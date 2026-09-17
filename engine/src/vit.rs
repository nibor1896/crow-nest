//! #VIT — the visual tower: the container's `"vit"` section (27 vision blocks +
//! patch embed + learned position table + merger, NVFP4 with bf16 keeps) run on
//! the existing f32/NVFP4 kernel machinery, plus the Crow image wire inputs
//! (OpenAI `image_url` data-URL blocks, base64, bytes exactly as on disk)
//! decoded and preprocessed to HF patch form.
//!
//! Reference (every formula here): transformers 5.16.1
//! `models/qwen4_exp/modeling_qwen4_exp.py` + `vision_utils.py` +
//! `models/Qwen3.8-Flash-Next-original/config.json` (vision_config) and
//! `preprocessor_config.json`. Config of record: hidden 1152, heads 16
//! (head_dim 72, rotary dim 36, theta 10000), depth 27, intermediate 4304,
//! gelu_pytorch_tanh in the block MLPs, exact-erf GELU in the merger
//! (`nn.GELU()`, default approximate='none'), patch 16, temporal patch 2,
//! spatial merge 2, num_position_embeddings 2304 (side 48),
//! out_hidden_size 2560 = the text hidden size, deepstack_visual_indexes [].
//!
//! Sequence order: patches are emitted in SPATIAL-MERGE-BLOCK order
//! (block_row, block_col, in_row, in_col), the order both
//! `get_vision_interpolation_indices_and_weights` (spatial_merge_size > 1) and
//! the merger's `view(-1, 4*hidden)` assume. With that order the merger is a
//! plain GEMV over contiguous rows and no permute kernel exists.
//!
//! Precision: the tower runs f32 (NVFP4 GEMV in the `gemv_fp4_b` op order via
//! `gemv_fp4_vit`, which also covers k_dim 4304 = 64*67 + 16 that the 64-wide
//! walk cannot). The oracle golden (f32 over the same container-dequantized
//! weights) differs by math precision only; the band is a measurement, not a
//! bit gate.
//!
//! Image size limits (preprocessor_config.json): smart_resize factor
//! patch*merge = 32, min_pixels 65536, max_pixels 16777216; an image may
//! produce at most `VIT_MAX_PATCHES` patches (4096 visual tokens — the
//! llama-server `--image-max-tokens` default), larger requests answer 400.

use crate::cuda::{self, CUdeviceptr as Dev};
use crate::weights::{dequant_fp4_dev, load_fp4, Fp4};
use crate::kernels::{launch_v, Kernels};
use crate::{cnq::Cnq, geo::{H, MIB}};

/// the `<|image_pad|>` token (tokenizer_config.json added_tokens_decoder)
pub const IMAGE_PAD: i64 = 248056;

pub const VIT_HIDDEN: usize = 1152;
pub const VIT_HEADS: usize = 16;
pub const VIT_HEAD_DIM: usize = VIT_HIDDEN / VIT_HEADS; // 72
pub const VIT_ROT: usize = VIT_HEAD_DIM / 2; // 36
pub const VIT_BLOCKS: usize = 27;
pub const VIT_INTER: usize = 4304;
pub const VIT_MERGED: usize = 4 * VIT_HIDDEN; // 4608 merger input row
pub const VIT_PATCH: usize = 16;
pub const VIT_TPATCH: usize = 2;
pub const VIT_MERGE: usize = 2;
pub const VIT_SIDE: usize = 48; // sqrt(num_position_embeddings)
pub const VIT_IN: usize = 3 * VIT_TPATCH * VIT_PATCH * VIT_PATCH; // 1536 conv row
pub const VIT_QKV: usize = 3 * VIT_HIDDEN; // 3456
/// hard patch cap per image (4096 patches = 1024 visual tokens; every
/// production VLM bounds image resolution - at 16384 the patch attention
/// alone costs hours, and no chat question needs more than this grid)
pub const VIT_MAX_PATCHES: usize = 4096;

/// min_pixels / max_pixels of the preprocessor config (size shortest/longest_edge)
const MIN_PIXELS: u64 = 65536;
const MAX_PIXELS: u64 = 16777216;

/// CROW_VIT: unset or any value but `0` = vision ON (the default after gates);
/// `0` = the text-only placeholder of record (no vit load, /props vision false)
pub fn vit_on() -> bool {
    std::env::var("CROW_VIT").as_deref() != Ok("0")
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

pub struct VitBlockW {
    pub n1w: Dev,
    pub n1b: Dev,
    pub n2w: Dev,
    pub n2b: Dev,
    pub qkv: Fp4,
    pub qkv_b: Dev,
    pub proj: Fp4,
    pub proj_b: Dev,
    pub fc1: Fp4,
    pub fc1_b: Dev,
    pub fc2: Fp4,
    pub fc2_b: Dev,
}

pub struct VitW {
    pub patch_proj: Fp4,
    pub patch_bias: Dev,
    /// f32 [2304][1152], dequantized once at load (the interp gathers read f32)
    pub pos_embed: Dev,
    pub blocks: Vec<VitBlockW>,
    pub m_norm_w: Dev,
    pub m_norm_b: Dev,
    pub m_fc1: Fp4,
    pub m_fc1_b: Dev,
    pub m_fc2: Fp4,
    pub m_fc2_b: Dev,
}

/// every vit keep is a bf16 tensor; read_f32 asserts that
unsafe fn load_f32_small(cnq: &mut Cnq, name: &str) -> Dev {
    cuda::to_f32_dev(&cnq.read_f32(name, "vit"))
}

/// fp4 with a loud assert — the sidecar of record has all 112 big vit tensors
/// NVFP4; a bf16 keep here would silently run the wrong GEMV
unsafe fn load_f4(cnq: &mut Cnq, name: &str) -> Fp4 {
    let t = cnq.find(name, "vit").clone();
    assert_eq!(t.dtype, "nvfp4", "{name}: the vit section of record is NVFP4, got {}", t.dtype);
    load_fp4(cnq, name, "vit")
}

impl VitW {
    pub unsafe fn load(cnq: &mut Cnq) -> VitW {
        let trace = std::env::var("CROW_VIT_TRACE").is_ok();
        macro_rules! say { ($m:expr) => { if trace { eprintln!("[vit-trace] {}", $m); } } }
        say!("patch_embed");
        let patch_proj = load_f4(cnq, "model.visual.patch_embed.proj.weight");
        let patch_bias = load_f32_small(cnq, "model.visual.patch_embed.proj.bias");
        say!("pos_embed");
        let pos_embed = dequant_fp4_dev(
            cnq, "model.visual.pos_embed.weight", "vit", VIT_SIDE * VIT_SIDE * VIT_HIDDEN);
        say!("pos_embed ok");
        let mut blocks = Vec::with_capacity(VIT_BLOCKS);
        for b in 0..VIT_BLOCKS {
            let p = |s: &str| format!("model.visual.blocks.{b}.{s}");
            blocks.push(VitBlockW {
                n1w: load_f32_small(cnq, &p("norm1.weight")),
                n1b: load_f32_small(cnq, &p("norm1.bias")),
                n2w: load_f32_small(cnq, &p("norm2.weight")),
                n2b: load_f32_small(cnq, &p("norm2.bias")),
                qkv: load_f4(cnq, &p("attn.qkv.weight")),
                qkv_b: load_f32_small(cnq, &p("attn.qkv.bias")),
                proj: load_f4(cnq, &p("attn.proj.weight")),
                proj_b: load_f32_small(cnq, &p("attn.proj.bias")),
                fc1: load_f4(cnq, &p("mlp.linear_fc1.weight")),
                fc1_b: load_f32_small(cnq, &p("mlp.linear_fc1.bias")),
                fc2: load_f4(cnq, &p("mlp.linear_fc2.weight")),
                fc2_b: load_f32_small(cnq, &p("mlp.linear_fc2.bias")),
            });
            if b % 8 == 7 { say!(format!("block {b}")); }
        }
        say!("merger");
        let w2 = VitW {
            patch_proj,
            patch_bias,
            pos_embed,
            blocks,
            m_norm_w: load_f32_small(cnq, "model.visual.merger.norm.weight"),
            m_norm_b: load_f32_small(cnq, "model.visual.merger.norm.bias"),
            m_fc1: load_f4(cnq, "model.visual.merger.linear_fc1.weight"),
            m_fc1_b: load_f32_small(cnq, "model.visual.merger.linear_fc1.bias"),
            m_fc2: load_f4(cnq, "model.visual.merger.linear_fc2.weight"),
            m_fc2_b: load_f32_small(cnq, "model.visual.merger.linear_fc2.bias"),
        };
        say!("weights ok");
        w2
    }
}


// ---------------------------------------------------------------------------
// scratch + the tower forward
// ---------------------------------------------------------------------------

pub struct Vit {
    pub w: VitW,
    pub cap: usize,
    /// #VIT cache: image-bytes hash -> (grid, n_visual, tower embeddings on
    /// the host). A conversation that re-sends its whole history (Crow does,
    /// base64 and all) pays for each picture ONCE per process, not per turn.
    image_cache: std::collections::HashMap<u64, ((usize, usize, usize), usize, Vec<f32>)>,
    /// cache keys, least recently used first: the cache is per PROCESS and a
    /// serve runs for days, so it is bounded by bytes (`CROW_VIT_CACHE_MB`)
    /// and evicted LRU. Unbounded it grew by up to 10 MiB per distinct image,
    /// forever (about 1 GiB per 100 screenshots).
    image_lru: Vec<u64>,
    image_cache_bytes: usize,
    x: Dev,        // [cap][1152] residual stream
    normed: Dev,   // [cap][1152] ln output / gemv scratch
    qkv: Dev,      // [cap][3456]
    attn: Dev,     // [cap][1152] attention output rows
    mlp: Dev,      // [cap][4304]
    m1: Dev,       // [cap/4][4608] merger fc1 output
    out: Dev,      // [cap/4][2560] the visual embeddings
    patches: Dev,  // [cap][1536] normalized patch input
    pe_idx: Dev,   // [cap*4] i32 pos-table taps
    pe_w: Dev,     // [cap*4] f32 tap weights
    cs: Dev,       // [cap][36] rope cos
    sn: Dev,       // [cap][36] rope sin
    /// device i32 scalars, indexed by the S_* consts
    s: Vec<Dev>,
    /// the scratch is allocated (lazily, on the first image request)
    scratch: bool,
}

const S_KD_IN: usize = 0;
const S_KD_HIDDEN: usize = 1;
const S_KD_INTER: usize = 2;
const S_KD_MERGED: usize = 3;
const S_COLS_HIDDEN: usize = 4;
const S_N: usize = 5;
const S_NV: usize = 6;
const S_STRIDE_HIDDEN: usize = 7;
const S_BIAS_QKV: usize = 8;  // stride 3456, cols 3456 (one pair packed [s,c])
const S_BIAS_INTER: usize = 9;
const S_BIAS_MERGED: usize = 10;
const S_BIAS_H: usize = 11;
const S_RESERVED_12: usize = 12; // reserved slot; every S_* index is a kernel argument position - never renumber
const _: () = assert!(S_RESERVED_12 + 1 == S_NP);
const S_NP: usize = 13;          // n * 1152, the add_flat residual count (refreshed per image)
const S_NMLP: usize = 14;        // n * 4304, the block MLP gelu count
const S_NMERGE: usize = 15;      // nv * 4608, the merger gelu count

impl Vit {
    /// weights + cap-sized scratch, all resident BEFORE the budget planner runs
    /// weights resident BEFORE the budget planner (they are part of the model);
    /// the cap-sized SCRATCH is allocated LAZILY on the first image request
    /// (ensure_scratch) so text-only boots keep the full planner budget — the
    /// F49 ten-task operating point refused with the scratch resident (found
    /// 2026-09-14, the TEN phase in `decode_out/srv-vit.log`). One-time,
    /// before the request's text prefill, then resident for the process life.
    pub unsafe fn new(cnq: &mut Cnq) -> Vit {
        let w = VitW::load(cnq);
        Vit {
            w,
            cap: VIT_MAX_PATCHES,
            x: 0,
            normed: 0,
            qkv: 0,
            attn: 0,
            mlp: 0,
            m1: 0,
            out: 0,
            patches: 0,
            pe_idx: 0,
            pe_w: 0,
            cs: 0,
            sn: 0,
            s: Vec::new(),
            scratch: false,
            image_cache: std::collections::HashMap::new(),
            image_lru: Vec::new(),
            image_cache_bytes: 0,
        }
    }

    /// one-time scratch allocation, called on the first image request; the
    /// scalar slots are sized here and refreshed per image by `run`
    unsafe fn ensure_scratch(&mut self) {
        if self.scratch {
            return;
        }
        let cap = self.cap;
        // every scratch buffer is n four-byte elements: the allocator counts
        // what it actually handed out and the boot line prints THAT. It used
        // to print `cap * 41_024`, a hand-summed bytes-per-patch literal that
        // no longer matched the twelve allocations below - and could not have
        // been noticed, because nothing derives from it.
        let mut bytes = 0usize;
        let mut alloc4 = |n: usize| { bytes += n * 4; cuda::alloc_zeroed(n * 4) };
        let scalars: Vec<i32> = vec![
            VIT_IN as i32, VIT_HIDDEN as i32, VIT_INTER as i32, VIT_MERGED as i32, VIT_HIDDEN as i32,
            0, 0,                          // S_N, S_NV (refreshed per image)
            0, 0, 0, 0, 0,                 // S_STRIDE_HIDDEN + S_BIAS*
            0,                             // S_RESERVED_12
            0, 0, 0,                       // S_NP, S_NMLP, S_NMERGE
        ];
        self.s = scalars.iter().map(|v| cuda::to_i32_dev(&[*v])).collect();
        self.x = alloc4(cap * VIT_HIDDEN);
        self.normed = alloc4(cap * VIT_HIDDEN);
        self.qkv = alloc4(cap * VIT_QKV);
        self.attn = alloc4(cap * VIT_HIDDEN);
        self.mlp = alloc4(cap * VIT_INTER);
        self.m1 = alloc4(cap / 4 * VIT_MERGED);
        self.out = alloc4(cap / 4 * H);
        self.patches = alloc4(cap * VIT_IN);
        self.pe_idx = alloc4(cap * 4);
        self.pe_w = alloc4(cap * 4);
        self.cs = alloc4(cap * VIT_ROT);
        self.sn = alloc4(cap * VIT_ROT);
        cuda::to_i32_into(self.s[S_STRIDE_HIDDEN], &[VIT_HIDDEN as i32]);
        cuda::to_i32_into(self.s[S_BIAS_QKV], &[VIT_QKV as i32]);
        cuda::to_i32_into(self.s[S_BIAS_INTER], &[VIT_INTER as i32]);
        cuda::to_i32_into(self.s[S_BIAS_MERGED], &[VIT_MERGED as i32]);
        cuda::to_i32_into(self.s[S_BIAS_H], &[H as i32]);
        cuda::sync();
        self.scratch = true;
        eprintln!("[vit] scratch allocated on the first image request ({:.0} MiB at cap {} patches)",
            bytes as f64 / MIB, cap);
    }

    /// one NVFP4 linear: y[t][rows] = w x^T with RAW f32 activations, via the
    /// tiled `gemm_fp4_f32x` (bit-identical product tree to gemv_fp4_vit,
    /// reads amortized over 32 tokens x 64 rows). rows_p/kd name SCALAR slots
    /// holding the row and k_dim counts (values reuse the k_dim/bias slots).
    unsafe fn gemv(&self, k: &Kernels, f: &Fp4, rows: usize, kd: usize, rows_p: usize, t: usize, x: Dev, y: Dev) {
        launch_v(k.f("gemm_fp4_f32x"), ((rows + 63) / 64) as u32, ((t + 31) / 32) as u32, 1, 256, &[
            f.w, x, f.gs, y, self.s[kd], self.s[rows_p], self.s[S_N]]);
    }

    /// out[t][stride] rows get +bias[0..cols]; stride == cols for every vision
    /// linear, so one scalar serves both kernel params
    unsafe fn bias(&self, k: &Kernels, bias: Dev, sp: usize, t: usize, y: Dev) {
        launch_v(k.f("vit_add_bias"), t as u32, 1, 1, 256, &[
            y, bias, self.s[sp], self.s[sp]]);
    }

    /// CROW_VIT_TRACE stage dump: sync, one D2H, one `stage-<tag>.f32` in
    /// CROW_VIT_DUMP. The seven trace points in `run` all had this body inline.
    unsafe fn trace_dump(&self, on: bool, tag: &str, buf: Dev, n: usize) {
        if !on { return; }
        cuda::sync();
        let dir = std::env::var("CROW_VIT_DUMP").unwrap_or_default();
        f32_file(&format!("{dir}/stage-{tag}.f32"), &cuda::dtoh(buf, n));
    }

    unsafe fn ln(&self, k: &Kernels, w: Dev, b: Dev, t: usize, x: Dev, out: Dev) {
        launch_v(k.f("vit_ln"), t as u32, 1, 1, 256, &[x, w, b, out, self.s[S_COLS_HIDDEN]]);
    }

    /// the full tower. `patches_host` [n][1536] normalized patch rows in
    /// merge-block order; `idx_host`/`w_host` the per-token 4-tap pos-table
    /// gathers; `cs_host`/`sn_host` the per-token [36] rotary tables — all
    /// four built by `vision_tables`. Returns the device visual embeddings
    /// [n/4][2560] f32 (the scratch `out`, valid until the next image).
    pub unsafe fn run(
        &mut self,
        k: &Kernels,
        patches_host: &[f32],
        idx_host: &[i32],
        w_host: &[f32],
        cs_host: &[f32],
        sn_host: &[f32],
        n: usize,
    ) -> Dev {
        self.ensure_scratch();
        let nv = n / (VIT_MERGE * VIT_MERGE);
        cuda::to_f32_into(self.patches, patches_host);
        cuda::to_i32_into(self.pe_idx, idx_host);
        cuda::to_f32_into(self.pe_w, w_host);
        cuda::to_f32_into(self.cs, cs_host);
        cuda::to_f32_into(self.sn, sn_host);
        cuda::to_i32_into(self.s[S_N], &[n as i32]);
        cuda::to_i32_into(self.s[S_NV], &[nv as i32]);
        cuda::to_i32_into(self.s[S_NP], &[((n * VIT_HIDDEN) as i32)]);
        cuda::to_i32_into(self.s[S_NMLP], &[((n * VIT_INTER) as i32)]);
        cuda::to_i32_into(self.s[S_NMERGE], &[((nv * VIT_MERGED) as i32)]);

        // patch embed: the Conv3d as GEMV 1536 → 1152, then bias, then the
        // bilinear-interpolated learned position embed
        self.gemv(k, &self.w.patch_proj, VIT_HIDDEN, S_KD_IN, S_KD_HIDDEN, n, self.patches, self.x);
        self.bias(k, self.w.patch_bias, S_STRIDE_HIDDEN, n, self.x);
        launch_v(k.f("vit_pe_add"), n as u32, 1, 1, 256, &[
            self.x, self.w.pos_embed, self.pe_idx, self.pe_w, self.s[S_N]]);
        // stage dumps for the oracle bisect (CROW_VIT_TRACE=1)
        let trace = std::env::var("CROW_VIT_TRACE").is_ok();
        if trace {
            self.trace_dump(true, "pe", self.x, n * VIT_HIDDEN);
            let dumpdir = std::env::var("CROW_VIT_DUMP").unwrap_or_default();
            f32_file(&format!("{dumpdir}/stage-cs.f32"), cs_host);
            f32_file(&format!("{dumpdir}/stage-sn.f32"), sn_host);
            eprintln!("[vit-trace] dumped stage-pe + cs/sn");
        }

        for (bi, b) in self.w.blocks.iter().enumerate() {
            // norm1 → fused qkv (+ bias), rotary in place on the q and k thirds
            self.ln(k, b.n1w, b.n1b, n, self.x, self.normed);
            let b0 = trace && bi == 0;
            self.trace_dump(b0, "b0-n1", self.normed, n * VIT_HIDDEN);
            self.gemv(k, &b.qkv, VIT_QKV, S_KD_HIDDEN, S_BIAS_QKV, n, self.normed, self.qkv);
            self.bias(k, b.qkv_b, S_BIAS_QKV, n, self.qkv);
            self.trace_dump(b0, "b0-qkv", self.qkv, n * VIT_QKV);
            launch_v(k.f("vit_rope"), VIT_HEADS as u32, n as u32, 1, 64, &[
                self.qkv, self.cs, self.sn]);
            self.trace_dump(b0, "b0-qkv-rope", self.qkv, n * VIT_QKV);
            // non-causal attention over the whole image, one query row per block
            launch_v(k.f("vit_attn"), n as u32, VIT_HEADS as u32, 1, 256, &[
                self.qkv, self.attn, self.s[S_N]]);
            self.trace_dump(b0, "b0-attn", self.attn, n * VIT_HIDDEN);
            // proj + residual
            self.gemv(k, &b.proj, VIT_HIDDEN, S_KD_HIDDEN, S_STRIDE_HIDDEN, n, self.attn, self.normed);
            self.bias(k, b.proj_b, S_STRIDE_HIDDEN, n, self.normed);
            launch_v(k.f("add_flat"), ((n * VIT_HIDDEN + 255) / 256) as u32, 1, 1, 256, &[
                self.normed, self.x, self.s[S_NP]]);
            self.trace_dump(b0, "b0-res", self.x, n * VIT_HIDDEN);
            // MLP: fc1 → gelu tanh → fc2 → residual
            self.ln(k, b.n2w, b.n2b, n, self.x, self.normed);
            self.gemv(k, &b.fc1, VIT_INTER, S_KD_HIDDEN, S_BIAS_INTER, n, self.normed, self.mlp);
            self.bias(k, b.fc1_b, S_BIAS_INTER, n, self.mlp);
            launch_v(k.f("gelu_tanh"), ((n * VIT_INTER + 255) / 256) as u32, 1, 1, 256, &[
                self.mlp, self.s[S_NMLP]]);
            self.trace_dump(b0, "b0-gelu", self.mlp, n * VIT_INTER);
            self.gemv(k, &b.fc2, VIT_HIDDEN, S_KD_INTER, S_STRIDE_HIDDEN, n, self.mlp, self.normed);
            self.bias(k, b.fc2_b, S_STRIDE_HIDDEN, n, self.normed);
            launch_v(k.f("add_flat"), ((n * VIT_HIDDEN + 255) / 256) as u32, 1, 1, 256, &[
                self.normed, self.x, self.s[S_NP]]);
            self.trace_dump(b0, "block0", self.x, n * VIT_HIDDEN);
            if b0 { eprintln!("[vit-trace] dumped stage-block0"); }
        }

        // merger: LN(1152) over patches → the [nv][4608] view is contiguous →
        // fc1 → exact-erf GELU → fc2 → the [nv][2560] visual embeddings
        self.ln(k, self.w.m_norm_w, self.w.m_norm_b, n, self.x, self.normed);
        self.gemv(k, &self.w.m_fc1, VIT_MERGED, S_KD_MERGED, S_BIAS_MERGED, nv, self.normed, self.m1);
        self.bias(k, self.w.m_fc1_b, S_BIAS_MERGED, nv, self.m1);
        launch_v(k.f("gelu_erf"), ((nv * VIT_MERGED + 255) / 256) as u32, 1, 1, 256, &[
            self.m1, self.s[S_NMERGE]]);
        self.gemv(k, &self.w.m_fc2, H, S_KD_MERGED, S_BIAS_H, nv, self.m1, self.out);
        self.bias(k, self.w.m_fc2_b, S_BIAS_H, nv, self.out);
        self.out
    }

    pub unsafe fn free(&mut self) {
        if !self.scratch {
            return;
        }
        cuda::free_dev(&mut self.x);
        cuda::free_dev(&mut self.normed);
        cuda::free_dev(&mut self.qkv);
        cuda::free_dev(&mut self.attn);
        cuda::free_dev(&mut self.mlp);
        cuda::free_dev(&mut self.m1);
        cuda::free_dev(&mut self.out);
        cuda::free_dev(&mut self.patches);
        cuda::free_dev(&mut self.pe_idx);
        cuda::free_dev(&mut self.pe_w);
        cuda::free_dev(&mut self.cs);
        cuda::free_dev(&mut self.sn);
        for d in self.s.iter_mut() {
            cuda::free_dev(d);
        }
        self.s.clear();
    }
}

// ---------------------------------------------------------------------------
// preprocessing: decode → smart_resize → bicubic → normalize → patchify
// ---------------------------------------------------------------------------

pub struct ImagePrep {
    /// [n_patches][1536] normalized, merge-block order
    pub patches: Vec<f32>,
    /// i32 [n_patches*4] pos-table taps (h*48 + w), tap order (h0w0, h0w1, h1w0, h1w1)
    pub pe_idx: Vec<i32>,
    /// f32 [n_patches*4] tap weights
    pub pe_w: Vec<f32>,
    /// f32 [n_patches][36] rotary cos
    pub cs: Vec<f32>,
    /// f32 [n_patches][36] rotary sin
    pub sn: Vec<f32>,
    pub n_patches: usize,
    /// patch grid (1, hp, wp) — the `image_grid_thw` row
    pub grid: (usize, usize, usize),
    /// resized source pixels (h, w), both multiples of 32
    pub resized: (usize, usize),
    pub n_visual: usize,
}

/// unit-test accessor for the private `smart_resize`
pub fn smart_resize_for_test(h: u64, w: u64) -> Result<(u64, u64), String> {
    smart_resize(h, w)
}

/// HF `smart_resize` (qwen2_vl image processing), factor 32. Python's
/// `round` is banker's rounding — round-half-EVEN — replicated here exactly.
fn smart_resize(height: u64, width: u64) -> Result<(u64, u64), String> {
    const FACTOR: u64 = (VIT_PATCH * VIT_MERGE) as u64;
    if height < FACTOR || width < FACTOR {
        return Err(format!(
            "image {height}x{width} smaller than the {FACTOR}px patch factor"));
    }
    if height.max(width) / height.min(width) > 200 {
        return Err(format!("image aspect ratio over 200 ({height}x{width})"));
    }
    let pyround = |x: f64| -> f64 {
        let f = x.floor();
        let d = x - f;
        if d > 0.5 { f + 1.0 } else if d < 0.5 { f } else if (f as i64) % 2 == 0 { f } else { f + 1.0 }
    };
    let h_bar = (pyround(height as f64 / FACTOR as f64) as u64) * FACTOR;
    let w_bar = (pyround(width as f64 / FACTOR as f64) as u64) * FACTOR;
    if h_bar * w_bar > MAX_PIXELS {
        let beta = ((height * width) as f64 / MAX_PIXELS as f64).sqrt();
        let h2 = ((height as f64 / beta / FACTOR as f64).floor() as u64) * FACTOR;
        let w2 = ((width as f64 / beta / FACTOR as f64).floor() as u64) * FACTOR;
        Ok((h2.max(FACTOR), w2.max(FACTOR)))
    } else if h_bar * w_bar < MIN_PIXELS {
        let beta = (MIN_PIXELS as f64 / ((height * width) as f64)).sqrt();
        let h2 = ((height as f64 * beta / FACTOR as f64).ceil() as u64) * FACTOR;
        let w2 = ((width as f64 * beta / FACTOR as f64).ceil() as u64) * FACTOR;
        Ok((h2, w2))
    } else {
        Ok((h_bar, w_bar))
    }
}

/// torchvision bicubic (Keys a = -0.75), antialias=True: separable, taps
/// windowed by `support = 2 * in/out` on downscale, weights normalized.
/// `src`/`dst` are row-major [rows][cols_in/cols_out] f32.
fn resize_axis(src: &[f32], rows: usize, cols_in: usize, cols_out: usize) -> Vec<f32> {
    let scale = cols_in as f64 / cols_out as f64; // > 1 = downscale
    let support = if scale > 1.0 { 2.0 * scale } else { 2.0 };
    let kernel = |x: f64| -> f64 {
        const A: f64 = -0.75;
        let x = x.abs();
        if x <= 1.0 {
            (A + 2.0) * x * x * x - (A + 3.0) * x * x + 1.0
        } else if x < 2.0 {
            A * x * x * x - 5.0 * A * x * x + 8.0 * A * x - 4.0 * A
        } else {
            0.0
        }
    };
    let mut out = vec![0f32; rows * cols_out];
    let mut wbuf = vec![0f64; 2 * cols_in.max(1)];
    for r in 0..rows {
        for i in 0..cols_out {
            let center = (i as f64 + 0.5) * scale - 0.5;
            let lo = ((center - support).ceil() as i64).max(0) as usize;
            let hi = ((center + support).floor() as i64).min(cols_in as i64 - 1) as usize;
            let mut sum = 0f64;
            let mut wsum = 0f64;
            let mut nw = 0usize;
            for j in lo..=hi {
                let t = (j as f64 - center) / scale;
                let w = kernel(t);
                if w == 0.0 {
                    continue;
                }
                wbuf[nw] = w;
                sum += w * src[r * cols_in + j] as f64;
                wsum += w;
                nw += 1;
            }
            out[r * cols_out + i] = if wsum != 0.0 { (sum / wsum) as f32 } else { 0.0 };
        }
    }
    out
}

/// transpose [rows][cols] → [cols][rows]
fn transpose(v: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; v.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = v[r * cols + c];
        }
    }
    out
}

/// decode raw image bytes (any of Crow's five formats) → RGB8
pub fn decode_rgb(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32), String> {
    let img = image::load_from_memory(bytes).map_err(|e| format!("image decode failed: {e}"))?;
    let (w, h) = (img.width(), img.height());
    Ok((img.to_rgb8().into_raw(), w, h))
}

/// the full preprocessing: HF Qwen2-VL fast-processor order — decode, resize
/// (bicubic antialias on the 0..255 values, rounded back to the u8 grid),
/// rescale 1/255, normalize mean/std 0.5, patchify in merge-block order with
/// the temporal frame duplicated — plus the host tables the tower launches
/// need (pos-table taps, rotary cos/sin).
pub fn prep_image(bytes: &[u8]) -> Result<ImagePrep, String> {
    let (rgb, w0, h0) = decode_rgb(bytes)?;
    let (rh, rw) = smart_resize(h0 as u64, w0 as u64)?;
    let (mut rh, mut rw) = (rh as usize, rw as usize);
    let (w0, h0) = (w0 as usize, h0 as usize);

    // #VIT cap: an image over the patch budget is DOWNSCALED further, never
    // refused - a screenshot must reach the model at the resolution a chat
    // needs, the same clamp every production VLM applies. Dims stay
    // FACTOR-multiples so the patch grid keeps its merge alignment.
    const FACTOR: usize = VIT_PATCH * VIT_MERGE;
    while (rh / VIT_PATCH) * (rw / VIT_PATCH) > VIT_MAX_PATCHES {
        let beta = ((rh * rw) as f64 / (VIT_MAX_PATCHES * VIT_PATCH * VIT_PATCH) as f64).sqrt() * 1.06;
        let nh = (((rh as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        let nw = (((rw as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        if nh >= rh && nw >= rw {
            break; // degenerate: cannot shrink further, accept as is
        }
        rh = nh;
        rw = nw;
    }
    let hp = rh / VIT_PATCH;
    let wp = rw / VIT_PATCH;
    let n_patches = hp * wp;
    let n_visual = hp * wp / (VIT_MERGE * VIT_MERGE);

    // resize: horizontal pass then vertical pass per channel. ONE channel plane
    // at a time: the f32 copy of the decoded raster is 4x its bytes, and holding
    // all three reached about 2 GiB of host RAM from a single 16 MiB PNG (the
    // decoder's own cap is 512 MiB). `plane` carries exactly the bytes the
    // [3][h0][w0] slice carried, so the tower sees the same input.
    let mut resized = Vec::with_capacity(3 * rh * rw);
    let mut plane = vec![0f32; h0 * w0];
    for c in 0..3 {
        for (i, px) in rgb.chunks_exact(3).enumerate() {
            plane[i] = px[c] as f32;                    // to f32 0..255
        }
        let hor = resize_axis(&plane, h0, w0, rw);      // [h0][rw]
        let ver_src = transpose(&hor, h0, rw);          // [rw][h0]
        let ver = resize_axis(&ver_src, rw, h0, rh);    // [rw][rh]
        let out = transpose(&ver, rw, rh);              // [rh][rw]
        resized.extend_from_slice(&out);
    }
    drop(plane);
    drop(rgb);
    // torchvision rounds the interpolated values back to the u8 grid before
    // rescale/normalize (the fast processor resizes the uint8 tensor)
    for v in resized.iter_mut() {
        *v = v.round().clamp(0.0, 255.0);
    }

    // normalize ((x/255) - 0.5) / 0.5 = x/127.5 - 1
    for v in resized.iter_mut() {
        *v = *v / 127.5 - 1.0;
    }

    // patchify, merge-block order; row = (c, t, y, x), t duplicated
    let at = |y: usize, x: usize, c: usize| resized[c * rh * rw + y * rw + x];
    let mut patches = vec![0f32; n_patches * VIT_IN];
    let mut pe_idx = vec![0i32; n_patches * 4];
    let mut pe_w = vec![0f32; n_patches * 4];
    let mut cs = vec![0f32; n_patches * VIT_ROT];
    let mut sn = vec![0f32; n_patches * VIT_ROT];
    let bw = wp / VIT_MERGE;
    // 18 frequencies (arange(0, 36, 2) / 36), REUSED for the w axis — the
    // oracle's rot vector is [h*inv_0..h*inv_17, w*inv_0..w*inv_17]
    let inv_freq: Vec<f32> = (0..VIT_ROT / 2)
        .map(|j| 10000f32.powf(-(2.0 * j as f32) / VIT_ROT as f32))
        .collect();
    for bi in 0..hp / VIT_MERGE {
        for bj in 0..bw {
            for iy in 0..VIT_MERGE {
                for ix in 0..VIT_MERGE {
                    let py = bi * VIT_MERGE + iy;
                    let px = bj * VIT_MERGE + ix;
                    let seq = (bi * bw + bj) * (VIT_MERGE * VIT_MERGE) + iy * VIT_MERGE + ix;
                    let row = &mut patches[seq * VIT_IN..(seq + 1) * VIT_IN];
                    for c in 0..3 {
                        for ty in 0..VIT_PATCH {
                            for tx in 0..VIT_PATCH {
                                let v = at(py * VIT_PATCH + ty, px * VIT_PATCH + tx, c);
                                for t in 0..VIT_TPATCH {
                                    row[((c * VIT_TPATCH + t) * VIT_PATCH + ty) * VIT_PATCH + tx] = v;
                                }
                            }
                        }
                    }
                    // position of this patch in the (row, col) patch grid —
                    // the same pair feeds the pos-table interp AND the rotary
                    let (pr, pc) = (py, px);
                    // bilinear taps, align_corners=True, resampling the 48x48
                    // learned table to this image's patch grid (vision_utils
                    // _interpolation_axis_taps_weights): src = p * (side-1) /
                    // (grid-1), taps floor/floor+1 clamped, weights (1-d, d)
                    let taps = |p: usize, grid: usize, side: usize| -> (usize, usize, f32, f32) {
                        if grid <= 1 {
                            return (0, 0, 1.0, 0.0);
                        }
                        let src = p as f64 * (side as f64 - 1.0) / (grid as f64 - 1.0);
                        let f = (src.floor() as usize).min(side - 1);
                        let d = (src - f as f64) as f32;
                        (f, (f + 1).min(side - 1), 1.0 - d, d)
                    };
                    let (r0, r1, wr0, wr1) = taps(pr, hp, VIT_SIDE);
                    let (c0, c1, wc0, wc1) = taps(pc, wp, VIT_SIDE);
                    for (k, (ri, ci, wt)) in [(r0, c0, wr0 * wc0), (r0, c1, wr0 * wc1),
                                              (r1, c0, wr1 * wc0), (r1, c1, wr1 * wc1)]
                        .into_iter()
                        .enumerate()
                    {
                        pe_idx[seq * 4 + k] = (ri * VIT_SIDE + ci) as i32;
                        pe_w[seq * 4 + k] = wt;
                    }
                    // rotary: freqs [h*inv_0..h*inv_17, w*inv_0..w*inv_17]
                    for j in 0..VIT_ROT {
                        let p = if j < VIT_ROT / 2 { pr } else { pc };
                        let f = p as f32 * inv_freq[j % (VIT_ROT / 2)];
                        cs[seq * VIT_ROT + j] = f.cos();
                        sn[seq * VIT_ROT + j] = f.sin();
                    }
                }
            }
        }
    }
    Ok(ImagePrep {
        patches,
        pe_idx,
        pe_w,
        cs,
        sn,
        n_patches,
        grid: (1, hp, wp),
        resized: (rh, rw),
        n_visual,
    })
}

// ---------------------------------------------------------------------------
// the text-side splice: id expansion, the mrope tables, the plan
// ---------------------------------------------------------------------------

/// one image's grid (t, hp, wp) — `image_grid_thw` row
pub type Grid = (usize, usize, usize);

/// the per-request vision plan: where the visual embeddings land in the
/// (already expanded) prompt id list, the embeddings themselves, and the
/// mrope delta the decode path needs
pub struct VisionPlan {
    /// the expanded prompt ids (every IMAGE_PAD replaced by its visual tokens)
    pub ids: Vec<u32>,
    /// device [n_visual][2560] f32 — the merged embeddings of ALL images,
    /// concatenated in message order
    pub embeds: Dev,
    /// the same rows on the host (dumps)
    pub embeds_host: Vec<f32>,
    /// per final-sequence row: flat visual embedding index, or -1 for text
    pub map: Vec<i32>,
    /// mm_token_type per row (0 text, 1 image) — the get_rope_index input
    pub types: Vec<u8>,
    pub grids: Vec<Grid>,
    pub n_visual: usize,
    /// mrope_position_deltas of the oracle (max_pos + 1 - seq_len)
    pub delta: i64,
}

/// expand every IMAGE_PAD token (one per image, template order) into
/// `counts[i]` copies
pub fn expand_ids(ids: &[u32], counts: &[usize]) -> Result<Vec<u32>, String> {
    let mut out = Vec::with_capacity(ids.len() + counts.iter().sum::<usize>());
    let mut it = counts.iter();
    for &id in ids {
        if id as i64 == IMAGE_PAD {
            let c = it.next().ok_or("more image_pad tokens than images")?;
            out.extend(std::iter::repeat(id).take(*c));
        } else {
            out.push(id);
        }
    }
    if it.next().is_some() {
        return Err("more images than image_pad tokens".into());
    }
    Ok(out)
}

/// `get_rope_index` (Qwen4ExpModel) for one sequence: text groups run
/// arange + current_pos on all three axes; each image group gets
/// T = current_pos, H = current_pos + hpos, W = current_pos + wpos over its
/// merged tokens (block-major order), then current_pos += max(hp, wp) / 2.
/// Returns per-row (T, H, W) and the mrope delta (max + 1 - seq_len).
pub fn mrope_positions(types: &[u8], grids: &[Grid]) -> (Vec<[i64; 3]>, i64) {
    let mut it = grids.iter();
    let mut out = Vec::with_capacity(types.len());
    let mut cur: i64 = 0;
    let mut i = 0usize;
    while i < types.len() {
        let ty = types[i];
        let start = i;
        while i < types.len() && types[i] == ty {
            i += 1;
        }
        let len = i - start;
        if ty == 0 {
            for k in 0..len {
                out.push([cur + k as i64; 3]);
            }
            cur += len as i64;
        } else {
            let g = it.next().expect("image type row without a grid");
            let (_t, hp, wp) = *g;
            let gw = wp / VIT_MERGE;
            for k in 0..len {
                // block-major token order: bh, bw, iy, ix
                let bh = k / (gw * VIT_MERGE * VIT_MERGE);
                let rem = k % (gw * VIT_MERGE * VIT_MERGE);
                let bw = rem / (VIT_MERGE * VIT_MERGE);
                let r2 = rem % (VIT_MERGE * VIT_MERGE);
                let iy = r2 / VIT_MERGE;
                let ix = r2 % VIT_MERGE;
                let t_ax = cur; // arange(t) = 0, + start_position after
                let h_ax = cur + (bh * VIT_MERGE + iy) as i64;
                let w_ax = cur + (bw * VIT_MERGE + ix) as i64;
                out.push([t_ax, h_ax, w_ax]);
            }
            cur += (hp.max(wp) / VIT_MERGE) as i64;
        }
    }
    let max_pos = out.iter().flat_map(|r| r.iter()).copied().max().unwrap_or(0);
    let delta = max_pos + 1 - types.len() as i64;
    (out, delta)
}

/// the interleaved-mrope cos/sin span tables (Qwen4ExpTextRotaryEmbedding,
/// `apply_interleaved_mrope`, section [11, 11, 10], partial rotary 0.25):
/// slot j carries T unless it is an H slot (j % 3 == 1, j <= 31) or a W slot
/// (j % 3 == 2, j <= 29). Rows `seq..span` are the decode rows: all three
/// axes at `row + delta`. The f32 expressions are byte-for-byte the ones in
/// `manager.rs` so a text-only span reproduces the load-time table bit for bit.
pub fn mrope_tables(pos: &[[i64; 3]], seq: usize, span: usize, delta: i64) -> (Vec<f32>, Vec<f32>) {
    // mrope section sizes [T, H, W] = [11, 11, 10]: T keeps every slot not taken by H / W
    let mut cos = vec![0f32; span * 32];
    let mut sin = vec![0f32; span * 32];
    for row in 0..span {
        let [pt, ph, pw] = if row < seq {
            pos[row]
        } else {
            let p = row as i64 + delta;
            [p, p, p]
        };
        for j in 0..32usize {
            let axis_pos = if j % 3 == 1 && j <= 31 {
                ph
            } else if j % 3 == 2 && j <= 29 {
                pw
            } else {
                pt
            };
            let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
            let f = axis_pos as f32 * inv;
            cos[row * 32 + j] = f.cos();
            sin[row * 32 + j] = f.sin();
        }
    }
    (cos, sin)
}

/// byte ceiling of the per-process image-embedding cache (`CROW_VIT_CACHE_MB`,
/// default 256 MiB = about 25 full-size images at 10 MiB each)
fn vit_cache_bytes() -> usize {
    static MB: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *MB.get_or_init(|| {
        crate::geo::env_parse::<usize>("CROW_VIT_CACHE_MB").unwrap_or(256)
    }) << 20
}

impl Vit {
    /// insert one tower output, then evict least-recently-used entries until the
    /// cache is back under `vit_cache_bytes()`. An entry larger than the whole
    /// ceiling is not cached at all (it would evict everything and then itself).
    fn cache_insert(&mut self, key: u64, grid: (usize, usize, usize), n_visual: usize, rows: Vec<f32>) {
        let bytes = rows.len() * 4;
        let ceiling = vit_cache_bytes();
        if bytes > ceiling {
            return;
        }
        if let Some((_, _, old)) = self.image_cache.insert(key, (grid, n_visual, rows)) {
            self.image_cache_bytes -= old.len() * 4;
            self.image_lru.retain(|&k| k != key);
        }
        self.image_cache_bytes += bytes;
        self.image_lru.push(key);
        while self.image_cache_bytes > ceiling && !self.image_lru.is_empty() {
            let oldest = self.image_lru.remove(0);
            if let Some((_, _, old)) = self.image_cache.remove(&oldest) {
                self.image_cache_bytes -= old.len() * 4;
            }
        }
    }

    /// decode + preprocess + run every image, expand the ids, and assemble the
    /// plan the prefill splice and the decode mrope need. `ids` are the
    /// RENDERED prompt ids (one IMAGE_PAD per image, message order).
    pub unsafe fn build_plan(
        &mut self,
        k: &Kernels,
        ids: &[u32],
        images: &[Vec<u8>],
    ) -> Result<VisionPlan, String> {
        // #VIT cache: the tower output depends on the image BYTES only, so
        // every image runs decode + preprocess + tower at most once per
        // process; re-sent history images are a hash lookup and a clone.
        let mut infos: Vec<((usize, usize, usize), usize)> = Vec::with_capacity(images.len());
        let mut all_rows: Vec<Vec<f32>> = Vec::with_capacity(images.len());
        let dump = std::env::var("CROW_VIT_DUMP").ok();
        if let Some(dir) = &dump {
            let _ = std::fs::create_dir_all(dir);
        }
        let mut misses = 0usize;
        for (i, bytes) in images.iter().enumerate() {
            use std::collections::hash_map::DefaultHasher;
            use std::hash::{Hash, Hasher};
            let mut hasher = DefaultHasher::new();
            bytes.hash(&mut hasher);
            let key = hasher.finish();
            if let Some((grid, n_visual, rows)) = self.image_cache.get(&key) {
                eprintln!("[vit-cache] image {i}: HIT grid {grid:?}, {n_visual} visual tokens");
                infos.push((*grid, *n_visual));
                all_rows.push(rows.clone());
                self.image_lru.retain(|&k| k != key);
                self.image_lru.push(key);
                continue;
            }
            let p = prep_image(bytes)
                .map_err(|e| format!("image {i}: {e}"))?;
            let out = self.run(k, &p.patches, &p.pe_idx, &p.pe_w, &p.cs, &p.sn, p.n_patches);
            let rows = cuda::dtoh(out, p.n_visual * H);
            if let Some(dir) = &dump {
                f32_file(&format!("{dir}/img{i}.patches.f32"), &p.patches);
                f32_file(&format!("{dir}/img{i}.pe-w.f32"), &p.pe_w);
                let meta = serde_json::json!({
                    "grid": p.grid, "n_patches": p.n_patches,
                    "resized": p.resized, "n_visual": p.n_visual,
                });
                let _ = std::fs::write(format!("{dir}/img{i}.meta.json"), meta.to_string());
            }
            infos.push((p.grid, p.n_visual));
            all_rows.push(rows.clone());
            self.cache_insert(key, p.grid, p.n_visual, rows);
            misses += 1;
        }
        eprintln!(
            "[vit-cache] {} image(s): {} through the tower, {} cached; cache {} entries, {:.1} MiB of {} MiB",
            images.len(), misses, images.len() - misses,
            self.image_cache.len(), self.image_cache_bytes as f64 / MIB,
            vit_cache_bytes() >> 20
        );
        let counts: Vec<usize> = infos.iter().map(|(_, n)| *n).collect();
        let expanded = expand_ids(ids, &counts)?;
        // walk the expanded ids, mark the image rows
        let mut types = vec![0u8; expanded.len()];
        let mut map = vec![-1i32; expanded.len()];
        let mut flat: i32 = 0;
        for (r, &id) in expanded.iter().enumerate() {
            if id as i64 == IMAGE_PAD {
                types[r] = 1;
                map[r] = flat;
                flat += 1;
            }
        }
        let (_pos, delta) = mrope_positions(&types, &infos.iter().map(|(g, _)| *g).collect::<Vec<_>>());
        // concatenate the per-image tower outputs (image order preserved)
        let mut embeds_host = Vec::new();
        for r in &all_rows {
            embeds_host.extend_from_slice(r);
        }
        cuda::sync();
        if let Some(dir) = &dump {
            f32_file(&format!("{dir}/vit-embeds.f32"), &embeds_host);
        }
        let embeds = cuda::to_f32_dev(&embeds_host);
        Ok(VisionPlan {
            ids: expanded,
            embeds,
            embeds_host,
            map,
            types,
            grids: infos.iter().map(|(g, _)| *g).collect(),
            n_visual: flat as usize,
            delta,
        })
    }
}

/// raw f32 little-endian file write, failures logged not raised.
/// `pub` because serve's `/v1` logits dump writes the same file the same way.
pub fn f32_file(path: &str, v: &[f32]) {
    if let Err(e) = cuda::write_le(path, v) {
        eprintln!("[vit-dump] write {path} failed: {e}");
    }
}

impl Drop for VisionPlan {
    fn drop(&mut self) {
        if self.embeds != 0 {
            unsafe { cuda::free_dev(&mut self.embeds) };
        }
    }
}
