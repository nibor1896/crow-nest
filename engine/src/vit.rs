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
//! Image size limits: smart_resize factor patch*merge = 32, so one visual token
//! covers 32x32 source pixels. The preprocessor_config.json window (min_pixels
//! 65536 = 64 tokens, max_pixels 16777216 = 16384 tokens) is NOT the serving
//! window: `VitBudget` (#107) resizes every image into
//! [`CROW_VIT_MIN_TOKENS`, `CROW_VIT_MAX_TOKENS`] visual tokens, default
//! [1024, 1280] - the floor is the llama.cpp operating point's
//! `--image-min-tokens 1024` (clip.cpp warns that Qwen-VL needs 1024), the cap
//! the upper end of the Qwen3-VL README's 256-1280 recommendation. An image
//! over the cap is DOWNSCALED (never refused); the tower scratch and the
//! planner reserve are sized from the cap (`scratch_bytes`).

use crate::cuda::{self, CUdeviceptr as Dev};
use crate::weights::{dequant_fp4_dev, load_fp4, Fp4};
use crate::kernels::{launch_v, Kernels};
use crate::{cnq::Cnq, geo::{H, MIB}};

/// the `<|image_pad|>` token (tokenizer_config.json added_tokens_decoder)
pub const IMAGE_PAD: i64 = 248056;

pub const VIT_HIDDEN: usize = 1152;
pub const VIT_HEADS: usize = 16;
pub const VIT_HEAD_DIM: usize = VIT_HIDDEN / VIT_HEADS; // 72
/// #98 step 2: query rows per `vit_attn` block (the `BR` of the kernel);
/// the launch grid is (ceil(n / VIT_ATTN_ROWS), VIT_HEADS)
pub const VIT_ATTN_ROWS: usize = 16;
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
/// min_pixels / max_pixels of the preprocessor config (size shortest/longest_edge):
/// the HF reference window `smart_resize_for_test` checks the port against. The
/// SERVING window is `VitBudget`.
const HF_MIN_PIXELS: u64 = 65536;
const HF_MAX_PIXELS: u64 = 16777216;

/// source pixels per visual token: (patch * merge)^2 = 32 * 32
pub const VIT_PIXELS_PER_TOKEN: u64 = ((VIT_PATCH * VIT_MERGE) * (VIT_PATCH * VIT_MERGE)) as u64;

/// #107: the default visual-token FLOOR per image. Before this the
/// floor was the preprocessor's 64 tokens, and Crow's renders reached the model
/// at 527-880 tokens where llama.cpp's operating point gives the same PNGs 1032
/// (`--image-min-tokens 1024`; at ~350 tokens the model said no image arrived,
/// Crow docs/operating-points.md). clip.cpp's load_hparams warns: "Qwen-VL models
/// require at minimum 1024 image tokens to function correctly on grounding tasks".
pub const VIT_MIN_TOKENS_DEFAULT: usize = 1024;
/// the default visual-token CAP per image (was 1024 = 4096 patches): the upper
/// end of the Qwen3-VL README's "256-1280" token budget. It sits above the floor
/// so the floor's ceil rounding (1000x560 -> 1032) is never cut by the cap.
pub const VIT_MAX_TOKENS_DEFAULT: usize = 1280;
/// the bounds either env var may take: clip.cpp `set_limit_image_tokens(8, 4096)`
/// for the Qwen-VL projectors (4096 tokens = 16384 patches, 914 MiB of scratch)
pub const VIT_TOKENS_LOWEST: usize = 8;
pub const VIT_TOKENS_HIGHEST: usize = 4096;

/// The per-image visual-token window the server resizes into, read once per
/// process from `CROW_VIT_MIN_TOKENS` / `CROW_VIT_MAX_TOKENS`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct VitBudget {
    pub min_tokens: usize,
    pub max_tokens: usize,
}

impl VitBudget {
    pub const DEFAULT: VitBudget =
        VitBudget { min_tokens: VIT_MIN_TOKENS_DEFAULT, max_tokens: VIT_MAX_TOKENS_DEFAULT };

    /// the env values as strings (None = unset); a malformed or out-of-range
    /// value is an ERROR, not a silent default - a typo must not change what
    /// the model sees without a word
    pub fn parse(min: Option<&str>, max: Option<&str>) -> Result<VitBudget, String> {
        let one = |key: &str, v: Option<&str>, dflt: usize| -> Result<usize, String> {
            let Some(v) = v else { return Ok(dflt) };
            let n: usize = v
                .trim()
                .parse()
                .map_err(|_| format!("{key}={v:?} is not a whole number of visual tokens"))?;
            if !(VIT_TOKENS_LOWEST..=VIT_TOKENS_HIGHEST).contains(&n) {
                return Err(format!(
                    "{key}={n} is outside {VIT_TOKENS_LOWEST}..={VIT_TOKENS_HIGHEST} visual tokens (clip.cpp's Qwen-VL limits)"
                ));
            }
            Ok(n)
        };
        let min_tokens = one("CROW_VIT_MIN_TOKENS", min, VIT_MIN_TOKENS_DEFAULT)?;
        // an explicit floor above the default cap lifts the default cap with it
        let max_default = VIT_MAX_TOKENS_DEFAULT.max(min_tokens);
        let max_tokens = one("CROW_VIT_MAX_TOKENS", max, max_default)?;
        if min_tokens > max_tokens {
            return Err(format!(
                "CROW_VIT_MIN_TOKENS={min_tokens} is above CROW_VIT_MAX_TOKENS={max_tokens}"
            ));
        }
        Ok(VitBudget { min_tokens, max_tokens })
    }

    pub fn from_env() -> Result<VitBudget, String> {
        let min = std::env::var("CROW_VIT_MIN_TOKENS").ok();
        let max = std::env::var("CROW_VIT_MAX_TOKENS").ok();
        VitBudget::parse(min.as_deref(), max.as_deref())
    }

    /// the patch cap the tower scratch is sized for (4 patches per visual token)
    pub const fn max_patches(&self) -> usize {
        self.max_tokens * VIT_MERGE * VIT_MERGE
    }
    pub const fn min_pixels(&self) -> u64 {
        self.min_tokens as u64 * VIT_PIXELS_PER_TOKEN
    }
    pub const fn max_pixels(&self) -> u64 {
        self.max_tokens as u64 * VIT_PIXELS_PER_TOKEN
    }
}

/// the process's budget: `CROW_VIT_MIN_TOKENS` / `CROW_VIT_MAX_TOKENS`, read
/// once. A bad value stops the boot with the reason (`Engine::load` calls this
/// before the tower loads, so it never fails inside a request).
pub fn budget() -> VitBudget {
    static B: std::sync::OnceLock<VitBudget> = std::sync::OnceLock::new();
    *B.get_or_init(|| VitBudget::from_env().unwrap_or_else(|e| panic!("[vit] {e}")))
}

/// the `[vit]` boot line's words for the budget, naming where each bound came from
pub fn budget_words(b: &VitBudget) -> String {
    let src = |key: &str| {
        if std::env::var(key).is_ok() { key.to_string() } else { "default".to_string() }
    };
    format!(
        "{}..{} visual tokens per image (floor {}, cap {}; {} patches at the cap)",
        b.min_tokens,
        b.max_tokens,
        src("CROW_VIT_MIN_TOKENS"),
        src("CROW_VIT_MAX_TOKENS"),
        b.max_patches()
    )
}

/// CROW_VIT: unset or any value but `0` = vision ON (the default after gates);
/// `0` = the text-only placeholder of record (no vit load, /props vision false)
pub fn vit_on() -> bool {
    std::env::var("CROW_VIT").as_deref() != Ok("0")
}

/// device buffers `ensure_scratch` takes (the 16 i32 scalar slots are counted
/// apart: 64 B in total, inside the same group unwind)
pub const VIT_SCRATCH_BUFFERS: usize = 12;

/// bytes of the tower scratch at a patch cap — the twelve buffers
/// `ensure_scratch` takes, counted from the same geometry the allocator uses
/// (58,496 B per patch: 4096 patches = 228.5 MiB, 5120 = 285.6 MiB)
pub const fn scratch_bytes_for(cap: usize) -> usize {
    let elems = cap * VIT_HIDDEN * 3   // x, normed, attn
        + cap * VIT_QKV
        + cap * VIT_INTER
        + cap / 4 * VIT_MERGED
        + cap / 4 * H
        + cap * VIT_IN
        + cap * 8                      // pe_idx + pe_w
        + cap * VIT_ROT * 2;           // cs + sn
    elems * 4
}

/// the scratch at this process's cap (`budget().max_patches()`)
pub fn scratch_bytes() -> usize {
    scratch_bytes_for(budget().max_patches())
}

/// bytes of the interleaved-mrope span tables at their widest: the whole
/// context, cos and sin, exactly the shape `Engine::begin_vision` allocates
pub const fn mrope_bytes(context: usize) -> usize {
    context * crate::geo::ROPE_PAIRS * 2 * 4
}

/// The VRAM the PLANNER sets aside for the image path when the tower is loaded
/// (TASK K, 2026-09-17).
///
/// Before this the vision path allocated everything lazily "on the first image
/// request so text-only boots keep the full planner budget" (#VIT): the planner
/// maximised the hot set against the free VRAM and an image request then had to
/// find its scratch and its mrope tables in whatever the
/// 512 MiB `manager::SAFETY` slack had left. On robin's 2026-09-17 serve session
/// it did not, and `cuda::ck` turned the refusal into a process panic that took
/// the rest of the session with it. The three numbers are named here, added to
/// the planner's `pending` bytes, and printed on the `[budget]` boot line.
///
/// `CROW_VIT_RESERVE_MB` replaces the derived value (0 = no reserve, the
/// pre-TASK-K behaviour, for a measurement that wants the old N back).
/// `CROW_VIT=0` never calls this at all.
pub fn reserve_bytes(context: usize) -> u64 {
    if let Some(mb) = crate::geo::env_parse::<u64>("CROW_VIT_RESERVE_MB") {
        return mb << 20;
    }
    (scratch_bytes() + mrope_bytes(context)) as u64
}

/// the `[budget]` line's own words for what `reserve_bytes` covers; with
/// `CROW_VIT_RESERVE_MB` set it names the override AND what the derived value
/// would have been, so a log that shows a bigger N still says what paid for it
pub fn reserve_line(context: usize, held: u64) -> String {
    let mib = |b: usize| b as f64 / MIB;
    let derived = (scratch_bytes() + mrope_bytes(context)) as f64 / MIB;
    let basis = match crate::geo::env_parse::<u64>("CROW_VIT_RESERVE_MB") {
        Some(_) => format!("CROW_VIT_RESERVE_MB, derived would be {derived:.1} MB"),
        None => format!(
            "tower scratch {:.1} + mrope span {:.1}",
            mib(scratch_bytes()),
            mib(mrope_bytes(context))
        ),
    };
    // #72: what the planner still has to SET ASIDE, now that the tower scratch and
    // the mrope span are taken at boot. With the derived reserve the two are the
    // same number and nothing is pending; a bigger CROW_VIT_RESERVE_MB pads the
    // plan by the difference, a smaller one is already covered by what is held.
    let pending = reserve_bytes(context).saturating_sub(held);
    let state = if held == 0 {
        "lazy, allocated on the first image request".to_string()
    } else if pending == 0 {
        format!("{:.1} MB HELD at boot, nothing left pending", held as f64 / MIB)
    } else {
        format!("{:.1} MB HELD at boot, {:.1} MB pending", held as f64 / MIB, pending as f64 / MIB)
    };
    format!(
        "vit reserve {:9.1} MB  ({basis}, CROW_VIT on) — {state}",
        reserve_bytes(context) as f64 / MIB
    )
}

// ---------------------------------------------------------------------------
// weights
// ---------------------------------------------------------------------------

/// one vision linear [rows][k] as the tower's GEMM reads it: the container's
/// NVFP4 pair (`gemm_fp4_f32x`) or #108's exact F16 copy from llama.cpp's
/// projector file (`gemm_f16_f32x`, row-major u16 [rows][k])
#[derive(Clone, Copy)]
pub enum Lin {
    Fp4(Fp4),
    F16(Dev),
}

pub struct VitBlockW {
    pub n1w: Dev,
    pub n1b: Dev,
    pub n2w: Dev,
    pub n2b: Dev,
    pub qkv: Lin,
    pub qkv_b: Dev,
    pub proj: Lin,
    pub proj_b: Dev,
    pub fc1: Lin,
    pub fc1_b: Dev,
    pub fc2: Lin,
    pub fc2_b: Dev,
}

pub struct VitW {
    pub patch_proj: Lin,
    pub patch_bias: Dev,
    /// f32 [2304][1152], dequantized once at load (the interp gathers read f32)
    pub pos_embed: Dev,
    pub blocks: Vec<VitBlockW>,
    pub m_norm_w: Dev,
    pub m_norm_b: Dev,
    pub m_fc1: Lin,
    pub m_fc1_b: Dev,
    pub m_fc2: Lin,
    pub m_fc2_b: Dev,
    /// the `[vit]` boot line's "mode": where the weights came from
    pub mode: String,
}

// ---------------------------------------------------------------------------
// #108: the F16 projector file as the tower's weight source
// ---------------------------------------------------------------------------

/// the file name Crow's llama.cpp operating point loads with `--mmproj`
pub const MMPROJ_FILE: &str = "mmproj-F16.gguf";

/// Where the tower's weights come from, decided once at load.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum VitSource {
    /// llama.cpp's F16 projector (the default whenever the file is found)
    Mmproj(String),
    /// the container's NVFP4 `vit` section; the string says why (the fallback)
    Container(String),
}

/// `CROW_VIT_MMPROJ` wins: `0` = the container's NVFP4 section, any other
/// value = that file (missing -> the container, with the reason). Unset: the
/// first `mmproj-F16.gguf` found in `models/` beside the container (the #94
/// checkpoint convention), `$CROW_MODELS` (Crow's models root) and
/// `~/.local/share/crow/models` (Crow's Linux install link). `exists` is the
/// file test, injected so the unit test runs without the files.
pub fn resolve_mmproj(
    env: Option<&str>,
    cnq_path: &str,
    crow_models: Option<&str>,
    home: Option<&str>,
    exists: &dyn Fn(&str) -> bool,
) -> VitSource {
    match env {
        Some("0") => return VitSource::Container("CROW_VIT_MMPROJ=0".to_string()),
        Some(p) if !p.is_empty() => {
            return if exists(p) {
                VitSource::Mmproj(p.to_string())
            } else {
                VitSource::Container(format!("CROW_VIT_MMPROJ={p} does not exist"))
            };
        }
        _ => {}
    }
    let mut tried = Vec::new();
    let cnq = std::path::Path::new(cnq_path);
    if let Some(root) = cnq.parent().and_then(|d| d.parent()) {
        tried.push(root.join("models").join(MMPROJ_FILE));
    }
    if let Some(m) = crow_models.filter(|m| !m.is_empty()) {
        tried.push(std::path::Path::new(m).join(MMPROJ_FILE));
    }
    if let Some(h) = home.filter(|h| !h.is_empty()) {
        tried.push(std::path::Path::new(h).join(".local/share/crow/models").join(MMPROJ_FILE));
    }
    for p in &tried {
        let s = p.to_string_lossy();
        if exists(&s) {
            return VitSource::Mmproj(s.into_owned());
        }
    }
    let list = tried.iter().map(|p| p.to_string_lossy().into_owned()).collect::<Vec<_>>().join(", ");
    VitSource::Container(format!("no {MMPROJ_FILE} in {list}"))
}

/// the process's choice, from the real env and filesystem
pub fn mmproj_source(cnq_path: &str) -> VitSource {
    let env = std::env::var("CROW_VIT_MMPROJ").ok();
    let models = std::env::var("CROW_MODELS").ok();
    let home = std::env::var("HOME").ok();
    resolve_mmproj(env.as_deref(), cnq_path, models.as_deref(), home.as_deref(), &|p| {
        std::path::Path::new(p).is_file()
    })
}

/// the kind of one projector tensor the tower reads
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MmKind {
    /// an F16 linear, ggml dims (k, rows)
    Linear,
    /// an F32 vector or table read as f32
    F32,
}

/// Every projector tensor the tower reads: (gguf name, ggml dims, kind, the
/// container tensor it replaces). llama.cpp's names (convert_hf_to_gguf.py,
/// Qwen3-VL): `v.blk.N.{ln1,ln2,attn_qkv,attn_out,ffn_up,ffn_down}`,
/// `v.post_ln` = merger.norm, `mm.0`/`mm.2` = merger.linear_fc1/fc2, the Conv3d
/// patch kernel split along its temporal axis into `v.patch_embd.weight`
/// (t = 0) and `v.patch_embd.weight.1` (t = 1), `v.position_embd.weight` F32.
pub fn mmproj_plan() -> Vec<(String, Vec<u64>, MmKind, String)> {
    let (h, i, q, m, o) = (VIT_HIDDEN as u64, VIT_INTER as u64, VIT_QKV as u64, VIT_MERGED as u64, H as u64);
    let p = VIT_PATCH as u64;
    let mut v: Vec<(String, Vec<u64>, MmKind, String)> = vec![
        ("v.patch_embd.weight".into(), vec![p, p, 3, h], MmKind::Linear, "model.visual.patch_embed.proj.weight".into()),
        ("v.patch_embd.weight.1".into(), vec![p, p, 3, h], MmKind::Linear, "model.visual.patch_embed.proj.weight".into()),
        ("v.patch_embd.bias".into(), vec![h], MmKind::F32, "model.visual.patch_embed.proj.bias".into()),
        ("v.position_embd.weight".into(), vec![h, (VIT_SIDE * VIT_SIDE) as u64], MmKind::F32, "model.visual.pos_embed.weight".into()),
        ("v.post_ln.weight".into(), vec![m / 4], MmKind::F32, "model.visual.merger.norm.weight".into()),
        ("v.post_ln.bias".into(), vec![m / 4], MmKind::F32, "model.visual.merger.norm.bias".into()),
        ("mm.0.weight".into(), vec![m, m], MmKind::Linear, "model.visual.merger.linear_fc1.weight".into()),
        ("mm.0.bias".into(), vec![m], MmKind::F32, "model.visual.merger.linear_fc1.bias".into()),
        ("mm.2.weight".into(), vec![m, o], MmKind::Linear, "model.visual.merger.linear_fc2.weight".into()),
        ("mm.2.bias".into(), vec![o], MmKind::F32, "model.visual.merger.linear_fc2.bias".into()),
    ];
    for b in 0..VIT_BLOCKS {
        let g = |s: &str| format!("v.blk.{b}.{s}");
        let c = |s: &str| format!("model.visual.blocks.{b}.{s}");
        v.extend([
            (g("ln1.weight"), vec![h], MmKind::F32, c("norm1.weight")),
            (g("ln1.bias"), vec![h], MmKind::F32, c("norm1.bias")),
            (g("ln2.weight"), vec![h], MmKind::F32, c("norm2.weight")),
            (g("ln2.bias"), vec![h], MmKind::F32, c("norm2.bias")),
            (g("attn_qkv.weight"), vec![h, q], MmKind::Linear, c("attn.qkv.weight")),
            (g("attn_qkv.bias"), vec![q], MmKind::F32, c("attn.qkv.bias")),
            (g("attn_out.weight"), vec![h, h], MmKind::Linear, c("attn.proj.weight")),
            (g("attn_out.bias"), vec![h], MmKind::F32, c("attn.proj.bias")),
            (g("ffn_up.weight"), vec![h, i], MmKind::Linear, c("mlp.linear_fc1.weight")),
            (g("ffn_up.bias"), vec![i], MmKind::F32, c("mlp.linear_fc1.bias")),
            (g("ffn_down.weight"), vec![i, h], MmKind::Linear, c("mlp.linear_fc2.weight")),
            (g("ffn_down.bias"), vec![h], MmKind::F32, c("mlp.linear_fc2.bias")),
        ]);
    }
    v
}

/// the header check before a byte is uploaded: the projector type and the
/// geometry keys this tower is built for, then every tensor of `mmproj_plan`
/// present with its exact dims and type (Linear = F16, F32 = F32). A file that
/// fails any of it is NOT loaded - the boot falls back to the container, loudly.
pub fn validate_mmproj(g: &crate::gguf::Gguf) -> Result<(), String> {
    use crate::gguf::{GGML_TYPE_F16, GGML_TYPE_F32};
    let want_str = [("clip.projector_type", "qwen3vl_merger")];
    for (k, v) in want_str {
        match g.kv_str(k) {
            Some(s) if s == v => {}
            other => return Err(format!("{}: {k} is {other:?}, the tower needs {v:?}", g.path)),
        }
    }
    let want_u = [
        ("clip.vision.block_count", VIT_BLOCKS as u64),
        ("clip.vision.embedding_length", VIT_HIDDEN as u64),
        ("clip.vision.feed_forward_length", VIT_INTER as u64),
        ("clip.vision.attention.head_count", VIT_HEADS as u64),
        ("clip.vision.patch_size", VIT_PATCH as u64),
        ("clip.vision.spatial_merge_size", VIT_MERGE as u64),
        ("clip.vision.projection_dim", H as u64),
    ];
    for (k, v) in want_u {
        if g.kv_u64(k) != Some(v) {
            return Err(format!("{}: {k} is {:?}, the tower needs {v}", g.path, g.kv_u64(k)));
        }
    }
    for (name, dims, kind, _) in mmproj_plan() {
        let t = g.find(&name).ok_or_else(|| format!("{}: tensor {name} missing", g.path))?;
        if t.dims != dims {
            return Err(format!("{}: tensor {name} dims {:?}, expected {dims:?}", g.path, t.dims));
        }
        let want = if kind == MmKind::Linear { GGML_TYPE_F16 } else { GGML_TYPE_F32 };
        if t.ggml_type != want {
            return Err(format!("{}: tensor {name} ggml type {}, expected {want}", g.path, t.ggml_type));
        }
    }
    Ok(())
}

/// the Conv3d patch kernel [1152][3][2][16][16] (our patch-row order
/// ((c*2 + t)*16 + ty)*16 + tx) from the two temporal halves llama.cpp stores,
/// each [1152][3][16][16] (ggml dims (16, 16, 3, 1152)); raw f16 bits
pub fn interleave_patch_kernel(t0: &[u16], t1: &[u16]) -> Vec<u16> {
    let pp = VIT_PATCH * VIT_PATCH;
    assert_eq!(t0.len(), VIT_HIDDEN * 3 * pp);
    assert_eq!(t1.len(), t0.len());
    let mut out = vec![0u16; VIT_HIDDEN * VIT_IN];
    for o in 0..VIT_HIDDEN {
        for c in 0..3 {
            let src = (o * 3 + c) * pp;
            for (t, half) in [t0, t1].iter().enumerate() {
                let dst = o * VIT_IN + (c * VIT_TPATCH + t) * pp;
                out[dst..dst + pp].copy_from_slice(&half[src..src + pp]);
            }
        }
    }
    out
}

fn u16s(raw: &[u8]) -> Vec<u16> {
    raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
}

fn f32s(raw: &[u8]) -> Vec<f32> {
    raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
}

/// every vit keep is a bf16 tensor; read_f32 asserts that
unsafe fn load_f32_small(cnq: &mut Cnq, name: &str) -> Dev {
    cuda::to_f32_dev(&cnq.read_f32(name, "vit"))
}

/// fp4 with a loud assert — the sidecar of record has all 112 big vit tensors
/// NVFP4; a bf16 keep here would silently run the wrong GEMV
unsafe fn load_f4(cnq: &mut Cnq, name: &str) -> Lin {
    let t = cnq.find(name, "vit").clone();
    assert_eq!(t.dtype, "nvfp4", "{name}: the vit section of record is NVFP4, got {}", t.dtype);
    Lin::Fp4(load_fp4(cnq, name, "vit"))
}

impl VitW {
    /// #108: the tower's weights from the source `mmproj_source` picks -
    /// llama.cpp's F16 projector when the file is there and passes
    /// `validate_mmproj`, else the container's NVFP4 section (the fallback,
    /// named on the `[vit]` boot line with its reason).
    pub unsafe fn load(cnq: &mut Cnq) -> VitW {
        match mmproj_source(&cnq.path) {
            VitSource::Mmproj(path) => match crate::gguf::Gguf::open(&path).and_then(|g| validate_mmproj(&g).map(|_| g)) {
                Ok(g) => VitW::load_mmproj(&g),
                Err(why) => {
                    tracing::error!(target: "vit", "[vit] the F16 projector is NOT used: {why} - falling back to the container's NVFP4 vit section");
                    VitW::load_container(cnq, &format!("fallback, {path} refused: {why}"))
                }
            },
            VitSource::Container(why) => VitW::load_container(cnq, &why),
        }
    }

    /// the projector file's tensors, F16 linears kept F16 on the device, the
    /// F32 vectors and the position table as f32. `g` passed `validate_mmproj`.
    unsafe fn load_mmproj(g: &crate::gguf::Gguf) -> VitW {
        let raw = |name: &str| -> Vec<u8> {
            let t = g.find(name).unwrap_or_else(|| panic!("{}: {name} vanished after validation", g.path));
            g.read_raw(t).unwrap_or_else(|e| panic!("[vit] {e}"))
        };
        let lin = |name: &str| -> Lin { Lin::F16(cuda::upload_dev_named("a vit F16 linear", &raw(name))) };
        let vec32 = |name: &str| -> Dev { cuda::to_f32_dev(&f32s(&raw(name))) };
        let t0 = u16s(&raw("v.patch_embd.weight"));
        let t1 = u16s(&raw("v.patch_embd.weight.1"));
        let pk = interleave_patch_kernel(&t0, &t1);
        let patch_proj = Lin::F16(cuda::upload_dev_named(
            "the vit F16 patch kernel",
            std::slice::from_raw_parts(pk.as_ptr() as *const u8, pk.len() * 2),
        ));
        let blocks = (0..VIT_BLOCKS)
            .map(|b| {
                let p = |s: &str| format!("v.blk.{b}.{s}");
                VitBlockW {
                    n1w: vec32(&p("ln1.weight")),
                    n1b: vec32(&p("ln1.bias")),
                    n2w: vec32(&p("ln2.weight")),
                    n2b: vec32(&p("ln2.bias")),
                    qkv: lin(&p("attn_qkv.weight")),
                    qkv_b: vec32(&p("attn_qkv.bias")),
                    proj: lin(&p("attn_out.weight")),
                    proj_b: vec32(&p("attn_out.bias")),
                    fc1: lin(&p("ffn_up.weight")),
                    fc1_b: vec32(&p("ffn_up.bias")),
                    fc2: lin(&p("ffn_down.weight")),
                    fc2_b: vec32(&p("ffn_down.bias")),
                }
            })
            .collect();
        VitW {
            patch_proj,
            patch_bias: vec32("v.patch_embd.bias"),
            pos_embed: vec32("v.position_embd.weight"),
            blocks,
            m_norm_w: vec32("v.post_ln.weight"),
            m_norm_b: vec32("v.post_ln.bias"),
            m_fc1: lin("mm.0.weight"),
            m_fc1_b: vec32("mm.0.bias"),
            m_fc2: lin("mm.2.weight"),
            m_fc2_b: vec32("mm.2.bias"),
            mode: format!("f16 (llama.cpp projector {}, 112 F16 linears)", g.path),
        }
    }

    unsafe fn load_container(cnq: &mut Cnq, why: &str) -> VitW {
        let trace = std::env::var("CROW_VIT_TRACE").is_ok();
        macro_rules! say { ($m:expr) => { if trace { tracing::info!(target: "vit", "[vit-trace] {}", $m); } } }
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
            mode: format!("nvfp4 (the container vit section; {why})"),
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
    /// the scratch is allocated (#72: at boot when the reserve is held, else
    /// lazily on the first image request)
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
    /// weights resident BEFORE the budget planner (they are part of the model).
    ///
    /// The cap-sized SCRATCH was allocated LAZILY on the first image request
    /// until #72 (2026-09-18): text-only boots kept the full planner budget,
    /// but the planner's `reserve_bytes` was then only a note - the 512 MiB
    /// `manager::SAFETY` slack it lived in was spent by the driver's own
    /// post-plan allocations, and robin's first image request found 35.7 MiB
    /// free. `Engine::load` now calls `arm_scratch` right after this, so the
    /// reserve is held; the lazy path stays as the fallback (`CROW_VIT_RESERVE_MB=0`).
    pub unsafe fn new(cnq: &mut Cnq) -> Vit {
        let w = VitW::load(cnq);
        Vit {
            w,
            cap: budget().max_patches(),
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
    /// scalar slots are sized here and refreshed per image by `run`.
    ///
    /// TASK K: every allocation here is FALLIBLE and named. The twelve buffers
    /// plus sixteen scalars are taken as a group, and a refusal in the middle
    /// frees the ones already taken before it raises - `scratch` stays false, no
    /// pointer is left dangling in `self`, and the engine is exactly as it was.
    /// Inside a request scope the raise is an `AllocFailed` payload, so `serve`
    /// answers 503 instead of the process dying (the planner's `vit::reserve_bytes`
    /// is what makes the refusal not happen in the first place).
    unsafe fn ensure_scratch(&mut self) {
        self.ensure_scratch_named("allocated on the first image request (the boot hold is off)");
    }

    /// #72: the same allocation, taken AT BOOT while the card is empty, so the
    /// planner's `reserve_bytes` is a HELD allocation and not a note. Called
    /// from `Engine::load` right after the tower weights, before the budget
    /// verify - `free0` then already excludes it, exactly like the dense
    /// weights, and nothing allocated after the plan (the decode graph, the
    /// driver's launch pools) can take the image path's bytes any more.
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other tower call
    pub unsafe fn arm_scratch(&mut self) {
        self.ensure_scratch_named("held at boot (#72: the vit reserve is a real allocation)");
    }

    /// `how` is what the one log line says about WHEN this happened
    unsafe fn ensure_scratch_named(&mut self, how: &str) {
        if self.scratch {
            return;
        }
        let cap = self.cap;
        // taken in order, freed in reverse if the card refuses one of them
        let mut taken: Vec<Dev> = Vec::with_capacity(28);
        let alloc_or_unwind = |what: &str, bytes: usize, taken: &mut Vec<Dev>| -> Dev {
            match cuda::try_alloc_zeroed(what, bytes) {
                Ok(d) => {
                    taken.push(d);
                    d
                }
                Err(e) => {
                    for d in taken.iter_mut() {
                        cuda::free_dev(d);
                    }
                    tracing::info!(target: "vit",
                        "[vit] scratch allocation refused after {} buffer(s) - they were freed, the tower stays unarmed",
                        taken.len()
                    );
                    e.raise()
                }
            }
        };
        // every scratch buffer is n four-byte elements: the allocator counts
        // what it actually handed out and the boot line prints THAT. It used
        // to print `cap * 41_024`, a hand-summed bytes-per-patch literal that
        // no longer matched the twelve allocations below - and could not have
        // been noticed, because nothing derives from it.
        let mut bytes = 0usize;
        let scalars: Vec<i32> = vec![
            VIT_IN as i32, VIT_HIDDEN as i32, VIT_INTER as i32, VIT_MERGED as i32, VIT_HIDDEN as i32,
            0, 0,                          // S_N, S_NV (refreshed per image)
            0, 0, 0, 0, 0,                 // S_STRIDE_HIDDEN + S_BIAS*
            0,                             // S_RESERVED_12
            0, 0, 0,                       // S_NP, S_NMLP, S_NMERGE
        ];
        let mut s = Vec::with_capacity(scalars.len());
        for v in &scalars {
            let d = alloc_or_unwind("the vit scalar slots", 4, &mut taken);
            // `into_dev_legacy`, not `to_i32_into`: this is the copy `to_i32_dev` made
            // before TASK K named the allocation - legacy stream plus its own sync, so
            // the slot is written the same way whatever stream the last request left active
            cuda::into_dev_legacy(d, &[*v]);
            s.push(d);
        }
        self.s = s;
        let mut alloc4 = |what: &str, n: usize, taken: &mut Vec<Dev>| {
            bytes += n * 4;
            alloc_or_unwind(what, n * 4, taken)
        };
        self.x = alloc4("the vit residual stream", cap * VIT_HIDDEN, &mut taken);
        self.normed = alloc4("the vit norm scratch", cap * VIT_HIDDEN, &mut taken);
        self.qkv = alloc4("the vit qkv scratch", cap * VIT_QKV, &mut taken);
        self.attn = alloc4("the vit attention output", cap * VIT_HIDDEN, &mut taken);
        self.mlp = alloc4("the vit block MLP scratch", cap * VIT_INTER, &mut taken);
        self.m1 = alloc4("the vit merger fc1 output", cap / 4 * VIT_MERGED, &mut taken);
        self.out = alloc4("the vit visual embeddings", cap / 4 * H, &mut taken);
        self.patches = alloc4("the vit patch input", cap * VIT_IN, &mut taken);
        self.pe_idx = alloc4("the vit position taps", cap * 4, &mut taken);
        self.pe_w = alloc4("the vit position tap weights", cap * 4, &mut taken);
        self.cs = alloc4("the vit rope cos", cap * VIT_ROT, &mut taken);
        self.sn = alloc4("the vit rope sin", cap * VIT_ROT, &mut taken);
        cuda::to_i32_into(self.s[S_STRIDE_HIDDEN], &[VIT_HIDDEN as i32]);
        cuda::to_i32_into(self.s[S_BIAS_QKV], &[VIT_QKV as i32]);
        cuda::to_i32_into(self.s[S_BIAS_INTER], &[VIT_INTER as i32]);
        cuda::to_i32_into(self.s[S_BIAS_MERGED], &[VIT_MERGED as i32]);
        cuda::to_i32_into(self.s[S_BIAS_H], &[H as i32]);
        cuda::sync();
        self.scratch = true;
        debug_assert_eq!(bytes, scratch_bytes(), "scratch_bytes() and ensure_scratch disagree");
        tracing::info!(target: "vit", "[vit] tower scratch {how}: {:.1} MiB at cap {} patches ({} buffers + {} scalars)",
            bytes as f64 / MIB, cap, VIT_SCRATCH_BUFFERS, scalars.len());
    }

    /// one NVFP4 linear: y[t][rows] = w x^T with RAW f32 activations, via the
    /// tiled `gemm_fp4_f32x` (bit-identical product tree to gemv_fp4_vit,
    /// reads amortized over 32 tokens x 64 rows). rows_p/kd name SCALAR slots
    /// holding the row and k_dim counts (values reuse the k_dim/bias slots).
    /// #108: an F16 linear runs `gemm_f16_f32x`, the same tile geometry
    /// with the exact f16 -> f32 widening in place of the e2m1 decode and scale.
    unsafe fn gemv(&self, k: &Kernels, f: &Lin, rows: usize, kd: usize, rows_p: usize, t: usize, x: Dev, y: Dev) {
        let grid = (rows.div_ceil(64) as u32, t.div_ceil(32) as u32);
        match f {
            Lin::Fp4(f) => launch_v(k.f("gemm_fp4_f32x"), grid.0, grid.1, 1, 256, &[
                f.w, x, f.gs, y, self.s[kd], self.s[rows_p], self.s[S_N]]),
            Lin::F16(w) => launch_v(k.f("gemm_f16_f32x"), grid.0, grid.1, 1, 256, &[
                *w, x, y, self.s[kd], self.s[rows_p], self.s[S_N]]),
        }
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
        // the scratch holds `cap` patch rows; prep_image sizes to the same budget
        assert!(n <= self.cap, "vit: {n} patches over the scratch cap {}", self.cap);
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
            tracing::info!(target: "vit", "[vit-trace] dumped stage-pe + cs/sn");
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
            // non-causal attention over the whole image, 16 query rows per block
            launch_v(k.f("vit_attn"), n.div_ceil(VIT_ATTN_ROWS) as u32, VIT_HEADS as u32, 1, 256, &[
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
            if b0 { tracing::info!(target: "vit", "[vit-trace] dumped stage-block0"); }
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
        // #73: the tower is ASYNC. Every launch above went to `cuda::cur_stream()`,
        // and the only reader of `self.out` is the blocking `cuMemcpyDtoH_v2` in
        // `build_plan`, which runs on the LEGACY null stream. A non-blocking
        // stream does not order against the null stream, so that D2H copied
        // whatever `self.out` still held - the PREVIOUS image's embeddings,
        // complete and plausible, one request stale. The whole visual path was
        // therefore lag-by-one: the FIRST image of a process read correctly (no
        // decode had run, so the active stream was still the legacy one), and
        // every image after it answered for its predecessor. That is what made
        // the #VIT smoke parity green - it only ever ran one image per process.
        // `sync()` synchronizes the stream the launches above actually used, so
        // it is correct whichever stream is active; it is the one point where
        // the tower has to meet the host, and it costs one image's latency.
        cuda::sync();
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

/// unit-test accessor for the private `smart_resize` at the preprocessor
/// config's own window (the HF reference cases)
pub fn smart_resize_for_test(h: u64, w: u64) -> Result<(u64, u64), String> {
    smart_resize(h, w, HF_MIN_PIXELS, HF_MAX_PIXELS)
}

/// HF `smart_resize` (qwen2_vl image processing), factor 32. Python's
/// `round` is banker's rounding — round-half-EVEN — replicated here exactly.
fn smart_resize(height: u64, width: u64, min_pixels: u64, max_pixels: u64) -> Result<(u64, u64), String> {
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
    if h_bar * w_bar > max_pixels {
        let beta = ((height * width) as f64 / max_pixels as f64).sqrt();
        let h2 = ((height as f64 / beta / FACTOR as f64).floor() as u64) * FACTOR;
        let w2 = ((width as f64 / beta / FACTOR as f64).floor() as u64) * FACTOR;
        Ok((h2.max(FACTOR), w2.max(FACTOR)))
    } else if h_bar * w_bar < min_pixels {
        let beta = (min_pixels as f64 / ((height * width) as f64)).sqrt();
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
/// #107: the resized pixel size (h, w) of an `h0` x `w0` source under
/// `b`: HF `smart_resize` with the budget's min/max pixels, then the patch-cap
/// clamp for the one case smart_resize can overshoot (the floor's ceil rounding
/// on an extreme aspect ratio). Pure host math, unit-tested against known grids.
pub fn resized_dims(h0: u64, w0: u64, b: &VitBudget) -> Result<(usize, usize), String> {
    let (rh, rw) = smart_resize(h0, w0, b.min_pixels(), b.max_pixels())?;
    let (mut rh, mut rw) = (rh as usize, rw as usize);
    let cap = b.max_patches();
    // #VIT cap: an image over the patch budget is DOWNSCALED further, never
    // refused - a screenshot must reach the model at the resolution a chat
    // needs, the same clamp every production VLM applies. Dims stay
    // FACTOR-multiples so the patch grid keeps its merge alignment.
    const FACTOR: usize = VIT_PATCH * VIT_MERGE;
    while (rh / VIT_PATCH) * (rw / VIT_PATCH) > cap {
        let beta = ((rh * rw) as f64 / (cap * VIT_PATCH * VIT_PATCH) as f64).sqrt() * 1.06;
        let nh = (((rh as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        let nw = (((rw as f64 / beta) / FACTOR as f64).floor() as usize).max(1) * FACTOR;
        if nh >= rh && nw >= rw {
            break; // degenerate: cannot shrink further, accept as is
        }
        rh = nh;
        rw = nw;
    }
    Ok((rh, rw))
}

/// the `image_grid_thw` row and the visual-token count `resized_dims` gives
pub fn grid_for(h0: u64, w0: u64, b: &VitBudget) -> Result<((usize, usize, usize), usize), String> {
    let (rh, rw) = resized_dims(h0, w0, b)?;
    let (hp, wp) = (rh / VIT_PATCH, rw / VIT_PATCH);
    Ok(((1, hp, wp), hp * wp / (VIT_MERGE * VIT_MERGE)))
}

pub fn prep_image(bytes: &[u8]) -> Result<ImagePrep, String> {
    prep_image_with(bytes, &budget())
}

/// `prep_image` under an explicit window (the layout tests pin the
/// preprocessor's own 64-token floor so a 256x256 raster is not resized)
pub fn prep_image_with(bytes: &[u8], b: &VitBudget) -> Result<ImagePrep, String> {
    let (rgb, w0, h0) = decode_rgb(bytes)?;
    let (rh, rw) = resized_dims(h0 as u64, w0 as u64, b)?;
    let (w0, h0) = (w0 as usize, h0 as usize);
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

/// - #114: one image's place in the EXPANDED prompt and its content identity
/// - `start..start + len` are the rows its visual tokens occupy (all `IMAGE_PAD` ids)
/// - `hash` is `image_key` of the image bytes, the key the tower-output cache uses
/// - the prefix cache compares these next to the ids: `IMAGE_PAD` ids alone say
///   nothing about WHICH image sits there (`cache::common_prefix_len_mm`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ImageSpan {
    pub start: usize,
    pub len: usize,
    pub hash: u64,
}

/// - #114: the content identity of one image: a 64-bit hash of its encoded bytes
/// - the ONE definition, shared by the tower-output cache (`Vit::build_plan`) and the
///   prefix cache's image spans, so both agree on what "the same image" is
/// - `DefaultHasher::new()` is SipHash-1-3 with fixed keys: stable within a process,
///   which is all either cache needs (neither outlives the process)
pub fn image_key(bytes: &[u8]) -> u64 {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::{Hash, Hasher};
    let mut hasher = DefaultHasher::new();
    bytes.hash(&mut hasher);
    hasher.finish()
}

/// - #114: the spans of the images in the EXPANDED prompt, from the RENDERED ids (one
///   `IMAGE_PAD` per image), each image's visual token count and its hash
/// - the same walk `expand_ids` does, so `start` is exactly where the splice puts it
pub fn image_spans(ids: &[u32], counts: &[usize], hashes: &[u64]) -> Result<Vec<ImageSpan>, String> {
    if counts.len() != hashes.len() {
        return Err(format!("{} image counts but {} image hashes", counts.len(), hashes.len()));
    }
    let mut out = Vec::with_capacity(counts.len());
    let mut row = 0usize;
    let mut k = 0usize;
    for &id in ids {
        if id as i64 == IMAGE_PAD {
            let len = *counts.get(k).ok_or("more image_pad tokens than images")?;
            out.push(ImageSpan { start: row, len, hash: hashes[k] });
            row += len;
            k += 1;
        } else {
            row += 1;
        }
    }
    if k != counts.len() {
        return Err("more images than image_pad tokens".into());
    }
    Ok(out)
}

/// the per-request vision plan: where the visual embeddings land in the
/// (already expanded) prompt id list, the embeddings themselves, and the
/// mrope delta the decode path needs
pub struct VisionPlan {
    /// the expanded prompt ids (every IMAGE_PAD replaced by its visual tokens)
    pub ids: Vec<u32>,
    /// [n_visual][2560] f32 on the HOST — the merged embeddings of ALL images,
    /// concatenated in message order. The prefill splices from here row by row
    /// into the chunk's embedding upload, so no per-request device buffer exists
    /// (the one there was, sum(n_visual) x 2560 f32, was never read and was what
    /// a 12-image history request of 2026-09-17 could not allocate).
    pub embeds_host: Vec<f32>,
    /// per final-sequence row: flat visual embedding index, or -1 for text
    pub map: Vec<i32>,
    /// mm_token_type per row (0 text, 1 image) — the get_rope_index input
    pub types: Vec<u8>,
    pub grids: Vec<Grid>,
    pub n_visual: usize,
    /// mrope_position_deltas of the oracle (max_pos + 1 - seq_len)
    pub delta: i64,
    /// #114: each image's rows in `ids` and its content hash, message order: what the
    /// prefix cache compares so two prompts are equal only if their images are
    pub spans: Vec<ImageSpan>,
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

/// #72: the image cache's ceiling for the `[budget]` post-plan line. HOST RAM -
/// the entries are `Vec<f32>` tower outputs, never device memory - so it is
/// listed on the host half of that line and costs the card nothing.
pub fn image_cache_budget_bytes() -> u64 {
    vit_cache_bytes() as u64
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
        let mut hashes: Vec<u64> = Vec::with_capacity(images.len());
        for (i, bytes) in images.iter().enumerate() {
            let key = image_key(bytes);
            hashes.push(key);
            if let Some((grid, n_visual, rows)) = self.image_cache.get(&key) {
                tracing::info!(target: "vit", "[vit-cache] image {i}: HIT grid {grid:?}, {n_visual} visual tokens");
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
        tracing::info!(target: "vit",
            "[vit-cache] {} image(s): {} through the tower, {} cached; cache {} entries, {:.1} MiB of {} MiB",
            images.len(), misses, images.len() - misses,
            self.image_cache.len(), self.image_cache_bytes as f64 / MIB,
            vit_cache_bytes() >> 20
        );
        let counts: Vec<usize> = infos.iter().map(|(_, n)| *n).collect();
        let expanded = expand_ids(ids, &counts)?;
        let spans = image_spans(ids, &counts, &hashes)?;
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
        Ok(VisionPlan {
            ids: expanded,
            embeds_host,
            map,
            types,
            grids: infos.iter().map(|(g, _)| *g).collect(),
            n_visual: flat as usize,
            delta,
            spans,
        })
    }
}

/// raw f32 little-endian file write, failures logged not raised.
/// `pub` because serve's `/v1` logits dump writes the same file the same way.
pub fn f32_file(path: &str, v: &[f32]) {
    if let Err(e) = cuda::write_le(path, v) {
        tracing::warn!(target: "vit", "[vit-dump] write {path} failed: {e}");
    }
}


#[cfg(test)]
mod reserve {
    //! TASK K: the numbers the planner subtracts before it chooses N. They are pure
    //! geometry, so they are asserted without a GPU and without a container; the
    //! `debug_assert_eq!` at the end of `ensure_scratch` is what keeps `scratch_bytes`
    //! tied to the twelve allocations it counts.
    use super::*;

    #[test]
    fn the_scratch_is_the_twelve_buffers_the_allocator_takes() {
        // x + normed + attn: 3 x 4096 x 1152, qkv 4096 x 3456, mlp 4096 x 4304,
        // m1 1024 x 4608, out 1024 x 2560, patches 4096 x 1536, pe_idx + pe_w 2 x 16384,
        // cs + sn 2 x 4096 x 36 - all f32 (the pre-#107 cap, 1024 tokens)
        assert_eq!(scratch_bytes_for(4096), 239_599_616);
        // 58,496 B per patch; the default cap 1280 tokens = 5120 patches = 285.6 MiB
        assert_eq!(scratch_bytes_for(5120), 299_499_520);
        assert_eq!(VitBudget::DEFAULT.max_patches(), 5120);
        assert_eq!(scratch_bytes(), scratch_bytes_for(budget().max_patches()));
    }

    /// the span tables are `n_ctx` rows since serve clamps the budget before arming them
    #[test]
    fn the_mrope_span_is_the_whole_context_cos_and_sin() {
        assert_eq!(mrope_bytes(200_000), 51_200_000);
        assert_eq!(mrope_bytes(262_144), 67_108_864);
    }

    #[test]
    fn the_reserve_is_the_sum_and_the_budget_line_names_its_parts() {
        let ctx = 200_000;
        assert_eq!(
            reserve_bytes(ctx),
            (scratch_bytes() + mrope_bytes(ctx)) as u64
        );
        let line = reserve_line(ctx, 0);
        assert!(line.starts_with("vit reserve"), "the [budget] label moved: {line}");
        assert!(line.contains("334.5 MB"), "the reserve total moved: {line}");
        assert!(line.contains("tower scratch 285.6"), "the scratch part moved: {line}");
        assert!(line.contains("mrope span 48.8"), "the mrope part moved: {line}");
    }

    /// #72 gate 1: the derived reserve is EXACTLY what `Engine::load` holds at
    /// boot - the twelve scratch buffers at the patch cap plus the two span
    /// tables at `n_ctx`. If the two ever drift apart the planner is back to
    /// promising VRAM nobody took.
    #[test]
    fn the_held_bytes_are_the_whole_derived_reserve_and_nothing_stays_pending() {
        for ctx in [200_000usize, 262_144] {
            let held = (scratch_bytes() + mrope_bytes(ctx)) as u64;
            assert_eq!(held, reserve_bytes(ctx), "ctx {ctx}: the hold is not the reserve");
            assert_eq!(reserve_bytes(ctx).saturating_sub(held), 0, "ctx {ctx}: bytes left pending");
        }
        // the twelve buffers, each counted from the geometry ensure_scratch uses
        let cap = budget().max_patches();
        let buffers: [usize; VIT_SCRATCH_BUFFERS] = [
            cap * VIT_HIDDEN,      // x
            cap * VIT_HIDDEN,      // normed
            cap * VIT_QKV,         // qkv
            cap * VIT_HIDDEN,      // attn
            cap * VIT_INTER,       // mlp
            cap / 4 * VIT_MERGED,  // m1
            cap / 4 * H,           // out
            cap * VIT_IN,          // patches
            cap * 4,               // pe_idx
            cap * 4,               // pe_w
            cap * VIT_ROT,         // cs
            cap * VIT_ROT,         // sn
        ];
        assert_eq!(buffers.iter().sum::<usize>() * 4, scratch_bytes());
    }

    /// #72 gate 2: the `[budget]` line says HELD once the loader holds it, and
    /// still says lazy when `CROW_VIT_RESERVE_MB=0` turns the hold off.
    #[test]
    fn the_budget_line_says_whether_the_reserve_is_held_or_only_planned() {
        let ctx = 200_000;
        let held = reserve_line(ctx, reserve_bytes(ctx));
        assert!(held.contains("HELD at boot"), "the held wording moved: {held}");
        assert!(held.contains("nothing left pending"), "the pending clause moved: {held}");
        let lazy = reserve_line(ctx, 0);
        assert!(lazy.contains("lazy, allocated on the first image request"), "the lazy wording moved: {lazy}");
        // a bigger override pads the plan by the difference, and says so
        let part = reserve_line(ctx, scratch_bytes() as u64);
        assert!(part.contains("48.8 MB pending"), "the partial-hold wording moved: {part}");
    }
}

#[cfg(test)]
mod layout {
    //! #73: the patch layout `prep_image` hands the tower, pinned against values
    //! computed by hand from the reference formulas. No GPU, no container.
    //!
    //! #73 itself was NOT a layout bug - it was the missing stream sync in
    //! `Vit::run`, which let the host read the previous image's embeddings back.
    //! But the deterministic colour permutation it produced (red reads as Black,
    //! green as Red) is exactly the signature a channel swap or a patch-order
    //! transpose would leave, and suspect 2 of the issue cost the longest to
    //! clear. These tests hold it cleared: a `plane[i] = px[c]` that picked BGR,
    //! a `((c * T + t) * P + ty) * P + tx` with two axes exchanged, a merge-block
    //! `seq` that walked columns before rows, an `align_corners` tap, or an h/w
    //! swap in the rotary would each move a literal below.

    use super::*;

    /// the preprocessor's own 64-token floor: a 256x256 raster stays 16x16
    /// patches (the serving floor of 1024 tokens would upscale it)
    const HF_WINDOW: VitBudget = VitBudget { min_tokens: 64, max_tokens: 1024 };

    /// encode an RGB8 raster as a PNG, which is what `prep_image` takes
    fn png(w: u32, h: u32, px: impl Fn(u32, u32) -> [u8; 3]) -> Vec<u8> {
        use image::ImageEncoder;
        let mut raw = Vec::with_capacity((w * h * 3) as usize);
        for y in 0..h {
            for x in 0..w {
                raw.extend_from_slice(&px(x, y));
            }
        }
        let mut out = Vec::new();
        image::codecs::png::PngEncoder::new(&mut out)
            .write_image(&raw, w, h, image::ExtendedColorType::Rgb8)
            .expect("png encode");
        out
    }

    /// `((c * VIT_TPATCH + t) * VIT_PATCH + ty) * VIT_PATCH + tx`, written out
    /// here so the test does not borrow the expression it is checking
    fn at(c: usize, t: usize, ty: usize, tx: usize) -> usize {
        c * 512 + t * 256 + ty * 16 + tx
    }

    fn close(got: f32, want: f32, what: &str) {
        assert!((got - want).abs() < 1e-6, "{what}: got {got}, want {want}");
    }

    /// A 256x256 image is a FACTOR multiple already, so `smart_resize` keeps it
    /// and the bicubic pass is the identity (scale 1, taps 1/0/0) - every value
    /// below is therefore a pixel of the source, normalized, and nothing depends
    /// on the resampler. Pixel (x, y) carries its own coordinates: R = x, G = y,
    /// B = 7, so a swapped channel or a transposed patch axis reads as a
    /// different NUMBER, not as a different shade.
    #[test]
    fn the_patch_rows_are_channel_major_with_the_temporal_frame_duplicated() {
        let p = prep_image_with(&png(256, 256, |x, y| [x as u8, y as u8, 7]), &HF_WINDOW).expect("prep");
        assert_eq!(p.grid, (1, 16, 16));
        assert_eq!(p.resized, (256, 256));
        assert_eq!(p.n_patches, 256);
        assert_eq!(p.n_visual, 64);
        assert_eq!(p.patches.len(), 256 * VIT_IN);

        // ---- seq 0 = block (0,0), in-block (0,0) = patch (row 0, col 0) ----
        let r0 = &p.patches[0..VIT_IN];
        // R plane carries x, G carries y, B is the constant 7
        close(r0[at(0, 0, 3, 5)], 5.0 / 127.5 - 1.0, "R at (tx 5, ty 3)");
        close(r0[at(1, 0, 3, 5)], 3.0 / 127.5 - 1.0, "G at (tx 5, ty 3)");
        close(r0[at(2, 0, 3, 5)], 7.0 / 127.5 - 1.0, "B at (tx 5, ty 3)");
        // the temporal patch is the SAME frame twice
        close(r0[at(0, 1, 3, 5)], r0[at(0, 0, 3, 5)], "R frame 1 duplicates frame 0");
        close(r0[at(2, 1, 3, 5)], r0[at(2, 0, 3, 5)], "B frame 1 duplicates frame 0");
        // the literals themselves, so a changed normalize is caught too
        close(r0[at(0, 0, 3, 5)], -0.960_784_3, "R literal");
        close(r0[at(1, 0, 3, 5)], -0.976_470_6, "G literal");
        close(r0[at(2, 0, 3, 5)], -0.945_098_04, "B literal");

        // ---- the merge block walks in-row before in-column, rows before blocks ----
        // seq 1 = in-block (iy 0, ix 1) = patch (row 0, col 1), pixels x 16..31
        let r1 = &p.patches[VIT_IN..2 * VIT_IN];
        close(r1[at(0, 0, 0, 0)], 16.0 / 127.5 - 1.0, "seq 1 is the RIGHT neighbour (R = x = 16)");
        close(r1[at(1, 0, 0, 0)], -1.0, "seq 1 stays on row 0 (G = y = 0)");
        // seq 2 = in-block (iy 1, ix 0) = patch (row 1, col 0), pixels y 16..31
        let r2 = &p.patches[2 * VIT_IN..3 * VIT_IN];
        close(r2[at(0, 0, 0, 0)], -1.0, "seq 2 stays on col 0 (R = x = 0)");
        close(r2[at(1, 0, 0, 0)], 16.0 / 127.5 - 1.0, "seq 2 is the patch BELOW (G = y = 16)");
        // seq 4 = the next BLOCK along the row = patch (row 0, col 2), pixels x 32..47
        let r4 = &p.patches[4 * VIT_IN..5 * VIT_IN];
        close(r4[at(0, 0, 0, 0)], 32.0 / 127.5 - 1.0, "seq 4 is the next block right (R = x = 32)");
        close(r4[at(1, 0, 0, 0)], -1.0, "seq 4 stays on block row 0 (G = y = 0)");
        // seq 32 = block row 1 (8 blocks of 4 per row) = patch (row 2, col 0)
        let r32 = &p.patches[32 * VIT_IN..33 * VIT_IN];
        close(r32[at(0, 0, 0, 0)], -1.0, "seq 32 stays on col 0 (R = x = 0)");
        close(r32[at(1, 0, 0, 0)], 32.0 / 127.5 - 1.0, "seq 32 is block row 1 (G = y = 32)");
    }

    /// The colour check the end-to-end probe makes, one step earlier: a solid
    /// image must give 256 identical rows whose three 512-value halves are the
    /// normalized R, G and B of that colour, in THAT order. This is the assertion
    /// `red -> Black` would have failed first if the cause had been the channels.
    #[test]
    fn a_solid_colour_fills_every_patch_row_with_that_colour_in_rgb_order() {
        let (r, g, b) = (230u8, 20u8, 20u8);
        let p = prep_image_with(&png(256, 256, |_, _| [r, g, b]), &HF_WINDOW).expect("prep");
        let (nr, ng, nb) = (
            r as f32 / 127.5 - 1.0,
            g as f32 / 127.5 - 1.0,
            b as f32 / 127.5 - 1.0,
        );
        close(nr, 0.803_921_6, "the red literal");
        close(ng, -0.843_137_25, "the green literal");
        for seq in [0usize, 1, 7, 128, 255] {
            let row = &p.patches[seq * VIT_IN..(seq + 1) * VIT_IN];
            // the three 512-value halves, in the order the conv row expects them
            for (name, want, half) in [
                ("R", nr, &row[0..512]),
                ("G", ng, &row[512..1024]),
                ("B", nb, &row[1024..1536]),
            ] {
                for (i, &v) in half.iter().enumerate() {
                    close(v, want, &format!("seq {seq} {name} slot {i}"));
                }
            }
        }
    }

    /// The learned 48x48 position table is resampled with align_corners=True:
    /// src = p * (side - 1) / (grid - 1), taps floor/floor+1, weights (1-d, d),
    /// emitted (h0w0, h0w1, h1w0, h1w1). Patch (row 1, col 0) of a 16x16 grid
    /// lands at 1 * 47 / 15 = 3.1333, so it splits 0.8667 / 0.1333 over rows 3
    /// and 4 of the table and sits whole on column 0.
    #[test]
    fn the_position_table_taps_are_align_corners_bilinear_in_h0w0_order() {
        let p = prep_image_with(&png(256, 256, |_, _| [9, 9, 9]), &HF_WINDOW).expect("prep");
        // seq 0 = patch (0, 0): the exact corner, all weight on the first tap
        assert_eq!(&p.pe_idx[0..4], &[0, 1, 48, 49], "seq 0 taps");
        close(p.pe_w[0], 1.0, "seq 0 w00");
        close(p.pe_w[1], 0.0, "seq 0 w01");
        close(p.pe_w[2], 0.0, "seq 0 w10");
        close(p.pe_w[3], 0.0, "seq 0 w11");
        // seq 2 = patch (1, 0): split over table rows 3 and 4, column 0
        assert_eq!(&p.pe_idx[8..12], &[144, 145, 192, 193], "seq 2 taps");
        close(p.pe_w[8], 0.866_666_7, "seq 2 w00");
        close(p.pe_w[9], 0.0, "seq 2 w01");
        close(p.pe_w[10], 0.133_333_3, "seq 2 w10");
        close(p.pe_w[11], 0.0, "seq 2 w11");
        // the four weights of any patch are a partition of one
        for seq in [0usize, 2, 5, 100, 255] {
            let s: f32 = p.pe_w[seq * 4..seq * 4 + 4].iter().sum();
            close(s, 1.0, &format!("seq {seq} weights sum"));
        }
    }

    /// The rotary vector is [h * inv_0 .. h * inv_17, w * inv_0 .. w * inv_17]:
    /// the FIRST half is the patch row, the second the patch column, and
    /// inv_freq[0] is 1. An h/w swap moves cos(1) and cos(2) past each other.
    #[test]
    fn the_rotary_half_is_the_patch_row_and_the_second_half_the_patch_column() {
        let p = prep_image_with(&png(256, 256, |_, _| [9, 9, 9]), &HF_WINDOW).expect("prep");
        let half = VIT_ROT / 2; // 18
        // seq 2 = patch (row 1, col 0)
        close(p.cs[2 * VIT_ROT], 1f32.cos(), "seq 2 h slot 0 = cos(1)");
        close(p.sn[2 * VIT_ROT], 1f32.sin(), "seq 2 h slot 0 = sin(1)");
        close(p.cs[2 * VIT_ROT + half], 1.0, "seq 2 w slot 0 = cos(0)");
        close(p.sn[2 * VIT_ROT + half], 0.0, "seq 2 w slot 0 = sin(0)");
        close(p.cs[2 * VIT_ROT], 0.540_302_3, "cos(1) literal");
        close(p.sn[2 * VIT_ROT], 0.841_471, "sin(1) literal");
        // seq 4 = patch (row 0, col 2): the mirror image of the above
        close(p.cs[4 * VIT_ROT], 1.0, "seq 4 h slot 0 = cos(0)");
        close(p.cs[4 * VIT_ROT + half], 2f32.cos(), "seq 4 w slot 0 = cos(2)");
        close(p.sn[4 * VIT_ROT + half], 2f32.sin(), "seq 4 w slot 0 = sin(2)");
        close(p.cs[4 * VIT_ROT + half], -0.416_146_84, "cos(2) literal");
    }
}

#[cfg(test)]
mod attn_98 {
    //! #98: the shipped `vit_attn` (step 2, 16-row tiles) against the kernel
    //! #98 replaced, on the GPU, bit for bit; step 1 rides along, checked and
    //! timed. The old kernel is kept
    //! here VERBATIM (renamed `vit_attn_pre98`, nothing else changed) as the
    //! oracle; the new one is cut out of KERNEL_SRC itself, so the test checks
    //! the text that ships. Synthetic q/k/v, no weights, no container, a few
    //! hundred MB of VRAM at the 4,000-patch shape.
    //!
    //! `#[ignore]`: CI has no GPU (ci.yml) and the gate counts the host tests.
    //! Run it with `cargo test --release attn_98 -- --ignored --nocapture`; it
    //! prints the per-layer kernel time old vs new (cuEvent) at the logged
    //! image shapes.
    use super::*;
    use cudarc::driver::sys;

    /// the pre-#98 kernel, byte for byte as of b6d57f8 except its name
    const PRE98: &str = r#"
extern "C" __global__ void vit_attn_pre98(const float* __restrict__ qkv, float* __restrict__ out,
                                    const int* __restrict__ n_p) {
    int n = *n_p;
    int qi = blockIdx.x;
    int h = blockIdx.y;
    int t = threadIdx.x;
    const float* qp = qkv + (size_t)qi * 3456 + h * 72;
    __shared__ float qs[72];
    __shared__ float red[256];   // per-tile scores, then the sum partials
    __shared__ float ps[256];    // the tile's probabilities, kept for the V walk
    if (t < 72) qs[t] = qp[t];
    __syncthreads();
    float acc[72];
    #pragma unroll
    for (int d = 0; d < 72; d++) acc[d] = 0.0f;
    float m = -__int_as_float(0x7f800000), l = 0.0f;
    const float SCALE = 0.11785113019775793f;   // 1/sqrt(72)
    for (int kt = 0; kt < n; kt += 256) {
        int j = kt + t;
        float s = -__int_as_float(0x7f800000);
        if (j < n) {
            const float* kp = qkv + (size_t)j * 3456 + 1152 + h * 72;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < 72; d++) dot += qs[d] * kp[d];
            s = dot * SCALE;
        }
        red[t] = s;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (t < st) red[t] = fmaxf(red[t], red[t + st]);
            __syncthreads();
        }
        float mn = red[0];
        __syncthreads();
        float mn_new = fmaxf(m, mn);
        float p = (j < n) ? expf(s - mn_new) : 0.0f;
        red[t] = p;
        ps[t] = p;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (t < st) red[t] += red[t + st];
            __syncthreads();
        }
        float ln = red[0];
        __syncthreads();
        float r = expf(m - mn_new);
        l = l * r + ln;
        #pragma unroll
        for (int d = 0; d < 72; d++) acc[d] *= r;
        m = mn_new;
        // EVERY thread t < 72 walks the WHOLE tile's probabilities — output
        // dim t accumulates p_j * v_j[t] over ALL keys, not one key per tile
        if (t < 72) {
            int lim = (n - kt) < 256 ? (n - kt) : 256;
            for (int jj = 0; jj < lim; jj++) {
                const float* vp = qkv + (size_t)(kt + jj) * 3456 + 2304 + h * 72;
                #pragma unroll
                for (int d = 0; d < 72; d++) acc[d] += ps[jj] * vp[d];
            }
        }
        __syncthreads();
    }
    if (t < 72) {
        float* op = out + ((size_t)qi * 16 + h) * 72;
        #pragma unroll
        for (int d = 0; d < 72; d++) op[d] = acc[d] / l;
    }
}"#;

    /// step 1 (834f983, one accumulator per dim, one query row per block),
    /// timed beside the other two so the record keeps both steps' gains
    const STEP1: &str = r#"
extern "C" __global__ void vit_attn_s1(const float* __restrict__ qkv, float* __restrict__ out,
                                    const int* __restrict__ n_p) {
    int n = *n_p;
    int qi = blockIdx.x;
    int h = blockIdx.y;
    int t = threadIdx.x;
    const float* qp = qkv + (size_t)qi * 3456 + h * 72;
    __shared__ float qs[72];
    __shared__ float red[256];   // per-tile scores, then the sum partials
    __shared__ float ps[256];    // the tile's probabilities, kept for the V walk
    if (t < 72) qs[t] = qp[t];
    __syncthreads();
    float acc = 0.0f;   // output dim t (threads t >= 72 carry an unused 0)
    float m = -__int_as_float(0x7f800000), l = 0.0f;
    const float SCALE = 0.11785113019775793f;   // 1/sqrt(72)
    for (int kt = 0; kt < n; kt += 256) {
        int j = kt + t;
        float s = -__int_as_float(0x7f800000);
        if (j < n) {
            const float* kp = qkv + (size_t)j * 3456 + 1152 + h * 72;
            float dot = 0.0f;
            #pragma unroll 8
            for (int d = 0; d < 72; d++) dot += qs[d] * kp[d];
            s = dot * SCALE;
        }
        red[t] = s;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (t < st) red[t] = fmaxf(red[t], red[t + st]);
            __syncthreads();
        }
        float mn = red[0];
        __syncthreads();
        float mn_new = fmaxf(m, mn);
        float p = (j < n) ? expf(s - mn_new) : 0.0f;
        red[t] = p;
        ps[t] = p;
        __syncthreads();
        for (int st = 128; st > 0; st >>= 1) {
            if (t < st) red[t] += red[t + st];
            __syncthreads();
        }
        float ln = red[0];
        __syncthreads();
        float r = expf(m - mn_new);
        l = l * r + ln;
        acc *= r;
        m = mn_new;
        // EVERY thread t < 72 walks the WHOLE tile's probabilities — output
        // dim t accumulates p_j * v_j[t] over ALL keys, not one key per tile;
        // the 72 v_j[t] reads of one jj are one coalesced 288-byte row
        if (t < 72) {
            int lim = (n - kt) < 256 ? (n - kt) : 256;
            const float* vp = qkv + (size_t)kt * 3456 + 2304 + h * 72 + t;
            for (int jj = 0; jj < lim; jj++) acc += ps[jj] * vp[(size_t)jj * 3456];
        }
        __syncthreads();
    }
    if (t < 72) out[((size_t)qi * 16 + h) * 72 + t] = acc / l;
}"#;

    /// the shipped `vit_attn`, from its `extern "C"` line to its closing brace
    fn shipped() -> &'static str {
        let src = crate::kernels::KERNEL_SRC;
        let a = src
            .find("extern \"C\" __global__ void vit_attn(")
            .expect("vit_attn is in KERNEL_SRC");
        let e = src[a..].find("\n}\n").expect("vit_attn closes") + a + 3;
        &src[a..e]
    }

    /// xorshift32, uniform in [-amp, amp): deterministic, no rand crate
    fn fill(n: usize, seed: u32, amp: f32) -> Vec<f32> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                ((x >> 8) as f32 / 16_777_216.0 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    /// f64 softmax(q k^T / sqrt 72) v for one (query, head): the sanity bound
    /// that the two kernels agreeing is not two kernels agreeing on garbage
    fn reference(qkv: &[f32], n: usize, qi: usize, h: usize) -> Vec<f64> {
        let at = |t: usize, third: usize, d: usize| qkv[t * VIT_QKV + third * 1152 + h * 72 + d] as f64;
        let s: Vec<f64> = (0..n)
            .map(|j| (0..72).map(|d| at(qi, 0, d) * at(j, 1, d)).sum::<f64>() / 72f64.sqrt())
            .collect();
        let m = s.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let p: Vec<f64> = s.iter().map(|x| (x - m).exp()).collect();
        let l: f64 = p.iter().sum();
        (0..72).map(|d| (0..n).map(|j| p[j] * at(j, 2, d)).sum::<f64>() / l).collect()
    }

    unsafe fn elapsed_ms(a: sys::CUevent, b: sys::CUevent) -> f64 {
        let mut ms = 0.0f32;
        cuda::ck(sys::cuEventElapsedTime_v2(&mut ms, a, b));
        ms as f64
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release attn_98 -- --ignored --nocapture"]
    fn the_shipped_vit_attn_is_bit_identical_to_the_pre_98_kernel() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut module = cuda::compile(&format!("{PRE98}\n{STEP1}\n{}", shipped()));
            let (f_old, f_s1, f_new) = (module.get("vit_attn_pre98"), module.get("vit_attn_s1"), module.get("vit_attn"));
            // timing events (`cuda::event_create` makes DISABLE_TIMING ones)
            let ev = || {
                let mut e: sys::CUevent = std::ptr::null_mut();
                cuda::ck(sys::cuEventCreate(&mut e, 0));
                e
            };
            let (e0, e1) = (ev(), ev());
            let s = cuda::cur_stream();
            // tile edges (1, 255/256/257, 512), the acceptance n = 1,000 and
            // the three logged image shapes (912 / 3,520 / 4,000 patches);
            // amp 3 gives peaked rows, amp 12 drives the running-max rescale
            let cases: &[(usize, f32)] = &[
                (1, 3.0), (255, 3.0), (256, 3.0), (257, 3.0), (512, 12.0),
                (912, 3.0), (1000, 3.0), (1000, 12.0), (3520, 3.0), (4000, 3.0),
            ];
            for (ci, &(n, amp)) in cases.iter().enumerate() {
                let host = fill(n * VIT_QKV, 0x9e37_79b9 ^ (ci as u32 * 7919 + n as u32), amp);
                let mut qkv = cuda::to_f32_dev(&host);
                let mut o_old = cuda::alloc_zeroed(n * VIT_HIDDEN * 4);
                let mut o_new = cuda::alloc_zeroed(n * VIT_HIDDEN * 4);
                let mut n_p = cuda::to_i32_dev(&[n as i32]);
                // the two row kernels launch one block per query row, the
                // shipped one a block per VIT_ATTN_ROWS rows
                let timed = |f, gx: usize, o: Dev| {
                    cuda::event_record(e0, s);
                    launch_v(f, gx as u32, VIT_HEADS as u32, 1, 256, &[qkv, o, n_p]);
                    cuda::event_record(e1, s);
                    cuda::stream_sync(s);
                    elapsed_ms(e0, e1)
                };
                let t_old = timed(f_old, n, o_old);
                let t_s1 = timed(f_s1, n, o_new);
                let s1_bits = cuda::dtoh_u32(o_new, n * VIT_HIDDEN);
                let t_new = timed(f_new, n.div_ceil(VIT_ATTN_ROWS), o_new);
                let (a, b) = (cuda::dtoh_u32(o_old, n * VIT_HIDDEN), cuda::dtoh_u32(o_new, n * VIT_HIDDEN));
                let diff = a.iter().zip(&b).filter(|(x, y)| x != y).count();
                assert_eq!(diff, 0, "n {n} amp {amp}: {diff} of {} outputs differ in their bits", a.len());
                assert!(s1_bits == a, "n {n} amp {amp}: step 1 left the pre-#98 bits");
                // both are the attention, not merely equal: f64 on three rows.
                // The f32 score error grows with |q||k| (amp^2), and so does
                // the band: 9e-5 at amp 3, 1.4e-3 at amp 12 (scores ~ +-200)
                let tol = 1e-5 * (amp as f64) * (amp as f64);
                let got: Vec<f32> = b.iter().map(|&u| f32::from_bits(u)).collect();
                for (qi, h) in [(0, 0), (n / 2, 7), (n - 1, VIT_HEADS - 1)] {
                    let want = reference(&host, n, qi, h);
                    for d in 0..72 {
                        let g = got[(qi * VIT_HEADS + h) * 72 + d] as f64;
                        assert!((g - want[d]).abs() < tol, "n {n} q {qi} h {h} d {d}: {g} vs {}", want[d]);
                    }
                }
                println!(
                    "[#98] n {n:>4} amp {amp:>4}: {} outputs bit-identical; per layer pre-#98 {t_old:9.3} ms, step 1 {t_s1:7.3} ms, shipped {t_new:7.3} ms (x{:.0} / x{:.1}); x27 layers {:.2} s -> {:.3} s -> {:.3} s",
                    a.len(), t_old / t_new, t_s1 / t_new, t_old * 27.0 / 1e3, t_s1 * 27.0 / 1e3, t_new * 27.0 / 1e3
                );
                for p in [&mut qkv, &mut o_old, &mut o_new, &mut n_p] {
                    cuda::free_dev(p);
                }
            }
            cuda::event_destroy(e0);
            cuda::event_destroy(e1);
            module.unload();
        }
    }
}

#[cfg(test)]
mod budget_and_mmproj {
    //! #107 and #108: the resize window, the projector file
    //! choice and the projector layout, all host-side (no GPU).
    use super::*;
    use crate::gguf::{self, Gguf, GGML_TYPE_F16, GGML_TYPE_F32};

    const OLD: VitBudget = VitBudget { min_tokens: 64, max_tokens: 1024 };

    #[test]
    fn the_old_window_reproduces_the_grids_the_engine_logged() {
        // [vit-chat] grids of record: 1280x720 -> (1, 44, 80) = 880 (#98 closing,
        // 2026-09-22); 1000x560 -> 558 and a 992x544 render -> (1, 34, 62) = 527
        // (engine.log, 2026-09-23)
        assert_eq!(grid_for(720, 1280, &OLD).unwrap(), ((1, 44, 80), 880));
        assert_eq!(grid_for(560, 1000, &OLD).unwrap(), ((1, 36, 62), 558));
        assert_eq!(grid_for(544, 992, &OLD).unwrap(), ((1, 34, 62), 527));
    }

    #[test]
    fn the_default_window_gives_crows_renders_the_llama_cpp_token_count() {
        let b = VitBudget::DEFAULT;
        assert_eq!((b.min_tokens, b.max_tokens), (1024, 1280));
        // llama.cpp at --image-min-tokens 1024 gives both renders 1032 tokens
        assert_eq!(grid_for(560, 1000, &b).unwrap(), ((1, 48, 86), 1032));
        assert_eq!(grid_for(720, 1280, &b).unwrap(), ((1, 48, 86), 1032));
        assert_eq!(grid_for(544, 992, &b).unwrap(), ((1, 48, 88), 1056));
        // the 1097x380 paste that read as "no image" at ~350 tokens
        assert_eq!(grid_for(380, 1097, &b).unwrap(), ((1, 38, 110), 1045));
        // over the cap: downscaled into it
        assert_eq!(grid_for(1080, 1920, &b).unwrap(), ((1, 52, 94), 1222));
        assert_eq!(grid_for(2160, 3840, &b).unwrap(), ((1, 52, 94), 1222));
        assert_eq!(grid_for(1024, 1024, &b).unwrap(), ((1, 64, 64), 1024));
    }

    #[test]
    fn every_grid_stays_inside_the_cap_and_on_the_merge_lattice() {
        for b in [OLD, VitBudget::DEFAULT, VitBudget { min_tokens: 1024, max_tokens: 1024 }] {
            for h in (32..3000u64).step_by(97) {
                for w in (32..3000u64).step_by(89) {
                    if h.max(w) / h.min(w) > 200 {
                        continue;
                    }
                    let (rh, rw) = resized_dims(h, w, &b).unwrap();
                    assert!(rh % 32 == 0 && rw % 32 == 0, "{h}x{w} -> {rh}x{rw}");
                    let n = (rh / 16) * (rw / 16);
                    assert!(n <= b.max_patches(), "{h}x{w} under {b:?}: {n} patches");
                    // the floor holds wherever the cap leaves room for it
                    if b.max_tokens >= b.min_tokens + b.min_tokens / 4 && h.max(w) / h.min(w) < 8 {
                        assert!(n / 4 >= b.min_tokens, "{h}x{w} under {b:?}: {} tokens", n / 4);
                    }
                }
            }
        }
    }

    #[test]
    fn the_env_window_parses_and_a_bad_value_is_an_error() {
        assert_eq!(VitBudget::parse(None, None), Ok(VitBudget::DEFAULT));
        assert_eq!(VitBudget::parse(Some("256"), Some("1280")), Ok(VitBudget { min_tokens: 256, max_tokens: 1280 }));
        // a floor above the default cap lifts the default cap with it
        assert_eq!(VitBudget::parse(Some("2048"), None), Ok(VitBudget { min_tokens: 2048, max_tokens: 2048 }));
        assert_eq!(VitBudget::parse(Some(" 64 "), Some("1024")), Ok(OLD));
        for (min, max, word) in [
            (Some("1k"), None, "not a whole number"),
            (None, Some("-1"), "not a whole number"),
            (Some("4"), None, "outside 8..=4096"),
            (None, Some("8192"), "outside 8..=4096"),
            (Some("2048"), Some("1024"), "is above"),
        ] {
            let e = VitBudget::parse(min, max).unwrap_err();
            assert!(e.contains(word), "{min:?}/{max:?}: {e}");
        }
        assert_eq!(VitBudget::DEFAULT.min_pixels(), 1_048_576);
        assert_eq!(VitBudget::DEFAULT.max_pixels(), 1_310_720);
    }

    #[test]
    fn the_projector_file_is_found_by_the_documented_order() {
        let cnq = "/r/crow-nest/converter/x.cnq";
        // CROW_VIT_MMPROJ wins, 0 forces the container
        assert_eq!(
            resolve_mmproj(Some("0"), cnq, None, None, &|_| true),
            VitSource::Container("CROW_VIT_MMPROJ=0".into())
        );
        assert_eq!(
            resolve_mmproj(Some("/a/p.gguf"), cnq, None, None, &|p| p == "/a/p.gguf"),
            VitSource::Mmproj("/a/p.gguf".into())
        );
        assert!(matches!(resolve_mmproj(Some("/a/p.gguf"), cnq, None, None, &|_| false),
            VitSource::Container(w) if w.contains("does not exist")));
        // unset: models/ beside the container, then $CROW_MODELS, then Crow's install link
        let m = |p: &str| resolve_mmproj(None, cnq, Some("/cm"), Some("/home/u"), &|q| q == p);
        for hit in [
            "/r/crow-nest/models/mmproj-F16.gguf",
            "/cm/mmproj-F16.gguf",
            "/home/u/.local/share/crow/models/mmproj-F16.gguf",
        ] {
            assert_eq!(m(hit), VitSource::Mmproj(hit.into()));
        }
        // the first hit wins when several exist
        assert_eq!(
            resolve_mmproj(None, cnq, Some("/cm"), Some("/home/u"), &|_| true),
            VitSource::Mmproj("/r/crow-nest/models/mmproj-F16.gguf".into())
        );
        match resolve_mmproj(None, cnq, Some("/cm"), Some("/home/u"), &|_| false) {
            VitSource::Container(w) => assert!(
                w.contains("/r/crow-nest/models/mmproj-F16.gguf") && w.contains("/cm/") && w.contains("/home/u/"),
                "{w}"
            ),
            other => panic!("{other:?}"),
        }
    }

    /// a header-only projector image with the given tensor table (no data: the
    /// parser is told the file is big enough, validation reads no bytes)
    fn header(kv_over: Option<(&str, u32)>, drop: Option<&str>, redim: Option<(&str, Vec<u64>)>) -> Gguf {
        let u = |v: u32| v.to_le_bytes().to_vec();
        let mut kvs: Vec<(&str, u32, Vec<u8>)> = vec![
            ("clip.projector_type", 8, gguf::tests::str_payload("qwen3vl_merger")),
            ("clip.vision.block_count", 4, u(27)),
            ("clip.vision.embedding_length", 4, u(1152)),
            ("clip.vision.feed_forward_length", 4, u(4304)),
            ("clip.vision.attention.head_count", 4, u(16)),
            ("clip.vision.patch_size", 4, u(16)),
            ("clip.vision.spatial_merge_size", 4, u(2)),
            ("clip.vision.projection_dim", 4, u(2560)),
        ];
        if let Some((k, v)) = kv_over {
            for e in kvs.iter_mut() {
                if e.0 == k {
                    e.2 = u(v);
                }
            }
        }
        let plan = mmproj_plan();
        let tensors: Vec<(&str, Vec<u64>, u32, Vec<u8>)> = plan
            .iter()
            .filter(|(n, ..)| Some(n.as_str()) != drop)
            .map(|(n, d, k, _)| {
                let d = match &redim {
                    Some((rn, rd)) if rn == n => rd.clone(),
                    _ => d.clone(),
                };
                (n.as_str(), d, if *k == MmKind::Linear { GGML_TYPE_F16 } else { GGML_TYPE_F32 }, Vec::new())
            })
            .collect();
        let img = gguf::tests::build(&kvs, &tensors);
        Gguf::parse(&img[..], 1 << 40, "synthetic.gguf").unwrap()
    }

    #[test]
    fn the_projector_layout_is_checked_before_a_byte_is_loaded() {
        let plan = mmproj_plan();
        assert_eq!(plan.len(), 334, "the unsloth file carries 334 tensors");
        assert_eq!(plan.iter().filter(|e| e.2 == MmKind::Linear).count(), 27 * 4 + 2 + 2);
        let names: std::collections::BTreeSet<_> = plan.iter().map(|e| e.0.clone()).collect();
        assert_eq!(names.len(), 334);
        let targets: std::collections::BTreeSet<_> = plan.iter().map(|e| e.3.clone()).collect();
        assert_eq!(targets.len(), 333, "the container's vit section has 333 tensors");
        assert_eq!(validate_mmproj(&header(None, None, None)), Ok(()));
        let e = validate_mmproj(&header(Some(("clip.vision.block_count", 26)), None, None)).unwrap_err();
        assert!(e.contains("block_count"), "{e}");
        let e = validate_mmproj(&header(None, Some("v.blk.13.ffn_down.weight"), None)).unwrap_err();
        assert!(e.contains("v.blk.13.ffn_down.weight missing"), "{e}");
        // fc2 stored transposed (rows and k swapped) must not load
        let e = validate_mmproj(&header(None, None, Some(("v.blk.0.ffn_down.weight", vec![1152, 4304])))).unwrap_err();
        assert!(e.contains("dims"), "{e}");
    }

    #[test]
    fn the_patch_kernel_halves_interleave_into_the_conv3d_row_order() {
        let pp = VIT_PATCH * VIT_PATCH;
        let n = VIT_HIDDEN * 3 * pp;
        // value = a code of (half, o, c, ty*16+tx)
        let code = |t: usize, o: usize, c: usize, i: usize| ((t * 7919 + o * 263 + c * 1031 + i) % 65521) as u16;
        let mk = |t: usize| (0..n).map(|j| code(t, j / (3 * pp), (j / pp) % 3, j % pp)).collect::<Vec<u16>>();
        let out = interleave_patch_kernel(&mk(0), &mk(1));
        for o in [0, 1, 577, VIT_HIDDEN - 1] {
            for c in 0..3 {
                for t in 0..2 {
                    for i in [0, 1, 17, pp - 1] {
                        // our patch row: ((c * 2 + t) * 16 + ty) * 16 + tx
                        assert_eq!(out[o * VIT_IN + (c * 2 + t) * pp + i], code(t, o, c, i));
                    }
                }
            }
        }
    }

    /// the real file, when this machine has it: the header passes validation
    /// (reads ~20 KB, no tensor data)
    #[test]
    fn the_projector_on_this_machine_passes_the_header_check() {
        let VitSource::Mmproj(p) = mmproj_source("../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq") else {
            eprintln!("no mmproj-F16.gguf on this machine - skipped");
            return;
        };
        let g = Gguf::open(&p).unwrap();
        assert_eq!(g.tensors.len(), 334);
        validate_mmproj(&g).unwrap();
    }

    /// cos(container NVFP4 dequant, mmproj F16) per tensor: a mapping or
    /// orientation error (a wrong name, a transposed fc2, swapped patch halves)
    /// drops far below NVFP4's own error. CPU only, reads ~1.1 GB:
    /// `CROW_CNQ=<container> cargo test --release mmproj_matches -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn mmproj_matches_the_container_tensor_by_tensor() {
        let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| "../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq".into());
        let VitSource::Mmproj(p) = mmproj_source(&cnq_path) else { panic!("no mmproj file") };
        let g = Gguf::open(&p).unwrap();
        let mut cnq = Cnq::open(&cnq_path);
        let cos = |a: &[f32], b: &[f32]| {
            assert_eq!(a.len(), b.len());
            let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
            for (x, y) in a.iter().zip(b) {
                ab += (*x as f64) * (*y as f64);
                aa += (*x as f64).powi(2);
                bb += (*y as f64).powi(2);
            }
            ab / (aa.sqrt() * bb.sqrt())
        };
        let raw = |name: &str| g.read_raw(g.find(name).unwrap()).unwrap();
        let mut worst = (1.0f64, String::new());
        for (gname, dims, kind, cname) in mmproj_plan() {
            if gname == "v.patch_embd.weight.1" {
                continue;
            }
            let n: usize = if gname == "v.patch_embd.weight" { VIT_HIDDEN * VIT_IN } else { dims.iter().product::<u64>() as usize };
            let t = cnq.find(&cname, "vit").clone();
            let ours: Vec<f32> = if t.dtype == "bf16" {
                cnq.read_f32(&cname, "vit")
            } else {
                let bytes = cnq.read_bytes(&t);
                let mut out = vec![0f32; n];
                let mut blk = [0f32; 64];
                for (b, chunk) in bytes.chunks_exact(36).enumerate() {
                    crate::cnq::dequant_block(chunk, t.global_scale, &mut blk);
                    let end = ((b + 1) * 64).min(n);
                    out[b * 64..end].copy_from_slice(&blk[..end - b * 64]);
                }
                out
            };
            let theirs: Vec<f32> = if gname == "v.patch_embd.weight" {
                let pk = interleave_patch_kernel(&u16s(&raw("v.patch_embd.weight")), &u16s(&raw("v.patch_embd.weight.1")));
                pk.iter().map(|h| gguf::f16_to_f32(*h)).collect()
            } else if kind == MmKind::Linear {
                u16s(&raw(&gname)).iter().map(|h| gguf::f16_to_f32(*h)).collect()
            } else {
                f32s(&raw(&gname))
            };
            let c = cos(&ours, &theirs);
            if gname == "v.patch_embd.weight" {
                // the control: the two temporal halves swapped must read worse
                let sw = interleave_patch_kernel(&u16s(&raw("v.patch_embd.weight.1")), &u16s(&raw("v.patch_embd.weight")));
                let sw: Vec<f32> = sw.iter().map(|h| gguf::f16_to_f32(*h)).collect();
                let cs = cos(&ours, &sw);
                eprintln!("patch kernel cos {c:.6}, halves swapped {cs:.6}");
                assert!(cs < c, "the patch test cannot tell the halves apart");
            }
            if c < worst.0 {
                worst = (c, gname.clone());
            }
            assert!(c > 0.995, "{gname} vs {cname}: cos {c:.6}");
        }
        eprintln!("worst cos {:.6} at {}", worst.0, worst.1);
    }
}

#[cfg(test)]
mod gemm_vit {
    //! #109 and #108 on the GPU: the two vision GEMMs
    //! (`gemm_fp4_f32x`, `gemm_f16_f32x`) against an f64 host reference at the
    //! tower's real shapes, INCLUDING the fc1 shape whose 4304 rows are not a
    //! multiple of the 64-row tile. The weight buffer carries a nonzero garbage
    //! tail, so a row tile that reads past the last row decodes nonzero values -
    //! and before the row guard it wrote them into the next token's first 48
    //! outputs. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release gemm_vit -- --ignored --nocapture` (a few MB of VRAM).
    use super::*;

    fn fill_u8(n: usize, seed: u32) -> Vec<u8> {
        let mut x = seed.max(1);
        (0..n)
            .map(|_| {
                x ^= x << 13;
                x ^= x >> 17;
                x ^= x << 5;
                (x >> 11) as u8
            })
            .collect()
    }

    fn fill_f32(n: usize, seed: u32) -> Vec<f32> {
        fill_u8(n * 3, seed).chunks_exact(3).map(|b| (b[0] as f32 - 127.5) / 64.0 + b[1] as f32 / 4096.0).collect()
    }

    /// max |gpu - ref| / (|ref| + 1e-3 * max|ref|) over every output
    fn worst(gpu: &[f32], reference: &[f64]) -> (f64, usize) {
        let scale = reference.iter().fold(0f64, |a, v| a.max(v.abs()));
        let mut w = (0f64, 0usize);
        for (i, (g, r)) in gpu.iter().zip(reference).enumerate() {
            let e = (*g as f64 - r).abs() / (r.abs() + 1e-3 * scale);
            if e > w.0 {
                w = (e, i);
            }
        }
        w
    }

    fn reference(wf: &[f32], x: &[f32], rows: usize, k: usize, t: usize) -> Vec<f64> {
        let mut y = vec![0f64; t * rows];
        for tok in 0..t {
            for r in 0..rows {
                y[tok * rows + r] = (0..k).map(|j| wf[r * k + j] as f64 * x[tok * k + j] as f64).sum();
            }
        }
        y
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release gemm_vit -- --ignored --nocapture"]
    fn both_vision_gemms_match_the_host_at_every_tower_shape_and_keep_the_row_tail_in_bounds() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let module = cuda::compile(crate::kernels::KERNEL_SRC);
            let (f_fp4, f_f16) = (module.get("gemm_fp4_f32x"), module.get("gemm_f16_f32x"));
            // (rows, k): qkv, proj, fc1 (row tail), fc2 (k tail), merger fc1, fc2, patch
            let shapes = [(3456, 1152), (1152, 1152), (4304, 1152), (1152, 4304), (4608, 4608), (2560, 4608), (1152, 1536)];
            let t = 70; // three token tiles, the last one partial
            for (si, &(rows, k)) in shapes.iter().enumerate() {
                let x = fill_f32(t * k, 0x51 + si as u32);
                let xd = cuda::to_f32_dev(&x);
                let (kd, rd, td) = (cuda::to_i32_dev(&[k as i32]), cuda::to_i32_dev(&[rows as i32]), cuda::to_i32_dev(&[t as i32]));
                // --- NVFP4: 36-byte blocks, scale bytes in a sane ue4m3 range, 1 MiB garbage tail
                let nblk = rows * k / 64;
                let mut wb = fill_u8(nblk * 36 + (1 << 20), 0x77 + si as u32);
                for b in 0..nblk {
                    for s in 0..4 {
                        wb[b * 36 + s] = 0x30 + (wb[b * 36 + s] & 0x0f);
                    }
                }
                let gs = 0.37f32;
                let mut wf = vec![0f32; rows * k];
                let mut blk = [0f32; 64];
                for b in 0..nblk {
                    crate::cnq::dequant_block(&wb[b * 36..b * 36 + 36], gs, &mut blk);
                    wf[b * 64..b * 64 + 64].copy_from_slice(&blk);
                }
                let want = reference(&wf, &x, rows, k, t);
                let wd = cuda::upload_dev(&wb);
                let gsd = cuda::to_f32_dev(&[gs]);
                let yd = cuda::alloc_zeroed((t + 1) * rows * 4);
                for rep in 0..5 {
                    launch_v(f_fp4, rows.div_ceil(64) as u32, t.div_ceil(32) as u32, 1, 256, &[wd, xd, gsd, yd, kd, rd, td]);
                    cuda::sync();
                    let got = cuda::dtoh(yd, (t + 1) * rows);
                    let (e, at) = worst(&got[..t * rows], &want);
                    assert!(e < 1e-3, "fp4 {rows}x{k} rep {rep}: rel err {e:.3e} at token {} row {}", at / rows, at % rows);
                    assert!(got[t * rows..].iter().all(|v| *v == 0.0), "fp4 {rows}x{k}: wrote past the last token");
                }
                // --- F16: the same values rounded to f16 bits, garbage tail again
                let wh: Vec<u16> = fill_u8(rows * k * 2, 0x99 + si as u32)
                    .chunks_exact(2)
                    .map(|b| {
                        // sign, exponent 10..17 (|w| ~ 1e-3..4), random mantissa
                        (((b[0] as u16) & 0x80) << 8) | ((10 + (b[0] as u16 & 7)) << 10) | (((b[1] as u16) << 2) & 0x3ff)
                    })
                    .collect();
                let wf16: Vec<f32> = wh.iter().map(|h| crate::gguf::f16_to_f32(*h)).collect();
                let want = reference(&wf16, &x, rows, k, t);
                let mut bytes: Vec<u8> = wh.iter().flat_map(|h| h.to_le_bytes()).collect();
                bytes.extend(fill_u8(1 << 20, 3));
                let wd = cuda::upload_dev(&bytes);
                let yd = cuda::alloc_zeroed((t + 1) * rows * 4);
                for rep in 0..5 {
                    launch_v(f_f16, rows.div_ceil(64) as u32, t.div_ceil(32) as u32, 1, 256, &[wd, xd, yd, kd, rd, td]);
                    cuda::sync();
                    let got = cuda::dtoh(yd, (t + 1) * rows);
                    let (e, at) = worst(&got[..t * rows], &want);
                    assert!(e < 1e-3, "f16 {rows}x{k} rep {rep}: rel err {e:.3e} at token {} row {}", at / rows, at % rows);
                    assert!(got[t * rows..].iter().all(|v| *v == 0.0), "f16 {rows}x{k}: wrote past the last token");
                }
                eprintln!("gemm_vit {rows}x{k} t {t}: fp4 and f16 within 1e-3 of the f64 host, 5 reps each");
            }
        }
    }
}

#[cfg(test)]
mod image_identity {
    //! #114: what the prefix cache compares next to the ids of an image
    use super::*;

    const P: u32 = IMAGE_PAD as u32;

    /// the spans land exactly where `expand_ids` puts the visual tokens
    #[test]
    fn the_spans_are_the_rows_expand_ids_fills() {
        let ids = [1u32, 2, P, 3, P, 4];
        let counts = [4usize, 2];
        let spans = image_spans(&ids, &counts, &[11, 22]).unwrap();
        assert_eq!(
            spans,
            vec![ImageSpan { start: 2, len: 4, hash: 11 }, ImageSpan { start: 7, len: 2, hash: 22 }]
        );
        let expanded = expand_ids(&ids, &counts).unwrap();
        for sp in &spans {
            assert!(expanded[sp.start..sp.start + sp.len].iter().all(|&v| v == P));
            assert_ne!(expanded[sp.start - 1], P);
        }
        assert!(image_spans(&ids, &[4], &[11]).is_err());
        assert!(image_spans(&ids, &[4, 2, 1], &[1, 2, 3]).is_err());
        assert!(image_spans(&ids, &[4, 2], &[1]).is_err());
    }

    /// two images that differ in one byte get different keys; the same bytes the same key
    #[test]
    fn the_image_key_is_the_content_of_the_bytes() {
        let red = vec![0x89u8, b'P', b'N', b'G', 255, 0, 0];
        let mut green = red.clone();
        green[4] = 0;
        green[5] = 255;
        assert_eq!(image_key(&red), image_key(&red.clone()));
        assert_ne!(image_key(&red), image_key(&green));
    }
}
