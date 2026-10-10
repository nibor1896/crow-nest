//! crow-nest #163 (GLM-5.3-Flash plan step 14, MLA/DSA part): the glm5_next MLA attention over a
//! BF16 latent cache and its DSA indexer with learned 4-token pooling, on the GPU.
//!
//! What one DSA layer computes is `docs/glm5-next-recipe.md` sections 7 and 8 (HF
//! `modeling_glm5_next.py:736-1256`); how this module computes it, its state and its evidence are
//! `docs/glm5-mla.md`. In short, per call of `t` rows at positions `pos0 .. pos0 + t`:
//!
//! - the latent `c = rms(kv_a x)` goes into the latent cache, BF16, `kv_lora` values per token
//!   (1,024 B, `Glm5Geo::latent_bytes_per_token`); K and V are never expanded: `kv_b` is absorbed into
//!   the query (`q~_h = W_k,h^T q_h`) and applied after attention (`o_h = W_v,h u_h`);
//! - the indexer row `[LayerNorm(wk x) | gate x | 1]` goes into the indexer cache in the HF layout,
//!   BF16, 257 values (514 B, `Glm5Geo::indexer_bytes_per_token`); pooled keys are recomputed from it
//!   (HF does the same, `:899-972`);
//! - the pool scores feed `qsa_select_fast` of `KERNEL_SRC` (pools of 4 from position 0, the tail
//!   appended, ties to the lowest pool index), at most `sel_max` = 2051 rows per query;
//! - split-K attention over the selected latent rows, then `W_v`, then `o_proj`.
//!
//! The projections run on `gm_gemm` (BF16 weights, f32 activations and accumulation; `gm_gemv`,
//! bit-identical, for calls of at most [`DECODE_T`] rows) through
//! [`MlaScratch::linear`]; an integrator with other weight codecs replaces those calls and keeps the
//! stages ([`MlaScratch::store_latent`], [`MlaScratch::store_index`], [`MlaScratch::select`],
//! [`MlaScratch::attend`]). Nothing in the engine calls this module yet (the lead wires it into
//! `gen.rs`); it depends on `cuda`, `geo` and `kernels` only.

use crate::cuda;
use crate::geo::Glm5Geo;
use crate::kernels::launch_v;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

/// every entry of `kernels::GLM5_MLA_SRC`
pub const NAMES: &[&str] = &[
    "gm_gemm",
    "gm_rmsnorm",
    "gm_latent_store",
    "gm_idx_store",
    "gm_idx_scores",
    "gm_sel_prep",
    "gm_absorb",
    "gm_attn",
    "gm_attn_merge",
    "gm_out_v",
    "gm_gemv",
    "gm_absorb1",
    "gm_out_v1",
    "gm_attn2",
];

/// the most rows a call may have to run on the decode kernels `gm_gemv`, `gm_absorb1`,
/// `gm_out_v1` (`GM_DT`); larger calls run on the tiled `gm_gemm`, `gm_absorb`, `gm_out_v`. Each
/// decode kernel keeps the summation chain of its tiled twin: bit-identical outputs.
pub const DECODE_T: usize = 4;
/// output rows of one `gm_gemv` block (`GM_GV_R`)
pub const GEMV_ROWS: usize = 8;

/// the pool size the selection kernel (`qsa_select_fast`) is built for
pub const KPOOL: usize = 4;
/// the most pools `qsa_select_fast` can rank (its shared bitmap holds 65,536 bits): 262,144 tokens
pub const MAX_POOLS: usize = 65_536;
/// RMSNorm eps (`rms_norm_eps`) and the indexer LayerNorm eps (`modeling_glm5_next.py:763`)
pub const RMS_EPS: f32 = 1e-5;
pub const LN_EPS: f32 = 1e-6;

/// The shapes of one MLA + DSA layer, compiled into the kernels as `GM_*` defines.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MlaDims {
    pub hidden: usize,
    pub heads: usize,
    /// per-head key width (`qk_nope_head_dim`; no RoPE part) and value width
    pub nope: usize,
    pub v: usize,
    /// the latent width (`kv_lora_rank`) and the query low rank (`q_lora_rank`)
    pub kv_lora: usize,
    pub q_lora: usize,
    /// the indexer: heads, head dim, pool size, selection budget in tokens
    pub idx_heads: usize,
    pub idx_dim: usize,
    pub kpool: usize,
    pub topk: usize,
}

impl MlaDims {
    /// the DSA layer of a glm5_next geometry
    pub const fn of(g: &Glm5Geo) -> MlaDims {
        MlaDims {
            hidden: g.hidden,
            heads: g.mla_heads,
            nope: g.nope_dim,
            v: g.v_dim,
            kv_lora: g.kv_lora,
            q_lora: g.q_lora,
            idx_heads: g.index_heads,
            idx_dim: g.index_head_dim,
            kpool: g.index_kpool,
            topk: g.index_topk,
        }
    }
    /// pools selected per query (`index_topk / kpool`, 512)
    pub const fn sel_pools(&self) -> usize {
        self.topk / self.kpool
    }
    /// the most rows one query attends (`index_topk + kpool - 1`, 2051)
    pub const fn sel_max(&self) -> usize {
        self.topk + self.kpool - 1
    }
    /// values of one indexer cache row, HF layout `[key | gate | valid]` (257)
    pub const fn idx_row(&self) -> usize {
        2 * self.idx_dim + 1
    }
    /// rows of the concatenated x-side indexer projection `[wk | compress_gate | weights_proj]` (288)
    pub const fn idx_proj(&self) -> usize {
        2 * self.idx_dim + self.idx_heads
    }
    /// latent cache bytes per token (BF16, 1,024)
    pub const fn latent_bytes_per_token(&self) -> u64 {
        (self.kv_lora * 2) as u64
    }
    /// indexer cache bytes per token (BF16, 514)
    pub const fn indexer_bytes_per_token(&self) -> u64 {
        (self.idx_row() * 2) as u64
    }
    /// the kernels' preconditions, by name
    pub fn check(&self) -> Result<(), String> {
        let mut bad = Vec::new();
        if self.kpool != KPOOL {
            bad.push(format!("kpool {} (qsa_select_fast pools {KPOOL} tokens)", self.kpool));
        }
        if self.topk % self.kpool != 0 || self.topk == 0 {
            bad.push(format!("index_topk {} is not a positive multiple of kpool", self.topk));
        }
        if self.kv_lora % 32 != 0 || self.kv_lora == 0 {
            bad.push(format!("kv_lora {} (gm_attn / gm_out_v split it over 32 lanes)", self.kv_lora));
        }
        for (n, v) in [("hidden", self.hidden), ("q_lora", self.q_lora), ("heads*v", self.heads * self.v), ("kv_lora", self.kv_lora)] {
            if v % 32 != 0 {
                bad.push(format!("{n} {v} (gm_gemm reads k in steps of 32)"));
            }
        }
        if self.idx_heads * self.idx_dim * 4 + 32 * (self.idx_dim + 1) * 4 > 48 * 1024 {
            bad.push(format!("indexer {} x {} does not fit gm_idx_scores' 48 KiB of shared memory", self.idx_heads, self.idx_dim));
        }
        if self.nope * 8 * 4 > 48 * 1024 || self.kv_lora * 8 * 4 > 48 * 1024 {
            bad.push("nope / kv_lora too wide for the 8-token shared tiles".into());
        }
        if bad.is_empty() { Ok(()) } else { Err(bad.join("; ")) }
    }
    /// the `#define` block in front of `kernels::GLM5_MLA_SRC`
    pub fn prelude(&self) -> String {
        let f = |v: f64| format!("{:e}f", v as f32);
        let defs: [(&str, String); 13] = [
            ("GM_HEADS", self.heads.to_string()),
            ("GM_NOPE", self.nope.to_string()),
            ("GM_V", self.v.to_string()),
            ("GM_LAT", self.kv_lora.to_string()),
            ("GM_IH", self.idx_heads.to_string()),
            ("GM_ID", self.idx_dim.to_string()),
            ("GM_ROW", self.idx_row().to_string()),
            ("GM_SEL_MAX", self.sel_max().to_string()),
            ("GM_EPS", f(RMS_EPS as f64)),
            ("GM_LN_EPS", f(LN_EPS as f64)),
            // HF: scaling = qk_head_dim^-0.5 (:1128), indexer softmax_scale = head_dim^-0.5 and
            // weights * n_heads^-0.5 (:825-830), each a Python float applied to an f32 tensor
            ("GM_SCALE", f((self.nope as f64).powf(-0.5))),
            ("GM_ISCALE", f((self.idx_dim as f64).powf(-0.5))),
            ("GM_WSCALE", f((self.idx_heads as f64).powf(-0.5))),
        ];
        let mut s = String::from("// crow-nest #163: glm5_mla::MlaDims::prelude()\n");
        for (n, v) in defs {
            s.push_str(&format!("#define {n} {v}\n"));
        }
        s
    }
    /// the text NVRTC compiles
    pub fn source(&self) -> String {
        format!("{}{}", self.prelude(), crate::kernels::GLM5_MLA_SRC)
    }
}

/// The compiled module, its entries and the selection kernel of the engine's main module.
pub struct MlaKernels {
    pub module: cuda::Module,
    pub dims: MlaDims,
    gemm: CUfunction,
    rmsnorm: CUfunction,
    latent_store: CUfunction,
    idx_store: CUfunction,
    idx_scores: CUfunction,
    sel_prep: CUfunction,
    absorb: CUfunction,
    attn: CUfunction,
    merge: CUfunction,
    /// #186: `gm_attn2`, the tensor-core prompt attention (`CROW_GLM_ATTN2=1`)
    attn2: CUfunction,
    out_v: CUfunction,
    gemv: CUfunction,
    absorb1: CUfunction,
    out_v1: CUfunction,
    select: CUfunction,
}

impl MlaKernels {
    /// `select` is `qsa_select_fast` of a module compiled from `KERNEL_SRC` (any `KernelGeo`: the
    /// entry reads no `CN_*` define).
    ///
    /// # Safety
    /// A CUDA context is current and `select` belongs to a module loaded in it.
    pub unsafe fn new(dims: MlaDims, select: CUfunction) -> MlaKernels {
        if let Err(e) = dims.check() {
            panic!("glm5_mla: {e}");
        }
        let module = crate::kernels::glm5_mla_module(&dims.prelude());
        MlaKernels {
            gemm: module.get("gm_gemm"),
            rmsnorm: module.get("gm_rmsnorm"),
            latent_store: module.get("gm_latent_store"),
            idx_store: module.get("gm_idx_store"),
            idx_scores: module.get("gm_idx_scores"),
            sel_prep: module.get("gm_sel_prep"),
            absorb: module.get("gm_absorb"),
            attn: module.get("gm_attn"),
            merge: module.get("gm_attn_merge"),
            attn2: module.get("gm_attn2"),
            out_v: module.get("gm_out_v"),
            gemv: module.get("gm_gemv"),
            absorb1: module.get("gm_absorb1"),
            out_v1: module.get("gm_out_v1"),
            select,
            module,
            dims,
        }
    }

    /// #161: `x[r][0..n] = w * (x[r] * rsqrt(mean(x[r]^2) + 1e-5))` in place for `rows` rows
    /// (`gm_rmsnorm`, HF `Glm5NextTextRMSNorm`: weight times the normed value, not `1 + w`); the
    /// layer driver's `input_layernorm` and `post_attention_layernorm`. `st` is any device
    /// buffer of two i32 (the kernel takes the call slot but reads no field of it).
    ///
    /// # Safety
    /// `x` holds `rows x n` f32, `w` `n` f32.
    pub unsafe fn rmsnorm_rows(&self, x: CUdeviceptr, w: CUdeviceptr, n: usize, rows: usize, st: CUdeviceptr) {
        launch_v(self.rmsnorm, rows as u32, 1, 1, 256, &[x, w, n as u64, st]);
    }
}

/// #161: the four MLA projections a caller may run in another weight codec (the container
/// stores them NVFP4); the indexer projections stay BF16 on `gm_gemm`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MlaProj {
    /// `x [t][hidden]` -> `qa [t][q_lora]` (normed after)
    QA,
    /// `qa [t][q_lora]` -> `q [t][heads * nope]`
    QB,
    /// `x [t][hidden]` -> `kva [t][kv_lora]`
    KVA,
    /// `o [t][heads * v]` -> `y [t][hidden]`
    O,
}

/// The per-sequence state of one DSA layer: the latent cache `[cap][kv_lora]` BF16 and the indexer
/// cache `[cap][2 idx_dim + 1]` BF16, both indexed by absolute position.
pub struct MlaCache {
    pub latent: CUdeviceptr,
    pub index: CUdeviceptr,
    pub cap: usize,
}

impl MlaCache {
    /// device bytes of a cache of `cap` tokens
    pub const fn bytes(d: &MlaDims, cap: usize) -> u64 {
        (d.latent_bytes_per_token() + d.indexer_bytes_per_token()) * cap as u64
    }
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(d: &MlaDims, cap: usize) -> MlaCache {
        assert!(cap / KPOOL <= MAX_POOLS, "glm5_mla: a cache of {cap} tokens has more than {MAX_POOLS} pools");
        MlaCache {
            latent: cuda::alloc_named("glm5 MLA latent cache", cap * d.latent_bytes_per_token() as usize),
            index: cuda::alloc_named("glm5 DSA indexer cache", cap * d.indexer_bytes_per_token() as usize),
            cap,
        }
    }
    /// # Safety
    /// No launch that reads the cache is pending.
    pub unsafe fn free(&mut self) {
        cuda::free_dev(&mut self.latent);
        cuda::free_dev(&mut self.index);
    }
}

/// Device pointers of one layer's weights. Matrices BF16 row-major `[out][in]` (the checkpoint
/// layout), vectors f32.
#[derive(Clone, Copy, Debug)]
pub struct MlaWeights {
    /// `[q_lora][hidden]`, norm `[q_lora]`
    pub q_a: CUdeviceptr,
    pub q_a_norm: CUdeviceptr,
    /// `[heads * nope][q_lora]`
    pub q_b: CUdeviceptr,
    /// `[kv_lora][hidden]`, norm `[kv_lora]`
    pub kv_a: CUdeviceptr,
    pub kv_a_norm: CUdeviceptr,
    /// `[heads * (nope + v)][kv_lora]`, read by `gm_absorb` and `gm_out_v` (always BF16)
    pub kv_b: CUdeviceptr,
    /// `[hidden][heads * v]`
    pub o_proj: CUdeviceptr,
    /// `[idx_heads * idx_dim][q_lora]`
    pub idx_wq_b: CUdeviceptr,
    /// `[wk; index_kpool_compress_gate; weights_proj]` stacked: `[2 idx_dim + idx_heads][hidden]`
    pub idx_x: CUdeviceptr,
    /// LayerNorm weight and bias `[idx_dim]`, `index_kpool_compress_ape` `[kpool][idx_dim]`
    pub idx_k_norm_w: CUdeviceptr,
    pub idx_k_norm_b: CUdeviceptr,
    pub idx_ape: CUdeviceptr,
}

/// split count of the decode-shaped attention: enough blocks for a few rows, one split from 16 rows
/// #186 `CROW_GLM_ATTN2=1`: the one-split (prompt) attention runs `gm_attn2` on tensor cores
/// (BF16 q~, P and latent rows, f32 accumulation and softmax); default off
pub fn attn2_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_GLM_ATTN2").ok().as_deref() == Some("1"))
}

/// the shapes `gm_attn2` tiles (64 heads per block, each latent half a multiple of 16)
pub const fn attn2_fits(d: &MlaDims) -> bool {
    d.heads % 64 == 0 && d.kv_lora % 32 == 0
}

pub fn attn_splits(t: usize) -> usize {
    (16 / t.max(1)).max(1)
}

/// the `idx_scores` grid of [`MlaScratch::select`] for rows `pos0 .. pos0 + t` (0 = not launched):
/// the one launch shape of a decode row that depends on the position (#190: `CROW_GLM_GRAPH`
/// keys its captured row on it)
pub fn score_grid(pos0: usize, t: usize) -> usize {
    ((pos0 + t) / KPOOL).div_ceil(32)
}

/// The scratch of one call shape (at most `max_t` rows against caches of `cap` tokens) and the
/// stage launches. Every stage queues on the current stream and reads the call set by [`begin`].
///
/// [`begin`]: MlaScratch::begin
pub struct MlaScratch {
    pub max_t: usize,
    pub cap: usize,
    t: usize,
    pos0: usize,
    /// the call `(pos0, t)` on the device; `prm` = `[sel_pools, cap / 4, sel_max]`
    st: CUdeviceptr,
    prm: CUdeviceptr,
    /// `[t][q_lora]`: q_a output, normed in place into q_resid
    pub qa: CUdeviceptr,
    /// `[t][heads][nope]`
    pub q: CUdeviceptr,
    /// `[t][kv_lora]`: kv_a output (the latent before its norm)
    pub kva: CUdeviceptr,
    /// `[t][2 idx_dim + idx_heads]`, `[t][idx_heads][idx_dim]`
    pub ip: CUdeviceptr,
    pub iq: CUdeviceptr,
    /// `[t][cap / 4]` pool scores
    pub scores: CUdeviceptr,
    pub ncb: CUdeviceptr,
    pub pos: CUdeviceptr,
    /// `[t][sel_max]` selected token ids, `[t]` their counts
    pub sel: CUdeviceptr,
    pub sel_n: CUdeviceptr,
    /// `[t][heads][kv_lora]` absorbed queries, attention partials, latent mix
    pub qt: CUdeviceptr,
    pub part_o: CUdeviceptr,
    pub part_ml: CUdeviceptr,
    pub u: CUdeviceptr,
    /// `[t][heads * v]`, the input of o_proj
    pub o: CUdeviceptr,
    /// the pinned sources of `st`'s uploads: [`ST_RING`] entries of `[pos0, t]` i32, each with
    /// the event of its copy (an entry is rewritten only after its copy ran)
    st_ring: cuda::Pinned,
    st_ev: Vec<cudarc::driver::sys::CUevent>,
    st_used: Vec<bool>,
    st_cur: usize,
    /// #196: the region of every activation but `sel` / `sel_n` when this scratch owns it (0: a
    /// view into a region its owner frees)
    region: CUdeviceptr,
    /// #196: rows of `sel` / `sel_n` (at least `max_t`); a call of [`MlaScratch::forward_rows_with`]
    /// writes its selection at row `row_off` of them
    pub sel_rows: usize,
    row_off: usize,
    /// #196: the rows whose split count ([`attn_splits`]) the call's attention takes (its own
    /// rows, or the whole prompt call's when it is a sub-call of one)
    split_t: usize,
    /// `sel_max` of the dims (the row stride of `sel`)
    sel_max: usize,
}

/// entries of the pinned `[pos0, t]` ring of [`MlaScratch::begin`] (a decode row with `CROW_GLM_LA`
/// has two rows in flight; 64 is never waited on in practice)
pub const ST_RING: usize = 64;

impl MlaScratch {
    /// f32 values of the activations in the region at `max_t` rows over caches of `cap` tokens,
    /// in field order: qa, q, kva, ip, iq, scores, ncb, pos, qt, part_o, part_ml, u, o
    fn widths(d: &MlaDims, max_t: usize, cap: usize) -> [usize; 13] {
        let slots = max_t.max(16); // t * attn_splits(t) <= max(t, 16)
        [
            max_t * d.q_lora,
            max_t * d.heads * d.nope,
            max_t * d.kv_lora,
            max_t * d.idx_proj(),
            max_t * d.idx_heads * d.idx_dim,
            max_t * (cap / KPOOL).max(1),
            max_t,
            max_t,
            max_t * d.heads * d.kv_lora,
            slots * d.heads * d.kv_lora,
            slots * d.heads * 2,
            max_t * d.heads * d.kv_lora,
            max_t * d.heads * d.v,
        ]
    }

    /// #196: the bytes of the region of a scratch of `max_t` rows over caches of `cap` tokens
    /// (every activation but the selection, `glm5_moe::carve`)
    pub fn region_bytes(d: &MlaDims, max_t: usize, cap: usize) -> usize {
        crate::glm5_moe::carve(&MlaScratch::widths(d, max_t, cap).map(|n| n * 4)).1
    }

    /// #196: the bytes of `sel` and `sel_n` at `rows` rows (allocated apart from the region)
    pub fn sel_bytes(d: &MlaDims, rows: usize) -> usize {
        4 * rows * (d.sel_max() + 1)
    }

    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(d: &MlaDims, max_t: usize, cap: usize) -> MlaScratch {
        assert!(max_t > 0 && cap / KPOOL <= MAX_POOLS);
        let region = cuda::alloc_named("glm5 mla scratch", MlaScratch::region_bytes(d, max_t, cap));
        let mut s = MlaScratch::new_in(d, max_t, max_t, cap, region);
        s.region = region;
        s
    }

    /// #196: a scratch of calls of up to `max_t` rows whose activations are views into `region`
    /// ([`MlaScratch::region_bytes`] long), which the caller allocates and frees, and whose
    /// selection holds `sel_rows` rows (its own allocation; [`MlaScratch::forward_rows_with`]
    /// writes sub-calls of a larger call at their rows)
    ///
    /// # Safety
    /// A CUDA context is current; `region` outlives every launch on this scratch.
    pub unsafe fn new_in(d: &MlaDims, max_t: usize, sel_rows: usize, cap: usize, region: CUdeviceptr) -> MlaScratch {
        assert!(max_t > 0 && sel_rows >= max_t && cap / KPOOL <= MAX_POOLS && region != 0);
        let (off, _) = crate::glm5_moe::carve(&MlaScratch::widths(d, max_t, cap).map(|n| n * 4));
        let at = |i: usize| region + off[i] as u64;
        let a = |what: &str, n: usize| cuda::alloc_named(what, n * 4);
        let prm = cuda::to_i32_dev(&[d.sel_pools() as i32, (cap / KPOOL) as i32, d.sel_max() as i32]);
        MlaScratch {
            max_t,
            cap,
            t: 0,
            pos0: 0,
            st: cuda::to_i32_dev(&[0i32, 0]),
            prm,
            qa: at(0),
            q: at(1),
            kva: at(2),
            ip: at(3),
            iq: at(4),
            scores: at(5),
            ncb: at(6),
            pos: at(7),
            sel: a("glm5 idx selection", sel_rows * d.sel_max()),
            sel_n: a("glm5 idx selection n", sel_rows),
            qt: at(8),
            part_o: at(9),
            part_ml: at(10),
            u: at(11),
            o: at(12),
            st_ring: cuda::Pinned::alloc(ST_RING * 8),
            st_ev: (0..ST_RING).map(|_| cuda::event_create()).collect(),
            st_used: vec![false; ST_RING],
            st_cur: 0,
            region: 0,
            sel_rows,
            row_off: 0,
            split_t: 0,
            sel_max: d.sel_max(),
        }
    }

    /// the call's `sel` / `sel_n` rows (#196: at `row_off`)
    fn sel_at(&self) -> (CUdeviceptr, CUdeviceptr) {
        ((self.sel + (self.row_off * self.sel_max * 4) as u64), self.sel_n + (self.row_off * 4) as u64)
    }

    /// rows of the current call
    pub fn t(&self) -> usize {
        self.t
    }

    /// Set the call: `t` rows at absolute positions `pos0 .. pos0 + t`.
    ///
    /// # Safety
    /// No launch of the previous call that reads `st` is pending on another stream.
    pub unsafe fn begin(&mut self, pos0: usize, t: usize) {
        assert!((1..=self.max_t).contains(&t), "glm5_mla: {t} rows per call (1..={})", self.max_t);
        assert!(pos0 + t <= self.cap, "glm5_mla: rows {pos0}..{} beyond the cache of {}", pos0 + t, self.cap);
        self.t = t;
        self.pos0 = pos0;
        (self.row_off, self.split_t) = (0, t);
        // #190: inside a CROW_GLM_GRAPH capture the row staged `st` already (a host upload would
        // be captured from this stack array and replayed stale)
        if !crate::glm5_graph::capturing() {
            // from a pinned ring entry, so the copy is a true async DMA: a pageable (stack)
            // source made the legacy stream synchronize here, and a row enqueued ahead
            // (`CROW_GLM_LA`) blocked the host at its first DSA layer. Same values, same stream
            // order, so every kernel reads what it read before.
            use cudarc::driver::sys;
            let i = self.st_cur;
            self.st_cur = (i + 1) % ST_RING;
            if self.st_used[i] {
                cuda::ck(sys::cuEventSynchronize(self.st_ev[i]));
            }
            let e = (self.st_ring.host as *mut i32).add(2 * i);
            e.write(pos0 as i32);
            e.add(1).write(t as i32);
            let s = cuda::cur_stream();
            cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.st, e as *const _, 8, s));
            cuda::event_record(self.st_ev[i], s);
            self.st_used[i] = true;
        }
    }

    /// #190: the device `[pos0, t]` the call's kernels read (`CROW_GLM_GRAPH` stages it per row)
    pub fn st_dev(&self) -> CUdeviceptr {
        self.st
    }

    /// `y[tok][0..n]` (row stride `ldy`) `= W x[tok]` (row stride `ldx`), W BF16 `[n][k]`
    ///
    /// # Safety
    /// The buffers hold the current call's rows.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn linear(&self, kn: &MlaKernels, w: CUdeviceptr, k: usize, n: usize, x: CUdeviceptr, ldx: usize, y: CUdeviceptr, ldy: usize) {
        assert!(k % 32 == 0, "gm_gemm: k = {k}");
        if self.t <= DECODE_T && w.is_multiple_of(16) && x.is_multiple_of(16) && ldx.is_multiple_of(4) {
            launch_v(kn.gemv, n.div_ceil(GEMV_ROWS) as u32, 1, 1, 128, &[w, x, y, k as u64, n as u64, ldx as u64, ldy as u64, self.st]);
            return;
        }
        launch_v(kn.gemm, n.div_ceil(64) as u32, self.t.div_ceil(16) as u32, 1, 256, &[w, x, y, k as u64, n as u64, ldx as u64, ldy as u64, self.st]);
    }

    /// the query side up to q_resid and q: `qa = rms(q_a x)`, `q = q_b qa` (x `[t][hidden]`)
    ///
    /// # Safety
    /// As [`MlaScratch::linear`].
    pub unsafe fn query(&self, kn: &MlaKernels, w: &MlaWeights, x: CUdeviceptr) {
        let d = &kn.dims;
        self.linear(kn, w.q_a, d.hidden, d.q_lora, x, d.hidden, self.qa, d.q_lora);
        launch_v(kn.rmsnorm, self.t as u32, 1, 1, 256, &[self.qa, w.q_a_norm, d.q_lora as u64, self.st]);
        self.linear(kn, w.q_b, d.q_lora, d.heads * d.nope, self.qa, d.q_lora, self.q, d.heads * d.nope);
    }

    /// `kva` (kv_a output, `[t][kv_lora]`) -> RMSNorm -> BF16 latent rows `pos0..pos0 + t`
    ///
    /// # Safety
    /// `kva` holds the call's rows; `c` is this sequence's cache of this layer.
    pub unsafe fn store_latent(&self, kn: &MlaKernels, w: &MlaWeights, c: &MlaCache) {
        assert_eq!(c.cap, self.cap);
        launch_v(kn.latent_store, self.t as u32, 1, 1, 256, &[self.kva, w.kv_a_norm, c.latent, self.st]);
    }

    /// `ip` (`[t][2 idx_dim + idx_heads]`) -> indexer rows `pos0..pos0 + t`
    ///
    /// # Safety
    /// As [`MlaScratch::store_latent`].
    pub unsafe fn store_index(&self, kn: &MlaKernels, w: &MlaWeights, c: &MlaCache) {
        assert_eq!(c.cap, self.cap);
        launch_v(kn.idx_store, self.t as u32, 1, 1, 256, &[self.ip, w.idx_k_norm_w, w.idx_k_norm_b, c.index, self.st]);
    }

    /// pool scores from `iq` and `ip` against the indexer rows `0..pos0 + t`, then the selection
    /// (`sel`, `sel_n`). The call's own rows must be stored ([`MlaScratch::store_index`]) first.
    ///
    /// # Safety
    /// As [`MlaScratch::store_latent`].
    pub unsafe fn select(&self, kn: &MlaKernels, w: &MlaWeights, c: &MlaCache) {
        let gx = score_grid(self.pos0, self.t);
        if gx > 0 {
            launch_v(kn.idx_scores, gx as u32, self.t.div_ceil(16) as u32, 1, 256, &[
                self.iq, self.ip, c.index, w.idx_ape, self.scores, (self.cap / KPOOL) as u64, self.st]);
        }
        launch_v(kn.sel_prep, self.t.div_ceil(256) as u32, 1, 1, 256, &[self.ncb, self.pos, self.st]);
        let (sel, sel_n) = self.sel_at();
        launch_v(kn.select, self.t as u32, 1, 1, 256, &[
            self.scores, self.ncb, sel, sel_n, self.prm, self.prm + 4, self.prm + 8, self.pos]);
    }

    /// absorbed attention: `q~ = W_k^T q`, split-K softmax over the selected latent rows, `o = W_v u`
    ///
    /// # Safety
    /// `q`, `sel`, `sel_n` hold the call's rows and the call's latent rows are stored.
    pub unsafe fn attend(&self, kn: &MlaKernels, w: &MlaWeights, c: &MlaCache) {
        let d = &kn.dims;
        let t = self.t as u32;
        self.absorb(kn, w.kv_b);
        // #196: a sub-call of a prompt call takes the whole call's split count (its rows' bits)
        let ns = attn_splits(self.split_t) as u64;
        let (sel, sel_n) = self.sel_at();
        if ns == 1 && attn2_on() && attn2_fits(d) {
            // #186: one split on tensor cores writes u directly
            launch_v(kn.attn2, t, (d.heads / 64) as u32, 1, 256, &[self.qt, c.latent, sel, sel_n, self.u, self.st]);
        } else {
            launch_v(kn.attn, d.heads.div_ceil(8) as u32, t, ns as u32, 256, &[self.qt, c.latent, sel, sel_n, self.part_o, self.part_ml, ns, self.st]);
            launch_v(kn.merge, d.heads as u32, t, 1, 256, &[self.part_o, self.part_ml, self.u, ns, self.st]);
        }
        self.out_v(kn, w.kv_b);
    }

    /// `qt = W_k^T q` per head: `gm_absorb1` for calls of at most [`DECODE_T`] rows, else the
    /// tiled `gm_absorb` (bit-identical)
    unsafe fn absorb(&self, kn: &MlaKernels, kv_b: CUdeviceptr) {
        let d = &kn.dims;
        if self.t <= DECODE_T {
            launch_v(kn.absorb1, d.heads as u32, d.kv_lora.div_ceil(128) as u32, 1, 64, &[self.q, kv_b, self.qt, self.st]);
        } else {
            launch_v(kn.absorb, d.heads as u32, self.t.div_ceil(8) as u32, 1, 256, &[self.q, kv_b, self.qt, self.st]);
        }
    }

    /// `o = W_v u` per head: `gm_out_v1` for calls of at most [`DECODE_T`] rows (its 16-byte loads
    /// need `kv_b` 16-byte aligned), else the tiled `gm_out_v` (bit-identical)
    unsafe fn out_v(&self, kn: &MlaKernels, kv_b: CUdeviceptr) {
        let d = &kn.dims;
        if self.t <= DECODE_T && (kv_b.is_multiple_of(16) || !(d.kv_lora / 32).is_multiple_of(8)) {
            launch_v(kn.out_v1, d.heads as u32, d.v.div_ceil(32) as u32, 1, 256, &[self.u, kv_b, self.o, self.st]);
        } else {
            launch_v(kn.out_v, d.heads as u32, self.t.div_ceil(8) as u32, 1, 256, &[self.u, kv_b, self.o, self.st]);
        }
    }

    /// The whole sub-block for `t` rows at `pos0..pos0 + t`: x `[t][hidden]` (the output of
    /// `input_layernorm`) -> y `[t][hidden]`, caches updated. Projections on `gm_gemm`.
    ///
    /// # Safety
    /// `x`, `y` hold `t` rows; `c` is this sequence's cache of this layer, rows `0..pos0` written by
    /// earlier calls.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward(&mut self, kn: &MlaKernels, w: &MlaWeights, c: &MlaCache, x: CUdeviceptr, y: CUdeviceptr, pos0: usize, t: usize) {
        let d = kn.dims;
        self.begin(pos0, t);
        self.query(kn, w, x);
        self.linear(kn, w.kv_a, d.hidden, d.kv_lora, x, d.hidden, self.kva, d.kv_lora);
        self.store_latent(kn, w, c);
        self.linear(kn, w.idx_x, d.hidden, d.idx_proj(), x, d.hidden, self.ip, d.idx_proj());
        self.store_index(kn, w, c);
        self.linear(kn, w.idx_wq_b, d.q_lora, d.idx_heads * d.idx_dim, self.qa, d.q_lora, self.iq, d.idx_heads * d.idx_dim);
        self.select(kn, w, c);
        self.attend(kn, w, c);
        self.linear(kn, w.o_proj, d.heads * d.v, d.hidden, self.o, d.heads * d.v, y, d.hidden);
    }

    /// #161: [`MlaScratch::forward`] with q_a, q_b, kv_a and o_proj queued by
    /// `proj(self, which, x, y)` (the caller-projection hook of the module doc; `w.q_a`, `w.q_b`,
    /// `w.kv_a`, `w.o_proj` are not read). Every other stage is `forward`'s, in its order;
    /// `forward` itself is unchanged (its G3 evidence of #163). The call's rows are `self.t()`.
    ///
    /// # Safety
    /// As [`MlaScratch::forward`]; `proj` queues on the current stream and writes `y` dense,
    /// `[t][out]`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_with(
        &mut self,
        kn: &MlaKernels,
        w: &MlaWeights,
        c: &MlaCache,
        x: CUdeviceptr,
        y: CUdeviceptr,
        pos0: usize,
        t: usize,
        proj: &mut dyn FnMut(&MlaScratch, MlaProj, CUdeviceptr, CUdeviceptr),
    ) {
        self.forward_rows_with(kn, w, c, x, y, pos0, t, 0, t, proj);
    }

    /// #196: [`MlaScratch::forward_with`] for a sub-call of a prompt call of `split_t` rows: the
    /// selection goes to rows `row_off ..` of `sel` / `sel_n` (`row_off + t <= sel_rows`) and the
    /// attention takes the split count of `split_t` rows (`split_t >= t`), so a call run as
    /// sub-calls in position order has the rows of one call. `forward_with` is `row_off` 0,
    /// `split_t` = `t`.
    ///
    /// # Safety
    /// As [`MlaScratch::forward_with`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn forward_rows_with(
        &mut self,
        kn: &MlaKernels,
        w: &MlaWeights,
        c: &MlaCache,
        x: CUdeviceptr,
        y: CUdeviceptr,
        pos0: usize,
        t: usize,
        row_off: usize,
        split_t: usize,
        proj: &mut dyn FnMut(&MlaScratch, MlaProj, CUdeviceptr, CUdeviceptr),
    ) {
        let d = kn.dims;
        self.begin(pos0, t);
        assert!(row_off + t <= self.sel_rows && split_t >= t, "glm5_mla: sub-call rows {row_off}..{} of {} (split rows {split_t})", row_off + t, self.sel_rows);
        (self.row_off, self.split_t) = (row_off, split_t);
        proj(self, MlaProj::QA, x, self.qa);
        launch_v(kn.rmsnorm, self.t as u32, 1, 1, 256, &[self.qa, w.q_a_norm, d.q_lora as u64, self.st]);
        proj(self, MlaProj::QB, self.qa, self.q);
        proj(self, MlaProj::KVA, x, self.kva);
        self.store_latent(kn, w, c);
        self.linear(kn, w.idx_x, d.hidden, d.idx_proj(), x, d.hidden, self.ip, d.idx_proj());
        self.store_index(kn, w, c);
        self.linear(kn, w.idx_wq_b, d.q_lora, d.idx_heads * d.idx_dim, self.qa, d.q_lora, self.iq, d.idx_heads * d.idx_dim);
        self.select(kn, w, c);
        self.attend(kn, w, c);
        proj(self, MlaProj::O, self.o, y);
    }

    /// # Safety
    /// No launch of this scratch is pending.
    pub unsafe fn free(&mut self) {
        for p in [&mut self.st, &mut self.prm, &mut self.sel, &mut self.sel_n, &mut self.region] {
            cuda::free_dev(p);
        }
        for p in [
            &mut self.qa, &mut self.q, &mut self.kva, &mut self.ip, &mut self.iq, &mut self.scores, &mut self.ncb, &mut self.pos,
            &mut self.qt, &mut self.part_o, &mut self.part_ml, &mut self.u, &mut self.o,
        ] {
            *p = 0;
        }
        for &e in &self.st_ev {
            cuda::ck(cudarc::driver::sys::cuEventSynchronize(e));
            cuda::event_destroy(e);
        }
        self.st_ev.clear();
        self.st_used.clear();
        self.st_ring.free();
    }
}

// ---------------------------------------------------------------- test kit

/// The synthetic weights of `oracle/export_glm5_mla_golden.py`, value for value: a counter-based
/// generator (splitmix64 of seed, stream and index), every weight rounded to BF16.
#[cfg(test)]
pub(crate) mod synth {
    use super::MlaDims;

    pub fn splitmix64(mut z: u64) -> u64 {
        z = z.wrapping_add(0x9E37_79B9_7F4A_7C15);
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// value `i` of a stream, in [-1, 1) and exact in f32: (top 24 bits of
    /// splitmix64(((stream << 32) | i) ^ seed * 0xD1B54A32D192ED03) - 2^23) / 2^23
    pub fn uniform_at(seed: u64, stream: u64, i: u64) -> f32 {
        let z = splitmix64(((stream << 32) | i) ^ seed.wrapping_mul(0xD1B5_4A32_D192_ED03));
        ((z >> 40) as i64 - (1 << 23)) as f32 / (1u32 << 23) as f32
    }

    pub fn uniform(seed: u64, stream: u64, n: usize) -> Vec<f32> {
        (0..n as u64).map(|i| uniform_at(seed, stream, i)).collect()
    }

    /// f32 -> nearest BF16 (ties to even) as f32
    pub fn bf16_round(v: f32) -> f32 {
        let b = v.to_bits() as u64;
        f32::from_bits(((b + 0x7FFF + ((b >> 16) & 1)) & 0xFFFF_0000) as u32)
    }

    pub fn bf16_bits(v: f32) -> u16 {
        (bf16_round(v).to_bits() >> 16) as u16
    }

    #[derive(Clone, Copy, PartialEq, Debug)]
    pub enum Kind {
        /// uniform * sqrt(3 / fan_in)
        Lin,
        /// 1 + p * uniform
        One(f32),
        /// p * uniform
        Lin0(f32),
    }

    /// (stream, HF module key, rows, cols, kind): the table of the Python generator (`STREAMS`)
    pub fn streams(d: &MlaDims) -> Vec<(u64, &'static str, usize, usize, Kind)> {
        use Kind::*;
        vec![
            (1, "q_a_proj.weight", d.q_lora, d.hidden, Lin),
            (2, "q_a_layernorm.weight", d.q_lora, 1, One(0.1)),
            (3, "q_b_proj.weight", d.heads * d.nope, d.q_lora, Lin),
            (4, "kv_a_proj_with_mqa.weight", d.kv_lora, d.hidden, Lin),
            (5, "kv_a_layernorm.weight", d.kv_lora, 1, One(0.1)),
            (6, "kv_b_proj.weight", d.heads * (d.nope + d.v), d.kv_lora, Lin),
            (7, "o_proj.weight", d.hidden, d.heads * d.v, Lin),
            (8, "indexer.wq_b.weight", d.idx_heads * d.idx_dim, d.q_lora, Lin),
            (9, "indexer.wk.weight", d.idx_dim, d.hidden, Lin),
            (10, "indexer.k_norm.weight", d.idx_dim, 1, One(0.1)),
            (11, "indexer.k_norm.bias", d.idx_dim, 1, Lin0(0.1)),
            (12, "indexer.weights_proj.weight", d.idx_heads, d.hidden, Lin),
            (13, "indexer.index_kpool_compress_gate", d.idx_dim, d.hidden, Lin),
            (14, "indexer.index_kpool_compress_ape", d.kpool, d.idx_dim, Lin0(0.8)),
        ]
    }
    pub const X_STREAM: u64 = 100;

    /// value `i` of a tensor with `cols` columns
    pub fn value(seed: u64, stream: u64, i: usize, cols: usize, kind: Kind) -> f32 {
        let u = uniform_at(seed, stream, i as u64);
        match kind {
            Kind::Lin => bf16_round(u * (3.0f64 / cols as f64).sqrt() as f32),
            Kind::One(p) => bf16_round(1.0f32 + p * u),
            Kind::Lin0(p) => bf16_round(p * u),
        }
    }

    pub fn tensor(seed: u64, stream: u64, rows: usize, cols: usize, kind: Kind) -> Vec<f32> {
        (0..rows * cols).map(|i| value(seed, stream, i, cols, kind)).collect()
    }

    /// input rows `[n][hidden]`: uniform * sqrt(3), plain f32
    pub fn x(seed: u64, n: usize, hidden: usize) -> Vec<f32> {
        let s = 3.0f64.sqrt() as f32;
        uniform(seed, X_STREAM, n * hidden).iter().map(|&v| v * s).collect()
    }

    /// one layer's weights on the host, f32 (BF16-representable)
    pub struct HostWeights {
        pub d: MlaDims,
        pub q_a: Vec<f32>,
        pub q_a_norm: Vec<f32>,
        pub q_b: Vec<f32>,
        pub kv_a: Vec<f32>,
        pub kv_a_norm: Vec<f32>,
        pub kv_b: Vec<f32>,
        pub o_proj: Vec<f32>,
        pub wq_b: Vec<f32>,
        /// `[wk; gate; weights_proj]` stacked
        pub idx_x: Vec<f32>,
        pub k_norm_w: Vec<f32>,
        pub k_norm_b: Vec<f32>,
        pub ape: Vec<f32>,
    }

    pub fn weights(d: &MlaDims, seed: u64) -> HostWeights {
        let mut t: Vec<Vec<f32>> = streams(d).into_iter().map(|(s, _, r, c, k)| tensor(seed, s, r, c, k)).collect();
        let mut take = |i: usize| std::mem::take(&mut t[i]);
        let (q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, o_proj, wq_b) = (take(0), take(1), take(2), take(3), take(4), take(5), take(6), take(7));
        let (wk, k_norm_w, k_norm_b, wproj, gate, ape) = (take(8), take(9), take(10), take(11), take(12), take(13));
        let idx_x = [wk, gate, wproj].concat();
        HostWeights { d: *d, q_a, q_a_norm, q_b, kv_a, kv_a_norm, kv_b, o_proj, wq_b, idx_x, k_norm_w, k_norm_b, ape }
    }
}

/// The host reference of one sub-block in f64, with the caches stored as the engine stores them
/// (BF16) or in f32, the attention in the absorbed (latent) or in HF's expanded order.
#[cfg(test)]
pub(crate) mod host {
    use super::synth::{bf16_round, HostWeights};
    use super::{MlaDims, LN_EPS, RMS_EPS};

    pub fn matvec(w: &[f32], rows: usize, k: usize, x: &[f64]) -> Vec<f64> {
        (0..rows).map(|r| w[r * k..(r + 1) * k].iter().zip(x).map(|(&a, &b)| a as f64 * b).sum()).collect()
    }

    pub fn rms(x: &[f64], w: &[f32]) -> Vec<f64> {
        let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + RMS_EPS as f64).sqrt();
        x.iter().zip(w).map(|(v, &g)| g as f64 * (v * r)).collect()
    }

    fn store(v: f64, bf16: bool) -> f64 {
        if bf16 { bf16_round(v as f32) as f64 } else { v as f32 as f64 }
    }

    /// `pk[c] = sum_j softmax_j(gate_j[c] + ape[j][c]) * key_j[c]` (`modeling_glm5_next.py:961-967`)
    pub fn pool(keys: &[&[f64]], gates: &[&[f64]], ape: &[f32], dim: usize) -> Vec<f64> {
        (0..dim)
            .map(|c| {
                let g: Vec<f64> = (0..keys.len()).map(|j| gates[j][c] + ape[j * dim + c] as f64).collect();
                let m = g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = g.iter().map(|v| (v - m).exp()).collect();
                let s: f64 = e.iter().sum();
                (0..keys.len()).map(|j| e[j] / s * keys[j][c]).sum()
            })
            .collect()
    }

    /// the token set of a query at `p` from its pool scores (`scores[P]`, P < (p+1)/4): every pool
    /// while there are at most `sel_pools`, else the `sel_pools` best (ties: lowest pool), then the tail
    pub fn select(scores: &[f64], p: usize, d: &MlaDims) -> Vec<usize> {
        let ncb = (p + 1) / d.kpool;
        let mut pools: Vec<usize> = (0..ncb).collect();
        if ncb > d.sel_pools() {
            pools.sort_by(|&a, &b| scores[b].partial_cmp(&scores[a]).unwrap().then(a.cmp(&b)));
            pools.truncate(d.sel_pools());
            pools.sort();
        }
        let mut s: Vec<usize> = pools.iter().flat_map(|&q| (0..d.kpool).map(move |j| q * d.kpool + j)).collect();
        s.extend(ncb * d.kpool..=p);
        s
    }

    pub struct Out {
        /// `[n][hidden]`
        pub y: Vec<f64>,
        /// per row: the selected token ids, ascending
        pub sel: Vec<Vec<usize>>,
    }

    /// every row of `x` (`[n][hidden]`) as a causal sequence from position 0
    pub fn forward(w: &HostWeights, x: &[f32], bf16_cache: bool, absorbed: bool) -> Out {
        let d = &w.d;
        let n = x.len() / d.hidden;
        let (id, ih, lat) = (d.idx_dim, d.idx_heads, d.kv_lora);
        let mut q = Vec::new();
        let mut lat_c = Vec::new();
        let mut keys = Vec::new();
        let mut gates = Vec::new();
        let mut iq = Vec::new();
        let mut wts = Vec::new();
        for r in 0..n {
            let xr: Vec<f64> = x[r * d.hidden..(r + 1) * d.hidden].iter().map(|&v| v as f64).collect();
            let qr = rms(&matvec(&w.q_a, d.q_lora, d.hidden, &xr), &w.q_a_norm);
            q.push(matvec(&w.q_b, d.heads * d.nope, d.q_lora, &qr));
            iq.push(matvec(&w.wq_b, ih * id, d.q_lora, &qr));
            let c = rms(&matvec(&w.kv_a, lat, d.hidden, &xr), &w.kv_a_norm);
            lat_c.push(c.iter().map(|&v| store(v, bf16_cache)).collect::<Vec<f64>>());
            let ip = matvec(&w.idx_x, d.idx_proj(), d.hidden, &xr);
            let k = &ip[..id];
            let mean = k.iter().sum::<f64>() / id as f64;
            let var = k.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / id as f64;
            let rr = 1.0 / (var + LN_EPS as f64).sqrt();
            keys.push((0..id).map(|c| store((k[c] - mean) * rr * w.k_norm_w[c] as f64 + w.k_norm_b[c] as f64, bf16_cache)).collect::<Vec<f64>>());
            gates.push(ip[id..2 * id].iter().map(|&v| store(v, bf16_cache)).collect::<Vec<f64>>());
            wts.push(ip[2 * id..].iter().map(|v| v * (ih as f64).powf(-0.5)).collect::<Vec<f64>>());
        }
        let pooled: Vec<Vec<f64>> = (0..n / d.kpool)
            .map(|pp| {
                let kr: Vec<&[f64]> = (0..d.kpool).map(|j| &keys[pp * d.kpool + j][..]).collect();
                let gr: Vec<&[f64]> = (0..d.kpool).map(|j| &gates[pp * d.kpool + j][..]).collect();
                pool(&kr, &gr, &w.ape, id)
            })
            .collect();
        let row = d.nope + d.v;
        let scale = (d.nope as f64).powf(-0.5);
        let mut y = vec![0.0; n * d.hidden];
        let mut sel = Vec::new();
        for p in 0..n {
            let ncb = (p + 1) / d.kpool;
            let scores: Vec<f64> = (0..ncb)
                .map(|pp| {
                    (0..ih)
                        .map(|h| {
                            let dot: f64 = (0..id).map(|c| iq[p][h * id + c] * pooled[pp][c]).sum();
                            wts[p][h] * (dot * (id as f64).powf(-0.5)).max(0.0)
                        })
                        .sum()
                })
                .collect();
            let s = select(&scores, p, d);
            let mut o = vec![0.0; d.heads * d.v];
            for h in 0..d.heads {
                let wk = |i: usize, j: usize| w.kv_b[(h * row + i) * lat + j] as f64;
                let wv = |i: usize, j: usize| w.kv_b[(h * row + d.nope + i) * lat + j] as f64;
                let qh = &q[p][h * d.nope..(h + 1) * d.nope];
                let logits: Vec<f64> = if absorbed {
                    let qt: Vec<f64> = (0..lat).map(|j| (0..d.nope).map(|i| qh[i] * wk(i, j)).sum()).collect();
                    s.iter().map(|&t| (0..lat).map(|j| qt[j] * lat_c[t][j]).sum::<f64>() * scale).collect()
                } else {
                    s.iter()
                        .map(|&t| (0..d.nope).map(|i| qh[i] * (0..lat).map(|j| wk(i, j) * lat_c[t][j]).sum::<f64>()).sum::<f64>() * scale)
                        .collect()
                };
                let m = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let e: Vec<f64> = logits.iter().map(|v| (v - m).exp()).collect();
                let z: f64 = e.iter().sum();
                if absorbed {
                    let u: Vec<f64> = (0..lat).map(|j| s.iter().zip(&e).map(|(&t, &a)| a / z * lat_c[t][j]).sum()).collect();
                    for i in 0..d.v {
                        o[h * d.v + i] = (0..lat).map(|j| wv(i, j) * u[j]).sum();
                    }
                } else {
                    for i in 0..d.v {
                        o[h * d.v + i] = s.iter().zip(&e).map(|(&t, &a)| a / z * (0..lat).map(|j| wv(i, j) * lat_c[t][j]).sum::<f64>()).sum();
                    }
                }
            }
            y[p * d.hidden..(p + 1) * d.hidden].copy_from_slice(&matvec(&w.o_proj, d.hidden, d.heads * d.v, &o));
            sel.push(s);
        }
        Out { y, sel }
    }
}

#[cfg(test)]
mod tests {
    //! Host tests: the family row's shapes, the generator against the oracle's probes, the exact ops
    //! of the host reference and the NVRTC build. No GPU.
    use super::host;
    use super::synth;
    use super::*;

    /// a small layer with the real structure (pools of 4, a sparse regime from position 19)
    pub(crate) const TINY: MlaDims = MlaDims { hidden: 128, heads: 4, nope: 32, v: 32, kv_lora: 64, q_lora: 64, idx_heads: 4, idx_dim: 32, kpool: 4, topk: 16 };

    pub(crate) fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5/mla")
    }

    pub(crate) fn manifest() -> serde_json::Value {
        let p = fixture_dir().join("manifest.json");
        serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap()
    }

    #[test]
    fn the_glm_5_3_flash_layer_has_the_planned_shapes_and_cache_bytes() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let d = MlaDims::of(&g);
        assert_eq!((d.heads, d.nope, d.v, d.kv_lora, d.q_lora), (64, 256, 256, 512, 1536));
        assert_eq!((d.idx_heads, d.idx_dim, d.kpool, d.topk), (32, 128, 4, 2048));
        assert_eq!((d.sel_pools(), d.sel_max(), d.idx_row(), d.idx_proj()), (512, 2051, 257, 288));
        assert_eq!(d.sel_max(), g.sel_max());
        // the #159 planner's per-token bytes are this module's cache rows
        assert_eq!((d.latent_bytes_per_token(), d.indexer_bytes_per_token()), (g.latent_bytes_per_token(), g.indexer_bytes_per_token()));
        assert_eq!(MlaCache::bytes(&d, 200_000), 307_600_000);
        assert_eq!(d.check(), Ok(()));
        let p = d.prelude();
        for line in ["#define GM_LAT 512\n", "#define GM_ROW 257\n", "#define GM_SEL_MAX 2051\n", "#define GM_SCALE 6.25e-2f\n"] {
            assert!(p.contains(line), "{line:?} not in\n{p}");
        }
        assert!(MlaDims { kpool: 16, ..d }.check().unwrap_err().contains("kpool 16"));
        assert_eq!((attn_splits(1), attn_splits(3), attn_splits(16), attn_splits(512)), (16, 5, 1, 1));
    }

    /// the Rust generator is the oracle's: the probe values the golden's manifest recorded
    #[test]
    fn the_generator_reproduces_the_oracle_probes() {
        let m = manifest();
        let seed = m["seed"].as_u64().unwrap();
        let d = MlaDims::of(&Glm5Geo::GLM_5_3_FLASH);
        let mut n = 0;
        for (s, key, rows, cols, kind) in synth::streams(&d) {
            let probes = m["probes"][key].as_array().unwrap_or_else(|| panic!("{key}: no probes"));
            assert_eq!(probes[3][0].as_u64().unwrap() as usize, rows * cols - 1, "{key}: shape");
            for pr in probes {
                let (i, bits) = (pr[0].as_u64().unwrap() as usize, pr[1].as_u64().unwrap() as u32);
                let v = synth::value(seed, s, i, cols, kind);
                assert_eq!(v.to_bits(), bits, "{key}[{i}]: {v} vs oracle {}", f32::from_bits(bits));
                n += 1;
            }
        }
        let x = synth::x(seed, 8, d.hidden);
        for pr in m["probes"]["x"].as_array().unwrap() {
            let (i, bits) = (pr[0].as_u64().unwrap() as usize, pr[1].as_u64().unwrap() as u32);
            assert_eq!(x[i].to_bits(), bits, "x[{i}]");
            n += 1;
        }
        assert_eq!(n, 15 * 4);
    }

    #[test]
    fn bf16_rounding_is_round_to_nearest_even() {
        assert_eq!(synth::bf16_round(1.0), 1.0);
        // 1 + 2^-8 is the midpoint of 1 and 1 + 2^-7: ties to the even mantissa (1.0)
        assert_eq!(synth::bf16_round(1.0 + 1.0 / 256.0), 1.0);
        assert_eq!(synth::bf16_round(1.0 + 3.0 / 256.0), 1.0 + 4.0 / 256.0);
        assert_eq!(synth::bf16_round(-1.0 - 1.0 / 512.0), -1.0);
        assert_eq!(synth::bf16_bits(1.0), 0x3F80);
    }

    /// pooling is the learned softmax mix, not the mean: equal gates give the mean, a dominant gate
    /// picks its row, and the ape is added per (position, channel)
    #[test]
    fn the_pooled_key_is_the_softmax_gate_mix_of_four_rows() {
        let k: [Vec<f64>; 4] = [vec![1.0, 10.0], vec![2.0, 20.0], vec![3.0, 30.0], vec![4.0, 40.0]];
        let kr: Vec<&[f64]> = k.iter().map(|v| &v[..]).collect();
        let zero = [vec![0.0; 2], vec![0.0; 2], vec![0.0; 2], vec![0.0; 2]];
        let zr: Vec<&[f64]> = zero.iter().map(|v| &v[..]).collect();
        let no_ape = [0.0f32; 8];
        assert_eq!(host::pool(&kr, &zr, &no_ape, 2), vec![2.5, 25.0]);
        let big = [vec![0.0; 2], vec![0.0; 2], vec![60.0, 0.0], vec![0.0; 2]];
        let br: Vec<&[f64]> = big.iter().map(|v| &v[..]).collect();
        let p = host::pool(&kr, &br, &no_ape, 2);
        assert!((p[0] - 3.0).abs() < 1e-20 && p[1] == 25.0, "{p:?}");
        // ape [4][2]: channel 1 of position 0 gets ln 3 -> weights 3/6, 1/6, 1/6, 1/6
        let mut ape = [0.0f32; 8];
        ape[1] = 3.0f32.ln();
        let p = host::pool(&kr, &zr, &ape, 2);
        let want = (3.0 * 10.0 + 20.0 + 30.0 + 40.0) / 6.0;
        assert!((p[1] - want).abs() < 1e-5, "{} vs {want}", p[1]);
    }

    /// the selection: dense below the budget, top pools + tail above it, ties to the lowest pool
    #[test]
    fn the_selection_is_the_top_pools_plus_the_tail() {
        let d = TINY; // 4 pools budget
        // p = 9: 2 complete pools, tail 8, 9 -> dense, 0..=9
        assert_eq!(host::select(&[0.0, 0.0], 9, &d), (0..=9).collect::<Vec<_>>());
        // p = 22: 5 complete pools, tail 20..=22; pool 1 has the lowest score and is dropped
        let s = host::select(&[5.0, -1.0, 3.0, 4.0, 2.0], 22, &d);
        let want: Vec<usize> = [0usize, 2, 3, 4].iter().flat_map(|&q| q * 4..q * 4 + 4).chain(20..=22).collect();
        assert_eq!(s, want);
        // an exact tie at the boundary: pools 1 and 4 both 1.0, the lower index stays
        let s = host::select(&[5.0, 1.0, 3.0, 4.0, 1.0], 19, &d);
        assert_eq!(s, (0..16).collect::<Vec<_>>());
        assert_eq!(s.len(), d.topk);
        assert!(host::select(&[0.0; 512], 2050, &MlaDims::of(&Glm5Geo::GLM_5_3_FLASH)).len() == 2051);
    }

    /// the absorbed (latent) attention equals HF's expanded K/V order on the same weights: the
    /// algebra of `q~_h = W_k,h^T q_h` and `o_h = W_v,h u_h` (recipe section 7), sparse rows included
    #[test]
    fn absorbed_attention_equals_the_expanded_form() {
        let w = synth::weights(&TINY, 7);
        let x = synth::x(7, 26, TINY.hidden);
        let a = host::forward(&w, &x, true, true);
        let e = host::forward(&w, &x, true, false);
        assert_eq!(a.sel, e.sel);
        assert!(a.sel[25].len() == TINY.topk + 2, "row 25 is sparse: {} rows", a.sel[25].len());
        let rmsy = (e.y.iter().map(|v| v * v).sum::<f64>() / e.y.len() as f64).sqrt();
        let worst = a.y.iter().zip(&e.y).map(|(p, q)| (p - q).abs()).fold(0.0, f64::max);
        assert!(worst < 1e-12 * rmsy.max(1.0), "absorbed vs expanded: max |d| {worst:e} at rms {rmsy:e}");
    }

    /// the module source compiles with the engine's option set into one PTX module with every entry
    /// of `NAMES`, at the real and at the tiny shapes (host only, NVRTC)
    #[test]
    fn the_source_compiles_with_every_entry() {
        for d in [MlaDims::of(&Glm5Geo::GLM_5_3_FLASH), TINY] {
            let ptx = crate::kernels::tests_300_c4::ptx(&d.source());
            let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|e| e.0).collect();
            assert_eq!(names.len(), NAMES.len(), "{names:?}");
            for n in NAMES {
                assert!(names.iter().any(|e| e == n), "{n} missing in {names:?}");
            }
        }
        // the decode-kernel constants the host routes and launches by
        let src = crate::kernels::GLM5_MLA_SRC;
        assert!(src.contains(&format!("#define GM_DT {DECODE_T} ")), "GM_DT != DECODE_T");
        assert!(src.contains(&format!("#define GM_GV_R {GEMV_ROWS} ")), "GM_GV_R != GEMV_ROWS");
        // the decode kernels keep their staging arrays in registers (#191: a run-time-indexed
        // local array was the dense GEMVs' bottleneck)
        for d in [MlaDims::of(&Glm5Geo::GLM_5_3_FLASH), TINY] {
            let ptx = crate::kernels::tests_300_c4::ptx(&d.source());
            for (n, body) in crate::kernels::tests_300_c4::entries(&ptx) {
                if ["gm_gemv", "gm_absorb1", "gm_out_v1"].contains(&n.as_str()) {
                    assert!(!body.contains(".local"), "{n} uses local memory");
                }
            }
        }
    }
}

#[cfg(test)]
mod tests_gpu {
    //! #163 acceptance on the GPU: the kernels against the host reference at small shapes, and the
    //! whole sub-block against the HF golden (`oracle/export_glm5_mla_golden.py`) at the real shapes on
    //! synthetic weights. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_mla::tests_gpu -- --ignored --nocapture --test-threads 1`.
    use super::host;
    use super::synth::{self, HostWeights};
    use super::tests::{fixture_dir, manifest, TINY};
    use super::*;

    unsafe fn bf16_dev(v: &[f32]) -> CUdeviceptr {
        cuda::to_dev(&v.iter().map(|&x| synth::bf16_bits(x)).collect::<Vec<u16>>())
    }

    unsafe fn upload(w: &HostWeights) -> MlaWeights {
        for v in [&w.q_a, &w.q_b, &w.kv_a, &w.kv_b, &w.o_proj, &w.wq_b, &w.idx_x] {
            assert!(v.iter().all(|&x| synth::bf16_round(x) == x), "a weight is not BF16-representable");
        }
        MlaWeights {
            q_a: bf16_dev(&w.q_a),
            q_a_norm: cuda::to_f32_dev(&w.q_a_norm),
            q_b: bf16_dev(&w.q_b),
            kv_a: bf16_dev(&w.kv_a),
            kv_a_norm: cuda::to_f32_dev(&w.kv_a_norm),
            kv_b: bf16_dev(&w.kv_b),
            o_proj: bf16_dev(&w.o_proj),
            idx_wq_b: bf16_dev(&w.wq_b),
            idx_x: bf16_dev(&w.idx_x),
            idx_k_norm_w: cuda::to_f32_dev(&w.k_norm_w),
            idx_k_norm_b: cuda::to_f32_dev(&w.k_norm_b),
            idx_ape: cuda::to_f32_dev(&w.ape),
        }
    }

    unsafe fn free_weights(w: &mut MlaWeights) {
        for p in [&mut w.q_a, &mut w.q_a_norm, &mut w.q_b, &mut w.kv_a, &mut w.kv_a_norm, &mut w.kv_b, &mut w.o_proj, &mut w.idx_wq_b, &mut w.idx_x, &mut w.idx_k_norm_w, &mut w.idx_k_norm_b, &mut w.idx_ape] {
            cuda::free_dev(p);
        }
    }

    /// `qsa_select_fast` from the engine's main module (Flash-Next prelude; the entry reads no define)
    unsafe fn main_select() -> (cuda::Module, CUfunction) {
        let m = cuda::compile(&crate::kernels::KernelGeo::flash_next().source());
        let f = m.get("qsa_select_fast");
        (m, f)
    }

    /// One run: prompt rows `0..t_prompt` in calls of `chunk`, then one row per call. Returns the
    /// output rows `[n][hidden]` and every row's selection (ascending token ids).
    unsafe fn run(kn: &MlaKernels, w: &MlaWeights, x: &[f32], t_prompt: usize, chunk: usize) -> (Vec<f32>, Vec<Vec<usize>>) {
        let d = kn.dims;
        let n = x.len() / d.hidden;
        let mut cache = MlaCache::new(&d, n);
        let mut s = MlaScratch::new(&d, chunk, n);
        let xd = cuda::alloc_zeroed(chunk * d.hidden * 4);
        let yd = cuda::alloc_zeroed(chunk * d.hidden * 4);
        let mut y = vec![0f32; n * d.hidden];
        let mut sel = Vec::with_capacity(n);
        let mut calls: Vec<(usize, usize)> = (0..t_prompt).step_by(chunk).map(|r| (r, (r + chunk).min(t_prompt))).collect();
        calls.extend((t_prompt..n).map(|r| (r, r + 1)));
        for (r0, r1) in calls {
            let t = r1 - r0;
            cuda::to_f32_into(xd, &x[r0 * d.hidden..r1 * d.hidden]);
            s.forward(kn, w, &cache, xd, yd, r0, t);
            cuda::sync();
            y[r0 * d.hidden..r1 * d.hidden].copy_from_slice(&cuda::dtoh(yd, t * d.hidden));
            let sn = cuda::dtoh_i32(s.sel_n, t);
            let sl = cuda::dtoh_i32(s.sel, t * d.sel_max());
            for (i, &k) in sn.iter().enumerate() {
                let mut v: Vec<usize> = sl[i * d.sel_max()..i * d.sel_max() + k as usize].iter().map(|&e| e as usize).collect();
                v.sort();
                sel.push(v);
            }
        }
        s.free();
        cache.free();
        let (mut xd, mut yd) = (xd, yd);
        cuda::free_dev(&mut xd);
        cuda::free_dev(&mut yd);
        (y, sel)
    }

    fn cosine(a: &[f32], b: &[f64]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0.0, 0.0, 0.0);
        for (&p, &q) in a.iter().zip(b) {
            ab += p as f64 * q;
            aa += p as f64 * p as f64;
            bb += q * q;
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    /// the kernels at small shapes against the f64 host reference with the same BF16 caches:
    /// prefill in calls of 12 rows, then decode rows; dense rows 0..18, sparse rows from 19 on
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mla::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mla_gpu_small_shapes_match_the_host_reference() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, select) = main_select();
            let kn = MlaKernels::new(TINY, select);
            let hw = synth::weights(&TINY, 11);
            let (t_prompt, n) = (40, 46);
            let x = synth::x(11, n, TINY.hidden);
            let r = host::forward(&hw, &x, true, true);
            let mut w = upload(&hw);
            let (y, sel) = run(&kn, &w, &x, t_prompt, 12);
            let mut worst_cos: f64 = 1.0;
            for p in 0..n {
                assert_eq!(sel[p], r.sel[p], "row {p}: selection");
                let h = TINY.hidden;
                let c = cosine(&y[p * h..(p + 1) * h], &r.y[p * h..(p + 1) * h]);
                worst_cos = worst_cos.min(c);
                let rms = (r.y[p * h..(p + 1) * h].iter().map(|v| v * v).sum::<f64>() / h as f64).sqrt();
                let md = y[p * h..(p + 1) * h].iter().zip(&r.y[p * h..(p + 1) * h]).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
                assert!(md <= 1e-4 * rms, "row {p}: max |d| {md:e} at rms {rms:e}");
            }
            assert!(sel[45].len() == TINY.topk + 2 && sel[18].len() == 19, "{} {}", sel[45].len(), sel[18].len());
            println!("small shapes: {n} rows, every selection identical, worst cosine {worst_cos:.9}");
            free_weights(&mut w);
        }
    }

    fn read_f32(name: &str) -> Vec<f32> {
        let b = std::fs::read(fixture_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    fn read_u16(name: &str) -> Vec<u16> {
        let b = std::fs::read(fixture_dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        b.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect()
    }

    /// Gate G3 of the plan on synthetic weights with the real block shapes: the sub-block output at
    /// every anchor row (prompt and decode) has cosine >= 0.9999 against the HF golden with the
    /// engine's cache precision (`bf16kv`), and every row's selection equals the golden's (dense
    /// rows: the causal set; sparse rows p >= 2051: the same 512 pools), ties excepted and counted.
    /// The engine runs the prompt in calls of 384 rows (the golden: 512), then 8 decode rows.
    /// The cost of the BF16 cache against HF in f32 is printed, not gated (docs/glm5-mla.md).
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mla::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mla_gpu_matches_the_hf_golden_at_real_shapes() {
        let m = manifest();
        let d = MlaDims::of(&Glm5Geo::GLM_5_3_FLASH);
        let (seed, t_prompt, n) = (m["seed"].as_u64().unwrap(), m["T"].as_u64().unwrap() as usize, m["N"].as_u64().unwrap() as usize);
        let anchors: Vec<usize> = m["anchors"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let sp = m["sparse_rows"].as_array().unwrap();
        let (s0, s1) = (sp[0].as_u64().unwrap() as usize, sp[1].as_u64().unwrap() as usize);
        assert_eq!((s0, s1), (2051, n - 1));
        let gold = read_f32("golden-bf16kv-anchors.f32");
        let gold32 = read_f32("golden-f32-anchors.f32");
        let tk = read_u16("topk-bf16kv-sparse.u16");
        let tk32 = read_u16("topk-f32-sparse.u16");
        let ties: Vec<usize> = m["tie_rows"]["bf16kv"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let hw = synth::weights(&d, seed);
        let x = synth::x(seed, n, d.hidden);
        let (y, sel) = unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, select) = main_select();
            let kn = MlaKernels::new(d, select);
            let mut w = upload(&hw);
            let out = run(&kn, &w, &x, t_prompt, 384);
            free_weights(&mut w);
            out
        };
        drop(hw);
        // selections, every row
        let (mut bad, mut tie_skips, mut vs_f32) = (Vec::new(), 0, 0);
        for p in 0..n {
            let ncb = (p + 1) / 4;
            if ncb <= d.sel_pools() {
                if sel[p] != (0..=p).collect::<Vec<_>>() {
                    bad.push(p);
                }
                continue;
            }
            let pools: Vec<usize> = sel[p].iter().filter(|&&t| t < 4 * ncb).step_by(4).map(|&t| t / 4).collect();
            let tail: Vec<usize> = sel[p].iter().cloned().filter(|&t| t >= 4 * ncb).collect();
            let i = p - s0;
            let want: Vec<usize> = tk[i * 512..(i + 1) * 512].iter().map(|&v| v as usize).collect();
            let want32: Vec<usize> = tk32[i * 512..(i + 1) * 512].iter().map(|&v| v as usize).collect();
            assert_eq!(tail, (4 * ncb..=p).collect::<Vec<_>>(), "row {p}: tail");
            assert_eq!(sel[p].len(), 2048 + tail.len(), "row {p}: whole pools");
            if pools != want {
                if ties.contains(&p) { tie_skips += 1 } else { bad.push(p) }
            }
            if pools != want32 {
                vs_f32 += 1;
            }
        }
        assert!(bad.is_empty(), "selection differs from the bf16kv golden on rows {bad:?}");
        // outputs at the anchors
        let h = d.hidden;
        let mut worst = (1.0f64, 0usize);
        let mut worst32 = (1.0f64, 0usize);
        for (k, &p) in anchors.iter().enumerate() {
            let g: Vec<f64> = gold[k * h..(k + 1) * h].iter().map(|&v| v as f64).collect();
            let g32: Vec<f64> = gold32[k * h..(k + 1) * h].iter().map(|&v| v as f64).collect();
            let c = cosine(&y[p * h..(p + 1) * h], &g);
            let c32 = cosine(&y[p * h..(p + 1) * h], &g32);
            let md = y[p * h..(p + 1) * h].iter().zip(&g).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
            let rms = (g.iter().map(|v| v * v).sum::<f64>() / h as f64).sqrt();
            println!("anchor {p:5} {}: cosine {c:.9}  max|d| {md:.3e}  rms {rms:.3e}  | vs f32 golden {c32:.9}",
                if p < t_prompt { "prompt" } else { "decode" });
            if c < worst.0 { worst = (c, p) }
            if c32 < worst32.0 { worst32 = (c32, p) }
        }
        println!("selection: {} rows, sparse rows {s0}..={s1}, 0 differ from bf16kv (tie rows skipped {tie_skips}); {vs_f32} sparse rows differ from the f32 golden", n);
        println!("worst anchor cosine vs bf16kv {:.9} (row {}), vs f32 {:.9} (row {})", worst.0, worst.1, worst32.0, worst32.1);
        assert!(worst.0 >= 0.9999, "G3: anchor {} cosine {:.9} < 0.9999", worst.1, worst.0);
    }

    /// #186 `CROW_GLM_ATTN2`: `gm_attn2` (tensor cores) against `gm_attn` + `gm_attn_merge` (one
    /// split, the prompt path) at the real shapes, 8192 prompt rows over a synthetic latent cache of
    /// 16384 rows, selections of 512 random pools of 4 plus 3 tail rows (short dense rows first);
    /// cosine of u and the time of both
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mla::tests_gpu::glm5_attn2 -- --ignored --nocapture"]
    fn glm5_attn2_matches_gm_attn_at_8192_prompt_rows() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let d = MlaDims::of(&Glm5Geo::GLM_5_3_FLASH);
            assert!(attn2_fits(&d));
            let m = crate::kernels::glm5_mla_module(&d.prelude());
            let (attn, merge, attn2) = (m.get("gm_attn"), m.get("gm_attn_merge"), m.get("gm_attn2"));
            let (t, cap, sm) = (8192usize, 16384usize, d.sel_max());
            let (h, lat) = (d.heads, d.kv_lora);
            let mut rng = 0x2545F4914F6CDD1Du64;
            let mut next = move || {
                rng ^= rng << 13;
                rng ^= rng >> 7;
                rng ^= rng << 17;
                rng
            };
            let unif = |r: u64| (r >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0;
            let lat_h: Vec<u16> = (0..cap * lat).map(|_| synth::bf16_bits(unif(next()) * 1.7)).collect();
            let qt_h: Vec<f32> = (0..t * h * lat).map(|_| unif(next()) * 3.5).collect();
            let mut sel_h = vec![0i32; t * sm];
            let mut sel_n_h = vec![0i32; t];
            for tok in 0..t {
                let row = &mut sel_h[tok * sm..(tok + 1) * sm];
                let n = if tok < 40 {
                    for (i, v) in row.iter_mut().take(tok + 1).enumerate() {
                        *v = i as i32;
                    }
                    tok + 1
                } else {
                    for p in 0..d.sel_pools() {
                        let pool = (next() % (cap as u64 / 4)) as i32;
                        for j in 0..4 {
                            row[p * 4 + j] = pool * 4 + j as i32;
                        }
                    }
                    for j in 0..3 {
                        row[d.sel_pools() * 4 + j] = (next() % cap as u64) as i32;
                    }
                    sm
                };
                sel_n_h[tok] = n as i32;
            }
            let latd = cuda::to_dev(&lat_h);
            let qtd = cuda::to_f32_dev(&qt_h);
            drop(qt_h);
            let seld = cuda::to_dev(&sel_h);
            let selnd = cuda::to_dev(&sel_n_h);
            let std_ = cuda::to_dev(&[0i32, t as i32]);
            let part_o = cuda::alloc_zeroed(t * h * lat * 4);
            let part_ml = cuda::alloc_zeroed(t * h * 2 * 4);
            let u_old = cuda::alloc_zeroed(t * h * lat * 4);
            let u_new = cuda::alloc_zeroed(t * h * lat * 4);
            let run_old = || {
                launch_v(attn, h.div_ceil(8) as u32, t as u32, 1, 256, &[qtd, latd, seld, selnd, part_o, part_ml, 1, std_]);
                launch_v(merge, h as u32, t as u32, 1, 256, &[part_o, part_ml, u_old, 1, std_]);
            };
            let run_new = || launch_v(attn2, t as u32, (h / 64) as u32, 1, 256, &[qtd, latd, seld, selnd, u_new, std_]);
            let time = |f: &dyn Fn()| {
                f();
                cuda::sync();
                let mut best = f64::MAX;
                for _ in 0..3 {
                    let s = std::time::Instant::now();
                    f();
                    cuda::sync();
                    best = best.min(s.elapsed().as_secs_f64() * 1e3);
                }
                best
            };
            let t_old = time(&run_old);
            let t_new = time(&run_new);
            // sub-call sizes: rows 4096 .. 4096 + ts (all sparse, full selections)
            for ts in [256usize, 1024] {
                let (q, sl, sn) = (qtd + (4096 * h * lat * 4) as u64, seld + (4096 * sm * 4) as u64, selnd + 4096 * 4);
                let o = || {
                    launch_v(attn, h.div_ceil(8) as u32, ts as u32, 1, 256, &[q, latd, sl, sn, part_o, part_ml, 1, std_]);
                    launch_v(merge, h as u32, ts as u32, 1, 256, &[part_o, part_ml, u_old, 1, std_]);
                };
                let nw = || launch_v(attn2, ts as u32, (h / 64) as u32, 1, 256, &[q, latd, sl, sn, u_new, std_]);
                let (a, b) = (time(&o), time(&nw));
                println!("{ts} rows: gm_attn + merge {a:.3} ms, gm_attn2 {b:.3} ms ({:.2}x)", a / b);
            }
            run_old();
            run_new();
            cuda::sync();
            let a = cuda::dtoh(u_old, t * h * lat);
            let b = cuda::dtoh(u_new, t * h * lat);
            let (mut ab, mut aa, mut bb, mut worst, mut worst_at) = (0f64, 0f64, 0f64, 1f64, 0usize);
            for r in 0..t * h {
                let (mut rab, mut raa, mut rbb) = (0f64, 0f64, 0f64);
                for j in 0..lat {
                    let (x, y) = (a[r * lat + j] as f64, b[r * lat + j] as f64);
                    rab += x * y;
                    raa += x * x;
                    rbb += y * y;
                }
                ab += rab;
                aa += raa;
                bb += rbb;
                let c = rab / (raa.sqrt() * rbb.sqrt());
                if !(c >= worst) {
                    (worst, worst_at) = (c, r);
                }
            }
            let cos = ab / (aa.sqrt() * bb.sqrt());
            println!("gm_attn + merge {t_old:.2} ms, gm_attn2 {t_new:.2} ms ({:.2}x) at {t} rows x {h} heads, sel {sm}", t_old / t_new);
            println!("u cosine {cos:.9}, worst (row, head) {worst:.9} at tok {} head {}", worst_at / h, worst_at % h);
            for p in [latd, qtd, seld, selnd, std_, part_o, part_ml, u_old, u_new] {
                let mut p = p;
                cuda::free_dev(&mut p);
            }
            assert!(cos >= 0.9999, "u cosine {cos:.9} < 0.9999");
            assert!(worst >= 0.999, "worst row cosine {worst:.9} < 0.999");
        }
    }
}

#[cfg(test)]
mod tests_decode_gpu {
    //! Follow-up of #191 on the GPU (RTX 5090, sm_120): the decode kernels `gm_gemv`,
    //! `gm_absorb1`, `gm_out_v1` (calls of at most `DECODE_T` rows) bit-identical to the tiled
    //! `gm_gemm`, `gm_absorb`, `gm_out_v` they replace there, and their time per decode row at
    //! the GLM-5.3-Flash shapes on synthetic BF16 weights. Each timed launch reads a different
    //! copy of its matrix (>= 256 MiB per shape), so the weights come from VRAM as in a decode
    //! row, not from the L2. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_mla_decode_gpu -- --ignored --nocapture --test-threads 1`.
    use super::tests::TINY;
    use super::*;
    use crate::cpu_mul1::testkit::Rng;
    use cudarc::driver::sys;

    unsafe fn kernels(d: MlaDims) -> (cuda::Module, MlaKernels) {
        let m = cuda::compile(&crate::kernels::KernelGeo::flash_next().source());
        let f = m.get("qsa_select_fast");
        (m, MlaKernels::new(d, f))
    }

    /// random BF16 weights: finite values, every 13th +0 or -0 (the signed-zero paths)
    fn bf16s(n: usize, rng: &mut Rng) -> Vec<u16> {
        (0..n)
            .map(|i| match i % 13 {
                4 => 0x0000,
                9 => 0x8000,
                _ => synth::bf16_bits(rng.f(0.5)),
            })
            .collect()
    }

    fn f32s(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|i| if i % 97 == 5 { -0.0 } else { rng.f(2.0) }).collect()
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    unsafe fn nan_dev(n: usize) -> CUdeviceptr {
        cuda::to_f32_dev(&vec![f32::NAN; n])
    }

    unsafe fn same(what: &str, a: CUdeviceptr, b: CUdeviceptr, n: usize) {
        cuda::sync();
        let (p, q) = (bits(&cuda::dtoh(a, n)), bits(&cuda::dtoh(b, n)));
        let differ = p.iter().zip(&q).filter(|(x, y)| x != y).count();
        assert_eq!(differ, 0, "{what}: {differ} of {n} outputs differ from the tiled kernel");
    }

    /// the tiled `gm_gemm` launch of `MlaScratch::linear` (the path before the decode kernels)
    #[allow(clippy::too_many_arguments)]
    unsafe fn gemm_tiled(kn: &MlaKernels, s: &MlaScratch, w: CUdeviceptr, k: usize, n: usize, x: CUdeviceptr, ldx: usize, y: CUdeviceptr, ldy: usize) {
        launch_v(kn.gemm, n.div_ceil(64) as u32, s.t().div_ceil(16) as u32, 1, 256, &[w, x, y, k as u64, n as u64, ldx as u64, ldy as u64, s.st]);
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mla_decode_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mla_decode_gpu_kernels_are_bit_identical_to_the_tiled_kernels() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut rng = Rng(0x6d1a_dec0);
            for d in [MlaDims::of(&Glm5Geo::GLM_5_3_FLASH), TINY] {
                let (_m, kn) = kernels(d);
                let mut s = MlaScratch::new(&d, DECODE_T, 64);
                // (k, n, ldx, ldy): the indexer x-side and q-side projections, the MTP eh_proj,
                // a row tail with padded strides
                let shapes = [
                    (d.hidden, d.idx_proj(), d.hidden, d.idx_proj()),
                    (d.q_lora, d.idx_heads * d.idx_dim, d.q_lora, d.idx_heads * d.idx_dim),
                    (2 * d.hidden, d.hidden, 2 * d.hidden, d.hidden),
                    (d.hidden, 13, d.hidden + 4, 17),
                ];
                for t in 1..=DECODE_T {
                    s.begin(0, t);
                    for &(k, n, ldx, ldy) in &shapes {
                        let w = cuda::to_dev(&bf16s(n * k, &mut rng));
                        let x = cuda::to_f32_dev(&f32s(t * ldx, &mut rng));
                        let (ya, yb) = (nan_dev(t * ldy), nan_dev(t * ldy));
                        gemm_tiled(&kn, &s, w, k, n, x, ldx, ya, ldy);
                        s.linear(&kn, w, k, n, x, ldx, yb, ldy);
                        same(&format!("gemv [{n} x {k}] t {t} (heads {})", d.heads), ya, yb, t * ldy);
                        for mut p in [w, x, ya, yb] {
                            cuda::free_dev(&mut p);
                        }
                    }
                    // absorb and out_v on one kv_b
                    let kvb = cuda::to_dev(&bf16s(d.heads * (d.nope + d.v) * d.kv_lora, &mut rng));
                    cuda::to_f32_into(s.q, &f32s(t * d.heads * d.nope, &mut rng));
                    cuda::to_f32_into(s.u, &f32s(t * d.heads * d.kv_lora, &mut rng));
                    let nq = t * d.heads * d.kv_lora;
                    let ref_qt = nan_dev(nq);
                    launch_v(kn.absorb, d.heads as u32, t.div_ceil(8) as u32, 1, 256, &[s.q, kvb, ref_qt, s.st]);
                    cuda::to_f32_into(s.qt, &vec![f32::NAN; nq]);
                    s.absorb(&kn, kvb);
                    same(&format!("absorb t {t} (heads {})", d.heads), ref_qt, s.qt, nq);
                    let no = t * d.heads * d.v;
                    let ref_o = nan_dev(no);
                    launch_v(kn.out_v, d.heads as u32, t.div_ceil(8) as u32, 1, 256, &[s.u, kvb, ref_o, s.st]);
                    cuda::to_f32_into(s.o, &vec![f32::NAN; no]);
                    s.out_v(&kn, kvb);
                    same(&format!("out_v t {t} (heads {})", d.heads), ref_o, s.o, no);
                    for mut p in [kvb, ref_qt, ref_o] {
                        cuda::free_dev(&mut p);
                    }
                }
                s.free();
            }
        }
    }

    /// (mean GPU time of one `f(i)` over `n` calls, events around the whole queue, us; mean host
    /// time to queue one, us)
    unsafe fn time2_us(n: usize, mut f: impl FnMut(usize)) -> (f64, f64) {
        f(0);
        cuda::sync();
        let mk = || {
            let mut e: sys::CUevent = std::ptr::null_mut();
            cuda::ck(sys::cuEventCreate(&mut e, 0));
            e
        };
        let (a, b) = (mk(), mk());
        let st = cuda::cur_stream();
        cuda::event_record(a, st);
        let t0 = std::time::Instant::now();
        for i in 0..n {
            f(i);
        }
        let host = t0.elapsed().as_secs_f64() * 1e6 / n as f64;
        cuda::event_record(b, st);
        cuda::sync();
        let mut ms = 0f32;
        cuda::ck(sys::cuEventElapsedTime_v2(&mut ms, a, b));
        cuda::event_destroy(a);
        cuda::event_destroy(b);
        (ms as f64 * 1e3 / n as f64, host)
    }

    /// acceptance bound, fixed before the after-measurement (ticket comment): the four decode
    /// launches (indexer x-side and q-side projections, absorb, out_v) of the 11 DSA layers of one
    /// decode row on synthetic VRAM-cold weights take at most this (us); before: ~8,600 us (Nsight)
    const MLA_ROW_US: f64 = 1000.0;

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mla_decode_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mla_decode_gpu_kernels_read_the_weights_at_vram_rate() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let g = Glm5Geo::GLM_5_3_FLASH;
            let d = MlaDims::of(&g);
            let (_m, kn) = kernels(d);
            let mut rng = Rng(0x6d1a_7173);
            let mut s = MlaScratch::new(&d, DECODE_T, 64);
            let layers = g.dsa_layers as f64;
            let (mut tiled_row, mut new_row, mut bytes_row) = (0f64, 0f64, 0f64);
            // (what, rows, cols): the two gm_gemm launches of a decode row, then kv_b (absorb, out_v)
            let kvb_n = d.heads * (d.nope + d.v) * d.kv_lora;
            let launches = [
                ("idx x-side", d.idx_proj(), d.hidden),
                ("idx q-side", d.idx_heads * d.idx_dim, d.q_lora),
                ("absorb kv_b", 0, 0),
                ("out_v kv_b", 0, 0),
            ];
            for (what, rows, cols) in launches {
                let kvb = rows == 0;
                let elems = if kvb { kvb_n } else { rows * cols };
                // absorb reads the W_k half of kv_b, out_v the W_v half
                let bytes = if kvb { elems } else { elems * 2 };
                let copies = (256usize << 20).div_ceil(elems * 2).max(2);
                let first = cuda::to_dev(&bf16s(elems, &mut rng));
                let mut ws = vec![first];
                for _ in 1..copies {
                    let p = cuda::alloc_zeroed(elems * 2);
                    cuda::memcpy_async(p, first, elems * 2);
                    ws.push(p);
                }
                let x = cuda::to_f32_dev(&f32s(DECODE_T * cols.max(1), &mut rng));
                let y = cuda::alloc_zeroed(DECODE_T * rows.max(1) * 4);
                cuda::to_f32_into(s.q, &f32s(DECODE_T * d.heads * d.nope, &mut rng));
                cuda::to_f32_into(s.u, &f32s(DECODE_T * d.heads * d.kv_lora, &mut rng));
                let n = (4 * copies).max(64);
                for t in 1..=DECODE_T {
                    s.begin(0, t);
                    let (tiled, new) = if kvb && what.starts_with("absorb") {
                        let a = time2_us(n, |i| launch_v(kn.absorb, d.heads as u32, t.div_ceil(8) as u32, 1, 256, &[s.q, ws[i % copies], s.qt, s.st])).0;
                        (a, time2_us(n, |i| s.absorb(&kn, ws[i % copies])).0)
                    } else if kvb {
                        let a = time2_us(n, |i| launch_v(kn.out_v, d.heads as u32, t.div_ceil(8) as u32, 1, 256, &[s.u, ws[i % copies], s.o, s.st])).0;
                        (a, time2_us(n, |i| s.out_v(&kn, ws[i % copies])).0)
                    } else {
                        let a = time2_us(n, |i| gemm_tiled(&kn, &s, ws[i % copies], cols, rows, x, cols, y, rows)).0;
                        (a, time2_us(n, |i| s.linear(&kn, ws[i % copies], cols, rows, x, cols, y, rows)).0)
                    };
                    let gbs = |us: f64| bytes as f64 / us / 1e3;
                    eprintln!("mla decode {what:<12} {bytes:>9} B t {t}: tiled {tiled:>7.1} us {:>5.0} GB/s | decode {new:>6.1} us {:>5.0} GB/s", gbs(tiled), gbs(new));
                    if t == 1 {
                        tiled_row += tiled * layers;
                        new_row += new * layers;
                        bytes_row += bytes as f64 * layers;
                    }
                }
                for mut p in ws.into_iter().chain([x, y]) {
                    cuda::free_dev(&mut p);
                }
            }
            s.free();
            eprintln!(
                "mla decode launches per decode row ({} DSA layers): {:.0} MB; tiled {:.3} ms ({:.0} GB/s), decode {:.3} ms ({:.0} GB/s)",
                g.dsa_layers,
                bytes_row / 1e6,
                tiled_row / 1e3,
                bytes_row / tiled_row / 1e3,
                new_row / 1e3,
                bytes_row / new_row / 1e3
            );
            assert!(new_row <= MLA_ROW_US, "the MLA decode launches take {:.3} ms per decode row (bound {:.3} ms)", new_row / 1e3, MLA_ROW_US / 1e3);
        }
    }
}
