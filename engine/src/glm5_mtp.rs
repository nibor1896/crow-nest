//! crow-nest #182 (GLM-5.3-Flash open point O1): the glm5_next MTP (NextN) block, checkpoint layer
//! 45, as one runnable step on the GPU, built from the trunk's blocks.
//!
//! HF has no MTP forward (it drops layer 45 on load); the formula of record is the one vLLM, SGLang
//! and llama.cpp run for this model (`docs/glm5-mtp.md` "Sources", reference
//! `oracle/glm5_mtp.py`). Per draft row `i` (trunk row `i` paired with the next token `t_{i+1}`):
//!
//! ```text
//! eh   = [ enorm(embed(t_{i+1})) | hnorm(h_i) ]       h_i = the trunk's lm_head input (post final norm)
//! x    = eh_proj · eh                                 eh_proj BF16 [H][2H], e columns first
//! y    = x + MLA_DSA(input_layernorm(x))              glm5_mla, the block's own cache and full indexer
//! z    = y + MoE(post_attention_layernorm(y))         glm5_moe: router + 288 MUL1 experts + shared
//! n    = shared_head.norm(z)                          the draft's lm_head input and the next h
//! draft = argmax(n · lm_head^T)                       the TRUNK's lm_head (glm5_head::Head::lm_head)
//! ```
//!
//! One stream, no mHC (the block has no `hc_*` tensors). Every RMSNorm is HF's
//! `w * (x * rsqrt(mean(x^2) + 1e-5))`. Row `i` sits at position `i` of the block's cache (the
//! vLLM/SGLang shift); [`Pos0`] names how the stacks differ at row 0.
//!
//! What this module adds besides the order: the fused `mtp_eh_norm` kernel (zero-at-position-0 +
//! two RMSNorms + concat, vLLM's `fused_eh_norm`), the residual add, the pairing rule
//! ([`pair_rows`]), the tensor plan of layer 45 ([`mtp_tensors`]: which tensors the 3-bit container
//! holds - only the 288 MUL1 experts - and what an overlay would cost), the record loader of the
//! container's `mtp` section ([`load_mtp_records`]), and the whole block from the container plus
//! the MTP overlay of `converter --mtp-overlay` ([`load_mtp`], checked by [`mtp_overlay_check`];
//! [`MtpBlock::call`] runs the NVFP4 projections). Nothing in the engine calls it yet: the decode
//! loop's speculative step (`gen.rs` `spec_step`, Flash-Next/27B) and a glm5 caller are the lead's
//! wiring; `docs/glm5-mtp.md` "Integration" lists the hooks.

use crate::cnq::{Cnq, ModelBlock, TensorInfo};
use crate::cuda;
use crate::geo::Glm5Geo;
use crate::glm5_mla::{MlaCache, MlaDims, MlaProj, MlaScratch, MlaWeights, RMS_EPS};
use crate::glm5_model::{expert_names, Glm5Kernels};
use crate::glm5_moe::{GpuFfnWeights, GpuMoePlan, GpuMoeWeights, GpuNvfp4, MoeGeo};
use crate::kernels::launch_v;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

pub type Dev = CUdeviceptr;

/// the container section of the block's routed-expert records (`converter/src/recipe.rs`
/// `mul1_expert_decision`)
pub const MTP_SECTION: &str = "mtp";

/// the checkpoint layer of the (single) MTP block: `num_hidden_layers`
pub fn mtp_layer(g: &Glm5Geo) -> usize {
    assert_eq!(g.mtp_layers, 1, "glm5_mtp: {} MTP blocks; the module runs one (num_nextn_predict_layers 1)", g.mtp_layers);
    g.layers
}

pub fn mtp_prefix(g: &Glm5Geo) -> String {
    format!("model.language_model.layers.{}.", mtp_layer(g))
}

// ---------------------------------------------------------------- the tensor plan of layer 45

/// how a tensor of the block would be stored in an overlay that follows the `cnq4.5-glm5-next`
/// decisions of the trunk's DSA + MoE layers (`docs/glm5-mtp.md` "Weights")
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Store {
    Bf16,
    F32,
    Nvfp4,
    /// the routed experts: MUL1 K=3 records, already in the 3-bit container (section `mtp`)
    Mul1,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MtpTensor {
    /// relative to [`mtp_prefix`]
    pub name: String,
    pub shape: Vec<usize>,
    /// E4M3 + `weight_scale_inv` in the FP8 originals
    pub ckpt_fp8: bool,
    pub store: Store,
}

impl MtpTensor {
    pub fn values(&self) -> usize {
        self.shape.iter().product()
    }
    /// bytes in the overlay (NVFP4: the engine layout, 36 B per 64 values; MUL1: not in the overlay)
    pub fn overlay_bytes(&self) -> u64 {
        match self.store {
            Store::Bf16 => self.values() as u64 * 2,
            Store::F32 => self.values() as u64 * 4,
            Store::Nvfp4 => crate::cpu_nvfp4::Nvfp4Matrix::byte_len(self.shape[0], self.shape[1..].iter().product()) as u64 + 4,
            Store::Mul1 => 0,
        }
    }
}

/// Every tensor of the block (25 + 3 x 288), names relative to [`mtp_prefix`], shapes from the
/// family row; the FP8 split is the checkpoint index's (rev eb9eb208: q_a, q_b, kv_a, o_proj and the
/// shared expert carry a `weight_scale_inv`, `test_glm5_mtp.Names` checks the names against it).
pub fn mtp_tensors(g: &Glm5Geo) -> Vec<MtpTensor> {
    use Store::*;
    let d = MlaDims::of(g);
    let (h, si) = (g.hidden, g.expert_inter * g.shared_experts);
    let t = |n: &str, s: &[usize], fp8: bool, st: Store| MtpTensor { name: n.into(), shape: s.to_vec(), ckpt_fp8: fp8, store: st };
    let mut v = vec![
        t("enorm.weight", &[h], false, Bf16),
        t("hnorm.weight", &[h], false, Bf16),
        t("eh_proj.weight", &[h, 2 * h], false, Bf16),
        t("input_layernorm.weight", &[h], false, Bf16),
        t("post_attention_layernorm.weight", &[h], false, Bf16),
        t("shared_head.norm.weight", &[h], false, Bf16),
        t("self_attn.q_a_proj.weight", &[d.q_lora, h], true, Nvfp4),
        t("self_attn.q_a_layernorm.weight", &[d.q_lora], false, Bf16),
        t("self_attn.q_b_proj.weight", &[d.heads * d.nope, d.q_lora], true, Nvfp4),
        t("self_attn.kv_a_proj_with_mqa.weight", &[d.kv_lora, h], true, Nvfp4),
        t("self_attn.kv_a_layernorm.weight", &[d.kv_lora], false, Bf16),
        t("self_attn.kv_b_proj.weight", &[d.heads * (d.nope + d.v), d.kv_lora], false, Nvfp4),
        t("self_attn.o_proj.weight", &[h, d.heads * d.v], true, Nvfp4),
        t("self_attn.indexer.wq_b.weight", &[d.idx_heads * d.idx_dim, d.q_lora], false, Bf16),
        t("self_attn.indexer.wk.weight", &[d.idx_dim, h], false, Bf16),
        t("self_attn.indexer.k_norm.weight", &[d.idx_dim], false, Bf16),
        t("self_attn.indexer.k_norm.bias", &[d.idx_dim], false, Bf16),
        t("self_attn.indexer.weights_proj.weight", &[d.idx_heads, h], false, Bf16),
        t("self_attn.indexer.index_kpool_compress_gate", &[d.idx_dim, h], false, Bf16),
        t("self_attn.indexer.index_kpool_compress_ape", &[d.kpool, d.idx_dim], false, Bf16),
        t("mlp.gate.weight", &[g.experts, h], false, Bf16),
        t("mlp.gate.e_score_correction_bias", &[g.experts], false, F32),
        t("mlp.shared_experts.gate_proj.weight", &[si, h], true, Nvfp4),
        t("mlp.shared_experts.up_proj.weight", &[si, h], true, Nvfp4),
        t("mlp.shared_experts.down_proj.weight", &[h, si], true, Nvfp4),
    ];
    for e in 0..g.experts {
        for (p, s) in [("gate", [g.expert_inter, h]), ("up", [g.expert_inter, h]), ("down", [h, g.expert_inter])] {
            v.push(t(&format!("mlp.experts.{e}.{p}_proj.weight"), &s, true, Mul1));
        }
    }
    v
}

/// The block's tensors a container lacks (full names). On the 3-bit container this is the 25
/// non-expert tensors: the `cnq4.5-glm5-next` row omits layer 45 and `--experts-mul1` adds only
/// its experts (`docs/glm-mul1-conversion.md` "Container").
pub fn missing_in(cnq: &Cnq, g: &Glm5Geo) -> Vec<String> {
    let p = mtp_prefix(g);
    mtp_tensors(g).into_iter().map(|t| format!("{p}{}", t.name)).filter(|n| !cnq.tensors.iter().any(|t| &t.name == n)).collect()
}

/// (bytes of an overlay with the trunk's codecs, bytes of an all-BF16 overlay): the 25 non-expert
/// tensors only, the experts stay the container's MUL1 records
pub fn overlay_bytes(g: &Glm5Geo) -> (u64, u64) {
    let ts: Vec<MtpTensor> = mtp_tensors(g).into_iter().filter(|t| t.store != Store::Mul1).collect();
    let mixed = ts.iter().map(MtpTensor::overlay_bytes).sum();
    let bf16 = ts.iter().map(|t| t.values() as u64 * if t.store == Store::F32 { 4 } else { 2 }).sum();
    (mixed, bf16)
}

// ---------------------------------------------------------------- the pairing

/// What row 0 sees; the stacks differ only there (`docs/glm5-mtp.md` "Differences").
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Pos0 {
    /// SGLang `deepseek_nextn.py`: row 0 = (t_1, h_0), nothing masked (the reference's primary)
    Keep,
    /// vLLM `deepseek_mtp.py` / `fused_eh_norm`: the embedding of the row at position 0 is zeroed
    ZeroEmbed,
    /// llama.cpp `speculative.cpp`: a leading row (t_0, h = 0) at position 0, then (t_{i+1}, h_i) at i+1
    LeadRow,
}

/// One row of the block's input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DraftRow {
    /// index into the ids of the embedded token
    pub token: usize,
    /// the trunk row whose head-norm state feeds hnorm (None: zeros)
    pub h_row: Option<usize>,
    /// position in the block's cache
    pub pos: usize,
    pub zero_embed: bool,
    /// the index of the id this row drafts (None: past the end, or the lead row)
    pub drafts: Option<usize>,
}

/// The block's rows for a sequence of `n` ids (trunk rows 0..n-1 computed).
pub fn pair_rows(n: usize, pos0: Pos0) -> Vec<DraftRow> {
    assert!(n >= 2, "glm5_mtp: {n} ids, the first draft row needs two");
    let lead = (pos0 == Pos0::LeadRow) as usize;
    let mut v = Vec::with_capacity(n - 1 + lead);
    if lead == 1 {
        v.push(DraftRow { token: 0, h_row: None, pos: 0, zero_embed: false, drafts: None });
    }
    for i in 0..n - 1 {
        let pos = i + lead;
        v.push(DraftRow { token: i + 1, h_row: Some(i), pos, zero_embed: pos0 == Pos0::ZeroEmbed && pos == 0, drafts: (i + 2 < n).then_some(i + 2) });
    }
    v
}

/// draft rows whose pick equals the trunk's own pick for the same id: (agree, rows). `trunk_next[i]`
/// is the trunk's argmax at row i+1 (it guesses the same id as draft row i).
pub fn agreement(draft: &[u32], trunk_next: &[u32]) -> (usize, usize) {
    let n = draft.len().min(trunk_next.len());
    ((0..n).filter(|&i| draft[i] == trunk_next[i]).count(), n)
}

// ---------------------------------------------------------------- host twin (f64)

/// `w * (x * rsqrt(mean(x^2) + eps))`
pub fn rms_ref(x: &[f64], w: &[f32], eps: f64) -> Vec<f64> {
    let r = 1.0 / (x.iter().map(|v| v * v).sum::<f64>() / x.len() as f64 + eps).sqrt();
    x.iter().zip(w).map(|(v, &g)| g as f64 * (v * r)).collect()
}

/// the eh_proj input of one row: `[enorm(e) | hnorm(h)]`, `e` read as zeros when `zero_embed`
pub fn eh_norm_ref(e: &[f32], h: &[f32], we: &[f32], wh: &[f32], zero_embed: bool) -> Vec<f64> {
    let eps = RMS_EPS as f64;
    let e64: Vec<f64> = e.iter().map(|&v| if zero_embed { 0.0 } else { v as f64 }).collect();
    let h64: Vec<f64> = h.iter().map(|&v| v as f64).collect();
    let mut out = rms_ref(&e64, we, eps);
    out.extend(rms_ref(&h64, wh, eps));
    out
}

/// `eh_proj · x` for one row, eh_proj BF16 `[H][2H]` row-major
pub fn eh_proj_ref(w: &[u16], x: &[f64]) -> Vec<f64> {
    let k = x.len();
    assert_eq!(w.len() % k, 0, "glm5_mtp: eh_proj is not [H][{k}]");
    w.chunks_exact(k).map(|row| row.iter().zip(x).map(|(&b, &v)| crate::glm5_head::bf16_to_f32(b) as f64 * v).sum()).collect()
}

// ---------------------------------------------------------------- GPU

/// the block's glue kernels (its own NVRTC module, one per process)
pub const MTP_SRC: &str = r#"
// crow-nest #182: glm5_next MTP glue (glm5_mtp.rs)
__device__ __forceinline__ float mtp_block_sum(float v, float* red) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    __syncthreads();
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    float s = 0.0f;
    for (int i = 0; i < (int)((blockDim.x + 31) >> 5); i++) s += red[i];
    return s;
}

// out[r][0..n] = we * rms(e[r]) (zeros when zero_pos0 and the row sits at position 0),
// out[r][n..2n] = wh * rms(h[r]). grid t, block 256; st = {pos0, t}.
extern "C" __global__ void mtp_eh_norm(const float* __restrict__ e, const float* __restrict__ h,
                                       const float* __restrict__ we, const float* __restrict__ wh,
                                       float* __restrict__ out, long long n, float eps, long long zero_pos0,
                                       const int* __restrict__ st) {
    __shared__ float red[32];
    int r = blockIdx.x;
    const float* ep = e + (long long)r * n;
    const float* hp = h + (long long)r * n;
    float* op = out + (long long)r * 2 * n;
    bool z = zero_pos0 != 0 && st[0] + r == 0;
    float se = 0.0f, sh = 0.0f;
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        if (!z) se += ep[i] * ep[i];
        sh += hp[i] * hp[i];
    }
    float re = rsqrtf(mtp_block_sum(se, red) / (float)n + eps);
    float rh = rsqrtf(mtp_block_sum(sh, red) / (float)n + eps);
    for (long long i = threadIdx.x; i < n; i += blockDim.x) {
        op[i] = z ? 0.0f : we[i] * (ep[i] * re);
        op[n + i] = wh[i] * (hp[i] * rh);
    }
}

// y[i] += a[i], i < n
extern "C" __global__ void mtp_add(const float* __restrict__ a, float* __restrict__ y, long long n) {
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[i];
}
"#;

pub const NAMES: &[&str] = &["mtp_eh_norm", "mtp_add"];

pub struct MtpKernels {
    pub module: cuda::Module,
    eh_norm: CUfunction,
    add: CUfunction,
}

impl MtpKernels {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new() -> MtpKernels {
        let module = cuda::compile(MTP_SRC);
        MtpKernels { eh_norm: module.get("mtp_eh_norm"), add: module.get("mtp_add"), module }
    }
}

/// Device weights of the block. Vectors f32, `eh_proj` BF16 `[H][2H]`; `attn` as `glm5_mla` reads it
/// (projections 0 when a codec hook runs them, as `glm5_model::MlaLayer`); `records` the 288 MUL1
/// records back to back behind the `[E]` u64 `table` (`GpuMoePlan::run`).
pub struct MtpWeights {
    pub enorm: Dev,
    pub hnorm: Dev,
    pub eh_proj: Dev,
    pub input_norm: Dev,
    pub post_norm: Dev,
    pub head_norm: Dev,
    pub attn: MlaWeights,
    pub moe: GpuMoeWeights,
    pub records: Dev,
    pub table: Dev,
}

impl MtpWeights {
    /// frees every pointer it holds (0 entries are skipped); NVFP4 projections a hook owns are the
    /// caller's
    ///
    /// # Safety
    /// No launch reading these weights is pending.
    pub unsafe fn free(&mut self) {
        let a = &mut self.attn;
        for p in [
            &mut self.enorm, &mut self.hnorm, &mut self.eh_proj, &mut self.input_norm, &mut self.post_norm, &mut self.head_norm,
            &mut a.q_a, &mut a.q_a_norm, &mut a.q_b, &mut a.kv_a, &mut a.kv_a_norm, &mut a.kv_b, &mut a.o_proj, &mut a.idx_wq_b,
            &mut a.idx_x, &mut a.idx_k_norm_w, &mut a.idx_k_norm_b, &mut a.idx_ape, &mut self.moe.router, &mut self.moe.bias,
            &mut self.records, &mut self.table,
        ] {
            cuda::free_dev(p);
        }
        for m in [&mut self.moe.shared.gate, &mut self.moe.shared.up, &mut self.moe.shared.down] {
            cuda::free_dev(&mut m.w);
            cuda::free_dev(&mut m.gs);
        }
    }
}

/// The block's 288 routed-expert records from the container's `mtp` section into one VRAM buffer
/// and the `[E]` table of their bases - the half of the block the 3-bit container holds.
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn load_mtp_records(cnq: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo) -> (Dev, Dev) {
    let (l, rb) = (mtp_layer(g), moe.record.bytes);
    let records = cuda::alloc_named("glm5 MTP expert records", (g.experts as u64 * rb) as usize);
    let mut bases = Vec::with_capacity(g.experts);
    for e in 0..g.experts {
        let [gate, _, _] = expert_names(l, e);
        let t = cnq.find(&gate, MTP_SECTION).clone();
        assert_eq!(t.dtype, "mul1", "glm5_mtp: {gate} is {}, the record table takes MUL1 records (#181)", t.dtype);
        let rec = cnq.read_range(&t, 0, rb as usize);
        let base = records + e as u64 * rb;
        cuda::into_dev(base, rec.as_slice());
        bases.push(base);
    }
    (records, cuda::to_u64_dev(&bases))
}

// ---------------------------------------------------------------- the MTP overlay

/// the index dtype of an overlay tensor stored as `s` (the converter's `--mtp-overlay` row)
pub fn overlay_dtype(s: Store) -> &'static str {
    match s {
        Store::Bf16 => "bf16",
        Store::F32 => "f32",
        Store::Nvfp4 => "nvfp4",
        Store::Mul1 => "mul1",
    }
}

/// #182: whether an MTP overlay (`converter --mtp-overlay`, `docs/glm-mul1-conversion.md` "MTP
/// overlay") belongs to this base container, before a byte of either is read. The overlay holds
/// exactly the block's 25 non-expert tensors of [`mtp_tensors`], in section `mtp`, each with its
/// planned dtype and shape; the base holds the block's 288 MUL1 records (section `mtp`); both were
/// converted from the same checkpoint (family, source repo and revision, `config.json` byte for
/// byte). `Err` names every misfit.
pub fn mtp_overlay_check(
    g: &Glm5Geo,
    base: &[TensorInfo],
    base_model: Option<&ModelBlock>,
    ov: &[TensorInfo],
    ov_model: Option<&ModelBlock>,
) -> Result<(), String> {
    let mut bad = Vec::new();
    match (base_model, ov_model) {
        (Some(b), Some(o)) => {
            for (what, x, y) in [
                ("family", &b.family, &o.family),
                ("source repo", &b.source_repo, &o.source_repo),
                ("source revision", &b.source_revision, &o.source_revision),
                ("recipe", &b.recipe, &o.recipe),
            ] {
                if x != y {
                    bad.push(format!("{what}: `{y}` in the overlay, `{x}` in the base container"));
                }
            }
            if b.config_json != o.config_json {
                bad.push("config.json differs between the overlay and the base container".into());
            }
        }
        _ => bad.push("both files must carry an index v2 `model` block (the converter of #300 C6 writes one)".into()),
    }
    let p = mtp_prefix(g);
    let plan = mtp_tensors(g);
    let mut want = 0;
    for t in plan.iter().filter(|t| t.store != Store::Mul1) {
        want += 1;
        let name = format!("{p}{}", t.name);
        match ov.iter().find(|o| o.name == name) {
            None => bad.push(format!("{name}: not in the overlay")),
            Some(o) => {
                let shape: Vec<u64> = t.shape.iter().map(|&v| v as u64).collect();
                if (o.section.as_str(), o.dtype.as_str(), &o.shape) != (MTP_SECTION, overlay_dtype(t.store), &shape) {
                    bad.push(format!("{name}: [{}] {} {:?} in the overlay, the plan says [{MTP_SECTION}] {} {shape:?}", o.section, o.dtype, o.shape, overlay_dtype(t.store)));
                }
            }
        }
    }
    if ov.len() != want {
        let extra: Vec<&str> = ov.iter().filter(|o| !plan.iter().any(|t| t.store != Store::Mul1 && format!("{p}{}", t.name) == o.name)).map(|o| o.name.as_str()).collect();
        bad.push(format!("the overlay holds {} tensors, the block has {want} non-expert ones (not of the block: {extra:?})", ov.len()));
    }
    let l = mtp_layer(g);
    let records = (0..g.experts).filter(|&e| base.iter().any(|b| b.name == expert_names(l, e)[0] && b.section == MTP_SECTION && b.dtype == "mul1")).count();
    if records != g.experts {
        bad.push(format!("the base container holds {records} of the block's {} MUL1 records (section {MTP_SECTION}; converter --experts-mul1)", g.experts));
    }
    if bad.is_empty() {
        Ok(())
    } else {
        Err(format!("refusing the MTP overlay (#182): {} misfit(s)\n  {}", bad.len(), bad.join("\n  ")))
    }
}

/// The whole MTP block on the device: [`MtpWeights`] (vectors, eh_proj, the BF16 indexer and
/// `kv_b`, router, shared expert, the 288 records) and the four NVFP4 MLA projections the codec
/// hook ([`MtpBlock::call`]) runs on `glm5_gemv_fp4` (#191), as the trunk's `glm5_model::MlaLayer` does.
pub struct MtpBlock {
    /// `attn.q_a`, `attn.q_b`, `attn.kv_a`, `attn.o_proj` are 0: the hook runs them
    pub w: MtpWeights,
    pub q_a: GpuNvfp4,
    pub q_b: GpuNvfp4,
    pub kv_a: GpuNvfp4,
    pub o: GpuNvfp4,
    /// device i32 `[8]`: the `cols`, then the `rows` of q_a, q_b, kv_a, o (the GEMV reads its k,
    /// output stride and row count from the device)
    cols: Dev,
    /// bytes uploaded; scale bytes 0x7F rewritten; `kv_b` values BF16 does not hold exactly
    pub bytes: u64,
    pub sanitized: u64,
    pub kv_b_inexact: u64,
}

/// the overlay reads of [`load_mtp`], every tensor checked by [`mtp_overlay_check`] first
struct OvLoader<'a> {
    cnq: &'a mut Cnq,
    prefix: String,
    bytes: u64,
    sanitized: u64,
}

impl OvLoader<'_> {
    fn raw(&mut self, n: &str) -> (TensorInfo, Vec<u8>) {
        let t = self.cnq.find(&format!("{}{n}", self.prefix), MTP_SECTION).clone();
        let raw = self.cnq.read_bytes(&t);
        (t, raw)
    }
    fn f32_host(&mut self, n: &str) -> Vec<f32> {
        let (t, raw) = self.raw(n);
        match t.dtype.as_str() {
            "f32" => crate::cnq::f32_bytes_to_f32(&raw),
            _ => crate::cnq::bf16_bytes_to_f32(&raw),
        }
    }
    unsafe fn f32(&mut self, n: &str) -> Dev {
        let v = self.f32_host(n);
        self.bytes += v.len() as u64 * 4;
        cuda::to_f32_dev(&v)
    }
    unsafe fn bf16(&mut self, n: &str) -> Dev {
        let (_, raw) = self.raw(n);
        self.bytes += raw.len() as u64;
        cuda::upload_dev(&raw)
    }
    fn fp4_host(&mut self, n: &str) -> (TensorInfo, Vec<u8>) {
        let (t, mut raw) = self.raw(n);
        self.sanitized += crate::residency::sanitize_sf_slab(&mut raw);
        (t, raw)
    }
    unsafe fn fp4(&mut self, n: &str) -> GpuNvfp4 {
        let (t, raw) = self.fp4_host(n);
        self.bytes += raw.len() as u64 + 4;
        GpuNvfp4 { w: cuda::upload_dev(&raw), gs: cuda::to_f32_dev(&[t.global_scale]), rows: t.shape[0] as usize, cols: t.shape[1..].iter().product::<u64>() as usize }
    }
}

/// #182: the whole MTP block from the 3-bit container (`base`: the 288 MUL1 records of section
/// `mtp`) and the MTP overlay (`overlay`: the 25 other tensors, `converter --mtp-overlay`), after
/// [`mtp_overlay_check`]. Takes as the trunk's DSA + MoE loader does (`glm5_model::load_layer`):
/// norms widened to f32, BF16 matrices as stored, `kv_b` NVFP4 decoded once to BF16, the indexer's
/// `wk | index_kpool_compress_gate | weights_proj` stacked, NVFP4 scale bytes 0x7F -> 0x7E (#177).
///
/// # Safety
/// A CUDA context is current.
pub unsafe fn load_mtp(base: &mut Cnq, overlay: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo) -> Result<MtpBlock, String> {
    mtp_overlay_check(g, &base.tensors, base.model(), &overlay.tensors, overlay.model())?;
    let mut ld = OvLoader { cnq: overlay, prefix: mtp_prefix(g), bytes: 0, sanitized: 0 };
    let mut idx_x = ld.raw("self_attn.indexer.wk.weight").1;
    idx_x.extend(ld.raw("self_attn.indexer.index_kpool_compress_gate").1);
    idx_x.extend(ld.raw("self_attn.indexer.weights_proj.weight").1);
    ld.bytes += idx_x.len() as u64;
    let (kvb_t, kvb_raw) = ld.fp4_host("self_attn.kv_b_proj.weight");
    let (kv_b, kv_b_inexact) = crate::glm5_model::nvfp4_to_bf16(&kvb_raw, kvb_t.global_scale, kvb_t.n_values as usize);
    ld.bytes += kv_b.len() as u64 * 2;
    let attn = MlaWeights {
        q_a: 0,
        q_a_norm: ld.f32("self_attn.q_a_layernorm.weight"),
        q_b: 0,
        kv_a: 0,
        kv_a_norm: ld.f32("self_attn.kv_a_layernorm.weight"),
        kv_b: cuda::to_dev(&kv_b),
        o_proj: 0,
        idx_wq_b: ld.bf16("self_attn.indexer.wq_b.weight"),
        idx_x: cuda::upload_dev(&idx_x),
        idx_k_norm_w: ld.f32("self_attn.indexer.k_norm.weight"),
        idx_k_norm_b: ld.f32("self_attn.indexer.k_norm.bias"),
        idx_ape: ld.f32("self_attn.indexer.index_kpool_compress_ape"),
    };
    let moe_w = GpuMoeWeights {
        router: ld.bf16("mlp.gate.weight"),
        bias: ld.f32("mlp.gate.e_score_correction_bias"),
        shared: GpuFfnWeights {
            gate: ld.fp4("mlp.shared_experts.gate_proj.weight"),
            up: ld.fp4("mlp.shared_experts.up_proj.weight"),
            down: ld.fp4("mlp.shared_experts.down_proj.weight"),
        },
    };
    let (q_a, q_b, kv_a, o) = (ld.fp4("self_attn.q_a_proj.weight"), ld.fp4("self_attn.q_b_proj.weight"), ld.fp4("self_attn.kv_a_proj_with_mqa.weight"), ld.fp4("self_attn.o_proj.weight"));
    let (enorm, hnorm, eh_proj) = (ld.f32("enorm.weight"), ld.f32("hnorm.weight"), ld.bf16("eh_proj.weight"));
    let (input_norm, post_norm, head_norm) = (ld.f32("input_layernorm.weight"), ld.f32("post_attention_layernorm.weight"), ld.f32("shared_head.norm.weight"));
    let (bytes, sanitized) = (ld.bytes, ld.sanitized);
    let (records, table) = load_mtp_records(base, g, moe);
    let i = |v: usize| i32::try_from(v).expect("glm5_mtp: GEMV k beyond i32");
    let cols = cuda::to_i32_dev(&[i(q_a.cols), i(q_b.cols), i(kv_a.cols), i(o.cols), i(q_a.rows), i(q_b.rows), i(kv_a.rows), i(o.rows)]);
    Ok(MtpBlock {
        w: MtpWeights { enorm, hnorm, eh_proj, input_norm, post_norm, head_norm, attn, moe: moe_w, records, table },
        q_a,
        q_b,
        kv_a,
        o,
        cols,
        bytes: bytes + g.experts as u64 * moe.record.bytes,
        sanitized,
        kv_b_inexact,
    })
}

impl MtpBlock {
    /// One [`MtpPass::call`] with the block's NVFP4 projections on `glm5_gemv_fp4` (#191,
    /// bit-identical to the record `gemv_fp4_b`; the trunk's MLA codec hook, `glm5_model`
    /// `fp4_gemv`); `taps` as [`MtpPass::call_tapped`].
    ///
    /// # Safety
    /// As [`MtpPass::call`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn call(&self, pass: &mut MtpPass, kn: &Glm5Kernels, mk: &MtpKernels, e: Dev, h: Dev, pos0: usize, t: usize, zero_pos0: bool, taps: Option<&mut MtpTaps>) {
        let mut proj = |s: &MlaScratch, p: MlaProj, xi: Dev, yo: Dev| {
            let (m, k) = match p {
                MlaProj::QA => (&self.q_a, 0u64),
                MlaProj::QB => (&self.q_b, 1),
                MlaProj::KVA => (&self.kv_a, 2),
                MlaProj::O => (&self.o, 3),
            };
            // k from the device, output stride = row count (dense rows), row count
            let (blocks, threads) = crate::kernels::glm5_moe::fp4_launch(m.rows, m.cols);
            let rows = self.cols + 4 * (4 + k);
            launch_v(kn.moe.fp4, blocks, s.t() as u32, 1, threads, &[m.w, xi, m.gs, yo, self.cols + 4 * k, rows, rows]);
        };
        pass.call_tapped(kn, mk, &self.w, e, h, pos0, t, zero_pos0, Some(&mut proj), taps);
    }

    /// # Safety
    /// No launch reading these weights is pending.
    pub unsafe fn free(&mut self) {
        self.w.free();
        for m in [&mut self.q_a, &mut self.q_b, &mut self.kv_a, &mut self.o] {
            cuda::free_dev(&mut m.w);
            cuda::free_dev(&mut m.gs);
        }
        cuda::free_dev(&mut self.cols);
    }
}

/// The block's intermediate rows of tapped calls, appended per call (`[rows][H]` each): the
/// oracle golden's `mtp-eh`, `mtp-attn-out`, `mtp-ffn-in`, `mtp-ffn-out`, `mtp-out`, and the DSA
/// selection per row (ascending positions).
#[derive(Default)]
pub struct MtpTaps {
    pub eh: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_in: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub out: Vec<f32>,
    pub selection: Vec<Vec<usize>>,
}

/// The block's per-sequence state and scratch: its own MLA latent + indexer cache (one more DSA
/// layer, `docs/glm5-next-recipe.md` section 13) and the buffers of one call of up to `max_t` rows.
pub struct MtpPass {
    pub g: Glm5Geo,
    pub moe: MoeGeo,
    pub max_t: usize,
    pub cap: usize,
    pub mla_sc: MlaScratch,
    pub mla_c: MlaCache,
    moe_plans: Vec<GpuMoePlan>,
    last_t: usize,
    st: Dev,
    /// `[t][2H]` the eh_proj input
    pub eh: Dev,
    /// `[t][H]` the residual stream (x, then y, then z)
    pub x: Dev,
    xn: Dev,
    /// `[t][H]` the sub-layer output (attention, then MoE)
    pub sub: Dev,
    /// `[t][H]` `shared_head.norm(z)`: the lm_head input and the next chained step's h
    pub normed: Dev,
}

impl MtpPass {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo, moe: MoeGeo, max_t: usize, cap: usize) -> MtpPass {
        assert!(max_t >= 1 && cap >= max_t, "glm5_mtp: calls of {max_t} rows over {cap}");
        let (md, h) = (MlaDims::of(g), g.hidden);
        let a = |what: &str, n: usize| cuda::alloc_named(what, n * 4);
        MtpPass {
            g: *g,
            moe,
            max_t,
            cap,
            mla_sc: MlaScratch::new(&md, max_t, cap),
            mla_c: MlaCache::new(&md, cap),
            moe_plans: Vec::new(),
            last_t: 0,
            st: cuda::to_i32_dev(&[0i32, 0]),
            eh: a("glm5 MTP eh", max_t * 2 * h),
            x: a("glm5 MTP residual", max_t * h),
            xn: a("glm5 MTP normed in", max_t * h),
            sub: a("glm5 MTP sublayer out", max_t * h),
            normed: a("glm5 MTP head norm", max_t * h),
        }
    }

    /// One call of `t` rows at cache positions `pos0 .. pos0 + t`: `e` `[t][H]` the embeddings of the
    /// rows' tokens, `h` `[t][H]` the trunk's head-norm rows (or the previous step's `normed`).
    /// `zero_pos0`: [`Pos0::ZeroEmbed`]. `proj`: the MLA codec hook (`MlaScratch::forward_with`; None
    /// = BF16 projections on `gm_gemm`). Result in [`MtpPass::normed`]. Queued on the current stream.
    ///
    /// # Safety
    /// `e`, `h` hold `t` rows; the calls come in position order from 0 (the block's cache).
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn call(
        &mut self,
        kn: &Glm5Kernels,
        mk: &MtpKernels,
        w: &MtpWeights,
        e: Dev,
        h: Dev,
        pos0: usize,
        t: usize,
        zero_pos0: bool,
        proj: Option<&mut dyn FnMut(&MlaScratch, MlaProj, Dev, Dev)>,
    ) {
        self.call_tapped(kn, mk, w, e, h, pos0, t, zero_pos0, proj, None);
    }

    /// [`MtpPass::call`], and with `taps` the call's intermediate rows appended to them (each tap
    /// synchronizes; `None` queues exactly what `call` queues).
    ///
    /// # Safety
    /// As [`MtpPass::call`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn call_tapped(
        &mut self,
        kn: &Glm5Kernels,
        mk: &MtpKernels,
        w: &MtpWeights,
        e: Dev,
        h: Dev,
        pos0: usize,
        t: usize,
        zero_pos0: bool,
        proj: Option<&mut dyn FnMut(&MlaScratch, MlaProj, Dev, Dev)>,
        mut taps: Option<&mut MtpTaps>,
    ) {
        assert!((1..=self.max_t).contains(&t) && pos0 + t <= self.cap, "glm5_mtp: call rows {pos0}..{} (max_t {}, cap {})", pos0 + t, self.max_t, self.cap);
        let hd = self.g.hidden;
        let bytes = t * hd * 4;
        let tap = |taps: &mut Option<&mut MtpTaps>, pick: fn(&mut MtpTaps) -> &mut Vec<f32>, src: Dev| {
            if let Some(tp) = taps.as_deref_mut() {
                cuda::sync();
                pick(tp).extend(cuda::dtoh(src, t * hd));
            }
        };
        cuda::to_i32_into(self.st, &[pos0 as i32, t as i32]);
        launch_v(mk.eh_norm, t as u32, 1, 1, 256, &[e, h, w.enorm, w.hnorm, self.eh, hd as u64, (self.g.rms_eps as f32).to_bits() as u64, zero_pos0 as u64, self.st]);
        self.mla_sc.begin(pos0, t);
        self.mla_sc.linear(&kn.mla, w.eh_proj, 2 * hd, hd, self.eh, 2 * hd, self.x, hd);
        tap(&mut taps, |tp| &mut tp.eh, self.x);
        cuda::d2d_async(self.xn, self.x, bytes);
        kn.mla.rmsnorm_rows(self.xn, w.input_norm, hd, t, self.st);
        match proj {
            Some(p) => self.mla_sc.forward_with(&kn.mla, &w.attn, &self.mla_c, self.xn, self.sub, pos0, t, p),
            None => self.mla_sc.forward(&kn.mla, &w.attn, &self.mla_c, self.xn, self.sub, pos0, t),
        }
        tap(&mut taps, |tp| &mut tp.attn_out, self.sub);
        if let Some(tp) = taps.as_deref_mut() {
            let sw = MlaDims::of(&self.g).sel_max();
            let n = cuda::dtoh_i32(self.mla_sc.sel_n, t);
            let s = cuda::dtoh_i32(self.mla_sc.sel, t * sw);
            for r in 0..t {
                let mut v: Vec<usize> = s[r * sw..r * sw + n[r] as usize].iter().map(|&x| x as usize).collect();
                v.sort_unstable();
                tp.selection.push(v);
            }
        }
        let n = (t * hd) as u64;
        launch_v(mk.add, n.div_ceil(256) as u32, 1, 1, 256, &[self.sub, self.x, n]);
        tap(&mut taps, |tp| &mut tp.ffn_in, self.x);
        cuda::d2d_async(self.xn, self.x, bytes);
        kn.mla.rmsnorm_rows(self.xn, w.post_norm, hd, t, self.st);
        if !self.moe_plans.iter().any(|p| p.tokens == t) {
            self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
        }
        let p = self.moe_plans.iter().find(|p| p.tokens == t).unwrap();
        p.run(&kn.k, &kn.mul1, &kn.moe, &w.moe, w.table, self.xn, self.sub);
        self.last_t = t;
        tap(&mut taps, |tp| &mut tp.ffn_out, self.sub);
        launch_v(mk.add, n.div_ceil(256) as u32, 1, 1, 256, &[self.sub, self.x, n]);
        tap(&mut taps, |tp| &mut tp.out, self.x);
        cuda::d2d_async(self.normed, self.x, bytes);
        kn.mla.rmsnorm_rows(self.normed, w.head_norm, hd, t, self.st);
    }

    /// the routing of the last call (synchronizes)
    ///
    /// # Safety
    /// A call ran.
    pub unsafe fn routing(&self) -> crate::glm5_moe::Routing {
        self.moe_plans.iter().find(|p| p.tokens == self.last_t).expect("glm5_mtp: no call yet").read_routing()
    }

    /// # Safety
    /// No launch of this pass is pending.
    pub unsafe fn free(&mut self) {
        self.mla_sc.free();
        self.mla_c.free();
        for p in self.moe_plans.iter_mut() {
            p.free();
        }
        for d in [&mut self.st, &mut self.eh, &mut self.x, &mut self.xn, &mut self.sub, &mut self.normed] {
            cuda::free_dev(d);
        }
    }
}

// ---------------------------------------------------------------- #192: speculative decoding

/// `CROW_GLM_MTP=N`: N draft tokens per decode step (0 or unset = off, today's path)
pub const MTP_ENV: &str = "CROW_GLM_MTP";
/// the MTP overlay `load_mtp` reads with the container (default [`GLM5_MTP_OVERLAY_CNQ`])
pub const OVERLAY_ENV: &str = "CROW_GLM_MTP_OVERLAY";
/// the overlay of #182, relative to the repo root (`geo::from_engine_dir` from `engine/`)
pub const GLM5_MTP_OVERLAY_CNQ: &str = "converter/GLM-5.3-Flash-MTP-overlay.cnq";
/// the most drafts per step (one KDA snapshot slot each: 34 x 4,489,216 B on GLM-5.3-Flash)
pub const MTP_MAX: usize = 4;
/// rows per call of the block over the prompt (the catch-up of every prompt row)
pub const MTP_CHUNK: usize = 16;

/// Parse `CROW_GLM_MTP`: unset, empty or `0` = off; `1..=MTP_MAX` drafts per step; anything
/// else refused by name.
pub fn draft_rows(v: Option<&str>) -> Result<usize, String> {
    match v.map(str::trim) {
        None | Some("") => Ok(0),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if n <= MTP_MAX => Ok(n),
            _ => Err(format!("{MTP_ENV}={s:?}: accepted 0 (off, default) .. {MTP_MAX} draft tokens per step")),
        },
    }
}

/// `CROW_GLM_MTP` from the environment (see [`draft_rows`])
pub fn draft_rows_from_env() -> Result<usize, String> {
    draft_rows(std::env::var(MTP_ENV).ok().as_deref())
}

/// What the speculative decode did since the start of the last `generate` (host counters).
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpecStats {
    /// drafts per step asked for (`CROW_GLM_MTP`)
    pub n: usize,
    /// verify steps (one multi-row trunk call each)
    pub steps: u64,
    /// ids the verify steps emitted (accepted drafts + the trunk's own id per step)
    pub tokens: u64,
    /// draft tokens verified, and those accepted
    pub drafts: u64,
    pub accepted: u64,
    /// trunk rows of the verify calls (1 + drafts per step)
    pub verify_rows: u64,
    /// steps by accepted drafts: `hist[a]` = steps that accepted `a` drafts, `a` in `0..=n`
    pub hist: Vec<u64>,
    /// rows the MTP block ran (prompt catch-up, per-step catch-up, chained drafts)
    pub mtp_rows: u64,
    /// KDA rollback: snapshot copies (one per KDA layer per draft row) and their bytes
    pub kda_snapshots: u64,
    pub kda_snapshot_bytes: u64,
    /// KDA rollback: steps that restored a snapshot (a rejected draft), copies and bytes
    pub kda_restore_steps: u64,
    pub kda_restores: u64,
    pub kda_restore_bytes: u64,
}

impl SpecStats {
    /// accepted / drafts (None before the first draft)
    pub fn acceptance(&self) -> Option<f64> {
        (self.drafts > 0).then(|| self.accepted as f64 / self.drafts as f64)
    }
}

/// a draft override (`(index of the generated id the draft guesses, draft) -> draft`): the GPU
/// tests force acceptances and rejections with it
pub type DraftHook = Box<dyn FnMut(usize, i64) -> i64>;

/// The device state of the speculative decode of one `Glm5Run` (`glm5_tiers`): the block, its
/// pass (own MLA + indexer cache), the verify buffers of `1 + n` rows, the KDA snapshot slots
/// (`n` per KDA layer), the counters.
pub struct Spec {
    pub n: usize,
    pub block: MtpBlock,
    pub mp: MtpPass,
    pub mk: MtpKernels,
    /// per decoder layer: `n` snapshot states for a KDA layer, none otherwise
    pub snaps: Vec<Vec<crate::glm5_kda::KdaState>>,
    /// verify rows: `x [1+n][4][H]`, `normed [1+n][H]`, `logits [1+n][V]`, `ids [1+n]` i32
    pub x: Dev,
    pub normed: Dev,
    pub logits: Dev,
    pub ids: Dev,
    /// the block's inputs: embeddings `[MTP_CHUNK][H]`, head-norm rows `[MTP_CHUNK][H]`
    pub e: Dev,
    pub h: Dev,
    /// a chained draft's logits `[V]` and greedy id
    pub dlogits: Dev,
    pub did: Dev,
    pub stats: SpecStats,
    pub hook: Option<DraftHook>,
    /// test probe: after every rollback, (last valid trunk row, every KDA state's S and conv bytes)
    #[cfg(test)]
    pub probe: Option<Box<dyn FnMut(usize, &[Vec<u8>])>>,
}

impl Spec {
    /// # Safety
    /// A CUDA context is current; `kda_layers` = the KDA flag of every decoder layer.
    pub unsafe fn new(g: &Glm5Geo, moe: MoeGeo, cap: usize, n: usize, block: MtpBlock, kda_layers: &[bool]) -> Spec {
        assert!((1..=MTP_MAX).contains(&n), "glm5_mtp: {n} drafts per step (1..={MTP_MAX})");
        let kd = crate::glm5_kda::KdaDims::of(g);
        let (h, v, t) = (g.hidden, g.vocab, 1 + n);
        let a = |what: &str, bytes: usize| cuda::alloc_named(what, bytes);
        Spec {
            n,
            block,
            mp: MtpPass::new(g, moe, MTP_CHUNK.max(t), cap),
            mk: MtpKernels::new(),
            snaps: kda_layers.iter().map(|&k| if k { (0..n).map(|_| crate::glm5_kda::KdaState::alloc(&kd)).collect() } else { Vec::new() }).collect(),
            x: a("glm5 MTP verify residual", t * g.hc_streams * h * 4),
            normed: a("glm5 MTP verify normed", t * h * 4),
            logits: a("glm5 MTP verify logits", t * v * 4),
            ids: a("glm5 MTP verify ids", t * 4),
            e: a("glm5 MTP embeddings", MTP_CHUNK.max(t) * h * 4),
            h: a("glm5 MTP head-norm rows", MTP_CHUNK.max(t) * h * 4),
            dlogits: a("glm5 MTP draft logits", v * 4),
            did: a("glm5 MTP draft id", 4),
            stats: SpecStats { n, hist: vec![0; n + 1], ..SpecStats::default() },
            hook: None,
            #[cfg(test)]
            probe: None,
        }
    }

    /// counters back to zero (a new `generate`)
    pub fn reset_stats(&mut self) {
        self.stats = SpecStats { n: self.n, hist: vec![0; self.n + 1], ..SpecStats::default() };
    }

    /// bytes of one snapshot slot over every KDA layer
    pub fn snapshot_bytes(&self) -> u64 {
        self.snaps.iter().filter_map(|s| s.first()).map(|s| s.bytes()).sum()
    }

    /// # Safety
    /// No launch reading this state is pending.
    pub unsafe fn free(&mut self) {
        cuda::sync();
        self.block.free();
        self.mp.free();
        self.mk.module.unload();
        for s in self.snaps.iter_mut().flatten() {
            s.free();
        }
        for d in [&mut self.x, &mut self.normed, &mut self.logits, &mut self.ids, &mut self.e, &mut self.h, &mut self.dlogits, &mut self.did] {
            cuda::free_dev(d);
        }
    }
}

/// The VRAM the speculative decode adds at `n` drafts and `cap` rows, derived (not measured; the
/// planner of `glm5_run` takes it off the free VRAM): the block's 288 records, its 25 overlay
/// tensors as loaded (norms f32, `kv_b` BF16, the rest as stored), the KDA snapshot slots, the
/// block's MLA cache and scratch, the verify buffers, the MoE plans of the block's calls.
pub fn spec_vram_bytes(g: &Glm5Geo, moe: &MoeGeo, cap: usize, n: usize) -> u64 {
    let md = MlaDims::of(g);
    let kd = crate::glm5_kda::KdaDims::of(g);
    let (h, v, t) = (g.hidden as u64, g.vocab as u64, 1 + n as u64);
    let records = g.experts as u64 * moe.record.bytes;
    let tensors: u64 = mtp_tensors(g)
        .iter()
        .filter(|x| x.store != Store::Mul1)
        .map(|x| match x.store {
            Store::Nvfp4 if x.name.contains("kv_b_proj") => x.values() as u64 * 2,
            Store::Bf16 if x.shape.len() == 1 => x.values() as u64 * 4,
            _ => x.overlay_bytes(),
        })
        .sum();
    let snaps = n as u64 * g.kda_layers as u64 * ((kd.state_floats() + kd.conv_floats()) * 4) as u64;
    let mt = MTP_CHUNK.max(n + 1) as u64;
    let cache = MlaCache::bytes(&md, cap);
    let scratch = 4 * mt * (md.q_lora + md.heads * md.nope + md.kv_lora + md.idx_proj() + md.idx_heads * md.idx_dim + (cap / crate::glm5_mla::KPOOL).max(1) + md.sel_max() + 2 * md.heads * md.kv_lora + md.heads * md.v) as u64
        + 4 * mt.max(16) * md.heads as u64 * (md.kv_lora as u64 + 2)
        + 4 * mt * h * 6;
    let verify = 4 * t * (g.hc_streams as u64 * h + h + v + 1) + 4 * (v + 1);
    // one MoE plan per call size: [c][H] gather + output, 3 x [c][inter], the MUL1 partials
    let plans: u64 = (1..=mt).map(|tt| 4 * tt * g.topk as u64 * (2 * h + 3 * g.expert_inter as u64) * 2).sum();
    records + tensors + snaps + cache + scratch + verify + plans
}

/// #192 test kit: a synthetic MTP block of `g` (bounded weights as `glm5_flags`' synthetic
/// trunk: NVFP4 codes random, scale bytes 0x30-0x38 under 0.2 / sqrt(cols); BF16 matrices
/// +-1 / sqrt(cols); norms 1 +- 0.05; vectors +-0.05; `g.experts` MUL1 records with random
/// trellis words and fp16 suh / svh of magnitude 0.06-0.12), all in VRAM.
///
/// # Safety
/// A CUDA context is current.
#[cfg(test)]
pub(crate) unsafe fn synthetic_block(g: &Glm5Geo, moe: &MoeGeo, seed: u64) -> MtpBlock {
    struct R(u64);
    impl R {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn sym(&mut self) -> f32 {
            (self.u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }
    fn f16(v: f32) -> u16 {
        let b = v.to_bits();
        let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        (((b >> 16) & 0x8000) | ((e as u32) << 10) | ((b & 0x7F_FFFF) >> 13)) as u16
    }
    let mut r = R(seed);
    let md = MlaDims::of(g);
    let h = g.hidden;
    let bf = |r: &mut R, rows: usize, cols: usize| -> Dev {
        let a = 1.0 / (cols as f32).sqrt();
        cuda::to_dev(&(0..rows * cols).map(|_| crate::glm5_model::f32_to_bf16_rne(a * r.sym())).collect::<Vec<u16>>())
    };
    let norm = |r: &mut R, n: usize| -> Dev { cuda::to_f32_dev(&(0..n).map(|_| 1.0 + 0.05 * r.sym()).collect::<Vec<f32>>()) };
    let vec = |r: &mut R, n: usize| -> Dev { cuda::to_f32_dev(&(0..n).map(|_| 0.05 * r.sym()).collect::<Vec<f32>>()) };
    let fp4 = |r: &mut R, rows: usize, cols: usize| -> GpuNvfp4 {
        let mut b = vec![0u8; (rows * cols).div_ceil(64) * 36];
        for blk in b.chunks_exact_mut(36) {
            for (i, x) in blk.iter_mut().enumerate() {
                *x = if i < 4 { 0x30 + (r.u64() % 9) as u8 } else { r.u64() as u8 };
            }
        }
        GpuNvfp4 { w: cuda::upload_dev(&b), gs: cuda::to_f32_dev(&[0.2 / (cols as f32).sqrt()]), rows, cols }
    };
    let attn = MlaWeights {
        q_a: 0,
        q_a_norm: norm(&mut r, md.q_lora),
        q_b: 0,
        kv_a: 0,
        kv_a_norm: norm(&mut r, md.kv_lora),
        kv_b: bf(&mut r, md.heads * (md.nope + md.v), md.kv_lora),
        o_proj: 0,
        idx_wq_b: bf(&mut r, md.idx_heads * md.idx_dim, md.q_lora),
        idx_x: bf(&mut r, md.idx_proj(), h),
        idx_k_norm_w: norm(&mut r, md.idx_dim),
        idx_k_norm_b: vec(&mut r, md.idx_dim),
        idx_ape: vec(&mut r, md.kpool * md.idx_dim),
    };
    let si = g.expert_inter * g.shared_experts;
    let moe_w = GpuMoeWeights {
        router: bf(&mut r, g.experts, h),
        bias: vec(&mut r, g.experts),
        shared: GpuFfnWeights { gate: fp4(&mut r, si, h), up: fp4(&mut r, si, h), down: fp4(&mut r, h, si) },
    };
    let (q_a, q_b, kv_a, o) = (fp4(&mut r, md.q_lora, h), fp4(&mut r, md.heads * md.nope, md.q_lora), fp4(&mut r, md.kv_lora, h), fp4(&mut r, h, md.heads * md.v));
    let rb = moe.record.bytes as usize;
    let specs = crate::kernels::mul1::record_specs(g.hidden, g.expert_inter, 3, false);
    let trellis = specs[0].suh_off;
    let records = cuda::alloc_named("glm5 MTP synthetic records", g.experts * rb);
    let mut bases = Vec::with_capacity(g.experts);
    for e in 0..g.experts {
        let mut rec = vec![0u8; rb];
        for w in rec[..trellis].chunks_exact_mut(8) {
            w.copy_from_slice(&r.u64().to_le_bytes());
        }
        for s in &specs {
            for (at, cnt) in [(s.suh_off, s.k), (s.svh_off, s.n)] {
                for i in 0..cnt {
                    let v = (0.06 + 0.03 * (r.sym() + 1.0)) * if r.u64() & 1 == 1 { -1.0 } else { 1.0 };
                    rec[at + 2 * i..at + 2 * i + 2].copy_from_slice(&f16(v).to_le_bytes());
                }
            }
        }
        let base = records + (e * rb) as u64;
        cuda::into_dev(base, rec.as_slice());
        bases.push(base);
    }
    let i = |v: usize| v as i32;
    let cols = cuda::to_i32_dev(&[i(q_a.cols), i(q_b.cols), i(kv_a.cols), i(o.cols), i(q_a.rows), i(q_b.rows), i(kv_a.rows), i(o.rows)]);
    MtpBlock {
        w: MtpWeights {
            enorm: norm(&mut r, h),
            hnorm: norm(&mut r, h),
            eh_proj: bf(&mut r, h, 2 * h),
            input_norm: norm(&mut r, h),
            post_norm: norm(&mut r, h),
            head_norm: norm(&mut r, h),
            attn,
            moe: moe_w,
            records,
            table: cuda::to_u64_dev(&bases),
        },
        q_a,
        q_b,
        kv_a,
        o,
        cols,
        bytes: 0,
        sanitized: 0,
        kv_b_inexact: 0,
    }
}

#[cfg(test)]
mod spec_gpu_tests {
    use super::*;

    /// #191 for the block: `MtpBlock::call` (its NVFP4 projections on `glm5_gemv_fp4`) gives the
    /// bits of the same call with the record kernel `gemv_fp4_b`, real shapes, a synthetic block,
    /// 3 rows then 1 row. Run with
    /// `cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1`.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM): cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_spec_gpu_block_gemv_is_the_record_kernel() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let moe = MoeGeo::new(&g, crate::geo::ExpertRecordSpec::new(crate::geo::ExpertCodec::Mul1, crate::cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Glm5Kernels::new(&g);
            let mk = MtpKernels::new();
            let mut blk = synthetic_block(&g, &moe, 0x0191);
            let h = g.hidden;
            let mut x = 0x5eedu64;
            let mut rnd = |n: usize, a: f32| -> Vec<f32> {
                (0..n)
                    .map(|_| {
                        x ^= x << 13;
                        x ^= x >> 7;
                        x ^= x << 17;
                        a * ((x >> 40) as f32 / (1u64 << 23) as f32 - 1.0)
                    })
                    .collect()
            };
            let (e, hv) = (cuda::to_f32_dev(&rnd(4 * h, 0.035)), cuda::to_f32_dev(&rnd(4 * h, 1.7)));
            let gemv = kn.k.f("gemv_fp4_b");
            let mut outs = Vec::new();
            for record in [false, true] {
                let mut pass = MtpPass::new(&g, moe, 3, 16);
                let mut got = Vec::new();
                for (p0, t) in [(0usize, 3usize), (3, 1)] {
                    let (eo, ho) = (e + (p0 * h * 4) as u64, hv + (p0 * h * 4) as u64);
                    if record {
                        let cols = cuda::to_i32_dev(&[blk.q_a.cols as i32, blk.q_b.cols as i32, blk.kv_a.cols as i32, blk.o.cols as i32]);
                        let b = &blk;
                        let mut proj = |s: &MlaScratch, p: MlaProj, xi: Dev, yo: Dev| {
                            let (m, k) = match p {
                                MlaProj::QA => (&b.q_a, 0u64),
                                MlaProj::QB => (&b.q_b, 1),
                                MlaProj::KVA => (&b.kv_a, 2),
                                MlaProj::O => (&b.o, 3),
                            };
                            launch_v(gemv, m.rows as u32, s.t() as u32, 1, 256, &[m.w, xi, m.gs, yo, cols + 4 * k]);
                        };
                        pass.call(&kn, &mk, &blk.w, eo, ho, p0, t, false, Some(&mut proj));
                        cuda::sync();
                        let mut c = cols;
                        cuda::free_dev(&mut c);
                    } else {
                        blk.call(&mut pass, &kn, &mk, eo, ho, p0, t, false, None);
                    }
                    cuda::sync();
                    got.extend(cuda::dtoh(pass.normed, t * h).into_iter().map(f32::to_bits));
                }
                pass.free();
                outs.push(got);
            }
            let differ = outs[0].iter().zip(&outs[1]).filter(|(a, b)| a != b).count();
            assert_eq!(differ, 0, "{differ} of {} head-norm values differ between glm5_gemv_fp4 and gemv_fp4_b", outs[0].len());
            assert!(outs[0].iter().all(|&b| f32::from_bits(b).is_finite()));
            blk.free();
        }
    }
}

#[cfg(test)]
mod spec_host_tests {
    use super::*;

    #[test]
    fn crow_glm_mtp_parses_and_refuses_by_name() {
        assert_eq!(draft_rows(None), Ok(0));
        assert_eq!(draft_rows(Some("")), Ok(0));
        assert_eq!(draft_rows(Some("0")), Ok(0));
        assert_eq!(draft_rows(Some(" 1 ")), Ok(1));
        assert_eq!(draft_rows(Some("4")), Ok(4));
        for bad in ["5", "-1", "on", "1.5"] {
            let e = draft_rows(Some(bad)).unwrap_err();
            assert!(e.starts_with(&format!("{MTP_ENV}=\"{bad}\": accepted 0 (off, default) .. 4")), "{e}");
        }
    }

    #[test]
    fn spec_stats_acceptance_and_vram_estimate() {
        let s = SpecStats { drafts: 10, accepted: 4, ..SpecStats::default() };
        assert_eq!(s.acceptance(), Some(0.4));
        assert_eq!(SpecStats::default().acceptance(), None);
        let g = Glm5Geo::GLM_5_3_FLASH;
        let moe = MoeGeo::new(&g, crate::geo::ExpertRecordSpec::new(crate::geo::ExpertCodec::Mul1, crate::cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
        let (b1, b2) = (spec_vram_bytes(&g, &moe, 4096, 1), spec_vram_bytes(&g, &moe, 4096, 2));
        // one more draft row adds at least one KDA snapshot slot: 34 x (4 MiB + 288 KiB)
        assert!(b2 - b1 >= 34 * 4_489_216, "{b1} {b2}");
        // the 288 records alone are 2,728,525,824 B
        assert!(b1 > 288 * 9_474_048, "{b1}");
    }
}

#[cfg(test)]
mod tests {
    //! Host tests: the tensor plan, the pairing, the host twin's exact ops, the kernel source.
    //! `glm5_mtp_glue_matches_the_oracle_golden` (ignored, needs the golden and the FP8 originals)
    //! holds the host twin against `oracle/glm5_mtp.py` on the real weights.
    use super::*;

    const G: Glm5Geo = Glm5Geo::GLM_5_3_FLASH;

    #[test]
    fn glm5_mtp_tensor_plan_is_layer_45() {
        assert_eq!(mtp_layer(&G), 45);
        assert_eq!(mtp_prefix(&G), "model.language_model.layers.45.");
        let ts = mtp_tensors(&G);
        assert_eq!(ts.len(), 25 + 3 * 288);
        let mut names: Vec<&str> = ts.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), ts.len(), "a name twice");
        let rest: Vec<&MtpTensor> = ts.iter().filter(|t| t.store != Store::Mul1).collect();
        assert_eq!(rest.len(), 25);
        assert_eq!(rest.iter().filter(|t| t.ckpt_fp8).count(), 7);
        let eh = ts.iter().find(|t| t.name == "eh_proj.weight").unwrap();
        assert_eq!((eh.shape.clone(), eh.ckpt_fp8, eh.store), (vec![4096, 8192], false, Store::Bf16));
        let kvb = ts.iter().find(|t| t.name == "self_attn.kv_b_proj.weight").unwrap();
        assert_eq!(kvb.shape, vec![64 * 512, 512]);
        // no hc_*, no embed / head of its own
        assert!(!ts.iter().any(|t| t.name.starts_with("hc_") || t.name.contains("embed") || t.name.contains("shared_head.head")));
    }

    #[test]
    fn glm5_mtp_overlay_sizes() {
        let (mixed, bf16) = overlay_bytes(&G);
        eprintln!("glm5_mtp overlay: trunk codecs {mixed} B ({:.1} MiB), all BF16 {bf16} B ({:.1} MiB)", mixed as f64 / 1048576.0, bf16 as f64 / 1048576.0);
        // eh_proj alone is 64 MiB BF16; the NVFP4 projections shrink the rest
        assert!(mixed < bf16 && mixed > 64 << 20, "{mixed} {bf16}");
    }

    #[test]
    fn glm5_mtp_pairing_rules() {
        let k = pair_rows(5, Pos0::Keep);
        assert_eq!(k.len(), 4);
        for (i, r) in k.iter().enumerate() {
            assert_eq!((r.token, r.h_row, r.pos, r.zero_embed), (i + 1, Some(i), i, false));
        }
        assert_eq!(k.iter().map(|r| r.drafts).collect::<Vec<_>>(), vec![Some(2), Some(3), Some(4), None]);
        let z = pair_rows(5, Pos0::ZeroEmbed);
        assert_eq!(z.iter().map(|r| r.zero_embed).collect::<Vec<_>>(), vec![true, false, false, false]);
        assert_eq!(z.iter().map(|r| (r.token, r.h_row)).collect::<Vec<_>>(), k.iter().map(|r| (r.token, r.h_row)).collect::<Vec<_>>());
        let l = pair_rows(5, Pos0::LeadRow);
        assert_eq!(l.len(), 5);
        assert_eq!(l[0], DraftRow { token: 0, h_row: None, pos: 0, zero_embed: false, drafts: None });
        for i in 0..4 {
            assert_eq!((l[i + 1].token, l[i + 1].h_row, l[i + 1].pos, l[i + 1].drafts), (k[i].token, k[i].h_row, i + 1, k[i].drafts));
        }
        assert_eq!(agreement(&[1, 2, 3], &[1, 5, 3, 9]), (2, 3));
    }

    #[test]
    fn glm5_mtp_eh_norm_ref_exact_ops() {
        let e = [3.0f32, -4.0, 0.0, 0.0];
        let h = [1.0f32, 1.0, 1.0, 1.0];
        let we = [1.0f32, 2.0, 1.0, 1.0];
        let wh = [0.5f32; 4];
        let v = eh_norm_ref(&e, &h, &we, &wh, false);
        let r = 1.0 / (25.0f64 / 4.0 + 1e-5).sqrt();
        assert_eq!(v.len(), 8);
        assert!((v[0] - 3.0 * r).abs() < 1e-12 && (v[1] - -8.0 * r).abs() < 1e-12);
        let rh = 1.0 / (1.0f64 + 1e-5).sqrt();
        assert!(v[4..].iter().all(|&x| (x - 0.5 * rh).abs() < 1e-12));
        // the zeroed embedding: the e half is zeros, the h half unchanged
        let z = eh_norm_ref(&e, &h, &we, &wh, true);
        assert!(z[..4].iter().all(|&x| x == 0.0) && z[4..] == v[4..]);
        // e first: eh_proj picks the e half with its first H columns
        let w: Vec<u16> = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0].iter().map(|&x| crate::glm5_model::f32_to_bf16_rne(x)).collect();
        assert_eq!(eh_proj_ref(&w, &v), vec![v[0]]);
    }

    fn ti(name: String, section: &str, dtype: &str, shape: Vec<u64>) -> TensorInfo {
        TensorInfo { name, section: section.into(), dtype: dtype.into(), offset: 0, n_values: shape.iter().product(), global_scale: 1.0, shape, overlay: false }
    }

    fn model_block(rev: &str) -> ModelBlock {
        ModelBlock {
            family: "Glm5Next".into(),
            model_type: "glm5_next_text".into(),
            recipe: "cnq4.5-glm5-next".into(),
            config_json: "{\"text_config\": {}}".into(),
            generation_config_json: "{}".into(),
            source_repo: "zai-org/GLM-5.3-Flash".into(),
            source_revision: rev.into(),
            geo: serde_json::Value::Null,
            shards: serde_json::Value::Null,
        }
    }

    /// (base index, overlay index) as the 3-bit container and `converter --mtp-overlay` write them
    fn base_and_overlay() -> (Vec<TensorInfo>, Vec<TensorInfo>) {
        let p = mtp_prefix(&G);
        let mut base = vec![ti("lm_head.weight".into(), "text", "bf16", vec![154880, 4096])];
        let mut ov = Vec::new();
        for t in mtp_tensors(&G) {
            let shape: Vec<u64> = t.shape.iter().map(|&v| v as u64).collect();
            match t.store {
                Store::Mul1 => base.push(ti(format!("{p}{}", t.name), MTP_SECTION, "mul1", shape)),
                s => ov.push(ti(format!("{p}{}", t.name), MTP_SECTION, overlay_dtype(s), shape)),
            }
        }
        (base, ov)
    }

    /// #182: the overlay of `converter --mtp-overlay` fits the 3-bit container; every way it can
    /// not fit is refused by name before a byte is read.
    #[test]
    fn glm5_mtp_overlay_check_accepts_the_overlay_and_names_every_misfit() {
        let (base, ov) = base_and_overlay();
        let (mb, mo) = (model_block("eb9eb208"), model_block("eb9eb208"));
        assert_eq!(ov.len(), 25);
        assert_eq!(mtp_overlay_check(&G, &base, Some(&mb), &ov, Some(&mo)), Ok(()));
        let refused = |base: &[TensorInfo], ov: &[TensorInfo], mo: &ModelBlock, want: &str| {
            let e = mtp_overlay_check(&G, base, Some(&mb), ov, Some(mo)).unwrap_err();
            assert!(e.contains(want), "{want:?} not in {e}");
        };
        // a tensor missing
        let gone: Vec<TensorInfo> = ov.iter().filter(|t| !t.name.ends_with("eh_proj.weight")).cloned().collect();
        refused(&base, &gone, &mo, "layers.45.eh_proj.weight: not in the overlay");
        // another codec than the plan's (an all-BF16 overlay is not this loader's)
        let mut bf = ov.clone();
        bf.iter_mut().find(|t| t.name.ends_with("q_a_proj.weight")).unwrap().dtype = "bf16".into();
        refused(&base, &bf, &mo, "q_a_proj.weight: [mtp] bf16");
        // another section
        let mut sec = ov.clone();
        sec[0].section = "text".into();
        refused(&base, &sec, &mo, "[text]");
        // a tensor that is not the block's
        let mut extra = ov.clone();
        extra.push(ti("model.language_model.layers.44.input_layernorm.weight".into(), MTP_SECTION, "bf16", vec![4096]));
        refused(&base, &extra, &mo, "layers.44.input_layernorm.weight");
        // another checkpoint revision
        refused(&base, &ov, &model_block("0000000"), "source revision: `0000000` in the overlay");
        // a base without the block's records
        let trunk_only: Vec<TensorInfo> = base.iter().filter(|t| t.section != MTP_SECTION).cloned().collect();
        refused(&trunk_only, &ov, &mo, "0 of the block's 288 MUL1 records");
        // no model block
        assert!(mtp_overlay_check(&G, &base, Some(&mb), &ov, None).unwrap_err().contains("index v2"));
    }

    /// #182: the real overlay against the real 3-bit container's index (trailers only, nothing
    /// mapped): `GLM5_CNQ=<GLM-5.3-Flash-MUL1K3.cnq> GLM5_MTP_OVERLAY=<GLM-5.3-Flash-MTP-overlay.cnq>`.
    #[test]
    #[ignore = "needs the 3-bit container and the overlay: GLM5_CNQ=.. GLM5_MTP_OVERLAY=.. cargo test --release --lib glm5_mtp_real_overlay -- --ignored --nocapture"]
    fn glm5_mtp_real_overlay_fits_the_real_container() {
        let peek = |k: &str| Cnq::peek_index(&std::env::var(k).unwrap_or_else(|_| panic!("{k}"))).unwrap_or_else(|e| panic!("{k}: {e}"));
        let (b, o) = (peek("GLM5_CNQ"), peek("GLM5_MTP_OVERLAY"));
        mtp_overlay_check(&G, &b.tensors, b.model(), &o.tensors, o.model()).unwrap_or_else(|e| panic!("{e}"));
        // the container lacks exactly the 25 the overlay brings
        let missing: Vec<String> = mtp_tensors(&G).into_iter().map(|t| format!("{}{}", mtp_prefix(&G), t.name)).filter(|n| !b.tensors.iter().any(|t| &t.name == n)).collect();
        let mut have: Vec<String> = o.tensors.iter().map(|t| t.name.clone()).collect();
        have.sort();
        let mut miss = missing.clone();
        miss.sort();
        assert_eq!(have, miss);
        let bytes: u64 = o.tensors.iter().map(Cnq::byte_len).sum::<u64>() + o.tensors.iter().filter(|t| t.dtype == "nvfp4").count() as u64 * 4;
        assert_eq!(bytes, overlay_bytes(&G).0, "the overlay's bytes (+4 per NVFP4 global scale) are the plan's");
        eprintln!("glm5_mtp overlay fits: {} tensors, {bytes} B incl. global scales", o.tensors.len());
    }

    #[test]
    fn glm5_mtp_source_compiles_with_every_entry() {
        let ptx = crate::kernels::tests_300_c4::ptx(MTP_SRC);
        let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), NAMES.len(), "{names:?}");
        for n in NAMES {
            assert!(names.iter().any(|m| m == n), "{n} missing in {names:?}");
        }
    }

    // ---------------------------------------------------------------- the oracle golden (real weights)

    /// one tensor of a safetensors shard set, raw bytes and dtype
    fn st_tensor(dir: &std::path::Path, name: &str) -> (String, Vec<usize>, Vec<u8>) {
        use std::io::{Read, Seek, SeekFrom};
        let idx: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(dir.join("model.safetensors.index.json")).unwrap()).unwrap();
        let shard = idx["weight_map"][name].as_str().unwrap_or_else(|| panic!("{name}: not in the index"));
        let mut f = std::fs::File::open(dir.join(shard)).unwrap();
        let mut n8 = [0u8; 8];
        f.read_exact(&mut n8).unwrap();
        let n = u64::from_le_bytes(n8) as usize;
        let mut hdr = vec![0u8; n];
        f.read_exact(&mut hdr).unwrap();
        let hv: serde_json::Value = serde_json::from_slice(&hdr).unwrap();
        let e = &hv[name];
        let off: Vec<u64> = e["data_offsets"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let shape: Vec<usize> = e["shape"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        f.seek(SeekFrom::Start(8 + n as u64 + off[0])).unwrap();
        let mut b = vec![0u8; (off[1] - off[0]) as usize];
        f.read_exact(&mut b).unwrap();
        (e["dtype"].as_str().unwrap().to_string(), shape, b)
    }

    fn f32s(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn bf16_as_f32(b: &[u8]) -> Vec<f32> {
        b.chunks_exact(2).map(|c| crate::glm5_head::bf16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect()
    }

    /// The host twin of the glue (eh_norm + eh_proj) on the oracle golden's inputs and the
    /// checkpoint's BF16 enorm / hnorm / eh_proj, against the oracle's `mtp-eh.f32`; and the
    /// golden's fallback names are exactly this plan's non-expert tensors.
    #[test]
    #[ignore = "needs the golden and the FP8 originals: GLM5_MTP_GOLDEN=<ref-mul1-mtp> GLM5_ORIGINALS=<dir> cargo test --release --lib glm5_mtp_glue -- --ignored --nocapture"]
    fn glm5_mtp_glue_matches_the_oracle_golden() {
        let gd = std::path::PathBuf::from(std::env::var("GLM5_MTP_GOLDEN").expect("GLM5_MTP_GOLDEN"));
        let od = std::path::PathBuf::from(std::env::var("GLM5_ORIGINALS").expect("GLM5_ORIGINALS"));
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(gd.join("manifest.json")).unwrap()).unwrap();
        let p = mtp_prefix(&G);
        let mut want: Vec<String> = mtp_tensors(&G).into_iter().filter(|t| t.store != Store::Mul1).map(|t| format!("{p}{}", t.name)).collect();
        want.sort();
        let mut got: Vec<String> = man["weights"]["fallback_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        got.sort();
        assert_eq!(got, want, "the golden's fallback names are not the 25 non-expert tensors");
        let rows = man["rows"].as_u64().unwrap() as usize;
        let hd = G.hidden;
        let rd = |f: &str| f32s(&std::fs::read(gd.join(f)).unwrap());
        let (e, h, eh) = (rd("mtp-embed.f32"), rd("mtp-h.f32"), rd("mtp-eh.f32"));
        assert_eq!((e.len(), h.len(), eh.len()), (rows * hd, rows * hd, rows * hd));
        let w = |n: &str| {
            let (dt, _, b) = st_tensor(&od, &format!("{p}{n}"));
            assert_eq!(dt, "BF16", "{n}");
            b
        };
        let we = bf16_as_f32(&w("enorm.weight"));
        let wh = bf16_as_f32(&w("hnorm.weight"));
        let wp: Vec<u16> = w("eh_proj.weight").chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        let (mut worst, mut num, mut den) = (0f64, 0f64, 0f64);
        for r in 0..rows {
            let x = eh_norm_ref(&e[r * hd..(r + 1) * hd], &h[r * hd..(r + 1) * hd], &we, &wh, false);
            let y = eh_proj_ref(&wp, &x);
            for (j, &v) in y.iter().enumerate() {
                let o = eh[r * hd + j] as f64;
                worst = worst.max((v - o).abs());
                num += (v - o) * (v - o);
                den += o * o;
            }
        }
        let rel = (num / den).sqrt();
        eprintln!("glm5_mtp glue vs oracle mtp-eh over {rows} rows: max abs {worst:.3e}, rel rms {rel:.3e}");
        assert!(rel < 1e-5, "rel rms {rel}");
    }
}

#[cfg(test)]
mod tests_gpu {
    //! The block on the GPU (real shapes, synthetic weights) against the host composition of the
    //! blocks' own references: the glue's host twin, `glm5_mla::host::forward` (absorbed, BF16
    //! cache) and `glm5_moe::moe_cpu` (the MUL1 CPU lane), calls of 3 prompt rows then decode rows.
    //! `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_mtp::tests_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::cpu_mul1::testkit::{self, Rng};
    use crate::cpu_nvfp4::{ExpertBlock, Nvfp4Matrix, Path};
    use crate::geo::{ExpertCodec, ExpertRecordSpec};
    use crate::glm5_mla::host;
    use crate::glm5_mla::synth;
    use crate::glm5_moe::{moe_cpu, ExpertRecords, GpuFfnWeights, GpuNvfp4, MoeLayerCpu, RouterWeights};
    use std::collections::BTreeMap;

    const G: Glm5Geo = Glm5Geo::GLM_5_3_FLASH;
    const N: usize = 6;
    const T_PROMPT: usize = 3;
    const NVFP4_GS: f32 = 0.3;

    fn bf16(v: f32) -> u16 {
        synth::bf16_bits(v)
    }

    fn nvfp4(rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
        let mut b = vec![0u8; Nvfp4Matrix::byte_len(rows, cols)];
        for blk in b.chunks_exact_mut(36) {
            for (i, v) in blk.iter_mut().enumerate() {
                *v = if i < 4 { 0x30 + (rng.next() % 16) as u8 } else { rng.next() as u8 };
            }
        }
        b
    }

    struct Records(BTreeMap<u32, Vec<u8>>);
    impl ExpertRecords for Records {
        fn record(&self, e: u32) -> Result<&[u8], String> {
            self.0.get(&e).map(|v| v.as_slice()).ok_or_else(|| format!("no record for expert {e}"))
        }
    }

    fn record(e: u32, moe: &MoeGeo) -> Vec<u8> {
        testkit::record(&testkit::Case { name: format!("mtp-e{e}"), source: format!("synth:{}", 0x1820 + 9 * e), bitrate: moe.bitrate, hidden: 4096, inter: 2048, want: [String::new(), String::new(), String::new()] })
    }

    fn cos(a: &[f64], b: &[f32]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x * y as f64;
            aa += x * x;
            bb += y as f64 * y as f64;
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mtp::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_gpu_block_matches_the_host_composition() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let moe = MoeGeo::new(&G, ExpertRecordSpec::new(ExpertCodec::Mul1, crate::cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
            let kn = Glm5Kernels::new(&G);
            let mk = MtpKernels::new();
            let (hd, e_n) = (G.hidden, G.experts);
            let md = MlaDims::of(&G);
            let mut rng = Rng(0x0182_3001);
            // inputs: embeddings ~ 0.02 (the real table's scale), trunk head-norm rows ~ 1
            let e: Vec<f32> = (0..N * hd).map(|_| rng.f(0.035)).collect();
            let h: Vec<f32> = (0..N * hd).map(|_| rng.f(1.7)).collect();
            let vecw = |rng: &mut Rng, c: f32, a: f32| -> Vec<f32> { (0..hd).map(|_| synth::bf16_round(c + rng.f(a))).collect() };
            let (we, wh, w_in, w_post, w_head) = (vecw(&mut rng, 1.0, 0.3), vecw(&mut rng, 1.0, 0.3), vecw(&mut rng, 1.0, 0.1), vecw(&mut rng, 0.17, 0.05), vecw(&mut rng, 1.0, 0.1));
            let amp = (3.0f32 / (2 * hd) as f32).sqrt();
            let eh_w: Vec<u16> = (0..hd * 2 * hd).map(|_| bf16(rng.f(amp))).collect();
            let mla = synth::weights(&md, 0x0182);
            let router: Vec<u16> = (0..e_n * hd).map(|_| bf16(rng.f(0.55))).collect();
            let bias: Vec<f32> = (0..e_n).map(|_| rng.f(0.09)).collect();
            let si = moe.shared_inter;
            let sh = [nvfp4(si, hd, &mut rng), nvfp4(si, hd, &mut rng), nvfp4(hd, si, &mut rng)];
            // ---- host composition
            let mut x = Vec::with_capacity(N * hd);
            let mut xin = Vec::with_capacity(N * hd);
            for r in 0..N {
                let v = eh_proj_ref(&eh_w, &eh_norm_ref(&e[r * hd..(r + 1) * hd], &h[r * hd..(r + 1) * hd], &we, &wh, false));
                xin.extend(rms_ref(&v, &w_in, 1e-5).iter().map(|&a| a as f32));
                x.extend(v);
            }
            let att = host::forward(&mla, &xin, true, true);
            let y: Vec<f64> = x.iter().zip(&att.y).map(|(a, b)| a + b).collect();
            let yin: Vec<f32> = y.chunks_exact(hd).flat_map(|r| rms_ref(r, &w_post, 1e-5)).map(|a| a as f32).collect();
            let blk = ExpertBlock::new(
                Nvfp4Matrix::new(&sh[0], si, hd, NVFP4_GS).unwrap(),
                Nvfp4Matrix::new(&sh[1], si, hd, NVFP4_GS).unwrap(),
                Nvfp4Matrix::new(&sh[2], hd, si, NVFP4_GS).unwrap(),
            )
            .unwrap();
            let rw = RouterWeights { weight: &router, bias: &bias };
            let mut recs = BTreeMap::new();
            for e in crate::glm5_moe::route(&moe, &rw, &yin).needed() {
                recs.insert(e, record(e, &moe));
            }
            let mut f = vec![0f32; N * hd];
            let route_host = moe_cpu(&moe, &MoeLayerCpu { router: rw, shared: blk }, &Records(recs.clone()), &yin, &mut f, 8, Path::Auto).unwrap();
            let z: Vec<f64> = y.iter().zip(&f).map(|(a, &b)| a + b as f64).collect();
            let want: Vec<f64> = z.chunks_exact(hd).flat_map(|r| rms_ref(r, &w_head, 1e-5)).collect();
            // ---- device
            let rb = moe.record.bytes;
            let first = *recs.keys().next().unwrap();
            let all: Vec<u8> = recs.values().flatten().copied().collect();
            let records = cuda::upload_dev(&all);
            let order: Vec<u32> = recs.keys().copied().collect();
            let bases: Vec<u64> = (0..e_n as u32).map(|e| records + rb * order.iter().position(|&n| n == e).unwrap_or(order.iter().position(|&n| n == first).unwrap()) as u64).collect();
            let up = |v: &[f32]| cuda::to_f32_dev(v);
            let bfd = |v: &[f32]| cuda::to_dev(&v.iter().map(|&x| bf16(x)).collect::<Vec<u16>>());
            let fp4 = |b: &Vec<u8>, r: usize, c: usize| GpuNvfp4 { w: cuda::upload_dev(b), gs: cuda::to_f32_dev(&[NVFP4_GS]), rows: r, cols: c };
            let mut w = MtpWeights {
                enorm: up(&we),
                hnorm: up(&wh),
                eh_proj: cuda::to_dev(&eh_w),
                input_norm: up(&w_in),
                post_norm: up(&w_post),
                head_norm: up(&w_head),
                attn: MlaWeights {
                    q_a: bfd(&mla.q_a),
                    q_a_norm: up(&mla.q_a_norm),
                    q_b: bfd(&mla.q_b),
                    kv_a: bfd(&mla.kv_a),
                    kv_a_norm: up(&mla.kv_a_norm),
                    kv_b: bfd(&mla.kv_b),
                    o_proj: bfd(&mla.o_proj),
                    idx_wq_b: bfd(&mla.wq_b),
                    idx_x: bfd(&mla.idx_x),
                    idx_k_norm_w: up(&mla.k_norm_w),
                    idx_k_norm_b: up(&mla.k_norm_b),
                    idx_ape: up(&mla.ape),
                },
                moe: GpuMoeWeights {
                    router: cuda::to_dev(&router),
                    bias: up(&bias),
                    shared: GpuFfnWeights { gate: fp4(&sh[0], si, hd), up: fp4(&sh[1], si, hd), down: fp4(&sh[2], hd, si) },
                },
                records,
                table: cuda::to_u64_dev(&bases),
            };
            let mut pass = MtpPass::new(&G, moe, T_PROMPT, 64);
            let (mut ed, mut hdv) = (up(&e), up(&h));
            let mut got = Vec::new();
            let mut ids_dev = Vec::new();
            let calls: Vec<(usize, usize)> = std::iter::once((0, T_PROMPT)).chain((T_PROMPT..N).map(|r| (r, 1))).collect();
            for &(p0, t) in &calls {
                pass.call(&kn, &mk, &w, ed + (p0 * hd * 4) as u64, hdv + (p0 * hd * 4) as u64, p0, t, false, None);
                cuda::sync();
                got.extend(cuda::dtoh(pass.normed, t * hd));
                ids_dev.extend(pass.routing().ids);
            }
            for r in 0..N {
                let c = cos(&want[r * hd..(r + 1) * hd], &got[r * hd..(r + 1) * hd]);
                let ma = want[r * hd..(r + 1) * hd].iter().zip(&got[r * hd..(r + 1) * hd]).map(|(a, &b)| (a - b as f64).abs()).fold(0f64, f64::max);
                eprintln!("glm5_mtp GPU row {r}: 1 - cosine {:.2e}, max abs {ma:.3e}", 1.0 - c);
                assert!(c >= 0.9999, "row {r}: cosine {c:.9}");
            }
            let set = |v: &[u32]| -> Vec<Vec<u32>> {
                v.chunks(G.topk).map(|c| {
                    let mut s = c.to_vec();
                    s.sort_unstable();
                    s
                }).collect()
            };
            assert_eq!(set(&ids_dev), set(&route_host.ids), "routing");
            // zero_pos0: only the row at position 0 changes its eh input
            pass.free();
            let mut p2 = MtpPass::new(&G, moe, T_PROMPT, 64);
            p2.call(&kn, &mk, &w, ed, hdv, 0, T_PROMPT, true, None);
            cuda::sync();
            let ehz = cuda::dtoh(p2.eh, T_PROMPT * 2 * hd);
            for r in 0..T_PROMPT {
                let want = eh_norm_ref(&e[r * hd..(r + 1) * hd], &h[r * hd..(r + 1) * hd], &we, &wh, r == 0);
                let c = cos(&want, &ehz[r * 2 * hd..(r + 1) * 2 * hd]);
                assert!(c > 0.999_999, "eh row {r}: {c}");
            }
            assert!(ehz[..hd].iter().all(|&v| v == 0.0));
            p2.free();
            w.free();
            for d in [&mut ed, &mut hdv] {
                cuda::free_dev(d);
            }
        }
    }

    fn read_f32(p: &std::path::Path, n: usize) -> Vec<f32> {
        let b = std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display()));
        assert_eq!(b.len(), n * 4, "{}", p.display());
        b.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn read_i32(p: &std::path::Path) -> Vec<i32> {
        std::fs::read(p).unwrap_or_else(|e| panic!("{}: {e}", p.display())).chunks_exact(4).map(|c| i32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    /// (min, mean) cosine per row of `a` against `b`, both `[rows][h]`
    fn row_cos(a: &[f32], b: &[f32], h: usize) -> (f64, f64) {
        let c: Vec<f64> = a.chunks_exact(h).zip(b.chunks_exact(h)).map(|(x, y)| cos(&x.iter().map(|&v| v as f64).collect::<Vec<_>>(), y)).collect();
        (c.iter().copied().fold(f64::INFINITY, f64::min), c.iter().sum::<f64>() / c.len() as f64)
    }

    /// #182: the whole block on the 3-bit container + the MTP overlay ([`load_mtp`]: layer 45 and
    /// the trunk's lm_head only, ~4 GB of reads) against the oracle golden `ref-mul1-mtp`, which
    /// took the same 288 records but the 25 other tensors from the FP8 originals (dequantized to
    /// f32). Same inputs (`mtp-embed`, `mtp-h`), same call plan (prompt rows in one call, decode
    /// rows singly), SGLang pairing (`Pos0::Keep`). The overlay's NVFP4 attention and shared expert
    /// are not the FP8 weights, so the rows are not bit-equal; the thresholds were fixed before the
    /// first run (#182 implementation comment): the eh stage (BF16 weights on both sides) cosine
    /// >= 0.9999 on every row; head-norm cosine mean >= 0.98 and min >= 0.90; draft top-1 equal
    /// to the golden's draft on >= 75 % of the rows. Prints every stage, the routing and DSA overlap
    /// and the draft-vs-trunk agreement (golden: 55 / 89).
    #[test]
    #[ignore = "needs the GPU, the golden, the container and the overlay: GLM5_MTP_GOLDEN=<ref-mul1-mtp> GLM5_CNQ=<MUL1K3.cnq> GLM5_MTP_OVERLAY=<MTP-overlay.cnq> cargo test --release --lib glm5_mtp_block_on_the_overlay -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_block_on_the_overlay_matches_the_oracle_golden() {
        let env = |k: &str| std::env::var(k).unwrap_or_else(|_| panic!("{k}"));
        let gd = std::path::PathBuf::from(env("GLM5_MTP_GOLDEN"));
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(gd.join("manifest.json")).unwrap()).unwrap();
        let (rows, prompt) = (man["rows"].as_u64().unwrap() as usize, man["prompt_rows"].as_u64().unwrap() as usize);
        assert_eq!((man["primary_variant"].as_str(), man["draft_steps"].as_u64(), man["prompt_chunk"].as_u64()), (Some("sglang"), Some(1), Some(0)));
        let hd = G.hidden;
        let rd = |f: &str| read_f32(&gd.join(f), rows * hd);
        let (e, h) = (rd("mtp-embed.f32"), rd("mtp-h.f32"));
        unsafe {
            let _ctx = cuda::Ctx::init();
            let t0 = std::time::Instant::now();
            let mut base = Cnq::open(&env("GLM5_CNQ"));
            let mut ov = Cnq::open(&env("GLM5_MTP_OVERLAY"));
            let moe = MoeGeo::new(&G, ExpertRecordSpec::new(ExpertCodec::Mul1, crate::cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
            let mut blk = load_mtp(&mut base, &mut ov, &G, &moe).unwrap_or_else(|e| panic!("{e}"));
            let mut rep = crate::glm5_model::LoadReport::default();
            let mut head_w = crate::glm5_model::load_head(&mut base, &G, &mut rep);
            eprintln!(
                "glm5_mtp load: block {} B (scale bytes rewritten {}, kv_b inexact {}), lm_head + norm {} B, {:.1} s",
                blk.bytes, blk.sanitized, blk.kv_b_inexact, rep.bytes, t0.elapsed().as_secs_f64()
            );
            let kn = Glm5Kernels::new(&G);
            let mk = MtpKernels::new();
            let mut head = crate::glm5_head::Head::new(crate::glm5_model::head_geo(&G));
            let mut pass = MtpPass::new(&G, moe, prompt, rows + 8);
            let (mut ed, mut hdv) = (cuda::to_f32_dev(&e), cuda::to_f32_dev(&h));
            let v = G.vocab;
            let mut logits = cuda::alloc_named("glm5 MTP test logits", prompt * v * 4);
            let mut ids = cuda::alloc_named("glm5 MTP test ids", prompt * 4);
            let mut taps = MtpTaps::default();
            let (mut normed, mut top1, mut routing) = (Vec::new(), Vec::new(), Vec::new());
            let calls: Vec<(usize, usize)> = std::iter::once((0, prompt)).chain((prompt..rows).map(|r| (r, 1))).collect();
            for &(p0, t) in &calls {
                let off = (p0 * hd * 4) as u64;
                blk.call(&mut pass, &kn, &mk, ed + off, hdv + off, p0, t, false, Some(&mut taps));
                head.lm_head(&kn.k, head_w.lm, pass.normed, logits, t);
                head.argmax(&kn.k, logits, ids, t);
                cuda::sync();
                normed.extend(cuda::dtoh(pass.normed, t * hd));
                top1.extend(cuda::dtoh_i32(ids, t));
                let r = pass.routing();
                routing.extend(r.ids.chunks(r.topk).map(|c| c.to_vec()));
            }
            let stage = |name: &str, got: &[f32], file: &str| -> (f64, f64) {
                let (mn, mean) = row_cos(got, &rd(file), hd);
                eprintln!("glm5_mtp {name:<10} vs {file:<18}: cosine min {mn:.6}, mean {mean:.6} ({rows} rows)");
                (mn, mean)
            };
            let (eh_min, _) = stage("eh", &taps.eh, "mtp-eh.f32");
            stage("attn_out", &taps.attn_out, "mtp-attn-out.f32");
            stage("ffn_in", &taps.ffn_in, "mtp-ffn-in.f32");
            stage("ffn_out", &taps.ffn_out, "mtp-ffn-out.f32");
            stage("out", &taps.out, "mtp-out.f32");
            let (hn_min, hn_mean) = stage("head_norm", &normed, "mtp-head-norm.f32");
            let (ro, rsame) = crate::glm5_model::golden::routing_overlap(&routing, &read_i32(&gd.join("mtp-routing-ids.i32")), G.topk);
            let gsel = read_i32(&gd.join("mtp-dsa-topk.i32"));
            let (so, ssame) = crate::glm5_model::golden::dsa_overlap(&taps.selection, &gsel, gsel.len() / rows);
            eprintln!("glm5_mtp routing overlap {ro:.4} ({rsame} / {rows} rows identical), DSA selection overlap {so:.4} ({ssame} / {rows} identical)");
            let gdraft = read_i32(&gd.join("mtp-draft-top1.i32"));
            let trunk = read_i32(&gd.join("trunk-next-top1.i32"));
            let same_draft = (0..rows).filter(|&r| top1[r] == gdraft[r]).count();
            let ours: Vec<u32> = top1.iter().map(|&x| x as u32).collect();
            let theirs: Vec<u32> = trunk.iter().map(|&x| x as u32).collect();
            let gold: Vec<u32> = gdraft.iter().map(|&x| x as u32).collect();
            let (agree, n) = agreement(&ours, &theirs);
            let (gagree, _) = agreement(&gold, &theirs);
            let (pa, _) = agreement(&ours[..prompt], &theirs[..prompt]);
            eprintln!(
                "glm5_mtp draft top-1 = golden draft top-1 on {same_draft} / {rows}; draft = trunk's next on {agree} / {n} (prompt {pa} / {prompt}, decode {} / {}); golden {gagree} / {n}",
                agree - pa,
                rows - prompt
            );
            assert!(eh_min >= 0.9999, "eh stage: cosine min {eh_min}");
            assert!(hn_mean >= 0.98 && hn_min >= 0.90, "head norm: cosine mean {hn_mean}, min {hn_min}");
            assert!(same_draft * 4 >= rows * 3, "draft top-1 = golden's on {same_draft} / {rows}");
            pass.free();
            blk.free();
            head.free();
            for d in [&mut ed, &mut hdv, &mut logits, &mut ids, &mut head_w.norm, &mut head_w.lm] {
                cuda::free_dev(d);
            }
        }
    }
}
