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
//! holds - only the 288 MUL1 experts - and what an overlay would cost), and the record loader of the
//! container's `mtp` section ([`load_mtp_records`]). Nothing in the engine calls it yet: the decode
//! loop's speculative step (`gen.rs` `spec_step`, Flash-Next/27B) and a glm5 caller are the lead's
//! wiring; `docs/glm5-mtp.md` "Integration" lists the hooks.

use crate::cnq::Cnq;
use crate::cuda;
use crate::geo::Glm5Geo;
use crate::glm5_mla::{MlaCache, MlaDims, MlaProj, MlaScratch, MlaWeights, RMS_EPS};
use crate::glm5_model::{expert_names, Glm5Kernels};
use crate::glm5_moe::{GpuMoePlan, GpuMoeWeights, MoeGeo};
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
        assert!((1..=self.max_t).contains(&t) && pos0 + t <= self.cap, "glm5_mtp: call rows {pos0}..{} (max_t {}, cap {})", pos0 + t, self.max_t, self.cap);
        let hd = self.g.hidden;
        let bytes = t * hd * 4;
        cuda::to_i32_into(self.st, &[pos0 as i32, t as i32]);
        launch_v(mk.eh_norm, t as u32, 1, 1, 256, &[e, h, w.enorm, w.hnorm, self.eh, hd as u64, (self.g.rms_eps as f32).to_bits() as u64, zero_pos0 as u64, self.st]);
        self.mla_sc.begin(pos0, t);
        self.mla_sc.linear(&kn.mla, w.eh_proj, 2 * hd, hd, self.eh, 2 * hd, self.x, hd);
        cuda::d2d_async(self.xn, self.x, bytes);
        kn.mla.rmsnorm_rows(self.xn, w.input_norm, hd, t, self.st);
        match proj {
            Some(p) => self.mla_sc.forward_with(&kn.mla, &w.attn, &self.mla_c, self.xn, self.sub, pos0, t, p),
            None => self.mla_sc.forward(&kn.mla, &w.attn, &self.mla_c, self.xn, self.sub, pos0, t),
        }
        let n = (t * hd) as u64;
        launch_v(mk.add, n.div_ceil(256) as u32, 1, 1, 256, &[self.sub, self.x, n]);
        cuda::d2d_async(self.xn, self.x, bytes);
        kn.mla.rmsnorm_rows(self.xn, w.post_norm, hd, t, self.st);
        if !self.moe_plans.iter().any(|p| p.tokens == t) {
            self.moe_plans.push(GpuMoePlan::new(&self.moe, t));
        }
        let p = self.moe_plans.iter().find(|p| p.tokens == t).unwrap();
        p.run(&kn.k, &kn.mul1, &kn.moe, &w.moe, w.table, self.xn, self.sub);
        self.last_t = t;
        launch_v(mk.add, n.div_ceil(256) as u32, 1, 1, 256, &[self.sub, self.x, n]);
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
}
