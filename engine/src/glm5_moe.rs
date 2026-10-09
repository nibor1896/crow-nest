//! crow-nest #164 (GLM-5.3-Flash plan step 13d / 15): the glm5_next FFN sub-block - the sigmoid
//! router with the selection-only `e_score_correction_bias`, the routed experts as MUL1 records
//! (#181) with the SwiGLU clamp, the shared expert and the dense FFN of layers 0-2 (NVFP4).
//! Nothing in the engine calls this yet: `gen.rs` / `boot.rs` wire it with the rest of the
//! glm5_next arms (the lead integrates). Page: `docs/glm5-moe.md`.
//!
//! # Math of record (transformers 5.16.1 `modeling_glm5_next.py`, docs/glm5-next-recipe.md 9-10)
//!
//! - Router (`M:158-183`): `logits = x W_r^T` in f32 (W_r BF16 [E][H]), `s = sigmoid(logits)`,
//!   `c = s + e_score_correction_bias` (f32) for the CHOICE only; `n_group` = `topk_group` = 1, so
//!   the group stage keeps every expert; `ids = top-K of c`; `w = s[ids]` (the unbiased score),
//!   `w = w / (sum w + 1e-20)`, `w = w * routed_scaling` (2.5). E = 288, K = 8.
//! - SwiGLU clamp in every FFN (`M:98-104`, `M:137-142`): `g = min(g, L)`, `u = clamp(u, -L, L)`,
//!   `h = silu(g) * u`, L = `swiglu_limit` 10. Only the gate's upper side is clamped. The clamp is
//!   torch's: a NaN stays NaN.
//! - MoE (`M:200-207`): `y = sum_k w_k * down_k(h_k(x)) + shared(x)`, the shared expert a plain
//!   MLP of width `expert_inter * shared_experts` (2048) without a gate. Dense layers 0-2: the same
//!   MLP at `dense_inter` (12,288).
//!
//! # Lanes
//!
//! - **CPU** ([`moe_cpu`], [`ffn_nvfp4_cpu`], [`expert_ffn_mul1_cpu`]): the routed experts on
//!   `cpu_mul1::gemv` (#180/#183, AVX2), the shared and dense FFN on `cpu_nvfp4::gemv` (#173),
//!   the clamp in between ([`swiglu_clamp`]). A record comes from an [`ExpertRecords`] source: the
//!   hook for the pinned tier, a RAM copy, or an NVMe fetch (#149) that delivered into a host buffer.
//! - **GPU** ([`GpuMoePlan`], [`GpuFfnPlan`]): router logits on the engine's `gemv_bf16_b`, the
//!   selection on `glm5_router_sig_topk` (`kernels_glm5_moe.cu`, sized by the plan: E <= 512,
//!   padded to 512 threads with -inf so the power-of-two reduction holds for E = 288), the
//!   experts on `kernels::mul1::GemvPlan` (gate, up, `glm5_swiglu_clamp`, down) with one slot per
//!   (token, k) combo, the shared and dense FFN on the engine's `gemv_fp4_b` with the same clamp
//!   kernel, `glm5_moe_combine` last. The record base of every expert comes from a device table
//!   `[E] u64` the caller owns: a VRAM slot or a pinned-host UVA address, as `gemv_fp4_ptrb` reads
//!   them - the hook for the residency and the dynamic cache (#175). No host sync inside `run`.
//!
//! Ties in the selection go to the lowest expert index (the rule of `router_top10`); torch's
//! `topk` gives no order among equal values, so routing is compared as a set per token, and the
//! oracle comparison reports the smallest 8th/9th choice gap it saw.

use crate::cpu_mul1::{self, Bitrate, Mul1Expert};
use crate::cpu_nvfp4::{self, ExpertBlock, Path};
use crate::cuda;
use crate::geo::{ExpertCodec, ExpertRecordSpec, Glm5Geo};
use crate::kernels::{self, launch_v, mul1};
use cudarc::driver::sys::CUdeviceptr;

/// The FFN geometry of one glm5_next model, from its [`Glm5Geo`] and its container's routed-expert
/// record ([`ExpertRecordSpec`], `nvme_source::glm5_record_of_container`).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MoeGeo {
    pub hidden: usize,
    pub experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    /// `expert_inter * shared_experts`
    pub shared_inter: usize,
    pub dense_inter: usize,
    pub routed_scaling: f32,
    pub swiglu_limit: f32,
    pub record: ExpertRecordSpec,
    /// the MUL1 bitrate whose record has `record.bytes` (K = 3: 9,474,048 B at GLM shapes)
    pub bitrate: Bitrate,
}

/// the bitrates a MUL1 record can have (`cpu_mul1::Bitrate::from_k`)
const MUL1_KS: [f64; 11] = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0];

/// The MUL1 bitrate of a `bytes`-long record of one `[hidden, inter]` expert, refused by name
/// when no bitrate gives that size.
pub fn mul1_bitrate(hidden: usize, inter: usize, bytes: u64) -> Result<Bitrate, String> {
    for k in MUL1_KS {
        let b = Bitrate::from_k(k)?;
        if cpu_mul1::record_bytes(hidden, inter, b) as u64 == bytes {
            return Ok(b);
        }
    }
    Err(format!(
        "#164: a MUL1 record of {bytes} B fits no bitrate at H {hidden} I {inter} (K = 3 is {} B, #181)",
        cpu_mul1::record_bytes(hidden, inter, Bitrate { bits: 3, half: false })
    ))
}

impl MoeGeo {
    pub fn new(g: &Glm5Geo, record: ExpertRecordSpec) -> Result<MoeGeo, String> {
        if record.codec != ExpertCodec::Mul1 {
            return Err(format!(
                "#164: the glm5_next routed-expert lane computes MUL1 records; this container's experts are {} ({} B) - refusing",
                record.codec.dtype(),
                record.bytes
            ));
        }
        if g.experts > kernels::glm5_moe::ROUTER_THREADS || g.topk == 0 || g.topk > kernels::glm5_moe::MAXK || g.topk > g.experts {
            return Err(format!(
                "#164: {} experts top-{}: the router takes 1 <= K <= {} <= E <= {}",
                g.experts,
                g.topk,
                kernels::glm5_moe::MAXK,
                kernels::glm5_moe::ROUTER_THREADS
            ));
        }
        let bitrate = mul1_bitrate(g.hidden, g.expert_inter, record.bytes)?;
        Ok(MoeGeo {
            hidden: g.hidden,
            experts: g.experts,
            topk: g.topk,
            expert_inter: g.expert_inter,
            shared_inter: g.expert_inter * g.shared_experts,
            dense_inter: g.dense_inter,
            routed_scaling: g.routed_scaling as f32,
            swiglu_limit: g.swiglu_limit as f32,
            record,
            bitrate,
        })
    }

    /// gate, up, down of one record (offsets for `kernels::mul1`)
    pub fn record_specs(&self) -> [mul1::MatSpec; 3] {
        mul1::record_specs(self.hidden, self.expert_inter, self.bitrate.bits, self.bitrate.half)
    }
}

// ---------------------------------------------------------------- exact ops (the CPU twins)

/// `1 / (1 + exp(-x))` in f32 (the router kernel's order)
#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// `silu(min(g, l)) * clamp(u, -l, l)` with `silu(v) = v / (1 + exp(-v))`, f32, NaN-propagating
/// like `torch.clamp` (`glm5_swiglu_clamp`)
#[inline]
pub fn swiglu_clamp(g: f32, u: f32, l: f32) -> f32 {
    let g = if g > l { l } else { g };
    let u = if u > l {
        l
    } else if u < -l {
        -l
    } else {
        u
    };
    (g / (1.0 + (-g).exp())) * u
}

/// BF16 bits to f32 (exact)
#[inline]
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// The routing of `tokens` rows: `ids` and `weights` `[T][topk]` in pick order (descending choice
/// score, ties to the lowest expert).
#[derive(Clone, Debug, PartialEq)]
pub struct Routing {
    pub topk: usize,
    pub ids: Vec<u32>,
    pub weights: Vec<f32>,
}

impl Routing {
    pub fn tokens(&self) -> usize {
        self.ids.len() / self.topk
    }

    /// the ids of row `t` (what `ExpertCache::observe_token` takes)
    pub fn ids_of(&self, t: usize) -> &[u32] {
        &self.ids[t * self.topk..(t + 1) * self.topk]
    }

    /// every expert some row routes to, ascending: the records a layer needs before it runs (the
    /// NVMe fetch list once the cache says which are not resident)
    pub fn needed(&self) -> Vec<u32> {
        let mut v = self.ids.clone();
        v.sort_unstable();
        v.dedup();
        v
    }

    /// row `t` as (ids ascending, their weights): the oracle's canonical order
    pub fn sorted_row(&self, t: usize) -> (Vec<u32>, Vec<f32>) {
        let k = self.topk;
        let mut p: Vec<(u32, f32)> = (0..k).map(|j| (self.ids[t * k + j], self.weights[t * k + j])).collect();
        p.sort_by_key(|q| q.0);
        p.into_iter().unzip()
    }
}

/// The selection of `glm5_router_sig_topk` on the host: `logits` `[T][E]`, `bias` `[E]`.
pub fn route_from_logits(geo: &MoeGeo, logits: &[f32], bias: &[f32]) -> Routing {
    let (e, k) = (geo.experts, geo.topk);
    assert_eq!(bias.len(), e, "glm5_moe: bias is not [{e}]");
    assert!(logits.len() % e == 0, "glm5_moe: logits are not [T][{e}]");
    let tokens = logits.len() / e;
    let (mut ids, mut weights) = (Vec::with_capacity(tokens * k), Vec::with_capacity(tokens * k));
    let mut s = vec![0f32; e];
    let mut c = vec![0f32; e];
    let mut taken = vec![false; e];
    for row in logits.chunks_exact(e) {
        for i in 0..e {
            s[i] = sigmoid(row[i]);
            let v = s[i] + bias[i];
            c[i] = if v.is_nan() { f32::NEG_INFINITY } else { v };
            taken[i] = false;
        }
        let mut w = [0f32; kernels::glm5_moe::MAXK];
        for wj in w.iter_mut().take(k) {
            // ties (and a row of -inf) to the lowest untaken expert
            let mut best = usize::MAX;
            for i in 0..e {
                if !taken[i] && (best == usize::MAX || c[i] > c[best]) {
                    best = i;
                }
            }
            taken[best] = true;
            ids.push(best as u32);
            *wj = s[best];
        }
        let mut sum = 0f32;
        for &v in &w[..k] {
            sum += v;
        }
        let den = sum + 1e-20f32;
        for &v in &w[..k] {
            weights.push((v / den) * geo.routed_scaling);
        }
    }
    Routing { topk: k, ids, weights }
}

/// The router weights of one MoE layer: `mlp.gate.weight` BF16 `[E][H]`, `e_score_correction_bias` f32 `[E]`.
#[derive(Clone, Copy, Debug)]
pub struct RouterWeights<'a> {
    pub weight: &'a [u16],
    pub bias: &'a [f32],
}

/// `logits[t][e] = sum_i x[t][i] W[e][i]` with f64 accumulation, one rounding to f32
pub fn router_logits(geo: &MoeGeo, r: &RouterWeights, x: &[f32]) -> Vec<f32> {
    let (h, e) = (geo.hidden, geo.experts);
    assert_eq!(r.weight.len(), e * h, "glm5_moe: router weight is not [{e}][{h}]");
    assert!(x.len() % h == 0, "glm5_moe: x is not [T][{h}]");
    let mut out = Vec::with_capacity(x.len() / h * e);
    for xt in x.chunks_exact(h) {
        for wr in r.weight.chunks_exact(h) {
            let mut acc = 0f64;
            for (&w, &v) in wr.iter().zip(xt) {
                acc += bf16_to_f32(w) as f64 * v as f64;
            }
            out.push(acc as f32);
        }
    }
    out
}

pub fn route(geo: &MoeGeo, r: &RouterWeights, x: &[f32]) -> Routing {
    route_from_logits(geo, &router_logits(geo, r, x), r.bias)
}

// ---------------------------------------------------------------- CPU lane

/// The clamped SwiGLU FFN `down(silu(min(gate x, l)) * clamp(up x, -l, l))` of NVFP4 matrices
/// (gate, up `[inter, hidden]`, down `[hidden, inter]`): the shared expert and the dense layers.
pub fn ffn_nvfp4_cpu(f: &ExpertBlock, limit: f32, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    let tokens = x.len() / f.hidden;
    let (mut g, mut u) = (vec![0f32; tokens * f.inter], vec![0f32; tokens * f.inter]);
    cpu_nvfp4::gemv(&f.gate, x, &mut g, threads, path);
    cpu_nvfp4::gemv(&f.up, x, &mut u, threads, path);
    let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu_clamp(g, u, limit)).collect();
    cpu_nvfp4::gemv(&f.down, &h, y, threads, path);
}

/// One routed expert of one MUL1 record (#181 layout) with the clamp: `x`, `y` `[T][hidden]`.
pub fn expert_ffn_mul1_cpu(geo: &MoeGeo, rec: &[u8], x: &[f32], y: &mut [f32], threads: usize, path: Path) -> Result<(), String> {
    let e = Mul1Expert::from_record(rec, geo.hidden, geo.expert_inter, geo.bitrate)?;
    let tokens = x.len() / geo.hidden;
    let (mut g, mut u) = (vec![0f32; tokens * geo.expert_inter], vec![0f32; tokens * geo.expert_inter]);
    cpu_mul1::gemv(&e.gate, x, &mut g, threads, path);
    cpu_mul1::gemv(&e.up, x, &mut u, threads, path);
    let h: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| swiglu_clamp(g, u, geo.swiglu_limit)).collect();
    cpu_mul1::gemv(&e.down, &h, y, threads, path);
    Ok(())
}

/// Where the CPU lane reads a routed expert's record (`record.bytes` long, #181 layout): the
/// pinned tier, a RAM copy, or a host buffer an NVMe fetch (#149, `ColdSource`) filled.
pub trait ExpertRecords {
    fn record(&self, expert: u32) -> Result<&[u8], String>;
}

/// The host weights of one MoE layer.
#[derive(Clone, Copy, Debug)]
pub struct MoeLayerCpu<'a> {
    pub router: RouterWeights<'a>,
    pub shared: ExpertBlock<'a>,
}

/// `y = sum_k w_k * y_k + ys` per row, `k` in pick order (`glm5_moe_combine`)
pub fn combine(topk: usize, ye: &[f32], weights: &[f32], ys: &[f32], y: &mut [f32]) {
    let h = ys.len() / (weights.len() / topk);
    for (t, yt) in y.chunks_exact_mut(h).enumerate() {
        for (j, v) in yt.iter_mut().enumerate() {
            let mut acc = 0f32;
            for k in 0..topk {
                acc += weights[t * topk + k] * ye[(t * topk + k) * h + j];
            }
            *v = acc + ys[t * h + j];
        }
    }
}

/// One MoE layer on the CPU: route, every needed expert once over the rows that chose it, the
/// shared expert, the combine. Returns the routing (for the cache counters and the dumps).
pub fn moe_cpu(geo: &MoeGeo, layer: &MoeLayerCpu, records: &dyn ExpertRecords, x: &[f32], y: &mut [f32], threads: usize, path: Path) -> Result<Routing, String> {
    let (h, k) = (geo.hidden, geo.topk);
    assert!(x.len() % h == 0 && y.len() == x.len(), "glm5_moe::moe_cpu: x, y are not [T][{h}]");
    if layer.shared.inter != geo.shared_inter || layer.shared.hidden != h {
        return Err(format!("#164: shared expert [{}, {}], the geometry says [{}, {h}]", layer.shared.inter, layer.shared.hidden, geo.shared_inter));
    }
    let r = route(geo, &layer.router, x);
    let mut ys = vec![0f32; x.len()];
    ffn_nvfp4_cpu(&layer.shared, geo.swiglu_limit, x, &mut ys, threads, path);
    let mut ye = vec![0f32; x.len() * k];
    for e in r.needed() {
        let combos: Vec<usize> = (0..r.ids.len()).filter(|&c| r.ids[c] == e).collect();
        let xs: Vec<f32> = combos.iter().flat_map(|&c| x[c / k * h..(c / k + 1) * h].iter().copied()).collect();
        let mut out = vec![0f32; xs.len()];
        let rec = records.record(e)?;
        if rec.len() as u64 != geo.record.bytes {
            return Err(format!("#164: expert {e}: record of {} B, the container says {} B", rec.len(), geo.record.bytes));
        }
        expert_ffn_mul1_cpu(geo, rec, &xs, &mut out, threads, path)?;
        for (n, &c) in combos.iter().enumerate() {
            ye[c * h..(c + 1) * h].copy_from_slice(&out[n * h..(n + 1) * h]);
        }
    }
    combine(k, &ye, &r.weights, &ys, y);
    Ok(r)
}

// ---------------------------------------------------------------- GPU lane

/// One NVFP4 matrix on the device: the engine layout (`gemv_fp4_b`) and its global scale as a
/// device f32. Scale bytes are read raw (0x7F would read as 480): pass sanitized bytes.
#[derive(Clone, Copy, Debug)]
pub struct GpuNvfp4 {
    pub w: CUdeviceptr,
    pub gs: CUdeviceptr,
    pub rows: usize,
    pub cols: usize,
}

/// gate, up `[inter, hidden]`, down `[hidden, inter]`
#[derive(Clone, Copy, Debug)]
pub struct GpuFfnWeights {
    pub gate: GpuNvfp4,
    pub up: GpuNvfp4,
    pub down: GpuNvfp4,
}

/// the device weights of one MoE layer besides its routed experts
#[derive(Clone, Copy, Debug)]
pub struct GpuMoeWeights {
    /// `mlp.gate.weight` BF16 `[E][H]`
    pub router: CUdeviceptr,
    /// `e_score_correction_bias` f32 `[E]`
    pub bias: CUdeviceptr,
    pub shared: GpuFfnWeights,
}

unsafe fn i32_dev(v: &[usize]) -> CUdeviceptr {
    cuda::to_i32_dev(&v.iter().map(|&x| i32::try_from(x).expect("glm5_moe: parameter beyond i32")).collect::<Vec<_>>())
}

/// The clamped SwiGLU FFN of NVFP4 matrices for `tokens` rows (shared expert, dense layers).
pub struct GpuFfnPlan {
    pub hidden: usize,
    pub inter: usize,
    pub tokens: usize,
    prm_kh: CUdeviceptr,
    prm_ki: CUdeviceptr,
    prm_n: CUdeviceptr,
    prm_f: CUdeviceptr,
    pub g: CUdeviceptr,
    pub u: CUdeviceptr,
    pub h: CUdeviceptr,
}

impl GpuFfnPlan {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(hidden: usize, inter: usize, tokens: usize, limit: f32) -> GpuFfnPlan {
        assert!(hidden % 64 == 0 && inter % 64 == 0 && tokens > 0, "glm5_moe: FFN [{inter}, {hidden}] x {tokens}");
        GpuFfnPlan {
            hidden,
            inter,
            tokens,
            prm_kh: i32_dev(&[hidden]),
            prm_ki: i32_dev(&[inter]),
            prm_n: i32_dev(&[tokens * inter]),
            prm_f: cuda::to_f32_dev(&[0.0, limit]),
            g: cuda::alloc_zeroed(tokens * inter * 4),
            u: cuda::alloc_zeroed(tokens * inter * 4),
            h: cuda::alloc_zeroed(tokens * inter * 4),
        }
    }

    /// queue `y = ffn(x)`, x, y `[T][hidden]` f32 (four launches on the current stream)
    ///
    /// # Safety
    /// `kn` comes from a module with `gemv_fp4_b`; the weights match the plan's shape.
    pub unsafe fn run(&self, kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuFfnWeights, x: CUdeviceptr, y: CUdeviceptr) {
        let (h, i, t) = (self.hidden, self.inter, self.tokens);
        assert!((w.gate.rows, w.gate.cols, w.up.rows, w.up.cols, w.down.rows, w.down.cols) == (i, h, i, h, h, i), "glm5_moe: FFN weights do not fit the plan");
        let fp4 = kn.f("gemv_fp4_b");
        launch_v(fp4, i as u32, t as u32, 1, 256, &[w.gate.w, x, w.gate.gs, self.g, self.prm_kh]);
        launch_v(fp4, i as u32, t as u32, 1, 256, &[w.up.w, x, w.up.gs, self.u, self.prm_kh]);
        launch_v(gk.act, (t * i).div_ceil(256) as u32, 1, 1, 256, &[self.g, self.u, self.h, self.prm_n, self.prm_f]);
        launch_v(fp4, h as u32, t as u32, 1, 256, &[w.down.w, self.h, w.down.gs, y, self.prm_ki]);
    }

    /// # Safety
    /// No launch of this plan is pending.
    pub unsafe fn free(&mut self) {
        for d in [&mut self.prm_kh, &mut self.prm_ki, &mut self.prm_n, &mut self.prm_f, &mut self.g, &mut self.u, &mut self.h] {
            cuda::free_dev(d);
        }
    }
}

/// One MoE layer on the GPU for `tokens` rows: `tokens * topk` combos, one MUL1 slot each.
pub struct GpuMoePlan {
    pub geo: MoeGeo,
    pub tokens: usize,
    prm_kh: CUdeviceptr,
    prm_route: CUdeviceptr,
    prm_f: CUdeviceptr,
    prm_kh2: CUdeviceptr,
    prm_n: CUdeviceptr,
    /// `[T][E]` router logits
    pub logits: CUdeviceptr,
    /// `[T][K]` i32 expert ids, pick order
    pub ids: CUdeviceptr,
    /// `[T][K]` f32 routing weights
    pub wts: CUdeviceptr,
    /// `[T * K]` u64 record bases of the combos
    pub ptrs: CUdeviceptr,
    xg: CUdeviceptr,
    ge: CUdeviceptr,
    ue: CUdeviceptr,
    /// `[T * K][inter]` the clamped activation the down GEMV read
    pub he: CUdeviceptr,
    /// `[T * K][H]` the routed experts' outputs (unweighted)
    pub ye: CUdeviceptr,
    /// `[T][H]` the shared expert's output
    pub ys: CUdeviceptr,
    pub shared: GpuFfnPlan,
    gate: mul1::GemvPlan,
    up: mul1::GemvPlan,
    down: mul1::GemvPlan,
}

impl GpuMoePlan {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(geo: &MoeGeo, tokens: usize) -> GpuMoePlan {
        let (h, e, k, i) = (geo.hidden, geo.experts, geo.topk, geo.expert_inter);
        let c = tokens * k;
        assert!(tokens > 0 && e <= kernels::glm5_moe::ROUTER_THREADS && (1..=kernels::glm5_moe::MAXK).contains(&k) && k <= e);
        let [sg, su, sd] = geo.record_specs();
        GpuMoePlan {
            geo: *geo,
            tokens,
            prm_kh: i32_dev(&[h]),
            prm_route: i32_dev(&[e, k]),
            prm_f: cuda::to_f32_dev(&[geo.routed_scaling, geo.swiglu_limit]),
            prm_kh2: i32_dev(&[k, h]),
            prm_n: i32_dev(&[c * i]),
            logits: cuda::alloc_zeroed(tokens * e * 4),
            ids: cuda::alloc_zeroed(c * 4),
            wts: cuda::alloc_zeroed(c * 4),
            ptrs: cuda::alloc_zeroed(c * 8),
            xg: cuda::alloc_zeroed(c * h * 4),
            ge: cuda::alloc_zeroed(c * i * 4),
            ue: cuda::alloc_zeroed(c * i * 4),
            he: cuda::alloc_zeroed(c * i * 4),
            ye: cuda::alloc_zeroed(c * h * 4),
            ys: cuda::alloc_zeroed(tokens * h * 4),
            shared: GpuFfnPlan::new(h, geo.shared_inter, tokens, geo.swiglu_limit),
            gate: mul1::GemvPlan::new(sg, c, 1),
            up: mul1::GemvPlan::new(su, c, 1),
            down: mul1::GemvPlan::new(sd, c, 1),
        }
    }

    /// queue the layer: `y = moe(x)`, x, y `[T][H]` f32 (13 launches on the current stream, no
    /// host sync, graph-capturable).
    ///
    /// # Safety
    /// `kn` comes from a module with `gemv_bf16_b` and `gemv_fp4_b`; `table` is a device `[E]` u64
    /// array whose every entry is a record base (`geo.record.bytes`, #181 layout) that stays
    /// readable until the launches finished - a VRAM slot or a pinned-host UVA address.
    pub unsafe fn run(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        table: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) {
        let (h, e, t) = (self.geo.hidden, self.geo.experts, self.tokens);
        let c = t * self.geo.topk;
        launch_v(kn.f("gemv_bf16_b"), e as u32, t as u32, 1, 256, &[w.router, x, self.logits, self.prm_kh]);
        launch_v(gk.router, t as u32, 1, 1, kernels::glm5_moe::ROUTER_THREADS as u32, &[self.logits, w.bias, self.ids, self.wts, self.prm_route, self.prm_f]);
        launch_v(gk.gather, h.div_ceil(256) as u32, c as u32, 1, 256, &[self.ids, table, x, self.ptrs, self.xg, self.prm_kh2]);
        self.gate.run(mk, self.ptrs, self.xg, self.ge);
        self.up.run(mk, self.ptrs, self.xg, self.ue);
        launch_v(gk.act, (c * self.geo.expert_inter).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
        self.down.run(mk, self.ptrs, self.he, self.ye);
        self.shared.run(kn, gk, &w.shared, x, self.ys);
        launch_v(gk.combine, h.div_ceil(256) as u32, t as u32, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
    }

    /// the routing of the last `run` (synchronizes)
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn read_routing(&self) -> Routing {
        cuda::sync();
        let c = self.tokens * self.geo.topk;
        Routing { topk: self.geo.topk, ids: cuda::dtoh_i32(self.ids, c).into_iter().map(|v| v as u32).collect(), weights: cuda::dtoh(self.wts, c) }
    }

    /// # Safety
    /// No launch of this plan is pending.
    pub unsafe fn free(&mut self) {
        for d in [
            &mut self.prm_kh,
            &mut self.prm_route,
            &mut self.prm_f,
            &mut self.prm_kh2,
            &mut self.prm_n,
            &mut self.logits,
            &mut self.ids,
            &mut self.wts,
            &mut self.ptrs,
            &mut self.xg,
            &mut self.ge,
            &mut self.ue,
            &mut self.he,
            &mut self.ye,
            &mut self.ys,
        ] {
            cuda::free_dev(d);
        }
        self.shared.free();
        self.gate.free();
        self.up.free();
        self.down.free();
    }
}

#[cfg(test)]
mod tests {
    //! Exact-op tests (host), the oracle comparison on synthetic weights with GLM block shapes
    //! (host, `cargo test --release --lib glm5_moe`), the oracle input writer and the GPU tests
    //! (`#[ignore]`). Goldens: `engine/tests/fixtures/glm5/moe/`, written by
    //! `oracle/export_glm5_moe_golden.py` from the inputs `glm5_moe_write_oracle_inputs` dumps.
    use super::*;
    use crate::cpu_mul1::testkit::{self, Rng};
    use sha2::{Digest, Sha256};
    use std::collections::BTreeMap;

    fn geo() -> MoeGeo {
        MoeGeo::new(&Glm5Geo::GLM_5_3_FLASH, ExpertRecordSpec::new(ExpertCodec::Mul1, cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap()
    }

    /// a small geometry for the exact-op tests (E not a power of two)
    fn tiny(e: usize, k: usize) -> MoeGeo {
        MoeGeo { experts: e, topk: k, ..geo() }
    }

    // ---------------------------------------------------------------- exact ops

    #[test]
    fn glm5_moe_geo_takes_the_mul1_record_and_refuses_the_rest() {
        let g = geo();
        assert_eq!((g.hidden, g.experts, g.topk, g.expert_inter, g.shared_inter, g.dense_inter), (4096, 288, 8, 2048, 2048, 12288));
        assert_eq!((g.routed_scaling, g.swiglu_limit), (2.5, 10.0));
        assert_eq!(g.bitrate, Bitrate { bits: 3, half: false });
        assert_eq!(g.record_specs(), mul1::record_specs(4096, 2048, 3, false));
        let nv = ExpertRecordSpec::new(ExpertCodec::Nvfp4, crate::geo::GLM5_NEXT_EXPERT_BLOCK_BYTES).unwrap();
        let why = MoeGeo::new(&Glm5Geo::GLM_5_3_FLASH, nv).unwrap_err();
        assert!(why.contains("MUL1") && why.contains("nvfp4"), "{why}");
        let odd = ExpertRecordSpec::new(ExpertCodec::Mul1, cpu_mul1::GLM_RECORD_BYTES_K3 as u64 + 4096).unwrap();
        let why = MoeGeo::new(&Glm5Geo::GLM_5_3_FLASH, odd).unwrap_err();
        assert!(why.contains("9474048"), "{why}");
        assert_eq!(mul1_bitrate(4096, 2048, cpu_mul1::record_bytes(4096, 2048, Bitrate { bits: 2, half: true }) as u64).unwrap(), Bitrate { bits: 2, half: true });
        let big = Glm5Geo { experts: 513, ..Glm5Geo::GLM_5_3_FLASH };
        assert!(MoeGeo::new(&big, g.record).is_err());
    }

    #[test]
    fn glm5_moe_swiglu_clamp_is_the_hf_rule() {
        let silu = |v: f32| v / (1.0 + (-v).exp());
        // gate clamped from above only, up on both sides
        assert_eq!(swiglu_clamp(25.0, 3.0, 10.0).to_bits(), (silu(10.0) * 3.0).to_bits());
        assert_eq!(swiglu_clamp(-25.0, 3.0, 10.0).to_bits(), (silu(-25.0) * 3.0).to_bits());
        assert_eq!(swiglu_clamp(2.0, 40.0, 10.0).to_bits(), (silu(2.0) * 10.0).to_bits());
        assert_eq!(swiglu_clamp(2.0, -40.0, 10.0).to_bits(), (silu(2.0) * -10.0).to_bits());
        assert_eq!(swiglu_clamp(10.0, -10.0, 10.0).to_bits(), (silu(10.0) * -10.0).to_bits());
        assert_eq!(swiglu_clamp(0.5, 0.25, 10.0).to_bits(), (silu(0.5) * 0.25).to_bits());
        // torch.clamp keeps a NaN
        assert!(swiglu_clamp(f32::NAN, 1.0, 10.0).is_nan());
        assert!(swiglu_clamp(1.0, f32::NAN, 10.0).is_nan());
    }

    #[test]
    fn glm5_moe_router_weights_come_from_the_unbiased_scores() {
        // 5 experts top-2: the bias moves expert 4 (lowest score) into the choice, but its weight
        // is its own sigmoid score, normalized over the two and scaled
        let g = tiny(5, 2);
        let logits = [2.0f32, 1.0, 0.5, 0.0, -3.0];
        let bias = [0.0f32, 0.0, 0.0, 0.0, 2.0];
        let r = route_from_logits(&g, &logits, &bias);
        assert_eq!(r.ids, vec![4, 0]);
        let (s4, s0) = (sigmoid(-3.0), sigmoid(2.0));
        let den = (s4 + s0) + 1e-20;
        assert_eq!(r.weights, vec![(s4 / den) * 2.5, (s0 / den) * 2.5]);
        let sum: f32 = r.weights.iter().sum();
        assert!((sum - 2.5).abs() < 1e-6, "{sum}");
    }

    #[test]
    fn glm5_moe_router_ties_go_to_the_lowest_expert_and_nan_never_wins() {
        let g = tiny(7, 3);
        let logits = [0.0f32, 1.0, 1.0, f32::NAN, 1.0, -1.0, 1.0];
        let bias = [0f32; 7];
        let r = route_from_logits(&g, &logits, &bias);
        assert_eq!(r.ids, vec![1, 2, 4]);
        // a row with every choice NaN still picks K distinct experts, lowest first
        let r = route_from_logits(&g, &[f32::NAN; 7], &bias);
        assert_eq!(r.ids, vec![0, 1, 2]);
        // 288 experts (not a power of two): the best one sits past 256
        let g = geo();
        let mut l = vec![-1f32; 288];
        for (j, e) in [287usize, 256, 3, 100, 200, 270, 1, 0].into_iter().enumerate() {
            l[e] = 5.0 - j as f32 * 0.25;
        }
        let r = route_from_logits(&g, &l, &[0f32; 288]);
        assert_eq!(r.ids, vec![287, 256, 3, 100, 200, 270, 1, 0]);
        assert_eq!(r.needed(), vec![0, 1, 3, 100, 200, 256, 270, 287]);
    }

    #[test]
    fn glm5_moe_combine_sums_in_pick_order_then_adds_shared() {
        let ye = [1.0f32, 2.0, 10.0, 20.0, 100.0, 200.0, 1000.0, 2000.0];
        let w = [0.5f32, 0.25, 2.0, 4.0];
        let ys = [7.0f32, 8.0, 9.0, 10.0];
        let mut y = [0f32; 4];
        combine(2, &ye, &w, &ys, &mut y);
        assert_eq!(y, [0.5 + 2.5 + 7.0, 1.0 + 5.0 + 8.0, 200.0 + 4000.0 + 9.0, 400.0 + 8000.0 + 10.0]);
    }

    // ---------------------------------------------------------------- synthetic weights, GLM block shapes

    /// router tokens, MoE tokens, dense tokens of the golden
    const TR: usize = 64;
    const TM: usize = 4;
    const TD: usize = 2;
    const SEED: u64 = 0x0164_6105;
    /// activations uniform in [-X_AMP, X_AMP): gate outputs ~ 6.6 rms, so part of them clamp
    const X_AMP: f32 = 0.17;
    const ROUTER_AMP: f32 = 0.55;
    const BIAS_AMP: f32 = 0.09;
    const NVFP4_GS: f32 = 0.3;
    /// expert e is the #181 synthetic record of seed `RECORD_SEED + 9 e` (trellis seeds s, s+3, s+6)
    const RECORD_SEED: u32 = 0x1640;

    fn f32_to_bf16(v: f32) -> u16 {
        let b = v.to_bits();
        ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) as u16
    }

    struct Synth {
        router_w: Vec<u16>,
        bias: Vec<f32>,
        shared: [(Vec<u8>, usize, usize); 3],
        dense: [(Vec<u8>, usize, usize); 3],
        x_route: Vec<f32>,
        x_moe: Vec<f32>,
        x_dense: Vec<f32>,
    }

    /// NVFP4 bytes of a `[rows, cols]` matrix: random codes, scale bytes 0x30..0x3F (0.5 .. 0.94),
    /// never the 0x7F code
    fn nvfp4(rows: usize, cols: usize, rng: &mut Rng) -> Vec<u8> {
        let mut b = vec![0u8; cpu_nvfp4::Nvfp4Matrix::byte_len(rows, cols)];
        for blk in b.chunks_exact_mut(36) {
            for (i, v) in blk.iter_mut().enumerate() {
                *v = if i < 4 { 0x30 + (rng.next() % 16) as u8 } else { rng.next() as u8 };
            }
        }
        b
    }

    fn synth() -> Synth {
        let g = geo();
        let (h, e) = (g.hidden, g.experts);
        let mut rng = Rng(SEED);
        let router_w = (0..e * h).map(|_| f32_to_bf16(rng.f(ROUTER_AMP))).collect();
        let bias = (0..e).map(|_| rng.f(BIAS_AMP)).collect();
        let mut mats = |inter: usize| -> [(Vec<u8>, usize, usize); 3] { [(nvfp4(inter, h, &mut rng), inter, h), (nvfp4(inter, h, &mut rng), inter, h), (nvfp4(h, inter, &mut rng), h, inter)] };
        let shared = mats(g.shared_inter);
        let dense = mats(g.dense_inter);
        let mut xs = |n: usize| -> Vec<f32> { (0..n * h).map(|_| rng.f(X_AMP)).collect() };
        let (x_route, x_moe, x_dense) = (xs(TR), xs(TM), xs(TD));
        Synth { router_w, bias, shared, dense, x_route, x_moe, x_dense }
    }

    fn block<'a>(m: &'a [(Vec<u8>, usize, usize); 3]) -> ExpertBlock<'a> {
        let mk = |i: usize| cpu_nvfp4::Nvfp4Matrix::new(&m[i].0, m[i].1, m[i].2, NVFP4_GS).unwrap();
        ExpertBlock::new(mk(0), mk(1), mk(2)).unwrap()
    }

    /// the #181 synthetic record of expert `e` (K = 3, GLM shapes)
    fn record(e: u32) -> Vec<u8> {
        let c = testkit::Case {
            name: format!("glm5-moe-e{e}"),
            source: format!("synth:{}", RECORD_SEED + 9 * e),
            bitrate: geo().bitrate,
            hidden: 4096,
            inter: 2048,
            want: [String::new(), String::new(), String::new()],
        };
        let r = testkit::record(&c);
        assert_eq!(r.len(), cpu_mul1::GLM_RECORD_BYTES_K3);
        r
    }

    struct Records(BTreeMap<u32, Vec<u8>>);
    impl ExpertRecords for Records {
        fn record(&self, e: u32) -> Result<&[u8], String> {
            self.0.get(&e).map(|v| v.as_slice()).ok_or_else(|| format!("no record for expert {e}"))
        }
    }

    fn sha(b: &[u8]) -> String {
        format!("{:x}", Sha256::digest(b))
    }

    fn le_f32(v: &[f32]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    fn le_u16(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    /// sha256 of every encoded input the engine reads (the golden's manifest repeats them)
    fn input_shas(s: &Synth) -> BTreeMap<String, String> {
        let mut m = BTreeMap::new();
        m.insert("router.bf16".into(), sha(&le_u16(&s.router_w)));
        m.insert("bias.f32".into(), sha(&le_f32(&s.bias)));
        for (name, mats) in [("shared", &s.shared), ("dense", &s.dense)] {
            for (p, mat) in ["gate", "up", "down"].iter().zip(mats.iter()) {
                m.insert(format!("{name}_{p}.nvfp4"), sha(&mat.0));
            }
        }
        m.insert("x_route.f32".into(), sha(&le_f32(&s.x_route)));
        m.insert("x_moe.f32".into(), sha(&le_f32(&s.x_moe)));
        m.insert("x_dense.f32".into(), sha(&le_f32(&s.x_dense)));
        m
    }

    // ---------------------------------------------------------------- the golden

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5/moe")
    }

    struct Golden {
        man: serde_json::Value,
        router_ids: Vec<i32>,
        router_w: Vec<f32>,
        moe_ids: Vec<i32>,
        moe_w: Vec<f32>,
        moe_y: Vec<f32>,
        dense_y: Vec<f32>,
    }

    fn read_raw<T: Copy>(name: &str, n: usize, conv: fn([u8; 4]) -> T) -> Vec<T> {
        let p = fixture_dir().join(name);
        let b = std::fs::read(&p).unwrap_or_else(|e| panic!("{}: {e} (oracle/export_glm5_moe_golden.py writes it)", p.display()));
        assert_eq!(b.len(), 4 * n, "{name}: {} B, want {}", b.len(), 4 * n);
        b.chunks_exact(4).map(|c| conv([c[0], c[1], c[2], c[3]])).collect()
    }

    fn golden() -> Golden {
        let p = fixture_dir().join("manifest.json");
        let man: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&p).unwrap_or_else(|e| panic!("{}: {e}", p.display()))).unwrap();
        let (k, h) = (8, 4096);
        Golden {
            router_ids: read_raw("router-ids.i32", TR * k, i32::from_le_bytes),
            router_w: read_raw("router-weights.f32", TR * k, f32::from_le_bytes),
            moe_ids: read_raw("moe-ids.i32", TM * k, i32::from_le_bytes),
            moe_w: read_raw("moe-weights.f32", TM * k, f32::from_le_bytes),
            moe_y: read_raw("moe-y.f32", TM * h, f32::from_le_bytes),
            dense_y: read_raw("dense-y.f32", TD * h, f32::from_le_bytes),
            man,
        }
    }

    /// the golden was computed from exactly these inputs
    fn check_inputs(gd: &Golden, s: &Synth) {
        let want = &gd.man["inputs_sha256"];
        for (name, got) in input_shas(s) {
            assert_eq!(want[&name].as_str(), Some(got.as_str()), "{name}: the golden was written from other inputs");
        }
    }

    fn cosine(a: &[f32], b: &[f32]) -> f64 {
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
    }

    fn rms(a: &[f32]) -> f64 {
        (a.iter().map(|&x| x as f64 * x as f64).sum::<f64>() / a.len() as f64).sqrt()
    }

    /// G3 (plan step 15): cosine >= 0.9999 per row against the oracle
    const COS_MIN: f64 = 0.9999;

    /// the routing equals the oracle's as a set per row, weights within `wtol`; returns max |dw|
    fn check_routing(r: &Routing, ids: &[i32], w: &[f32], wtol: f64, what: &str) -> f64 {
        let mut worst = 0f64;
        for t in 0..r.tokens() {
            let (gi, gw) = r.sorted_row(t);
            let want: Vec<u32> = ids[t * 8..(t + 1) * 8].iter().map(|&v| v as u32).collect();
            assert_eq!(gi, want, "{what} row {t}: routed set differs from the oracle");
            worst = worst.max(max_abs(&gw, &w[t * 8..(t + 1) * 8]));
        }
        assert!(worst <= wtol, "{what}: routing weights {worst:.3e} from the oracle (> {wtol:.0e})");
        worst
    }

    /// Acceptance: the routed id SET of every row equals the oracle's (64 router rows + the 4 MoE
    /// rows), weights within 1e-6 absolute (they sum to 2.5).
    #[test]
    fn glm5_moe_router_matches_the_oracle() {
        let (s, gd, g) = (synth(), golden(), geo());
        check_inputs(&gd, &s);
        let rw = RouterWeights { weight: &s.router_w, bias: &s.bias };
        let r = route(&g, &rw, &s.x_route);
        let w1 = check_routing(&r, &gd.router_ids, &gd.router_w, 1e-6, "router");
        let r = route(&g, &rw, &s.x_moe);
        let w2 = check_routing(&r, &gd.moe_ids, &gd.moe_w, 1e-6, "moe");
        eprintln!(
            "glm5_moe router: {} rows, routed sets 100 % equal, max |dw| {:.2e} / {:.2e}, oracle min 8th-9th choice gap {}",
            TR + TM,
            w1,
            w2,
            gd.man["router_min_gap"]
        );
    }

    /// Acceptance: the MoE layer output (MUL1 routed experts on the CPU lane + NVFP4 shared expert)
    /// against HF `Glm5NextTextMoE` on the same decoded weights: cosine >= 0.9999 per row.
    #[test]
    fn glm5_moe_cpu_layer_matches_the_oracle() {
        let (s, gd, g) = (synth(), golden(), geo());
        check_inputs(&gd, &s);
        let mut recs = BTreeMap::new();
        for &e in &gd.moe_ids {
            recs.entry(e as u32).or_insert_with(|| record(e as u32));
        }
        for (e, r) in &recs {
            assert_eq!(gd.man["records_sha256"][e.to_string()].as_str(), Some(sha(r).as_str()), "expert {e}: the golden saw another record");
        }
        let layer = MoeLayerCpu { router: RouterWeights { weight: &s.router_w, bias: &s.bias }, shared: block(&s.shared) };
        let mut y = vec![0f32; TM * 4096];
        let r = moe_cpu(&g, &layer, &Records(recs), &s.x_moe, &mut y, 4, Path::Auto).unwrap();
        check_routing(&r, &gd.moe_ids, &gd.moe_w, 1e-6, "moe");
        for t in 0..TM {
            let (a, b) = (&y[t * 4096..(t + 1) * 4096], &gd.moe_y[t * 4096..(t + 1) * 4096]);
            let c = cosine(a, b);
            eprintln!("glm5_moe MoE row {t}: 1 - cosine {:.2e}, max abs {:.3e} at rms {:.3e}", 1.0 - c, max_abs(a, b), rms(b));
            assert!(c >= COS_MIN, "MoE row {t}: cosine {c:.9} < {COS_MIN}");
        }
    }

    /// Acceptance: the dense FFN (layers 0-2, width 12,288, NVFP4) against HF `Glm5NextTextMLP`.
    #[test]
    fn glm5_moe_cpu_dense_matches_the_oracle() {
        let (s, gd, g) = (synth(), golden(), geo());
        check_inputs(&gd, &s);
        let f = block(&s.dense);
        assert_eq!(f.inter, g.dense_inter);
        let mut y = vec![0f32; TD * 4096];
        ffn_nvfp4_cpu(&f, g.swiglu_limit, &s.x_dense, &mut y, 4, Path::Auto);
        for t in 0..TD {
            let (a, b) = (&y[t * 4096..(t + 1) * 4096], &gd.dense_y[t * 4096..(t + 1) * 4096]);
            let c = cosine(a, b);
            eprintln!("glm5_moe dense row {t}: 1 - cosine {:.2e}, max abs {:.3e} at rms {:.3e}", 1.0 - c, max_abs(a, b), rms(b));
            assert!(c >= COS_MIN, "dense row {t}: cosine {c:.9} < {COS_MIN}");
        }
    }

    /// `GLM5_MOE_DUMP=<dir>`: the oracle's inputs - every encoded input decoded to f32 in HF's
    /// layout (`[out][in]`), the records of the experts this lane routes the MoE rows to, and
    /// `inputs.json` (sha256 of the encoded inputs and records, the stats). Not a check.
    #[test]
    #[ignore = "writes the oracle inputs: GLM5_MOE_DUMP=<dir> cargo test --release --lib glm5_moe_write_oracle_inputs -- --ignored"]
    fn glm5_moe_write_oracle_inputs() {
        let dir = std::path::PathBuf::from(std::env::var("GLM5_MOE_DUMP").expect("GLM5_MOE_DUMP=<dir>"));
        std::fs::create_dir_all(&dir).unwrap();
        let (s, g) = (synth(), geo());
        let put = |name: &str, b: &[u8]| std::fs::write(dir.join(name), b).unwrap();
        put("router.bf16", &le_u16(&s.router_w));
        put("bias.f32", &le_f32(&s.bias));
        put("x_route.f32", &le_f32(&s.x_route));
        put("x_moe.f32", &le_f32(&s.x_moe));
        put("x_dense.f32", &le_f32(&s.x_dense));
        let lut = cpu_nvfp4::scale_lut(NVFP4_GS);
        const E2M1: [f64; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
        let dec = |m: &(Vec<u8>, usize, usize)| -> Vec<f32> {
            let (rows, cols) = (m.1, m.2);
            let mut w = vec![0f32; rows * cols];
            for r in 0..rows {
                for c in 0..cols {
                    let blk = &m.0[(r * (cols / 64) + c / 64) * 36..][..36];
                    let idx = c % 64;
                    let byte = blk[4 + idx / 2];
                    let nib = if idx % 2 == 0 { byte & 15 } else { byte >> 4 };
                    let v = E2M1[(nib & 7) as usize] * if nib & 8 != 0 { -1.0 } else { 1.0 };
                    w[r * cols + c] = (v * lut[blk[idx / 16] as usize] as f64) as f32;
                }
            }
            w
        };
        for (name, mats) in [("shared", &s.shared), ("dense", &s.dense)] {
            for (p, mat) in ["gate", "up", "down"].iter().zip(mats.iter()) {
                put(&format!("{name}_{p}.f32"), &le_f32(&dec(mat)));
            }
        }
        // this lane's routing picks the records to dump; the oracle refuses a pick without a dump
        let r = route(&g, &RouterWeights { weight: &s.router_w, bias: &s.bias }, &s.x_moe);
        let mut rec_sha = serde_json::Map::new();
        let (mut clamped_g, mut clamped_u, mut n_act) = (0usize, 0usize, 0usize);
        for e in r.needed() {
            let rec = record(e);
            rec_sha.insert(e.to_string(), sha(&rec).into());
            let ex = Mul1Expert::from_record(&rec, 4096, 2048, g.bitrate).unwrap();
            let lin = |m: &cpu_mul1::Mul1Matrix| -> Vec<f32> {
                // RefLinear.w is [in][out] in f64; HF wants [out][in]
                let rl = testkit::RefLinear::new(m);
                let mut w = vec![0f32; m.k * m.n];
                for i in 0..m.k {
                    for o in 0..m.n {
                        w[o * m.k + i] = rl.w[i * m.n + o] as f32;
                    }
                }
                w
            };
            let mut gu = lin(&ex.gate);
            gu.extend(lin(&ex.up));
            put(&format!("expert_{e}_gate_up.f32"), &le_f32(&gu));
            put(&format!("expert_{e}_down.f32"), &le_f32(&lin(&ex.down)));
            // how much of the activation the clamp touches (the evidence it is exercised)
            let rows: Vec<usize> = (0..TM).filter(|&t| r.ids_of(t).contains(&e)).collect();
            let xs: Vec<f32> = rows.iter().flat_map(|&t| s.x_moe[t * 4096..(t + 1) * 4096].iter().copied()).collect();
            let (mut gg, mut uu) = (vec![0f32; rows.len() * 2048], vec![0f32; rows.len() * 2048]);
            cpu_mul1::gemv(&ex.gate, &xs, &mut gg, 4, Path::Auto);
            cpu_mul1::gemv(&ex.up, &xs, &mut uu, 4, Path::Auto);
            clamped_g += gg.iter().filter(|&&v| v > 10.0).count();
            clamped_u += uu.iter().filter(|&&v| v.abs() > 10.0).count();
            n_act += gg.len();
            eprintln!("expert {e}: dumped");
        }
        let shas: serde_json::Map<String, serde_json::Value> = input_shas(&s).into_iter().map(|(k, v)| (k, v.into())).collect();
        let man = serde_json::json!({
            "ticket": "crow-nest #164",
            "seed": SEED, "record_seed": RECORD_SEED, "record_seed_step": 9,
            "x_amp": X_AMP, "router_amp": ROUTER_AMP, "bias_amp": BIAS_AMP, "nvfp4_global_scale": NVFP4_GS,
            "tokens": {"router": TR, "moe": TM, "dense": TD},
            "inputs_sha256": shas,
            "records_sha256": rec_sha,
            "lane_moe_ids": r.ids,
            "clamp_fraction": {"gate_gt_10": clamped_g as f64 / n_act as f64, "up_abs_gt_10": clamped_u as f64 / n_act as f64},
        });
        put("inputs.json", serde_json::to_string_pretty(&man).unwrap().as_bytes());
        eprintln!("wrote {}: {} experts, clamp fractions {}", dir.display(), r.needed().len(), man["clamp_fraction"]);
    }

    // ---------------------------------------------------------------- the GPU lane

    /// `GLM5_MOE_SRC` compiles to one PTX module with every entry `NAMES` lists (NVRTC, no GPU)
    #[test]
    fn glm5_moe_source_compiles_with_every_entry() {
        let ptx = crate::kernels::tests_300_c4::ptx(crate::kernels::GLM5_MOE_SRC);
        let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), kernels::glm5_moe::NAMES.len(), "{names:?}");
        for n in kernels::glm5_moe::NAMES {
            assert!(names.iter().any(|m| m == n), "{n} missing in {names:?}");
        }
    }

    unsafe fn main_kernels() -> (cuda::Module, kernels::Kernels) {
        let m = cuda::compile(&kernels::KernelGeo::flash_next().source());
        let k = kernels::Kernels::new(&m, false);
        (m, k)
    }

    unsafe fn gpu_nvfp4(m: &(Vec<u8>, usize, usize)) -> GpuNvfp4 {
        GpuNvfp4 { w: cuda::upload_dev(&m.0), gs: cuda::to_f32_dev(&[NVFP4_GS]), rows: m.1, cols: m.2 }
    }

    unsafe fn gpu_ffn(m: &[(Vec<u8>, usize, usize); 3]) -> GpuFfnWeights {
        GpuFfnWeights { gate: gpu_nvfp4(&m[0]), up: gpu_nvfp4(&m[1]), down: gpu_nvfp4(&m[2]) }
    }

    unsafe fn free_ffn(f: GpuFfnWeights) {
        for mut m in [f.gate, f.up, f.down] {
            cuda::free_dev(&mut m.w);
            cuda::free_dev(&mut m.gs);
        }
    }

    /// The router kernel against `route_from_logits`: random rows, a row of ties, a row of NaN,
    /// E 288 and a tiny E: the same ids in the same order, weights within 2 ulp of 2.5.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_router_is_the_host_selection() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let gk = kernels::glm5_moe::Kernels::new();
            let mut rng = Rng(0x164);
            for g in [geo(), tiny(7, 3), tiny(32, 8)] {
                let (e, k, t) = (g.experts, g.topk, 6);
                let mut logits: Vec<f32> = (0..t * e).map(|_| rng.f(4.0)).collect();
                logits[e..2 * e].iter_mut().for_each(|v| *v = 1.0);
                logits[2 * e..3 * e].iter_mut().for_each(|v| *v = f32::NAN);
                let bias: Vec<f32> = (0..e).map(|_| rng.f(0.1)).collect();
                let want = route_from_logits(&g, &logits, &bias);
                let (mut ld, mut bd) = (cuda::to_f32_dev(&logits), cuda::to_f32_dev(&bias));
                let (mut ids, mut wts) = (cuda::alloc_zeroed(t * k * 4), cuda::alloc_zeroed(t * k * 4));
                let (mut pi, mut pf) = (i32_dev(&[e, k]), cuda::to_f32_dev(&[g.routed_scaling, g.swiglu_limit]));
                kernels::launch_sync(gk.router, t as u32, 1, 1, kernels::glm5_moe::ROUTER_THREADS as u32, &[ld, bd, ids, wts, pi, pf]);
                let got_ids: Vec<u32> = cuda::dtoh_i32(ids, t * k).into_iter().map(|v| v as u32).collect();
                let got_w = cuda::dtoh(wts, t * k);
                assert_eq!(got_ids, want.ids, "E {e} K {k}");
                let dw = max_abs(&got_w[..k], &want.weights[..k]).max(max_abs(&got_w[3 * k..], &want.weights[3 * k..]));
                assert!(dw <= 5e-7, "E {e}: weights {dw:.2e} from the host");
                for d in [&mut ld, &mut bd, &mut ids, &mut wts, &mut pi, &mut pf] {
                    cuda::free_dev(d);
                }
            }
        }
    }

    /// The GPU lane against the oracle: the MoE layer with the records in VRAM and in pinned RAM
    /// (same bits), routing = the oracle's, cosine >= 0.9999 per row; the dense FFN likewise; the
    /// clamp kernel equals `swiglu_clamp` bit for bit.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_layer_matches_the_oracle() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, gd, g) = (synth(), golden(), geo());
            check_inputs(&gd, &s);
            // the clamp kernel, bit for bit
            let gv: Vec<f32> = (0..4096).map(|i| (i as f32 - 2048.0) * 0.0123).collect();
            let uv: Vec<f32> = (0..4096).map(|i| (2048.0 - i as f32) * 0.0071).collect();
            let (mut a, mut b, mut c) = (cuda::to_f32_dev(&gv), cuda::to_f32_dev(&uv), cuda::alloc_zeroed(4096 * 4));
            let (mut pn, mut pf) = (i32_dev(&[4096]), cuda::to_f32_dev(&[0.0, 10.0]));
            kernels::launch_sync(gk.act, 16, 1, 1, 256, &[a, b, c, pn, pf]);
            let got = cuda::dtoh(c, 4096);
            for i in 0..4096 {
                assert_eq!(got[i].to_bits(), swiglu_clamp(gv[i], uv[i], 10.0).to_bits(), "clamp at {i}");
            }
            for d in [&mut a, &mut b, &mut c, &mut pn, &mut pf] {
                cuda::free_dev(d);
            }
            // records: every needed expert once; every other table entry points at the first record
            let needed: Vec<u32> = {
                let mut v: Vec<u32> = gd.moe_ids.iter().map(|&e| e as u32).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let all: Vec<u8> = needed.iter().flat_map(|&e| record(e)).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3 as u64;
            let mut vram = cuda::upload_dev(&all);
            let mut pinned = cuda::Pinned::alloc_cold(all.len());
            pinned.write_bytes(0, &all);
            let table = |base: u64| -> Vec<u64> { (0..g.experts as u32).map(|e| base + rb * needed.iter().position(|&n| n == e).unwrap_or(0) as u64).collect() };
            let (mut tv, mut tp) = (cuda::to_u64_dev(&table(vram)), cuda::to_u64_dev(&table(pinned.dev)));
            let w = GpuMoeWeights { router: cuda::upload_dev(&le_u16(&s.router_w)), bias: cuda::to_f32_dev(&s.bias), shared: gpu_ffn(&s.shared) };
            let mut plan = GpuMoePlan::new(&g, TM);
            let (mut xd, mut yd) = (cuda::to_f32_dev(&s.x_moe), cuda::alloc_zeroed(TM * 4096 * 4));
            plan.run(&kn, &mk, &gk, &w, tv, xd, yd);
            let r = plan.read_routing();
            let yv = cuda::dtoh(yd, TM * 4096);
            plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
            cuda::sync();
            let yp = cuda::dtoh(yd, TM * 4096);
            assert!(yv.iter().zip(&yp).all(|(a, b)| a.to_bits() == b.to_bits()), "VRAM and pinned lanes differ");
            check_routing(&r, &gd.moe_ids, &gd.moe_w, 1e-6, "gpu moe");
            for t in 0..TM {
                let (a, b) = (&yv[t * 4096..(t + 1) * 4096], &gd.moe_y[t * 4096..(t + 1) * 4096]);
                let c = cosine(a, b);
                eprintln!("glm5_moe GPU MoE row {t}: 1 - cosine {:.2e}, max abs {:.3e} at rms {:.3e}", 1.0 - c, max_abs(a, b), rms(b));
                assert!(c >= COS_MIN, "GPU MoE row {t}: cosine {c:.9}");
            }
            // dense
            let dw = gpu_ffn(&s.dense);
            let mut dp = GpuFfnPlan::new(4096, g.dense_inter, TD, g.swiglu_limit);
            let (mut xdd, mut ydd) = (cuda::to_f32_dev(&s.x_dense), cuda::alloc_zeroed(TD * 4096 * 4));
            dp.run(&kn, &gk, &dw, xdd, ydd);
            cuda::sync();
            let yd2 = cuda::dtoh(ydd, TD * 4096);
            for t in 0..TD {
                let (a, b) = (&yd2[t * 4096..(t + 1) * 4096], &gd.dense_y[t * 4096..(t + 1) * 4096]);
                let c = cosine(a, b);
                eprintln!("glm5_moe GPU dense row {t}: 1 - cosine {:.2e}, max abs {:.3e} at rms {:.3e}", 1.0 - c, max_abs(a, b), rms(b));
                assert!(c >= COS_MIN, "GPU dense row {t}: cosine {c:.9}");
            }
            plan.free();
            dp.free();
            free_ffn(w.shared);
            free_ffn(dw);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut vram, &mut tv, &mut tp, &mut xd, &mut yd, &mut xdd, &mut ydd, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }
}
