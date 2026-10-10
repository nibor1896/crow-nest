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
//!   (token, k) combo, the shared and dense FFN on `glm5_gemv_fp4` (#191, bit-identical to the
//!   engine's `gemv_fp4_b`) with the same clamp kernel, `glm5_moe_combine` last. The record base
//!   of every expert comes from a device table `[E] u64` the caller owns: a VRAM slot or a pinned-host UVA address, as `gemv_fp4_ptrb` reads
//!   them - the hook for the residency and the dynamic cache (#175). No host sync inside `run`.
//! - **CPU lane in the GPU layer** (#188, [`lane`]): a decode call's combos posted by the tiers
//!   are split; the CPU computes its share from the pinned records while the GPU computes the
//!   rest, the combine is the GPU's (`GpuMoePlan::experts`, one host wait for x).
//! - **Prefill, expert-major** ([`ExpertMajor`], [`GpuMoeGroupedPlan`]): a prompt call groups
//!   its combos by expert (0xSero's glm53-flash-offload prefill, exllamav3's grouped MoE); the
//!   tiers serve each selected expert once for the call, and `mul1_gemm_grp` decodes an expert's
//!   trellis once per tile of [`GROUP_ROWS`] rows routed to it (had_in, the k-split GEMV and
//!   had_out in one block, the clamp fused into the down GEMM's input). Every row has the bits
//!   of the per-combo path; scratch about 0.3 MB per row instead of about 5 MB.
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
    /// #196: `g`, `u`, `h` are this plan's allocations (else views into a region its owner frees)
    owned: bool,
}

/// #196: device alignment of every buffer carved out of a shared scratch region
pub const CARVE_ALIGN: usize = 256;

/// #196: the offsets of buffers of `bytes` each, one after the other in a region, every one on a
/// [`CARVE_ALIGN`] boundary, and the region's size
pub fn carve(bytes: &[usize]) -> (Vec<usize>, usize) {
    let mut off = Vec::with_capacity(bytes.len());
    let mut at = 0usize;
    for &b in bytes {
        off.push(at);
        at += b.div_ceil(CARVE_ALIGN) * CARVE_ALIGN;
    }
    (off, at)
}

impl GpuFfnPlan {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(hidden: usize, inter: usize, tokens: usize, limit: f32) -> GpuFfnPlan {
        assert!(hidden % 64 == 0 && inter % 64 == 0 && tokens > 0, "glm5_moe: FFN [{inter}, {hidden}] x {tokens}");
        let mut p = GpuFfnPlan::new_in(hidden, inter, tokens, limit, 0);
        p.g = cuda::alloc_zeroed(tokens * inter * 4);
        p.u = cuda::alloc_zeroed(tokens * inter * 4);
        p.h = cuda::alloc_zeroed(tokens * inter * 4);
        p.owned = true;
        p
    }

    /// #196: the region bytes of `g`, `u`, `h` of a plan of `tokens` rows ([`carve`])
    pub fn region_bytes(inter: usize, tokens: usize) -> usize {
        carve(&[tokens * inter * 4; 3]).1
    }

    /// #196: a plan whose `g`, `u`, `h` are views into `region` ([`GpuFfnPlan::region_bytes`]
    /// long; 0 = none yet), which its owner allocates and frees; [`GpuFfnPlan::free`] frees only
    /// the parameter arrays
    ///
    /// # Safety
    /// A CUDA context is current; `region` outlives every launch of the plan.
    pub unsafe fn new_in(hidden: usize, inter: usize, tokens: usize, limit: f32, region: CUdeviceptr) -> GpuFfnPlan {
        assert!(hidden % 64 == 0 && inter % 64 == 0 && tokens > 0, "glm5_moe: FFN [{inter}, {hidden}] x {tokens}");
        let (off, _) = carve(&[tokens * inter * 4; 3]);
        let at = |i: usize| if region == 0 { 0 } else { region + off[i] as u64 };
        GpuFfnPlan {
            hidden,
            inter,
            tokens,
            prm_kh: i32_dev(&[hidden]),
            prm_ki: i32_dev(&[inter]),
            prm_n: i32_dev(&[tokens * inter]),
            prm_f: cuda::to_f32_dev(&[0.0, limit]),
            g: at(0),
            u: at(1),
            h: at(2),
            owned: false,
        }
    }

    /// queue `y = ffn(x)`, x, y `[T][hidden]` f32 (four launches on the current stream); the
    /// three NVFP4 GEMVs on `glm5_gemv_fp4` (#191, bit-identical to the record `gemv_fp4_b`)
    ///
    /// # Safety
    /// The weights match the plan's shape. `_kn` (the engine kernel table) is no longer read.
    pub unsafe fn run(&self, _kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuFfnWeights, x: CUdeviceptr, y: CUdeviceptr) {
        self.run_rows(gk, w, x, y, self.tokens);
    }

    /// [`GpuFfnPlan::run`] on the first `t` rows only (every launch is per row or per element,
    /// so each row has the bits of `run`)
    ///
    /// # Safety
    /// As [`GpuFfnPlan::run`]; `1 <= t <= tokens`.
    pub unsafe fn run_rows(&self, gk: &kernels::glm5_moe::Kernels, w: &GpuFfnWeights, x: CUdeviceptr, y: CUdeviceptr, t: usize) {
        self.run_rows_of(gk, w, x, y, t, t);
    }

    /// [`GpuFfnPlan::run_rows`] for `t` rows of a call of `call_t` rows run in pieces
    /// (`glm5_model`'s prompt call: `t = call_t`): #186 `CROW_GLM_DENSE_GEMM=1` picks the kernel by `call_t`, so
    /// every row of the call takes the kernel one `call_t`-row plan gives it (each kernel computes
    /// a row from that row alone: the pieces have the bits of one plan)
    ///
    /// # Safety
    /// As [`GpuFfnPlan::run`]; `1 <= t <= tokens`, `t <= call_t`.
    pub unsafe fn run_rows_of(&self, gk: &kernels::glm5_moe::Kernels, w: &GpuFfnWeights, x: CUdeviceptr, y: CUdeviceptr, t: usize, call_t: usize) {
        assert!((1..=self.tokens).contains(&t) && t <= call_t, "glm5_moe: FFN rows {t} of a {}-row plan, call {call_t}", self.tokens);
        let (h, i) = (self.hidden, self.inter);
        assert!((w.gate.rows, w.gate.cols, w.up.rows, w.up.cols, w.down.rows, w.down.cols) == (i, h, i, h, h, i), "glm5_moe: FFN weights do not fit the plan");
        if gk.dense_tc(call_t) {
            // #186 `CROW_GLM_DENSE_GEMM=1`: the three projections on the FP16 tensor-core GEMM
            let small = kernels::glm5_moe::tc_small(call_t);
            gk.gemm_tc_on(small, [w.gate.w; 3], [w.gate.gs; 3], 1, x, self.g, self.prm_kh, self.prm_ki, self.prm_ki, i, t);
            gk.gemm_tc_on(small, [w.up.w; 3], [w.up.gs; 3], 1, x, self.u, self.prm_kh, self.prm_ki, self.prm_ki, i, t);
            launch_v(gk.act, (t * i).div_ceil(256) as u32, 1, 1, 256, &[self.g, self.u, self.h, self.prm_n, self.prm_f]);
            gk.gemm_tc_on(small, [w.down.w; 3], [w.down.gs; 3], 1, self.h, y, self.prm_ki, self.prm_kh, self.prm_kh, h, t);
            return;
        }
        let ((gu, bu), (gd, bd)) = (kernels::glm5_moe::fp4_launch(i, h), kernels::glm5_moe::fp4_launch(h, i));
        // [w, x, gs, y, K, ldy, rows]: gate / up K = hidden, ldy = rows = inter; down the reverse
        launch_v(gk.fp4, gu, t as u32, 1, bu, &[w.gate.w, x, w.gate.gs, self.g, self.prm_kh, self.prm_ki, self.prm_ki]);
        launch_v(gk.fp4, gu, t as u32, 1, bu, &[w.up.w, x, w.up.gs, self.u, self.prm_kh, self.prm_ki, self.prm_ki]);
        launch_v(gk.act, (t * i).div_ceil(256) as u32, 1, 1, 256, &[self.g, self.u, self.h, self.prm_n, self.prm_f]);
        launch_v(gk.fp4, gd, t as u32, 1, bd, &[w.down.w, self.h, w.down.gs, y, self.prm_ki, self.prm_kh, self.prm_kh]);
    }

    /// # Safety
    /// No launch of this plan is pending.
    pub unsafe fn free(&mut self) {
        for d in [&mut self.prm_kh, &mut self.prm_ki, &mut self.prm_n, &mut self.prm_f] {
            cuda::free_dev(d);
        }
        if self.owned {
            for d in [&mut self.g, &mut self.u, &mut self.h] {
                cuda::free_dev(d);
            }
        }
        (self.g, self.u, self.h) = (0, 0, 0);
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
    /// `CROW_GLM_SHARED_OVERLAP`: [`GpuMoePlan::shared_early`] queued `ys` for the next `experts`
    shared_queued: std::cell::Cell<bool>,
    gate: mul1::GemvPlan,
    up: mul1::GemvPlan,
    down: mul1::GemvPlan,
    /// #188 CPU lane: host buffer and event, made on the first lane call
    lane: std::cell::OnceCell<LaneBuf>,
    /// #202 early reply: the late pass's slots, made on its first call
    late: std::cell::OnceCell<LatePass>,
    /// #202 RT2: [`GpuMoePlan::lane_x_early`] queued x into the lane's host buffer for the
    /// coming `experts` call (consumed by it)
    x_early: std::cell::Cell<bool>,
    /// #202 RT2: the slot and row tables of [`GpuMoePlan::experts_rt2`], made on its first call
    rt2: std::cell::OnceCell<Rt2Buf>,
}

/// #202 RT2: `[2 * K]` u64 on the host (mapped) and on the device: the GPU slots' record bases
/// (early slots first, then late), then every combo's output row address; a zeroed `[E]` table
/// the gather reads (its `ptrs` are unused here, so it never reads the device table the stager
/// stream may be writing) and the event behind the resident experts
struct Rt2Buf {
    host: cuda::Pinned,
    dev: CUdeviceptr,
    zero_table: CUdeviceptr,
    ev_early: u64,
}

/// #202 early reply: the late pass of [`GpuMoePlan::experts_late`]: `slots` record bases, the
/// combo of each slot (i32, -1 none) and the slots' outputs `[slots][H]` f32
struct LatePass {
    slots: usize,
    ptrs: CUdeviceptr,
    idx: CUdeviceptr,
    ye: CUdeviceptr,
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
            shared_queued: std::cell::Cell::new(false),
            gate:mul1::GemvPlan::new(sg, c, 1),
            up: mul1::GemvPlan::new(su, c, 1),
            down: mul1::GemvPlan::new(sd, c, 1),
            lane: std::cell::OnceCell::new(),
            late: std::cell::OnceCell::new(),
            x_early: std::cell::Cell::new(false),
            rt2: std::cell::OnceCell::new(),
        }
    }

    /// #202 RT2 (one row): queue the MoE input row `x` `[H]` f32 into the CPU lane's host buffer
    /// and its event now, ahead of the router's publish, so the host has x once it saw the flag;
    /// the coming [`GpuMoePlan::experts`] call does not copy it again (and drops the mark).
    ///
    /// # Safety
    /// `x` holds the MoE input written by work queued before; `tokens == 1`.
    pub unsafe fn lane_x_early(&self, x: CUdeviceptr) {
        use cudarc::driver::sys;
        assert_eq!(self.tokens, 1, "glm5_moe: RT2's early x is a one-row call");
        let (h, k) = (self.geo.hidden, self.geo.topk);
        let buf = self.lane.get_or_init(|| LaneBuf::new(h, k, 1));
        let s = cuda::cur_stream();
        cuda::ck(sys::cuMemcpyDtoHAsync_v2(buf.host.host, x, h * 4, s));
        cuda::event_record(buf.ev as sys::CUevent, s);
        self.x_early.set(true);
    }

    /// queue the layer: `y = moe(x)`, x, y `[T][H]` f32 (13 launches on the current stream, no
    /// host sync, graph-capturable).
    ///
    /// # Safety
    /// `kn` comes from a module with `gemv_bf16_b`; `table` is a device `[E]` u64
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
        self.route(kn, gk, w, x);
        self.experts(kn, mk, gk, w, table, x, y);
    }

    /// the first two launches of [`GpuMoePlan::run`]: router logits and the top-K selection into
    /// `ids` / `wts`. #175: the three-tier path reads `ids` between `route` and `experts` to put
    /// the selected records in place and fill `table`.
    ///
    /// # Safety
    /// As [`GpuMoePlan::run`].
    pub unsafe fn route(&self, kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuMoeWeights, x: CUdeviceptr) {
        let (e, t) = (self.geo.experts, self.tokens);
        launch_v(kn.f("gemv_bf16_b"), e as u32, t as u32, 1, 256, &[w.router, x, self.logits, self.prm_kh]);
        launch_v(gk.router, t as u32, 1, 1, kernels::glm5_moe::ROUTER_THREADS as u32, &[self.logits, w.bias, self.ids, self.wts, self.prm_route, self.prm_f]);
    }

    /// the rest of [`GpuMoePlan::run`] after [`GpuMoePlan::route`]: gather through `table`, the
    /// routed experts, the shared expert, the combine (11 launches, no host sync)
    ///
    /// # Safety
    /// As [`GpuMoePlan::run`]; `route` ran on this plan with the same `x`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        table: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
    ) {
        let x_queued = self.x_early.replace(false);
        if let Some(mut call) = lane::take(table, self.tokens, self.geo.topk) {
            if let Some(rt) = call.rt2.take() {
                return self.experts_rt2(kn, mk, gk, w, x, y, call, rt, x_queued);
            }
            return self.experts_lane(kn, mk, gk, w, table, x, y, call, x_queued);
        }
        self.experts_merge(kn, mk, gk, w, table, x, y, &mut |_| {});
    }

    /// [`GpuMoePlan::experts`] on the GPU path with `merge(ye)` queued between the routed
    /// experts (and the shared one) and the combine. `CROW_GLM_CONTROLLER` with the CPU lane: the
    /// device waits there for the controller thread's CPU rows and copies them over their combos'
    /// rows of `ye` (`glm5_flags::DevLane`); an empty `merge` is [`GpuMoePlan::experts`]'s GPU path.
    ///
    /// # Safety
    /// As [`GpuMoePlan::experts`]; `merge` only queues work on the current stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_merge(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        table: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        merge: &mut dyn FnMut(CUdeviceptr),
    ) {
        let (h, t) = (self.geo.hidden, self.tokens);
        let c = t * self.geo.topk;
        launch_v(gk.gather, h.div_ceil(256) as u32, c as u32, 1, 256, &[self.ids, table, x, self.ptrs, self.xg, self.prm_kh2]);
        self.gate.run(mk, self.ptrs, self.xg, self.ge);
        self.up.run(mk, self.ptrs, self.xg, self.ue);
        launch_v(gk.act, (c * self.geo.expert_inter).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
        self.down.run(mk, self.ptrs, self.he, self.ye);
        if !self.shared_queued.replace(false) {
            self.shared.run(kn, gk, &w.shared, x, self.ys);
        }
        merge(self.ye);
        launch_v(gk.combine, h.div_ceil(256) as u32, t as u32, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
    }

    /// #202 early reply (`CROW_GLM_CONTROLLER`, one row): [`GpuMoePlan::experts_merge`] with the
    /// late pass between the experts (and the shared one) and the combine: `fill(ids, ptrs,
    /// late ptrs, late idx)` queues what puts the late experts into the `slots` late slots (their
    /// record bases, their combos; a spare slot a readable record and combo -1), then gate / up /
    /// act / down over those slots (`run_slots`: every slot as in `run`, slots never interact;
    /// the slots read `xg`, which holds x in every row at t = 1, and reuse `ge` / `ue` / `he`,
    /// which the experts' down GEMV has read by then), and `scatter(ye, late ye, late idx)` puts
    /// each slot's row over its combo's row of `ye`. A late combo's row from the experts (its
    /// table entry a stand-in record) is replaced, so the combine sees what `experts` gives.
    ///
    /// # Safety
    /// As [`GpuMoePlan::experts`], `tokens == 1`; `fill` and `scatter` only queue work on the
    /// current stream.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_late(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        table: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        slots: usize,
        fill: &mut dyn FnMut(CUdeviceptr, CUdeviceptr, CUdeviceptr, CUdeviceptr),
        scatter: &mut dyn FnMut(CUdeviceptr, CUdeviceptr, CUdeviceptr),
    ) {
        let (h, k, i) = (self.geo.hidden, self.geo.topk, self.geo.expert_inter);
        assert!(self.tokens == 1 && (1..=k).contains(&slots), "glm5_moe: a late pass of {slots} slots on a plan of {} rows x {k}", self.tokens);
        // the CPU lane's host path is not posted for the controller's tables (consumed as `experts` does)
        let _ = lane::take(table, self.tokens, k);
        // made for top-k slots once (#202 lanes runs k, the former pass fewer)
        let lp = self.late.get_or_init(|| LatePass { slots: k, ptrs: cuda::alloc_zeroed(k * 8), idx: cuda::alloc_zeroed(k * 4), ye: cuda::alloc_zeroed(k * h * 4) });
        assert!(slots <= lp.slots, "glm5_moe: the late pass was made for {} slots", lp.slots);
        self.experts_merge(kn, mk, gk, w, table, x, y, &mut |ye| {
            fill(self.ids, self.ptrs, lp.ptrs, lp.idx);
            self.gate.run_slots(mk, slots, lp.ptrs, self.xg, self.ge);
            self.up.run_slots(mk, slots, lp.ptrs, self.xg, self.ue);
            launch_v(gk.act, (k * i).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
            self.down.run_slots(mk, slots, lp.ptrs, self.he, lp.ye);
            scatter(ye, lp.ye, lp.idx);
        });
    }

    /// `CROW_GLM_SHARED_OVERLAP`: the shared expert of the coming [`GpuMoePlan::experts`] on `x`
    /// now (4 launches), so it runs while the host hands the routing over; that `experts` call
    /// then skips it. The same launches on the same input into the same `ys`, so the layer's
    /// output keeps its bits.
    ///
    /// # Safety
    /// As [`GpuMoePlan::run`]; the next `experts` call of this plan has the same `w` and `x`.
    pub unsafe fn shared_early(&self, kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuMoeWeights, x: CUdeviceptr) {
        self.shared.run(kn, gk, &w.shared, x, self.ys);
        self.shared_queued.set(true);
    }

    /// `CROW_GLM_SHARED_OVERLAP`: the call after [`GpuMoePlan::shared_early`] failed before its
    /// `experts`; the next call computes its shared expert again
    pub fn shared_forget(&self) {
        self.shared_queued.set(false);
    }

    /// #188 CPU lane: [`GpuMoePlan::experts`] for a decode call (`t = 1`) whose combos
    /// `call.combos` (pick order) are split between the GPU and the CPU. Queued: x to the host
    /// (async, event), gather, the GPU combos' record bases compacted into the first slots (one
    /// H2D), gate / up / act / down over those slots only, the shared expert. Then, while the
    /// GPU runs them: wait for x, every CPU combo's expert in ONE pool run
    /// ([`cpu_mul1::experts_ffn`] with [`swiglu_clamp`], [`LANE_THREADS`]) straight from its
    /// pinned record, the bits of [`expert_ffn_mul1_cpu`]. Then the
    /// compact GPU rows go to their combo rows of `ye` (D2D, descending, so no row is overwritten
    /// before it is read), the CPU rows into theirs (H2D from the plan's pinned buffer), and the
    /// unchanged `glm5_moe_combine` sums `w_k ye_k` in pick order + shared. GPU combos give the
    /// bits of [`GpuMoePlan::experts`]; CPU combos the bits of `cpu_mul1::expert_ffn` (not the
    /// GPU kernels': another f32 order in the GEMVs, `docs/mul1-gemv.md` section 3, and the
    /// host `exp` in the clamp).
    ///
    /// The host buffer is rewritten by the next lane call only after the host waited for that
    /// call's routing, which the stream orders after every copy queued here.
    #[allow(clippy::too_many_arguments)]
    unsafe fn experts_lane(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        table: CUdeviceptr,
        x: CUdeviceptr,
        y: CUdeviceptr,
        call: lane::Call,
        x_queued: bool,
    ) {
        use cudarc::driver::sys;
        let g = &self.geo;
        let (h, k, t) = (g.hidden, g.topk, self.tokens);
        // CROW_GLM_MAX_BATCH: a decode step of t rows; combo c belongs to row c / k
        let ca = t * k;
        let rb = g.record.bytes as usize;
        let buf = self.lane.get_or_init(|| LaneBuf::new(h, ca, t));
        let host = buf.host.host as *mut u8;
        let (xh, ph, yh) = (host as *mut f32, host.add(t * h * 4) as *mut u64, host.add(t * h * 4 + ca * 8) as *mut f32);
        let s = cuda::cur_stream();
        let ev = buf.ev as sys::CUevent;
        // #202 RT2: x already queued ahead of the router flag (the same bytes, the same event)
        if !x_queued {
            cuda::ck(sys::cuMemcpyDtoHAsync_v2(xh as *mut _, x, t * h * 4, s));
            cuda::event_record(ev, s);
        }
        launch_v(gk.gather, h.div_ceil(256) as u32, ca as u32, 1, 256, &[self.ids, table, x, self.ptrs, self.xg, self.prm_kh2]);
        let mut gpu = Vec::with_capacity(ca);
        let mut cpu = Vec::with_capacity(ca);
        for (c, combo) in call.combos.iter().enumerate() {
            match *combo {
                lane::Combo::Gpu(base) => {
                    *ph.add(gpu.len()) = base;
                    gpu.push(c);
                }
                lane::Combo::Cpu(rec) => cpu.push((c, rec)),
            }
        }
        let n = gpu.len();
        if n > 0 {
            // the gather wrote xg for every combo: at t = 1 every row is x, so slot j reads x too;
            // for t > 1 slot j takes its combo's row (ascending: every source c >= its slot j, so
            // no source row is overwritten before it is read)
            if t > 1 {
                for (j, &c) in gpu.iter().enumerate() {
                    if c != j {
                        cuda::d2d_async(self.xg + (j * h * 4) as u64, self.xg + (c * h * 4) as u64, h * 4);
                    }
                }
            }
            cuda::upload_from_pinned(self.ptrs, ph as *const _, n * 8);
            self.gate.run_slots(mk, n, self.ptrs, self.xg, self.ge);
            self.up.run_slots(mk, n, self.ptrs, self.xg, self.ue);
            launch_v(gk.act, (ca * g.expert_inter).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
            self.down.run_slots(mk, n, self.ptrs, self.he, self.ye);
        }
        if !self.shared_queued.replace(false) {
            self.shared.run(kn, gk, &w.shared, x, self.ys);
        }
        // hand the queued launches to the GPU (WDDM batches them), then wait for x only (and for
        // the moves that put the CPU's records in place, when the tiers queued them async)
        let _ = sys::cuStreamQuery(s);
        cuda::ck(sys::cuEventSynchronize(ev));
        if let Some(r) = call.ready {
            cuda::ck(sys::cuEventSynchronize(r as sys::CUevent));
        }
        let t0 = std::time::Instant::now();
        let limit = g.swiglu_limit;
        // one pool run per row over the row's CPU combos in pick order (one row: every CPU
        // combo, the run of record); `cpu` is in pick order, so the rows land in its order
        let mut at = 0;
        for r in 0..t {
            let mine: Vec<(usize, *const u8)> = cpu.iter().copied().filter(|&(c, _)| c / k == r).collect();
            if mine.is_empty() {
                continue;
            }
            let es: Vec<Mul1Expert> = mine
                .iter()
                .map(|&(c, rec)| {
                    Mul1Expert::from_record(std::slice::from_raw_parts(rec, rb), h, g.expert_inter, g.bitrate).unwrap_or_else(|e| panic!("glm5_moe CPU lane: combo {c}: {e}"))
                })
                .collect();
            let xs = std::slice::from_raw_parts((xh as *const f32).add(r * h), h);
            let ys = std::slice::from_raw_parts_mut(yh.add(at * h), mine.len() * h);
            cpu_mul1::experts_ffn(&es, xs, ys, &move |a, b| swiglu_clamp(a, b, limit), lane::threads(), Path::Auto);
            at += mine.len();
        }
        call.clock.add(t0.elapsed(), cpu.len());
        for (j, &c) in gpu.iter().enumerate().rev() {
            if c != j {
                cuda::d2d_async(self.ye + (c * h * 4) as u64, self.ye + (j * h * 4) as u64, h * 4);
            }
        }
        for (i, &(c, _)) in cpu.iter().enumerate() {
            cuda::upload_from_pinned(self.ye + (c * h * 4) as u64, yh.add(i * h) as *const _, h * 4);
        }
        launch_v(gk.combine, h.div_ceil(256) as u32, t as u32, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
    }

    /// #202 RT2 (`CROW_GLM_RT2=1`, one row, the stager on): [`GpuMoePlan::experts_lane`] with
    /// the call's GPU combos split by `rt.late`. Queued at once: the gather (x into every slot,
    /// through a zeroed table), the slot and row tables (one H2D), gate / up / act / down over the
    /// early slots (records the stager's batch does not touch), an event, the compute stream's
    /// wait for the stager's event, the same four over the late slots (their records landed or
    /// copied by then), the shared expert unless queued. So the GPU runs its resident experts while
    /// the NVMe reads land, and the late ones as soon as the stager's batch ran, without the host.
    /// Then the host runs the CPU lane (resident records only: nothing to wait for but x, which
    /// the router's publish ordered before its flag when [`GpuMoePlan::lane_x_early`] ran) while
    /// the GPU works, and queues one `glm5_moe_combine_rows` that reads each combo's row where it
    /// is: a GPU slot of `ye` or the lane's row in mapped host memory. Bits: every GPU combo's row
    /// is the row `experts_lane` computes (slots never interact; the act launch covers every slot
    /// and recomputes the early slots' values from unchanged inputs), every CPU row the lane's,
    /// and the combine the same sum in the same order, so the layer's output is `experts_lane`'s
    /// for the same CPU / GPU split.
    ///
    /// The compute stream always waits for the stager's event before the combine, so as before
    /// every stager batch has run before the next router flag.
    #[allow(clippy::too_many_arguments)]
    unsafe fn experts_rt2(
        &self,
        kn: &kernels::Kernels,
        mk: &mul1::Kernels,
        gk: &kernels::glm5_moe::Kernels,
        w: &GpuMoeWeights,
        x: CUdeviceptr,
        y: CUdeviceptr,
        call: lane::Call,
        rt: lane::Rt2,
        x_queued: bool,
    ) {
        use cudarc::driver::sys;
        let g = &self.geo;
        let (h, k, ie) = (g.hidden, g.topk, g.expert_inter);
        assert!(self.tokens == 1 && call.combos.len() == k && rt.late.len() == k, "glm5_moe: RT2 is a one-row call of top-k combos");
        lane::RT2_CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let rb = g.record.bytes as usize;
        let buf = self.lane.get_or_init(|| LaneBuf::new(h, k, 1));
        let tb = self.rt2.get_or_init(|| Rt2Buf {
            host: cuda::Pinned::alloc(2 * k * 8),
            dev: cuda::alloc_zeroed(2 * k * 8),
            zero_table: cuda::alloc_zeroed(g.experts * 8),
            ev_early: cuda::event_create() as u64,
        });
        let host = buf.host.host as *mut u8;
        let (xh, yh) = (host as *const f32, host.add(h * 4 + k * 8) as *mut f32);
        let yh_dev = buf.host.dev + (h * 4 + k * 8) as u64;
        let s = cuda::cur_stream();
        let ev = buf.ev as sys::CUevent;
        let (mut early, mut late, mut cpu) = (Vec::with_capacity(k), Vec::with_capacity(k), Vec::with_capacity(k));
        for (c, combo) in call.combos.iter().enumerate() {
            match *combo {
                lane::Combo::Gpu(base) if rt.late[c] => late.push((c, base)),
                lane::Combo::Gpu(base) => early.push((c, base)),
                lane::Combo::Cpu(rec) => {
                    debug_assert!(!rt.late[c], "glm5_moe RT2: a CPU combo on a record the stager touches");
                    cpu.push((c, rec));
                }
            }
        }
        if !x_queued && !cpu.is_empty() {
            cuda::ck(sys::cuMemcpyDtoHAsync_v2(xh as *mut _, x, h * 4, s));
            cuda::event_record(ev, s);
        }
        let (ne, nl) = (early.len(), late.len());
        // slot j: early then late; the row of combo c: its slot's `ye` row or its lane row. The
        // previous call's upload of these words ran before this call's router flag.
        let hp = tb.host.host as *mut u64;
        for (j, &(c, base)) in early.iter().chain(&late).enumerate() {
            *hp.add(j) = base;
            *hp.add(k + c) = self.ye + (j * h * 4) as u64;
        }
        for (i, &(c, _)) in cpu.iter().enumerate() {
            *hp.add(k + c) = yh_dev + (i * h * 4) as u64;
        }
        launch_v(gk.gather, h.div_ceil(256) as u32, k as u32, 1, 256, &[self.ids, tb.zero_table, x, self.ptrs, self.xg, self.prm_kh2]);
        cuda::upload_from_pinned(tb.dev, hp as *const _, 2 * k * 8);
        let pass = |off: usize, n: usize| {
            let p = tb.dev + (off * 8) as u64;
            self.gate.run_slots(mk, n, p, self.xg + (off * h * 4) as u64, self.ge + (off * ie * 4) as u64);
            self.up.run_slots(mk, n, p, self.xg + (off * h * 4) as u64, self.ue + (off * ie * 4) as u64);
            launch_v(gk.act, (k * ie).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
            self.down.run_slots(mk, n, p, self.he + (off * ie * 4) as u64, self.ye + (off * h * 4) as u64);
        };
        if ne > 0 {
            pass(0, ne);
        }
        let ev_early = tb.ev_early as sys::CUevent;
        cuda::event_record(ev_early, s);
        cuda::stream_wait_event(s, rt.event as sys::CUevent);
        if nl > 0 {
            pass(ne, nl);
        }
        if !self.shared_queued.replace(false) {
            self.shared.run(kn, gk, &w.shared, x, self.ys);
        }
        // hand the queued launches to the GPU (WDDM batches them until a query or a sync)
        let _ = sys::cuStreamQuery(s);
        if !cpu.is_empty() {
            // x: ordered before the router flag the host already saw (or queued just above);
            // polled, not cuEventSynchronize
            loop {
                match sys::cuEventQuery(ev) {
                    sys::CUresult::CUDA_SUCCESS => break,
                    sys::CUresult::CUDA_ERROR_NOT_READY => std::hint::spin_loop(),
                    e => panic!("glm5_moe RT2: the lane's x event: {e:?}"),
                }
            }
            let busy = || sys::cuEventQuery(ev_early) == sys::CUresult::CUDA_ERROR_NOT_READY;
            let start_in_gpu = ne > 0 && busy();
            let t0 = std::time::Instant::now();
            let limit = g.swiglu_limit;
            let es: Vec<Mul1Expert> = cpu
                .iter()
                .map(|&(c, rec)| Mul1Expert::from_record(std::slice::from_raw_parts(rec, rb), h, ie, g.bitrate).unwrap_or_else(|e| panic!("glm5_moe RT2 lane: combo {c}: {e}")))
                .collect();
            let ys = std::slice::from_raw_parts_mut(yh, cpu.len() * h);
            cpu_mul1::experts_ffn(&es, std::slice::from_raw_parts(xh, h), ys, &move |a, b| swiglu_clamp(a, b, limit), lane::threads(), Path::Auto);
            call.clock.add(t0.elapsed(), cpu.len());
            call.clock.add_rt2(start_in_gpu, ne > 0 && busy());
        }
        launch_v(gk.combine_rows, h.div_ceil(256) as u32, 1, 1, 256, &[tb.dev + (k * 8) as u64, self.wts, self.ys, y, self.prm_kh2]);
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
        if let Some(mut lp) = self.late.take() {
            cuda::free_dev(&mut lp.ptrs);
            cuda::free_dev(&mut lp.idx);
            cuda::free_dev(&mut lp.ye);
        }
        if let Some(mut b) = self.lane.take() {
            b.host.free();
            cuda::event_destroy(b.ev as cudarc::driver::sys::CUevent);
        }
        if let Some(mut b) = self.rt2.take() {
            b.host.free();
            cuda::free_dev(&mut b.dev);
            cuda::free_dev(&mut b.zero_table);
            cuda::event_destroy(b.ev_early as cudarc::driver::sys::CUevent);
        }
    }
}

// ---------------------------------------------------------------- prefill: expert-major

/// rows of one work item of `mul1_gemm_grp` (`MUL1_GT` in `kernels_mul1.cu`)
pub const GROUP_ROWS: usize = 16;
/// activation rows one k-split of `mul1_gemm_grp` stages (`MUL1_GXROWS`)
pub const GROUP_XROWS: usize = 256;
/// the grouped GEMM's entry in `kernels::MUL1_SRC`
pub const GROUP_ENTRY: &str = "mul1_gemm_grp";

/// #186: the rows of one expert (in one `experts_items` range) from which `CROW_GLM_MOE_TC=1`
/// runs it as FP16 reconstruction + tensor-core GEMM instead of `mul1_gemm_grp` (exllamav3
/// `exl3.py` 151539c7 L137-144: FP16 reconstruction + hgemm above 144 rows)
pub const TC_MIN_ROWS: usize = 144;
/// #186: the most rows of one tensor-core chunk (the scratch's `xh` / `part` / `g` / `u` rows)
pub const TC_CHUNK_ROWS: usize = 1024;

/// #186: `CROW_GLM_MOE_TC`: `1` or `tc` the per-expert tensor-core path ([`MoeTc`]), `2` the
/// grouped one ([`MoeTc2`]); unset / anything else 0, off (the default)
pub fn moe_tc_mode_from_env() -> u8 {
    match std::env::var("CROW_GLM_MOE_TC").ok().as_deref() {
        Some("1") | Some("tc") => 1,
        Some("2") => 2,
        _ => 0,
    }
}

/// #186: `CROW_GLM_MOE_TC` is on (mode 1 or 2, [`moe_tc_mode_from_env`])
pub fn moe_tc_from_env() -> bool {
    moe_tc_mode_from_env() != 0
}

/// #186: the tensor-core expert scratch of `mode` ([`moe_tc_mode_from_env`]) of a plan of `tokens`
/// rows: [`moe_tc_bytes`] for 1, [`moe_tc2_bytes`] of [`Tc2Cfg::default_for`] for 2, else 0
pub fn moe_tc_mode_bytes(mode: u8, hidden: usize, inter: usize, experts: usize, topk: usize, tokens: usize) -> u64 {
    match mode {
        1 => moe_tc_bytes(hidden, inter, tokens),
        2 => moe_tc2_bytes(hidden, inter, experts, topk, tokens, &Tc2Cfg::default_for(tokens)),
        _ => 0,
    }
}

/// #186: the device bytes of the tensor-core expert scratch of a plan of `tokens` rows ([`MoeTc`]):
/// `xh` and `part` `[rows][max(H, I)]` f32, gate / up `[rows][I]` f32 (rows = min(tokens,
/// [`TC_CHUNK_ROWS`])), the three FP16 matrices of one expert; 0 below [`TC_MIN_ROWS`] tokens
pub fn moe_tc_bytes(hidden: usize, inter: usize, tokens: usize) -> u64 {
    if tokens < TC_MIN_ROWS {
        return 0;
    }
    carve(&moe_tc_parts(hidden, inter, tokens.min(TC_CHUNK_ROWS))).1 as u64
}

fn moe_tc_parts(hidden: usize, inter: usize, rows: usize) -> [usize; 7] {
    let m = hidden.max(inter);
    let w = hidden * inter * 2;
    [rows * m * 4, rows * m * 4, rows * inter * 4, rows * inter * 4, w, w, w]
}

/// #186: the tensor-core expert path of a [`GpuMoeGroupedPlan`] (`CROW_GLM_MOE_TC=1`): one
/// expert's gate, up and down reconstructed to FP16 `[n][k]` (`mul1_tc_recon`, the codec's FP16
/// values), then per chunk of rows `mul1_tc_in` (x * suh, FWHT), `mul1_gemm_tc` (FP16 MMA, FP32
/// accumulate), `mul1_tc_out` (FWHT, / 128, * svh); down reads the clamped SwiGLU of the chunk's
/// gate / up rows and writes `ye`. Not bit-identical to `mul1_gemm_grp`.
pub struct MoeTc {
    region: CUdeviceptr,
    rows: usize,
    xh: CUdeviceptr,
    part: CUdeviceptr,
    g: CUdeviceptr,
    u: CUdeviceptr,
    w: [CUdeviceptr; 3],
    f_in: u64,
    f_recon: u64,
    f_out: u64,
    f_gemm: u64,
}

impl MoeTc {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(hidden: usize, inter: usize, tokens: usize, mk: &mul1::Kernels) -> MoeTc {
        let rows = tokens.clamp(1, TC_CHUNK_ROWS);
        let (off, bytes) = carve(&moe_tc_parts(hidden, inter, rows));
        let region = cuda::alloc_named("glm5 MoE tensor-core expert scratch", bytes);
        let at = |n: usize| region + off[n] as u64;
        MoeTc {
            region,
            rows,
            xh: at(0),
            part: at(1),
            g: at(2),
            u: at(3),
            w: [at(4), at(5), at(6)],
            f_in: mk.module.get("mul1_tc_in") as u64,
            f_recon: mk.module.get("mul1_tc_recon") as u64,
            f_out: mk.module.get("mul1_tc_out") as u64,
            f_gemm: mk.module.get("mul1_gemm_tc") as u64,
        }
    }

    /// # Safety
    /// No launch on the scratch is pending.
    pub unsafe fn free(&mut self) {
        cuda::free_dev(&mut self.region);
    }
}

/// #186 `CROW_GLM_MOE_TC=2`: an expert with at least this many rows in an `experts_items` range
/// takes the grouped tensor-core pass ([`MoeTc2`]), fewer `mul1_gemm_grp` (micro-bench
/// `glm5_moe_gpu_tc2_bench`, one GLM layer at 8192 rows: 1, 16 and 32 within 1 %, 64 +8 %,
/// 144 +76 %)
pub const TC2_MIN_ROWS: usize = 16;
/// #186: the most packed rows of one grouped batch at the default (raised to the plan's tokens:
/// one expert never has more), 32 KiB of scratch per row at GLM-5.3-Flash shapes (micro-bench:
/// 16384 rows 1.5 % faster than 8192 for twice the scratch; 32768 slower)
pub const TC2_ROWS: usize = 8192;

/// #186: the knobs of [`MoeTc2`] (`CROW_GLM_MOE_TC=2`)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tc2Cfg {
    /// decode the weights inside the GEMM (`mul1_tc2_gemm*f`, no weight scratch) instead of
    /// `mul1_tc2_recon` into FP16 weight slots
    pub fused: bool,
    /// the most experts of one batch; without `fused` also the FP16 weight slots (two matrices
    /// of `H * I * 2` bytes each, 32 MiB per slot at GLM shapes)
    pub slots: usize,
    /// the most packed rows of one batch (raised to the plan's tokens)
    pub rows: usize,
    /// row tile of the grouped GEMM, 64 or 128
    pub bm: usize,
    /// an expert takes the tensor cores from this many rows in a range
    pub min_rows: usize,
}

impl Tc2Cfg {
    /// the default of a plan of `tokens` rows: fused decode, 128-row tiles, batches of up to
    /// [`TC2_ROWS`] rows (micro-bench `glm5_moe_gpu_tc2_bench`)
    pub fn default_for(tokens: usize) -> Tc2Cfg {
        Tc2Cfg { fused: true, slots: usize::MAX, rows: TC2_ROWS.max(tokens), bm: 128, min_rows: TC2_MIN_ROWS }
    }

    /// `rows` raised to `tokens`
    fn rows_for(&self, tokens: usize) -> usize {
        self.rows.max(tokens)
    }
}

fn moe_tc2_parts(hidden: usize, inter: usize, experts: usize, topk: usize, tokens: usize, cfg: &Tc2Cfg) -> [usize; 12] {
    let (c, r) = (tokens * topk, cfg.rows_for(tokens));
    let work = grouped_work_cap(experts, topk, tokens);
    [
        if cfg.fused { 0 } else { cfg.slots * 2 * hidden * inter * 2 },
        r * hidden.max(inter) * 2,
        r * hidden * 2,
        r * inter * 4,
        r * inter * 4,
        c * 4,
        c * 4,
        experts * 4,
        (experts + 1) * 4,
        (experts + 1) * 4,
        work * 12,
        c * 4,
    ]
}

/// #186: the device bytes of [`MoeTc2`] of `cfg` for a plan of `tokens` rows: without `fused` the
/// FP16 weights of `slots` experts (two matrices at a time: gate and up, then down); per batch
/// row `xg` FP16 `[max(H, I)]` (also the down input), `xu` FP16 `[H]`, `g` / `u` f32 `[I]` (32 KiB
/// per row at GLM shapes); the schedule (combo lists, offsets, the `mul1_gemm_grp` items of the
/// small experts); 0 below [`TC2_MIN_ROWS`] tokens
pub fn moe_tc2_bytes(hidden: usize, inter: usize, experts: usize, topk: usize, tokens: usize, cfg: &Tc2Cfg) -> u64 {
    if tokens < cfg.min_rows.max(1) {
        return 0;
    }
    carve(&moe_tc2_parts(hidden, inter, experts, topk, tokens, cfg)).1 as u64
}

/// #186: the host side of one schedule on [`MoeTc2`] ([`Tc2Sched::new`]): which experts take the
/// grouped tensor-core pass, the `mul1_gemm_grp` items of the others, and the device arrays
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Tc2Sched {
    /// per big expert (at least the minimum rows, schedule order): the expert id
    pub big_e: Vec<i32>,
    /// per big expert: its first packed row in `list_big` (len big + 1)
    pub row_off: Vec<i32>,
    /// per big expert: its first row tile of `bm` rows (len big + 1)
    pub mt_pre: Vec<i32>,
    /// the big experts' combos, packed in schedule order
    pub list_big: Vec<i32>,
    /// per packed row: its expert id
    pub row_e: Vec<i32>,
    /// the small experts' combos, packed in schedule order
    pub list2: Vec<i32>,
    /// the small experts' work items `[expert, first entry of list2, rows]`
    pub work2: Vec<[i32; 3]>,
    /// the rows of every item of `work2`
    pub rows2: Vec<usize>,
    /// per item index i of the schedule (len items + 1): small items before i
    pub small_pre: Vec<usize>,
    /// per item index i (len items + 1): big experts whose first item is before i
    pub big_pre: Vec<usize>,
    /// per item index: the item belongs to a big expert and is not its first
    inner_big: Vec<bool>,
}

impl Tc2Sched {
    /// the split of a schedule's `work` items over `list` (an [`ExpertMajor`]'s): experts of at
    /// least `min_rows` rows big, row tiles of `bm` rows
    pub fn new(work: &[[i32; 3]], list: &[i32], min_rows: usize, bm: usize) -> Tc2Sched {
        let mut s = Tc2Sched { row_off: vec![0], mt_pre: vec![0], ..Default::default() };
        let n = work.len();
        let mut a = 0;
        while a < n {
            let e = work[a][0];
            let (mut b, mut rows) = (a, 0usize);
            while b < n && work[b][0] == e {
                rows += work[b][2] as usize;
                b += 1;
            }
            let big = rows >= min_rows;
            for (i, it) in work.iter().enumerate().take(b).skip(a) {
                s.small_pre.push(s.work2.len());
                s.big_pre.push(s.big_e.len());
                s.inner_big.push(big && i > a);
                let (f, r) = (it[1] as usize, it[2] as usize);
                if big {
                    s.list_big.extend_from_slice(&list[f..f + r]);
                    s.row_e.extend(std::iter::repeat_n(e, r));
                } else {
                    s.work2.push([e, s.list2.len() as i32, r as i32]);
                    s.rows2.push(r);
                    s.list2.extend_from_slice(&list[f..f + r]);
                }
            }
            if big {
                s.big_e.push(e);
                s.row_off.push(s.list_big.len() as i32);
                s.mt_pre.push(s.mt_pre.last().unwrap() + rows.div_ceil(bm) as i32);
            }
            a = b;
        }
        s.small_pre.push(s.work2.len());
        s.big_pre.push(s.big_e.len());
        s
    }

    /// the `work2` items and the big experts of the schedule's items `items`; refused by name for
    /// a range that splits a big expert
    pub fn split(&self, items: std::ops::Range<usize>) -> (std::ops::Range<usize>, std::ops::Range<usize>) {
        assert!(items.end < self.small_pre.len(), "glm5_moe: items {items:?} of a schedule of {}", self.small_pre.len() - 1);
        for i in [items.start, items.end] {
            assert!(!self.inner_big.get(i).copied().unwrap_or(false), "glm5_moe: items {items:?} split a tensor-core expert at item {i}");
        }
        (self.small_pre[items.start]..self.small_pre[items.end], self.big_pre[items.start]..self.big_pre[items.end])
    }
}

/// #186 `CROW_GLM_MOE_TC=2`: the grouped tensor-core expert path of a [`GpuMoeGroupedPlan`]. The
/// experts of an `experts_items` range with at least `min_rows` rows go in batches of up to
/// `slots` experts, each batch: `mul1_tc2_recon` (gate and up of every expert to FP16),
/// `mul1_tc2_in` (H (x * suh) of gate and up, FP16), one grouped `mul1_tc2_gemm*` launch for gate
/// and up (FP32 accumulate, epilogue H / 128 * svh into `g` / `u`), `mul1_tc2_recon` (down),
/// `mul1_tc2_act` (clamped SwiGLU, H (. * suh), FP16), one grouped GEMM launch for down into `ye`;
/// the other experts' items run `mul1_gemm_grp` in one call (their rows bit for bit the default
/// path's). Not bit-identical to `mul1_gemm_grp`.
pub struct MoeTc2 {
    region: CUdeviceptr,
    /// the knobs
    pub cfg: Tc2Cfg,
    /// packed rows per batch (`cfg.rows` raised to the plan's tokens)
    rows: usize,
    w: CUdeviceptr,
    xg: CUdeviceptr,
    xu: CUdeviceptr,
    g: CUdeviceptr,
    u: CUdeviceptr,
    list_big: CUdeviceptr,
    row_e: CUdeviceptr,
    big_e: CUdeviceptr,
    row_off: CUdeviceptr,
    mt_pre: CUdeviceptr,
    work2: CUdeviceptr,
    list2: CUdeviceptr,
    sch: std::cell::RefCell<Tc2Sched>,
    f_recon: u64,
    f_in: u64,
    f_act: u64,
    f_gemm64: u64,
    f_gemm128: u64,
    f_gemm64f: u64,
    f_gemm128f: u64,
}

impl MoeTc2 {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(geo: &MoeGeo, tokens: usize, cfg: Tc2Cfg, mk: &mul1::Kernels) -> MoeTc2 {
        let (h, i) = (geo.hidden, geo.expert_inter);
        let rows = cfg.rows_for(tokens);
        assert!(cfg.slots > 0 && matches!(cfg.bm, 64 | 128), "glm5_moe: grouped tensor-core knobs {cfg:?}");
        assert!(h % 128 == 0 && i % 128 == 0 && (h / 16) * (i / 16) % 8 == 0 && rows <= 65_535, "glm5_moe: grouped tensor-core shapes H {h} I {i} rows {rows}");
        let (off, bytes) = carve(&moe_tc2_parts(h, i, geo.experts, geo.topk, tokens, &cfg));
        let region = cuda::alloc_named("glm5 MoE grouped tensor-core expert scratch", bytes);
        let at = |n: usize| region + off[n] as u64;
        MoeTc2 {
            region,
            cfg,
            rows,
            w: at(0),
            xg: at(1),
            xu: at(2),
            g: at(3),
            u: at(4),
            list_big: at(5),
            row_e: at(6),
            big_e: at(7),
            row_off: at(8),
            mt_pre: at(9),
            work2: at(10),
            list2: at(11),
            sch: std::cell::RefCell::new(Tc2Sched::default()),
            f_recon: mk.module.get("mul1_tc2_recon") as u64,
            f_in: mk.module.get("mul1_tc2_in") as u64,
            f_act: mk.module.get("mul1_tc2_act") as u64,
            f_gemm64: mk.module.get("mul1_tc2_gemm64") as u64,
            f_gemm128: mk.module.get("mul1_tc2_gemm128") as u64,
            f_gemm64f: mk.module.get("mul1_tc2_gemm64f") as u64,
            f_gemm128f: mk.module.get("mul1_tc2_gemm128f") as u64,
        }
    }

    /// the split of an uploaded schedule to the device (blocking copies)
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading the schedule arrays is pending.
    unsafe fn upload(&self, work: &[[i32; 3]], list: &[i32]) {
        let s = Tc2Sched::new(work, list, self.cfg.min_rows.max(1), self.cfg.bm);
        for (d, v) in [(self.list_big, &s.list_big), (self.row_e, &s.row_e), (self.big_e, &s.big_e), (self.row_off, &s.row_off), (self.mt_pre, &s.mt_pre), (self.list2, &s.list2)] {
            if !v.is_empty() {
                cuda::to_i32_into(d, v);
            }
        }
        if !s.work2.is_empty() {
            cuda::to_i32_into(self.work2, &s.work2.iter().flatten().copied().collect::<Vec<_>>());
        }
        *self.sch.borrow_mut() = s;
    }

    /// # Safety
    /// No launch on the scratch is pending.
    pub unsafe fn free(&mut self) {
        cuda::free_dev(&mut self.region);
    }
}

/// #186: the tensor-core expert path of a [`GpuMoeGroupedPlan`] (`CROW_GLM_MOE_TC`)
enum Tc {
    /// mode 1: per expert ([`MoeTc`])
    Per(MoeTc),
    /// mode 2: grouped ([`MoeTc2`])
    Grouped(MoeTc2),
}

impl Tc {
    unsafe fn free(&mut self) {
        match self {
            Tc::Per(t) => t.free(),
            Tc::Grouped(t) => t.free(),
        }
    }
}

/// The expert-major schedule of one prompt call, the prefill of 0xSero's glm53-flash-offload
/// (exllamav3's grouped MoE): every selected expert is fetched once for the call and applied to
/// all rows routed to it. Built on the host from the call's `[t][topk]` router ids (the routing
/// sync the prompt call has anyway).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ExpertMajor {
    pub topk: usize,
    /// the distinct selected experts, ascending
    pub experts: Vec<u32>,
    /// every combo `c = row * topk + pick` once, grouped by expert in `experts` order, `c`
    /// ascending within an expert
    pub list: Vec<i32>,
    /// work items `[expert, first entry of list, rows]`, `1 <= rows <= tile`, in `experts` order
    pub work: Vec<[i32; 3]>,
    /// per expert of `experts`: the end of its work items in `work`
    work_end: Vec<usize>,
}

impl ExpertMajor {
    /// the schedule of `ids` (`[t][topk]` i32, pick order) over `experts` experts in work items
    /// of up to `tile` rows; refused by name for an id outside the experts or a ragged selection
    pub fn new(ids: &[i32], topk: usize, experts: usize, tile: usize) -> Result<ExpertMajor, String> {
        if topk == 0 || tile == 0 || ids.is_empty() || ids.len() % topk != 0 {
            return Err(format!("glm5_moe: a selection of {} ids is no [rows][{topk}]", ids.len()));
        }
        let mut start = vec![0usize; experts + 1];
        for &e in ids {
            if e < 0 || e as usize >= experts {
                return Err(format!("glm5_moe: the router selected id {e}, outside 0..{experts}"));
            }
            start[e as usize + 1] += 1;
        }
        for e in 0..experts {
            start[e + 1] += start[e];
        }
        let mut fill = start.clone();
        let mut list = vec![0i32; ids.len()];
        for (c, &e) in ids.iter().enumerate() {
            list[fill[e as usize]] = c as i32;
            fill[e as usize] += 1;
        }
        let (mut ex, mut work, mut work_end) = (Vec::new(), Vec::new(), Vec::new());
        for e in 0..experts {
            let (mut f, end) = (start[e], start[e + 1]);
            if f == end {
                continue;
            }
            ex.push(e as u32);
            while f < end {
                let r = tile.min(end - f);
                work.push([e as i32, f as i32, r as i32]);
                f += r;
            }
            work_end.push(work.len());
        }
        Ok(ExpertMajor { topk, experts: ex, list, work, work_end })
    }

    /// pseudo-rows of [`ExpertMajor::sel`]
    pub fn pseudo_rows(&self) -> usize {
        self.experts.len().div_ceil(self.topk)
    }

    /// The selection the tier hook serves (`ExpertTiers::tables_for_chunk`): the distinct experts
    /// packed `topk` per pseudo-row in `experts` order, the last pseudo-row padded with its own
    /// first expert. Each expert is in exactly one pseudo-row, so whatever pseudo-row sub-batches
    /// the tiers split it into, every expert is put in place once for the call.
    pub fn sel(&self) -> Vec<i32> {
        let k = self.topk;
        let mut v = Vec::with_capacity(self.pseudo_rows() * k);
        for row in self.experts.chunks(k) {
            v.extend(row.iter().map(|&e| e as i32));
            v.extend(std::iter::repeat_n(row[0] as i32, k - row.len()));
        }
        v
    }

    /// the work items of the experts in pseudo-rows `r0 .. r0 + rows` (a contiguous range)
    pub fn work_of_rows(&self, r0: usize, rows: usize) -> std::ops::Range<usize> {
        let n = self.experts.len();
        let (a, b) = ((r0 * self.topk).min(n), ((r0 + rows) * self.topk).min(n));
        if a >= b {
            return 0..0;
        }
        (if a == 0 { 0 } else { self.work_end[a - 1] })..self.work_end[b - 1]
    }
}

/// the most work items of a [`GpuMoeGroupedPlan`] call of `tokens` rows over `experts` experts
/// top-`topk`: every expert's last item may be short
pub fn grouped_work_cap(experts: usize, topk: usize, tokens: usize) -> usize {
    let c = tokens * topk;
    c.div_ceil(GROUP_ROWS) + experts.min(c)
}

/// #196: the most combos one `mul1_gemm_grp` piece holds gate / up activations for (`ge` / `ue`
/// `[piece][expert_inter]`, 64 MiB at GLM-5.3-Flash shapes): a sub-batch's work items run in
/// pieces of at most this many combos, each gate / up then down (positions only, every row's
/// bits unchanged)
pub const GROUP_PIECE_COMBOS: usize = 4096;

/// #196: the region of a [`GpuMoeGroupedPlan`]: the MoE layer's buffers (`ye` `[c][hidden]`, `ys`
/// `[t][hidden]`, `ge` / `ue` `[piece][expert_inter]`, the shared expert's g / u / h) and, over the
/// same bytes, the dense FFN's g / u / h of `tokens` rows (a call runs one or the other): the
/// offsets of the MoE buffers and the region's size, the larger of the two
fn grouped_region(hidden: usize, topk: usize, expert_inter: usize, shared_inter: usize, dense_inter: usize, tokens: usize, piece: usize) -> (Vec<usize>, usize) {
    let (t, c) = (tokens, tokens * topk);
    let (off, moe) = carve(&[c * hidden * 4, t * hidden * 4, piece * expert_inter * 4, piece * expert_inter * 4, GpuFfnPlan::region_bytes(shared_inter, t)]);
    (off, moe.max(GpuFfnPlan::region_bytes(dense_inter, t)))
}

/// The device bytes of a [`GpuMoeGroupedPlan`] of `tokens` rows, without its parameter arrays,
/// from the model's geometry alone (the planner's booking, `manager::glm5_chunk_scratch_bytes`):
/// logits, ids, wts, the combo list, the work items, and its region (#196: `ye`, `ys`, `ge` /
/// `ue` of one piece of at most [`GROUP_PIECE_COMBOS`] combos and the shared expert's g / u / h,
/// shared with the dense FFN plan of `tokens` rows).
pub fn grouped_plan_bytes(hidden: usize, experts: usize, topk: usize, expert_inter: usize, shared_inter: usize, dense_inter: usize, tokens: usize) -> u64 {
    let c = tokens * topk;
    let own = 4 * (tokens * experts + 2 * c + c + 3 * grouped_work_cap(experts, topk, tokens));
    (own + grouped_region(hidden, topk, expert_inter, shared_inter, dense_inter, tokens, c.min(GROUP_PIECE_COMBOS)).1) as u64
}

/// One MoE layer of a prompt call of up to `tokens` rows, expert-major ([`ExpertMajor`]): the
/// router as [`GpuMoePlan::route`], then per tier sub-batch two `mul1_gemm_grp` launches (gate
/// and up in one, down with the clamp fused into its input), then the shared expert and
/// `glm5_moe_combine` once. Every routed output row `ye[c]` has the bits of [`GpuMoePlan`]'s
/// T = 1 slot of combo `c`, so `y` has the bits of [`GpuMoePlan::run`]. Scratch: `ye`
/// `[c][hidden]` for `c = tokens * topk` combos, `ge`, `ue` `[piece][inter]` (#196: the work
/// items run in pieces of at most `piece` combos), no per-slot GEMV plans; one region holds them
/// and the shared expert, and over the same bytes the dense FFN plan of `tokens` rows
/// ([`GpuMoeGroupedPlan::dense`]; about 0.18 MB per row at GLM-5.3-Flash shapes against
/// GpuMoePlan's about 5 MB).
pub struct GpuMoeGroupedPlan {
    pub geo: MoeGeo,
    pub tokens: usize,
    prm_kh: CUdeviceptr,
    prm_route: CUdeviceptr,
    prm_f: CUdeviceptr,
    prm_kh2: CUdeviceptr,
    prm_gu: CUdeviceptr,
    prm_d: CUdeviceptr,
    /// `[T][E]` router logits
    pub logits: CUdeviceptr,
    /// `[T][K]` i32 expert ids, pick order
    pub ids: CUdeviceptr,
    /// `[T][K]` f32 routing weights
    pub wts: CUdeviceptr,
    ge: CUdeviceptr,
    ue: CUdeviceptr,
    /// `[T * K][H]` the routed experts' outputs (unweighted)
    pub ye: CUdeviceptr,
    /// `[T][H]` the shared expert's output
    pub ys: CUdeviceptr,
    pub shared: GpuFfnPlan,
    /// #196: the dense FFN of up to `tokens` rows, over the plan's region (a dense layer's call)
    pub dense: GpuFfnPlan,
    list: CUdeviceptr,
    work: CUdeviceptr,
    work_cap: usize,
    /// #196: the region `ye`, `ys`, `ge`, `ue`, `shared` and `dense` are views into
    region: CUdeviceptr,
    /// #196: the most combos of one gate / up piece (`ge` / `ue` rows)
    piece: usize,
    /// #196: the rows of every work item of the uploaded schedule (host copy, for the pieces)
    item_rows: std::cell::RefCell<Vec<usize>>,
    /// `mul1_gemm_grp` (`CUfunction` as an integer, as `LaneBuf::ev`)
    grp: u64,
    /// #186: the uploaded schedule's work items (host copy, for the tensor-core experts)
    item_work: std::cell::RefCell<Vec<[i32; 3]>>,
    /// #186: the tensor-core expert path (`CROW_GLM_MOE_TC=1` and at least `TC_MIN_ROWS` tokens,
    /// or `CROW_GLM_MOE_TC=2` and at least `TC2_MIN_ROWS`)
    tc: Option<Tc>,
}

impl GpuMoeGroupedPlan {
    /// the most work items a call of `tokens` rows can have: every expert's last item may be short
    pub fn work_cap(geo: &MoeGeo, tokens: usize) -> usize {
        grouped_work_cap(geo.experts, geo.topk, tokens)
    }

    /// the device bytes [`GpuMoeGroupedPlan::new`] allocates, without its parameter arrays
    /// ([`grouped_plan_bytes`] of the plan's geometry)
    pub fn bytes(geo: &MoeGeo, tokens: usize) -> u64 {
        grouped_plan_bytes(geo.hidden, geo.experts, geo.topk, geo.expert_inter, geo.shared_inter, geo.dense_inter, tokens)
    }

    /// # Safety
    /// A CUDA context is current; `mk` is the module the launches run on.
    pub unsafe fn new(geo: &MoeGeo, tokens: usize, mk: &mul1::Kernels) -> GpuMoeGroupedPlan {
        GpuMoeGroupedPlan::with_piece(geo, tokens, mk, GROUP_PIECE_COMBOS)
    }

    /// [`GpuMoeGroupedPlan::new`] with gate / up pieces of at most `piece` combos (`usize::MAX`:
    /// one piece for every combo of `tokens` rows, the pre-#196 scratch)
    ///
    /// # Safety
    /// As [`GpuMoeGroupedPlan::new`].
    pub unsafe fn with_piece(geo: &MoeGeo, tokens: usize, mk: &mul1::Kernels, piece: usize) -> GpuMoeGroupedPlan {
        let (h, e, k, i) = (geo.hidden, geo.experts, geo.topk, geo.expert_inter);
        let c = tokens * k;
        let piece = piece.min(c);
        assert!(piece >= GROUP_ROWS.min(c), "glm5_moe: gate / up pieces of {piece} combos hold no work item of {GROUP_ROWS}");
        assert!(tokens > 0 && e <= kernels::glm5_moe::ROUTER_THREADS && (1..=kernels::glm5_moe::MAXK).contains(&k) && k <= e);
        let [sg, su, sd] = geo.record_specs();
        let (s_gu, s_d) = (mul1::ksplit(h), mul1::ksplit(i));
        assert!(h / s_gu <= GROUP_XROWS && i / s_d <= GROUP_XROWS && h % 128 == 0 && i % 128 == 0, "glm5_moe: grouped GEMM shapes H {h} I {i}");
        let work_cap = Self::work_cap(geo, tokens);
        assert!(work_cap <= 65_535, "glm5_moe: {work_cap} work items exceed the grid");
        let prm = |v: [usize; 16]| v.map(|x| i32::try_from(x).expect("glm5_moe: parameter beyond i32"));
        // [k, n, S, n32, bits, half, in_div, mode, limit bits, (tr, suh, svh) x 2, loc]; #196 loc:
        // gate / up write and down reads `ge` / `ue` at the entry's position in its piece
        let gu = prm([h, i, s_gu, sg.n32(), sg.bits as usize, sg.half as usize, k, 0, 0, sg.tr_off, sg.suh_off, sg.svh_off, su.tr_off, su.suh_off, su.svh_off, 1]);
        let mut d = prm([i, h, s_d, sd.n32(), sd.bits as usize, sd.half as usize, 1, 1, 0, sd.tr_off, sd.suh_off, sd.svh_off, 0, 0, 0, 1]);
        d[8] = geo.swiglu_limit.to_bits() as i32;
        let (off, bytes) = grouped_region(h, k, i, geo.shared_inter, geo.dense_inter, tokens, piece);
        let region = cuda::alloc_named("glm5 grouped MoE / dense FFN region", bytes);
        let at = |n: usize| region + off[n] as u64;
        GpuMoeGroupedPlan {
            geo: *geo,
            tokens,
            prm_kh: i32_dev(&[h]),
            prm_route: i32_dev(&[e, k]),
            prm_f: cuda::to_f32_dev(&[geo.routed_scaling, geo.swiglu_limit]),
            prm_kh2: i32_dev(&[k, h]),
            prm_gu: cuda::to_i32_dev(&gu),
            prm_d: cuda::to_i32_dev(&d),
            logits: cuda::alloc_zeroed(tokens * e * 4),
            ids: cuda::alloc_zeroed(c * 4),
            wts: cuda::alloc_zeroed(c * 4),
            ye: at(0),
            ys: at(1),
            ge: at(2),
            ue: at(3),
            shared: GpuFfnPlan::new_in(h, geo.shared_inter, tokens, geo.swiglu_limit, at(4)),
            dense: GpuFfnPlan::new_in(h, geo.dense_inter, tokens, geo.swiglu_limit, region),
            list: cuda::alloc_zeroed(c * 4),
            work: cuda::alloc_zeroed(work_cap * 3 * 4),
            work_cap,
            region,
            piece,
            item_rows: std::cell::RefCell::new(Vec::new()),
            grp: mk.module.get(GROUP_ENTRY) as u64,
            item_work: std::cell::RefCell::new(Vec::new()),
            tc: match moe_tc_mode_from_env() {
                1 if tokens >= TC_MIN_ROWS => Some(Tc::Per(MoeTc::new(h, i, tokens, mk))),
                2 if tokens >= TC2_MIN_ROWS => Some(Tc::Grouped(MoeTc2::new(geo, tokens, Tc2Cfg::default_for(tokens), mk))),
                _ => None,
            },
        }
    }

    /// #186: switch the tensor-core expert path on or off (the A/B arm of the tests and the
    /// micro-bench; `CROW_GLM_MOE_TC` picks it at construction)
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this plan is pending.
    pub unsafe fn set_tc(&mut self, on: bool, mk: &mul1::Kernels) {
        self.set_tc_mode(if on { 1 } else { 0 }, mk);
    }

    /// #186: switch to tensor-core mode `mode` (0 off, 1 per expert, 2 grouped with
    /// [`Tc2Cfg::default_for`]; `CROW_GLM_MOE_TC` picks it at construction). Upload the schedule
    /// after a switch.
    ///
    /// # Safety
    /// As [`GpuMoeGroupedPlan::set_tc`].
    pub unsafe fn set_tc_mode(&mut self, mode: u8, mk: &mul1::Kernels) {
        if mode == 2 {
            return self.set_tc2(Tc2Cfg::default_for(self.tokens), mk);
        }
        if matches!((&self.tc, mode), (None, 0) | (Some(Tc::Per(_)), 1)) {
            return;
        }
        if let Some(mut t) = self.tc.take() {
            t.free();
        }
        if mode == 1 {
            self.tc = Some(Tc::Per(MoeTc::new(self.geo.hidden, self.geo.expert_inter, self.tokens, mk)));
        }
    }

    /// #186: the grouped tensor-core path (mode 2) with the knobs `cfg` (rebuilt when they
    /// differ); upload the schedule after a switch
    ///
    /// # Safety
    /// As [`GpuMoeGroupedPlan::set_tc`].
    pub unsafe fn set_tc2(&mut self, cfg: Tc2Cfg, mk: &mul1::Kernels) {
        if matches!(&self.tc, Some(Tc::Grouped(t)) if t.cfg == cfg) {
            return;
        }
        if let Some(mut t) = self.tc.take() {
            t.free();
        }
        self.tc = Some(Tc::Grouped(MoeTc2::new(&self.geo, self.tokens, cfg, mk)));
    }

    /// #196: the most combos of one gate / up piece
    pub fn piece(&self) -> usize {
        self.piece
    }

    /// the router on the first `t` rows of `x` into `ids` / `wts` (the launches of
    /// [`GpuMoePlan::route`], `t` rows)
    ///
    /// # Safety
    /// As [`GpuMoePlan::route`]; `1 <= t <= tokens`.
    pub unsafe fn route(&self, kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuMoeWeights, x: CUdeviceptr, t: usize) {
        assert!((1..=self.tokens).contains(&t), "glm5_moe: route {t} rows of a {}-row plan", self.tokens);
        launch_v(kn.f("gemv_bf16_b"), self.geo.experts as u32, t as u32, 1, 256, &[w.router, x, self.logits, self.prm_kh]);
        launch_v(gk.router, t as u32, 1, 1, kernels::glm5_moe::ROUTER_THREADS as u32, &[self.logits, w.bias, self.ids, self.wts, self.prm_route, self.prm_f]);
    }

    /// the schedule's combo list and work items to the device (blocking copies)
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading `list` / `work` is pending.
    pub unsafe fn upload(&self, s: &ExpertMajor) {
        assert!(s.list.len() <= self.tokens * self.geo.topk && s.work.len() <= self.work_cap, "glm5_moe: a schedule of {} combos / {} items", s.list.len(), s.work.len());
        cuda::to_i32_into(self.list, &s.list);
        let flat: Vec<i32> = s.work.iter().flatten().copied().collect();
        cuda::to_i32_into(self.work, &flat);
        *self.item_rows.borrow_mut() = s.work.iter().map(|w| w[2] as usize).collect();
        *self.item_work.borrow_mut() = s.work.clone();
        if let Some(Tc::Grouped(t)) = &self.tc {
            t.upload(&s.work, &s.list);
        }
    }

    /// queue the routed experts of work items `items` of the uploaded schedule through `table`:
    /// `ye[c]` of their combos from rows `c / topk` of `x`, in pieces of at most
    /// [`GpuMoeGroupedPlan::piece`] combos (#196), two launches each (gate / up, then down)
    ///
    /// # Safety
    /// `table` points every expert of these items at a readable record until the launches
    /// finished; `x` holds the call's rows.
    pub unsafe fn experts_items(&self, table: CUdeviceptr, x: CUdeviceptr, items: std::ops::Range<usize>) {
        let tc = match &self.tc {
            None => return self.experts_items_grp(table, x, items),
            Some(Tc::Grouped(t)) => return self.experts_items_tc2(t, table, x, items),
            Some(Tc::Per(t)) => t,
        };
        // #186: an expert of at least TC_MIN_ROWS rows in the range takes the tensor cores, the
        // items between such experts mul1_gemm_grp as before
        let work = self.item_work.borrow();
        assert!(items.end <= work.len(), "glm5_moe: items {items:?} of a schedule of {}", work.len());
        let (mut a, mut from) = (items.start, items.start);
        while a < items.end {
            let e = work[a][0];
            let (mut b, mut rows) = (a, 0usize);
            while b < items.end && work[b][0] == e {
                rows += work[b][2] as usize;
                b += 1;
            }
            if rows >= TC_MIN_ROWS {
                self.experts_items_grp(table, x, from..a);
                self.expert_tc(tc, table, x, e as u64, work[a][1] as usize, rows);
                from = b;
            }
            a = b;
        }
        self.experts_items_grp(table, x, from..items.end);
    }

    /// #186: the `rows` list entries from `f` of expert `e` on the tensor cores into `ye` ([`MoeTc`])
    unsafe fn expert_tc(&self, tc: &MoeTc, table: CUdeviceptr, x: CUdeviceptr, e: u64, f: usize, rows: usize) {
        use cudarc::driver::sys::CUfunction;
        let (h, i, k) = (self.geo.hidden, self.geo.expert_inter, self.geo.topk);
        let [sg, su, sd] = self.geo.record_specs();
        let u = |v: usize| v as u64;
        let (fi, fr, fo, fg) = (tc.f_in as CUfunction, tc.f_recon as CUfunction, tc.f_out as CUfunction, tc.f_gemm as CUfunction);
        for (m, s) in [sg, su, sd].iter().enumerate() {
            launch_v(fr, (s.n / 128) as u32, (s.k / 16) as u32, 1, 256, &[table, tc.w[m], e, u(s.k), u(s.n), u(s.n32()), s.bits as u64, s.half as u64, u(s.tr_off)]);
        }
        let mut r0 = 0;
        while r0 < rows {
            let (r, f0) = (tc.rows.min(rows - r0), u(f + r0));
            let tb = r.div_ceil(128) as u32;
            for (m, s, dst) in [(0usize, sg, tc.g), (1, su, tc.u)] {
                launch_v(fi, (h / 128) as u32, r as u32, 1, 32, &[table, self.list, x, x, tc.xh, e, f0, u(h), u(k), 0, u(s.suh_off), 0]);
                launch_v(fg, (i / 128) as u32, tb, 1, 256, &[tc.w[m], tc.xh, tc.part, u(h), u(i), u(r)]);
                launch_v(fo, (i / 128) as u32, r as u32, 1, 32, &[table, self.list, tc.part, dst, e, f0, u(i), u(s.svh_off), 0]);
            }
            launch_v(fi, (i / 128) as u32, r as u32, 1, 32, &[table, self.list, tc.g, tc.u, tc.xh, e, f0, u(i), 1, 1, u(sd.suh_off), self.geo.swiglu_limit.to_bits() as u64]);
            launch_v(fg, (h / 128) as u32, tb, 1, 256, &[tc.w[2], tc.xh, tc.part, u(i), u(h), u(r)]);
            launch_v(fo, (h / 128) as u32, r as u32, 1, 32, &[table, self.list, tc.part, self.ye, e, f0, u(h), u(sd.svh_off), 1]);
            r0 += r;
        }
    }

    /// the `mul1_gemm_grp` path of [`GpuMoeGroupedPlan::experts_items`] (the default)
    unsafe fn experts_items_grp(&self, table: CUdeviceptr, x: CUdeviceptr, items: std::ops::Range<usize>) {
        let rows = self.item_rows.borrow();
        assert!(items.end <= self.work_cap && items.end <= rows.len(), "glm5_moe: items {items:?} of a schedule of {}", rows.len());
        self.grp_on(table, x, self.work, self.list, &rows, items);
    }

    /// the `mul1_gemm_grp` launches of items `items` of the work items at `work` (device,
    /// `[expert, first entry of list, rows]`) over the combo list `list`, `rows` their row counts
    unsafe fn grp_on(&self, table: CUdeviceptr, x: CUdeviceptr, work: CUdeviceptr, list: CUdeviceptr, rows: &[usize], items: std::ops::Range<usize>) {
        if items.is_empty() {
            return;
        }
        let (h, i) = (self.geo.hidden, self.geo.expert_inter);
        let f = self.grp as cudarc::driver::sys::CUfunction;
        let mut a = items.start;
        while a < items.end {
            let (mut b, mut n) = (a, 0usize);
            while b < items.end && n + rows[b] <= self.piece {
                n += rows[b];
                b += 1;
            }
            assert!(b > a, "glm5_moe: a work item of {} rows in pieces of {}", rows[a], self.piece);
            let (wk, w) = (work + (a * 12) as u64, (b - a) as u32);
            launch_v(f, (i / 128) as u32, w, 2, 256, &[table, wk, list, x, x, self.ge, self.ue, self.prm_gu]);
            launch_v(f, (h / 128) as u32, w, 1, 256, &[table, wk, list, self.ge, self.ue, self.ye, self.ye, self.prm_d]);
            a = b;
        }
    }

    /// #186 `CROW_GLM_MOE_TC=2`: the items `items` on [`MoeTc2`]: the small experts' items in one
    /// `mul1_gemm_grp` call, the big experts in batches of up to `slots` experts
    unsafe fn experts_items_tc2(&self, tc: &MoeTc2, table: CUdeviceptr, x: CUdeviceptr, items: std::ops::Range<usize>) {
        let s = tc.sch.borrow();
        let (small, big) = s.split(items);
        self.grp_on(table, x, tc.work2, tc.list2, &s.rows2, small);
        let mut j = big.start;
        while j < big.end {
            let mut j1 = j + 1;
            while j1 < big.end && j1 - j < tc.cfg.slots && (s.row_off[j1 + 1] - s.row_off[j]) as usize <= tc.rows {
                j1 += 1;
            }
            assert!((s.row_off[j1] - s.row_off[j]) as usize <= tc.rows, "glm5_moe: an expert of {} rows in a scratch of {}", s.row_off[j1] - s.row_off[j], tc.rows);
            self.tc2_batch(tc, &s, table, x, j..j1);
            j = j1;
        }
    }

    /// #186: one batch of big experts `js` (indices into the [`Tc2Sched`]) on [`MoeTc2`]
    unsafe fn tc2_batch(&self, tc: &MoeTc2, s: &Tc2Sched, table: CUdeviceptr, x: CUdeviceptr, js: std::ops::Range<usize>) {
        use cudarc::driver::sys::CUfunction;
        let (h, i, k) = (self.geo.hidden, self.geo.expert_inter, self.geo.topk);
        let [sg, su, sd] = self.geo.record_specs();
        let u = |v: usize| v as u64;
        let (b0, nexp) = (js.start, js.len());
        let pbase = s.row_off[b0] as usize;
        let rows = s.row_off[js.end] as usize - pbase;
        let mtiles = (s.mt_pre[js.end] - s.mt_pre[b0]) as usize;
        let mat = h * i * 2;
        let slot = 2 * mat;
        let tiles = (h / 16) * (i / 16);
        let recon = |m0: usize, mats: u32| {
            launch_v(tc.f_recon as CUfunction, (tiles / 8) as u32, mats, nexp as u32, 256, &[table, tc.big_e, tc.w, u(b0), u(sg.n32()), sg.bits as u64, sg.half as u64, u(m0), u(sg.tr_off), u(su.tr_off), u(sd.tr_off), u(slot), u(mat)]);
        };
        let gemm = match (tc.cfg.bm, tc.cfg.fused) {
            (128, false) => tc.f_gemm128,
            (128, true) => tc.f_gemm128f,
            (_, false) => tc.f_gemm64,
            (_, true) => tc.f_gemm64f,
        } as CUfunction;
        let dec = [u(sg.n32()), sg.bits as u64, sg.half as u64];
        if !tc.cfg.fused {
            recon(0, 2);
        }
        launch_v(tc.f_in as CUfunction, h.div_ceil(1024) as u32, rows as u32, 2, 256, &[table, tc.list_big, tc.row_e, x, tc.xg, tc.xu, u(pbase), u(h), u(k), u(sg.suh_off), u(su.suh_off)]);
        launch_v(gemm, (mtiles * (i / 128) * 2) as u32, 1, 1, 128, &[tc.w, tc.xg, tc.xu, tc.big_e, tc.row_off, tc.mt_pre, tc.list_big, table, tc.g, tc.u, u(b0), u(nexp), u(h), u(i), 2, u(sg.svh_off), u(su.svh_off), 0, u(slot), u(mat), u(sg.tr_off), u(su.tr_off), dec[0], dec[1], dec[2]]);
        if !tc.cfg.fused {
            recon(2, 1);
        }
        launch_v(tc.f_act as CUfunction, i.div_ceil(1024) as u32, rows as u32, 1, 256, &[table, tc.row_e, tc.g, tc.u, tc.xg, u(pbase), u(i), u(sd.suh_off), self.geo.swiglu_limit.to_bits() as u64]);
        launch_v(gemm, (mtiles * (h / 128)) as u32, 1, 1, 128, &[tc.w, tc.xg, tc.xg, tc.big_e, tc.row_off, tc.mt_pre, tc.list_big, table, self.ye, self.ye, u(b0), u(nexp), u(i), u(h), 1, u(sd.svh_off), u(sd.svh_off), 1, u(slot), u(mat), u(sd.tr_off), u(sd.tr_off), dec[0], dec[1], dec[2]]);
    }

    /// queue the shared expert on the first `t` rows of `x` and the combine into `y`
    ///
    /// # Safety
    /// Every combo of the `t` rows has its `ye` row queued before.
    pub unsafe fn finish(&self, gk: &kernels::glm5_moe::Kernels, w: &GpuMoeWeights, x: CUdeviceptr, y: CUdeviceptr, t: usize) {
        self.shared.run_rows(gk, &w.shared, x, self.ys, t);
        launch_v(gk.combine, self.geo.hidden.div_ceil(256) as u32, t as u32, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
    }

    /// The whole layer on the first `t` rows with every selected record in `table`: route, one
    /// host sync for the ids, the schedule, every item, finish. The tiered prompt call
    /// (`Glm5Pass::call_with_expert_batches`) runs the same steps around its tier sub-batches.
    ///
    /// # Safety
    /// As [`GpuMoePlan::run`]; synchronizes.
    pub unsafe fn run(&self, kn: &kernels::Kernels, gk: &kernels::glm5_moe::Kernels, w: &GpuMoeWeights, table: CUdeviceptr, x: CUdeviceptr, y: CUdeviceptr, t: usize) -> Result<ExpertMajor, String> {
        self.route(kn, gk, w, x, t);
        cuda::sync();
        let s = ExpertMajor::new(&cuda::dtoh_i32(self.ids, t * self.geo.topk), self.geo.topk, self.geo.experts, GROUP_ROWS)?;
        self.upload(&s);
        self.experts_items(table, x, 0..s.work.len());
        self.finish(gk, w, x, y, t);
        Ok(s)
    }

    /// # Safety
    /// No launch of this plan is pending.
    pub unsafe fn free(&mut self) {
        for d in [
            &mut self.prm_kh,
            &mut self.prm_route,
            &mut self.prm_f,
            &mut self.prm_kh2,
            &mut self.prm_gu,
            &mut self.prm_d,
            &mut self.logits,
            &mut self.ids,
            &mut self.wts,
            &mut self.list,
            &mut self.work,
            &mut self.region,
        ] {
            cuda::free_dev(d);
        }
        (self.ge, self.ue, self.ye, self.ys) = (0, 0, 0, 0);
        self.shared.free();
        self.dense.free();
        if let Some(mut t) = self.tc.take() {
            t.free();
        }
    }
}

/// #188: worker threads of one CPU-lane pool run (the GPU host thread is worker 0): the #183 C1
/// point of record (8 threads: `cpu_mul1` FFN 20.77 GB/s, 0.456 ms per 3-bit expert, T 1)
pub const LANE_THREADS: usize = 8;

/// #188: the CPU lane's pinned host buffer of one plan (`[hidden]` x, `[topk]` record bases,
/// `[topk][hidden]` CPU outputs; cacheable, the CPU reads and writes it) and its event
struct LaneBuf {
    host: cuda::Pinned,
    /// `CUevent` as an integer (the plan stays `Send` like the other device handles)
    ev: u64,
}

impl LaneBuf {
    /// `t` rows of x, `c` combos' record bases and output rows
    unsafe fn new(h: usize, c: usize, t: usize) -> LaneBuf {
        LaneBuf { host: cuda::Pinned::alloc(t * h * 4 + c * 8 + c * h * 4), ev: cuda::event_create() as u64 }
    }
}

/// #188 CPU lane hand-off. `glm5_tiers::ExpertTiers::table_for` (the expert hook) posts the
/// combos of a decode call here; [`GpuMoePlan::experts`] of the same thread takes them when its
/// `table` is the posted one, so the expert hook and its call site (`glm5_model.rs`
/// `call_with_experts`) stay as they are. Anything else (no post, another table, `t > 1`, no CPU
/// combo) runs the GPU path unchanged; a post is consumed by the next `experts` call either way.
/// Every CPU combo's table entry stays a valid pinned record base, so the GPU path is also
/// correct for a post that is not taken.
pub mod lane {
    use cudarc::driver::sys::CUdeviceptr;
    use std::cell::RefCell;
    use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
    use std::sync::Arc;

    /// worker threads of the lane's pool run (`CROW_GLM_LANE_THREADS`, stored by
    /// `glm5_tiers::ExpertTiers::new`); 0 = [`super::LANE_THREADS`]
    pub static THREADS: AtomicUsize = AtomicUsize::new(0);

    /// the lane's thread count: [`THREADS`], or [`super::LANE_THREADS`] while it is 0
    pub fn threads() -> usize {
        match THREADS.load(Ordering::Relaxed) {
            0 => super::LANE_THREADS,
            n => n,
        }
    }

    /// where one combo (pick order) is computed
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Combo {
        /// on the GPU, from this record base (VRAM, pinned UVA or staging)
        Gpu(u64),
        /// on the CPU, from this host record (a pinned slot; `record.bytes` long)
        Cpu(*const u8),
    }

    /// #202 `CROW_GLM_RT2=1` (only `1` turns it on), read by `glm5_tiers::ExpertTiers::new`
    pub const ENV_RT2: &str = "CROW_GLM_RT2";

    /// `CROW_GLM_RT2=1`
    pub fn rt2_on() -> bool {
        std::env::var(ENV_RT2).ok().as_deref() == Some("1")
    }

    /// #202 RT2: the decode call queues the MoE input row to the CPU lane's host buffer before
    /// its router flag (`glm5_model` `call_inner`), so the lane reads x as soon as the host saw
    /// the flag; stored by `glm5_tiers::ExpertTiers::new` from [`rt2_on`]
    pub static X_EARLY: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

    /// #202 RT2: calls run by `GpuMoePlan::experts_rt2` since the process started (tests)
    pub static RT2_CALLS: AtomicU64 = AtomicU64::new(0);

    /// the CPU lane's wall time and expert count, summed (shared with the counters' owner)
    #[derive(Debug, Default)]
    pub struct Clock {
        ns: AtomicU64,
        experts: AtomicU64,
        runs: AtomicU64,
        /// #202 RT2: lane runs, and of them those that started / ended while the GPU still ran
        /// the call's resident experts (the lane overlapped the GPU's expert work)
        rt2_runs: AtomicU64,
        start_in_gpu: AtomicU64,
        end_in_gpu: AtomicU64,
    }

    impl Clock {
        pub fn add(&self, d: std::time::Duration, experts: usize) {
            self.ns.fetch_add(d.as_nanos() as u64, Ordering::Relaxed);
            self.experts.fetch_add(experts as u64, Ordering::Relaxed);
            self.runs.fetch_add(1, Ordering::Relaxed);
        }
        /// (nanoseconds in pool runs, experts computed, pool runs) since construction
        pub fn read(&self) -> (u64, u64, u64) {
            (self.ns.load(Ordering::Relaxed), self.experts.load(Ordering::Relaxed), self.runs.load(Ordering::Relaxed))
        }
        /// #202 RT2: one lane run, whether the GPU's resident experts were still running at its
        /// start and at its end
        pub fn add_rt2(&self, start_in_gpu: bool, end_in_gpu: bool) {
            self.rt2_runs.fetch_add(1, Ordering::Relaxed);
            self.start_in_gpu.fetch_add(start_in_gpu as u64, Ordering::Relaxed);
            self.end_in_gpu.fetch_add(end_in_gpu as u64, Ordering::Relaxed);
        }
        /// #202 RT2: (lane runs, runs started while the GPU ran its experts, runs that also
        /// ended before the GPU did)
        pub fn read_rt2(&self) -> (u64, u64, u64) {
            (self.rt2_runs.load(Ordering::Relaxed), self.start_in_gpu.load(Ordering::Relaxed), self.end_in_gpu.load(Ordering::Relaxed))
        }
    }

    /// #202 RT2 (`CROW_GLM_RT2=1`): a decode call (one row) with the stager. `event` is the
    /// stager's event behind the call's moves (a `CUevent` as an integer); `late[c]` marks the
    /// GPU combos whose record that batch writes or waits for (an NVMe landing, a staging copy, a
    /// VRAM entrant, a write-back still landing). The other GPU combos and the CPU lane (only
    /// resident records) run before the compute stream waits for `event`; the late ones after.
    #[derive(Debug)]
    pub struct Rt2 {
        pub event: u64,
        pub late: Vec<bool>,
    }

    /// one decode call's split
    #[derive(Debug)]
    pub struct Call {
        /// the device table of the call (`experts` takes the post only for this table)
        pub table: CUdeviceptr,
        /// one per combo, pick order (`[rows][topk]`)
        pub combos: Vec<Combo>,
        pub clock: Arc<Clock>,
        /// a `CUevent` (as an integer) the host waits for before it reads a CPU combo's record:
        /// the async moves that put the records in place (`CROW_GLM_STAGER`); `None` = in place
        pub ready: Option<u64>,
        /// #202 `CROW_GLM_RT2`: the early / late split (`None`: the former path)
        pub rt2: Option<Rt2>,
    }

    thread_local! {
        static POSTED: RefCell<Option<Call>> = const { RefCell::new(None) };
    }

    /// post (or, with `None`, clear) this thread's next call
    pub fn post(call: Option<Call>) {
        POSTED.with(|p| *p.borrow_mut() = call);
    }

    /// the posted call, if it is for `table`, a decode call of `tokens` rows (one, or a batched
    /// step of `CROW_GLM_MAX_BATCH`) with `tokens x topk` combos, at least one on the CPU; the
    /// post is consumed either way
    pub(crate) fn take(table: CUdeviceptr, tokens: usize, topk: usize) -> Option<Call> {
        let c = POSTED.with(|p| p.borrow_mut().take())?;
        let rt2 = c.rt2.as_ref().is_some_and(|r| tokens == 1 && r.late.len() == c.combos.len());
        (c.table == table && c.combos.len() == tokens * topk && (rt2 || c.combos.iter().any(|x| matches!(x, Combo::Cpu(_))))).then_some(c)
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

    // ---------------------------------------------------------------- prefill: expert-major

    /// `t` rows of `k` distinct picks each over `e` experts, skewed toward low ids (`skew` > 1
    /// concentrates the picks, as a real router's hot experts do)
    fn skewed_ids(t: usize, k: usize, e: usize, skew: f64, seed: u64) -> Vec<i32> {
        let mut rng = Rng(seed);
        let mut ids = Vec::with_capacity(t * k);
        for _ in 0..t {
            let mut row: Vec<i32> = Vec::with_capacity(k);
            while row.len() < k {
                let u = (rng.next() >> 11) as f64 / (1u64 << 53) as f64;
                let x = ((u.powf(skew) * e as f64) as usize).min(e - 1) as i32;
                if !row.contains(&x) {
                    row.push(x);
                }
            }
            ids.extend(row);
        }
        ids
    }

    /// The prompt call's expert-major schedule (0xSero's prefill): every combo exactly once,
    /// grouped by expert, work items of at most `GROUP_ROWS` rows covering each expert's rows in
    /// order; its pseudo-row selection names each selected expert exactly once, and pseudo-row
    /// sub-batches map to disjoint work ranges covering every item. Served through the tiers' own
    /// `serve_chunk` (LRU, the RTX 5090 plan's proportions V 50 + P 124 of 288, a 64-slot prefill
    /// set, cold), every selected expert is visited, and read from NVMe, at most once in the
    /// call: the prefill fetches each needed expert once. (The token-row selection the prompt
    /// call handed the tiers before visits experts in many sub-batches; printed for comparison.)
    #[test]
    fn glm5_moe_prompt_call_fetches_each_expert_once() {
        use crate::expert_cache::{ExpertCache, Policy, Scope};
        use crate::glm5_tiers::{serve_chunk, Dst, LayerSlots, Mover, Served, TierSizes};
        struct Count(u64);
        impl Mover for Count {
            fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
                self.0 += jobs.len() as u64;
                Ok(0)
            }
            fn landing_to_stage(&mut self, _: u32) {}
            fn pinned_to_stage(&mut self, _: u32, _: u32) {}
            fn vram_to_stage(&mut self, _: u32, _: u32) {}
            fn barrier(&mut self) {}
            fn vram_to_pinned(&mut self, _: u32, _: u32) {}
            fn stage_to_vram(&mut self, _: u32, _: u32) {}
        }
        let (e, k) = (288usize, 8usize);
        // the visits of every expert, the sub-batches and the NVMe reads of one call
        let serve = |sel: &[i32]| -> (Vec<u32>, usize, u64) {
            let sizes = TierSizes { vram: 50, pinned: 124 };
            let mut cache = ExpertCache::new(Policy::Lru, Scope::PerLayer, 1, e, sizes.vram, sizes.pinned).unwrap();
            let mut slots = LayerSlots::new(e, sizes);
            let mut m = Count(0);
            let mut visits = vec![0u32; e];
            let mut each = |_: usize, _: usize, s: &Served| -> Result<(), String> {
                for &(x, _) in &s.locs {
                    visits[x as usize] += 1;
                }
                Ok(())
            };
            let b = serve_chunk(&mut cache, 0, &mut slots, sel, k, 64, &mut m, &mut each).unwrap();
            (visits, b, m.0)
        };
        for (t, skew, seed) in [(1usize, 1.0, 1u64), (5, 1.0, 2), (32, 1.0, 3), (256, 1.0, 4), (256, 3.0, 5), (2048, 2.0, 6)] {
            let ids = skewed_ids(t, k, e, skew, seed);
            let s = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
            // the schedule
            let mut seen = vec![0u32; t * k];
            for &c in &s.list {
                seen[c as usize] += 1;
            }
            assert!(seen.iter().all(|&n| n == 1), "t {t}: every combo once");
            let mut want: Vec<u32> = ids.iter().map(|&x| x as u32).collect();
            want.sort_unstable();
            want.dedup();
            assert_eq!(s.experts, want, "t {t}: the distinct experts, ascending");
            let mut next = 0usize;
            for (j, &x) in s.experts.iter().enumerate() {
                let mine: Vec<(usize, [i32; 3])> = s.work.iter().copied().enumerate().filter(|(_, w)| w[0] == x as i32).collect();
                assert!(!mine.is_empty() && mine.iter().all(|(_, w)| (1..=GROUP_ROWS as i32).contains(&w[2])), "t {t} expert {x}: items");
                let first = next;
                for (wi, w) in &mine {
                    assert!(s.work_of_rows(j / k, 1).contains(wi), "t {t} expert {x}: item {wi} outside its pseudo-row's range");
                    assert_eq!(w[1] as usize, next, "t {t} expert {x}: items are consecutive runs of the list");
                    for &c in &s.list[w[1] as usize..(w[1] + w[2]) as usize] {
                        assert_eq!(ids[c as usize], x as i32, "t {t}: combo {c} is not expert {x}'s");
                    }
                    next += w[2] as usize;
                }
                let rows: Vec<usize> = s.list[first..next].iter().map(|&c| c as usize / k).collect();
                assert!(rows.windows(2).all(|r| r[0] < r[1]), "t {t} expert {x}: rows ascending");
            }
            assert_eq!(next, t * k);
            // the pseudo-rows and their work ranges
            let sel = s.sel();
            assert_eq!(sel.len(), s.pseudo_rows() * k);
            for r in 0..s.pseudo_rows() {
                let mut row = sel[r * k..(r + 1) * k].to_vec();
                row.sort_unstable();
                row.dedup();
                let names: Vec<i32> = s.experts[r * k..((r + 1) * k).min(s.experts.len())].iter().map(|&x| x as i32).collect();
                assert_eq!(row, names, "t {t} pseudo-row {r}");
            }
            for split in [1usize, 2, 3, 7] {
                let (mut at, mut r0) = (0usize, 0usize);
                while r0 < s.pseudo_rows() {
                    let rows = split.min(s.pseudo_rows() - r0);
                    let w = s.work_of_rows(r0, rows);
                    assert_eq!(w.start, at, "t {t}: pseudo-row ranges are disjoint and in order");
                    at = w.end;
                    r0 += rows;
                }
                assert_eq!(at, s.work.len(), "t {t}: the pseudo-rows cover every item");
            }
            // through the tiers: each selected expert served, and read from NVMe, at most once
            let (visits, batches, reads) = serve(&sel);
            for x in 0..e {
                assert_eq!(visits[x], u32::from(s.experts.contains(&(x as u32))), "t {t}: expert {x} served {} times in the call", visits[x]);
            }
            assert!(reads <= s.experts.len() as u64, "t {t}: {reads} NVMe reads for {} experts", s.experts.len());
            let (rv, rb, rr) = serve(&ids);
            eprintln!(
                "glm5_moe expert-major t {t} skew {skew}: {} experts, {} items; tiers {batches} sub-batches, {reads} NVMe reads, max visits 1 (token rows: {rb} sub-batches, {rr} NVMe reads, max visits {})",
                s.experts.len(),
                s.work.len(),
                rv.iter().max().unwrap()
            );
        }
        // refused by name
        assert!(ExpertMajor::new(&[0, 1, 2], 2, 8, 16).unwrap_err().contains("no [rows][2]"));
        assert!(ExpertMajor::new(&[0, 9], 2, 8, 16).unwrap_err().contains("outside 0..8"));
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
        // #191: the host's grid arithmetic matches the kernel's rows per block
        let src = crate::kernels::GLM5_MOE_SRC;
        let def = |name: &str| -> usize {
            let pat = format!("#define {name} ");
            let i = src.find(&pat).unwrap_or_else(|| panic!("no #define {name}")) + pat.len();
            src[i..].split_whitespace().next().unwrap().parse().unwrap()
        };
        assert_eq!(def("GLM5_FP4_RB"), kernels::glm5_moe::FP4_ROWS_PER_BLOCK);
        // one thread per 36-byte block, whole warps, at most the record's 256
        let rb = kernels::glm5_moe::FP4_ROWS_PER_BLOCK;
        for (rows, k, threads) in [(2048, 4096, 64), (4096, 2048, 32), (512, 4096, 64), (16384, 1536, 32), (4096, 16384, 256), (1, 64, 32), (4096, 32768, 256)] {
            assert_eq!(kernels::glm5_moe::fp4_launch(rows, k), (rows.div_ceil(rb) as u32, threads), "[{rows}, {k}]");
        }
    }

    /// #186: the tensor-core GEMMs' tiles are the host's grid arithmetic, and the planner books
    /// the row table exactly when a pass of the chunk holds it (`CROW_GLM_DENSE_GEMM=1` and calls
    /// of at least `TC_MIN_ROWS` rows)
    #[test]
    fn glm5_dense_tc_tiles_and_row_table_booking() {
        use crate::kernels::glm5_moe::{TCS_N, TCS_TILE, TC_MIN_ROWS, TC_ROWS_BYTES, TC_SMALL_MAX_ROWS, TC_TILE};
        let src = crate::kernels::GLM5_MOE_SRC;
        let def = |name: &str| -> usize {
            let pat = format!("#define {name} ");
            let i = src.find(&pat).unwrap_or_else(|| panic!("no #define {name}")) + pat.len();
            src[i..].split_whitespace().next().unwrap().parse().unwrap()
        };
        assert_eq!((def("GLM5_TC_BM"), def("GLM5_TC_BN"), def("GLM5_TCS_T"), 8 * def("GLM5_TCS_NI")), (TC_TILE, TC_TILE, TCS_TILE, TCS_N));
        assert!(TC_MIN_ROWS <= TC_SMALL_MAX_ROWS);
        let g = crate::geo::Glm5Geo::GLM_5_3_FLASH;
        for (chunk, extra) in [(1, 0), (TC_MIN_ROWS - 1, 0), (TC_MIN_ROWS, TC_ROWS_BYTES), (8192, TC_ROWS_BYTES)] {
            let on = crate::manager::glm5_chunk_scratch_bytes_tc(&g, chunk, 200_000, true);
            let off = crate::manager::glm5_chunk_scratch_bytes_tc(&g, chunk, 200_000, false);
            assert_eq!(on - off, extra, "chunk {chunk}");
        }
        assert_eq!(TC_ROWS_BYTES, 262_144);
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

    /// #188 CPU lane, synthetic GLM layer (the records of `glm5_moe_gpu_layer_matches_the_oracle`
    /// in cacheable pinned RAM, x = the oracle's MoE row 0, T 1). For combo masks from one CPU
    /// combo (first, last) through alternating, 6 of 8, 7 of 8 and all 8: every GPU combo's `ye`
    /// row has the bits of the GPU-only run, every CPU combo's row is within 1 - cosine <= 1e-6 of
    /// `expert_ffn_mul1_cpu` of its record (the lane's int16 branch at <= 2 rows is held to the
    /// accuracy bar, not to bits; `ye` filled with NaN before each run, so a row the
    /// lane fails to write shows), `y` = `glm5_moe_combine` of that `ye`, cosine to the oracle >=
    /// `COS_MIN`; max abs and cosine against the GPU-only `y` printed. A post for another table,
    /// or without a CPU combo, runs the GPU path: `y` bit-identical to the GPU-only run.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_cpu_lane_rows_are_the_cpu_and_gpu_experts() {
        use std::sync::Arc;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, gd, g) = (synth(), golden(), geo());
            let (h, k) = (4096usize, g.topk);
            let needed: Vec<u32> = {
                let mut v: Vec<u32> = gd.moe_ids.iter().map(|&e| e as u32).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let all: Vec<u8> = needed.iter().flat_map(|&e| record(e)).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3;
            let mut pinned = cuda::Pinned::alloc(all.len());
            pinned.write_bytes(0, &all);
            let pos = |e: u32| needed.iter().position(|&n| n == e).unwrap_or(0);
            let table: Vec<u64> = (0..g.experts as u32).map(|e| pinned.dev + (rb * pos(e)) as u64).collect();
            let mut tp = cuda::to_u64_dev(&table);
            let w = GpuMoeWeights { router: cuda::upload_dev(&le_u16(&s.router_w)), bias: cuda::to_f32_dev(&s.bias), shared: gpu_ffn(&s.shared) };
            let mut plan = GpuMoePlan::new(&g, 1);
            let x = &s.x_moe[..h];
            let (mut xd, mut yd, mut y2) = (cuda::to_f32_dev(x), cuda::alloc_zeroed(h * 4), cuda::alloc_zeroed(h * 4));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            lane::post(None);
            plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
            cuda::sync();
            let (y_gpu, ye_gpu) = (cuda::dtoh(yd, h), cuda::dtoh(plan.ye, k * h));
            let ids: Vec<u32> = cuda::dtoh_i32(plan.ids, k).into_iter().map(|v| v as u32).collect();
            let rec = |e: u32| &all[rb * pos(e)..rb * (pos(e) + 1)];
            let ye_cpu: Vec<Vec<f32>> = ids
                .iter()
                .map(|&e| {
                    let mut y = vec![0f32; h];
                    expert_ffn_mul1_cpu(&g, rec(e), x, &mut y, 8, Path::Auto).unwrap();
                    y
                })
                .collect();
            let host = pinned.host as *const u8;
            let combos = |mask: u32| -> Vec<lane::Combo> {
                ids.iter()
                    .enumerate()
                    .map(|(c, &e)| if mask >> c & 1 == 1 { lane::Combo::Cpu(host.add(rb * pos(e))) } else { lane::Combo::Gpu(table[e as usize]) })
                    .collect()
            };
            let clock = Arc::new(lane::Clock::default());
            let mut want_experts = 0u64;
            for mask in [0b0000_0001u32, 0b1000_0000, 0b0101_0101, 0b0011_1100, 0b1111_1100, 0b1111_1110, 0b1111_1111] {
                cuda::to_f32_into(plan.ye, &vec![f32::NAN; k * h]);
                lane::post(Some(lane::Call { table: tp, combos: combos(mask), clock: clock.clone(), ready: None, rt2: None }));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::sync();
                want_experts += mask.count_ones() as u64;
                let (y, ye) = (cuda::dtoh(yd, h), cuda::dtoh(plan.ye, k * h));
                for c in 0..k {
                    let got = &ye[c * h..(c + 1) * h];
                    if mask >> c & 1 == 1 {
                        let d = 1.0 - cosine(got, &ye_cpu[c]);
                        assert!(d <= 1e-6, "mask {mask:08b} combo {c} (CPU): the ye row is 1 - cos {d:.3e} from expert_ffn_mul1_cpu");
                    } else {
                        assert!(bits(got) == bits(&ye_gpu[c * h..(c + 1) * h]), "mask {mask:08b} combo {c} (GPU): the ye row differs");
                    }
                }
                launch_v(gk.combine, h.div_ceil(256) as u32, 1, 1, 256, &[plan.ye, plan.wts, plan.ys, y2, plan.prm_kh2]);
                cuda::sync();
                assert!(bits(&y) == bits(&cuda::dtoh(y2, h)), "mask {mask:08b}: y is not the combine of ye");
                let (co, cg) = (cosine(&y, &gd.moe_y[..h]), cosine(&y, &y_gpu));
                eprintln!(
                    "glm5_moe CPU lane mask {mask:08b} ({} CPU): vs GPU-only max abs {:.3e} at rms {:.3e}, 1 - cosine {:.2e}; vs oracle 1 - cosine {:.2e} (GPU-only {:.2e})",
                    mask.count_ones(),
                    max_abs(&y, &y_gpu),
                    rms(&y_gpu),
                    1.0 - cg,
                    1.0 - co,
                    1.0 - cosine(&y_gpu, &gd.moe_y[..h])
                );
                assert!(co >= COS_MIN, "mask {mask:08b}: cosine to the oracle {co:.9}");
            }
            let (ns, n_exp, runs) = clock.read();
            assert_eq!((n_exp, runs), (want_experts, 7), "the lane clock");
            eprintln!("glm5_moe CPU lane: {n_exp} experts in {runs} pool runs, {:.3} ms (test harness, not a measurement)", ns as f64 / 1e6);
            // posts the GPU path must not take: another table, no CPU combo
            for post in [lane::Call { table: tp + 8, combos: combos(0xFF), clock: clock.clone(), ready: None, rt2: None }, lane::Call { table: tp, combos: combos(0), clock: clock.clone(), ready: None, rt2: None }] {
                lane::post(Some(post));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::sync();
                assert!(bits(&cuda::dtoh(yd, h)) == bits(&y_gpu), "a post that does not apply changed y");
            }
            assert_eq!(clock.read().1, want_experts, "a post that does not apply ran the CPU");
            plan.free();
            free_ffn(w.shared);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut tp, &mut xd, &mut yd, &mut y2, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// #202 RT2 (`GpuMoePlan::experts_rt2`), synthetic GLM layer (the records of the lane test in
    /// cacheable pinned RAM, T 1). The stager's event is held: it sits behind a
    /// `cuStreamWaitValue64` on a side stream that waits for a mapped word the host raises only
    /// later. For CPU / late masks: `run` returns with the CPU lane done and the layer's output
    /// not written (the compute stream waits for the held event), the event behind the resident
    /// GPU experts completes while the hold stands (so those experts and the lane ran before the
    /// "landing"), and once the word is raised `y` has the bits of `experts_lane` for the same
    /// CPU set (a mask without a CPU combo: the GPU-only run's bits). The lane clock counts the
    /// runs that started while the GPU still ran its resident experts: the lane overlapped them.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_rt2_lane_and_resident_experts_run_before_the_stager_event() {
        use cudarc::driver::sys;
        use std::sync::Arc;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, gd, g) = (synth(), golden(), geo());
            let (h, k) = (4096usize, g.topk);
            let needed: Vec<u32> = {
                let mut v: Vec<u32> = gd.moe_ids.iter().map(|&e| e as u32).collect();
                v.sort_unstable();
                v.dedup();
                v
            };
            let all: Vec<u8> = needed.iter().flat_map(|&e| record(e)).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3;
            let mut pinned = cuda::Pinned::alloc(all.len());
            pinned.write_bytes(0, &all);
            let pos = |e: u32| needed.iter().position(|&n| n == e).unwrap_or(0);
            let table: Vec<u64> = (0..g.experts as u32).map(|e| pinned.dev + (rb * pos(e)) as u64).collect();
            let mut tp = cuda::to_u64_dev(&table);
            let w = GpuMoeWeights { router: cuda::upload_dev(&le_u16(&s.router_w)), bias: cuda::to_f32_dev(&s.bias), shared: gpu_ffn(&s.shared) };
            let mut plan = GpuMoePlan::new(&g, 1);
            let x = &s.x_moe[..h];
            let (mut xd, mut yd) = (cuda::to_f32_dev(x), cuda::alloc_zeroed(h * 4));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            lane::post(None);
            plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
            cuda::sync();
            let y_gpu = cuda::dtoh(yd, h);
            let ids: Vec<u32> = cuda::dtoh_i32(plan.ids, k).into_iter().map(|v| v as u32).collect();
            let host = pinned.host as *const u8;
            let combos = |mask: u32| -> Vec<lane::Combo> {
                ids.iter()
                    .enumerate()
                    .map(|(c, &e)| if mask >> c & 1 == 1 { lane::Combo::Cpu(host.add(rb * pos(e))) } else { lane::Combo::Gpu(table[e as usize]) })
                    .collect()
            };
            let clock = Arc::new(lane::Clock::default());
            // the held "stager": a side stream waiting for a mapped word, its event
            let mut word = cuda::Pinned::alloc(4096);
            std::ptr::write_bytes(word.host as *mut u8, 0, word.bytes);
            let side = cuda::stream_create_non_blocking();
            let held = cuda::event_create();
            let fin = cuda::event_create();
            let mut overlapped_cases = 0;
            for (i, (cpu, late)) in [(0b0000_0011u32, 0b1100_0000u32), (0b0101_0000, 0b0000_0101), (0b0000_0000, 0b1000_0001), (0b0000_1111, 0b1111_0000), (0b0011_1100, 0b0000_0000), (0b1111_1111, 0)].into_iter().enumerate() {
                // the former lane path for this CPU set
                lane::post((cpu != 0).then(|| lane::Call { table: tp, combos: combos(cpu), clock: clock.clone(), ready: None, rt2: None }));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::sync();
                let y_ref = cuda::dtoh(yd, h);
                if cpu == 0 {
                    assert!(bits(&y_ref) == bits(&y_gpu), "the GPU path without a post");
                }
                cuda::to_f32_into(yd, &vec![f32::NAN; h]);
                cuda::sync();
                let v = i as u64 + 1;
                cuda::ck(sys::cuStreamWaitValue64_v2(side, word.dev, v, 0));
                cuda::event_record(held, side);
                cuda::stream_query(side);
                let late_v: Vec<bool> = (0..k).map(|c| late >> c & 1 == 1 && cpu >> c & 1 == 0).collect();
                let runs0 = clock.read_rt2().0;
                lane::post(Some(lane::Call { table: tp, combos: combos(cpu), clock: clock.clone(), ready: None, rt2: Some(lane::Rt2 { event: held as u64, late: late_v.clone() }) }));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::event_record(fin, cuda::cur_stream());
                cuda::stream_query(cuda::cur_stream());
                // the lane ran (run returned) while the layer's end still waits for the hold
                assert_eq!(clock.read_rt2().0, runs0 + (cpu != 0) as u64, "case {i}: the lane's run");
                assert_eq!(sys::cuEventQuery(fin), sys::CUresult::CUDA_ERROR_NOT_READY, "case {i}: the layer ended before the stager's event");
                // the resident GPU experts complete under the hold
                let ev_early = plan.rt2.get().expect("the RT2 tables").ev_early as sys::CUevent;
                let t0 = std::time::Instant::now();
                while sys::cuEventQuery(ev_early) == sys::CUresult::CUDA_ERROR_NOT_READY {
                    assert!(t0.elapsed() < std::time::Duration::from_secs(10), "case {i}: the resident experts did not run under the hold");
                    std::hint::spin_loop();
                }
                let early_ms = t0.elapsed().as_secs_f64() * 1e3;
                assert_eq!(sys::cuEventQuery(fin), sys::CUresult::CUDA_ERROR_NOT_READY, "case {i}: the layer ended before the stager's event");
                std::ptr::write_volatile(word.host as *mut u64, v);
                cuda::sync();
                let y = cuda::dtoh(yd, h);
                assert!(bits(&y) == bits(&y_ref), "case {i} cpu {cpu:08b} late {late:08b}: y differs from experts_lane's");
                let (runs, start_in, end_in) = clock.read_rt2();
                eprintln!("glm5_moe RT2 case {i} cpu {cpu:08b} late {late:08b}: y = experts_lane's bits; resident experts done {early_ms:.3} ms after run returned; lane runs {runs}, started in GPU work {start_in}, ended in GPU work {end_in}");
                overlapped_cases = start_in;
            }
            // cases 0, 1 and 4 have CPU and resident GPU combos: the lane starts while the GPU works
            assert!(overlapped_cases >= 3, "the lane started during the GPU's resident experts in {overlapped_cases} of 3 runs");
            // how long the route and 6 resident experts take with the stager's event free and
            // held (the held side stream sits in a `cuStreamWaitValue64`, as the stager stream
            // does on a landing): a held memop wait must not slow the compute stream
            let mut v = 100u64;
            let mut ms = |hold: bool| -> f64 {
                v += 1;
                if !hold {
                    std::ptr::write_volatile(word.host as *mut u64, v);
                }
                cuda::ck(sys::cuStreamWaitValue64_v2(side, word.dev, v, 0));
                cuda::event_record(held, side);
                cuda::stream_query(side);
                let late = (0..k).map(|c| c == 0 || c == k - 1).collect();
                lane::post(Some(lane::Call { table: tp, combos: combos(0), clock: clock.clone(), ready: None, rt2: Some(lane::Rt2 { event: held as u64, late }) }));
                let t0 = std::time::Instant::now();
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                let ev_early = plan.rt2.get().expect("the RT2 tables").ev_early as sys::CUevent;
                while sys::cuEventQuery(ev_early) == sys::CUresult::CUDA_ERROR_NOT_READY {
                    std::hint::spin_loop();
                }
                let t = t0.elapsed().as_secs_f64() * 1e3;
                std::ptr::write_volatile(word.host as *mut u64, v);
                cuda::sync();
                t
            };
            let mut free: Vec<f64> = (0..9).map(|_| ms(false)).collect();
            let mut hold: Vec<f64> = (0..9).map(|_| ms(true)).collect();
            free.sort_by(f64::total_cmp);
            hold.sort_by(f64::total_cmp);
            eprintln!("glm5_moe RT2: route + 6 resident experts (pinned, zero-copy) to their event, median of 9: stager event free {:.3} ms, held {:.3} ms (test harness)", free[4], hold[4]);
            assert!(hold[4] < 2.0 * free[4] + 0.5, "a held stager wait slowed the resident experts: {:.3} ms vs {:.3} ms", hold[4], free[4]);
            lane::post(None);
            cuda::sync();
            cuda::event_destroy(held);
            cuda::event_destroy(fin);
            cuda::stream_destroy(side);
            word.free();
            plan.free();
            free_ffn(w.shared);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut tp, &mut xd, &mut yd, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// Cross-wiring 2 (`CROW_GLM_MAX_BATCH` with the CPU lane): a lane call of two rows. Row 1 is
    /// x reversed, so the rows route differently; every expert reads its record from pinned. For
    /// several CPU/GPU masks over the 16 combos (both rows mixed, one row all CPU, all CPU): every
    /// GPU combo's `ye` row has the bits of the GPU-only two-row run (its xg row moved to its
    /// slot), every CPU combo's row within 1 - cosine <= 1e-6 of `expert_ffn_mul1_cpu` of its
    /// record on its own row's x (int16 branch: accuracy, not bits), and `y` (both rows) is
    /// `glm5_moe_combine` of that `ye`.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_cpu_lane_rows_of_a_batched_step() {
        use std::sync::Arc;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, g) = (synth(), geo());
            let (h, k, t) = (4096usize, g.topk, 2usize);
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3;
            // one record per expert id modulo 4 (any record serves; the table and the CPU agree)
            let recs: Vec<u8> = (0..4u32).flat_map(record).collect();
            let mut pinned = cuda::Pinned::alloc(recs.len());
            pinned.write_bytes(0, &recs);
            let slot = |e: u32| (e % 4) as usize;
            let table: Vec<u64> = (0..g.experts as u32).map(|e| pinned.dev + (rb * slot(e)) as u64).collect();
            let mut tp = cuda::to_u64_dev(&table);
            let w = GpuMoeWeights { router: cuda::upload_dev(&le_u16(&s.router_w)), bias: cuda::to_f32_dev(&s.bias), shared: gpu_ffn(&s.shared) };
            let mut plan = GpuMoePlan::new(&g, t);
            let x0 = &s.x_moe[..h];
            let x1: Vec<f32> = x0.iter().rev().copied().collect();
            let xs: Vec<f32> = x0.iter().chain(&x1).copied().collect();
            let (mut xd, mut yd, mut y2) = (cuda::to_f32_dev(&xs), cuda::alloc_zeroed(t * h * 4), cuda::alloc_zeroed(t * h * 4));
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            lane::post(None);
            plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
            cuda::sync();
            let ye_gpu = cuda::dtoh(plan.ye, t * k * h);
            let ids: Vec<u32> = cuda::dtoh_i32(plan.ids, t * k).into_iter().map(|v| v as u32).collect();
            assert_ne!(ids[..k], ids[k..], "the two rows must route differently for the test to mean something");
            let host = pinned.host as *const u8;
            let ye_cpu: Vec<Vec<f32>> = ids
                .iter()
                .enumerate()
                .map(|(c, &e)| {
                    let mut y = vec![0f32; h];
                    let x = if c < k { x0 } else { &x1[..] };
                    expert_ffn_mul1_cpu(&g, &recs[rb * slot(e)..rb * (slot(e) + 1)], x, &mut y, 8, Path::Auto).unwrap();
                    y
                })
                .collect();
            let combos = |mask: u32| -> Vec<lane::Combo> {
                ids.iter()
                    .enumerate()
                    .map(|(c, &e)| if mask >> c & 1 == 1 { lane::Combo::Cpu(host.add(rb * slot(e))) } else { lane::Combo::Gpu(table[e as usize]) })
                    .collect()
            };
            let clock = Arc::new(lane::Clock::default());
            for mask in [0x0001u32, 0x8000, 0x5555, 0xA5A5, 0x00FF, 0xFF00, 0x0FF0, 0xFFFF] {
                cuda::to_f32_into(plan.ye, &vec![f32::NAN; t * k * h]);
                lane::post(Some(lane::Call { table: tp, combos: combos(mask), clock: clock.clone(), ready: None, rt2: None }));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::sync();
                let (y, ye) = (cuda::dtoh(yd, t * h), cuda::dtoh(plan.ye, t * k * h));
                for c in 0..t * k {
                    let got = &ye[c * h..(c + 1) * h];
                    if mask >> c & 1 == 1 {
                        let d = 1.0 - cosine(got, &ye_cpu[c]);
                        assert!(d <= 1e-6, "mask {mask:016b} combo {c} (CPU): the ye row is 1 - cos {d:.3e} from expert_ffn_mul1_cpu");
                    } else {
                        assert!(bits(got) == bits(&ye_gpu[c * h..(c + 1) * h]), "mask {mask:016b} combo {c} (GPU): the ye row differs");
                    }
                }
                launch_v(gk.combine, h.div_ceil(256) as u32, t as u32, 1, 256, &[plan.ye, plan.wts, plan.ys, y2, plan.prm_kh2]);
                cuda::sync();
                assert!(bits(&y) == bits(&cuda::dtoh(y2, t * h)), "mask {mask:016b}: y is not the combine of ye");
            }
            eprintln!("glm5_moe CPU lane, two rows: 8 masks, every GPU ye row the GPU's bits, every CPU row within 1e-6 of the CPU's");
            plan.free();
            free_ffn(w.shared);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut tp, &mut xd, &mut yd, &mut y2, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// #188 bench (why the CPU lane refuses a write-combined pinned tier): the CPU lane's FFN
    /// (`experts_ffn`, 1 expert, T 1, `LANE_THREADS`) on one synthetic 3-bit GLM record in a
    /// write-combined pinned block (`CROW_PINNED_ALLOC=wc`, the Windows default), a cacheable
    /// pinned block (`host`) and the heap; median of 15 calls after 3 warm-up calls each, the
    /// three alternating. Prints ms per expert and GB/s of the record.
    #[test]
    #[ignore = "bench, needs the GPU (pinned allocation only): cargo test --release --lib glm5_moe_gpu_lane_wc_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_lane_wc_bench() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let g = geo();
            let r = record(1);
            let mut wc = cuda::Pinned::alloc_wc(r.len());
            let mut host = cuda::Pinned::alloc(r.len());
            wc.write_bytes(0, &r);
            host.write_bytes(0, &r);
            let x: Vec<f32> = { let mut rng = Rng(0x188); (0..4096).map(|_| rng.f(X_AMP)).collect() };
            let srcs: [(&str, *const u8); 3] = [("wc pinned", wc.host as *const u8), ("host pinned", host.host as *const u8), ("heap", r.as_ptr())];
            let mut t: [Vec<f64>; 3] = Default::default();
            let mut out: [Vec<f32>; 3] = Default::default();
            for it in 0..18 {
                for (i, &(_, p)) in srcs.iter().enumerate() {
                    let e = Mul1Expert::from_record(std::slice::from_raw_parts(p, r.len()), 4096, g.expert_inter, g.bitrate).unwrap();
                    let mut y = vec![0f32; 4096];
                    let t0 = std::time::Instant::now();
                    cpu_mul1::experts_ffn(&[e], &x, &mut y, &|a, b| swiglu_clamp(a, b, 10.0), LANE_THREADS, Path::Auto);
                    if it >= 3 {
                        t[i].push(t0.elapsed().as_secs_f64());
                    }
                    out[i] = y;
                }
            }
            for (i, &(name, _)) in srcs.iter().enumerate() {
                t[i].sort_by(|a, b| a.total_cmp(b));
                let m = t[i][t[i].len() / 2];
                eprintln!("glm5_moe lane bench {name}: median {:.3} ms per expert, {:.2} GB/s (min {:.3}, max {:.3} ms, n {})", m * 1e3, r.len() as f64 / m / 1e9, t[i][0] * 1e3, t[i][t[i].len() - 1] * 1e3, t[i].len());
            }
            assert!(out[0] == out[2] && out[1] == out[2], "the three copies give other outputs");
            wc.free();
            host.free();
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

    // ---------------------------------------------------------------- prefill: expert-major (GPU)

    fn router_w(s: &Synth) -> (CUdeviceptr, CUdeviceptr) {
        unsafe { (cuda::upload_dev(&le_u16(&s.router_w)), cuda::to_f32_dev(&s.bias)) }
    }

    /// The expert-major layer (`GpuMoeGroupedPlan`, `mul1_gemm_grp`) against the per-combo layer
    /// (`GpuMoePlan`, one T = 1 MUL1 slot per combo), synthetic GLM layer, 6 distinct 3-bit records
    /// (expert x reads record x % 6). The router of both plans gives the same ids and weights;
    /// then, for a skewed routing written in place of the router's (up to 64 rows on one expert:
    /// items of 16 rows and shorter ones), the grouped layer run in pseudo-row sub-batches of 2,
    /// alternately through a VRAM and a pinned (zero-copy) table, gives every `ye` row and `y`
    /// bit for bit as the per-combo layer; `GpuMoeGroupedPlan::run` on a larger plan (t of
    /// `t + 3` rows) gives `GpuMoePlan::run`'s `y` bit for bit. t = 1, 37, 64.
    #[test]
    #[ignore = "needs the GPU (about 1 GB VRAM): cargo test --release --lib glm5_moe_gpu_grouped -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_grouped_is_the_per_combo_layer_bit_for_bit() {
        const NR: usize = 6;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, g) = (synth(), geo());
            let (h, k, e) = (g.hidden, g.topk, g.experts);
            let recs: Vec<u8> = (0..NR as u32).flat_map(record).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3 as u64;
            let mut vram = cuda::upload_dev(&recs);
            let mut pinned = cuda::Pinned::alloc_cold(recs.len());
            pinned.write_bytes(0, &recs);
            let table = |base: u64| -> Vec<u64> { (0..e).map(|x| base + rb * (x % NR) as u64).collect() };
            let (mut tv, mut tp) = (cuda::to_u64_dev(&table(vram)), cuda::to_u64_dev(&table(pinned.dev)));
            let (router, bias) = router_w(&s);
            let w = GpuMoeWeights { router, bias, shared: gpu_ffn(&s.shared) };
            let bits = |v: &[f32]| v.iter().map(|f| f.to_bits()).collect::<Vec<_>>();
            lane::post(None);
            for (t, skew, seed) in [(1usize, 1.0, 11u64), (37, 1.0, 12), (64, 6.0, 13)] {
                let mut rng = Rng(seed);
                let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
                let (mut xd, mut ya, mut yb) = (cuda::to_f32_dev(&x), cuda::alloc_zeroed(t * h * 4), cuda::alloc_zeroed(t * h * 4));
                let mut a = GpuMoePlan::new(&g, t);
                let before = cuda::live_dev().1;
                let mut b = GpuMoeGroupedPlan::new(&g, t + 3, &mk);
                let got = cuda::live_dev().1 - before;
                assert!(got.abs_diff(GpuMoeGroupedPlan::bytes(&g, t + 3)) <= 4 << 10, "t {t}: the plan allocated {got} B, bytes() says {}", GpuMoeGroupedPlan::bytes(&g, t + 3));
                a.route(&kn, &gk, &w, xd);
                b.route(&kn, &gk, &w, xd, t);
                cuda::sync();
                assert_eq!(cuda::dtoh_i32(a.ids, t * k), cuda::dtoh_i32(b.ids, t * k), "t {t}: router ids");
                assert!(bits(&cuda::dtoh(a.wts, t * k)) == bits(&cuda::dtoh(b.wts, t * k)), "t {t}: router weights");
                // a skewed routing in place of the router's
                let ids = skewed_ids(t, k, e, skew, seed);
                cuda::to_i32_into(a.ids, &ids);
                cuda::to_i32_into(b.ids, &ids);
                a.experts(&kn, &mk, &gk, &w, tv, xd, ya);
                let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
                cuda::sync();
                b.upload(&sch);
                let (mut r0, mut n) = (0usize, 0usize);
                while r0 < sch.pseudo_rows() {
                    let rows = 2.min(sch.pseudo_rows() - r0);
                    b.experts_items(if n % 2 == 0 { tv } else { tp }, xd, sch.work_of_rows(r0, rows));
                    r0 += rows;
                    n += 1;
                }
                b.finish(&gk, &w, xd, yb, t);
                cuda::sync();
                let (yea, yeb) = (cuda::dtoh(a.ye, t * k * h), cuda::dtoh(b.ye, t * k * h));
                for c in 0..t * k {
                    assert!(bits(&yea[c * h..(c + 1) * h]) == bits(&yeb[c * h..(c + 1) * h]), "t {t}: ye of combo {c} (expert {}) differs", ids[c]);
                }
                assert!(bits(&cuda::dtoh(ya, t * h)) == bits(&cuda::dtoh(yb, t * h)), "t {t}: y differs");
                let most = sch.work.iter().fold(std::collections::HashMap::<i32, i32>::new(), |mut m, w| {
                    *m.entry(w[0]).or_default() += w[2];
                    m
                });
                eprintln!(
                    "glm5_moe grouped t {t}: {} experts, {} items, {n} sub-batches, at most {} rows on one expert: ye and y bit-identical to the per-combo layer",
                    sch.experts.len(),
                    sch.work.len(),
                    most.values().max().unwrap()
                );
                // the whole layer, the router's own ids
                a.run(&kn, &mk, &gk, &w, tv, xd, ya);
                b.run(&kn, &gk, &w, tp, xd, yb, t).unwrap();
                cuda::sync();
                assert!(bits(&cuda::dtoh(ya, t * h)) == bits(&cuda::dtoh(yb, t * h)), "t {t}: run differs");
                a.free();
                b.free();
                for d in [&mut xd, &mut ya, &mut yb] {
                    cuda::free_dev(d);
                }
            }
            free_ffn(w.shared);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut vram, &mut tv, &mut tp, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// #186 `CROW_GLM_MOE_TC`: the tensor-core expert path (`MoeTc`: FP16 reconstruction +
    /// `mul1_gemm_tc`) against `mul1_gemm_grp`, synthetic GLM layer, 6 distinct 3-bit records
    /// (expert x reads record x % 6). t = 300 rows, every row picks expert 0 (300 rows: the
    /// tensor cores) and 7 distinct others (a few rows each: `mul1_gemm_grp` in both arms). Every
    /// `ye` row of expert 0 has cosine >= 0.9999 to the default path's, every other row is bit for
    /// bit the default path's; the same through a pinned (zero-copy) table.
    #[test]
    #[ignore = "needs the GPU (about 1 GB VRAM): cargo test --release --lib glm5_moe_gpu_tc -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_tc_experts_match_the_grouped_kernel() {
        const NR: usize = 6;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mk = mul1::Kernels::new();
            let g = geo();
            let (h, k, e) = (g.hidden, g.topk, g.experts);
            let recs: Vec<u8> = (0..NR as u32).flat_map(record).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3 as u64;
            let mut vram = cuda::upload_dev(&recs);
            let mut pinned = cuda::Pinned::alloc_cold(recs.len());
            pinned.write_bytes(0, &recs);
            let table = |base: u64| -> Vec<u64> { (0..e).map(|x| base + rb * (x % NR) as u64).collect() };
            let (mut tv, mut tp) = (cuda::to_u64_dev(&table(vram)), cuda::to_u64_dev(&table(pinned.dev)));
            let t = 300usize;
            let mut rng = Rng(186);
            let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
            let mut ids = Vec::with_capacity(t * k);
            for r in 0..t {
                ids.push(0i32);
                let mut picked = vec![0i32];
                while picked.len() < k {
                    let c = 1 + ((r * 7919 + picked.len() * 104_729 + (rng.f(1.0).to_bits() as usize)) % (e - 1)) as i32;
                    if !picked.contains(&c) {
                        picked.push(c);
                    }
                }
                ids.extend_from_slice(&picked[1..]);
            }
            let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
            let mut xd = cuda::to_f32_dev(&x);
            let mut a = GpuMoeGroupedPlan::new(&g, t, &mk);
            a.set_tc(false, &mk);
            let mut b = GpuMoeGroupedPlan::new(&g, t, &mk);
            b.set_tc(true, &mk);
            for (name, tab) in [("vram", tv), ("pinned", tp)] {
                a.upload(&sch);
                b.upload(&sch);
                a.experts_items(tab, xd, 0..sch.work.len());
                b.experts_items(tab, xd, 0..sch.work.len());
                cuda::sync();
                let (ya, yb) = (cuda::dtoh(a.ye, t * k * h), cuda::dtoh(b.ye, t * k * h));
                let (mut worst, mut n_tc) = (1f64, 0usize);
                let (mut all_a, mut all_b) = (Vec::new(), Vec::new());
                for c in 0..t * k {
                    let (ra, rb) = (&ya[c * h..(c + 1) * h], &yb[c * h..(c + 1) * h]);
                    if ids[c] == 0 {
                        let cs = cosine(ra, rb);
                        worst = worst.min(cs);
                        n_tc += 1;
                        all_a.extend_from_slice(ra);
                        all_b.extend_from_slice(rb);
                    } else {
                        assert!(ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits()), "{name}: ye of combo {c} (expert {}, grouped path) differs", ids[c]);
                    }
                }
                let whole = cosine(&all_a, &all_b);
                eprintln!("glm5_moe tc {name}: expert 0, {n_tc} rows on the tensor cores: worst row cosine {worst:.8}, all rows {whole:.8}; the other experts' rows bit-identical");
                assert_eq!(n_tc, t);
                assert!(worst >= 0.9999, "{name}: worst row cosine {worst} < 0.9999");
            }
            a.free();
            b.free();
            for d in [&mut xd, &mut vram, &mut tv, &mut tp] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// #186 micro-bench: one expert (gate, up, down; synthetic 3-bit record in VRAM) at 2048 rows
    /// (top-1, every row on expert 0): `mul1_gemm_grp` (the default) against the tensor-core path
    /// (`CROW_GLM_MOE_TC`: reconstruction + `mul1_gemm_tc`), median of 9 calls after 2 warm-up
    /// calls, host wall around a stream sync; the cosine of the two outputs.
    #[test]
    #[ignore = "bench, needs the GPU (about 1 GB VRAM): cargo test --release --lib glm5_moe_gpu_tc_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_tc_bench() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mk = mul1::Kernels::new();
            let g = MoeGeo { topk: 1, ..geo() };
            let (h, i, e) = (g.hidden, g.expert_inter, g.experts);
            let rec = record(3);
            let mut one = cuda::upload_dev(&rec);
            let mut tv = cuda::to_u64_dev(&vec![one; e]);
            for t in [144usize, 256, 512, 2048] {
                let mut rng = Rng(t as u64);
                let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
                let mut xd = cuda::to_f32_dev(&x);
                let sch = ExpertMajor::new(&vec![0i32; t], 1, e, GROUP_ROWS).unwrap();
                let mut p = GpuMoeGroupedPlan::new(&g, t, &mk);
                let mut med = |p: &GpuMoeGroupedPlan| -> f64 {
                    p.upload(&sch);
                    let mut v = Vec::new();
                    for n in 0..11 {
                        let t0 = std::time::Instant::now();
                        p.experts_items(tv, xd, 0..sch.work.len());
                        cuda::sync();
                        if n >= 2 {
                            v.push(t0.elapsed().as_secs_f64());
                        }
                    }
                    v.sort_by(|a, b| a.total_cmp(b));
                    v[v.len() / 2]
                };
                p.set_tc(false, &mk);
                let old = med(&p);
                let ya = cuda::dtoh(p.ye, t * h);
                p.set_tc(true, &mk);
                let new = med(&p);
                let yb = cuda::dtoh(p.ye, t * h);
                let cs = cosine(&ya, &yb);
                let flop = 2.0 * t as f64 * 3.0 * (h * i) as f64;
                eprintln!(
                    "glm5_moe tc bench: one expert, {t} rows: mul1_gemm_grp {:.3} ms ({:.1} TFLOPS), tensor cores {:.3} ms ({:.1} TFLOPS), {:.1}x; cosine {cs:.8}",
                    old * 1e3,
                    flop / old / 1e12,
                    new * 1e3,
                    flop / new / 1e12,
                    old / new
                );
                assert!(cs >= 0.9999, "{t} rows: cosine {cs}");
                p.free();
                cuda::free_dev(&mut xd);
            }
            for d in [&mut one, &mut tv] {
                cuda::free_dev(d);
            }
        }
    }

    /// Bench: the routed experts of one MoE layer, per-combo (`GpuMoePlan::experts`: gather, one
    /// T = 1 MUL1 slot per combo for gate / up / down, clamp, shared, combine) against
    /// expert-major (`GpuMoeGroupedPlan::experts_items` + `finish`: two `mul1_gemm_grp`
    /// launches, shared, combine), one synthetic 3-bit record copied to 288 distinct VRAM slots
    /// (no L2 sharing between experts). (a) every row picks the same 8 experts, so each expert
    /// has t rows: expert-rows/s and ms per expert; (b) a uniform routing over 288 experts;
    /// (c) as (b) with 64 records in pinned RAM (zero-copy, expert x reads record x % 64).
    /// Median of 9 calls after 2 warm-up calls, host wall around a stream sync.
    #[test]
    #[ignore = "bench, needs the GPU (about 5 GB VRAM, 0.6 GB pinned): cargo test --release --lib glm5_moe_gpu_grouped_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_grouped_bench() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let (_m, kn) = main_kernels();
            let mk = mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let (s, g) = (synth(), geo());
            let (h, k, e) = (g.hidden, g.topk, g.experts);
            let rec = record(3);
            let rb = rec.len();
            let one = cuda::upload_dev(&rec);
            let mut arena = cuda::alloc_named("bench records", e * rb);
            for x in 0..e {
                cuda::d2d_async(arena + (x * rb) as u64, one, rb);
            }
            cuda::sync();
            let mut one = one;
            cuda::free_dev(&mut one);
            const NP: usize = 64;
            let mut pinned = cuda::Pinned::alloc_cold(NP * rb);
            for x in 0..NP {
                pinned.write_bytes(x * rb, &rec);
            }
            let mut tv = cuda::to_u64_dev(&(0..e).map(|x| arena + (x * rb) as u64).collect::<Vec<_>>());
            let mut tp = cuda::to_u64_dev(&(0..e).map(|x| pinned.dev + ((x % NP) * rb) as u64).collect::<Vec<_>>());
            let (router, bias) = router_w(&s);
            let w = GpuMoeWeights { router, bias, shared: gpu_ffn(&s.shared) };
            lane::post(None);
            {
                use cudarc::driver::sys::{cuFuncGetAttribute, CUfunction_attribute as A};
                let f = mk.module.get(GROUP_ENTRY);
                let at = |a: A| -> i32 {
                    let mut v = 0i32;
                    cuda::ck(cuFuncGetAttribute(&mut v, a, f));
                    v
                };
                eprintln!(
                    "glm5_moe grouped bench: {GROUP_ENTRY} {} registers, {} B local (spills), {} B static shared",
                    at(A::CU_FUNC_ATTRIBUTE_NUM_REGS),
                    at(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES),
                    at(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES)
                );
            }
            let med = |f: &mut dyn FnMut()| -> f64 {
                let mut v = Vec::new();
                for i in 0..11 {
                    let t0 = std::time::Instant::now();
                    f();
                    cuda::sync();
                    if i >= 2 {
                        v.push(t0.elapsed().as_secs_f64());
                    }
                }
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let mut rng = Rng(0x9e57);
            let cases: Vec<(&str, usize, Vec<i32>, bool)> = vec![
                ("same 8 experts", 1, (0..8).collect(), false),
                ("same 8 experts", 4, (0..4).flat_map(|_| 0..8).collect(), false),
                ("same 8 experts", 16, (0..16).flat_map(|_| 0..8).collect(), false),
                ("same 8 experts", 64, (0..64).flat_map(|_| 0..8).collect(), false),
                ("same 8 experts", 228, (0..228).flat_map(|_| 0..8).collect(), false),
                ("uniform over 288", 64, skewed_ids(64, k, e, 1.0, rng.next()), false),
                ("uniform over 288", 256, skewed_ids(256, k, e, 1.0, rng.next()), false),
                ("uniform over 288", 2048, skewed_ids(2048, k, e, 1.0, rng.next()), false),
                ("uniform over 288", 8192, skewed_ids(8192, k, e, 1.0, rng.next()), false),
                ("uniform over 288, pinned", 64, skewed_ids(64, k, e, 1.0, rng.next()), true),
                ("uniform over 288, pinned", 256, skewed_ids(256, k, e, 1.0, rng.next()), true),
                ("uniform over 288, pinned", 2048, skewed_ids(2048, k, e, 1.0, rng.next()), true),
            ];
            for (name, t, ids, pin) in cases {
                let table = if pin { tp } else { tv };
                let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
                let (mut xd, mut yd) = (cuda::to_f32_dev(&x), cuda::alloc_zeroed(t * h * 4));
                let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
                let mut b = GpuMoeGroupedPlan::new(&g, t, &mk);
                cuda::to_i32_into(b.ids, &ids);
                b.upload(&sch);
                let tg = med(&mut || {
                    b.experts_items(table, xd, 0..sch.work.len());
                    b.finish(&gk, &w, xd, yd, t);
                });
                let yg = cuda::dtoh(yd, t * h);
                b.free();
                let combos = (t * k) as f64;
                let per = if t <= 256 {
                    let mut a = GpuMoePlan::new(&g, t);
                    cuda::to_i32_into(a.ids, &ids);
                    let tc = med(&mut || a.experts(&kn, &mk, &gk, &w, table, xd, yd));
                    let same = cuda::dtoh(yd, t * h).iter().zip(&yg).all(|(p, q)| p.to_bits() == q.to_bits());
                    a.free();
                    assert!(same, "{name} t {t}: the two paths differ");
                    format!("per-combo {:.3} ms ({:.0} expert-rows/s), speedup {:.1}x", tc * 1e3, combos / tc, tc / tg)
                } else {
                    "per-combo not run (its plan needs about 5 MB per row)".to_string()
                };
                eprintln!(
                    "glm5_moe grouped bench {name}, t {t}: {} experts, {} items; grouped {:.3} ms ({:.0} expert-rows/s, {:.1} us per expert); {per}",
                    sch.experts.len(),
                    sch.work.len(),
                    tg * 1e3,
                    combos / tg,
                    tg * 1e6 / sch.experts.len() as f64
                );
                for d in [&mut xd, &mut yd] {
                    cuda::free_dev(d);
                }
            }
            free_ffn(w.shared);
            let (mut wr, mut wb) = (w.router, w.bias);
            for d in [&mut arena, &mut tv, &mut tp, &mut wr, &mut wb] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    // ---------------------------------------------------------------- #186 CROW_GLM_MOE_TC=2

    /// `t` rows of `k` distinct picks over `e` experts with log-normal expert popularity
    /// (`sigma`): at 8192 rows top-8 over 288, sigma 0.6 puts about 200 experts at 144 rows or
    /// more, as the profile of one real 8192-token chunk (profile-20261010e: about 196 experts per
    /// layer at >= 144 rows, mean 227, a few above 1024)
    fn realistic_ids(t: usize, k: usize, e: usize, sigma: f64, seed: u64) -> Vec<i32> {
        let mut rng = Rng(seed);
        let mut unit = || ((rng.next() >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
        let mut cdf = Vec::with_capacity(e);
        let mut acc = 0.0;
        for _ in 0..e {
            let n = (-2.0 * unit().ln()).sqrt() * (std::f64::consts::TAU * unit()).cos();
            acc += (sigma * n).exp();
            cdf.push(acc);
        }
        let mut ids = Vec::with_capacity(t * k);
        for _ in 0..t {
            let mut row: Vec<i32> = Vec::with_capacity(k);
            while row.len() < k {
                let u = unit() * acc;
                let x = cdf.partition_point(|&c| c < u).min(e - 1) as i32;
                if !row.contains(&x) {
                    row.push(x);
                }
            }
            ids.extend(row);
        }
        ids
    }

    /// the rows per expert of `ids`: (experts with >= 144 rows, the most rows)
    fn routing_stats(ids: &[i32], e: usize) -> (usize, usize) {
        let mut n = vec![0usize; e];
        for &x in ids {
            n[x as usize] += 1;
        }
        (n.iter().filter(|&&r| r >= 144).count(), *n.iter().max().unwrap())
    }

    /// #186: the split of a schedule for `CROW_GLM_MOE_TC=2`: every combo once, either in the big
    /// experts' packed list (experts of at least the minimum rows, whole) or in the small items'
    /// list, items rebased onto it; row offsets and row-tile prefixes; a range at expert
    /// boundaries splits into the matching small items and big experts, and a range inside a big
    /// expert is refused.
    #[test]
    fn glm5_moe_tc2_schedule_splits_big_and_small_experts() {
        let (t, k, e) = (300usize, 8usize, 288usize);
        let ids = skewed_ids(t, k, e, 3.0, 186);
        let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
        for (min_rows, bm) in [(1usize, 64usize), (40, 64), (40, 128), (100_000, 64)] {
            let s = Tc2Sched::new(&sch.work, &sch.list, min_rows, bm);
            let mut rows = std::collections::BTreeMap::<i32, usize>::new();
            for &x in &ids {
                *rows.entry(x).or_default() += 1;
            }
            let big: Vec<i32> = rows.iter().filter(|(_, &r)| r >= min_rows).map(|(&x, _)| x).collect();
            assert_eq!(s.big_e, big, "min {min_rows}: the big experts");
            let mut all: Vec<i32> = s.list_big.iter().chain(&s.list2).copied().collect();
            all.sort();
            assert_eq!(all, (0..(t * k) as i32).collect::<Vec<_>>(), "min {min_rows}: every combo once");
            for (j, &x) in s.big_e.iter().enumerate() {
                let (a, b) = (s.row_off[j] as usize, s.row_off[j + 1] as usize);
                assert_eq!(b - a, rows[&x]);
                assert!(s.list_big[a..b].iter().all(|&c| ids[c as usize] == x) && s.row_e[a..b].iter().all(|&y| y == x));
                assert_eq!((s.mt_pre[j + 1] - s.mt_pre[j]) as usize, rows[&x].div_ceil(bm));
            }
            let mut f = 0;
            for (w, &r) in s.work2.iter().zip(&s.rows2) {
                assert!(rows[&w[0]] < min_rows && w[1] == f && w[2] as usize == r && r <= GROUP_ROWS);
                assert!(s.list2[f as usize..f as usize + r].iter().all(|&c| ids[c as usize] == w[0]));
                f += r as i32;
            }
            // pseudo-row sub-batches (expert boundaries) cover every small item and big expert once
            let (mut small, mut bigs) = (0..0, 0..0);
            let mut r0 = 0;
            while r0 < sch.pseudo_rows() {
                let (sm, bg) = s.split(sch.work_of_rows(r0, 3.min(sch.pseudo_rows() - r0)));
                assert_eq!((sm.start, bg.start), (small.end, bigs.end));
                (small, bigs) = (small.start..sm.end, bigs.start..bg.end);
                r0 += 3;
            }
            assert_eq!((small, bigs), (0..s.work2.len(), 0..s.big_e.len()));
            if let Some(j) = (0..sch.work.len()).find(|&i| i > 0 && sch.work[i][0] == sch.work[i - 1][0] && rows[&sch.work[i][0]] >= min_rows) {
                assert!(std::panic::catch_unwind(|| s.split(j..sch.work.len())).is_err(), "min {min_rows}: a range from item {j} splits a big expert");
            }
        }
    }

    /// #186 `CROW_GLM_MOE_TC=2`: the grouped tensor-core path against `mul1_gemm_grp`, synthetic
    /// GLM layer (6 distinct 3-bit records, expert x reads record x % 6), 8192 rows, a realistic
    /// routing (`realistic_ids`). Every `ye` row of an expert on the tensor cores has cosine >=
    /// 0.9999 to the default path's, every other row is bit for bit the default path's: (a) 64-row
    /// tiles, 5 slots (many batches), pseudo-row sub-batches through a VRAM table; (b) 128-row
    /// tiles, 16 slots, the whole range through a pinned (zero-copy) table.
    #[test]
    #[ignore = "needs the GPU (about 5 GB VRAM): cargo test --release --lib glm5_moe_gpu_tc2 -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_tc2_experts_match_the_grouped_kernel() {
        const NR: usize = 6;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mk = mul1::Kernels::new();
            let g = geo();
            let (h, k, e) = (g.hidden, g.topk, g.experts);
            let recs: Vec<u8> = (0..NR as u32).flat_map(record).collect();
            let rb = cpu_mul1::GLM_RECORD_BYTES_K3 as u64;
            let mut vram = cuda::upload_dev(&recs);
            let mut pinned = cuda::Pinned::alloc_cold(recs.len());
            pinned.write_bytes(0, &recs);
            let table = |base: u64| -> Vec<u64> { (0..e).map(|x| base + rb * (x % NR) as u64).collect() };
            let (mut tv, mut tp) = (cuda::to_u64_dev(&table(vram)), cuda::to_u64_dev(&table(pinned.dev)));
            let t = 8192usize;
            let ids = realistic_ids(t, k, e, 0.6, 1861);
            let (n144, most) = routing_stats(&ids, e);
            let mut rng = Rng(1862);
            let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
            let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
            let mut xd = cuda::to_f32_dev(&x);
            let mut a = GpuMoeGroupedPlan::new(&g, t, &mk);
            a.set_tc(false, &mk);
            a.upload(&sch);
            a.experts_items(tv, xd, 0..sch.work.len());
            cuda::sync();
            let ya = cuda::dtoh(a.ye, t * k * h);
            a.free();
            let mut b = GpuMoeGroupedPlan::new(&g, t, &mk);
            for (name, tab, bm, slots, min_rows, sub, fused) in [
                ("recon, 64-row tiles, 5 slots, >= 144 rows, sub-batches, vram", tv, 64usize, 5usize, 144usize, true, false),
                ("recon, 128-row tiles, 16 slots, >= 16 rows, pinned", tp, 128, 16, TC2_MIN_ROWS, false, false),
                ("fused decode, 64-row tiles, 5 slots, >= 144 rows, sub-batches, vram", tv, 64, 5, 144, true, true),
                ("fused decode (default), pinned", tp, 128, usize::MAX, TC2_MIN_ROWS, false, true),
            ] {
                b.set_tc2(Tc2Cfg { fused, slots, rows: TC2_ROWS, bm, min_rows }, &mk);
                cuda::ck(cudarc::driver::sys::cuMemsetD8_v2(b.ye, 0, t * k * h * 4));
                b.upload(&sch);
                if sub {
                    let mut r0 = 0;
                    while r0 < sch.pseudo_rows() {
                        let rows = 5.min(sch.pseudo_rows() - r0);
                        b.experts_items(tab, xd, sch.work_of_rows(r0, rows));
                        r0 += rows;
                    }
                } else {
                    b.experts_items(tab, xd, 0..sch.work.len());
                }
                cuda::sync();
                let yb = cuda::dtoh(b.ye, t * k * h);
                let mut n_rows = vec![0usize; e];
                for &id in &ids {
                    n_rows[id as usize] += 1;
                }
                let (mut worst, mut n_tc, mut n_grp) = (1f64, 0usize, 0usize);
                let (mut dot, mut na, mut nb) = (0f64, 0f64, 0f64);
                for c in 0..t * k {
                    let (ra, rb) = (&ya[c * h..(c + 1) * h], &yb[c * h..(c + 1) * h]);
                    if n_rows[ids[c] as usize] >= min_rows {
                        worst = worst.min(cosine(ra, rb));
                        n_tc += 1;
                        for (p, q) in ra.iter().zip(rb) {
                            dot += *p as f64 * *q as f64;
                            na += *p as f64 * *p as f64;
                            nb += *q as f64 * *q as f64;
                        }
                    } else {
                        assert!(ra.iter().zip(rb).all(|(p, q)| p.to_bits() == q.to_bits()), "{name}: ye of combo {c} (expert {}, grouped path) differs", ids[c]);
                        n_grp += 1;
                    }
                }
                let whole = dot / (na.sqrt() * nb.sqrt());
                eprintln!(
                    "glm5_moe tc2 {name}: {t} rows, {n144} experts >= 144 rows, at most {most}; {n_tc} combos on the tensor cores: worst row cosine {worst:.8}, all {whole:.8}; {n_grp} combos mul1_gemm_grp bit-identical"
                );
                assert!(n_tc > 0 && worst >= 0.9999, "{name}: worst row cosine {worst} < 0.9999");
            }
            b.free();
            for d in [&mut xd, &mut vram, &mut tv, &mut tp] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }

    /// #186 micro-bench: the routed experts of one full GLM MoE layer at 8192 rows (realistic
    /// routing, `realistic_ids`; one synthetic 3-bit record copied to 288 distinct VRAM slots):
    /// `mul1_gemm_grp` (TC off), the per-expert tensor-core path (`CROW_GLM_MOE_TC=1`) and the
    /// grouped one (`=2`) over its knobs (minimum rows, row tile, slots). Median of 5 calls after
    /// 1 warm-up, host wall around a stream sync; TFLOPS over every routed combo (2 * 8192 * 8 *
    /// 3 * H * I); the cosine of the grouped default against `mul1_gemm_grp` over every row.
    #[test]
    #[ignore = "bench, needs the GPU (about 8 GB VRAM): cargo test --release --lib glm5_moe_gpu_tc2_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_moe_gpu_tc2_bench() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mk = mul1::Kernels::new();
            let g = geo();
            let (h, i, k, e) = (g.hidden, g.expert_inter, g.topk, g.experts);
            let rec = record(3);
            let rb = rec.len();
            let one = cuda::upload_dev(&rec);
            let mut arena = cuda::alloc_named("bench records", e * rb);
            for x in 0..e {
                cuda::d2d_async(arena + (x * rb) as u64, one, rb);
            }
            cuda::sync();
            let mut one = one;
            cuda::free_dev(&mut one);
            let mut tv = cuda::to_u64_dev(&(0..e).map(|x| arena + (x * rb) as u64).collect::<Vec<_>>());
            let t = 8192usize;
            let ids = realistic_ids(t, k, e, 0.6, 1863);
            let (n144, most) = routing_stats(&ids, e);
            let mut rng = Rng(1864);
            let x: Vec<f32> = (0..t * h).map(|_| rng.f(X_AMP)).collect();
            let mut xd = cuda::to_f32_dev(&x);
            let sch = ExpertMajor::new(&ids, k, e, GROUP_ROWS).unwrap();
            let flop = 2.0 * (t * k) as f64 * 3.0 * (h * i) as f64;
            {
                use cudarc::driver::sys::{cuFuncGetAttribute, CUfunction_attribute as A};
                for name in ["mul1_tc2_gemm64", "mul1_tc2_gemm128", "mul1_tc2_recon"] {
                    let f = mk.module.get(name);
                    let at = |a: A| -> i32 {
                        let mut v = 0i32;
                        cuda::ck(cuFuncGetAttribute(&mut v, a, f));
                        v
                    };
                    eprintln!("glm5_moe tc2 bench: {name} {} registers, {} B local (spills), {} B static shared", at(A::CU_FUNC_ATTRIBUTE_NUM_REGS), at(A::CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES), at(A::CU_FUNC_ATTRIBUTE_SHARED_SIZE_BYTES));
                }
            }
            eprintln!("glm5_moe tc2 bench: {t} rows top-{k} over {e} experts: {} experts routed, {n144} with >= 144 rows, at most {most}; {:.2} TFLOP per layer", sch.experts.len(), flop / 1e12);
            let mut p = GpuMoeGroupedPlan::new(&g, t, &mk);
            let mut med = |p: &GpuMoeGroupedPlan| -> f64 {
                p.upload(&sch);
                let mut v = Vec::new();
                for n in 0..6 {
                    let t0 = std::time::Instant::now();
                    p.experts_items(tv, xd, 0..sch.work.len());
                    cuda::sync();
                    if n >= 1 {
                        v.push(t0.elapsed().as_secs_f64());
                    }
                }
                v.sort_by(|a, b| a.total_cmp(b));
                v[v.len() / 2]
            };
            let say = |what: &str, s: f64| eprintln!("glm5_moe tc2 bench: {what}: {:.2} ms per layer ({:.1} TFLOPS; x42 layers {:.2} s)", s * 1e3, flop / s / 1e12, s * 42.0);
            p.set_tc_mode(0, &mk);
            let t0 = med(&p);
            say("mul1_gemm_grp (TC off)", t0);
            let ya = cuda::dtoh(p.ye, t * k * h);
            p.set_tc_mode(1, &mk);
            let t1 = med(&p);
            say("per expert (TC=1)", t1);
            let dflt = Tc2Cfg::default_for(t);
            let sweep: Vec<Tc2Cfg> = if std::env::var("CROW_GLM_TC2_SWEEP").is_ok() {
                let mut v = Vec::new();
                for fused in [false, true] {
                    for slots in [8, 32] {
                        for bm in [64, 128] {
                            for min_rows in [1, 16, 32, 64, 144] {
                                v.push(Tc2Cfg { fused, slots, rows: t, bm, min_rows });
                            }
                        }
                    }
                }
                for rows in [8192, 16_384, 32_768] {
                    v.push(Tc2Cfg { rows, ..dflt });
                }
                v
            } else {
                vec![Tc2Cfg { fused: false, slots: 16, rows: t, bm: 64, min_rows: 32 }, Tc2Cfg { bm: 64, ..dflt }, Tc2Cfg { rows: t, ..dflt }]
            };
            let label = |c: &Tc2Cfg| {
                format!(
                    "grouped (TC=2) {}, {} experts / {} rows per batch, {}-row tiles, >= {} rows ({} MiB scratch)",
                    if c.fused { "fused decode" } else { "recon" },
                    if c.slots == usize::MAX { "all".to_string() } else { c.slots.to_string() },
                    c.rows_for(t),
                    c.bm,
                    c.min_rows,
                    moe_tc2_bytes(h, i, e, k, t, c) >> 20
                )
            };
            for c in sweep {
                p.set_tc2(c, &mk);
                let s = med(&p);
                say(&label(&c), s);
            }
            p.set_tc_mode(2, &mk);
            let t2 = med(&p);
            say(&format!("default {}", label(&dflt)), t2);
            let yb = cuda::dtoh(p.ye, t * k * h);
            let mut worst = 1f64;
            for c in 0..t * k {
                worst = worst.min(cosine(&ya[c * h..(c + 1) * h], &yb[c * h..(c + 1) * h]));
            }
            let whole = cosine(&ya, &yb);
            eprintln!(
                "glm5_moe tc2 bench: TC=1 {:.2} ms -> TC=2 {:.2} ms per layer ({:.1}x; vs TC off {:.1}x); TC=2 vs mul1_gemm_grp: worst row cosine {worst:.8}, all rows {whole:.8}; scratch {} B",
                t1 * 1e3,
                t2 * 1e3,
                t1 / t2,
                t0 / t2,
                moe_tc2_bytes(h, i, e, k, t, &dflt)
            );
            assert!(worst >= 0.9999, "worst row cosine {worst}");
            p.free();
            for d in [&mut xd, &mut arena, &mut tv] {
                cuda::free_dev(d);
            }
        }
    }
}
