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
        assert!((1..=self.tokens).contains(&t), "glm5_moe: FFN rows {t} of a {}-row plan", self.tokens);
        let (h, i) = (self.hidden, self.inter);
        assert!((w.gate.rows, w.gate.cols, w.up.rows, w.up.cols, w.down.rows, w.down.cols) == (i, h, i, h, h, i), "glm5_moe: FFN weights do not fit the plan");
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
    /// #188 CPU lane: host buffer and event, made on the first lane call
    lane: std::cell::OnceCell<LaneBuf>,
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
            lane: std::cell::OnceCell::new(),
        }
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
        if let Some(call) = lane::take(table, self.tokens, self.geo.topk) {
            return self.experts_lane(kn, mk, gk, w, table, x, y, call);
        }
        let (h, t) = (self.geo.hidden, self.tokens);
        let c = t * self.geo.topk;
        launch_v(gk.gather, h.div_ceil(256) as u32, c as u32, 1, 256, &[self.ids, table, x, self.ptrs, self.xg, self.prm_kh2]);
        self.gate.run(mk, self.ptrs, self.xg, self.ge);
        self.up.run(mk, self.ptrs, self.xg, self.ue);
        launch_v(gk.act, (c * self.geo.expert_inter).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
        self.down.run(mk, self.ptrs, self.he, self.ye);
        self.shared.run(kn, gk, &w.shared, x, self.ys);
        launch_v(gk.combine, h.div_ceil(256) as u32, t as u32, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
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
    ) {
        use cudarc::driver::sys;
        let g = &self.geo;
        let (h, k) = (g.hidden, g.topk);
        let rb = g.record.bytes as usize;
        let buf = self.lane.get_or_init(|| LaneBuf::new(h, k));
        let host = buf.host.host as *mut u8;
        let (xh, ph, yh) = (host as *mut f32, host.add(h * 4) as *mut u64, host.add(h * 4 + k * 8) as *mut f32);
        let s = cuda::cur_stream();
        let ev = buf.ev as sys::CUevent;
        cuda::ck(sys::cuMemcpyDtoHAsync_v2(xh as *mut _, x, h * 4, s));
        cuda::event_record(ev, s);
        launch_v(gk.gather, h.div_ceil(256) as u32, k as u32, 1, 256, &[self.ids, table, x, self.ptrs, self.xg, self.prm_kh2]);
        let mut gpu = Vec::with_capacity(k);
        let mut cpu = Vec::with_capacity(k);
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
            // the gather wrote xg for every combo: at t = 1 every row is x, so slot j reads x too
            cuda::upload_from_pinned(self.ptrs, ph as *const _, n * 8);
            self.gate.run_slots(mk, n, self.ptrs, self.xg, self.ge);
            self.up.run_slots(mk, n, self.ptrs, self.xg, self.ue);
            launch_v(gk.act, (k * g.expert_inter).div_ceil(256) as u32, 1, 1, 256, &[self.ge, self.ue, self.he, self.prm_n, self.prm_f]);
            self.down.run_slots(mk, n, self.ptrs, self.he, self.ye);
        }
        self.shared.run(kn, gk, &w.shared, x, self.ys);
        // hand the queued launches to the GPU (WDDM batches them), then wait for x only
        let _ = sys::cuStreamQuery(s);
        cuda::ck(sys::cuEventSynchronize(ev));
        let t0 = std::time::Instant::now();
        let es: Vec<Mul1Expert> = cpu
            .iter()
            .map(|&(c, rec)| {
                Mul1Expert::from_record(std::slice::from_raw_parts(rec, rb), h, g.expert_inter, g.bitrate).unwrap_or_else(|e| panic!("glm5_moe CPU lane: combo {c}: {e}"))
            })
            .collect();
        let xs = std::slice::from_raw_parts(xh as *const f32, h);
        let ys = std::slice::from_raw_parts_mut(yh, cpu.len() * h);
        let limit = g.swiglu_limit;
        cpu_mul1::experts_ffn(&es, xs, ys, &move |a, b| swiglu_clamp(a, b, limit), LANE_THREADS, Path::Auto);
        call.clock.add(t0.elapsed(), cpu.len());
        for (j, &c) in gpu.iter().enumerate().rev() {
            if c != j {
                cuda::d2d_async(self.ye + (c * h * 4) as u64, self.ye + (j * h * 4) as u64, h * 4);
            }
        }
        for (i, &(c, _)) in cpu.iter().enumerate() {
            cuda::upload_from_pinned(self.ye + (c * h * 4) as u64, yh.add(i * h) as *const _, h * 4);
        }
        launch_v(gk.combine, h.div_ceil(256) as u32, 1, 1, 256, &[self.ye, self.wts, self.ys, y, self.prm_kh2]);
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
        if let Some(mut b) = self.lane.take() {
            b.host.free();
            cuda::event_destroy(b.ev as cudarc::driver::sys::CUevent);
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

/// One MoE layer of a prompt call of up to `tokens` rows, expert-major ([`ExpertMajor`]): the
/// router as [`GpuMoePlan::route`], then per tier sub-batch two `mul1_gemm_grp` launches (gate
/// and up in one, down with the clamp fused into its input), then the shared expert and
/// `glm5_moe_combine` once. Every routed output row `ye[c]` has the bits of [`GpuMoePlan`]'s
/// T = 1 slot of combo `c`, so `y` has the bits of [`GpuMoePlan::run`]. Scratch: `ge`, `ue`
/// `[c][inter]` and `ye` `[c][hidden]` for `c = tokens * topk` combos, no per-slot GEMV plans
/// (about 0.35 MB per row at GLM-5.3-Flash shapes against GpuMoePlan's about 5 MB).
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
    list: CUdeviceptr,
    work: CUdeviceptr,
    work_cap: usize,
    /// `mul1_gemm_grp` (`CUfunction` as an integer, as `LaneBuf::ev`)
    grp: u64,
}

impl GpuMoeGroupedPlan {
    /// the most work items a call of `tokens` rows can have: every expert's last item may be short
    pub fn work_cap(geo: &MoeGeo, tokens: usize) -> usize {
        let c = tokens * geo.topk;
        c.div_ceil(GROUP_ROWS) + geo.experts.min(c)
    }

    /// the device bytes [`GpuMoeGroupedPlan::new`] allocates, without its parameter arrays
    pub fn bytes(geo: &MoeGeo, tokens: usize) -> u64 {
        let (t, c) = (tokens as u64, (tokens * geo.topk) as u64);
        let (h, e, i, s) = (geo.hidden as u64, geo.experts as u64, geo.expert_inter as u64, geo.shared_inter as u64);
        // logits, ids, wts, ge / ue, ye, ys, list, work, the shared expert's g / u / h
        4 * (t * e + 2 * c + 2 * c * i + c * h + t * h + c + 3 * Self::work_cap(geo, tokens) as u64 + 3 * t * s)
    }

    /// # Safety
    /// A CUDA context is current; `mk` is the module the launches run on.
    pub unsafe fn new(geo: &MoeGeo, tokens: usize, mk: &mul1::Kernels) -> GpuMoeGroupedPlan {
        let (h, e, k, i) = (geo.hidden, geo.experts, geo.topk, geo.expert_inter);
        let c = tokens * k;
        assert!(tokens > 0 && e <= kernels::glm5_moe::ROUTER_THREADS && (1..=kernels::glm5_moe::MAXK).contains(&k) && k <= e);
        let [sg, su, sd] = geo.record_specs();
        let (s_gu, s_d) = (mul1::ksplit(h), mul1::ksplit(i));
        assert!(h / s_gu <= GROUP_XROWS && i / s_d <= GROUP_XROWS && h % 128 == 0 && i % 128 == 0, "glm5_moe: grouped GEMM shapes H {h} I {i}");
        let work_cap = Self::work_cap(geo, tokens);
        assert!(work_cap <= 65_535, "glm5_moe: {work_cap} work items exceed the grid");
        let prm = |v: [usize; 15]| v.map(|x| i32::try_from(x).expect("glm5_moe: parameter beyond i32"));
        // [k, n, S, n32, bits, half, in_div, mode, limit bits, (tr, suh, svh) x 2]
        let gu = prm([h, i, s_gu, sg.n32(), sg.bits as usize, sg.half as usize, k, 0, 0, sg.tr_off, sg.suh_off, sg.svh_off, su.tr_off, su.suh_off, su.svh_off]);
        let mut d = prm([i, h, s_d, sd.n32(), sd.bits as usize, sd.half as usize, 1, 1, 0, sd.tr_off, sd.suh_off, sd.svh_off, 0, 0, 0]);
        d[8] = geo.swiglu_limit.to_bits() as i32;
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
            ge: cuda::alloc_zeroed(c * i * 4),
            ue: cuda::alloc_zeroed(c * i * 4),
            ye: cuda::alloc_zeroed(c * h * 4),
            ys: cuda::alloc_zeroed(tokens * h * 4),
            shared: GpuFfnPlan::new(h, geo.shared_inter, tokens, geo.swiglu_limit),
            list: cuda::alloc_zeroed(c * 4),
            work: cuda::alloc_zeroed(work_cap * 3 * 4),
            work_cap,
            grp: mk.module.get(GROUP_ENTRY) as u64,
        }
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
    }

    /// queue the routed experts of work items `items` of the uploaded schedule through `table`:
    /// `ye[c]` of their combos from rows `c / topk` of `x` (two launches)
    ///
    /// # Safety
    /// `table` points every expert of these items at a readable record until the launches
    /// finished; `x` holds the call's rows.
    pub unsafe fn experts_items(&self, table: CUdeviceptr, x: CUdeviceptr, items: std::ops::Range<usize>) {
        if items.is_empty() {
            return;
        }
        assert!(items.end <= self.work_cap);
        let (h, i) = (self.geo.hidden, self.geo.expert_inter);
        let (f, n) = (self.grp as cudarc::driver::sys::CUfunction, items.len() as u32);
        let work = self.work + (items.start * 12) as u64;
        launch_v(f, (i / 128) as u32, n, 2, 256, &[table, work, self.list, x, x, self.ge, self.ue, self.prm_gu]);
        launch_v(f, (h / 128) as u32, n, 1, 256, &[table, work, self.list, self.ge, self.ue, self.ye, self.ye, self.prm_d]);
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
            &mut self.ge,
            &mut self.ue,
            &mut self.ye,
            &mut self.ys,
            &mut self.list,
            &mut self.work,
        ] {
            cuda::free_dev(d);
        }
        self.shared.free();
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
    unsafe fn new(h: usize, k: usize) -> LaneBuf {
        LaneBuf { host: cuda::Pinned::alloc(h * 4 + k * 8 + k * h * 4), ev: cuda::event_create() as u64 }
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
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::sync::Arc;

    /// where one combo (pick order) is computed
    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    pub enum Combo {
        /// on the GPU, from this record base (VRAM, pinned UVA or staging)
        Gpu(u64),
        /// on the CPU, from this host record (a pinned slot; `record.bytes` long)
        Cpu(*const u8),
    }

    /// the CPU lane's wall time and expert count, summed (shared with the counters' owner)
    #[derive(Debug, Default)]
    pub struct Clock {
        ns: AtomicU64,
        experts: AtomicU64,
        runs: AtomicU64,
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
    }

    /// one decode call's split
    #[derive(Debug)]
    pub struct Call {
        /// the device table of the call (`experts` takes the post only for this table)
        pub table: CUdeviceptr,
        /// one per combo, pick order
        pub combos: Vec<Combo>,
        pub clock: Arc<Clock>,
    }

    thread_local! {
        static POSTED: RefCell<Option<Call>> = const { RefCell::new(None) };
    }

    /// post (or, with `None`, clear) this thread's next call
    pub fn post(call: Option<Call>) {
        POSTED.with(|p| *p.borrow_mut() = call);
    }

    /// the posted call, if it is for `table`, a decode call (`tokens` 1) of `topk` combos with at
    /// least one on the CPU; the post is consumed either way
    pub(crate) fn take(table: CUdeviceptr, tokens: usize, topk: usize) -> Option<Call> {
        let c = POSTED.with(|p| p.borrow_mut().take())?;
        (c.table == table && tokens == 1 && c.combos.len() == topk && c.combos.iter().any(|x| matches!(x, Combo::Cpu(_)))).then_some(c)
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
    /// row has the bits of the GPU-only run, every CPU combo's row the bits of
    /// `expert_ffn_mul1_cpu` of its record (`ye` filled with NaN before each run, so a row the
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
                lane::post(Some(lane::Call { table: tp, combos: combos(mask), clock: clock.clone() }));
                plan.run(&kn, &mk, &gk, &w, tp, xd, yd);
                cuda::sync();
                want_experts += mask.count_ones() as u64;
                let (y, ye) = (cuda::dtoh(yd, h), cuda::dtoh(plan.ye, k * h));
                for c in 0..k {
                    let (got, want) = (&ye[c * h..(c + 1) * h], if mask >> c & 1 == 1 { &ye_cpu[c][..] } else { &ye_gpu[c * h..(c + 1) * h] });
                    assert!(bits(got) == bits(want), "mask {mask:08b} combo {c} ({}): the ye row differs", if mask >> c & 1 == 1 { "CPU" } else { "GPU" });
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
            for post in [lane::Call { table: tp + 8, combos: combos(0xFF), clock: clock.clone() }, lane::Call { table: tp, combos: combos(0), clock: clock.clone() }] {
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
}
