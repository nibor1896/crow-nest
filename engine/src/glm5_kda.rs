//! #162 (GLM-5.3-Flash plan step 13b, KDA part): one KDA (Kimi delta attention) sub-block of
//! glm5_next on the GPU — `a = KDA(h)` of `docs/glm5-next-recipe.md` section 6, from the
//! input-layernormed row `h` [T][hidden] to the `o_proj` output [T][hidden], with the per-sequence
//! state (short-conv window + recurrent state S) carried across prompt chunks and decode steps.
//!
//! Reuse, not a parallel path (docs/glm5-kda.md): the q|k|v projection, causal conv + window,
//! split, l2norm and gated RMSNorm are the GDN kernels of `KERNEL_SRC`, compiled with the KDA
//! geometry ([`kernel_geo`]); the BF16 projections are `gemm_bf16_dense` (prompt) and
//! `gemv_bf16_w` (decode). New are only the three kernels of `kernels::GLM5_KDA_SRC`: the bounded
//! forget gate (`kda_gate`) and the delta rule with a per-KEY-CHANNEL decay (`kda_persist_r`,
//! `kda_step_r`; GDN decays the whole head by one scalar).
//!
//! Not wired into `gen.rs` / `boot.rs`: the layer loop of glm5_next (mHC, MLA/DSA, FFN, head) is
//! integrated by the lead with #161, #163-#165. Nothing in the engine calls this yet.

use crate::cuda;
use crate::geo::{GateAct, Glm5Geo};
use crate::kernels::{self, launch_v, KernelGeo};
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

pub type Dev = CUdeviceptr;

/// the KDA numbers of a glm5_next geometry
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KdaDims {
    pub hidden: usize,
    pub heads: usize,
    pub head_dim: usize,
    /// short conv kernel (4); the window holds `conv - 1` rows
    pub conv: usize,
    /// forget-gate lower bound (`gate_lower_bound`, -5)
    pub lower_bound: f32,
    /// gated RMSNorm eps (`rms_norm_eps`, 1e-5)
    pub eps: f32,
}

impl KdaDims {
    /// panics by name on a shape `GLM5_KDA_SRC` / the reused kernels are not built for
    pub fn of(g: &Glm5Geo) -> KdaDims {
        assert_eq!(g.kda_head_dim, kernels::glm5_kda::HEAD_DIM, "#162: KDA head dim {} != KDA_D", g.kda_head_dim);
        assert_eq!(g.kda_conv, 4, "#162: KDA short conv {} != 4 (conv_silu / conv_step are kernel 4)", g.kda_conv);
        assert!(g.hidden % 16 == 0 && g.kda_heads > 0, "#162: hidden {} / heads {}", g.hidden, g.kda_heads);
        KdaDims {
            hidden: g.hidden,
            heads: g.kda_heads,
            head_dim: g.kda_head_dim,
            conv: g.kda_conv,
            lower_bound: g.kda_lower_bound as f32,
            eps: g.rms_eps as f32,
        }
    }
    /// q, k, v and gate width: heads x head dim (8192)
    pub const fn width(&self) -> usize {
        self.heads * self.head_dim
    }
    /// conv channels: q | k | v (24576)
    pub const fn conv_ch(&self) -> usize {
        3 * self.width()
    }
    /// recurrent state floats per sequence: `[heads][key][value]` (4 MiB at f32)
    pub const fn state_floats(&self) -> usize {
        self.heads * self.head_dim * self.head_dim
    }
    /// conv window floats per sequence: `[conv_ch][conv - 1]` (288 KiB at f32)
    pub const fn conv_floats(&self) -> usize {
        self.conv_ch() * (self.conv - 1)
    }
}

/// The `KernelGeo` the reused GDN kernels compile with for KDA: GDN key heads = value heads =
/// KDA heads (no repeat in `l2norm_repeat`), head dims 128, conv channels q|k|v, eps
/// `rms_norm_eps`, sigmoid gate (HF `Glm5NextTextRMSNormGated`), hidden 4096. Every field
/// glm5_next has no GDN counterpart for (attention, experts, QSA, mHC) keeps its Flash-Next
/// value: compile-only, never launched by this module, until the family's `Geo` arm owns them.
pub fn kernel_geo(g: &Glm5Geo) -> KernelGeo {
    let kd = KdaDims::of(g);
    let mut kg = KernelGeo::flash_next();
    kg.d.h = kd.hidden;
    kg.d.gdn_kheads = kd.heads;
    kg.d.gdn_vheads = kd.heads;
    kg.d.gd = kd.head_dim;
    kg.d.gdv = kd.head_dim;
    kg.d.gdn_key = kd.width();
    kg.d.gdn_val = kd.width();
    kg.d.gdn_conv = kd.conv_ch();
    kg.d.conv_kernel = kd.conv;
    kg.eps = kd.eps;
    kg.gate_act = GateAct::Sigmoid;
    kg.p2 = false;
    kg.q8kv = false;
    kg
}

/// the reused `KERNEL_SRC` entries a KDA layer launches
pub const BASE_NAMES: &[&str] = &[
    "gemm_bf16_dense",
    "gemv_bf16_w",
    "transpose_rt",
    "conv_silu",
    "conv_state_update",
    "conv_step",
    "split_qkv",
    "l2norm_repeat",
    "rmsnorm_gated",
];

/// both modules of a KDA layer
pub struct KdaKernels {
    pub d: KdaDims,
    /// `KERNEL_SRC` at [`kernel_geo`]
    pub base: cuda::Module,
    pub kda: kernels::glm5_kda::Kernels,
    gemm: CUfunction,
    gemv: CUfunction,
    transpose: CUfunction,
    conv_silu: CUfunction,
    conv_state_update: CUfunction,
    conv_step: CUfunction,
    split_qkv: CUfunction,
    l2norm: CUfunction,
    gated_norm: CUfunction,
}

impl KdaKernels {
    /// Compiles `KERNEL_SRC` at [`kernel_geo`] and `GLM5_KDA_SRC`.
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo) -> KdaKernels {
        KdaKernels::with_base(g, cuda::compile(&kernel_geo(g).source()))
    }

    /// #161: the KDA kernels on a `KERNEL_SRC` module the caller compiled at [`kernel_geo`] (the
    /// one shared glm5 compile of `glm5_model::Glm5Kernels`); `base` may be a second handle to
    /// that module, whose owner unloads it. Compiles `GLM5_KDA_SRC`.
    /// # Safety
    /// A CUDA context is current and `base` holds `KERNEL_SRC` at [`kernel_geo`] of `g`.
    pub unsafe fn with_base(g: &Glm5Geo, base: cuda::Module) -> KdaKernels {
        KdaKernels {
            d: KdaDims::of(g),
            gemm: base.get("gemm_bf16_dense"),
            gemv: base.get("gemv_bf16_w"),
            transpose: base.get("transpose_rt"),
            conv_silu: base.get("conv_silu"),
            conv_state_update: base.get("conv_state_update"),
            conv_step: base.get("conv_step"),
            split_qkv: base.get("split_qkv"),
            l2norm: base.get("l2norm_repeat"),
            gated_norm: base.get("rmsnorm_gated"),
            base,
            kda: kernels::glm5_kda::Kernels::new(),
        }
    }
}

/// One KDA layer's tensors on the host, module layout (`docs/glm5-next-recipe.md` section 6).
/// BF16 bit patterns (`u16`) for the projections, f32 for the tensors HF keeps in f32.
pub struct KdaHostWeights {
    /// `self_attn.{q,k,v}_proj.weight`, each `[width][hidden]`
    pub q: Vec<u16>,
    pub k: Vec<u16>,
    pub v: Vec<u16>,
    /// `self_attn.{q,k,v}_conv1d.weight` concatenated q, k, v: `[conv_ch][conv]`
    pub conv: Vec<f32>,
    /// `self_attn.f_a_proj.weight` `[head_dim][hidden]`, `f_b_proj.weight` `[width][head_dim]`
    pub f_a: Vec<u16>,
    pub f_b: Vec<u16>,
    /// `self_attn.dt_bias` `[width]`, `self_attn.A_log` `[heads]`
    pub dt_bias: Vec<f32>,
    pub a_log: Vec<f32>,
    /// `self_attn.b_proj.weight` `[heads][hidden]`
    pub b: Vec<u16>,
    /// `self_attn.g_a_proj.weight` `[head_dim][hidden]`, `g_b_proj.weight` `[width][head_dim]`
    pub g_a: Vec<u16>,
    pub g_b: Vec<u16>,
    /// `self_attn.o_norm.weight` `[head_dim]`
    pub o_norm: Vec<f32>,
    /// `self_attn.o_proj.weight` `[hidden][width]`
    pub o_proj: Vec<u16>,
}

/// one KDA layer's tensors in VRAM; `qkv` is q, k, v stacked to `[conv_ch][hidden]`
pub struct KdaWeights {
    pub qkv: Dev,
    pub conv: Dev,
    pub f_a: Dev,
    pub f_b: Dev,
    pub dt_bias: Dev,
    pub a_log: Dev,
    pub b: Dev,
    pub g_a: Dev,
    pub g_b: Dev,
    pub o_norm: Dev,
    pub o_proj: Dev,
}

impl KdaWeights {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn upload(d: &KdaDims, h: &KdaHostWeights) -> KdaWeights {
        let (hd, w, hid) = (d.head_dim, d.width(), d.hidden);
        let chk = |n: &str, got: usize, want: usize| assert_eq!(got, want, "#162: KDA tensor {n} has {got} values, want {want}");
        chk("q_proj", h.q.len(), w * hid);
        chk("k_proj", h.k.len(), w * hid);
        chk("v_proj", h.v.len(), w * hid);
        chk("conv1d", h.conv.len(), d.conv_ch() * d.conv);
        chk("f_a_proj", h.f_a.len(), hd * hid);
        chk("f_b_proj", h.f_b.len(), w * hd);
        chk("dt_bias", h.dt_bias.len(), w);
        chk("A_log", h.a_log.len(), d.heads);
        chk("b_proj", h.b.len(), d.heads * hid);
        chk("g_a_proj", h.g_a.len(), hd * hid);
        chk("g_b_proj", h.g_b.len(), w * hd);
        chk("o_norm", h.o_norm.len(), hd);
        chk("o_proj", h.o_proj.len(), hid * w);
        let qkv = cuda::alloc_named("kda qkv", 3 * w * hid * 2);
        for (i, part) in [&h.q, &h.k, &h.v].into_iter().enumerate() {
            cuda::into_dev(qkv + (i * w * hid * 2) as u64, part.as_slice());
        }
        KdaWeights {
            qkv,
            conv: cuda::to_dev(h.conv.as_slice()),
            f_a: cuda::to_dev(h.f_a.as_slice()),
            f_b: cuda::to_dev(h.f_b.as_slice()),
            dt_bias: cuda::to_dev(h.dt_bias.as_slice()),
            a_log: cuda::to_dev(h.a_log.as_slice()),
            b: cuda::to_dev(h.b.as_slice()),
            g_a: cuda::to_dev(h.g_a.as_slice()),
            g_b: cuda::to_dev(h.g_b.as_slice()),
            o_norm: cuda::to_dev(h.o_norm.as_slice()),
            o_proj: cuda::to_dev(h.o_proj.as_slice()),
        }
    }

    /// # Safety
    /// A CUDA context is current; no launch still reads these buffers.
    pub unsafe fn free(&mut self) {
        for p in [
            &mut self.qkv, &mut self.conv, &mut self.f_a, &mut self.f_b, &mut self.dt_bias, &mut self.a_log,
            &mut self.b, &mut self.g_a, &mut self.g_b, &mut self.o_norm, &mut self.o_proj,
        ] {
            cuda::free_dev(p);
        }
    }
}

/// The per-layer, per-sequence KDA state: the recurrent state `s` `[heads][key][value]` f32 and the
/// conv window `conv` `[conv_ch][conv - 1]` f32 (the last pre-conv q|k|v rows, oldest first).
/// Zero = the start of a sequence (HF: the conv's left zero padding and `initial_state=None`).
pub struct KdaState {
    pub s: Dev,
    pub conv: Dev,
    d: KdaDims,
}

impl KdaState {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn alloc(d: &KdaDims) -> KdaState {
        KdaState { s: cuda::alloc_named("kda state", d.state_floats() * 4), conv: cuda::alloc_named("kda conv", d.conv_floats() * 4), d: *d }
    }
    /// back to the start of a sequence
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn reset(&self) {
        cuda::sync();
        cuda::ck(cudarc::driver::sys::cuMemsetD8_v2(self.s, 0, self.d.state_floats() * 4));
        cuda::ck(cudarc::driver::sys::cuMemsetD8_v2(self.conv, 0, self.d.conv_floats() * 4));
    }
    /// # Safety
    /// A CUDA context is current; no launch still reads these buffers.
    pub unsafe fn free(&mut self) {
        cuda::free_dev(&mut self.s);
        cuda::free_dev(&mut self.conv);
    }
}

/// #192 (MTP rollback): a KDA state as a snapshot slot of another
impl KdaState {
    /// device bytes of one state: S plus the conv window
    pub fn bytes(&self) -> u64 {
        ((self.d.state_floats() + self.d.conv_floats()) * 4) as u64
    }

    /// queue `self = src` (S and conv window, D2D on the current stream)
    ///
    /// # Safety
    /// A CUDA context is current; both states have the same dims.
    pub unsafe fn copy_from(&self, src: &KdaState) {
        assert_eq!(self.d, src.d, "glm5_kda: a snapshot of other dims");
        cuda::d2d_async(self.s, src.s, self.d.state_floats() * 4);
        cuda::d2d_async(self.conv, src.conv, self.d.conv_floats() * 4);
    }
}

// the scalar argument slots of `KdaScratch::params` (every scalar is a device pointer, the p5 rule)
const P_T: usize = 0;
const P_HIDDEN: usize = 1;
const P_WIDTH: usize = 2;
const P_CONV: usize = 3;
const P_HD: usize = 4;
const P_HEADS: usize = 5;
const P_ZERO: usize = 6;
const P_ONE: usize = 7;

/// the activations of up to `max_t` rows per prompt call, carved out of one region (#196: the
/// region may be shared with other scratch that is never live at the same time, `glm5_model`'s
/// attention region)
pub struct KdaScratch {
    pub max_t: usize,
    /// #196: the region this scratch owns (0: a view into a region its owner frees)
    region: Dev,
    params: Dev,
    lb: Dev,
    qkv: Dev,
    qkv_t: Dev,
    conv_t: Dev,
    q: Dev,
    k: Dev,
    v: Dev,
    qn: Dev,
    kn: Dev,
    fa: Dev,
    f: Dev,
    b: Dev,
    g: Dev,
    beta: Dev,
    o: Dev,
    ga: Dev,
    gate: Dev,
    normed: Dev,
}

impl KdaScratch {
    /// f32 values per row of the activations, in field order: qkv, qkv_t, conv_t, q, k, v, qn,
    /// kn, fa, f, b, g, beta, o, ga, gate, normed
    fn widths(d: &KdaDims) -> [usize; 17] {
        let (cc, w, hd, h) = (d.conv_ch(), d.width(), d.head_dim, d.heads);
        [cc, cc, cc, w, w, w, w, w, hd, w, h, w, h, w, hd, w, w]
    }

    /// #196: the bytes of the region of a scratch of `max_t` rows (`glm5_moe::carve`)
    pub fn region_bytes(d: &KdaDims, max_t: usize) -> usize {
        crate::glm5_moe::carve(&KdaScratch::widths(d).map(|n| max_t * n * 4)).1
    }

    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn alloc(d: &KdaDims, max_t: usize) -> KdaScratch {
        assert!(max_t >= 1);
        let region = cuda::alloc_named("kda scratch", KdaScratch::region_bytes(d, max_t));
        let mut s = KdaScratch::alloc_in(d, max_t, region);
        s.region = region;
        s
    }

    /// #196: a scratch of `max_t` rows whose activations are views into `region`
    /// ([`KdaScratch::region_bytes`] long), which the caller allocates and frees;
    /// [`KdaScratch::free`] frees only the parameter arrays
    ///
    /// # Safety
    /// A CUDA context is current; `region` outlives every launch on this scratch.
    pub unsafe fn alloc_in(d: &KdaDims, max_t: usize, region: Dev) -> KdaScratch {
        assert!(max_t >= 1 && region != 0);
        let (off, _) = crate::glm5_moe::carve(&KdaScratch::widths(d).map(|n| max_t * n * 4));
        let f = |n: usize| region + off[n] as u64;
        let i = |v: usize| i32::try_from(v).expect("#162: KDA dim beyond i32");
        let params = cuda::to_i32_dev(&[0, i(d.hidden), i(d.width()), i(d.conv_ch()), i(d.head_dim), i(d.heads), 0, 1]);
        KdaScratch {
            max_t,
            region: 0,
            params,
            lb: cuda::to_f32_dev(&[d.lower_bound]),
            qkv: f(0),
            qkv_t: f(1),
            conv_t: f(2),
            q: f(3),
            k: f(4),
            v: f(5),
            qn: f(6),
            kn: f(7),
            fa: f(8),
            f: f(9),
            b: f(10),
            g: f(11),
            beta: f(12),
            o: f(13),
            ga: f(14),
            gate: f(15),
            normed: f(16),
        }
    }
    fn p(&self, slot: usize) -> u64 {
        self.params + (slot * 4) as u64
    }
    /// # Safety
    /// A CUDA context is current; no launch still reads these buffers.
    pub unsafe fn free(&mut self) {
        for p in [&mut self.params, &mut self.lb, &mut self.region] {
            cuda::free_dev(p);
        }
        for p in [
            &mut self.qkv, &mut self.qkv_t, &mut self.conv_t, &mut self.q, &mut self.k, &mut self.v, &mut self.qn, &mut self.kn,
            &mut self.fa, &mut self.f, &mut self.b, &mut self.g, &mut self.beta, &mut self.o, &mut self.ga, &mut self.gate, &mut self.normed,
        ] {
            *p = 0;
        }
    }
}

/// `y[t][rows] = W[rows][k] x[t][k]`, BF16 W, prompt form (`gemm_bf16_dense`, hi+lo activation split)
unsafe fn gemm(kk: &KdaKernels, sc: &KdaScratch, w: Dev, x: Dev, y: Dev, k_slot: usize, rows: usize, rows_slot: usize, t: usize) {
    launch_v(kk.gemm, rows.div_ceil(64) as u32, t.div_ceil(8) as u32, 1, 128, &[w, x, y, sc.p(k_slot), sc.p(rows_slot), sc.p(P_T)]);
}

/// one row of `gemm` (`gemv_bf16_w`, warp per output row)
unsafe fn gemv(kk: &KdaKernels, sc: &KdaScratch, w: Dev, x: Dev, y: Dev, k_slot: usize, rows: usize, rows_slot: usize) {
    launch_v(kk.gemv, rows.div_ceil(8) as u32, 1, 1, 256, &[w, x, y, sc.p(k_slot), sc.p(rows_slot)]);
}

/// A prompt call: rows `x` `[t][hidden]` f32 (the input-layernormed `h`) through the KDA sub-block
/// into `out` `[t][hidden]` f32, continuing `st` (zero after `KdaState::reset` = a new sequence).
/// Any split of a prompt into calls gives the one-call result up to f32 summation order; the
/// recurrence runs token by token (no chunked WY form). Queued on the current stream.
/// # Safety
/// A CUDA context is current; `x` / `out` hold `t` rows; `t <= sc.max_t`.
pub unsafe fn prompt(kk: &KdaKernels, w: &KdaWeights, st: &KdaState, sc: &KdaScratch, x: Dev, t: usize, out: Dev) {
    let d = &kk.d;
    assert!((1..=sc.max_t).contains(&t), "#162: KDA prompt call of {t} rows (scratch holds {})", sc.max_t);
    cuda::to_i32_into(sc.params, &[t as i32]);
    let (cc, hd, heads) = (d.conv_ch(), d.head_dim as u32, d.heads as u32);
    // q | k | v rows, then the causal conv over time on [C][T] with the window of the previous call
    gemm(kk, sc, w.qkv, x, sc.qkv, P_HIDDEN, cc, P_CONV, t);
    launch_v(kk.transpose, cc as u32, 1, 1, 256, &[sc.qkv, sc.qkv_t, sc.p(P_T), sc.p(P_CONV)]);
    launch_v(kk.conv_silu, cc as u32, 1, 1, 256, &[sc.qkv_t, w.conv, sc.conv_t, sc.p(P_T), st.conv]);
    launch_v(kk.conv_state_update, cc as u32, 1, 1, (d.conv - 1) as u32, &[sc.qkv_t, st.conv, sc.p(P_T)]);
    launch_v(kk.split_qkv, (t * cc).div_ceil(256) as u32, 1, 1, 256, &[sc.conv_t, sc.q, sc.k, sc.v, sc.p(P_T)]);
    launch_v(kk.l2norm, heads, t as u32, 1, hd, &[sc.q, sc.k, sc.qn, sc.kn]);
    // forget gate (per key channel) and beta (per head)
    gemm(kk, sc, w.f_a, x, sc.fa, P_HIDDEN, d.head_dim, P_HD, t);
    gemm(kk, sc, w.f_b, sc.fa, sc.f, P_HD, d.width(), P_WIDTH, t);
    gemm(kk, sc, w.b, x, sc.b, P_HIDDEN, d.heads, P_HEADS, t);
    launch_v(kk.kda.gate, (t * d.width()).div_ceil(256) as u32, 1, 1, 256, &[
        sc.f, w.dt_bias, w.a_log, sc.b, sc.g, sc.beta, sc.p(P_T), sc.p(P_HEADS), sc.lb]);
    launch_v(kk.kda.persist, heads, 1, 1, hd, &[sc.qn, sc.kn, sc.v, sc.g, sc.beta, sc.o, st.s, sc.p(P_T), sc.p(P_ZERO)]);
    // output gate, gated RMSNorm per head, o_proj
    gemm(kk, sc, w.g_a, x, sc.ga, P_HIDDEN, d.head_dim, P_HD, t);
    gemm(kk, sc, w.g_b, sc.ga, sc.gate, P_HD, d.width(), P_WIDTH, t);
    launch_v(kk.gated_norm, heads, t as u32, 1, hd, &[sc.o, sc.gate, w.o_norm, sc.normed]);
    gemm(kk, sc, w.o_proj, sc.normed, out, P_WIDTH, d.hidden, P_HIDDEN, t);
}

/// A decode step: one row `x` `[hidden]` into `out` `[hidden]`, continuing `st`
/// (HF `causal_conv1d_update` + `recurrent_kimi_delta_attention`). Queued on the current stream.
/// # Safety
/// A CUDA context is current.
pub unsafe fn step(kk: &KdaKernels, w: &KdaWeights, st: &KdaState, sc: &KdaScratch, x: Dev, out: Dev) {
    let d = &kk.d;
    let (cc, wb, hd, heads) = (d.conv_ch(), (d.width() * 4) as u64, d.head_dim as u32, d.heads as u32);
    gemv(kk, sc, w.qkv, x, sc.qkv, P_HIDDEN, cc, P_CONV);
    launch_v(kk.conv_step, cc.div_ceil(256) as u32, 1, 1, 256, &[sc.qkv, w.conv, st.conv, sc.conv_t]);
    // conv_t is q | k | v of the one row, each [heads][head_dim]
    launch_v(kk.l2norm, heads, 1, 1, hd, &[sc.conv_t, sc.conv_t + wb, sc.qn, sc.kn]);
    gemv(kk, sc, w.f_a, x, sc.fa, P_HIDDEN, d.head_dim, P_HD);
    gemv(kk, sc, w.f_b, sc.fa, sc.f, P_HD, d.width(), P_WIDTH);
    gemv(kk, sc, w.b, x, sc.b, P_HIDDEN, d.heads, P_HEADS);
    launch_v(kk.kda.gate, d.width().div_ceil(256) as u32, 1, 1, 256, &[
        sc.f, w.dt_bias, w.a_log, sc.b, sc.g, sc.beta, sc.p(P_ONE), sc.p(P_HEADS), sc.lb]);
    launch_v(kk.kda.step, heads, 1, 1, hd, &[st.s, sc.qn, sc.kn, sc.conv_t + 2 * wb, sc.g, sc.beta, sc.o]);
    gemv(kk, sc, w.g_a, x, sc.ga, P_HIDDEN, d.head_dim, P_HD);
    gemv(kk, sc, w.g_b, sc.ga, sc.gate, P_HD, d.width(), P_WIDTH);
    launch_v(kk.gated_norm, heads, 1, 1, hd, &[sc.o, sc.gate, w.o_norm, sc.normed]);
    gemv(kk, sc, w.o_proj, sc.normed, out, P_WIDTH, d.hidden, P_HIDDEN);
}

/// #161: the two projections of a KDA layer a caller may run in another weight codec (the
/// container stores them NVFP4, `converter/src/recipe.rs`); the gate projections stay BF16.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KdaProj {
    /// `x [t][hidden]` -> `y [t][conv_ch]`, q | k | v per row
    Qkv,
    /// `x [t][width]` (the gated-normed `o`) -> `y [t][hidden]`
    O,
}

/// [`prompt`] with the q|k|v and o_proj projections queued by `proj(which, x, y, t)` instead of
/// the BF16 `gemm` (`w.qkv` and `w.o_proj` are not read); every other stage is `prompt`'s,
/// launch for launch. `prompt` itself is unchanged (its G3 evidence of #162).
/// # Safety
/// As [`prompt`]; `proj` queues on the current stream and writes `y` in the layout above.
#[allow(clippy::too_many_arguments)]
pub unsafe fn prompt_with(kk: &KdaKernels, w: &KdaWeights, st: &KdaState, sc: &KdaScratch, x: Dev, t: usize, out: Dev, proj: &mut dyn FnMut(KdaProj, Dev, Dev, usize)) {
    let d = &kk.d;
    assert!((1..=sc.max_t).contains(&t), "#162: KDA prompt call of {t} rows (scratch holds {})", sc.max_t);
    cuda::to_i32_into(sc.params, &[t as i32]);
    let (cc, hd, heads) = (d.conv_ch(), d.head_dim as u32, d.heads as u32);
    proj(KdaProj::Qkv, x, sc.qkv, t);
    launch_v(kk.transpose, cc as u32, 1, 1, 256, &[sc.qkv, sc.qkv_t, sc.p(P_T), sc.p(P_CONV)]);
    launch_v(kk.conv_silu, cc as u32, 1, 1, 256, &[sc.qkv_t, w.conv, sc.conv_t, sc.p(P_T), st.conv]);
    launch_v(kk.conv_state_update, cc as u32, 1, 1, (d.conv - 1) as u32, &[sc.qkv_t, st.conv, sc.p(P_T)]);
    launch_v(kk.split_qkv, (t * cc).div_ceil(256) as u32, 1, 1, 256, &[sc.conv_t, sc.q, sc.k, sc.v, sc.p(P_T)]);
    launch_v(kk.l2norm, heads, t as u32, 1, hd, &[sc.q, sc.k, sc.qn, sc.kn]);
    gemm(kk, sc, w.f_a, x, sc.fa, P_HIDDEN, d.head_dim, P_HD, t);
    gemm(kk, sc, w.f_b, sc.fa, sc.f, P_HD, d.width(), P_WIDTH, t);
    gemm(kk, sc, w.b, x, sc.b, P_HIDDEN, d.heads, P_HEADS, t);
    launch_v(kk.kda.gate, (t * d.width()).div_ceil(256) as u32, 1, 1, 256, &[
        sc.f, w.dt_bias, w.a_log, sc.b, sc.g, sc.beta, sc.p(P_T), sc.p(P_HEADS), sc.lb]);
    launch_v(kk.kda.persist, heads, 1, 1, hd, &[sc.qn, sc.kn, sc.v, sc.g, sc.beta, sc.o, st.s, sc.p(P_T), sc.p(P_ZERO)]);
    gemm(kk, sc, w.g_a, x, sc.ga, P_HIDDEN, d.head_dim, P_HD, t);
    gemm(kk, sc, w.g_b, sc.ga, sc.gate, P_HD, d.width(), P_WIDTH, t);
    launch_v(kk.gated_norm, heads, t as u32, 1, hd, &[sc.o, sc.gate, w.o_norm, sc.normed]);
    proj(KdaProj::O, sc.normed, out, t);
}

/// #196: [`prompt_with`] for a call of any `t` rows on a scratch of fewer: the rows in sub-blocks
/// of `sc.max_t` in position order, the state (S and conv window) carried from one to the next
/// (`kda_persist_r` stores S after a block and loads it at the next, the conv window likewise),
/// so the sequential scan sees every row once in order; every other stage computes a row from
/// that row alone.
/// # Safety
/// As [`prompt_with`], without the `t <= sc.max_t` bound.
#[allow(clippy::too_many_arguments)]
pub unsafe fn prompt_rows_with(kk: &KdaKernels, w: &KdaWeights, st: &KdaState, sc: &KdaScratch, x: Dev, t: usize, out: Dev, proj: &mut dyn FnMut(KdaProj, Dev, Dev, usize)) {
    let rb = (kk.d.hidden * 4) as u64;
    let mut r = 0;
    while r < t {
        let n = sc.max_t.min(t - r);
        prompt_with(kk, w, st, sc, x + r as u64 * rb, n, out + r as u64 * rb, proj);
        r += n;
    }
}

/// [`step`] with the two projections queued by `proj(which, x, y, 1)`, as [`prompt_with`].
/// # Safety
/// As [`step`]; `proj` as in [`prompt_with`].
pub unsafe fn step_with(kk: &KdaKernels, w: &KdaWeights, st: &KdaState, sc: &KdaScratch, x: Dev, out: Dev, proj: &mut dyn FnMut(KdaProj, Dev, Dev, usize)) {
    let d = &kk.d;
    let (cc, wb, hd, heads) = (d.conv_ch(), (d.width() * 4) as u64, d.head_dim as u32, d.heads as u32);
    proj(KdaProj::Qkv, x, sc.qkv, 1);
    launch_v(kk.conv_step, cc.div_ceil(256) as u32, 1, 1, 256, &[sc.qkv, w.conv, st.conv, sc.conv_t]);
    launch_v(kk.l2norm, heads, 1, 1, hd, &[sc.conv_t, sc.conv_t + wb, sc.qn, sc.kn]);
    gemv(kk, sc, w.f_a, x, sc.fa, P_HIDDEN, d.head_dim, P_HD);
    gemv(kk, sc, w.f_b, sc.fa, sc.f, P_HD, d.width(), P_WIDTH);
    gemv(kk, sc, w.b, x, sc.b, P_HIDDEN, d.heads, P_HEADS);
    launch_v(kk.kda.gate, d.width().div_ceil(256) as u32, 1, 1, 256, &[
        sc.f, w.dt_bias, w.a_log, sc.b, sc.g, sc.beta, sc.p(P_ONE), sc.p(P_HEADS), sc.lb]);
    launch_v(kk.kda.step, heads, 1, 1, hd, &[st.s, sc.qn, sc.kn, sc.conv_t + 2 * wb, sc.g, sc.beta, sc.o]);
    gemv(kk, sc, w.g_a, x, sc.ga, P_HIDDEN, d.head_dim, P_HD);
    gemv(kk, sc, w.g_b, sc.ga, sc.gate, P_HD, d.width(), P_WIDTH);
    launch_v(kk.gated_norm, heads, 1, 1, hd, &[sc.o, sc.gate, w.o_norm, sc.normed]);
    proj(KdaProj::O, sc.normed, out, 1);
}

/// The synthetic KDA layer of the #162 goldens: the counter-based generator of
/// `oracle/export_glm5_kda_golden.py`, bit for bit. Every value is `q * 2^-p` (or the `Dt` /
/// `Norm` rule) with an integer `q` in [-128, 127]: exact in BF16 and f32 on both sides.
pub mod synth {
    use super::{KdaDims, KdaHostWeights};

    pub const SEED: u64 = 0x6C6D_354B_4441;
    const GOLD: u64 = 0x9E37_79B9_7F4A_7C15;

    /// the value rule of one tensor
    #[derive(Clone, Copy, Debug)]
    pub enum Rule {
        /// `q * 2^-p`
        P(i32),
        /// `q * 2^-5 - 2` (dt_bias, in [-6, 2))
        Dt,
        /// `(64 + (q >> 2)) * 2^-6` (o_norm, in [0.5, 1.5))
        Norm,
    }

    /// (id, module name, shape, rule) of the exporter's `TENSORS`; the input rows are id 0, `P(6)`
    pub const TENSORS: &[(u64, &str, &[usize], Rule)] = &[
        (1, "q_proj.weight", &[8192, 4096], Rule::P(13)),
        (2, "k_proj.weight", &[8192, 4096], Rule::P(13)),
        (3, "v_proj.weight", &[8192, 4096], Rule::P(13)),
        (4, "conv1d.weight", &[24576, 1, 4], Rule::P(8)),
        (5, "forget_gate.f_a_proj.weight", &[128, 4096], Rule::P(13)),
        (6, "forget_gate.f_b_proj.weight", &[8192, 128], Rule::P(9)),
        (7, "forget_gate.dt_bias", &[8192], Rule::Dt),
        (8, "forget_gate.A_log", &[64], Rule::P(7)),
        (9, "b_proj.weight", &[64, 4096], Rule::P(13)),
        (10, "g_a_proj.weight", &[128, 4096], Rule::P(13)),
        (11, "g_b_proj.weight", &[8192, 128], Rule::P(9)),
        (12, "o_norm.weight", &[128], Rule::Norm),
        (13, "o_proj.weight", &[4096, 8192], Rule::P(14)),
    ];
    pub const X_ID: u64 = 0;
    pub const X_RULE: Rule = Rule::P(6);

    /// the top byte of splitmix64(seed_t + (i + 1) * GOLD) minus 128, seed_t = SEED + id * 2^32
    pub fn int(id: u64, i: usize) -> i32 {
        let seed = SEED.wrapping_add(id << 32);
        let mut z = seed.wrapping_add((i as u64 + 1).wrapping_mul(GOLD));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 56) as i32 - 128
    }

    pub fn values(id: u64, n: usize, rule: Rule) -> Vec<f32> {
        (0..n)
            .map(|i| {
                let q = int(id, i);
                match rule {
                    Rule::P(p) => q as f32 * (2f32).powi(-p),
                    Rule::Dt => q as f32 * (2f32).powi(-5) - 2.0,
                    Rule::Norm => (64 + (q >> 2)) as f32 * (2f32).powi(-6),
                }
            })
            .collect()
    }

    /// the BF16 bit pattern of BF16-exact values (panics on any other)
    pub fn bf16(v: &[f32]) -> Vec<u16> {
        v.iter()
            .map(|x| {
                let b = x.to_bits();
                assert_eq!(b & 0xFFFF, 0, "#162: {x} is not exact in BF16");
                (b >> 16) as u16
            })
            .collect()
    }

    /// every tensor of the table at its module name, f32
    pub fn tensors() -> Vec<(&'static str, Vec<f32>)> {
        TENSORS.iter().map(|&(id, name, shape, rule)| (name, values(id, shape.iter().product(), rule))).collect()
    }

    /// the synthetic layer at the real GLM-5.3-Flash KDA shapes
    pub fn weights(d: &KdaDims) -> KdaHostWeights {
        assert_eq!((d.hidden, d.heads, d.head_dim, d.conv), (4096, 64, 128, 4), "#162: the synthetic layer is real-shape only");
        let mut t = tensors().into_iter().map(|(_, v)| v);
        let mut nx = || t.next().unwrap();
        let (q, k, v, conv, f_a, f_b, dt_bias, a_log, b, g_a, g_b, o_norm, o_proj) =
            (nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx(), nx());
        KdaHostWeights {
            q: bf16(&q),
            k: bf16(&k),
            v: bf16(&v),
            conv,
            f_a: bf16(&f_a),
            f_b: bf16(&f_b),
            dt_bias,
            a_log,
            b: bf16(&b),
            g_a: bf16(&g_a),
            g_b: bf16(&g_b),
            o_norm,
            o_proj: bf16(&o_proj),
        }
    }

    /// input rows `[n][hidden]`
    pub fn input(n: usize, hidden: usize) -> Vec<f32> {
        values(X_ID, n * hidden, X_RULE)
    }
}

/// f64 references of the two new kernels, for the tiny op tests and any harness
pub mod reference {
    /// `kda_gate`: (g `[t][heads*hd]`, beta `[t][heads]`)
    pub fn gate(f: &[f32], dt_bias: &[f32], a_log: &[f32], b: &[f32], heads: usize, hd: usize, lb: f64) -> (Vec<f64>, Vec<f64>) {
        let sig = |x: f64| 1.0 / (1.0 + (-x).exp());
        let w = heads * hd;
        let g = f.iter().enumerate().map(|(i, &fv)| {
            let c = i % w;
            lb * sig((a_log[c / hd] as f64).exp() * (fv as f64 + dt_bias[c] as f64))
        });
        (g.collect(), b.iter().map(|&x| sig(x as f64)).collect())
    }

    /// the KDA recurrence over `t` tokens from state `s` `[heads][hd][hd]` (updated in place);
    /// q, k, v, g `[t][heads][hd]`, beta `[t][heads]`; returns o `[t][heads][hd]`
    #[allow(clippy::too_many_arguments)]
    pub fn recurrence(q: &[f32], k: &[f32], v: &[f32], g: &[f32], beta: &[f32], s: &mut [f64], t: usize, heads: usize, hd: usize) -> Vec<f64> {
        let mut o = vec![0f64; t * heads * hd];
        for tt in 0..t {
            for h in 0..heads {
                let r = (tt * heads + h) * hd;
                let sh = &mut s[h * hd * hd..(h + 1) * hd * hd];
                for dk in 0..hd {
                    let e = (g[r + dk] as f64).exp();
                    for dv in 0..hd {
                        sh[dk * hd + dv] *= e;
                    }
                }
                let bt = beta[tt * heads + h] as f64;
                for dv in 0..hd {
                    let kv: f64 = (0..hd).map(|dk| sh[dk * hd + dv] * k[r + dk] as f64).sum();
                    let delta = (v[r + dv] as f64 - kv) * bt;
                    for dk in 0..hd {
                        sh[dk * hd + dv] += k[r + dk] as f64 * delta;
                    }
                }
                for dv in 0..hd {
                    o[r + dv] = (0..hd).map(|dk| sh[dk * hd + dv] * q[r + dk] as f64).sum();
                }
            }
        }
        o
    }
}

#[cfg(test)]
mod tests {
    //! Host-only checks (no GPU): the dims are the family row, the generator is the exporter's
    //! bit for bit, the reused KERNEL_SRC entries compile at the KDA geometry.
    use super::*;
    use sha2::{Digest, Sha256};

    pub(super) const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/kda");

    pub(super) fn manifest() -> serde_json::Value {
        serde_json::from_str(&std::fs::read_to_string(format!("{FIX}/manifest.json")).unwrap()).unwrap()
    }

    fn sha(v: &[f32]) -> String {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn kda_dims_are_the_family_row() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let d = KdaDims::of(&g);
        assert_eq!((d.width(), d.conv_ch(), d.lower_bound, d.eps), (8192, 24576, -5.0, 1e-5));
        assert_eq!((d.state_floats() * 4) as u64, g.kda_state_bytes());
        assert_eq!((d.conv_floats() * 4) as u64, g.kda_conv_bytes());
        let kg = kernel_geo(&g);
        let defs: std::collections::HashMap<_, _> = kg.defines().into_iter().collect();
        for (k, v) in [
            ("CN_GDN_KHEADS", "64"), ("CN_GDN_VHEADS", "64"), ("CN_GD", "128"), ("CN_GDV", "128"),
            ("CN_GDN_KEY", "8192"), ("CN_GDN_VAL", "8192"), ("CN_GDN_CONV", "24576"), ("CN_EPS", "1e-5f"), ("CN_GATE_ACT", "0"),
        ] {
            assert_eq!(defs[k], v, "{k}");
        }
    }

    #[test]
    fn kda_synth_generator_is_the_exporters_bit_for_bit() {
        let man = manifest();
        let want = &man["generated_sha256"];
        let (t, h) = (man["N"].as_u64().unwrap() as usize, man["kda"]["hidden"].as_u64().unwrap() as usize);
        assert_eq!(sha(&synth::input(t, h)), want["x"].as_str().unwrap(), "input rows");
        for (name, v) in synth::tensors() {
            assert_eq!(sha(&v), want[name].as_str().unwrap(), "{name}");
            synth::bf16(&v); // every value BF16-exact
        }
        assert_eq!(man["seed"].as_u64().unwrap(), synth::SEED);
    }

    #[test]
    fn kda_reused_kernels_compile_at_the_kda_geometry() {
        let ptx = crate::kernels::tests_300_c4::ptx(&kernel_geo(&Glm5Geo::GLM_5_3_FLASH).source());
        let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|(n, _)| n).collect();
        for n in BASE_NAMES {
            assert!(names.iter().any(|m| m == n), "{n} missing at the KDA geometry");
        }
    }
}

#[cfg(test)]
mod tests_gpu {
    //! #162 acceptance on the GPU (RTX 5090, sm_120). `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_kda::tests_gpu -- --ignored --nocapture --test-threads 1`.
    //! G3 (runs/glm53-flash/PREREG.md, per layer): cosine >= 0.9999 of the sub-block output against
    //! the HF golden on the same (synthetic, real-shape) weights, max |deviation| reported; prompt
    //! rows and decode rows are separate anchors.
    use super::tests::{manifest, FIX};
    use super::*;

    const G3: f64 = 0.9999;

    fn read_f32(name: &str) -> Vec<f32> {
        std::fs::read(format!("{FIX}/{name}")).unwrap().chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()
    }

    fn cos(a: &[f32], b: &[f32]) -> f64 {
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

    fn cos64(a: &[f32], b: &[f64]) -> f64 {
        let b32: Vec<f32> = b.iter().map(|&x| x as f32).collect();
        cos(a, &b32)
    }

    struct Layer {
        kk: KdaKernels,
        w: KdaWeights,
        st: KdaState,
        sc: KdaScratch,
    }

    unsafe fn layer(max_t: usize) -> Layer {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let kk = KdaKernels::new(&g);
        let w = KdaWeights::upload(&kk.d, &synth::weights(&kk.d));
        let st = KdaState::alloc(&kk.d);
        st.reset();
        let sc = KdaScratch::alloc(&kk.d, max_t);
        Layer { kk, w, st, sc }
    }

    /// the prompt in calls of `split` rows, then one step per decode row; returns (out [N][H],
    /// conv window and state S after the prompt, S after the last row)
    unsafe fn run(l: &Layer, x: &[f32], t: usize, n: usize, split: &[usize]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let h = l.kk.d.hidden;
        l.st.reset();
        let mut xd = cuda::to_f32_dev(x);
        let mut od = cuda::alloc_zeroed(n * h * 4);
        let mut r0 = 0;
        for &c in split {
            prompt(&l.kk, &l.w, &l.st, &l.sc, xd + (r0 * h * 4) as u64, c, od + (r0 * h * 4) as u64);
            r0 += c;
        }
        assert_eq!(r0, t);
        cuda::sync();
        let conv = cuda::dtoh(l.st.conv, l.kk.d.conv_floats());
        let s_p = cuda::dtoh(l.st.s, l.kk.d.state_floats());
        for r in t..n {
            step(&l.kk, &l.w, &l.st, &l.sc, xd + (r * h * 4) as u64, od + (r * h * 4) as u64);
        }
        cuda::sync();
        let out = cuda::dtoh(od, n * h);
        let s_f = cuda::dtoh(l.st.s, l.kk.d.state_floats());
        cuda::free_dev(&mut xd);
        cuda::free_dev(&mut od);
        (out, conv, s_p, s_f)
    }

    /// heads `hs` of a full state `[heads][hd][hd]`
    fn heads_of(s: &[f32], hs: &[usize], hd: usize) -> Vec<f32> {
        hs.iter().flat_map(|&h| s[h * hd * hd..(h + 1) * hd * hd].iter().copied()).collect()
    }

    /// the G3 verdict of one engine run against the golden; panics below the gate
    fn judge(tag: &str, got: &(Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>)) {
        let man = manifest();
        let (t, n, h) = (man["T"].as_u64().unwrap() as usize, man["N"].as_u64().unwrap() as usize, 4096);
        let hs: Vec<usize> = man["state_heads"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let gold = read_f32("out.f32");
        let (out, conv, s_p, s_f) = got;
        for (anchor, a, b) in [("prompt", 0, t), ("decode", t, n)] {
            let c = cos(&out[a * h..b * h], &gold[a * h..b * h]);
            let rows: Vec<f64> = (a..b).map(|r| cos(&out[r * h..(r + 1) * h], &gold[r * h..(r + 1) * h])).collect();
            let worst = rows.iter().cloned().fold(1.0, f64::min);
            let m = max_abs(&out[a * h..b * h], &gold[a * h..b * h]);
            eprintln!("{tag} {anchor} rows {a}..{b}: cosine {c:.9} (worst row {worst:.9}), max_abs {m:.3e}");
            if anchor == "decode" {
                let drift: Vec<String> = rows.iter().map(|c| format!("{:.2e}", 1.0 - c)).collect();
                eprintln!("{tag} decode 1-cos by step: {}", drift.join(" "));
            }
            assert!(c >= G3 && worst >= G3, "{tag} {anchor}: cosine {c} / worst row {worst} < {G3}");
        }
        let hd = 128;
        let gconv = read_f32("conv-prompt.f32");
        let (cc, cm) = (cos(conv, &gconv), max_abs(conv, &gconv));
        for (name, s, gname, nkey) in [("prompt", s_p, "state-prompt.f32", "state_norm_prompt"), ("final", s_f, "state-final.f32", "state_norm_final")] {
            let sub = heads_of(s, &hs, hd);
            let gs = read_f32(gname);
            let (c, m) = (cos(&sub, &gs), max_abs(&sub, &gs));
            let norms = man[nkey].as_array().unwrap();
            let worst_norm = (0..64)
                .map(|hh| {
                    let n: f64 = s[hh * hd * hd..(hh + 1) * hd * hd].iter().map(|&x| x as f64 * x as f64).sum::<f64>().sqrt();
                    (n / norms[hh].as_f64().unwrap() - 1.0).abs()
                })
                .fold(0.0, f64::max);
            eprintln!("{tag} state S after {name}: heads {hs:?} cosine {c:.9}, max_abs {m:.3e}; |norm/golden - 1| <= {worst_norm:.2e} over 64 heads");
            assert!(c >= G3 && worst_norm < 1e-3, "{tag} state {name}: cosine {c}, norm deviation {worst_norm}");
        }
        eprintln!("{tag} conv window after the prompt: cosine {cc:.9}, max_abs {cm:.3e}");
        assert!(cc >= G3, "{tag} conv window: cosine {cc}");
    }

    /// G3, prompt in one call (96 rows: one 64-row HF chunk + a partial one) and 8 decode steps
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_kda::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn kda_gpu_layer_matches_the_hf_golden() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let man = manifest();
            let (t, n) = (man["T"].as_u64().unwrap() as usize, man["N"].as_u64().unwrap() as usize);
            let mut l = layer(t);
            let x = synth::input(n, 4096);
            let got = run(&l, &x, t, n, &[t]);
            judge("one call", &got);
            l.w.free();
            l.st.free();
            l.sc.free();
        }
    }

    /// G3 with the prompt in calls of 40, 2 and 54 rows (a call shorter than the conv window, the
    /// recurrence and the window carried across calls), then the 8 decode steps; and the split
    /// against the one-call run of the engine itself
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_kda::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn kda_gpu_chunked_prompt_matches_the_hf_golden() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let man = manifest();
            let (t, n) = (man["T"].as_u64().unwrap() as usize, man["N"].as_u64().unwrap() as usize);
            let mut l = layer(t);
            let x = synth::input(n, 4096);
            let split = run(&l, &x, t, n, &[40, 2, 54]);
            judge("calls 40+2+54", &split);
            let one = run(&l, &x, t, n, &[t]);
            let m = max_abs(&split.0, &one.0);
            let ms = max_abs(&split.2, &one.2);
            eprintln!("calls 40+2+54 vs one call (engine): out max_abs {m:.3e}, state after prompt max_abs {ms:.3e}");
            assert!(m < 1e-4 && ms < 1e-4, "the split changes the result: out {m:.3e}, state {ms:.3e}");
            l.w.free();
            l.st.free();
            l.sc.free();
        }
    }

    /// The two new kernels against their f64 references at the real head shape (64 x 128): the gate,
    /// the per-channel decay over 6 tokens from a non-zero state, and kda_persist_r over 6 tokens
    /// bit-identical to 6 kda_step_r calls.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_kda::tests_gpu -- --ignored --nocapture --test-threads 1"]
    fn kda_gpu_gate_and_recurrence_match_the_f64_reference() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = kernels::glm5_kda::Kernels::new();
            let (heads, hd, t) = (64usize, 128usize, 6usize);
            let w = heads * hd;
            let val = |id: u64, n: usize, p: i32| synth::values(1000 + id, n, synth::Rule::P(p));
            // gate inputs: f, dt_bias, A_log, b
            let (f, dt, al, b) = (val(1, t * w, 5), synth::values(1002, w, synth::Rule::Dt), val(3, heads, 7), val(4, t * heads, 5));
            let dev = |v: &[f32]| cuda::to_f32_dev(v);
            let (fd, dtd, ald, bd) = (dev(&f), dev(&dt), dev(&al), dev(&b));
            let (gd, betad) = (cuda::alloc_zeroed(t * w * 4), cuda::alloc_zeroed(t * heads * 4));
            let par = cuda::to_i32_dev(&[t as i32, heads as i32, 0, 1]);
            let lb = cuda::to_f32_dev(&[-5.0f32]);
            kernels::launch_sync(kn.gate, (t * w).div_ceil(256) as u32, 1, 1, 256, &[fd, dtd, ald, bd, gd, betad, par, par + 4, lb]);
            let (g, beta) = (cuda::dtoh(gd, t * w), cuda::dtoh(betad, t * heads));
            let (gr, br) = reference::gate(&f, &dt, &al, &b, heads, hd, -5.0);
            let eg = g.iter().zip(&gr).map(|(&a, &r)| (a as f64 - r).abs()).fold(0.0, f64::max);
            let eb = beta.iter().zip(&br).map(|(&a, &r)| (a as f64 - r).abs()).fold(0.0, f64::max);
            let spread = g.chunks(hd).map(|c| c.iter().cloned().fold(f32::MIN, f32::max) - c.iter().cloned().fold(f32::MAX, f32::min)).fold(0f32, f32::max);
            eprintln!("kda_gate: max |g - ref| {eg:.3e}, max |beta - ref| {eb:.3e}; g spread within a head up to {spread:.3}");
            assert!(eg < 1e-5 && eb < 1e-6, "kda_gate off its reference: g {eg:.3e}, beta {eb:.3e}");
            assert!(spread > 1.0, "the test needs channels with different decay");
            // recurrence inputs: unit-norm-ish q, k, v, the gate's g / beta, a non-zero start state
            let l2 = |mut v: Vec<f32>, scale: f32| {
                for c in v.chunks_mut(hd) {
                    let n = c.iter().map(|x| x * x).sum::<f32>().sqrt();
                    c.iter_mut().for_each(|x| *x *= scale / n);
                }
                v
            };
            let q = l2(val(5, t * w, 6), 1.0 / (hd as f32).sqrt());
            let k = l2(val(6, t * w, 6), 1.0);
            let v = val(7, t * w, 6);
            let s0 = val(8, heads * hd * hd, 9);
            let (qd, kd, vd) = (dev(&q), dev(&k), dev(&v));
            let (sp, ss) = (dev(&s0), dev(&s0));
            let (op, os) = (cuda::alloc_zeroed(t * w * 4), cuda::alloc_zeroed(t * w * 4));
            kernels::launch_sync(kn.persist, heads as u32, 1, 1, hd as u32, &[qd, kd, vd, gd, betad, op, sp, par, par + 8]);
            for i in 0..t {
                let (o4, b4) = ((i * w * 4) as u64, (i * heads * 4) as u64);
                kernels::launch_sync(kn.step, heads as u32, 1, 1, hd as u32, &[ss, qd + o4, kd + o4, vd + o4, gd + o4, betad + b4, os + o4]);
            }
            let (o_p, o_s) = (cuda::dtoh(op, t * w), cuda::dtoh(os, t * w));
            let (s_p, s_s) = (cuda::dtoh(sp, heads * hd * hd), cuda::dtoh(ss, heads * hd * hd));
            let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
            assert_eq!(bits(&o_p), bits(&o_s), "kda_persist_r and kda_step_r outputs differ");
            assert_eq!(bits(&s_p), bits(&s_s), "kda_persist_r and kda_step_r states differ");
            let mut sr: Vec<f64> = s0.iter().map(|&x| x as f64).collect();
            let or = reference::recurrence(&q, &k, &v, &g, &beta, &mut sr, t, heads, hd);
            let (co, cs) = (cos64(&o_p, &or), cos64(&s_p, &sr));
            let eo = o_p.iter().zip(&or).map(|(&a, &r)| (a as f64 - r).abs()).fold(0.0, f64::max);
            eprintln!("kda_persist_r over {t} tokens: o cosine {co:.12}, max_abs {eo:.3e}; state cosine {cs:.12}");
            assert!(co > 0.999_999 && cs > 0.999_999, "the recurrence is off its f64 reference: o {co}, state {cs}");
            for p in [fd, dtd, ald, bd, gd, betad, par, lb, qd, kd, vd, sp, ss, op, os].iter_mut() {
                cuda::free_dev(p);
            }
        }
    }
}
