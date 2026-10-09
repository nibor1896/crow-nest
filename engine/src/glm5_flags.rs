//! #149 path B / #189: the glm5_next decode switches of `Glm5Run` (glm5_next only; nothing on the
//! Flash-Next / 27B path constructs or calls anything here, gate R). All default OFF; off, the
//! path is the one before them, call for call.
//!
//! - **`CROW_GLM_FLAGS=1`** ([`Routed`]): a MoE layer's router ids reach the host through mapped
//!   pinned memory. A one-block kernel queued right behind the router copies the selected ids
//!   into mapped host memory, `__threadfence_system`, then raises a 64-bit sequence flag there
//!   (the device-side publication of `docs/architecture.md` 3.2 and #7: a kernel-side mapped write,
//!   no memop on the legacy stream). The host spins on the flag instead of `cuStreamSynchronize`
//!   plus a blocking pageable `cuMemcpyDtoH`. What the host still waits for: the flag itself, i.e.
//!   every launch of the stream up to and including the router (the same GPU work the stream sync
//!   waited for), and inside `ExpertTiers::table_for` its own syncs (the phase-A barrier, the NVMe
//!   reads, the synchronous landing and table uploads), unless the stager below is on.
//! - **`CROW_GLM_STAGER=1`** (needs `CROW_GLM_FLAGS=1`, `glm5_tiers::stager_on`): the landed half
//!   of path B (`glm5_tiers::Stager`). `table_for` queues the moves and the table upload on a
//!   stager stream from persistent pinned sources, the NVMe readers raise a per-expert landed flag
//!   the stager stream waits on (`cuStreamWaitValue64_v2`), and the compute stream waits on the
//!   stager's event. The host's one wait per MoE layer is then this module's router flag.
//! - **`CROW_GLM_LOOKAHEAD=1`** ([`Feed`], [`Readback`]): `Glm5Run::generate` queues decode row
//!   k+1 before the host reads token k. The head's greedy id stays on the GPU; [`Feed::gather`]
//!   writes that id's embedding row (BF16 widened exactly, `cnq::bf16_bytes_to_f32`) into all four
//!   residual streams (`glm5_model::trunk_input`); the id and the logits go to pinned host memory by
//!   an async copy, read by the host at row k+1's first MoE layer (after its stream sync or flag).
//!   The embedding table sits in VRAM (154,880 x 4096 BF16 = 1,268,776,960 B for GLM-5.3-Flash);
//!   it is loaded only when the switch is on. `Glm5Run::row` (serve) does not look ahead: serve
//!   needs token k before it decides on step k+1 (end of turn).
//! - **PUBFAST** (the reference's `GLM53_NV_PUBFAST`): not built. The reference's warp-ballot publish replaces two serial
//!   thread-0 scans over the experts inside its publish kernel (dedup + request positions,
//!   sybil-solutions/glm-flash-lite `kernels/nv2/nv2_dev.cu` `nv_pub_k` / `nv_pub_fast_k`). Here
//!   the publish kernel copies the raw top-k ids (8 values) and the host dedups them in
//!   `glm5_tiers::distinct_ids` (sort of 8); there is no scan over the experts to replace.
//!   Still not built with the prefetch below: its guess is published the same way (8 raw ids +
//!   the guessed layer) and the host filters the 8 by residency (`ExpertCache::tier`), whose
//!   state lives on the host; a device ballot would need a device copy of that state per call.
//! - **`CROW_GLM_PREFETCH=1`** (needs `CROW_GLM_FLAGS=1`; [`Predict`], [`Prefetch`]): the
//!   reference's layer-ahead prefetch (`GLM53_NV_PREFETCH`, sybil-solutions/glm53-flash-offload
//!   `6769b27` `glm53/nv2.py` `_predict` / `layer`, `kernels/nv2/nv2_host.cpp` `plan_and_reply`). In
//!   a decode call (one row) of MoE layer l, layer l+1's router (`gemv_bf16_b` + the fused
//!   `glm5_router_sig_topk`: sigmoid, selection bias, top-k) runs on layer l's MoE input into
//!   private buffers; the publish kernel hands its ids and the guessed layer to the host with
//!   layer l's ids under the one flag. After `ExpertTiers::table_for` of layer l queued its demand
//!   moves, the guessed experts that are on NVMe only are read into a pinned prefetch store (two
//!   halves of top-k records, by layer parity) behind the demand reads, each with a landed flag.
//!   When layer l+1's call stages one of them from NVMe, [`PrefetchMover`] takes the store's record
//!   instead of reading it again (the stager stream waits on its landed flag, the synchronous
//!   mover on its ticket). A hint only: the cache's decisions and every record byte the kernels
//!   read are those of the path without it, so ids, logits and row reports are the same.
//! - **`CROW_GLM_PREFETCH_SIDE=1`** (needs `CROW_GLM_PREFETCH=1`; the reference's
//!   `GLM53_NV_PFSIDE`, `nv2.py` `predict_early`): the guess runs on a side stream, launched before
//!   layer l's own router so the two overlap; the compute stream joins it before the publish. One
//!   row only; inside a `CROW_GLM_GRAPH` capture it stays on the compute stream.
//! - **`CROW_GLM_SHARED_OVERLAP=1`** (needs `CROW_GLM_FLAGS=1`; the reference's `GLM53_K_OVL`,
//!   `glm53/k_overlap.py`): in a decode call the shared expert is queued right behind the publish,
//!   before the host's hand-off, so the GPU computes it while the host waits for the flag and plans
//!   the layer; `experts` then skips it. Same kernels on the same input into the same buffer, so
//!   bit-identical (the reference moves it out of a fused kernel and is not). Not inside a
//!   `CROW_GLM_GRAPH` capture (the shared expert stays in the replayed segment).

use crate::cnq::{self, Cnq};
use crate::cuda::{self, Pinned};
use crate::geo::Glm5Geo;
use crate::kernels::launch_v;
use cudarc::driver::sys::{self, CUfunction};

pub type Dev = sys::CUdeviceptr;

pub const ENV_FLAGS: &str = "CROW_GLM_FLAGS";
pub const ENV_LOOKAHEAD: &str = "CROW_GLM_LOOKAHEAD";
pub const ENV_PREFETCH: &str = "CROW_GLM_PREFETCH";
pub const ENV_PREFETCH_SIDE: &str = "CROW_GLM_PREFETCH_SIDE";
pub const ENV_SHARED_OVERLAP: &str = "CROW_GLM_SHARED_OVERLAP";

/// how long the host spins for a router's ids before it gives up by name (a layer's router is
/// well under a second; the WDDM TDR is 2 s)
const ROUTED_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);

/// the decode switches of `Glm5Run`
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Switches {
    pub flags: bool,
    pub lookahead: bool,
    /// `CROW_GLM_PREFETCH`: the next layer's router guess and its prefetch reads
    pub prefetch: bool,
    /// `CROW_GLM_PREFETCH_SIDE`: the guess on a side stream
    pub pf_side: bool,
    /// `CROW_GLM_SHARED_OVERLAP`: the shared expert before the host's hand-off
    pub overlap: bool,
}

impl Switches {
    /// `1` turns a switch on; unset or any other value leaves it off (the repo's `CROW_*` rule)
    pub fn parse(get: &dyn Fn(&str) -> Option<String>) -> Switches {
        let on = |k: &str| get(k).as_deref() == Some("1");
        Switches { flags: on(ENV_FLAGS), lookahead: on(ENV_LOOKAHEAD), prefetch: on(ENV_PREFETCH), pf_side: on(ENV_PREFETCH_SIDE), overlap: on(ENV_SHARED_OVERLAP) }
    }

    /// the combinations refused by name: the prefetch and the overlap need the flags (the guess
    /// travels with the router's flag; the overlap sits between the publish and the host's wait
    /// for the flag), the side stream needs the prefetch
    pub fn check(&self) -> Result<(), String> {
        if self.prefetch && !self.flags {
            return Err(format!("{ENV_PREFETCH}=1 needs {ENV_FLAGS}=1: the next layer's guess reaches the host with the router's flag"));
        }
        if self.pf_side && !self.prefetch {
            return Err(format!("{ENV_PREFETCH_SIDE}=1 needs {ENV_PREFETCH}=1: it moves the prefetch's guess to a side stream"));
        }
        if self.overlap && !self.flags {
            return Err(format!("{ENV_SHARED_OVERLAP}=1 needs {ENV_FLAGS}=1: the shared expert runs between the router's publish and the host's wait for its flag"));
        }
        Ok(())
    }

    pub fn from_env() -> Switches {
        Switches::parse(&|k| std::env::var(k).ok())
    }

    /// `[glm5_run]` line part: `flags on, lookahead off`
    pub fn label(&self) -> String {
        let s = |b: bool| if b { "on" } else { "off" };
        let mut l = format!("{} {}, {} {}", ENV_FLAGS, s(self.flags), ENV_LOOKAHEAD, s(self.lookahead));
        for (on, k) in [(self.prefetch, ENV_PREFETCH), (self.pf_side, ENV_PREFETCH_SIDE), (self.overlap, ENV_SHARED_OVERLAP)] {
            if on {
                l += &format!(", {k} on");
            }
        }
        l
    }
}

fn src(g: &Glm5Geo) -> String {
    format!(
        r#"
#define H {h}
#define V {v}
#define S {s}
// #189: the embedding row of the greedy id, BF16 widened exactly, into every residual stream
extern "C" __global__ void glm5_feed(const unsigned short* __restrict__ table, const int* __restrict__ id, float* __restrict__ x)
{{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= H) return;
    const int t = *id;
    const float v = (t < 0 || t >= V) ? __int_as_float(0x7fc00000) : __uint_as_float(((unsigned int) table[(size_t) t * H + i]) << 16);
    for (int s = 0; s < S; ++s) x[(size_t) s * H + i] = v;
}}
// #149 path B: the router's ids into mapped host memory, then the sequence flag (one block)
extern "C" __global__ void glm5_publish(const int* __restrict__ ids, const int* __restrict__ n, volatile int* host_ids, unsigned long long* ctr, volatile unsigned long long* flag)
{{
    const int nn = *n;
    for (int j = threadIdx.x; j < nn; j += blockDim.x) host_ids[j] = ids[j];
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {{
        const unsigned long long q = *ctr + 1;
        *ctr = q;
        __threadfence_system();
        *flag = q;
        __threadfence_system();
    }}
}}
// CROW_GLM_PREFETCH: the guess's layer, written behind its router (a kernel, so a graph capture holds it)
extern "C" __global__ void glm5_pred_tag(int* tag, int v)
{{
    if (threadIdx.x == 0) *tag = v;
}}
// CROW_GLM_PREFETCH: glm5_publish + the guess (k ids) and its layer tag, which goes back to -1 on
// the device, so a later publish without a guess hands over no stale one
extern "C" __global__ void glm5_publish_pred(const int* __restrict__ ids, const int* __restrict__ n, volatile int* host_ids, unsigned long long* ctr, volatile unsigned long long* flag,
                                             const int* __restrict__ pids, int k, volatile int* host_pids, int* dtag, volatile int* host_tag)
{{
    const int nn = *n;
    for (int j = threadIdx.x; j < nn; j += blockDim.x) host_ids[j] = ids[j];
    for (int j = threadIdx.x; j < k; j += blockDim.x) host_pids[j] = pids[j];
    __syncthreads();
    if (threadIdx.x == 0) {{
        *host_tag = *dtag;
        *dtag = -1;
    }}
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {{
        const unsigned long long q = *ctr + 1;
        *ctr = q;
        __threadfence_system();
        *flag = q;
        __threadfence_system();
    }}
}}
"#,
        h = g.hidden,
        v = g.vocab,
        s = g.hc_streams
    )
}

/// the switches' one NVRTC module (compiled only when a switch is on)
pub struct Kernels {
    module: cuda::Module,
    feed: CUfunction,
    publish: CUfunction,
    publish_pred: CUfunction,
    pred_tag: CUfunction,
}

impl Kernels {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo) -> Kernels {
        let module = cuda::compile(&src(g));
        Kernels { feed: module.get("glm5_feed"), publish: module.get("glm5_publish"), publish_pred: module.get("glm5_publish_pred"), pred_tag: module.get("glm5_pred_tag"), module }
    }

    /// # Safety
    /// No launch of the module is pending.
    pub unsafe fn free(&mut self) {
        self.module.unload();
    }
}

// ---------------------------------------------------------------- CROW_GLM_FLAGS

/// #149 path B: the router ids of one MoE call published into mapped pinned memory behind a
/// sequence flag (see the module doc)
pub struct Routed {
    /// `[0]` the flag (u64), `[8]` the guess's layer (i32, prefetch), `[64..]` the ids (i32),
    /// then the guess's `topk` ids (prefetch)
    host: Pinned,
    /// device: `[0]` the publish counter (u64), `[8]` the id count (i32)
    dev: Dev,
    n: usize,
    seq: u64,
    publish: CUfunction,
    publish_pred: CUfunction,
    pred_tag: CUfunction,
    /// calls since construction, and how many of them found the flag already up on the first look
    pub calls: u64,
    pub ready_first_look: u64,
    /// `CROW_GLM_PREFETCH`: the next layer's router guess (`None` = off)
    pred: Option<Predict>,
    /// the guess the last call handed over: (layer, ids)
    last_guess: Option<(usize, Vec<i32>)>,
    /// `CROW_GLM_PREFETCH`: how good the guesses were
    pub guess: GuessStats,
}

const IDS_AT: usize = 64;
const TAG_AT: usize = 8;

/// `CROW_GLM_PREFETCH`: the guesses compared with the layer's own routing when it came
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GuessStats {
    /// guesses launched on the GPU (all, of them on the side stream)
    pub launched: u64,
    pub side: u64,
    /// guesses compared with the guessed layer's own selection, its picks, the picks guessed right
    pub compared: u64,
    pub picks: u64,
    pub hits: u64,
}

impl GuessStats {
    /// guessed picks that the layer then selected, of its picks
    pub fn hit_rate(&self) -> f64 {
        if self.picks == 0 {
            0.0
        } else {
            self.hits as f64 / self.picks as f64
        }
    }
}

/// `CROW_GLM_PREFETCH`: layer l+1's router on layer l's MoE input, into private buffers
pub struct Predict {
    /// per decoder layer: the MoE router (`mlp.gate.weight` BF16 `[E][H]`, bias f32 `[E]`)
    w: Vec<Option<(Dev, Dev)>>,
    topk: usize,
    experts: usize,
    logits: Dev,
    ids: Dev,
    wts: Dev,
    /// i32: the layer of the guess in `ids`, -1 none (the publish resets it)
    tag: Dev,
    prm_kh: Dev,
    prm_route: Dev,
    prm_f: Dev,
    /// `CROW_GLM_PREFETCH_SIDE`: the side stream, the input-ready and done events
    side: Option<(sys::CUstream, sys::CUevent, sys::CUevent)>,
    /// no side guess is waiting for the compute stream's join
    joined: bool,
}

impl Predict {
    /// # Safety
    /// A CUDA context is current; the layers' router weights outlive this guess.
    unsafe fn new(layers: &[crate::glm5_model::LayerW], moe: &crate::glm5_moe::MoeGeo, side: bool) -> Predict {
        let mut w = Vec::new();
        for lw in layers {
            if w.len() <= lw.layer {
                w.resize(lw.layer + 1, None);
            }
            if let crate::glm5_model::FfnW::Moe { w: mw, .. } = &lw.ffn {
                w[lw.layer] = Some((mw.router, mw.bias));
            }
        }
        Predict::with(w, moe.experts, moe.topk, moe.hidden, moe.routed_scaling, moe.swiglu_limit, side)
    }

    /// the guess over `w` (per decoder layer: router BF16 `[e][h]`, bias f32 `[e]`), top-`k`
    ///
    /// # Safety
    /// As [`Predict::new`].
    unsafe fn with(w: Vec<Option<(Dev, Dev)>>, e: usize, k: usize, h: usize, scaling: f32, limit: f32, side: bool) -> Predict {
        Predict {
            w,
            topk: k,
            experts: e,
            logits: cuda::alloc_zeroed(e * 4),
            ids: cuda::alloc_zeroed(k * 4),
            wts: cuda::alloc_zeroed(k * 4),
            tag: cuda::to_i32_dev(&[-1]),
            prm_kh: cuda::to_i32_dev(&[h as i32]),
            prm_route: cuda::to_i32_dev(&[e as i32, k as i32]),
            prm_f: cuda::to_f32_dev(&[scaling, limit]),
            side: side.then(|| (cuda::stream_create_non_blocking(), cuda::event_create(), cuda::event_create())),
            joined: true,
        }
    }

    /// the router weights of `layer + 1` when that is a MoE layer
    fn next(&self, layer: usize) -> Option<(Dev, Dev)> {
        self.w.get(layer + 1).copied().flatten()
    }

    /// the three launches of the guess on the current stream
    unsafe fn launch(&self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, tagk: CUfunction, layer: usize, x: Dev, (r, b): (Dev, Dev)) {
        launch_v(kn.f("gemv_bf16_b"), self.experts as u32, 1, 1, 256, &[r, x, self.logits, self.prm_kh]);
        launch_v(gk.router, 1, 1, 1, crate::kernels::glm5_moe::ROUTER_THREADS as u32, &[self.logits, b, self.ids, self.wts, self.prm_route, self.prm_f]);
        launch_v(tagk, 1, 1, 1, 32, &[self.tag, (layer + 1) as u64]);
    }

    unsafe fn free(&mut self) {
        for d in [&mut self.logits, &mut self.ids, &mut self.wts, &mut self.tag, &mut self.prm_kh, &mut self.prm_route, &mut self.prm_f] {
            cuda::free_dev(d);
        }
        if let Some((s, e0, e1)) = self.side.take() {
            cuda::stream_sync(s);
            cuda::event_destroy(e0);
            cuda::event_destroy(e1);
            cuda::stream_destroy(s);
        }
    }
}

impl Routed {
    /// `n` ids per call (`t x topk` of the pass's calls)
    ///
    /// # Safety
    /// A CUDA context is current; `k` outlives this publisher.
    pub unsafe fn new(k: &Kernels, n: usize) -> Routed {
        assert!(n > 0);
        let host = Pinned::alloc((IDS_AT + (n + crate::kernels::glm5_moe::MAXK) * 4).next_multiple_of(4096));
        std::ptr::write_bytes(host.host as *mut u8, 0, host.bytes);
        let dev = cuda::alloc_named("glm5 routed-ids counter", 16);
        cuda::to_i32_into(dev + 8, &[n as i32]);
        Routed {
            host,
            dev,
            n,
            seq: 0,
            publish: k.publish,
            publish_pred: k.publish_pred,
            pred_tag: k.pred_tag,
            calls: 0,
            ready_first_look: 0,
            pred: None,
            last_guess: None,
            guess: GuessStats::default(),
        }
    }

    /// `CROW_GLM_PREFETCH`: guess layer l+1's routing in every one-row call of a MoE layer l
    /// whose next layer is a MoE layer of `layers` (their router weights are read in place);
    /// `side`: `CROW_GLM_PREFETCH_SIDE`
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this publisher is pending; the weights outlive it.
    pub unsafe fn predict_on(&mut self, layers: &[crate::glm5_model::LayerW], moe: &crate::glm5_moe::MoeGeo, side: bool) {
        if let Some(mut p) = self.pred.take() {
            p.free();
        }
        self.pred = Some(Predict::new(layers, moe, side));
    }

    /// the guess is on
    pub fn predicting(&self) -> bool {
        self.pred.is_some()
    }

    /// `CROW_GLM_PREFETCH_SIDE`: before layer `layer`'s router, launch the guess of layer + 1 on
    /// the side stream (it waits for `x` on the current stream). A no-op without the side stream,
    /// for a last MoE layer, and inside a graph capture (the guess then runs in [`Routed::predict`]).
    ///
    /// # Safety
    /// `x` holds the MoE input `[hidden]` f32, written by work queued before.
    pub unsafe fn predict_early(&mut self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, layer: usize, x: Dev) {
        let tagk = self.pred_tag;
        let Some(p) = self.pred.as_mut() else { return };
        let (Some((s, e0, e1)), Some(w)) = (p.side, p.next(layer)) else { return };
        if crate::glm5_graph::capturing() {
            return;
        }
        let main = cuda::cur_stream();
        cuda::event_record(e0, main);
        cuda::stream_wait_event(s, e0);
        cuda::set_stream(s as u64);
        p.launch(kn, gk, tagk, layer, x, w);
        cuda::set_stream(main as u64);
        cuda::event_record(e1, s);
        p.joined = false;
        self.guess.launched += 1;
        self.guess.side += 1;
    }

    /// After layer `layer`'s router: the guess of layer + 1 on the current stream, or the join of
    /// the side guess [`Routed::predict_early`] launched. A no-op when the guess is off or `layer`
    /// is the last MoE layer.
    ///
    /// # Safety
    /// As [`Routed::predict_early`].
    pub unsafe fn predict(&mut self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, layer: usize, x: Dev) {
        let tagk = self.pred_tag;
        let Some(p) = self.pred.as_mut() else { return };
        if !p.joined {
            let (_, _, e1) = p.side.expect("a side guess without its stream");
            cuda::stream_wait_event(cuda::cur_stream(), e1);
            p.joined = true;
            return;
        }
        let Some(w) = p.next(layer) else { return };
        p.launch(kn, gk, tagk, layer, x, w);
        self.guess.launched += 1;
    }

    fn flag(&self) -> u64 {
        // SAFETY: the first 8 B of a live page-aligned pinned block; the GPU writes it
        unsafe { std::ptr::read_volatile(self.host.host as *const u64) }
    }

    /// Queue on the current stream: the `n` i32 ids at `ids` into mapped memory (with the guess
    /// and its layer when the guess is on), then the flag to the next sequence number; then
    /// submit the stream (WDDM batches launches until a query or a sync).
    ///
    /// # Safety
    /// A CUDA context is current; `ids` holds `n` i32 written by work queued before.
    pub unsafe fn publish(&mut self, ids: Dev, n: usize) {
        assert_eq!(n, self.n, "glm5 flags: a call of {n} ids on a publisher of {}", self.n);
        match self.pred.as_ref() {
            None => launch_v(self.publish, 1, 1, 1, 32, &[ids, self.dev + 8, self.host.dev + IDS_AT as u64, self.dev, self.host.dev]),
            Some(p) => launch_v(
                self.publish_pred,
                1,
                1,
                1,
                32,
                &[
                    ids,
                    self.dev + 8,
                    self.host.dev + IDS_AT as u64,
                    self.dev,
                    self.host.dev,
                    p.ids,
                    p.topk as u64,
                    self.host.dev + (IDS_AT + self.n * 4) as u64,
                    p.tag,
                    self.host.dev + TAG_AT as u64,
                ],
            ),
        }
        self.seq += 1;
        cuda::stream_query(cuda::cur_stream());
    }

    /// Spin until the flag shows the last publish; its ids. `Err` by name after
    /// [`ROUTED_TIMEOUT`].
    pub fn wait(&mut self) -> Result<Vec<i32>, String> {
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        self.calls += 1;
        if self.flag() >= self.seq {
            self.ready_first_look += 1;
        }
        while self.flag() < self.seq {
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 {
                // SAFETY: a status query; it also submits a pending WDDM command buffer
                unsafe { cuda::stream_query(cuda::cur_stream()) };
                if t0.elapsed() > ROUTED_TIMEOUT {
                    return Err(format!("{ENV_FLAGS}: the router's ids did not arrive in {} s (flag {}, want {})", ROUTED_TIMEOUT.as_secs(), self.flag(), self.seq));
                }
            }
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        let p = unsafe { (self.host.host as *const u8).add(IDS_AT) as *const i32 };
        // SAFETY: `n` i32 inside the block; the kernel wrote them before the flag (fence_system)
        Ok((0..self.n).map(|i| unsafe { std::ptr::read_volatile(p.add(i)) }).collect())
    }

    /// [`Routed::wait`] for the call of decoder layer `layer`. With the guess on, the ids are
    /// scored against the guess an earlier call made for `layer`, and this call's guess (when the
    /// device made one) goes to [`take_hint`] for `ExpertTiers::table_for`.
    pub fn wait_layer(&mut self, layer: usize) -> Result<Vec<i32>, String> {
        let ids = self.wait()?;
        let Some(p) = self.pred.as_ref() else { return Ok(ids) };
        if let Some((gl, g)) = self.last_guess.take() {
            if gl == layer {
                self.guess.compared += 1;
                self.guess.picks += ids.len() as u64;
                self.guess.hits += ids.iter().filter(|e| g.contains(e)).count() as u64;
            }
        }
        // SAFETY: inside the block, written by the publish before the flag
        let tag = unsafe { std::ptr::read_volatile((self.host.host as *const u8).add(TAG_AT) as *const i32) };
        if tag >= 0 {
            let q = unsafe { (self.host.host as *const u8).add(IDS_AT + self.n * 4) as *const i32 };
            let g: Vec<i32> = (0..p.topk).map(|i| unsafe { std::ptr::read_volatile(q.add(i)) }).collect();
            self.last_guess = Some((tag as usize, g.clone()));
            post_hint(Some(Hint { layer: tag as usize, ids: g }));
        } else {
            post_hint(None);
        }
        Ok(ids)
    }

    /// # Safety
    /// No launch of a publish or a guess is pending.
    pub unsafe fn free(&mut self) {
        if let Some(mut p) = self.pred.take() {
            p.free();
        }
        self.host.free();
        cuda::free_dev(&mut self.dev);
    }
}

// ---------------------------------------------------------------- CROW_GLM_PREFETCH: the host side

/// the guess of one call: the next MoE layer (decoder index) and its `topk` ids
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hint {
    pub layer: usize,
    pub ids: Vec<i32>,
}

thread_local! {
    static HINT: std::cell::RefCell<Option<Hint>> = const { std::cell::RefCell::new(None) };
}

/// [`Routed::wait_layer`] leaves the call's guess here: the hand-off to the expert hook, the way
/// `glm5_moe::lane::post` hands the CPU lane its call
pub fn post_hint(h: Option<Hint>) {
    HINT.with(|c| *c.borrow_mut() = h);
}

/// the guess the last [`Routed::wait_layer`] left, once
pub fn take_hint() -> Option<Hint> {
    HINT.with(|c| c.borrow_mut().take())
}

/// `CU_STREAM_WAIT_VALUE_GEQ`
const WAIT_GEQ: u32 = 0;

/// `CROW_GLM_PREFETCH`: host-side counts since the store was made
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PrefetchStats {
    /// guesses handed to `table_for` for its next layer
    pub hints: u64,
    /// guessed experts already in VRAM or pinned (no read)
    pub resident: u64,
    /// records read into the store, and their bytes
    pub issued: u64,
    pub bytes: u64,
    /// store records a call staged instead of reading them (prefetch hits)
    pub used: u64,
    /// store records dropped unused (their slot was needed again, or the store was emptied)
    pub wasted: u64,
}

/// `CROW_GLM_PREFETCH`: pinned records read ahead from NVMe, two halves of `k` by layer parity,
/// each slot with a landed flag in mapped memory.
///
/// Why a half is free to be rewritten when the guess of layer m arrives (in `table_for` of m - 1):
/// its last reader is the staging copy of `table_for` of m - 2, queued before m - 2's experts;
/// the host is in `table_for` of m - 1 only after m - 1's router flag, so (in-order compute
/// stream, the stager's event before those experts) that copy has run. A read still in flight in
/// the half (a guess never used) is waited for before its slot is rewritten.
pub struct Prefetch {
    buf: Pinned,
    flags: Pinned,
    rb: u64,
    k: usize,
    /// per slot: (cache layer, expert)
    key: Vec<Option<(usize, u32)>>,
    ticket: Vec<Option<crate::nvme_source::Ticket>>,
    value: Vec<u64>,
    used: Vec<bool>,
    bytes: Vec<u64>,
    seq: u64,
    pub stats: PrefetchStats,
}

impl Prefetch {
    /// `k` records per half of `rb` bytes each
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(k: usize, rb: u64) -> Prefetch {
        let n = 2 * k;
        let flags = Pinned::alloc((n * 8).next_multiple_of(4096));
        std::ptr::write_bytes(flags.host as *mut u8, 0, flags.bytes);
        Prefetch {
            buf: Pinned::alloc(n * rb as usize),
            flags,
            rb,
            k,
            key: vec![None; n],
            ticket: (0..n).map(|_| None).collect(),
            value: vec![0; n],
            used: vec![false; n],
            bytes: vec![0; n],
            seq: 0,
            stats: PrefetchStats::default(),
        }
    }

    /// the pinned bytes of the store (records and flags)
    pub fn pinned_bytes(&self) -> u64 {
        (self.buf.bytes + self.flags.bytes) as u64
    }

    /// the slot holding expert `e` of cache layer `l`
    pub fn find(&self, l: usize, e: u32) -> Option<usize> {
        self.key.iter().position(|k| *k == Some((l, e)))
    }

    fn host(&self, i: usize) -> *mut u8 {
        // SAFETY: slot i < 2k of the store
        unsafe { (self.buf.host as *mut u8).add(i * self.rb as usize) }
    }

    /// drop slot `i`: its read reports (waits if it still runs), an unused record counts as wasted
    fn retire(&mut self, src: &crate::nvme_source::NvmeSource, i: usize) -> Result<(), String> {
        use crate::nvme_source::ColdSource;
        let r = self.ticket[i].take().map_or(Ok(()), |t| src.wait(t).map(|_| ()));
        if self.key[i].is_some() && !self.used[i] {
            self.stats.wasted += 1;
        }
        self.key[i] = None;
        self.used[i] = false;
        r.map_err(|e| format!("{ENV_PREFETCH}: a prefetch read: {e}"))
    }

    /// Read `want` (experts of cache layer `l`, on NVMe only) into the half of `l`'s parity,
    /// keeping the slots that already hold one of `keep` (of layer `l`); at most `k` records.
    ///
    /// # Safety
    /// `recs` are layer `l`'s records; no queued copy still reads the half (see the type doc).
    pub unsafe fn issue(&mut self, src: &crate::nvme_source::NvmeSource, recs: &[crate::nvme_source::ExpertRecord], l: usize, keep: &[u32], want: &[u32]) -> Result<usize, String> {
        let half = (l % 2) * self.k..(l % 2 + 1) * self.k;
        let mut err = Ok(());
        for i in half.clone() {
            if !matches!(self.key[i], Some((kl, ke)) if kl == l && keep.contains(&ke)) {
                let r = self.retire(src, i);
                if err.is_ok() {
                    err = r;
                }
            }
        }
        err?;
        let mut n = 0;
        for &e in want {
            let Some(i) = half.clone().find(|&i| self.key[i].is_none()) else { break };
            let dst = crate::nvme_source::RecordDst { gu: self.host(i), dn: std::ptr::null_mut() };
            let rec = recs[e as usize];
            let bytes = rec.parts(&dst).iter().map(|p| p.1.len as u64).sum::<u64>();
            self.seq += 1;
            self.value[i] = self.seq;
            let flag = crate::nvme_source::Landed { flag: (self.flags.host as *mut u64).add(i), value: self.seq };
            // SAFETY: the destination is store slot i (`rb` bytes, nothing reads it until its
            // flag / ticket), the flag its own word, written only by this read until retired
            self.ticket[i] = Some(src.fetch_landed(&[(rec, dst)], &[flag])?);
            self.key[i] = Some((l, e));
            self.bytes[i] = bytes;
            self.stats.issued += 1;
            self.stats.bytes += bytes;
            n += 1;
        }
        Ok(n)
    }

    /// empty the store (every read reports; unused records count as wasted)
    pub fn forget(&mut self, src: &crate::nvme_source::NvmeSource) -> Result<(), String> {
        let mut err = Ok(());
        for i in 0..self.key.len() {
            let r = self.retire(src, i);
            if err.is_ok() {
                err = r;
            }
        }
        err
    }

    /// # Safety
    /// No queued copy reads the store.
    pub unsafe fn free(&mut self, src: &crate::nvme_source::NvmeSource) -> Result<(), String> {
        let r = self.forget(src);
        self.buf.free();
        self.flags.free();
        r
    }
}

/// `CROW_GLM_PREFETCH`, after `table_for` of decoder layer `layer` queued its moves: the guess
/// [`Routed::wait_layer`] left for `layer + 1`, its experts on NVMe only read into the store
/// behind the demand reads (`records` and `cache` by cache layer, `first_moe` the first MoE
/// decoder layer). A guess for another layer is dropped.
///
/// # Safety
/// As [`Prefetch::issue`].
pub unsafe fn prefetch_hinted(
    pf: &mut Prefetch,
    src: &crate::nvme_source::NvmeSource,
    records: &[Vec<crate::nvme_source::ExpertRecord>],
    cache: &crate::expert_cache::ExpertCache,
    first_moe: usize,
    layer: usize,
) -> Result<(), String> {
    let Some(h) = take_hint() else { return Ok(()) };
    if h.layer != layer + 1 {
        return Ok(());
    }
    let Some(l) = h.layer.checked_sub(first_moe).filter(|&l| l < records.len()) else { return Ok(()) };
    pf.stats.hints += 1;
    let (mut keep, mut want) = (Vec::new(), Vec::new());
    for &e in &h.ids {
        if e < 0 || e as usize >= cache.experts {
            continue;
        }
        let e = e as u32;
        if cache.tier(l, e) != crate::expert_cache::Tier::Nvme {
            pf.stats.resident += 1;
        } else if pf.find(l, e).is_some() {
            keep.push(e);
        } else if !want.contains(&e) {
            want.push(e);
        }
    }
    if want.is_empty() {
        return Ok(());
    }
    pf.issue(src, &records[l], l, &keep, &want).map(|_| ())
}

/// `CROW_GLM_PREFETCH`: `serve`'s mover of one call with the store in front of `inner`. An NVMe
/// read into staging slot s of an expert the store holds is not issued; the landing-to-staging
/// copy of s takes the store's record instead (on `stream`, the stager's, behind a wait on its
/// landed flag; without a stream, after the host waited for its read, on the current stream
/// before `serve`'s barrier). Every other move goes to `inner` unchanged; a read into a pinned
/// slot is never taken from the store.
pub struct PrefetchMover<'a> {
    inner: &'a mut dyn crate::glm5_tiers::Mover,
    pf: &'a mut Prefetch,
    src: &'a crate::nvme_source::NvmeSource,
    l: usize,
    stage: Dev,
    stream: Option<sys::CUstream>,
    /// (staging slot, store slot)
    redirect: Vec<(u32, usize)>,
}

impl<'a> PrefetchMover<'a> {
    pub fn new(inner: &'a mut dyn crate::glm5_tiers::Mover, pf: &'a mut Prefetch, src: &'a crate::nvme_source::NvmeSource, l: usize, stage: Dev, stream: Option<sys::CUstream>) -> PrefetchMover<'a> {
        PrefetchMover { inner, pf, src, l, stage, stream, redirect: Vec::new() }
    }
}

impl crate::glm5_tiers::Mover for PrefetchMover<'_> {
    fn nvme(&mut self, jobs: &[(u32, crate::glm5_tiers::Dst)]) -> Result<u64, String> {
        use crate::glm5_tiers::Dst;
        let mut rest = Vec::with_capacity(jobs.len());
        let mut bytes = 0;
        for &(e, d) in jobs {
            match (d, self.pf.find(self.l, e)) {
                (Dst::Landing(s), Some(i)) => {
                    match self.stream {
                        // SAFETY: the store's mapped flag word of slot i
                        Some(st) => unsafe { cuda::ck(sys::cuStreamWaitValue64_v2(st, self.pf.flags.dev + (i * 8) as u64, self.pf.value[i], WAIT_GEQ)) },
                        None => {
                            use crate::nvme_source::ColdSource;
                            if let Some(t) = self.pf.ticket[i].take() {
                                self.src.wait(t).map_err(|e| format!("{ENV_PREFETCH}: a prefetch read: {e}"))?;
                            }
                        }
                    }
                    if !self.pf.used[i] {
                        self.pf.used[i] = true;
                        self.pf.stats.used += 1;
                    }
                    bytes += self.pf.bytes[i];
                    self.redirect.push((s, i));
                }
                _ => rest.push((e, d)),
            }
        }
        if !rest.is_empty() {
            bytes += self.inner.nvme(&rest)?;
        }
        Ok(bytes)
    }
    fn landing_to_stage(&mut self, s: u32) {
        let Some(&(_, i)) = self.redirect.iter().find(|r| r.0 == s) else {
            return self.inner.landing_to_stage(s);
        };
        let (dst, src, rb) = (self.stage + s as u64 * self.pf.rb, self.pf.host(i) as *const std::ffi::c_void, self.pf.rb as usize);
        // SAFETY: a staging slot and a store record, `rb` bytes each; the record has landed (the
        // stream waits on its flag, or the host waited for its read)
        unsafe {
            match self.stream {
                Some(st) => cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, src, rb, st)),
                None => cuda::upload_from_pinned(dst, src, rb),
            }
        }
    }
    fn pinned_to_stage(&mut self, q: u32, s: u32) {
        self.inner.pinned_to_stage(q, s)
    }
    fn vram_to_stage(&mut self, v: u32, s: u32) {
        self.inner.vram_to_stage(v, s)
    }
    fn barrier(&mut self) {
        self.inner.barrier()
    }
    fn vram_to_pinned(&mut self, v: u32, q: u32) {
        self.inner.vram_to_pinned(v, q)
    }
    fn stage_to_vram(&mut self, s: u32, v: u32) {
        self.inner.stage_to_vram(s, v)
    }
}

// ---------------------------------------------------------------- CROW_GLM_LOOKAHEAD

/// #189: the token embedding in VRAM and the gather that feeds a device id into the residual
pub struct Feed {
    pub table: Dev,
    pub bytes: u64,
    hidden: usize,
    feed: CUfunction,
}

impl Feed {
    /// the BF16 `embed_tokens` of the container, as stored, into VRAM (`glm5_model::embed_rows`'
    /// tensor and checks)
    ///
    /// # Safety
    /// A CUDA context is current; `k` outlives this feed.
    pub unsafe fn load(k: &Kernels, cnq: &mut Cnq, g: &Glm5Geo) -> Feed {
        let t = cnq.find("model.language_model.embed_tokens.weight", "text").clone();
        assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("bf16", &[g.vocab as u64, g.hidden as u64][..]), "glm5 lookahead: embed_tokens");
        Feed::from_table(k, &cnq.read_bytes(&t), g)
    }

    /// # Safety
    /// A CUDA context is current; `raw` is `[vocab][hidden]` BF16; `k` outlives this feed.
    pub unsafe fn from_table(k: &Kernels, raw: &[u8], g: &Glm5Geo) -> Feed {
        assert_eq!(raw.len(), g.vocab * g.hidden * 2, "glm5 lookahead: the embedding table is not [{}][{}] BF16", g.vocab, g.hidden);
        Feed { table: cuda::upload_dev_named("glm5 lookahead embedding table", raw), bytes: raw.len() as u64, hidden: g.hidden, feed: k.feed }
    }

    /// Queue: `x` `[streams][hidden]` f32 = the embedding row of the i32 id at `id`, in every
    /// stream (an id outside the vocab writes NaN; the host refuses that id when it reads it).
    ///
    /// # Safety
    /// `x` holds `streams x hidden` f32, `id` one i32 written by work queued before.
    pub unsafe fn gather(&self, id: Dev, x: Dev) {
        launch_v(self.feed, self.hidden.div_ceil(256) as u32, 1, 1, 256, &[self.table, id, x]);
    }

    /// # Safety
    /// No gather is pending.
    pub unsafe fn free(&mut self) {
        cuda::free_dev(&mut self.table);
    }
}

/// #189: the head's greedy id (and its logits, when kept) copied async into pinned memory, read
/// by the host after a later sync point of the same stream
pub struct Readback {
    id: Pinned,
    logits: Pinned,
    vocab: usize,
}

impl Readback {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(vocab: usize) -> Readback {
        Readback { id: Pinned::alloc(4096), logits: Pinned::alloc((vocab * 4).next_multiple_of(4096)), vocab }
    }

    /// Queue on the current stream: the id at `next` (and the `vocab` logits at `logits`).
    ///
    /// # Safety
    /// The sources hold that much, written by work queued before.
    pub unsafe fn enqueue(&self, next: Dev, logits: Option<Dev>) {
        cuda::ck(sys::cuMemcpyDtoHAsync_v2(self.id.host, next, 4, cuda::cur_stream()));
        if let Some(l) = logits {
            cuda::ck(sys::cuMemcpyDtoHAsync_v2(self.logits.host, l, self.vocab * 4, cuda::cur_stream()));
        }
    }

    /// the id of the last [`Readback::enqueue`]
    ///
    /// # Safety
    /// The stream passed a sync point (a stream sync, a routed flag) after the enqueue.
    pub unsafe fn id(&self) -> i32 {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        std::ptr::read_volatile(self.id.host as *const i32)
    }

    /// the logits of the last [`Readback::enqueue`] with logits
    ///
    /// # Safety
    /// As [`Readback::id`].
    pub unsafe fn logits(&self) -> Vec<f32> {
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        std::slice::from_raw_parts(self.logits.host as *const f32, self.vocab).to_vec()
    }

    /// # Safety
    /// No copy into it is pending.
    pub unsafe fn free(&mut self) {
        self.id.free();
        self.logits.free();
    }
}

/// the host reference the gather must equal bit for bit: `trunk_input(bf16 row widened)`
pub fn host_feed(raw: &[u8], g: &Glm5Geo, id: usize) -> Vec<f32> {
    let row = g.hidden * 2;
    crate::glm5_model::trunk_input(&cnq::bf16_bytes_to_f32(&raw[id * row..(id + 1) * row]), g.hidden, g.hc_streams)
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use crate::glm5_model::{self as gm, Take};
    use crate::glm5_moe::MoeGeo;
    use crate::glm5_tiers::{ExpertTiers, Generated, Glm5Run, TierSizes, TokenReport};

    #[test]
    fn only_1_turns_a_switch_on() {
        let parse = |pairs: Vec<(&str, &str)>| Switches::parse(&|k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string()));
        assert_eq!(parse(vec![]), Switches::default());
        assert_eq!(parse(vec![(ENV_FLAGS, "1")]), Switches { flags: true, lookahead: false, ..Switches::default() });
        assert_eq!(parse(vec![(ENV_LOOKAHEAD, "1")]), Switches { flags: false, lookahead: true, ..Switches::default() });
        assert_eq!(parse(vec![(ENV_FLAGS, "1"), (ENV_LOOKAHEAD, "1")]), Switches { flags: true, lookahead: true, ..Switches::default() });
        for v in ["0", "", "on", "true", "yes", " 1", "2"] {
            assert_eq!(parse(vec![(ENV_FLAGS, v), (ENV_LOOKAHEAD, v)]), Switches::default(), "{v:?}");
        }
        assert_eq!(Switches { flags: true, lookahead: false, ..Switches::default() }.label(), "CROW_GLM_FLAGS on, CROW_GLM_LOOKAHEAD off");
    }

    /// splitmix64
    struct Rng(u64);

    impl Rng {
        fn u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// uniform in [-1, 1)
        fn sym(&mut self) -> f32 {
            (self.u64() >> 40) as f32 / (1u64 << 23) as f32 - 1.0
        }
    }

    /// The device gather against the host feed (`embed_rows` + `trunk_input`), bit for bit, for
    /// every id of a synthetic table that holds NaN, +-Inf, -0, denormals and random values; an
    /// id outside the vocab writes NaN everywhere.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_the_feed_is_the_host_feed_bit_for_bit() {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        g.vocab = 64;
        let mut rng = Rng(0x189);
        let mut raw = vec![0u8; g.vocab * g.hidden * 2];
        for w in raw.chunks_exact_mut(2) {
            w.copy_from_slice(&(rng.u64() as u16).to_le_bytes());
        }
        let specials: [u16; 7] = [0x7FC0, 0x7F80, 0xFF80, 0x8000, 0x0001, 0x807F, 0xFFFF];
        for (i, v) in specials.iter().enumerate() {
            for id in [0usize, 7, 63] {
                let at = (id * g.hidden + i * 97) * 2;
                raw[at..at + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut f = Feed::from_table(&k, &raw, &g);
            let n = g.hc_streams * g.hidden;
            let mut x = cuda::alloc_named("test feed x", n * 4);
            let mut id = cuda::alloc_named("test feed id", 4);
            for t in 0..g.vocab {
                cuda::to_i32_into(id, &[t as i32]);
                f.gather(id, x);
                cuda::sync();
                let got: Vec<u32> = cuda::dtoh_t(x, n);
                let want: Vec<u32> = host_feed(&raw, &g, t).iter().map(|v| v.to_bits()).collect();
                assert!(got == want, "id {t}: {} of {n} values differ in bits", got.iter().zip(&want).filter(|(a, b)| a != b).count());
            }
            cuda::to_i32_into(id, &[g.vocab as i32]);
            f.gather(id, x);
            cuda::sync();
            let got: Vec<f32> = cuda::dtoh(x, n);
            assert!(got.iter().all(|v| v.is_nan()), "an id outside the vocab must write NaN");
            cuda::free_dev(&mut x);
            cuda::free_dev(&mut id);
            f.free();
            k.free();
        }
    }

    /// The router-ids publisher never lets the host see a flag before the ids it covers: a
    /// kernel that holds the stream for 0 / 0.3 / 2 ms and then writes fresh ids, then
    /// `publish`; `wait` must return exactly those ids, 300 calls in a row.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_routed_ids_arrive_with_their_flag() {
        const SLOW: &str = r#"
extern "C" __global__ void slow_ids(int* ids, const int* base, const long long* ns)
{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    do { asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t)); } while ((long long) (t - t0) < *ns);
    if (threadIdx.x < 8) ids[threadIdx.x] = *base * 8 + threadIdx.x;
}
"#;
        let g = Glm5Geo::GLM_5_3_FLASH;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut m = cuda::compile(SLOW);
            let slow = m.get("slow_ids");
            let mut r = Routed::new(&k, 8);
            let mut ids = cuda::alloc_named("test ids", 32);
            let mut base = cuda::alloc_named("test base", 4);
            let mut ns = cuda::alloc_named("test ns", 8);
            for i in 1..=300i32 {
                cuda::to_i32_into(base, &[i]);
                cuda::upload_into(ns, &([0i64, 300_000, 2_000_000][i as usize % 3]).to_le_bytes());
                launch_v(slow, 1, 1, 1, 32, &[ids, base, ns]);
                r.publish(ids, 8);
                let got = r.wait().unwrap();
                assert_eq!(got, (0..8).map(|j| i * 8 + j).collect::<Vec<_>>(), "call {i}");
            }
            eprintln!("glm5 flags: {} calls, {} with the flag already up on the first look", r.calls, r.ready_first_look);
            r.free();
            for d in [&mut ids, &mut base, &mut ns] {
                cuda::free_dev(d);
            }
            m.unload();
            k.free();
        }
    }

    /// a synthetic glm5_next container, removed on drop
    pub(crate) struct Synth {
        dir: std::path::PathBuf,
        pub(crate) path: String,
    }

    impl Drop for Synth {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// fp16 bits of a value of normal fp16 range (truncating the mantissa)
    fn f16_bits(v: f32) -> u16 {
        let b = v.to_bits();
        let e = ((b >> 23) & 0xFF) as i32 - 127 + 15;
        assert!((1..31).contains(&e), "{v} is outside the normal fp16 range");
        (((b >> 16) & 0x8000) | ((e as u32) << 10) | ((b & 0x7F_FFFF) >> 13)) as u16
    }

    /// The synthetic model of `g` (the GLM-5.3-Flash layer shapes, `g.layers` layers, the last
    /// `g.layers - g.dense_prefix` of them MoE with `g.experts` MUL1 records of `rec` bytes each),
    /// every tensor of the `glm5_model` plan at its planned dtype and shape, values bounded so the
    /// trunk stays finite: NVFP4 blocks with random codes and scale bytes 0.5-1.0 under a global
    /// scale of 0.2 / sqrt(cols), BF16 matrices uniform in +-1 / sqrt(cols) (the embedding +-0.5),
    /// f32 norms 1 +- 0.05 and other vectors +-0.05, MUL1 records with random trellis words and
    /// fp16 suh / svh of magnitude 0.06-0.12.
    pub(crate) fn synth_model(g: &Glm5Geo, rec: u64) -> Synth {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("crow-glm5-flags-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-glm5-model.cnq");
        let mut plan: Vec<gm::Planned> = (0..g.layers).flat_map(|l| gm::layer_tensors(g, l)).collect();
        plan.extend(gm::model_tensors(g));
        let mut rng = Rng(0x0189_F1A6);
        let mut tensors = Vec::new();
        let mut off = 0u64;
        let mut f = std::io::BufWriter::with_capacity(1 << 22, std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        for p in &plan {
            let n: u64 = p.shape.iter().product();
            let cols = *p.shape.last().unwrap() as f32;
            let (dtype, bytes): (&str, Vec<u8>) = match p.take {
                Take::Bf16 => {
                    let a = if p.name.contains("embed_tokens") { 0.5 } else { 1.0 / cols.sqrt() };
                    ("bf16", (0..n).flat_map(|_| gm::f32_to_bf16_rne(a * rng.sym()).to_le_bytes()).collect())
                }
                Take::F32 => {
                    let v = |r: &mut Rng| {
                        if p.name.ends_with("norm.weight") {
                            1.0 + 0.05 * r.sym()
                        } else if p.name.ends_with("A_log") {
                            0.5 + 0.5 * r.sym()
                        } else {
                            0.05 * r.sym()
                        }
                    };
                    ("f32", (0..n).flat_map(|_| v(&mut rng).to_le_bytes()).collect())
                }
                Take::Fp4 | Take::Fp4ToF32 | Take::Fp4ToBf16 => {
                    let mut b = vec![0u8; (n.div_ceil(64) * 36) as usize];
                    for blk in b.chunks_exact_mut(36) {
                        for s in blk.iter_mut().take(4) {
                            *s = 0x30 + (rng.u64() % 9) as u8;
                        }
                        for c in blk.iter_mut().skip(4) {
                            *c = rng.u64() as u8;
                        }
                    }
                    ("nvfp4", b)
                }
            };
            let mut t = serde_json::json!({ "name": p.name, "section": "text", "dtype": dtype, "offset": off, "n_values": n, "shape": p.shape });
            if dtype == "nvfp4" {
                t["global_scale"] = serde_json::json!(0.2 / cols.sqrt());
            }
            tensors.push(t);
            f.write_all(&bytes).unwrap();
            off += bytes.len() as u64;
        }
        // the records start on a 4096-B file offset (blob at 12)
        let pad = (4096 - (12 + off) % 4096) % 4096;
        f.write_all(&vec![0u8; pad as usize]).unwrap();
        off += pad;
        let specs = crate::kernels::mul1::record_specs(g.hidden, g.expert_inter, 3, false);
        let trellis = specs[0].suh_off;
        for l in g.dense_prefix..g.layers {
            for e in 0..g.experts as u32 {
                let mut r = vec![0u8; rec as usize];
                for w in r[..trellis].chunks_exact_mut(8) {
                    w.copy_from_slice(&rng.u64().to_le_bytes());
                }
                for s in &specs {
                    for (at, cnt) in [(s.suh_off, s.k), (s.svh_off, s.n)] {
                        for i in 0..cnt {
                            let v = (0.06 + 0.03 * (rng.sym() + 1.0)) * if rng.u64() & 1 == 1 { -1.0 } else { 1.0 };
                            r[at + 2 * i..at + 2 * i + 2].copy_from_slice(&f16_bits(v).to_le_bytes());
                        }
                    }
                }
                for (k, q) in ["gate", "up", "down"].into_iter().enumerate() {
                    tensors.push(serde_json::json!({ "name": crate::nvme_source::glm5_expert_tensor_name(l as u32, e, q), "section": "text", "dtype": "mul1",
                        "offset": off + k as u64 * (rec / 3), "n_values": 2048u64 * 4096, "shape": [2048, 4096] }));
                }
                f.write_all(&r).unwrap();
                off += rec;
            }
        }
        let sha = |s: &str| cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "blob_offset": 12, "recipe": "synthetic-glm5-flags",
            "model": { "family": "Glm5Next", "model_type": "glm5_next_text", "config_json": "{}", "config_json_sha256": sha("{}"),
                "generation_config_json": "{}", "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-glm5-flags", "revision": "189", "shards": [] }, "geo": {} },
            "tensors": tensors
        });
        let ib = serde_json::to_vec(&index).unwrap();
        f.write_all(&ib).unwrap();
        f.write_all(&(ib.len() as u64).to_le_bytes()).unwrap();
        f.flush().unwrap();
        Synth { dir, path: path.to_str().unwrap().to_string() }
    }

    /// a report without its clock (the only field the switches may change)
    pub(crate) fn unclocked(r: &TokenReport) -> TokenReport {
        TokenReport { secs: 0.0, ..r.clone() }
    }

    /// The switches are invisible in the output. A synthetic glm5_next model (the real layer
    /// shapes; layers 0-2 KDA + dense SwiGLU, layer 3 MLA/DSA + MoE with 16 MUL1 experts, top-8,
    /// vocab 2048), a 5-id prompt and 6 greedy ids, VRAM 3 + pinned 4 slots (every kind of move),
    /// a fresh store per arm: with `CROW_GLM_FLAGS`, `CROW_GLM_LOOKAHEAD` and both, the ids, every
    /// logit's bits and every row report except its clock equal the switch-off run. Also serve's
    /// door: `row` with the flags gives the switch-off ids and logits.
    #[test]
    #[ignore = "needs the GPU (about 1 GB VRAM, a 0.9 GB synthetic container in the temp dir): cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_the_switches_are_invisible_in_ids_logits_and_reports() {
        const REC: u64 = 9_474_048;
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (4, 3, 16, 8, 2048);
        let t0 = std::time::Instant::now();
        let s = synth_model(&g, REC);
        eprintln!("glm5 flags: synthetic model written in {:.1} s", t0.elapsed().as_secs_f64());
        let (spec, records) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        assert_eq!((spec.bytes, records), (REC, 16));
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let n = 6;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let arms = [
            Switches::default(),
            Switches { flags: true, lookahead: false, ..Switches::default() },
            Switches { flags: false, lookahead: true, ..Switches::default() },
            Switches { flags: true, lookahead: true, ..Switches::default() },
        ];
        let mut outs: Vec<(Switches, Generated, Vec<TokenReport>)> = Vec::new();
        let mut rows_flags: (Vec<i64>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            for sw in arms {
                run.set_switches(&mut cnq, sw);
                assert_eq!(run.switches(), sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                let mut reps: Vec<TokenReport> = Vec::new();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                eprintln!("glm5 flags {}: ids {:?}, NVMe reads {}", sw.label(), gen.ids, tiers.nvme_reads);
                tiers.free();
                outs.push((sw, gen, reps));
            }
            // serve's door with the flags: the prompt rows, then one decode row per id
            run.set_switches(&mut cnq, Switches { flags: true, lookahead: false, ..Switches::default() });
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            for i in 0..n {
                rows_flags.0.push(tok);
                rows_flags.1.push(cuda::dtoh(run.logits_dev(), g.vocab));
                if i + 1 < n {
                    tok = run.row(&mut cnq, &mut tiers, tok, prompt.len() + i, true).unwrap().unwrap();
                }
            }
            tiers.free();
            run.free();
        }
        drop(cnq);
        let (_, g0, r0) = &outs[0];
        assert_eq!((g0.ids.len(), g0.logits.len(), r0.len()), (n, n, prompt.len() + n - 1));
        assert_eq!(r0.iter().map(|r| r.pos).collect::<Vec<_>>(), (0..prompt.len() + n - 1).collect::<Vec<_>>());
        let finite = g0.logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        assert!(r0.iter().map(|r| r.moves.iter().map(|m| m.nvme_reads()).sum::<u64>()).sum::<u64>() > 0, "the run must read records from NVMe");
        let bits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect::<Vec<_>>();
        for (sw, gx, rx) in &outs[1..] {
            assert_eq!(gx.ids, g0.ids, "{}: ids", sw.label());
            assert_eq!(gx.logits.len(), g0.logits.len(), "{}: logit rows", sw.label());
            let diff = bits(&gx.logits, &g0.logits);
            assert!(diff.iter().all(|&d| d == 0), "{}: logits differ in bits per generated position {diff:?}", sw.label());
            assert_eq!(rx.iter().map(unclocked).collect::<Vec<_>>(), r0.iter().map(unclocked).collect::<Vec<_>>(), "{}: row reports", sw.label());
        }
        assert_eq!(rows_flags.0, g0.ids, "row() with the flags: ids");
        let diff = bits(&rows_flags.1, &g0.logits);
        assert!(diff.iter().all(|&d| d == 0), "row() with the flags: logits differ in bits {diff:?}");
    }

    // ---------------------------------------------------------------- #149 path B: the stager

    #[test]
    fn the_stager_switch_needs_the_flags_and_refuses_the_cpu_lane() {
        use crate::glm5_tiers::stager_on;
        assert_eq!(stager_on(None, None, None), Ok(false));
        for v in ["0", "", "on", "true", " 1", "2"] {
            assert_eq!(stager_on(Some(v), None, Some("1")), Ok(false), "{v:?}");
        }
        assert_eq!(stager_on(Some("1"), Some("1"), None), Ok(true));
        assert_eq!(stager_on(Some("1"), Some("1"), Some("0")), Ok(true));
        for flags in [None, Some("0"), Some("on")] {
            let e = stager_on(Some("1"), flags, None).unwrap_err();
            assert!(e.starts_with("CROW_GLM_STAGER=1 needs CROW_GLM_FLAGS=1"), "{e}");
        }
        let e = stager_on(Some("1"), Some("1"), Some("1")).unwrap_err();
        assert!(e.starts_with("CROW_GLM_STAGER=1 and CROW_GLM_CPU_LANE=1"), "{e}");
        // what the plan books for the stager: GLM-5.3-Flash, 42 MoE layers x 288, the 3-bit record, top-8
        let g = Glm5Geo::GLM_5_3_FLASH;
        assert_eq!(crate::glm5_tiers::stager_pinned_bytes(g.moe_layers(), g.experts, g.topk, 9_474_048), 75_988_992);
    }

    /// the synthetic 8-layer model of the stager tests (the #190 shape): layers 0-2 KDA + dense,
    /// 3-7 MoE with 16 MUL1 experts, top-8, DSA at 3 and 7, vocab 2048
    fn geo8() -> Glm5Geo {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (8, 3, 16, 8, 2048);
        g
    }

    /// one arm of the stager test
    #[derive(Clone, Copy, Debug)]
    struct Arm {
        name: &'static str,
        sw: Switches,
        graph: bool,
        stager: bool,
        zerocopy: bool,
        lfu: bool,
    }

    /// The stager is invisible in the output. The synthetic 8-layer model (5 MoE layers, so the
    /// staging slots, the landing buffer and the table rows are reused across layers and rows), a
    /// 5-id prompt and 6 greedy ids, VRAM 3 + pinned 4 slots, a fresh store per arm. Against the
    /// synchronous arm of the same cache rule: `CROW_GLM_STAGER` with the flags, with the
    /// lookahead, with `CROW_GLM_GRAPH`, with `CROW_GLM_PINNED=zerocopy`, and under LFU (whose
    /// NVMe misses can enter pinned while another pick leaves it: the gate) give the same ids,
    /// every logit's bits and every row report except its clock. Every stager arm reads from NVMe
    /// only through landed flags (`landed_reads` = the store's NVMe reads). Also serve's door: `row`
    /// with flags + stager gives the switch-off ids and logits.
    #[test]
    #[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_flags_gpu_stager -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_stager_is_invisible_in_ids_logits_and_reports() {
        use crate::expert_cache::{ExpertCache, Policy, Scope};
        use crate::glm5_tiers::PinnedUse;
        const REC: u64 = 9_474_048;
        let g = geo8();
        let t0 = std::time::Instant::now();
        let s = synth_model(&g, REC);
        eprintln!("glm5 stager: synthetic model written in {:.1} s", t0.elapsed().as_secs_f64());
        let (spec, records) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        assert_eq!((spec.bytes, records), (REC, 16 * 5));
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let n = 6;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let off = Switches::default();
        let flags = Switches { flags: true, lookahead: false, ..Switches::default() };
        let both = Switches { flags: true, lookahead: true, ..Switches::default() };
        let arm = |name, sw, graph, stager, zerocopy, lfu| Arm { name, sw, graph, stager, zerocopy, lfu };
        // each stager arm is compared with the synchronous arm of its cache rule (`base`)
        let arms = [
            (arm("off", off, false, false, false, false), None),
            (arm("flags", flags, false, false, false, false), Some(0)),
            (arm("flags+stager", flags, false, true, false, false), Some(0)),
            (arm("flags+lookahead+stager", both, false, true, false, false), Some(0)),
            (arm("graph+flags+stager", flags, true, true, false, false), Some(0)),
            (arm("off zerocopy", off, false, false, true, false), None),
            (arm("flags+stager zerocopy", flags, false, true, true, false), Some(5)),
            (arm("off lfu", off, false, false, false, true), None),
            (arm("flags+stager lfu", flags, false, true, false, true), Some(7)),
        ];
        let mut outs: Vec<(Generated, Vec<TokenReport>)> = Vec::new();
        let mut door: (Vec<i64>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        let mut gates = 0u64;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            for (a, _) in arms {
                run.set_graph(a.graph);
                run.set_switches(&mut cnq, a.sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                if a.lfu {
                    tiers.cache = ExpertCache::new(Policy::Lfu { decay: 0.5 }, Scope::PerLayer, g.layers - g.dense_prefix, g.experts, sizes.vram, sizes.pinned).unwrap();
                }
                if a.zerocopy {
                    tiers.set_pinned_use(PinnedUse { stay: true, cpu_lane: false }).unwrap();
                }
                tiers.set_stager(a.stager).unwrap();
                let mut reps: Vec<TokenReport> = Vec::new();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                let st = tiers.stager_stats();
                eprintln!("glm5 stager {}: ids {:?}, NVMe reads {}, stager {st:?}", a.name, gen.ids, tiers.nvme_reads);
                if let Some(st) = st {
                    assert_eq!(st.calls, (5 * (prompt.len() + n - 1)) as u64, "{}: one stager call per MoE layer per row", a.name);
                    assert_eq!(st.landed_reads, tiers.nvme_reads, "{}: every NVMe read carries a landed flag", a.name);
                    gates += st.gate_syncs;
                }
                tiers.free();
                outs.push((gen, reps));
            }
            // serve's door with flags + stager
            run.set_graph(false);
            run.set_switches(&mut cnq, flags);
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            tiers.set_stager(true).unwrap();
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            for i in 0..n {
                door.0.push(tok);
                door.1.push(cuda::dtoh(run.logits_dev(), g.vocab));
                if i + 1 < n {
                    tok = run.row(&mut cnq, &mut tiers, tok, prompt.len() + i, true).unwrap().unwrap();
                }
            }
            tiers.free();
            run.free();
        }
        drop(cnq);
        eprintln!("glm5 stager: gate syncs over all stager arms {gates}");
        let (g0, r0) = &outs[0];
        assert_eq!((g0.ids.len(), g0.logits.len(), r0.len()), (n, n, prompt.len() + n - 1));
        let finite = g0.logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        let bits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect::<Vec<_>>();
        for (i, (a, base)) in arms.iter().enumerate() {
            let (gx, rx) = &outs[i];
            // every arm, whatever its cache rule, gives the switch-off ids and logits (#188 / #190
            // hold the cache rules lossless; the stager must not change that)
            assert_eq!(gx.ids, g0.ids, "{}: ids", a.name);
            let diff = bits(&gx.logits, &g0.logits);
            assert!(diff.iter().all(|&d| d == 0), "{}: logits differ in bits per generated position {diff:?}", a.name);
            assert!(rx.iter().map(|r| r.moves.iter().map(|m| m.nvme_reads()).sum::<u64>()).sum::<u64>() > 0, "{}: the run must read records from NVMe", a.name);
            if let Some(b) = base {
                let (_, rb) = &outs[*b];
                assert_eq!(rx.iter().map(unclocked).collect::<Vec<_>>(), rb.iter().map(unclocked).collect::<Vec<_>>(), "{}: row reports against {}", a.name, arms[*b].0.name);
            }
        }
        assert_eq!(door.0, g0.ids, "row() with flags + stager: ids");
        let diff = bits(&door.1, &g0.logits);
        assert!(diff.iter().all(|&d| d == 0), "row() with flags + stager: logits differ in bits {diff:?}");
    }

    /// The sync-count harness of #149 path B (no assertion beyond the run): the synthetic 8-layer
    /// model with VRAM 3 + pinned 4 slots (records move every row), `GLM_STAGER_PROFILE_ARM` =
    /// `off` or a `+` list of `flags`, `stager`, `graph`, `prefetch`, `side`, `overlap` (e.g.
    /// `flags+stager`, `graph+flags+stager+prefetch+overlap`), 5 prompt rows and 3 warm
    /// decode rows through `row`, then `cuProfilerStart`, 8 decode rows, `cuProfilerStop`. Under
    /// `nsys profile -t cuda --capture-range=cudaProfilerApi` the API counts divided by 8 are the
    /// per-row counts of decode rows (5 MoE layers each).
    #[test]
    #[ignore = "measurement: nsys profile -t cuda --capture-range=cudaProfilerApi <test exe> glm5_flags_gpu_stager_profile_rows --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_stager_profile_rows() {
        const REC: u64 = 9_474_048;
        let arm = std::env::var("GLM_STAGER_PROFILE_ARM").unwrap_or_else(|_| "flags".into());
        // `off` or a `+` list of flags, stager, graph, prefetch, side, overlap
        let toks: Vec<&str> = arm.split('+').collect();
        for t in &toks {
            assert!(["off", "flags", "stager", "graph", "prefetch", "side", "overlap"].contains(t), "GLM_STAGER_PROFILE_ARM={arm:?}: off | a + list of flags, stager, graph, prefetch, side, overlap");
        }
        let on = |t: &str| toks.contains(&t);
        let (graph, stager) = (on("graph"), on("stager"));
        let sw = Switches { flags: on("flags"), lookahead: false, prefetch: on("prefetch"), pf_side: on("side"), overlap: on("overlap") };
        sw.check().unwrap();
        let g = geo8();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let (warm, rows) = (3usize, 8usize);
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + warm + rows, &mut |s| eprintln!("{s}"));
            run.set_graph(graph);
            run.set_switches(&mut cnq, sw);
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, g.topk).unwrap();
            tiers.set_stager(stager).unwrap();
            tiers.set_prefetch(sw.prefetch);
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            let mut pos = prompt.len();
            for _ in 0..warm {
                tok = run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap();
                pos += 1;
            }
            cuda::sync();
            let (reads0, moves0) = (tiers.nvme_reads, tiers.moves.iter().map(|m| m.h2d() + m.vram_to_stage + m.stage_to_vram + m.vram_to_pinned).sum::<u64>());
            let st0 = tiers.stager_stats().unwrap_or_default();
            crate::glm5_graph::profiler(true);
            let t0 = std::time::Instant::now();
            for _ in 0..rows {
                tok = run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap();
                pos += 1;
            }
            crate::glm5_graph::profiler(false);
            let copies = tiers.moves.iter().map(|m| m.h2d() + m.vram_to_stage + m.stage_to_vram + m.vram_to_pinned).sum::<u64>() - moves0;
            let st = tiers.stager_stats().unwrap_or_default();
            eprintln!("glm5 stager profile arm {arm}: prefetch {:?}, guess {:?}", tiers.prefetch_stats(), run.guess_stats());
            eprintln!(
                "glm5 stager profile arm {arm}: {rows} decode rows in {:.4} s (synthetic model, V 3 P 4); NVMe reads {}, record copies {copies}; stager calls {}, landed reads {}, gate syncs {}",
                t0.elapsed().as_secs_f64(),
                tiers.nvme_reads - reads0,
                st.calls - st0.calls,
                st.landed_reads - st0.landed_reads,
                st.gate_syncs - st0.gate_syncs
            );
            tiers.free();
            run.free();
        }
        drop(cnq);
    }

    // ---------------------------------------------------------------- CROW_GLM_PREFETCH

    #[test]
    fn the_prefetch_switches_parse_and_refuse_by_name() {
        let parse = |pairs: Vec<(&str, &str)>| Switches::parse(&|k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string()));
        let all = parse(vec![(ENV_FLAGS, "1"), (ENV_PREFETCH, "1"), (ENV_PREFETCH_SIDE, "1"), (ENV_SHARED_OVERLAP, "1")]);
        assert_eq!(all, Switches { flags: true, lookahead: false, prefetch: true, pf_side: true, overlap: true });
        assert_eq!(all.check(), Ok(()));
        assert_eq!(all.label(), "CROW_GLM_FLAGS on, CROW_GLM_LOOKAHEAD off, CROW_GLM_PREFETCH on, CROW_GLM_PREFETCH_SIDE on, CROW_GLM_SHARED_OVERLAP on");
        for v in ["0", "", "on", " 1"] {
            assert_eq!(parse(vec![(ENV_PREFETCH, v), (ENV_PREFETCH_SIDE, v), (ENV_SHARED_OVERLAP, v)]), Switches::default(), "{v:?}");
        }
        let e = parse(vec![(ENV_PREFETCH, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_PREFETCH=1 needs CROW_GLM_FLAGS=1"), "{e}");
        let e = parse(vec![(ENV_FLAGS, "1"), (ENV_PREFETCH_SIDE, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_PREFETCH_SIDE=1 needs CROW_GLM_PREFETCH=1"), "{e}");
        let e = parse(vec![(ENV_SHARED_OVERLAP, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_SHARED_OVERLAP=1 needs CROW_GLM_FLAGS=1"), "{e}");
        assert_eq!(Switches::default().check(), Ok(()));
    }

    #[test]
    fn a_hint_is_taken_once() {
        post_hint(Some(Hint { layer: 4, ids: vec![1, 2] }));
        assert_eq!(take_hint(), Some(Hint { layer: 4, ids: vec![1, 2] }));
        assert_eq!(take_hint(), None);
        post_hint(Some(Hint { layer: 5, ids: vec![3] }));
        post_hint(None);
        assert_eq!(take_hint(), None);
    }

    /// the router's selection on the host: sigmoid, + bias, K times the largest (ties to the
    /// lowest expert), the order of `glm5_router_sig_topk`
    fn host_select(logits: &[f32], bias: &[f32], k: usize) -> Vec<i32> {
        let mut c: Vec<f32> = logits.iter().zip(bias).map(|(&l, &b)| 1.0 / (1.0 + (-l).exp()) + b).collect();
        (0..k)
            .map(|_| {
                let mut best = 0;
                for e in 1..c.len() {
                    if c[e] > c[best] {
                        best = e;
                    }
                }
                c[best] = f32::NEG_INFINITY;
                best as i32
            })
            .collect()
    }

    /// The guess reaches the host with its flag: layer 1's router (random BF16 weights over
    /// E 288, a random bias) on a random input, guessed in a call of layer 0 on the compute
    /// stream and on the side stream; the published guess is the router kernel's top-8 of the
    /// guess's own logits (which are the BF16 GEMV of the host within 1e-4), handed over once as
    /// the hint for layer 1 and scored against layer 1's ids. A later publish without a guess
    /// hands over none (the device tag went back to -1).
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_the_guess_arrives_with_the_flag() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (e, k, h) = (g.experts, g.topk, g.hidden);
        let mut rng = Rng(0x0149_0E7C);
        let w_bf: Vec<u16> = (0..e * h).map(|_| gm::f32_to_bf16_rne(rng.sym() / (h as f32).sqrt())).collect();
        let w_f: Vec<f32> = w_bf.iter().map(|&b| f32::from_bits((b as u32) << 16)).collect();
        let bias: Vec<f32> = (0..e).map(|_| 0.05 * rng.sym()).collect();
        let x: Vec<f32> = (0..h).map(|_| rng.sym()).collect();
        let host_logits: Vec<f32> = (0..e).map(|r| (0..h).map(|c| w_f[r * h + c] as f64 * x[c] as f64).sum::<f64>() as f32).collect();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut km = cuda::compile(&crate::kernels::KernelGeo::flash_next().source());
            let kn = crate::kernels::Kernels::new(&km, false);
            let gk = crate::kernels::glm5_moe::Kernels::new();
            let mut k5 = Kernels::new(&g);
            let mut wr = cuda::upload_dev(&w_bf.iter().flat_map(|v| v.to_le_bytes()).collect::<Vec<u8>>());
            let mut wb = cuda::to_f32_dev(&bias);
            let mut xd = cuda::to_f32_dev(&x);
            let mut ids = cuda::to_i32_dev(&[7, 1, 2, 3, 4, 5, 6, 0]);
            for side in [false, true] {
                let mut r = Routed::new(&k5, k);
                r.pred = Some(Predict::with(vec![None, Some((wr, wb))], e, k, h, 2.5, 7.0, side));
                assert!(r.predicting());
                for call in 0..3 {
                    cuda::to_i32_into(ids, &[7, 1, 2, 3, 4, 5, 6, 0]);
                    r.predict_early(&kn, &gk, 0, xd);
                    r.predict(&kn, &gk, 0, xd);
                    r.publish(ids, k);
                    let got = r.wait_layer(0).unwrap();
                    assert_eq!(got, vec![7, 1, 2, 3, 4, 5, 6, 0], "side {side} call {call}: layer 0's own ids");
                    let logits = cuda::dtoh(r.pred.as_ref().unwrap().logits, e);
                    let dmax = logits.iter().zip(&host_logits).map(|(a, b)| (a - b).abs()).fold(0f32, f32::max);
                    assert!(dmax < 1e-4, "side {side}: the guess's logits are off the host GEMV by {dmax}");
                    let want = host_select(&logits, &bias, k);
                    assert_eq!(take_hint(), Some(Hint { layer: 1, ids: want.clone() }), "side {side} call {call}: the hint");
                    assert_eq!(take_hint(), None, "a hint is handed over once");
                    // layer 1's own call (its ids: the guess, 6 of them in another order, 2 others):
                    // no guess (layer 2 is no MoE layer here), scored against the guess
                    let mut own: Vec<i32> = want.iter().rev().copied().take(6).collect();
                    own.extend([-1, e as i32]);
                    cuda::to_i32_into(ids, &own);
                    r.predict_early(&kn, &gk, 1, xd);
                    r.predict(&kn, &gk, 1, xd);
                    r.publish(ids, k);
                    r.wait_layer(1).unwrap();
                    assert_eq!(take_hint(), None, "side {side} call {call}: a publish without a guess hands over none");
                }
                let gs = r.guess;
                assert_eq!((gs.launched, gs.side, gs.compared, gs.picks, gs.hits), (3, if side { 3 } else { 0 }, 3, 3 * k as u64, 3 * 6), "side {side}: {gs:?}");
                eprintln!("glm5 guess side {side}: {gs:?}");
                r.free();
            }
            for d in [&mut wr, &mut wb, &mut xd, &mut ids] {
                cuda::free_dev(d);
            }
            km.unload();
            k5.free();
        }
    }

    /// the synthetic 8-layer model's arms of the prefetch test
    #[derive(Clone, Copy, Debug)]
    struct PfArm {
        name: &'static str,
        sw: Switches,
        graph: bool,
        stager: bool,
        zerocopy: bool,
    }

    /// The prefetch, its side stream and the shared-expert overlap are invisible in the output.
    /// The synthetic 8-layer model (5 MoE layers, so a guess exists for 4 of them), a 5-id prompt
    /// and 6 greedy ids, VRAM 3 + pinned 4 slots, a fresh store per arm: every arm gives the
    /// switch-off ids and logits bit for bit, and the row reports of the arm without the prefetch
    /// (same cache rule, same stager) except the clock. Every prefetch arm handed guesses to the
    /// store and read records into it; over the arms, staged records came from the store. Also
    /// serve's door: `row` with flags + stager + prefetch + overlap gives the switch-off ids and
    /// logits.
    #[test]
    #[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_flags_gpu_prefetch -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_prefetch_is_invisible_in_ids_logits_and_reports() {
        use crate::glm5_tiers::PinnedUse;
        const REC: u64 = 9_474_048;
        let g = geo8();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let n = 6;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let sw = |flags, lookahead, prefetch, pf_side, overlap| Switches { flags, lookahead, prefetch, pf_side, overlap };
        let arm = |name, sw, graph, stager, zerocopy| PfArm { name, sw, graph, stager, zerocopy };
        let arms = [
            (arm("off", Switches::default(), false, false, false), None),
            (arm("flags", sw(true, false, false, false, false), false, false, false), None),
            (arm("flags+stager", sw(true, false, false, false, false), false, true, false), None),
            (arm("off zerocopy", Switches::default(), false, false, true), None),
            (arm("flags+prefetch", sw(true, false, true, false, false), false, false, false), Some(1)),
            (arm("flags+prefetch+side", sw(true, false, true, true, false), false, false, false), Some(1)),
            (arm("flags+overlap", sw(true, false, false, false, true), false, false, false), Some(1)),
            (arm("flags+stager+prefetch", sw(true, false, true, false, false), false, true, false), Some(2)),
            (arm("flags+stager+prefetch+side+overlap", sw(true, false, true, true, true), false, true, false), Some(2)),
            (arm("flags+lookahead+stager+prefetch+overlap", sw(true, true, true, false, true), false, true, false), Some(2)),
            (arm("graph+flags+stager+prefetch+side+overlap", sw(true, false, true, true, true), true, true, false), Some(2)),
            (arm("flags+stager+prefetch+overlap zerocopy", sw(true, false, true, false, true), false, true, true), Some(3)),
        ];
        let mut outs: Vec<(Generated, Vec<TokenReport>)> = Vec::new();
        let mut door: (Vec<i64>, Vec<Vec<f32>>) = (Vec::new(), Vec::new());
        let mut used = 0u64;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            for (a, _) in arms {
                run.set_graph(a.graph);
                run.set_switches(&mut cnq, a.sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                if a.zerocopy {
                    tiers.set_pinned_use(PinnedUse { stay: true, cpu_lane: false }).unwrap();
                }
                tiers.set_stager(a.stager).unwrap();
                tiers.set_prefetch(a.sw.prefetch);
                let mut reps: Vec<TokenReport> = Vec::new();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                let (ps, gs) = (tiers.prefetch_stats(), run.guess_stats());
                eprintln!(
                    "glm5 prefetch {}: ids {:?}, NVMe reads {}, prefetch {ps:?}, guess {gs:?} hit rate {:.3}",
                    a.name,
                    gen.ids,
                    tiers.nvme_reads,
                    gs.map_or(0.0, |g| g.hit_rate())
                );
                assert_eq!(ps.is_some(), a.sw.prefetch, "{}: the store follows the switch", a.name);
                assert_eq!(gs.is_some(), a.sw.prefetch, "{}: the guess follows the switch", a.name);
                if let (Some(ps), Some(gs)) = (ps, gs) {
                    // 4 of 5 MoE layers have a next MoE layer, every row (graph replays included)
                    assert_eq!(gs.compared, (4 * (prompt.len() + n - 1)) as u64, "{}: one guess per MoE layer with a next one, per row", a.name);
                    assert_eq!(ps.hints, gs.compared, "{}: every guess reached the store", a.name);
                    assert!(ps.issued > 0, "{}: the store read guessed records", a.name);
                    assert_eq!(gs.side > 0, a.sw.pf_side && !a.graph, "{}: side guesses", a.name);
                    used += ps.used;
                }
                tiers.free();
                outs.push((gen, reps));
            }
            // serve's door with flags + stager + prefetch + overlap
            run.set_graph(false);
            run.set_switches(&mut cnq, sw(true, false, true, false, true));
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            tiers.set_stager(true).unwrap();
            tiers.set_prefetch(true);
            run.kda_states().for_each(|k| k.reset());
            let mut tok = 0i64;
            for (pos, &p) in prompt.iter().enumerate() {
                if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                    tok = id;
                }
            }
            for i in 0..n {
                door.0.push(tok);
                door.1.push(cuda::dtoh(run.logits_dev(), g.vocab));
                if i + 1 < n {
                    tok = run.row(&mut cnq, &mut tiers, tok, prompt.len() + i, true).unwrap().unwrap();
                }
            }
            used += tiers.prefetch_stats().unwrap().used;
            tiers.free();
            run.free();
        }
        drop(cnq);
        eprintln!("glm5 prefetch: staged records taken from the store over all arms {used}");
        assert!(used > 0, "some staged record must come from the store");
        let (g0, r0) = &outs[0];
        assert_eq!((g0.ids.len(), g0.logits.len(), r0.len()), (n, n, prompt.len() + n - 1));
        let finite = g0.logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        let bits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect::<Vec<_>>();
        for (i, (a, base)) in arms.iter().enumerate() {
            let (gx, rx) = &outs[i];
            assert_eq!(gx.ids, g0.ids, "{}: ids", a.name);
            let diff = bits(&gx.logits, &g0.logits);
            assert!(diff.iter().all(|&d| d == 0), "{}: logits differ in bits per generated position {diff:?}", a.name);
            if let Some(b) = base {
                let (_, rb) = &outs[*b];
                assert_eq!(rx.iter().map(unclocked).collect::<Vec<_>>(), rb.iter().map(unclocked).collect::<Vec<_>>(), "{}: row reports against {}", a.name, arms[*b].0.name);
            }
        }
        assert_eq!(door.0, g0.ids, "row() with flags + stager + prefetch + overlap: ids");
        let diff = bits(&door.1, &g0.logits);
        assert!(diff.iter().all(|&d| d == 0), "row() with flags + stager + prefetch + overlap: logits differ in bits {diff:?}");
    }
}
