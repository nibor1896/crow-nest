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
//! - **`CROW_GLM_SIDE_NOJOIN=1`** (#202 S, with `CROW_GLM_PREFETCH_SIDE=1`; [`ENV_SIDE_NOJOIN`]):
//!   the side guess publishes itself. Behind its router the side stream copies the guess and its
//!   layer into its own area of the mapped block and raises its own sequence word
//!   (`glm5_guess_publish`); the compute stream publishes the router's ids alone (`glm5_publish`)
//!   without waiting for the side stream, and joins it only after the layer's experts are queued
//!   ([`Routed::join_late`]: the guess's input row must not be overwritten before the guess read
//!   it). The host reads the guess where the next layer's reads are issued ([`take_guess`]),
//!   spinning at most [`SIDE_WAIT`] for its word, else the guess is dropped (a hint only). Not
//!   under `CROW_GLM_CONTROLLER` (its request carries the guess) nor inside a graph capture (the
//!   guess then runs on the compute stream as without the switch).
//! - **`CROW_GLM_GUESS_TRIM=1`** (#202 N2, with `CROW_GLM_PREFETCH=1`; [`ENV_GUESS_TRIM`]): with
//!   the stager on the global arena a guessed read of layer l still queued when layer l's routing
//!   came without it is taken out of the reader queue before it reaches the drive
//!   (`NvmeSource::cancel`; its pinned slot goes back to the arena's free list); with
//!   [`ENV_GUESS_TRIM_K`] set a guess is also cut to its K best-scored ids (unset: no cut. The
//!   simulation's top-3, `docs/glm-tier-simulation.md` §9 on glm-pf2, doubled the demand reads on
//!   the real container: smoke 2026-10-10, 14.0 -> 28.5 per token).
//! - **`CROW_GLM_SHARED_OVERLAP=1`** (needs `CROW_GLM_FLAGS=1`; the reference's `GLM53_K_OVL`,
//!   `glm53/k_overlap.py`): in a decode call the shared expert is queued right behind the publish,
//!   before the host's hand-off, so the GPU computes it while the host waits for the flag and plans
//!   the layer; `experts` then skips it. Same kernels on the same input into the same buffer, so
//!   bit-identical (the reference moves it out of a fused kernel and is not). Not inside a
//!   `CROW_GLM_GRAPH` capture (the shared expert stays in the replayed segment).
//! - **`CROW_GLM_CONTROLLER=1`** (needs `CROW_GLM_FLAGS=1` and `CROW_GLM_STAGER=1`; [`Ctl`],
//!   [`Worker`]): the reference's nv2 controller (`kernels/nv2/nv2_shared.h`, `nv2_dev.cu`,
//!   `nv2_host.cpp`). A decode row's MoE layers hand their routing to a host controller thread
//!   through a mapped request ring; the thread serves each layer through the stager and the
//!   stager stream raises the reply word; the compute stream waits for it on the device (one
//!   bounded spinning thread, at most [`CTL_WAIT_NS`], under the WDDM TDR). The host enqueues a
//!   whole row without waiting for any routing. Same record bytes and tables as the stager path,
//!   so ids and logits are the same. Refused with `CROW_GLM_GRAPH`, `CROW_GLM_LOOKAHEAD` and MTP.
//!   The controller thread waits neither for a landing nor for the CPU: it plans, starts the NVMe
//!   reads, queues the reply and hands the CPU lane's job to the lane's own thread ([`DevLane`])
//!   (a read into a pinned slot a copy still uses goes to the stager's deferred-read thread), which
//!   waits per expert for its record to land and sums its experts into one row that
//!   [`Ctl::combine_lane`] adds behind the combine (#202 D-A, D-C; held to accuracy, not bits).
//!   [`CtlClock`] counts plan / reply / serve per layer.
//! - **`CROW_GLM_LA=1`** (needs `CROW_GLM_CONTROLLER=1`): the reference's decode lookahead
//!   (`glm53/k_lookahead.py`, `GLM53_LA`). After row k's head the next row is enqueued on the
//!   device's greedy id (its embedding gathered from a host-mapped table, as the reference's
//!   `prep_model`) before the host reads token k; the host then waits for token k only (an event
//!   behind the head's readback). `Glm5Run::generate` launches no row past the last one; serve's
//!   door (`Glm5Run::decode_la`) keeps the KDA states of the launched row's start and drops the row
//!   (states back, its MLA rows left to be overwritten) when the next id is not the one it ran on
//!   (end of turn, a forced or redrawn id) or anything else touches the sequence.

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
pub const ENV_CONTROLLER: &str = "CROW_GLM_CONTROLLER";
pub const ENV_LA: &str = "CROW_GLM_LA";
/// #202 S: `1` lets the side guess publish itself (see the module doc); unset or anything else off
pub const ENV_SIDE_NOJOIN: &str = "CROW_GLM_SIDE_NOJOIN";
/// #202 N2: `1` caps a guess by score and drops its stale queued reads; unset or anything else off
pub const ENV_GUESS_TRIM: &str = "CROW_GLM_GUESS_TRIM";
/// #202 N2: the guess's best-scored ids kept under [`ENV_GUESS_TRIM`], a whole number from 1;
/// unset/empty = every id (no cut); anything else refused by name
pub const ENV_GUESS_TRIM_K: &str = "CROW_GLM_GUESS_TRIM_K";
/// #202 S: how long the host spins for a self-published side guess before it drops it
pub const SIDE_WAIT: std::time::Duration = std::time::Duration::from_micros(500);

/// `CROW_GLM_SIDE_NOJOIN=1` (the repo's `CROW_*` rule: only `1` turns it on)
pub fn side_nojoin_on() -> bool {
    std::env::var(ENV_SIDE_NOJOIN).ok().as_deref() == Some("1")
}

/// #202 N2: the rank cap of [`ENV_GUESS_TRIM`] from the variables' values: `None` = off; `k`
/// unset/empty = `usize::MAX` (on, no cut); a `k` that is no whole number from 1 refused by name
pub fn guess_trim_from(on: Option<&str>, k: Option<&str>) -> Result<Option<usize>, String> {
    if on != Some("1") {
        return Ok(None);
    }
    match k.map(str::trim) {
        None | Some("") => Ok(Some(usize::MAX)),
        Some(v) => match v.parse::<usize>() {
            Ok(n) if n >= 1 => Ok(Some(n)),
            _ => Err(format!("{ENV_GUESS_TRIM_K}={v:?}: the guess's best-scored ids kept, a whole number from 1")),
        },
    }
}
/// #202 lanes: `1` lets the GPU experts, the CPU lane and the NVMe reads of a controlled decode
/// layer run side by side (see [`lanes2_on`]); unset or anything else keeps the former path
pub const ENV_LANES2: &str = "CROW_GLM_LANES2";

/// `CROW_GLM_LANES2=1` (the repo's `CROW_*` rule: only `1` turns it on). Under the controller's
/// early reply: the CPU lane takes only resident records (a record still landing goes to the
/// GPU's late pass, so the lane never waits for a landing or a write-back); the late pass holds
/// top-k slots ([`late_slots`]) and its spare slots read nothing (a null record: the expert
/// kernels return at once), so no layer waits for all its landings before its experts and a
/// layer without late experts pays no late GEMV; the device waits for the stager's moves word only
/// when the call queued a copy the experts read (else the moves are no wait of the experts at
/// all: the early experts read the mapped table row); the next layer's guessed reads go to the
/// NVMe pool's `Prefetch` queue (at most [`LANES2_PREFETCH_WORKERS`] workers on it) and a guessed
/// read the layer then needs moves to the `Demand` queue.
pub fn lanes2_on() -> bool {
    std::env::var(ENV_LANES2).ok().as_deref() == Some("1")
}

/// #202 lanes: workers of the NVMe piece pool that may read `Prefetch` pieces at once under
/// `CROW_GLM_LANES2` (the others stay free for a layer's demand reads;
/// `nvme_source::tests::bench_lanes_pool_under_spin`, 2026-10-10: demand 2 x 9.47 MB p50 2.48
/// ms at 4 of 16 against 2.85 ms uncapped)
pub const LANES2_PREFETCH_WORKERS: usize = 4;

/// the late pass's slots: [`LATE_SLOTS`], or top-k `k` under `CROW_GLM_LANES2`
pub fn late_slots(lanes2: bool, k: usize) -> usize {
    if lanes2 {
        k
    } else {
        LATE_SLOTS
    }
}

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
    /// `CROW_GLM_CONTROLLER`: routing through the request ring and the controller thread
    pub controller: bool,
    /// `CROW_GLM_LA`: the next decode row enqueued before the host reads the token
    pub la: bool,
}

impl Switches {
    /// `1` turns a switch on; unset or any other value leaves it off (the repo's `CROW_*` rule)
    pub fn parse(get: &dyn Fn(&str) -> Option<String>) -> Switches {
        let on = |k: &str| get(k).as_deref() == Some("1");
        Switches { flags: on(ENV_FLAGS), lookahead: on(ENV_LOOKAHEAD), prefetch: on(ENV_PREFETCH), pf_side: on(ENV_PREFETCH_SIDE), overlap: on(ENV_SHARED_OVERLAP), controller: on(ENV_CONTROLLER), la: on(ENV_LA) }
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
        if self.controller && !self.flags {
            return Err(format!("{ENV_CONTROLLER}=1 needs {ENV_FLAGS}=1 and CROW_GLM_STAGER=1: the controller serves the layers through the stager"));
        }
        if self.controller && self.lookahead {
            return Err(format!("{ENV_CONTROLLER}=1 and {ENV_LOOKAHEAD}=1: the controller's lookahead is {ENV_LA}=1"));
        }
        if self.la && !self.controller {
            return Err(format!("{ENV_LA}=1 needs {ENV_CONTROLLER}=1: a row is enqueued ahead only when no MoE layer waits for the host"));
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
        for (on, k) in [(self.prefetch, ENV_PREFETCH), (self.pf_side, ENV_PREFETCH_SIDE), (self.overlap, ENV_SHARED_OVERLAP), (self.controller, ENV_CONTROLLER), (self.la, ENV_LA)] {
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
// #202 S (CROW_GLM_SIDE_NOJOIN): the side stream's own publish of its guess: the k ids and the
// guessed layer into mapped host memory, then its own sequence word (one block)
extern "C" __global__ void glm5_guess_publish(const int* __restrict__ pids, int k, volatile int* host_pids, volatile int* host_tag, int tag, unsigned long long* ctr, volatile unsigned long long* flag)
{{
    for (int j = threadIdx.x; j < k; j += blockDim.x) host_pids[j] = pids[j];
    if (threadIdx.x == 0) *host_tag = tag;
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
// CROW_GLM_CONTROLLER: one request into ring entry q % {ring} ({entry} B: seq u64, layer i32,
// guess layer i32, ids i32 [16..], guess i32 [80..], routing weights f32 [144..]), its sequence
// number written last
extern "C" __global__ void glm5_ctl_publish(const int* __restrict__ ids, int n, unsigned char* ring, unsigned long long* ctr, int layer,
                                            const int* __restrict__ pids, int k, int* dtag, const float* __restrict__ wts)
{{
    __shared__ unsigned long long q;
    if (threadIdx.x == 0) {{
        q = *ctr + 1;
        *ctr = q;
    }}
    __syncthreads();
    unsigned char* e = ring + (q % {ring}) * {entry};
    volatile int* eids = (volatile int*) (e + 16);
    volatile int* eg = (volatile int*) (e + 80);
    volatile float* ew = (volatile float*) (e + 144);
    for (int j = threadIdx.x; j < n; j += blockDim.x) eids[j] = ids[j];
    if (wts)
        for (int j = threadIdx.x; j < n; j += blockDim.x) ew[j] = wts[j];
    if (pids)
        for (int j = threadIdx.x; j < k; j += blockDim.x) eg[j] = pids[j];
    __syncthreads();
    if (threadIdx.x == 0) {{
        ((volatile int*) (e + 8))[0] = layer;
        ((volatile int*) (e + 12))[0] = pids ? *dtag : -1;
        if (pids) *dtag = -1;
    }}
    __threadfence_system();
    __syncthreads();
    if (threadIdx.x == 0) {{
        *(volatile unsigned long long*) e = q;
        __threadfence_system();
    }}
}}
// CROW_GLM_CONTROLLER with the CPU lane (the template's nv_combine_k, sybil-solutions/
// glm53-flash-offload 6769b27 kernels/nv2/nv2_dev.cu#L452-L478): behind the layer's combine (the
// GPU computed the CPU's combos from a zeroed record, so they added 0), thread 0 of each block
// waits until the lane flag reaches the request's number, at most timeout_ns (then the number
// goes to the error word and y stays as it is); then, when the host wrote any CPU expert
// (flag[1] != 0), the CPU's one row (every CPU expert of the layer, weighted and summed on the
// host, host-mapped cacheable memory) is added to y. mode 0: volatile loads, 1: __ldcv, 2:
// __ldcv of float4 (h % 4 == 0, 16-byte aligned rows)
extern "C" __global__ void glm5_lane_add(float* y, const float* cpu, const unsigned long long* ctr, const volatile unsigned long long* flag, volatile unsigned long long* err, long long timeout_ns, long long h, int mode)
{{
    __shared__ int go;
    if (threadIdx.x == 0) {{
        const unsigned long long q = *ctr;
        unsigned long long t0, t;
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
        go = 1;
        while (flag[0] < q) {{
            asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
            if ((long long) (t - t0) > timeout_ns) {{
                if (*err == 0) *err = q;
                go = 0;
                break;
            }}
            __nanosleep(200);
        }}
        __threadfence_system();
        if (go && flag[1] == 0) go = 0;
    }}
    __syncthreads();
    if (!go) return;
    const long long stride = (long long) gridDim.x * blockDim.x;
    const long long i0 = blockIdx.x * (long long) blockDim.x + threadIdx.x;
    if (mode == 2) {{
        float4* y4 = (float4*) y;
        const float4* c4 = (const float4*) cpu;
        for (long long i = i0; i < h / 4; i += stride) {{
            const float4 v = __ldcv(c4 + i);
            float4 a = y4[i];
            a.x = __fadd_rn(a.x, v.x);
            a.y = __fadd_rn(a.y, v.y);
            a.z = __fadd_rn(a.z, v.z);
            a.w = __fadd_rn(a.w, v.w);
            y4[i] = a;
        }}
    }} else if (mode == 1) {{
        for (long long i = i0; i < h; i += stride) y[i] = __fadd_rn(y[i], __ldcv(cpu + i));
    }} else {{
        const volatile float* vc = cpu;
        for (long long i = i0; i < h; i += stride) y[i] = __fadd_rn(y[i], vc[i]);
    }}
}}
// CROW_GLM_CONTROLLER: the stream waits until the reply word reaches the last request's number,
// at most timeout_ns (then the number goes to the error word and the stream goes on)
extern "C" __global__ void glm5_ctl_wait(const unsigned long long* ctr, volatile unsigned long long* reply, volatile unsigned long long* err, long long timeout_ns)
{{
    const unsigned long long q = *ctr;
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (*reply < q) {{
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if ((long long) (t - t0) > timeout_ns) {{
            if (*err == 0) *err = q;
            break;
        }}
        __nanosleep(500);
    }}
    __threadfence_system();
}}
// #202 early reply: the late ring entry of request q ({late_w} u64): [0] q, [1] mode (1: the moves
// word before the experts, 2: every late item waited before the experts, 4: the spare late slots
// read a null record), [2] items, [3] a VRAM
// record the spare late slots read, [4] the moves word's address, [5] its value; from [8] four
// words per item: expert, record address, wait word address, wait value
__device__ bool glm5_late_spin(const volatile unsigned long long* w, unsigned long long v, unsigned long long q, volatile unsigned long long* err, long long timeout_ns)
{{
    unsigned long long t0, t;
    asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t0));
    while (*w < v) {{
        asm volatile("mov.u64 %0, %%globaltimer;" : "=l"(t));
        if ((long long) (t - t0) > timeout_ns) {{
            if (*err == 0) *err = q;
            return false;
        }}
        __nanosleep(200);
    }}
    return true;
}}
// before the layer's experts: the moves word (records the stager stream copies before any
// landing wait), and with mode 2 every late item (more late experts than the late pass holds)
extern "C" __global__ void glm5_ctl_pre(const unsigned long long* ctr, const unsigned long long* late, volatile unsigned long long* err, long long timeout_ns)
{{
    const unsigned long long q = *ctr;
    const volatile unsigned long long* en = (const volatile unsigned long long*) (late + (q % {ring}) * {late_w});
    if (en[0] != q) return;
    const unsigned long long mode = en[1];
    bool ok = true;
    if (mode & 1) ok = glm5_late_spin((const volatile unsigned long long*) en[4], en[5], q, err, timeout_ns);
    if (ok && (mode & 2)) {{
        const unsigned long long n = en[2];
        for (unsigned long long i = 0; i < n && ok; ++i) ok = glm5_late_spin((const volatile unsigned long long*) en[8 + 4 * i + 2], en[8 + 4 * i + 3], q, err, timeout_ns);
    }}
    __threadfence_system();
}}
// after the layer's other experts: each late expert waits on its own word (its landed flag),
// then goes into a late slot (its record, its combo); the spare slots read the VRAM record and
// write nowhere (idx -1)
extern "C" __global__ void glm5_ctl_late(const unsigned long long* ctr, const unsigned long long* late, const int* __restrict__ ids, int k, const unsigned long long* __restrict__ ptrs1,
                                         unsigned long long* ptrs2, int* idx, int slots, volatile unsigned long long* err, long long timeout_ns)
{{
    const unsigned long long q = *ctr;
    const volatile unsigned long long* en = (const volatile unsigned long long*) (late + (q % {ring}) * {late_w});
    const bool mine = en[0] == q;
    // mode 4 (CROW_GLM_LANES2): a spare slot reads nothing (a null record, the expert kernels
    // return at once)
    const unsigned long long spare = (mine && (en[1] & 4)) ? 0ull : (mine && en[3]) ? en[3] : ptrs1[0];
    int j = 0;
    if (mine && !(en[1] & 2)) {{
        const unsigned long long n = en[2];
        bool ok = true;
        for (unsigned long long i = 0; i < n; ++i) {{
            if (ok) ok = glm5_late_spin((const volatile unsigned long long*) en[8 + 4 * i + 2], en[8 + 4 * i + 3], q, err, timeout_ns);
            const int e = (int) en[8 + 4 * i];
            for (int c = 0; c < k && j < slots; ++c)
                if (ids[c] == e) {{
                    ptrs2[j] = en[8 + 4 * i + 1];
                    idx[j] = c;
                    ++j;
                    break;
                }}
        }}
    }}
    for (; j < slots; ++j) {{
        ptrs2[j] = spare;
        idx[j] = -1;
    }}
    __threadfence_system();
}}
// the late slots' outputs over their combos' rows of ye (blockIdx.y = slot)
extern "C" __global__ void glm5_ctl_scatter(float* ye, const float* __restrict__ ye2, const int* __restrict__ idx)
{{
    const int c = idx[blockIdx.y];
    if (c < 0) return;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < H; i += gridDim.x * blockDim.x) ye[(size_t) c * H + i] = ye2[(size_t) blockIdx.y * H + i];
}}
"#,
        h = g.hidden,
        v = g.vocab,
        s = g.hc_streams,
        ring = CTL_RING,
        entry = CTL_ENTRY,
        late_w = LATE_WORDS
    )
}

/// the switches' one NVRTC module (compiled only when a switch is on)
pub struct Kernels {
    module: cuda::Module,
    feed: CUfunction,
    publish: CUfunction,
    publish_pred: CUfunction,
    pred_tag: CUfunction,
    guess_publish: CUfunction,
    ctl_publish: CUfunction,
    ctl_wait: CUfunction,
    lane_add: CUfunction,
    ctl_pre: CUfunction,
    ctl_late: CUfunction,
    ctl_scatter: CUfunction,
}

impl Kernels {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(g: &Glm5Geo) -> Kernels {
        let module = cuda::compile(&src(g));
        Kernels {
            feed: module.get("glm5_feed"),
            publish: module.get("glm5_publish"),
            publish_pred: module.get("glm5_publish_pred"),
            pred_tag: module.get("glm5_pred_tag"),
            guess_publish: module.get("glm5_guess_publish"),
            ctl_publish: module.get("glm5_ctl_publish"),
            ctl_wait: module.get("glm5_ctl_wait"),
            lane_add: module.get("glm5_lane_add"),
            ctl_pre: module.get("glm5_ctl_pre"),
            ctl_late: module.get("glm5_ctl_late"),
            ctl_scatter: module.get("glm5_ctl_scatter"),
            module,
        }
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
    /// then the guess's `topk` ids (prefetch); at [`side_at`] the side guess's own publish (#202
    /// S): its sequence word (u64), its layer (i32), its ids (i32)
    host: Pinned,
    /// device: `[0]` the publish counter (u64), `[8]` the id count (i32)
    dev: Dev,
    n: usize,
    seq: u64,
    publish: CUfunction,
    publish_pred: CUfunction,
    pred_tag: CUfunction,
    guess_publish: CUfunction,
    /// #202 S: side guesses that published themselves, and the one the next publish leaves to
    /// the host: (guessed layer, its sequence number)
    side_seq: u64,
    side_pending: Option<(usize, u64)>,
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

/// #202 S: where the side guess's own publish starts in a block of `n` ids
fn side_at(n: usize) -> usize {
    (IDS_AT + (n + crate::kernels::glm5_moe::MAXK) * 4).next_multiple_of(64)
}

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
    /// #202 S (`CROW_GLM_SIDE_NOJOIN`, with the side stream): the device counter of the side
    /// guess's own publish (0 = off); the side guess in flight published itself (the router's
    /// publish does not wait for it); the compute stream still owes it the join behind the layer
    /// ([`Routed::join_late`])
    sctr: Dev,
    unjoined: bool,
    late: bool,
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
            sctr: 0,
            unjoined: false,
            late: false,
        }
    }

    /// #202 S: the side guess may publish itself (a counter for its sequence word); a no-op
    /// without the side stream
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn nojoin_on(&mut self) {
        if self.side.is_some() && self.sctr == 0 {
            self.sctr = cuda::alloc_zeroed(8);
        }
    }

    /// the router weights of `layer + 1` when that is a MoE layer
    fn next(&self, layer: usize) -> Option<(Dev, Dev)> {
        self.w.get(layer + 1).copied().flatten()
    }

    /// the three launches of the guess on the current stream
    unsafe fn launch(&self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, tagk: CUfunction, layer: usize, x: Dev, w: (Dev, Dev)) {
        self.launch_router(kn, gk, x, w);
        launch_v(tagk, 1, 1, 1, 32, &[self.tag, (layer + 1) as u64]);
    }

    /// the guess's router (GEMV + selection) on the current stream
    unsafe fn launch_router(&self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, x: Dev, (r, b): (Dev, Dev)) {
        launch_v(kn.f("gemv_bf16_b"), self.experts as u32, 1, 1, 256, &[r, x, self.logits, self.prm_kh]);
        launch_v(gk.router, 1, 1, 1, crate::kernels::glm5_moe::ROUTER_THREADS as u32, &[self.logits, b, self.ids, self.wts, self.prm_route, self.prm_f]);
    }

    unsafe fn free(&mut self) {
        for d in [&mut self.logits, &mut self.ids, &mut self.wts, &mut self.tag, &mut self.prm_kh, &mut self.prm_route, &mut self.prm_f] {
            cuda::free_dev(d);
        }
        if self.sctr != 0 {
            cuda::free_dev(&mut self.sctr);
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
        let host = Pinned::alloc((side_at(n) + 16 + crate::kernels::glm5_moe::MAXK * 4).next_multiple_of(4096));
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
            guess_publish: k.guess_publish,
            side_seq: 0,
            side_pending: None,
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
        let mut p = Predict::new(layers, moe, side);
        // #202 S: read once here, with the side stream only
        if side_nojoin_on() {
            p.nojoin_on();
        }
        self.pred = Some(p);
    }

    /// #202 S (`CROW_GLM_SIDE_NOJOIN`): the side guess can publish itself
    pub fn side_nojoin(&self) -> bool {
        self.pred.as_ref().is_some_and(|p| p.sctr != 0)
    }

    /// the guess is on
    pub fn predicting(&self) -> bool {
        self.pred.is_some()
    }

    /// `CROW_GLM_CONTROLLER`: the guess's device buffers `(ids, layer tag)` for the request ring
    pub fn guess_bufs(&self) -> Option<(Dev, Dev)> {
        self.pred.as_ref().map(|p| (p.ids, p.tag))
    }

    /// `CROW_GLM_PREFETCH_SIDE`: before layer `layer`'s router, launch the guess of layer + 1 on
    /// the side stream (it waits for `x` on the current stream). A no-op without the side stream,
    /// for a last MoE layer, and inside a graph capture (the guess then runs in [`Routed::predict`]).
    ///
    /// # Safety
    /// `x` holds the MoE input `[hidden]` f32, written by work queued before.
    pub unsafe fn predict_early(&mut self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, layer: usize, x: Dev) {
        self.predict_early_with(kn, gk, layer, x, false)
    }

    /// #202 S: [`Routed::predict_early`] whose side guess publishes itself when
    /// `CROW_GLM_SIDE_NOJOIN` is on ([`Routed::side_nojoin`]): the next [`Routed::predict`] then
    /// does not join it, the next [`Routed::publish`] publishes the router's ids alone and
    /// [`Routed::wait_layer`] leaves the guess for [`take_guess`]; the caller joins it with
    /// [`Routed::join_late`] once the layer's work that reads `x` is queued. Without the switch
    /// (or in a capture, or without a next MoE layer) as [`Routed::predict_early`].
    ///
    /// # Safety
    /// As [`Routed::predict_early`].
    pub unsafe fn predict_early_unjoined(&mut self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, layer: usize, x: Dev) {
        let on = self.side_nojoin();
        self.predict_early_with(kn, gk, layer, x, on)
    }

    unsafe fn predict_early_with(&mut self, kn: &crate::kernels::Kernels, gk: &crate::kernels::glm5_moe::Kernels, layer: usize, x: Dev, unjoined: bool) {
        let (tagk, gpub, host, sat) = (self.pred_tag, self.guess_publish, self.host.dev, side_at(self.n) as u64);
        let Some(p) = self.pred.as_mut() else { return };
        let (Some((s, e0, e1)), Some(w)) = (p.side, p.next(layer)) else { return };
        if crate::glm5_graph::capturing() {
            return;
        }
        let main = cuda::cur_stream();
        // #202 S: a guess not joined yet (an error path skipped its late join) is joined first
        if p.late {
            cuda::stream_wait_event(main, e1);
            p.late = false;
        }
        cuda::event_record(e0, main);
        cuda::stream_wait_event(s, e0);
        cuda::set_stream(s as u64);
        if unjoined {
            p.launch_router(kn, gk, x, w);
            launch_v(gpub, 1, 1, 1, 32, &[p.ids, p.topk as u64, host + sat + 16, host + sat + 8, (layer + 1) as u64, p.sctr, host + sat]);
        } else {
            p.launch(kn, gk, tagk, layer, x, w);
        }
        cuda::set_stream(main as u64);
        cuda::event_record(e1, s);
        p.joined = false;
        p.unjoined = unjoined;
        if unjoined {
            // submit the side stream now: nothing on the compute stream waits for it any more
            // (WDDM batches launches until a query or a sync)
            cuda::stream_query(s);
            self.side_seq += 1;
            self.side_pending = Some((layer + 1, self.side_seq));
        }
        self.guess.launched += 1;
        self.guess.side += 1;
    }

    /// #202 S: the compute stream's join of a self-published side guess, owed since
    /// [`Routed::predict`] skipped it: from here on the compute stream may overwrite the guess's
    /// input row. A no-op when none is owed.
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn join_late(&mut self) {
        let Some(p) = self.pred.as_mut() else { return };
        // (inside a graph capture no side guess is launched; one owed from before waits for the
        // next launch's own join)
        if p.late && !crate::glm5_graph::capturing() {
            let (_, _, e1) = p.side.expect("a side guess without its stream");
            cuda::stream_wait_event(cuda::cur_stream(), e1);
            p.late = false;
        }
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
        if !p.joined && p.unjoined {
            // #202 S: the side guess published itself; the join waits behind the layer
            p.joined = true;
            p.unjoined = false;
            p.late = true;
            return;
        }
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
        // #202 S: the call's guess published itself on the side stream: the ids alone
        match self.pred.as_ref().filter(|_| self.side_pending.is_none()) {
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
        // #202 S: a self-published guess is scored as the host read it ([`take_guess`])
        if let Some((gl, g)) = self.last_guess.take().or_else(take_side_read) {
            if gl == layer {
                self.guess.compared += 1;
                self.guess.picks += ids.len() as u64;
                self.guess.hits += ids.iter().filter(|e| g.contains(e)).count() as u64;
            }
        }
        if let Some((gl, seq)) = self.side_pending.take() {
            // SAFETY: the side area of this live block (`side_at`)
            let base = unsafe { (self.host.host as *const u8).add(side_at(self.n)) };
            post_side(SideHint { word: base as *const u64, tag: unsafe { base.add(8) } as *const i32, ids: unsafe { base.add(16) } as *const i32, k: p.topk, layer: gl, seq });
            return Ok(ids);
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

/// #202 S: a guess the side stream publishes itself: its sequence word, layer and ids in the
/// mapped block of a live [`Routed`], the layer it guesses and the sequence number to wait for
#[derive(Clone, Copy, Debug)]
pub struct SideHint {
    word: *const u64,
    tag: *const i32,
    ids: *const i32,
    k: usize,
    layer: usize,
    seq: u64,
}

thread_local! {
    static HINT: std::cell::RefCell<Option<Hint>> = const { std::cell::RefCell::new(None) };
    static SIDE: std::cell::Cell<Option<SideHint>> = const { std::cell::Cell::new(None) };
    static SIDE_READ: std::cell::RefCell<Option<(usize, Vec<i32>)>> = const { std::cell::RefCell::new(None) };
}

/// [`Routed::wait_layer`] leaves the call's guess here: the hand-off to the expert hook, the way
/// `glm5_moe::lane::post` hands the CPU lane its call
pub fn post_hint(h: Option<Hint>) {
    SIDE.with(|c| c.set(None));
    HINT.with(|c| *c.borrow_mut() = h);
}

/// #202 S: [`post_hint`] of a guess still on its way from the side stream
fn post_side(s: SideHint) {
    HINT.with(|c| *c.borrow_mut() = None);
    SIDE.with(|c| c.set(Some(s)));
}

/// #202 S: the self-published guess the host read last (for [`Routed::wait_layer`]'s score), once
fn take_side_read() -> Option<(usize, Vec<i32>)> {
    SIDE_READ.with(|c| c.borrow_mut().take())
}

/// the guess the last [`Routed::wait_layer`] left, once (a self-published side guess is read by
/// [`take_guess`] only)
pub fn take_hint() -> Option<Hint> {
    HINT.with(|c| c.borrow_mut().take())
}

/// #202 S: how the host found self-published side guesses
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SideRead {
    /// up at the first look, up after a spin, dropped after [`SIDE_WAIT`]
    pub ready: u64,
    pub waited: u64,
    pub late: u64,
}

/// #202 S: the self-published guess `s`, its word waited for at most `wait`
fn read_side(s: SideHint, wait: std::time::Duration, how: &mut SideRead) -> Option<Hint> {
    // SAFETY: the words of a live block the side stream's publish writes (`SideHint`)
    let word = || unsafe { std::ptr::read_volatile(s.word) };
    if word() >= s.seq {
        how.ready += 1;
    } else {
        let t0 = std::time::Instant::now();
        while word() < s.seq {
            if t0.elapsed() > wait {
                how.late += 1;
                return None;
            }
            std::hint::spin_loop();
        }
        how.waited += 1;
    }
    std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
    // SAFETY: as above; the kernel wrote them before the word (fence_system)
    let tag = unsafe { std::ptr::read_volatile(s.tag) };
    if tag != s.layer as i32 {
        return None;
    }
    let ids: Vec<i32> = (0..s.k).map(|i| unsafe { std::ptr::read_volatile(s.ids.add(i)) }).collect();
    Some(Hint { layer: s.layer, ids })
}

/// The guess for the expert hook's next reads: [`take_hint`], or #202 S the self-published side
/// guess (read once its word is up, at most [`SIDE_WAIT`]; [`Routed::wait_layer`] scores it as
/// read here); #202 N2 cut to the store's rank cap ([`ENV_GUESS_TRIM`]). The counts go to
/// `pf.stats`.
pub fn take_guess(pf: &mut Prefetch) -> Option<Hint> {
    let mut h = match take_hint() {
        Some(h) => h,
        None => {
            let s = SIDE.with(|c| c.take())?;
            let mut how = SideRead::default();
            let h = read_side(s, SIDE_WAIT, &mut how);
            pf.stats.side_ready += how.ready;
            pf.stats.side_waited += how.waited;
            pf.stats.side_late += how.late;
            let h = h?;
            SIDE_READ.with(|c| *c.borrow_mut() = Some((h.layer, h.ids.clone())));
            h
        }
    };
    if let Some(k) = pf.trim {
        if h.ids.len() > k {
            pf.stats.capped += (h.ids.len() - k) as u64;
            h.ids.truncate(k);
        }
    }
    Some(h)
}

/// `CU_STREAM_WAIT_VALUE_GEQ`
const WAIT_GEQ: u32 = 0;

/// `CROW_GLM_PREFETCH` with `CROW_NVME_POOL=1`: the store's reads join the piece pool's
/// `Prefetch` queue (every queued demand piece first, as nv2's two queues); the per-reader
/// backends have one FIFO each and keep `Demand`, the read of record, unless #202 N1
/// (`CROW_NVME_DEMAND_FIRST=1`) gives them the two queues too
pub fn prefetch_priority(src: &crate::nvme_source::NvmeSource) -> crate::nvme_source::ReadPriority {
    if src.pool().is_some() || src.demand_first() {
        crate::nvme_source::ReadPriority::Prefetch
    } else {
        crate::nvme_source::ReadPriority::Demand
    }
}

/// `CROW_GLM_PREFETCH`: the pinned bytes of the store (2 x `topk` records + the flag page), the
/// bytes `glm5_tiers::plan_for_rows` takes off the pinned budget
pub fn prefetch_pinned_bytes(topk: usize, record_bytes: u64) -> u64 {
    2 * topk as u64 * record_bytes + 4096
}

/// `CROW_GLM_LA`: the pinned bytes of the host-mapped embedding table (BF16 `[vocab][hidden]`),
/// taken off the pinned budget by `glm5_tiers::plan_for_rows`
pub fn feed_pinned_bytes(g: &Glm5Geo) -> u64 {
    ((g.vocab * g.hidden * 2) as u64).next_multiple_of(4096)
}

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
    /// #203: demands for a record whose read was still in flight (they joined it)
    pub joins: u64,
    /// the used records the call's NVMe read count still holds (the store: every used record,
    /// staged as an NVMe read; the RAM tier: none, a used record is a pinned hit)
    pub covered: u64,
    /// #202 N2 (`CROW_GLM_GUESS_TRIM`): guessed ids cut by the rank cap, and guessed reads taken
    /// out of the reader queue before the drive (their layer came without them)
    pub capped: u64,
    pub dropped: u64,
    /// #202 S (`CROW_GLM_SIDE_NOJOIN`): self-published side guesses up at the host's first look,
    /// up after a spin, dropped after [`SIDE_WAIT`]
    pub side_ready: u64,
    pub side_waited: u64,
    pub side_late: u64,
    /// #202 N1 (`CROW_NVME_DEMAND_FIRST`): the NVMe source's demand fetches that went ahead of a
    /// queued guess, and guessed records moved up to the demand queue (`NvmeSource::queue_counts`,
    /// since the source opened)
    pub overtakes: u64,
    pub promoted: u64,
}

impl PrefetchStats {
    /// `self - o` field by field (a later snapshot minus an earlier one)
    pub fn since(&self, o: &PrefetchStats) -> PrefetchStats {
        PrefetchStats {
            hints: self.hints - o.hints,
            resident: self.resident - o.resident,
            issued: self.issued - o.issued,
            bytes: self.bytes - o.bytes,
            used: self.used - o.used,
            wasted: self.wasted - o.wasted,
            joins: self.joins - o.joins,
            covered: self.covered - o.covered,
            capped: self.capped - o.capped,
            dropped: self.dropped - o.dropped,
            side_ready: self.side_ready - o.side_ready,
            side_waited: self.side_waited - o.side_waited,
            side_late: self.side_late - o.side_late,
            overtakes: self.overtakes - o.overtakes,
            promoted: self.promoted - o.promoted,
        }
    }

    pub fn add(&mut self, o: &PrefetchStats) {
        self.hints += o.hints;
        self.resident += o.resident;
        self.issued += o.issued;
        self.bytes += o.bytes;
        self.used += o.used;
        self.wasted += o.wasted;
        self.joins += o.joins;
        self.covered += o.covered;
        self.capped += o.capped;
        self.dropped += o.dropped;
        self.side_ready += o.side_ready;
        self.side_waited += o.side_waited;
        self.side_late += o.side_late;
        self.overtakes += o.overtakes;
        self.promoted += o.promoted;
    }
}

/// `CROW_GLM_PREFETCH`: pinned records read ahead from NVMe, two halves of `k` by layer parity,
/// each slot with a landed flag in mapped memory.
///
/// Why a half is free to be rewritten when the guess of layer m arrives (in `table_for` of m - 1):
/// its last reader is the staging copy of `table_for` of m - 2, queued before m - 2's experts;
/// the host is in `table_for` of m - 1 only after m - 1's router flag, so (in-order compute
/// stream, the stager's event before those experts) that copy has run. A read still in flight in
/// the half (a guess never used) is not waited for (#202 D4): its slot stays out of use until the
/// read completes (polled without blocking), so the controller never blocks on the store.
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
    /// a copy of `stats` after every call, for a report callback while the run holds the store
    shared: std::sync::Arc<std::sync::Mutex<PrefetchStats>>,
    /// #202 N2 ([`ENV_GUESS_TRIM`], read by [`Prefetch::new`]): on (the stale drop) with the
    /// guess's best-scored ids kept (`usize::MAX`: every id); `None` = off
    pub trim: Option<usize>,
}

impl Prefetch {
    /// `k` records per half of `rb` bytes each
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(k: usize, rb: u64) -> Prefetch {
        let n = 2 * k;
        assert!(n * 8 <= 4096, "the store's flags fit one page");
        let flags = Pinned::alloc(4096);
        std::ptr::write_bytes(flags.host as *mut u8, 0, flags.bytes);
        let p = Prefetch {
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
            shared: Default::default(),
            trim: guess_trim_from(std::env::var(ENV_GUESS_TRIM).ok().as_deref(), std::env::var(ENV_GUESS_TRIM_K).ok().as_deref()).unwrap_or_else(|e| panic!("{e}")),
        };
        assert_eq!(p.pinned_bytes(), prefetch_pinned_bytes(k, rb), "the prefetch store allocates what the plan books");
        p
    }

    /// the counters as of the last call, shared (`ExpertTiers::prefetch_clock`)
    pub fn clock(&self) -> std::sync::Arc<std::sync::Mutex<PrefetchStats>> {
        self.shared.clone()
    }

    pub fn publish_stats(&self) {
        if let Ok(mut g) = self.shared.lock() {
            *g = self.stats;
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

    /// Drop slot `i` without blocking (an unused record counts as wasted): a finished read
    /// reports; one still running keeps its ticket, and the slot stays out of use until
    /// [`Prefetch::poll`] sees it done.
    fn retire(&mut self, i: usize) -> Result<(), String> {
        if self.key[i].is_some() && !self.used[i] {
            self.stats.wasted += 1;
        }
        self.key[i] = None;
        self.used[i] = false;
        self.poll(i)
    }

    /// slot `i`'s read, if it finished, reports and frees the ticket (never blocks)
    fn poll(&mut self, i: usize) -> Result<(), String> {
        let r = match self.ticket[i].as_mut().and_then(|t| t.try_done()) {
            Some(r) => {
                self.ticket[i] = None;
                r.map(|_| ())
            }
            None => Ok(()),
        };
        r.map_err(|e| format!("{ENV_PREFETCH}: a prefetch read: {e}"))
    }

    /// drop slot `i`, waiting for its read (emptying the store)
    fn retire_wait(&mut self, src: &crate::nvme_source::NvmeSource, i: usize) -> Result<(), String> {
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
            let r = if matches!(self.key[i], Some((kl, ke)) if kl == l && keep.contains(&ke)) { Ok(()) } else { self.retire(i) };
            if err.is_ok() {
                err = r;
            }
        }
        err?;
        let mut n = 0;
        for &e in want {
            // a free slot: no record and no read still running into it
            let Some(i) = half.clone().find(|&i| self.key[i].is_none() && self.ticket[i].is_none()) else { break };
            let dst = crate::nvme_source::RecordDst { gu: self.host(i), dn: std::ptr::null_mut() };
            let rec = recs[e as usize];
            let bytes = rec.parts(&dst).iter().map(|p| p.1.len as u64).sum::<u64>();
            self.seq += 1;
            self.value[i] = self.seq;
            let flag = crate::nvme_source::Landed { flag: (self.flags.host as *mut u64).add(i), value: self.seq };
            // SAFETY: the destination is store slot i (`rb` bytes, nothing reads it until its
            // flag / ticket), the flag its own word, written only by this read until retired
            // under the piece pool behind every queued demand piece (`prefetch_priority`)
            self.ticket[i] = Some(src.fetch_prio(&[(rec, dst)], Some(&[flag]), prefetch_priority(src))?);
            self.key[i] = Some((l, e));
            self.bytes[i] = bytes;
            self.stats.issued += 1;
            self.stats.bytes += bytes;
            n += 1;
        }
        Ok(n)
    }

    /// the landed flag of slot `i` holds the value of its latest read (that read has landed)
    pub fn landed_value(&self, i: usize) -> bool {
        // SAFETY: slot i's word of the store's flag page, written by its reader
        unsafe { std::ptr::read_volatile((self.flags.host as *const u64).add(i)) == self.value[i] }
    }

    /// empty the store (every read reports; unused records count as wasted)
    pub fn forget(&mut self, src: &crate::nvme_source::NvmeSource) -> Result<(), String> {
        let mut err = Ok(());
        for i in 0..self.key.len() {
            let r = self.retire_wait(src, i);
            if err.is_ok() {
                err = r;
            }
        }
        self.publish_stats();
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
/// behind the demand reads (`records` by cache layer, `experts` per layer, `on_nvme(l, e)` true
/// when expert `e` of cache layer `l` is neither in VRAM nor pinned: the per-layer cache's tier or
/// the global arena's place; `first_moe` the first MoE decoder layer). A guess for another layer is
/// dropped.
///
/// # Safety
/// As [`Prefetch::issue`].
pub unsafe fn prefetch_hinted(
    pf: &mut Prefetch,
    src: &crate::nvme_source::NvmeSource,
    records: &[Vec<crate::nvme_source::ExpertRecord>],
    experts: usize,
    on_nvme: &dyn Fn(usize, u32) -> bool,
    first_moe: usize,
    layer: usize,
) -> Result<(), String> {
    let r = hinted(pf, src, records, experts, on_nvme, first_moe, layer);
    pf.publish_stats();
    r
}

unsafe fn hinted(
    pf: &mut Prefetch,
    src: &crate::nvme_source::NvmeSource,
    records: &[Vec<crate::nvme_source::ExpertRecord>],
    experts: usize,
    on_nvme: &dyn Fn(usize, u32) -> bool,
    first_moe: usize,
    layer: usize,
) -> Result<(), String> {
    let Some(h) = take_guess(pf) else { return Ok(()) };
    if h.layer != layer + 1 {
        return Ok(());
    }
    let Some(l) = h.layer.checked_sub(first_moe).filter(|&l| l < records.len()) else { return Ok(()) };
    pf.stats.hints += 1;
    let (mut keep, mut want) = (Vec::new(), Vec::new());
    for &e in &h.ids {
        if e < 0 || e as usize >= experts {
            continue;
        }
        let e = e as u32;
        if !on_nvme(l, e) {
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
    fn join(&mut self, e: u32, landed: u64) {
        self.inner.join(e, landed)
    }
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
                        self.pf.stats.covered += 1;
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

// ---------------------------------------------------------------- CROW_GLM_CONTROLLER

/// ring entries of the controller's request ring (the device publishes request s only after its
/// wait for request s - 1 passed, so the host is never more than one entry behind; 8 for slack)
pub const CTL_RING: usize = 8;
/// bytes per ring entry: `[0]` seq u64, `[8]` layer i32, `[12]` guess layer i32 (-1 none),
/// `[16..]` the ids (i32, at most `MAXK`), `[80..]` the guess (i32, `MAXK`), `[144..]` the
/// routing weights (f32, `MAXK`; the CPU lane weights its experts with them)
const CTL_ENTRY: usize = 256;
/// `glm5_lane_add`'s load of the CPU row: 2 = `__ldcv` of float4 (fastest in
/// `glm5_flags_gpu_lane_combine_bench`, 2026-10-10)
const LANE_ADD_MODE: u64 = 2;
/// a device wait of the controller gives up after this long (the WDDM TDR is 2 s; the kernel
/// stays well under it and the host then refuses the row by name)
pub const CTL_WAIT_NS: u64 = 1_000_000_000;
/// the host gives up on a request the device did not publish after this long
const CTL_HOST_TIMEOUT: std::time::Duration = std::time::Duration::from_secs(30);
/// requests the host may queue past the last one the controller answered ([`Ctl::pace`]): a
/// decode layer is a few dozen launches and milliseconds of device time against microseconds of
/// host enqueue, so 4 layers keep the device fed and the stream far below the launch queue that
/// filled inside the first controlled row of GLM-5.3-Flash (2026-10-10)
pub const CTL_AHEAD: u64 = 4;
/// #202: `1` keeps the former reply of the controller, written by the stager stream behind the
/// layer's moves and NVMe landings; unset, the controller answers at once (a host store into the
/// mapped reply word right after the plan, as the template's `serve`, sybil-solutions/
/// glm53-flash-offload 6769b27 `kernels/nv2/nv2_host.cpp#L484-L520`) and every expert not yet in
/// place waits on the device for its own word ([`EarlyReply`])
pub const ENV_CTL_LATE_REPLY: &str = "CROW_GLM_CTL_LATE_REPLY";
/// u64 words per entry of the late ring (see `glm5_ctl_pre`): a header of 8, then 4 per item
pub const LATE_WORDS: usize = 128;
/// most items one late entry holds
pub const LATE_MAX: usize = (LATE_WORDS - 8) / 4;
/// #202: experts per MoE layer the late pass computes after their own landing (more go through
/// `glm5_ctl_pre`'s wait before the layer's experts); the pass costs this many expert GEMV sets
/// per layer, also with no late expert
pub const LATE_SLOTS: usize = 2;

/// #202 early reply: one expert of a controlled call whose record is not in place at the reply:
/// the GPU computes it from `addr` in the late pass once the u64 at device address `word`
/// reached `value` (its landed flag, or the stager's done word)
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct LateItem {
    pub e: u32,
    pub addr: Dev,
    pub word: Dev,
    pub value: u64,
}

/// #202 early reply: the host side of one request's answer: the mapped reply word and the
/// request's entry of the late ring
#[derive(Clone, Copy, Debug)]
pub struct EarlyReply {
    pub reply: *mut u64,
    pub entry: *mut u64,
}

unsafe impl Send for EarlyReply {}

impl EarlyReply {
    /// Write request `q`'s late entry: `mode` (1: the device waits for the moves word `moves`
    /// before the experts; 2: for every item there too), `spare` the VRAM record the unused late
    /// slots read, `items` the late experts (at most [`LATE_MAX`]). The entry's sequence number
    /// goes last.
    ///
    /// # Safety
    /// `entry` is request `q`'s live entry; the device reads it only after the reply reached `q`.
    pub unsafe fn entry(&self, q: u64, mode: u64, moves: (Dev, u64), spare: Dev, items: &[LateItem]) {
        assert!(items.len() <= LATE_MAX, "glm5 controller: {} late experts, the entry holds {LATE_MAX}", items.len());
        let w = |i: usize, v: u64| std::ptr::write_volatile(self.entry.add(i), v);
        w(1, mode);
        w(2, items.len() as u64);
        w(3, spare);
        w(4, moves.0);
        w(5, moves.1);
        for (i, it) in items.iter().enumerate() {
            w(8 + 4 * i, it.e as u64);
            w(8 + 4 * i + 1, it.addr);
            w(8 + 4 * i + 2, it.word);
            w(8 + 4 * i + 3, it.value);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        w(0, q);
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }

    /// the reply: `q` into the mapped reply word (a host store, no stream); never lowers it (a
    /// failed job left `u64::MAX`)
    ///
    /// # Safety
    /// `reply` is the live mapped reply word.
    pub unsafe fn reply(&self, q: u64) {
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
        if std::ptr::read_volatile(self.reply) < q {
            std::ptr::write_volatile(self.reply, q);
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

/// `CROW_GLM_CONTROLLER`: the device side of the reference's nv2 controller
/// (`kernels/nv2/nv2_shared.h` `Req` ring, `nv2_dev.cu` `nv_pub_k` / the waits of `nv_step`).
/// Per MoE layer of a decode row the compute stream runs `glm5_ctl_publish` (the router's ids, the
/// next layer's guess and the layer into ring entry `q % CTL_RING`, then the entry's sequence
/// number `q`, counted on the device) and `glm5_ctl_wait` (one thread spins on the mapped reply
/// word until it reaches `q`, at most [`CTL_WAIT_NS`]; on a timeout it writes `q` to the error
/// word and lets the stream go on). The host's controller thread ([`Worker`] running the job of
/// `glm5_tiers`) reads the ring in order, serves the layer through the stager
/// (`ExpertTiers::table_reply`) and lets the stager stream write `q` into the reply word behind
/// the layer's moves (`cuStreamWriteValue64`). The experts then read the layer's fixed record
/// table. So the host enqueues a whole row without waiting for any routing.
pub struct Ctl {
    ring: Pinned,
    /// `[0]` the reply (u64, written by the stager stream or, on a failure, by the host),
    /// `[8]` the error word (u64, the device's timed-out sequence number)
    reply: Pinned,
    /// device u64: the publish counter
    ctr: Dev,
    publish: CUfunction,
    wait: CUfunction,
    add: CUfunction,
    k: usize,
    /// per decoder layer: the record table its experts read (0: no MoE layer); set per row
    pub tables: Vec<Dev>,
    /// the CPU lane under the controller (`ExpertTiers::dev_lane`); set per row, `None` = off
    pub lane: Option<DevLaneDev>,
    /// rows of the pass go through the controller (set around a controlled row)
    pub active: bool,
    /// requests handed to controller jobs so far (the host's ring position)
    host_seq: u64,
    /// requests queued on the compute stream so far ([`Ctl::publish`]; the device counter follows)
    queued: u64,
    /// raised by the host when a row it enqueued is abandoned: a job stops waiting for its requests
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// the guesses scored by the controller jobs
    pub score: std::sync::Arc<std::sync::Mutex<GuessScore>>,
    /// a job failed: the reply word was forced up; no further controlled row until the switches
    /// are set again
    pub poisoned: bool,
    /// #202: the controller answers at once ([`ENV_CTL_LATE_REPLY`] unset): the late ring
    /// (`CTL_RING` entries of [`LATE_WORDS`] u64, mapped) and the late pass's kernels
    pub early: bool,
    late: Pinned,
    pre: CUfunction,
    late_k: CUfunction,
    scatter: CUfunction,
    /// #202 lanes ([`ENV_LANES2`], read at `new`): the late pass's slots ([`late_slots`]); the
    /// controller job writes its entries to match ([`Ctl::set_lanes2`])
    pub late_slots: usize,
}

/// `CROW_GLM_CONTROLLER` with `CROW_GLM_PREFETCH`: the guesses as the controller saw them
#[derive(Clone, Debug, Default)]
pub struct GuessScore {
    pub stats: GuessStats,
    last: Option<(usize, Vec<i32>)>,
}

impl GuessScore {
    /// layer `layer`'s ids came: score the guess made for it; remember the request's own guess
    pub fn see(&mut self, layer: usize, ids: &[i32], guess: Option<&Hint>) {
        if let Some((gl, g)) = self.last.take() {
            if gl == layer {
                self.stats.compared += 1;
                self.stats.picks += ids.len() as u64;
                self.stats.hits += ids.iter().filter(|e| g.contains(e)).count() as u64;
            }
        }
        self.last = guess.map(|h| (h.layer, h.ids.clone()));
    }
}

/// `CROW_GLM_CONTROLLER`: the controller thread's and the CPU lane's clocks, summed over the
/// requests (the template's counters, sybil-solutions/glm53-flash-offload 6769b27
/// `kernels/nv2/nv2_host.cpp#L140-L149`: `plan_ns`, `reply_ns`, `ctl_busy_ns` / `serve_max_ns`,
/// `cpu_busy_ns`, `cpu_wait_land_ns`). Every span starts when the controller thread sees the
/// request in the ring.
#[derive(Debug, Default)]
pub struct CtlClock {
    requests: std::sync::atomic::AtomicU64,
    /// up to the plan's end: the cache's moves issued (NVMe reads started), the table row and the
    /// CPU lane's share decided, the next layer's guess read; before the reply is queued
    plan_ns: std::sync::atomic::AtomicU64,
    /// up to the reply word's write queued on the stager stream (and submitted)
    reply_ns: std::sync::atomic::AtomicU64,
    /// up to the controller thread being free for the next request
    serve_ns: std::sync::atomic::AtomicU64,
    serve_max_ns: std::sync::atomic::AtomicU64,
    /// CPU lane jobs, their time from start to the lane flag, and the part of it spent waiting
    /// for records to land
    cpu_jobs: std::sync::atomic::AtomicU64,
    cpu_busy_ns: std::sync::atomic::AtomicU64,
    cpu_wait_land_ns: std::sync::atomic::AtomicU64,
}

/// a snapshot of [`CtlClock`]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CtlTimes {
    pub requests: u64,
    pub plan_ns: u64,
    pub reply_ns: u64,
    pub serve_ns: u64,
    pub serve_max_ns: u64,
    pub cpu_jobs: u64,
    pub cpu_busy_ns: u64,
    pub cpu_wait_land_ns: u64,
}

impl CtlClock {
    /// one request served: `plan`, `reply`, `serve` from the request's sight
    pub fn served(&self, plan: std::time::Duration, reply: std::time::Duration, serve: std::time::Duration) {
        use std::sync::atomic::Ordering::Relaxed;
        self.requests.fetch_add(1, Relaxed);
        self.plan_ns.fetch_add(plan.as_nanos() as u64, Relaxed);
        self.reply_ns.fetch_add(reply.as_nanos() as u64, Relaxed);
        self.serve_ns.fetch_add(serve.as_nanos() as u64, Relaxed);
        self.serve_max_ns.fetch_max(serve.as_nanos() as u64, Relaxed);
    }

    /// one CPU lane job: `busy` from its start to its flag, `wait` of it waiting for landings
    pub fn cpu_job(&self, busy: std::time::Duration, wait: std::time::Duration) {
        use std::sync::atomic::Ordering::Relaxed;
        self.cpu_jobs.fetch_add(1, Relaxed);
        self.cpu_busy_ns.fetch_add(busy.as_nanos() as u64, Relaxed);
        self.cpu_wait_land_ns.fetch_add(wait.as_nanos() as u64, Relaxed);
    }

    pub fn read(&self) -> CtlTimes {
        use std::sync::atomic::Ordering::Relaxed;
        CtlTimes {
            requests: self.requests.load(Relaxed),
            plan_ns: self.plan_ns.load(Relaxed),
            reply_ns: self.reply_ns.load(Relaxed),
            serve_ns: self.serve_ns.load(Relaxed),
            serve_max_ns: self.serve_max_ns.load(Relaxed),
            cpu_jobs: self.cpu_jobs.load(Relaxed),
            cpu_busy_ns: self.cpu_busy_ns.load(Relaxed),
            cpu_wait_land_ns: self.cpu_wait_land_ns.load(Relaxed),
        }
    }
}

impl CtlTimes {
    /// `self - o` (a later snapshot minus an earlier one; the maximum is the later one's)
    pub fn since(&self, o: &CtlTimes) -> CtlTimes {
        CtlTimes {
            requests: self.requests - o.requests,
            plan_ns: self.plan_ns - o.plan_ns,
            reply_ns: self.reply_ns - o.reply_ns,
            serve_ns: self.serve_ns - o.serve_ns,
            serve_max_ns: self.serve_max_ns,
            cpu_jobs: self.cpu_jobs - o.cpu_jobs,
            cpu_busy_ns: self.cpu_busy_ns - o.cpu_busy_ns,
            cpu_wait_land_ns: self.cpu_wait_land_ns - o.cpu_wait_land_ns,
        }
    }

    /// microseconds per request (per controlled MoE layer) of a nanosecond sum
    pub fn per_layer_us(&self, ns: u64) -> f64 {
        ns as f64 / 1e3 / self.requests.max(1) as f64
    }
}

/// one request of the ring
#[derive(Clone, Debug, PartialEq)]
pub struct Request {
    pub seq: u64,
    pub layer: usize,
    pub ids: Vec<i32>,
    pub guess: Option<Hint>,
    /// the routing weights of `ids` (pick order; 0 when the publish had none)
    pub wts: Vec<f32>,
}

/// a raw pointer the controller thread may hold (the protocol, not the type, keeps it exclusive)
pub struct SendPtr<T>(pub *mut T);
unsafe impl<T> Send for SendPtr<T> {}

/// the host side of the ring for one controller job: `n` requests from `seq + 1` on
pub struct RingReader {
    ring: SendPtr<u8>,
    reply: SendPtr<u64>,
    /// the reply word's device address (the stager stream writes it)
    pub reply_dev: Dev,
    seq: u64,
    k: usize,
    cancel: std::sync::Arc<std::sync::atomic::AtomicBool>,
    /// #202: the late ring (host), `None` with [`ENV_CTL_LATE_REPLY`]=1
    late: Option<SendPtr<u64>>,
}

impl RingReader {
    /// #202: request `q`'s early answer (its late entry and the reply word); `None` when the
    /// controller keeps the former reply ([`ENV_CTL_LATE_REPLY`]=1)
    pub fn early(&self, q: u64) -> Option<EarlyReply> {
        // SAFETY: entry q % CTL_RING of the live mapped late ring
        self.late.as_ref().map(|l| EarlyReply { reply: self.reply.0, entry: unsafe { l.0.add((q as usize % CTL_RING) * LATE_WORDS) } })
    }

    /// the next request, once its entry's sequence number shows it; `Err` by name after
    /// [`CTL_HOST_TIMEOUT`] or when the host cancelled the row
    pub fn next(&mut self) -> Result<Request, String> {
        self.seq += 1;
        let s = self.seq;
        // SAFETY: entry s % CTL_RING of the live mapped ring
        let e = unsafe { self.ring.0.add((s as usize % CTL_RING) * CTL_ENTRY) };
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        while unsafe { std::ptr::read_volatile(e as *const u64) } != s {
            spins = spins.wrapping_add(1);
            if spins % 1024 == 0 {
                if self.cancel.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(format!("{ENV_CONTROLLER}: the row was abandoned before request {s}"));
                }
                if t0.elapsed() > CTL_HOST_TIMEOUT {
                    return Err(format!("{ENV_CONTROLLER}: request {s} did not arrive in {} s", CTL_HOST_TIMEOUT.as_secs()));
                }
            }
            std::hint::spin_loop();
        }
        std::sync::atomic::fence(std::sync::atomic::Ordering::Acquire);
        // SAFETY: the entry's fields, written by the publish before its sequence number
        unsafe {
            let rd = |at: usize| std::ptr::read_volatile(e.add(at) as *const i32);
            let layer = rd(8);
            let tag = rd(12);
            let ids = (0..self.k).map(|i| rd(16 + 4 * i)).collect();
            let guess = (tag >= 0).then(|| Hint { layer: tag as usize, ids: (0..self.k).map(|i| rd(80 + 4 * i)).collect() });
            let wts = (0..self.k).map(|i| std::ptr::read_volatile(e.add(144 + 4 * i) as *const f32)).collect();
            Ok(Request { seq: s, layer: layer as usize, ids, guess, wts })
        }
    }

    /// after a failure: every wait of the device passes from now on (the reply word to u64::MAX;
    /// the caller has drained the stager stream, so no queued reply write lowers it again)
    pub fn release_all(&self) {
        // SAFETY: the live mapped reply word
        unsafe { std::ptr::write_volatile(self.reply.0, u64::MAX) };
        std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
    }
}

impl Ctl {
    /// `k` ids per request, `layers` decoder layers
    ///
    /// # Safety
    /// A CUDA context is current; `kn` outlives this controller.
    pub unsafe fn new(kn: &Kernels, k: usize, layers: usize) -> Ctl {
        assert!(k <= crate::kernels::glm5_moe::MAXK);
        let ring = Pinned::alloc(CTL_RING * CTL_ENTRY);
        std::ptr::write_bytes(ring.host as *mut u8, 0, ring.bytes);
        let reply = Pinned::alloc(4096);
        std::ptr::write_bytes(reply.host as *mut u8, 0, reply.bytes);
        let ctr = cuda::alloc_named("glm5 controller counter", 8);
        cuda::memset_zero_sync(ctr, 8);
        let late = Pinned::alloc(CTL_RING * LATE_WORDS * 8);
        std::ptr::write_bytes(late.host as *mut u8, 0, late.bytes);
        Ctl {
            late_slots: late_slots(lanes2_on(), k),
            early: std::env::var(ENV_CTL_LATE_REPLY).ok().as_deref() != Some("1"),
            late,
            pre: kn.ctl_pre,
            late_k: kn.ctl_late,
            scatter: kn.ctl_scatter,
            ring,
            reply,
            ctr,
            publish: kn.ctl_publish,
            wait: kn.ctl_wait,
            add: kn.lane_add,
            k,
            tables: vec![0; layers],
            lane: None,
            active: false,
            host_seq: 0,
            queued: 0,
            cancel: Default::default(),
            score: Default::default(),
            poisoned: false,
        }
    }

    /// Hold the host before it queues request `queued + 1` until the controller has answered
    /// request `queued + 1 - CTL_AHEAD` (the reply word, read on the host: no driver call). So
    /// the compute stream never holds more than [`CTL_AHEAD`] layers of launches behind a device
    /// wait. Unpaced, a row of the real model (42 MoE layers) went into the stream whole behind
    /// the first wait, `cuLaunchKernel` blocked on the full launch queue, the controller thread's
    /// next driver call (a stager copy) waited behind that blocked launch, and the device's wait
    /// gave up after [`CTL_WAIT_NS`]: the experts read a stale table (CUDA_ERROR_ILLEGAL_ADDRESS,
    /// 2026-10-10). After a failed job the reply word is `u64::MAX`, so this passes. `Err` by
    /// name when a device wait timed out, the row was cancelled or [`CTL_HOST_TIMEOUT`] passed.
    pub fn pace(&self) -> Result<(), String> {
        let want = (self.queued + 1).saturating_sub(CTL_AHEAD);
        if want == 0 {
            return Ok(());
        }
        let t0 = std::time::Instant::now();
        let mut spins = 0u32;
        // SAFETY: the live mapped reply word
        while unsafe { std::ptr::read_volatile(self.reply.host as *const u64) } < want {
            spins = spins.wrapping_add(1);
            if spins % 64 == 0 {
                let q = self.timed_out();
                if q != 0 {
                    return Err(format!("{ENV_CONTROLLER}: the device's wait for request {q} gave up after {} ms", CTL_WAIT_NS / 1_000_000));
                }
                if self.cancel.load(std::sync::atomic::Ordering::Acquire) {
                    return Err(format!("{ENV_CONTROLLER}: the row was abandoned before request {want} was answered"));
                }
                if t0.elapsed() > CTL_HOST_TIMEOUT {
                    return Err(format!("{ENV_CONTROLLER}: request {want} was not answered in {} s", CTL_HOST_TIMEOUT.as_secs()));
                }
                std::thread::yield_now();
            }
            std::hint::spin_loop();
        }
        Ok(())
    }

    /// Queue the request of decoder layer `layer`: its `n` ids at `ids`, their routing weights at
    /// `wts` (0: none) and, with the guess on, the guess (`(ids, tag)` of
    /// [`Routed::guess_bufs`]), into the ring.
    ///
    /// # Safety
    /// `ids` holds `n` i32 (and `wts`, when not 0, `n` f32) written by work queued before; `n` is
    /// this controller's `k`.
    pub unsafe fn publish(&mut self, layer: usize, ids: Dev, n: usize, guess: Option<(Dev, Dev)>, wts: Dev) {
        assert_eq!(n, self.k, "glm5 controller: a request of {n} ids on a ring of {}", self.k);
        let (pids, dtag) = guess.unwrap_or((0, 0));
        launch_v(self.publish, 1, 1, 1, 32, &[ids, n as u64, self.ring.dev, self.ctr, layer as u64, pids, self.k as u64, dtag, wts]);
        self.queued += 1;
        // WDDM: hand the request to the GPU now, the controller thread waits for it
        cuda::stream_query(cuda::cur_stream());
    }

    /// Queue the device wait for the reply to the last published request.
    ///
    /// # Safety
    /// [`Ctl::publish`] was queued before.
    pub unsafe fn wait_reply(&self) {
        launch_v(self.wait, 1, 1, 1, 1, &[self.ctr, self.reply.dev, self.reply.dev + 8, CTL_WAIT_NS]);
        cuda::stream_query(cuda::cur_stream());
    }

    /// The CPU lane under the controller, queued behind the layer's combine (one kernel, the
    /// template's `nv_combine_k`): the device waits (bounded as [`Ctl::wait_reply`]) until the
    /// lane flag reaches the last request's number, then adds the CPU's one row (all its experts,
    /// weighted and summed on the host) to the layer's output `y` (`h` f32).
    ///
    /// # Safety
    /// [`Ctl::publish`] was queued before; `y` is the combined output of the layer's call.
    pub unsafe fn combine_lane(&self, dl: &DevLaneDev, y: Dev, h: usize) {
        lane_add(self.add, y, dl, self.ctr, self.reply.dev + 8, CTL_WAIT_NS, h, LANE_ADD_MODE);
        cuda::stream_query(cuda::cur_stream());
    }

    /// the reader of the next `n` requests (the host's ring position moves past them)
    pub fn reader(&mut self, n: usize) -> RingReader {
        self.cancel.store(false, std::sync::atomic::Ordering::Release);
        let r = RingReader {
            ring: SendPtr(self.ring.host as *mut u8),
            reply: SendPtr(self.reply.host as *mut u64),
            reply_dev: self.reply.dev,
            seq: self.host_seq,
            k: self.k,
            cancel: self.cancel.clone(),
            late: self.early.then(|| SendPtr(self.late.host as *mut u64)),
        };
        self.host_seq += n as u64;
        r
    }

    /// #202 early reply: the experts of the last published request's layer behind
    /// [`Ctl::wait_reply`] (the table `table` read where the controller wrote it): `glm5_ctl_pre`
    /// (the moves word; with more late experts than the late slots every late one), the experts
    /// (a late one reads a zeroed VRAM record there), the shared one, then the late pass:
    /// `glm5_ctl_late` waits for each late expert's own word and puts it into a late slot, the
    /// expert GEMVs over the late slots ([`Ctl::late_slots`]), `glm5_ctl_scatter` over their combos' rows, the
    /// combine. One row (`t = 1`).
    ///
    /// # Safety
    /// [`Ctl::publish`] and [`Ctl::wait_reply`] were queued for this layer; `p` routed `x`.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_early(
        &self,
        p: &crate::glm5_moe::GpuMoePlan,
        kn: &crate::kernels::Kernels,
        mk: &crate::kernels::mul1::Kernels,
        gk: &crate::kernels::glm5_moe::Kernels,
        w: &crate::glm5_moe::GpuMoeWeights,
        table: Dev,
        x: Dev,
        y: Dev,
    ) {
        self.experts_early_marked(p, kn, mk, gk, w, table, x, y, None);
    }

    /// [`Ctl::experts_early`] with `mark` (tests) recorded on the stream after the layer's other
    /// experts and the shared one, right before the late pass's wait
    ///
    /// # Safety
    /// As [`Ctl::experts_early`]; `mark` is a live event.
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn experts_early_marked(
        &self,
        p: &crate::glm5_moe::GpuMoePlan,
        kn: &crate::kernels::Kernels,
        mk: &crate::kernels::mul1::Kernels,
        gk: &crate::kernels::glm5_moe::Kernels,
        w: &crate::glm5_moe::GpuMoeWeights,
        table: Dev,
        x: Dev,
        y: Dev,
        mark: Option<sys::CUevent>,
    ) {
        let err = self.reply.dev + 8;
        launch_v(self.pre, 1, 1, 1, 1, &[self.ctr, self.late.dev, err, CTL_WAIT_NS]);
        let (ctr, late, k, fill, scatter, slots) = (self.ctr, self.late.dev, self.k, self.late_k, self.scatter, self.late_slots);
        let h = p.geo.hidden;
        p.experts_late(
            kn,
            mk,
            gk,
            w,
            table,
            x,
            y,
            slots,
            &mut |ids, ptrs1, ptrs2, idx| {
                if let Some(m) = mark {
                    cuda::event_record(m, cuda::cur_stream());
                }
                launch_v(fill, 1, 1, 1, 1, &[ctr, late, ids, k as u64, ptrs1, ptrs2, idx, slots as u64, err, CTL_WAIT_NS])
            },
            &mut |ye, ye2, idx| launch_v(scatter, h.div_ceil(256).min(64) as u32, slots as u32, 1, 256, &[ye, ye2, idx]),
        );
        cuda::stream_query(cuda::cur_stream());
    }

    /// #202 lanes: the late pass of [`late_slots`] (`CROW_GLM_LANES2` on or off); the tiers
    /// serving this controller must be set alike (`ExpertTiers::set_lanes2`)
    pub fn set_lanes2(&mut self, on: bool) {
        self.late_slots = late_slots(on, self.k);
    }

    /// no request was handed to a job yet (a controller made by the last `set_switches`)
    pub fn fresh(&self) -> bool {
        self.host_seq == 0
    }

    /// a row the host enqueued only in part: its job stops waiting
    pub fn cancel(&self) {
        self.cancel.store(true, std::sync::atomic::Ordering::Release);
    }

    /// the sequence number of the first device wait that timed out (0: none)
    pub fn timed_out(&self) -> u64 {
        // SAFETY: the live mapped error word
        unsafe { std::ptr::read_volatile((self.reply.host as *const u8).add(8) as *const u64) }
    }

    /// # Safety
    /// No launch of this controller is pending.
    pub unsafe fn free(&mut self) {
        self.ring.free();
        self.reply.free();
        self.late.free();
        cuda::free_dev(&mut self.ctr);
    }
}

/// `glm5_lane_add` (mode 0 volatile, 1 `__ldcv`, 2 `__ldcv` of float4) on `y` (`h` f32) with the
/// lane's row and flag, its wait bounded by `timeout_ns` (then the number goes to `err`)
///
/// # Safety
/// `y` holds `h` f32; `ctr` the device's publish counter; `err` a mapped u64.
#[allow(clippy::too_many_arguments)]
pub unsafe fn lane_add(f: CUfunction, y: Dev, dl: &DevLaneDev, ctr: Dev, err: Dev, timeout_ns: u64, h: usize, mode: u64) {
    let n = if mode == 2 { h / 4 } else { h };
    launch_v(f, n.div_ceil(256).max(1) as u32, 1, 1, 256, &[y, dl.y, ctr, dl.flag, err, timeout_ns, h as u64, mode]);
}

/// `CROW_GLM_CONTROLLER` with `CROW_GLM_CPU_LANE`: the device addresses a controlled row's MoE
/// layer needs (copied into [`Ctl::lane`] per row)
#[derive(Clone, Copy, Debug)]
pub struct DevLaneDev {
    /// host address of the x row (the compute stream copies the MoE input there before publish)
    pub x_host: u64,
    /// mapped: the lane flag (u64) and the CPU expert count of its job (u64, behind it), the CPU's
    /// summed row (`[h]` f32)
    pub flag: Dev,
    pub y: Dev,
}

impl DevLaneDev {
    /// queue the MoE input row `x` (`h` f32) to the host before the layer's request is published
    ///
    /// # Safety
    /// `x` holds the row; nothing reads the host row until the request is seen.
    pub unsafe fn queue_x(&self, x: Dev, h: usize) {
        cuda::ck(sys::cuMemcpyDtoHAsync_v2(self.x_host as *mut std::ffi::c_void, x, h * 4, cuda::cur_stream()));
    }
}

const DL_X: usize = 4096;

/// What a CPU pick waits for before the CPU reads its pinned record (the template's per-expert
/// `land_cv` wait in `cpu_loop`, nv2_host.cpp#L553-L569): an NVMe read's landed flag (host word,
/// raised by the reader after the record's last byte) and CUDA events of copies still writing
/// the slot, polled with `cuEventQuery` (the template polls its write-backs the same way,
/// #L382-L392).
#[derive(Clone, Debug, Default)]
pub struct Ready {
    pub flag: Option<(*const u64, u64)>,
    pub events: Vec<u64>,
}

impl Ready {
    /// every condition met (an event in error counts as done: its stream fails loudly elsewhere)
    ///
    /// # Safety
    /// The flag word and the events are alive.
    unsafe fn landed(&self) -> bool {
        if let Some((w, v)) = self.flag {
            if std::ptr::read_volatile(w) < v {
                return false;
            }
        }
        self.events.iter().all(|&e| sys::cuEventQuery(e as sys::CUevent) != sys::cudaError_enum::CUDA_ERROR_NOT_READY)
    }
}

/// one CPU expert of a layer's call: its pinned record, its routing weight, what it waits for
#[derive(Clone, Debug)]
pub struct LanePick {
    pub rec: *const u8,
    pub w: f32,
    pub ready: Ready,
}

/// one layer's CPU job: request `q`, its experts (pick order)
#[derive(Debug)]
pub struct LaneJob {
    pub q: u64,
    pub picks: Vec<LanePick>,
}
// SAFETY: the records, flag words and events outlive the job (the controller's protocol: the next
// request, whose call may move them, comes after the device passed this job's flag)
unsafe impl Send for LaneJob {}

/// what the lane's worker thread holds
struct LaneCtx {
    base: SendPtr<u8>,
    h: usize,
    geo: crate::glm5_moe::MoeGeo,
    clock: std::sync::Arc<crate::glm5_moe::lane::Clock>,
    ctl: std::sync::Arc<CtlClock>,
    done: std::sync::Arc<std::sync::atomic::AtomicU64>,
    /// the worker's thread id (tests: the lane runs off the controller thread)
    thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::ThreadId>>>,
}

/// the flag word of a lane buffer as an atomic (host side; the device only reads it)
///
/// # Safety
/// `base` is the live, 8-byte aligned lane buffer.
unsafe fn lane_flag<'a>(base: *mut u8) -> &'a std::sync::atomic::AtomicU64 {
    std::sync::atomic::AtomicU64::from_ptr(base as *mut u64)
}

impl LaneCtx {
    /// One job: the picks whose records have landed are computed together (`cpu_mul1::experts_ffn`
    /// in one pool run, clamped SwiGLU), weighted and summed into one row; then the next landed
    /// ones, until none is left (resident records first, records still landing after: ProMoE
    /// section 4.3). The row into the mapped buffer, the expert count, then the flag to `q` (never
    /// lowered: a released flag stays released).
    ///
    /// # Safety
    /// The job's records are readable once landed; the x row holds the request's MoE input.
    unsafe fn run(&self, job: LaneJob) {
        let (h, base) = (self.h, self.base.0);
        let t0 = std::time::Instant::now();
        let mut wait = std::time::Duration::ZERO;
        let mut acc = vec![0f32; h];
        let n = job.picks.len();
        let mut left = job.picks;
        let xs = std::slice::from_raw_parts(base.add(DL_X) as *const f32, h);
        let rb = self.geo.record.bytes as usize;
        let limit = self.geo.swiglu_limit;
        let mut ok = true;
        while !left.is_empty() {
            let (ready, rest): (Vec<LanePick>, Vec<LanePick>) = left.into_iter().partition(|p| p.ready.landed());
            left = rest;
            if ready.is_empty() {
                let tw = std::time::Instant::now();
                let mut spins = 0u32;
                while !left.iter().any(|p| p.ready.landed()) {
                    spins = spins.wrapping_add(1);
                    if spins % 64 == 0 {
                        if tw.elapsed() > CTL_HOST_TIMEOUT {
                            eprintln!("{ENV_CONTROLLER}: CPU lane: request {}: {} records did not land in {} s", job.q, left.len(), CTL_HOST_TIMEOUT.as_secs());
                            ok = false;
                            break;
                        }
                        std::thread::yield_now();
                    }
                    std::hint::spin_loop();
                }
                wait += tw.elapsed();
                if !ok {
                    break;
                }
                continue;
            }
            let tc = std::time::Instant::now();
            let es: Vec<crate::cpu_mul1::Mul1Expert> = ready
                .iter()
                .map(|p| {
                    crate::cpu_mul1::Mul1Expert::from_record(std::slice::from_raw_parts(p.rec, rb), h, self.geo.expert_inter, self.geo.bitrate)
                        .unwrap_or_else(|e| panic!("glm5 controller CPU lane: {e}"))
                })
                .collect();
            let mut ys = vec![0f32; ready.len() * h];
            crate::cpu_mul1::experts_ffn(&es, xs, &mut ys, &move |a, b| crate::glm5_moe::swiglu_clamp(a, b, limit), crate::glm5_moe::lane::threads(), crate::cpu_mul1::Path::Auto);
            for (p, y) in ready.iter().zip(ys.chunks_exact(h)) {
                for (a, v) in acc.iter_mut().zip(y) {
                    *a += p.w * v;
                }
            }
            self.clock.add(tc.elapsed(), ready.len());
        }
        if ok {
            std::ptr::copy_nonoverlapping(acc.as_ptr(), base.add(DL_X + h * 4) as *mut f32, h);
            std::ptr::write_volatile(base.add(8) as *mut u64, n as u64);
            std::sync::atomic::fence(std::sync::atomic::Ordering::SeqCst);
            lane_flag(base).fetch_max(job.q, std::sync::atomic::Ordering::SeqCst);
        }
        self.ctl.cpu_job(t0.elapsed(), wait);
    }
}

/// `CROW_GLM_CONTROLLER` with `CROW_GLM_CPU_LANE`: the CPU lane of a controlled row (the
/// template's nv2 CPU job: the controller hands the job over and replies at once, a worker thread
/// of its own computes it, the device's combine waits for its flag; nv2_host.cpp `serve` /
/// `cpu_loop`, #L484-L569). The host enqueues a whole row without knowing the routing, so the
/// GPU computes every combo; the controller points the table entries of the CPU's experts at a
/// zeroed VRAM record (no PCIe read; the combo adds 0), writes the reply and
/// [`DevLane::submit`]s the job. The worker waits per expert for its record to land ([`Ready`]),
/// computes the CPU's experts from their pinned records on the MoE input row the device copied to
/// the host, sums them weighted into one f32 row and raises the lane flag; [`Ctl::combine_lane`]
/// adds that row behind the layer's combine. Another f32 order than the GPU combine (the CPU's
/// experts summed first), so not the bits of the host lane; held to accuracy (#202 D-C).
pub struct DevLane {
    buf: Pinned,
    /// one zeroed record in VRAM: the table entry of every combo the CPU computes
    pub dummy: Dev,
    h: usize,
    tx: Option<std::sync::mpsc::Sender<LaneJob>>,
    th: Option<std::thread::JoinHandle<()>>,
    sent: std::sync::atomic::AtomicU64,
    done: std::sync::Arc<std::sync::atomic::AtomicU64>,
    thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::ThreadId>>>,
}

impl DevLane {
    /// The lane's buffer and its worker thread (the creating thread's CUDA context current on it,
    /// for `cuEventQuery`); `clock` and `ctl` count its pool runs and jobs.
    ///
    /// # Safety
    /// A CUDA context is current and outlives the lane.
    pub unsafe fn new(h: usize, geo: crate::glm5_moe::MoeGeo, clock: std::sync::Arc<crate::glm5_moe::lane::Clock>, ctl: std::sync::Arc<CtlClock>) -> DevLane {
        assert!(h % 4 == 0, "glm5 controller CPU lane: hidden {h} is not a multiple of 4");
        let buf = Pinned::alloc(DL_X + 2 * h * 4);
        std::ptr::write_bytes(buf.host as *mut u8, 0, buf.bytes);
        let dummy = cuda::alloc_zeroed(geo.record.bytes as usize);
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        cuda::ck(sys::cuCtxGetCurrent(&mut ctx));
        let ctx = SendPtr(ctx as *mut u8);
        let done: std::sync::Arc<std::sync::atomic::AtomicU64> = Default::default();
        let thread: std::sync::Arc<std::sync::Mutex<Option<std::thread::ThreadId>>> = Default::default();
        let lc = LaneCtx { base: SendPtr(buf.host as *mut u8), h, geo, clock, ctl, done: done.clone(), thread: thread.clone() };
        let (tx, rx) = std::sync::mpsc::channel::<LaneJob>();
        let th = std::thread::Builder::new()
            .name("glm5-cpu-lane".into())
            .spawn(move || {
                let (c, lc) = (ctx, lc);
                // SAFETY: the creator's context, alive while the lane runs
                unsafe { cuda::ck(sys::cuCtxSetCurrent(c.0 as sys::CUcontext)) };
                if let Ok(mut t) = lc.thread.lock() {
                    *t = Some(std::thread::current().id());
                }
                while let Ok(job) = rx.recv() {
                    // SAFETY: see `LaneJob`
                    unsafe { lc.run(job) };
                    lc.done.fetch_add(1, std::sync::atomic::Ordering::Release);
                }
            })
            .expect("glm5 controller: spawning the CPU lane thread");
        DevLane { buf, dummy, h, tx: Some(tx), th: Some(th), sent: Default::default(), done, thread }
    }

    pub fn dev(&self) -> DevLaneDev {
        DevLaneDev { x_host: self.buf.host as u64 + DL_X as u64, flag: self.buf.dev, y: self.buf.dev + (DL_X + self.h * 4) as u64 }
    }

    /// Hand request `q`'s CPU job to the worker and return at once (also with no pick: the device
    /// waits for the flag on every layer).
    ///
    /// # Safety
    /// The device published request `q` (so the x row is in place and the last combine ran);
    /// every pick's record is readable once its [`Ready`] holds.
    pub unsafe fn submit(&self, q: u64, picks: Vec<LanePick>) {
        self.sent.fetch_add(1, std::sync::atomic::Ordering::AcqRel);
        self.tx.as_ref().expect("glm5 controller: the CPU lane is gone").send(LaneJob { q, picks }).expect("glm5 controller: the CPU lane thread is gone");
    }

    /// wait until the worker finished every job handed to it
    pub fn idle(&self) {
        while self.done.load(std::sync::atomic::Ordering::Acquire) < self.sent.load(std::sync::atomic::Ordering::Acquire) {
            if self.th.as_ref().is_none_or(|t| t.is_finished()) {
                return;
            }
            std::thread::yield_now();
        }
    }

    /// the worker thread's id (`None` before it started)
    pub fn thread_id(&self) -> Option<std::thread::ThreadId> {
        self.thread.lock().ok().and_then(|t| *t)
    }

    /// the lane flag now (the last request whose CPU row is in place)
    pub fn flag_now(&self) -> u64 {
        // SAFETY: the live mapped flag word
        unsafe { lane_flag(self.buf.host as *mut u8).load(std::sync::atomic::Ordering::Acquire) }
    }

    /// the CPU's summed row of the last job (host side of `DevLaneDev::y`)
    pub fn row(&self) -> Vec<f32> {
        // SAFETY: the live mapped row, `h` f32
        unsafe { std::slice::from_raw_parts((self.buf.host as *const u8).add(DL_X + self.h * 4) as *const f32, self.h).to_vec() }
    }

    /// after a failure: every lane wait of the device passes from now on
    pub fn release(&self) {
        // SAFETY: the live mapped flag word
        unsafe { lane_flag(self.buf.host as *mut u8).store(u64::MAX, std::sync::atomic::Ordering::SeqCst) };
    }

    /// back to a fresh controller (its counter restarts at 0 with new switches); the worker
    /// finishes first
    pub fn reset(&self) {
        self.idle();
        // SAFETY: the live mapped flag word
        unsafe { lane_flag(self.buf.host as *mut u8).store(0, std::sync::atomic::Ordering::SeqCst) };
    }

    /// # Safety
    /// No launch reading the lane is pending.
    pub unsafe fn free(&mut self) {
        self.tx = None;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
        self.buf.free();
        cuda::free_dev(&mut self.dummy);
    }
}

/// `CROW_GLM_CONTROLLER`: the host controller thread. It makes the creating thread's CUDA
/// context current and runs the jobs it is sent, in order; each job's result comes back in order.
pub struct Worker<T: Send + 'static> {
    tx: Option<std::sync::mpsc::Sender<Box<dyn FnOnce() -> Result<T, String> + Send>>>,
    rx: std::sync::mpsc::Receiver<Result<T, String>>,
    th: Option<std::thread::JoinHandle<()>>,
    /// jobs sent whose result has not been taken
    pub pending: usize,
}

impl<T: Send + 'static> Worker<T> {
    /// # Safety
    /// A CUDA context is current on this thread and outlives the worker.
    pub unsafe fn new() -> Worker<T> {
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        cuda::ck(sys::cuCtxGetCurrent(&mut ctx));
        let ctx = SendPtr(ctx as *mut u8);
        let (tx, jobs) = std::sync::mpsc::channel::<Box<dyn FnOnce() -> Result<T, String> + Send>>();
        let (done, rx) = std::sync::mpsc::channel();
        let th = std::thread::Builder::new()
            .name("glm5-controller".into())
            .spawn(move || {
                let c = ctx;
                // SAFETY: the creator's context, alive while the worker runs
                unsafe { cuda::ck(sys::cuCtxSetCurrent(c.0 as sys::CUcontext)) };
                while let Ok(job) = jobs.recv() {
                    if done.send(job()).is_err() {
                        break;
                    }
                }
            })
            .expect("glm5 controller: spawning the thread");
        Worker { tx: Some(tx), rx, th: Some(th), pending: 0 }
    }

    pub fn send(&mut self, job: Box<dyn FnOnce() -> Result<T, String> + Send>) {
        self.tx.as_ref().expect("glm5 controller: the worker is gone").send(job).expect("glm5 controller: the thread is gone");
        self.pending += 1;
    }

    /// the result of the oldest job not taken yet
    pub fn wait(&mut self) -> Result<T, String> {
        assert!(self.pending > 0, "glm5 controller: no job to wait for");
        self.pending -= 1;
        self.rx.recv().map_err(|_| format!("{ENV_CONTROLLER}: the controller thread is gone"))?
    }

    /// every pending job's result is dropped, the thread ends
    pub fn free(&mut self) {
        while self.pending > 0 {
            let _ = self.wait();
        }
        self.tx = None;
        if let Some(t) = self.th.take() {
            let _ = t.join();
        }
    }
}

impl<T: Send + 'static> Drop for Worker<T> {
    fn drop(&mut self) {
        self.free();
    }
}

// ---------------------------------------------------------------- CROW_GLM_LOOKAHEAD

/// #189: the token embedding in VRAM and the gather that feeds a device id into the residual
pub struct Feed {
    pub table: Dev,
    pub bytes: u64,
    hidden: usize,
    feed: CUfunction,
    /// `CROW_GLM_LA`: the table in host-mapped pinned memory (`table` is its device alias)
    host: Option<Pinned>,
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
        Feed { table: cuda::upload_dev_named("glm5 lookahead embedding table", raw), bytes: raw.len() as u64, hidden: g.hidden, feed: k.feed, host: None }
    }

    /// `CROW_GLM_LA`: [`Feed::load`] with the table in host-mapped pinned memory, read
    /// zero-copy by the gather (one row per token; the reference's `prep_model`)
    ///
    /// # Safety
    /// As [`Feed::load`].
    pub unsafe fn load_mapped(k: &Kernels, cnq: &mut Cnq, g: &Glm5Geo) -> Feed {
        let t = cnq.find("model.language_model.embed_tokens.weight", "text").clone();
        assert_eq!((t.dtype.as_str(), t.shape.as_slice()), ("bf16", &[g.vocab as u64, g.hidden as u64][..]), "glm5 lookahead: embed_tokens");
        Feed::from_table_mapped(k, &cnq.read_bytes(&t), g)
    }

    /// # Safety
    /// As [`Feed::from_table`].
    pub unsafe fn from_table_mapped(k: &Kernels, raw: &[u8], g: &Glm5Geo) -> Feed {
        assert_eq!(raw.len(), g.vocab * g.hidden * 2, "glm5 lookahead: the embedding table is not [{}][{}] BF16", g.vocab, g.hidden);
        let p = Pinned::alloc(feed_pinned_bytes(g) as usize);
        std::ptr::copy_nonoverlapping(raw.as_ptr(), p.host as *mut u8, raw.len());
        Feed { table: p.dev, bytes: p.bytes as u64, hidden: g.hidden, feed: k.feed, host: Some(p) }
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
        match self.host.take() {
            Some(mut p) => {
                p.free();
                self.table = 0;
            }
            None => cuda::free_dev(&mut self.table),
        }
    }
}

/// #189: the head's greedy id (and its logits, when kept) copied async into pinned memory, read
/// by the host after a later sync point of the same stream
pub struct Readback {
    id: Pinned,
    logits: Pinned,
    vocab: usize,
    /// `CROW_GLM_LA`: recorded behind [`Readback::enqueue_marked`]
    ev: sys::CUevent,
}

impl Readback {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(vocab: usize) -> Readback {
        Readback { id: Pinned::alloc(4096), logits: Pinned::alloc((vocab * 4).next_multiple_of(4096)), vocab, ev: cuda::event_create() }
    }

    /// `CROW_GLM_LA`: [`Readback::enqueue`], then an event the host waits on with
    /// [`Readback::wait_marked`] (the copies only, not what is queued behind them)
    ///
    /// # Safety
    /// As [`Readback::enqueue`].
    pub unsafe fn enqueue_marked(&self, next: Dev, logits: Option<Dev>) {
        self.enqueue(next, logits);
        cuda::event_record(self.ev, cuda::cur_stream());
        cuda::stream_query(cuda::cur_stream());
    }

    /// # Safety
    /// [`Readback::enqueue_marked`] ran.
    pub unsafe fn wait_marked(&self) {
        cuda::ck(sys::cuEventSynchronize(self.ev));
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
        cuda::event_destroy(self.ev);
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

    /// #202 N2: the trim is off unless `1`, its cap defaults to no cut, a cap that is no whole
    /// number from 1 is refused by name
    #[test]
    fn n2_the_guess_trim_is_off_unless_asked() {
        for off in [None, Some(""), Some("0"), Some("yes")] {
            assert_eq!(guess_trim_from(off, Some("5")), Ok(None));
        }
        assert_eq!(guess_trim_from(Some("1"), None), Ok(Some(usize::MAX)));
        assert_eq!(guess_trim_from(Some("1"), Some(" ")), Ok(Some(usize::MAX)));
        assert_eq!(guess_trim_from(Some("1"), Some("3")), Ok(Some(3)));
        assert_eq!(guess_trim_from(Some("1"), Some("8")), Ok(Some(8)));
        for bad in ["0", "-1", "three"] {
            let e = guess_trim_from(Some("1"), Some(bad)).unwrap_err();
            assert!(e.contains(ENV_GUESS_TRIM_K) && e.contains(bad), "{e}");
        }
    }

    /// #202 S, the host side of a self-published guess on a block of host words: read at once
    /// when its word is up (counted ready), dropped after the wait when it is not (counted late),
    /// dropped when the block holds another layer's guess
    #[test]
    fn s_a_side_guess_is_read_when_its_word_is_up_and_dropped_when_late() {
        let mut block = [0u64; 8];
        let base = block.as_mut_ptr() as *mut u8;
        let ids = [5i32, 9, 2];
        unsafe {
            std::ptr::write(base as *mut u64, 4);
            std::ptr::write(base.add(8) as *mut i32, 7);
            for (i, &v) in ids.iter().enumerate() {
                std::ptr::write((base.add(16) as *mut i32).add(i), v);
            }
        }
        let hint = |layer: usize, seq: u64| SideHint { word: base as *const u64, tag: unsafe { base.add(8) } as *const i32, ids: unsafe { base.add(16) } as *const i32, k: 3, layer, seq };
        let mut how = SideRead::default();
        assert_eq!(read_side(hint(7, 4), SIDE_WAIT, &mut how), Some(Hint { layer: 7, ids: ids.to_vec() }));
        assert_eq!(how, SideRead { ready: 1, waited: 0, late: 0 });
        assert_eq!(read_side(hint(7, 5), std::time::Duration::from_micros(50), &mut how), None);
        assert_eq!(how.late, 1);
        assert_eq!(read_side(hint(6, 4), SIDE_WAIT, &mut how), None, "another layer's guess");
        // the hand-off: posted once, a later post of a plain hint replaces it
        post_side(hint(7, 4));
        assert_eq!(take_hint(), None);
        post_hint(None);
        assert!(SIDE.with(|c| c.get()).is_none());
    }

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
    fn the_stager_switch_needs_the_flags_and_takes_the_cpu_lane() {
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
        // the CPU lane reads its records after the stager's moves (integration, cross-wiring 2)
        assert_eq!(stager_on(Some("1"), Some("1"), Some("1")), Ok(true));
        assert_eq!(stager_on(Some("1"), Some("1"), Some("split")), Ok(true));
        // what the plan books for the stager: GLM-5.3-Flash, 42 MoE layers x 288, the 3-bit record, top-8
        let g = Glm5Geo::GLM_5_3_FLASH;
        assert_eq!(crate::glm5_tiers::stager_pinned_bytes(g.moe_layers(), g.experts, g.topk, 9_474_048), 75_988_992);
    }

    /// the synthetic 8-layer model of the stager tests (the #190 shape): layers 0-2 KDA + dense,
    /// 3-7 MoE with 16 MUL1 experts, top-8, DSA at 3 and 7, vocab 2048
    pub(crate) fn geo8() -> Glm5Geo {
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
            assert!(["off", "flags", "stager", "graph", "prefetch", "side", "overlap", "ctl", "la"].contains(t), "GLM_STAGER_PROFILE_ARM={arm:?}: off | a + list of flags, stager, graph, prefetch, side, overlap, ctl, la");
        }
        let on = |t: &str| toks.contains(&t);
        let (graph, stager) = (on("graph"), on("stager"));
        let sw = Switches { flags: on("flags"), lookahead: false, prefetch: on("prefetch"), pf_side: on("side"), overlap: on("overlap"), controller: on("ctl"), la: on("la") };
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
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + warm + rows + 1, &mut |s| eprintln!("{s}"));
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
                tok = if sw.la { run.decode_la(&mut cnq, &mut tiers, tok, pos).unwrap() } else { run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap() };
                pos += 1;
            }
            cuda::sync();
            let (reads0, moves0) = (tiers.nvme_reads, tiers.moves.iter().map(|m| m.h2d() + m.vram_to_stage + m.stage_to_vram + m.vram_to_pinned).sum::<u64>());
            let st0 = tiers.stager_stats().unwrap_or_default();
            crate::glm5_graph::profiler(true);
            let t0 = std::time::Instant::now();
            for _ in 0..rows {
                tok = if sw.la { run.decode_la(&mut cnq, &mut tiers, tok, pos).unwrap() } else { run.row(&mut cnq, &mut tiers, tok, pos, true).unwrap().unwrap() };
                pos += 1;
            }
            crate::glm5_graph::profiler(false);
            run.settle_ahead(&mut tiers).unwrap();
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
        assert_eq!(all, Switches { flags: true, lookahead: false, prefetch: true, pf_side: true, overlap: true, ..Switches::default() });
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

    /// #202 S (`CROW_GLM_SIDE_NOJOIN`): the side guess publishes itself. Layer 0's call on the
    /// side stream, unjoined: the router's publish carries the ids alone (no hint by
    /// [`take_hint`]), [`take_guess`] reads the guess the side stream published (the router
    /// kernel's top-8 of its own logits, for layer 1) once, the trim cap cuts it to its 3
    /// best-scored ids, and layer 1's call scores the guess as read; [`Routed::join_late`] then
    /// owes nothing. Without the switch the same calls hand the guess over with the flag.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_s_the_side_guess_publishes_itself() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (e, k, h) = (g.experts, g.topk, g.hidden);
        let mut rng = Rng(0x0202_5E1F);
        let w_bf: Vec<u16> = (0..e * h).map(|_| gm::f32_to_bf16_rne(rng.sym() / (h as f32).sqrt())).collect();
        let bias: Vec<f32> = (0..e).map(|_| 0.05 * rng.sym()).collect();
        let x: Vec<f32> = (0..h).map(|_| rng.sym()).collect();
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
            let mut pf = Prefetch::new(k, 4096);
            for nojoin in [true, false] {
                let mut r = Routed::new(&k5, k);
                let mut p = Predict::with(vec![None, Some((wr, wb))], e, k, h, 2.5, 7.0, true);
                if nojoin {
                    p.nojoin_on();
                }
                r.pred = Some(p);
                assert_eq!(r.side_nojoin(), nojoin);
                let s0 = pf.stats;
                for call in 0..3 {
                    cuda::to_i32_into(ids, &[7, 1, 2, 3, 4, 5, 6, 0]);
                    r.predict_early_unjoined(&kn, &gk, 0, xd);
                    r.predict(&kn, &gk, 0, xd);
                    r.publish(ids, k);
                    let got = r.wait_layer(0).unwrap();
                    assert_eq!(got, vec![7, 1, 2, 3, 4, 5, 6, 0], "nojoin {nojoin} call {call}: layer 0's own ids");
                    let want = {
                        let pr = r.pred.as_ref().unwrap();
                        let (_, _, e1) = pr.side.unwrap();
                        cuda::ck(sys::cuEventSynchronize(e1));
                        host_select(&cuda::dtoh(pr.logits, e), &bias, k)
                    };
                    if nojoin {
                        assert_eq!(take_hint(), None, "call {call}: the router's publish carried a guess");
                    }
                    pf.trim = None;
                    let hint = take_guess(&mut pf);
                    assert_eq!(hint, Some(Hint { layer: 1, ids: want.clone() }), "nojoin {nojoin} call {call}: the guess");
                    assert_eq!(take_guess(&mut pf), None, "a guess is handed over once");
                    r.join_late();
                    assert!(!r.pred.as_ref().unwrap().late, "the late join is still owed");
                    // layer 1's own call (no next MoE layer: no guess), scored against the guess
                    let mut own: Vec<i32> = want.iter().rev().copied().take(6).collect();
                    own.extend([-1, e as i32]);
                    cuda::to_i32_into(ids, &own);
                    r.predict_early_unjoined(&kn, &gk, 1, xd);
                    r.predict(&kn, &gk, 1, xd);
                    r.publish(ids, k);
                    r.wait_layer(1).unwrap();
                    assert_eq!(take_guess(&mut pf), None, "nojoin {nojoin} call {call}: a call without a guess hands over none");
                }
                let gs = r.guess;
                assert_eq!((gs.launched, gs.side, gs.compared, gs.picks, gs.hits), (3, 3, 3, 3 * k as u64, 3 * 6), "nojoin {nojoin}: {gs:?}");
                let d = pf.stats.since(&s0);
                let read = d.side_ready + d.side_waited;
                assert_eq!((read, d.side_late), (if nojoin { 3 } else { 0 }, 0), "nojoin {nojoin}: {d:?}");
                // the rank cap
                cuda::to_i32_into(ids, &[7, 1, 2, 3, 4, 5, 6, 0]);
                r.predict_early_unjoined(&kn, &gk, 0, xd);
                r.predict(&kn, &gk, 0, xd);
                r.publish(ids, k);
                r.wait_layer(0).unwrap();
                pf.trim = Some(3);
                let c0 = pf.stats.capped;
                let hint = take_guess(&mut pf).expect("the guess");
                assert_eq!((hint.ids.len(), pf.stats.capped - c0), (3, k as u64 - 3), "nojoin {nojoin}: the cap");
                r.join_late();
                eprintln!("glm5 guess S nojoin {nojoin}: {gs:?}, side reads {d:?}");
                r.free();
            }
            pf.buf.free();
            pf.flags.free();
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
        let sw = |flags, lookahead, prefetch, pf_side, overlap| Switches { flags, lookahead, prefetch, pf_side, overlap, ..Switches::default() };
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

    // ---------------------------------------------------------------- CROW_GLM_CONTROLLER / CROW_GLM_LA

    #[test]
    fn the_controller_switches_refuse_by_name_and_book_their_pinned_bytes() {
        let parse = |pairs: Vec<(&str, &str)>| Switches::parse(&|k: &str| pairs.iter().find(|p| p.0 == k).map(|p| p.1.to_string()));
        let all = parse(vec![(ENV_FLAGS, "1"), (ENV_CONTROLLER, "1"), (ENV_LA, "1")]);
        assert_eq!(all, Switches { flags: true, controller: true, la: true, ..Switches::default() });
        assert_eq!(all.check(), Ok(()));
        assert_eq!(all.label(), "CROW_GLM_FLAGS on, CROW_GLM_LOOKAHEAD off, CROW_GLM_CONTROLLER on, CROW_GLM_LA on");
        let e = parse(vec![(ENV_CONTROLLER, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_CONTROLLER=1 needs CROW_GLM_FLAGS=1 and CROW_GLM_STAGER=1"), "{e}");
        let e = parse(vec![(ENV_FLAGS, "1"), (ENV_LA, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_LA=1 needs CROW_GLM_CONTROLLER=1"), "{e}");
        let e = parse(vec![(ENV_FLAGS, "1"), (ENV_CONTROLLER, "1"), (ENV_LOOKAHEAD, "1")]).check().unwrap_err();
        assert!(e.starts_with("CROW_GLM_CONTROLLER=1 and CROW_GLM_LOOKAHEAD=1"), "{e}");
        // what the plan books: GLM-5.3-Flash, the 3-bit record, top-8; the BF16 embedding table
        let g = Glm5Geo::GLM_5_3_FLASH;
        assert_eq!(prefetch_pinned_bytes(g.topk, 9_474_048), 151_588_864);
        assert_eq!(feed_pinned_bytes(&g), 1_268_776_960);
        let b = |sw: Switches| crate::glm5_tiers::decode_switch_pinned_bytes(&g, &sw, 9_474_048);
        assert_eq!(b(Switches::default()), 0);
        assert_eq!(b(Switches { flags: true, prefetch: true, ..Switches::default() }), 151_588_864);
        assert_eq!(b(Switches { flags: true, prefetch: true, controller: true, la: true, ..Switches::default() }), 151_588_864 + 1_268_776_960);
    }

    #[test]
    fn the_controller_clock_sums_spans_per_layer() {
        let ms = std::time::Duration::from_millis;
        let c = CtlClock::default();
        c.served(ms(1), ms(2), ms(4));
        let a = c.read();
        c.served(ms(3), ms(4), ms(10));
        c.cpu_job(ms(5), ms(1));
        let d = c.read().since(&a);
        assert_eq!((d.requests, d.plan_ns, d.reply_ns, d.serve_ns, d.serve_max_ns), (1, 3_000_000, 4_000_000, 10_000_000, 10_000_000));
        assert_eq!((d.cpu_jobs, d.cpu_busy_ns, d.cpu_wait_land_ns), (1, 5_000_000, 1_000_000));
        assert_eq!(c.read().per_layer_us(c.read().serve_ns), 7_000.0);
    }

    #[test]
    fn the_controller_scores_a_guess_against_the_layer_it_named() {
        let mut s = GuessScore::default();
        s.see(3, &[1, 2, 3, 4], Some(&Hint { layer: 4, ids: vec![2, 3, 9, 8] }));
        assert_eq!(s.stats.compared, 0, "no guess for layer 3");
        s.see(4, &[3, 2, 7, 6], None);
        assert_eq!((s.stats.compared, s.stats.picks, s.stats.hits), (1, 4, 2));
        s.see(5, &[1, 2, 3, 4], None);
        assert_eq!(s.stats.compared, 1, "layer 4's call made no guess");
        s.see(6, &[1, 2, 3, 4], Some(&Hint { layer: 7, ids: vec![1, 2, 3, 4] }));
        s.see(8, &[1, 2, 3, 4], None);
        assert_eq!(s.stats.compared, 1, "a guess for layer 7 does not score layer 8");
    }

    /// The device wait of the controller is bounded: a request nobody serves lets the stream go on
    /// after `CTL_WAIT_NS` and names its sequence number in the error word (the WDDM TDR guard).
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_an_unserved_request_times_out_on_the_device() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut c = Ctl::new(&k, g.topk, g.layers);
            let mut ids = cuda::to_i32_dev(&[5, 4, 3, 2, 1, 0, 6, 7]);
            let mut rd = c.reader(1);
            let t0 = std::time::Instant::now();
            let mut wts = cuda::to_f32_dev(&[0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625]);
            c.publish(3, ids, g.topk, None, wts);
            c.wait_reply();
            let rq = rd.next().unwrap();
            assert_eq!(rq, Request { seq: 1, layer: 3, ids: vec![5, 4, 3, 2, 1, 0, 6, 7], guess: None, wts: vec![0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0.0078125, 0.00390625] });
            cuda::sync();
            let dt = t0.elapsed().as_secs_f64();
            assert_eq!(c.timed_out(), 1, "the unserved request is named");
            assert!((0.9..1.9).contains(&dt), "the wait gave up after {dt:.3} s");
            eprintln!("glm5 controller: an unserved request released the stream after {dt:.3} s");
            cuda::free_dev(&mut ids);
            cuda::free_dev(&mut wts);
            c.free();
            k.free();
        }
    }

    /// A row as deep as the real model's keeps its controller served: 64 requests, each followed
    /// by 40 launches on the compute stream (a MoE layer's worth), the host enqueuing the row
    /// whole while a controller thread answers every request with a copy and the reply write on a
    /// stream of its own (the stager's calls). Without [`Ctl::pace`] the host ran thousands of
    /// launches ahead of the device's wait, `cuLaunchKernel` blocked on the full launch queue, the
    /// controller's next driver call waited behind it, and the device's wait gave up after
    /// `CTL_WAIT_NS` (the GLM-5.3-Flash ILLEGAL_ADDRESS of 2026-10-10: the experts then read a
    /// stale table). With it no wait times out and every request is answered once.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_a_deep_row_keeps_the_controller_served() {
        const REQUESTS: usize = 64;
        const FILL: usize = 40;
        let g = Glm5Geo::GLM_5_3_FLASH;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut c = Ctl::new(&k, g.topk, g.layers);
            let mut w: Worker<usize> = Worker::new();
            let mut ids = cuda::to_i32_dev(&[5, 4, 3, 2, 1, 0, 6, 7]);
            // the filler: glm5_pred_tag, one int written
            let mut tag = cuda::alloc_named("glm5 controller test tag", 4);
            let mut dst = cuda::alloc_named("glm5 controller test copy", 4096);
            let mut src = Pinned::alloc(4096);
            let s = cuda::stream_create_non_blocking();
            let (sp, srcp, dstp) = (s as u64, src.host as u64, dst);
            let mut rd = c.reader(REQUESTS);
            w.send(Box::new(move || {
                let mut served = 0;
                for _ in 0..REQUESTS {
                    let rq = rd.next()?;
                    let st = sp as sys::CUstream;
                    cuda::ck(sys::cuMemcpyHtoDAsync_v2(dstp, srcp as *const std::ffi::c_void, 4096, st));
                    cuda::ck(sys::cuStreamWriteValue64_v2(st, rd.reply_dev, rq.seq, 0));
                    cuda::stream_query(st);
                    served += 1;
                }
                Ok(served)
            }));
            let t0 = std::time::Instant::now();
            for layer in 0..REQUESTS {
                c.pace().unwrap();
                c.publish(layer, ids, g.topk, None, 0);
                c.wait_reply();
                for _ in 0..FILL {
                    launch_v(k.pred_tag, 1, 1, 1, 32, &[tag, 0]);
                }
            }
            cuda::sync();
            let served = w.wait().unwrap();
            let dt = t0.elapsed().as_secs_f64();
            eprintln!("glm5 controller: {REQUESTS} requests x {FILL} launches in {dt:.3} s, first timed-out request {}", c.timed_out());
            assert_eq!(c.timed_out(), 0, "a device wait gave up: the controller was starved by the host's launches");
            assert_eq!(served, REQUESTS);
            w.free();
            cuda::stream_sync(s);
            cuda::stream_destroy(s);
            src.free();
            for d in [&mut ids, &mut tag, &mut dst] {
                cuda::free_dev(d);
            }
            c.free();
            k.free();
        }
    }

    /// a lane buffer as `DevLane` lays it out (flag, count, the row at byte 64) in `p`
    fn test_lane(p: &Pinned) -> DevLaneDev {
        DevLaneDev { x_host: 0, flag: p.dev, y: p.dev + 64 }
    }

    /// #202 D-C: `glm5_lane_add` adds the CPU's one row to the combined output, in every load mode,
    /// as the host's f32 add bit for bit (one `__fadd_rn` per value); with no CPU expert in the job
    /// (count 0) it leaves y alone; a flag that never comes gives up after the timeout, names the
    /// request in the error word and leaves y alone.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_the_lane_row_adds_to_the_combine() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let h = g.hidden;
        let mut rng = Rng(0x202c);
        let y0: Vec<f32> = (0..h).map(|_| rng.sym() * 300.0).collect();
        let row: Vec<f32> = (0..h).map(|_| rng.sym() * 7.0).collect();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut p = Pinned::alloc(64 + h * 4);
            let base = p.host as *mut u8;
            std::ptr::copy_nonoverlapping(row.as_ptr(), base.add(64) as *mut f32, h);
            let dl = test_lane(&p);
            let mut ctr = cuda::to_u64_dev(&[5]);
            let mut errw = Pinned::alloc(4096);
            let err_host = errw.host as *mut u64;
            let mut y = cuda::to_f32_dev(&y0);
            let want: Vec<u32> = y0.iter().zip(&row).map(|(a, b)| (a + b).to_bits()).collect();
            for mode in 0..3u64 {
                cuda::to_f32_into(y, &y0);
                std::ptr::write_volatile(base as *mut u64, 5);
                std::ptr::write_volatile(base.add(8) as *mut u64, 3);
                lane_add(k.lane_add, y, &dl, ctr, errw.dev, CTL_WAIT_NS, h, mode);
                cuda::sync();
                let got: Vec<u32> = cuda::dtoh(y, h).iter().map(|v| v.to_bits()).collect();
                assert!(got == want, "mode {mode}: {} of {h} values are not the host's sum", got.iter().zip(&want).filter(|(a, b)| a != b).count());
            }
            // no CPU expert in the job: y stays
            cuda::to_f32_into(y, &y0);
            std::ptr::write_volatile(base.add(8) as *mut u64, 0);
            lane_add(k.lane_add, y, &dl, ctr, errw.dev, CTL_WAIT_NS, h, 2);
            cuda::sync();
            assert!(cuda::dtoh(y, h) == y0, "a job without CPU experts changed y");
            // the flag stays below the request: the wait gives up after 5 ms, y stays
            std::ptr::write_volatile(base as *mut u64, 4);
            std::ptr::write_volatile(base.add(8) as *mut u64, 3);
            std::ptr::write_volatile(err_host, 0);
            let t0 = std::time::Instant::now();
            lane_add(k.lane_add, y, &dl, ctr, errw.dev, 5_000_000, h, 2);
            cuda::sync();
            let dt = t0.elapsed();
            assert_eq!(std::ptr::read_volatile(err_host), 5, "the timed-out request is named");
            assert!(cuda::dtoh(y, h) == y0, "a timed-out wait changed y");
            eprintln!("glm5 lane add: modes 0/1/2 equal the host's sum bit for bit; count 0 and a timeout ({dt:?}) leave y alone");
            for d in [&mut ctr, &mut y] {
                cuda::free_dev(d);
            }
            p.free();
            errw.free();
            k.free();
        }
    }

    /// #202 D-C micro-bench, median of 200 per arm on the GPU's clock (CUDA events), the host
    /// rewriting the CPU rows before every launch (dirty lines in the CPU's cache, as after a
    /// lane job): before = `glm5_ctl_wait` + the former `glm5_lane_merge` (k rows of h f32 copied
    /// over `ye` with volatile scalar loads, 3 and 8 of 8 combos on the CPU); after = the one
    /// `glm5_lane_add` behind the combine reading one row, with volatile, `__ldcv` and `__ldcv`
    /// float4 loads. The flag is up: the read and the launch are timed, not a wait.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_flags_gpu_lane_combine_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_lane_combine_bench() {
        const OLD: &str = r#"
extern "C" __global__ void old_lane_merge(float* ye, const volatile float* y, const volatile int* mask, long long h)
{
    const int c = blockIdx.y;
    if (mask[c] == 0) return;
    for (long long i = blockIdx.x * (long long) blockDim.x + threadIdx.x; i < h; i += (long long) gridDim.x * blockDim.x)
        ye[c * h + i] = y[c * h + i];
}
"#;
        const ITERS: usize = 200;
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (h, kk) = (g.hidden, g.topk);
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut k = Kernels::new(&g);
            let mut m = cuda::compile(OLD);
            let old_merge = m.get("old_lane_merge");
            // old: flag, mask [k] at 64, rows [k][h] at 4096; new: flag, count, the row at 64
            let mut po = Pinned::alloc(4096 + kk * h * 4);
            let mut pn = Pinned::alloc(64 + h * 4);
            std::ptr::write_bytes(po.host as *mut u8, 0, po.bytes);
            std::ptr::write_bytes(pn.host as *mut u8, 0, pn.bytes);
            let (bo, bn) = (po.host as *mut u8, pn.host as *mut u8);
            std::ptr::write_volatile(bo as *mut u64, 1);
            std::ptr::write_volatile(bn as *mut u64, 1);
            std::ptr::write_volatile(bn.add(8) as *mut u64, 1);
            let dl = test_lane(&pn);
            let mut ctr = cuda::to_u64_dev(&[1]);
            let mut errw = Pinned::alloc(4096);
            let mut ye = cuda::alloc_zeroed(kk * h * 4);
            let mut y = cuda::alloc_zeroed(h * 4);
            let mk = |flags: u32| {
                let mut e: sys::CUevent = std::ptr::null_mut();
                cuda::ck(sys::cuEventCreate(&mut e, flags));
                e
            };
            let (e0, e1) = (mk(0), mk(0));
            let s = cuda::cur_stream();
            let mut time = |rows: usize, words: usize, launch: &dyn Fn()| -> f64 {
                let base = if rows == 0 { bn.add(64) } else { bo.add(4096) };
                let mut v = Vec::with_capacity(ITERS);
                for it in 0..ITERS {
                    let f = std::slice::from_raw_parts_mut(base as *mut f32, words);
                    f.fill(it as f32);
                    cuda::event_record(e0, s);
                    launch();
                    cuda::event_record(e1, s);
                    cuda::sync();
                    let mut ms = 0f32;
                    cuda::ck(sys::cuEventElapsedTime_v2(&mut ms, e0, e1));
                    v.push(ms as f64 * 1e3);
                }
                v.sort_by(|a, b| a.total_cmp(b));
                v[ITERS / 2]
            };
            let mut line = Vec::new();
            for cpu in [3usize, 8] {
                let mask = std::slice::from_raw_parts_mut(bo.add(64) as *mut i32, kk);
                for (c, w) in mask.iter_mut().enumerate() {
                    *w = (c < cpu) as i32;
                }
                let us = time(1, kk * h, &|| {
                    launch_v(k.ctl_wait, 1, 1, 1, 1, &[ctr, po.dev, errw.dev, CTL_WAIT_NS]);
                    launch_v(old_merge, h.div_ceil(256) as u32, kk as u32, 1, 256, &[ye, po.dev + 4096, po.dev + 64, h as u64]);
                });
                eprintln!("glm5 lane combine bench: before, wait + glm5_lane_merge, {cpu} of {kk} combos on the CPU: {us:.2} us per layer");
                line.push(format!("before {cpu}/{kk} {us:.2}"));
            }
            for (mode, name) in [(0u64, "volatile"), (1, "__ldcv"), (2, "__ldcv float4")] {
                let us = time(0, h, &|| lane_add(k.lane_add, y, &dl, ctr, errw.dev, CTL_WAIT_NS, h, mode));
                eprintln!("glm5 lane combine bench: after, glm5_lane_add {name}: {us:.2} us per layer");
                line.push(format!("after {name} {us:.2}"));
            }
            eprintln!("glm5 lane combine bench (us per layer, median of {ITERS}): {}", line.join(" | "));
            cuda::ck(sys::cuEventDestroy_v2(e0));
            cuda::ck(sys::cuEventDestroy_v2(e1));
            for d in [&mut ctr, &mut ye, &mut y] {
                cuda::free_dev(d);
            }
            po.free();
            pn.free();
            errw.free();
            m.unload();
            k.free();
        }
    }

    /// #202 D-A / D-D, a measurement on the synthetic model: the controller's clocks per
    /// controlled MoE layer (plan, reply, serve = the controller thread busy; the CPU lane's busy
    /// time and its wait for landings) under the measurement arm of the GLM-5.3-Flash rows with
    /// small budgets (global arena with warm start, NVMe piece pool of 8 workers, flags + stager +
    /// prefetch on the side stream + shared overlap + controller + LA, CPU lane split on host
    /// pinned memory), a 20-id prompt and 16 greedy ids, at V 3 + P 4 (NVMe reads) and V 0 + P 16
    /// (every expert pinned: the lane computes). Prints; holds that every decode row went through
    /// the controller and the lane ran.
    #[test]
    #[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_flags_gpu_controller_clock -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_controller_clock_per_layer() {
        use crate::glm5_int_tests::{synth_warm, Env};
        const REC: u64 = 9_474_048;
        let g = geo8();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let dir = std::env::temp_dir().join(format!("crow-ctl-clock-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let warm = synth_warm(&dir);
        let gib = |records: f64| format!("{}", records * REC as f64 / (1u64 << 30) as f64);
        let env: Vec<(&str, String)> = vec![
            ("CROW_NVME_POOL", "1".into()),
            ("CROW_NVME_POOL_THREADS", "8".into()),
            ("CROW_GLM_CPU_LANE", "split".into()),
            ("CROW_PINNED_ALLOC", "host".into()),
            ("CROW_GLM_ARENA", "global".into()),
            ("CROW_GLM_ARENA_WARM", warm),
            ("CROW_GLM_ARENA_ELASTIC_GB", gib(6.0)),
            ("CROW_GLM_ARENA_STAGE_GB", gib(12.0)),
            ("CROW_GLM_FLAGS", "1".into()),
            ("CROW_GLM_STAGER", "1".into()),
            ("CROW_GLM_PREFETCH", "1".into()),
            ("CROW_GLM_PREFETCH_SIDE", "1".into()),
            ("CROW_GLM_SHARED_OVERLAP", "1".into()),
            ("CROW_GLM_CONTROLLER", "1".into()),
            ("CROW_GLM_LA", "1".into()),
        ];
        let prompt: Vec<i64> = (0..20).map(|i| (i * 53 + 11) % 2048).collect();
        let n = 16;
        unsafe {
            let _ctx = cuda::Ctx::init();
            for sizes in [TierSizes { vram: 3, pinned: 4 }, TierSizes { vram: 0, pinned: 16 }] {
                let _env = Env::set(&env);
                let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |_| {});
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                let clock = tiers.ctl_clock();
                let t0 = std::time::Instant::now();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, false, &mut |_| {}).unwrap();
                let wall = t0.elapsed().as_secs_f64();
                let c = clock.read();
                let lane = tiers.cpu_lane_clock().read();
                eprintln!(
                    "glm5 controller clock V {} P {}: {} requests, per layer plan {:.1} us reply {:.1} us serve {:.1} us (max {:.1} us); CPU lane {} jobs ({} experts) busy {:.1} us, waiting for landings {:.1} us per layer; NVMe reads {}; generate {wall:.3} s; ids {:?}",
                    sizes.vram,
                    sizes.pinned,
                    c.requests,
                    c.per_layer_us(c.plan_ns),
                    c.per_layer_us(c.reply_ns),
                    c.per_layer_us(c.serve_ns),
                    c.serve_max_ns as f64 / 1e3,
                    c.cpu_jobs,
                    lane.1,
                    c.per_layer_us(c.cpu_busy_ns),
                    c.per_layer_us(c.cpu_wait_land_ns),
                    tiers.nvme_reads,
                    gen.ids
                );
                assert!(c.requests >= 5 * (n as u64 - 1), "every MoE layer of every decode row through the controller ({} requests)", c.requests);
                if sizes.vram == 0 {
                    assert!(lane.1 > 0, "the lane computed nothing");
                }
                tiers.free();
                run.free();
            }
        }
        drop(cnq);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// The controller and its lookahead are invisible in the output. The synthetic 8-layer model,
    /// a 5-id prompt and 6 greedy ids, VRAM 3 + pinned 4 slots, a fresh store per arm: every arm
    /// gives the switch-off ids and logits bit for bit and the row reports of flags + stager (the
    /// same moves through the same stager) except the clock; under the controller no layer's
    /// routing is read through the flag (`Routed::calls` stays 0) and every MoE layer of every row
    /// went through the ring. Then serve's door (`decode_la`) against `row` over the same fed ids,
    /// with a forced id in the middle (the row enqueued ahead on the greedy id is dropped and its
    /// KDA states come back) and the turn's end (the last row ahead dropped): the same ids and
    /// logits per step, and the same KDA states after the drop.
    #[test]
    #[ignore = "needs the GPU (about 2 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_flags_gpu_controller -- --ignored --nocapture --test-threads 1"]
    fn glm5_flags_gpu_controller_and_la_are_invisible_in_ids_logits_and_reports() {
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
        let flags = Switches { flags: true, ..Switches::default() };
        let ctl = Switches { flags: true, controller: true, ..Switches::default() };
        let arms: [(&str, Switches, bool); 7] = [
            ("off", Switches::default(), false),
            ("flags+stager", flags, false),
            ("ctl", ctl, false),
            ("ctl+prefetch+overlap", Switches { prefetch: true, overlap: true, ..ctl }, false),
            ("ctl+la", Switches { la: true, ..ctl }, false),
            ("ctl+la+prefetch+side+overlap", Switches { la: true, prefetch: true, pf_side: true, overlap: true, ..ctl }, false),
            ("ctl+la+prefetch zerocopy", Switches { la: true, prefetch: true, ..ctl }, true),
        ];
        let mut outs: Vec<(Generated, Vec<TokenReport>)> = Vec::new();
        let moe_rows = 5 * (prompt.len() + n - 1);
        // serve's door: the fed ids (step 2 forced to 7) and their logits / greedy ids
        let forced = 2usize;
        let mut door_ref: (Vec<i64>, Vec<Vec<f32>>, Vec<Vec<u8>>) = Default::default();
        let mut door_la: (Vec<i64>, Vec<Vec<f32>>, Vec<Vec<u8>>) = Default::default();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, prompt.len() + n + 1, &mut |s| eprintln!("{s}"));
            for (name, sw, zc) in arms {
                run.set_switches(&mut cnq, sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                if zc {
                    tiers.set_pinned_use(PinnedUse { stay: true, cpu_lane: false }).unwrap();
                }
                tiers.set_stager(sw.flags).unwrap();
                tiers.set_prefetch(sw.prefetch);
                let mut reps: Vec<TokenReport> = Vec::new();
                let gen = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                eprintln!("glm5 controller {name}: ids {:?}, NVMe reads {}, prefetch {:?}, guess {:?}", gen.ids, tiers.nvme_reads, tiers.prefetch_stats(), run.guess_stats());
                if sw.controller {
                    let st = tiers.stager_stats().unwrap();
                    assert_eq!(st.calls, moe_rows as u64, "{name}: every MoE layer of every row through the controller");
                    assert_eq!(run.flag_calls(), 0, "{name}: no routing read through the flag");
                }
                if sw.prefetch {
                    let gs = run.guess_stats().unwrap();
                    assert_eq!(gs.compared, (4 * (prompt.len() + n - 1)) as u64, "{name}: the controller scored every guess");
                }
                tiers.free();
                outs.push((gen, reps));
            }
            // serve's door: `row` without switches, then `decode_la` under ctl + la + prefetch
            let feed = |step: usize, greedy: i64| if step == forced { 7 } else { greedy };
            let kda_bytes = |run: &Glm5Run| {
                let kd = crate::glm5_kda::KdaDims::of(&g);
                run.kda_states().flat_map(|k| [cuda::dtoh_t::<u8>(k.s, kd.state_floats() * 4), cuda::dtoh_t::<u8>(k.conv, kd.conv_floats() * 4)]).collect::<Vec<_>>()
            };
            for la in [false, true] {
                let sw = if la { Switches { la: true, prefetch: true, ..ctl } } else { Switches::default() };
                run.set_switches(&mut cnq, sw);
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                tiers.set_stager(la).unwrap();
                tiers.set_prefetch(sw.prefetch);
                run.kda_states().for_each(|k| k.reset());
                let mut tok = 0i64;
                for (pos, &p) in prompt.iter().enumerate() {
                    if let Some(id) = run.row(&mut cnq, &mut tiers, p, pos, pos + 1 == prompt.len()).unwrap() {
                        tok = id;
                    }
                }
                let out = if la { &mut door_la } else { &mut door_ref };
                for step in 0..n {
                    if step == forced {
                        assert_ne!(tok, 7, "the forced id must differ from the greedy one for the drop to mean something");
                    }
                    let fed = feed(step, tok);
                    out.0.push(fed);
                    let pos = prompt.len() + step;
                    tok = if la { run.decode_la(&mut cnq, &mut tiers, fed, pos).unwrap() } else { run.row(&mut cnq, &mut tiers, fed, pos, true).unwrap().unwrap() };
                    // as `Glm5Device::logits`: the stream first (a row may be running ahead)
                    cuda::sync();
                    out.1.push(cuda::dtoh(run.logits_dev(), g.vocab));
                }
                if la {
                    assert!(run.has_ahead(), "the last step enqueued a row ahead");
                    run.settle_ahead(&mut tiers).unwrap();
                    assert!(!run.has_ahead());
                }
                cuda::sync();
                out.2 = kda_bytes(&run);
                tiers.free();
            }
            run.free();
        }
        drop(cnq);
        let (g0, r0) = &outs[0];
        let finite = g0.logits.iter().flatten().filter(|v| v.is_finite()).count();
        assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
        let bits = |a: &[Vec<f32>], b: &[Vec<f32>]| a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect::<Vec<_>>();
        let (_, rs) = &outs[1];
        for (i, (name, _, _)) in arms.iter().enumerate() {
            let (gx, rx) = &outs[i];
            assert_eq!(gx.ids, g0.ids, "{name}: ids");
            let diff = bits(&gx.logits, &g0.logits);
            assert!(diff.iter().all(|&d| d == 0), "{name}: logits differ in bits per generated position {diff:?}");
            if i >= 2 && !arms[i].2 {
                assert_eq!(rx.iter().map(unclocked).collect::<Vec<_>>(), rs.iter().map(unclocked).collect::<Vec<_>>(), "{name}: row reports against flags+stager");
            }
        }
        assert_eq!(r0.len(), prompt.len() + n - 1);
        assert_eq!(door_la.0, door_ref.0, "serve's door: the fed ids");
        let diff = bits(&door_la.1, &door_ref.1);
        assert!(diff.iter().all(|&d| d == 0), "serve's door with the lookahead: logits differ in bits per step {diff:?}");
        assert!(door_la.2 == door_ref.2, "serve's door: the KDA states after the last row ahead was dropped");
    }
}
