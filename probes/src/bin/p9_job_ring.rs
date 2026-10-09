//! Probe 9 (Crow #7): job ring + stream memops on Windows — strand-2 probe.
//!
//! Spec section 3.2 pattern, acceptance 3.6: pinned 64-B-aligned descriptor ring,
//! device-side mapped writes for descriptors, flag publication via
//! cuStreamWriteValue32, host stager busy-polls (no sync primitives), raises the
//! done flag stream-ordered on a copy stream, GPU consumer resumes behind
//! cuStreamWaitValue32. What Windows/WDDM actually allows is THE open question
//! (exl3 memops unexercised on Windows), so the probe auto-detects:
//!
//!   Path 1 (primary): cuStreamWriteValue32 onto the HOST-MAPPED pub flag.
//!   Path 2 (fallback): a stream-ordered publish kernel writes the flag via
//!     mapped write + __threadfence_system (deviation from spec wording,
//!     same visibility contract).
//!
//! Stage A (mechanics): 2 full laps over 256 slots — descriptor fields verified
//! host-side after publication, sequence numbers guard slot reuse, consumer
//! verifies payload + done flag behind the wait.
//!
//! Stage B (round trip): per-rep latency with sync (resolution: host Instant,
//! WDDM jitter dominated), stager on its own thread:
//!   mode 1 signaling-only (no payload copy), 1000 reps
//!   mode 2 with 1 MiB staged H2D before the done flag, 300 reps
//!   baseline: empty-kernel launch+sync per rep for context.
//!
//! Stage C (crow-nest #149, plan step 17a): `cuStreamWaitValue64_v2` inside a CUDA graph under
//! WDDM. The flag lives in two places, each with three builds of the same wait + consumer:
//!   flag: device memory (raised by an 8-B H2D on the copy stream, the stager pattern measured
//!         in stage B) and mapped pinned host memory (raised by a plain host store);
//!   U  uncaptured: `cuStreamWaitValue64_v2` (EQ 1) + consumer launch per replay (reference);
//!   C1 stream capture of exactly that pair, node types printed (wait must be BATCH_MEM_OP, 12);
//!   C2 explicit graph: `cuGraphAddBatchMemOpNode` (WAIT_VALUE_64, EQ 1) -> kernel node.
//! A writer thread raises the flag a fixed delay after each launch (300 us; 50 ms for the hold
//! check) and times wait-to-start: from its flag write to the consumer's start marker landing
//! in mapped host memory. Checks per replay: the marker must NOT be there before the flag is
//! raised (ordering), the consumer must have read flag == 1, the counter must equal the replay.
//! A negative control (consumer without the wait) must fail both the hold and the ordering
//! check, else the checks prove nothing and the probe exits 1.
//! TDR guard (WDDM default TdrDelay 2 s, never changed here): the flag is raised at most 50 ms
//! after a launch (the writer, or the main thread as rescue at delay + 250 ms); a consumer that
//! has not started 200 ms after the flag ends the probe (exit 2) after a bounded 200 ms drain.
//!
//! Scalars travel in device buffers (p5 lesson); HtoD async + sync (WDDM rule).

use std::ffi::CString;
use std::sync::atomic::AtomicBool;
use std::sync::atomic::{AtomicU32, AtomicU64};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUfunction, CUresult};

// cudarc 0.19.9 only generates the stream-memops bindings for cuda-11.x
// features, but the CUDA 13 driver exports them — load the two symbols
// directly from the same DLL cudarc already uses.
struct Memops {
    write32: unsafe extern "C" fn(sys::CUstream, CUdeviceptr, u32, u32) -> CUresult,
    #[allow(dead_code)]
    wait32: unsafe extern "C" fn(sys::CUstream, CUdeviceptr, u32, u32) -> CUresult,
}

fn load_memops() -> Memops {
    let lib = Box::leak(Box::new(unsafe { libloading::Library::new("nvcuda.dll").expect("open nvcuda.dll") }));
    type MemopFn = unsafe extern "C" fn(sys::CUstream, CUdeviceptr, u32, u32) -> CUresult;
    let write32: MemopFn = unsafe { *lib.get::<MemopFn>(b"cuStreamWriteValue32").expect("cuStreamWriteValue32") };
    let wait32: MemopFn = unsafe { *lib.get::<MemopFn>(b"cuStreamWaitValue32").expect("cuStreamWaitValue32") };
    Memops { write32, wait32 }
}
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

// stage C: the graph entry points, loaded out of the driver DLL the same way the engine does
// (`engine/src/cuda.rs` graph_sym: cudarc's bindings were not usable for them there)
type FnBeginCapture = unsafe extern "system" fn(sys::CUstream, sys::CUstreamCaptureMode) -> CUresult;
type FnEndCapture = unsafe extern "system" fn(sys::CUstream, *mut sys::CUgraph) -> CUresult;
type FnGraphCreate = unsafe extern "system" fn(*mut sys::CUgraph, u32) -> CUresult;
type FnGraphGetNodes = unsafe extern "system" fn(sys::CUgraph, *mut sys::CUgraphNode, *mut usize) -> CUresult;
type FnNodeGetType = unsafe extern "system" fn(sys::CUgraphNode, *mut u32) -> CUresult;
type FnAddBatchMemOp = unsafe extern "system" fn(
    *mut sys::CUgraphNode,
    sys::CUgraph,
    *const sys::CUgraphNode,
    usize,
    *const sys::CUDA_BATCH_MEM_OP_NODE_PARAMS,
) -> CUresult;
type FnAddKernelNode = unsafe extern "system" fn(
    *mut sys::CUgraphNode,
    sys::CUgraph,
    *const sys::CUgraphNode,
    usize,
    *const sys::CUDA_KERNEL_NODE_PARAMS_v2,
) -> CUresult;
type FnGetAttr = unsafe extern "system" fn(*mut i32, i32, sys::CUdevice) -> CUresult;
type FnInstantiate = unsafe extern "system" fn(*mut sys::CUgraphExec, sys::CUgraph, u64) -> CUresult;
type FnGraphLaunch = unsafe extern "system" fn(sys::CUgraphExec, sys::CUstream) -> CUresult;
type FnGraphDestroy = unsafe extern "system" fn(sys::CUgraph) -> CUresult;
type FnExecDestroy = unsafe extern "system" fn(sys::CUgraphExec) -> CUresult;

struct GraphApi {
    begin_capture: FnBeginCapture,
    end_capture: FnEndCapture,
    create: FnGraphCreate,
    get_nodes: FnGraphGetNodes,
    node_type: FnNodeGetType,
    add_batch_memop: FnAddBatchMemOp,
    add_kernel_node: FnAddKernelNode,
    get_attr: FnGetAttr,
    instantiate: FnInstantiate,
    launch: FnGraphLaunch,
    destroy: FnGraphDestroy,
    exec_destroy: FnExecDestroy,
}

fn load_graph_api() -> GraphApi {
    let lib = Box::leak(Box::new(unsafe { libloading::Library::new("nvcuda.dll").expect("open nvcuda.dll") }));
    unsafe fn sym<T: Copy>(lib: &libloading::Library, n: &[u8]) -> T {
        unsafe { *lib.get::<T>(n).unwrap_or_else(|e| panic!("nvcuda.dll {}: {e}", String::from_utf8_lossy(n))) }
    }
    unsafe {
        GraphApi {
            begin_capture: sym(lib, b"cuStreamBeginCapture_v2\0"),
            end_capture: sym(lib, b"cuStreamEndCapture\0"),
            create: sym(lib, b"cuGraphCreate\0"),
            get_nodes: sym(lib, b"cuGraphGetNodes\0"),
            node_type: sym(lib, b"cuGraphNodeGetType\0"),
            add_batch_memop: sym(lib, b"cuGraphAddBatchMemOpNode\0"),
            add_kernel_node: sym(lib, b"cuGraphAddKernelNode_v2\0"),
            get_attr: sym(lib, b"cuDeviceGetAttribute\0"),
            instantiate: sym(lib, b"cuGraphInstantiateWithFlags\0"),
            launch: sym(lib, b"cuGraphLaunch\0"),
            destroy: sym(lib, b"cuGraphDestroy\0"),
            exec_destroy: sym(lib, b"cuGraphExecDestroy\0"),
        }
    }
}

/// CU_GRAPH_NODE_TYPE_BATCH_MEM_OP
const NODE_BATCH_MEM_OP: u32 = 12;
const REPS_GRAPH: usize = 1000;
const REPS_CONTROL: usize = 50;
/// writer delay between a launch and its flag write (replays) and for the hold check
const WRITER_DELAY_US: u64 = 300;
const HOLD_DELAY_US: u64 = 50_000;
/// TDR guard: a consumer that has not started this long after its flag ends the probe
const GUARD_MS: u64 = 200;
const ST_RELEASED: u32 = 0;
const ST_EARLY: u32 = 1;
const ST_NO_START: u32 = 2;

const SLOTS: usize = 256;
const DESC_U32: usize = 16; // 64 B per descriptor slot
const PUB_STRIDE: usize = 16; // pub flags padded to 64 B to kill false sharing
const PAYLOAD_BYTES: usize = 1 << 20; // 1 MiB staged copy (mode 2)
const REPS_SIGNAL: usize = 1000;
const REPS_COPY: usize = 300;
const WRITE_DEFAULT: u32 = 0; // CU_STREAM_WRITE_VALUE_DEFAULT

const KERNEL_SRC: &str = r#"
// job descriptor written straight into HOST-MAPPED pinned memory
extern "C" __global__ void producer(volatile unsigned int* __restrict__ desc_host,
                                    const unsigned int* __restrict__ prm) {
    unsigned int slot = prm[0];
    volatile unsigned int* d = desc_host + slot * 16;
    if (threadIdx.x == 0) {
        d[0] = prm[1];             // layer
        d[1] = prm[2];             // kind
        d[2] = prm[3];             // n_cold
        d[3] = prm[4];             // seq
        for (int j = 0; j < 10; j++) d[4 + j] = prm[1] * 7u + (unsigned)j * 13u + slot * 31u;
    }
    __threadfence_system();
}

// fallback publication: flag written from device after the descriptor is visible
extern "C" __global__ void publish_flag(volatile unsigned int* __restrict__ pub_host,
                                        const unsigned int* __restrict__ prm) {
    if (threadIdx.x == 0) {
        unsigned int slot = prm[0];
        __threadfence_system();
        pub_host[slot * 16] = prm[4];
    }
}

// WDDM wait: polls the HOST-MAPPED done flag (mapped into device space) with a
// nanosleep backoff, then runs. This replaces cuStreamWaitValue32, which is
// unsupported/crashes on WDDM (see main() notes) — one tiny launch instead of
// a driver memop, still zero host involvement in the release path.
extern "C" __global__ void consumer_poll(const volatile unsigned int* __restrict__ done_host,
                                         const float* __restrict__ payload,
                                         float* __restrict__ result,
                                         const unsigned int* __restrict__ prm) {
    unsigned int slot = prm[0];
    unsigned int seq = prm[4];
    const volatile unsigned int* f = done_host + slot * 16;
    unsigned int spins = 0;
    while (*f != seq) {
        ++spins;
        if ((spins & 1023u) == 0) __nanosleep(250);
    }
    if (threadIdx.x == 0) {
        result[slot] = payload[0] + (float)seq;
    }
}

// memops-path consumer: runs BEHIND cuStreamWaitValue64_v2, no polling needed
extern "C" __global__ void consumer_memop(const float* __restrict__ payload,
                                          float* __restrict__ result,
                                          const unsigned int* __restrict__ prm) {
    unsigned int slot = prm[0];
    unsigned int seq = prm[4];
    if (threadIdx.x == 0) {
        result[slot] = payload[0] + (float)seq;
    }
}

// stage C consumer: runs behind the 64-bit wait (EQ 1). Reads the flag it was released on,
// counts the replay, re-arms the flag (-> 0) and publishes a start marker into mapped host
// memory: marker = (replay << 4) | (flag value seen & 0xf). The host times flag -> marker.
extern "C" __global__ void ring_consumer(unsigned long long* flag,
                                         unsigned long long* __restrict__ counter,
                                         volatile unsigned long long* marker_host) {
    if (threadIdx.x == 0) {
        unsigned long long f = *(volatile unsigned long long*)flag;
        unsigned long long s = *counter + 1ull;
        *counter = s;
        *(volatile unsigned long long*)flag = 0ull;
        __threadfence_system();
        *marker_host = (s << 4) | (f & 0xfull);
        __threadfence_system();
    }
}

// baseline for the timing table: empty kernel launch cost
extern "C" __global__ void noop() {}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

unsafe fn alloc_zeroed(bytes: usize) -> CUdeviceptr {
    let mut d: CUdeviceptr = 0;
    ck(sys::cuMemAlloc_v2(&mut d, bytes));
    ck(sys::cuMemsetD8_v2(d, 0, bytes));
    d
}

/// expected descriptor fields for job i (deterministic, mirrors the kernel)
fn expect(i: usize) -> [u32; 16] {
    let slot = (i % SLOTS) as u32;
    let layer = (i % 48) as u32;
    let seq = (i + 1) as u32;
    let mut d = [0u32; 16];
    d[0] = layer;
    d[1] = 1;
    d[2] = ((i % 10) + 1) as u32;
    d[3] = seq;
    for j in 0..10 {
        d[4 + j] = layer * 7 + (j as u32) * 13 + slot * 31;
    }
    d
}

#[derive(Clone)]
struct SendPtr(*mut std::ffi::c_void);
unsafe impl Send for SendPtr {}

/// everything the stager thread needs — all fields Send by construction
struct StagerArgs {
    host_pub: SendPtr,
    host_done: SendPtr,
    payload_src: SendPtr,
    ctx: SendPtr,
    copy_stream: SendPtr,
    done64: u64,
    payload_dst: u64,
    memops: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
}

fn stager_loop(a: StagerArgs) {
    let copy_stream: sys::CUstream = a.copy_stream.0 as *mut sys::CUstream_st;
    unsafe {
        ck(sys::cuCtxSetCurrent(a.ctx.0 as *mut sys::CUctx_st));
    }
    let mut mode_copy = false;
    for r in 0..(REPS_SIGNAL + REPS_COPY) {
        let slot = r % SLOTS;
        let seq = (r + 1) as u32;
        let flags = unsafe {
            std::slice::from_raw_parts(a.host_pub.0 as *const u32, SLOTS * PUB_STRIDE)
        };
        unsafe {
            while std::ptr::read_volatile(flags.as_ptr().add(slot * PUB_STRIDE)) != seq {
                if a.stop.load(Ordering::Relaxed) {
                    return;
                }
                std::hint::spin_loop();
            }
            if mode_copy {
                ck(sys::cuMemcpyHtoDAsync_v2(
                    a.payload_dst,
                    a.payload_src.0 as *const std::ffi::c_void,
                    PAYLOAD_BYTES,
                    copy_stream,
                ));
                ck(sys::cuStreamSynchronize(copy_stream));
            }
            if a.memops.load(Ordering::Relaxed) {
                let sv = seq as u64;
                ck(sys::cuMemcpyHtoDAsync_v2(
                    a.done64 + (slot * 8) as u64,
                    &sv as *const u64 as *const std::ffi::c_void,
                    8,
                    copy_stream,
                ));
            } else {
                let done_view = std::slice::from_raw_parts_mut(
                    a.host_done.0 as *mut u32,
                    SLOTS * PUB_STRIDE,
                );
                std::ptr::write_volatile(&mut done_view[slot * PUB_STRIDE], seq);
            }
        }
        if r == REPS_SIGNAL - 1 {
            mode_copy = true;
        }
    }
}

fn pct(v: &mut Vec<f64>, p: f64) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = (((v.len() as f64) * p) as usize).min(v.len() - 1);
    v[idx]
}

fn main() {
    unsafe {
        ck(sys::cuInit(0));
        let memops = load_memops();
        let mut dev = 0;
        ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx = std::ptr::null_mut();
        ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        ck(sys::cuCtxSetCurrent(ctx));

        // ---- pinned host ring (PORTABLE | DEVICEMAP) + device views ----
        let mut host_desc: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut host_desc,
            SLOTS * DESC_U32 * 4,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
        ));
        std::ptr::write_bytes(host_desc as *mut u8, 0, SLOTS * DESC_U32 * 4);
        let mut host_pub: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut host_pub,
            SLOTS * PUB_STRIDE * 4,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
        ));
        std::ptr::write_bytes(host_pub as *mut u8, 0, SLOTS * PUB_STRIDE * 4);

        let mut desc_dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut desc_dev, host_desc, 0));
        // WRITE-COMBINED: the GPU polls this buffer, so host writes must reach
        // DRAM uncached — otherwise the GPU's mapped read can serve a stale line
        let mut host_done: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut host_done,
            SLOTS * PUB_STRIDE * 4,
            sys::CU_MEMHOSTALLOC_PORTABLE
                | sys::CU_MEMHOSTALLOC_DEVICEMAP
                | sys::CU_MEMHOSTALLOC_WRITECOMBINED,
        ));
        std::ptr::write_bytes(host_done as *mut u8, 0, SLOTS * PUB_STRIDE * 4);
        let mut pub_dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut pub_dev, host_pub, 0));
        let mut done_dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut done_dev, host_done, 0));
        println!("p9: ring allocated — descriptors {} B, pub flags {} B (pinned, device-mapped)", SLOTS * 64, SLOTS * 64);

        // device-side state
        let payload_src = {
            let mut p: *mut std::ffi::c_void = std::ptr::null_mut();
            ck(sys::cuMemHostAlloc(
                &mut p,
                PAYLOAD_BYTES,
                sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
            ));
            let f = p as *mut f32;
            for k in 0..PAYLOAD_BYTES / 4 {
                *f.add(k) = (k % 977) as f32 * 0.5;
            }
            p
        };
        let mut payload_src_dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut payload_src_dev, payload_src, 0));
        let mut payload_dst = alloc_zeroed(PAYLOAD_BYTES);
        let mut result = alloc_zeroed(SLOTS * 4);
        let mut params = alloc_zeroed(8 * 4);

        // streams: dedicated compute + copy stream (never the legacy NULL stream —
        // stream memops on WDDM are a suspected offender there)
        let mut copy_stream: sys::CUstream = std::ptr::null_mut();
        ck(sys::cuStreamCreate(&mut copy_stream, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32));
        let mut compute: sys::CUstream = std::ptr::null_mut();
        ck(sys::cuStreamCreate(&mut compute, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32));

        // 64-bit _v2 memops (cudarc exposes these; handoff-bench proved them on WDDM)
        let mut done64 = alloc_zeroed(SLOTS * 8); // u64 completion flags, device
        let memops64 = {
            let probe = alloc_zeroed(8);
            let v: u64 = 0xDEAD_BEEF;
            let w = sys::cuStreamWriteValue64_v2(std::ptr::null_mut(), probe, v, 0);
            let _ = sys::cuStreamSynchronize(std::ptr::null_mut());
            let mut back = 0u64;
            let g = sys::cuMemcpyDtoH_v2(
                &mut back as *mut u64 as *mut std::ffi::c_void,
                probe,
                8,
            );
            w == CUresult::CUDA_SUCCESS && g == CUresult::CUDA_SUCCESS && back == v
        };
        println!("p9: 64-bit _v2 memops on device memory: {}", if memops64 { "AVAILABLE" } else { "unavailable" });

        // ---- compile ----
        let opts = CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).expect("nvrtc");
        let c_ptx = CString::new(ptx.to_src()).unwrap();
        let mut module = std::ptr::null_mut();
        ck(sys::cuModuleLoadData(&mut module, c_ptx.as_ptr() as *const _));
        let get_fn = |n: &str| {
            let mut fu: CUfunction = std::ptr::null_mut();
            ck(sys::cuModuleGetFunction(
                &mut fu,
                module,
                CString::new(n).unwrap().as_ptr(),
            ));
            fu
        };
        let f_producer = get_fn("producer");
        let f_publish = get_fn("publish_flag");
        let f_consumer = get_fn("consumer_poll");
        let f_consumer_memop = get_fn("consumer_memop");
        let f_noop = get_fn("noop");
        let f_ring = get_fn("ring_consumer");

        let launch1 = |f: CUfunction, args: &mut [*mut std::ffi::c_void]| {
            ck(sys::cuLaunchKernel(
                f,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                std::ptr::null_mut(),
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ));
        };
        let launch1s = |f: CUfunction, stream: sys::CUstream, args: &mut [*mut std::ffi::c_void]| {
            ck(sys::cuLaunchKernel(
                f,
                1,
                1,
                1,
                32,
                1,
                1,
                0,
                stream,
                args.as_mut_ptr(),
                std::ptr::null_mut(),
            ));
        };

        // scalar params per job travel via device buffer (stream-ordered H2D)
        fn job_params(prm: CUdeviceptr, compute: sys::CUstream, slot: u32, layer: u32, kind: u32, n_cold: u32, seq: u32) {
            let v = [slot, layer, kind, n_cold, seq, 0, 0, 0];
            ck(unsafe { sys::cuMemcpyHtoDAsync_v2(prm, v.as_ptr() as *const std::ffi::c_void, 32, compute) });
        }

        // ================= Stage A — mechanics =================
        println!("p9: stage A — 2 laps × {SLOTS} slots, descriptor + flag mechanics");
        let mut publish_path_kernel = false;
        let lap_jobs = SLOTS * 2;
        for i in 0..lap_jobs {
            let slot = (i % SLOTS) as u32;
            let seq = (i + 1) as u32;
            let e = expect(i);
            job_params(params, compute, slot, e[0], e[1], e[2], seq);
            launch1s(f_producer, compute, &mut [
                &mut desc_dev as *mut _ as *mut _,
                &mut params as *mut _ as *mut _,
            ]);
            // publication: try the spec's cuStreamWriteValue32 onto host-mapped flag
            // ONCE (job 0 only) — on this driver it returns NOT_SUPPORTED, and the
            // device-memory variant is not attempted: it crashes WDDM outright
            // (access violation observed on 616.56). Kernel-side mapped publication
            // is the WDDM path (deviation from spec wording documented).
            if !publish_path_kernel && i == 0 {
                let r = (memops.write32)(
                    compute,
                    pub_dev + (slot as usize * PUB_STRIDE * 4) as u64,
                    seq,
                    WRITE_DEFAULT,
                );
                if r != CUresult::CUDA_SUCCESS {
                    println!("p9: cuStreamWriteValue32 on host-mapped flag: {r:?} — 32-bit non-_v2 memops rejected on WDDM (device-target variant even crashes); trying the 64-bit _v2 variants (handoff-bench precedent)");
                    publish_path_kernel = true;
                }
            }
            if publish_path_kernel {
                launch1s(f_publish, compute, &mut [
                    &mut pub_dev as *mut _ as *mut _,
                    &mut params as *mut _ as *mut _,
                ]);
            }
            ck(sys::cuStreamSynchronize(compute));

            // host verifies the descriptor became visible
            let d = std::slice::from_raw_parts(host_desc as *const u32, SLOTS * DESC_U32);
            let base = slot as usize * DESC_U32;
            assert_eq!(&d[base..base + 14], &e[0..14], "descriptor mismatch at job {i}");
            let pubr = std::slice::from_raw_parts(host_pub as *const u32, SLOTS * PUB_STRIDE);
            assert_eq!(pubr[slot as usize * PUB_STRIDE], seq, "pub flag mismatch at job {i}");

            // done flag raised as a stream op onto DEVICE memory (supported
            // everywhere); the host-mapped question lives on the PUB side only
            // release: memops64 path enqueues waitvalue64_v2 + raises via a
            // stream write; poll path enqueues the spinning consumer and the
            // host raises via volatile write into the WRITE-COMBINED buffer
            if memops64 {
                ck(sys::cuStreamWaitValue64_v2(
                    compute,
                    done64 + (slot as usize * 8) as u64,
                    seq as u64,
                    0, // GEQ: monotonic sequences, never reset
                ));
                launch1s(f_consumer_memop, compute, &mut [
                    &mut payload_dst as *mut _ as *mut _,
                    &mut result as *mut _ as *mut _,
                    &mut params as *mut _ as *mut _,
                ]);
                let sv = seq as u64;
                ck(sys::cuMemcpyHtoDAsync_v2(
                    done64 + (slot as usize * 8) as u64,
                    &sv as *const u64 as *const std::ffi::c_void,
                    8,
                    copy_stream,
                ));
            } else {
                launch1s(f_consumer, compute, &mut [
                    &mut done_dev as *mut _ as *mut _,
                    &mut payload_dst as *mut _ as *mut _,
                    &mut result as *mut _ as *mut _,
                    &mut params as *mut _ as *mut _,
                ]);
                let done_view = std::slice::from_raw_parts_mut(
                    host_done as *mut u32,
                    SLOTS * PUB_STRIDE,
                );
                std::ptr::write_volatile(&mut done_view[slot as usize * PUB_STRIDE], seq);
            }
            ck(sys::cuStreamSynchronize(compute));
            let mut rv = 0f32;
            ck(sys::cuMemcpyDtoH_v2(
                &mut rv as *mut f32 as *mut std::ffi::c_void,
                result + (slot as usize * 4) as u64,
                4,
            ));
            assert_eq!(rv, seq as f32, "consumer result wrong at job {i}");
        }
        println!(
            "p9: stage A PASS — mapped descriptor writes, fences, seq reuse ({lap_jobs} jobs), wait/release verified (publish path: {})",
            if publish_path_kernel { "kernel-side" } else { "cuStreamWriteValue32 on host-mapped flag" }
        );

        // ================= Stage B — round trip =================
        println!("p9: stage B — round-trip latency (sync per rep, stager on own thread)");
        let host_pub_ptr = SendPtr(host_pub);
        let host_done_ptr = SendPtr(host_done);
        let copy_stream_ptr = SendPtr(copy_stream as *mut std::ffi::c_void);
        let done64_for_stager: u64 = done64;
        let payload_dst_for_stager: u64 = payload_dst;
        let payload_src_ptr = SendPtr(payload_src);
        let ctx_ptr = SendPtr(ctx as *mut std::ffi::c_void);
        let memops_mode = Arc::new(AtomicBool::new(memops64));
        let stager_memops = memops_mode.clone();
        // stager is spawned per release round below (its loop is finite per round)

        // baseline first (no ring): empty kernel + sync
        let mut baseline: Vec<f64> = Vec::new();
        for _ in 0..300 {
            let t0 = Instant::now();
            launch1(f_noop, &mut []);
            ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            baseline.push(t0.elapsed().as_secs_f64() * 1e6);
        }

        // release-path loop: when 64-bit memops exist, time BOTH paths
        let mut release_rounds: Vec<(&str, bool)> = Vec::new();
        if memops64 {
            release_rounds.push(("waitvalue64_v2 (memops)", true));
        }
        release_rounds.push(("poll-consumer (WDDM fallback)", false));
        for (label, use_memops) in release_rounds {
            memops_mode.store(use_memops, Ordering::Relaxed);
            let consumer = if use_memops { f_consumer_memop } else { f_consumer };
            let mut mode1: Vec<f64> = Vec::new();
            let mut mode2: Vec<f64> = Vec::new();
            let round_stop = Arc::new(AtomicBool::new(false));
            let stager_stop = round_stop.clone();
            let stager = {
                let stager_memops = stager_memops.clone();
                let stager_stop = stager_stop.clone();
                let hp = host_pub_ptr.clone();
                let hd = host_done_ptr.clone();
                let ps = payload_src_ptr.clone();
                let cx = ctx_ptr.clone();
                let cs = copy_stream_ptr.clone();
                std::thread::spawn(move || {
                    stager_loop(StagerArgs {
                        host_pub: hp,
                        host_done: hd,
                        payload_src: ps,
                        ctx: cx,
                        copy_stream: cs,
                        done64: done64_for_stager,
                        payload_dst: payload_dst_for_stager,
                        memops: stager_memops,
                        stop: stager_stop,
                    })
                })
            };
            for r in 0..(REPS_SIGNAL + REPS_COPY) {
                let slot = (r % SLOTS) as u32;
                let seq = (r + 1) as u32;
                let e = expect(r);
                let mode_copy = r >= REPS_SIGNAL;
                let t0 = Instant::now();
                if r == 0 {
                }
                job_params(params, compute, slot, e[0], e[1], e[2], seq);
                launch1s(
                    f_producer,
                    compute,
                    &mut [
                        &mut desc_dev as *mut _ as *mut _,
                        &mut params as *mut _ as *mut _,
                    ],
                );
                launch1s(
                    f_publish,
                    compute,
                    &mut [
                        &mut pub_dev as *mut _ as *mut _,
                        &mut params as *mut _ as *mut _,
                    ],
                );
                if use_memops {
                    ck(sys::cuStreamWaitValue64_v2(
                        compute,
                        done64 + (slot as usize * 8) as u64,
                        seq as u64,
                        0,
                    ));
                    // consumer_memop takes (payload, result, prm) — no done flag
                    launch1s(consumer, compute, &mut [
                        &mut payload_dst as *mut _ as *mut _,
                        &mut result as *mut _ as *mut _,
                        &mut params as *mut _ as *mut _,
                    ]);
                } else {
                    launch1s(consumer, compute, &mut [
                        &mut done_dev as *mut _ as *mut _,
                        &mut payload_dst as *mut _ as *mut _,
                        &mut result as *mut _ as *mut _,
                        &mut params as *mut _ as *mut _,
                    ]);
                }
                if r == 0 {
                }
                ck(sys::cuStreamSynchronize(compute));
                if r == 0 {
                }
                let us = t0.elapsed().as_secs_f64() * 1e6;
                if mode_copy {
                    mode2.push(us);
                } else {
                    mode1.push(us);
                }
                let mut rv = 0f32;
                ck(sys::cuMemcpyDtoH_v2(
                    &mut rv as *mut f32 as *mut std::ffi::c_void,
                    result + (slot as usize * 4) as u64,
                    4,
                ));
                assert_eq!(rv, seq as f32, "round trip {r} ({label}): consumer result wrong");
                if r % 200 == 0 {
                }
            }
            println!("p9: release path = {label}");
            for (name, v) in [("signaling-only", &mut mode1), ("with 1 MiB staged copy", &mut mode2)] {
                println!(
                    "p9:   {name:<24} n={:4}  p50={:8.1} µs  p95={:8.1} µs  p99={:8.1} µs  max={:8.1} µs",
                    v.len(),
                    pct(v, 0.50),
                    pct(v, 0.95),
                    pct(v, 0.99),
                    v.last().unwrap()
                );
            }
            round_stop.store(true, Ordering::Relaxed);
            let _ = stager.join();
        }

        println!("p9: baseline launch+sync p50 = {:.1} µs", pct(&mut baseline, 0.50));

        // ================= Stage C — WaitValue64_v2 inside a CUDA graph (crow-nest #149, step 17a) =================
        let (stage_c, stage_c_green) = if !memops64 {
            println!("p9: stage C skipped — 64-bit _v2 memops unavailable on this device");
            ("skipped (no 64-bit memops)".to_string(), false)
        } else {
            stage_c(ctx, dev, compute, copy_stream, f_ring)
        };

        println!(
            "p9: reference points — swapped barrier 4–6 ms (#186); WDDM ring round trips 3.1–3.4 ms/layer (measured 2026-09-01, retired variant A)"
        );
        println!("p9: PASS — job ring + stream memops verified end to end on Windows (stage A mechanics + stage B timing)");
        println!("p9: stage C (WaitValue64_v2 in a CUDA graph) — {stage_c}");
        if !stage_c_green {
            std::process::exit(3);
        }
    }
}

// ===================== stage C (crow-nest #149, plan step 17a) =====================

#[derive(Clone, Copy, PartialEq, Debug)]
enum FlagAt {
    Device,
    Host,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Build {
    NoWait,
    Uncaptured,
    Captured,
    Explicit,
}

/// stage C addresses as plain integers, so the writer thread can own a copy
#[derive(Clone, Copy)]
struct Flags {
    host_flag: usize,   // host address of the mapped u64 flag
    host_flag_dev: u64, // its device address
    dev_flag: u64,      // u64 flag in device memory
    one_pinned: usize,  // pinned u64 = 1, H2D source when the device flag is raised
    marker: usize,      // host address of the consumer's start marker (mapped)
    marker_dev: u64,
    copy_stream: usize,
    ctx: usize,
}

impl Flags {
    fn ptr(&self, at: FlagAt) -> u64 {
        match at {
            FlagAt::Device => self.dev_flag,
            FlagAt::Host => self.host_flag_dev,
        }
    }
    unsafe fn marker(&self) -> u64 {
        unsafe { std::ptr::read_volatile(self.marker as *const u64) }
    }
    unsafe fn raise(&self, at: FlagAt) {
        unsafe {
            match at {
                FlagAt::Host => {
                    std::ptr::write_volatile(self.host_flag as *mut u64, 1);
                    std::sync::atomic::fence(Ordering::SeqCst);
                }
                FlagAt::Device => {
                    let cs = self.copy_stream as sys::CUstream;
                    ck(sys::cuMemcpyHtoDAsync_v2(self.dev_flag, self.one_pinned as *const std::ffi::c_void, 8, cs));
                    let _ = sys::cuStreamQuery(cs); // WDDM batches submissions: push the copy out now
                }
            }
        }
    }
}

/// hand-off between the launching thread and the flag writer
struct Writer {
    ticket: AtomicU64, // bumped once per launch; the writer serves each ticket once
    seq: AtomicU64,    // replay number the consumer will publish
    delay_us: AtomicU64,
    at_host: AtomicBool,
    done: AtomicU64, // ticket answered
    status: AtomicU32,
    lat_ns: AtomicU64,
    stop: AtomicBool,
}

fn writer_loop(w: Arc<Writer>, f: Flags) {
    unsafe {
        ck(sys::cuCtxSetCurrent(f.ctx as sys::CUcontext));
    }
    let mut last = 0u64;
    loop {
        let ticket = loop {
            if w.stop.load(Ordering::Acquire) {
                return;
            }
            let t = w.ticket.load(Ordering::Acquire);
            if t != last {
                break t;
            }
            std::hint::spin_loop();
        };
        last = ticket;
        let seq = w.seq.load(Ordering::Relaxed);
        let at = if w.at_host.load(Ordering::Relaxed) { FlagAt::Host } else { FlagAt::Device };
        let delay = std::time::Duration::from_micros(w.delay_us.load(Ordering::Relaxed));
        let t_arm = Instant::now();
        while t_arm.elapsed() < delay {
            std::hint::spin_loop();
        }
        let (status, lat) = unsafe {
            if f.marker() >> 4 == seq {
                (ST_EARLY, 0) // the consumer ran before the flag: the wait did not hold
            } else {
                let t_w = Instant::now();
                f.raise(at);
                loop {
                    if f.marker() >> 4 == seq {
                        break (ST_RELEASED, t_w.elapsed().as_nanos() as u64);
                    }
                    if t_w.elapsed().as_millis() as u64 >= GUARD_MS {
                        break (ST_NO_START, 0);
                    }
                    std::hint::spin_loop();
                }
            }
        };
        w.status.store(status, Ordering::Relaxed);
        w.lat_ns.store(lat, Ordering::Relaxed);
        w.done.store(ticket, Ordering::Release);
    }
}

struct StageC {
    g: GraphApi,
    compute: sys::CUstream,
    f_ring: CUfunction,
    counter: CUdeviceptr,
    flags: Flags,
    w: Arc<Writer>,
    ticket: std::cell::Cell<u64>,
}

struct Row {
    label: String,
    build: String,
    hold: Option<bool>,
    n: usize,
    ok: usize,
    early: usize,
    lat: Vec<f64>,
    api: Vec<f64>,
    err: Option<String>,
}

impl Row {
    fn green(&self) -> bool {
        self.err.is_none() && self.hold == Some(true) && self.n == REPS_GRAPH && self.ok == REPS_GRAPH
    }
}

const EQ: u32 = sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_EQ as u32;

unsafe fn launch_ring(c: &StageC, at: FlagAt) -> CUresult {
    let mut a: [u64; 3] = [c.flags.ptr(at), c.counter, c.flags.marker_dev];
    let mut k: [*mut std::ffi::c_void; 3] = [
        &mut a[0] as *mut _ as *mut std::ffi::c_void,
        &mut a[1] as *mut _ as *mut std::ffi::c_void,
        &mut a[2] as *mut _ as *mut std::ffi::c_void,
    ];
    unsafe { sys::cuLaunchKernel(c.f_ring, 1, 1, 1, 32, 1, 1, 0, c.compute, k.as_mut_ptr(), std::ptr::null_mut()) }
}

unsafe fn sync_bounded(s: sys::CUstream, ms: u64) -> Result<(), String> {
    let t = Instant::now();
    loop {
        let q = unsafe { sys::cuStreamQuery(s) };
        if q == CUresult::CUDA_SUCCESS {
            return Ok(());
        }
        if q != CUresult::CUDA_ERROR_NOT_READY {
            return Err(format!("cuStreamQuery {q:?}"));
        }
        if t.elapsed().as_millis() as u64 >= ms {
            return Err(format!("stream not drained {ms} ms later"));
        }
        std::hint::spin_loop();
    }
}

/// C1 / C2: build and instantiate the graph; Ok((graph, exec, node description))
unsafe fn build_graph(c: &StageC, build: Build, at: FlagAt) -> Result<(sys::CUgraph, sys::CUgraphExec, String), String> {
    unsafe {
        let g = &c.g;
        let mut graph: sys::CUgraph = std::ptr::null_mut();
        if build == Build::Captured {
            let r_b = (g.begin_capture)(c.compute, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if r_b != CUresult::CUDA_SUCCESS {
                return Err(format!("cuStreamBeginCapture_v2 {r_b:?}"));
            }
            let r_w = sys::cuStreamWaitValue64_v2(c.compute, c.flags.ptr(at), 1, EQ);
            let r_k = launch_ring(c, at);
            let r_e = (g.end_capture)(c.compute, &mut graph); // always end, also after a failed op
            if r_w != CUresult::CUDA_SUCCESS || r_k != CUresult::CUDA_SUCCESS || r_e != CUresult::CUDA_SUCCESS {
                if !graph.is_null() {
                    let _ = (g.destroy)(graph);
                }
                return Err(format!("capture: cuStreamWaitValue64_v2 {r_w:?}, kernel {r_k:?}, cuStreamEndCapture {r_e:?}"));
            }
        } else {
            ck((g.create)(&mut graph, 0));
            let mut op: sys::CUstreamBatchMemOpParams = std::mem::zeroed();
            op.waitValue.operation = sys::CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WAIT_VALUE_64;
            op.waitValue.address = c.flags.ptr(at);
            op.waitValue.__bindgen_anon_1.value64 = 1;
            op.waitValue.flags = EQ;
            let prm = sys::CUDA_BATCH_MEM_OP_NODE_PARAMS { ctx: c.flags.ctx as sys::CUcontext, count: 1, paramArray: &mut op, flags: 0 };
            let mut wait_node: sys::CUgraphNode = std::ptr::null_mut();
            let r_add = (g.add_batch_memop)(&mut wait_node, graph, std::ptr::null(), 0, &prm);
            if r_add != CUresult::CUDA_SUCCESS {
                let _ = (g.destroy)(graph);
                return Err(format!("cuGraphAddBatchMemOpNode {r_add:?}"));
            }
            let mut a: [u64; 3] = [c.flags.ptr(at), c.counter, c.flags.marker_dev];
            let mut k: [*mut std::ffi::c_void; 3] = [
                &mut a[0] as *mut _ as *mut std::ffi::c_void,
                &mut a[1] as *mut _ as *mut std::ffi::c_void,
                &mut a[2] as *mut _ as *mut std::ffi::c_void,
            ];
            let kp = sys::CUDA_KERNEL_NODE_PARAMS_v2 {
                func: c.f_ring,
                gridDimX: 1,
                gridDimY: 1,
                gridDimZ: 1,
                blockDimX: 32,
                blockDimY: 1,
                blockDimZ: 1,
                sharedMemBytes: 0,
                kernelParams: k.as_mut_ptr(),
                extra: std::ptr::null_mut(),
                kern: std::ptr::null_mut(),
                ctx: std::ptr::null_mut(),
            };
            let mut kernel_node: sys::CUgraphNode = std::ptr::null_mut();
            let r_kn = (g.add_kernel_node)(&mut kernel_node, graph, &wait_node, 1, &kp);
            if r_kn != CUresult::CUDA_SUCCESS {
                let _ = (g.destroy)(graph);
                return Err(format!("cuGraphAddKernelNode_v2 {r_kn:?}"));
            }
        }
        let mut n = 0usize;
        ck((g.get_nodes)(graph, std::ptr::null_mut(), &mut n));
        let mut nodes: Vec<sys::CUgraphNode> = vec![std::ptr::null_mut(); n];
        ck((g.get_nodes)(graph, nodes.as_mut_ptr(), &mut n));
        let types: Vec<u32> = nodes
            .iter()
            .map(|&nd| {
                let mut t = u32::MAX;
                ck((g.node_type)(nd, &mut t));
                t
            })
            .collect();
        let desc = format!(
            "{n} nodes {types:?} ({})",
            if types.contains(&NODE_BATCH_MEM_OP) { "wait = BATCH_MEM_OP" } else { "NO BATCH_MEM_OP node" }
        );
        let mut exec: sys::CUgraphExec = std::ptr::null_mut();
        let r_inst = (g.instantiate)(&mut exec, graph, 0);
        if r_inst != CUresult::CUDA_SUCCESS {
            let _ = (g.destroy)(graph);
            return Err(format!("{desc}; cuGraphInstantiateWithFlags {r_inst:?}"));
        }
        Ok((graph, exec, desc))
    }
}

/// one variant: hold check (replay 1) + replays; Err = the TDR guard fired (caller exits 2)
unsafe fn run_variant(c: &StageC, label: &str, build: Build, at: FlagAt, exec: sys::CUgraphExec, reps: usize, build_desc: String) -> Result<Row, Row> {
    unsafe {
        let mut row = Row {
            label: label.to_string(),
            build: build_desc,
            hold: None,
            n: 0,
            ok: 0,
            early: 0,
            lat: Vec::with_capacity(reps),
            api: Vec::with_capacity(reps),
            err: None,
        };
        // reset: flag 0, counter 0, marker 0
        std::ptr::write_volatile(c.flags.host_flag as *mut u64, 0);
        ck(sys::cuMemsetD8_v2(c.flags.dev_flag, 0, 8));
        ck(sys::cuMemsetD8_v2(c.counter, 0, 8));
        ck(sys::cuCtxSynchronize());
        std::ptr::write_volatile(c.flags.marker as *mut u64, 0);
        c.w.at_host.store(at == FlagAt::Host, Ordering::Relaxed);
        let waits = build != Build::NoWait;
        for rep in 0..reps {
            let seq = (rep + 1) as u64;
            let hold = rep == 0;
            let delay = if hold { HOLD_DELAY_US } else { WRITER_DELAY_US };
            let t0 = Instant::now();
            let r = match build {
                Build::NoWait => launch_ring(c, at),
                Build::Uncaptured => {
                    let r = sys::cuStreamWaitValue64_v2(c.compute, c.flags.ptr(at), 1, EQ);
                    if r != CUresult::CUDA_SUCCESS {
                        r
                    } else {
                        launch_ring(c, at)
                    }
                }
                Build::Captured | Build::Explicit => (c.g.launch)(exec, c.compute),
            };
            if r != CUresult::CUDA_SUCCESS {
                row.err = Some(format!("launch {r:?} at replay {rep}"));
                // a wait may have been enqueued before the failing call: raise and drain
                c.flags.raise(at);
                let _ = sync_bounded(c.compute, GUARD_MS);
                return Ok(row);
            }
            row.api.push(t0.elapsed().as_secs_f64() * 1e6);
            let q0 = sys::cuStreamQuery(c.compute); // flush the WDDM batch; NOT_READY = the wait holds
            let ticket = c.ticket.get() + 1;
            c.ticket.set(ticket);
            c.w.seq.store(seq, Ordering::Relaxed);
            c.w.delay_us.store(delay, Ordering::Relaxed);
            c.w.ticket.store(ticket, Ordering::Release);
            if hold {
                std::thread::sleep(std::time::Duration::from_millis(20));
                let q = sys::cuStreamQuery(c.compute);
                let m = c.flags.marker();
                let held = q == CUresult::CUDA_ERROR_NOT_READY && m >> 4 != seq;
                row.hold = Some(held);
                println!(
                    "p9:   {label}: hold check, flag 0 for 20 ms after launch: cuStreamQuery {q:?}, start marker {} -> {}",
                    if m >> 4 == seq { "present" } else { "absent" },
                    if held { "held" } else { "NOT held" }
                );
            }
            let deadline = std::time::Duration::from_micros(delay) + std::time::Duration::from_millis(GUARD_MS + 50);
            let ta = Instant::now();
            while c.w.done.load(Ordering::Acquire) != ticket {
                if ta.elapsed() > deadline {
                    c.flags.raise(at); // rescue: the flag is up no matter what the writer does
                    row.err = Some(format!("writer did not answer replay {rep}; flag raised by the main thread"));
                    let d = sync_bounded(c.compute, GUARD_MS);
                    println!("p9:   {label}: FAIL — {} (drain: {d:?})", row.err.as_ref().unwrap());
                    return Err(row);
                }
                std::hint::spin_loop();
            }
            let status = c.w.status.load(Ordering::Relaxed);
            if status == ST_NO_START {
                row.err = Some(format!("consumer not started {GUARD_MS} ms after the flag (replay {rep}, q0 {q0:?})"));
                let d = sync_bounded(c.compute, GUARD_MS);
                println!("p9:   {label}: FAIL — {} (drain: {d:?})", row.err.as_ref().unwrap());
                return Err(row);
            }
            if let Err(e) = sync_bounded(c.compute, GUARD_MS) {
                row.err = Some(format!("replay {rep}: {e}"));
                println!("p9:   {label}: FAIL — {}", row.err.as_ref().unwrap());
                return Err(row);
            }
            row.n += 1;
            let m = c.flags.marker();
            if status == ST_EARLY {
                row.early += 1;
            } else {
                row.lat.push(c.w.lat_ns.load(Ordering::Relaxed) as f64 / 1e3);
            }
            // ordering: released by the writer, the consumer saw flag == 1, and it is replay `seq`
            if waits && status == ST_RELEASED && m >> 4 == seq && m & 0xf == 1 {
                row.ok += 1;
            }
            if !waits && status == ST_EARLY && m >> 4 == seq {
                row.ok += 1;
            }
        }
        Ok(row)
    }
}

fn stats(v: &[f64]) -> String {
    if v.is_empty() {
        return format!("{:>33}", "—");
    }
    let mut s = v.to_vec();
    let p50 = pct(&mut s, 0.50);
    let p95 = pct(&mut s, 0.95);
    let p99 = pct(&mut s, 0.99);
    format!("{p50:7.1} {p95:7.1} {p99:7.1} {:8.1}", s.last().unwrap())
}

fn print_rows(rows: &[Row]) {
    println!("p9: stage C results (wait->start = writer's flag store -> consumer's start marker seen by the host, µs)");
    println!(
        "p9:   {:<32} {:<5} {:>9} {:>5}   {:>7} {:>7} {:>7} {:>8}   {:>8}   build",
        "variant", "hold", "ok/n", "early", "p50", "p95", "p99", "max", "API p50"
    );
    for r in rows {
        let api = if r.api.is_empty() { "—".to_string() } else { format!("{:8.1}", pct(&mut r.api.clone(), 0.5)) };
        println!(
            "p9:   {:<32} {:<5} {:>9} {:>5}   {}   {:>8}   {}{}",
            r.label,
            match r.hold {
                Some(true) => "yes",
                Some(false) => "no",
                None => "—",
            },
            format!("{}/{}", r.ok, r.n),
            r.early,
            stats(&r.lat),
            api,
            r.build,
            r.err.as_ref().map(|e| format!("  ERROR: {e}")).unwrap_or_default()
        );
    }
}

unsafe fn stage_c(ctx: sys::CUcontext, dev: sys::CUdevice, compute: sys::CUstream, copy_stream: sys::CUstream, f_ring: CUfunction) -> (String, bool) {
    unsafe {
        let g = load_graph_api();
        let mut drv = 0i32;
        ck(sys::cuDriverGetVersion(&mut drv));
        let attrs: [(&str, i32); 8] = [
            ("TCC_DRIVER", 35),
            ("CAN_USE_HOST_POINTER_FOR_REGISTERED_MEM", 91),
            ("CAN_USE_STREAM_MEM_OPS_V1", 92),
            ("CAN_USE_64_BIT_STREAM_MEM_OPS_V1", 93),
            ("CAN_USE_STREAM_WAIT_VALUE_NOR_V1", 94),
            ("CAN_FLUSH_REMOTE_WRITES", 98),
            ("CAN_USE_64_BIT_STREAM_MEM_OPS (v2)", 122),
            ("CAN_USE_STREAM_WAIT_VALUE_NOR (v2)", 123),
        ];
        let attr_line: Vec<String> = attrs
            .iter()
            .map(|&(name, id)| {
                let mut v = -1i32;
                let r = (g.get_attr)(&mut v, id, dev);
                if r == CUresult::CUDA_SUCCESS {
                    format!("{name}={v}")
                } else {
                    format!("{name}: {r:?}")
                }
            })
            .collect();
        println!("p9: stage C — cuStreamWaitValue64_v2 inside a CUDA graph under WDDM (crow-nest #149, 17a); driver API {drv}");
        println!("p9:   device attributes: {}", attr_line.join(", "));

        // mapped pinned host block: flag at +0, start marker at +64 (own cache line)
        let mut host_blk: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(&mut host_blk, 4096, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP));
        std::ptr::write_bytes(host_blk as *mut u8, 0, 4096);
        let mut host_blk_dev: CUdeviceptr = 0;
        ck(sys::cuMemHostGetDevicePointer_v2(&mut host_blk_dev, host_blk, 0));
        let mut one: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(&mut one, 64, sys::CU_MEMHOSTALLOC_PORTABLE));
        *(one as *mut u64) = 1;
        let dev_flag = alloc_zeroed(8);
        let counter = alloc_zeroed(8);
        let flags = Flags {
            host_flag: host_blk as usize,
            host_flag_dev: host_blk_dev,
            dev_flag,
            one_pinned: one as usize,
            marker: host_blk as usize + 64,
            marker_dev: host_blk_dev + 64,
            copy_stream: copy_stream as usize,
            ctx: ctx as usize,
        };
        let w = Arc::new(Writer {
            ticket: AtomicU64::new(0),
            seq: AtomicU64::new(0),
            delay_us: AtomicU64::new(0),
            at_host: AtomicBool::new(false),
            done: AtomicU64::new(0),
            status: AtomicU32::new(0),
            lat_ns: AtomicU64::new(0),
            stop: AtomicBool::new(false),
        });
        let writer = {
            let w = w.clone();
            std::thread::spawn(move || writer_loop(w, flags))
        };
        let c = StageC { g, compute, f_ring, counter, flags, w, ticket: std::cell::Cell::new(0) };
        let mut rows: Vec<Row> = Vec::new();
        let abort = |rows: &mut Vec<Row>, r: Row| -> ! {
            rows.push(r);
            print_rows(rows);
            println!("p9: stage C ABORTED by the TDR guard (exit 2): a wait did not release within {GUARD_MS} ms of its flag");
            std::process::exit(2);
        };

        // negative control: the consumer without any wait must fail the hold and ordering checks
        match run_variant(&c, "control: no wait", Build::NoWait, FlagAt::Device, std::ptr::null_mut(), REPS_CONTROL, "consumer only".into()) {
            Ok(r) => {
                let control_ok = r.hold == Some(false) && r.early > 0;
                println!(
                    "p9:   control: hold {:?}, {} of {} replays ran before their flag -> the checks {}",
                    r.hold,
                    r.early,
                    r.n,
                    if control_ok { "detect a missing wait" } else { "do NOT detect a missing wait" }
                );
                rows.push(r);
                if !control_ok {
                    print_rows(&rows);
                    println!("p9: stage C FAIL — negative control not caught, results would prove nothing");
                    std::process::exit(1);
                }
            }
            Err(r) => abort(&mut rows, r),
        }

        // device flag first: its uncaptured form is the path stage B measures; host-mapped after
        for at in [FlagAt::Device, FlagAt::Host] {
            let loc = match at {
                FlagAt::Device => "device flag",
                FlagAt::Host => "host-mapped flag",
            };
            for build in [Build::Uncaptured, Build::Captured, Build::Explicit] {
                let label = format!(
                    "{loc}, {}",
                    match build {
                        Build::Uncaptured => "U uncaptured",
                        Build::Captured => "C1 captured",
                        _ => "C2 explicit",
                    }
                );
                let (graph, exec, desc) = if build == Build::Uncaptured {
                    (std::ptr::null_mut(), std::ptr::null_mut(), "stream: WaitValue64_v2 + launch".to_string())
                } else {
                    match build_graph(&c, build, at) {
                        Ok(x) => x,
                        Err(e) => {
                            println!("p9:   {label}: build failed — {e}");
                            rows.push(Row { label, build: String::new(), hold: None, n: 0, ok: 0, early: 0, lat: vec![], api: vec![], err: Some(e) });
                            continue;
                        }
                    }
                };
                println!("p9:   {label}: {desc}");
                let res = run_variant(&c, &label, build, at, exec, REPS_GRAPH, desc);
                if !exec.is_null() {
                    let _ = (c.g.exec_destroy)(exec);
                    let _ = (c.g.destroy)(graph);
                }
                match res {
                    Ok(r) => rows.push(r),
                    Err(r) => abort(&mut rows, r),
                }
            }
        }
        c.w.stop.store(true, Ordering::Release);
        let _ = writer.join();
        ck(sys::cuMemFree_v2(dev_flag));
        ck(sys::cuMemFree_v2(counter));
        ck(sys::cuMemFreeHost(host_blk));
        ck(sys::cuMemFreeHost(one));
        print_rows(&rows);

        let find = |l: &str| rows.iter().find(|r| r.label == l).map(|r| r.green()).unwrap_or(false);
        let mut parts = Vec::new();
        let mut green = false;
        for loc in ["device flag", "host-mapped flag"] {
            let (u, c1, c2) = (
                find(&format!("{loc}, U uncaptured")),
                find(&format!("{loc}, C1 captured")),
                find(&format!("{loc}, C2 explicit")),
            );
            green |= c1 || c2;
            let gr = |b: bool| if b { "green" } else { "red" };
            parts.push(format!("{loc}: U {} / C1 {} / C2 {}", gr(u), gr(c1), gr(c2)));
        }
        let line = format!("{} — {}", if green { "GREEN" } else { "RED" }, parts.join("; "));
        println!("p9: stage C verdict (17a): {line}");
        (line, green)
    }
}
