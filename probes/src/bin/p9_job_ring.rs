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
//! Stage C (crow-nest #149, plan step 17a, 2026-10-08): `cuStreamWaitValue64_v2` inside a CUDA
//! graph under WDDM. Two builds of the same wait, each instantiated and replayed:
//!   C1 stream capture: begin capture, `cuStreamWaitValue64_v2` (EQ 1 on a device flag), the
//!      consumer kernel, end capture; the graph's node types are printed and the wait must be
//!      a BATCH_MEM_OP node (type 12);
//!   C2 explicit node: `cuGraphAddBatchMemOpNode` with one WAIT_VALUE_64 op, the consumer
//!      launched behind the graph on the same stream.
//! Per build: a hold check (graph launched with the flag at 0, 20 ms later `cuStreamQuery` must
//! say NOT_READY, i.e. the replayed wait really holds the stream), then 1000 replays, each:
//! params H2D, graph launch, host raises the flag (8 B H2D on the copy stream), sync; the
//! consumer resets the flag in-graph. Every result is checked; p50/p95/p99 per replay printed.
//! A wait that never releases is reported after 5 s and the probe exits 1.
//!
//! Scalars travel in device buffers (p5 lesson); HtoD async + sync (WDDM rule).

use std::ffi::CString;
use std::sync::atomic::AtomicBool;
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

// stage C consumer: runs behind the graph's WaitValue64 node, then re-arms the flag
// in-graph (EQ 1 -> 0), so the next replay holds again until the host raises it
extern "C" __global__ void consumer_graph(unsigned long long* __restrict__ done_g,
                                          const float* __restrict__ payload,
                                          float* __restrict__ result,
                                          const unsigned int* __restrict__ prm) {
    if (threadIdx.x == 0) {
        unsigned int slot = prm[0];
        result[slot] = payload[0] + (float)prm[4];
        *done_g = 0ull;
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
        let f_consumer_graph = get_fn("consumer_graph");

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
        let stage_c = if !memops64 {
            println!("p9: stage C skipped — 64-bit _v2 memops unavailable on this device");
            "skipped (no 64-bit memops)".to_string()
        } else {
            let g = load_graph_api();
            let done_g = alloc_zeroed(8); // the stage-C flag: device u64, EQ 1 releases
            // copies of the device addresses: the closures below hold no borrow of the stage A/B state
            let (pd, rs, pm) = (payload_dst, result, params);
            let graph_launch = g.launch;
            let eq = sys::CUstreamWaitValue_flags::CU_STREAM_WAIT_VALUE_EQ as u32;
            let raise = move |copy_stream: sys::CUstream| {
                let one: u64 = 1;
                ck(sys::cuMemcpyHtoDAsync_v2(done_g, &one as *const u64 as *const std::ffi::c_void, 8, copy_stream));
                ck(sys::cuStreamSynchronize(copy_stream));
            };
            // sync with a deadline: a wait that never releases must not hang the probe silently
            let sync_or_die = move |what: &str| {
                let t0 = Instant::now();
                loop {
                    let q = sys::cuStreamQuery(compute);
                    if q == CUresult::CUDA_SUCCESS {
                        return;
                    }
                    if q != CUresult::CUDA_ERROR_NOT_READY {
                        panic!("p9: stage C {what}: cuStreamQuery {q:?}");
                    }
                    if t0.elapsed().as_secs_f64() > 5.0 {
                        println!("p9: stage C FAIL — {what}: the graph's wait did not release within 5 s after the flag was raised");
                        std::process::exit(1);
                    }
                    std::hint::spin_loop();
                }
            };
            // one hold check + REPS_GRAPH replays of an instantiated graph; `tail_consumer` launches
            // the consumer behind the graph (C2: the graph holds the wait node only)
            let replay = move |label: &str, exec: sys::CUgraphExec, tail_consumer: bool| -> bool {
                let launch_one = |seq: u32| {
                    job_params(pm, compute, 0, 0, 1, 1, seq);
                    ck(graph_launch(exec, compute));
                    if tail_consumer {
                        let mut a = [done_g, pd, rs, pm];
                        launch1s(f_consumer_graph, compute, &mut [
                            &mut a[0] as *mut _ as *mut _,
                            &mut a[1] as *mut _ as *mut _,
                            &mut a[2] as *mut _ as *mut _,
                            &mut a[3] as *mut _ as *mut _,
                        ]);
                    }
                };
                let check = |seq: u32, what: &str| {
                    let mut rv = 0f32;
                    ck(sys::cuMemcpyDtoH_v2(&mut rv as *mut f32 as *mut std::ffi::c_void, rs, 4));
                    assert_eq!(rv, seq as f32, "p9: stage C {label} {what}: consumer result wrong");
                };
                // hold check: flag at 0, the stream must still be blocked 20 ms after the launch
                launch_one(1);
                std::thread::sleep(std::time::Duration::from_millis(20));
                let q = sys::cuStreamQuery(compute);
                let held = q == CUresult::CUDA_ERROR_NOT_READY;
                println!(
                    "p9:   {label}: hold check — cuStreamQuery 20 ms after launch, flag 0: {q:?} ({})",
                    if held { "the replayed wait holds the stream" } else { "NOT held: the wait node did not block" }
                );
                raise(copy_stream);
                sync_or_die(&format!("{label} hold check"));
                check(1, "hold check");
                let mut lat = Vec::with_capacity(REPS_GRAPH);
                for r in 0..REPS_GRAPH {
                    let seq = (r + 2) as u32;
                    let t0 = Instant::now();
                    launch_one(seq);
                    raise(copy_stream);
                    sync_or_die(&format!("{label} replay {r}"));
                    lat.push(t0.elapsed().as_secs_f64() * 1e6);
                    check(seq, &format!("replay {r}"));
                }
                println!(
                    "p9:   {label}: {REPS_GRAPH} replays, results verified  p50={:8.1} µs  p95={:8.1} µs  p99={:8.1} µs  max={:8.1} µs",
                    pct(&mut lat, 0.50),
                    pct(&mut lat, 0.95),
                    pct(&mut lat, 0.99),
                    lat.last().unwrap()
                );
                held
            };

            println!("p9: stage C — cuStreamWaitValue64_v2 inside a CUDA graph (WDDM), hold check + {REPS_GRAPH} replays per build");
            // ---- C1: stream capture ----
            let c1;
            let r_begin = (g.begin_capture)(compute, sys::CUstreamCaptureMode::CU_STREAM_CAPTURE_MODE_THREAD_LOCAL);
            if r_begin != CUresult::CUDA_SUCCESS {
                c1 = format!("cuStreamBeginCapture_v2 {r_begin:?}");
            } else {
                let r_wait = sys::cuStreamWaitValue64_v2(compute, done_g, 1, eq);
                let mut a = [done_g, pd, rs, pm];
                let mut kargs: [*mut std::ffi::c_void; 4] = [
                    &mut a[0] as *mut _ as *mut std::ffi::c_void,
                    &mut a[1] as *mut _ as *mut std::ffi::c_void,
                    &mut a[2] as *mut _ as *mut std::ffi::c_void,
                    &mut a[3] as *mut _ as *mut std::ffi::c_void,
                ];
                let r_kernel = sys::cuLaunchKernel(
                    f_consumer_graph,
                    1,
                    1,
                    1,
                    32,
                    1,
                    1,
                    0,
                    compute,
                    kargs.as_mut_ptr(),
                    std::ptr::null_mut(),
                );
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                let r_end = (g.end_capture)(compute, &mut graph); // always end, also after a failed op
                println!("p9:   C1 capture: begin {r_begin:?}, WaitValue64_v2 {r_wait:?}, consumer {r_kernel:?}, end {r_end:?}");
                if r_wait == CUresult::CUDA_SUCCESS && r_kernel == CUresult::CUDA_SUCCESS && r_end == CUresult::CUDA_SUCCESS {
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
                    let memop_node = types.contains(&NODE_BATCH_MEM_OP);
                    println!(
                        "p9:   C1 graph: {n} nodes, types {types:?} (12 = BATCH_MEM_OP, 0 = KERNEL) — wait captured as a batch mem-op node: {}",
                        if memop_node { "YES" } else { "NO" }
                    );
                    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
                    let r_inst = (g.instantiate)(&mut exec, graph, 0);
                    println!("p9:   C1 instantiate: {r_inst:?}");
                    if r_inst == CUresult::CUDA_SUCCESS {
                        let held = replay("C1 captured", exec, false);
                        c1 = format!(
                            "captured {} node, instantiated, hold {}, {REPS_GRAPH} replays verified",
                            if memop_node { "as BATCH_MEM_OP" } else { "WITHOUT a BATCH_MEM_OP" },
                            if held { "yes" } else { "NO" }
                        );
                        ck((g.exec_destroy)(exec));
                    } else {
                        c1 = format!("captured, instantiate {r_inst:?}");
                    }
                } else {
                    c1 = format!("capture failed (wait {r_wait:?}, consumer {r_kernel:?}, end {r_end:?})");
                }
                if !graph.is_null() {
                    let _ = (g.destroy)(graph);
                }
            }
            // ---- C2: explicit batch mem-op node ----
            let c2 = {
                let mut op: sys::CUstreamBatchMemOpParams = std::mem::zeroed();
                op.waitValue.operation = sys::CUstreamBatchMemOpType::CU_STREAM_MEM_OP_WAIT_VALUE_64;
                op.waitValue.address = done_g;
                op.waitValue.__bindgen_anon_1.value64 = 1;
                op.waitValue.flags = eq;
                let prm = sys::CUDA_BATCH_MEM_OP_NODE_PARAMS { ctx, count: 1, paramArray: &mut op, flags: 0 };
                let mut graph: sys::CUgraph = std::ptr::null_mut();
                ck((g.create)(&mut graph, 0));
                let mut node: sys::CUgraphNode = std::ptr::null_mut();
                let r_add = (g.add_batch_memop)(&mut node, graph, std::ptr::null(), 0, &prm);
                println!("p9:   C2 cuGraphAddBatchMemOpNode (WAIT_VALUE_64, EQ 1): {r_add:?}");
                let out = if r_add != CUresult::CUDA_SUCCESS {
                    format!("cuGraphAddBatchMemOpNode {r_add:?}")
                } else {
                    let mut exec: sys::CUgraphExec = std::ptr::null_mut();
                    let r_inst = (g.instantiate)(&mut exec, graph, 0);
                    println!("p9:   C2 instantiate: {r_inst:?}");
                    if r_inst == CUresult::CUDA_SUCCESS {
                        let held = replay("C2 explicit", exec, true);
                        ck((g.exec_destroy)(exec));
                        format!("instantiated, hold {}, {REPS_GRAPH} replays verified", if held { "yes" } else { "NO" })
                    } else {
                        format!("instantiate {r_inst:?}")
                    }
                };
                let _ = (g.destroy)(graph);
                out
            };
            ck(sys::cuMemFree_v2(done_g));
            let line = format!("C1 stream capture: {c1}; C2 explicit node: {c2}");
            println!("p9: stage C — {line}");
            line
        };

        println!(
            "p9: reference points — swapped barrier 4–6 ms (#186); WDDM ring round trips 3.1–3.4 ms/layer (measured 2026-09-01, retired variant A)"
        );
        println!("p9: PASS — job ring + stream memops verified end to end on Windows (stage A mechanics + stage B timing)");
        println!("p9: stage C (WaitValue64_v2 in a CUDA graph) — {stage_c}");
    }
}
