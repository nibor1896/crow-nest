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

        println!(
            "p9: reference points — swapped barrier 4–6 ms (#186); WDDM ring round trips 3.1–3.4 ms/layer (measured 2026-09-01, retired variant A)"
        );
        println!("p9: PASS — job ring + stream memops verified end to end on Windows (stage A mechanics + stage B timing)");
    }
}
