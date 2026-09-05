//! handoff-bench — crow-nest #7: the job-ring handoff, measured on Windows first.
//!
//! Spec section 3.2/3.6: GPU publishes a cold-expert job via device-side mapped writes;
//! a host stager thread busy-polls the pinned ring, "gathers" one 2.76 MB expert block
//! (pinned-to-pinned copy, standing in for the RAM-tier gather), ships it H2D on a
//! dedicated copy stream and raises the completion flag with a **64-bit stream memop on
//! that stream** (`cuStreamWriteValue64_v2`) — the GPU itself sets the flag when the copy
//! lands, no host synchronization anywhere; the GPU waits via `cuStreamWaitValue64_v2`
//! (GEQ) and consumes. Sequence numbers are u64 and monotonic — no resets, no
//! wraparound in the engine's lifetime. No host in the hot loop, no kernel launch for
//! synchronization, no full-device barrier.
//!
//! Design principle (robin, 2026-09-02): state of the art only — the modern `_v2` memops,
//! 64-bit flags, `compute_120a` — and every choice swappable, not hardcoded.
//!
//! Measured, resolution stated:
//!   1. raw memop overhead: WriteValue64 + WaitValue64 (pre-set) round trip
//!   2. latency mode: full publish -> stager -> H2D -> consume round trip, serialized
//!   3. throughput mode: K iterations back-to-back, stager overlapping (the real shape)
//! Reference: the borrowed engine's swapped barrier costs 4-6 ms (Crow #186).

use std::ffi::CString;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const RING_SLOTS: usize = 256;
const DESC_BYTES: usize = 64;
const PAYLOAD_BYTES: usize = 2_760_000; // one expert block, NVFP4 (spec 2.1)
const ITERS: usize = 200;
const WARMUP: usize = 20;

// Mapped host layout (UVA: the host pointer is directly usable as a device pointer):
// [ descriptors: RING_SLOTS * DESC_BYTES ][ job flags: RING_SLOTS * u64 ]
// [ completion flag: u64 ][ scratch: 8 B ][ source payload ][ staging payload ]
const OFF_FLAGS: usize = RING_SLOTS * DESC_BYTES;
const OFF_COMPLETE: usize = OFF_FLAGS + RING_SLOTS * 8;
const OFF_RESULT: usize = OFF_COMPLETE + 8;
const OFF_PROBE: usize = OFF_RESULT + 32;
const OFF_SRC: usize = OFF_PROBE + 8;
const OFF_STAGE: usize = OFF_SRC + PAYLOAD_BYTES;
const RING_BYTES: usize = OFF_STAGE + PAYLOAD_BYTES;

static HOST_BASE: std::sync::OnceLock<usize> = std::sync::OnceLock::new();

const KERNEL_SRC: &str = r#"
#define RING_SLOTS 256

// One thread publishes the descriptor, then the job flag; system-scope atomicity orders
// the descriptor before the flag so the stager never sees a flag without its payload id.
extern "C" __global__ void publisher(unsigned long long* desc,
                                     unsigned long long* job_flag,
                                     unsigned long long seq) {
    if (threadIdx.x == 0) {
        unsigned long long slot = (seq % RING_SLOTS) * 8;
        atomicExch_system((unsigned long long*)&desc[slot + 0], seq);
        atomicExch_system((unsigned long long*)&desc[slot + 1], 1ull);
        atomicExch_system((unsigned long long*)&desc[slot + 2], 0x0102030405060708ull);
        __threadfence_system();
        atomicExch_system(&job_flag[seq % RING_SLOTS], seq);
    }
}

// Consumer folds the freshly copied device block; proves the H2D landed.
extern "C" __global__ void consumer(const unsigned int* payload, unsigned long long* result, int n) {
    unsigned int s = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s ^= payload[i];
    // warp reduction: thread 0 must report the WHOLE warp's xor, not its own stripe
    for (int off = 16; off > 0; off >>= 1) s ^= __shfl_down_sync(0xFFFFFFFFu, s, off);
    if (threadIdx.x == 0) {
        atomicExch_system(&result[0], (unsigned long long)s);
        atomicExch_system(&result[1], (unsigned long long)n); // what the kernel actually saw
        atomicExch_system(&result[2], (unsigned long long)payload[0]); // kernel's data view
        atomicExch_system(&result[3], (unsigned long long)payload[1]);
    }
}
"#;

/// Raw CUDA handles are `Send` by driver-API contract; cudarc's types just don't mark it.
struct SendPtr(*mut sys::CUstream_st);
unsafe impl Send for SendPtr {}
struct CtxPtr(*mut sys::CUctx_st);
unsafe impl Send for CtxPtr {}

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

fn report(v: &[f64]) {
    if v.is_empty() {
        return;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean: f64 = s.iter().sum::<f64>() / s.len() as f64;
    println!(
        "   mean {:.3} ms | p50 {:.3} | p99 {:.3} | min {:.3} | max {:.3}",
        mean,
        s[s.len() / 2],
        s[(s.len() as f64 * 0.99) as usize],
        s[0],
        s[s.len() - 1]
    );
}


fn stager_loop(
    stop: Arc<AtomicBool>,
    served: Arc<AtomicU64>,
    published: Arc<AtomicU64>,
    base: usize,
    dev_payload: CUdeviceptr,
    complete_dev: CUdeviceptr,
    s_copy: SendPtr,
    ctx: CtxPtr,
) {
    let s_copy = s_copy.0;
    #[cfg(feature = "stager_ctx")]
    unsafe { ck(sys::cuCtxSetCurrent(ctx.0)); }
    // Driver-API calls from this thread need the context current HERE — the primary
    // context is per-process but "current" is per-thread.
                            while !stop.load(Ordering::Relaxed) {
                let next = served.load(Ordering::Relaxed);
                #[cfg(feature = "no_pub_kernel")]
                {
                    // no GPU-published flag: wait for the host-side counter instead
                    if published.load(Ordering::Relaxed) <= next {
                        std::thread::yield_now();
                        continue;
                    }
                }
                #[cfg(not(feature = "no_pub_kernel"))]
                {
                    let flag =
                        (base + OFF_FLAGS + ((next as usize % RING_SLOTS) * 8)) as *const u64;
                    let seen = unsafe { flag.read_volatile() };
                    if seen != next {
                        std::thread::yield_now();
                        continue;
                    }
                }
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        (base + OFF_SRC) as *const u8,
                        (base + OFF_STAGE) as *mut u8,
                        PAYLOAD_BYTES,
                    );
                    std::sync::atomic::fence(Ordering::SeqCst);
                    let r = sys::cuMemcpyHtoDAsync_v2(
                        dev_payload,
                        (base + OFF_STAGE) as *const std::ffi::c_void,
                        PAYLOAD_BYTES,
                        s_copy,
                    );
                    if r != CUresult::CUDA_SUCCESS {
                        eprintln!("stager: cuMemcpyHtoDAsync_v2 failed: {r:?} at job {next}");
                        return;
                    }
                    // GPU sets the completion flag itself when the copy lands.
                    #[cfg(feature = "completion_memop")]
                    {
                        // GPU raises the flag itself when the copy lands (spec 3.2 design
                        // path). FINDING (2026-09-02): on this WDDM stack the stream-
                        // ordered memop-after-H2D did NOT hold — consumers saw torn
                        // payloads. Retest on Linux before ever trusting it there.
                        let r2 = sys::cuStreamWriteValue64_v2(s_copy, complete_dev, next + 1, 0);
                        if r2 != CUresult::CUDA_SUCCESS {
                            eprintln!("stager: WriteValue64_v2 failed: {r2:?} at job {next}");
                            return;
                        }
                    }
                    #[cfg(not(feature = "completion_memop"))]
                    {
                        // WDDM-safe default: the stager waits for ITS OWN copy (host wait,
                        // off the GPU critical path — resident-expert compute continues)
                        // and then raises the flag with a plain host store, exl3-style.
                        let rs = sys::cuStreamSynchronize(s_copy);
                        if rs != CUresult::CUDA_SUCCESS {
                            eprintln!("stager: sync failed: {rs:?} at job {next}");
                            return;
                        }
                        ((base + OFF_COMPLETE) as *mut u64).write_volatile(next + 1);
                        std::sync::atomic::fence(Ordering::SeqCst);
                    }
                }
                served.store(next + 1, Ordering::Relaxed);
            }
}


// --- crash forensics: print exception address + faulting module on 0xC0000005 ---
#[repr(C)]
struct ExceptionRecord {
    code: u32, flags: u32, record: *mut ExceptionRecord, address: *const u8,
    n_params: u32, info: [usize; 15],
}
#[repr(C)]
struct ExceptionPointers { record: *mut ExceptionRecord, context: *mut u8 }

extern "system" {
    fn AddVectoredExceptionHandler(first: u32, handler: usize) -> *mut std::ffi::c_void;
    fn GetModuleHandleExW(flags: u32, addr: *const u8, module: *mut *mut std::ffi::c_void) -> i32;
    fn GetModuleFileNameW(module: *mut std::ffi::c_void, buf: *mut u16, len: u32) -> u32;
}

unsafe extern "system" fn veh(ep: *mut ExceptionPointers) -> i32 {
    let rec = unsafe { (*ep).record };
    let code = unsafe { (*rec).code };
    if code == 0xC0000005 {
        let addr = unsafe { (*rec).address as *const u8 };
        let mut module = std::ptr::null_mut();
        let mut buf = [0u16; 512];
        let ok = unsafe {
            GetModuleHandleExW(0x4 | 0x2, addr, &mut module) // FROM_ADDRESS | UNCHANGED_REFCOUNT
        };
        let (name, offset) = if ok != 0 {
            unsafe { GetModuleFileNameW(module, buf.as_mut_ptr(), 512) };
            let w: String = buf.iter().take_while(|&&c| c != 0).map(|c| char::from_u32(u32::from(*c)).unwrap_or('?')).collect();
            let mbase = module as usize;
            (w, (addr as usize).wrapping_sub(mbase))
        } else {
            ("<unknown module>".into(), addr as usize)
        };
        let rw = unsafe { (*rec).info[0] };
        let target_addr = unsafe { (*rec).info[1] };
        eprintln!("VEH: ACCESS_VIOLATION at {:p} in {name} +{offset:#x}  op={} target_addr={target_addr:#x}",
            addr, if rw == 0 { "READ" } else { "WRITE" });
    }
    0 // EXCEPTION_CONTINUE_SEARCH
}

fn main() {
    unsafe { AddVectoredExceptionHandler(1, veh as usize); }

    unsafe {
        ck(sys::cuInit(0));
        let mut dev = 0;
        ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx = std::ptr::null_mut();
        #[cfg(feature = "sched_spin")]
        ck(sys::cuDevicePrimaryCtxSetFlags(dev, sys::CU_CTX_SCHED_SPIN));
        ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        ck(sys::cuCtxSetCurrent(ctx));

        let opts = CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).expect("nvrtc");
        std::fs::write("handoff_emitted.ptx", ptx.to_src()).expect("write ptx");
        let c_ptx = CString::new(ptx.to_src()).unwrap();
        let mut module = std::ptr::null_mut();
        ck(sys::cuModuleLoadData(&mut module, c_ptx.as_ptr() as *const _));
        let get_fn = |s: &str| {
            let mut f = std::ptr::null_mut();
            ck(sys::cuModuleGetFunction(
                &mut f,
                module,
                CString::new(s).unwrap().as_ptr(),
            ));
            f
        };
        let pub_fn = get_fn("publisher");
        let con_fn = get_fn("consumer");

        // Mapped pinned ring (PORTABLE | DEVICEMAP); under UVA the host pointer is the
        // device pointer.
        let mut host_ring: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut host_ring,
            RING_BYTES,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
        ));
        let base = host_ring as usize;
        HOST_BASE.set(base).expect("host base once");
        let job_flags_dev = (base + OFF_FLAGS) as CUdeviceptr;
        let complete_dev = (base + OFF_COMPLETE) as CUdeviceptr;

        // Zero flags + payloads BEFORE any GPU work: GEQ waits against uninitialized
        // mapped memory would be satisfied instantly by garbage (measured, cost a
        // debugging round).
        std::ptr::write_bytes(host_ring as *mut u8, 0, RING_BYTES);

        // Deterministic source payload; the consumer's xor must reproduce this.
        let src =
            std::slice::from_raw_parts_mut((base + OFF_SRC) as *mut u32, PAYLOAD_BYTES / 4);
        let mut x: u32 = 0x9E3779B9;
        let mut expect = 0u32;
        for v in src.iter_mut() {
            x ^= x << 13;
            x ^= x >> 17;
            x ^= x << 5;
            *v = x;
            expect ^= x;
        }

        let mut dev_payload: CUdeviceptr = 0;
        let mut dev_result: CUdeviceptr = 0;
        ck(sys::cuMemAlloc_v2(&mut dev_payload, PAYLOAD_BYTES));
        ck(sys::cuMemAlloc_v2(&mut dev_result, 12));

        let mut s_main = std::ptr::null_mut();
        let mut s_copy = std::ptr::null_mut();
        ck(sys::cuStreamCreate(
            &mut s_main,
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
        ));
        ck(sys::cuStreamCreate(
            &mut s_copy,
            sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32,
        ));

        // --- 0. kernel smoke tests, NO stager thread running yet ---
        {
            let mut p_payload = dev_payload as *mut std::ffi::c_void;
            let mut p_result = (base + OFF_RESULT) as *mut std::ffi::c_void;
            let mut n = (PAYLOAD_BYTES / 4) as i32;
            eprintln!("smoke: consumer launch");
            ck(sys::cuLaunchKernel(con_fn, 1, 1, 1, 32, 1, 1, 0, s_main,
                [&mut p_payload as *mut _ as *mut _,
                 &mut p_result as *mut _ as *mut _,
                 &mut n as *mut _ as *mut _].as_mut_ptr(), std::ptr::null_mut()));
            eprintln!("smoke: consumer sync");
            ck(sys::cuStreamSynchronize(s_main));
            eprintln!("smoke: consumer ok");
            // XOR check on a known buffer, NO ring involved: fill dev_payload via
            // synchronous H2D from src, run consumer, read mapped result.
            let src = std::slice::from_raw_parts((base + OFF_SRC) as *const u32, PAYLOAD_BYTES / 4);
            let mut hx = 0u32;
            for v in src { hx ^= *v; }
            ck(sys::cuMemcpyHtoD_v2(dev_payload, (base + OFF_SRC) as *const std::ffi::c_void, PAYLOAD_BYTES));
            let mut p_payload = dev_payload as *mut std::ffi::c_void;
            let mut p_result = (base + OFF_RESULT) as *mut std::ffi::c_void;
            let mut n = (PAYLOAD_BYTES / 4) as i32;
            ck(sys::cuLaunchKernel(con_fn, 1, 1, 1, 32, 1, 1, 0, s_main,
                [&mut p_payload as *mut _ as *mut _,
                 &mut p_result as *mut _ as *mut _,
                 &mut n as *mut _ as *mut _].as_mut_ptr(), std::ptr::null_mut()));
            ck(sys::cuStreamSynchronize(s_main));
            let kx = ((base + OFF_RESULT) as *const u64).read_volatile() as u32;
            let kn = (((base + OFF_RESULT + 8) as *const u64).read_volatile()) as u32;
            // full-buffer diff AFTER a device-side sync
            let mut full = vec![0u32; PAYLOAD_BYTES / 4];
            ck(sys::cuMemcpyDtoH_v2(full.as_mut_ptr() as *mut std::ffi::c_void, dev_payload, PAYLOAD_BYTES));
            let diffs: usize = full.iter().zip(src).filter(|(a, b)| a != b).count();
            let mut dx = 0u32;
            for v in &full { dx ^= *v; }
            println!("smoke: host_xor={hx:#010x} kernel_xor={kx:#010x} n_seen={kn} diffs={diffs} dtoh_xor={dx:#010x} match={}", hx == kx);
        }
        {
            let mut p_desc = base as *mut std::ffi::c_void;
            let mut p_flags = job_flags_dev as *mut std::ffi::c_void;
            let mut seq_v = 0u64;
            eprintln!("smoke: publisher launch");
            ck(sys::cuLaunchKernel(pub_fn, 1, 1, 1, 32, 1, 1, 0, s_main,
                [&mut p_desc as *mut _ as *mut _,
                 &mut p_flags as *mut _ as *mut _,
                 &mut seq_v as *mut _ as *mut _].as_mut_ptr(), std::ptr::null_mut()));
            eprintln!("smoke: publisher sync");
            ck(sys::cuStreamSynchronize(s_main));
            eprintln!("smoke: publisher ok, flag readback = {}", unsafe {
                (base + OFF_FLAGS) as *const u64
            }.read_volatile());
        }

        // Monotonic counters, shared with the stager; no resets anywhere.
        let published = Arc::new(AtomicU64::new(0));
        let served = Arc::new(AtomicU64::new(0));
        let stop = Arc::new(AtomicBool::new(false));
        let (stop2, served2, published2) = (stop.clone(), served.clone(), published.clone());        let s_copy_send = SendPtr(s_copy);
        let ctx_send = CtxPtr(ctx);
        let stager = std::thread::spawn(move || stager_loop(stop2, served2.clone(), published2.clone(), base, dev_payload, complete_dev, s_copy_send, ctx_send));


        let launch1 = |f, args: &mut [*mut std::ffi::c_void]| {
            ck(sys::cuLaunchKernel(
                f, 1, 1, 1, 32, 1, 1, 0, s_main, args.as_mut_ptr(), std::ptr::null_mut(),
            ));
        };

        // --- 1. raw memop overhead (64-bit) ---
        let probe = (base + OFF_PROBE) as CUdeviceptr;
        const MOPS: u64 = 2000;
        for v in 0..MOPS {
            ck(sys::cuStreamWriteValue64_v2(s_main, probe, v, 0));
            ck(sys::cuStreamWaitValue64_v2(s_main, probe, v, 0));
        }
        ck(sys::cuStreamSynchronize(s_main));
        let t = Instant::now();
        for v in MOPS..MOPS * 2 {
            ck(sys::cuStreamWriteValue64_v2(s_main, probe, v, 0));
            ck(sys::cuStreamWaitValue64_v2(s_main, probe, v, 0));
        }
        ck(sys::cuStreamSynchronize(s_main));
        let mop_us = t.elapsed().as_secs_f64() * 1e6 / MOPS as f64 / 2.0;
        println!("1) raw memop (64-bit): {mop_us:.2} us per Write/WaitValue64_v2 (enqueue+execute, n={MOPS})");

        let mut ev_mark = std::ptr::null_mut();
        ck(sys::cuEventCreate(&mut ev_mark, 0));

        // --- 2 + 3. full handoff ---
        let run = |pipelined: bool, label: &str| -> (Vec<f64>, f64) {
            let base_seq = published.load(Ordering::Relaxed);
            ck(sys::cuStreamSynchronize(s_main));
            ck(sys::cuStreamSynchronize(s_copy));
            let mut times = Vec::new();
            let t_start = Instant::now();
            for i in 0..(ITERS + WARMUP) as u64 {
                let seq = base_seq + i;
                let t0 = Instant::now();
                let mut p_desc = (base + (seq as usize % RING_SLOTS) * DESC_BYTES)
                    as *mut std::ffi::c_void;
                let mut p_flags = job_flags_dev as *mut std::ffi::c_void;
                let mut seq_v = seq;
                launch1(
                    pub_fn,
                    &mut [
                        &mut p_desc as *mut _ as *mut _,
                        &mut p_flags as *mut _ as *mut _,
                        &mut seq_v as *mut _ as *mut _,
                    ],
                );
                published.store(seq + 1, Ordering::Relaxed);

                let mut p_payload = dev_payload as *mut std::ffi::c_void;
                let mut p_result = (base + OFF_RESULT) as *mut std::ffi::c_void;
                let mut n = (PAYLOAD_BYTES / 4) as i32;
                ck(sys::cuStreamWaitValue64_v2(s_main, complete_dev, seq + 1, 0));
                launch1(
                    con_fn,
                    &mut [
                        &mut p_payload as *mut _ as *mut _,
                        &mut p_result as *mut _ as *mut _,
                        &mut n as *mut _ as *mut _,
                    ],
                );
                if !pipelined {
                    #[cfg(feature = "busy_poll")]
                    {
                        ck(sys::cuEventRecord(ev_mark, s_main));
                        loop {
                            let r = sys::cuEventQuery(ev_mark);
                            if r != CUresult::CUDA_ERROR_NOT_READY { ck(r); break; }
                            std::hint::spin_loop();
                        }
                    }
                    #[cfg(not(feature = "busy_poll"))]
                    ck(sys::cuStreamSynchronize(s_main));
                    if i >= WARMUP as u64 {
                        times.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                } else if i == ITERS as u64 + WARMUP as u64 - 1 {
                    ck(sys::cuStreamSynchronize(s_main));
                }
            }
            ck(sys::cuStreamSynchronize(s_main));
            ck(sys::cuStreamSynchronize(s_copy));
            // result lives in mapped memory, written by the kernel with system-scope
            // atomics; both streams are drained, so a host read is safe here.
            let res3 = unsafe {
                [
                    ((base + OFF_RESULT) as *const u64).read_volatile() as u32,
                    ((base + OFF_RESULT + 8) as *const u64).read_volatile() as u32,
                    ((base + OFF_RESULT + 16) as *const u64).read_volatile() as u32,
                ]
            };
            eprintln!("DIAG kernel_payload_ptr={:#x} kernel_result_ptr={:#x} host_dev_payload={dev_payload:#x}",
                (((base + OFF_RESULT + 16) as *const u64).read_volatile()),
                (((base + OFF_RESULT + 24) as *const u64).read_volatile()));
            // DIAGNOSTIC: what does the device buffer actually hold?
            let mut probe4 = [0u32; 4];
            ck(sys::cuMemcpyDtoH_v2(
                probe4.as_mut_ptr() as *mut std::ffi::c_void,
                dev_payload,
                16,
            ));
            let src4 = [
                ((base + OFF_SRC) as *const u32).read_volatile(),
                ((base + OFF_SRC + 4) as *const u32).read_volatile(),
                ((base + OFF_STAGE) as *const u32).read_volatile(),
                ((base + OFF_STAGE + 4) as *const u32).read_volatile(),
            ];
            eprintln!("DIAG served={} n_seen={} sentinel={:#x} dev[:2]={:?} src[:2]={:?}",
                served.load(Ordering::Relaxed), res3[1], res3[2], &probe4[..2], &src4[..2]);

            if res3[0] != expect {
                println!("   WARNING: consumer xor {:#010x} != expected {expect:#010x} (n_seen={}) ({label})", res3[0], res3[1]);
            } else {
                println!("   consumer payload verified ({label})");
            }
            {
                let mut full = vec![0u32; PAYLOAD_BYTES / 4];
                ck(sys::cuMemcpyDtoH_v2(full.as_mut_ptr() as *mut std::ffi::c_void, dev_payload, PAYLOAD_BYTES));
                let mut dx = 0u32;
                for v in &full { dx ^= *v; }
                let mut first_diff = None;
                for (i, v) in full.iter().enumerate() {
                    if *v != src[i] { first_diff = Some((i, *v, src[i])); break; }
                }
                eprintln!("DIAG dev_xor={dx:#010x} first_diff={first_diff:?} kernview=[{:x},{:x}] srcview=[{:#x},{:#x}]",
                ((base + OFF_RESULT + 16) as *const u64).read_volatile(),
                ((base + OFF_RESULT + 24) as *const u64).read_volatile(),
                ((base + OFF_SRC) as *const u32).read_volatile(),
                ((base + OFF_SRC + 4) as *const u32).read_volatile());
            }
            let _ = res3;
            (times, t_start.elapsed().as_secs_f64() * 1e3)
        };

        println!("2) latency mode (serialized round trips, n={ITERS}, warmup {WARMUP}):");
        let (lat, _) = run(false, "latency");
        report(&lat);
        println!("3) throughput mode ({ITERS} back-to-back, stager overlapping):");
        let (_, wall) = run(true, "pipelined");
        println!(
            "   {ITERS} handoffs in {wall:.1} ms -> {:.2} ms/handoff pipelined",
            wall / ITERS as f64
        );
        println!("\nreference: borrowed-engine barrier swap = 4-6 ms (Crow #186)");

        stop.store(true, Ordering::Relaxed);
        stager.join().expect("stager");
        let _ = served.load(Ordering::Relaxed);
    }
}
