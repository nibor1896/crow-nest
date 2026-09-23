//! #82 probe (2026-09-20): which teardown order gives the pinned UVA pages back
//! to nvidia_uvm on the open module (610.57.04), and which one HANGS. The serve
//! proof round of 7586234 printed the shutdown line and then stalled >10 min with
//! the RAM still held - the two candidates are `cuMemFreeHost` across the 45 GiB
//! tier and `cuDevicePrimaryCtxReset_v2` called while our own primary-ctx retain
//! is still held (serve.rs calls the reset BEFORE the `Ctx` drop releases it).
//! No engine here: pin <gib> in engine-shaped slabs (96 x PORTABLE|DEVICEMAP,
//! device pointer taken, every page touched = the resident cold tier), then run
//! ONE teardown order per process, every step timed and followed by MemAvailable
//! and the nvidia_uvm refcount, and a plain line before each call that may never
//! return - the timeout kill plus that line IS the localization.
//!   ctx_reset_probe free-exit      pin touch free -> setCurrent(NULL) -> release
//!   ctx_reset_probe reset-hold     pin touch free -> setCurrent(NULL) -> RESET (retain held) -> release  [7586234 serve order]
//!   ctx_reset_probe release-reset  pin touch free -> setCurrent(NULL) -> release -> RESET               [candidate fix order]
//!   ctx_reset_probe leak-exit      pin touch -> setCurrent(NULL) -> release (frees skipped: #82 baseline)

use crow_nest_engine::cuda;
use crow_nest_engine::geo::{GIB, MIB};
use cudarc::driver::sys;
use std::time::Instant;

/// the nvidia_uvm refcount out of /proc/modules (4 on a clean boot of this
/// machine; the leak holds it higher)
fn uvm_refcount() -> u32 {
    std::fs::read_to_string("/proc/modules")
        .ok()
        .and_then(|t| {
            t.lines()
                .find(|l| l.starts_with("nvidia_uvm"))
                .and_then(|l| l.split_whitespace().nth(2).and_then(|r| r.parse().ok()))
        })
        .unwrap_or(0)
}

/// MemAvailable in GiB, the same figure `free -g` prints
fn mem_available_gib() -> f64 {
    let Ok(t) = std::fs::read_to_string("/proc/meminfo") else { return 0.0 };
    for l in t.lines() {
        if let Some(rest) = l.strip_prefix("MemAvailable:") {
            let kb: f64 = rest.trim_end_matches(" kB").trim().parse().unwrap_or(0.0);
            return kb / (1u64 << 20) as f64;
        }
    }
    0.0
}

/// one evidence line per step; stdout is line-buffered, so a killed process has
/// said everything it had said
fn step(tag: &str) {
    println!("[ctx_reset_probe] {tag}: MemAvailable {:.2} GiB, nvidia_uvm refcount {}", mem_available_gib(), uvm_refcount());
}

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let mode = a.get(1).cloned().unwrap_or_else(|| "free-exit".into());
    let gib: f64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(2.0);
    let slabs: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(96);
    let per = ((gib * GIB) as usize / slabs) & !0xfff;
    unsafe {
        // raw init, NOT cuda::Ctx: the release has to happen exactly once and in
        // the order the mode dictates, so the probe owns the retain itself
        cuda::ck(sys::cuInit(0));
        let mut dev: sys::CUdevice = 0;
        cuda::ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx: sys::CUcontext = std::ptr::null_mut();
        cuda::ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        cuda::ck(sys::cuCtxSetCurrent(ctx));
        step(&format!("ctx init, mode {mode}, {slabs} x {:.1} MiB = {:.2} GiB", per as f64 / MIB, (per * slabs) as f64 / GIB));

        // pin + touch, the engine's cold-tier shape (Pinned::alloc flags, page
        // per page resident the way the load fill leaves them)
        let flags = sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP;
        let t0 = Instant::now();
        let mut held: Vec<*mut std::ffi::c_void> = Vec::with_capacity(slabs);
        for _ in 0..slabs {
            let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
            cuda::ck(sys::cuMemHostAlloc(&mut host, per, flags));
            let mut devp: sys::CUdeviceptr = 0;
            cuda::ck(sys::cuMemHostGetDevicePointer_v2(&mut devp, host, 0));
            let p = host as *mut u8;
            let mut o = 0usize;
            while o < per {
                std::ptr::write_volatile(p.add(o), 1);
                o += 4096;
            }
            held.push(host);
        }
        println!("[ctx_reset_probe] pinned + touched in {:.1} s", t0.elapsed().as_secs_f64());
        step("tier resident");

        if mode != "leak-exit" {
            let t2 = Instant::now();
            for h in held.drain(..) {
                println!("[ctx_reset_probe] about to cuMemFreeHost one slab");
                cuda::ck(sys::cuMemFreeHost(h));
            }
            println!("[ctx_reset_probe] cuMemFreeHost x{slabs} in {:.1} s", t2.elapsed().as_secs_f64());
            step("tier freed");
        }

        println!("[ctx_reset_probe] about to cuCtxSetCurrent(NULL)");
        let _ = sys::cuCtxSetCurrent(std::ptr::null_mut());
        match mode.as_str() {
            "free-exit" => {
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxRelease_v2");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxRelease_v2(dev);
                println!("[ctx_reset_probe] release -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("released");
            }
            "reset-hold" => {
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxReset_v2 (retain still held - the 7586234 serve order)");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxReset_v2(dev);
                println!("[ctx_reset_probe] reset -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("reset, retain held");
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxRelease_v2");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxRelease_v2(dev);
                println!("[ctx_reset_probe] release -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("released after reset");
            }
            "release-reset" => {
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxRelease_v2");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxRelease_v2(dev);
                println!("[ctx_reset_probe] release -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("released");
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxReset_v2 (retain gone - the candidate fix order)");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxReset_v2(dev);
                println!("[ctx_reset_probe] reset -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("reset after release");
            }
            "leak-exit" => {
                println!("[ctx_reset_probe] about to cuDevicePrimaryCtxRelease_v2 (frees were SKIPPED)");
                let t = Instant::now();
                let r = sys::cuDevicePrimaryCtxRelease_v2(dev);
                println!("[ctx_reset_probe] release -> {r:?} in {:.1} s", t.elapsed().as_secs_f64());
                step("released, frees skipped");
            }
            other => {
                eprintln!("[ctx_reset_probe] unknown mode {other}");
                std::process::exit(2);
            }
        }
        step("process end");
    }
}
