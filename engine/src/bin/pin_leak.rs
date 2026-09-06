//! #18 probe (2026-09-05): does the driver keep device memory per pinned
//! DEVICEMAP allocation after cuMemFreeHost? No engine: allocate the cold-tier
//! shape (2 x 48 pinned slabs, PORTABLE|DEVICEMAP[|WRITECOMBINED], device pointer
//! taken), free everything, read cuMemGetInfo; repeat n cycles.
//!   pin_leak <cycles=3> <total_gib=44> <slabs=96> [wc=1] [touch=0]
//! touch=1 writes one byte per 4 KiB page from the host (the engine fills the
//! tier at load, so the pages are resident in production).
use crow_nest_engine::cuda;
use cudarc::driver::sys;

fn main() {
    let a: Vec<String> = std::env::args().collect();
    let cycles: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(3);
    let total_gib: f64 = a.get(2).and_then(|v| v.parse().ok()).unwrap_or(44.0);
    let slabs: usize = a.get(3).and_then(|v| v.parse().ok()).unwrap_or(96);
    let wc = a.get(4).map(|v| v != "0").unwrap_or(true);
    let touch = a.get(5).map(|v| v == "1").unwrap_or(false);
    let per = ((total_gib * (1u64 << 30) as f64) as usize / slabs) & !0xfff;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let flags = sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP | if wc { sys::CU_MEMHOSTALLOC_WRITECOMBINED } else { 0 };
        let f_start = cuda::free_vram_bytes();
        println!("[pin_leak] {slabs} x {:.1} MiB = {:.2} GiB, wc={wc} touch={touch}; free VRAM at start {:.1} MB, free RAM {:.1} GiB",
            per as f64 / (1 << 20) as f64, (per * slabs) as f64 / (1u64 << 30) as f64, f_start as f64 / 1e6, cuda::free_physical_ram() as f64 / (1u64 << 30) as f64);
        for c in 0..cycles {
            let t0 = std::time::Instant::now();
            let mut held = Vec::with_capacity(slabs);
            for _ in 0..slabs {
                let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
                cuda::ck(sys::cuMemHostAlloc(&mut host, per, flags));
                let mut dev: sys::CUdeviceptr = 0;
                cuda::ck(sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0));
                if touch {
                    let p = host as *mut u8;
                    let mut o = 0usize;
                    while o < per { std::ptr::write_volatile(p.add(o), 1); o += 4096; }
                }
                held.push(host);
            }
            let f_held = cuda::free_vram_bytes();
            let t1 = t0.elapsed().as_secs_f64();
            for h in held.drain(..) { cuda::ck(sys::cuMemFreeHost(h)); }
            // PIN_LEAK_HOLD_S=<s>: stay alive after the free so an outside sampler can
            // read what the process still holds host-side (2026-09-06 harness reload check)
            if let Some(s) = std::env::var("PIN_LEAK_HOLD_S").ok().and_then(|v| v.parse::<u64>().ok()) {
                println!("[pin_leak] cycle {c}: freed; free RAM now {:.2} GiB, holding {s} s", cuda::free_physical_ram() as f64 / (1u64 << 30) as f64);
                std::thread::sleep(std::time::Duration::from_secs(s));
                println!("[pin_leak] cycle {c}: after hold free RAM {:.2} GiB", cuda::free_physical_ram() as f64 / (1u64 << 30) as f64);
            }
            let f_after = cuda::free_vram_bytes();
            println!("[pin_leak] cycle {c}: alloc {:.1} s; free VRAM while held {:.1} MB (delta {:+.1}), after free {:.1} MB (leak vs start {:+.1} MB)",
                t1, f_held as f64 / 1e6, (f_held as f64 - f_start as f64) / 1e6, f_after as f64 / 1e6, (f_start as f64 - f_after as f64) / 1e6);
        }
        // does cuCtxSynchronize / a device sync return anything?
        cuda::ck(sys::cuCtxSynchronize());
        println!("[pin_leak] after cuCtxSynchronize: free VRAM {:.1} MB (leak vs start {:+.1} MB)", cuda::free_vram_bytes() as f64 / 1e6, (f_start as f64 - cuda::free_vram_bytes() as f64) / 1e6);
    }
}
