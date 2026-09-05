//! Measurement C-D1a/D1b (2026-09-04): what limits pinned (cuMemHostAlloc,
//! PORTABLE|DEVICEMAP) host memory on this machine - total bytes, or the
//! number/size of mappings? Three passes, everything freed after each:
//!   1) incremental 1 GiB allocations until the first failure (total ceiling)
//!   2) one single allocation, descending from the cap in 2 GiB steps
//!   3) 96 equal allocations (today's cold tier shape) sized to the cap
//! Cap = min(free physical RAM - 4 GiB, 50 GiB) so the system stays responsive.
use crow_nest_engine::cuda;
use cudarc::driver::sys;

const GIB: u64 = 1 << 30;

unsafe fn try_alloc(bytes: u64) -> Result<*mut std::ffi::c_void, sys::CUresult> {
    let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
    let r = sys::cuMemHostAlloc(&mut host, bytes as usize, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP);
    if r == sys::CUresult::CUDA_SUCCESS {
        let mut dev: sys::CUdeviceptr = 0;
        let r2 = sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0);
        if r2 != sys::CUresult::CUDA_SUCCESS {
            sys::cuMemFreeHost(host);
            return Err(r2);
        }
        Ok(host)
    } else {
        Err(r)
    }
}

fn main() {
    unsafe {
        let _ctx = cuda::Ctx::init();
        let free = cuda::free_physical_ram();
        let cap = (free.saturating_sub(4 * GIB)).min(50 * GIB);
        println!("[pin] free physical RAM {:.2} GiB, probe cap {:.2} GiB", free as f64 / GIB as f64, cap as f64 / GIB as f64);
        // 1) incremental
        let mut held: Vec<*mut std::ffi::c_void> = Vec::new();
        let mut total = 0u64;
        loop {
            if total + GIB > cap { println!("[pin] 1) reached cap without failure at {:.1} GiB", total as f64 / GIB as f64); break; }
            match try_alloc(GIB) {
                Ok(h) => { held.push(h); total += GIB; }
                Err(e) => { println!("[pin] 1) incremental 1 GiB allocations: failed at {:.1} GiB total with {e:?}", total as f64 / GIB as f64); break; }
            }
        }
        let free_now = cuda::free_physical_ram();
        println!("[pin] 1) held {:.1} GiB pinned, free physical RAM now {:.2} GiB", total as f64 / GIB as f64, free_now as f64 / GIB as f64);
        for h in held.drain(..) { sys::cuMemFreeHost(h); }
        // 2) single allocation, descending
        let mut size = cap / GIB * GIB;
        loop {
            if size == 0 { println!("[pin] 2) no single allocation succeeded"); break; }
            match try_alloc(size) {
                Ok(h) => { println!("[pin] 2) single allocation OK at {:.1} GiB", size as f64 / GIB as f64); sys::cuMemFreeHost(h); break; }
                Err(e) => { println!("[pin] 2) single {:.1} GiB failed: {e:?}", size as f64 / GIB as f64); size = size.saturating_sub(2 * GIB); }
            }
        }
        // 3) 96 equal allocations (today's tier shape)
        let per = cap / 96 / (1 << 20) * (1 << 20);
        let mut n_ok = 0;
        let mut last_err = None;
        for _ in 0..96 {
            match try_alloc(per) {
                Ok(h) => { held.push(h); n_ok += 1; }
                Err(e) => { last_err = Some(e); break; }
            }
        }
        println!("[pin] 3) 96 x {:.0} MiB: {} succeeded ({:.1} GiB){}", per as f64 / (1 << 20) as f64, n_ok, (n_ok as u64 * per) as f64 / GIB as f64,
            match last_err { Some(e) => format!(", first failure {e:?}"), None => String::new() });
        for h in held.drain(..) { sys::cuMemFreeHost(h); }
        println!("[pin] done, everything freed");
    }
}
