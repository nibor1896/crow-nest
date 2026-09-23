//! #103, the #82 follow-up probe (2026-09-23): where do pinned cold-tier pages go after
//! free / exit / SIGKILL, and what does the alternative allocation cost?
//!
//! Two allocation kinds are compared:
//!   host / host-wc   cuMemHostAlloc(PORTABLE|DEVICEMAP[|WRITECOMBINED]): the
//!                    pages come from the NVIDIA driver (nv_alloc_system_pages)
//!                    and go back to its sysmem page pool on free (nv-vm.c,
//!                    NVreg_EnableSystemMemoryPools, default 0x211)
//!   reg / reg-huge   anonymous mmap (+ MADV_HUGEPAGE for reg-huge), touched,
//!                    then cuMemHostRegister(PORTABLE|DEVICEMAP): the pages are
//!                    ordinary kernel anon pages the driver only pins
//!
//!   pin_return_probe snap
//!       one meminfo line, no CUDA
//!   pin_return_probe hold <kind> <gib> <end> [slabs=8] [hold_s=0]
//!       allocate <gib> in <slabs> equal slabs, touch every page, then
//!       end = free  free every slab, release the context, exit 0
//!             exit  release the context and exit WITHOUT freeing
//!             kill  SIGKILL ourselves while holding everything
//!   pin_return_probe bw [kinds...]
//!       1 GiB per kind, GPU read bandwidth of the stage pattern
//!       (production stage_cold_ca shape: cp.async.cg 4 KB tiles, 40 x 256)
//!       plus the plain copy16 kernel and the copy engine
//!
//! Every step prints the `cuda::HostRam` view (free_for_pin, MemAvailable,
//! the driver-held estimate) so an outside reader sees what the engine sees.
use crow_nest_engine::cuda;
use crow_nest_engine::geo::GIB;
use crow_nest_engine::kernels::launch_v;
use cudarc::driver::sys;

const ONE_GIB: usize = 1 << 30;

/// /proc/vmstat nr_foll_pin_acquired - nr_foll_pin_released: pages pinned by
/// pin_user_pages (cuMemHostRegister) and not yet unpinned, machine-wide
fn foll_pin_outstanding() -> i64 {
    let t = std::fs::read_to_string("/proc/vmstat").unwrap_or_default();
    let g = |k: &str| t.lines().find_map(|l| l.strip_prefix(k).and_then(|v| v.trim().parse::<i64>().ok())).unwrap_or(0);
    g("nr_foll_pin_acquired ") - g("nr_foll_pin_released ")
}

fn snap(tag: &str) {
    let r = cuda::free_physical_ram_parts();
    let m = cuda::meminfo();
    println!(
        "[pin_return] {tag:<28} MemFree {:6.2}  MemAvailable {:6.2}  AnonPages {:6.2}  Shmem {:6.2}  driver-held {:6.2} (live {:5.2})  free_for_pin {:6.2}  swap used {:5.2} GiB  FOLL_PIN outstanding {} pages{}",
        m.get("MemFree") as f64 / GIB,
        r.mem_available as f64 / GIB,
        m.get("AnonPages") as f64 / GIB,
        m.get("Shmem") as f64 / GIB,
        r.driver_held as f64 / GIB,
        r.driver_live as f64 / GIB,
        r.free_for_pin as f64 / GIB,
        (m.get("SwapTotal").saturating_sub(m.get("SwapFree"))) as f64 / GIB,
        foll_pin_outstanding(),
        if r.other_cuda { "  (other CUDA process alive)" } else { "" }
    );
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Kind {
    Host,
    HostWc,
    Reg,
    RegHuge,
    /// the engine's own `cuda::Pinned::alloc_registered`
    Lib,
}

fn kind(s: &str) -> Kind {
    match s {
        "host" => Kind::Host,
        "host-wc" => Kind::HostWc,
        "reg" => Kind::Reg,
        "reg-huge" => Kind::RegHuge,
        "lib" => Kind::Lib,
        _ => panic!("kind is host | host-wc | reg | reg-huge, got {s}"),
    }
}

struct Slab {
    host: *mut std::ffi::c_void,
    dev: u64,
    bytes: usize,
    kind: Kind,
    lib: Option<cuda::Pinned>,
}

unsafe fn alloc(k: Kind, bytes: usize) -> Slab {
    let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
    if k == Kind::Lib {
        let mut p = cuda::Pinned::alloc_registered(bytes);
        let mut o = 0;
        while o < bytes {
            std::ptr::write_volatile((p.host as *mut u8).add(o), 1);
            o += 4096;
        }
        let (host, dev) = (p.host, p.dev);
        p.write_bytes(0, &[1]);
        return Slab { host, dev, bytes, kind: k, lib: Some(p) };
    }
    match k {
        Kind::Lib => unreachable!(),
        Kind::Host | Kind::HostWc => {
            let wc = if k == Kind::HostWc { sys::CU_MEMHOSTALLOC_WRITECOMBINED } else { 0 };
            cuda::ck(sys::cuMemHostAlloc(&mut host, bytes, sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP | wc));
            // touch: the engine fills the whole tier at load
            let p = host as *mut u8;
            let mut o = 0;
            while o < bytes {
                std::ptr::write_volatile(p.add(o), 1);
                o += 4096;
            }
        }
        Kind::Reg | Kind::RegHuge => {
            host = libc::mmap(
                std::ptr::null_mut(),
                bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            );
            assert!(host != libc::MAP_FAILED, "mmap {bytes} B failed");
            if k == Kind::RegHuge {
                libc::madvise(host, bytes, libc::MADV_HUGEPAGE);
            }
            let p = host as *mut u8;
            let mut o = 0;
            while o < bytes {
                std::ptr::write_volatile(p.add(o), 1);
                o += 4096;
            }
            cuda::ck(sys::cuMemHostRegister_v2(host, bytes, sys::CU_MEMHOSTREGISTER_PORTABLE | sys::CU_MEMHOSTREGISTER_DEVICEMAP));
        }
    }
    let mut dev: sys::CUdeviceptr = 0;
    cuda::ck(sys::cuMemHostGetDevicePointer_v2(&mut dev, host, 0));
    Slab { host, dev, bytes, kind: k, lib: None }
}

unsafe fn free(mut s: Slab) {
    if let Some(p) = s.lib.as_mut() {
        p.free();
        return;
    }
    match s.kind {
        Kind::Lib => unreachable!(),
        Kind::Host | Kind::HostWc => cuda::ck(sys::cuMemFreeHost(s.host)),
        Kind::Reg | Kind::RegHuge => {
            cuda::ck(sys::cuMemHostUnregister(s.host));
            assert_eq!(libc::munmap(s.host, s.bytes), 0);
        }
    }
}

const SRC: &str = r#"
extern "C" __global__ void copy16(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a = src[i], b = src[i + stride], c = src[i + 2 * stride], d = src[i + 3 * stride];
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = src[i];
}
extern "C" __global__ void copy_cpasync(const char* __restrict__ src, char* __restrict__ dst, const int* __restrict__ n_p, const int* __restrict__ tile_p) {
    extern __shared__ char smem[];
    size_t total = ((size_t)(*n_p)) << 4;
    size_t tile = (size_t)(*tile_p);
    size_t gstride = (size_t)gridDim.x * tile;
    for (size_t base = (size_t)blockIdx.x * tile; base < total; base += gstride) {
        size_t left = total - base;
        size_t cur = left < tile ? left : tile;
        unsigned int ncur = (unsigned int)(cur >> 4);
        for (unsigned int j = threadIdx.x; j < ncur; j += blockDim.x) {
            unsigned int s = (unsigned int)__cvta_generic_to_shared(smem + ((size_t)j << 4));
            const char* g = src + base + ((size_t)j << 4);
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(s), "l"(g) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        for (unsigned int j = threadIdx.x; j < ncur; j += blockDim.x) {
            *(uint4*)(dst + base + ((size_t)j << 4)) = *(const uint4*)(smem + ((size_t)j << 4));
        }
        __syncthreads();
    }
}
extern "C" __global__ void gather_cpa(const char* __restrict__ src, const unsigned long long* __restrict__ offs, const int* __restrict__ geo, char* __restrict__ dst) {
    extern __shared__ char smem[];
    size_t tpc = (size_t)geo[0];   // 4 KB tiles per chunk
    size_t total = tpc * (size_t)geo[1];
    for (size_t t = blockIdx.x; t < total; t += gridDim.x) {
        size_t c = t / tpc, w = (t % tpc) << 12;
        const char* g0 = src + offs[c] + w;
        for (unsigned int j = threadIdx.x; j < 256; j += blockDim.x) {
            unsigned int sa = (unsigned int)__cvta_generic_to_shared(smem + (j << 4));
            asm volatile("cp.async.cg.shared.global [%0], [%1], 16;" :: "r"(sa), "l"(g0 + (j << 4)) : "memory");
        }
        asm volatile("cp.async.commit_group;" ::: "memory");
        asm volatile("cp.async.wait_group 0;" ::: "memory");
        __syncthreads();
        char* d = dst + c * (tpc << 12) + w;
        for (unsigned int j = threadIdx.x; j < 256; j += blockDim.x)
            *(uint4*)(d + (j << 4)) = *(const uint4*)(smem + (j << 4));
        __syncthreads();
    }
}
extern "C" __global__ void cmp16(const uint4* __restrict__ a, const uint4* __restrict__ b, const int* __restrict__ n_p, unsigned int* __restrict__ out) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 x = a[i], y = b[i];
        if (x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w) atomicMin(out, (unsigned int)i);
    }
}
"#;

/// median GB/s of `reps` timed runs of `f` (one warm-up first)
unsafe fn med(reps: usize, mut f: impl FnMut()) -> (f64, f64) {
    f();
    cuda::sync();
    let mut v = Vec::with_capacity(reps);
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        f();
        cuda::sync();
        v.push(ONE_GIB as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    (v[v.len() / 2], v[0])
}

unsafe fn bw(kinds: &[Kind]) {
    let module = cuda::compile(SRC);
    let f16 = module.get("copy16");
    let fca = module.get("copy_cpasync");
    let fcmp = module.get("cmp16");
    cuda::ck(sys::cuFuncSetAttribute(fca, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536));
    let dst = cuda::alloc_zeroed(ONE_GIB);
    let n_p = cuda::to_i32_dev(&[(ONE_GIB / 16) as i32]);
    let tile_p = cuda::to_i32_dev(&[4096]);
    let out = cuda::alloc_zeroed(4);
    let reps = 9;
    for &k in kinds {
        let s = alloc(k, ONE_GIB);
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let hp = s.host as *mut u64;
        for i in 0..(ONE_GIB / 8) {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *hp.add(i) = x;
        }
        let (ca, ca_min) = med(reps, || {
            let args = [s.dev, dst, n_p, tile_p];
            let mut ptrs: Vec<*mut std::ffi::c_void> = args.iter().map(|v| v as *const u64 as *mut _).collect();
            cuda::ck(sys::cuLaunchKernel(fca, 40, 1, 1, 256, 1, 1, 4096, std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut()));
        });
        cuda::upload_into(out, &u32::MAX.to_le_bytes());
        launch_v(fcmp, 1024, 1, 1, 256, &[s.dev, dst, n_p, out]);
        cuda::sync();
        let eq = cuda::dtoh_u32(out, 1)[0] == u32::MAX;
        let (c16, c16_min) = med(reps, || launch_v(f16, 2048, 1, 1, 256, &[s.dev, dst, n_p]));
        let (ce, ce_min) = med(reps, || cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, s.host, ONE_GIB, std::ptr::null_mut())));
        println!(
            "[pin_return] bw {:<8} stage cp.async 4KB 40x256 {ca:5.1} GB/s (min {ca_min:5.1}, bytes equal {eq}) | copy16 2048x256 {c16:5.1} (min {c16_min:5.1}) | copy engine HtoD {ce:5.1} (min {ce_min:5.1})",
            format!("{k:?}")
        );
        free(s);
    }
}

fn main() {
    let a: Vec<String> = std::env::args().skip(1).collect();
    let cmd = a.first().map(|s| s.as_str()).unwrap_or("snap");
    if cmd == "snap" {
        snap("snapshot");
        return;
    }
    unsafe {
        let ctx = cuda::Ctx::init();
        match cmd {
            "bw" => {
                let kinds: Vec<Kind> = if a.len() > 1 {
                    a[1..].iter().map(|s| kind(s)).collect()
                } else {
                    vec![Kind::HostWc, Kind::Host, Kind::Reg, Kind::RegHuge]
                };
                bw(&kinds);
                drop(ctx);
            }
            "scatter" => {
                // expert-shaped random reads over a large tier: <gib> buffer, 1.76 MB
                // chunks (one gate_up expert), 24 chunks per launch, 200 launches
                let gib: usize = a.get(1).and_then(|v| v.parse().ok()).unwrap_or(8);
                let kinds: Vec<Kind> = if a.len() > 2 { a[2..].iter().map(|s| kind(s)).collect() } else { vec![Kind::HostWc, Kind::Host, Kind::Reg, Kind::RegHuge] };
                let module = cuda::compile(SRC);
                let fg = module.get("gather_cpa");
                cuda::ck(sys::cuFuncSetAttribute(fg, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536));
                let tpc = 430usize; // 430 x 4 KiB = 1.76 MB
                let chunk = tpc << 12;
                let per_launch = 24usize;
                let dst = cuda::alloc_zeroed(chunk * per_launch);
                let geo = cuda::to_i32_dev(&[tpc as i32, per_launch as i32]);
                let launches = 200usize;
                let mut x: u64 = 0x2545F4914F6CDD1D;
                let per = (gib << 30) / 8 & !0x1f_ffff;
                let nslots = per / chunk; // offsets inside one slab, the slab is drawn per launch
                let offs_host: Vec<u64> = (0..launches * per_launch).map(|_| { x ^= x << 13; x ^= x >> 7; x ^= x << 17; (x % nslots as u64) * chunk as u64 }).collect();
                let offs = cuda::to_u64_dev(&offs_host);
                for &k in &kinds {
                    let held: Vec<Slab> = (0..8).map(|_| alloc(k, per)).collect();
                    // one virtual tier: offsets index slab (off / per) at (off % per)
                    let run = |warm: bool| {
                        let t0 = std::time::Instant::now();
                        for l in 0..launches {
                            // 8 slabs like the engine's per-layer tier, one slab per launch (= one layer)
                            let slab = &held[(l * 5 + 3) % 8];
                            let offs_l = offs + (l * per_launch * 8) as u64;
                            let args = [slab.dev, offs_l, geo, dst];
                            let mut ptrs: Vec<*mut std::ffi::c_void> = args.iter().map(|v| v as *const u64 as *mut _).collect();
                            cuda::ck(sys::cuLaunchKernel(fg, 40, 1, 1, 256, 1, 1, 4096, std::ptr::null_mut(), ptrs.as_mut_ptr(), std::ptr::null_mut()));
                        }
                        cuda::sync();
                        let s = t0.elapsed().as_secs_f64();
                        if !warm { Some((launches * per_launch * chunk) as f64 / s / 1e9) } else { None }
                    };
                    let _ = run(true);
                    let mut v: Vec<f64> = (0..5).filter_map(|_| run(false)).collect();
                    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    println!("[pin_return] scatter {:<8} {gib} GiB tier, {launches} x {per_launch} random 1.76 MB chunks, 40x256 cp.async: median {:5.1} GB/s (min {:5.1})", format!("{k:?}"), v[2], v[0]);
                    for s in held { free(s); }
                }
            }
            "hold" => {
                let k = kind(a.get(1).expect("kind"));
                let gib: f64 = a.get(2).and_then(|v| v.parse().ok()).expect("gib");
                let end = a.get(3).cloned().unwrap_or_else(|| "free".into());
                let slabs: usize = a.get(4).and_then(|v| v.parse().ok()).unwrap_or(8);
                let hold_s: u64 = a.get(5).and_then(|v| v.parse().ok()).unwrap_or(0);
                let per = ((gib * GIB) as usize / slabs) & !0x1f_ffff; // 2 MiB aligned
                // under memory pressure this probe must be the OOM victim, never the desktop
                let _ = std::fs::write("/proc/self/oom_score_adj", "1000");
                snap("start");
                let t0 = std::time::Instant::now();
                let held: Vec<Slab> = (0..slabs).map(|_| alloc(k, per)).collect();
                println!("[pin_return] {slabs} x {:.0} MiB {k:?} allocated + touched in {:.2} s", per as f64 / (1 << 20) as f64, t0.elapsed().as_secs_f64());
                snap("held");
                if hold_s > 0 {
                    std::thread::sleep(std::time::Duration::from_secs(hold_s));
                }
                match end.as_str() {
                    "free" => {
                        let t0 = std::time::Instant::now();
                        for s in held {
                            free(s);
                        }
                        println!("[pin_return] freed in {:.2} s", t0.elapsed().as_secs_f64());
                        snap("after free (ctx alive)");
                        drop(ctx);
                        snap("after ctx release");
                    }
                    "exit" => {
                        std::mem::forget(held);
                        drop(ctx);
                        snap("exit without free");
                    }
                    "kill" => {
                        println!("[pin_return] SIGKILL self now");
                        libc::kill(libc::getpid(), libc::SIGKILL);
                    }
                    _ => panic!("end is free | exit | kill"),
                }
            }
            _ => panic!("command is snap | hold | bw"),
        }
    }
}
