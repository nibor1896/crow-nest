//! Measurement A-V1 / C-D3a (2026-09-04): is the ~34 GB/s cold-expert staging
//! rate the PCIe link, the host DRAM, the copy kernel's occupancy, or the
//! device-initiated-read path? Variants over a fixed 1 GiB volume:
//!   a) kernel copy pinned->VRAM, grid like today's stage_cold (160 blocks x 256)
//!   b) kernel copy with 512 / 2048 / 8192 blocks, block 256 and 512
//!   c) same kernel with a VRAM source (removes PCIe: the kernel's own ceiling)
//!   d) two concurrent kernels on two streams (does aggregate exceed one?)
//!   e) cuMemcpyHtoDAsync pinned->VRAM (copy engine DMA)
//!   f) cuMemcpyDtoDAsync from the mapped host pointer (driver-side DMA path)
//!   g) write-combined pinned source with variant a
//! No model is loaded; 1 GiB pinned + 2 GiB VRAM.
//!
//! #19c (2026-09-11): device-issued copy forms that today's kernel does not use,
//! same 1 GiB volume, same process, same buffers as a / e / f:
//!   i) 256-bit loads: ld.global.v8.b32 asm, and 8 outstanding uint4 loads
//!   j) non-coherent loads: __ldg 4 and 8 deep, ld.global.nc.L1::no_allocate
//!   k) cp.async.cg.shared.global into shared, then coalesced stores to VRAM
//!   l) TMA 1D bulk copy cp.async.bulk into shared, bulk store to VRAM
//!   x) the three fastest forms at 20 to 80 blocks (beyond the brief's grids)
//!   m) the best of i to x on 2 and 4 concurrent streams (aggregate)
//!   n) byte check of every variant against the pinned source (cmp16 kernel)
//! A variant whose PTX the JIT rejects on compute_120a is recorded with the
//! error text as "not available", never skipped. Group filter on argv:
//! `pcie_probe base i j k l x m` (no argument runs every group).
use crow_nest_engine::cuda::{self, Pinned};
use crow_nest_engine::gen::launch_v;
use cudarc::driver::sys;

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
"#;

/// #19c n) byte check: first differing 16 B element index, 0xFFFFFFFF = equal
const SRC_CMP: &str = r##"
extern "C" __global__ void cmp16(const uint4* __restrict__ a, const uint4* __restrict__ b, const int* __restrict__ n_p, unsigned int* __restrict__ out) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        uint4 x = a[i], y = b[i];
        if (x.x != y.x || x.y != y.y || x.z != y.z || x.w != y.w) atomicMin(out, (unsigned int)i);
    }
}
"##;

/// #19c i) 256-bit loads, one 32 B load and one 32 B store per thread per step
const SRC_V8: &str = r##"
extern "C" __global__ void copy_v8(const char* __restrict__ src, char* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n32 = ((size_t)(*n_p)) >> 1;
    size_t stride = (size_t)gridDim.x * blockDim.x;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n32; i += stride) {
        unsigned int r0, r1, r2, r3, r4, r5, r6, r7;
        const char* p = src + (i << 5);
        char* q = dst + (i << 5);
        asm volatile("ld.global.v8.b32 {%0,%1,%2,%3,%4,%5,%6,%7}, [%8];"
            : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3), "=r"(r4), "=r"(r5), "=r"(r6), "=r"(r7)
            : "l"(p) : "memory");
        asm volatile("st.global.v8.b32 [%0], {%1,%2,%3,%4,%5,%6,%7,%8};"
            :: "l"(q), "r"(r0), "r"(r1), "r"(r2), "r"(r3), "r"(r4), "r"(r5), "r"(r6), "r"(r7)
            : "memory");
    }
}
"##;

/// #19c i) 8 outstanding 16 B loads per thread (128 B in flight per thread)
const SRC_X8: &str = r##"
extern "C" __global__ void copy16_x8(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 7 * stride < n; i += 8 * stride) {
        uint4 v0 = src[i], v1 = src[i + stride], v2 = src[i + 2 * stride], v3 = src[i + 3 * stride];
        uint4 v4 = src[i + 4 * stride], v5 = src[i + 5 * stride], v6 = src[i + 6 * stride], v7 = src[i + 7 * stride];
        dst[i] = v0; dst[i + stride] = v1; dst[i + 2 * stride] = v2; dst[i + 3 * stride] = v3;
        dst[i + 4 * stride] = v4; dst[i + 5 * stride] = v5; dst[i + 6 * stride] = v6; dst[i + 7 * stride] = v7;
    }
    for (; i < n; i += stride) dst[i] = src[i];
}
"##;

/// #19c j) non-coherent cached loads through __ldg, 4 and 8 in flight
const SRC_NC: &str = r##"
extern "C" __global__ void copy_nc4(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a = __ldg(&src[i]), b = __ldg(&src[i + stride]), c = __ldg(&src[i + 2 * stride]), d = __ldg(&src[i + 3 * stride]);
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = __ldg(&src[i]);
}
extern "C" __global__ void copy_nc8(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 7 * stride < n; i += 8 * stride) {
        uint4 v0 = __ldg(&src[i]), v1 = __ldg(&src[i + stride]), v2 = __ldg(&src[i + 2 * stride]), v3 = __ldg(&src[i + 3 * stride]);
        uint4 v4 = __ldg(&src[i + 4 * stride]), v5 = __ldg(&src[i + 5 * stride]), v6 = __ldg(&src[i + 6 * stride]), v7 = __ldg(&src[i + 7 * stride]);
        dst[i] = v0; dst[i + stride] = v1; dst[i + 2 * stride] = v2; dst[i + 3 * stride] = v3;
        dst[i + 4 * stride] = v4; dst[i + 5 * stride] = v5; dst[i + 6 * stride] = v6; dst[i + 7 * stride] = v7;
    }
    for (; i < n; i += stride) dst[i] = __ldg(&src[i]);
}
"##;

/// #19c j) ld.global.nc.L1::no_allocate, 4 x 16 B in flight per thread
const SRC_NCNA: &str = r##"
extern "C" __global__ void copy_nc_na(const uint4* __restrict__ src, uint4* __restrict__ dst, const int* __restrict__ n_p) {
    size_t n = (size_t)(*n_p);
    size_t stride = (size_t)gridDim.x * blockDim.x;
    size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    for (; i + 3 * stride < n; i += 4 * stride) {
        uint4 a, b, c, d;
        const uint4* p0 = &src[i];
        const uint4* p1 = &src[i + stride];
        const uint4* p2 = &src[i + 2 * stride];
        const uint4* p3 = &src[i + 3 * stride];
        asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(a.x), "=r"(a.y), "=r"(a.z), "=r"(a.w) : "l"(p0) : "memory");
        asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(b.x), "=r"(b.y), "=r"(b.z), "=r"(b.w) : "l"(p1) : "memory");
        asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(c.x), "=r"(c.y), "=r"(c.z), "=r"(c.w) : "l"(p2) : "memory");
        asm volatile("ld.global.nc.L1::no_allocate.v4.b32 {%0,%1,%2,%3}, [%4];"
            : "=r"(d.x), "=r"(d.y), "=r"(d.z), "=r"(d.w) : "l"(p3) : "memory");
        dst[i] = a; dst[i + stride] = b; dst[i + 2 * stride] = c; dst[i + 3 * stride] = d;
    }
    for (; i < n; i += stride) dst[i] = src[i];
}
"##;

/// #19c k) cp.async.cg.shared.global 16 B per thread into a shared tile
const SRC_CPA: &str = r##"
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
"##;

/// #19c l) TMA 1D bulk copy, one mbarrier per block, bulk store back to VRAM
const SRC_TMA: &str = r##"
extern "C" __global__ void copy_tma(const char* __restrict__ src, char* __restrict__ dst, const int* __restrict__ n_p, const int* __restrict__ tile_p) {
    extern __shared__ char smem_raw[];
    __shared__ unsigned long long bar;
    size_t total = ((size_t)(*n_p)) << 4;
    size_t tile = (size_t)(*tile_p);
    unsigned int sbar = (unsigned int)__cvta_generic_to_shared(&bar);
    unsigned int ssm = ((unsigned int)__cvta_generic_to_shared(smem_raw) + 127u) & ~127u;
    if (threadIdx.x == 0) asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" :: "r"(sbar) : "memory");
    __syncthreads();
    unsigned int phase = 0;
    size_t gstride = (size_t)gridDim.x * tile;
    for (size_t base = (size_t)blockIdx.x * tile; base < total; base += gstride) {
        size_t left = total - base;
        unsigned int cur = (unsigned int)(left < tile ? left : tile);
        if (threadIdx.x == 0) {
            asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(sbar), "r"(cur) : "memory");
            asm volatile("cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];"
                :: "r"(ssm), "l"(src + base), "r"(cur), "r"(sbar) : "memory");
        }
        unsigned int done = 0;
        while (done == 0) {
            asm volatile("{ .reg .pred P; mbarrier.try_wait.parity.shared::cta.b64 P, [%1], %2; selp.b32 %0, 1, 0, P; }"
                : "=r"(done) : "r"(sbar), "r"(phase) : "memory");
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;"
                :: "l"(dst + base), "r"(ssm), "r"(cur) : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
            asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
        }
        __syncthreads();
        phase ^= 1;
    }
}
"##;

const GIB: usize = 1 << 30;

unsafe fn time_kernel(f: sys::CUfunction, src: u64, dst: u64, n_p: u64, gx: u32, bx: u32, reps: u32) -> f64 {
    launch_v(f, gx, 1, 1, bx, &[src, dst, n_p]);
    cuda::sync();
    let t0 = std::time::Instant::now();
    for _ in 0..reps {
        launch_v(f, gx, 1, 1, bx, &[src, dst, n_p]);
    }
    cuda::sync();
    let s = t0.elapsed().as_secs_f64() / reps as f64;
    GIB as f64 / s / 1e9
}

// ---------- #19c helpers ----------

/// NVRTC for compute_120a plus a JIT load that returns the PTX error text
/// instead of panicking (a variant the JIT rejects is a result, not a stop).
unsafe fn try_compile(src: &str) -> Result<cuda::Module, String> {
    let opts = cudarc::nvrtc::CompileOptions {
        options: vec!["--gpu-architecture=compute_120a".into()],
        ..Default::default()
    };
    let ptx = match cudarc::nvrtc::compile_ptx_with_opts(src, opts) {
        Ok(p) => p,
        Err(e) => {
            let s = format!("nvrtc: {e:?}").replace('\n', " | ");
            return Err(s.chars().take(400).collect());
        }
    };
    let c = std::ffi::CString::new(ptx.to_src()).unwrap();
    let mut m: sys::CUmodule = std::ptr::null_mut();
    let mut log = vec![0u8; 8192];
    let mut jopt = [
        sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER,
        sys::CUjit_option_enum::CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    ];
    let mut jval: [*mut std::ffi::c_void; 2] =
        [log.as_mut_ptr() as *mut std::ffi::c_void, 8192usize as *mut std::ffi::c_void];
    let r = sys::cuModuleLoadDataEx(
        &mut m,
        c.as_ptr() as *const std::ffi::c_void,
        2,
        jopt.as_mut_ptr(),
        jval.as_mut_ptr(),
    );
    if r != sys::CUresult::CUDA_SUCCESS {
        let end = log.iter().position(|&b| b == 0).unwrap_or(0);
        let txt = String::from_utf8_lossy(&log[..end]).replace('\n', " | ");
        return Err(format!("jit: {r:?} {}", txt.chars().take(400).collect::<String>()));
    }
    Ok(cuda::Module(m))
}

fn soft(r: sys::CUresult, what: &str) -> Result<(), String> {
    if r == sys::CUresult::CUDA_SUCCESS {
        Ok(())
    } else {
        Err(format!("{what} {r:?}"))
    }
}

unsafe fn launch_raw(
    f: sys::CUfunction,
    gx: u32,
    bx: u32,
    smem: u32,
    stream: sys::CUstream,
    vals: &[u64],
) -> sys::CUresult {
    let mut ptrs: Vec<*mut std::ffi::c_void> = vals
        .iter()
        .map(|v| v as *const u64 as *mut std::ffi::c_void)
        .collect();
    sys::cuLaunchKernel(
        f,
        gx,
        1,
        1,
        bx,
        1,
        1,
        smem,
        stream,
        ptrs.as_mut_ptr(),
        std::ptr::null_mut(),
    )
}

unsafe fn ctx_alive() -> bool {
    sys::cuCtxSynchronize() == sys::CUresult::CUDA_SUCCESS
}

/// 0xAA over the destination so that a variant which copies nothing cannot
/// inherit the previous variant's correct bytes and pass the byte check
unsafe fn poison(dst: u64) {
    let _ = soft(sys::cuMemsetD8_v2(dst, 0xAA, GIB), "memset");
    let _ = soft(sys::cuStreamSynchronize(std::ptr::null_mut()), "sync");
}

/// median and min GB/s over `reps` timed launches, warm-up launch first
unsafe fn measure(
    f: sys::CUfunction,
    gx: u32,
    bx: u32,
    smem: u32,
    src: u64,
    dst: u64,
    n_p: u64,
    tile_p: Option<u64>,
    reps: u32,
) -> Result<(f64, f64), String> {
    let mut args: Vec<u64> = vec![src, dst, n_p];
    if let Some(t) = tile_p {
        args.push(t);
    }
    soft(launch_raw(f, gx, bx, smem, std::ptr::null_mut(), &args), "launch")?;
    soft(sys::cuStreamSynchronize(std::ptr::null_mut()), "sync")?;
    let mut v: Vec<f64> = Vec::new();
    for _ in 0..reps {
        let t0 = std::time::Instant::now();
        soft(launch_raw(f, gx, bx, smem, std::ptr::null_mut(), &args), "launch")?;
        soft(sys::cuStreamSynchronize(std::ptr::null_mut()), "sync")?;
        v.push(GIB as f64 / t0.elapsed().as_secs_f64() / 1e9);
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Ok((v[v.len() / 2], v[0]))
}

/// n) byte check: the whole 1 GiB destination against the pinned source
unsafe fn bytes_eq(cmp_f: sys::CUfunction, src: u64, dst: u64, n_p: u64, out: u64) -> String {
    cuda::upload_into(out, &u32::MAX.to_le_bytes());
    if let Err(e) = soft(
        launch_raw(cmp_f, 1024, 256, 0, std::ptr::null_mut(), &[src, dst, n_p, out]),
        "cmp launch",
    ) {
        return format!("check failed ({e})");
    }
    if let Err(e) = soft(sys::cuStreamSynchronize(std::ptr::null_mut()), "cmp sync") {
        return format!("check failed ({e})");
    }
    let r = cuda::dtoh_u32(out, 1)[0];
    if r == u32::MAX {
        "yes".to_string()
    } else {
        format!("no, first diff at byte {}", r as u64 * 16)
    }
}

#[derive(Clone)]
struct Cand {
    label: String,
    f: sys::CUfunction,
    gx: u32,
    bx: u32,
    smem: u32,
    tile_p: Option<u64>,
    med: f64,
}

fn main() {
    unsafe {
        let argv: Vec<String> = std::env::args().skip(1).collect();
        let want = |g: &str| argv.is_empty() || argv.iter().any(|a| a == g);
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(SRC);
        let f = module.get("copy16");
        let cmp_mod = match try_compile(SRC_CMP) {
            Ok(m) => m,
            Err(e) => {
                println!("[pcie] n) cmp16 not available: {e}");
                return;
            }
        };
        let cmp_f = cmp_mod.get("cmp16");
        let mut pinned = Pinned::alloc(GIB);
        let pinned_wc = Pinned::alloc_wc(GIB);
        // #19c: the whole 1 GiB carries data so that the byte check is a real
        // check (before 2026-09-11 only every 4096th byte was written)
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let hp = pinned.host as *mut u64;
        let wp = pinned_wc.host as *mut u64;
        for i in 0..(GIB / 8) {
            x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            *hp.add(i) = x;
            *wp.add(i) = x;
        }
        for i in (0..GIB).step_by(4096) {
            *(pinned.host as *mut u8).add(i) = (i >> 12) as u8;
            *(pinned_wc.host as *mut u8).add(i) = (i >> 12) as u8;
        }
        let dst = cuda::alloc_zeroed(GIB);
        let vsrc = cuda::alloc_zeroed(GIB);
        let n_p = cuda::to_i32_dev(&[(GIB / 16) as i32]);
        let half_p = cuda::to_i32_dev(&[(GIB / 32) as i32]);
        let quarter_p = cuda::to_i32_dev(&[(GIB / 64) as i32]);
        let cmp_out = cuda::alloc_zeroed(4);
        let reps = 5;
        println!("PCIe/staging probe, 1 GiB per measurement, {reps} reps each");
        println!("[pcie] groups requested: {}", if argv.is_empty() { "all".to_string() } else { argv.join(",") });
        if want("base") {
            // a) today's shape
            println!("[pcie] a) kernel pinned->VRAM 160x256  : {:6.1} GB/s", time_kernel(f, pinned.dev, dst, n_p, 160, 256, reps));
            println!("[pcie] n) byte check a) 160x256          : bytes equal: {}", bytes_eq(cmp_f, pinned.dev, dst, n_p, cmp_out));
            // b) more blocks
            for &(gx, bx) in &[(512u32, 256u32), (2048, 256), (8192, 256), (2048, 512), (8192, 512)] {
                println!("[pcie] b) kernel pinned->VRAM {gx}x{bx}: {:6.1} GB/s", time_kernel(f, pinned.dev, dst, n_p, gx, bx, reps));
            }
            // g) write-combined source
            println!("[pcie] g) kernel WC-pinned->VRAM 2048x256: {:6.1} GB/s", time_kernel(f, pinned_wc.dev, dst, n_p, 2048, 256, reps));
            // c) VRAM source: kernel ceiling without PCIe
            println!("[pcie] c) kernel VRAM->VRAM 2048x256   : {:6.1} GB/s", time_kernel(f, vsrc, dst, n_p, 2048, 256, reps));
            // d) two concurrent kernels on two streams, each half
            let s1 = cuda::stream_create_non_blocking();
            let s2 = cuda::stream_create_non_blocking();
            let half = (GIB / 2) as u64;
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                cuda::set_stream(s1 as u64);
                launch_v(f, 2048, 1, 1, 256, &[pinned.dev, dst, half_p]);
                cuda::set_stream(s2 as u64);
                launch_v(f, 2048, 1, 1, 256, &[pinned.dev + half, dst + half, half_p]);
            }
            cuda::ck(sys::cuStreamSynchronize(s1));
            cuda::ck(sys::cuStreamSynchronize(s2));
            cuda::set_stream(0);
            println!("[pcie] d) two streams, 2 x 0.5 GiB     : {:6.1} GB/s aggregate", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
            // e) copy engine HtoD
            poison(dst);
            cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, pinned.host, GIB, std::ptr::null_mut()));
            cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, pinned.host, GIB, std::ptr::null_mut()));
            }
            cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            println!("[pcie] e) cuMemcpyHtoDAsync pinned->VRAM : {:6.1} GB/s (copy engine)", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
            println!("[pcie] n) byte check e) copy engine      : bytes equal: {}", bytes_eq(cmp_f, pinned.dev, dst, n_p, cmp_out));
            // f) DtoD from the mapped pointer
            poison(dst);
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                cuda::ck(sys::cuMemcpyDtoDAsync_v2(dst, pinned.dev, GIB, std::ptr::null_mut()));
            }
            cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            println!("[pcie] f) cuMemcpyDtoDAsync mapped->VRAM : {:6.1} GB/s", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
            println!("[pcie] n) byte check f) mapped DtoD      : bytes equal: {}", bytes_eq(cmp_f, pinned.dev, dst, n_p, cmp_out));
            // e2) copy engine from WC pinned
            let t0 = std::time::Instant::now();
            for _ in 0..reps {
                cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, pinned_wc.host, GIB, std::ptr::null_mut()));
            }
            cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
            println!("[pcie] e2) cuMemcpyHtoDAsync WC-pinned   : {:6.1} GB/s", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
            // host DRAM read ceiling (single thread memcpy 1 GiB) for wall-1 context
            let mut hbuf = vec![0u8; GIB];
            let t0 = std::time::Instant::now();
            std::ptr::copy_nonoverlapping(pinned.host as *const u8, hbuf.as_mut_ptr(), GIB);
            println!("[pcie] h) host memcpy pinned->heap 1 thread: {:6.1} GB/s ({})", GIB as f64 / t0.elapsed().as_secs_f64() / 1e9, hbuf[4096]);
        }

        // ---------- #19c device-issued variants ----------
        let mut cands: Vec<Cand> = Vec::new();
        let mut tile_devs: Vec<(u32, u64)> = Vec::new();
        for &t in &[4096u32, 16384, 32768, 49152] {
            tile_devs.push((t, cuda::to_i32_dev(&[t as i32])));
        }
        let tile_dev = |t: u32| -> u64 { tile_devs.iter().find(|e| e.0 == t).unwrap().1 };
        let run = |label: String,
                       shape: String,
                       inflight: String,
                       fk: sys::CUfunction,
                       gx: u32,
                       bx: u32,
                       smem: u32,
                       tile_p: Option<u64>,
                       cands: &mut Vec<Cand>|
         -> bool {
            poison(dst);
            match measure(fk, gx, bx, smem, pinned.dev, dst, n_p, tile_p, reps) {
                Ok((med, mn)) => {
                    let eq = bytes_eq(cmp_f, pinned.dev, dst, n_p, cmp_out);
                    println!(
                        "[pcie] {label} {shape} : med {med:6.1} GB/s  min {mn:6.1} GB/s  in-flight {inflight}  bytes equal: {eq}"
                    );
                    if eq == "yes" {
                        cands.push(Cand { label: format!("{label} {shape}"), f: fk, gx, bx, smem, tile_p, med });
                    }
                    true
                }
                Err(e) => {
                    println!("[pcie] {label} {shape} : FAILED {e}");
                    ctx_alive()
                }
            }
        };
        let mut remaining: Vec<&str> = Vec::new();
        let groups = ["i", "j", "k", "l", "m"];
        // kept for the x) low-block bracket after l; Module has no Drop, the
        // function handle stays valid after the Module value goes out of scope
        let mut f_v8: Option<sys::CUfunction> = None;
        let mut f_cpa: Option<sys::CUfunction> = None;
        let mut f_tma: Option<sys::CUfunction> = None;

        // i) 256-bit loads
        if want("i") {
            match try_compile(SRC_V8) {
                Ok(m) => {
                    let fk = m.get("copy_v8");
                    f_v8 = Some(fk);
                    for &gx in &[160u32, 320] {
                        if !run("i-a) ld.global.v8.b32".into(), format!("{gx}x256"), "32 B/thread".into(), fk, gx, 256, 0, None, &mut cands) {
                            remaining = groups.iter().skip_while(|g| **g != "i").skip(1).cloned().collect();
                        }
                    }
                }
                Err(e) => println!("[pcie] i-a) ld.global.v8.b32 : not available on sm_120a: {e}"),
            }
            match try_compile(SRC_X8) {
                Ok(m) => {
                    let fk = m.get("copy16_x8");
                    for &gx in &[160u32, 320] {
                        run("i-b) 8x uint4 in flight".into(), format!("{gx}x256"), "128 B/thread".into(), fk, gx, 256, 0, None, &mut cands);
                    }
                }
                Err(e) => println!("[pcie] i-b) 8x uint4 in flight : not available on sm_120a: {e}"),
            }
        }

        // j) non-coherent loads
        if want("j") && remaining.is_empty() {
            match try_compile(SRC_NC) {
                Ok(m) => {
                    let f4 = m.get("copy_nc4");
                    let f8 = m.get("copy_nc8");
                    run("j-a) __ldg 4 in flight".into(), "160x256".into(), "64 B/thread".into(), f4, 160, 256, 0, None, &mut cands);
                    run("j-b) __ldg 8 in flight".into(), "160x256".into(), "128 B/thread".into(), f8, 160, 256, 0, None, &mut cands);
                }
                Err(e) => println!("[pcie] j-a/j-b) __ldg : not available on sm_120a: {e}"),
            }
            match try_compile(SRC_NCNA) {
                Ok(m) => {
                    let fk = m.get("copy_nc_na");
                    run("j-c) nc.L1::no_allocate".into(), "160x256".into(), "64 B/thread".into(), fk, 160, 256, 0, None, &mut cands);
                }
                Err(e) => println!("[pcie] j-c) nc.L1::no_allocate : not available on sm_120a: {e}"),
            }
        }

        // k) cp.async into shared, coalesced store to VRAM
        if want("k") && remaining.is_empty() {
            match try_compile(SRC_CPA) {
                Ok(m) => {
                    let fk = m.get("copy_cpasync");
                    f_cpa = Some(fk);
                    let _ = soft(
                        sys::cuFuncSetAttribute(fk, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 65536),
                        "smem attr",
                    );
                    for &tile in &[4096u32, 16384, 32768] {
                        for &gx in &[160u32, 320, 640] {
                            run(
                                format!("k) cp.async.cg tile {} KB", tile / 1024),
                                format!("{gx}x256"),
                                format!("{} B/block", tile),
                                fk,
                                gx,
                                256,
                                tile,
                                Some(tile_dev(tile)),
                                &mut cands,
                            );
                        }
                    }
                }
                Err(e) => println!("[pcie] k) cp.async.cg.shared.global : not available on sm_120a: {e}"),
            }
        }

        // l) TMA 1D bulk copy
        if want("l") && remaining.is_empty() {
            match try_compile(SRC_TMA) {
                Ok(m) => {
                    let fk = m.get("copy_tma");
                    f_tma = Some(fk);
                    let _ = soft(
                        sys::cuFuncSetAttribute(fk, sys::CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES, 98304),
                        "smem attr",
                    );
                    // l0) one block from the VRAM source: does the instruction run at all
                    poison(dst);
                    let smoke_vram = measure(fk, 1, 128, 16384 + 128, vsrc, dst, n_p, Some(tile_dev(16384)), 1);
                    match &smoke_vram {
                        Ok(_) => println!("[pcie] l0) TMA smoke VRAM source 1x128 : ran, instruction available on sm_120a"),
                        Err(e) => println!("[pcie] l0) TMA smoke VRAM source 1x128 : FAILED {e}"),
                    }
                    if smoke_vram.is_ok() && ctx_alive() {
                        // l1) one block from the pinned host UVA address
                        poison(dst);
                        let smoke_host = measure(fk, 1, 128, 16384 + 128, pinned.dev, dst, n_p, Some(tile_dev(16384)), 1);
                        match &smoke_host {
                            Ok(_) => println!("[pcie] l1) TMA smoke host UVA 1x128   : ran, host pointer accepted"),
                            Err(e) => println!("[pcie] l1) TMA smoke host UVA 1x128   : FAILED {e}"),
                        }
                        if smoke_host.is_ok() && ctx_alive() {
                            for &tile in &[16384u32, 32768, 49152] {
                                for &gx in &[80u32, 160, 320] {
                                    if !run(
                                        format!("l) TMA bulk tile {} KB", tile / 1024),
                                        format!("{gx}x128"),
                                        format!("{} B/block", tile),
                                        fk,
                                        gx,
                                        128,
                                        tile + 128,
                                        Some(tile_dev(tile)),
                                        &mut cands,
                                    ) {
                                        remaining = vec!["m"];
                                    }
                                }
                            }
                        } else if !ctx_alive() {
                            remaining = vec!["m"];
                        }
                    } else if !ctx_alive() {
                        remaining = vec!["m"];
                    }
                }
                Err(e) => println!("[pcie] l) TMA cp.async.bulk : not available on sm_120a: {e}"),
            }
        }

        // x) beyond the #19c brief's grid list: every form of i to l rises as the
        // block count falls, so the three fastest forms get two smaller grids to
        // bracket the optimum (rows marked x, the brief's rows are unchanged)
        if want("x") && remaining.is_empty() {
            if let Some(fk) = f_v8 {
                for &gx in &[80u32, 40] {
                    run("x-i-a) ld.global.v8.b32".into(), format!("{gx}x256"), "32 B/thread".into(), fk, gx, 256, 0, None, &mut cands);
                }
            }
            if let Some(fk) = f_cpa {
                for &gx in &[80u32, 40] {
                    run("x-k) cp.async.cg tile 4 KB".into(), format!("{gx}x256"), "4096 B/block".into(), fk, gx, 256, 4096, Some(tile_dev(4096)), &mut cands);
                }
            }
            if let Some(fk) = f_tma {
                for &gx in &[40u32, 20] {
                    run("x-l) TMA bulk tile 16 KB".into(), format!("{gx}x128"), "16384 B/block".into(), fk, gx, 128, 16384 + 128, Some(tile_dev(16384)), &mut cands);
                }
            }
        }

        // m) the best device-issued variant on 2 and 4 concurrent streams
        if want("m") && remaining.is_empty() {
            let best = cands.iter().cloned().fold(None::<Cand>, |acc, c| match acc {
                Some(b) if b.med >= c.med => Some(b),
                _ => Some(c),
            });
            match best {
                None => println!("[pcie] m) no device-issued variant measured, nothing to aggregate"),
                Some(b) => {
                    println!("[pcie] m) best single-stream variant: {} at {:6.1} GB/s median", b.label, b.med);
                    for &ns in &[2usize, 4usize] {
                        // two forms: the winner's grid on EVERY stream (more total
                        // blocks, the form of variant d), and the winner's grid
                        // SPLIT over the streams (same total blocks)
                        for &(gxs, form) in &[
                            (b.gx, "grid per stream"),
                            (std::cmp::max(1, b.gx / ns as u32), "grid split"),
                        ] {
                            let part = GIB / ns;
                            let np = if ns == 2 { half_p } else { quarter_p };
                            let streams: Vec<sys::CUstream> = (0..ns).map(|_| cuda::stream_create_non_blocking()).collect();
                            poison(dst);
                            let one = || -> Result<(), String> {
                                for (s, st) in streams.iter().enumerate() {
                                    let mut args: Vec<u64> = vec![pinned.dev + (s * part) as u64, dst + (s * part) as u64, np];
                                    if let Some(t) = b.tile_p {
                                        args.push(t);
                                    }
                                    soft(launch_raw(b.f, gxs, b.bx, b.smem, *st, &args), "launch")?;
                                }
                                for st in streams.iter() {
                                    soft(sys::cuStreamSynchronize(*st), "sync")?;
                                }
                                Ok(())
                            };
                            match one() {
                                Err(e) => println!("[pcie] m) {ns} streams {form} {gxs} blocks : FAILED {e}"),
                                Ok(()) => {
                                    let mut v: Vec<f64> = Vec::new();
                                    let mut err: Option<String> = None;
                                    for _ in 0..reps {
                                        let t0 = std::time::Instant::now();
                                        if let Err(e) = one() {
                                            err = Some(e);
                                            break;
                                        }
                                        v.push(GIB as f64 / t0.elapsed().as_secs_f64() / 1e9);
                                    }
                                    match err {
                                        Some(e) => println!("[pcie] m) {ns} streams {form} {gxs} blocks : FAILED {e}"),
                                        None => {
                                            v.sort_by(|a, c| a.partial_cmp(c).unwrap());
                                            let eq = bytes_eq(cmp_f, pinned.dev, dst, n_p, cmp_out);
                                            println!(
                                                "[pcie] m) {ns} streams {form} {gxs}x{} of {} : med {:6.1} GB/s  min {:6.1} GB/s aggregate  bytes equal: {eq}",
                                                b.bx,
                                                b.label,
                                                v[v.len() / 2],
                                                v[0]
                                            );
                                        }
                                    }
                                }
                            }
                            for st in streams.iter() {
                                cuda::stream_destroy(*st);
                            }
                        }
                    }
                }
            }
        }

        if !remaining.is_empty() {
            println!("[pcie] POISONED context, remaining groups need a second invocation: {}", remaining.join(" "));
            pinned.free();
            std::process::exit(3);
        }
        println!("[pcie] done, context alive: {}", ctx_alive());
    }
}
