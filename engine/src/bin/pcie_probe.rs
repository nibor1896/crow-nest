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

fn main() {
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(SRC);
        let f = module.get("copy16");
        let pinned = Pinned::alloc(GIB);
        let pinned_wc = Pinned::alloc_wc(GIB);
        for i in (0..GIB).step_by(4096) {
            *(pinned.host as *mut u8).add(i) = (i >> 12) as u8;
            *(pinned_wc.host as *mut u8).add(i) = (i >> 12) as u8;
        }
        let dst = cuda::alloc_zeroed(GIB);
        let vsrc = cuda::alloc_zeroed(GIB);
        let n_p = cuda::to_i32_dev(&[(GIB / 16) as i32]);
        let half_p = cuda::to_i32_dev(&[(GIB / 32) as i32]);
        let reps = 5;
        println!("PCIe/staging probe, 1 GiB per measurement, {reps} reps each");
        // a) today's shape
        println!("[pcie] a) kernel pinned->VRAM 160x256  : {:6.1} GB/s", time_kernel(f, pinned.dev, dst, n_p, 160, 256, reps));
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
        cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, pinned.host, GIB, std::ptr::null_mut()));
        cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            cuda::ck(sys::cuMemcpyHtoDAsync_v2(dst, pinned.host, GIB, std::ptr::null_mut()));
        }
        cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
        println!("[pcie] e) cuMemcpyHtoDAsync pinned->VRAM : {:6.1} GB/s (copy engine)", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
        // f) DtoD from the mapped pointer
        let t0 = std::time::Instant::now();
        for _ in 0..reps {
            cuda::ck(sys::cuMemcpyDtoDAsync_v2(dst, pinned.dev, GIB, std::ptr::null_mut()));
        }
        cuda::ck(sys::cuStreamSynchronize(std::ptr::null_mut()));
        println!("[pcie] f) cuMemcpyDtoDAsync mapped->VRAM : {:6.1} GB/s", GIB as f64 / (t0.elapsed().as_secs_f64() / reps as f64) / 1e9);
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
}
