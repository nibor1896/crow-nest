//! Probe 3 (Crow #8 pre-study): zero-copy read bandwidth — GPU reading PINNED HOST
//! memory directly (UVA), no copy, no WDDM submission. This is the cold-path
//! candidate: expert GEMMs reading their weights straight from the RAM tier.
//!
//! Kernel: grid-strided xor-reduce over 1 GiB of mapped host memory (coalesced u32
//! loads), one output word per block. The xor chain prevents load elimination.
//! Measures: effective host-read GB/s at 5 buffer sizes × 3 reps.

use std::ffi::CString;
use std::time::Instant;

use cudarc::driver::sys::{self, CUdeviceptr, CUresult};
use cudarc::nvrtc::{compile_ptx_with_opts, CompileOptions};

const KERNEL_SRC: &str = r#"
extern "C" __global__ void read_bw(const unsigned int* __restrict__ src,
                                   unsigned int* __restrict__ out,
                                   unsigned long long n_words) {
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int acc = 0;
    for (; i < n_words; i += stride) {
        acc ^= src[i] + 0x9E3779B9u;   // dependency chain: loads cannot be eliminated
        acc = (acc << 7) | (acc >> 25);
    }
    if (threadIdx.x == 0) out[blockIdx.x] = acc;
}
"#;

fn ck(r: CUresult) {
    if r != CUresult::CUDA_SUCCESS {
        panic!("CUDA error: {r:?}");
    }
}

fn main() {
    unsafe {
        ck(sys::cuInit(0));
        let mut dev = 0;
        ck(sys::cuDeviceGet(&mut dev, 0));
        let mut ctx = std::ptr::null_mut();
        ck(sys::cuDevicePrimaryCtxRetain(&mut ctx, dev));
        ck(sys::cuCtxSetCurrent(ctx));

        let opts = CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = compile_ptx_with_opts(KERNEL_SRC, opts).expect("nvrtc");
        let c_ptx = CString::new(ptx.to_src()).unwrap();
        let mut module = std::ptr::null_mut();
        ck(sys::cuModuleLoadData(&mut module, c_ptx.as_ptr() as *const _));
        let mut fn_read = std::ptr::null_mut();
        ck(sys::cuModuleGetFunction(&mut fn_read, module, CString::new("read_bw").unwrap().as_ptr()));

        // pinned, device-mapped source buffer (1 GiB)
        let bytes: usize = 1 << 30;
        let mut host: *mut std::ffi::c_void = std::ptr::null_mut();
        ck(sys::cuMemHostAlloc(
            &mut host,
            bytes,
            sys::CU_MEMHOSTALLOC_PORTABLE | sys::CU_MEMHOSTALLOC_DEVICEMAP,
        ));
        // fill via host pointer (xor pattern so no page is zero-trivial)
        let words = bytes / 4;
        let src = host as *mut u32;
        for i in 0..words {
            *src.add(i) = (i as u32) ^ 0xA5A5A5A5;
        }
        let src_dev = host as CUdeviceptr;

        let mut out_dev: CUdeviceptr = 0;
        ck(sys::cuMemAlloc_v2(&mut out_dev, 65536));
        let mut s = std::ptr::null_mut();
        ck(sys::cuStreamCreate(&mut s, sys::CUstream_flags::CU_STREAM_NON_BLOCKING as u32));
        let mut ev0 = std::ptr::null_mut();
        let mut ev1 = std::ptr::null_mut();
        ck(sys::cuEventCreate(&mut ev0, 0));
        ck(sys::cuEventCreate(&mut ev1, 0));

        let grid: u32 = 170 * 8; // 8 blocks per SM on the 5090
        let block: u32 = 256;
        let mut out_host = [0u32; 65536];

        println!("p3: zero-copy host-read bandwidth (grid {grid} x block {block}), 1 GiB mapped pinned buffer");
        for rep in 0..5 {
            // warmup + timed run
            let mut n = words as u64;
            let mut p_src = src_dev as *mut std::ffi::c_void;
            let mut p_out = out_dev as *mut std::ffi::c_void;
            let args = &mut [
                &mut p_src as *mut _ as *mut _,
                &mut p_out as *mut _ as *mut _,
                &mut n as *mut _ as *mut _,
            ];
            ck(sys::cuLaunchKernel(fn_read, grid, 1, 1, block, 1, 1, 0, s, args.as_mut_ptr(), std::ptr::null_mut()));
            ck(sys::cuStreamSynchronize(s));

            // wall-clock around launch+sync: cuEventElapsedTime is not exposed under
            // the cuda-13030 feature; the kernel is long enough that host overhead is
            // negligible for a bandwidth measurement
            let t = Instant::now();
            ck(sys::cuLaunchKernel(fn_read, grid, 1, 1, block, 1, 1, 0, s, args.as_mut_ptr(), std::ptr::null_mut()));
            ck(sys::cuStreamSynchronize(s));
            let ms = t.elapsed().as_secs_f64() * 1e3;
            let gbs = bytes as f64 / 1e9 / (ms / 1e3);
            println!("  rep {rep}: {ms:.2} ms -> {gbs:.1} GB/s effective host-read");
        }

        // correctness: the out words must be nonzero (loads actually happened)
        ck(sys::cuMemcpyDtoH_v2(
            out_host.as_mut_ptr() as *mut std::ffi::c_void,
            out_dev,
            4096,
        ));
        let nonzero = out_host.iter().filter(|&&v| v != 0).count();
        println!("  verification: {nonzero}/4096 block results nonzero (loads happened)");
    }
}
