//! EVALUATION ARTEFACT, not a production path (`docs/cuda-rust-evaluation.md`, 2026-09-17).
//! Behind the default-off `cutile-pilot` feature: one kernel of `KERNEL_SRC` — `gelu_tanh`,
//! the vision-MLP activation of `vit.rs:394` — written a second time in cuTile Rust, run over
//! an engine-owned `cuda::alloc_zeroed` buffer on the engine's own stream, and compared
//! bit-for-bit with the NVRTC form. Nothing in the engine calls this module; `KERNEL_SRC` and
//! the production launch are untouched.
//!
//! Run: `CUDA_HOME=<a CUDA 13.3 toolkit whose bin/ holds a REAL tileiras — a copy, not a
//! symlink — beside ptxas, with nvvm/lib64 a sibling of bin/>
//! cargo test --release --features cutile-pilot -- --nocapture`.

use crate::cuda::{self, CUdeviceptr};
use cutile::cuda_async::device_buffer::DeviceAllocation;
use cutile::cuda_core::{sys::CUdeviceptr as CuCUdeviceptr, Device, Stream};
use cutile::prelude::*;
use std::sync::Arc;

/// The tile form of the frozen kernel: the CUDA expression of `kernels.rs:4473` with the same
/// constants and the same associativity,
/// `0.5f*v*(1.0f+tanhf(0.7978845608028654f*(v+0.044715f*v*v*v)))`.
#[cutile::module]
pub mod tile {
    use cutile::core::*;

    #[cutile::entry()]
    pub fn gelu_tanh<const S: [i32; 1]>(out: &mut Tensor<f32, S>, x: &Tensor<f32, { [-1] }>) {
        let v: Tile<f32, S> = x.load_like(out);
        let c_in = constant(0.044715f32, out.shape());
        let c_sq = constant(0.7978845608028654f32, out.shape());
        let one = constant(1.0f32, out.shape());
        let half = constant(0.5f32, out.shape());
        let t = tanh(c_sq * (v + c_in * v * v * v));
        out.store(half * v * (one + t));
    }
}

/// An engine-owned device allocation lent to cuTile: the engine allocates and frees it, this
/// wrapper only describes it and frees nothing on drop.
pub struct EngineBuffer {
    pub dptr: CUdeviceptr,
    pub len_bytes: usize,
}

// SAFETY: `dptr` is a live `len_bytes` allocation on device 0 for the whole lifetime of this
// wrapper — the engine frees it only after every tensor built from it has been dropped.
unsafe impl DeviceAllocation for EngineBuffer {
    fn device_ptr(&self) -> CuCUdeviceptr {
        self.dptr as CuCUdeviceptr
    }
    fn len_bytes(&self) -> usize {
        self.len_bytes
    }
    fn device_id(&self) -> usize {
        0
    }
}

/// `gelu_tanh` sliced out of the frozen `KERNEL_SRC`: the pilot NVRTC-compiles the production
/// text, never a copy of it.
pub fn nvrtc_gelu_tanh_src() -> String {
    let s = crate::kernels::KERNEL_SRC;
    let i = s.find("// tanh-approx GELU").expect("gelu_tanh comment gone from KERNEL_SRC");
    let j = s[i..].find("\n}\n").expect("gelu_tanh body unterminated") + i + 3;
    s[i..j].to_string()
}

/// Borrow the engine's primary context and one of its streams as cuTile handles.
///
/// # Safety
/// `ctx` must be current and `stream` must outlive every returned handle.
pub unsafe fn borrow(ctx: &cuda::Ctx, stream: u64) -> (Arc<Device>, Arc<Stream>) {
    let dev = unsafe { Device::borrow_raw(ctx.ctx as *mut std::ffi::c_void, ctx.dev, 0) };
    let st = unsafe { Stream::borrow_raw(stream as *mut std::ffi::c_void, &dev) };
    (dev, st)
}

/// Two tensors over ONE engine buffer: the read view and the written view.
///
/// # Safety
/// In place is what the CUDA kernel does. Every tile block loads exactly the tile it stores, so
/// each byte is read before it is written, by the same block, in program order — the aliasing
/// cuTile cannot see is benign here.
pub unsafe fn wrap_inplace(dptr: CUdeviceptr, n: usize) -> (Arc<Tensor<f32>>, Tensor<f32>) {
    let (bytes, shape) = (n * 4, vec![n as i32]);
    let buf = || Arc::new(EngineBuffer { dptr, len_bytes: bytes });
    unsafe {
        let rd = Tensor::<f32>::from_foreign(buf(), shape.clone(), vec![1]);
        (Arc::new(rd), Tensor::<f32>::from_foreign(buf(), shape, vec![1]))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cudarc::driver::sys;
    use cutile::cuda_async::device_operation::{DeviceOp, ExecutionContext};

    const N: usize = 1 << 20; // 1,048,576 f32 = 4 MiB, a multiple of TILE
    const TILE: usize = 256; // the CUDA launch's block width
    const REPS: usize = 1000;
    // Measured 2026-09-17, RTX 5090 sm_120, NVRTC 13.3.33 against tileiras 13.3.36: 1,060 of
    // 1,048,576 elements differ, all in the negative GELU tail where `1 + tanh(z)` cancels;
    // the largest ABSOLUTE difference over the million is 2.38e-7.
    const MAX_ULP_OF_RECORD: i64 = 33_135;

    /// Deterministic pattern: 16 edge values, then xorshift32 bits (the full f32 spread,
    /// denormals and NaNs included) alternating with a [-12, 12] sweep.
    fn pattern() -> Vec<f32> {
        let edges: [u32; 16] = [
            0x0000_0000, 0x8000_0000, 0x0080_0000, 0x0000_0001, 0x8000_0001, 0x007f_ffff,
            0x7f7f_ffff, 0xff7f_ffff, 0x7f80_0000, 0xff80_0000, 0x7fc0_0000, 0xffc0_0000,
            0x7fc0_0001, 0x7f80_0001, 0x3f80_0000, 0xbf80_0000,
        ];
        let mut v: Vec<f32> = edges.iter().map(|b| f32::from_bits(*b)).collect();
        let mut s: u32 = 0x1357_9bdf;
        while v.len() < N {
            s ^= s << 13;
            s ^= s >> 17;
            s ^= s << 5;
            let sweep = (s as f64 / u32::MAX as f64 * 24.0 - 12.0) as f32;
            v.push(if v.len().is_multiple_of(2) { f32::from_bits(s) } else { sweep });
        }
        v
    }

    /// Monotonic order key, so an ulp distance is a subtraction.
    fn ord(b: u32) -> i64 {
        (if b & 0x8000_0000 != 0 { !b } else { b | 0x8000_0000 }) as i64
    }

    /// The same expression in f64 over the same constants — the two CUDA literals rounded to f32
    /// exactly as nvcc rounds them, then widened. The arbiter where the two forms disagree.
    fn reference(x: f32) -> f32 {
        let (c_in, c_sq) = (0.044_715_f64 as f32 as f64, 0.797_884_560_802_865_4_f64 as f32 as f64);
        let v = x as f64;
        (0.5 * v * (1.0 + (c_sq * (v + c_in * v * v * v)).tanh())) as f32
    }

    unsafe fn ev() -> sys::CUevent {
        let mut e: sys::CUevent = std::ptr::null_mut();
        cuda::ck(unsafe { sys::cuEventCreate(&mut e, 0) });
        e
    }
    unsafe fn us_each(a: sys::CUevent, b: sys::CUevent) -> f64 {
        let mut ms = 0.0f32;
        cuda::ck(unsafe { sys::cuEventElapsedTime_v2(&mut ms, a, b) });
        ms as f64 * 1e3 / REPS as f64
    }

    #[test]
    fn gelu_tanh_cutile_matches_nvrtc() {
        unsafe { run() }
    }

    unsafe fn run() {
        let ctx = unsafe { cuda::Ctx::init() };
        let host = pattern();

        // The boot step the report weighs AOT against: all of KERNEL_SRC, the two halves of
        // `cuda::compile` (gen.rs:990) apart — NVRTC source to PTX, then the driver JIT of that
        // PTX. `~/.nv/ComputeCache` caches both; `CUDA_CACHE_DISABLE=1` gives the cold number.
        let t = std::time::Instant::now();
        let opts = cudarc::nvrtc::CompileOptions {
            options: vec!["--gpu-architecture=compute_120a".into()],
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(crate::kernels::KERNEL_SRC, opts)
            .expect("nvrtc compile");
        let (nvrtc_s, t) = (t.elapsed().as_secs_f64(), std::time::Instant::now());
        let c = std::ffi::CString::new(ptx.to_src()).unwrap();
        let mut whole: sys::CUmodule = std::ptr::null_mut();
        cuda::ck(unsafe { sys::cuModuleLoadData(&mut whole, c.as_ptr() as *const _) });
        let jit_s = t.elapsed().as_secs_f64();
        cuda::ck(unsafe { sys::cuModuleUnload(whole) });

        let t = std::time::Instant::now();
        let mut module = unsafe { cuda::compile(&nvrtc_gelu_tanh_src()) };
        let nvrtc_ms = t.elapsed().as_secs_f64() * 1e3;
        let f = unsafe { module.get("gelu_tanh") };

        let (mut a, mut b) = unsafe { (cuda::alloc_zeroed(N * 4), cuda::alloc_zeroed(N * 4)) };
        let mut n_p = unsafe { cuda::to_i32_dev(&[N as i32]) };
        unsafe {
            cuda::to_f32_into(a, &host);
            cuda::to_f32_into(b, &host);
        }

        // The engine's own stream: `launch_v` reads it, cuTile borrows it.
        let stream = unsafe { cuda::stream_create_non_blocking() };
        cuda::set_stream(stream as u64);
        let (_dev, st) = unsafe { borrow(&ctx, stream as u64) };

        let grid = N.div_ceil(TILE) as u32;
        unsafe {
            crate::kernels::launch_v(f, grid, 1, 1, TILE as u32, &[a, n_p]);
            cuda::stream_sync(stream);
        }

        let (x, out) = unsafe { wrap_inplace(b, N) };
        let t = std::time::Instant::now();
        let (mut part, mut x) =
            tile::gelu_tanh(out.partition([TILE]), x).sync_on(&st).expect("cutile gelu_tanh");
        let jit_ms = t.elapsed().as_secs_f64() * 1e3;

        let (ga, gb) = unsafe { (cuda::dtoh_u32(a, N), cuda::dtoh_u32(b, N)) };
        let (mut same, mut nan2, mut diff, mut max_ulp, mut worst) = (0usize, 0usize, 0usize, 0i64, 0);
        let (mut max_abs, mut ref_nv, mut ref_ct) = (0.0f32, 0i64, 0i64);
        let mut hist = [0usize; 5]; // 1, 2, 3-4, 5-16, >16 ulp
        for i in 0..N {
            if ga[i] == gb[i] {
                same += 1;
            } else if f32::from_bits(ga[i]).is_nan() && f32::from_bits(gb[i]).is_nan() {
                nan2 += 1; // both NaN, different payload
            } else {
                diff += 1;
                let u = (ord(ga[i]) - ord(gb[i])).abs();
                if u > max_ulp {
                    (max_ulp, worst) = (u, i);
                }
                hist[match u { 1 => 0, 2 => 1, 3..=4 => 2, 5..=16 => 3, _ => 4 }] += 1;
                let r = reference(host[i]).to_bits();
                ref_nv = ref_nv.max((ord(ga[i]) - ord(r)).abs());
                ref_ct = ref_ct.max((ord(gb[i]) - ord(r)).abs());
                max_abs = max_abs.max((f32::from_bits(ga[i]) - f32::from_bits(gb[i])).abs());
            }
        }

        // REPS launches back to back on the same stream, both forms, cuEvent-timed.
        let (e0, e1, e2) = unsafe { (ev(), ev(), ev()) };
        unsafe {
            cuda::ck(sys::cuEventRecord(e0, stream));
            for _ in 0..REPS {
                crate::kernels::launch_v(f, grid, 1, 1, TILE as u32, &[a, n_p]);
            }
            cuda::ck(sys::cuEventRecord(e1, stream));
        }
        let ectx = ExecutionContext::new(st.clone());
        for _ in 0..REPS {
            (part, x) = unsafe { tile::gelu_tanh(part, x).execute(&ectx) }.expect("execute");
        }
        unsafe {
            cuda::ck(sys::cuEventRecord(e2, stream));
            cuda::stream_sync(stream);
        }
        let (nv_us, ct_us) = unsafe { (us_each(e0, e1), us_each(e1, e2)) };

        println!(
            "[cutile-pilot] gelu_tanh n={N} tile={TILE}\n\
             [cutile-pilot] whole KERNEL_SRC: nvrtc {nvrtc_s:.2} s + driver JIT of the PTX {jit_s:.2} s\n\
             [cutile-pilot] nvrtc compile {nvrtc_ms:.1} ms | cutile first launch (JIT) {jit_ms:.1} ms\n\
             [cutile-pilot] per launch over {REPS}: nvrtc {nv_us:.2} us, cutile {ct_us:.2} us\n\
             [cutile-pilot] bit-equal {same}/{N}, NaN-vs-NaN {nan2}, differing {diff}, max {max_ulp} ulp, \
             max abs {max_abs:e} at i={worst} (in {:08x} nvrtc {:08x} cutile {:08x})\n\
             [cutile-pilot] ulp histogram 1/2/3-4/5-16/>16: {hist:?}\n\
             [cutile-pilot] against the f64 reference, over the differing elements: \
             nvrtc max {ref_nv} ulp, cutile max {ref_ct} ulp",
            host[worst].to_bits(), ga[worst], gb[worst],
        );

        drop((part, x));
        unsafe {
            module.unload();
            cuda::free_dev(&mut a);
            cuda::free_dev(&mut b);
            cuda::free_dev(&mut n_p);
            cuda::set_stream(0);
            cuda::stream_destroy(stream);
        }
        assert_eq!(same + nan2 + diff, N);
        // Bit-equality is the target; MAX_ULP_OF_RECORD is what this box actually gives, and the
        // printed table above is the result of record (doc section 5).
        assert!(max_ulp <= MAX_ULP_OF_RECORD, "{diff} differ, max {max_ulp} ulp");
    }
}
