//! fast validation: NVRTC compile of the consolidated kernel source + FP8
//! E4M3 device/Rust-twin bit-exactness sweep (no model load).
use crow_nest_engine::cnq::e4m3_to_f32;
use crow_nest_engine::cnq::f32_to_e4m3;
use crow_nest_engine::cuda;
use crow_nest_engine::kernels::Kernels;

fn main() {
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let k = Kernels::new(&module);

        // ---- FP8 sweep: device enc/dec vs Rust twin, bit-exact expected ----
        let mut vals: Vec<f32> = Vec::new();
        for i in -6200..6200 {
            vals.push(i as f32 * 0.1);
        }
        for e in -20..9 {
            for m in 0..64 {
                vals.push(m as f32 / 8.0 * 2f32.powi(e));
                vals.push(-(m as f32) / 8.0 * 2f32.powi(e));
            }
        }
        for i in 0..1000u32 {
            let b = i.wrapping_mul(2654435761);
            vals.push(f32::from_bits(b));
        }
        vals.extend([0.0, -0.0, 448.0, 464.0, 1e30, -1e30, 1e-30, 2f32.powi(-9), 2f32.powi(-10), 2f32.powi(-6)]);
        let n = vals.len();
        let x = cuda::to_f32_dev(&vals);
        let bytes = cuda::alloc_zeroed(n);
        let back = cuda::alloc_zeroed(n * 4);
        let n_dev = cuda::to_i32_dev(&[n as i32]);
        crow_nest_engine::kernels::launch_sync(k.f("cast_e4m3_flat"), ((n as u32) + 255) / 256, 1, 1, 256, &[
            x as u64, bytes as u64, n_dev as u64]);
        let host_bytes = {
            let mut v = vec![0u8; n];
            cuda::ck(cudarc::driver::sys::cuMemcpyDtoH_v2(
                v.as_mut_ptr() as *mut std::ffi::c_void, bytes, n));
            v
        };
        crow_nest_engine::kernels::launch_sync(k.f("dec_e4m3_flat"), ((n as u32) + 255) / 256, 1, 1, 256, &[
            bytes as u64, back as u64, n_dev as u64]);
        let host_back = cuda::dtoh(back, n);

        let mut enc_mismatch = 0usize;
        let mut worst_rt = 0.0f64;
        let mut worst_rt_val = 0f32;
        for i in 0..n {
            let rust_enc = f32_to_e4m3(vals[i]);
            if rust_enc != host_bytes[i] {
                if enc_mismatch < 8 {
                    println!("  enc mismatch @{}: v={:e} rust={:#04x} gpu={:#04x}", i, vals[i], rust_enc, host_bytes[i]);
                }
                enc_mismatch += 1;
            }
            let rt_gpu = e4m3_to_f32(host_bytes[i]);
            let d = if rt_gpu.is_nan() && host_back[i].is_nan() {
                0.0
            } else {
                (rt_gpu - host_back[i]).abs() as f64
            };
            if d > worst_rt {
                worst_rt = d;
                worst_rt_val = vals[i];
            }
            // roundtrip error bound of the FORMAT (device decode == rust decode)
            let v = vals[i];
            if v.is_finite() && v.abs() <= 448.0 && host_bytes[i] & 0x7f != 0x7f {
                let dec = rt_gpu;
                let err = ((dec - v) as f64).abs() / (v.abs() as f64).max(1e-30);
                if err > 0.07 { // E4M3 relative step max 1/16 + subnormal edges
                    // subnormals have large rel error — only flag normals
                    if v.abs() >= 0.015625 {
                        println!("  large rt err: v={v:e} dec={dec:e} rel={err}");
                    }
                }
            }
            let _ = worst_rt_val;
        }
        println!(
            "kcheck: FP8 sweep n={n}: enc mismatches device-vs-rust = {enc_mismatch} (expect 0), worst decode delta = {worst_rt:e}"
        );
        if enc_mismatch == 0 {
            println!("kcheck: OK");
        } else {
            println!("kcheck: FAIL");
            std::process::exit(1);
        }
    }
}
