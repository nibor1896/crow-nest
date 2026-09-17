//! #10d numeric probe: the `CROW_ROUTER_GEMM=1` router form against the
//! `gemv_b` form of record, on synthetic production-scale inputs (the
//! qsa_probe pattern: no model, no container, no engine lock; it compiles the
//! kernel source and runs the router forms on device buffers).
//!
//! The router switch (gen.rs moe_run, the `t >= 8` prefill branch) runs the
//! bf16 twin `router_bf` through the ONE bf16 dense launch helper instead of
//! `gemv_b` per token. The twin is EXACT: the router tensor is a BF16 keep,
//! so `load_bf16_twin`'s top-16-bit truncation carries the same values as the
//! f32 router; the only mover of the logits is the summation order (gemv_b:
//! 256-lane strided f32 sum + shared tree reduce; the dense forms: bf16
//! mma.sync m16n8k16 with f32 accumulate over 16-wide k steps, activations
//! entering as hi + residual lo so they stay effectively f32 exact).
//! Both dense forms are probed because the #10d pairs stack carries
//! `CROW_PF_GEMM_B=1`, which selects `gemm_bf16_dense_b` (32-token tiles)
//! inside the same helper; #10c proved the _b form bit-identical to the
//! 8-token form on the dense sites, and this probe measures each against
//! `gemv_b` directly.
//!
//! Statistics per form at each t (2048 = the one-chunk form of the brief,
//! 8 = the prefill boundary):
//!   - max_abs over all t x 512 logits;
//!   - raw max_rel over ALL logits (the cancellation-dominated near-zero
//!     logits included, for the record only);
//!   - masked max_rel, the acceptance number: over logits with
//!     |ref| >= 1e-3 * (max |ref| of that token's row), i.e. values bounded
//!     away from cancellation; the 1e-2-class hard line of the 10a plan
//!     applies HERE;
//!   - top-10 set agreement per token (the router_top10 consumer's
//!     consequence), with the first differing token index.
//!
//! usage: router_probe
use crow_nest_engine::cuda;
use crow_nest_engine::sample::Rng;
use crow_nest_engine::kernels::launch_v;

/// uniform in [-a, a] over the shared deterministic xorshift64* stream
fn uni(rng: &mut Rng, a: f32) -> f32 {
    (rng.f01() * 2.0 - 1.0) * a
}

/// top-10 indices by (value desc, index asc); ties are measure zero on
/// continuous random logits and the router_top10 kernel breaks ties by index,
/// the same rule
fn top10(v: &[f32]) -> Vec<usize> {
    let mut idx: Vec<usize> = (0..v.len()).collect();
    idx.sort_by(|&a, &b| v[b].partial_cmp(&v[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(10);
    idx
}

fn probe_one(label: &str, t: usize, rows: usize, r: &[f32], c: &[f32]) -> bool {
    let mut max_abs: f32 = 0.0;
    let mut raw_rel: f32 = 0.0;
    let mut mask_rel: f32 = 0.0;
    let mut scale_max: f32 = 0.0;
    let mut diff_tokens: usize = 0;
    let mut first_tok: Option<usize> = None;
    let mut nan = 0usize;
    for tok in 0..t {
        let row = &r[tok * rows..(tok + 1) * rows];
        let rowc = &c[tok * rows..(tok + 1) * rows];
        let rmax = row.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        scale_max = scale_max.max(rmax);
        let mask = 1.0e-3 * rmax;
        for i in 0..rows {
            let (a, b) = (row[i], rowc[i]);
            if a.is_nan() || b.is_nan() {
                nan += 1;
                continue;
            }
            let d = (a - b).abs();
            max_abs = max_abs.max(d);
            raw_rel = raw_rel.max(d / a.abs().max(f32::MIN_POSITIVE));
            if a.abs() >= mask {
                mask_rel = mask_rel.max(d / a.abs());
            }
        }
        if top10(row) != top10(rowc) {
            diff_tokens += 1;
            if first_tok.is_none() {
                first_tok = Some(tok);
            }
        }
    }
    let pass = mask_rel <= 1.0e-2 && nan == 0;
    println!(
        "[router-probe] t={t} {label}: logit_scale {scale_max:.4}  max_abs {max_abs:.3e}  raw_max_rel {raw_rel:.3e}  masked_max_rel(>=1e-3*rowmax) {mask_rel:.3e}  top10_diff {diff_tokens}/{t} first_diff_tok {}  nan {nan}  -> {}",
        first_tok.map(|i| i.to_string()).unwrap_or_else(|| "none".into()),
        if pass { "PASS (1e-2 line)" } else { "FAIL" }
    );
    pass
}

/// f64 host dot of one (row, token) pair from the ORIGINAL f32 values; the
/// arbitration reference that tells which GPU side deviates and by how much
fn f64_dot(w: &[f32], x: &[f32], k: usize, row: usize, tok: usize) -> f64 {
    let wr = &w[row * k..(row + 1) * k];
    let xr = &x[tok * k..(tok + 1) * k];
    let mut acc = 0.0f64;
    for i in 0..k {
        acc += wr[i] as f64 * xr[i] as f64;
    }
    acc
}

fn main() {
    let mut ok = true;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f_ref = module.get("gemv_b");
        let f_g8 = module.get("gemm_bf16_dense");
        let f_g32 = module.get("gemm_bf16_dense_b");

        const ROWS: usize = 512; // E, the router row count (nr512)
        const K: usize = 2560; // H, the router input width (n2560)

        // the production launch shapes of gen.rs moe_run (gemv_b grid
        // (E, t) block 256; the dense helper grid (rows/64, ceil(t/8 or 32))
        // block 128), and the production int args via device buffers
        // (scalar args travel as device pointers on this raw-launch path)
        for &t in [2048usize, 8usize].iter() {
            let mut rng = Rng::from_state(0x10d_5eED_2026_0914);
            // activations: post-norm f32 hidden states are O(1); uniform [-2, 2]
            let x: Vec<f32> = (0..t * K).map(|_| uni(&mut rng, 2.0)).collect();
            // weights: uniform [-0.1, 0.1] then quantized to EXACT bf16 values
            // (truncate, re-widen): the engine invariant this probe depends on
            // is that the f32 router (load_f32 of the BF16 keep) and the bf16
            // twin (load_bf16_twin) carry THE SAME values, so both forms dot
            // identical weights and the only remaining delta is summation
            // order plus the dense form's hi+lo activation representation
            let w: Vec<f32> = (0..ROWS * K)
                .map(|_| {
                    let v: f32 = uni(&mut rng, 0.1);
                    f32::from_bits(v.to_bits() & 0xFFFF_0000)
                })
                .collect();
            let wb: Vec<u16> = w.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let d_w = cuda::upload_dev(std::slice::from_raw_parts(
                w.as_ptr() as *const u8,
                w.len() * 4,
            ));
            let d_wb = cuda::upload_dev(std::slice::from_raw_parts(
                wb.as_ptr() as *const u8,
                wb.len() * 2,
            ));
            let d_x = cuda::to_f32_dev(&x);
            let d_kdim = cuda::to_i32_dev(&[K as i32]);
            let d_rows = cuda::to_i32_dev(&[ROWS as i32]);
            let d_t = cuda::to_i32_dev(&[t as i32]);
            let n = t * ROWS;
            let d_ref = cuda::alloc_zeroed(n * 4);
            let d_g8 = cuda::alloc_zeroed(n * 4);
            let d_g32 = cuda::alloc_zeroed(n * 4);

            launch_v(f_ref, ROWS as u32, t as u32, 1, 256, &[
                d_w as u64, d_x as u64, d_ref as u64, d_kdim as u64]);
            launch_v(f_g8, ((ROWS + 63) / 64) as u32, ((t + 7) / 8) as u32, 1, 128, &[
                d_wb as u64, d_x as u64, d_g8 as u64, d_kdim as u64, d_rows as u64, d_t as u64]);
            launch_v(f_g32, ((ROWS + 63) / 64) as u32, ((t + 31) / 32) as u32, 1, 128, &[
                d_wb as u64, d_x as u64, d_g32 as u64, d_kdim as u64, d_rows as u64, d_t as u64]);
            cuda::sync();

            let r = cuda::dtoh(d_ref, n);
            let g8 = cuda::dtoh(d_g8, n);
            let g32 = cuda::dtoh(d_g32, n);

            // spot arbitration vs the f64 host dot: gemv_b must sit at f64
            // order-noise level; wherever the dense form lands tells whether
            // the delta is representation noise or something structural
            let mut argmax = (0usize, 0usize);
            let mut best = 0.0f64;
            for tok in 0..t {
                for row in 0..ROWS {
                    let d = (r[tok * ROWS + row] as f64 - g8[tok * ROWS + row] as f64).abs();
                    if d > best {
                        best = d;
                        argmax = (tok, row);
                    }
                }
            }
            for &(tok, row) in [(0usize, 0usize), (1, 17), (6, 511), (t.min(100) - 1, 256), argmax].iter() {
                let (tok, row) = (tok.min(t - 1), row % ROWS);
                let f = f64_dot(&w, &x, K, row, tok);
                println!(
                    "[router-probe] spot t={t} tok={tok} row={row}: f64 {f:.6}  gemv_b {:.6} (d {:+.2e})  gemm8 {:.6} (d {:+.2e})",
                    r[tok * ROWS + row],
                    r[tok * ROWS + row] as f64 - f,
                    g8[tok * ROWS + row],
                    g8[tok * ROWS + row] as f64 - f
                );
            }

            ok &= probe_one("gemm_bf16_dense (8-token tiles)", t, ROWS, &r, &g8);
            ok &= probe_one("gemm_bf16_dense_b (32-token tiles)", t, ROWS, &r, &g32);

            cuda::free_dev(&mut { d_w });
            cuda::free_dev(&mut { d_wb });
            cuda::free_dev(&mut { d_x });
            cuda::free_dev(&mut { d_kdim });
            cuda::free_dev(&mut { d_rows });
            cuda::free_dev(&mut { d_t });
            cuda::free_dev(&mut { d_ref });
            cuda::free_dev(&mut { d_g8 });
            cuda::free_dev(&mut { d_g32 });
        }
    }
    println!(
        "[router-probe] seed 0x10d_5eED_2026_0914, x uniform [-2,2], w uniform [-0.1,0.1], rows 512, k 2560, t 2048 + 8"
    );
    println!(
        "[router-probe] VERDICT: {}",
        if ok { "PASS, every masked_max_rel within the 1e-2 line" } else { "FAIL, the 1e-2 line is broken" }
    );
    if !ok {
        std::process::exit(1);
    }
}
