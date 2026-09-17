//! MMA FP4 path gate (ticket: gemv_fp4_ptrb -> tensor-core mma.sync m16n8k64).
//!
//! mma_gate [gate|bench|all]
//!
//! Stages:
//!   1. dyadic single-tile transport check (delta must be EXACTLY 0.0)
//!   2. random numerics vs the p10-verified naive `gemv_fp4_ptrb` AND two CPU
//!      references (exact-f32 chain, quantized-activation chain) on BOTH MoE
//!      shapes (gate_up [1280,2560], down [2560,640]), 12 combos, one combo
//!      served from PINNED host memory (zero-copy cold path)
//!   3. microbench on the real per-layer routed sequence, t=1 decode and
//!      t=2048 prefill: naive (ptrb+silu+ptrb) vs mma (quant+mma+silu+quant+mma)
//!
//! Layouts: p2 (LSB-first nibbles, sf byte i -> k-subblock i, D fragment) +
//! mma_probe differential lane map (sf_a: lane L -> row (L>>2)+8*(L&3), t<2;
//! sf_b: lane L -> col L>>2, t==0).

use crow_nest_engine::cnq;
use crow_nest_engine::cuda::{self, CUdeviceptr, Pinned};
use crow_nest_engine::geo::*;
use crow_nest_engine::sample::Rng;
use crow_nest_engine::kernels::{launch_sync, launch_v, Kernels};

// ---------------- Rust twins of the device math ----------------

fn q_e2m1(v: f32) -> u32 {
    let s = if v < 0.0 { 8u32 } else { 0 };
    let r = v.abs();
    if r <= 0.25 { s }
    else if r < 0.75 { s | 1 }
    else if r <= 1.25 { s | 2 }
    else if r < 1.75 { s | 3 }
    else if r <= 2.5 { s | 4 }
    else if r < 3.5 { s | 5 }
    else if r <= 5.0 { s | 6 }
    else { s | 7 }
}

fn enc_ue4m3_up(s: f32) -> u8 {
    if !(s > 0.0) { return 0; }
    for e in 0..16i32 {
        let mul = if e == 0 { 512.0f32 } else { (2.0f32).powi(10 - e) };
        let m = (s * mul).ceil() - if e == 0 { 0.0 } else { 8.0 };
        if (0.0..=7.0).contains(&m) {
            return ((e << 3) | m as i32) as u8;
        }
    }
    0x7E
}

/// dequantize a [rows][k_dim] NVFP4 slab (36-byte 64-blocks) — container math
fn dequant_slab(w: &[u8], rows: usize, k_dim: usize, gs: f32) -> Vec<f32> {
    let bpr = k_dim / 64;
    let mut out = vec![0f32; rows * k_dim];
    for r in 0..rows {
        for b in 0..bpr {
            let blk = &w[(r * bpr + b) * 36..(r * bpr + b) * 36 + 36];
            for idx in 0..64usize {
                let byte = blk[4 + (idx >> 1)] as u32;
                let nib = if idx & 1 == 1 { byte >> 4 } else { byte & 0xF };
                out[r * k_dim + b * 64 + idx] =
                    cnq::e2m1(nib) * cnq::ue4m3(blk[idx >> 4] as u32) * gs;
            }
        }
    }
    out
}

/// Rust twin of quant_x_fp4: THREE levels per 16-block (level 0: smallest
/// ue4m3>=amax/6 scale + RNE ties-to-even e2m1; levels 1-2: the residual
/// quantized the same way). Returns [level0 | level1 | level2] bytes.
fn quant_row(x: &[f32], k_dim: usize) -> Vec<u8> {
    let bpr = k_dim / 64;
    let mut out = vec![0u8; bpr * 108];
    for b in 0..bpr {
        for s in 0..4usize {
            let p = &x[b * 64 + s * 16..b * 64 + s * 16 + 16];
            let mut src = p.to_vec();
            for lv in 0..3usize {
                let base = lv * bpr * 36 + b * 36;
                let amax = src.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let sc = enc_ue4m3_up(amax / 6.0f32);
                let inv = 1.0f32 / cnq::ue4m3(sc as u32);
                let scf = cnq::ue4m3(sc as u32);
                out[base + s] = sc;
                let mut r = [0f32; 16];
                for j in 0..8usize {
                    let n0 = q_e2m1(src[2 * j] * inv);
                    let n1 = q_e2m1(src[2 * j + 1] * inv);
                    out[base + 4 + s * 8 + j] = (n0 | (n1 << 4)) as u8;
                    r[2 * j] = src[2 * j] - cnq::e2m1(n0) * scf;
                    r[2 * j + 1] = src[2 * j + 1] - cnq::e2m1(n1) * scf;
                }
                src = r.to_vec();
            }
        }
    }
    out
}

fn dequant_quant_row_2l(q: &[u8], k_dim: usize) -> Vec<f32> {
    let bpr = k_dim / 64;
    let mut out = dequant_quant_row(&q[..bpr * 36], k_dim);
    for lv in 1..3usize {
        let l = dequant_quant_row(&q[lv * bpr * 36..(lv + 1) * bpr * 36], k_dim);
        for (o, v) in out.iter_mut().zip(l.iter()) {
            *o += v;
        }
    }
    out
}

fn dequant_quant_row(q: &[u8], k_dim: usize) -> Vec<f32> {
    let bpr = k_dim / 64;
    let mut out = vec![0f32; k_dim];
    for b in 0..bpr {
        let blk = &q[b * 36..b * 36 + 36];
        for idx in 0..64usize {
            let byte = blk[4 + (idx >> 1)] as u32;
            let nib = if idx & 1 == 1 { byte >> 4 } else { byte & 0xF };
            out[b * 64 + idx] = cnq::e2m1(nib) * cnq::ue4m3(blk[idx >> 4] as u32);
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum()
}

// ---------------- random data ----------------

/// symmetric uniform in [-1, 1) over the shared xorshift64* stream
fn sym(rng: &mut Rng) -> f32 {
    rng.f01() * 2.0 - 1.0
}

/// Box-Muller normal over the same stream
fn gauss(rng: &mut Rng) -> f32 {
    let u1 = ((rng.next_u64() >> 11) as f64 + 1.0) / (1u64 << 53) as f64;
    let u2 = (rng.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
    ((-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()) as f32
}

fn rand_slab(rng: &mut Rng, rows: usize, k_dim: usize) -> Vec<u8> {
    let bpr = k_dim / 64;
    let mut v = vec![0u8; rows * bpr * 36];
    for b in v.chunks_exact_mut(36) {
        for s in 0..4usize {
            b[s] = 0x28 + (rng.next_u64() % 0x21) as u8; // scales ~0.13..4.8
        }
        for j in 0..32usize {
            b[4 + j] = (rng.next_u64() % 256) as u8;
        }
    }
    v
}

fn rand_rows(rng: &mut Rng, rows: usize, k_dim: usize) -> Vec<f32> {
    (0..rows * k_dim).map(|_| gauss(rng) * 1.5).collect()
}

// ---------------- stats ----------------

fn stats(name: &str, got: &[f32], want: &[f32], gate: Option<f64>) -> bool {
    let mut max_abs = 0f32;
    let (mut se, mut sg, mut sen, mut sgn) = (0f64, 0f64, 0f64, 0f64);
    let mut nan = 0usize;
    for i in 0..got.len() {
        if got[i].is_nan() { nan += 1; }
        let d = (got[i] - want[i]).abs();
        max_abs = max_abs.max(d);
        let e = got[i] as f64 - want[i] as f64;
        se += e * e;
        sg += want[i] as f64 * want[i] as f64;
        if want[i].abs() > 1e-5 {
            sen += e * e;
            sgn += want[i] as f64 * want[i] as f64;
        }
    }
    let rel = se.sqrt() / sg.sqrt().max(1e-30);
    let rel_nz = sen.sqrt() / sgn.sqrt().max(1e-30);
    println!(
        "  {name}: max_abs={max_abs:.3e} rel_L2={rel:.3e} rel_L2(nz)={rel_nz:.3e} NaN={nan} max|ref|={:.3e}",
        want.iter().fold(0f32, |a, &v| a.max(v.abs()))
    );
    match gate {
        Some(g) => {
            let ok = rel <= g && nan == 0;
            println!("    gate rel_L2 <= {g:.0e} && NaN=0: {}", if ok { "PASS" } else { "FAIL" });
            ok
        }
        None => nan == 0,
    }
}

unsafe fn dtoh_bytes(src: CUdeviceptr, n: usize) -> Vec<u8> {
    let mut v = vec![0u8; n];
    cuda::ck(cudarc::driver::sys::cuMemcpyDtoH_v2(
        v.as_mut_ptr() as *mut std::ffi::c_void, src, n));
    v
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("gate");
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let k = Kernels::new(&module);

        let n2560 = cuda::to_i32_dev(&[H as i32]);
        let n640 = cuda::to_i32_dev(&[INTER as i32]);
        let n64 = cuda::to_i32_dev(&[64i32]);
        let one = cuda::to_i32_dev(&[1]);
        let k_top10 = cuda::to_i32_dev(&[TOPK as i32]);

        if mode == "gate" || mode == "all" {
            let ok = gate_stage(&k, n2560, n640, n64, one, k_top10);
            if !ok {
                std::process::exit(1);
            }
        }
        if mode == "bench" || mode == "all" {
            bench_stage(&k, n2560, n640, one, k_top10);
        }
        if mode == "dense" || mode == "alld" {
            let ok = dense_gate_stage(&k);
            if !ok {
                std::process::exit(1);
            }
        }
        if mode == "densebench" || mode == "alld" {
            dense_bench_stage(&k);
        }
    }
}

unsafe fn gate_stage(k: &Kernels, n2560: CUdeviceptr, n640: CUdeviceptr, n64: CUdeviceptr, one: CUdeviceptr, k_top10: CUdeviceptr) -> bool {
    let mut all_ok = true;

    println!("=== MMA gate stage 1: dyadic single-tile transport (expect EXACT 0) ===");
    {
        let mut rng = Rng::from_state(0xDA1A);
        let (rows, k_dim) = (64usize, 64usize); // ONE m16n8k64 tile per warp
        let mut w = vec![0u8; rows * 36];
        for b in w.chunks_exact_mut(36) {
            for (s, sb) in b.iter_mut().take(4).enumerate() {
                *sb = [0x30, 0x38, 0x40, 0x38][s]; // 0.5 / 1 / 2 / 1
            }
            for j in 0..32usize {
                let lo = (((rng.next_u64() & 1) << 3) | 2) as u8; // +/- 1.0
                let hi = (((rng.next_u64() & 1) << 3) | 1) as u8; // +/- 0.5
                b[4 + j] = lo | (hi << 4);
            }
        }
        let x: Vec<f32> = (0..k_dim)
            .map(|i| [1.0f32, 0.5, 2.0, 1.5, -1.0, -0.5, -2.0, -1.5][i % 8])
            .collect();
        let xq = quant_row(&x, k_dim);
        let gs = 0.5f32;

        let w_dev = cuda::upload_dev(&w);
        let xq_dev = cuda::upload_dev(&xq);
        let gs_dev = cuda::to_f32_dev(&[gs]);
        let y_dev = cuda::alloc_zeroed(rows * 4);
        let ptr_dev = cuda::to_u64_dev(&[w_dev as u64]);
        launch_sync(k.f("gemv_fp4_mma"), (rows / 64) as u32, 1, 1, 128, &[
            ptr_dev as u64, xq_dev as u64, gs_dev as u64, y_dev as u64, n64 as u64, one as u64]);
        let y = cuda::dtoh(y_dev, rows);

        let wq = dequant_slab(&w, rows, k_dim, gs);
        let xqf = dequant_quant_row_2l(&xq, k_dim);
        let refy: Vec<f32> = (0..rows)
            .map(|r| dot(&wq[r * k_dim..(r + 1) * k_dim], &xqf))
            .collect();
        let max_d = y.iter().zip(refy.iter()).map(|(&a, &b)| (a - b).abs()).fold(0f32, f32::max);
        println!("  dyadic max delta = {max_d:.6e} (must be exactly 0.0)");
        if max_d != 0.0 {
            println!("  stage 1 FAIL — MMA layout wrong");
            all_ok = false;
        }
        for mut d in [w_dev, xq_dev, gs_dev, y_dev, ptr_dev] {
            cuda::free_dev(&mut d);
        }
    }

    println!("=== MMA gate stage 2: random numerics vs naive ptrb + CPU references ===");
    for &(name, rows, k_dim, x_div) in &[("gate_up", 2 * INTER, H, TOPK), ("down", H, INTER, 1usize)] {
        println!("--- shape {name} [{rows},{k_dim}] ---");
        let mut rng = Rng::from_state(if rows == 2 * INTER { 0x6A7E1 } else { 0x6A7E2 });
        let bpr = k_dim / 64;
        let n_combos = 12usize; // >= 8 required
        let n_tokens = (n_combos + x_div - 1) / x_div;

        // one expert per combo; combo 11 served from PINNED host RAM (zero-copy)
        let mut slabs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..n_combos {
            slabs.push(rand_slab(&mut rng, rows, k_dim));
        }
        let mut pinned = Pinned::alloc(slabs[11].len());
        pinned.write_bytes(0, &slabs[11]);
        let slab_devs: Vec<CUdeviceptr> = slabs[..11].iter().map(|s| cuda::upload_dev(s)).collect();
        let mut table: Vec<u64> = slab_devs.iter().map(|&d| d as u64).collect();
        table.push(pinned.dev as u64);

        let gs = 0.25f32 + 0.5 * sym(&mut rng).abs();
        let x_rows = if x_div > 1 { n_tokens } else { n_combos };
        let x: Vec<f32> = rand_rows(&mut rng, x_rows, k_dim);

        let ptr_dev = cuda::to_u64_dev(&table);
        let x_dev = cuda::to_f32_dev(&x);
        let gs_dev = cuda::to_f32_dev(&[gs]);
        let y_naive = cuda::alloc_zeroed(n_combos * rows * 4);
        let y_mma = cuda::alloc_zeroed(n_combos * rows * 4);
        let xq_dev = cuda::alloc_zeroed(x_rows * bpr * 108);

        // naive path (p10-verified reference)
        launch_sync(k.f("gemv_fp4_ptrb"), rows as u32, n_combos as u32, 1, 256, &[
            ptr_dev as u64, x_dev as u64, gs_dev as u64, y_naive as u64,
            if x_div > 1 { n2560 as u64 } else { n640 as u64 },
            if x_div > 1 { k_top10 as u64 } else { one as u64 },
            if x_div > 1 { n2560 as u64 } else { n640 as u64 }]);
        // mma path
        launch_sync(k.f("quant_x_fp4"), x_rows as u32, 1, 1, 128, &[
            x_dev as u64, xq_dev as u64,
            if x_div > 1 { n2560 as u64 } else { n640 as u64 }, one as u64,
            if x_div > 1 { n2560 as u64 } else { n640 as u64 }]);
        launch_sync(k.f("gemv_fp4_mma"), (rows / 64) as u32, n_combos as u32, 1, 128, &[
            ptr_dev as u64, xq_dev as u64, gs_dev as u64, y_mma as u64,
            if x_div > 1 { n2560 as u64 } else { n640 as u64 },
            if x_div > 1 { k_top10 as u64 } else { one as u64 }]);

        let y_n = cuda::dtoh(y_naive, n_combos * rows);
        let y_m = cuda::dtoh(y_mma, n_combos * rows);
        let xq_host = dtoh_bytes(xq_dev, x_rows * bpr * 108);

        // device quant vs Rust twin must agree BYTE-EXACT (all levels)
        let mut q_mismatch = 0usize;
        for r in 0..x_rows {
            let twin = quant_row(&x[r * k_dim..(r + 1) * k_dim], k_dim);
            let dev = &xq_host[r * bpr * 108..(r + 1) * bpr * 108];
            if twin[..] != dev[..] {
                q_mismatch += 1;
            }
        }
        println!("  quant kernel vs Rust twin: {q_mismatch}/{x_rows} rows differ (expect 0)");

        // CPU references
        let mut ref_f32 = vec![0f32; n_combos * rows];
        let mut ref_q = vec![0f32; n_combos * rows];
        for c in 0..n_combos {
            let xr = if x_div > 1 { c / x_div } else { c };
            let xrow = &x[xr * k_dim..(xr + 1) * k_dim];
            let qf = dequant_quant_row_2l(&quant_row(xrow, k_dim), k_dim);
            let wc = dequant_slab(&slabs[c], rows, k_dim, gs);
            for r in 0..rows {
                ref_f32[c * rows + r] = dot(&wc[r * k_dim..(r + 1) * k_dim], xrow);
                ref_q[c * rows + r] = dot(&wc[r * k_dim..(r + 1) * k_dim], &qf);
            }
        }

        println!("  (combo 11 weights served zero-copy from pinned host RAM)");
        all_ok &= stats("naive vs CPU exact-f32 chain", &y_n, &ref_f32, Some(1e-4));
        all_ok &= stats("mma   vs CPU quantized chain", &y_m, &ref_q, Some(1e-4));
        all_ok &= stats("mma   vs naive (adds act-quant error)", &y_m, &y_n, None);
        all_ok &= stats("mma   vs CPU exact-f32 chain (total)", &y_m, &ref_f32, None);

        for mut d in slab_devs {
            cuda::free_dev(&mut d);
        }
        for mut d in [ptr_dev, x_dev, gs_dev, y_naive, y_mma, xq_dev] {
            cuda::free_dev(&mut d);
        }
        pinned.free();
    }
    println!("=== MMA numerics gate: {} ===", if all_ok { "PASS" } else { "FAIL" });
    all_ok
}

unsafe fn bench_stage(k: &Kernels, n2560: CUdeviceptr, n640: CUdeviceptr, one: CUdeviceptr, k_top10: CUdeviceptr) {
    for &(label, t) in &[("t=1 decode", 1usize), ("t=2048 prefill chunk", 2048usize)] {
        println!("=== MMA bench: routed MoE GEMVs per layer, {label} ===");
        let mut rng = Rng::from_state(0xBEEF + t as u64);
        let combos = t * TOPK;
        let tokens = t;
        let bpr_gu = H / 64; // 40 (k=2560)
        let bpr_dn = INTER / 64; // 10 (k=640)

        // 10 hot experts (VRAM slabs) — the compute-relevant case
        let mut gu_slabs: Vec<Vec<u8>> = Vec::new();
        let mut dn_slabs: Vec<Vec<u8>> = Vec::new();
        for _ in 0..TOPK {
            gu_slabs.push(rand_slab(&mut rng, 2 * INTER, H));
            dn_slabs.push(rand_slab(&mut rng, H, INTER));
        }
        let gu_devs: Vec<CUdeviceptr> = gu_slabs.iter().map(|s| cuda::upload_dev(s)).collect();
        let dn_devs: Vec<CUdeviceptr> = dn_slabs.iter().map(|s| cuda::upload_dev(s)).collect();
        let gu_table: Vec<u64> = (0..combos).map(|c| gu_devs[c % TOPK] as u64).collect();
        let dn_table: Vec<u64> = (0..combos).map(|c| dn_devs[c % TOPK] as u64).collect();
        let gu_ptrs = cuda::to_u64_dev(&gu_table);
        let dn_ptrs = cuda::to_u64_dev(&dn_table);
        let gs_dev = cuda::to_f32_dev(&[0.5f32, 0.5]);

        let x = rand_rows(&mut rng, tokens, H);
        let x_dev = cuda::to_f32_dev(&x);
        let h1 = cuda::alloc_zeroed(combos * 2 * INTER * 4);
        let h2 = cuda::alloc_zeroed(combos * INTER * 4);
        let eo = cuda::alloc_zeroed(combos * H * 4);
        let xq_gu = cuda::alloc_zeroed(tokens * bpr_gu * 108);
        let xq_dn = cuda::alloc_zeroed(combos * bpr_dn * 108);
        let nt_combo = cuda::to_i32_dev(&[(combos * INTER) as i32]);

        let seq_naive = || {
            launch_v(k.f("gemv_fp4_ptrb"), (2 * INTER) as u32, combos as u32, 1, 256, &[
                gu_ptrs as u64, x_dev as u64, gs_dev as u64, h1 as u64,
                n2560 as u64, k_top10 as u64, n2560 as u64]);
            launch_v(k.f("silu_mul_combo"), ((combos * INTER + 255) / 256) as u32, 1, 1, 256, &[
                h1 as u64, h2 as u64, nt_combo as u64]);
            launch_v(k.f("gemv_fp4_ptrb"), H as u32, combos as u32, 1, 256, &[
                dn_ptrs as u64, h2 as u64, (gs_dev + 4) as u64, eo as u64,
                n640 as u64, one as u64, n640 as u64]);
        };
        let seq_mma = || {
            launch_v(k.f("quant_x_fp4"), tokens as u32, 1, 1, 128, &[
                x_dev as u64, xq_gu as u64, n2560 as u64, one as u64, n2560 as u64]);
            launch_v(k.f("gemv_fp4_mma"), ((2 * INTER) / 64) as u32, combos as u32, 1, 128, &[
                gu_ptrs as u64, xq_gu as u64, gs_dev as u64, h1 as u64, n2560 as u64, k_top10 as u64]);
            launch_v(k.f("silu_mul_combo"), ((combos * INTER + 255) / 256) as u32, 1, 1, 256, &[
                h1 as u64, h2 as u64, nt_combo as u64]);
            launch_v(k.f("quant_x_fp4"), combos as u32, 1, 1, 128, &[
                h2 as u64, xq_dn as u64, n640 as u64, one as u64, n640 as u64]);
            launch_v(k.f("gemv_fp4_mma"), (H / 64) as u32, combos as u32, 1, 128, &[
                dn_ptrs as u64, xq_dn as u64, (gs_dev + 4) as u64, eo as u64,
                n640 as u64, one as u64]);
        };

        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..3 { f(); }
            cuda::sync();
            let reps = if t == 1 { 100 } else { 15 };
            let t0 = std::time::Instant::now();
            for _ in 0..reps { f(); }
            cuda::sync();
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        };

        let ms_naive = time(&seq_naive);
        let ms_mma = time(&seq_mma);
        println!(
            "  naive: {ms_naive:.3} ms/layer   mma: {ms_mma:.3} ms/layer   speedup x{:.1}   projected/token (x48 layers): naive {:.1} ms -> mma {:.1} ms",
            ms_naive / ms_mma,
            ms_naive * 48.0,
            ms_mma * 48.0
        );

        let all: Vec<CUdeviceptr> = gu_devs.iter().copied().chain(dn_devs.iter().copied()).collect();
        for mut d in all {
            cuda::free_dev(&mut d);
        }
        for mut d in [gu_ptrs, dn_ptrs, gs_dev, x_dev, h1, h2, eo, xq_gu, xq_dn, nt_combo] {
            cuda::free_dev(&mut d);
        }
    }
}

// ---------------- dense GEMV groups (#10 extension) ----------------
// gate: gemv_fp4_mma_d numerics vs the p10-verified naive kernels AND CPU
// quantized-chain references — covers the row-guard shapes (48/4 rows), the
// batched-t path and the sh12 [t][1280] stride layout.
unsafe fn dense_gate_stage(k: &Kernels) -> bool {
    let mut all_ok = true;
    println!("=== MMA-d gate: dense shapes vs naive gemv_fp4_b + CPU quant chain ===");
    let shapes: &[(&str, usize, usize, usize)] = &[
        ("gdn b/a [48,2560] row-guard, t=3", GDN_VHEADS, H, 3),
        ("hc inject [4,10240] row-guard, t=2", HCN, HCT, 2),
        ("attn v [512,2560], t=2", KV_ROWS, H, 2),
        ("indexer qk [640,2560], t=2", QSA_QK_ROWS, H, 2),
        ("gdn out [2560,6144], t=2", H, GDN_VAL, 2),
        ("shared down [2560,640], t=2", H, INTER, 2),
    ];
    for &(name, rows, k_dim, tokens) in shapes {
        println!("--- shape {name} ---");
        let mut rng = Rng::from_state(0xD00D + rows as u64 * 131 + k_dim as u64);
        let bpr = k_dim / 64;
        let w = rand_slab(&mut rng, rows, k_dim);
        let x = rand_rows(&mut rng, tokens, k_dim);
        let gs = 0.25f32 + 0.5 * sym(&mut rng).abs();

        let w_dev = cuda::upload_dev(&w);
        let x_dev = cuda::to_f32_dev(&x);
        let gs_dev = cuda::to_f32_dev(&[gs]);
        let k_dim_dev = cuda::to_i32_dev(&[k_dim as i32]);
        let rows_dev = cuda::to_i32_dev(&[rows as i32]);
        let one = cuda::to_i32_dev(&[1]);
        let y_naive = cuda::alloc_zeroed(tokens * rows * 4);
        let y_mma = cuda::alloc_zeroed(tokens * rows * 4);
        let xq_dev = cuda::alloc_zeroed(tokens * bpr * 108);

        launch_sync(k.f("gemv_fp4_b"), rows as u32, tokens as u32, 1, 256, &[
            w_dev as u64, x_dev as u64, gs_dev as u64, y_naive as u64, k_dim_dev as u64]);
        launch_sync(k.f("quant_x_fp4"), tokens as u32, 1, 1, 128, &[
            x_dev as u64, xq_dev as u64, k_dim_dev as u64, one as u64, k_dim_dev as u64]);
        launch_sync(k.f("gemv_fp4_mma_d"), ((rows + 63) / 64) as u32, tokens as u32, 1, 128, &[
            w_dev as u64, xq_dev as u64, gs_dev as u64, y_mma as u64,
            k_dim_dev as u64, rows_dev as u64, rows_dev as u64]);

        let y_n = cuda::dtoh(y_naive, tokens * rows);
        let y_m = cuda::dtoh(y_mma, tokens * rows);

        // CPU references: exact-f32 chain + quantized-activation chain per token
        let mut ref_f32 = vec![0f32; tokens * rows];
        let mut ref_q = vec![0f32; tokens * rows];
        let wq = dequant_slab(&w, rows, k_dim, gs);
        for tt in 0..tokens {
            let xrow = &x[tt * k_dim..(tt + 1) * k_dim];
            let qf = dequant_quant_row_2l(&quant_row(xrow, k_dim), k_dim);
            for r in 0..rows {
                ref_f32[tt * rows + r] = dot(&wq[r * k_dim..(r + 1) * k_dim], xrow);
                ref_q[tt * rows + r] = dot(&wq[r * k_dim..(r + 1) * k_dim], &qf);
            }
        }
        all_ok &= stats("mma_d vs CPU quantized chain", &y_m, &ref_q, Some(1e-4));
        all_ok &= stats("naive  vs CPU exact-f32 chain", &y_n, &ref_f32, Some(1e-4));
        all_ok &= stats("mma_d  vs naive (adds act-quant error)", &y_m, &y_n, None);

        for mut d in [w_dev, x_dev, gs_dev, k_dim_dev, rows_dev, one, y_naive, y_mma, xq_dev] {
            cuda::free_dev(&mut d);
        }
    }

    println!("--- sh12 stride layout: sg/su [640,2560] into ONE [t][1280] buffer, t=2 ---");
    {
        let mut rng = Rng::from_state(0x5E12);
        let (rows, k_dim, tokens) = (INTER, H, 2);
        let bpr = k_dim / 64;
        let wsg = rand_slab(&mut rng, rows, k_dim);
        let wsu = rand_slab(&mut rng, rows, k_dim);
        let x = rand_rows(&mut rng, tokens, k_dim);
        let gs = 0.25f32 + 0.5 * sym(&mut rng).abs();

        let wsg_dev = cuda::upload_dev(&wsg);
        let wsu_dev = cuda::upload_dev(&wsu);
        let x_dev = cuda::to_f32_dev(&x);
        let gs_dev = cuda::to_f32_dev(&[gs]);
        let n2560 = cuda::to_i32_dev(&[H as i32]);
        let n640 = cuda::to_i32_dev(&[INTER as i32]);
        let n1280 = cuda::to_i32_dev(&[(2 * INTER) as i32]);
        let one = cuda::to_i32_dev(&[1]);
        let sh12_naive = cuda::alloc_zeroed(tokens * 2 * INTER * 4);
        let sh12_mma = cuda::alloc_zeroed(tokens * 2 * INTER * 4);
        let xq_dev = cuda::alloc_zeroed(tokens * bpr * 108);

        launch_sync(k.f("gemv_fp4_bs"), rows as u32, tokens as u32, 1, 256, &[
            wsg_dev as u64, x_dev as u64, gs_dev as u64, sh12_naive as u64,
            n2560 as u64, n1280 as u64]);
        launch_sync(k.f("gemv_fp4_bs"), rows as u32, tokens as u32, 1, 256, &[
            wsu_dev as u64, x_dev as u64, gs_dev as u64, (sh12_naive + (INTER * 4) as u64),
            n2560 as u64, n1280 as u64]);
        launch_sync(k.f("quant_x_fp4"), tokens as u32, 1, 1, 128, &[
            x_dev as u64, xq_dev as u64, n2560 as u64, one as u64, n2560 as u64]);
        launch_sync(k.f("gemv_fp4_mma_d"), (rows / 64) as u32, tokens as u32, 1, 128, &[
            wsg_dev as u64, xq_dev as u64, gs_dev as u64, sh12_mma as u64,
            n2560 as u64, n640 as u64, n1280 as u64]);
        launch_sync(k.f("gemv_fp4_mma_d"), (rows / 64) as u32, tokens as u32, 1, 128, &[
            wsu_dev as u64, xq_dev as u64, gs_dev as u64, (sh12_mma + (INTER * 4) as u64),
            n2560 as u64, n640 as u64, n1280 as u64]);

        let y_n = cuda::dtoh(sh12_naive, tokens * 2 * INTER);
        let y_m = cuda::dtoh(sh12_mma, tokens * 2 * INTER);
        let wqg = dequant_slab(&wsg, rows, k_dim, gs);
        let wqu = dequant_slab(&wsu, rows, k_dim, gs);
        let mut ref_q = vec![0f32; tokens * 2 * INTER];
        for tt in 0..tokens {
            let qf = dequant_quant_row_2l(&quant_row(&x[tt * k_dim..(tt + 1) * k_dim], k_dim), k_dim);
            for r in 0..rows {
                ref_q[tt * 1280 + r] = dot(&wqg[r * k_dim..(r + 1) * k_dim], &qf);
                ref_q[tt * 1280 + 640 + r] = dot(&wqu[r * k_dim..(r + 1) * k_dim], &qf);
            }
        }
        all_ok &= stats("mma_d stride vs CPU quant chain", &y_m, &ref_q, Some(1e-4));
        all_ok &= stats("mma_d stride vs naive bs", &y_m, &y_n, None);

        for mut d in [wsg_dev, wsu_dev, x_dev, gs_dev, n2560, n640, n1280, one, sh12_naive, sh12_mma, xq_dev] {
            cuda::free_dev(&mut d);
        }
    }
    println!("=== MMA-d numerics gate: {} ===", if all_ok { "PASS" } else { "FAIL" });
    all_ok
}

// bench: per-callsite-group dense GEMV sequence, naive vs quant+mma_d,
// decode (t=1) and prefill-chunk (t=512) — the #10 per-group ms/layer data.
// Filters: densebench [t] [group] ; GATE_SYNC=1 syncs after every launch
// (crash localization).
unsafe fn dense_bench_stage(k: &Kernels) {
    let t_f: Option<usize> = std::env::args().nth(2).and_then(|v| v.parse().ok());
    let g_f = std::env::args().nth(3);
    let sync_each = std::env::var("GATE_SYNC").as_deref() == Ok("1");
    let lv = |name: &str, gx: u32, gy: u32, vals: &[u64]| {
        if sync_each {
            eprintln!("[launch] {name} gx={gx} gy={gy}");
            launch_sync(k.f(name), gx, gy, 1, 128, vals);
        } else {
            launch_v(k.f(name), gx, gy, 1, 128, vals);
        }
    };
    let lvn = |name: &str, gx: u32, gy: u32, vals: &[u64]| {
        if sync_each {
            eprintln!("[launch] {name} gx={gx} gy={gy}");
            launch_sync(k.f(name), gx, gy, 1, 256, vals);
        } else {
            launch_v(k.f(name), gx, gy, 1, 256, vals);
        }
    };
    let want = |name: &str| g_f.as_deref().map(|g| g == name).unwrap_or(true);
    let one = cuda::to_i32_dev(&[1]);
    let n2560 = cuda::to_i32_dev(&[H as i32]);
    let n640 = cuda::to_i32_dev(&[INTER as i32]);
    let n1280 = cuda::to_i32_dev(&[(2 * INTER) as i32]);
    let n6144 = cuda::to_i32_dev(&[GDN_VAL as i32]);
    let n10240 = cuda::to_i32_dev(&[HCT as i32]);
    let n48 = cuda::to_i32_dev(&[GDN_VHEADS as i32]);
    let n512 = cuda::to_i32_dev(&[KV_ROWS as i32]);
    // persistent-param integrity watch (corruption hunter)
    let empty: Vec<(&str, CUdeviceptr, i32)> = vec![];
    let watch: Vec<(&str, CUdeviceptr, i32)> = if std::env::var("GATE_WATCH").as_deref() == Ok("1") {
        vec![("one", one, 1), ("n2560", n2560, H as i32), ("n640", n640, INTER as i32),
             ("n1280", n1280, (2 * INTER) as i32), ("n6144", n6144, GDN_VAL as i32),
             ("n10240", n10240, HCT as i32), ("n48", n48, GDN_VHEADS as i32),
             ("n512", n512, KV_ROWS as i32)]
    } else {
        empty
    };
    let check = |tag: &str, w: &Vec<(&str, CUdeviceptr, i32)>| {
        for (name, dev, want_v) in w {
            let v = cuda::dtoh_i32(*dev, 1)[0];
            if v != *want_v {
                eprintln!("[corrupt {tag}] {name}: {v} != {want_v}");
            }
        }
    };
    for &(label, t) in &[("t=1 decode", 1usize), ("t=512 prefill chunk", 512usize)] {
        if t_f.map(|tf| tf != t).unwrap_or(false) {
            continue;
        }
        println!("=== MMA-d bench: dense GEMV groups per layer, {label} ===");
        check("pre", &watch);
        let mut rng = Rng::from_state(0xD00B + t as u64);
        let gs_dev = cuda::to_f32_dev(&[0.5f32]);
        let time = |f: &dyn Fn()| -> f64 {
            for _ in 0..3 { f(); }
            cuda::sync();
            let reps = if t == 1 { 100 } else { 8 };
            let t0 = std::time::Instant::now();
            for _ in 0..reps { f(); }
            cuda::sync();
            t0.elapsed().as_secs_f64() * 1e3 / reps as f64
        };
        // per group: (name, layers/token, naive closure, mma closure) — built
        // inline per group so buffers scope correctly.
        macro_rules! group {
            ($name:expr, $per_layer:expr, $alloc:expr, $naive:expr, $mma:expr) => {{
                $alloc;
                let ms_n = time(&$naive);
                let ms_m = time(&$mma);
                println!(
                    "  {:<10} naive {:8.3} ms   mma {:8.3} ms   x{:.1}   projected/token (x{}): {:.1} -> {:.1} ms",
                    $name, ms_n, ms_m, ms_n / ms_m, $per_layer,
                    ms_n * $per_layer as f64, ms_m * $per_layer as f64
                );
            }};
        }

        // HC inject [4][10240] — runs 2x per layer (hc + hc2)
        if want("hc") {
        let rows = HCN;
        let k_dim = HCT;
        let w = rand_slab(&mut rng, rows, k_dim);
        let x = rand_rows(&mut rng, t, k_dim);
        let w_dev = cuda::upload_dev(&w);
        let x_dev = cuda::to_f32_dev(&x);
        let y = cuda::alloc_zeroed(t * rows * 4);
        let xq = cuda::alloc_zeroed(t * (k_dim / 64) * 108);
        let rows_dev = cuda::to_i32_dev(&[rows as i32]);
        group!("hc-inject", 96,
            (),
            || {
                lvn("gemv_fp4_b", rows as u32, t as u32, &[
                    w_dev as u64, x_dev as u64, gs_dev as u64, y as u64, n10240 as u64]);
            },
            || {
                lv("quant_x_fp4", t as u32, 1, &[
                    x_dev as u64, xq as u64, n10240 as u64, one as u64, n10240 as u64]);
                lv("gemv_fp4_mma_d", 1, t as u32, &[
                    w_dev as u64, xq as u64, gs_dev as u64, y as u64,
                    n10240 as u64, rows_dev as u64, rows_dev as u64]);
            });
        for mut d in [w_dev, x_dev, y, xq, rows_dev] { cuda::free_dev(&mut d); }
        }
        check("hc", &watch);

        // GDN block: qkv[10240,2560] z[6144,2560] b[48,2560] a[48,2560] out[2560,6144]
        if want("gdn") {
        let wqkv = cuda::upload_dev(&rand_slab(&mut rng, GDN_CONV, H));
        let wz = cuda::upload_dev(&rand_slab(&mut rng, GDN_VAL, H));
        let wb = cuda::upload_dev(&rand_slab(&mut rng, GDN_VHEADS, H));
        let wa = cuda::upload_dev(&rand_slab(&mut rng, GDN_VHEADS, H));
        let wout = cuda::upload_dev(&rand_slab(&mut rng, H, GDN_VAL));
        let xm = cuda::to_f32_dev(&rand_rows(&mut rng, t, H));
        let xg = cuda::to_f32_dev(&rand_rows(&mut rng, t, GDN_VAL));
        let xq_m = cuda::alloc_zeroed(t * (H / 64) * 108);
        let xq_v = cuda::alloc_zeroed(t * (GDN_VAL / 64) * 108);
        let o_mq = cuda::alloc_zeroed(t * GDN_CONV * 4);
        let o_gz = cuda::alloc_zeroed(t * GDN_VAL * 4);
        let o_gb = cuda::alloc_zeroed(t * GDN_VHEADS * 4);
        let o_ga = cuda::alloc_zeroed(t * GDN_VHEADS * 4);
        let o_gout = cuda::alloc_zeroed(t * H * 4);
        group!("gdn", 36,
            (),
            || {
                lvn("gemv_fp4_b", GDN_CONV as u32, t as u32, &[
                    wqkv as u64, xm as u64, gs_dev as u64, o_mq as u64, n2560 as u64]);
                lvn("gemv_fp4_b", GDN_VAL as u32, t as u32, &[
                    wz as u64, xm as u64, gs_dev as u64, o_gz as u64, n2560 as u64]);
                lvn("gemv_fp4_b", GDN_VHEADS as u32, t as u32, &[
                    wb as u64, xm as u64, gs_dev as u64, o_gb as u64, n2560 as u64]);
                lvn("gemv_fp4_b", GDN_VHEADS as u32, t as u32, &[
                    wa as u64, xm as u64, gs_dev as u64, o_ga as u64, n2560 as u64]);
                lvn("gemv_fp4_b", H as u32, t as u32, &[
                    wout as u64, xg as u64, gs_dev as u64, o_gout as u64, n6144 as u64]);
            },
            || {
                lv("quant_x_fp4", t as u32, 1, &[
                    xm as u64, xq_m as u64, n2560 as u64, one as u64, n2560 as u64]);
                lv("gemv_fp4_mma_d", (GDN_CONV / 64) as u32, t as u32, &[
                    wqkv as u64, xq_m as u64, gs_dev as u64, o_mq as u64,
                    n2560 as u64, n10240 as u64, n10240 as u64]);
                lv("gemv_fp4_mma_d", (GDN_VAL / 64) as u32, t as u32, &[
                    wz as u64, xq_m as u64, gs_dev as u64, o_gz as u64,
                    n2560 as u64, n6144 as u64, n6144 as u64]);
                lv("gemv_fp4_mma_d", 1, t as u32, &[
                    wb as u64, xq_m as u64, gs_dev as u64, o_gb as u64,
                    n2560 as u64, n48 as u64, n48 as u64]);
                lv("gemv_fp4_mma_d", 1, t as u32, &[
                    wa as u64, xq_m as u64, gs_dev as u64, o_ga as u64,
                    n2560 as u64, n48 as u64, n48 as u64]);
                lv("quant_x_fp4", t as u32, 1, &[
                    xg as u64, xq_v as u64, n6144 as u64, one as u64, n6144 as u64]);
                lv("gemv_fp4_mma_d", (H / 64) as u32, t as u32, &[
                    wout as u64, xq_v as u64, gs_dev as u64, o_gout as u64,
                    n6144 as u64, n2560 as u64, n2560 as u64]);
            });
        for mut d in [wqkv, wz, wb, wa, wout, xm, xg, xq_m, xq_v, o_mq, o_gz, o_gb, o_ga, o_gout] {
            cuda::free_dev(&mut d);
        }
        }
        check("gdn", &watch);

        // Attention: v[512,2560] iqk[640,2560] o[2560,6144]
        if want("attn") {
        let wv = cuda::upload_dev(&rand_slab(&mut rng, KV_ROWS, H));
        let wiqk = cuda::upload_dev(&rand_slab(&mut rng, QSA_QK_ROWS, H));
        let wo = cuda::upload_dev(&rand_slab(&mut rng, H, GDN_VAL));
        let xm = cuda::to_f32_dev(&rand_rows(&mut rng, t, H));
        let xg = cuda::to_f32_dev(&rand_rows(&mut rng, t, GDN_VAL));
        let xq_m = cuda::alloc_zeroed(t * (H / 64) * 108);
        let xq_v = cuda::alloc_zeroed(t * (GDN_VAL / 64) * 108);
        let o_v = cuda::alloc_zeroed(t * KV_ROWS * 4);
        let o_qk = cuda::alloc_zeroed(t * QSA_QK_ROWS * 4);
        let o_ay = cuda::alloc_zeroed(t * H * 4);
        group!("attn", 12,
            (),
            || {
                lvn("gemv_fp4_b", KV_ROWS as u32, t as u32, &[
                    wv as u64, xm as u64, gs_dev as u64, o_v as u64, n2560 as u64]);
                lvn("gemv_fp4_b", QSA_QK_ROWS as u32, t as u32, &[
                    wiqk as u64, xm as u64, gs_dev as u64, o_qk as u64, n2560 as u64]);
                lvn("gemv_fp4_b", H as u32, t as u32, &[
                    wo as u64, xg as u64, gs_dev as u64, o_ay as u64, n6144 as u64]);
            },
            || {
                lv("quant_x_fp4", t as u32, 1, &[
                    xm as u64, xq_m as u64, n2560 as u64, one as u64, n2560 as u64]);
                lv("gemv_fp4_mma_d", (KV_ROWS / 64) as u32, t as u32, &[
                    wv as u64, xq_m as u64, gs_dev as u64, o_v as u64,
                    n2560 as u64, n512 as u64, n512 as u64]);
                lv("gemv_fp4_mma_d", (QSA_QK_ROWS / 64) as u32, t as u32, &[
                    wiqk as u64, xq_m as u64, gs_dev as u64, o_qk as u64,
                    n2560 as u64, n640 as u64, n640 as u64]);
                lv("quant_x_fp4", t as u32, 1, &[
                    xg as u64, xq_v as u64, n6144 as u64, one as u64, n6144 as u64]);
                lv("gemv_fp4_mma_d", (H / 64) as u32, t as u32, &[
                    wo as u64, xq_v as u64, gs_dev as u64, o_ay as u64,
                    n6144 as u64, n2560 as u64, n2560 as u64]);
            });
        for mut d in [wv, wiqk, wo, xm, xg, xq_m, xq_v, o_v, o_qk, o_ay] {
            cuda::free_dev(&mut d);
        }
        }
        check("attn", &watch);

        // Shared expert: sg[640,2560] su[640,2560] (stride 1280) + sdn[2560,640]
        if want("shared") {
        let wsg = cuda::upload_dev(&rand_slab(&mut rng, INTER, H));
        let wsu = cuda::upload_dev(&rand_slab(&mut rng, INTER, H));
        let wsdn = cuda::upload_dev(&rand_slab(&mut rng, H, INTER));
        let xm = cuda::to_f32_dev(&rand_rows(&mut rng, t, H));
        let xs = cuda::to_f32_dev(&rand_rows(&mut rng, t, INTER));
        let xq_gu = cuda::alloc_zeroed(t * (H / 64) * 108);
        let xq_s = cuda::alloc_zeroed(t * (INTER / 64) * 108);
        let sh12 = cuda::alloc_zeroed(t * 2 * INTER * 4);
        let sdown = cuda::alloc_zeroed(t * H * 4);
        group!("shared", 48,
            (),
            || {
                lvn("gemv_fp4_bs", INTER as u32, t as u32, &[
                    wsg as u64, xm as u64, gs_dev as u64, sh12 as u64, n2560 as u64, n1280 as u64]);
                lvn("gemv_fp4_bs", INTER as u32, t as u32, &[
                    wsu as u64, xm as u64, gs_dev as u64, (sh12 + (INTER * 4) as u64), n2560 as u64, n1280 as u64]);
                lvn("gemv_fp4_b", H as u32, t as u32, &[
                    wsdn as u64, xs as u64, gs_dev as u64, sdown as u64, n640 as u64]);
            },
            || {
                lv("gemv_fp4_mma_d", (INTER / 64) as u32, t as u32, &[
                    wsg as u64, xq_gu as u64, gs_dev as u64, sh12 as u64,
                    n2560 as u64, n640 as u64, n1280 as u64]);
                lv("gemv_fp4_mma_d", (INTER / 64) as u32, t as u32, &[
                    wsu as u64, xq_gu as u64, gs_dev as u64, (sh12 + (INTER * 4) as u64),
                    n2560 as u64, n640 as u64, n1280 as u64]);
                lv("quant_x_fp4", t as u32, 1, &[
                    xs as u64, xq_s as u64, n640 as u64, one as u64, n640 as u64]);
                lv("gemv_fp4_mma_d", (H / 64) as u32, t as u32, &[
                    wsdn as u64, xq_s as u64, gs_dev as u64, sdown as u64,
                    n640 as u64, n2560 as u64, n2560 as u64]);
            });
        for mut d in [wsg, wsu, wsdn, xm, xs, xq_gu, xq_s, sh12, sdown] {
            cuda::free_dev(&mut d);
        }
        }
        check("shared", &watch);

        // PLE (layer 1 only): key[10240,2560] value[2560,2560]
        if want("ple") {
        let wk = cuda::upload_dev(&rand_slab(&mut rng, HCT, H));
        let wv2 = cuda::upload_dev(&rand_slab(&mut rng, H, H));
        let xe = cuda::to_f32_dev(&rand_rows(&mut rng, t, H));
        let xq_e = cuda::alloc_zeroed(t * (H / 64) * 108);
        let ok = cuda::alloc_zeroed(t * HCT * 4);
        let ov = cuda::alloc_zeroed(t * H * 4);
        group!("ple", 1,
            (),
            || {
                lvn("gemv_fp4_b", HCT as u32, t as u32, &[
                    wk as u64, xe as u64, gs_dev as u64, ok as u64, n2560 as u64]);
                lvn("gemv_fp4_b", H as u32, t as u32, &[
                    wv2 as u64, xe as u64, gs_dev as u64, ov as u64, n2560 as u64]);
            },
            || {
                lv("quant_x_fp4", t as u32, 1, &[
                    xe as u64, xq_e as u64, n2560 as u64, one as u64, n2560 as u64]);
                lv("gemv_fp4_mma_d", (HCT / 64) as u32, t as u32, &[
                    wk as u64, xq_e as u64, gs_dev as u64, ok as u64,
                    n2560 as u64, n10240 as u64, n10240 as u64]);
                lv("gemv_fp4_mma_d", (H / 64) as u32, t as u32, &[
                    wv2 as u64, xq_e as u64, gs_dev as u64, ov as u64,
                    n2560 as u64, n2560 as u64, n2560 as u64]);
            });
        for mut d in [wk, wv2, xe, xq_e, ok, ov] { cuda::free_dev(&mut d); }
        }
        check("ple", &watch);

        for mut d in [one, n2560, n640, n1280, n6144, n10240, n48, n512, gs_dev] {
            cuda::free_dev(&mut d);
        }
    }
}
