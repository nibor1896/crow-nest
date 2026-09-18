//! #61a/#61b unit gate: qsa_select_par (two launches) against qsa_select_fast
//! (one block) on synthetic score rows. Compares sel_n and sel_list[0..sel_n]
//! byte for byte. No model, no container, no engine lock: it compiles the
//! kernel source and runs the two selections on device buffers.
//!
//! #61b (61a review finding I3) extends the case set:
//!   - sel_max is the PRODUCTION constant QSA_SEL_MAX (geo.rs, 2051) on every
//!     row, no longer the generous max(4*K+3, pos+1);
//!   - positions follow the production form pos = 4*ncb + tail - 1 with the
//!     real tail range 0..=3 (tail = (pos+1) mod 4), so the no-tail rows and
//!     the empty edge cases are covered and the old four-tail-token row
//!     (never a production row) is gone;
//!   - several cap values (one smaller, the production 65536, one larger),
//!     each case also run with two query rows so the cap row stride
//!     (row = scores + qi * cap) is really exercised;
//!   - the maximum supported block count ncb = 65536 (the score cap at the
//!     262k context ceiling) is a shape;
//!   - canaries: after every row the slots [sel_n..sel_max) of both lists
//!     must still hold the poison, and h1 must be zero again.
//!
//! usage: qsa_probe [--rows N] [--bench]
use crow_nest_engine::cuda;
use crow_nest_engine::gen::{launch_qsa_par_e, QSA_PAR_BINS};
use crow_nest_engine::kernels::launch_v;
use crow_nest_engine::geo::QSA_SEL_MAX;
/// deterministic xorshift64*: the same rows on every machine and every run
use crow_nest_engine::sample::Rng;

/// score distributions: the narrow ones are the hard cases for a radix top-k
/// (every key shares the top digits), the tie ones exercise the lowest-index
/// fill. Tiny holds 1e-30-scale magnitudes: those are ordinary normal-range
/// floats (FP32 min normal is about 1.18e-38), so Tiny is NOT a denormal or
/// subnormal test and no denormal coverage is claimed (61a review, doc
/// corrections).
#[derive(Clone, Copy, Debug)]
enum Dist {
    Uniform,
    Narrow,
    Ties(u32),
    AllEqual,
    Signed,
    Tiny,
}

fn fill_row(d: Dist, n: usize, rng: &mut Rng, out: &mut [f32]) {
    for v in out.iter_mut().take(n) {
        *v = match d {
            Dist::Uniform => rng.f01() * 20.0 - 10.0,
            Dist::Narrow => 1.0 + rng.f01() * 1.0e-3,
            Dist::Ties(m) => ((rng.next_u64() % m as u64) as f32) * 0.25,
            Dist::AllEqual => 1.25,
            Dist::Signed => {
                let a = rng.f01() * 4.0;
                if rng.next_u64() & 1 == 0 { -a } else { a }
            }
            Dist::Tiny => (rng.f01() - 0.5) * 1.0e-30,
        };
    }
}

fn main() {
    // #13: the logging subscriber of this process. Every library line this bin
    // triggers (`[prefill]`, `[load]`, `[budget]`, `[ple]`, ...) is a `tracing`
    // event now, so without this call they go nowhere. The guard drains the two
    // writer threads when `main` returns; an `exit` below calls `shutdown` first.
    let _log = crow_nest_engine::log::init();
    let args: Vec<String> = std::env::args().collect();
    let bench = args.iter().any(|a| a == "--bench");
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f_fast = module.get("qsa_select_fast");
        let f_par_h = module.get("qsa_select_par_h");
        let f_par_e = module.get("qsa_select_par_e");

        // production sel_max semantics: the list stride is the CONSTANT
        // QSA_SEL_MAX (geo.rs, 4*512 + 3 = 2051 for the 512-block budget); it
        // never grows with K or pos, so every case must keep its sel_n within
        // it. Caps: 4096 (smaller), 65536 (the production score cap at the
        // 262k context ceiling), 131072 (larger).
        const CAPS: &[usize] = &[4096, 65536, 131072];
        const CAP_MAX: usize = 131072;
        const NQ_MAX: usize = 2;
        const MAX_NCB: usize = 65536; // the maximum supported block count
        let d_scores = cuda::alloc_zeroed(NQ_MAX * CAP_MAX * 4);
        let d_sel_a = cuda::alloc_zeroed(NQ_MAX * QSA_SEL_MAX * 4);
        let d_sel_b = cuda::alloc_zeroed(NQ_MAX * QSA_SEL_MAX * 4);
        let d_n_a = cuda::alloc_zeroed(NQ_MAX * 4);
        let d_n_b = cuda::alloc_zeroed(NQ_MAX * 4);
        let d_h1 = cuda::alloc_zeroed(NQ_MAX * QSA_PAR_BINS * 4);
        let d_ncb = cuda::to_i32_dev(&[0]);
        let d_k = cuda::to_i32_dev(&[0]);
        let d_cap = cuda::to_i32_dev(&[0]);
        let d_selmax = cuda::to_i32_dev(&[QSA_SEL_MAX as i32]);
        let d_pos = cuda::to_i32_dev(&[0]);

        // (ncb, K) shapes; every shape runs with tails 0..=3 so pos sweeps the
        // production form pos = 4*ncb + tail - 1. Sparse regime (K < ncb):
        // sel_n = 4*selected + tail <= 4*K + 3. Dense regime (K >= ncb): the
        // list is 0..=pos, so pos + 1 <= QSA_SEL_MAX caps dense ncb at 512.
        let shapes: &[(usize, usize)] = &[
            (65536, 512), // MAXIMUM supported block count (score cap, 262k ceiling), sparse
            (4080, 512),  // the 16,320 row of the operating point, sparse
            (4096, 512),  // power of two, sparse
            (3000, 300),  // sparse, K off the 512 budget
            (2048, 512),  // sparse
            (1024, 512),  // sparse
            (513, 512),   // one block above the budget, sparse
            (512, 512),   // dense boundary K == ncb: sel_n = 2048 + tail fits 2051 exactly
            (256, 512),   // dense
            (100, 512),   // dense
            (100, 5000),  // dense, K far above ncb
            (63, 7),      // sparse, small K
            (7, 1),       // sparse, K = 1
            (1, 512),     // dense, single block
            (0, 512),     // dense, empty: tail 0 gives pos = -1 and sel_n = 0
        ];
        for &(ncb, k) in shapes {
            assert!(
                ncb <= MAX_NCB,
                "shape ncb {ncb} above the maximum supported block count {MAX_NCB}");
            if k >= ncb {
                assert!(
                    4 * ncb + 3 <= QSA_SEL_MAX,
                    "dense shape ncb {ncb} overflows the production sel_max {QSA_SEL_MAX}");
            } else {
                assert!(
                    4 * k + 3 <= QSA_SEL_MAX,
                    "sparse shape K {k} overflows the production sel_max {QSA_SEL_MAX}");
            }
        }

        let dists: &[Dist] = &[
            Dist::Uniform,
            Dist::Narrow,
            Dist::Ties(9),
            Dist::Ties(64),
            Dist::AllEqual,
            Dist::Signed,
            Dist::Tiny,
        ];
        let want_rows: usize = args
            .iter()
            .position(|a| a == "--rows")
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(0);

        let mut host = vec![0f32; NQ_MAX * CAP_MAX];
        let mut rng = Rng::from_state(0x1234_5678_9abc_def1);
        let mut rows = 0usize;
        let mut diffs = 0usize;
        let mut first_diff: Option<String> = None;
        let per_seed: usize = CAPS
            .iter()
            .map(|&cap| shapes.iter().filter(|&&(ncb, _)| ncb <= cap).count())
            .sum::<usize>()
            * NQ_MAX
            * 4
            * dists.len();
        let mut seeds = 2usize;
        if want_rows > 0 {
            seeds = ((want_rows + per_seed - 1) / per_seed).max(1);
        }
        for seed in 0..seeds {
            for &cap in CAPS {
                for &(ncb, k) in shapes {
                    if ncb > cap {
                        continue; // the score row must fit the cap
                    }
                    for nq in 1..=NQ_MAX {
                        for tail in 0..=3usize {
                            // production position form: ncb = (pos+1)/4
                            // complete blocks, tail = (pos+1) mod 4; tail 0 is
                            // the no-tail row, and ncb 0 with tail 0 gives
                            // pos = -1, the fully empty case sel_n = 0
                            let pos = 4 * ncb + tail - 1;
                            for &dist in dists {
                                for r in 0..nq {
                                    fill_row(dist, ncb.max(1), &mut rng,
                                        &mut host[r * cap..r * cap + ncb.max(1)]);
                                    cuda::to_f32_into(
                                        (d_scores as u64 + (r * cap * 4) as u64)
                                            as cuda::CUdeviceptr,
                                        &host[r * cap..r * cap + ncb.max(1)]);
                                }
                                cuda::to_i32_into(d_ncb, &[ncb as i32]);
                                cuda::to_i32_into(d_k, &[k as i32]);
                                cuda::to_i32_into(d_cap, &[cap as i32]);
                                cuda::to_i32_into(d_pos, &[pos as i32]);
                                // poison both lists (every query row) so a
                                // short write and an over-write past sel_n
                                // both show up
                                let poison = vec![-7i32; nq * QSA_SEL_MAX];
                                cuda::to_i32_into(d_sel_a, &poison);
                                cuda::to_i32_into(d_sel_b, &poison);
                                cuda::sync();

                                launch_v(f_fast, nq as u32, 1, 1, 256, &[
                                    d_scores as u64, d_ncb as u64, d_sel_a as u64, d_n_a as u64,
                                    d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64]);
                                // G = 32, the production CROW_QSA_PAR_BLOCKS default
                                launch_v(f_par_h, 32, nq as u32, 1, 256, &[
                                    d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64,
                                    d_cap as u64]);
                                launch_qsa_par_e(f_par_e, nq as u32, &[
                                    d_scores as u64, d_ncb as u64, d_sel_b as u64, d_n_b as u64,
                                    d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64,
                                    d_h1 as u64]);
                                cuda::sync();

                                let na = cuda::dtoh_i32(d_n_a, nq);
                                let nb = cuda::dtoh_i32(d_n_b, nq);
                                let la = cuda::dtoh_i32(d_sel_a, nq * QSA_SEL_MAX);
                                let lb = cuda::dtoh_i32(d_sel_b, nq * QSA_SEL_MAX);
                                let h = cuda::dtoh_u32(d_h1, nq * QSA_PAR_BINS);
                                rows += 1;
                                let mut bad: Option<String> = None;
                                for qi in 0..nq {
                                    let (a, b) = (na[qi], nb[qi]);
                                    let laq = &la[qi * QSA_SEL_MAX..(qi + 1) * QSA_SEL_MAX];
                                    let lbq = &lb[qi * QSA_SEL_MAX..(qi + 1) * QSA_SEL_MAX];
                                    if a != b {
                                        bad = Some(format!("qi {qi}: sel_n {a} vs {b}"));
                                    } else {
                                        let n = a.max(0) as usize;
                                        for i in 0..n {
                                            if laq[i] != lbq[i] {
                                                bad = Some(format!(
                                                    "qi {qi}: sel_list[{i}] {} vs {}",
                                                    laq[i], lbq[i]));
                                                break;
                                            }
                                        }
                                        // canary: nothing may be written past sel_n
                                        if bad.is_none() {
                                            for i in n..QSA_SEL_MAX {
                                                if laq[i] != -7 || lbq[i] != -7 {
                                                    bad = Some(format!(
                                                        "qi {qi}: canary sel_list[{i}] after sel_n {n}: {} vs {}",
                                                        laq[i], lbq[i]));
                                                    break;
                                                }
                                            }
                                        }
                                    }
                                    if bad.is_some() {
                                        break;
                                    }
                                }
                                // the histogram must be zero again for the next call
                                if bad.is_none() && h.iter().any(|&v| v != 0) {
                                    bad = Some("h1 not zero after the emit kernel".to_string());
                                }
                                if let Some(m) = bad {
                                    diffs += 1;
                                    if first_diff.is_none() {
                                        first_diff = Some(format!(
                                            "cap {cap} nq {nq} ncb {ncb} tail {tail} K {k} dist {dist:?} seed {seed}: {m}"));
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        println!(
            "[qsa-probe] sel_max {QSA_SEL_MAX} (production QSA_SEL_MAX), caps {CAPS:?}, max ncb {MAX_NCB}, tails 0..=3, nq 1..={NQ_MAX}, G 32");
        println!("[qsa-probe] rows {rows}  differences {diffs}");
        if let Some(m) = first_diff {
            println!("[qsa-probe] first difference: {m}");
        }

        if bench {
            let ncb = 4080usize;
            let pos = 4 * ncb - 1; // the 16,320 operating point row: tail 0
            let k = 512usize;
            let sel_max = QSA_SEL_MAX;
            fill_row(Dist::Narrow, ncb, &mut rng, &mut host);
            cuda::to_f32_into(d_scores, &host[..ncb]);
            cuda::to_i32_into(d_ncb, &[ncb as i32]);
            cuda::to_i32_into(d_k, &[k as i32]);
            cuda::to_i32_into(d_selmax, &[sel_max as i32]);
            cuda::to_i32_into(d_pos, &[pos as i32]);
            cuda::sync();
            let reps = 2000u32;
            for g in [4u32, 8, 16, 32, 64, 128, 256] {
                for _ in 0..50 {
                    launch_v(f_par_h, g, 1, 1, 256, &[
                        d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64, d_cap as u64]);
                    launch_qsa_par_e(f_par_e, 1, &[
                        d_scores as u64, d_ncb as u64, d_sel_b as u64, d_n_b as u64,
                        d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64, d_h1 as u64]);
                }
                cuda::sync();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    launch_v(f_par_h, g, 1, 1, 256, &[
                        d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64, d_cap as u64]);
                    launch_qsa_par_e(f_par_e, 1, &[
                        d_scores as u64, d_ncb as u64, d_sel_b as u64, d_n_b as u64,
                        d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64, d_h1 as u64]);
                }
                cuda::sync();
                let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
                println!("[qsa-probe] par  G={g:3}  {us:7.2} us per call (2 launches)");
            }
            for _ in 0..50 {
                launch_v(f_fast, 1, 1, 1, 256, &[
                    d_scores as u64, d_ncb as u64, d_sel_a as u64, d_n_a as u64,
                    d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64]);
            }
            cuda::sync();
            let t = std::time::Instant::now();
            for _ in 0..reps {
                launch_v(f_fast, 1, 1, 1, 256, &[
                    d_scores as u64, d_ncb as u64, d_sel_a as u64, d_n_a as u64,
                    d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64]);
            }
            cuda::sync();
            let us = t.elapsed().as_secs_f64() * 1e6 / reps as f64;
            println!("[qsa-probe] fast G=  1  {us:7.2} us per call (1 launch)");
        }
        if diffs > 0 {
            crow_nest_engine::log::shutdown();
            std::process::exit(1);
        }
    }
}
