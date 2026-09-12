//! #61a unit gate: qsa_select_par (two launches) against qsa_select_fast (one
//! block) on synthetic score rows. Compares sel_n and sel_list[0..sel_n] byte
//! for byte. No model, no container, no engine lock: it compiles the kernel
//! source and runs the two selections on device buffers.
//!
//! usage: qsa_probe [--rows N] [--bench]
use crow_nest_engine::cuda;
use crow_nest_engine::gen::launch_v;

/// deterministic xorshift64*: the same rows on every machine and every run
struct Rng(u64);
impl Rng {
    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn f01(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / (1u32 << 24) as f32
    }
}

/// score distributions: the narrow ones are the hard cases for a radix top-k
/// (every key shares the top digits), the tie ones exercise the lowest-index fill
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
    let args: Vec<String> = std::env::args().collect();
    let bench = args.iter().any(|a| a == "--bench");
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f_fast = module.get("qsa_select_fast");
        let f_par_h = module.get("qsa_select_par_h");
        let f_par_e = module.get("qsa_select_par_e");

        const CAP: usize = 65536;
        const SEL_CAP: usize = 70000;
        let d_scores = cuda::alloc_zeroed(CAP * 4);
        let d_sel_a = cuda::alloc_zeroed(SEL_CAP * 4);
        let d_sel_b = cuda::alloc_zeroed(SEL_CAP * 4);
        let d_n_a = cuda::alloc_zeroed(4);
        let d_n_b = cuda::alloc_zeroed(4);
        let d_h1 = cuda::alloc_zeroed(4096 * 4);
        let d_ncb = cuda::to_i32_dev(&[0]);
        let d_k = cuda::to_i32_dev(&[0]);
        let d_cap = cuda::to_i32_dev(&[CAP as i32]);
        let d_selmax = cuda::to_i32_dev(&[0]);
        let d_pos = cuda::to_i32_dev(&[0]);

        // (ncb, tail, K) cases: the 16,320 row of the operating point, the dense
        // regime (K >= ncb), tiny rows, and K values around the 512 budget
        let shapes: &[(usize, usize, usize)] = &[
            (4080, 0, 512),
            (4080, 3, 512),
            (4080, 1, 1),
            (4080, 2, 7),
            (4080, 0, 4079),
            (4080, 0, 4080),
            (4080, 2, 5000),
            (2048, 1, 512),
            (1024, 3, 512),
            (513, 0, 512),
            (512, 1, 512),
            (256, 2, 512),
            (100, 3, 512),
            (1, 0, 512),
            (0, 3, 512),
            (63, 1, 7),
            (4096, 0, 512),
            (3000, 2, 300),
        ];
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

        let mut host = vec![0f32; CAP];
        let mut rng = Rng(0x1234_5678_9abc_def1);
        let mut rows = 0usize;
        let mut diffs = 0usize;
        let mut first_diff: Option<String> = None;
        let per_seed = shapes.len() * dists.len();
        let mut seeds = 2usize;
        if want_rows > 0 {
            seeds = ((want_rows + per_seed - 1) / per_seed).max(1);
        }
        for seed in 0..seeds {
            for &(ncb, tail, k) in shapes {
                for &dist in dists {
                    let pos = 4 * ncb + tail;
                    let sel_max = std::cmp::max(4 * k + 3, pos + 1);
                    assert!(sel_max <= SEL_CAP, "sel_max {sel_max} over the probe buffer");
                    fill_row(dist, ncb.max(1), &mut rng, &mut host);
                    cuda::to_f32_into(d_scores, &host[..ncb.max(1)]);
                    cuda::to_i32_into(d_ncb, &[ncb as i32]);
                    cuda::to_i32_into(d_k, &[k as i32]);
                    cuda::to_i32_into(d_selmax, &[sel_max as i32]);
                    cuda::to_i32_into(d_pos, &[pos as i32]);
                    // poison both lists so a short write shows up as a difference
                    let poison = vec![-7i32; sel_max];
                    cuda::to_i32_into(d_sel_a, &poison);
                    cuda::to_i32_into(d_sel_b, &poison);
                    cuda::sync();

                    launch_v(f_fast, 1, 1, 1, 256, &[
                        d_scores as u64, d_ncb as u64, d_sel_a as u64, d_n_a as u64,
                        d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64]);
                    launch_v(f_par_h, 32, 1, 1, 256, &[
                        d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64, d_cap as u64]);
                    launch_v(f_par_e, 1, 1, 1, 1024, &[
                        d_scores as u64, d_ncb as u64, d_sel_b as u64, d_n_b as u64,
                        d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64, d_h1 as u64]);
                    cuda::sync();

                    let na = cuda::dtoh_i32(d_n_a, 1)[0];
                    let nb = cuda::dtoh_i32(d_n_b, 1)[0];
                    let la = cuda::dtoh_i32(d_sel_a, sel_max);
                    let lb = cuda::dtoh_i32(d_sel_b, sel_max);
                    rows += 1;
                    let mut bad = None;
                    if na != nb {
                        bad = Some(format!("sel_n {na} vs {nb}"));
                    } else {
                        for i in 0..(na.max(0) as usize) {
                            if la[i] != lb[i] {
                                bad = Some(format!("sel_list[{i}] {} vs {}", la[i], lb[i]));
                                break;
                            }
                        }
                    }
                    // the histogram must be zero again for the next call
                    let h = cuda::dtoh_u32(d_h1, 4096);
                    if bad.is_none() && h.iter().any(|&v| v != 0) {
                        bad = Some("h1 not zero after the emit kernel".to_string());
                    }
                    if let Some(m) = bad {
                        diffs += 1;
                        if first_diff.is_none() {
                            first_diff = Some(format!(
                                "ncb {ncb} tail {tail} K {k} dist {dist:?} seed {seed}: {m}"));
                        }
                    }
                }
            }
        }
        println!("[qsa-probe] rows {rows}  differences {diffs}");
        if let Some(m) = first_diff {
            println!("[qsa-probe] first difference: {m}");
        }

        if bench {
            let ncb = 4080usize;
            let pos = 4 * ncb;
            let k = 512usize;
            let sel_max = 4 * k + 3;
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
                    launch_v(f_par_e, 1, 1, 1, 1024, &[
                        d_scores as u64, d_ncb as u64, d_sel_b as u64, d_n_b as u64,
                        d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64, d_h1 as u64]);
                }
                cuda::sync();
                let t = std::time::Instant::now();
                for _ in 0..reps {
                    launch_v(f_par_h, g, 1, 1, 256, &[
                        d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64, d_cap as u64]);
                    launch_v(f_par_e, 1, 1, 1, 1024, &[
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
            std::process::exit(1);
        }
    }
}
