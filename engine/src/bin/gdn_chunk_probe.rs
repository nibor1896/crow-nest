//! #89 numerics probe P2 (docs/numerics-diff.md §3 P2): the GDN prefill
//! numerics class — crow's token-recurrent prefill (`delta_rule_persist_r`,
//! the register-resident form gen.rs launches by default) vs HF's CHUNKED
//! prefill (`torch_chunk_gated_delta_rule`, chunk 64, modeling:266-344), on
//! identical synthetic f32 inputs.
//!
//! Inputs come from the torch dumper (.venv-oracle; /tmp/p2_gdn_dump.py):
//! for T in {64, 512, 4096}, 48 heads x 128 dims, normed q/k (q pre-scaled
//! 1/sqrt(128) — exactly the `l2norm_repeat` output class), v/g/beta drawn
//! O(1), plus BOTH torch references on the same bits: the step-recurrent
//! form (crow's algorithm class, different device reduction order) and the
//! chunked form (the HF production prefill). Dump layout (f32 LE sections,
//! in order): q, k, v [T,48,128]; g, beta [T,48]; out_rec, out_chunk
//! [T,48,128]; S_rec, S_chunk [48,128,128] — all crow layouts (t-major,
//! state [h][dk][dv], matching delta_rule_persist_r's S[dk*128+d]).
//!
//! Measurements: whole-tensor and worst-head rel_L2 + max_abs of
//!   crow out vs out_chunk (the class bound — the 1e-5 gate),
//!   crow out vs out_rec (device-noise floor for the same algorithm),
//!   crow S   vs S_chunk / S_rec (state compounding across T).
//!
//! usage: gdn_chunk_probe [dumpdir]   (default /tmp/crow-gdn-p2)
use crow_nest_engine::cuda;
use crow_nest_engine::geo::{GDN_VHEADS as H, GD as D};
use crow_nest_engine::kernels::launch_v;

fn read_f32(path: &std::path::Path) -> Vec<f32> {
    let b = std::fs::read(path).unwrap_or_else(|e| panic!("read {}: {e}", path.display()));
    assert_eq!(b.len() % 4, 0, "{}: not f32-aligned", path.display());
    b.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

/// rel_L2 (||a-b||/||b||), max_abs, and worst per-head rel_L2 over [T][H][D]
fn stats(a: &[f32], b: &[f32], t: usize) -> (f64, f32, f64) {
    let hd = H * D;
    assert_eq!(a.len(), t * hd);
    let (mut n2, mut b2, mut mx) = (0f64, 0f64, 0f32);
    let mut worst_head = 0f64;
    for h in 0..H {
        let (mut hn, mut hb) = (0f64, 0f64);
        for i in 0..t {
            for d in 0..D {
                let (x, y) = (a[i * hd + h * D + d], b[i * hd + h * D + d]);
                let dd = (x - y) as f64;
                hn += dd * dd;
                hb += (y as f64) * (y as f64);
                mx = mx.max((x - y).abs());
            }
        }
        n2 += hn;
        b2 += hb;
        worst_head = worst_head.max((hn / hb).sqrt());
    }
    ((n2 / b2).sqrt(), mx, worst_head)
}

/// rel_L2 / max_abs for the [H][D][D] state
fn stats_s(a: &[f32], b: &[f32]) -> (f64, f32) {
    let (mut n2, mut b2, mut mx) = (0f64, 0f64, 0f32);
    for (&x, &y) in a.iter().zip(b) {
        let dd = (x - y) as f64;
        n2 += dd * dd;
        b2 += (y as f64) * (y as f64);
        mx = mx.max((x - y).abs());
    }
    ((n2 / b2).sqrt(), mx)
}

fn main() {
    let _log = crow_nest_engine::log::init();
    let dir = std::env::args().nth(1).unwrap_or_else(|| "/tmp/crow-gdn-p2".into());
    let mut ok = true;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f = module.get("delta_rule_persist_r");

        for &t in &[64usize, 512, 4096] {
            let path = std::path::Path::new(&dir).join(format!("gdn_T{t}.bin"));
            let buf = read_f32(&path);
            let nthd = t * H * D;
            let nth = t * H;
            let ns = H * D * D;
            // section offsets in dump order
            let mut o = 0usize;
            let sec = |o: &mut usize, n: usize| {
                let s = &buf[*o..*o + n];
                *o += n;
                s
            };
            let q = sec(&mut o, nthd);
            let k = sec(&mut o, nthd);
            let v = sec(&mut o, nthd);
            let g = sec(&mut o, nth);
            let beta = sec(&mut o, nth);
            let out_rec = sec(&mut o, nthd);
            let out_chunk = sec(&mut o, nthd);
            let s_rec = sec(&mut o, ns);
            let s_chunk = sec(&mut o, ns);
            assert_eq!(o, buf.len(), "{}: trailing bytes", path.display());

            let d_q = cuda::to_f32_dev(q);
            let d_k = cuda::to_f32_dev(k);
            let d_v = cuda::to_f32_dev(v);
            let d_g = cuda::to_f32_dev(g);
            let d_beta = cuda::to_f32_dev(beta);
            let d_out = cuda::alloc_zeroed(nthd * 4);
            let d_s = cuda::alloc_zeroed(ns * 4);
            let d_t = cuda::to_i32_dev(&[t as i32]);
            let d_init = cuda::to_i32_dev(&[1]);
            // production launch shape (gen.rs:2362): grid (48 heads) x 128
            launch_v(f, H as u32, 1, 1, D as u32, &[
                d_q as u64, d_k as u64, d_v as u64, d_g as u64, d_beta as u64,
                d_out as u64, d_s as u64, d_t as u64, d_init as u64,
            ]);
            cuda::sync();
            let out = cuda::dtoh(d_out, nthd);
            let s = cuda::dtoh(d_s, ns);

            let (rc, mxc, whc) = stats(&out, out_chunk, t);
            let (rr, mxr, whr) = stats(&out, out_rec, t);
            let (sc, mxsc) = stats_s(&s, s_chunk);
            let (sr, mxsr) = stats_s(&s, s_rec);
            let norm_s = s.iter().map(|&x| (x as f64) * (x as f64)).sum::<f64>().sqrt();
            // torch's own rec-vs-chunk floor on the same bits (printed by the
            // dumper) for context
            let pass = rc <= 1e-5 && whc <= 1e-5;
            ok &= pass;
            println!(
                "[gdn-chunk] T={t:<5} out vs CHUNKED: rel_L2 {rc:.3e} (worst head {whc:.3e}) max_abs {mxc:.3e}  -> {} (1e-5 line)",
                if pass { "PASS" } else { "FAIL" }
            );
            println!(
                "[gdn-chunk]        out vs recurrent: rel_L2 {rr:.3e} (worst head {whr:.3e}) max_abs {mxr:.3e}  (same-algorithm device floor)"
            );
            println!(
                "[gdn-chunk]        S vs chunked {sc:.3e} (max_abs {mxsc:.3e})  S vs recurrent {sr:.3e} (max_abs {mxsr:.3e})  ||S|| {norm_s:.2}"
            );
        }
    }
    if ok {
        println!("[gdn-chunk] PASS: crow recurrent prefill within 1e-5 rel_L2 of the HF chunked reference at every T");
    } else {
        std::process::exit(1);
    }
}
