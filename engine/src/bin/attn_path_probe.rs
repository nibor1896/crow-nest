//! #89 numerics probe: prefill-vs-decode QSA attention ROW EQUALITY (row 21
//! of docs/numerics-diff.md). The prefill path (attn_sel and its bit-identical
//! faster twins) normalizes each softmax weight `p[j]/sum` and then accumulates
//! `o += w*v` in list order; the decode path (attn_sel_split + attn_merge)
//! accumulates UNNORMALIZED partials per split block and divides `o/L` once
//! after the merge (kernels.rs:1886-1901 vs 3740-3791). Same token set, same
//! f32 class, different rounding schedule — this probe measures that floor on
//! the worst production shape: the full sparse list (n = QSA_SEL_MAX = 2051
//! tokens), 24 heads x 256 dims, bf16 KV (mode 1, so the probe can produce
//! cache bytes on the host by f32->bf16 truncation exactly like the engine's
//! store_kv keep path).
//!
//! attn_sel (base list kernel) is the reference form all prefill twins claim
//! bit-identity to; attn_sel_split S in {1, 2, 8} covers the decode split
//! counts (S=1 collapses to a single partial, S=8 the production long-list
//! shape). Report: per-element max_abs, rel_L2 = ||a-b||2 / ||a||2 over the
//! whole 24x256 output row, argmax element. Soft pass at rel_L2 <= 1e-5
//! (expected ~1e-7: only division placement and split-partial reduction
//! order differ; the token set is identical by construction).
//!
//! usage: attn_path_probe
use crow_nest_engine::cuda;
use crow_nest_engine::kernels::launch_v;
use crow_nest_engine::geo::{AHD, NKV, NQ, QSA_SEL_MAX};
use crow_nest_engine::sample::Rng;

const TMAX: usize = 4096; // cache rows per kv head (> QSA_SEL_MAX)
const MODE: i32 = 1; // bf16 KV: host can emit exact bf16 bytes by truncation

fn main() {
    let _log = crow_nest_engine::log::init();
    let mut ok = true;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f_sel = module.get("attn_sel");
        let f_split = module.get("attn_sel_split");
        let f_merge = module.get("attn_merge");

        let mut rng = Rng::from_state(0x89_DEC0DE_0002);

        // q: [1][24][256] f32 in [-1, 1] -> logits = q.k * 0.0625 in O(1)
        let q: Vec<f32> = (0..NQ * AHD).map(|_| (rng.f01() * 2.0 - 1.0) as f32).collect();
        // KV: [NKV][TMAX][256] bf16, distinct values, produced by f32 -> bf16
        // truncation (the top 16 bits), the exact cast store_kv mode-1 does
        let kv_f32: Vec<f32> = (0..NKV * TMAX * AHD).map(|_| (rng.f01() * 2.0 - 1.0) as f32).collect();
        let kv_bf16: Vec<u16> = kv_f32.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
        let kv_bytes: Vec<u8> = kv_bf16
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();

        // the selected list: n = QSA_SEL_MAX distinct ascending tokens (the
        // even positions of the first 2*QSA_SEL_MAX rows), the sparse worst
        // case; sel_n = n
        let n = QSA_SEL_MAX;
        let sel: Vec<i32> = (0..n).map(|i| (2 * i) as i32).collect();

        let d_q = cuda::to_f32_dev(&q);
        let d_kc = cuda::upload_dev(&kv_bytes);
        let d_vc = cuda::upload_dev(&kv_bytes);
        let d_sel = cuda::to_i32_dev(&sel);
        let d_sel_n = cuda::to_i32_dev(&[n as i32]);
        let d_tmax = cuda::to_i32_dev(&[TMAX as i32]);
        let d_mode = cuda::to_i32_dev(&[MODE]);
        let d_selmax = cuda::to_i32_dev(&[QSA_SEL_MAX as i32]);

        // prefill reference row: attn_sel grid (24, 1) block 256
        let out_n = NQ * AHD;
        let d_out_a = cuda::alloc_zeroed(out_n * 4);
        launch_v(f_sel, NQ as u32, 1, 1, AHD as u32, &[
            d_q as u64, d_kc as u64, d_vc as u64, d_sel as u64, d_sel_n as u64,
            d_tmax as u64, d_mode as u64, d_selmax as u64, d_out_a as u64]);

        let a = cuda::dtoh(d_out_a, out_n);
        let norm_a = a.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>().sqrt();

        for s in [1u32, 2, 8] {
            // decode form: attn_sel_split grid (24, 1, S) block 256 emits
            // (part_o[256], part_ml[2]) per (head, split), attn_merge combines
            let d_part_o = cuda::alloc_zeroed((NQ * s as usize) * AHD * 4);
            let d_part_ml = cuda::alloc_zeroed((NQ * s as usize) * 2 * 4);
            let d_out_b = cuda::alloc_zeroed(out_n * 4);
            let d_s = cuda::to_i32_dev(&[s as i32]);
            launch_v(f_split, NQ as u32, 1, s, AHD as u32, &[
                d_q as u64, d_kc as u64, d_vc as u64, d_sel as u64, d_sel_n as u64,
                d_tmax as u64, d_mode as u64, d_selmax as u64,
                d_part_o as u64, d_part_ml as u64]);
            launch_v(f_merge, NQ as u32, 1, 1, AHD as u32, &[
                d_part_o as u64, d_part_ml as u64, d_out_b as u64, d_s as u64]);
            cuda::sync();

            let b = cuda::dtoh(d_out_b, out_n);
            let mut max_abs: f32 = 0.0;
            let mut max_i = 0usize;
            let mut l2 = 0.0f64;
            for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
                let d = (*x - *y).abs();
                if d > max_abs {
                    max_abs = d;
                    max_i = i;
                }
                let dd = (*x as f64) - (*y as f64);
                l2 += dd * dd;
            }
            let rel_l2 = (l2.sqrt() / norm_a) as f32;
            let pass = rel_l2 <= 1.0e-5 && b.iter().all(|v| v.is_finite());
            println!(
                "[attn-path] n={n} S={s}: max_abs {max_abs:.3e} at [{},{}]  rel_L2 {rel_l2:.3e}  ||a|| {norm_a:.4}  -> {}",
                max_i / AHD, max_i % AHD,
                if pass { "PASS (1e-5 line)" } else { "FAIL" },
            );
            ok &= pass;
        }
    }
    if ok {
        println!("[attn-path] prefill attn_sel vs decode attn_sel_split+attn_merge agree within the 1e-5 line at the full sparse list length");
    } else {
        std::process::exit(1);
    }
}
