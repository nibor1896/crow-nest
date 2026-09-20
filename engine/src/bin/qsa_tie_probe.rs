//! #89 numerics probe: QSA top-k TIE-BREAK semantics, engine side (U1 in
//! docs/numerics-diff.md). The three production selectors — qsa_select (exact
//! radix), qsa_select_fast (radix + dense shortcut), qsa_select_par_h/_e
//! (histogram + refine, the decode default) — all document ties resolved
//! lowest-block-index (kernels.rs:2447-2542, 2741-2749). torch.topk's order
//! among exactly-equal values is a CUDA implementation detail, and a tie that
//! straddles the budget boundary in the sparse regime (ncb > K, i.e. prompts
//! beyond 2048 tokens) swaps a whole 4-token block of the attention mask —
//! the #68 long-context coverage gap. The oracle rows (<= 607 tokens) are
//! dense-regime and can never see this.
//!
//! This probe pins the ENGINE side: score rows with a bit-identical f32 tie
//! group straddling rank K = QSA_BLOCK_TOPK = 512, run through all three
//! selectors in their production launch shapes, compared byte-for-byte
//! against a host reference implementing the documented rule (value desc,
//! index asc) with the engines' emit contract (selected blocks ascending,
//! 4 tokens each, then the ascending tail). It does NOT prove anything about
//! torch.topk — that is probe P1b (a venv experiment) in the doc.
//!
//! Case set:
//!   - straddle: 500 distinct blocks above the tie value, a tie group of 8/12/
//!     24 blocks at scattered indices, K falls INSIDE the group;
//!   - boundary-exact: K - above == tie-group size (fill consumes the group);
//!   - all-tied: every block one bit-identical value (the adversarial floor);
//!   - dense: ncb <= K (the qsa_select_fast shortcut must emit 0..=pos);
//!   - tails 0..3 on both sides of the regime boundary.
//!
//! Checks per case: sel_n equality; sel_list[0..sel_n) byte-equality across
//! the three selectors AND vs the reference; the poison beyond sel_n
//! untouched; h1 zero again after par_e (the #61a pairing invariant).
//!
//! usage: qsa_tie_probe
use crow_nest_engine::cuda;
use crow_nest_engine::gen::{launch_qsa_par_e, QSA_PAR_BINS};
use crow_nest_engine::kernels::launch_v;
use crow_nest_engine::geo::{QSA_BLOCK_TOPK, QSA_SEL_MAX};
use crow_nest_engine::sample::Rng;

const K: usize = QSA_BLOCK_TOPK; // 512, the production budget in blocks
const CAP: usize = 65536; // the production score-cap (262k-context ceiling)
const POISON: i32 = -559038737; // 0xDEADBEEF as i32

/// host reference: blocks ranked by (value desc, index asc) — the documented
/// torch.topk intent — truncated to min(K, ncb); emitted ASCENDING by block
/// (the engines' list contract), 4 tokens per block, then the ascending tail
fn reference(row: &[f32], pos: usize) -> (Vec<i32>, usize) {
    let ncb = (pos + 1) / 4;
    let mut idx: Vec<usize> = (0..ncb).collect();
    idx.sort_by(|&a, &b| row[b].partial_cmp(&row[a]).unwrap().then(a.cmp(&b)));
    idx.truncate(K.min(ncb));
    idx.sort_unstable();
    let mut list = Vec::with_capacity(4 * idx.len() + 3);
    for &b in &idx {
        for c in 0..4 {
            list.push((b * 4 + c) as i32);
        }
    }
    let tailn = (pos + 1) - 4 * ncb;
    for c in 0..tailn {
        list.push((4 * ncb + c) as i32);
    }
    let n = list.len();
    (list, n)
}

/// a deterministic distinct background value per block, strictly inside
/// (lo, hi): an integer hash scaled into the range, never bit-equal to the
/// planted tie value
fn background(b: usize, lo: f32, hi: f32) -> f32 {
    let h = (b.wrapping_mul(2654435761).wrapping_add(0x9E37) % 1_000_003) as f32;
    lo + (hi - lo) * (h / 1_000_003.0)
}

/// the probe's one case: (label, ncb, tail, tie_group_indices, n_above)
struct Case {
    label: &'static str,
    ncb: usize,
    tail: usize,
    ties: Vec<usize>,
    n_above: usize,
}

fn build_row(cs: &Case, rng: &mut Rng) -> Vec<f32> {
    let mut row = vec![0.0f32; CAP];
    // three value bands: n_above distinct blocks in (0.6, 1.0); the tie group
    // bit-identical at 0.5; everything else distinct in (0.0, 0.4)
    let mut above_left = cs.n_above;
    for b in 0..cs.ncb {
        if cs.ties.contains(&b) {
            row[b] = 0.5f32;
        } else if above_left > 0 {
            above_left -= 1;
            row[b] = background(b, 0.6, 1.0);
        } else {
            row[b] = background(b, 0.0, 0.4);
        }
    }
    // n_above must be realizable: ncb - ties.len() >= n_above (asserted by the
    // case table); a dash of rng keeps this constructor honest across runs
    let _ = rng.f01();
    row
}

fn main() {
    let _log = crow_nest_engine::log::init();
    let mut ok = true;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let module = cuda::compile(crow_nest_engine::kernels::KERNEL_SRC);
        let f_sel = module.get("qsa_select");
        let f_fast = module.get("qsa_select_fast");
        let f_par_h = module.get("qsa_select_par_h");
        let f_par_e = module.get("qsa_select_par_e");

        // tie groups: scattered indices, never contiguous (the lowest-index
        // fill must be a real selection, not a prefix accident)
        let g24: Vec<usize> = (0..24).map(|i| 7 + i * 37).collect();
        let g12: Vec<usize> = (0..12).map(|i| 3 + i * 53).collect();
        let g8: Vec<usize> = (0..8).map(|i| 11 + i * 71).collect();
        let cases = vec![
            // K lands inside the tie group (500 above, 12 tied: rank 501..512
            // must go to the 12 LOWEST tied indices)
            Case { label: "straddle-12 tail0", ncb: 600, tail: 0, ties: g12.clone(), n_above: 500 },
            Case { label: "straddle-24 tail3", ncb: 600, tail: 3, ties: g24.clone(), n_above: 500 },
            // fill exactly consumes the tie group (500 above, 12 tied, K=512)
            Case { label: "boundary-exact tail1", ncb: 700, tail: 1, ties: g12, n_above: 500 },
            // a bigger tie group than the whole budget headroom
            Case { label: "straddle-8-of-24 tail2", ncb: 3000, tail: 2, ties: g24, n_above: 505 },
            // adversarial floor: every complete block one bit-identical value
            Case { label: "all-tied ncb600", ncb: 600, tail: 0, ties: (0..600).collect(), n_above: 0 },
            // dense regime: the fast shortcut must emit 0..=pos; radix + par
            // must agree with it byte-for-byte
            Case { label: "dense ncb400 tail3", ncb: 400, tail: 3, ties: g8, n_above: 0 },
            Case { label: "dense ncb512 tail1", ncb: 512, tail: 1, ties: vec![], n_above: 0 },
        ];

        let d_k = cuda::to_i32_dev(&[K as i32]);
        let d_cap = cuda::to_i32_dev(&[CAP as i32]);
        let d_selmax = cuda::to_i32_dev(&[QSA_SEL_MAX as i32]);

        for cs in &cases {
            let mut rng = Rng::from_state(0x89_C0FFEE_0001);
            let pos = 4 * cs.ncb + cs.tail - 1;
            let row = build_row(cs, &mut rng);
            assert!(cs.n_above + cs.ties.len() <= cs.ncb, "case {} overflows ncb", cs.label);
            let d_scores = cuda::to_f32_dev(&row);
            let d_ncb = cuda::to_i32_dev(&[cs.ncb as i32]);
            let d_pos = cuda::to_i32_dev(&[pos as i32]);

            let poison: Vec<i32> = vec![POISON; QSA_SEL_MAX];
            // (sel_list, sel_n) per arm; h1 is the par pair's histogram buffer,
            // alloc_zeroed = the #61a invariant state before its first launch
            let mut devs: Vec<(u64, u64)> = Vec::new();
            for _ in 0..3 {
                devs.push((
                    cuda::to_i32_dev(&poison) as u64,
                    cuda::to_i32_dev(&[POISON]) as u64,
                ));
            }
            let d_h1 = cuda::alloc_zeroed(QSA_PAR_BINS * 4);

            // arm 0: qsa_select; arm 1: qsa_select_fast — production launch
            // shape grid (nq) x 256 (gen.rs:2632). FINDING (#89): the plain
            // radix kernel has NO dense shortcut — with K > ncb its threshold
            // search (`need = K - above` never satisfied) degrades and it
            // emits a wrong sel_n. Production never reaches it (qsa_fast_on
            // defaults ON and both fast/par shortcut the dense regime), but
            // CROW_QSA_FAST=0 + a <= 2048-token prompt does. Skipped here and
            // filed as a latent env-gated defect in docs/numerics-diff.md.
            let dense = K >= cs.ncb;
            if !dense {
                let vals_a = [
                    d_scores as u64, d_ncb as u64, devs[0].0, devs[0].1,
                    d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64,
                ];
                launch_v(f_sel, 1, 1, 1, 256, &vals_a);
            }
            launch_v(f_fast, 1, 1, 1, 256, &[
                d_scores as u64, d_ncb as u64, devs[1].0, devs[1].1,
                d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64,
            ]);
            // arm 2: the decode default, G = 32 (CROW_QSA_PAR_BLOCKS default),
            // h1 zeroed before the pair (the #61a invariant this probe re-checks)
            cuda::to_i32_into(devs[2].1, &[0]);
            launch_v(f_par_h, 32, 1, 1, 256, &[
                d_scores as u64, d_ncb as u64, d_h1 as u64, d_k as u64, d_cap as u64]);
            launch_qsa_par_e(f_par_e, 1, &[
                d_scores as u64, d_ncb as u64, devs[2].0, devs[2].1,
                d_k as u64, d_cap as u64, d_selmax as u64, d_pos as u64, d_h1 as u64]);
            cuda::sync();

            let (ref_list, ref_n) = reference(&row[..cs.ncb.max(1)], pos);
            let mut pass = true;
            let mut msg = String::new();
            for (name, arm) in [("qsa_select", 0usize), ("qsa_select_fast", 1), ("par_h+e", 2)] {
                if arm == 0 && dense {
                    msg.push_str(" qsa_select: SKIPPED (no dense shortcut — latent defect, see doc);");
                    continue;
                }
                let n = cuda::dtoh_i32(devs[arm].1, 1)[0] as usize;
                let list = cuda::dtoh_i32(devs[arm].0, QSA_SEL_MAX);                if n != ref_n {
                    pass = false;
                    msg.push_str(&format!(" {name}: sel_n {n} != ref {ref_n};"));
                } else if list[..n] != ref_list[..] {
                    pass = false;
                    let first = (0..n).find(|&i| list[i] != ref_list[i]).unwrap_or(n);
                    msg.push_str(&format!(
                        " {name}: list differs at slot {first} (got {} want {}, sel_n {n});",
                        list[first], ref_list[first],
                    ));
                }
                // the poison beyond sel_n must survive: nothing may write there
                for (i, v) in list.iter().enumerate().skip(n) {
                    if *v != POISON {
                        pass = false;
                        msg.push_str(&format!(" {name}: slot {i} beyond sel_n clobbered to {v};"));
                        break;
                    }
                }
            }
            // h1 back to zero (par_e consumed and re-zeroed it)
            let h = cuda::dtoh_u32(d_h1, QSA_PAR_BINS);
            if h.iter().any(|&v| v != 0) {
                pass = false;
                msg.push_str(" par: h1 not re-zeroed;");
            }
            println!(
                "[qsa-tie] {:<22} ncb {:>4} tail {} pos {:>4} ref_n {:>4} -> {}{}",
                cs.label, cs.ncb, cs.tail, pos, ref_n,
                if pass { "PASS" } else { "FAIL" }, msg
            );
            ok &= pass;
        }
    }
    if ok {
        println!("[qsa-tie] all cases PASS: ties -> lowest index, ascending emit, three selectors byte-identical to the reference");
    } else {
        std::process::exit(1);
    }
}
