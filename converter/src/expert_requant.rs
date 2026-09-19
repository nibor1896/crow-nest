//! #79: NVFP4 for the routed experts with importance-weighted sub-block scales.
//!
//! The container's own `quantize_nvfp4` picks each 16-wide sub-block's `ue4m3` scale by the
//! UNWEIGHTED squared error of the 16 weights (`--scales mse`). Every one of those 16 values
//! belongs to a DIFFERENT input column of the same output row — NVFP4's block runs along the
//! row — so a scale chosen without knowing what those columns carry spends its resolution
//! evenly over columns the model never excites and columns it lives on.
//!
//! llama.cpp has done the weighted version since q4_K: in `ggml/src/ggml-quants.c`, both
//! `quantize_row_q4_K_impl` and `quantize_row_iq4_nl_impl` build
//!
//! ```text
//! sigma2   = 2 * sum(x^2) / super_block_size
//! weight[l] = qw[l] * sqrtf(sigma2 + x[l]*x[l])
//! ```
//!
//! and then minimize the WEIGHTED squared error over the scale. `iq4_nl` is the closer twin of
//! what is needed here, because it searches a scale for a FIXED non-uniform 16-level grid, the
//! same shape of problem NVFP4's E2M1 grid poses: it takes `d = sumqx/sumq2` (the weighted
//! least-squares scale for the current levels) and then tries neighbours, keeping the best by
//! `sumqx^2/sumq2` — which is exactly the smallest weighted SSE.
//!
//! There is no NVFP4 prior art to copy: llama.cpp's own `quantize_nvfp4` ignores
//! `quant_weights` entirely. What is copied is the METHOD, and the three differences are named:
//!
//! - the super block is the 64-value NVFP4 block (the unit four sub-block scales share), so
//!   `sigma2 = 2 * sum_64(x^2) / 64`;
//! - the scale is not a free float but one of 128 `ue4m3` ladder steps times the tensor's f32
//!   global scale, so the analytic optimum is BRACKETED by its two neighbouring steps and
//!   scored against them, exactly as `encode_subblock_mse` already does unweighted;
//! - the GLOBAL scale convention does not move. It is still `max over sub-blocks(max|x|/6)/448`
//!   computed from the same values, so every byte of the container's geometry, every kernel and
//!   the engine's `gs_dev` see what they saw.
//!
//! `Rule::FourSix` adds NVIDIA's Nemotron 3 Ultra candidate (arXiv 2512.02010): per block,
//! choose whether the block maximum sits at the top of the E2M1 grid (6) or at 4, by
//! reconstruction error. Here that is one more pair of ladder candidates — `max|x|/4` — scored
//! by the same weighted SSE, so it can only ever be chosen when it is better.

use crate::{decode_ue4m3, e2m1_index, encode_subblock_ceil, encode_subblock_mse, quant_dequant, E2M1_GRID, UE4M3_MAX};

/// Which sub-block scale rule to run.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rule {
    /// (a) the CONTROL: the container's own `--scales mse`, unweighted. Byte-identical output.
    Mse,
    /// (b) llama.cpp's form: weighted least squares on the E2M1 shape, bracketed on the ladder.
    Imatrix,
    /// (c) (b) plus the four-over-six candidate.
    ImatrixFourSix,
}

impl Rule {
    pub fn parse(s: &str) -> Option<Rule> {
        match s {
            "mse" => Some(Rule::Mse),
            "imatrix" => Some(Rule::Imatrix),
            "imatrix46" => Some(Rule::ImatrixFourSix),
            _ => None,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            Rule::Mse => "mse",
            Rule::Imatrix => "imatrix",
            Rule::ImatrixFourSix => "imatrix46",
        }
    }
    fn weighted(self) -> bool {
        self != Rule::Mse
    }
}

/// Weight-space numbers for one tensor under one rule. Everything here is measured on the
/// bytes that were actually written, by dequantizing them again.
#[derive(Clone, Debug, Default)]
pub struct ExpertStats {
    pub n: u64,
    /// sum of squared error, unweighted
    pub sse: f64,
    /// sum of `imatrix_j * err^2`, with the PLAIN importance weight (mean squared activation),
    /// not the `sqrt(sigma2 + x^2)` factor of the search
    pub wsse: f64,
    /// sum of the same `imatrix_j` over every value, so `wsse / wsum` is a weighted MSE
    pub wsum: f64,
    pub max_abs_err: f32,
    /// values whose magnitude ran past `6 * sub-block scale` and saturated
    pub clipped: u64,
    /// sub-blocks whose chosen ladder byte differs from what `Rule::Mse` would have chosen
    pub scales_moved: u64,
    pub sub_blocks: u64,
}

impl ExpertStats {
    fn absorb(&mut self, o: &ExpertStats) {
        self.n += o.n;
        self.sse += o.sse;
        self.wsse += o.wsse;
        self.wsum += o.wsum;
        self.max_abs_err = self.max_abs_err.max(o.max_abs_err);
        self.clipped += o.clipped;
        self.scales_moved += o.scales_moved;
        self.sub_blocks += o.sub_blocks;
    }
    pub fn mse(&self) -> f64 {
        if self.n == 0 {
            0.0
        } else {
            self.sse / self.n as f64
        }
    }
    /// `sum(imatrix * err^2) / sum(imatrix)` — the quantity rule (b) is supposed to lower, and
    /// the first-order proxy for the squared error this rounding puts on the layer's OUTPUT:
    /// `E[(dw . x)^2] = sum_j dw_j^2 E[x_j^2]` for independent columns.
    pub fn weighted_mse(&self) -> f64 {
        if self.wsum <= 0.0 {
            0.0
        } else {
            self.wsse / self.wsum
        }
    }
}

/// Weighted SSE of one 16-wide sub-block at decode scale `s`, with the SAME rounding the
/// packing uses, so a candidate's score describes exactly the bytes it would write.
fn subblock_wsse(sub: &[f32], w: &[f64], s: f32) -> f64 {
    if s <= 0.0 {
        return sub.iter().zip(w).map(|(v, wl)| wl * (*v as f64) * (*v as f64)).sum();
    }
    let inv = 1.0 / s;
    let mut acc = 0.0f64;
    for (v, wl) in sub.iter().zip(w) {
        let e = (quant_dequant(*v, s, inv) - *v) as f64;
        acc += wl * e * e;
    }
    acc
}

/// The two ladder steps bracketing `t` (a scale in DIVIDED units), appended to `out`.
fn bracket(out: &mut Vec<u32>, t: f32) {
    if !t.is_finite() || t <= 0.0 {
        return;
    }
    let hi = encode_subblock_ceil(t);
    let lo = if hi == 0 {
        0
    } else if decode_ue4m3(hi) == t {
        hi
    } else {
        hi - 1
    };
    out.push(hi);
    out.push(lo);
}

/// The importance-weighted sub-block scale: llama.cpp's `iq4_nl` search on NVFP4's ladder.
///
/// `w[l]` is `qw[j_l] * sqrt(sigma2 + x_l^2)` — the caller builds it, because only the caller
/// knows which input column each of the 16 values belongs to.
fn encode_subblock_weighted(sub: &[f32], w: &[f64], global: f32, ceil_byte: u32, four_six: bool) -> u32 {
    let max_abs = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs <= 0.0 {
        return 0;
    }
    // d = sumqx/sumq2 for the current E2M1 shape, twice — the weighted twin of the generalized
    // Lloyd step `encode_subblock_mse` runs unweighted. Saturated elements sit at g = +/-6, so
    // clipping is priced into the same expression.
    let mut s = max_abs / 6.0;
    for _ in 0..2 {
        let inv = 1.0 / s;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for (l, &v) in sub.iter().enumerate() {
            let g = quant_dequant(v, 1.0, inv) as f64;
            num += w[l] * (v as f64) * g;
            den += w[l] * g * g;
        }
        if den > 0.0 {
            let s_next = (num / den) as f32;
            if s_next.is_finite() && s_next > 0.0 {
                s = s_next;
            } else {
                break;
            }
        } else {
            break;
        }
    }
    let mut cands: Vec<u32> = Vec::with_capacity(6);
    bracket(&mut cands, s / global);
    if four_six {
        // NVIDIA's four-over-six: the block maximum at grid level 4 instead of 6
        bracket(&mut cands, (max_abs / 4.0) / global);
    }
    let mut best = ceil_byte;
    let mut best_sse = subblock_wsse(sub, w, decode_ue4m3(ceil_byte) * global);
    for b in cands {
        if b == 0 || b == best {
            continue;
        }
        let sse = subblock_wsse(sub, w, decode_ue4m3(b) * global);
        if sse < best_sse {
            best_sse = sse;
            best = b;
        }
    }
    best
}

/// The tensor's f32 global scale, by the container's own convention: the largest sub-block
/// `max|x|/6` divided by 448. Unchanged from `quantize_nvfp4` on purpose — the overlay has to
/// carry the same convention the kernels and `gs_dev` already read.
pub fn global_scale(values: &[f32]) -> f32 {
    let mut max_scale = 0.0f32;
    for sub in values.chunks(16) {
        let max = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        max_scale = max_scale.max(max / 6.0);
    }
    if max_scale > 0.0 {
        max_scale / UE4M3_MAX
    } else {
        1.0
    }
}

/// Quantize ONE routed-expert tensor.
///
/// `values` is the tensor in row-major order, `per_expert` values per expert and `n_cols`
/// values per row, so value index `i` belongs to expert `i / per_expert` and input column
/// `i % n_cols`. `imw` is `n_experts * n_cols` importance weights, already divided by the
/// counts (see `imatrix::ExpertImatrix::row`); it is ignored by `Rule::Mse`.
///
/// Both `per_expert` and `n_cols` are multiples of 64 in this architecture (1280*2560 and 2560,
/// 2560*640 and 640), which is what makes a 16-wide sub-block belong to ONE expert and ONE row
/// and therefore to 16 consecutive input columns. It is asserted, not assumed.
pub fn quantize_expert_tensor(
    values: &[f32],
    per_expert: usize,
    n_cols: usize,
    imw: &[f32],
    rule: Rule,
    threads: usize,
) -> (Vec<u8>, f32, ExpertStats) {
    assert!(values.len() % 64 == 0, "{} values is not a multiple of 64", values.len());
    assert!(per_expert % 64 == 0 && n_cols % 64 == 0, "expert stride {per_expert} / row stride {n_cols}");
    assert!(values.len() % per_expert == 0);
    let n_experts = values.len() / per_expert;
    if rule.weighted() {
        assert_eq!(imw.len(), n_experts * n_cols, "importance matrix is {} x {}", n_experts, n_cols);
    }
    let global = global_scale(values);
    let n_blocks = values.len() / 64;
    let mut out = vec![0u8; n_blocks * 36];

    let nt = threads.max(1).min(n_blocks.max(1));
    let per_thread = n_blocks.div_ceil(nt);
    let mut stats_all = ExpertStats::default();
    let parts: Vec<ExpertStats> = std::thread::scope(|scope| {
        let mut handles = Vec::new();
        for (ti, chunk) in out.chunks_mut(per_thread * 36).enumerate() {
            let first_block = ti * per_thread;
            handles.push(scope.spawn(move || {
                quantize_blocks(values, first_block, chunk, per_expert, n_cols, imw, rule, global)
            }));
        }
        handles.into_iter().map(|h| h.join().expect("quantize thread")).collect()
    });
    for p in &parts {
        stats_all.absorb(p);
    }
    (out, global, stats_all)
}

/// One thread's slice of blocks. `first_block` is where `dst` starts in the tensor.
#[allow(clippy::too_many_arguments)]
fn quantize_blocks(
    values: &[f32],
    first_block: usize,
    dst: &mut [u8],
    per_expert: usize,
    n_cols: usize,
    imw: &[f32],
    rule: Rule,
    global: f32,
) -> ExpertStats {
    let mut st = ExpertStats::default();
    let mut w = [0.0f64; 16];
    for (bi, blk) in dst.chunks_exact_mut(36).enumerate() {
        let b = first_block + bi;
        let base = b * 64;
        let chunk = &values[base..base + 64];
        // sigma2 over the 64-value NVFP4 block, llama.cpp's `2 * sum(x^2) / super_block_size`
        let sigma2 = if rule.weighted() {
            2.0f32 * chunk.iter().map(|v| (*v as f64) * (*v as f64)).sum::<f64>() as f32 / 64.0
        } else {
            0.0
        };
        let expert = base / per_expert;
        let imw_row = if rule.weighted() { &imw[expert * n_cols..(expert + 1) * n_cols] } else { &[][..] };
        let mut scales = [0u32; 4];
        let mut nibbles = [0u32; 32];
        for (sb, sub) in chunk.chunks(16).enumerate() {
            let max_abs = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            let ceil_byte = encode_subblock_ceil((max_abs / 6.0) / global);
            let stored = if rule.weighted() {
                let col0 = (base + sb * 16) % n_cols;
                for (l, &v) in sub.iter().enumerate() {
                    w[l] = (imw_row[col0 + l] as f64) * ((sigma2 + v * v).sqrt() as f64);
                }
                encode_subblock_weighted(sub, &w[..sub.len()], global, ceil_byte, rule == Rule::ImatrixFourSix)
            } else {
                encode_subblock_mse(sub, global, ceil_byte).0
            };
            if rule.weighted() {
                let mse_byte = encode_subblock_mse(sub, global, ceil_byte).0;
                if mse_byte != stored {
                    st.scales_moved += 1;
                }
            }
            st.sub_blocks += 1;
            scales[sb] = stored;
            let dec = decode_ue4m3(stored) * global;
            let inv = 1.0 / dec;
            for (j, v) in sub.iter().enumerate() {
                let nib = e2m1_index(v.abs() * inv) as u32 | (((*v < 0.0) as u32) << 3);
                let byte_idx = sb * 16 + j;
                nibbles[byte_idx / 2] |= nib << (4 * (byte_idx % 2));
                // the report is read back off the written encoding, never off the search
                let d = if nib & 0x8 != 0 { -E2M1_GRID[(nib & 0x7) as usize] } else { E2M1_GRID[(nib & 0x7) as usize] } * dec;
                let err = (d - *v) as f64;
                st.sse += err * err;
                let col = (base + sb * 16 + j) % n_cols;
                let qw = if imw_row.is_empty() { 1.0f64 } else { imw_row[col] as f64 };
                st.wsse += qw * err * err;
                st.wsum += qw;
                st.n += 1;
                if (err.abs() as f32) > st.max_abs_err {
                    st.max_abs_err = err.abs() as f32;
                }
                if v.abs() > 6.0 * dec * (1.0 + 1e-6) {
                    st.clipped += 1;
                }
            }
        }
        for (i, s) in scales.iter().enumerate() {
            blk[i] = *s as u8;
        }
        for (i, nib) in nibbles.iter().enumerate() {
            blk[4 + i] = *nib as u8;
        }
    }
    st
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{quantize_nvfp4, ScalesMode};

    fn flat_imw(n_experts: usize, n_cols: usize) -> Vec<f32> {
        vec![1.0f32; n_experts * n_cols]
    }

    fn sample(n: usize) -> Vec<f32> {
        // a deterministic spread with outliers, the shape a real weight row has
        let mut s = 123456789u64;
        (0..n)
            .map(|i| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let u = ((s >> 33) as f64 / (1u64 << 31) as f64) - 0.5;
                let v = (u * 0.08) as f32;
                if i % 97 == 0 {
                    v * 12.0
                } else {
                    v
                }
            })
            .collect()
    }

    /// The whole control rests on this: rule (a) through the NEW code path must emit exactly
    /// what the conversion's own `quantize_nvfp4` emits, bytes and global scale.
    #[test]
    fn rule_mse_is_byte_identical_to_the_conversions_own_quantizer() {
        let v = sample(64 * 40);
        let (want, want_g, _, _) = quantize_nvfp4(&v, ScalesMode::Mse);
        for threads in [1usize, 3, 8] {
            let (got, got_g, st) = quantize_expert_tensor(&v, 64 * 40, 64, &[], Rule::Mse, threads);
            assert_eq!(got, want, "threads {threads}");
            assert_eq!(got_g.to_bits(), want_g.to_bits(), "threads {threads}");
            assert_eq!(st.n, v.len() as u64);
            assert_eq!(st.scales_moved, 0, "the mse rule never reports a moved scale");
        }
    }

    /// The thread split may not be visible in the output, ever.
    #[test]
    fn the_thread_count_does_not_change_one_byte() {
        let v = sample(64 * 37);
        let imw = flat_imw(1, 64);
        let (a, ga, sa) = quantize_expert_tensor(&v, 64 * 37, 64, &imw, Rule::Imatrix, 1);
        for threads in [2usize, 5, 16] {
            let (b, gb, sb) = quantize_expert_tensor(&v, 64 * 37, 64, &imw, Rule::Imatrix, threads);
            assert_eq!(a, b, "threads {threads}");
            assert_eq!(ga.to_bits(), gb.to_bits());
            assert_eq!((sa.n, sa.clipped, sa.sub_blocks), (sb.n, sb.clipped, sb.sub_blocks));
            assert!((sa.sse - sb.sse).abs() < 1e-9 * sa.sse.max(1e-30));
        }
    }

    /// A FLAT importance matrix is still not the unweighted rule — llama.cpp's
    /// `sqrt(sigma2 + x^2)` factor stays — but the global scale convention must not move, and
    /// the result must stay a valid encoding of the same geometry.
    #[test]
    fn the_global_scale_convention_is_the_containers_whatever_the_rule() {
        let v = sample(64 * 20);
        let (_, g_mse, _, _) = quantize_nvfp4(&v, ScalesMode::Mse);
        for rule in [Rule::Imatrix, Rule::ImatrixFourSix] {
            let (out, g, _) = quantize_expert_tensor(&v, 64 * 20, 64, &flat_imw(1, 64), rule, 4);
            assert_eq!(g.to_bits(), g_mse.to_bits(), "{rule:?}");
            assert_eq!(out.len(), 20 * 36);
        }
    }

    /// The point of the exercise: with a real importance matrix the weighted error must go
    /// DOWN against the unweighted rule, and it is allowed to cost plain MSE.
    #[test]
    fn the_weighted_rule_lowers_the_weighted_error_and_may_cost_plain_mse() {
        let n_cols = 64;
        let n_experts = 4;
        let per_expert = n_cols * 8;
        let v = sample(per_expert * n_experts);
        // a spiky importance profile: a few columns carry almost everything
        let mut imw = vec![0.01f32; n_experts * n_cols];
        for e in 0..n_experts {
            for j in 0..n_cols {
                if (j + e) % 11 == 0 {
                    imw[e * n_cols + j] = 300.0;
                }
            }
        }
        let (_, _, a) = quantize_expert_tensor(&v, per_expert, n_cols, &imw, Rule::Mse, 4);
        let (_, _, b) = quantize_expert_tensor(&v, per_expert, n_cols, &imw, Rule::Imatrix, 4);
        let (_, _, c) = quantize_expert_tensor(&v, per_expert, n_cols, &imw, Rule::ImatrixFourSix, 4);
        assert!(b.weighted_mse() < a.weighted_mse(), "{} !< {}", b.weighted_mse(), a.weighted_mse());
        assert!(c.weighted_mse() <= b.weighted_mse(), "{} !<= {}", c.weighted_mse(), b.weighted_mse());
        assert!(b.scales_moved > 0, "a spiky imatrix has to move some scale");
        // the unweighted mse rule is by construction the best UNWEIGHTED one of the three
        assert!(a.mse() <= b.mse() * 1.000001);
    }

    /// Every candidate the four-over-six rule adds is scored, so it can never be worse than
    /// the rule it extends on its own objective.
    #[test]
    fn four_over_six_only_ever_adds_candidates() {
        let n_cols = 128;
        let v = sample(n_cols * 16);
        let mut imw = vec![1.0f32; n_cols];
        for (j, w) in imw.iter_mut().enumerate() {
            *w = 1.0 + (j % 7) as f32;
        }
        let (_, _, b) = quantize_expert_tensor(&v, n_cols * 16, n_cols, &imw, Rule::Imatrix, 2);
        let (_, _, c) = quantize_expert_tensor(&v, n_cols * 16, n_cols, &imw, Rule::ImatrixFourSix, 2);
        assert!(c.weighted_mse() <= b.weighted_mse());
    }

    /// An all-zero sub-block encodes as scale byte 0 and zero nibbles, in every rule.
    #[test]
    fn an_all_zero_block_stays_all_zero() {
        let v = vec![0.0f32; 128];
        for rule in [Rule::Mse, Rule::Imatrix, Rule::ImatrixFourSix] {
            let (out, g, st) = quantize_expert_tensor(&v, 128, 64, &vec![1.0f32; 64], rule, 1);
            assert_eq!(out, vec![0u8; 72], "{rule:?}");
            assert_eq!(g, 1.0);
            assert_eq!(st.sse, 0.0);
            assert_eq!(st.clipped, 0);
        }
    }

    /// The expert a value belongs to decides which importance row weights it. Two experts with
    /// swapped profiles must not produce the same bytes as one shared profile would.
    #[test]
    fn each_expert_is_weighted_by_its_own_row() {
        let n_cols = 64;
        let per_expert = n_cols * 4;
        let v = sample(per_expert * 2);
        let mut a = vec![1.0f32; 2 * n_cols];
        let mut b = vec![1.0f32; 2 * n_cols];
        for j in 0..n_cols {
            a[j] = if j < 8 { 500.0 } else { 0.002 };
            a[n_cols + j] = if j >= 56 { 500.0 } else { 0.002 };
            b[j] = a[n_cols + j];
            b[n_cols + j] = a[j];
        }
        let (oa, _, _) = quantize_expert_tensor(&v, per_expert, n_cols, &a, Rule::Imatrix, 1);
        let (ob, _, _) = quantize_expert_tensor(&v, per_expert, n_cols, &b, Rule::Imatrix, 1);
        assert_ne!(oa, ob, "swapping the two experts' importance rows must change the bytes");
    }

    #[test]
    fn the_rule_names_round_trip() {
        for r in [Rule::Mse, Rule::Imatrix, Rule::ImatrixFourSix] {
            assert_eq!(Rule::parse(r.name()), Some(r));
        }
        assert_eq!(Rule::parse("nonsense"), None);
    }
}
