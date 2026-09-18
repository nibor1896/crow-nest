//! converter — crow-nest #5, stage 1: streaming RTN quantizer to the OWN container format.
//!
//! Spec section 1 (approved 2026-09-02): input = the original 131-shard safetensors
//! directory (or a single file), streamed shard by shard and tensor by tensor — never
//! the model in RAM; output = CNQ v1 container — `CNQ1` magic, streamed payload, JSON
//! index as TRAILER (the payload streams to disk without knowing the index size up
//! front; the old index-first layout needed the whole blob in RAM).
//!
//! Quantization: NVFP4 per ggml geometry — 64 values per block, 4 ue4m3 sub-block
//! scales (one per 16-wide k-block), 32 B packed E2M1 = 36 B per 64 values = 4.5 bpw —
//! plus ONE global f32 scale per tensor (two-level scaling). Calibration-free (RTN).
//!
//! BF16 keep-set (spec 1.2, approved): embeddings + lm_head, router GEMM,
//! shared_expert_gate, all norms — plus implementation rules flagged for review:
//! every 1-D tensor (biases, A_log, dt_bias, gates ride along; sub-0.1 % of bytes,
//! and 1-D recurrence params must not go through a GEMM quant path) and any tensor
//! whose length is not a multiple of 64 stays BF16.
//!
//! Section tags (decision 2026-09-02): `ple` (ngram_embedding — NVFP4, block
//! exchangeable to FP8), `vit` (model.visual — carried, optional to load), `mtp`
//! (carried, optional to load), `text` (default).
//!
//! Verification sidecar: `<out>.cnq.sidecar.jsonl`, one JSON line per tensor with
//! max/mean error against the sub-block scales — computed during quantization by
//! dequantizing in place. Gate 0 of the measurement ladder. Exit code 1 on any bound
//! violation.
//!
//! Scale policies (`--scales`, default `ceil`):
//!   ceil — smallest ue4m3 ladder step >= sub-block max. stored >= raw ALWAYS:
//!          values never clamp, per-element error is bounded by one E2M1 half-gap,
//!          max_rel <= 1.0 (gate: exit 1 on any violation).
//!   mse  — per 16-wide sub-block, pick the ue4m3 ladder step minimizing the sum
//!          of squared errors (SSE) of the 16 e2m1-quantized values (round-to-
//!          nearest, saturation at ±6·scale). ANALYTIC PRE-SELECTION + LOCAL
//!          REFINEMENT, no global scan: moment-match seed -> two fixed-shape
//!          least-squares refinements -> explicit SSE check of only the 2-3
//!          bracketing ladder steps (see encode_subblock_mse). This DELIBERATELY
//!          allows clipping: elements above 6·scale are cut, so the old
//!          "max_rel <= 1.0" guarantee is VOID under `mse` — no fixed
//!          relative bound holds anymore. The rel-bound exit gate is disabled in
//!          this mode; quality is REPORTED instead: every nvfp4 sidecar line gains
//!          `mse` (written encoding), `mse_ceil` (same weights re-encoded with
//!          ceiling scales), `mse_ratio` and `max_abs_clipped` (COUNT of clipped
//!          elements, |v| > 6·scale — the name is per report spec, it is a count),
//!          plus one `record: "section_summary"` line per nvfp4 section
//!          (text/vit/ple/mtp) with the aggregated numbers.
//!          The GLOBAL tensor scale stays max-based in BOTH modes, so ladder
//!          utilization is unchanged; only the SUBBLOCK scale choice differs.
//!          Cost: ~2-3x the per-value quantization work of `ceil` (a handful of
//!          SSE evaluations per sub-block instead of one scale choice).
//!
//! Container layout v1:
//!   [0..4)    magic "CNQ1"
//!   [4..12)   reserved (zeros)
//!   [12..)    payload blob (streamed)
//!   [end-8-index_len..end-8)  index JSON (UTF-8)
//!   [end-8..end)              u64 LE index_len
//! Tensor offsets in the index are relative to blob start (12).
//!
//! Subcommand `requant-check` (#76, 2026-09-18) — ADDITIVE and read-only. It takes the BF16
//! originals of the dense text path, fetched back by `tools/fetch-dense-originals.py`, runs
//! them through the SAME `quantize_nvfp4` above and compares the blocks and the global scale
//! with what a container already stores. The conversion path below is untouched by it: a
//! run without the word `requant-check` parses, reads and writes exactly what it did before.
//! See `src/requant_check.rs` and `docs/dense-originals.md`.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};

mod requant_check;

const E2M1_GRID: [f32; 8] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0];
const E2M1_MID: [f32; 7] = [0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0];
const UE4M3_MAX: f32 = 448.0;
const MAGIC: &[u8; 4] = b"CNQ1";

fn decode_e2m1(nibble: u32) -> f32 {
    let v = E2M1_GRID[(nibble & 0x7) as usize];
    if nibble & 0x8 != 0 { -v } else { v }
}

/// Branchless nearest-E2M1-level index for a magnitude m (0..=7): the count of
/// midpoints <= m. Unrolled compare chain — no data-dependent branches, lets the
/// sub-block loops vectorize (the mse hot path runs this several times per value).
#[inline]
fn e2m1_index(m: f32) -> usize {
    (m >= E2M1_MID[0]) as usize
        + (m >= E2M1_MID[1]) as usize
        + (m >= E2M1_MID[2]) as usize
        + (m >= E2M1_MID[3]) as usize
        + (m >= E2M1_MID[4]) as usize
        + (m >= E2M1_MID[5]) as usize
        + (m >= E2M1_MID[6]) as usize
}

/// Dequantized value of `v` at scale `s` (`inv` = 1/s precomputed once per scale)
/// — same round-to-nearest-midpoint rounding as encode/decode_e2m1, but with
/// reciprocal-multiply instead of divide. ALL quantization paths (MSE search,
/// packing) share this one arithmetic so reported stats describe exactly the
/// written bytes; at exact midpoints results are identical, elsewhere they match
/// up to ulps, which the 1e-6 report tolerances absorb.
#[inline]
fn quant_dequant(v: f32, s: f32, inv: f32) -> f32 {
    let idx = e2m1_index(v.abs() * inv);
    (if v < 0.0 { -E2M1_GRID[idx] } else { E2M1_GRID[idx] }) * s
}

/// Smallest ue4m3-representable value >= v (CEILING scale policy).
///
/// Round-to-nearest let small scales land up to 29 % BELOW their value in the linear
/// subnormal range — the sub-block's largest value then clamps at E2M1's 6·scale and
/// blows the error bound (found on real ViT weights, 2026-09-02: 25 violations,
/// max rel err 2.9). With ceiling scales, stored >= raw ALWAYS holds: values never
/// clamp and the per-element error is bounded by one E2M1 half-gap — violations
/// become structurally impossible instead of rare.
fn encode_ue4m3_ceil(v: f32) -> u32 {
    if v <= 0.0 {
        return 0;
    }
    let mut e = ((v.log2().floor() as i32) + 7).clamp(0, 15) as u32;
    loop {
        if e == 0 {
            // subnormal ladder: m * 2^-9, m = 0..7
            let m = (v * 512.0).ceil() as u32;
            if m <= 7 {
                return m;
            }
            e = 1; // carry into the normal range
            continue;
        }
        let unit = 2.0f32.powi(e as i32 - 7);
        let m = ((v / unit - 1.0) * 8.0).ceil() as i32;
        if m < 0 {
            return e << 3; // v just below this exponent's minimum — unit itself suffices
        }
        if m <= 7 {
            return (e << 3) | (m as u32);
        }
        if e >= 15 {
            return (15 << 3) | 7; // format maximum
        }
        e += 1; // carry: v above this exponent's max — next exponent, m = 0
    }
}

fn decode_ue4m3(byte: u32) -> f32 {
    let e = (byte >> 3) & 0xF;
    let m = byte & 0x7;
    if e == 0 {
        (m as f32) * 2.0f32.powi(-9)
    } else {
        (1.0 + (m as f32) / 8.0) * 2.0f32.powi(e as i32 - 7)
    }
}

// ---------------- manifest ----------------

struct TensorEntry {
    name: String,
    shape: Vec<usize>,
    n_values: usize,
    dtype_out: &'static str, // "nvfp4" | "bf16"
    section: &'static str,   // "text" | "ple" | "vit" | "mtp"
    shard: String,
    data_begin: u64, // byte offset inside the shard file (incl. header)
    data_end: u64,
}

fn section_of(name: &str) -> &'static str {
    if name.contains("ngram_embedding") {
        "ple"
    } else if name.contains(".visual.") || name.starts_with("model.visual") {
        "vit"
    } else if name.contains(".mtp.") || name.starts_with("mtp") {
        "mtp"
    } else {
        "text"
    }
}

/// BF16 keep-set per spec 1.2 — precision where routing and norm stability live.
fn keep_bf16(name: &str, shape: &[usize], n: usize) -> bool {
    if shape.len() == 1 {
        return true; // 1-D: norms/biases/A_log/dt_bias/gates — flagged in the ticket comment
    }
    if name.contains("norm")
        || name.contains("embed_tokens")
        || name.contains("lm_head")
        || name.contains("mlp.gate.weight") // router GEMM
        || name.contains("shared_expert_gate")
    {
        return true;
    }
    // Amendment 2026-09-03 (robin GO, mix of options 1+2 on the #11 gate
    // finding): FP4 compounding through 48 layers breaks argmax parity —
    // keep the residual-path HC mix projections and attention q/k in BF16.
    if name.contains("input_mix_weight_down")
        || name.contains("input_mix_weight_up")
        || name.contains("self_attn.q_proj.weight")
        || name.contains("self_attn.k_proj.weight")
    {
        return true;
    }
    n % 64 != 0 || n < 64
}

fn read_safetensors_header(path: &std::path::Path) -> std::io::Result<(serde_json::Value, u64)> {
    let mut f = std::fs::File::open(path)?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let header_len = u64::from_le_bytes(len_buf);
    let mut header_buf = vec![0u8; header_len as usize];
    f.read_exact(&mut header_buf)?;
    let header: serde_json::Value = serde_json::from_slice(&header_buf)?;
    Ok((header, 8 + header_len))
}

fn elem_size_of(dt: &str) -> Option<usize> {
    match dt {
        "F32" => Some(4),
        "BF16" | "F16" => Some(2),
        "I64" => Some(8),
        _ => None,
    }
}

fn bytes_to_f32(raw: &[u8], dt: &str) -> Vec<f32> {
    match dt {
        "F32" => raw
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        "BF16" => raw
            .chunks_exact(2)
            .map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16))
            .collect(),
        "F16" => raw
            .chunks_exact(2)
            .map(|c| half_bits_to_f32(u16::from_le_bytes([c[0], c[1]])))
            .collect(),
        _ => unreachable!(),
    }
}

fn half_bits_to_f32(bits: u16) -> f32 {
    let s = ((bits >> 15) as u32) << 31;
    let e = ((bits >> 10) & 0x1F) as u32;
    let m = (bits & 0x3FF) as u32;
    if e == 0 {
        f32::from_bits(s | (m << 13))
    } else if e == 31 {
        f32::from_bits(s | 0x7F80_0000 | (m << 13))
    } else {
        f32::from_bits(s | ((e + 112) << 23) | (m << 13))
    }
}

#[derive(Clone)]
struct QuantStats {
    max_abs_err: f32,
    sum_abs_err: f64,
    sum_sq_err: f64,
    max_rel_err: f32,
    violations: u64,
    clipped: u64,
}

impl QuantStats {
    fn zero() -> Self {
        QuantStats {
            max_abs_err: 0.0,
            sum_abs_err: 0.0,
            sum_sq_err: 0.0,
            max_rel_err: 0.0,
            violations: 0,
            clipped: 0,
        }
    }

    fn absorb(&mut self, o: &QuantStats) {
        self.sum_abs_err += o.sum_abs_err;
        self.sum_sq_err += o.sum_sq_err;
        if o.max_abs_err > self.max_abs_err {
            self.max_abs_err = o.max_abs_err;
        }
        if o.max_rel_err > self.max_rel_err {
            self.max_rel_err = o.max_rel_err;
        }
        self.violations += o.violations;
        self.clipped += o.clipped;
    }
}

/// Per-section (text/vit/ple/mtp) aggregate for the MSE verification report.
struct SectAgg {
    tensors: u64,
    n: u64,
    sse_new: f64,
    sse_ceil: f64,
    clipped: u64,
}

/// Sub-block scale policy (see `--scales` in the header docs).
#[derive(Clone, Copy, PartialEq, Eq)]
enum ScalesMode {
    /// smallest ue4m3 step >= block max — stored >= raw ALWAYS, no clipping
    Ceil,
    /// per-sub-block SSE-minimizing step (analytic pre-selection + local
    /// refinement) — clipping allowed, max_rel bound void
    Mse,
}

/// Ceiling byte for a sub-block in DIVIDED units: `encode_ue4m3_ceil` plus the
/// float-division guard (bump one ladder step while the decoded value is still
/// below raw; byte +1 walks representables monotonically, 0x7F is the format max).
fn encode_subblock_ceil(raw: f32) -> u32 {
    let mut stored = encode_ue4m3_ceil(raw);
    while stored < 0x7F && decode_ue4m3(stored) < raw {
        stored += 1;
    }
    stored
}

/// SSE of one 16-wide sub-block quantized at decode scale `s` — uses the SAME
/// rounding as block packing (`quant_dequant`), so candidate scores describe
/// exactly the bytes that would be written.
fn subblock_sse(sub: &[f32], s: f32) -> f64 {
    let inv = 1.0 / s;
    let mut sse = 0.0f64;
    for &v in sub {
        let e = quant_dequant(v, s, inv) - v;
        sse += (e as f64) * (e as f64);
    }
    sse
}

/// MSE sub-block scale (`--scales mse`): analytic pre-selection + local ladder
/// refinement. NO global scan over all 127 ue4m3 steps — that would multiply
/// conversion time roughly 10x and blow the Vollauf budget:
///   1. seed s0 = mean|w| / mean|E2M1 grid| (moment match; grid mean over all 8
///      levels = 18/8 = 2.25),
///   2. two fixed-shape refinements (generalized Lloyd step): quantize at the
///      current scale, then s' = Σ(w·g)/Σ(g²) is the exact least-squares scale
///      FOR THAT e2m1 shape — saturated elements sit at g = ±6, so clipping is
///      priced in correctly,
///   3. explicit SSE evaluation (packing arithmetic) of the two ue4m3 ladder
///      steps bracketing the analytic optimum, plus the policy-ceiling step —
///      the ceiling candidate keeps MSE_neu <= MSE_ceil per sub-block BY
///      CONSTRUCTION; minimum wins (ties keep the clamp-free ceiling scale).
/// The ceiling step's SSE is computed once here and also serves as the report
/// reference (returned alongside), so no extra pass is spent on it.
/// Clipping stays deliberate: elements above 6·s are cut, so the old
/// "max_rel <= 1.0" guarantee does NOT hold in this mode — quality is reported
/// via the sidecar MSE fields instead.
/// Returns (chosen ladder byte, its SSE, the ceiling scale's SSE).
fn encode_subblock_mse(sub: &[f32], global: f32, ceil_byte: u32) -> (u32, f64, f64) {
    let max_abs = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    if max_abs <= 0.0 {
        let sse = 0.0f64; // all-zero sub-block: scale 0, nibbles 0, error 0
        return (0, sse, sse);
    }
    let mean_abs = sub.iter().fold(0.0f32, |m, v| m + v.abs()) / sub.len() as f32;
    let mut s = mean_abs / 2.25; // seed, real units
    for _ in 0..2 {
        let inv = 1.0 / s;
        let (mut num, mut den) = (0.0f64, 0.0f64);
        for &v in sub {
            let g = quant_dequant(v, 1.0, inv) as f64; // unit-scale e2m1 shape
            num += v as f64 * g;
            den += g * g;
        }
        if den > 0.0 {
            let s_next = (num / den) as f32;
            if s_next.is_finite() && s_next > 0.0 {
                s = s_next;
            } else {
                break;
            }
        } else {
            break; // shape quantized to all-zeros — current scale already too large
        }
    }
    // bracket the analytic optimum with its two adjacent ue4m3 ladder steps
    // (divided units; the ladder is strictly monotonic in the byte value)
    let t = s / global;
    let hi = encode_subblock_ceil(t); // smallest ladder step >= t
    let lo = if hi == 0 {
        0
    } else if decode_ue4m3(hi) == t {
        hi
    } else {
        hi - 1
    };
    let mut cands = [lo, hi];
    cands.sort_unstable();
    if cands[1] == cands[0] {
        cands[1] = 0; // dedup: 0 is skipped as a candidate anyway
    }
    let ceil_sse = subblock_sse(sub, decode_ue4m3(ceil_byte) * global);
    let mut best = ceil_byte;
    let mut best_sse = ceil_sse;
    for b in cands {
        if b == 0 || b == best {
            continue;
        }
        let sse = subblock_sse(sub, decode_ue4m3(b) * global);
        if sse < best_sse {
            best_sse = sse;
            best = b;
        }
    }
    (best, best_sse, ceil_sse)
}

/// NVFP4-RTN with in-place dequant statistics (gate 0 of the measurement ladder).
/// Returns (packed blocks, global scale, stats of the WRITTEN encoding, ceiling-
/// reference SSE on the SAME weights — identical to the written SSE in Ceil mode).
/// The GLOBAL scale stays max-based in both modes: ladder utilization unchanged;
/// only the sub-block scale choice differs.
fn quantize_nvfp4(values: &[f32], mode: ScalesMode) -> (Vec<u8>, f32, QuantStats, f64) {
    assert!(values.len() % 64 == 0);
    let n_blocks = values.len() / 64;
    let mut raw_scales = vec![0.0f32; n_blocks * 4];
    for (b, chunk) in values.chunks(64).enumerate() {
        for (sb, sub) in chunk.chunks(16).enumerate() {
            let max = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
            raw_scales[b * 4 + sb] = max / 6.0;
        }
    }
    let max_scale = raw_scales.iter().fold(0.0f32, |m, s| m.max(*s));
    let global = if max_scale > 0.0 { max_scale / UE4M3_MAX } else { 1.0 };

    let mut out = Vec::with_capacity(n_blocks * 36);
    let mut stats = QuantStats::zero();
    let mut sse_ceil = 0.0f64;
    for (b, chunk) in values.chunks(64).enumerate() {
        let mut scales = [0u32; 4];
        let mut nibbles = [0u32; 32];
        for (sb, sub) in chunk.chunks(16).enumerate() {
            let raw = raw_scales[b * 4 + sb] / global;
            let ceil_byte = encode_subblock_ceil(raw);
            let stored = match mode {
                ScalesMode::Ceil => ceil_byte,
                ScalesMode::Mse => {
                    let (byte, _sse, ceil_sse) = encode_subblock_mse(sub, global, ceil_byte);
                    sse_ceil += ceil_sse;
                    byte
                }
            };
            scales[sb] = stored;
            let dec = decode_ue4m3(stored) * global;
            let inv = 1.0 / dec;
            for (j, v) in sub.iter().enumerate() {
                let nib = e2m1_index(v.abs() * inv) as u32 | (((*v < 0.0) as u32) << 3);
                let byte_idx = sb * 16 + j;
                nibbles[byte_idx / 2] |= nib << (4 * (byte_idx % 2));
            }
            // gate 0 (written encoding): dequantize in place and measure against
            // the sub-block scale
            let mut sb_stats = QuantStats::zero();
            for (j, v) in sub.iter().enumerate() {
                let byte_idx = sb * 16 + j;
                let nib = (nibbles[byte_idx / 2] >> (4 * (byte_idx % 2))) & 0xF;
                let d = decode_e2m1(nib) * dec;
                let err = (d - *v).abs();
                sb_stats.sum_abs_err += err as f64;
                sb_stats.sum_sq_err += (err as f64) * (err as f64);
                if err > sb_stats.max_abs_err {
                    sb_stats.max_abs_err = err;
                }
                // clipping: |v| beyond e2m1's ±6·dec saturates (1e-6 relative
                // slack absorbs the ulp lost round-tripping raw/global ->
                // decode*global — without it, exactly-fitting ceiling scales
                // flagged their block max as clipped)
                if v.abs() > 6.0 * dec * (1.0 + 1e-6) {
                    sb_stats.clipped += 1;
                }
                if dec > 0.0 {
                    let rel = err / dec;
                    if rel > sb_stats.max_rel_err {
                        sb_stats.max_rel_err = rel;
                    }
                    if rel > 1.08 {
                        sb_stats.violations += 1;
                    }
                }
            }
            stats.absorb(&sb_stats);
        }
        for sb in 0..4 {
            out.push(scales[sb] as u8);
        }
        for nib in nibbles {
            out.push(nib as u8);
        }
    }
    if mode == ScalesMode::Ceil {
        sse_ceil = stats.sum_sq_err; // the written encoding IS the ceiling encoding
    }
    (out, global, stats, sse_ceil)
}

const HELP: &str = "usage: converter [--scales ceil|mse] <model-dir | file.safetensors> <out.cnq>\n  --scales ceil  ceiling sub-block scales: stored >= raw always, max_rel <= 1.0 (default)\n  --scales mse   per-sub-block SSE-minimizing scales: clipping allowed, quality via MSE report\n       converter [--scales ceil|mse] requant-check <dense.safetensors> <container.cnq>\n  re-quantizes fetched originals and compares them with the container's own bytes (#76)";

fn main() {
    // #76: the additive read-only subcommand is taken off the front before the conversion
    // path sees anything. Without the word `requant-check` nothing below this block changes.
    let all: Vec<String> = std::env::args().skip(1).collect();
    if let Some(at) = all.iter().position(|a| a == "requant-check") {
        let mut mode = ScalesMode::Ceil;
        let mut explicit = false;
        let mut before = all[..at].iter();
        while let Some(a) = before.next() {
            match a.as_str() {
                "--scales" => match before.next().map(|s| s.as_str()) {
                    Some("ceil") => {
                        mode = ScalesMode::Ceil;
                        explicit = true;
                    }
                    Some("mse") => {
                        mode = ScalesMode::Mse;
                        explicit = true;
                    }
                    other => {
                        eprintln!("--scales needs `ceil` or `mse`, got {other:?}\n{HELP}");
                        std::process::exit(2);
                    }
                },
                other => {
                    eprintln!("unexpected argument {other} before requant-check\n{}", requant_check::HELP);
                    std::process::exit(2);
                }
            }
        }
        std::process::exit(requant_check::run(&all[at + 1..], mode, explicit));
    }

    let mut positional: Vec<String> = Vec::new();
    let mut mode = ScalesMode::Ceil;
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--scales" => match argv.next().as_deref() {
                Some("ceil") => mode = ScalesMode::Ceil,
                Some("mse") => mode = ScalesMode::Mse,
                other => {
                    eprintln!("--scales needs `ceil` or `mse`, got {other:?}\n{HELP}");
                    std::process::exit(2);
                }
            },
            a if a.starts_with("--") => {
                eprintln!("unknown flag {a}\n{HELP}");
                std::process::exit(2);
            }
            a => positional.push(a.to_string()),
        }
    }
    if positional.len() != 2 {
        eprintln!("{HELP}");
        std::process::exit(2);
    }
    let input = std::path::PathBuf::from(&positional[0]);
    let out_path = std::path::PathBuf::from(&positional[1]);
    let scales_mode_str = if mode == ScalesMode::Mse { "mse" } else { "ceil" };
    let single_file = input.is_file();
    let t_start = std::time::Instant::now();

    // ---- manifest: scan headers only (fast), collect every tensor's location ----
    let mut tensors: Vec<TensorEntry> = Vec::new();
    let mut weight_map: Option<BTreeMap<String, String>> = None;
    let shard_files: Vec<std::path::PathBuf> = if single_file {
        vec![input.clone()]
    } else {
        let index: serde_json::Value = serde_json::from_slice(
            &std::fs::read(input.join("model.safetensors.index.json")).expect("read model index"),
        )
        .expect("parse model index");
        let mut wm = BTreeMap::new();
        for (name, shard) in index["weight_map"].as_object().expect("weight_map") {
            wm.insert(name.clone(), shard.as_str().unwrap().to_string());
        }
        weight_map = Some(wm);
        let mut files: Vec<String> = weight_map
            .as_ref()
            .unwrap()
            .values()
            .cloned()
            .collect();
        files.sort();
        files.dedup();
        files.iter().map(|f| input.join(f)).collect()
    };

    for path in &shard_files {
        let shard_name = path
            .file_name()
            .unwrap()
            .to_str()
            .unwrap()
            .to_string();
        let (header, data_start) = read_safetensors_header(path).expect("shard header");
        if let Some(entries) = header.as_object() {
            for (name, info) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dt = info["dtype"].as_str().unwrap_or("F32");
                let Some(es) = elem_size_of(dt) else {
                    eprintln!("skip {name}: unsupported dtype {dt}");
                    continue;
                };
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                    .unwrap_or_default();
                let n: usize = shape.iter().product();
                if n == 0 {
                    continue;
                }
                let begin = info["data_offsets"][0].as_u64().unwrap();
                let end = info["data_offsets"][1].as_u64().unwrap();
                assert_eq!(end - begin, (n * es) as u64, "{name}: length mismatch");
                // I64: integer metadata (PLE index tables — layer_multipliers,
                // ngram_heads_offsets, ngram_heads_vocab_sizes) — raw carry, never quantize
                let dtype_out = if dt == "I64" || keep_bf16(name, &shape, n) { "bf16" } else { "nvfp4" };
                tensors.push(TensorEntry {
                    name: name.clone(),
                    shape,
                    n_values: n,
                    dtype_out,
                    section: section_of(name),
                    shard: shard_name.clone(),
                    data_begin: data_start + begin,
                    data_end: data_start + end,
                });
            }
        }
    }
    tensors.sort_by(|a, b| a.shard.cmp(&b.shard).then(a.data_begin.cmp(&b.data_begin)));
    eprintln!(
        "manifest: {} tensors from {} shard file(s), {:.1} GB to read",
        tensors.len(),
        shard_files.len(),
        tensors.iter().map(|t| (t.data_end - t.data_begin) as u64).sum::<u64>() as f64 / 1e9
    );

    // ---- write container: magic, streamed blob, index trailer ----
    let mut out = std::fs::File::create(&out_path).expect("create output");
    out.write_all(MAGIC).expect("magic");
    out.write_all(&0u64.to_le_bytes()).expect("reserved");
    let blob_start: u64 = 12;

    let mut index_tensors: Vec<serde_json::Value> = Vec::new();
    let mut sidecar = std::io::BufWriter::new(
        std::fs::File::create(out_path.with_extension("cnq.sidecar.jsonl")).expect("sidecar"),
    );
    let mut blob_len: u64 = 0;
    let mut nvfp4_count = 0usize;
    let mut bf16_count = 0usize;
    let mut total_violations: u64 = 0;
    let mut section_agg: BTreeMap<&'static str, SectAgg> = BTreeMap::new();

    for (i, t) in tensors.iter().enumerate() {
        let shard_path = if single_file {
            input.clone()
        } else {
            input.join(&t.shard)
        };
        let mut f = std::fs::File::open(&shard_path).expect("open shard");
        f.seek(SeekFrom::Start(t.data_begin)).expect("seek tensor");
        let mut raw = vec![0u8; (t.data_end - t.data_begin) as usize];
        f.read_exact(&mut raw).expect("read tensor");
        drop(f);

        let is_i64 = raw.len() == t.n_values * 8;
        let dt_in = if is_i64 { "I64" } else if raw.len() == t.n_values * 4 { "F32" } else { "BF16" };
        let mut entry_json = serde_json::json!({
            "name": t.name, "shape": t.shape, "section": t.section,
            "n_values": t.n_values, "offset": blob_len,
        });

        if t.dtype_out == "nvfp4" {
            let values = bytes_to_f32(&raw, dt_in);
            let (blocks, global, stats, sse_ceil) = quantize_nvfp4(&values, mode);
            total_violations += stats.violations;
            let mse = stats.sum_sq_err / t.n_values as f64;
            let mse_ceil = sse_ceil / t.n_values as f64;
            let mse_ratio = if mse_ceil > 0.0 { mse / mse_ceil } else { 1.0 };
            let agg = section_agg.entry(t.section).or_insert(SectAgg {
                tensors: 0,
                n: 0,
                sse_new: 0.0,
                sse_ceil: 0.0,
                clipped: 0,
            });
            agg.tensors += 1;
            agg.n += t.n_values as u64;
            agg.sse_new += stats.sum_sq_err;
            agg.sse_ceil += sse_ceil;
            agg.clipped += stats.clipped;
            out.write_all(&blocks).expect("write blocks");
            entry_json["dtype"] = serde_json::Value::from("nvfp4");
            entry_json["global_scale"] = serde_json::Value::from(global);
            entry_json["len"] = serde_json::Value::from(blocks.len() as u64);
            blob_len += blocks.len() as u64;
            nvfp4_count += 1;
            serde_json::to_writer(
                &mut sidecar,
                &serde_json::json!({
                    "name": t.name, "section": t.section, "dtype": "nvfp4",
                    "n": t.n_values, "global_scale": global,
                    "max_abs_err": stats.max_abs_err,
                    "mean_abs_err": stats.sum_abs_err / t.n_values as f64,
                    "max_rel_err": stats.max_rel_err,
                    "violations": stats.violations,
                    "scales_mode": scales_mode_str,
                    "mse": mse,
                    "mse_ceil": mse_ceil,
                    "mse_ratio": mse_ratio,
                    "max_abs_clipped": stats.clipped,
                }),
            )
            .expect("sidecar line");
            writeln!(sidecar).expect("sidecar newline");
        } else {
            out.write_all(&raw).expect("write bf16");
            entry_json["dtype"] = serde_json::Value::from(if is_i64 { "i64" } else { "bf16" });
            entry_json["len"] = serde_json::Value::from(raw.len() as u64);
            blob_len += raw.len() as u64;
            bf16_count += 1;
            serde_json::to_writer(
                &mut sidecar,
                &serde_json::json!({
                    "name": t.name, "section": t.section, "dtype": "bf16", "n": t.n_values,
                }),
            )
            .expect("sidecar line");
            writeln!(sidecar).expect("sidecar newline");
        }
        index_tensors.push(entry_json);
        if i % 100 == 0 {
            eprintln!(
                "[{:>5}/{}] {} ({}) — {:.2} GB written, {:.0} s",
                i + 1,
                tensors.len(),
                t.name,
                t.section,
                blob_len as f64 / 1e9,
                t_start.elapsed().as_secs_f64()
            );
        }
    }
    // per-section verification report (nvfp4 only): aggregate MSE of the written
    // encoding vs the ceiling-scale reference on the SAME weights + clipped counts
    for (section, a) in &section_agg {
        let mse = a.sse_new / a.n as f64;
        let mse_ceil = a.sse_ceil / a.n as f64;
        let mse_ratio = if mse_ceil > 0.0 { mse / mse_ceil } else { 1.0 };
        println!(
            "section {section}: {} nvfp4 tensors — MSE {mse:.4e} vs ceil {mse_ceil:.4e} (ratio {mse_ratio:.4}), clipped {}",
            a.tensors, a.clipped
        );
        serde_json::to_writer(
            &mut sidecar,
            &serde_json::json!({
                "record": "section_summary", "section": section, "dtype": "nvfp4",
                "scales_mode": scales_mode_str,
                "tensors": a.tensors, "n": a.n,
                "mse": mse, "mse_ceil": mse_ceil, "mse_ratio": mse_ratio,
                "max_abs_clipped": a.clipped,
            }),
        )
        .expect("sidecar summary line");
        writeln!(sidecar).expect("sidecar summary newline");
    }
    sidecar.flush().ok();

    // coverage check in dir mode: every tensor the model index knows must be in the output
    if let Some(wm) = &weight_map {
        let mut missing = 0usize;
        for name in wm.keys() {
            if !tensors.iter().any(|t| &t.name == name) {
                eprintln!("MISSING from output: {name}");
                missing += 1;
            }
        }
        if missing > 0 {
            eprintln!("coverage check FAILED: {missing} tensors missing");
            std::process::exit(3);
        }
        eprintln!("coverage check: all {} weight_map tensors present", wm.len());
    }

    let index = serde_json::json!({
        "format": "crow-nest-quant", "version": 1,
        "block_geometry": { "values": 64, "sub_block": 16, "bytes_per_block": 36, "bpw": 4.5 },
        "sections": {
            "ple": { "format": "nvfp4", "exchangeable_to": "fp8" },
            "vit": { "optional_to_load": true },
            "mtp": { "optional_to_load": true }
        },
        "blob_offset": blob_start,
        "tensors": index_tensors,
    });
    let index_json = serde_json::to_vec_pretty(&index).expect("index json");
    out.write_all(&index_json).expect("index");
    out.write_all(&(index_json.len() as u64).to_le_bytes()).expect("index len");
    out.flush().ok();

    println!(
        "wrote {}: {} tensors ({} nvfp4, {} bf16-keep), payload {:.2} GB, scales {scales_mode_str}, \
violations {total_violations}, elapsed {:.0} s",
        out_path.display(),
        tensors.len(),
        nvfp4_count,
        bf16_count,
        blob_len as f64 / 1e9,
        t_start.elapsed().as_secs_f64()
    );
    if total_violations > 0 {
        if mode == ScalesMode::Mse {
            // the old per-element relative bound is void BY DESIGN here (clipping
            // allowed); quality is carried by the MSE fields of the report above
            eprintln!(
                "NOTE: {total_violations} elements exceed the old max-rel bound — \
expected under --scales mse, see the MSE report"
            );
        } else {
            eprintln!("WARNING: {total_violations} bound violations — check the sidecar");
            std::process::exit(1);
        }
    }
}

// ---------------- tests ----------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn e2m1_grid_roundtrip() {
        // decode and quant_dequant must reproduce the grid exactly (s = 1, inv = 1)
        for nib in 0u32..16 {
            let v = decode_e2m1(nib);
            assert_eq!(quant_dequant(v, 1.0, 1.0), if v == 0.0 { 0.0 } else { v });
        }
    }

    #[test]
    fn ue4m3_ladder_is_strictly_monotonic() {
        let mut prev = 0.0f32;
        for byte in 1u32..=0x7F {
            let v = decode_ue4m3(byte);
            assert!(v > prev && v.is_finite(), "ladder broken at byte {byte}");
            prev = v;
        }
    }

    #[test]
    fn ceiling_scale_never_clips() {
        let sub: Vec<f32> = (0..16).map(|i| ((i as f32) * 37.7 - 300.0) * 1e-3).collect();
        let max = sub.iter().fold(0.0f32, |m, v| m.max(v.abs()));
        let b = encode_subblock_ceil(max / 6.0);
        let s = decode_ue4m3(b);
        assert!(s >= max / 6.0);
        // no element may exceed the representable radius 6*s (tolerance absorbs ulps)
        for &v in &sub {
            assert!(v.abs() <= 6.0 * s * (1.0 + 1e-6));
        }
    }

    #[test]
    fn mse_hits_exact_scale_on_uniform_block() {
        // 16 equal magnitudes: SSE 0 is achievable at s = 1.0 (a grid point);
        // the analytic refinement must land on the ladder step that reaches it.
        let sub = [1.0f32; 16];
        let global = 1.0;
        let ceil_b = encode_subblock_ceil(1.0 / 6.0);
        let (b, sse, _) = encode_subblock_mse(&sub, global, ceil_b);
        assert_eq!(subblock_sse(&sub, decode_ue4m3(b) * global), 0.0);
        assert_eq!(sse, 0.0, "uniform block must reach zero SSE");
    }

    #[test]
    fn mse_beats_ceil_on_moderate_outlier() {
        // 15 well-scaled values + a moderately large outlier: the ceiling scale
        // inflates everyone's rounding error; MSE accepts clipping the outlier
        // and must come out strictly ahead.
        let mut sub = [1.0f32; 16];
        sub[15] = 8.0;
        let global = 0.125;
        let ceil_b = encode_subblock_ceil(8.0 / 6.0 / global);
        let (b, sse, ceil_sse) = encode_subblock_mse(&sub, global, ceil_b);
        assert!(
            sse < ceil_sse,
            "mse sse {sse} must strictly beat ceil sse {ceil_sse}"
        );
        assert_eq!(subblock_sse(&sub, decode_ue4m3(b) * global), sse);
    }

    #[test]
    fn packed_block_roundtrip_both_modes() {
        // 64 deterministic pseudo-random values through the real quantize path,
        // decoded back per CONTAINER layout (engine/src/cnq.rs dequant_block):
        // 4 B ue4m3 scales (byte i = sub-block i) THEN 32 B LSB-first packed E2M1
        // (element j at bit 4j); dequant = e2m1 * ue4m3 * global.
        let mut vals: Vec<f32> = Vec::with_capacity(64);
        let mut x: u32 = 0x1234_5678;
        for i in 0..64 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = ((x >> 8) % 1000) as f32 / 1000.0;
            vals.push((m - 0.5) * if i % 9 == 0 { 30.0 } else { 0.8 });
        }
        for mode in [ScalesMode::Ceil, ScalesMode::Mse] {
            let (blocks, global, st, sse_ceil) = quantize_nvfp4(&vals, mode);
            assert_eq!(blocks.len(), 36);
            for sb in 0..4usize {
                let s = decode_ue4m3(blocks[sb] as u32) * global;
                for j in 0..16usize {
                    let byte_idx = sb * 16 + j;
                    let byte = blocks[4 + byte_idx / 2] as u32;
                    let nib = (byte >> (4 * (byte_idx % 2))) & 0xF;
                    let d = decode_e2m1(nib) * s;
                    let raw = vals[byte_idx];
                    assert!(d == 0.0 || d.signum() == raw.signum(), "sign flip at {byte_idx}");
                    if mode == ScalesMode::Ceil {
                        // ceiling guarantee: no clipping, error <= one e2m1 half-gap
                        assert!(raw.abs() <= 6.0 * s * (1.0 + 1e-6));
                        assert!((d - raw).abs() <= s * (1.0 + 1e-6));
                    }
                }
            }
            if mode == ScalesMode::Mse {
                assert!(st.sum_sq_err <= sse_ceil * 1.000_000_1);
            } else {
                assert_eq!(st.sum_sq_err, sse_ceil);
            }
        }
    }

    #[test]
    fn mse_dominates_ceil_per_tensor() {
        // heavy-tailed tensor: per-sub-block SSE of the written encoding must never
        // exceed the ceiling reference (the ceil byte is always in the candidate
        // set), and must be strictly better in aggregate on skewed weight data
        let mut vals: Vec<f32> = Vec::with_capacity(64 * 128);
        let mut x: u32 = 42;
        for _ in 0..64 * 128 {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = ((x >> 8) % 1000) as f32 / 1000.0;
            let outlier = (x >> 20) % 16 == 0;
            vals.push((m - 0.5) * if outlier { 40.0 } else { 1.0 });
        }
        let (_b, _g, st, sse_ceil) = quantize_nvfp4(&vals, ScalesMode::Mse);
        assert!(st.sum_sq_err <= sse_ceil * 1.000_000_1);
        assert!(st.sum_sq_err < sse_ceil, "MSE must strictly win overall");
    }
}
