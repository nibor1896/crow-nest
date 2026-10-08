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
//! exchangeable to FP8), `vit` (model.visual — carried, optional to load; the dense 27B row
//! omits it, `recipe::omitted`), `mtp`
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
//!   diag — Crow #300 phase 2 (`--diag-stats <f.json>`): all 126 finite ladder steps per
//!          sub-block, scored by the ACTIVATION-weighted SSE sum_j d_j (q_j - w_j)^2
//!          (d_j = sum x_j^2 of the input column over calibration tokens,
//!          `oracle/calib_qwen35_stats.py` + `oracle/export_diag_stats.py`); searched on
//!          every core. Clipping allowed and reported as under `mse`. The 27B's scale
//!          policy since 2026-09-26 (decode_out/p2-lh: KLD 0.290 -> 0.223).
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
//!
//! crow-nest #154/#155 (2026-10-08, GLM-5.3-Flash steps 4 and 5): FP8 E4M3 input with 128x128
//! block scales (`src/fp8.rs`), the `cnq4.5-glm5-next` row (`recipe::decide_glm5_next`), a code
//! histogram per NVFP4 tensor and the exact static order-0 coder sizes in the sidecar
//! (`src/entropy.rs`), header caches (`--headers`), the shard-window mode (`--consume`), and a
//! journal every conversion resumes from. Every family but GLM keeps its write order and bytes.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom, Write};

mod dense_overlay;
mod entropy;
mod expert_overlay;
mod expert_requant;
mod fp8;
mod imatrix;
mod dequant;
mod layer_rule_overlay;
mod mul1;
mod partial;
mod recipe;
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

/// The f32 scale of one NVFP4 sub-block: its ue4m3 byte times the tensor's global scale.
/// crow-nest #156: gate 0 (the sidecar, `quantize_nvfp4_w`) and `converter dequant`
/// (`dequant_nvfp4`) share this and [`nvfp4_value`], so both see the same f32 per value.
#[inline]
fn nvfp4_scale(byte: u32, global: f32) -> f32 {
    decode_ue4m3(byte) * global
}

/// One decoded NVFP4 value: the E2M1 code times its sub-block scale.
#[inline]
fn nvfp4_value(nib: u32, dec: f32) -> f32 {
    decode_e2m1(nib) * dec
}

/// Decode whole 36-byte NVFP4 blocks (4 ue4m3 sub-block scales, then 32 bytes of E2M1 codes,
/// value 2k in the low nibble of byte k, 2k+1 in the high one) to f32, exactly as gate 0
/// decodes the written encoding.
fn dequant_nvfp4(blocks: &[u8], global: f32) -> Vec<f32> {
    assert!(blocks.len() % 36 == 0, "{} B is not whole 36-byte NVFP4 blocks", blocks.len());
    let mut out = Vec::with_capacity(blocks.len() / 36 * 64);
    for b in blocks.chunks_exact(36) {
        for sb in 0..4 {
            let dec = nvfp4_scale(b[sb] as u32, global);
            for j in 0..16 {
                let byte_idx = sb * 16 + j;
                let nib = (b[4 + byte_idx / 2] as u32 >> (4 * (byte_idx % 2))) & 0xF;
                out.push(nvfp4_value(nib, dec));
            }
        }
    }
    out
}

// ---------------- manifest ----------------

struct TensorEntry {
    name: String,
    shape: Vec<usize>,
    n_values: usize,
    /// the safetensors dtype of the source (`BF16`, `F32`, `F16`, `I64`)
    src_dtype: String,
    /// Crow #300 C6: the recipe's decision (dtype, section and the row that made it)
    decision: recipe::Decision,
    shard: String,
    data_begin: u64, // byte offset inside the shard file (incl. header)
    data_end: u64,
    /// #155: the 128x128 block scale of an F8_E4M3 weight (`X.weight_scale_inv`)
    scale: Option<ScaleSrc>,
}

// The BF16 keep set and the section patterns moved to `recipe.rs` with Crow #300 C6: they are
// now one row per model family. The Flash-Next row (`recipe::decide_flash_next`) is the keep set
// that stood here at `64c242b`, verbatim, and is proved against every tensor of CNQ4.5-M.

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
        // #154/#155: the GLM-5.3-Flash originals; read only through their weight_scale_inv
        "F8_E4M3" => Some(1),
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
    /// Crow #300 phase 2 (decode_out/p2-lh): every one of the 126 finite ue4m3 steps is tried
    /// and the one with the smallest ACTIVATION-WEIGHTED error sum_j d_j (q_j - w_j)^2 wins,
    /// d_j = sum over calibration tokens of x_j^2 for the sub-block's input column j
    /// (`--diag-stats`, `oracle/export_diag_stats.py`) — clipping allowed, as `Mse`
    Diag,
}

/// Crow #300 phase 2: the `--scales diag` weights, one f32 vector per input group
/// (`oracle/calib_qwen35_stats.py` groups: `layers.N.attn_in`, `o_in`, `gdn_in`, `gdn_out_in`,
/// `mlp_in`, `down_in`, `head_in`), read from `oracle/export_diag_stats.py`'s JSON
struct DiagStats {
    groups: BTreeMap<String, Vec<f32>>,
    path: String,
    stats_sha256: String,
}

impl DiagStats {
    fn load(path: &str) -> Result<DiagStats, String> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(path).map_err(|e| format!("{path}: {e}"))?)
            .map_err(|e| format!("{path}: {e}"))?;
        let mut groups = BTreeMap::new();
        for (k, arr) in v["groups"].as_object().ok_or(format!("{path}: no groups"))? {
            let vals: Vec<f32> = arr.as_array().ok_or(format!("{path}: {k} is not an array"))?
                .iter().map(|x| x.as_f64().map(|f| f as f32).ok_or(format!("{path}: {k}: not a number")))
                .collect::<Result<_, _>>()?;
            groups.insert(k.clone(), vals);
        }
        Ok(DiagStats { groups, path: path.to_string(), stats_sha256: v["stats_sha256"].as_str().unwrap_or("").to_string() })
    }

    /// the input group of one NVFP4 projection of the dense family, None for any other tensor
    fn key_of(name: &str) -> Option<String> {
        if name == "lm_head.weight" {
            return Some("head_in".into());
        }
        let rest = name.strip_prefix("model.language_model.layers.")?;
        let (layer, proj) = rest.split_once('.')?;
        let group = match proj {
            "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => "attn_in",
            "self_attn.o_proj.weight" => "o_in",
            "linear_attn.in_proj_qkv.weight" | "linear_attn.in_proj_z.weight" => "gdn_in",
            "linear_attn.out_proj.weight" => "gdn_out_in",
            "mlp.gate_proj.weight" | "mlp.up_proj.weight" => "mlp_in",
            "mlp.down_proj.weight" => "down_in",
            _ => return None,
        };
        Some(format!("layers.{layer}.{group}"))
    }

    /// the weights of one tensor's input columns, or a named refusal
    fn weights_for(&self, name: &str, cols: usize) -> Result<&[f32], String> {
        let key = DiagStats::key_of(name).ok_or(format!("--scales diag: no calibration group for {name}"))?;
        let d = self.groups.get(&key).ok_or(format!("--scales diag: {} has no group {key} (for {name})", self.path))?;
        if d.len() != cols {
            return Err(format!("--scales diag: group {key} has {} columns, {name} has {cols}", d.len()));
        }
        Ok(d)
    }
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

/// `--scales diag`: all 126 finite ue4m3 steps (bytes 1..=0x7E; 0x7F is the NaN code) scored by
/// sum_j (d_j * (q_j - w_j)^2 summed in order), the packing arithmetic of `quant_dequant`; the
/// smallest wins, ties keep the smaller byte, an all-zero sub-block keeps byte 0. The same
/// search as `oracle/nvfp4_sim.py` `search("diag")`, which the simulation of record ran.
fn encode_subblock_diag(sub: &[f32], global: f32, d: &[f32]) -> u32 {
    if sub.iter().all(|v| *v == 0.0) {
        return 0;
    }
    let mut best = 0u32;
    let mut best_err = f32::INFINITY;
    for b in 1..=126u32 {
        let s = decode_ue4m3(b) * global;
        let inv = 1.0 / s;
        let mut err = 0.0f32;
        for (v, w) in sub.iter().zip(d) {
            let e = quant_dequant(*v, s, inv) - *v;
            err += e * e * *w;
        }
        if err < best_err {
            best_err = err;
            best = b;
        }
    }
    best
}

/// NVFP4-RTN with in-place dequant statistics (gate 0 of the measurement ladder).
/// Returns (packed blocks, global scale, stats of the WRITTEN encoding, ceiling-
/// reference SSE on the SAME weights — identical to the written SSE in Ceil mode).
/// The GLOBAL scale stays max-based in both modes: ladder utilization unchanged;
/// only the sub-block scale choice differs.
fn quantize_nvfp4(values: &[f32], mode: ScalesMode) -> (Vec<u8>, f32, QuantStats, f64) {
    quantize_nvfp4_w(values, mode, None)
}

/// `quantize_nvfp4` with the `--scales diag` weights: `diag` = (d over the input columns, the
/// row length); a sub-block is 16 consecutive columns of one row (row length % 16 == 0)
fn quantize_nvfp4_w(values: &[f32], mode: ScalesMode, diag: Option<(&[f32], usize)>) -> (Vec<u8>, f32, QuantStats, f64) {
    quantize_nvfp4_cap(values, mode, diag, 0x7F)
}

/// #177: `quantize_nvfp4_w` with every sub-block scale byte capped at `max_byte`
/// (`recipe::Family::scale_byte_max`). The cap is applied BEFORE the E2M1 codes are chosen, so
/// the codes round against the scale that is written and the statistics describe the written
/// bytes. `0x7F` is no cap (the families of record, byte-identical to before); `0x7E` keeps the
/// E4M3 NaN code out of a container. In `ceil` mode the "ceiling reference" is then the capped
/// ceiling, the encoding that is written.
fn quantize_nvfp4_cap(values: &[f32], mode: ScalesMode, diag: Option<(&[f32], usize)>, max_byte: u32) -> (Vec<u8>, f32, QuantStats, f64) {
    assert!(values.len() % 64 == 0);
    assert_eq!(mode == ScalesMode::Diag, diag.is_some(), "--scales diag needs its weights, and only it");
    if let Some((d, cols)) = diag {
        assert!(cols % 16 == 0 && d.len() == cols && values.len() % cols == 0);
    }
    // `--scales diag` tries 126 steps per sub-block (~2.5 h on one core for the 27B): the bytes
    // are chosen up front on every core, each sub-block independently, so the result is the
    // same as the serial search byte for byte
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
    let diag_bytes: Vec<u32> = match diag {
        Some((d, cols)) => {
            let n_sub = values.len() / 16;
            let mut out = vec![0u32; n_sub];
            let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
            let per = n_sub.div_ceil(threads).max(1);
            std::thread::scope(|sc| {
                for (c, chunk) in out.chunks_mut(per).enumerate() {
                    sc.spawn(move || {
                        for (k, byte) in chunk.iter_mut().enumerate() {
                            let i = c * per + k;
                            let col = (i * 16) % cols;
                            *byte = encode_subblock_diag(&values[i * 16..i * 16 + 16], global, &d[col..col + 16]);
                        }
                    });
                }
            });
            out
        }
        None => Vec::new(),
    };

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
                ScalesMode::Diag => {
                    sse_ceil += subblock_sse(sub, decode_ue4m3(ceil_byte) * global);
                    diag_bytes[b * 4 + sb]
                }
            };
            let stored = stored.min(max_byte);
            scales[sb] = stored;
            let dec = nvfp4_scale(stored, global);
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
                let d = nvfp4_value(nib, dec);
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

const HELP: &str = "usage: converter [--scales ceil|mse] --source-repo <org/name> [--revision <sha>] <model-dir | file.safetensors> <out.cnq>\n  writes an index v2 container: config.json + generation_config.json verbatim, the family's recipe, source repo/revision/shard sha256\n  (--revision defaults to the Hugging Face cache in the model dir; Crow #300 C6)\n  --scales ceil  ceiling sub-block scales: stored >= raw always, max_rel <= 1.0 (default)\n  --scales mse   per-sub-block SSE-minimizing scales: clipping allowed, quality via MSE report\n  --scales diag --diag-stats <f.json>  all 126 ue4m3 steps scored by the activation-weighted error (Crow #300 p2-lh)\n  --headers <dir>  read the shard headers from a header cache (<dir>/<shard>.json) (crow-nest #154)\n  --consume <shard-dir>  convert while shards come and go: wait for <shard>.verified, write <shard>.done, never delete; needs --headers (crow-nest #155)\n  an interrupted conversion resumes from <out>.cnq.journal.jsonl (crow-nest #155)\n  --layers <spec> [--with-embed-head]  a partial container: text layers <spec> only (0-3, 0,3) [+ token embedding, lm_head, final norm]; the rest is filtered and the index says so (crow-nest #156)\n       converter [--scales ceil|mse] requant-check <dense.safetensors> <container.cnq>\n  re-quantizes fetched originals and compares them with the container's own bytes (#76)\n       converter dequant <container.cnq> (<name>[:<r0>:<r1>] ... | --names -)\n  writes the named tensors (rows r0..r1) to stdout as f32 little endian, decoded as gate 0 decodes them (crow-nest #156)\n       converter dense-overlay --base <container.cnq> --out <overlay.cnq> (--from-originals <f.safetensors> | --from-container <base.cnq>) [--kinds ...]\n  builds a bf16 overlay container over the dense text tensors (#77)\n       converter expert-overlay --base <container.cnq> --out <overlay.cnq> --originals <dir> --layers 1,7,... --rule mse|mse46|imatrix|imatrix46 [--imatrix <f.gguf>]\n  builds an nvfp4 overlay container over the routed experts of those layers (#79)\n       converter layer-rule-overlay --base <container.cnq> --out <overlay.cnq> (--from-originals <f.safetensors> | --from-container <base.cnq>) --arm attn-v-out|ffn-down-rule|ffn-down-all\n  builds a bf16 overlay container for one llama.cpp-shaped layer-rule arm (#91 phase 1)\n       converter imatrix-show <imatrix.gguf> [tensor ...]\n  prints the importance matrix header and named tensors (#79)\n       converter plan [--headers <dir>] [--source-repo <org/name>] [--revision <sha>] <model-dir | file.safetensors>\n  the dry run: family, recipe, per-tensor dtype/section table, GPU / host byte totals (Crow #300 C6)";

/// `converter imatrix-show <imatrix.gguf> [tensor ...]` — #79. Read-only: the kv block, the
/// tensor count, and for every named tensor its dims, its data offset, its first eight values,
/// its last three and its sum. It exists to be compared with a second, independent reader
/// (`tools/` has none; a stdlib Python GGUF parser was used, see `docs/expert-requant.md`),
/// because a header parser that is wrong by one field reads plausible numbers out of the wrong
/// place.
fn imatrix_show(args: &[String]) -> i32 {
    let Some(path) = args.first() else {
        eprintln!("usage: converter imatrix-show <imatrix.gguf> [tensor ...]");
        return 2;
    };
    let g = match imatrix::Gguf::open(std::path::Path::new(path)) {
        Ok(g) => g,
        Err(e) => {
            eprintln!("{e}");
            return 2;
        }
    };
    println!(
        "{}: GGUF v{}, {} tensors, {} kv, alignment {}, data_start {}, {} B",
        path,
        g.version,
        g.tensors.len(),
        g.kv.len(),
        g.alignment,
        g.data_start,
        g.file_len
    );
    for (k, v) in &g.kv {
        let s = v.to_string();
        println!("  kv {k} = {}", if s.len() > 120 { format!("{}… ({} chars)", &s[..120], s.len()) } else { s });
    }
    for name in &args[1..] {
        let Some(t) = g.find(name) else {
            println!("  {name}: NOT IN THIS FILE");
            return 1;
        };
        let v = match g.read_f32(name) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("{e}");
                return 1;
            }
        };
        let sum: f64 = v.iter().map(|x| *x as f64).sum();
        println!(
            "  {name}  dims {:?}  type {}  offset {}  n {}\n    first8 {:?}\n    last3  {:?}\n    sum    {:.9e}",
            t.dims,
            t.ggml_type,
            t.offset,
            v.len(),
            &v[..8.min(v.len())],
            &v[v.len().saturating_sub(3)..],
            sum
        );
    }
    0
}

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
    // #77: the same rule for the overlay builder — it is taken off the front, it writes a NEW
    // file and it never reaches the conversion path below. Without the word `dense-overlay`
    // nothing about this binary changed.
    if let Some(at) = all.iter().position(|a| a == "dense-overlay") {
        if at != 0 {
            eprintln!("unexpected argument {} before dense-overlay\n{}", all[0], dense_overlay::HELP);
            std::process::exit(2);
        }
        std::process::exit(dense_overlay::run(&all[1..]));
    }
    // #79: and the same rule once more for the routed-expert overlay. Four subcommands now
    // sit in front of the conversion path and none of them can be reached by accident: each
    // one demands its own word as argument zero.
    if let Some(at) = all.iter().position(|a| a == "expert-overlay") {
        if at != 0 {
            eprintln!("unexpected argument {} before expert-overlay\n{}", all[0], expert_overlay::HELP);
            std::process::exit(2);
        }
        std::process::exit(expert_overlay::run(&all[1..]));
    }
    // #91 phase 1: the llama.cpp layer-rule arms, as bf16 overlays over the dense path. Same
    // additive rule as the three above: its own word as argument zero, and the conversion path
    // below never sees it.
    if let Some(at) = all.iter().position(|a| a == "layer-rule-overlay") {
        if at != 0 {
            eprintln!("unexpected argument {} before layer-rule-overlay\n{}", all[0], layer_rule_overlay::HELP);
            std::process::exit(2);
        }
        std::process::exit(layer_rule_overlay::run(&all[1..]));
    }
    // #79: read the importance matrix back out and print it, so its numbers can be checked
    // against an independent (stdlib Python) GGUF reader.
    if let Some(at) = all.iter().position(|a| a == "imatrix-show") {
        if at != 0 {
            eprintln!("unexpected argument {} before imatrix-show", all[0]);
            std::process::exit(2);
        }
        std::process::exit(imatrix_show(&all[1..]));
    }

    // crow-nest #156: the read-only decoder of a written container (the oracle's `--weights
    // container` back end reads its weights through it). Same additive rule: its own word as
    // argument zero, and it never reaches the conversion path.
    if let Some(at) = all.iter().position(|a| a == "dequant") {
        if at != 0 {
            eprintln!("unexpected argument {} before dequant\n{}", all[0], dequant::HELP);
            std::process::exit(2);
        }
        std::process::exit(dequant::run(&all[1..]));
    }

    // Crow #300 C6: the read-only plan. Same additive rule as every subcommand above: its own
    // word as argument zero, and it reads the index, the shard headers and the config only.
    if let Some(at) = all.iter().position(|a| a == "plan") {
        if at != 0 {
            eprintln!("unexpected argument {} before plan\n{PLAN_HELP}", all[0]);
            std::process::exit(2);
        }
        std::process::exit(plan(&all[1..]));
    }

    let mut positional: Vec<String> = Vec::new();
    let mut mode = ScalesMode::Ceil;
    let mut prov = Provenance::default();
    let mut diag_path: Option<String> = None;
    let mut opts = ConvertOpts::default();
    let mut layers_spec: Option<String> = None;
    let mut with_embed_head = false;
    let mut argv = std::env::args().skip(1);
    while let Some(arg) = argv.next() {
        match arg.as_str() {
            "--scales" => match argv.next().as_deref() {
                Some("ceil") => mode = ScalesMode::Ceil,
                Some("mse") => mode = ScalesMode::Mse,
                Some("diag") => mode = ScalesMode::Diag,
                other => {
                    eprintln!("--scales needs `ceil`, `mse` or `diag`, got {other:?}\n{HELP}");
                    std::process::exit(2);
                }
            },
            "--diag-stats" => diag_path = argv.next(),
            "--source-repo" => prov.repo = argv.next(),
            "--revision" => prov.revision = argv.next(),
            "--headers" => opts.headers = argv.next().map(std::path::PathBuf::from),
            "--consume" => opts.consume = argv.next().map(std::path::PathBuf::from),
            "--layers" => layers_spec = argv.next(),
            "--with-embed-head" => with_embed_head = true,
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
    match (layers_spec, with_embed_head) {
        (Some(spec), e) => match partial::LayerFilter::parse(&spec, e) {
            Ok(f) => opts.filter = Some(f),
            Err(e) => {
                eprintln!("conversion refused: {e}\n{HELP}");
                std::process::exit(2);
            }
        },
        (None, true) => {
            eprintln!("conversion refused: --with-embed-head belongs to a partial container and needs --layers <spec>\n{HELP}");
            std::process::exit(2);
        }
        (None, false) => {}
    }
    let input = std::path::PathBuf::from(&positional[0]);
    let out_path = std::path::PathBuf::from(&positional[1]);
    let diag = match diag_path.as_deref().map(DiagStats::load) {
        None => None,
        Some(Ok(d)) => {
            eprintln!("--diag-stats {}: {} groups (stats sha256 {})", d.path, d.groups.len(), d.stats_sha256);
            Some(d)
        }
        Some(Err(e)) => {
            eprintln!("conversion refused: {e}");
            std::process::exit(2);
        }
    };
    std::process::exit(convert_with(&input, &out_path, mode, &prov, diag.as_ref(), &opts));
}

/// `--source-repo` / `--revision`: where the checkpoint came from, for the index v2 `model`
/// block. The revision defaults to the Hugging Face local-dir cache's (`recipe::read_hf_tree`)
/// or to `sha` of `hf-revision.json` (`recipe::read_hf_api_info`, crow-nest #155).
#[derive(Default, Clone)]
struct Provenance {
    repo: Option<String>,
    revision: Option<String>,
}

/// One FP8 weight's `weight_scale_inv` (crow-nest #155): where its bytes are.
#[derive(Clone)]
struct ScaleSrc {
    name: String,
    dtype: String,
    shard: String,
    data_begin: u64,
    data_end: u64,
}

/// The header scan of a conversion (and of `plan`): the config, the family, and every tensor
/// with its recipe decision. Reads the model index, the shard headers (from the shard files, or
/// from a header cache, `--headers`) and the two config files, never a tensor's payload.
struct Manifest {
    family: recipe::Family,
    config_json: String,
    generation_config_json: String,
    config: serde_json::Value,
    tensors: Vec<TensorEntry>,
    /// tensors the recipe does not write (`recipe::omitted`): reason -> (count, source bytes);
    /// an omitted FP8 weight's `weight_scale_inv` counts under the same reason
    omitted: BTreeMap<&'static str, (usize, u64)>,
    shard_files: Vec<std::path::PathBuf>,
    weight_map: Option<BTreeMap<String, String>>,
    single_file: bool,
    /// the directory the configs (and the HF cache) are read from
    model_dir: std::path::PathBuf,
    /// #155: the header cache the manifest was built from (`--headers`): shard files need not
    /// exist for the plan, and a conversion reads each shard only when it gets to it
    headers: Option<std::path::PathBuf>,
    /// #155: the shard sizes the header cache recorded (`size`), file name -> bytes
    shard_size: BTreeMap<String, u64>,
}

/// The `(header, data_start, size)` of one shard: from the file itself, or from the header
/// cache `<headers>/<shard>.json` (`{"header", "data_start"[, "shard", "size"]}`, the format of
/// `tools/fetch-dense-originals.py` `RangeFetcher.read_header`).
fn shard_header(path: &std::path::Path, headers: Option<&std::path::Path>) -> Result<(serde_json::Value, u64, Option<u64>), String> {
    let file = path.file_name().and_then(|f| f.to_str()).unwrap_or_default().to_string();
    let Some(dir) = headers else {
        let (h, s) = read_safetensors_header(path).map_err(|e| format!("{}: shard header: {e}", path.display()))?;
        return Ok((h, s, None));
    };
    let p = dir.join(format!("{file}.json"));
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| format!("{}: header cache: {e}", p.display()))?)
        .map_err(|e| format!("{}: {e}", p.display()))?;
    if let Some(s) = v["shard"].as_str() {
        if s != file {
            return Err(format!("{}: the header cache is for {s}, not {file}", p.display()));
        }
    }
    let start = v["data_start"].as_u64().ok_or(format!("{}: no data_start", p.display()))?;
    if !v["header"].is_object() {
        return Err(format!("{}: no header object", p.display()));
    }
    Ok((v["header"].clone(), start, v["size"].as_u64()))
}

#[cfg_attr(not(test), allow(dead_code))]
fn build_manifest(input: &std::path::Path) -> Result<Manifest, String> {
    build_manifest_from(input, None)
}

fn build_manifest_from(input: &std::path::Path, headers: Option<&std::path::Path>) -> Result<Manifest, String> {
    let single_file = input.is_file();
    let model_dir = if single_file { input.parent().map(|p| p.to_path_buf()).unwrap_or_default() } else { input.to_path_buf() };
    let model_dir = if model_dir.as_os_str().is_empty() { std::path::PathBuf::from(".") } else { model_dir };
    let (family, config_json, generation_config_json, config) = recipe::read_model_configs(&model_dir)?;
    recipe::check_family_config(family, &config)?;
    let mut tensors: Vec<TensorEntry> = Vec::new();
    let mut weight_map: Option<BTreeMap<String, String>> = None;
    let shard_files: Vec<std::path::PathBuf> = if single_file {
        vec![input.to_path_buf()]
    } else {
        let p = input.join("model.safetensors.index.json");
        let index: serde_json::Value = serde_json::from_slice(&std::fs::read(&p).map_err(|e| format!("{}: {e}", p.display()))?)
            .map_err(|e| format!("{}: {e}", p.display()))?;
        let mut wm = BTreeMap::new();
        for (name, shard) in index["weight_map"].as_object().ok_or("model index: no weight_map")? {
            wm.insert(name.clone(), shard.as_str().ok_or("model index: weight_map value is not a string")?.to_string());
        }
        let mut files: Vec<String> = wm.values().cloned().collect();
        files.sort();
        files.dedup();
        weight_map = Some(wm);
        files.iter().map(|f| input.join(f)).collect()
    };
    let mut refusals: Vec<String> = Vec::new();
    let mut omitted: BTreeMap<&'static str, (usize, u64)> = BTreeMap::new();
    let mut scales: BTreeMap<String, ScaleSrc> = BTreeMap::new();
    let mut scale_shape: BTreeMap<String, Vec<usize>> = BTreeMap::new();
    let mut shard_size: BTreeMap<String, u64> = BTreeMap::new();
    for path in &shard_files {
        let shard_name = path.file_name().unwrap().to_str().unwrap().to_string();
        let (header, data_start, size) = shard_header(path, headers)?;
        if let Some(s) = size {
            shard_size.insert(shard_name.clone(), s);
        }
        if let Some(entries) = header.as_object() {
            for (name, info) in entries {
                if name == "__metadata__" {
                    continue;
                }
                let dt = info["dtype"].as_str().unwrap_or("F32");
                // crow-nest #154: an unknown dtype used to be skipped with a line on stderr
                // (`eprintln!("skip ...")`), which would have dropped every F8_E4M3 tensor of
                // GLM-5.3-Flash from the plan; it is a refusal by name now
                let Some(es) = elem_size_of(dt) else {
                    refusals.push(format!("{name}: dtype {dt} - this converter reads F32, BF16, F16, I64 and F8_E4M3 only (refused, not skipped)"));
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
                if let Some(why) = recipe::omitted(family, name) {
                    let e = omitted.entry(why).or_default();
                    e.0 += 1;
                    e.1 += end - begin;
                    continue;
                }
                if name.ends_with(".weight_scale_inv") {
                    scale_shape.insert(name.clone(), shape);
                    scales.insert(
                        name.clone(),
                        ScaleSrc { name: name.clone(), dtype: dt.to_string(), shard: shard_name.clone(), data_begin: data_start + begin, data_end: data_start + end },
                    );
                    continue;
                }
                let decision = match recipe::decide(family, name, &shape, dt) {
                    Ok(d) => d,
                    Err(why) => {
                        refusals.push(why);
                        continue;
                    }
                };
                tensors.push(TensorEntry {
                    name: name.clone(),
                    shape,
                    n_values: n,
                    src_dtype: dt.to_string(),
                    decision,
                    shard: shard_name.clone(),
                    data_begin: data_start + begin,
                    data_end: data_start + end,
                    scale: None,
                });
            }
        }
    }
    // #155: pair every FP8 weight with its `weight_scale_inv` (128x128 grid), by name
    for t in tensors.iter_mut() {
        let key = format!("{}_scale_inv", t.name);
        match scales.remove(&key) {
            Some(s) => {
                let shp = &scale_shape[&key];
                if t.src_dtype != "F8_E4M3" {
                    refusals.push(format!("{key}: a block scale beside a {} weight - only F8_E4M3 weights are read through their scale", t.src_dtype));
                } else if t.shape.len() != 2 {
                    refusals.push(format!("{}: an F8_E4M3 tensor of shape {:?} - only 2-D FP8 weights with 128x128 block scales are read", t.name, t.shape));
                } else if shp.as_slice() != fp8::scale_grid(t.shape[0], t.shape[1]) {
                    refusals.push(format!("{key}: shape {shp:?}, the 128x128 grid of {:?} is {:?}", t.shape, fp8::scale_grid(t.shape[0], t.shape[1])));
                } else if s.dtype != "F32" && s.dtype != "BF16" {
                    refusals.push(format!("{key}: dtype {} - block scales are read as F32 or BF16 only", s.dtype));
                } else {
                    t.scale = Some(s);
                }
            }
            None if t.src_dtype == "F8_E4M3" => refusals.push(format!("{}: an F8_E4M3 weight without its {key}", t.name)),
            None => {}
        }
    }
    for k in scales.keys() {
        refusals.push(format!("{k}: a weight_scale_inv without its F8_E4M3 weight"));
    }
    if !refusals.is_empty() {
        refusals.sort();
        return Err(format!("{} tensor(s) refused by the {} recipe:\n  {}", refusals.len(), family.recipe(), refusals.join("\n  ")));
    }
    tensors.sort_by(|a, b| a.shard.cmp(&b.shard).then(a.data_begin.cmp(&b.data_begin)));
    let geo = recipe::derive_geo(family, &config);
    let named: Vec<(String, Vec<usize>)> = tensors.iter().map(|t| (t.name.clone(), t.shape.clone())).collect();
    recipe::check_geo_against_tensors(&geo, &named)?;
    Ok(Manifest {
        family,
        config_json,
        generation_config_json,
        config,
        tensors,
        omitted,
        shard_files,
        weight_map,
        single_file,
        model_dir,
        headers: headers.map(|p| p.to_path_buf()),
        shard_size,
    })
}

/// The index v2 `model` block's provenance: repo from `--source-repo`, revision from
/// `--revision`, the HF cache or `hf-revision.json` (all that are given must agree), and one
/// record per shard. `compute` hashes a shard that has no LFS record (the conversion); the plan
/// passes `false` and reports those shards as "computed at conversion". #155: a shard the
/// model-info JSON has an LFS record for is recorded from it (`sha256_from: hf-lfs`) without
/// reading it; under `--headers` it need not be on disk, and when it is, its size must match.
fn provenance(m: &Manifest, prov: &Provenance, compute: bool) -> Result<(String, String, Vec<recipe::ShardRecord>, usize), String> {
    let tree = recipe::read_hf_tree(&m.model_dir)?;
    let api = recipe::read_hf_api_info(&m.model_dir)?;
    let mut revision = prov.revision.clone();
    for (src, rev) in [
        ("the Hugging Face cache in this directory", tree.as_ref().map(|t| t.revision.clone())),
        ("hf-revision.json in this directory", api.as_ref().and_then(|a| a.revision.clone())),
    ] {
        match (&revision, rev) {
            (Some(a), Some(b)) if *a != b => return Err(format!("--revision {a}, but {src} is revision {b}")),
            (None, Some(b)) => revision = Some(b),
            _ => {}
        }
    }
    let Some(revision) = revision else {
        return Err("no --revision given and no Hugging Face cache (.cache/huggingface/trees) or hf-revision.json to read it from".into());
    };
    let repo = prov.repo.clone().ok_or("--source-repo <org/name> is required: the index v2 names the checkpoint it was converted from")?;
    let mut shards = Vec::new();
    let mut pending = 0usize;
    for p in &m.shard_files {
        let file = p.file_name().unwrap().to_string_lossy().to_string();
        let in_tree = tree.as_ref().is_some_and(|t| t.lfs.contains_key(&file));
        if let (false, Some((sha, size))) = (in_tree, api.as_ref().and_then(|a| a.lfs.get(&file))) {
            match std::fs::metadata(p) {
                Ok(md) if md.len() != *size => {
                    return Err(format!("{file}: {} B on disk, but Hugging Face recorded {size} B - an incomplete or foreign file", md.len()))
                }
                Ok(_) => {}
                Err(_) if m.headers.is_some() => {}
                Err(e) => return Err(format!("{}: {e}", p.display())),
            }
            if let Some(hs) = m.shard_size.get(&file) {
                if hs != size {
                    return Err(format!("{file}: the header cache says {hs} B, Hugging Face recorded {size} B"));
                }
            }
            shards.push(recipe::ShardRecord { file, size: *size, sha256: sha.clone(), sha256_from: "hf-lfs" });
            continue;
        }
        match recipe::shard_record(p, tree.as_ref(), compute)? {
            Some(r) => shards.push(r),
            None => pending += 1,
        }
    }
    Ok((repo, revision, shards, pending))
}

fn gib(b: u64) -> f64 {
    b as f64 / (1u64 << 30) as f64
}

/// #155: the write order of a conversion, in units. A unit is written back to back; a unit with
/// `align` starts at an absolute file offset that is a multiple of it (zeros before it). Every
/// family but GLM: one tensor per unit, no alignment (the order and the bytes of record). GLM:
/// the gate, up and down projections of one routed expert of one layer form one unit (in that
/// order), aligned to 4096 so an expert is one aligned read for the NVMe tier; the unit sits
/// where its first projection appears in (shard, offset) order.
struct Unit {
    tensors: Vec<usize>,
    align: u64,
}

const EXPERT_ALIGN: u64 = 4096;

fn write_units(m: &Manifest) -> Vec<Unit> {
    if m.family != recipe::Family::Glm5Next {
        return (0..m.tensors.len()).map(|i| Unit { tensors: vec![i], align: 0 }).collect();
    }
    let mut groups: BTreeMap<(u64, u64), [Option<usize>; 3]> = BTreeMap::new();
    for (i, t) in m.tensors.iter().enumerate() {
        if let Some((l, e, p)) = recipe::glm_expert(&t.name) {
            let slot = match p {
                "gate" => 0,
                "up" => 1,
                _ => 2,
            };
            groups.entry((l, e)).or_default()[slot] = Some(i);
        }
    }
    let mut emitted = vec![false; m.tensors.len()];
    let mut units = Vec::new();
    for (i, t) in m.tensors.iter().enumerate() {
        if emitted[i] {
            continue;
        }
        match recipe::glm_expert(&t.name) {
            Some((l, e, _)) => {
                let g: Vec<usize> = groups[&(l, e)].iter().flatten().copied().collect();
                for &k in &g {
                    emitted[k] = true;
                }
                units.push(Unit { tensors: g, align: EXPERT_ALIGN });
            }
            None => {
                emitted[i] = true;
                units.push(Unit { tensors: vec![i], align: 0 });
            }
        }
    }
    units
}

/// zeros before a unit that starts at blob offset `blob_len` with alignment `align`
fn pad_for(blob_len: u64, align: u64) -> u64 {
    if align == 0 {
        0
    } else {
        (align - (12 + blob_len) % align) % align
    }
}

/// The blob length a conversion of `m` writes (payload incl. alignment zeros) and the zeros.
fn layout(m: &Manifest, units: &[Unit]) -> (u64, u64) {
    let (mut blob, mut pads) = (0u64, 0u64);
    for u in units {
        let p = pad_for(blob, u.align);
        pads += p;
        blob += p;
        for &i in &u.tensors {
            blob += m.tensors[i].decision.dtype.bytes(m.tensors[i].n_values);
        }
    }
    (blob, pads)
}

/// The shards one tensor reads (its own and its scale's).
fn shards_of(t: &TensorEntry) -> Vec<&str> {
    let mut v = vec![t.shard.as_str()];
    if let Some(s) = &t.scale {
        if s.shard != t.shard {
            v.push(s.shard.as_str());
        }
    }
    v
}

const PLAN_HELP: &str = "usage: converter plan [--headers <dir>] [--source-repo <org/name>] [--revision <sha>] <model-dir | file.safetensors>\n  reads the model index, the shard headers and config.json only; prints the family, the recipe,\n  the per-tensor dtype/section decision table, the per-row summary and the GPU / host byte totals\n  --headers <dir>  read the shard headers from a header cache (<dir>/<shard>.json) instead of the shard files (crow-nest #154)";

/// `converter plan <model-dir>` (Crow #300 C6): the dry run. Nothing is quantized and no
/// tensor payload is read.
fn plan(args: &[String]) -> i32 {
    let mut prov = Provenance::default();
    let mut input: Option<std::path::PathBuf> = None;
    let mut headers: Option<std::path::PathBuf> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--source-repo" => prov.repo = it.next().cloned(),
            "--revision" => prov.revision = it.next().cloned(),
            "--headers" => headers = it.next().map(std::path::PathBuf::from),
            s if s.starts_with("--") || input.is_some() => {
                eprintln!("unexpected argument {s}\n{PLAN_HELP}");
                return 2;
            }
            s => input = Some(std::path::PathBuf::from(s)),
        }
    }
    let Some(input) = input else {
        eprintln!("{PLAN_HELP}");
        return 2;
    };
    let m = match build_manifest_from(&input, headers.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("plan refused: {e}");
            return 2;
        }
    };
    println!(
        "plan      {}: family {} ({}), recipe {}, {} tensors from {} shard file(s){}",
        input.display(),
        m.family.name(),
        m.family.model_type(),
        m.family.recipe(),
        m.tensors.len(),
        m.shard_files.len(),
        match &m.headers {
            Some(h) => format!(", headers from the cache {}", h.display()),
            None => String::new(),
        }
    );
    println!(
        "config    config.json {} B sha256 {}, generation_config.json {} B sha256 {} (stored verbatim in the index v2)",
        m.config_json.len(),
        recipe::sha256_hex(m.config_json.as_bytes()),
        m.generation_config_json.len(),
        recipe::sha256_hex(m.generation_config_json.as_bytes())
    );
    let prov_repo = prov.repo.clone();
    match provenance(&m, &Provenance { repo: prov_repo.or(Some("<--source-repo required at conversion>".into())), ..prov }, false) {
        Ok((repo, rev, shards, pending)) => {
            let lfs = shards.iter().filter(|s| s.sha256_from == "hf-lfs").count();
            let from = match recipe::read_hf_api_info(&m.model_dir) {
                Ok(Some(a)) => format!(" ({})", a.source),
                _ => String::new(),
            };
            println!("source    repo {repo}, revision {rev}; shard sha256: {lfs} from the HF LFS record{from}, {pending} computed at conversion");
        }
        Err(e) => println!("source    NOT READY for a conversion: {e}"),
    }
    println!("geo       {}", recipe::derive_geo(m.family, &m.config));
    println!();
    println!("{:<72} {:>22} {:>7} -> {:<5} {:<4} {:>13}  rule", "tensor", "shape", "src", "out", "sect", "bytes");
    let mut sorted: Vec<&TensorEntry> = m.tensors.iter().collect();
    sorted.sort_by(|a, b| a.name.cmp(&b.name));
    for t in &sorted {
        let d = t.decision;
        println!(
            "{:<72} {:>22} {:>7} -> {:<5} {:<4} {:>13}  {}",
            t.name,
            format!("{:?}", t.shape),
            t.src_dtype,
            d.dtype.as_str(),
            d.section,
            d.dtype.bytes(t.n_values),
            d.rule
        );
    }
    // per row
    let mut rows: BTreeMap<(&str, &str, &str), (usize, u64, u64)> = BTreeMap::new();
    for t in &m.tensors {
        let d = t.decision;
        let e = rows.entry((d.section, d.dtype.as_str(), d.rule)).or_default();
        e.0 += 1;
        e.1 += t.n_values as u64;
        e.2 += d.dtype.bytes(t.n_values);
    }
    println!();
    println!("{:<4} {:<5} {:<72} {:>7} {:>15} {:>15} {:>10}", "sect", "dtype", "rule", "tensors", "values", "bytes", "GiB");
    for ((s, dt, rule), (c, v, b)) in &rows {
        println!("{s:<4} {dt:<5} {rule:<72} {c:>7} {v:>15} {b:>15} {:>10.4}", gib(*b));
    }
    // per histogram class (#155): the classes the code summary of a conversion reports
    let mut classes: BTreeMap<&str, (usize, u64)> = BTreeMap::new();
    for t in m.tensors.iter().filter(|t| t.decision.dtype == recipe::DtypeOut::Nvfp4) {
        let e = classes.entry(recipe::tensor_class(m.family, &t.name, &m.config).0).or_default();
        e.0 += 1;
        e.1 += t.decision.dtype.bytes(t.n_values);
    }
    println!();
    println!("{:<16} {:>7} {:>15} {:>10}   (nvfp4 tensors per code-histogram class)", "class", "tensors", "bytes", "GiB");
    for (c, (n, b)) in &classes {
        println!("{c:<16} {n:>7} {b:>15} {:>10.4}", gib(*b));
    }
    // totals: the token embedding is the one text tensor that lives in host RAM (a table
    // lookup, no GEMM); `vit` and `mtp` are optional to load; `ple` is Flash-Next's own tier
    let is_embed = |t: &TensorEntry| t.name.ends_with("language_model.embed_tokens.weight");
    let is_routed = |t: &TensorEntry| {
        let c = recipe::tensor_class(m.family, &t.name, &m.config).0;
        c == "expert" || c.starts_with("expert_")
    };
    let sum = |f: &dyn Fn(&TensorEntry) -> bool| -> u64 { m.tensors.iter().filter(|t| f(t)).map(|t| t.decision.dtype.bytes(t.n_values)).sum() };
    let text_nvfp4 = sum(&|t| t.decision.section == "text" && t.decision.dtype == recipe::DtypeOut::Nvfp4);
    let text_keep = sum(&|t| t.decision.section == "text" && t.decision.dtype != recipe::DtypeOut::Nvfp4 && !is_embed(t));
    let host_embed = sum(&|t| is_embed(t));
    let routed = sum(&|t| t.decision.section == "text" && is_routed(t));
    let dense = sum(&|t| t.decision.section == "text" && !is_routed(t) && !is_embed(t));
    let mtp = sum(&|t| t.decision.section == "mtp");
    let vit = sum(&|t| t.decision.section == "vit");
    let ple = sum(&|t| t.decision.section == "ple");
    let total = sum(&|_| true);
    let units = write_units(&m);
    let (blob, pads) = layout(&m, &units);
    println!();
    println!("totals    (GiB = 2^30 B, GB = 10^9 B; nvfp4 = 36 B per 64 values, plus one f32 global scale per tensor in the index, not counted)");
    println!("  GPU, text weights NVFP4          {:>15} B  {:>8.3} GiB", text_nvfp4, gib(text_nvfp4));
    println!("  GPU, text keeps (bf16/f32/i64)   {:>15} B  {:>8.3} GiB  (without the token embedding)", text_keep, gib(text_keep));
    println!("  GPU, text subtotal               {:>15} B  {:>8.3} GiB", text_nvfp4 + text_keep, gib(text_nvfp4 + text_keep));
    println!("  host RAM, token embedding        {:>15} B  {:>8.3} GiB", host_embed, gib(host_embed));
    if routed > 0 {
        println!("  routed experts (tiered)          {:>15} B  {:>8.3} GiB", routed, gib(routed));
        println!("  dense resident part              {:>15} B  {:>8.3} GiB  {:>7.3} GB  (text without routed experts and the token embedding)", dense, gib(dense), dense as f64 / 1e9);
    }
    println!("  optional section mtp             {:>15} B  {:>8.3} GiB", mtp, gib(mtp));
    println!("  optional section vit             {:>15} B  {:>8.3} GiB", vit, gib(vit));
    if ple > 0 {
        println!("  section ple (Flash-Next tier)    {:>15} B  {:>8.3} GiB", ple, gib(ple));
    }
    println!("  container payload                {:>15} B  {:>8.3} GiB", total, gib(total));
    for (why, (c, b)) in &m.omitted {
        println!("  omitted, not written             {c:>7} tensors, {b:>15} B source  {:>8.3} GiB  ({why})", gib(*b));
    }
    // #155: FP8 pairs, the write layout and the container size as one number
    let fp8: Vec<&TensorEntry> = m.tensors.iter().filter(|t| t.scale.is_some()).collect();
    if !fp8.is_empty() {
        let sb: u64 = fp8.iter().map(|t| t.scale.as_ref().map(|s| s.data_end - s.data_begin).unwrap_or(0)).sum();
        let straddle = fp8.iter().filter(|t| t.scale.as_ref().is_some_and(|s| s.shard != t.shard)).count();
        println!("  FP8 E4M3 weights read            {:>7} tensors with their weight_scale_inv ({sb} B of scales, read and not written); pairs across two shards: {straddle}", fp8.len());
    }
    let span = units
        .iter()
        .map(|u| {
            let mut s: Vec<&str> = u.tensors.iter().flat_map(|&i| shards_of(&m.tensors[i])).collect();
            s.sort();
            s.dedup();
            s.len()
        })
        .max()
        .unwrap_or(0);
    let aligned = units.iter().filter(|u| u.align > 0).count();
    println!("  write units                      {:>7} ({aligned} aligned to {EXPERT_ALIGN} B), alignment zeros {pads} B; most shards one unit reads: {span}", units.len());
    println!("  container without index trailer  {:>15} B  {:>8.3} GiB  {:>8.3} GB  (12 B head + payload + alignment zeros)", 12 + blob, gib(12 + blob), (12 + blob) as f64 / 1e9);
    // which shards each layer reads (the partial container of step 6 and the window of step 9)
    if m.family == recipe::Family::Glm5Next {
        let mut per: BTreeMap<String, std::collections::BTreeSet<String>> = BTreeMap::new();
        for t in &m.tensors {
            let key = match recipe::glm_parts(&t.name).and_then(|p| p.0) {
                Some(l) => format!("layer {l:>2}"),
                None => "model".to_string(),
            };
            for s in shards_of(t) {
                per.entry(key.clone()).or_default().insert(s.to_string());
            }
        }
        println!();
        println!("shards per layer (written tensors and their scales):");
        for (k, s) in &per {
            println!("  {k:<9} {}", s.iter().map(|x| x.as_str()).collect::<Vec<_>>().join(" "));
        }
    }
    0
}

/// #155: the options of a conversion beyond the scale policy.
#[derive(Clone)]
struct ConvertOpts {
    /// the header cache (`--headers`)
    headers: Option<std::path::PathBuf>,
    /// `--consume <shard-dir>`: shards appear there one by one, each with `<shard>.verified`
    /// (size and sha256 checked by the downloader); the converter writes `<shard>.done` when
    /// every tensor that reads from it is written and synced; it never deletes a shard
    consume: Option<std::path::PathBuf>,
    /// tests only: stop after this many tensors are journalled, as a kill would (no trailer)
    stop_after: Option<usize>,
    poll: std::time::Duration,
    /// #156: `--layers <spec> [--with-embed-head]`, a partial container (`partial.rs`)
    filter: Option<partial::LayerFilter>,
}

impl Default for ConvertOpts {
    fn default() -> Self {
        ConvertOpts { headers: None, consume: None, stop_after: None, poll: std::time::Duration::from_secs(2), filter: None }
    }
}

/// The conversion: the manifest, the streamed payload, the index v2 trailer. Returns the exit
/// code (0, 1 on a bound violation under `--scales ceil`, 2 on a refusal, 3 on a coverage gap).
#[cfg_attr(not(test), allow(dead_code))]
fn convert(input: &std::path::Path, out_path: &std::path::Path, mode: ScalesMode, prov: &Provenance, diag: Option<&DiagStats>) -> i32 {
    convert_with(input, out_path, mode, prov, diag, &ConvertOpts::default())
}

/// What one tensor turned into: the bytes, its index record, its sidecar line and the numbers
/// the section summary sums.
struct Encoded {
    bytes: Vec<u8>,
    entry: serde_json::Value,
    sidecar: serde_json::Value,
    acc: serde_json::Value,
}

/// Read one tensor's source bytes (and its scale's) from `dir/<shard>`.
fn read_source(dir: &std::path::Path, single_file: Option<&std::path::Path>, shard: &str, begin: u64, end: u64) -> Result<Vec<u8>, String> {
    let p = match single_file {
        Some(f) => f.to_path_buf(),
        None => dir.join(shard),
    };
    let mut f = std::fs::File::open(&p).map_err(|e| format!("{}: {e}", p.display()))?;
    f.seek(SeekFrom::Start(begin)).map_err(|e| format!("{}: seek: {e}", p.display()))?;
    let mut raw = vec![0u8; (end - begin) as usize];
    f.read_exact(&mut raw).map_err(|e| format!("{}: read [{begin}, {end}): {e}", p.display()))?;
    Ok(raw)
}

#[allow(clippy::too_many_arguments)]
fn encode_tensor(
    m: &Manifest,
    t: &TensorEntry,
    raw: Vec<u8>,
    scale_raw: Option<Vec<u8>>,
    offset: u64,
    mode: ScalesMode,
    scales_mode_str: &str,
    diag: Option<&DiagStats>,
) -> Result<Encoded, String> {
    // C6: the source dtype comes from the shard header. It used to be guessed from the
    // byte length, which read an F16 tensor as BF16; for the BF16/F32/I64 sources both
    // checkpoints of record carry, the two agree.
    let dt_in = t.src_dtype.as_str();
    let section = t.decision.section;
    let mut entry_json = serde_json::json!({
        "name": t.name, "shape": t.shape, "section": section,
        "n_values": t.n_values, "offset": offset,
    });
    if t.decision.dtype == recipe::DtypeOut::Nvfp4 {
        let values = if dt_in == "F8_E4M3" {
            // #155: FP8 E4M3 with 128x128 block scales, one f32 multiply per value (`fp8.rs`)
            let s = t.scale.as_ref().ok_or(format!("{}: FP8 without a scale", t.name))?;
            let sc = fp8::scales_to_f32(&scale_raw.ok_or(format!("{}: scale bytes not read", t.name))?, &s.dtype)?;
            fp8::dequant_fp8_block(&t.name, &raw, t.shape[0], t.shape[1], &sc)?
        } else {
            bytes_to_f32(&raw, dt_in)
        };
        let w = match (mode, diag) {
            (ScalesMode::Diag, Some(ds)) => {
                let cols = *t.shape.last().expect("an nvfp4 tensor has a shape");
                Some((ds.weights_for(&t.name, cols)?, cols))
            }
            _ => None,
        };
        // #177: the family's scale cap (0x7E for glm5_next: no E4M3 NaN code in its containers)
        let (blocks, global, stats, sse_ceil) = quantize_nvfp4_cap(&values, mode, w, m.family.scale_byte_max());
        let mse = stats.sum_sq_err / t.n_values as f64;
        let mse_ceil = sse_ceil / t.n_values as f64;
        let mse_ratio = if mse_ceil > 0.0 { mse / mse_ceil } else { 1.0 };
        entry_json["dtype"] = serde_json::Value::from("nvfp4");
        entry_json["global_scale"] = serde_json::Value::from(global);
        entry_json["len"] = serde_json::Value::from(blocks.len() as u64);
        // #155 (b): the code histograms of the WRITTEN bytes (`entropy.rs`)
        let h = entropy::Hist::of_blocks(&blocks);
        let (hc, hs) = h.to_json();
        let (class, layer, expert) = recipe::tensor_class(m.family, &t.name, &m.config);
        let sidecar = serde_json::json!({
            "name": t.name, "section": section, "dtype": "nvfp4",
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
            "class": class, "layer": layer, "expert": expert,
            "h_codes": hc, "h_scales": hs,
        });
        let acc = serde_json::json!({
            "nvfp4": true, "section": section, "n": t.n_values, "sse": stats.sum_sq_err, "sse_ceil": sse_ceil,
            "clipped": stats.clipped, "violations": stats.violations,
        });
        Ok(Encoded { bytes: blocks, entry: entry_json, sidecar, acc })
    } else {
        // bf16 and i64: the source's raw bytes (`recipe::decide` guarantees a bf16 keep
        // has a BF16 source). f32 (C6, the dense row's A_log): an F32 source is carried,
        // a BF16 source is widened exactly (bf16 is the top half of an f32).
        let out_dtype = t.decision.dtype;
        let payload: Vec<u8> = if out_dtype == recipe::DtypeOut::F32 && dt_in == "BF16" {
            bytes_to_f32(&raw, "BF16").iter().flat_map(|v| v.to_le_bytes()).collect()
        } else {
            raw
        };
        assert_eq!(payload.len() as u64, out_dtype.bytes(t.n_values), "{}: payload length", t.name);
        entry_json["dtype"] = serde_json::Value::from(out_dtype.as_str());
        entry_json["len"] = serde_json::Value::from(payload.len() as u64);
        let sidecar = serde_json::json!({
            "name": t.name, "section": section, "dtype": out_dtype.as_str(), "n": t.n_values,
        });
        Ok(Encoded { bytes: payload, entry: entry_json, sidecar, acc: serde_json::json!({ "nvfp4": false }) })
    }
}

/// The running totals of a conversion, rebuilt from the journal on a resume.
struct Totals {
    nvfp4_count: usize,
    bf16_count: usize,
    total_violations: u64,
    section_agg: BTreeMap<String, SectAgg>,
    hists: Vec<entropy::TensorHist>,
}

impl Totals {
    fn absorb(&mut self, acc: &serde_json::Value, sidecar: &serde_json::Value) {
        if acc["nvfp4"] == true {
            self.nvfp4_count += 1;
            self.total_violations += acc["violations"].as_u64().unwrap_or(0);
            let agg = self.section_agg.entry(acc["section"].as_str().unwrap_or("").to_string()).or_insert(SectAgg {
                tensors: 0,
                n: 0,
                sse_new: 0.0,
                sse_ceil: 0.0,
                clipped: 0,
            });
            agg.tensors += 1;
            agg.n += acc["n"].as_u64().unwrap_or(0);
            agg.sse_new += acc["sse"].as_f64().unwrap_or(0.0);
            agg.sse_ceil += acc["sse_ceil"].as_f64().unwrap_or(0.0);
            agg.clipped += acc["clipped"].as_u64().unwrap_or(0);
            if let Some(h) = entropy::Hist::from_json(&sidecar["h_codes"], &sidecar["h_scales"]) {
                self.hists.push(entropy::TensorHist {
                    class: sidecar["class"].as_str().unwrap_or("rest").to_string(),
                    layer: sidecar["layer"].as_u64(),
                    expert: sidecar["expert"].as_u64(),
                    hist: h,
                });
            }
        } else {
            self.bf16_count += 1;
        }
    }
}

fn write_line(w: &mut impl Write, v: &serde_json::Value) -> std::io::Result<()> {
    serde_json::to_writer(&mut *w, v)?;
    writeln!(w)
}

/// #155 (d): the journal of a conversion, `<out>.cnq.journal.jsonl`. Line 1 names the plan (the
/// recipe, the scale policy and the sha256 of the write order); then one record per written
/// tensor, appended only after the container bytes are synced, and synced itself. A resume
/// keeps the records whose bytes re-hash correctly (in order, up to the first that does not),
/// truncates the container behind the last one, and rewrites the sidecar from them.
struct Resumed {
    records: Vec<serde_json::Value>,
    journal_len: u64,
}

fn read_journal(journal: &std::path::Path, out: &std::path::Path, head: &serde_json::Value, order: &[&TensorEntry], unit_pad_at: &[bool]) -> Result<Resumed, String> {
    use sha2::Digest;
    let text = std::fs::read(journal).map_err(|e| format!("{}: {e}", journal.display()))?;
    let mut pos = 0usize;
    let mut lines = Vec::new();
    while let Some(nl) = text[pos..].iter().position(|b| *b == b'\n') {
        lines.push((pos, pos + nl + 1));
        pos += nl + 1;
    }
    let Some(&(h0, h1)) = lines.first() else {
        return Ok(Resumed { records: vec![], journal_len: 0 });
    };
    let got: serde_json::Value = serde_json::from_slice(&text[h0..h1]).map_err(|e| format!("{}: line 1: {e}", journal.display()))?;
    if got != *head {
        return Err(format!(
            "{} belongs to another conversion ({} vs {}); remove it and {} to start over",
            journal.display(),
            got,
            head,
            out.display()
        ));
    }
    let mut f = std::fs::File::open(out).map_err(|e| format!("{}: {e}", out.display()))?;
    let flen = f.metadata().map_err(|e| e.to_string())?.len();
    let mut records = Vec::new();
    let mut journal_len = h1 as u64;
    let mut blob = 0u64;
    for &(a, b) in &lines[1..] {
        let Ok(r) = serde_json::from_slice::<serde_json::Value>(&text[a..b]) else { break };
        let k = records.len();
        let (Some(off), Some(len), Some(pad), Some(sha)) = (r["entry"]["offset"].as_u64(), r["entry"]["len"].as_u64(), r["pad"].as_u64(), r["sha256"].as_str()) else { break };
        if k >= order.len() || r["entry"]["name"] != order[k].name.as_str() || r["seq"] != k as u64 || off != blob + pad || (pad > 0 && !unit_pad_at[k]) {
            break;
        }
        if 12 + off + len > flen {
            break;
        }
        let mut buf = vec![0u8; (pad + len) as usize];
        f.seek(SeekFrom::Start(12 + blob)).map_err(|e| e.to_string())?;
        f.read_exact(&mut buf).map_err(|e| e.to_string())?;
        if buf[..pad as usize].iter().any(|x| *x != 0) {
            break;
        }
        let h: String = sha2::Sha256::digest(&buf[pad as usize..]).iter().map(|x| format!("{x:02x}")).collect();
        if h != sha {
            break;
        }
        blob = off + len;
        journal_len = b as u64;
        records.push(r);
    }
    Ok(Resumed { records, journal_len })
}

/// Wait for `<dir>/<shard>.verified` (written by the downloader after size + sha256), then check
/// the file's size against the recorded one and the marker's sha256 against the HF LFS record.
fn wait_verified(dir: &std::path::Path, shard: &str, want: Option<&(String, u64)>, header_size: Option<u64>, poll: std::time::Duration) -> Result<(), String> {
    let marker = dir.join(format!("{shard}.verified"));
    let mut said = false;
    while !marker.exists() {
        if !said {
            eprintln!("consume: waiting for {}", marker.display());
            said = true;
        }
        std::thread::sleep(poll);
    }
    let v: serde_json::Value = std::fs::read(&marker).ok().and_then(|b| serde_json::from_slice(&b).ok()).unwrap_or(serde_json::Value::Null);
    if let (Some(a), Some((b, _))) = (v["sha256"].as_str(), want) {
        if a != b {
            return Err(format!("{}: sha256 {a}, Hugging Face recorded {b} - not converting this shard", marker.display()));
        }
    }
    let p = dir.join(shard);
    let size = std::fs::metadata(&p).map_err(|e| format!("{}: {e} (it has a .verified marker)", p.display()))?.len();
    for want_size in [want.map(|w| w.1), header_size].into_iter().flatten() {
        if size != want_size {
            return Err(format!("{}: {size} B, the record says {want_size} B - not converting this shard", p.display()));
        }
    }
    Ok(())
}

fn convert_with(input: &std::path::Path, out_path: &std::path::Path, mode: ScalesMode, prov: &Provenance, diag: Option<&DiagStats>, opts: &ConvertOpts) -> i32 {
    if (mode == ScalesMode::Diag) != diag.is_some() {
        eprintln!("conversion refused: --scales diag and --diag-stats go together\n{HELP}");
        return 2;
    }
    if opts.consume.is_some() && opts.headers.is_none() {
        eprintln!("conversion refused: --consume reads the shard headers from --headers <dir>; the shards are not all there to read them from\n{HELP}");
        return 2;
    }
    let scales_mode_str = match mode {
        ScalesMode::Ceil => "ceil",
        ScalesMode::Mse => "mse",
        ScalesMode::Diag => "diag",
    };
    let t_start = std::time::Instant::now();

    // ---- manifest: scan headers only (fast), collect every tensor's location ----
    let mut m = match build_manifest_from(input, opts.headers.as_deref()) {
        Ok(m) => m,
        Err(e) => {
            eprintln!("conversion refused: {e}");
            return 2;
        }
    };
    // ---- #156: a partial container keeps the filtered tensors only. The manifest above saw
    // every tensor (recipe whitelist, FP8 pairing, geometry); what the filter drops is named
    // as filtered, weight and block scale, never as missing ----
    let mut filtered: std::collections::HashSet<String> = std::collections::HashSet::new();
    if let Some(f) = &opts.filter {
        let before = m.tensors.len();
        for t in m.tensors.iter().filter(|t| !f.keeps(&t.name)) {
            filtered.insert(t.name.clone());
            if let Some(s) = &t.scale {
                filtered.insert(s.name.clone());
            }
        }
        m.tensors.retain(|t| f.keeps(&t.name));
        for l in &f.layers {
            if !m.tensors.iter().any(|t| partial::LayerFilter::layer_of(&t.name) == Some(*l)) {
                eprintln!("conversion refused: {}: layer {l} has no tensor the {} recipe writes", f.describe(), m.family.recipe());
                return 2;
            }
        }
        if f.embed_head_norm && !m.tensors.iter().any(|t| partial::LayerFilter::is_embed_head_norm(&t.name)) {
            eprintln!("conversion refused: {}: no token embedding, lm_head or final norm among the tensors", f.describe());
            return 2;
        }
        eprintln!("partial container {}: {} of {before} tensors kept, {} weight_map names filtered (block scales included)", f.describe(), m.tensors.len(), filtered.len());
    }
    // ---- the index v2 `model` block: provenance first, so a missing flag costs nothing ----
    let (repo, revision, shards, _) = match provenance(&m, prov, true) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("conversion refused: {e}");
            return 2;
        }
    };
    let api = match recipe::read_hf_api_info(&m.model_dir) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("conversion refused: {e}");
            return 2;
        }
    };
    let model = recipe::ModelSource {
        family: m.family,
        config_json: m.config_json.clone(),
        generation_config_json: m.generation_config_json.clone(),
        config: m.config.clone(),
        repo,
        revision,
        shards,
    };
    let tensors = &m.tensors;
    eprintln!(
        "manifest: {} tensors from {} shard file(s), {:.1} GB to read — family {}, recipe {}",
        tensors.len(),
        m.shard_files.len(),
        tensors.iter().map(|t| t.data_end - t.data_begin).sum::<u64>() as f64 / 1e9,
        m.family.name(),
        m.family.recipe()
    );

    // ---- the write order (#155): units, and per tensor whether a unit starts there ----
    let units = write_units(&m);
    let mut order: Vec<&TensorEntry> = Vec::with_capacity(tensors.len());
    let mut unit_start: Vec<Option<u64>> = Vec::with_capacity(tensors.len()); // Some(align) at a unit's first tensor
    for u in &units {
        for (k, &i) in u.tensors.iter().enumerate() {
            order.push(&tensors[i]);
            unit_start.push(if k == 0 { Some(u.align) } else { None });
        }
    }
    let unit_pad_at: Vec<bool> = unit_start.iter().map(|a| a.is_some_and(|x| x > 0)).collect();
    // the last write position that reads each shard: `<shard>.done` once it is journalled
    let mut last_reader: BTreeMap<&str, usize> = BTreeMap::new();
    for (k, t) in order.iter().enumerate() {
        for s in shards_of(t) {
            last_reader.insert(s, k);
        }
    }
    let plan_sha = {
        let mut s = String::new();
        for t in &order {
            s.push_str(&format!("{}\t{}\t{}\t{}\n", t.name, t.shard, t.data_begin, t.data_end));
        }
        if let Some(d) = diag {
            s.push_str(&format!("diag {}\n", d.stats_sha256));
        }
        recipe::sha256_hex(s.as_bytes())
    };
    let mut head = serde_json::json!({
        "journal": "crow-nest converter", "version": 1, "recipe": m.family.recipe(), "scales": scales_mode_str,
        "tensors": order.len(), "order_sha256": plan_sha,
    });
    if let Some(f) = &opts.filter {
        head["partial"] = serde_json::Value::from(f.describe());
    }

    // ---- write container: magic, streamed blob, index trailer; resume from the journal ----
    let journal_path = out_path.with_extension("cnq.journal.jsonl");
    let sidecar_path = out_path.with_extension("cnq.sidecar.jsonl");
    let blob_start: u64 = 12;
    let resumed = if journal_path.exists() && out_path.exists() {
        match read_journal(&journal_path, out_path, &head, &order, &unit_pad_at) {
            Ok(r) => r,
            Err(e) => {
                eprintln!("conversion refused: {e}");
                return 2;
            }
        }
    } else {
        Resumed { records: vec![], journal_len: 0 }
    };
    let mut blob_len: u64 = resumed.records.last().map(|r| r["entry"]["offset"].as_u64().unwrap() + r["entry"]["len"].as_u64().unwrap()).unwrap_or(0);
    let (mut out, mut journal) = if resumed.journal_len > 0 {
        eprintln!("resume: {} of {} tensors journalled and re-hashed, container truncated to {} B", resumed.records.len(), order.len(), blob_start + blob_len);
        let mut out = std::fs::OpenOptions::new().read(true).write(true).open(out_path).expect("open output");
        out.set_len(blob_start + blob_len).expect("truncate output");
        out.seek(SeekFrom::End(0)).expect("seek output");
        let mut journal = std::fs::OpenOptions::new().read(true).write(true).open(&journal_path).expect("open journal");
        journal.set_len(resumed.journal_len).expect("truncate journal");
        journal.seek(SeekFrom::End(0)).expect("seek journal");
        (out, journal)
    } else {
        let mut out = std::fs::File::create(out_path).expect("create output");
        out.write_all(MAGIC).expect("magic");
        out.write_all(&0u64.to_le_bytes()).expect("reserved");
        let mut journal = std::fs::File::create(&journal_path).expect("create journal");
        write_line(&mut journal, &head).expect("journal head");
        journal.sync_data().expect("sync journal");
        (out, journal)
    };

    let mut index_tensors: Vec<serde_json::Value> = Vec::new();
    let mut sidecar = std::io::BufWriter::new(std::fs::File::create(&sidecar_path).expect("sidecar"));
    let mut tot = Totals { nvfp4_count: 0, bf16_count: 0, total_violations: 0, section_agg: BTreeMap::new(), hists: Vec::new() };
    for r in &resumed.records {
        tot.absorb(&r["acc"], &r["sidecar"]);
        write_line(&mut sidecar, &r["sidecar"]).expect("sidecar line");
        index_tensors.push(r["entry"].clone());
    }
    let mut pad_total: u64 = resumed.records.iter().map(|r| r["pad"].as_u64().unwrap_or(0)).sum();

    // ---- consume mode: which shards are verified, which are done ----
    let mut verified: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let mut done: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
    let shard_names: Vec<String> = m.shard_files.iter().map(|p| p.file_name().unwrap().to_string_lossy().to_string()).collect();
    let mark_done = |k_written: Option<usize>, done: &mut std::collections::BTreeSet<String>| -> Result<(), String> {
        let Some(dir) = &opts.consume else { return Ok(()) };
        for s in &shard_names {
            if done.contains(s) {
                continue;
            }
            let ready = match last_reader.get(s.as_str()) {
                Some(&last) => k_written.is_some_and(|k| k >= last),
                None => dir.join(format!("{s}.verified")).exists(), // nothing of it is written
            };
            if ready {
                let p = dir.join(format!("{s}.done"));
                std::fs::write(&p, serde_json::to_vec(&serde_json::json!({ "shard": s, "out": out_path.display().to_string() })).unwrap())
                    .map_err(|e| format!("{}: {e}", p.display()))?;
                done.insert(s.clone());
            }
        }
        Ok(())
    };
    if let Err(e) = mark_done(resumed.records.len().checked_sub(1), &mut done) {
        eprintln!("conversion refused: {e}");
        return 2;
    }

    let src_dir = opts.consume.clone().unwrap_or_else(|| input.to_path_buf());
    let single = if m.single_file { Some(input) } else { None };
    for k in resumed.records.len()..order.len() {
        let t = order[k];
        let fail = |e: String| -> i32 {
            eprintln!("conversion stopped at {} ({}/{}): {e} - the journal keeps what is written; run again to resume", t.name, k + 1, order.len());
            2
        };
        if let Some(dir) = &opts.consume {
            for s in shards_of(t) {
                if verified.contains(s) {
                    continue;
                }
                if let Err(e) = wait_verified(dir, s, api.as_ref().and_then(|a| a.lfs.get(s)), m.shard_size.get(s).copied(), opts.poll) {
                    return fail(e);
                }
                verified.insert(s.to_string());
            }
        }
        let raw = match read_source(&src_dir, single, &t.shard, t.data_begin, t.data_end) {
            Ok(r) => r,
            Err(e) => return fail(e),
        };
        let scale_raw = match &t.scale {
            Some(s) => match read_source(&src_dir, single, &s.shard, s.data_begin, s.data_end) {
                Ok(r) => Some(r),
                Err(e) => return fail(e),
            },
            None => None,
        };
        let pad = pad_for(blob_len, unit_start[k].unwrap_or(0));
        let enc = match encode_tensor(&m, t, raw, scale_raw, blob_len + pad, mode, scales_mode_str, diag) {
            Ok(e) => e,
            Err(e) => {
                eprintln!("conversion refused: {e}");
                return 2;
            }
        };
        if pad > 0 {
            out.write_all(&vec![0u8; pad as usize]).expect("write alignment zeros");
        }
        out.write_all(&enc.bytes).expect("write tensor");
        out.sync_data().expect("sync output");
        blob_len += pad + enc.bytes.len() as u64;
        pad_total += pad;
        let rec = serde_json::json!({
            "seq": k, "pad": pad, "sha256": recipe::sha256_hex(&enc.bytes),
            "entry": enc.entry, "sidecar": enc.sidecar, "acc": enc.acc,
        });
        write_line(&mut journal, &rec).expect("journal line");
        journal.sync_data().expect("sync journal");
        tot.absorb(&rec["acc"], &rec["sidecar"]);
        write_line(&mut sidecar, &rec["sidecar"]).expect("sidecar line");
        index_tensors.push(rec["entry"].clone());
        if let Err(e) = mark_done(Some(k), &mut done) {
            return fail(e);
        }
        if k % 100 == 0 {
            eprintln!(
                "[{:>5}/{}] {} ({}) — {:.2} GB written, {:.0} s",
                k + 1,
                order.len(),
                t.name,
                t.decision.section,
                blob_len as f64 / 1e9,
                t_start.elapsed().as_secs_f64()
            );
        }
        if opts.stop_after == Some(k + 1) {
            sidecar.flush().ok();
            return 4;
        }
    }
    // per-section verification report (nvfp4 only): aggregate MSE of the written
    // encoding vs the ceiling-scale reference on the SAME weights + clipped counts
    for (section, a) in &tot.section_agg {
        let mse = a.sse_new / a.n as f64;
        let mse_ceil = a.sse_ceil / a.n as f64;
        let mse_ratio = if mse_ceil > 0.0 { mse / mse_ceil } else { 1.0 };
        println!(
            "section {section}: {} nvfp4 tensors — MSE {mse:.4e} vs ceil {mse_ceil:.4e} (ratio {mse_ratio:.4}), clipped {}",
            a.tensors, a.clipped
        );
        write_line(
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
    }
    // #155 (c): the exact static order-0 coder sizes, tables included, from the histograms
    for r in entropy::code_summary(&tot.hists) {
        if r["scope"] == "class" || r["scope"] == "expert_blocks" {
            println!(
                "code {:<13} {:<14} raw {:>15} B, coded {:>15} B incl. {} B tables: saving {:>7.3} % (order-0 entropy bound {:.3} %)",
                r["scope"].as_str().unwrap_or(""),
                r["class"].as_str().unwrap_or("all routed"),
                r["raw_bytes"],
                r["coded_bytes"],
                r["table_bytes"],
                100.0 * r["saving"].as_f64().unwrap_or(0.0),
                100.0 * r["saving_entropy"].as_f64().unwrap_or(0.0)
            );
        }
        write_line(&mut sidecar, &r).expect("sidecar code summary line");
    }
    sidecar.flush().ok();

    // coverage check in dir mode: every tensor the model index knows must be in the output,
    // be the block scale of one that is (#155), or be one the recipe omits by name
    // (`recipe::omitted`, Crow #300: the dense vision tower)
    if let Some(wm) = &m.weight_map {
        let mut covered: std::collections::HashSet<&str> = tensors.iter().map(|t| t.name.as_str()).collect();
        covered.extend(tensors.iter().filter_map(|t| t.scale.as_ref().map(|s| s.name.as_str())));
        let (mut missing, mut omitted, mut n_filtered) = (0usize, 0usize, 0usize);
        for name in wm.keys() {
            if !covered.contains(name.as_str()) {
                if recipe::omitted(m.family, name).is_some() {
                    omitted += 1;
                    continue;
                }
                if filtered.contains(name) {
                    n_filtered += 1;
                    continue;
                }
                eprintln!("MISSING from output: {name}");
                missing += 1;
            }
        }
        if missing > 0 {
            eprintln!("coverage check FAILED: {missing} tensors missing");
            return 3;
        }
        match &opts.filter {
            None => eprintln!("coverage check: all {} weight_map tensors present ({omitted} omitted by the {} recipe)", wm.len(), m.family.recipe()),
            Some(f) => eprintln!(
                "coverage check (PARTIAL container, {}): {} of {} weight_map tensors present, {n_filtered} filtered by the flags, {omitted} omitted by the {} recipe, 0 missing",
                f.describe(),
                wm.len() - n_filtered - omitted,
                wm.len(),
                m.family.recipe()
            ),
        }
    }

    let mut index = index_v2(&model, scales_mode_str, blob_start, index_tensors);
    if let Some(f) = &opts.filter {
        index["partial"] = f.index_block(tensors.len(), filtered.len());
    }
    let index_json = serde_json::to_vec_pretty(&index).expect("index json");
    out.write_all(&index_json).expect("index");
    out.write_all(&(index_json.len() as u64).to_le_bytes()).expect("index len");
    out.flush().ok();
    out.sync_all().expect("sync output");
    drop(journal);
    std::fs::remove_file(&journal_path).ok();

    println!(
        "wrote {}: {} tensors ({} nvfp4, {} bf16-keep), payload {:.2} GB, alignment zeros {pad_total} B, scales {scales_mode_str}, \
violations {}, elapsed {:.0} s",
        out_path.display(),
        tensors.len(),
        tot.nvfp4_count,
        tot.bf16_count,
        blob_len as f64 / 1e9,
        tot.total_violations,
        t_start.elapsed().as_secs_f64()
    );
    let total_violations = tot.total_violations;
    if total_violations > 0 {
        if mode != ScalesMode::Ceil {
            // the old per-element relative bound is void BY DESIGN here (clipping
            // allowed); quality is carried by the MSE fields of the report above
            eprintln!(
                "NOTE: {total_violations} elements exceed the old max-rel bound — \
expected under --scales mse, see the MSE report"
            );
        } else {
            eprintln!("WARNING: {total_violations} bound violations — check the sidecar");
            return 1;
        }
    }
    0
}

/// The index v2 trailer (Crow #300 C6). What changed against v1 (`64c242b`):
///
/// - `format_version: 2` replaces `version: 1`; the engine reads a v1 index only for the one
///   container of record (`engine/src/cnq.rs`, `CNQ45M_INDEX_SHA256`);
/// - `recipe` names the family row that decided every tensor's dtype, `scales` the sub-block
///   scale policy;
/// - `model` carries the checkpoint: `config.json` and `generation_config.json` as verbatim
///   strings with their sha256, the family, the derived geometry and the source repo, revision
///   and per-shard sha256 (`recipe::ModelSource::model_block`);
/// - `sections` lists only the sections the container has (a dense model has no `ple`).
///
/// The tensor records, the block geometry and `blob_offset` are unchanged.
fn index_v2(model: &recipe::ModelSource, scales: &str, blob_offset: u64, tensors: Vec<serde_json::Value>) -> serde_json::Value {
    let present = |s: &str| tensors.iter().any(|t| t["section"] == s);
    let mut sections = serde_json::Map::new();
    if present("ple") {
        sections.insert("ple".into(), serde_json::json!({ "format": "nvfp4", "exchangeable_to": "fp8" }));
    }
    if present("vit") {
        sections.insert("vit".into(), serde_json::json!({ "optional_to_load": true }));
    }
    if present("mtp") {
        sections.insert("mtp".into(), serde_json::json!({ "optional_to_load": true }));
    }
    serde_json::json!({
        "format": "crow-nest-quant",
        "format_version": 2,
        "recipe": model.family.recipe(),
        "scales": scales,
        "model": model.model_block(),
        "block_geometry": { "values": 64, "sub_block": 16, "bytes_per_block": 36, "bpw": 4.5 },
        "sections": sections,
        "blob_offset": blob_offset,
        "tensors": tensors,
    })
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

    // ---- Crow #300 C6: the index v2 round trip ----

    /// The synthetic dense model of the index v2 fixture: every row of the dense recipe once
    /// (nvfp4, bf16 keeps, the f32 A_log, the mtp and vit sections), BF16 values from a fixed
    /// LCG. Returns (name, shape) in file order.
    fn synthetic_v2_tensors() -> Vec<(&'static str, Vec<usize>)> {
        vec![
            ("model.language_model.embed_tokens.weight", vec![128, 64]),
            ("model.language_model.layers.0.input_layernorm.weight", vec![64]),
            ("model.language_model.layers.0.linear_attn.A_log", vec![2]),
            ("model.language_model.layers.0.linear_attn.dt_bias", vec![2]),
            ("model.language_model.layers.0.linear_attn.conv1d.weight", vec![128, 1, 4]),
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", vec![2, 64]),
            ("model.language_model.layers.0.linear_attn.in_proj_qkv.weight", vec![128, 64]),
            ("model.language_model.layers.0.linear_attn.out_proj.weight", vec![64, 64]),
            ("model.language_model.layers.0.mlp.gate_proj.weight", vec![128, 64]),
            ("model.language_model.layers.1.self_attn.q_proj.weight", vec![128, 64]),
            ("model.language_model.layers.1.self_attn.k_norm.weight", vec![32]),
            ("model.language_model.layers.1.mlp.down_proj.weight", vec![64, 128]),
            ("model.language_model.norm.weight", vec![64]),
            ("lm_head.weight", vec![128, 64]),
            ("mtp.fc.weight", vec![64, 128]),
            ("model.visual.blocks.0.attn.qkv.weight", vec![192, 64]),
        ]
    }

    /// Write the synthetic model as one BF16 `.safetensors` file.
    fn write_synthetic_safetensors(path: &std::path::Path) {
        let mut header = serde_json::Map::new();
        let mut data: Vec<u8> = Vec::new();
        let mut x: u32 = 0xC6C6_0300;
        for (name, shape) in synthetic_v2_tensors() {
            let n: usize = shape.iter().product();
            let begin = data.len();
            for _ in 0..n {
                x = x.wrapping_mul(1664525).wrapping_add(1013904223);
                let v = (((x >> 8) % 2001) as f32 - 1000.0) / 4000.0;
                data.extend_from_slice(&((v.to_bits() >> 16) as u16).to_le_bytes());
            }
            header.insert(name.into(), serde_json::json!({ "dtype": "BF16", "shape": shape, "data_offsets": [begin, data.len()] }));
        }
        header.insert("__metadata__".into(), serde_json::json!({ "format": "pt" }));
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&data).unwrap();
    }

    fn fixture_dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../engine/tests/fixtures/synthetic-v2")
    }

    fn trailer(bytes: &[u8]) -> serde_json::Value {
        let n = u64::from_le_bytes(bytes[bytes.len() - 8..].try_into().unwrap()) as usize;
        serde_json::from_slice(&bytes[bytes.len() - 8 - n..bytes.len() - 8]).unwrap()
    }

    /// A synthetic v2 container written by `convert` carries both config files byte for byte,
    /// with their sha256, the family, the recipe, the geometry and the provenance; the f32 row
    /// is the exact widening of the BF16 source; and the container is byte-identical to the
    /// engine's fixture `engine/tests/fixtures/synthetic-v2/synthetic-v2.cnq`, which
    /// `engine/src/cnq.rs` reads back. That pair is the round trip across the two crates.
    /// `CNQ_C6_REGEN=1` rewrites the fixture instead of comparing.
    #[test]
    fn a_synthetic_v2_container_carries_its_config_verbatim_and_is_the_engine_fixture() {
        let dir = std::env::temp_dir().join(format!("cnq-c6-v2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = std::fs::read(fixture_dir().join("config.json")).unwrap();
        let gen = std::fs::read(fixture_dir().join("generation_config.json")).unwrap();
        std::fs::write(dir.join("config.json"), &cfg).unwrap();
        std::fs::write(dir.join("generation_config.json"), &gen).unwrap();
        let st = dir.join("model.safetensors");
        write_synthetic_safetensors(&st);
        let out = dir.join("synthetic-v2.cnq");
        let prov = Provenance { repo: Some("crow-nest/synthetic-v2".into()), revision: Some("c6".into()) };
        assert_eq!(convert(&st, &out, ScalesMode::Ceil, &prov, None), 0);
        let bytes = std::fs::read(&out).unwrap();
        assert_eq!(&bytes[..4], b"CNQ1");
        let idx = trailer(&bytes);
        assert_eq!(idx["format_version"], 2);
        assert!(idx.get("version").is_none());
        assert_eq!(idx["recipe"], "cnq4.5-qwen35-dense");
        let m = &idx["model"];
        assert_eq!(m["family"], "Qwen35Dense");
        assert_eq!(m["config_json"].as_str().unwrap().as_bytes(), cfg.as_slice());
        assert_eq!(m["generation_config_json"].as_str().unwrap().as_bytes(), gen.as_slice());
        assert_eq!(m["config_json_sha256"], recipe::sha256_hex(&cfg));
        assert_eq!(m["generation_config_json_sha256"], recipe::sha256_hex(&gen));
        assert_eq!(m["geo"]["hidden"], 64);
        assert_eq!(m["geo"]["gqa"], 2);
        assert_eq!(m["source"]["repo"], "crow-nest/synthetic-v2");
        assert_eq!(m["source"]["revision"], "c6");
        let sh = &m["source"]["shards"][0];
        assert_eq!(sh["sha256"], recipe::sha256_file(&st).unwrap());
        assert_eq!(sh["sha256_from"], "computed");
        // the vision tower is omitted by the dense row: no section, no record (#300 projector decision)
        assert_eq!(idx["sections"].as_object().unwrap().keys().collect::<Vec<_>>(), ["mtp"]);
        // dtypes per the dense row, and the f32 A_log is the BF16 source widened exactly
        let ts = idx["tensors"].as_array().unwrap();
        let get = |n: &str| ts.iter().find(|t| t["name"] == n).unwrap();
        assert_eq!(get("lm_head.weight")["dtype"], "nvfp4");
        assert_eq!(get("model.language_model.layers.0.linear_attn.in_proj_a.weight")["dtype"], "bf16");
        assert_eq!(get("model.language_model.layers.0.linear_attn.conv1d.weight")["dtype"], "bf16");
        assert_eq!(get("mtp.fc.weight")["section"], "mtp");
        assert!(ts.iter().all(|t| t["section"] != "vit" && !t["name"].as_str().unwrap().contains("visual")));
        let a = get("model.language_model.layers.0.linear_attn.A_log");
        assert_eq!((a["dtype"].as_str(), a["len"].as_u64()), (Some("f32"), Some(8)));
        let (hdr, start) = read_safetensors_header(&st).unwrap();
        let src_off = start + hdr["model.language_model.layers.0.linear_attn.A_log"]["data_offsets"][0].as_u64().unwrap();
        let src = std::fs::read(&st).unwrap();
        let at = (12 + a["offset"].as_u64().unwrap()) as usize;
        for i in 0..2 {
            let b = u16::from_le_bytes([src[src_off as usize + 2 * i], src[src_off as usize + 2 * i + 1]]);
            let w = u32::from_le_bytes(bytes[at + 4 * i..at + 4 * i + 4].try_into().unwrap());
            assert_eq!(w, (b as u32) << 16, "A_log[{i}]");
        }
        // the engine fixture
        let fx = fixture_dir().join("synthetic-v2.cnq");
        if std::env::var("CNQ_C6_REGEN").as_deref() == Ok("1") {
            std::fs::write(&fx, &bytes).unwrap();
        }
        let want = std::fs::read(&fx).expect("engine/tests/fixtures/synthetic-v2/synthetic-v2.cnq (CNQ_C6_REGEN=1 writes it)");
        assert!(want == bytes, "the converter no longer writes the engine's fixture byte for byte ({} vs {} B)", bytes.len(), want.len());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// No config, no container: an index v2 carries the checkpoint's config, so a conversion
    /// without `config.json` beside the input is refused before anything is written; so is one
    /// without `--source-repo`.
    #[test]
    fn a_conversion_without_config_or_source_repo_is_refused() {
        let dir = std::env::temp_dir().join(format!("cnq-c6-noconf-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let st = dir.join("model.safetensors");
        write_synthetic_safetensors(&st);
        let out = dir.join("x.cnq");
        let prov = Provenance { repo: Some("r".into()), revision: Some("v".into()) };
        let m = build_manifest(&st).err().unwrap();
        assert!(m.contains("carries the checkpoint's config.json"), "{m}");
        assert_eq!(convert(&st, &out, ScalesMode::Ceil, &prov, None), 2);
        assert!(!out.exists());
        std::fs::copy(fixture_dir().join("config.json"), dir.join("config.json")).unwrap();
        std::fs::copy(fixture_dir().join("generation_config.json"), dir.join("generation_config.json")).unwrap();
        let m = build_manifest(&st).unwrap();
        let e = provenance(&m, &Provenance { repo: None, revision: Some("v".into()) }, false).err().unwrap();
        assert!(e.contains("--source-repo"), "{e}");
        let e = provenance(&m, &Provenance { repo: Some("r".into()), revision: None }, false).err().unwrap();
        assert!(e.contains("no --revision"), "{e}");
        assert_eq!(convert(&st, &out, ScalesMode::Ceil, &Provenance::default(), None), 2);
        assert!(!out.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// Dir mode (a `model.safetensors.index.json` beside the shards) runs the coverage check
    /// against the checkpoint's `weight_map`. A tensor the recipe omits (`recipe::omitted`: the
    /// dense vision tower) is absent from the output on purpose and must not fail it; the 27B
    /// conversion of 2026-09-26 exited 3 with "333 tensors missing" before this was known.
    #[test]
    fn a_dir_conversion_passes_the_coverage_check_with_the_omitted_vision_tower() {
        let dir = std::env::temp_dir().join(format!("cnq-300-omit-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::copy(fixture_dir().join("config.json"), dir.join("config.json")).unwrap();
        std::fs::copy(fixture_dir().join("generation_config.json"), dir.join("generation_config.json")).unwrap();
        write_synthetic_safetensors(&dir.join("model.safetensors"));
        let wm: serde_json::Map<String, serde_json::Value> =
            synthetic_v2_tensors().iter().map(|(n, _)| (n.to_string(), "model.safetensors".into())).collect();
        assert!(wm.keys().any(|n| recipe::omitted(recipe::Family::Qwen35Dense, n).is_some()), "the fixture carries a vision tensor");
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({ "weight_map": wm })).unwrap()).unwrap();
        let out = dir.join("x.cnq");
        let prov = Provenance { repo: Some("crow-nest/synthetic-v2".into()), revision: Some("c6".into()) };
        assert_eq!(convert(&dir, &out, ScalesMode::Ceil, &prov, None), 0);
        let idx = trailer(&std::fs::read(&out).unwrap());
        assert!(idx["tensors"].as_array().unwrap().iter().all(|t| !t["name"].as_str().unwrap().contains("visual")));
        std::fs::remove_dir_all(&dir).ok();
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

    fn weighted_err(sub: &[f32], s: f32, d: &[f32]) -> f32 {
        let inv = 1.0 / s;
        sub.iter().zip(d).fold(0.0f32, |a, (v, w)| {
            let e = quant_dequant(*v, s, inv) - *v;
            a + e * e * *w
        })
    }

    fn lcg_values(n: usize, seed: u32) -> Vec<f32> {
        let mut x = seed;
        (0..n).map(|_| {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            let m = ((x >> 8) % 1000) as f32 / 1000.0;
            (m - 0.5) * if (x >> 20) % 16 == 0 { 40.0 } else { 1.0 }
        }).collect()
    }

    #[test]
    fn diag_picks_the_smallest_weighted_error_of_all_126_steps() {
        // Crow #300 p2-lh: the chosen byte is the brute-force minimum over 1..=126, ties to the
        // smaller byte, and never the NaN code 0x7F
        let vals = lcg_values(16 * 64, 7);
        let d: Vec<f32> = (0..16).map(|j| 0.1 + (j * j) as f32).collect();
        let global = 0.01f32;
        for sub in vals.chunks(16) {
            let b = encode_subblock_diag(sub, global, &d);
            assert!((1..=126).contains(&b));
            let e_b = weighted_err(sub, decode_ue4m3(b) * global, &d);
            for c in 1..=126u32 {
                let e_c = weighted_err(sub, decode_ue4m3(c) * global, &d);
                assert!(e_b < e_c || (e_b == e_c && b <= c) || b == c, "byte {b} ({e_b}) vs {c} ({e_c})");
            }
        }
        assert_eq!(encode_subblock_diag(&[0.0; 16], global, &d), 0);
    }

    #[test]
    fn diag_never_loses_to_mse_on_its_own_objective_and_reads_its_columns() {
        // per sub-block the weighted error of the diag bytes <= that of the mse bytes (mse's
        // byte is one of the 126 candidates); a row of 128 columns uses d[col..col + 16]
        let (rows, cols) = (8usize, 128usize);
        let vals = lcg_values(rows * cols, 11);
        let d: Vec<f32> = (0..cols).map(|j| if j % 32 < 4 { 50.0 } else { 0.5 }).collect();
        let (bd, g, _, _) = quantize_nvfp4_w(&vals, ScalesMode::Diag, Some((&d, cols)));
        let (bm, g2, _, _) = quantize_nvfp4(&vals, ScalesMode::Mse);
        assert_eq!(g, g2, "the global scale does not depend on the scale rule");
        let byte_of = |blocks: &[u8], sb: usize| blocks[(sb / 4) * 36 + sb % 4] as u32;
        let (mut tot_d, mut tot_m) = (0.0f64, 0.0f64);
        for (i, sub) in vals.chunks(16).enumerate() {
            let col = (i * 16) % cols;
            let w = &d[col..col + 16];
            let ed = weighted_err(sub, decode_ue4m3(byte_of(&bd, i)) * g, w);
            let em = weighted_err(sub, decode_ue4m3(byte_of(&bm, i)) * g, w);
            assert!(ed <= em, "sub-block {i}: diag {ed} > mse {em}");
            tot_d += ed as f64;
            tot_m += em as f64;
        }
        assert!(tot_d < tot_m, "diag must win on the weighted error overall");
    }

    #[test]
    fn diag_stats_map_every_dense_projection_to_its_input_group() {
        let k = |n: &str| DiagStats::key_of(n);
        assert_eq!(k("model.language_model.layers.3.self_attn.k_proj.weight").as_deref(), Some("layers.3.attn_in"));
        assert_eq!(k("model.language_model.layers.3.self_attn.o_proj.weight").as_deref(), Some("layers.3.o_in"));
        assert_eq!(k("model.language_model.layers.0.linear_attn.in_proj_z.weight").as_deref(), Some("layers.0.gdn_in"));
        assert_eq!(k("model.language_model.layers.0.linear_attn.out_proj.weight").as_deref(), Some("layers.0.gdn_out_in"));
        assert_eq!(k("model.language_model.layers.63.mlp.up_proj.weight").as_deref(), Some("layers.63.mlp_in"));
        assert_eq!(k("model.language_model.layers.63.mlp.down_proj.weight").as_deref(), Some("layers.63.down_in"));
        assert_eq!(k("lm_head.weight").as_deref(), Some("head_in"));
        assert_eq!(k("model.language_model.layers.0.linear_attn.in_proj_a.weight"), None);
        let ds = DiagStats { groups: [("head_in".to_string(), vec![1.0f32; 32])].into_iter().collect(), path: "t".into(), stats_sha256: String::new() };
        assert!(ds.weights_for("lm_head.weight", 32).is_ok());
        assert!(ds.weights_for("lm_head.weight", 48).is_err());
        assert!(ds.weights_for("model.language_model.layers.1.mlp.up_proj.weight", 32).is_err());
    }

    // ---- crow-nest #155: FP8 input, the GLM row, histograms, journal and --consume ----

    /// One synthetic tensor: name, safetensors dtype, shape, raw bytes.
    type SynthTensor = (String, &'static str, Vec<usize>, Vec<u8>);

    fn lcg_bytes(n: usize, seed: u32, f: impl Fn(u32) -> Vec<u8>) -> Vec<u8> {
        let mut x = seed;
        let mut v = Vec::new();
        while v.len() < n {
            x = x.wrapping_mul(1664525).wrapping_add(1013904223);
            v.extend(f(x));
        }
        v.truncate(n);
        v
    }

    fn synth_tensor(name: &str, dtype: &'static str, shape: &[usize], seed: u32) -> SynthTensor {
        let n: usize = shape.iter().product();
        let bytes = match dtype {
            // FP8: any byte but the two NaN codes
            "F8_E4M3" => lcg_bytes(n, seed, |x| {
                let b = (x >> 13) as u8;
                vec![if b & 0x7F == 0x7F { b ^ 1 } else { b }]
            }),
            "BF16" => lcg_bytes(2 * n, seed, |x| {
                let v = (((x >> 8) % 2001) as f32 - 1000.0) / 4000.0;
                ((v.to_bits() >> 16) as u16).to_le_bytes().to_vec()
            }),
            "F32" => lcg_bytes(4 * n, seed, |x| ((((x >> 8) % 1000) as f32 + 1.0) * 1e-5).to_le_bytes().to_vec()),
            other => lcg_bytes(n * elem_size_of(other).unwrap_or(1), seed, |x| vec![(x >> 9) as u8]),
        };
        (name.to_string(), dtype, shape.to_vec(), bytes)
    }

    /// FP8 weight + its F32 128x128 block scale (scales around 2^-7)
    fn fp8_pair(name: &str, shape: &[usize], seed: u32) -> (SynthTensor, SynthTensor) {
        let g = fp8::scale_grid(shape[0], shape[1]);
        let s: Vec<u8> = (0..g[0] * g[1]).flat_map(|j| (0.0078125f32 * (1.0 + j as f32 / 8.0)).to_le_bytes()).collect();
        (synth_tensor(name, "F8_E4M3", shape, seed), (format!("{name}_scale_inv"), "F32", g.to_vec(), s))
    }

    /// Write one safetensors file; returns (data_start, size).
    fn write_st(path: &std::path::Path, ts: &[SynthTensor]) -> (u64, u64) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dt, shape, b) in ts {
            header.insert(name.clone(), serde_json::json!({ "dtype": dt, "shape": shape, "data_offsets": [data.len(), data.len() + b.len()] }));
            data.extend_from_slice(b);
        }
        let h = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
        let mut f = std::fs::File::create(path).unwrap();
        f.write_all(&(h.len() as u64).to_le_bytes()).unwrap();
        f.write_all(&h).unwrap();
        f.write_all(&data).unwrap();
        (8 + h.len() as u64, 8 + (h.len() + data.len()) as u64)
    }

    fn glm_config() -> serde_json::Value {
        let lt: Vec<&str> = (0..45).map(|i| if i % 4 == 3 { "deepseek_sparse_attention" } else { "linear_attention" }).collect();
        serde_json::json!({
            "text_config": {
                "model_type": "glm5_next_text", "num_hidden_layers": 45, "num_nextn_predict_layers": 1,
                "hidden_size": 128, "vocab_size": 64, "layer_types": lt, "n_routed_experts": 2,
                "num_experts_per_tok": 1, "moe_intermediate_size": 256, "first_k_dense_replace": 3,
                "num_attention_heads": 2, "num_key_value_heads": 2,
            },
            "quantization_config": { "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128] },
        })
    }

    const GLM_SHARDS: [&str; 3] = ["model-00001-of-00003.safetensors", "model-00002-of-00003.safetensors", "model-00003-of-00003.safetensors"];

    /// A 3-shard GLM-5.3-Flash miniature: every kind of row once, FP8 weights with block scales
    /// (one with partial 128-tiles on both axes), the MTP layer 45 and a vision tensor (omitted),
    /// and expert 1 of layer 3 straddling shards 2 and 3 (its down_proj weight in shard 2, the
    /// scale in shard 3). Writes the shards into `shard_dir`, and into `dir` the configs, the
    /// model index, `hf-revision.json` (sha256 and size per shard) and the header cache
    /// `headers/<shard>.json`.
    fn write_glm_synth(dir: &std::path::Path, shard_dir: &std::path::Path) {
        std::fs::create_dir_all(dir.join("headers")).unwrap();
        std::fs::create_dir_all(shard_dir).unwrap();
        let l = |n: &str| format!("model.language_model.layers.{n}");
        let mut s1 = vec![
            synth_tensor("lm_head.weight", "BF16", &[64, 128], 1),
            synth_tensor("model.language_model.embed_tokens.weight", "BF16", &[64, 128], 2),
        ];
        let (w, s) = fp8_pair(&l("45.mlp.experts.0.up_proj.weight"), &[256, 128], 3);
        s1.extend([s, w]);
        let mut s2 = vec![
            synth_tensor(&l("0.input_layernorm.weight"), "BF16", &[128], 4),
            synth_tensor(&l("0.self_attn.q_proj.weight"), "BF16", &[128, 128], 5),
            synth_tensor(&l("0.self_attn.A_log"), "F32", &[4], 6),
            synth_tensor(&l("0.hc_attn_fn"), "BF16", &[24, 512], 7),
            synth_tensor(&l("0.hc_attn_base"), "F32", &[24], 8),
        ];
        let mut seed = 20;
        let mut pair = |name: String, shape: &[usize], v: &mut Vec<SynthTensor>| {
            seed += 1;
            let (w, s) = fp8_pair(&name, shape, seed);
            v.extend([s, w]);
        };
        pair(l("0.mlp.gate_proj.weight"), &[256, 128], &mut s2);
        pair(l("3.mlp.experts.0.gate_proj.weight"), &[256, 128], &mut s2);
        pair(l("3.mlp.experts.0.up_proj.weight"), &[256, 128], &mut s2);
        pair(l("3.mlp.experts.0.down_proj.weight"), &[128, 256], &mut s2);
        pair(l("3.mlp.experts.1.gate_proj.weight"), &[256, 128], &mut s2);
        let (dw, ds) = fp8_pair(&l("3.mlp.experts.1.down_proj.weight"), &[128, 256], 90);
        s2.push(dw);
        s2.push(synth_tensor(&l("3.mlp.gate.weight"), "BF16", &[2, 128], 9));
        s2.push(synth_tensor(&l("3.mlp.gate.e_score_correction_bias"), "F32", &[2], 10));
        let mut s3 = vec![ds];
        pair(l("3.mlp.experts.1.up_proj.weight"), &[256, 128], &mut s3);
        pair(l("3.self_attn.q_a_proj.weight"), &[192, 200], &mut s3);
        pair(l("3.mlp.shared_experts.down_proj.weight"), &[128, 256], &mut s3);
        s3.push(synth_tensor(&l("44.post_attention_layernorm.weight"), "BF16", &[128], 11));
        s3.push(synth_tensor("model.language_model.norm.weight", "BF16", &[128], 12));
        s3.push(synth_tensor("model.visual.blocks.0.norm1.weight", "BF16", &[16], 13));
        let mut wm = serde_json::Map::new();
        let mut siblings = Vec::new();
        for (shard, ts) in GLM_SHARDS.iter().zip([&s1, &s2, &s3]) {
            let p = shard_dir.join(shard);
            let (start, size) = write_st(&p, ts);
            let (hdr, _) = read_safetensors_header(&p).unwrap();
            std::fs::write(
                dir.join("headers").join(format!("{shard}.json")),
                serde_json::to_vec(&serde_json::json!({ "shard": shard, "size": size, "data_start": start, "header": hdr })).unwrap(),
            )
            .unwrap();
            for t in ts.iter() {
                wm.insert(t.0.clone(), serde_json::json!(shard));
            }
            siblings.push(serde_json::json!({ "rfilename": shard, "size": size, "lfs": { "sha256": recipe::sha256_file(&p).unwrap(), "size": size } }));
        }
        std::fs::write(dir.join("config.json"), serde_json::to_vec_pretty(&glm_config()).unwrap()).unwrap();
        std::fs::write(dir.join("generation_config.json"), b"{}").unwrap();
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&serde_json::json!({ "weight_map": wm })).unwrap()).unwrap();
        std::fs::write(dir.join("hf-revision.json"), serde_json::to_vec(&serde_json::json!({ "sha": "glm-synth", "siblings": siblings })).unwrap()).unwrap();
    }

    fn tmp(tag: &str) -> std::path::PathBuf {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let k = N.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
        let d = std::env::temp_dir().join(format!("cnq-155-{tag}-{}-{k}", std::process::id()));
        std::fs::remove_dir_all(&d).ok();
        std::fs::create_dir_all(&d).unwrap();
        d
    }

    fn glm_prov() -> Provenance {
        Provenance { repo: Some("crow-nest/glm-synth".into()), revision: None }
    }

    /// The uninterrupted reference conversion of the miniature: (container, sidecar) bytes.
    fn glm_reference(mode: ScalesMode) -> (Vec<u8>, Vec<u8>) {
        let dir = tmp(&format!("ref-{}", mode == ScalesMode::Mse));
        write_glm_synth(&dir, &dir);
        let out = dir.join("glm.cnq");
        assert_eq!(convert_with(&dir, &out, mode, &glm_prov(), None, &ConvertOpts::default()), 0);
        let r = (std::fs::read(&out).unwrap(), std::fs::read(dir.join("glm.cnq.sidecar.jsonl")).unwrap());
        assert!(!dir.join("glm.cnq.journal.jsonl").exists(), "the journal is removed after the trailer");
        std::fs::remove_dir_all(&dir).ok();
        r
    }

    /// End to end on the miniature: the FP8 tensor with partial tiles is stored as exactly the
    /// NVFP4 of its known dequantization; F32 carries are the source bytes; MTP, vision and the
    /// scales are not written; every routed-expert block starts at an absolute offset that is a
    /// multiple of 4096, its gate/up/down back to back; provenance comes from hf-revision.json.
    #[test]
    fn a_glm_conversion_reads_fp8_and_aligns_every_expert_block() {
        let dir = tmp("e2e");
        write_glm_synth(&dir, &dir);
        let out = dir.join("glm.cnq");
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &ConvertOpts::default()), 0);
        let bytes = std::fs::read(&out).unwrap();
        let idx = trailer(&bytes);
        assert_eq!((idx["recipe"].as_str(), idx["model"]["family"].as_str()), (Some("cnq4.5-glm5-next"), Some("Glm5Next")));
        assert_eq!(idx["model"]["source"]["revision"], "glm-synth");
        let sh = idx["model"]["source"]["shards"].as_array().unwrap();
        assert_eq!(sh.len(), 3);
        assert!(sh.iter().all(|s| s["sha256_from"] == "hf-lfs"));
        let ts = idx["tensors"].as_array().unwrap();
        assert_eq!(ts.len(), 20);
        assert!(ts.iter().all(|t| {
            let n = t["name"].as_str().unwrap();
            !n.contains("layers.45.") && !n.contains("visual") && !n.ends_with("_scale_inv")
        }));
        let get = |n: &str| ts.iter().find(|t| t["name"] == n).unwrap_or_else(|| panic!("{n}"));
        // the FP8 tensor with partial tiles: known result = quantize(dequant(fp8, scale))
        let name = "model.language_model.layers.3.self_attn.q_a_proj.weight";
        let (hdr, start) = read_safetensors_header(&dir.join(GLM_SHARDS[2])).unwrap();
        let src = std::fs::read(dir.join(GLM_SHARDS[2])).unwrap();
        let at = |k: &str| {
            let o = &hdr[k]["data_offsets"];
            &src[(start + o[0].as_u64().unwrap()) as usize..(start + o[1].as_u64().unwrap()) as usize]
        };
        let sc = fp8::scales_to_f32(at(&format!("{name}_scale_inv")), "F32").unwrap();
        let vals = fp8::dequant_fp8_block(name, at(name), 192, 200, &sc).unwrap();
        let (want, global, _, _) = quantize_nvfp4(&vals, ScalesMode::Mse);
        let t = get(name);
        let off = 12 + t["offset"].as_u64().unwrap() as usize;
        assert_eq!(&bytes[off..off + want.len()], want.as_slice());
        assert_eq!(t["global_scale"].as_f64().unwrap() as f32, global);
        assert_eq!(t["section"], "text");
        // f32 carries are the source bytes
        let e = get("model.language_model.layers.3.mlp.gate.e_score_correction_bias");
        assert_eq!(e["dtype"], "f32");
        let (hdr2, start2) = read_safetensors_header(&dir.join(GLM_SHARDS[1])).unwrap();
        let src2 = std::fs::read(dir.join(GLM_SHARDS[1])).unwrap();
        let o = &hdr2["model.language_model.layers.3.mlp.gate.e_score_correction_bias"]["data_offsets"];
        let off = 12 + e["offset"].as_u64().unwrap() as usize;
        assert_eq!(&bytes[off..off + 8], &src2[(start2 + o[0].as_u64().unwrap()) as usize..(start2 + o[1].as_u64().unwrap()) as usize]);
        assert_eq!(get("model.language_model.layers.0.hc_attn_fn")["dtype"], "bf16");
        assert_eq!(get("model.language_model.layers.0.self_attn.q_proj.weight")["dtype"], "nvfp4");
        // expert blocks: aligned, back to back
        for x in 0..2 {
            let p = |w: &str| get(&format!("model.language_model.layers.3.mlp.experts.{x}.{w}_proj.weight"));
            let (g, u, d) = (p("gate"), p("up"), p("down"));
            let off = |t: &serde_json::Value| t["offset"].as_u64().unwrap();
            let len = |t: &serde_json::Value| t["len"].as_u64().unwrap();
            assert_eq!((12 + off(g)) % 4096, 0, "expert {x}");
            assert_eq!(off(u), off(g) + len(g));
            assert_eq!(off(d), off(u) + len(u));
        }
        std::fs::remove_dir_all(&dir).ok();
    }

    /// #155 failure mode: the histogram must count what is written, also what `mse` clips:
    /// sum(h_codes) == n and sum(h_scales) == n / 16 on every NVFP4 sidecar line, and the code
    /// summary has one table per expert block.
    #[test]
    fn every_histogram_sums_to_the_value_count() {
        let (_, sidecar) = glm_reference(ScalesMode::Mse);
        let lines: Vec<serde_json::Value> = String::from_utf8(sidecar).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let mut n_nvfp4 = 0;
        for v in lines.iter().filter(|v| v["dtype"] == "nvfp4" && v.get("record").is_none()) {
            let h = entropy::Hist::from_json(&v["h_codes"], &v["h_scales"]).unwrap();
            let n = v["n"].as_u64().unwrap();
            assert_eq!((h.n_codes(), h.n_scales()), (n, n / 16), "{}", v["name"]);
            n_nvfp4 += 1;
        }
        assert_eq!(n_nvfp4, 10);
        let eb = lines.iter().find(|v| v["record"] == "code_summary" && v["scope"] == "expert_blocks").unwrap();
        assert_eq!(eb["blocks"], 2);
        assert!(lines.iter().any(|v| v["record"] == "code_summary" && v["scope"] == "class" && v["class"] == "attn_mla"));
        // forced clipping: 15 values of 1 and one of 8 per sub-block, which `mse` clips
        // (the case of `mse_beats_ceil_on_moderate_outlier`)
        let vals: Vec<f32> = (0..64 * 64).map(|i| if i % 16 == 15 { 8.0 } else if i % 3 == 0 { -1.0 } else { 1.0 }).collect();
        for mode in [ScalesMode::Ceil, ScalesMode::Mse] {
            let (blocks, _, st, _) = quantize_nvfp4(&vals, mode);
            if mode == ScalesMode::Mse {
                assert!(st.clipped > 0, "the test must clip");
            }
            let h = entropy::Hist::of_blocks(&blocks);
            assert_eq!((h.n_codes(), h.n_scales()), (vals.len() as u64, vals.len() as u64 / 16));
        }
    }

    /// #177: a tensor whose largest sub-block scale lands above 448 in f32 (`max_scale /
    /// (max_scale / 448)` rounds up) gets scale byte 0x7F, the E4M3 NaN code, from the ceiling
    /// rule. The glm5_next rule writes no 0x7F, and its codes are rounded against the 448 that is
    /// written (not the engine's byte-only rewrite, which would shrink the block max by 6.7 %).
    /// The rule of the families of record writes the same bytes as before, 0x7F included.
    #[test]
    fn the_glm_rule_keeps_the_e4m3_nan_code_out_of_the_scales() {
        let v = (0..100_000)
            .map(|k| 1.0f32 + k as f32 * 1e-4)
            .find(|v| {
                let s = *v / 6.0;
                s / (s / UE4M3_MAX) > UE4M3_MAX
            })
            .expect("a block max whose divided scale rounds above 448");
        // block 0, sub-block 0 is the tensor max (16 x +-v, so `mse` has nothing to clip);
        // everything else is smaller
        let vals: Vec<f32> = (0..256).map(|i| if i < 16 { if i % 2 == 0 { v } else { -v } } else { ((i % 13) as f32 - 6.0) * 0.05 }).collect();
        let scale_bytes = |b: &[u8]| b.chunks_exact(36).flat_map(|blk| blk[..4].to_vec()).collect::<Vec<u8>>();
        let (old, _, _, _) = quantize_nvfp4(&vals, ScalesMode::Ceil);
        assert_eq!(old[0], 0x7F, "the fixture must hit the NaN code under the ceiling rule (v = {v})");
        for mode in [ScalesMode::Ceil, ScalesMode::Mse] {
            let mn = if mode == ScalesMode::Mse { "mse" } else { "ceil" };
            // the families of record: byte-identical to `quantize_nvfp4`
            for f in [recipe::Family::FlashNext, recipe::Family::Qwen35Dense] {
                let (want, wg, _, _) = quantize_nvfp4(&vals, mode);
                let (got, gg, _, _) = quantize_nvfp4_cap(&vals, mode, None, f.scale_byte_max());
                assert_eq!((got, gg), (want, wg), "{f:?} {mn}");
            }
            // glm5_next: no 0x7F, and the block max decodes within 1 % (a byte-only rewrite: 6.7 %)
            let (blocks, global, _, _) = quantize_nvfp4_cap(&vals, mode, None, recipe::Family::Glm5Next.scale_byte_max());
            assert!(!scale_bytes(&blocks).contains(&0x7F), "{mn}: a glm5_next scale byte is 0x7F");
            let d = dequant_nvfp4(&blocks, global);
            assert!((d[0] - v).abs() / v < 0.01, "{mn}: block max {v} decodes to {}", d[0]);
        }
    }

    /// #177 end to end: the GLM miniature's container carries no scale byte 0x7F, read from the
    /// written histograms (`h_scales[0x7F]`) of every NVFP4 sidecar line, in both scale modes.
    #[test]
    fn a_glm_container_carries_no_scale_byte_0x7f() {
        for mode in [ScalesMode::Ceil, ScalesMode::Mse] {
            let mn = if mode == ScalesMode::Mse { "mse" } else { "ceil" };
            let (_, sidecar) = glm_reference(mode);
            let lines: Vec<serde_json::Value> = String::from_utf8(sidecar).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
            let nv: Vec<&serde_json::Value> = lines.iter().filter(|v| v["dtype"] == "nvfp4" && v.get("record").is_none()).collect();
            assert_eq!(nv.len(), 10);
            for v in nv {
                assert_eq!(v["h_scales"][0x7F], 0, "{mn} {}", v["name"]);
            }
        }
    }

    /// A conversion killed after k tensors (no trailer, sometimes with a torn tail on the
    /// container and on the journal) and run again ends with the same container and sidecar
    /// bytes as an uninterrupted run, for k at the start, inside an expert block, at the
    /// straddling pair and before the last tensor. A journal of another plan is refused.
    #[test]
    fn a_glm_conversion_killed_and_resumed_is_byte_identical() {
        let (want, want_side) = glm_reference(ScalesMode::Mse);
        for k in [1usize, 9, 13, 18] {
            let dir = tmp(&format!("kill-{k}"));
            write_glm_synth(&dir, &dir);
            let out = dir.join("glm.cnq");
            let opts = ConvertOpts { stop_after: Some(k), ..ConvertOpts::default() };
            assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &opts), 4, "k {k}");
            assert!(dir.join("glm.cnq.journal.jsonl").exists());
            if k % 2 == 1 {
                // a kill mid-write: garbage behind the last synced tensor, half a journal line
                std::fs::OpenOptions::new().append(true).open(&out).unwrap().write_all(&[0xAB; 777]).unwrap();
                std::fs::OpenOptions::new().append(true).open(dir.join("glm.cnq.journal.jsonl")).unwrap().write_all(b"{\"seq\": 99, \"pad").unwrap();
            }
            assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &ConvertOpts::default()), 0, "k {k}");
            assert!(std::fs::read(&out).unwrap() == want, "k {k}: container differs from the uninterrupted run");
            let got_side = std::fs::read(dir.join("glm.cnq.sidecar.jsonl")).unwrap();
            if got_side != want_side {
                let (g, w) = (String::from_utf8(got_side).unwrap(), String::from_utf8(want_side.clone()).unwrap());
                for (a, b) in g.lines().zip(w.lines()) {
                    if a != b {
                        panic!("k {k}: sidecar differs
 got  {a}
 want {b}");
                    }
                }
                panic!("k {k}: sidecar differs in length {} vs {}", g.lines().count(), w.lines().count());
            }
            std::fs::remove_dir_all(&dir).ok();
        }
        // a corrupted journalled tensor: everything from it on is written again
        let dir = tmp("kill-corrupt");
        write_glm_synth(&dir, &dir);
        let out = dir.join("glm.cnq");
        let opts = ConvertOpts { stop_after: Some(12), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &opts), 4);
        let mut b = std::fs::read(&out).unwrap();
        let n = b.len();
        b[n - 100] ^= 0xFF;
        std::fs::write(&out, &b).unwrap();
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &ConvertOpts::default()), 0);
        assert!(std::fs::read(&out).unwrap() == want);
        // another plan (scale policy) is refused, not mixed
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &ConvertOpts { stop_after: Some(3), ..ConvertOpts::default() }), 4);
        assert_eq!(convert_with(&dir, &out, ScalesMode::Ceil, &glm_prov(), None, &ConvertOpts::default()), 2);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// `--consume`: the shards live in their own directory and become readable one by one when a
    /// driver writes `<shard>.verified`; the converter never reads ahead of the markers, writes
    /// `<shard>.done` when everything that reads a shard is synced, never deletes a shard, and
    /// writes the same container as a conversion of the whole directory. A marker whose sha256
    /// disagrees with the HF record is refused.
    #[test]
    fn consume_mode_follows_the_verified_markers_and_writes_the_same_container() {
        let (want, _) = glm_reference(ScalesMode::Mse);
        let dir = tmp("consume");
        let shards = dir.join("incoming");
        write_glm_synth(&dir, &shards);
        let api = recipe::read_hf_api_info(&dir).unwrap().unwrap();
        // the driver: a window of two shards, the next one verified when the one before is done;
        // it records which shards were done at the moment it verified a later one
        let driver = {
            let shards = shards.clone();
            let api_lfs = api.lfs.clone();
            std::thread::spawn(move || {
                let verify = |s: &str| {
                    let (sha, size) = &api_lfs[s];
                    std::fs::write(shards.join(format!("{s}.verified")), serde_json::to_vec(&serde_json::json!({ "sha256": sha, "size": size })).unwrap()).unwrap();
                };
                verify(GLM_SHARDS[0]);
                verify(GLM_SHARDS[1]);
                let t0 = std::time::Instant::now();
                while !shards.join(format!("{}.done", GLM_SHARDS[0])).exists() {
                    assert!(t0.elapsed().as_secs() < 60, "shard 1 never done");
                    std::thread::sleep(std::time::Duration::from_millis(5));
                }
                verify(GLM_SHARDS[2]);
            })
        };
        let out = dir.join("glm.cnq");
        let opts = ConvertOpts { headers: Some(dir.join("headers")), consume: Some(shards.clone()), poll: std::time::Duration::from_millis(5), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &opts), 0);
        driver.join().unwrap();
        assert!(std::fs::read(&out).unwrap() == want, "--consume wrote another container");
        for s in GLM_SHARDS {
            assert!(shards.join(format!("{s}.done")).exists(), "{s}: no .done");
            assert!(shards.join(s).exists(), "{s}: the converter never deletes a shard");
        }
        // a marker with another sha256: refused before a byte of that shard is converted
        let dir2 = tmp("consume-bad");
        let shards2 = dir2.join("incoming");
        write_glm_synth(&dir2, &shards2);
        std::fs::write(shards2.join(format!("{}.verified", GLM_SHARDS[0])), br#"{"sha256": "00"}"#).unwrap();
        let opts = ConvertOpts { headers: Some(dir2.join("headers")), consume: Some(shards2.clone()), poll: std::time::Duration::from_millis(5), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir2, &dir2.join("glm.cnq"), ScalesMode::Mse, &glm_prov(), None, &opts), 2);
        // --consume without --headers is refused
        let opts = ConvertOpts { consume: Some(shards2.clone()), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir2, &dir2.join("glm2.cnq"), ScalesMode::Mse, &glm_prov(), None, &opts), 2);
        std::fs::remove_dir_all(&dir).ok();
        std::fs::remove_dir_all(&dir2).ok();
    }

    /// `plan --headers` builds the same manifest as reading the shard files, and needs no shard
    /// on disk: provenance then comes from hf-revision.json without reading one.
    #[test]
    fn the_manifest_from_the_header_cache_equals_the_one_from_the_shards() {
        let dir = tmp("headers");
        write_glm_synth(&dir, &dir);
        let key = |m: &Manifest| -> Vec<String> {
            m.tensors
                .iter()
                .map(|t| format!("{} {} {} {} {:?} {:?}", t.name, t.shard, t.data_begin, t.data_end, t.decision, t.scale.as_ref().map(|s| (&s.shard, s.data_begin, s.data_end))))
                .collect()
        };
        let a = build_manifest(&dir).unwrap();
        let b = build_manifest_from(&dir, Some(&dir.join("headers"))).unwrap();
        assert_eq!(key(&a), key(&b));
        assert_eq!(a.omitted, b.omitted);
        assert_eq!(a.omitted.values().map(|v| v.0).sum::<usize>(), 3, "MTP weight + its scale, the vision tensor");
        for s in GLM_SHARDS {
            std::fs::remove_file(dir.join(s)).unwrap();
        }
        assert!(build_manifest(&dir).is_err(), "without --headers a missing shard is a refusal");
        let c = build_manifest_from(&dir, Some(&dir.join("headers"))).unwrap();
        assert_eq!(key(&a), key(&c));
        let (_, rev, shards, pending) = provenance(&c, &glm_prov(), true).unwrap();
        assert_eq!((rev.as_str(), shards.len(), pending), ("glm-synth", 3, 0));
        assert_eq!(plan(&["--headers".into(), dir.join("headers").display().to_string(), "--source-repo".into(), "r".into(), dir.display().to_string()]), 0);
        std::fs::remove_dir_all(&dir).ok();
    }

    /// crow-nest #154: an unknown dtype is refused by name (it used to be skipped with a line on
    /// stderr, which would have dropped every F8_E4M3 tensor of GLM-5.3-Flash); an FP8 weight
    /// without its scale and a scale without its weight are refused by name too.
    #[test]
    fn unknown_dtypes_and_unpaired_scales_are_refused_by_name() {
        let case = |tag: &str, ts: Vec<SynthTensor>| -> String {
            let dir = tmp(tag);
            std::fs::write(dir.join("config.json"), serde_json::to_vec(&glm_config()).unwrap()).unwrap();
            std::fs::write(dir.join("generation_config.json"), b"{}").unwrap();
            let st = dir.join("model.safetensors");
            write_st(&st, &ts);
            let e = build_manifest(&st).err().expect("refused");
            std::fs::remove_dir_all(&dir).ok();
            e
        };
        let (w, s) = fp8_pair("model.language_model.layers.3.mlp.shared_experts.up_proj.weight", &[256, 128], 1);
        let e = case("dtype", vec![synth_tensor("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight", "F8_E5M2", &[256, 128], 2), w.clone(), s.clone()]);
        assert!(e.contains("model.language_model.layers.3.mlp.shared_experts.gate_proj.weight: dtype F8_E5M2"), "{e}");
        let e = case("noscale", vec![w.clone()]);
        assert!(e.contains("up_proj.weight: an F8_E4M3 weight without its"), "{e}");
        let e = case("noweight", vec![s.clone()]);
        assert!(e.contains("up_proj.weight_scale_inv: a weight_scale_inv without its F8_E4M3 weight"), "{e}");
        let bad = (s.0.clone(), "F32", vec![1, 1], vec![0u8; 4]);
        let e = case("grid", vec![w, bad]);
        assert!(e.contains("the 128x128 grid of [256, 128] is [2, 1]"), "{e}");
    }

    /// crow-nest #156: `--layers 0,3 --with-embed-head` on the miniature writes exactly layers 0
    /// and 3 plus embedding, lm_head and final norm, each with the bytes the full conversion
    /// writes for it; layer 44 is filtered (not missing: exit 0, not 3); the index says
    /// `partial`. A layer the checkpoint does not have is refused.
    #[test]
    fn a_partial_conversion_writes_the_named_layers_with_the_full_bytes() {
        let (full, _) = glm_reference(ScalesMode::Mse);
        let fidx = trailer(&full);
        let dir = tmp("partial");
        write_glm_synth(&dir, &dir);
        let out = dir.join("glm-l03.cnq");
        let f = partial::LayerFilter::parse("0,3", true).unwrap();
        let opts = ConvertOpts { headers: Some(dir.join("headers")), filter: Some(f), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &opts), 0);
        let bytes = std::fs::read(&out).unwrap();
        let idx = trailer(&bytes);
        let ts = idx["tensors"].as_array().unwrap();
        let names: Vec<&str> = ts.iter().map(|t| t["name"].as_str().unwrap()).collect();
        assert!(names.iter().all(|n| !n.contains("layers.44.")), "{names:?}");
        assert_eq!(ts.len(), fidx["tensors"].as_array().unwrap().len() - 1, "only layer 44's norm is filtered");
        for n in ["lm_head.weight", "model.language_model.embed_tokens.weight", "model.language_model.norm.weight"] {
            assert!(names.contains(&n), "{n}");
        }
        assert_eq!(idx["partial"]["layers"], serde_json::json!([0, 3]));
        assert_eq!(idx["partial"]["embed_head_norm"], true);
        assert_eq!(idx["partial"]["tensors_filtered"], 1);
        let body = |b: &[u8], t: &serde_json::Value| {
            let o = 12 + t["offset"].as_u64().unwrap() as usize;
            b[o..o + t["len"].as_u64().unwrap() as usize].to_vec()
        };
        for t in ts {
            let ft = fidx["tensors"].as_array().unwrap().iter().find(|x| x["name"] == t["name"]).unwrap();
            assert!(body(&bytes, t) == body(&full, ft), "{}: other bytes than the full conversion", t["name"]);
            assert_eq!(t["global_scale"], ft["global_scale"]);
        }
        // without --with-embed-head: no embedding, head or final norm
        let out2 = dir.join("glm-l03-noeh.cnq");
        let opts = ConvertOpts { headers: Some(dir.join("headers")), filter: Some(partial::LayerFilter::parse("0,3", false).unwrap()), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir, &out2, ScalesMode::Mse, &glm_prov(), None, &opts), 0);
        let idx2 = trailer(&std::fs::read(&out2).unwrap());
        assert!(idx2["tensors"].as_array().unwrap().iter().all(|t| !partial::LayerFilter::is_embed_head_norm(t["name"].as_str().unwrap())));
        // layers 1 and 2 do not exist in the miniature: refused before a byte is written
        let out3 = dir.join("glm-l0-3.cnq");
        let opts = ConvertOpts { headers: Some(dir.join("headers")), filter: Some(partial::LayerFilter::parse("0-3", true).unwrap()), ..ConvertOpts::default() };
        assert_eq!(convert_with(&dir, &out3, ScalesMode::Mse, &glm_prov(), None, &opts), 2);
        assert!(!out3.exists());
        std::fs::remove_dir_all(&dir).ok();
    }

    /// crow-nest #156: `converter dequant` decodes what gate 0 measured. Re-decoding the written
    /// blocks reproduces the sidecar's squared-error sum and max error bit for bit; a decode with
    /// the nibble order or the scale order swapped does not.
    #[test]
    fn dequant_reproduces_gate_zero_bit_for_bit() {
        let vals: Vec<f32> = (0..64 * 40).map(|i| ((i as f32 * 0.37).sin() * 3.0 + if i % 29 == 0 { 9.0 } else { 0.0 }) * 1e-2).collect();
        for mode in [ScalesMode::Ceil, ScalesMode::Mse] {
            let (blocks, global, st, _) = quantize_nvfp4(&vals, mode);
            let d = dequant_nvfp4(&blocks, global);
            assert_eq!(d.len(), vals.len());
            // summed per 16-value sub-block, then over sub-blocks: gate 0's order (`QuantStats::absorb`)
            let (mut sse, mut max) = (0.0f64, 0.0f32);
            for (ds, vs) in d.chunks(16).zip(vals.chunks(16)) {
                let mut sub = 0.0f64;
                for (a, b) in ds.iter().zip(vs) {
                    let e = (a - b).abs();
                    sub += (e as f64) * (e as f64);
                    max = max.max(e);
                }
                sse += sub;
            }
            assert_eq!(sse.to_bits(), st.sum_sq_err.to_bits(), "mode {}: the decode is not gate 0's", mode == ScalesMode::Mse);
            assert_eq!(max.to_bits(), st.max_abs_err.to_bits());
            // the controls: a swapped nibble order and a swapped scale order are seen
            let mut swapped = blocks.clone();
            for b in swapped.chunks_exact_mut(36) {
                for k in 4..36 {
                    b[k] = b[k].rotate_left(4);
                }
            }
            assert_ne!(dequant_nvfp4(&swapped, global), d);
            let mut sc = blocks.clone();
            for b in sc.chunks_exact_mut(36) {
                b[..4].reverse();
            }
            assert_ne!(dequant_nvfp4(&sc, global), d);
        }
    }

    /// crow-nest #156: the subcommand's decode on a written container: a whole NVFP4 tensor, a row
    /// range of it, and a BF16 and an F32 tensor, against the bytes in the file.
    #[test]
    fn dequant_reads_the_container_records() {
        let dir = tmp("dequant");
        write_glm_synth(&dir, &dir);
        let out = dir.join("glm.cnq");
        assert_eq!(convert_with(&dir, &out, ScalesMode::Mse, &glm_prov(), None, &ConvertOpts::default()), 0);
        let bytes = std::fs::read(&out).unwrap();
        let idx = trailer(&bytes);
        let get = |n: &str| idx["tensors"].as_array().unwrap().iter().find(|t| t["name"] == n).unwrap().clone();
        let body = |t: &serde_json::Value| {
            let o = 12 + t["offset"].as_u64().unwrap() as usize;
            bytes[o..o + t["len"].as_u64().unwrap() as usize].to_vec()
        };
        let mut f = std::fs::File::open(&out).unwrap();
        let q = get("model.language_model.layers.3.self_attn.q_a_proj.weight"); // [192, 200], nvfp4
        assert_eq!(q["dtype"], "nvfp4");
        let want = dequant_nvfp4(&body(&q), q["global_scale"].as_f64().unwrap() as f32);
        let all = dequant::decode(&mut f, 12, &q, None).unwrap();
        assert!(all == want[..192 * 200]);
        // row 1 of a 200-wide tensor is not whole 64-value blocks: refused
        assert!(dequant::decode(&mut f, 12, &q, Some((1, 2))).is_err());
        let g = get("model.language_model.layers.3.mlp.experts.0.gate_proj.weight"); // [256, 128]
        let gall = dequant::decode(&mut f, 12, &g, None).unwrap();
        assert!(dequant::decode(&mut f, 12, &g, Some((5, 9))).unwrap() == gall[5 * 128..9 * 128]);
        let e = get("lm_head.weight");
        assert_eq!(e["dtype"], "bf16");
        assert!(dequant::decode(&mut f, 12, &e, None).unwrap() == bytes_to_f32(&body(&e), "BF16"));
        let c = get("model.language_model.layers.3.mlp.gate.e_score_correction_bias");
        assert!(dequant::decode(&mut f, 12, &c, None).unwrap() == bytes_to_f32(&body(&c), "F32"));
        std::fs::remove_dir_all(&dir).ok();
    }
}
