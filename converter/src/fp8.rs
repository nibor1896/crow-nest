//! crow-nest #155 (GLM-5.3-Flash step 5): FP8 E4M3 input with 128x128 block scales.
//!
//! The GLM-5.3-Flash originals store most weights as `F8_E4M3` with one f32 `weight_scale_inv`
//! per 128x128 tile (`quantization_config`: `quant_method fp8`, `fmt e4m3`,
//! `weight_block_size [128, 128]`). This module turns such a pair into f32 values, exactly:
//!
//! - **E4M3** is the `e4m3fn` variant (torch `float8_e4m3fn`, OCP OFP8): bias 7, no infinities,
//!   `S.1111.111` is NaN (codes 0x7F and 0xFF), max 448, subnormals `m * 2^-9`. Every finite
//!   code is exactly representable in f32, so the table below is exact.
//! - **Block dequant** is one f32 multiply per value, `fp8_as_f32 * scale[r / 128][c / 128]`,
//!   the scale grid row-major with `ceil(cols / 128)` columns and partial edge tiles allowed.
//!   That is DeepSeek-V3 `inference/kernel.py` `weight_dequant` (`x.to(f32) * s`, ceil grid,
//!   masked edges) and transformers `Fp8Dequantize._dequantize_one` (`q.float() * s` in f32;
//!   it refuses shapes not divisible by the grid). One rounding, the same one, on all three.
//!
//! A NaN code is refused by name: a weight file with a NaN in it is not something to quantize.

/// The 256 E4M3FN codes as f32; NaN for 0x7F and 0xFF.
pub fn e4m3fn_to_f32(b: u8) -> f32 {
    let sign = if b & 0x80 != 0 { -1.0f32 } else { 1.0f32 };
    let e = ((b >> 3) & 0xF) as i32;
    let m = (b & 0x7) as f32;
    if e == 15 && (b & 0x7) == 7 {
        return f32::NAN;
    }
    let mag = if e == 0 {
        m * (2.0f32).powi(-9) // subnormal: (m / 8) * 2^(1 - 7)
    } else {
        (1.0 + m / 8.0) * (2.0f32).powi(e - 7)
    };
    sign * mag
}

/// The lookup table, built once per call site (256 entries, cheap).
pub fn lut() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (i, v) in t.iter_mut().enumerate() {
        *v = e4m3fn_to_f32(i as u8);
    }
    t
}

/// The scale grid of a `[rows, cols]` FP8 weight under 128x128 blocks.
pub fn scale_grid(rows: usize, cols: usize) -> [usize; 2] {
    [rows.div_ceil(128), cols.div_ceil(128)]
}

/// Scale bytes (`F32`, or `BF16` widened exactly) to f32.
pub fn scales_to_f32(raw: &[u8], dtype: &str) -> Result<Vec<f32>, String> {
    match dtype {
        "F32" => Ok(raw.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()),
        "BF16" => Ok(raw.chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect()),
        other => Err(format!("weight_scale_inv dtype {other}: only F32 or BF16 scales are read")),
    }
}

/// Dequantize one `[rows, cols]` FP8 E4M3FN tensor with its 128x128 block scales (row-major
/// grid `[ceil(rows/128), ceil(cols/128)]`) to f32: `lut[q] * s`, one f32 multiply per value.
pub fn dequant_fp8_block(name: &str, raw: &[u8], rows: usize, cols: usize, scale: &[f32]) -> Result<Vec<f32>, String> {
    if raw.len() != rows * cols {
        return Err(format!("{name}: {} FP8 bytes for shape [{rows}, {cols}]", raw.len()));
    }
    let [sr, sc] = scale_grid(rows, cols);
    if scale.len() != sr * sc {
        return Err(format!("{name}: {} scales, the 128x128 grid of [{rows}, {cols}] needs [{sr}, {sc}] = {}", scale.len(), sr * sc));
    }
    if let Some(bad) = scale.iter().position(|s| !s.is_finite()) {
        return Err(format!("{name}: weight_scale_inv[{bad}] is not finite ({})", scale[bad]));
    }
    let t = lut();
    let mut out = Vec::with_capacity(rows * cols);
    for r in 0..rows {
        let srow = &scale[(r / 128) * sc..(r / 128) * sc + sc];
        let row = &raw[r * cols..(r + 1) * cols];
        for (c, &q) in row.iter().enumerate() {
            if q & 0x7F == 0x7F {
                return Err(format!("{name}: FP8 NaN code 0x{q:02X} at [{r}, {c}] - refusing to quantize a NaN"));
            }
            out.push(t[q as usize] * srow[c / 128]);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::Digest;

    /// The table equals torch `float8_e4m3fn` on all 254 finite codes, bit for bit
    /// (`tests/fixtures/fp8-e4m3fn-lut.tsv`, exported from `.venv-oracle` torch 2.13.0+cpu by
    /// `tests/fixtures/fp8_fixtures.py`), and both NaN codes are NaN.
    #[test]
    fn the_e4m3fn_table_equals_torch_float8_e4m3fn() {
        let tsv = include_str!("../tests/fixtures/fp8-e4m3fn-lut.tsv");
        let mut n = 0;
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let code: u8 = f[0].parse().unwrap();
            let v = e4m3fn_to_f32(code);
            if f[1] == "nan" {
                assert!(v.is_nan(), "code {code}");
            } else {
                let bits = u32::from_str_radix(f[1], 16).unwrap();
                assert_eq!(v.to_bits(), bits, "code {code}: {v} vs torch {}", f32::from_bits(bits));
                n += 1;
            }
        }
        assert_eq!(n, 254);
        assert!(e4m3fn_to_f32(0x7F).is_nan() && e4m3fn_to_f32(0xFF).is_nan());
        assert_eq!(e4m3fn_to_f32(0x7E), 448.0);
        assert_eq!(e4m3fn_to_f32(0x01), 2.0f32.powi(-9));
    }

    /// The deterministic synthetic input both sides generate (`fp8_fixtures.py` `synth`):
    /// bytes from a multiplicative hash with the NaN codes moved one step down, scales from
    /// fixed f32 bit patterns.
    pub(crate) fn synth(rows: usize, cols: usize) -> (Vec<u8>, Vec<f32>) {
        let raw: Vec<u8> = (0..rows * cols)
            .map(|i| {
                let b = (((i as u64).wrapping_mul(2654435761) >> 7) & 0xFF) as u8;
                if b & 0x7F == 0x7F { b ^ 0x01 } else { b }
            })
            .collect();
        let [sr, sc] = scale_grid(rows, cols);
        let scale: Vec<f32> = (0..sr * sc).map(|j| f32::from_bits(0x3A00_0000 + ((j as u32).wrapping_mul(7919) % 0x0080_0000) * 3)).collect();
        (raw, scale)
    }

    fn sha_f32(v: &[f32]) -> String {
        let mut h = sha2::Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        h.finalize().iter().map(|b| format!("{b:02x}")).collect()
    }

    /// A 300 x 200 tensor (partial 128-tiles on both axes) dequantizes bit-identical to the
    /// DeepSeek-V3 `weight_dequant` formula (ceil grid, `x.to(f32) * s`) computed with torch;
    /// a 256 x 256 tensor bit-identical to transformers `Fp8Dequantize._dequantize_one`.
    /// The expected sha256 of the f32 LE bytes are in `tests/fixtures/fp8-dequant.tsv`.
    #[test]
    fn a_synthetic_fp8_tensor_dequantizes_like_deepseek_and_transformers() {
        let tsv = include_str!("../tests/fixtures/fp8-dequant.tsv");
        let mut seen = 0;
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let (who, rows, cols, want_sha, first) = (f[0], f[1].parse().unwrap(), f[2].parse().unwrap(), f[3], f[4]);
            let (raw, scale) = synth(rows, cols);
            let got = dequant_fp8_block(who, &raw, rows, cols, &scale).unwrap();
            assert_eq!(format!("{:08x}", got[rows * cols - 1].to_bits()), first, "{who}: last value");
            assert_eq!(sha_f32(&got), want_sha, "{who} {rows}x{cols}");
            seen += 1;
        }
        assert_eq!(seen, 2);
    }

    #[test]
    fn nan_codes_and_wrong_grids_are_refused_by_name() {
        let (mut raw, scale) = synth(130, 64);
        raw[129 * 64 + 3] = 0xFF;
        let e = dequant_fp8_block("w", &raw, 130, 64, &scale).unwrap_err();
        assert!(e.contains("NaN code 0xFF at [129, 3]"), "{e}");
        raw[129 * 64 + 3] = 0x7F;
        assert!(dequant_fp8_block("w", &raw, 130, 64, &scale).unwrap_err().contains("0x7F"));
        let e = dequant_fp8_block("w", &raw, 130, 64, &scale[..1]).unwrap_err();
        assert!(e.contains("needs [2, 1]"), "{e}");
        assert!(scales_to_f32(&[0; 4], "F16").is_err());
        assert_eq!(scales_to_f32(&[0x80, 0x3F], "BF16").unwrap(), vec![1.0]);
    }
}
