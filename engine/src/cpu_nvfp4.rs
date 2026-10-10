//! crow-nest #173: the CPU NVFP4 expert FFN - lever 2 of the GLM measurement book, section F
//! ("CPU on miss"): a routed expert that sits in the pinned RAM tier can be computed where it
//! lies instead of crossing PCIe. This module is the kernel only; nothing in the engine calls it
//! yet (the wiring is the glm5_next port, #159 and plan steps 14-15).
//!
//! # Layout (the engine's, bit for bit)
//!
//! - An NVFP4 matrix `[rows, cols]` is `rows * cols / 64` blocks of 36 B, row-major: 4 ue4m3
//!   sub-block scales, then 32 B of E2M1 codes; value `idx` of the block is in byte
//!   `4 + idx / 2`, low nibble for even `idx` (`kernels.rs` `gemv_fp4`, converter
//!   `dequant_nvfp4`).
//! - Sub-block scale = `cnq::ue4m3(byte) * global_scale` in f32, with the expert-slab rule of
//!   `residency::sanitize_sf_slab` applied first: byte 0x7F (the E4M3 NaN code, read as 480 by
//!   the scalar device decoder) is 0x7E (448). The kernel applies the rule itself, so raw
//!   container bytes and sanitized slabs give identical results.
//! - One GLM-5.3-Flash expert unit (converter `write_units`): gate `[I, H]` | up `[I, H]` |
//!   down `[H, I]` back to back, each with its own global scale; H 4096, I 2048 give
//!   3 x 4,718,592 = 14,155,776 B.
//! - `y = down(silu(gate(x)) * up(x))` with `silu(g) * u = g / (1 + exp(-g)) * u`, the formula
//!   of `silu_mul640`.
//!
//! # The f32 order (one order, both paths)
//!
//! Per output row and token there are 8 lane accumulators. For every 16-value sub-block, lane l
//! takes values 2l and 2l+1 (the low and high nibble of code byte l):
//! `p = w[2l] * x[2l] + w[2l+1] * x[2l+1]`, then `acc[l] = acc[l] + p * s`, every operation
//! rounded on its own (no FMA; Rust never contracts). The row result is
//! `acc[0] + acc[1] + ... + acc[7]` left to right. The AVX2 path and the scalar path execute
//! exactly these operations, so they are bit-identical, and so is every thread count (threads
//! split rows, never a sum). Each term passes at most `K/16 + 10` roundings, which gives the
//! bound the tests hold: `|y - y_exact| <= gamma(K/16 + 10) * sum |w x|`.

use crate::cnq::ue4m3;
use std::sync::Barrier;

/// bytes of one NVFP4 block (4 scale bytes + 32 code bytes)
pub const BLOCK_BYTES: usize = 36;
/// values of one NVFP4 block
pub const BLOCK_VALUES: usize = 64;
/// GLM-5.3-Flash `hidden_size`
pub const GLM_HIDDEN: usize = 4096;
/// GLM-5.3-Flash `moe_intermediate_size`
pub const GLM_INTER: usize = 2048;
/// one GLM-5.3-Flash routed expert unit: gate + up + down (`docs/glm-entropy.md` section 2)
pub const GLM_UNIT_BYTES: usize = 3 * GLM_HIDDEN * GLM_INTER / BLOCK_VALUES * BLOCK_BYTES;
const _: () = assert!(GLM_UNIT_BYTES == 14_155_776);

/// tokens computed per pass over a weight row (the decoded codes are reused per token)
const TOK_TILE: usize = 4;

/// E2M1 code -> value, the table of `cnq::e2m1` (asserted equal in the tests)
const E2M1: [f32; 16] = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0];

/// which implementation computes the rows
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Path {
    /// AVX2 when the CPU has it, else scalar
    Auto,
    /// the portable path
    Scalar,
    /// AVX2; panics on a CPU without it
    Avx2,
}

/// does this CPU run the AVX2 path
pub fn avx2_available() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        std::arch::is_x86_feature_detected!("avx2")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

fn use_avx2(path: Path) -> bool {
    match path {
        Path::Auto => avx2_available(),
        Path::Scalar => false,
        Path::Avx2 => {
            assert!(avx2_available(), "cpu_nvfp4: Path::Avx2 on a CPU without AVX2");
            true
        }
    }
}

/// The scale byte the engine decodes: `residency::sanitize_sf_slab`'s rule, 0x7F -> 0x7E.
#[inline]
pub fn scale_rule(byte: u8) -> u8 {
    if byte == 0x7F {
        0x7E
    } else {
        byte
    }
}

/// f32 sub-block scale for every scale byte: `ue4m3(scale_rule(b)) * global_scale`
pub fn scale_lut(global_scale: f32) -> [f32; 256] {
    let mut lut = [0f32; 256];
    for (b, v) in lut.iter_mut().enumerate() {
        *v = ue4m3(scale_rule(b as u8) as u32) * global_scale;
    }
    lut
}

/// one NVFP4 weight matrix `[rows, cols]` over borrowed bytes
#[derive(Clone, Copy, Debug)]
pub struct Nvfp4Matrix<'a> {
    pub bytes: &'a [u8],
    pub rows: usize,
    pub cols: usize,
    pub global_scale: f32,
}

impl<'a> Nvfp4Matrix<'a> {
    /// bytes of a `[rows, cols]` NVFP4 matrix
    pub const fn byte_len(rows: usize, cols: usize) -> usize {
        rows * (cols / BLOCK_VALUES) * BLOCK_BYTES
    }

    pub fn new(bytes: &'a [u8], rows: usize, cols: usize, global_scale: f32) -> Result<Self, String> {
        if cols == 0 || cols % BLOCK_VALUES != 0 {
            return Err(format!("cpu_nvfp4: {cols} columns are not whole 64-value NVFP4 blocks"));
        }
        let want = Self::byte_len(rows, cols);
        if bytes.len() != want {
            return Err(format!("cpu_nvfp4: [{rows}, {cols}] needs {want} B, got {}", bytes.len()));
        }
        Ok(Nvfp4Matrix { bytes, rows, cols, global_scale })
    }

    fn row_bytes(&self) -> usize {
        self.cols / BLOCK_VALUES * BLOCK_BYTES
    }

    fn row(&self, r: usize) -> &'a [u8] {
        let n = self.row_bytes();
        &self.bytes[r * n..(r + 1) * n]
    }
}

/// one routed expert: gate `[inter, hidden]`, up `[inter, hidden]`, down `[hidden, inter]`
#[derive(Clone, Copy, Debug)]
pub struct ExpertBlock<'a> {
    pub gate: Nvfp4Matrix<'a>,
    pub up: Nvfp4Matrix<'a>,
    pub down: Nvfp4Matrix<'a>,
    pub hidden: usize,
    pub inter: usize,
}

impl<'a> ExpertBlock<'a> {
    pub fn new(gate: Nvfp4Matrix<'a>, up: Nvfp4Matrix<'a>, down: Nvfp4Matrix<'a>) -> Result<Self, String> {
        let (inter, hidden) = (gate.rows, gate.cols);
        if (up.rows, up.cols) != (inter, hidden) || (down.rows, down.cols) != (hidden, inter) {
            return Err(format!(
                "cpu_nvfp4: expert shapes gate [{}, {}], up [{}, {}], down [{}, {}] do not fit together",
                gate.rows, gate.cols, up.rows, up.cols, down.rows, down.cols
            ));
        }
        Ok(ExpertBlock { gate, up, down, hidden, inter })
    }

    /// One expert unit as the GLM converter writes it: gate | up | down back to back, global
    /// scales in that order (`converter/src/main.rs` `write_units`).
    pub fn from_unit(unit: &'a [u8], hidden: usize, inter: usize, global_scales: [f32; 3]) -> Result<Self, String> {
        let gu = Nvfp4Matrix::byte_len(inter, hidden);
        let dn = Nvfp4Matrix::byte_len(hidden, inter);
        if unit.len() != 2 * gu + dn {
            return Err(format!("cpu_nvfp4: expert unit H {hidden} I {inter} needs {} B, got {}", 2 * gu + dn, unit.len()));
        }
        ExpertBlock::new(
            Nvfp4Matrix::new(&unit[..gu], inter, hidden, global_scales[0])?,
            Nvfp4Matrix::new(&unit[gu..2 * gu], inter, hidden, global_scales[1])?,
            Nvfp4Matrix::new(&unit[2 * gu..], hidden, inter, global_scales[2])?,
        )
    }
}

/// position of value `j` in the lane order: per 16-value group, even values first, then odd
#[inline]
fn lane_pos(j: usize) -> usize {
    let k = j & 15;
    (j & !15) | if k & 1 == 0 { k >> 1 } else { 8 + (k >> 1) }
}

/// `[T][cols]` activations into the lane order
fn permute(x: &[f32], cols: usize) -> Vec<f32> {
    let mut out = vec![0f32; x.len()];
    for (src, dst) in x.chunks_exact(cols).zip(out.chunks_exact_mut(cols)) {
        for (j, &v) in src.iter().enumerate() {
            dst[lane_pos(j)] = v;
        }
    }
    out
}

/// rows `n * w / k .. n * (w + 1) / k` of worker `w`
#[inline]
fn split(n: usize, k: usize, w: usize) -> (usize, usize) {
    (n * w / k, n * (w + 1) / k)
}

#[inline]
fn row_scalar<const NT: usize>(row: &[u8], lut: &[f32; 256], xp: &[f32], cols: usize, t0: usize, out: &mut [f32]) {
    let mut acc = [[0f32; 8]; NT];
    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        for sb in 0..4 {
            let s = lut[blk[sb] as usize];
            let codes = &blk[4 + sb * 8..4 + sb * 8 + 8];
            let mut lo = [0f32; 8];
            let mut hi = [0f32; 8];
            for l in 0..8 {
                lo[l] = E2M1[(codes[l] & 0xF) as usize];
                hi[l] = E2M1[(codes[l] >> 4) as usize];
            }
            let base = b * BLOCK_VALUES + sb * 16;
            for (t, a) in acc.iter_mut().enumerate() {
                let x = &xp[(t0 + t) * cols + base..(t0 + t) * cols + base + 16];
                for l in 0..8 {
                    let p = lo[l] * x[l] + hi[l] * x[8 + l];
                    a[l] += p * s;
                }
            }
        }
    }
    for (t, a) in acc.iter().enumerate() {
        let mut r = 0f32;
        for v in a {
            r += *v;
        }
        out[t] = r;
    }
}

/// # Safety
/// The CPU must have AVX2.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn row_avx2<const NT: usize>(row: &[u8], lut: &[f32; 256], xp: &[f32], cols: usize, t0: usize, out: &mut [f32]) {
    use std::arch::x86_64::*;
    assert!(xp.len() >= (t0 + NT) * cols && out.len() >= NT && row.len() == cols / BLOCK_VALUES * BLOCK_BYTES);
    let mag = _mm256_setr_ps(0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0);
    let lo4 = _mm256_set1_epi32(0xF);
    let bit3 = _mm256_set1_epi32(0x8);
    let mut acc = [_mm256_setzero_ps(); NT];
    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
        for sb in 0..4 {
            let s = _mm256_set1_ps(lut[blk[sb] as usize]);
            // SAFETY: bytes 4 + 8 sb .. 12 + 8 sb lie inside the 36-byte block
            let raw = unsafe { _mm_loadl_epi64(blk.as_ptr().add(4 + sb * 8) as *const __m128i) };
            let v = _mm256_cvtepu8_epi32(raw);
            let nib_lo = _mm256_and_si256(v, lo4);
            let nib_hi = _mm256_srli_epi32::<4>(v);
            // magnitude by table (permutevar reads the low 3 bits), sign bit from code bit 3
            let w_lo = _mm256_xor_ps(
                _mm256_permutevar8x32_ps(mag, nib_lo),
                _mm256_castsi256_ps(_mm256_slli_epi32::<28>(_mm256_and_si256(nib_lo, bit3))),
            );
            let w_hi = _mm256_xor_ps(
                _mm256_permutevar8x32_ps(mag, nib_hi),
                _mm256_castsi256_ps(_mm256_slli_epi32::<28>(_mm256_and_si256(nib_hi, bit3))),
            );
            let base = b * BLOCK_VALUES + sb * 16;
            for (t, a) in acc.iter_mut().enumerate() {
                // SAFETY: (t0 + t) * cols + base + 16 <= (t0 + NT) * cols <= xp.len() (asserted)
                let (x0, x1) = unsafe {
                    let p = xp.as_ptr().add((t0 + t) * cols + base);
                    (_mm256_loadu_ps(p), _mm256_loadu_ps(p.add(8)))
                };
                let p = _mm256_add_ps(_mm256_mul_ps(w_lo, x0), _mm256_mul_ps(w_hi, x1));
                *a = _mm256_add_ps(*a, _mm256_mul_ps(p, s));
            }
        }
    }
    for (t, a) in acc.iter().enumerate() {
        let mut lanes = [0f32; 8];
        // SAFETY: 8 f32 into an 8-element array
        unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), *a) };
        let mut r = 0f32;
        for v in lanes {
            r += v;
        }
        out[t] = r;
    }
}

/// One weight row against tokens `t0 .. t0 + nt` (`nt <= TOK_TILE`) of the lane-ordered `xp`.
#[inline]
fn row_dot(avx2: bool, row: &[u8], lut: &[f32; 256], xp: &[f32], cols: usize, t0: usize, nt: usize, out: &mut [f32]) {
    #[cfg(target_arch = "x86_64")]
    if avx2 {
        // SAFETY: `avx2` is true only after `use_avx2` saw the feature on this CPU
        unsafe {
            match nt {
                1 => row_avx2::<1>(row, lut, xp, cols, t0, out),
                2 => row_avx2::<2>(row, lut, xp, cols, t0, out),
                3 => row_avx2::<3>(row, lut, xp, cols, t0, out),
                _ => row_avx2::<4>(row, lut, xp, cols, t0, out),
            }
        }
        return;
    }
    let _ = avx2;
    match nt {
        1 => row_scalar::<1>(row, lut, xp, cols, t0, out),
        2 => row_scalar::<2>(row, lut, xp, cols, t0, out),
        3 => row_scalar::<3>(row, lut, xp, cols, t0, out),
        _ => row_scalar::<4>(row, lut, xp, cols, t0, out),
    }
}

/// Raw output pointer shared by the workers; each worker writes disjoint indices only.
#[derive(Clone, Copy)]
struct Out(*mut f32, usize);
// SAFETY: workers write disjoint indices (row ranges from `split`), readers start after a
// `Barrier::wait` or after the scope joined
unsafe impl Send for Out {}
unsafe impl Sync for Out {}
impl Out {
    #[inline]
    fn put(self, i: usize, v: f32) {
        assert!(i < self.1);
        // SAFETY: in bounds (asserted), index owned by exactly one worker
        unsafe { *self.0.add(i) = v };
    }
}

/// all rows `r0 .. r1` of `m` against every token of `xp`, written to `out[t * m.rows + r]`
fn rows_into(avx2: bool, m: &Nvfp4Matrix, lut: &[f32; 256], xp: &[f32], tokens: usize, r0: usize, r1: usize, out: Out) {
    let mut tmp = [0f32; TOK_TILE];
    for r in r0..r1 {
        let row = m.row(r);
        let mut t0 = 0;
        while t0 < tokens {
            let nt = (tokens - t0).min(TOK_TILE);
            row_dot(avx2, row, lut, xp, m.cols, t0, nt, &mut tmp);
            for t in 0..nt {
                out.put((t0 + t) * m.rows + r, tmp[t]);
            }
            t0 += nt;
        }
    }
}

/// `y[t][r] = sum_k W[r][k] * x[t][k]` for `x` `[T][cols]`, `y` `[T][rows]`, rows split over
/// `threads` workers (the caller's thread among them).
pub fn gemv(m: &Nvfp4Matrix, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    assert!(x.len() % m.cols == 0, "cpu_nvfp4::gemv: x is not [T][{}]", m.cols);
    let tokens = x.len() / m.cols;
    assert_eq!(y.len(), tokens * m.rows, "cpu_nvfp4::gemv: y is not [T][{}]", m.rows);
    let avx2 = use_avx2(path);
    let xp = permute(x, m.cols);
    let lut = scale_lut(m.global_scale);
    let k = threads.clamp(1, m.rows.max(1));
    let out = Out(y.as_mut_ptr(), y.len());
    let work = |w: usize| {
        let (r0, r1) = split(m.rows, k, w);
        rows_into(avx2, m, &lut, &xp, tokens, r0, r1, out);
    };
    if k == 1 {
        work(0);
    } else {
        let work = &work;
        std::thread::scope(|s| {
            for w in 1..k {
                s.spawn(move || work(w));
            }
            work(0);
        });
    }
}

/// The expert FFN `y = down(silu(gate(x)) * up(x))` for `x`, `y` `[T][hidden]` (decode is T 1).
/// One scope: phase 1 splits the `inter` rows of gate and up over the workers, a barrier, phase
/// 2 splits the `hidden` rows of down.
pub fn expert_ffn(e: &ExpertBlock, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    let (h, i) = (e.hidden, e.inter);
    assert!(x.len() % h == 0, "cpu_nvfp4::expert_ffn: x is not [T][{h}]");
    let tokens = x.len() / h;
    assert_eq!(y.len(), tokens * h, "cpu_nvfp4::expert_ffn: y is not [T][{h}]");
    let avx2 = use_avx2(path);
    let xp = permute(x, h);
    let (lg, lu, ld) = (scale_lut(e.gate.global_scale), scale_lut(e.up.global_scale), scale_lut(e.down.global_scale));
    // the activation, written straight into the lane order the down rows read
    let mut hp = vec![0f32; tokens * i];
    let hp_out = Out(hp.as_mut_ptr(), hp.len());
    let y_out = Out(y.as_mut_ptr(), y.len());
    let k = threads.clamp(1, i.min(h).max(1));
    let barrier = Barrier::new(k);
    let work = |w: usize| {
        let (j0, j1) = split(i, k, w);
        let (mut g, mut u) = ([0f32; TOK_TILE], [0f32; TOK_TILE]);
        for j in j0..j1 {
            let (gr, ur) = (e.gate.row(j), e.up.row(j));
            let mut t0 = 0;
            while t0 < tokens {
                let nt = (tokens - t0).min(TOK_TILE);
                row_dot(avx2, gr, &lg, &xp, h, t0, nt, &mut g);
                row_dot(avx2, ur, &lu, &xp, h, t0, nt, &mut u);
                for t in 0..nt {
                    let gate = g[t];
                    hp_out.put((t0 + t) * i + lane_pos(j), (gate / (1.0 + (-gate).exp())) * u[t]);
                }
                t0 += nt;
            }
        }
        barrier.wait();
        // SAFETY: every write to `hp` happened before the barrier; from here on it is read only
        let hp_read = unsafe { std::slice::from_raw_parts(hp_out.0 as *const f32, hp_out.1) };
        let (r0, r1) = split(h, k, w);
        rows_into(avx2, &e.down, &ld, hp_read, tokens, r0, r1, y_out);
    };
    if k == 1 {
        work(0);
    } else {
        let work = &work;
        std::thread::scope(|s| {
            for w in 1..k {
                s.spawn(move || work(w));
            }
            work(0);
        });
    }
    drop(hp);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cnq::e2m1;

    /// splitmix64: deterministic synthetic blocks without a dependency
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        fn byte(&mut self) -> u8 {
            self.next() as u8
        }
        /// uniform in [-a, a)
        fn f(&mut self, a: f32) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
        }
    }

    /// Synthetic NVFP4 matrix: random codes, scale bytes from `scale(rng)`.
    fn synth(rows: usize, cols: usize, rng: &mut Rng, scale: &mut dyn FnMut(&mut Rng) -> u8, code_mask: u8) -> Vec<u8> {
        let mut v = vec![0u8; Nvfp4Matrix::byte_len(rows, cols)];
        for blk in v.chunks_exact_mut(BLOCK_BYTES) {
            for b in blk.iter_mut().take(4) {
                *b = scale(rng);
            }
            for b in blk.iter_mut().skip(4) {
                *b = rng.byte() & code_mask;
            }
        }
        v
    }

    /// realistic scale bytes: 0x30..0x7F (2^-1 .. 480 before the rule), ~1 in 16 is 0x7F
    fn scale_mix(rng: &mut Rng) -> u8 {
        let b = rng.byte();
        if b & 15 == 0 {
            0x7F
        } else {
            0x30 + (b >> 4) * 3
        }
    }

    fn xs(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| rng.f(2.0)).collect()
    }

    /// a row as the engine serves it: `rule` runs the engine's own `residency::sanitize_sf_slab`
    /// on a copy (independent of this module's `scale_rule`); false leaves 0x7F (read as 480)
    fn served_row(m: &Nvfp4Matrix, r: usize, rule: bool) -> Vec<u8> {
        let mut row = m.row(r).to_vec();
        if rule {
            crate::residency::sanitize_sf_slab(&mut row);
        }
        row
    }

    /// dequantize-then-matmul in f64: (y_ref, sum |w x|) per [t][r]
    fn reference(m: &Nvfp4Matrix, x: &[f32], rule: bool) -> (Vec<f64>, Vec<f64>) {
        let tokens = x.len() / m.cols;
        let (mut y, mut a) = (vec![0f64; tokens * m.rows], vec![0f64; tokens * m.rows]);
        for r in 0..m.rows {
            let row = served_row(m, r, rule);
            for t in 0..tokens {
                let (mut s1, mut s2) = (0f64, 0f64);
                for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
                    for idx in 0..BLOCK_VALUES {
                        let s = ue4m3(blk[idx / 16] as u32) * m.global_scale; // the engine's f32 scale
                        let nib = (blk[4 + idx / 2] >> (4 * (idx % 2))) & 0xF;
                        let w = e2m1(nib as u32) as f64 * s as f64;
                        let p = w * x[t * m.cols + b * BLOCK_VALUES + idx] as f64;
                        s1 += p;
                        s2 += p.abs();
                    }
                }
                y[t * m.rows + r] = s1;
                a[t * m.rows + r] = s2;
            }
        }
        (y, a)
    }

    /// gamma_n with n = K/16 + 10, u = 2^-24 (the ticket's bound, fixed before the first result)
    fn gamma(cols: usize) -> f64 {
        let nu = (cols / 16 + 10) as f64 * (2f64).powi(-24);
        nu / (1.0 - nu)
    }

    fn run(m: &Nvfp4Matrix, x: &[f32], threads: usize, path: Path) -> Vec<f32> {
        let mut y = vec![0f32; x.len() / m.cols * m.rows];
        gemv(m, x, &mut y, threads, path);
        y
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }

    /// worst |y - y_ref| / (gamma * sum|w x|) over all outputs (<= 1 holds the bound)
    fn worst_ratio(y: &[f32], yr: &[f64], ab: &[f64], cols: usize) -> f64 {
        let g = gamma(cols);
        y.iter().zip(yr).zip(ab).map(|((&a, &b), &s)| (a as f64 - b).abs() / (g * s).max(f64::MIN_POSITIVE)).fold(0.0, f64::max)
    }

    #[test]
    fn cpu_nvfp4_tables_are_the_engine_decoders() {
        for n in 0..16u32 {
            assert_eq!(E2M1[n as usize].to_bits(), e2m1(n).to_bits(), "code {n}");
        }
        let gs = 3.0517578e-5f32;
        let lut = scale_lut(gs);
        for b in 0..256usize {
            let want = if b == 0x7F { 448.0 * gs } else { ue4m3(b as u32) * gs };
            assert_eq!(lut[b].to_bits(), want.to_bits(), "scale byte {b:#04x}");
        }
        for j in 0..64 {
            let p = lane_pos(j);
            assert_eq!((p / 16, p % 16), (j / 16, if j % 2 == 0 { (j % 16) / 2 } else { 8 + (j % 16) / 2 }));
        }
    }

    #[test]
    fn cpu_nvfp4_shapes_are_checked() {
        let b = vec![0u8; Nvfp4Matrix::byte_len(3, 128)];
        assert!(Nvfp4Matrix::new(&b, 3, 128, 1.0).is_ok());
        assert!(Nvfp4Matrix::new(&b, 3, 96, 1.0).is_err());
        assert!(Nvfp4Matrix::new(&b[1..], 3, 128, 1.0).is_err());
        let unit = vec![0u8; GLM_UNIT_BYTES];
        let e = ExpertBlock::from_unit(&unit, GLM_HIDDEN, GLM_INTER, [1.0; 3]).unwrap();
        assert_eq!((e.gate.bytes.len(), e.up.bytes.len(), e.down.bytes.len()), (4_718_592, 4_718_592, 4_718_592));
        assert!(ExpertBlock::from_unit(&unit[..GLM_UNIT_BYTES - 36], GLM_HIDDEN, GLM_INTER, [1.0; 3]).is_err());
        let g = Nvfp4Matrix::new(&b, 3, 128, 1.0).unwrap();
        assert!(ExpertBlock::new(g, g, g).is_err());
    }

    /// Acceptance 1: every output within gamma(K/16 + 10) * sum|w x| of the f64 reference.
    #[test]
    fn cpu_nvfp4_gemv_holds_the_fp32_bound() {
        let mut rng = Rng(173);
        for &(rows, cols) in &[(37usize, 256usize), (16, 2048), (9, 4096)] {
            let bytes = synth(rows, cols, &mut rng, &mut scale_mix, 0xFF);
            let m = Nvfp4Matrix::new(&bytes, rows, cols, 1.7e-3).unwrap();
            for &t in &[1usize, 2, 3, 5, 8] {
                let x = xs(t * cols, &mut rng);
                let (yr, ab) = reference(&m, &x, true);
                for path in [Path::Scalar, Path::Auto] {
                    let y = run(&m, &x, 3, path);
                    let w = worst_ratio(&y, &yr, &ab, cols);
                    assert!(w <= 1.0, "[{rows}, {cols}] T {t} {path:?}: worst |dy| = {w:.3} x bound");
                }
            }
        }
    }

    /// Acceptance 2: AVX2 == scalar bit for bit, for every token count and thread count.
    #[test]
    fn cpu_nvfp4_avx2_equals_scalar_bits() {
        if !avx2_available() {
            eprintln!("cpu_nvfp4: no AVX2 on this CPU, the AVX2 path is not exercised");
            return;
        }
        let mut rng = Rng(0xC0FFEE);
        let (rows, cols) = (41usize, 1024usize);
        let bytes = synth(rows, cols, &mut rng, &mut |r| r.byte(), 0xFF); // every scale byte, bit 7 too
        let m = Nvfp4Matrix::new(&bytes, rows, cols, 2.3e-4).unwrap();
        for &t in &[1usize, 2, 3, 5, 8] {
            let x = xs(t * cols, &mut rng);
            let s1 = bits(&run(&m, &x, 1, Path::Scalar));
            for th in [1usize, 3, 8] {
                assert_eq!(bits(&run(&m, &x, th, Path::Scalar)), s1, "scalar T {t} threads {th}");
                assert_eq!(bits(&run(&m, &x, th, Path::Avx2)), s1, "avx2 T {t} threads {th}");
            }
        }
    }

    /// Acceptance 3: the 0x7F -> 0x7E rule. Positive codes and activations (no cancellation), so
    /// 448 vs 480 on half the sub-blocks moves every output far beyond the fp32 bound. Red with
    /// `scale_rule` returning the byte unchanged.
    #[test]
    fn cpu_nvfp4_scale_rule_is_the_slab_rule() {
        let mut rng = Rng(0x7F);
        let (rows, cols) = (12usize, 512usize);
        let bytes = synth(rows, cols, &mut rng, &mut |r| if r.byte() & 1 == 0 { 0x7F } else { 0x70 + (r.byte() & 7) }, 0x77);
        let m = Nvfp4Matrix::new(&bytes, rows, cols, 9.5e-4).unwrap();
        let x: Vec<f32> = (0..2 * cols).map(|_| rng.f(1.0).abs() + 0.01).collect();
        let (y448, ab448) = reference(&m, &x, true);
        let (y480, ab480) = reference(&m, &x, false);
        for path in [Path::Scalar, Path::Auto] {
            let y = run(&m, &x, 2, path);
            let in_448 = worst_ratio(&y, &y448, &ab448, cols);
            assert!(in_448 <= 1.0, "{path:?}: {in_448:.3} x bound off the 0x7F->0x7E reference");
            let g = gamma(cols);
            for (n, ((&v, &r), &s)) in y.iter().zip(&y480).zip(&ab480).enumerate() {
                assert!((v as f64 - r).abs() > g * s, "{path:?} output {n}: matches the 480 reading of 0x7F");
            }
        }
        // a slab sanitized by the engine's own function gives the same bits as the raw bytes
        let mut clean = bytes.clone();
        assert!(crate::residency::sanitize_sf_slab(&mut clean) > 0);
        let mc = Nvfp4Matrix::new(&clean, rows, cols, 9.5e-4).unwrap();
        assert_eq!(bits(&run(&mc, &x, 2, Path::Auto)), bits(&run(&m, &x, 2, Path::Auto)));
    }

    fn silu_mul(g: f64, u: f64) -> f64 {
        g / (1.0 + (-g).exp()) * u
    }

    /// Acceptance 4 (and 2 for the FFN): the fused FFN equals gemv + activation + gemv bit for
    /// bit, the down stage holds the bound on its own input, the whole FFN is within 1e-4 of an
    /// all-f64 reference, AVX2 == scalar and every thread count gives the same bits.
    #[test]
    fn cpu_nvfp4_expert_ffn_matches_reference() {
        let mut rng = Rng(0xF00D);
        let (h, i) = (192usize, 128usize);
        let gb = synth(i, h, &mut rng, &mut scale_mix, 0xFF);
        let ub = synth(i, h, &mut rng, &mut scale_mix, 0xFF);
        let db = synth(h, i, &mut rng, &mut scale_mix, 0xFF);
        let mut unit = gb.clone();
        unit.extend_from_slice(&ub);
        unit.extend_from_slice(&db);
        let gs = [2.1e-3f32, 1.9e-3, 2.6e-3];
        let e = ExpertBlock::from_unit(&unit, h, i, gs).unwrap();
        for &t in &[1usize, 2, 5] {
            let x = xs(t * h, &mut rng);
            let mut y = vec![0f32; t * h];
            expert_ffn(&e, &x, &mut y, 1, Path::Scalar);
            // the same arithmetic through the public gemv
            let (g, u) = (run(&e.gate, &x, 1, Path::Scalar), run(&e.up, &x, 1, Path::Scalar));
            let hk: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| (g / (1.0 + (-g).exp())) * u).collect();
            assert_eq!(bits(&run(&e.down, &hk, 1, Path::Scalar)), bits(&y), "T {t}: fused != staged");
            // down stage on the kernel's own activation: the fp32 bound
            let (yr, ab) = reference(&e.down, &hk, true);
            let w = worst_ratio(&y, &yr, &ab, i);
            assert!(w <= 1.0, "T {t}: down stage {w:.3} x bound");
            // end to end against f64 throughout
            let (gr, _) = reference(&e.gate, &x, true);
            let (ur, _) = reference(&e.up, &x, true);
            let h64: Vec<f64> = gr.iter().zip(&ur).map(|(&g, &u)| silu_mul(g, u)).collect();
            for tt in 0..t {
                for r in 0..h {
                    let (mut want, mut mag) = (0f64, 0f64);
                    let row = served_row(&e.down, r, true);
                    for (b, blk) in row.chunks_exact(BLOCK_BYTES).enumerate() {
                        for idx in 0..BLOCK_VALUES {
                            let s = ue4m3(blk[idx / 16] as u32) * gs[2];
                            let nib = (blk[4 + idx / 2] >> (4 * (idx % 2))) & 0xF;
                            let p = e2m1(nib as u32) as f64 * s as f64 * h64[tt * i + b * BLOCK_VALUES + idx];
                            want += p;
                            mag += p.abs();
                        }
                    }
                    let got = y[tt * h + r] as f64;
                    assert!((got - want).abs() <= 1e-4 * mag, "T {t} token {tt} row {r}: {got} vs {want} (sum|w h| {mag})");
                }
            }
            for th in [1usize, 3, 8] {
                for path in [Path::Scalar, Path::Auto] {
                    let mut y2 = vec![0f32; t * h];
                    expert_ffn(&e, &x, &mut y2, th, path);
                    assert_eq!(bits(&y2), bits(&y), "T {t} threads {th} {path:?}");
                }
            }
        }
    }

    /// One full-size GLM-5.3-Flash unit (14,155,776 B): AVX2 == scalar for the whole FFN, and the
    /// gate rows hold the bound at K 4096 (64 rows checked against f64).
    #[test]
    fn cpu_nvfp4_full_glm_unit() {
        let mut rng = Rng(0x6C4D);
        let mut unit = vec![0u8; GLM_UNIT_BYTES];
        for blk in unit.chunks_exact_mut(BLOCK_BYTES) {
            for b in blk.iter_mut().take(4) {
                *b = scale_mix(&mut rng);
            }
            for b in blk.iter_mut().skip(4) {
                *b = rng.byte();
            }
        }
        let e = ExpertBlock::from_unit(&unit, GLM_HIDDEN, GLM_INTER, [1.1e-3, 1.2e-3, 1.3e-3]).unwrap();
        let x = xs(GLM_HIDDEN, &mut rng);
        let mut ys = vec![0f32; GLM_HIDDEN];
        expert_ffn(&e, &x, &mut ys, 8, Path::Scalar);
        assert!(ys.iter().all(|v| v.is_finite()));
        if avx2_available() {
            let mut ya = vec![0f32; GLM_HIDDEN];
            expert_ffn(&e, &x, &mut ya, 8, Path::Avx2);
            assert_eq!(bits(&ya), bits(&ys));
        }
        let g64 = Nvfp4Matrix::new(&e.gate.bytes[..Nvfp4Matrix::byte_len(64, GLM_HIDDEN)], 64, GLM_HIDDEN, e.gate.global_scale).unwrap();
        let (yr, ab) = reference(&g64, &x, true);
        let w = worst_ratio(&run(&g64, &x, 4, Path::Auto), &yr, &ab, GLM_HIDDEN);
        assert!(w <= 1.0, "gate rows at K 4096: {w:.3} x bound");
    }

    /// Micro-benchmark, not a gate: one FFN over GLM units rotated across 8 distinct units
    /// (113 MB, more than the L3), T 1 and 4, a few thread counts. Expert bytes / wall time.
    /// `cargo test --release -p crow-nest-engine cpu_nvfp4_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cpu_nvfp4_bench() {
        let mut rng = Rng(1);
        let units: Vec<Vec<u8>> = (0..8)
            .map(|_| {
                let mut u = vec![0u8; GLM_UNIT_BYTES];
                for blk in u.chunks_exact_mut(BLOCK_BYTES) {
                    for b in blk.iter_mut().take(4) {
                        *b = scale_mix(&mut rng);
                    }
                    for b in blk.iter_mut().skip(4) {
                        *b = rng.byte();
                    }
                }
                u
            })
            .collect();
        let blocks: Vec<ExpertBlock> = units.iter().map(|u| ExpertBlock::from_unit(u, GLM_HIDDEN, GLM_INTER, [1e-3; 3]).unwrap()).collect();
        let path = if avx2_available() { Path::Avx2 } else { Path::Scalar };
        eprintln!("cpu_nvfp4 bench: {path:?}, {} threads available", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
        for &t in &[1usize, 4] {
            let x = xs(t * GLM_HIDDEN, &mut rng);
            let mut y = vec![0f32; t * GLM_HIDDEN];
            for &th in &[1usize, 4, 8, 16, 24] {
                for b in &blocks {
                    expert_ffn(b, &x, &mut y, th, path); // warm-up: page faults, thread start
                }
                let mut ms = Vec::new();
                for rep in 0..48 {
                    let t0 = std::time::Instant::now();
                    expert_ffn(&blocks[rep % blocks.len()], &x, &mut y, th, path);
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = ms[ms.len() / 2];
                eprintln!(
                    "  T {t} threads {th:2}: median {med:.3} ms per unit ({:.2} GB/s of expert bytes), min {:.3} max {:.3} ms, n {}",
                    GLM_UNIT_BYTES as f64 / (med * 1e-3) / 1e9,
                    ms[0],
                    ms[ms.len() - 1],
                    ms.len()
                );
            }
        }
    }
}
