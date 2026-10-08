//! crow-nest #180: the CPU MUL1 trellis expert GEMV and FFN (plan step 10, CPU half; the GPU half
//! is `kernels::mul1` / `kernels_mul1.cu`). A routed GLM-5.3-Flash expert stored as a CNQ MUL1
//! record (#181, `converter/src/mul1.rs`) is computed where it lies, without a dequant pass.
//! Nothing in the engine calls this yet (the CPU-lane wiring is plan step 19).
//!
//! # Format (the #181 record, bit for bit)
//!
//! - One expert record: `[gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh]
//!   [up.suh][up.svh][down.suh][down.svh] [zeros to a multiple of 4096]`. gate and up are
//!   `[k = hidden, n = inter]`, down is `[inter, hidden]` (exllamav3 stores `[in, out]`). One GLM
//!   expert at K = 3: 3 x 3,145,728 B + 36,864 B = 9,474,048 B.
//! - A trellis is `(k/16)(n/16)` tiles of `16 K` u16 words; tile `kb * n/16 + nb` holds rows
//!   `16 kb ..`, columns `16 nb ..`. Trellis position `8 L + j` (lane L, j = 0..7) is the 16-bit
//!   window ending at bit `end_bit(8 L + j)` of the tile's circular MSB-first stream of
//!   little-endian u32 words, at row `2 (L % 4) + [0, 1, 8, 9][j % 4]`, column `L / 4 + 8 (j / 4)`.
//! - A state decodes to the codec's fp16 value `w = rne16(s * k_inv - 3.453125)` with
//!   `s = bytesum(state * 0x83DCD12D)`, `k_inv = fp16(0x1eee)` (exllamav3 `codebook.cuh:38-52`
//!   at `151539c7`; #181 `mul1_decode`). `s * k_inv - 3.453125` is exact in f32, so `w` is one RNE
//!   rounding to fp16, held exactly in f32 (the scalar path reads a 1021-entry table, the AVX2 path
//!   rounds with sybil's magic add, `ft_core.h:218-246`; both tested against the codec for every
//!   state).
//! - `y = x W`, `W = diag(suh) H W_hat H diag(svh) / 128`, H the 128-wide Sylvester Hadamard:
//!   exllamav3 `LinearEXL3.get_weight_tensor` / `reconstruct_hgemm` (`exl3.py:193-249`). The
//!   kernel computes `xh = H (x * suh)`, `y' = xh W_hat` (decoding W_hat in the inner loop) and
//!   `y = (H y') / 128 * svh`. Activations stay f32 throughout: exllamav3's own CPU path
//!   (`moe_mul1.cpp:35-45`) quantizes them to int8 (~0.9 % output RMS in its author's words); that
//!   lossy arm is out of this module.
//!
//! # The f32 order (one order, both paths)
//!
//! `x * suh` (1 rounding), the natural-order FWHT in 7 stages `(a, b) -> (a + b, a - b)` (7).
//! For every output tile column, per token, 8 x 8 lane accumulators `acc[C][j]` (sybil's layout,
//! `ft_core.h:218-246`): for each tile row kb ascending, C = 0..8, q = 0..4, trellis lane
//! `g = 4 C + q` decodes its 8 weights and `acc[C][j] = acc[C][j] + w[j] * xh[16 kb + 2 q +
//! [0, 1, 8, 9][j % 4]]`, product and sum rounded on their own (no FMA; Rust never contracts):
//! a chain of `k / 4` roundings. Column `16 nb + C` is `(acc[C][0] + acc[C][1]) + (acc[C][2] +
//! acc[C][3])`, column `16 nb + C + 8` the same over j = 4..7 (2). Then the FWHT (7), `* 2^-7`
//! (exact) and `* svh` (1). Every term passes at most `n = k / 4 + 18` roundings
//! ([`rounding_steps`]). AVX2 and scalar execute exactly these operations, threads split output
//! columns and never a sum, so every path and thread count gives the same bits. The bound the
//! tests hold, its form and the FFN chain are in `docs/mul1-gemv.md` section 3 (amended on #180,
//! 2026-10-09, before the first comparison).
//!
//! Written after exllamav3 `exllamav3_ext/cpu/moe_mul1.cpp` (turboderp-org/exllamav3 @
//! `151539c7`, MIT) and sybil-solutions/glm53-flash-offload `kernels/cpu_avx2/ft_core.h`
//! (`df0b439`, MIT, Copyright (c) 2026 0xSero): the decode identity, the k-major tile walk over a
//! band of output tiles and the stream-order accumulator layout. Code is not copied; both are MIT.
//! API and threading are those of `cpu_nvfp4` (#173): rows (here output columns) split over
//! `threads` scoped workers, one barrier between the FFN phases.

pub use crate::cpu_nvfp4::{avx2_available, Path};
use std::sync::Barrier;

/// GLM-5.3-Flash `hidden_size`
pub const GLM_HIDDEN: usize = 4096;
/// GLM-5.3-Flash `moe_intermediate_size`
pub const GLM_INTER: usize = 2048;
/// one GLM-5.3-Flash expert record at K = 3 (#181 `glm_record_size_at_3_bit`)
pub const GLM_RECORD_BYTES_K3: usize = 9_474_048;
/// the record alignment of #181 (`EXPERT_ALIGN`)
pub const RECORD_ALIGN: usize = 4096;
/// the Hadamard block
pub const HAD: usize = 128;

/// tokens computed per pass over the trellis (the decoded weights are reused per token)
const TOK_TILE: usize = 4;
/// output tile columns per work unit (64 outputs); a thread owns whole units
const UNIT_TILES: usize = 4;
/// tile rows the AVX2 path prefetches ahead
const PF_ROWS: usize = 6;

/// `0x1eee` = 1774 * 2^-18
const KINV: f32 = 1774.0 / 262144.0;
/// `1024 * k_inv + fp16(0xc931)` = 6.9296875 - 10.3828125
const CAFF: f32 = -3.453125;
const MUL1: u32 = 0x83DC_D12D;
const ROW_OFF: [usize; 4] = [0, 1, 8, 9];

/// The f32 rounding count of one GEMV with input width `k` (module doc).
pub const fn rounding_steps(k: usize) -> usize {
    k / 4 + 18
}

/// Bitrate of a trellis: `bits` per step plus one more on every odd step when `half`
/// (#181 `Bitrate`, the set exllamav3's `bits_from_K` accepts for mul1).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitrate {
    pub bits: u32,
    pub half: bool,
}

impl Bitrate {
    pub fn from_k(k: f64) -> Result<Bitrate, String> {
        let bits = k.trunc();
        let f = k - bits;
        let ok = (1.0..=8.0).contains(&bits) && (f == 0.0 || (f == 0.5 && bits <= 3.0));
        if !ok {
            return Err(format!("cpu_mul1: unsupported bitrate {k} (integer 1..8, or 1.5 / 2.5 / 3.5)"));
        }
        Ok(Bitrate { bits: bits as u32, half: f == 0.5 })
    }

    pub fn k(self) -> f64 {
        self.bits as f64 + if self.half { 0.5 } else { 0.0 }
    }

    /// u16 words per 16x16 tile, `16 K`
    pub fn words_per_tile(self) -> usize {
        16 * self.bits as usize + if self.half { 8 } else { 0 }
    }

    pub fn tile_bytes(self) -> usize {
        2 * self.words_per_tile()
    }

    /// exclusive end bit of position `t` in the tile's stream
    pub fn end_bit(self, t: usize) -> usize {
        let b = self.bits as usize;
        if !self.half {
            (t + 1) * b
        } else if t % 2 == 1 {
            (t / 2 + 1) * (2 * b + 1)
        } else {
            (t / 2) * (2 * b + 1) + b
        }
    }
}

fn use_avx2(path: Path) -> bool {
    match path {
        Path::Auto => avx2_available(),
        Path::Scalar => false,
        Path::Avx2 => {
            assert!(avx2_available(), "cpu_mul1: Path::Avx2 on a CPU without AVX2");
            true
        }
    }
}

/// fp16 bits -> f32 (exact)
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as i32;
    let m = (h & 0x3ff) as f32;
    match e {
        0 => sign * m * (2f32).powi(-24),
        31 => {
            if m == 0.0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => sign * (1024.0 + m) * (2f32).powi(e - 25),
    }
}

/// `v` rounded once, to nearest even, to fp16 precision (finite, normal-range fp16 results)
fn rne16(v: f32) -> f32 {
    let a = (v as f64).abs();
    if a == 0.0 {
        return v;
    }
    let e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
    let ulp = (2f64).powi(e.max(-14) - 10);
    ((v as f64 / ulp).round_ties_even() * ulp) as f32
}

/// the weight of byte sum `s` (0..=1020): `rne16(s * k_inv - 3.453125)`
fn weight_lut() -> &'static [f32; 1021] {
    static LUT: std::sync::OnceLock<[f32; 1021]> = std::sync::OnceLock::new();
    LUT.get_or_init(|| {
        let mut t = [0f32; 1021];
        for (s, w) in t.iter_mut().enumerate() {
            *w = rne16(s as f32 * KINV + CAFF);
        }
        t
    })
}

#[inline]
fn bytesum(state: u32) -> usize {
    let x = state.wrapping_mul(MUL1);
    ((x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24)) as usize
}

/// The codec weight of one 16-bit trellis state, as the scalar path decodes it.
pub fn weight(state: u16) -> f32 {
    weight_lut()[bytesum(state as u32)]
}

#[inline]
fn rd32(tile: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([tile[4 * i], tile[4 * i + 1], tile[4 * i + 2], tile[4 * i + 3]])
}

/// state of the 16-bit window starting at stream bit `lo` (MSB first over the u32 words)
#[inline]
fn state_at(tile: &[u8], n32: usize, lo: usize) -> u32 {
    let (i, o) = (lo >> 5, lo & 31);
    let a = rd32(tile, i) as u64;
    let b = rd32(tile, if i + 1 == n32 { 0 } else { i + 1 }) as u64;
    (((a << 32 | b) >> (48 - o)) & 0xffff) as u32
}

/// the AVX2 view of one trellis lane: a 4-word window and, per position, the two words and the shift
#[derive(Clone, Copy, Debug)]
struct Group {
    w0: usize,
    wrap: bool,
    ia: [i32; 8],
    ib: [i32; 8],
    sh: [i32; 8],
    shr: [i32; 8],
}

/// per-bitrate position tables
#[derive(Clone, Debug)]
struct Tables {
    n32: usize,
    /// window start bit of every trellis position
    lo: [usize; 256],
    groups: [Group; 32],
}

impl Tables {
    fn new(b: Bitrate) -> Tables {
        let n32 = b.words_per_tile() / 2;
        let s = 32 * n32;
        let mut lo = [0usize; 256];
        for (p, l) in lo.iter_mut().enumerate() {
            *l = (b.end_bit(p) + s - 16) % s;
        }
        let mut groups = [Group { w0: 0, wrap: false, ia: [0; 8], ib: [0; 8], sh: [0; 8], shr: [0; 8] }; 32];
        for (g, gr) in groups.iter_mut().enumerate() {
            let w0 = lo[8 * g] / 32;
            gr.w0 = w0;
            gr.wrap = w0 + 3 >= n32;
            for j in 0..8 {
                let rel = (lo[8 * g + j] + s - 32 * w0) % s;
                let ia = rel / 32;
                assert!(ia + 1 <= 3, "cpu_mul1: K = {} lane {g} position {j} leaves the 4-word window", b.k());
                gr.ia[j] = ia as i32;
                gr.ib[j] = ia as i32 + 1;
                gr.sh[j] = (rel % 32) as i32;
                gr.shr[j] = 32 - (rel % 32) as i32;
            }
        }
        Tables { n32, lo, groups }
    }
}

/// One MUL1 linear `[k = in, n = out]` over borrowed record bytes.
#[derive(Clone, Copy, Debug)]
pub struct Mul1Matrix<'a> {
    pub trellis: &'a [u8],
    /// fp16 LE, `k` values
    pub suh: &'a [u8],
    /// fp16 LE, `n` values
    pub svh: &'a [u8],
    pub k: usize,
    pub n: usize,
    pub bitrate: Bitrate,
}

impl<'a> Mul1Matrix<'a> {
    pub fn trellis_bytes(k: usize, n: usize, b: Bitrate) -> usize {
        k / 16 * (n / 16) * b.tile_bytes()
    }

    pub fn new(trellis: &'a [u8], suh: &'a [u8], svh: &'a [u8], k: usize, n: usize, bitrate: Bitrate) -> Result<Self, String> {
        if k == 0 || n == 0 || k % HAD != 0 || n % HAD != 0 {
            return Err(format!("cpu_mul1: [{k}, {n}] is not a positive multiple of 128 on both sides"));
        }
        let want = Self::trellis_bytes(k, n, bitrate);
        if trellis.len() != want || suh.len() != 2 * k || svh.len() != 2 * n {
            return Err(format!(
                "cpu_mul1: [{k}, {n}] K = {} needs trellis {want} B, suh {} B, svh {} B; got {}, {}, {}",
                bitrate.k(),
                2 * k,
                2 * n,
                trellis.len(),
                suh.len(),
                svh.len()
            ));
        }
        Ok(Mul1Matrix { trellis, suh, svh, k, n, bitrate })
    }

    fn scale(v: &[u8], i: usize) -> f32 {
        f16_to_f32(u16::from_le_bytes([v[2 * i], v[2 * i + 1]]))
    }
}

/// The bytes of one record: payload rounded up to [`RECORD_ALIGN`] (#181 `RecordLayout::size`).
pub fn record_bytes(hidden: usize, inter: usize, b: Bitrate) -> usize {
    let payload = 3 * Mul1Matrix::trellis_bytes(hidden, inter, b) + 6 * (hidden + inter);
    payload.div_ceil(RECORD_ALIGN) * RECORD_ALIGN
}

/// one routed expert: gate `[hidden, inter]`, up `[hidden, inter]`, down `[inter, hidden]`
#[derive(Clone, Copy, Debug)]
pub struct Mul1Expert<'a> {
    pub gate: Mul1Matrix<'a>,
    pub up: Mul1Matrix<'a>,
    pub down: Mul1Matrix<'a>,
    pub hidden: usize,
    pub inter: usize,
}

impl<'a> Mul1Expert<'a> {
    /// One record as #181 `write_record` lays it out.
    pub fn from_record(rec: &'a [u8], hidden: usize, inter: usize, b: Bitrate) -> Result<Self, String> {
        let size = record_bytes(hidden, inter, b);
        if rec.len() != size {
            return Err(format!("cpu_mul1: record H {hidden} I {inter} K = {} is {size} B, got {}", b.k(), rec.len()));
        }
        let tb = Mul1Matrix::trellis_bytes(hidden, inter, b);
        let (h2, i2) = (2 * hidden, 2 * inter);
        let mut at = 3 * tb;
        let mut take = |n: usize| {
            let s = &rec[at..at + n];
            at += n;
            s
        };
        let (gs, gv, us, uv, ds, dv) = (take(h2), take(i2), take(h2), take(i2), take(i2), take(h2));
        Ok(Mul1Expert {
            gate: Mul1Matrix::new(&rec[..tb], gs, gv, hidden, inter, b)?,
            up: Mul1Matrix::new(&rec[tb..2 * tb], us, uv, hidden, inter, b)?,
            down: Mul1Matrix::new(&rec[2 * tb..3 * tb], ds, dv, inter, hidden, b)?,
            hidden,
            inter,
        })
    }
}

/// In-place natural-order Walsh-Hadamard transform of 128 values, stages h = 1, 2, .., 64,
/// `(a, b) -> (a + b, a - b)` (the order of `kernels_mul1.cu` `mul1_fwht128`).
pub fn fwht128(v: &mut [f32]) {
    debug_assert_eq!(v.len(), HAD);
    let mut h = 1;
    while h < HAD {
        let mut i = 0;
        while i < HAD {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
            i += 2 * h;
        }
        h *= 2;
    }
}

/// `xh = H (x * suh)` per 128-block, `x` `[T][k]`
fn had_in(x: &[f32], m: &Mul1Matrix) -> Vec<f32> {
    let mut xh = vec![0f32; x.len()];
    for (src, dst) in x.chunks_exact(m.k).zip(xh.chunks_exact_mut(m.k)) {
        for (i, (d, &s)) in dst.iter_mut().zip(src).enumerate() {
            *d = s * Mul1Matrix::scale(m.suh, i);
        }
        for blk in dst.chunks_exact_mut(HAD) {
            fwht128(blk);
        }
    }
    xh
}

/// `y = (H y') / 128 * svh` per 128-block, `raw`, `y` `[T][n]`
fn had_out(raw: &[f32], m: &Mul1Matrix, y: &mut [f32]) {
    for (src, dst) in raw.chunks_exact(m.n).zip(y.chunks_exact_mut(m.n)) {
        dst.copy_from_slice(src);
        for blk in dst.chunks_exact_mut(HAD) {
            fwht128(blk);
        }
        for (i, d) in dst.iter_mut().enumerate() {
            *d = (*d * (1.0 / 128.0)) * Mul1Matrix::scale(m.svh, i);
        }
    }
}

/// `xh` `[T][k]` into the lane order: `xp[t][kb][q][j] = xh[t][16 kb + 2 q + ROW_OFF[j % 4]]`
fn permute(xh: &[f32], k: usize) -> Vec<f32> {
    let tk = k / 16;
    let mut xp = vec![0f32; xh.len() / k * tk * 32];
    for (src, dst) in xh.chunks_exact(k).zip(xp.chunks_exact_mut(tk * 32)) {
        for kb in 0..tk {
            for q in 0..4 {
                for j in 0..8 {
                    dst[kb * 32 + q * 8 + j] = src[16 * kb + 2 * q + ROW_OFF[j % 4]];
                }
            }
        }
    }
    xp
}

/// columns `n * w / k .. n * (w + 1) / k` of worker `w`
#[inline]
fn split(n: usize, k: usize, w: usize) -> (usize, usize) {
    (n * w / k, n * (w + 1) / k)
}

/// Raw output pointer shared by the workers; each worker writes disjoint indices only.
#[derive(Clone, Copy)]
struct Out(*mut f32, usize);
// SAFETY: workers write disjoint indices (whole units from `split`), readers start after a
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

/// the two column sums of one accumulator row (module doc)
#[inline]
fn fold(a: &[f32; 8]) -> (f32, f32) {
    ((a[0] + a[1]) + (a[2] + a[3]), (a[4] + a[5]) + (a[6] + a[7]))
}

/// Raw `y'` of tile columns `nb0 .. nb0 + ntc` (`ntc <= UNIT_TILES`) for tokens `t0 .. t0 + NT`
/// of the lane-ordered `xp`, written to `raw[t * n + col]`.
fn unit_scalar<const NT: usize>(m: &Mul1Matrix, tab: &Tables, xp: &[f32], t0: usize, nb0: usize, ntc: usize, raw: Out) {
    let (tk, tn, tb) = (m.k / 16, m.n / 16, m.bitrate.tile_bytes());
    let lut = weight_lut();
    let mut acc = [[[[0f32; 8]; NT]; 8]; UNIT_TILES];
    for kb in 0..tk {
        for (tc, acc_tc) in acc.iter_mut().enumerate().take(ntc) {
            let off = (kb * tn + nb0 + tc) * tb;
            let tile = &m.trellis[off..off + tb];
            for (c, acc_c) in acc_tc.iter_mut().enumerate() {
                for q in 0..4 {
                    let g = 4 * c + q;
                    let mut w = [0f32; 8];
                    for (j, wj) in w.iter_mut().enumerate() {
                        *wj = lut[bytesum(state_at(tile, tab.n32, tab.lo[8 * g + j]))];
                    }
                    for (t, a) in acc_c.iter_mut().enumerate() {
                        let xv = &xp[((t0 + t) * tk + kb) * 32 + q * 8..][..8];
                        for j in 0..8 {
                            a[j] += w[j] * xv[j];
                        }
                    }
                }
            }
        }
    }
    for (tc, acc_tc) in acc.iter().enumerate().take(ntc) {
        for (c, acc_c) in acc_tc.iter().enumerate() {
            for (t, a) in acc_c.iter().enumerate() {
                let (lo, hi) = fold(a);
                let col = (nb0 + tc) * 16 + c;
                raw.put((t0 + t) * m.n + col, lo);
                raw.put((t0 + t) * m.n + col + 8, hi);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::*;
    use std::arch::x86_64::*;

    pub(super) struct Consts {
        mul: __m256i,
        one8: __m256i,
        one16: __m256i,
        expm: __m256i,
        magic: __m256i,
        kinv: __m256,
        caff: __m256,
    }

    /// # Safety
    /// The CPU must have AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn consts() -> Consts {
        Consts {
            mul: _mm256_set1_epi32(MUL1 as i32),
            one8: _mm256_set1_epi8(1),
            one16: _mm256_set1_epi16(1),
            expm: _mm256_set1_epi32(0x7F80_0000),
            magic: _mm256_set1_epi32(0x06C0_0000),
            kinv: _mm256_set1_ps(KINV),
            caff: _mm256_set1_ps(CAFF),
        }
    }

    /// The 8 weights of trellis lane `g` of one tile, in position order j = 0..7.
    ///
    /// # Safety
    /// The CPU must have AVX2; `tile` points at `4 * n32` readable bytes.
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn decode8(tile: *const u8, n32: usize, g: &Group, k: &Consts) -> __m256 {
        // SAFETY (all loads): word indices are < n32 (wrap taken modulo, else w0 + 3 < n32)
        let win128 = if g.wrap {
            let rd = |i: usize| unsafe { (tile.add(4 * ((g.w0 + i) % n32)) as *const i32).read_unaligned() };
            _mm_setr_epi32(rd(0), rd(1), rd(2), rd(3))
        } else {
            unsafe { _mm_loadu_si128(tile.add(4 * g.w0) as *const __m128i) }
        };
        let win = _mm256_broadcastsi128_si256(win128);
        let (ia, ib, sh, shr) = unsafe {
            (
                _mm256_loadu_si256(g.ia.as_ptr() as *const __m256i),
                _mm256_loadu_si256(g.ib.as_ptr() as *const __m256i),
                _mm256_loadu_si256(g.sh.as_ptr() as *const __m256i),
                _mm256_loadu_si256(g.shr.as_ptr() as *const __m256i),
            )
        };
        let a = _mm256_permutevar8x32_epi32(win, ia);
        let b = _mm256_permutevar8x32_epi32(win, ib);
        // (a << o | b >> (32 - o)) >> 16: the 16 bits from o, MSB first (srlv by 32 gives 0)
        let st = _mm256_srli_epi32::<16>(_mm256_or_si256(_mm256_sllv_epi32(a, sh), _mm256_srlv_epi32(b, shr)));
        let x = _mm256_mullo_epi32(st, k.mul);
        // byte sum: u8 x 1 into i16 pairs (<= 510), then i16 x 1 into i32
        let s = _mm256_madd_epi16(_mm256_maddubs_epi16(x, k.one8), k.one16);
        // exact in f32, then one RNE rounding to fp16 precision: c = 1.5 * 2^(e + 13) from v's
        // exponent e, (v + c) - c rounds v to 2^(e - 10), the fp16 ulp (ft_core.h:224-232);
        // every v lies in the fp16 normal range (|v| >= 0.0018)
        let v = _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(s), k.kinv), k.caff);
        let c = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_and_si256(_mm256_castps_si256(v), k.expm), k.magic));
        _mm256_sub_ps(_mm256_add_ps(v, c), c)
    }

    /// The AVX2 twin of `unit_scalar`, operation for operation.
    ///
    /// # Safety
    /// The CPU must have AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn unit<const NT: usize>(m: &Mul1Matrix, tab: &Tables, xp: &[f32], t0: usize, nb0: usize, ntc: usize, raw: Out) {
        let (tk, tn, tb) = (m.k / 16, m.n / 16, m.bitrate.tile_bytes());
        assert!(ntc <= UNIT_TILES && nb0 + ntc <= tn && xp.len() >= (t0 + NT) * tk * 32);
        let k = unsafe { consts() };
        let mut acc = [[[_mm256_setzero_ps(); NT]; 8]; UNIT_TILES];
        let base = m.trellis.as_ptr();
        for kb in 0..tk {
            if kb + PF_ROWS < tk {
                let pf = ((kb + PF_ROWS) * tn + nb0) * tb;
                let mut o = 0;
                while o < ntc * tb {
                    // SAFETY: inside the trellis (row kb + PF_ROWS exists, columns nb0 .. nb0 + ntc)
                    unsafe { _mm_prefetch::<_MM_HINT_T0>(base.add(pf + o) as *const i8) };
                    o += 64;
                }
            }
            for (tc, acc_tc) in acc.iter_mut().enumerate().take(ntc) {
                let off = (kb * tn + nb0 + tc) * tb;
                assert!(off + tb <= m.trellis.len());
                // SAFETY: off + tb <= len (asserted)
                let tile = unsafe { base.add(off) };
                for (c, acc_c) in acc_tc.iter_mut().enumerate() {
                    let mut a = *acc_c;
                    for q in 0..4 {
                        // SAFETY: AVX2 (caller), the tile has 4 * n32 bytes
                        let w = unsafe { decode8(tile, tab.n32, &tab.groups[4 * c + q], &k) };
                        for (t, at) in a.iter_mut().enumerate() {
                            // SAFETY: ((t0 + t) * tk + kb) * 32 + q * 8 + 8 <= xp.len() (asserted)
                            let xv = unsafe { _mm256_loadu_ps(xp.as_ptr().add(((t0 + t) * tk + kb) * 32 + q * 8)) };
                            *at = _mm256_add_ps(*at, _mm256_mul_ps(w, xv));
                        }
                    }
                    *acc_c = a;
                }
            }
        }
        for (tc, acc_tc) in acc.iter().enumerate().take(ntc) {
            for (c, acc_c) in acc_tc.iter().enumerate() {
                for (t, a) in acc_c.iter().enumerate() {
                    let mut lanes = [0f32; 8];
                    // SAFETY: 8 f32 into an 8-element array
                    unsafe { _mm256_storeu_ps(lanes.as_mut_ptr(), *a) };
                    let (lo, hi) = fold(&lanes);
                    let col = (nb0 + tc) * 16 + c;
                    raw.put((t0 + t) * m.n + col, lo);
                    raw.put((t0 + t) * m.n + col + 8, hi);
                }
            }
        }
    }
}

/// One work unit (tile columns `UNIT_TILES u ..`) for every token of `xp`.
fn unit_into(avx2: bool, m: &Mul1Matrix, tab: &Tables, xp: &[f32], tokens: usize, u: usize, raw: Out) {
    let tn = m.n / 16;
    let nb0 = u * UNIT_TILES;
    let ntc = UNIT_TILES.min(tn - nb0);
    let mut t0 = 0;
    while t0 < tokens {
        let nt = (tokens - t0).min(TOK_TILE);
        #[cfg(target_arch = "x86_64")]
        if avx2 {
            // SAFETY: `avx2` is true only after `use_avx2` saw the feature on this CPU
            unsafe {
                match nt {
                    1 => avx2::unit::<1>(m, tab, xp, t0, nb0, ntc, raw),
                    2 => avx2::unit::<2>(m, tab, xp, t0, nb0, ntc, raw),
                    3 => avx2::unit::<3>(m, tab, xp, t0, nb0, ntc, raw),
                    _ => avx2::unit::<4>(m, tab, xp, t0, nb0, ntc, raw),
                }
            }
            t0 += nt;
            continue;
        }
        let _ = avx2;
        match nt {
            1 => unit_scalar::<1>(m, tab, xp, t0, nb0, ntc, raw),
            2 => unit_scalar::<2>(m, tab, xp, t0, nb0, ntc, raw),
            3 => unit_scalar::<3>(m, tab, xp, t0, nb0, ntc, raw),
            _ => unit_scalar::<4>(m, tab, xp, t0, nb0, ntc, raw),
        }
        t0 += nt;
    }
}

/// The activation `silu(g) * u = g / (1 + exp(-g)) * u` (`cpu_nvfp4`, `silu_mul640`), in f32.
#[inline]
pub fn silu_mul(g: f32, u: f32) -> f32 {
    (g / (1.0 + (-g).exp())) * u
}

fn units_of(m: &Mul1Matrix) -> usize {
    (m.n / 16).div_ceil(UNIT_TILES)
}

/// run `work(w)` on `k` workers, the caller's thread among them
fn scoped(k: usize, work: &(dyn Fn(usize) + Sync)) {
    if k == 1 {
        work(0);
    } else {
        std::thread::scope(|s| {
            for w in 1..k {
                s.spawn(move || work(w));
            }
            work(0);
        });
    }
}

/// `y[t] = x[t] W` for `x` `[T][k]`, `y` `[T][n]`, output columns split over `threads` workers.
pub fn gemv(m: &Mul1Matrix, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    assert!(x.len() % m.k == 0, "cpu_mul1::gemv: x is not [T][{}]", m.k);
    let tokens = x.len() / m.k;
    assert_eq!(y.len(), tokens * m.n, "cpu_mul1::gemv: y is not [T][{}]", m.n);
    let avx2 = use_avx2(path);
    let tab = Tables::new(m.bitrate);
    let xp = permute(&had_in(x, m), m.k);
    let mut raw = vec![0f32; y.len()];
    let out = Out(raw.as_mut_ptr(), raw.len());
    let units = units_of(m);
    let k = threads.clamp(1, units);
    scoped(k, &|w| {
        let (u0, u1) = split(units, k, w);
        for u in u0..u1 {
            unit_into(avx2, m, &tab, &xp, tokens, u, out);
        }
    });
    had_out(&raw, m, y);
}

/// The expert FFN `y = down(silu(gate(x)) * up(x))` for `x`, `y` `[T][hidden]`, with
/// `silu(g) * u = g / (1 + exp(-g)) * u` (`cpu_nvfp4`, `silu_mul640`). One scope: phase 1 splits
/// the gate and up units over the workers, a barrier, worker 0 finishes gate and up, applies the
/// activation and prepares down's input, a barrier, phase 2 splits the down units; the caller
/// finishes down. Bit for bit `gemv(down, silu(gemv(gate, x)) * gemv(up, x))`.
pub fn expert_ffn(e: &Mul1Expert, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    let (h, i) = (e.hidden, e.inter);
    assert!(x.len() % h == 0, "cpu_mul1::expert_ffn: x is not [T][{h}]");
    let tokens = x.len() / h;
    assert_eq!(y.len(), tokens * h, "cpu_mul1::expert_ffn: y is not [T][{h}]");
    let avx2 = use_avx2(path);
    let tab = Tables::new(e.gate.bitrate);
    let tab_d = Tables::new(e.down.bitrate);
    let (xp_g, xp_u) = (permute(&had_in(x, &e.gate), h), permute(&had_in(x, &e.up), h));
    let mut raw_g = vec![0f32; tokens * i];
    let mut raw_u = vec![0f32; tokens * i];
    let mut xp_d = vec![0f32; tokens * (i / 16) * 32];
    let mut raw_d = vec![0f32; tokens * h];
    let (og, ou, od) = (Out(raw_g.as_mut_ptr(), raw_g.len()), Out(raw_u.as_mut_ptr(), raw_u.len()), Out(raw_d.as_mut_ptr(), raw_d.len()));
    let oxd = Out(xp_d.as_mut_ptr(), xp_d.len());
    let (ug, ud) = (units_of(&e.gate), units_of(&e.down));
    let k = threads.clamp(1, (2 * ug).min(ud));
    let barrier = Barrier::new(k);
    scoped(k, &|w| {
        let (u0, u1) = split(2 * ug, k, w);
        for u in u0..u1 {
            if u < ug {
                unit_into(avx2, &e.gate, &tab, &xp_g, tokens, u, og);
            } else {
                unit_into(avx2, &e.up, &tab, &xp_u, tokens, u - ug, ou);
            }
        }
        barrier.wait();
        if w == 0 {
            // SAFETY: every write to raw_g / raw_u happened before the barrier; read only here
            let (rg, ru) = unsafe { (std::slice::from_raw_parts(og.0 as *const f32, og.1), std::slice::from_raw_parts(ou.0 as *const f32, ou.1)) };
            let (mut g, mut u) = (vec![0f32; og.1], vec![0f32; ou.1]);
            had_out(rg, &e.gate, &mut g);
            had_out(ru, &e.up, &mut u);
            let act: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| silu_mul(g, u)).collect();
            let xp = permute(&had_in(&act, &e.down), i);
            for (n, v) in xp.into_iter().enumerate() {
                oxd.put(n, v);
            }
        }
        barrier.wait();
        // SAFETY: worker 0 wrote xp_d before the second barrier; read only from here on
        let xd = unsafe { std::slice::from_raw_parts(oxd.0 as *const f32, oxd.1) };
        let (d0, d1) = split(ud, k, w);
        for u in d0..d1 {
            unit_into(avx2, &e.down, &tab_d, xd, tokens, u, od);
        }
    });
    drop((raw_g, raw_u, xp_d));
    had_out(&raw_d, &e.down, y);
}

/// Test kit shared with `kernels::tests_mul1_gpu`: the #181 decoder (ported), the f64 reference,
/// the fixtures and the synthetic experts.
#[cfg(test)]
pub(crate) mod testkit {
    use super::*;
    use sha2::{Digest, Sha256};

    /// splitmix64
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = self.0;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            z ^ (z >> 31)
        }
        /// uniform in [-a, a)
        pub fn f(&mut self, a: f32) -> f32 {
            ((self.next() >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a
        }
    }

    pub fn xs(n: usize, rng: &mut Rng) -> Vec<f32> {
        (0..n).map(|_| rng.f(2.0)).collect()
    }

    // ---- the #181 decoder, ported verbatim from converter/src/mul1.rs (5bc1e72); checked
    // against the exllamav3 reconstruct digests #181 checks it against
    // (`cpu_mul1_codec_port_is_the_181_decoder`) ----

    pub fn f16_to_f64(h: u16) -> f64 {
        let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
        let e = ((h >> 10) & 0x1f) as i32;
        let m = (h & 0x3ff) as f64;
        match e {
            0 => sign * m * 2f64.powi(-24),
            31 => {
                if m == 0.0 {
                    sign * f64::INFINITY
                } else {
                    f64::NAN
                }
            }
            _ => sign * (1024.0 + m) * 2f64.powi(e - 25),
        }
    }

    pub fn f64_to_f16(x: f64) -> u16 {
        let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
        let a = x.abs();
        if a == 0.0 {
            return sign;
        }
        if a < 2f64.powi(-14) {
            return sign | (a * 2f64.powi(24)).round_ties_even() as u16;
        }
        let mut e = ((a.to_bits() >> 52) & 0x7ff) as i32 - 1023;
        let mut q = (a * 2f64.powi(10 - e)).round_ties_even() as u32;
        if q == 2048 {
            q = 1024;
            e += 1;
        }
        if e + 15 >= 31 {
            return sign | 0x7c00;
        }
        sign | (((e + 15) as u16) << 10) | (q - 1024) as u16
    }

    pub fn mul1_decode(state: u16) -> u16 {
        let x = (state as u32).wrapping_mul(0x83DC_D12D);
        let sum = 0x6400u32 + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
        let h = f16_to_f64(sum as u16);
        f64_to_f16(h * f16_to_f64(0x1eee) + f16_to_f64(0xc931))
    }

    pub fn tile_index(p: usize) -> usize {
        let (lane, j) = (p / 8, p % 8);
        let row = 2 * (lane % 4) + [0, 1, 8, 9][j % 4];
        let col = lane / 4 + 8 * (j / 4);
        row * 16 + col
    }

    pub fn decode_tile(words: &[u16], b: Bitrate) -> [u16; 256] {
        assert_eq!(words.len(), b.words_per_tile());
        let w32: Vec<u32> = words.chunks_exact(2).map(|p| p[0] as u32 | (p[1] as u32) << 16).collect();
        let n32 = w32.len();
        let s = b.words_per_tile() * 16;
        let mut out = [0u16; 256];
        for p in 0..256 {
            let lo = (b.end_bit(p) + s - 16) % s;
            let (i, o) = (lo / 32, lo % 32);
            let pair = (w32[i] as u64) << 32 | w32[(i + 1) % n32] as u64;
            let state = ((pair >> (48 - o)) & 0xffff) as u16;
            out[tile_index(p)] = mul1_decode(state);
        }
        out
    }

    pub fn reconstruct(trellis: &[u16], k: usize, n: usize, b: Bitrate) -> Vec<u16> {
        let wpt = b.words_per_tile();
        assert_eq!(trellis.len(), k / 16 * (n / 16) * wpt);
        let mut w = vec![0u16; k * n];
        for kb in 0..k / 16 {
            for nb in 0..n / 16 {
                let t = kb * (n / 16) + nb;
                let tile = decode_tile(&trellis[t * wpt..(t + 1) * wpt], b);
                for r in 0..16 {
                    let row = (kb * 16 + r) * n + nb * 16;
                    w[row..row + 16].copy_from_slice(&tile[r * 16..r * 16 + 16]);
                }
            }
        }
        w
    }

    /// `exl3_mul1_fixtures.py` / #181 `synth_words`
    pub fn synth_words(count: usize, seed: u32) -> Vec<u16> {
        (0..count as u32)
            .map(|i| {
                let mut x = i.wrapping_mul(0x9E37_79B1).wrapping_add(seed.wrapping_mul(0x85EB_CA6B));
                x ^= x >> 15;
                x = x.wrapping_mul(0x2C1B_3C6D);
                x ^= x >> 12;
                (x & 0xffff) as u16
            })
            .collect()
    }

    pub fn u16s(b: &[u8]) -> Vec<u16> {
        b.chunks_exact(2).map(|p| u16::from_le_bytes([p[0], p[1]])).collect()
    }

    fn le(v: &[u16]) -> Vec<u8> {
        v.iter().flat_map(|x| x.to_le_bytes()).collect()
    }

    pub fn sha_u16(v: &[u16]) -> String {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }

    pub const DECODE_TSV: &str = include_str!("../../converter/tests/fixtures/exl3-mul1-decode.tsv");

    /// the exllamav3 quantizer outputs #181 committed (unpadded records)
    pub fn fixture_bin(name: &str) -> &'static [u8] {
        match name {
            "exl3-mul1-q-k3.bin" => include_bytes!("../../converter/tests/fixtures/exl3-mul1-q-k3.bin"),
            "exl3-mul1-q-k2.bin" => include_bytes!("../../converter/tests/fixtures/exl3-mul1-q-k2.bin"),
            "exl3-mul1-q-k4.bin" => include_bytes!("../../converter/tests/fixtures/exl3-mul1-q-k4.bin"),
            "exl3-mul1-q-k3.5.bin" => include_bytes!("../../converter/tests/fixtures/exl3-mul1-q-k3.5.bin"),
            _ => panic!("no fixture {name}"),
        }
    }

    /// one row of `exl3-mul1-decode.tsv`
    pub struct Case {
        pub name: String,
        pub source: String,
        pub bitrate: Bitrate,
        pub hidden: usize,
        pub inter: usize,
        pub want: [String; 3],
    }

    pub fn cases() -> Vec<Case> {
        DECODE_TSV
            .lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                assert_eq!(f.len(), 8);
                Case {
                    name: f[0].into(),
                    source: f[1].into(),
                    bitrate: Bitrate::from_k(f[2].parse().unwrap()).unwrap(),
                    hidden: f[3].parse().unwrap(),
                    inter: f[4].parse().unwrap(),
                    want: [f[5].into(), f[6].into(), f[7].into()],
                }
            })
            .collect()
    }

    /// synthetic suh / svh: fp16 values of random sign and magnitude 0.5 .. 1.5
    fn synth_scales(n: usize, rng: &mut Rng) -> Vec<u16> {
        (0..n).map(|_| f64_to_f16((1.0 + rng.f(0.5) as f64) * if rng.next() & 1 == 0 { 1.0 } else { -1.0 })).collect()
    }

    /// The record of one case, padded to `record_bytes`: the quantizer's bytes, or for a
    /// "synth:<seed>" row the #181 synthetic trellis streams (`seed + 3 i`) with synthetic
    /// scales (#181's synthetic scale words are arbitrary fp16 bit patterns, NaN among them).
    pub fn record(c: &Case) -> Vec<u8> {
        let size = record_bytes(c.hidden, c.inter, c.bitrate);
        let mut rec = if let Some(seed) = c.source.strip_prefix("synth:") {
            let seed: u32 = seed.parse().unwrap();
            let words = Mul1Matrix::trellis_bytes(c.hidden, c.inter, c.bitrate) / 2;
            let mut rng = Rng(seed as u64 ^ 0x180);
            let mut rec = Vec::with_capacity(size);
            for i in 0..3u32 {
                rec.extend(le(&synth_words(words, seed + 3 * i)));
            }
            for (k, n) in [(c.hidden, c.inter), (c.hidden, c.inter), (c.inter, c.hidden)] {
                rec.extend(le(&synth_scales(k, &mut rng)));
                rec.extend(le(&synth_scales(n, &mut rng)));
            }
            rec
        } else {
            fixture_bin(&c.source).to_vec()
        };
        rec.resize(size, 0);
        rec
    }

    /// The three GLM-shaped experts of the #181 synthetic set (K = 3, 2, 3.5).
    pub fn glm_cases() -> Vec<Case> {
        cases().into_iter().filter(|c| c.source.starts_with("synth:")).collect()
    }

    pub fn quant_cases() -> Vec<Case> {
        cases().into_iter().filter(|c| !c.source.starts_with("synth:")).collect()
    }

    pub fn gamma32(n: usize) -> f64 {
        let nu = n as f64 * 2f64.powi(-24);
        nu / (1.0 - nu)
    }

    pub fn gamma64(n: usize) -> f64 {
        let nu = n as f64 * 2f64.powi(-53);
        nu / (1.0 - nu)
    }

    /// in-place f64 Walsh-Hadamard over 128 values (reference; checked against the explicit matrix)
    pub fn fwht64(v: &mut [f64]) {
        let mut h = 1;
        while h < HAD {
            for i in (0..HAD).step_by(2 * h) {
                for j in i..i + h {
                    let (a, b) = (v[j], v[j + h]);
                    v[j] = a + b;
                    v[j + h] = a - b;
                }
            }
            h *= 2;
        }
    }

    /// One linear decoded through the #181 decoder, everything in f64.
    pub struct RefLinear {
        pub k: usize,
        pub n: usize,
        /// rotated weight `[k][n]`
        pub w_hat: Vec<f64>,
        pub suh: Vec<f64>,
        pub svh: Vec<f64>,
        /// original-basis weight `diag(suh) H W_hat H diag(svh) / 128`, `[k][n]`
        pub w: Vec<f64>,
    }

    impl RefLinear {
        pub fn new(m: &Mul1Matrix) -> RefLinear {
            let (k, n) = (m.k, m.n);
            let w_hat: Vec<f64> = reconstruct(&u16s(m.trellis), k, n, m.bitrate).into_iter().map(f16_to_f64).collect();
            let suh: Vec<f64> = u16s(m.suh).into_iter().map(f16_to_f64).collect();
            let svh: Vec<f64> = u16s(m.svh).into_iter().map(f16_to_f64).collect();
            let mut w = w_hat.clone();
            // left H: along k inside each 128-row block, for every column
            let mut col = vec![0f64; HAD];
            for c in 0..n {
                for rb in 0..k / HAD {
                    for r in 0..HAD {
                        col[r] = w[(rb * HAD + r) * n + c];
                    }
                    fwht64(&mut col);
                    for r in 0..HAD {
                        w[(rb * HAD + r) * n + c] = col[r];
                    }
                }
            }
            for r in 0..k {
                for blk in w[r * n..(r + 1) * n].chunks_exact_mut(HAD) {
                    fwht64(blk);
                }
                for c in 0..n {
                    w[r * n + c] *= suh[r] * svh[c] / 128.0;
                }
            }
            RefLinear { k, n, w_hat, suh, svh, w }
        }

        /// per `[t][j]`: (y_ref = sum_i x_i W_ij, A = sum_i |x_i W_ij|, M = the |.| sum over every
        /// product path of the factorized form)
        pub fn eval(&self, x: &[f64]) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
            let (k, n) = (self.k, self.n);
            let tokens = x.len() / k;
            let (mut y, mut a, mut mm) = (vec![0f64; tokens * n], vec![0f64; tokens * n], vec![0f64; tokens * n]);
            for t in 0..tokens {
                let xt = &x[t * k..(t + 1) * k];
                let (yt, at) = (&mut y[t * n..(t + 1) * n], &mut a[t * n..(t + 1) * n]);
                for i in 0..k {
                    let row = &self.w[i * n..(i + 1) * n];
                    for j in 0..n {
                        let p = xt[i] * row[j];
                        yt[j] += p;
                        at[j] += p.abs();
                    }
                }
                // M: s_i = sum over i's block |suh x|, col_c = sum_i |w_hat_ic| s_i,
                // M_j = |svh_j| / 128 * sum over j's block col_c
                let mut s = vec![0f64; k];
                for b in 0..k / HAD {
                    let v: f64 = (b * HAD..(b + 1) * HAD).map(|i| (self.suh[i] * xt[i]).abs()).sum();
                    s[b * HAD..(b + 1) * HAD].iter_mut().for_each(|z| *z = v);
                }
                let mut colsum = vec![0f64; n];
                for i in 0..k {
                    let row = &self.w_hat[i * n..(i + 1) * n];
                    for c in 0..n {
                        colsum[c] += row[c].abs() * s[i];
                    }
                }
                for b in 0..n / HAD {
                    let v: f64 = colsum[b * HAD..(b + 1) * HAD].iter().sum();
                    for j in b * HAD..(b + 1) * HAD {
                        mm[t * n + j] = self.svh[j].abs() / 128.0 * v;
                    }
                }
            }
            (y, a, mm)
        }
    }

    /// Worst ratios of `|y - y_ref|` to (the ticket form `gamma32(n) A + gamma64(k + 20) M`, the
    /// rigorous form `(gamma32(n) + gamma64(k + 20)) M`); both must be <= 1.
    pub fn ratios(y: &[f32], yr: &[f64], a: &[f64], mm: &[f64], k: usize, n: usize) -> (f64, f64) {
        let (g, g64) = (gamma32(n), gamma64(k + 20));
        let mut w = (0f64, 0f64);
        for i in 0..y.len() {
            let d = (y[i] as f64 - yr[i]).abs();
            w.0 = w.0.max(d / (g * a[i] + g64 * mm[i]).max(f64::MIN_POSITIVE));
            w.1 = w.1.max(d / ((g + g64) * mm[i]).max(f64::MIN_POSITIVE));
        }
        w
    }

    pub fn f64s(v: &[f32]) -> Vec<f64> {
        v.iter().map(|&x| x as f64).collect()
    }

    pub fn silu_mul64(g: f64, u: f64) -> f64 {
        g / (1.0 + (-g).exp()) * u
    }

    /// sup |silu'| = 1.0998 (at g = 2.3994); the FFN bound uses 1.1
    pub const SILU_LIP: f64 = 1.1;

    /// The FFN chain bound of docs/mul1-gemv.md 3.3 for one path, per `[t][r]`:
    /// `gamma32(n_d) sum_j |Wd_jr h_hat_j| + gamma64 M_d + sum_j |Wd_jr| dh_j` with
    /// `dh_j = 1.1 Eg_j |u_hat_j| + |silu(g_j)| Eu_j + gamma32(9) |h_hat_j|` (gamma32(8) relative
    /// to the exact activation of the path's own inputs, `cpu_mul1_activation_within_8_roundings`),
    /// `Eg = gamma32(n_gu) A_g + gamma64 M_g` (the GEMV bound of gate), same for up.
    /// `g_hat`, `u_hat`, `h_hat` are the path's own values. Returns (y_ref all-f64, bound).
    #[allow(clippy::too_many_arguments)]
    pub fn ffn_bound(
        rg: &RefLinear,
        ru: &RefLinear,
        rd: &RefLinear,
        x: &[f32],
        u_hat: &[f32],
        h_hat: &[f32],
        n_gu: usize,
        n_d: usize,
    ) -> (Vec<f64>, Vec<f64>) {
        let x64 = f64s(x);
        let (g, ag, mg) = rg.eval(&x64);
        let (u, au, mu) = ru.eval(&x64);
        let h: Vec<f64> = g.iter().zip(&u).map(|(&g, &u)| silu_mul64(g, u)).collect();
        let (yr, _, _) = rd.eval(&h);
        let (_, ad_hat, md_hat) = rd.eval(&f64s(h_hat));
        let (gk, g64) = (gamma32(n_gu), gamma64(rg.k + 20));
        let dh: Vec<f64> = (0..h.len())
            .map(|j| {
                let eg = gk * ag[j] + g64 * mg[j];
                let eu = gk * au[j] + g64 * mu[j];
                let sg = g[j] / (1.0 + (-g[j]).exp());
                SILU_LIP * eg * (u_hat[j] as f64).abs() + sg.abs() * eu + gamma32(9) * (h_hat[j] as f64).abs()
            })
            .collect();
        // sum_j |Wd_jr| dh_j
        let (ki, n) = (rd.k, rd.n);
        let tokens = h.len() / ki;
        let mut prop = vec![0f64; tokens * n];
        for t in 0..tokens {
            for j in 0..ki {
                let row = &rd.w[j * n..(j + 1) * n];
                for r in 0..n {
                    prop[t * n + r] += row[r].abs() * dh[t * ki + j];
                }
            }
        }
        let (gd, g64d) = (gamma32(n_d), gamma64(rd.k + 20));
        let bound: Vec<f64> = (0..yr.len()).map(|i| gd * ad_hat[i] + g64d * md_hat[i] + prop[i]).collect();
        (yr, bound)
    }

    pub fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|f| f.to_bits()).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;

    fn run(m: &Mul1Matrix, x: &[f32], threads: usize, path: Path) -> Vec<f32> {
        let mut y = vec![0f32; x.len() / m.k * m.n];
        gemv(m, x, &mut y, threads, path);
        y
    }

    fn ffn(e: &Mul1Expert, x: &[f32], threads: usize, path: Path) -> Vec<f32> {
        let mut y = vec![0f32; x.len()];
        expert_ffn(e, x, &mut y, threads, path);
        y
    }

    /// The test reference is the #181 decoder: the port gives the exllamav3 `reconstruct` digests
    /// of every row of `exl3-mul1-decode.tsv` (the digests #181's own acceptance test holds).
    #[test]
    fn cpu_mul1_codec_port_is_the_181_decoder() {
        let cs = cases();
        assert_eq!(cs.len(), 7);
        for c in &cs {
            let rec = record(c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            for (m, want) in [e.gate, e.up, e.down].iter().zip(&c.want) {
                assert_eq!(&sha_u16(&reconstruct(&u16s(m.trellis), m.k, m.n, c.bitrate)), want, "{} [{}, {}]", c.name, m.k, m.n);
            }
        }
    }

    /// Both decoders give the codec's fp16 value of every state; the AVX2 lane decode gives the
    /// codec tile at every position, for every bitrate.
    #[test]
    fn cpu_mul1_weights_are_the_codec_values() {
        for s in 0..=u16::MAX {
            assert_eq!(weight(s).to_bits(), (f16_to_f64(mul1_decode(s)) as f32).to_bits(), "state {s:#06x}");
        }
        let mut rng = Rng(0x7E11);
        for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0] {
            let b = Bitrate::from_k(k).unwrap();
            let tab = Tables::new(b);
            for _ in 0..8 {
                let words: Vec<u16> = (0..b.words_per_tile()).map(|_| rng.next() as u16).collect();
                let tile: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let want = decode_tile(&words, b);
                for p in 0..256 {
                    let w = weight(state_at(&tile, tab.n32, tab.lo[p]) as u16);
                    assert_eq!(w.to_bits(), (f16_to_f64(want[tile_index(p)]) as f32).to_bits(), "scalar K = {k} p {p}");
                }
                #[cfg(target_arch = "x86_64")]
                if avx2_available() {
                    for g in 0..32 {
                        let mut lanes = [0f32; 8];
                        // SAFETY: AVX2 checked; the tile holds 4 * n32 bytes
                        unsafe {
                            let k = avx2::consts();
                            let v = avx2::decode8(tile.as_ptr(), tab.n32, &tab.groups[g], &k);
                            std::arch::x86_64::_mm256_storeu_ps(lanes.as_mut_ptr(), v);
                        }
                        for (j, &l) in lanes.iter().enumerate() {
                            let p = 8 * g + j;
                            assert_eq!(l.to_bits(), (f16_to_f64(want[tile_index(p)]) as f32).to_bits(), "avx2 K = {k} p {p}");
                        }
                    }
                }
            }
        }
    }

    #[test]
    fn cpu_mul1_shapes_are_checked() {
        let b3 = Bitrate::from_k(3.0).unwrap();
        assert_eq!(record_bytes(GLM_HIDDEN, GLM_INTER, b3), GLM_RECORD_BYTES_K3);
        let rec = vec![0u8; GLM_RECORD_BYTES_K3];
        let e = Mul1Expert::from_record(&rec, GLM_HIDDEN, GLM_INTER, b3).unwrap();
        assert_eq!((e.gate.trellis.len(), e.up.trellis.len(), e.down.trellis.len()), (3_145_728, 3_145_728, 3_145_728));
        assert_eq!((e.gate.k, e.gate.n, e.down.k, e.down.n), (4096, 2048, 2048, 4096));
        assert!(Mul1Expert::from_record(&rec[..GLM_RECORD_BYTES_K3 - 4096], GLM_HIDDEN, GLM_INTER, b3).is_err());
        assert!(Mul1Matrix::new(&rec[..96], &rec[..32], &rec[..32], 16, 16, b3).is_err());
        assert!(Bitrate::from_k(3.25).is_err());
        assert_eq!(rounding_steps(4096), 1042);
        assert_eq!(rounding_steps(2048), 530);
        // the f64 reference transform: the FWHT is the explicit Sylvester matrix (-1)^popcount(i & j)
        let mut rng = Rng(0x48);
        let v: Vec<f64> = (0..HAD).map(|_| rng.f(1.0) as f64).collect();
        let mut f = v.clone();
        fwht64(&mut f);
        for (i, &fi) in f.iter().enumerate() {
            let want: f64 = (0..HAD).map(|j| if (i & j).count_ones() % 2 == 0 { v[j] } else { -v[j] }).sum();
            assert!((fi - want).abs() < 1e-12, "row {i}");
        }
        let mut f32v: Vec<f32> = v.iter().map(|&x| x as f32).collect();
        let mut f64v: Vec<f64> = f32v.iter().map(|&x| x as f64).collect();
        fwht128(&mut f32v);
        fwht64(&mut f64v);
        for i in 0..HAD {
            assert!((f32v[i] as f64 - f64v[i]).abs() <= gamma32(7) * 128.0 * 2.0, "fwht128 {i}");
        }
    }

    /// Acceptance 1 (CPU): every GEMV output within the amended bound of the f64 evaluation of
    /// the #181-decoded tensor: the 4 exllamav3 quantizer experts and the 3 GLM-shaped synthetic
    /// experts, gate / up / down, T 1, 2, 4.
    #[test]
    fn cpu_mul1_gemv_holds_the_bound() {
        let mut rng = Rng(180);
        for c in cases() {
            let rec = record(&c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            for (name, m) in [("gate", e.gate), ("up", e.up), ("down", e.down)] {
                let r = RefLinear::new(&m);
                for t in [1usize, 2, 4] {
                    let x = xs(t * m.k, &mut rng);
                    let (yr, a, mm) = r.eval(&f64s(&x));
                    for path in [Path::Scalar, Path::Auto] {
                        let y = run(&m, &x, 3, path);
                        let (w, wr) = ratios(&y, &yr, &a, &mm, m.k, rounding_steps(m.k));
                        assert!(w <= 1.0 && wr <= 1.0, "{} {name} T {t} {path:?}: {w:.3} x bound ({wr:.2e} x rigorous)", c.name);
                        if path == Path::Auto && t == 1 {
                            eprintln!("{} {name} [{}, {}]: worst |dy| = {w:.4} x bound, {wr:.2e} x rigorous", c.name, m.k, m.n);
                        }
                    }
                }
            }
        }
    }

    /// Acceptance 2: AVX2 == scalar bit for bit for T 1, 2, 3, 5, 8 and threads 1, 3, 8, 16, on
    /// every bitrate, and the FFN of one full GLM-shaped expert.
    #[test]
    fn cpu_mul1_avx2_equals_scalar_bits() {
        if !avx2_available() {
            eprintln!("cpu_mul1: no AVX2 on this CPU, the AVX2 path is not exercised");
            return;
        }
        let mut rng = Rng(0xC0FFEE);
        for c in quant_cases() {
            let rec = record(&c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            for m in [e.gate, e.down] {
                for t in [1usize, 2, 3, 5, 8] {
                    let x = xs(t * m.k, &mut rng);
                    let s1 = bits(&run(&m, &x, 1, Path::Scalar));
                    for th in [1usize, 3, 8, 16] {
                        assert_eq!(bits(&run(&m, &x, th, Path::Scalar)), s1, "{} scalar T {t} threads {th}", c.name);
                        assert_eq!(bits(&run(&m, &x, th, Path::Avx2)), s1, "{} avx2 T {t} threads {th}", c.name);
                    }
                }
            }
        }
        let c = &glm_cases()[0];
        let rec = record(c);
        let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
        let x = xs(2 * GLM_HIDDEN, &mut rng);
        let s = bits(&ffn(&e, &x, 8, Path::Scalar));
        assert_eq!(bits(&ffn(&e, &x, 16, Path::Avx2)), s, "full GLM expert FFN");
        assert_eq!(bits(&ffn(&e, &x, 3, Path::Avx2)), s, "full GLM expert FFN, 3 threads");
    }

    /// The activation assumption of the FFN bound: the f32 `g / (1 + exp(-g)) * u` is within
    /// gamma32(8) relative of the exact value, over g in [-40, 40].
    #[test]
    fn cpu_mul1_activation_within_8_roundings() {
        let mut worst = 0f64;
        for i in 0..=800_000 {
            let g = -40.0 + i as f32 * 1e-4;
            let u = 1.0f32 + (i % 7) as f32 * 0.37;
            let got = super::silu_mul(g, u);
            let want = silu_mul64(g as f64, u as f64);
            if want != 0.0 {
                worst = worst.max((got as f64 - want).abs() / want.abs());
            }
        }
        eprintln!("activation: worst relative error {worst:.3e} (gamma32(8) = {:.3e})", gamma32(8));
        assert!(worst <= gamma32(8), "{worst:.3e}");
    }

    /// Acceptance 4 (CPU): the FFN equals gemv + activation + gemv bit for bit, the down stage
    /// holds the GEMV bound on its own input, the whole FFN holds the chain bound against an
    /// all-f64 reference, and every thread count / path gives the same bits.
    #[test]
    fn cpu_mul1_expert_ffn_matches_reference() {
        let mut rng = Rng(0xF00D);
        let mut all = quant_cases();
        all.push(glm_cases().remove(0));
        for c in all {
            let rec = record(&c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            let (rg, ru, rd) = (RefLinear::new(&e.gate), RefLinear::new(&e.up), RefLinear::new(&e.down));
            let (h, i) = (c.hidden, c.inter);
            for t in [1usize, 2, 4] {
                let x = xs(t * h, &mut rng);
                let y = ffn(&e, &x, 1, Path::Scalar);
                let (g, u) = (run(&e.gate, &x, 1, Path::Scalar), run(&e.up, &x, 1, Path::Scalar));
                let hk: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| super::silu_mul(g, u)).collect();
                assert_eq!(bits(&run(&e.down, &hk, 1, Path::Scalar)), bits(&y), "{} T {t}: fused != staged", c.name);
                let (yr, a, mm) = rd.eval(&f64s(&hk));
                let (w, _) = ratios(&y, &yr, &a, &mm, i, rounding_steps(i));
                assert!(w <= 1.0, "{} T {t}: down stage {w:.3} x bound", c.name);
                let (yr, bound) = ffn_bound(&rg, &ru, &rd, &x, &u, &hk, rounding_steps(h), rounding_steps(i));
                let worst = y.iter().zip(&yr).zip(&bound).map(|((&a, &b), &s)| (a as f64 - b).abs() / s).fold(0.0, f64::max);
                assert!(worst <= 1.0, "{} T {t}: FFN {worst:.3} x chain bound", c.name);
                if t == 1 {
                    eprintln!("{} FFN H {h} I {i}: worst |dy| = {worst:.4} x chain bound", c.name);
                }
                for th in [1usize, 3, 8] {
                    for path in [Path::Scalar, Path::Auto] {
                        assert_eq!(bits(&ffn(&e, &x, th, path)), bits(&y), "{} T {t} threads {th} {path:?}", c.name);
                    }
                }
            }
        }
    }

    /// Micro-benchmark, not a gate: one FFN over GLM-shaped K = 3 records rotated across 8 distinct
    /// records (76 MB, more than the L3), T 1 and 4, threads 1, 4, 8, 16, 24. Trellis + scale
    /// bytes / wall time.
    /// `cargo test --release -p crow-nest-engine cpu_mul1_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cpu_mul1_bench() {
        let c = &glm_cases()[0];
        assert_eq!(c.bitrate.k(), 3.0);
        let base = record(c);
        let recs: Vec<Vec<u8>> = (0..8u8)
            .map(|i| {
                let mut r = base.clone();
                let tb = 3 * Mul1Matrix::trellis_bytes(c.hidden, c.inter, c.bitrate);
                r[..tb].iter_mut().for_each(|b| *b = b.rotate_left(i as u32));
                r
            })
            .collect();
        let experts: Vec<Mul1Expert> = recs.iter().map(|r| Mul1Expert::from_record(r, c.hidden, c.inter, c.bitrate).unwrap()).collect();
        let path = if avx2_available() { Path::Avx2 } else { Path::Scalar };
        let bytes = 3 * Mul1Matrix::trellis_bytes(c.hidden, c.inter, c.bitrate) + 6 * (c.hidden + c.inter);
        eprintln!("cpu_mul1 bench: {path:?}, {} threads available, {bytes} B per expert", std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0));
        let mut rng = Rng(1);
        for &t in &[1usize, 4] {
            let x = xs(t * GLM_HIDDEN, &mut rng);
            let mut y = vec![0f32; t * GLM_HIDDEN];
            for &th in &[1usize, 4, 8, 16, 24] {
                for e in &experts {
                    expert_ffn(e, &x, &mut y, th, path);
                }
                let mut ms = Vec::new();
                for rep in 0..24 {
                    let t0 = std::time::Instant::now();
                    expert_ffn(&experts[rep % experts.len()], &x, &mut y, th, path);
                    ms.push(t0.elapsed().as_secs_f64() * 1e3);
                }
                ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = ms[ms.len() / 2];
                eprintln!(
                    "  T {t} threads {th:2}: median {med:.3} ms per expert ({:.2} GB/s of record bytes), min {:.3} max {:.3} ms, n {}",
                    bytes as f64 / (med * 1e-3) / 1e9,
                    ms[0],
                    ms[ms.len() - 1],
                    ms.len()
                );
            }
        }
    }
}
