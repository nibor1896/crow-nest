//! crow-nest #180: the CPU MUL1 trellis expert GEMV and FFN (plan step 10, CPU half; the GPU half
//! is `kernels::mul1` / `kernels_mul1.cu`). A routed GLM-5.3-Flash expert stored as a CNQ MUL1
//! record (#181, `converter/src/mul1.rs`) is computed where it lies, without a dequant pass.
//! Its callers are `glm5_moe` (#164) and, behind `CROW_GLM_CPU_LANE=1`, the #188 CPU lane
//! ([`experts_ffn`]: every CPU expert of a MoE layer in one pool run).
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
//!   state). The AVX2 path gets a lane's 8 states with one byte shuffle of a 16-byte window, one
//!   variable shift and a mask (#183, sybil `ft_core.h:129-160, 175-191`, exllamav3
//!   `moe_mul1.cpp:1004-1030`), the same states as the scalar windows; since #183 C1 the window
//!   needs no division ([`Lane`]) and at K = 3 the 32 lanes are compile-time constants.
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
//! The API is that of `cpu_nvfp4` (#173). Threading (#183 C1, see [`Impl`]): one run of the
//! persistent [`pool`] per call; tile columns go out in guided chunks from an atomic counter, so on
//! a hybrid CPU a P-core takes more than an E-core and the last chunk is one column; the worker
//! that completes a 128-column block finishes it; a phase waits for completed work, never for
//! workers. A column is computed by one worker with the same operations whichever it is and
//! whatever chunk holds it, so the bits do not depend on the thread count.

pub use crate::cpu_nvfp4::{avx2_available, Path};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::OnceLock;

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
/// output tile columns per work unit (128 outputs, one block); a thread owns whole units. 8
/// since the split planner (2026-10-09): a unit reads 768 contiguous bytes per tile row instead
/// of 384, the #188 lane's 8 experts at 8 threads 0.490 -> 0.426 ms per expert
/// (`cpu_mul1_lane_bench`); the bits do not depend on it
const UNIT_TILES: usize = 8;
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
    pub const fn words_per_tile(self) -> usize {
        16 * self.bits as usize + if self.half { 8 } else { 0 }
    }

    pub const fn tile_bytes(self) -> usize {
        2 * self.words_per_tile()
    }

    /// exclusive end bit of position `t` in the tile's stream
    pub const fn end_bit(self, t: usize) -> usize {
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

/// fp16 bits -> f32 (exact). A normal value is assembled from its bits (sign, exponent + 112,
/// mantissa << 13), the same f32 as `sign * (1024 + m) * 2^(e - 25)`; NaN gives `f32::NAN`
/// (#183 C1: the scales are converted per call, `powi` made the input transform ~80 us).
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    match e {
        0 => sign * m as f32 * (1.0 / 16_777_216.0),
        31 => {
            if m == 0 {
                sign * f32::INFINITY
            } else {
                f32::NAN
            }
        }
        _ => f32::from_bits((h as u32 & 0x8000) << 16 | (e + 112) << 23 | m << 13),
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

/// The AVX2 view of one trellis lane: a 16-byte window of 4 words from `w0` (circular) and, per
/// position j, a byte shuffle that puts the window bytes holding stream bytes `Q0, Q0 + 1, Q0 + 2`
/// (`Q0 = rel / 8`) into bytes 3, 2, 1 of 32-bit lane j (byte 0 zero), so the lane reads the
/// stream MSB first, and the right shift `16 - rel % 8` that leaves the 16-bit state in the low
/// half (sybil `ft_core.h:129-160, 188-207`, generalised from K = 3 to every bitrate).
/// `ia`, `ib`, `sh`, `shr` are the two-word form of the first #180 decoder (`avx2::decode8_v1`,
/// test reference only).
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
struct Group {
    ctrl: [u8; 32],
    shv: [i32; 8],
    w0: usize,
    wrap: bool,
    ia: [i32; 8],
    ib: [i32; 8],
    sh: [i32; 8],
    shr: [i32; 8],
}

/// The window of one trellis lane as the C1 kernel (#183 follow-up) reads it, with no division
/// in the inner loop: a plain 16-byte load at byte `off`, or for a lane whose bits run over the
/// circular end of the stream (`alignr` = 4, 8 or 12) `_mm_alignr_epi8` of the tile's first 16
/// bytes and its last 16 bytes at `off`, so window word i is stream word `(w0 + i) % n32` (sybil
/// `ft_core.h:180-185` does this for K = 3 lane 0). `ctrl` and `shv` as in [`Group`], relative to
/// the window's first word; a lane whose bits end in the tile's last words without wrapping
/// reads the last 16 bytes instead of running past the tile.
#[derive(Clone, Copy, Debug)]
#[repr(C, align(32))]
struct Lane {
    ctrl: [u8; 32],
    shv: [i32; 8],
    off: usize,
    alignr: u8,
}

/// The [`Lane`] table of one bitrate, a `const fn` so the K = 3 table is built (and its window
/// checks run) at compile time ([`K3_LANES`]).
const fn lane_table(b: Bitrate) -> [Lane; 32] {
    let n32 = b.words_per_tile() / 2;
    let s = 32 * n32;
    let mut lanes = [Lane { ctrl: [0x80; 32], shv: [0; 8], off: 0, alignr: 0 }; 32];
    let mut g = 0;
    while g < 32 {
        let first = (b.end_bit(8 * g) + s - 16) % s;
        // bits from the lane's first window start to the end of its last state (circular)
        let span = ((b.end_bit(8 * g + 7) + s - 16) % s + s - first) % s + 16;
        let w0 = first / 32;
        let wraps = w0 + 4 > n32 && first + span > s;
        let ws = if w0 + 4 <= n32 || wraps { w0 } else { n32 - 4 };
        lanes[g].off = 4 * if wraps { n32 - 4 } else { ws };
        lanes[g].alignr = if wraps { (4 * (w0 + 4 - n32)) as u8 } else { 0 };
        let mut j = 0;
        while j < 8 {
            let rel = ((b.end_bit(8 * g + j) + s - 16) % s + s - 32 * ws) % s;
            let q0 = rel / 8;
            // a state on a byte boundary needs only Q0, Q0 + 1 (the shift by 16 drops the third
            // byte): that byte may lie past the window and is zeroed (control 0x80)
            let need = if rel % 8 == 0 { q0 + 1 } else { q0 + 2 };
            assert!(need < 16, "cpu_mul1: a trellis lane leaves the 16-byte window");
            let mut d = 0;
            while d < 3 {
                // stream byte q of the window sits at window byte 4 (q / 4) + 3 - q % 4 (LE words)
                let q = q0 + d;
                lanes[g].ctrl[4 * (j % 4) + 16 * (j / 4) + 3 - d] = if q < 16 { (4 * (q / 4) + 3 - q % 4) as u8 } else { 0x80 };
                d += 1;
            }
            lanes[g].shv[j] = 16 - (rel % 8) as i32;
            j += 1;
        }
        g += 1;
    }
    lanes
}

/// the GLM experts' bitrate
const K3: Bitrate = Bitrate { bits: 3, half: false };

/// the K = 3 lane table, built at compile time
static K3_LANES: [Lane; 32] = lane_table(K3);

/// The tables of bitrate `b`, built once per process.
fn tables(b: Bitrate) -> &'static Tables {
    static T: [OnceLock<Tables>; 18] = [const { OnceLock::new() }; 18];
    T[2 * b.bits as usize + b.half as usize].get_or_init(|| Tables::new(b))
}

/// per-bitrate position tables
#[derive(Clone, Debug)]
struct Tables {
    n32: usize,
    /// window start bit of every trellis position
    lo: [usize; 256],
    groups: [Group; 32],
    lanes: [Lane; 32],
}

impl Tables {
    fn new(b: Bitrate) -> Tables {
        let n32 = b.words_per_tile() / 2;
        let s = 32 * n32;
        let mut lo = [0usize; 256];
        for (p, l) in lo.iter_mut().enumerate() {
            *l = (b.end_bit(p) + s - 16) % s;
        }
        let groups = {
            let mut groups = [Group { ctrl: [0x80; 32], shv: [0; 8], w0: 0, wrap: false, ia: [0; 8], ib: [0; 8], sh: [0; 8], shr: [0; 8] }; 32];
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
                    // stream byte q of the window sits at window byte 4 (q / 4) + 3 - q % 4 (LE words)
                    let q0 = rel / 8;
                    assert!(q0 + 2 < 16, "cpu_mul1: K = {} lane {g} position {j} leaves the 16-byte window", b.k());
                    for (lb, q) in [(3, q0), (2, q0 + 1), (1, q0 + 2)] {
                        gr.ctrl[4 * (j % 4) + 16 * (j / 4) + lb] = (4 * (q / 4) + 3 - q % 4) as u8;
                    }
                    gr.shv[j] = 16 - (rel % 8) as i32;
                }
            }
            groups
        };
        let lanes = lane_table(b);
        Tables { n32, lo, groups, lanes }
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

/// `xh = H (x * suh)` per 128-block, `x` `[T][k]` (the reference arms and tests;
/// the production path transforms per block, `prep_block`)
#[cfg(test)]
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

/// `y = (H y') / 128 * svh` per 128-block, `raw`, `y` `[T][n]` (the reference arms; the production
/// path applies it per block, `had_out_block`)
#[cfg(test)]
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
/// (the reference arms and tests; the production path transforms per block, `prep_block`)
#[cfg(test)]
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

    /// read back an index whose writer finished before (block counter or barrier)
    #[inline]
    fn get(self, i: usize) -> f32 {
        assert!(i < self.1);
        // SAFETY: in bounds (asserted); the write happened before (the callers' contract)
        unsafe { *self.0.add(i) }
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
        m16: __m256i,
        mul: __m256i,
        one8: __m256i,
        one16: __m256i,
        expm: __m256i,
        magic: __m256i,
        kinv: __m256,
        caff: __m256,
        sh13: __m256i,
        f1024: __m256i,
        caff1024: __m256,
        kinv8192: __m256,
        caff_v: __m256,
    }

    /// # Safety
    /// The CPU must have AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn consts() -> Consts {
        Consts {
            m16: _mm256_set1_epi32(0xffff),
            mul: _mm256_set1_epi32(MUL1 as i32),
            one8: _mm256_set1_epi8(1),
            one16: _mm256_set1_epi16(1),
            expm: _mm256_set1_epi32(0x7F80_0000),
            magic: _mm256_set1_epi32(0x06C0_0000),
            kinv: _mm256_set1_ps(KINV),
            caff: _mm256_set1_ps(CAFF),
            sh13: _mm256_set1_epi16(1 << 13),
            f1024: _mm256_set1_epi32(0x4480_0000),
            caff1024: _mm256_set1_ps(CAFF - 1024.0 * KINV),
            kinv8192: _mm256_set1_ps(8192.0 * KINV),
            caff_v: _mm256_set1_ps(CAFF - 1024.0 * (8192.0 * KINV)),
        }
    }

    /// The 16-byte window of lane `ln`, in both 128-bit halves (no division, no scalar loads).
    ///
    /// # Safety
    /// The CPU must have AVX2; `tile` points at `4 * n32` readable bytes of the lane's table.
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn window(tile: *const u8, ln: &Lane) -> __m256i {
        // SAFETY (all loads): `off + 16 <= 4 n32` and `16 <= 4 n32` (n32 >= 8, Tables::new)
        let w = unsafe {
            let hi = _mm_loadu_si128(tile.add(ln.off) as *const __m128i);
            match ln.alignr {
                0 => hi,
                4 => _mm_alignr_epi8::<4>(_mm_loadu_si128(tile as *const __m128i), hi),
                8 => _mm_alignr_epi8::<8>(_mm_loadu_si128(tile as *const __m128i), hi),
                _ => _mm_alignr_epi8::<12>(_mm_loadu_si128(tile as *const __m128i), hi),
            }
        };
        _mm256_broadcastsi128_si256(w)
    }

    /// The C1 lane decoder: the states of `decode8` from a division-free window, and the weight
    /// through `weights_fast`. Same bits as `decode8` (tested for every lane of every bitrate and,
    /// through `weights_fast`, every state).
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA; `tile` points at `4 * n32` readable bytes.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn decode8_fast(tile: *const u8, ln: &Lane, k: &Consts) -> __m256 {
        // SAFETY: the caller's contract; Lane is 32-byte aligned, ctrl and shv its first 64 bytes
        unsafe {
            let win = window(tile, ln);
            let (ctrl, shv) = (_mm256_load_si256(ln.ctrl.as_ptr() as *const __m256i), _mm256_load_si256(ln.shv.as_ptr() as *const __m256i));
            weights_fast(_mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(win, ctrl), shv), k.m16), k)
        }
    }

    /// `weights` with the same value from fewer and shorter steps: the byte sum times 2^13 straight
    /// from `vpmaddwd` (pairs <= 510 times 8192), added to the bits of 1024.0 gives the f32 `1024 + s`
    /// exactly (s < 1024 fills the mantissa below 2^10); `(1024 + s) k_inv + (-3.453125 - 1024 k_inv)`
    /// in one FMA. Both products are exact in f32 ((1024 + s) * 1774 < 2^22, times 2^-18) and the sum
    /// is a multiple of 2^-18 below 4 in magnitude, so the FMA, the separate mul + add and the
    /// `s k_inv - 3.453125` of `weights` are one exact value; the fp16 rounding is unchanged.
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn weights_fast(st: __m256i, k: &Consts) -> __m256 {
        let x = _mm256_mullo_epi32(st, k.mul);
        let s13 = _mm256_madd_epi16(_mm256_maddubs_epi16(x, k.one8), k.sh13);
        let f = _mm256_castsi256_ps(_mm256_add_epi32(s13, k.f1024));
        let v = _mm256_fmadd_ps(f, k.kinv, k.caff1024);
        let c = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_and_si256(_mm256_castps_si256(v), k.expm), k.magic));
        _mm256_sub_ps(_mm256_add_ps(v, c), c)
    }

    /// `weights_fast` with the byte sum in one `vpdpbusd` (AVX-VNNI): the four product bytes
    /// times 1 summed into the bits of 1024.0 give the f32 `1024 + s 2^-13` exactly (one ulp of
    /// 1024.0 is 2^-13 and s <= 1020 stays inside its binade), and `f (8192 k_inv) +
    /// (-3.453125 - 1024 (8192 k_inv))` in one FMA is the exact `s k_inv - 3.453125` (the exact
    /// sum is a multiple of 2^-18 below 4 in magnitude, so the FMA's one rounding keeps it; both
    /// constants are exact in f32). One instruction for `vpmaddubsw` + `vpmaddwd` + `vpaddd`; the
    /// fp16 rounding is unchanged. Same bits as `weights` for every state (tested).
    ///
    /// # Safety
    /// The CPU must have AVX2, FMA and AVX-VNNI.
    #[inline]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    pub(super) unsafe fn weights_vnni(st: __m256i, k: &Consts) -> __m256 {
        let x = _mm256_mullo_epi32(st, k.mul);
        let f = _mm256_castsi256_ps(_mm256_dpbusd_avx_epi32(k.f1024, x, k.one8));
        let v = _mm256_fmadd_ps(f, k.kinv8192, k.caff_v);
        let c = _mm256_castsi256_ps(_mm256_add_epi32(_mm256_and_si256(_mm256_castps_si256(v), k.expm), k.magic));
        _mm256_sub_ps(_mm256_add_ps(v, c), c)
    }

    /// The C1 unit: `unit` operation for operation on the products and sums (separate
    /// `vmulps` + `vaddps`, never fused), with `decode8_fast`, no bounds checks in the k loop
    /// (asserted once per unit) and every line of the next-but-`PF_ROWS` tile row prefetched.
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA.
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn unit_fast<const NT: usize>(m: &Mul1Matrix, tab: &Tables, xp: &[f32], t0: usize, nb0: usize, ntc: usize, raw: Out) {
        let (tk, tn, tb) = (m.k / 16, m.n / 16, m.bitrate.tile_bytes());
        assert!(ntc <= UNIT_TILES && nb0 + ntc <= tn && xp.len() >= (t0 + NT) * tk * 32);
        assert!(m.trellis.len() >= tk * tn * tb && tb == 4 * tab.n32);
        let k = unsafe { consts() };
        let mut acc = [[[_mm256_setzero_ps(); NT]; 8]; UNIT_TILES];
        let base = m.trellis.as_ptr();
        let xb = xp.as_ptr();
        let run = ntc * tb;
        for kb in 0..tk {
            if kb + PF_ROWS < tk {
                let pf = ((kb + PF_ROWS) * tn + nb0) * tb;
                // every 64-byte line of the run [pf, pf + run)
                let mut o = 0;
                while o < run + 63 {
                    // SAFETY: pf + min(o, run - 1) lies inside the trellis (row kb + PF_ROWS exists)
                    unsafe { _mm_prefetch::<_MM_HINT_T0>(base.add(pf + o.min(run - 1)) as *const i8) };
                    o += 64;
                }
            }
            // SAFETY: token t's lane-ordered row kb starts at ((t0 + t) tk + kb) 32 < xp.len() (asserted)
            let xr = unsafe { xb.add((t0 * tk + kb) * 32) };
            for (tc, acc_tc) in acc.iter_mut().enumerate().take(ntc) {
                // SAFETY: (kb tn + nb0 + tc + 1) tb <= tk tn tb <= len (asserted)
                let tile = unsafe { base.add((kb * tn + nb0 + tc) * tb) };
                for (c, acc_c) in acc_tc.iter_mut().enumerate() {
                    let mut a = *acc_c;
                    for q in 0..4 {
                        // SAFETY: AVX2 + FMA (caller), the tile has 4 n32 bytes, 4 c + q < 32
                        let w = unsafe { decode8_fast(tile, tab.lanes.get_unchecked(4 * c + q), &k) };
                        for (t, at) in a.iter_mut().enumerate() {
                            // SAFETY: inside xp (asserted above)
                            let xv = unsafe { _mm256_loadu_ps(xr.add(t * tk * 32 + q * 8)) };
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

    /// Lane `G` of a K = 3 tile (its window, control and shifts constants of [`K3_LANES`]):
    /// decode, then `a[t] = a[t] + w * x[t]` for the NT tokens, as `unit_fast`'s inner step.
    ///
    /// # Safety
    /// The CPU must have AVX2 and FMA; `tile` points at a K = 3 tile (96 bytes), `xr` at token
    /// 0's 32 lane-ordered inputs of the tile row, token t's at `xr + t tk32`.
    #[inline]
    #[target_feature(enable = "avx2,fma")]
    unsafe fn group_k3<const G: usize, const NT: usize>(tile: *const u8, xr: *const f32, tk32: usize, a: &mut [__m256; NT], k: &Consts) {
        // SAFETY: the caller's contract
        unsafe {
            let w = decode8_fast(tile, &K3_LANES[G], k);
            for (t, at) in a.iter_mut().enumerate() {
                let xv = _mm256_loadu_ps(xr.add(t * tk32 + (G % 4) * 8));
                *at = _mm256_add_ps(*at, _mm256_mul_ps(w, xv));
            }
        }
    }

    /// `group_k3` with `weights_vnni`: the same weights, products and sums.
    ///
    /// # Safety
    /// As `group_k3`; the CPU must also have AVX-VNNI.
    #[inline]
    #[target_feature(enable = "avx2,fma,avxvnni")]
    unsafe fn group_k3_vnni<const G: usize, const NT: usize>(tile: *const u8, xr: *const f32, tk32: usize, a: &mut [__m256; NT], k: &Consts) {
        // SAFETY: the caller's contract; Lane is 32-byte aligned, ctrl and shv its first 64 bytes
        unsafe {
            let ln = &K3_LANES[G];
            let (ctrl, shv) = (_mm256_load_si256(ln.ctrl.as_ptr() as *const __m256i), _mm256_load_si256(ln.shv.as_ptr() as *const __m256i));
            let w = weights_vnni(_mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(window(tile, ln), ctrl), shv), k.m16), k);
            for (t, at) in a.iter_mut().enumerate() {
                let xv = _mm256_loadu_ps(xr.add(t * tk32 + (G % 4) * 8));
                *at = _mm256_add_ps(*at, _mm256_mul_ps(w, xv));
            }
        }
    }

    /// Defines `$name`, one K = 3 tile of a unit: the 32 lanes through `$group` in stream order,
    /// lanes 4 C .. 4 C + 3 into the accumulators of column group C.
    macro_rules! tile_k3_fn {
        ($name:ident, $group:ident, $feat:literal) => {
            /// # Safety
            /// As the lane function it calls.
            #[inline]
            #[target_feature(enable = $feat)]
            unsafe fn $name<const NT: usize>(tile: *const u8, xr: *const f32, tk32: usize, acc: &mut [[__m256; NT]; 8], k: &Consts) {
                // SAFETY: the caller's contract
                unsafe {
                    $group::<0, NT>(tile, xr, tk32, &mut acc[0], k);
                    $group::<1, NT>(tile, xr, tk32, &mut acc[0], k);
                    $group::<2, NT>(tile, xr, tk32, &mut acc[0], k);
                    $group::<3, NT>(tile, xr, tk32, &mut acc[0], k);
                    $group::<4, NT>(tile, xr, tk32, &mut acc[1], k);
                    $group::<5, NT>(tile, xr, tk32, &mut acc[1], k);
                    $group::<6, NT>(tile, xr, tk32, &mut acc[1], k);
                    $group::<7, NT>(tile, xr, tk32, &mut acc[1], k);
                    $group::<8, NT>(tile, xr, tk32, &mut acc[2], k);
                    $group::<9, NT>(tile, xr, tk32, &mut acc[2], k);
                    $group::<10, NT>(tile, xr, tk32, &mut acc[2], k);
                    $group::<11, NT>(tile, xr, tk32, &mut acc[2], k);
                    $group::<12, NT>(tile, xr, tk32, &mut acc[3], k);
                    $group::<13, NT>(tile, xr, tk32, &mut acc[3], k);
                    $group::<14, NT>(tile, xr, tk32, &mut acc[3], k);
                    $group::<15, NT>(tile, xr, tk32, &mut acc[3], k);
                    $group::<16, NT>(tile, xr, tk32, &mut acc[4], k);
                    $group::<17, NT>(tile, xr, tk32, &mut acc[4], k);
                    $group::<18, NT>(tile, xr, tk32, &mut acc[4], k);
                    $group::<19, NT>(tile, xr, tk32, &mut acc[4], k);
                    $group::<20, NT>(tile, xr, tk32, &mut acc[5], k);
                    $group::<21, NT>(tile, xr, tk32, &mut acc[5], k);
                    $group::<22, NT>(tile, xr, tk32, &mut acc[5], k);
                    $group::<23, NT>(tile, xr, tk32, &mut acc[5], k);
                    $group::<24, NT>(tile, xr, tk32, &mut acc[6], k);
                    $group::<25, NT>(tile, xr, tk32, &mut acc[6], k);
                    $group::<26, NT>(tile, xr, tk32, &mut acc[6], k);
                    $group::<27, NT>(tile, xr, tk32, &mut acc[6], k);
                    $group::<28, NT>(tile, xr, tk32, &mut acc[7], k);
                    $group::<29, NT>(tile, xr, tk32, &mut acc[7], k);
                    $group::<30, NT>(tile, xr, tk32, &mut acc[7], k);
                    $group::<31, NT>(tile, xr, tk32, &mut acc[7], k);
                }
            }
        };
    }
    tile_k3_fn!(tile_k3, group_k3, "avx2,fma");
    tile_k3_fn!(tile_k3_vnni, group_k3_vnni, "avx2,fma,avxvnni");

    /// Defines `$name`, the K = 3 unit over the tile function `$tile` with the target features
    /// `$feat`: `unit_k3` and its AVX-VNNI twin `unit_k3_vnni` (same code, one decoder each).
    macro_rules! unit_k3_fn {
        ($(#[$doc:meta])* $name:ident, $tile:ident, $feat:literal) => {
            $(#[$doc])*
            #[target_feature(enable = $feat)]
            pub(super) unsafe fn $name<const NT: usize>(m: &Mul1Matrix, xp: &[f32], t0: usize, nb0: usize, ntc: usize, raw: Out) {
                let (tk, tn, tb) = (m.k / 16, m.n / 16, m.bitrate.tile_bytes());
                assert!(m.bitrate == K3 && tb == 96);
                assert!(ntc <= UNIT_TILES && nb0 + ntc <= tn && xp.len() >= (t0 + NT) * tk * 32);
                assert!(m.trellis.len() >= tk * tn * tb);
                let k = unsafe { consts() };
                let mut acc = [[[_mm256_setzero_ps(); NT]; 8]; UNIT_TILES];
                let base = m.trellis.as_ptr();
                let xb = xp.as_ptr();
                let run = ntc * tb;
                let tk32 = tk * 32;
                for kb in 0..tk {
                    if kb + PF_ROWS < tk {
                        let pf = ((kb + PF_ROWS) * tn + nb0) * tb;
                        let mut o = 0;
                        while o < run + 63 {
                            // SAFETY: pf + min(o, run - 1) lies inside the trellis (row kb + PF_ROWS exists)
                            unsafe { _mm_prefetch::<_MM_HINT_T0>(base.add(pf + o.min(run - 1)) as *const i8) };
                            o += 64;
                        }
                    }
                    // SAFETY: inside xp (asserted)
                    let xr = unsafe { xb.add((t0 * tk + kb) * 32) };
                    for (tc, acc_tc) in acc.iter_mut().enumerate().take(ntc) {
                        // SAFETY: (kb tn + nb0 + tc + 1) tb <= len (asserted); the target features (caller)
                        unsafe { $tile::<NT>(base.add((kb * tn + nb0 + tc) * tb), xr, tk32, acc_tc, &k) };
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
        };
    }
    unit_k3_fn!(
        /// `unit_fast` for K = 3 with the 32 lanes of a tile unrolled over compile-time lane
        /// constants (sybil `ft_core.h:176-249`, `decode8<g>` / `tile_accum<C>`): the same operations
        /// in the same order per accumulator, no per-lane table loads or branches.
        ///
        /// # Safety
        /// The CPU must have AVX2 and FMA; `m` is a K = 3 matrix.
        unit_k3, tile_k3, "avx2,fma"
    );
    unit_k3_fn!(
        /// `unit_k3` with `weights_vnni` (one `vpdpbusd` for the byte sum): the same bits.
        ///
        /// # Safety
        /// The CPU must have AVX2, FMA and AVX-VNNI; `m` is a K = 3 matrix.
        unit_k3_vnni, tile_k3_vnni, "avx2,fma,avxvnni"
    );

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
        // SAFETY: Group is 32-byte aligned (repr align 32), ctrl and shv are its first 64 bytes
        let (ctrl, shv) = unsafe { (_mm256_load_si256(g.ctrl.as_ptr() as *const __m256i), _mm256_load_si256(g.shv.as_ptr() as *const __m256i)) };
        // lane j = stream bytes Q0, Q0 + 1, Q0 + 2 MSB first; >> (16 - rel % 8), low 16 bits
        let st = _mm256_and_si256(_mm256_srlv_epi32(_mm256_shuffle_epi8(win, ctrl), shv), k.m16);
        weights(st, k)
    }

    /// the codec weight of eight 16-bit states, one per 32-bit lane
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn weights(st: __m256i, k: &Consts) -> __m256 {
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

    /// The first #180 lane decoder (two `permutevar8x32` word selects and a funnel shift), kept as
    /// the reference of `cpu_mul1_decode_and_schedule_equal_v1_bits` and the benchmark arms.
    ///
    /// # Safety
    /// The CPU must have AVX2; `tile` points at `4 * n32` readable bytes.
    #[cfg(test)]
    #[inline]
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn decode8_v1(tile: *const u8, n32: usize, g: &Group, k: &Consts) -> __m256 {
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
        weights(st, k)
    }

    /// `decode8`, or under test with `V1` the first #180 decoder `decode8_v1` (same states)
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn dec<const V1: bool>(tile: *const u8, n32: usize, g: &Group, k: &Consts) -> __m256 {
        #[cfg(test)]
        if V1 {
            // SAFETY: as decode8
            return unsafe { decode8_v1(tile, n32, g, k) };
        }
        // SAFETY: the caller's contract
        unsafe { decode8(tile, n32, g, k) }
    }

    /// The AVX2 twin of `unit_scalar`, operation for operation.
    ///
    /// # Safety
    /// The CPU must have AVX2.
    #[target_feature(enable = "avx2")]
    pub(super) unsafe fn unit<const NT: usize, const V1: bool>(m: &Mul1Matrix, tab: &Tables, xp: &[f32], t0: usize, nb0: usize, ntc: usize, raw: Out) {
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
                        let w = unsafe { dec::<V1>(tile, tab.n32, &tab.groups[4 * c + q], &k) };
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

/// Which kernel runs: `NEW`, the production path (#183 C1 follow-up): `avx2::unit_fast` units
/// handed out by an atomic counter on the persistent [`pool`], each 128-column block finished
/// (Hadamard out, activation, down's input transform) by the worker that completes its last unit.
/// Under test also the two earlier kernels, the reference arms of the bit-identity tests and the
/// benchmark, kept as they were in [`reference`]: `V2` (#183: `decode8` with the wrap taken
/// modulo, units from a counter, scoped threads per call, `std::sync::Barrier`, the activation
/// on worker 0) and `V1` (#180: the two-word decoder, units split statically by worker). All
/// give the same bits: the same states and f32 operations per output column, and a unit or a
/// block is computed by exactly one worker whichever it is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Impl {
    arm: u8,
}

impl Impl {
    pub(crate) const NEW: Impl = Impl { arm: 0 };
    #[cfg(test)]
    pub(crate) const V1: Impl = Impl { arm: 1 };
    #[cfg(test)]
    pub(crate) const V2: Impl = Impl { arm: 2 };
    /// the production path with `Kern::Avx2`, as on an AVX2 CPU without FMA (test only)
    #[cfg(test)]
    pub(crate) const NEW_NO_FMA: Impl = Impl { arm: 3 };

    /// the reference arms run their own code (test only)
    #[cfg(test)]
    fn reference(self) -> bool {
        self.arm == 1 || self.arm == 2
    }

    /// the inner kernel this arm uses on `path`
    fn kern(self, path: Path) -> Kern {
        match kern(path) {
            Kern::Fast if self.arm == 3 => Kern::Avx2,
            k => k,
        }
    }
}

/// the inner kernel of the production path
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kern {
    Scalar,
    /// `avx2::unit` with `decode8`, the #183 unit: an AVX2 CPU without FMA
    Avx2,
    /// `avx2::unit_k3` / `avx2::unit_fast` (AVX2 + FMA)
    Fast,
}

/// `Kern::Fast` needs FMA beside AVX2 (every Intel CPU since Haswell and AMD since Zen has
/// both); an AVX2 CPU without FMA keeps the #183 AVX2 unit. All three give the same bits.
fn kern(path: Path) -> Kern {
    if !use_avx2(path) {
        return Kern::Scalar;
    }
    #[cfg(target_arch = "x86_64")]
    if std::arch::is_x86_feature_detected!("fma") {
        return Kern::Fast;
    }
    Kern::Avx2
}

/// The K = 3 unit of `Kern::Fast` takes the byte sum from `vpdpbusd` (`avx2::unit_k3_vnni`, the
/// same bits) on a CPU with AVX-VNNI (Intel since Alder Lake, AMD since Zen 5).
#[cfg(target_arch = "x86_64")]
fn vnni() -> bool {
    std::arch::is_x86_feature_detected!("avxvnni")
}

/// One chunk (tile columns `nb0 .. nb0 + ntc`, `ntc <= UNIT_TILES`) for every token of `xp`.
fn unit_new(kern: Kern, m: &Mul1Matrix, tab: &Tables, xp: &[f32], tokens: usize, nb0: usize, ntc: usize, raw: Out) {
    let mut t0 = 0;
    while t0 < tokens {
        let nt = (tokens - t0).min(TOK_TILE);
        #[cfg(target_arch = "x86_64")]
        if kern == Kern::Fast && m.bitrate == K3 && vnni() {
            // SAFETY: `Kern::Fast` is chosen only after AVX2 and FMA were detected on this CPU,
            // `vnni` checked AVX-VNNI
            unsafe {
                match nt {
                    1 => avx2::unit_k3_vnni::<1>(m, xp, t0, nb0, ntc, raw),
                    2 => avx2::unit_k3_vnni::<2>(m, xp, t0, nb0, ntc, raw),
                    3 => avx2::unit_k3_vnni::<3>(m, xp, t0, nb0, ntc, raw),
                    _ => avx2::unit_k3_vnni::<4>(m, xp, t0, nb0, ntc, raw),
                }
            }
            t0 += nt;
            continue;
        }
        #[cfg(target_arch = "x86_64")]
        if kern == Kern::Fast && m.bitrate == K3 {
            // SAFETY: `Kern::Fast` is chosen only after AVX2 and FMA were detected on this CPU
            unsafe {
                match nt {
                    1 => avx2::unit_k3::<1>(m, xp, t0, nb0, ntc, raw),
                    2 => avx2::unit_k3::<2>(m, xp, t0, nb0, ntc, raw),
                    3 => avx2::unit_k3::<3>(m, xp, t0, nb0, ntc, raw),
                    _ => avx2::unit_k3::<4>(m, xp, t0, nb0, ntc, raw),
                }
            }
            t0 += nt;
            continue;
        }
        #[cfg(target_arch = "x86_64")]
        if kern == Kern::Fast {
            // SAFETY: `Kern::Fast` is chosen only after AVX2 and FMA were detected on this CPU
            unsafe {
                match nt {
                    1 => avx2::unit_fast::<1>(m, tab, xp, t0, nb0, ntc, raw),
                    2 => avx2::unit_fast::<2>(m, tab, xp, t0, nb0, ntc, raw),
                    3 => avx2::unit_fast::<3>(m, tab, xp, t0, nb0, ntc, raw),
                    _ => avx2::unit_fast::<4>(m, tab, xp, t0, nb0, ntc, raw),
                }
            }
            t0 += nt;
            continue;
        }
        #[cfg(target_arch = "x86_64")]
        if kern == Kern::Avx2 {
            // SAFETY: `Kern::Avx2` is chosen only after AVX2 was detected on this CPU
            unsafe {
                match nt {
                    1 => avx2::unit::<1, false>(m, tab, xp, t0, nb0, ntc, raw),
                    2 => avx2::unit::<2, false>(m, tab, xp, t0, nb0, ntc, raw),
                    3 => avx2::unit::<3, false>(m, tab, xp, t0, nb0, ntc, raw),
                    _ => avx2::unit::<4, false>(m, tab, xp, t0, nb0, ntc, raw),
                }
            }
            t0 += nt;
            continue;
        }
        let _ = kern;
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

/// tile columns per 128-column block
const BLOCK_TILES: usize = HAD / 16;

/// Guided chunk sizes (#183 C1): over `tiles` tile columns taken in order by `k` workers,
/// chunks of `UNIT_TILES` while more than `2 k` of them remain, then of 2, then of 1, so the last
/// chunk a slow core (an E-core takes ~2.2 x a P-core's time per unit, measured) starts is one
/// tile column. A chunk starts at a multiple of its size and every size divides `BLOCK_TILES`, so
/// no chunk crosses a block. A column's sums do not depend on the chunk it is in.
fn chunks(tiles: usize, k: usize) -> Vec<(usize, usize)> {
    let mut v = Vec::with_capacity(tiles);
    let mut t = 0;
    while t < tiles {
        let rem = tiles - t;
        let sz = if k == 1 || rem > 2 * k * UNIT_TILES {
            UNIT_TILES
        } else if rem > 2 * k * 2 {
            2
        } else {
            1
        };
        let sz = sz.min(rem);
        debug_assert!(t % sz == 0 && BLOCK_TILES % sz == 0);
        v.push((t, sz));
        t += sz;
    }
    v
}

/// the next chunk from the shared counter (a P-core takes more than an E-core)
#[inline]
fn take(next: &AtomicUsize, plan: &[(usize, usize)]) -> Option<(usize, usize)> {
    plan.get(next.fetch_add(1, Ordering::Relaxed)).copied()
}

/// Tile columns left per 128-column block; the worker whose chunk completes a block finishes it.
struct Blocks(Vec<AtomicUsize>);

impl Blocks {
    fn new(blocks: usize, tiles_each: usize) -> Blocks {
        Blocks((0..blocks).map(|_| AtomicUsize::new(tiles_each)).collect())
    }

    /// true for exactly one caller per block: the one that hands in its last `ntc` tiles.
    /// AcqRel: the finisher sees every write of the block's chunks (release sequence).
    fn complete(&self, b: usize, ntc: usize) -> bool {
        self.0[b].fetch_sub(ntc, Ordering::AcqRel) == ntc
    }
}

/// `had_out` of output block `b` (columns `128 b ..`) for every token: the same operations per
/// element as `had_out`, written to `y`.
fn had_out_block(raw: Out, m: &Mul1Matrix, b: usize, tokens: usize, y: Out) {
    let mut v = [0f32; HAD];
    for t in 0..tokens {
        let at = t * m.n + b * HAD;
        for (i, d) in v.iter_mut().enumerate() {
            *d = raw.get(at + i);
        }
        fwht128(&mut v);
        for (i, d) in v.iter().enumerate() {
            y.put(at + i, (*d * (1.0 / 128.0)) * Mul1Matrix::scale(m.svh, b * HAD + i));
        }
    }
}

/// The lane permutation of one 128-block `h` (block `b` of a `[T][k]` input, token `t`) into
/// `xp`: `permute`'s `xp[t][kb][q][j] = h[16 kb' + 2 q + ROW_OFF[j % 4]]` for its 8 tile rows.
#[inline]
fn permute_block(h: &[f32; HAD], k: usize, b: usize, t: usize, xp: Out) {
    let tk = k / 16;
    for kl in 0..BLOCK_TILES {
        for q in 0..4 {
            for j in 0..8 {
                xp.put((t * tk + b * BLOCK_TILES + kl) * 32 + q * 8 + j, h[16 * kl + 2 * q + ROW_OFF[j % 4]]);
            }
        }
    }
}

/// `had_in` + `permute` of input block `b` (rows `128 b ..` of `m`) for every token: the same
/// operations per element (`x * suh`, the FWHT of the block, the lane order), written to `xp`.
fn prep_block(x: &[f32], m: &Mul1Matrix, b: usize, tokens: usize, xp: Out) {
    let mut v = [0f32; HAD];
    for t in 0..tokens {
        for (r, d) in v.iter_mut().enumerate() {
            *d = x[t * m.k + b * HAD + r] * Mul1Matrix::scale(m.suh, b * HAD + r);
        }
        fwht128(&mut v);
        permute_block(&v, m.k, b, t, xp);
    }
}

/// the FFN's activation `h = act(gate, up)`: [`silu_mul`] in [`expert_ffn`]; the GLM routed
/// experts' clamped SwiGLU (`glm5_moe::swiglu_clamp`) in the #188 CPU lane
pub type Act = dyn Fn(f32, f32) -> f32 + Sync;

/// FFN block `b` of the intermediate dimension, for every token: `had_out` of gate and up,
/// `act` (`silu_mul` in `expert_ffn`), `had_in` of down and the lane permutation into `xd`, element for element the
/// operations of the whole-vector functions (each is local to its 128-block).
fn act_block(e: &Mul1Expert, b: usize, tokens: usize, rg: Out, ru: Out, xd: Out, act: &Act) {
    let i = e.inter;
    let (mut g, mut u) = ([0f32; HAD], [0f32; HAD]);
    for t in 0..tokens {
        let at = t * i + b * HAD;
        for (m, raw, v) in [(&e.gate, rg, &mut g), (&e.up, ru, &mut u)] {
            for (r, d) in v.iter_mut().enumerate() {
                *d = raw.get(at + r);
            }
            fwht128(v);
            for (r, d) in v.iter_mut().enumerate() {
                *d = (*d * (1.0 / 128.0)) * Mul1Matrix::scale(m.svh, b * HAD + r);
            }
        }
        let mut h = [0f32; HAD];
        for (r, d) in h.iter_mut().enumerate() {
            *d = act(g[r], u[r]) * Mul1Matrix::scale(e.down.suh, b * HAD + r);
        }
        fwht128(&mut h);
        permute_block(&h, i, b, t, xd);
    }
}

/// Per-call buffers from a per-thread scratch vector, reused across calls (a fresh zeroed
/// allocation of the FFN's 112 KB at T 1 cost ~20 us per call, measured). Every region is
/// written before it is read in a call, so stale values never reach an output. A call while the
/// scratch is in use (nested), or one needing more than `SCRATCH_KEEP` floats, allocates instead.
#[cfg(test)]
thread_local! {
    static POISON_SCRATCH: std::cell::Cell<bool> = const { std::cell::Cell::new(true) };
}

fn with_scratch<R>(len: usize, f: impl FnOnce(&mut [f32]) -> R) -> R {
    const SCRATCH_KEEP: usize = 1 << 20;
    thread_local! {
        static SCRATCH: std::cell::RefCell<Vec<f32>> = const { std::cell::RefCell::new(Vec::new()) };
    }
    if len > SCRATCH_KEEP {
        return f(&mut vec![0f32; len]);
    }
    SCRATCH.with(|s| match s.try_borrow_mut() {
        Ok(mut v) => {
            if v.len() < len {
                v.resize(len, 0.0);
            }
            // under test every call starts from NaN, so a region a call fails to write cannot
            // pass on the previous call's values (the benchmark switches this off)
            #[cfg(test)]
            if POISON_SCRATCH.with(|p| p.get()) {
                v[..len].fill(f32::NAN);
            }
            f(&mut v[..len])
        }
        Err(_) => f(&mut vec![0f32; len]),
    })
}

/// One phase of a pool run: items `0 .. total` handed out by `next`, each counted in `done`
/// after it is complete. A later phase waits for the work (`wait`), never for the workers, so a
/// worker that starts late, or is descheduled between items, holds nobody up.
struct Phase {
    next: AtomicUsize,
    done: AtomicUsize,
    total: usize,
}

impl Phase {
    fn new(total: usize) -> Phase {
        Phase { next: AtomicUsize::new(0), done: AtomicUsize::new(0), total }
    }

    /// run `f` on items until none is left
    fn work(&self, mut f: impl FnMut(usize)) {
        loop {
            let j = self.next.fetch_add(1, Ordering::Relaxed);
            if j >= self.total {
                return;
            }
            f(j);
            self.done.fetch_add(1, Ordering::Release);
        }
    }

    /// wait until every item is complete
    fn wait(&self) {
        wait_for(&self.done, self.total);
    }
}

/// Spin (then yield) until `done` reaches `total`. Acquire: each counted piece of work happened
/// before its release increment, and the load that sees `total` synchronizes with all of them.
fn wait_for(done: &AtomicUsize, total: usize) {
    let mut spins = 0u32;
    while done.load(Ordering::Acquire) < total {
        if spins < pool::SPINS {
            spins += 1;
            std::hint::spin_loop();
        } else {
            std::thread::yield_now();
        }
    }
}

/// `y[t] = x[t] W` for `x` `[T][k]`, `y` `[T][n]`, output columns in units over `threads` workers.
pub fn gemv(m: &Mul1Matrix, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    gemv_with(Impl::NEW, m, x, y, threads, path)
}

pub(crate) fn gemv_with(im: Impl, m: &Mul1Matrix, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    #[cfg(test)]
    if im.reference() {
        return reference::gemv(im, m, x, y, threads, path);
    }
    assert!(x.len() % m.k == 0, "cpu_mul1::gemv: x is not [T][{}]", m.k);
    let tokens = x.len() / m.k;
    assert_eq!(y.len(), tokens * m.n, "cpu_mul1::gemv: y is not [T][{}]", m.n);
    let (kern, tab) = (im.kern(path), tables(m.bitrate));
    let nxp = tokens * (m.k / 16) * 32;
    with_scratch(nxp + y.len(), |s| {
        let (xp, raw) = s.split_at_mut(nxp);
        let (oxp, out, yo) = (Out(xp.as_mut_ptr(), xp.len()), Out(raw.as_mut_ptr(), raw.len()), Out(y.as_mut_ptr(), y.len()));
        let k = threads.clamp(1, units_of(m));
        let plan = chunks(m.n / 16, k);
        let blocks = Blocks::new(m.n / HAD, BLOCK_TILES);
        let (prep, cols) = (Phase::new(m.k / HAD), AtomicUsize::new(0));
        pool::run(k, &|_| {
            // the input transform, one 128-block of x at a time, then the columns
            prep.work(|b| prep_block(x, m, b, tokens, oxp));
            prep.wait();
            // SAFETY: every write to xp is complete (`prep.wait`); read only from here on
            let xp = unsafe { std::slice::from_raw_parts(oxp.0 as *const f32, oxp.1) };
            while let Some((nb0, ntc)) = take(&cols, &plan) {
                unit_new(kern, m, tab, xp, tokens, nb0, ntc, out);
                let b = nb0 / BLOCK_TILES;
                if blocks.complete(b, ntc) {
                    had_out_block(out, m, b, tokens, yo);
                }
            }
        });
    });
}

/// The expert FFN `y = down(silu(gate(x)) * up(x))` for `x`, `y` `[T][hidden]`, with
/// `silu(g) * u = g / (1 + exp(-g)) * u` (`cpu_nvfp4`, `silu_mul640`). One pool run in three
/// phases: the input transforms of gate and up per 128-block of x; the gate and up tile columns
/// block by block in guided chunks, the worker that completes a block applying `had_out`, the
/// activation and down's input transform to it; the down tile columns, the worker that completes
/// a block of `y` applying `had_out`. Bit for bit `gemv(down, silu(gemv(gate, x)) * gemv(up, x))`.
pub fn expert_ffn(e: &Mul1Expert, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    expert_ffn_with(Impl::NEW, e, x, y, threads, path)
}

pub(crate) fn expert_ffn_with(im: Impl, e: &Mul1Expert, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
    #[cfg(test)]
    if im.reference() {
        return reference::expert_ffn(im, e, x, y, threads, path);
    }
    experts_ffn_with(im, std::slice::from_ref(e), x, y, &silu_mul, threads, path)
}

/// #188 CPU lane: the expert FFN of every expert in `es` on the same rows `x` `[T][hidden]`, in
/// ONE pool run; `ys` `[n][T][hidden]`, expert `j`'s rows at `j T hidden`. The three phases of
/// [`expert_ffn`] run over the experts side by side (input transforms of every expert, then the
/// gate / up columns of every expert, then, once every expert's activation is complete, the down
/// columns of every expert). A column, a block and its finisher do the operations of
/// `expert_ffn` with `act` in place of `silu_mul`, so with `act = silu_mul` every output is bit
/// for bit `expert_ffn` of that expert alone, and with any `act` bit for bit
/// `gemv(down, act(gemv(gate, x), gemv(up, x)))`. Every expert has the shape and bitrate of the
/// first.
pub fn experts_ffn(es: &[Mul1Expert], x: &[f32], ys: &mut [f32], act: &Act, threads: usize, path: Path) {
    experts_ffn_with(Impl::NEW, es, x, ys, act, threads, path)
}

fn experts_ffn_with(im: Impl, es: &[Mul1Expert], x: &[f32], ys: &mut [f32], act: &Act, threads: usize, path: Path) {
    let n = es.len();
    if n == 0 {
        return;
    }
    let e0 = &es[0];
    let (h, i) = (e0.hidden, e0.inter);
    for e in es {
        assert!(
            e.hidden == h && e.inter == i && e.gate.bitrate == e0.gate.bitrate && e.up.bitrate == e0.up.bitrate && e.down.bitrate == e0.down.bitrate,
            "cpu_mul1::experts_ffn: the experts differ in shape or bitrate"
        );
    }
    assert!(x.len() % h == 0, "cpu_mul1::expert_ffn: x is not [T][{h}]");
    let tokens = x.len() / h;
    assert_eq!(ys.len(), n * tokens * h, "cpu_mul1::expert_ffn: y is not [{n}][T][{h}]");
    let (kern, tab, tab_d) = (im.kern(path), tables(e0.gate.bitrate), tables(e0.down.bitrate));
    // scratch per expert: xp_g, xp_u [T][h/16][32], raw_g, raw_u [T][i], xp_d [T][i/16][32],
    // raw_d [T][h]; expert j's regions at j x the per-expert length of each
    let (nxh, nxi) = (tokens * (h / 16) * 32, tokens * (i / 16) * 32);
    let (ni, nh) = (tokens * i, tokens * h);
    with_scratch(n * (2 * nxh + 2 * ni + nxi + nh), |s| {
        let o = |v: &mut [f32]| Out(v.as_mut_ptr(), v.len());
        let (xp_g, s) = s.split_at_mut(n * nxh);
        let (xp_u, s) = s.split_at_mut(n * nxh);
        let (raw_g, s) = s.split_at_mut(n * ni);
        let (raw_u, s) = s.split_at_mut(n * ni);
        let (xp_d, raw_d) = s.split_at_mut(n * nxi);
        // the sub-buffer of expert j inside one of the regions above
        let sub = |v: Out, len: usize, j: usize| Out(unsafe { v.0.add(j * len) }, len);
        let (oxg, oxu, og, ou, oxd, od, yo) = (o(xp_g), o(xp_u), o(raw_g), o(raw_u), o(xp_d), o(raw_d), o(ys));
        let (ug, ud) = (units_of(&e0.gate), units_of(&e0.down));
        let k = threads.clamp(1, n * (2 * ug).min(ud));
        // the gate/up phase in block order: block b's 8 gate tiles, then its 8 up tiles; expert
        // j's columns follow expert j - 1's (a multiple of 16 tiles: no chunk crosses experts)
        let (c1, c2) = (2 * (i / 16), h / 16);
        let (plan1, plan2) = (chunks(n * c1, k), chunks(n * c2, k));
        let (blk1, blk2) = (Blocks::new(n * (i / HAD), 2 * BLOCK_TILES), Blocks::new(n * (h / HAD), BLOCK_TILES));
        let (prep, acts) = (Phase::new(n * 2 * (h / HAD)), AtomicUsize::new(0));
        let (cols1, cols2) = (AtomicUsize::new(0), AtomicUsize::new(0));
        pool::run(k, &|_| {
            prep.work(|jj| {
                let (j, b) = (jj / (2 * (h / HAD)), jj % (2 * (h / HAD)));
                if b < h / HAD {
                    prep_block(x, &es[j].gate, b, tokens, sub(oxg, nxh, j));
                } else {
                    prep_block(x, &es[j].up, b - h / HAD, tokens, sub(oxu, nxh, j));
                }
            });
            prep.wait();
            while let Some((q0, ntc)) = take(&cols1, &plan1) {
                let (j, p0) = (q0 / c1, q0 % c1);
                // SAFETY: every write to xp_g / xp_u is complete (`prep.wait`); read only from here on
                let (xg, xu) = unsafe {
                    (std::slice::from_raw_parts(oxg.0.add(j * nxh) as *const f32, nxh), std::slice::from_raw_parts(oxu.0.add(j * nxh) as *const f32, nxh))
                };
                let (b, r) = (p0 / (2 * BLOCK_TILES), p0 % (2 * BLOCK_TILES));
                let (gj, uj) = (sub(og, ni, j), sub(ou, ni, j));
                if r < BLOCK_TILES {
                    unit_new(kern, &es[j].gate, tab, xg, tokens, b * BLOCK_TILES + r, ntc, gj);
                } else {
                    unit_new(kern, &es[j].up, tab, xu, tokens, b * BLOCK_TILES + r - BLOCK_TILES, ntc, uj);
                }
                if blk1.complete(j * (i / HAD) + b, ntc) {
                    act_block(&es[j], b, tokens, gj, uj, sub(oxd, nxi, j), act);
                    acts.fetch_add(1, Ordering::Release);
                }
            }
            // down needs every block of its input: wait for the work, not for the workers
            wait_for(&acts, n * (i / HAD));
            while let Some((q0, ntc)) = take(&cols2, &plan2) {
                let (j, nb0) = (q0 / c2, q0 % c2);
                // SAFETY: every block of xp_d is complete (acquire above); read only from here on
                let xd = unsafe { std::slice::from_raw_parts(oxd.0.add(j * nxi) as *const f32, nxi) };
                let dj = sub(od, nh, j);
                unit_new(kern, &es[j].down, tab_d, xd, tokens, nb0, ntc, dj);
                let b = nb0 / BLOCK_TILES;
                if blk2.complete(j * (h / HAD) + b, ntc) {
                    had_out_block(dj, &es[j].down, b, tokens, sub(yo, nh, j));
                }
            }
        });
    });
}

/// The persistent worker pool of the CPU lane (#183 C1 follow-up). Starting and joining 8 scoped
/// threads cost 171 us per call on the 285K (#183, empty work); here `run(n, job)` publishes the
/// job to up to `n - 1` parked or spinning workers and runs `job(0)` on the caller. A worker
/// spins on one dispatch word for [`SPINS`] pause iterations after its last job, then parks; a
/// dispatch wakes the parked participants. When `job(0)` returns the dispatch is closed: a worker
/// that had not started by then skips it, and `run` waits only for the workers that entered, so
/// one descheduled thread does not stall the call. Jobs must therefore take their work from
/// shared counters and wait only for work to complete, never for a number of workers.
/// Written after exllamav3 `moe_mul1.cpp:2050-2170` (`Pool`: generation and participant count in
/// one word, master = worker 0, a surplus worker neither runs nor acks; turboderp-org/exllamav3 @
/// `151539c7`, MIT) and sybil `ft_core.h:411-455` (spin, then sleep). No pinning: the OS places the
/// threads and the shared counters balance P- and E-cores. One run at a time: a call while the
/// pool is busy (another thread, or a nested call) runs on scoped threads instead.
mod pool {
    use std::cell::UnsafeCell;
    use std::panic::{catch_unwind, resume_unwind, AssertUnwindSafe};
    use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
    use std::sync::{Mutex, OnceLock};
    use std::thread::Thread;

    pub(super) type Job<'a> = dyn Fn(usize) + Sync + 'a;

    /// pause iterations an idle worker (or a waiter) spins before it parks (or yields)
    pub(super) const SPINS: u32 = 1 << 16;
    const NW_BITS: u32 = 16;
    /// `gate`: generation in the high 32 bits, bit 31 closed, entered workers below
    const CLOSED: u64 = 1 << 31;

    struct Shared {
        /// `generation << NW_BITS | participants`, one word, so a worker never pairs one
        /// dispatch's generation with another's participant count
        dispatch: AtomicU64,
        /// `generation << 32 | CLOSED? | entered`: a worker enters only an open gate of the
        /// generation it saw (CAS), so it is either waited for or never touches the job
        gate: AtomicU64,
        job: UnsafeCell<Option<*const Job<'static>>>,
        done: AtomicUsize,
        panicked: AtomicBool,
    }

    // SAFETY: `job` is written only by the dispatcher, which holds `Pool::inner`, after every
    // worker that entered the previous dispatch acked and before the release store of `dispatch`;
    // a worker reads it only after entering the open gate of that generation, and before it acks.
    unsafe impl Sync for Shared {}
    unsafe impl Send for Shared {}

    struct Worker {
        thread: Thread,
        sleeping: &'static AtomicBool,
    }

    struct Inner {
        workers: Vec<Worker>,
        generation: u64,
    }

    struct Pool {
        shared: &'static Shared,
        inner: Mutex<Inner>,
    }

    fn pool() -> &'static Pool {
        static POOL: OnceLock<Pool> = OnceLock::new();
        POOL.get_or_init(|| Pool {
            shared: Box::leak(Box::new(Shared {
                dispatch: AtomicU64::new(0),
                gate: AtomicU64::new(0),
                job: UnsafeCell::new(None),
                done: AtomicUsize::new(0),
                panicked: AtomicBool::new(false),
            })),
            inner: Mutex::new(Inner { workers: Vec::new(), generation: 0 }),
        })
    }

    /// enter the open gate of `generation`; false if it is closed or another generation's
    fn enter(sh: &Shared, generation: u64) -> bool {
        let mut v = sh.gate.load(Ordering::Acquire);
        loop {
            if v >> 32 != generation & 0xffff_ffff || v & CLOSED != 0 {
                return false;
            }
            match sh.gate.compare_exchange_weak(v, v + 1, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return true,
                Err(now) => v = now,
            }
        }
    }

    fn worker_loop(sh: &'static Shared, sleeping: &'static AtomicBool, idx: usize, mut seen: u64) {
        let mut spins = 0u32;
        loop {
            let d = sh.dispatch.load(Ordering::Acquire);
            if d >> NW_BITS == seen {
                if spins < SPINS {
                    spins += 1;
                    std::hint::spin_loop();
                    continue;
                }
                // SeqCst pairs with the dispatcher's store of `dispatch` and load of `sleeping`:
                // either this load sees the new generation or the dispatcher sees `sleeping` and
                // unparks (an unpark before the park is kept as the thread's token)
                sleeping.store(true, Ordering::SeqCst);
                if sh.dispatch.load(Ordering::SeqCst) >> NW_BITS == seen {
                    std::thread::park();
                }
                sleeping.store(false, Ordering::SeqCst);
                continue;
            }
            spins = 0;
            seen = d >> NW_BITS;
            let nw = (d & ((1 << NW_BITS) - 1)) as usize;
            if idx < nw && enter(sh, seen) {
                // SAFETY: published before the dispatch this worker entered (struct comment); the
                // dispatcher keeps the job alive until this worker acks below
                let job = unsafe { (*sh.job.get()).expect("cpu_mul1 pool: dispatch without a job") };
                // SAFETY: as above
                if catch_unwind(AssertUnwindSafe(|| unsafe { (*job)(idx) })).is_err() {
                    sh.panicked.store(true, Ordering::Relaxed);
                }
                sh.done.fetch_add(1, Ordering::Release);
            }
        }
    }

    /// scoped threads per call (the pool is busy)
    fn scoped(n: usize, job: &Job<'_>) {
        std::thread::scope(|s| {
            for w in 1..n {
                s.spawn(move || job(w));
            }
            job(0);
        });
    }

    /// Run `job(0)` on the caller and `job(w)` on each worker w in 1..n that starts before
    /// `job(0)` returns; return when all of those are done. A panic in any of them is raised
    /// here after all of them finished.
    pub(super) fn run(n: usize, job: &Job<'_>) {
        if n <= 1 {
            job(0);
            return;
        }
        assert!(n < 1 << NW_BITS, "cpu_mul1 pool: {n} workers");
        let p = pool();
        let Ok(mut inner) = p.inner.try_lock() else {
            return scoped(n, job);
        };
        let sh = p.shared;
        while inner.workers.len() < n - 1 {
            let idx = inner.workers.len() + 1;
            let sleeping: &'static AtomicBool = Box::leak(Box::new(AtomicBool::new(false)));
            let seen = inner.generation;
            let h = std::thread::Builder::new()
                .name(format!("cpu_mul1-{idx}"))
                .spawn(move || worker_loop(sh, sleeping, idx, seen))
                .expect("cpu_mul1 pool: cannot start a worker thread");
            inner.workers.push(Worker { thread: h.thread().clone(), sleeping });
        }
        // SAFETY: only the lifetime is erased; this function clears the pointer and returns only
        // after every worker that entered acked, so no worker calls the job after the borrow ends
        let ptr = unsafe { std::mem::transmute::<*const Job<'_>, *const Job<'static>>(job as *const Job<'_>) };
        inner.generation += 1;
        let generation = inner.generation;
        // SAFETY: no worker reads `job` now (all entered workers acked, the gate of the next
        // generation is not open yet)
        unsafe { *sh.job.get() = Some(ptr) };
        sh.done.store(0, Ordering::Relaxed);
        sh.gate.store((generation & 0xffff_ffff) << 32, Ordering::Release);
        sh.dispatch.store(generation << NW_BITS | n as u64, Ordering::SeqCst);
        for w in &inner.workers[..n - 1] {
            if w.sleeping.load(Ordering::SeqCst) {
                w.thread.unpark();
            }
        }
        let mine = catch_unwind(AssertUnwindSafe(|| job(0)));
        let entered = (sh.gate.fetch_or(CLOSED, Ordering::AcqRel) & (CLOSED - 1)) as usize;
        let mut spins = 0u32;
        while sh.done.load(Ordering::Acquire) < entered {
            if spins < SPINS {
                spins += 1;
                std::hint::spin_loop();
            } else {
                std::thread::yield_now();
            }
        }
        // SAFETY: every worker that entered acked; the gate is closed to the others
        unsafe { *sh.job.get() = None };
        let theirs = sh.panicked.swap(false, Ordering::Relaxed);
        drop(inner);
        if let Err(e) = mine {
            resume_unwind(e);
        }
        assert!(!theirs, "cpu_mul1 pool: a worker panicked");
    }

    /// Test hook: park every idle worker now (no spinning left), so a reference arm timed next
    /// does not share the cores with spinning workers (`cpu_mul1_bench`).
    #[cfg(test)]
    pub(super) fn park_idle() {
        let p = pool();
        let Ok(inner) = p.inner.lock() else { return };
        // no dispatch can start while the lock is held; wait (at most 50 ms) for every park
        let t0 = std::time::Instant::now();
        while inner.workers.iter().any(|w| !w.sleeping.load(Ordering::SeqCst)) && t0.elapsed().as_millis() < 50 {
            std::thread::yield_now();
        }
    }

    /// Test hook: how many workers the pool has started.
    #[cfg(test)]
    pub(super) fn workers() -> usize {
        pool().inner.lock().map(|i| i.workers.len()).unwrap_or(0)
    }
}

/// The #180 and #183 kernels as they were, the reference arms of the bit-identity tests and of
/// `cpu_mul1_bench` (test only).
#[cfg(test)]
mod reference {
    use super::*;
    use std::sync::Barrier;

    /// columns `n * w / k .. n * (w + 1) / k` of worker `w`
    #[inline]
    fn split(n: usize, k: usize, w: usize) -> (usize, usize) {
        (n * w / k, n * (w + 1) / k)
    }

    /// Worker `w` of `k` runs `f(u)` on its units of `0 .. units`: taken one at a time from `next`
    /// (V2), or under `V1` the static block of `split`.
    fn for_units(im: Impl, next: &AtomicUsize, units: usize, k: usize, w: usize, mut f: impl FnMut(usize)) {
        if im == Impl::V1 {
            let (u0, u1) = split(units, k, w);
            (u0..u1).for_each(f);
            return;
        }
        loop {
            let u = next.fetch_add(1, Ordering::Relaxed);
            if u >= units {
                break;
            }
            f(u);
        }
    }

    /// One work unit (tile columns `UNIT_TILES u ..`) for every token of `xp`.
    fn unit_into(avx2: bool, im: Impl, m: &Mul1Matrix, tab: &Tables, xp: &[f32], tokens: usize, u: usize, raw: Out) {
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
                    if im == Impl::V1 {
                        match nt {
                            1 => avx2::unit::<1, true>(m, tab, xp, t0, nb0, ntc, raw),
                            2 => avx2::unit::<2, true>(m, tab, xp, t0, nb0, ntc, raw),
                            3 => avx2::unit::<3, true>(m, tab, xp, t0, nb0, ntc, raw),
                            _ => avx2::unit::<4, true>(m, tab, xp, t0, nb0, ntc, raw),
                        }
                    } else {
                        match nt {
                            1 => avx2::unit::<1, false>(m, tab, xp, t0, nb0, ntc, raw),
                            2 => avx2::unit::<2, false>(m, tab, xp, t0, nb0, ntc, raw),
                            3 => avx2::unit::<3, false>(m, tab, xp, t0, nb0, ntc, raw),
                            _ => avx2::unit::<4, false>(m, tab, xp, t0, nb0, ntc, raw),
                        }
                    }
                }
                t0 += nt;
                continue;
            }
            let _ = (avx2, im);
            match nt {
                1 => unit_scalar::<1>(m, tab, xp, t0, nb0, ntc, raw),
                2 => unit_scalar::<2>(m, tab, xp, t0, nb0, ntc, raw),
                3 => unit_scalar::<3>(m, tab, xp, t0, nb0, ntc, raw),
                _ => unit_scalar::<4>(m, tab, xp, t0, nb0, ntc, raw),
            }
            t0 += nt;
        }
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

    pub(super) fn gemv(im: Impl, m: &Mul1Matrix, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
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
        let next = AtomicUsize::new(0);
        scoped(k, &|w| for_units(im, &next, units, k, w, |u| unit_into(avx2, im, m, &tab, &xp, tokens, u, out)));
        had_out(&raw, m, y);
    }

    pub(super) fn expert_ffn(im: Impl, e: &Mul1Expert, x: &[f32], y: &mut [f32], threads: usize, path: Path) {
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
        let (next1, next2) = (AtomicUsize::new(0), AtomicUsize::new(0));
        scoped(k, &|w| {
            for_units(im, &next1, 2 * ug, k, w, |u| {
                if u < ug {
                    unit_into(avx2, im, &e.gate, &tab, &xp_g, tokens, u, og);
                } else {
                    unit_into(avx2, im, &e.up, &tab, &xp_u, tokens, u - ug, ou);
                }
            });
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
            for_units(im, &next2, ud, k, w, |u| unit_into(avx2, im, &e.down, &tab_d, xd, tokens, u, od));
        });
        drop((raw_g, raw_u, xp_d));
        had_out(&raw_d, &e.down, y);
    }
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

    /// #183: the shuffle decoder and the counter scheduler give the bits of the first
    /// #180 kernel (`Impl::V1`): every lane decode of every bitrate (random tiles, wrap included),
    /// GEMV of the 4 quantizer experts and of synthetic K = 1, 1.5, 2.5, 5, 8 at T 1, 2, 3, 5, 8 x
    /// threads 1, 3, 8, 16, and the FFN of a full GLM expert at T 1 and 4 x threads 1, 8, 16, 24.
    #[test]
    fn cpu_mul1_decode_and_schedule_equal_v1_bits() {
        if !avx2_available() {
            eprintln!("cpu_mul1: no AVX2 on this CPU, the AVX2 path is not exercised");
            return;
        }
        let mut rng = Rng(0xD1CE);
        for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0] {
            let b = Bitrate::from_k(k).unwrap();
            let tab = Tables::new(b);
            for _ in 0..64 {
                let tile: Vec<u8> = (0..b.tile_bytes()).map(|_| rng.next() as u8).collect();
                for g in 0..32 {
                    let (mut new, mut old) = ([0f32; 8], [0f32; 8]);
                    // SAFETY: AVX2 checked; the tile holds 4 * n32 bytes
                    unsafe {
                        let kc = avx2::consts();
                        std::arch::x86_64::_mm256_storeu_ps(new.as_mut_ptr(), avx2::decode8(tile.as_ptr(), tab.n32, &tab.groups[g], &kc));
                        std::arch::x86_64::_mm256_storeu_ps(old.as_mut_ptr(), avx2::decode8_v1(tile.as_ptr(), tab.n32, &tab.groups[g], &kc));
                    }
                    assert_eq!(bits(&new), bits(&old), "K = {k} lane {g}");
                }
            }
        }
        let mut all = quant_cases();
        for (i, k) in [1.0, 1.5, 2.5, 5.0, 8.0].into_iter().enumerate() {
            all.push(Case {
                name: format!("synth K = {k}"),
                source: format!("synth:{}", 60 + i),
                bitrate: Bitrate::from_k(k).unwrap(),
                hidden: 512,
                inter: 256,
                want: [String::new(), String::new(), String::new()],
            });
        }
        for c in &all {
            let rec = record(c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            for m in [e.gate, e.down] {
                for t in [1usize, 2, 3, 5, 8] {
                    let x = xs(t * m.k, &mut rng);
                    let mut want = vec![0f32; t * m.n];
                    gemv_with(Impl::V1, &m, &x, &mut want, 1, Path::Avx2);
                    for th in [1usize, 3, 8, 16] {
                        let mut y = vec![0f32; t * m.n];
                        gemv_with(Impl::NEW, &m, &x, &mut y, th, Path::Avx2);
                        assert_eq!(bits(&y), bits(&want), "{} [{}, {}] T {t} threads {th}", c.name, m.k, m.n);
                    }
                }
            }
        }
        let c = &glm_cases()[0];
        let rec = record(c);
        let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
        for t in [1usize, 4] {
            let x = xs(t * GLM_HIDDEN, &mut rng);
            let mut want = vec![0f32; x.len()];
            expert_ffn_with(Impl::V1, &e, &x, &mut want, 8, Path::Avx2);
            for th in [1usize, 8, 16, 24] {
                let mut y = vec![0f32; x.len()];
                expert_ffn_with(Impl::NEW, &e, &x, &mut y, th, Path::Avx2);
                assert_eq!(bits(&y), bits(&want), "GLM FFN T {t} threads {th}");
            }
        }
    }

    /// #183 C1: the production path gives the bits of the #183 kernel (`Impl::V2`): the
    /// division-free lane decoder (`decode8_fast`, every lane of every bitrate on random tiles,
    /// wrapping lanes included) == `decode8`; `weights_fast` == `weights` for all 65,536 states;
    /// `weights_vnni` == `weights` and the AVX-VNNI K = 3 unit == `unit_fast` (on a CPU with
    /// AVX-VNNI); the K = 3 unrolled unit == the table-driven `unit_fast`; GEMV of the 4 quantizer experts
    /// and of synthetic K = 1, 1.5, 2.5, 5, 8 at T 1, 2, 3, 5, 8 x threads 1, 3, 8, 16, 24 and of
    /// the GLM K = 3 gate and down at T 1, 4; the FFN of a full GLM expert at T 1, 2, 4, 5 x
    /// threads 1, 2, 8, 16, 24; and the production path with the #183 unit (`NEW_NO_FMA`, an AVX2
    /// CPU without FMA) on the same GEMVs and FFNs at 8 threads.
    #[test]
    fn cpu_mul1_c1_kernel_equals_v2_bits() {
        if !avx2_available() || !std::arch::is_x86_feature_detected!("fma") {
            eprintln!("cpu_mul1: no AVX2 + FMA on this CPU, the C1 kernel is not exercised");
            return;
        }
        use std::arch::x86_64::*;
        // SAFETY (whole test): AVX2 and FMA checked above; every tile holds 4 * n32 bytes
        unsafe {
            let kc = avx2::consts();
            for s0 in (0..=u16::MAX as i32).step_by(8) {
                let st = _mm256_setr_epi32(s0, s0 + 1, s0 + 2, s0 + 3, s0 + 4, s0 + 5, s0 + 6, s0 + 7);
                let (mut a, mut b) = ([0f32; 8], [0f32; 8]);
                _mm256_storeu_ps(a.as_mut_ptr(), avx2::weights_fast(st, &kc));
                _mm256_storeu_ps(b.as_mut_ptr(), avx2::weights(st, &kc));
                assert_eq!(bits(&a), bits(&b), "states {s0:#06x}..");
                if vnni() {
                    _mm256_storeu_ps(a.as_mut_ptr(), avx2::weights_vnni(st, &kc));
                    assert_eq!(bits(&a), bits(&b), "states {s0:#06x}.. (VNNI)");
                }
            }
            let mut rng = Rng(0xC1);
            for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0] {
                let b = Bitrate::from_k(k).unwrap();
                let tab = Tables::new(b);
                for _ in 0..64 {
                    let tile: Vec<u8> = (0..b.tile_bytes()).map(|_| rng.next() as u8).collect();
                    for g in 0..32 {
                        let (mut new, mut old) = ([0f32; 8], [0f32; 8]);
                        _mm256_storeu_ps(new.as_mut_ptr(), avx2::decode8_fast(tile.as_ptr(), &tab.lanes[g], &kc));
                        _mm256_storeu_ps(old.as_mut_ptr(), avx2::decode8(tile.as_ptr(), tab.n32, &tab.groups[g], &kc));
                        assert_eq!(bits(&new), bits(&old), "K = {k} lane {g}");
                    }
                }
            }
        }
        let mut rng = Rng(0xC1C1);
        let c = &glm_cases()[0];
        let rec = record(c);
        let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
        for t in [1usize, 2, 3, 4] {
            let x = xs(t * GLM_HIDDEN, &mut rng);
            let xp = permute(&had_in(&x, &e.gate), GLM_HIDDEN);
            let (mut a, mut b) = (vec![0f32; t * GLM_INTER], vec![0f32; t * GLM_INTER]);
            let (oa, ob) = (Out(a.as_mut_ptr(), a.len()), Out(b.as_mut_ptr(), b.len()));
            for (nb0, ntc) in chunks(GLM_INTER / 16, 8) {
                // SAFETY: AVX2 + FMA checked above; e.gate is K = 3
                unsafe {
                    match t {
                        1 => (avx2::unit_k3::<1>(&e.gate, &xp, 0, nb0, ntc, oa), avx2::unit_fast::<1>(&e.gate, tables(K3), &xp, 0, nb0, ntc, ob)),
                        2 => (avx2::unit_k3::<2>(&e.gate, &xp, 0, nb0, ntc, oa), avx2::unit_fast::<2>(&e.gate, tables(K3), &xp, 0, nb0, ntc, ob)),
                        3 => (avx2::unit_k3::<3>(&e.gate, &xp, 0, nb0, ntc, oa), avx2::unit_fast::<3>(&e.gate, tables(K3), &xp, 0, nb0, ntc, ob)),
                        _ => (avx2::unit_k3::<4>(&e.gate, &xp, 0, nb0, ntc, oa), avx2::unit_fast::<4>(&e.gate, tables(K3), &xp, 0, nb0, ntc, ob)),
                    };
                }
            }
            assert_eq!(bits(&a), bits(&b), "unit_k3 != unit_fast, T {t}");
            if vnni() {
                a.fill(f32::NAN);
                for (nb0, ntc) in chunks(GLM_INTER / 16, 8) {
                    // SAFETY: AVX2, FMA and AVX-VNNI checked; e.gate is K = 3
                    unsafe {
                        match t {
                            1 => avx2::unit_k3_vnni::<1>(&e.gate, &xp, 0, nb0, ntc, oa),
                            2 => avx2::unit_k3_vnni::<2>(&e.gate, &xp, 0, nb0, ntc, oa),
                            3 => avx2::unit_k3_vnni::<3>(&e.gate, &xp, 0, nb0, ntc, oa),
                            _ => avx2::unit_k3_vnni::<4>(&e.gate, &xp, 0, nb0, ntc, oa),
                        }
                    }
                }
                assert_eq!(bits(&a), bits(&b), "unit_k3_vnni != unit_fast, T {t}");
            }
        }
        let mut all = quant_cases();
        for (i, k) in [1.0, 1.5, 2.5, 5.0, 8.0].into_iter().enumerate() {
            all.push(Case {
                name: format!("synth K = {k}"),
                source: format!("synth:{}", 70 + i),
                bitrate: Bitrate::from_k(k).unwrap(),
                hidden: 512,
                inter: 256,
                want: [String::new(), String::new(), String::new()],
            });
        }
        for c in &all {
            let rec = record(c);
            let e = Mul1Expert::from_record(&rec, c.hidden, c.inter, c.bitrate).unwrap();
            for m in [e.gate, e.down] {
                for t in [1usize, 2, 3, 5, 8] {
                    let x = xs(t * m.k, &mut rng);
                    let mut want = vec![0f32; t * m.n];
                    gemv_with(Impl::V2, &m, &x, &mut want, 3, Path::Avx2);
                    for th in [1usize, 3, 8, 16, 24] {
                        let mut y = vec![0f32; t * m.n];
                        gemv_with(Impl::NEW, &m, &x, &mut y, th, Path::Avx2);
                        assert_eq!(bits(&y), bits(&want), "{} [{}, {}] T {t} threads {th}", c.name, m.k, m.n);
                    }
                    let mut y = vec![0f32; t * m.n];
                    gemv_with(Impl::NEW_NO_FMA, &m, &x, &mut y, 8, Path::Avx2);
                    assert_eq!(bits(&y), bits(&want), "{} [{}, {}] T {t} without FMA", c.name, m.k, m.n);
                }
            }
        }
        for m in [e.gate, e.down] {
            for t in [1usize, 4] {
                let x = xs(t * m.k, &mut rng);
                let (mut want, mut y) = (vec![0f32; t * m.n], vec![0f32; t * m.n]);
                gemv_with(Impl::V2, &m, &x, &mut want, 8, Path::Avx2);
                gemv_with(Impl::NEW, &m, &x, &mut y, 8, Path::Avx2);
                assert_eq!(bits(&y), bits(&want), "GLM [{}, {}] T {t}", m.k, m.n);
            }
        }
        for t in [1usize, 2, 4, 5] {
            let x = xs(t * GLM_HIDDEN, &mut rng);
            let mut want = vec![0f32; x.len()];
            expert_ffn_with(Impl::V2, &e, &x, &mut want, 8, Path::Avx2);
            for th in [1usize, 2, 8, 16, 24] {
                let mut y = vec![0f32; x.len()];
                expert_ffn_with(Impl::NEW, &e, &x, &mut y, th, Path::Avx2);
                assert_eq!(bits(&y), bits(&want), "GLM FFN T {t} threads {th}");
            }
            let mut y = vec![0f32; x.len()];
            expert_ffn_with(Impl::NEW_NO_FMA, &e, &x, &mut y, 8, Path::Avx2);
            assert_eq!(bits(&y), bits(&want), "GLM FFN T {t} without FMA");
        }
    }

    /// #188 CPU lane: `experts_ffn` (all experts of a call in one pool run) gives every expert
    /// the bits of `expert_ffn` on that expert alone: 1, 2, 3, 5 and 8 GLM experts (K = 3, T 1
    /// and 2) and 3 small K = 2.5 experts (T 1, 3), threads 1, 2, 8, 16, 24, the outputs NaN
    /// before the call; with the clamped SwiGLU as `act` (8 threads), the bits of the staged
    /// `gemv(down, act(gemv(gate, x), gemv(up, x)))`.
    #[test]
    fn cpu_mul1_experts_ffn_is_expert_ffn_per_expert_bits() {
        let mut rng = Rng(0x188);
        let mk = |j: usize, k: f64, hidden: usize, inter: usize| Case {
            name: format!("lane e{j}"),
            source: format!("synth:{}", 0x1880 + 9 * j),
            bitrate: Bitrate::from_k(k).unwrap(),
            hidden,
            inter,
            want: [String::new(), String::new(), String::new()],
        };
        let glm: Vec<Vec<u8>> = (0..8).map(|j| record(&mk(j, 3.0, GLM_HIDDEN, GLM_INTER))).collect();
        let small: Vec<Vec<u8>> = (0..3).map(|j| record(&mk(10 + j, 2.5, 512, 256))).collect();
        let check = |recs: &[Vec<u8>], c: &Case, n: usize, tokens: &[usize], rng: &mut Rng| {
            let es: Vec<Mul1Expert> = recs[..n].iter().map(|r| Mul1Expert::from_record(r, c.hidden, c.inter, c.bitrate).unwrap()).collect();
            for &t in tokens {
                let x = xs(t * c.hidden, rng);
                let want: Vec<Vec<f32>> = es
                    .iter()
                    .map(|e| {
                        let mut y = vec![0f32; x.len()];
                        expert_ffn(e, &x, &mut y, 8, Path::Auto);
                        y
                    })
                    .collect();
                for th in [1usize, 2, 8, 16, 24] {
                    let mut ys = vec![f32::NAN; n * x.len()];
                    experts_ffn(&es, &x, &mut ys, &silu_mul, th, Path::Auto);
                    for (j, w) in want.iter().enumerate() {
                        assert_eq!(bits(&ys[j * x.len()..(j + 1) * x.len()]), bits(w), "{n} experts [{}, {}] T {t} threads {th}: expert {j}", c.hidden, c.inter);
                    }
                }
                // another activation (GLM's clamped SwiGLU, limit 10): the staged chain's bits
                let clamp = |g: f32, u: f32| {
                    let g = if g > 10.0 { 10.0 } else { g };
                    let u = if u > 10.0 { 10.0 } else if u < -10.0 { -10.0 } else { u };
                    (g / (1.0 + (-g).exp())) * u
                };
                let mut ys = vec![f32::NAN; n * x.len()];
                experts_ffn(&es, &x, &mut ys, &clamp, 8, Path::Auto);
                for (j, e) in es.iter().enumerate() {
                    let (mut g, mut u) = (vec![0f32; t * c.inter], vec![0f32; t * c.inter]);
                    gemv(&e.gate, &x, &mut g, 8, Path::Auto);
                    gemv(&e.up, &x, &mut u, 8, Path::Auto);
                    let hv: Vec<f32> = g.iter().zip(&u).map(|(&g, &u)| clamp(g, u)).collect();
                    let mut w = vec![0f32; x.len()];
                    gemv(&e.down, &hv, &mut w, 8, Path::Auto);
                    assert_eq!(bits(&ys[j * x.len()..(j + 1) * x.len()]), bits(&w), "{n} experts [{}, {}] T {t} clamped act: expert {j}", c.hidden, c.inter);
                }
            }
        };
        let cg = mk(0, 3.0, GLM_HIDDEN, GLM_INTER);
        for n in [1usize, 2, 3, 5, 8] {
            check(&glm, &cg, n, &[1, 2], &mut rng);
        }
        check(&small, &mk(10, 2.5, 512, 256), 3, &[1, 3], &mut rng);
    }

    /// #183 C1: the bit-built `f16_to_f32` is the former formula for all 65,536 inputs, and the
    /// guided chunk plan covers every tile column once, in order, in aligned chunks of
    /// `UNIT_TILES` / 2 / 1
    /// that never cross a 128-column block, ending in one-column chunks when k > 1.
    #[test]
    fn cpu_mul1_scales_and_chunk_plan() {
        for h in 0..=u16::MAX {
            let sign = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
            let (e, m) = (((h >> 10) & 0x1f) as i32, (h & 0x3ff) as f32);
            let want = match e {
                0 => sign * m * (2f32).powi(-24),
                31 if m == 0.0 => sign * f32::INFINITY,
                31 => f32::NAN,
                _ => sign * (1024.0 + m) * (2f32).powi(e - 25),
            };
            assert_eq!(f16_to_f32(h).to_bits(), want.to_bits(), "fp16 {h:#06x}");
        }
        for tiles in [8usize, 16, 64, 128, 256, 512] {
            for k in (1..=24).chain([32, 64]) {
                let plan = chunks(tiles, k);
                let mut at = 0;
                for &(t0, n) in &plan {
                    assert_eq!(t0, at, "tiles {tiles} k {k}: gap or overlap at {t0}");
                    assert!(matches!(n, 1 | 2 | UNIT_TILES) && t0 % n == 0 && t0 / BLOCK_TILES == (t0 + n - 1) / BLOCK_TILES, "tiles {tiles} k {k}: chunk ({t0}, {n})");
                    at += n;
                }
                assert_eq!(at, tiles, "tiles {tiles} k {k}: not covered");
                if k > 1 {
                    assert_eq!(plan.last().unwrap().1, 1, "tiles {tiles} k {k}: the last chunk is not one column");
                } else {
                    assert!(plan.iter().all(|&(_, n)| n == UNIT_TILES.min(tiles)), "k 1 takes whole units");
                }
            }
        }
    }

    /// #183 C1: the pool. Every run does all of its work before it returns (item sums), job(0)
    /// runs exactly once and no worker index runs twice or at or past n; a phase's `wait` returns
    /// only after all of its items (timed items); two threads calling at
    /// once (one falls back to scoped threads) and a nested call complete; a panicking worker is
    /// raised in the caller and the pool keeps working.
    #[test]
    fn cpu_mul1_pool_runs_all_work_once() {
        use std::sync::atomic::AtomicU64;
        fn check(n: usize, items: usize) {
            let seen: Vec<AtomicUsize> = (0..n + 1).map(|_| AtomicUsize::new(0)).collect();
            let ph = Phase::new(items);
            let sum = AtomicU64::new(0);
            pool::run(n, &|w| {
                seen[w.min(n)].fetch_add(1, Ordering::Relaxed);
                ph.work(|j| {
                    sum.fetch_add(j as u64 + 1, Ordering::Relaxed);
                });
                ph.wait();
            });
            assert_eq!(sum.load(Ordering::Relaxed), (items * (items + 1) / 2) as u64, "n {n}: work missing");
            assert_eq!(seen[0].load(Ordering::Relaxed), 1, "n {n}: job(0)");
            assert!(seen[1..n].iter().all(|s| s.load(Ordering::Relaxed) <= 1), "n {n}: a worker ran twice");
            assert_eq!(seen[n].load(Ordering::Relaxed), 0, "n {n}: a worker index >= n ran");
        }
        for rep in 0..300 {
            check([2, 3, 8, 16, 24][rep % 5], 1 + rep % 97);
        }
        // a later phase sees every item of the earlier one: items take 20..60 us, so a worker that
        // ran out of items while others are inside theirs would pass an early wait
        for n in [2usize, 4, 8, 16] {
            let a = Phase::new(4 * n);
            let flags: Vec<std::sync::atomic::AtomicBool> = (0..4 * n).map(|_| std::sync::atomic::AtomicBool::new(false)).collect();
            let early = AtomicUsize::new(0);
            pool::run(n, &|_| {
                a.work(|j| {
                    let t0 = std::time::Instant::now();
                    while t0.elapsed() < std::time::Duration::from_micros(20 + 10 * (j % 5) as u64) {
                        std::hint::spin_loop();
                    }
                    flags[j].store(true, Ordering::Relaxed);
                });
                a.wait();
                if flags.iter().any(|f| !f.load(Ordering::Relaxed)) {
                    early.fetch_add(1, Ordering::Relaxed);
                }
            });
            assert_eq!(early.load(Ordering::Relaxed), 0, "n {n}: a worker passed Phase::wait before the phase was complete");
        }
        assert!(pool::workers() >= 23, "the pool did not grow to 24 workers");
        std::thread::scope(|s| {
            for _ in 0..2 {
                s.spawn(|| {
                    for rep in 0..100 {
                        check(1 + rep % 9, 50);
                    }
                });
            }
        });
        let outer = AtomicUsize::new(0);
        pool::run(4, &|w| {
            if w == 0 {
                check(4, 64);
            }
            outer.fetch_add(1, Ordering::Relaxed);
        });
        assert!((1..=4).contains(&outer.load(Ordering::Relaxed)));
        let started = std::sync::atomic::AtomicBool::new(false);
        let r = std::panic::catch_unwind(|| {
            pool::run(8, &|w| {
                // job(0) waits (at most 2 s) until a worker entered; every other worker panics
                if w == 0 {
                    let t0 = std::time::Instant::now();
                    while !started.load(Ordering::Acquire) && t0.elapsed().as_secs() < 2 {
                        std::thread::yield_now();
                    }
                } else {
                    started.store(true, Ordering::Release);
                    panic!("worker {w} fails on purpose");
                }
            })
        });
        assert!(r.is_err(), "a worker's panic was not raised");
        check(8, 100);
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
    /// bytes / wall time. Two arms alternating call by call (#183 C1): `old` = `Impl::V2` (the
    /// #183 kernel as at `bd52328`) on even calls, timed after the pool's idle workers parked
    /// (`pool::park_idle`, so no spinning worker shares its cores), `new` = `Impl::NEW` on odd
    /// calls, which therefore starts from parked workers every time; median of 24 calls per arm.
    /// Then `new warm`: 24 `NEW` calls back to back (workers spinning between calls, as between
    /// the MoE layers of a decode step), median.
    /// `cargo test --release -p crow-nest-engine cpu_mul1_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cpu_mul1_bench() {
        super::POISON_SCRATCH.with(|p| p.set(false));
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
                    expert_ffn_with(Impl::V2, e, &x, &mut y, th, path);
                    expert_ffn_with(Impl::NEW, e, &x, &mut y, th, path);
                }
                let mut ms = [Vec::new(), Vec::new(), Vec::new()];
                for rep in 0..48 {
                    let im = if rep % 2 == 0 { Impl::V2 } else { Impl::NEW };
                    if im == Impl::V2 {
                        pool::park_idle();
                    }
                    let t0 = std::time::Instant::now();
                    expert_ffn_with(im, &experts[(rep / 2) % experts.len()], &x, &mut y, th, path);
                    ms[rep % 2].push(t0.elapsed().as_secs_f64() * 1e3);
                }
                for rep in 0..24 {
                    let t0 = std::time::Instant::now();
                    expert_ffn_with(Impl::NEW, &experts[rep % experts.len()], &x, &mut y, th, path);
                    ms[2].push(t0.elapsed().as_secs_f64() * 1e3);
                }
                for (arm, ms) in ["old     ", "new     ", "new warm"].iter().zip(ms.iter_mut()) {
                    ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let med = ms[ms.len() / 2];
                    eprintln!(
                        "  T {t} threads {th:2} {arm}: median {med:.3} ms per expert ({:.2} GB/s of record bytes), min {:.3} max {:.3} ms, n {}",
                        bytes as f64 / (med * 1e-3) / 1e9,
                        ms[0],
                        ms[ms.len() - 1],
                        ms.len()
                    );
                }
            }
        }
    }

    /// Micro-benchmark, not a gate (the split planner's CPU cost, `glm5_tiers::SplitCost`): the
    /// #188 lane's call, `experts_ffn` over n = 1..8 distinct GLM-shaped K = 3 records in one pool
    /// run, T 1, `swiglu`-free `silu_mul`, records rotated over 32 distinct copies (303 MB, far
    /// beyond the L3) so every call streams its records from DRAM; threads 8, 12, 16, 20, 24.
    /// Median of 32 calls per point after 4 warm-up calls, workers warm (spinning between calls, as
    /// between the MoE layers of a decode step). Prints ms per call, ms per expert and the least
    /// squares line `a + b n` per thread count (a = the run's fixed cost, b = ms per expert).
    /// `cargo test --release --lib cpu_mul1_lane_bench -- --ignored --nocapture`
    #[test]
    #[ignore]
    fn cpu_mul1_lane_bench() {
        super::POISON_SCRATCH.with(|p| p.set(false));
        let c = &glm_cases()[0];
        let base = record(c);
        let tb = 3 * Mul1Matrix::trellis_bytes(c.hidden, c.inter, c.bitrate);
        let recs: Vec<Vec<u8>> = (0..32u32)
            .map(|i| {
                let mut r = base.clone();
                r[..tb].iter_mut().for_each(|b| *b = b.rotate_left(i % 8) ^ ((i / 8) as u8).wrapping_mul(37));
                r
            })
            .collect();
        let experts: Vec<Mul1Expert> = recs.iter().map(|r| Mul1Expert::from_record(r, c.hidden, c.inter, c.bitrate).unwrap()).collect();
        let mut rng = Rng(0x5917);
        let x = xs(GLM_HIDDEN, &mut rng);
        let mut ys = vec![0f32; 8 * GLM_HIDDEN];
        eprintln!("cpu_mul1 lane bench: {:?}, {} threads available, {} B per record", kern(Path::Auto), std::thread::available_parallelism().map(|n| n.get()).unwrap_or(0), recs[0].len());
        for &th in &[8usize, 12, 16, 20, 24] {
            let mut pts = Vec::new();
            for n in 1..=8usize {
                let mut ms = Vec::new();
                let mut at = 0usize;
                for rep in 0..36 {
                    let es: Vec<Mul1Expert> = (0..n).map(|j| experts[(at + j) % experts.len()]).collect();
                    at += n;
                    let t0 = std::time::Instant::now();
                    experts_ffn(&es, &x, &mut ys[..n * GLM_HIDDEN], &silu_mul, th, Path::Auto);
                    if rep >= 4 {
                        ms.push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
                ms.sort_by(|a, b| a.total_cmp(b));
                pts.push((n as f64, ms[ms.len() / 2]));
            }
            let m = pts.len() as f64;
            let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
            let (sxx, sxy) = (pts.iter().map(|p| p.0 * p.0).sum::<f64>(), pts.iter().map(|p| p.0 * p.1).sum::<f64>());
            let b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
            let a = (sy - b * sx) / m;
            let row: Vec<String> = pts.iter().map(|&(n, t)| format!("{n:.0}:{t:.3}")).collect();
            eprintln!("  threads {th:2}: ms per call {}; fit {a:.3} + {b:.3} n ms; 8 experts {:.3} ms per expert", row.join(" "), pts[7].1 / 8.0);
        }
    }
}
