//! #181: the CNQ expert codec "MUL1 trellis" for GLM-5.3-Flash's routed experts.
//!
//! Encoding is exllamav3's (MIT, `turboderp-org/exllamav3` v1.6.0, commit `151539c7`): its
//! `quantize_exl3` (QTIP-style LDLQ over a tail-biting trellis, `modules/quant/exl3_lib/quantize.py`)
//! with the `mul1` codebook turns one linear `[in, out]` into three tensors: `trellis` int16
//! `[in/16, out/16, 16 K]`, `suh` fp16 `[in]`, `svh` fp16 `[out]`. This module does not quantize;
//! it stores what exllamav3 wrote and decodes it, byte for byte as exllamav3's own `reconstruct`
//! kernel does (`exllamav3_ext/quant/reconstruct.cu`, `exl3_dq.cuh`, `codebook.cuh`).
//!
//! Bitrate K: an integer 1..8, or 1.5 / 2.5 / 3.5 (alternating K-1/2 and K+1/2-bit steps, odd
//! positions carry the extra bit, `bits_k.cuh`, `exl3_dq.cuh` `dq8_half`). One 16x16 tile is a
//! circular bit stream of 256 K bits; position t's 16-bit trellis state is the 16 bits ending at
//! bit e(t) (e(t) = (t + 1) K for integer K), read MSB first over 32-bit words, each word being
//! `u16[2i] | u16[2i + 1] << 16` (the packer stores the words half-swapped, `pack.cu` `SWAP16`).
//! The state decodes through the mul1 codebook to one fp16 value:
//!
//! ```text
//! x   = state * 0x83DCD12D                     (u32, wrapping)
//! sum = 0x6400 + byte0 + byte1 + byte2 + byte3  (__dp4a(x, 0x01010101, 0x6400))
//! w   = hfma(half(sum), half(0x1eee), half(0xc931))   (one rounding, RNE)
//! ```
//!
//! Position 8 L + j of a tile (lane L, j = 0..7) lands at row `2 (L % 4) + [0, 1, 8, 9][j % 4]`,
//! column `L / 4 + 8 (j / 4)` (the tensor-core order, `quantize.py` `tensor_core_perm`); tile
//! (kb, nb) of a `[k, n]` matrix is trellis tile `kb * n/16 + nb`. The decoded matrix is the
//! ROTATED weight W_hat; the original-basis weight is `diag(suh) H W_hat H diag(svh)` with 128-wide
//! Hadamard blocks, which exllamav3 applies in fp32 cuBLAS (`get_weight_tensor`) or folds into the
//! activations at inference. That step is not part of this codec (the kernel is plan step 10).
//!
//! Record: one per expert, `[gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh]
//! [up.suh][up.svh][down.suh][down.svh] [zeros to the next multiple of EXPERT_ALIGN]`, all little
//! endian — the layout of sybil-solutions/glm53-flash-offload `scripts/pack_glm53_store.py`
//! (`df0b439`), so record i of a store starts at `i * size`, a multiple of 4096. gate and up are
//! `[hidden, inter]`, down is `[inter, hidden]` (exllamav3 stores `[in, out]`).
//!
//! The conversion writes these records from a quantizer store (`mul1_store.rs`, `--experts-mul1`,
//! plan step 12, #182). `converter dequant` (`dequant.rs`, #156) decodes them with `decode_tile`
//! and applies the Hadamard step above in f64, so the oracle reads the original-basis weight;
//! `reconstruct` is the whole-matrix form its tests hold against exllamav3.
#![cfg_attr(not(test), allow(dead_code))]

use crate::EXPERT_ALIGN;
use std::sync::OnceLock;

/// The index `dtype` of the routed-expert tensors in this codec — the name the engine reads
/// (`geo::ExpertCodec::from_dtype`, `nvme_source::glm5_record_from_index` on local main `54ef024`).
pub const DTYPE: &str = "mul1";

/// The bitrate of a trellis tensor: `bits` per step, plus one extra bit on every odd step when
/// `half` (K = bits + 0.5).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Bitrate {
    bits: u32,
    half: bool,
}

impl Bitrate {
    /// K = 1..8, or 1.5 / 2.5 / 3.5 — the set exllamav3's `bits_from_K` accepts for mul1.
    pub fn from_k(k: f64) -> Result<Bitrate, String> {
        let bits = k.trunc();
        let f = k - bits;
        let ok = (1.0..=8.0).contains(&bits) && (f == 0.0 || (f == 0.5 && bits <= 3.0));
        if !ok {
            return Err(format!("MUL1 trellis: unsupported bitrate {k} (integer 1..8, or 1.5 / 2.5 / 3.5)"));
        }
        Ok(Bitrate { bits: bits as u32, half: f == 0.5 })
    }

    pub fn k(self) -> f64 {
        self.bits as f64 + if self.half { 0.5 } else { 0.0 }
    }

    /// u16 words per 16x16 tile: `16 K` (exllamav3 `trellis_words`).
    pub fn words_per_tile(self) -> usize {
        16 * self.bits as usize + if self.half { 8 } else { 0 }
    }

    fn stream_bits(self) -> usize {
        self.words_per_tile() * 16
    }

    /// End bit (exclusive) of position `t`'s state in the tile's stream.
    fn end_bit(self, t: usize) -> usize {
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

pub(crate) fn f16_to_f64(h: u16) -> f64 {
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

/// fp16 bits of `x`, rounded once to nearest even (`x` finite).
fn f64_to_f16(x: f64) -> u16 {
    let sign: u16 = if x.is_sign_negative() { 0x8000 } else { 0 };
    let a = x.abs();
    if a == 0.0 {
        return sign;
    }
    if a < 2f64.powi(-14) {
        // subnormal: units of 2^-24; a round-up to 1024 is the smallest normal, bit pattern 0x400
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

/// exllamav3 `decode_3inst<2>` (mul1): fp16 bits of one 16-bit trellis state. The product
/// `half(sum) * k_inv + k_bias` is exact in f64 (11 x 11 mantissa bits plus an addend), so one
/// RNE conversion is the single rounding of `__hfma`.
pub fn mul1_decode(state: u16) -> u16 {
    let x = (state as u32).wrapping_mul(0x83DC_D12D);
    let sum = 0x6400u32 + (x & 0xff) + ((x >> 8) & 0xff) + ((x >> 16) & 0xff) + (x >> 24);
    let h = f16_to_f64(sum as u16);
    f64_to_f16(h * f16_to_f64(0x1eee) + f16_to_f64(0xc931))
}

fn mul1_lut() -> &'static [u16] {
    static LUT: OnceLock<Vec<u16>> = OnceLock::new();
    LUT.get_or_init(|| (0..=u16::MAX).map(mul1_decode).collect())
}

/// Row-major index inside a 16x16 tile of trellis position `p` (`quantize.py` `tensor_core_perm`).
fn tile_index(p: usize) -> usize {
    let (lane, j) = (p / 8, p % 8);
    let row = 2 * (lane % 4) + [0, 1, 8, 9][j % 4];
    let col = lane / 4 + 8 * (j / 4);
    row * 16 + col
}

/// Decode one packed tile (`words_per_tile` u16) into 256 fp16 bit patterns, row major.
pub fn decode_tile(words: &[u16], b: Bitrate) -> [u16; 256] {
    assert_eq!(words.len(), b.words_per_tile(), "tile length for K = {}", b.k());
    let w32: Vec<u32> = words.chunks_exact(2).map(|p| p[0] as u32 | (p[1] as u32) << 16).collect();
    let n32 = w32.len();
    let s = b.stream_bits();
    let lut = mul1_lut();
    let mut out = [0u16; 256];
    for p in 0..256 {
        // first bit of the 16-bit window, taken in the circular stream (tail-biting)
        let lo = (b.end_bit(p) + s - 16) % s;
        let (i, o) = (lo / 32, lo % 32);
        let pair = (w32[i] as u64) << 32 | w32[(i + 1) % n32] as u64;
        let state = ((pair >> (48 - o)) & 0xffff) as u16;
        out[tile_index(p)] = lut[state as usize];
    }
    out
}

/// exllamav3 `reconstruct`: the rotated weight W_hat `[k, n]` (fp16 bits, row major) of a trellis
/// `[k/16, n/16, words_per_tile]`.
pub fn reconstruct(trellis: &[u16], k: usize, n: usize, b: Bitrate) -> Vec<u16> {
    assert!(k % 16 == 0 && n % 16 == 0, "MUL1 trellis: [{k}, {n}] is not a multiple of 16");
    let wpt = b.words_per_tile();
    assert_eq!(trellis.len(), k / 16 * (n / 16) * wpt, "trellis length for [{k}, {n}] at K = {}", b.k());
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

/// One exllamav3 linear `[k = in, n = out]`: trellis words, suh (k fp16) and svh (n fp16).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Linear {
    pub k: usize,
    pub n: usize,
    pub trellis: Vec<u16>,
    pub suh: Vec<u16>,
    pub svh: Vec<u16>,
}

/// Byte layout of one expert record.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RecordLayout {
    pub hidden: usize,
    pub inter: usize,
    pub bitrate: Bitrate,
    /// bytes of one trellis (gate, up and down hold the same number of weights)
    pub trellis_bytes: u64,
    /// bytes before the padding: three trellis plus six scale vectors
    pub payload_bytes: u64,
    /// the record's size, `payload_bytes` rounded up to `EXPERT_ALIGN`
    pub size: u64,
}

impl RecordLayout {
    pub fn new(hidden: usize, inter: usize, bitrate: Bitrate) -> Result<RecordLayout, String> {
        if hidden == 0 || inter == 0 || hidden % 128 != 0 || inter % 128 != 0 {
            return Err(format!("MUL1 record: hidden {hidden} and inter {inter} must be positive multiples of 128"));
        }
        let trellis_bytes = (hidden / 16 * (inter / 16) * bitrate.words_per_tile() * 2) as u64;
        let scales = 3 * 2 * (hidden + inter) as u64;
        let payload_bytes = 3 * trellis_bytes + scales;
        let size = payload_bytes.div_ceil(EXPERT_ALIGN) * EXPERT_ALIGN;
        Ok(RecordLayout { hidden, inter, bitrate, trellis_bytes, payload_bytes, size })
    }

    /// `[k, n]` of gate, up, down.
    pub fn shapes(&self) -> [(usize, usize); 3] {
        [(self.hidden, self.inter), (self.hidden, self.inter), (self.inter, self.hidden)]
    }

    /// Offsets inside the record of the gate, up and down entries an index names: the starts of
    /// the three trellis blocks (their suh/svh sit in the record's tail, at fixed places). The
    /// engine takes `next tensor offset - gate offset` as the record size, i.e. `size`.
    pub fn tensor_offsets(&self) -> [u64; 3] {
        [0, self.trellis_bytes, 2 * self.trellis_bytes]
    }

    /// File offset of record `index` in a store whose first record starts at 0.
    pub fn offset(&self, index: u64) -> u64 {
        index * self.size
    }
}

fn put_u16s(out: &mut Vec<u8>, v: &[u16]) {
    for x in v {
        out.extend_from_slice(&x.to_le_bytes());
    }
}

/// The bytes of one record from the expert's gate, up and down linears.
pub fn write_record(layout: &RecordLayout, mats: [&Linear; 3]) -> Result<Vec<u8>, String> {
    let wpt = layout.bitrate.words_per_tile();
    for (m, (k, n)) in mats.iter().zip(layout.shapes()) {
        if m.k != k || m.n != n || m.trellis.len() != k / 16 * (n / 16) * wpt || m.suh.len() != k || m.svh.len() != n {
            return Err(format!(
                "MUL1 record: linear [{}, {}] (trellis {}, suh {}, svh {}) does not fit [{k}, {n}] at K = {}",
                m.k,
                m.n,
                m.trellis.len(),
                m.suh.len(),
                m.svh.len(),
                layout.bitrate.k()
            ));
        }
    }
    let mut out = Vec::with_capacity(layout.size as usize);
    for m in mats {
        put_u16s(&mut out, &m.trellis);
    }
    for m in mats {
        put_u16s(&mut out, &m.suh);
        put_u16s(&mut out, &m.svh);
    }
    debug_assert_eq!(out.len() as u64, layout.payload_bytes);
    out.resize(layout.size as usize, 0);
    Ok(out)
}

/// Gate, up and down back from one record's bytes.
pub fn read_record(layout: &RecordLayout, rec: &[u8]) -> Result<[Linear; 3], String> {
    if rec.len() as u64 != layout.size {
        return Err(format!("MUL1 record: {} bytes, the layout says {}", rec.len(), layout.size));
    }
    let mut pos = 0usize;
    let mut take = |words: usize| -> Vec<u16> {
        let v = rec[pos..pos + 2 * words].chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect();
        pos += 2 * words;
        v
    };
    let shapes = layout.shapes();
    let wpt = layout.bitrate.words_per_tile();
    let trellis: Vec<Vec<u16>> = shapes.iter().map(|&(k, n)| take(k / 16 * (n / 16) * wpt)).collect();
    let mut mats = Vec::with_capacity(3);
    for (t, &(k, n)) in trellis.into_iter().zip(shapes.iter()) {
        let suh = take(k);
        let svh = take(n);
        mats.push(Linear { k, n, trellis: t, suh, svh });
    }
    let [g, u, d]: [Linear; 3] = mats.try_into().map_err(|_| "three linears".to_string())?;
    Ok([g, u, d])
}

#[cfg(test)]
mod tests {
    use super::*;
    use sha2::{Digest, Sha256};

    const FIX: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures");

    /// The synthetic u16 stream of `exl3_mul1_fixtures.py` `synth_words` (same formula).
    fn synth_words(count: usize, seed: u32) -> Vec<u16> {
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

    fn sha_u16(v: &[u16]) -> String {
        let mut h = Sha256::new();
        for x in v {
            h.update(x.to_le_bytes());
        }
        format!("{:x}", h.finalize())
    }

    struct Case {
        name: String,
        source: String,
        k: f64,
        hidden: usize,
        inter: usize,
        want: [String; 3],
    }

    fn cases() -> Vec<Case> {
        let tsv = include_str!("../tests/fixtures/exl3-mul1-decode.tsv");
        tsv.lines()
            .filter(|l| !l.starts_with('#') && !l.trim().is_empty())
            .map(|l| {
                let f: Vec<&str> = l.split('\t').collect();
                assert_eq!(f.len(), 8, "exl3-mul1-decode.tsv row {l}");
                Case {
                    name: f[0].to_string(),
                    source: f[1].to_string(),
                    k: f[2].parse().unwrap(),
                    hidden: f[3].parse().unwrap(),
                    inter: f[4].parse().unwrap(),
                    want: [f[5].to_string(), f[6].to_string(), f[7].to_string()],
                }
            })
            .collect()
    }

    /// The expert of one case: synthetic words (seed = the `source` field after `synth:`), or the
    /// exllamav3 quantizer output stored as one unpadded record in `exl3-mul1-<name>.bin`.
    fn expert(c: &Case) -> [Linear; 3] {
        let b = Bitrate::from_k(c.k).unwrap();
        let lay = RecordLayout::new(c.hidden, c.inter, b).unwrap();
        if let Some(seed) = c.source.strip_prefix("synth:") {
            let seed: u32 = seed.parse().unwrap();
            let wpt = b.words_per_tile();
            let mats: Vec<Linear> = lay
                .shapes()
                .iter()
                .enumerate()
                .map(|(i, &(k, n))| Linear {
                    k,
                    n,
                    trellis: synth_words(k / 16 * (n / 16) * wpt, seed + 3 * i as u32),
                    suh: synth_words(k, seed + 3 * i as u32 + 1),
                    svh: synth_words(n, seed + 3 * i as u32 + 2),
                })
                .collect();
            mats.try_into().unwrap()
        } else {
            let mut raw = std::fs::read(format!("{FIX}/{}", c.source)).unwrap_or_else(|e| panic!("{}: {e}", c.source));
            assert_eq!(raw.len() as u64, lay.payload_bytes, "{}: unpadded record size", c.source);
            raw.resize(lay.size as usize, 0);
            read_record(&lay, &raw).unwrap()
        }
    }

    /// The acceptance test: every matrix decodes to exactly the fp16 bytes exllamav3's own
    /// `reconstruct` kernel produced (sha256 of the `[k, n]` LE bytes, `exl3-mul1-decode.tsv`,
    /// written by `exl3_mul1_fixtures.py` on an RTX 5090). Quantizer outputs at K = 2, 3, 4, 3.5
    /// and synthetic streams at GLM-5.3-Flash's routed-expert shapes [4096, 2048] / [2048, 4096].
    #[test]
    fn decode_is_byte_identical_to_exllamav3() {
        let cs = cases();
        assert!(cs.iter().any(|c| c.hidden == 4096 && c.inter == 2048), "a GLM-shaped case");
        assert!(cs.iter().any(|c| !c.source.starts_with("synth:")), "a quantizer case");
        for c in &cs {
            let b = Bitrate::from_k(c.k).unwrap();
            for (m, want) in expert(c).iter().zip(&c.want) {
                if want == "-" {
                    continue;
                }
                let got = sha_u16(&reconstruct(&m.trellis, m.k, m.n, b));
                assert_eq!(&got, want, "{} [{}, {}] K = {}", c.name, m.k, m.n, c.k);
            }
        }
    }

    /// Records go into a store back to back: every record offset is a multiple of 4096, also
    /// for shapes whose payload is not (the small quantizer cases), and each record reads back
    /// to the same linears with zero padding.
    #[test]
    fn record_offsets_are_4096_aligned() {
        let mut store = Vec::new();
        let mut offsets = Vec::new();
        for c in cases() {
            let lay = RecordLayout::new(c.hidden, c.inter, Bitrate::from_k(c.k).unwrap()).unwrap();
            if c.hidden > 512 {
                continue; // the GLM rows: their size is checked in glm_record_size_at_3_bit
            }
            let mats = expert(&c);
            for e in 0..3u64 {
                let rec = write_record(&lay, [&mats[0], &mats[1], &mats[2]]).unwrap();
                assert_eq!(rec.len() as u64, lay.size);
                assert!(rec[lay.payload_bytes as usize..].iter().all(|&x| x == 0), "padding is zeros");
                assert_eq!(read_record(&lay, &rec).unwrap(), mats, "{} round trip", c.name);
                if c.name == "q-k3" {
                    assert_eq!(lay.offset(e), e * lay.size);
                }
                offsets.push((c.name.clone(), store.len() as u64, lay.payload_bytes));
                store.extend_from_slice(&rec);
            }
        }
        assert!(offsets.len() >= 12);
        assert!(offsets.iter().any(|o| o.2 % EXPERT_ALIGN != 0), "a payload that needs padding");
        for (name, off, _) in &offsets {
            assert_eq!(off % 4096, 0, "{name}: record at {off}");
        }
        assert_eq!(store.len() as u64 % 4096, 0);
    }

    /// One GLM-5.3-Flash routed expert (gate/up [4096, 2048], down [2048, 4096]) at K = 3, the
    /// expert bitrate of the 3.05 bpw EXL3 checkpoint: 3 x 3,145,728 B trellis + 36,864 B of
    /// fp16 suh/svh = 9,474,048 B = 2313 x 4096, no padding. The tensor sizes are those
    /// exllamav3's quantizer wrote for a synthetic expert of these shapes (`exl3-mul1-glm-sizes.tsv`).
    #[test]
    fn glm_record_size_at_3_bit() {
        let lay = RecordLayout::new(4096, 2048, Bitrate::from_k(3.0).unwrap()).unwrap();
        assert_eq!(lay.trellis_bytes, 3_145_728);
        assert_eq!(lay.payload_bytes, 9_474_048);
        assert_eq!(lay.size, 9_474_048);
        assert_eq!(lay.size % 4096, 0);
        let tsv = include_str!("../tests/fixtures/exl3-mul1-glm-sizes.tsv");
        let mut sum = 0u64;
        let mut rows = 0;
        for l in tsv.lines().filter(|l| !l.starts_with('#') && !l.trim().is_empty()) {
            let f: Vec<&str> = l.split('\t').collect();
            let (trellis, suh, svh): (u64, u64, u64) = (f[2].parse().unwrap(), f[3].parse().unwrap(), f[4].parse().unwrap());
            assert_eq!(trellis, lay.trellis_bytes, "{}", f[0]);
            sum += trellis + suh + svh;
            rows += 1;
        }
        assert_eq!(rows, 3);
        assert_eq!(sum, lay.payload_bytes);
    }

    /// The engine's reading of a glm5_next expert unit (local main `54ef024`,
    /// `nvme_source::glm5_record_from_index`): dtype `mul1`, gate < up < down, the record from the
    /// gate offset to the next unit a non-zero multiple of 4096.
    #[test]
    fn index_entries_follow_the_engine_contract() {
        assert_eq!(DTYPE, "mul1");
        for (h, i, k) in [(4096, 2048, 3.0), (256, 128, 3.0), (384, 256, 2.0), (128, 128, 4.0), (256, 128, 3.5)] {
            let lay = RecordLayout::new(h, i, Bitrate::from_k(k).unwrap()).unwrap();
            let [g, u, d] = lay.tensor_offsets();
            assert!(g == 0 && g < u && u < d && d + lay.trellis_bytes <= lay.payload_bytes, "[{h}, {i}] K = {k}");
            let next = lay.offset(1);
            assert!(next - g == lay.size && lay.size > 0 && lay.size % 4096 == 0, "[{h}, {i}] K = {k}");
        }
    }

    #[test]
    fn bitrate_set_is_exllamav3s() {
        for k in [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 8.0] {
            assert_eq!(Bitrate::from_k(k).unwrap().k(), k);
        }
        for k in [0.0, 0.5, 4.5, 9.0, 3.05, 3.25] {
            assert!(Bitrate::from_k(k).is_err(), "{k}");
        }
        assert_eq!(Bitrate::from_k(3.0).unwrap().words_per_tile(), 48);
        assert_eq!(Bitrate::from_k(3.5).unwrap().words_per_tile(), 56);
    }

    /// Manual check against a quantizer output at the GLM shapes that is too large to commit:
    /// `exl3_mul1_fixtures.py --glm-out DIR` writes `glm-k3.bin` (one unpadded record) and
    /// `glm-k3.tsv` (exllamav3's reconstruct digests); run with
    /// `CNQ_MUL1_GLM_DIR=DIR cargo test glm_quantizer_expert -- --ignored`.
    #[test]
    #[ignore]
    fn glm_quantizer_expert_decodes_byte_identical() {
        let dir = std::env::var("CNQ_MUL1_GLM_DIR").expect("CNQ_MUL1_GLM_DIR");
        let row = std::fs::read_to_string(format!("{dir}/glm-k3.tsv")).unwrap();
        let f: Vec<String> = row.trim().split('\t').map(str::to_string).collect();
        let lay = RecordLayout::new(4096, 2048, Bitrate::from_k(3.0).unwrap()).unwrap();
        let raw = std::fs::read(format!("{dir}/glm-k3.bin")).unwrap();
        assert_eq!(raw.len() as u64, lay.size);
        let mats = read_record(&lay, &raw).unwrap();
        for (m, want) in mats.iter().zip(&f) {
            assert_eq!(&sha_u16(&reconstruct(&m.trellis, m.k, m.n, lay.bitrate)), want);
        }
    }
}
