//! A small GGUF reader for Unsloth's importance matrix (#79), F32 tensors only.
//!
//! `models/unsloth-imatrix/imatrix_unsloth.gguf` is a GGUF v3 file with 1852 tensors, all of
//! them F32, and four kv entries. It is not a model: it is what `llama-imatrix` wrote while
//! 45 chunks of 18,432 tokens ran through the BF16 model. Per layer N it carries
//!
//! ```text
//! blk.N.ffn_gate_exps.weight.in_sum2   [2560, 512]   blk.N.ffn_gate_exps.weight.counts   [1, 512]
//! blk.N.ffn_up_exps.weight.in_sum2     [2560, 512]   blk.N.ffn_up_exps.weight.counts     [1, 512]
//! blk.N.ffn_down_exps.weight.in_sum2   [ 640, 512]   blk.N.ffn_down_exps.weight.counts   [1, 512]
//! ```
//!
//! GGUF lists dimensions innermost first, so `ne[0] = 2560` is the INPUT column index and
//! `ne[1] = 512` the expert: element `[e][j]` sits at `e * ne[0] + j`.
//!
//! The semantics are `tools/imatrix/imatrix.cpp`'s, read out of the source and not guessed:
//! `in_sum2[e][j]` is the sum of `x[j]^2` over every token routed to expert `e`, and
//! `counts[e]` is how many routings that was. `tools/quantize/quantize.cpp` (the GGUF branch,
//! `is_legacy == false`) turns that into the weight the quantizer uses:
//!
//! ```text
//! qw[e][j] = counts[e] > 0 ? in_sum2[e][j] / counts[e] : 1
//! ```
//!
//! — the mean squared activation of input column `j` for expert `e`, and a flat 1 for an
//! expert the calibration never routed to. That fallback is llama.cpp's and it is kept here on
//! purpose: MoEQuant (arXiv 2505.03804) names rare experts as the weak spot of any
//! calibration-based method, and an expert with no counts must fall back to the unweighted
//! rule rather than to a row of zeros, which would let its weights be rounded to anything.
//!
//! This reader is deliberately minimal: it refuses anything but F32, it never allocates a
//! tensor it was not asked for, and its unit tests build GGUF bytes by hand.

use std::collections::BTreeMap;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

/// GGUF metadata value types (`gguf.h`). Only the ones this file needs are decoded; an array
/// is decoded element by element with the same table.
const T_UINT8: u32 = 0;
const T_INT8: u32 = 1;
const T_UINT16: u32 = 2;
const T_INT16: u32 = 3;
const T_UINT32: u32 = 4;
const T_INT32: u32 = 5;
const T_FLOAT32: u32 = 6;
const T_BOOL: u32 = 7;
const T_STRING: u32 = 8;
const T_ARRAY: u32 = 9;
const T_UINT64: u32 = 10;
const T_INT64: u32 = 11;
const T_FLOAT64: u32 = 12;

/// `GGML_TYPE_F32`. The importance matrix stores nothing else, and this reader refuses
/// anything else rather than mis-reading a quantized block as floats.
pub const GGML_TYPE_F32: u32 = 0;

#[derive(Debug)]
pub struct GgufTensor {
    pub name: String,
    /// as GGUF lists them: innermost (fastest-varying) first
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// relative to the start of the tensor data area
    pub offset: u64,
}

impl GgufTensor {
    pub fn n_elements(&self) -> u64 {
        self.dims.iter().product()
    }
}

#[derive(Debug)]
pub struct Gguf {
    pub path: PathBuf,
    pub version: u32,
    pub alignment: u64,
    pub kv: BTreeMap<String, serde_json::Value>,
    pub tensors: Vec<GgufTensor>,
    /// absolute file offset of the tensor data area
    pub data_start: u64,
    pub file_len: u64,
}

/// A cursor over the header prefix. Every read is bounds-checked and says how far it got, so
/// `Gguf::open` can ask for more bytes instead of guessing the header size.
struct Cur<'a> {
    b: &'a [u8],
    o: usize,
}

/// `Err(None)` means "the prefix was too short" — the caller reads more and retries.
/// `Err(Some(msg))` is a real refusal.
type CurResult<T> = Result<T, Option<String>>;

impl<'a> Cur<'a> {
    fn take(&mut self, n: usize) -> CurResult<&'a [u8]> {
        if self.o + n > self.b.len() {
            return Err(None);
        }
        let s = &self.b[self.o..self.o + n];
        self.o += n;
        Ok(s)
    }
    fn u32(&mut self) -> CurResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> CurResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> CurResult<String> {
        let n = self.u64()?;
        if n > (1 << 24) {
            return Err(Some(format!("a GGUF string of {n} B is not a name this reader accepts")));
        }
        let raw = self.take(n as usize)?;
        String::from_utf8(raw.to_vec()).map_err(|e| Some(format!("GGUF string is not utf-8: {e}")))
    }
    fn value(&mut self, t: u32) -> CurResult<serde_json::Value> {
        use serde_json::json;
        Ok(match t {
            T_UINT8 => json!(self.take(1)?[0]),
            T_INT8 => json!(self.take(1)?[0] as i8),
            T_UINT16 => json!(u16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            T_INT16 => json!(i16::from_le_bytes(self.take(2)?.try_into().unwrap())),
            T_UINT32 => json!(self.u32()?),
            T_INT32 => json!(i32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            T_FLOAT32 => json!(f32::from_le_bytes(self.take(4)?.try_into().unwrap())),
            T_BOOL => json!(self.take(1)?[0] != 0),
            T_STRING => json!(self.string()?),
            T_UINT64 => json!(self.u64()?),
            T_INT64 => json!(i64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_FLOAT64 => json!(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            T_ARRAY => {
                let et = self.u32()?;
                let n = self.u64()?;
                if et == T_ARRAY {
                    return Err(Some("a GGUF array of arrays is not something this reader knows".into()));
                }
                let mut out = Vec::new();
                for _ in 0..n {
                    out.push(self.value(et)?);
                }
                serde_json::Value::Array(out)
            }
            other => return Err(Some(format!("GGUF value type {other} is not one this reader knows"))),
        })
    }
}

/// Parse a GGUF header out of a PREFIX of the file. `Ok(None)` means the prefix was too short.
pub fn parse_header(prefix: &[u8], file_len: u64) -> Result<Option<Gguf>, String> {
    let mut c = Cur { b: prefix, o: 0 };
    let magic = match c.take(4) {
        Ok(m) => m,
        Err(_) => return Ok(None),
    };
    if magic != b"GGUF" {
        return Err(format!("magic is {magic:?}, not GGUF"));
    }
    let mut go = || -> CurResult<Gguf> {
        let version = c.u32()?;
        if version != 3 {
            return Err(Some(format!("GGUF version {version}; this reader was written against v3")));
        }
        let n_tensors = c.u64()?;
        let n_kv = c.u64()?;
        if n_tensors > 1 << 22 || n_kv > 1 << 16 {
            return Err(Some(format!("{n_tensors} tensors / {n_kv} kv entries is not a file this reader accepts")));
        }
        let mut kv = BTreeMap::new();
        for _ in 0..n_kv {
            let k = c.string()?;
            let t = c.u32()?;
            let v = c.value(t)?;
            kv.insert(k, v);
        }
        let mut tensors = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = c.string()?;
            let nd = c.u32()?;
            if nd > 4 {
                return Err(Some(format!("{name}: {nd} dimensions, GGUF allows at most 4")));
            }
            let mut dims = Vec::with_capacity(nd as usize);
            for _ in 0..nd {
                dims.push(c.u64()?);
            }
            let ggml_type = c.u32()?;
            let offset = c.u64()?;
            tensors.push(GgufTensor { name, dims, ggml_type, offset });
        }
        let alignment = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
        if alignment == 0 || !alignment.is_power_of_two() {
            return Err(Some(format!("general.alignment {alignment} is not a power of two")));
        }
        let data_start = (c.o as u64).div_ceil(alignment) * alignment;
        Ok(Gguf {
            path: PathBuf::new(),
            version,
            alignment,
            kv,
            tensors,
            data_start,
            file_len,
        })
    };
    match go() {
        Ok(g) => Ok(Some(g)),
        Err(None) => Ok(None),
        Err(Some(m)) => Err(m),
    }
}

impl Gguf {
    /// Open a GGUF file and parse its header. The header size is not known in advance, so the
    /// prefix is read and doubled until it covers it (8 MiB is already ten times the 122,496 B
    /// header of the file this was written for).
    pub fn open(path: &Path) -> Result<Gguf, String> {
        let mut f = std::fs::File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        let file_len = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
        let mut want = (8usize << 20).min(file_len as usize);
        loop {
            let mut buf = vec![0u8; want];
            f.seek(SeekFrom::Start(0)).map_err(|e| format!("{}: {e}", path.display()))?;
            f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
            match parse_header(&buf, file_len)? {
                Some(mut g) => {
                    g.path = path.to_path_buf();
                    return Ok(g);
                }
                None => {
                    if want as u64 >= file_len {
                        return Err(format!("{}: the header does not fit the file", path.display()));
                    }
                    want = (want * 2).min(file_len as usize);
                }
            }
        }
    }

    pub fn find(&self, name: &str) -> Option<&GgufTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// The raw little-endian bytes of one F32 tensor.
    pub fn read_f32_bytes(&self, name: &str) -> Result<Vec<u8>, String> {
        let t = self.find(name).ok_or_else(|| format!("{}: no tensor named {name}", self.path.display()))?;
        if t.ggml_type != GGML_TYPE_F32 {
            return Err(format!("{name}: ggml type {} — this reader reads F32 only", t.ggml_type));
        }
        let n = t.n_elements();
        let nbytes = n * 4;
        let begin = self.data_start + t.offset;
        if begin + nbytes > self.file_len {
            return Err(format!("{name}: runs to byte {} of a {} B file", begin + nbytes, self.file_len));
        }
        let mut f = std::fs::File::open(&self.path).map_err(|e| format!("{}: {e}", self.path.display()))?;
        f.seek(SeekFrom::Start(begin)).map_err(|e| format!("{}: {e}", self.path.display()))?;
        let mut raw = vec![0u8; nbytes as usize];
        f.read_exact(&mut raw).map_err(|e| format!("{}: {e}", self.path.display()))?;
        Ok(raw)
    }

    /// One F32 tensor as floats.
    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, String> {
        let raw = self.read_f32_bytes(name)?;
        Ok(raw.chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect())
    }
}

#[derive(Debug)]
/// The importance matrix of ONE routed-expert tensor of one layer: `n_experts` rows of
/// `n_cols` mean squared activations, already divided by the counts.
pub struct ExpertImatrix {
    pub name: String,
    pub n_experts: usize,
    pub n_cols: usize,
    /// `in_sum2`, as stored: `[e * n_cols + j]`
    pub sums: Vec<f32>,
    /// `counts[e]`, as stored (a float that holds an integer)
    pub counts: Vec<f32>,
}

impl ExpertImatrix {
    /// The weights `llama-quantize` would hand the quantizer for expert `e`:
    /// `in_sum2[e][j] / counts[e]`, or a flat 1 when the calibration never routed to `e`.
    pub fn row(&self, e: usize) -> Vec<f32> {
        let c = self.counts[e];
        let base = e * self.n_cols;
        if c > 0.0 {
            self.sums[base..base + self.n_cols].iter().map(|v| v / c).collect()
        } else {
            vec![1.0f32; self.n_cols]
        }
    }

    /// experts the calibration never routed to
    pub fn zero_count_experts(&self) -> Vec<usize> {
        (0..self.n_experts).filter(|&e| self.counts[e] <= 0.0).collect()
    }

    /// experts routed to fewer than `n` times (rare experts — MoEQuant's weak spot)
    pub fn tiny_count_experts(&self, n: f32) -> Vec<usize> {
        (0..self.n_experts).filter(|&e| self.counts[e] > 0.0 && self.counts[e] < n).collect()
    }
}

/// Which routed-expert tensor of a layer an importance matrix is wanted for.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ExpertTensor {
    /// the FUSED `mlp.experts.gate_up_proj` — see `load` for why one imatrix covers both halves
    GateUp,
    Down,
}

/// Read the importance matrix of one layer's routed experts.
///
/// For the FUSED `gate_up_proj` this asks for `ffn_gate_exps` and then REFUSES unless
/// `ffn_up_exps.in_sum2` is bitwise identical to it. That is not an optimization, it is what
/// makes the fused layout irrelevant: gate and up read the SAME activation row, so
/// `imatrix.cpp` accumulates the same `x[j]^2` into both, and if that ever stopped being true
/// the gate/up split inside the 1280 rows would have to be known. Checked, not assumed.
pub fn load(gguf: &Gguf, layer: usize, which: ExpertTensor) -> Result<ExpertImatrix, String> {
    let stem = match which {
        ExpertTensor::GateUp => format!("blk.{layer}.ffn_gate_exps.weight"),
        ExpertTensor::Down => format!("blk.{layer}.ffn_down_exps.weight"),
    };
    let sums_name = format!("{stem}.in_sum2");
    let counts_name = format!("{stem}.counts");
    let st = gguf
        .find(&sums_name)
        .ok_or_else(|| format!("{}: no tensor named {sums_name}", gguf.path.display()))?;
    if st.dims.len() != 2 {
        return Err(format!("{sums_name}: dims {:?}, expected [n_cols, n_experts]", st.dims));
    }
    let n_cols = st.dims[0] as usize;
    let n_experts = st.dims[1] as usize;
    let sums = gguf.read_f32(&sums_name)?;
    let counts = gguf.read_f32(&counts_name)?;
    if counts.len() != n_experts {
        return Err(format!("{counts_name}: {} counts for {n_experts} experts", counts.len()));
    }
    if sums.len() != n_cols * n_experts {
        return Err(format!("{sums_name}: {} values for {n_cols} x {n_experts}", sums.len()));
    }
    for (e, c) in counts.iter().enumerate() {
        if !c.is_finite() || *c < 0.0 {
            return Err(format!("{counts_name}: expert {e} has count {c}"));
        }
    }
    for (i, v) in sums.iter().enumerate() {
        if !v.is_finite() || *v < 0.0 {
            return Err(format!("{sums_name}: element {i} is {v}; in_sum2 is a sum of squares"));
        }
    }
    if which == ExpertTensor::GateUp {
        let up = format!("blk.{layer}.ffn_up_exps.weight.in_sum2");
        let a = gguf.read_f32_bytes(&sums_name)?;
        let b = gguf.read_f32_bytes(&up)?;
        if a != b {
            return Err(format!(
                "{sums_name} and {up} differ — the fused gate_up_proj can only take ONE imatrix \
while gate and up see the same activation; they do not here, so the gate/up split inside the \
fused rows would have to be known first"
            ));
        }
    }
    Ok(ExpertImatrix { name: sums_name, n_experts, n_cols, sums, counts })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a GGUF v3 file in memory: kv entries, then F32 tensors in the given order.
    fn gguf_bytes(kv: &[(&str, u32, Vec<u8>)], tensors: &[(&str, Vec<u64>, Vec<f32>)], alignment: u64) -> Vec<u8> {
        let mut h = Vec::new();
        h.extend_from_slice(b"GGUF");
        h.extend_from_slice(&3u32.to_le_bytes());
        h.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
        h.extend_from_slice(&(kv.len() as u64).to_le_bytes());
        for (k, t, v) in kv {
            h.extend_from_slice(&(k.len() as u64).to_le_bytes());
            h.extend_from_slice(k.as_bytes());
            h.extend_from_slice(&t.to_le_bytes());
            h.extend_from_slice(v);
        }
        let mut off = 0u64;
        for (name, dims, vals) in tensors {
            h.extend_from_slice(&(name.len() as u64).to_le_bytes());
            h.extend_from_slice(name.as_bytes());
            h.extend_from_slice(&(dims.len() as u32).to_le_bytes());
            for d in dims {
                h.extend_from_slice(&d.to_le_bytes());
            }
            h.extend_from_slice(&GGML_TYPE_F32.to_le_bytes());
            h.extend_from_slice(&off.to_le_bytes());
            off += (vals.len() * 4) as u64;
            // every tensor starts on an `alignment` boundary in the real writer
            off = off.div_ceil(alignment) * alignment;
        }
        let pad = (h.len() as u64).div_ceil(alignment) * alignment - h.len() as u64;
        h.extend(std::iter::repeat_n(0u8, pad as usize));
        let data_start = h.len();
        let mut off = 0u64;
        for (_, _, vals) in tensors {
            h.resize(data_start + off as usize, 0);
            for v in vals {
                h.extend_from_slice(&v.to_le_bytes());
            }
            off += (vals.len() * 4) as u64;
            off = off.div_ceil(alignment) * alignment;
        }
        h.resize(data_start + off as usize, 0);
        h
    }

    fn u64v(v: u64) -> Vec<u8> {
        v.to_le_bytes().to_vec()
    }

    fn write_tmp(bytes: &[u8], name: &str) -> PathBuf {
        let p = std::env::temp_dir().join(format!("crow-nest-imatrix-test-{name}-{}.gguf", std::process::id()));
        std::fs::write(&p, bytes).unwrap();
        p
    }

    /// Two experts, three input columns. The numbers are the ones a stdlib Python GGUF reader
    /// prints back out of the same bytes, and the derived weights are hand-computed:
    /// expert 0 has 4 counts, expert 1 has none.
    fn toy() -> Vec<u8> {
        gguf_bytes(
            &[
                ("general.type", T_STRING, {
                    let mut v = u64v(7);
                    v.extend_from_slice(b"imatrix");
                    v
                }),
                ("imatrix.chunk_count", T_UINT32, 45u32.to_le_bytes().to_vec()),
                ("general.alignment", T_UINT32, 32u32.to_le_bytes().to_vec()),
            ],
            &[
                ("blk.5.ffn_gate_exps.weight.in_sum2", vec![3, 2], vec![8.0, 4.0, 2.0, 100.0, 200.0, 300.0]),
                ("blk.5.ffn_gate_exps.weight.counts", vec![1, 2], vec![4.0, 0.0]),
                ("blk.5.ffn_up_exps.weight.in_sum2", vec![3, 2], vec![8.0, 4.0, 2.0, 100.0, 200.0, 300.0]),
                ("blk.5.ffn_up_exps.weight.counts", vec![1, 2], vec![4.0, 0.0]),
                ("blk.5.ffn_down_exps.weight.in_sum2", vec![2, 2], vec![1.0, 3.0, 5.0, 7.0]),
                ("blk.5.ffn_down_exps.weight.counts", vec![1, 2], vec![2.0, 8.0]),
            ],
            32,
        )
    }

    #[test]
    fn the_header_of_a_hand_built_gguf_parses_to_its_kv_and_tensors() {
        let raw = toy();
        let g = parse_header(&raw, raw.len() as u64).unwrap().unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.alignment, 32);
        assert_eq!(g.kv["general.type"], serde_json::json!("imatrix"));
        assert_eq!(g.kv["imatrix.chunk_count"], serde_json::json!(45));
        assert_eq!(g.tensors.len(), 6);
        let t = g.find("blk.5.ffn_gate_exps.weight.in_sum2").unwrap();
        assert_eq!(t.dims, vec![3, 2]);
        assert_eq!(t.ggml_type, GGML_TYPE_F32);
        assert_eq!(t.n_elements(), 6);
        assert_eq!(g.data_start % 32, 0);
    }

    #[test]
    fn a_prefix_shorter_than_the_header_says_so_instead_of_guessing() {
        let raw = toy();
        assert!(parse_header(&raw[..40], raw.len() as u64).unwrap().is_none());
        assert!(parse_header(&raw[..4], raw.len() as u64).unwrap().is_none());
        assert!(parse_header(b"GGU", raw.len() as u64).unwrap().is_none());
    }

    #[test]
    fn a_file_that_is_not_a_gguf_v3_is_refused_by_name() {
        let mut bad = toy();
        bad[0] = b'X';
        assert!(parse_header(&bad, 100).unwrap_err().contains("not GGUF"));
        let mut v2 = toy();
        v2[4..8].copy_from_slice(&2u32.to_le_bytes());
        assert!(parse_header(&v2, 100).unwrap_err().contains("version 2"));
    }

    #[test]
    fn the_values_come_back_the_way_a_python_reader_prints_them() {
        let raw = toy();
        let p = write_tmp(&raw, "values");
        let g = Gguf::open(&p).unwrap();
        assert_eq!(
            g.read_f32("blk.5.ffn_gate_exps.weight.in_sum2").unwrap(),
            vec![8.0, 4.0, 2.0, 100.0, 200.0, 300.0]
        );
        assert_eq!(g.read_f32("blk.5.ffn_down_exps.weight.in_sum2").unwrap(), vec![1.0, 3.0, 5.0, 7.0]);
        assert_eq!(g.read_f32("blk.5.ffn_gate_exps.weight.counts").unwrap(), vec![4.0, 0.0]);
        assert!(g.read_f32("blk.5.nope").unwrap_err().contains("no tensor named"));
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn the_weight_rule_is_llama_quantizes_sums_over_counts_with_its_zero_count_fallback() {
        let raw = toy();
        let p = write_tmp(&raw, "weights");
        let g = Gguf::open(&p).unwrap();
        let im = load(&g, 5, ExpertTensor::GateUp).unwrap();
        assert_eq!((im.n_experts, im.n_cols), (2, 3));
        // expert 0: counts 4 -> 8/4, 4/4, 2/4
        assert_eq!(im.row(0), vec![2.0, 1.0, 0.5]);
        // expert 1: counts 0 -> the flat fallback, NOT the raw sums and NOT zeros
        assert_eq!(im.row(1), vec![1.0, 1.0, 1.0]);
        assert_eq!(im.zero_count_experts(), vec![1]);
        assert_eq!(im.tiny_count_experts(10.0), vec![0]);

        let dn = load(&g, 5, ExpertTensor::Down).unwrap();
        assert_eq!((dn.n_experts, dn.n_cols), (2, 2));
        assert_eq!(dn.row(0), vec![0.5, 1.5]);
        assert_eq!(dn.row(1), vec![0.625, 0.875]);
        assert!(dn.zero_count_experts().is_empty());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_gate_and_up_imatrix_that_differ_refuse_the_fused_tensor() {
        // the whole reason the fused gate|up layout never has to be known: gate and up see the
        // same activation row, so their in_sum2 are the same bytes. If they were not, one
        // imatrix could not cover 1280 fused rows and this must fail loudly.
        let mut tensors: Vec<(&str, Vec<u64>, Vec<f32>)> = vec![
            ("blk.5.ffn_gate_exps.weight.in_sum2", vec![3, 2], vec![8.0, 4.0, 2.0, 100.0, 200.0, 300.0]),
            ("blk.5.ffn_gate_exps.weight.counts", vec![1, 2], vec![4.0, 1.0]),
            ("blk.5.ffn_up_exps.weight.in_sum2", vec![3, 2], vec![8.0, 4.0, 2.0, 100.0, 200.0, 301.0]),
            ("blk.5.ffn_up_exps.weight.counts", vec![1, 2], vec![4.0, 1.0]),
        ];
        let raw = gguf_bytes(&[("general.alignment", T_UINT32, 32u32.to_le_bytes().to_vec())], &tensors, 32);
        let p = write_tmp(&raw, "gateup-differ");
        let g = Gguf::open(&p).unwrap();
        let err = load(&g, 5, ExpertTensor::GateUp).unwrap_err();
        assert!(err.contains("differ"), "{err}");
        // and the same file with gate == up is accepted
        tensors[2].2 = tensors[0].2.clone();
        let raw = gguf_bytes(&[("general.alignment", T_UINT32, 32u32.to_le_bytes().to_vec())], &tensors, 32);
        std::fs::write(&p, &raw).unwrap();
        let g = Gguf::open(&p).unwrap();
        assert!(load(&g, 5, ExpertTensor::GateUp).is_ok());
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_negative_or_non_finite_in_sum2_is_refused() {
        // in_sum2 is a sum of squares: a negative value means the file is not what it claims
        let raw = gguf_bytes(
            &[("general.alignment", T_UINT32, 32u32.to_le_bytes().to_vec())],
            &[
                ("blk.0.ffn_down_exps.weight.in_sum2", vec![2, 1], vec![1.0, -3.0]),
                ("blk.0.ffn_down_exps.weight.counts", vec![1, 1], vec![2.0]),
            ],
            32,
        );
        let p = write_tmp(&raw, "negative");
        let g = Gguf::open(&p).unwrap();
        let err = load(&g, 0, ExpertTensor::Down).unwrap_err();
        assert!(err.contains("sum of squares"), "{err}");
        std::fs::remove_file(&p).ok();
    }

    #[test]
    fn a_non_f32_tensor_is_refused_rather_than_read_as_floats() {
        let mut raw = toy();
        // rewrite the first tensor's ggml type to Q4_K (12) in the header
        let at = raw
            .windows(34)
            .position(|w| w.starts_with(b"blk.5.ffn_gate_exps.weight.in_sum2"))
            .unwrap();
        let type_at = at + 34 + 4 + 2 * 8; // name, n_dims u32, two u64 dims
        raw[type_at..type_at + 4].copy_from_slice(&12u32.to_le_bytes());
        let p = write_tmp(&raw, "nonf32");
        let g = Gguf::open(&p).unwrap();
        let err = g.read_f32("blk.5.ffn_gate_exps.weight.in_sum2").unwrap_err();
        assert!(err.contains("F32 only"), "{err}");
        std::fs::remove_file(&p).ok();
    }
}
