//! #108: a minimal GGUF v2/v3 reader, just enough to load llama.cpp's
//! vision projector (`mmproj-F16.gguf`) as the vit tower's weights.
//!
//! Format (ggml `docs/gguf.md`, "File Structure"): magic `GGUF`, u32 version,
//! u64 tensor count, u64 metadata-kv count, the kv pairs (u64-length strings,
//! u32 value type, typed value; arrays are u32 element type + u64 count), then
//! per tensor: name, u32 n_dims, n_dims x u64 dims (ne0 = the CONTIGUOUS axis
//! first), u32 ggml type, u64 offset relative to the data section. The data
//! section starts at the next multiple of `general.alignment` (default 32)
//! after the tensor infos. Little-endian only (a big-endian file fails the
//! magic/version check, as ggml's own reader does).
//!
//! Only the types the projector uses are named: F32 (0), F16 (1), BF16 (30).
//! Everything is read with `std::fs` - no mmap, no third-party crate - and
//! every tensor read checks its byte range against the file length.

use std::collections::HashMap;
use std::io::{Read, Seek, SeekFrom};

pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_BF16: u32 = 30;

#[derive(Clone, Debug, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    /// arrays are kept only as their length: the projector reader needs none of them
    Array(u64),
}

impl Value {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (fastest) axis
    pub dims: Vec<u64>,
    pub ggml_type: u32,
    /// absolute file offset of the first byte
    pub offset: u64,
}

impl TensorInfo {
    pub fn numel(&self) -> u64 {
        self.dims.iter().product()
    }
    /// bytes of the types this reader names; None for any other type
    pub fn byte_len(&self) -> Option<u64> {
        let per = match self.ggml_type {
            GGML_TYPE_F32 => 4,
            GGML_TYPE_F16 | GGML_TYPE_BF16 => 2,
            _ => return None,
        };
        Some(self.numel() * per)
    }
}

pub struct Gguf {
    pub path: String,
    pub version: u32,
    pub kv: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
    file_len: u64,
}

struct Rd<R: Read> {
    r: R,
    pos: u64,
}

impl<R: Read> Rd<R> {
    fn bytes(&mut self, n: usize) -> Result<Vec<u8>, String> {
        let mut b = vec![0u8; n];
        self.r.read_exact(&mut b).map_err(|e| format!("gguf: truncated header at byte {}: {e}", self.pos))?;
        self.pos += n as u64;
        Ok(b)
    }
    fn u8(&mut self) -> Result<u8, String> {
        Ok(self.bytes(1)?[0])
    }
    fn u32(&mut self) -> Result<u32, String> {
        Ok(u32::from_le_bytes(self.bytes(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Result<u64, String> {
        Ok(u64::from_le_bytes(self.bytes(8)?.try_into().unwrap()))
    }
    fn string(&mut self) -> Result<String, String> {
        let n = self.u64()?;
        if n > 1 << 20 {
            return Err(format!("gguf: string of {n} bytes at byte {} (corrupt header?)", self.pos));
        }
        let b = self.bytes(n as usize)?;
        String::from_utf8(b).map_err(|_| format!("gguf: non-UTF-8 string at byte {}", self.pos))
    }
    /// one typed value (gguf_type: 0 u8, 1 i8, 2 u16, 3 i16, 4 u32, 5 i32,
    /// 6 f32, 7 bool, 8 string, 9 array, 10 u64, 11 i64, 12 f64)
    fn value(&mut self, t: u32, depth: u32) -> Result<Value, String> {
        Ok(match t {
            0 => Value::U64(self.u8()? as u64),
            1 => Value::I64(self.u8()? as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.bytes(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_bits(self.u32()?) as f64),
            7 => Value::Bool(self.u8()? != 0),
            8 => Value::Str(self.string()?),
            9 => {
                if depth > 2 {
                    return Err("gguf: arrays nested deeper than 2".into());
                }
                let et = self.u32()?;
                let n = self.u64()?;
                if n > 1 << 24 {
                    return Err(format!("gguf: array of {n} elements (corrupt header?)"));
                }
                for _ in 0..n {
                    self.value(et, depth + 1)?;
                }
                Value::Array(n)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(self.u64()? as i64),
            12 => Value::F64(f64::from_bits(self.u64()?)),
            _ => return Err(format!("gguf: unknown value type {t} at byte {}", self.pos)),
        })
    }
}

impl Gguf {
    /// parse the header of `path` (the tensor data is read on demand)
    pub fn open(path: &str) -> Result<Gguf, String> {
        let f = std::fs::File::open(path).map_err(|e| format!("{path}: {e}"))?;
        let file_len = f.metadata().map_err(|e| format!("{path}: {e}"))?.len();
        Gguf::parse(std::io::BufReader::new(f), file_len, path)
    }

    /// the header parser behind `open`, over any reader (the unit tests feed it bytes)
    pub fn parse<R: Read>(r: R, file_len: u64, path: &str) -> Result<Gguf, String> {
        let mut rd = Rd { r, pos: 0 };
        if rd.bytes(4)? != b"GGUF" {
            return Err(format!("{path}: not a GGUF file (magic)"));
        }
        let version = rd.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("{path}: GGUF version {version}, this reader knows 2 and 3"));
        }
        let n_tensors = rd.u64()?;
        let n_kv = rd.u64()?;
        if n_tensors > 1 << 20 || n_kv > 1 << 20 {
            return Err(format!("{path}: {n_tensors} tensors / {n_kv} kv pairs (corrupt header?)"));
        }
        let mut kv = HashMap::new();
        for _ in 0..n_kv {
            let k = rd.string()?;
            let t = rd.u32()?;
            let v = rd.value(t, 0)?;
            kv.insert(k, v);
        }
        let align = kv.get("general.alignment").and_then(|v| v.as_u64()).unwrap_or(32);
        if align == 0 || !align.is_power_of_two() {
            return Err(format!("{path}: general.alignment {align} is not a power of two"));
        }
        let mut raw = Vec::with_capacity(n_tensors as usize);
        for _ in 0..n_tensors {
            let name = rd.string()?;
            let nd = rd.u32()?;
            if nd == 0 || nd > 4 {
                return Err(format!("{path}: tensor {name} has {nd} dims"));
            }
            let dims = (0..nd).map(|_| rd.u64()).collect::<Result<Vec<_>, _>>()?;
            let ggml_type = rd.u32()?;
            let rel = rd.u64()?;
            raw.push((name, dims, ggml_type, rel));
        }
        let data0 = rd.pos.div_ceil(align) * align;
        let mut tensors = Vec::with_capacity(raw.len());
        let mut index = HashMap::new();
        for (name, dims, ggml_type, rel) in raw {
            if rel % align != 0 {
                return Err(format!("{path}: tensor {name} offset {rel} not aligned to {align}"));
            }
            let t = TensorInfo { name: name.clone(), dims, ggml_type, offset: data0 + rel };
            if let Some(len) = t.byte_len() {
                if t.offset + len > file_len {
                    return Err(format!(
                        "{path}: tensor {name} [{}, +{len}) runs past the file end {file_len}",
                        t.offset
                    ));
                }
            }
            if index.insert(name.clone(), tensors.len()).is_some() {
                return Err(format!("{path}: tensor {name} named twice"));
            }
            tensors.push(t);
        }
        Ok(Gguf { path: path.to_string(), version, kv, tensors, index, file_len })
    }

    pub fn find(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn kv_str(&self, key: &str) -> Option<&str> {
        self.kv.get(key).and_then(|v| v.as_str())
    }

    pub fn kv_u64(&self, key: &str) -> Option<u64> {
        self.kv.get(key).and_then(|v| v.as_u64())
    }

    /// the raw bytes of one tensor of a named type
    pub fn read_raw(&self, t: &TensorInfo) -> Result<Vec<u8>, String> {
        let len = t
            .byte_len()
            .ok_or_else(|| format!("{}: tensor {} has ggml type {}, not F32/F16/BF16", self.path, t.name, t.ggml_type))?;
        if t.offset + len > self.file_len {
            return Err(format!("{}: tensor {} runs past the file end", self.path, t.name));
        }
        let mut f = std::fs::File::open(&self.path).map_err(|e| format!("{}: {e}", self.path))?;
        f.seek(SeekFrom::Start(t.offset)).map_err(|e| format!("{}: {e}", self.path))?;
        let mut b = vec![0u8; len as usize];
        f.read_exact(&mut b).map_err(|e| format!("{}: tensor {}: {e}", self.path, t.name))?;
        Ok(b)
    }
}

/// IEEE binary16 -> f32, exact (every f16 value, subnormals included, is an f32)
pub fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h >> 15) as u32) << 31;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            // subnormal: normalize the mantissa
            let mut e = 127 - 15 + 1;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | ((e as u32) << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

#[cfg(test)]
pub mod tests {
    use super::*;

    /// a tiny GGUF v3 image in memory: kv pairs as (key, type, payload bytes),
    /// tensors as (name, dims, type, data bytes); alignment 32
    pub fn build(kvs: &[(&str, u32, Vec<u8>)], tensors: &[(&str, Vec<u64>, u32, Vec<u8>)]) -> Vec<u8> {
        let s = |b: &mut Vec<u8>, x: &str| {
            b.extend((x.len() as u64).to_le_bytes());
            b.extend(x.as_bytes());
        };
        let mut b = Vec::new();
        b.extend(b"GGUF");
        b.extend(3u32.to_le_bytes());
        b.extend((tensors.len() as u64).to_le_bytes());
        b.extend((kvs.len() as u64).to_le_bytes());
        for (k, t, payload) in kvs {
            s(&mut b, k);
            b.extend(t.to_le_bytes());
            b.extend(payload);
        }
        let mut rel = 0u64;
        let mut offs = Vec::new();
        for (name, dims, t, data) in tensors {
            s(&mut b, name);
            b.extend((dims.len() as u32).to_le_bytes());
            for d in dims {
                b.extend(d.to_le_bytes());
            }
            b.extend(t.to_le_bytes());
            b.extend(rel.to_le_bytes());
            offs.push(rel);
            rel += (data.len() as u64).div_ceil(32) * 32;
        }
        while b.len() % 32 != 0 {
            b.push(0);
        }
        let data0 = b.len();
        for ((_, _, _, data), off) in tensors.iter().zip(offs) {
            b.resize(data0 + off as usize, 0);
            b.extend(data);
        }
        b
    }

    pub fn str_payload(x: &str) -> Vec<u8> {
        let mut b = (x.len() as u64).to_le_bytes().to_vec();
        b.extend(x.as_bytes());
        b
    }

    #[test]
    fn the_header_the_tensor_table_and_the_data_offsets_parse() {
        let arr: Vec<u8> = [7u32.to_le_bytes().to_vec(), 3u64.to_le_bytes().to_vec(), vec![1, 0, 1]].concat();
        let img = build(
            &[
                ("clip.projector_type", 8, str_payload("qwen3vl_merger")),
                ("clip.vision.block_count", 4, 27u32.to_le_bytes().to_vec()),
                ("clip.vision.is_deepstack_layers", 9, arr),
            ],
            &[
                ("a.bias", vec![3], GGML_TYPE_F32, [1f32, 2.0, 3.0].iter().flat_map(|v| v.to_le_bytes()).collect()),
                ("a.weight", vec![2, 3], GGML_TYPE_F16, [0x3c00u16, 0xc000, 0, 0x3800, 0x7bff, 1].iter().flat_map(|v| v.to_le_bytes()).collect()),
            ],
        );
        let path = std::env::temp_dir().join(format!("crow-gguf-test-{}.gguf", std::process::id()));
        std::fs::write(&path, &img).unwrap();
        let g = Gguf::open(path.to_str().unwrap()).unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.kv_str("clip.projector_type"), Some("qwen3vl_merger"));
        assert_eq!(g.kv_u64("clip.vision.block_count"), Some(27));
        assert_eq!(g.kv.get("clip.vision.is_deepstack_layers"), Some(&Value::Array(3)));
        let t = g.find("a.weight").unwrap().clone();
        assert_eq!(t.dims, vec![2, 3]);
        assert_eq!(t.offset % 32, 0);
        let raw = g.read_raw(&t).unwrap();
        let h: Vec<f32> = raw.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes([c[0], c[1]]))).collect();
        assert_eq!(h, vec![1.0, -2.0, 0.0, 0.5, 65504.0, 5.960_464_5e-8]);
        let bias = g.read_raw(g.find("a.bias").unwrap()).unwrap();
        assert_eq!(f32::from_le_bytes(bias[8..12].try_into().unwrap()), 3.0);
        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn a_truncated_or_foreign_file_is_refused_by_name() {
        let img = build(&[], &[("w", vec![4], GGML_TYPE_F32, vec![0u8; 16])]);
        // cut into the data section: the tensor runs past the end
        let short = &img[..img.len() - 4];
        let e = Gguf::parse(short, short.len() as u64, "x.gguf").err().unwrap();
        assert!(e.contains("runs past the file end"), "{e}");
        let e = Gguf::parse(&b"GGML\x03\0\0\0"[..], 8, "y.gguf").err().unwrap();
        assert!(e.contains("not a GGUF file"), "{e}");
        let e = Gguf::parse(&img[..20], 20, "z.gguf").err().unwrap();
        assert!(e.contains("truncated header"), "{e}");
    }

    #[test]
    fn f16_decodes_exactly_including_subnormals_inf_and_nan() {
        assert_eq!(f16_to_f32(0x0000), 0.0);
        assert_eq!(f16_to_f32(0x8000).to_bits(), (-0.0f32).to_bits());
        assert_eq!(f16_to_f32(0x3555), 0.333_251_95);
        assert_eq!(f16_to_f32(0x0001), 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x03ff), 1023.0 * 2f32.powi(-24));
        assert_eq!(f16_to_f32(0x0400), 2f32.powi(-14));
        assert_eq!(f16_to_f32(0x7c00), f32::INFINITY);
        assert!(f16_to_f32(0x7e00).is_nan());
        // the positive finite codes 0x0000..=0x7bff map strictly increasing (no
        // two codes collide, no gap folds back), and the negatives mirror them
        for h in 0u16..0x7bff {
            assert!(f16_to_f32(h) < f16_to_f32(h + 1), "h {h:#06x}");
            assert_eq!(f16_to_f32(h | 0x8000), -f16_to_f32(h));
        }
        // the value formula: (-1)^s * 2^(e-15) * (1 + m/1024), e >= 1
        assert_eq!(f16_to_f32(0x3c01), 1.0 + 1.0 / 1024.0);
        assert_eq!(f16_to_f32(0x7bff), 65504.0);
    }
}
