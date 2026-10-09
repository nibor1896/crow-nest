//! `converter dequant` (crow-nest #156, GLM-5.3-Flash plan step 6): the decoder of a written
//! container, read-only.
//!
//! It writes the named tensors to stdout as f32 little endian, one after the other, in the order
//! asked, with no header: NVFP4 through `dequant_nvfp4` (the arithmetic gate 0 measures the
//! written encoding with, `nvfp4_scale` / `nvfp4_value` in `main.rs`), BF16 widened exactly, F32
//! as stored. A name may carry a row range `name:r0:r1` (a 2-D tensor's rows r0..r1; NVFP4 rows
//! must be whole 64-value blocks). `--names -` reads the names from stdin, one per line, so a
//! whole layer (864 routed-expert tensors) is one call.
//!
//! The oracle's `--weights container` back end (`oracle/glm5_common.py`, `ContainerSource`) reads
//! every container weight through this subcommand, so the reference and gate 0 decode the same
//! bytes with the same code.
//!
//! MUL1 (GLM-5.3-Flash's routed experts, index dtype `mul1`, #181/#182): the entry's record (one
//! 4096-aligned gate|up|down unit, `mul1.record_offset` / `mul1.record_bytes`) is read whole and
//! parsed by `mul1::read_record`; the projection's trellis is decoded by the #181 reference decoder
//! (`mul1::decode_tile`, byte-identical to exllamav3 v1.6.0 `reconstruct`) to the rotated fp16
//! W_hat `[in, out]`, and the original-basis weight is exllamav3's `get_weight_tensor`,
//! `W = diag(suh) H W_hat H diag(svh) / 128` (H the 128-wide Sylvester Hadamard), computed in f64
//! and rounded to f32 once (exllamav3 rounds to fp16 between the steps; this does not). Out goes
//! the checkpoint layout `[out, in]` = W^T, so a row range is a range of output features. Up to 16
//! threads per projection, each on whole 128-column blocks; the bits do not depend on the thread
//! count.

use std::io::{BufRead, Read, Seek, SeekFrom, Write};

pub const HELP: &str = "usage: converter dequant <container.cnq> (<name>[:<r0>:<r1>] ... | --names -)\n  writes each named tensor (or rows r0..r1 of a 2-D tensor) to stdout as f32 little endian, no header,\n  in the order given; nvfp4 decoded as gate 0 decodes it, bf16 widened exactly, f32 as stored (crow-nest #156);\n  mul1 (MUL1 K=3 expert records) decoded by the #181 reference decoder to the original-basis weight\n  diag(suh) H W_hat H diag(svh) / 128 in f64, rounded to f32 once, in the checkpoint's [out, in] layout";

/// One request: a tensor name and an optional row range.
fn parse_spec(spec: &str) -> Result<(String, Option<(usize, usize)>), String> {
    let parts: Vec<&str> = spec.split(':').collect();
    match parts.as_slice() {
        [n] => Ok((n.to_string(), None)),
        [n, a, b] => {
            let r0 = a.parse::<usize>().map_err(|_| format!("{spec}: row `{a}` is not a number"))?;
            let r1 = b.parse::<usize>().map_err(|_| format!("{spec}: row `{b}` is not a number"))?;
            if r0 > r1 {
                return Err(format!("{spec}: rows run backwards"));
            }
            Ok((n.to_string(), Some((r0, r1))))
        }
        _ => Err(format!("{spec}: expected <name> or <name>:<r0>:<r1>")),
    }
}

/// The byte range of the request inside the blob and the f32 values it decodes to.
pub(crate) fn decode(f: &mut std::fs::File, blob_offset: u64, t: &serde_json::Value, rows: Option<(usize, usize)>) -> Result<Vec<f32>, String> {
    let name = t["name"].as_str().unwrap_or("?");
    let dtype = t["dtype"].as_str().unwrap_or("?");
    if dtype == crate::mul1::DTYPE {
        return decode_mul1(f, blob_offset, t, rows);
    }
    let n = t["n_values"].as_u64().ok_or(format!("{name}: no n_values"))? as usize;
    let shape: Vec<usize> = t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
    let (v0, v1) = match rows {
        None => (0, n),
        Some((r0, r1)) => {
            if shape.len() != 2 || r1 > shape[0] {
                return Err(format!("{name}: rows {r0}..{r1} of shape {shape:?}"));
            }
            (r0 * shape[1], r1 * shape[1])
        }
    };
    let (b0, b1) = match dtype {
        "nvfp4" => {
            if v0 % 64 != 0 || v1 % 64 != 0 {
                return Err(format!("{name}: rows of {:?} are not whole 64-value NVFP4 blocks", shape));
            }
            (v0 / 64 * 36, v1.div_ceil(64) * 36)
        }
        "bf16" => (v0 * 2, v1 * 2),
        "f32" => (v0 * 4, v1 * 4),
        other => return Err(format!("{name}: dtype {other} is not decoded to f32 here")),
    };
    let off = t["offset"].as_u64().ok_or(format!("{name}: no offset"))?;
    let len = t["len"].as_u64().ok_or(format!("{name}: no len"))?;
    if b1 as u64 > len {
        return Err(format!("{name}: [{b0}, {b1}) beyond its {len} B"));
    }
    f.seek(SeekFrom::Start(blob_offset + off + b0 as u64)).map_err(|e| format!("{name}: seek: {e}"))?;
    let mut raw = vec![0u8; b1 - b0];
    f.read_exact(&mut raw).map_err(|e| format!("{name}: read: {e}"))?;
    Ok(match dtype {
        "nvfp4" => {
            let g = t["global_scale"].as_f64().ok_or(format!("{name}: no global_scale"))? as f32;
            let mut v = crate::dequant_nvfp4(&raw, g);
            v.truncate(v1 - v0); // a tensor whose length is not a multiple of 64 ends inside its last block
            v
        }
        "bf16" => crate::bytes_to_f32(&raw, "BF16"),
        _ => crate::bytes_to_f32(&raw, "F32"),
    })
}

/// The most threads one MUL1 projection is decoded with.
const MUL1_MAX_THREADS: usize = 16;

/// A MUL1 index entry: its record read whole, the projection decoded to the original basis.
fn decode_mul1(f: &mut std::fs::File, blob_offset: u64, t: &serde_json::Value, rows: Option<(usize, usize)>) -> Result<Vec<f32>, String> {
    use crate::mul1::{read_record, Bitrate, RecordLayout};
    let name = t["name"].as_str().unwrap_or("?");
    let shape: Vec<usize> = t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
    let (_, _, proj) = crate::recipe::glm_expert(name).ok_or(format!("{name}: a mul1 entry that is not a routed-expert projection"))?;
    if shape.len() != 2 {
        return Err(format!("{name}: mul1 shape {shape:?} is not 2-D"));
    }
    // the checkpoint shape is [out, in]: gate/up [inter, hidden], down [hidden, inter]
    let (hidden, inter, which) = match proj {
        "gate" => (shape[1], shape[0], 0),
        "up" => (shape[1], shape[0], 1),
        _ => (shape[0], shape[1], 2),
    };
    let m = &t["mul1"];
    let k = m["k"].as_f64().ok_or(format!("{name}: no mul1.k"))?;
    let rec_off = m["record_offset"].as_u64().ok_or(format!("{name}: no mul1.record_offset"))?;
    let rec_bytes = m["record_bytes"].as_u64().ok_or(format!("{name}: no mul1.record_bytes"))?;
    let lay = RecordLayout::new(hidden, inter, Bitrate::from_k(k)?)?;
    if lay.size != rec_bytes {
        return Err(format!("{name}: record of {rec_bytes} B, a [{hidden}, {inter}] K = {k} record is {} B", lay.size));
    }
    let (r0, r1) = rows.unwrap_or((0, shape[0]));
    if r1 > shape[0] {
        return Err(format!("{name}: rows {r0}..{r1} of shape {shape:?}"));
    }
    f.seek(SeekFrom::Start(blob_offset + rec_off)).map_err(|e| format!("{name}: seek: {e}"))?;
    let mut rec = vec![0u8; rec_bytes as usize];
    f.read_exact(&mut rec).map_err(|e| format!("{name}: read: {e}"))?;
    let [g, u, d] = read_record(&lay, &rec)?;
    let lin = [&g, &u, &d][which];
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).min(MUL1_MAX_THREADS);
    Ok(mul1_original_basis(lin, lay.bitrate, r0, r1, threads))
}

/// Natural-order (Sylvester) Walsh-Hadamard butterflies over 128 rows of `width` values each
/// (`x` row major), unnormalized: row i becomes sum_a H[i][a] row a.
fn fwht128_rows(x: &mut [f64], width: usize) {
    let mut h = 1;
    while h < 128 {
        for i in (0..128).step_by(2 * h) {
            for j in i..i + h {
                let (lo, hi) = x.split_at_mut((j + h) * width);
                for (a, b) in lo[j * width..(j + 1) * width].iter_mut().zip(&mut hi[..width]) {
                    let (p, q) = (*a, *b);
                    *a = p + q;
                    *b = p - q;
                }
            }
        }
        h *= 2;
    }
}

/// Rows `r0..r1` of the checkpoint-layout weight `[out = lin.n, in = lin.k]` of one MUL1 linear:
/// `W^T` with `W = diag(suh) H W_hat H diag(svh) / 128`, f64 throughout, one f32 rounding.
/// The work is split by 128-wide output blocks (the right Hadamard's) over `threads` workers.
pub(crate) fn mul1_original_basis(lin: &crate::mul1::Linear, b: crate::mul1::Bitrate, r0: usize, r1: usize, threads: usize) -> Vec<f32> {
    use crate::mul1::{decode_tile, f16_to_f64};
    let (k, n) = (lin.k, lin.n);
    assert!(k % 128 == 0 && n % 128 == 0 && r0 <= r1 && r1 <= n, "MUL1 [{k}, {n}] rows {r0}..{r1}");
    let (b0, b1) = (r0 / 128, r1.div_ceil(128));
    let mut full = vec![0.0f32; (b1 - b0) * 128 * k];
    let wpt = b.words_per_tile();
    let suh: Vec<f64> = lin.suh.iter().map(|&h| f16_to_f64(h)).collect();
    let svh: Vec<f64> = lin.svh.iter().map(|&h| f16_to_f64(h)).collect();
    let block = |cb: usize, out: &mut [f32]| {
        // W_hat[:, 128 cb .. 128 cb + 128] transposed: wt[c][i], [128][k]
        let mut wt = vec![0.0f64; 128 * k];
        for kb in 0..k / 16 {
            for nbl in 0..8 {
                let t = kb * (n / 16) + cb * 8 + nbl;
                let tile = decode_tile(&lin.trellis[t * wpt..(t + 1) * wpt], b);
                for r in 0..16 {
                    for c in 0..16 {
                        wt[(nbl * 16 + c) * k + kb * 16 + r] = f16_to_f64(tile[r * 16 + c]);
                    }
                }
            }
        }
        // right Hadamard (over the 128 columns c, each a row of wt), then the left one (over each
        // 128-wide block of i, on the transposed rows: wt^T row blocks)
        fwht128_rows(&mut wt, k);
        let mut col = vec![0.0f64; 128 * 128];
        for ib in 0..k / 128 {
            for c in 0..128 {
                for a in 0..128 {
                    col[a * 128 + c] = wt[c * k + ib * 128 + a];
                }
            }
            fwht128_rows(&mut col, 128);
            for c in 0..128 {
                let j = cb * 128 + c;
                for a in 0..128 {
                    let i = ib * 128 + a;
                    out[c * k + i] = (col[a * 128 + c] * suh[i] * svh[j] / 128.0) as f32;
                }
            }
        }
    };
    let mut work: Vec<(usize, &mut [f32])> = full.chunks_mut(128 * k).enumerate().map(|(i, c)| (b0 + i, c)).collect();
    let per = work.len().div_ceil(threads.max(1)).max(1);
    std::thread::scope(|s| {
        for group in work.chunks_mut(per) {
            let block = &block;
            s.spawn(move || {
                for (cb, out) in group.iter_mut() {
                    block(*cb, out);
                }
            });
        }
    });
    let start = (r0 - b0 * 128) * k;
    full.truncate((r1 - b0 * 128) * k);
    full.drain(..start);
    full
}

pub fn run(args: &[String]) -> i32 {
    let (Some(path), rest) = (args.first(), args.get(1..).unwrap_or(&[])) else {
        eprintln!("{HELP}");
        return 2;
    };
    let mut specs: Vec<String> = Vec::new();
    let mut it = rest.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--names" => match it.next().map(|s| s.as_str()) {
                Some("-") => {
                    for line in std::io::stdin().lock().lines() {
                        match line {
                            Ok(l) if !l.trim().is_empty() => specs.push(l.trim().to_string()),
                            Ok(_) => {}
                            Err(e) => {
                                eprintln!("dequant: stdin: {e}");
                                return 2;
                            }
                        }
                    }
                }
                other => {
                    eprintln!("--names takes `-` (stdin), got {other:?}\n{HELP}");
                    return 2;
                }
            },
            s if s.starts_with("--") => {
                eprintln!("unknown flag {s}\n{HELP}");
                return 2;
            }
            s => specs.push(s.to_string()),
        }
    }
    if specs.is_empty() {
        eprintln!("{HELP}");
        return 2;
    }
    let p = std::path::Path::new(path);
    let index = match crate::dense_overlay::read_index_json(p) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("dequant: {}: {e}", p.display());
            return 2;
        }
    };
    let blob_offset = index["blob_offset"].as_u64().unwrap_or(12);
    let by_name: std::collections::HashMap<&str, &serde_json::Value> =
        index["tensors"].as_array().map(|a| a.iter().filter_map(|t| t["name"].as_str().map(|n| (n, t))).collect()).unwrap_or_default();
    // every request is checked before the first byte goes out
    let mut reqs = Vec::with_capacity(specs.len());
    for s in &specs {
        match parse_spec(s) {
            Ok((n, r)) => match by_name.get(n.as_str()) {
                Some(t) => reqs.push((*t, r)),
                None => {
                    eprintln!("dequant: {n}: not in {}", p.display());
                    return 2;
                }
            },
            Err(e) => {
                eprintln!("dequant: {e}");
                return 2;
            }
        }
    }
    let mut f = match std::fs::File::open(p) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("dequant: {}: {e}", p.display());
            return 2;
        }
    };
    let stdout = std::io::stdout();
    let mut out = std::io::BufWriter::with_capacity(1 << 22, stdout.lock());
    for (t, r) in reqs {
        let v = match decode(&mut f, blob_offset, t, r) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("dequant: {e}");
                return 2;
            }
        };
        let bytes: Vec<u8> = v.iter().flat_map(|x| x.to_le_bytes()).collect();
        if let Err(e) = out.write_all(&bytes) {
            eprintln!("dequant: stdout: {e}");
            return 2;
        }
    }
    if let Err(e) = out.flush() {
        eprintln!("dequant: stdout: {e}");
        return 2;
    }
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn specs_parse_with_and_without_rows() {
        assert_eq!(parse_spec("lm_head.weight").unwrap(), ("lm_head.weight".to_string(), None));
        assert_eq!(parse_spec("lm_head.weight:16384:32768").unwrap(), ("lm_head.weight".to_string(), Some((16384, 32768))));
        assert!(parse_spec("a:2:1").is_err());
        assert!(parse_spec("a:1").is_err());
        assert!(parse_spec("a:x:2").is_err());
    }
}
