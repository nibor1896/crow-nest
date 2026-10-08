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

use std::io::{BufRead, Read, Seek, SeekFrom, Write};

pub const HELP: &str = "usage: converter dequant <container.cnq> (<name>[:<r0>:<r1>] ... | --names -)\n  writes each named tensor (or rows r0..r1 of a 2-D tensor) to stdout as f32 little endian, no header,\n  in the order given; nvfp4 decoded as gate 0 decodes it, bf16 widened exactly, f32 as stored (crow-nest #156)";

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
