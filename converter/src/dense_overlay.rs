//! `converter dense-overlay` — #77, step 3 of the requant series. ADDITIVE: it writes a
//! NEW file and never touches the conversion path, so `converter <model-dir> <out.cnq>` and
//! `converter requant-check` read and emit exactly what they did before.
//!
//! What it builds: an OVERLAY container in the SAME CNQ1 format (magic, blob, index trailer)
//! that holds ONLY the dense text tensors, as `dtype bf16`. The engine opens it BESIDE the
//! 104.73 GB base container (`CROW_CNQ_OVERLAY`) and a tensor named in the overlay shadows the
//! base tensor of the same name and section. No second 105 GB file, and the default path is
//! untouched because an engine without the overlay never reads this file.
//!
//! Two sources, and the second one is the whole reason the first can be trusted:
//!
//! - `--from-originals <dense.safetensors>` — the BF16 originals of #76, byte for byte. The
//!   overlay carries the ORIGINAL weights, which is what the experiment is about.
//! - `--from-container <base.cnq>` — the CONTROL. The base container's own NVFP4 values,
//!   dequantized to f32 and rounded to BF16. It carries NO new information: every value is one
//!   the FP4 kernels would have decoded anyway (up to the f32 -> bf16 rounding, which is
//!   reported). An engine that answers differently under THIS overlay has a wiring difference,
//!   not a weight difference.
//!
//! `--kinds a,b,c` restricts the overlay to some of the 17 dense kinds, so a later ticket can
//! ablate per kind without a new tool. The kind of a tensor is its name with the
//! `model.language_model.layers.<N>.` prefix and the `.weight` suffix removed.
//!
//! The index trailer carries an `overlay` block: base container name and byte size (the engine
//! refuses an overlay built against a different base), the source, the kinds, the counts and
//! the build date.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::{decode_e2m1, decode_ue4m3, read_safetensors_header, MAGIC};

pub const HELP: &str = "usage: converter dense-overlay --base <container.cnq> --out <overlay.cnq> \\\n                              (--from-originals <dense.safetensors> | --from-container <base.cnq>) \\\n                              [--kinds k1,k2,...]\n  builds a bf16 OVERLAY container over the dense text tensors of <container.cnq> (#77)\n  --from-originals  the BF16 originals fetched by tools/fetch-dense-originals.py (#76)\n  --from-container  the CONTROL: the container's own NVFP4 values dequantized to bf16\n  --kinds           a subset of the 17 dense kinds (default: all of them)";

/// One tensor as the base container's index trailer describes it.
pub struct BaseEntry {
    pub name: String,
    pub section: String,
    pub dtype: String,
    pub offset: u64,
    pub len: u64,
    pub n_values: usize,
    pub global_scale: f32,
    pub shape: Vec<u64>,
}

/// The base container's index. Same trailer rule as `requant_check`: the last 8 bytes are the
/// u64 length of the JSON that sits in front of them.
pub fn read_base_index(path: &Path) -> std::io::Result<Vec<BaseEntry>> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::other(format!("{}: magic is {magic:?}, not CNQ1", path.display())));
    }
    let trailer = f.seek(SeekFrom::End(-8))?;
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let index_len = u64::from_le_bytes(len_buf);
    f.seek(SeekFrom::Start(trailer - index_len))?;
    let mut buf = vec![0u8; index_len as usize];
    f.read_exact(&mut buf)?;
    let index: serde_json::Value = serde_json::from_slice(&buf)?;
    let blob_offset = index["blob_offset"].as_u64().unwrap_or(12);
    if blob_offset != 12 {
        return Err(std::io::Error::other(format!(
            "{}: blob_offset is {blob_offset}, this reader knows 12",
            path.display()
        )));
    }
    let mut out = Vec::new();
    for t in index["tensors"].as_array().into_iter().flatten() {
        out.push(BaseEntry {
            name: t["name"].as_str().unwrap_or("").to_string(),
            section: t["section"].as_str().unwrap_or("").to_string(),
            dtype: t["dtype"].as_str().unwrap_or("").to_string(),
            offset: t["offset"].as_u64().unwrap_or(0),
            len: t["len"].as_u64().unwrap_or(0),
            n_values: t["n_values"].as_u64().unwrap_or(0) as usize,
            global_scale: t["global_scale"].as_f64().unwrap_or(1.0) as f32,
            shape: t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default(),
        });
    }
    Ok(out)
}

/// The #76 selection rule, derived from the container's own index: a text-section NVFP4 tensor
/// that is not a routed expert. 495 of them in CNQ4.5-M.
pub fn is_dense_text(e: &BaseEntry) -> bool {
    e.section == "text" && e.dtype == "nvfp4" && !e.name.contains(".mlp.experts.")
}

/// The KIND of a dense tensor: its name without the per-layer prefix and without `.weight`.
/// `model.language_model.layers.7.linear_attn.in_proj_qkv.weight` -> `linear_attn.in_proj_qkv`.
pub fn kind_of(name: &str) -> String {
    let mut s = name;
    if let Some(rest) = s.strip_prefix("model.language_model.layers.") {
        if let Some(dot) = rest.find('.') {
            if rest[..dot].bytes().all(|b| b.is_ascii_digit()) {
                s = &rest[dot + 1..];
            }
        }
    }
    s.strip_suffix(".weight").unwrap_or(s).to_string()
}

/// f32 -> bf16 bits, round to nearest with ties to even. The base container's own `bf16` keeps
/// were never rounded (they are the original bytes), so this rule is only ever applied to the
/// CONTROL overlay, where it is the one and only place information is lost.
pub fn f32_to_bf16_rne(v: f32) -> u16 {
    let bits = v.to_bits();
    if (bits & 0x7F80_0000) == 0x7F80_0000 && (bits & 0x007F_FFFF) != 0 {
        // NaN: keep it a NaN with a non-zero payload after the shift
        return ((bits >> 16) | 0x0040) as u16;
    }
    let rounding = 0x7FFF + ((bits >> 16) & 1);
    ((bits.wrapping_add(rounding)) >> 16) as u16
}

/// NVFP4 blocks -> bf16 little-endian bytes. 36 B per 64 values: 4 ue4m3 sub-block scale bytes
/// then 32 B of LSB-first packed E2M1 nibbles; value = e2m1 * ue4m3 * global. This is the same
/// arithmetic `cnq::dequant_block` runs in the engine and `gemv_fp4_b` runs on the card.
/// `pub` since #91: `layer_rule_overlay` re-uses it for its `--from-container` control, so the
/// two controls dequant by the one arithmetic, not two copies of it.
pub fn blocks_to_bf16(raw: &[u8], global: f32, n_values: usize, out: &mut Vec<u8>) -> u64 {
    out.clear();
    out.reserve(n_values * 2);
    let mut inexact = 0u64;
    for blk in raw.chunks_exact(36) {
        for idx in 0..64usize {
            if out.len() / 2 >= n_values {
                break;
            }
            let s = decode_ue4m3(blk[idx >> 4] as u32) * global;
            let byte = blk[4 + (idx >> 1)] as u32;
            let nib = if idx & 1 == 1 { (byte >> 4) & 0xF } else { byte & 0xF };
            let v = decode_e2m1(nib) * s;
            let b = f32_to_bf16_rne(v);
            if f32::from_bits((b as u32) << 16) != v {
                inexact += 1;
            }
            out.extend_from_slice(&b.to_le_bytes());
        }
    }
    inexact
}

enum Source {
    Originals(PathBuf),
    Container(PathBuf),
}

/// `YYYY-MM-DDTHH:MM:SSZ` from the wall clock, with the Gregorian calendar spelled out — the
/// converter carries no date crate and the overlay header has to say when it was built.
pub fn utc_date_string() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0);
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let (h, mi, s) = (rem / 3600, (rem % 3600) / 60, rem % 60);
    // civil_from_days (Howard Hinnant's algorithm), era-based, no leap-second table
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = if m <= 2 { y + 1 } else { y };
    format!("{y:04}-{m:02}-{d:02}T{h:02}:{mi:02}:{s:02}Z")
}

pub fn run(args: &[String]) -> i32 {
    let mut base: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut source: Option<Source> = None;
    let mut kinds_arg: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base" => base = it.next().map(PathBuf::from),
            "--out" => out_path = it.next().map(PathBuf::from),
            "--from-originals" => source = it.next().map(|p| Source::Originals(PathBuf::from(p))),
            "--from-container" => source = it.next().map(|p| Source::Container(PathBuf::from(p))),
            "--kinds" => kinds_arg = it.next().cloned(),
            s => {
                eprintln!("unexpected argument {s}\n{HELP}");
                return 2;
            }
        }
    }
    let (Some(base), Some(out_path), Some(source)) = (base, out_path, source) else {
        eprintln!("{HELP}");
        return 2;
    };
    let base_len = match std::fs::metadata(&base) {
        Ok(m) => m.len(),
        Err(e) => {
            eprintln!("{}: {e}", base.display());
            return 2;
        }
    };
    let index = match read_base_index(&base) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("base container index: {e}");
            return 2;
        }
    };
    let dense: Vec<&BaseEntry> = index.iter().filter(|e| is_dense_text(e)).collect();
    let all_kinds: std::collections::BTreeSet<String> = dense.iter().map(|e| kind_of(&e.name)).collect();
    println!(
        "base      {}: {} B, {} tensors, {} dense text nvfp4 in {} kinds",
        base.display(),
        base_len,
        index.len(),
        dense.len(),
        all_kinds.len()
    );

    // --kinds: every name must be one the BASE actually carries. A typo that silently
    // produced an empty overlay would be measured as "the originals change nothing".
    let wanted: std::collections::BTreeSet<String> = match &kinds_arg {
        None => all_kinds.clone(),
        Some(s) => {
            let asked: Vec<String> = s.split(',').map(|k| k.trim().to_string()).filter(|k| !k.is_empty()).collect();
            let unknown: Vec<&String> = asked.iter().filter(|k| !all_kinds.contains(*k)).collect();
            if asked.is_empty() || !unknown.is_empty() {
                eprintln!("--kinds: unknown kind(s) {unknown:?}; the base carries:");
                for k in &all_kinds {
                    eprintln!("  {k}");
                }
                return 2;
            }
            asked.into_iter().collect()
        }
    };
    let selected: Vec<&BaseEntry> = dense.into_iter().filter(|e| wanted.contains(&kind_of(&e.name))).collect();
    if selected.is_empty() {
        eprintln!("no tensor selected — refusing to write an empty overlay");
        return 2;
    }

    // ---- the source side ----
    let (source_str, mut originals): (String, std::collections::BTreeMap<String, (u64, u64, String, usize)>) = match &source {
        Source::Originals(p) => {
            let (header, data_start) = match read_safetensors_header(p) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{}: {e}", p.display());
                    return 2;
                }
            };
            let mut map = std::collections::BTreeMap::new();
            for (name, info) in header.as_object().into_iter().flatten() {
                if name == "__metadata__" {
                    continue;
                }
                let dt = info["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> =
                    info["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
                let n: usize = shape.iter().product();
                let begin = info["data_offsets"][0].as_u64().unwrap_or(0);
                let end = info["data_offsets"][1].as_u64().unwrap_or(0);
                map.insert(name.clone(), (data_start + begin, data_start + end, dt, n));
            }
            (format!("originals:{}", p.display()), map)
        }
        Source::Container(p) => {
            if std::fs::canonicalize(p).ok() != std::fs::canonicalize(&base).ok() {
                eprintln!(
                    "--from-container {} is not the same file as --base {} — the control overlay must \
be built from the container it shadows",
                    p.display(),
                    base.display()
                );
                return 2;
            }
            ("container-dequant".to_string(), std::collections::BTreeMap::new())
        }
    };
    if let Source::Originals(p) = &source {
        let missing: Vec<&str> = selected.iter().map(|e| e.name.as_str()).filter(|n| !originals.contains_key(*n)).collect();
        if !missing.is_empty() {
            eprintln!("{} selected tensors are not in {}:", missing.len(), p.display());
            for n in missing.iter().take(10) {
                eprintln!("  {n}");
            }
            return 1;
        }
    }

    // ---- write the overlay: magic, blob, index trailer (the CNQ1 layout verbatim) ----
    let t0 = std::time::Instant::now();
    let mut out = match std::fs::File::create(&out_path) {
        Ok(f) => std::io::BufWriter::with_capacity(8 << 20, f),
        Err(e) => {
            eprintln!("{}: {e}", out_path.display());
            return 2;
        }
    };
    out.write_all(MAGIC).expect("magic");
    out.write_all(&0u64.to_le_bytes()).expect("reserved");

    let mut src_file = match &source {
        Source::Originals(p) => std::fs::File::open(p).expect("open originals"),
        Source::Container(p) => std::fs::File::open(p).expect("open base container"),
    };
    let mut index_tensors: Vec<serde_json::Value> = Vec::new();
    let mut blob_len: u64 = 0;
    let mut values_total: u64 = 0;
    let mut inexact_total: u64 = 0;
    let mut per_kind: std::collections::BTreeMap<String, (usize, u64)> = std::collections::BTreeMap::new();
    let mut deq = Vec::new();
    let mut raw = Vec::new();

    for (i, e) in selected.iter().enumerate() {
        let n = e.n_values;
        let bf16: Vec<u8> = match &source {
            Source::Originals(p) => {
                let (begin, end, dt, sn) = originals.remove(&e.name).expect("checked above");
                if sn != n {
                    eprintln!("{}: {sn} values in {}, {n} in the base container", e.name, p.display());
                    return 1;
                }
                if dt != "BF16" {
                    eprintln!("{}: dtype {dt} in {} — this overlay stores bf16 only", e.name, p.display());
                    return 1;
                }
                raw.resize((end - begin) as usize, 0);
                src_file.seek(SeekFrom::Start(begin)).expect("seek originals");
                src_file.read_exact(&mut raw).expect("read original tensor");
                raw.clone()
            }
            Source::Container(_) => {
                let len = e.len.max(((n as u64) + 63) / 64 * 36);
                raw.resize(len as usize, 0);
                src_file.seek(SeekFrom::Start(12 + e.offset)).expect("seek base blob");
                src_file.read_exact(&mut raw).expect("read base tensor");
                inexact_total += blocks_to_bf16(&raw, e.global_scale, n, &mut deq);
                deq.clone()
            }
        };
        if bf16.len() != n * 2 {
            eprintln!("{}: {} bytes for {n} bf16 values", e.name, bf16.len());
            return 1;
        }
        out.write_all(&bf16).expect("write overlay tensor");
        index_tensors.push(serde_json::json!({
            "name": e.name, "shape": e.shape, "section": e.section,
            "n_values": n, "offset": blob_len, "dtype": "bf16", "len": bf16.len() as u64,
        }));
        blob_len += bf16.len() as u64;
        values_total += n as u64;
        let slot = per_kind.entry(kind_of(&e.name)).or_insert((0, 0));
        slot.0 += 1;
        slot.1 += n as u64;
        if i % 100 == 0 {
            eprintln!("[{:>4}/{}] {} — {:.2} GB written", i + 1, selected.len(), e.name, blob_len as f64 / 1e9);
        }
    }

    let built = utc_date_string();
    let index = serde_json::json!({
        "format": "crow-nest-quant", "version": 1,
        "overlay": {
            "kind": "dense-bf16",
            "issue": 77,
            "base_name": base.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            "base_bytes": base_len,
            "source": source_str,
            "kinds": wanted.iter().cloned().collect::<Vec<String>>(),
            "tensors": index_tensors.len(),
            "values": values_total,
            "bytes": blob_len,
            "bf16_inexact_values": inexact_total,
            "built": built,
        },
        "blob_offset": 12u64,
        "tensors": index_tensors,
    });
    let index_json = serde_json::to_vec_pretty(&index).expect("index json");
    out.write_all(&index_json).expect("index");
    out.write_all(&(index_json.len() as u64).to_le_bytes()).expect("index len");
    out.flush().ok();
    drop(out);

    println!("overlay   {}: {} tensors, {} values, payload {:.2} GB, source {source_str}",
        out_path.display(), index_tensors.len(), values_total, blob_len as f64 / 1e9);
    for (k, (c, n)) in &per_kind {
        println!("  {c:>3} x {k}  ({n} values)");
    }
    if matches!(source, Source::Container(_)) {
        println!(
            "  control: {inexact_total} of {values_total} dequantized values ({:.4} %) did not fit bf16 exactly and were rounded to nearest-even",
            inexact_total as f64 * 100.0 / values_total.max(1) as f64
        );
    }
    println!("dense-overlay: done in {:.0} s", t0.elapsed().as_secs_f64());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_kind_is_the_name_without_the_layer_prefix_and_the_weight_suffix() {
        assert_eq!(kind_of("model.language_model.layers.7.linear_attn.in_proj_qkv.weight"), "linear_attn.in_proj_qkv");
        assert_eq!(kind_of("model.language_model.layers.0.self_attn.indexer.index_qk_proj.weight"), "self_attn.indexer.index_qk_proj");
        assert_eq!(kind_of("model.language_model.layers.1.ple.key_proj.weight"), "ple.key_proj");
        assert_eq!(kind_of("model.language_model.layers.47.mlp.shared_expert.down_proj.weight"), "mlp.shared_expert.down_proj");
        // a name that does not carry the per-layer prefix keeps its own shape
        assert_eq!(kind_of("model.language_model.hyper_connection_mixer.input_mix_weight_up.weight"), "model.language_model.hyper_connection_mixer.input_mix_weight_up");
    }

    #[test]
    fn bf16_rounding_is_nearest_even_and_exact_when_it_can_be() {
        // exactly representable: the low 16 bits are zero
        for v in [0.0f32, 1.0, -2.5, 6.0, 1.5258789e-5] {
            let b = f32_to_bf16_rne(v);
            assert_eq!(f32::from_bits((b as u32) << 16), v, "{v} should survive bf16");
        }
        // a tie rounds to the even bf16, not away from zero
        let tie = f32::from_bits(0x3F80_8000); // 1.0 + half a bf16 ulp
        assert_eq!(f32_to_bf16_rne(tie), 0x3F80);
        let tie_up = f32::from_bits(0x3F81_8000); // 1.0 + 1.5 bf16 ulp
        assert_eq!(f32_to_bf16_rne(tie_up), 0x3F82);
        // truncation (the >> 16 the engine's bf16 twin uses) would give 0x3F80 for both
        assert_ne!(f32_to_bf16_rne(tie_up), 0x3F80);
    }

    #[test]
    fn the_control_dequant_is_the_engines_block_walk() {
        // one block: sub-block scales 0x38 0x38 0x38 0x38 (= 1.0), nibbles 0,1,2,...
        let mut blk = vec![0u8; 36];
        for sb in 0..4 {
            blk[sb] = 0x38;
        }
        for i in 0..32 {
            blk[4 + i] = 0x21; // low nibble 1 (0.5), high nibble 2 (1.0)
        }
        let mut out = Vec::new();
        let inexact = blocks_to_bf16(&blk, 1.0, 64, &mut out);
        assert_eq!(out.len(), 128);
        assert_eq!(inexact, 0, "0.5 and 1.0 are exact in bf16");
        let first = f32::from_bits((u16::from_le_bytes([out[0], out[1]]) as u32) << 16);
        let second = f32::from_bits((u16::from_le_bytes([out[2], out[3]]) as u32) << 16);
        assert_eq!(decode_ue4m3(0x38), 1.0);
        assert_eq!(first, 0.5);
        assert_eq!(second, 1.0);
    }

    #[test]
    fn a_short_tensor_stops_at_its_value_count() {
        // 64-value block, but the tensor claims 16 values: the walk stops there
        let mut blk = vec![0u8; 36];
        for sb in 0..4 {
            blk[sb] = 0x38;
        }
        for i in 0..32 {
            blk[4 + i] = 0x22;
        }
        let mut out = Vec::new();
        blocks_to_bf16(&blk, 1.0, 16, &mut out);
        assert_eq!(out.len(), 32);
    }
}
