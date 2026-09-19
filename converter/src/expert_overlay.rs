//! `converter expert-overlay` — #79, step 5 of the requant series. ADDITIVE, exactly as
//! `requant-check` (#76) and `dense-overlay` (#77) are: the word `expert-overlay` is taken off
//! the front of the argument list and the conversion path never sees it.
//!
//! What it builds: an OVERLAY container in the SAME CNQ1 format holding only the ROUTED EXPERT
//! tensors of the chosen layers, as `dtype nvfp4` of the SAME byte length, the SAME shape and
//! the SAME global-scale convention as the base container's. The engine opens it beside the
//! base (`CROW_CNQ_OVERLAY`) and those tensors shadow the base ones — same slabs, same
//! residency plan, same kernels, same VRAM. Only the 36-byte blocks differ.
//!
//! Three rules, and the first one is what makes the other two readable:
//!
//! - `--rule mse` — the CONTROL. The conversion's own `--scales mse`, re-run over the fetched
//!   BF16 originals. #76 proved that reproduces the container's bytes for the dense path;
//!   here the same must hold for the experts, so the overlay is BYTE-IDENTICAL to the base and
//!   an engine that computes anything else under it has a wiring difference.
//! - `--rule imatrix` — llama.cpp's importance-weighted scale search (see `expert_requant`).
//! - `--rule imatrix46` — the same plus NVIDIA's four-over-six candidate.
//!
//! Whatever the rule, every tensor is compared with the base container's bytes as it is
//! written, so the report always says how far the new encoding moved from the shipped one.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::dense_overlay::{read_base_index, BaseEntry};
use crate::expert_requant::{quantize_expert_tensor, ExpertStats, Rule};
use crate::imatrix::{self, ExpertTensor, Gguf};
use crate::{bytes_to_f32, read_safetensors_header, MAGIC};

pub const HELP: &str = "usage: converter expert-overlay --base <container.cnq> --out <overlay.cnq> \\\n                               --originals <dir> --layers 1,7,13,... --rule mse|imatrix|imatrix46 \\\n                               [--imatrix <imatrix.gguf>] [--threads N] [--report <report.json>]\n  builds an nvfp4 OVERLAY container over the routed experts of the given layers (#79)\n  --originals  the directory of tools/fetch-dense-originals.py --experts (layer-NN.safetensors)\n  --rule mse   the CONTROL: the conversion's own unweighted rule, byte-identical to the base\n  --rule imatrix / imatrix46  importance-weighted scales; --imatrix is then required";

/// The routed-expert selection, derived from the container's own index — the mirror of
/// `dense_overlay::is_dense_text`, and the twin of the fetch tool's `select_expert_tensors`.
pub fn is_expert_text(e: &BaseEntry) -> bool {
    e.section == "text" && e.dtype == "nvfp4" && e.name.contains(".mlp.experts.")
}

/// The layer number in `model.language_model.layers.<N>....`, or None.
pub fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<usize>().ok()
}

/// `1,7,13` or `2-5,9` -> a sorted list of distinct layers. The twin of the fetch tool's
/// `parse_layers`, refusals included.
pub fn parse_layers(spec: &str) -> Result<Vec<usize>, String> {
    let mut out: Vec<usize> = Vec::new();
    for part in spec.split(',') {
        let part = part.trim();
        if part.is_empty() {
            continue;
        }
        if let Some((lo, hi)) = part.split_once('-') {
            let lo: usize = lo.trim().parse().map_err(|_| format!("{part:?} is not a layer range"))?;
            let hi: usize = hi.trim().parse().map_err(|_| format!("{part:?} is not a layer range"))?;
            if hi < lo {
                return Err(format!("layer range {part:?} runs backwards"));
            }
            out.extend(lo..=hi);
        } else {
            out.push(part.parse().map_err(|_| format!("{part:?} is not a layer number"))?);
        }
    }
    out.sort_unstable();
    out.dedup();
    if out.is_empty() {
        return Err("no layers selected".into());
    }
    Ok(out)
}

/// Which of the two routed-expert tensors a name is, and the geometry that follows from its
/// shape: `[n_experts, rows_per_expert, n_cols]`.
#[derive(Debug)]
struct ExpertGeometry {
    which: ExpertTensor,
    n_experts: usize,
    per_expert: usize,
    n_cols: usize,
}

fn geometry_of(name: &str, shape: &[u64], n_values: usize) -> Result<ExpertGeometry, String> {
    let which = if name.ends_with("gate_up_proj") {
        ExpertTensor::GateUp
    } else if name.ends_with("down_proj") {
        ExpertTensor::Down
    } else {
        return Err(format!("{name}: not a routed-expert tensor this tool knows"));
    };
    if shape.len() != 3 {
        return Err(format!("{name}: shape {shape:?}, expected [experts, rows, cols]"));
    }
    let n_experts = shape[0] as usize;
    let per_expert = (shape[1] * shape[2]) as usize;
    let n_cols = shape[2] as usize;
    if n_experts * per_expert != n_values {
        return Err(format!("{name}: shape {shape:?} is not {n_values} values"));
    }
    if per_expert % 64 != 0 || n_cols % 64 != 0 {
        return Err(format!(
            "{name}: expert stride {per_expert} / row stride {n_cols} - an NVFP4 sub-block would \
cross an expert or a row boundary and its 16 values would not be 16 input columns"
        ));
    }
    Ok(ExpertGeometry { which, n_experts, per_expert, n_cols })
}

/// How the newly written bytes compare with the base container's.
struct BlockDiff {
    blocks: usize,
    diff_blocks: usize,
    diff_scale_bytes: u64,
    diff_nibble_bytes: u64,
}

fn compare(got: &[u8], want: &[u8]) -> BlockDiff {
    let n = got.len() / 36;
    let mut d = BlockDiff { blocks: n, diff_blocks: 0, diff_scale_bytes: 0, diff_nibble_bytes: 0 };
    for b in 0..n {
        let g = &got[b * 36..b * 36 + 36];
        let w = &want[b * 36..b * 36 + 36];
        if g == w {
            continue;
        }
        d.diff_blocks += 1;
        for i in 0..4 {
            if g[i] != w[i] {
                d.diff_scale_bytes += 1;
            }
        }
        for i in 4..36 {
            if g[i] != w[i] {
                d.diff_nibble_bytes += 1;
            }
        }
    }
    d
}

pub fn run(args: &[String]) -> i32 {
    let mut base: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut originals: Option<PathBuf> = None;
    let mut imatrix_path: Option<PathBuf> = None;
    let mut report_path: Option<PathBuf> = None;
    let mut layers_arg: Option<String> = None;
    let mut rule = Rule::Mse;
    let mut rule_given = false;
    let mut threads = 0usize;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base" => base = it.next().map(PathBuf::from),
            "--out" => out_path = it.next().map(PathBuf::from),
            "--originals" => originals = it.next().map(PathBuf::from),
            "--imatrix" => imatrix_path = it.next().map(PathBuf::from),
            "--report" => report_path = it.next().map(PathBuf::from),
            "--layers" => layers_arg = it.next().cloned(),
            "--threads" => threads = it.next().and_then(|s| s.parse().ok()).unwrap_or(0),
            "--rule" => {
                let Some(name) = it.next() else {
                    eprintln!("--rule needs a value\n{HELP}");
                    return 2;
                };
                let Some(r) = Rule::parse(name) else {
                    eprintln!("--rule {name}: not one of mse, imatrix, imatrix46\n{HELP}");
                    return 2;
                };
                rule = r;
                rule_given = true;
            }
            s => {
                eprintln!("unexpected argument {s}\n{HELP}");
                return 2;
            }
        }
    }
    let (Some(base), Some(out_path), Some(originals), Some(layers_arg)) = (base, out_path, originals, layers_arg)
    else {
        eprintln!("{HELP}");
        return 2;
    };
    if !rule_given {
        eprintln!("--rule is not optional: the control (mse) and the experiments (imatrix, imatrix46) \nmust never be confused for one another\n{HELP}");
        return 2;
    }
    if rule != Rule::Mse && imatrix_path.is_none() {
        eprintln!("--rule {} needs --imatrix <imatrix.gguf>\n{HELP}", rule.name());
        return 2;
    }
    let layers = match parse_layers(&layers_arg) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("--layers: {e}");
            return 2;
        }
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
    let all_experts: Vec<&BaseEntry> = index.iter().filter(|e| is_expert_text(e)).collect();
    let selected: Vec<&BaseEntry> = all_experts
        .iter()
        .copied()
        .filter(|e| layer_of(&e.name).map(|l| layers.contains(&l)).unwrap_or(false))
        .collect();
    println!(
        "base      {}: {} B, {} tensors, {} routed-expert nvfp4, {} selected over layers {:?}",
        base.display(),
        base_len,
        index.len(),
        all_experts.len(),
        selected.len(),
        layers
    );
    if selected.len() != layers.len() * 2 {
        eprintln!(
            "expected 2 expert tensors per layer, the container gives {} for {} layers",
            selected.len(),
            layers.len()
        );
        return 2;
    }

    let gguf = match &imatrix_path {
        None => None,
        Some(p) => match Gguf::open(p) {
            Ok(g) => {
                println!(
                    "imatrix   {}: GGUF v{}, {} tensors, chunk_count {}, chunk_size {}",
                    p.display(),
                    g.version,
                    g.tensors.len(),
                    g.kv.get("imatrix.chunk_count").map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                    g.kv.get("imatrix.chunk_size").map(|v| v.to_string()).unwrap_or_else(|| "?".into()),
                );
                Some(g)
            }
            Err(e) => {
                eprintln!("{e}");
                return 2;
            }
        },
    };

    let nt = if threads > 0 {
        threads
    } else {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4)
    };
    println!("rule      {} on {nt} threads", rule.name());

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

    let mut base_file = std::fs::File::open(&base).expect("open base container");
    let mut index_tensors: Vec<serde_json::Value> = Vec::new();
    let mut rows: Vec<serde_json::Value> = Vec::new();
    let mut blob_len: u64 = 0;
    let mut values_total: u64 = 0;
    let mut identical_tensors = 0usize;
    let mut zero_count_total = 0usize;
    let mut tiny_count_total = 0usize;

    println!(
        "{:<58} {:>13} {:>13} {:>10} {:>9} {:>11}",
        "tensor", "mse", "imatrix-mse", "clipped%", "scales%", "diff blocks"
    );
    for e in &selected {
        let layer = layer_of(&e.name).expect("checked above");
        let geo = match geometry_of(&e.name, &e.shape, e.n_values) {
            Ok(g) => g,
            Err(m) => {
                eprintln!("{m}");
                return 1;
            }
        };
        let src = originals.join(format!("layer-{layer:02}.safetensors"));
        let (header, data_start) = match read_safetensors_header(&src) {
            Ok(v) => v,
            Err(err) => {
                eprintln!("{}: {err}", src.display());
                return 2;
            }
        };
        let Some(info) = header.get(&e.name) else {
            eprintln!("{}: does not carry {}", src.display(), e.name);
            return 1;
        };
        let dt = info["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<u64> = info["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0)).collect()).unwrap_or_default();
        if shape != e.shape {
            eprintln!("{}: shape {:?} in {}, {:?} in the base container", e.name, shape, src.display(), e.shape);
            return 1;
        }
        let begin = data_start + info["data_offsets"][0].as_u64().unwrap_or(0);
        let end = data_start + info["data_offsets"][1].as_u64().unwrap_or(0);
        let mut raw = vec![0u8; (end - begin) as usize];
        let mut sf = std::fs::File::open(&src).expect("open originals");
        sf.seek(SeekFrom::Start(begin)).expect("seek originals");
        sf.read_exact(&mut raw).expect("read original tensor");
        let values = bytes_to_f32(&raw, &dt);
        drop(raw);
        if values.len() != e.n_values {
            eprintln!("{}: {} values fetched, {} in the base container", e.name, values.len(), e.n_values);
            return 1;
        }

        // the importance matrix of THIS layer and THIS tensor, divided by the counts
        let mut imw: Vec<f32> = Vec::new();
        let mut zero_experts: Vec<usize> = Vec::new();
        let mut tiny_experts: Vec<usize> = Vec::new();
        if rule != Rule::Mse {
            let g = gguf.as_ref().expect("checked above");
            let im = match imatrix::load(g, layer, geo.which) {
                Ok(v) => v,
                Err(m) => {
                    eprintln!("{m}");
                    return 1;
                }
            };
            if im.n_experts != geo.n_experts || im.n_cols != geo.n_cols {
                eprintln!(
                    "{}: the imatrix is {} x {}, the tensor is {} experts x {} input columns",
                    e.name, im.n_experts, im.n_cols, geo.n_experts, geo.n_cols
                );
                return 1;
            }
            zero_experts = im.zero_count_experts();
            tiny_experts = im.tiny_count_experts(100.0);
            zero_count_total += zero_experts.len();
            tiny_count_total += tiny_experts.len();
            imw.reserve(geo.n_experts * geo.n_cols);
            for ex in 0..geo.n_experts {
                imw.extend_from_slice(&im.row(ex));
            }
        }

        let (blocks, global, st) = quantize_expert_tensor(&values, geo.per_expert, geo.n_cols, &imw, rule, nt);
        drop(values);
        if blocks.len() as u64 != e.len {
            eprintln!("{}: {} B produced, the base container stores {} B", e.name, blocks.len(), e.len);
            return 1;
        }
        // ALWAYS compared with the base: for --rule mse this is the control's proof, for the
        // other rules it is how far the encoding moved.
        let mut want = vec![0u8; e.len as usize];
        base_file.seek(SeekFrom::Start(12 + e.offset)).expect("seek base blob");
        base_file.read_exact(&mut want).expect("read base tensor");
        let d = compare(&blocks, &want);
        let scale_ok = global.to_bits() == e.global_scale.to_bits();
        drop(want);
        if d.diff_blocks == 0 && scale_ok {
            identical_tensors += 1;
        }

        // the report row, and the same numbers on stdout
        let mse = st.mse();
        let wmse = st.weighted_mse();
        println!(
            "{:<58} {:>13.6e} {:>13.6e} {:>10.4} {:>9.4} {:>11}",
            short_name(&e.name),
            mse,
            wmse,
            st.clipped as f64 * 100.0 / st.n.max(1) as f64,
            st.scales_moved as f64 * 100.0 / st.sub_blocks.max(1) as f64,
            d.diff_blocks
        );
        rows.push(serde_json::json!({
            "name": e.name, "layer": layer, "rule": rule.name(),
            "n_values": st.n, "shape": e.shape,
            "mse": mse, "imatrix_mse": wmse, "max_abs_err": st.max_abs_err,
            "clipped": st.clipped, "clipped_share": st.clipped as f64 / st.n.max(1) as f64,
            "sub_blocks": st.sub_blocks, "scales_moved": st.scales_moved,
            "scales_moved_share": st.scales_moved as f64 / st.sub_blocks.max(1) as f64,
            "blocks": d.blocks, "diff_blocks": d.diff_blocks,
            "diff_scale_bytes": d.diff_scale_bytes, "diff_nibble_bytes": d.diff_nibble_bytes,
            "global_scale": global, "global_scale_matches_base": scale_ok,
            "zero_count_experts": zero_experts, "tiny_count_experts": tiny_experts,
        }));

        out.write_all(&blocks).expect("write overlay tensor");
        index_tensors.push(serde_json::json!({
            "name": e.name, "shape": e.shape, "section": e.section,
            "n_values": e.n_values, "offset": blob_len, "dtype": "nvfp4",
            "len": blocks.len() as u64, "global_scale": global,
        }));
        blob_len += blocks.len() as u64;
        values_total += e.n_values as u64;
    }

    let built = crate::dense_overlay::utc_date_string();
    let source = format!("originals:{} rule:{}", originals.display(), rule.name());
    let header = serde_json::json!({
        "format": "crow-nest-quant", "version": 1,
        "overlay": {
            "kind": "expert-nvfp4",
            "issue": 79,
            "base_name": base.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            "base_bytes": base_len,
            "source": source,
            "rule": rule.name(),
            "imatrix": imatrix_path.as_ref().map(|p| p.display().to_string()),
            "layers": layers.clone(),
            "tensors": index_tensors.len(),
            "values": values_total,
            "bytes": blob_len,
            "identical_to_base": identical_tensors,
            "built": built,
        },
        "blob_offset": 12u64,
        "tensors": index_tensors,
    });
    let index_json = serde_json::to_vec_pretty(&header).expect("index json");
    out.write_all(&index_json).expect("index");
    out.write_all(&(index_json.len() as u64).to_le_bytes()).expect("index len");
    out.flush().ok();
    drop(out);

    println!(
        "overlay   {}: {} tensors, {} values, payload {:.2} GB, rule {}",
        out_path.display(),
        index_tensors.len(),
        values_total,
        blob_len as f64 / 1e9,
        rule.name()
    );
    println!(
        "control   {} of {} tensors byte-identical to the base container (including the global scale)",
        identical_tensors,
        index_tensors.len()
    );
    if rule != Rule::Mse {
        println!("imatrix   {zero_count_total} zero-count and {tiny_count_total} tiny-count (<100) expert slots over the selected tensors");
    }
    println!("expert-overlay: done in {:.0} s", t0.elapsed().as_secs_f64());

    if let Some(p) = &report_path {
        let doc = serde_json::json!({
            "tool": "converter expert-overlay", "issue": 79,
            "base": base.display().to_string(), "base_bytes": base_len,
            "out": out_path.display().to_string(),
            "originals": originals.display().to_string(),
            "imatrix": imatrix_path.as_ref().map(|p| p.display().to_string()),
            "rule": rule.name(), "layers": layers, "threads": nt,
            "built": crate::dense_overlay::utc_date_string(),
            "identical_to_base": identical_tensors,
            "tensors": rows,
        });
        if let Err(e) = std::fs::write(p, serde_json::to_vec_pretty(&doc).expect("report json")) {
            eprintln!("{}: {e}", p.display());
            return 1;
        }
        println!("report    {}", p.display());
    }
    // the control must be byte-identical or it is not a control
    if rule == Rule::Mse && identical_tensors != index_tensors.len() {
        eprintln!(
            "CONTROL FAILED: {} of {} tensors differ from the base container",
            index_tensors.len() - identical_tensors,
            index_tensors.len()
        );
        return 1;
    }
    0
}

/// `…layers.13.mlp.experts.gate_up_proj` -> `L13 mlp.experts.gate_up_proj`, so the table fits.
fn short_name(name: &str) -> String {
    match layer_of(name) {
        Some(l) => format!("L{l:02} {}", crate::dense_overlay::kind_of(name)),
        None => name.to_string(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_layer_spec_takes_a_list_and_a_range_and_refuses_the_rest() {
        assert_eq!(parse_layers("1,7,13").unwrap(), vec![1, 7, 13]);
        assert_eq!(parse_layers("13,1,7,1").unwrap(), vec![1, 7, 13]);
        assert_eq!(parse_layers("2-5,9").unwrap(), vec![2, 3, 4, 5, 9]);
        assert_eq!(parse_layers(" 3 ").unwrap(), vec![3]);
        for bad in ["", ",", "a", "5-2", "-1", "1..3"] {
            assert!(parse_layers(bad).is_err(), "{bad:?} should be refused");
        }
    }

    #[test]
    fn the_expert_rule_and_the_dense_rule_partition_the_text_nvfp4_tensors() {
        let e = |name: &str, section: &str, dtype: &str| BaseEntry {
            name: name.into(),
            section: section.into(),
            dtype: dtype.into(),
            offset: 0,
            len: 0,
            n_values: 0,
            global_scale: 1.0,
            shape: vec![],
        };
        let all = [
            e("model.language_model.layers.1.mlp.experts.gate_up_proj", "text", "nvfp4"),
            e("model.language_model.layers.1.linear_attn.out_proj.weight", "text", "nvfp4"),
            e("model.language_model.layers.1.mlp.experts.gate_up_proj", "mtp", "nvfp4"),
            e("model.language_model.layers.1.input_layernorm.weight", "text", "bf16"),
        ];
        let experts: Vec<&str> = all.iter().filter(|t| is_expert_text(t)).map(|t| t.name.as_str()).collect();
        let dense: Vec<&str> =
            all.iter().filter(|t| crate::dense_overlay::is_dense_text(t)).map(|t| t.name.as_str()).collect();
        assert_eq!(experts, vec!["model.language_model.layers.1.mlp.experts.gate_up_proj"]);
        assert_eq!(dense, vec!["model.language_model.layers.1.linear_attn.out_proj.weight"]);
        // the mtp copy of the same name is in neither: both rules filter on `section == text`
        assert_eq!(experts.len() + dense.len(), 2);
    }

    #[test]
    fn the_geometry_comes_out_of_the_shape_and_refuses_one_that_would_split_a_sub_block() {
        let g = geometry_of("m.experts.gate_up_proj", &[512, 1280, 2560], 512 * 1280 * 2560).unwrap();
        assert_eq!(g.which, ExpertTensor::GateUp);
        assert_eq!((g.n_experts, g.per_expert, g.n_cols), (512, 1280 * 2560, 2560));
        let d = geometry_of("m.experts.down_proj", &[512, 2560, 640], 512 * 2560 * 640).unwrap();
        assert_eq!(d.which, ExpertTensor::Down);
        assert_eq!((d.n_experts, d.per_expert, d.n_cols), (512, 2560 * 640, 640));
        // a row stride that is not a multiple of 64 would put two input columns of DIFFERENT
        // rows in one sub-block, and the importance weights would be read off the wrong column
        let bad = geometry_of("m.experts.down_proj", &[2, 4, 100], 800).unwrap_err();
        assert!(bad.contains("cross an expert or a row boundary"), "{bad}");
        // a shape that does not multiply out is refused before anything is read
        assert!(geometry_of("m.experts.down_proj", &[2, 4, 64], 999).is_err());
        assert!(geometry_of("m.experts.no_proj", &[2, 4, 64], 512).is_err());
    }

    #[test]
    fn the_block_comparison_names_scale_bytes_and_nibble_bytes_apart() {
        let a = vec![7u8; 72];
        let mut b = a.clone();
        assert_eq!(compare(&a, &b).diff_blocks, 0);
        b[2] = 9; // a scale byte of block 0
        b[40] = 9; // a nibble byte of block 1
        let d = compare(&a, &b);
        assert_eq!((d.blocks, d.diff_blocks), (2, 2));
        assert_eq!((d.diff_scale_bytes, d.diff_nibble_bytes), (1, 1));
    }

    #[test]
    fn the_short_name_keeps_the_layer_and_the_kind() {
        assert_eq!(
            short_name("model.language_model.layers.13.mlp.experts.gate_up_proj"),
            "L13 mlp.experts.gate_up_proj"
        );
    }
}
