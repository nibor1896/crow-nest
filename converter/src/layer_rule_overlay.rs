//! `converter layer-rule-overlay` — #91 phase 1: the llama.cpp LAYER-RULE SHAPE, ported onto
//! this container's precision tiers as bf16 overlay arms. ADDITIVE exactly as `requant-check`
//! (#76), `dense-overlay` (#77) and `expert-overlay` (#79) are: the word `layer-rule-overlay`
//! is taken off the front of the argument list and the conversion path never sees it.
//!
//! What is ported. `src/llama-quant.cpp` (ggml-org/llama.cpp, master 2026-09) shapes a mixed
//! quantization by TENSOR CATEGORY and LAYER POSITION, not by weight-space error:
//!
//! - `attn_v` (and the other v-like categories) is the most sensitive attention tensor and
//!   gets the highest tier short of f16 — Q6_K under `Q4_K_M`, Q8_0 outright for the 8-expert
//!   model ("trades just ~128MB", their comment);
//! - `output.weight` gets Q6_K by default;
//! - `ffn_down` gets `use_more_bits` —
//!   `i < n/8 || i >= 7*n/8 || (i - n/8) % 3 == 2` — first eighth, last eighth, and every
//!   third middle layer counting from `n/8 + 2`.
//!
//! On the NVFP4 grid of `CNQ4.5-M` the only tier above nvfp4 the container and the engine
//! have is `bf16`, so "one tier up" IS bf16, and the overlay mechanism (#77 plumbing,
//! `CROW_CNQ_OVERLAY`) is how it is expressed without touching the 104.73 GB base. The arms:
//!
//! - `attn-v-out` — `self_attn.v_proj`, `self_attn.o_proj` and `linear_attn.out_proj` over ALL
//!   layers: the attention value and output projections join q/k (bf16 since the 2026-09-03
//!   keep-set amendment) at full precision. The router (`mlp.gate.weight`) and `lm_head` that
//!   the issue names in the same breath are ALREADY bf16 in the base — there is nothing to
//!   elevate and this tool says so instead of silently writing a no-op.
//! - `ffn-down-rule` — `mlp.shared_expert.down_proj` over exactly llama.cpp's `use_more_bits`
//!   layer set of the base's own 48 layers.
//! - `ffn-down-all` — the same kind over ALL layers, the companion that separates "the layer
//!   rule matters" from "the tensor matters".
//!
//! What is NOT here, on purpose. Arm (b) of the issue — routed experts elevated to bf16 on a
//! layer rule — is not expressible tonight, for two independent reasons, both of which the
//! artifacts must state rather than paper over:
//!
//! 1. DATA: the BF16 originals of the routed experts are gone. #79 fetched layers
//!    1,7,13,19,25,31,37,43 (37.5 GiB), built its overlays, and the payloads were deleted to
//!    free disk — `models/Qwen3.8-Flash-Next-original/experts/` now carries only the eight
//!    `layer-NN.manifest.json` records and the shard headers. `dense/dense.safetensors`
//!    (5.17 GB) carries the dense-path originals ONLY: its selection rule is
//!    `section == "text" and dtype == "nvfp4" and ".mlp.experts." not in name`.
//! 2. ENGINE: `engine/src/cnq.rs::overlay_refusal` refuses a bf16 tensor whose name contains
//!    `.mlp.experts.` — "a routed expert may only be shadowed as nvfp4 of the same byte
//!    length, the expert slabs and the kernels are cut from it" (#79 widened that table on
//!    purpose: `residency` cuts per-expert slabs out of `byte_len / 512` and hands them to
//!    kernels that index 36 B per 64 values). Expressing a bf16 expert needs engine slab work,
//!    not converter work.
//!
//! And the #79 STOP RULE is binding on every arm here: nothing in this file tunes, scores or
//! selects by weight-space MSE. The arms are fixed shapes copied from llama.cpp's rule table;
//! their acceptance instrument is the oracle-KLD of #90, never an error term. The report the
//! tool prints is byte counts, value counts and — for the `--from-container` control — the
//! bf16 rounding residue, which is wiring information (#77's control discipline), not quality.
//!
//! The two sources are #77's, unchanged:
//!
//! - `--from-originals <dense.safetensors>` — the BF16 originals, byte for byte;
//! - `--from-container <base.cnq>` — the CONTROL: the base's own NVFP4 values dequantized and
//!   rounded to bf16. No new information; an engine that answers differently under it has a
//!   wiring difference, not a weight difference.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use crate::dense_overlay::{blocks_to_bf16, is_dense_text, kind_of, read_base_index, utc_date_string, BaseEntry};
use crate::expert_overlay::layer_of;
use crate::{read_safetensors_header, MAGIC};

pub const HELP: &str = "usage: converter layer-rule-overlay --base <container.cnq> --out <overlay.cnq> \\\n                                     (--from-originals <dense.safetensors> | --from-container <base.cnq>) \\\n                                     --arm attn-v-out|ffn-down-rule|ffn-down-all\n  builds a bf16 OVERLAY container for one llama.cpp-shaped layer-rule arm (#91 phase 1)\n  --arm attn-v-out      self_attn.v_proj + self_attn.o_proj + linear_attn.out_proj, all layers\n                        (the router mlp.gate.weight and lm_head are already bf16 in the base)\n  --arm ffn-down-rule   mlp.shared_expert.down_proj on llama.cpp's use_more_bits layers\n                        (first eighth, last eighth, every third middle layer)\n  --arm ffn-down-all    mlp.shared_expert.down_proj on all layers\n  --from-originals     the BF16 originals of #76 — the real weights\n  --from-container     the CONTROL: the base's own NVFP4 dequantized to bf16";

/// llama.cpp's `use_more_bits` (src/llama-quant.cpp, `llama_tensor_get_type_impl`), ported
/// verbatim including its C++ precedence `7 * n_layers / 8`:
///
/// ```text
/// i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8) % 3 == 2
/// ```
///
/// First eighth, last eighth, and every third middle layer counting from `n/8 + 2` (so for
/// 48 layers: 0-5, 42-47, and 8, 11, ..., 41 — 24 of 48). The middle clause's subtraction is
/// guarded because `i < n/8` has already returned; C++ short-circuits the same way.
pub fn llama_use_more_bits(i_layer: usize, n_layers: usize) -> bool {
    let eighth = n_layers / 8;
    i_layer < eighth || i_layer >= 7 * n_layers / 8 || (i_layer >= eighth && (i_layer - eighth) % 3 == 2)
}

/// The layers of `use_more_bits` over `n_layers`, ascending.
pub fn use_more_bits_layers(n_layers: usize) -> Vec<usize> {
    (0..n_layers).filter(|&i| llama_use_more_bits(i, n_layers)).collect()
}

/// Which layers an arm covers.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum LayerRule {
    All,
    /// llama.cpp's `use_more_bits` over the base's own layer count.
    LlamaUseMoreBits,
}

impl LayerRule {
    fn name(self) -> &'static str {
        match self {
            LayerRule::All => "all",
            LayerRule::LlamaUseMoreBits => "llama-use-more-bits",
        }
    }
}

/// One arm of the #91 phase-1 table: a fixed kind set plus a fixed layer rule. The kinds are
/// the container's own (`dense_overlay::kind_of`); nothing here is searched, tuned or scored.
pub struct Arm {
    pub name: &'static str,
    pub what: &'static str,
    pub kinds: &'static [&'static str],
    pub layer_rule: LayerRule,
}

pub const ARMS: &[Arm] = &[
    {
        Arm {
            name: "attn-v-out",
            what: "attention v/out projections to bf16 on every layer (llama.cpp: attn_v is the \
most sensitive attention tensor and gets the top tier; the router and lm_head are already bf16)",
            kinds: &["self_attn.v_proj", "self_attn.o_proj", "linear_attn.out_proj"],
            layer_rule: LayerRule::All,
        }
    },
    {
        Arm {
            name: "ffn-down-rule",
            what: "the ffn_down-equivalent dense tensor to bf16 on llama.cpp's use_more_bits \
layers: first eighth, last eighth, every third middle layer",
            kinds: &["mlp.shared_expert.down_proj"],
            layer_rule: LayerRule::LlamaUseMoreBits,
        }
    },
    {
        Arm {
            name: "ffn-down-all",
            what: "the ffn_down-equivalent dense tensor to bf16 on every layer — the companion \
that separates the layer rule from the tensor",
            kinds: &["mlp.shared_expert.down_proj"],
            layer_rule: LayerRule::All,
        }
    },
];

pub fn arm_by_name(name: &str) -> Option<&'static Arm> {
    ARMS.iter().find(|a| a.name == name)
}

/// The base's own layer count, derived the way every selection in this crate is derived: from
/// the container's index trailer, not from an out-of-band config. One plus the largest layer
/// number over text tensors that carry the per-layer prefix.
pub fn n_layers_of(index: &[BaseEntry]) -> usize {
    index
        .iter()
        .filter(|e| e.section == "text")
        .filter_map(|e| layer_of(&e.name))
        .map(|l| l + 1)
        .max()
        .unwrap_or(0)
}

/// Does `e` belong to the arm? The dense-text rule of #77 (text section, nvfp4, not a routed
/// expert), then the arm's kinds, then the arm's layer rule. A kind that matches no tensor is
/// the empty-overlay hazard #77 refuses, so the caller refuses it too.
pub fn arm_selects(arm: &Arm, e: &BaseEntry, n_layers: usize) -> bool {
    if !is_dense_text(e) {
        return false;
    }
    if !arm.kinds.contains(&kind_of(&e.name).as_str()) {
        return false;
    }
    match arm.layer_rule {
        LayerRule::All => true,
        LayerRule::LlamaUseMoreBits => layer_of(&e.name).map(|l| llama_use_more_bits(l, n_layers)).unwrap_or(false),
    }
}

enum Source {
    Originals(PathBuf),
    Container(PathBuf),
}

/// Re-read the overlay that was just written and compare every tensor's bytes with the source
/// they were claimed to come from. The originals arm must be byte-for-byte the #76 file; the
/// control arm must be the deterministic re-dequant of the base's own bytes. This is the
/// converter's own verification of the phase-1 artifacts — the engine's boot refusal table is
/// the second, but the machine it runs on is not available in phase 1.
fn verify_written(
    overlay_path: &Path,
    selected: &[&BaseEntry],
    source: &Source,
    originals: &std::collections::BTreeMap<String, (u64, u64, String, usize)>,
) -> Result<(usize, u64), String> {
    let back = read_base_index(overlay_path).map_err(|e| format!("re-read: {e}"))?;
    if back.len() != selected.len() {
        return Err(format!("re-read: {} tensors in the overlay, {} were selected", back.len(), selected.len()));
    }
    let mut ov_file = std::fs::File::open(overlay_path).map_err(|e| format!("re-read: {e}"))?;
    let mut src_file = match source {
        Source::Originals(p) => std::fs::File::open(p).map_err(|e| format!("re-read {}: {e}", p.display()))?,
        Source::Container(p) => std::fs::File::open(p).map_err(|e| format!("re-read {}: {e}", p.display()))?,
    };
    let mut ov_buf = Vec::new();
    let mut want_buf = Vec::new();
    let mut deq = Vec::new();
    let mut checked_bytes: u64 = 0;
    for (got, e) in back.iter().zip(selected.iter()) {
        if got.name != e.name || got.n_values != e.n_values || got.dtype != "bf16" {
            return Err(format!("re-read: {} does not match the selection ({} {})", got.name, got.dtype, got.n_values));
        }
        let want: Vec<u8> = match source {
            Source::Originals(_) => {
                let Some(&(begin, end, ..)) = originals.get(&e.name) else {
                    return Err(format!("re-read: {} is not in the originals map", e.name));
                };
                want_buf.resize((end - begin) as usize, 0);
                src_file.seek(SeekFrom::Start(begin)).map_err(|e| e.to_string())?;
                src_file.read_exact(&mut want_buf).map_err(|e| e.to_string())?;
                want_buf.clone()
            }
            Source::Container(_) => {
                let len = e.len.max(((e.n_values as u64) + 63) / 64 * 36);
                want_buf.resize(len as usize, 0);
                src_file.seek(SeekFrom::Start(12 + e.offset)).map_err(|e| e.to_string())?;
                src_file.read_exact(&mut want_buf).map_err(|e| e.to_string())?;
                deq.clear();
                blocks_to_bf16(&want_buf, e.global_scale, e.n_values, &mut deq);
                deq.clone()
            }
        };
        ov_buf.resize(got.len as usize, 0);
        ov_file.seek(SeekFrom::Start(12 + got.offset)).map_err(|e| e.to_string())?;
        ov_file.read_exact(&mut ov_buf).map_err(|e| e.to_string())?;
        if ov_buf != want {
            return Err(format!("re-read: {} differs from its source bytes", e.name));
        }
        checked_bytes += want.len() as u64;
    }
    Ok((back.len(), checked_bytes))
}

pub fn run(args: &[String]) -> i32 {
    let mut base: Option<PathBuf> = None;
    let mut out_path: Option<PathBuf> = None;
    let mut source: Option<Source> = None;
    let mut arm_name: Option<String> = None;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--base" => base = it.next().map(PathBuf::from),
            "--out" => out_path = it.next().map(PathBuf::from),
            "--from-originals" => source = it.next().map(|p| Source::Originals(PathBuf::from(p))),
            "--from-container" => source = it.next().map(|p| Source::Container(PathBuf::from(p))),
            "--arm" => arm_name = it.next().cloned(),
            s => {
                eprintln!("unexpected argument {s}\n{HELP}");
                return 2;
            }
        }
    }
    let (Some(base), Some(out_path), Some(source), Some(arm_name)) = (base, out_path, source, arm_name) else {
        eprintln!("{HELP}");
        return 2;
    };
    let Some(arm) = arm_by_name(&arm_name) else {
        eprintln!("--arm {arm_name}: not one of the #91 phase-1 arms. The arms are:");
        for a in ARMS {
            eprintln!("  {:<14} {}", a.name, a.what);
        }
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
    let n_layers = n_layers_of(&index);
    if n_layers == 0 {
        eprintln!("the base index carries no model.language_model.layers.<N> text tensors");
        return 2;
    }
    let selected: Vec<&BaseEntry> = index.iter().filter(|e| arm_selects(arm, e, n_layers)).collect();
    let layers: Vec<usize> = selected.iter().filter_map(|e| layer_of(&e.name)).collect::<std::collections::BTreeSet<_>>().into_iter().collect();
    println!(
        "base      {}: {} B, {} tensors, {} layers — arm {} (kinds {}, layer rule {})",
        base.display(),
        base_len,
        index.len(),
        n_layers,
        arm.name,
        arm.kinds.join(","),
        arm.layer_rule.name()
    );
    match arm.layer_rule {
        LayerRule::LlamaUseMoreBits => println!(
            "rule      use_more_bits over {n_layers} layers: first 0..{}, last {}..{}, every third middle -> {} of {n_layers} layers {:?}",
            n_layers / 8 - 1,
            7 * n_layers / 8,
            n_layers - 1,
            layers.len(),
            layers
        ),
        LayerRule::All => println!("rule      all {n_layers} layers"),
    }
    // a kind that selects nothing is the #77 empty-overlay hazard: it would be measured as
    // "the originals change nothing"
    for k in arm.kinds {
        if !selected.iter().any(|e| kind_of(&e.name) == *k) {
            eprintln!("kind {k} of arm {} selects no tensor of this base — refusing", arm.name);
            return 2;
        }
    }
    if selected.is_empty() {
        eprintln!("no tensor selected — refusing to write an empty overlay");
        return 2;
    }
    // the issue's arm (a) names the router and the output head beside v/out: in THIS container
    // both are already bf16, and a tool that quietly wrote a no-op would be read as one
    if arm.name == "attn-v-out" {
        let already: Vec<&str> = index
            .iter()
            .filter(|e| {
                e.section == "text"
                    && e.dtype == "bf16"
                    && (e.name.contains("mlp.gate.weight") || e.name == "lm_head.weight" || e.name.contains("shared_expert_gate"))
            })
            .map(|e| e.name.as_str())
            .collect();
        println!(
            "already bf16 in the base, nothing to elevate: {} router/gate/head tensors (e.g. {:?})",
            already.len(),
            &already[..already.len().min(3)]
        );
    }

    // ---- the source side (#77 verbatim) ----
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
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                    .unwrap_or_default();
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
            "kind": "layer-rule-bf16",
            "issue": 91,
            "base_name": base.file_name().map(|s| s.to_string_lossy().to_string()).unwrap_or_default(),
            "base_bytes": base_len,
            "source": source_str,
            "arm": arm.name,
            "arm_what": arm.what,
            "kinds": arm.kinds,
            "layer_rule": arm.layer_rule.name(),
            "n_layers": n_layers,
            "layers": layers,
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

    // ---- the converter's own verification: re-read and compare with the source ----
    // (the originals map was drained by the write loop; rebuild it for the verifier)
    let mut originals_again = std::collections::BTreeMap::new();
    if let Source::Originals(p) = &source {
        if let Ok((header, data_start)) = read_safetensors_header(p) {
            for (name, info) in header.as_object().into_iter().flatten() {
                if name == "__metadata__" {
                    continue;
                }
                let dt = info["dtype"].as_str().unwrap_or("").to_string();
                let shape: Vec<usize> = info["shape"]
                    .as_array()
                    .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
                    .unwrap_or_default();
                let n: usize = shape.iter().product();
                let begin = info["data_offsets"][0].as_u64().unwrap_or(0);
                let end = info["data_offsets"][1].as_u64().unwrap_or(0);
                originals_again.insert(name.clone(), (data_start + begin, data_start + end, dt, n));
            }
        }
    }
    match verify_written(&out_path, &selected, &source, &originals_again) {
        Ok((n, bytes)) => println!(
            "verified  re-read {} tensors, {bytes} B — every byte is its source's ({} arm)",
            n,
            if matches!(source, Source::Container(_)) { "control" } else { "originals" }
        ),
        Err(e) => {
            eprintln!("VERIFY FAILED: {e}");
            return 1;
        }
    }

    println!(
        "overlay   {}: {} tensors, {} values, payload {:.3} GB, arm {} ({})",
        out_path.display(),
        index_tensors.len(),
        values_total,
        blob_len as f64 / 1e9,
        arm.name,
        source_str
    );
    for (k, (c, n)) in &per_kind {
        println!("  {c:>3} x {k}  ({n} values)");
    }
    if matches!(source, Source::Container(_)) {
        println!(
            "  control: {inexact_total} of {values_total} dequantized values ({:.4} %) did not fit bf16 exactly and were rounded to nearest-even",
            inexact_total as f64 * 100.0 / values_total.max(1) as f64
        );
    }
    println!("layer-rule-overlay: done in {:.0} s", t0.elapsed().as_secs_f64());
    0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn e(name: &str, dtype: &str) -> BaseEntry {
        BaseEntry {
            name: name.into(),
            section: "text".into(),
            dtype: dtype.into(),
            offset: 0,
            len: 0,
            n_values: 0,
            global_scale: 1.0,
            shape: vec![],
        }
    }

    /// The exact 48-layer set of llama.cpp's rule: first 6, last 6, and every third middle
    /// layer counting from 8 — 24 of 48. Ported line by line, so the set is spelled out.
    #[test]
    fn use_more_bits_is_llama_cpps_on_48_layers() {
        let want: Vec<usize> = vec![
            0, 1, 2, 3, 4, 5, // first eighth
            8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41, // every third middle
            42, 43, 44, 45, 46, 47, // last eighth
        ];
        assert_eq!(use_more_bits_layers(48), want);
        assert_eq!(want.len(), 24);
    }

    /// C++ semantics on the small-model edge: n_layers = 8 has eighth 1, so {0} front, {7}
    /// back, and (i-1) % 3 == 2 fires at 3 and 6. n_layers = 4 has eighth 0, so the middle
    /// clause is (i-0) % 3 == 2 and fires at 2, and `i >= 7*4/8 = 3` keeps the back — the
    /// C++ `7*n/8` precedence, computed the same way.
    #[test]
    fn use_more_bits_keeps_the_cpp_edges() {
        assert_eq!(use_more_bits_layers(8), vec![0, 3, 6, 7]);
        assert_eq!(use_more_bits_layers(4), vec![2, 3]); // (i-0)%3==2 -> {2}, 7*4/8 = 3 -> {3}
        // the subtraction must not underflow: 0 < 6 already returned before (0-6) is evaluated
        assert!(llama_use_more_bits(0, 48));
    }

    #[test]
    fn the_arms_parse_and_the_rest_is_refused() {
        for a in ARMS {
            assert_eq!(arm_by_name(a.name).map(|x| x.name), Some(a.name));
        }
        assert!(arm_by_name("everything").is_none());
        assert!(arm_by_name("").is_none());
    }

    #[test]
    fn attn_v_out_selects_the_attention_projections_and_not_the_router() {
        let idx = [
            e("model.language_model.layers.3.self_attn.v_proj.weight", "nvfp4"),
            e("model.language_model.layers.3.self_attn.o_proj.weight", "nvfp4"),
            e("model.language_model.layers.3.linear_attn.out_proj.weight", "nvfp4"),
            // the router is ALREADY bf16: is_dense_text (dtype nvfp4) must keep it out
            e("model.language_model.layers.3.mlp.gate.weight", "bf16"),
            e("model.language_model.layers.3.mlp.experts.gate_up_proj", "nvfp4"),
            e("model.language_model.layers.3.mlp.shared_expert.down_proj.weight", "nvfp4"),
        ];
        let arm = arm_by_name("attn-v-out").unwrap();
        let got: Vec<&str> = idx.iter().filter(|t| arm_selects(arm, t, 48)).map(|t| t.name.as_str()).collect();
        assert_eq!(got, vec![
            "model.language_model.layers.3.self_attn.v_proj.weight",
            "model.language_model.layers.3.self_attn.o_proj.weight",
            "model.language_model.layers.3.linear_attn.out_proj.weight",
        ]);
    }

    #[test]
    fn ffn_down_rule_takes_the_use_more_bits_layers_and_all_takes_all() {
        let mut idx = Vec::new();
        for l in 0..48usize {
            idx.push(e(&format!("model.language_model.layers.{l}.mlp.shared_expert.down_proj.weight"), "nvfp4"));
        }
        let rule = arm_by_name("ffn-down-rule").unwrap();
        let got: Vec<usize> = idx
            .iter()
            .filter(|t| arm_selects(rule, t, 48))
            .map(|t| layer_of(&t.name).unwrap())
            .collect();
        assert_eq!(got, use_more_bits_layers(48));
        let all = arm_by_name("ffn-down-all").unwrap();
        assert_eq!(idx.iter().filter(|t| arm_selects(all, t, 48)).count(), 48);
        // layer 6 and 7 are in NO arm: the first eighth ends at 5 and the middle stride starts at 8
        assert!(!llama_use_more_bits(6, 48) && !llama_use_more_bits(7, 48));
    }

    #[test]
    fn the_layer_count_comes_off_the_base_index() {
        let idx = [
            e("model.language_model.layers.0.self_attn.v_proj.weight", "nvfp4"),
            e("model.language_model.layers.47.mlp.shared_expert.down_proj.weight", "nvfp4"),
            e("lm_head.weight", "bf16"),                       // no layer prefix
            e("mtp.layers.0.mlp.gate.weight", "bf16"),         // not the text section
        ];
        assert_eq!(n_layers_of(&idx), 48);
        assert_eq!(n_layers_of(&[]), 0);
    }

    /// The control rounding rule is #77's and is re-used, not re-implemented: a spot check
    /// that the bf16 twin in this module's write path is the same function the dense overlay
    /// tests pin (ties to even).
    #[test]
    fn the_control_rounds_like_dense_overlay() {
        let tie = f32::from_bits(0x3F81_8000); // 1 + 1.5 bf16 ulp -> 0x3F82, not the truncation
        assert_eq!(crate::dense_overlay::f32_to_bf16_rne(tie), 0x3F82);
    }
}
