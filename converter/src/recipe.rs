//! Crow #300 C6: the per-model recipe table and the index v2 `model` block.
//!
//! Before C6 the converter knew one model. Its keep set (`keep_bf16` in `main.rs` at `64c242b`)
//! named Flash-Next tensors, and its index trailer said nothing about which checkpoint it came
//! from. C6 makes both a function of the model FAMILY, read from the checkpoint's own
//! `config.json` (`text_config.model_type`), the same key `engine/src/meta.rs` `Family::detect`
//! reads:
//!
//! | family | `text_config.model_type` | recipe | row |
//! |---|---|---|---|
//! | `FlashNext` | `qwen4_exp_text` | `cnq4.5-flash-next` | [`decide_flash_next`]: the pre-C6 keep set, verbatim |
//! | `Qwen35Dense` | `qwen3_5_text` | `cnq4.5-qwen35-dense` | [`decide_qwen35_dense`]: the phase 2 recipe (#300 "Phase 2", research brief) |
//!
//! The Flash-Next row is today's behaviour exactly. It is proved against every one of the 1658
//! tensors of the CNQ4.5-M index trailer (`tests/fixtures/cnq45m-index.tsv`), not against a
//! sample.
//!
//! **Latent finding, recorded and NOT changed on Flash-Next** (research brief 2026-09-25): the
//! Flash-Next row lets `linear_attn.in_proj_a` / `in_proj_b` (`[48, 2560]`) and
//! `linear_attn.conv1d` (`[10240, 1, 4]`) fall through to its last rule, `n % 64 != 0 || n < 64`,
//! and both are multiples of 64, so all three are NVFP4 in CNQ4.5-M (108 tensors, the "rest"
//! group of `group91-manifest.md`). A 64-value block of `conv1d` spans the four taps of 16
//! channels. ModelOpt's `default_disabled_quantizers.yaml` leaves them unquantized, and Unsloth
//! stores them F16/Q8_0 (alpha/beta) and F32 (conv1d). The dense row keeps all three BF16 on
//! purpose; changing the Flash-Next row would change the container of record and is not C6's
//! call.
//!
//! The dense row is a WHITELIST: every text tensor must match one named row, or the plan and
//! the conversion refuse and name the tensor. The Flash-Next row keeps its catch-all, because
//! that catch-all is part of the behaviour of record.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

/// The model families the converter has a recipe for. The names are the engine's
/// (`meta::Family` `Debug`), so the index's `model.family` reads the same on both sides.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Family {
    FlashNext,
    Qwen35Dense,
}

impl Family {
    pub const ALL: [Family; 2] = [Family::FlashNext, Family::Qwen35Dense];

    pub fn name(self) -> &'static str {
        match self {
            Family::FlashNext => "FlashNext",
            Family::Qwen35Dense => "Qwen35Dense",
        }
    }

    pub fn model_type(self) -> &'static str {
        match self {
            Family::FlashNext => "qwen4_exp_text",
            Family::Qwen35Dense => "qwen3_5_text",
        }
    }

    /// The `recipe` name the index v2 carries.
    pub fn recipe(self) -> &'static str {
        match self {
            Family::FlashNext => "cnq4.5-flash-next",
            Family::Qwen35Dense => "cnq4.5-qwen35-dense",
        }
    }

    pub fn from_name(name: &str) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.name() == name)
    }

    /// `text_config.model_type` of a parsed `config.json`, refused by name when it is missing
    /// or not a family this converter has a recipe for (the engine's wording, `meta.rs:79`).
    pub fn detect(config: &serde_json::Value) -> Result<Family, String> {
        let Some(mt) = config["text_config"]["model_type"].as_str() else {
            return Err("config.json has no text_config.model_type - the converter cannot pick a recipe (Crow #300 C6)".into());
        };
        Family::ALL.into_iter().find(|f| f.model_type() == mt).ok_or_else(|| {
            format!(
                "text_config.model_type '{mt}' is not a model family this converter has a recipe for \
(known: qwen4_exp_text = Qwen3.8-Flash-Next, qwen3_5_text = dense Qwen3.5/3.8) - refusing (Crow #300 C6)"
            )
        })
    }
}

/// What the converter writes for one tensor.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DtypeOut {
    Nvfp4,
    Bf16,
    /// widened (or carried) to f32: the dense row's `A_log`
    F32,
    /// raw integer metadata (the Flash-Next PLE tables), never quantized
    I64,
}

impl DtypeOut {
    pub fn as_str(self) -> &'static str {
        match self {
            DtypeOut::Nvfp4 => "nvfp4",
            DtypeOut::Bf16 => "bf16",
            DtypeOut::F32 => "f32",
            DtypeOut::I64 => "i64",
        }
    }

    /// Bytes this tensor takes in the container, the twin of `engine/src/cnq.rs::Cnq::byte_len`.
    pub fn bytes(self, n: usize) -> u64 {
        let n = n as u64;
        match self {
            DtypeOut::Nvfp4 => n.div_ceil(64) * 36,
            DtypeOut::Bf16 => n * 2,
            DtypeOut::F32 => n * 4,
            DtypeOut::I64 => n * 8,
        }
    }
}

/// One recipe decision: the dtype, the section and the name of the row that made it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Decision {
    pub dtype: DtypeOut,
    pub section: &'static str,
    pub rule: &'static str,
}

/// The section of a tensor. The same name patterns for both families (the dense checkpoint has
/// no `ngram_embedding`, so it never yields `ple`); this is `section_of` of `64c242b`.
pub fn section_of(name: &str) -> &'static str {
    if name.contains("ngram_embedding") {
        "ple"
    } else if name.contains(".visual.") || name.starts_with("model.visual") {
        "vit"
    } else if name.contains(".mtp.") || name.starts_with("mtp") {
        "mtp"
    } else {
        "text"
    }
}

/// The decision for one tensor under `family`'s recipe. `src_dtype` is the safetensors dtype
/// (`BF16`, `F32`, `F16`, `I64`).
pub fn decide(family: Family, name: &str, shape: &[usize], src_dtype: &str) -> Result<Decision, String> {
    let d = match family {
        Family::FlashNext => decide_flash_next(name, shape, src_dtype),
        Family::Qwen35Dense => decide_qwen35_dense(name, shape, src_dtype)?,
    };
    // A bf16 keep is written as the source's raw bytes (the pre-C6 write path, kept). That is
    // only a bf16 tensor when the source IS bf16: an F32 or F16 source would land in the
    // container as 4-byte or f16 values under a `bf16` label, and every reader would take the
    // wrong length or the wrong bits. Refused by name instead of converted quietly; neither
    // checkpoint of record has such a tensor (27B: 1199 of 1199 BF16; CNQ4.5-M: every bf16
    // entry has `len == 2 * n_values`).
    if d.dtype == DtypeOut::Bf16 && src_dtype != "BF16" {
        return Err(format!(
            "{name}: source dtype {src_dtype}, but the {} recipe keeps it bf16 and a bf16 keep is \
the source's raw bytes - refusing rather than writing {src_dtype} bytes under a bf16 label",
            family.recipe()
        ));
    }
    if d.dtype == DtypeOut::F32 && !(src_dtype == "BF16" || src_dtype == "F32") {
        return Err(format!("{name}: source dtype {src_dtype} cannot be carried as f32 by this converter"));
    }
    if d.dtype == DtypeOut::Nvfp4 && shape.iter().product::<usize>() % 64 != 0 {
        return Err(format!("{name}: {shape:?} is not a whole number of 64-value NVFP4 blocks"));
    }
    Ok(d)
}

/// The Flash-Next row: `keep_bf16` and the I64 carry of `main.rs` at `64c242b`, byte for byte
/// in its rule order. Every rule name below is only a label for the plan's table.
pub fn decide_flash_next(name: &str, shape: &[usize], src_dtype: &str) -> Decision {
    let section = section_of(name);
    let n: usize = shape.iter().product();
    let d = |dtype, rule| Decision { dtype, section, rule };
    // I64: integer metadata (PLE index tables - layer_multipliers, ngram_heads_offsets,
    // ngram_heads_vocab_sizes) - raw carry, never quantize
    if src_dtype == "I64" {
        return d(DtypeOut::I64, "i64 metadata carry");
    }
    if shape.len() == 1 {
        return d(DtypeOut::Bf16, "1-D (norms, biases, A_log, dt_bias, gates)");
    }
    if name.contains("norm")
        || name.contains("embed_tokens")
        || name.contains("lm_head")
        || name.contains("mlp.gate.weight") // router GEMM
        || name.contains("shared_expert_gate")
    {
        return d(DtypeOut::Bf16, "spec 1.2 keep (norm, embed_tokens, lm_head, router, shared_expert_gate)");
    }
    // Amendment 2026-09-03 (robin GO, mix of options 1+2 on the #11 gate finding): FP4
    // compounding through 48 layers breaks argmax parity - keep the residual-path HC mix
    // projections and attention q/k in BF16.
    if name.contains("input_mix_weight_down")
        || name.contains("input_mix_weight_up")
        || name.contains("self_attn.q_proj.weight")
        || name.contains("self_attn.k_proj.weight")
    {
        return d(DtypeOut::Bf16, "2026-09-03 amendment (HC mix, attention q/k)");
    }
    if n % 64 != 0 || n < 64 {
        return d(DtypeOut::Bf16, "not a whole 64-value block");
    }
    d(DtypeOut::Nvfp4, "default nvfp4 (incl. in_proj_a/b, conv1d: the latent finding)")
}

/// The dense Qwen3.5/3.8 row (phase 2 recipe, #300 "Phase 2" and the research brief):
///
/// - NVFP4: MLP `gate_proj` / `up_proj` / `down_proj`; GDN `in_proj_qkv` / `in_proj_z` /
///   `out_proj`; attention `q_proj` / `k_proj` / `v_proj` / `o_proj`; `lm_head`;
/// - BF16: `in_proj_a` / `in_proj_b`, `conv1d`, every norm, `dt_bias`; the token embedding
///   (it lives in host RAM);
/// - f32: `A_log`;
/// - `vit` and `mtp`: their own sections, every tensor BF16 (the published NVIDIA NVFP4 27B
///   keeps vision and MTP BF16; robin wants BF16 vision on crow-nest). Both are optional to
///   load, so they cost GPU memory only on the cards that load them.
///
/// A text tensor matching none of the rows is refused by name.
pub fn decide_qwen35_dense(name: &str, shape: &[usize], src_dtype: &str) -> Result<Decision, String> {
    let section = section_of(name);
    let d = |dtype, rule| Ok(Decision { dtype, section, rule });
    if src_dtype == "I64" {
        return Err(format!("{name}: an I64 tensor in a dense checkpoint - the dense recipe has no row for it"));
    }
    match section {
        "vit" => return d(DtypeOut::Bf16, "vision tower BF16 (own section)"),
        "mtp" => return d(DtypeOut::Bf16, "MTP BF16 (own section)"),
        "ple" => return Err(format!("{name}: a `ple` tensor in a dense checkpoint - the dense family has no PLE")),
        _ => {}
    }
    let has = |k: &str| name.contains(k);
    if name.ends_with("linear_attn.A_log") {
        return d(DtypeOut::F32, "A_log f32");
    }
    if has("embed_tokens") {
        return d(DtypeOut::Bf16, "token embedding BF16 (host RAM)");
    }
    if has("norm") {
        return d(DtypeOut::Bf16, "norm BF16");
    }
    if name.ends_with("linear_attn.dt_bias") {
        return d(DtypeOut::Bf16, "dt_bias BF16");
    }
    if has("linear_attn.in_proj_a.") || has("linear_attn.in_proj_b.") {
        return d(DtypeOut::Bf16, "GDN in_proj_a/b BF16");
    }
    if has("linear_attn.conv1d.") {
        return d(DtypeOut::Bf16, "GDN conv1d BF16");
    }
    if shape.len() >= 2 {
        if has("mlp.gate_proj.") || has("mlp.up_proj.") || has("mlp.down_proj.") {
            return d(DtypeOut::Nvfp4, "MLP gate/up/down NVFP4");
        }
        if has("linear_attn.in_proj_qkv.") || has("linear_attn.in_proj_z.") || has("linear_attn.out_proj.") {
            return d(DtypeOut::Nvfp4, "GDN in_proj_qkv/z, out_proj NVFP4");
        }
        if has("self_attn.q_proj.") || has("self_attn.k_proj.") || has("self_attn.v_proj.") || has("self_attn.o_proj.") {
            return d(DtypeOut::Nvfp4, "attention q/k/v/o NVFP4");
        }
        if name == "lm_head.weight" {
            return d(DtypeOut::Nvfp4, "lm_head NVFP4");
        }
    }
    Err(format!(
        "{name} {shape:?}: matches no row of the {} recipe - refusing (the dense row is a whitelist; \
add a row with a reason, do not let it fall through)",
        Family::Qwen35Dense.recipe()
    ))
}

// ---------------------------------------------------------------------------------------------
// the `model` block of the index v2
// ---------------------------------------------------------------------------------------------

/// Hex sha256 of `bytes`.
pub fn sha256_hex(bytes: &[u8]) -> String {
    use sha2::Digest;
    hex(&sha2::Sha256::digest(bytes))
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

/// Hex sha256 of a whole file, streamed in 8 MiB reads (never the file in RAM).
pub fn sha256_file(path: &Path) -> std::io::Result<String> {
    use sha2::Digest;
    let mut f = std::fs::File::open(path)?;
    let mut h = sha2::Sha256::new();
    let mut buf = vec![0u8; 8 << 20];
    loop {
        let n = f.read(&mut buf)?;
        if n == 0 {
            break;
        }
        h.update(&buf[..n]);
    }
    Ok(hex(&h.finalize()))
}

/// One source shard as the index v2 records it.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShardRecord {
    pub file: String,
    pub size: u64,
    pub sha256: String,
    /// `hf-lfs` (the `lfs_sha256` Hugging Face recorded for the download) or `computed`
    pub sha256_from: &'static str,
}

/// What `hf download --local-dir` leaves in `<dir>/.cache/huggingface/`: the revision (the one
/// `trees/<revision>.json`) and, per file, the LFS sha256 and size.
pub struct HfTree {
    pub revision: String,
    /// file -> (lfs_sha256, lfs_size)
    pub lfs: BTreeMap<String, (String, u64)>,
}

/// Read the HF local-dir cache, `None` when there is none (the Flash-Next originals were
/// fetched another way). More than one tree file is ambiguous and refused.
pub fn read_hf_tree(model_dir: &Path) -> Result<Option<HfTree>, String> {
    let trees = model_dir.join(".cache/huggingface/trees");
    let Ok(rd) = std::fs::read_dir(&trees) else { return Ok(None) };
    let mut files: Vec<PathBuf> = rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|p| p.extension().is_some_and(|x| x == "json")).collect();
    files.sort();
    if files.is_empty() {
        return Ok(None);
    }
    if files.len() > 1 {
        return Err(format!("{}: {} tree files - which revision is this directory? (pass --revision)", trees.display(), files.len()));
    }
    let revision = files[0].file_stem().unwrap().to_string_lossy().to_string();
    let v: serde_json::Value = serde_json::from_slice(&std::fs::read(&files[0]).map_err(|e| format!("{}: {e}", files[0].display()))?)
        .map_err(|e| format!("{}: {e}", files[0].display()))?;
    let mut lfs = BTreeMap::new();
    for (name, rec) in v["files"].as_object().into_iter().flatten() {
        if let (Some(sha), Some(size)) = (rec["lfs_sha256"].as_str(), rec["lfs_size"].as_u64()) {
            lfs.insert(name.clone(), (sha.to_string(), size));
        }
    }
    Ok(Some(HfTree { revision, lfs }))
}

/// The per-shard record: the HF LFS sha256 when the cache has one for this file AND its size
/// equals the file on disk; else `computed` - hashed here when `compute` is set (the
/// conversion), or left as `None` (the plan, which never reads a shard's payload).
pub fn shard_record(path: &Path, tree: Option<&HfTree>, compute: bool) -> Result<Option<ShardRecord>, String> {
    let file = path.file_name().unwrap().to_string_lossy().to_string();
    let size = std::fs::metadata(path).map_err(|e| format!("{}: {e}", path.display()))?.len();
    if let Some((sha, lfs_size)) = tree.and_then(|t| t.lfs.get(&file)) {
        if *lfs_size != size {
            return Err(format!("{file}: {size} B on disk, but Hugging Face recorded {lfs_size} B - an incomplete or foreign file"));
        }
        return Ok(Some(ShardRecord { file, size, sha256: sha.clone(), sha256_from: "hf-lfs" }));
    }
    if !compute {
        return Ok(None);
    }
    let sha256 = sha256_file(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(Some(ShardRecord { file, size, sha256, sha256_from: "computed" }))
}

/// The geometry the converter derives from `config.json`, written into the index beside the
/// verbatim config. Informational: the engine derives its own `Geo` from `config_json`
/// (`meta.rs`); this block exists so a human, or `converter plan`, can read the shape without a
/// JSON tool, and so the conversion can check the config against the tensors it is converting.
pub fn derive_geo(family: Family, config: &serde_json::Value) -> serde_json::Value {
    let t = &config["text_config"];
    let layer_types: Vec<&str> = t["layer_types"].as_array().into_iter().flatten().filter_map(|v| v.as_str()).collect();
    let attn = layer_types.iter().filter(|s| **s == "full_attention").count();
    let gdn = layer_types.iter().filter(|s| **s == "linear_attention").count();
    let mut g = serde_json::json!({
        "family": family.name(),
        "hidden": t["hidden_size"],
        "layers": t["num_hidden_layers"],
        "attn_layers": attn,
        "gdn_layers": gdn,
        "q_heads": t["num_attention_heads"],
        "kv_heads": t["num_key_value_heads"],
        "head_dim": t["head_dim"],
        "vocab": t["vocab_size"],
        "gdn_k_heads": t["linear_num_key_heads"],
        "gdn_v_heads": t["linear_num_value_heads"],
        "gdn_k_dim": t["linear_key_head_dim"],
        "gdn_v_dim": t["linear_value_head_dim"],
        "conv_kernel": t["linear_conv_kernel_dim"],
        "mtp_layers": t["mtp_num_hidden_layers"],
        "tie_word_embeddings": config["tie_word_embeddings"],
        "vision_depth": config["vision_config"]["depth"],
        "vision_hidden": config["vision_config"]["hidden_size"],
        "vision_out": config["vision_config"]["out_hidden_size"],
    });
    match family {
        Family::FlashNext => {
            g["ffn"] = serde_json::json!("moe");
            g["experts"] = t["num_experts"].clone();
            g["experts_per_tok"] = t["num_experts_per_tok"].clone();
            g["moe_inter"] = t["moe_intermediate_size"].clone();
            g["shared_inter"] = t["shared_expert_intermediate_size"].clone();
        }
        Family::Qwen35Dense => {
            g["ffn"] = serde_json::json!("dense");
            g["inter"] = t["intermediate_size"].clone();
        }
    }
    if let (Some(q), Some(kv)) = (t["num_attention_heads"].as_u64(), t["num_key_value_heads"].as_u64()) {
        if let Some(gqa) = q.checked_div(kv) {
            g["gqa"] = serde_json::json!(gqa);
        }
    }
    g
}

/// Check the config's geometry against the tensors actually being converted: the token
/// embedding's `[vocab, hidden]` and the text layer count. A config copied from another
/// checkpoint would otherwise travel inside the container as if it described it.
pub fn check_geo_against_tensors(geo: &serde_json::Value, tensors: &[(String, Vec<usize>)]) -> Result<(), String> {
    let (Some(hidden), Some(vocab), Some(layers)) = (geo["hidden"].as_u64(), geo["vocab"].as_u64(), geo["layers"].as_u64()) else {
        return Err("config.json: text_config lacks hidden_size, vocab_size or num_hidden_layers".into());
    };
    let Some((_, emb)) = tensors.iter().find(|(n, _)| n.ends_with("language_model.embed_tokens.weight")) else {
        return Err("no model.language_model.embed_tokens.weight among the tensors".into());
    };
    if emb.as_slice() != [vocab as usize, hidden as usize] {
        return Err(format!("embed_tokens is {emb:?}, config.json says [vocab {vocab}, hidden {hidden}] - config and weights disagree"));
    }
    let seen = tensors
        .iter()
        .filter(|(n, _)| section_of(n) == "text")
        .filter_map(|(n, _)| n.strip_prefix("model.language_model.layers.").and_then(|r| r.split('.').next()).and_then(|s| s.parse::<u64>().ok()))
        .max()
        .map(|l| l + 1)
        .unwrap_or(0);
    if seen != layers {
        return Err(format!("the weights carry {seen} text layers, config.json says num_hidden_layers {layers} - config and weights disagree"));
    }
    Ok(())
}

/// Everything the index v2 `model` block is built from.
pub struct ModelSource {
    pub family: Family,
    /// `config.json` and `generation_config.json`, the bytes as read
    pub config_json: String,
    pub generation_config_json: String,
    pub config: serde_json::Value,
    pub repo: String,
    pub revision: String,
    pub shards: Vec<ShardRecord>,
}

/// Read the two config files next to the input (the model directory, or the directory of a
/// single `.safetensors` file) and detect the family. Both files are required: a v2 container
/// carries its config, so a conversion without one is refused.
pub fn read_model_configs(dir: &Path) -> Result<(Family, String, String, serde_json::Value), String> {
    let read = |f: &str| {
        let p = dir.join(f);
        let b = std::fs::read(&p).map_err(|e| format!("{}: {e} - an index v2 container carries the checkpoint's {f}", p.display()))?;
        String::from_utf8(b).map_err(|e| format!("{}: not UTF-8 ({e})", p.display()))
    };
    let config_json = read("config.json")?;
    let generation_config_json = read("generation_config.json")?;
    let config: serde_json::Value = serde_json::from_str(&config_json).map_err(|e| format!("config.json: {e}"))?;
    serde_json::from_str::<serde_json::Value>(&generation_config_json).map_err(|e| format!("generation_config.json: {e}"))?;
    let family = Family::detect(&config)?;
    Ok((family, config_json, generation_config_json, config))
}

impl ModelSource {
    /// The index v2 `model` block. The two config files are JSON STRINGS, not parsed objects:
    /// a string survives the index round trip byte for byte (and so does its sha256), a
    /// re-serialized object would not (key order, number formatting, whitespace).
    pub fn model_block(&self) -> serde_json::Value {
        serde_json::json!({
            "family": self.family.name(),
            "model_type": self.family.model_type(),
            "config_json": self.config_json,
            "config_json_sha256": sha256_hex(self.config_json.as_bytes()),
            "generation_config_json": self.generation_config_json,
            "generation_config_json_sha256": sha256_hex(self.generation_config_json.as_bytes()),
            "geo": derive_geo(self.family, &self.config),
            "source": {
                "repo": self.repo,
                "revision": self.revision,
                "shards": self.shards.iter().map(|s| serde_json::json!({
                    "file": s.file, "size": s.size, "sha256": s.sha256, "sha256_from": s.sha256_from,
                })).collect::<Vec<_>>(),
            },
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shape(s: &str) -> Vec<usize> {
        s.split('x').map(|v| v.parse().unwrap()).collect()
    }

    /// The pre-C6 keep set, frozen verbatim from `converter/src/main.rs` at `64c242b` (`:187-210`
    /// plus the I64 rule at `:728`). The Flash-Next row must agree with it on every input.
    fn keep_bf16_64c242b(name: &str, shape: &[usize], n: usize) -> bool {
        if shape.len() == 1 {
            return true;
        }
        if name.contains("norm")
            || name.contains("embed_tokens")
            || name.contains("lm_head")
            || name.contains("mlp.gate.weight")
            || name.contains("shared_expert_gate")
        {
            return true;
        }
        if name.contains("input_mix_weight_down")
            || name.contains("input_mix_weight_up")
            || name.contains("self_attn.q_proj.weight")
            || name.contains("self_attn.k_proj.weight")
        {
            return true;
        }
        n % 64 != 0 || n < 64
    }

    /// The Flash-Next row reproduces the container of record: every one of the 1658 tensors of
    /// the CNQ4.5-M index trailer gets the dtype and the section it has there, and the frozen
    /// pre-C6 function agrees on each of them.
    #[test]
    fn the_flash_next_row_reproduces_every_tensor_of_cnq45m() {
        let tsv = include_str!("../tests/fixtures/cnq45m-index.tsv");
        let mut n = 0usize;
        let mut by_dtype: BTreeMap<&str, usize> = BTreeMap::new();
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let (name, sh, section, dtype) = (f[0], shape(f[1]), f[2], f[3]);
            let src = if dtype == "i64" { "I64" } else { "BF16" };
            let d = decide(Family::FlashNext, name, &sh, src).unwrap();
            assert_eq!(d.dtype.as_str(), dtype, "{name}: dtype ({})", d.rule);
            assert_eq!(d.section, section, "{name}: section");
            if src != "I64" {
                let old = if keep_bf16_64c242b(name, &sh, sh.iter().product()) { "bf16" } else { "nvfp4" };
                assert_eq!(old, dtype, "{name}: the frozen pre-C6 keep set disagrees with the container");
            }
            *by_dtype.entry(dtype).or_default() += 1;
            n += 1;
        }
        assert_eq!(n, 1658);
        assert_eq!(by_dtype, BTreeMap::from([("bf16", 812), ("i64", 3), ("nvfp4", 843)]));
    }

    /// The latent finding, pinned so it cannot be "fixed" on Flash-Next by accident: the
    /// Flash-Next row quantizes in_proj_a/b and conv1d (as CNQ4.5-M does), the dense row keeps
    /// them BF16.
    #[test]
    fn in_proj_a_b_and_conv1d_are_nvfp4_on_flash_next_and_bf16_on_the_dense_row() {
        for (name, sh) in [
            ("model.language_model.layers.0.linear_attn.in_proj_a.weight", vec![48, 2560]),
            ("model.language_model.layers.0.linear_attn.in_proj_b.weight", vec![48, 2560]),
            ("model.language_model.layers.0.linear_attn.conv1d.weight", vec![10240, 1, 4]),
        ] {
            assert_eq!(decide(Family::FlashNext, name, &sh, "BF16").unwrap().dtype, DtypeOut::Nvfp4, "{name}");
            assert_eq!(decide(Family::Qwen35Dense, name, &sh, "BF16").unwrap().dtype, DtypeOut::Bf16, "{name}");
        }
    }

    /// Every tensor of the 27B (names and shapes from its 18 shard headers) is decided by a
    /// named row of the dense recipe, none is refused, and the per-row counts are the recipe.
    #[test]
    fn the_dense_row_decides_every_tensor_of_the_27b() {
        let tsv = include_str!("../tests/fixtures/qwen3.8-27b-tensors.tsv");
        let mut per: BTreeMap<(&str, &str, &str), usize> = BTreeMap::new();
        let mut n = 0;
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            let d = decide(Family::Qwen35Dense, f[0], &shape(f[1]), f[2]).unwrap_or_else(|e| panic!("{e}"));
            *per.entry((d.section, d.dtype.as_str(), d.rule)).or_default() += 1;
            n += 1;
        }
        assert_eq!(n, 1199);
        let want = BTreeMap::from([
            (("mtp", "bf16", "MTP BF16 (own section)"), 15),
            (("text", "bf16", "GDN conv1d BF16"), 48),
            (("text", "bf16", "GDN in_proj_a/b BF16"), 96),
            (("text", "bf16", "dt_bias BF16"), 48),
            (("text", "bf16", "norm BF16"), 64 + 64 + 48 + 16 + 16 + 1),
            (("text", "bf16", "token embedding BF16 (host RAM)"), 1),
            (("text", "f32", "A_log f32"), 48),
            (("text", "nvfp4", "GDN in_proj_qkv/z, out_proj NVFP4"), 144),
            (("text", "nvfp4", "MLP gate/up/down NVFP4"), 192),
            (("text", "nvfp4", "attention q/k/v/o NVFP4"), 64),
            (("text", "nvfp4", "lm_head NVFP4"), 1),
            (("vit", "bf16", "vision tower BF16 (own section)"), 27 * 12 + 6 + 3),
        ]);
        assert_eq!(per, want);
    }

    /// The dense row is a whitelist: a tensor it has no row for is refused by name, and so is a
    /// PLE table or an I64 tensor in a dense checkpoint.
    #[test]
    fn the_dense_row_refuses_what_it_has_no_row_for() {
        let m = decide(Family::Qwen35Dense, "model.language_model.layers.3.mlp.experts.gate_up_proj", &[512, 1280, 2560], "BF16").unwrap_err();
        assert!(m.contains("matches no row of the cnq4.5-qwen35-dense recipe"), "{m}");
        let m = decide(Family::Qwen35Dense, "model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", &[64, 64], "BF16").unwrap_err();
        assert!(m.contains("no PLE"), "{m}");
        let m = decide(Family::Qwen35Dense, "model.language_model.layers.1.ple.ple_embedding.layer_multipliers", &[3], "I64").unwrap_err();
        assert!(m.contains("I64"), "{m}");
    }

    /// A bf16 keep is the source's raw bytes, so a non-BF16 source under a bf16 decision would
    /// be a mislabelled tensor. Both families refuse it; f32 widening is allowed.
    #[test]
    fn a_bf16_keep_of_a_non_bf16_source_is_refused() {
        for fam in Family::ALL {
            for src in ["F32", "F16"] {
                let m = decide(fam, "model.language_model.norm.weight", &[5120], src).unwrap_err();
                assert!(m.contains("under a bf16 label"), "{fam:?} {src}: {m}");
            }
        }
        assert_eq!(decide(Family::Qwen35Dense, "model.language_model.layers.0.linear_attn.A_log", &[48], "F32").unwrap().dtype, DtypeOut::F32);
    }

    #[test]
    fn the_family_comes_from_text_config_model_type_and_an_unknown_one_is_refused() {
        let cfg = |mt: &str| serde_json::json!({ "text_config": { "model_type": mt } });
        assert_eq!(Family::detect(&cfg("qwen4_exp_text")).unwrap(), Family::FlashNext);
        assert_eq!(Family::detect(&cfg("qwen3_5_text")).unwrap(), Family::Qwen35Dense);
        let m = Family::detect(&cfg("llama")).unwrap_err();
        assert!(m.contains("'llama' is not a model family"), "{m}");
        let m = Family::detect(&serde_json::json!({ "model_type": "qwen3_5" })).unwrap_err();
        assert!(m.contains("no text_config.model_type"), "{m}");
    }

    /// A config that does not describe the weights is refused before anything is written.
    #[test]
    fn a_config_that_disagrees_with_the_weights_is_refused() {
        let geo = serde_json::json!({ "hidden": 64, "vocab": 128, "layers": 2 });
        let ok = vec![
            ("model.language_model.embed_tokens.weight".to_string(), vec![128, 64]),
            ("model.language_model.layers.1.mlp.up_proj.weight".to_string(), vec![128, 64]),
            ("mtp.layers.5.mlp.up_proj.weight".to_string(), vec![128, 64]),
        ];
        check_geo_against_tensors(&geo, &ok).unwrap();
        let mut wrong = ok.clone();
        wrong[0].1 = vec![128, 32];
        assert!(check_geo_against_tensors(&geo, &wrong).unwrap_err().contains("config and weights disagree"));
        let mut layers = ok.clone();
        layers[1].0 = "model.language_model.layers.2.mlp.up_proj.weight".into();
        assert!(check_geo_against_tensors(&geo, &layers).unwrap_err().contains("3 text layers"));
    }

    /// The HF local-dir cache gives the revision and the LFS sha256; a size that does not match
    /// the file on disk is refused, a file with no LFS record is hashed only when asked.
    #[test]
    fn shard_sha256_comes_from_the_hf_lfs_record_or_is_computed() {
        let dir = std::env::temp_dir().join(format!("cnq-c6-hf-{}", std::process::id()));
        let trees = dir.join(".cache/huggingface/trees");
        std::fs::create_dir_all(&trees).unwrap();
        std::fs::write(dir.join("a.safetensors"), b"abc").unwrap();
        std::fs::write(dir.join("b.safetensors"), b"abc").unwrap();
        std::fs::write(dir.join("c.safetensors"), b"abc").unwrap();
        std::fs::write(
            trees.join("0123abcd.json"),
            br#"{"format_version":1,"files":{"a.safetensors":{"size":3,"lfs_sha256":"feed","lfs_size":3},"c.safetensors":{"size":3,"lfs_sha256":"feed","lfs_size":4}}}"#,
        )
        .unwrap();
        let t = read_hf_tree(&dir).unwrap().unwrap();
        assert_eq!(t.revision, "0123abcd");
        let a = shard_record(&dir.join("a.safetensors"), Some(&t), false).unwrap().unwrap();
        assert_eq!((a.sha256.as_str(), a.sha256_from), ("feed", "hf-lfs"));
        assert!(shard_record(&dir.join("b.safetensors"), Some(&t), false).unwrap().is_none());
        let b = shard_record(&dir.join("b.safetensors"), Some(&t), true).unwrap().unwrap();
        // sha256("abc"), FIPS 180-2 appendix B.1
        assert_eq!((b.sha256.as_str(), b.sha256_from), ("ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad", "computed"));
        let m = shard_record(&dir.join("c.safetensors"), Some(&t), true).unwrap_err();
        assert!(m.contains("Hugging Face recorded 4 B"), "{m}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
