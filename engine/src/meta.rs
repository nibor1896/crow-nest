//! #94 phase 1 — the metadata gate: the checkpoint's config parsed at boot and
//! asserted against the pinned constants, with ZERO numeric change.
//!
//! Every formula constant this engine computes with is compile-time pinned
//! (geo.rs calls itself "probe-pinned" on purpose): rms eps 1e-6 in every norm
//! kernel, attention scale 0.0625 in five kernel variants, rope theta 1e7 in
//! the boot table, GQA `head / 12`, ROPE_PAIRS 32, H 2560, vocab 248320, the
//! EOS pair in `sample`. All of it is correct for Qwen3.8-Flash-Next — and a
//! different checkpoint would compute silently wrong values everywhere. The
//! llama.cpp discipline (their `get_key` ledger: #7327, #14892, #28068) is that
//! metadata is READ, never assumed, and that an unknown or deviating value is a
//! loud, NAMED load failure.
//!
//! Phase 1 is the plumbing, not the migration: at boot — before the container
//! is mapped, before the CUDA context exists — `assert_pinned` reads
//! `models/<name>/config.json` + `generation_config.json` next to the container
//! (the CNQ trailer carries only quant geometry: `blob_offset`, `block_geometry`,
//! `format`, `sections`, `tensors`, `version` — no model constants, verified
//! 2026-09-20), derives every constant the way the checkpoint says it should be
//! derived, and compares each against the pin. Green on the checkpoint of
//! record proves the plumbing; a future checkpoint that differs dies at the
//! front door with a table of every wrong constant instead of degrading output
//! quietly for days. No kernel, no math, no pinned value changed.
//!
//! Deviations that are decisions, not gaps:
//!
//! - **config.json absent → one WARN line, boot continues.** The selftest
//!   package (`tools/selftest.sh`) deliberately ships WITHOUT `models/` —
//!   "without the originals" is the whole positive control — so a hard refusal
//!   there would break the release gate on the download package. This is the
//!   llama.cpp "missing fields" WARNING class; everything below it (file there
//!   but unparseable / missing a field / a value that differs / a special id
//!   out of vocab range) is the hard-error class and panics.
//! - **`CROW_MODEL_DIR`** names the checkpoint dir when it is not the
//!   `models/` sibling of the container.
//! - `sample.rs` keeps its EOS pin; this module READS `sample::EOS_IDS` and
//!   asserts the config against it (the pin and the truth stay one concept,
//!   written once in sample, compared once here).

use crate::geo;
use crate::sample;
use serde_json::Value;

/// one constant of the gate: the pinned engine value against the value derived
/// from the checkpoint config, `ok` when they are equal. The name is stable —
/// the unit tests and the panic table key on it.
pub struct Check {
    pub name: &'static str,
    /// the pin: value + where it lives in the engine
    pub pinned: String,
    /// what the config says, rendered
    pub config: String,
    /// the config key(s) the value came from
    pub source: String,
    pub ok: bool,
}

impl Check {
    fn cmp<T: PartialEq + std::fmt::Debug>(name: &'static str, pinned: T, config: T, pin_at: &str, source: &str) -> Check {
        Check {
            name,
            pinned: format!("{pinned:?} ({pin_at})"),
            config: format!("{config:?}"),
            source: source.to_string(),
            ok: pinned == config,
        }
    }
    /// the one-line form of a failing check, as the panic table renders it
    pub fn line(&self) -> String {
        format!("{}: pinned {}, config {} ({})", self.name, self.pinned, self.config, self.source)
    }
}

/// the model metadata the gate is built from — every field REQUIRED at parse
/// time (a missing key is a named error, never a guess), except `bos`, which
/// some checkpoints only carry in one of the two files.
#[derive(Debug)]
pub struct ModelMeta {
    /// the config.json this was parsed from (for the log lines)
    pub config_path: String,
    /// the generation_config.json next to it, when there was one
    pub generation_config_path: Option<String>,
    pub model_type: String,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    /// which config key supplied `rope_theta` (it nests in `rope_parameters`)
    pub rope_theta_source: String,
    pub partial_rotary_factor: f64,
    pub head_dim: u64,
    pub hidden_size: u64,
    pub num_attention_heads: u64,
    pub num_key_value_heads: u64,
    pub num_hidden_layers: u64,
    /// the hybrid layout, one entry per layer ("linear_attention"/"full_attention")
    pub layer_types: Vec<String>,
    pub full_attention_interval: Option<u64>,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    /// every generation-stop id, and whether generation_config.json supplied
    /// them (else the text config's single id did)
    pub eos_token_ids: Vec<i64>,
    pub eos_from_generation: bool,
    /// the text config's OWN eos id — 248044 here, the id that doubles as the
    /// PLE shard end marker (`geo::PLE_EOS`)
    pub text_eos_token_id: Option<i64>,
    /// the generation bos (either file), and the text config's own bos for the
    /// cross-file agreement check
    pub bos_token_id: Option<i64>,
    pub text_bos_token_id: Option<i64>,
    /// `mrope_section` when the config carries it ([11, 11, 10] here)
    pub mrope_section: Option<Vec<u64>>,
}

// ---- parsing ----

/// the config object the text-tower constants live in: `text_config` when the
/// checkpoint nests it (Qwen multimodal form), the top level when it is flat
fn text_config(config: &Value) -> &Value {
    match config.get("text_config") {
        Some(tc) if tc.is_object() => tc,
        _ => config,
    }
}

fn f64_of(v: &Value, key: &str) -> Option<f64> {
    v.get(key).and_then(Value::as_f64)
}

fn u64_of(v: &Value, key: &str) -> Option<u64> {
    v.get(key).and_then(Value::as_u64)
}

/// eos/bos ids appear as one int or as an array in either file
fn ids_of(v: &Value, key: &str) -> Option<Vec<i64>> {
    match v.get(key) {
        Some(Value::Array(a)) => Some(a.iter().filter_map(Value::as_i64).collect()),
        Some(Value::Number(n)) => n.as_i64().map(|id| vec![id]),
        _ => None,
    }
}

fn first_id(v: &Value, key: &str) -> Option<i64> {
    ids_of(v, key).and_then(|ids| ids.first().copied())
}

impl ModelMeta {
    /// Parse a config.json (+ optional generation_config.json). `Err` carries
    /// every missing/malformed key by name — the llama.cpp rule: a missing
    /// field is a hard, named error, never a defaulted guess.
    pub fn from_config_files(config_path: &str, generation_config_path: Option<&str>) -> Result<ModelMeta, String> {
        let config_text = std::fs::read_to_string(config_path).map_err(|e| format!("{config_path}: {e}"))?;
        let config: Value =
            serde_json::from_str(&config_text).map_err(|e| format!("{config_path}: not valid json: {e}"))?;
        let tc = text_config(&config);

        // rope fields nest in `rope_parameters` in this checkpoint; a flat
        // `rope_theta` / `partial_rotary_factor` is the fallback of the same read
        let rp = tc.get("rope_parameters").filter(|r| r.is_object());
        let (rope_theta, rope_theta_source) = match rp.and_then(|r| f64_of(r, "rope_theta")) {
            Some(t) => (t, "text_config.rope_parameters.rope_theta".to_string()),
            None => match f64_of(tc, "rope_theta") {
                Some(t) => (t, "text_config.rope_theta".to_string()),
                None => (f64::NAN, String::new()),
            },
        };
        let partial_rotary_factor = match rp.and_then(|r| f64_of(r, "partial_rotary_factor")) {
            Some(f) => Some((f, "text_config.rope_parameters.partial_rotary_factor")),
            None => f64_of(tc, "partial_rotary_factor").map(|f| (f, "text_config.partial_rotary_factor")),
        };
        let mrope_section = rp
            .and_then(|r| r.get("mrope_section"))
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(Value::as_u64).collect());

        // generation_config.json is the source of record for the stop ids; the
        // text config's own eos/bos is the fallback and the cross-check
        let generation = match generation_config_path {
            Some(p) => {
                let s = std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?;
                Some(serde_json::from_str::<Value>(&s).map_err(|e| format!("{p}: not valid json: {e}"))?)
            }
            None => None,
        };
        let eos_from_generation = generation.as_ref().and_then(|g| ids_of(g, "eos_token_id")).is_some();
        let eos_token_ids = generation
            .as_ref()
            .and_then(|g| ids_of(g, "eos_token_id"))
            .or_else(|| ids_of(tc, "eos_token_id"));
        let bos_token_id = generation
            .as_ref()
            .and_then(|g| first_id(g, "bos_token_id"))
            .or_else(|| first_id(tc, "bos_token_id"));
        let text_eos_token_id = first_id(tc, "eos_token_id");
        let text_bos_token_id = first_id(tc, "bos_token_id");

        // collect EVERY missing key in one pass, so the error names them all
        let mut missing: Vec<String> = Vec::new();
        let mut need = |cond: bool, key: &str| {
            if !cond {
                missing.push(key.to_string());
            }
        };
        let layer_types: Vec<String> = tc
            .get("layer_types")
            .and_then(Value::as_array)
            .map(|a| a.iter().filter_map(|v| v.as_str().map(str::to_string)).collect())
            .unwrap_or_default();
        need(f64_of(tc, "rms_norm_eps").is_some(), "text_config.rms_norm_eps");
        need(!rope_theta_source.is_empty(), "text_config.rope_parameters.rope_theta (or text_config.rope_theta)");
        need(partial_rotary_factor.is_some(), "text_config.partial_rotary_factor (or rope_parameters.partial_rotary_factor)");
        need(u64_of(tc, "head_dim").is_some(), "text_config.head_dim");
        need(u64_of(tc, "hidden_size").is_some(), "text_config.hidden_size");
        need(u64_of(tc, "num_attention_heads").is_some(), "text_config.num_attention_heads");
        need(u64_of(tc, "num_key_value_heads").is_some(), "text_config.num_key_value_heads");
        need(u64_of(tc, "num_hidden_layers").is_some(), "text_config.num_hidden_layers");
        need(!layer_types.is_empty(), "text_config.layer_types");
        need(u64_of(tc, "vocab_size").is_some(), "text_config.vocab_size");
        need(u64_of(tc, "max_position_embeddings").is_some(), "text_config.max_position_embeddings");
        need(eos_token_ids.as_ref().is_some_and(|v| !v.is_empty()), "generation_config.json eos_token_id (or text_config.eos_token_id)");
        if !missing.is_empty() {
            return Err(format!("{config_path}: missing required key(s): {}", missing.join(", ")));
        }

        Ok(ModelMeta {
            config_path: config_path.to_string(),
            generation_config_path: generation_config_path.map(str::to_string),
            model_type: tc.get("model_type").and_then(Value::as_str).unwrap_or("?").to_string(),
            rms_norm_eps: f64_of(tc, "rms_norm_eps").unwrap(),
            rope_theta,
            rope_theta_source,
            partial_rotary_factor: partial_rotary_factor.unwrap().0,
            head_dim: u64_of(tc, "head_dim").unwrap(),
            hidden_size: u64_of(tc, "hidden_size").unwrap(),
            num_attention_heads: u64_of(tc, "num_attention_heads").unwrap(),
            num_key_value_heads: u64_of(tc, "num_key_value_heads").unwrap(),
            num_hidden_layers: u64_of(tc, "num_hidden_layers").unwrap(),
            layer_types,
            full_attention_interval: u64_of(tc, "full_attention_interval"),
            vocab_size: u64_of(tc, "vocab_size").unwrap(),
            max_position_embeddings: u64_of(tc, "max_position_embeddings").unwrap(),
            eos_token_ids: eos_token_ids.unwrap(),
            eos_from_generation,
            text_eos_token_id,
            bos_token_id,
            text_bos_token_id,
            mrope_section,
        })
    }

    // ---- the comparisons ----

    /// every check, green and red, in table order. The count is what the boot
    /// INFO line reports as "N constants verified".
    pub fn checks(&self) -> Vec<Check> {
        let mut c = Vec::with_capacity(20);
        c.push(Check::cmp("rms_norm_eps", 1e-6f64, self.rms_norm_eps, "kernels.rs, every rms + LayerNorm site", "text_config.rms_norm_eps"));
        c.push(Check::cmp("rope_theta", 1e7f64, self.rope_theta, "manager.rs boot RoPE table", &self.rope_theta_source));
        // partial rotary: factor x head_dim rotary dims, ROPE_PAIRS of them
        let pairs = self.partial_rotary_factor * self.head_dim as f64 / 2.0;
        c.push(Check {
            name: "rope_pairs",
            pinned: format!("{:?} (geo::ROPE_PAIRS)", geo::ROPE_PAIRS),
            config: format!("{pairs}"),
            source: format!("{} x head_dim {} / 2", self.partial_rotary_factor, self.head_dim),
            ok: pairs == geo::ROPE_PAIRS as f64,
        });
        // the attention scale every variant hardcodes is 1/sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        c.push(Check::cmp("attention_scale", 0.0625f64, scale, "kernels.rs, 5 attention variants", "1/sqrt(text_config.head_dim)"));
        // GQA: the kernels derive kvh from the head count with a pinned divisor
        let gqa = if self.num_key_value_heads > 0 && self.num_attention_heads % self.num_key_value_heads == 0 {
            self.num_attention_heads / self.num_key_value_heads
        } else {
            0
        };
        c.push(Check::cmp("gqa_ratio", 12u64, gqa, "kernels.rs `head / 12`", "num_attention_heads / num_key_value_heads"));
        c.push(Check::cmp("hidden_size", geo::H as u64, self.hidden_size, "geo::H", "text_config.hidden_size"));
        c.push(Check::cmp("head_dim", geo::AHD as u64, self.head_dim, "geo::AHD", "text_config.head_dim"));
        c.push(Check::cmp("q_heads", geo::NQ as u64, self.num_attention_heads, "geo::NQ", "text_config.num_attention_heads"));
        c.push(Check::cmp("kv_heads", geo::NKV as u64, self.num_key_value_heads, "geo::NKV", "text_config.num_key_value_heads"));
        c.push(Check::cmp("num_hidden_layers", geo::LAYERS as u64, self.num_hidden_layers, "geo::LAYERS", "text_config.num_hidden_layers"));
        // the hybrid layout: geo dispatches on `layer % 4 == 3`; the config's
        // layer_types must be exactly that layout, with the pinned layer counts
        let unknown: Vec<&str> = self
            .layer_types
            .iter()
            .map(|s| s.as_str())
            .filter(|s| *s != "linear_attention" && *s != "full_attention")
            .collect();
        let full_at: Vec<usize> = self
            .layer_types
            .iter()
            .enumerate()
            .filter(|(_, t)| t.as_str() == "full_attention")
            .map(|(i, _)| i)
            .collect();
        let pinned_at: Vec<usize> = (0..geo::LAYERS).filter(|i| geo::is_attn(*i)).collect();
        let layout_ok = self.layer_types.len() == geo::LAYERS
            && full_at == pinned_at
            && full_at.len() == geo::ATTN_LAYERS
            && self.layer_types.len() - full_at.len() == geo::GDN_LAYERS
            && unknown.is_empty();
        c.push(Check {
            name: "layer_layout",
            pinned: format!(
                "layer % 4 == 3 is full attention: {} attn + {} gdn of {} (geo::is_attn, ATTN_LAYERS, GDN_LAYERS)",
                geo::ATTN_LAYERS, geo::GDN_LAYERS, geo::LAYERS
            ),
            config: format!(
                "{} layers, {} full_attention at {:?}, {} other{}",
                self.layer_types.len(),
                full_at.len(),
                &full_at[..full_at.len().min(8)],
                self.layer_types.len() - full_at.len(),
                if unknown.is_empty() { String::new() } else { format!(", UNKNOWN types {unknown:?}") }
            ),
            source: "text_config.layer_types".to_string(),
            ok: layout_ok,
        });
        if let Some(interval) = self.full_attention_interval {
            c.push(Check::cmp("full_attention_interval", 4u64, interval, "geo::is_attn `layer % 4`", "text_config.full_attention_interval"));
        }
        c.push(Check::cmp("vocab_size", geo::V as u64, self.vocab_size, "geo::V", "text_config.vocab_size"));
        // Config::default().context is the checkpoint's context budget; the
        // 200_000 floor is an engine policy, not a model constant
        c.push(Check::cmp(
            "max_position_embeddings",
            geo::Config::default().context as u64,
            self.max_position_embeddings,
            "geo::Config::default().context",
            "text_config.max_position_embeddings",
        ));
        // the sampler's stop ids — sample.rs keeps the pin, this reads it live
        let config_eos: Vec<usize> = self.eos_token_ids.iter().map(|id| *id as usize).collect();
        c.push(Check {
            name: "eos_ids",
            pinned: format!("{:?} (sample::EOS_IDS)", sample::EOS_IDS),
            config: format!("{:?}", self.eos_token_ids),
            source: if self.eos_from_generation {
                "generation_config.json eos_token_id".to_string()
            } else {
                "text_config.eos_token_id".to_string()
            },
            ok: config_eos == sample::EOS_IDS.to_vec(),
        });
        // 248044 doubles as the PLE shard end marker (geo::PLE_EOS): the text
        // config's own eos must BE that id, or the PLE reader and the sampler
        // disagree about what "end" means
        c.push(Check {
            name: "ple_eos",
            pinned: format!("{:?} (geo::PLE_EOS, the PLE shard end marker / gen.rs filler)", geo::PLE_EOS),
            config: format!("{:?}", self.text_eos_token_id),
            source: "text_config.eos_token_id".to_string(),
            ok: self.text_eos_token_id == Some(geo::PLE_EOS),
        });
        // llama.cpp special-id discipline: every stop/start id inside the vocab
        let vocab = self.vocab_size as i64;
        let eos_in = !self.eos_token_ids.is_empty() && self.eos_token_ids.iter().all(|id| *id >= 0 && *id < vocab);
        c.push(Check {
            name: "eos_ids_in_vocab",
            pinned: format!("0 <= id < {}", geo::V),
            config: format!("{:?} against vocab {}", self.eos_token_ids, self.vocab_size),
            source: "generation_config.json eos_token_id vs text_config.vocab_size".to_string(),
            ok: eos_in,
        });
        if let Some(bos) = self.bos_token_id {
            c.push(Check {
                name: "bos_id_in_vocab",
                pinned: format!("0 <= id < {}", geo::V),
                config: format!("{bos} against vocab {}", self.vocab_size),
                source: "generation_config.json bos_token_id vs text_config.vocab_size".to_string(),
                ok: bos >= 0 && bos < vocab,
            });
        }
        // the two files must agree about bos when both carry it
        if let (Some(gen_bos), Some(text_bos)) = (self.bos_token_id, self.text_bos_token_id) {
            c.push(Check::cmp("bos_id_agrees", text_bos, gen_bos, "text_config.bos_token_id", "generation_config.json bos_token_id"));
        }
        // the vit mrope sections sum to the SAME pair count the rope table builds
        if let Some(sec) = &self.mrope_section {
            let sum: u64 = sec.iter().sum();
            c.push(Check::cmp("mrope_section_pairs", geo::ROPE_PAIRS as u64, sum, "geo::ROPE_PAIRS (manager.rs table)", "text_config.rope_parameters.mrope_section sum"));
        }
        c
    }

    /// the named mismatches — [`ModelMeta::checks`] with the green rows
    /// removed. Empty means the plumbing is byte-identical to the pins.
    pub fn verify(&self) -> Vec<Check> {
        self.checks().into_iter().filter(|c| !c.ok).collect()
    }
}

// ---- locating the checkpoint's config next to the container ----

/// `CROW_MODEL_DIR` wins; otherwise `models/` beside the container (the
/// container lives in `<repo>/converter/`, so the repo root is its parent
/// directory's parent — the same convention both cwd forms of the bins give:
/// `converter/x.cnq` from the repo root, `../converter/x.cnq` from `engine/`).
/// `Ok(None)` = no config found (the selftest package): the caller warns.
pub fn from_container(cnq_path: &str) -> Result<Option<ModelMeta>, String> {
    let config_dir = if let Ok(dir) = std::env::var("CROW_MODEL_DIR") {
        if dir.is_empty() {
            return Err("CROW_MODEL_DIR is set but empty".to_string());
        }
        Some(std::path::PathBuf::from(dir))
    } else {
        config_dir_for_container(cnq_path)
    };
    let Some(dir) = config_dir else {
        return Ok(None);
    };
    let config = dir.join("config.json");
    if !config.is_file() {
        // a dir with no config.json is a broken tree, not a "keep going"
        return Err(format!("{}: no config.json in it", dir.display()));
    }
    let generation = dir.join("generation_config.json");
    let generation = if generation.is_file() { Some(generation.to_string_lossy().into_owned()) } else { None };
    Ok(Some(ModelMeta::from_config_files(
        &config.to_string_lossy(),
        generation.as_deref(),
    )?))
}

/// the `models/<name>` dir of the checkpoint this container was built from, or
/// `None` when the tree holds no candidate. When several checkpoints sit under
/// `models/`, the one whose directory name carries the container's model name
/// (the file stem up to `-CNQ…`) wins; a lone candidate needs no hint.
pub fn config_dir_for_container(cnq_path: &str) -> Option<std::path::PathBuf> {
    let hint = std::path::Path::new(cnq_path)
        .file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .map(|stem| {
            stem.split_once("-CNQ")
                .or_else(|| stem.split_once("-cnq"))
                .map(|(h, _)| h.to_string())
                .unwrap_or(stem)
        })
        .unwrap_or_default();
    let p = std::path::Path::new(cnq_path);
    // `…/repo/converter/x.cnq` -> `…/repo`; `../converter/x.cnq` -> `..`
    for root in p.parent().and_then(|d| d.parent()).into_iter().chain([std::path::Path::new(".")]) {
        if let Some(dir) = config_dir_in_root(root, &hint) {
            return Some(dir);
        }
    }
    None
}

fn config_dir_in_root(root: &std::path::Path, hint: &str) -> Option<std::path::PathBuf> {
    let read = std::fs::read_dir(root.join("models")).ok()?;
    let mut cands: Vec<std::path::PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .collect();
    cands.sort();
    match cands.len() {
        0 => None,
        1 => Some(cands[0].clone()),
        // several checkpoints: take the one the container's name names
        _ => cands
            .into_iter()
            .find(|c| c.file_name().map(|n| n.to_string_lossy().starts_with(hint)).unwrap_or(false)),
    }
}

// ---- the boot door ----

/// The gate `boot::open_model` calls FIRST, before the container is mapped and
/// the CUDA context created:
///
/// - config.json found and every check green → one INFO line, the meta returned
///   for the later phases;
/// - config.json found and anything red or unreadable → a panic carrying the
///   whole table (llama.cpp-style loud failure);
/// - no config.json anywhere next to the container → one WARN line, boot
///   continues (the selftest package ships without `models/` on purpose).
pub fn assert_pinned(cnq_path: &str) -> Option<ModelMeta> {
    let meta = match from_container(cnq_path) {
        Ok(Some(meta)) => meta,
        Ok(None) => {
            tracing::warn!(
                target: "meta",
                "meta: no config.json beside {cnq_path} - 0 constants verified, the pins stand unchecked \
(set CROW_MODEL_DIR to the checkpoint directory to arm the #94 gate)"
            );
            return None;
        }
        Err(why) => panic!("{why} - the #94 metadata gate refuses to boot on an unreadable truth source"),
    };
    let all = meta.checks();
    let bad: Vec<&Check> = all.iter().filter(|c| !c.ok).collect();
    if !bad.is_empty() {
        let table = bad.iter().map(|c| format!("  {}", c.line())).collect::<Vec<_>>().join("\n");
        panic!(
            "[meta] {} of {} constants differ from the engine pins - refusing to boot (issue #94):\n{}\n\
[meta]   config: {}\n\
[meta] a different checkpoint must be ported consciously, not silently - see docs/acceptance/issue-94.md",
            bad.len(),
            all.len(),
            table,
            meta.config_path
        );
    }
    tracing::info!(
        target: "meta",
        "meta: {} constants verified against config.json (zero numeric change) [{}]",
        all.len(),
        meta.config_path
    );
    Some(meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// the checkpoint of record, as seen from the engine crate the tests run in
    const REAL_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../models/Qwen3.8-Flash-Next-original");

    fn real_meta() -> ModelMeta {
        ModelMeta::from_config_files(&format!("{REAL_DIR}/config.json"), Some(&format!("{REAL_DIR}/generation_config.json"))).unwrap()
    }

    /// write a doctored copy of the real config (+ generation config) into a
    /// temp dir and parse it — every red test below doctors the truth, never the pin
    fn doctored(mutate: impl FnOnce(&mut Value, &mut Value)) -> ModelMeta {
        let mut config: Value = serde_json::from_str(&std::fs::read_to_string(format!("{REAL_DIR}/config.json")).unwrap()).unwrap();
        let mut generation: Value = serde_json::from_str(&std::fs::read_to_string(format!("{REAL_DIR}/generation_config.json")).unwrap()).unwrap();
        mutate(&mut config, &mut generation);
        // tests run in parallel threads of ONE process, so the pid alone is not
        // a unique dir: every doctored copy gets its own counter slot
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("crow-meta-test-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
        let m = ModelMeta::from_config_files(&dir.join("config.json").to_string_lossy(), Some(&dir.join("generation_config.json").to_string_lossy())).unwrap();
        std::fs::remove_dir_all(&dir).ok();
        m
    }

    fn fired(meta: &ModelMeta, name: &str) -> bool {
        meta.verify().iter().any(|c| c.name == name)
    }

    /// the checkpoint of record is ALL GREEN - the phase-1 proof: the plumbing
    /// reads back every pinned constant byte-identically
    #[test]
    fn the_real_config_passes_every_check() {
        let meta = real_meta();
        let all = meta.checks();
        assert!(all.len() >= 20, "the gate must carry the full constant table, got {}", all.len());
        for c in &all {
            assert!(c.ok, "expected green: {}", c.line());
        }
        assert!(meta.verify().is_empty());
        assert_eq!(meta.model_type, "qwen4_exp_text");
    }

    /// both cwd forms the bins use — repo root (`converter/…`, serve/parity)
    /// and engine/ (`../converter/…`, decode) — must resolve to the checkpoint
    #[test]
    fn the_container_convention_finds_the_repo_checkpoint() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let root = std::path::Path::new(manifest).parent().unwrap();
        for cnq in [
            root.join("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq").to_string_lossy().into_owned(),
            format!("{}/../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", manifest),
        ] {
            let dir = config_dir_for_container(&cnq).expect("the repo layout must resolve");
            assert!(dir.join("config.json").is_file(), "{dir:?}");
        }
    }

    #[test]
    fn a_doctored_eps_fires_by_name() {
        let m = doctored(|c, _| c["text_config"]["rms_norm_eps"] = json!(1e-5));
        assert!(fired(&m, "rms_norm_eps"));
    }

    #[test]
    fn a_doctored_theta_fires_by_name() {
        let m = doctored(|c, _| c["text_config"]["rope_parameters"]["rope_theta"] = json!(5e6));
        assert!(fired(&m, "rope_theta"));
    }

    #[test]
    fn a_doctored_partial_rotary_fires_rope_pairs() {
        let m = doctored(|c, _| {
            c["text_config"]["rope_parameters"]["partial_rotary_factor"] = json!(0.5);
            c["text_config"]["partial_rotary_factor"] = json!(0.5);
        });
        assert!(fired(&m, "rope_pairs"));
    }

    #[test]
    fn a_doctored_head_dim_fires_scale_and_pairs_and_head_dim() {
        let m = doctored(|c, _| c["text_config"]["head_dim"] = json!(128));
        assert!(fired(&m, "attention_scale"));
        assert!(fired(&m, "rope_pairs"));
        assert!(fired(&m, "head_dim"));
    }

    #[test]
    fn a_doctored_hidden_size_fires_by_name() {
        let m = doctored(|c, _| c["text_config"]["hidden_size"] = json!(4096));
        assert!(fired(&m, "hidden_size"));
    }

    #[test]
    fn doctored_head_counts_fire_gqa_and_the_pins() {
        let m = doctored(|c, _| c["text_config"]["num_attention_heads"] = json!(32));
        assert!(fired(&m, "gqa_ratio"));
        assert!(fired(&m, "q_heads"));
        let m = doctored(|c, _| c["text_config"]["num_key_value_heads"] = json!(4));
        assert!(fired(&m, "gqa_ratio"));
        assert!(fired(&m, "kv_heads"));
    }

    #[test]
    fn a_doctored_layer_count_and_layout_fire_by_name() {
        let m = doctored(|c, _| c["text_config"]["num_hidden_layers"] = json!(24));
        assert!(fired(&m, "num_hidden_layers"));
        // move ONE full_attention onto a linear slot: count stays 12, the positions break
        let m = doctored(|c, _| {
            c["text_config"]["layer_types"][3] = json!("linear_attention");
            c["text_config"]["layer_types"][4] = json!("full_attention");
        });
        assert!(fired(&m, "layer_layout"));
        // an unknown layer type is named, not guessed
        let m = doctored(|c, _| c["text_config"]["layer_types"][3] = json!("sliding_attention"));
        assert!(fired(&m, "layer_layout"));
        assert!(m.verify().iter().any(|c| c.config.contains("UNKNOWN")));
        // the interval is what geo's `layer % 4` dispatch is pinned to
        let m = doctored(|c, _| c["text_config"]["full_attention_interval"] = json!(8));
        assert!(fired(&m, "full_attention_interval"));
    }

    #[test]
    fn doctored_vocab_and_positions_fire_by_name() {
        let m = doctored(|c, _| c["text_config"]["vocab_size"] = json!(151936));
        assert!(fired(&m, "vocab_size"));
        let m = doctored(|c, _| c["text_config"]["max_position_embeddings"] = json!(32768));
        assert!(fired(&m, "max_position_embeddings"));
    }

    #[test]
    fn doctored_generation_ids_fire_by_name() {
        let m = doctored(|_, g| g["eos_token_id"] = json!([151645, 151643]));
        assert!(fired(&m, "eos_ids"));
        // bos inside the vocab but disagreeing between the two files
        let m = doctored(|_, g| g["bos_token_id"] = json!(151643));
        assert!(fired(&m, "bos_id_agrees"));
    }

    #[test]
    fn an_out_of_range_special_id_is_a_named_mismatch() {
        let m = doctored(|_, g| g["eos_token_id"] = json!([999999999]));
        assert!(fired(&m, "eos_ids_in_vocab"));
        assert!(fired(&m, "eos_ids"));
    }

    #[test]
    fn a_missing_field_is_a_named_error() {
        let mut config: Value = serde_json::from_str(&std::fs::read_to_string(format!("{REAL_DIR}/config.json")).unwrap()).unwrap();
        config["text_config"].as_object_mut().unwrap().remove("head_dim");
        let dir = std::env::temp_dir().join(format!("crow-meta-missing-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let err = ModelMeta::from_config_files(&dir.join("config.json").to_string_lossy(), None).unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(err.contains("head_dim"), "the error must name the key: {err}");
    }
}
