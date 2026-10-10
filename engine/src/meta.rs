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
//!   `models/` sibling of the container. Crow #300 C7: that is the index v1
//!   path (the CNQ4.5-M container of record). An index v2 container carries its
//!   config in its `model` block and the gate reads it from there; a set
//!   `CROW_MODEL_DIR` is then only a sha256 cross-check ([`gate`]).
//! - `sample.rs` keeps its EOS pin; this module READS `sample::EOS_IDS` and
//!   asserts the config against it (the pin and the truth stay one concept,
//!   written once in sample, compared once here).
//!
//! #96 extends the same reader with `rope_scaling`: the object is parsed when
//! the config carries one (absent on the checkpoint of record = no scaling,
//! byte-identical everywhere), an unsupported type or a missing field is the
//! hard-error class, and the green boot path stashes the scaling + the training
//! context for the two readers that build the rope table (manager.rs) and
//! thread the YaRN mscale into the attention scale (kernels.rs).
//!
//! Crow #300 phase 1 (C1 + C2) makes the reader family-aware:
//!
//! - **family detection** from `text_config.model_type`: `qwen4_exp_text` is
//!   Flash-Next, `qwen3_5_text` the dense Qwen3.5/3.8 family (Qwen3.8-27B); any
//!   other string refuses by name.
//! - **one expected-values row per family** ([`Expected`]): the Flash-Next row
//!   IS today's pins (read out of `geo` and `sample`, the same 21 checks as
//!   before); the dense row is the Qwen/Qwen3.8-27B @ 1d4bf0f2 checkpoint.
//! - **unknown `text_config` keys refuse the parse by name.** Every key is
//!   either consumed (it feeds a check or the `Geo`), or on a named ignore list
//!   with the reason it has no effect on inference ([`IGNORED_KEYS`]); keys of
//!   the other family count as unknown.
//! - **the runtime [`geo::Geo`]** ([`ModelMeta::geo`]) derived from the config.
//!   A Flash-Next boot asserts it equal to `Geo::FLASH_NEXT` (a mismatch table
//!   otherwise); a dense boot prints it and passes it on. C5: the refusal of a
//!   family moved from here to its first unbuilt block (`Geo::built`, called by
//!   `boot::model_geo` right after this gate, still before the container and CUDA).

use crate::geo;
use crate::geo::{Attn, Family, FinalNorm, Ffn, GateAct, Geo, Glm5Geo, PleGeo, Residual, GLM5_NEXT_MODEL_TYPE, GLM5_NEXT_SOURCE};
use crate::cnq::{sha256_hex, IndexPeek, ModelBlock};
use crate::sample;
use serde_json::Value;

// ---- C1: the family table ----

/// the source of the dense row of [`Expected`]
pub const QWEN35_DENSE_SOURCE: &str = "Qwen/Qwen3.8-27B @ 1d4bf0f2 config.json";

impl Family {
    /// family detection: `text_config.model_type` (else the flat top level's)
    pub fn detect(model_type: &str) -> Result<Family, String> {
        Family::KNOWN.into_iter().find(|f| f.model_type() == model_type).ok_or_else(|| {
            format!(
                "text_config.model_type '{model_type}' is not a model family this engine knows \
(known: qwen4_exp_text = Qwen3.8-Flash-Next, qwen3_5_text = dense Qwen3.5/3.8, glm5_next_text = GLM-5.3-Flash) - refusing (Crow #300)"
            )
        })
    }
}

/// the expected values of one family: what the config must say for the
/// engine's pins (Flash-Next) or the family row (dense) to hold
#[derive(Clone, Copy, Debug)]
pub struct Expected {
    pub family: Family,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub rope_pairs: usize,
    pub attention_scale: f64,
    pub gqa_ratio: u64,
    pub hidden_size: u64,
    pub head_dim: u64,
    pub q_heads: u64,
    pub kv_heads: u64,
    pub layers: usize,
    pub attn_layers: usize,
    pub gdn_layers: usize,
    pub attn_interval: usize,
    pub vocab_size: u64,
    pub max_position_embeddings: u64,
    pub eos_ids: [usize; 2],
    /// the PLE shard end marker the text eos must equal (`None` = no PLE)
    pub ple_eos: Option<i64>,
    pub rope_type: &'static str,
    /// the context the boot allocates at least, per family (C5: Flash-Next
    /// `geo::CONTEXT_FLOOR`, dense `geo::DENSE_CONTEXT_FLOOR`)
    pub context_floor: usize,
}

impl Expected {
    /// the Flash-Next row: today's pins, read out of `geo` and `sample`
    pub const FLASH_NEXT: Expected = Expected {
        family: Family::FlashNext,
        rms_norm_eps: 1e-6,
        rope_theta: 1e7,
        rope_pairs: geo::ROPE_PAIRS,
        attention_scale: 0.0625,
        gqa_ratio: 12,
        hidden_size: geo::H as u64,
        head_dim: geo::AHD as u64,
        q_heads: geo::NQ as u64,
        kv_heads: geo::NKV as u64,
        layers: geo::LAYERS,
        attn_layers: geo::ATTN_LAYERS,
        gdn_layers: geo::GDN_LAYERS,
        attn_interval: 4,
        vocab_size: geo::V as u64,
        max_position_embeddings: 262_144,
        eos_ids: sample::EOS_IDS,
        ple_eos: Some(geo::PLE_EOS),
        rope_type: "default",
        context_floor: geo::CONTEXT_FLOOR,
    };

    /// the dense row: Qwen/Qwen3.8-27B @ 1d4bf0f2 (verified 2026-09-25,
    /// crow-nest#116): 64 layers = 16 x (3 GDN + 1 attention), H 5120,
    /// 24 q / 4 kv heads x 256, no PLE
    pub const QWEN35_DENSE: Expected = Expected {
        family: Family::Qwen35Dense,
        rms_norm_eps: 1e-6,
        rope_theta: 1e7,
        rope_pairs: 32,
        attention_scale: 0.0625,
        gqa_ratio: 6,
        hidden_size: 5120,
        head_dim: 256,
        q_heads: 24,
        kv_heads: 4,
        layers: 64,
        attn_layers: 16,
        gdn_layers: 48,
        attn_interval: 4,
        vocab_size: 248_320,
        max_position_embeddings: 262_144,
        eos_ids: [248046, 248044],
        ple_eos: None,
        rope_type: "default",
        // C5: the per-family floor; the per-card planner above it is phase 2
        context_floor: geo::DENSE_CONTEXT_FLOOR,
    };

    pub fn of(family: Family) -> Expected {
        match family {
            Family::FlashNext => Expected::FLASH_NEXT,
            Family::Qwen35Dense => Expected::QWEN35_DENSE,
            // #159: `checks` and `geo` branch before they ask for a row
            Family::Glm5Next => panic!("Expected::of: glm5_next has no Qwen row; its row is Glm5Geo::GLM_5_3_FLASH (#159)"),
        }
    }

    /// where a pinned value lives: the engine site on Flash-Next, the family
    /// row otherwise
    fn at(&self, flash_site: &str) -> String {
        match self.family {
            Family::FlashNext => flash_site.to_string(),
            Family::Qwen35Dense => format!("qwen3_5_text family row, {QWEN35_DENSE_SOURCE}"),
            Family::Glm5Next => format!("glm5_next_text family row, {GLM5_NEXT_SOURCE}"),
        }
    }
}

// ---- C1: the text_config key ledger ----

/// keys every family consumes (they feed a check or the `Geo`)
const COMMON_KEYS: &[&str] = &[
    "attention_bias",
    "bos_token_id",
    "eos_token_id",
    "full_attention_interval",
    "head_dim",
    "hidden_act",
    "hidden_size",
    "layer_types",
    "linear_conv_kernel_dim",
    "linear_key_head_dim",
    "linear_num_key_heads",
    "linear_num_value_heads",
    "linear_value_head_dim",
    "mamba_ssm_dtype",
    "max_position_embeddings",
    "model_type",
    "mtp_num_hidden_layers",
    "num_attention_heads",
    "num_hidden_layers",
    "num_key_value_heads",
    "output_gate_type",
    "partial_rotary_factor",
    "rms_norm_eps",
    "rope_parameters",
    "rope_scaling",
    "rope_theta",
    "rope_type",
    "tie_word_embeddings",
    "vocab_size",
];

/// keys only Flash-Next consumes: hyper-connections, MoE, the QSA indexer, PLE
const FLASH_NEXT_KEYS: &[&str] = &[
    "hc_count",
    "hc_lowrank",
    "heads_per_ngram",
    "indexer_budget",
    "indexer_compress_ratio",
    "indexer_head_dim",
    "indexer_kv_heads",
    "indexer_n_heads",
    "moe_intermediate_size",
    "ngram_size",
    "num_experts",
    "num_experts_per_tok",
    "ple_conv_kernel_size",
    "ple_embed_dim",
    "ple_layer_ids",
    "shared_expert_intermediate_size",
];

/// keys only the dense family consumes
const QWEN35_DENSE_KEYS: &[&str] = &["attn_output_gate", "intermediate_size"];

// ---- #159: the glm5_next key ledger (its own: no RoPE keys, no GDN keys, no COMMON_KEYS) ----

/// glm5_next keys read as unsigned integers
const GLM5_NEXT_U64_KEYS: &[&str] = &[
    "first_k_dense_replace",
    "hc_mult",
    "hc_sinkhorn_iters",
    "head_dim",
    "hidden_size",
    "index_head_dim",
    "index_kpool",
    "index_n_heads",
    "index_topk",
    "intermediate_size",
    "kv_lora_rank",
    "max_position_embeddings",
    "moe_intermediate_size",
    "n_group",
    "n_routed_experts",
    "n_shared_experts",
    "num_attention_heads",
    "num_experts_per_tok",
    "num_hidden_layers",
    "num_key_value_heads",
    "num_nextn_predict_layers",
    "q_lora_rank",
    "qk_head_dim",
    "qk_nope_head_dim",
    "qk_rope_head_dim",
    "topk_group",
    "v_head_dim",
    "vocab_size",
];
/// glm5_next keys read as numbers
const GLM5_NEXT_F64_KEYS: &[&str] = &["hc_eps", "rms_norm_eps", "routed_scaling_factor", "swiglu_limit"];
/// glm5_next keys read as booleans
const GLM5_NEXT_BOOL_KEYS: &[&str] = &[
    "attention_bias",
    "index_kpool_always_select_tail",
    "index_kpool_compress",
    "mhc",
    "mla_use_nope",
    "norm_topk_prob",
    "tie_word_embeddings",
];
/// glm5_next keys read as strings
const GLM5_NEXT_STR_KEYS: &[&str] = &["hidden_act", "model_type", "moe_router_dtype", "scoring_func", "topk_method"];
/// glm5_next keys read as per-layer string lists (one entry per trunk layer)
const GLM5_NEXT_LIST_KEYS: &[&str] = &["indexer_types", "layer_types", "mlp_layer_types"];
/// the other glm5_next keys: the stop ids and the KDA object
const GLM5_NEXT_OTHER_KEYS: &[&str] = &["eos_token_id", "linear_attn_config"];
/// the keys of `linear_attn_config` (the KDA layers) the reader consumes
const LINEAR_ATTN_CONFIG_KEYS: &[&str] = &["full_attn_layers", "gate_lower_bound", "head_dim", "kda_layers", "num_heads", "short_conv_kernel_size"];

/// every glm5_next key the reader consumes
fn glm5_next_consumes(k: &str) -> bool {
    [GLM5_NEXT_U64_KEYS, GLM5_NEXT_F64_KEYS, GLM5_NEXT_BOOL_KEYS, GLM5_NEXT_STR_KEYS, GLM5_NEXT_LIST_KEYS, GLM5_NEXT_OTHER_KEYS]
        .iter()
        .any(|l| l.contains(&k))
}

/// keys READ AND DROPPED on purpose, per family (`None` = every family), each
/// with the reason it cannot change a logit of this engine
pub const IGNORED_KEYS: &[(Option<Family>, &str, &str)] = &[
    (None, "attention_dropout", "training-only dropout"),
    (None, "dtype", "checkpoint storage dtype; the container carries its own per-tensor dtypes"),
    (None, "initializer_range", "training-only initialisation"),
    (None, "mtp_use_dedicated_embeddings", "MTP is not executed (crow-nest#95)"),
    (None, "pad_token_id", "the engine never pads a batch"),
    (None, "use_cache", "HF runtime switch"),
    (Some(Family::FlashNext), "make_ngram_vocab_size_divisible_by", "PLE table sizing; the container's shard shapes are the truth"),
    (Some(Family::FlashNext), "mtp", "MTP layer descriptor; MTP is not executed (crow-nest#95)"),
    (Some(Family::FlashNext), "ngram_vocab_size_base", "PLE table sizing; the container's shard shapes are the truth"),
    (Some(Family::FlashNext), "output_router_logits", "training-only router loss"),
    (Some(Family::FlashNext), "router_aux_loss_coef", "training-only router loss"),
    (Some(Family::FlashNext), "split_ngram_parts", "PLE table sizing; the container's shard shapes are the truth"),
    (Some(Family::Glm5Next), "index_share_for_mtp_iteration", "MTP is not executed (GLM plan step 21; docs/glm5-next-recipe.md O1)"),
    (Some(Family::Glm5Next), "indexer_rope_interleave", "no RoPE dims exist (qk_rope_head_dim 0): the indexer applies no rotation (docs/glm5-next-recipe.md section 14)"),
    (Some(Family::Glm5Next), "output_router_logits", "training-only router loss"),
    (Some(Family::Glm5Next), "router_aux_loss_coef", "training-only router loss"),
];

/// the multimodal wrapper keys a FLAT config (no `text_config`) carries beside
/// the text keys; the vision tower is not part of the text ledger
const WRAPPER_KEYS: &[&str] = &[
    "architectures",
    "image_token_id",
    "language_model_only",
    "transformers_version",
    "video_token_id",
    "vision_config",
    "vision_end_token_id",
    "vision_start_token_id",
];

/// the keys of `rope_parameters` the reader consumes
const ROPE_PARAMETER_KEYS: &[&str] = &["mrope_interleaved", "mrope_section", "partial_rotary_factor", "rope_theta", "rope_type"];

/// every key of `tc` (and of its `rope_parameters`) that is neither consumed
/// nor ignored for `family`, sorted — the unknown-key refusal's list
pub fn unknown_keys(tc: &Value, family: Family, flat: bool) -> Vec<String> {
    let family_keys = match family {
        Family::FlashNext => FLASH_NEXT_KEYS,
        Family::Qwen35Dense => QWEN35_DENSE_KEYS,
        Family::Glm5Next => &[],
    };
    // #159: glm5_next has its own ledger; the Qwen COMMON_KEYS are unknown to it
    let glm = family == Family::Glm5Next;
    let known = |k: &str| {
        (!glm && COMMON_KEYS.contains(&k))
            || (glm && glm5_next_consumes(k))
            || family_keys.contains(&k)
            || IGNORED_KEYS.iter().any(|(f, key, _)| *key == k && f.is_none_or(|f| f == family))
            || (flat && WRAPPER_KEYS.contains(&k))
    };
    let mut out: Vec<String> = tc
        .as_object()
        .into_iter()
        .flat_map(|o| o.keys())
        .filter(|k| !known(k))
        .map(|k| format!("text_config.{k}"))
        .collect();
    if let Some(rp) = tc.get("rope_parameters").and_then(Value::as_object) {
        out.extend(
            rp.keys()
                .filter(|k| !ROPE_PARAMETER_KEYS.contains(&k.as_str()))
                .map(|k| format!("text_config.rope_parameters.{k}")),
        );
    }
    if let Some(lac) = tc.get("linear_attn_config").and_then(Value::as_object).filter(|_| glm) {
        out.extend(
            lac.keys()
                .filter(|k| !LINEAR_ATTN_CONFIG_KEYS.contains(&k.as_str()))
                .map(|k| format!("text_config.linear_attn_config.{k}")),
        );
    }
    out.sort();
    out
}

/// the keys only one family consumes, parsed; which variant is present IS the
/// family
#[derive(Debug, Clone, PartialEq)]
pub enum FamilyKeys {
    FlashNext {
        hc_count: u64,
        hc_lowrank: u64,
        num_experts: u64,
        num_experts_per_tok: u64,
        moe_intermediate_size: u64,
        shared_expert_intermediate_size: u64,
        indexer_n_heads: u64,
        indexer_kv_heads: u64,
        indexer_head_dim: u64,
        indexer_compress_ratio: u64,
        indexer_budget: u64,
        /// 1-based, as the config writes them
        ple_layer_ids: Vec<u64>,
        ngram_size: u64,
        heads_per_ngram: u64,
        ple_embed_dim: u64,
        ple_conv_kernel_size: u64,
    },
    Qwen35Dense {
        intermediate_size: u64,
        attn_output_gate: bool,
    },
    /// #159: GLM-5.3-Flash; the layer lists are 0-based positions in the trunk
    Glm5Next {
        first_k_dense_replace: u64,
        hc_mult: u64,
        hc_sinkhorn_iters: u64,
        hc_eps: f64,
        q_lora_rank: u64,
        kv_lora_rank: u64,
        qk_nope_head_dim: u64,
        v_head_dim: u64,
        index_n_heads: u64,
        index_head_dim: u64,
        index_kpool: u64,
        index_topk: u64,
        linear_num_heads: u64,
        linear_head_dim: u64,
        linear_conv_kernel: u64,
        linear_lower_bound: f64,
        intermediate_size: u64,
        n_routed_experts: u64,
        num_experts_per_tok: u64,
        n_shared_experts: u64,
        moe_intermediate_size: u64,
        routed_scaling_factor: f64,
        swiglu_limit: f64,
        dsa_layers: Vec<u64>,
        kda_layers: Vec<u64>,
    },
}

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
    /// `rope_parameters.rope_type` (or flat `rope_type`): "default" here —
    /// checked, because any other value is a checkpoint announcing scaling
    /// through a channel nothing reads (scaling lives in `rope_scaling`)
    pub rope_type: String,
    pub rope_type_source: String,
    /// the `rope_scaling` object of #96 — `None` on the checkpoint of record,
    /// which means no scaling anywhere: the boot table, the kernel scale and
    /// every logit stay byte-identical
    pub rope_scaling: Option<RopeScaling>,
    // ---- Crow #300 C1: the family and the keys the Geo is derived from ----
    pub family: Family,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub mamba_ssm_dtype: String,
    pub output_gate_type: String,
    pub mrope_interleaved: Option<bool>,
    pub linear_num_key_heads: u64,
    pub linear_num_value_heads: u64,
    pub linear_key_head_dim: u64,
    pub linear_value_head_dim: u64,
    pub linear_conv_kernel_dim: u64,
    pub tie_word_embeddings: bool,
    pub mtp_num_hidden_layers: u64,
    /// `vision_config.out_hidden_size` (the merger's output width), when the
    /// config carries a vision tower
    pub vision_out_hidden: Option<u64>,
    pub family_keys: FamilyKeys,
}

// ---- #96: rope scaling ----

/// which `rope_scaling` flavor the checkpoint asks for. Everything but `Yarn`
/// and the base rewrites is implemented from the llama.cpp reference; an
/// unknown type string refuses the parse by name rather than being ignored.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RopeKind {
    /// "default" / "none": the object configures nothing
    Default,
    /// naive position interpolation (freq_scale on every pair, no mscale)
    Linear,
    /// the YaRN ramp: low/high-frequency mixing + the mscale temperature
    Yarn,
    /// NTK-aware theta rewrite: base·factor^(dim/(dim-2)), nothing else
    NtkAware,
}

impl RopeKind {
    fn parse(s: &str) -> Result<RopeKind, String> {
        match s {
            "default" | "none" => Ok(RopeKind::Default),
            "linear" => Ok(RopeKind::Linear),
            "yarn" => Ok(RopeKind::Yarn),
            "ntk-aware" | "ntk_aware" => Ok(RopeKind::NtkAware),
            other => Err(format!(
                "rope_scaling type '{other}' is not implemented (supported: default, linear, yarn, ntk-aware) \
- refusing rather than silently ignoring the checkpoint's scaling (issue #96)"
            )),
        }
    }
}

/// the parsed `rope_scaling` object (#96), with the derived numbers the boot
/// table builder and the attention-scale arm need. Every formula is the
/// llama.cpp reference (`ggml_rope_yarn_corr_dims`, `rope_yarn`, `get_mscale`,
/// the NTK base rewrite of the converter), which is the YaRN paper's reference
/// implementation. `Copy`, so the boot stash below hands it out freely.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RopeScaling {
    pub kind: RopeKind,
    /// the stretch factor (>= 1 in every sane config; llama.cpp `factor`)
    pub factor: f64,
    /// the context the model was TRAINED at: the object's own
    /// `original_max_position_embeddings`, falling back to the config's
    /// `max_position_embeddings` (llama.cpp's `n_ctx_orig_yarn` fallback)
    pub original_context: u64,
    /// YaRN ramp bounds, llama.cpp defaults 32 / 1 when the object omits them
    pub beta_fast: f32,
    pub beta_slow: f32,
    /// optional HF `attention_factor`, a direct multiplier on the mscale
    /// (llama.cpp `rope.attention.factor`)
    pub attention_factor: Option<f64>,
}

impl RopeScaling {
    /// parse the `rope_scaling` object. `fallback_ctx` is the config's
    /// `max_position_embeddings`. A missing `rope_type`/`type` or `factor`, or
    /// a non-positive factor, is a named error — the #94 rule: never a guess.
    fn from_config(v: &Value, fallback_ctx: u64) -> Result<RopeScaling, String> {
        let type_str = v
            .get("rope_type")
            .and_then(Value::as_str)
            .or_else(|| v.get("type").and_then(Value::as_str))
            .ok_or("missing required key(s): rope_scaling.rope_type (or rope_scaling.type)")?;
        let kind = RopeKind::parse(type_str)?;
        let factor = f64_of(v, "factor").ok_or("missing required key(s): rope_scaling.factor")?;
        if !(factor.is_finite() && factor > 0.0) {
            return Err(format!("rope_scaling.factor {factor} is not a positive number"));
        }
        Ok(RopeScaling {
            kind,
            factor,
            original_context: u64_of(v, "original_max_position_embeddings").unwrap_or(fallback_ctx),
            beta_fast: f64_of(v, "beta_fast").unwrap_or(32.0) as f32,
            beta_slow: f64_of(v, "beta_slow").unwrap_or(1.0) as f32,
            attention_factor: f64_of(v, "attention_factor"),
        })
    }

    /// llama.cpp `rope_freq_scale` = 1/factor: what an interpolated angle is
    /// multiplied by
    pub fn freq_scale(&self) -> f32 {
        (1.0 / self.factor) as f32
    }

    /// the YaRN correction range, ggml.c `ggml_rope_yarn_corr_dims` verbatim:
    /// pairs below `lo` extrapolate (no scaling — the high-frequency bands
    /// YaRN exists to protect), pairs above `hi` interpolate, the span between
    /// ramps. The bounds are in DIM units against a PAIR index, exactly as
    /// llama.cpp and HF ship it (both compare `i/2` against dim-unit bounds).
    pub fn yarn_corr_range(&self, dim: usize, base: f64) -> (f32, f32) {
        let corr = |beta: f32| {
            dim as f32 * (self.original_context as f32 / (beta * 2.0 * std::f32::consts::PI)).ln()
                / (2.0 * (base as f32).ln())
        };
        (corr(self.beta_fast).floor().max(0.0), corr(self.beta_slow).ceil().min(dim as f32 - 1.0))
    }

    /// the YaRN mscale (llama.cpp `get_mscale(factor, 1.0)`, the YaRN paper's
    /// attention temperature): `1 + 0.1·ln(1/freq_scale)` = `1 + 0.1·ln(factor)`.
    /// 1.0 unless yarn with factor > 1 — linear and ntk change no temperature.
    /// `attention_factor`, when the object carries it, multiplies the result.
    pub fn mscale(&self) -> f64 {
        let m = match self.kind {
            RopeKind::Yarn if self.factor > 1.0 => 1.0 + 0.1 * self.factor.ln(),
            _ => 1.0,
        };
        m * self.attention_factor.unwrap_or(1.0)
    }

    /// the NTK-aware theta rewrite the llama.cpp converter bakes into the rope
    /// freq base: `base · factor^(dim/(dim-2))` — slows the low-frequency walk
    /// without interpolating any pair
    pub fn ntk_base(&self, dim: usize, base: f64) -> f64 {
        base * self.factor.powf(dim as f64 / (dim as f64 - 2.0))
    }
}

// ---- the #96 boot stash ----

/// The rope truth the boot door parsed, for the two boot-time readers that
/// cannot be handed it as a parameter: `ThreeStates::allocate` (manager.rs)
/// builds the table and `Kernels::new` (kernels.rs) threads the mscale — both
/// are called from `gen.rs` with signatures this issue does not own. `assert_pinned`
/// sets this BEFORE the container is mapped, so both readers see it; a process
/// that never passed the front door (unit tests, the probe bins) reads the
/// `None` default, which is today's behavior everywhere.
static BOOT_ROPE_SCALING: std::sync::OnceLock<Option<RopeScaling>> = std::sync::OnceLock::new();
static BOOT_TRAINING_CONTEXT: std::sync::OnceLock<u64> = std::sync::OnceLock::new();

/// the boot-parsed `rope_scaling` (`None` = no scaling: byte-identical behavior)
pub fn boot_rope_scaling() -> Option<RopeScaling> {
    BOOT_ROPE_SCALING.get().copied().flatten()
}

/// the context the checkpoint was TRAINED at — `rope_scaling.original_context`
/// when the object carries one, else `max_position_embeddings` (262,144 here)
pub fn boot_training_context() -> Option<u64> {
    BOOT_TRAINING_CONTEXT.get().copied()
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
        let generation = match generation_config_path {
            Some(p) => Some((std::fs::read_to_string(p).map_err(|e| format!("{p}: {e}"))?, p)),
            None => None,
        };
        ModelMeta::from_config_texts(&config_text, config_path, generation.as_ref().map(|(t, p)| (t.as_str(), *p)))
    }

    /// `from_config_files` on the two files' TEXTS (Crow #300 C7: an index v2 container carries
    /// them verbatim in its `model` block). `config_path` and the generation label are what the
    /// log lines and the refusals name as the source: a file path, or the container's
    /// `<cnq> [index v2 model.config_json]`.
    pub fn from_config_texts(
        config_text: &str,
        config_path: &str,
        generation_config: Option<(&str, &str)>,
    ) -> Result<ModelMeta, String> {
        let generation_config_path = generation_config.map(|(_, p)| p);
        let config: Value =
            serde_json::from_str(config_text).map_err(|e| format!("{config_path}: not valid json: {e}"))?;
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
        // #96: the rope_type token, same nested-then-flat fallback as rope_theta
        let (rope_type, rope_type_source) = match rp.and_then(|r| r.get("rope_type")).and_then(Value::as_str) {
            Some(t) => (t.to_string(), "text_config.rope_parameters.rope_type".to_string()),
            None => match tc.get("rope_type").and_then(Value::as_str) {
                Some(t) => (t.to_string(), "text_config.rope_type".to_string()),
                None => (String::new(), String::new()),
            },
        };

        // generation_config.json is the source of record for the stop ids; the
        // text config's own eos/bos is the fallback and the cross-check
        let generation = match generation_config {
            Some((s, p)) => Some(serde_json::from_str::<Value>(s).map_err(|e| format!("{p}: not valid json: {e}"))?),
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

        // #159: glm5_next has its own key ledger (no RoPE, no GDN keys), parsed apart so the
        // Qwen families' path below is unchanged
        if tc.get("model_type").and_then(Value::as_str) == Some(GLM5_NEXT_MODEL_TYPE) {
            let ids = GenIds { eos_token_ids, eos_from_generation, text_eos_token_id, bos_token_id, text_bos_token_id };
            return ModelMeta::from_glm5_next(&config, tc, config_path, generation_config_path, ids);
        }

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
        need(!rope_type_source.is_empty(), "text_config.rope_parameters.rope_type (or text_config.rope_type)");
        need(eos_token_ids.as_ref().is_some_and(|v| !v.is_empty()), "generation_config.json eos_token_id (or text_config.eos_token_id)");

        // Crow #300 C1: the family, then the keys it requires. An unknown
        // model_type refuses here, by name, before anything else is judged.
        let model_type = tc.get("model_type").and_then(Value::as_str);
        let family = match model_type {
            Some(mt) => Some(Family::detect(mt).map_err(|why| format!("{config_path}: {why}"))?),
            None => None,
        };
        need(model_type.is_some(), "text_config.model_type");
        let str_of = |key: &str| tc.get(key).and_then(Value::as_str).map(str::to_string);
        let bool_of = |key: &str| tc.get(key).and_then(Value::as_bool);
        for key in [
            "linear_num_key_heads",
            "linear_num_value_heads",
            "linear_key_head_dim",
            "linear_value_head_dim",
            "linear_conv_kernel_dim",
            "mtp_num_hidden_layers",
        ] {
            need(u64_of(tc, key).is_some(), &format!("text_config.{key}"));
        }
        for key in ["hidden_act", "mamba_ssm_dtype", "output_gate_type"] {
            need(str_of(key).is_some(), &format!("text_config.{key}"));
        }
        for key in ["attention_bias", "tie_word_embeddings"] {
            need(bool_of(key).is_some(), &format!("text_config.{key}"));
        }
        match family {
            Some(Family::FlashNext) => {
                for key in FLASH_NEXT_KEYS.iter().filter(|k| **k != "ple_layer_ids") {
                    need(u64_of(tc, key).is_some(), &format!("text_config.{key}"));
                }
                need(tc.get("ple_layer_ids").and_then(Value::as_array).is_some(), "text_config.ple_layer_ids");
            }
            Some(Family::Qwen35Dense) => {
                need(u64_of(tc, "intermediate_size").is_some(), "text_config.intermediate_size");
                need(bool_of("attn_output_gate").is_some(), "text_config.attn_output_gate");
            }
            Some(Family::Glm5Next) => unreachable!("glm5_next is parsed by from_glm5_next (#159)"),
            None => {}
        }
        if !missing.is_empty() {
            return Err(format!("{config_path}: missing required key(s): {}", missing.join(", ")));
        }
        let family = family.unwrap();

        // Crow #300 C1: every text_config key is consumed or ignored by name
        let flat = !config.get("text_config").is_some_and(Value::is_object);
        let unknown = unknown_keys(tc, family, flat);
        if !unknown.is_empty() {
            return Err(format!(
                "{config_path}: key(s) this engine does not know for family {family:?} ({}): {} - refusing rather than \
silently ignoring them (Crow #300); each must be consumed or put on the ignore list in meta.rs with its reason",
                family.model_type(),
                unknown.join(", ")
            ));
        }
        // formula facts with exactly one implemented value, refused by name
        let unsupported = |key: &str, got: &str, have: &[&str]| -> Result<(), String> {
            if have.contains(&got) {
                Ok(())
            } else {
                Err(format!("{config_path}: text_config.{key} '{got}' is not implemented (implemented: {}) - refusing (Crow #300)", have.join(", ")))
            }
        };
        let hidden_act = str_of("hidden_act").unwrap();
        let mamba_ssm_dtype = str_of("mamba_ssm_dtype").unwrap();
        let output_gate_type = str_of("output_gate_type").unwrap();
        unsupported("hidden_act", &hidden_act, &["silu"])?;
        unsupported("mamba_ssm_dtype", &mamba_ssm_dtype, &["float32"])?;
        unsupported("output_gate_type", &output_gate_type, &["sigmoid", "swish", "silu"])?;
        let tie_word_embeddings = bool_of("tie_word_embeddings").unwrap();
        if let Some(top) = config.get("tie_word_embeddings").and_then(Value::as_bool) {
            if top != tie_word_embeddings {
                return Err(format!(
                    "{config_path}: tie_word_embeddings {top} (top level) disagrees with text_config.tie_word_embeddings {tie_word_embeddings} - refusing (Crow #300)"
                ));
            }
        }
        let u = |key: &str| u64_of(tc, key).unwrap();
        let family_keys = match family {
            Family::FlashNext => FamilyKeys::FlashNext {
                hc_count: u("hc_count"),
                hc_lowrank: u("hc_lowrank"),
                num_experts: u("num_experts"),
                num_experts_per_tok: u("num_experts_per_tok"),
                moe_intermediate_size: u("moe_intermediate_size"),
                shared_expert_intermediate_size: u("shared_expert_intermediate_size"),
                indexer_n_heads: u("indexer_n_heads"),
                indexer_kv_heads: u("indexer_kv_heads"),
                indexer_head_dim: u("indexer_head_dim"),
                indexer_compress_ratio: u("indexer_compress_ratio"),
                indexer_budget: u("indexer_budget"),
                ple_layer_ids: tc["ple_layer_ids"].as_array().unwrap().iter().filter_map(Value::as_u64).collect(),
                ngram_size: u("ngram_size"),
                heads_per_ngram: u("heads_per_ngram"),
                ple_embed_dim: u("ple_embed_dim"),
                ple_conv_kernel_size: u("ple_conv_kernel_size"),
            },
            Family::Qwen35Dense => FamilyKeys::Qwen35Dense {
                intermediate_size: u("intermediate_size"),
                attn_output_gate: bool_of("attn_output_gate").unwrap(),
            },
            Family::Glm5Next => unreachable!("glm5_next is parsed by from_glm5_next (#159)"),
        };

        // #96: rope_scaling, after the required-key pass so the fallback context
        // is known. Absent (the checkpoint of record) = None = no scaling; a
        // present object must parse or the whole config refuses by name.
        let rope_scaling = match tc.get("rope_scaling") {
            Some(rs) if rs.is_object() => {
                Some(RopeScaling::from_config(rs, u64_of(tc, "max_position_embeddings").unwrap())
                    .map_err(|why| format!("{config_path}: {why}"))?)
            }
            _ => None,
        };

        Ok(ModelMeta {
            config_path: config_path.to_string(),
            generation_config_path: generation_config_path.map(str::to_string),
            model_type: family.model_type().to_string(),
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
            rope_type,
            rope_type_source,
            rope_scaling,
            family,
            hidden_act,
            attention_bias: bool_of("attention_bias").unwrap(),
            mamba_ssm_dtype,
            output_gate_type,
            mrope_interleaved: rp.and_then(|r| r.get("mrope_interleaved")).and_then(Value::as_bool),
            linear_num_key_heads: u("linear_num_key_heads"),
            linear_num_value_heads: u("linear_num_value_heads"),
            linear_key_head_dim: u("linear_key_head_dim"),
            linear_value_head_dim: u("linear_value_head_dim"),
            linear_conv_kernel_dim: u("linear_conv_kernel_dim"),
            tie_word_embeddings,
            mtp_num_hidden_layers: u("mtp_num_hidden_layers"),
            vision_out_hidden: config.get("vision_config").and_then(|v| u64_of(v, "out_hidden_size")),
            family_keys,
        })
    }

    // ---- C2: the runtime geometry ----

    /// Derive the runtime [`Geo`] from the parsed config and the family table.
    /// `Err` names the value that has no `Geo` form (a non-integral rope pair
    /// count, an eos list that is not two ids, a malformed mrope section).
    pub fn geo(&self) -> Result<Geo, String> {
        // #159: glm5_next has no runtime `Geo` (its layers run in `glm5_model`, #161); its geometry is `glm5_geo`
        if self.family == Family::Glm5Next {
            return Err(glm5_not_built(self.checks().len(), &self.config_path));
        }
        let e = Expected::of(self.family);
        let pairs = self.partial_rotary_factor * self.head_dim as f64 / 2.0;
        if pairs.fract() != 0.0 || pairs <= 0.0 {
            return Err(format!("rope pairs {pairs} (partial_rotary_factor x head_dim / 2) is not a whole number"));
        }
        let eos_ids: [usize; 2] = match self.eos_token_ids.as_slice() {
            [a, b] if *a >= 0 && *b >= 0 => [*a as usize, *b as usize],
            other => return Err(format!("eos_token_id {other:?}: the Geo holds exactly two stop ids (sample::EOS_IDS)")),
        };
        let mrope_section: [usize; 3] = match self.mrope_section.as_deref() {
            Some([a, b, c]) => [*a as usize, *b as usize, *c as usize],
            other => return Err(format!("text_config.rope_parameters.mrope_section {other:?}: the Geo needs three sections")),
        };
        let attn_layers = self.layer_types.iter().filter(|t| t.as_str() == "full_attention").count();
        let gdn_layers = self.layer_types.iter().filter(|t| t.as_str() == "linear_attention").count();
        // the interval the config names, else the first full-attention slot + 1
        let attn_interval = match self.full_attention_interval {
            Some(i) => i as usize,
            None => self.layer_types.iter().position(|t| t == "full_attention").map_or(0, |p| p + 1),
        };
        let gate_act = match self.output_gate_type.as_str() {
            "sigmoid" => GateAct::Sigmoid,
            _ => GateAct::Silu, // "swish" / "silu", the parse refused anything else
        };
        let z = |v: u64| v as usize;
        let (residual, ffn, attn, ple, final_norm, attn_output_gate) = match &self.family_keys {
            FamilyKeys::FlashNext {
                hc_count,
                hc_lowrank,
                num_experts,
                num_experts_per_tok,
                moe_intermediate_size,
                shared_expert_intermediate_size,
                indexer_n_heads,
                indexer_kv_heads,
                indexer_head_dim,
                indexer_compress_ratio,
                indexer_budget,
                ple_layer_ids,
                ngram_size,
                heads_per_ngram,
                ple_embed_dim,
                ple_conv_kernel_size,
            } => {
                let ple_layer = match ple_layer_ids.as_slice() {
                    [one] if *one >= 1 => z(*one) - 1,
                    other => return Err(format!("text_config.ple_layer_ids {other:?}: the Geo holds exactly one 1-based PLE layer")),
                };
                if *indexer_compress_ratio == 0 || indexer_budget % indexer_compress_ratio != 0 {
                    return Err(format!(
                        "text_config.indexer_budget {indexer_budget} is not a multiple of indexer_compress_ratio {indexer_compress_ratio}"
                    ));
                }
                (
                    Residual::Hc { streams: z(*hc_count), lowrank: z(*hc_lowrank) },
                    Ffn::Moe {
                        experts: z(*num_experts),
                        topk: z(*num_experts_per_tok),
                        expert_inter: z(*moe_intermediate_size),
                        shared_inter: z(*shared_expert_intermediate_size),
                    },
                    Attn::Qsa {
                        heads: z(*indexer_n_heads),
                        kv_heads: z(*indexer_kv_heads),
                        head_dim: z(*indexer_head_dim),
                        compress: z(*indexer_compress_ratio),
                        block_topk: z(indexer_budget / indexer_compress_ratio),
                    },
                    Some(PleGeo {
                        layer: ple_layer,
                        ngram: z(*ngram_size),
                        heads_per_ngram: z(*heads_per_ngram),
                        embed: z(*ple_embed_dim),
                        conv_kernel: z(*ple_conv_kernel_size),
                        eos: self.text_eos_token_id.unwrap_or(-1),
                    }),
                    FinalNorm::HcMixer,
                    // Flash-Next names only the gate's activation; the gate is always there
                    true,
                )
            }
            FamilyKeys::Qwen35Dense { intermediate_size, attn_output_gate } => (
                Residual::Plain,
                Ffn::Dense { inter: z(*intermediate_size) },
                Attn::Full,
                None,
                FinalNorm::Rms,
                *attn_output_gate,
            ),
            FamilyKeys::Glm5Next { .. } => unreachable!("glm5_next returned above"),
        };
        Ok(Geo {
            family: self.family,
            hidden: z(self.hidden_size),
            residual,
            layers: z(self.num_hidden_layers),
            gdn_layers,
            attn_layers,
            attn_interval,
            q_heads: z(self.num_attention_heads),
            kv_heads: z(self.num_key_value_heads),
            head_dim: z(self.head_dim),
            attn,
            attn_output_gate,
            gate_act,
            attention_bias: self.attention_bias,
            rope_pairs: pairs as usize,
            rope_theta: self.rope_theta,
            mrope_section,
            mrope_interleaved: self.mrope_interleaved.unwrap_or(false),
            gdn_key_heads: z(self.linear_num_key_heads),
            gdn_value_heads: z(self.linear_num_value_heads),
            gdn_key_dim: z(self.linear_key_head_dim),
            gdn_value_dim: z(self.linear_value_head_dim),
            conv_kernel: z(self.linear_conv_kernel_dim),
            ffn,
            ple,
            final_norm,
            // both families' RMSNorm is the zero-centred (1 + w) form
            norm_one_plus_w: true,
            rms_eps: self.rms_norm_eps,
            vocab: z(self.vocab_size),
            tie_word_embeddings: self.tie_word_embeddings,
            context_max: z(self.max_position_embeddings),
            context_floor: e.context_floor,
            eos_ids,
            mtp_layers: z(self.mtp_num_hidden_layers),
            vision_out_hidden: self.vision_out_hidden.map(z),
        })
    }

    // ---- the comparisons ----

    /// every check, green and red, in table order, against the family's
    /// [`Expected`] row. The count is what the boot INFO line reports as "N
    /// constants verified" (21 on Flash-Next, whose row is today's pins; the
    /// dense row has no PLE, so no `ple_eos`).
    pub fn checks(&self) -> Vec<Check> {
        if self.family == Family::Glm5Next {
            return self.glm5_checks();
        }
        let e = Expected::of(self.family);
        let mut c = Vec::with_capacity(20);
        c.push(Check::cmp("rms_norm_eps", e.rms_norm_eps, self.rms_norm_eps, &e.at("kernels.rs, every rms + LayerNorm site"), "text_config.rms_norm_eps"));
        c.push(Check::cmp("rope_theta", e.rope_theta, self.rope_theta, &e.at("manager.rs boot RoPE table"), &self.rope_theta_source));
        // partial rotary: factor x head_dim rotary dims, ROPE_PAIRS of them
        let pairs = self.partial_rotary_factor * self.head_dim as f64 / 2.0;
        c.push(Check {
            name: "rope_pairs",
            pinned: format!("{:?} ({})", e.rope_pairs, e.at("geo::ROPE_PAIRS")),
            config: format!("{pairs}"),
            source: format!("{} x head_dim {} / 2", self.partial_rotary_factor, self.head_dim),
            ok: pairs == e.rope_pairs as f64,
        });
        // the attention scale every variant hardcodes is 1/sqrt(head_dim)
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        c.push(Check::cmp("attention_scale", e.attention_scale, scale, &e.at("kernels.rs, 5 attention variants"), "1/sqrt(text_config.head_dim)"));
        // GQA: the kernels derive kvh from the head count with a pinned divisor
        let gqa = if self.num_key_value_heads > 0 && self.num_attention_heads % self.num_key_value_heads == 0 {
            self.num_attention_heads / self.num_key_value_heads
        } else {
            0
        };
        c.push(Check::cmp("gqa_ratio", e.gqa_ratio, gqa, &e.at("kernels.rs `head / 12`"), "num_attention_heads / num_key_value_heads"));
        c.push(Check::cmp("hidden_size", e.hidden_size, self.hidden_size, &e.at("geo::H"), "text_config.hidden_size"));
        c.push(Check::cmp("head_dim", e.head_dim, self.head_dim, &e.at("geo::AHD"), "text_config.head_dim"));
        c.push(Check::cmp("q_heads", e.q_heads, self.num_attention_heads, &e.at("geo::NQ"), "text_config.num_attention_heads"));
        c.push(Check::cmp("kv_heads", e.kv_heads, self.num_key_value_heads, &e.at("geo::NKV"), "text_config.num_key_value_heads"));
        c.push(Check::cmp("num_hidden_layers", e.layers as u64, self.num_hidden_layers, &e.at("geo::LAYERS"), "text_config.num_hidden_layers"));
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
        let pinned_at: Vec<usize> = (0..e.layers).filter(|i| i % e.attn_interval == e.attn_interval - 1).collect();
        let layout_ok = self.layer_types.len() == e.layers
            && full_at == pinned_at
            && full_at.len() == e.attn_layers
            && self.layer_types.len() - full_at.len() == e.gdn_layers
            && unknown.is_empty();
        c.push(Check {
            name: "layer_layout",
            pinned: format!(
                "layer % {} == {} is full attention: {} attn + {} gdn of {} ({})",
                e.attn_interval,
                e.attn_interval - 1,
                e.attn_layers,
                e.gdn_layers,
                e.layers,
                e.at("geo::is_attn, ATTN_LAYERS, GDN_LAYERS")
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
            c.push(Check::cmp("full_attention_interval", e.attn_interval as u64, interval, &e.at("geo::is_attn `layer % 4`"), "text_config.full_attention_interval"));
        }
        c.push(Check::cmp("vocab_size", e.vocab_size, self.vocab_size, &e.at("geo::V"), "text_config.vocab_size"));
        // Config::default().context is the checkpoint's context budget; the
        // 200_000 floor is an engine policy, not a model constant
        c.push(Check::cmp(
            "max_position_embeddings",
            e.max_position_embeddings,
            self.max_position_embeddings,
            &e.at("geo::Config::default().context"),
            "text_config.max_position_embeddings",
        ));
        // the sampler's stop ids — sample.rs keeps the pin, this reads it live
        let config_eos: Vec<usize> = self.eos_token_ids.iter().map(|id| *id as usize).collect();
        c.push(Check {
            name: "eos_ids",
            pinned: format!("{:?} ({})", e.eos_ids, e.at("sample::EOS_IDS")),
            config: format!("{:?}", self.eos_token_ids),
            source: if self.eos_from_generation {
                "generation_config.json eos_token_id".to_string()
            } else {
                "text_config.eos_token_id".to_string()
            },
            ok: config_eos == e.eos_ids.to_vec(),
        });
        // 248044 doubles as the PLE shard end marker (geo::PLE_EOS): the text
        // config's own eos must BE that id, or the PLE reader and the sampler
        // disagree about what "end" means (a family without PLE has no marker)
        if let Some(ple_eos) = e.ple_eos {
            c.push(Check {
                name: "ple_eos",
                pinned: format!("{:?} (geo::PLE_EOS, the PLE shard end marker / gen.rs filler)", ple_eos),
                config: format!("{:?}", self.text_eos_token_id),
                source: "text_config.eos_token_id".to_string(),
                ok: self.text_eos_token_id == Some(ple_eos),
            });
        }
        // llama.cpp special-id discipline: every stop/start id inside the vocab
        let vocab = self.vocab_size as i64;
        let eos_in = !self.eos_token_ids.is_empty() && self.eos_token_ids.iter().all(|id| *id >= 0 && *id < vocab);
        c.push(Check {
            name: "eos_ids_in_vocab",
            pinned: format!("0 <= id < {}", e.vocab_size),
            config: format!("{:?} against vocab {}", self.eos_token_ids, self.vocab_size),
            source: "generation_config.json eos_token_id vs text_config.vocab_size".to_string(),
            ok: eos_in,
        });
        if let Some(bos) = self.bos_token_id {
            c.push(Check {
                name: "bos_id_in_vocab",
                pinned: format!("0 <= id < {}", e.vocab_size),
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
            c.push(Check::cmp("mrope_section_pairs", e.rope_pairs as u64, sum, &e.at("geo::ROPE_PAIRS (manager.rs table)"), "text_config.rope_parameters.mrope_section sum"));
        }
        // #96: the rope_type token must say "default" — the scaling this engine
        // reads lives in the rope_scaling object, parsed separately; a different
        // value here is a checkpoint announcing scaling through a channel
        // nothing reads (and any scaling it meant must come as rope_scaling)
        c.push(Check::cmp(
            "rope_type",
            e.rope_type,
            self.rope_type.as_str(),
            &e.at("manager.rs boot RoPE table (no scaling path taken)"),
            &self.rope_type_source,
        ));
        c
    }

    /// the context the checkpoint was TRAINED at: the rope_scaling object's own
    /// `original_max_position_embeddings` when it carries one, else the config's
    /// `max_position_embeddings` (#96: the warn's threshold)
    pub fn training_context(&self) -> u64 {
        self.rope_scaling.map(|s| s.original_context).unwrap_or(self.max_position_embeddings)
    }

    /// the named mismatches — [`ModelMeta::checks`] with the green rows
    /// removed. Empty means the plumbing is byte-identical to the pins.
    pub fn verify(&self) -> Vec<Check> {
        self.checks().into_iter().filter(|c| !c.ok).collect()
    }
}

// ---- locating the checkpoint's config next to the container ----

/// The index v1 path (the CNQ4.5-M container of record, which carries no
/// config): `CROW_MODEL_DIR` (`model_dir`, read by the caller) wins; otherwise
/// `models/` beside the container (the container lives in `<repo>/converter/`,
/// so the repo root is its parent directory's parent — the same convention both
/// cwd forms of the bins give: `converter/x.cnq` from the repo root,
/// `../converter/x.cnq` from `engine/`). `Ok(None)` = no config found (the
/// selftest package): the caller warns.
pub fn from_container(cnq_path: &str, model_dir: Option<&str>) -> Result<Option<ModelMeta>, String> {
    let config_dir = if let Some(dir) = model_dir {
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
/// `None` when the tree holds no candidate. Only a checkpoint whose directory
/// name carries the container's model name (the file stem up to `-CNQ…`) is a
/// candidate, also when it is the only one under `models/`.
pub fn config_dir_for_container(cnq_path: &str) -> Option<std::path::PathBuf> {
    config_dir_beside(cnq_path, std::path::Path::new("."))
}

/// [`config_dir_for_container`] with the cwd fallback root passed in (the unit
/// test builds the install layout in a temp dir instead of changing the cwd)
fn config_dir_beside(cnq_path: &str, cwd: &std::path::Path) -> Option<std::path::PathBuf> {
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
    for root in p.parent().and_then(|d| d.parent()).into_iter().chain([cwd]) {
        if let Some(dir) = config_dir_in_root(root, &hint) {
            return Some(dir);
        }
    }
    None
}

/// only a checkpoint whose directory name starts with the container's model
/// name, a lone candidate included: Crow's install keeps every model under
/// `models/`, so a lone `models/whisper-small/` is another model, not this
/// container's truth source. No match (or no name to match) -> `None`.
fn config_dir_in_root(root: &std::path::Path, hint: &str) -> Option<std::path::PathBuf> {
    if hint.is_empty() {
        return None;
    }
    let read = std::fs::read_dir(root.join("models")).ok()?;
    let mut cands: Vec<std::path::PathBuf> = read
        .flatten()
        .map(|e| e.path())
        .filter(|p| p.join("config.json").is_file())
        .filter(|c| c.file_name().map(|n| n.to_string_lossy().starts_with(hint)).unwrap_or(false))
        .collect();
    cands.sort();
    cands.into_iter().next()
}

// ---- Crow #300 C7: which config the gate judges, and the container it must fit ----

/// the log / refusal label of an index v2 config: the container, and the block key
pub fn v2_config_label(cnq_path: &str, key: &str) -> String {
    format!("{cnq_path} [index v2 model.{key}]")
}

/// The config the gate judges, by the container's index kind:
///
/// - **index v2**: the `model` block's `config_json` / `generation_config_json`,
///   the bytes the converter read (their sha256 were checked by
///   `cnq::classify_index`). The container's config wins: neither `models/`
///   beside the container nor `CROW_MODEL_DIR` is read for it
///   ([`check_model_dir_override`] only compares a set `CROW_MODEL_DIR`).
/// - **index v1** (the CNQ4.5-M container of record, no config inside):
///   [`from_container`], the pre-C7 path: `CROW_MODEL_DIR`, else `models/`.
pub fn config_for_container(cnq_path: &str, peek: &IndexPeek, model_dir: Option<&str>) -> Result<Option<ModelMeta>, String> {
    match peek.model() {
        None => from_container(cnq_path, model_dir),
        Some(m) => {
            let label = v2_config_label(cnq_path, "config_json");
            let glabel = v2_config_label(cnq_path, "generation_config_json");
            ModelMeta::from_config_texts(&m.config_json, &label, Some((m.generation_config_json.as_str(), glabel.as_str()))).map(Some)
        }
    }
}

/// Crow #300 C7: `CROW_MODEL_DIR` beside an index v2 container. The container's
/// config wins, so the variable is only a cross-check: its `config.json` (and its
/// `generation_config.json`, when the directory has one) must hash to the bytes
/// the container carries, or the boot is refused by name. A caller that points
/// the variable at one checkpoint and the container at another is told so,
/// instead of one of the two being silently ignored.
pub fn check_model_dir_override(dir: &str, m: &ModelBlock) -> Result<(), String> {
    if dir.is_empty() {
        return Err("CROW_MODEL_DIR is set but empty".to_string());
    }
    let d = std::path::Path::new(dir);
    for (file, carried, key, required) in [
        ("config.json", m.config_json.as_str(), "config_json", true),
        ("generation_config.json", m.generation_config_json.as_str(), "generation_config_json", false),
    ] {
        let p = d.join(file);
        let bytes = match std::fs::read(&p) {
            Ok(b) => b,
            Err(_) if !required => continue,
            Err(e) => return Err(format!("CROW_MODEL_DIR {dir}: {file}: {e} - refusing (Crow #300 C7)")),
        };
        let (got, want) = (sha256_hex(&bytes), sha256_hex(carried.as_bytes()));
        if got != want {
            return Err(format!(
                "[meta] CROW_MODEL_DIR {} sha256 {got} differs from the container's model.{key} sha256 {want} \
(family {}, {} @ {}) - an index v2 container carries its own config; unset CROW_MODEL_DIR or point it at the \
checkpoint this container was converted from - refusing to boot (Crow #300)",
                p.display(),
                m.family,
                m.source_repo,
                m.source_revision
            ));
        }
    }
    Ok(())
}

/// The container facts a config must fit, read off the index trailer: the
/// family it was converted as, the token embedding's `[vocab, hidden]`, and the
/// number of text layers. A config of another model passes its own family row
/// (`verdict`), so this is what catches it, before `Cnq::open` and before any
/// CUDA work: `Err` is a table with one row per differing fact, `container X,
/// config Y`. Crow #300 phase 1 acceptance: "a container with the other model's
/// config dies at boot with a named mismatch table".
pub fn container_mismatch(geo: &Geo, peek: &IndexPeek, cnq_path: &str, config_label: &str) -> Result<(), String> {
    let text = || peek.tensors.iter().filter(|t| t.section == "text");
    let embed = text()
        .find(|t| t.name.ends_with("embed_tokens.weight"))
        .map(|t| format!("{:?}", t.shape))
        .unwrap_or_else(|| "no embed_tokens tensor".to_string());
    let layers = text()
        .filter_map(|t| t.name.split(".layers.").nth(1)?.split('.').next()?.parse::<usize>().ok())
        .collect::<std::collections::BTreeSet<_>>()
        .len();
    let family = format!("{:?}", geo.family);
    let rows = [
        ("family", format!("{} (index v{})", peek.family(), if peek.model().is_some() { 2 } else { 1 }), family.clone(), peek.family() == family),
        ("embed_tokens [vocab, hidden]", embed.clone(), format!("{:?}", [geo.vocab, geo.hidden]), embed == format!("{:?}", [geo.vocab, geo.hidden])),
        ("text layers", layers.to_string(), geo.layers.to_string(), layers == geo.layers),
    ];
    let bad: Vec<_> = rows.iter().filter(|r| !r.3).collect();
    if bad.is_empty() {
        return Ok(());
    }
    let table = bad.iter().map(|(n, c, g, _)| format!("  {n}: container {c}, config {g}")).collect::<Vec<_>>().join("\n");
    Err(format!(
        "[meta] {} of {} container facts differ from the config - the container and its config describe different models, \
refusing to boot (Crow #300):\n{table}\n[meta]   container: {cnq_path}\n[meta]   config: {config_label}",
        bad.len(),
        rows.len()
    ))
}

/// Why the gate refused: a config it could not read (the #94 "unreadable truth
/// source" class), or a config it read and refused (a table, a named reason).
#[derive(Debug)]
pub enum GateRefusal {
    Unreadable(String),
    Refused(String),
}

/// The metadata gate, pure (no panic, no log, no stash), so every branch is
/// unit-tested on a container's peeked index: the `CROW_MODEL_DIR` cross-check
/// of an index v2, the config by index kind ([`config_for_container`]), the
/// family row and the `Geo` ([`verdict`]), then the container facts
/// ([`container_mismatch`]). `Ok(None)` = an index v1 with no config found (the
/// selftest package).
pub fn gate(cnq_path: &str, peek: &IndexPeek, model_dir: Option<&str>) -> Result<Option<(ModelMeta, Geo)>, GateRefusal> {
    if let (Some(m), Some(dir)) = (peek.model(), model_dir) {
        check_model_dir_override(dir, m).map_err(GateRefusal::Refused)?;
    }
    let Some(meta) = config_for_container(cnq_path, peek, model_dir).map_err(GateRefusal::Unreadable)? else {
        return Ok(None);
    };
    let geo = verdict(&meta).map_err(GateRefusal::Refused)?;
    container_mismatch(&geo, peek, cnq_path, &meta.config_path).map_err(GateRefusal::Refused)?;
    Ok(Some((meta, geo)))
}

// ---- #159: the glm5_next family at the gate ----

/// the stop and start ids `from_config_texts` read before it knew the family
struct GenIds {
    eos_token_ids: Option<Vec<i64>>,
    eos_from_generation: bool,
    text_eos_token_id: Option<i64>,
    bos_token_id: Option<i64>,
    text_bos_token_id: Option<i64>,
}

/// #159: the boot refusal of a glm5_next config that passed its family row. The gate accepts
/// the family (parse, key ledger, `Glm5Geo` equal to the row) and the planner plans it. #161:
/// the layer arms exist (`glm5_model`, run layer by layer by `decode glmgolden`); what the boot
/// still lacks is named here, so it stops before the container is mapped and before any CUDA
/// call.
pub fn glm5_not_built(checks: usize, config_path: &str) -> String {
    format!(
        "[meta] family Glm5Next ({GLM5_NEXT_MODEL_TYPE}): {checks} constants verified against the family row \
({GLM5_NEXT_SOURCE}) [{config_path}]; its layers are built (mHC, KDA, MLA + DSA, router and MUL1 experts, head: \
#161-#165, layer by layer in `decode glmgolden`), its boot is not yet: the dynamic expert cache (#175), the NVMe \
expert tier and the 200k boot (#149, plan step 14) and the vision tower (plan step 20) for family Glm5Next not built \
yet - refusing to boot (#159); `states --plan` prints its three-tier plan"
    )
}

impl ModelMeta {
    /// #159: parse a glm5_next config. Its own ledger: every key in the `GLM5_NEXT_*` lists is
    /// required with its type (one error names every missing key), every other key refuses by
    /// name unless it is on the ignore list, and the formula facts this family has exactly one
    /// form of refuse any other value by name.
    fn from_glm5_next(
        config: &Value,
        tc: &Value,
        config_path: &str,
        generation_config_path: Option<&str>,
        ids: GenIds,
    ) -> Result<ModelMeta, String> {
        let family = Family::Glm5Next;
        let str_list = |key: &str| -> Option<Vec<String>> {
            tc.get(key)?.as_array()?.iter().map(|v| v.as_str().map(str::to_string)).collect()
        };
        let lac = tc.get("linear_attn_config").filter(|v| v.is_object());
        let lac_list = |key: &str| -> Option<Vec<u64>> { lac?.get(key)?.as_array()?.iter().map(Value::as_u64).collect() };
        let mut missing: Vec<String> = Vec::new();
        for k in GLM5_NEXT_U64_KEYS {
            if u64_of(tc, k).is_none() {
                missing.push(format!("text_config.{k}"));
            }
        }
        for k in GLM5_NEXT_F64_KEYS {
            if f64_of(tc, k).is_none() {
                missing.push(format!("text_config.{k}"));
            }
        }
        for k in GLM5_NEXT_BOOL_KEYS {
            if tc.get(*k).and_then(Value::as_bool).is_none() {
                missing.push(format!("text_config.{k}"));
            }
        }
        for k in GLM5_NEXT_STR_KEYS {
            if tc.get(*k).and_then(Value::as_str).is_none() {
                missing.push(format!("text_config.{k}"));
            }
        }
        for k in GLM5_NEXT_LIST_KEYS {
            if str_list(k).is_none_or(|l| l.is_empty()) {
                missing.push(format!("text_config.{k}"));
            }
        }
        if ids.eos_token_ids.as_ref().is_none_or(|v| v.is_empty()) {
            missing.push("generation_config.json eos_token_id (or text_config.eos_token_id)".to_string());
        }
        match lac {
            None => missing.push("text_config.linear_attn_config".to_string()),
            Some(l) => {
                for k in ["num_heads", "head_dim", "short_conv_kernel_size"] {
                    if u64_of(l, k).is_none() {
                        missing.push(format!("text_config.linear_attn_config.{k}"));
                    }
                }
                if f64_of(l, "gate_lower_bound").is_none() {
                    missing.push("text_config.linear_attn_config.gate_lower_bound".to_string());
                }
                for k in ["kda_layers", "full_attn_layers"] {
                    if lac_list(k).is_none() {
                        missing.push(format!("text_config.linear_attn_config.{k}"));
                    }
                }
            }
        }
        if !missing.is_empty() {
            return Err(format!("{config_path}: missing required key(s): {}", missing.join(", ")));
        }
        let flat = !config.get("text_config").is_some_and(Value::is_object);
        let unknown = unknown_keys(tc, family, flat);
        if !unknown.is_empty() {
            return Err(format!(
                "{config_path}: key(s) this engine does not know for family {family:?} ({}): {} - refusing rather than \
silently ignoring them (Crow #300, #159); each must be consumed or put on the ignore list in meta.rs with its reason",
                family.model_type(),
                unknown.join(", ")
            ));
        }
        let u = |key: &str| u64_of(tc, key).unwrap();
        let f = |key: &str| f64_of(tc, key).unwrap();
        let b = |key: &str| tc.get(key).and_then(Value::as_bool).unwrap();
        let st = |key: &str| tc.get(key).and_then(Value::as_str).unwrap().to_string();
        let lu = |key: &str| u64_of(lac.unwrap(), key).unwrap();
        // the formula facts with exactly one implemented form, refused by name
        let refuse = |key: &str, got: String, want: String| -> Result<(), String> {
            Err(format!(
                "{config_path}: text_config.{key} {got} is not implemented for family Glm5Next (implemented: {want}) - refusing (#159)"
            ))
        };
        for (key, want) in [("scoring_func", "sigmoid"), ("topk_method", "noaux_tc"), ("moe_router_dtype", "float32"), ("hidden_act", "silu")] {
            if st(key) != want {
                refuse(key, format!("'{}'", st(key)), want.to_string())?;
            }
        }
        for (key, want) in [
            ("mhc", true),
            ("mla_use_nope", true),
            ("norm_topk_prob", true),
            ("index_kpool_compress", true),
            ("index_kpool_always_select_tail", true),
            ("attention_bias", false),
        ] {
            if b(key) != want {
                refuse(key, b(key).to_string(), want.to_string())?;
            }
        }
        for (key, want) in [("n_group", 1), ("topk_group", 1), ("qk_rope_head_dim", 0)] {
            if u(key) != want {
                refuse(key, u(key).to_string(), want.to_string())?;
            }
        }
        // the head dims: qk = nope + rope (HF), `head_dim` is HF's placeholder 0 or that sum
        let qk = u("qk_nope_head_dim") + u("qk_rope_head_dim");
        if u("qk_head_dim") != qk {
            refuse("qk_head_dim", u("qk_head_dim").to_string(), format!("qk_nope_head_dim + qk_rope_head_dim = {qk}"))?;
        }
        if u("head_dim") != 0 && u("head_dim") != qk {
            refuse("head_dim", u("head_dim").to_string(), format!("0 or qk_head_dim {qk}"))?;
        }
        // HF `validate_architecture`: the selection budget is whole pools
        if u("index_kpool") == 0 || u("index_topk") % u("index_kpool") != 0 {
            refuse("index_topk", u("index_topk").to_string(), format!("a multiple of index_kpool {}", u("index_kpool")))?;
        }
        // the per-layer lists: one entry per trunk layer, each a form the family row names
        let layers = u("num_hidden_layers") as usize;
        let positions = |list: &[String], what: &str| -> Vec<u64> {
            list.iter().enumerate().filter(|(_, t)| t.as_str() == what).map(|(i, _)| i as u64).collect()
        };
        let layer_types = str_list("layer_types").unwrap();
        let mlp_types = str_list("mlp_layer_types").unwrap();
        let indexer_types = str_list("indexer_types").unwrap();
        for (key, list, forms) in [
            ("layer_types", &layer_types, &["linear_attention", "deepseek_sparse_attention"][..]),
            ("mlp_layer_types", &mlp_types, &["dense", "sparse"][..]),
            // a "shared" layer reuses another layer's top-k: no arm for it (#159)
            ("indexer_types", &indexer_types, &["full"][..]),
        ] {
            if list.len() != layers {
                refuse(key, format!("with {} entries", list.len()), format!("one per layer, {layers}"))?;
            }
            if let Some((i, t)) = list.iter().enumerate().find(|(_, t)| !forms.contains(&t.as_str())) {
                refuse(&format!("{key}[{i}]"), format!("'{t}'"), forms.join(", "))?;
            }
        }
        let dsa_layers = positions(&layer_types, "deepseek_sparse_attention");
        let kda_layers = positions(&layer_types, "linear_attention");
        for (key, from_types) in [("full_attn_layers", &dsa_layers), ("kda_layers", &kda_layers)] {
            if lac_list(key).unwrap() != *from_types {
                refuse(&format!("linear_attn_config.{key}"), format!("{:?}", lac_list(key).unwrap()), format!("the positions layer_types names, {from_types:?}"))?;
            }
        }
        let dense = positions(&mlp_types, "dense");
        let k = u("first_k_dense_replace");
        if dense != (0..k).collect::<Vec<_>>() {
            refuse("mlp_layer_types", format!("dense at {dense:?}"), format!("dense at the first first_k_dense_replace = {k} layers"))?;
        }
        let tie_word_embeddings = b("tie_word_embeddings");
        if let Some(top) = config.get("tie_word_embeddings").and_then(Value::as_bool) {
            if top != tie_word_embeddings {
                return Err(format!(
                    "{config_path}: tie_word_embeddings {top} (top level) disagrees with text_config.tie_word_embeddings {tie_word_embeddings} - refusing (Crow #300)"
                ));
            }
        }
        let family_keys = FamilyKeys::Glm5Next {
            first_k_dense_replace: k,
            hc_mult: u("hc_mult"),
            hc_sinkhorn_iters: u("hc_sinkhorn_iters"),
            hc_eps: f("hc_eps"),
            q_lora_rank: u("q_lora_rank"),
            kv_lora_rank: u("kv_lora_rank"),
            qk_nope_head_dim: u("qk_nope_head_dim"),
            v_head_dim: u("v_head_dim"),
            index_n_heads: u("index_n_heads"),
            index_head_dim: u("index_head_dim"),
            index_kpool: u("index_kpool"),
            index_topk: u("index_topk"),
            linear_num_heads: lu("num_heads"),
            linear_head_dim: lu("head_dim"),
            linear_conv_kernel: lu("short_conv_kernel_size"),
            linear_lower_bound: f64_of(lac.unwrap(), "gate_lower_bound").unwrap(),
            intermediate_size: u("intermediate_size"),
            n_routed_experts: u("n_routed_experts"),
            num_experts_per_tok: u("num_experts_per_tok"),
            n_shared_experts: u("n_shared_experts"),
            moe_intermediate_size: u("moe_intermediate_size"),
            routed_scaling_factor: f("routed_scaling_factor"),
            swiglu_limit: f("swiglu_limit"),
            dsa_layers,
            kda_layers,
        };
        // no RoPE: the Qwen rope fields read as "none"; the GDN-named fields carry the KDA numbers
        let no_rope = "none: glm5_next has no RoPE (text_config.qk_rope_head_dim 0)".to_string();
        Ok(ModelMeta {
            config_path: config_path.to_string(),
            generation_config_path: generation_config_path.map(str::to_string),
            model_type: family.model_type().to_string(),
            rms_norm_eps: f("rms_norm_eps"),
            rope_theta: 0.0,
            rope_theta_source: no_rope.clone(),
            partial_rotary_factor: 0.0,
            head_dim: u("head_dim"),
            hidden_size: u("hidden_size"),
            num_attention_heads: u("num_attention_heads"),
            num_key_value_heads: u("num_key_value_heads"),
            num_hidden_layers: u("num_hidden_layers"),
            layer_types,
            full_attention_interval: None,
            vocab_size: u("vocab_size"),
            max_position_embeddings: u("max_position_embeddings"),
            eos_token_ids: ids.eos_token_ids.unwrap(),
            eos_from_generation: ids.eos_from_generation,
            text_eos_token_id: ids.text_eos_token_id,
            bos_token_id: ids.bos_token_id,
            text_bos_token_id: ids.text_bos_token_id,
            mrope_section: None,
            rope_type: "none".to_string(),
            rope_type_source: no_rope,
            rope_scaling: None,
            family,
            hidden_act: st("hidden_act"),
            attention_bias: b("attention_bias"),
            mamba_ssm_dtype: String::new(),
            output_gate_type: String::new(),
            mrope_interleaved: None,
            linear_num_key_heads: lu("num_heads"),
            linear_num_value_heads: lu("num_heads"),
            linear_key_head_dim: lu("head_dim"),
            linear_value_head_dim: lu("head_dim"),
            linear_conv_kernel_dim: lu("short_conv_kernel_size"),
            tie_word_embeddings,
            mtp_num_hidden_layers: u("num_nextn_predict_layers"),
            vision_out_hidden: config.get("vision_config").and_then(|v| u64_of(v, "out_hidden_size")),
            family_keys,
        })
    }

    /// #159: the glm5_next geometry of a parsed glm5_next config. `Err` names the value that
    /// has no `Glm5Geo` form (another family, a stop-id list that is not three ids, MLA + DSA
    /// layers that are not every n-th layer).
    pub fn glm5_geo(&self) -> Result<Glm5Geo, String> {
        let FamilyKeys::Glm5Next {
            first_k_dense_replace,
            hc_mult,
            hc_sinkhorn_iters,
            hc_eps,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            v_head_dim,
            index_n_heads,
            index_head_dim,
            index_kpool,
            index_topk,
            linear_num_heads,
            linear_head_dim,
            linear_conv_kernel,
            linear_lower_bound,
            intermediate_size,
            n_routed_experts,
            num_experts_per_tok,
            n_shared_experts,
            moe_intermediate_size,
            routed_scaling_factor,
            swiglu_limit,
            dsa_layers,
            kda_layers,
        } = &self.family_keys
        else {
            return Err(format!("family {:?} has no Glm5Geo", self.family));
        };
        let z = |v: u64| v as usize;
        let eos_ids: [usize; 3] = match self.eos_token_ids.as_slice() {
            [a, b, c] if *a >= 0 && *b >= 0 && *c >= 0 => [*a as usize, *b as usize, *c as usize],
            other => return Err(format!("eos_token_id {other:?}: the glm5_next row holds exactly three stop ids")),
        };
        let layers = z(self.num_hidden_layers);
        let attn_interval = dsa_layers.first().map_or(0, |p| z(*p) + 1);
        let periodic: Vec<u64> = (0..layers as u64).filter(|l| attn_interval > 0 && *l as usize % attn_interval == attn_interval - 1).collect();
        if attn_interval == 0 || *dsa_layers != periodic {
            return Err(format!("text_config.layer_types: deepseek_sparse_attention at {dsa_layers:?} is not every n-th layer"));
        }
        Ok(Glm5Geo {
            hidden: z(self.hidden_size),
            layers,
            attn_interval,
            dsa_layers: dsa_layers.len(),
            kda_layers: kda_layers.len(),
            hc_streams: z(*hc_mult),
            hc_sinkhorn_iters: z(*hc_sinkhorn_iters),
            hc_eps: *hc_eps,
            mla_heads: z(self.num_attention_heads),
            q_lora: z(*q_lora_rank),
            kv_lora: z(*kv_lora_rank),
            nope_dim: z(*qk_nope_head_dim),
            v_dim: z(*v_head_dim),
            index_heads: z(*index_n_heads),
            index_head_dim: z(*index_head_dim),
            index_kpool: z(*index_kpool),
            index_topk: z(*index_topk),
            kda_heads: z(*linear_num_heads),
            kda_head_dim: z(*linear_head_dim),
            kda_conv: z(*linear_conv_kernel),
            kda_lower_bound: *linear_lower_bound,
            dense_prefix: z(*first_k_dense_replace),
            dense_inter: z(*intermediate_size),
            experts: z(*n_routed_experts),
            topk: z(*num_experts_per_tok),
            shared_experts: z(*n_shared_experts),
            expert_inter: z(*moe_intermediate_size),
            routed_scaling: *routed_scaling_factor,
            swiglu_limit: *swiglu_limit,
            rms_eps: self.rms_norm_eps,
            vocab: z(self.vocab_size),
            tie_word_embeddings: self.tie_word_embeddings,
            context_max: z(self.max_position_embeddings),
            context_floor: geo::CONTEXT_FLOOR,
            eos_ids,
            mtp_layers: z(self.mtp_num_hidden_layers),
            vision_out_hidden: self.vision_out_hidden.map(z),
        })
    }

    /// #159: the glm5_next checks: every `Glm5Geo` field against the family row
    /// ([`Glm5Geo::GLM_5_3_FLASH`]), then the stop ids inside the vocab. A config with no
    /// `Glm5Geo` form is one red row naming why.
    fn glm5_checks(&self) -> Vec<Check> {
        let at = format!("glm5_next_text family row, {GLM5_NEXT_SOURCE}");
        let source = "config.json via meta::ModelMeta::glm5_geo".to_string();
        let mut c: Vec<Check> = match self.glm5_geo() {
            Ok(g) => g
                .rows()
                .into_iter()
                .zip(Glm5Geo::GLM_5_3_FLASH.rows())
                .map(|((name, got), (_, pin))| Check { name, pinned: format!("{pin} ({at})"), ok: got == pin, config: got, source: source.clone() })
                .collect(),
            Err(why) => vec![Check { name: "glm5_geo", pinned: format!("a Glm5Geo ({at})"), config: why, source, ok: false }],
        };
        let vocab = self.vocab_size as i64;
        c.push(Check {
            name: "eos_ids_in_vocab",
            pinned: format!("0 <= id < {}", Glm5Geo::GLM_5_3_FLASH.vocab),
            config: format!("{:?} against vocab {}", self.eos_token_ids, self.vocab_size),
            source: "generation_config.json eos_token_id vs text_config.vocab_size".to_string(),
            ok: !self.eos_token_ids.is_empty() && self.eos_token_ids.iter().all(|id| *id >= 0 && *id < vocab),
        });
        c
    }
}

// ---- the boot door ----

/// Crow #300 C5: the 27B fixture's derived `Geo` (engine/tests/fixtures/Qwen3.8-27B),
/// for the tests of the modules that plan per family
#[cfg(test)]
pub(crate) fn dense_fixture_geo() -> Geo {
    let dir = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/Qwen3.8-27B");
    ModelMeta::from_config_files(&format!("{dir}/config.json"), Some(&format!("{dir}/generation_config.json")))
        .and_then(|m| m.geo())
        .unwrap()
}

/// the geometry as the boot prints it, one `[meta]` row per `Geo` field
pub fn geo_table(g: &Geo) -> String {
    g.rows().iter().map(|(n, v)| format!("[meta]   {n:<20} {v}")).collect::<Vec<_>>().join("\n")
}

/// The boot decision on a parsed config, pure (no panic, no stash), so every
/// branch is unit-tested:
///
/// - any check of the family row red → `Err`, the #94 table;
/// - the `Geo` not derivable → `Err`, the named reason;
/// - Flash-Next whose `Geo` differs from `Geo::FLASH_NEXT` → `Err`, a table of
///   every differing field (derived vs pinned);
/// - Flash-Next, everything equal → `Ok(geo)`;
/// - dense → `Ok(geo)` (C5: the family's first unbuilt block refuses it next,
///   `Geo::built` in `boot::model_geo`).
pub fn verdict(meta: &ModelMeta) -> Result<Geo, String> {
    let all = meta.checks();
    let bad: Vec<&Check> = all.iter().filter(|c| !c.ok).collect();
    if !bad.is_empty() {
        let table = bad.iter().map(|c| format!("  {}", c.line())).collect::<Vec<_>>().join("\n");
        return Err(format!(
            "[meta] {} of {} constants differ from the engine pins - refusing to boot (issue #94):\n{}\n\
[meta]   config: {}\n\
[meta] a different checkpoint must be ported consciously, not silently - see docs/acceptance/issue-94.md",
            bad.len(),
            all.len(),
            table,
            meta.config_path
        ));
    }
    // #159: glm5_next passed its family row; its boot refuses naming what is not built (#161)
    if meta.family == Family::Glm5Next {
        return Err(glm5_not_built(all.len(), &meta.config_path));
    }
    let geo = meta.geo().map_err(|why| format!("[meta] {why} - refusing to boot (Crow #300) [{}]", meta.config_path))?;
    match meta.family {
        Family::FlashNext => {
            let diff = geo.diff(&Geo::FLASH_NEXT);
            if diff.is_empty() {
                return Ok(geo);
            }
            let table = diff
                .iter()
                .map(|(name, got, want)| format!("  {name}: config derives {got}, Geo::FLASH_NEXT pins {want}"))
                .collect::<Vec<_>>()
                .join("\n");
            Err(format!(
                "[meta] {} of {} runtime Geo fields differ from Geo::FLASH_NEXT - refusing to boot (Crow #300):\n{}\n\
[meta]   config: {}",
                diff.len(),
                geo.rows().len(),
                table,
                meta.config_path
            ))
        }
        Family::Qwen35Dense => Ok(geo),
        Family::Glm5Next => unreachable!("glm5_next returned above"),
    }
}

/// The gate `boot::open_model` calls FIRST, before the container is mapped and
/// the CUDA context created ([`gate`], plus the log lines, the panic and the #96
/// stash). Since Crow #300 C7 it judges the config the container's index names:
/// an index v2 container's own `model` block, or for the index v1 container of
/// record `CROW_MODEL_DIR` / `models/` (`model_dir` is the variable, read by the
/// caller):
///
/// - config.json found, every check green and (Flash-Next) the runtime `Geo`
///   equal to `Geo::FLASH_NEXT` → one INFO line, the meta and the `Geo`
///   returned for the later phases;
/// - a dense checkpoint, every check green → the INFO line and the derived
///   geometry, returned the same way (C5: `boot::model_geo` refuses it next, at
///   its first unbuilt block);
/// - config.json found and anything red or unreadable → a panic carrying the
///   whole table (llama.cpp-style loud failure); since C7 that includes a
///   config that does not fit the container (`container_mismatch`) and, for an
///   index v2, a `CROW_MODEL_DIR` whose config is not the container's;
/// - no config.json anywhere next to the container → one WARN line, boot
///   continues (the selftest package ships without `models/` on purpose).
pub fn assert_pinned(cnq_path: &str, peek: &IndexPeek, model_dir: Option<&str>) -> Option<(ModelMeta, Geo)> {
    let (meta, geo) = match gate(cnq_path, peek, model_dir) {
        Ok(Some(pair)) => pair,
        Ok(None) => {
            tracing::warn!(
                target: "meta",
                "meta: no config.json beside {cnq_path} - 0 constants verified, the pins stand unchecked \
(set CROW_MODEL_DIR to the checkpoint directory to arm the #94 gate)"
            );
            return None;
        }
        Err(GateRefusal::Unreadable(why)) => panic!("{why} - the #94 metadata gate refuses to boot on an unreadable truth source"),
        Err(GateRefusal::Refused(refusal)) => {
            // the table goes to the log too, not only to the panic text
            for line in refusal.lines() {
                tracing::error!(target: "meta", "{line}");
            }
            panic!("{refusal}")
        }
    };
    match meta.family {
        Family::FlashNext => tracing::info!(
            target: "meta",
            "meta: {} constants verified against config.json (zero numeric change) [{}]; family {:?} ({}), runtime Geo == Geo::FLASH_NEXT ({} fields)",
            meta.checks().len(),
            meta.config_path,
            meta.family,
            meta.family.model_type(),
            geo.rows().len()
        ),
        Family::Qwen35Dense => {
            tracing::info!(
                target: "meta",
                "meta: {} constants verified against config.json ({QWEN35_DENSE_SOURCE}) [{}]; family {:?} ({}), runtime Geo derived ({} fields):",
                meta.checks().len(),
                meta.config_path,
                meta.family,
                meta.family.model_type(),
                geo.rows().len()
            );
            for line in geo_table(&geo).lines() {
                tracing::info!(target: "meta", "{line}");
            }
        }
        Family::Glm5Next => unreachable!("verdict refuses glm5_next at its first unbuilt arm (#159)"),
    }
    // #96: stash the rope truth for the two boot-time readers that cannot be
    // handed it as a parameter (ThreeStates::allocate builds the table,
    // Kernels::new threads the mscale). Set only on the green path: a red
    // config panicked above, a missing one has nothing to say — both leave the
    // `None` default in force, which is the byte-identical behavior.
    let _ = BOOT_ROPE_SCALING.set(meta.rope_scaling);
    let _ = BOOT_TRAINING_CONTEXT.set(meta.training_context());
    Some((meta, geo))
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
        doctored_from(REAL_DIR, mutate).unwrap()
    }

    /// `doctored` on any checkpoint dir, the parse result handed back as is
    fn doctored_from(src: &str, mutate: impl FnOnce(&mut Value, &mut Value)) -> Result<ModelMeta, String> {
        let mut config: Value = serde_json::from_str(&std::fs::read_to_string(format!("{src}/config.json")).unwrap()).unwrap();
        let mut generation: Value = serde_json::from_str(&std::fs::read_to_string(format!("{src}/generation_config.json")).unwrap()).unwrap();
        mutate(&mut config, &mut generation);
        // tests run in parallel threads of ONE process, so the pid alone is not
        // a unique dir: every doctored copy gets its own counter slot
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("crow-meta-test-{}-{}", std::process::id(), n));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        std::fs::write(dir.join("generation_config.json"), generation.to_string()).unwrap();
        let m = ModelMeta::from_config_files(&dir.join("config.json").to_string_lossy(), Some(&dir.join("generation_config.json").to_string_lossy()));
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

    /// Crow's install (cwd `%LOCALAPPDATA%\Crow`, the container in
    /// `models/Qwen3.8-Flash-Next-CNQ4.5-M/` with no config.json): a lone
    /// unrelated checkpoint under `models/` (whisper-small, 2026-10-01) must not
    /// become the truth source - the lookup finds nothing and boot warns
    #[test]
    fn a_lone_unrelated_checkpoint_is_not_the_containers_config() {
        let t = std::env::temp_dir().join(format!("crow-meta-cfgdir-{}", std::process::id()));
        std::fs::remove_dir_all(&t).ok();
        let whisper = t.join("models").join("whisper-small");
        let own = t.join("models").join("Qwen3.8-Flash-Next-CNQ4.5-M");
        std::fs::create_dir_all(&whisper).unwrap();
        std::fs::create_dir_all(&own).unwrap();
        std::fs::write(whisper.join("config.json"), "{}").unwrap();
        let cnq = own.join("Qwen3.8-Flash-Next-CNQ4.5-M.cnq").to_string_lossy().into_owned();
        let none = config_dir_beside(&cnq, &t);
        // the checkpoint the name names is still found beside the unrelated one
        let original = t.join("models").join("Qwen3.8-Flash-Next-original");
        std::fs::create_dir_all(&original).unwrap();
        std::fs::write(original.join("config.json"), "{}").unwrap();
        let found = config_dir_beside(&cnq, &t);
        std::fs::remove_dir_all(&t).ok();
        assert_eq!(none, None, "an unrelated lone checkpoint was taken");
        assert_eq!(found, Some(original));
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

    // ---- #96: rope_scaling ----

    /// the checkpoint of record carries NO rope_scaling object and rope_type
    /// "default" — the two facts the byte-identity contract stands on
    #[test]
    fn the_real_config_has_no_scaling_and_a_default_rope_type() {
        let m = real_meta();
        assert!(m.rope_scaling.is_none(), "the checkpoint of record must parse scaling-free");
        assert_eq!(m.rope_type, "default");
        assert_eq!(m.training_context(), 262_144);
        assert!(!fired(&m, "rope_type"), "the new rope_type check must be green on the checkpoint of record");
    }

    /// a doctored rope_type fires the new check by name (a checkpoint announcing
    /// scaling through the token nothing reads is a loud mismatch, not a guess)
    #[test]
    fn a_doctored_rope_type_fires_by_name() {
        let m = doctored(|c, _| c["text_config"]["rope_parameters"]["rope_type"] = json!("yarn"));
        assert!(fired(&m, "rope_type"));
    }

    /// config-driven YaRN: the full object parses, and every derived number is
    /// the llama.cpp reference math (the corr range 14..22 is the checkpoint's
    /// own dims/base/context with the default betas)
    #[test]
    fn a_yarn_rope_scaling_parses_with_the_reference_math() {
        let m = doctored(|c, _| {
            c["text_config"]["rope_scaling"] = json!({
                "rope_type": "yarn",
                "factor": 4.0,
                "original_max_position_embeddings": 262_144,
                "beta_fast": 32.0,
                "beta_slow": 1.0
            });
        });
        let s = m.rope_scaling.expect("the object must parse");
        assert_eq!(s.kind, RopeKind::Yarn);
        assert_eq!(s.factor, 4.0);
        assert_eq!(s.freq_scale(), 0.25f32);
        assert_eq!(s.original_context, 262_144);
        assert_eq!(m.training_context(), 262_144);
        // get_mscale(4, 1) = 1 + 0.1·ln 4
        assert!((s.mscale() - (1.0 + 0.1 * 4.0f64.ln())).abs() < 1e-12, "{}", s.mscale());
        // ggml_rope_yarn_corr_dims(64, 262144, 1e7, 32, 1) = floor(14.23), min(63, ceil(21.11))
        assert_eq!(s.yarn_corr_range(64, 1e7), (14.0, 22.0));
        // a parse must never arm the engine: only assert_pinned's green path sets the stash
        assert!(boot_rope_scaling().is_none(), "from_config_files must not touch the boot stash");
    }

    /// the legacy "type" spelling parses, and the llama.cpp defaults apply
    /// (beta 32/1, original context falling back to max_position_embeddings)
    #[test]
    fn a_legacy_typed_rope_scaling_takes_the_defaults() {
        let m = doctored(|c, _| {
            c["text_config"]["rope_scaling"] = json!({ "type": "yarn", "factor": 2.5 });
        });
        let s = m.rope_scaling.expect("the legacy spelling must parse");
        assert_eq!(s.kind, RopeKind::Yarn);
        assert_eq!(s.beta_fast, 32.0);
        assert_eq!(s.beta_slow, 1.0);
        assert_eq!(s.original_context, 262_144, "no original_max_position_embeddings: the config max is the fallback");
        assert!((s.freq_scale() - 0.4).abs() < 1e-6);
    }

    /// linear and ntk-aware parse too, and neither touches the attention
    /// temperature — only yarn has an mscale
    #[test]
    fn linear_and_ntk_parse_with_no_mscale() {
        let lin = doctored(|c, _| c["text_config"]["rope_scaling"] = json!({ "rope_type": "linear", "factor": 4.0 }));
        let lin = lin.rope_scaling.unwrap();
        assert_eq!(lin.kind, RopeKind::Linear);
        assert_eq!(lin.mscale(), 1.0);
        assert_eq!(lin.freq_scale(), 0.25f32);
        let ntk = doctored(|c, _| c["text_config"]["rope_scaling"] = json!({ "rope_type": "ntk-aware", "factor": 4.0 }));
        let ntk = ntk.rope_scaling.unwrap();
        assert_eq!(ntk.kind, RopeKind::NtkAware);
        assert_eq!(ntk.mscale(), 1.0);
        // base' = 1e7 · 4^(64/62): the NTK-aware theta rewrite
        assert!((ntk.ntk_base(64, 1e7) - 1e7 * 4.0f64.powf(64.0 / 62.0)).abs() < 1e-6 * 1e7);
        // a "default" scaling object configures nothing
        let dflt = doctored(|c, _| c["text_config"]["rope_scaling"] = json!({ "rope_type": "default", "factor": 4.0 }));
        assert_eq!(dflt.rope_scaling.unwrap().kind, RopeKind::Default);
    }

    /// an unsupported scaling type (longrope's factor tables, dynamic, su-rope)
    /// is a NAMED parse error, never a silently ignored object
    #[test]
    fn an_unsupported_rope_scaling_type_is_a_named_error() {
        let mut config: Value = serde_json::from_str(&std::fs::read_to_string(format!("{REAL_DIR}/config.json")).unwrap()).unwrap();
        config["text_config"]["rope_scaling"] = json!({ "rope_type": "longrope", "factor": 4.0 });
        let dir = std::env::temp_dir().join(format!("crow-meta-longrope-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let err = ModelMeta::from_config_files(&dir.join("config.json").to_string_lossy(), None).unwrap_err();
        assert!(err.contains("longrope"), "the error must name the type: {err}");
        assert!(err.contains("issue #96"), "the error must point at the refusing gate: {err}");
        // a missing factor is a named missing key of the same class
        config["text_config"]["rope_scaling"] = json!({ "rope_type": "yarn" });
        std::fs::create_dir_all(&dir).unwrap(); // the remove above took the dir; the second write needs it back
        std::fs::write(dir.join("config.json"), config.to_string()).unwrap();
        let err = ModelMeta::from_config_files(&dir.join("config.json").to_string_lossy(), None).unwrap_err();
        std::fs::remove_dir_all(&dir).ok();
        assert!(err.contains("rope_scaling.factor"), "the error must name the key: {err}");
    }

    // ---- Crow #300 phase 1: C1 family table + key ledger, C2 runtime Geo ----

    /// the dense fixture: Qwen/Qwen3.8-27B @ 1d4bf0f2 (revision
    /// 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0), config.json sha256 191e0af2…,
    /// generation_config.json sha256 e70c136c… (byte-identical to Flash-Next's),
    /// copied from models/Qwen3.8-27B/ (sha256-checked 2026-09-25)
    const DENSE_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/Qwen3.8-27B");

    fn dense_meta() -> ModelMeta {
        ModelMeta::from_config_files(&format!("{DENSE_DIR}/config.json"), Some(&format!("{DENSE_DIR}/generation_config.json"))).unwrap()
    }

    #[test]
    fn family_detection_names_both_families_and_refuses_an_unknown_one() {
        assert_eq!(real_meta().family, Family::FlashNext);
        assert_eq!(dense_meta().family, Family::Qwen35Dense);
        assert_eq!(dense_meta().model_type, "qwen3_5_text");
        let err = doctored_from(DENSE_DIR, |c, _| c["text_config"]["model_type"] = json!("qwen3_5_moe_text")).unwrap_err();
        assert!(err.contains("'qwen3_5_moe_text' is not a model family"), "{err}");
        assert!(err.contains("Crow #300"), "{err}");
    }

    /// the 27B fixture parses, is all green against its own family row, and
    /// derives the geometry of crow-nest#116 / Crow #300
    #[test]
    fn the_27b_fixture_parses_and_derives_its_geo() {
        let m = dense_meta();
        for c in m.checks() {
            assert!(c.ok, "expected green against the dense row: {}", c.line());
        }
        assert!(!m.checks().iter().any(|c| c.name == "ple_eos"), "no PLE, no PLE end marker check");
        let g = m.geo().unwrap();
        assert_eq!(g.family, Family::Qwen35Dense);
        assert_eq!((g.layers, g.gdn_layers, g.attn_layers, g.attn_interval), (64, 48, 16, 4));
        assert_eq!(g.hidden, 5120);
        assert_eq!(g.ffn, Ffn::Dense { inter: 17408 });
        assert_eq!((g.q_heads, g.kv_heads, g.head_dim, g.gqa()), (24, 4, 256, 6));
        assert_eq!(g.gate_act, GateAct::Silu, "output_gate_type swish");
        assert!(g.attn_output_gate);
        assert_eq!((g.q_rows(), g.kv_rows(), g.core()), (12288, 1024, 6144));
        assert_eq!(g.vocab, 248_320);
        assert!(!g.tie_word_embeddings, "untied lm_head");
        assert_eq!(g.mtp_layers, 1);
        assert_eq!((g.residual, g.attn, g.ple, g.final_norm), (Residual::Plain, Attn::Full, None, FinalNorm::Rms));
        assert_eq!(g.attn.sel_max(), None, "uncapped causal attention");
        assert_eq!((g.gdn_key_heads, g.gdn_value_heads, g.gdn_key_dim, g.gdn_value_dim, g.conv_kernel), (16, 48, 128, 128, 4));
        assert_eq!((g.rope_pairs, g.rope_theta, g.rms_eps), (32, 1e7, 1e-6));
        assert_eq!(g.eos_ids, [248046, 248044]);
        assert_eq!(g.context_max, 262_144);
        assert_eq!(g.vision_out_hidden, Some(5120));
        for l in 0..64 {
            assert_eq!(g.is_attn(l), m.layer_types[l] == "full_attention", "layer {l}");
        }
    }

    /// the checkpoint of record derives exactly `Geo::FLASH_NEXT`, and its
    /// check table is still today's 21 constants (#94 phase 1 + #96 rope_type)
    #[test]
    fn the_flash_next_config_derives_geo_flash_next() {
        let m = real_meta();
        assert_eq!(m.checks().len(), 21, "the 20 of #94 phase 1 + rope_type of #96");
        let g = m.geo().unwrap();
        assert_eq!(g.diff(&Geo::FLASH_NEXT), vec![]);
        assert_eq!(g, Geo::FLASH_NEXT);
        assert_eq!(verdict(&m), Ok(Geo::FLASH_NEXT));
    }

    /// an unknown text_config key refuses the parse BY NAME; so does a key of
    /// the other family and an unknown rope_parameters key; a key on the
    /// ignore list does not
    #[test]
    fn an_unknown_text_config_key_is_refused_by_name() {
        let err = doctored_from(REAL_DIR, |c, _| c["text_config"]["sliding_window"] = json!(4096)).unwrap_err();
        assert!(err.contains("text_config.sliding_window"), "{err}");
        assert!(err.contains("Crow #300"), "{err}");
        let err = doctored_from(REAL_DIR, |c, _| c["text_config"]["intermediate_size"] = json!(17408)).unwrap_err();
        assert!(err.contains("text_config.intermediate_size") && err.contains("FlashNext"), "{err}");
        let err = doctored_from(DENSE_DIR, |c, _| c["text_config"]["num_experts"] = json!(512)).unwrap_err();
        assert!(err.contains("text_config.num_experts") && err.contains("Qwen35Dense"), "{err}");
        let err = doctored_from(DENSE_DIR, |c, _| c["text_config"]["rope_parameters"]["mrope_scale"] = json!(2)).unwrap_err();
        assert!(err.contains("text_config.rope_parameters.mrope_scale"), "{err}");
        // on the ignore list: read and dropped, no refusal
        assert!(doctored_from(DENSE_DIR, |c, _| c["text_config"]["initializer_range"] = json!(0.01)).is_ok());
        assert!(IGNORED_KEYS.iter().all(|(_, _, why)| !why.is_empty()));
    }

    /// formula facts with one implemented value refuse other values by name
    #[test]
    fn an_unimplemented_formula_value_is_refused_by_name() {
        let err = doctored_from(DENSE_DIR, |c, _| c["text_config"]["hidden_act"] = json!("gelu")).unwrap_err();
        assert!(err.contains("text_config.hidden_act 'gelu' is not implemented"), "{err}");
        let err = doctored_from(REAL_DIR, |c, _| c["text_config"]["output_gate_type"] = json!("tanh")).unwrap_err();
        assert!(err.contains("text_config.output_gate_type 'tanh'"), "{err}");
        let err = doctored_from(REAL_DIR, |c, _| c["tie_word_embeddings"] = json!(true)).unwrap_err();
        assert!(err.contains("tie_word_embeddings true (top level) disagrees"), "{err}");
    }

    /// a Flash-Next config that passes every one of the 21 checks but derives a
    /// different runtime Geo refuses the boot with a table naming the fields
    #[test]
    fn a_geo_mismatch_refuses_with_a_table() {
        let m = doctored(|c, _| {
            c["text_config"]["num_experts"] = json!(256);
            c["text_config"]["output_gate_type"] = json!("swish");
        });
        assert!(m.verify().is_empty(), "the 21 checks stay green");
        let err = verdict(&m).unwrap_err();
        assert!(err.contains("2 of 35 runtime Geo fields differ from Geo::FLASH_NEXT"), "{err}");
        assert!(err.contains("  ffn: config derives Moe { experts: 256"), "{err}");
        assert!(err.contains("  gate_act: config derives Silu, Geo::FLASH_NEXT pins Sigmoid"), "{err}");
    }

    /// C5: a dense checkpoint passes the metadata gate with its derived geometry
    /// (the table the boot prints) and the family check (`Geo::built`, which
    /// `boot::model_geo` calls before the container and the GPU; Crow #300 phase 2
    /// built the dense arms)
    #[test]
    fn a_dense_checkpoint_passes_the_gate_and_the_family_check() {
        let geo = verdict(&dense_meta()).unwrap();
        assert_eq!(geo, dense_fixture_geo());
        let table = geo_table(&geo);
        assert!(table.contains("[meta]   hidden               5120"), "{table}");
        assert!(table.contains("[meta]   ffn                  Dense { inter: 17408 }"), "{table}");
        assert_eq!(dense_meta().checks().len(), 20);
        assert_eq!(geo.built(), Ok(()));
        // the per-family floor: 64k (the image stack's point, Crow #300 2026-09-27), Flash-Next keeps 200k
        assert_eq!((geo.context_floor, Geo::FLASH_NEXT.context_floor), (65_536, 200_000));
        // a doctored dense config is refused by the dense row, not by the pins
        let m = doctored_from(DENSE_DIR, |c, _| c["text_config"]["hidden_size"] = json!(4096)).unwrap();
        let err = verdict(&m).unwrap_err();
        assert!(err.contains("hidden_size: pinned 5120 (qwen3_5_text family row"), "{err}");
    }

    // ---- #159: the glm5_next family at the gate ----

    const GLM_DIR: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/GLM-5.3-Flash");
    const FN_FIXTURE: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/Qwen3.8-Flash-Next");

    fn glm_meta() -> ModelMeta {
        ModelMeta::from_config_files(&format!("{GLM_DIR}/config.json"), Some(&format!("{GLM_DIR}/generation_config.json"))).unwrap()
    }

    /// the doctored GLM config's parse error, which must name `want`
    fn glm_refused(want: &str, mutate: impl FnOnce(&mut Value, &mut Value)) -> String {
        let err = doctored_from(GLM_DIR, mutate).unwrap_err();
        assert!(err.contains(want), "expected {want:?} in: {err}");
        err
    }

    /// #159: the GLM-5.3-Flash config (rev eb9eb208, sha256 bb8f01c4...) parses as glm5_next,
    /// every check of its family row is green, and its geometry is the row
    #[test]
    fn the_glm_config_parses_into_its_family_row() {
        let m = glm_meta();
        assert_eq!((m.family, m.model_type.as_str()), (Family::Glm5Next, "glm5_next_text"));
        for c in m.checks() {
            assert!(c.ok, "expected green against the glm5_next row: {}", c.line());
        }
        assert_eq!(m.checks().len(), 38, "37 Glm5Geo rows + eos_ids_in_vocab");
        assert_eq!(m.glm5_geo(), Ok(Glm5Geo::GLM_5_3_FLASH));
        assert_eq!(m.eos_token_ids, vec![154_820, 154_827, 154_829]);
        assert!(m.eos_from_generation);
        assert_eq!((m.rope_scaling, m.rope_type.as_str()), (None, "none"), "no RoPE");
        assert!(matches!(m.family_keys, FamilyKeys::Glm5Next { .. }));
    }

    /// #159: the gate accepts the family, the boot refuses it at its first unbuilt arm, by name,
    /// in the gate (before the container is mapped and before CUDA)
    #[test]
    fn a_glm_boot_refuses_at_its_first_unbuilt_arm() {
        let m = glm_meta();
        let err = verdict(&m).unwrap_err();
        assert!(err.contains("family Glm5Next (glm5_next_text): 38 constants verified"), "{err}");
        assert!(err.contains("its layers are built (mHC, KDA, MLA + DSA, router and MUL1 experts, head: #161-#165"), "{err}");
        assert!(err.contains("the dynamic expert cache (#175), the NVMe expert tier and the 200k boot (#149, plan step 14) and the vision tower (plan step 20) for family Glm5Next not built yet"), "{err}");
        assert!(err.contains("refusing to boot (#159)"), "{err}");
        assert_eq!(m.geo().unwrap_err(), err, "no runtime Geo until the arms exist");
    }

    /// #159: a model type that is not exactly glm5_next_text (another variant of the name, the
    /// HF wrapper's type) is refused by name; the refusal lists glm5_next_text as known
    #[test]
    fn an_unknown_glm_model_type_variant_is_refused_by_name() {
        for variant in ["glm5_next", "glm5_text", "glm5_next_vl_text", "GLM5_NEXT_TEXT"] {
            let err = glm_refused(&format!("'{variant}' is not a model family"), |c, _| c["text_config"]["model_type"] = json!(variant));
            assert!(err.contains("glm5_next_text = GLM-5.3-Flash"), "{err}");
        }
    }

    /// #159: every missing glm5_next key is named in one error, nested KDA keys included
    #[test]
    fn a_missing_glm_key_is_a_named_error() {
        let err = glm_refused("missing required key(s)", |c, _| {
            let tc = c["text_config"].as_object_mut().unwrap();
            tc.remove("kv_lora_rank");
            tc.remove("mlp_layer_types");
            tc.remove("scoring_func");
            c["text_config"]["linear_attn_config"].as_object_mut().unwrap().remove("num_heads");
        });
        for key in ["text_config.kv_lora_rank", "text_config.mlp_layer_types", "text_config.scoring_func", "text_config.linear_attn_config.num_heads"] {
            assert!(err.contains(key), "{key} not named: {err}");
        }
        glm_refused("generation_config.json eos_token_id", |c, g| {
            c["text_config"].as_object_mut().unwrap().remove("eos_token_id");
            g.as_object_mut().unwrap().remove("eos_token_id");
        });
    }

    /// #159: glm5_next has its own ledger: an unknown key, a Qwen key and an unknown KDA key
    /// refuse by name; a key on the glm5_next ignore list does not
    #[test]
    fn an_unknown_glm_key_is_refused_by_name() {
        glm_refused("text_config.sliding_window", |c, _| c["text_config"]["sliding_window"] = json!(4096));
        let err = glm_refused("text_config.partial_rotary_factor", |c, _| c["text_config"]["partial_rotary_factor"] = json!(0.25));
        assert!(err.contains("family Glm5Next"), "{err}");
        glm_refused("text_config.linear_attn_config.chunk_size", |c, _| c["text_config"]["linear_attn_config"]["chunk_size"] = json!(64));
        assert!(doctored_from(GLM_DIR, |c, _| c["text_config"]["router_aux_loss_coef"] = json!(0.01)).is_ok());
        assert_eq!(IGNORED_KEYS.iter().filter(|(f, _, _)| *f == Some(Family::Glm5Next)).count(), 4);
    }

    /// #159: each formula fact glm5_next has one form of refuses another value by its key
    #[test]
    fn each_glm_formula_fact_refuses_another_value_by_name() {
        glm_refused("text_config.scoring_func 'softmax' is not implemented for family Glm5Next", |c, _| c["text_config"]["scoring_func"] = json!("softmax"));
        glm_refused("text_config.topk_method 'greedy'", |c, _| c["text_config"]["topk_method"] = json!("greedy"));
        glm_refused("text_config.moe_router_dtype 'bfloat16'", |c, _| c["text_config"]["moe_router_dtype"] = json!("bfloat16"));
        glm_refused("text_config.mhc false", |c, _| c["text_config"]["mhc"] = json!(false));
        glm_refused("text_config.mla_use_nope false", |c, _| c["text_config"]["mla_use_nope"] = json!(false));
        glm_refused("text_config.norm_topk_prob false", |c, _| c["text_config"]["norm_topk_prob"] = json!(false));
        glm_refused("text_config.n_group 2", |c, _| c["text_config"]["n_group"] = json!(2));
        glm_refused("text_config.qk_rope_head_dim 64", |c, _| c["text_config"]["qk_rope_head_dim"] = json!(64));
        glm_refused("text_config.index_topk 2050", |c, _| c["text_config"]["index_topk"] = json!(2050));
        glm_refused("text_config.indexer_types[7] 'shared'", |c, _| c["text_config"]["indexer_types"][7] = json!("shared"));
        glm_refused("text_config.layer_types[5] 'full_attention'", |c, _| c["text_config"]["layer_types"][5] = json!("full_attention"));
        glm_refused("text_config.linear_attn_config.kda_layers", |c, _| c["text_config"]["linear_attn_config"]["kda_layers"][0] = json!(3));
        glm_refused("text_config.mlp_layer_types dense at [0, 1, 2, 4]", |c, _| c["text_config"]["mlp_layer_types"][4] = json!("dense"));
        glm_refused("tie_word_embeddings true (top level) disagrees", |c, _| c["tie_word_embeddings"] = json!(true));
    }

    /// #159: a value the parse accepts but the family row pins otherwise fires its row by
    /// name, and the gate refuses with the table (not with the not-built line)
    #[test]
    fn a_doctored_glm_value_fires_its_row_by_name() {
        let m = doctored_from(GLM_DIR, |c, _| {
            c["text_config"]["hidden_size"] = json!(5120);
            c["text_config"]["n_routed_experts"] = json!(256);
        })
        .unwrap();
        assert!(fired(&m, "hidden") && fired(&m, "experts"), "{:?}", m.verify().iter().map(Check::line).collect::<Vec<_>>());
        let err = verdict(&m).unwrap_err();
        assert!(err.starts_with("[meta] 2 of 38 constants differ"), "{err}");
        assert!(err.contains("hidden: pinned 4096 (glm5_next_text family row, zai-org/GLM-5.3-Flash @ eb9eb208"), "{err}");
        let m = doctored_from(GLM_DIR, |_, g| g["eos_token_id"] = json!([154820, 154827])).unwrap();
        assert!(fired(&m, "glm5_geo"), "two stop ids have no Glm5Geo form");
        assert!(m.verify()[0].config.contains("exactly three stop ids"), "{}", m.verify()[0].line());
    }

    /// #159: the Flash-Next fixture still passes its 21 checks into `Geo::FLASH_NEXT`, and the
    /// 27B its 20, with the glm5_next family beside them
    #[test]
    fn the_families_of_record_pass_the_gate_as_before() {
        let m = ModelMeta::from_config_files(&format!("{FN_FIXTURE}/config.json"), Some(&format!("{FN_FIXTURE}/generation_config.json"))).unwrap();
        assert_eq!((m.family, m.checks().len(), m.verify().len()), (Family::FlashNext, 21, 0));
        assert_eq!(verdict(&m), Ok(Geo::FLASH_NEXT));
        assert!(m.glm5_geo().is_err());
        assert_eq!((dense_meta().checks().len(), verdict(&dense_meta()).map(|g| g.family)), (20, Ok(Family::Qwen35Dense)));
    }
}
