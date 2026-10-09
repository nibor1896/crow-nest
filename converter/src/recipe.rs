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
//! | `Glm5Next` | `glm5_next_text` | `cnq4.5-glm5-next` | [`decide_glm5_next`]: GLM-5.3-Flash, FP8 originals (crow-nest #154/#155), a whitelist |
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
    Glm5Next,
}

impl Family {
    pub const ALL: [Family; 3] = [Family::FlashNext, Family::Qwen35Dense, Family::Glm5Next];

    pub fn name(self) -> &'static str {
        match self {
            Family::FlashNext => "FlashNext",
            Family::Qwen35Dense => "Qwen35Dense",
            Family::Glm5Next => "Glm5Next",
        }
    }

    pub fn model_type(self) -> &'static str {
        match self {
            Family::FlashNext => "qwen4_exp_text",
            Family::Qwen35Dense => "qwen3_5_text",
            Family::Glm5Next => "glm5_next_text",
        }
    }

    /// The `recipe` name the index v2 carries.
    pub fn recipe(self) -> &'static str {
        match self {
            Family::FlashNext => "cnq4.5-flash-next",
            Family::Qwen35Dense => "cnq4.5-qwen35-dense",
            Family::Glm5Next => "cnq4.5-glm5-next",
        }
    }

    /// #177: the largest ue4m3 sub-block scale byte the family's containers may carry. 0x7F
    /// decodes as 480 in this converter and the engine's scalar path, but it is the E4M3 NaN
    /// code (S.1111.111, arXiv:2209.05433 Table 1) and the mxf4nvf4 MMA instruction reads it as
    /// NaN (`engine/src/residency.rs` `sanitize_sf_slab`). A `cnq4.5-glm5-next` container stops
    /// at 0x7E (448); Flash-Next and the 27B keep 0x7F, so their containers stay byte-identical.
    pub fn scale_byte_max(self) -> u32 {
        match self {
            Family::FlashNext | Family::Qwen35Dense => 0x7F,
            Family::Glm5Next => 0x7E,
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
(known: qwen4_exp_text = Qwen3.8-Flash-Next, qwen3_5_text = dense Qwen3.5/3.8, glm5_next_text = GLM-5.3-Flash) - refusing (Crow #300 C6)"
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
    /// crow-nest #182: a GLM routed-expert projection stored as a MUL1 K = 3 trellis inside its
    /// expert's record (`mul1.rs`, `--experts-mul1`); only [`mul1_expert_decision`] gives it
    Mul1,
}

impl DtypeOut {
    pub fn as_str(self) -> &'static str {
        match self {
            DtypeOut::Nvfp4 => "nvfp4",
            DtypeOut::Bf16 => "bf16",
            DtypeOut::F32 => "f32",
            DtypeOut::I64 => "i64",
            DtypeOut::Mul1 => crate::mul1::DTYPE,
        }
    }

    /// Bytes this tensor takes in the container, the twin of `engine/src/cnq.rs::Cnq::byte_len`.
    /// `Mul1`: the trellis at K = 3 (3/8 B per value); the record adds the six fp16 scale vectors
    /// and the zeros to 4096, which the conversion counts per record (`mul1::RecordLayout::size`).
    pub fn bytes(self, n: usize) -> u64 {
        let n = n as u64;
        match self {
            DtypeOut::Nvfp4 => n.div_ceil(64) * 36,
            DtypeOut::Bf16 => n * 2,
            DtypeOut::F32 => n * 4,
            DtypeOut::I64 => n * 8,
            DtypeOut::Mul1 => n * 3 / 8,
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

/// Why the dense row omits the vision tower: the 27B sees images through the F16 projector.
pub const OMIT_VIT_DENSE: &str = "vision tower not written: the 27B sees images through the F16 projector mmproj-F16.gguf (Crow #300)";

/// Tensors `family`'s recipe does not write at all, with the reason the plan prints. The
/// manifest drops them before [`decide`], so they have no index record, no payload and no
/// section. The dense row omits `model.visual.*` (#300 decision 2026-09-26: the 27B reads
/// images through llama.cpp's F16 projector on the GPU, the container gets no vision part).
/// Flash-Next omits nothing: its `vit` section is part of the container of record.
pub fn omitted(family: Family, name: &str) -> Option<&'static str> {
    match family {
        Family::Qwen35Dense if section_of(name) == "vit" => Some(OMIT_VIT_DENSE),
        Family::Glm5Next if section_of(name) == "vit" => Some(OMIT_VIT_GLM),
        Family::Glm5Next if glm_parts(name).and_then(|p| p.0).is_some_and(|l| l >= GLM5_NEXT_TEXT_LAYERS) => Some(OMIT_MTP_GLM),
        _ => None,
    }
}

/// The decision for one tensor under `family`'s recipe. `src_dtype` is the safetensors dtype
/// (`BF16`, `F32`, `F16`, `I64`).
pub fn decide(family: Family, name: &str, shape: &[usize], src_dtype: &str) -> Result<Decision, String> {
    let d = match family {
        Family::FlashNext => decide_flash_next(name, shape, src_dtype),
        Family::Qwen35Dense => decide_qwen35_dense(name, shape, src_dtype)?,
        Family::Glm5Next => decide_glm5_next(name, shape, src_dtype)?,
    };
    check_decision(family, name, shape, src_dtype, d)
}

/// The checks every decision passes, whichever row made it: what the write path can carry.
fn check_decision(family: Family, name: &str, shape: &[usize], src_dtype: &str, d: Decision) -> Result<Decision, String> {
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
    if d.dtype != DtypeOut::Nvfp4 && src_dtype == "F8_E4M3" {
        return Err(format!("{name}: an F8_E4M3 source is only read through its weight_scale_inv into NVFP4; the {} recipe keeps it {} - refusing", family.recipe(), d.dtype.as_str()));
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
/// - `mtp`: its own section, every tensor BF16 (the published NVIDIA NVFP4 27B keeps MTP BF16),
///   optional to load, so it costs GPU memory only on the cards that load it;
/// - `vit`: NOT WRITTEN, see [`omitted`]. It never reaches this row; if it does, it is refused.
///
/// A text tensor matching none of the rows is refused by name.
pub fn decide_qwen35_dense(name: &str, shape: &[usize], src_dtype: &str) -> Result<Decision, String> {
    let section = section_of(name);
    let d = |dtype, rule| Ok(Decision { dtype, section, rule });
    if src_dtype == "I64" {
        return Err(format!("{name}: an I64 tensor in a dense checkpoint - the dense recipe has no row for it"));
    }
    match section {
        "vit" => return Err(format!("{name}: the dense recipe omits the vision tower (`recipe::omitted`), it never decides it")),
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
// GLM-5.3-Flash (crow-nest #154 step 4, #155 step 5; plan of record: vault
// `glm-5-3-flash-laeuft-nur-auf-crow-nest-...`, PREREG `runs/glm53-flash/PREREG.md`)
// ---------------------------------------------------------------------------------------------

/// The text layers the `cnq4.5-glm5-next` row was written for (`num_hidden_layers` of revision
/// `eb9eb208`). Layer 45 of the checkpoint is the MTP block (`num_nextn_predict_layers` 1; HF
/// ignores `layers.45.` on load, `modeling_glm5_next.py:1359`). [`check_family_config`] refuses a
/// config with another count, so this constant cannot silently disagree with the checkpoint.
pub const GLM5_NEXT_TEXT_LAYERS: u64 = 45;

/// Plan of record, "Was bewusst NICHT gebaut wird": no vision tower in v1.
pub const OMIT_VIT_GLM: &str = "vision tower not converted in v1 (GLM-5.3-Flash plan: no vision tower and no MTP in v1)";
/// Plan of record, same line: the MTP block (layer 45: its experts, attention, eh_proj, enorm,
/// hnorm, shared_head) is not converted in v1; step 21 would add it on its own trigger.
pub const OMIT_MTP_GLM: &str = "MTP block (layer 45) not converted in v1 (GLM-5.3-Flash plan: no vision tower and no MTP in v1; step 21)";

/// `(layer, the name below the layer)` of a GLM text tensor: `model.language_model.layers.N.x`
/// -> `(Some(N), "x")`, `model.language_model.x` -> `(None, "x")`, `lm_head.weight` -> `(None,
/// "lm_head.weight")`. `None` for anything else (the vision tower).
pub fn glm_parts(name: &str) -> Option<(Option<u64>, &str)> {
    if name == "lm_head.weight" {
        return Some((None, name));
    }
    let rest = name.strip_prefix("model.language_model.")?;
    if let Some(r) = rest.strip_prefix("layers.") {
        let (l, x) = r.split_once('.')?;
        return Some((Some(l.parse().ok()?), x));
    }
    Some((None, rest))
}

/// `(layer, expert, projection)` of a routed-expert weight `...layers.L.mlp.experts.E.P_proj.weight`.
pub fn glm_expert(name: &str) -> Option<(u64, u64, &str)> {
    let (l, r) = glm_parts(name)?;
    let r = r.strip_prefix("mlp.experts.")?;
    let (e, p) = r.split_once('.')?;
    let p = p.strip_suffix("_proj.weight")?;
    if !matches!(p, "gate" | "up" | "down") {
        return None;
    }
    Some((l?, e.parse().ok()?, p))
}

/// crow-nest #182: the decision for a routed-expert weight of GLM-5.3-Flash when the experts go
/// to MUL1 trellis records (`--experts-mul1`): the trunk's experts (layers 3-44) in `text`, the
/// MTP block's (layer 45, which [`omitted`] otherwise drops whole) in `mtp`. The rest of the MTP
/// block stays omitted, every other tensor keeps its `cnq4.5-glm5-next` decision. `None` for a
/// name that is not a routed-expert projection.
pub fn mul1_expert_decision(name: &str) -> Option<Decision> {
    let (l, _, _) = glm_expert(name)?;
    let (section, rule) = if l >= GLM5_NEXT_TEXT_LAYERS {
        ("mtp", "MTP routed expert gate/up/down MUL1 K=3 trellis (exllamav3 quantizer, #182)")
    } else {
        ("text", "routed expert gate/up/down MUL1 K=3 trellis (exllamav3 quantizer, #182)")
    };
    Some(Decision { dtype: DtypeOut::Mul1, section, rule })
}

/// The section of an MTP overlay (#182): the MTP block's non-expert tensors, the same section the
/// 3-bit container gives the block's routed-expert records ([`mul1_expert_decision`]).
pub const MTP_SECTION: &str = "mtp";

/// crow-nest #182: the decision for a non-expert tensor of GLM-5.3-Flash's MTP block (layer 45) in
/// an MTP overlay (`converter --mtp-overlay`): the codec the trunk's DSA + MoE layers carry for the
/// same tensor ([`glm5_layer_row`]: NVFP4 q_a/q_b/kv_a/kv_b/o_proj and shared expert, BF16 norms,
/// indexer and router, F32 score bias), in section `mtp`. `eh_proj` has no trunk twin and stays
/// BF16 (llama.cpp keeps `nextn.eh_proj` at Q8_0 or above, `docs/glm5-next-recipe.md` D10).
/// `None` for a name that is not a non-expert tensor of layer 45 (the routed experts are the 3-bit
/// container's MUL1 records); the same checks as [`decide`] run on the result.
pub fn mtp_overlay_decision(name: &str, shape: &[usize], src_dtype: &str) -> Option<Result<Decision, String>> {
    if name.ends_with("_scale_inv") || glm_expert(name).is_some() {
        return None;
    }
    let (layer, r) = glm_parts(name)?;
    if layer != Some(GLM5_NEXT_TEXT_LAYERS) {
        return None;
    }
    let d = if r == "eh_proj.weight" {
        Ok(Decision { dtype: DtypeOut::Bf16, section: MTP_SECTION, rule: "MTP eh_proj BF16 (no trunk twin; llama.cpp keeps nextn.eh_proj >= Q8_0, D10)" })
    } else {
        glm5_layer_row(name, layer, r, shape, src_dtype, MTP_SECTION)
    };
    Some(d.and_then(|d| check_decision(Family::Glm5Next, name, shape, src_dtype, d)))
}

/// The GLM-5.3-Flash row, a WHITELIST. Keep set = the plan's step 4 (PREREG "Fixed for the
/// whole series"): embeddings, `lm_head`, router, `e_score_correction_bias`, norms, 1-D, mHC
/// `hc_*`, indexer, KDA gates, anything not a whole 64-value block. Keeps carry the source
/// dtype (BF16 as BF16, F32 as F32: `e_score_correction_bias`, `A_log`, `dt_bias`, `hc_*_base`,
/// `hc_*_scale` are F32 in the checkpoint). Everything else is NVFP4: routed and shared experts,
/// the dense MLP of layers 0-2, MLA attention, KDA q/k/v/o and the KDA short convolutions.
/// The rule names say which of these had a BF16 source (the checkpoint's own
/// `modules_to_not_convert`) so the plan's table shows what that costs. Vision and the MTP
/// block never reach this row ([`omitted`]); `weight_scale_inv` never does either (it is read
/// with its weight). Any other tensor is refused by name.
pub fn decide_glm5_next(name: &str, shape: &[usize], src_dtype: &str) -> Result<Decision, String> {
    let refuse = |why: &str| -> Result<Decision, String> {
        Err(format!(
            "{name} {shape:?} {src_dtype}: {why} - refusing (the {} row is a whitelist; add a row with a reason, do not let it fall through)",
            Family::Glm5Next.recipe()
        ))
    };
    if name.ends_with("_scale_inv") {
        return refuse("a weight_scale_inv is read with its FP8 weight, never decided on its own");
    }
    let Some((layer, r)) = glm_parts(name) else {
        return refuse("not a text tensor of GLM-5.3-Flash (vision and MTP are omitted before the row)");
    };
    if layer.is_some_and(|l| l >= GLM5_NEXT_TEXT_LAYERS) {
        return refuse("an MTP-block tensor (omitted in v1)");
    }
    glm5_layer_row(name, layer, r, shape, src_dtype, "text")
}

/// The body of the GLM-5.3-Flash row below the layer check: one tensor `r` of layer `layer` (None:
/// a model-level tensor), written to `section`. [`decide_glm5_next`] passes `text`; the MTP
/// overlay ([`mtp_overlay_decision`], #182) passes `mtp`, so the MTP block's attention, router and
/// shared expert get exactly the codec the trunk's DSA + MoE layers carry.
fn glm5_layer_row(name: &str, layer: Option<u64>, r: &str, shape: &[usize], src_dtype: &str, section: &'static str) -> Result<Decision, String> {
    let refuse = |why: &str| -> Result<Decision, String> {
        Err(format!(
            "{name} {shape:?} {src_dtype}: {why} - refusing (the {} row is a whitelist; add a row with a reason, do not let it fall through)",
            Family::Glm5Next.recipe()
        ))
    };
    // a keep is bf16: `decide` refuses a non-BF16 source under that label (a norm in F32 would be
    // a surprise worth a refusal). Only the tensors the checkpoint stores in F32 on purpose
    // (HF `_keep_in_fp32_modules_strict`, `modeling_glm5_next.py:1358`, plus the mHC base/scale)
    // are carried in their source dtype.
    let keep = |rule: &'static str| -> Result<Decision, String> { Ok(Decision { dtype: DtypeOut::Bf16, section, rule }) };
    let carry = |rule: &'static str| -> Result<Decision, String> {
        let dtype = if src_dtype == "F32" { DtypeOut::F32 } else { DtypeOut::Bf16 };
        Ok(Decision { dtype, section, rule })
    };
    let nv = |rule: &'static str| -> Result<Decision, String> { Ok(Decision { dtype: DtypeOut::Nvfp4, section, rule }) };
    let n: usize = shape.iter().product();
    match (layer, r) {
        (None, "embed_tokens.weight") => return keep("token embedding BF16 (host RAM)"),
        (None, "lm_head.weight") => return keep("lm_head BF16"),
        (None, "norm.weight") => return keep("norm BF16"),
        (None, _) => return refuse("matches no row of the model-level tensors"),
        _ => {}
    }
    if r.starts_with("hc_") {
        return carry("mHC hc_* keep (fn BF16, base/scale F32)");
    }
    if r.contains("norm.") {
        return keep("norm BF16");
    }
    if r == "mlp.gate.weight" {
        return keep("router BF16");
    }
    if r == "mlp.gate.e_score_correction_bias" {
        return carry("e_score_correction_bias f32 carry");
    }
    if r.starts_with("self_attn.indexer.") {
        return keep("DSA indexer keep (BF16)");
    }
    if matches!(r, "self_attn.f_a_proj.weight" | "self_attn.f_b_proj.weight" | "self_attn.g_a_proj.weight" | "self_attn.g_b_proj.weight" | "self_attn.b_proj.weight") {
        return keep("KDA gates f_a/f_b/g_a/g_b/b_proj BF16");
    }
    if matches!(r, "self_attn.A_log" | "self_attn.dt_bias") {
        return carry("KDA A_log/dt_bias f32 carry");
    }
    if shape.len() == 1 {
        return keep("1-D keep");
    }
    if n % 64 != 0 || n < 64 {
        return keep("not a whole 64-value block");
    }
    if glm_expert(name).is_some() {
        return nv("routed expert gate/up/down NVFP4 (FP8 source)");
    }
    if matches!(r, "mlp.shared_experts.gate_proj.weight" | "mlp.shared_experts.up_proj.weight" | "mlp.shared_experts.down_proj.weight") {
        return nv("shared expert NVFP4 (FP8 source)");
    }
    if matches!(r, "mlp.gate_proj.weight" | "mlp.up_proj.weight" | "mlp.down_proj.weight") {
        return nv("dense MLP layers 0-2 NVFP4 (FP8 source)");
    }
    if matches!(r, "self_attn.q_a_proj.weight" | "self_attn.q_b_proj.weight" | "self_attn.kv_a_proj_with_mqa.weight") {
        return nv("MLA q_a/q_b/kv_a NVFP4 (FP8 source)");
    }
    if r == "self_attn.kv_b_proj.weight" {
        return nv("MLA kv_b NVFP4 (BF16 source)");
    }
    if r == "self_attn.o_proj.weight" {
        return if src_dtype == "F8_E4M3" { nv("MLA o_proj NVFP4 (FP8 source)") } else { nv("KDA o_proj NVFP4 (BF16 source)") };
    }
    if matches!(r, "self_attn.q_proj.weight" | "self_attn.k_proj.weight" | "self_attn.v_proj.weight") {
        return nv("KDA q/k/v NVFP4 (BF16 source)");
    }
    if matches!(r, "self_attn.q_conv1d.weight" | "self_attn.k_conv1d.weight" | "self_attn.v_conv1d.weight") {
        return nv("KDA short conv q/k/v_conv1d NVFP4 (BF16 source, not in the step-4 keep set)");
    }
    refuse("matches no row")
}

/// The tensor class of the code histograms (#155 (b)) and the layer and expert it belongs to.
/// GLM classes: `expert_gate`, `expert_up`, `expert_down`, `shared_expert`, `attn_mla`,
/// `attn_kda`, `dense_mlp`, `indexer`, `rest`; KDA vs MLA from `text_config.layer_types`.
/// The other families get the same names where the tensor names allow (`expert` for a fused
/// expert tensor, `attn`, `attn_linear`), so their sidecar carries a histogram too.
pub fn tensor_class(family: Family, name: &str, config: &serde_json::Value) -> (&'static str, Option<u64>, Option<u64>) {
    let layer = name.split("layers.").nth(1).and_then(|r| r.split('.').next()).and_then(|s| s.parse::<u64>().ok());
    if family == Family::Glm5Next {
        if let Some((l, e, p)) = glm_expert(name) {
            let c = match p {
                "gate" => "expert_gate",
                "up" => "expert_up",
                _ => "expert_down",
            };
            return (c, Some(l), Some(e));
        }
        let kda = layer.and_then(|l| config["text_config"]["layer_types"][l as usize].as_str()) == Some("linear_attention");
        let c = if name.contains(".mlp.shared_experts.") {
            "shared_expert"
        } else if name.contains(".self_attn.indexer.") {
            "indexer"
        } else if name.contains(".self_attn.") {
            if kda { "attn_kda" } else { "attn_mla" }
        } else if name.contains(".mlp.") && !name.contains(".mlp.gate.") {
            "dense_mlp"
        } else {
            "rest"
        };
        return (c, layer, None);
    }
    let c = if name.contains(".mlp.experts.") {
        "expert"
    } else if name.contains("shared_expert") {
        "shared_expert"
    } else if name.contains(".linear_attn.") {
        "attn_linear"
    } else if name.contains(".self_attn.") {
        "attn"
    } else if name.contains(".mlp.") {
        "dense_mlp"
    } else {
        "rest"
    };
    (c, layer, None)
}

/// Family-specific checks of `config.json` that the recipe row relies on. GLM: 45 text layers
/// + 1 MTP layer ([`GLM5_NEXT_TEXT_LAYERS`]) and FP8 E4M3 with 128x128 block scales (what
/// `fp8.rs` decodes). The other families: nothing.
pub fn check_family_config(family: Family, config: &serde_json::Value) -> Result<(), String> {
    if family != Family::Glm5Next {
        return Ok(());
    }
    let t = &config["text_config"];
    if t["num_hidden_layers"].as_u64() != Some(GLM5_NEXT_TEXT_LAYERS) || t["num_nextn_predict_layers"].as_u64() != Some(1) {
        return Err(format!(
            "config.json: num_hidden_layers {} / num_nextn_predict_layers {}, the {} row was written for 45 + 1 (GLM-5.3-Flash rev eb9eb208)",
            t["num_hidden_layers"],
            t["num_nextn_predict_layers"],
            family.recipe()
        ));
    }
    let q = &config["quantization_config"];
    if q["quant_method"] != "fp8" || q["fmt"] != "e4m3" || q["weight_block_size"] != serde_json::json!([128, 128]) {
        let mut q = q.clone();
        if let Some(o) = q.as_object_mut() {
            o.remove("modules_to_not_convert");
        }
        return Err(format!("config.json: quantization_config {q} - the converter reads FP8 e4m3 with 128x128 block scales only"));
    }
    Ok(())
}

/// The Hugging Face model-info JSON saved beside the originals (`hf-revision.json`: the API's
/// `/api/models/<repo>/revision/<rev>` with `sha` and `siblings[].lfs.sha256/size`) or a tree
/// listing (`hf-tree.json`: `[{path, size, lfs: {oid, size}}]`, no revision). Per LFS file:
/// (sha256, size). `None` when neither file is there.
pub struct HfApiInfo {
    pub revision: Option<String>,
    pub lfs: BTreeMap<String, (String, u64)>,
    pub source: &'static str,
}

pub fn read_hf_api_info(model_dir: &Path) -> Result<Option<HfApiInfo>, String> {
    let rd = |f: &str| -> Result<Option<serde_json::Value>, String> {
        let p = model_dir.join(f);
        match std::fs::read(&p) {
            Ok(b) => serde_json::from_slice(&b).map(Some).map_err(|e| format!("{}: {e}", p.display())),
            Err(_) => Ok(None),
        }
    };
    if let Some(v) = rd("hf-revision.json")? {
        let mut lfs = BTreeMap::new();
        for s in v["siblings"].as_array().ok_or("hf-revision.json: no siblings[]")? {
            if let (Some(f), Some(sha), Some(size)) = (s["rfilename"].as_str(), s["lfs"]["sha256"].as_str(), s["lfs"]["size"].as_u64()) {
                lfs.insert(f.to_string(), (sha.to_string(), size));
            }
        }
        return Ok(Some(HfApiInfo { revision: v["sha"].as_str().map(String::from), lfs, source: "hf-revision.json" }));
    }
    if let Some(v) = rd("hf-tree.json")? {
        let mut lfs = BTreeMap::new();
        for s in v.as_array().ok_or("hf-tree.json: not a tree listing (array)")? {
            if let (Some(f), Some(sha), Some(size)) = (s["path"].as_str(), s["lfs"]["oid"].as_str(), s["lfs"]["size"].as_u64()) {
                lfs.insert(f.to_string(), (sha.to_string(), size));
            }
        }
        return Ok(Some(HfApiInfo { revision: None, lfs, source: "hf-tree.json" }));
    }
    Ok(None)
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
        Family::Glm5Next => {
            let o = g.as_object_mut().unwrap();
            for k in ["gdn_layers", "gdn_k_heads", "gdn_v_heads", "gdn_k_dim", "gdn_v_dim", "conv_kernel"] {
                o.remove(k);
            }
            g["attn_layers"] = serde_json::json!(layer_types.iter().filter(|s| **s == "deepseek_sparse_attention").count());
            g["kda_layers"] = serde_json::json!(layer_types.iter().filter(|s| **s == "linear_attention").count());
            g["kda_heads"] = t["linear_attn_config"]["num_heads"].clone();
            g["kda_head_dim"] = t["linear_attn_config"]["head_dim"].clone();
            g["conv_kernel"] = t["linear_attn_config"]["short_conv_kernel_size"].clone();
            g["mtp_layers"] = t["num_nextn_predict_layers"].clone();
            g["ffn"] = serde_json::json!("moe");
            g["dense_layers"] = t["first_k_dense_replace"].clone();
            g["inter"] = t["intermediate_size"].clone();
            g["experts"] = t["n_routed_experts"].clone();
            g["experts_per_tok"] = t["num_experts_per_tok"].clone();
            g["moe_inter"] = t["moe_intermediate_size"].clone();
            g["shared_experts"] = t["n_shared_experts"].clone();
            g["kv_lora_rank"] = t["kv_lora_rank"].clone();
            g["q_lora_rank"] = t["q_lora_rank"].clone();
            g["hc_mult"] = t["hc_mult"].clone();
            g["index_topk"] = t["index_topk"].clone();
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
            assert_eq!(omitted(Family::FlashNext, name), None, "{name}: Flash-Next omits nothing");
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

    /// Every tensor of the 27B (names and shapes from its 18 shard headers) is either omitted
    /// (the 333 vision-tower tensors, and only those) or decided by a named row of the dense
    /// recipe; none is refused, and the per-row counts are the recipe.
    #[test]
    fn the_dense_row_decides_every_tensor_of_the_27b() {
        let tsv = include_str!("../tests/fixtures/qwen3.8-27b-tensors.tsv");
        let mut per: BTreeMap<(&str, &str, &str), usize> = BTreeMap::new();
        let mut n = 0;
        let mut omit = 0;
        for line in tsv.lines().filter(|l| !l.starts_with('#')) {
            let f: Vec<&str> = line.split('\t').collect();
            n += 1;
            if let Some(why) = omitted(Family::Qwen35Dense, f[0]) {
                assert_eq!(why, OMIT_VIT_DENSE);
                assert!(decide(Family::Qwen35Dense, f[0], &shape(f[1]), f[2]).unwrap_err().contains("omits the vision tower"));
                omit += 1;
                continue;
            }
            let d = decide(Family::Qwen35Dense, f[0], &shape(f[1]), f[2]).unwrap_or_else(|e| panic!("{e}"));
            *per.entry((d.section, d.dtype.as_str(), d.rule)).or_default() += 1;
        }
        assert_eq!(n, 1199);
        assert_eq!(omit, 27 * 12 + 6 + 3, "the whole vision tower, and nothing else, is omitted");
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

    /// The GLM-5.3-Flash tensor table (crow-nest #154), expanded: (name, dtype, shape, shard,
    /// bytes) per tensor, plus the shard records (data_start, size).
    fn glm53_table() -> (Vec<(String, String, Vec<usize>, u32, u64)>, Vec<(u64, u64)>) {
        let tsv = include_str!("../tests/fixtures/glm53-flash-tensors.tsv");
        let (mut rows, mut shards) = (Vec::new(), Vec::new());
        for line in tsv.lines() {
            let f: Vec<&str> = line.split('\t').collect();
            if f[0] == "# shard" {
                shards.push((f[2].parse().unwrap(), f[3].parse().unwrap()));
                continue;
            }
            if line.starts_with('#') {
                continue;
            }
            let (dt, sh, nr, b) = (f[1].to_string(), shape(f[2]), f[3].parse().unwrap(), f[4].parse().unwrap());
            match f[0].split_once('{') {
                Some((pre, rest)) => {
                    let (range, post) = rest.split_once('}').unwrap();
                    let (a, z) = range.split_once("..").unwrap();
                    for e in a.parse::<u32>().unwrap()..=z.parse().unwrap() {
                        rows.push((format!("{pre}{e}{post}"), dt.clone(), sh.clone(), nr, b));
                    }
                }
                None => rows.push((f[0].to_string(), dt, sh, nr, b)),
            }
        }
        (rows, shards)
    }

    /// Step 4's abort criterion on the real headers: 76,108 tensors whose bytes plus the 62
    /// headers are the 328,337,455,672 B of the shards; every tensor is omitted (vision, MTP),
    /// a block scale paired with an FP8 weight (128x128 grid), or decided by a named row of the
    /// whitelist, never refused; the per-row counts are the recipe.
    #[test]
    fn the_glm_row_decides_every_tensor_of_glm53_flash() {
        let (rows, shards) = glm53_table();
        assert_eq!(rows.len(), 76_108);
        assert_eq!(shards.len(), 62);
        let tensor_bytes: u64 = rows.iter().map(|r| r.4).sum();
        let header_bytes: u64 = shards.iter().map(|s| s.0).sum();
        assert_eq!(tensor_bytes + header_bytes, 328_337_455_672);
        assert_eq!(shards.iter().map(|s| s.1).sum::<u64>(), 328_337_455_672);
        let by_name: BTreeMap<&str, &(String, String, Vec<usize>, u32, u64)> = rows.iter().map(|r| (r.0.as_str(), r)).collect();
        let mut per: BTreeMap<(&str, &str), usize> = BTreeMap::new();
        let (mut omit_vit, mut omit_mtp, mut scales) = (0, 0, 0);
        for (name, dt, sh, _, _) in &rows {
            match omitted(Family::Glm5Next, name) {
                Some(OMIT_VIT_GLM) => {
                    omit_vit += 1;
                    continue;
                }
                Some(OMIT_MTP_GLM) => {
                    omit_mtp += 1;
                    continue;
                }
                Some(other) => panic!("{name}: {other}"),
                None => {}
            }
            if let Some(w) = name.strip_suffix("_scale_inv") {
                let wr = by_name.get(w).unwrap_or_else(|| panic!("{name}: no weight"));
                assert_eq!((wr.1.as_str(), dt.as_str()), ("F8_E4M3", "F32"), "{name}");
                assert_eq!(sh.as_slice(), [wr.2[0].div_ceil(128), wr.2[1].div_ceil(128)], "{name}: grid");
                scales += 1;
                continue;
            }
            if dt == "F8_E4M3" {
                assert!(by_name.contains_key(format!("{name}_scale_inv").as_str()), "{name}: FP8 without its scale");
            }
            let d = decide(Family::Glm5Next, name, sh, dt).unwrap_or_else(|e| panic!("{e}"));
            assert_eq!(d.section, "text", "{name}");
            *per.entry((d.dtype.as_str(), d.rule)).or_default() += 1;
        }
        assert_eq!((omit_vit, omit_mtp, scales), (347, 1760, 36_467));
        let want = BTreeMap::from([
            (("bf16", "DSA indexer keep (BF16)"), 55),
            (("bf16", "KDA gates f_a/f_b/g_a/g_b/b_proj BF16"), 170),
            (("bf16", "lm_head BF16"), 1),
            (("bf16", "mHC hc_* keep (fn BF16, base/scale F32)"), 90),
            (("bf16", "norm BF16"), 169),
            (("bf16", "router BF16"), 42),
            (("bf16", "token embedding BF16 (host RAM)"), 1),
            (("f32", "KDA A_log/dt_bias f32 carry"), 68),
            (("f32", "e_score_correction_bias f32 carry"), 42),
            (("f32", "mHC hc_* keep (fn BF16, base/scale F32)"), 180),
            (("nvfp4", "KDA o_proj NVFP4 (BF16 source)"), 34),
            (("nvfp4", "KDA q/k/v NVFP4 (BF16 source)"), 102),
            (("nvfp4", "KDA short conv q/k/v_conv1d NVFP4 (BF16 source, not in the step-4 keep set)"), 102),
            (("nvfp4", "MLA kv_b NVFP4 (BF16 source)"), 11),
            (("nvfp4", "MLA o_proj NVFP4 (FP8 source)"), 11),
            (("nvfp4", "MLA q_a/q_b/kv_a NVFP4 (FP8 source)"), 33),
            (("nvfp4", "dense MLP layers 0-2 NVFP4 (FP8 source)"), 9),
            (("nvfp4", "routed expert gate/up/down NVFP4 (FP8 source)"), 36_288),
            (("nvfp4", "shared expert NVFP4 (FP8 source)"), 126),
        ]);
        assert_eq!(per, want);
    }

    /// #182: the MTP overlay row on the real tensor table. Exactly the 25 non-expert tensors of
    /// layer 45 get a decision (section `mtp`), never a scale or a routed expert; every one with a
    /// twin in the trunk's DSA + MoE layer 7 gets the twin's codec and rule; `eh_proj` is BF16.
    #[test]
    fn the_mtp_overlay_row_gives_layer_45_the_trunks_dsa_moe_codecs() {
        let (rows, _) = glm53_table();
        let by_name: BTreeMap<&str, &(String, String, Vec<usize>, u32, u64)> = rows.iter().map(|r| (r.0.as_str(), r)).collect();
        let mut per: BTreeMap<&str, usize> = BTreeMap::new();
        let mut twins = 0;
        for (name, dt, sh, _, _) in &rows {
            let Some(d) = mtp_overlay_decision(name, sh, dt) else {
                assert!(!(name.contains(".layers.45.") && !name.ends_with("_scale_inv") && glm_expert(name).is_none()), "{name}: no MTP overlay decision");
                continue;
            };
            let d = d.unwrap_or_else(|e| panic!("{e}"));
            assert!(name.starts_with("model.language_model.layers.45.") && glm_expert(name).is_none(), "{name}");
            assert_eq!(d.section, MTP_SECTION, "{name}");
            *per.entry(d.dtype.as_str()).or_default() += 1;
            let twin = name.replace(".layers.45.", ".layers.7.");
            match by_name.get(twin.as_str()) {
                Some(t) => {
                    let td = decide(Family::Glm5Next, &t.0, &t.2, &t.1).unwrap();
                    assert_eq!((d.dtype, d.rule), (td.dtype, td.rule), "{name} vs its trunk twin {twin}");
                    twins += 1;
                }
                None => assert!(["eh_proj.weight", "enorm.weight", "hnorm.weight", "shared_head.norm.weight"].iter().any(|s| name.ends_with(s)), "{name}: no twin in layer 7"),
            }
        }
        assert_eq!(per, BTreeMap::from([("bf16", 16), ("f32", 1), ("nvfp4", 8)]));
        assert_eq!(twins, 21);
        let eh = mtp_overlay_decision("model.language_model.layers.45.eh_proj.weight", &[4096, 8192], "BF16").unwrap().unwrap();
        assert_eq!(eh.dtype, DtypeOut::Bf16);
        // an F8 eh_proj would be refused by the write-path checks, not written under a bf16 label
        assert!(mtp_overlay_decision("model.language_model.layers.45.eh_proj.weight", &[4096, 8192], "F8_E4M3").unwrap().is_err());
        // the trunk's row is unchanged: layer 45 is still refused there
        assert!(decide(Family::Glm5Next, "model.language_model.layers.45.eh_proj.weight", &[4096, 8192], "BF16").is_err());
    }

    /// The GLM row is a whitelist: a tensor it has no row for is refused BY NAME, and so are a
    /// stray block scale, an FP8 weight it would keep, and a model-level tensor it does not know.
    #[test]
    fn the_glm_row_refuses_an_unknown_tensor_by_name() {
        let m = decide(Family::Glm5Next, "model.language_model.layers.7.self_attn.mystery_proj.weight", &[128, 128], "BF16").unwrap_err();
        assert!(m.contains("model.language_model.layers.7.self_attn.mystery_proj.weight") && m.contains("whitelist"), "{m}");
        let m = decide(Family::Glm5Next, "model.language_model.mystery.weight", &[128, 128], "BF16").unwrap_err();
        assert!(m.contains("model.language_model.mystery.weight"), "{m}");
        let m = decide(Family::Glm5Next, "model.language_model.layers.3.mlp.gate_proj.weight_scale_inv", &[1, 1], "F32").unwrap_err();
        assert!(m.contains("read with its FP8 weight"), "{m}");
        let m = decide(Family::Glm5Next, "model.language_model.layers.3.mlp.gate.weight", &[288, 4096], "F8_E4M3").unwrap_err();
        assert!(m.contains("F8_E4M3"), "{m}");
        let m = decide(Family::Glm5Next, "model.language_model.layers.45.mlp.experts.0.up_proj.weight", &[2048, 4096], "F8_E4M3").unwrap_err();
        assert!(m.contains("MTP"), "{m}");
        assert_eq!(omitted(Family::Glm5Next, "model.language_model.layers.45.eh_proj.weight"), Some(OMIT_MTP_GLM));
        assert_eq!(omitted(Family::Glm5Next, "model.language_model.layers.44.eh_proj.weight"), None);
        assert_eq!(omitted(Family::Glm5Next, "model.visual.blocks.0.attn.qkv.weight"), Some(OMIT_VIT_GLM));
        assert_eq!(omitted(Family::FlashNext, "model.language_model.layers.45.mlp.experts.gate_up_proj"), None);
    }

    /// #177: only the glm5_next recipe caps the scale byte below the E4M3 NaN code
    #[test]
    fn only_the_glm_recipe_caps_the_scale_byte_at_0x7e() {
        assert_eq!(Family::Glm5Next.scale_byte_max(), 0x7E);
        assert_eq!(Family::FlashNext.scale_byte_max(), 0x7F);
        assert_eq!(Family::Qwen35Dense.scale_byte_max(), 0x7F);
    }

    #[test]
    fn the_glm_config_is_checked_before_the_row_is_trusted() {
        let mut c = serde_json::json!({
            "text_config": { "model_type": "glm5_next_text", "num_hidden_layers": 45, "num_nextn_predict_layers": 1 },
            "quantization_config": { "quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128] },
        });
        assert_eq!(Family::detect(&c).unwrap(), Family::Glm5Next);
        check_family_config(Family::Glm5Next, &c).unwrap();
        c["quantization_config"]["weight_block_size"] = serde_json::json!([1, 32]);
        assert!(check_family_config(Family::Glm5Next, &c).unwrap_err().contains("128x128"));
        c["quantization_config"]["weight_block_size"] = serde_json::json!([128, 128]);
        c["text_config"]["num_hidden_layers"] = serde_json::json!(46);
        assert!(check_family_config(Family::Glm5Next, &c).unwrap_err().contains("45 + 1"));
        check_family_config(Family::FlashNext, &c).unwrap();
    }
}
