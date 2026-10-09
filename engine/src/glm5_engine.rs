//! #185 (GLM-5.3-Flash plan steps 20/22), part 1 of 2: the glm5_next engine behind `bin/serve`.
//!
//! `serve` dispatches by the container's family before it maps the container
//! (`bin/serve.rs`, `engine_kind`): Flash-Next and the 27B boot `gen::Engine` exactly as
//! before, and `glm5_next` boots this type. Its request layer is wired in `bin/serve.rs`
//! (the five items of `docs/glm5-tokenizer.md` "What serve still needs"): the template
//! variables, the 400s for `enable_thinking:false` and the refused reasoning words, the GLM
//! tool markup in the parser and the grammar, the reasoning filter starting `Inside`, and the
//! stop ids [`glm5_template::EOS_IDS`].
//!
//! What this type does NOT have yet is a body. The token generator it wraps (`glm5_model`
//! with the expert tiers: the dynamic expert cache of #175 and the NVMe expert tier and 200k
//! boot of #149) is being built and is not merged, so [`Glm5Engine::boot`] refuses by name,
//! before the container is mapped and before any CUDA call. The type holds an
//! [`std::convert::Infallible`], so no value of it exists and every method that would need
//! the generator is statically unreachable. Part 2 replaces that field with the generator and
//! gives the methods their bodies: boot (the #159 plan at 200k), prefill a chunk, decode one
//! step, the logits row, snapshot / restore at a prefix length (KDA recurrent + conv states;
//! MLA latent and DSA indexer rows truncate), reset.

use crate::geo::Family;
use crate::glm5_template;
use crate::meta::ModelMeta;
use crate::toolcall::Markup;

/// the words of the boot refusal that name what is missing; the tests and the docs quote them
pub const NEEDS: &str = "needs the expert tiers of #175/#149";

/// - #185: the glm5_next engine `serve` boots for a GLM-5.3-Flash container
/// - part 1: no value exists (`never`), so [`Glm5Engine::boot`] is the only door and it refuses
pub struct Glm5Engine {
    never: std::convert::Infallible,
}

impl Glm5Engine {
    /// - #185: boot the glm5_next engine for the container `cnq_path` whose config is `meta`
    /// - part 1: refuses by name, naming what part 2 needs; a config of another family is
    ///   refused too (the dispatch in `serve` never sends one here)
    /// - pure: no container mapping, no CUDA call, no log line (the caller logs the reason)
    pub fn boot(cnq_path: &str, meta: &ModelMeta) -> Result<Glm5Engine, String> {
        if meta.family != Family::Glm5Next {
            return Err(format!(
                "[glm5] {cnq_path}: family {:?} ({}) is not glm5_next - Glm5Engine boots GLM-5.3-Flash only",
                meta.family, meta.config_path
            ));
        }
        Err(format!(
            "[glm5] {cnq_path}: serve's glm5_next request layer is wired (#185 part 1), its engine is not: \
             Glm5Engine::boot {NEEDS} (the dynamic expert cache, the NVMe expert tier and the 200k boot; \
             config {}) - refusing to boot",
            meta.config_path
        ))
    }

    /// - part 1: the proof that no value exists; `serve`'s dispatch matches on it after a
    ///   boot that cannot succeed, so the `Ok` arm compiles without a request loop
    /// - part 2 deletes it together with the `never` field
    pub fn placeholder(self) -> std::convert::Infallible {
        self.never
    }

    /// the context the boot allocated
    pub fn n_ctx(&self) -> usize {
        match self.never {}
    }

    /// the vocabulary size of the loaded head (`Glm5Geo::vocab`)
    pub fn vocab(&self) -> usize {
        match self.never {}
    }

    /// the three end-of-turn ids of GLM-5.3-Flash (`generation_config.json` `eos_token_id`)
    pub fn stop_ids(&self) -> &'static [u32] {
        &glm5_template::EOS_IDS
    }

    /// the tool-call markup this family's template writes
    pub fn markup(&self) -> Markup {
        Markup::Glm
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the checkpoint's own config beside the repository root (tests run from engine/);
    /// `None` when this machine has not downloaded it (`models/` is not in git)
    fn meta(dir: &str) -> Option<ModelMeta> {
        let c = format!("../models/{dir}/config.json");
        if !std::path::Path::new(&c).is_file() {
            eprintln!("no {c} on this machine - skipped");
            return None;
        }
        let g = format!("../models/{dir}/generation_config.json");
        let g = std::path::Path::new(&g).is_file().then_some(g);
        Some(ModelMeta::from_config_files(&c, g.as_deref()).expect("config parses"))
    }

    /// #185 part 1: the boot refuses by name, naming the tiers part 2 needs
    #[test]
    fn the_boot_refuses_naming_the_expert_tiers() {
        let Some(m) = meta("GLM-5.3-Flash-original") else { return };
        assert_eq!(m.family, Family::Glm5Next);
        let e = Glm5Engine::boot("converter/GLM-5.3-Flash-MUL1K3.cnq", &m).err().expect("refused");
        assert!(e.contains(NEEDS), "{e}");
        assert!(e.contains("converter/GLM-5.3-Flash-MUL1K3.cnq"), "{e}");
        assert!(e.contains("refusing to boot"), "{e}");
    }

    /// a Qwen config is never booted as glm5_next
    #[test]
    fn a_qwen_config_is_not_a_glm5_engine() {
        let Some(m) = meta("Qwen3.8-27B") else { return };
        let e = Glm5Engine::boot("converter/Qwen3.8-27B-CNQ4.5.cnq", &m).err().expect("refused");
        assert!(e.contains("is not glm5_next"), "{e}");
        assert!(!e.contains(NEEDS), "{e}");
    }
}
