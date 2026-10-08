//! crow-nest #156 (GLM-5.3-Flash plan step 6): the tensor filter of a partial container.
//!
//! `--layers <spec>` keeps the text decoder layers named in `<spec>` (`0-3`, `0,3`, `0-2,7`);
//! `--with-embed-head` adds the token embedding, `lm_head` and the final norm. Everything else
//! the recipe would write is FILTERED: not read, not written, and named as filtered (not as
//! missing) by the coverage check. The index trailer then carries a `partial` block, so a
//! partial container can never pass for a whole one.
//!
//! The filter runs after the manifest is built, so the recipe's whitelist and the config-vs-
//! weights geometry check still see every tensor of the checkpoint.

use std::collections::BTreeSet;

/// What a partial conversion keeps.
#[derive(Clone, Debug, PartialEq)]
pub struct LayerFilter {
    pub layers: BTreeSet<u64>,
    pub embed_head_norm: bool,
}

impl LayerFilter {
    /// `spec`: comma list of layers and inclusive ranges, e.g. `0-3` or `0,3,5-7`.
    pub fn parse(spec: &str, embed_head_norm: bool) -> Result<Self, String> {
        let mut layers = BTreeSet::new();
        for part in spec.split(',') {
            let part = part.trim();
            let num = |s: &str| s.trim().parse::<u64>().map_err(|_| format!("--layers {spec}: `{s}` is not a layer number"));
            match part.split_once('-') {
                Some((a, b)) => {
                    let (a, b) = (num(a)?, num(b)?);
                    if a > b {
                        return Err(format!("--layers {spec}: range {a}-{b} runs backwards"));
                    }
                    layers.extend(a..=b);
                }
                None => {
                    layers.insert(num(part)?);
                }
            }
        }
        Ok(LayerFilter { layers, embed_head_norm })
    }

    /// The text decoder layer a tensor belongs to: the number after a `layers` path segment,
    /// outside the vision tower. `None` for embeddings, head, final norm and the vision tower.
    pub fn layer_of(name: &str) -> Option<u64> {
        if crate::recipe::section_of(name) == "vit" {
            return None;
        }
        let parts: Vec<&str> = name.split('.').collect();
        parts.windows(2).find(|w| w[0] == "layers").and_then(|w| w[1].parse().ok())
    }

    /// The token embedding, `lm_head` and the final norm of the text model.
    pub fn is_embed_head_norm(name: &str) -> bool {
        name == "lm_head.weight"
            || name.ends_with("language_model.embed_tokens.weight")
            || name == "model.embed_tokens.weight"
            || name.ends_with("language_model.norm.weight")
            || name == "model.norm.weight"
    }

    pub fn keeps(&self, name: &str) -> bool {
        match Self::layer_of(name) {
            Some(l) => self.layers.contains(&l),
            None => self.embed_head_norm && Self::is_embed_head_norm(name),
        }
    }

    /// The flags as given (normalised), for logs and the index.
    pub fn describe(&self) -> String {
        let mut runs: Vec<String> = Vec::new();
        let v: Vec<u64> = self.layers.iter().copied().collect();
        let mut i = 0;
        while i < v.len() {
            let mut j = i;
            while j + 1 < v.len() && v[j + 1] == v[j] + 1 {
                j += 1;
            }
            runs.push(if i == j { v[i].to_string() } else { format!("{}-{}", v[i], v[j]) });
            i = j + 1;
        }
        format!("--layers {}{}", runs.join(","), if self.embed_head_norm { " --with-embed-head" } else { "" })
    }

    /// The index trailer's `partial` block.
    pub fn index_block(&self, kept: usize, filtered: usize) -> serde_json::Value {
        serde_json::json!({
            "layers": self.layers.iter().collect::<Vec<_>>(),
            "embed_head_norm": self.embed_head_norm,
            "filter": self.describe(),
            "tensors_written": kept,
            "tensors_filtered": filtered,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_spec_parses_ranges_and_lists() {
        let f = LayerFilter::parse("0-3", true).unwrap();
        assert_eq!(f.layers.iter().copied().collect::<Vec<_>>(), vec![0, 1, 2, 3]);
        let f = LayerFilter::parse("0,3,5-7", false).unwrap();
        assert_eq!(f.layers.iter().copied().collect::<Vec<_>>(), vec![0, 3, 5, 6, 7]);
        assert_eq!(f.describe(), "--layers 0,3,5-7");
        assert_eq!(LayerFilter::parse("0-3", true).unwrap().describe(), "--layers 0-3 --with-embed-head");
        assert!(LayerFilter::parse("3-0", false).is_err());
        assert!(LayerFilter::parse("a", false).is_err());
        assert!(LayerFilter::parse("", false).is_err());
    }

    #[test]
    fn it_keeps_the_named_layers_and_the_embed_head_norm_only() {
        let f = LayerFilter::parse("0-3", true).unwrap();
        for keep in [
            "model.language_model.layers.0.self_attn.q_proj.weight",
            "model.language_model.layers.3.mlp.experts.287.down_proj.weight",
            "model.language_model.layers.3.mlp.experts.0.gate_proj.weight_scale_inv",
            "model.language_model.embed_tokens.weight",
            "lm_head.weight",
            "model.language_model.norm.weight",
        ] {
            assert!(f.keeps(keep), "{keep}");
        }
        for drop in [
            "model.language_model.layers.4.input_layernorm.weight",
            "model.language_model.layers.30.mlp.gate.weight",
            "model.language_model.layers.45.mlp.experts.0.up_proj.weight",
            // a layer norm is not the final norm
            "model.language_model.layers.13.post_attention_layernorm.weight",
            "model.visual.blocks.0.norm1.weight",
        ] {
            assert!(!f.keeps(drop), "{drop}");
        }
        let no_head = LayerFilter::parse("0-3", false).unwrap();
        assert!(!no_head.keeps("lm_head.weight"));
        assert!(!no_head.keeps("model.language_model.embed_tokens.weight"));
        assert!(no_head.keeps("model.language_model.layers.2.mlp.down_proj.weight"));
    }
}
