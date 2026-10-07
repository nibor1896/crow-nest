//! #160 (GLM-5.3-Flash plan step 12): what is GLM-specific about the prompt side - the
//! family test, the reasoning words, the stop tokens, the tool-call markup - for
//! `zai-org/GLM-5.3-Flash` rev `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`.
//!
//! The render itself is `tokenizer::ChatTokenizer::render_chat_with`: the template is the
//! model's own `chat_template.jinja` (10,950 B, sha256 `0c4099f3...09c5`), read by
//! `ChatTokenizer::load` beside `tokenizer_config.json` (761 B, no `chat_template` field).
//!
//! What the template does (read 2026-10-08 from the downloaded file):
//!
//! | input | template | rendered |
//! |---|---|---|
//! | always | `[gMASK]<sop>` | the first two ids, 154822 154824 |
//! | `reasoning_effort` `low` / `high` | `effective_reasoning_effort` = the word | `<\|system\|>Reasoning Effort: Low` / `High` |
//! | `reasoning_effort` undefined or ANY other word | `'max'` | `<\|system\|>Reasoning Effort: Max` |
//! | `clear_thinking` | default false: every past turn keeps its `<think>..</think>`; true: only turns after the last user message | |
//! | `add_generation_prompt` | `<\|assistant\|><think>` | the prompt ENDS in an open think block, no newline |
//! | `enable_thinking` | not read at all | no byte moves (golden `sys_user_enable_thinking_false`) |
//!
//! - "off" is NOT representable: no variable of this template closes the think block of the
//!   generation prompt, and `effective_reasoning_effort` is never none. A closed block would
//!   be string surgery after the render, which #25's rule (template variables, not string
//!   surgery) excludes. So `none` is refused by name (`reasoning_level`).
//! - `medium` would silently render `max`, so it is refused by name too. The template has
//!   no `xhigh`: `high` and `xhigh` both map to `high`.
//!
//! Rendering table (goldens, `tests/fixtures/GLM-5.3-Flash/tokenizer-goldens.json`, system +
//! user message, `add_generation_prompt` true; sha256 of the UTF-8 render):
//!
//! | request word | template variable | rendered level | sha256 |
//! |---|---|---|---|
//! | absent | undefined | Max | `a28b9e93db2ccc8fb3c5cb1095cdef88ded6bd38d26a10582b1bc99d48b885b1` |
//! | `low` | `low` | Low | `05a7e44d1453c37c60293d62f707356e91a558edde0645972695b8a10836a218` |
//! | `high`, `xhigh` | `high` | High | `7c349ba45c32be4266bacc4ed862f64b090f6869af934353f7fa326295815e03` |
//! | `max` | undefined | Max | `a28b9e93...b885b1` (= absent) |
//! | `medium` | refused | (would be Max: `a28b9e93...b885b1`) | |
//! | `none` | refused | (would be Max: "off" is not representable) | |
//!
//! Token ids (`tokenizer.json`, 20,217,442 B, sha256 `19e77364...a82d`), all added tokens:
//!
//! | token | id | `special` |
//! |---|---|---|
//! | `<\|endoftext\|>` `<\|user\|>` `<\|observation\|>` | 154820 154827 154829 | true - the three `eos_token_id` of `generation_config.json` |
//! | `[gMASK]` `<sop>` `<\|system\|>` `<\|assistant\|>` | 154822 154824 154826 154828 | true |
//! | `<think>` `</think>` | 154841 154842 | false |
//! | `<tool_call>` `</tool_call>` | 154843 154844 | false |
//! | `<arg_key>` `</arg_key>` `<arg_value>` `</arg_value>` | 154847 154848 154849 154850 | false |
//!
//! - The six tool markers are NOT special, so `decode(.., skip_special_tokens=true)` keeps
//!   them and the tool grammar's vocabulary trie carries them as text: the parser
//!   (`toolcall::Markup::Glm`) and the grammar (`toolgrammar`, `Markup::Glm`) work on the
//!   same bytes the model writes. `<tool_call>` arms both by id, as for Qwen.

use crate::tokenizer::ChatTokenizer;
use crate::toolcall::Markup;

/// the source repository and revision the goldens were made from
pub const REPO: &str = "zai-org/GLM-5.3-Flash";
pub const REVISION: &str = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a";

/// the three end-of-turn tokens (`generation_config.json` `eos_token_id`)
pub const EOS_TOKENS: [&str; 3] = ["<|endoftext|>", "<|user|>", "<|observation|>"];
/// their ids in rev `eb9eb208`
pub const EOS_IDS: [u32; 3] = [154820, 154827, 154829];

/// the generation prompt's last bytes: the reasoning filter starts `Inside`
pub const PROMPT_END: &str = "<|assistant|><think>";

/// - the tokenizer is GLM-5's when it carries `[gMASK]` and the four argument markers as
///   added tokens; no Qwen vocabulary has either
pub fn is_glm5(tk: &ChatTokenizer) -> bool {
    ["[gMASK]", "<sop>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"]
        .iter()
        .all(|t| tk.token_id(t).is_some())
}

/// the tool-call markup of the loaded tokenizer's family
pub fn markup_of(tk: &ChatTokenizer) -> Markup {
    if is_glm5(tk) {
        Markup::Glm
    } else {
        Markup::Qwen
    }
}

/// - the GLM request words, as `REASONING_WORDS` lists them for a refusal
pub const REASONING_WORDS: &str = "low, high, xhigh, max";

/// - #160: the request's `reasoning_effort` word -> the template variable, `Ok(None)` = leave
///   it UNDEFINED (the template's default `max`)
///
/// | request | variable | rendered |
/// |---|---|---|
/// | absent | undefined | Max |
/// | `low` | `low` | Low |
/// | `high`, `xhigh` | `high` | High |
/// | `max` | undefined | Max |
/// | `medium` | refused | the template would render Max, a silent promotion |
/// | `none` | refused | "off" is not representable: the prompt always ends in `<think>` |
/// | anything else | refused | |
///
/// - matching is exact and lower case, as `bin/serve.rs`'s Qwen mapping matches
pub fn reasoning_level(word: Option<&str>) -> Result<Option<&'static str>, String> {
    match word {
        None | Some("max") => Ok(None),
        Some("low") => Ok(Some("low")),
        Some("high") | Some("xhigh") => Ok(Some("high")),
        Some("none") => Err(
            "reasoning_effort \"none\" cannot be rendered by GLM-5.3-Flash's template: its \
             generation prompt always ends in <think> and no template variable closes it \
             (accepted: low, high, xhigh, max)"
                .to_string(),
        ),
        Some("medium") => Err(
            "reasoning_effort \"medium\" is not a level of GLM-5.3-Flash's template (it would \
             silently render max; accepted: low, high, xhigh, max)"
                .to_string(),
        ),
        Some(other) => Err(format!(
            "reasoning_effort \"{other}\" is not one of {REASONING_WORDS} (GLM-5.3-Flash's \
             template has low, high and max)"
        )),
    }
}

/// - #160: the template variables of one GLM request: `reasoning_effort` only when
///   `reasoning_level` gave a word, `clear_thinking` only when the request set it
pub fn template_vars(level: Option<&str>, clear_thinking: Option<bool>) -> serde_json::Value {
    let mut m = serde_json::Map::new();
    if let Some(l) = level {
        m.insert("reasoning_effort".to_string(), serde_json::Value::from(l));
    }
    if let Some(c) = clear_thinking {
        m.insert("clear_thinking".to_string(), serde_json::Value::from(c));
    }
    serde_json::Value::Object(m)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::Value;

    /// the downloaded original, beside the repository root (tests run from engine/)
    const DIR: &str = "../models/GLM-5.3-Flash-original";
    const GOLDENS: &str = include_str!("../tests/fixtures/GLM-5.3-Flash/tokenizer-goldens.json");

    fn goldens() -> Value {
        serde_json::from_str(GOLDENS).expect("goldens parse")
    }

    /// the GLM tokenizer, or `None` when this machine has not downloaded it (`models/` is
    /// not in git); the parser and grammar tests do not need it
    fn tk() -> Option<ChatTokenizer> {
        let t = format!("{DIR}/tokenizer.json");
        if !std::path::Path::new(&t).is_file() {
            eprintln!("no {t} on this machine - skipped");
            return None;
        }
        Some(ChatTokenizer::load(&t, &crate::tokenizer::sibling_config(&t)).expect("GLM tokenizer loads"))
    }

    fn sha256_hex(s: &str) -> String {
        use sha2::Digest;
        let d = sha2::Sha256::digest(s.as_bytes());
        d.iter().map(|b| format!("{b:02x}")).collect()
    }

    #[test]
    fn the_template_comes_from_chat_template_jinja() {
        let Some(t) = tk() else { return };
        assert!(t.template_path().ends_with("/chat_template.jinja"), "{}", t.template_path());
        assert!(is_glm5(&t));
        assert_eq!(markup_of(&t), Markup::Glm);
        for (tok, id) in EOS_TOKENS.iter().zip(EOS_IDS) {
            assert_eq!(t.token_id(tok), Some(id), "{tok}");
            assert!(t.is_special(id), "{tok} is special");
        }
        let g = goldens();
        for (tok, id) in g["special_ids"].as_object().unwrap() {
            assert_eq!(t.token_id(tok), Some(id.as_u64().unwrap() as u32), "{tok}");
        }
        // the six tool markers are NOT special: decode keeps them, the trie carries them
        for m in ["<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>"] {
            let id = t.token_id(m).unwrap();
            assert!(!t.is_special(id), "{m}");
            assert_eq!(t.token_bytes(id), m.as_bytes(), "{m}");
        }
    }

    /// every encode golden: the ids equal transformers', both decodes byte-equal
    #[test]
    fn encode_and_decode_are_byte_identical_to_transformers() {
        let Some(t) = tk() else { return };
        let g = goldens();
        let cases = g["encode"].as_array().unwrap();
        assert_eq!(cases.len(), 10);
        for c in cases {
            let name = c["name"].as_str().unwrap();
            let text = c["text"].as_str().unwrap();
            let want: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            let ids = t.encode_raw(text).unwrap();
            assert_eq!(ids, want, "{name}: encode");
            assert_eq!(t.decode(&ids).unwrap(), c["decode_skip_special"].as_str().unwrap(), "{name}: decode");
            // the bytes of every id concatenate to the text (the logprobs `bytes` contract)
            let bytes: Vec<u8> = ids.iter().flat_map(|&i| t.token_bytes(i)).collect();
            assert_eq!(bytes, c["decode_keep_special"].as_str().unwrap().as_bytes(), "{name}: token_bytes");
        }
    }

    /// every render golden: the render byte-equal, its sha256, its ids
    #[test]
    fn every_render_is_byte_identical_to_transformers() {
        let Some(t) = tk() else { return };
        let g = goldens();
        let cases = g["render"].as_array().unwrap();
        assert_eq!(cases.len(), 16);
        let mut bad = Vec::new();
        for c in cases {
            let name = c["name"].as_str().unwrap();
            let tools = c.get("tools").filter(|v| !v.is_null());
            let agp = c["add_generation_prompt"].as_bool().unwrap();
            let got = match t.render_chat_with(&c["messages"], tools, agp, &c["template_vars"]) {
                Ok(s) => s,
                Err(e) => {
                    bad.push(format!("{name}: {e}"));
                    continue;
                }
            };
            let want = c["render"].as_str().unwrap();
            if got != want {
                let at = got.bytes().zip(want.bytes()).take_while(|(a, b)| a == b).count();
                bad.push(format!(
                    "{name}: differs at byte {at}: got {:?} want {:?}",
                    &got[at.saturating_sub(40).min(got.len())..(at + 60).min(got.len())],
                    &want[at.saturating_sub(40).min(want.len())..(at + 60).min(want.len())]
                ));
                continue;
            }
            assert_eq!(sha256_hex(&got), c["sha256"].as_str().unwrap(), "{name}: sha256");
            let want_ids: Vec<u32> = c["ids"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
            assert_eq!(t.encode_raw(&got).unwrap(), want_ids, "{name}: ids");
        }
        assert!(bad.is_empty(), "{} of {} renders differ:\n{}", bad.len(), cases.len(), bad.join("\n"));
    }

    /// #160: the wire words and what each renders; the hashes are the table of the module
    /// header, re-derived from the goldens and from this engine's own render
    #[test]
    fn reasoning_words_map_to_the_template_levels() {
        assert_eq!(reasoning_level(None), Ok(None));
        assert_eq!(reasoning_level(Some("max")), Ok(None));
        assert_eq!(reasoning_level(Some("low")), Ok(Some("low")));
        assert_eq!(reasoning_level(Some("high")), Ok(Some("high")));
        assert_eq!(reasoning_level(Some("xhigh")), Ok(Some("high")));
        for w in ["none", "medium", "High", "minimal", "off", ""] {
            let e = reasoning_level(Some(w)).expect_err(w);
            assert!(e.contains(&format!("\"{w}\"")), "{w}: {e}");
        }
        assert!(reasoning_level(Some("none")).unwrap_err().contains("always ends in <think>"));
        assert!(reasoning_level(Some("medium")).unwrap_err().contains("silently render max"));
        assert_eq!(template_vars(None, None), serde_json::json!({}));
        assert_eq!(template_vars(Some("low"), Some(true)), serde_json::json!({"reasoning_effort": "low", "clear_thinking": true}));

        let g = goldens();
        let hash = |name: &str| -> String {
            let c = g["render"].as_array().unwrap().iter().find(|c| c["name"] == name).unwrap();
            c["sha256"].as_str().unwrap().to_string()
        };
        const MAX: &str = "a28b9e93db2ccc8fb3c5cb1095cdef88ded6bd38d26a10582b1bc99d48b885b1";
        const LOW: &str = "05a7e44d1453c37c60293d62f707356e91a558edde0645972695b8a10836a218";
        const HIGH: &str = "7c349ba45c32be4266bacc4ed862f64b090f6869af934353f7fa326295815e03";
        assert_eq!((hash("sys_user_default"), hash("sys_user_low"), hash("sys_user_high")), (MAX.into(), LOW.into(), HIGH.into()));
        // every word the template does not know renders max, the reason they are refused
        for n in ["sys_user_medium", "sys_user_none", "sys_user_max", "sys_user_enable_thinking_false"] {
            assert_eq!(hash(n), MAX, "{n}");
        }
        let Some(t) = tk() else { return };
        let case = g["render"].as_array().unwrap().iter().find(|c| c["name"] == "sys_user_default").unwrap();
        for (word, want) in [(None, MAX), (Some("max"), MAX), (Some("low"), LOW), (Some("high"), HIGH), (Some("xhigh"), HIGH)] {
            let vars = template_vars(reasoning_level(word).unwrap(), None);
            let s = t.render_chat_with(&case["messages"], None, true, &vars).unwrap();
            assert_eq!(sha256_hex(&s), want, "{word:?}");
        }
    }

    /// #160: "off" is not representable - every generation prompt ends in the open think
    /// block, whatever the variables say, so the reasoning filter starts `Inside`
    #[test]
    fn off_is_not_representable_every_generation_prompt_opens_think() {
        let g = goldens();
        let mut n = 0;
        for c in g["render"].as_array().unwrap() {
            if c["add_generation_prompt"].as_bool().unwrap() {
                assert!(c["render"].as_str().unwrap().ends_with(PROMPT_END), "{}", c["name"]);
                n += 1;
            }
        }
        assert_eq!(n, 15);
        let Some(t) = tk() else { return };
        let m = crate::tokenizer::user_message("Hallo");
        for vars in [
            serde_json::json!({"enable_thinking": false}),
            serde_json::json!({"reasoning_effort": "none"}),
            serde_json::json!({"clear_thinking": true, "reasoning_effort": "low"}),
        ] {
            let s = t.render_chat_with(&m, None, true, &vars).unwrap();
            assert!(s.ends_with(PROMPT_END), "{vars}: {s:?}");
            assert!(!s.ends_with("</think>"), "{vars}");
        }
        // the last two ids are `<|assistant|>` and `<think>`
        let ids = t.encode_chat_with(&m, None, true, &serde_json::json!({})).unwrap();
        assert_eq!(&ids[ids.len() - 2..], &[154828, 154841]);
        assert_eq!(&ids[..2], &[154822, 154824], "[gMASK]<sop>");
    }
}
