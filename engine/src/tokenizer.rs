//! #25 A3 - in-engine tokenizer and chat template (the server's product path).
//!
//! What this module replaces:
//!
//! - `tools/tokenize_ids.py` (transformers 5.16.1, `.venv-oracle`) as a PRODUCT dependency.
//! - The Python script stays, as the GATE ORACLE only.
//! - No Python process is started from here, ever.
//!
//! Inputs:
//!
//! | file | default | env override |
//! |---|---|---|
//! | HF tokenizer | `models/Qwen3.8-Flash-Next-original/tokenizer.json` | `CROW_TOKENIZER` |
//! | chat template | sibling `tokenizer_config.json`, field `chat_template` | `CROW_TOKENIZER_CONFIG` |
//!
//! Python semantics reproduced (`tools/tokenize_ids.py:26-45`):
//!
//! - `apply_chat_template(messages, add_generation_prompt=True, tokenize=True, enable_thinking=False)`.
//! - `add_generation_prompt` and `enable_thinking` are TEMPLATE VARIABLES, not string surgery.
//! - `tools` and `documents` are passed as template variables too (`none` when absent).
//! - `apply_chat_template` tokenizes the rendered string with `add_special_tokens=False`.
//! - Raw mode: `tok(text, add_special_tokens=False)["input_ids"]`.
//!
//! Jinja environment, matched to `transformers/utils/chat_template_utils.py:489-493`:
//!
//! | setting | value | reason |
//! |---|---|---|
//! | `trim_blocks` | true | `ImmutableSandboxedEnvironment(trim_blocks=True, ...)` |
//! | `lstrip_blocks` | true | same call |
//! | `keep_trailing_newline` | false | jinja2 default, minijinja default |
//! | `raise_exception` | function | jinja_env.globals, raises a template error |
//! | `tojson` | filter | jinja_env.filters, `json.dumps` defaults, separators `", "` and `": "` |
//! | unknown method callback | `py_method` | jinja2 runs on Python objects, minijinja does not |
//!
//! Python methods the callback answers (jinja2 gets them for free, minijinja does not):
//!
//! - string: `startswith`, `endswith`, `strip`, `lstrip`, `rstrip`, `lower`, `upper`.
//! - string: `replace`, `split`, `splitlines`.
//! - mapping: `items`, `get`; `get` on a NON mapping stays `UnknownMethod`, as in Python.
//! - `content.startswith('<tool_response>')` in the Qwen3 template is why this exists.
//! - Anything else stays `ErrorKind::UnknownMethod`, so a silent wrong render is impossible.
//!
//! Template features used by the Qwen3 template (minijinja features enabled for them):
//!
//! - `namespace`, macros with default arguments (`macros`, default feature).
//! - `messages[::-1]` slicing, `loop.previtem` / `loop.nextitem` (`adjacent_loop_items`, default).
//! - `tojson`, `items`, `default`, `trim`, `string`, `safe` (`builtins`, `json`).
//! - `loop_controls` enabled to mirror `jinja2.ext.loopcontrols` (this template does not use it).
//!
//! Key order of every JSON object the template renders (`tools`, later `tool_call.arguments`):
//!
//! | crate | feature | without it |
//! |---|---|---|
//! | `serde_json` | `preserve_order` | `Map` is a `BTreeMap`, `py_json` walks it SORTED |
//! | `minijinja` | `preserve_order` | the value map is a `BTreeMap`, sorted BEFORE `tojson` runs |
//!
//! - Python renders INSERTION order, so both features are needed for oracle identical ids.
//! - Measured: without them `parameters` renders `additionalProperties, properties, required, type`.
//! - `a_tools_render_is_byte_identical_to_the_oracle` is the test that pins this.
//!
//! Load cost:
//!
//! - `tokenizer.json` is 12.8 MB; `global()` loads it once per process (`OnceLock`).
//! - The chat template is compiled once into the owned `Environment`.

use minijinja::value::{from_args, ValueKind};
use minijinja::{Environment, Error as JErr, ErrorKind as JErrKind, State, Value as JVal};
use serde_json::Value;
use std::sync::OnceLock;
use tokenizers::tokenizer::Tokenizer;

/// HF tokenizer file, repository relative; `CROW_TOKENIZER` overrides it
pub const DEFAULT_TOKENIZER: &str = "models/Qwen3.8-Flash-Next-original/tokenizer.json";
/// template name inside the minijinja environment
const TEMPLATE_NAME: &str = "chat";

/// tokenizer plus compiled chat template, one per process
pub struct ChatTokenizer {
    tok: Tokenizer,
    env: Environment<'static>,
    tokenizer_path: String,
    config_path: String,
}

// ------------------------------------------------------------ jinja plumbing

/// `json.dumps(x, ensure_ascii=False, indent=None, separators=None)`:
///
/// - item separator `", "`, key separator `": "` (Python defaults, NOT compact).
/// - non ASCII stays raw (`ensure_ascii=False`).
/// - control characters are escaped `\uXXXX`, as `json.dumps` does.
/// - key order is INSERTION order, from `serde_json`'s `preserve_order` feature.
/// - without that feature the `Map` is a `BTreeMap` and `tools` would render sorted.
/// - minijinja's own `tojson` is compact, so it is replaced.
fn py_json(v: &Value, out: &mut String) {
    match v {
        Value::Array(a) => {
            out.push('[');
            for (i, x) in a.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                py_json(x, out);
            }
            out.push(']');
        }
        Value::Object(m) => {
            out.push('{');
            for (i, (k, x)) in m.iter().enumerate() {
                if i > 0 {
                    out.push_str(", ");
                }
                out.push_str(&Value::String(k.clone()).to_string());
                out.push_str(": ");
                py_json(x, out);
            }
            out.push('}');
        }
        other => out.push_str(&other.to_string()),
    }
}

/// `str.strip()` / `lstrip` / `rstrip`: no argument strips whitespace, an argument
/// strips any character out of that set (Python semantics, not a suffix trim).
fn py_strip<'a>(s: &'a str, chars: Option<&str>, left: bool, right: bool) -> &'a str {
    let mut out = s;
    match chars {
        None => {
            if left {
                out = out.trim_start();
            }
            if right {
                out = out.trim_end();
            }
        }
        Some(set) => {
            if left {
                out = out.trim_start_matches(|c| set.contains(c));
            }
            if right {
                out = out.trim_end_matches(|c| set.contains(c));
            }
        }
    }
    out
}

/// - the Python methods jinja2 gets from the Python data model, which minijinja has not
/// - the list is explicit; an unlisted method stays `UnknownMethod`
fn py_method(state: &State, value: &JVal, method: &str, args: &[JVal]) -> Result<JVal, JErr> {
    if let Some(s) = value.as_str() {
        match method {
            "startswith" => {
                let (p,): (&str,) = from_args(args)?;
                return Ok(JVal::from(s.starts_with(p)));
            }
            "endswith" => {
                let (p,): (&str,) = from_args(args)?;
                return Ok(JVal::from(s.ends_with(p)));
            }
            "strip" | "lstrip" | "rstrip" => {
                let (c,): (Option<&str>,) = from_args(args)?;
                let (l, r) = (method != "rstrip", method != "lstrip");
                return Ok(JVal::from(py_strip(s, c, l, r)));
            }
            "lower" => {
                let _: () = from_args(args)?;
                return Ok(JVal::from(s.to_lowercase()));
            }
            "upper" => {
                let _: () = from_args(args)?;
                return Ok(JVal::from(s.to_uppercase()));
            }
            "replace" => {
                let (a, b): (&str, &str) = from_args(args)?;
                return Ok(JVal::from(s.replace(a, b)));
            }
            "split" => {
                let (sep,): (Option<&str>,) = from_args(args)?;
                let parts: Vec<JVal> = match sep {
                    // Python: no separator splits on runs of whitespace, empties dropped
                    None => s.split_whitespace().map(JVal::from).collect(),
                    Some(sep) => s.split(sep).map(JVal::from).collect(),
                };
                return Ok(JVal::from(parts));
            }
            "splitlines" => {
                let _: () = from_args(args)?;
                let parts: Vec<JVal> = s.lines().map(JVal::from).collect();
                return Ok(JVal::from(parts));
            }
            _ => {}
        }
    }
    match method {
        "items" => {
            let _: () = from_args(args)?;
            return state.apply_filter("items", &[value.clone()]);
        }
        // Python: only a mapping has .get. On a string, a list or none this must be LOUD,
        // otherwise a template that diverges from Python renders a plausible wrong string.
        "get" if value.kind() == ValueKind::Map => {
            let (k, d): (JVal, Option<JVal>) = from_args(args)?;
            let hit = value.get_item(&k).ok().filter(|v| !v.is_undefined());
            return Ok(hit.unwrap_or_else(|| d.unwrap_or_else(|| JVal::from(()))));
        }
        _ => {}
    }
    Err(JErr::new(
        JErrKind::UnknownMethod,
        format!("{method} is not one of the Python methods this environment answers"),
    ))
}

/// the environment `render_chat` renders in; template is owned, so it outlives the call
fn build_env(template: String) -> Result<Environment<'static>, String> {
    let mut env = Environment::new();
    env.set_trim_blocks(true);
    env.set_lstrip_blocks(true);
    env.set_keep_trailing_newline(false);
    env.set_unknown_method_callback(py_method);
    env.add_function("raise_exception", |msg: String| -> Result<JVal, JErr> {
        Err(JErr::new(JErrKind::InvalidOperation, msg))
    });
    env.add_filter("tojson", |v: JVal| -> Result<JVal, JErr> {
        let j: Value = serde_json::to_value(&v)
            .map_err(|e| JErr::new(JErrKind::InvalidOperation, format!("tojson: {e}")))?;
        let mut s = String::new();
        py_json(&j, &mut s);
        Ok(JVal::from_safe_string(s))
    });
    env.add_template_owned(TEMPLATE_NAME, template)
        .map_err(|e| format!("chat template does not compile: {e:#}"))?;
    Ok(env)
}

// ------------------------------------------------------------- construction

/// sibling `tokenizer_config.json` of a `tokenizer.json` path
pub fn sibling_config(tokenizer_path: &str) -> String {
    let p = tokenizer_path.replace('\\', "/");
    match p.rfind('/') {
        Some(i) => format!("{}/tokenizer_config.json", &p[..i]),
        None => "tokenizer_config.json".to_string(),
    }
}

impl ChatTokenizer {
    /// - `tokenizer_path` is an HF `tokenizer.json`
    /// - `config_path` is the `tokenizer_config.json` carrying `chat_template`
    /// - `Err` carries the operator message, no panic
    pub fn load(tokenizer_path: &str, config_path: &str) -> Result<Self, String> {
        let tok = Tokenizer::from_file(tokenizer_path)
            .map_err(|e| format!("cannot load {tokenizer_path}: {e}"))?;
        let raw = std::fs::read(config_path).map_err(|e| format!("cannot read {config_path}: {e}"))?;
        let cfg: Value =
            serde_json::from_slice(&raw).map_err(|e| format!("{config_path} is not JSON: {e}"))?;
        let template = cfg
            .get("chat_template")
            .and_then(|v| v.as_str())
            .ok_or_else(|| format!("{config_path} has no string field chat_template"))?
            .to_string();
        Ok(ChatTokenizer {
            tok,
            env: build_env(template)?,
            tokenizer_path: tokenizer_path.to_string(),
            config_path: config_path.to_string(),
        })
    }

    /// paths this instance was built from, for the `/props` document and the gate log
    pub fn paths(&self) -> (&str, &str) {
        (&self.tokenizer_path, &self.config_path)
    }

    /// `tok(text, add_special_tokens=False)["input_ids"]`
    pub fn encode_raw(&self, text: &str) -> Result<Vec<u32>, String> {
        let enc = self
            .tok
            .encode(text, false)
            .map_err(|e| format!("encode failed: {e}"))?;
        Ok(enc.get_ids().to_vec())
    }

    /// - `messages` is the OpenAI shape, a JSON array of `{role, content, ...}`
    /// - `tools` is passed through as the template variable `tools` (`none` when `None`)
    /// - `add_generation_prompt` and `enable_thinking` are template variables
    /// - `documents` is `none`, as `render_jinja_template` passes it
    pub fn render_chat(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<String, String> {
        let tmpl = self
            .env
            .get_template(TEMPLATE_NAME)
            .map_err(|e| format!("chat template missing: {e:#}"))?;
        let ctx = minijinja::context! {
            messages => JVal::from_serialize(messages),
            tools => match tools {
                Some(t) => JVal::from_serialize(t),
                None => JVal::from(()),
            },
            documents => JVal::from(()),
            add_generation_prompt => add_generation_prompt,
            enable_thinking => enable_thinking,
        };
        tmpl.render(ctx)
            .map_err(|e| format!("chat template render failed: {e:#}"))
    }

    /// `render_chat` then `encode(..., add_special_tokens=False)`, as `apply_chat_template` does
    pub fn encode_chat(
        &self,
        messages: &Value,
        tools: Option<&Value>,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> Result<Vec<u32>, String> {
        let s = self.render_chat(messages, tools, add_generation_prompt, enable_thinking)?;
        self.encode_raw(&s)
    }

    /// the one message shape `tokenize_ids.py --chat` builds, and its ids
    pub fn encode_chat_user(&self, text: &str) -> Result<Vec<u32>, String> {
        self.encode_chat(&user_message(text), None, true, false)
    }

    /// `tok.decode(ids, skip_special_tokens=True)`, the shape `tools/detokenize_ids.py` prints
    pub fn decode(&self, ids: &[u32]) -> Result<String, String> {
        self.tok
            .decode(ids, true)
            .map_err(|e| format!("decode failed: {e}"))
    }

    /// `tok.decode(ids, skip_special_tokens=False)`, for stop token inspection
    pub fn decode_with_specials(&self, ids: &[u32]) -> Result<String, String> {
        self.tok
            .decode(ids, false)
            .map_err(|e| format!("decode failed: {e}"))
    }

    /// id of a special token, e.g. `<|im_end|>` as the stop token for A4
    pub fn token_id(&self, token: &str) -> Option<u32> {
        self.tok.token_to_id(token)
    }
}

/// the single user message `tokenize_ids.py --chat` sends
pub fn user_message(text: &str) -> Value {
    serde_json::json!([{ "role": "user", "content": text }])
}

/// - `CROW_TOKENIZER` or `DEFAULT_TOKENIZER`
/// - `CROW_TOKENIZER_CONFIG` or the sibling `tokenizer_config.json`
pub fn default_paths() -> (String, String) {
    let t = std::env::var("CROW_TOKENIZER").unwrap_or_else(|_| DEFAULT_TOKENIZER.to_string());
    let c = std::env::var("CROW_TOKENIZER_CONFIG").unwrap_or_else(|_| sibling_config(&t));
    (t, c)
}

static GLOBAL: OnceLock<Result<ChatTokenizer, String>> = OnceLock::new();

/// - process wide instance, the 12.8 MB file is read once
/// - `Err` is cached too, so a bad path does not re-read on every request
pub fn global() -> Result<&'static ChatTokenizer, &'static str> {
    match GLOBAL.get_or_init(|| {
        let (t, c) = default_paths();
        ChatTokenizer::load(&t, &c)
    }) {
        Ok(k) => Ok(k),
        Err(e) => Err(e.as_str()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tk() -> ChatTokenizer {
        // tests run from engine/, the model lives at the repository root
        let t = format!("../{DEFAULT_TOKENIZER}");
        ChatTokenizer::load(&t, &sibling_config(&t)).expect("tokenizer loads")
    }

    #[test]
    fn config_path_is_the_sibling_of_the_tokenizer() {
        assert_eq!(sibling_config("models/x/tokenizer.json"), "models/x/tokenizer_config.json");
        assert_eq!(sibling_config("C:\\m\\x\\tokenizer.json"), "C:/m/x/tokenizer_config.json");
        assert_eq!(sibling_config("tokenizer.json"), "tokenizer_config.json");
    }

    #[test]
    fn tojson_uses_the_python_separators() {
        let mut s = String::new();
        py_json(&serde_json::json!({"a": 1, "b": [1, 2]}), &mut s);
        assert_eq!(s, "{\"a\": 1, \"b\": [1, 2]}");
        let mut s = String::new();
        py_json(&serde_json::json!([]), &mut s);
        assert_eq!(s, "[]");
    }

    /// Provenance of `PY_JSON`: run once on 2026-09-09 with
    ///   `.venv-oracle/Scripts/python.exe` (CPython 3.13.3), `PYTHONIOENCODING=utf-8`,
    ///   `json.dumps({"groesse": "Groesse in Zoll", "ctrl": "ab\nc",
    ///                "zoll": 27.5, "n": [1, 2.0]}, ensure_ascii=False)`
    ///   with the two `groesse` spelled with the real umlaut and sharp s.
    /// Covers: non ASCII raw, a control character escaped, an escaped newline,
    /// a float, and an int next to a float in one array.
    #[test]
    fn py_json_matches_python_json_dumps_ensure_ascii_false() {
        const PY_JSON: &str = "{\"gr\u{f6}\u{df}e\": \"Gr\u{f6}\u{df}e in Zoll\", \
                               \"ctrl\": \"a\\u001fb\\nc\", \"zoll\": 27.5, \"n\": [1, 2.0]}";
        let v = serde_json::json!({
            "gr\u{f6}\u{df}e": "Gr\u{f6}\u{df}e in Zoll",
            "ctrl": "a\u{1f}b\nc",
            "zoll": 27.5,
            "n": [1, 2.0]
        });
        let mut s = String::new();
        py_json(&v, &mut s);
        assert_eq!(s, PY_JSON);
    }

    #[test]
    fn get_on_a_non_mapping_is_an_unknown_method_not_an_answer() {
        let render = |src: &str, ctx: JVal| -> Result<String, JErr> {
            let env = build_env(src.to_string()).expect("template compiles");
            env.get_template(TEMPLATE_NAME)?.render(ctx)
        };
        // a mapping answers .get, so the callback is reachable at all
        let m = JVal::from_serialize(serde_json::json!({ "a": 1 }));
        let ok = render("{{ m.get('a') }}|{{ m.get('zz', 'dflt') }}", minijinja::context! { m })
            .expect("get on a mapping answers");
        assert_eq!(ok, "1|dflt");
        // a string does not; Python raises AttributeError, this raises UnknownMethod
        let e = render("{{ s.get('a') }}", minijinja::context! { s => "text" })
            .expect_err("get on a string is loud");
        assert_eq!(e.kind(), JErrKind::UnknownMethod, "error was {e:#}");
    }

    #[test]
    fn ascii_round_trips_through_encode_and_decode() {
        let t = tk();
        // oracle: .venv-oracle python, tok("Hello world", add_special_tokens=False) -> [9419, 1814]
        let ids = t.encode_raw("Hello world").unwrap();
        assert_eq!(ids, vec![9419u32, 1814]);
        assert_eq!(t.decode(&ids).unwrap(), "Hello world");
    }

    #[test]
    fn utf8_round_trips_through_encode_and_decode() {
        let t = tk();
        // "Gruesse ae oe ue, Adler" with the real umlauts and an emoji, written as escapes
        // so this file stays ASCII
        let s = "Gr\u{fc}\u{df}e \u{e4}\u{f6}\u{fc} \u{1f985}";
        let ids = t.encode_raw(s).unwrap();
        assert!(ids.len() > 3, "multi byte text does not collapse to one token");
        assert_eq!(t.decode(&ids).unwrap(), s);
    }

    /// Provenance of the expected string: run once on 2026-09-09 with
    ///   .venv-oracle/Scripts/python.exe, transformers 5.16.1,
    ///   AutoTokenizer.from_pretrained("models/Qwen3.8-Flash-Next-original")
    ///   .apply_chat_template([{"role":"user","content":"Hello"}],
    ///        add_generation_prompt=True, tokenize=False, enable_thinking=False)
    /// printed with json.dumps ->
    ///   "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    /// and the same call with tokenize=True ->
    ///   [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
    #[test]
    fn chat_render_ends_with_the_assistant_header_and_the_empty_think_block() {
        let t = tk();
        const ORACLE: &str =
            "<|im_start|>user\nHello<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n";
        let s = t.render_chat(&user_message("Hello"), None, true, false).unwrap();
        assert_eq!(s, ORACLE);
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(
            t.encode_chat_user("Hello").unwrap(),
            vec![248045u32, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271]
        );
    }

    #[test]
    fn enable_thinking_and_add_generation_prompt_are_template_variables() {
        let t = tk();
        let m = user_message("Hello");
        // thinking on: the header opens an unclosed think block
        let on = t.render_chat(&m, None, true, true).unwrap();
        assert!(on.ends_with("<|im_start|>assistant\n<think>\n"));
        assert!(!on.contains("</think>"));
        // no generation prompt: the render stops after the user turn
        let off = t.render_chat(&m, None, false, false).unwrap();
        assert_eq!(off, "<|im_start|>user\nHello<|im_end|>\n");
    }

    #[test]
    fn a_tools_render_does_not_fail() {
        let t = tk();
        let tools = serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "weather for a city",
                "parameters": { "type": "object", "properties": { "city": { "type": "string" } } }
            }
        }]);
        let s = t
            .render_chat(&user_message("Hello"), Some(&tools), true, false)
            .expect("tools render");
        assert!(s.starts_with("<|im_start|>system\n"));
        assert!(s.contains("<tools>"));
        assert!(s.contains("get_weather"));
        assert!(s.ends_with("<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    /// the tools case the reviewer asked for: `parameters` has four keys in NON alphabetical
    /// order (`type`, `properties`, `required`, `additionalProperties`) and a non ASCII
    /// description, so a sorted `Map` renders a different string and different ids.
    fn oracle_tools() -> Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "get_monitor",
                "description": "Monitordaten",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "zoll": { "type": "number", "description": "Gr\u{f6}\u{df}e in Zoll" },
                        "marke": { "type": "string" },
                        "aktiv": { "type": "boolean" },
                        "id": { "type": "integer" }
                    },
                    "required": ["zoll", "marke"],
                    "additionalProperties": false
                }
            }
        }])
    }

    /// Provenance of `ORACLE_RENDER` and `ORACLE_IDS`: run once on 2026-09-09 with
    ///   `.venv-oracle/Scripts/python.exe` (CPython 3.13.3), `PYTHONIOENCODING=utf-8`,
    ///   transformers 5.16.1,
    ///   `AutoTokenizer.from_pretrained("models/Qwen3.8-Flash-Next-original")`
    ///   `.apply_chat_template([{"role":"user","content":"Wie gross ist der Monitor?"}],`
    ///   `    tools=TOOLS, add_generation_prompt=True, tokenize=False, enable_thinking=False)`
    ///   and the same call with `tokenize=True` for the 322 ids.
    ///   `TOOLS` is `oracle_tools()` above, key for key in that order;
    ///   the two `gross` in the message and in the description carry the real sharp s.
    /// Without `serde_json`'s `preserve_order` the `parameters` object renders
    ///   `additionalProperties, properties, required, type` and both assertions fail.
    #[test]
    fn a_tools_render_is_byte_identical_to_the_oracle() {
        let t = tk();
        const ORACLE_RENDER: &str = concat!(
            "<|im_start|>system\n",
            "# Tools\n",
            "\n",
            "You have access to the following functions:\n",
            "\n",
            "<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"get_monitor\", \"description\": \"Monitordaten\", \"parameters\": {\"type\": \"object\", \"properties\": {\"zoll\": {\"type\": \"number\", \"description\": \"Gr\u{f6}\u{df}e in Zoll\"}, \"marke\": {\"type\": \"string\"}, \"aktiv\": {\"type\": \"boolean\"}, \"id\": {\"type\": \"integer\"}}, \"required\": [\"zoll\", \"marke\"], \"additionalProperties\": false}}}\n",
            "</tools>\n",
            "\n",
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
            "\n",
            "<tool_call>\n",
            "<function=example_function_name>\n",
            "<parameter=example_parameter_1>\n",
            "value_1\n",
            "</parameter>\n",
            "<parameter=example_parameter_2>\n",
            "This is the value for the second parameter\n",
            "that can span\n",
            "multiple lines\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>\n",
            "\n",
            "<IMPORTANT>\n",
            "Reminder:\n",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
            "- Required parameters MUST be specified\n",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
            "</IMPORTANT><|im_end|>\n",
            "<|im_start|>user\n",
            "Wie gro\u{df} ist der Monitor?<|im_end|>\n",
            "<|im_start|>assistant\n",
            "<think>\n",
            "\n",
            "</think>\n",
            "\n",
        );
        const ORACLE_IDS: [u32; 322] = [
            248045, 8678, 198, 2, 13455, 271, 2523, 599, 2528, 310, 279, 2614, 5568, 25, 271, 27,
            15449, 29, 198, 4754, 1267, 763, 328, 1628, 487, 328, 1628, 763, 5046, 591, 763, 328,
            447, 38780, 487, 328, 4532, 763, 328, 10772, 275, 526, 13128, 487, 328, 13390, 763,
            5046, 1267, 763, 328, 1640, 487, 328, 12811, 763, 5046, 89, 935, 763, 5046, 1267, 763,
            328, 3946, 487, 328, 4532, 763, 328, 6262, 76239, 303, 195643, 13933, 328, 5437, 432,
            763, 5046, 1267, 763, 328, 889, 13933, 328, 71106, 763, 5046, 1267, 763, 328, 5925,
            13933, 328, 306, 763, 5046, 1267, 763, 328, 11326, 8934, 2069, 328, 6081, 763, 4241,
            89, 935, 487, 328, 5437, 432, 7664, 328, 34325, 7654, 763, 867, 72964, 198, 510, 15449,
            29, 271, 2592, 488, 4992, 310, 1562, 264, 709, 25835, 9559, 303, 279, 2614, 3443, 440,
            5486, 19900, 25, 271, 248058, 198, 27, 1628, 28, 8422, 8901, 1224, 29, 198, 27, 15704,
            28, 8422, 24109, 62, 16, 29, 198, 927, 62, 16, 198, 510, 15704, 29, 198, 27, 15704, 28,
            8422, 24109, 62, 17, 29, 198, 1919, 369, 279, 869, 364, 279, 2018, 5555, 198, 8761,
            628, 9111, 198, 34493, 4965, 198, 510, 15704, 29, 198, 510, 1628, 29, 198, 248059, 271,
            27, 95328, 29, 198, 92065, 25, 198, 12, 5534, 6526, 26834, 1732, 279, 5024, 3443, 25,
            449, 8906, 361, 1628, 28, 1076, 1419, 1628, 29, 2424, 1902, 381, 23283, 2785, 220,
            248058, 248059, 11535, 9212, 198, 12, 12296, 4868, 26834, 381, 5024, 198, 12, 1394,
            1189, 3300, 9801, 31626, 364, 678, 709, 1562, 303, 5629, 3992, 54588, 279, 709, 1562,
            11, 694, 4045, 1238, 198, 12, 1368, 1017, 369, 874, 709, 1562, 2420, 11, 4087, 279,
            3296, 1040, 4472, 440, 678, 1428, 6337, 321, 635, 524, 3184, 279, 1156, 883, 709, 6526,
            198, 510, 95328, 29, 248046, 198, 248045, 846, 198, 63614, 64468, 5810, 2607, 22784,
            30, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271,
        ];
        let tools = oracle_tools();
        let msg = user_message("Wie gro\u{df} ist der Monitor?");
        let s = t.render_chat(&msg, Some(&tools), true, false).expect("tools render");
        assert_eq!(s, ORACLE_RENDER);
        // the key order the reviewer measured, spelled out so a regression names itself
        assert!(s.contains(
            "{\"type\": \"object\", \"properties\": {\"zoll\": {\"type\": \"number\", \
             \"description\": \"Gr\u{f6}\u{df}e in Zoll\"}"
        ));
        let ids = t.encode_chat(&msg, Some(&tools), true, false).expect("tools encode");
        assert_eq!(ids.len(), ORACLE_IDS.len());
        assert_eq!(ids, ORACLE_IDS.to_vec());
    }

    #[test]
    fn an_empty_message_list_is_a_template_error_not_a_panic() {
        let t = tk();
        let e = t
            .render_chat(&serde_json::json!([]), None, true, false)
            .expect_err("raise_exception fires");
        assert!(e.contains("No messages provided"), "message was {e}");
    }

    #[test]
    fn the_stop_token_resolves() {
        let t = tk();
        assert!(t.token_id("<|im_end|>").is_some());
        assert_eq!(t.token_id("<|im_end|>"), Some(248046));
    }
}
