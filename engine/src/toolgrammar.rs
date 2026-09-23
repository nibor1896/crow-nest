//! #93: LAZY, schema-derived constrained decoding of the model's tool-call
//! markup - llama.cpp's `common_chat_params_init_qwen3_coder` semantics (pinned tree
//! cbca449, `common/chat.cpp:1162-1330`), built for this engine's token loop.
//!
//! What it constrains:
//!
//! - Free text stays free. The grammar idles (`Ph::Idle`) until the model emits the
//!   `<tool_call>` token ID (248058) - the same event that arms `toolcall::ToolStream`, so
//!   the parser and the grammar can never disagree about where a call starts. llama.cpp's
//!   trigger is the WORD `<tool_call>` (`grammar_triggers`, `chat.cpp:1323-1328`); this
//!   model always emits it as the added token, and crow-nest's parser only opens a call on
//!   the id, so the id is the trigger here.
//! - From the trigger on, every byte must continue the frame of the chat template:
//!
//! ```text
//! <tool_call>\n<function=NAME>\n<parameter=P>\nVALUE\n</parameter>\n ... </function>\n</tool_call>
//! ```
//!
//!   NAME is one of the request's declared tools; P one of THAT tool's declared parameters,
//!   each at most once; `</function>` only once every required parameter was written.
//! - VALUE, by the parameter's schema (llama.cpp `resolves_to_string`,
//!   `common/json-schema-to-grammar.cpp:1125`):
//!
//! | schema | VALUE | llama.cpp |
//! |---|---|---|
//! | `type` string (or a type list / anyOf / oneOf with a string branch, a string `const`) | free text up to `\n</parameter>\n`; the text may not contain `</parameter>` or `</tool_call>` at all | `until("\n</parameter>\n")` - a `</parameter>` WITHOUT the newline, or `</tool_call>`, is allowed inside the value there; here `toolcall` would cut the value at it, so it is refused |
//! | string `enum` (every option a one-line string) | exactly one option, raw | free text (stricter here, the #93 ask) |
//! | integer | `-?(0|[1-9][0-9]{0,15})` | same (`integral-part`); `minimum`/`maximum` NOT enforced here |
//! | number | `-?int(\.[0-9]{1,16})?([eE][-+]?int)?` | same |
//! | boolean / null / enum / const | the JSON literals | same |
//! | array | JSON array, `items` schema per element | same (`minItems`/`maxItems` not enforced) |
//! | object with `properties` | JSON object, declared keys only (unless `additionalProperties` is true or a schema), each once, any order, every `required` key before `}` | declared keys, required keys in declared ORDER first |
//! | anything else (`$ref`, `allOf`, untyped) | any JSON value | the resolved schema |
//!
//!   A JSON value may be followed by `[ \t]*` before `\n</parameter>\n` (llama.cpp's
//!   `space` allows up to two newlines as well); whitespace inside containers is bounded to
//!   20 bytes per gap, as llama.cpp's `space` rule is.
//! - Parameter ORDER is free (llama.cpp: required ones first, permuted, then optionals);
//!   a parameter can NOT repeat (llama.cpp's `zero_or_more` over the optionals lets an
//!   optional one repeat; `toolcall` would then emit a duplicate JSON key).
//! - After `</tool_call>`: `space` (at most two newlines and 22 whitespace bytes), then
//!   either end of generation or - when `parallel_tool_calls` is true - the next
//!   `<tool_call>` token. No prose after a call: llama.cpp's root is
//!   `repeat(calls, min, 1)` and an empty stack only admits EOG
//!   (`src/llama-grammar.cpp:1361-1383`). The template's own instruction is the same
//!   ("ONLY reply in the following format with NO suffix").
//! - `tool_choice: "required"`: the same lazy grammar, plus end of generation is refused
//!   until one call has closed (llama.cpp: `min_calls` 1, `chat.cpp:1297`).
//!
//! Token level (the part llama.cpp does in `llama_grammar_reject_candidates`):
//!
//! - The state machine is BYTE level (`ToolGrammar::step`); a token is allowed when every
//!   byte of it steps (`token_ok`), so a token may span any number of grammar boundaries
//!   (`>\n`, `\n</parameter>\n<parameter=`...).
//! - The full mask (`Vocab::mask`) is a walk over a PREORDER BYTE TRIE of the vocabulary:
//!   a node whose byte is refused skips its whole subtree, so the walk costs the number of
//!   LIVE trie nodes, not V x token length. Special tokens (`<|im_end|>` ...) are not in
//!   the trie: inside a call they are refused, except EOS where the grammar is complete.
//!   `<tool_call>` itself is handled by id, never by its text.
//! - Rejection sampling, llama.cpp's default (`common/sampling.cpp` `common_sampler_sample`:
//!   sample, check the ONE drawn token, only on a reject apply the grammar to the whole row
//!   and draw again): the per-token price inside a call is one `token_ok` over the drawn
//!   token's bytes (27 ns per id measured, `tests::mask_cost_on_cpu`); the mask is built
//!   only on a reject (4.75 ms in a string value, <= 0.01 ms elsewhere) and cached per state.
//! - Outside a call the price is `Gate::armed` (0.38 ns) plus the advance that watches for
//!   the opener (2.4 ns) per generated id.
//!
//! What this module does NOT do: sampling, the logits row, the device - `bin/serve.rs`
//! owns the redraw (`chat_generate`) and `gen.rs` the device re-booking.

use std::collections::HashMap;

/// JSON container nesting the machine tracks; deeper schemas degrade to `Any`
pub const MAX_JSON_DEPTH: usize = 6;
/// bytes of whitespace allowed in one JSON gap / after a JSON value (llama.cpp `space`
/// is `| " " | "\n"{1,2} [ \t]{0,20}`)
const WS_MAX: u8 = 20;
/// whitespace after `</tool_call>` (llama.cpp's `space`: at most 2 newlines + 20)
const BETWEEN_WS_MAX: u8 = 22;
const BETWEEN_NL_MAX: u8 = 2;

// ------------------------------------------------------------------ the frame literals

const L_OPEN: u8 = 0; // after the `<tool_call>` id
const L_FGT: u8 = 1; // after the function name
const L_PARAM: u8 = 2; // entered at offset 2 from `BodyLt`
const L_PGT: u8 = 3; // after a parameter name
const L_FCLOSE: u8 = 4; // entered at offset 2 from `BodyLt`
const L_CALL_CLOSE: u8 = 5;
const L_PCLOSE: u8 = 6; // after a JSON value / an enum option
const L_NL: u8 = 7; // after the `\n</parameter>` a raw string's guard detected
const LITS: [&[u8]; 8] = [
    b"\n<function=",
    b">\n",
    b"<parameter=",
    b">\n",
    b"</function>",
    b"\n</tool_call>",
    b"\n</parameter>\n",
    b"\n",
];

/// the two markers a value may not contain: `toolcall` cuts a value at either
const GUARD: [&[u8]; 2] = [b"</parameter>", b"</tool_call>"];
const GUARD_LEN: u8 = 12;

// ------------------------------------------------------------------ the compiled schema

/// one JSON schema node (the subset in the module table)
#[derive(Debug, Clone, PartialEq)]
enum Node {
    Any,
    Str,
    Int,
    Num,
    Bool,
    Null,
    /// enum / const: the options' compact JSON texts, sorted, deduplicated
    Choice(Vec<Vec<u8>>),
    /// array, element node
    Arr(u16),
    /// strict object: sorted JSON-encoded keys (quotes included), their value nodes,
    /// the required bit set over the sorted key index
    Obj { keys: Vec<Vec<u8>>, vals: Vec<u16>, required: u32 },
    /// any keys, any values
    OpenObj,
}

const N_ANY: u16 = 0;
const N_OPEN_OBJ: u16 = 1;
const N_ANY_ARR: u16 = 2;

/// how one parameter's VALUE is written
#[derive(Debug, Clone, PartialEq)]
enum PKind {
    RawStr,
    /// the raw options, sorted
    RawEnum(Vec<Vec<u8>>),
    Json(u16),
}

#[derive(Debug, Clone)]
struct Tool {
    /// sorted parameter names
    pnames: Vec<Vec<u8>>,
    kinds: Vec<PKind>,
    /// bit i = pnames[i] is required
    required: u64,
}

/// `tool_choice`, as far as the grammar is concerned (`none` builds no grammar)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Mode {
    /// lazy: prose free, calls constrained, end of generation free outside a call
    Auto,
    /// lazy, and end of generation refused until one call closed
    Required,
}

/// the grammar of ONE request: the declared tools compiled
#[derive(Debug, Clone)]
pub struct ToolGrammar {
    /// sorted tool names
    names: Vec<Vec<u8>>,
    tools: Vec<Tool>,
    nodes: Vec<Node>,
    mode: Mode,
    parallel: bool,
}

// ------------------------------------------------------------------ the matcher state

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
struct Frame {
    node: u16,
    st: u8,
    key: u8,
    seen: u32,
}

const OBJ_FIRST: u8 = 0; // after `{`: key or `}`
const OBJ_KEY: u8 = 1; // after `,`: key
const OBJ_COLON: u8 = 2; // after a key: `:`
const OBJ_VAL: u8 = 3; // after `:`: value
const OBJ_NEXT: u8 = 4; // after a value: `,` or `}`
const ARR_FIRST: u8 = 5; // after `[`: value or `]`
const ARR_VAL: u8 = 6; // after `,`: value
const ARR_NEXT: u8 = 7; // after a value: `,` or `]`

/// the scalar being scanned inside the JSON machine
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Sc {
    None,
    /// a free JSON string, after its opening quote: guard progress, escape state
    /// (1 = after `\`, 10+k = k hex digits of `\u` still owed)
    Str { g: u8, pat: u8, esc: u8 },
    /// a strict object's key, a choice over its encoded keys
    Key { lo: u8, hi: u8, off: u16 },
    /// an enum / const value
    Ch { node: u16, lo: u16, hi: u16, off: u16 },
    /// a number; `ph` is the position in the number automaton, `n` the digit count
    Num { int: bool, ph: u8, n: u8 },
    /// `true` / `false` / `null`
    Lit { which: u8, off: u8 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct Js {
    top: u16,
    depth: u8,
    done: bool,
    ws: u8,
    sc: Sc,
    fr: [Frame; MAX_JSON_DEPTH],
}

impl Js {
    const EMPTY: Js = Js {
        top: 0,
        depth: 0,
        done: false,
        ws: 0,
        sc: Sc::None,
        fr: [Frame { node: 0, st: 0, key: 0, seen: 0 }; MAX_JSON_DEPTH],
    };
    fn new(top: u16) -> Js {
        Js { top, ..Js::EMPTY }
    }
}

/// where the frame machine stands
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum Ph {
    /// outside any call: every byte is free
    Idle,
    /// matching `LITS[id]`, `off` bytes done
    Lit { id: u8, off: u8 },
    FName { lo: u16, hi: u16, off: u8 },
    /// inside the function block: `<`
    Body,
    /// after `<`: `p` (a parameter) or `/` (`</function>`)
    BodyLt,
    PName { lo: u16, hi: u16, off: u8 },
    /// a free string value: guard progress `g` in `GUARD[pat]`, `nlb` = a newline stood
    /// before the guard's `<`, `nl` = the last byte was a newline
    RawStr { g: u8, pat: u8, nlb: bool, nl: bool },
    REnum { lo: u16, hi: u16, off: u16 },
    /// a JSON value, in `St::j`
    Json,
    /// after a JSON value: `[ \t]*` then `\n</parameter>\n`
    JTail { ws: u8 },
    /// after `</tool_call>`: whitespace, then EOS or (parallel) the next `<tool_call>`
    Between { ws: u8, nls: u8 },
}

/// the matcher state of one generation; `Copy`, and `Hash` so masks can be cached per state
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct St {
    ph: Ph,
    tool: u16,
    param: u16,
    seen: u64,
    j: Js,
}

impl St {
    /// the state of a generation that has not opened a call
    pub const IDLE: St = St { ph: Ph::Idle, tool: 0, param: 0, seen: 0, j: Js::EMPTY };

    /// inside a call (or between calls): the grammar constrains the next token
    pub fn in_call(&self) -> bool {
        self.ph != Ph::Idle
    }
}

/// a choice step over sorted byte strings sharing the prefix `[..off]`
enum ChoiceR {
    /// consumed; the new range
    Cont(usize, usize),
    /// the option `opts[i]` is complete and `b` does not continue any live option;
    /// `b` was NOT consumed
    Done(usize),
    Rej,
}

fn choice_step(opts: &[Vec<u8>], lo: usize, hi: usize, off: usize, b: u8, alive: impl Fn(usize) -> bool) -> ChoiceR {
    let r = &opts[lo..hi];
    // options longer than `off` whose byte at `off` is `b`: contiguous, sorted
    let a = lo + r.partition_point(|o| o.len() <= off || o[off] < b);
    let z = lo + r.partition_point(|o| o.len() <= off || o[off] <= b);
    // `partition_point` needs a monotone predicate: options of length `off` sort FIRST
    // (a prefix sorts before its extensions), so both predicates are monotone
    if a < z && (a..z).any(&alive) {
        return ChoiceR::Cont(a, z);
    }
    if opts[lo].len() == off && alive(lo) {
        return ChoiceR::Done(lo);
    }
    ChoiceR::Rej
}

/// the next state of a guard scan (`</parameter>` / `</tool_call>`), `Err(pat)` when a
/// marker just COMPLETED
fn guard_step(g: u8, pat: u8, b: u8) -> Result<(u8, u8), u8> {
    if g > 0 {
        let hit = match g {
            1 => (b == b'/').then_some(0),
            2 => match b {
                b'p' => Some(0),
                b't' => Some(1),
                _ => None,
            },
            _ => (GUARD[pat as usize][g as usize] == b).then_some(pat),
        };
        if let Some(p) = hit {
            let g = g + 1;
            if g == GUARD_LEN {
                return Err(p);
            }
            return Ok((g, p));
        }
    }
    if b == b'<' {
        Ok((1, 0))
    } else {
        Ok((0, 0))
    }
}

fn is_ws(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

/// the result of one JSON byte
enum R {
    Ok,
    Rej,
    /// the top-level value is complete and `b` is not part of it
    Pass,
}

// ------------------------------------------------------------------ building

fn ty_list(s: &serde_json::Value) -> Vec<&str> {
    match s.get("type") {
        Some(serde_json::Value::String(t)) => vec![t.as_str()],
        Some(serde_json::Value::Array(a)) => a.iter().filter_map(|t| t.as_str()).collect(),
        _ => Vec::new(),
    }
}

/// llama.cpp `common_schema_info::resolves_to_string` without `$ref` resolution
fn resolves_to_string(s: &serde_json::Value) -> bool {
    if !s.is_object() {
        return false;
    }
    if ty_list(s).contains(&"string") {
        return true;
    }
    for k in ["oneOf", "anyOf"] {
        if let Some(a) = s.get(k).and_then(|v| v.as_array()) {
            if a.iter().any(resolves_to_string) {
                return true;
            }
        }
    }
    if let Some(a) = s.get("allOf").and_then(|v| v.as_array()) {
        if !a.is_empty() && a.iter().all(resolves_to_string) {
            return true;
        }
    }
    matches!(s.get("const"), Some(serde_json::Value::String(_)))
}

impl ToolGrammar {
    /// - compile the request's `tools` array; `only` restricts it to one named function
    ///   (`tool_choice: {"type":"function","function":{"name":...}}`)
    /// - `Err` names why no grammar can be built (no usable tool, a tool with more than
    ///   64 parameters, ...); the caller runs the request unconstrained and says so
    pub fn build(tools: &serde_json::Value, mode: Mode, parallel: bool, only: Option<&str>) -> Result<Self, String> {
        let arr = tools.as_array().ok_or("tools is not an array")?;
        let mut g = ToolGrammar { names: Vec::new(), tools: Vec::new(), nodes: Vec::new(), mode, parallel };
        g.nodes.push(Node::Any);
        g.nodes.push(Node::OpenObj);
        g.nodes.push(Node::Arr(N_ANY));
        let mut by_name: Vec<(Vec<u8>, Tool)> = Vec::new();
        for t in arr {
            let Some(f) = t.get("function") else { continue };
            let Some(name) = f.get("name").and_then(|n| n.as_str()) else { continue };
            if only.is_some_and(|o| o != name) {
                continue;
            }
            if name.is_empty() || name.len() > 255 || name.contains('>') || name.contains('\n') {
                return Err(format!("tool name {name:?} cannot be written in the markup"));
            }
            if by_name.iter().any(|(n, _)| n == name.as_bytes()) {
                continue; // first declaration wins, as the parser's HashMap insert order does not matter
            }
            let params = f.get("parameters");
            let props = params.and_then(|p| p.get("properties")).and_then(|p| p.as_object());
            let req: Vec<&str> = params
                .and_then(|p| p.get("required"))
                .and_then(|r| r.as_array())
                .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                .unwrap_or_default();
            let mut ps: Vec<(Vec<u8>, &serde_json::Value)> = Vec::new();
            if let Some(props) = props {
                for (k, v) in props {
                    if k.is_empty() || k.len() > 255 || k.contains('>') || k.contains('\n') {
                        return Err(format!("parameter {k:?} of {name:?} cannot be written in the markup"));
                    }
                    ps.push((k.as_bytes().to_vec(), v));
                }
            }
            if ps.len() > 64 {
                return Err(format!("tool {name:?} declares {} parameters (at most 64)", ps.len()));
            }
            ps.sort_by(|a, b| a.0.cmp(&b.0));
            let mut tool = Tool { pnames: Vec::new(), kinds: Vec::new(), required: 0 };
            for (i, (k, v)) in ps.into_iter().enumerate() {
                let kind = g.param_kind(v)?;
                if req.iter().any(|r| r.as_bytes() == k.as_slice()) {
                    tool.required |= 1u64 << i;
                }
                tool.pnames.push(k);
                tool.kinds.push(kind);
            }
            by_name.push((name.as_bytes().to_vec(), tool));
        }
        if by_name.is_empty() {
            return Err(match only {
                Some(o) => format!("tool_choice names {o:?}, which no tool declares"),
                None => "no tool with a function name".to_string(),
            });
        }
        by_name.sort_by(|a, b| a.0.cmp(&b.0));
        for (n, t) in by_name {
            g.names.push(n);
            g.tools.push(t);
        }
        Ok(g)
    }

    fn param_kind(&mut self, s: &serde_json::Value) -> Result<PKind, String> {
        if let Some(opts) = Self::raw_enum(s) {
            return Ok(PKind::RawEnum(opts));
        }
        if resolves_to_string(s) {
            return Ok(PKind::RawStr);
        }
        Ok(PKind::Json(self.node(s, 0)?))
    }

    /// a string enum whose options are all one-line strings, written raw
    fn raw_enum(s: &serde_json::Value) -> Option<Vec<Vec<u8>>> {
        let tl = ty_list(s);
        if !(tl.is_empty() || tl == ["string"]) {
            return None;
        }
        let e = s.get("enum")?.as_array()?;
        let mut v = Vec::new();
        for o in e {
            let o = o.as_str()?;
            if o.contains('\n') {
                return None;
            }
            v.push(o.as_bytes().to_vec());
        }
        if v.is_empty() {
            return None;
        }
        v.sort();
        v.dedup();
        Some(v)
    }

    fn push(&mut self, n: Node) -> Result<u16, String> {
        if self.nodes.len() >= u16::MAX as usize {
            return Err("schema too large".to_string());
        }
        self.nodes.push(n);
        Ok((self.nodes.len() - 1) as u16)
    }

    /// compile one schema at container nesting `d`
    fn node(&mut self, s: &serde_json::Value, d: usize) -> Result<u16, String> {
        if !s.is_object() {
            return Ok(N_ANY);
        }
        let choice = |vals: &[serde_json::Value]| -> Vec<Vec<u8>> {
            let mut v: Vec<Vec<u8>> = vals.iter().map(|x| x.to_string().into_bytes()).collect();
            v.sort();
            v.dedup();
            v
        };
        if let Some(c) = s.get("const") {
            return self.push(Node::Choice(choice(std::slice::from_ref(c))));
        }
        if let Some(e) = s.get("enum").and_then(|e| e.as_array()) {
            if !e.is_empty() {
                return self.push(Node::Choice(choice(e)));
            }
        }
        let tl = ty_list(s);
        if tl.len() != 1 {
            return Ok(N_ANY); // untyped, a type union, $ref, anyOf...: any JSON value
        }
        match tl[0] {
            "string" => self.push(Node::Str),
            "integer" => self.push(Node::Int),
            "number" => self.push(Node::Num),
            "boolean" => self.push(Node::Bool),
            "null" => self.push(Node::Null),
            "array" => {
                if d + 1 >= MAX_JSON_DEPTH {
                    return Ok(N_ANY);
                }
                match s.get("items") {
                    Some(it) if it.is_object() => {
                        let i = self.node(it, d + 1)?;
                        self.push(Node::Arr(i))
                    }
                    _ => Ok(N_ANY_ARR),
                }
            }
            "object" => {
                if d + 1 >= MAX_JSON_DEPTH {
                    return Ok(N_ANY);
                }
                let props = s.get("properties").and_then(|p| p.as_object());
                let open = match s.get("additionalProperties") {
                    Some(serde_json::Value::Bool(b)) => *b,
                    Some(v) if v.is_object() => true,
                    _ => false,
                };
                match props {
                    Some(p) if !open && !p.is_empty() && p.len() <= 32 => {
                        let req: Vec<&str> = s
                            .get("required")
                            .and_then(|r| r.as_array())
                            .map(|a| a.iter().filter_map(|v| v.as_str()).collect())
                            .unwrap_or_default();
                        let mut kv: Vec<(Vec<u8>, &str, &serde_json::Value)> = p
                            .iter()
                            .map(|(k, v)| (serde_json::Value::String(k.clone()).to_string().into_bytes(), k.as_str(), v))
                            .collect();
                        kv.sort_by(|a, b| a.0.cmp(&b.0));
                        let mut keys = Vec::new();
                        let mut vals = Vec::new();
                        let mut required = 0u32;
                        for (i, (ek, k, v)) in kv.into_iter().enumerate() {
                            if req.contains(&k) {
                                required |= 1 << i;
                            }
                            keys.push(ek);
                            vals.push(self.node(v, d + 1)?);
                        }
                        self.push(Node::Obj { keys, vals, required })
                    }
                    _ => Ok(N_OPEN_OBJ),
                }
            }
            _ => Ok(N_ANY),
        }
    }

    /// declared tools, for the request line
    pub fn n_tools(&self) -> usize {
        self.tools.len()
    }

    /// declared parameters over all tools, for the request line
    pub fn n_params(&self) -> usize {
        self.tools.iter().map(|t| t.pnames.len()).sum()
    }

    pub fn mode(&self) -> Mode {
        self.mode
    }

    pub fn parallel(&self) -> bool {
        self.parallel
    }

    // -------------------------------------------------------------- the byte machine

    fn unseen_param(&self, s: &St) -> bool {
        let n = self.tools[s.tool as usize].pnames.len();
        n > 0 && s.seen.count_ones() < n as u32
    }

    fn after_lit(&self, s: &mut St, id: u8) {
        s.ph = match id {
            L_OPEN => Ph::FName { lo: 0, hi: self.names.len() as u16, off: 0 },
            L_FGT | L_PCLOSE | L_NL => Ph::Body,
            L_PARAM => Ph::PName { lo: 0, hi: self.tools[s.tool as usize].pnames.len() as u16, off: 0 },
            L_PGT => match &self.tools[s.tool as usize].kinds[s.param as usize] {
                PKind::RawStr => Ph::RawStr { g: 0, pat: 0, nlb: false, nl: false },
                PKind::RawEnum(o) => Ph::REnum { lo: 0, hi: o.len() as u16, off: 0 },
                PKind::Json(n) => {
                    s.j = Js::new(*n);
                    Ph::Json
                }
            },
            L_FCLOSE => Ph::Lit { id: L_CALL_CLOSE, off: 0 },
            L_CALL_CLOSE => Ph::Between { ws: 0, nls: 0 },
            _ => unreachable!("literal id"),
        };
    }

    /// one BYTE through the machine; `false` = refused (the state is then unspecified)
    pub fn step(&self, s: &mut St, b: u8) -> bool {
        loop {
            match s.ph {
                Ph::Idle => return true,
                Ph::Lit { id, off } => {
                    let lit = LITS[id as usize];
                    if lit[off as usize] != b {
                        return false;
                    }
                    let off = off + 1;
                    if off as usize == lit.len() {
                        self.after_lit(s, id);
                    } else {
                        s.ph = Ph::Lit { id, off };
                    }
                    return true;
                }
                Ph::FName { lo, hi, off } => {
                    match choice_step(&self.names, lo as usize, hi as usize, off as usize, b, |_| true) {
                        ChoiceR::Cont(a, z) => {
                            s.ph = Ph::FName { lo: a as u16, hi: z as u16, off: off + 1 };
                            return true;
                        }
                        ChoiceR::Done(i) => {
                            s.tool = i as u16;
                            s.seen = 0;
                            s.ph = Ph::Lit { id: L_FGT, off: 0 };
                        }
                        ChoiceR::Rej => return false,
                    }
                }
                Ph::Body => {
                    if b != b'<' {
                        return false;
                    }
                    s.ph = Ph::BodyLt;
                    return true;
                }
                Ph::BodyLt => {
                    s.ph = match b {
                        b'p' if self.unseen_param(s) => Ph::Lit { id: L_PARAM, off: 2 },
                        // the call may close with a required parameter missing: the client
                        // reports that as a schema error the model can repair, while refusing
                        // `/` FORCED `<parameter=` plus filler text into the model's context
                        b'/' => Ph::Lit { id: L_FCLOSE, off: 2 },
                        _ => return false,
                    };
                    return true;
                }
                Ph::PName { lo, hi, off } => {
                    let seen = s.seen;
                    let t = &self.tools[s.tool as usize];
                    match choice_step(&t.pnames, lo as usize, hi as usize, off as usize, b, |i| (seen & (1u64 << i)) == 0) {
                        ChoiceR::Cont(a, z) => {
                            s.ph = Ph::PName { lo: a as u16, hi: z as u16, off: off + 1 };
                            return true;
                        }
                        ChoiceR::Done(i) => {
                            s.param = i as u16;
                            s.seen |= 1u64 << i;
                            s.ph = Ph::Lit { id: L_PGT, off: 0 };
                        }
                        ChoiceR::Rej => return false,
                    }
                }
                Ph::RawStr { g, pat, nlb, nl } => {
                    match guard_step(g, pat, b) {
                        Err(p) => {
                            // a marker completed: `</parameter>` closes the value whether or
                            // not a newline stood before it - `toolcall::find_marker` cuts there
                            // either way, so refusing it only FORCED a junk token into the
                            // model's context (live 2026-09-23: `src/scene.js\nparameter_path`,
                            // a `task` of `parameter`). `</tool_call>` inside a value stays refused.
                            if p == 0 {
                                s.ph = Ph::Lit { id: L_NL, off: 0 };
                                return true;
                            }
                            return false;
                        }
                        Ok((g2, p2)) => {
                            s.ph = if g2 == 0 {
                                Ph::RawStr { g: 0, pat: 0, nlb: false, nl: b == b'\n' }
                            } else if g2 == 1 {
                                // a fresh `<`: remember whether a newline stood before it; after
                                // a mismatch that restarted on this `<`, the byte before is the
                                // mismatching one, never a newline
                                Ph::RawStr { g: 1, pat: 0, nlb: g == 0 && nl, nl: false }
                            } else {
                                Ph::RawStr { g: g2, pat: p2, nlb, nl: false }
                            };
                            return true;
                        }
                    }
                }
                Ph::REnum { lo, hi, off } => {
                    let PKind::RawEnum(opts) = &self.tools[s.tool as usize].kinds[s.param as usize] else {
                        unreachable!("REnum on a non-enum parameter")
                    };
                    match choice_step(opts, lo as usize, hi as usize, off as usize, b, |_| true) {
                        ChoiceR::Cont(a, z) => {
                            s.ph = Ph::REnum { lo: a as u16, hi: z as u16, off: off + 1 };
                            return true;
                        }
                        ChoiceR::Done(_) => s.ph = Ph::Lit { id: L_PCLOSE, off: 0 },
                        ChoiceR::Rej => return false,
                    }
                }
                Ph::Json => match self.j_step(&mut s.j, b) {
                    R::Ok => return true,
                    R::Rej => return false,
                    R::Pass => {
                        s.j = Js::EMPTY;
                        s.ph = Ph::JTail { ws: 0 };
                    }
                },
                Ph::JTail { ws } => {
                    s.ph = match b {
                        b' ' | b'\t' if ws < WS_MAX => Ph::JTail { ws: ws + 1 },
                        b'\n' => Ph::Lit { id: L_PCLOSE, off: 1 },
                        _ => return false,
                    };
                    return true;
                }
                Ph::Between { ws, nls } => {
                    if ws >= BETWEEN_WS_MAX {
                        return false;
                    }
                    s.ph = match b {
                        b' ' | b'\t' => Ph::Between { ws: ws + 1, nls },
                        b'\n' if nls < BETWEEN_NL_MAX => Ph::Between { ws: ws + 1, nls: nls + 1 },
                        _ => return false,
                    };
                    return true;
                }
            }
        }
    }

    // -------------------------------------------------------------- the JSON machine

    fn j_value_done(&self, j: &mut Js) {
        j.sc = Sc::None;
        j.ws = 0;
        if j.depth == 0 {
            j.done = true;
            return;
        }
        let f = &mut j.fr[j.depth as usize - 1];
        match f.st {
            OBJ_VAL => {
                if matches!(self.nodes[f.node as usize], Node::Obj { .. }) {
                    f.seen |= 1u32 << f.key;
                }
                f.st = OBJ_NEXT;
            }
            ARR_FIRST | ARR_VAL => f.st = ARR_NEXT,
            _ => {}
        }
    }

    fn j_push(&self, j: &mut Js, node: u16, st: u8) -> R {
        if j.depth as usize >= MAX_JSON_DEPTH {
            return R::Rej;
        }
        j.fr[j.depth as usize] = Frame { node, st, key: 0, seen: 0 };
        j.depth += 1;
        j.ws = 0;
        R::Ok
    }

    fn j_pop(&self, j: &mut Js) -> R {
        j.depth -= 1;
        j.fr[j.depth as usize] = Frame::default();
        self.j_value_done(j);
        R::Ok
    }

    /// the first byte of a value of `node`
    fn j_start(&self, j: &mut Js, node: u16, b: u8) -> R {
        let n = &self.nodes[node as usize];
        let num = |int: bool, j: &mut Js| -> R {
            j.sc = Sc::Num { int, ph: 0, n: 0 };
            self.j_num(j, int, 0, 0, b)
        };
        match (n, b) {
            (Node::Choice(o), _) => {
                j.sc = Sc::Ch { node, lo: 0, hi: o.len() as u16, off: 0 };
                self.j_choice(j, node, 0, o.len() as u16, 0, b)
            }
            (Node::Any | Node::Str, b'"') => {
                j.sc = Sc::Str { g: 0, pat: 0, esc: 0 };
                R::Ok
            }
            (Node::Any | Node::Num, b'-' | b'0'..=b'9') => num(false, j),
            (Node::Int, b'-' | b'0'..=b'9') => num(true, j),
            (Node::Any | Node::Bool, b't') => {
                j.sc = Sc::Lit { which: 0, off: 1 };
                R::Ok
            }
            (Node::Any | Node::Bool, b'f') => {
                j.sc = Sc::Lit { which: 1, off: 1 };
                R::Ok
            }
            (Node::Any | Node::Null, b'n') => {
                j.sc = Sc::Lit { which: 2, off: 1 };
                R::Ok
            }
            (Node::Any, b'{') => self.j_push(j, N_OPEN_OBJ, OBJ_FIRST),
            (Node::Any, b'[') => self.j_push(j, N_ANY_ARR, ARR_FIRST),
            (Node::Obj { .. } | Node::OpenObj, b'{') => self.j_push(j, node, OBJ_FIRST),
            (Node::Arr(_), b'[') => self.j_push(j, node, ARR_FIRST),
            _ => R::Rej,
        }
    }

    fn j_choice(&self, j: &mut Js, node: u16, lo: u16, hi: u16, off: u16, b: u8) -> R {
        let Node::Choice(o) = &self.nodes[node as usize] else { unreachable!("Ch on a non-choice node") };
        match choice_step(o, lo as usize, hi as usize, off as usize, b, |_| true) {
            ChoiceR::Cont(a, z) => {
                let off = off + 1;
                // a string option ends with its quote and is no prefix of another: done now
                if z - a == 1 && o[a].len() == off as usize && o[a].last() == Some(&b'"') {
                    self.j_value_done(j);
                } else {
                    j.sc = Sc::Ch { node, lo: a as u16, hi: z as u16, off };
                }
                R::Ok
            }
            ChoiceR::Done(_) => {
                if off == 0 {
                    return R::Rej; // an empty option cannot be written
                }
                self.j_value_done(j);
                self.j_struct(j, b)
            }
            ChoiceR::Rej => R::Rej,
        }
    }

    /// the number automaton: 0 start, 1 after `-`, 2 int `0`, 3 int digits, 4 after `.`,
    /// 5 fraction digits, 6 after `e`, 7 after the exponent sign, 8 exponent `0`,
    /// 9 exponent digits. Accepting: 2, 3, 5, 8, 9. Digit runs cap at 16 (llama.cpp).
    fn j_num(&self, j: &mut Js, int: bool, ph: u8, n: u8, b: u8) -> R {
        let d = b.is_ascii_digit();
        let next = match (ph, b) {
            (0, b'-') => Some((1, 0)),
            (0 | 1, b'0') => Some((2, 1)),
            (0 | 1, b'1'..=b'9') => Some((3, 1)),
            (3, _) if d && n < 16 => Some((3, n + 1)),
            (2 | 3, b'.') if !int => Some((4, 0)),
            (2 | 3 | 5, b'e' | b'E') if !int => Some((6, 0)),
            (4, _) if d => Some((5, 1)),
            (5, _) if d && n < 16 => Some((5, n + 1)),
            (6, b'+' | b'-') => Some((7, 0)),
            (6 | 7, b'0') => Some((8, 1)),
            (6 | 7, b'1'..=b'9') => Some((9, 1)),
            (9, _) if d && n < 16 => Some((9, n + 1)),
            _ => None,
        };
        if let Some((ph, n)) = next {
            j.sc = Sc::Num { int, ph, n };
            return R::Ok;
        }
        if matches!(ph, 2 | 3 | 5 | 8 | 9) {
            self.j_value_done(j);
            return self.j_struct(j, b);
        }
        R::Rej
    }

    fn j_step(&self, j: &mut Js, b: u8) -> R {
        match j.sc {
            Sc::None => self.j_struct(j, b),
            Sc::Str { g, pat, esc } => {
                if esc == 1 {
                    j.sc = match b {
                        b'"' | b'\\' | b'/' | b'b' | b'f' | b'n' | b'r' | b't' => Sc::Str { g: 0, pat: 0, esc: 0 },
                        b'u' => Sc::Str { g: 0, pat: 0, esc: 14 },
                        _ => return R::Rej,
                    };
                    return R::Ok;
                }
                if esc >= 10 {
                    if !b.is_ascii_hexdigit() {
                        return R::Rej;
                    }
                    j.sc = Sc::Str { g: 0, pat: 0, esc: if esc == 11 { 0 } else { esc - 1 } };
                    return R::Ok;
                }
                match b {
                    b'"' => {
                        let key = j.depth > 0 && matches!(j.fr[j.depth as usize - 1].st, OBJ_FIRST | OBJ_KEY);
                        if key {
                            j.fr[j.depth as usize - 1].st = OBJ_COLON;
                            j.sc = Sc::None;
                            j.ws = 0;
                        } else {
                            self.j_value_done(j);
                        }
                        R::Ok
                    }
                    b'\\' => {
                        j.sc = Sc::Str { g: 0, pat: 0, esc: 1 };
                        R::Ok
                    }
                    0..=0x1f => R::Rej,
                    _ => match guard_step(g, pat, b) {
                        Ok((g, pat)) => {
                            j.sc = Sc::Str { g, pat, esc: 0 };
                            R::Ok
                        }
                        Err(_) => R::Rej,
                    },
                }
            }
            Sc::Key { lo, hi, off } => {
                let f = j.fr[j.depth as usize - 1];
                let Node::Obj { keys, .. } = &self.nodes[f.node as usize] else { unreachable!("Key on a non-object") };
                match choice_step(keys, lo as usize, hi as usize, off as usize, b, |i| (f.seen & (1u32 << i)) == 0) {
                    ChoiceR::Cont(a, z) => {
                        let off = off + 1;
                        if keys[a].len() == off as usize {
                            // an encoded key ends with its quote: complete, and unique
                            let top = &mut j.fr[j.depth as usize - 1];
                            top.key = a as u8;
                            top.st = OBJ_COLON;
                            j.sc = Sc::None;
                            j.ws = 0;
                        } else {
                            j.sc = Sc::Key { lo: a as u8, hi: z as u8, off };
                        }
                        R::Ok
                    }
                    _ => R::Rej,
                }
            }
            Sc::Ch { node, lo, hi, off } => self.j_choice(j, node, lo, hi, off, b),
            Sc::Num { int, ph, n } => self.j_num(j, int, ph, n, b),
            Sc::Lit { which, off } => {
                let lit: &[u8] = [&b"true"[..], b"false", b"null"][which as usize];
                if lit[off as usize] != b {
                    return R::Rej;
                }
                if off as usize + 1 == lit.len() {
                    self.j_value_done(j);
                } else {
                    j.sc = Sc::Lit { which, off: off + 1 };
                }
                R::Ok
            }
        }
    }

    /// a byte between scalars
    fn j_struct(&self, j: &mut Js, b: u8) -> R {
        if j.depth == 0 {
            if j.done {
                return R::Pass;
            }
            return self.j_start(j, j.top, b); // no leading whitespace (llama.cpp)
        }
        if is_ws(b) {
            if j.ws >= WS_MAX {
                return R::Rej;
            }
            j.ws += 1;
            return R::Ok;
        }
        j.ws = 0;
        let di = j.depth as usize - 1;
        let f = j.fr[di];
        let node = &self.nodes[f.node as usize];
        match f.st {
            OBJ_FIRST | OBJ_KEY => {
                if b == b'}' && f.st == OBJ_FIRST {
                    return self.j_close_obj(j);
                }
                if b != b'"' {
                    return R::Rej;
                }
                match node {
                    Node::Obj { keys, .. } => {
                        let n = keys.len();
                        match choice_step(keys, 0, n, 0, b, |i| (f.seen & (1u32 << i)) == 0) {
                            ChoiceR::Cont(a, z) => {
                                j.sc = Sc::Key { lo: a as u8, hi: z as u8, off: 1 };
                                R::Ok
                            }
                            _ => R::Rej,
                        }
                    }
                    _ => {
                        j.sc = Sc::Str { g: 0, pat: 0, esc: 0 };
                        R::Ok
                    }
                }
            }
            OBJ_COLON => {
                if b != b':' {
                    return R::Rej;
                }
                j.fr[di].st = OBJ_VAL;
                R::Ok
            }
            OBJ_VAL => {
                let vn = match node {
                    Node::Obj { vals, .. } => vals[f.key as usize],
                    _ => N_ANY,
                };
                self.j_start(j, vn, b)
            }
            OBJ_NEXT => match b {
                b'}' => self.j_close_obj(j),
                b',' => {
                    if let Node::Obj { keys, .. } = node {
                        // a comma must leave a key to write
                        if f.seen.count_ones() as usize >= keys.len() {
                            return R::Rej;
                        }
                    }
                    j.fr[di].st = OBJ_KEY;
                    R::Ok
                }
                _ => R::Rej,
            },
            ARR_FIRST | ARR_VAL => {
                if b == b']' && f.st == ARR_FIRST {
                    return self.j_pop(j);
                }
                let Node::Arr(item) = node else { unreachable!("array frame on a non-array") };
                self.j_start(j, *item, b)
            }
            ARR_NEXT => match b {
                b']' => self.j_pop(j),
                b',' => {
                    j.fr[di].st = ARR_VAL;
                    R::Ok
                }
                _ => R::Rej,
            },
            _ => R::Rej,
        }
    }

    fn j_close_obj(&self, j: &mut Js) -> R {
        let f = j.fr[j.depth as usize - 1];
        if let Node::Obj { required, .. } = &self.nodes[f.node as usize] {
            if (f.seen & required) != *required {
                return R::Rej;
            }
        }
        self.j_pop(j)
    }

    // -------------------------------------------------------------- token level

    /// end of generation is allowed in this state
    pub fn eos_ok(&self, s: &St) -> bool {
        match s.ph {
            Ph::Idle => self.mode == Mode::Auto,
            Ph::Between { .. } => true,
            _ => false,
        }
    }

    /// the `<tool_call>` token is allowed in this state
    pub fn open_ok(&self, s: &St) -> bool {
        match s.ph {
            Ph::Idle => true,
            Ph::Between { .. } => self.parallel,
            _ => false,
        }
    }
}

// ------------------------------------------------------------------ the vocabulary

/// the vocabulary as the grammar sees it: every id's bytes, the special ids, and a
/// preorder byte trie of every id that may appear inside a call
pub struct Vocab {
    n: usize,
    eos: Vec<u32>,
    tool_open: u32,
    /// bytes of id i: `bytes[off[i]..off[i+1]]`
    off: Vec<u32>,
    bytes: Vec<u8>,
    /// ids that are never allowed inside a call by their bytes: special tokens, ids
    /// without bytes, EOS and `<tool_call>` (those two go by id)
    by_id_only: Vec<bool>,
    // the trie, nodes in preorder; node 0 is the root
    t_depth: Vec<u16>,
    t_byte: Vec<u8>,
    t_end: Vec<u32>,
    /// ids ending at node i: `t_toks[t_tok[i]..t_tok[i+1]]`
    t_tok: Vec<u32>,
    t_toks: Vec<u32>,
    max_len: usize,
}

impl Vocab {
    /// - `n` ids; `bytes_of(id)` the exact bytes an id decodes to, `special(id)` whether
    ///   the decoder SKIPS it (`decode(.., skip_special_tokens = true)` in `tokenizer.rs`)
    pub fn build(n: usize, bytes_of: impl Fn(u32) -> Vec<u8>, special: impl Fn(u32) -> bool, eos: &[u32], tool_open: u32) -> Vocab {
        let mut off = Vec::with_capacity(n + 1);
        let mut bytes = Vec::new();
        let mut by_id_only = vec![false; n];
        off.push(0u32);
        for id in 0..n as u32 {
            let b = bytes_of(id);
            by_id_only[id as usize] = b.is_empty() || special(id) || eos.contains(&id) || id == tool_open;
            bytes.extend_from_slice(&b);
            off.push(bytes.len() as u32);
        }
        let mut ids: Vec<u32> = (0..n as u32).filter(|&i| !by_id_only[i as usize]).collect();
        let bo = |i: u32| &bytes[off[i as usize] as usize..off[i as usize + 1] as usize];
        ids.sort_by(|&a, &b| bo(a).cmp(bo(b)).then(a.cmp(&b)));
        let mut v = Vocab {
            n,
            eos: eos.to_vec(),
            tool_open,
            t_depth: vec![0],
            t_byte: vec![0],
            t_end: vec![0],
            t_tok: vec![0],
            t_toks: Vec::with_capacity(ids.len()),
            max_len: 0,
            off: Vec::new(),
            bytes: Vec::new(),
            by_id_only: Vec::new(),
        };
        let mut stack: Vec<usize> = vec![0];
        let mut prev: &[u8] = &[];
        for &id in &ids {
            let cur = bo(id);
            let lcp = prev.iter().zip(cur).take_while(|(a, b)| a == b).count();
            while stack.len() - 1 > lcp {
                let nd = stack.pop().expect("non-empty");
                v.t_end[nd] = v.t_depth.len() as u32;
            }
            for (k, &byte) in cur.iter().enumerate().skip(lcp) {
                v.t_depth.push((k + 1) as u16);
                v.t_byte.push(byte);
                v.t_end.push(0);
                v.t_tok.push(v.t_toks.len() as u32);
                stack.push(v.t_depth.len() - 1);
            }
            // a duplicate byte string lands on the node created last (sorted order)
            v.t_toks.push(id);
            v.max_len = v.max_len.max(cur.len());
            prev = cur;
        }
        while let Some(nd) = stack.pop() {
            v.t_end[nd] = v.t_depth.len() as u32;
        }
        v.t_tok.push(v.t_toks.len() as u32);
        v.off = off;
        v.bytes = bytes;
        v.by_id_only = by_id_only;
        v
    }

    /// ids in the vocabulary
    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// trie nodes (the root included)
    pub fn trie_nodes(&self) -> usize {
        self.t_depth.len()
    }

    fn bytes_of(&self, id: u32) -> &[u8] {
        &self.bytes[self.off[id as usize] as usize..self.off[id as usize + 1] as usize]
    }

    /// may `id` be generated in state `s`? The per-token check of the rejection path:
    /// one pass over the id's bytes, no allocation.
    pub fn token_ok(&self, g: &ToolGrammar, s: &St, id: u32) -> bool {
        if (id as usize) >= self.n {
            return false;
        }
        if self.eos.contains(&id) {
            return g.eos_ok(s);
        }
        if id == self.tool_open {
            return g.open_ok(s);
        }
        if s.ph == Ph::Idle {
            return true;
        }
        if self.by_id_only[id as usize] {
            return false;
        }
        let mut t = *s;
        self.bytes_of(id).iter().all(|&b| g.step(&mut t, b))
    }

    /// advance `s` by an id that was GENERATED (sampled, or forced by #81). `false`: the id
    /// is not in the grammar here; `s` is then left unchanged and the caller decides.
    pub fn accept(&self, g: &ToolGrammar, s: &mut St, id: u32) -> bool {
        if !self.token_ok(g, s, id) {
            return false;
        }
        if id == self.tool_open {
            *s = St { ph: Ph::Lit { id: L_OPEN, off: 0 }, ..St::IDLE };
            return true;
        }
        if s.ph == Ph::Idle || self.eos.contains(&id) {
            return true;
        }
        for &b in self.bytes_of(id) {
            g.step(s, b);
        }
        true
    }

    /// - the full mask of state `s`: bit `id` set = `token_ok(g, s, id)`
    /// - a walk over the preorder trie: a refused byte skips its subtree
    pub fn mask(&self, g: &ToolGrammar, s: &St, bits: &mut Vec<u64>) {
        bits.clear();
        bits.resize(self.n.div_ceil(64), 0);
        let mut set = |id: u32| bits[id as usize / 64] |= 1u64 << (id % 64);
        for &e in &self.eos {
            if (e as usize) < self.n && g.eos_ok(s) {
                set(e);
            }
        }
        if (self.tool_open as usize) < self.n && g.open_ok(s) {
            set(self.tool_open);
        }
        if s.ph == Ph::Idle {
            for &id in &self.t_toks {
                set(id);
            }
            // ids by id only that are neither EOS nor the opener: free text outside a call
            for id in 0..self.n as u32 {
                if self.by_id_only[id as usize] && !self.eos.contains(&id) && id != self.tool_open {
                    set(id);
                }
            }
            return;
        }
        let mut stack: Vec<St> = vec![*s; self.max_len + 1];
        let mut i = 1usize;
        let n_nodes = self.t_depth.len();
        while i < n_nodes {
            let d = self.t_depth[i] as usize;
            let mut t = stack[d - 1];
            if g.step(&mut t, self.t_byte[i]) {
                stack[d] = t;
                for k in self.t_tok[i]..self.t_tok[i + 1] {
                    set(self.t_toks[k as usize]);
                }
                i += 1;
            } else {
                i = self.t_end[i] as usize;
            }
        }
    }
}

/// is bit `id` set in a mask
pub fn allowed(bits: &[u64], id: usize) -> bool {
    bits.get(id / 64).is_some_and(|w| ((w >> (id % 64)) & 1) == 1)
}

// ------------------------------------------------------------------ the per-request gate

/// what the gate did over one request, for the `[chat]` line
#[derive(Debug, Default, Clone, Copy)]
pub struct GateStats {
    /// generated ids checked while a call was open (or, `required`, before one)
    pub checked: usize,
    /// of those, refused by the grammar and redrawn under the mask
    pub redrawn: usize,
    /// masks built (the rest came from the cache)
    pub masks_built: usize,
    /// wall of the mask builds, ms
    pub mask_ms: f64,
    /// calls the grammar saw close
    pub calls_closed: usize,
    /// a forced id (#81) the grammar did not admit: the grammar stepped aside
    pub stepped_aside: bool,
}

/// the grammar of one request, its state and its mask cache
pub struct Gate<'v> {
    pub g: ToolGrammar,
    pub v: &'v Vocab,
    pub st: St,
    cache: HashMap<St, std::sync::Arc<Vec<u64>>>,
    pub stats: GateStats,
    /// the grammar gave up (a forced id outside it); from here on nothing is checked
    off: bool,
}

/// masks kept per request; a mask is V/8 = 31 KB
const CACHE_MAX: usize = 256;

impl<'v> Gate<'v> {
    pub fn new(g: ToolGrammar, v: &'v Vocab) -> Self {
        Gate { g, v, st: St::IDLE, cache: HashMap::new(), stats: GateStats::default(), off: false }
    }

    /// does the NEXT id need a check at all? `false` outside a call in `auto` mode:
    /// the whole cost of the grammar on free text is this one branch
    pub fn armed(&self) -> bool {
        !self.off && (self.st.in_call() || self.g.mode == Mode::Required)
    }

    /// the per-token check (counted)
    pub fn check(&mut self, id: u32) -> bool {
        self.stats.checked += 1;
        self.v.token_ok(&self.g, &self.st, id)
    }

    /// the mask of the current state, cached
    pub fn mask(&mut self) -> std::sync::Arc<Vec<u64>> {
        if let Some(m) = self.cache.get(&self.st) {
            return m.clone();
        }
        let t = std::time::Instant::now();
        let mut bits = Vec::new();
        self.v.mask(&self.g, &self.st, &mut bits);
        self.stats.mask_ms += t.elapsed().as_secs_f64() * 1e3;
        self.stats.masks_built += 1;
        if self.cache.len() >= CACHE_MAX {
            self.cache.clear();
        }
        let m = std::sync::Arc::new(bits);
        self.cache.insert(self.st, m.clone());
        m
    }

    /// `-inf` on every id the grammar refuses here
    pub fn mask_row(&mut self, row: &mut [f32]) {
        let m = self.mask();
        for (i, l) in row.iter_mut().enumerate() {
            if !allowed(&m, i) {
                *l = f32::NEG_INFINITY;
            }
        }
    }

    /// advance by a generated id; a refused (forced) id switches the gate off for the rest
    /// of the request and returns `false`
    pub fn accept(&mut self, id: u32) -> bool {
        if self.off {
            return true;
        }
        let was_between = matches!(self.st.ph, Ph::Between { .. });
        let before = self.st.ph;
        if self.v.accept(&self.g, &mut self.st, id) {
            if matches!(self.st.ph, Ph::Between { .. }) && !was_between && before != Ph::Idle {
                self.stats.calls_closed += 1;
            }
            return true;
        }
        self.off = true;
        self.stats.stepped_aside = true;
        false
    }

    /// the gate gave up on this request
    pub fn is_off(&self) -> bool {
        self.off
    }

    /// a short name of the current phase, for log lines
    pub fn phase(&self) -> &'static str {
        match self.st.ph {
            Ph::Idle => "idle",
            Ph::Lit { .. } => "markup",
            Ph::FName { .. } => "function name",
            Ph::Body | Ph::BodyLt => "function body",
            Ph::PName { .. } => "parameter name",
            Ph::RawStr { .. } => "string value",
            Ph::REnum { .. } => "enum value",
            Ph::Json | Ph::JTail { .. } => "JSON value",
            Ph::Between { .. } => "after </tool_call>",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const TOOLS_3DBC015: &str = include_str!("../../tools/corpora/crow-3dbc015-tools.json");

    fn crow_tools() -> serde_json::Value {
        serde_json::from_str(TOOLS_3DBC015).expect("fixture parses")
    }

    fn crow() -> ToolGrammar {
        ToolGrammar::build(&crow_tools(), Mode::Auto, true, None).expect("builds")
    }

    /// the byte machine alone, from just after the `<tool_call>` token; `Ok(state)` when
    /// every byte stepped, `Err(offset)` at the first refused byte
    fn run(g: &ToolGrammar, text: &str) -> Result<St, usize> {
        let mut s = St { ph: Ph::Lit { id: L_OPEN, off: 0 }, ..St::IDLE };
        for (i, &b) in text.as_bytes().iter().enumerate() {
            if !g.step(&mut s, b) {
                return Err(i);
            }
        }
        Ok(s)
    }

    fn complete(g: &ToolGrammar, text: &str) -> bool {
        matches!(run(g, text), Ok(s) if g.eos_ok(&s))
    }

    // ------------------------------------------------------------- construction

    #[test]
    fn the_crow_tools_compile_to_26_tools_with_their_required_sets() {
        let g = crow();
        assert_eq!(g.n_tools(), 26);
        assert_eq!(g.n_params(), 51);
        let edit = g.names.iter().position(|n| n == b"edit_file").expect("edit_file");
        let t = &g.tools[edit];
        assert_eq!(t.pnames, vec![b"new".to_vec(), b"old".to_vec(), b"path".to_vec()]);
        assert_eq!(t.required, 0b111);
        assert!(t.kinds.iter().all(|k| *k == PKind::RawStr));
        let rf = g.names.iter().position(|n| n == b"read_file").expect("read_file");
        let t = &g.tools[rf];
        assert_eq!(t.pnames, vec![b"end_line".to_vec(), b"path".to_vec(), b"start_line".to_vec()]);
        assert_eq!(t.required, 0b010);
        assert!(matches!(t.kinds[0], PKind::Json(n) if g.nodes[n as usize] == Node::Int));
        let gc = g.names.iter().position(|n| n == b"git_commit").expect("git_commit");
        let PKind::Json(n) = g.tools[gc].kinds[1] else { panic!("paths is JSON") };
        let Node::Arr(item) = g.nodes[n as usize] else { panic!("paths is an array") };
        assert_eq!(g.nodes[item as usize], Node::Str);
        let gd = g.names.iter().position(|n| n == b"git_diff").expect("git_diff");
        assert!(matches!(g.tools[gd].kinds[1], PKind::Json(n) if g.nodes[n as usize] == Node::Bool));
    }

    #[test]
    fn schema_edge_cases_compile_to_the_documented_kinds() {
        let tools = serde_json::json!([{"type":"function","function":{"name":"f","parameters":{"type":"object",
            "properties":{
                "mode":{"type":"string","enum":["fast","slow"]},
                "opt":{"type":["string","null"]},
                "any":{},
                "lvl":{"enum":[1,2,12]},
                "obj":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]},
                "free":{"type":"object"}
            },"required":["mode"]}}}]);
        let g = ToolGrammar::build(&tools, Mode::Auto, true, None).unwrap();
        let t = &g.tools[0];
        let k = |name: &str| &t.kinds[t.pnames.iter().position(|p| p == name.as_bytes()).unwrap()];
        assert_eq!(*k("mode"), PKind::RawEnum(vec![b"fast".to_vec(), b"slow".to_vec()]));
        assert_eq!(*k("opt"), PKind::RawStr);
        assert_eq!(*k("any"), PKind::Json(N_ANY));
        assert_eq!(*k("free"), PKind::Json(N_OPEN_OBJ));
        assert!(matches!(k("lvl"), PKind::Json(n) if matches!(&g.nodes[*n as usize], Node::Choice(o) if o.len() == 3)));
        // tool_choice by name: only that tool
        assert!(ToolGrammar::build(&crow_tools(), Mode::Required, true, Some("nope")).is_err());
        let one = ToolGrammar::build(&crow_tools(), Mode::Required, true, Some("read_file")).unwrap();
        assert_eq!(one.n_tools(), 1);
    }

    // ------------------------------------------------------------- the byte machine

    #[test]
    fn well_formed_crow_calls_are_accepted_and_complete() {
        let g = crow();
        for text in [
            "\n<function=read_file>\n<parameter=path>\n/home/nibor1896/x.md\n</parameter>\n</function>\n</tool_call>",
            "\n<function=read_file>\n<parameter=start_line>\n10\n</parameter>\n<parameter=path>\na\n</parameter>\n<parameter=end_line>\n-3\n</parameter>\n</function>\n</tool_call>",
            "\n<function=edit_file>\n<parameter=path>\np\n</parameter>\n<parameter=old>\nfn a() {\n    x < y && y > z\n}\n</parameter>\n<parameter=new>\n\n</parameter>\n</function>\n</tool_call>\n",
            "\n<function=git_status>\n</function>\n</tool_call>",
            "\n<function=git_commit>\n<parameter=message>\nm\n</parameter>\n<parameter=paths>\n[\"a\", \"b/c\"]  \n</parameter>\n</function>\n</tool_call>",
            "\n<function=git_diff>\n<parameter=staged>\ntrue\n</parameter>\n</function>\n</tool_call>",
            "\n<function=goal_step>\n<parameter=step>\n3\n</parameter>\n<parameter=status>\ndone\n</parameter>\n</function>\n</tool_call>",
        ] {
            assert!(complete(&g, text), "{text:?}: {:?}", run(&g, text));
        }
    }

    #[test]
    fn the_observed_failure_shapes_are_refused_at_the_first_wrong_byte() {
        let g = crow();
        // 2026-09-22 15:17:12 and three more: `<tool_call>\n\n</function>\n</tool_call>`
        assert_eq!(run(&g, "\n\n</function>\n</tool_call>"), Err(1));
        // `old_string` for edit_file's `old` (22 of 302 calls of the stored session): the
        // name may run to `old`, then only `>` closes it
        let p = "\n<function=edit_file>\n<parameter=path>\np\n</parameter>\n<parameter=old";
        assert!(run(&g, p).is_ok());
        assert_eq!(run(&g, &format!("{p}_string>")), Err(p.len()));
        // an undeclared tool
        assert_eq!(run(&g, "\n<function=edit>"), Err("\n<function=edit".len()));
        // a required parameter missing: `</function>` is ACCEPTED since 2026-09-23 - the
        // client reports the missing parameter; refusing forced filler into the context
        let p = "\n<function=edit_file>\n<parameter=path>\np\n</parameter>\n<";
        assert!(run(&g, p).is_ok());
        assert!(run(&g, &format!("{p}/function>")).is_ok());
        // a value closed without a newline before `</parameter>` is accepted too (the
        // parser cuts there anyway); `</tool_call>` inside a value stays refused
        assert!(run(&g, "\n<function=read_file>\n<parameter=path>\na</parameter>\n</function>").is_ok());
        assert!(run(&g, "\n<function=read_file>\n<parameter=path>\n</parameter>\n</function>").is_ok());
        assert!(run(&g, "\n<function=read_file>\n<parameter=path>\na</tool_call>").is_err());
        // a non-integer for an integer, in each spelling the model reaches for
        let p = "\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=start_line>\n";
        for bad in ["ten", "\"10\"", "1.5", "10a", "01", " 10", "+1"] {
            assert!(run(&g, &format!("{p}{bad}\n</parameter>")).is_err(), "{bad:?} accepted for an integer");
        }
        // a parameter written twice: refused at its first byte, `p` leads to no unseen name
        let p = "\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=";
        assert_eq!(run(&g, &format!("{p}path>")), Err(p.len()));
        // `</parameter>` without the newline CLOSES the string since 2026-09-23 (the parser
        // cuts there too); `</tool_call>` inside it stays refused
        let p = "\n<function=read_file>\n<parameter=path>\nab";
        assert!(run(&g, &format!("{p}</parameter>")).is_ok());
        assert!(run(&g, &format!("{p}\n</tool_call>")).is_err());
        // prose after a call
        let p = "\n<function=git_status>\n</function>\n</tool_call>\n\n";
        assert!(run(&g, p).is_ok());
        assert!(run(&g, &format!("{p}Done")).is_err());
        assert!(run(&g, &format!("{p}\n")).is_err(), "at most two newlines of space");
        // a JSON array for a declared array: typed items
        let p = "\n<function=git_commit>\n<parameter=message>\nm\n</parameter>\n<parameter=paths>\n";
        assert!(run(&g, &format!("{p}[1]")).is_err());
        assert!(run(&g, &format!("{p}\"a\"")).is_err());
        assert!(run(&g, &format!("{p}[\"a\",]")).is_err(), "no trailing comma");
    }

    #[test]
    fn values_follow_the_json_subset() {
        let tools = serde_json::json!([{"type":"function","function":{"name":"f","parameters":{"type":"object",
            "properties":{
                "n":{"type":"number"},
                "mode":{"type":"string","enum":["fast","faster"]},
                "lvl":{"enum":[1,12,"x"]},
                "obj":{"type":"object","properties":{"a":{"type":"integer"},"b":{"type":"string"}},"required":["a"]},
                "any":{}
            }}}}]);
        let g = ToolGrammar::build(&tools, Mode::Auto, true, None).unwrap();
        let ok = |p: &str, v: &str| {
            let t = format!("\n<function=f>\n<parameter={p}>\n{v}\n</parameter>\n</function>\n</tool_call>");
            complete(&g, &t)
        };
        for v in ["0", "-0.5", "1e9", "2.5E-3", "12"] {
            assert!(ok("n", v), "number {v}");
        }
        for v in ["1.", ".5", "--1", "1e", "0x1"] {
            assert!(!ok("n", v), "number {v}");
        }
        assert!(ok("mode", "fast") && ok("mode", "faster"));
        assert!(!ok("mode", "fas") && !ok("mode", "\"fast\""));
        assert!(ok("lvl", "1") && ok("lvl", "12") && ok("lvl", "\"x\""));
        assert!(!ok("lvl", "2") && !ok("lvl", "123"));
        assert!(ok("obj", "{\"a\": 1}") && ok("obj", "{\"b\":\"s\", \"a\":2}"));
        assert!(!ok("obj", "{\"b\":\"s\"}"), "required key missing");
        assert!(!ok("obj", "{\"a\":1,\"a\":2}"), "duplicate key");
        assert!(!ok("obj", "{\"c\":1}"), "undeclared key");
        assert!(!ok("obj", "{\"a\":\"1\"}"), "typed value");
        for v in ["{\"k\": [1, {\"z\": null}], \"s\": \"\\u00e9\\n\"}", "[]", "\"s\"", "false", "-1"] {
            assert!(ok("any", v), "any {v}");
        }
        for v in ["{k:1}", "[1 2]", "\"a\nb\"", "\"\\x\"", "tru"] {
            assert!(!ok("any", v), "any {v}");
        }
        // too deep for the machine: refused, never a dead end
        assert!(!ok("any", "[[[[[[[1]]]]]]]"));
    }

    #[test]
    fn required_mode_refuses_eos_until_a_call_closed_and_parallel_off_refuses_a_second_call() {
        let g = ToolGrammar::build(&crow_tools(), Mode::Required, false, None).unwrap();
        assert!(!g.eos_ok(&St::IDLE));
        assert!(g.open_ok(&St::IDLE));
        let s = run(&g, "\n<function=git_status>\n</function>\n</tool_call>").unwrap();
        assert!(g.eos_ok(&s));
        assert!(!g.open_ok(&s), "parallel_tool_calls false");
        let g = crow();
        assert!(g.eos_ok(&St::IDLE) && g.open_ok(&s));
    }

    // ------------------------------------------------------------- token level, real tokenizer

    fn tokenizer() -> crate::tokenizer::ChatTokenizer {
        let t = format!("../{}", crate::tokenizer::DEFAULT_TOKENIZER);
        let c = t.replace("tokenizer.json", "tokenizer_config.json");
        crate::tokenizer::ChatTokenizer::load(&t, &c).expect("tokenizer loads")
    }

    fn vocab(tk: &crate::tokenizer::ChatTokenizer) -> Vocab {
        let eos: Vec<u32> = crate::sample::EOS_IDS.iter().map(|&e| e as u32).collect();
        let open = tk.token_id("<tool_call>").expect("<tool_call> id");
        Vocab::build(crate::geo::V, |id| tk.token_bytes(id), |id| tk.is_special(id), &eos, open)
    }

    /// the model's ids for `text` after the `<tool_call>` token, as the tokenizer splits it
    fn ids(tk: &crate::tokenizer::ChatTokenizer, text: &str) -> Vec<u32> {
        let mut v = vec![tk.token_id("<tool_call>").unwrap()];
        v.extend(tk.encode_raw(text).expect("encodes"));
        v
    }

    #[test]
    fn token_masks_on_the_real_vocabulary_agree_with_the_per_token_check() {
        let tk = tokenizer();
        let v = vocab(&tk);
        let g = crow();
        let text = "\n<function=edit_file>\n<parameter=path>\n/home/nibor1896/a.rs\n</parameter>\n<parameter=old>\nlet x = 1;\n</parameter>\n<parameter=new>\nlet x = 2;\n</parameter>\n</function>\n</tool_call>";
        let seq = ids(&tk, text);
        let mut s = St::IDLE;
        let mut bits = Vec::new();
        let mut rng = crate::sample::Rng::new(7);
        for (k, &id) in seq.iter().enumerate() {
            v.mask(&g, &s, &mut bits);
            assert!(allowed(&bits, id as usize), "the tokenizer's own split refused at id #{k} ({id})");
            // the mask and the per-token walk are the same predicate: every id of a
            // random sample, plus every EOS and the opener
            let mut probe: Vec<u32> = (0..400).map(|_| (rng.next_u64() % crate::geo::V as u64) as u32).collect();
            probe.extend(crate::sample::EOS_IDS.iter().map(|&e| e as u32));
            probe.push(tk.token_id("<tool_call>").unwrap());
            for p in probe {
                assert_eq!(allowed(&bits, p as usize), v.token_ok(&g, &s, p), "state #{k} id {p}");
            }
            // and the mask is never empty: no dead end
            assert!(bits.iter().any(|&w| w != 0));
            assert!(v.accept(&g, &mut s, id));
        }
        assert!(g.eos_ok(&s), "the call closed");
    }

    #[test]
    fn the_observed_failure_shapes_are_masked_on_the_real_vocabulary() {
        let tk = tokenizer();
        let v = vocab(&tk);
        let g = crow();
        let eos = crate::sample::EOS_IDS[0] as u32;
        // walk `prefix`, then report whether each id of `bad` is allowed next
        let at = |prefix: &str| -> St {
            let mut s = St::IDLE;
            for id in ids(&tk, prefix) {
                assert!(v.accept(&g, &mut s, id), "prefix {prefix:?} refused");
            }
            s
        };
        // `<tool_call>\n\n</function>`: after `<tool_call>` only `\n<function=` may follow;
        // the model's own split of each shape is walked until an id is refused
        let refused = |s: St, text: &str| -> Option<(usize, u32)> {
            let mut t = s;
            for (k, id) in tk.encode_raw(text).unwrap().into_iter().enumerate() {
                if !v.accept(&g, &mut t, id) {
                    return Some((k, id));
                }
            }
            None
        };
        let s = at("");
        for bad in ["\n\n</function>\n</tool_call>", "\n</function>\n</tool_call>", "</function>"] {
            assert!(refused(s, bad).is_some(), "{bad:?} walked through after <tool_call>");
        }
        assert_eq!(refused(s, "\n\n</function>").map(|r| r.0), Some(0), "the `\n\n` id itself");
        assert!(!v.token_ok(&g, &s, eos), "EOS inside a call");
        // `old_string`: `_` (and every id starting with it) refused after `old`
        let s = at("\n<function=edit_file>\n<parameter=path>\np\n</parameter>\n<parameter=old");
        let under = tk.encode_raw("_string>").unwrap()[0];
        assert!(!v.token_ok(&g, &s, under), "the `_string` piece ({under}) allowed after `old`");
        let gt = tk.encode_raw(">\n").unwrap();
        assert!(v.token_ok(&g, &s, gt[0]), "`>` closes `old`");
        // missing required: `</function>` ACCEPTED while `new` and `old` are owed (the
        // client reports it; refusing forced filler) - its tokens all walk through
        let s = at("\n<function=edit_file>\n<parameter=path>\np\n</parameter>\n");
        let close = tk.encode_raw("</function>").unwrap();
        let mut bits = Vec::new();
        let mut t = s;
        let refused = close.iter().any(|&id| {
            let ok = v.token_ok(&g, &t, id);
            if ok {
                v.accept(&g, &mut t, id);
            }
            !ok
        });
        assert!(!refused, "`</function>` with required parameters owed must close the call");
        // a non-integer for an integer: `ten`, `"`, `.` refused at their first id
        let s = at("\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=start_line>\n");
        for bad in ["ten", "\"", "x"] {
            let id = tk.encode_raw(bad).unwrap()[0];
            assert!(!v.token_ok(&g, &s, id), "{bad:?} allowed for an integer");
        }
        let s10 = at("\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=start_line>\n10");
        let dot = tk.encode_raw(".").unwrap()[0];
        assert!(!v.token_ok(&g, &s10, dot), "`.` after an integer");
        // free text: the whole file body may be any id that is not special
        let s = at("\n<function=write_file>\n<parameter=path>\na\n</parameter>\n<parameter=content>\n");
        v.mask(&g, &s, &mut bits);
        let n_ok = bits.iter().map(|w| w.count_ones() as usize).sum::<usize>();
        assert!(n_ok > 240_000, "a string value allows the vocabulary ({n_ok})");
        assert!(!allowed(&bits, eos as usize));
        assert!(!allowed(&bits, tk.token_id("<|im_start|>").unwrap() as usize));
        assert!(!allowed(&bits, tk.token_id("<tool_call>").unwrap() as usize));
    }

    #[test]
    fn a_grammar_accepted_call_parses_into_the_declared_arguments() {
        // the grammar and `toolcall` agree: what the grammar lets through, the parser turns
        // into ONE call with valid JSON arguments of the declared types
        let text = "<tool_call>\n<function=read_file>\n<parameter=start_line>\n10\n</parameter>\n<parameter=path>\n/a b/c.md\n</parameter>\n</function>\n</tool_call>";
        assert!(complete(&crow(), &text["<tool_call>".len()..]));
        let tools = crow_tools();
        let mut ts = crate::toolcall::ToolStream::new(Some(&tools));
        ts.arm();
        let mut out = ts.feed(text);
        assert!(!ts.finish(&mut out));
        let args: String = out
            .iter()
            .filter_map(|e| match e {
                crate::toolcall::Emit::Args { text, .. } => Some(text.as_str()),
                _ => None,
            })
            .collect();
        let v: serde_json::Value = serde_json::from_str(&args).unwrap();
        assert_eq!(v, serde_json::json!({"start_line": 10, "path": "/a b/c.md"}));
        assert!(ts.malformed().is_empty());
    }

    #[test]
    fn a_random_walk_over_allowed_ids_never_reaches_a_dead_end() {
        // liveness: from every state the walk reaches, some id is allowed; ids are drawn
        // from the mask with a bias to short ones so the walk also leaves long values
        let tk = tokenizer();
        let v = vocab(&tk);
        let g = crow();
        let mut rng = crate::sample::Rng::new(11);
        let mut bits = Vec::new();
        for round in 0..6 {
            let mut s = St::IDLE;
            assert!(v.accept(&g, &mut s, tk.token_id("<tool_call>").unwrap()));
            for step in 0..300 {
                v.mask(&g, &s, &mut bits);
                let ok: Vec<u32> = (0..crate::geo::V as u32).filter(|&i| allowed(&bits, i as usize)).collect();
                assert!(!ok.is_empty(), "dead end in round {round} step {step}: {s:?}");
                if g.eos_ok(&s) {
                    break;
                }
                let short: Vec<u32> = ok.iter().copied().filter(|&i| v.bytes_of(i).len() <= 2).collect();
                let pool = if !short.is_empty() && !rng.next_u64().is_multiple_of(4) { &short } else { &ok };
                let id = pool[(rng.next_u64() % pool.len() as u64) as usize];
                assert!(v.accept(&g, &mut s, id));
            }
        }
    }

    #[test]
    fn mask_cost_on_cpu() {
        // the measurement of record for the ticket: build, per-token check, full mask per
        // phase. Printed with --nocapture; the bound is loose (a debug-build safety net).
        let tk = tokenizer();
        let t0 = std::time::Instant::now();
        let v = vocab(&tk);
        let build_ms = t0.elapsed().as_secs_f64() * 1e3;
        let g = crow();
        let at = |prefix: &str| -> St {
            let mut s = St::IDLE;
            for id in ids(&tk, prefix) {
                assert!(v.accept(&g, &mut s, id));
            }
            s
        };
        let states = [
            ("after <tool_call>", at("")),
            ("function name", at("\n<function=re")),
            ("parameter name", at("\n<function=edit_file>\n<parameter=")),
            ("string value", at("\n<function=write_file>\n<parameter=content>\nfn main() {")),
            ("integer value", at("\n<function=read_file>\n<parameter=path>\na\n</parameter>\n<parameter=start_line>\n1")),
            ("JSON array", at("\n<function=git_commit>\n<parameter=paths>\n[\"a\", ")),
        ];
        let mut bits = Vec::new();
        let mut line = format!("vocab {} ids, trie {} nodes, built in {build_ms:.1} ms;", v.len(), v.trie_nodes());
        let mut worst = 0.0f64;
        for (name, s) in states {
            let reps = 5;
            let t = std::time::Instant::now();
            for _ in 0..reps {
                v.mask(&g, &s, &mut bits);
            }
            let ms = t.elapsed().as_secs_f64() * 1e3 / reps as f64;
            worst = worst.max(ms);
            let n = bits.iter().map(|w| w.count_ones()).sum::<u32>();
            line += &format!(" mask[{name}] {ms:.2} ms ({n} allowed);");
        }
        // the per-token check: every id of a realistic call, many times
        let s = at("\n<function=write_file>\n<parameter=content>\n");
        let body = tk.encode_raw("fn main() {\n    println!(\"hello </b> world\");\n}\n").unwrap();
        let reps = 20_000;
        let t = std::time::Instant::now();
        let mut n_ok = 0usize;
        for _ in 0..reps {
            let mut st = s;
            for &id in &body {
                if v.token_ok(&g, &st, id) {
                    n_ok += 1;
                }
                v.accept(&g, &mut st, id);
            }
        }
        let ns = t.elapsed().as_secs_f64() * 1e9 / (reps * body.len()) as f64;
        assert_eq!(n_ok, reps * body.len());
        // idle: the branch serve pays per token outside a call
        let gate = Gate::new(crow(), &v);
        let t = std::time::Instant::now();
        let mut armed = 0usize;
        for _ in 0..1_000_000 {
            armed += std::hint::black_box(&gate).armed() as usize;
        }
        let idle_ns = t.elapsed().as_secs_f64() * 1e9 / 1e6;
        assert_eq!(armed, 0);
        // and the advance serve runs on every kept id outside a call (only the opener moves it)
        let mut gate = Gate::new(crow(), &v);
        let t = std::time::Instant::now();
        for i in 0..1_000_000u32 {
            gate.accept(std::hint::black_box(i % 200_000));
        }
        let idle_acc_ns = t.elapsed().as_secs_f64() * 1e9 / 1e6;
        assert!(!gate.st.in_call());
        line += &format!(
            " token_ok+accept {ns:.0} ns/token in a string value; outside a call: armed() {idle_ns:.2} ns + accept {idle_acc_ns:.2} ns per token"
        );
        eprintln!("[toolgrammar cost] {line}");
        assert!(worst < 2000.0, "{line}");
    }
}
