//! #29 A7 toolcall: the model's tool-call markup turned into OpenAI `delta.tool_calls`
//! fragments, plus the state machine that reads it out of the decoded text stream.
//!
//! Scope:
//!
//! | lives here | lives in `bin/serve.rs` |
//! |---|---|
//! | `ToolStream`, `TState`, `Emit`, `find_marker`, the JSON escaping helpers | the SSE chunk builders (`chunk_tool_open`, `chunk_tool_args`) |
//! | the parser unit tests | `send_emits` and the `chat_stream` call sites |
//!
//! - Moved out of `bin/serve.rs` on 2026-09-09 (#29 review); no logic lives twice.
//! - The only behaviour change of that move is the `Tail` row of the end-of-generation table.
//!
//! The markup this model is instructed to emit (chat template, `tokenizer_config.json`):
//!
//! ```text
//! <tool_call>
//! <function=read_file>
//! <parameter=path>
//! C:/x/y.md
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! - It is NOT the `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` JSON form.
//! - `<tool_call>` is added token 248058, `</tool_call>` is 248059, both `special: false`.
//! - `<tool_call>` is matched by TOKEN ID, through `arm()`; `</tool_call>` is matched by TEXT.
//! - The id is what opens a call, so the literal text `<tool_call>` stays content.
//! - The other four markers are ordinary text and split across tokens.
//!
//! End of generation (`finish`), by the state the parser stands in:
//!
//! | state | what `finish` does | return |
//! |---|---|---|
//! | `Text` | the rest is content, or dropped and counted when a call already ran | `false` |
//! | `Tail`, after `</function>` | the call is CLOSED, the trailing markup is dropped and counted | `false` |
//! | any other | MALFORMED: the `arguments` object is closed and marked, the raw markup of the call in flight goes out as content | `true` |
//!
//! - `Tail` means the call is COMPLETE: `</function>` already emitted the closing `}`.
//! - So the client gets a parseable call, `finish_reason` `tool_calls`, and NOT the raw markup.
//! - One stderr line names it: `tool call closed at EOS without </tool_call>`.
//! - Only a truncated call (no `</function>`) is malformed.
//!
//! The `arguments` invariant (TASK J, 2026-09-17):
//!
//! - For EVERY index this parser names, the concatenation of its `Emit::Args` fragments is a
//!   PARSEABLE JSON OBJECT. There is no input that makes it anything else.
//! - A complete call closes with `}` (or `{}` when it carried no parameter), as before.
//! - An ABANDONED call (truncated at `max_tokens`, or markup this parser gives up on) closes
//!   through `close_args_truncated`: the value in flight is closed the way `</parameter>`
//!   would have closed it, then `,"_truncated":true}` (or `{"_truncated":true}`) goes out.
//! - Why it matters: Crow concatenates the fragments into a STRING and stores that string in
//!   its history verbatim (`crow_core.py:5068`, `:3756-3761`), then re-sends the whole history
//!   every turn (`:3783-3785`, `:4866`). The chat template iterates `arguments|items`, which
//!   needs a mapping, so ONE unterminated call used to kill every later turn of that session
//!   with a 400, and Crow has no way to drop the message again (`:13485-13487`).
//! - Why the marker instead of a bare `}`: `{"path":"C:/x/y.m"}` parses, so a client would
//!   run half a command. `_truncated` is declared by no tool, so the call fails as an
//!   argument error instead. The old contract bought the same safety by leaving the JSON
//!   broken; that is what poisoned the history.
//! - `finish_reason` is unchanged: a malformed call keeps `stop` or `length`, never
//!   `tool_calls`, and the raw markup still goes out as content before the first call.
//!
//! Byte accounting:
//!
//! - `dropped()` counts every content or markup byte the parser swallowed without emitting it.
//! - That is content after the first call, the `Tail` remainder, and a `give_up` after a call.
//! - `serve` prints the total in one stderr line per request, so the number is exact.

/// `<tool_call>`, added token id 248058 of this model, `special: false`
pub const TOOL_OPEN: &str = "<tool_call>";

/// The declaration shape Crow's `_fn` builds (`crow_core.py:569-574`): two
/// parameters of two different declared types, so both value paths are
/// exercised. `pub` and outside `cfg(test)` because `bin/serve.rs`'s oracle
/// render was captured from THIS json - a second copy over there could drift
/// from the captured bytes and nothing would catch it.
pub fn a7_tools_fixture() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the file." },
                        "start_line": { "type": "integer", "description": "First line, 1-based." }
                    },
                    "required": ["path"]
                }
            }
        }])
}
/// `</tool_call>`, added token id 248059 of this model, `special: false`
const TOOL_CLOSE: &str = "</tool_call>";
/// opens the function block; NOT an added token, it arrives as ordinary pieces
const FUNCTION_OPEN: &str = "<function=";
/// closes the function block
const FUNCTION_CLOSE: &str = "</function>";
/// opens one parameter block
const PARAM_OPEN: &str = "<parameter=";
/// closes one parameter block
const PARAM_CLOSE: &str = "</parameter>";
/// a function name longer than this is markup that never closed, not a name
const MAX_TOOL_NAME: usize = 128;
/// TASK J: the key a TRUNCATED call's `arguments` object carries, and the only key in it
/// that no tool declares; `pub` because `bin/serve.rs` names it in its contract table test
pub const TRUNCATED_KEY: &str = "_truncated";

/// what the parser wants the stream writer to send next
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Emit {
    /// `delta.content`, the A4 path
    Content(String),
    /// `delta.tool_calls[0]` with `index`, `id`, `type` and `function.name`, once per call
    Call { index: usize, id: String, name: String },
    /// `delta.tool_calls[0].function.arguments`, a raw JSON text fragment
    Args { index: usize, text: String },
}

/// where the parser stands inside the model's tool-call markup
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum TState {
    /// outside any tool call: text is content
    Text,
    /// after `<tool_call>`, looking for `<function=`
    Func,
    /// collecting the function name, until `>`
    FuncName,
    /// inside the function block, looking for `<parameter=`, `</function>` or `</tool_call>`
    Body,
    /// collecting a parameter name, until `>`
    ParamName,
    /// collecting a parameter value, until `</parameter>`
    ParamValue,
    /// after `</function>`, looking for `</tool_call>`
    Tail,
}

/// - one JSON string literal, quotes included, exactly as `serde_json` writes it
fn json_str(s: &str) -> String {
    serde_json::Value::String(s.to_string()).to_string()
}

/// - the inside of a JSON string literal: `json_str` without the two quotes
/// - safe per piece: every escape is produced whole, none can span two calls
fn json_escape(s: &str) -> String {
    let q = json_str(s);
    q[1..q.len() - 1].to_string()
}

/// - `Some((at, i))`: `markers[i]` starts at byte `at` of `buf`
/// - `None`: no marker is complete; the returned length is what can never be part of one
/// - the returned length is a char boundary, because every marker is ASCII
fn find_marker(buf: &str, markers: &[&str]) -> (Option<(usize, usize)>, usize) {
    let mut best: Option<(usize, usize)> = None;
    for (i, m) in markers.iter().enumerate() {
        if let Some(at) = buf.find(m) {
            if best.map_or(true, |(b, _)| at < b) {
                best = Some((at, i));
            }
        }
    }
    if let Some(hit) = best {
        return (Some(hit), hit.0);
    }
    let mut hold = 0usize;
    for m in markers {
        let max = (m.len() - 1).min(buf.len());
        for k in (hold + 1..=max).rev() {
            let cut = buf.len() - k;
            if buf.is_char_boundary(cut) && buf.as_bytes()[cut..] == m.as_bytes()[..k] {
                hold = k;
                break;
            }
        }
    }
    (None, buf.len() - hold)
}

/// - `{function name: {parameter name: declared JSON type}}` out of the request's `tools`
/// - the declared type decides whether a value streams as a JSON string or is buffered
/// - a tool, a parameter or a type this map does not carry falls back to the value heuristic
fn tool_param_types(
    tools: Option<&serde_json::Value>,
) -> std::collections::HashMap<String, std::collections::HashMap<String, String>> {
    let mut out = std::collections::HashMap::new();
    let arr = match tools.and_then(|t| t.as_array()) {
        Some(a) => a,
        None => return out,
    };
    for t in arr {
        let f = match t.get("function") {
            Some(f) => f,
            None => continue,
        };
        let name = match f.get("name").and_then(|v| v.as_str()) {
            Some(n) => n.to_string(),
            None => continue,
        };
        let mut params = std::collections::HashMap::new();
        if let Some(props) = f
            .get("parameters")
            .and_then(|p| p.get("properties"))
            .and_then(|p| p.as_object())
        {
            for (k, v) in props {
                if let Some(ty) = v.get("type").and_then(|v| v.as_str()) {
                    params.insert(k.clone(), ty.to_string());
                }
            }
        }
        out.insert(name, params);
    }
    out
}

/// - #29 A7: the model's tool-call markup turned into OpenAI `delta.tool_calls` fragments
/// - the markup itself is in the module header; it is NOT the JSON form
/// - `<tool_call>` and `</tool_call>` are added tokens (248058, 248059), the rest is ordinary text
/// - the arguments JSON object therefore has to be BUILT, and it is built in fragments
///
/// | markup event | fragment that goes out |
/// |---|---|
/// | `<function=NAME>` closes | `Emit::Call { index, id: "call_<index>", name }` |
/// | first `<parameter=P>` closes | `{"P":` (plus `"` when P is a declared string) |
/// | later `<parameter=P>` closes | `,"P":` (plus `"` when P is a declared string) |
/// | value text of a declared string | the escaped piece, one fragment per decoded piece |
/// | value text of any other type | buffered, one fragment at `</parameter>` |
/// | `</parameter>` of a string | `"` |
/// | `</function>` or `</tool_call>` | `}`, or `{}` when the call carried no parameter |
/// | the call is ABANDONED (TASK J) | the value in flight is closed, then `,"_truncated":true}` (or `{"_truncated":true}`) |
///
/// - The concatenation of every fragment of one index is exactly the arguments JSON text.
/// - `id` and `name` ride on the FIRST fragment only; no later fragment carries either key.
/// - A value of an undeclared parameter is emitted raw when it parses as JSON, else quoted.
/// - Text before the first `<tool_call>` is content; after it content is DROPPED and counted.
///
/// Robustness:
///
/// - Markers may split across pieces: an incomplete marker suffix is held back, never emitted.
/// - `<parameter=P>\n` and `\n</parameter>` are template separators, not value bytes.
/// - Entering a call needs `arm()`, which the decode loop calls on token id 248058 ONLY.
/// - So `<tool_call>` is matched by TOKEN ID; `</tool_call>` is matched by TEXT.
/// - The literal text `<tool_call>` out of ordinary tokens therefore stays content.
/// - `finish()` in `Tail` (after `</function>`) CLOSES the call: complete, no raw markup.
/// - `finish()` in any other state but `Text` is MALFORMED: the raw markup goes out as content.
pub struct ToolStream {
    /// declared parameter types, from the request's `tools`
    types: std::collections::HashMap<String, std::collections::HashMap<String, String>>,
    state: TState,
    /// text consumed from the wire but not yet acted on
    buf: String,
    /// `<tool_call>` token ids seen and not yet matched in the text
    armed: usize,
    /// index of the call being built, and the count of calls started
    index: usize,
    /// calls whose `</tool_call>` arrived
    closed: usize,
    /// a `Call` fragment went out for `index`
    named: bool,
    /// the function name of the call being built
    name: String,
    /// the parameter name being collected
    pname: String,
    /// `{` already went out for this call
    open_brace: bool,
    /// the current value streams as a JSON string
    vstring: bool,
    /// the current value of a non string parameter
    vbuf: String,
    /// the value region has not yet dropped the newline after `<parameter=P>`
    skip_nl: bool,
    /// a trailing newline held back: it may be the separator before `</parameter>`
    held_nl: bool,
    /// raw markup of the call in flight, for the malformed flush
    raw: String,
    /// length of `raw` when `</function>` closed: everything past it is trailing markup
    tail_from: usize,
    /// bytes swallowed without being emitted: content after a call, or a `Tail` remainder
    dropped: usize,
}

impl ToolStream {
    pub fn new(tools: Option<&serde_json::Value>) -> Self {
        ToolStream {
            types: tool_param_types(tools),
            state: TState::Text,
            buf: String::new(),
            armed: 0,
            index: 0,
            closed: 0,
            named: false,
            name: String::new(),
            pname: String::new(),
            open_brace: false,
            vstring: false,
            vbuf: String::new(),
            skip_nl: false,
            held_nl: false,
            raw: String::new(),
            tail_from: 0,
            dropped: 0,
        }
    }

    /// the decode loop saw the `<tool_call>` token ID; only this permits entering a call
    pub fn arm(&mut self) {
        self.armed += 1;
    }

    /// calls whose `</tool_call>` arrived
    pub fn closed(&self) -> usize {
        self.closed
    }

    /// - bytes the parser swallowed without emitting them, over the whole request
    /// - content after the first call, a `Tail` remainder, a `give_up` after a call
    pub fn dropped(&self) -> usize {
        self.dropped
    }

    /// one decoded text piece in, the fragments it completes out
    pub fn feed(&mut self, piece: &str) -> Vec<Emit> {
        self.buf.push_str(piece);
        let mut out = Vec::new();
        self.run(&mut out);
        out
    }

    /// - end of generation (EOS or `max_tokens`): flush what is left
    /// - `Text`: the rest is content, or dropped and counted when a call already ran
    /// - `Tail`: the call is COMPLETE (`</function>` emitted the closing `}`), so it is
    ///   CLOSED here and the trailing markup is dropped, never replayed as content
    /// - any other state: MALFORMED, the raw markup of the call in flight goes out as content
    /// - `true` means MALFORMED, and only that; `Tail` returns `false`
    pub fn finish(&mut self, out: &mut Vec<Emit>) -> bool {
        if self.state == TState::Text {
            let rest = std::mem::take(&mut self.buf);
            self.content(&rest, out);
            return false;
        }
        if self.state == TState::Tail {
            // everything past `</function>` is markup this call does not need: drop it, count
            // it, and close the call so `finish_reason` is `tool_calls` and the client gets a
            // complete, parseable call instead of a complete call PLUS its raw markup
            self.dropped += self.raw.len().saturating_sub(self.tail_from) + self.buf.len();
            self.buf.clear();
            self.close_call();
            eprintln!("[toolcall] tool call closed at EOS without </tool_call>");
            return false;
        }
        // TASK J: the fragments already on the wire are closed into a JSON object and
        // marked, so Crow's stored history renders and the half call is not runnable
        self.close_args_truncated(out);
        self.flush_raw(out);
        self.state = TState::Text;
        true
    }

    /// - the raw markup of the call in flight leaves the parser
    /// - before the first completed call it goes out as content, after one it is DROPPED
    /// - either way every byte is accounted for, so the `dropped` line is exact
    fn flush_raw(&mut self, out: &mut Vec<Emit>) {
        let mut raw = std::mem::take(&mut self.raw);
        raw.push_str(&std::mem::take(&mut self.buf));
        if raw.is_empty() {
            return;
        }
        if self.index > 0 {
            self.dropped += raw.len();
            return;
        }
        out.push(Emit::Content(raw));
    }

    /// content before the first call goes out; after it, it is dropped and counted
    fn content(&mut self, s: &str, out: &mut Vec<Emit>) {
        if s.is_empty() {
            return;
        }
        if self.index > 0 || self.named {
            self.dropped += s.len();
            return;
        }
        out.push(Emit::Content(s.to_string()));
    }

    /// take `n` bytes off `buf` into the raw markup record of the call in flight
    fn eat(&mut self, n: usize) -> String {
        let s: String = self.buf[..n].to_string();
        self.raw.push_str(&s);
        self.buf.drain(..n);
        s
    }

    /// the fragment that opens a parameter: `{"p":` or `,"p":`, plus `"` for a string value
    fn open_param(&mut self, out: &mut Vec<Emit>) {
        let ty = self
            .types
            .get(&self.name)
            .and_then(|m| m.get(&self.pname))
            .map(|s| s.as_str());
        self.vstring = ty == Some("string");
        self.vbuf.clear();
        self.skip_nl = true;
        self.held_nl = false;
        let mut frag = String::new();
        frag.push(if self.open_brace { ',' } else { '{' });
        self.open_brace = true;
        frag.push_str(&json_str(&self.pname));
        frag.push(':');
        if self.vstring {
            frag.push('"');
        }
        out.push(Emit::Args { index: self.index, text: frag });
    }

    /// - value bytes, with the template separators taken out
    /// - `terminal` means `</parameter>` was reached, so a trailing newline is the separator
    fn value(&mut self, text: &str, terminal: bool, out: &mut Vec<Emit>) {
        let mut t = String::new();
        if self.held_nl {
            t.push('\n');
            self.held_nl = false;
        }
        t.push_str(text);
        if terminal {
            if t.ends_with('\n') {
                t.pop();
            }
        } else if t.ends_with('\n') {
            t.pop();
            self.held_nl = true;
        }
        if self.vstring {
            if !t.is_empty() {
                out.push(Emit::Args { index: self.index, text: json_escape(&t) });
            }
        } else {
            self.vbuf.push_str(&t);
        }
    }

    /// `</parameter>`: close a string value, or emit the buffered one as raw JSON
    fn close_param(&mut self, out: &mut Vec<Emit>) {
        if self.vstring {
            out.push(Emit::Args { index: self.index, text: "\"".to_string() });
        } else {
            let v = std::mem::take(&mut self.vbuf);
            let t = v.trim();
            let text = match serde_json::from_str::<serde_json::Value>(t) {
                Ok(_) if !t.is_empty() => t.to_string(),
                _ => json_str(&v),
            };
            out.push(Emit::Args { index: self.index, text });
        }
        self.skip_nl = false;
        self.held_nl = false;
    }

    /// `}` closes the arguments object; a call without a parameter gets `{}`
    fn close_args(&mut self, out: &mut Vec<Emit>) {
        if !self.named {
            return;
        }
        let frag = if self.open_brace { "}" } else { "{}" };
        out.push(Emit::Args { index: self.index, text: frag.to_string() });
        self.open_brace = false;
    }

    /// - TASK J: the call in flight is ABANDONED (truncated at `max_tokens`, or markup this
    ///   parser cannot read), and its `arguments` fragments are already on the wire
    /// - those fragments are a PREFIX of a JSON object, so this closes the object
    /// - the value in flight is closed the way `</parameter>` would have closed it, then
    ///   `TRUNCATED_KEY` is added and the object is closed
    /// - the invariant it buys: the concatenation of every `Emit::Args` of one index is a
    ///   PARSEABLE JSON OBJECT, always, so a client that stores it and sends it back cannot
    ///   poison its own history (the chat template iterates `arguments|items`)
    /// - the marker is what keeps a truncated call from being RUN as if it were complete:
    ///   no tool declares `_truncated`, so a caller that passes the object on gets an
    ///   argument error, not half a command
    fn close_args_truncated(&mut self, out: &mut Vec<Emit>) {
        if !self.named {
            return;
        }
        if self.state == TState::ParamValue {
            self.close_param(out);
        }
        let frag = if self.open_brace {
            format!(",{}:true}}", json_str(TRUNCATED_KEY))
        } else {
            format!("{{{}:true}}", json_str(TRUNCATED_KEY))
        };
        out.push(Emit::Args { index: self.index, text: frag });
        self.open_brace = false;
    }

    /// `</tool_call>`: the call is complete, the next one gets the next index
    fn close_call(&mut self) {
        if self.named {
            self.closed += 1;
            self.index += 1;
        }
        self.named = false;
        self.name.clear();
        self.open_brace = false;
        self.raw.clear();
        self.tail_from = 0;
        self.state = TState::Text;
    }

    /// the call in flight is markup this parser cannot read: back to content
    /// Both `<function=NAME>` and `<parameter=NAME>` read the same way: the text
    /// up to the next `>`, trimmed and length-capped. `Ok(None)` = need more input,
    /// `Err(())` = malformed, give up.
    fn read_name_to_gt(&mut self) -> Result<Option<String>, ()> {
        match self.buf.find('>') {
            Some(at) => {
                let s = self.eat(at + 1);
                let name = s[..at].trim().to_string();
                if name.is_empty() || name.len() > MAX_TOOL_NAME {
                    return Err(());
                }
                Ok(Some(name))
            }
            None => {
                if self.buf.len() > MAX_TOOL_NAME {
                    return Err(());
                }
                Ok(None)
            }
        }
    }

    fn give_up(&mut self, out: &mut Vec<Emit>) {
        // TASK J: whatever went out for this index stays a parseable JSON object
        self.close_args_truncated(out);
        self.flush_raw(out);
        // an abandoned call keeps its index: a later call must not land in the same slot
        if self.named {
            self.index += 1;
        }
        self.named = false;
        self.open_brace = false;
        self.state = TState::Text;
    }

    fn run(&mut self, out: &mut Vec<Emit>) {
        loop {
            match self.state {
                TState::Text => {
                    let (hit, safe) = find_marker(&self.buf, &[TOOL_OPEN]);
                    match hit {
                        Some((at, _)) if self.armed > 0 => {
                            let head = self.buf[..at].to_string();
                            self.content(&head, out);
                            self.buf.drain(..at + TOOL_OPEN.len());
                            self.armed -= 1;
                            self.raw.clear();
                            self.raw.push_str(TOOL_OPEN);
                            self.named = false;
                            self.name.clear();
                            self.open_brace = false;
                            self.state = TState::Func;
                        }
                        Some((at, _)) => {
                            let take = at + TOOL_OPEN.len();
                            let s = self.buf[..take].to_string();
                            self.content(&s, out);
                            self.buf.drain(..take);
                        }
                        None => {
                            if safe == 0 {
                                return;
                            }
                            let s = self.buf[..safe].to_string();
                            self.content(&s, out);
                            self.buf.drain(..safe);
                            return;
                        }
                    }
                }
                TState::Func => {
                    let (hit, safe) = find_marker(&self.buf, &[FUNCTION_OPEN, TOOL_CLOSE]);
                    match hit {
                        Some((at, 0)) => {
                            self.eat(at + FUNCTION_OPEN.len());
                            self.state = TState::FuncName;
                        }
                        Some((at, _)) => {
                            self.eat(at + TOOL_CLOSE.len());
                            self.give_up(out);
                        }
                        None => {
                            self.eat(safe);
                            return;
                        }
                    }
                }
                TState::FuncName => match self.read_name_to_gt() {
                    Ok(Some(name)) => {
                        self.name = name.clone();
                        self.named = true;
                        out.push(Emit::Call {
                            index: self.index,
                            id: format!("call_{}", self.index),
                            name,
                        });
                        self.state = TState::Body;
                    }
                    Ok(None) => return,
                    Err(()) => {
                        self.give_up(out);
                        continue;
                    }
                },
                TState::Body => {
                    let (hit, safe) =
                        find_marker(&self.buf, &[PARAM_OPEN, FUNCTION_CLOSE, TOOL_CLOSE]);
                    match hit {
                        Some((at, 0)) => {
                            self.eat(at + PARAM_OPEN.len());
                            self.pname.clear();
                            self.state = TState::ParamName;
                        }
                        Some((at, 1)) => {
                            self.eat(at + FUNCTION_CLOSE.len());
                            self.close_args(out);
                            // the arguments object is closed, so everything from here on is
                            // trailing markup: `finish` in `Tail` drops it and counts it
                            self.tail_from = self.raw.len();
                            self.state = TState::Tail;
                        }
                        Some((at, _)) => {
                            self.eat(at + TOOL_CLOSE.len());
                            self.close_args(out);
                            self.close_call();
                        }
                        None => {
                            self.eat(safe);
                            return;
                        }
                    }
                }
                TState::ParamName => match self.read_name_to_gt() {
                    Ok(Some(p)) => {
                        self.pname = p;
                        self.open_param(out);
                        self.state = TState::ParamValue;
                    }
                    Ok(None) => return,
                    Err(()) => {
                        self.give_up(out);
                        continue;
                    }
                },
                TState::ParamValue => {
                    if self.skip_nl {
                        if self.buf.starts_with('\n') {
                            self.eat(1);
                            self.skip_nl = false;
                        } else if self.buf.starts_with("\r\n") {
                            self.eat(2);
                            self.skip_nl = false;
                        } else if self.buf.is_empty() {
                            return;
                        } else {
                            self.skip_nl = false;
                        }
                    }
                    let (hit, safe) = find_marker(&self.buf, &[PARAM_CLOSE, TOOL_CLOSE]);
                    match hit {
                        Some((at, 0)) => {
                            let s = self.eat(at + PARAM_CLOSE.len());
                            let v = s[..at].to_string();
                            self.value(&v, true, out);
                            self.close_param(out);
                            self.state = TState::Body;
                        }
                        Some((at, _)) => {
                            let s = self.eat(at + TOOL_CLOSE.len());
                            let v = s[..at].to_string();
                            self.value(&v, true, out);
                            self.close_param(out);
                            self.close_args(out);
                            self.close_call();
                        }
                        None => {
                            if safe == 0 {
                                return;
                            }
                            let s = self.eat(safe);
                            self.value(&s, false, out);
                            return;
                        }
                    }
                }
                TState::Tail => {
                    let (hit, safe) = find_marker(&self.buf, &[TOOL_CLOSE]);
                    match hit {
                        Some((at, _)) => {
                            self.eat(at + TOOL_CLOSE.len());
                            self.close_call();
                        }
                        None => {
                            self.eat(safe);
                            return;
                        }
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// - the parser over a sequence of text pieces, with `arms` `<tool_call>` ids seen
    /// - returns the fragments, whether the end was malformed, and the completed calls
    fn drive(
        tools: Option<&serde_json::Value>,
        arms: usize,
        pieces: &[&str],
    ) -> (Vec<Emit>, bool, usize) {
        let mut ts = ToolStream::new(tools);
        for _ in 0..arms {
            ts.arm();
        }
        let mut out = Vec::new();
        for p in pieces {
            out.extend(ts.feed(p));
        }
        let bad = ts.finish(&mut out);
        (out, bad, ts.closed())
    }

    /// every `arguments` fragment of one index, in order
    fn frags(es: &[Emit], index: usize) -> Vec<String> {
        es.iter()
            .filter_map(|e| match e {
                Emit::Args { index: i, text } if *i == index => Some(text.clone()),
                _ => None,
            })
            .collect()
    }

    /// the concatenation of the `arguments` fragments of one index
    fn args_of(es: &[Emit], index: usize) -> String {
        frags(es, index).concat()
    }

    /// every `content` piece, concatenated
    fn content_of(es: &[Emit]) -> String {
        es.iter()
            .filter_map(|e| match e {
                Emit::Content(t) => Some(t.as_str()),
                _ => None,
            })
            .collect()
    }

    /// cut a string into pieces of at most `n` bytes, never inside a character
    fn cut(s: &str, n: usize) -> Vec<&str> {
        let mut out = Vec::new();
        let mut i = 0usize;
        while i < s.len() {
            let mut j = (i + n).min(s.len());
            while !s.is_char_boundary(j) {
                j -= 1;
            }
            out.push(&s[i..j]);
            i = j;
        }
        out
    }

    const A7_CALL: &str = "<tool_call>\n<function=read_file>\n<parameter=path>\n\
                           C:/Users/robin/dev/crow-nest/docs/ten-tasks.md\n</parameter>\n\
                           <parameter=start_line>\n1\n</parameter>\n</function>\n</tool_call>";

    #[test]
    fn one_tool_call_becomes_a_name_and_argument_fragments_that_concatenate() {
        let tools = a7_tools_fixture();
        let (es, bad, closed) = drive(Some(&tools), 1, &[A7_CALL]);
        assert!(!bad, "a complete call is not malformed");
        assert_eq!(closed, 1);
        // the name goes out exactly once, with a non empty id
        let calls: Vec<&Emit> = es.iter().filter(|e| matches!(e, Emit::Call { .. })).collect();
        assert_eq!(calls.len(), 1, "one Call fragment: {es:?}");
        assert_eq!(
            *calls[0],
            Emit::Call { index: 0, id: "call_0".to_string(), name: "read_file".to_string() }
        );
        // the concatenation is the arguments JSON object, and it parses
        let args = args_of(&es, 0);
        assert_eq!(
            args,
            "{\"path\":\"C:/Users/robin/dev/crow-nest/docs/ten-tasks.md\",\"start_line\":1}"
        );
        let v: serde_json::Value = serde_json::from_str(&args).expect("arguments parse");
        assert_eq!(v["path"], "C:/Users/robin/dev/crow-nest/docs/ten-tasks.md");
        assert_eq!(v["start_line"], 1);
        // the gate's clause: more than one fragment, so the reassembly is exercised
        assert!(frags(&es, 0).len() >= 2, "fragments: {:?}", frags(&es, 0));
        assert_eq!(content_of(&es), "", "no markup leaked into content");
    }

    #[test]
    fn the_same_call_split_byte_by_byte_gives_the_same_fragments() {
        let tools = a7_tools_fixture();
        let whole = args_of(&drive(Some(&tools), 1, &[A7_CALL]).0, 0);
        for n in [1usize, 2, 3, 5, 7, 11, 13] {
            let pieces = cut(A7_CALL, n);
            let (es, bad, closed) = drive(Some(&tools), 1, &pieces);
            assert!(!bad, "split {n} was called malformed");
            assert_eq!(closed, 1, "split {n}");
            assert_eq!(args_of(&es, 0), whole, "split {n}");
            assert_eq!(content_of(&es), "", "split {n} leaked markup into content");
            assert!(frags(&es, 0).len() >= 2, "split {n}");
        }
    }

    #[test]
    fn text_before_the_first_call_is_content_and_after_it_is_dropped() {
        let tools = a7_tools_fixture();
        let (es, _, closed) = drive(
            Some(&tools),
            1,
            &["I will read it.\n", A7_CALL, "\nand some trailing prose"],
        );
        assert_eq!(closed, 1);
        assert_eq!(content_of(&es), "I will read it.\n");
        // the content chunk comes BEFORE the call, so a reader sees the prose first
        assert!(matches!(es[0], Emit::Content(_)), "{es:?}");
        assert!(matches!(es[1], Emit::Call { .. }), "{es:?}");
    }

    /// one `run_command` declaration; the caller supplies the `properties`,
    /// so the only difference between the tests using it is one line
    fn run_command_tools(props: serde_json::Value) -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "run_command",
                "parameters": { "type": "object", "properties": props }
            }
        }])
    }

    #[test]
    fn braces_quotes_and_newlines_inside_a_value_survive_as_json() {
        let tools = run_command_tools(serde_json::json!({ "command": { "type": "string" } }));
        // a value with nested braces, a quote, a backslash and two real newlines
        let value = "python -c \"print({'a': {'b': 1}})\"\nC:\\tmp\\x\nend";
        let markup = format!(
            "<tool_call>\n<function=run_command>\n<parameter=command>\n{value}\n\
             </parameter>\n</function>\n</tool_call>"
        );
        for n in [1usize, 4, 9, 1000] {
            let pieces = cut(&markup, n);
            let (es, bad, closed) = drive(Some(&tools), 1, &pieces);
            assert!(!bad);
            assert_eq!(closed, 1);
            let args = args_of(&es, 0);
            let v: serde_json::Value = serde_json::from_str(&args).expect("arguments parse");
            assert_eq!(v["command"], value, "split {n}");
        }
    }

    #[test]
    fn an_undeclared_parameter_is_raw_json_when_it_is_json_and_a_string_otherwise() {
        // no tools at all: every type falls back to the value heuristic
        let markup = "<tool_call>\n<function=f>\n<parameter=n>\n42\n</parameter>\n\
                      <parameter=flag>\ntrue\n</parameter>\n\
                      <parameter=obj>\n{\"a\": [1, 2]}\n</parameter>\n\
                      <parameter=s>\nnot json\n</parameter>\n</function>\n</tool_call>";
        let (es, bad, closed) = drive(None, 1, &[markup]);
        assert!(!bad);
        assert_eq!(closed, 1);
        let args = args_of(&es, 0);
        let v: serde_json::Value = serde_json::from_str(&args).expect("arguments parse");
        assert_eq!(v["n"], 42);
        assert_eq!(v["flag"], true);
        assert_eq!(v["obj"], serde_json::json!({"a": [1, 2]}));
        assert_eq!(v["s"], "not json");
    }

    #[test]
    fn a_call_without_a_parameter_gets_an_empty_arguments_object() {
        let markup = "<tool_call>\n<function=list_dir>\n</function>\n</tool_call>";
        let (es, bad, closed) = drive(None, 1, &[markup]);
        assert!(!bad);
        assert_eq!(closed, 1);
        assert_eq!(args_of(&es, 0), "{}");
        assert_eq!(
            es[0],
            Emit::Call { index: 0, id: "call_0".to_string(), name: "list_dir".to_string() }
        );
    }

    #[test]
    fn two_calls_get_index_zero_and_one_with_their_own_ids() {
        let tools = a7_tools_fixture();
        let two = format!("{A7_CALL}\n{A7_CALL}");
        let (es, bad, closed) = drive(Some(&tools), 2, &cut(&two, 6));
        assert!(!bad);
        assert_eq!(closed, 2);
        let calls: Vec<&Emit> = es.iter().filter(|e| matches!(e, Emit::Call { .. })).collect();
        assert_eq!(calls.len(), 2);
        assert_eq!(
            *calls[0],
            Emit::Call { index: 0, id: "call_0".to_string(), name: "read_file".to_string() }
        );
        assert_eq!(
            *calls[1],
            Emit::Call { index: 1, id: "call_1".to_string(), name: "read_file".to_string() }
        );
        let a0 = args_of(&es, 0);
        assert_eq!(a0, args_of(&es, 1));
        serde_json::from_str::<serde_json::Value>(&a0).expect("arguments parse");
        // the newline between the two calls is dropped, not sent as content
        assert_eq!(content_of(&es), "");
    }

    #[test]
    fn a_call_that_never_closes_goes_out_as_content_and_is_malformed() {
        let tools = a7_tools_fixture();
        let cut_off = "<tool_call>\n<function=read_file>\n<parameter=path>\nC:/x/y.md";
        let (es, bad, closed) = drive(Some(&tools), 1, &cut(cut_off, 5));
        assert!(bad, "an unclosed call is malformed");
        assert_eq!(closed, 0);
        assert_eq!(content_of(&es), cut_off, "the raw markup is what the client sees");
        // TASK J: the half built arguments are CLOSED into a JSON object and marked. They
        // used to stay unterminated on purpose, and that unterminated string is what Crow
        // stored and re-sent until the chat template refused the whole session with a 400.
        let v: serde_json::Value =
            serde_json::from_str(&args_of(&es, 0)).expect("a truncated call still parses");
        assert!(v.is_object(), "{v}");
        assert_eq!(v[TRUNCATED_KEY], true, "the truncation is named: {v}");
        // the bytes the model did produce are still there, so the log shows what was asked
        assert_eq!(v["path"], "C:/x/y.md");
    }

    /// TASK J, the invariant that keeps a session alive: whatever the parser is fed, every
    /// index it NAMED leaves a parseable JSON OBJECT behind. The corpus is every byte prefix
    /// of a real call plus the give-up shapes, at four piece sizes.
    #[test]
    fn every_abandoned_call_leaves_a_parseable_arguments_object() {
        let tools = a7_tools_fixture();
        let long_name = "x".repeat(MAX_TOOL_NAME + 8);
        let corpus = [
            A7_CALL.to_string(),
            // the give-up shapes: a parameter name that never closes, after a complete one
            format!("<tool_call>\n<function=read_file>\n<parameter=path>\na.md\n</parameter>\n<parameter={long_name}"),
            // `</tool_call>` before `<function=`
            "<tool_call>\nplain text</tool_call>".to_string(),
            // a function name that never closes
            format!("<tool_call>\n<function={long_name}"),
            // a value that is not JSON, of a parameter that is not declared
            "<tool_call>\n<function=f>\n<parameter=n>\nnot json".to_string(),
        ];
        for markup in &corpus {
            for end in 1..=markup.len() {
                if !markup.is_char_boundary(end) {
                    continue;
                }
                for n in [1usize, 3, 7, 4096] {
                    let (es, _, _) = drive(Some(&tools), 2, &cut(&markup[..end], n));
                    let named: Vec<usize> = es
                        .iter()
                        .filter_map(|e| match e {
                            Emit::Call { index, .. } => Some(*index),
                            _ => None,
                        })
                        .collect();
                    for i in named {
                        let args = args_of(&es, i);
                        let v = serde_json::from_str::<serde_json::Value>(&args).unwrap_or_else(
                            |e| panic!("prefix {end} split {n} index {i}: {args:?} does not parse: {e}"),
                        );
                        assert!(v.is_object(), "prefix {end} split {n} index {i}: {args:?}");
                    }
                }
            }
        }
    }

    /// TASK J: the give-up path closes the arguments it had already opened, and the call
    /// that follows it lands in its own index with its own complete object.
    #[test]
    fn a_parameter_name_that_never_completes_closes_what_it_opened() {
        let long_name = "y".repeat(MAX_TOOL_NAME + 1);
        let markup = format!(
            "<tool_call>\n<function=f>\n<parameter=a>\n1\n</parameter>\n<parameter={long_name}>\n2\n</parameter>\n</function>\n</tool_call>"
        );
        let (es, _, _) = drive(None, 1, &[markup.as_str()]);
        let v: serde_json::Value =
            serde_json::from_str(&args_of(&es, 0)).expect("the abandoned call parses");
        assert_eq!(v["a"], 1);
        assert_eq!(v[TRUNCATED_KEY], true);
    }

    /// TASK J: a call the parser named but that ended before its first parameter gets an
    /// object too, not the empty string the open chunk started with.
    #[test]
    fn a_call_truncated_before_its_first_parameter_gets_a_marked_object() {
        let tools = a7_tools_fixture();
        for markup in ["<tool_call>\n<function=read_file>", "<tool_call>\n<function=read_file>\n<parameter=path"] {
            let (es, bad, _) = drive(Some(&tools), 1, &cut(markup, 2));
            assert!(bad, "{markup:?} is malformed");
            let args = args_of(&es, 0);
            assert_eq!(args, "{\"_truncated\":true}", "{markup:?}");
            let v: serde_json::Value = serde_json::from_str(&args).expect("parses");
            assert_eq!(v[TRUNCATED_KEY], true);
        }
    }

    #[test]
    fn a_function_name_that_never_completes_is_malformed() {
        let markup = "<tool_call>\n<function=read_file";
        let (es, bad, closed) = drive(None, 1, &[markup]);
        assert!(bad);
        assert_eq!(closed, 0);
        assert_eq!(content_of(&es), markup);
        assert!(es.iter().all(|e| matches!(e, Emit::Content(_))), "{es:?}");
    }

    #[test]
    fn only_the_tool_call_token_id_opens_a_call() {
        let tools = a7_tools_fixture();
        // the same bytes, but the decode loop never saw token 248058: it stays content
        let (es, bad, closed) = drive(Some(&tools), 0, &cut(A7_CALL, 4));
        assert!(!bad, "unarmed markup is ordinary text, not a broken call");
        assert_eq!(closed, 0);
        assert_eq!(content_of(&es), A7_CALL);
        assert!(es.iter().all(|e| matches!(e, Emit::Content(_))));
    }

    /// the round trip the gate measures: the parser's fragments, put back together the way
    /// `crow_core.py:4864-4877` does, give the call the model meant
    #[test]
    fn crows_reassembly_of_the_fragments_gives_one_complete_call() {
        let tools = a7_tools_fixture();
        let (es, _, _) = drive(Some(&tools), 1, &cut(A7_CALL, 3));
        // the reader: id and name only on a truthy value, arguments concatenated per index
        let mut id = String::new();
        let mut name = String::new();
        let mut args = String::new();
        let mut fragments = 0usize;
        for e in &es {
            match e {
                Emit::Call { id: cid, name: n, .. } => {
                    if !cid.is_empty() {
                        id = cid.clone();
                    }
                    if !n.is_empty() {
                        name = n.clone();
                    }
                }
                Emit::Args { text, .. } => {
                    if !text.is_empty() {
                        args.push_str(text);
                        fragments += 1;
                    }
                }
                Emit::Content(_) => panic!("content in a pure tool call: {es:?}"),
            }
        }
        assert_eq!(id, "call_0");
        assert_eq!(name, "read_file");
        assert!(fragments >= 2, "fragments {fragments}");
        let v: serde_json::Value = serde_json::from_str(&args).expect("json.loads equivalent");
        assert_eq!(v["path"], "C:/Users/robin/dev/crow-nest/docs/ten-tasks.md");
    }

    /// #29 review: `</function>` arrived, then EOS or `max_tokens` before `</tool_call>`.
    /// The call is COMPLETE, so the client must get the call and NOT its raw markup as well.
    #[test]
    fn end_of_stream_after_function_close_still_closes_the_call() {
        let tools = a7_tools_fixture();
        let head = A7_CALL
            .strip_suffix("\n</tool_call>")
            .expect("A7_CALL ends with the close");
        // the bare cut, and the cut with the newline the template puts before `</tool_call>`
        for tail in ["", "\n"] {
            let markup = format!("{head}{tail}");
            for n in [1usize, 3, 7, 1000] {
                let (es, bad, closed) = drive(Some(&tools), 1, &cut(&markup, n));
                assert!(!bad, "tail {tail:?} split {n}: a complete call is not malformed");
                assert_eq!(closed, 1, "tail {tail:?} split {n}: one call is emitted");
                let args = args_of(&es, 0);
                let v: serde_json::Value =
                    serde_json::from_str(&args).expect("arguments parse");
                assert_eq!(v["path"], "C:/Users/robin/dev/crow-nest/docs/ten-tasks.md");
                assert_eq!(v["start_line"], 1);
                assert_eq!(
                    content_of(&es),
                    "",
                    "tail {tail:?} split {n}: the raw markup must not be replayed"
                );
                let calls: Vec<&Emit> =
                    es.iter().filter(|e| matches!(e, Emit::Call { .. })).collect();
                assert_eq!(calls.len(), 1, "tail {tail:?} split {n}: {es:?}");
                assert_eq!(
                    *calls[0],
                    Emit::Call { index: 0, id: "call_0".to_string(), name: "read_file".to_string() }
                );
            }
        }
    }

    /// a `<` inside a value is not the start of a marker; `find_marker` holds it back for one
    /// piece at most and then lets it through
    #[test]
    fn a_less_than_inside_a_value_survives() {
        let tools = run_command_tools(serde_json::json!({ "command": { "type": "string" }, "note": {} }));
        for value in ["a < b", "x<y>z", "a<b</c>d", "<parameter=", "</param"] {
            let markup = format!(
                "<tool_call>\n<function=run_command>\n<parameter=command>\n{value}\n\
                 </parameter>\n</function>\n</tool_call>"
            );
            for n in [1usize, 2, 5, 1000] {
                let (es, bad, closed) = drive(Some(&tools), 1, &cut(&markup, n));
                assert!(!bad, "value {value:?} split {n}");
                assert_eq!(closed, 1, "value {value:?} split {n}");
                let args = args_of(&es, 0);
                let v: serde_json::Value =
                    serde_json::from_str(&args).expect("arguments parse");
                assert_eq!(v["command"], value, "value {value:?} split {n}");
                assert_eq!(content_of(&es), "", "value {value:?} split {n}");
            }
        }
    }
}

#[cfg(test)]
mod arguments_contract {
    //! TASK K: the `arguments` invariant of e2b9845 (module header, "The `arguments`
    //! invariant") is not argued here, it is SWEPT. robin's 2026-09-17 session carried a
    //! 3,112-byte `arguments` string with a multi-line HTML `content` parameter that
    //! `serde_json` refused, and the question was whether THIS parser can produce such a
    //! string. These two tests answer it: over ten hand-written hazards and four thousand
    //! random markups built from the marker fragments, quotes, backslashes, control
    //! characters and non-BMP scalars a `content` value can carry, at eight piece sizes
    //! and with the parameter types declared AND undeclared, every index the parser names
    //! concatenates to a parseable JSON OBJECT. Not one input produced anything else.
    use super::*;

    /// drive the parser over `markup` in `piece`-byte slices (on char boundaries), with one
    /// `arm()` per `<tool_call>` in it, and return the per-index `Args` concatenation
    fn run_case(tools: Option<&serde_json::Value>, markup: &str, piece: usize) -> Vec<String> {
        let mut ts = ToolStream::new(tools);
        let mut out = Vec::new();
        for _ in 0..markup.matches(TOOL_OPEN).count() {
            ts.arm();
        }
        let b = markup.as_bytes();
        let mut i = 0usize;
        while i < b.len() {
            let mut j = (i + piece).min(b.len());
            while !markup.is_char_boundary(j) {
                j += 1;
            }
            out.extend(ts.feed(&markup[i..j]));
            i = j;
        }
        ts.finish(&mut out);
        let mut args: Vec<String> = Vec::new();
        for e in &out {
            if let Emit::Args { index, text } = e {
                while args.len() <= *index {
                    args.push(String::new());
                }
                args[*index].push_str(text);
            }
        }
        args
    }

    /// the `write_file` shape of robin's session: `path` plus a `content` parameter
    /// declared as a string
    fn write_file_tools() -> serde_json::Value {
        serde_json::json!([{ "type": "function", "function": {
            "name": "write_file",
            "parameters": {"type": "object", "properties": {
                "path": {"type": "string"}, "content": {"type": "string"}, "n": {"type": "integer"}}},
        }}])
    }

    fn check(name: &str, args: &[String], bad: &mut usize, ctx: &str) {
        for (i, a) in args.iter().enumerate() {
            match serde_json::from_str::<serde_json::Value>(a) {
                Ok(serde_json::Value::Object(_)) => {}
                other => {
                    *bad += 1;
                    if *bad < 6 {
                        eprintln!("{name}: call {i} is not a JSON object ({ctx}): {other:?}\n  args {a:?}");
                    }
                }
            }
        }
    }

    /// the hazards a `write_file` call with an HTML body actually carries
    #[test]
    fn an_html_content_parameter_always_concatenates_to_an_object() {
        let tools = write_file_tools();
        let html = "<!doctype html>\n<html lang=\"en\" data-theme=\"dark\">\n<head>\n<meta charset=\"utf-8\">\n<title>C\\rowsheet</title>\n<style>\n  .a { content: \"\\201C\"; }\n\ttab\tindented\n</style>\n<script>\nconst re = /a\\\\b\"c/g;\nif (a < b && c > d) { console.log(\"x\\ty\"); }\n</script>\n</head>\n<body>\n<p>\u{e9}moji \u{1f426}\u{200d}\u{2b1b} and \u{1d54f} and \u{7f} and \u{b} vertical tab</p>\n</body>\n</html>\n";
        let cases: Vec<(&str, String)> = vec![
            ("plain html", format!("<tool_call>\n<function=write_file>\n<parameter=path>\n/home/x/a.html\n</parameter>\n<parameter=content>\n{html}</parameter>\n</function>\n</tool_call>")),
            ("content holds </parameter>", "<tool_call>\n<function=write_file>\n<parameter=path>\n/home/x/a.html\n</parameter>\n<parameter=content>\n<p>see </parameter> inside</p>\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("content holds <parameter=", "<tool_call>\n<function=write_file>\n<parameter=path>\n/x\n</parameter>\n<parameter=content>\n<parameter=evil>text\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("undeclared tool", format!("<tool_call>\n<function=unknown_tool>\n<parameter=content>\n{html}</parameter>\n</function>\n</tool_call>")),
            ("weird parameter name", "<tool_call>\n<function=write_file>\n<parameter=a\"b\\c\td>\nv\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("empty value", "<tool_call>\n<function=write_file>\n<parameter=content>\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("crlf value", "<tool_call>\n<function=write_file>\n<parameter=content>\r\nline1\r\nline2\r\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("value ends with a backslash", "<tool_call>\n<function=write_file>\n<parameter=content>\nends with \\\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("no separator newline", "<tool_call>\n<function=write_file>\n<parameter=content>x\ny\n</parameter>\n</function>\n</tool_call>".to_string()),
            ("cut in the middle of the html", format!("<tool_call>\n<function=write_file>\n<parameter=path>\n/x\n</parameter>\n<parameter=content>\n{}", &html[..60])),
        ];
        let mut bad = 0;
        for (name, markup) in &cases {
            for piece in [1usize, 2, 3, 5, 7, 13, 64, 100_000] {
                for t in [Some(&tools), None] {
                    let args = run_case(t, markup, piece);
                    check(name, &args, &mut bad, &format!("piece {piece}, tools {}", t.is_some()));
                }
            }
        }
        assert_eq!(bad, 0, "{bad} arguments strings did not parse as an object");
    }

    fn lcg(st: &mut u64) -> u64 {
        *st = st.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        *st >> 33
    }

    /// four thousand random markups over the alphabet that can break JSON escaping or the
    /// marker scan, each at four piece sizes with and without declared types
    #[test]
    fn random_parameter_values_always_concatenate_to_an_object() {
        let tools = write_file_tools();
        let atoms: Vec<&str> = vec![
            "<", ">", "/", "parameter", "function", "tool_call", "=", "<parameter=", "</parameter>",
            "</function>", "</tool_call>", "<tool_call>", "\\", "\"", "\n", "\r", "\t", "\u{b}", "\u{0}",
            "\u{7f}", "\u{1b}", "a", " ", "\u{e9}", "\u{1f426}", "\u{2028}", "</", "<!", "-->",
            "{", "}", "[", "]", ":", ",", "0", "true", "null", "\u{fffd}",
        ];
        let mut st: u64 = 0x5eed_1234_9abc_def0;
        let mut bad = 0usize;
        for case in 0..4000u32 {
            let nparam = 1 + (lcg(&mut st) % 3) as usize;
            let mut markup = String::from("<tool_call>\n<function=write_file>\n");
            for pi in 0..nparam {
                let pname = ["path", "content", "n", "weird name", "a\"b"][(lcg(&mut st) % 5) as usize];
                markup.push_str(&format!("<parameter={pname}>\n"));
                let n = (lcg(&mut st) % 24) as usize;
                for _ in 0..n {
                    markup.push_str(atoms[(lcg(&mut st) % atoms.len() as u64) as usize]);
                }
                if !lcg(&mut st).is_multiple_of(8) || pi + 1 < nparam {
                    markup.push_str("\n</parameter>\n");
                }
            }
            if !lcg(&mut st).is_multiple_of(6) {
                markup.push_str("</function>\n");
            }
            if !lcg(&mut st).is_multiple_of(6) {
                markup.push_str("</tool_call>");
            }
            for piece in [1usize, 3, 17, 100_000] {
                for t in [Some(&tools), None] {
                    let args = run_case(t, &markup, piece);
                    check("random", &args, &mut bad, &format!("case {case}, piece {piece}, markup {markup:?}"));
                }
            }
        }
        assert_eq!(bad, 0, "{bad} random arguments strings did not parse as an object");
    }
}
