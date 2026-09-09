//! #24 serve: the HTTP face of the engine (spec section 7).
//!
//! Endpoints:
//!
//! - `GET /health` answers `{"status":"ok"}`.
//! - `GET /props` answers the operating point as a JSON document.
//! - `/props` field names mirror llama-server, so Crow's readers work unchanged.
//! - Crow readers served: `check_endpoint`, `fetch_n_ctx`, `fetch_model_name`.
//! - Crow readers served: `refuse_images`, `server_model_path`.
//! - `POST /v1/chat/completions` streams the answer as SSE (#26 A4).
//! - Anything else answers 404 with a JSON body.
//!
//! Connection handling:
//!
//! - Read and write timeout of 10 s on every accepted connection.
//! - A timeout or read error logs one stderr line and closes that connection.
//! - The accept loop continues after any single connection failed.
//! - Head (request line plus headers) capped at 64 KiB total.
//! - The reader itself is bounded, it never buffers past that cap.
//! - Over the head cap: 431 JSON, then close.
//! - `Content-Length` is parsed case insensitively.
//! - A malformed or repeated `Content-Length` answers 400 JSON.
//! - `Transfer-Encoding: chunked` answers 501 JSON, A3 may implement it.
//! - The body is read and discarded before the response is written.
//! - Body capped at 16 MiB; over the cap: 413 JSON, then close.
//! - A bodied POST to an unknown route therefore gets the 404 JSON, not a reset.
//! - A garbage request line answers 400 JSON with `{"error":"bad request"}`.
//! - A client that sent nothing is closed silently, without a response.
//! - The write half is shut down after the response is flushed.
//! - The parsed body is kept as `Vec<u8>` for A3 and A4; A2 routes ignore it.
//!
//! Operating point (M1 decisions, robin 2026-09-09):
//!
//! - Container default `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`.
//! - `CROW_CNQ` overrides the container path.
//! - Hot sets default `decode_out/hotsets-M-longctx2100-n160.json`.
//! - `CROW_HOTSETS` overrides the hot set sidecar.
//! - Prompt chunk pinned at 2048 for the whole process.
//! - The per prompt policy `geo::apply_chunk_policy` is NOT applied.
//! - Same result as `CROW_CHUNK=2048 CROW_CHUNK_AUTO=0` before `Engine::load`.
//! - Context `CONTEXT_FLOOR` (200,000), the operating point of `decode run`.
//! - `n_ctx` is read back from the loaded states (`Engine::st.context`).
//! - Process wide defaults since d36353a need no env variable.
//! - Those defaults: `CROW_PF_ASYNC=2`, PF_TG 64, PLE 128 MB, attn_sel_s8l.
//!
//! Environment defaults set by `serve` itself (#26 review):
//!
//! | variable | behaviour |
//! |---|---|
//! | `CROW_GRAPH`, `CROW_MMA` | default 1 in serve (the gated configuration); env overrides |
//!
//! - Both are set in `main` BEFORE `cuda::Ctx::init` and before the first kernel call.
//! - `gen.rs` reads each one once through a `OnceLock`, so the order is the whole contract.
//! - Only an unset variable is set; an explicit `CROW_GRAPH=0` still turns graphs off.
//! - One stderr line per variable carries the effective value.
//!
//! Hot set adaptation (#26 review):
//!
//! - `apply_adapt_policy` prints a `[policy] ... every N ...` line at start.
//! - `serve` calls NEITHER `trickle_tick` NOR `adapt_tick`; `decode.rs:216-228` is the only caller.
//! - The hot set therefore stays the loaded one for the whole process life.
//! - One stderr line after the policy line says so, so the log cannot mislead.
//! - Turning it on is an A6/A9 decision: greedy identity comes first.
//!
//! Concurrency (constant 4: one binary, no runtime):
//!
//! - `std::net::TcpListener`, blocking, one request at a time, no async stack.
//! - A second connection waits in the accept queue; no 503.
//!
//! One engine per machine:
//!
//! - `Engine::load` takes `engine/.engine.lock` before anything is pinned.
//! - A second `serve` therefore exits non zero with the lock message.
//! - The engine loads BEFORE the socket binds, so the lock speaks first.
//!
//! Unsafe surface:
//!
//! - `cuda::Ctx::init` and `Engine::load` are the only unsafe calls.
//! - Argument parsing, the socket loop and the routes are safe code.
//!
//! Run:
//!
//! - `serve [--port <n>]` from the repository root (paths are repo relative).
//! - Default port 8099, bind address 127.0.0.1.
//!
//! Subcommand `tokenize` (#25 A3, no engine, no GPU, no lock):
//!
//! | invocation | effect |
//! |---|---|
//! | `serve tokenize --chat --file <prompts.json> --out <ids.json>` | every prompt as one user message |
//! | `serve tokenize --chat --text "<text>"` | one prompt, id list on stdout |
//! | `serve tokenize --raw --text "<text>"` | `add_special_tokens=false`, id list on stdout |
//!
//! - `--chat` is `add_generation_prompt=true`, `enable_thinking=false`, as `tokenize_ids.py --chat`.
//! - `<prompts.json>` is read as `{id: text}` or as `[{"id": ..., "text": ...}]`.
//! - `<ids.json>` is written as `{task_id: [ids]}`.
//! - Per task the id count goes to stderr, so the gate log carries the ten lengths.
//! - The subcommand returns before `cuda::Ctx::init`, so no CUDA context and no `.engine.lock`.
//! - No Python process is started; `crow_nest_engine::tokenizer` is the whole path.
//!
//! Exit codes of `serve tokenize`:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | ids written (`--file`) or printed (`--text`) |
//! | 2 | usage: bad, missing, doubled or conflicting arguments |
//! | 3 | tokenizer load failed (`tokenizer.json` or `tokenizer_config.json`) |
//! | 4 | IO or encode: prompt file unreadable, not JSON, wrong shape, encode failed, output unwritable |
//!
//! Start of `serve` (no subcommand):
//!
//! - The tokenizer is warmed up right after argument parsing, BEFORE `cuda::Ctx::init`.
//! - A missing or broken tokenizer therefore fails in a second, not after the engine load.
//! - Warm-up failure prints the error plus both paths and exits 3.
//! - Warm-up success logs the two loaded paths, one stderr line each.
//!
//! `POST /v1/chat/completions` (#26 A4), request body:
//!
//! | field | A4 behaviour |
//! |---|---|
//! | `messages` | required, non empty array, every entry needs a string `role` |
//! | `stream` | `true` streams; `false` or absent answers 501 (A5) |
//! | `max_tokens` | default 1024, capped at 32768 |
//! | `model` | echoed into every chunk, default `crow-nest` |
//! | `chat_template_kwargs.enable_thinking` | template variable, default false |
//! | `temperature`, `top_p`, `min_p`, `top_k` | ACCEPTED AND IGNORED, A4 is greedy (A6 samples) |
//! | `tools` | ACCEPTED AND IGNORED, the render carries `messages` only (A7) |
//! | `stream_options`, `timings_per_token` | ACCEPTED AND IGNORED (A5 adds usage and timings) |
//!
//! Rendering and generation:
//!
//! - `tokenizer::render_chat(messages, None, add_generation_prompt=true, enable_thinking)`.
//! - Greedy decode: `Engine::prefill` gives the first id, `Engine::decode_step` the rest.
//! - Stops on `sample::EOS_IDS` (`finish_reason` `stop`) or at `max_tokens` (`length`).
//! - `prompt ids >= n_ctx` answers 413 before any GPU work.
//! - Otherwise `max_tokens` is CLAMPED to `n_ctx - prompt ids` and to 32768.
//! - A clamp logs one stderr line and the request is served, not refused.
//!
//! Stream shape (llama-server / OpenAI, `crow_core.py:4831-4877`):
//!
//! | order | line |
//! |---|---|
//! | 1 | `data: {... "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` |
//! | 2..n | `data: {... "delta":{"content":"..."},"finish_reason":null}` |
//! | n+1 | `data: {... "delta":{},"finish_reason":"stop"}` |
//! | n+2 | `data: [DONE]` |
//!
//! - Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: close`.
//! - No `Content-Length` and no chunked framing: the body is close delimited.
//! - Every frame is flushed on its own; one token can never wait for the next.
//!
//! Incremental detokenization:
//!
//! - The accumulated generated ids are decoded after every token.
//! - Only the new byte suffix is sent, so the text arrives exactly once.
//! - A tail that is not a whole character is HELD BACK, not sent.
//! - The byte level decoder renders an incomplete UTF-8 sequence as U+FFFD.
//! - No chunk therefore carries a replacement character from a split token.
//! - Assumed of `decode`: prefix stability across one more id; a violation is lossy, never a panic.
//! - Cost: one decode of the whole answer per token, microseconds against ms of GPU.
//!
//! One conversation at a time (M1):
//!
//! - `Engine::reset_to_zero` runs before every prefill, so request k equals a fresh process.
//! - The reset field list and its evidence live in `engine/src/reset.rs`.
//! - A write error aborts the generation loop; the next request resets the state anyway.
//! - One stderr line per request: prompt tokens, generated tokens, prefill ms, decode ms.
//! - One stderr line per request with the generated ids, for the A4 identity gate.

use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::gen::Engine;
use crow_nest_engine::geo::{apply_adapt_policy, Config, CONTEXT_FLOOR};
use crow_nest_engine::sample::EOS_IDS;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PORT: u16 = 8099;
const DEFAULT_CNQ: &str = "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq";
const DEFAULT_HOTSETS: &str = "decode_out/hotsets-M-longctx2100-n160.json";
/// pinned for the process (M1): every request prefills at chunk 2048
const SERVE_CHUNK: usize = 2048;
/// read and write timeout per connection, so one stalled client cannot hold the loop
const IO_TIMEOUT_SECS: u64 = 10;
/// request line plus headers, 64 KiB total; over it the answer is 431
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// declared `Content-Length`, 16 MiB; over it the answer is 413
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// `max_tokens` when the request carries none
const DEFAULT_MAX_TOKENS: usize = 1024;
/// ceiling for `max_tokens`, so one request cannot hold the process forever
const MAX_MAX_TOKENS: usize = 32768;
/// the last line of every stream
const SSE_DONE: &str = "data: [DONE]

";

// ---------------------------------------------------------------- pure parts

/// the route a `(method, path)` pair dispatches to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `GET /health`
    Health,
    /// `GET /props`
    Props,
    /// `POST /v1/chat/completions` (#26 A4)
    Chat,
    /// everything else, a wrong method on a known path included
    NotFound,
}

/// the whole dispatch table, pure so the test can drive it
fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/health") => Route::Health,
        ("GET", "/props") => Route::Props,
        ("POST", "/v1/chat/completions") => Route::Chat,
        _ => Route::NotFound,
    }
}

/// what one connection sent, as far as the head reader could tell
#[derive(Debug, PartialEq, Eq)]
enum Head {
    /// the client closed without sending a byte; no response is written
    Empty,
    /// there was a request line, but it is not HTTP; the answer is 400
    Bad,
    /// the head passed `MAX_HEAD_BYTES`; the answer is 431
    HeadTooLarge,
    /// the declared body passed `MAX_BODY_BYTES`; the answer is 413
    BodyTooLarge(usize),
    /// the head asks for chunked transfer encoding; the answer is 501
    Chunked,
    /// a well formed request; `body` is drained and kept for A3 and A4
    Req { method: String, target: String, body: Vec<u8> },
}

/// - `--port <n>` and `--port=<n>` out of the argument vector
/// - argv[0] is skipped
/// - no `--port` flag means the default port
/// - `Err` carries the message for the operator
fn parse_port(args: &[String]) -> Result<u16, String> {
    let mut i = 1;
    let mut port = DEFAULT_PORT;
    while i < args.len() {
        let a = args[i].as_str();
        let val = if a == "--port" {
            i += 1;
            args.get(i).cloned().ok_or_else(|| "--port needs a number".to_string())?
        } else if let Some(v) = a.strip_prefix("--port=") {
            v.to_string()
        } else {
            return Err(format!("unknown argument {a:?} (usage: serve [--port <n>])"));
        };
        port = val.parse::<u16>().map_err(|_| format!("bad port {val:?}"))?;
        i += 1;
    }
    Ok(port)
}

// ------------------------------------------------- tokenize subcommand (A3)

/// what `serve tokenize ...` was asked to do
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    /// every prompt of `file` through the chat template, ids written to `out`
    File { chat: bool, file: String, out: String },
    /// one text, id list on stdout
    Text { chat: bool, text: String },
}

/// - argument vector AFTER the `tokenize` word
/// - exactly one of `--chat` and `--raw`
/// - `--file` needs `--out`; `--file` and `--text` are exclusive
fn parse_tokenize(rest: &[String]) -> Result<Tok, String> {
    const USAGE: &str = "usage: serve tokenize (--chat|--raw) (--file <prompts.json> --out <ids.json> | --text <text>)";
    let mut chat: Option<bool> = None;
    let (mut file, mut out, mut text) = (None, None, None);
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].as_str();
        let slot: &mut Option<String> = match a {
            "--chat" | "--raw" => {
                if chat.replace(a == "--chat").is_some() {
                    return Err(format!("mode given twice ({USAGE})"));
                }
                i += 1;
                continue;
            }
            "--file" => &mut file,
            "--out" => &mut out,
            "--text" => &mut text,
            other => return Err(format!("unknown argument {other:?} ({USAGE})")),
        };
        i += 1;
        let v = rest.get(i).ok_or_else(|| format!("{a} needs a value ({USAGE})"))?;
        if slot.replace(v.clone()).is_some() {
            return Err(format!("{a} given twice ({USAGE})"));
        }
        i += 1;
    }
    let chat = chat.ok_or_else(|| format!("one of --chat and --raw is required ({USAGE})"))?;
    match (file, out, text) {
        (Some(_), _, Some(_)) => Err(format!("--file and --text are exclusive ({USAGE})")),
        (Some(f), Some(o), None) => Ok(Tok::File { chat, file: f, out: o }),
        (Some(_), None, None) => Err(format!("--file needs --out ({USAGE})")),
        (None, _, Some(t)) => Ok(Tok::Text { chat, text: t }),
        (None, _, None) => Err(format!("one of --file and --text is required ({USAGE})")),
    }
}

/// - `{id: text}` and `[{"id": ..., "text": ...}]` both give the same pairs
/// - array shape: the array order
/// - object shape: the order the keys stand in the file (`serde_json` feature `preserve_order`)
/// - without that feature the object shape would come back sorted by key
fn prompts_from_json(v: &serde_json::Value) -> Result<Vec<(String, String)>, String> {
    if let Some(a) = v.as_array() {
        let mut out = Vec::with_capacity(a.len());
        for (i, p) in a.iter().enumerate() {
            let id = p["id"].as_str().ok_or_else(|| format!("entry {i} has no string id"))?;
            let text = p["text"].as_str().ok_or_else(|| format!("entry {id} has no string text"))?;
            out.push((id.to_string(), text.to_string()));
        }
        return Ok(out);
    }
    if let Some(m) = v.as_object() {
        let mut out = Vec::with_capacity(m.len());
        for (k, t) in m {
            let text = t.as_str().ok_or_else(|| format!("entry {k} is not a string"))?;
            out.push((k.clone(), text.to_string()));
        }
        return Ok(out);
    }
    Err("prompts file is neither an object nor an array".to_string())
}

/// `serve tokenize ...`, the gate arm; exit code is the return value
fn tokenize_main(rest: &[String]) -> i32 {
    let cmd = match parse_tokenize(rest) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[tokenize] {e}");
            return 2;
        }
    };
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[tokenize] {e}");
            return 3;
        }
    };
    let (tp, cp) = tk.paths();
    eprintln!("[tokenize] tokenizer {tp}");
    eprintln!("[tokenize] chat template {cp}");

    let ids_of = |chat: bool, text: &str| -> Result<Vec<u32>, String> {
        if chat {
            tk.encode_chat_user(text)
        } else {
            tk.encode_raw(text)
        }
    };

    match cmd {
        Tok::Text { chat, text } => match ids_of(chat, &text) {
            Ok(ids) => {
                println!("{}", serde_json::json!(ids));
                eprintln!("[tokenize] {} tokens", ids.len());
                0
            }
            Err(e) => {
                eprintln!("[tokenize] {e}");
                4
            }
        },
        Tok::File { chat, file, out } => {
            let raw = match std::fs::read(&file) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[tokenize] cannot read {file}: {e}");
                    return 4;
                }
            };
            let doc: serde_json::Value = match serde_json::from_slice(&raw) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[tokenize] {file} is not JSON: {e}");
                    return 4;
                }
            };
            let prompts = match prompts_from_json(&doc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[tokenize] {file}: {e}");
                    return 4;
                }
            };
            let mut map = serde_json::Map::new();
            for (id, text) in &prompts {
                match ids_of(chat, text) {
                    Ok(ids) => {
                        eprintln!("[tokenize] {id} chars {} tokens {}", text.chars().count(), ids.len());
                        map.insert(id.clone(), serde_json::json!(ids));
                    }
                    Err(e) => {
                        eprintln!("[tokenize] {id}: {e}");
                        return 4;
                    }
                }
            }
            let text = serde_json::Value::Object(map).to_string();
            if let Err(e) = std::fs::write(&out, text) {
                eprintln!("[tokenize] cannot write {out}: {e}");
                return 4;
            }
            eprintln!("[tokenize] {} prompts -> {out}", prompts.len());
            0
        }
    }
}

/// method + request target of an HTTP request line, `None` when it is not one.
fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut it = line.trim_end_matches(['\r', '\n']).split_whitespace();
    let method = it.next()?.to_string();
    let target = it.next()?.to_string();
    let version = it.next()?;
    if !version.starts_with("HTTP/") || !target.starts_with('/') {
        return None;
    }
    Some((method, target))
}

/// value of one header line when it carries `name`, matched case insensitively
fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) {
        Some(v.trim())
    } else {
        None
    }
}

/// request target without query string or fragment
fn route_path(target: &str) -> &str {
    let cut = target.find(['?', '#']).unwrap_or(target.len());
    let p = &target[..cut];
    // "/props/" and "/props" are the same route
    if p.len() > 1 { p.trim_end_matches('/') } else { p }
}

/// container file stem, the name Crow shows (`fetch_model_name`)
fn model_name(model_path: &str) -> String {
    let base = model_path.replace('\\', "/");
    let base = base.rsplit('/').next().unwrap_or(&base);
    base.strip_suffix(".cnq").unwrap_or(base).to_string()
}

/// the `/props` document; `n_ctx` comes from the loaded engine, not a constant
fn props_json(model_path: &str, n_ctx: usize, prompt_chunk: usize) -> serde_json::Value {
    serde_json::json!({
        "model_path": model_path,
        "model": model_name(model_path),
        "n_ctx": n_ctx,
        "default_generation_settings": { "n_ctx": n_ctx },
        // crow-nest is text only; Crow's refuse_images answers BLIND_SERVER_HINT
        "modalities": { "vision": false },
        "prompt_chunk": prompt_chunk,
        "build": "crow-nest-engine 0.1.0",
    })
}

fn not_found_json(path: &str) -> serde_json::Value {
    serde_json::json!({
        "error": { "code": 404, "message": format!("no route {path} (serve answers GET /health, GET /props, POST /v1/chat/completions)") }
    })
}

/// flat error document for 400, 413, 431 and 501
fn error_json(msg: &str) -> serde_json::Value {
    serde_json::json!({ "error": msg })
}

// --------------------------------------------- chat completions (#26 A4)

/// the fields of a `POST /v1/chat/completions` body that A4 acts on
#[derive(Debug, Clone, PartialEq)]
struct ChatReq {
    /// echoed into every chunk's `model`
    model: String,
    /// the OpenAI message array, handed to `render_chat` unchanged
    messages: serde_json::Value,
    /// `false` (or absent) answers 501 in A4
    stream: bool,
    /// generation budget, `DEFAULT_MAX_TOKENS` when absent
    max_tokens: usize,
    /// `chat_template_kwargs.enable_thinking`, a template variable
    enable_thinking: bool,
}

/// - the request body, as Crow sends it (`crow_core.py:4672-4700`)
/// - unknown fields are accepted and ignored, as llama-server does
/// - `temperature`, `top_p`, `min_p`, `top_k`, `tools` fall under that rule in A4
/// - `Err` carries the message for the 400 body
fn parse_chat(body: &[u8]) -> Result<ChatReq, String> {
    let doc: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| "body is not a JSON object".to_string())?;
    let messages = obj
        .get("messages")
        .ok_or_else(|| "no messages in the body".to_string())?;
    let arr = messages
        .as_array()
        .ok_or_else(|| "messages is not an array".to_string())?;
    if arr.is_empty() {
        return Err("messages is empty".to_string());
    }
    for (i, m) in arr.iter().enumerate() {
        if !m.get("role").map(|r| r.is_string()).unwrap_or(false) {
            return Err(format!("message {i} has no string role"));
        }
    }
    let stream = match obj.get("stream") {
        None | Some(serde_json::Value::Null) => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "stream is not a boolean".to_string())?,
    };
    let max_tokens = match obj.get("max_tokens") {
        None | Some(serde_json::Value::Null) => DEFAULT_MAX_TOKENS,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| "max_tokens is not a positive integer".to_string())?;
            if n == 0 {
                return Err("max_tokens is 0".to_string());
            }
            (n as usize).min(MAX_MAX_TOKENS)
        }
    };
    let enable_thinking = obj
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = obj
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("crow-nest")
        .to_string();
    Ok(ChatReq { model, messages: messages.clone(), stream, max_tokens, enable_thinking })
}

/// one `chat.completion.chunk`, the only object shape this endpoint streams
fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": match finish {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
        }],
    })
}

/// first chunk: the role, no content (`crow_core.py:4877` reads content only)
fn chunk_role(id: &str, created: u64, model: &str) -> serde_json::Value {
    chunk(id, created, model, serde_json::json!({ "role": "assistant" }), None)
}

/// a content delta chunk, one per emitted text piece
fn chunk_content(id: &str, created: u64, model: &str, text: &str) -> serde_json::Value {
    chunk(id, created, model, serde_json::json!({ "content": text }), None)
}

/// last chunk before `[DONE]`: empty delta, the finish reason
fn chunk_finish(id: &str, created: u64, model: &str, reason: &str) -> serde_json::Value {
    chunk(id, created, model, serde_json::json!({}), Some(reason))
}

/// one SSE event: `data: <compact json>` plus the blank line that ends it
fn sse_frame(doc: &serde_json::Value) -> String {
    format!("data: {doc}\n\n")
}

/// - the generation budget one request may actually spend
/// - `None` means the prompt alone does not fit `n_ctx`: the answer is 413
/// - `Some(n)`: `max_tokens`, clamped to the free context and to `MAX_MAX_TOKENS`
/// - a budget that does not fit is clamped, never refused (llama-server does the same)
/// - pure: no engine, no socket, so the test drives the arithmetic directly
fn clamped_max_tokens(prompt_ids: usize, max_tokens: usize, n_ctx: usize) -> Option<usize> {
    if prompt_ids >= n_ctx {
        return None;
    }
    Some(max_tokens.min(n_ctx - prompt_ids).min(MAX_MAX_TOKENS))
}

/// - `full` is `decode` over EVERY generated id so far
/// - `emitted` is how many BYTES of `full` already left as content
/// - `None` holds the delta back until the tail is a whole character
/// - the byte level decoder renders an incomplete UTF-8 sequence as U+FFFD
/// - so no chunk this returns ends in a replacement character
/// - ASSUMED of `decode`: prefix stability, `decode(ids[..k])` is a byte prefix of `decode(ids[..k+1])`
/// - a violation is LOSSY, never a panic: the changed bytes below `emitted` are never re-sent,
///   and the `is_char_boundary` guard keeps the slice legal for any input
fn next_delta(full: &str, emitted: usize) -> Option<&str> {
    if full.len() <= emitted || !full.is_char_boundary(emitted) {
        return None;
    }
    if full.ends_with(char::REPLACEMENT_CHARACTER) {
        return None;
    }
    Some(&full[emitted..])
}

/// write one SSE frame and flush it; `false` means the client is gone
fn sse_send(stream: &mut TcpStream, text: &str) -> bool {
    match stream.write_all(text.as_bytes()).and_then(|_| stream.flush()) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[chat] write failed, aborting the generation: {e}");
            false
        }
    }
}

/// a JSON response plus its status, so the caller can log one label
fn respond_json(
    stream: &mut TcpStream,
    status: &'static str,
    doc: &serde_json::Value,
) -> &'static str {
    if let Err(e) = respond(stream, status, &doc.to_string()) {
        eprintln!("[serve] response write failed: {e}");
    }
    status
}

/// - `POST /v1/chat/completions`
/// - every rejection happens BEFORE the first stream byte, as a JSON response
/// - the return value is the status for the access log line
fn chat_route(stream: &mut TcpStream, srv: &mut Srv, body: &[u8]) -> &'static str {
    let mut req = match parse_chat(body) {
        Ok(r) => r,
        Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
    };
    if !req.stream {
        return respond_json(
            stream,
            "501 Not Implemented",
            &error_json("stream:false is not implemented yet (A5)"),
        );
    }
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };
    // A4 ignores `tools`: the render carries the messages only
    let ids = match tk.encode_chat(&req.messages, None, true, req.enable_thinking) {
        Ok(v) => v,
        Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
    };
    if ids.is_empty() {
        return respond_json(stream, "400 Bad Request", &error_json("the rendered prompt is empty"));
    }
    // #26 review: the prompt alone is the only 413 case; a budget that does not fit is CLAMPED,
    // not refused (the old combined check was dead, `max_tokens` is capped at 32768 first)
    let budget = match clamped_max_tokens(ids.len(), req.max_tokens, srv.n_ctx) {
        Some(n) => n,
        None => {
            let m = format!("prompt {} tokens over n_ctx {}", ids.len(), srv.n_ctx);
            return respond_json(stream, "413 Payload Too Large", &error_json(&m));
        }
    };
    if budget != req.max_tokens {
        eprintln!(
            "[chat] max_tokens {} clamped to {} (prompt {}, n_ctx {})",
            req.max_tokens,
            budget,
            ids.len(),
            srv.n_ctx
        );
        req.max_tokens = budget;
    }
    chat_stream(stream, srv, &req, &ids)
}

/// - prefill, then greedy decode, one flushed SSE frame per emitted delta
/// - the engine is reset to position 0 first, so request k equals a fresh process
/// - a write error breaks the loop; the engine stays dirty and the next reset cleans it
fn chat_stream(stream: &mut TcpStream, srv: &mut Srv, req: &ChatReq, ids: &[u32]) -> &'static str {
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };
    srv.seq += 1;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let id = format!("chatcmpl-{created}-{}", srv.seq);
    let model = req.model.clone();

    const HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if !sse_send(stream, HEAD) {
        return "200 OK (client gone)";
    }

    let prompt: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
    let t_pre = Instant::now();
    // unsafe: engine kernels; the CUDA context and engine/.engine.lock are this process's
    let mut next = unsafe {
        srv.eng.reset_to_zero();
        srv.eng.prefill(srv.cnq, &prompt, None)
    };
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1e3;

    let mut aborted = !sse_send(stream, &sse_frame(&chunk_role(&id, created, &model)));

    let t_dec = Instant::now();
    let mut out: Vec<u32> = Vec::with_capacity(req.max_tokens);
    // bytes of the accumulated decode that already left as content
    let mut emitted = 0usize;
    let mut content_chunks = 0usize;
    let mut finish = "length";
    let mut decode_ms = 0.0f64;
    if !aborted {
        for i in 0..req.max_tokens {
            if EOS_IDS.contains(&next) {
                finish = "stop";
                break;
            }
            out.push(next as u32);
            let full = match tk.decode(&out) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[chat] detokenize failed: {e}");
                    String::new()
                }
            };
            if let Some(delta) = next_delta(&full, emitted) {
                if !sse_send(stream, &sse_frame(&chunk_content(&id, created, &model, delta))) {
                    aborted = true;
                    break;
                }
                content_chunks += 1;
                emitted = full.len();
            }
            // the budget is spent: no decode_step whose token nobody reads
            if i + 1 == req.max_tokens {
                break;
            }
            next = unsafe { srv.eng.decode_step(srv.cnq, next as i64) };
        }
        decode_ms = t_dec.elapsed().as_secs_f64() * 1e3;
    }

    // the tail the hold back kept (a never completed sequence, or a real U+FFFD)
    if !aborted {
        let full = tk.decode(&out).unwrap_or_default();
        if full.len() > emitted && full.is_char_boundary(emitted) {
            if sse_send(stream, &sse_frame(&chunk_content(&id, created, &model, &full[emitted..]))) {
                content_chunks += 1;
            } else {
                aborted = true;
            }
        }
    }
    if !aborted {
        let _ = sse_send(stream, &sse_frame(&chunk_finish(&id, created, &model, finish)))
            && sse_send(stream, SSE_DONE);
    }

    let gen = out.len();
    eprintln!(
        "[chat] prompt {} tok, generated {gen} tok, prefill {prefill_ms:.1} ms, decode {decode_ms:.1} ms, {:.1} tok/s, finish {finish}, content chunks {content_chunks}{}",
        ids.len(),
        (gen.saturating_sub(1)) as f64 * 1000.0 / decode_ms.max(1e-9),
        if aborted { ", client gone" } else { "" }
    );
    eprintln!("[chat] ids {out:?}");
    if aborted {
        "200 OK (client gone)"
    } else {
        "200 OK (text/event-stream)"
    }
}

// ---------------------------------------------------------------- the socket

/// one response, HTTP/1.1 with Content-Length; the connection closes after it
fn respond(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// - reads one line into `buf`, at most `cap` bytes from `r`
/// - the cap is what keeps an endless line from growing the buffer
/// - returns the bytes taken; `buf` ends in a newline on a complete line
fn read_line_capped<R: BufRead>(r: &mut R, cap: usize, buf: &mut Vec<u8>) -> std::io::Result<usize> {
    r.by_ref().take(cap as u64).read_until(b'\n', buf)
}

/// - request line, headers and body of one connection, over any `BufRead`
/// - the read is bounded: never more than `MAX_HEAD_BYTES` are buffered
/// - a head that spends the cap without a newline gives `Head::HeadTooLarge`
/// - `Content-Length` must be one well formed value, else `Head::Bad`
/// - `Transfer-Encoding: chunked` gives `Head::Chunked`
/// - the body is drained after the blank line
/// - draining first is what lets a bodied POST see the 404 instead of a reset
fn read_head_from<R: BufRead>(r: &mut R) -> std::io::Result<Head> {
    let mut used = 0usize;

    let mut first = Vec::new();
    let n = read_line_capped(r, MAX_HEAD_BYTES, &mut first)?;
    if n == 0 {
        return Ok(Head::Empty);
    }
    used += n;
    if !first.ends_with(b"\n") && used >= MAX_HEAD_BYTES {
        return Ok(Head::HeadTooLarge);
    }
    let first = String::from_utf8_lossy(&first).into_owned();

    let mut content_length: Option<usize> = None;
    let mut bad_length = false;
    let mut chunked = false;
    loop {
        let budget = MAX_HEAD_BYTES - used;
        if budget == 0 {
            return Ok(Head::HeadTooLarge);
        }
        let mut raw = Vec::new();
        let n = read_line_capped(r, budget, &mut raw)?;
        if n == 0 {
            break;
        }
        used += n;
        if !raw.ends_with(b"\n") && used >= MAX_HEAD_BYTES {
            return Ok(Head::HeadTooLarge);
        }
        let line = String::from_utf8_lossy(&raw).into_owned();
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = header_value(&line, "content-length") {
            match (content_length, v.parse::<usize>()) {
                (None, Ok(len)) => content_length = Some(len),
                // a second Content-Length, or one that does not parse, is a 400
                _ => bad_length = true,
            }
        }
        if let Some(v) = header_value(&line, "transfer-encoding") {
            if v.split(',').any(|c| c.trim().eq_ignore_ascii_case("chunked")) {
                chunked = true;
            }
        }
    }

    // chunked first: there is no declared length to trust or to drain
    if chunked {
        return Ok(Head::Chunked);
    }
    if bad_length {
        return Ok(Head::Bad);
    }

    let content_length = content_length.unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return Ok(Head::BodyTooLarge(content_length));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        r.read_exact(&mut body)?;
    }

    match parse_request_line(&first) {
        Some((method, target)) => Ok(Head::Req { method, target, body }),
        None => Ok(Head::Bad),
    }
}

/// - `read_head_from` over one connection, buffered
fn read_head(stream: &TcpStream) -> std::io::Result<Head> {
    let mut r = BufReader::new(stream);
    read_head_from(&mut r)
}

/// what a connection needs beyond the socket: the engine, its container, the static props
struct Srv<'a> {
    /// the one loaded engine of this process; every chat request resets it first
    eng: &'a mut Engine,
    /// the container handle `prefill` fills PLE rows from
    cnq: &'a mut Cnq,
    /// `/props` `model_path`
    model_path: &'a str,
    /// `/props` `n_ctx`, read back from the loaded states
    n_ctx: usize,
    /// `/props` `prompt_chunk`, pinned for the process
    prompt_chunk: usize,
    /// per process request counter, the tail of every chunk `id`
    seq: u64,
}

fn serve_one(stream: &mut TcpStream, srv: &mut Srv) {
    let t = Duration::from_secs(IO_TIMEOUT_SECS);
    if let Err(e) = stream.set_read_timeout(Some(t)) {
        eprintln!("[serve] no read timeout on this connection, closing: {e}");
        return;
    }
    if let Err(e) = stream.set_write_timeout(Some(t)) {
        eprintln!("[serve] no write timeout on this connection, closing: {e}");
        return;
    }

    let head = match read_head(stream) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[serve] read failed or timed out after {IO_TIMEOUT_SECS}s, closing: {e}");
            return;
        }
    };

    let (label, status, doc) = match head {
        Head::Empty => return,
        Head::Bad => (
            "<no HTTP request line>".to_string(),
            "400 Bad Request",
            error_json("bad request"),
        ),
        Head::HeadTooLarge => (
            format!("<head over {MAX_HEAD_BYTES} bytes>"),
            "431 Request Header Fields Too Large",
            error_json("request header fields too large"),
        ),
        Head::BodyTooLarge(n) => (
            format!("<Content-Length {n} over {MAX_BODY_BYTES} bytes>"),
            "413 Payload Too Large",
            error_json("payload too large"),
        ),
        Head::Chunked => (
            "<Transfer-Encoding: chunked>".to_string(),
            "501 Not Implemented",
            error_json("chunked transfer encoding not supported"),
        ),
        Head::Req { method, target, body } => {
            let path = route_path(&target).to_string();
            let label = format!("{method} {target} (body {} bytes)", body.len());
            match route(&method, &path) {
                // the chat route writes its own response: SSE, or a JSON error
                Route::Chat => {
                    let status = chat_route(stream, srv, &body);
                    eprintln!("[serve] {label} -> {status}");
                    let _ = stream.shutdown(Shutdown::Write);
                    return;
                }
                Route::Health => (label, "200 OK", serde_json::json!({ "status": "ok" })),
                Route::Props => (
                    label,
                    "200 OK",
                    props_json(srv.model_path, srv.n_ctx, srv.prompt_chunk),
                ),
                Route::NotFound => (label, "404 Not Found", not_found_json(&path)),
            }
        }
    };

    let text = doc.to_string();
    eprintln!("[serve] {label} -> {status}");
    if let Err(e) = respond(stream, status, &text) {
        eprintln!("[serve] response write failed: {e}");
    }
    // half close, so the client reads EOF instead of a reset
    let _ = stream.shutdown(Shutdown::Write);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // A3: the tokenize arm returns HERE, before the CUDA context and before Engine::load
    // takes engine/.engine.lock; it never starts a Python process
    if args.get(1).map(|s| s.as_str()) == Some("tokenize") {
        std::process::exit(tokenize_main(&args[2..]));
    }
    let port = match parse_port(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[serve] {e}");
            std::process::exit(2);
        }
    };

    // #25 A3: warm up the tokenizer BEFORE the CUDA context and before Engine::load.
    // A missing or broken tokenizer must fail in a second, not after the engine is pinned.
    let (tok_path, tok_cfg) = crow_nest_engine::tokenizer::default_paths();
    match crow_nest_engine::tokenizer::global() {
        Ok(tk) => {
            let (tp, cp) = tk.paths();
            eprintln!("[serve] tokenizer {tp}");
            eprintln!("[serve] chat template {cp}");
        }
        Err(e) => {
            eprintln!("[serve] {e}");
            eprintln!("[serve] tokenizer {tok_path}");
            eprintln!("[serve] chat template {tok_cfg}");
            std::process::exit(3);
        }
    }

    // #26 review: the gated configuration is CROW_GRAPH=1 and CROW_MMA=1 (every A2..A4 gate and
    // the engine's ten-task gates ran with both on). gen.rs reads each one ONCE through a
    // OnceLock (`mma_on` gen.rs:1075, `graph_on` gen.rs:1086, `dense_mma_on` gen.rs:1156), so
    // setting them here, before cuda::Ctx::init and before any kernel call, is the whole switch.
    // Only an UNSET variable is set, so an explicit value still overrides and diagnostics stay
    // possible. serve is single threaded at this point: no other thread can read the environment.
    for key in ["CROW_GRAPH", "CROW_MMA"] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, "1");
        }
        eprintln!("[serve] {key}={}", std::env::var(key).unwrap_or_default());
    }

    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| DEFAULT_CNQ.into());
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or_else(|_| DEFAULT_HOTSETS.into());
    let mut cnq = Cnq::open(&cnq_path);

    // unsafe: creates the CUDA context; it must outlive every device allocation
    let _ctx = unsafe { crow_nest_engine::cuda::Ctx::init() };

    let mut cfg = Config::default();
    cfg.context = CONTEXT_FLOOR;
    // M1: chunk pinned for the process, no per prompt policy
    cfg.prompt_chunk = SERVE_CHUNK;
    apply_adapt_policy(&mut cfg);
    // #26 review: the "[policy] ... every N ..." line above comes from apply_adapt_policy, but
    // serve calls neither trickle_tick nor adapt_tick (decode.rs:216-228 is the only caller)
    eprintln!(
        "[serve] hot-set adaptation is not ticked by serve in this build (greedy identity first; A6/A9 decide)"
    );

    // unsafe: pins device and host memory; takes engine/.engine.lock, a second serve dies here
    let (eng, _rep) = unsafe {
        Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| eprintln!("[load] {m}"))
    };
    // `Cnq::open` handed the container to the loader; the engine keeps a raw
    // pointer to it (`Engine::cnq`), and prefill needs it back as `&mut`
    let n_ctx = eng.st.context;
    let prompt_chunk = eng.cfg.prompt_chunk;

    eprintln!("[serve] container {cnq_path}");
    eprintln!("[serve] hotsets {sidecar}");
    eprintln!("[serve] n_ctx {n_ctx}");
    eprintln!("[serve] prompt_chunk {prompt_chunk}");

    let addr = format!("127.0.0.1:{port}");
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[serve] cannot bind {addr}: {e}");
            std::process::exit(3);
        }
    };
    eprintln!("[serve] listening on http://{addr} (blocking, one request at a time)");

    let mut eng = eng;
    let mut srv = Srv {
        eng: &mut eng,
        cnq: &mut cnq,
        model_path: &cnq_path,
        n_ctx,
        prompt_chunk,
        seq: 0,
    };
    for conn in listener.incoming() {
        match conn {
            Ok(mut s) => serve_one(&mut s, &mut srv),
            Err(e) => eprintln!("[serve] accept failed: {e}"),
        }
    }
    drop(srv);
    drop(eng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn request_line_is_method_and_target() {
        assert_eq!(
            parse_request_line("GET /props HTTP/1.1\r\n"),
            Some(("GET".into(), "/props".into()))
        );
        assert_eq!(
            parse_request_line("POST /v1/chat/completions HTTP/1.1"),
            Some(("POST".into(), "/v1/chat/completions".into()))
        );
        assert_eq!(parse_request_line(""), None);
        assert_eq!(parse_request_line("GET /props"), None);
        assert_eq!(parse_request_line("GARBAGE\r\n"), None);
        // a proxy style absolute target is not a route this server answers
        assert_eq!(parse_request_line("GET http://x/props HTTP/1.1"), None);
    }

    #[test]
    fn route_drops_query_and_trailing_slash() {
        assert_eq!(route_path("/props"), "/props");
        assert_eq!(route_path("/props?x=1"), "/props");
        assert_eq!(route_path("/health/"), "/health");
        assert_eq!(route_path("/health#frag"), "/health");
        assert_eq!(route_path("/"), "/");
    }

    #[test]
    fn dispatch_is_method_and_path() {
        // the request line goes through the same two helpers serve_one uses
        let d = |line: &str| {
            let (m, t) = parse_request_line(line).expect("request line parses");
            route(&m, route_path(&t))
        };
        assert_eq!(d("GET /health HTTP/1.1\r\n"), Route::Health);
        assert_eq!(d("GET /health/ HTTP/1.1\r\n"), Route::Health);
        assert_eq!(d("GET /props HTTP/1.1\r\n"), Route::Props);
        assert_eq!(d("GET /props?x=1 HTTP/1.1\r\n"), Route::Props);
        // unknown path
        assert_eq!(d("GET /nope HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("GET /v1/models HTTP/1.1\r\n"), Route::NotFound);
        // non GET method, on a known path and on a future one
        assert_eq!(d("POST /health HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("HEAD /props HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("POST /v1/chat/completions HTTP/1.1\r\n"), Route::Chat);
        assert_eq!(d("POST /v1/chat/completions?x=1 HTTP/1.1\r\n"), Route::Chat);
        // the chat path answers POST only
        assert_eq!(d("GET /v1/chat/completions HTTP/1.1\r\n"), Route::NotFound);
    }

    #[test]
    fn content_length_header_is_case_insensitive() {
        assert_eq!(header_value("Content-Length: 7\r\n", "content-length"), Some("7"));
        assert_eq!(header_value("content-length:7", "content-length"), Some("7"));
        assert_eq!(header_value("CONTENT-LENGTH:  42  \r\n", "content-length"), Some("42"));
        assert_eq!(header_value("Host: 127.0.0.1\r\n", "content-length"), None);
        assert_eq!(header_value("no colon here", "content-length"), None);
    }

    #[test]
    fn props_carries_the_fields_crow_reads() {
        let doc = props_json("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", 200_000, 2048);
        // server_model_path (crow_core.py:1408)
        assert_eq!(doc["model_path"], "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq");
        // fetch_model_name (crow_core.py:14884)
        assert_eq!(doc["model"], "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert!(doc["model"].as_str().unwrap().contains("Flash-Next"));
        // fetch_n_ctx (crow_core.py:14837): settings first, bare key as fallback
        assert_eq!(doc["default_generation_settings"]["n_ctx"], 200_000);
        assert_eq!(doc["n_ctx"], 200_000);
        assert!(doc["n_ctx"].as_u64().unwrap() >= 200_000);
        // refuse_images (crow_core.py:1429): false = BLIND_SERVER_HINT, not an error
        assert_eq!(doc["modalities"]["vision"], serde_json::Value::Bool(false));
        assert_eq!(doc["prompt_chunk"], 2048);
    }

    #[test]
    fn model_name_is_the_container_stem() {
        assert_eq!(model_name("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq"), "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert_eq!(model_name("C:\\x\\converter\\Qwen3.8-Flash-Next-CNQ4.5-M.cnq"), "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert_eq!(model_name("plain"), "plain");
    }

    #[test]
    fn port_flag_beats_the_default() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_port(&v(&["serve"])), Ok(DEFAULT_PORT));
        assert_eq!(parse_port(&v(&["serve", "--port", "8099"])), Ok(8099));
        assert_eq!(parse_port(&v(&["serve", "--port=1234"])), Ok(1234));
        assert!(parse_port(&v(&["serve", "--port"])).is_err());
        assert!(parse_port(&v(&["serve", "--port", "no"])).is_err());
        assert!(parse_port(&v(&["serve", "-p", "1"])).is_err());
    }

    #[test]
    fn tokenize_arguments_parse_into_one_job() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_tokenize(&v(&["--chat", "--file", "p.json", "--out", "i.json"])),
            Ok(Tok::File { chat: true, file: "p.json".into(), out: "i.json".into() })
        );
        assert_eq!(
            parse_tokenize(&v(&["--raw", "--text", "hi"])),
            Ok(Tok::Text { chat: false, text: "hi".into() })
        );
        assert_eq!(
            parse_tokenize(&v(&["--chat", "--text", "hi"])),
            Ok(Tok::Text { chat: true, text: "hi".into() })
        );
        // no mode, both modes, missing --out, both inputs, unknown flag, missing value
        assert!(parse_tokenize(&v(&["--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--raw", "--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--file", "p.json"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--file", "p.json", "--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--text"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--nope", "x"])).is_err());
    }

    #[test]
    fn both_prompt_file_shapes_give_the_same_pairs() {
        // docs/ten-task-prompts-crowlab.json is {id: text}; with serde_json's preserve_order
        // the object comes back in FILE order, not sorted by key (t2 stands first here)
        let obj = serde_json::json!({ "t2-write": "b", "t1-read": "a" });
        assert_eq!(
            prompts_from_json(&obj).unwrap(),
            vec![("t2-write".to_string(), "b".to_string()), ("t1-read".to_string(), "a".to_string())]
        );
        // the same shape parsed from bytes, so this pins the parser, not just the json! macro
        let parsed: serde_json::Value =
            serde_json::from_str(r#"{"t2-write": "b", "t1-read": "a"}"#).unwrap();
        assert_eq!(
            prompts_from_json(&parsed).unwrap(),
            vec![("t2-write".to_string(), "b".to_string()), ("t1-read".to_string(), "a".to_string())]
        );
        // decode_out/ten-tasks.json is the parity harness array
        let arr = serde_json::json!([
            { "id": "t1-read", "text": "a", "max_tokens": 1024 },
            { "id": "t2-write", "text": "b", "max_tokens": 1280 }
        ]);
        assert_eq!(
            prompts_from_json(&arr).unwrap(),
            vec![("t1-read".to_string(), "a".to_string()), ("t2-write".to_string(), "b".to_string())]
        );
        assert!(prompts_from_json(&serde_json::json!([{ "text": "a" }])).is_err());
        assert!(prompts_from_json(&serde_json::json!({ "t": 1 })).is_err());
        assert!(prompts_from_json(&serde_json::json!("nope")).is_err());
    }

    #[test]
    fn unknown_route_is_a_404_document() {
        let doc = not_found_json("/v1/models");
        assert_eq!(doc["error"]["code"], 404);
        assert!(doc["error"]["message"].as_str().unwrap().contains("/v1/models"));
    }

    #[test]
    fn flat_error_documents_name_the_reason() {
        assert_eq!(error_json("bad request")["error"], "bad request");
        assert_eq!(error_json("payload too large")["error"], "payload too large");
    }
    // ------------------------------------------------ chat completions (#26 A4)

    #[test]
    fn chat_body_parses_messages_max_tokens_and_stream() {
        // the body Crow sends (crow_core.py:4672-4700), trimmed to what A4 reads
        let raw = br#"{"model":"crow-nest",
            "messages":[{"role":"user","content":"Say the word ready."}],
            "temperature":0,"top_p":0.95,"min_p":0.01,"stream":true,
            "stream_options":{"include_usage":true},"timings_per_token":true,
            "max_tokens":16}"#;
        let r = parse_chat(raw).expect("the body parses");
        assert_eq!(r.model, "crow-nest");
        assert_eq!(r.max_tokens, 16);
        assert!(r.stream);
        assert!(!r.enable_thinking);
        assert_eq!(r.messages[0]["content"], "Say the word ready.");

        // defaults: no stream, no max_tokens, no model
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(r.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!r.stream);
        assert_eq!(r.model, "crow-nest");

        // the digest path sends chat_template_kwargs (crow_core.py:2969)
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        assert!(r.enable_thinking);

        // max_tokens is capped, never trusted
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":999999}"#)
            .unwrap();
        assert_eq!(r.max_tokens, MAX_MAX_TOKENS);

        // tools and unknown sampler fields are accepted and ignored
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "tools":[{"type":"function","function":{"name":"read"}}],"top_k":20}"#,
        )
        .unwrap();
        assert_eq!(r.max_tokens, DEFAULT_MAX_TOKENS);

        // rejections
        assert!(parse_chat(b"not json").is_err());
        assert!(parse_chat(b"[]").is_err());
        assert!(parse_chat(br#"{"messages":[]}"#).is_err());
        assert!(parse_chat(br#"{"model":"x"}"#).is_err());
        assert!(parse_chat(br#"{"messages":"hi"}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"content":"hi"}]}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"stream":"yes"}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}"#).is_err());
    }

    #[test]
    fn the_budget_is_clamped_to_the_free_context_and_413_only_without_one() {
        // the normal case: the whole budget fits, nothing is touched
        assert_eq!(clamped_max_tokens(18, 64, 200_000), Some(64));
        // the prompt eats most of the context: the budget shrinks to what is left
        assert_eq!(clamped_max_tokens(199_990, 64, 200_000), Some(10));
        assert_eq!(clamped_max_tokens(1, 1024, 2), Some(1));
        // the 32768 cap still binds when the free context is larger
        assert_eq!(clamped_max_tokens(18, 999_999, 200_000), Some(MAX_MAX_TOKENS));
        assert_eq!(clamped_max_tokens(18, MAX_MAX_TOKENS, 200_000), Some(MAX_MAX_TOKENS));
        // exactly one token of room, and exactly none
        assert_eq!(clamped_max_tokens(199_999, 64, 200_000), Some(1));
        assert_eq!(clamped_max_tokens(200_000, 64, 200_000), None);
        assert_eq!(clamped_max_tokens(200_001, 1, 200_000), None);
        // the clamp is never 0 when it returns Some: a served request always has a budget
        for prompt in [0usize, 1, 17, 199_999] {
            let n = clamped_max_tokens(prompt, DEFAULT_MAX_TOKENS, 200_000).unwrap();
            assert!(n >= 1, "prompt {prompt} got budget {n}");
            assert!(prompt + n <= 200_000);
        }
    }

    #[test]
    fn a_chunk_carries_exactly_what_crow_reads() {
        // crow_core.py:4846 finish_reason, :4877 delta.content
        let role = chunk_role("chatcmpl-1", 1_757_000_000, "crow-nest");
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["id"], "chatcmpl-1");
        assert_eq!(role["created"], 1_757_000_000u64);
        assert_eq!(role["model"], "crow-nest");
        assert_eq!(role["choices"][0]["index"], 0);
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], serde_json::Value::Null);
        assert!(role["choices"][0]["delta"].get("content").is_none());

        let c = chunk_content("chatcmpl-1", 1, "crow-nest", " ready");
        assert_eq!(c["choices"][0]["delta"]["content"], " ready");
        assert_eq!(c["choices"][0]["finish_reason"], serde_json::Value::Null);

        // the last chunk: empty delta, a finish reason, and only ONE of them exists
        let f = chunk_finish("chatcmpl-1", 1, "crow-nest", "stop");
        assert_eq!(f["choices"][0]["finish_reason"], "stop");
        assert_eq!(f["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(chunk_finish("i", 1, "m", "length")["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn an_sse_frame_is_one_data_line_and_a_blank_line() {
        let f = sse_frame(&chunk_content("id", 7, "m", "hi"));
        assert!(f.starts_with("data: {"));
        assert!(f.ends_with("\n\n"));
        // exactly one event: one `data:` line, then the terminator
        assert_eq!(f.matches("data: ").count(), 1);
        assert_eq!(f.trim_end_matches('\n').matches('\n').count(), 0);
        // a newline inside the content is escaped by the JSON writer, never raw
        let f = sse_frame(&chunk_content("id", 7, "m", "a\nb"));
        assert!(f.contains(r#""content":"a\nb""#));
        assert_eq!(f.trim_end_matches('\n').matches('\n').count(), 0);
        // the closing line of every stream
        assert_eq!(SSE_DONE, "data: [DONE]\n\n");
    }

    #[test]
    fn an_incomplete_utf8_tail_is_held_back() {
        // "Gr" = 2 bytes, "u umlaut" = 2 bytes: a token split inside it decodes
        // to a replacement character until the second byte arrives
        assert_eq!(next_delta("Gr\u{FFFD}", 0), None);
        assert_eq!(next_delta("Gr\u{00FC}", 0), Some("Gr\u{00FC}"));
        // the next token continues the same string; only the new suffix goes out
        assert_eq!(next_delta("Gr\u{00FC}\u{00DF}e", 4), Some("\u{00DF}e"));
        // nothing new, nothing sent
        assert_eq!(next_delta("Gr\u{00FC}", 4), None);
        // never split a character, even if the caller asks for it
        assert_eq!(next_delta("Gr\u{00FC}", 3), None);
    }

    #[test]
    fn a_split_multibyte_token_never_reaches_a_chunk() {
        // the real tokenizer, no GPU: an emoji is one 4 byte character whose
        // byte level tokens can end mid sequence
        // tests run from engine/, the model lives at the repository root
        let t = "../models/Qwen3.8-Flash-Next-original/tokenizer.json";
        let tk = crow_nest_engine::tokenizer::ChatTokenizer::load(
            t,
            &t.replace("tokenizer.json", "tokenizer_config.json"),
        )
        .expect("tokenizer loads");
        let ids = tk.encode_raw("\u{1F985}\u{1F985}").expect("encode");
        assert!(ids.len() >= 2, "the emoji pair must be more than one token: {ids:?}");
        let mut emitted = 0usize;
        let mut sent = String::new();
        let mut held = 0usize;
        for n in 1..=ids.len() {
            let full = tk.decode(&ids[..n]).expect("decode");
            match next_delta(&full, emitted) {
                Some(d) => {
                    assert!(
                        !d.contains(char::REPLACEMENT_CHARACTER),
                        "chunk {d:?} carries U+FFFD"
                    );
                    sent.push_str(d);
                    emitted = full.len();
                }
                None => held += 1,
            }
        }
        // at least one token was held back, and the text still arrived whole
        assert!(held > 0, "no token was held back, the split was never exercised");
        assert_eq!(sent, "\u{1F985}\u{1F985}");
    }

    #[test]
    fn a_request_line_over_the_cap_stops_the_reader_at_the_cap() {
        let mut raw = b"GET /".to_vec();
        raw.resize(70_000, b'a');
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::HeadTooLarge);
        // bounded: the cap was never passed, the 70,000 byte line was never buffered
        assert!(c.position() <= MAX_HEAD_BYTES as u64);
        assert_eq!(c.position(), MAX_HEAD_BYTES as u64);
    }

    #[test]
    fn header_lines_over_the_cap_stop_the_reader_at_the_cap() {
        let mut raw = b"GET /health HTTP/1.1\r\n".to_vec();
        while raw.len() < MAX_HEAD_BYTES + 4096 {
            raw.extend_from_slice(b"X-Pad: 0123456789012345678901234567890123456789\r\n");
        }
        raw.extend_from_slice(b"\r\n");
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::HeadTooLarge);
        assert!(c.position() <= MAX_HEAD_BYTES as u64);
    }

    #[test]
    fn a_plain_get_parses_out_of_a_cursor() {
        let mut c = Cursor::new(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n".to_vec());
        assert_eq!(
            read_head_from(&mut c).unwrap(),
            Head::Req { method: "GET".into(), target: "/health".into(), body: Vec::new() }
        );
    }

    #[test]
    fn a_declared_body_is_read_whole() {
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello".to_vec());
        match read_head_from(&mut c).unwrap() {
            Head::Req { method, target, body } => {
                assert_eq!(method, "POST");
                assert_eq!(target, "/x");
                assert_eq!(body, b"hello".to_vec());
            }
            other => panic!("expected a request, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_stream_gets_no_response() {
        let mut c = Cursor::new(Vec::new());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Empty);
    }

    #[test]
    fn a_garbage_request_line_is_a_bad_request() {
        let mut c = Cursor::new(b"NOT HTTP AT ALL\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn a_malformed_content_length_is_a_bad_request() {
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: -1\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn a_repeated_content_length_is_a_bad_request() {
        let raw = b"POST /x HTTP/1.1\r\nContent-Length: 5\r\ncontent-length: 5\r\n\r\nhello".to_vec();
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn chunked_transfer_encoding_is_rejected() {
        let raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n".to_vec();
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Chunked);
        // case insensitive, and inside a coding list
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\ntransfer-encoding: gzip, Chunked\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Chunked);
        assert_eq!(
            error_json("chunked transfer encoding not supported")["error"],
            "chunked transfer encoding not supported"
        );
    }
}
