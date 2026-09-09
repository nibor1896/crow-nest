//! #24 serve: the HTTP face of the engine (spec section 7).
//!
//! Endpoints (A2 skeleton, no chat endpoint yet):
//!
//! - `GET /health` answers `{"status":"ok"}`.
//! - `GET /props` answers the operating point as a JSON document.
//! - `/props` field names mirror llama-server, so Crow's readers work unchanged.
//! - Crow readers served: `check_endpoint`, `fetch_n_ctx`, `fetch_model_name`.
//! - Crow readers served: `refuse_images`, `server_model_path`.
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

use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::gen::Engine;
use crow_nest_engine::geo::{apply_adapt_policy, Config, CONTEXT_FLOOR};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::Duration;

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

// ---------------------------------------------------------------- pure parts

/// the route a `(method, path)` pair dispatches to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `GET /health`
    Health,
    /// `GET /props`
    Props,
    /// everything else, a wrong method on a known path included
    NotFound,
}

/// the whole dispatch table of A2, pure so the test can drive it
fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/health") => Route::Health,
        ("GET", "/props") => Route::Props,
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
        "error": { "code": 404, "message": format!("no route {path} (serve answers GET /health, GET /props)") }
    })
}

/// flat error document for 400, 413, 431 and 501
fn error_json(msg: &str) -> serde_json::Value {
    serde_json::json!({ "error": msg })
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

fn serve_one(stream: &mut TcpStream, model_path: &str, n_ctx: usize, prompt_chunk: usize) {
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
            let (status, doc) = match route(&method, &path) {
                Route::Health => ("200 OK", serde_json::json!({ "status": "ok" })),
                Route::Props => ("200 OK", props_json(model_path, n_ctx, prompt_chunk)),
                Route::NotFound => ("404 Not Found", not_found_json(&path)),
            };
            (format!("{method} {target} (body {} bytes)", body.len()), status, doc)
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
    let port = match parse_port(&args) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("[serve] {e}");
            std::process::exit(2);
        }
    };

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

    // unsafe: pins device and host memory; takes engine/.engine.lock, a second serve dies here
    let (eng, _rep) = unsafe {
        Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| eprintln!("[load] {m}"))
    };
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

    for conn in listener.incoming() {
        match conn {
            Ok(mut s) => serve_one(&mut s, &cnq_path, n_ctx, prompt_chunk),
            Err(e) => eprintln!("[serve] accept failed: {e}"),
        }
    }
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
        assert_eq!(d("POST /v1/chat/completions HTTP/1.1\r\n"), Route::NotFound);
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
