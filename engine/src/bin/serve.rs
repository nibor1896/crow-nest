//! #24 serve: the HTTP face of the engine (spec section 7).
//!
//! Endpoints (A2 skeleton, no chat endpoint yet):
//!
//! - `GET /health` answers `{"status":"ok"}`.
//! - `GET /props` answers `model_path`, `model`, `n_ctx`,
//!   `default_generation_settings.n_ctx`, `modalities.vision` (llama-server field
//!   names, so Crow's `check_endpoint` / `fetch_n_ctx` / `fetch_model_name` /
//!   `refuse_images` / `server_model_path` work unchanged).
//! - Anything else answers 404 with a JSON body.
//!
//! Operating point (M1 decisions, robin 2026-09-09):
//!
//! - Container default `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`; `CROW_CNQ` overrides.
//! - Hot sets default `decode_out/hotsets-M-longctx2100-n160.json`; `CROW_HOTSETS` overrides.
//! - Prompt chunk pinned at 2048 for the whole process (the per prompt policy of
//!   `geo::apply_chunk_policy` is NOT applied; this is `CROW_CHUNK=2048
//!   CROW_CHUNK_AUTO=0` set programmatically before `Engine::load`).
//! - Context `CONTEXT_FLOOR` (200,000), the operating point of `decode run`.
//! - `n_ctx` is read back from the loaded states (`Engine::st.context`), never a constant.
//! - Process wide defaults since d36353a need no env: `CROW_PF_ASYNC=2`, PF_TG 64,
//!   PLE 128 MB, attention kernel attn_sel_s8l.
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
//! Run:
//!
//! - `serve [--port <n>]` from the repository root (paths are repo relative).
//! - Default port 8099, bind address 127.0.0.1.

use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::gen::Engine;
use crow_nest_engine::geo::{apply_adapt_policy, Config, CONTEXT_FLOOR};
use std::io::{BufRead, BufReader, Write};
use std::net::{TcpListener, TcpStream};

const DEFAULT_PORT: u16 = 8099;
const DEFAULT_CNQ: &str = "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq";
const DEFAULT_HOTSETS: &str = "decode_out/hotsets-M-longctx2100-n160.json";
/// pinned for the process (M1): every request prefills at chunk 2048
const SERVE_CHUNK: usize = 2048;

// ---------------------------------------------------------------- pure parts

/// `--port <n>` / `--port=<n>` out of the argument vector (argv[0] included).
/// `Err` carries the message for the operator; missing flag = default port.
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

/// request line + headers; the headers are read to the blank line so the client
/// never sees a reset before it finished writing
fn read_head(stream: &TcpStream) -> std::io::Result<Option<(String, String)>> {
    let mut r = BufReader::new(stream);
    let mut first = String::new();
    if r.read_line(&mut first)? == 0 {
        return Ok(None);
    }
    loop {
        let mut h = String::new();
        if r.read_line(&mut h)? == 0 || h.trim().is_empty() {
            break;
        }
    }
    Ok(parse_request_line(&first))
}

fn serve_one(stream: &mut TcpStream, model_path: &str, n_ctx: usize, prompt_chunk: usize) {
    let head = match read_head(stream) {
        Ok(Some(h)) => h,
        Ok(None) => return,
        Err(e) => {
            eprintln!("[serve] request read failed: {e}");
            return;
        }
    };
    let (method, target) = head;
    let path = route_path(&target).to_string();
    let (status, body) = match (method.as_str(), path.as_str()) {
        ("GET", "/health") => ("200 OK", serde_json::json!({ "status": "ok" })),
        ("GET", "/props") => ("200 OK", props_json(model_path, n_ctx, prompt_chunk)),
        _ => ("404 Not Found", not_found_json(&path)),
    };
    let text = body.to_string();
    eprintln!("[serve] {method} {target} -> {status}");
    if let Err(e) = respond(stream, status, &text) {
        eprintln!("[serve] response write failed: {e}");
    }
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

    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        let mut cfg = Config::default();
        cfg.context = CONTEXT_FLOOR;
        // M1: chunk pinned for the process, no per prompt policy
        cfg.prompt_chunk = SERVE_CHUNK;
        apply_adapt_policy(&mut cfg);

        // Engine::load takes engine/.engine.lock; a second serve dies here
        let (eng, _rep) = Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| eprintln!("[load] {m}"));
        let n_ctx = eng.st.context;

        eprintln!("[serve] container {cnq_path}");
        eprintln!("[serve] hotsets {sidecar}");
        eprintln!("[serve] n_ctx {n_ctx}");
        eprintln!("[serve] prompt_chunk {}", eng.cfg.prompt_chunk);

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
                Ok(mut s) => serve_one(&mut s, &cnq_path, n_ctx, eng.cfg.prompt_chunk),
                Err(e) => eprintln!("[serve] accept failed: {e}"),
            }
        }
        drop(eng);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
}
