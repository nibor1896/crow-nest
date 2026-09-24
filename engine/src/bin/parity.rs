//! #11 — standing parity harness (spec 5.2, methodology #159): A/B against
//! llama.cpp on this machine, same session, one variable, every number with
//! its operating point.
//!
//!   parity run   <prompts.json> [llama-url]
//!   parity phase <run_index> <crow|llama> <prompts.json> [llama-url] [outprefix]
//!
//! Execution mode (2026-09-03, measured constraint): the crow engine's pinned
//! cold tier (~44.6 GB host RAM, ~29 GB VRAM) cannot coexist with the
//! llama-server model (mmap page cache ~50 GB host, ~12 GB VRAM) on this 64 GB
//! / 32 GB machine — the loader would clamp or refuse. The ten-task gate is
//! therefore run ARM-PHASED: each arm drives the full rotated task order in
//! its own phase with exclusive resources. The #159 pairing discipline is
//! kept exactly: both arms see the IDENTICAL rotated order per run, the arm
//! that moves first swaps per run, one discarded cold prefill per start.
//! Interleaved execution stays available via `run` for a machine that fits
//! both engines.
//!
//! prompts.json: [{"id": "...", "text": "...", "max_tokens": 64}, ...]
//! Text is tokenized through tools/tokenize_ids.py (oracle venv) for the
//! crow-nest side; llama.cpp receives the raw text (same tokenizer family).
//! Reports land in decode_out/<prefix>-run<index>-<arm>.json; crow answers
//! are recorded as greedy token ids (detokenized in batch afterwards via
//! tools/detokenize_ids.py).
//!
//! llama-server baseline config (crow-lab levers-159):
//!   llama-server.exe -m <UD-Q2_K_XL.gguf> --port 8083 -c 200000 -b 4096
//!     -ub 4096 -ctk q8_0 -ctv q8_0 -ncmoe 40 --fit off --load-mode none
//!     -np 1 --jinja

use crow_nest_engine::geo::{DEFAULT_CNQ, DEFAULT_HOTSETS};
use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::sync::Mutex;
use std::time::Instant;

#[derive(Clone)]
struct Prompt {
    id: String,
    text: String,
    max_tokens: usize,
}

// ---- the two oracle children (#65) ---------------------------------------------------
//
// Both python children of this harness are spawned through `oracle_child`, which is the
// guard of issue #65: a tokenize child that dies ONCE costs the single task a second
// attempt, not the ten-task phase a re-run (~15 min). The diagnosis of a dead child is
// `child_verdict`, the pure half, and it leads with the EXIT CODE because #65's three
// failures had nothing else — see `child_verdict`'s own comment for why.

/// the oracle venv's interpreter — the ONE place the per-OS path is written
/// (`.venv-oracle` is gitignored; the unix arm arrived with the Linux port, #15).
#[cfg(windows)]
const ORACLE_PY: &str = ".venv-oracle/Scripts/python.exe";
#[cfg(unix)]
const ORACLE_PY: &str = ".venv-oracle/bin/python";

/// One retry per entry, and the pause before it. Three attempts, 7 s of waiting in the
/// worst case against the ~15 min a lost phase costs. The pauses are seconds and not
/// milliseconds on purpose: the leading suspicion of #65 is host-resource pressure at
/// the moment of the spawn (the harness re-maps the container and re-pins ~44.6 GB per
/// task, and engine/README's Windows rule is to wait for 50.5 GiB of free RAM before a
/// load), and a resource that is being reclaimed needs wall-clock time, not a tight loop.
const ORACLE_BACKOFF_MS: [u64; 2] = [2_000, 5_000];

/// #65 (2026-09-18): what the oracle children of THIS phase had to be asked twice for.
/// One entry per FAILED attempt, so a retry lands in the run record instead of only in
/// the console; `label` is the task the harness was on when the child was spawned, which
/// is the one thing the three #65 reports had to reconstruct from the task order.
struct OracleLog {
    label: String,
    attempts: Vec<serde_json::Value>,
}

static ORACLE_LOG: Mutex<OracleLog> =
    Mutex::new(OracleLog { label: String::new(), attempts: Vec::new() });

/// names the task every later oracle child belongs to (`"warm-up"`, `"t5-agent (position 4)"`).
fn oracle_label(label: &str) {
    ORACLE_LOG.lock().unwrap().label = label.to_string();
}

/// takes the failed attempts recorded since the last drain — the caller writes them into
/// the record it is building, and an empty vector adds no key, so a clean phase produces
/// exactly the record shape it produced before #65.
fn drain_oracle_attempts() -> Vec<serde_json::Value> {
    std::mem::take(&mut ORACLE_LOG.lock().unwrap().attempts)
}

/// The exit codes whose NAME is the whole message, because a process that dies this way
/// never reaches its own error path and writes nothing to stderr. This is the list #65
/// needed and did not have: the harness printed the child's stderr and threw the status
/// away, and the stderr was empty in all three occurrences.
fn exit_code_name(code: i32) -> Option<&'static str> {
    Some(match code as u32 {
        // Windows NTSTATUS, reported as the process exit code
        0xC000_0005 => "STATUS_ACCESS_VIOLATION - a native crash (an extension module), no Python traceback",
        0xC000_0017 => "STATUS_NO_MEMORY",
        0xC000_0142 => "STATUS_DLL_INIT_FAILED - the loader could not initialise a DLL of the venv, before any Python code ran",
        0xC000_012D => "STATUS_COMMITMENT_LIMIT - the system could not commit the child's memory",
        0xC000_00FD => "STATUS_STACK_OVERFLOW",
        0xC000_013A => "STATUS_CONTROL_C_EXIT - the child was interrupted",
        0xC000_00FC => "STATUS_FILE_CORRUPT_ERROR",
        // CPython's own silent one
        120 => "CPython could not flush its std streams at exit",
        _ => return None,
    })
}

/// The pure half of the guard: what ONE finished oracle child means, as a `Result` whose
/// error is the whole diagnosis in one string.
///
/// It exists because of the shape of #65. The call site was
/// `assert!(out.status.success(), "tokenize stderr: {}", ..)`: it printed the child's
/// stderr and dropped the exit status, the stdout length and the size of what had been
/// sent. All three #65 failures had an EMPTY stderr — which is not a Python error at all,
/// since a Python error writes its traceback to that pipe. An empty stderr with a
/// non-zero status is a process that died without running Python's error path, and on
/// Windows the NAME of that death is the exit code (see `exit_code_name`). So the exit
/// code comes first, `signal` covers the unix arm, and a child that exits 0 with an
/// EMPTY stdout is a failure too: it said nothing, and nothing is not a token stream.
fn child_verdict(
    code: Option<i32>,
    signal: Option<i32>,
    stderr: &str,
    stdout_len: usize,
) -> Result<(), String> {
    let status = match (code, signal) {
        (Some(0), _) if stdout_len > 0 => return Ok(()),
        (Some(0), _) => "exit code 0 but 0 B on stdout".to_string(),
        (Some(c), _) => match exit_code_name(c) {
            Some(name) => format!("exit code {c} (0x{:08x}) — {name}", c as u32),
            None => format!("exit code {c} (0x{:08x})", c as u32),
        },
        (None, Some(s)) => format!("killed by signal {s}"),
        (None, None) => "no exit code and no signal".to_string(),
    };
    let said = if stderr.trim().is_empty() {
        // the #65 signature, spelled out so the next report does not have to guess
        "stderr EMPTY — the child never reached Python's own error path, so its exit \
         code above is the whole message (a loader failure, a native crash or an \
         outside kill; a Python exception would have written a traceback here)"
            .to_string()
    } else {
        let s = stderr.trim();
        format!("stderr ({} B): {}", stderr.len(), &s[..s.len().min(4000)])
    };
    Err(format!("{status}; {stdout_len} B on stdout; {said}"))
}

/// Spawn one oracle child, hand it `stdin_bytes`, and give back its stdout — retrying a
/// failed attempt up to `backoff_ms.len()` times. `what` names the child in the log and
/// in the record. Fail-closed: when every attempt fails this panics with the whole table
/// of attempts, so the phase records nothing for the task it was on.
///
/// `program` and `args` are arguments and not constants so the retry itself is unit
/// testable against a stub child, with no oracle venv, no llama-server and no GPU.
fn oracle_child(
    program: &str,
    args: &[&str],
    stdin_bytes: &[u8],
    what: &str,
    backoff_ms: &[u64],
) -> Vec<u8> {
    let tries = backoff_ms.len() + 1;
    let mut failures: Vec<String> = Vec::new();
    for attempt in 1..=tries {
        let t0 = Instant::now();
        match oracle_attempt(program, args, stdin_bytes) {
            Ok(stdout) => {
                if attempt > 1 {
                    eprintln!(
                        "[oracle] {what}: attempt {attempt} of {tries} succeeded ({} B on stdout) — the task goes on, the failed attempts are in the record",
                        stdout.len()
                    );
                }
                return stdout;
            }
            Err(why) => {
                let ms = t0.elapsed().as_secs_f64() * 1000.0;
                // loud and immediate: the process may not live to write the record
                eprintln!("[oracle] {what}: attempt {attempt} of {tries} FAILED after {ms:.0} ms — {why}");
                {
                    let mut log = ORACLE_LOG.lock().unwrap();
                    let at = log.label.clone();
                    log.attempts.push(serde_json::json!({
                        "at": at, "child": what, "attempt": attempt, "of": tries,
                        "stdin_bytes": stdin_bytes.len(), "ms": ms, "diagnosis": why,
                    }));
                }
                failures.push(format!("attempt {attempt}: {why}"));
                if attempt < tries {
                    std::thread::sleep(std::time::Duration::from_millis(backoff_ms[attempt - 1]));
                }
            }
        }
    }
    panic!(
        "{what}: all {tries} attempts failed — fail-closed, nothing is recorded for this task\n  {}",
        failures.join("\n  ")
    );
}

/// ONE attempt. Every pipe hazard this call site could have is closed here by
/// construction, which is how #65 excluded them as causes:
///
/// - the payload goes through STDIN, never through the command line (`CreateProcess`
///   caps at 32,767 chars and two of the ten frozen prompts are longer than that —
///   t1-read at 60,290 and t1b-read-lang at 53,889, measured 2026-09-18);
/// - stdin is CLOSED by this scope, so the child's read-to-EOF always ends;
/// - `wait_with_output` drains stdout AND stderr while it waits, so neither pipe can
///   fill and stall the child (8 KiB on Windows, 64 KiB on Linux);
/// - a failed write is NOT unwrapped — the child knows why its read end went away, so
///   the error is kept and the child is still waited for and still asked for its stderr;
/// - and a short write can never be accepted, even behind a zero exit.
fn oracle_attempt(program: &str, args: &[&str], stdin_bytes: &[u8]) -> Result<Vec<u8>, String> {
    let mut child = Command::new(program)
        .args(args)
        // #34: the harness sets the oracle transport itself, not the shell. Python 3.13
        // on Windows decodes STDIN as cp1252 without it; measured 2026-09-09 on
        // t4-prose: 9,522 ids bare against 9,398 ids with UTF-8.
        .env("PYTHONIOENCODING", "utf-8")
        .env("PYTHONUTF8", "1")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| format!("{program} did not start: {e}"))?;
    let write_err = {
        let mut sin = child.stdin.take().expect("stdin was piped");
        sin.write_all(stdin_bytes).and_then(|()| sin.flush()).err()
    };
    let out = child.wait_with_output().map_err(|e| format!("wait failed: {e}"))?;
    #[cfg(unix)]
    let signal = std::os::unix::process::ExitStatusExt::signal(&out.status);
    #[cfg(not(unix))]
    let signal: Option<i32> = None;
    let stderr = String::from_utf8_lossy(&out.stderr);
    if let Err(diagnosis) = child_verdict(out.status.code(), signal, &stderr, out.stdout.len()) {
        return Err(match write_err {
            Some(e) => format!("{diagnosis}; the parent's write to its stdin failed too: {e}"),
            None => diagnosis,
        });
    }
    if let Some(e) = write_err {
        return Err(format!(
            "the child exited 0 with {} B on stdout, but the parent could not write all {} B of its stdin: {e}",
            out.stdout.len(),
            stdin_bytes.len()
        ));
    }
    Ok(out.stdout)
}

fn tokenize(text: &str) -> Vec<i64> {
    // CWD must be the crow-nest repo root (all harness paths are repo-relative)
    // long prompts go through STDIN — the Windows command line caps at 32,767 chars
    // --chat: the crow arm receives the SAME token stream as the llama.cpp arm
    // (chat template, thinking disabled) — parity-gate fairness, fable gate 2026-09-03
    let stdout = oracle_child(
        ORACLE_PY,
        &["tools/tokenize_ids.py", "--chat"],
        text.as_bytes(),
        "tokenize_ids.py --chat",
        &ORACLE_BACKOFF_MS,
    );
    let v: serde_json::Value = serde_json::from_slice(&stdout).unwrap_or_else(|e| {
        panic!(
            "tokenize_ids.py printed {} B that are not JSON ({e}): {:?}",
            stdout.len(),
            String::from_utf8_lossy(&stdout[..stdout.len().min(200)])
        )
    });
    let ids: Vec<i64> = v
        .as_array()
        .unwrap_or_else(|| panic!("tokenize_ids.py printed {v}, which is not an array of ids"))
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect();
    // fail-closed on the one shape a ZERO exit could still hide: a child that read a
    // truncated stdin and printed an empty id list would otherwise be recorded as a
    // measurement of a prompt that was never sent.
    assert!(!ids.is_empty(), "tokenize_ids.py returned 0 ids for {} B of prompt text", text.len());
    ids
}

fn detokenize_all(map: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    let inp = serde_json::json!(map).to_string();
    let tmp = "decode_out/_detok_in.json";
    std::fs::write(tmp, inp).unwrap();
    // the ids reach this child through a FILE, so it needs nothing on its stdin; it gets
    // the same bounded retry, because it runs after the last task has already been paid
    // for and its failure would cost the phase its answer TEXT.
    let stdout = oracle_child(
        ORACLE_PY,
        &["tools/detokenize_ids.py", tmp],
        &[],
        "detokenize_ids.py",
        &ORACLE_BACKOFF_MS,
    );
    serde_json::from_slice(&stdout).unwrap_or_else(|e| {
        panic!(
            "detokenize_ids.py printed {} B that are not JSON ({e})",
            stdout.len()
        )
    })
}

/// Minimal HTTP/1.1 POST with real headers (the server is local; Connection:
/// close makes read-to-EOF the whole response).
fn http_post_json(host: &str, path: &str, body: &str) -> String {
    let mut s = std::net::TcpStream::connect(host).expect("llama-server not reachable");
    s.set_read_timeout(Some(std::time::Duration::from_secs(1800))).unwrap();
    s.set_write_timeout(Some(std::time::Duration::from_secs(120))).unwrap();
    let req = format!(
        "POST {path} HTTP/1.1\r\nHost: {host}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).unwrap();
    let mut raw = String::new();
    s.read_to_string(&mut raw).unwrap();
    let (head, resp_body) = raw
        .split_once("\r\n\r\n")
        .unwrap_or_else(|| panic!("no HTTP header/body split; got {} bytes", raw.len()));
    if head.to_ascii_lowercase().contains("transfer-encoding: chunked") {
        // dechunk: hex-size line, payload, repeat until 0
        let mut out = String::new();
        let mut rest = resp_body;
        loop {
            let Some((sz_line, after)) = rest.split_once("\r\n") else { break };
            let sz = usize::from_str_radix(sz_line.trim().split(';').next().unwrap(), 16)
                .unwrap_or_else(|_| panic!("bad chunk size {sz_line:?}"));
            if sz == 0 { break }
            out.push_str(&after[..sz.min(after.len())]);
            rest = &after[sz.min(after.len())..];
            rest = rest.strip_prefix("\r\n").unwrap_or(rest);
        }
        out
    } else {
        resp_body.to_string()
    }
}

/// llama.cpp baseline arm via the CHAT endpoint — the baseline's pinned
/// operating mode (--jinja). Raw /completion without chat markup makes this
/// chat-trained model emit EOS mid-think (4/5 empty answers measured);
/// enable_thinking:false gives deterministic direct answers (instruct mode).
fn llama_complete(url: &str, text: &str, max_tokens: usize) -> (f64, f64, String) {
    let host = url.trim_start_matches("http://").trim_end_matches('/').to_string();
    let body = serde_json::json!({
        "messages": [{ "role": "user", "content": text }],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "cache_prompt": true,
        "chat_template_kwargs": { "enable_thinking": false },
    });
    let resp = http_post_json(&host, "/v1/chat/completions", &body.to_string());
    let v: serde_json::Value = serde_json::from_str(&resp).unwrap_or_else(|e| {
        panic!("bad JSON from llama-server: {e}; first 400 B: {}", &resp[..resp.len().min(400)])
    });
    let msg = &v["choices"][0]["message"];
    let content = msg["content"].as_str().unwrap_or("").to_string();
    // llama-server appends its timings block to OpenAI-format responses too
    let timings = &v["timings"];
    let pre = timings["prompt_per_second"].as_f64().unwrap_or(0.0);
    let dec = timings["predicted_per_second"].as_f64().unwrap_or(0.0);
    (pre, dec, content)
}

// eos_token_id from the checkpoint's generation_config (248046 = <|im_end|>,
// 248044 = bos/pad/eos). Symmetry with the llama arm: the server stops on
// EOS, so the crow arm must too — otherwise every correct short answer is
// scored as degeneration (fable gate 2026-09-03). Raw traces stay recorded.

fn crow_complete(text: &str, max_tokens: usize) -> (f64, f64, String, Vec<i64>) {    // same production switches as `decode run` (2026-09-04): sidecar override,
    // prefill chunk, prompt-adaptive hot set after the prefill (charged to prefill)
    // defaults (#48): the production -M container and the id-sorted rectangular
    // sidecar serve.rs loads, both relative to the repo root (see the header at :44)
    //
    // #65 (2026-09-18): the oracle child is spawned BEFORE `open_model`, not between it
    // and `Engine::load`. Numerically inert — `tokenize` runs no kernel, reads no
    // config and the chunk policy still sees the same `ids.len()` before the load — but
    // it takes the python child out of the worst host-memory window this process has:
    // the container of this task is not mapped yet, the CUDA primary context does not
    // exist yet, and the ~44.6 GB the previous task pinned has had the whole return
    // path to itself. That window is the leading suspect of #65 (engine/README's
    // Windows rule is to wait for 50.5 GiB of free host RAM before a load).
    let ids = tokenize(text);
    let (mut cnq, _ctx, mut cfg, _cnq_path, sidecar) = unsafe {
        crow_nest_engine::boot::open_model(DEFAULT_CNQ.into(), DEFAULT_HOTSETS.into())
    };
    unsafe {
        // #16: CROW_CHUNK explicit, else auto by prompt length (geo.rs)
        crow_nest_engine::geo::apply_chunk_policy(&mut cfg, ids.len());
        let mut eng = crow_nest_engine::gen::Engine::load(
            &mut cnq, cfg, None, &sidecar, false, &mut |_| {},
        );
        let t0 = Instant::now();
        let mut next = eng.prefill(&mut cnq, &ids, None);
        if let Some((sw, cap0)) = eng.adapt_after_prefill() {
            eprintln!("[adapt] {sw} hot-slot swaps from the prompt routing (cap {cap0}/layer, 0 = unbounded)");
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        // same decode-time adaptation knobs as `decode run` (#17/#21):
        // CROW_ADAPT_EVERY=K re-cuts the hot set every K tokens, <= CROW_ADAPT_MAX swaps/layer
        // (#17: from geo::apply_adapt_policy - env in manual mode, else the
        // long-context switch: stream trickle 7 / 8 / 7 at chunk 2048 only)
        let (adapt_stream, adapt_every, adapt_max) = eng.cfg.adapt.knobs();
        let mut trickle_swaps = 0usize;
        let c0 = eng.drain_counters();
        let (ple_r0, ple_m0) = (eng.ple().req, eng.ple().miss);
        // #20: CROW_SAMPLE=1 -> sampling with the data-sheet profile: on the device
        // (sample_k behind argmax_k) unless CROW_SAMPLE_HOST=1 keeps the host path
        let mut sampler = crow_nest_engine::sample::Sampler::from_env();
        // #85/#92: a host-only knob (DRY, the #92 tier) takes the host path too
        let sample_host = crow_nest_engine::sample::host_forced()
            || sampler.as_ref().is_some_and(|s| s.host_route());
        if let Some(s) = &mut sampler {
            eprintln!("[{}]", s.describe());
            if sample_host {
                let lg = crow_nest_engine::cuda::dtoh(eng.logits(), crow_nest_engine::geo::V);
                next = s.sample(&lg);
                s.observe(next);
            } else {
                next = eng.arm_sampler(s);
            }
        }
        let mut answer: Vec<i64> = vec![next as i64];
        let mut stopped_eos = crow_nest_engine::sample::EOS_IDS_I64.contains(&(next as i64));
        let t1 = Instant::now();
        let mut steps = 1usize;
        while answer.len() < max_tokens && !stopped_eos {
            let s0 = Instant::now();
            if adapt_stream && adapt_every > 0 {
                trickle_swaps += eng.trickle_tick(steps % adapt_every == 0, adapt_max);
            } else if adapt_every > 0 && steps % adapt_every == 0 {
                trickle_swaps += eng.adapt_tick(adapt_max);
            }
            next = eng.decode_step(&mut cnq, next as i64);
            if sample_host {
                if let Some(s) = &mut sampler {
                    let lg = crow_nest_engine::cuda::dtoh(eng.logits(), crow_nest_engine::geo::V);
                    next = s.sample(&lg);
                    s.observe(next);
                }
            }
            answer.push(next as i64);
            steps += 1;
            if crow_nest_engine::sample::EOS_IDS_I64.contains(&(next as i64)) {
                stopped_eos = true;
            }
            if steps % 10 == 0 || answer.len() >= max_tokens || stopped_eos {
                eprintln!(
                    "[dec {}] {:.0} ms/tok — {:.1} tok/s",
                    answer.len(),
                    s0.elapsed().as_secs_f64() * 1000.0,
                    answer.len() as f64 / t1.elapsed().as_secs_f64().max(1e-9)
                );
            }
        }
        let dec_s = t1.elapsed().as_secs_f64();
        if adapt_stream && adapt_every > 0 {
            let _ = eng.trickle_drain();
        }
        {
            let c = eng.drain_counters();
            let (sel, cold): (u64, u64) = (
                c.iter().zip(c0.iter()).map(|(x, b)| x[0] - b[0]).sum(),
                c.iter().zip(c0.iter()).map(|(x, b)| x[1] - b[1]).sum(),
            );
            let n = (steps as f64 - 1.0).max(1.0);
            let (r, m) = (eng.ple().req - ple_r0, eng.ple().miss - ple_m0);
            eprintln!("[decode-stats] {} tokens: cold experts/token {:.1} of {:.0}, {:.0} MB/token zero-copy; ple misses/token {:.2} of {:.1}; trickle swaps {} (every {}, max {}/layer)",
                steps - 1, cold as f64 / n, sel as f64 / n, cold as f64 / n * (eng.residency().gu_bytes + eng.residency().dn_bytes) as f64 / 1e6,
                m as f64 / n, r as f64 / n, trickle_swaps, adapt_every, adapt_max);
        }
        (
            ids.len() as f64 / prefill_s,
            (steps as f64 - if stopped_eos { 0.0 } else { 1.0 }).max(1.0) / dec_s.max(1e-9),
            format!(
                "prompt_tokens {}{}{}",
                ids.len(),
                if stopped_eos {
                    format!("; stopped_eos at {}", answer.len())
                } else {
                    String::new()
                },
                // #20: a sampled answer names its profile and seed in the record
                match &sampler { Some(s) => format!("; {}", s.describe()), None => String::new() }
            ),
            answer,
        )
    }
}

/// The greedy operating point. It is what the LLAMA arm always is: `llama_complete`
/// sends `"temperature": 0.0` on every request (`:136`) and reads no engine env var,
/// so nothing a crow phase sets can change what that arm drew.
const GREEDY_POINT: &str = "200k floor, -np 1, greedy, temperature 0";

/// #53: the record header names the sampler that produced the answer — and since
/// #60 it names the sampler of the ARM whose record it heads.
///
/// | arm | case | `operating_point` |
/// |---|---|---|
/// | `crow` | `CROW_SAMPLE` unset | `200k floor, -np 1, greedy, temperature 0` |
/// | `crow` | `CROW_SAMPLE=1` | `200k floor, -np 1, sample: temp <t> top_p <p> top_k <k> presence <pr> seed <s>` |
/// | `llama` | either | `200k floor, -np 1, greedy, temperature 0` |
///
/// - Source for the crow arm is `sample::Sampler::from_env`, the call `crow_complete`
///   already makes for `measurements[].note` (`sample.rs:86`).
/// - The note keeps the `gpu` or `host` suffix of `Sampler::describe`; the header
///   names the profile and the seed only.
/// - Before #53 both record sites stamped `greedy` on sampled runs too.
/// - #60 (2026-09-18): `CROW_SAMPLE` is a CROW-side switch, and the header was
///   arm-independent. A llama phase started with `CROW_SAMPLE=1` still in the
///   environment — the arms run in separate phases on this machine, one shell — would
///   have stamped a sampler profile and a seed on answers drawn at temperature 0.
///   Latent (no llama record on this branch carries it), and now impossible by
///   construction: the arm decides, and `point_for` is pinned by the tests below.
fn operating_point(arm: &str) -> String {
    point_for(arm, crow_nest_engine::sample::Sampler::from_env().as_ref())
}

/// the pure half of `operating_point`: the arm, and the sampler that arm used. No
/// environment and no server, so the arm rule is unit testable on any machine.
fn point_for(arm: &str, sampler: Option<&crow_nest_engine::sample::Sampler>) -> String {
    match sampler {
        Some(s) if arm != "llama" => format!(
            "200k floor, -np 1, sample: temp {} top_p {} top_k {} presence {} seed {}",
            s.temperature, s.top_p, s.top_k, s.presence_penalty, s.seed
        ),
        // the llama arm, and every unsampled crow arm
        _ => GREEDY_POINT.to_string(),
    }
}

fn rotate(prompts_all: &[Prompt], run_index: usize) -> (Vec<Prompt>, &'static str, &'static str) {
    let n = prompts_all.len();
    let shift = (run_index * 3) % n; // coprime-ish stride; reverse on odd runs
    let mut prompts: Vec<Prompt> = prompts_all[shift..]
        .iter()
        .chain(prompts_all[..shift].iter())
        .cloned()
        .collect();
    if run_index % 2 == 1 {
        prompts.reverse();
    }
    // first mover per run: even run → crow moves first, odd run → llama
    let first = if run_index % 2 == 0 { "crow" } else { "llama" };
    let second = if run_index % 2 == 0 { "llama" } else { "crow" };
    (prompts, first, second)
}

fn load_prompts(path: &str) -> Vec<Prompt> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| Prompt {
            id: p["id"].as_str().unwrap().into(),
            text: p["text"].as_str().unwrap().into(),
            max_tokens: p["max_tokens"].as_u64().unwrap_or(64) as usize,
        })
        .collect()
}

fn main() {
    // #13: the logging subscriber of this process. Every library line this bin
    // triggers (`[prefill]`, `[load]`, `[budget]`, `[ple]`, ...) is a `tracing`
    // event now, so without this call they go nowhere. The guard drains the two
    // writer threads when `main` returns; an `exit` below calls `shutdown` first.
    let _log = crow_nest_engine::log::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        println!("usage: parity <run> <prompts.json> [llama-url]\n       parity <phase> <run_index> <crow|llama> <prompts.json> [llama-url] [outprefix]\n  run   = interleaved (needs both engines co-resident)\n  phase = one arm over the full rotated order (RAM/VRAM discipline on this machine)");
        return;
    }
    let url = args
        .iter()
        .find(|a| a.starts_with("http://"))
        .cloned()
        .unwrap_or_else(|| "http://127.0.0.1:8083".into());
    let mode = args[1].as_str();

    match mode {
        // ---- arm-phased execution (used for the ten-task gate on this machine)
        "phase" => {
            let run_index: usize = args[2].parse().unwrap_or(0);
            let arm = args[3].as_str();
            assert!(arm == "crow" || arm == "llama", "arm must be crow|llama");
            let prompts_all = load_prompts(&args[4]);
            let prefix = args
                .iter()
                .skip(5)
                .find(|a| !a.starts_with("http://"))
                .map(|s| s.as_str())
                .unwrap_or("parity");
            let (prompts, first, _second) = rotate(&prompts_all, run_index);

            println!("warm-up (discarded): {arm} throwaway request …");
            // #65: every oracle child from here on is stamped with the task it serves,
            // so a retry in the record names the task instead of leaving it to be
            // reconstructed from the rotated order.
            oracle_label("warm-up");
            if arm == "llama" {
                let _ = llama_complete(&url, "warm-up", 1);
            } else {
                let _ = crow_complete("warm-up", 1);
            }

            let mut report = vec![serde_json::json!({
                "run_index": run_index,
                "arm": arm,
                "execution": "arm-phased (exclusive resources; paired rotated order per arm, identical to the other arm's)",
                "first_mover_rule": first,
                "order": prompts.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
                "warmup": "one cold prefill per phase start, discarded (spec 0.3)",
                "operating_point": operating_point(arm),
                "crow_container": std::env::var("CROW_CNQ").unwrap_or_else(|_| DEFAULT_CNQ.into()),
            })];
            // #65: keys that appear only when something actually had to be retried, so a
            // clean phase writes exactly the record it wrote before the guard existed.
            let note_retries = |row: &mut serde_json::Value, key: &str| {
                let retried = drain_oracle_attempts();
                if !retried.is_empty() {
                    row[key] = serde_json::json!(retried);
                }
            };
            note_retries(&mut report[0], "oracle_retries_warmup");
            let out = format!("decode_out/{prefix}-run{run_index}-{arm}.json");
            let mut answers = serde_json::Map::new();
            // incremental persistence: each completed task is written out — a
            // crash mid-phase keeps the finished answers (lesson 2026-09-03:
            // a 3h crow phase crashed at task 1 and lost the whole arm)
            let write_report = |report: &[serde_json::Value],
                                answers: &serde_json::Map<String, serde_json::Value>,
                                detok: &serde_json::Value,
                                out: &str| {
                std::fs::write(
                    out,
                    serde_json::to_string_pretty(&serde_json::json!({
                        "meta": report[0], "measurements": report[1..].to_vec(),
                        "answers": answers, "crow_answers_text": detok,
                    })).unwrap(),
                ).unwrap();
            };
            for (i, p) in prompts.iter().enumerate() {
                let t0 = Instant::now();
                oracle_label(&format!("{} (position {i})", p.id));
                let (pre, dec, note, answer_ids) = if arm == "crow" {
                    let (pre, dec, note, ids) = crow_complete(&p.text, p.max_tokens);
                    answers.insert(
                        p.id.clone(),
                        serde_json::json!({ "answer_ids": ids, "max_tokens": p.max_tokens }),
                    );
                    (pre, dec, note, ids)
                } else {
                    let (pre, dec, content) = llama_complete(&url, &p.text, p.max_tokens);
                    answers.insert(
                        p.id.clone(),
                        serde_json::json!({ "answer_text": content, "max_tokens": p.max_tokens }),
                    );
                    (pre, dec, format!("answer_chars {}", content.len()), Vec::new())
                };
                let _ = answer_ids;
                let wall = t0.elapsed().as_secs_f64();
                report.push(serde_json::json!({
                    "prompt": p.id, "arm": arm, "position_in_series": i,
                    "prefill_tok_s": pre, "decode_tok_s": dec, "wall_s": wall,
                    "note": note,
                }));
                let last = report.len() - 1;
                note_retries(&mut report[last], "oracle_retries");
                // incremental snapshot keeps raw ids; detokenize runs once at
                // the end (a python+transformers start per task would waste ~15s each)
                let detok = serde_json::json!(null);
                write_report(&report, &answers, &detok, &out);
                println!("{:>5} {:6} pos {i} pre {pre:8.1} tok/s  dec {dec:8.1} tok/s  wall {wall:7.1}s",
                    p.id, arm);
            }
            oracle_label("detokenize (end of phase)");
            let detok = if arm == "crow" { detokenize_all(&answers) } else { serde_json::json!(null) };
            note_retries(&mut report[0], "oracle_retries_detok");
            write_report(&report, &answers, &detok, &out);
            println!("phase done: written {out}");
        }
        // ---- interleaved execution (original skeleton path; kept for a
        // machine where both engines fit side by side)
        "run" => {
            let run_index: usize = args[2].parse().unwrap_or(0);
            let prompts_all = load_prompts(&args[3]);
            let (prompts, first, second) = rotate(&prompts_all, run_index);

            println!("warm-up (discarded): llama.cpp throwaway request …");
            let _ = llama_complete(&url, "warm-up", 1);
            println!("warm-up (discarded): crow-nest throwaway request …");
            let _ = crow_complete("warm-up", 1);

            let mut report = vec![serde_json::json!({
                "run_index": run_index,
                "order": prompts.iter().map(|p| p.id.clone()).collect::<Vec<_>>(),
                "first_mover": first,
                "warmup": "one cold prefill per arm, discarded (server start / engine load)",
            })];
            for (i, p) in prompts.iter().enumerate() {
                let interleave = if (i + run_index) % 2 == 0 { (first, second) } else { (second, first) };
                for which in [interleave.0, interleave.1] {
                    let t0 = Instant::now();
                    oracle_label(&format!("{} (position {i}, {which})", p.id));
                    let (pre, dec, note, _ids) = if which == "crow" {
                        crow_complete(&p.text, p.max_tokens)
                    } else {
                        let (pre, dec, content) = llama_complete(&url, &p.text, p.max_tokens);
                        (pre, dec, content, Vec::new())
                    };
                    let wall = t0.elapsed().as_secs_f64();
                    report.push(serde_json::json!({
                        "prompt": p.id, "engine": which, "position_in_series": i,
                        "prefill_tok_s": pre, "decode_tok_s": dec, "wall_s": wall,
                        "note": note,
                        "operating_point": operating_point(which),
                    }));
                    let retried = drain_oracle_attempts();
                    if !retried.is_empty() {
                        let last = report.len() - 1;
                        report[last]["oracle_retries"] = serde_json::json!(retried);
                    }
                    println!("{:>5} {:6} pre {pre:8.1} tok/s  dec {dec:8.1} tok/s", p.id, which);
                }
            }
            let out = format!("decode_out/parity-run{run_index}.json");
            std::fs::write(&out, serde_json::to_string_pretty(&report).unwrap()).unwrap();
            println!("parity: written {out}");
        }
        _ => println!("unknown mode {mode}"),
    }
}

/// #60 (2026-09-18): the arm rule of the record header, pinned without a
/// llama-server, without the oracle venv and without a GPU — `point_for` is pure
/// and `Sampler::new` is the constructor that reads no environment.
#[cfg(test)]
mod tests {
    use super::{operating_point, point_for, GREEDY_POINT};
    use crow_nest_engine::sample::Sampler;

    /// The llama arm draws through `llama_complete`, which sends `temperature 0.0`
    /// on every request, so its header says greedy whatever sampler the crow arm of
    /// the same series used. The crow arm keeps the #53 behaviour.
    #[test]
    fn the_llama_arm_header_is_greedy_whatever_the_sampler_is() {
        let s = Sampler::new(4); // data-sheet instruct profile, seed 4
        assert_eq!(point_for("llama", Some(&s)), GREEDY_POINT);
        assert_eq!(point_for("llama", None), GREEDY_POINT);
        assert_eq!(point_for("crow", None), GREEDY_POINT);
        assert_eq!(
            point_for("crow", Some(&s)),
            "200k floor, -np 1, sample: temp 0.7 top_p 0.8 top_k 20 presence 1.5 seed 4"
        );
    }

    /// The same rule through the env-reading front door, with the profile of a crow
    /// phase left in the environment — the shape of the accident #60 names: one
    /// shell, the crow phase first with `CROW_SAMPLE=1`, then the llama phase.
    #[test]
    fn a_llama_phase_started_with_crow_sample_still_records_greedy() {
        std::env::set_var("CROW_SAMPLE", "1");
        assert_eq!(operating_point("llama"), GREEDY_POINT);
        let crow = operating_point("crow");
        assert!(crow.starts_with("200k floor, -np 1, sample: temp "), "{crow}");
        std::env::remove_var("CROW_SAMPLE");
        assert_eq!(operating_point("crow"), GREEDY_POINT);
    }
}

/// #65 (2026-09-18): the guard around the two oracle children, pinned WITHOUT the oracle
/// venv, without a llama-server and without a GPU. `child_verdict` is pure; the retry
/// itself is driven against a `/bin/sh` stub child, which is why `oracle_child` takes its
/// program and its arguments instead of reading `ORACLE_PY`.
#[cfg(test)]
mod oracle_guard {
    use super::child_verdict;

    /// The whole reason #65 could not be pinned: the call site printed the child's stderr
    /// and threw the exit status away, and all three failures had an EMPTY stderr. This is
    /// the table that must never be silent again.
    #[test]
    fn a_child_that_says_nothing_is_still_named() {
        // green is a zero exit WITH something on stdout
        assert!(child_verdict(Some(0), None, "", 12).is_ok());
        assert!(child_verdict(Some(0), None, "a warning\n", 12).is_ok());
        // a zero exit with an EMPTY stdout is not green: nothing is not a token stream
        let d = child_verdict(Some(0), None, "", 0).unwrap_err();
        assert!(d.contains("exit code 0 but 0 B on stdout"), "{d}");
        // the #65 shape itself: non-zero, nothing on stderr. The exit code has to be in
        // the text, because it is the only thing such a failure says.
        let d = child_verdict(Some(1), None, "", 0).unwrap_err();
        assert!(d.contains("exit code 1 (0x00000001)"), "{d}");
        assert!(d.contains("stderr EMPTY"), "{d}");
        assert!(d.contains("0 B on stdout"), "{d}");
        // the Windows deaths whose NAME is the whole message — a loader failure and a
        // native crash both reach the parent as an exit code and an empty stderr
        let d = child_verdict(Some(0xC000_0142_u32 as i32), None, "", 0).unwrap_err();
        assert!(d.contains("0xc0000142") && d.contains("STATUS_DLL_INIT_FAILED"), "{d}");
        let d = child_verdict(Some(0xC000_0005_u32 as i32), None, "", 0).unwrap_err();
        assert!(d.contains("STATUS_ACCESS_VIOLATION"), "{d}");
        let d = child_verdict(Some(120), None, "", 0).unwrap_err();
        assert!(d.contains("flush its std streams"), "{d}");
        // a child that DID reach Python's own error path is quoted, and is not called silent
        let d = child_verdict(Some(1), None, "ImportError: no jinja2\n", 0).unwrap_err();
        assert!(d.contains("ImportError: no jinja2"), "{d}");
        assert!(!d.contains("stderr EMPTY"), "{d}");
        // the unix arm: a killed child has no exit code at all
        let d = child_verdict(None, Some(9), "", 0).unwrap_err();
        assert!(d.contains("killed by signal 9"), "{d}");
        let d = child_verdict(None, None, "", 0).unwrap_err();
        assert!(d.contains("no exit code and no signal"), "{d}");
    }

    /// The retry against a REAL child process. `/bin/sh` is the stub, so the test needs no
    /// venv and no GPU; the Windows arm of the harness runs the same pure verdict above and
    /// the same `oracle_child` loop, which has nothing platform-specific in it.
    #[cfg(unix)]
    mod stub_child {
        use super::super::{drain_oracle_attempts, oracle_child, oracle_label};
        use std::sync::Mutex;

        /// `ORACLE_LOG` is one process-wide log, so the tests that read it take turns.
        static TURN: Mutex<()> = Mutex::new(());

        /// writes a stub child and returns (script path, marker path), both unique to `tag`
        fn stub(tag: &str, body: &str) -> (String, String) {
            let dir = std::env::temp_dir().join(format!("crow-oracle-guard-{tag}-{}", std::process::id()));
            std::fs::create_dir_all(&dir).unwrap();
            let sh = dir.join("stub.sh");
            let marker = dir.join("marker");
            let _ = std::fs::remove_file(&marker);
            std::fs::write(&sh, body).unwrap();
            (sh.to_str().unwrap().to_string(), marker.to_str().unwrap().to_string())
        }

        /// The accident of #65 in miniature: the child fails ONCE with an empty stderr and
        /// a non-zero exit, and the second attempt is green. The task goes on, the phase is
        /// not restarted, and the failed attempt is in the record with its exit code.
        #[test]
        fn a_child_that_fails_once_with_an_empty_stderr_is_retried() {
            let _turn = TURN.lock().unwrap();
            let (sh, marker) = stub(
                "once",
                // reads its stdin to EOF first, the way tokenize_ids.py does, then dies
                // the #65 death on the first call only: no stderr, exit 1
                "cat > /dev/null\nif [ ! -f \"$1\" ]; then : > \"$1\"; exit 1; fi\nprintf '[11,22,33]'\n",
            );
            let _ = drain_oracle_attempts();
            oracle_label("t5-agent (position 4)");
            // backoff 0 ms: the pause is the production constant's business, not the loop's
            let out = oracle_child("/bin/sh", &[&sh, &marker], b"the prompt", "stub --chat", &[0]);
            assert_eq!(String::from_utf8_lossy(&out), "[11,22,33]");
            let rec = drain_oracle_attempts();
            assert_eq!(rec.len(), 1, "{rec:?}");
            assert_eq!(rec[0]["at"], "t5-agent (position 4)");
            assert_eq!(rec[0]["child"], "stub --chat");
            assert_eq!(rec[0]["attempt"], 1);
            assert_eq!(rec[0]["of"], 2);
            assert_eq!(rec[0]["stdin_bytes"], 10);
            let d = rec[0]["diagnosis"].as_str().unwrap();
            assert!(d.contains("exit code 1") && d.contains("stderr EMPTY"), "{d}");
        }

        /// Fail-closed: a child that fails EVERY attempt still fails the task, with the
        /// whole table of attempts in the panic and every attempt in the record.
        #[test]
        fn a_child_that_fails_every_attempt_still_fails_the_task() {
            let _turn = TURN.lock().unwrap();
            let (sh, _m) = stub("always", "cat > /dev/null\nexit 3\n");
            let _ = drain_oracle_attempts();
            oracle_label("t1-read (position 0)");
            let hook = std::panic::take_hook();
            std::panic::set_hook(Box::new(|_| {})); // the panic is the assertion, not noise
            let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                oracle_child("/bin/sh", &[&sh], b"x", "stub --chat", &[0, 0])
            }));
            std::panic::set_hook(hook);
            let msg = *res.unwrap_err().downcast::<String>().unwrap();
            assert!(msg.contains("all 3 attempts failed"), "{msg}");
            assert!(msg.contains("fail-closed"), "{msg}");
            assert!(msg.contains("attempt 3: exit code 3"), "{msg}");
            let rec = drain_oracle_attempts();
            assert_eq!(rec.len(), 3, "{rec:?}");
            assert_eq!(rec[2]["attempt"], 3);
            assert_eq!(rec[0]["at"], "t1-read (position 0)");
        }

        /// The payload hazard the ticket names: the prompt goes through STDIN and never
        /// through the command line (`CreateProcess` caps at 32,767 chars, and the longest
        /// of the ten frozen prompts is 60,290), it arrives WHOLE, and the child's
        /// read-to-EOF ends — which it only can because this scope closes stdin.
        #[test]
        fn the_whole_payload_reaches_the_child_through_stdin() {
            let (sh, _m) = stub("stdin", "printf '[%s]' \"$(cat | wc -c)\"\n");
            let payload = vec![b'x'; 60_290];
            let out = oracle_child("/bin/sh", &[&sh], &payload, "stub --chat", &[]);
            assert_eq!(String::from_utf8_lossy(&out).replace(' ', ""), "[60290]");
        }
    }
}
