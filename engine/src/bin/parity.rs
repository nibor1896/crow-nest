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

use std::io::{Read, Write};
use std::process::{Command, Stdio};
use std::time::Instant;

#[derive(Clone)]
struct Prompt {
    id: String,
    text: String,
    max_tokens: usize,
}

fn tokenize(text: &str) -> Vec<i64> {
    // CWD must be the crow-nest repo root (all harness paths are repo-relative)
    // long prompts go through STDIN — the Windows command line caps at ~32k chars
    // --chat: the crow arm receives the SAME token stream as the llama.cpp arm
    // (chat template, thinking disabled) — parity-gate fairness, fable gate 2026-09-03
    let py = ".venv-oracle/Scripts/python.exe";
    let mut child = Command::new(py)
        .arg("tools/tokenize_ids.py")
        .arg("--chat")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .expect("tokenize_ids.py (oracle venv) failed to start");
    child.stdin.as_mut().unwrap().write_all(text.as_bytes()).unwrap();
    let out = child.wait_with_output().expect("tokenize_ids.py (oracle venv) failed");
    assert!(out.status.success(), "tokenize stderr: {}", String::from_utf8_lossy(&out.stderr));
    let v: serde_json::Value = serde_json::from_slice(&out.stdout).unwrap();
    v.as_array().unwrap().iter().map(|x| x.as_i64().unwrap()).collect()
}

fn detokenize_all(map: &serde_json::Map<String, serde_json::Value>) -> serde_json::Value {
    let inp = serde_json::json!(map).to_string();
    let tmp = "decode_out/_detok_in.json";
    std::fs::write(tmp, inp).unwrap();
    let py = ".venv-oracle/Scripts/python.exe";
    let out = std::process::Command::new(py)
        .arg("tools/detokenize_ids.py")
        .arg(tmp)
        .output()
        .expect("detokenize_ids.py (oracle venv) failed");
    assert!(out.status.success(), "detok stderr: {}", String::from_utf8_lossy(&out.stderr));
    serde_json::from_slice(&out.stdout).unwrap()
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
    let reasoning = msg["reasoning_content"].as_str().unwrap_or("");
    let note = if reasoning.is_empty() {
        format!("answer_chars {}", content.len())
    } else {
        format!("answer_chars {} + reasoning_chars {}", content.len(), reasoning.len())
    };
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
const EOS_STOP: [i64; 2] = [248046, 248044];

fn crow_complete(text: &str, max_tokens: usize) -> (f64, f64, String, Vec<i64>) {    let cnq_path = std::env::var("CROW_CNQ")
        .unwrap_or_else(|_| "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq".into());
    // same production switches as `decode run` (2026-09-04): sidecar override,
    // prefill chunk, prompt-adaptive hot set after the prefill (charged to prefill)
    // defaults (#48): the production -M container and the id-sorted rectangular
    // sidecar serve.rs loads, both relative to the repo root (see the header at :44)
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or_else(|_| "decode_out/hotsets-M-longctx2100-n160.json".into());
    let mut cnq = crow_nest_engine::cnq::Cnq::open(&cnq_path);
    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        let mut cfg = crow_nest_engine::geo::Config::default();
        cfg.context = crow_nest_engine::geo::CONTEXT_FLOOR;
        let ids = tokenize(text);
        // #16: CROW_CHUNK explicit, else auto by prompt length (geo.rs)
        crow_nest_engine::geo::apply_chunk_policy(&mut cfg, ids.len());
        let (mut eng, _) = crow_nest_engine::gen::Engine::load(
            &mut cnq, cfg, None, &sidecar, false, &mut |_| {},
        );
        let t0 = Instant::now();
        let mut next = eng.prefill(&mut cnq, &ids, None);
        if std::env::var("CROW_ADAPT").as_deref() == Ok("1") {
            let cap0: usize = std::env::var("CROW_ADAPT_MAX0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
            let sw = eng.adapt_hot_set(cap0);
            eprintln!("[adapt] {sw} hot-slot swaps from the prompt routing (cap {cap0}/layer, 0 = unbounded)");
        }
        let prefill_s = t0.elapsed().as_secs_f64();
        // same decode-time adaptation knobs as `decode run` (#17/#21):
        // CROW_ADAPT_EVERY=K re-cuts the hot set every K tokens, <= CROW_ADAPT_MAX swaps/layer
        // (#17: from geo::apply_adapt_policy - env in manual mode, else the
        // long-context switch: stream trickle 7 / 16 / 7 at chunk 2048 only)
        let crow_nest_engine::geo::Adapt { stream: adapt_stream, every: adapt_every, max: adapt_max, .. } = eng.cfg.adapt;
        let mut trickle_swaps = 0usize;
        let c0 = eng.drain_counters();
        let (ple_r0, ple_m0) = (eng.ple.req, eng.ple.miss);
        // #20: CROW_SAMPLE=1 -> sampling with the data-sheet profile: on the device
        // (sample_k behind argmax_k) unless CROW_SAMPLE_HOST=1 keeps the host path
        let mut sampler = crow_nest_engine::sample::Sampler::from_env();
        let sample_host = crow_nest_engine::sample::host_forced();
        if let Some(s) = &mut sampler {
            eprintln!("[{}]", s.describe());
            if sample_host {
                let lg = crow_nest_engine::cuda::dtoh(eng.s.logits, crow_nest_engine::geo::V);
                next = s.sample(&lg);
                s.observe(next);
            } else {
                eng.enable_dev_sampler(s);
                next = eng.sample_last();
            }
        }
        let mut answer: Vec<i64> = vec![next as i64];
        let mut stopped_eos = EOS_STOP.contains(&(next as i64));
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
                    let lg = crow_nest_engine::cuda::dtoh(eng.s.logits, crow_nest_engine::geo::V);
                    next = s.sample(&lg);
                    s.observe(next);
                }
            }
            answer.push(next as i64);
            steps += 1;
            if EOS_STOP.contains(&(next as i64)) {
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
            let (r, m) = (eng.ple.req - ple_r0, eng.ple.miss - ple_m0);
            eprintln!("[decode-stats] {} tokens: cold experts/token {:.1} of {:.0}, {:.0} MB/token zero-copy; ple misses/token {:.2} of {:.1}; trickle swaps {} (every {}, max {}/layer)",
                steps - 1, cold as f64 / n, sel as f64 / n, cold as f64 / n * (eng.res.gu_bytes + eng.res.dn_bytes) as f64 / 1e6,
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
                "operating_point": "200k floor, -np 1, greedy, temperature 0",
                "crow_container": std::env::var("CROW_CNQ").unwrap_or_else(|_| "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq".into()),
            })];
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
                // incremental snapshot keeps raw ids; detokenize runs once at
                // the end (a python+transformers start per task would waste ~15s each)
                let detok = serde_json::json!(null);
                write_report(&report, &answers, &detok, &out);
                println!("{:>5} {:6} pos {i} pre {pre:8.1} tok/s  dec {dec:8.1} tok/s  wall {wall:7.1}s",
                    p.id, arm);
            }
            let detok = if arm == "crow" { detokenize_all(&answers) } else { serde_json::json!(null) };
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
                    let (pre, dec, note, ids) = if which == "crow" {
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
                        "operating_point": "200k floor, -np 1, greedy",
                    }));
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
