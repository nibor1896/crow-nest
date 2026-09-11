//! #11 — first end-to-end production decode.
//!
//! Modes:
//!   decode parity <ids.json> <out_dir>   — short-T run, ALL positions' logits
//!       dumped for the oracle gate (f32 [T][248320]), greedy trace json.
//!   decode run <ids.json> <gen> <out_dir> — prefill + N decode steps, greedy
//!       trace + per-step timing (the standing-series engine side).
//!   decode longctx <prompt_ids> <fill> <gen> — QSA sparse regime: fill the
//!       context past the 2048 budget with real prefill, then decode steps.
//!
//! Token ids come from tools/tokenize.py (oracle venv, transformers 5.16.1
//! tokenizer — the same tokenization the reference side uses).

use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::geo::*;
use crow_nest_engine::gen::{Engine, PW, SubW};

fn read_ids(path: &str) -> Vec<i64> {
    let v: serde_json::Value =
        serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
    v.as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_i64().unwrap())
        .collect()
}

fn write_f32(path: &str, v: &[f32]) {
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    std::fs::write(path, b).unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mode = args.get(1).map(|s| s.as_str()).unwrap_or("help");

    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| "../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq".into());
    // CROW_CNQ and CROW_HOTSETS override container and hot-set sidecar
    // (e.g. a sidecar warmed on real traffic via `decode warmup`)
    // defaults (#48): the production -M container and the id-sorted rectangular
    // sidecar serve.rs loads, both relative to engine/, the cwd of `decode`
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or_else(|_| "../decode_out/hotsets-M-longctx2100-n160.json".into());
    let mut cnq = Cnq::open(&cnq_path);

    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        let mut cfg = Config::default();
        match mode {
            "parity" => {
                let ids_path = args[2].clone();
                let out = args[3].clone();
                let ids = read_ids(&ids_path);
                // parity runs short — chunk = whole prompt, dense QSA regime
                cfg.context = CONTEXT_FLOOR;
                cfg.prompt_chunk = ids.len().max(1);
                // CROW_CHUNK: scratch/chunk size override (determinism bisect: C=8 vs 512)
                if let Some(c) = std::env::var("CROW_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()) {
                    cfg.prompt_chunk = c.max(ids.len());
                }
                // parity ladder switches (bisect FP4 / FP8-KV / PLE)
                if std::env::var("CROW_KV").as_deref() == Ok("bf16") {
                    cfg.kv = KvDtype::Bf16;
                }
                if std::env::var("CROW_PLE").as_deref() == Ok("off") {
                    cfg.ple = false;
                }
                // CROW_PARITY_PREFILL=<n> (#11, 2026-09-05): prefill only ids[..n] and feed
                // ids[n..] teacher-forced through decode_step (one logits row per step) —
                // decode-path rows against prefill-path rows under the same context
                let tf_split: usize = std::env::var("CROW_PARITY_PREFILL").ok()
                    .and_then(|v| v.parse().ok()).unwrap_or(ids.len()).clamp(1, ids.len());
                if tf_split < ids.len() {
                    cfg.prompt_chunk = tf_split;
                }
                std::fs::create_dir_all(&out).unwrap();
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| println!("[load] {m}"));
                println!("decode/parity: {} prompt tokens ({} prefilled, {} teacher-forced), collecting all logits …",
                    ids.len(), tf_split, ids.len() - tf_split);
                let mut logits = Vec::new();
                let t0 = std::time::Instant::now();
                let mut tok = eng.prefill(&mut cnq, &ids[..tf_split], Some(&mut logits));
                println!("prefill+logits in {:.1} s", t0.elapsed().as_secs_f64());
                let mut tf_trace: Vec<usize> = vec![tok];
                for &fed in &ids[tf_split..] {
                    tok = eng.decode_step(&mut cnq, fed);
                    logits.push(crow_nest_engine::cuda::dtoh(eng.s.logits, V));
                    tf_trace.push(tok);
                }
                if tf_split < ids.len() {
                    println!("teacher-forced decode: {} steps, engine greedy trace {:?}", ids.len() - tf_split, tf_trace);
                }
                if std::env::var("CROW_ADAPT").as_deref() == Ok("1") {
                    // CROW_ADAPT_MAX0=<n>: cap the post-prefill swaps per layer (#21); 0 = unbounded
                    let cap0: usize = std::env::var("CROW_ADAPT_MAX0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let sw = eng.adapt_hot_set(cap0);
                    println!("adapt: {sw} hot-slot swaps");
                }

                // a few decode steps (persistent state, logits collected too)
                let mut all_ids = ids.clone();
                let mut next = tok;
                for _ in 0..4 {
                    all_ids.push(next as i64);
                    let pos = eng.pos;
                    let t0 = std::time::Instant::now();
                    next = eng.decode_step(&mut cnq, next as i64);
                    println!("  decode pos {pos} → {next} ({:.1} ms)", t0.elapsed().as_secs_f64() * 1e3);
                    // recompute logits row for this position (decode wrote row 0)
                    // — captured via head_run inside decode_step; read it back:
                    let lg = crow_nest_engine::cuda::dtoh(eng.s.logits, V);
                    logits.push(lg);
                }
                all_ids.push(next as i64);

                let mut nan = 0usize;
                for row in &logits {
                    for &x in row {
                        if x.is_nan() {
                            nan += 1;
                        }
                    }
                }
                write_f32(&format!("{out}/gpu-logits.f32"), &logits.concat());
                serde_json::to_writer(
                    std::fs::File::create(format!("{out}/gen-sequence.json")).unwrap(),
                    &serde_json::json!({
                        "prompt_len": ids.len(),
                        "prefill_len": tf_split,
                        "tf_trace": tf_trace,
                        "all_ids": all_ids,
                        "rows": logits.len(),
                        "nan": nan,
                        "note": "production FP4 weights, FP8-KV, QSA dense (short T); logits [rows][248320] f32",
                    }),
                )
                .unwrap();
                println!("decode/parity: {} rows → {out}/gpu-logits.f32 (NaN={nan})", logits.len());
            }
            "run" | "longctx" => {
                let ids_path = args[2].clone();
                let gen: usize = args.get(3).and_then(|v| v.parse().ok()).unwrap_or(32);
                let ids = read_ids(&ids_path);
                cfg.context = CONTEXT_FLOOR;
                // #16: CROW_CHUNK explicit, else auto by prompt length (geo.rs)
                crow_nest_engine::geo::apply_chunk_policy(&mut cfg, ids.len());
                std::fs::create_dir_all("decode_out").unwrap();
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| println!("[load] {m}"));
                println!("decode/run: {} prompt tokens → {gen} steps", ids.len());
                let t0 = std::time::Instant::now();
                let mut next = eng.prefill(&mut cnq, &ids, None);
                let prefill_s = t0.elapsed().as_secs_f64();
                if std::env::var("CROW_ADAPT").as_deref() == Ok("1") {
                    let ta = std::time::Instant::now();
                    // CROW_ADAPT_MAX0=<n>: cap the post-prefill swaps per layer (#21); 0 = unbounded
                    let cap0: usize = std::env::var("CROW_ADAPT_MAX0").ok().and_then(|v| v.parse().ok()).unwrap_or(0);
                    let sw = eng.adapt_hot_set(cap0);
                    println!("adapt: {sw} hot-slot swaps from the prompt routing in {:.2} s", ta.elapsed().as_secs_f64());
                }
                println!(
                    "prefill done in {:.2} s ({:.0} tok/s completed-prompt average)",
                    prefill_s,
                    ids.len() as f64 / prefill_s
                );

                // #20: CROW_SAMPLE=1 samples with the data-sheet profile (env-
                // overridable) - on the device (sample_k behind argmax_k, captured
                // with the graph, so it must be enabled before the warm-up step)
                // unless CROW_SAMPLE_HOST=1 keeps the host path (logits readback)
                let mut sampler = crow_nest_engine::sample::Sampler::from_env();
                let sample_host = crow_nest_engine::sample::host_forced();
                if let Some(s) = &sampler {
                    println!("{}", s.describe());
                    if !sample_host {
                        eng.enable_dev_sampler(s);
                    }
                }
                // warm-up step discarded (spec 0.3 measurement discipline)
                let t0 = std::time::Instant::now();
                next = eng.decode_step(&mut cnq, next as i64);
                let warm = t0.elapsed().as_secs_f64() * 1e3;
                println!("warm-up decode step (discarded): {warm:.2} ms → {next}");

                // counter baseline AFTER prefill + warm-up: cold/token below is per
                // timed decode token (the counters are cumulative since load)
                let c0 = eng.drain_counters();
                let (ple_r0, ple_m0) = (eng.ple.req, eng.ple.miss);
                println!("ple rows during prefill+warm-up: {} requested, {} misses ({:.1} %), cache slots {}",
                    ple_r0, ple_m0, 100.0 * ple_m0 as f64 / ple_r0.max(1) as f64, eng.ple.n_slots);
                {
                    let (s0, k0): (u64, u64) = (c0.iter().map(|x| x[0]).sum(), c0.iter().map(|x| x[1]).sum());
                    println!("cold experts during prefill+warm-up: {:.1} per token of {:.0} selections ({} tokens)",
                        k0 as f64 / (ids.len() + 1) as f64, s0 as f64 / (ids.len() + 1) as f64, ids.len() + 1);
                }
                let mut lat = Vec::new();
                let mut trace = vec![next];
                // host path: re-draw the warm-up token from its logits row;
                // CROW_STOP_EOS=1 ends the run at EOS
                if sample_host {
                    if let Some(s) = &mut sampler {
                        let lg = crow_nest_engine::cuda::dtoh(eng.s.logits, V);
                        next = s.sample(&lg);
                        s.observe(next);
                        trace[0] = next;
                    }
                }
                let stop_eos = crow_nest_engine::sample::stop_on_eos();
                let mut stopped_eos = stop_eos && crow_nest_engine::sample::EOS_IDS.contains(&next);
                // CROW_ADAPT_EVERY=K: re-cut the hot set from the cumulative routing
                // every K decode tokens (<= CROW_ADAPT_MAX swaps per layer, default 8);
                // the swap time is charged to that token's latency (amortized cost)
                // (#17: the knobs come from geo::apply_adapt_policy - env in manual
                // mode, the long-context switch otherwise; see the [policy] line)
                let crow_nest_engine::geo::Adapt { stream: adapt_stream, every: adapt_every, max: adapt_max, .. } = eng.cfg.adapt;
                // CROW_ADAPT_STREAM=1: the same trickle, but the copies run on a
                // side stream overlapping the next token (A-P3c); the tick's
                // host bookkeeping is the only part still inside the token time
                let mut trickle_swaps = 0usize;
                let mut tick_us = 0u128;
                crow_nest_engine::gen::stage_dma_reset();
                for i in 1..gen {
                    let t0 = std::time::Instant::now();
                    if adapt_stream && adapt_every > 0 {
                        trickle_swaps += eng.trickle_tick(i % adapt_every == 0, adapt_max);
                        tick_us += t0.elapsed().as_micros();
                    } else if adapt_every > 0 && i % adapt_every == 0 {
                        trickle_swaps += eng.adapt_tick(adapt_max);
                    }
                    if stopped_eos {
                        break;
                    }
                    next = eng.decode_step(&mut cnq, next as i64);
                    if sample_host {
                        if let Some(s) = &mut sampler {
                            let lg = crow_nest_engine::cuda::dtoh(eng.s.logits, V);
                            next = s.sample(&lg);
                            s.observe(next);
                        }
                    }
                    lat.push(t0.elapsed().as_secs_f64() * 1e3);
                    trace.push(next);
                    if stop_eos && crow_nest_engine::sample::EOS_IDS.contains(&next) {
                        stopped_eos = true;
                    }
                }
                if stopped_eos {
                    println!("stopped at EOS after {} generated tokens", trace.len());
                }
                if adapt_stream && adapt_every > 0 {
                    let drained = eng.trickle_drain();
                    println!("adapt trickle (stream): {trickle_swaps} swaps started over {} decode tokens (every {adapt_every}, max {adapt_max}/layer, {drained} total, tick host time {:.2} ms/token)",
                        gen - 1, tick_us as f64 / 1e3 / (gen - 1).max(1) as f64);
                } else if adapt_every > 0 {
                    println!("adapt trickle: {trickle_swaps} swaps over {} decode tokens (every {adapt_every}, max {adapt_max}/layer)", gen - 1);
                }
                let mean: f64 = lat.iter().sum::<f64>() / lat.len().max(1) as f64;
                lat.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let p50 = lat[lat.len() / 2];
                let c = eng.drain_counters();
                let (sel, cold): (u64, u64) = (
                    c.iter().zip(c0.iter()).map(|(x, b)| x[0] - b[0]).sum(),
                    c.iter().zip(c0.iter()).map(|(x, b)| x[1] - b[1]).sum(),
                );
                let gen_timed = (gen - 1).max(1) as f64;
                {
                    let (r, m) = (eng.ple.req - ple_r0, eng.ple.miss - ple_m0);
                    println!("ple rows per timed decode token: {:.1} requested, {:.2} misses ({:.1} %)",
                        r as f64 / gen_timed, m as f64 / gen_timed, 100.0 * m as f64 / r.max(1) as f64);
                }
                println!("cold experts per timed decode token: {:.1} of {:.0} selections -> {:.0} MB/token zero-copy",
                    cold as f64 / gen_timed, sel as f64 / gen_timed,
                    cold as f64 / gen_timed * (eng.res.gu_bytes + eng.res.dn_bytes) as f64 / 1e6);
                crow_nest_engine::gen::stage_dma_report(gen_timed as u64);
            if std::env::var("CROW_PROFILE").is_ok() {
                crow_nest_engine::gen::prof::kprof_report(gen as u64);
                crow_nest_engine::gen::prof::report();
            }
                println!(
                    "decode: mean {mean:.2} ms  p50 {p50:.2} ms  ({:.1} tok/s)  context {}  cold experts/token {:.1}",
                    1000.0 / mean,
                    eng.pos,
                    cold as f64 / (gen as f64)
                );
                println!("trace: {trace:?}");
                serde_json::to_writer(
                    std::fs::File::create("decode_out/run.json").unwrap(),
                    &serde_json::json!({
                        "prompt": ids.len(), "generated": trace.len(), "budget": gen, "stopped_eos": stopped_eos,
                        "prefill_s": prefill_s,
                        "prefill_tok_s": ids.len() as f64 / prefill_s,
                        "warmup_ms": warm, "mean_ms": mean, "p50_ms": p50,
                        "tok_s": 1000.0 / mean,
                        "context": eng.pos,
                        "sel_per_tok": sel as f64 / gen as f64,
                        "cold_per_tok": cold as f64 / gen as f64,
                        "trace": trace,
                        "kv": cfg.kv.name(),
                        "n_hot": eng.res.n,
                    }),
                )
                .unwrap();
            }
            "routestats" => {
                // decode routestats <ids.json> <out.json> [gen]: prefill the prompt,
                // snapshot the per-expert prefill selection counts, then decode `gen`
                // tokens (non-graph) logging the routed ids per layer. Measurement
                // A-V3: does a prompt's own routing predict its decode routing?
                let ids_path = args[2].clone();
                let out = args[3].clone();
                let gen: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(64);
                let ids = read_ids(&ids_path);
                cfg.context = CONTEXT_FLOOR;
                std::env::set_var("CROW_ROUTE_DUMP", "1");
                std::env::set_var("CROW_GRAPH", "0");
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| println!("[load] {m}"));
                let t0 = std::time::Instant::now();
                let mut next = eng.prefill(&mut cnq, &ids, None);
                println!("routestats: prefill {} tokens in {:.1} s", ids.len(), t0.elapsed().as_secs_f64());
                let pre_counts = eng.drain_sel_counts();
                let mut trace = vec![next as i64];
                for _ in 0..gen {
                    next = eng.decode_step(&mut cnq, next as i64);
                    trace.push(next as i64);
                }
                let routes: Vec<Vec<[i32; 10]>> = eng.route_log.clone();
                serde_json::to_writer(
                    std::fs::File::create(&out).unwrap(),
                    &serde_json::json!({
                        "prompt_tokens": ids.len(), "gen": gen, "n_hot": eng.res.n,
                        "hot_sets": eng.res.sets,
                        "prefill_counts": pre_counts,
                        "decode_routes": routes,
                        "trace": trace,
                    }),
                ).unwrap();
                println!("routestats: {} decode tokens logged -> {out}", routes.len());
            }
            "warmup" => {
                // decode warmup <ids.json> <out-sidecar.json> [N]: prefill REAL token
                // ids with an even (id-ordered) hot set, drain the per-expert
                // selection counts, persist top-N per layer in frequency order.
                // Never overwrites: the output path must not exist.
                let ids_path = args[2].clone();
                let out = args[3].clone();
                let n: usize = args.get(4).and_then(|v| v.parse().ok()).unwrap_or(160);
                assert!(!std::path::Path::new(&out).exists(), "warmup: {out} exists - refusing to overwrite");
                let ids = read_ids(&ids_path);
                cfg.context = CONTEXT_FLOOR;
                cfg.n_hot = n;
                let even: [[u64; E]; LAYERS] = [[1u64; E]; LAYERS];
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, Some(&even), &out, false, &mut |m| println!("[load] {m}"));
                println!("warmup: prefill over {} real tokens (chunk {}) …", ids.len(), cfg.prompt_chunk);
                let t0 = std::time::Instant::now();
                let _ = eng.prefill(&mut cnq, &ids, None);
                println!("warmup: prefill done in {:.1} s", t0.elapsed().as_secs_f64());
                let counts = eng.drain_sel_counts();
                let sets: Vec<Vec<u32>> = counts
                    .iter()
                    .map(|c| {
                        let mut ord: Vec<u32> = (0..E as u32).collect();
                        ord.sort_by(|&a, &b| c[b as usize].cmp(&c[a as usize]).then(a.cmp(&b)));
                        ord.truncate(n);
                        ord // frequency order (truncate-safe)
                    })
                    .collect();
                // coverage estimate on the warm-up traffic itself
                let mut hit = 0u64;
                let mut tot = 0u64;
                for (l, c) in counts.iter().enumerate() {
                    tot += c.iter().sum::<u64>();
                    hit += sets[l].iter().map(|&e| c[e as usize]).sum::<u64>();
                }
                let slabs = crow_nest_engine::residency::expert_slab_info(&cnq, 0, "text");
                crow_nest_engine::residency::persist_sidecar(
                    &out, n, &sets, &slabs,
                    &format!("decode warmup, {} real tokens from {}, top-{n}/layer, frequency order", ids.len(), ids_path),
                );
                println!("warmup: sidecar written -> {out}  (in-sample coverage {:.1} % of {} selections)", 100.0 * hit as f64 / tot.max(1) as f64, tot);
            }
            "reloadcheck" => {
                // #18: load, drop, load again in ONE process; report cuMemGetInfo
                // before/after each cycle (acceptance: delta < 64 MB, second load succeeds)
                let n: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(2);
                cfg.context = CONTEXT_FLOOR;
                if let Some(c) = std::env::var("CROW_CHUNK").ok().and_then(|v| v.parse::<usize>().ok()) {
                    cfg.prompt_chunk = c.max(1);
                }
                let f0 = crow_nest_engine::cuda::free_vram_bytes();
                println!("reloadcheck: free VRAM before any load {:.1} MB", f0 as f64 / 1e6);
                for i in 0..n {
                    let (mut eng, _rep) =
                        Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |_| {});
                    let ids = [760i64, 3841, 13477, 37550, 33075, 888, 279, 15217];
                    let mut next = eng.prefill(&mut cnq, &ids, None);
                    for _ in 0..3 {
                        next = eng.decode_step(&mut cnq, next as i64);
                    }
                    let f_loaded = crow_nest_engine::cuda::free_vram_bytes();
                    drop(eng);
                    crow_nest_engine::cuda::drop_dbg("after Engine dropped");
                    let f_after = crow_nest_engine::cuda::free_vram_bytes();
                    println!("reloadcheck cycle {i}: loaded {:.1} MB free, after drop {:.1} MB free, leak vs start {:.1} MB, last token {next}",
                        f_loaded as f64 / 1e6, f_after as f64 / 1e6, (f0 as f64 - f_after as f64) / 1e6);
                }
            }
            "layercheck" => {
                // layer-0 assembly check vs the p10 golden: load engine, feed
                // oracle/golden/layer0-input.f32, run layer 0, compare to
                // oracle/golden/layer0-golden-output.f32
                cfg.context = CONTEXT_FLOOR;
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| println!("[load] {m}"));
                let inp = std::fs::read("../oracle/golden/layer0-input.f32").unwrap();
                let gold = std::fs::read("../oracle/golden/layer0-golden-output.f32").unwrap();
                let t = 8usize;
                assert_eq!(inp.len(), t * HCT * 4);
                let to_f32 = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
                let x = to_f32(&inp);
                let g = to_f32(&gold);
                // the golden input is ALREADY the [T][10240] HC stream (p10 fed it as x0)
                let h_in: Vec<f32> = x;
                let out = eng.run_layer0_with_stage_dumps(&h_in, t, "../probes/engine-p8debug");
                let mut nan = 0usize;
                let mut max_abs = 0f32;
                for i in 0..t * HCT {
                    if out[i].is_nan() { nan += 1; }
                    max_abs = max_abs.max((out[i] - g[i]).abs());
                }
                println!("layercheck: max_abs={max_abs:.3e} NaN={nan} (p10 measured max_abs=0.125, rel_L2 1.73e-2)");
                println!("layercheck: max_abs precise = {max_abs:.6e}");
            }
            "layercheck3" => {
                // layer-3 (first FULL-ATTENTION) sub-block check vs the p7
                // golden: feed oracle/golden/layer3-attn-input.f32 (the [8][2560]
                // `mixed` input) into the SubW::Attn flow in isolation and
                // compare the [8][2560] o_proj output to layer3-attn-output.f32.
                // Reference marks (p16, all-proj FP4 vs this golden): rel_L2
                // 0.165, max_abs 0.582 — the "expected bad" FP4 attention delta.
                cfg.context = CONTEXT_FLOOR;
                let (mut eng, _rep) =
                    Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| println!("[load] {m}"));
                let to_f32 = |b: &[u8]| b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect::<Vec<f32>>();
                let inp = std::fs::read("../oracle/golden/layer3-attn-input.f32").unwrap();
                let gold = std::fs::read("../oracle/golden/layer3-attn-output.f32").unwrap();
                let t = 8usize;
                assert_eq!(inp.len(), t * H * 4, "golden input shape [8][2560] f32");
                assert_eq!(gold.len(), t * H * 4, "golden output shape [8][2560] f32");
                let x = to_f32(&inp);
                let g = to_f32(&gold);
                match &eng.w.sub[3] {
                    SubW::Attn { q, k, .. } => {
                        let d = |p: &PW| match p { PW::Fp4(..) => "fp4", PW::Bf16(_) => "bf16" };
                        println!("layercheck3: layer-3 q/k projection dtype = {}/{}", d(q), d(k));
                    }
                    _ => panic!("layer 3 is not attention"),
                }
                let out = eng.run_attn_subblock(3, &x, t, 0);
                let n = out.len();
                let mut nan = 0usize;
                let mut max_abs = 0f32;
                let mut sum_abs = 0f64;
                for i in 0..n {
                    if out[i].is_nan() { nan += 1; }
                    let a = (out[i] - g[i]).abs();
                    max_abs = max_abs.max(a);
                    sum_abs += a as f64;
                }
                // rel_L2 of the error vs the golden norm — near-zero golden
                // entries (|g| ≤ 1e-5) excluded from the ratio (p16 rule)
                let (mut se, mut sg, mut sen, mut sgn) = (0f64, 0f64, 0f64, 0f64);
                for i in 0..n {
                    let e = out[i] as f64 - g[i] as f64;
                    se += e * e;
                    sg += g[i] as f64 * g[i] as f64;
                    if g[i].abs() > 1e-5 {
                        sen += e * e;
                        sgn += g[i] as f64 * g[i] as f64;
                    }
                }
                let rel_all = se.sqrt() / sg.sqrt().max(1e-30);
                let rel_nz = sen.sqrt() / sgn.sqrt().max(1e-30);
                // Pearson correlation over ALL elements
                let nf = n as f64;
                let mo = out.iter().map(|&v| v as f64).sum::<f64>() / nf;
                let mg = g.iter().map(|&v| v as f64).sum::<f64>() / nf;
                let (mut cov, mut vo, mut vg) = (0f64, 0f64, 0f64);
                for i in 0..n {
                    let a = out[i] as f64 - mo;
                    let b = g[i] as f64 - mg;
                    cov += a * b;
                    vo += a * a;
                    vg += b * b;
                }
                let corr = cov / (vo.sqrt() * vg.sqrt()).max(1e-30);
                println!(
                    "layercheck3: max_abs={max_abs:.4} mean_abs={:.4} rel_L2={rel_all:.4} rel_L2(nz)={rel_nz:.4} corr={corr:.5} NaN={nan}",
                    sum_abs / nf
                );
                println!("layercheck3: p16 FP4 mark for this golden: rel_L2 0.165, max_abs 0.582");

                // stepwise pass: the same 8 rows ONE token at a time with
                // advancing pos (persistent KV + QSA state, single-row GEMV
                // path `gemv_bf16`/`gemv_fp4` — the shape free generation runs)
                let mut s_max = 0f32;
                let mut s_se = 0f64;
                let mut s_sg = 0f64;
                let mut s_nan = 0usize;
                for i in 0..t {
                    let row = eng.run_attn_subblock(3, &x[i * H..(i + 1) * H], 1, i);
                    let gold_row = &g[i * H..(i + 1) * H];
                    let mut r_max = 0f32;
                    for j in 0..H {
                        if row[j].is_nan() { s_nan += 1; }
                        let a = (row[j] - gold_row[j]).abs();
                        r_max = r_max.max(a);
                        let e = (row[j] - gold_row[j]) as f64;
                        s_se += e * e;
                        s_sg += gold_row[j] as f64 * gold_row[j] as f64;
                    }
                    s_max = s_max.max(r_max);
                }
                let s_rel = s_se.sqrt() / s_sg.sqrt().max(1e-30);
                println!(
                    "layercheck3 stepwise: max_abs={s_max:.4} rel_L2={s_rel:.4} NaN={s_nan} (batched==stepped pin, p11/p12 pattern)"
                );
            }
            _ => {
                println!("usage: decode parity <ids.json> <out> | decode run <ids.json> <gen> | decode layercheck | decode layercheck3");
            }
        }
    }
}
