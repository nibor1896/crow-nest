//! #8 acceptance demo — residency scheduler on demo Crow-like traffic:
//!   1. warm-up: prefill traffic through the full engine, per-expert GPU
//!      selection counts drained once per chunk (spec 2.2 — every routed
//!      choice counts), top-N promoted per layer, sidecar persisted.
//!   2. reload from the sidecar, held-out measurement pass: prefill + decode
//!      steps with a per-token report of selections / cold experts /
//!      cold bytes / layers fully resident (routing-gated skip, §3.3/§3.6).
//!   3. coverage of the persisted sets on held-out traffic (vs the #3 finding
//!      that per-layer top-N ≥ global top-N).
//!
//! Demo traffic is tokenized with a stable word hash (NOT the model
//! tokenizer) — it only exercises router locality; parity runs use the real
//! tokenizer via tools/tokenize.py.

use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::geo::*;
use crow_nest_engine::gen::Engine;

const WARMUP: &str = include_str!("warmup_text.txt");
const HOLDOUT: &str = include_str!("holdout_text.txt");

fn demo_tokenize(text: &str, max_tokens: usize) -> Vec<i64> {
    let mut out = Vec::new();
    for w in text.split_whitespace() {
        let h: u64 = w
            .bytes()
            .fold(2166136261u64, |a, b| (a ^ b as u64).wrapping_mul(16777619));
        out.push((100 + h % 199_900) as i64);
        if out.len() >= max_tokens {
            break;
        }
    }
    out
}

fn totals(c: &[[u64; 2]]) -> (u64, u64) {
    (c.iter().map(|x| x[0]).sum(), c.iter().map(|x| x[1]).sum())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let warm_tokens: usize = args.get(1).and_then(|v| v.parse().ok()).unwrap_or(384);
    let n_requested: usize = args.get(2).and_then(|v| v.parse().ok()).unwrap_or(160);

    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| "../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq".into());
    let sidecar = format!("{cnq_path}.hotsets.json");
    let mut cnq = Cnq::open(&cnq_path);

    unsafe {
        let _ctx = crow_nest_engine::cuda::Ctx::init();
        let mut cfg = Config::default();
        cfg.context = CONTEXT_FLOOR;
        cfg.n_hot = n_requested;

        if std::path::Path::new(&sidecar).exists() {
            println!("residency: sidecar exists — skipping warm-up ({sidecar})");
        } else {
            println!("residency: warm-up phase on {warm_tokens} demo tokens …");
            let even: [[u64; E]; LAYERS] = [[1u64; E]; LAYERS];
            let (mut eng0, _) = Engine::load(&mut cnq, cfg, Some(&even), &sidecar, false, &mut |m| {
                eprintln!("[load0] {m}");
            });
            let warm_ids = demo_tokenize(WARMUP, warm_tokens);
            let t0 = std::time::Instant::now();
            let _ = eng0.prefill(&mut cnq, &warm_ids, None);
            println!(
                "warm-up prefill done in {:.1} s — promoting top-{} per layer",
                t0.elapsed().as_secs_f64(),
                cfg.n_hot
            );
            let counts = eng0.drain_sel_counts();
            let sets: Vec<Vec<u32>> = counts
                .iter()
                .map(|c| {
                    let mut ord: Vec<u32> = (0..E as u32).collect();
                    ord.sort_by(|&a, &b| c[b as usize].cmp(&c[a as usize]).then(a.cmp(&b)));
                    ord.truncate(cfg.n_hot);
                    // keep FREQUENCY order (matches the residency.rs writer fix
                    // 2026-09-03) — an ID re-sort here would make a later
                    // N-truncate keep the lowest ids instead of the hottest
                    ord
                })
                .collect();
            let slabs = crow_nest_engine::residency::expert_slab_info(&cnq, 0, "text");
            crow_nest_engine::residency::persist_sidecar(
                &sidecar,
                cfg.n_hot,
                &sets,
                &slabs,
                &format!("engine warm-up, {warm_tokens} demo tokens, {n_requested}/layer"),
            );
            println!("sidecar persisted → reloading with real hot sets …");
            drop(eng0);
        }

        let (mut eng, _rep) =
            Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| eprintln!("[load] {m}"));
        println!(
            "residency ready: N={} ({}) — pinned cold tier {:.2} GiB",
            eng.res.n,
            eng.res.source,
            eng.res.pinned_bytes() as f64 / (1 << 30) as f64
        );

        // ---- held-out measurement pass ----
        let hold_ids = demo_tokenize(HOLDOUT, 256);
        println!("\nholdout prefill: {} tokens …", hold_ids.len());
        let t0 = std::time::Instant::now();
        let tok = eng.prefill(&mut cnq, &hold_ids, None);
        println!(
            "holdout prefill done in {:.2} s ({:.0} tok/s) → greedy token {tok}",
            t0.elapsed().as_secs_f64(),
            hold_ids.len() as f64 / t0.elapsed().as_secs_f64()
        );

        let steps = 8;
        let mut next = tok;
        let mut cum = (0u64, 0u64);
        let mut prev = eng.drain_counters();
        println!("\n=== per-token report (held-out decode, spec 3.6) ===");
        for i in 0..steps {
            let t0 = std::time::Instant::now();
            next = eng.decode_step(&mut cnq, next as i64);
            let dt = t0.elapsed().as_secs_f64() * 1e3;
            let c = eng.drain_counters();
            let sel: u64 = c.iter().zip(prev.iter()).map(|(x, p)| x[0] - p[0]).sum();
            let cold: u64 = c.iter().zip(prev.iter()).map(|(x, p)| x[1] - p[1]).sum();
            let layers_cold = c.iter().zip(prev.iter()).filter(|(x, p)| x[1] > p[1]).count();
            let bytes = cold as f64 * (eng.res.gu_bytes + eng.res.dn_bytes) as f64 / (1 << 20) as f64;
            println!(
                "decode {i}: {dt:7.2} ms  selections {sel:3}  cold {cold:3}  cold-bytes {bytes:6.1} MB  layers fully resident {}/{}",
                LAYERS - layers_cold,
                LAYERS
            );
            cum.0 += sel;
            cum.1 += cold;
            prev = c;
        }
        let bytes = cum.1 as f64 * (eng.res.gu_bytes + eng.res.dn_bytes) as f64 / (1 << 20) as f64;
        println!(
            "\ntoken means: selections {:.0}  cold {:.1}  cold-bytes/token ~{:.1} MB (zero-copy, spec 3.4)",
            cum.0 as f64 / steps as f64,
            cum.1 as f64 / steps as f64,
            bytes / steps as f64
        );

        // coverage of the hot sets on held-out routing
        let counts = eng.drain_sel_counts();
        let total: u64 = counts.iter().flatten().sum();
        let hot: u64 = counts
            .iter()
            .zip(eng.res.sets.iter())
            .map(|(c, set)| set.iter().map(|&id| c[id as usize]).sum::<u64>())
            .sum();
        println!(
            "\nhot-set coverage on held-out traffic: {}/{} = {:.1} %  (#3: per-layer top-N ≥ global)",
            hot,
            total,
            100.0 * hot as f64 / total.max(1) as f64
        );
        println!("residency: DONE");
    }
}
