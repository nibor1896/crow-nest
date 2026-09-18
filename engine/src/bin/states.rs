//! #9 acceptance demo — three-state manager:
//!   * allocation plan (byte counts per state kind) at the 262k default AND
//!     the 200k floor, FP8-KV vs BF16-KV,
//!   * loader budget verify with measured allocations, auto-clamp demonstrated
//!     by forcing N=192,
//!   * refusal of configurations below the 200k context floor.
//! State allocations are REAL; expert/dense sizes come from the container
//! index (the decode binary does the full measured load).

use crow_nest_engine::cuda;
use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::geo::*;
use crow_nest_engine::manager::{ThreeStates, StateSizes};

fn plan_table(context: usize, kv: KvDtype, expert_per_unit: u64, dense_hot: u64, n: usize) {
    let s = StateSizes::plan(context, kv, 512);
    println!("--- plan: context {context}, KV {}, N={n} ---", kv.name());
    println!("KV cache      {:>10.1} MiB", s.kv_bytes as f64 / MIB);
    println!("QSA keys      {:>10.1} MiB", s.qsa_keys_bytes as f64 / MIB);
    println!("QSA pooled    {:>10.1} MiB", s.qsa_pooled_bytes as f64 / MIB);
    println!("GDN state     {:>10.1} MiB", (s.gdn_s_bytes + s.gdn_conv_bytes) as f64 / MIB);
    println!("RoPE tables   {:>10.1} MiB", s.rope_bytes as f64 / MIB);
    println!("dense (index) {:>10.1} MiB", dense_hot as f64 / MIB);
    println!(
        "hot experts   {:>10.1} MiB  ({} x {} layers x {:.2} MiB)",
        n as f64 * expert_per_unit as f64 / MIB,
        n,
        LAYERS,
        expert_per_unit as f64 / LAYERS as f64 / MIB
    );
    let total = s.kv_bytes
        + s.qsa_keys_bytes
        + s.qsa_pooled_bytes
        + s.gdn_s_bytes
        + s.gdn_conv_bytes
        + s.rope_bytes
        + dense_hot
        + n as u64 * expert_per_unit;
    println!("TOTAL         {:>10.2} GiB  of 32.00 GiB VRAM", total as f64 / GIB);
}

fn main() {
    // #13: the logging subscriber of this process. Every library line this bin
    // triggers (`[prefill]`, `[load]`, `[budget]`, `[ple]`, ...) is a `tracing`
    // event now, so without this call they go nowhere. The guard drains the two
    // writer threads when `main` returns; an `exit` below calls `shutdown` first.
    let _log = crow_nest_engine::log::init();
    // #60 (2026-09-18): the demo defaults to the production -M container, like
    // `decode` and `parity` (#51) and the two generator bins (#52), and reads
    // CROW_CNQ like they do. It used to hard-code the pre-#51 container, so the
    // byte counts it printed were the geometry of a file that is no longer of
    // record. Read only: it opens the container INDEX, never a weight.
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| from_engine_dir(DEFAULT_CNQ));
    println!("states: opening container index for expert geometry … ({cnq_path})");
    let mut cnq = Cnq::open(&cnq_path);
    let slabs = crow_nest_engine::residency::expert_slab_info(&mut cnq, 0, "text");
    let expert_per_unit = (slabs.gu_bytes + slabs.dn_bytes) * LAYERS as u64;
    println!(
        "expert slab: gate_up {:.2} MiB + down {:.2} MiB = {:.2} MiB per expert per layer (gs {:.3}/{:.3})",
        slabs.gu_bytes as f64 / MIB,
        slabs.dn_bytes as f64 / MIB,
        (slabs.gu_bytes + slabs.dn_bytes) as f64 / MIB,
        slabs.gu_gs,
        slabs.dn_gs
    );
    let mut dense_bytes = 0u64;
    for t in &cnq.tensors {
        if t.section != "text" {
            continue;
        }
        let is_expert = t.name.contains("mlp.experts.");
        if !is_expert && t.name != "model.language_model.embed_tokens.weight" {
            dense_bytes += Cnq::byte_len(t);
        }
    }
    println!(
        "dense (index): {:.1} MiB non-expert, non-embedding (lm_head BF16 included)",
        dense_bytes as f64 / MIB
    );

    plan_table(262_144, KvDtype::Fp8E4m3, expert_per_unit, dense_bytes, 160);
    plan_table(262_144, KvDtype::Bf16, expert_per_unit, dense_bytes, 160);
    plan_table(200_000, KvDtype::Fp8E4m3, expert_per_unit, dense_bytes, 160);
    plan_table(200_000, KvDtype::Bf16, expert_per_unit, dense_bytes, 160);

    unsafe {
        let _ctx = cuda::Ctx::init();
        let cfg = Config::default();
        let (_, rep) = ThreeStates::allocate(&cfg, dense_bytes, expert_per_unit, expert_per_unit, false);
        println!("\n=== measured load @262k FP8-KV (default) ===");
        for l in &rep.lines {
            println!("{l}");
        }
        let cfg192 = Config { n_hot: 192, ..Default::default() };
        let (_, rep) = ThreeStates::allocate(&cfg192, dense_bytes, expert_per_unit, expert_per_unit, false);
        println!("\n=== forced N=192 @262k (auto-clamp expected, spec 2.6) ===");
        for l in &rep.lines {
            println!("{l}");
        }
        let cfgbf = Config { context: 200_000, kv: KvDtype::Bf16, ..Default::default() };
        let (_, rep) = ThreeStates::allocate(&cfgbf, dense_bytes, expert_per_unit, expert_per_unit, false);
        println!("\n=== fallback point 200k BF16-KV ===");
        for l in &rep.lines {
            println!("{l}");
        }
        println!("\n=== refusing 150k context (must refuse) ===");
        let cfgbad = Config { context: 150_000, ..Default::default() };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ThreeStates::allocate(&cfgbad, dense_bytes, expert_per_unit, expert_per_unit, false)
        }));
        match r {
            Ok(_) => {
                eprintln!("states: FAIL — 150k context was not refused");
                crow_nest_engine::log::shutdown();
                std::process::exit(1);
            }
            Err(_) => println!("refused as required (spec 0.2 floor)"),
        }
    }
    println!("\nstates: DONE");
}
