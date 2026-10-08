//! #9 acceptance demo — three-state manager:
//!   * allocation plan (byte counts per state kind) at the 262k default AND
//!     the 200k floor, FP8-KV vs BF16-KV,
//!   * loader budget verify with measured allocations, auto-clamp demonstrated
//!     by forcing N=192,
//!   * refusal of configurations below the 200k context floor.
//! State allocations are REAL; expert/dense sizes come from the container
//! index (the decode binary does the full measured load).
//!
//! #159: `states --plan [--vram-mib N]` is the dry plan of a glm5_next checkpoint: it reads
//! the checkpoint's config.json + generation_config.json (`CROW_MODEL_DIR`, else
//! `models/GLM-5.3-Flash-original`), runs them through the metadata gate's glm5_next arm and
//! prints the three-tier plan (`manager::glm5_plan_table`). No CUDA context, no container
//! mapping, no allocation: the card is `--vram-mib` (default the RTX 5090's 32,607 MiB), the
//! pinned budget `CROW_PINNED_BUDGET_GB` or `HOST_PINNED_CAP`, the context `CROW_CONTEXT` or
//! the family floor, the dense part and the expert block the converter dry run's.

use crow_nest_engine::cuda;
use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::geo::*;
use crow_nest_engine::manager::{ThreeStates, StateSizes};

fn plan_table(geo: &Geo, context: usize, kv: KvDtype, expert_per_unit: u64, dense_hot: u64, n: usize) {
    let s = StateSizes::plan(geo, context, kv, 512);
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
        geo.layers,
        expert_per_unit as f64 / geo.layers as f64 / MIB
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

/// the RTX 5090 of record (`docs/system-landscape.md:12`): 32,607 MiB
const CARD_MIB_OF_RECORD: u64 = 32_607;

/// #159: the dry glm5_next plan; exits 1 with the refusal on any gate or planner refusal
fn plan_only(args: &[String]) {
    use crow_nest_engine::manager::{glm5_plan_table, plan_glm5_next};
    use crow_nest_engine::meta::ModelMeta;
    let fail = |why: String| -> ! {
        eprintln!("states --plan: {why}");
        std::process::exit(1)
    };
    let vram_mib = match args.iter().position(|a| a == "--vram-mib") {
        Some(i) => args.get(i + 1).and_then(|v| v.parse::<u64>().ok()).unwrap_or_else(|| fail("--vram-mib needs a whole number of MiB".into())),
        None => CARD_MIB_OF_RECORD,
    };
    let dir = std::env::var("CROW_MODEL_DIR").unwrap_or_else(|_| from_engine_dir("models/GLM-5.3-Flash-original"));
    let config = format!("{dir}/config.json");
    let generation = format!("{dir}/generation_config.json");
    let generation = std::path::Path::new(&generation).is_file().then_some(generation);
    let meta = ModelMeta::from_config_files(&config, generation.as_deref()).unwrap_or_else(|why| fail(why));
    if meta.family != Family::Glm5Next {
        fail(format!("{config}: family {:?}; --plan is the glm5_next dry plan (#159), `states` without it measures the families that run", meta.family));
    }
    let bad: Vec<String> = meta.verify().iter().map(|c| format!("  {}", c.line())).collect();
    if !bad.is_empty() {
        fail(format!("{} of {} constants differ from the glm5_next family row:
{}", bad.len(), meta.checks().len(), bad.join("
")));
    }
    let g = meta.glm5_geo().unwrap_or_else(|why| fail(why));
    let context = crow_nest_engine::boot::context_from_env(std::env::var("CROW_CONTEXT").ok().as_deref(), g.context_floor, g.context_max)
        .unwrap_or_else(|why| fail(why));
    let (pinned, pinned_src) = match env_parse::<u64>("CROW_PINNED_BUDGET_GB") {
        Some(gb) => (gb << 30, "CROW_PINNED_BUDGET_GB".to_string()),
        None => (HOST_PINNED_CAP, "HOST_PINNED_CAP; the boot takes min(cap, free RAM - CROW_RAM_MARGIN_GB 1 GiB), free RAM not read here".to_string()),
    };
    let (states, input, plan) = plan_glm5_next(
        &g, context, vram_mib << 20, pinned, GLM5_NEXT_DENSE_BYTES, GLM5_NEXT_EXPERT_BLOCK_BYTES,
        crow_nest_engine::gen::pf_tg(), crow_nest_engine::gen::pf_async_on(),
    )
    .unwrap_or_else(|why| fail(why));
    println!("states --plan: {config} ({} constants verified against {GLM5_NEXT_SOURCE}); no CUDA, no container read", meta.checks().len());
    let sources = [
        ("dense", "converter dry run, GLM measurement book".to_string()),
        ("expert", format!("converter dry run; config derives {} B", g.expert_block_bytes())),
        ("vram", if vram_mib == CARD_MIB_OF_RECORD { "RTX 5090 of record, docs/system-landscape.md:12; not free VRAM".to_string() } else { "--vram-mib".to_string() }),
        ("pinned", pinned_src),
    ];
    println!("{}", glm5_plan_table(&g, &states, &input, &plan, &sources));
}

fn main() {
    // #159: the dry plan touches no GPU and no container, so it runs before the log and CUDA
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--plan") {
        plan_only(&args);
        return;
    }
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
    // Crow #300 C3: the model's Geo through the same metadata gate the front door uses
    let geo = crow_nest_engine::boot::model_geo(&cnq_path);
    let mut cnq = Cnq::open(&cnq_path);
    let slabs = crow_nest_engine::residency::expert_slab_info(&mut cnq, 0, "text", geo.moe().experts);
    let expert_per_unit = (slabs.gu_bytes + slabs.dn_bytes) * geo.layers as u64;
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

    plan_table(&geo, 262_144, KvDtype::Fp8E4m3, expert_per_unit, dense_bytes, 160);
    plan_table(&geo, 262_144, KvDtype::Bf16, expert_per_unit, dense_bytes, 160);
    plan_table(&geo, 200_000, KvDtype::Fp8E4m3, expert_per_unit, dense_bytes, 160);
    plan_table(&geo, 200_000, KvDtype::Bf16, expert_per_unit, dense_bytes, 160);

    unsafe {
        let _ctx = cuda::Ctx::init();
        let cfg = Config::default();
        let (_, rep) = ThreeStates::allocate(&cfg, &geo, dense_bytes, expert_per_unit, expert_per_unit, false, 0);
        println!("\n=== measured load @262k FP8-KV (default) ===");
        for l in &rep.lines {
            println!("{l}");
        }
        let cfg192 = Config { n_hot: 192, ..Default::default() };
        let (_, rep) = ThreeStates::allocate(&cfg192, &geo, dense_bytes, expert_per_unit, expert_per_unit, false, 0);
        println!("\n=== forced N=192 @262k (auto-clamp expected, spec 2.6) ===");
        for l in &rep.lines {
            println!("{l}");
        }
        let cfgbf = Config { context: 200_000, kv: KvDtype::Bf16, ..Default::default() };
        let (_, rep) = ThreeStates::allocate(&cfgbf, &geo, dense_bytes, expert_per_unit, expert_per_unit, false, 0);
        println!("\n=== fallback point 200k BF16-KV ===");
        for l in &rep.lines {
            println!("{l}");
        }
        println!("\n=== refusing 150k context (must refuse) ===");
        let cfgbad = Config { context: 150_000, ..Default::default() };
        let r = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            ThreeStates::allocate(&cfgbad, &geo, dense_bytes, expert_per_unit, expert_per_unit, false, 0)
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
