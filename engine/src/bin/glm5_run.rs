//! #175 + #149 (GLM-5.3-Flash plan steps 16-17): greedy generation on the glm5_next container
//! with the routed experts in three tiers (VRAM / pinned / NVMe), all 45 layers resident, token
//! by token (`crow_nest_engine::glm5_tiers`).
//!
//! ```text
//! glm5_run [-n N] [--cnq PATH] [--ids a,b,c | --prompt TEXT --tokenizer tokenizer.json]
//!          [--tokenizer tokenizer.json] [--vram-slots N] [--pinned-slots N] [--readers N]
//! ```
//!
//! - container: `--cnq`, else `CROW_CNQ`, else `converter/GLM-5.3-Flash-MUL1K3.cnq` (relative
//!   to the repo root, run from `engine/`); refused by name unless it passes the glm5_next family
//!   row with MUL1 records (the checks of `decode glmgolden`)
//! - prompt: `--ids`, or `--prompt` through `--tokenizer` (one user message, generation prompt),
//!   else the tokenizer golden `sys_user_default` (38 ids); `--tokenizer` also decodes the output
//! - tiers per MoE layer: the #159 plan at this card's free VRAM, `CROW_CONTEXT` (the family
//!   floor 200,000 when unset) and the derived pinned budget (`HOST_PINNED_CAP` 46 GiB or less,
//!   free RAM - `CROW_RAM_MARGIN_GB`); `--vram-slots` / `--pinned-slots` may only ask for less
//! - NVMe readers: `--readers`, default 1 (PREREG amendment 5)
//!
//! Prints the plan, then one line per row: the row, the greedy id, seconds, the NVMe reads of
//! the row and the `[vram pinned nvme]` accesses of every MoE layer; then the totals. The
//! seconds are not a speed figure (every MoE layer synchronizes for its routing).
use crow_nest_engine::cuda;
use crow_nest_engine::geo::{from_engine_dir, GLM5_NEXT_DENSE_BYTES, HOST_PINNED_CAP};
use crow_nest_engine::glm5_model::GLM5_MUL1K3_CNQ;
use crow_nest_engine::glm5_tiers::{self as gt, ExpertTiers, Glm5Run, TokenReport};
use crow_nest_engine::manager::{derive_host_pinned_budget, plan_glm5_next};

fn main() {
    let _log = crow_nest_engine::log::init();
    let args: Vec<String> = std::env::args().collect();
    let code = match run(&args) {
        Ok(()) => 0,
        Err(e) => {
            eprintln!("[glm5_run] refused: {e}");
            1
        }
    };
    crow_nest_engine::log::shutdown();
    unsafe { cuda::ctx_hard_reset() };
    std::process::exit(code);
}

fn run(args: &[String]) -> Result<(), String> {
    let flag = |n: &str| -> Result<Option<String>, String> {
        match args.iter().position(|a| a == n) {
            None => Ok(None),
            Some(i) => args.get(i + 1).cloned().map(Some).ok_or_else(|| format!("{n} needs a value")),
        }
    };
    let num = |n: &str| -> Result<Option<usize>, String> { flag(n)?.map(|v| v.parse::<usize>().map_err(|_| format!("{n} {v:?} is not a whole number"))).transpose() };
    let n = num("-n")?.unwrap_or(16);
    let readers = num("--readers")?.unwrap_or(1);
    let path = flag("--cnq")?.or_else(|| std::env::var("CROW_CNQ").ok()).unwrap_or_else(|| from_engine_dir(GLM5_MUL1K3_CNQ));
    let tok = match flag("--tokenizer")? {
        Some(t) => {
            let cfg = crow_nest_engine::tokenizer::sibling_config(&t);
            Some(crow_nest_engine::tokenizer::ChatTokenizer::load(&t, &cfg)?)
        }
        None => None,
    };
    let prompt: Vec<i64> = match (flag("--ids")?, flag("--prompt")?) {
        (Some(_), Some(_)) => return Err("give the prompt once: --ids or --prompt".into()),
        (Some(ids), None) => ids.split(',').map(|v| v.trim().parse::<i64>().map_err(|_| format!("--ids: {v:?} is not a token id"))).collect::<Result<_, _>>()?,
        (None, Some(text)) => {
            let t = tok.as_ref().ok_or("--prompt needs --tokenizer <tokenizer.json>")?;
            t.encode_chat_user(&text)?.into_iter().map(i64::from).collect()
        }
        (None, None) => gt::fixed_prompt(),
    };
    let mut o = gt::open_container(&path)?;
    println!(
        "[glm5_run] container {path}: {} constants verified, routed experts {} x {} B ({} records, codec {}); prompt {} ids, generating {n}",
        o.constants,
        o.g.experts,
        o.spec.bytes,
        o.records,
        o.spec.codec.dtype(),
        prompt.len()
    );
    let context = crow_nest_engine::boot::context_from_env(std::env::var("CROW_CONTEXT").ok().as_deref(), o.g.context_floor, o.g.context_max)?;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let free = cuda::free_vram_bytes();
        let budget = derive_host_pinned_budget(HOST_PINNED_CAP, &mut |s| println!("{s}"));
        let (_, _, plan) = plan_glm5_next(&o.g, context, free, budget, GLM5_NEXT_DENSE_BYTES, o.spec.bytes, crow_nest_engine::gen::pf_tg(), crow_nest_engine::gen::pf_async_on())?;
        let sizes = gt::tier_sizes(&plan, num("--vram-slots")?, num("--pinned-slots")?)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        println!(
            "[glm5_run] #159 plan at free VRAM {:.2} GiB, context {context}, pinned budget {:.2} GiB: per MoE layer VRAM {} / pinned {} / NVMe {}; running VRAM {} / pinned {} / NVMe {} x {} MoE layers, NVMe readers {readers}",
            gib(free),
            gib(budget),
            plan.hot,
            plan.pinned,
            plan.nvme,
            sizes.vram,
            sizes.pinned,
            o.g.experts - sizes.vram - sizes.pinned,
            gt::moe_layers(&o.g)
        );
        let cap = prompt.len() + n;
        let mut run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, cap, &mut |s| println!("{s}"));
        let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, readers, o.g.topk)?;
        println!(
            "[glm5_run] tiers: {:.2} GiB VRAM (slots, {} staging, tables), {:.2} GiB pinned; free VRAM now {:.2} GiB; policy {:?}, cache empty at start",
            gib(tiers.vram_bytes()),
            tiers.stage_cap,
            gib(tiers.pinned_bytes()),
            gib(cuda::free_vram_bytes()),
            tiers.cache.policy
        );
        let first_moe = tiers.first_moe;
        let mut report = |r: &TokenReport| {
            let per: Vec<String> = r.tiers.iter().enumerate().map(|(i, c)| format!("l{} {}/{}/{}", first_moe + i, c[0], c[1], c[2])).collect();
            let sum = r.tiers.iter().fold([0u64; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
            let id = r.next.map_or("-".to_string(), |v| v.to_string());
            println!(
                "glm5_run row {:>4} {} next {id:>6}  {:.2} s  NVMe reads {:>3} ({:.1} MB)  tiers v/p/n {}/{}/{}  [{}]",
                r.pos,
                if r.prompt { "prompt" } else { "gen   " },
                r.secs,
                r.nvme_reads,
                r.nvme_bytes as f64 / 1e6,
                sum[0],
                sum[1],
                sum[2],
                per.join(" ")
            );
        };
        let t0 = std::time::Instant::now();
        let out = run.generate(&mut o.cnq, &mut tiers, &prompt, n, false, &mut report)?;
        let secs = t0.elapsed().as_secs_f64();
        let tot = tiers.cache.counters().iter().fold([0u64; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
        let all = (tot[0] + tot[1] + tot[2]).max(1) as f64;
        println!("glm5_run ids {:?}", out.ids);
        if let Some(t) = &tok {
            let ids: Vec<u32> = out.ids.iter().map(|&v| v as u32).collect();
            println!("glm5_run text {:?}", t.decode(&ids)?);
        }
        println!(
            "glm5_run totals: {} rows in {secs:.1} s; accesses vram {} ({:.1} %) pinned {} ({:.1} %) nvme {} ({:.1} %); NVMe reads {} = {:.2} GB, {:.1} per row",
            prompt.len() + n - 1,
            tot[0],
            100.0 * tot[0] as f64 / all,
            tot[1],
            100.0 * tot[1] as f64 / all,
            tot[2],
            100.0 * tot[2] as f64 / all,
            tiers.nvme_reads,
            tiers.nvme_bytes as f64 / 1e9,
            tiers.nvme_reads as f64 / (prompt.len() + n - 1) as f64
        );
        tiers.free();
        run.free();
    }
    Ok(())
}
