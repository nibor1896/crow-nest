//! #175 + #149 (GLM-5.3-Flash plan steps 16-17): greedy generation on the glm5_next container
//! with the routed experts in three tiers (VRAM / pinned / NVMe), all 45 layers resident, token
//! by token (`crow_nest_engine::glm5_tiers`). #187: the lever A/B instrument: prefill and decode
//! timed and counted apart, repetitions, prompt length, machine state, JSON.
//!
//! ```text
//! glm5_run [-n N] [--cnq PATH] [--ids a,b,c | --prompt TEXT --tokenizer tokenizer.json | --prompt-ids PATH]
//!          [--prompt-tokens N] [--reps N] [--cold] [--json PATH]   (env: CROW_CHUNK=N, #186)
//!          [--tokenizer tokenizer.json] [--vram-slots N] [--pinned-slots N] [--readers N]
//! ```
//!
//! - container: `--cnq`, else `CROW_CNQ`, else `converter/GLM-5.3-Flash-MUL1K3.cnq` (relative
//!   to the repo root, run from `engine/`); refused by name unless it passes the glm5_next family
//!   row with MUL1 records (the checks of `decode glmgolden`)
//! - prompt: `--ids`, `--prompt` through `--tokenizer` (one user message, generation prompt), or
//!   `--prompt-ids` (a file of ids separated by commas, blanks or newlines; `[ ]` allowed), else
//!   the tokenizer golden `sys_user_default` (31 ids); `--prompt-tokens N` repeats that prompt
//!   cyclically and cuts it to N ids (synthetic, deterministic); `--tokenizer` also decodes the
//!   output
//! - `--reps N` (default 1): `generate` N times on the one loaded model. Rep 1 starts with the
//!   cache empty (cold); later reps start with the cache the previous rep left (warm), unless
//!   `--cold`, which empties it before every rep (`ExpertTiers::reset_cache`, no allocation)
//! - tiers per MoE layer: the #159 plan at this card's free VRAM, `CROW_CONTEXT` (the family
//!   floor 200,000 when unset) and the derived pinned budget (`HOST_PINNED_CAP` 46 GiB or less,
//!   free RAM - `CROW_RAM_MARGIN_GB`); `--vram-slots` / `--pinned-slots` may only ask for less
//! - NVMe readers: `--readers`, default 1 (PREREG amendment 5)
//! - #186 prompt chunk: `CROW_CHUNK=N` runs the prompt in prompt calls of up to N rows
//!   (`Glm5Run::prefill`, one routing sync per MoE layer per call, the prefill staging set of
//!   `glm5_tiers::prefill_stage_slots` slots); unset or 1 = every prompt row one decode call. The
//!   plan books the chunk (`manager::plan_glm5_next_chunk`). A prompt call reports once (its
//!   rows, routing syncs and sub-batches); its seconds are the call's, not a row's
//! - `--json PATH`: every rep, both phases, per MoE layer, args, env, machine state (rewritten
//!   after every rep)
//! - #192 `CROW_GLM_MTP=N` (1..=4): speculative decode with the MTP block (the container's
//!   section `mtp` + `CROW_GLM_MTP_OVERLAY`, default `converter/GLM-5.3-Flash-MTP-overlay.cnq`);
//!   the plan runs at free VRAM minus the block's derived bytes, the tiers get `(1 + N) x top-k`
//!   staging slots, every rep prints and records the MTP counters per decode token. With
//!   `CROW_CHUNK` the prompt runs in prompt calls and the block follows each call. A step's
//!   clock is shared by the ids it emits (`secs` = step / ids), its tier counters sit on its
//!   first id.
//!
//! Prints the plan, one line per row (row, phase, rep, greedy id, seconds, NVMe reads, the
//! `[vram pinned nvme]` accesses of every MoE layer), then per rep the prefill and decode lines,
//! their counters, the per-layer spread and the machine state, then the summary over the reps.
//!
//! Clocks (no sync added, `docs/glm5-model.md` 6.1): the row seconds are `TokenReport::secs`,
//! read inside `Glm5Run::generate` from the row's start (before the embedding) to after the
//! row's closing `cuda::sync()` and the greedy id's `dtoh`; TTFT and the phase wall times are
//! read here, on entry of the report callback, which `generate` calls after that sync. Every
//! MoE layer synchronizes for its routing (`glm5_model.rs`, `call_with_experts`), so these are
//! the times of this synchronous path, not of a graph-captured one.
use crow_nest_engine::cuda;
use crow_nest_engine::geo::{from_engine_dir, HOST_PINNED_CAP};
use crow_nest_engine::glm5_flags::PrefetchStats;
use crow_nest_engine::glm5_model::GLM5_MUL1K3_CNQ;
use crow_nest_engine::glm5_mtp::{self as mtp, SpecStats};
use crow_nest_engine::glm5_tiers::{self as gt, ExpertTiers, Glm5Run, Moves, TokenReport};
use crow_nest_engine::manager::derive_host_pinned_budget;
use serde_json::{json, Value};

/// the spread rule of the gates (`runs/glm53-flash/PREREG.md`): max / min over repetitions
const SPREAD_RULE: f64 = 1.15;

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

// ---------------------------------------------------------------- prompt

/// ids of a `--prompt-ids` file: separated by commas, blanks or newlines, `[` `]` ignored
fn parse_ids(text: &str) -> Result<Vec<i64>, String> {
    let ids: Vec<i64> = text
        .split(|c: char| c == ',' || c == '[' || c == ']' || c.is_whitespace())
        .filter(|v| !v.is_empty())
        .map(|v| v.parse::<i64>().map_err(|_| format!("--prompt-ids: {v:?} is not a token id")))
        .collect::<Result<_, _>>()?;
    if ids.is_empty() {
        return Err("--prompt-ids: the file holds no id".into());
    }
    Ok(ids)
}

/// `--prompt-tokens N`: `base` repeated cyclically and cut to `n` ids
fn prompt_of_len(base: &[i64], n: usize) -> Result<Vec<i64>, String> {
    if base.is_empty() || n == 0 {
        return Err("--prompt-tokens needs a positive count and a non-empty base prompt".into());
    }
    Ok(base.iter().copied().cycle().take(n).collect())
}

// ---------------------------------------------------------------- statistics

/// one row as the report callback saw it; `at` = seconds since the `generate` call, read on
/// entry of the callback
#[derive(Clone, Debug)]
struct Row {
    r: TokenReport,
    at: f64,
    /// #188: seconds the CPU lane's pool runs took in this row (`ExpertTiers::cpu_lane_clock`)
    lane_s: f64,
    /// `CROW_GLM_PREFETCH`: the store's counters since the previous row (`None`: the store is off)
    pf: Option<PrefetchStats>,
}

fn median(v: &[f64]) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let n = s.len();
    Some(if n % 2 == 1 { s[n / 2] } else { (s[n / 2 - 1] + s[n / 2]) / 2.0 })
}

/// nearest rank: the smallest value with at least `p` percent of the values at or below it
fn percentile(v: &[f64], p: f64) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    let mut s = v.to_vec();
    s.sort_by(|a, b| a.total_cmp(b));
    let rank = ((p / 100.0) * s.len() as f64).ceil().max(1.0) as usize;
    Some(s[rank.min(s.len()) - 1])
}

/// max / min (None for fewer than two values or a non-positive minimum)
fn spread(v: &[f64]) -> Option<f64> {
    if v.len() < 2 {
        return None;
    }
    let min = v.iter().copied().fold(f64::INFINITY, f64::min);
    let max = v.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    (min > 0.0).then(|| max / min)
}

/// one phase of one rep: timings and the counters summed over its rows
#[derive(Clone, Debug, Default)]
struct Phase {
    tokens: usize,
    /// seconds of the phase on the callback clock: prefill = TTFT, decode = last row - TTFT
    wall: f64,
    /// `TokenReport::secs` of every row of the phase
    lat: Vec<f64>,
    /// `[vram, pinned, nvme]` accesses per MoE layer
    tiers: Vec<[u64; 3]>,
    moves: Vec<Moves>,
    nvme_reads: u64,
    nvme_bytes: u64,
    /// #188: CPU lane pool-run seconds summed over the rows
    lane_s: f64,
    /// `CROW_GLM_PREFETCH`: the store's counters summed over the rows (`None`: off)
    pf: Option<PrefetchStats>,
    /// #186: reports (one per row, or one per prompt call), routing syncs, serve sub-batches
    reports: usize,
    routing_syncs: u64,
    sub_batches: u64,
}

impl Phase {
    fn of(rows: &[&Row], layers: usize, wall: f64) -> Phase {
        let tokens = rows.iter().map(|r| r.r.rows.max(1)).sum();
        let mut p = Phase { tokens, wall, reports: rows.len(), tiers: vec![[0; 3]; layers], moves: vec![Moves::default(); layers], ..Phase::default() };
        for row in rows {
            p.routing_syncs += row.r.routing_syncs;
            p.sub_batches += row.r.sub_batches;
            p.lat.push(row.r.secs);
            p.nvme_reads += row.r.nvme_reads;
            p.nvme_bytes += row.r.nvme_bytes;
            p.lane_s += row.lane_s;
            if let Some(d) = row.pf {
                p.pf.get_or_insert_with(PrefetchStats::default).add(&d);
            }
            for (a, b) in p.tiers.iter_mut().zip(&row.r.tiers) {
                for i in 0..3 {
                    a[i] += b[i];
                }
            }
            for (a, b) in p.moves.iter_mut().zip(&row.r.moves) {
                a.add(b);
            }
        }
        p
    }

    fn total_tiers(&self) -> [u64; 3] {
        self.tiers.iter().fold([0; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]])
    }

    fn total_moves(&self) -> Moves {
        let mut m = Moves::default();
        for x in &self.moves {
            m.add(x);
        }
        m
    }

    /// tokens per second of the phase on its wall clock
    fn rate(&self) -> Option<f64> {
        (self.tokens > 0 && self.wall > 0.0).then(|| self.tokens as f64 / self.wall)
    }

    /// per-token rates `1 / secs` of the rows
    fn token_rates(&self) -> Vec<f64> {
        self.lat.iter().filter(|&&s| s > 0.0).map(|s| 1.0 / s).collect()
    }
}

/// Split the reports of one `generate` into prefill (the prompt rows or prompt calls; the last one
/// yields the first id) and decode (every later row, one generated id each). TTFT = the callback
/// clock of the last prompt report; the decode wall = the last row's minus TTFT.
fn phases(rows: &[Row], layers: usize) -> (Phase, Phase, f64) {
    let ttft = rows.iter().rfind(|r| r.r.prompt).map_or(0.0, |r| r.at);
    let pre: Vec<&Row> = rows.iter().filter(|r| r.r.prompt).collect();
    let dec: Vec<&Row> = rows.iter().filter(|r| !r.r.prompt).collect();
    let end = rows.last().map_or(ttft, |r| r.at);
    (Phase::of(&pre, layers, ttft), Phase::of(&dec, layers, if dec.is_empty() { 0.0 } else { end - ttft }), ttft)
}

/// the counters of a phase per token; `rb` the record bytes
fn counters_json(p: &Phase, rb: u64) -> Value {
    let t = p.tokens.max(1) as f64;
    let tot = p.total_tiers();
    let m = p.total_moves();
    let visits = m.visits.max(1) as f64;
    let gb = |records: u64| records as f64 * rb as f64 / 1e9 / t;
    json!({
        "tokens": p.tokens,
        "visits": m.visits, "visits_per_token": m.visits as f64 / t,
        "hits": { "vram": tot[0], "pinned": tot[1], "nvme": tot[2] },
        "hit_rate": { "vram": tot[0] as f64 / visits, "pinned": tot[1] as f64 / visits, "nvme": tot[2] as f64 / visits },
        "tier_sum_equals_visits": tot[0] + tot[1] + tot[2] == m.visits,
        "nvme_reads": p.nvme_reads, "r_nvme_reads_per_token": p.nvme_reads as f64 / t,
        "nvme_gb_per_token": p.nvme_bytes as f64 / 1e9 / t,
        "m_nvme_share": tot[2] as f64 / visits,
        "h2d_gb_per_token": gb(m.h2d()),
        "zero_copy_gb_per_token": gb(m.zero_copy),
        "host_dram_to_gpu_gb_per_token": gb(m.h2d() + m.zero_copy),
        "d2h_gb_per_token": gb(m.vram_to_pinned),
        "d2d_gb_per_token": gb(m.vram_to_stage + m.stage_to_vram),
        "promotions_per_token": m.promotions() as f64 / t,
        "evictions_per_token": m.evictions() as f64 / t,
        "cpu_lane_per_token": m.cpu_lane as f64 / t,
        "cpu_lane_s_per_token": p.lane_s / t,
        "moves": moves_json(&m),
        "prefetch": prefetch_json(p, t),
    })
}

/// `CROW_GLM_PREFETCH`: the store's counters of a phase. A staged NVMe record the store held
/// (`used`) was read before its layer asked; the demand misses it did not cover are the phase's
/// NVMe reads minus those.
fn prefetch_json(p: &Phase, t: f64) -> Value {
    let f = p.pf.unwrap_or_default();
    let uncovered = p.nvme_reads.saturating_sub(f.used);
    json!({
        "mode": if p.pf.is_some() { "next-layer" } else { "none" },
        "hints": f.hints, "resident": f.resident, "issued": f.issued, "issued_bytes": f.bytes, "used": f.used, "wasted": f.wasted,
        "demand_misses_uncovered": uncovered, "demand_misses_uncovered_per_token": uncovered as f64 / t,
    })
}

/// #192: the speculative decode's counters of one rep, per decode token (`tokens` = the ids the
/// verify steps emitted = the decode phase's tokens)
fn mtp_json(s: &SpecStats) -> Value {
    let t = s.tokens.max(1) as f64;
    let steps = s.steps.max(1) as f64;
    json!({
        "drafts_per_step_max": s.n, "steps": s.steps, "tokens": s.tokens, "drafts": s.drafts, "accepted": s.accepted,
        "acceptance_rate": s.acceptance(), "tokens_per_step": s.tokens as f64 / steps,
        "drafts_per_token": s.drafts as f64 / t, "accepted_per_token": s.accepted as f64 / t,
        "verify_rows": s.verify_rows, "verify_rows_per_token": s.verify_rows as f64 / t,
        "accepted_hist": s.hist, "mtp_rows": s.mtp_rows, "mtp_rows_per_token": s.mtp_rows as f64 / t,
        "kda_snapshots": s.kda_snapshots, "kda_snapshot_gb_per_token": s.kda_snapshot_bytes as f64 / 1e9 / t,
        "kda_restore_steps": s.kda_restore_steps, "kda_restores": s.kda_restores, "kda_restore_gb_per_token": s.kda_restore_bytes as f64 / 1e9 / t,
    })
}

fn mtp_line(s: &SpecStats) -> String {
    let c = mtp_json(s);
    let f = |k: &str| c[k].as_f64().unwrap_or(0.0);
    format!(
        "N {}, steps {}, tokens/step {:.3}, drafts {} accepted {} (acceptance {}), per token: drafts {:.3} accepted {:.3} verify rows {:.3} MTP rows {:.3}, accepted per step {:?}, KDA snapshots {:.4} GB/token, restores {} steps ({:.4} GB/token)",
        s.n,
        s.steps,
        f("tokens_per_step"),
        s.drafts,
        s.accepted,
        s.acceptance().map_or("-".to_string(), |a| format!("{:.1} %", 100.0 * a)),
        f("drafts_per_token"),
        f("accepted_per_token"),
        f("verify_rows_per_token"),
        f("mtp_rows_per_token"),
        s.hist,
        f("kda_snapshot_gb_per_token"),
        s.kda_restore_steps,
        f("kda_restore_gb_per_token")
    )
}

fn moves_json(m: &Moves) -> Value {
    Value::Object(Moves::NAMES.iter().zip(m.fields()).map(|(k, v)| (k.to_string(), json!(v))).collect())
}

fn layers_json(p: &Phase, first_moe: usize) -> Value {
    let t = p.tokens.max(1) as f64;
    Value::Array(
        p.tiers
            .iter()
            .zip(&p.moves)
            .enumerate()
            .map(|(i, (c, m))| {
                let v = m.visits.max(1) as f64;
                json!({ "layer": first_moe + i, "hits": { "vram": c[0], "pinned": c[1], "nvme": c[2] },
                        "hit_rate": { "vram": c[0] as f64 / v, "pinned": c[1] as f64 / v, "nvme": c[2] as f64 / v },
                        "nvme_reads_per_token": m.nvme_reads() as f64 / t, "promotions_per_token": m.promotions() as f64 / t,
                        "evictions_per_token": m.evictions() as f64 / t, "moves": moves_json(m) })
            })
            .collect(),
    )
}

fn timing_json(p: &Phase, ttft: Option<f64>) -> Value {
    let rates = p.token_rates();
    json!({
        "tokens": p.tokens, "wall_s": p.wall, "ttft_s": ttft, "wall_tok_s": p.rate(),
        "tok_s_median_over_tokens": median(&rates), "tok_s_min": rates.iter().copied().reduce(f64::min), "tok_s_max": rates.iter().copied().reduce(f64::max),
        "latency_p50_s": percentile(&p.lat, 50.0), "latency_p99_s": percentile(&p.lat, 99.0), "latency_s": p.lat,
    })
}

fn opt(v: Option<f64>, prec: usize) -> String {
    v.map_or("-".to_string(), |x| format!("{x:.prec$}"))
}

fn counters_line(p: &Phase, rb: u64) -> String {
    let c = counters_json(p, rb);
    let f = |k: &str| c[k].as_f64().unwrap_or(0.0);
    let h = |k: &str| 100.0 * c["hit_rate"][k].as_f64().unwrap_or(0.0);
    format!(
        "visits {:.1}, hits vram {:.1} % pinned {:.1} % nvme {:.1} %, r {:.2} NVMe reads ({:.3} GB), m {:.4}, H2D {:.3} GB, zero-copy {:.3} GB, host DRAM->GPU {:.3} GB, D2H {:.3} GB, promotions {:.2}, evictions {:.2}, prefetch none (issued 0, used 0, wasted 0, demand misses uncovered {:.2}), CPU lane {:.2} experts {:.4} s",
        f("visits_per_token"),
        h("vram"),
        h("pinned"),
        h("nvme"),
        f("r_nvme_reads_per_token"),
        f("nvme_gb_per_token"),
        f("m_nvme_share"),
        f("h2d_gb_per_token"),
        f("zero_copy_gb_per_token"),
        f("host_dram_to_gpu_gb_per_token"),
        f("d2h_gb_per_token"),
        f("promotions_per_token"),
        f("evictions_per_token"),
        c["prefetch"]["demand_misses_uncovered_per_token"].as_f64().unwrap_or(0.0),
        f("cpu_lane_per_token"),
        f("cpu_lane_s_per_token")
    )
}

/// the per-layer spread of a phase in one line: NVMe reads per token and VRAM hit rate, the
/// lowest and highest layer of each
fn layers_line(p: &Phase, first_moe: usize) -> String {
    if p.tokens == 0 || p.moves.is_empty() {
        return "no rows".into();
    }
    let t = p.tokens as f64;
    let r: Vec<f64> = p.moves.iter().map(|m| m.nvme_reads() as f64 / t).collect();
    let h: Vec<f64> = p.tiers.iter().zip(&p.moves).map(|(c, m)| 100.0 * c[0] as f64 / m.visits.max(1) as f64).collect();
    let arg = |v: &[f64], max: bool| -> usize {
        let mut best = 0;
        for i in 1..v.len() {
            if (max && v[i] > v[best]) || (!max && v[i] < v[best]) {
                best = i;
            }
        }
        best
    };
    let (rl, rh, hl, hh) = (arg(&r, false), arg(&r, true), arg(&h, false), arg(&h, true));
    format!(
        "NVMe reads/tok min l{} {:.2} median {:.2} max l{} {:.2}; VRAM hit min l{} {:.1} % max l{} {:.1} %",
        first_moe + rl,
        r[rl],
        median(&r).unwrap_or(0.0),
        first_moe + rh,
        r[rh],
        first_moe + hl,
        h[hl],
        first_moe + hh,
        h[hh]
    )
}

/// median and spread of one metric over the reps, with the 1.15 rule named (no gate verdict)
fn summary_json(v: &[f64]) -> Value {
    let s = spread(v);
    json!({ "values": v, "median": median(v), "spread_max_over_min": s, "within_spread_rule": s.map(|x| x <= SPREAD_RULE) })
}

// ---------------------------------------------------------------- machine state

/// (working set, peak working set, private commit) of this process in bytes, as the OS reports
/// them (Windows `K32GetProcessMemoryInfo`: `WorkingSetSize`, `PeakWorkingSetSize`,
/// `PagefileUsage`; unix `/proc/self/status` `VmRSS`, `VmHWM`, no commit); zeros if the query fails
#[cfg(windows)]
fn process_memory() -> (u64, u64, u64) {
    #[repr(C)]
    #[derive(Default)]
    struct Pmc {
        cb: u32,
        page_fault_count: u32,
        peak_working_set_size: usize,
        working_set_size: usize,
        quota_peak_paged_pool_usage: usize,
        quota_paged_pool_usage: usize,
        quota_peak_non_paged_pool_usage: usize,
        quota_non_paged_pool_usage: usize,
        pagefile_usage: usize,
        peak_pagefile_usage: usize,
    }
    type FnCur = unsafe extern "system" fn() -> *mut std::ffi::c_void;
    type FnPmi = unsafe extern "system" fn(*mut std::ffi::c_void, *mut Pmc, u32) -> i32;
    unsafe {
        let Ok(lib) = libloading::Library::new("kernel32.dll") else { return (0, 0, 0) };
        let (Ok(cur), Ok(pmi)) = (lib.get::<FnCur>(b"GetCurrentProcess\0"), lib.get::<FnPmi>(b"K32GetProcessMemoryInfo\0")) else { return (0, 0, 0) };
        let mut c = Pmc { cb: std::mem::size_of::<Pmc>() as u32, ..Pmc::default() };
        let cb = c.cb;
        if pmi(cur(), &mut c, cb) == 0 {
            return (0, 0, 0);
        }
        (c.working_set_size as u64, c.peak_working_set_size as u64, c.pagefile_usage as u64)
    }
}

#[cfg(unix)]
fn process_memory() -> (u64, u64, u64) {
    let t = std::fs::read_to_string("/proc/self/status").unwrap_or_default();
    let kb = |k: &str| t.lines().find(|l| l.starts_with(k)).and_then(|l| l.split_whitespace().nth(1)).and_then(|v| v.parse::<u64>().ok()).unwrap_or(0) * 1024;
    (kb("VmRSS:"), kb("VmHWM:"), 0)
}

/// VRAM, RAM and commit now, plus the store's pinned and VRAM bytes
///
/// # Safety
/// A CUDA context is current.
unsafe fn machine(tiers: &ExpertTiers) -> Value {
    let (free, total) = (cuda::free_vram_bytes(), cuda::total_vram_bytes());
    let (ws, peak, private) = process_memory();
    let (commit_free, commit_limit) = cuda::commit_bytes();
    json!({
        "vram_used_bytes": total.saturating_sub(free), "vram_free_bytes": free, "vram_total_bytes": total,
        "working_set_bytes": ws, "peak_working_set_bytes": peak, "private_commit_bytes": private,
        "free_ram_bytes": cuda::free_physical_ram(), "system_commit_free_bytes": commit_free, "system_commit_limit_bytes": commit_limit,
        "tiers_pinned_bytes": tiers.pinned_bytes(), "tiers_vram_bytes": tiers.vram_bytes(),
    })
}

fn machine_line(m: &Value) -> String {
    let g = |k: &str| m[k].as_u64().unwrap_or(0) as f64 / (1u64 << 30) as f64;
    format!(
        "VRAM used {:.2} / free {:.2} GiB, working set {:.2} GiB (peak {:.2}), private commit {:.2} GiB, free RAM {:.2} GiB, system commit free {:.2} / limit {:.2} GiB, pinned {:.2} GiB",
        g("vram_used_bytes"),
        g("vram_free_bytes"),
        g("working_set_bytes"),
        g("peak_working_set_bytes"),
        g("private_commit_bytes"),
        g("free_ram_bytes"),
        g("system_commit_free_bytes"),
        g("system_commit_limit_bytes"),
        g("tiers_pinned_bytes")
    )
}

/// HEAD of the checkout this runs in (`git rev-parse HEAD`, `-dirty` with tracked changes);
/// the binary's own build commit is not recorded (no build script), its path and mtime are
fn commit() -> String {
    let git = |a: &[&str]| std::process::Command::new("git").args(a).output().ok().filter(|o| o.status.success()).map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string());
    match git(&["rev-parse", "HEAD"]) {
        Some(h) if git(&["status", "--porcelain", "--untracked-files=no"]).is_some_and(|s| !s.is_empty()) => format!("{h}-dirty"),
        Some(h) => h,
        None => "unknown".into(),
    }
}

fn env_json() -> Value {
    let mut v: Vec<(String, String)> = std::env::vars().filter(|(k, _)| k.starts_with("CROW_") || k == "CUDA_VISIBLE_DEVICES").collect();
    v.sort();
    Value::Object(v.into_iter().map(|(k, x)| (k, json!(x))).collect())
}

fn write_json(path: &str, v: &Value) -> Result<(), String> {
    std::fs::write(path, serde_json::to_string_pretty(v).map_err(|e| e.to_string())?).map_err(|e| format!("--json {path}: {e}"))
}

// ---------------------------------------------------------------- the run

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
    let reps = num("--reps")?.unwrap_or(1);
    if reps == 0 {
        return Err("--reps 0: nothing to run".into());
    }
    let cold = args.iter().any(|a| a == "--cold");
    let json_path = flag("--json")?;
    let prompt_tokens = num("--prompt-tokens")?;
    let path = flag("--cnq")?.or_else(|| std::env::var("CROW_CNQ").ok()).unwrap_or_else(|| from_engine_dir(GLM5_MUL1K3_CNQ));
    let tok = match flag("--tokenizer")? {
        Some(t) => {
            let cfg = crow_nest_engine::tokenizer::sibling_config(&t);
            Some(crow_nest_engine::tokenizer::ChatTokenizer::load(&t, &cfg)?)
        }
        None => None,
    };
    let (ids_flag, text_flag, file_flag) = (flag("--ids")?, flag("--prompt")?, flag("--prompt-ids")?);
    if [ids_flag.is_some(), text_flag.is_some(), file_flag.is_some()].iter().filter(|&&b| b).count() > 1 {
        return Err("give the prompt once: --ids, --prompt or --prompt-ids".into());
    }
    let (base, source): (Vec<i64>, String) = if let Some(ids) = ids_flag {
        (ids.split(',').map(|v| v.trim().parse::<i64>().map_err(|_| format!("--ids: {v:?} is not a token id"))).collect::<Result<_, _>>()?, "--ids".into())
    } else if let Some(text) = text_flag {
        let t = tok.as_ref().ok_or("--prompt needs --tokenizer <tokenizer.json>")?;
        (t.encode_chat_user(&text)?.into_iter().map(i64::from).collect(), "--prompt".into())
    } else if let Some(f) = file_flag {
        (parse_ids(&std::fs::read_to_string(&f).map_err(|e| format!("--prompt-ids {f}: {e}"))?)?, format!("--prompt-ids {f}"))
    } else {
        (gt::fixed_prompt(), "fixed sys_user_default".into())
    };
    let prompt = match prompt_tokens {
        Some(k) => prompt_of_len(&base, k)?,
        None => base.clone(),
    };
    let t_open = std::time::Instant::now();
    let mut o = gt::open_container(&path)?;
    let open_s = t_open.elapsed().as_secs_f64();
    let container_bytes = std::fs::metadata(&path).map(|m| m.len()).unwrap_or(0);
    println!(
        "[glm5_run] container {path} ({container_bytes} B): {} constants verified, routed experts {} x {} B ({} records, codec {}); prompt {} ids ({source}{}), generating {n}, reps {reps}{}",
        o.constants,
        o.g.experts,
        o.spec.bytes,
        o.records,
        o.spec.codec.dtype(),
        prompt.len(),
        if prompt_tokens.is_some() { format!(", base {} ids repeated / cut", base.len()) } else { String::new() },
        if cold { ", --cold" } else { "" }
    );
    let context = crow_nest_engine::boot::context_from_env(std::env::var("CROW_CONTEXT").ok().as_deref(), o.g.context_floor, o.g.context_max)?;
    let commit = commit();
    let exe = std::env::current_exe().ok();
    let exe_mtime = exe.as_ref().and_then(|p| std::fs::metadata(p).ok()).and_then(|m| m.modified().ok()).and_then(|t| t.duration_since(std::time::UNIX_EPOCH).ok()).map(|d| d.as_secs());
    // #192: CROW_GLM_MTP=N drafts per step (0 = off); its VRAM comes off the planner's free VRAM
    let drafts = mtp::draft_rows_from_env()?;
    let cap = prompt.len() + n;
    unsafe {
        let _ctx = cuda::Ctx::init();
        let free = cuda::free_vram_bytes();
        let mtp_reserved = if drafts > 0 { mtp::spec_vram_bytes(&o.g, &o.moe, cap, drafts) } else { 0 };
        if drafts > 0 {
            println!("[glm5_run] {}={drafts}: the #159 plan runs at free VRAM minus {mtp_reserved} B (the MTP block, its caches, {drafts} KDA snapshot slots; derived)", mtp::MTP_ENV);
        }
        let budget = derive_host_pinned_budget(HOST_PINNED_CAP, &mut |s| println!("{s}"));
        // #186: the prompt chunk Glm5Run::load takes (CROW_CHUNK, at most the run's rows), booked by the plan
        let (_, input, plan) = gt::plan_for_rows(&o.g, context, prompt.len() + n, free.saturating_sub(mtp_reserved), budget, o.spec.bytes)?;
        let chunk = input.chunk;
        let sizes = gt::tier_sizes(&plan, num("--vram-slots")?, num("--pinned-slots")?)?;
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        let moe_layers = gt::moe_layers(&o.g);
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
            moe_layers
        );
        if chunk > 1 {
            println!(
                "[glm5_run] #186 prompt chunk {chunk} (CROW_CHUNK): prompt calls {:?}, plan books {:.2} GiB for them (FFN plans of {:?} rows) and the {} B pageable prefill landing off the pinned budget",
                gt::prompt_calls(prompt.len(), chunk).iter().map(|c| c.1).collect::<Vec<_>>(),
                gib(input.chunk_scratch_bytes),
                crow_nest_engine::manager::glm5_prompt_call_sizes(chunk),
                gt::prefill_stage_slots(o.g.topk) as u64 * o.spec.bytes
            );
        } else {
            println!("[glm5_run] #186 prompt chunk 1 (CROW_CHUNK unset or 1): every prompt row one decode call");
        }
        if drafts > 0 && chunk > 1 {
            println!("[glm5_run] {}={drafts}: the prompt runs in the prompt calls above; after each call the MTP block takes its rows' head-norm rows in the windows of the row path ({} rows from position 0)", mtp::MTP_ENV, mtp::MTP_CHUNK);
        }
        let t_load = std::time::Instant::now();
        let mut run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, cap, &mut |s| println!("{s}"));
        let mtp_n = run.mtp_from_env(&mut o.cnq, &mut |s| println!("{s}"))?;
        let load_s = t_load.elapsed().as_secs_f64();
        let t_tiers = std::time::Instant::now();
        // #192: a verify call stages the experts of 1 + N rows at once
        let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, readers, (1 + mtp_n) * o.g.topk)?;
        if run.prompt_chunk() > 1 {
            // #186: allocated here, so the first prompt call's clock does not hold it
            tiers.alloc_prefill_stage(gt::prefill_stage_slots(o.g.topk))?;
        }
        let tiers_s = t_tiers.elapsed().as_secs_f64();
        println!(
            "[glm5_run] tiers: {:.2} GiB VRAM (slots, {} decode + {} prefill staging, tables), {:.2} GiB pinned; free VRAM now {:.2} GiB; policy {:?}, cache empty at start; pinned {:?}",
            gib(tiers.vram_bytes()),
            tiers.stage_cap,
            tiers.prefill_cap(),
            gib(tiers.pinned_bytes()),
            gib(cuda::free_vram_bytes()),
            tiers.cache.policy,
            tiers.pinned_use
        );
        println!("glm5_run setup: open {open_s:.2} s, load {load_s:.2} s (dense part + head), tiers {tiers_s:.2} s; commit {commit}");
        let m_setup = machine(&tiers);
        println!("glm5_run machine after setup: {}", machine_line(&m_setup));
        let first_moe = tiers.first_moe;
        let rb = tiers.rb;
        let mut doc = json!({
            "tool": "glm5_run", "ticket": 187, "args": args, "env": env_json(), "commit": commit,
            "binary": { "path": exe.as_ref().map(|p| p.display().to_string()), "mtime_unix_s": exe_mtime },
            "container": { "path": path, "bytes": container_bytes, "record_bytes": rb, "experts": o.g.experts, "codec": o.spec.codec.dtype() },
            "prompt": { "source": source, "base_ids": base.len(), "ids": prompt.len(), "prompt_tokens_flag": prompt_tokens },
            "generate": n, "reps": reps, "cold": cold, "context": context,
            "tiers": { "plan": { "vram": plan.hot, "pinned": plan.pinned, "nvme": plan.nvme }, "vram_slots": sizes.vram, "pinned_slots": sizes.pinned,
                       "nvme": o.g.experts - sizes.vram - sizes.pinned, "moe_layers": moe_layers, "first_moe_layer": first_moe, "readers": readers,
                       "policy": format!("{:?}", tiers.cache.policy), "pinned_use": format!("{:?}", tiers.pinned_use), "staging_slots": tiers.stage_cap, "pinned_budget_bytes": budget, "free_vram_at_plan_bytes": free,
                       "prefill_staging_slots": tiers.prefill_cap() },
            "prompt_chunk": { "chunk": run.prompt_chunk(), "calls": gt::prompt_calls(prompt.len(), run.prompt_chunk()).iter().map(|c| c.1).collect::<Vec<_>>(), "plan_chunk_scratch_bytes": input.chunk_scratch_bytes },
            "setup": { "open_s": open_s, "load_s": load_s, "tiers_s": tiers_s },
            "mtp": { "drafts_per_step": mtp_n, "planner_vram_reserved_bytes": mtp_reserved, "overlay": std::env::var(mtp::OVERLAY_ENV).ok() },
            "machine_after_setup": m_setup,
            "clocks": "row seconds: TokenReport::secs, row start to after the row's closing cuda::sync + greedy-id dtoh; ttft and phase wall: on entry of the report callback, after that sync; no sync added",
            "reps_detail": [],
        });
        let lane = tiers.cpu_lane_clock();
        let pf_clock = tiers.prefetch_clock();
        let mut per_rep: Vec<(Phase, Phase, f64, f64)> = Vec::new();
        let mut all_ids: Vec<Vec<i64>> = Vec::new();
        for rep in 1..=reps {
            let state = if rep == 1 || cold { "cold" } else { "warm" };
            if cold && rep > 1 {
                tiers.reset_cache()?;
            }
            let mut rows: Vec<Row> = Vec::with_capacity(cap);
            let mut lane_ns = lane.read().0;
            let pf_read = || pf_clock.as_ref().map(|c| c.lock().map(|g| *g).unwrap_or_default());
            let mut pf_last = pf_read();
            let t0 = std::time::Instant::now();
            let mut report = |r: &TokenReport| {
                let at = t0.elapsed().as_secs_f64();
                let ns = lane.read().0;
                let lane_s = (ns - lane_ns) as f64 / 1e9;
                lane_ns = ns;
                let per: Vec<String> = r.tiers.iter().enumerate().map(|(i, c)| format!("l{} {}/{}/{}", first_moe + i, c[0], c[1], c[2])).collect();
                let sum = r.tiers.iter().fold([0u64; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
                let id = r.next.map_or("-".to_string(), |v| v.to_string());
                let call = if r.rows > 1 { format!(" ({} rows, {} routing syncs, {} sub-batches)", r.rows, r.routing_syncs, r.sub_batches) } else { String::new() };
                println!(
                    "glm5_run row {:>4} {}{call} rep {rep} next {id:>6}  {:.4} s  NVMe reads {:>3} ({:.1} MB)  tiers v/p/n {}/{}/{}  [{}]",
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
                let pf_now = pf_read();
                let pf = pf_now.zip(pf_last).map(|(a, b)| a.since(&b));
                pf_last = pf_now;
                rows.push(Row { r: r.clone(), at, lane_s, pf });
            };
            let out = run.generate(&mut o.cnq, &mut tiers, &prompt, n, false, &mut report)?;
            let wall = t0.elapsed().as_secs_f64();
            let (pre, dec, ttft) = phases(&rows, moe_layers);
            let tag = format!("glm5_run rep {rep}/{reps} cache {state}");
            println!("{tag} ids {:?}", out.ids);
            if let Some(t) = &tok {
                let ids: Vec<u32> = out.ids.iter().map(|&v| v as u32).collect();
                println!("{tag} text {:?}", t.decode(&ids)?);
            }
            println!(
                "{tag} prefill: {} tok, TTFT {ttft:.3} s, {} tok/s, {} latency p50 {} s p99 {} s; prompt chunk {}: {} reports, {} routing syncs, {} serve sub-batches",
                pre.tokens,
                opt(pre.rate(), 2),
                if run.prompt_chunk() > 1 { "call" } else { "row" },
                opt(percentile(&pre.lat, 50.0), 4),
                opt(percentile(&pre.lat, 99.0), 4),
                run.prompt_chunk(),
                pre.reports,
                pre.routing_syncs,
                pre.sub_batches
            );
            let dr = dec.token_rates();
            if dec.tokens == 0 {
                println!("{tag} decode: 0 tok (-n 1: the only id comes from the last prompt row)");
            } else {
                println!(
                    "{tag} decode: {} tok, {} tok/s median over tokens (min {}, max {}), wall {:.3} s = {} tok/s, latency p50 {} s p99 {} s",
                    dec.tokens,
                    opt(median(&dr), 2),
                    opt(dr.iter().copied().reduce(f64::min), 2),
                    opt(dr.iter().copied().reduce(f64::max), 2),
                    dec.wall,
                    opt(dec.rate(), 2),
                    opt(percentile(&dec.lat, 50.0), 4),
                    opt(percentile(&dec.lat, 99.0), 4)
                );
            }
            println!("{tag} prefill counters per token: {}", counters_line(&pre, rb));
            println!("{tag} decode counters per token: {}", counters_line(&dec, rb));
            println!("{tag} prefill layers: {}", layers_line(&pre, first_moe));
            println!("{tag} decode layers: {}", layers_line(&dec, first_moe));
            let spec = run.mtp_stats().cloned();
            if let Some(s) = &spec {
                println!("{tag} decode MTP: {}", mtp_line(s));
            }
            println!("{tag} wall {wall:.3} s (generate call)");
            let m_rep = machine(&tiers);
            println!("glm5_run machine after rep {rep}: {}", machine_line(&m_rep));
            doc["reps_detail"].as_array_mut().expect("reps array").push(json!({
                "rep": rep, "cache": state, "wall_s": wall, "ids": out.ids,
                "prefill": { "timing": timing_json(&pre, Some(ttft)), "counters": counters_json(&pre, rb), "layers": layers_json(&pre, first_moe),
                             "reports": pre.reports, "routing_syncs": pre.routing_syncs, "sub_batches": pre.sub_batches },
                "decode": { "timing": timing_json(&dec, None), "counters": counters_json(&dec, rb), "layers": layers_json(&dec, first_moe),
                            "mtp": spec.as_ref().map(mtp_json) },
                "rows": rows.iter().map(|x| json!({ "pos": x.r.pos, "prompt": x.r.prompt, "next": x.r.next, "secs": x.r.secs, "at_s": x.at,
                    "nvme_reads": x.r.nvme_reads, "nvme_bytes": x.r.nvme_bytes, "rows": x.r.rows, "routing_syncs": x.r.routing_syncs, "sub_batches": x.r.sub_batches })).collect::<Vec<_>>(),
                "machine": m_rep,
            }));
            all_ids.push(out.ids);
            per_rep.push((pre, dec, ttft, wall));
            if let Some(p) = &json_path {
                write_json(p, &doc)?;
            }
        }
        let (text, summary) = summarize(&per_rep, &all_ids, cold);
        for l in &text {
            println!("{l}");
        }
        doc["summary"] = summary;
        if let Some(p) = &json_path {
            write_json(p, &doc)?;
            println!("glm5_run json {p}");
        }
        tiers.free();
        run.free();
    }
    Ok(())
}

/// the summary over the reps: median and spread max/min of each metric, the 1.15 rule named,
/// ids compared; (text lines, JSON)
fn summarize(per_rep: &[(Phase, Phase, f64, f64)], all_ids: &[Vec<i64>], cold: bool) -> (Vec<String>, Value) {
    let reps = per_rep.len();
    let col = |f: &dyn Fn(&(Phase, Phase, f64, f64)) -> Option<f64>| -> Vec<f64> { per_rep.iter().filter_map(f).collect() };
    let metrics: Vec<(&str, Vec<f64>)> = vec![
        ("ttft_s", col(&|x| Some(x.2))),
        ("prefill_tok_s", col(&|x| x.0.rate())),
        ("decode_tok_s_median", col(&|x| median(&x.1.token_rates()))),
        ("decode_wall_tok_s", col(&|x| x.1.rate())),
        ("decode_latency_p50_s", col(&|x| percentile(&x.1.lat, 50.0))),
        ("decode_latency_p99_s", col(&|x| percentile(&x.1.lat, 99.0))),
        ("rep_wall_s", col(&|x| Some(x.3))),
        ("decode_r_nvme_reads_per_token", col(&|x| (x.1.tokens > 0).then(|| x.1.nvme_reads as f64 / x.1.tokens as f64))),
    ];
    let same_ids = all_ids.windows(2).all(|w| w[0] == w[1]);
    let mut summary = serde_json::Map::new();
    let mut parts = Vec::new();
    for (k, v) in &metrics {
        let s = summary_json(v);
        parts.push(format!("{k} {} (spread {})", opt(s["median"].as_f64(), 4), opt(s["spread_max_over_min"].as_f64(), 3)));
        summary.insert(k.to_string(), s);
    }
    let above: Vec<&str> = metrics.iter().filter(|(_, v)| spread(v).is_some_and(|s| s > SPREAD_RULE)).map(|(k, _)| *k).collect();
    let states: Vec<&str> = (1..=reps).map(|r| if r == 1 || cold { "cold" } else { "warm" }).collect();
    summary.insert("ids_identical_across_reps".into(), json!(same_ids));
    summary.insert("spread_rule".into(), json!(SPREAD_RULE));
    summary.insert("above_spread_rule".into(), json!(above));
    summary.insert("cache_per_rep".into(), json!(states));
    let text = vec![
        format!("glm5_run summary over {reps} reps (median, spread max/min): {}", parts.join("; ")),
        format!(
            "glm5_run summary spread rule <= {SPREAD_RULE}: {}; ids identical across reps: {}; cache per rep: {}",
            if reps < 2 {
                "n/a (1 rep)".to_string()
            } else if above.is_empty() {
                "every metric within".to_string()
            } else {
                format!("above for {}", above.join(", "))
            },
            if same_ids { "yes" } else { "NO" },
            states.join(" ")
        ),
    ];
    (text, Value::Object(summary))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn moves(nvme: u64) -> Moves {
        Moves { visits: 8, nvme_to_landing: nvme, landing_to_stage: nvme, n2v: 1, v2p: 1, zero_copy: 2, ..Moves::default() }
    }

    /// a synthetic run of 3 prompt rows and 3 generated ids (5 rows: the last prompt row yields
    /// the first id), 2 MoE layers, 8 visits per layer per row; the callback clock runs 0.01 s
    /// ahead of the row seconds per row (the print)
    fn synthetic() -> Vec<Row> {
        let secs = [0.5, 0.25, 0.25, 0.2, 0.1];
        let mut at = 0.0;
        (0..5)
            .map(|pos| {
                at += secs[pos] + 0.01;
                let prompt = pos < 3;
                let nv = if prompt { 4 } else { 1 };
                Row {
                    r: TokenReport {
                        pos,
                        prompt,
                        next: (pos >= 2).then_some(pos as i64),
                        secs: secs[pos],
                        nvme_reads: 2 * nv,
                        nvme_bytes: 2 * nv * 100,
                        tiers: vec![[2, 6 - nv, nv]; 2],
                        moves: vec![moves(nv); 2],
                        rows: 1,
                        routing_syncs: 2,
                        sub_batches: 2,
                    },
                    at,
                    lane_s: 0.0,
                    // the store on in decode rows only: 1 record used of 2 read per row
                    pf: (!prompt).then_some(PrefetchStats { hints: 1, issued: 2, bytes: 200, used: 1, wasted: 1, ..PrefetchStats::default() }),
                }
            })
            .collect()
    }

    #[test]
    fn the_phases_split_at_the_last_prompt_row_and_ttft_is_its_report() {
        let rows = synthetic();
        let (pre, dec, ttft) = phases(&rows, 2);
        assert_eq!((pre.tokens, dec.tokens), (3, 2));
        // TTFT: the callback clock of row 2 (the last prompt row, which yields the first id)
        assert!((ttft - (0.5 + 0.25 + 0.25 + 0.03)).abs() < 1e-12, "{ttft}");
        assert!((pre.rate().unwrap() - 3.0 / ttft).abs() < 1e-12);
        // decode wall = last row's clock - TTFT = 0.2 + 0.1 + 2 x 0.01
        assert!((dec.wall - 0.32).abs() < 1e-12, "{}", dec.wall);
        assert_eq!(dec.lat, vec![0.2, 0.1]);
        // per-token rates 5 and 10: median 7.5, min 5, max 10
        let r = dec.token_rates();
        assert_eq!((median(&r), r.iter().copied().reduce(f64::min), r.iter().copied().reduce(f64::max)), (Some(7.5), Some(5.0), Some(10.0)));
        let tj = timing_json(&dec, None);
        assert_eq!((tj["latency_p50_s"].as_f64(), tj["latency_p99_s"].as_f64()), (Some(0.1), Some(0.2)));
        // counters: 3 prompt rows x 2 layers x 8 visits; the tiers sum to the visits
        let c = counters_json(&pre, 1_000_000_000);
        assert_eq!(c["visits"], 48);
        assert_eq!(c["tier_sum_equals_visits"], true);
        assert_eq!(c["visits_per_token"], 16.0);
        assert_eq!(c["nvme_reads"], 24);
        assert_eq!(c["r_nvme_reads_per_token"], 8.0);
        // 1 GB records: 4 landing->stage x 2 layers per row = 8 GB H2D per token, 2 x 2 zero-copy
        assert_eq!(c["h2d_gb_per_token"], 8.0);
        assert_eq!(c["zero_copy_gb_per_token"], 4.0);
        assert_eq!(c["host_dram_to_gpu_gb_per_token"], 12.0);
        assert_eq!((c["promotions_per_token"].as_f64(), c["evictions_per_token"].as_f64()), (Some(2.0), Some(2.0)));
        assert_eq!((c["prefetch"]["mode"].as_str(), c["prefetch"]["issued"].as_u64(), c["prefetch"]["demand_misses_uncovered"].as_u64()), (Some("none"), Some(0), Some(24)));
        assert_eq!(c["hit_rate"]["nvme"], 0.5);
        let d = counters_json(&dec, 1);
        let rows_dec = dec.tokens as u64;
        assert_eq!(d["prefetch"]["mode"], "next-layer");
        assert_eq!((d["prefetch"]["issued"].as_u64(), d["prefetch"]["used"].as_u64(), d["prefetch"]["wasted"].as_u64()), (Some(2 * rows_dec), Some(rows_dec), Some(rows_dec)));
        assert_eq!(d["prefetch"]["demand_misses_uncovered"].as_u64(), Some(dec.nvme_reads - rows_dec));
        assert_eq!(d["m_nvme_share"], 2.0 / 16.0);
        let l = layers_json(&dec, 3);
        assert_eq!(l.as_array().unwrap().len(), 2);
        assert_eq!(l[1]["layer"], 4);
        assert_eq!(l[1]["nvme_reads_per_token"], 1.0);
        assert!(layers_line(&dec, 3).starts_with("NVMe reads/tok min l3 1.00 median 1.00 max l3 1.00"), "{}", layers_line(&dec, 3));
        assert!(counters_line(&pre, 1_000_000_000).starts_with("visits 16.0, hits vram 25.0 % pinned 25.0 % nvme 50.0 %, r 8.00 NVMe reads"), "{}", counters_line(&pre, 1_000_000_000));
    }

    /// `-n 1`: the only id comes from the last prompt row; the decode phase is empty, not NaN
    #[test]
    fn one_generated_id_leaves_an_empty_decode_phase() {
        let rows: Vec<Row> = synthetic().into_iter().take(3).collect();
        let (pre, dec, ttft) = phases(&rows, 2);
        assert_eq!((pre.tokens, dec.tokens, dec.wall), (3, 0, 0.0));
        assert!(ttft > 0.0);
        assert_eq!(dec.rate(), None);
        let t = timing_json(&dec, None);
        assert!(t["tok_s_median_over_tokens"].is_null() && t["latency_p99_s"].is_null());
        assert_eq!(counters_json(&dec, 1)["tokens"], 0);
        assert_eq!(layers_line(&dec, 3), "no rows");
    }

    #[test]
    fn median_percentile_and_spread() {
        assert_eq!(median(&[3.0, 1.0, 2.0]), Some(2.0));
        assert_eq!(median(&[4.0, 1.0, 2.0, 3.0]), Some(2.5));
        assert_eq!(median(&[]), None);
        let v: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        assert_eq!((percentile(&v, 50.0), percentile(&v, 99.0), percentile(&v, 100.0)), (Some(50.0), Some(99.0), Some(100.0)));
        assert_eq!(percentile(&[0.3, 0.1, 0.2], 99.0), Some(0.3));
        assert_eq!(percentile(&[0.3], 50.0), Some(0.3));
        assert_eq!(spread(&[2.0, 2.2, 2.1]), Some(2.2 / 2.0));
        assert_eq!(spread(&[2.0]), None);
        let s = summary_json(&[10.0, 12.0, 11.0]);
        assert_eq!((s["median"].as_f64(), s["within_spread_rule"].as_bool()), (Some(11.0), Some(false)));
        assert_eq!(summary_json(&[10.0, 11.0])["within_spread_rule"], true);
    }

    /// #186: a prompt phase of 3 rows as one prompt call (one report of 3 rows, its routing syncs
    /// and sub-batches) counts 3 prefill tokens, TTFT is that report's clock, and decode is unchanged
    #[test]
    fn a_prompt_call_report_counts_its_rows() {
        let mut rows = synthetic();
        let call = Row { r: TokenReport { rows: 3, routing_syncs: 2, sub_batches: 5, ..rows[2].r.clone() }, at: rows[2].at, lane_s: 0.0, pf: None };
        rows.splice(0..3, [call]);
        let (pre, dec, ttft) = phases(&rows, 2);
        assert_eq!((pre.tokens, pre.reports, pre.routing_syncs, pre.sub_batches, dec.tokens), (3, 1, 2, 5, 2));
        assert!((ttft - (0.5 + 0.25 + 0.25 + 0.03)).abs() < 1e-12, "{ttft}");
        assert!((pre.rate().unwrap() - 3.0 / ttft).abs() < 1e-12);
        assert!((dec.wall - 0.32).abs() < 1e-12, "{}", dec.wall);
    }

    /// three reps of the synthetic run, the third slower: medians over the reps, the spread rule
    /// named per metric, ids compared, warm after rep 1 unless --cold
    #[test]
    fn the_summary_gives_medians_spreads_and_the_cache_state_per_rep() {
        let rows = synthetic();
        let (pre, dec, ttft) = phases(&rows, 2);
        let mut slow = rows.clone();
        for r in slow.iter_mut() {
            r.r.secs *= 1.5;
            r.at *= 1.5;
        }
        let (pre3, dec3, ttft3) = phases(&slow, 2);
        let per = vec![(pre.clone(), dec.clone(), ttft, 2.0), (pre, dec, ttft, 2.0), (pre3, dec3, ttft3, 3.0)];
        let ids = vec![vec![2, 3, 4]; 3];
        let (text, s) = summarize(&per, &ids, false);
        assert_eq!(s["ttft_s"]["median"].as_f64(), Some(ttft));
        assert!((s["ttft_s"]["spread_max_over_min"].as_f64().unwrap() - 1.5).abs() < 1e-12);
        assert_eq!(s["decode_tok_s_median"]["values"].as_array().unwrap().len(), 3);
        assert_eq!(s["ids_identical_across_reps"], true);
        assert_eq!(s["cache_per_rep"], json!(["cold", "warm", "warm"]));
        assert!(text[0].starts_with("glm5_run summary over 3 reps (median, spread max/min): ttft_s 1.0300 (spread 1.500)"), "{}", text[0]);
        assert!(text[1].contains("above for ttft_s, prefill_tok_s, decode_tok_s_median"), "{}", text[1]);
        assert!(text[1].ends_with("ids identical across reps: yes; cache per rep: cold warm warm"), "{}", text[1]);
        let (text, s) = summarize(&per[..1], &ids[..1], true);
        assert_eq!(s["cache_per_rep"], json!(["cold"]));
        assert!(text[1].starts_with("glm5_run summary spread rule <= 1.15: n/a (1 rep)"), "{}", text[1]);
        let (_, s) = summarize(&per[..2], &[vec![1], vec![2]], true);
        assert_eq!((s["ids_identical_across_reps"].as_bool(), s["cache_per_rep"].clone()), (Some(false), json!(["cold", "cold"])));
    }

    #[test]
    fn the_prompt_length_repeats_the_base_and_an_ids_file_parses() {
        // the default prompt: the tokenizer golden sys_user_default, 31 ids
        let base = gt::fixed_prompt();
        assert_eq!(base.len(), 31);
        let p = prompt_of_len(&base, 100).unwrap();
        assert_eq!(p.len(), 100);
        assert_eq!(&p[..31], &base[..]);
        assert_eq!(&p[31..62], &base[..]);
        assert_eq!(&p[62..93], &base[..]);
        assert_eq!(&p[93..], &base[..7]);
        assert_eq!(prompt_of_len(&base, 5).unwrap(), base[..5].to_vec());
        assert_eq!(prompt_of_len(&base, 31).unwrap(), base);
        assert!(prompt_of_len(&base, 0).is_err());
        assert_eq!(parse_ids("[1, 2,3]\n4 5\r\n").unwrap(), vec![1, 2, 3, 4, 5]);
        assert!(parse_ids("1, x").unwrap_err().contains("\"x\" is not a token id"));
        assert!(parse_ids(" \n").is_err());
    }

    /// #192: the MTP counters per decode token: 10 ids from 6 steps of N = 2 (12 drafts, 4
    /// accepted), 3 rejected steps restored
    #[test]
    fn the_mtp_counters_are_per_decode_token() {
        let s = SpecStats {
            n: 2,
            steps: 6,
            tokens: 10,
            drafts: 12,
            accepted: 4,
            verify_rows: 18,
            hist: vec![3, 1, 1],
            mtp_rows: 20,
            kda_snapshots: 12 * 34,
            kda_snapshot_bytes: 12 * 1_000_000_000,
            kda_restore_steps: 4,
            kda_restores: 4 * 34,
            kda_restore_bytes: 4 * 1_000_000_000,
        };
        let j = mtp_json(&s);
        assert_eq!(j["acceptance_rate"].as_f64(), Some(4.0 / 12.0));
        assert_eq!(j["tokens_per_step"].as_f64(), Some(10.0 / 6.0));
        assert_eq!((j["drafts_per_token"].as_f64(), j["verify_rows_per_token"].as_f64()), (Some(1.2), Some(1.8)));
        assert_eq!((j["kda_snapshot_gb_per_token"].as_f64(), j["kda_restore_gb_per_token"].as_f64()), (Some(1.2), Some(0.4)));
        assert_eq!(j["accepted_hist"], json!([3, 1, 1]));
        let l = mtp_line(&s);
        assert!(l.starts_with("N 2, steps 6, tokens/step 1.667, drafts 12 accepted 4 (acceptance 33.3 %), per token: drafts 1.200 accepted 0.400 verify rows 1.800"), "{l}");
        assert!(mtp_line(&SpecStats::default()).contains("(acceptance -)"));
    }

    #[test]
    fn the_process_memory_query_answers() {
        let (ws, peak, _) = process_memory();
        assert!(ws > 0 && peak >= ws, "working set {ws}, peak {peak}");
    }
}
