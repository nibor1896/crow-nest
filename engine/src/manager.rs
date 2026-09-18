//! #9 — the three-state memory manager. OWNS every state allocation of the
//! engine (spec 2.5): KV cache (12 full-attention layers), GDN recurrent state
//! (36 layers, f32, fixed), QSA indexer key + pooled-block caches (per
//! attention layer, full-length like the reference `StaticIndexedLayer` — the
//! 2048 budget is the SELECTION budget, not the cache size; finding recorded
//! from transformers 5.16.1 cache_utils `update_indexer`).
//!
//! Loader rule (spec 2.1, binding): N=160 is the TARGET; the loader verifies
//! the full budget with MEASURED overheads at load time and auto-clamps N; it
//! refuses configurations that fall below the 200k context floor — it never
//! silently degrades context.

use crate::cuda;
use crate::geo::*;
use cudarc::driver::sys::CUdeviceptr;

/// The planner refusal text (spec 2.1) as a pure function, factored by #10b
/// (2026-09-13) so the refusal path carries a panic-message test that needs
/// no GPU and no container. The text is byte-identical to the inline panic
/// it replaces.
pub fn planner_refusal_msg(free0: u64, host_pinned_budget: u64) -> String {
    format!(
        "refusing config: no hot-set size fits BOTH the VRAM budget (free {:.2} GiB) and the host pinned budget ({:.1} GiB) — shrink the chunk/scratch, the PLE cache, or the keep-set (spec 2.1)",
        free0 as f64 / GIB,
        host_pinned_budget as f64 / GIB
    )
}

/// physical RAM that must stay free after the cold tier is pinned
/// (`CROW_RAM_MARGIN_GB`, default 3 GiB). One number for both readers: the
/// budget derived below and the pre-pin gate in `residency::build`.
pub fn ram_margin_bytes() -> u64 {
    env_parse::<u64>("CROW_RAM_MARGIN_GB").unwrap_or(3) << 30
}

/// The host pinned budget, DERIVED at boot instead of assumed (issue #15).
///
/// `cap` is the configured ceiling (`Config::host_pinned_budget`, 46 GiB by
/// default - the measured ceiling of the 64 GB machine this engine grew up on).
/// The budget is the smaller of that cap and what this host can really give,
/// `free_for_pin - margin`, where `free_for_pin` is the reclaimable-aware
/// figure of `cuda::free_physical_ram_parts` (NOT `MemAvailable`).
///
/// A small budget is not a refusal. It is an input to the two-sided loop in
/// `ThreeStates::allocate`, which RAISES N - more experts hot in VRAM, fewer
/// pinned - until the cold tier fits, and refuses only when no N satisfies both
/// sides (`planner_refusal_msg`). Before this, a host with less free RAM than
/// the hard-coded 46 GiB pinned whatever the cap allowed and then died in the
/// gate at `residency::build` (the Windows boot of 2026-09-14,
/// `decode_out/hotfix-serve.log`: "refusing to pin 44.62 GiB with only
/// 46.47 GiB physical RAM free").
///
/// `CROW_PINNED_BUDGET_GB` pins the budget for a measurement and skips the
/// derivation entirely.
pub fn derive_host_pinned_budget(cap: u64, log: &mut dyn FnMut(&str)) -> u64 {
    let gib = |b: u64| b as f64 / GIB;
    let ram = cuda::free_physical_ram_parts();
    let (free_for_pin, mem_available) = (ram.free_for_pin, ram.mem_available);
    let margin = ram_margin_bytes();
    let (budget, basis) = match env_parse::<u64>("CROW_PINNED_BUDGET_GB") {
        Some(g) => (g << 30, "CROW_PINNED_BUDGET_GB".to_string()),
        // free_for_pin == 0 means the query failed: keep the configured cap,
        // the pre-pin gate in residency::build is then the only guard left
        None if free_for_pin == 0 => (cap, "configured cap, free RAM unknown".to_string()),
        None => {
            let room = free_for_pin.saturating_sub(margin);
            if room < cap {
                (room, format!("free for pinning {:.2} GiB - margin {:.0} GiB", gib(free_for_pin), gib(margin)))
            } else {
                (cap, "configured cap".to_string())
            }
        }
    };
    // #15 follow-up: the driver's pinned pool counts as free only while we are the
    // only CUDA process; with another one alive the figure IS MemAvailable
    let basis_ram = if ram.other_cuda { " (another CUDA process is alive: using MemAvailable)" } else { "" };
    log(&format!(
        "[budget] host pinned budget {:.2} GiB ({basis}); free for pinning {:.2} GiB{basis_ram}, MemAvailable {:.2} GiB, cap {:.2} GiB",
        gib(budget), gib(free_for_pin), gib(mem_available), gib(cap)
    ));
    budget
}

pub struct StateSizes {
    pub kv_bytes: u64,
    pub qsa_keys_bytes: u64,
    /// rows of the raw indexer-key ring per attention layer (CROW_QSA_FULL=1:
    /// the full context, the pre-2026-09-05 layout)
    pub qsa_ring_rows: usize,
    pub qsa_pooled_bytes: u64,
    pub gdn_s_bytes: u64,
    pub gdn_conv_bytes: u64,
    pub rope_bytes: u64,
}

impl StateSizes {
    /// all byte counts derived from geometry + context (measured shapes)
    pub fn plan(context: usize, kv: KvDtype, prompt_chunk: usize) -> StateSizes {
        let bpv = kv.byte_per_value() as u64;
        // raw keys are consumed by pool4_cache right after they are appended
        // (the pooled per-block keys are the long-lived cache): a 4-aligned ring
        // of chunk + 4 rows holds every row a chunk can still pool (up to 3 rows
        // of the previous chunk's incomplete block) — measured 2026-09-05
        let ring = if std::env::var("CROW_QSA_FULL").as_deref() == Ok("1") {
            context
        } else {
            ((prompt_chunk + QSA_COMPRESS + QSA_COMPRESS - 1) / QSA_COMPRESS * QSA_COMPRESS).min(context)
        };
        StateSizes {
            kv_bytes: (ATTN_LAYERS * 2 * NKV * AHD * context) as u64 * bpv,
            qsa_keys_bytes: (ATTN_LAYERS * ring * QSA_HIDD) as u64 * 4,
            qsa_ring_rows: ring,
            qsa_pooled_bytes: (ATTN_LAYERS * ((context + QSA_COMPRESS - 1) / QSA_COMPRESS) * QSA_HIDD) as u64 * 4,
            gdn_s_bytes: (GDN_LAYERS * GDN_VHEADS * GD * GD) as u64 * 4,
            gdn_conv_bytes: (GDN_LAYERS * GDN_CONV * 3) as u64 * 4,
            rope_bytes: (context * ROPE_PAIRS * 2) as u64 * 4,
        }
    }
}

pub struct ThreeStates {
    pub context: usize,
    pub kv: KvDtype,
    pub kv_buf: CUdeviceptr,      // [12][2][nkv][t][256] — one accounting
    pub qsa_keys: Vec<CUdeviceptr>, // [12][ring][128] f32 (row = pos % ring)
    pub qsa_ring_rows: usize,
    pub qsa_pooled: Vec<CUdeviceptr>, // [12][ceil(t/4)][128] f32
    pub gdn_s: Vec<CUdeviceptr>,    // [36][48*128*128] f32
    pub gdn_conv: Vec<CUdeviceptr>, // [36][10240*3] f32
    pub cos: CUdeviceptr,
    pub sin: CUdeviceptr,
    pub sizes: StateSizes,
    pub report: AllocReport,
}

#[derive(Default, Clone)]
pub struct AllocReport {
    pub lines: Vec<String>,
    pub total_bytes: u64,
    pub effective_n: usize,
}

impl ThreeStates {
    /// verify plan + allocate. `pending_bytes` is the loader's measured
    /// non-state footprint (dense weights + hot experts + PLE cache + graph
    /// slack); the deficit clamps N, the context floor refuses configs.
    pub unsafe fn allocate(
        cfg: &Config,
        pending_bytes: u64, // planned but NOT yet resident (dense already sits inside free0)
        expert_bytes_per_n_unit: u64,
        // host side: bytes per hot-set unit in the PINNED tier (record size of a
        // low-bit tier, else the NVFP4 slab size) and whether the tier is FULL
        // (every expert pinned: constant size, independent of N)
        cold_bytes_per_n_unit: u64,
        cold_fixed: bool,
    ) -> (ThreeStates, AllocReport) {
        assert!(
            cfg.context >= CONTEXT_FLOOR,
            "refusing config: context {} below the 200k floor (spec 0.2)",
            cfg.context
        );
        let mut rep = AllocReport::default();
        let total = cuda::total_vram_bytes();
        let free0 = cuda::free_vram_bytes();
        rep.lines.push(format!(
            "VRAM total {:.2} GiB, free at start {:.2} GiB",
            total as f64 / GIB,
            free0 as f64 / GIB
        ));

        // auto-clamp N with measured numbers, never the context (spec 2.1).
        // TWO-sided: VRAM lowers N, the HOST pinned budget RAISES it (fewer
        // cold experts) — measured host ceiling ~48.5 GB on this machine.
        let mut n = cfg.n_hot;
        let sizes = StateSizes::plan(cfg.context, cfg.kv, cfg.prompt_chunk);
        // the state bytes do not depend on N (the hot-expert count): one plan for the whole clamp loop
        let states_bytes = sizes.kv_bytes + sizes.qsa_keys_bytes + sizes.qsa_pooled_bytes
            + sizes.gdn_s_bytes + sizes.gdn_conv_bytes + sizes.rope_bytes;
        // Termination guard (2026-09-04): when VRAM pushes N down and the host
        // budget pushes it up, no N is feasible. Without this the loop
        // oscillated forever and grew `rep.lines` without bound -> the whole
        // machine froze from RAM exhaustion (chunk 1024 on the M container).
        let mut went_down = false;
        let mut went_up = false;
        let mut iters = 0u32;
        let spare = cfg.adapt.spare; // #17: from the policy in geo.rs, not the env
        loop {
            let sum = states_bytes + pending_bytes + n as u64 * expert_bytes_per_n_unit;
            let cold = (if cold_fixed { E } else { E - n.min(E) + spare }) as u64 * cold_bytes_per_n_unit;
            if sum + SAFETY < free0 && cold <= cfg.host_pinned_budget {
                break;
            }
            if n == N_MIN {
                break;
            }
            iters += 1;
            if (went_down && went_up) || iters > 2 * E as u32 {
                rep.lines.push(format!(
                    "no feasible N: VRAM allows at most N={} while the host pinned budget needs more — refusing",
                    n
                ));
                // #10b: the message moved into planner_refusal_msg (tested)
                panic!("{}", planner_refusal_msg(free0, cfg.host_pinned_budget));
            }
            if sum + SAFETY >= free0 {
                went_down = true;
                n -= 1;
                if n % 8 == 0 {
                    rep.lines.push(format!(
                        "VRAM budget over by {:.0} MB at N={} — clamping",
                        (sum + SAFETY - free0) as f64 / MIB,
                        n
                    ));
                }
            } else {
                went_up = true;
                n += 1;
                if n % 8 == 0 {
                    rep.lines.push(format!(
                        "host pinned tier over by {:.0} MB at N={} — raising N",
                        (cold - cfg.host_pinned_budget) as f64 / MIB,
                        n
                    ));
                }
            }
            if cfg_n_dbg() {
                tracing::info!(target: "manager", "[clamp] n={n} vram_sum={:.0} MB cold={:.0} MB free0={:.0} MB",
                    states_bytes as f64 / MIB,
                    ((if cold_fixed { E } else { E - n.min(E) + spare }) as u64 * cold_bytes_per_n_unit) as f64 / MIB,
                    free0 as f64 / MIB);
            }
        }
        let cold_final = (if cold_fixed { E } else { E - n.min(E) + spare }) as u64 * cold_bytes_per_n_unit;
        if cold_final > cfg.host_pinned_budget {
            panic!(
                "refusing config: hot set N={n} would pin {:.1} GiB cold > budget {:.1} GiB — no feasible N (spec 2.1)",
                cold_final as f64 / GIB,
                cfg.host_pinned_budget as f64 / GIB
            );
        }
        if n < cfg.n_hot {
            rep.lines.push(format!(
                "loader auto-clamped hot set: N {} -> {} (measured budget, spec 2.6)",
                cfg.n_hot, n
            ));
        }
        if n == N_MIN {
            let sum = sizes.kv_bytes
                + pending_bytes
                + N_MIN as u64 * expert_bytes_per_n_unit;
            if sum + SAFETY > free0 {
                panic!(
                    "refusing config: even N={N_MIN} does not fit context {} states (need {:.2} GiB, free {:.2} GiB)",
                    cfg.context,
                    sum as f64 / GIB,
                    free0 as f64 / GIB
                );
            }
        }

        // ---- allocate (the real allocations ARE the measurement) ----
        let kv_buf = cuda::alloc_zeroed(sizes.kv_bytes as usize);
        rep.lines.push(format!(
            "KV        {:9.1} MB  (12 layers × 2 kv-heads × 256 × {context} × {})",
            sizes.kv_bytes as f64 / MIB,
            cfg.kv.name(),
            context = cfg.context
        ));
        let mut qsa_keys = Vec::with_capacity(ATTN_LAYERS);
        for _ in 0..ATTN_LAYERS {
            qsa_keys.push(cuda::alloc_zeroed((sizes.qsa_ring_rows * QSA_HIDD * 4) as usize));
        }
        rep.lines.push(format!(
            "QSA keys  {:9.1} MB  (12 layers × {} × 128 f32 — raw-key ring, pooled cache stays full-length)",
            sizes.qsa_keys_bytes as f64 / MIB,
            sizes.qsa_ring_rows
        ));
        let mut qsa_pooled = Vec::with_capacity(ATTN_LAYERS);
        let cap_blocks = (cfg.context + 3) / 4;
        for _ in 0..ATTN_LAYERS {
            qsa_pooled.push(cuda::alloc_zeroed((cap_blocks * QSA_HIDD * 4) as usize));
        }
        rep.lines.push(format!(
            "QSA pooled{:9.1} MB  (12 layers × {} blocks × 128 f32)",
            sizes.qsa_pooled_bytes as f64 / MIB,
            cap_blocks
        ));
        let mut gdn_s = Vec::with_capacity(GDN_LAYERS);
        for _ in 0..GDN_LAYERS {
            gdn_s.push(cuda::alloc_zeroed((GDN_VHEADS * GD * GD * 4) as usize));
        }
        let mut gdn_conv = Vec::with_capacity(GDN_LAYERS);
        for _ in 0..GDN_LAYERS {
            gdn_conv.push(cuda::alloc_zeroed((GDN_CONV * 3 * 4) as usize));
        }
        rep.lines.push(format!(
            "GDN state {:9.1} MB  (36 × S[48][128][128] + conv[10240][3], f32, fixed)",
            (sizes.gdn_s_bytes + sizes.gdn_conv_bytes) as f64 / MIB
        ));
        let mut cos_h = vec![0f32; cfg.context * ROPE_PAIRS];
        let mut sin_h = vec![0f32; cfg.context * ROPE_PAIRS];
        for t in 0..cfg.context {
            for j in 0..ROPE_PAIRS {
                let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
                let f = t as f32 * inv;
                cos_h[t * ROPE_PAIRS + j] = f.cos();
                sin_h[t * ROPE_PAIRS + j] = f.sin();
            }
        }
        let cos = cuda::to_f32_dev(&cos_h);
        let sin = cuda::to_f32_dev(&sin_h);
        drop(cos_h);
        drop(sin_h);
        rep.lines.push(format!(
            "RoPE tbl  {:9.1} MB  ({} positions × 32 pairs × cos+sin)",
            sizes.rope_bytes as f64 / MIB,
            cfg.context
        ));

        let free1 = cuda::free_vram_bytes();
        let measured = free0 - free1;
        rep.total_bytes = measured;
        rep.effective_n = n;
        rep.lines.push(format!(
            "states+dense+hot measured in VRAM: {:.1} MiB (planned {:.1} MiB, N={n})",
            measured as f64 / MIB,
            (sizes.kv_bytes
                + sizes.qsa_keys_bytes
                + sizes.qsa_pooled_bytes
                + sizes.gdn_s_bytes
                + sizes.gdn_conv_bytes
                + sizes.rope_bytes
                + pending_bytes
                + n as u64 * expert_bytes_per_n_unit) as f64 / MIB
        ));

        (
            ThreeStates {
                context: cfg.context,
                kv: cfg.kv,
                kv_buf,
                qsa_keys,
                qsa_ring_rows: sizes.qsa_ring_rows,
                qsa_pooled,
                gdn_s,
                gdn_conv,
                cos,
                sin,
                sizes,
                report: rep.clone(),
            },
            rep,
        )
    }

    pub unsafe fn kv_row_ptr(&self, layer: usize, is_k: bool, kvh: usize, slot: usize) -> u64 {
        let b = (self.kv.byte_per_value()) as u64;
        self.kv_buf as u64
            + ((layer * 2 + if is_k { 0 } else { 1 }) * NKV * self.context
                + kvh * self.context
                + slot) as u64
            * AHD as u64
            * b
    }
}

pub const SAFETY: u64 = 512 << 20; // launch pools, scratch, telemetry slack
pub const N_MIN: usize = 32;

impl Drop for ThreeStates {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before ThreeStates"); }
        unsafe {
            cuda::free_dev(&mut self.kv_buf);
            for v in self.qsa_keys.iter_mut() { cuda::free_dev(v); }
            for v in self.qsa_pooled.iter_mut() { cuda::free_dev(v); }
            for v in self.gdn_s.iter_mut() { cuda::free_dev(v); }
            for v in self.gdn_conv.iter_mut() { cuda::free_dev(v); }
            cuda::free_dev(&mut self.cos);
            cuda::free_dev(&mut self.sin);
        }
    }
}

fn cfg_n_dbg() -> bool {
    std::env::var("ENGINE_DEBUG_SYNC").is_ok()
}

#[cfg(test)]
mod tests_10b {
    use super::planner_refusal_msg;

    /// #10b gate 5: the planner refusal path keeps its panic-message text
    /// (manager.rs:137 of the F52 record), asserted without a GPU.
    #[test]
    fn planner_refusal_message_names_both_budgets_and_the_escape_hatch() {
        let m = planner_refusal_msg(20 * (1u64 << 30), 46 * (1u64 << 30));
        assert!(
            m.starts_with("refusing config: no hot-set size fits BOTH the VRAM budget"),
            "message must name the VRAM budget first, got: {m}"
        );
        assert!(m.contains("free 20.00 GiB"), "free GiB formatting moved: {m}");
        assert!(
            m.contains("the host pinned budget (46.0 GiB)"),
            "pinned budget formatting moved: {m}"
        );
        assert!(
            m.contains("shrink the chunk/scratch, the PLE cache, or the keep-set (spec 2.1)"),
            "escape-hatch clause moved: {m}"
        );
    }
}
