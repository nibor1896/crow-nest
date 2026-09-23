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
use crate::meta::RopeKind;
use crate::meta::RopeScaling;
use cudarc::driver::sys::CUdeviceptr;

// ---------------- #96: rope scaling (YaRN / NTK / linear) ----------------

/// How the per-pair angle blends, from the `rope_scaling` the boot parsed.
/// `Plain` is the default path; `Interp` is naive linear scaling; `Yarn`
/// carries (freq_scale, corr_lo, corr_hi) — the low/high-frequency ramp.
enum Blend {
    Plain,
    Interp(f32),
    Yarn(f32, f32, f32),
}

/// llama.cpp `rope_yarn_ramp` verbatim: 1 - clamp((pair - low) / max(0.001,
/// high - low), 0, 1). 1 at and below `low` (extrapolate), 0 at and above
/// `high` (interpolate), linear between.
fn yarn_ramp(pair: usize, low: f32, high: f32) -> f32 {
    let y = (pair as f32 - low) / (high - low).max(0.001);
    1.0 - y.clamp(0.0, 1.0)
}

/// The boot RoPE table (#96), pure host f32 math so the byte-identity gate can
/// build it both ways in a unit test with no GPU and no container.
///
/// The `None` / default arm is the loop this replaces, VERBATIM — theta 1e7
/// (`10_000_000f32.powf(-(2j)/64)`), `t · inv`, f32 cos/sin, t-major layout —
/// pinned byte-identical by `the_unscaled_table_is_byte_identical_to_the_loop`.
///
/// The scaled arms are the llama.cpp reference math (ggml `rope_yarn`, MIT,
/// jquesnelle/yarn — the YaRN paper's implementation), with the YaRN
/// extrapolation factor pinned at the llama.cpp default 1.0:
/// - **yarn**: per pair j, `theta = interp·(1−mix) + extrap·mix` with
///   `interp = freq_scale·extrap` and `mix = ramp(j)` over the corr range from
///   beta_fast/beta_slow — the high-frequency pairs below `lo` keep their
///   trained angles, everything above `hi` interpolates;
/// - **linear**: `freq_scale·extrap` on every pair (the naive form the paper
///   shows collapsing — kept because it is one line and one config away);
/// - **ntk-aware**: theta rewritten to `base·factor^(dim/(dim−2))`, every pair
///   otherwise untouched (what the llama.cpp converter bakes into the base).
///
/// The mscale is deliberately NOT folded in here: it belongs to the attention
/// temperature, which lives in the kernels (`d_attn_scale`, set once at boot).
///
/// NOTE the table is shared: `rope`/`rope_p` (text attention) AND `rope64`
/// (the QSA indexer keys) read these cos/sin rows, so a scaled table scales
/// both — the single-table consequence, flagged in docs/acceptance/issue-96.md.
pub fn build_rope_table(context: usize, scaling: Option<&RopeScaling>) -> (Vec<f32>, Vec<f32>) {
    let dim = 2 * ROPE_PAIRS; // 64 rotary dims of the 256-dim head
    let base = match scaling {
        Some(s) if s.kind == RopeKind::NtkAware => s.ntk_base(dim, 10_000_000f64) as f32,
        _ => 10_000_000f32,
    };
    let blend = match scaling {
        None | Some(&RopeScaling { kind: RopeKind::Default, .. }) => Blend::Plain,
        Some(&RopeScaling { kind: RopeKind::Linear, factor, .. }) => {
            Blend::Interp((1.0 / factor) as f32)
        }
        Some(s) if s.kind == RopeKind::Yarn => {
            let (lo, hi) = s.yarn_corr_range(dim, 10_000_000f64);
            Blend::Yarn(s.freq_scale(), lo, hi)
        }
        Some(_) => Blend::Plain, // NtkAware: handled by the base above
    };
    let mut cos_h = vec![0f32; context * ROPE_PAIRS];
    let mut sin_h = vec![0f32; context * ROPE_PAIRS];
    for t in 0..context {
        for j in 0..ROPE_PAIRS {
            let inv = base.powf(-(2.0 * j as f32) / dim as f32);
            let extrap = t as f32 * inv;
            let f = match blend {
                Blend::Plain => extrap,
                Blend::Interp(fs) => fs * extrap,
                Blend::Yarn(fs, lo, hi) => {
                    let mix = yarn_ramp(j, lo, hi); // ext_factor 1.0 (llama.cpp yarn default)
                    (fs * extrap) * (1.0 - mix) + extrap * mix
                }
            };
            cos_h[t * ROPE_PAIRS + j] = f.cos();
            sin_h[t * ROPE_PAIRS + j] = f.sin();
        }
    }
    (cos_h, sin_h)
}

/// #96 phase 3 — the exceed-training-context warning, a pure function so every
/// side of the decision is unit-testable without a GPU. Fires when the
/// effective context runs past the positions the checkpoint was trained on AND
/// no scaling is armed: positions beyond the training window walk RoPE
/// frequencies the model never saw, and the failure mode is silent quality
/// collapse — llama.cpp's "n_ctx > n_ctx_train … quality will be degraded"
/// discipline, one loud boot WARN, not an error.
pub fn exceed_training_warning(
    context: usize,
    training: Option<u64>,
    scaling: Option<&RopeScaling>,
) -> Option<String> {
    let training = training?;
    let armed = scaling.is_some_and(|s| s.kind != RopeKind::Default);
    if armed || context as u64 <= training {
        return None;
    }
    Some(format!(
        "context {} exceeds the {} positions this checkpoint was trained on and no rope_scaling is armed - \
output quality WILL be degraded past position {training} (issue #96: configure rope_scaling in the checkpoint config)",
        context, training
    ))
}

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
    // #103: the driver's page pool counts as free (it is reclaimable), the
    // driver memory live processes still map does not - whether or not another CUDA
    // process is alive (the old MemAvailable fallback refused boots next to a pool)
    let other = if ram.other_cuda { ", another CUDA process is alive" } else { "" };
    log(&format!(
        "[budget] host pinned budget {:.2} GiB ({basis}); free for pinning {:.2} GiB, MemAvailable {:.2} GiB, cap {:.2} GiB; NVIDIA driver pages {:.2} GiB of which {:.2} GiB mapped by live processes (subtracted){other}",
        gib(budget), gib(free_for_pin), gib(mem_available), gib(cap), gib(ram.driver_held), gib(ram.driver_live)
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
        // ---- RoPE table (#96: the builder is factored out below; None scaling
        // builds the byte-identical table the inline loop always built) ----
        // phase 3 FIRST (the cheap hazard close): effective context past the
        // training window with no scaling armed is one loud boot WARN — the
        // llama.cpp "quality will be degraded" discipline. `None` training
        // context (a boot that never saw a config.json — the selftest package)
        // stays silent.
        let scaling = crate::meta::boot_rope_scaling();
        if let Some(w) = exceed_training_warning(cfg.context, crate::meta::boot_training_context(), scaling.as_ref()) {
            tracing::warn!(target: "rope", "[rope] {w}");
        }
        let (cos_h, sin_h) = build_rope_table(cfg.context, scaling.as_ref());
        let cos = cuda::to_f32_dev(&cos_h);
        let sin = cuda::to_f32_dev(&sin_h);
        drop(cos_h);
        drop(sin_h);
        rep.lines.push(format!(
            "RoPE tbl  {:9.1} MB  ({} positions × 32 pairs × cos+sin)",
            sizes.rope_bytes as f64 / MIB,
            cfg.context
        ));
        // the armed line: what the config asked for and what was derived from
        // it, next to the table it changed (silent — nothing is armed by default)
        if let Some(s) = scaling.filter(|s| s.kind != crate::meta::RopeKind::Default) {
            let detail = match s.kind {
                crate::meta::RopeKind::Yarn => {
                    let (lo, hi) = s.yarn_corr_range(2 * ROPE_PAIRS, 10_000_000f64);
                    format!(
                        "YaRN: factor {} ({} training positions -> {}), corr dims [{lo}, {hi}] of {} (beta {} / {}), mscale {:.4} on the attention scale",
                        s.factor, s.original_context, cfg.context, 2 * ROPE_PAIRS, s.beta_fast, s.beta_slow, s.mscale()
                    )
                }
                crate::meta::RopeKind::Linear => {
                    format!("linear: factor {} (every pair interpolated by 1/{})", s.factor, s.factor)
                }
                crate::meta::RopeKind::NtkAware => {
                    format!("ntk-aware: theta 1e7 -> {:.0}", s.ntk_base(2 * ROPE_PAIRS, 10_000_000f64))
                }
                crate::meta::RopeKind::Default => unreachable!("filtered above"),
            };
            rep.lines.push(format!("[rope] rope_scaling armed — {detail}"));
        }

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

/// #72: VRAM that must STILL be free once every boot allocation is resident.
///
/// `SAFETY` (512 MiB) is what the clamp loop leaves above the planned sum; this
/// is what the loader VERIFIES afterwards, against the card. It covers only the
/// allocations no engine code makes: the decode graph the first `decode_step`
/// captures, the driver's launch pools and the local-memory backing store. The
/// image path used to live in here too, which is the whole of #72 - on robin's
/// 2026-09-18 session the tower found 35.7 MiB of it left.
pub const POST_PLAN_FLOOR: u64 = 256 << 20;

/// where a boot allocation lives; the #72 audit found the biggest suspect
/// (the prefix-cache snapshots, 3 x 124.6 MiB) on the HOST side, not the card
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Site {
    Vram,
    Host,
}

/// #72: the allocations that used to happen AFTER the planner had chosen N, listed
/// with their bytes and their side of the bus. Pure bookkeeping - `Engine::load`
/// fills it with the numbers it really allocated and prints one `[budget]` line
/// from it, so a lazily allocated buffer that is not in this list is a bug.
#[derive(Default)]
pub struct PostPlan {
    pub items: Vec<(String, u64, Site)>,
}

impl PostPlan {
    pub fn new() -> PostPlan {
        PostPlan { items: Vec::new() }
    }
    /// a device allocation the boot now HOLDS
    pub fn vram(&mut self, what: &str, bytes: u64) -> &mut PostPlan {
        self.items.push((what.to_string(), bytes, Site::Vram));
        self
    }
    /// host RAM, named so the reader does not look for it on the card
    pub fn host(&mut self, what: &str, bytes: u64) -> &mut PostPlan {
        self.items.push((what.to_string(), bytes, Site::Host));
        self
    }
    pub fn vram_bytes(&self) -> u64 {
        self.items.iter().filter(|i| i.2 == Site::Vram).map(|i| i.1).sum()
    }
    pub fn host_bytes(&self) -> u64 {
        self.items.iter().filter(|i| i.2 == Site::Host).map(|i| i.1).sum()
    }
    /// the `[budget]` line: what is held on the card, then what is host RAM only
    pub fn line(&self) -> String {
        let mb = |b: u64| b as f64 / MIB;
        let names = |site: Site| -> String {
            self.items
                .iter()
                .filter(|i| i.2 == site)
                .map(|i| format!("{} {:.1} MB", i.0, mb(i.1)))
                .collect::<Vec<_>>()
                .join(" + ")
        };
        let vram = names(Site::Vram);
        let host = names(Site::Host);
        let vram = if vram.is_empty() { "nothing".to_string() } else { vram };
        format!(
            "post-plan allocations held at boot: {vram} = {:.1} MB VRAM; host RAM only (never on the card): {}",
            mb(self.vram_bytes()),
            if host.is_empty() { "nothing".to_string() } else { host }
        )
    }
}

/// #72: the floor check the loader prints after everything is resident. `free`
/// is what the card reports once the load is done; below `floor` the boot says
/// so loudly, because the decode graph is captured after this line.
pub fn headroom_line(free: u64, floor: u64) -> String {
    let gib = |b: u64| b as f64 / GIB;
    if free >= floor {
        format!(
            "free VRAM after load {:.2} GiB >= floor {:.2} GiB — the image path is held, not borrowed from here",
            gib(free), gib(floor)
        )
    } else {
        format!(
            "SHORT: free VRAM after load {:.2} GiB is BELOW the floor {:.2} GiB — the decode graph and the driver pools still have to fit; lower the hot-set target or the context",
            gib(free), gib(floor)
        )
    }
}

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
mod tests_72 {
    //! #72: the post-plan ledger and the headroom floor, as pure arithmetic.
    //! The bug they close: the planner set 277.3 MB aside for the image path and
    //! nothing held it, so by the seventh round of robin's 2026-09-18 session the
    //! first image request found 35.7 MiB free and answered 503.
    use super::*;

    fn ledger() -> PostPlan {
        let mut p = PostPlan::new();
        p.vram("vit tower scratch", 239_599_616);
        p.vram("vit mrope span", 51_200_000);
        p.vram("device sampler", crate::gen::sampler_bytes());
        p.host("prefix cache (3 snapshots)", 3 * 130_646_016);
        p.host("vit image cache (CROW_VIT_CACHE_MB, LRU)", 256 << 20);
        p
    }

    #[test]
    fn the_post_plan_vram_total_is_the_reserve_plus_the_sampler() {
        let p = ledger();
        assert_eq!(p.vram_bytes(), 239_599_616 + 51_200_000 + crate::gen::sampler_bytes());
        // the reserve of record, 277.3 MB, plus ~0.7 MB of sampler
        // (#83, 2026-09-20: params grew 16 -> 36 B for min_p + ln(min_p);
        // #84 the same day: the windowed penalties added counts [V] u16 +
        // the 1026-i32 ring, so the pin moved 281_112 -> 281_132 -> 781_876)
        assert_eq!(p.vram_bytes(), 291_581_492);
        assert_eq!(crate::gen::sampler_bytes(), 781_876);
    }

    /// the biggest post-plan allocation of the process is the prefix cache, and it
    /// is HOST RAM (`cache.rs`: `Vec<f32>`): it must never be counted against the
    /// card, or the planner would pay 392 MB of hot experts for nothing.
    #[test]
    fn the_prefix_cache_snapshots_are_host_ram_and_stay_out_of_the_vram_total() {
        let p = ledger();
        assert_eq!(p.host_bytes(), 3 * 130_646_016 + (256 << 20));
        assert!(p.vram_bytes() < 3 * 130_646_016 + (256 << 20));
        let line = p.line();
        assert!(line.starts_with("post-plan allocations held at boot:"), "the label moved: {line}");
        assert!(line.contains("vit tower scratch 228.5 MB"), "the scratch entry moved: {line}");
        assert!(line.contains("vit mrope span 48.8 MB"), "the mrope entry moved: {line}");
        assert!(line.contains("= 278.1 MB VRAM"), "the VRAM total moved: {line}");
        assert!(line.contains("host RAM only (never on the card)"), "the host clause moved: {line}");
        assert!(line.contains("prefix cache (3 snapshots) 373.8 MB"), "the snapshot entry moved: {line}");
    }

    #[test]
    fn an_empty_ledger_says_nothing_rather_than_an_empty_list() {
        let p = PostPlan::new();
        assert_eq!(p.vram_bytes(), 0);
        let line = p.line();
        assert!(line.contains("held at boot: nothing = 0.0 MB VRAM"), "{line}");
        assert!(line.contains("never on the card): nothing"), "{line}");
    }

    #[test]
    fn the_headroom_floor_is_named_and_the_short_case_is_loud() {
        assert_eq!(POST_PLAN_FLOOR, 256 << 20);
        let ok = headroom_line(600 << 20, POST_PLAN_FLOOR);
        assert!(ok.starts_with("free VRAM after load 0.59 GiB >= floor 0.25 GiB"), "{ok}");
        assert!(ok.contains("the image path is held, not borrowed from here"), "{ok}");
        // the 35.7 MiB of the issue: below the floor, and the line says SHORT first
        let short = headroom_line(37_450_000, POST_PLAN_FLOOR);
        assert!(short.starts_with("SHORT:"), "{short}");
        assert!(short.contains("BELOW the floor 0.25 GiB"), "{short}");
    }
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

#[cfg(test)]
mod tests_96 {
    //! #96: the scaled rope-table builder and the exceed-training-context warn,
    //! pure host math — no GPU, no container. The first test is THE gate: the
    //! refactor that factored the boot table into `build_rope_table` changed
    //! not one byte of the default path, and a present-but-"default"
    //! rope_scaling object configures nothing either.
    use super::*;

    fn scaling(kind: RopeKind) -> RopeScaling {
        RopeScaling {
            kind,
            factor: 4.0,
            original_context: 262_144,
            beta_fast: 32.0,
            beta_slow: 1.0,
            attention_factor: None,
        }
    }

    /// the pre-#96 boot loop, copied VERBATIM from the manager.rs that stood
    /// before the refactor (theta 1e7, f32 powf/cos/sin, t-major): the oracle
    /// every byte-identity claim below is measured against
    #[test]
    fn the_unscaled_table_is_byte_identical_to_the_loop() {
        let context = 8192;
        let mut cos_ref = vec![0f32; context * ROPE_PAIRS];
        let mut sin_ref = vec![0f32; context * ROPE_PAIRS];
        for t in 0..context {
            for j in 0..ROPE_PAIRS {
                let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
                let f = t as f32 * inv;
                cos_ref[t * ROPE_PAIRS + j] = f.cos();
                sin_ref[t * ROPE_PAIRS + j] = f.sin();
            }
        }
        let (cos, sin) = build_rope_table(context, None);
        assert_eq!(cos, cos_ref, "the default cos table moved");
        assert_eq!(sin, sin_ref, "the default sin table moved");
        // a present-but-"default" rope_scaling object configures nothing: same bytes
        let (c2, s2) = build_rope_table(context, Some(&scaling(RopeKind::Default)));
        assert_eq!(c2, cos_ref, "a default rope_scaling must not move the table");
        assert_eq!(s2, sin_ref, "a default rope_scaling must not move the table");
    }

    /// YaRN shape, against the same oracle: pairs at/below the corr-range floor
    /// keep their angles BYTE-FOR-BYTE (the high-frequency bands YaRN exists to
    /// protect), pairs at/above the ceiling take the interpolated angle, and the
    /// corr range itself is the reference math for this checkpoint's dims
    #[test]
    fn yarn_extrapolates_low_pairs_and_interpolates_high_ones() {
        let s = scaling(RopeKind::Yarn);
        assert_eq!(s.yarn_corr_range(64, 1e7), (14.0, 22.0), "corr dims for dim 64 / base 1e7 / 262144 ctx / beta 32+1");
        let (cos_y, _) = build_rope_table(1024, Some(&s));
        let (cos_0, _) = build_rope_table(1024, None);
        for t in [0usize, 1, 100, 1023] {
            // j <= 14: ramp 1 -> pure extrapolation -> identical bytes
            for j in [0usize, 7, 13, 14] {
                assert_eq!(
                    cos_y[t * ROPE_PAIRS + j], cos_0[t * ROPE_PAIRS + j],
                    "pair {j} at t={t} is high frequency: YaRN keeps the trained angle"
                );
            }
            // j >= 22: ramp 0 -> pure interpolation by 1/factor
            for j in [22usize, 28, 31] {
                let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
                assert_eq!(
                    cos_y[t * ROPE_PAIRS + j],
                    ((t as f32 * inv) * 0.25f32).cos(),
                    "pair {j} at t={t} is low frequency: the interpolated angle"
                );
                // a witness only where the angles are wide enough to differ in
                // f32: cos(x) - cos(x/4) ~ 3x²/8, invisible below ~1e-4 rad
                // (t=0 maps every scaling to angle 0; tiny t x low freq ditto)
                if t as f32 * inv > 1e-3 {
                    assert_ne!(cos_y[t * ROPE_PAIRS + j], cos_0[t * ROPE_PAIRS + j]);
                }
            }
        }
        // the ramp band (14 < j < 22) is neither extreme
        let (t, j) = (1023usize, 18usize);
        let inv = 10_000_000f32.powf(-(2.0 * j as f32) / 64.0);
        let extrap = t as f32 * inv;
        let blended = (0.25f32 * extrap) * 0.5 + extrap * 0.5;
        assert_eq!(cos_y[t * ROPE_PAIRS + j], blended.cos(), "the ramp midpoint is the half-and-half angle");
    }

    /// linear interpolates EVERY pair (pair 0 included — exactly the collapse
    /// the YaRN paper shows); ntk-aware leaves the angles alone and rewrites
    /// the theta the pairs are computed from
    #[test]
    fn linear_scales_every_pair_and_ntk_rewrites_the_base() {
        let (cos_l, _) = build_rope_table(64, Some(&scaling(RopeKind::Linear)));
        let (cos_0, _) = build_rope_table(64, None);
        let (t, j) = (63usize, 0usize);
        assert_eq!(cos_l[t * ROPE_PAIRS + j], (63f32 * 0.25f32).cos(), "linear: even pair 0 interpolates");
        assert_ne!(cos_l[t * ROPE_PAIRS + j], cos_0[t * ROPE_PAIRS + j]);
        let s = scaling(RopeKind::NtkAware);
        let b = s.ntk_base(64, 1e7) as f32;
        let (cos_n, _) = build_rope_table(64, Some(&s));
        // pair 31 at t=63 is NOT a usable witness for the base rewrite: both
        // angles are ~1e-5 rad and f32 cos rounds both to exactly 1.0. Pair 8
        // has base^(-0.25) scale angles O(1) rad, where the rewrite is visible.
        let (t, j) = (63usize, 8usize);
        let inv8 = b.powf(-(2.0 * 8f32) / 64.0);
        assert_eq!(cos_n[t * ROPE_PAIRS + j], (63f32 * inv8).cos(), "ntk-aware: theta' = 1e7·4^(64/62) rewrites every pair's base");
        assert_ne!(cos_n[t * ROPE_PAIRS + j], cos_0[t * ROPE_PAIRS + j], "the rewritten base moves pair 8's angle");
        // and the far pair keeps the formula (the O(1e-5) angle both bases)
        let inv31 = b.powf(-(2.0 * 31f32) / 64.0);
        assert_eq!(cos_n[63 * ROPE_PAIRS + 31], (63f32 * inv31).cos());
    }

    /// phase 3: the warn fires only when the effective context exceeds the
    /// training context AND nothing is armed — every other side is silent
    #[test]
    fn the_exceed_training_warning_fires_only_unscaled_and_oversized() {
        // today's shape: 200k boot against 262144 training positions — silent
        assert!(exceed_training_warning(200_000, Some(262_144), None).is_none());
        // equal context is not exceeded
        assert!(exceed_training_warning(262_144, Some(262_144), None).is_none());
        // the loud line: oversized and unscaled
        let w = exceed_training_warning(300_000, Some(262_144), None).expect("oversized + unscaled must warn");
        assert!(w.contains("300000 exceeds the 262144 positions"), "{w}");
        assert!(w.contains("quality WILL be degraded"), "{w}");
        assert!(w.contains("issue #96"), "{w}");
        // armed yarn: silent — that is what the scaling is for
        assert!(exceed_training_warning(300_000, Some(262_144), Some(&scaling(RopeKind::Yarn))).is_none());
        // a "default" scaling object configures nothing: the warn stands
        assert!(exceed_training_warning(300_000, Some(262_144), Some(&scaling(RopeKind::Default))).is_some());
        // no config seen at all (the selftest package): never a guessed threshold
        assert!(exceed_training_warning(300_000, None, None).is_none());
    }
}
