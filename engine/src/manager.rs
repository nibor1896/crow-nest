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
pub fn build_rope_table(geo: &Geo, context: usize, scaling: Option<&RopeScaling>) -> (Vec<f32>, Vec<f32>) {
    // C3: pairs and theta from the model's Geo (Flash-Next: 32 pairs = 64 rotary
    // dims of the 256-dim head, theta 1e7; `1e7f64 as f32` is exactly 10_000_000f32)
    let pairs = geo.rope_pairs;
    let dim = 2 * pairs;
    let base = match scaling {
        Some(s) if s.kind == RopeKind::NtkAware => s.ntk_base(dim, geo.rope_theta) as f32,
        _ => geo.rope_theta as f32,
    };
    let blend = match scaling {
        None | Some(&RopeScaling { kind: RopeKind::Default, .. }) => Blend::Plain,
        Some(&RopeScaling { kind: RopeKind::Linear, factor, .. }) => {
            Blend::Interp((1.0 / factor) as f32)
        }
        Some(s) if s.kind == RopeKind::Yarn => {
            let (lo, hi) = s.yarn_corr_range(dim, geo.rope_theta);
            Blend::Yarn(s.freq_scale(), lo, hi)
        }
        Some(_) => Blend::Plain, // NtkAware: handled by the base above
    };
    let mut cos_h = vec![0f32; context * pairs];
    let mut sin_h = vec![0f32; context * pairs];
    for t in 0..context {
        for j in 0..pairs {
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
            cos_h[t * pairs + j] = f.cos();
            sin_h[t * pairs + j] = f.sin();
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

/// The default of `CROW_RAM_MARGIN_GB`. 1 GiB since 2026-10-06 (was 3),
/// robin's decision: on Windows with a page file, Flash-Next boots at 1 GiB
/// with a loaded desktop and decodes as fast as at 3 GiB (40.3-41.1 vs
/// 39.0-39.9 tok/s, three ~30k-token prompts per arm; free RAM fell to 59 MB),
/// while 3 GiB refused a boot at 48.31 GiB free and 2 GiB refused three boots
/// at 12:23-12:29 the same day (46.97-48.13 GiB free at the start; the load
/// itself takes ~1.4 GiB after the budget check). crow-lab/runs/fn-ram-margin-win-20261006,
/// fn-boot-refusals-win-20261006.
pub const RAM_MARGIN_DEFAULT_GB: u64 = 1;

/// physical RAM that must stay free after the cold tier is pinned
/// (`CROW_RAM_MARGIN_GB`, default [`RAM_MARGIN_DEFAULT_GB`]). One number for
/// both readers: the budget derived below and the pre-pin gate in
/// `residency::build`.
pub fn ram_margin_bytes() -> u64 {
    env_parse::<u64>("CROW_RAM_MARGIN_GB").unwrap_or(RAM_MARGIN_DEFAULT_GB) << 30
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
    /// all byte counts derived from geometry + context (measured shapes); C3: the
    /// geometry is the model's `Geo`. C5: the QSA ring and pooled cache are the
    /// `Attn::Qsa` arm; full attention plans none of them (zero bytes, zero rows).
    pub fn plan(geo: &Geo, context: usize, kv: KvDtype, prompt_chunk: usize) -> StateSizes {
        // #88: the bytes of one KV row (a q8 row carries its scales: 272 B at head dim 256)
        let row = kv.row_bytes(geo.head_dim) as u64;
        let (attn_layers, gdn_layers) = (geo.attn_layers, geo.gdn_layers);
        let (qsa_keys_bytes, ring, qsa_pooled_bytes) = match geo.attn {
            Attn::Qsa { .. } => {
                let q = geo.qsa();
                // raw keys are consumed by pool4_cache right after they are appended
                // (the pooled per-block keys are the long-lived cache): a 4-aligned ring
                // of chunk + 4 rows holds every row a chunk can still pool (up to 3 rows
                // of the previous chunk's incomplete block) — measured 2026-09-05
                let ring = if std::env::var("CROW_QSA_FULL").as_deref() == Ok("1") {
                    context
                } else {
                    ((prompt_chunk + q.compress + q.compress - 1) / q.compress * q.compress).min(context)
                };
                (
                    (attn_layers * ring * q.hidd()) as u64 * 4,
                    ring,
                    (attn_layers * ((context + q.compress - 1) / q.compress) * q.hidd()) as u64 * 4,
                )
            }
            Attn::Full => (0, 0, 0),
        };
        StateSizes {
            kv_bytes: (attn_layers * 2 * geo.kv_heads * context) as u64 * row,
            qsa_keys_bytes,
            qsa_ring_rows: ring,
            qsa_pooled_bytes,
            gdn_s_bytes: (gdn_layers * geo.gdn_value_heads * geo.gdn_key_dim * geo.gdn_value_dim) as u64 * 4,
            gdn_conv_bytes: (gdn_layers * gdn_conv_state_len(geo)) as u64 * 4,
            rope_bytes: (context * geo.rope_pairs * 2) as u64 * 4,
        }
    }
}

/// C3: f32 of ONE GDN layer's recurrent state S, `[value heads][key dim][value dim]`
/// (Flash-Next `[48][128][128]`)
pub const fn gdn_s_state_len(geo: &Geo) -> usize {
    geo.gdn_value_heads * geo.gdn_key_dim * geo.gdn_value_dim
}

/// C3: f32 of ONE GDN layer's causal conv state, `[conv channels][conv_kernel - 1]`
/// (Flash-Next `[10240][3]`)
pub const fn gdn_conv_state_len(geo: &Geo) -> usize {
    geo.gdn_conv() * (geo.conv_kernel - 1)
}

/// C3: f32 of the PLE dilated conv state (`gen.rs` `Ple::state`), `[residual width][9]`
/// (Flash-Next `[10240][9]`; the 9 taps are the PLE kernel's own, pinned in kernels.rs).
/// C5: zero for a model without PLE (the `None` arm has no state).
pub const fn ple_state_len(geo: &Geo) -> usize {
    match geo.ple {
        Some(_) => geo.residual_width() * 9,
        None => 0,
    }
}

impl StateSizes {
    /// every state byte the planner sets aside before the hot set (KV at the
    /// config's dtype - bf16 doubles `kv_bytes`, #102, q8 is 17/16 of fp8, #88 -
    /// plus QSA, GDN, rope)
    pub fn total(&self) -> u64 {
        self.kv_bytes + self.qsa_keys_bytes + self.qsa_pooled_bytes
            + self.gdn_s_bytes + self.gdn_conv_bytes + self.rope_bytes
    }
}

pub struct ThreeStates {
    /// C3: the geometry the states were shaped for (a copy of the engine's `Geo`)
    pub geo: Geo,
    pub context: usize,
    pub kv: KvDtype,
    pub kv_buf: CUdeviceptr,      // [12][2][nkv][t][KvDtype::row_bytes(256)] — one accounting
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
    /// #110 follow-up: the render reserve GRANTED (bytes), <= the requested one
    pub render_reserve: u64,
}

impl ThreeStates {
    /// verify plan + allocate. `pending_bytes` is the loader's measured
    /// non-state footprint (dense weights + hot experts + PLE cache + graph
    /// slack); the deficit clamps N, the context floor refuses configs.
    pub unsafe fn allocate(
        cfg: &Config,
        geo: &Geo,
        pending_bytes: u64, // planned but NOT yet resident (dense already sits inside free0)
        expert_bytes_per_n_unit: u64,
        // host side: bytes per hot-set unit in the PINNED tier (record size of a
        // low-bit tier, else the NVFP4 slab size) and whether the tier is FULL
        // (every expert pinned: constant size, independent of N)
        cold_bytes_per_n_unit: u64,
        cold_fixed: bool,
        // #110 follow-up: the REQUESTED render reserve (NOT in `pending_bytes`);
        // granted best-effort by `grant_render_reserve`, never a boot refusal
        render_reserve_requested: u64,
    ) -> (ThreeStates, AllocReport) {
        assert!(
            cfg.context >= geo.context_floor,
            "refusing config: context {} below the {} floor (spec 0.2)",
            cfg.context,
            geo.context_floor
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
        let sizes = StateSizes::plan(geo, cfg.context, cfg.kv, cfg.prompt_chunk);
        // the state bytes do not depend on N (the hot-expert count): one plan for the whole clamp loop
        let states_bytes = sizes.total();
        // C5: the hot-set clamp is the `Ffn::Moe` arm of the planner; a dense FFN has no
        // hot set, so its plan is one sum (Crow #300 phase 2)
        let n = match geo.ffn {
            Ffn::Moe { experts, .. } => {
                let spare = cfg.adapt.spare; // #17: from the policy in geo.rs, not the env
                let base = ClampInput {
                    n_hot: cfg.n_hot,
                    experts,
                    states_bytes,
                    pending_bytes,
                    expert_bytes_per_n_unit,
                    cold_bytes_per_n_unit,
                    cold_fixed,
                    spare,
                    free0,
                    host_pinned_budget: cfg.host_pinned_budget,
                };
                let n = match grant_render_reserve(&base, render_reserve_requested) {
                    Ok(g) => {
                        rep.lines.extend(g.lines);
                        rep.lines.push(g.line);
                        rep.render_reserve = g.granted;
                        g.n
                    }
                    // the refusal WITHOUT any reserve: the planner's own, as before #110
                    Err(msg) => panic!("{msg}"),
                };
                // the rest of the plan counts the granted reserve as pending
                let pending_bytes = pending_bytes + rep.render_reserve;
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
                n
            }
            Ffn::Dense { .. } => {
                let (granted, lines) = dense_fit(free0, states_bytes, pending_bytes, render_reserve_requested, cfg.context)
                    .unwrap_or_else(|msg| panic!("{msg}"));
                rep.lines.extend(lines);
                rep.render_reserve = granted;
                0
            }
        };

        // ---- allocate (the real allocations ARE the measurement) ----
        let kv_buf = cuda::alloc_zeroed(sizes.kv_bytes as usize);
        rep.lines.push(format!(
            "KV        {:9.1} MB  ({} layers × {} kv-heads × {} × {context} × {})",
            sizes.kv_bytes as f64 / MIB,
            geo.attn_layers,
            geo.kv_heads,
            geo.head_dim,
            cfg.kv.name(),
            context = cfg.context
        ));
        // C5: the indexer's raw-key ring and pooled blocks are the `Attn::Qsa` arm;
        // full attention allocates neither (and prints no QSA line)
        let (qsa_keys, qsa_pooled) = match geo.attn {
            Attn::Qsa { .. } => {
                let q = geo.qsa();
                let mut qsa_keys = Vec::with_capacity(geo.attn_layers);
                for _ in 0..geo.attn_layers {
                    qsa_keys.push(cuda::alloc_zeroed((sizes.qsa_ring_rows * q.hidd() * 4) as usize));
                }
                rep.lines.push(format!(
                    "QSA keys  {:9.1} MB  ({} layers × {} × {} f32 — raw-key ring, pooled cache stays full-length)",
                    sizes.qsa_keys_bytes as f64 / MIB,
                    geo.attn_layers,
                    sizes.qsa_ring_rows,
                    q.hidd()
                ));
                let mut qsa_pooled = Vec::with_capacity(geo.attn_layers);
                let cap_blocks = cfg.context.div_ceil(q.compress);
                for _ in 0..geo.attn_layers {
                    qsa_pooled.push(cuda::alloc_zeroed((cap_blocks * q.hidd() * 4) as usize));
                }
                rep.lines.push(format!(
                    "QSA pooled{:9.1} MB  ({} layers × {} blocks × {} f32)",
                    sizes.qsa_pooled_bytes as f64 / MIB,
                    geo.attn_layers,
                    cap_blocks,
                    q.hidd()
                ));
                (qsa_keys, qsa_pooled)
            }
            Attn::Full => (Vec::new(), Vec::new()),
        };
        let mut gdn_s = Vec::with_capacity(geo.gdn_layers);
        for _ in 0..geo.gdn_layers {
            gdn_s.push(cuda::alloc_zeroed(gdn_s_state_len(geo) * 4));
        }
        let mut gdn_conv = Vec::with_capacity(geo.gdn_layers);
        for _ in 0..geo.gdn_layers {
            gdn_conv.push(cuda::alloc_zeroed(gdn_conv_state_len(geo) * 4));
        }
        rep.lines.push(format!(
            "GDN state {:9.1} MB  ({} × S[{}][{}][{}] + conv[{}][{}], f32, fixed)",
            (sizes.gdn_s_bytes + sizes.gdn_conv_bytes) as f64 / MIB,
            geo.gdn_layers,
            geo.gdn_value_heads,
            geo.gdn_key_dim,
            geo.gdn_value_dim,
            geo.gdn_conv(),
            geo.conv_kernel - 1
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
        let (cos_h, sin_h) = build_rope_table(geo, cfg.context, scaling.as_ref());
        let cos = cuda::to_f32_dev(&cos_h);
        let sin = cuda::to_f32_dev(&sin_h);
        drop(cos_h);
        drop(sin_h);
        rep.lines.push(format!(
            "RoPE tbl  {:9.1} MB  ({} positions × {} pairs × cos+sin)",
            sizes.rope_bytes as f64 / MIB,
            cfg.context,
            geo.rope_pairs
        ));
        // the armed line: what the config asked for and what was derived from
        // it, next to the table it changed (silent — nothing is armed by default)
        if let Some(s) = scaling.filter(|s| s.kind != crate::meta::RopeKind::Default) {
            let detail = match s.kind {
                crate::meta::RopeKind::Yarn => {
                    let (lo, hi) = s.yarn_corr_range(2 * geo.rope_pairs, geo.rope_theta);
                    format!(
                        "YaRN: factor {} ({} training positions -> {}), corr dims [{lo}, {hi}] of {} (beta {} / {}), mscale {:.4} on the attention scale",
                        s.factor, s.original_context, cfg.context, 2 * geo.rope_pairs, s.beta_fast, s.beta_slow, s.mscale()
                    )
                }
                crate::meta::RopeKind::Linear => {
                    format!("linear: factor {} (every pair interpolated by 1/{})", s.factor, s.factor)
                }
                crate::meta::RopeKind::NtkAware => {
                    format!("ntk-aware: theta {:e} -> {:.0}", geo.rope_theta, s.ntk_base(2 * geo.rope_pairs, geo.rope_theta))
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
            (sizes.total()
                + pending_bytes
                + n as u64 * expert_bytes_per_n_unit) as f64 / MIB
        ));

        (
            ThreeStates {
                geo: *geo,
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
        self.kv_buf as u64
            + ((layer * 2 + if is_k { 0 } else { 1 }) * self.geo.kv_heads * self.context
                + kvh * self.context
                + slot) as u64
            * self.kv.row_bytes(self.geo.head_dim) as u64
    }
}

/// The inputs of the two-sided hot-set clamp, all measured or planned bytes.
#[derive(Clone, Copy, Debug)]
pub struct ClampInput {
    pub n_hot: usize,
    /// C3: routed experts per layer (`Geo::moe().experts`), the ceiling of N
    pub experts: usize,
    /// `StateSizes::total()`: KV (at the config's dtype) + QSA + GDN + rope
    pub states_bytes: u64,
    pub pending_bytes: u64,
    pub expert_bytes_per_n_unit: u64,
    pub cold_bytes_per_n_unit: u64,
    pub cold_fixed: bool,
    pub spare: usize,
    pub free0: u64,
    pub host_pinned_budget: u64,
}

/// The hot-set size N the planner picks, as pure arithmetic (#102: factored
/// out of `ThreeStates::allocate` unchanged so the KV-dtype effect on N is
/// testable without a GPU). VRAM lowers N, the host pinned budget RAISES it
/// (fewer cold experts). `Err` is the refusal text the caller panics with.
impl ClampInput {
    /// bytes of the pinned cold tier at hot-set size `n` (the clamp's own formula)
    pub fn cold_at(&self, n: usize) -> u64 {
        let e = self.experts;
        (if self.cold_fixed { e } else { e - n.min(e) + self.spare }) as u64 * self.cold_bytes_per_n_unit
    }
}

pub fn clamp_hot_n(c: &ClampInput) -> Result<(usize, Vec<String>), String> {
    let mut lines = Vec::new();
    let mut n = c.n_hot;
    let cold_of = |n: usize| c.cold_at(n);
    // Termination guard (2026-09-04): when VRAM pushes N down and the host
    // budget pushes it up, no N is feasible. Without this the loop
    // oscillated forever and grew the report lines without bound -> the whole
    // machine froze from RAM exhaustion (chunk 1024 on the M container).
    let mut went_down = false;
    let mut went_up = false;
    let mut iters = 0u32;
    loop {
        let sum = c.states_bytes + c.pending_bytes + n as u64 * c.expert_bytes_per_n_unit;
        let cold = cold_of(n);
        if sum + SAFETY < c.free0 && cold <= c.host_pinned_budget {
            break;
        }
        if n == N_MIN {
            break;
        }
        iters += 1;
        if (went_down && went_up) || iters > 2 * c.experts as u32 {
            // #10b: the message moved into planner_refusal_msg (tested)
            return Err(planner_refusal_msg(c.free0, c.host_pinned_budget));
        }
        if sum + SAFETY >= c.free0 {
            went_down = true;
            n -= 1;
            if n % 8 == 0 {
                lines.push(format!(
                    "VRAM budget over by {:.0} MB at N={} — clamping",
                    (sum + SAFETY - c.free0) as f64 / MIB,
                    n
                ));
            }
        } else {
            went_up = true;
            n += 1;
            if n % 8 == 0 {
                lines.push(format!(
                    "host pinned tier over by {:.0} MB at N={} — raising N",
                    (cold - c.host_pinned_budget) as f64 / MIB,
                    n
                ));
            }
        }
        if cfg_n_dbg() {
            tracing::info!(target: "manager", "[clamp] n={n} vram_sum={:.0} MB cold={:.0} MB free0={:.0} MB",
                c.states_bytes as f64 / MIB,
                cold_of(n) as f64 / MIB,
                c.free0 as f64 / MIB);
        }
    }
    let cold_final = cold_of(n);
    if cold_final > c.host_pinned_budget {
        return Err(format!(
            "refusing config: hot set N={n} would pin {:.1} GiB cold > budget {:.1} GiB — no feasible N (spec 2.1)",
            cold_final as f64 / GIB,
            c.host_pinned_budget as f64 / GIB
        ));
    }
    Ok((n, lines))
}

pub const SAFETY: u64 = 512 << 20; // launch pools, scratch, telemetry slack
/// VRAM the planner leaves free for launch/param plumbing, booked as pending (`planner_pending`).
/// Sibling of `SAFETY`: two reserves, two sums, two numbers. #159: one constant for
/// `Engine::load` and the glm5_next plan (it was a local const of `Engine::load`).
pub const LAUNCH_SLACK: u64 = 128 << 20;
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

/// #110: VRAM the planner leaves FREE for a co-resident GPU client (Crow's
/// `render_page`, a browser, the compositor), in MiB, when
/// `CROW_RENDER_RESERVE_MB` is unset. robin's decision of 2026-09-25: under
/// serve only 73-185 MiB stayed free and every Crow capture fell back to
/// SwiftShader (Crow's GPU gate needs 512 MiB). 1536 MiB (lead, 2026-09-25, under
/// robin's go; first 1024) also covers Crow's gate with its browser panel open
/// (`crow_platform.py` `_GPU_HEADROOM_PANEL_MIB`, Crow #279) and is above
/// llama.cpp's default per-device `--fit-target` margin of 1024 MiB.
///
/// #110 follow-up (2026-09-25): the default is 0 (off). The 1536 MiB default did
/// not boot on the machine it was written for: at N=150 the pinned cold tier is
/// 45.61 GiB against the 46.00 GiB host cap, every hot unit given up moves
/// 126.6 MiB into that tier, and `clamp_hot_n` refused the config
/// (`manager.rs:311` panic). Crow now borrows VRAM per render through serve's
/// lending endpoints (#117); the variable stays for machines with host RAM
/// headroom, and is granted best-effort (`grant_render_reserve`).
pub const RENDER_RESERVE_DEFAULT_MB: u64 = 0;

/// #110: `CROW_RENDER_RESERVE_MB` as bytes. `None` (unset) or an empty value =
/// the default; `0` = off; anything that is not a whole number of MiB is an
/// error the boot stops on.
pub fn render_reserve_from(v: Option<&str>) -> Result<u64, String> {
    let v = match v.map(str::trim) {
        None | Some("") => return Ok(RENDER_RESERVE_DEFAULT_MB << 20),
        Some(v) => v,
    };
    let mb: u64 = v.parse().map_err(|_| {
        format!(
            "refusing config: CROW_RENDER_RESERVE_MB={v:?} is not a whole number of MiB (default {RENDER_RESERVE_DEFAULT_MB}, 0 = off)"
        )
    })?;
    mb.checked_mul(1 << 20)
        .ok_or_else(|| format!("refusing config: CROW_RENDER_RESERVE_MB={mb} is out of range"))
}

/// #110: the process's REQUESTED render reserve, read from the environment. A
/// bad value panics with the reason; `Engine::load` calls this before it loads a
/// byte. What is granted is decided by `grant_render_reserve` against both budgets.
pub fn render_reserve_bytes() -> u64 {
    render_reserve_from(std::env::var("CROW_RENDER_RESERVE_MB").ok().as_deref())
        .unwrap_or_else(|e| panic!("{e}"))
}

/// #110: the planner's `pending` bytes: VRAM that is planned but not resident
/// when N is chosen. The render reserve is never allocated by the engine, so
/// including it here is what keeps it free after the load.
pub fn planner_pending(launch_slack: u64, ring_reserve: u64, vit_reserve: u64, render_reserve: u64) -> u64 {
    launch_slack + ring_reserve + vit_reserve + render_reserve
}

/// #159: `planner_pending` plus the VRAM a family's stability policy keeps off the plan
/// (`Stability::planner_reserve`: glm5_next 2 GiB of headroom on a 32,607 MiB card, capped at
/// 31.9 GiB; Flash-Next and the 27B `OF_RECORD`, 0 B, so their pending is unchanged)
pub fn planner_pending_for(
    stability: &Stability,
    vram_total: u64,
    launch_slack: u64,
    ring_reserve: u64,
    vit_reserve: u64,
    render_reserve: u64,
) -> u64 {
    planner_pending(launch_slack, ring_reserve, vit_reserve, render_reserve) + stability.planner_reserve(vram_total)
}

// ---------------- #159: the glm5_next plan (VRAM, pinned RAM, NVMe) ----------------

/// #159: the state bytes of a glm5_next model at `context` tokens, from its `Glm5Geo`
/// (`docs/glm5-next-recipe.md` section 13). The MTP block (checkpoint layer 45) is not
/// executed (plan step 21), so it books nothing.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Glm5States {
    pub context: usize,
    /// MLA latent per token per DSA layer (BF16, 1,024 B) and the indexer row (HF layout, 514 B)
    pub latent_per_token: u64,
    pub indexer_per_token: u64,
    /// the two caches over every DSA layer at `context` tokens
    pub latent_bytes: u64,
    pub indexer_bytes: u64,
    /// per sequence, independent of the context: KDA state (4 MiB per layer) and conv window
    pub kda_state_bytes: u64,
    pub kda_conv_bytes: u64,
}

impl Glm5States {
    pub fn plan(g: &Glm5Geo, context: usize) -> Glm5States {
        let dsa = g.dsa_layers as u64;
        let kda = g.kda_layers as u64;
        Glm5States {
            context,
            latent_per_token: g.latent_bytes_per_token(),
            indexer_per_token: g.indexer_bytes_per_token(),
            latent_bytes: dsa * context as u64 * g.latent_bytes_per_token(),
            indexer_bytes: dsa * context as u64 * g.indexer_bytes_per_token(),
            kda_state_bytes: kda * g.kda_state_bytes(),
            kda_conv_bytes: kda * g.kda_conv_bytes(),
        }
    }
    /// the cache bytes of one token over every DSA layer
    pub fn per_token(&self, g: &Glm5Geo) -> u64 {
        g.dsa_layers as u64 * (self.latent_per_token + self.indexer_per_token)
    }
    pub fn total(&self) -> u64 {
        self.latent_bytes + self.indexer_bytes + self.kda_state_bytes + self.kda_conv_bytes
    }
}

/// #159: the inputs of the three-tier expert plan, every one of them named in the printout
#[derive(Clone, Copy, Debug)]
pub struct TierInput {
    /// the card's VRAM (the boot replaces it by the measured free VRAM)
    pub vram_total: u64,
    /// the family's policy: headroom and cap (`Stability::planner_reserve`)
    pub stability: Stability,
    /// VRAM-resident bytes that are not routed experts (the container's dense part)
    pub dense_bytes: u64,
    /// #161: what the load adds on top of the container's dense part: `kv_b_proj` decoded from
    /// NVFP4 to BF16 (`Glm5Geo::kv_b_decode_bytes`, 265,289,728 B on GLM-5.3-Flash)
    pub decoded_bytes: u64,
    /// `Glm5States::total` at the boot context
    pub states_bytes: u64,
    /// the cold staging slots (`Stability::stage_slots`) times one expert block
    pub staging_bytes: u64,
    /// #186: the prompt phase's device bytes at its chunk above a one-row pass
    /// ([`glm5_chunk_scratch_bytes`]); 0 at chunk 1 (every row one decode call)
    pub chunk_scratch_bytes: u64,
    /// #186: the prompt chunk the plan books (1 = row by row)
    pub chunk: usize,
    /// `LAUNCH_SLACK`
    pub launch_slack: u64,
    /// one routed expert of one MoE layer
    pub expert_block_bytes: u64,
    pub moe_layers: usize,
    /// routed experts per MoE layer
    pub experts: usize,
    /// the pinned host budget: `HOST_PINNED_CAP`, or what the boot derives below it
    pub host_pinned_budget: u64,
}

/// #159: where the routed experts of every MoE layer live: N in VRAM, P in pinned host RAM,
/// the rest read from NVMe (#149). One unit is one expert in every MoE layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierPlan {
    /// `Stability::vram_ceiling(vram_total)`
    pub vram_ceiling: u64,
    /// dense + decoded + states + staging + pending (with the policy reserve) + `SAFETY`
    pub fixed_bytes: u64,
    /// `expert_block_bytes x moe_layers`
    pub unit_bytes: u64,
    pub hot: usize,
    pub pinned: usize,
    pub nvme: usize,
}

impl TierPlan {
    pub fn hot_bytes(&self) -> u64 {
        self.hot as u64 * self.unit_bytes
    }
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned as u64 * self.unit_bytes
    }
    pub fn nvme_bytes(&self) -> u64 {
        self.nvme as u64 * self.unit_bytes
    }
}

/// #159: the three-tier plan. VRAM takes the dense part, the states, the staging, the launch
/// slack and `SAFETY`, then as many expert units as fit under the family's VRAM ceiling
/// (`hot`, at most `experts`); pinned RAM takes as many of the rest as fit the host budget
/// (`pinned`); NVMe holds what is left (`nvme`). `Err` when the dense part and the states do
/// not fit the ceiling with no expert at all: no expert tier changes that.
pub fn plan_three_tiers(i: &TierInput) -> Result<TierPlan, String> {
    let gib = |b: u64| b as f64 / GIB;
    let vram_ceiling = i.stability.vram_ceiling(i.vram_total);
    // the clamp's own sum (`clamp_hot_n`): what must fit the card, the policy reserve inside pending
    let pending = planner_pending_for(&i.stability, i.vram_total, i.launch_slack, 0, 0, 0);
    let need = i.dense_bytes + i.decoded_bytes + i.states_bytes + i.staging_bytes + i.chunk_scratch_bytes + pending + SAFETY;
    // the same sum without the reserve: what lands below the ceiling
    let fixed_bytes = need - i.stability.planner_reserve(i.vram_total);
    if need > i.vram_total {
        return Err(format!(
            "refusing config: family Glm5Next needs {:.2} GiB before any expert (dense part {:.2} + kv_b at BF16 {:.2} + states {:.2} + staging {:.2}{} + launch slack {:.2} + safety {:.2}) against a VRAM plan ceiling of {:.2} GiB (card {:.2} GiB - headroom {:.2} GiB, cap {}) - the dense part does not fit, no expert tier changes that (#159)",
            gib(fixed_bytes), gib(i.dense_bytes), gib(i.decoded_bytes), gib(i.states_bytes), gib(i.staging_bytes),
            if i.chunk > 1 { format!(" + prompt chunk {} (CROW_CHUNK, #186) {:.2}", i.chunk, gib(i.chunk_scratch_bytes)) } else { String::new() },
            gib(i.launch_slack), gib(SAFETY),
            gib(vram_ceiling), gib(i.vram_total), gib(i.stability.vram_headroom),
            i.stability.vram_cap.map_or("none".to_string(), |c| format!("{:.2} GiB", gib(c)))
        ));
    }
    let unit_bytes = i.expert_block_bytes * i.moe_layers as u64;
    let unit = unit_bytes.max(1);
    let hot = (((i.vram_total - need) / unit) as usize).min(i.experts);
    let pinned = ((i.host_pinned_budget / unit) as usize).min(i.experts - hot);
    Ok(TierPlan { vram_ceiling, fixed_bytes, unit_bytes, hot, pinned, nvme: i.experts - hot - pinned })
}

/// the records of the pinned NVMe landing ring of glm5_next prompt calls: prefill NVMe reads land
/// there and the copy engine moves them into the staging slots (two `MAX_IN_FLIGHT` batches)
pub const GLM5_PREFILL_RING: usize = 2 * crate::nvme_source::MAX_IN_FLIGHT;

/// #186: the call sizes a glm5_next prompt phase at `chunk` rows per call rounds a borrowed
/// scratch up to (`glm5_tiers::prompt_borrow_rows`) besides the one-row decode call: every power
/// of two above 1 and below `chunk`, then `chunk`. Empty at chunk 1. (#196: a call's dense FFN
/// and MoE run on the one expert-major plan of the pass's rows, `glm5_moe::GpuMoeGroupedPlan`.)
pub fn glm5_prompt_call_sizes(chunk: usize) -> Vec<usize> {
    let mut v: Vec<usize> = (1..usize::BITS).map(|i| 1usize << i).take_while(|&p| p < chunk).collect();
    if chunk > 1 {
        v.push(chunk);
    }
    v
}

/// #186: the device bytes the glm5_next prompt phase holds at `chunk` rows per call over a cache
/// of `cap` rows, above what a one-row pass holds ([`glm5_chunk_scratch_parts`], summed): the pass
/// scratch at `max_t = chunk` minus at 1, the residual's `chunk - 1` more rows, the one
/// expert-major plan of `chunk` rows with the dense FFN over its region, and with
/// `CROW_GLM_DENSE_GEMM=1` the tensor-core path's row table (`kernels::glm5_moe::TC_ROWS_BYTES`,
/// #186). The small parameter arrays are left out (the GPU test
/// `glm5_tiers_gpu_186_chunk_bytes_are_what_the_prompt_phase_allocates` holds the sum to the
/// bytes the allocations register). 0 at chunk 1.
pub fn glm5_chunk_scratch_bytes(g: &Glm5Geo, chunk: usize, cap: usize) -> u64 {
    glm5_chunk_scratch_bytes_tc(g, chunk, cap, crate::kernels::glm5_moe::dense_gemm_from_env())
}

/// [`glm5_chunk_scratch_bytes`] with `CROW_GLM_DENSE_GEMM` given (`tc`)
pub fn glm5_chunk_scratch_bytes_tc(g: &Glm5Geo, chunk: usize, cap: usize, tc: bool) -> u64 {
    glm5_chunk_scratch_parts(g, chunk, cap, tc).iter().map(|p| p.1).sum()
}

/// #196: [`glm5_chunk_scratch_bytes_tc`] by buffer, above a one-row pass: (name, bytes) of
/// - `mhc` the mHC plan (`glm5_mhc::Plan`: logits, pre, post, comb, done), `rows` the collapsed
///   and sublayer rows, `residual` the residual's `chunk - 1` more rows;
/// - `attention` the region the KDA and the MLA scratch share (`glm5_model::attn_region_bytes`:
///   the larger of `KdaScratch` at `min(chunk, KDA_SUB_ROWS)` rows and `MlaScratch` at
///   `min(chunk, MLA_SUB_ROWS)` rows with its pool scores, `cap / kpool` per row), `selection` the
///   MLA selection of the call's rows (`MlaScratch::sel_bytes`);
/// - `moe` the expert-major plan of `chunk` rows (`glm5_moe::grouped_plan_bytes`: router logits,
///   ids, weights, combo list, work items, and its region: the experts' outputs, the shared
///   expert, gate / up of one piece of at most `GROUP_PIECE_COMBOS` combos, over the same bytes
///   the dense FFN of `chunk` rows);
/// - `tc` the tensor-core row table.
pub fn glm5_chunk_scratch_parts(g: &Glm5Geo, chunk: usize, cap: usize, tc: bool) -> Vec<(&'static str, u64)> {
    use crate::glm5_mhc::{HC, MIX};
    use crate::glm5_mla::{MlaDims, MlaScratch};
    use crate::glm5_model::{attn_region_bytes, attn_rows};
    if chunk <= 1 {
        return Vec::new();
    }
    let (md, h) = (MlaDims::of(g), g.hidden);
    let above = |f: &dyn Fn(usize) -> usize| (f(chunk) - f(1)) as u64;
    let moe = crate::glm5_moe::grouped_plan_bytes(h, g.experts, g.topk, g.expert_inter, g.expert_inter * g.shared_experts, g.dense_inter, chunk);
    // #186: the tensor-core path's device row counts (`CROW_GLM_DENSE_GEMM=1`), held by a pass of `chunk` rows
    let tc = if tc && chunk >= crate::kernels::glm5_moe::TC_MIN_ROWS { crate::kernels::glm5_moe::TC_ROWS_BYTES } else { 0 };
    vec![
        ("mhc", above(&|m| 4 * m * (MIX + HC + HC + HC * HC + 1))),
        ("rows", above(&|m| 4 * 2 * m * h)),
        ("residual", ((chunk - 1) * g.hc_streams * h * 4) as u64),
        ("attention", above(&|m| attn_region_bytes(g, attn_rows(m), cap))),
        ("selection", above(&|m| MlaScratch::sel_bytes(&md, m))),
        ("moe", moe),
        ("tc", tc),
    ]
}

/// #159: the glm5_next plan from its geometry: the states at `context`, the staging of its
/// stability policy (`pf_tg` tiles per prefill group, `pf_async` = two group sets), the three
/// tiers. `dense_bytes` and `expert_block_bytes` are the container's: the record of one routed
/// expert of one MoE layer in the container's codec (`ExpertRecordSpec::bytes`; 14,155,776 B at
/// NVFP4, the plan's 9,474,048 B at 3.05-bpw MUL1). A record that is not a whole number of
/// 4096-B sectors is refused by name (`geo::expert_record_refusal`).
#[allow(clippy::too_many_arguments)]
pub fn plan_glm5_next(
    g: &Glm5Geo,
    context: usize,
    vram_total: u64,
    host_pinned_budget: u64,
    dense_bytes: u64,
    expert_block_bytes: u64,
    pf_tg: usize,
    pf_async: bool,
) -> Result<(Glm5States, TierInput, TierPlan), String> {
    plan_glm5_next_chunk(g, context, vram_total, host_pinned_budget, dense_bytes, expert_block_bytes, pf_tg, pf_async, 1)
}

/// #186: [`plan_glm5_next`] for a prompt phase in calls of up to `chunk` rows (`CROW_CHUNK`,
/// `glm5_tiers::prompt_chunk_from_env`). Above chunk 1 the plan also books the prompt phase's
/// device bytes ([`glm5_chunk_scratch_bytes`], in VRAM before any expert) and takes the pinned
/// NVMe landing ring of prompt calls ([`GLM5_PREFILL_RING`] records, the async H2D source of
/// `glm5_tiers::PrefillMover`) off the pinned budget. Chunk 1 is `plan_glm5_next` byte for byte.
/// A chunk whose bytes do not fit is refused by name (`plan_three_tiers`).
#[allow(clippy::too_many_arguments)]
pub fn plan_glm5_next_chunk(
    g: &Glm5Geo,
    context: usize,
    vram_total: u64,
    host_pinned_budget: u64,
    dense_bytes: u64,
    expert_block_bytes: u64,
    pf_tg: usize,
    pf_async: bool,
    chunk: usize,
) -> Result<(Glm5States, TierInput, TierPlan), String> {
    if let Some(why) = expert_record_refusal(expert_block_bytes) {
        return Err(why);
    }
    let chunk = chunk.clamp(1, context.max(1));
    let stability = Stability::of(Family::Glm5Next);
    let states = Glm5States::plan(g, context);
    let slots = stability.stage_slots(g.topk, pf_tg, pf_async);
    let landing = if chunk > 1 { GLM5_PREFILL_RING as u64 * expert_block_bytes } else { 0 };
    let input = TierInput {
        vram_total,
        stability,
        dense_bytes,
        // #161: kv_b is stored NVFP4 and decoded once to BF16 at load (the MLA kernels read BF16)
        decoded_bytes: g.kv_b_decode_bytes(),
        states_bytes: states.total(),
        // decode and prefill sets apart (#176): both are held, one record per slot
        staging_bytes: slots.bytes(expert_block_bytes),
        chunk_scratch_bytes: glm5_chunk_scratch_bytes(g, chunk, context),
        chunk,
        launch_slack: LAUNCH_SLACK,
        expert_block_bytes,
        moe_layers: g.moe_layers(),
        experts: g.experts,
        host_pinned_budget: host_pinned_budget.saturating_sub(landing),
    };
    let plan = plan_three_tiers(&input)?;
    Ok((states, input, plan))
}

/// #159: the expert record `states --plan` plans with, and where it came from
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct PlanRecord {
    /// the codec; `None` when `--expert-bytes` came without `--expert-codec` (the plan needs
    /// the bytes only)
    pub codec: Option<ExpertCodec>,
    pub bytes: u64,
    /// the printout's source line for the record
    pub source: String,
}

/// #159: the expert record of `states --plan`, from exactly one of: a container (`--cnq PATH`:
/// its index names codec and record, `nvme_source::glm5_record_of_container`) or an explicit
/// override for planning without one (`--expert-bytes N [--expert-codec nvfp4|mul1]`). There is
/// no default: neither, or both, is refused by name, so a 3-bit plan can never silently take
/// the 4.5-bit 14,155,776 B. An NVFP4 record must equal what the config derives
/// (`Glm5Geo::expert_block_bytes`).
pub fn glm5_plan_record(g: &Glm5Geo, cnq: Option<&str>, expert_bytes: Option<&str>, expert_codec: Option<&str>) -> Result<PlanRecord, String> {
    let nvfp4_check = |codec: Option<ExpertCodec>, bytes: u64, what: &str| -> Result<(), String> {
        if codec == Some(ExpertCodec::Nvfp4) && bytes != g.expert_block_bytes() {
            return Err(format!("{what}: an nvfp4 record of {bytes} B, the config derives {} B at nvfp4 - not this model's record", g.expert_block_bytes()));
        }
        Ok(())
    };
    match (cnq, expert_bytes) {
        (Some(_), Some(_)) => Err("give the expert record once: --cnq <container> or --expert-bytes N, not both".into()),
        (None, None) => Err(
            "the glm5_next plan needs the routed-expert record: --cnq <container> (its index names codec and record) or --expert-bytes N [--expert-codec nvfp4|mul1] (planning without a container); there is no default (#159)"
                .into(),
        ),
        (Some(path), None) => {
            if expert_codec.is_some() {
                return Err("--expert-codec goes with --expert-bytes; a container names its own codec".into());
            }
            let (spec, n) = crate::nvme_source::glm5_record_of_container(path)?;
            nvfp4_check(Some(spec.codec), spec.bytes, path)?;
            Ok(PlanRecord {
                codec: Some(spec.codec),
                bytes: spec.bytes,
                source: format!("container index {path}: codec {}, {n} records of {} B", spec.codec.dtype(), spec.bytes),
            })
        }
        (None, Some(b)) => {
            let bytes: u64 = b.trim().parse().map_err(|_| format!("--expert-bytes {b:?} is not a whole number of bytes"))?;
            if let Some(why) = expert_record_refusal(bytes) {
                return Err(format!("--expert-bytes: {why}"));
            }
            let codec = expert_codec.map(ExpertCodec::from_dtype).transpose()?;
            nvfp4_check(codec, bytes, "--expert-bytes")?;
            let c = codec.map_or("not given (--expert-codec; the plan needs the bytes only)".to_string(), |c| c.dtype().to_string());
            Ok(PlanRecord { codec, bytes, source: format!("--expert-bytes, no container read; codec {c}") })
        }
    }
}

/// #159: the plan as `states --plan` prints it: every input with its source, then the three
/// tiers per MoE layer and in bytes. `sources` names where the dense part, the expert block,
/// the card and the pinned budget came from.
pub fn glm5_plan_table(g: &Glm5Geo, s: &Glm5States, i: &TierInput, p: &TierPlan, sources: &[(&str, String)]) -> String {
    let gib = |b: u64| b as f64 / GIB;
    let src = |k: &str| sources.iter().find(|(n, _)| *n == k).map_or(String::new(), |(_, v)| format!(" ({v})"));
    let slots = i.staging_bytes / i.expert_block_bytes.max(1);
    let mut o = Vec::new();
    o.push(format!("--- glm5_next plan (#159): context {}, three expert tiers ---", s.context));
    o.push("inputs".to_string());
    o.push(format!("  dense part           {:>16} B  {:>7.2} GiB{}", i.dense_bytes, gib(i.dense_bytes), src("dense")));
    o.push(format!("  expert block         {:>16} B  {:>7.2} MiB  one expert of one MoE layer{}", i.expert_block_bytes, i.expert_block_bytes as f64 / MIB, src("expert")));
    o.push(format!("  MoE layers x experts {:>16}    {} x {} (top-{}), MTP layer 45 not executed (step 21): no cache, no experts", "", i.moe_layers, i.experts, g.topk));
    o.push(format!("  KV per token         {:>16} B  {} DSA layers x (MLA latent {} B BF16 + indexer {} B HF layout)", s.per_token(g), g.dsa_layers, s.latent_per_token, s.indexer_per_token));
    o.push(format!("  card VRAM            {:>16} B  {:>7.2} GiB{}", i.vram_total, gib(i.vram_total), src("vram")));
    o.push(format!("  pinned budget        {:>16} B  {:>7.2} GiB{}", i.host_pinned_budget, gib(i.host_pinned_budget), src("pinned")));
    o.push("VRAM".to_string());
    o.push(format!("  plan ceiling         {:>16} B  {:>7.2} GiB  card - headroom {:.2} GiB, cap {} (Stability::GLM5_NEXT, #176)", p.vram_ceiling, gib(p.vram_ceiling), gib(i.stability.vram_headroom),
        i.stability.vram_cap.map_or("none".to_string(), |c| format!("{:.2} GiB", gib(c)))));
    o.push(format!("  dense part           {:>16} B  {:>7.2} GiB", i.dense_bytes, gib(i.dense_bytes)));
    o.push(format!("  kv_b at BF16         {:>16} B  {:>7.2} GiB  {} DSA layers: kv_b decoded at load, BF16 {} B - container NVFP4 {} B (#161)", i.decoded_bytes, gib(i.decoded_bytes), g.dsa_layers, g.kv_b_bf16_bytes(), g.kv_b_nvfp4_bytes()));
    o.push(format!("  MLA latent           {:>16} B  {:>7.2} GiB  {} x {} tokens x {} B", s.latent_bytes, gib(s.latent_bytes), g.dsa_layers, s.context, s.latent_per_token));
    o.push(format!("  indexer cache        {:>16} B  {:>7.2} GiB  {} x {} tokens x {} B", s.indexer_bytes, gib(s.indexer_bytes), g.dsa_layers, s.context, s.indexer_per_token));
    o.push(format!("  KDA state + conv     {:>16} B  {:>7.2} GiB  {} layers x ({} + {} B) per sequence", s.kda_state_bytes + s.kda_conv_bytes, gib(s.kda_state_bytes + s.kda_conv_bytes), g.kda_layers, g.kda_state_bytes(), g.kda_conv_bytes()));
    o.push(format!("  cold staging         {:>16} B  {:>7.2} GiB  {} slots (decode and prefill sets apart, #176)", i.staging_bytes, gib(i.staging_bytes), slots));
    if i.chunk > 1 {
        o.push(format!("  prompt chunk         {:>16} B  {:>7.2} GiB  {} rows per prompt call (CROW_CHUNK, #186): pass scratch (KDA | MLA in one region at their sub-block rows, #196), residual rows, the expert-major MoE plan with the dense FFN over its region", i.chunk_scratch_bytes, gib(i.chunk_scratch_bytes), i.chunk));
    }
    o.push(format!("  launch slack + safety{:>16} B  {:>7.2} GiB", i.launch_slack + SAFETY, gib(i.launch_slack + SAFETY)));
    o.push("  activations/scratch  not measured (GLM widths; the step-14 boot measures them against free VRAM)".to_string());
    o.push(format!("  room for experts     {:>16} B  {:>7.2} GiB", p.vram_ceiling - p.fixed_bytes, gib(p.vram_ceiling - p.fixed_bytes)));
    o.push(format!("expert tiers per MoE layer (unit = {} B = {} layers x one block)", p.unit_bytes, i.moe_layers));
    let share = |n: usize| 100.0 * n as f64 / i.experts.max(1) as f64;
    o.push(format!("  VRAM    N = {:>3}  {:>5.1} %  {:>16} B  {:>7.2} GiB", p.hot, share(p.hot), p.hot_bytes(), gib(p.hot_bytes())));
    o.push(format!("  pinned  P = {:>3}  {:>5.1} %  {:>16} B  {:>7.2} GiB", p.pinned, share(p.pinned), p.pinned_bytes(), gib(p.pinned_bytes())));
    o.push(format!("  NVMe        {:>3}  {:>5.1} %  {:>16} B  {:>7.2} GiB  (CROW_NVME_TIER, #149)", p.nvme, share(p.nvme), p.nvme_bytes(), gib(p.nvme_bytes())));
    o.push(format!("  resident (N + P) / {} = {:.1} %; expert cache policy default lru (CROW_EXPERT_CACHE, #175)", i.experts, share(p.hot + p.pinned)));
    o.push("  G1d capacities (plan step 3): not measured yet - these are the planner's own numbers".to_string());
    o.join("\n")
}

/// #110: what must still be free once the load is done: the #72 floor for the
/// engine's own post-plan allocations plus the render reserve.
pub const fn post_plan_floor(render_reserve: u64) -> u64 {
    POST_PLAN_FLOOR + render_reserve
}

/// #110: the render reserve's own `[budget]` line, with its cost in hot-set
/// units (`unit` = the bytes of one hot expert across all layers)
pub fn render_reserve_line(render_reserve: u64, unit: u64, from_env: bool) -> String {
    let src = if from_env { "CROW_RENDER_RESERVE_MB" } else { "default, CROW_RENDER_RESERVE_MB unset" };
    if render_reserve == 0 {
        return format!(
            "render reserve    0.0 MB  ({src}: off) — no VRAM is kept for a co-resident renderer; Crow's render_page borrows it per render through POST /v1/crow/vram/lend (#117)"
        );
    }
    let units = if unit > 0 { render_reserve as f64 / unit as f64 } else { 0.0 };
    format!(
        "render reserve {:9.1} MB  ({src}) — kept FREE for a co-resident GPU client (Crow's render_page, a browser); never allocated by the engine; costs {units:.1} hot-set units",
        render_reserve as f64 / MIB
    )
}

/// #110 follow-up: what the planner GRANTS of a requested render reserve.
#[derive(Debug, Clone, PartialEq)]
pub struct ReserveGrant {
    pub requested: u64,
    pub granted: u64,
    /// the hot-set size chosen with the granted reserve
    pub n: usize,
    /// the clamp's own lines for that N
    pub lines: Vec<String>,
    /// the `[budget] render reserve: requested X MiB, granted Y MiB — ...` line
    pub line: String,
}

/// #110 follow-up (2026-09-25): the render reserve is BEST-EFFORT and never
/// blocks the boot. `base` is the clamp input WITHOUT the reserve in
/// `pending_bytes`. The whole request is granted when both budgets hold with it;
/// otherwise the largest multiple of one hot-set unit (`expert_bytes_per_n_unit`)
/// below the request that still fits both the VRAM budget and the host pinned
/// budget, down to 0. `Err` only when the config does not fit even WITHOUT a
/// reserve - that is the planner's own refusal, not the reserve's.
///
/// Measured case (engine.log 2026-09-25 07:44 UTC, 56a9740 boot 10:34 UTC): N=150,
/// cold tier 369 units = 45.61 GiB against a 46.00 GiB cap. N may not drop below
/// 147, so 1536 MiB (12.1 units) panicked at `clamp_hot_n`; granted here: 3 units.
/// Crow #300 phase 2: the plan of a dense FFN, which has no hot set to clamp: the
/// states and the pending bytes must fit the free VRAM with `SAFETY` to spare, or
/// the boot refuses by name (the #102 rule: say what does not fit, never page).
/// The render reserve is granted from what is left, best-effort as for MoE (#110).
/// Returns (granted reserve, the `[budget]` lines).
pub fn dense_fit(free0: u64, states_bytes: u64, pending_bytes: u64, requested: u64, context: usize) -> Result<(u64, Vec<String>), String> {
    let gib = |b: u64| b as f64 / GIB;
    let need = states_bytes + pending_bytes + SAFETY;
    if need > free0 {
        return Err(format!(
            "refusing config: context {context} needs {:.2} GiB of states + {:.2} GiB pending + {:.2} GiB safety = {:.2} GiB, free {:.2} GiB - lower the context or use a smaller KV dtype (CROW_KV)",
            gib(states_bytes), gib(pending_bytes), gib(SAFETY), gib(need), gib(free0)
        ));
    }
    let room = free0 - need;
    let granted = requested.min(room);
    let mib = |b: u64| b as f64 / MIB;
    let reserve = if requested == 0 {
        "render reserve: requested 0 MiB, granted 0 MiB — off (CROW_RENDER_RESERVE_MB 0 or unset)".to_string()
    } else if granted == requested {
        format!("render reserve: requested {:.0} MiB, granted {:.0} MiB — the VRAM budget holds", mib(requested), mib(granted))
    } else {
        format!("render reserve: requested {:.0} MiB, granted {:.1} MiB — the VRAM budget binds", mib(requested), mib(granted))
    };
    Ok((granted, vec![
        format!(
            "dense FFN, no hot set: states {:.2} GiB + pending {:.2} GiB + safety {:.2} GiB fit free {:.2} GiB ({:.2} GiB left)",
            gib(states_bytes), gib(pending_bytes), gib(SAFETY), gib(free0), gib(room)
        ),
        reserve,
    ]))
}

pub fn grant_render_reserve(base: &ClampInput, requested: u64) -> Result<ReserveGrant, String> {
    let with = |r: u64| clamp_hot_n(&ClampInput { pending_bytes: base.pending_bytes + r, ..*base });
    let mib = |b: u64| b as f64 / MIB;
    let unit = base.expert_bytes_per_n_unit.max(1);
    if requested == 0 {
        let (n, lines) = with(0)?;
        return Ok(ReserveGrant { requested, granted: 0, n, lines,
            line: "render reserve: requested 0 MiB, granted 0 MiB — off (CROW_RENDER_RESERVE_MB 0 or unset)".into() });
    }
    if let Ok((n, lines)) = with(requested) {
        return Ok(ReserveGrant { requested, granted: requested, n, lines,
            line: format!("render reserve: requested {:.0} MiB, granted {:.0} MiB — both budgets hold (N={n})", mib(requested), mib(requested)) });
    }
    // the largest k*unit strictly below the request, then down to 0
    let mut k = (requested - 1) / unit;
    loop {
        let r = k * unit;
        match with(r) {
            Ok((n, lines)) => {
                let failed = requested.min((k + 1) * unit);
                let why = reserve_binding(base, failed);
                return Ok(ReserveGrant { requested, granted: r, n, lines,
                    line: format!(
                        "render reserve: requested {:.0} MiB, granted {:.1} MiB ({k} hot-set unit(s), N={n}) — {why}",
                        mib(requested), mib(r)
                    ) });
            }
            Err(e) if k == 0 => return Err(e),
            Err(_) => k -= 1,
        }
    }
}

/// which budget stops a reserve of `r` bytes: the host pinned cap (N may not
/// drop far enough for the VRAM side) or the VRAM budget (even N_MIN is too big)
fn reserve_binding(base: &ClampInput, r: u64) -> String {
    let gib = |b: u64| b as f64 / GIB;
    let vram_fits = |n: usize| base.states_bytes + base.pending_bytes + r + n as u64 * base.expert_bytes_per_n_unit + SAFETY < base.free0;
    let n_vram = (N_MIN..=base.experts).rev().find(|&n| vram_fits(n));
    let n_host = (N_MIN..=base.experts).find(|&n| base.cold_at(n) <= base.host_pinned_budget);
    match (n_vram, n_host) {
        (None, _) => format!("the VRAM budget binds: even N={N_MIN} does not leave {:.2} GiB free (free {:.2} GiB)", gib(r), gib(base.free0)),
        (Some(v), Some(h)) if v < h => format!(
            "the host pinned budget binds: N must stay >= {h} so the cold tier fits {:.2} GiB, and VRAM would need N <= {v} for more",
            gib(base.host_pinned_budget)),
        (Some(v), None) => format!("the host pinned budget binds: no N fits the cold tier in {:.2} GiB (VRAM alone allows N <= {v})", gib(base.host_pinned_budget)),
        (Some(v), Some(h)) => format!("the clamp refused (VRAM allows N <= {v}, host needs N >= {h})"),
    }
}

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
/// is what the card reports once the load is done; below the floor the boot says
/// so loudly, because the decode graph is captured after this line.
/// #110: the floor is `POST_PLAN_FLOOR + render_reserve`, and the line names both.
pub fn headroom_line(free: u64, render_reserve: u64) -> String {
    let gib = |b: u64| b as f64 / GIB;
    let floor = post_plan_floor(render_reserve);
    let parts = if render_reserve > 0 {
        format!(" (post-plan {:.2} + render reserve {:.2})", gib(POST_PLAN_FLOOR), gib(render_reserve))
    } else {
        String::new()
    };
    if headroom_ok(free, render_reserve) {
        format!(
            "free VRAM after load {:.2} GiB >= floor {:.2} GiB{parts} — the image path is held, not borrowed from here",
            gib(free), gib(floor)
        )
    } else {
        format!(
            "SHORT: free VRAM after load {:.2} GiB is BELOW the floor {:.2} GiB{parts} — the decode graph, the driver pools and the render reserve still have to fit; lower the hot-set target, the context or CROW_RENDER_RESERVE_MB",
            gib(free), gib(floor)
        )
    }
}

/// #110: the post-plan check itself, so the line and the loader cannot disagree
pub fn headroom_ok(free: u64, render_reserve: u64) -> bool {
    free >= post_plan_floor(render_reserve)
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
        p.vram("device sampler", crate::gen::sampler_bytes(V));
        p.host("prefix cache (3 snapshots)", 3 * 130_646_016);
        p.host("vit image cache (CROW_VIT_CACHE_MB, LRU)", 256 << 20);
        p
    }

    #[test]
    fn the_post_plan_vram_total_is_the_reserve_plus_the_sampler() {
        let p = ledger();
        assert_eq!(p.vram_bytes(), 239_599_616 + 51_200_000 + crate::gen::sampler_bytes(V));
        // the reserve of record, 277.3 MB, plus ~0.7 MB of sampler
        // (#83, 2026-09-20: params grew 16 -> 36 B for min_p + ln(min_p);
        // #84 the same day: the windowed penalties added counts [V] u16 +
        // the 1026-i32 ring, so the pin moved 281_112 -> 281_132 -> 781_876)
        assert_eq!(p.vram_bytes(), 291_581_492);
        assert_eq!(crate::gen::sampler_bytes(V), 781_876);
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
        // render reserve 0 (#110 off): the #72 floor alone, worded as before
        let ok = headroom_line(600 << 20, 0);
        assert!(ok.starts_with("free VRAM after load 0.59 GiB >= floor 0.25 GiB — "), "{ok}");
        assert!(ok.contains("the image path is held, not borrowed from here"), "{ok}");
        // the 35.7 MiB of the issue: below the floor, and the line says SHORT first
        let short = headroom_line(37_450_000, 0);
        assert!(short.starts_with("SHORT:"), "{short}");
        assert!(short.contains("BELOW the floor 0.25 GiB"), "{short}");
    }
}

#[cfg(test)]
mod tests_110 {
    //! #110: the render reserve. While serve ran, 73-185 MiB of VRAM stayed free
    //! and Crow's render_page (GPU gate 512 MiB) fell back to SwiftShader in every
    //! capture of 2026-09-23/24. robin, 2026-09-25: 1536 MiB by default (Crow's panel gate), kept FREE
    //! by the planner, checked after the load. Pure arithmetic, no GPU.
    use super::*;

    /// one hot-set unit: 48 layers x 2,764,800 B per expert (MEAS-0923 serve logs)
    const UNIT: u64 = 48 * 2_764_800;
    /// `gen.rs` `Engine::load`'s launch slack, the other term of `pending`
    const LAUNCH_SLACK: u64 = 128 << 20;

    /// a card that lands the planner on N=155 with no render reserve (the #72
    /// serve boot of 2026-09-18: N 160 -> 155), the vit reserve already held
    fn card(render: u64) -> ClampInput {
        let states = StateSizes::plan(&Geo::FLASH_NEXT, 200_000, KvDtype::Fp8E4m3, 2048).total();
        ClampInput {
            n_hot: 160,
            experts: E,
            states_bytes: states,
            pending_bytes: planner_pending(LAUNCH_SLACK, 0, 0, render),
            expert_bytes_per_n_unit: UNIT,
            cold_bytes_per_n_unit: UNIT,
            cold_fixed: false,
            spare: 7,
            free0: states + planner_pending(LAUNCH_SLACK, 0, 0, 0) + 155 * UNIT + SAFETY + 1,
            host_pinned_budget: 60 << 30,
        }
    }

    #[test]
    fn the_render_reserve_defaults_to_off_now_that_lending_exists() {
        // #110 follow-up: 0 by default (the 1536 MiB default refused the boot)
        assert_eq!(render_reserve_from(None), Ok(0));
        assert_eq!(render_reserve_from(Some("")), Ok(0));
        assert_eq!(render_reserve_from(Some("0")), Ok(0));
        assert_eq!(render_reserve_from(Some(" 1024 ")), Ok(1024 << 20));
        for bad in ["1g", "-1", "1.5", "off", "18446744073709551615"] {
            let e = render_reserve_from(Some(bad)).unwrap_err();
            assert!(e.starts_with("refusing config: CROW_RENDER_RESERVE_MB="), "{bad:?}: {e}");
        }
    }

    #[test]
    fn a_1536_mib_render_reserve_costs_about_twelve_units_and_stays_free_after_the_plan() {
        let (n_off, _) = clamp_hot_n(&card(0)).unwrap();
        let reserve = render_reserve_from(Some("1536")).unwrap();
        let c = card(reserve);
        let (n_on, lines) = clamp_hot_n(&c).unwrap();
        assert_eq!(n_off, 155);
        // 1536 MiB / 132,710,400 B = 12.14 units: on this card (155 lands exactly on
        // the boundary) the first N that fits is 13 lower
        assert_eq!(n_on, 142, "the reserve did not lower N: {lines:?}");
        // what the plan leaves on the card once states, launch slack and the hot set
        // are counted: the clamp's SAFETY for the engine AND the whole reserve
        let planned = c.states_bytes + planner_pending(LAUNCH_SLACK, 0, 0, 0) + n_on as u64 * UNIT;
        assert!(c.free0 - planned > SAFETY + reserve, "the reserve is not left free");
    }

    #[test]
    fn the_post_plan_check_requires_the_floor_plus_the_render_reserve() {
        let reserve = render_reserve_from(Some("1536")).unwrap();
        assert_eq!(post_plan_floor(reserve), (256 + 1536) << 20);
        // 551 MiB: the #72 live check's free VRAM after load WITHOUT the reserve -
        // enough for the old floor, SHORT once the renderer's 1.5 GiB is owed
        let free = 551 << 20;
        assert!(headroom_ok(free, 0));
        assert!(!headroom_ok(free, reserve));
        let short = headroom_line(free, reserve);
        assert!(short.starts_with("SHORT: free VRAM after load 0.54 GiB is BELOW the floor 1.75 GiB (post-plan 0.25 + render reserve 1.50)"), "{short}");
        assert!(short.contains("CROW_RENDER_RESERVE_MB"), "{short}");
        // the same boot with the reserve kept: 551 + 1536 MiB
        let ok = headroom_line(free + reserve, reserve);
        assert!(ok.starts_with("free VRAM after load 2.04 GiB >= floor 1.75 GiB (post-plan 0.25 + render reserve 1.50)"), "{ok}");
        // the boundary is inclusive, one byte less is SHORT
        assert!(headroom_ok(post_plan_floor(reserve), reserve));
        assert!(!headroom_ok(post_plan_floor(reserve) - 1, reserve));
    }

    #[test]
    fn the_render_reserve_has_its_own_budget_line() {
        let on = render_reserve_line(1536 << 20, UNIT, false);
        assert!(on.starts_with("render reserve    1536.0 MB  (default, CROW_RENDER_RESERVE_MB unset)"), "{on}");
        assert!(on.contains("never allocated by the engine"), "{on}");
        assert!(on.contains("costs 12.1 hot-set units"), "{on}");
        let set = render_reserve_line(1024 << 20, UNIT, true);
        assert!(set.contains("(CROW_RENDER_RESERVE_MB)") && set.contains("costs 8.1 hot-set units"), "{set}");
        let off = render_reserve_line(0, UNIT, true);
        assert!(off.starts_with("render reserve    0.0 MB  (CROW_RENDER_RESERVE_MB: off)"), "{off}");
        assert!(off.contains("POST /v1/crow/vram/lend (#117)"), "{off}");
    }
}

#[cfg(test)]
mod tests_110_boot {
    //! #110 follow-up: the static render reserve must never block the boot.
    //! The real numbers of 2026-09-25 (engine.log 07:44 UTC boot without a
    //! reserve, and the 10:34 UTC boot at 56a9740 that panicked at
    //! `manager.rs:311`): N 160 -> 150 (143 logical + 7 spare), cold tier 369
    //! units = 45.61 GiB, host pinned cap 46.00 GiB, reserve 1536 MiB.
    use super::*;

    const UNIT: u64 = 48 * 2_764_800; // 126.6 MiB, the measured hot-set unit
    const LAUNCH_SLACK: u64 = 128 << 20;

    /// the planner input that lands on N=150 without a reserve (VRAM-bound)
    fn real_card() -> ClampInput {
        let states = StateSizes::plan(&Geo::FLASH_NEXT, 200_000, KvDtype::Fp8E4m3, 2048).total();
        let pending = planner_pending(LAUNCH_SLACK, 0, 0, 0);
        ClampInput {
            n_hot: 160,
            experts: E,
            states_bytes: states,
            pending_bytes: pending,
            expert_bytes_per_n_unit: UNIT,
            cold_bytes_per_n_unit: UNIT,
            cold_fixed: false,
            spare: 7,
            free0: states + pending + 150 * UNIT + SAFETY + 1,
            host_pinned_budget: 46 << 30,
        }
    }

    #[test]
    fn the_real_card_is_the_measured_one() {
        let c = real_card();
        let (n, _) = clamp_hot_n(&c).unwrap();
        assert_eq!(n, 150);
        // 369 cold units = 45.61 GiB of the 46.00 GiB cap
        assert_eq!(c.cold_at(150), 369 * UNIT);
        assert_eq!(format!("{:.2}", c.cold_at(150) as f64 / GIB), "45.61");
        // and 1536 MiB with the old arithmetic (reserve inside pending) is the refusal
        let old = ClampInput { pending_bytes: c.pending_bytes + (1536 << 20), ..c };
        assert!(clamp_hot_n(&old).unwrap_err().starts_with("refusing config: no hot-set size fits BOTH"));
    }

    #[test]
    fn a_1536_mib_reserve_on_the_real_card_boots_with_what_fits() {
        let c = real_card();
        // the boot's own decision: red at 56a9740, where the reserve sat inside
        // `pending` and `clamp_hot_n` refused (the panic at manager.rs:311)
        let g = grant_render_reserve(&c, 1536 << 20).expect("the render reserve blocked the boot");
        // N may drop to 147 (cold 372 units = 45.98 GiB <= 46.00): 3 units granted
        assert_eq!(g.n, 147, "{g:?}");
        assert_eq!(g.granted, 3 * UNIT, "{g:?}");
        assert!(g.line.starts_with("render reserve: requested 1536 MiB, granted 379.7 MiB (3 hot-set unit(s), N=147) — the host pinned budget binds: N must stay >= 147"), "{}", g.line);
        // the grant leaves both budgets satisfied
        let planned = c.states_bytes + c.pending_bytes + g.granted + g.n as u64 * UNIT + SAFETY;
        assert!(planned < c.free0);
        assert!(c.cold_at(g.n) <= c.host_pinned_budget);
    }

    #[test]
    fn a_reserve_that_fits_is_granted_whole_and_0_is_off() {
        let mut c = real_card();
        c.host_pinned_budget = 60 << 30; // RAM headroom: N may drop freely
        let g = grant_render_reserve(&c, 1536 << 20).unwrap();
        assert_eq!(g.granted, 1536 << 20);
        assert!(g.line.contains("granted 1536 MiB — both budgets hold"), "{}", g.line);
        let off = grant_render_reserve(&real_card(), 0).unwrap();
        assert_eq!((off.granted, off.n), (0, 150));
        assert!(off.line.starts_with("render reserve: requested 0 MiB, granted 0 MiB — off"), "{}", off.line);
    }

    #[test]
    fn with_no_room_at_all_the_grant_is_zero_not_a_panic() {
        // cap exactly the cold tier at N=150: N may not drop at all
        let mut c = real_card();
        c.host_pinned_budget = c.cold_at(150);
        let g = grant_render_reserve(&c, 1536 << 20).unwrap();
        assert_eq!((g.granted, g.n), (0, 150));
        assert!(g.line.contains("granted 0.0 MiB (0 hot-set unit(s), N=150) — the host pinned budget binds"), "{}", g.line);
    }

    #[test]
    fn a_config_that_does_not_fit_without_a_reserve_is_still_refused() {
        let mut c = real_card();
        c.free0 = c.states_bytes; // no room for anything
        assert!(grant_render_reserve(&c, 1536 << 20).is_err());
        assert!(grant_render_reserve(&c, 0).is_err());
    }
}

#[cfg(test)]
mod tests_kv_dtype {
    //! #102: CROW_KV=bf16 doubles the KV bytes and the planner pays for them in
    //! hot experts (VRAM) and therefore in pinned cold bytes (host). Pure arithmetic.
    use super::*;

    /// 48 layers x 2,764,800 B per expert (`operating_point.cold_path.expert_bytes`
    /// of the MEAS-0923 serve logs, 2026-09-23) = the bytes of one hot-set unit
    const UNIT: u64 = 48 * 2_764_800;

    #[test]
    fn bf16_kv_is_exactly_twice_fp8_and_nothing_else_moves() {
        let f = StateSizes::plan(&Geo::FLASH_NEXT, 200_000, KvDtype::Fp8E4m3, 2048);
        let b = StateSizes::plan(&Geo::FLASH_NEXT, 200_000, KvDtype::Bf16, 2048);
        // 12 layers x 2 (k,v) x 2 kv-heads x 256 x 200000 x 1 B: the 2343.8 MB of the
        // `[budget] KV` line in serve-tf-dense-kv.log
        assert_eq!(f.kv_bytes, 2_457_600_000);
        assert_eq!(b.kv_bytes, 2 * f.kv_bytes);
        assert_eq!(b.total() - f.total(), f.kv_bytes);
        assert_eq!(
            (b.qsa_keys_bytes, b.qsa_pooled_bytes, b.gdn_s_bytes, b.gdn_conv_bytes, b.rope_bytes),
            (f.qsa_keys_bytes, f.qsa_pooled_bytes, f.gdn_s_bytes, f.gdn_conv_bytes, f.rope_bytes)
        );
        assert_eq!(format!("{:.1}", f.kv_bytes as f64 / MIB), "2343.8");
        assert_eq!(format!("{:.1}", b.kv_bytes as f64 / MIB), "4687.5");
    }

    /// a card sized so that FP8 KV lands on N=155 (the serve-bare boot of MEAS-0923)
    fn card(kv: KvDtype, budget_gib: u64) -> ClampInput {
        let fp8 = StateSizes::plan(&Geo::FLASH_NEXT, 200_000, KvDtype::Fp8E4m3, 2048).total();
        let pending = 1 << 30;
        ClampInput {
            n_hot: 160,
            experts: E,
            states_bytes: StateSizes::plan(&Geo::FLASH_NEXT, 200_000, kv, 2048).total(),
            pending_bytes: pending,
            expert_bytes_per_n_unit: UNIT,
            cold_bytes_per_n_unit: UNIT,
            cold_fixed: false,
            spare: 7,
            free0: fp8 + pending + 155 * UNIT + SAFETY + 1,
            host_pinned_budget: budget_gib << 30,
        }
    }

    #[test]
    fn bf16_kv_costs_19_hot_experts_per_layer_at_200k() {
        let (n8, _) = clamp_hot_n(&card(KvDtype::Fp8E4m3, 48)).unwrap();
        let (n16, lines) = clamp_hot_n(&card(KvDtype::Bf16, 48)).unwrap();
        assert_eq!(n8, 155);
        // 2,457,600,000 B / 132,710,400 B = 18.52 -> the first N that fits is 19 lower
        assert_eq!(n16, 136);
        assert!(lines.iter().any(|l| l.contains("clamping")), "{lines:?}");
    }

    #[test]
    fn bf16_kv_at_the_46_gib_default_cap_is_refused_loudly_not_squeezed() {
        // N=136 pins (512 - 136 + 7) x UNIT = 47.34 GiB > 46 GiB: no feasible N
        assert!(clamp_hot_n(&card(KvDtype::Fp8E4m3, 46)).is_ok());
        let e = clamp_hot_n(&card(KvDtype::Bf16, 46)).unwrap_err();
        assert!(e.starts_with("refusing config: no hot-set size fits BOTH"), "{e}");
    }

    #[test]
    fn crow_kv_words_parse_and_a_typo_is_an_error() {
        assert_eq!(KvDtype::parse("bf16"), Ok(KvDtype::Bf16));
        assert_eq!(KvDtype::parse("BF16"), Ok(KvDtype::Bf16));
        assert_eq!(KvDtype::parse("fp8"), Ok(KvDtype::Fp8E4m3));
        assert_eq!(KvDtype::parse("fp8_e4m3"), Ok(KvDtype::Fp8E4m3));
        for k in [KvDtype::Fp8E4m3, KvDtype::Bf16] {
            assert_eq!(KvDtype::parse(k.name()), Ok(k), "name() must round-trip");
        }
        for bad in ["", "f16", "bf-16", "fp8e4m3", "1", " bf16"] {
            let e = KvDtype::parse(bad).unwrap_err();
            assert!(e.contains("accepted: bf16, fp8, fp8_e4m3"), "{bad:?}: {e}");
        }
        assert_eq!(KvDtype::from_env_value(None), Ok(None));
        assert_eq!(KvDtype::from_env_value(Some("bf16")), Ok(Some(KvDtype::Bf16)));
        assert!(KvDtype::from_env_value(Some("bf61")).is_err());
    }
}

#[cfg(test)]
mod tests_88_q8kv {
    //! #88: `CROW_KV=q8` - a KV row of 256 values is 272 B (256 int8 + 8 f16 scales) and the
    //! planner, the park and the slot header pay exactly that. Pure arithmetic, no GPU.
    use super::*;

    /// The dense 27B at 200,000 (ticket #88, 2026-10-01: the BF16 boot was refused at
    /// `dense_fit`, "needs 12.40 GiB of states ... free 12.17 GiB"): q8 KV is 17/32 of BF16
    /// KV, nothing else in the plan moves, and the same free VRAM holds it.
    #[test]
    fn q8_kv_plans_17_32_of_bf16_and_fits_the_27b_at_200k_where_bf16_was_refused() {
        let g = crate::meta::dense_fixture_geo();
        assert_eq!((g.attn_layers, g.kv_heads, g.head_dim), (16, 4, 256));
        assert_eq!((KvDtype::Fp8E4m3.row_bytes(256), KvDtype::Bf16.row_bytes(256), KvDtype::Q8Block.row_bytes(256)), (256, 512, 272));
        let b = StateSizes::plan(&g, 200_000, KvDtype::Bf16, 2048);
        let q = StateSizes::plan(&g, 200_000, KvDtype::Q8Block, 2048);
        // 16 layers x 2 (k, v) x 4 kv-heads x 200,000 rows x 272 B (bf16: x 512 B)
        assert_eq!(q.kv_bytes, 6_963_200_000);
        assert_eq!(b.kv_bytes, 13_107_200_000);
        assert_eq!(q.kv_bytes * 32, b.kv_bytes * 17);
        assert_eq!(
            (q.qsa_keys_bytes, q.qsa_pooled_bytes, q.gdn_s_bytes, q.gdn_conv_bytes, q.rope_bytes),
            (b.qsa_keys_bytes, b.qsa_pooled_bytes, b.gdn_s_bytes, b.gdn_conv_bytes, b.rope_bytes)
        );
        let (free, pending) = ((12.17 * GIB) as u64, (0.12 * GIB) as u64);
        let e = dense_fit(free, b.total(), pending, 0, 200_000).unwrap_err();
        assert!(e.contains("context 200000 needs 12.40 GiB of states") && e.contains("free 12.17 GiB"), "{e}");
        let (_, lines) = dense_fit(free, q.total(), pending, 0, 200_000).unwrap();
        assert!(lines[0].starts_with("dense FFN, no hot set: states 6.68 GiB + pending 0.12 GiB"), "{}", lines[0]);
        // the park stash and the slot rows take the same row size
        assert_eq!(
            crate::cache::park_host_bytes_rows(&g, 8_192, 200_000, 16, KvDtype::Q8Block.row_bytes(g.head_dim)),
            8_192 * 16 * 2 * 4 * 272
        );
    }

    /// `q8` parses and round-trips; it is refused on a model whose attention is not
    /// `Attn::Full` (Flash-Next's `attn_sel*` have no q8 path) and on a head dim that is not
    /// whole 32-value blocks; unset and the other words resolve as before
    #[test]
    fn crow_kv_q8_parses_and_is_refused_where_the_attention_is_not_full() {
        assert_eq!(KvDtype::parse("q8"), Ok(KvDtype::Q8Block));
        assert_eq!(KvDtype::parse("Q8"), Ok(KvDtype::Q8Block));
        assert_eq!(KvDtype::parse(KvDtype::Q8Block.name()), Ok(KvDtype::Q8Block), "name() must round-trip");
        for bad in ["q4", "int8", "q8 ", "8"] {
            let e = KvDtype::parse(bad).unwrap_err();
            assert!(e.contains("accepted: bf16, fp8, fp8_e4m3, q8"), "{bad:?}: {e}");
        }
        let dense = crate::meta::dense_fixture_geo();
        let fx = Geo::FLASH_NEXT;
        assert_eq!(KvDtype::kv_for(&dense, Some(KvDtype::Q8Block)), Ok(KvDtype::Q8Block));
        assert_eq!(KvDtype::kv_for(&dense, None), Ok(KvDtype::Bf16));
        assert_eq!(KvDtype::kv_for(&fx, None), Ok(KvDtype::Fp8E4m3));
        for k in [KvDtype::Fp8E4m3, KvDtype::Bf16] {
            assert_eq!(KvDtype::kv_for(&dense, Some(k)), Ok(k));
            assert_eq!(KvDtype::kv_for(&fx, Some(k)), Ok(k));
        }
        let e = KvDtype::kv_for(&fx, Some(KvDtype::Q8Block)).unwrap_err();
        assert!(e.contains("full attention only"), "{e}");
        let odd = Geo { head_dim: 80, ..dense };
        let e = KvDtype::kv_for(&odd, Some(KvDtype::Q8Block)).unwrap_err();
        assert!(e.contains("multiple of 32"), "{e}");
        // the q8 kernel text never reaches a module without the phase 2 kernels
        let fk = crate::kernels::KernelGeo { q8kv: true, ..crate::kernels::KernelGeo::flash_next() };
        assert!(std::panic::catch_unwind(|| fk.source()).is_err());
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
        let (cos, sin) = build_rope_table(&Geo::FLASH_NEXT, context, None);
        assert_eq!(cos, cos_ref, "the default cos table moved");
        assert_eq!(sin, sin_ref, "the default sin table moved");
        // a present-but-"default" rope_scaling object configures nothing: same bytes
        let (c2, s2) = build_rope_table(&Geo::FLASH_NEXT, context, Some(&scaling(RopeKind::Default)));
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
        let (cos_y, _) = build_rope_table(&Geo::FLASH_NEXT, 1024, Some(&s));
        let (cos_0, _) = build_rope_table(&Geo::FLASH_NEXT, 1024, None);
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
        let (cos_l, _) = build_rope_table(&Geo::FLASH_NEXT, 64, Some(&scaling(RopeKind::Linear)));
        let (cos_0, _) = build_rope_table(&Geo::FLASH_NEXT, 64, None);
        let (t, j) = (63usize, 0usize);
        assert_eq!(cos_l[t * ROPE_PAIRS + j], (63f32 * 0.25f32).cos(), "linear: even pair 0 interpolates");
        assert_ne!(cos_l[t * ROPE_PAIRS + j], cos_0[t * ROPE_PAIRS + j]);
        let s = scaling(RopeKind::NtkAware);
        let b = s.ntk_base(64, 1e7) as f32;
        let (cos_n, _) = build_rope_table(&Geo::FLASH_NEXT, 64, Some(&s));
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

#[cfg(test)]
mod tests_300_c5 {
    //! Crow #300 C5: the state plan per family. Pure arithmetic, no GPU.
    use super::*;
    use crate::cache::{park_host_bytes, Shape};

    /// Flash-Next selects the plan of record: every byte count of the gate's
    /// `[budget]` lines (200k context, fp8 KV, chunk 512: KV 2343.8 MB, QSA keys
    /// 3.0 MB on a 516-row ring, QSA pooled 293.0 MB, GDN 112.2 MB, RoPE 48.8 MB),
    /// the 121,208,832 B snapshot and the #118 park
    #[test]
    fn flash_next_selects_the_plan_of_record() {
        let g = Geo::FLASH_NEXT;
        let s = StateSizes::plan(&g, 200_000, KvDtype::Fp8E4m3, 512);
        assert_eq!(
            (s.kv_bytes, s.qsa_keys_bytes, s.qsa_ring_rows, s.qsa_pooled_bytes, s.gdn_s_bytes, s.gdn_conv_bytes, s.rope_bytes),
            (2_457_600_000, 3_170_304, 516, 307_200_000, 113_246_208, 4_423_680, 51_200_000)
        );
        assert_eq!(ple_state_len(&g), 92_160, "[10240][9]");
        assert_eq!(Shape::with_geo(&g, 36, 12, 516).snapshot_bytes(), 121_208_832);
        assert_eq!(park_host_bytes(&g, 8_192, 200_000, 12, 1), 113_252_352);
    }

    /// a dense Geo (the 27B fixture) plans no QSA ring or pool, no PLE state and
    /// no hot set, and passes the family check (Crow #300 phase 2)
    #[test]
    fn a_dense_geo_plans_no_ple_qsa_or_hot_set_parts() {
        let g = crate::meta::dense_fixture_geo();
        let s = StateSizes::plan(&g, g.context_floor, KvDtype::Fp8E4m3, 512);
        assert_eq!((s.qsa_keys_bytes, s.qsa_ring_rows, s.qsa_pooled_bytes), (0, 0, 0), "no QSA indexer");
        // 16 attention layers x 2 x 4 kv-heads x 256 x 64k fp8; 48 GDN layers
        assert_eq!(s.kv_bytes, 2_147_483_648);
        assert_eq!((s.gdn_s_bytes, s.gdn_conv_bytes, s.rope_bytes), (150_994_944, 5_898_240, 16_777_216));
        assert_eq!(s.total(), 2_147_483_648 + 150_994_944 + 5_898_240 + 16_777_216);
        assert_eq!(ple_state_len(&g), 0, "no PLE state");
        let shape = Shape::with_geo(&g, 48, 16, 516);
        assert_eq!((shape.qsa_ring_len, shape.ple_len), (0, 0));
        assert_eq!(shape.snapshot_bytes(), 4 * 48 * (48 * 128 * 128 + 10_240 * 3));
        assert_eq!(park_host_bytes(&g, 8_192, 100_000, 16, 1), 8_192 * 16 * 2 * 4 * 256, "KV rows only, no pooled blocks");
        assert_eq!(crate::boot::hot_set_sidecar(&g, Some("hot.json".into()), "d.json".into()), None, "no hot set");
        assert_eq!(g.context_floor, DENSE_CONTEXT_FLOOR);
        assert_eq!(g.built(), Ok(()));
    }

    /// Crow #300 phase 2: the dense plan is one sum. It fits, grants the render reserve
    /// from what is left (best-effort), and refuses by name when the states do not fit.
    #[test]
    fn the_dense_plan_fits_grants_the_reserve_best_effort_and_refuses_by_name() {
        const G: u64 = 1 << 30;
        let (granted, lines) = dense_fit(20 * G, 7 * G, G, 2 * G, 200_000).unwrap();
        assert_eq!(granted, 2 * G);
        assert!(lines[0].starts_with("dense FFN, no hot set: states 7.00 GiB + pending 1.00 GiB"), "{}", lines[0]);
        assert!(lines[1].contains("both") || lines[1].contains("holds"), "{}", lines[1]);
        let left = 20 * G - 8 * G - SAFETY;
        let (granted, lines) = dense_fit(20 * G, 7 * G, G, 64 * G, 200_000).unwrap();
        assert_eq!(granted, left, "granted what is left, not refused");
        assert!(lines[1].contains("binds"), "{}", lines[1]);
        let why = dense_fit(8 * G, 7 * G, G, 0, 200_000).unwrap_err();
        assert!(why.starts_with("refusing config: context 200000 needs 7.00 GiB of states"), "{why}");
    }
}

#[cfg(test)]
mod tests_ram_margin {
    use super::*;

    /// 2026-10-06: the default margin is 1 GiB (was 3; 2 for a few hours).
    /// Without the env var the two readers (budget, pre-pin gate) get 1 GiB.
    #[test]
    fn the_default_ram_margin_is_one_gib() {
        assert_eq!(RAM_MARGIN_DEFAULT_GB, 1);
        if std::env::var_os("CROW_RAM_MARGIN_GB").is_none() {
            assert_eq!(ram_margin_bytes(), 1u64 << 30);
        }
    }
}

#[cfg(test)]
mod tests_159_record_plan {
    //! #159: the plan of Flash-Next and the 27B, rendered as text and pinned. The text was
    //! captured at 0c90901 (before the glm5_next family existed) with exactly this test body;
    //! the family, its planner and its Stability wiring must leave every number unchanged.
    use super::*;

    /// one hot-set unit of Flash-Next: 48 layers x 2,764,800 B per expert (tests_110)
    const UNIT: u64 = 48 * 2_764_800;

    fn render(geo: &Geo) -> String {
        let mut out = format!("family {:?} code {}\n", geo.family, geo.family.code());
        for context in [geo.context_floor, 262_144] {
            for kv in [KvDtype::Fp8E4m3, KvDtype::Bf16, KvDtype::Q8Block] {
                for chunk in [512, 4096] {
                    let s = StateSizes::plan(geo, context, kv, chunk);
                    out += &format!(
                        "ctx {context} {} chunk {chunk}: kv {} qsa_keys {} ring {} pooled {} gdn_s {} gdn_conv {} rope {} total {}\n",
                        kv.name(), s.kv_bytes, s.qsa_keys_bytes, s.qsa_ring_rows, s.qsa_pooled_bytes,
                        s.gdn_s_bytes, s.gdn_conv_bytes, s.rope_bytes, s.total()
                    );
                }
            }
        }
        let card = 32_607u64 << 20;
        let st = Stability::of(geo.family);
        let slots = st.stage_slots(geo.dims().topk, 64, true);
        out += &format!(
            "stability {:?} ceiling {} reserve {} slots {:?} held {} pending {}\n",
            st, st.vram_ceiling(card), st.planner_reserve(card), slots, slots.held(),
            planner_pending(128 << 20, 0, 0, 0)
        );
        let states = StateSizes::plan(geo, geo.context_floor, geo.family.default_kv(), 2048).total();
        match geo.ffn {
            Ffn::Moe { experts, .. } => {
                for (free0, budget) in [(30u64 << 30, 46u64 << 30), (31 << 30, 40 << 30), (28 << 30, 46 << 30)] {
                    let c = ClampInput {
                        n_hot: 160,
                        experts,
                        states_bytes: states,
                        pending_bytes: planner_pending(128 << 20, 0, 0, 0),
                        expert_bytes_per_n_unit: UNIT,
                        cold_bytes_per_n_unit: UNIT,
                        cold_fixed: false,
                        spare: 7,
                        free0,
                        host_pinned_budget: budget,
                    };
                    out += &format!("clamp free0 {free0} budget {budget}: {:?}\n", clamp_hot_n(&c).map(|(n, _)| n));
                }
            }
            Ffn::Dense { .. } => {
                for free0 in [20u64 << 30, 30 << 30] {
                    out += &format!("dense_fit free0 {free0}: {:?}\n", dense_fit(free0, states, 128 << 20, 0, geo.context_floor).map(|(g, _)| g));
                }
            }
        }
        out
    }

    const FLASH_NEXT_PLAN: &str = "family FlashNext code 1\nctx 200000 fp8_e4m3 chunk 512: kv 2457600000 qsa_keys 3170304 ring 516 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 2936840192\nctx 200000 fp8_e4m3 chunk 4096: kv 2457600000 qsa_keys 25190400 ring 4100 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 2958860288\nctx 200000 bf16 chunk 512: kv 4915200000 qsa_keys 3170304 ring 516 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 5394440192\nctx 200000 bf16 chunk 4096: kv 4915200000 qsa_keys 25190400 ring 4100 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 5416460288\nctx 200000 q8 chunk 512: kv 2611200000 qsa_keys 3170304 ring 516 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 3090440192\nctx 200000 q8 chunk 4096: kv 2611200000 qsa_keys 25190400 ring 4100 pooled 307200000 gdn_s 113246208 gdn_conv 4423680 rope 51200000 total 3112460288\nctx 262144 fp8_e4m3 chunk 512: kv 3221225472 qsa_keys 3170304 ring 516 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 3811827712\nctx 262144 fp8_e4m3 chunk 4096: kv 3221225472 qsa_keys 25190400 ring 4100 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 3833847808\nctx 262144 bf16 chunk 512: kv 6442450944 qsa_keys 3170304 ring 516 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 7033053184\nctx 262144 bf16 chunk 4096: kv 6442450944 qsa_keys 25190400 ring 4100 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 7055073280\nctx 262144 q8 chunk 512: kv 3422552064 qsa_keys 3170304 ring 516 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 4013154304\nctx 262144 q8 chunk 4096: kv 3422552064 qsa_keys 25190400 ring 4100 pooled 402653184 gdn_s 113246208 gdn_conv 4423680 rope 67108864 total 4035174400\nstability Stability { vram_headroom: 0, vram_cap: None, decode_stage_rows: None } ceiling 34190917632 reserve 0 slots StageSlots { decode: 20, prefill: 128, shared: true } held 128 pending 134217728\nclamp free0 32212254720 budget 49392123904: Ok(160)\nclamp free0 33285996544 budget 42949672960: Ok(196)\nclamp free0 30064771072 budget 49392123904: Ok(160)\n";
    const DENSE_27B_PLAN: &str = "family Qwen35Dense code 2\nctx 65536 fp8_e4m3 chunk 512: kv 2147483648 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 2321154048\nctx 65536 fp8_e4m3 chunk 4096: kv 2147483648 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 2321154048\nctx 65536 bf16 chunk 512: kv 4294967296 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 4468637696\nctx 65536 bf16 chunk 4096: kv 4294967296 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 4468637696\nctx 65536 q8 chunk 512: kv 2281701376 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 2455371776\nctx 65536 q8 chunk 4096: kv 2281701376 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 16777216 total 2455371776\nctx 262144 fp8_e4m3 chunk 512: kv 8589934592 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 8813936640\nctx 262144 fp8_e4m3 chunk 4096: kv 8589934592 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 8813936640\nctx 262144 bf16 chunk 512: kv 17179869184 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 17403871232\nctx 262144 bf16 chunk 4096: kv 17179869184 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 17403871232\nctx 262144 q8 chunk 512: kv 9126805504 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 9350807552\nctx 262144 q8 chunk 4096: kv 9126805504 qsa_keys 0 ring 0 pooled 0 gdn_s 150994944 gdn_conv 5898240 rope 67108864 total 9350807552\nstability Stability { vram_headroom: 0, vram_cap: None, decode_stage_rows: None } ceiling 34190917632 reserve 0 slots StageSlots { decode: 0, prefill: 128, shared: true } held 128 pending 134217728\ndense_fit free0 21474836480: Ok(0)\ndense_fit free0 32212254720: Ok(0)\n";

    #[test]
    fn flash_next_and_the_27b_plan_exactly_as_before_159() {
        assert_eq!(render(&Geo::FLASH_NEXT), FLASH_NEXT_PLAN);
        assert_eq!(render(&crate::meta::dense_fixture_geo()), DENSE_27B_PLAN);
    }
}


#[cfg(test)]
mod tests_159_glm_plan {
    //! #159: the glm5_next plan: states from the recipe's per-token and per-layer bytes, the
    //! #176 policy through `planner_pending_for`, three expert tiers. Pure arithmetic, no GPU.
    use super::*;

    const CARD: u64 = 32_607 << 20; // RTX 5090, docs/system-landscape.md:12

    fn plan(card: u64, pinned: u64, context: usize) -> Result<(Glm5States, TierInput, TierPlan), String> {
        plan_glm5_next(&Glm5Geo::GLM_5_3_FLASH, context, card, pinned, GLM5_NEXT_DENSE_BYTES, GLM5_NEXT_EXPERT_BLOCK_BYTES, 64, true)
    }

    /// the plan of record at the boot context 200,000 on the RTX 5090 with the 46 GiB cap:
    /// N 32 in VRAM, P 83 pinned, 173 on NVMe per MoE layer; KV from the latent, not per head
    #[test]
    fn the_glm_plan_at_the_boot_context_is_n32_p83_nvme173() {
        let (s, i, p) = plan(CARD, HOST_PINNED_CAP, 200_000).unwrap();
        assert_eq!((s.latent_bytes, s.indexer_bytes), (2_252_800_000, 1_130_800_000));
        assert_eq!((s.kda_state_bytes, s.kda_conv_bytes), (142_606_336, 10_027_008));
        assert_eq!(s.per_token(&Glm5Geo::GLM_5_3_FLASH), 16_918);
        // the per-head K/V form the ticket names as the failure mode: 11 x 2 x 64 x 200k x 256 B
        assert_ne!(s.latent_bytes, 72_089_600_000);
        assert_eq!(i.staging_bytes, 160 * GLM5_NEXT_EXPERT_BLOCK_BYTES, "32 decode + 128 prefill slots (#176)");
        assert_eq!(p.vram_ceiling, 32_043_433_984);
        assert_eq!(p.unit_bytes, 594_542_592);
        assert_eq!((p.hot, p.pinned, p.nvme), (32, 83, 173));
        assert_eq!(p.fixed_bytes, GLM5_NEXT_DENSE_BYTES + Glm5Geo::GLM_5_3_FLASH.kv_b_decode_bytes() + s.total() + i.staging_bytes + LAUNCH_SLACK + SAFETY);
        assert!(p.fixed_bytes + p.hot_bytes() <= p.vram_ceiling && p.fixed_bytes + p.hot_bytes() + p.unit_bytes > p.vram_ceiling);
        assert!(p.pinned_bytes() <= HOST_PINNED_CAP);
        assert_eq!(p.hot_bytes() + p.pinned_bytes() + p.nvme_bytes(), GLM5_NEXT_EXPERT_BLOCKS * GLM5_NEXT_EXPERT_BLOCK_BYTES);
    }

    /// #161: the plan books kv_b decoded to BF16 at load (the MLA kernels read BF16; the container
    /// stores it NVFP4, inside the dense part): 11 x 16,777,216 values, BF16 369,098,752 B minus
    /// NVFP4 103,809,024 B = 265,289,728 B more VRAM before any expert, named in the printout
    #[test]
    fn the_glm_plan_books_kv_b_decoded_to_bf16() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        assert_eq!((g.kv_b_values(), g.kv_b_bf16_bytes(), g.kv_b_nvfp4_bytes(), g.kv_b_decode_bytes()), (16_777_216, 369_098_752, 103_809_024, 265_289_728));
        let (s, i, p) = plan(CARD, HOST_PINNED_CAP, 200_000).unwrap();
        assert_eq!(i.decoded_bytes, 265_289_728);
        assert_eq!(p.fixed_bytes, GLM5_NEXT_DENSE_BYTES + 265_289_728 + s.total() + i.staging_bytes + LAUNCH_SLACK + SAFETY);
        let t = glm5_plan_table(&g, &s, &i, &p, &[]);
        let want = "  kv_b at BF16                265289728 B     0.25 GiB  11 DSA layers: kv_b decoded at load, BF16 369098752 B - container NVFP4 103809024 B (#161)";
        assert!(t.contains(want), "{want:?} missing:
{t}");
    }

    /// a card that cannot hold the dense part and the states is refused by name; the
    /// smallest card that holds them plans N 0 and puts every other expert below VRAM
    #[test]
    fn a_budget_that_cannot_hold_the_dense_part_is_refused() {
        let why = plan(8 << 30, HOST_PINNED_CAP, 200_000).unwrap_err();
        assert!(why.starts_with("refusing config: family Glm5Next needs"), "{why}");
        assert!(why.contains("the dense part does not fit, no expert tier changes that (#159)"), "{why}");
        let (_, i, p) = plan(CARD, HOST_PINNED_CAP, 200_000).unwrap();
        let tight = p.fixed_bytes + i.stability.vram_headroom;
        let (_, _, q) = plan(tight, HOST_PINNED_CAP, 200_000).unwrap();
        assert_eq!((q.hot, q.pinned, q.nvme), (0, 83, 205));
        assert!(plan(tight - 1, HOST_PINNED_CAP, 200_000).is_err());
    }

    /// the tiers follow their inputs: no pinned budget sends the rest to NVMe, a long context
    /// takes VRAM from N, the 31.9 GiB cap binds on a larger card
    #[test]
    fn the_glm_tiers_follow_the_budget_the_context_and_the_cap() {
        let (_, _, p) = plan(CARD, 0, 200_000).unwrap();
        assert_eq!((p.hot, p.pinned, p.nvme), (32, 0, 256));
        let (s, _, p) = plan(CARD, HOST_PINNED_CAP, 1_048_576).unwrap();
        assert_eq!(s.latent_bytes, 11 * 1_048_576 * 1_024);
        assert!(p.hot < 32, "N {}", p.hot);
        let (_, _, p) = plan(48 << 30, HOST_PINNED_CAP, 200_000).unwrap();
        assert_eq!(p.vram_ceiling, GLM5_NEXT_VRAM_CAP);
    }

    /// `planner_pending_for` adds the policy reserve: 0 B for Flash-Next and the 27B (their
    /// pending unchanged), 2 GiB for glm5_next on the RTX 5090
    #[test]
    fn planner_pending_for_adds_only_the_glm_reserve() {
        for f in Family::ALL {
            assert_eq!(planner_pending_for(&Stability::of(f), CARD, LAUNCH_SLACK, 7, 11, 13), planner_pending(LAUNCH_SLACK, 7, 11, 13), "{f:?}");
        }
        assert_eq!(planner_pending_for(&Stability::of(Family::Glm5Next), CARD, LAUNCH_SLACK, 0, 0, 0), LAUNCH_SLACK + (2 << 30));
        assert_eq!(LAUNCH_SLACK, 128 << 20);
    }

    /// the printout names its inputs and the three tiers, and says step 3 is pending
    #[test]
    fn the_glm_plan_table_names_its_inputs_and_the_three_tiers() {
        let (s, i, p) = plan(CARD, HOST_PINNED_CAP, 200_000).unwrap();
        let t = glm5_plan_table(&Glm5Geo::GLM_5_3_FLASH, &s, &i, &p, &[("dense", "converter dry run".into())]);
        for want in [
            "context 200000",
            "dense part                 5981546744 B     5.57 GiB (converter dry run)",
            "expert block                 14155776 B",
            "KV per token                    16918 B  11 DSA layers x (MLA latent 1024 B BF16 + indexer 514 B HF layout)",
            "plan ceiling              32043433984 B",
            "VRAM    N =  32",
            "pinned  P =  83",
            "NVMe        173",
            "G1d capacities (plan step 3): not measured yet",
        ] {
            assert!(t.contains(want), "{want:?} missing:\n{t}");
        }
    }
}

#[cfg(test)]
mod tests_rec_size {
    //! #159 / #176 / #149: the glm5_next plan takes the routed-expert record as a parameter.
    //! The 14,155,776-B plan of record stays golden in `tests_159_glm_plan` (unchanged); this
    //! module plans the 3.05-bpw MUL1 figure of 9,474,048 B and the refusals.
    use super::*;

    const CARD: u64 = 32_607 << 20; // RTX 5090, docs/system-landscape.md:12
    /// the plan's 3.05-bpw MUL1 record (2313 x 4096)
    const MUL1_PLAN: u64 = 9_474_048;

    fn plan(rec: u64, pinned: u64) -> Result<(Glm5States, TierInput, TierPlan), String> {
        plan_glm5_next(&Glm5Geo::GLM_5_3_FLASH, 200_000, CARD, pinned, GLM5_NEXT_DENSE_BYTES, rec, 64, true)
    }

    /// at 9,474,048 B per record on the RTX 5090, 200,000 tokens, the 46 GiB cap: the unit is
    /// 42 x 9,474,048 = 397,910,016 B, the staging 160 x 9,474,048 = 1,515,847,680 B, and the
    /// tiers per MoE layer are N 50 in VRAM, P 124 pinned, 114 on NVMe (4.5 bit: 32 / 83 / 173).
    /// #161: was N 51 / NVMe 113 before the plan booked kv_b's BF16 decode (265,289,728 B the
    /// MLA path allocates on top of the container's dense part): one VRAM unit fewer.
    #[test]
    fn the_glm_plan_at_the_3bit_record_is_n50_p124_nvme114() {
        let (s, i, p) = plan(MUL1_PLAN, HOST_PINNED_CAP).unwrap();
        assert_eq!(i.expert_block_bytes, MUL1_PLAN);
        assert_eq!(i.staging_bytes, 1_515_847_680, "32 decode + 128 prefill slots of 9,474,048 B (#176)");
        assert_eq!(p.unit_bytes, 397_910_016);
        assert_eq!((p.hot, p.pinned, p.nvme), (50, 124, 114));
        assert_eq!(p.fixed_bytes, GLM5_NEXT_DENSE_BYTES + Glm5Geo::GLM_5_3_FLASH.kv_b_decode_bytes() + s.total() + i.staging_bytes + LAUNCH_SLACK + SAFETY);
        assert!(p.fixed_bytes + p.hot_bytes() <= p.vram_ceiling && p.fixed_bytes + p.hot_bytes() + p.unit_bytes > p.vram_ceiling);
        assert!(p.pinned_bytes() <= HOST_PINNED_CAP && p.pinned_bytes() + p.unit_bytes > HOST_PINNED_CAP);
        let t = glm5_plan_table(&Glm5Geo::GLM_5_3_FLASH, &s, &i, &p, &[]);
        for want in [
            format!("  expert block         {:>16} B", MUL1_PLAN),
            format!("  cold staging         {:>16} B", 1_515_847_680u64),
            "unit = 397910016 B = 42 layers x one block".to_string(),
            "VRAM    N =  50".to_string(),
            "pinned  P = 124".to_string(),
            "NVMe        114".to_string(),
        ] {
            assert!(t.contains(&want), "{want:?} missing:\n{t}");
        }
    }

    /// the staging and the unit scale with the record; with the same inputs the 14,155,776-B
    /// plan is the one of record
    #[test]
    fn staging_and_capacities_follow_the_record() {
        let (_, i45, p45) = plan(GLM5_NEXT_EXPERT_BLOCK_BYTES, HOST_PINNED_CAP).unwrap();
        let (_, i3, p3) = plan(MUL1_PLAN, HOST_PINNED_CAP).unwrap();
        assert_eq!((p45.hot, p45.pinned, p45.nvme), (32, 83, 173));
        assert_eq!(i45.staging_bytes / GLM5_NEXT_EXPERT_BLOCK_BYTES, i3.staging_bytes / MUL1_PLAN);
        assert_eq!(p3.unit_bytes * GLM5_NEXT_EXPERT_BLOCK_BYTES, p45.unit_bytes * MUL1_PLAN);
        assert!(p3.hot > p45.hot && p3.pinned > p45.pinned && p3.nvme < p45.nvme);
    }

    /// a record that is not a whole number of 4096-B sectors, or 0 B, is refused by name
    #[test]
    fn the_planner_refuses_a_record_off_the_sector_grid() {
        let why = plan(MUL1_PLAN - 48, HOST_PINNED_CAP).unwrap_err();
        assert!(why.starts_with("refusing expert record of 9474000 B: not a multiple of 4096 B"), "{why}");
        assert!(plan(0, HOST_PINNED_CAP).unwrap_err().contains("0 B"));
    }

    /// `states --plan` takes the record from exactly one source; no default
    #[test]
    fn the_plan_record_has_one_source_and_no_default() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let why = glm5_plan_record(&g, None, None, None).unwrap_err();
        assert!(why.contains("--cnq <container>") && why.contains("--expert-bytes N") && why.contains("there is no default"), "{why}");
        assert!(glm5_plan_record(&g, Some("x.cnq"), Some("9474048"), None).unwrap_err().contains("not both"));
        assert!(glm5_plan_record(&g, Some("x.cnq"), None, Some("mul1")).unwrap_err().contains("--expert-codec goes with --expert-bytes"));
        let r = glm5_plan_record(&g, None, Some("9474048"), Some("mul1")).unwrap();
        assert_eq!((r.codec, r.bytes), (Some(ExpertCodec::Mul1), MUL1_PLAN));
        assert!(r.source.contains("codec mul1"), "{}", r.source);
        let r = glm5_plan_record(&g, None, Some("9474048"), None).unwrap();
        assert_eq!((r.codec, r.bytes), (None, MUL1_PLAN));
        assert!(glm5_plan_record(&g, None, Some("9474000"), Some("mul1")).unwrap_err().contains("not a multiple of 4096 B"));
        assert!(glm5_plan_record(&g, None, Some("9.4e6"), None).unwrap_err().contains("not a whole number of bytes"));
        assert!(glm5_plan_record(&g, None, Some("9474048"), Some("q3k")).unwrap_err().starts_with("refusing expert codec \"q3k\""));
        // nvfp4 is the config's own record or nothing
        assert_eq!(glm5_plan_record(&g, None, Some("14155776"), Some("nvfp4")).unwrap().bytes, GLM5_NEXT_EXPERT_BLOCK_BYTES);
        assert!(glm5_plan_record(&g, None, Some("9474048"), Some("nvfp4")).unwrap_err().contains("the config derives 14155776 B at nvfp4"));
        // a container that is not there is a named error, not a fallback
        assert!(glm5_plan_record(&g, Some("no-such-container.cnq"), None, None).is_err());
    }
}

#[cfg(test)]
mod tests_186_chunk_plan {
    //! #186: the glm5_next plan with a prompt chunk (`CROW_CHUNK`): chunk 1 is the plan of
    //! record byte for byte; a chunk books its prompt-phase bytes before any expert and the
    //! prefill landing off the pinned budget; a chunk that does not fit is refused by name.
    use super::*;

    const CARD: u64 = 32_607 << 20; // RTX 5090, docs/system-landscape.md:12
    const MUL1_PLAN: u64 = 9_474_048;

    fn plan(chunk: usize) -> Result<(Glm5States, TierInput, TierPlan), String> {
        plan_glm5_next_chunk(&Glm5Geo::GLM_5_3_FLASH, 200_000, CARD, HOST_PINNED_CAP, GLM5_NEXT_DENSE_BYTES, MUL1_PLAN, 64, true, chunk)
    }

    #[test]
    fn chunk_one_is_the_plan_of_record_and_a_chunk_books_its_bytes() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (s0, i0, p0) = plan_glm5_next(&g, 200_000, CARD, HOST_PINNED_CAP, GLM5_NEXT_DENSE_BYTES, MUL1_PLAN, 64, true).unwrap();
        let (s1, i1, p1) = plan(1).unwrap();
        assert_eq!((s0.total(), p0), (s1.total(), p1));
        assert_eq!((i1.chunk, i1.chunk_scratch_bytes, i1.host_pinned_budget), (1, 0, i0.host_pinned_budget));
        assert_eq!(glm5_chunk_scratch_bytes(&g, 1, 200_000), 0);
        let mut last = 0;
        // the prompt phase books one expert-major MoE plan of `chunk` rows (about 0.3 MB per row),
        // so chunk 8192 (glm53-flash-offload's chunk_size) plans on the card
        for chunk in [2, 16, 32, 128, 1024, 2048, 4096, 8192] {
            let (_, i, p) = plan(chunk).unwrap();
            let b = glm5_chunk_scratch_bytes(&g, chunk, 200_000);
            assert!(b > last, "chunk {chunk}: {b} B, not above {last}");
            last = b;
            assert_eq!((i.chunk, i.chunk_scratch_bytes), (chunk, b));
            assert_eq!(i.host_pinned_budget, HOST_PINNED_CAP - GLM5_PREFILL_RING as u64 * MUL1_PLAN, "the pinned prefill landing ring comes off the pinned budget");
            assert_eq!(p.fixed_bytes, p0.fixed_bytes + b, "the chunk's bytes are booked before any expert");
            assert!(p.hot <= p0.hot && p.hot + p.pinned + p.nvme == g.experts);
            let t = glm5_plan_table(&g, &s0, &i, &p, &[]);
            assert!(t.contains(&format!("{chunk} rows per prompt call (CROW_CHUNK, #186)")), "{t}");
            eprintln!("glm5 plan #186 chunk {chunk}: prompt-phase bytes {b} ({:.2} GiB), N {} P {} NVMe {} (chunk 1: {} / {} / {})", b as f64 / GIB, p.hot, p.pinned, p.nvme, p0.hot, p0.pinned, p0.nvme);
        }
        // #196: 65,536 rows fit since the scratch is about 0.37 MB per row; twice that does not
        let e = plan(131_072).unwrap_err();
        assert!(e.contains("prompt chunk 131072 (CROW_CHUNK, #186)"), "{e}");
    }
}

#[cfg(test)]
mod tests_196_scratch {
    //! #196: the prompt-pass scratch the plan books at 8192 rows per call over a cache of
    //! 200,000 rows (`CROW_CHUNK=8192`, the template's chunk) is at most 3.5 GiB: the KDA and MLA
    //! scratch share one region at their sub-block rows, the indexer scores are one block of 256
    //! rows, gate / up live per piece and the dense FFN shares the MoE plan's region. Printed per
    //! row by buffer.
    use super::*;

    #[test]
    fn glm5_prompt_scratch_at_8192_rows_is_at_most_3_5_gib() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (chunk, cap) = (8192usize, 200_000usize);
        let parts = glm5_chunk_scratch_parts(&g, chunk, cap, false);
        let total: u64 = parts.iter().map(|p| p.1).sum();
        assert_eq!(total, glm5_chunk_scratch_bytes_tc(&g, chunk, cap, false));
        for (name, b) in &parts {
            eprintln!("#196 chunk {chunk} cap {cap}: {name:<10} {b:>13} B  {:>9.1} B/row", *b as f64 / chunk as f64);
        }
        eprintln!("#196 chunk {chunk} cap {cap}: total      {total:>13} B  {:>9.1} B/row  {:.3} GiB", total as f64 / chunk as f64, total as f64 / GIB);
        assert!(total <= 7 << 29, "the prompt scratch at {chunk} rows books {total} B ({:.2} GiB), above 3.5 GiB", total as f64 / GIB);
        // the attention region is the larger of the two scratch kinds, not their sum
        let attn = parts.iter().find(|p| p.0 == "attention").unwrap().1;
        let (kd, md) = (crate::glm5_kda::KdaDims::of(&g), crate::glm5_mla::MlaDims::of(&g));
        let (k, m) = crate::glm5_model::attn_rows(chunk);
        let (kb, mb) = (crate::glm5_kda::KdaScratch::region_bytes(&kd, k) as u64, crate::glm5_mla::MlaScratch::region_bytes(&md, m, cap) as u64);
        let one = crate::glm5_model::attn_region_bytes(&g, crate::glm5_model::attn_rows(1), cap) as u64;
        assert_eq!(attn + one, kb.max(mb));
        // the indexer scores do not grow with the call: a 262,144-row cache adds one 256-row block
        let big = glm5_chunk_scratch_bytes_tc(&g, chunk, 262_144, false);
        assert!(big <= 7 << 29, "cap 262144: {big} B");
    }
}
