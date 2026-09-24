//! Model geometry (probe-pinned, qwen4_exp text tower) + runtime config.

pub const H: usize = 2560; // hidden
pub const HCN: usize = 4; // hyper-connection streams
pub const HCT: usize = HCN * H; // 10240 residual stream
pub const LOWRANK: usize = 320;
pub const E: usize = 512; // experts per layer
pub const TOPK: usize = 10;
pub const INTER: usize = 640; // per-expert intermediate
pub const GDN_KEY: usize = GDN_KHEADS * GD; // 2048
pub const GDN_VAL: usize = 6144;
pub const GDN_CONV: usize = GDN_KEY * 2 + GDN_VAL; // 10240
pub const GDN_KHEADS: usize = 16;
pub const GDN_VHEADS: usize = 48;
pub const GD: usize = 128; // GDN head dim
pub const NQ: usize = 24; // attention q heads
pub const NKV: usize = 2;
pub const AHD: usize = 256; // attention head dim
pub const Q_ROWS: usize = NQ * AHD * 2; // 12288 (query+gate)
pub const KV_ROWS: usize = NKV * AHD; // 512
pub const CORE: usize = NQ * AHD; // 6144
pub const V: usize = 248320; // vocab
pub const LAYERS: usize = 48;
pub const GDN_LAYERS: usize = 36;
pub const ATTN_LAYERS: usize = 12; // every 4th (layer % 4 == 3)
pub const ROPE_PAIRS: usize = 32; // partial rotary 64 of 256 dims
pub const PLE_LAYER: usize = 1; // layer index (config ple_layer_ids [2] is 1-based)
pub const PLE_NGRAM: usize = 3;
pub const PLE_CTX: usize = PLE_NGRAM - 1;
pub const PLE_HEADS_PER_NGRAM: usize = 8;
pub const PLE_NHEADS: usize = PLE_CTX * PLE_HEADS_PER_NGRAM; // 16
pub const PLE_EMB_DIM: usize = 160;
pub const PLE_EMBED: usize = PLE_NHEADS * PLE_EMB_DIM; // 2560
pub const PLE_ROWS_PER_SHARD: i64 = 2_500_012;
pub const PLE_EOS: i64 = 248044;
/// CAP on the pinned cold tier, and the ONE place to lower it for a smaller
/// host. Measured host ceiling ~48.5 GB with ~2.5 GB of margin - only their
/// DIFFERENCE is a number this code can hold, so it is written as one
/// constant. The operating value is DERIVED from the running host at boot by
/// `manager::derive_host_pinned_budget`, which takes the smaller of this cap
/// and `free_for_pin - CROW_RAM_MARGIN_GB`.
pub const HOST_PINNED_CAP: u64 = 46 << 30;

// ---- the prompt-chunk family (see `apply_chunk_policy`) ----

/// prompt chunks are rounded UP to a multiple of this and never fall below it;
/// it is also the `Config::default` chunk (the decode/parity operating point)
pub const CHUNK_ROUND: usize = 512;
/// #10b: the auto policy never raises the chunk above this (the scratch diet
/// lifted the F52 wall from 2048 to 4096)
pub const CHUNK_CAP: usize = 4096;
/// At or above this chunk the default adapt policy arms the STREAM trickle -
/// and it is exactly the chunk `serve` pins (`bin/serve.rs::SERVE_CHUNK`
/// derives from it). That is not a coincidence: serve's chunk choice is what
/// arms the trickle, and the 4096 experiment of 2026-09-14 collapsed the live
/// serve prefill because the trickle/adapt policy is tuned for THIS number.
pub const TRICKLE_CHUNK_THRESHOLD: usize = 2048;
/// Decode tokens between two trickle ticks in the long-context policy (TASK I,
/// 2026-09-17: 8, was 16 since #17 — the table in `apply_adapt_policy` says what
/// each value measured on the multi-turn serve shape and on the 16k single prompt).
pub const TRICKLE_EVERY: usize = 8;

// ---- units and default artefact paths ----

/// f64 divisors for the byte-size log lines. `(1u64 << 20) as f64` is EXACT in
/// f64, so every number printed through these is bit-identical to the inline
/// `/ (1 << 20) as f64` form that used to be written out at 54 sites.
pub const MIB: f64 = (1u64 << 20) as f64;
pub const GIB: f64 = (1u64 << 30) as f64;

/// The container and hot-set sidecar of record, RELATIVE TO THE REPO ROOT.
/// `serve` and `parity` run from there and use them as written; the bins that
/// run from `engine/` wrap them in `from_engine_dir`. Nine sites used to spell
/// these two strings out across two CWD conventions.
pub const DEFAULT_CNQ: &str = "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq";
// 2026-09-24: cut on the generated positions of three Crow goal-mode sessions
// (tools/hotset-eval.py, docs/hotset-calibration.md); the gates of record keep
// pinning hotsets-M-longctx2100-n160.json through CROW_HOTSETS (tools/gate-linux.sh).
pub const DEFAULT_HOTSETS: &str = "decode_out/hotsets-M-crow0924-n160.json";

/// a repo-root-relative path as seen from `engine/` (where `cargo run`,
/// `cargo test` and the probe bins start)
pub fn from_engine_dir(p: &str) -> String {
    format!("../{p}")
}

pub const QSA_HEADS: usize = 4;
pub const QSA_KVHEADS: usize = 1;
pub const QSA_HD: usize = 128;
pub const QSA_QK_ROWS: usize = (QSA_HEADS + QSA_KVHEADS) * QSA_HD; // 640
pub const QSA_COMPRESS: usize = 4;
pub const QSA_BLOCK_TOPK: usize = 512; // budget 2048 / ratio 4
pub const QSA_SEL_MAX: usize = QSA_BLOCK_TOPK * QSA_COMPRESS + QSA_COMPRESS - 1; // 2051
pub const QSA_HIDD: usize = 128; // indexer raw key width

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum KvDtype {
    Fp8E4m3,
    Bf16,
}

impl KvDtype {
    pub fn byte_per_value(self) -> usize {
        match self {
            KvDtype::Fp8E4m3 => 1,
            KvDtype::Bf16 => 2,
        }
    }
    pub fn name(self) -> &'static str {
        match self {
            KvDtype::Fp8E4m3 => "fp8_e4m3",
            KvDtype::Bf16 => "bf16",
        }
    }

    /// The one parser of a KV dtype word (#102). `bf16`, `fp8` and `fp8_e4m3`
    /// (the `name()` spelling, so a logged value round-trips), ASCII case
    /// ignored. Anything else, the empty string included, is an error that
    /// names the accepted words: a typo must not silently fall back to FP8.
    pub fn parse(s: &str) -> Result<KvDtype, String> {
        match s.to_ascii_lowercase().as_str() {
            "bf16" => Ok(KvDtype::Bf16),
            "fp8" | "fp8_e4m3" => Ok(KvDtype::Fp8E4m3),
            _ => Err(format!("CROW_KV={s:?} is not a KV dtype; accepted: bf16, fp8, fp8_e4m3 (unset = fp8_e4m3)")),
        }
    }

    /// `CROW_KV` as read at the `boot::open_model` front door, shared by
    /// `decode`, `parity` and `serve` (#102). `None` = unset, keep the default;
    /// a set value goes through `parse`, including the error.
    pub fn from_env_value(v: Option<&str>) -> Result<Option<KvDtype>, String> {
        v.map(KvDtype::parse).transpose()
    }
}

/// runtime configuration — the knobs the loader honours (spec 0.2, 2.1)
#[derive(Clone, Copy)]
pub struct Config {
    pub context: usize,       // default 262_144, floor 200_000
    pub n_hot: usize,         // target 160 experts per layer, loader clamps
    pub kv: KvDtype,          // FP8 default; CROW_KV=bf16 at boot::open_model (#102)
    pub ple_cache_bytes: u64, // hot-row cache, default 128 MB (#16, 2026-09-05; was 1 GB)
    pub prompt_chunk: usize,  // prefill chunk size (correctness stage: 256..512)
    /// pinned-host budget for the cold tier (spec 3.4: ~43-47 GB of 64 GB);
    /// the loader RAISES N if the cold tier would exceed this. The value here is
    /// the CAP: `Engine::load` replaces it with
    /// `manager::derive_host_pinned_budget`, which takes the smaller of this cap
    /// and the measured `free_for_pin - CROW_RAM_MARGIN_GB` of the running host.
    pub host_pinned_budget: u64,
    /// PLE layer on/off (bisector switch for the parity ladder)
    pub ple: bool,
    /// decode-time hot-set adaptation (#17), set by `apply_adapt_policy`
    pub adapt: Adapt,
}

/// Decode-time hot-set adaptation knobs (#17). `stream`: swaps run on a side
/// stream overlapping the next token, with `spare` hot slots per layer taken
/// out of the planned N; otherwise the swaps run on the compute stream inside
/// the token. `every`: re-cut the hot set every K decode tokens (0 = never),
/// at most `max` swaps per layer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Adapt {
    pub stream: bool,
    pub spare: usize,
    pub every: usize,
    pub max: usize,
}

impl Adapt {
    /// the three knobs every decode loop binds, in the order it binds them
    pub fn knobs(&self) -> (bool, usize, usize) {
        (self.stream, self.every, self.max)
    }
}

impl Default for Adapt {
    fn default() -> Self {
        Adapt { stream: false, spare: 0, every: 0, max: 8 }
    }
}

impl Default for Config {
    fn default() -> Self {
        Config {
            context: 262_144,
            n_hot: 160,
            kv: KvDtype::Fp8E4m3,
            ple_cache_bytes: 128 << 20, // #16: measured 2026-09-05, +0.2 % misses vs 1 GB, ~7 units freed
            prompt_chunk: CHUNK_ROUND,
            host_pinned_budget: HOST_PINNED_CAP,
            ple: true,
            adapt: Adapt::default(),
        }
    }
}

pub const CONTEXT_FLOOR: usize = 200_000;

/// per-layer type dispatch (layer % 4 == 3 is full attention — probe-pinned)
pub fn is_attn(layer: usize) -> bool {
    layer % 4 == 3
}
pub fn gdn_index(layer: usize) -> usize {
    layer - layer / 4
}
pub fn attn_index(layer: usize) -> usize {
    layer / 4
}

/// one definition of "a number read from the environment"; every filter, clamp and default stays at the call site
pub fn env_parse<T: std::str::FromStr>(key: &str) -> Option<T> {
    std::env::var(key).ok().and_then(|v| v.parse().ok())
}

/// Prefill chunk policy (#16). `CROW_CHUNK=<n>` is authoritative; without it the
/// chunk follows the prompt length (rounded up to 512, at most 4096), so a short
/// prompt keeps the chunk-512 scratch and its larger hot set and a long prompt
/// takes the fewest PCIe passes over the cold tier. Cap 4096 since #10b
/// (2026-09-13): the per-chunk scratch diet cut the per-token scratch from
/// 1.23 MiB to about 0.42 MiB, so chunk 4096 passes the VRAM planner with the
/// hot set at the pinned-budget side (the cap was 2048 before, N 140 at 2048).
/// `CROW_CHUNK_AUTO=1` applies the policy on top of an explicit `CROW_CHUNK`
/// (cap = max(CROW_CHUNK, 4096)); `CROW_CHUNK_AUTO=0` disables it.
/// Default since 2026-09-05 (auto on; opt-in before). Gated: chunk 1024 and 2048
/// deterministic since #22, chunk 4096 gated by #10b (F47 env-vs-env parity,
/// the 16k parity form, ten-task final4 identity, the F49 prefill pairs).
pub fn apply_chunk_policy(cfg: &mut Config, n_prompt: usize) {
    let explicit = env_parse::<usize>("CROW_CHUNK");
    if let Some(c) = explicit {
        cfg.prompt_chunk = c.max(1);
    }
    let auto = match std::env::var("CROW_CHUNK_AUTO").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => explicit.is_none(),
    };
    if auto {
        let need = ((n_prompt + CHUNK_ROUND - 1) / CHUNK_ROUND * CHUNK_ROUND).max(CHUNK_ROUND);
        cfg.prompt_chunk = need.min(cfg.prompt_chunk.max(CHUNK_CAP));
    }
    apply_adapt_policy(cfg);
}

/// Adaptation policy (#17, 2026-09-05). Measured on the ten-task series
/// (final4 vs trk16): the side-stream trickle with 7 spares / every 16 / max 7
/// gains 0.1 to 0.8 tok/s on every prompt that lands on chunk 2048 (N 147, the
/// spares come out of a hot set that no longer covers the routing) and loses
/// 0.2 to 1.0 tok/s on every prompt at chunk 512 (N 157 already covers it, the
/// seven surrendered slots are pure cost). So the trickle is a long-context
/// switch: with `CROW_ADAPT_STREAM` unset, chunk >= 2048 gets stream / 7 / 8 / 7
/// regardless of `CROW_ADAPT_EVERY` / `CROW_ADAPT_MAX` (which keep describing
/// the short-prompt form), a smaller chunk gets the compute-stream swaps from
/// `CROW_ADAPT_EVERY` (default 0 = none) / `CROW_ADAPT_MAX` (default 8) with
/// `CROW_ADAPT_SPARE` (default 0). `CROW_ADAPT_STREAM=1` / `=0` is the manual
/// mode: every knob from its own variable, spare default 1 / 0, no policy.
///
/// TASK I, 2026-09-17: the tick interval is 8, not the 16 of #17. #17 measured
/// ONE prompt per process, where the hot set has one prefill to be wrong about;
/// a chat session prefills a new turn against the same document again and again,
/// and every expert the hot set does not hold is 2.76 MB over PCIe per turn per
/// layer that holds it. On robin's 6-turn replay (3,296-token prefix, turns of
/// 39 to 101 new ids, `CROW_CHUNK=2048`) the staged cold bytes per warm turn and
/// the prefill wall read, mean over turns 1-6:
///
/// | every | cold bytes staged, turn 1 -> turn 6 | prefill ms | decode ms (6 turns) | swaps |
/// |---|---|---|---|---|
/// | 16 (#17) | 10.1 GB -> 7.9 GB | 267.7 | 5216 | 4,030 |
/// | 8 (this) | 9.4 GB -> 6.3 GB | 231.3 | 5033 | 9,160 |
/// | 4 | 8.1 GB -> 6.0 GB | 209.3 | 5128 | 16,134 |
///
/// and the ids of all three arms are identical, turn for turn (a swap moves
/// bytes between tiers, it never changes a number). 8 is the value taken: it
/// wins on both shapes measured (the replay above and `decode run` on the 16k
/// t1-read prompt: 26.89 / 26.89 ms per decode token at 16 against 26.76 / 26.68
/// at 8, cold experts per token 230.9 -> 216.5). 4 buys another 22 ms of prefill
/// but doubles the swap traffic again for a total turn time inside 0.6 % of 8,
/// and it ranks the hot set on a `CROW_ADAPT_DECAY` window of four decode tokens.
/// Both stay reachable through the manual mode (`CROW_ADAPT_STREAM=1` plus
/// `CROW_ADAPT_EVERY`).
pub fn apply_adapt_policy(cfg: &mut Config) {
    let num = env_parse::<usize>;
    let (every, max, spare) = (num("CROW_ADAPT_EVERY"), num("CROW_ADAPT_MAX"), num("CROW_ADAPT_SPARE"));
    let stream = std::env::var("CROW_ADAPT_STREAM").ok();
    cfg.adapt = match stream.as_deref() {
        Some("1") => Adapt { stream: true, spare: spare.unwrap_or(1), every: every.unwrap_or(0), max: max.unwrap_or(8) },
        Some(_) => Adapt { stream: false, spare: spare.unwrap_or(0), every: every.unwrap_or(0), max: max.unwrap_or(8) },
        None if cfg.prompt_chunk >= TRICKLE_CHUNK_THRESHOLD => Adapt { stream: true, spare: 7, every: TRICKLE_EVERY, max: 7 },
        None => Adapt { stream: false, spare: spare.unwrap_or(0), every: every.unwrap_or(0), max: max.unwrap_or(8) },
    };
    let a = cfg.adapt;
    tracing::info!(target: "policy",
        "[policy] chunk {} -> {} ({}), {} spare hot slot(s), every {}, max {}/layer",
        cfg.prompt_chunk,
        if a.stream { "stream trickle" } else { "compute-stream swaps" },
        if stream.is_some() { "manual CROW_ADAPT_STREAM" } else { "policy" },
        a.spare, a.every, a.max
    );
}
