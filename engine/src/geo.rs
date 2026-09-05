//! Model geometry (probe-pinned, qwen4_exp text tower) + runtime config.

pub const H: usize = 2560; // hidden
pub const HCN: usize = 4; // hyper-connection streams
pub const HCT: usize = HCN * H; // 10240 residual stream
pub const LOWRANK: usize = 320;
pub const E: usize = 512; // experts per layer
pub const TOPK: usize = 10;
pub const INTER: usize = 640; // per-expert intermediate
pub const GDN_KEY: usize = 2048;
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
pub const PLE_NHEADS: usize = (PLE_NGRAM - 1) * PLE_HEADS_PER_NGRAM; // 16
pub const PLE_EMB_DIM: usize = 160;
pub const PLE_EMBED: usize = PLE_NHEADS * PLE_EMB_DIM; // 2560
pub const PLE_ROWS_PER_SHARD: i64 = 2_500_012;
pub const PLE_EOS: i64 = 248044;
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
}

/// runtime configuration — the knobs the loader honours (spec 0.2, 2.1)
#[derive(Clone, Copy)]
pub struct Config {
    pub context: usize,       // default 262_144, floor 200_000
    pub n_hot: usize,         // target 160 experts per layer, loader clamps
    pub kv: KvDtype,          // FP8 default, BF16 fallback via config
    pub ple_cache_bytes: u64, // hot-row cache, default 128 MB (#16, 2026-09-05; was 1 GB)
    pub prompt_chunk: usize,  // prefill chunk size (correctness stage: 256..512)
    /// pinned-host budget for the cold tier (spec 3.4: ~43-47 GB of 64 GB);
    /// the loader RAISES N if the cold tier would exceed this
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
            prompt_chunk: 512,
            host_pinned_budget: 46 << 30, // measured host ceiling ~48.5 GB, 2.5 GB margin
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

/// Prefill chunk policy (#16). `CROW_CHUNK=<n>` is authoritative; without it the
/// chunk follows the prompt length (rounded up to 512, at most 2048), so a short
/// prompt keeps the chunk-512 scratch and its larger hot set (N 157 vs 140 at
/// chunk 2048) and a long prompt takes the fewest PCIe passes over the cold tier.
/// `CROW_CHUNK_AUTO=1` applies the policy on top of an explicit `CROW_CHUNK`
/// (cap = max(CROW_CHUNK, 2048)); `CROW_CHUNK_AUTO=0` disables it.
/// Default since 2026-09-05 (auto on; opt-in before). Gated: chunk 1024 and 2048
/// deterministic since #22, long-prompt references = ten-task series final4.
pub fn apply_chunk_policy(cfg: &mut Config, n_prompt: usize) {
    let explicit = std::env::var("CROW_CHUNK").ok().and_then(|v| v.parse::<usize>().ok());
    if let Some(c) = explicit {
        cfg.prompt_chunk = c.max(1);
    }
    let auto = match std::env::var("CROW_CHUNK_AUTO").as_deref() {
        Ok("0") => false,
        Ok("1") => true,
        _ => explicit.is_none(),
    };
    if auto {
        let need = ((n_prompt + 511) / 512 * 512).max(512);
        cfg.prompt_chunk = need.min(cfg.prompt_chunk.max(2048));
    }
    apply_adapt_policy(cfg);
}

/// Adaptation policy (#17, 2026-09-05). Measured on the ten-task series
/// (final4 vs trk16): the side-stream trickle with 7 spares / every 16 / max 7
/// gains 0.1 to 0.8 tok/s on every prompt that lands on chunk 2048 (N 147, the
/// spares come out of a hot set that no longer covers the routing) and loses
/// 0.2 to 1.0 tok/s on every prompt at chunk 512 (N 157 already covers it, the
/// seven surrendered slots are pure cost). So the trickle is a long-context
/// switch: with `CROW_ADAPT_STREAM` unset, chunk >= 2048 gets stream / 7 / 16 / 7
/// regardless of `CROW_ADAPT_EVERY` / `CROW_ADAPT_MAX` (which keep describing
/// the short-prompt form), a smaller chunk gets the compute-stream swaps from
/// `CROW_ADAPT_EVERY` (default 0 = none) / `CROW_ADAPT_MAX` (default 8) with
/// `CROW_ADAPT_SPARE` (default 0). `CROW_ADAPT_STREAM=1` / `=0` is the manual
/// mode: every knob from its own variable, spare default 1 / 0, no policy.
pub fn apply_adapt_policy(cfg: &mut Config) {
    let num = |k: &str| std::env::var(k).ok().and_then(|v| v.parse::<usize>().ok());
    let (every, max, spare) = (num("CROW_ADAPT_EVERY"), num("CROW_ADAPT_MAX"), num("CROW_ADAPT_SPARE"));
    let stream = std::env::var("CROW_ADAPT_STREAM").ok();
    cfg.adapt = match stream.as_deref() {
        Some("1") => Adapt { stream: true, spare: spare.unwrap_or(1), every: every.unwrap_or(0), max: max.unwrap_or(8) },
        Some(_) => Adapt { stream: false, spare: spare.unwrap_or(0), every: every.unwrap_or(0), max: max.unwrap_or(8) },
        None if cfg.prompt_chunk >= 2048 => Adapt { stream: true, spare: 7, every: 16, max: 7 },
        None => Adapt { stream: false, spare: spare.unwrap_or(0), every: every.unwrap_or(0), max: max.unwrap_or(8) },
    };
    let a = cfg.adapt;
    eprintln!(
        "[policy] chunk {} -> {} ({}), {} spare hot slot(s), every {}, max {}/layer",
        cfg.prompt_chunk,
        if a.stream { "stream trickle" } else { "compute-stream swaps" },
        if stream.is_some() { "manual CROW_ADAPT_STREAM" } else { "policy" },
        a.spare, a.every, a.max
    );
}
