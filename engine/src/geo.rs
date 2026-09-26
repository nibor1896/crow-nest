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
            _ => Err(format!("CROW_KV={s:?} is not a KV dtype; accepted: bf16, fp8, fp8_e4m3 (unset = the family's default, Family::default_kv)")),
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

/// the Flash-Next context floor (`Geo::FLASH_NEXT.context_floor`): the boot allocates
/// at least this many positions and refuses below it (spec 0.2)
pub const CONTEXT_FLOOR: usize = 200_000;

/// Crow #300 C5: the dense family's context floor (`Geo::context_floor` of a
/// `Qwen35Dense` checkpoint). The phase 2 plan picks the context from the card's
/// free VRAM: 16 GB -> 100k with a 4-bit KV cache, 20-24 GB -> 200k with FP8 KV,
/// 32 GB -> 200k FP8 plus vision and MTP. The floor is the smallest of those points.
/// TODO(Crow #300 phase 2): the per-card planner that raises the context above this
/// floor; until it exists a dense boot is refused before it would use the value.
pub const DENSE_CONTEXT_FLOOR: usize = 100_000;

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

// ---- Crow #300 phase 1 (C2): the runtime geometry ----
//
// `Geo` is the geometry a checkpoint's config DERIVES (`meta::ModelMeta::geo`),
// as one runtime value instead of the compile-time consts above. C2 only builds
// it and asserts it at boot: on the Flash-Next checkpoint the derived `Geo` must
// equal `Geo::FLASH_NEXT`, which is written FROM the consts above, so a config
// that derives anything else refuses the boot with a table of the differing
// fields. Every call site still reads the consts; moving them onto a `Geo` is
// C3. Nothing here is read by a kernel, a buffer size or a loader yet.

/// the model families this engine can parse (`text_config.model_type`)
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Family {
    /// `qwen4_exp_text`: Qwen3.8-Flash-Next, the checkpoint of record — hyper-
    /// connection residual, 512-expert MoE, QSA attention, one PLE layer
    FlashNext,
    /// `qwen3_5_text`: the dense Qwen3.5 / Qwen3.8 family (Qwen3.8-27B) — plain
    /// pre-norm residual, dense SwiGLU, full causal attention; runs since Crow #300
    /// phase 2
    Qwen35Dense,
}

impl Family {
    /// the `text_config.model_type` string of the family
    pub fn model_type(self) -> &'static str {
        match self {
            Family::FlashNext => "qwen4_exp_text",
            Family::Qwen35Dense => "qwen3_5_text",
        }
    }
    /// Crow #300 phase 2: the KV cache dtype a boot takes when `CROW_KV` is unset. Flash-Next
    /// keeps FP8 e4m3 (its gate values of record are computed with it). The dense family takes
    /// BF16: on the 27B the raw FP8 cast failed the pre-registered long-context criterion at
    /// 5 of 6 anchors (KL(BF16 || FP8) 0.04 to 4.4 against the limit 0.073,
    /// `decode_out/p2-kld/PREREG.md`), while BF16 KV tracks the f32 reference (median KL 4e-5)
    pub fn default_kv(self) -> KvDtype {
        match self {
            Family::FlashNext => KvDtype::Fp8E4m3,
            Family::Qwen35Dense => KvDtype::Bf16,
        }
    }
    /// every family, in table order
    pub const ALL: [Family; 2] = [Family::FlashNext, Family::Qwen35Dense];

    /// Crow #300 C3: the family's number in a slot file header (`slot::Header::model_family`);
    /// 0 is never written, so a zeroed field reads as no family
    pub const fn code(self) -> u64 {
        match self {
            Family::FlashNext => 1,
            Family::Qwen35Dense => 2,
        }
    }
    /// the family a slot-file code names, `None` for a code no family carries
    pub fn from_code(code: u64) -> Option<Family> {
        Family::ALL.into_iter().find(|f| f.code() == code)
    }
}

/// the residual stream: hyper-connection streams (mixed in and out of every
/// sub-block) or one plain pre-norm residual
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Residual {
    Hc { streams: usize, lowrank: usize },
    Plain,
}

/// the feed-forward block of every layer
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Ffn {
    /// routed experts plus one shared expert (and its sigmoid gate)
    Moe { experts: usize, topk: usize, expert_inter: usize, shared_inter: usize },
    /// one dense SwiGLU of width `inter`
    Dense { inter: usize },
}

/// what the full-attention layers attend over
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Attn {
    /// the QSA indexer picks `block_topk` compressed blocks (`sel_max` rows)
    Qsa { heads: usize, kv_heads: usize, head_dim: usize, compress: usize, block_topk: usize },
    /// uncapped causal attention over every row
    Full,
}

impl Attn {
    /// the most rows one query attends (`QSA_SEL_MAX`); `None` = uncapped
    pub const fn sel_max(self) -> Option<usize> {
        match self {
            Attn::Qsa { compress, block_topk, .. } => Some(block_topk * compress + compress - 1),
            Attn::Full => None,
        }
    }
}

/// the per-layer n-gram embedding (PLE) of one layer
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct PleGeo {
    /// 0-based layer index (config `ple_layer_ids` is 1-based)
    pub layer: usize,
    pub ngram: usize,
    pub heads_per_ngram: usize,
    /// the concatenated embedding width (`ple_embed_dim`)
    pub embed: usize,
    pub conv_kernel: usize,
    /// the shard end marker, the text config's own eos id
    pub eos: i64,
}

impl PleGeo {
    pub const fn nheads(self) -> usize {
        (self.ngram - 1) * self.heads_per_ngram
    }
    pub const fn emb_dim(self) -> usize {
        self.embed / self.nheads()
    }
    /// the n-gram context: the ids before the current one (`PLE_CTX`)
    pub const fn ctx(self) -> usize {
        self.ngram - 1
    }
}

/// Crow #300 C3: the numbers `gen.rs` computes with, as one flat `Copy` value derived
/// from the model's `Geo` (`Geo::dims`) - one field per `geo.rs` const it replaces,
/// the const's name in lower case (`H` -> `h`, `QSA_SEL_MAX` -> `qsa_sel_max`). It is
/// a CACHE of the `Geo`, never a second source: `Engine::load` builds it from
/// `Engine::geo` once, so the per-launch host code reads a field instead of matching
/// the family enums. C5: the family switches match on the `Geo` enums; `Dims` carries
/// the numbers of the blocks that exist (a model without PLE gets zero PLE fields).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct Dims {
    pub h: usize,
    pub hcn: usize,
    pub hct: usize,
    pub lowrank: usize,
    pub e: usize,
    pub topk: usize,
    pub inter: usize,
    pub gdn_key: usize,
    pub gdn_val: usize,
    pub gdn_conv: usize,
    pub gdn_kheads: usize,
    pub gdn_vheads: usize,
    /// GDN key head dim (`GD`)
    pub gd: usize,
    /// GDN value head dim (`GDN_VAL / GDN_VHEADS`, 128 = `GD` on Flash-Next)
    pub gdv: usize,
    pub conv_kernel: usize,
    pub nq: usize,
    pub nkv: usize,
    pub ahd: usize,
    pub q_rows: usize,
    pub kv_rows: usize,
    pub core: usize,
    pub v: usize,
    pub layers: usize,
    pub gdn_layers: usize,
    pub attn_layers: usize,
    pub attn_interval: usize,
    pub rope_pairs: usize,
    pub ple_layer: usize,
    pub ple_ngram: usize,
    pub ple_ctx: usize,
    pub ple_heads_per_ngram: usize,
    pub ple_nheads: usize,
    pub ple_emb_dim: usize,
    pub ple_embed: usize,
    pub ple_eos: i64,
    pub qsa_heads: usize,
    pub qsa_kvheads: usize,
    pub qsa_hd: usize,
    pub qsa_qk_rows: usize,
    pub qsa_compress: usize,
    pub qsa_block_topk: usize,
    pub qsa_sel_max: usize,
    pub qsa_hidd: usize,
    /// Crow #300 phase 2: the dense SwiGLU width (`Ffn::Dense`), 0 on a MoE family
    pub dense_inter: usize,
}

impl Dims {
    /// layer % interval == interval - 1 is full attention (`geo::is_attn`)
    pub const fn is_attn(&self, layer: usize) -> bool {
        layer % self.attn_interval == self.attn_interval - 1
    }
    /// `geo::attn_index`
    pub const fn attn_index(&self, layer: usize) -> usize {
        layer / self.attn_interval
    }
    /// `geo::gdn_index`
    pub const fn gdn_index(&self, layer: usize) -> usize {
        layer - layer / self.attn_interval
    }
}

/// C3: the routed-expert numbers of a MoE family, as `Geo::moe` hands them out
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct MoeGeo {
    pub experts: usize,
    pub topk: usize,
    pub expert_inter: usize,
    pub shared_inter: usize,
}

/// C3: the QSA indexer numbers of a QSA family, as `Geo::qsa` hands them out
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub struct QsaGeo {
    pub heads: usize,
    pub kv_heads: usize,
    /// the indexer head dim, also the raw indexer key width (`QSA_HD` = `QSA_HIDD`)
    pub head_dim: usize,
    pub compress: usize,
    pub block_topk: usize,
}

impl QsaGeo {
    /// rows of `index_qk_proj`: query heads plus the key head (`QSA_QK_ROWS`)
    pub const fn qk_rows(self) -> usize {
        (self.heads + self.kv_heads) * self.head_dim
    }
    /// the most rows one query attends (`QSA_SEL_MAX`)
    pub const fn sel_max(self) -> usize {
        self.block_topk * self.compress + self.compress - 1
    }
    /// the raw indexer key width (`QSA_HIDD`): the key head of `index_qk_proj`
    pub const fn hidd(self) -> usize {
        self.kv_heads * self.head_dim
    }
}

/// the GDN gated RMSNorm's activation (`output_gate_type`; HF qwen4_exp builds
/// `RMSNormGated(activation=config.output_gate_type)`, qwen3_5 hard-codes silu).
/// The attention output gate is `sigmoid(gate)` in both families and is not
/// this field (Crow #300 C4 found the old wording wrong).
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum GateAct {
    Sigmoid,
    /// `swish` in the config: x·sigmoid(x)
    Silu,
}

/// what turns the last layer's residual into the lm_head input
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum FinalNorm {
    /// the model-level hyper-connection mixer (`head_run`)
    HcMixer,
    /// one RMSNorm
    Rms,
}

/// the runtime geometry of one checkpoint (C2). Every field is a number or a
/// formula switch the engine computes with; `meta::ModelMeta::geo` derives it
/// from config.json + generation_config.json + the family table.
#[derive(Clone, Copy, PartialEq, Debug)]
pub struct Geo {
    pub family: Family,
    pub hidden: usize,
    pub residual: Residual,
    pub layers: usize,
    pub gdn_layers: usize,
    pub attn_layers: usize,
    /// layer % interval == interval - 1 is full attention
    pub attn_interval: usize,
    pub q_heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub attn: Attn,
    pub attn_output_gate: bool,
    pub gate_act: GateAct,
    pub attention_bias: bool,
    pub rope_pairs: usize,
    pub rope_theta: f64,
    pub mrope_section: [usize; 3],
    pub mrope_interleaved: bool,
    pub gdn_key_heads: usize,
    pub gdn_value_heads: usize,
    pub gdn_key_dim: usize,
    pub gdn_value_dim: usize,
    pub conv_kernel: usize,
    pub ffn: Ffn,
    pub ple: Option<PleGeo>,
    pub final_norm: FinalNorm,
    /// RMSNorm weight applied as (1 + w) (the zero-centred gamma of the family)
    pub norm_one_plus_w: bool,
    pub rms_eps: f64,
    pub vocab: usize,
    pub tie_word_embeddings: bool,
    /// the checkpoint's `max_position_embeddings` (`Config::default().context`)
    pub context_max: usize,
    /// the context the boot allocates at least (`CONTEXT_FLOOR`)
    pub context_floor: usize,
    /// the generation stop ids (`sample::EOS_IDS`)
    pub eos_ids: [usize; 2],
    pub mtp_layers: usize,
    /// the vision merger's output width; `None` = no vision tower in the config
    pub vision_out_hidden: Option<usize>,
}

impl Geo {
    /// today's compile-time geometry, written from the consts above (and the
    /// literals the kernels and the boot table pin: eps 1e-6, theta 1e7, the
    /// sigmoid gate, the (1 + w) norm, conv kernel 4). A Flash-Next boot asserts
    /// the config-derived `Geo` equal to this.
    pub const FLASH_NEXT: Geo = Geo {
        family: Family::FlashNext,
        hidden: H,
        residual: Residual::Hc { streams: HCN, lowrank: LOWRANK },
        layers: LAYERS,
        gdn_layers: GDN_LAYERS,
        attn_layers: ATTN_LAYERS,
        attn_interval: 4,
        q_heads: NQ,
        kv_heads: NKV,
        head_dim: AHD,
        attn: Attn::Qsa {
            heads: QSA_HEADS,
            kv_heads: QSA_KVHEADS,
            head_dim: QSA_HD,
            compress: QSA_COMPRESS,
            block_topk: QSA_BLOCK_TOPK,
        },
        attn_output_gate: true,
        gate_act: GateAct::Sigmoid,
        attention_bias: false,
        rope_pairs: ROPE_PAIRS,
        rope_theta: 1e7,
        mrope_section: [11, 11, 10],
        mrope_interleaved: true,
        gdn_key_heads: GDN_KHEADS,
        gdn_value_heads: GDN_VHEADS,
        gdn_key_dim: GD,
        gdn_value_dim: GDN_VAL / GDN_VHEADS,
        conv_kernel: 4,
        // the shared expert runs through the same 640/1280 silu·mul chain
        ffn: Ffn::Moe { experts: E, topk: TOPK, expert_inter: INTER, shared_inter: INTER },
        ple: Some(PleGeo {
            layer: PLE_LAYER,
            ngram: PLE_NGRAM,
            heads_per_ngram: PLE_HEADS_PER_NGRAM,
            embed: PLE_EMBED,
            conv_kernel: 4,
            eos: PLE_EOS,
        }),
        final_norm: FinalNorm::HcMixer,
        norm_one_plus_w: true,
        rms_eps: 1e-6,
        vocab: V,
        tie_word_embeddings: false,
        context_max: 262_144,
        context_floor: CONTEXT_FLOOR,
        // = sample::EOS_IDS (sample depends on geo, so the equality is a test)
        eos_ids: [248046, PLE_EOS as usize],
        mtp_layers: 1,
        vision_out_hidden: Some(H),
    };

    // the derivation chain of the consts above, per Geo

    /// hyper-connection streams (1 for a plain residual)
    pub const fn hc_streams(&self) -> usize {
        match self.residual {
            Residual::Hc { streams, .. } => streams,
            Residual::Plain => 1,
        }
    }
    /// the residual stream width (`HCT` on Flash-Next)
    pub const fn residual_width(&self) -> usize {
        self.hc_streams() * self.hidden
    }
    /// query + output gate rows (`Q_ROWS`)
    pub const fn q_rows(&self) -> usize {
        self.q_heads * self.head_dim * if self.attn_output_gate { 2 } else { 1 }
    }
    pub const fn kv_rows(&self) -> usize {
        self.kv_heads * self.head_dim
    }
    /// the attention core width (`CORE`)
    pub const fn core(&self) -> usize {
        self.q_heads * self.head_dim
    }
    /// query heads per kv head (the kernels' `head / 12`)
    pub const fn gqa(&self) -> usize {
        self.q_heads / self.kv_heads
    }
    pub const fn gdn_key(&self) -> usize {
        self.gdn_key_heads * self.gdn_key_dim
    }
    pub const fn gdn_val(&self) -> usize {
        self.gdn_value_heads * self.gdn_value_dim
    }
    /// the GDN conv channels: q, k and v (`GDN_CONV`)
    pub const fn gdn_conv(&self) -> usize {
        self.gdn_key() * 2 + self.gdn_val()
    }
    pub const fn is_attn(&self, layer: usize) -> bool {
        layer % self.attn_interval == self.attn_interval - 1
    }
    /// the attention-layer index of full-attention layer `layer` (`geo::attn_index`)
    pub const fn attn_index(&self, layer: usize) -> usize {
        layer / self.attn_interval
    }
    /// the GDN-layer index of linear-attention layer `layer` (`geo::gdn_index`):
    /// the layers below it minus the attention layers among them
    pub const fn gdn_index(&self, layer: usize) -> usize {
        layer - layer / self.attn_interval
    }

    // ---- C3: the family-specific numbers the host sites read ----
    //
    // These accessors hand a block's numbers to the code inside that block's
    // family arm and refuse by name on a family that has no such block. C5: a
    // family whose arms are not built dies at `Geo::built` in the boot door
    // (`boot::model_geo`) before any of them can run, so the refusals are a
    // guard, not a path.

    /// the hyper-connection low-rank width (`LOWRANK`)
    pub const fn hc_lowrank(&self) -> usize {
        match self.residual {
            Residual::Hc { lowrank, .. } => lowrank,
            Residual::Plain => panic!("Geo::hc_lowrank: a plain residual has no hyper-connection mixer (Crow #300 phase 2 builds that path)"),
        }
    }
    /// the routed-expert block (`E`, `TOPK`, `INTER`)
    pub const fn moe(&self) -> MoeGeo {
        match self.ffn {
            Ffn::Moe { experts, topk, expert_inter, shared_inter } => MoeGeo { experts, topk, expert_inter, shared_inter },
            Ffn::Dense { .. } => panic!("Geo::moe: a dense FFN has no experts (Crow #300 phase 2 builds that path)"),
        }
    }
    /// the QSA indexer (`QSA_*`)
    pub const fn qsa(&self) -> QsaGeo {
        match self.attn {
            Attn::Qsa { heads, kv_heads, head_dim, compress, block_topk } => QsaGeo { heads, kv_heads, head_dim, compress, block_topk },
            Attn::Full => panic!("Geo::qsa: full causal attention has no QSA indexer (Crow #300 phase 2 builds that path)"),
        }
    }
    /// the PLE layer (`PLE_*`)
    pub const fn ple_geo(&self) -> PleGeo {
        match self.ple {
            Some(p) => p,
            None => panic!("Geo::ple_geo: this model has no PLE layer (C5: its PLE arm is `None`, skipped by the loader and the layer loop)"),
        }
    }
    /// the vision merger's output width (`H` in vit.rs)
    pub const fn vision_out(&self) -> usize {
        match self.vision_out_hidden {
            Some(w) => w,
            None => panic!("Geo::vision_out: this config has no vision tower"),
        }
    }

    /// every field as (name, rendered value), in declaration order — the boot
    /// print and the mismatch table are both this list
    pub fn rows(&self) -> Vec<(&'static str, String)> {
        vec![
            ("family", format!("{:?} ({})", self.family, self.family.model_type())),
            ("hidden", self.hidden.to_string()),
            ("residual", format!("{:?}", self.residual)),
            ("layers", self.layers.to_string()),
            ("gdn_layers", self.gdn_layers.to_string()),
            ("attn_layers", self.attn_layers.to_string()),
            ("attn_interval", self.attn_interval.to_string()),
            ("q_heads", self.q_heads.to_string()),
            ("kv_heads", self.kv_heads.to_string()),
            ("head_dim", self.head_dim.to_string()),
            ("attn", format!("{:?}", self.attn)),
            ("attn_output_gate", self.attn_output_gate.to_string()),
            ("gate_act", format!("{:?}", self.gate_act)),
            ("attention_bias", self.attention_bias.to_string()),
            ("rope_pairs", self.rope_pairs.to_string()),
            ("rope_theta", format!("{:?}", self.rope_theta)),
            ("mrope_section", format!("{:?}", self.mrope_section)),
            ("mrope_interleaved", self.mrope_interleaved.to_string()),
            ("gdn_key_heads", self.gdn_key_heads.to_string()),
            ("gdn_value_heads", self.gdn_value_heads.to_string()),
            ("gdn_key_dim", self.gdn_key_dim.to_string()),
            ("gdn_value_dim", self.gdn_value_dim.to_string()),
            ("conv_kernel", self.conv_kernel.to_string()),
            ("ffn", format!("{:?}", self.ffn)),
            ("ple", format!("{:?}", self.ple)),
            ("final_norm", format!("{:?}", self.final_norm)),
            ("norm_one_plus_w", self.norm_one_plus_w.to_string()),
            ("rms_eps", format!("{:?}", self.rms_eps)),
            ("vocab", self.vocab.to_string()),
            ("tie_word_embeddings", self.tie_word_embeddings.to_string()),
            ("context_max", self.context_max.to_string()),
            ("context_floor", self.context_floor.to_string()),
            ("eos_ids", format!("{:?}", self.eos_ids)),
            ("mtp_layers", self.mtp_layers.to_string()),
            ("vision_out_hidden", format!("{:?}", self.vision_out_hidden)),
        ]
    }

    /// Crow #300 C3: the flat view `gen.rs` computes with (`Dims`), one field per
    /// const it replaces. Built once per `Engine::load`; panics by name (through
    /// `moe` / `qsa` / `hc_lowrank`) on a family whose arms the engine does not
    /// build yet (`Geo::built` refuses those at boot, before this runs). C5: a model
    /// without PLE gets zero PLE fields; every reader sits behind the PLE arm.
    pub const fn dims(&self) -> Dims {
        // Crow #300 phase 2: a block the family does not have reads as zeros in the
        // flat view (no experts, no indexer, no mixer); no host site of that family
        // computes with them, and `KernelGeo::defines` gives the kernels that are
        // compiled but never launched a compile-only value
        let m = match self.ffn {
            Ffn::Moe { .. } => self.moe(),
            Ffn::Dense { .. } => MoeGeo { experts: 0, topk: 0, expert_inter: 0, shared_inter: 0 },
        };
        let q = match self.attn {
            Attn::Qsa { .. } => self.qsa(),
            Attn::Full => QsaGeo { heads: 0, kv_heads: 0, head_dim: 0, compress: 0, block_topk: 0 },
        };
        let lowrank = match self.residual {
            Residual::Hc { lowrank, .. } => lowrank,
            Residual::Plain => 0,
        };
        let sel_max = match self.attn {
            Attn::Qsa { .. } => q.sel_max(),
            Attn::Full => 0,
        };
        let dense_inter = match self.ffn {
            Ffn::Moe { .. } => 0,
            Ffn::Dense { inter } => inter,
        };
        // (layer, ngram, ctx, heads per ngram, nheads, emb dim, embed, eos)
        let p = match self.ple {
            Some(p) => (p.layer, p.ngram, p.ctx(), p.heads_per_ngram, p.nheads(), p.emb_dim(), p.embed, p.eos),
            None => (0, 0, 0, 0, 0, 0, 0, 0),
        };
        Dims {
            h: self.hidden,
            hcn: self.hc_streams(),
            hct: self.residual_width(),
            lowrank,
            e: m.experts,
            topk: m.topk,
            inter: m.expert_inter,
            dense_inter,
            gdn_key: self.gdn_key(),
            gdn_val: self.gdn_val(),
            gdn_conv: self.gdn_conv(),
            gdn_kheads: self.gdn_key_heads,
            gdn_vheads: self.gdn_value_heads,
            gd: self.gdn_key_dim,
            gdv: self.gdn_value_dim,
            conv_kernel: self.conv_kernel,
            nq: self.q_heads,
            nkv: self.kv_heads,
            ahd: self.head_dim,
            q_rows: self.q_rows(),
            kv_rows: self.kv_rows(),
            core: self.core(),
            v: self.vocab,
            layers: self.layers,
            gdn_layers: self.gdn_layers,
            attn_layers: self.attn_layers,
            attn_interval: self.attn_interval,
            rope_pairs: self.rope_pairs,
            ple_layer: p.0,
            ple_ngram: p.1,
            ple_ctx: p.2,
            ple_heads_per_ngram: p.3,
            ple_nheads: p.4,
            ple_emb_dim: p.5,
            ple_embed: p.6,
            ple_eos: p.7,
            qsa_heads: q.heads,
            qsa_kvheads: q.kv_heads,
            qsa_hd: q.head_dim,
            qsa_qk_rows: q.qk_rows(),
            qsa_compress: q.compress,
            qsa_block_topk: q.block_topk,
            qsa_sel_max: sel_max,
            qsa_hidd: q.hidd(),
        }
    }

    /// Crow #300 C3: a 64-bit fingerprint of this geometry - fnv1a-64 over the
    /// `rows` rendering (`name=value` lines, declaration order). The slot file
    /// carries it next to the family (`slot::Header::geo_hash`), so a slot saved
    /// by one model is refused by another even where every buffer SIZE agrees.
    /// It is only as stable as the `rows` text: a renamed field or a changed
    /// Debug rendering moves it, which refuses old slot files - the safe side.
    pub fn fingerprint(&self) -> u64 {
        let mut h: u64 = 0xcbf2_9ce4_8422_2325;
        for (name, value) in self.rows() {
            for b in name.bytes().chain(std::iter::once(b'=')).chain(value.bytes()).chain(std::iter::once(b'\n')) {
                h ^= b as u64;
                h = h.wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        h
    }

    /// the fields where `self` differs from `want`: (name, self, want)
    pub fn diff(&self, want: &Geo) -> Vec<(&'static str, String, String)> {
        let mut d: Vec<_> = self
            .rows()
            .into_iter()
            .zip(want.rows())
            .filter(|((_, a), (_, b))| a != b)
            .map(|((name, a), (_, b))| (name, a, b))
            .collect();
        // a field `rows` forgot must still refuse: the derived PartialEq decides
        if d.is_empty() && self != want {
            d.push(("(a field missing from Geo::rows)", format!("{self:?}"), format!("{want:?}")));
        }
        d
    }

    /// Crow #300 C5: the boot check of the family switches. It walks the blocks in
    /// forward order (the residual stream the embedding writes, then per layer the
    /// PLE add, the attention and the FFN, then the final norm) and refuses at the
    /// FIRST arm this engine has not built, by name ([`not_built`]). As phase 2
    /// builds arms, a checkpoint gets exactly as far as the engine supports.
    /// `boot::model_geo` calls it before the container is mapped and before the
    /// CUDA context exists; `Engine::load` calls it again before its first byte.
    ///
    /// | block | Flash-Next | Qwen35Dense (Crow #300 phase 2) |
    /// |---|---|---|
    /// | residual | `Residual::Hc` | `Residual::Plain` |
    /// | PLE | `Some` (loaded, added at its layer) | `None` (skipped) |
    /// | attention | `Attn::Qsa` | `Attn::Full` |
    /// | FFN | `Ffn::Moe` (hot sets, residency, expert slabs, cold staging) | `Ffn::Dense` |
    /// | final norm | `FinalNorm::HcMixer` | `FinalNorm::Rms` |
    ///
    /// Since phase 2 every arm of every block is built, so this returns `Ok` for every
    /// `Geo`; a new enum variant must get its arm here (the matches are exhaustive), and
    /// until its engine path exists it refuses with [`not_built`].
    pub fn built(&self) -> Result<(), String> {
        match self.residual {
            Residual::Hc { .. } | Residual::Plain => {}
        }
        match self.ple {
            Some(_) | None => {}
        }
        match self.attn {
            Attn::Qsa { .. } | Attn::Full => {}
        }
        match self.ffn {
            Ffn::Moe { .. } | Ffn::Dense { .. } => {}
        }
        match self.final_norm {
            FinalNorm::HcMixer | FinalNorm::Rms => {}
        }
        Ok(())
    }
}

/// Crow #300 C5: the refusal of an arm the engine has not built:
/// "<block> for family <F> not built yet (Crow #300 phase 2)"
pub fn not_built(block: &str, family: Family) -> String {
    format!("{block} for family {family:?} not built yet (Crow #300 phase 2)")
}

/// Crow #300 C5: an unbuilt arm reached past the boot check. `Geo::built` refuses
/// every such `Geo` at boot, so this is a guard for a `Geo` that did not come
/// through `boot::model_geo`, never a path a request can take.
#[track_caller]
pub fn unbuilt_arm(block: &str, family: Family) -> ! {
    panic!("{} - reached past the boot check (Geo::built refuses it at boot)", not_built(block, family))
}

#[cfg(test)]
mod tests_300 {
    use super::*;

    /// `Geo::FLASH_NEXT` and its derivation chain reproduce every compile-time
    /// const the call sites read today (the C3 migration moves a site from the
    /// const to the Geo; this is the table it must agree with)
    #[test]
    fn flash_next_geo_derives_every_pinned_const() {
        let g = Geo::FLASH_NEXT;
        assert_eq!(g.residual_width(), HCT);
        assert_eq!(g.q_rows(), Q_ROWS);
        assert_eq!(g.kv_rows(), KV_ROWS);
        assert_eq!(g.core(), CORE);
        assert_eq!(g.gqa(), 12, "the kernels' `head / 12`");
        assert_eq!(g.gdn_key(), GDN_KEY);
        assert_eq!(g.gdn_val(), GDN_VAL);
        assert_eq!(g.gdn_conv(), GDN_CONV);
        assert_eq!(g.attn.sel_max(), Some(QSA_SEL_MAX));
        let ple = g.ple.unwrap();
        assert_eq!(ple.nheads(), PLE_NHEADS);
        assert_eq!(ple.emb_dim(), PLE_EMB_DIM);
        assert_eq!(g.context_max, Config::default().context);
        assert_eq!(g.eos_ids, crate::sample::EOS_IDS);
        for l in 0..LAYERS {
            assert_eq!(g.is_attn(l), is_attn(l), "layer {l}");
        }
        assert_eq!((0..LAYERS).filter(|l| g.is_attn(*l)).count(), ATTN_LAYERS);
        assert_eq!(g.rows().len(), 35, "one row per Geo field");
        assert!(g.diff(&Geo::FLASH_NEXT).is_empty());
    }

    /// C3: the accessors the migrated host sites read hand out exactly the consts
    /// they replaced, and refuse by name on a family without that block
    #[test]
    fn c3_accessors_reproduce_the_consts_they_replace() {
        let g = Geo::FLASH_NEXT;
        assert_eq!(g.hc_lowrank(), LOWRANK);
        let m = g.moe();
        assert_eq!((m.experts, m.topk, m.expert_inter, m.shared_inter), (E, TOPK, INTER, INTER));
        let q = g.qsa();
        assert_eq!((q.heads, q.kv_heads, q.head_dim, q.compress, q.block_topk), (QSA_HEADS, QSA_KVHEADS, QSA_HD, QSA_COMPRESS, QSA_BLOCK_TOPK));
        assert_eq!((q.qk_rows(), q.sel_max(), q.hidd()), (QSA_QK_ROWS, QSA_SEL_MAX, QSA_HIDD));
        let p = g.ple_geo();
        assert_eq!((p.layer, p.ngram, p.ctx(), p.heads_per_ngram, p.embed), (PLE_LAYER, PLE_NGRAM, PLE_CTX, PLE_HEADS_PER_NGRAM, PLE_EMBED));
        assert_eq!(g.vision_out(), H);
        for l in 0..LAYERS {
            if is_attn(l) {
                assert_eq!(g.attn_index(l), attn_index(l), "layer {l}");
            } else {
                assert_eq!(g.gdn_index(l), gdn_index(l), "layer {l}");
            }
        }
        // `Dims`, the flat view gen.rs computes with: every field is the const it replaces
        let d = g.dims();
        assert_eq!((d.h, d.hcn, d.hct, d.lowrank, d.e, d.topk, d.inter), (H, HCN, HCT, LOWRANK, E, TOPK, INTER));
        assert_eq!(
            (d.gdn_key, d.gdn_val, d.gdn_conv, d.gdn_kheads, d.gdn_vheads, d.gd, d.gdv),
            (GDN_KEY, GDN_VAL, GDN_CONV, GDN_KHEADS, GDN_VHEADS, GD, GD)
        );
        assert_eq!(
            (d.nq, d.nkv, d.ahd, d.q_rows, d.kv_rows, d.core, d.v, d.layers, d.gdn_layers, d.attn_layers, d.rope_pairs, d.conv_kernel),
            (NQ, NKV, AHD, Q_ROWS, KV_ROWS, CORE, V, LAYERS, GDN_LAYERS, ATTN_LAYERS, ROPE_PAIRS, 4)
        );
        assert_eq!(
            (d.ple_layer, d.ple_ngram, d.ple_ctx, d.ple_heads_per_ngram, d.ple_nheads, d.ple_emb_dim, d.ple_embed, d.ple_eos),
            (PLE_LAYER, PLE_NGRAM, PLE_CTX, PLE_HEADS_PER_NGRAM, PLE_NHEADS, PLE_EMB_DIM, PLE_EMBED, PLE_EOS)
        );
        assert_eq!(
            (d.qsa_heads, d.qsa_kvheads, d.qsa_hd, d.qsa_qk_rows, d.qsa_compress, d.qsa_block_topk, d.qsa_sel_max, d.qsa_hidd),
            (QSA_HEADS, QSA_KVHEADS, QSA_HD, QSA_QK_ROWS, QSA_COMPRESS, QSA_BLOCK_TOPK, QSA_SEL_MAX, QSA_HIDD)
        );
        for l in 0..LAYERS {
            assert_eq!((d.is_attn(l), d.attn_index(l), d.gdn_index(l)), (is_attn(l), attn_index(l), gdn_index(l)), "layer {l}");
        }
        assert_eq!(crate::manager::gdn_s_state_len(&g), GDN_VHEADS * GD * GD);
        assert_eq!(crate::manager::gdn_conv_state_len(&g), GDN_CONV * 3);
        assert_eq!(crate::manager::ple_state_len(&g), GDN_CONV * 9, "the PLE conv runs over the 10240-wide residual");
        // the fingerprint of record: a change here refuses every slot file saved before it
        assert_eq!(g.fingerprint(), 0x7191_8a73_24c6_5fdd, "Geo::FLASH_NEXT fingerprint");
        assert_eq!(Family::from_code(g.family.code()), Some(Family::FlashNext));
        assert_eq!(Family::from_code(0), None);
        // a family without the block refuses the block's accessor by name; its flat
        // `dims()` reads the missing blocks as zeros instead (Crow #300 phase 2: the
        // dense engine computes with the Dims of a family that has none of them)
        let dense = Geo { residual: Residual::Plain, ffn: Ffn::Dense { inter: 17408 }, attn: Attn::Full, ple: None, ..g };
        let dd = dense.dims();
        assert_eq!((dd.e, dd.topk, dd.inter, dd.lowrank, dd.qsa_sel_max, dd.qsa_hidd, dd.dense_inter, dd.hcn, dd.hct), (0, 0, 0, 0, 0, 0, 17408, 1, g.hidden));
        assert_eq!(g.dims().dense_inter, 0, "Flash-Next has no dense FFN");
        for (what, r) in [
            ("hc_lowrank", std::panic::catch_unwind(|| dense.hc_lowrank())),
            ("moe", std::panic::catch_unwind(|| dense.moe().experts)),
            ("qsa", std::panic::catch_unwind(|| dense.qsa().heads)),
            ("ple_geo", std::panic::catch_unwind(|| dense.ple_geo().layer)),
        ] {
            let e = r.unwrap_err();
            let msg = e.downcast_ref::<&str>().copied().map(String::from).or_else(|| e.downcast_ref::<String>().cloned()).unwrap();
            let why = if what == "ple_geo" { "its PLE arm is `None`" } else { "Crow #300 phase 2" };
            assert!(msg.contains(&format!("Geo::{what}")) && msg.contains(why), "{what}: {msg}");
        }
    }
}

#[cfg(test)]
mod tests_300_c5 {
    #[test]
    fn an_unset_crow_kv_is_fp8_on_flash_next_and_bf16_on_the_dense_family() {
        // Crow #300 phase 2: Flash-Next's gate values of record are FP8; the 27B failed the
        // long-context criterion with FP8 KV, so its default is BF16
        use super::{Family, KvDtype};
        assert_eq!(Family::FlashNext.default_kv(), KvDtype::Fp8E4m3);
        assert_eq!(Family::Qwen35Dense.default_kv(), KvDtype::Bf16);
    }

    use super::*;

    /// C5: Flash-Next takes every built arm, so the boot check passes, and a
    /// Flash-Next without its PLE layer is built too (the `None` arm is a skip):
    /// its `Dims` carry zero PLE fields and every other field unchanged
    #[test]
    fn flash_next_passes_the_family_check_and_ple_is_optional() {
        let g = Geo::FLASH_NEXT;
        assert_eq!(g.built(), Ok(()));
        assert_eq!(g.context_floor, 200_000, "the Flash-Next floor of record");
        let no_ple = Geo { ple: None, ..g };
        assert_eq!(no_ple.built(), Ok(()), "a model without PLE skips the block");
        let (d, dn) = (g.dims(), no_ple.dims());
        assert_eq!((dn.ple_layer, dn.ple_ctx, dn.ple_nheads, dn.ple_emb_dim, dn.ple_embed, dn.ple_eos), (0, 0, 0, 0, 0, 0));
        assert_eq!(Dims { ple_layer: d.ple_layer, ple_ngram: d.ple_ngram, ple_ctx: d.ple_ctx, ple_heads_per_ngram: d.ple_heads_per_ngram,
            ple_nheads: d.ple_nheads, ple_emb_dim: d.ple_emb_dim, ple_embed: d.ple_embed, ple_eos: d.ple_eos, ..dn }, d);
    }

    /// C5: each unbuilt arm refuses by its own name, the first one in forward
    /// order wins, and building an arm moves the refusal to the next block
    #[test]
    fn every_arm_is_built_and_the_refusal_still_names_its_block() {
        let g = Geo::FLASH_NEXT;
        let f = Family::Qwen35Dense;
        let dense = Geo { family: f, residual: Residual::Plain, attn: Attn::Full, ffn: Ffn::Dense { inter: 17408 }, ple: None, final_norm: FinalNorm::Rms, ..g };
        // phase 2: the dense arms and every mix of them with the Flash-Next ones pass
        assert_eq!(dense.built(), Ok(()));
        assert_eq!(Geo { residual: g.residual, ..dense }.built(), Ok(()));
        assert_eq!(Geo { attn: g.attn, ffn: g.ffn, ..dense }.built(), Ok(()));
        // the refusal a future unbuilt arm takes (`not_built`, `unbuilt_arm`) keeps its form
        assert_eq!(not_built("X::Y (a block)", f), "X::Y (a block) for family Qwen35Dense not built yet (Crow #300 phase 2)");
        let e = std::panic::catch_unwind(|| unbuilt_arm("X::Y (a block)", f)).unwrap_err();
        let msg = e.downcast_ref::<String>().cloned().unwrap();
        assert!(msg.starts_with("X::Y (a block) for family Qwen35Dense not built yet (Crow #300 phase 2)"), "{msg}");
    }
}
