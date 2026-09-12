//! #11 — the production generation loop: embeddings → 48 decoder layers
//! (36× GDN, 12× full attention with QSA + FP8/BF16-KV) under hyper-connections,
//! PLE at layer 1, MoE with the #8 scheduler (GPU router + zero-copy cold path),
//! model-level hyper_connection_mixer → untied BF16 lm_head → greedy argmax.
//!
//! Every sub-flow is the probe-verified math (p6–p16). Host touches per decode
//! token: embedding row, PLE index math (spec 3.5 — the ONE host touch), scalar
//! updates are device buffers, and the end-of-token argmax readback. Layers run
//! entirely from queued kernels — no per-layer sync, no DtoH of routing.

use crate::cuda::{self, CUdeviceptr as Dev};
use crate::geo::*;
use crate::kernels::Kernels;
use crate::manager::ThreeStates;
use crate::residency::{Residency, PendingSwap};
use crate::cnq::{self, Cnq};
use std::collections::HashMap;

/// live per-section profile (env CROW_PROFILE=1): CPU-side microseconds per
/// section, accumulated over all steps. `prof::report` prints ms/step — the
/// breakdown that says where the per-token milliseconds actually go.
pub mod prof {
    use std::sync::atomic::{AtomicU64, Ordering};
    pub static SCALAR: AtomicU64 = AtomicU64::new(0); // host→device scalar refreshes (each = HtoD + full sync)
    pub static EMBED: AtomicU64 = AtomicU64::new(0);
    pub static PLE: AtomicU64 = AtomicU64::new(0);
    pub static HC: AtomicU64 = AtomicU64::new(0);
    pub static SUB_GDN: AtomicU64 = AtomicU64::new(0);
    pub static SUB_ATTN: AtomicU64 = AtomicU64::new(0);
    pub static MOE: AtomicU64 = AtomicU64::new(0);
    pub static HEAD: AtomicU64 = AtomicU64::new(0);
    pub static TAIL: AtomicU64 = AtomicU64::new(0); // end-of-token sync + argmax readback
    pub static LAUNCHES: AtomicU64 = AtomicU64::new(0);
    pub static STEPS: AtomicU64 = AtomicU64::new(0);
    pub fn add(c: &AtomicU64, us: u64) {
        c.fetch_add(us, Ordering::Relaxed);
    }
    pub fn launch() {
        LAUNCHES.fetch_add(1, Ordering::Relaxed);
    }
    pub fn reset() {
        for c in [&SCALAR, &EMBED, &PLE, &HC, &SUB_GDN, &SUB_ATTN, &MOE, &HEAD, &TAIL, &LAUNCHES, &STEPS] {
            c.store(0, Ordering::Relaxed);
        }
    }
    pub fn report() {
        let steps = STEPS.swap(0, Ordering::Relaxed).max(1);
        let ms = |c: &AtomicU64| c.swap(0, Ordering::Relaxed) as f64 / 1000.0 / steps as f64;
        let launches = LAUNCHES.swap(0, Ordering::Relaxed) as f64 / steps as f64;
        eprintln!(
            "[profile] ms/step over {steps} steps — scalar(sync) {:.2} | embed {:.2} | ple {:.2} | hc {:.2} | gdn {:.2} | attn {:.2} | moe {:.2} | head {:.2} | tail(sync+dtoh) {:.2} | launches/step {:.0}",
            ms(&SCALAR), ms(&EMBED), ms(&PLE), ms(&HC), ms(&SUB_GDN), ms(&SUB_ATTN), ms(&MOE), ms(&HEAD), ms(&TAIL), launches
        );
    }
    pub fn on() -> bool {
        std::env::var("CROW_PROFILE").is_ok()
    }
    // ---- per-kernel profile (CROW_KPROF=1) ----
    pub static KPROF: std::sync::Mutex<Option<std::collections::HashMap<String, (u64, u64)>>> =
        std::sync::Mutex::new(None);
    pub fn kprof_add(name: String, us: u64) {
        let mut g = KPROF.lock().unwrap();
        let m = g.get_or_insert_with(std::collections::HashMap::new);
        let e = m.entry(name).or_insert((0, 0));
        e.0 += 1;
        e.1 += us;
    }
    /// per-kernel table (sorted by total time), normalized per `steps`
    pub fn kprof_report(steps: u64) {
        let g = KPROF.lock().unwrap();
        let Some(m) = g.as_ref() else { return };
        let mut rows: Vec<(&String, &(u64, u64))> = m.iter().collect();
        rows.sort_by(|a, b| b.1 .1.cmp(&a.1 .1));
        let total: u64 = rows.iter().map(|r| r.1 .1).sum();
        let st = steps.max(1) as f64;
        eprintln!("[kprof] kernel time incl. one launch latency each, over {steps} steps (total {:.1} ms/step)", total as f64 / 1000.0 / st);
        eprintln!("[kprof] {:<22} {:>9} {:>11} {:>9} {:>6}", "kernel", "calls/step", "ms/step", "us/call", "share");
        for (n, (c, us)) in rows {
            eprintln!("[kprof] {:<22} {:>9.1} {:>11.3} {:>9.1} {:>5.1}%", n, *c as f64 / st, *us as f64 / 1000.0 / st, *us as f64 / *c as f64, 100.0 * *us as f64 / total as f64);
        }
    }
}

pub struct Weights {
    pub layer_sec: String,
    // head
    pub embed_host: Vec<u16>, // [V][H] BF16 bits (keep; rows widen to f32 at gather - halves the host RAM vs f32)
    pub lm_head: Dev,         // BF16 raw [V][2560]
    pub mx_norm: Dev,
    pub mx_down: PW,
    pub mx_up: PW,
    // per layer: hc = attn HC block, hc2 = mlp HC block
    pub hc: Vec<HcW>,
    pub hc2: Vec<HcW>,
    pub sub: Vec<SubW>,
    pub moe: Vec<MoeW>,
}

#[derive(Clone, Copy)]
pub struct Fp4 {
    pub w: Dev,
    pub gs: Dev,
}

/// Projection weight: NVFP4 (dequant on the fly) or BF16 keep (exact bit
/// shift inside gemv_bf16) — converter keep-set amendment 2026-09-03.
#[derive(Clone, Copy)]
pub enum PW {
    Fp4(Dev, Dev), // w, gs
    Bf16(Dev),
}

impl PW {
    /// `rows_p` = device i32 with the row count (warp-per-row BF16 kernel)
    pub unsafe fn launch_gemv(&self, k: &Kernels, rows: usize, rows_p: u64, t: usize, t_p: u64, x: u64, y: u64, k_dim: u64) {
        match self {
            PW::Fp4(w, gs) => launch_v(k.f("gemv_fp4_b"), rows as u32, t as u32, 1, 256, &[
                *w as u64, x, *gs as u64, y, k_dim]),
            // prefill-sized batches: 8-token bf16 tile GEMM (CROW_BF16_GEMM=0 -> warp GEMV)
            PW::Bf16(w) if t >= 8 && std::env::var("CROW_BF16_GEMM").as_deref() != Ok("0") => {
                launch_v(k.f("gemm_bf16_dense"), ((rows + 63) / 64) as u32, ((t + 7) / 8) as u32, 1, 128, &[
                    *w as u64, x, y, k_dim, rows_p, t_p])
            }
            PW::Bf16(w) => if bf16_w_on() {
                launch_v(k.f("gemv_bf16_w"), ((rows + 7) / 8) as u32, t as u32, 1, 256, &[*w as u64, x, y, k_dim, rows_p])
            } else {
                launch_v(k.f("gemv_bf16_b"), rows as u32, t as u32, 1, 256, &[*w as u64, x, y, k_dim])
            },
        }
    }
    pub unsafe fn launch_gemv1(&self, k: &Kernels, rows: usize, rows_p: u64, x: u64, y: u64, k_dim: u64) {
        match self {
            PW::Fp4(w, gs) => launch_v(k.f("gemv_fp4"), rows as u32, 1, 1, 256, &[
                *w as u64, x, *gs as u64, y, k_dim]),
            PW::Bf16(w) => if bf16_w_on() {
                launch_v(k.f("gemv_bf16_w"), ((rows + 7) / 8) as u32, 1, 1, 256, &[*w as u64, x, y, k_dim, rows_p])
            } else {
                launch_v(k.f("gemv_bf16"), rows as u32, 1, 1, 256, &[*w as u64, x, y, k_dim])
            },
        }
    }
}

pub struct HcW {
    pub norm: Dev, // f32 [4][2560] (keep)
    pub down: PW,  // [320][10240] — BF16 keep since the 2026-09-03 amendment
    pub up: PW,    // [10240][320] — BF16 keep
    pub inj: Fp4,  // [4][10240]
}

pub enum SubW {
    Gdn {
        qkv: Fp4,      // [10240][2560]
        conv: Dev,     // f32 [10240][4] (dequant at load)
        z: Fp4,        // [6144][2560]
        b: Fp4,        // [48][2560]
        a: Fp4,        // [48][2560]
        alog: Dev,     // f32 [48]
        dt: Dev,       // f32 [48]
        norm: Dev,     // f32 [128]
        out: Fp4,      // [2560][6144]
    },
    Attn {
        q: PW,         // [12288][2560] — BF16 keep (2026-09-03 amendment)
        k: PW,         // [512][2560] — BF16 keep
        v: Fp4,        // [512][2560]
        o: Fp4,        // [2560][6144]
        qn: Dev,       // f32 [256]
        kn: Dev,       // f32 [256]
        iqk: Fp4,      // [640][2560] QSA indexer
        iqln: Dev,     // f32 [128]
        ikln: Dev,     // f32 [128]
    },
}

pub struct MoeW {
    pub router: Dev,  // f32 [512][2560] (BF16 keep, dequant at load)
    pub router_bf: Dev, // bf16 [512][2560] twin for gemm_bf16_dense (prefill, CROW_ROUTER_GEMM=1)
    pub sg: Fp4,
    pub su: Fp4,
    pub sdn: Fp4,
    pub sgate: Dev, // f32 [1][2560]
}

pub struct Ple {
    pub key: Fp4,     // [10240][2560]
    pub value: Fp4,   // [2560][2560]
    pub norm_key: Dev,
    pub norm_query: Dev,
    pub norm_conv: Dev,
    pub conv: Dev,    // f32 [10240][4]
    pub multipliers: Vec<i64>,
    pub vocab_sizes: Vec<i64>,
    pub offsets: Vec<i64>,
    pub cache: Dev,   // [n_slots][108] raw NVFP4 rows
    pub gs: Dev,      // [n_slots] f32 per-slot global scale
    pub gs_host: Vec<f32>,
    pub n_slots: usize,
    pub slot_map: Vec<i32>,
    pub batch_tag: Vec<u32>, // #22: slot claimed in batch `batch` (collision probe within one chunk)
    pub batch: u32,
    pub req: u64,  // PLE rows requested (hit-rate counter, #16)
    pub miss: u64, // PLE rows filled from the container
    pub state: Dev,   // [10240][9] conv state
    pub shards: Vec<(crate::cnq::TensorInfo, f32)>,
}

pub struct Params {
    pub n320: Dev,
    pub n128: Dev,
    pub n2560: Dev,
    pub n640: Dev,
    pub n1280: Dev,
    pub n6144: Dev,
    pub n10240: Dev,
    pub n2048: Dev,
    pub n12288: Dev,
    pub n_conv_deq: Dev, // 10240*4
    pub n_selmax: Dev,   // 2051
    pub k_top: Dev,      // 512
    pub cap: Dev,        // pooled blocks capacity
    pub tmax: Dev,       // context
    pub mode: Dev,       // kv dtype mode
    // per-chunk refreshables
    pub t: Dev,
    pub init: Dev,
    pub pos_base: Dev,
    pub pos_base_sb: Dev, // [n_sb] pos_base + i * ATTN_SB, refreshed per attn_prompt (#16)
    pub pos_base_b4: Dev, // block base * 4 (rope for pooled rows)
    pub slot_base: Dev,
    pub ncb: Dev,        // [C] per-query complete block counts
    pub pos_row: Dev,    // [C] global positions (select tail)
    pub block_base: Dev,
    pub n_new: Dev,
    pub row_base: Dev,
    pub n_rows: Dev,
    pub slot1: Dev,      // [1] decode slot
    pub one: Dev,
    pub zero: Dev,
    pub n_vocab: Dev,
    // t-dependent elementwise counts (refreshed once per chunk/step)
    pub nt_low: Dev,
    pub nt_hct: Dev,
    pub nt_hc: Dev,
    pub nt_6144: Dev,
    pub nt_combo: Dev,
    // QSA fixed params
    pub q_heads4: Dev,
    pub q_heads1: Dev,
    pub pos_mul1: Dev,
    pub pos_mul4: Dev,
    pub stride512: Dev,
    pub stride128: Dev,
    pub keys_ring: Dev,  // i32 rows of the raw-key ring (row = pos % ring)
    pub qk_stride: Dev,
    pub ncb1: Dev,
    pub pos_row1: Dev,
    pub k_top10: Dev,
    pub nt_hct1: Dev,
    // dense MMA row counts (rows not covered by an existing n* param)
    pub nr4: Dev,   // HCN inject rows
    pub nr48: Dev,  // GDN b/a rows
    pub nr512: Dev, // KV_ROWS (attention v)
    pub n_splits: Dev, // ATTN_SPLITS
}

pub struct Scratch {
    pub h: Dev,        // [C][10240]
    pub emb: Dev,      // [C][2560] (ple gather out)
    pub mixed: Dev,    // [C][2560]
    pub low: Dev,      // [C][320]
    pub sil: Dev,
    pub mixw: Dev,     // [C][10240]
    pub normed: Dev,   // [C][10240] (HC rms output, mix_streams input)
    pub injr: Dev,     // [C][4]
    pub injw: Dev,
    pub x1: Dev,       // [C][10240]
    pub mixed_m: Dev,
    pub moe_out: Dev,  // [C][2560]
    pub h1: Dev,       // [C*10][1280]
    pub h2: Dev,       // [C*10][640]
    pub eo: Dev,       // [C*10][2560]
    pub xq_gu: Dev,    // [C][3*40*36] MMA quantized MoE input (per token, 3 levels)
    pub xq_dn: Dev,    // [C*10][3*10*36] MMA quantized silu·up (per combo, 3 levels)
    pub xq_m: Dev,     // [C][3*40*36] quantized mixed rows k2560 (GDN qkv/z/b/a, attn v/iqk)
    pub xq_v: Dev,     // [C][3*96*36] quantized 6144 rows (GDN out, attn o)
    pub xq_s: Dev,     // [C][3*10*36] quantized 640 rows (shared down input sh2)
    pub xq_e: Dev,     // [C][3*40*36] quantized PLE embed rows k2560
    pub sdown: Dev,    // [C][2560]
    pub sh12: Dev,     // [C][1280] shared-expert gate|up (p13 layout)
    pub sh2: Dev,      // [C][640]
    pub sgv: Dev,      // [C][1]
    pub rlog: Dev,     // [C][512]
    pub rids: Dev,     // [C][10] i32
    pub rwts: Dev,     // [C][10] f32
    pub gu_ptrs: Dev,  // [C][10] u64
    pub dn_ptrs: Dev,
    pub cold: Dev,     // [C] u32
    // gdn
    pub mq: Dev,       // [C][10240]
    pub mq_t: Dev,     // [10240][C]
    pub cout_t: Dev,   // [10240][C]
    pub gq: Dev, pub gk: Dev, pub gv: Dev, // [C][2048]/[C][2048]/[C][6144]
    pub gz: Dev,       // [C][6144]
    pub gb: Dev, pub ga: Dev, pub gbeta: Dev, pub gg: Dev, // [C][48]
    pub gqr: Dev, pub gkr: Dev, // [C][6144]
    pub gcore: Dev,    // [C][6144]
    pub gnorm: Dev,    // [C][6144]
    pub gout: Dev,     // [C][2560]
    // attention
    pub qg: Dev,       // [C][12288]
    pub aq: Dev, pub agate: Dev, // [C][6144]
    pub aqn: Dev, pub aqr: Dev,
    pub ak: Dev, pub akn: Dev, pub akr: Dev,
    pub av: Dev,       // [C][512]
    pub aout: Dev,     // [C][6144]
    pub agated: Dev,
    pub ay: Dev,       // [C][2560]
    // qsa
    pub qk: Dev,       // [C][640]
    pub q_nrm: Dev,    // [C][512]
    pub q_rot: Dev,
    pub pool_raw: Dev, // [Ccap][128]
    pub pool_nrm: Dev,
    pub pool_rot: Dev,
    pub scores: Dev,   // [C][cap]
    pub sel: Dev,      // [C][2051] i32
    pub sel_n: Dev,    // [C] i32
    // ple
    pub ple_key: Dev, pub ple_kn: Dev, // [C][10240]
    pub ple_val: Dev,  // [C][2560]
    pub ple_qn: Dev,   // [C][10240]
    pub ple_gate: Dev, // [C][4]
    pub ple_gs: Dev,   // [C][4] gate_signed
    pub ple_gated: Dev, // [C][10240]
    pub ple_gn: Dev,   // [C][10240]
    pub ple_out: Dev,  // [C][10240]
    pub ple_slots: Dev, // [C][16] i32
    // head
    pub part_o: Dev,   // [24][ATTN_SPLITS_MAX][256] decode attention partials
    pub part_ml: Dev,  // [24][ATTN_SPLITS_MAX][2]
    pub qsa_h1: Dev,   // #61a [QSA_PAR_BINS] u32, the CROW_QSA_PAR key histogram
    pub logits: Dev,   // [C][V]
    pub argmax: Dev,   // [1] i32
    pub mixed_final: Dev, // [C][2560]
    pub scratch_gn: Dev, // elementwise n scratch
}

pub struct Engine {
    pub k: Kernels,
    pub module: cuda::Module, // #18: kept so Drop can unload it after every kernel user is gone
    pub st: ThreeStates,
    pub res: Residency,
    pub w: Weights,
    pub ple: Ple,
    pub p: Params,
    pub s: Scratch,
    pub cfg: Config,
    pub pos: usize,
    pub history: Vec<i64>,
    pub done_blocks: usize,
    pub p_n: usize,
    pub sel_counts: Dev, // [48][512] u64 per-expert routing counts
    pub cnq: *mut Cnq,   // container handle for PLE row fills (loader thread)
    // ---- CUDA Graphs (CROW_GRAPH=1) ----
    pub graph_exec: u64, // CUgraphExec handle, 0 = not instantiated
    pub cap_stream: u64, // capture stream handle, 0 = not created
    pub scalar_stage: cuda::Pinned, // pinned [16] i32 staging for scalar refreshes
    pub embed_buf: Vec<f32>,        // persistent embedding staging (async HtoD source)
    // host staging slots for scalars uploaded inside the captured region —
    // the graph stores these HOST pointers, so they must be stable for the
    // engine's lifetime and each replay reads the freshest value
    pub sb_pack: Box<[i32; 2]>,
    // ---- cold-expert staging (CROW_STAGE, decode: t*TOPK <= stage.max) ----
    pub stage: Stage,
    pub pf_ncombo: Dev, // i32 t*TOPK (refreshed per prefill chunk; grouped MoE path)
    /// copy-engine prefetch ring (A-P3b): [2] VRAM copies of a layer's pinned
    /// cold slab (gu, dn), side stream + fill/done events; `pf_dma_live` is
    /// set for the chunk while the ring is being used (moe_run reads it)
    pub pf_ring_gu: [Dev; 2],
    pub pf_ring_dn: [Dev; 2],
    pub pf_stream: cudarc::driver::sys::CUstream,
    pub pf_ev_filled: [cudarc::driver::sys::CUevent; 2],
    pub pf_ev_done: [cudarc::driver::sys::CUevent; 2],
    pub pf_dma_live: std::cell::Cell<bool>,
    /// CROW_PF_ASYNC: staging side stream + plan/filled/done events (two slot sets)
    pub pa_stream: cudarc::driver::sys::CUstream,
    pub pa_ev_plan: cudarc::driver::sys::CUevent,
    pub pa_ev_filled: [cudarc::driver::sys::CUevent; 2],
    pub pa_ev_done: [cudarc::driver::sys::CUevent; 2],
    /// CROW_ROUTE_DUMP=1 (non-graph decode): routed expert ids per token per
    /// layer, [token][layer][10] - measurement A-V3 (hot-set coverage study)
    pub route_log: Vec<Vec<[i32; 10]>>,
    /// stream-side trickle adaptation (A-P3c, CROW_ADAPT_STREAM=1): created
    /// on first use by `trickle_tick`
    pub trickle: Option<Trickle>,
    /// #63b (the deferred order, default since #63c): the copies `trickle_tick`
    /// parked so `decode_step` can issue them AFTER the graph launch; always
    /// `None` under `CROW_TRICKLE_DEFER=0`, the eager fallback
    pub trickle_pend: Option<TrickleBatch>,
    /// decode-window adaptation (#17): routing counts at the last adaptation and
    /// an exponentially decayed count of the selections since (CROW_ADAPT_WINDOW=1)
    pub adapt_base: Vec<u64>,
    pub adapt_ema: Vec<f64>,
    /// #20: device-side sampler (CROW_SAMPLE=1 without CROW_SAMPLE_HOST=1) -
    /// the sample_k node behind argmax_k; its state buffers live here
    pub dev_sampler: Option<DevSampler>,
}

/// device state of the #20 sampler: presence mask [V] u8, xorshift64* state
/// [1] u64, profile {temp, top_p, presence: f32, top_k: i32}; `in_graph` is
/// set once the launch was captured into the decode graph
pub struct DevSampler {
    pub mask: Dev,
    pub rng: Dev,
    pub params: Dev,
    /// v2: per-slice top-k candidates [SAMPLE_PARTS * SAMPLE_MAXK] f32 / i32
    pub cand_v: Dev,
    pub cand_i: Dev,
    pub in_graph: std::cell::Cell<bool>,
}

/// side stream + events + the swaps in flight of the stream-side trickle
pub struct Trickle {
    pub stream: cudarc::driver::sys::CUstream,
    pub ev_side: cudarc::driver::sys::CUevent,   // recorded on the side stream after its copies
    pub ev_commit: cudarc::driver::sys::CUevent, // recorded on the compute stream after the table flips
    pub in_a: Vec<PendingSwap>, // phase A issued, commit A pending
    pub in_b: Vec<PendingSwap>, // phase B issued, commit B pending
    pub swaps: usize,
}

/// #63b: the side-stream copies of ONE tick, parked by `trickle_tick` unless
/// `CROW_TRICKLE_DEFER=0`, issued by `Engine::trickle_drain_after_launch`.
pub struct TrickleBatch {
    pub to_b: Vec<PendingSwap>,          // phase B: evicted hot slot -> pinned slot
    pub new_a: Vec<(usize, usize, u32)>, // phase A: (layer, evict slot, incoming id)
}

/// VRAM staging slots for cold routed experts + the rewritten combo
/// pointer tables the GEMVs consume when staging is active.
pub struct Stage {
    pub gu: Dev,      // [max][gu_bytes]
    pub dn: Dev,      // [max][dn_bytes]
    pub sgu: Dev,     // [max] u64 combo pointers (staged or hot)
    pub sdn: Dev,
    pub gu_b: Dev,    // i32 gu_bytes
    pub dn_b: Dev,    // i32 dn_bytes
    pub gu_bytes: u64, // host copy of the same two counts (#19d fix I3 assert)
    pub dn_bytes: u64,
    pub max: usize,   // combos with a staging slot
    // CROW_STAGE_DMA scratch: [max] u64 gu ptrs | [max] u64 dn ptrs | [..] u32 cold mask
    pub dma: Option<cuda::Pinned>,
    // ---- prefill grouped-GEMM plan (CROW_PF_GEMM) ----
    pub counts: Dev,   // [512] u32 (zeroed by moe_plan)
    pub offsets: Dev,  // [513] i32
    pub cursor: Dev,   // [512] u32
    pub perm: Dev,     // [C*10] i32 combos sorted by expert
    pub tiles: Dev,    // [max_tiles] int4
    pub n_tiles: Dev,  // [1] i32
    pub eptr: Dev,     // [max_tiles][2] u64
    pub grp: Dev,      // [MAX_GROUPS] i32 = 0..MAX_GROUPS (device group index per launch)
    pub tg: Dev,       // i32 PF_TG
    pub max_tiles_p: Dev,
    pub max_tiles: usize,
}

/// tiles (<= 8 tokens of one expert) per staged group: <= PF_TG distinct
/// experts share the staging slots with the decode path (stage.max slots).
/// Default 64 since 2026-09-09 (#10, robin's call, gated on ten tasks with CROW_PF_ASYNC=2; 32 before)
pub const PF_TG: usize = 64;
/// runtime tiles-per-group (CROW_PF_TG, default PF_TG): larger groups cut the
/// 5-launch chain per group and raise the tile-GEMM grid; costs
/// CROW_PF_TG x 2.76 MB of staging slots
pub fn pf_tg() -> usize {
    use std::sync::OnceLock;
    static V: OnceLock<usize> = OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_PF_TG").ok().and_then(|v| v.parse().ok()).unwrap_or(PF_TG).clamp(8, 512))
}
pub const PF_MAX_GROUPS: usize = 256;

/// CROW_PF_GEMM (default on): expert-grouped tile GEMM for prefill-sized
/// MoE batches; =0 keeps the per-combo GEMV path.
/// CROW_GDN_REG=0 restores the global-memory delta-rule scan (bit-identical fallback)
/// (1 = both kernels, p = prefill scan only, s = decode step only, 0 = off)
fn gdn_reg_mode() -> u8 {
    use std::sync::OnceLock;
    static V: OnceLock<u8> = OnceLock::new();
    *V.get_or_init(|| match std::env::var("CROW_GDN_REG").as_deref() { Ok("0") => 0, Ok("p") => 1, Ok("s") => 2, _ => 3 })
}
fn gdn_reg_on() -> bool { gdn_reg_mode() != 0 }

/// CROW_PF_DMA=0 disables the copy-engine prefetch of the cold tier during
/// prefill (a 2-slot VRAM ring of one layer's pinned cold slab; costs
/// ~2 x cold-slab bytes of VRAM = fewer hot experts)
fn pf_dma_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    // opt-in (CROW_PF_DMA=1): with the exact NVFP4 tier the host-pinned budget
    // already pins N >= ~139 on the 64 GB box, so the ring makes the plan
    // infeasible (measured 2026-09-04); it fits with a low-bit cold-only tier
    *ON.get_or_init(|| std::env::var("CROW_PF_DMA").as_deref() == Ok("1"))
}

fn pf_gemm_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_PF_GEMM").as_deref() != Ok("0"))
}

/// CROW_PF_ASYNC=1: the prefill staging copies (`stage_tiles`, SM reads over
/// PCIe) run on a side stream into two slot sets (group parity), overlapping
/// the previous group's tile GEMMs; costs one extra set of PF_TG staging slots
/// (measured 2026-09-06: stage_tiles = 40 % of the prefill kernel time at 16k).
/// CROW_PF_ASYNC=2 (the copy engine stages, host memcpys ahead on the side stream) is the default since 2026-09-09
/// (#10, robin's call, gated: parity 8/512/1024 + ten tasks ids == final4 without the env); CROW_PF_ASYNC=0 = the
/// previous synchronous staging (one slot set).
pub fn pf_async_mode() -> i32 {
    static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_PF_ASYNC").ok().and_then(|v| v.parse().ok()).unwrap_or(2))
}
pub fn pf_async_on() -> bool { pf_async_mode() >= 1 }

/// blocks per expert in the staging copies (CROW_STAGE_SPLIT, default 8; 16/32/64
/// keep more PCIe requests in flight - measured 2026-09-04 with the WC tier)
pub fn stage_split() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_STAGE_SPLIT").ok().and_then(|v| v.parse().ok()).filter(|&k| [1u32, 2, 4, 8, 16, 32, 64].contains(&k)).unwrap_or(8))
}

/// CROW_STAGE (default on): stage cold experts into VRAM with coalesced
/// PCIe reads before the routed GEMVs; CROW_STAGE=0 = direct zero-copy.
fn stage_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_STAGE").as_deref() != Ok("0"))
}

/// CROW_STAGE_DMA (default off, measurement, #19b): the cold combos of a layer are
/// staged by the COPY ENGINE (one cuMemcpyDtoDAsync per cold combo and matrix from the
/// mapped host pointer, pcie_probe variant f = 55.0 GB/s) instead of the stage_cold
/// kernel (31.5 GB/s SM-read ceiling). The routed pointers are known on the HOST only
/// after router_top10 of that layer, so the path needs the decode graph off:
/// graph_on() forces CROW_GRAPH off when this switch is on.
pub fn stage_dma_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_STAGE_DMA").as_deref() == Ok("1"))
}

/// CROW_STAGE_KERNEL (default 2 = stage_cold_ca, the #19d persistent cp.async.cg 4 KB
/// tile copy; `1` selects the old stage_cold, kept as the fallback; any other value
/// and unset take the default). DEFAULT SINCE #19e, 2026-09-12: stage_cold_ca ran the
/// #59 profile arm at 26.46 ms per decode token against 29.68 for stage_cold, a gain
/// of 3.22 ms per token at 19.5 x the baseline spread (decode_out/srv-19d.log, RTX
/// 5090, 2026-09-11). The variant is a pure copy, so both settings must be
/// byte-identical in parity.
pub fn stage_kernel_ca() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_STAGE_KERNEL").as_deref() != Ok("1"))
}

/// CROW_STAGE_BLOCKS (default 40, accepted 8 to 512, other values fall back): the
/// persistent grid of stage_cold_ca, read whenever that kernel runs, which is the
/// default since #19e. 19c: fewer blocks are faster, 40 x 256 was the best row
/// (52.9 GB/s), 20 x 256 gave 52.6 and 160 x 256 gave 52.4. 19d measured 40 and 80
/// tied inside their own spreads and 20 worse by 0.1532 ms per token.
pub fn stage_blocks() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_STAGE_BLOCKS").ok().and_then(|v| v.parse().ok()).filter(|&g| (8..=512).contains(&g)).unwrap_or(40))
}

pub struct LoadReport {
    pub dense_bytes: u64,
    pub residency_report: Vec<String>,
    pub states_report: Vec<String>,
    pub lm_head_bytes: u64,
    pub ple_cache_bytes: u64,
}

impl Engine {
    /// Full load: dense weights → states (with N clamp) → residency → scratch.
    /// `warmup_counts` feeds warm-up promotion when no sidecar exists yet.
    pub unsafe fn load(
        cnq: &mut Cnq,
        cfg: Config,
        warmup_counts: Option<&[[u64; E]; LAYERS]>,
        sidecar_path: &str,
        persist: bool,
        log: &mut dyn FnMut(&str),
    ) -> (Engine, LoadReport) {
        let sec = "text";
        let t0 = std::time::Instant::now();
        engine_lock_acquire();

        // ---- head + dense (residents before the budget verify) ----
        log("loading embeddings (BF16 keep → host f32) …");
        let emb_t = cnq.find("model.language_model.embed_tokens.weight", sec).clone();
        let emb_raw = cnq.read_bytes(&emb_t);
        let embed_host: Vec<u16> = emb_raw.chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect();
        assert_eq!(embed_host.len(), V * H);
        drop(emb_raw);

        let lm_t = cnq.find("lm_head.weight", sec).clone();
        let lm_raw = cnq.read_bytes(&lm_t);
        let lm_head = cuda::upload_dev(&lm_raw);
        let lm_head_bytes = lm_raw.len() as u64;
        drop(lm_raw);

        let mx_norm = load_f32(cnq, "model.language_model.hyper_connection_mixer.hc_norm.weight", sec);
        let mx_down = load_pw(cnq, "model.language_model.hyper_connection_mixer.input_mix_weight_down.weight", sec);
        let mx_up = load_pw(cnq, "model.language_model.hyper_connection_mixer.input_mix_weight_up.weight", sec);

        log("loading 48 dense layer bundles (FP4 + keeps) …");
        let mut hc = Vec::with_capacity(LAYERS);
        let mut sub = Vec::with_capacity(LAYERS);
        let mut moe = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let pfx = |s: &str| format!("model.language_model.layers.{l}.{s}");
            hc.push(HcW {
                norm: load_f32(cnq, &pfx("attn_hyper_connection.hc_norm.weight"), sec),
                down: load_pw(cnq, &pfx("attn_hyper_connection.input_mix_weight_down.weight"), sec),
                up: load_pw(cnq, &pfx("attn_hyper_connection.input_mix_weight_up.weight"), sec),
                inj: load_fp4(cnq, &pfx("attn_hyper_connection.block_inject_weight.weight"), sec),
            });
            if is_attn(l) {
                sub.push(SubW::Attn {
                    q: load_pw(cnq, &pfx("self_attn.q_proj.weight"), sec),
                    k: load_pw(cnq, &pfx("self_attn.k_proj.weight"), sec),
                    v: load_fp4(cnq, &pfx("self_attn.v_proj.weight"), sec),
                    o: load_fp4(cnq, &pfx("self_attn.o_proj.weight"), sec),
                    qn: load_f32(cnq, &pfx("self_attn.q_norm.weight"), sec),
                    kn: load_f32(cnq, &pfx("self_attn.k_norm.weight"), sec),
                    iqk: load_fp4(cnq, &pfx("self_attn.indexer.index_qk_proj.weight"), sec),
                    iqln: load_f32(cnq, &pfx("self_attn.indexer.q_layernorm.weight"), sec),
                    ikln: load_f32(cnq, &pfx("self_attn.indexer.k_layernorm.weight"), sec),
                });
            } else {
                sub.push(SubW::Gdn {
                    qkv: load_fp4(cnq, &pfx("linear_attn.in_proj_qkv.weight"), sec),
                    conv: dequant_fp4_dev(cnq, &pfx("linear_attn.conv1d.weight"), sec, GDN_CONV * 4),
                    z: load_fp4(cnq, &pfx("linear_attn.in_proj_z.weight"), sec),
                    b: load_fp4(cnq, &pfx("linear_attn.in_proj_b.weight"), sec),
                    a: load_fp4(cnq, &pfx("linear_attn.in_proj_a.weight"), sec),
                    alog: load_small_f32(cnq, &pfx("linear_attn.A_log"), sec, 48),
                    dt: load_small_f32(cnq, &pfx("linear_attn.dt_bias"), sec, 48),
                    norm: load_f32(cnq, &pfx("linear_attn.norm.weight"), sec),
                    out: load_fp4(cnq, &pfx("linear_attn.out_proj.weight"), sec),
                });
            }
            moe.push(MoeW {
                router: load_f32(cnq, &pfx("mlp.gate.weight"), sec),
                router_bf: load_bf16_twin(cnq, &pfx("mlp.gate.weight"), sec),
                sg: load_fp4(cnq, &pfx("mlp.shared_expert.gate_proj.weight"), sec),
                su: load_fp4(cnq, &pfx("mlp.shared_expert.up_proj.weight"), sec),
                sdn: load_fp4(cnq, &pfx("mlp.shared_expert.down_proj.weight"), sec),
                sgate: load_f32(cnq, &pfx("mlp.shared_expert_gate.weight"), sec),
            });
            if l % 12 == 0 {
                log(&format!("  dense layers {l}/48"));
            }
        }
        // mlp HC bundle (second per-layer HC block)
        let mut hc2 = Vec::with_capacity(LAYERS);
        for l in 0..LAYERS {
            let pfx = |s: &str| format!("model.language_model.layers.{l}.{s}");
            hc2.push(HcW {
                norm: load_f32(cnq, &pfx("mlp_hyper_connection.hc_norm.weight"), sec),
                down: load_pw(cnq, &pfx("mlp_hyper_connection.input_mix_weight_down.weight"), sec),
                up: load_pw(cnq, &pfx("mlp_hyper_connection.input_mix_weight_up.weight"), sec),
                inj: load_fp4(cnq, &pfx("mlp_hyper_connection.block_inject_weight.weight"), sec),
            });
        }
        let dense_measured = cuda::total_vram_bytes() - cuda::free_vram_bytes();

        // ---- PLE weights (layer index 1) ----
        log("loading PLE (layer 1) weights + row cache …");
        // CROW_PLE_CACHE_MB=<n>: hot-row cache size override (VRAM diet, 2026-09-05)
        let ple_bytes = std::env::var("CROW_PLE_CACHE_MB").ok().and_then(|v| v.parse::<u64>().ok())
            .map(|mb| mb << 20).unwrap_or(cfg.ple_cache_bytes);
        let ple = Ple::load(cnq, ple_bytes);
        let ple_cache_bytes = ple.n_slots as u64 * (108 + 4);
        log("PLE done — budget verify next");

        // ---- scratch + staging BEFORE the budget: the planner measures free VRAM,
        // so everything chunk-sized must already be resident (C=512 scratch is
        // ~0.7 GB; an unplanned allocation past the card limit gets paged by
        // WDDM and silently costs 3x per token - found 2026-09-04 at C=1024) ----
        let slabs = crate::residency::expert_slab_info(cnq, 0, sec);
        let per_expert_unit = (slabs.gu_bytes + slabs.dn_bytes) * LAYERS as u64;
        let s = Scratch::alloc(cfg.prompt_chunk);
        // cold staging: decode-sized batches only (t*TOPK <= stage_max)
        let stage_max = (2 * TOPK).max(pf_tg() * if pf_async_on() { 2 } else { 1 }); // 2 x 64 slots x 2.76 MB = 354 MB (default since 2026-09-09; CROW_PF_ASYNC=0 CROW_PF_TG=32 = 88 MB)
        // #19e fix C4 of 19d: CROW_STAGE_SPLIT shapes the stage_cold grid only, so its
        // assert gates the kernel 1 fallback only. Kernel 2, the default, carries no
        // tail tile and needs 4 KB multiples instead, asserted here at load and again
        // at the launch site.
        if stage_kernel_ca() {
            assert!(slabs.gu_bytes % 4096 == 0 && slabs.dn_bytes % 4096 == 0,
                "stage_cold_ca needs both expert slab byte counts to be multiples of 4096 (gate_up {} B, down {} B); CROW_STAGE_KERNEL=1 falls back to stage_cold",
                slabs.gu_bytes, slabs.dn_bytes);
        } else {
            assert!(slabs.gu_bytes % (16 * stage_split() as u64) == 0 && slabs.dn_bytes % (16 * stage_split() as u64) == 0,
                "expert slab bytes must split into 16-byte units x STAGE_SPLIT");
        }
        // #19e, 2026-09-12: ONE line per engine process names the staging kernel that
        // will run, so every future log says which kernel produced it. It sits in the
        // [load] block and not next to the [policy] line of geo::apply_adapt_policy,
        // because decode parity (decode.rs, the parity gate) never calls that function
        // and would print no such line at all (parity.exe does, through apply_chunk_policy).
        // CROW_STAGE_DMA and the low-bit tier
        // still take precedence at the launch site (moe_run).
        let sk = std::env::var("CROW_STAGE_KERNEL").unwrap_or_else(|_| "unset".to_string());
        if stage_kernel_ca() {
            println!("[stage] kernel stage_cold_ca, CROW_STAGE_KERNEL {} (default 2), {} blocks x 256 threads, 4096 B tiles (gate_up {} B = {} tiles, down {} B = {} tiles)",
                sk, stage_blocks(), slabs.gu_bytes, slabs.gu_bytes / 4096, slabs.dn_bytes, slabs.dn_bytes / 4096);
        } else {
            println!("[stage] kernel stage_cold, CROW_STAGE_KERNEL {}, grid t * TOPK x 2 x {} (CROW_STAGE_SPLIT) x 256 threads (gate_up {} B, down {} B)",
                sk, stage_split(), slabs.gu_bytes, slabs.dn_bytes);
        }
        // #63c, 2026-09-12: ONE line per engine process names the trickle issue
        // point, next to the [stage] line and for the same reason: every future
        // log says which order produced it. It sits in the [load] block, so the
        // parity gate prints it too (decode parity never calls apply_adapt_policy).
        let td = std::env::var("CROW_TRICKLE_DEFER").unwrap_or_else(|_| "unset".to_string());
        println!("[trickle] copies issued {}, CROW_TRICKLE_DEFER {} (unset or any value but 0 defers)",
            if trickle_defer_on() { "AFTER the graph launch, in decode_step (deferred, default)" }
            else { "BEFORE the launch, in trickle_tick (eager fallback)" }, td);
        // worst case every expert ends with a partial tile: t*10/8 + 512 tiles
        let max_tiles = cfg.prompt_chunk * TOPK / 8 + E + 1;
        let stage = Stage {
            gu: cuda::alloc_zeroed(stage_max * slabs.gu_bytes as usize),
            dn: cuda::alloc_zeroed(stage_max * slabs.dn_bytes as usize),
            sgu: cuda::alloc_zeroed(stage_max * 8),
            sdn: cuda::alloc_zeroed(stage_max * 8),
            gu_b: cuda::to_i32_dev(&[slabs.gu_bytes as i32]),
            dn_b: cuda::to_i32_dev(&[slabs.dn_bytes as i32]),
            gu_bytes: slabs.gu_bytes,
            dn_bytes: slabs.dn_bytes,
            max: stage_max,
            dma: if stage_dma_on() { Some(cuda::Pinned::alloc(stage_max * 16 + 256)) } else { None },
            counts: cuda::alloc_zeroed(E * 4),
            offsets: cuda::alloc_zeroed((E + 1) * 4),
            cursor: cuda::alloc_zeroed(E * 4),
            perm: cuda::alloc_zeroed(cfg.prompt_chunk.max(1) * TOPK * 4),
            tiles: cuda::alloc_zeroed(max_tiles * 16),
            n_tiles: cuda::alloc_zeroed(4),
            eptr: cuda::alloc_zeroed(max_tiles * 16),
            grp: cuda::to_i32_dev(&(0..PF_MAX_GROUPS as i32).collect::<Vec<_>>()),
            tg: cuda::to_i32_dev(&[pf_tg() as i32, if pf_async_on() { 2 } else { 1 }, pf_async_mode()]),
            max_tiles_p: cuda::to_i32_dev(&[max_tiles as i32]),
            max_tiles,
        };
        if pf_async_on() {
            log(&format!("prefill staging on a side stream (CROW_PF_ASYNC={}): 2 x {} slots", pf_async_mode(), pf_tg()));
        }
        let scratch_measured = cuda::total_vram_bytes() - cuda::free_vram_bytes() - dense_measured;
        log(&format!("scratch + staging resident: {:.0} MiB (chunk {})", scratch_measured as f64 / (1 << 20) as f64, cfg.prompt_chunk));

        // ---- budget verify + states (#9) ----
        // dense + PLE row cache + scratch + staging are all resident already
        // (free0 excludes them); pending = launch/param slack only. The old
        // planner double-counted the 1 GiB PLE cache here while the ~0.9 GB
        // scratch went unplanned.
        let _ = ple_cache_bytes;
        // + the prefetch ring (2 x one layer's cold slab; sized for N down to
        //   n_hot-24, verified after the residency build)
        let ring_reserve = if pf_dma_on() && mma_on() && pf_gemm_on() {
            2 * (E - cfg.n_hot.saturating_sub(24).min(E)) as u64 * (slabs.gu_bytes + slabs.dn_bytes)
        } else { 0 };
        let pending = (128u64 << 20) + ring_reserve;
        // pinned-side sizing follows the cold tier actually used (record size
        // of a low-bit tier, full tier = constant; see residency::build)
        let (cold_unit, cold_fixed) = match std::env::var("CROW_COLD_TIER").ok() {
            Some(p) => {
                let (h, _) = crate::residency::read_coldtier_header(&p);
                let rec = h["gu_record_bytes"].as_u64().unwrap() + h["dn_record_bytes"].as_u64().unwrap();
                let unit = rec * LAYERS as u64;
                // FULL tier (every expert pinned, enables the prompt-adaptive hot set)
                // when it fits the host budget, else cold-only; CROW_COLD_FULL=1/0 forces
                let full = match std::env::var("CROW_COLD_FULL").as_deref() {
                    Ok("1") => true, Ok("0") => false, _ => E as u64 * unit <= cfg.host_pinned_budget };
                (unit, full)
            }
            None => (per_expert_unit, false),
        };
        let (st, st_rep) = ThreeStates::allocate(&cfg, pending, per_expert_unit, cold_unit, cold_fixed);
        for l in &st_rep.lines {
            log(&format!("  [budget] {l}"));
        }

        // ---- residency (#8) ----
        log("building residency (hot VRAM slabs + pinned cold tier) …");
        let res = Residency::build(cnq, sec, st_res_n(&st_rep, cfg.n_hot), warmup_counts, sidecar_path, persist, cold_fixed, cfg.adapt.spare, &mut |m| {
            log(&format!("  [residency] {m}"));
        });
        // ---- copy-engine prefetch ring (A-P3b) ----
        let (ring_gu, ring_dn, pf_stream) = if ring_reserve > 0 {
            let need_gu = res.cold_gu.iter().map(|p| p.bytes as u64).max().unwrap_or(0);
            let need_dn = res.cold_dn.iter().map(|p| p.bytes as u64).max().unwrap_or(0);
            if 2 * (need_gu + need_dn) > ring_reserve {
                log(&format!("prefetch ring needs {:.0} MiB but only {:.0} MiB were planned (N clamped below n_hot-24) - CROW_PF_DMA disabled for this run",
                    2.0 * (need_gu + need_dn) as f64 / (1 << 20) as f64, ring_reserve as f64 / (1 << 20) as f64));
                ([0, 0], [0, 0], std::ptr::null_mut())
            } else {
                log(&format!("prefetch ring: 2 x ({:.0} + {:.0}) MiB VRAM, copy engine on a side stream",
                    need_gu as f64 / (1 << 20) as f64, need_dn as f64 / (1 << 20) as f64));
                ([cuda::alloc_zeroed(need_gu as usize), cuda::alloc_zeroed(need_gu as usize)],
                 [cuda::alloc_zeroed(need_dn as usize), cuda::alloc_zeroed(need_dn as usize)],
                 cuda::stream_create_non_blocking())
            }
        } else { ([0, 0], [0, 0], std::ptr::null_mut()) };

        // ---- kernels + params + scratch ----
        crate::kernels::kprof_init();
        let module = cuda::compile(&crate::kernels::KERNEL_SRC);
        let k = Kernels::new(&module);
        let p = Params::setup(&cfg, &st);
        let sel_counts = cuda::alloc_zeroed(LAYERS * E * 8);
        let w = Weights {
            layer_sec: sec.to_string(),
            embed_host,
            lm_head,
            mx_norm,
            mx_down,
            mx_up,
            hc,
            hc2,
            sub,
            moe,
        };

        let dense_after = cuda::total_vram_bytes() - cuda::free_vram_bytes();
        log(&format!(
            "load done in {:.0} s — VRAM used {:.2} GiB (dense {:.2} GiB + hot experts {} × 48 × {:.2} MB + states)",
            t0.elapsed().as_secs_f64(),
            dense_after as f64 / (1 << 30) as f64,
            dense_measured as f64 / (1 << 30) as f64,
            res.n,
            (res.gu_bytes + res.dn_bytes) as f64 / (1 << 20) as f64
        ));

        let report = LoadReport {
            dense_bytes: dense_measured,
            residency_report: vec![format!("n={} source={}", res.n, res.source)],
            states_report: st_rep.lines,
            lm_head_bytes,
            ple_cache_bytes,
        };

        (
            Engine {
                k,
                module,
                st,
                res,
                w,
                ple,
                p,
                s,
                p_n: cfg.prompt_chunk,
                cfg,
                pos: 0,
                history: Vec::new(),
                done_blocks: 0,
                sel_counts,
                stage,
                pf_ncombo: cuda::to_i32_dev(&[TOPK as i32]),
                pf_ring_gu: ring_gu,
                pf_ring_dn: ring_dn,
                pf_stream,
                pf_ev_filled: [cuda::event_create(), cuda::event_create()],
                pf_ev_done: [cuda::event_create(), cuda::event_create()],
                pf_dma_live: std::cell::Cell::new(false),
                pa_stream: if pf_async_on() { cuda::stream_create_priority() } else { std::ptr::null_mut() },
                pa_ev_plan: cuda::event_create(),
                pa_ev_filled: [cuda::event_create(), cuda::event_create()],
                pa_ev_done: [cuda::event_create(), cuda::event_create()],
                route_log: Vec::new(),
                trickle: None,
                trickle_pend: None,
                adapt_base: Vec::new(),
                adapt_ema: Vec::new(),
                dev_sampler: None,
                cnq: std::ptr::null_mut(),
                graph_exec: 0,
                cap_stream: 0,
                scalar_stage: unsafe { cuda::Pinned::alloc(16 * 4) },
                embed_buf: vec![0f32; HCT],
                sb_pack: Box::new([0i32; 2]),
            },
            report,
        )
    }
}

fn st_res_n(rep: &crate::manager::AllocReport, target: usize) -> usize {
    if rep.effective_n > 0 && rep.effective_n <= target {
        rep.effective_n
    } else {
        target
    }
}

// ---------------- tensor loaders ----------------

/// bf16 device copy of a BF16-keep tensor (exact: the keep is bf16)
pub unsafe fn load_bf16_twin(cnq: &mut Cnq, name: &str, sec: &str) -> Dev {
    let v = cnq.read_f32(name, sec);
    let b: Vec<u16> = v.iter().map(|x| (x.to_bits() >> 16) as u16).collect();
    cuda::upload_dev(std::slice::from_raw_parts(b.as_ptr() as *const u8, b.len() * 2))
}

pub unsafe fn load_f32(cnq: &mut Cnq, name: &str, sec: &str) -> Dev {
    let v = cnq.read_f32(name, sec);
    cuda::to_f32_dev(&v)
}

pub unsafe fn load_fp4(cnq: &mut Cnq, name: &str, sec: &str) -> Fp4 {
    let t = cnq.find(name, sec).clone();
    assert_ne!(t.dtype, "bf16", "{name}: expected NVFP4");
    let raw = cnq.read_bytes(&t);
    let w = cuda::upload_dev(&raw);
    let gs = cuda::to_f32_dev(&[t.global_scale]);
    Fp4 { w, gs }
}

/// dtype-agnostic loader: nvfp4 → FP4 GEMV, bf16 keep → BF16 GEMV
pub unsafe fn load_pw(cnq: &mut Cnq, name: &str, sec: &str) -> PW {
    let t = cnq.find(name, sec).clone();
    if t.dtype == "bf16" {
        let raw = cnq.read_bytes(&t);
        PW::Bf16(cuda::upload_dev(&raw))
    } else {
        let Fp4 { w, gs } = load_fp4(cnq, name, sec);
        PW::Fp4(w, gs)
    }
}

pub unsafe fn dequant_fp4_dev(cnq: &mut Cnq, name: &str, sec: &str, n: usize) -> Dev {
    let t = cnq.find(name, sec).clone();
    let raw = cnq.read_bytes(&t);
    let mut out = vec![0f32; n];
    let mut blk = [0f32; 64];
    for (b, chunk) in raw.chunks_exact(36).enumerate() {
        cnq::dequant_block(chunk, t.global_scale, &mut blk);
        out[b * 64..(b + 1) * 64].copy_from_slice(&blk);
    }
    cuda::to_f32_dev(&out)
}

/// tiny tensors (A_log, dt_bias): any dtype → host f32 → device
pub unsafe fn load_small_f32(cnq: &mut Cnq, name: &str, sec: &str, n: usize) -> Dev {
    let t = cnq.find(name, sec).clone();
    let raw = cnq.read_bytes(&t);
    let v: Vec<f32> = match t.dtype.as_str() {
        "bf16" => cnq::bf16_bytes_to_f32(&raw),
        "i64" => panic!("{name}: i64 unexpected here"),
        _ => {
            let mut out = vec![0f32; n];
            let mut blk = [0f32; 64];
            for (b, chunk) in raw.chunks_exact(36).enumerate() {
                cnq::dequant_block(chunk, t.global_scale, &mut blk);
                out[b * 64..(b + 1) * 64].copy_from_slice(&blk);
            }
            out
        }
    };
    assert_eq!(v.len(), n, "{name}: size mismatch");
    cuda::to_f32_dev(&v)
}

impl Ple {
    pub unsafe fn load(cnq: &mut Cnq, cache_bytes: u64) -> Ple {
        // projections/norms/conv + the I64 tables live in the `text` section;
        // only the 128 big shard tables carry the `ple` section tag
        let sec = "text";
        let P = |s: &str| format!("model.language_model.layers.1.ple.{s}");
        let key = load_fp4(cnq, &P("key_proj.weight"), sec);
        let value = load_fp4(cnq, &P("value_proj.weight"), sec);
        let norm_key = load_f32(cnq, &P("norm_key.weight"), sec);
        let norm_query = load_f32(cnq, &P("norm_query.weight"), sec);
        let norm_conv = load_f32(cnq, &P("norm_conv.weight"), sec);
        let conv = dequant_fp4_dev(cnq, &P("conv1d.weight"), sec, GDN_CONV * 4);
        let multipliers = cnq.read_i64(&P("ple_embedding.layer_multipliers"), sec);
        let vocab_sizes = cnq.read_i64(&P("ple_embedding.ngram_heads_vocab_sizes"), sec);
        let offsets = cnq.read_i64(&P("ple_embedding.ngram_heads_offsets"), sec);

        let n_slots = (cache_bytes / 112) as usize; // 108 B row + 4 B gs per slot
        let cache = cuda::alloc_zeroed(n_slots * 108);
        let gs = cuda::alloc_zeroed(n_slots * 4);
        let state = cuda::alloc_zeroed(GDN_CONV * 9 * 4);

        // shard tensor infos + global scales (128 shards)
        let mut shards = Vec::with_capacity(128);
        for i in 0..128 {
            let name = format!("model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{i}.weight");
            let t = cnq.find(&name, "ple").clone();
            shards.push((t, 1.0f32));
        }

        Ple {
            key,
            value,
            norm_key,
            norm_query,
            norm_conv,
            conv,
            multipliers,
            vocab_sizes,
            offsets,
            cache,
            gs,
            gs_host: vec![0f32; n_slots],
            n_slots,
            slot_map: vec![-1i32; n_slots],
            batch_tag: vec![0u32; n_slots],
            batch: 0,
            req: 0,
            miss: 0,
            state,
            shards,
        }
    }

    /// host n-gram index math (p15-verified) with a real-history prefix
    pub fn ngram_ids(&self, history_prefix: &[i64], ids: &[i64]) -> Vec<Vec<i64>> {
        let mut full: Vec<i64> = history_prefix.to_vec();
        full.extend_from_slice(ids);
        let multipliers = &self.multipliers;
        let vocab_sizes = &self.vocab_sizes;
        let offsets = &self.offsets;
        let n = full.len();
        let history = &full[..];
        let shifted: Vec<Vec<i64>> = (0..PLE_NGRAM)
            .map(|shift| {
                let mut eos_pos = vec![-1i64; n];
                for (i, &v) in history.iter().enumerate() {
                    if v == PLE_EOS {
                        eos_pos[i] = i as i64;
                    }
                }
                let mut prev_incl = vec![0i64; n];
                let mut run = i64::MIN;
                for i in 0..n {
                    run = run.max(eos_pos[i]);
                    prev_incl[i] = run;
                }
                (0..n)
                    .map(|i| {
                        let prev = if i == 0 { -1 } else { prev_incl[i - 1] };
                        let segment_start = prev + 1;
                        let pos_in_seg = i as i64 - segment_start;
                        let src = i as i64 - shift as i64;
                        let gather = src.max(0) as usize;
                        let valid = pos_in_seg >= shift as i64 && src >= 0;
                        if valid { history[gather] } else { PLE_EOS }
                    })
                    .collect()
            })
            .collect();
        let mut out = Vec::new();
        for i in 0..n {
            let mut row = Vec::with_capacity(PLE_NHEADS);
            for ngram in 2..=PLE_NGRAM {
                let start = (ngram - 2) * PLE_HEADS_PER_NGRAM;
                let mut mixed = (shifted[0][i] as u64).wrapping_mul(multipliers[0] as u64);
                for p in 1..ngram {
                    mixed ^= (shifted[p][i] as u64).wrapping_mul(multipliers[p] as u64);
                }
                let mixed = mixed as i64;
                for h in 0..PLE_HEADS_PER_NGRAM {
                    row.push(mixed.rem_euclid(vocab_sizes[start + h]) + offsets[start + h]);
                }
            }
            out.push(row);
        }
        out
    }

    /// absolute container offsets of the n-gram rows a token chunk will need
    /// (prefetch pipeline: the next chunk's rows are touched on a helper
    /// thread while the current chunk computes, so ensure_rows hits the page
    /// cache instead of NVMe - HugeCTR-HPS pattern, spec R6)
    pub fn row_offsets(&self, cnq: &Cnq, prefix: &[i64], chunk: &[i64]) -> Vec<u64> {
        let all = self.ngram_ids(prefix, chunk);
        let mut out = Vec::with_capacity(chunk.len() * PLE_NHEADS);
        for row in &all[prefix.len()..] {
            for &id in row {
                let slot = (id as u64 % self.n_slots as u64) as usize;
                if self.slot_map[slot] == id as i32 {
                    continue; // already cached in VRAM
                }
                let shard = (id / PLE_ROWS_PER_SHARD) as usize;
                let r = (id % PLE_ROWS_PER_SHARD) as u64;
                out.push(cnq.abs_offset(&self.shards[shard].0, r * 108));
            }
        }
        out
    }

    /// map rows to cache slots, filling misses from the container (NVMe → VRAM)
    pub unsafe fn ensure_rows(&mut self, cnq: &mut Cnq, ngids: &[i64]) -> Vec<i32> {
        let mut slots = vec![0i32; ngids.len()];
        let mut fill_rows: Vec<(usize, i64)> = Vec::new();
        // #22 (2026-09-05): direct-mapped slots collided WITHIN one chunk (16 rows per
        // token, 16k rows at t = 1024): two ids claimed the same slot, the later
        // upload won, the other token read a foreign row - and which one won followed
        // the HashMap iteration order below, i.e. differed per process. Now a slot
        // claimed earlier in this batch by another id is probed past (linear), so
        // every id of the batch owns its slot; cross-batch eviction stays direct-mapped.
        self.batch = self.batch.wrapping_add(1);
        if self.batch == 0 { for v in self.batch_tag.iter_mut() { *v = 0; } self.batch = 1; }
        for (i, &id) in ngids.iter().enumerate() {
            let mut slot = (id as u64 % self.n_slots as u64) as usize;
            let mut probes = 0usize;
            while self.batch_tag[slot] == self.batch && self.slot_map[slot] != id as i32 {
                slot = (slot + 1) % self.n_slots;
                probes += 1;
                assert!(probes < self.n_slots, "PLE row cache: batch larger than the cache");
            }
            if self.batch_tag[slot] != self.batch {
                self.batch_tag[slot] = self.batch;
                if self.slot_map[slot] != id as i32 {
                    self.slot_map[slot] = id as i32;
                    fill_rows.push((slot, id));
                }
            }
            slots[i] = slot as i32;
        }
        self.req += ngids.len() as u64;
        self.miss += fill_rows.len() as u64;
        if !fill_rows.is_empty() {
            let rows_per_shard = PLE_ROWS_PER_SHARD;
            // group by shard for sequential reads
            let mut by_shard: HashMap<i64, Vec<(usize, i64)>> = HashMap::new();
            for &(slot, id) in &fill_rows {
                let shard = id / rows_per_shard;
                by_shard.entry(shard).or_default().push((slot, id));
            }
            let mut shards: Vec<i64> = by_shard.keys().copied().collect();
            shards.sort_unstable();
            for shard in shards {
                let mut rows = by_shard.remove(&shard).unwrap();
                rows.sort_by_key(|&(_, id)| id);
                let (t, _) = &self.shards[shard as usize];
                let gs = t.global_scale;
                for &(slot, id) in &rows {
                    let row = (id % rows_per_shard) as u64;
                    let raw = cnq.read_range(t, row * 108, 108);
                    cuda::upload_into(self.cache + (slot * 108) as u64, &raw);
                    self.gs_host[slot] = gs;
                    cuda::to_f32_into(self.gs + (slot * 4) as u64, &[gs]);
                }
            }
        }
        slots
    }
}

// ---------------- launch plumbing ----------------

/// CROW_MMA=1 gate for the tensor-core FP4 paths (routed + dense GEMVs).
/// Read once — the env is fixed per process.
fn mma_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_MMA").as_deref() == Ok("1"))
}

/// CUDA graphs replay switch: capture the per-token kernel sequence once,
/// then replay (WDDM launch overhead elimination, fable gate 2026-09-03).
fn swap_bundle_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_SWAP_BUNDLE").as_deref() == Ok("1"))
}
fn graph_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        let on = std::env::var("CROW_GRAPH").as_deref() == Ok("1");
        if on && stage_dma_on() {
            println!("[stage] CROW_STAGE_DMA=1: decode graph disabled, host-issued staging");
            return false;
        }
        on
    })
}

/// dense-GEMV MMA switch: follows CROW_MMA unless CROW_MMA_DENSE=0 (A/B knob
/// for the dense #10 paths — routed MoE MMA stays on with CROW_MMA=1).
/// dense FP4 projection over t tokens: t >= 8 -> 8-token tile GEMM (weights
/// read once per 8 tokens), else the per-token MMA GEMV. `args` = the 7
/// gemv_fp4_mma_d arguments (w, xq, gs, y, k_dim, rows, y_stride).
pub unsafe fn launch_mma_d(k: &Kernels, gx: u32, t: usize, t_p: u64, args: &[u64]) {
    if t >= 8 && std::env::var("CROW_DENSE_GEMM").as_deref() != Ok("0") {
        let mut a: Vec<u64> = args.to_vec();
        a.push(t_p);
        launch_v(k.f("gemm_fp4_dense"), gx, ((t + 7) / 8) as u32, 1, mma_bx(), &a);
    } else {
        launch_v(k.f("gemv_fp4_mma_d"), gx, t as u32, 1, mma_bx(), args);
    }
}

/// MMA k-split factor (CROW_MMA_KS, default 4): block = 128 * KS threads;
/// KS warps-groups split the k-blocks and reduce deterministically in smem.
/// KS=1 is the original single-slice kernel (bit-identical).
pub fn mma_bx() -> u32 {
    static KS: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    128 * *KS.get_or_init(|| {
        std::env::var("CROW_MMA_KS").ok().and_then(|v| v.parse().ok()).filter(|&k| (1..=4).contains(&k)).unwrap_or(4)
    })
}

/// step-2 kernel switches (default on; =0 selects the previous kernel)
fn env_on(name: &str) -> bool {
    std::env::var(name).as_deref() != Ok("0")
}
/// CROW_TRICKLE_DEFER (default 1 = deferred; `0` selects the eager order, kept
/// as the fallback; any other value and unset take the default): the stream
/// trickle's copies are parked in `trickle_tick` and issued in `decode_step`
/// right after the graph launch, so they overlap the replay instead of blocking
/// it (63a: one async copy engine, the burst holds it, and the graph launch is
/// stream ordered behind the scalar refreshes that queue on that engine).
/// DEFAULT SINCE #63c, 2026-09-12: the deferred order ran the #59 profile arm at
/// 24.80 ms per decode token against 26.42 for the eager order, a gain of 1.62
/// ms per token at 16.4 x the baseline spread (decode_out/srv-63b.log, RTX 5090,
/// 2026-09-12). The switch moves the HOST issue order only, so both settings
/// must be byte-identical in parity.
fn trickle_defer_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_TRICKLE_DEFER").as_deref() != Ok("0"))
}
fn qsa_fast_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_QSA_FAST")) }
fn bf16_w_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_BF16_W")) }
fn inj_1k_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_INJ_1K")) }

/// CROW_QFUSE (default on): producers emit the NVFP4 activation cascade
/// themselves (no separate quant_x_fp4 launch). Bit-identical.
fn qfuse_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_QFUSE")) }
/// CROW_ATTN_SPLIT (default on): decode attention as S=8 partials + merge,
/// QSA scores warp-per-block (both graph-static; cost no longer grows
/// linearly with the context inside one block)
fn attn_split_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_ATTN_SPLIT")) }
/// #10 step 5 (2026-09-06): CROW_ATTN_R=4 selects attn_sel_s8l (attn_sel_s8 + shared e4m3 LUT), 5 selects attn_sel_g
/// (one block per (KV head, query), 12 warps = the 12 q heads sharing the rows; grid (NKV, Tq), 384 threads).
/// #10 step 4 (2026-09-06): CROW_ATTN_R=2 selects attn_sel_s (K/V rows staged through 16 KB shared memory per chunk,
/// same op chain as attn_sel_r -> bit-identical by parity), 3 selects attn_sel_s8 (8 KB chunks, higher occupancy).
/// #10 step 3 (2026-09-06): CROW_ATTN_R=1 selects attn_sel_r (q in registers, weights normalised once, V loop
/// unrolled x4; meant bit-identical, gate = parity); 8 / 9 are DIAGNOSTICS with wrong output (no K dot / no V loop).
/// Unset = attn_sel, byte-identical to before.
/// Default since 2026-09-06 late (#10, robin's call, gated by parity 8/512/1024 against the previous build and ten tasks):
/// unset = attn_sel_s8l (shared-memory staging + e4m3 LUT, 5.5 ms per call); CROW_ATTN_R=0 = the previous attn_sel.
fn attn_r_mode() -> i32 { static V: std::sync::OnceLock<i32> = std::sync::OnceLock::new(); *V.get_or_init(|| std::env::var("CROW_ATTN_R").ok().and_then(|v| v.parse().ok()).unwrap_or(-1)) }
fn attn_sel_name() -> &'static str { match attn_r_mode() { 0 => "attn_sel", 1 => "attn_sel_r", 2 => "attn_sel_s", 3 => "attn_sel_s8", 4 => "attn_sel_s8l", 5 => "attn_sel_g", 8 => "attn_sel_d8", 9 => "attn_sel_d9", _ => "attn_sel_s8l" } }
/// (grid.x, block) of the selected attention kernel: one block per q head (NQ, AHD) or per KV head (NKV, 384).
fn attn_sel_gx_bx() -> (u32, u32) { if attn_r_mode() == 5 { (NKV as u32, 384) } else { (NQ as u32, AHD as u32) } }
/// #16 (2026-09-05): prompt attention in sub-batches of ATTN_SB tokens. The QSA
/// score buffer [chunk][cap] f32 (256 KB per token, 512 MB at chunk 2048) shrinks
/// to [ATTN_SB][cap]; scores / select / attn_sel take pointer offsets per
/// sub-batch and a per-sub-batch pos_base scalar. Same arithmetic, same order
/// per token: bit-identical by construction. CROW_ATTN_SB=0 restores the
/// full-chunk buffer (one sub-batch).
pub const ATTN_SB: usize = 512;
fn attn_sb_on() -> bool { static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new(); *ON.get_or_init(|| env_on("CROW_ATTN_SB")) }
fn attn_sb(chunk: usize) -> usize { if attn_sb_on() { chunk.min(ATTN_SB).max(1) } else { chunk.max(1) } }
pub const ATTN_SPLITS: usize = 8;
/// #61a: the partial buffers are sized for the largest allowed split count,
/// so CROW_ATTN_SPLITS can be raised at runtime without a reallocation.
pub const ATTN_SPLITS_MAX: usize = 32;
pub const QSA_PAR_BLOCKS: u32 = 512;
/// #61a: bin count of the qsa_select_par round A histogram (12 top key bits);
/// must match QSA_PAR_BINS in kernels.rs.
pub const QSA_PAR_BINS: usize = 4096;
/// #61a CROW_ATTN_SPLITS (default 8, allowed 4 8 16 32, anything else falls
/// back to 8): the split count of the decode attention (grid.z of
/// attn_sel_split, the device scalar p.n_splits read by attn_merge).
/// MEASUREMENT ONLY: a different split count changes the merge order of the
/// flash-decoding partials, so the last bits of the logits may move.
pub fn attn_splits() -> usize {
    static V: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_ATTN_SPLITS").ok().and_then(|v| v.parse().ok())
        .filter(|s| matches!(s, 4 | 8 | 16 | 32)).unwrap_or(ATTN_SPLITS))
}
/// #61a CROW_QSA_PAR (default off, `1` selects it): the decode QSA top-k runs
/// as qsa_select_par_h (G blocks, 12-bit histogram) plus qsa_select_par_e
/// (one block of 1024, threshold refine and ascending emit) instead of
/// qsa_select_fast on one block. Same selection list in the same order:
/// bit-identical by parity.
fn qsa_par_on() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("CROW_QSA_PAR").as_deref() == Ok("1"))
}
/// #61a CROW_QSA_PAR_BLOCKS (default 32, clamped to 4 .. 256): the block count
/// of qsa_select_par_h. The histogram is order free, so the count never moves
/// a bit of the selection list.
fn qsa_par_blocks() -> u32 {
    static V: std::sync::OnceLock<u32> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("CROW_QSA_PAR_BLOCKS").ok().and_then(|v| v.parse().ok())
        .unwrap_or(32u32).clamp(4, 256))
}

fn dense_mma_on() -> bool {
    mma_on() && std::env::var("CROW_MMA_DENSE").as_deref() != Ok("0")
}

/// every kernel arg is a device address (scalars live in device buffers — the
/// p5 rule), so the arg list is just u64 values.
pub unsafe fn launch_sync(
    f: cudarc::driver::sys::CUfunction,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    vals: &[u64],
) {
    launch_v(f, gx, gy, gz, bx, vals);
    cuda::sync();
}

pub unsafe fn launch_v(
    f: cudarc::driver::sys::CUfunction,
    gx: u32,
    gy: u32,
    gz: u32,
    bx: u32,
    vals: &[u64],
) {
    use cudarc::driver::sys;
    use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
    static DBG: AtomicBool = AtomicBool::new(false);
    static INIT: AtomicBool = AtomicBool::new(false);
    static LAUNCH_N: AtomicU64 = AtomicU64::new(0);
    if !INIT.swap(true, Ordering::Relaxed) {
        DBG.store(std::env::var("ENGINE_DEBUG_SYNC").is_ok(), Ordering::Relaxed);
    }
    let mut ptrs: Vec<*mut std::ffi::c_void> = vals
        .iter()
        .map(|v| v as *const u64 as *mut std::ffi::c_void)
        .collect();
    let stream = cuda::cur_stream();
    let kprof = crate::kernels::KPROF_ON.load(Ordering::Relaxed);
    if kprof {
        cuda::sync();
    }
    let t_k = std::time::Instant::now();
    let r = sys::cuLaunchKernel(
        f, gx, gy, gz, bx, 1, 1, 0,
        stream,
        ptrs.as_mut_ptr(),
        std::ptr::null_mut(),
    );
    if kprof {
        cuda::sync();
        prof::kprof_add(format!("{}[{}x{}]", crate::kernels::last_name(), gx, gy), t_k.elapsed().as_micros() as u64);
    }
    if std::env::var("ENGINE_DEBUG_LAUNCH").is_ok() {
        eprintln!("[launch] stream={:p} gx={gx} gy={gy} r={:?}", stream as *mut std::ffi::c_void, r);
    }
    cuda::ck(r);
    if DBG.load(Ordering::Relaxed) {
        let n = LAUNCH_N.fetch_add(1, Ordering::Relaxed);
        if std::env::var("ENGINE_DEBUG_TRACE").as_deref() == Ok("1") {
            eprintln!(
                "[launch {n}] go gx={gx} gy={gy} gz={gz} bx={bx} a0={:x} a1={:x} a2={:x}",
                vals.get(0).copied().unwrap_or(0),
                vals.get(1).copied().unwrap_or(0),
                vals.get(2).copied().unwrap_or(0)
            );
        }
        cuda::sync();
    }
}

impl Params {
    pub unsafe fn setup(cfg: &Config, _st: &ThreeStates) -> Params {
        let cap_blocks = (cfg.context + 3) / 4;
        Params {
            n320: cuda::to_i32_dev(&[LOWRANK as i32]),
            n128: cuda::to_i32_dev(&[QSA_HD as i32]),
            n2560: cuda::to_i32_dev(&[H as i32]),
            n640: cuda::to_i32_dev(&[INTER as i32]),
            n1280: cuda::to_i32_dev(&[(2 * INTER) as i32]),
            n6144: cuda::to_i32_dev(&[GDN_VAL as i32]),
            n10240: cuda::to_i32_dev(&[HCT as i32]),
            n2048: cuda::to_i32_dev(&[GDN_KEY as i32]),
            n12288: cuda::to_i32_dev(&[Q_ROWS as i32]),
            n_conv_deq: cuda::to_i32_dev(&[(GDN_CONV * 4) as i32]),
            n_selmax: cuda::to_i32_dev(&[QSA_SEL_MAX as i32]),
            k_top: cuda::to_i32_dev(&[QSA_BLOCK_TOPK as i32]),
            cap: cuda::to_i32_dev(&[cap_blocks as i32]),
            tmax: cuda::to_i32_dev(&[cfg.context as i32]),
            mode: cuda::to_i32_dev(&[match cfg.kv { KvDtype::Fp8E4m3 => 0, KvDtype::Bf16 => 1 }]),
            t: cuda::to_i32_dev(&[1]),
            init: cuda::to_i32_dev(&[1]),
            pos_base: cuda::to_i32_dev(&[0]),
            pos_base_sb: cuda::to_i32_dev(&vec![0i32; (cfg.prompt_chunk.max(1) + ATTN_SB - 1) / ATTN_SB]),
            pos_base_b4: cuda::to_i32_dev(&[0]),
            slot_base: cuda::to_i32_dev(&[0]),
            ncb: cuda::to_i32_dev(&vec![0i32; cfg.prompt_chunk.max(1)]),
            pos_row: cuda::to_i32_dev(&vec![0i32; cfg.prompt_chunk.max(1)]),
            block_base: cuda::to_i32_dev(&[0]),
            n_new: cuda::to_i32_dev(&[0]),
            row_base: cuda::to_i32_dev(&[0]),
            n_rows: cuda::to_i32_dev(&[1]),
            slot1: cuda::to_i32_dev(&[0]),
            one: cuda::to_i32_dev(&[1]),
            zero: cuda::to_i32_dev(&[0]),
            n_vocab: cuda::to_i32_dev(&[V as i32]),
            nt_low: cuda::to_i32_dev(&[1]),
            nt_hct: cuda::to_i32_dev(&[1]),
            nt_hc: cuda::to_i32_dev(&[1]),
            nt_6144: cuda::to_i32_dev(&[1]),
            nt_combo: cuda::to_i32_dev(&[1]),
            q_heads4: cuda::to_i32_dev(&[QSA_HEADS as i32]),
            q_heads1: cuda::to_i32_dev(&[QSA_KVHEADS as i32]),
            pos_mul1: cuda::to_i32_dev(&[1]),
            pos_mul4: cuda::to_i32_dev(&[QSA_COMPRESS as i32]),
            stride512: cuda::to_i32_dev(&[(QSA_HEADS * QSA_HD) as i32]),
            stride128: cuda::to_i32_dev(&[QSA_HD as i32]),
            keys_ring: cuda::to_i32_dev(&[_st.qsa_ring_rows as i32]),
            qk_stride: cuda::to_i32_dev(&[QSA_QK_ROWS as i32]),
            ncb1: cuda::to_i32_dev(&[0]),
            pos_row1: cuda::to_i32_dev(&[0]),
            k_top10: cuda::to_i32_dev(&[TOPK as i32]),
            nt_hct1: cuda::to_i32_dev(&[HCT as i32]),
            nr4: cuda::to_i32_dev(&[HCN as i32]),
            nr48: cuda::to_i32_dev(&[GDN_VHEADS as i32]),
            nr512: cuda::to_i32_dev(&[KV_ROWS as i32]),
            n_splits: cuda::to_i32_dev(&[attn_splits() as i32]),
        }
    }
}

impl Scratch {
    pub unsafe fn alloc(chunk: usize) -> Scratch {
        let c = chunk;
        let cap_blocks = 65536usize; // pooled/score cap at the 262k ceiling
        let d = |n: usize| cuda::alloc_zeroed(n * 4);
        let db = |n: usize| cuda::alloc_zeroed(n);
        Scratch {
            h: d(c * HCT),
            emb: d(c * PLE_EMBED),
            mixed: d(c * H),
            low: d(c * LOWRANK),
            sil: d(c * LOWRANK),
            mixw: d(c * HCT),
            normed: d(c * HCT),
            injr: d(c * HCN),
            injw: d(c * HCN),
            x1: d(c * HCT),
            mixed_m: d(c * H),
            moe_out: d(c * H),
            h1: d(c * TOPK * 2 * INTER),
            h2: d(c * TOPK * INTER),
            eo: d(c * TOPK * H),
            xq_gu: db(c * 3 * (H / 64) * 36),
            xq_dn: db(c * TOPK * 3 * (INTER / 64) * 36),
            xq_m: db(c * 3 * (H / 64) * 36),
            xq_v: db(c * 3 * (GDN_VAL / 64) * 36),
            xq_s: db(c * 3 * (INTER / 64) * 36),
            xq_e: db(c * 3 * (H / 64) * 36),
            sdown: d(c * H),
            sh12: d(c * 2 * INTER),
            sh2: d(c * INTER),
            sgv: d(c),
            rlog: d(c * E),
            rids: db(c * TOPK * 4),
            rwts: d(c * TOPK),
            gu_ptrs: db(c * TOPK * 8),
            dn_ptrs: db(c * TOPK * 8),
            cold: db(c * 4),
            mq: d(c * GDN_CONV),
            mq_t: d(GDN_CONV * c),
            cout_t: d(GDN_CONV * c),
            gq: d(c * GDN_KEY),
            gk: d(c * GDN_KEY),
            gv: d(c * GDN_VAL),
            gz: d(c * GDN_VAL),
            gb: d(c * GDN_VHEADS),
            ga: d(c * GDN_VHEADS),
            gbeta: d(c * GDN_VHEADS),
            gg: d(c * GDN_VHEADS),
            gqr: d(c * GDN_VAL),
            gkr: d(c * GDN_VAL),
            gcore: d(c * GDN_VAL),
            gnorm: d(c * GDN_VAL),
            gout: d(c * H),
            qg: d(c * Q_ROWS),
            aq: d(c * CORE),
            agate: d(c * CORE),
            aqn: d(c * CORE),
            aqr: d(c * CORE),
            ak: d(c * KV_ROWS),
            akn: d(c * KV_ROWS),
            akr: d(c * KV_ROWS),
            av: d(c * KV_ROWS),
            aout: d(c * CORE),
            agated: d(c * CORE),
            ay: d(c * H),
            qk: d(c * QSA_QK_ROWS),
            q_nrm: d(c * QSA_HEADS * QSA_HD),
            q_rot: d(c * QSA_HEADS * QSA_HD),
            pool_raw: d(cap_blocks * QSA_HID),
            pool_nrm: d(cap_blocks * QSA_HID),
            pool_rot: d(cap_blocks * QSA_HID),
            scores: d(attn_sb(c) * cap_blocks), // #16: [ATTN_SB][cap] when sub-batched
            sel: db(c * QSA_SEL_MAX * 4),
            sel_n: db(c * 4),
            ple_key: d(c * HCT),
            ple_kn: d(c * HCT),
            ple_val: d(c * PLE_EMBED),
            ple_qn: d(c * HCT),
            ple_gate: d(c * HCN),
            ple_gs: d(c * HCN),
            ple_gated: d(c * HCT),
            ple_gn: d(c * HCT),
            ple_out: d(c * HCT),
            ple_slots: db(c * PLE_NHEADS * 4),
            part_o: d(NQ * ATTN_SPLITS_MAX * AHD),  // #61a: sized for CROW_ATTN_SPLITS=32
            part_ml: d(NQ * ATTN_SPLITS_MAX * 2),   // #61a: sized for CROW_ATTN_SPLITS=32
            qsa_h1: db(QSA_PAR_BINS * 4),           // #61a: CROW_QSA_PAR histogram, zero between calls
            logits: d(V), // one row: lm_head_row always writes the base row (was c*V = 508 MB at C=512)
            argmax: db(4),
            mixed_final: d(c * H),
            scratch_gn: d(1),
        }
    }
}

pub const QSA_HID: usize = QSA_HIDD;

impl Engine {
    // (kc, vc, qsa_keys, qsa_pooled) device addresses of a layer's caches
    fn layer_cache_ptrs(&self, layer: usize) -> (u64, u64, u64, u64) {
        let ai = attn_index(layer);
        let bpv = self.st.kv.byte_per_value() as u64;
        let per_layer = (NKV * self.st.context * AHD) as u64 * bpv;
        // kv_buf holds ATTN_LAYERS layer slots — index by ATTENTION index,
        // not by the raw layer id (36 GDN layers shift the numbering!)
        let kc = self.st.kv_buf as u64 + (ai * 2) as u64 * per_layer;
        let vc = kc + per_layer;
        (kc, vc, self.st.qsa_keys[ai] as u64, self.st.qsa_pooled[ai] as u64)
    }

    /// GatedResidual block: norm → down → silu/4 → up → sigmoid → mix + inject
    unsafe fn hc_run(&self, w: &HcW, x: Dev, t: usize, mixed: Dev, injw: Dev) {
        // the mixed row is quantized in the same launch when it feeds FP4
        // projections: attn-HC -> xq_m (GDN qkv/z/b/a, attention v/indexer),
        // mlp-HC -> xq_gu (routed gate_up + shared gate/up)
        let xq = if mixed == self.s.mixed { self.s.xq_m } else if mixed == self.s.mixed_m { self.s.xq_gu } else { 0 };
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        launch_v(k.f("rms_group"), 4, t as u32, 1, 256, &[x as u64, w.norm as u64, s.normed as u64]);
        w.down.launch_gemv(k, LOWRANK, p.n320 as u64, t, p.t as u64, s.normed as u64, s.low as u64, p.n10240 as u64);
        launch_v(k.f("silu_div4"), ((t * LOWRANK + 255) / 256) as u32, 1, 1, 256, &[
            s.low as u64, s.sil as u64, p.nt_low as u64]);
        w.up.launch_gemv(k, HCT, p.n10240 as u64, t, p.t as u64, s.sil as u64, s.mixw as u64, p.n320 as u64);
        launch_v(k.f("sigmoid_el"), ((t * HCT + 255) / 256) as u32, 1, 1, 256, &[
            s.mixw as u64, p.nt_hct as u64]);
        if qfuse_on() && xq != 0 {
            launch_v(k.f("mix_streams_q"), 10, t as u32, 1, 256, &[
                s.mixw as u64, s.normed as u64, mixed as u64, xq as u64]);
        } else {
            launch_v(k.f("mix_streams"), 10, t as u32, 1, 256, &[
                s.mixw as u64, s.normed as u64, mixed as u64]);
        }
        // block-inject stays on the naive GEMV by measurement (mma_gate dense):
        // rows=4 saturates the naive kernel (4×256 threads) while the MMA tile
        // is a single active warp on one SM — mma_d is ~10x SLOWER at t=1 and
        // still behind at t=512. Documented skip, quality/perf over uniformity.
        if inj_1k_on() {
            launch_v(k.f("gemv_fp4_b1k"), HCN as u32, t as u32, 1, 1024, &[
                w.inj.w as u64, s.normed as u64, w.inj.gs as u64, s.injr as u64, p.n10240 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), HCN as u32, t as u32, 1, 256, &[
                w.inj.w as u64, s.normed as u64, w.inj.gs as u64, s.injr as u64, p.n10240 as u64]);
        }
        launch_v(k.f("sig2_div4"), ((t * HCN + 255) / 256) as u32, 1, 1, 256, &[
            s.injr as u64, injw as u64, p.nt_hc as u64]);
    }

    unsafe fn gdn_prompt(&self, l: usize, mixed: Dev, t: usize, _first: bool) -> Dev {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let gi = gdn_index(l);
        let SubW::Gdn { qkv, conv, z, b, a, alog, dt, norm, out } = &self.w.sub[l] else {
            panic!("layer {l} is not GDN");
        };
        if dense_mma_on() {
            // one quantized `mixed` row set serves qkv/z/b/a (all k=2560)
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                mixed as u64, s.xq_m as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]); }
            launch_mma_d(k, (GDN_CONV / 64) as u32, t, p.t as u64, &[
                qkv.w as u64, s.xq_m as u64, qkv.gs as u64, s.mq as u64,
                p.n2560 as u64, p.n10240 as u64, p.n10240 as u64]);
            launch_mma_d(k, (GDN_VAL / 64) as u32, t, p.t as u64, &[
                z.w as u64, s.xq_m as u64, z.gs as u64, s.gz as u64,
                p.n2560 as u64, p.n6144 as u64, p.n6144 as u64]);
            launch_mma_d(k, 1, t, p.t as u64, &[
                b.w as u64, s.xq_m as u64, b.gs as u64, s.gb as u64,
                p.n2560 as u64, p.nr48 as u64, p.nr48 as u64]);
            launch_mma_d(k, 1, t, p.t as u64, &[
                a.w as u64, s.xq_m as u64, a.gs as u64, s.ga as u64,
                p.n2560 as u64, p.nr48 as u64, p.nr48 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), GDN_CONV as u32, t as u32, 1, 256, &[
                qkv.w as u64, mixed as u64, qkv.gs as u64, s.mq as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4_b"), GDN_VAL as u32, t as u32, 1, 256, &[
                z.w as u64, mixed as u64, z.gs as u64, s.gz as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4_b"), 48, t as u32, 1, 256, &[
                b.w as u64, mixed as u64, b.gs as u64, s.gb as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4_b"), 48, t as u32, 1, 256, &[
                a.w as u64, mixed as u64, a.gs as u64, s.ga as u64, p.n2560 as u64]);
        }
        launch_v(k.f("transpose_rt"), GDN_CONV as u32, 1, 1, 256, &[
            s.mq as u64, s.mq_t as u64, p.t as u64, p.n10240 as u64]);
        launch_v(k.f("conv_silu"), GDN_CONV as u32, 1, 1, 256, &[
            s.mq_t as u64, *conv as u64, s.cout_t as u64, p.t as u64, self.st.gdn_conv[gi] as u64]);
        launch_v(k.f("conv_state_update"), GDN_CONV as u32, 1, 1, 3, &[
            s.mq_t as u64, self.st.gdn_conv[gi] as u64, p.t as u64]);
        launch_v(k.f("split_qkv"), ((t * GDN_CONV + 255) / 256) as u32, 1, 1, 256, &[
            s.cout_t as u64, s.gq as u64, s.gk as u64, s.gv as u64, p.t as u64]);
        launch_v(k.f("beta_g"), ((t * 48 + 255) / 256) as u32, 1, 1, 256, &[
            s.gb as u64, s.ga as u64, *alog as u64, *dt as u64, s.gbeta as u64, s.gg as u64, p.t as u64]);
launch_v(k.f("l2norm_repeat"), 48, t as u32, 1, 128, &[
            s.gq as u64, s.gk as u64, s.gqr as u64, s.gkr as u64]);
        launch_v(k.f(if gdn_reg_mode() & 1 != 0 { "delta_rule_persist_r" } else { "delta_rule_persist" }), 48, 1, 1, 128, &[
            s.gqr as u64, s.gkr as u64, s.gv as u64, s.gg as u64, s.gbeta as u64,
            s.gcore as u64, self.st.gdn_s[gi] as u64, p.t as u64, p.init as u64]);
        if qfuse_on() {
            launch_v(k.f("rmsnorm_gated_q"), 48, t as u32, 1, 128, &[
                s.gcore as u64, s.gz as u64, *norm as u64, s.gnorm as u64, s.xq_v as u64]);
        } else {
            launch_v(k.f("rmsnorm_gated"), 48, t as u32, 1, 128, &[
                s.gcore as u64, s.gz as u64, *norm as u64, s.gnorm as u64]);
        }
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                s.gnorm as u64, s.xq_v as u64, p.n6144 as u64, p.one as u64, p.n6144 as u64]); }
            launch_mma_d(k, (H / 64) as u32, t, p.t as u64, &[
                out.w as u64, s.xq_v as u64, out.gs as u64, s.gout as u64,
                p.n6144 as u64, p.n2560 as u64, p.n2560 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), H as u32, t as u32, 1, 256, &[
                out.w as u64, s.gnorm as u64, out.gs as u64, s.gout as u64, p.n6144 as u64]);
        }
        s.gout
    }

    unsafe fn gdn_step(&self, l: usize, mixed: Dev) -> Dev {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let gi = gdn_index(l);
        let SubW::Gdn { qkv, conv, z, b, a, alog, dt, norm, out } = &self.w.sub[l] else {
            panic!("layer {l} is not GDN");
        };
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), 1, 1, 1, 128, &[
                mixed as u64, s.xq_m as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]); }
            launch_v(k.f("gemv_fp4_mma_d"), (GDN_CONV / 64) as u32, 1, 1, mma_bx(), &[
                qkv.w as u64, s.xq_m as u64, qkv.gs as u64, s.mq as u64,
                p.n2560 as u64, p.n10240 as u64, p.n10240 as u64]);
            launch_v(k.f("gemv_fp4_mma_d"), (GDN_VAL / 64) as u32, 1, 1, mma_bx(), &[
                z.w as u64, s.xq_m as u64, z.gs as u64, s.gz as u64,
                p.n2560 as u64, p.n6144 as u64, p.n6144 as u64]);
            launch_v(k.f("gemv_fp4_mma_d"), 1, 1, 1, mma_bx(), &[
                b.w as u64, s.xq_m as u64, b.gs as u64, s.gb as u64,
                p.n2560 as u64, p.nr48 as u64, p.nr48 as u64]);
            launch_v(k.f("gemv_fp4_mma_d"), 1, 1, 1, mma_bx(), &[
                a.w as u64, s.xq_m as u64, a.gs as u64, s.ga as u64,
                p.n2560 as u64, p.nr48 as u64, p.nr48 as u64]);
        } else {
            launch_v(k.f("gemv_fp4"), GDN_CONV as u32, 1, 1, 256, &[
                qkv.w as u64, mixed as u64, qkv.gs as u64, s.mq as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4"), GDN_VAL as u32, 1, 1, 256, &[
                z.w as u64, mixed as u64, z.gs as u64, s.gz as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4"), 48, 1, 1, 256, &[
                b.w as u64, mixed as u64, b.gs as u64, s.gb as u64, p.n2560 as u64]);
            launch_v(k.f("gemv_fp4"), 48, 1, 1, 256, &[
                a.w as u64, mixed as u64, a.gs as u64, s.ga as u64, p.n2560 as u64]);
        }
        launch_v(k.f("conv_step"), (GDN_CONV as u32 + 255) / 256, 1, 1, 256, &[
            s.mq as u64, *conv as u64, self.st.gdn_conv[gi] as u64, s.cout_t as u64]);
        // cout layout [10240] = q | k | v (pointer slices, no copy)
        let (q_p, k_p, v_p) = (s.cout_t as u64, s.cout_t as u64 + (GDN_KEY * 4) as u64, s.cout_t as u64 + (2 * GDN_KEY * 4) as u64);
        launch_v(k.f("beta_g"), 1, 1, 1, 48, &[
            s.gb as u64, s.ga as u64, *alog as u64, *dt as u64, s.gbeta as u64, s.gg as u64, p.t as u64]);
        launch_v(k.f("l2norm_repeat"), 48, 1, 1, 128, &[q_p, k_p, s.gqr as u64, s.gkr as u64]);
        launch_v(k.f(if gdn_reg_mode() & 2 != 0 { "delta_rule_step_r" } else { "delta_rule_step" }), 48, 1, 1, 128, &[
            self.st.gdn_s[gi] as u64, s.gqr as u64, s.gkr as u64, v_p, s.gg as u64,
            s.gbeta as u64, s.gcore as u64]);
        if qfuse_on() {
            launch_v(k.f("rmsnorm_gated_q"), 48, 1, 1, 128, &[
                s.gcore as u64, s.gz as u64, *norm as u64, s.gnorm as u64, s.xq_v as u64]);
        } else {
            launch_v(k.f("rmsnorm_gated"), 48, 1, 1, 128, &[
                s.gcore as u64, s.gz as u64, *norm as u64, s.gnorm as u64]);
        }
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), 1, 1, 1, 128, &[
                s.gnorm as u64, s.xq_v as u64, p.n6144 as u64, p.one as u64, p.n6144 as u64]); }
            launch_v(k.f("gemv_fp4_mma_d"), (H / 64) as u32, 1, 1, mma_bx(), &[
                out.w as u64, s.xq_v as u64, out.gs as u64, s.gout as u64,
                p.n6144 as u64, p.n2560 as u64, p.n2560 as u64]);
        } else {
            launch_v(k.f("gemv_fp4"), H as u32, 1, 1, 256, &[
                out.w as u64, s.gnorm as u64, out.gs as u64, s.gout as u64, p.n6144 as u64]);
        }
        s.gout
    }
}

impl Engine {
    unsafe fn attn_prompt(&self, l: usize, mixed: Dev, t: usize, pos_base: usize) -> Dev {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let dbg = std::env::var("ENGINE_DEBUG_STEP").is_ok();
        macro_rules! step {
            ($m:expr) => {{
                if dbg {
                    cuda::sync();
                    eprintln!("[attn {}] {}", l, $m);
                }
            }};
        }
        let SubW::Attn { q, k: kk, v, o, qn, kn, iqk, iqln, ikln } = &self.w.sub[l] else {
            panic!("layer {l} is not attention");
        };
        let (kc, vc, keys, pooled) = self.layer_cache_ptrs(l);
        let cos = self.st.cos as u64 + (pos_base * ROPE_PAIRS * 4) as u64;
        let sin = self.st.sin as u64 + (pos_base * ROPE_PAIRS * 4) as u64;

        q.launch_gemv(k, Q_ROWS, p.n12288 as u64, t, p.t as u64, mixed as u64, s.qg as u64, p.n2560 as u64);
        step!("launch #1");
        step!("q gemv");
        launch_v(k.f("split_qg"), NQ as u32, t as u32, 1, AHD as u32, &[
            s.qg as u64, s.aq as u64, s.agate as u64]);
        step!("launch #2");
        step!("split_qg");
        launch_v(k.f("rmsnorm_1pw"), NQ as u32, t as u32, 1, AHD as u32, &[
            s.aq as u64, *qn as u64, s.aqn as u64]);
        step!("launch #3");
        launch_v(k.f("rope"), NQ as u32, t as u32, 1, AHD as u32, &[
            s.aqn as u64, cos, sin, s.aqr as u64]);
        step!("launch #4");
        kk.launch_gemv(k, KV_ROWS, p.nr512 as u64, t, p.t as u64, mixed as u64, s.ak as u64, p.n2560 as u64);
        step!("launch #5");
        launch_v(k.f("rmsnorm_1pw"), NKV as u32, t as u32, 1, AHD as u32, &[
            s.ak as u64, *kn as u64, s.akn as u64]);
        step!("launch #6");
        launch_v(k.f("rope"), NKV as u32, t as u32, 1, AHD as u32, &[
            s.akn as u64, cos, sin, s.akr as u64]);
        step!("launch #7");
        if dense_mma_on() {
            // one quantized `mixed` row set serves v + indexer qk (both k=2560)
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                mixed as u64, s.xq_m as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]); }
            launch_mma_d(k, (KV_ROWS / 64) as u32, t, p.t as u64, &[
                v.w as u64, s.xq_m as u64, v.gs as u64, s.av as u64,
                p.n2560 as u64, p.nr512 as u64, p.nr512 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), KV_ROWS as u32, t as u32, 1, 256, &[
                v.w as u64, mixed as u64, v.gs as u64, s.av as u64, p.n2560 as u64]);
        }
        step!("launch #8");
        launch_v(k.f("store_kv"), 4, t as u32, 1, AHD as u32, &[
            s.akr as u64, s.av as u64, kc, vc, p.pos_base as u64, p.tmax as u64, p.mode as u64]);
        step!("launch #9");

        // ---- QSA indexer (raw keys append + block pooling + scores + select) ----
        if dense_mma_on() {
            launch_mma_d(k, (QSA_QK_ROWS / 64) as u32, t, p.t as u64, &[
                iqk.w as u64, s.xq_m as u64, iqk.gs as u64, s.qk as u64,
                p.n2560 as u64, p.n640 as u64, p.n640 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), QSA_QK_ROWS as u32, t as u32, 1, 256, &[
                iqk.w as u64, mixed as u64, iqk.gs as u64, s.qk as u64, p.n2560 as u64]);
        }
        step!("launch #10");
        launch_v(k.f("rms128"), QSA_HEADS as u32, t as u32, 1, QSA_HD as u32, &[
            s.qk as u64, *iqln as u64, s.q_nrm as u64, p.q_heads4 as u64, p.qk_stride as u64]);
        step!("launch #11");
        launch_v(k.f("rope64"), QSA_HEADS as u32, t as u32, 1, QSA_HD as u32, &[
            s.q_nrm as u64, self.st.cos as u64, self.st.sin as u64, s.q_rot as u64,
            p.q_heads4 as u64, p.pos_mul1 as u64, p.stride512 as u64, p.pos_base as u64]);
        step!("launch #12");
        launch_v(k.f("qk_k_append"), ((t * QSA_HD + 255) / 256) as u32, 1, 1, 256, &[
            s.qk as u64, keys, p.pos_base as u64, p.t as u64, p.keys_ring as u64]);
        step!("launch #13");
        let new_done = (pos_base + t) / 4;
        let n_new = new_done - self.done_blocks;
        if n_new > 0 {
            cuda::to_i32_into(p.block_base, &[self.done_blocks as i32]);
            cuda::to_i32_into(p.n_new, &[n_new as i32]);
            launch_v(k.f("pool4_cache"), n_new as u32, 1, 1, QSA_HD as u32, &[
                keys, s.pool_raw as u64, p.stride128 as u64, p.block_base as u64, p.n_new as u64, p.keys_ring as u64]);
        step!("launch #14");
            launch_v(k.f("rms128"), 1, n_new as u32, 1, QSA_HD as u32, &[
                s.pool_raw as u64, *ikln as u64, s.pool_nrm as u64, p.q_heads1 as u64, p.stride128 as u64]);
        step!("launch #15");
            launch_v(k.f("rope64"), 1, n_new as u32, 1, QSA_HD as u32, &[
                s.pool_nrm as u64, self.st.cos as u64, self.st.sin as u64,
                pooled + (self.done_blocks * QSA_HD * 4) as u64,
                p.q_heads1 as u64, p.pos_mul4 as u64, p.stride128 as u64, p.pos_base_b4 as u64]);
        step!("launch #16");
        }
        // #16: scores / select / attn_sel per sub-batch of ATTN_SB tokens (the score
        // buffer is [ATTN_SB][cap]); every other buffer is indexed by the token
        // row, so a base-pointer offset per sub-batch is the whole change.
        let sb = attn_sb(self.cfg.prompt_chunk);
        let n_sb = (t + sb - 1) / sb;
        let pb: Vec<i32> = (0..n_sb).map(|i| (pos_base + i * sb) as i32).collect();
        cuda::to_i32_into(p.pos_base_sb, &pb);
        for i in 0..n_sb {
            let t0 = i * sb;
            let tb = (t - t0).min(sb);
            let pos_base_i = p.pos_base_sb as u64 + (i * 4) as u64;
            let q_rot_i = s.q_rot as u64 + (t0 * QSA_HEADS * QSA_HD * 4) as u64;
            let sel_i = s.sel as u64 + (t0 * QSA_SEL_MAX * 4) as u64;
            let sel_n_i = s.sel_n as u64 + (t0 * 4) as u64;
            if attn_split_on() {
                let ncb_max = ((pos_base + t0 + tb + 3) / 4).min(65536);
                launch_v(k.f("qsa_scores_par"), (((ncb_max + 3) / 4).clamp(1, 512)) as u32, tb as u32, 1, 128, &[
                    q_rot_i, pooled, s.scores as u64, p.cap as u64, pos_base_i]);
            } else {
                launch_v(k.f("qsa_scores"), tb as u32, 1, 1, QSA_HD as u32, &[
                    q_rot_i, pooled, s.scores as u64, p.cap as u64, pos_base_i]);
            }
            step!("launch #17");
            launch_v(k.f(if qsa_fast_on() { "qsa_select_fast" } else { "qsa_select" }), tb as u32, 1, 1, 256, &[
                s.scores as u64, p.ncb as u64 + (t0 * 4) as u64, sel_i, sel_n_i, p.k_top as u64,
                p.cap as u64, p.n_selmax as u64, p.pos_row as u64 + (t0 * 4) as u64]);
            step!("launch #18");
            launch_v(k.f(attn_sel_name()), attn_sel_gx_bx().0, tb as u32, 1, attn_sel_gx_bx().1, &[
                s.aqr as u64 + (t0 * CORE * 4) as u64, kc, vc, sel_i, sel_n_i, p.tmax as u64, p.mode as u64,
                p.n_selmax as u64, s.aout as u64 + (t0 * CORE * 4) as u64]);
        }

        if dbg {
            cuda::sync();
            let sn = cuda::dtoh_i32(s.sel_n, t.min(8));
            let sl = cuda::dtoh_i32(s.sel, 12);
            eprintln!("[attn {l}] sel_n[0..{}]={:?} sel[0..12]={:?}", sn.len(), sn, sl);
        }
        step!("launch #19");        step!("launch #19");
        if qfuse_on() {
            launch_v(k.f("gate_mul_q"), ((t * CORE + 255) / 256) as u32, 1, 1, 256, &[
                s.aout as u64, s.agate as u64, s.agated as u64, s.xq_v as u64]);
        } else {
            launch_v(k.f("gate_mul"), ((t * CORE + 255) / 256) as u32, 1, 1, 256, &[
                s.aout as u64, s.agate as u64, s.agated as u64]);
        }
        step!("launch #20");
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                s.agated as u64, s.xq_v as u64, p.n6144 as u64, p.one as u64, p.n6144 as u64]); }
            launch_mma_d(k, (H / 64) as u32, t, p.t as u64, &[
                o.w as u64, s.xq_v as u64, o.gs as u64, s.ay as u64,
                p.n6144 as u64, p.n2560 as u64, p.n2560 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), H as u32, t as u32, 1, 256, &[
                o.w as u64, s.agated as u64, o.gs as u64, s.ay as u64, p.n6144 as u64]);
        }
        step!("launch #21");
        s.ay
    }

    unsafe fn attn_step(&self, l: usize, mixed: Dev, pos: usize, sb: &[i32; 2], graph: bool) -> Dev {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let SubW::Attn { q, k: kk, v, o, qn, kn, iqk, iqln, ikln } = &self.w.sub[l] else {
            panic!("layer {l} is not attention");
        };
        let (kc, vc, keys, pooled) = self.layer_cache_ptrs(l);
        // rope reads its table row from the DEVICE scalar p.pos_base (refreshed
        // per token) - a host-computed pointer offset would be baked into a
        // captured graph and replay the capture token position forever
        let (cos, sin) = (self.st.cos as u64, self.st.sin as u64);

        q.launch_gemv1(k, Q_ROWS, p.n12288 as u64, mixed as u64, s.qg as u64, p.n2560 as u64);
        launch_v(k.f("split_qg"), NQ as u32, 1, 1, AHD as u32, &[s.qg as u64, s.aq as u64, s.agate as u64]);
        launch_v(k.f("rmsnorm_1pw"), NQ as u32, 1, 1, AHD as u32, &[s.aq as u64, *qn as u64, s.aqn as u64]);
        launch_v(k.f("rope_p"), NQ as u32, 1, 1, AHD as u32, &[s.aqn as u64, cos, sin, s.aqr as u64, p.pos_base as u64]);
        kk.launch_gemv1(k, KV_ROWS, p.nr512 as u64, mixed as u64, s.ak as u64, p.n2560 as u64);
        launch_v(k.f("rmsnorm_1pw"), NKV as u32, 1, 1, AHD as u32, &[s.ak as u64, *kn as u64, s.akn as u64]);
        launch_v(k.f("rope_p"), NKV as u32, 1, 1, AHD as u32, &[s.akn as u64, cos, sin, s.akr as u64, p.pos_base as u64]);
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), 1, 1, 1, 128, &[
                mixed as u64, s.xq_m as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]); }
            launch_v(k.f("gemv_fp4_mma_d"), (KV_ROWS / 64) as u32, 1, 1, mma_bx(), &[
                v.w as u64, s.xq_m as u64, v.gs as u64, s.av as u64,
                p.n2560 as u64, p.nr512 as u64, p.nr512 as u64]);
        } else {
            launch_v(k.f("gemv_fp4"), KV_ROWS as u32, 1, 1, 256, &[
                v.w as u64, mixed as u64, v.gs as u64, s.av as u64, p.n2560 as u64]);
        }
        launch_v(k.f("store_kv"), 4, 1, 1, AHD as u32, &[
            s.akr as u64, s.av as u64, kc, vc, p.slot1 as u64, p.tmax as u64, p.mode as u64]);

        if dense_mma_on() {
            launch_v(k.f("gemv_fp4_mma_d"), (QSA_QK_ROWS / 64) as u32, 1, 1, mma_bx(), &[
                iqk.w as u64, s.xq_m as u64, iqk.gs as u64, s.qk as u64,
                p.n2560 as u64, p.n640 as u64, p.n640 as u64]);
        } else {
            launch_v(k.f("gemv_fp4"), QSA_QK_ROWS as u32, 1, 1, 256, &[
                iqk.w as u64, mixed as u64, iqk.gs as u64, s.qk as u64, p.n2560 as u64]);
        }
        launch_v(k.f("rms128"), QSA_HEADS as u32, 1, 1, QSA_HD as u32, &[
            s.qk as u64, *iqln as u64, s.q_nrm as u64, p.q_heads4 as u64, p.qk_stride as u64]);
        launch_v(k.f("rope64"), QSA_HEADS as u32, 1, 1, QSA_HD as u32, &[
            s.q_nrm as u64, self.st.cos as u64, self.st.sin as u64, s.q_rot as u64,
            p.q_heads4 as u64, p.pos_mul1 as u64, p.stride512 as u64, p.pos_base as u64]);
        launch_v(k.f("qk_k_append"), 1, 1, 1, QSA_HD as u32, &[
            s.qk as u64, keys, p.pos_base as u64, p.one as u64, p.keys_ring as u64]);
        if graph {
            // graph mode: static launch sequence. n_new may be 0 — grids stay
            // at 1 and pool4_cache guards on n_blocks; garbage lands beyond
            // done_blocks, which is never selected. The staging slots (sb) are
            // stable host storage — the graph re-reads their current values on
            // every replay. rope64 writes the staging buffer; the d2d copy
            // takes the dynamic offset from p.block_base (device, fresh).
            // p.block_base / p.n_new were refreshed per token from PINNED
            // staging BEFORE the graph launch - no HtoD inside the captured
            // region (pageable memcpy may sync the stream = capture-illegal)
            let nn1 = sb[1].max(0).max(1) as u32;
            launch_v(k.f("pool4_cache"), nn1, 1, 1, QSA_HD as u32, &[
                keys, s.pool_raw as u64, p.stride128 as u64, p.block_base as u64, p.n_new as u64, p.keys_ring as u64]);
            launch_v(k.f("rms128"), 1, nn1, 1, QSA_HD as u32, &[
                s.pool_raw as u64, *ikln as u64, s.pool_nrm as u64, p.q_heads1 as u64, p.stride128 as u64]);
            launch_v(k.f("rope64"), 1, nn1, 1, QSA_HD as u32, &[
                s.pool_nrm as u64, self.st.cos as u64, self.st.sin as u64, s.pool_nrm as u64,
                p.q_heads1 as u64, p.pos_mul4 as u64, p.stride128 as u64, p.pos_base_b4 as u64]);
            launch_v(k.f("d2d_block"), 1, 1, 1, 128, &[
                pooled as u64, s.pool_nrm as u64, p.block_base as u64, p.n128 as u64]);
        } else if (pos + 1) % 4 == 0 {
            let bb = (pos + 1) / 4 - 1;
            cuda::to_i32_into(p.block_base, &[bb as i32]);
            cuda::to_i32_into(p.n_new, &[1]);
            launch_v(k.f("pool4_cache"), 1, 1, 1, QSA_HD as u32, &[
                keys, s.pool_raw as u64, p.stride128 as u64, p.block_base as u64, p.one as u64, p.keys_ring as u64]);
            launch_v(k.f("rms128"), 1, 1, 1, QSA_HD as u32, &[
                s.pool_raw as u64, *ikln as u64, s.pool_nrm as u64, p.q_heads1 as u64, p.stride128 as u64]);
            launch_v(k.f("rope64"), 1, 1, 1, QSA_HD as u32, &[
                s.pool_nrm as u64, self.st.cos as u64, self.st.sin as u64,
                pooled + (bb * QSA_HD * 4) as u64,
                p.q_heads1 as u64, p.pos_mul4 as u64, p.stride128 as u64, p.pos_base_b4 as u64]);
        }
        if attn_split_on() {
            launch_v(k.f("qsa_scores_par"), QSA_PAR_BLOCKS, 1, 1, 128, &[
                s.q_rot as u64, pooled, s.scores as u64, p.cap as u64, p.pos_base as u64]);
        } else {
            launch_v(k.f("qsa_scores"), 1, 1, 1, QSA_HD as u32, &[
                s.q_rot as u64, pooled, s.scores as u64, p.cap as u64, p.pos_base as u64]);
        }
        if qsa_par_on() {
            // #61a: two launches, the histogram over G blocks then one emit block
            launch_v(k.f("qsa_select_par_h"), qsa_par_blocks(), 1, 1, 256, &[
                s.scores as u64, p.ncb1 as u64, s.qsa_h1 as u64, p.k_top as u64, p.cap as u64]);
            launch_v(k.f("qsa_select_par_e"), 1, 1, 1, 1024, &[
                s.scores as u64, p.ncb1 as u64, s.sel as u64, s.sel_n as u64, p.k_top as u64,
                p.cap as u64, p.n_selmax as u64, p.pos_row1 as u64, s.qsa_h1 as u64]);
        } else {
        launch_v(k.f(if qsa_fast_on() { "qsa_select_fast" } else { "qsa_select" }), 1, 1, 1, 256, &[
            s.scores as u64, p.ncb1 as u64, s.sel as u64, s.sel_n as u64, p.k_top as u64,
            p.cap as u64, p.n_selmax as u64, p.pos_row1 as u64]);
        }
        if attn_split_on() {
            launch_v(k.f("attn_sel_split"), NQ as u32, 1, attn_splits() as u32, AHD as u32, &[
                s.aqr as u64, kc, vc, s.sel as u64, s.sel_n as u64, p.tmax as u64, p.mode as u64,
                p.n_selmax as u64, s.part_o as u64, s.part_ml as u64]);
            launch_v(k.f("attn_merge"), NQ as u32, 1, 1, AHD as u32, &[
                s.part_o as u64, s.part_ml as u64, s.aout as u64, p.n_splits as u64]);
        } else {
            launch_v(k.f(attn_sel_name()), attn_sel_gx_bx().0, 1, 1, attn_sel_gx_bx().1, &[
                s.aqr as u64, kc, vc, s.sel as u64, s.sel_n as u64, p.tmax as u64, p.mode as u64,
                p.n_selmax as u64, s.aout as u64]);
        }
        if qfuse_on() {
            launch_v(k.f("gate_mul_q"), (CORE as u32 + 255) / 256, 1, 1, 256, &[
                s.aout as u64, s.agate as u64, s.agated as u64, s.xq_v as u64]);
        } else {
            launch_v(k.f("gate_mul"), (CORE as u32 + 255) / 256, 1, 1, 256, &[
                s.aout as u64, s.agate as u64, s.agated as u64]);
        }
        if dense_mma_on() {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), 1, 1, 1, 128, &[
                s.agated as u64, s.xq_v as u64, p.n6144 as u64, p.one as u64, p.n6144 as u64]); }
            launch_v(k.f("gemv_fp4_mma_d"), (H / 64) as u32, 1, 1, mma_bx(), &[
                o.w as u64, s.xq_v as u64, o.gs as u64, s.ay as u64,
                p.n6144 as u64, p.n2560 as u64, p.n2560 as u64]);
        } else {
            launch_v(k.f("gemv_fp4"), H as u32, 1, 1, 256, &[
                o.w as u64, s.agated as u64, o.gs as u64, s.ay as u64, p.n6144 as u64]);
        }
        s.ay
    }
}

/// CROW_STAGE_DMA counters (cumulative, reset by stage_dma_reset before the timed window)
static DMA_SYNCS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DMA_COPIES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static DMA_BYTES: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// host nanoseconds inside the per-layer read-back (DtoH issue plus the one sync)
static DMA_NS_READ: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// host nanoseconds inside the per-layer copy issue loop plus the pointer-table HtoD
static DMA_NS_ISSUE: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

pub fn stage_dma_reset() {
    use std::sync::atomic::Ordering;
    DMA_SYNCS.store(0, Ordering::Relaxed);
    DMA_COPIES.store(0, Ordering::Relaxed);
    DMA_BYTES.store(0, Ordering::Relaxed);
    DMA_NS_READ.store(0, Ordering::Relaxed);
    DMA_NS_ISSUE.store(0, Ordering::Relaxed);
}

/// one line with the host cost of the DMA staging path; silent when it never ran
pub fn stage_dma_report(tokens: u64) {
    use std::sync::atomic::Ordering;
    let sy = DMA_SYNCS.load(Ordering::Relaxed);
    if sy == 0 { return; }
    let co = DMA_COPIES.load(Ordering::Relaxed);
    let by = DMA_BYTES.load(Ordering::Relaxed);
    let rd = DMA_NS_READ.load(Ordering::Relaxed);
    let is = DMA_NS_ISSUE.load(Ordering::Relaxed);
    let d = tokens.max(1) as f64;
    println!("stage dma per timed decode token: {:.1} host syncs, {:.1} copy-engine copies, {:.0} MB, read-back+sync {:.3} ms, issue {:.3} ms",
        sy as f64 / d, co as f64 / d, by as f64 / 1e6 / d, rd as f64 / 1e6 / d, is as f64 / 1e6 / d);
}

impl Engine {
    /// CROW_STAGE_DMA: host-issued cold staging through the copy engine.
    /// Same shape as the PREFILL staging of CROW_PF_ASYNC=2 (`moe_run`, the `ce` branch:
    /// host read-back of the device plan, then one copy-engine memcpy per cold record
    /// into the staging slots), reduced to the decode case: the plan of a decode layer is
    /// the cold mask plus the two routed pointer tables of `router_top10`, and the copies
    /// go on the COMPUTE stream, so no `pa_ev_filled`/`pa_ev_done` event pair is needed.
    /// Per layer: one async DtoH of the cold mask and both routed pointer tables,
    /// exactly ONE host sync, one cuMemcpyDtoDAsync per cold combo and matrix
    /// (mapped host pointer -> staging slot, pcie_probe variant f), then one HtoD
    /// of the rewritten pointer tables. Hot combos keep their VRAM slab pointer.
    /// Byte-for-byte the same slots and the same pointers as the stage_cold kernel.
    unsafe fn stage_cold_dma(&self, t: usize) {
        use std::sync::atomic::Ordering;
        let s = &self.s;
        let st = &self.stage;
        let pin = st.dma.as_ref().expect("CROW_STAGE_DMA needs the pinned staging scratch");
        let n = t * TOPK;
        let (gu_b, dn_b) = (self.res.gu_bytes as usize, self.res.dn_bytes as usize);
        let h = pin.host as *mut u8;
        let h_gu = h as *mut u64;
        let h_dn = h.add(st.max * 8) as *mut u64;
        let h_cold = h.add(st.max * 16) as *mut u32;
        // one DtoH batch on the compute stream, then exactly one host sync per layer
        let t_read = std::time::Instant::now();
        cuda::memcpy_async(h_gu as Dev, s.gu_ptrs, n * 8);
        cuda::memcpy_async(h_dn as Dev, s.dn_ptrs, n * 8);
        cuda::memcpy_async(h_cold as Dev, s.cold, t * 4);
        cuda::sync();
        DMA_SYNCS.fetch_add(1, Ordering::Relaxed);
        DMA_NS_READ.fetch_add(t_read.elapsed().as_nanos() as u64, Ordering::Relaxed);
        let t_issue = std::time::Instant::now();
        // the copies go on the COMPUTE stream in issue order: the routed GEMVs of
        // this layer are issued after them, so no event and no second stream is
        // needed (a side stream would need one event per layer for the same order)
        let (mut copies, mut bytes) = (0u64, 0u64);
        for c in 0..n {
            if (*h_cold.add(c / TOPK) >> (c % TOPK)) & 1 == 0 { continue; }
            let (dgu, ddn) = (st.gu + (c * gu_b) as Dev, st.dn + (c * dn_b) as Dev);
            cuda::d2d_async(dgu, *h_gu.add(c), gu_b);
            cuda::d2d_async(ddn, *h_dn.add(c), dn_b);
            *h_gu.add(c) = dgu;
            *h_dn.add(c) = ddn;
            copies += 2;
            bytes += (gu_b + dn_b) as u64;
        }
        DMA_COPIES.fetch_add(copies, Ordering::Relaxed);
        DMA_BYTES.fetch_add(bytes, Ordering::Relaxed);
        // the rewritten tables land after the copies, before the GEMVs (stream order)
        cuda::memcpy_async(st.sgu, h_gu as Dev, n * 8);
        cuda::memcpy_async(st.sdn, h_dn as Dev, n * 8);
        DMA_NS_ISSUE.fetch_add(t_issue.elapsed().as_nanos() as u64, Ordering::Relaxed);
    }

    unsafe fn moe_run(&self, l: usize, mixed_m: Dev, t: usize) -> Dev {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let m = &self.w.moe[l];
        let (table, bitmap, counters, _, _) = self.res.layer_ptrs(l);
        let sel_counts = self.sel_counts as u64 + (l * E * 8) as u64;

        if t >= 8 && std::env::var("CROW_ROUTER_GEMM").as_deref() == Ok("1") {
            // bf16 tensor-core GEMM on the exact bf16 twin (summation order differs
            // from gemv_b -> not bit-identical; env-gated until validated)
            launch_v(k.f("gemm_bf16_dense"), ((E + 63) / 64) as u32, ((t + 7) / 8) as u32, 1, 128, &[
                m.router_bf as u64, mixed_m as u64, s.rlog as u64, p.n2560 as u64, p.nr512 as u64, p.t as u64]);
        } else {
            launch_v(k.f("gemv_b"), E as u32, t as u32, 1, 256, &[
                m.router as u64, mixed_m as u64, s.rlog as u64, p.n2560 as u64]);
        }
        launch_v(k.f("router_top10"), t as u32, 1, 1, 512, &[
            s.rlog as u64, bitmap, s.rids as u64, s.rwts as u64, s.gu_ptrs as u64,
            s.dn_ptrs as u64, table, s.cold as u64, counters, sel_counts]);
        // cold staging (decode-sized batches): coalesced PCIe pull into VRAM
        // slots + rewritten combo pointers; prefill chunks stay zero-copy
        let lb = self.res.lb.as_ref();
        let staged = stage_on() && t * TOPK <= self.stage.max;
        assert!(staged || lb.is_none() || (mma_on() && pf_gemm_on()),
            "low-bit cold tier needs a staging path (decode: t*TOPK <= stage.max; prefill: CROW_PF_GEMM)");
        let (gu_ptrs, dn_ptrs) = if staged {
            if stage_dma_on() && lb.is_none() {
                self.stage_cold_dma(t);
            } else if let Some(lb) = lb {
                launch_v(k.f("stage_cold_lb"), (t * TOPK) as u32, 2, stage_split(), 256, &[
                    s.gu_ptrs as u64, s.dn_ptrs as u64, s.cold as u64,
                    self.stage.gu as u64, self.stage.dn as u64,
                    self.stage.sgu as u64, self.stage.sdn as u64,
                    self.stage.gu_b as u64, self.stage.dn_b as u64, lb.bits_dev as u64, lb.lut_dev as u64]);
            } else if stage_kernel_ca() {
                // #19d: persistent cp.async.cg 4 KB tile copy, grid CROW_STAGE_BLOCKS
                // x 1 x 1; the tenth argument is the combo count BY VALUE, the very
                // `t * TOPK` that `staged` bounds-checked against stage.max above, so
                // the device item count and the host bound cannot disagree (fix I2)
                // #19e: this is the DEFAULT branch since 2026-09-12; CROW_STAGE_KERNEL=1
                // takes the stage_cold branch below
                // fix I3: the kernel has no tail tile, so both counts must be 4 KB
                assert!(self.stage.gu_bytes % 4096 == 0 && self.stage.dn_bytes % 4096 == 0,
                    "stage_cold_ca needs both staged byte counts to be multiples of 4096 (gate_up {} B, down {} B); CROW_STAGE_KERNEL=1 falls back to stage_cold",
                    self.stage.gu_bytes, self.stage.dn_bytes);
                launch_v(k.f("stage_cold_ca"), stage_blocks(), 1, 1, 256, &[
                    s.gu_ptrs as u64, s.dn_ptrs as u64, s.cold as u64,
                    self.stage.gu as u64, self.stage.dn as u64,
                    self.stage.sgu as u64, self.stage.sdn as u64,
                    self.stage.gu_b as u64, self.stage.dn_b as u64, (t * TOPK) as u64]);
            } else {
                launch_v(k.f("stage_cold"), (t * TOPK) as u32, 2, stage_split(), 256, &[
                    s.gu_ptrs as u64, s.dn_ptrs as u64, s.cold as u64,
                    self.stage.gu as u64, self.stage.dn as u64,
                    self.stage.sgu as u64, self.stage.sdn as u64,
                    self.stage.gu_b as u64, self.stage.dn_b as u64]);
            }
            (self.stage.sgu as u64, self.stage.sdn as u64)
        } else {
            (s.gu_ptrs as u64, s.dn_ptrs as u64)
        };

        // CROW_MMA=1: tensor-core path (quant_x_fp4 + gemv_fp4_mma[_d], p2/
        // mma_probe layouts) — the naive scalar GEMVs stay as reference/fallback.
        let mma = mma_on();
        if mma {
            // quantize mixed_m ONCE — the quantized rows feed the shared expert
            // gate|up AND all TOPK routed gate_up combos of a token
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                mixed_m as u64, s.xq_gu as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]); }
        }
        // shared expert — gate|up into ONE [t][1280] buffer (p13 layout, the
        // contract silu_mul640 reads); explicit stride, never the grid
        let dense = dense_mma_on();
        if dense {
            launch_mma_d(k, (INTER / 64) as u32, t, p.t as u64, &[
                m.sg.w as u64, s.xq_gu as u64, m.sg.gs as u64, s.sh12 as u64,
                p.n2560 as u64, p.n640 as u64, p.n1280 as u64]);
            launch_mma_d(k, (INTER / 64) as u32, t, p.t as u64, &[
                m.su.w as u64, s.xq_gu as u64, m.su.gs as u64, (s.sh12 + (INTER * 4) as u64),
                p.n2560 as u64, p.n640 as u64, p.n1280 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_bs"), INTER as u32, t as u32, 1, 256, &[
                m.sg.w as u64, mixed_m as u64, m.sg.gs as u64, s.sh12 as u64, p.n2560 as u64, p.n1280 as u64]);
            launch_v(k.f("gemv_fp4_bs"), INTER as u32, t as u32, 1, 256, &[
                m.su.w as u64, mixed_m as u64, m.su.gs as u64, (s.sh12 + (INTER * 4) as u64), p.n2560 as u64, p.n1280 as u64]);
        }
        if qfuse_on() {
            launch_v(k.f("silu_mul640_q"), 3, t as u32, 1, 256, &[s.sh12 as u64, s.sh2 as u64, s.xq_s as u64]);
        } else {
            launch_v(k.f("silu_mul640"), 3, t as u32, 1, 256, &[s.sh12 as u64, s.sh2 as u64]);
        }
        if dense {
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                s.sh2 as u64, s.xq_s as u64, p.n640 as u64, p.one as u64, p.n640 as u64]); }
            launch_mma_d(k, (H / 64) as u32, t, p.t as u64, &[
                m.sdn.w as u64, s.xq_s as u64, m.sdn.gs as u64, s.sdown as u64,
                p.n640 as u64, p.n2560 as u64, p.n2560 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_b"), H as u32, t as u32, 1, 256, &[
                m.sdn.w as u64, s.sh2 as u64, m.sdn.gs as u64, s.sdown as u64, p.n640 as u64]);
        }
        launch_v(k.f("gemv_b"), 1, t as u32, 1, 256, &[
            m.sgate as u64, mixed_m as u64, s.sgv as u64, p.n2560 as u64]);
        // moe_out is ASSIGNED by gate_shared (first writer) - no memset: the
        // synchronous cuMemsetD8 ran on the legacy stream and was never part
        // of a captured graph (replay would accumulate across layers)
        launch_v(k.f("gate_shared"), 10, t as u32, 1, 256, &[
            s.sdown as u64, s.sgv as u64, s.moe_out as u64]);

        // routed experts through the residency pointer tables (zero-copy cold)
        // gs_dev is [LAYERS][2] f32 — per-tensor global scales differ per layer
        if mma && pf_gemm_on() && !staged {
            // prefill-sized batch: expert-grouped tile GEMM (each cold expert
            // crosses PCIe ONCE per layer, n = 8 tokens per mma)
            let st = &self.stage;
            launch_v(k.f("moe_count"), ((t * TOPK + 255) / 256) as u32, 1, 1, 256, &[
                s.rids as u64, st.counts as u64, self.pf_ncombo as u64]);
            launch_v(k.f("moe_plan"), 1, 1, 1, 1024, &[
                st.counts as u64, st.offsets as u64, st.tiles as u64, st.n_tiles as u64,
                st.cursor as u64, st.tg as u64, st.max_tiles_p as u64]);
            launch_v(k.f("moe_scatter"), ((t * TOPK + 255) / 256) as u32, 1, 1, 256, &[
                s.rids as u64, st.offsets as u64, st.cursor as u64, st.perm as u64, self.pf_ncombo as u64]);
            let max_tiles_now = t * TOPK / 8 + E + 1;
            let tg = pf_tg();
            let n_groups = (max_tiles_now + tg - 1) / tg;
            assert!(n_groups <= PF_MAX_GROUPS);
            let gs_gu = (self.res.gs_dev + (l * 8) as u64) as u64;
            let gs_dn = (self.res.gs_dev + (l * 8 + 4) as u64) as u64;
            let pa = pf_async_on();
            let cs = cuda::cur_stream();
            if pa {
                // the plan (tiles/perm) is complete on the compute stream
                cuda::event_record(self.pa_ev_plan, cs);
                cuda::stream_wait_event(self.pa_stream, self.pa_ev_plan);
            }
            // CROW_PF_ASYNC=2: the copy engine stages the cold experts - the host
            // reads the plan back once per layer (sync) and issues one memcpy per
            // cold record ahead of the stage kernel on the side stream
            let ce = pa && (pf_async_mode() == 2 || pf_async_mode() == 4) && self.res.lb.is_none();
            let (tiles_h, hot_h) = if ce {
                let nt = cuda::dtoh_i32(st.n_tiles, 1)[0] as usize;
                let tl = cuda::dtoh_i32(st.tiles, nt * 4);
                let mut hot = vec![false; E];
                for &id in &self.res.sets[l] { if (id as usize) < E { hot[id as usize] = true; } }
                (tl, hot)
            } else { (Vec::new(), Vec::new()) };
            // staging of one tile group (side stream + slot set gi%2 with CROW_PF_ASYNC)
            let stage_group = |gi: usize| unsafe {
                let grp = st.grp as u64 + (gi * 4) as u64;
                let (pin_gu, pin_dn) = (self.res.cold_gu[l].dev as u64, self.res.cold_dn[l].dev as u64);
                let (ring_gu, ring_dn) = if self.pf_dma_live.get() {
                    (self.pf_ring_gu[l % 2] as u64, self.pf_ring_dn[l % 2] as u64)
                } else { (0u64, 0u64) };
                if pa {
                    // slot set gi%2 is free once group gi-2's GEMMs are done
                    cuda::stream_wait_event(self.pa_stream, self.pa_ev_done[gi % 2]);
                    cuda::set_stream(self.pa_stream as u64);
                }
                if ce {
                    let t1 = ((gi + 1) * tg).min(tiles_h.len() / 4);
                    for ti in gi * tg..t1 {
                        let e = tiles_h[ti * 4] as usize;
                        let w = tiles_h[ti * 4 + 3];
                        if (w >> 16) & 1 == 1 && !hot_h[e] {
                            let slot = ((w & 0xFFFF) as usize + (gi & 1) * tg) as u64;
                            let rec = *self.res.cold_index[l].get(&(e as u32)).expect("cold expert without a pinned record") as u64;
                            cuda::memcpy_async_on((st.gu as u64 + slot * self.res.gu_bytes) as cudarc::driver::sys::CUdeviceptr,
                                self.res.cold_gu[l].dev + rec * self.res.gu_bytes, self.res.gu_bytes as usize, self.pa_stream);
                            cuda::memcpy_async_on((st.dn as u64 + slot * self.res.dn_bytes) as cudarc::driver::sys::CUdeviceptr,
                                self.res.cold_dn[l].dev + rec * self.res.dn_bytes, self.res.dn_bytes as usize, self.pa_stream);
                        }
                    }
                }
                if let Some(lb) = self.res.lb.as_ref() {
                    launch_v(k.f("stage_tiles_lb"), tg as u32, 2, stage_split(), 256, &[
                        st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, table, bitmap,
                        st.gu as u64, st.dn as u64, st.eptr as u64, st.gu_b as u64, st.dn_b as u64,
                        lb.bits_dev as u64, lb.lut_dev as u64, pin_gu, pin_dn, ring_gu, ring_dn]);
                } else {
                    launch_v(k.f("stage_tiles"), tg as u32, 2, stage_split(), 256, &[
                        st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, table, bitmap,
                        st.gu as u64, st.dn as u64, st.eptr as u64, st.gu_b as u64, st.dn_b as u64,
                        pin_gu, pin_dn, ring_gu, ring_dn]);
                }
                if pa {
                    cuda::event_record(self.pa_ev_filled[gi % 2], self.pa_stream);
                    cuda::stream_query(self.pa_stream); // WDDM: submit the side-stream batch now
                    cuda::set_stream(cs as u64);
                }
            };
            if pa {
                stage_group(0);
            }
            for gi in 0..n_groups {
                let grp = st.grp as u64 + (gi * 4) as u64;
                if pa {
                    // pipelined host order: group gi+1's copy is issued before group gi's GEMMs
                    if gi + 1 < n_groups {
                        stage_group(gi + 1);
                    }
                    cuda::stream_wait_event(cs, self.pa_ev_filled[gi % 2]);
                } else {
                    stage_group(gi);
                }
                if pf_async_mode() != 4 { // DIAGNOSTIC (=4): copy-only floor, no tile GEMMs
                launch_v(k.f("gemm_fp4_tiles"), ((2 * INTER) / 64) as u32, tg as u32, 1, mma_bx(), &[
                    st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, st.eptr as u64, p.zero as u64,
                    s.xq_gu as u64, st.perm as u64, s.h1 as u64, p.n2560 as u64, p.k_top10 as u64, gs_gu]);
                launch_v(k.f("silu_tiles"), tg as u32, 8, 1, 256, &[
                    s.h1 as u64, s.h2 as u64, st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, st.perm as u64]);
                launch_v(k.f("quant_tiles"), tg as u32, 8, 1, 128, &[
                    s.h2 as u64, s.xq_dn as u64, st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, st.perm as u64]);
                launch_v(k.f("gemm_fp4_tiles"), (H / 64) as u32, tg as u32, 1, mma_bx(), &[
                    st.tiles as u64, st.n_tiles as u64, grp, st.tg as u64, st.eptr as u64, p.one as u64,
                    s.xq_dn as u64, st.perm as u64, s.eo as u64, p.n640 as u64, p.one as u64, gs_dn]);
                }
                if pa {
                    cuda::event_record(self.pa_ev_done[gi % 2], cs);
                }
            }
            launch_v(k.f("acc_combo"), (H / 256) as u32, t as u32, 1, 256, &[
                s.eo as u64, s.rwts as u64, s.moe_out as u64]);
            return s.moe_out;
        }
        if mma {
            // xq_gu was quantized above (shared with the shared expert)
            launch_v(k.f("gemv_fp4_mma"), ((2 * INTER) / 64) as u32, (t * TOPK) as u32, 1, mma_bx(), &[
                gu_ptrs, s.xq_gu as u64, (self.res.gs_dev + (l * 8) as u64) as u64,
                s.h1 as u64, p.n2560 as u64, p.k_top10 as u64]);
        } else {
            launch_v(k.f("gemv_fp4_ptrb"), (2 * INTER) as u32, (t * TOPK) as u32, 1, 256, &[
                gu_ptrs, mixed_m as u64, (self.res.gs_dev + (l * 8) as u64) as u64, s.h1 as u64,
                p.n2560 as u64, p.k_top10 as u64, p.n2560 as u64]);
        }
        if qfuse_on() && mma {
            launch_v(k.f("silu_mul_combo_q"), ((t * TOPK * INTER + 255) / 256) as u32, 1, 1, 256, &[
                s.h1 as u64, s.h2 as u64, p.nt_combo as u64, s.xq_dn as u64]);
        } else {
            launch_v(k.f("silu_mul_combo"), ((t * TOPK * INTER + 255) / 256) as u32, 1, 1, 256, &[
                s.h1 as u64, s.h2 as u64, p.nt_combo as u64]);
        }
        if mma {
            // h2 is per-combo ([t*TOPK][640]) — quantize per combo row
            if !qfuse_on() { launch_v(k.f("quant_x_fp4"), (t * TOPK) as u32, 1, 1, 128, &[
                s.h2 as u64, s.xq_dn as u64, p.n640 as u64, p.one as u64, p.n640 as u64]); }
            launch_v(k.f("gemv_fp4_mma"), (H / 64) as u32, (t * TOPK) as u32, 1, mma_bx(), &[
                dn_ptrs, s.xq_dn as u64, (self.res.gs_dev + (l * 8 + 4) as u64) as u64,
                s.eo as u64, p.n640 as u64, p.one as u64]);
        } else {
            launch_v(k.f("gemv_fp4_ptrb"), H as u32, (t * TOPK) as u32, 1, 256, &[
                dn_ptrs, s.h2 as u64, (self.res.gs_dev + (l * 8 + 4) as u64) as u64, s.eo as u64,
                p.n640 as u64, p.one as u64, p.n640 as u64]);
        }
        launch_v(k.f("acc_combo"), (H / 256) as u32, t as u32, 1, 256, &[
            s.eo as u64, s.rwts as u64, s.moe_out as u64]);
        s.moe_out
    }

    unsafe fn ple_run(&mut self, cnq: &mut Cnq, t: usize, prefix: &[i64], chunk_ids: &[i64], is_step: bool) {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let dbg = std::env::var("ENGINE_DEBUG_STEP").is_ok();
        if dbg { eprintln!("[ple] enter"); }
        let pl = &mut self.ple;
        // p15 semantics: the returned rows cover prefix+chunk; keep only the
        // chunk's rows (the prefix rows are history, already processed)
        let all_ngids = pl.ngram_ids(prefix, chunk_ids);
        let skip = prefix.len();
        let ngids = all_ngids[skip..].to_vec();
        if dbg { eprintln!("[ple] ngids {}x{}", ngids.len(), ngids.first().map(|r| r.len()).unwrap_or(0)); }
        let flat: Vec<i64> = ngids.concat();
        if dbg { eprintln!("[ple] flat {} rows", flat.len()); }
        let slots = { pl.ensure_rows(cnq, &flat) };
        if dbg { eprintln!("[ple] ensured {} slots", slots.len()); }
        cuda::to_i32_into(s.ple_slots, &slots.iter().map(|&v| v as i32).collect::<Vec<_>>());
        if dbg { eprintln!("[ple] slots uploaded"); }

        if is_step {
            // #11 (2026-09-05): the step's kernels run inside the layer loop at
            // PLE_LAYER (ple_step_kernels) - the reference adds ple(h) at the TOP
            // of layer 1, i.e. to layer 0's output; only the host prep (n-gram
            // ids, row cache, slot upload) stays here, before any graph capture
        } else {
            launch_v(k.f("gather_ple_fp4"), PLE_NHEADS as u32, t as u32, 1, PLE_EMB_DIM as u32, &[
                pl.cache as u64, pl.gs as u64, s.ple_slots as u64, s.emb as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 11"); }
            if dense_mma_on() {
                launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                    s.emb as u64, s.xq_e as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]);
                launch_mma_d(k, (HCT / 64) as u32, t, p.t as u64, &[
                    pl.key.w as u64, s.xq_e as u64, pl.key.gs as u64, s.ple_key as u64,
                    p.n2560 as u64, p.n10240 as u64, p.n10240 as u64]);
            } else {
                launch_v(k.f("gemv_fp4_b"), HCT as u32, t as u32, 1, 256, &[
                    pl.key.w as u64, s.emb as u64, pl.key.gs as u64, s.ple_key as u64, p.n2560 as u64]);
            }
        if dbg { cuda::sync(); eprintln!("[ple] step 12"); }
            launch_v(k.f("rms_group"), 4, t as u32, 1, 256, &[
                s.ple_key as u64, pl.norm_key as u64, s.ple_kn as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 13"); }
            if dense_mma_on() {
                launch_mma_d(k, (H / 64) as u32, t, p.t as u64, &[
                    pl.value.w as u64, s.xq_e as u64, pl.value.gs as u64, s.ple_val as u64,
                    p.n2560 as u64, p.n2560 as u64, p.n2560 as u64]);
            } else {
                launch_v(k.f("gemv_fp4_b"), H as u32, t as u32, 1, 256, &[
                    pl.value.w as u64, s.emb as u64, pl.value.gs as u64, s.ple_val as u64, p.n2560 as u64]);
            }
        if dbg { cuda::sync(); eprintln!("[ple] step 14"); }
            if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                cuda::sync();
                let dump = |tag: &str, v: Vec<f32>| {
                    let mut b = Vec::with_capacity(v.len() * 4);
                    for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                    std::fs::write(format!("{dir}/ple-{tag}.f32"), b).unwrap();
                };
                dump("h-before-qn", cuda::dtoh(s.h, t * HCT));
                dump("norm-query-w", cuda::dtoh(pl.norm_query, HCT));
            }
            launch_v(k.f("rms_group"), 4, t as u32, 1, 256, &[
                s.h as u64, pl.norm_query as u64, s.ple_qn as u64]);
            if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                cuda::sync();
                let v = cuda::dtoh(s.ple_qn, t * HCT);
                let mut b = Vec::with_capacity(v.len() * 4);
                for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                std::fs::write(format!("{dir}/ple-qn-immediate.f32"), b).unwrap();
            }
        if dbg { cuda::sync(); eprintln!("[ple] step 15"); }
            launch_v(k.f("gate_dot"), 4, t as u32, 1, 256, &[
                s.ple_kn as u64, s.ple_qn as u64, s.ple_gate as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 16"); }
            launch_v(k.f("gate_apply"), 4, t as u32, 1, 256, &[
                s.ple_gate as u64, s.ple_val as u64, s.ple_gs as u64, s.ple_gated as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 17"); }
            launch_v(k.f("rms_group"), 4, t as u32, 1, 256, &[
                s.ple_gated as u64, pl.norm_conv as u64, s.ple_gn as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 18"); }
            launch_v(k.f("ple_conv"), GDN_CONV as u32, 1, 1, 256, &[
                s.ple_gn as u64, pl.conv as u64, s.ple_gated as u64, s.ple_out as u64,
                p.t as u64, pl.state as u64]);
        if dbg {
            cuda::sync();
            let emb = cuda::dtoh(s.emb, t * PLE_EMBED);
            let kn = cuda::dtoh(s.ple_kn, t * HCT);
            let gd = cuda::dtoh(s.ple_gated, t * HCT);
            let gn = cuda::dtoh(s.ple_gn, t * HCT);
            let po = cuda::dtoh(s.ple_out, t * HCT);
            let cnt = |v: &[f32]| v.iter().filter(|x| x.is_nan()).count();
            let mut rows = Vec::new();
            for r in 0..t {
                let lo = r * HCT;
                rows.push(cnt(&po[lo..lo + HCT]));
            }
            eprintln!("[ple dbg] nan emb={} key_n={} gated={} gn={} out={} out_rows={:?} slots0={:?}",
                cnt(&emb), cnt(&kn), cnt(&gd), cnt(&gn), cnt(&po), rows,
                &cuda::dtoh_i32(s.ple_slots, 16));
        }
        if dbg { cuda::sync(); eprintln!("[ple] step 19"); }
            launch_v(k.f("ple_state_update"), (GDN_CONV as u32 + 255) / 256, 1, 1, 256, &[
                s.ple_gn as u64, pl.state as u64, p.t as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 20"); }
            launch_v(k.f("add_flat"), ((t * HCT + 255) / 256) as u32, 1, 1, 256, &[
                s.ple_out as u64, s.h as u64, p.nt_hct as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 21"); }
            if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                cuda::sync();
                let dump = |tag: &str, v: Vec<f32>| {
                    let mut b = Vec::with_capacity(v.len() * 4);
                    for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                    std::fs::write(format!("{dir}/ple-{tag}.f32"), b).unwrap();
                };
                dump("emb", cuda::dtoh(s.emb, t * PLE_EMBED));
                dump("key", cuda::dtoh(s.ple_key, t * HCT));
                dump("val", cuda::dtoh(s.ple_val, t * H));
                dump("qn", cuda::dtoh(s.ple_qn, t * HCT));
                dump("gate", cuda::dtoh(s.ple_gate, t * 4));
                dump("gated", cuda::dtoh(s.ple_gated, t * HCT));
                dump("gn", cuda::dtoh(s.ple_gn, t * HCT));
                dump("out", cuda::dtoh(s.ple_out, t * HCT));
                let sl = cuda::dtoh_i32(s.ple_slots, t * PLE_NHEADS);
                std::fs::write(format!("{dir}/ple-slots.json"), format!("{sl:?}")).unwrap();
                std::fs::write(format!("{dir}/ple-ngids.json"), format!("{flat:?}")).unwrap();
            }
        }
    }

    /// PLE step kernels (decode, t = 1): launched from decode_step at l == PLE_LAYER,
    /// after ple_run(.., is_step = true) prepared the row slots (#11, 2026-09-05)
    unsafe fn ple_step_kernels(&self) {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        let pl = &self.ple;
        let t = 1usize;
        let dbg = std::env::var("ENGINE_DEBUG_STEP").is_ok();
            launch_v(k.f("gather_ple_fp4"), PLE_NHEADS as u32, 1, 1, PLE_EMB_DIM as u32, &[
                pl.cache as u64, pl.gs as u64, s.ple_slots as u64, s.emb as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 1"); }
            if dense_mma_on() {
                // one quantized embed row set serves key + value (both k=2560)
                launch_v(k.f("quant_x_fp4"), t as u32, 1, 1, 128, &[
                    s.emb as u64, s.xq_e as u64, p.n2560 as u64, p.one as u64, p.n2560 as u64]);
                launch_mma_d(k, (HCT / 64) as u32, t, p.t as u64, &[
                    pl.key.w as u64, s.xq_e as u64, pl.key.gs as u64, s.ple_key as u64,
                    p.n2560 as u64, p.n10240 as u64, p.n10240 as u64]);
            } else {
                launch_v(k.f("gemv_fp4"), HCT as u32, 1, 1, 256, &[
                    pl.key.w as u64, s.emb as u64, pl.key.gs as u64, s.ple_key as u64, p.n2560 as u64]);
            }
        if dbg { cuda::sync(); eprintln!("[ple] step 2"); }
            launch_v(k.f("rms_group"), 4, 1, 1, 256, &[
                s.ple_key as u64, pl.norm_key as u64, s.ple_kn as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 3"); }
            if dense_mma_on() {
                launch_mma_d(k, (H / 64) as u32, t, p.t as u64, &[
                    pl.value.w as u64, s.xq_e as u64, pl.value.gs as u64, s.ple_val as u64,
                    p.n2560 as u64, p.n2560 as u64, p.n2560 as u64]);
            } else {
                launch_v(k.f("gemv_fp4"), H as u32, 1, 1, 256, &[
                    pl.value.w as u64, s.emb as u64, pl.value.gs as u64, s.ple_val as u64, p.n2560 as u64]);
            }
        if dbg { cuda::sync(); eprintln!("[ple] step 4"); }
            launch_v(k.f("rms_group"), 4, 1, 1, 256, &[
                s.h as u64, pl.norm_query as u64, s.ple_qn as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 5"); }
            launch_v(k.f("gate_dot"), 4, 1, 1, 256, &[
                s.ple_kn as u64, s.ple_qn as u64, s.ple_gate as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 6"); }
            launch_v(k.f("gate_apply"), 4, 1, 1, 256, &[
                s.ple_gate as u64, s.ple_val as u64, s.ple_gs as u64, s.ple_gated as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 7"); }
            launch_v(k.f("rms_group"), 4, 1, 1, 256, &[
                s.ple_gated as u64, pl.norm_conv as u64, s.ple_gn as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 8"); }
            // #11 (2026-09-05): the step passed s.h as gated_row, so the token got
            // h + (h + silu(conv)) = 2h + silu(conv) instead of h + gated + silu(conv)
            // (prefill's ple_conv uses ple_gated). Decode rows drifted 5-15 logit
            // units from the prefill rows over the same context; with PLE off both
            // paths agreed within 1.8. gated_row is the gated value row, as in prefill.
            launch_v(k.f("ple_conv_step"), (GDN_CONV as u32 + 255) / 256, 1, 1, 256, &[
                s.ple_gn as u64, s.ple_gated as u64, pl.conv as u64, pl.state as u64, s.ple_out as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 9"); }
            launch_v(k.f("add_flat"), (HCT as u32 + 255) / 256, 1, 1, 256, &[
                s.ple_out as u64, s.h as u64, p.nt_hct1 as u64]);
        if dbg { cuda::sync(); eprintln!("[ple] step 10"); }
    }

    unsafe fn head_run(&self, rows: usize) {
        let k = &self.k;
        let p = &self.p;
        let s = &self.s;
        launch_v(k.f("rms_group"), 4, rows as u32, 1, 256, &[
            s.h as u64, self.w.mx_norm as u64, s.normed as u64]);
        self.w.mx_down.launch_gemv(k, LOWRANK, p.n320 as u64, rows, p.t as u64, s.normed as u64, s.low as u64, p.n10240 as u64);
        launch_v(k.f("silu_div4"), ((rows * LOWRANK + 255) / 256) as u32, 1, 1, 256, &[
            s.low as u64, s.sil as u64, p.nt_low as u64]);
        self.w.mx_up.launch_gemv(k, HCT, p.n10240 as u64, rows, p.t as u64, s.sil as u64, s.mixw as u64, p.n320 as u64);
        launch_v(k.f("sigmoid_el"), ((rows * HCT + 255) / 256) as u32, 1, 1, 256, &[
            s.mixw as u64, p.nt_hct as u64]);
        launch_v(k.f("mix_streams"), 10, rows as u32, 1, 256, &[
            s.mixw as u64, s.normed as u64, s.mixed_final as u64]);
    }

    unsafe fn lm_head_row(&self, row: usize) {
        let dst = self.s.logits as u64;
        if bf16_w_on() {
            launch_v(self.k.f("gemv_bf16_w"), ((V + 7) / 8) as u32, 1, 1, 256, &[
                self.w.lm_head as u64, (self.s.mixed_final as u64 + (row * H * 4) as u64),
                dst, self.p.n2560 as u64, self.p.n_vocab as u64]);
        } else {
            launch_v(self.k.f("gemv_bf16"), V as u32, 1, 1, 256, &[
                self.w.lm_head as u64, (self.s.mixed_final as u64 + (row * H * 4) as u64),
                dst, self.p.n2560 as u64]);
        }
    }
}

impl Engine {
    /// layer-0 chain with p8-style stage dumps for the oracle comparator
    pub unsafe fn run_layer0_with_stage_dumps(&mut self, h_host: &[f32], t: usize, dump_dir: &str) -> Vec<f32> {
        std::fs::create_dir_all(dump_dir).unwrap();
        let dir = dump_dir;
        let l = 0usize;
        let pos_base = 0usize;
        cuda::to_f32_into(self.s.h, h_host);
        cuda::to_i32_into(self.p.t, &[t as i32]);
        cuda::to_i32_into(self.p.init, &[1]);
        cuda::to_i32_into(self.p.pos_base, &[pos_base as i32]);
        cuda::to_i32_into(self.p.slot_base, &[pos_base as i32]);
        cuda::to_i32_into(self.p.nt_hct, &[((t * HCT) as i32)]);
        cuda::to_i32_into(self.p.nt_hc, &[((t * HCN) as i32)]);
        cuda::to_i32_into(self.p.nt_low, &[((t * LOWRANK) as i32)]);
        cuda::to_i32_into(self.p.nt_6144, &[((t * GDN_VAL) as i32)]);
        cuda::to_i32_into(self.p.nt_combo, &[((t * TOPK * INTER) as i32)]);
        cuda::to_i32_into(self.pf_ncombo, &[(t * TOPK) as i32]);
        let dump = |dir: &str, tag: &str, v: &[f32]| {
            let mut b = Vec::with_capacity(v.len() * 4);
            for x in v { b.extend_from_slice(&x.to_le_bytes()); }
            std::fs::write(format!("{dir}/gpu-{tag}.f32"), b).unwrap();
        };
        self.hc_run(&self.w.hc[l], self.s.h, t, self.s.mixed, self.s.injw);
        cuda::sync();
        dump(dir, "mixed-a", &cuda::dtoh(self.s.mixed, t * H));
        if std::env::var("ENGINE_DEBUG_HC").is_ok() {
            dump(dir, "hc-normed-a", &cuda::dtoh(self.s.normed, t * HCT));
            dump(dir, "hc-low", &cuda::dtoh(self.s.low, t * LOWRANK));
            dump(dir, "hc-sil", &cuda::dtoh(self.s.sil, t * LOWRANK));
            dump(dir, "hc-mixw-a", &cuda::dtoh(self.s.mixw, t * HCT));
            dump(dir, "hc-injw-a", &cuda::dtoh(self.s.injw, t * HCN));
        }
        let sub = self.gdn_prompt(l, self.s.mixed, t, true);
        cuda::sync();
        dump(dir, "gdn-out", &cuda::dtoh(sub, t * H));
        launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
            self.s.h as u64, sub as u64, self.s.injw as u64, self.s.x1 as u64]);
        cuda::sync();
        dump(dir, "x1", &cuda::dtoh(self.s.x1, t * HCT));
        self.hc_run(&self.w.hc2[l], self.s.x1, t, self.s.mixed_m, self.s.injw);
        cuda::sync();
        dump(dir, "mixed-m", &cuda::dtoh(self.s.mixed_m, t * H));
        let moe = self.moe_run(l, self.s.mixed_m, t);
        cuda::sync();
        dump(dir, "moe", &cuda::dtoh(moe, t * H));
        launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
            self.s.x1 as u64, moe as u64, self.s.injw as u64, self.s.h as u64]);
        cuda::sync();
        cuda::dtoh(self.s.h, t * HCT)
    }

    /// DEBUG/parity helper: run ONE decoder layer's full chain (hc → sub →
    /// inject → hc2 → moe → inject) on the given [T][HCT] host stream.
    pub unsafe fn run_single_layer(&mut self, l: usize, h_host: &[f32], t: usize, pos_base: usize, first: bool) -> Vec<f32> {
        cuda::to_f32_into(self.s.h, h_host);
        cuda::to_i32_into(self.p.t, &[t as i32]);
        cuda::to_i32_into(self.p.init, &[if first { 1 } else { 0 }]);
        cuda::to_i32_into(self.p.pos_base, &[pos_base as i32]);
        cuda::to_i32_into(self.p.pos_base_b4, &[((pos_base / 4)) as i32]);
        cuda::to_i32_into(self.p.slot_base, &[pos_base as i32]);
        let ncb: Vec<i32> = (0..t)
            .map(|i| ((pos_base + i + 1) / 4).min((self.st.context + 3) / 4) as i32)
            .collect();
        cuda::to_i32_into(self.p.ncb, &ncb);
        let posrows: Vec<i32> = (0..t).map(|i| (pos_base + i) as i32).collect();
        cuda::to_i32_into(self.p.pos_row, &posrows);
        cuda::to_i32_into(self.p.nt_low, &[((t * LOWRANK) as i32)]);
        cuda::to_i32_into(self.p.nt_hct, &[((t * HCT) as i32)]);
        cuda::to_i32_into(self.p.nt_hc, &[((t * HCN) as i32)]);
        cuda::to_i32_into(self.p.nt_combo, &[((t * TOPK * INTER) as i32)]);
        cuda::to_i32_into(self.pf_ncombo, &[(t * TOPK) as i32]);

        self.hc_run(&self.w.hc[l], self.s.h, t, self.s.mixed, self.s.injw);
        let sub = match &self.w.sub[l] {
            SubW::Gdn { .. } => self.gdn_prompt(l, self.s.mixed, t, first),
            SubW::Attn { .. } => self.attn_prompt(l, self.s.mixed, t, pos_base),
        };
        launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
            self.s.h as u64, sub as u64, self.s.injw as u64, self.s.x1 as u64]);
        self.hc_run(&self.w.hc2[l], self.s.x1, t, self.s.mixed_m, self.s.injw);
        let moe = self.moe_run(l, self.s.mixed_m, t);
        launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
            self.s.x1 as u64, moe as u64, self.s.injw as u64, self.s.h as u64]);
        cuda::sync();
        cuda::dtoh(self.s.h, t * HCT)
    }

    /// DEBUG/layercheck helper: run ONLY the attention sub-block of layer `l`
    /// (q/k/v GEMVs → split → q_norm/k_norm → rotary → KV store → QSA select →
    /// attention → sigmoid gate → o_proj) on a host [T][H] `mixed` input,
    /// returning the [T][H] o_proj output (the p7-golden contract).
    pub unsafe fn run_attn_subblock(&mut self, l: usize, x_host: &[f32], t: usize, pos_base: usize) -> Vec<f32> {
        assert!(is_attn(l), "layer {l} is not an attention layer");
        assert_eq!(x_host.len(), t * H, "attn subblock input must be [T][H]");
        cuda::to_f32_into(self.s.mixed, x_host);
        cuda::to_i32_into(self.p.t, &[t as i32]);
        cuda::to_i32_into(self.p.init, &[1]);
        cuda::to_i32_into(self.p.pos_base, &[pos_base as i32]);
        cuda::to_i32_into(self.p.pos_base_b4, &[(pos_base / 4) as i32]);
        cuda::to_i32_into(self.p.slot_base, &[pos_base as i32]);
        let ncb: Vec<i32> = (0..t)
            .map(|i| ((pos_base + i + 1) / 4).min((self.st.context + 3) / 4) as i32)
            .collect();
        cuda::to_i32_into(self.p.ncb, &ncb);
        let posrows: Vec<i32> = (0..t).map(|i| (pos_base + i) as i32).collect();
        cuda::to_i32_into(self.p.pos_row, &posrows);
        self.attn_prompt(l, self.s.mixed, t, pos_base);
        cuda::sync();
        cuda::dtoh(self.s.ay, t * H)
    }

    /// chunked prefill; appends state; returns the last position's greedy token.
    /// `collect_logits` stores every position's logits host-side (parity mode).
    pub unsafe fn prefill(
        &mut self,
        cnq: &mut Cnq,
        ids: &[i64],
        mut collect_logits: Option<&mut Vec<Vec<f32>>>,
    ) -> usize {
        let mut start = 0usize;
        let first = self.pos == 0;
        // live prefill progress on stderr (unbuffered — visible immediately in
        // any redirected log; requested 2026-09-03: a first number with the
        // START of prefill, not only after the task completes)
        let pre_total = ids.len();
        let pre_t0 = std::time::Instant::now();
        eprintln!("[prefill] START: {pre_total} tokens, chunk {}", self.cfg.prompt_chunk);
        let mut prefetch: Option<std::thread::JoinHandle<()>> = None;
        // balanced chunks: 2100 tokens at chunk 1024 become 2 x 1050 instead of
        // 1024 + 1024 + 52 - every chunk costs one full PCIe pass over the cold
        // tier, so a small tail chunk is nearly as expensive as a full one
        // (exact: the per-token math does not depend on the chunk cut)
        let n_chunks = (ids.len() + self.cfg.prompt_chunk - 1) / self.cfg.prompt_chunk.max(1);
        // measured 2026-09-04: 3 x 700 (382 tok/s) LOSES to 1024+1024+52 (414 tok/s) -
        // a small tail chunk touches fewer cold experts than a full pass; opt-in only
        let balanced = if std::env::var("CROW_CHUNK_BALANCE").as_deref() != Ok("1") || n_chunks <= 1 {
            self.cfg.prompt_chunk
        } else {
            (((ids.len() + n_chunks - 1) / n_chunks + 3) / 4 * 4).min(self.cfg.prompt_chunk)
        };
        while start < ids.len() {
            let t = (ids.len() - start).min(balanced);
            let chunk = &ids[start..start + t];
            let pos_base = self.pos;
            // wait for the row touches of THIS chunk (issued during the previous one)
            if let Some(h) = prefetch.take() {
                let _ = h.join();
            }
            // prefetch the NEXT chunk's PLE rows on a helper thread (page-cache
            // warm-up through the file mapping; no-op without a mapping)
            if cnq.map != 0 && self.cfg.ple && start + t < ids.len() && std::env::var("CROW_PLE_PREFETCH").as_deref() != Ok("0") {
                let nt = (ids.len() - start - t).min(self.cfg.prompt_chunk);
                let next = &ids[start + t..start + t + nt];
                let hist_end = pos_base + t; // position after this chunk
                let pre: Vec<i64> = {
                    let mut v: Vec<i64> = Vec::new();
                    let full: Vec<i64> = self.history.iter().copied().chain(chunk.iter().copied()).collect();
                    let _ = hist_end;
                    for k in 0..2 {
                        let idx = full.len() as i64 - 2 + k;
                        v.push(if idx >= 0 { full[idx as usize] } else { PLE_EOS });
                    }
                    v
                };
                let offsets = self.ple.row_offsets(cnq, &pre, next);
                let base = cnq.map;
                let len = cnq.map_len;
                prefetch = Some(std::thread::spawn(move || {
                    let mut sink = 0u8;
                    for off in offsets {
                        if off + 108 <= len {
                            // touch first and last byte of the row (rows may straddle a page)
                            unsafe {
                                sink ^= std::ptr::read_volatile((base as *const u8).add(off as usize));
                                sink ^= std::ptr::read_volatile((base as *const u8).add(off as usize + 107));
                            }
                        }
                    }
                    std::hint::black_box(sink);
                }));
            }
            let p = &self.p;

            // per-chunk scalar refresh (device buffers, one sync each — chunk level)
            cuda::to_i32_into(p.t, &[t as i32]);
            cuda::to_i32_into(p.init, &[if first && start == 0 { 1 } else { 0 }]);
            cuda::to_i32_into(p.pos_base, &[pos_base as i32]);
            // rope base for pooled blocks = BLOCK index base (kernel multiplies by 4)
            cuda::to_i32_into(p.pos_base_b4, &[((pos_base / 4)) as i32]);
            cuda::to_i32_into(p.slot_base, &[pos_base as i32]);
            let ncb: Vec<i32> = (0..t)
                .map(|i| ((pos_base + i + 1) / 4).min((self.st.context + 3) / 4) as i32)
                .collect();
            cuda::to_i32_into(p.ncb, &ncb);
            let posrows: Vec<i32> = (0..t).map(|i| (pos_base + i) as i32).collect();
            cuda::to_i32_into(p.pos_row, &posrows);
            cuda::to_i32_into(p.nt_low, &[((t * LOWRANK) as i32)]);
            cuda::to_i32_into(p.nt_hct, &[((t * HCT) as i32)]);
            cuda::to_i32_into(p.nt_hc, &[((t * HCN) as i32)]);
            cuda::to_i32_into(p.nt_6144, &[((t * GDN_VAL) as i32)]);
            cuda::to_i32_into(p.nt_combo, &[((t * TOPK * INTER) as i32)]);
            cuda::to_i32_into(self.pf_ncombo, &[(t * TOPK) as i32]);

            // embeddings → [t][10240] (each token row replicated over 4 HC streams)
            let mut h_host = vec![0f32; t * HCT];
            for (i, &id) in chunk.iter().enumerate() {
                let row = &self.w.embed_host[id as usize * H..(id as usize + 1) * H];
                for g in 0..HCN {
                    for (dst, &b) in h_host[i * HCT + g * H..i * HCT + (g + 1) * H].iter_mut().zip(row) {
                        *dst = f32::from_bits((b as u32) << 16);
                    }
                }
            }
            cuda::to_f32_into(self.s.h, &h_host);

            // copy-engine prefetch of the cold tier, two layers ahead (grouped
            // prefill path only; decode-sized chunks use the staged path)
            let dma = self.pf_ring_gu[0] != 0 && mma_on() && pf_gemm_on() && !(stage_on() && t * TOPK <= self.stage.max);
            self.pf_dma_live.set(dma);
            if dma {
                self.pf_issue(0);
                self.pf_issue(1);
            }
            // 48 decoder layers
            for l in 0..LAYERS {
                if std::env::var("ENGINE_DEBUG_NAN").is_ok() && l <= 3 {
                    cuda::sync();
                    let hst = cuda::dtoh(self.s.h, t * HCT);
                    let ni = hst.iter().filter(|x| x.is_infinite()).count();
                    let nn = hst.iter().filter(|x| x.is_nan()).count();
                    let mx = hst.iter().filter(|x| x.is_finite()).fold(0f32, |a, &b| a.max(b.abs()));
                    eprintln!("[nanwatch] layer {l} ENTRY: nan={nn} inf={ni} max_abs={mx:.3e}");
                }
                if l == PLE_LAYER && self.cfg.ple {
                    // reference: hidden += ple(hidden, input_ids) at the TOP
                    // of layer 1's forward — host index math, spec 3.5
                    let prefix: Vec<i64> = if pos_base >= 2 {
                        self.history[pos_base - 2..pos_base].to_vec()
                    } else {
                        vec![PLE_EOS; 2 - pos_base]
                    };
                    self.ple_run(cnq, t, &prefix, chunk, false);
                    if std::env::var("ENGINE_DEBUG_NAN").is_ok() {
                        cuda::sync();
                        let hst = cuda::dtoh(self.s.h, t * HCT);
                        let nn = hst.iter().filter(|x| x.is_nan()).count();
                        let ni = hst.iter().filter(|x| x.is_infinite()).count();
                        eprintln!("[nanwatch] layer 1 AFTER PLE: nan={nn} inf={ni}");
                    }
                }
                let (mut mixed, mut injw) = (self.s.mixed, self.s.injw);
                self.hc_run(&self.w.hc[l], self.s.h, t, mixed, injw);
                let dbg = std::env::var("ENGINE_DEBUG_SYNC").is_ok();
                if dbg {
                    eprintln!("[prefill layer {l}] enter");
                }
                let nan_watch = std::env::var("ENGINE_DEBUG_NAN").is_ok();
                // CROW_DUMP_H: layer-0 stage dumps (determinism bisect)
                let dump0 = |tag: &str, ptr: Dev, n: usize| {
                    if l == 0 { if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                        cuda::sync();
                        let v = cuda::dtoh(ptr, n);
                        let mut b = Vec::with_capacity(v.len() * 4);
                        for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                        std::fs::write(format!("{dir}/l0-{tag}.f32"), b).unwrap();
                    } }
                };
                dump0("mixed", mixed, t * H);
                dump0("injw", injw, t * HCN);
                let sub = match &self.w.sub[l] {
                    SubW::Gdn { .. } => self.gdn_prompt(l, mixed, t, first && start == 0),
                    SubW::Attn { .. } => self.attn_prompt(l, mixed, t, pos_base),
                };
                dump0("sub", sub, t * H);
                dump0("gdn-mq", self.s.mq, t * GDN_CONV);
                dump0("gdn-gq", self.s.gq, t * GDN_KEY);
                dump0("gdn-gv", self.s.gv, t * GDN_VAL);
                dump0("gdn-gz", self.s.gz, t * GDN_VAL);
                dump0("gdn-gb", self.s.gb, t * 48);
                dump0("gdn-ga", self.s.ga, t * 48);
                dump0("gdn-core", self.s.gcore, t * GDN_VAL);
                dump0("gdn-norm", self.s.gnorm, t * GDN_VAL);
                if dbg {
                    cuda::sync();
                    eprintln!("[prefill layer {l}] done");
                }
                if nan_watch {
                    cuda::sync();
                    let hst = cuda::dtoh(self.s.h, t * HCT);
                    let (mut nn, mut ni, mut mx) = (0usize, 0usize, 0f32);
                    let mut first_nan = 0usize;
                    let mut first_inf = 0usize;
                    for (i, &v) in hst.iter().enumerate() {
                        if v.is_nan() {
                            if nn == 0 { first_nan = i; }
                            nn += 1;
                        }
                        if v.is_infinite() {
                            if ni == 0 { first_inf = i; }
                            ni += 1;
                        }
                        if v.is_finite() { mx = mx.max(v.abs()); }
                    }
                    if nn > 0 || ni > 0 {
                        let row = first_nan / HCT;
                        let rest = first_nan % HCT;
                        let stream = rest / H;
                        let irow = first_inf / HCT;
                        eprintln!("[nanwatch] after layer {l}: nan={nn} inf={ni} max_abs={mx:.3e} first_nan idx={first_nan} (row {row}, stream {stream}) first_inf row {irow}");
                    } else {
                        eprintln!("[nanwatch] after layer {l}: nan=0 inf=0 max_abs={mx:.3e}");
                    }
                }
                // x1 = h + sub ⊗ injw
                launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
                    self.s.h as u64, sub as u64, self.s.injw as u64, self.s.x1 as u64]);
                if nan_watch && l == 0 {
                    cuda::sync();
                    let v = cuda::dtoh(self.s.x1, t * HCT);
                    let rowmax = |r: usize, w: usize| v[r * w..(r + 1) * w].iter().fold(0f32, |a, &b| a.max(b.abs()));
                    eprintln!("[nanwatch] l0 x1: nan={} inf={} rowmax4={:.3e} rowmax7={:.3e}",
                        v.iter().filter(|x| x.is_nan()).count(),
                        v.iter().filter(|x| x.is_infinite()).count(),
                        rowmax(4, HCT), rowmax(7, HCT));
                }
                self.hc_run(&self.w.hc2[l], self.s.x1, t, self.s.mixed_m, self.s.injw);
                if nan_watch && l == 0 {
                    cuda::sync();
                    let v = cuda::dtoh(self.s.mixed_m, t * H);
                    let rowmax = |r: usize| v[r * H..(r + 1) * H].iter().fold(0f32, |a, &b| a.max(b.abs()));
                    eprintln!("[nanwatch] l0 mixed_m: nan={} inf={} max0={:.3e} max4={:.3e} max7={:.3e}",
                        v.iter().filter(|x| x.is_nan()).count(),
                        v.iter().filter(|x| x.is_infinite()).count(),
                        rowmax(0), rowmax(4), rowmax(7));
                }
                if dma {
                    cuda::stream_wait_event(cuda::cur_stream(), self.pf_ev_filled[l % 2]);
                }
                dump0("moe-mixed_m", self.s.mixed_m, t * H);
                let moe = self.moe_run(l, self.s.mixed_m, t);
                dump0("moe-rlog", self.s.rlog, t * E);
                dump0("moe-rwts", self.s.rwts, t * TOPK);
                dump0("moe-h1", self.s.h1, t * TOPK * 2 * INTER);
                dump0("moe-eo", self.s.eo, t * TOPK * H);
                dump0("moe-out", moe, t * H);
                if l == 0 { if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                    // grouped-GEMM plan of layer 0: perm [t*10] i32, tiles [n_tiles] int4, rids [t*10]
                    cuda::sync();
                    let wr = |tag: &str, v: Vec<i32>| {
                        let mut b = Vec::with_capacity(v.len() * 4);
                        for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                        std::fs::write(format!("{dir}/l0-{tag}.i32"), b).unwrap();
                    };
                    let nt = cuda::dtoh_i32(self.stage.n_tiles, 1)[0] as usize;
                    wr("moe-perm", cuda::dtoh_i32(self.stage.perm, t * TOPK));
                    wr("moe-tiles", cuda::dtoh_i32(self.stage.tiles, nt * 4));
                    wr("moe-rids", cuda::dtoh_i32(self.s.rids, t * TOPK));
                } }
                if dma {
                    cuda::event_record(self.pf_ev_done[l % 2], cuda::cur_stream());
                    if l + 2 < LAYERS {
                        self.pf_issue(l + 2);
                    }
                }
                if nan_watch && l == 0 {
                    cuda::sync();
                    let v = cuda::dtoh(self.s.moe_out, t * H);
                    eprintln!("[nanwatch] l0 moe_out: nan={} inf={}",
                        v.iter().filter(|x| x.is_nan()).count(),
                        v.iter().filter(|x| x.is_infinite()).count());
                }
                launch_v(self.k.f("inject_residual"), 4, (t * 10) as u32, 1, 256, &[
                    self.s.x1 as u64, moe as u64, self.s.injw as u64, self.s.h as u64]);
                // CROW_DUMP_H=<dir>: per-layer residual-stream dump (determinism bisect)
                if let Ok(dir) = std::env::var("CROW_DUMP_H") {
                    cuda::sync();
                    let v = cuda::dtoh(self.s.h, t * HCT);
                    let mut b = Vec::with_capacity(v.len() * 4);
                    for x in &v { b.extend_from_slice(&x.to_le_bytes()); }
                    std::fs::write(format!("{dir}/h-chunk{start}-layer{l:02}.f32"), b).unwrap();
                }
            }
            self.pf_dma_live.set(false);
            self.done_blocks = (pos_base + t) / 4;
            self.pos = pos_base + t;
            self.history.extend_from_slice(chunk);
            start += t;
            let done = start;
            let el = pre_t0.elapsed().as_secs_f64().max(1e-9);
            eprintln!(
                "[prefill] {done}/{pre_total} tok — {:.1} tok/s ({:.0} s)",
                done as f64 / el,
                el
            );

            // head: mixer + lm_head (+ optional logits collection, + argmax last)
            self.head_run(t);
            if let Some(out) = collect_logits.as_deref_mut() {
                for i in 0..t {
                    self.lm_head_row(i);
                    launch_v(self.k.f("argmax_k"), 1, 1, 1, 1024, &[
                        self.s.logits as u64, self.s.argmax as u64, self.p.n_vocab as u64]);
                    cuda::sync();
                    let lg = cuda::dtoh(self.s.logits, V);
                    out.push(lg);
                }
            }
            self.lm_head_row(t - 1);
            launch_v(self.k.f("argmax_k"), 1, 1, 1, 1024, &[
                self.s.logits as u64, self.s.argmax as u64, self.p.n_vocab as u64]);
            cuda::sync();
        }
        let tok = cuda::dtoh_i32(self.s.argmax, 1)[0] as usize;
        tok
    }

    /// single decode step; returns the greedy token. Host touches: embedding
    /// row, PLE index math, decode-slot scalar, argmax readback.
    pub unsafe fn decode_step(&mut self, cnq: &mut Cnq, id: i64) -> usize {
        let pos = self.pos;
        let p = &self.p;
        let graph = graph_on();
        let prof = prof::on();
        let t0 = std::time::Instant::now();
        if prof {
            prof::STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
        if graph && self.cap_stream == 0 {
            self.cap_stream = cuda::stream_create_non_blocking() as u64;
            cuda::set_stream(self.cap_stream);
        }

        let bb_if_complete = if (pos + 1) % 4 == 0 { (pos + 1) / 4 - 1 } else { 0 };
        let ncb1 = (((pos + 1) / 4).min((self.st.context + 3) / 4)) as i32;
        // QSA block bookkeeping for this token: a block completes when
        // (pos+1) % 4 == 0; otherwise block_base names the NEXT (incomplete)
        // block = done_blocks, which the selector never scans (ncb = done_blocks)
        let complete = (pos + 1) % 4 == 0;
        let blk_base = if complete { (pos + 1) / 4 - 1 } else { self.done_blocks };
        let gdbg = graph && std::env::var("CROW_GRAPH_DBG").is_ok();
        let route_dump = std::env::var("CROW_ROUTE_DUMP").is_ok();
        if graph {
            // scalar refreshes from PINNED staging — async HtoD from pinned is
            // safe without sync (stack-sourced HtoD dies with the frame on a
            // non-blocking stream)
            let sb = self.scalar_stage.host as *mut i32;
            *sb.add(0) = 1;
            *sb.add(1) = 0;
            *sb.add(2) = pos as i32;
            *sb.add(3) = bb_if_complete as i32;
            *sb.add(4) = pos as i32;
            *sb.add(6) = ncb1;
            *sb.add(7) = pos as i32;
            *sb.add(8) = LOWRANK as i32;
            *sb.add(9) = HCT as i32;
            *sb.add(10) = HCN as i32;
            *sb.add(11) = (TOPK * INTER) as i32;
            *sb.add(12) = blk_base as i32;
            *sb.add(13) = if complete { 1 } else { 0 };
            for (dev, off) in [
                (p.t, 0usize), (p.init, 1), (p.pos_base, 2), (p.pos_base_b4, 3),
                (p.slot1, 4), (p.pos_row1, 7), (p.ncb1, 6),
                (p.nt_low, 8), (p.nt_hct, 9), (p.nt_hc, 10), (p.nt_combo, 11),
                (p.block_base, 12), (p.n_new, 13),
            ] {
                cuda::upload_from_pinned(dev, sb.add(off) as _, 4);
            }
            if prof {
                prof::add(&prof::SCALAR, t0.elapsed().as_micros() as u64);
            }
        } else {
            cuda::to_i32_into(p.t, &[1]);
            cuda::to_i32_into(p.init, &[0]);
            cuda::to_i32_into(p.pos_base, &[pos as i32]);
            cuda::to_i32_into(p.pos_base_b4, &[bb_if_complete as i32]);
            cuda::to_i32_into(p.slot1, &[pos as i32]);
            cuda::to_i32_into(p.ncb1, &[ncb1]);
            cuda::to_i32_into(p.pos_row1, &[pos as i32]);
            cuda::to_i32_into(p.nt_low, &[LOWRANK as i32]);
            cuda::to_i32_into(p.nt_hct, &[HCT as i32]);
            cuda::to_i32_into(p.nt_hc, &[HCN as i32]);
            cuda::to_i32_into(p.nt_combo, &[(TOPK * INTER) as i32]);
            if prof {
                prof::add(&prof::SCALAR, t0.elapsed().as_micros() as u64);
            }
        }

        // embedding row → [1][10240]; the staging buffer is engine-owned so
        // the async HtoD source outlives the call (graph-mode requirement)
        let t_emb = std::time::Instant::now();
        let row = &self.w.embed_host[id as usize * H..(id as usize + 1) * H];
        for g in 0..HCN {
            for (dst, &b) in self.embed_buf[g * H..(g + 1) * H].iter_mut().zip(row) {
                *dst = f32::from_bits((b as u32) << 16);
            }
        }
        cuda::to_f32_into(self.s.h, &self.embed_buf);
        if prof {
            prof::add(&prof::EMBED, t_emb.elapsed().as_micros() as u64);
        }

        if gdbg { eprintln!("[graph-dbg] M: vor PLE"); }
        // PLE host prep (the one host touch per token, spec 3.5); the kernels
        // run at PLE_LAYER inside the layer loop (#11, 2026-09-05)
        if self.cfg.ple {
            let t_ple = std::time::Instant::now();
            let prefix: Vec<i64> = if pos >= 2 {
                self.history[pos - 2..pos].to_vec()
            } else {
                vec![PLE_EOS; 2 - pos]
            };
            self.ple_run(cnq, 1, &prefix, &[id], true);
            if prof {
                prof::add(&prof::PLE, t_ple.elapsed().as_micros() as u64);
            }
        }

        if gdbg { eprintln!("[graph-dbg] N: nach PLE"); }
        // graph mode captures this kernel sequence once (stream capture —
        // the debug syncs above must stay off here) and replays it below
        let capturing = graph && self.graph_exec == 0;
        let replay = graph && self.graph_exec != 0;
        if capturing {
            if gdbg { eprintln!("[graph-dbg] O: begin_capture"); }
            cuda::begin_capture(self.cap_stream as cudarc::driver::sys::CUstream);
            if gdbg { eprintln!("[graph-dbg] P: capture status = {}", cuda::capture_status(self.cap_stream as cudarc::driver::sys::CUstream)); }
        }
        // ENGINE_DEBUG_NAN inserts syncs — incompatible with stream capture
        let nan_watch = !graph && std::env::var("ENGINE_DEBUG_NAN").is_ok();
        let t_layer = std::time::Instant::now();
        // replay: the captured graph carries the whole layer loop + head -
        // launching it eagerly as well would double the work and keep every
        // WDDM launch on the critical path
        let mut t_head = t_layer;
        if !replay {
        for l in 0..LAYERS {
            if l == PLE_LAYER && self.cfg.ple {
                // reference order: hidden += ple(hidden, ids) at the top of layer 1
                self.ple_step_kernels();
            }
            let t_hc = std::time::Instant::now();
            self.hc_run(&self.w.hc[l], self.s.h, 1, self.s.mixed, self.s.injw);
            let d_hc = t_hc.elapsed().as_micros() as u64;
            if prof {
                prof::add(&prof::HC, d_hc);
            }
            if nan_watch && l >= 11 && l <= 31 {
                cuda::sync();
                let v = cuda::dtoh(self.s.mixed, H);
                eprintln!("[nanwatch-decode] l{l} mixed: nan={} inf={}",
                    v.iter().filter(|x| x.is_nan()).count(),
                    v.iter().filter(|x| x.is_infinite()).count());
            }
            let t_sub = std::time::Instant::now();
            self.sb_pack[0] = blk_base as i32;
            self.sb_pack[1] = if complete { 1 } else { 0 };
            let sub = match &self.w.sub[l] {
                SubW::Gdn { .. } => self.gdn_step(l, self.s.mixed),
                SubW::Attn { .. } => self.attn_step(l, self.s.mixed, pos, &self.sb_pack, graph),
            };
            let d_sub = t_sub.elapsed().as_micros() as u64;
            if prof {
                if matches!(&self.w.sub[l], SubW::Gdn { .. }) {
                    prof::add(&prof::SUB_GDN, d_sub);
                } else {
                    prof::add(&prof::SUB_ATTN, d_sub);
                }
            }
            launch_v(self.k.f("inject_residual"), 4, 10, 1, 256, &[
                self.s.h as u64, sub as u64, self.s.injw as u64, self.s.x1 as u64]);
            self.hc_run(&self.w.hc2[l], self.s.x1, 1, self.s.mixed_m, self.s.injw);
            let t_moe = std::time::Instant::now();
            let moe = self.moe_run(l, self.s.mixed_m, 1);
            if prof {
                prof::add(&prof::MOE, t_moe.elapsed().as_micros() as u64);
            }
            if !graph && route_dump {
                cuda::sync();
                let ids = cuda::dtoh_i32(self.s.rids, TOPK);
                let mut a = [0i32; 10];
                a.copy_from_slice(&ids[..10]);
                if l == 0 { self.route_log.push(Vec::with_capacity(LAYERS)); }
                self.route_log.last_mut().unwrap().push(a);
            }
            launch_v(self.k.f("inject_residual"), 4, 10, 1, 256, &[
                self.s.x1 as u64, moe as u64, self.s.injw as u64, self.s.h as u64]);
        }
        t_head = std::time::Instant::now();
        self.head_run(1);
        self.lm_head_row(0);
        launch_v(self.k.f("argmax_k"), 1, 1, 1, 1024, &[
            self.s.logits as u64, self.s.argmax as u64, self.p.n_vocab as u64]);
        // #20: the device sampler overwrites the argmax slot with its draw
        if let Some(ds) = &self.dev_sampler {
            self.launch_sample(ds);
            if capturing { ds.in_graph.set(true); }
        }
        } // !replay
        if gdbg { eprintln!("[graph-dbg] Q: layer-loop fertig, capture status = {}", cuda::capture_status(self.cap_stream as cudarc::driver::sys::CUstream)); }
        if capturing {
            self.graph_exec = cuda::end_capture_instantiate(self.cap_stream as cudarc::driver::sys::CUstream) as u64;
            if gdbg { eprintln!("[graph-dbg] R: instantiated"); }
            // execute what was just captured — during capture the launches are
            // recorded, not run, so token 1 would otherwise produce nothing
            cuda::launch_graph(
                self.graph_exec as cudarc::driver::sys::CUgraphExec,
                self.cap_stream as cudarc::driver::sys::CUstream,
            );
        } else if graph {
            cuda::launch_graph(
                self.graph_exec as cudarc::driver::sys::CUgraphExec,
                self.cap_stream as cudarc::driver::sys::CUstream,
            );
            // #20: sampler enabled after the capture -> eager launch behind the replay
            if let Some(ds) = &self.dev_sampler {
                if !ds.in_graph.get() { self.launch_sample(ds); }
            }
        }
        // #63b: the parked trickle copies go out HERE, after the launch of
        // this token (graph replay, capture, or the eager kernels above) and
        // before the end-of-step sync, so they overlap the replay instead of
        // blocking it. No-op under CROW_TRICKLE_DEFER=0: nothing is parked.
        // Under CROW_PROFILE=1 the host time of this issue (about 0.24 ms per
        // token, 63c review I2) lands in the HEAD bucket, not in TAIL.
        self.trickle_drain_after_launch();
        if prof {
            prof::add(&prof::HEAD, t_head.elapsed().as_micros() as u64);
        }
        let t_tail = std::time::Instant::now();
        if gdbg { eprintln!("[graph-dbg] T1: vor sync"); }
        cuda::sync();
        if gdbg { eprintln!("[graph-dbg] T2: sync ok"); }
        let _prof_tok = cuda::dtoh_i32(self.s.argmax, 1)[0] as usize;
        if gdbg { eprintln!("[graph-dbg] T3: dtoh ok"); }
        if prof {
            prof::add(&prof::TAIL, t_tail.elapsed().as_micros() as u64);
        }

        self.pos = pos + 1;
        self.history.push(id);
        if complete {
            self.done_blocks = (pos + 1) / 4;
        }
        let tok = cuda::dtoh_i32(self.s.argmax, 1)[0] as usize;
        tok
    }

    /// #20: switch this answer to device-side sampling with the profile of `s`
    /// (mask cleared, rng = the sampler's seeded state). Call after prefill and
    /// before the first decode_step of the process so the node is captured with
    /// the graph; enabled later it runs as an eager launch behind each replay.
    pub unsafe fn enable_dev_sampler(&mut self, s: &crate::sample::Sampler) {
        let ds = self.dev_sampler.get_or_insert_with(|| DevSampler {
            mask: cuda::alloc_zeroed(V),
            rng: cuda::alloc_zeroed(8),
            params: cuda::alloc_zeroed(16),
            cand_v: cuda::alloc_zeroed(64 * 64 * 4),
            cand_i: cuda::alloc_zeroed(64 * 64 * 4),
            in_graph: std::cell::Cell::new(false),
        });
        let zero = vec![0u8; V];
        cuda::upload_into(ds.mask, &zero);
        cuda::to_u64_into(ds.rng, &[s.rng.state()]);
        let mut pb = [0u8; 16];
        pb[0..4].copy_from_slice(&s.temperature.to_le_bytes());
        pb[4..8].copy_from_slice(&s.top_p.to_le_bytes());
        pb[8..12].copy_from_slice(&s.presence_penalty.to_le_bytes());
        pb[12..16].copy_from_slice(&(s.top_k as i32).to_le_bytes());
        cuda::upload_into(ds.params, &pb);
        cuda::sync();
        // invariant: in_graph is true only while the CURRENT decode graph carries the
        // sampler node; arming re-arms for the next capture, so clear it here
        ds.in_graph.set(false);
    }

    unsafe fn launch_sample(&self, ds: &DevSampler) {
        // v2: 64 blocks pick their slice's top-k, one block merges and draws
        launch_v(self.k.f("sample_topk_part"), 64, 1, 1, 256, &[
            self.s.logits as u64, self.p.n_vocab as u64, ds.mask as u64, ds.params as u64,
            ds.cand_v as u64, ds.cand_i as u64]);
        launch_v(self.k.f("sample_k"), 1, 1, 1, 256, &[
            ds.cand_v as u64, ds.cand_i as u64, self.s.argmax as u64, self.p.n_vocab as u64,
            ds.mask as u64, ds.rng as u64, ds.params as u64]);
    }

    /// #20: draw a token from the logits row currently in `s.logits` (the last
    /// prefill row) with the device sampler; eager launch + readback.
    pub unsafe fn sample_last(&mut self) -> usize {
        let ds = self.dev_sampler.as_ref().expect("sample_last: device sampler not enabled");
        self.launch_sample(ds);
        cuda::sync();
        cuda::dtoh_i32(self.s.argmax, 1)[0] as usize
    }

    /// Prompt-adaptive residency (A-P3): re-cut every layer's hot set from the
    /// selection counts accumulated so far (the prompt's own routing), swapping
    /// the least-selected resident experts for the most-selected absent ones.
    /// `max_swaps` bounds the work per layer (0 = unbounded). Returns swaps done.
    /// Needs CROW_COLD_TIER (full low-bit tier). Stream-ordered; call between
    /// prefill and decode (never inside the captured graph).
    pub unsafe fn adapt_hot_set(&mut self, max_swaps: usize) -> usize {
        if self.res.lb.is_some() && !self.res.full {
            eprintln!("[adapt] cold-only low-bit tier: hot experts have no record to fall back to - no adaptation");
            return 0;
        }
        if let Some(tr) = &self.trickle {
            assert!(tr.in_a.is_empty() && tr.in_b.is_empty(), "adapt_hot_set while stream swaps are in flight");
        }
        let counts = self.drain_sel_counts();
        self.adapt_base = counts.concat();
        self.adapt_ema = vec![0.0; LAYERS * E];
        let nblk_gu = cuda::to_i32_dev(&[((2 * INTER * H) / 64) as i32]);
        let nblk_dn = cuda::to_i32_dev(&[((H * INTER) / 64) as i32]);
        let mut total = 0usize;
        let none = std::collections::HashSet::new();
        for l in 0..LAYERS {
            let plan = self.res.plan_swaps(l, &counts[l], max_swaps, &none);
            for &(slot, evict, new_id) in &plan {
                self.res.swap_in(&self.k, l, slot, evict, new_id, nblk_gu, nblk_dn);
                total += 1;
            }
            if !plan.is_empty() {
                self.res.flush_layer_tables(l);
            }
        }
        cuda::sync();
        let (mut a, mut b) = (nblk_gu, nblk_dn);
        cuda::free_dev(&mut a);
        cuda::free_dev(&mut b);
        total
    }

    /// Decode-time adaptation tick (#17). Default: the cumulative re-cut
    /// (`adapt_hot_set`), whose counts stay dominated by the prompt. With
    /// CROW_ADAPT_WINDOW=1 the plan uses an exponentially decayed count of the
    /// selections SINCE the last tick (decay CROW_ADAPT_DECAY, default 0.5), so
    /// the hot set follows the answer's routing instead of the prompt's. Swaps
    /// are the same exact three-way exchange; numerics are untouched.
    /// CROW_ADAPT_WINDOW=1: per-layer u64 view of the decayed selection window
    /// (selections since the last tick, decayed by CROW_ADAPT_DECAY), scaled by
    /// 1024 for the planner's integer ranking. None when the window is off.
    pub unsafe fn window_counts(&mut self) -> Option<Vec<Vec<u64>>> {
        if std::env::var("CROW_ADAPT_WINDOW").as_deref() != Ok("1") {
            return None;
        }
        let decay: f64 = std::env::var("CROW_ADAPT_DECAY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let now = self.drain_sel_counts().concat();
        if self.adapt_base.len() != now.len() {
            self.adapt_base = now.clone();
            self.adapt_ema = vec![0.0; now.len()];
        }
        for i in 0..now.len() {
            let d = now[i].saturating_sub(self.adapt_base[i]) as f64;
            self.adapt_ema[i] = self.adapt_ema[i] * decay + d;
        }
        self.adapt_base = now;
        Some((0..LAYERS).map(|l| self.adapt_ema[l * E..(l + 1) * E].iter().map(|v| (v * 1024.0) as u64).collect()).collect())
    }

    pub unsafe fn adapt_tick(&mut self, max_swaps: usize) -> usize {
        if std::env::var("CROW_ADAPT_WINDOW").as_deref() != Ok("1") {
            return self.adapt_hot_set(max_swaps);
        }
        if self.res.lb.is_some() && !self.res.full {
            return 0;
        }
        let decay: f64 = std::env::var("CROW_ADAPT_DECAY").ok().and_then(|v| v.parse().ok()).unwrap_or(0.5);
        let now = self.drain_sel_counts().concat();
        if self.adapt_base.len() != now.len() {
            self.adapt_base = now.clone();
            self.adapt_ema = vec![0.0; now.len()];
        }
        for i in 0..now.len() {
            let d = now[i].saturating_sub(self.adapt_base[i]) as f64;
            self.adapt_ema[i] = self.adapt_ema[i] * decay + d;
        }
        self.adapt_base = now;
        let nblk_gu = cuda::to_i32_dev(&[((2 * INTER * H) / 64) as i32]);
        let nblk_dn = cuda::to_i32_dev(&[((H * INTER) / 64) as i32]);
        let mut total = 0usize;
        let none = std::collections::HashSet::new();
        // #17: CROW_SWAP_BUNDLE=1 exchanges all pairs of the tick in one launch per
        // size class (48 layers x <= max_swaps pairs) instead of 6 memcpys per swap
        let bundle = swap_bundle_on() && self.res.lb.is_none();
        let (mut gu_pairs, mut dn_pairs) = (Vec::new(), Vec::new());
        let mut touched: Vec<usize> = Vec::new();
        for l in 0..LAYERS {
            // plan_swaps ranks by u64 counts: scale the decayed window by 1024
            let c: Vec<u64> = self.adapt_ema[l * E..(l + 1) * E].iter().map(|v| (v * 1024.0) as u64).collect();
            let plan = self.res.plan_swaps(l, &c, max_swaps, &none);
            for &(slot, evict, new_id) in &plan {
                if bundle {
                    self.res.swap_in_bundled(l, slot, evict, new_id, &mut gu_pairs, &mut dn_pairs);
                } else {
                    self.res.swap_in(&self.k, l, slot, evict, new_id, nblk_gu, nblk_dn);
                }
                total += 1;
            }
            if !plan.is_empty() {
                if bundle { touched.push(l); } else { self.res.flush_layer_tables(l); }
            }
        }
        if bundle {
            self.res.swap_pairs_launch(&self.k, &gu_pairs, &dn_pairs);
            for &l in &touched { self.res.flush_layer_tables(l); }
        }
        cuda::sync();
        let (mut a, mut b) = (nblk_gu, nblk_dn);
        cuda::free_dev(&mut a);
        cuda::free_dev(&mut b);
        total
    }

    /// Stream-side trickle (A-P3c): one tick per token boundary, BEFORE the
    /// token's graph launch. Commits the swaps whose copies landed (table
    /// flips on the compute stream, ordered after the side stream's event),
    /// then issues phase B for the just-committed ones and - when `plan` -
    /// phase A for up to `max_per_layer` new swaps per layer into the free
    /// spare slots, all on the side stream, overlapping the next token's
    /// compute. Exact three-way exchange as `swap_in`, host RAM never grows.
    /// Returns the number of new swaps started.
    pub unsafe fn trickle_tick(&mut self, plan: bool, max_per_layer: usize) -> usize {
        assert!(self.res.lb.is_none(), "stream trickle: exact NVFP4 tier only");
        assert!(self.res.stride > self.res.n, "stream trickle needs CROW_ADAPT_STREAM=1 (spare hot slots)");
        // #63b: a tick whose `decode_step` never ran (EOS stop in
        // `bin/decode.rs:233`) leaves a parked batch; issue it FIRST, so this
        // tick's wait on `ev_side` and its commits see the copies they need.
        if self.trickle_pend.is_some() {
            self.trickle_drain_after_launch();
        }
        let cap = cuda::cur_stream();
        let mut tr = self.trickle.take().unwrap_or_else(|| Trickle {
            stream: cuda::stream_create_non_blocking(),
            ev_side: cuda::event_create(),
            ev_commit: cuda::event_create(),
            in_a: Vec::new(),
            in_b: Vec::new(),
            swaps: 0,
        });
        // the compute stream must see the side stream's copies before the
        // table flips below (and before the next graph reads the new slots)
        if !tr.in_a.is_empty() || !tr.in_b.is_empty() {
            cuda::stream_wait_event(cap, tr.ev_side);
        }
        let mut touched: std::collections::HashSet<usize> = std::collections::HashSet::new();
        for p in tr.in_b.drain(..) {
            self.res.swap_commit_b(&p);
            touched.insert(p.l);
        }
        let to_b: Vec<PendingSwap> = tr.in_a.drain(..).collect();
        for p in &to_b {
            self.res.swap_commit_a(p);
            touched.insert(p.l);
        }
        // new swaps into the free spares (slots of swaps in flight are excluded)
        let mut new_a: Vec<(usize, usize, u32)> = Vec::new();
        if plan {
            let counts = self.window_counts().unwrap_or_else(|| self.drain_sel_counts());
            for l in 0..LAYERS {
                let free = self.res.spare_free[l].len();
                if free == 0 { continue; }
                let k = if max_per_layer == 0 { free } else { max_per_layer.min(free) };
                let excl: std::collections::HashSet<usize> = to_b.iter().filter(|p| p.l == l)
                    .flat_map(|p| [p.evict_slot, p.spare]).collect();
                for (slot, _evict, new_id) in self.res.plan_swaps(l, &counts[l], k, &excl) {
                    new_a.push((l, slot, new_id));
                }
            }
        }
        if touched.len() >= 8 {
            self.res.flush_all_tables();
        } else {
            for &l in &touched {
                self.res.flush_layer_tables(l);
            }
        }
        if !to_b.is_empty() || !new_a.is_empty() {
            // side stream waits for the flips (and thus for every graph that
            // could still read the slots about to be overwritten)
            cuda::event_record(tr.ev_commit, cap);
            if trickle_defer_on() {
                // #63b: park phase B and phase A; `decode_step` issues them
                // behind its graph launch. `ev_commit` is already recorded on
                // the compute stream, so the side stream's queue order is the
                // same as below: wait ev_commit, phase B, phase A, ev_side.
                let started = new_a.len();
                tr.swaps += started;
                self.trickle = Some(tr);
                self.trickle_pend = Some(TrickleBatch { to_b, new_a });
                return started;
            }
            cuda::stream_wait_event(tr.stream, tr.ev_commit);
            for p in &to_b {
                self.res.swap_stream_b(p, tr.stream);
            }
            tr.in_b = to_b;
            let started = new_a.len();
            for (l, slot, new_id) in new_a {
                let p = self.res.swap_stream_a(l, slot, new_id, tr.stream);
                tr.in_a.push(p);
            }
            cuda::event_record(tr.ev_side, tr.stream);
            tr.swaps += started;
            self.trickle = Some(tr);
            started
        } else {
            self.trickle = Some(tr);
            0
        }
    }

    /// #63b (the deferred order, default since #63c): issue the parked batch,
    /// on the side stream, AFTER the graph launch of the same token, so the
    /// copies overlap the replay instead of holding the one async copy engine
    /// while the graph launch waits behind the scalar refreshes (63a F5).
    /// No-op under `CROW_TRICKLE_DEFER=0`: nothing is ever parked.
    ///
    /// Safety, unchanged from the eager form (63a review, CUDA stream order
    /// follows ENQUEUE order per stream, not host wall clock):
    /// - the table flip happened before `ev_commit`, which the side stream
    ///   waits for here, so no graph reads a slot being overwritten
    /// - phase A writes only spare slots, which no table points at
    /// - phase B reads the evicted hot slot, which stays readable until
    ///   commit B at the next tick, and writes the pinned slot the incoming
    ///   expert vacated, which no table points at since commit A
    /// - the compute stream still waits `ev_side` at the next tick before the
    ///   commits and the flush, and `ev_side` is recorded here, strictly
    ///   before `decode_step` returns and before the next tick runs
    pub unsafe fn trickle_drain_after_launch(&mut self) {
        let b = match self.trickle_pend.take() {
            Some(b) => b,
            None => return,
        };
        let mut tr = self.trickle.take().expect("parked trickle batch without a trickle");
        cuda::stream_wait_event(tr.stream, tr.ev_commit);
        for p in &b.to_b {
            self.res.swap_stream_b(p, tr.stream);
        }
        tr.in_b = b.to_b;
        for (l, slot, new_id) in b.new_a {
            let p = self.res.swap_stream_a(l, slot, new_id, tr.stream);
            tr.in_a.push(p);
        }
        cuda::event_record(tr.ev_side, tr.stream);
        self.trickle = Some(tr);
    }

    /// finish the stream-side trickle: commit everything in flight (two more
    /// ticks without new plans) and drain the side stream
    pub unsafe fn trickle_drain(&mut self) -> usize {
        // #63b: a parked batch is still in flight for this purpose; issue it
        // before the emptiness test below, and after each of the two ticks.
        self.trickle_drain_after_launch();
        let mut total = 0;
        if let Some(tr) = &self.trickle {
            total = tr.swaps;
            if tr.in_a.is_empty() && tr.in_b.is_empty() {
                return total;
            }
        } else {
            return 0;
        }
        self.trickle_tick(false, 0);
        self.trickle_drain_after_launch();
        self.trickle_tick(false, 0);
        self.trickle_drain_after_launch();
        if let Some(tr) = &self.trickle {
            assert!(tr.in_a.is_empty() && tr.in_b.is_empty());
            cuda::stream_sync(tr.stream);
        }
        cuda::sync();
        total
    }

    /// DMA layer `l`'s pinned cold slab into ring slot l%2 on the side stream:
    /// waits until the compute stream released that slot (layer l-2 done),
    /// then records the slot's "filled" event for the compute stream.
    unsafe fn pf_issue(&self, l: usize) {
        let i = l % 2;
        cuda::stream_wait_event(self.pf_stream, self.pf_ev_done[i]);
        let (gu, dn) = (&self.res.cold_gu[l], &self.res.cold_dn[l]);
        cuda::memcpy_async_on(self.pf_ring_gu[i], gu.dev, gu.bytes, self.pf_stream);
        cuda::memcpy_async_on(self.pf_ring_dn[i], dn.dev, dn.bytes, self.pf_stream);
        cuda::event_record(self.pf_ev_filled[i], self.pf_stream);
    }

    /// drain per-layer [selections, cold] counters (control plane, between tokens)
    pub unsafe fn drain_counters(&self) -> Vec<[u64; 2]> {
        self.res.drain_counters()
    }
    /// drain per-expert selection counts (warm-up bookkeeping, [48][512])
    pub unsafe fn drain_sel_counts(&self) -> Vec<Vec<u64>> {
        let raw = cuda::dtoh_u64(self.sel_counts, LAYERS * E);
        (0..LAYERS).map(|l| raw[l * E..(l + 1) * E].to_vec()).collect()
    }
}

fn free_pw(p: &mut PW) {
    unsafe {
        match p {
            PW::Fp4(a, b) => { cuda::free_dev(a); cuda::free_dev(b); }
            PW::Bf16(w) => cuda::free_dev(w),
        }
    }
}

/// One engine per machine (2026-09-05): two engines starting together pin
/// 2 x 45 GiB on a 64 GB host and froze the machine twice on 2026-09-04. The
/// lock file carries the owner's PID; a stale lock (dead PID) is taken over,
/// a live one refuses the start BEFORE anything is pinned. CROW_LOCK=0 disables,
/// CROW_LOCK=<path> relocates (default: engine/.engine.lock next to the exe's crate).
fn engine_lock_path() -> Option<std::path::PathBuf> {
    match std::env::var("CROW_LOCK").as_deref() {
        Ok("0") => None,
        Ok(p) if !p.is_empty() => Some(std::path::PathBuf::from(p)),
        _ => Some(std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(".engine.lock")),
    }
}

fn pid_alive(pid: u32) -> bool {
    // tasklist prints the image line only for a live PID (Windows; no ptrace here)
    match std::process::Command::new("tasklist").args(["/FI", &format!("PID eq {pid}"), "/NH"]).output() {
        Ok(o) => String::from_utf8_lossy(&o.stdout).contains(&format!(" {pid} ")),
        Err(_) => true, // cannot tell: stay on the safe side
    }
}

fn engine_lock_acquire() {
    let Some(path) = engine_lock_path() else { return };
    let me = std::process::id();
    if let Ok(txt) = std::fs::read_to_string(&path) {
        if let Ok(pid) = txt.trim().parse::<u32>() {
            if pid != me && pid_alive(pid) {
                panic!(
                    "refusing to start: another engine (pid {pid}) holds {} — two engines pin 2 x 45 GiB and freeze this 64 GB machine (CROW_LOCK=0 overrides)",
                    path.display()
                );
            }
        }
    }
    std::fs::write(&path, format!("{me}
")).expect("engine lock file");
}

fn engine_lock_release() {
    let Some(path) = engine_lock_path() else { return };
    if let Ok(txt) = std::fs::read_to_string(&path) {
        if txt.trim().parse::<u32>().ok() == Some(std::process::id()) {
            let _ = std::fs::remove_file(&path);
        }
    }
}

impl Drop for Engine {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Engine"); }
        // #18 (2026-09-05): everything the field Drops do not cover — the graph
        // exec and capture stream, the staging slots and grouped-GEMM plan, the
        // prefetch rings and events, the routing counters, the trickle stream.
        // The planner of the next load in the same process reads cuMemGetInfo;
        // ~400 MB of slack decided between "fits" and "refuses".
        unsafe {
            cuda::sync();
            if self.graph_exec != 0 {
                cuda::graph_exec_destroy(self.graph_exec as cudarc::driver::sys::CUgraphExec);
                self.graph_exec = 0;
            }
            if self.cap_stream != 0 {
                cuda::set_stream(0);
                cuda::stream_destroy(self.cap_stream as cudarc::driver::sys::CUstream);
                self.cap_stream = 0;
            }
            if let Some(tr) = self.trickle.take() {
                cuda::stream_sync(tr.stream);
                cuda::event_destroy(tr.ev_side);
                cuda::event_destroy(tr.ev_commit);
                cuda::stream_destroy(tr.stream);
            }
            for e in self.pf_ev_filled.iter().chain(self.pf_ev_done.iter()) {
                cuda::event_destroy(*e);
            }
            if !self.pa_stream.is_null() {
                cuda::stream_sync(self.pa_stream);
                cuda::stream_destroy(self.pa_stream);
            }
            cuda::event_destroy(self.pa_ev_plan);
            for e in self.pa_ev_filled.iter().chain(self.pa_ev_done.iter()) {
                cuda::event_destroy(*e);
            }
            for d in self.pf_ring_gu.iter_mut().chain(self.pf_ring_dn.iter_mut()) {
                cuda::free_dev(d);
            }
            let st = &mut self.stage;
            for d in [&mut st.gu, &mut st.dn, &mut st.sgu, &mut st.sdn, &mut st.gu_b, &mut st.dn_b,
                      &mut st.counts, &mut st.offsets, &mut st.cursor, &mut st.perm, &mut st.tiles,
                      &mut st.n_tiles, &mut st.eptr, &mut st.grp, &mut st.tg, &mut st.max_tiles_p] {
                cuda::free_dev(d);
            }
            // #18: the pinned DMA scratch of CROW_STAGE_DMA belongs to the same rule
            if let Some(d) = st.dma.as_mut() {
                d.free();
            }
            cuda::free_dev(&mut self.pf_ncombo);
            cuda::free_dev(&mut self.sel_counts);
            if let Some(ds) = &mut self.dev_sampler {
                cuda::free_dev(&mut ds.mask);
                cuda::free_dev(&mut ds.rng);
                cuda::free_dev(&mut ds.params);
                cuda::free_dev(&mut ds.cand_v);
                cuda::free_dev(&mut ds.cand_i);
            }
            // the module last: every CUfunction in self.k points into it
            self.module.unload();
        }
        engine_lock_release();
    }
}

impl Drop for Weights {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Weights"); }
        unsafe {
            cuda::free_dev(&mut self.lm_head);
            cuda::free_dev(&mut self.mx_norm);
            free_pw(&mut self.mx_down);
            free_pw(&mut self.mx_up);
            for h in self.hc.iter_mut().chain(self.hc2.iter_mut()) {
                cuda::free_dev(&mut h.norm);
                free_pw(&mut h.down);
                free_pw(&mut h.up);
                for f in [&mut h.inj.w, &mut h.inj.gs] {
                    cuda::free_dev(f);
                }
            }
            for s in self.sub.iter_mut() {
                match s {
                    SubW::Gdn { qkv, conv, z, b, a, alog, dt, norm, out } => {
                        for f in [&mut qkv.w, &mut qkv.gs, &mut z.w, &mut z.gs, &mut b.w, &mut b.gs, &mut a.w, &mut a.gs, &mut out.w, &mut out.gs] {
                            cuda::free_dev(f);
                        }
                        cuda::free_dev(conv);
                        cuda::free_dev(alog);
                        cuda::free_dev(dt);
                        cuda::free_dev(norm);
                    }
                    SubW::Attn { q, k, v, o, qn, kn, iqk, iqln, ikln } => {
                        free_pw(q);
                        free_pw(k);
                        for f in [&mut v.w, &mut v.gs, &mut o.w, &mut o.gs, &mut iqk.w, &mut iqk.gs] {
                            cuda::free_dev(f);
                        }
                        cuda::free_dev(qn);
                        cuda::free_dev(kn);
                        cuda::free_dev(iqln);
                        cuda::free_dev(ikln);
                    }
                }
            }
            for m in self.moe.iter_mut() {
                cuda::free_dev(&mut m.router);
                // #18: the bf16 router twin (2.62 MB x 48 layers = 125.8 MB) was never
                // freed - the engine-side part of the 214 MB per reload (2026-09-05)
                cuda::free_dev(&mut m.router_bf);
                cuda::free_dev(&mut m.sgate);
                for f in [&mut m.sg.w, &mut m.sg.gs, &mut m.su.w, &mut m.su.gs, &mut m.sdn.w, &mut m.sdn.gs] {
                    cuda::free_dev(f);
                }
            }
        }
    }
}

impl Drop for Ple {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Ple"); }
        unsafe {
            for f in [&mut self.key.w, &mut self.key.gs, &mut self.value.w, &mut self.value.gs] {
                cuda::free_dev(f);
            }
            cuda::free_dev(&mut self.norm_key);
            cuda::free_dev(&mut self.norm_query);
            cuda::free_dev(&mut self.norm_conv);
            cuda::free_dev(&mut self.conv);
            cuda::free_dev(&mut self.cache);
            cuda::free_dev(&mut self.gs);
            cuda::free_dev(&mut self.state);
        }
    }
}

impl Drop for Params {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Params"); }
        unsafe {
            for f in [&mut self.n320, &mut self.n2560, &mut self.n640, &mut self.n6144, &mut self.n10240,
                      &mut self.n2048, &mut self.n12288, &mut self.n_conv_deq, &mut self.n_selmax,
                      &mut self.k_top, &mut self.cap, &mut self.tmax, &mut self.mode, &mut self.t,
                      &mut self.init, &mut self.pos_base, &mut self.pos_base_sb, &mut self.pos_base_b4, &mut self.slot_base,
                      &mut self.ncb, &mut self.pos_row, &mut self.block_base, &mut self.n_new,
                      &mut self.row_base, &mut self.n_rows, &mut self.slot1, &mut self.one,
                      &mut self.zero, &mut self.n_vocab, &mut self.nt_low, &mut self.nt_hct,
                      &mut self.nt_hc, &mut self.nt_6144, &mut self.nt_combo, &mut self.q_heads4,
                      &mut self.q_heads1, &mut self.pos_mul1, &mut self.pos_mul4, &mut self.stride512,
                      &mut self.stride128, &mut self.keys_ring, &mut self.qk_stride, &mut self.ncb1, &mut self.pos_row1,
                      &mut self.k_top10, &mut self.nt_hct1,
                      &mut self.nr4, &mut self.nr48, &mut self.nr512] {
                cuda::free_dev(f);
            }
        }
    }
}

impl Drop for Scratch {
    fn drop(&mut self) {
        unsafe { cuda::drop_dbg("before Scratch"); }
        unsafe {
            for f in [&mut self.h,
                      &mut self.part_o, // #18: the two attention-split partials were never freed
                      &mut self.part_ml,
                      &mut self.qsa_h1, // #61a: the CROW_QSA_PAR histogram
                      &mut self.emb,
                      &mut self.mixed,
                      &mut self.low,
                      &mut self.sil,
                      &mut self.mixw,
                      &mut self.normed,
                      &mut self.injr,
                      &mut self.injw,
                      &mut self.x1,
                      &mut self.mixed_m,
                      &mut self.moe_out,
                      &mut self.h1,
                      &mut self.h2,
                      &mut self.eo,
                      &mut self.xq_gu,
                      &mut self.xq_dn,
                      &mut self.xq_m,
                      &mut self.xq_v,
                      &mut self.xq_s,
                      &mut self.xq_e,
                      &mut self.sdown,
                      &mut self.sh12,
                      &mut self.sh2,
                      &mut self.sgv,
                      &mut self.rlog,
                      &mut self.rids,
                      &mut self.rwts,
                      &mut self.gu_ptrs,
                      &mut self.dn_ptrs,
                      &mut self.cold,
                      &mut self.mq,
                      &mut self.mq_t,
                      &mut self.cout_t,
                      &mut self.gq,
                      &mut self.gk,
                      &mut self.gv,
                      &mut self.gz,
                      &mut self.gb,
                      &mut self.ga,
                      &mut self.gbeta,
                      &mut self.gg,
                      &mut self.gqr,
                      &mut self.gkr,
                      &mut self.gcore,
                      &mut self.gnorm,
                      &mut self.gout,
                      &mut self.qg,
                      &mut self.aq,
                      &mut self.agate,
                      &mut self.aqn,
                      &mut self.aqr,
                      &mut self.ak,
                      &mut self.akn,
                      &mut self.akr,
                      &mut self.av,
                      &mut self.aout,
                      &mut self.agated,
                      &mut self.ay,
                      &mut self.qk,
                      &mut self.q_nrm,
                      &mut self.q_rot,
                      &mut self.pool_raw,
                      &mut self.pool_nrm,
                      &mut self.pool_rot,
                      &mut self.scores,
                      &mut self.sel,
                      &mut self.sel_n,
                      &mut self.ple_key,
                      &mut self.ple_kn,
                      &mut self.ple_val,
                      &mut self.ple_qn,
                      &mut self.ple_gate,
                      &mut self.ple_gs,
                      &mut self.ple_gated,
                      &mut self.ple_gn,
                      &mut self.ple_out,
                      &mut self.ple_slots,
                      &mut self.logits,
                      &mut self.argmax,
                      &mut self.mixed_final,
                      &mut self.scratch_gn] {
                cuda::free_dev(f);
            }
        }
    }
}
