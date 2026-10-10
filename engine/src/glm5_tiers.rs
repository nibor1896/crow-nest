//! #175 + #149 (GLM-5.3-Flash plan steps 16-17): the glm5_next routed experts in three tiers,
//! all 45 layers resident, token by token. glm5_next only: nothing on the `Engine` / `Geo` path
//! of Flash-Next or the 27B constructs or calls anything here (gate R, #174).
//!
//! - **Sizes** ([`tier_sizes`]): VRAM and pinned slots per MoE layer are the #159 planner's
//!   `hot` and `pinned` (`manager::plan_glm5_next`; RTX 5090 of record, 200,000 tokens, 3-bit
//!   record: 50 + 124, NVMe 114). A smaller figure may be asked for (the cache-size test, a
//!   smoke on a smaller machine); a larger one is refused by name.
//! - **Policy**: [`ExpertCache`] (#175) per layer (G1d's arena `layer`), LRU for the family
//!   (`expert_cache::policy_for`, `CROW_EXPERT_CACHE`; G1d chose `LRU s 0.00 P 0`: no seed, so
//!   the cache starts empty). Counters `[vram, pinned, nvme]` per MoE layer.
//! - **Physical slots** ([`LayerSlots`], [`serve`]): per MoE layer one VRAM arena of `vram`
//!   records and one pinned arena of `pinned` records (`Pinned::alloc_cold`, mapped: the MUL1
//!   kernels read a pinned record zero-copy). After the router of a call, the cache observes the
//!   selected ids; the tier change of every expert is then executed in three phases so that no
//!   slot is overwritten before it is read: (A) every expert entering VRAM, and every selected
//!   expert the policy leaves on NVMe, is staged into a VRAM staging slot (from its pinned slot,
//!   its old VRAM slot, or the container through `nvme_source` into a 4096-aligned landing
//!   buffer); barrier; (B) every expert entering pinned gets a freed pinned slot (D2H from its
//!   VRAM slot, or read from the container straight into the slot); (C) the VRAM entrants go
//!   from staging into freed VRAM slots. The `[E]` record table then points every selected id at
//!   its VRAM slot, its pinned slot (zero-copy) or its staging slot.
//! - **Pinned hits** (#188, [`PinnedUse`]): `CROW_GLM_PINNED=zerocopy` keeps a pinned hit in
//!   pinned (no promotion, the kernels read it zero-copy); `CROW_GLM_CPU_LANE=1` also computes a
//!   decode call's pinned ids on the CPU (`glm5_moe::lane`, posted by [`ExpertTiers::table_for`]);
//!   `CROW_GLM_CPU_LANE=split` computes only the pinned ids [`plan_split`] gives the CPU (the
//!   greedy min-max split of sybil's `nv2_host.cpp` `plan_and_reply`, cost model [`SplitCost`]),
//!   the rest zero-copy on the GPU. All off by default: the exchange rule above, bit for bit.
//! - **NVMe** (#149): [`NvmeSource`], one handle per reader, `FILE_FLAG_NO_BUFFERING`, 1 reader by
//!   default (B = 6.994 GB/s, PREREG amendment 5); every record (9,474,048 B at 3 bit) is located
//!   once at setup ([`ExpertRecord::glm5_table`]).
//! - **Driver** ([`Glm5Run`]): the dense part of every layer stays in VRAM
//!   (`glm5_model::load_layer_without_experts`), one KDA state per KDA layer and one MLA cache
//!   per DSA layer are swapped into the `Glm5Pass` around the layer's call, every generated row
//!   runs as one decode call, greedy head. The prompt rows run as decode calls too (`CROW_CHUNK`
//!   unset or 1), or (#186, `CROW_CHUNK=N`) in prompt calls of up to N rows ([`prompt_calls`],
//!   [`Glm5Run::prefill`]): one routing sync per MoE layer per call, the call's selection served
//!   through the prefill staging set in row sub-batches that fit it ([`serve_chunk`]).
//!
//! The cache decides only where a record is read from; the bytes are the container's, and the
//! MUL1 kernels read VRAM and pinned records alike (`glm5_moe_gpu_layer_matches_the_oracle`:
//! VRAM and pinned lanes bit-identical). So the logits must not depend on the cache size: the
//! GPU test [`tests::glm5_tiers_gpu_cache_size_is_invisible_in_the_logits`] holds that (plan
//! step 16 abort criterion, `docs/architecture.md` A9).

use crate::cnq::Cnq;
use crate::cuda::{self, Pinned};
use crate::expert_cache::{self, ExpertCache, Scope, Tier};
use crate::geo::{ExpertRecordSpec, Glm5Geo};
use crate::glm5_flags::{self, Feed, Readback, Routed, Switches};
use crate::glm5_graph;
use crate::glm5_head::Head;
use crate::glm5_kda::{KdaDims, KdaState};
use crate::glm5_mla::{MlaCache, MlaDims};
use crate::glm5_model::{self as gm, AttnKind, FfnKind, Glm5Pass, HeadW, LayerW, LoadReport};
use crate::glm5_moe::MoeGeo;
use crate::manager::TierPlan;
use crate::nvme_source::{ColdSource, ExpertRecord, NvmeConfig, NvmeSource, RecordDst, MAX_IN_FLIGHT};
use cudarc::driver::sys;

pub type Dev = cudarc::driver::sys::CUdeviceptr;

const NONE: u32 = u32::MAX;

// ---------------------------------------------------------------- sizes

/// slots per MoE layer
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TierSizes {
    pub vram: usize,
    pub pinned: usize,
}

/// The tier sizes per MoE layer: the #159 plan's `hot` and `pinned`, or a smaller figure asked
/// for; a figure above the plan is refused by name (the plan is what fits the card and the
/// pinned budget).
pub fn tier_sizes(plan: &TierPlan, vram: Option<usize>, pinned: Option<usize>) -> Result<TierSizes, String> {
    let pick = |what: &str, ask: Option<usize>, planned: usize| -> Result<usize, String> {
        match ask {
            None => Ok(planned),
            Some(n) if n <= planned => Ok(n),
            Some(n) => Err(format!(
                "{what} {n} per MoE layer is above the #159 plan's {planned}: the plan is what fits this card and the pinned budget (VRAM {} / pinned {} / NVMe {})",
                plan.hot, plan.pinned, plan.nvme
            )),
        }
    };
    Ok(TierSizes { vram: pick("--vram-slots", vram, plan.hot)?, pinned: pick("--pinned-slots", pinned, plan.pinned)? })
}

// ---------------------------------------------------------------- #188: how the pinned tier is read

/// `promote` (default) | `zerocopy`
pub const PINNED_ENV: &str = "CROW_GLM_PINNED";
/// `1` = the CPU lane, every pinned id of a decode call; `split` = the CPU lane on the pinned ids
/// [`plan_split`] gives the CPU; unset / `0` = off (default)
pub const CPU_LANE_ENV: &str = "CROW_GLM_CPU_LANE";

/// #188: what happens to a selected expert the cache holds in pinned. Default (both false): it
/// is promoted into VRAM (staged H2D, its VRAM victim back to pinned by D2H), today's rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinnedUse {
    /// `CROW_GLM_PINNED=zerocopy` (or implied by the CPU lane): a pinned hit stays in pinned
    /// ([`ExpertCache::set_pinned_stays`]); the GPU kernels read it zero-copy from its slot
    pub stay: bool,
    /// `CROW_GLM_CPU_LANE=1` (or `split`): in a decode call the selected ids in pinned are
    /// computed by the CPU from their slot (`glm5_moe::lane`), the others on the GPU; with
    /// `split` ([`ExpertTiers::split`]) only those [`plan_split`] gives the CPU
    pub cpu_lane: bool,
}

/// Parse the two #188 switches (`pinned` = `CROW_GLM_PINNED`, `lane` = `CROW_GLM_CPU_LANE`).
/// The CPU lane implies `zerocopy` (a promoted expert is no longer in pinned); asking for the
/// lane with an explicit `promote` is refused by name, as is any other value.
pub fn pinned_use(pinned: Option<&str>, lane: Option<&str>) -> Result<PinnedUse, String> {
    let promote_explicit = match pinned.map(str::trim) {
        None | Some("") => None,
        Some("promote") => Some(true),
        Some("zerocopy") => Some(false),
        Some(v) => return Err(format!("{PINNED_ENV}={v:?}: accepted promote (default), zerocopy")),
    };
    let cpu_lane = match lane.map(str::trim) {
        None | Some("") | Some("0") => false,
        Some("1") | Some("split") => true,
        Some(v) => return Err(format!("{CPU_LANE_ENV}={v:?}: accepted 0 (default), 1, split")),
    };
    if cpu_lane && promote_explicit == Some(true) {
        let v = lane.map(str::trim).unwrap_or_default();
        return Err(format!("{CPU_LANE_ENV}={v} reads the selected pinned experts where they lie, {PINNED_ENV}=promote moves them to VRAM first: unset {PINNED_ENV} or set it to zerocopy"));
    }
    Ok(PinnedUse { stay: cpu_lane || promote_explicit == Some(false), cpu_lane })
}

/// the CPU lane reads the pinned records on the CPU: refused by name on a write-combined arena
fn lane_on_wc(u: PinnedUse, wc: bool) -> Result<(), String> {
    if u.cpu_lane && wc {
        return Err(format!(
            "{CPU_LANE_ENV}=1: the pinned tier is write-combined (CROW_PINNED_ALLOC unset on Windows, or wc), which most CPUs cannot read efficiently (CUDA Driver API, cuMemHostAlloc, CU_MEMHOSTALLOC_WRITECOMBINED); set CROW_PINNED_ALLOC=host"
        ));
    }
    Ok(())
}

/// #188: the combos of one decode call for the CPU lane, pick order: a selected id served from
/// its pinned slot goes to the CPU (its host record), every other to the GPU (its table entry).
/// Returns the combos and the number on the CPU.
pub fn lane_combos(sel: &[i32], locs: &[(u32, Loc)], gpu_base: impl Fn(u32, Loc) -> u64, cpu_host: impl Fn(u32) -> *const u8) -> (Vec<crate::glm5_moe::lane::Combo>, usize) {
    lane_combos_where(sel, locs, gpu_base, cpu_host, |_| true)
}

/// [`lane_combos`] with the CPU restricted to the pinned ids `on_cpu` accepts (`split`): a pinned
/// id it refuses stays on the GPU, read zero-copy through its table entry.
pub fn lane_combos_where(
    sel: &[i32],
    locs: &[(u32, Loc)],
    gpu_base: impl Fn(u32, Loc) -> u64,
    cpu_host: impl Fn(u32) -> *const u8,
    on_cpu: impl Fn(u32) -> bool,
) -> (Vec<crate::glm5_moe::lane::Combo>, usize) {
    use crate::glm5_moe::lane::Combo;
    let mut n = 0;
    let combos = sel
        .iter()
        .map(|&e| {
            let loc = locs.iter().find(|x| x.0 == e as u32).expect("every selected id has a location").1;
            match loc {
                Loc::Pinned(q) if on_cpu(e as u32) => {
                    n += 1;
                    Combo::Cpu(cpu_host(q))
                }
                _ => Combo::Gpu(gpu_base(e as u32, loc)),
            }
        })
        .collect();
    (combos, n)
}

// ---------------------------------------------------------------- the CPU/GPU lane split

/// `CROW_GLM_CPU_LANE=split`: the lane takes only the pinned ids [`plan_split`] gives the CPU
/// (`1` keeps #188: every pinned id on the CPU). Values other than `split` are `pinned_use`'s.
pub fn lane_split(lane: Option<&str>) -> bool {
    lane.map(str::trim) == Some("split")
}

/// worker threads of the CPU lane's pool run (`glm5_moe::lane::THREADS`)
pub const LANE_THREADS_ENV: &str = "CROW_GLM_LANE_THREADS";

/// the lane's thread count under `CROW_GLM_CPU_LANE=split` when `CROW_GLM_LANE_THREADS` is
/// unset: the count whose best split was fastest in `glm5_tiers_gpu_split_cost_bench`
pub const SPLIT_LANE_THREADS: usize = 20;

/// Parse `CROW_GLM_LANE_THREADS` (`threads`) against `CROW_GLM_CPU_LANE` (`lane`): a whole
/// number from 1 to 256; unset or empty = `glm5_moe::LANE_THREADS` (8, the #188 lane as it was)
/// or, with `split`, [`SPLIT_LANE_THREADS`]; anything else refused by name.
pub fn lane_threads(threads: Option<&str>, lane: Option<&str>) -> Result<usize, String> {
    match threads.map(str::trim) {
        None | Some("") => Ok(if lane_split(lane) { SPLIT_LANE_THREADS } else { crate::glm5_moe::LANE_THREADS }),
        Some(v) => match v.parse::<usize>() {
            Ok(n) if (1..=256).contains(&n) => Ok(n),
            _ => Err(format!(
                "{LANE_THREADS_ENV}={v:?}: accepted a whole number of threads from 1 to 256 (unset: {}, or {SPLIT_LANE_THREADS} with {CPU_LANE_ENV}=split)",
                crate::glm5_moe::LANE_THREADS
            )),
        },
    }
}

/// The fixed cost model of the split (ms per MoE layer of one decode row), sybil's `GLM53_NV_POL`
/// (`g0, thit, tzc, ca, cb`, `glm53/nv2.py`) for this engine: the GPU lane costs `g0` (gather,
/// shared expert, launches: it runs in every layer, so unlike sybil's it is charged with no VRAM
/// hit too) + `thit` per selected id read from VRAM (slot or staging) + `tzc` per pinned id read
/// zero-copy over PCIe; the CPU lane `ca` (x to the host, the pool run's fixed part, the rows
/// back) + `cb` per expert. At most `maxcpu` ids per layer on the CPU (sybil: 32).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SplitCost {
    pub g0: f64,
    pub thit: f64,
    pub tzc: f64,
    pub ca: f64,
    pub cb: f64,
    pub maxcpu: usize,
}

impl SplitCost {
    /// RTX 5090 (PCIe 5.0 x16), Core Ultra 9 285K, 2 x 32 GB DDR5-5600, by the lane's thread
    /// count: `glm5_tiers_gpu_split_cost_bench` (2026-10-10), the real lane over 24 (VRAM, CPU)
    /// splits of one synthetic GLM layer at 8, 12, 16 and 20 threads, records rotated beyond the
    /// L3 and the GPU's L2; CPU and GPU zero-copy share the DRAM, so the fit takes `tzc` and `cb`
    /// under each other's load (`thit` 0.0148 from the solo VRAM run). Best split per V 0 / 2 /
    /// 4 / 6, ms: 8 threads 1.626 / 1.335 / 1.098 / 0.717, 12 1.675 / 1.349 / 1.022 / 0.630,
    /// 16 1.535 / 1.226 / 0.913 / 0.614, 20 1.503 / 1.261 / 0.841 / 0.572 (all-GPU V 0 2.08-2.47).
    pub const RTX5090_285K_BY_THREADS: [(usize, SplitCost); 4] = [
        (8, SplitCost { g0: 0.040, thit: 0.0148, tzc: 0.2441, ca: 0.140, cb: 0.4017, maxcpu: 32 }),
        (12, SplitCost { g0: 0.050, thit: 0.0148, tzc: 0.2589, ca: 0.110, cb: 0.3459, maxcpu: 32 }),
        (16, SplitCost { g0: 0.220, thit: 0.0148, tzc: 0.2737, ca: 0.360, cb: 0.2849, maxcpu: 32 }),
        (20, SplitCost { g0: 0.290, thit: 0.0148, tzc: 0.2367, ca: 0.350, cb: 0.2481, maxcpu: 32 }),
    ];

    /// the model at [`SPLIT_LANE_THREADS`], the split's default
    pub const RTX5090_285K: SplitCost = SplitCost::for_threads(SPLIT_LANE_THREADS);

    /// the calibrated model of the thread count nearest `n` (a tie to the lower count); a count
    /// outside 8..20 takes the nearest end, whose per-expert CPU cost no longer fits it
    pub const fn for_threads(n: usize) -> SplitCost {
        let t = &SplitCost::RTX5090_285K_BY_THREADS;
        let mut best = 0;
        let mut i = 1;
        while i < t.len() {
            if t[i].0.abs_diff(n) < t[best].0.abs_diff(n) {
                best = i;
            }
            i += 1;
        }
        t[best].1
    }
}

/// The CPU/GPU split of one decode call's MoE layer (sybil `nv2_host.cpp` `plan_and_reply`,
/// `glm53-flash-offload` @ `6769b27`, the loop over the RAM picks): `gpu_hits` selected ids the
/// GPU reads from VRAM, `ram` the selected ids in pinned with their heat (selections so far).
/// Colder ids first (lower heat, then lower id); each goes to the CPU when that does not raise
/// the layer's time `max(cpu, gpu)` above sending it to the GPU (`max(c + cb, g) <= max(c, g +
/// tzc)`), up to `maxcpu`; the other pinned ids are read zero-copy by the GPU, so both lanes
/// finish as close together as the per-expert steps allow. Returns the CPU ids, planning order.
pub fn plan_split(cost: &SplitCost, gpu_hits: usize, ram: &[(u32, u32)]) -> Vec<u32> {
    let mut order = ram.to_vec();
    order.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
    let mut g = cost.g0 + cost.thit * gpu_hits as f64;
    let mut c = if ram.is_empty() { 0.0 } else { cost.ca };
    let mut cpu = Vec::new();
    for &(e, _) in &order {
        let (ce, ge) = (c + cost.cb, g + cost.tzc);
        if cpu.len() < cost.maxcpu && ce.max(g) <= c.max(ge) {
            c = ce;
            cpu.push(e);
        } else {
            g = ge;
        }
    }
    cpu
}

// ---------------------------------------------------------------- #149 path B: the stager switch

/// `1` = the stager ([`ExpertTiers::set_stager`]); unset / anything else = off (default)
pub const STAGER_ENV: &str = "CROW_GLM_STAGER";

/// Parse `CROW_GLM_STAGER` (`stager`) against the switches it depends on: `1` turns it on (the
/// repo's `CROW_*` rule), and then it needs `CROW_GLM_FLAGS=1` (`flags`: the host takes a layer's
/// ids from the router's flag, not from a stream sync that would wait for the stager's copies
/// anyway) and refuses `CROW_GLM_CPU_LANE=1` (`lane`: the CPU reads the pinned records on the
/// host at launch time, before a landed flag could hold it back). Both refusals by name.
pub fn stager_on(stager: Option<&str>, flags: Option<&str>, lane: Option<&str>) -> Result<bool, String> {
    if stager != Some("1") {
        return Ok(false);
    }
    if flags != Some("1") {
        return Err(format!("{STAGER_ENV}=1 needs {}=1: the stager replaces the host's waits inside a MoE layer, the router's flag is the one wait it keeps", glm5_flags::ENV_FLAGS));
    }
    // the CPU lane reads its pinned records after the stager's moves of the call are done (the
    // stager's event, `lane::Call::ready`; under the controller `DevLane::serve` after it)
    let _ = lane;
    Ok(true)
}

/// #149 path B: the pinned bytes the stager holds for `moe_layers` x `experts` with `stage_cap`
/// staging slots of `record_bytes`: the landing (one record per staging slot) plus the table rows
/// and the landed flags (`[moe_layers][experts]` u64 each, rounded up to 4096 B). GLM-5.3-Flash,
/// 3-bit record, top-8: 75,792,384 + 2 x 98,304 = 75,988,992 B. [`plan_for_rows`] takes it off the
/// pinned budget when the stager is on.
pub fn stager_pinned_bytes(moe_layers: usize, experts: usize, stage_cap: usize, record_bytes: u64) -> u64 {
    stage_cap as u64 * record_bytes + 2 * (moe_layers * experts * 8).next_multiple_of(4096) as u64
}

// ---------------------------------------------------------------- slots and the mover (host)

/// Where a selected record is read from in this call.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Loc {
    Vram(u32),
    Pinned(u32),
    /// a VRAM staging slot: a record the policy did not cache (or evicted within the call)
    Stage(u32),
}

/// the destination of one NVMe read
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dst {
    /// the landing buffer of staging slot `i`
    Landing(u32),
    Pinned(u32),
}

/// The physical moves of [`serve`], in the order it issues them. The GPU side queues the
/// copies on the current stream; `barrier` waits for every copy queued so far; `nvme` returns
/// when its records are in their destinations.
pub trait Mover {
    /// read the records of `jobs` (expert of this layer, destination) from the container;
    /// at most `MAX_IN_FLIGHT`; returns the bytes read
    fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String>;
    fn landing_to_stage(&mut self, i: u32);
    fn pinned_to_stage(&mut self, q: u32, s: u32);
    fn vram_to_stage(&mut self, v: u32, s: u32);
    fn barrier(&mut self);
    fn vram_to_pinned(&mut self, v: u32, q: u32);
    fn stage_to_vram(&mut self, s: u32, v: u32);
    /// [`serve_chunk_prefill`]: sub-batch `j` of a prompt call starts its moves (default: nothing)
    fn begin_batch(&mut self, _j: usize) {}
    /// [`serve_chunk_prefill`]: the moves of the current sub-batch are all issued; the kernels
    /// queued after this read them (default: nothing, the moves are on the compute stream)
    fn end_batch(&mut self) {}
}

/// The slot of every expert of one MoE layer in its tier, and the free slots.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LayerSlots {
    pub vram_of: Vec<u32>,
    pub pin_of: Vec<u32>,
    free_vram: Vec<u32>,
    free_pin: Vec<u32>,
}

impl LayerSlots {
    pub fn new(experts: usize, s: TierSizes) -> LayerSlots {
        LayerSlots {
            vram_of: vec![NONE; experts],
            pin_of: vec![NONE; experts],
            // popped from the back: slot 0 first
            free_vram: (0..s.vram as u32).rev().collect(),
            free_pin: (0..s.pinned as u32).rev().collect(),
        }
    }
}

/// what one [`serve`] did
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Served {
    /// every selected id (ascending, once) and where its record is read from
    pub locs: Vec<(u32, Loc)>,
    pub nvme_reads: usize,
    pub nvme_bytes: u64,
    /// #187: the moves and tier changes of this call, counted on the host
    pub moves: Moves,
}

/// #187: host-side counts of what [`serve`] did, one record each (bytes = count x record
/// bytes). Counted where the moves are issued; no GPU call. Summed per MoE layer in
/// [`ExpertTiers::moves`] and per row in [`TokenReport::moves`].
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Moves {
    /// distinct selected ids (expert visits) of the call(s)
    pub visits: u64,
    /// selected ids the kernels read zero-copy from their pinned slot (over PCIe from host DRAM)
    pub zero_copy: u64,
    /// #188 (`CROW_GLM_CPU_LANE=1`): selected ids computed by the CPU lane from their pinned
    /// slot (not counted in `zero_copy`; no PCIe read of the record)
    pub cpu_lane: u64,
    /// NVMe reads into the pageable landing buffer (then H2D to staging)
    pub nvme_to_landing: u64,
    /// NVMe reads straight into a pinned slot
    pub nvme_to_pinned: u64,
    /// H2D: landing -> staging
    pub landing_to_stage: u64,
    /// H2D: pinned slot -> staging
    pub pinned_to_stage: u64,
    /// D2D: VRAM slot -> staging
    pub vram_to_stage: u64,
    /// D2H: VRAM slot -> pinned slot
    pub vram_to_pinned: u64,
    /// D2D: staging -> VRAM slot
    pub stage_to_vram: u64,
    /// tier transitions of the policy (any expert, selected or not): NVMe -> VRAM, pinned ->
    /// VRAM, NVMe -> pinned (promotions); VRAM -> pinned, VRAM -> NVMe, pinned -> NVMe (evictions)
    pub n2v: u64,
    pub p2v: u64,
    pub n2p: u64,
    pub v2p: u64,
    pub v2n: u64,
    pub p2n: u64,
}

impl Moves {
    /// field names in the order of [`Moves::fields`] (the JSON keys of `glm5_run`)
    pub const NAMES: [&'static str; 16] =
        ["visits", "zero_copy", "cpu_lane", "nvme_to_landing", "nvme_to_pinned", "landing_to_stage", "pinned_to_stage", "vram_to_stage", "vram_to_pinned", "stage_to_vram", "n2v", "p2v", "n2p", "v2p", "v2n", "p2n"];

    pub fn fields(&self) -> [u64; 16] {
        [
            self.visits,
            self.zero_copy,
            self.cpu_lane,
            self.nvme_to_landing,
            self.nvme_to_pinned,
            self.landing_to_stage,
            self.pinned_to_stage,
            self.vram_to_stage,
            self.vram_to_pinned,
            self.stage_to_vram,
            self.n2v,
            self.p2v,
            self.n2p,
            self.v2p,
            self.v2n,
            self.p2n,
        ]
    }

    fn fields_mut(&mut self) -> [&mut u64; 16] {
        [
            &mut self.visits,
            &mut self.zero_copy,
            &mut self.cpu_lane,
            &mut self.nvme_to_landing,
            &mut self.nvme_to_pinned,
            &mut self.landing_to_stage,
            &mut self.pinned_to_stage,
            &mut self.vram_to_stage,
            &mut self.vram_to_pinned,
            &mut self.stage_to_vram,
            &mut self.n2v,
            &mut self.p2v,
            &mut self.n2p,
            &mut self.v2p,
            &mut self.v2n,
            &mut self.p2n,
        ]
    }

    pub fn add(&mut self, o: &Moves) {
        for (x, y) in self.fields_mut().into_iter().zip(o.fields()) {
            *x += y;
        }
    }

    /// `self - o` field by field (a later snapshot minus an earlier one)
    pub fn since(&self, o: &Moves) -> Moves {
        let mut d = *self;
        for (x, y) in d.fields_mut().into_iter().zip(o.fields()) {
            *x -= y;
        }
        d
    }

    pub fn nvme_reads(&self) -> u64 {
        self.nvme_to_landing + self.nvme_to_pinned
    }

    pub fn promotions(&self) -> u64 {
        self.n2v + self.p2v + self.n2p
    }

    pub fn evictions(&self) -> u64 {
        self.v2p + self.v2n + self.p2n
    }

    /// records the mover copies host -> device (landing and pinned slots into staging)
    pub fn h2d(&self) -> u64 {
        self.landing_to_stage + self.pinned_to_stage
    }
}

/// One call of MoE layer `l` (the cache's layer index): `ids` the distinct selected experts,
/// ascending. The cache observes them (one tick); the tier changes are executed through `m`
/// (phases A, B, C of the module doc); returns where each selected record is.
pub fn serve(cache: &mut ExpertCache, l: usize, slots: &mut LayerSlots, ids: &[u32], stage_cap: usize, m: &mut dyn Mover) -> Result<Served, String> {
    let n = cache.experts;
    let tiers = |c: &ExpertCache| -> Vec<Tier> { (0..n as u32).map(|e| c.tier(l, e)).collect() };
    let before = tiers(cache);
    cache.observe_token(l, ids);
    let after = tiers(cache);
    let staged = staged_of(&before, &after, ids);
    if staged.len() > stage_cap {
        return Err(format!("expert tiers: layer {l} stages {} records in one call, {stage_cap} staging slots", staged.len()));
    }
    let mut out = Served::default();
    out.moves.visits = ids.len() as u64;
    for (b, a) in before.iter().zip(&after) {
        let mv = &mut out.moves;
        match (b, a) {
            (Tier::Nvme, Tier::Vram) => mv.n2v += 1,
            (Tier::Pinned, Tier::Vram) => mv.p2v += 1,
            (Tier::Nvme, Tier::Pinned) => mv.n2p += 1,
            (Tier::Vram, Tier::Pinned) => mv.v2p += 1,
            (Tier::Vram, Tier::Nvme) => mv.v2n += 1,
            (Tier::Pinned, Tier::Nvme) => mv.p2n += 1,
            _ => {}
        }
    }
    // phase A: into staging, from pinned / VRAM (queued) or the container (via the landing buffer)
    let mut from_nvme = Vec::new();
    for (s, &e) in staged.iter().enumerate() {
        match before[e as usize] {
            Tier::Vram => {
                m.vram_to_stage(slots.vram_of[e as usize], s as u32);
                out.moves.vram_to_stage += 1;
            }
            Tier::Pinned => {
                m.pinned_to_stage(slots.pin_of[e as usize], s as u32);
                out.moves.pinned_to_stage += 1;
            }
            Tier::Nvme => from_nvme.push((e, Dst::Landing(s as u32))),
        }
    }
    for c in from_nvme.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
    }
    for &(_, d) in &from_nvme {
        if let Dst::Landing(i) = d {
            m.landing_to_stage(i);
            out.moves.landing_to_stage += 1;
        }
    }
    m.barrier();
    // phase B: the pinned slots of experts that left pinned are free now (their bytes are staged
    // or not needed); the entrants take them
    for e in 0..n {
        if before[e] == Tier::Pinned && after[e] != Tier::Pinned {
            slots.free_pin.push(slots.pin_of[e]);
            slots.pin_of[e] = NONE;
        }
    }
    let mut to_pinned = Vec::new();
    for e in 0..n {
        if after[e] == Tier::Pinned && before[e] != Tier::Pinned {
            let q = slots.free_pin.pop().ok_or_else(|| format!("expert tiers: layer {l} has no free pinned slot for expert {e}"))?;
            slots.pin_of[e] = q;
            match before[e] {
                Tier::Vram => {
                    m.vram_to_pinned(slots.vram_of[e], q);
                    out.moves.vram_to_pinned += 1;
                }
                _ => to_pinned.push((e as u32, Dst::Pinned(q))),
            }
        }
    }
    for c in to_pinned.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
    }
    // phase C: the VRAM slots of experts that left VRAM (read in phase A / B, queued before)
    for e in 0..n {
        if before[e] == Tier::Vram && after[e] != Tier::Vram {
            slots.free_vram.push(slots.vram_of[e]);
            slots.vram_of[e] = NONE;
        }
    }
    for (s, &e) in staged.iter().enumerate() {
        if after[e as usize] == Tier::Vram {
            let v = slots.free_vram.pop().ok_or_else(|| format!("expert tiers: layer {l} has no free VRAM slot for expert {e}"))?;
            slots.vram_of[e as usize] = v;
            m.stage_to_vram(s as u32, v);
            out.moves.stage_to_vram += 1;
        }
    }
    out.nvme_reads = from_nvme.len() + to_pinned.len();
    out.moves.nvme_to_landing = from_nvme.len() as u64;
    out.moves.nvme_to_pinned = to_pinned.len() as u64;
    out.moves.zero_copy = ids.iter().filter(|&&e| after[e as usize] == Tier::Pinned).count() as u64;
    out.locs = ids
        .iter()
        .map(|&e| {
            let loc = match after[e as usize] {
                Tier::Vram => Loc::Vram(slots.vram_of[e as usize]),
                Tier::Pinned => Loc::Pinned(slots.pin_of[e as usize]),
                Tier::Nvme => Loc::Stage(staged.iter().position(|&x| x == e).expect("a selected NVMe expert is staged") as u32),
            };
            (e, loc)
        })
        .collect();
    Ok(out)
}

/// #187: a cache and its slot maps as new: every expert on NVMe, every slot free. The bytes
/// left in the arenas are referenced by no table entry afterwards (the table of a call is built
/// from the slot maps), so the next call of a layer reads its selection as a cold one.
pub fn reset_cache(cache: &mut ExpertCache, slots: &mut [LayerSlots], sizes: TierSizes) -> Result<(), String> {
    let stays = cache.pinned_stays();
    *cache = ExpertCache::new(cache.policy, cache.scope, cache.layers, cache.experts, cache.vram, cache.pinned)?;
    cache.set_pinned_stays(stays);
    for s in slots.iter_mut() {
        *s = LayerSlots::new(cache.experts, sizes);
    }
    Ok(())
}

/// the distinct ids of a `[t][k]` selection, ascending (the order `glm_tier_sim` replays)
pub fn distinct_ids(sel: &[i32], experts: usize) -> Result<Vec<u32>, String> {
    let mut v = Vec::with_capacity(sel.len());
    for &e in sel {
        if e < 0 || e as usize >= experts {
            return Err(format!("expert tiers: the router selected id {e}, outside 0..{experts}"));
        }
        v.push(e as u32);
    }
    v.sort_unstable();
    v.dedup();
    Ok(v)
}

/// the records [`serve`] stages for a call that moves the tiers from `before` to `after`: every
/// expert entering VRAM (any id), then every selected id the policy leaves on NVMe
fn staged_of(before: &[Tier], after: &[Tier], ids: &[u32]) -> Vec<u32> {
    let mut staged: Vec<u32> = (0..before.len() as u32).filter(|&e| after[e as usize] == Tier::Vram && before[e as usize] != Tier::Vram).collect();
    staged.extend(ids.iter().copied().filter(|&e| after[e as usize] == Tier::Nvme));
    staged
}

/// #186: the records one [`serve`] of `ids` (distinct, ascending) in layer `l` would stage; the
/// cache is left as it is ([`ExpertCache::tiers_after`])
pub fn staged_count(cache: &ExpertCache, l: usize, ids: &[u32]) -> usize {
    let before: Vec<Tier> = (0..cache.experts as u32).map(|e| cache.tier(l, e)).collect();
    staged_of(&before, &cache.tiers_after(l, ids), ids).len()
}

/// #186: how many rows of a prompt call's selection `sel` (`[rows][k]` i32, from its first row)
/// one [`serve`] with `cap` staging slots takes: every row when their staged set fits, else the
/// largest power of two of rows that fits (the call sizes the plan books,
/// `manager::glm5_prompt_call_sizes`). Refused by name when one row does not fit.
pub fn fitting_rows(cache: &ExpertCache, l: usize, sel: &[i32], k: usize, cap: usize) -> Result<usize, String> {
    fitting_rows_by(cache, l, sel, k, cap, &|ids| staged_count(cache, l, ids))
}

/// [`fitting_rows`] with the staged records of a sub-batch's distinct ids counted by `staged`
fn fitting_rows_by(cache: &ExpertCache, l: usize, sel: &[i32], k: usize, cap: usize, staged: &dyn Fn(&[u32]) -> usize) -> Result<usize, String> {
    let t = sel.len() / k;
    if t == 0 || sel.len() != t * k {
        return Err(format!("expert tiers: layer {l}: a selection of {} ids is no [rows][{k}]", sel.len()));
    }
    let fits = |r: usize| -> Result<bool, String> { Ok(staged(&distinct_ids(&sel[..r * k], cache.experts)?) <= cap) };
    if fits(t)? {
        return Ok(t);
    }
    let mut r = 1usize << (usize::BITS - 1 - t.leading_zeros());
    if r == t {
        r /= 2;
    }
    while r >= 1 {
        if fits(r)? {
            return Ok(r);
        }
        r /= 2;
    }
    let one = staged(&distinct_ids(&sel[..k], cache.experts)?);
    Err(format!("expert tiers: layer {l}: one row stages {one} records, {cap} staging slots"))
}

/// Prefill never admits (glm53-flash-offload: "Prefill (large picks) runs with admission off:
/// ... the decode working set survives"): one sub-batch of a prompt call of MoE layer `l`. The
/// cache observes the VRAM hits only (their recency / score / reference bit, as a decode tick
/// marks them; a VRAM hit moves no tier under any policy), so no tier changes; every other
/// selected record is staged for the call - a pinned one from its slot, an NVMe one through the
/// landing buffer - and its VRAM and pinned slots stay as they are. No barrier: the moves are
/// the staging copies only, ordered on the stream.
pub fn serve_prefill(cache: &mut ExpertCache, l: usize, slots: &LayerSlots, ids: &[u32], stage_cap: usize, m: &mut dyn Mover) -> Result<Served, String> {
    let hits: Vec<u32> = ids.iter().copied().filter(|&e| cache.tier(l, e) == Tier::Vram).collect();
    cache.observe_token(l, &hits);
    if let Some(&e) = hits.iter().find(|&&e| cache.tier(l, e) != Tier::Vram) {
        return Err(format!("expert tiers: layer {l}: marking the VRAM hit {e} moved it out of VRAM"));
    }
    let staged: Vec<u32> = ids.iter().copied().filter(|&e| cache.tier(l, e) != Tier::Vram).collect();
    if staged.len() > stage_cap {
        return Err(format!("expert tiers: layer {l} stages {} records in one call, {stage_cap} staging slots", staged.len()));
    }
    let mut out = Served::default();
    out.moves.visits = ids.len() as u64;
    let mut from_nvme = Vec::new();
    for (s, &e) in staged.iter().enumerate() {
        match cache.tier(l, e) {
            Tier::Pinned => {
                m.pinned_to_stage(slots.pin_of[e as usize], s as u32);
                out.moves.pinned_to_stage += 1;
            }
            _ => from_nvme.push((e, Dst::Landing(s as u32))),
        }
    }
    // each read batch goes on to staging before the next lands (a landing ring may be smaller
    // than the sub-batch)
    for c in from_nvme.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
        for &(_, d) in c {
            if let Dst::Landing(i) = d {
                m.landing_to_stage(i);
            }
        }
    }
    out.nvme_reads = from_nvme.len();
    out.moves.nvme_to_landing = from_nvme.len() as u64;
    out.moves.landing_to_stage = from_nvme.len() as u64;
    let mut si = 0u32;
    out.locs = ids
        .iter()
        .map(|&e| {
            if cache.tier(l, e) == Tier::Vram {
                (e, Loc::Vram(slots.vram_of[e as usize]))
            } else {
                si += 1;
                (e, Loc::Stage(si - 1))
            }
        })
        .collect();
    Ok(out)
}

/// #186: one prompt call of MoE layer `l`: the `[t][k]` selection `sel` served in consecutive row
/// sub-batches, each one [`serve`] (one cache tick) of its rows' distinct ids whose staged set fits
/// `cap` ([`fitting_rows`]). `each(row0, rows, served)` sees every sub-batch in row order, right
/// after its records moved, and queues its experts; before every sub-batch after the first the
/// mover's barrier waits for the previous one's (they read the staging slots this one reuses).
/// Returns the number of sub-batches.
#[allow(clippy::too_many_arguments)]
pub fn serve_chunk(
    cache: &mut ExpertCache,
    l: usize,
    slots: &mut LayerSlots,
    sel: &[i32],
    k: usize,
    cap: usize,
    m: &mut dyn Mover,
    each: &mut dyn FnMut(usize, usize, &Served) -> Result<(), String>,
) -> Result<usize, String> {
    let t = sel.len() / k;
    let (mut r0, mut batches) = (0, 0);
    while r0 < t {
        if r0 > 0 {
            m.barrier();
        }
        let rows = fitting_rows(cache, l, &sel[r0 * k..], k, cap)?;
        let ids = distinct_ids(&sel[r0 * k..(r0 + rows) * k], cache.experts)?;
        let served = serve(cache, l, slots, &ids, cap, m)?;
        each(r0, rows, &served)?;
        batches += 1;
        r0 += rows;
    }
    Ok(batches)
}

/// [`serve_chunk`] for a prompt call: every sub-batch through [`serve_prefill`] (no admission,
/// staged set = the selected records not in VRAM) and no barrier before a later sub-batch. With
/// no tier change, nothing a queued sub-batch reads is moved: a later sub-batch's staging copies
/// and its table upload are ordered behind it on the stream, and its NVMe reads into the host
/// landing buffer (whose earlier copies the upload already completed) run while the GPU computes
/// the queued sub-batches.
#[allow(clippy::too_many_arguments)]
pub fn serve_chunk_prefill(
    cache: &mut ExpertCache,
    l: usize,
    slots: &LayerSlots,
    sel: &[i32],
    k: usize,
    cap: usize,
    m: &mut dyn Mover,
    each: &mut dyn FnMut(usize, usize, &Served) -> Result<(), String>,
) -> Result<usize, String> {
    let t = sel.len() / k;
    let (mut r0, mut batches) = (0, 0);
    while r0 < t {
        let rows = {
            let c = &*cache;
            fitting_rows_by(c, l, &sel[r0 * k..], k, cap, &|ids| ids.iter().filter(|&&e| c.tier(l, e) != Tier::Vram).count())?
        };
        let ids = distinct_ids(&sel[r0 * k..(r0 + rows) * k], cache.experts)?;
        m.begin_batch(batches);
        let served = serve_prefill(cache, l, slots, &ids, cap, m)?;
        m.end_batch();
        each(r0, rows, &served)?;
        batches += 1;
        r0 += rows;
    }
    Ok(batches)
}

/// #186: the calls of a prompt phase of `n` rows at up to `chunk` rows per call, `(first row,
/// rows)`: full chunks, then the remainder as one call (its FFN runs in the plan's sizes,
/// `Glm5Pass::call_with_expert_batches`). Chunk 1: one row each.
/// `CROW_GLM_ARENA` elastic, on the plan's own numbers: the VRAM expert slots per MoE layer during
/// decode. The plan's hot set `plan.hot` plus the elastic chunks (one layer's `plan.hot` slots,
/// `record` bytes each, shared by every layer in the global arena) that fit the VRAM the plan
/// leaves under its ceiling (`vram_ceiling - fixed_bytes - hot_bytes`), plus the prompt scratch
/// `scratch` when the prompt borrows it instead of keeping it booked, above
/// [`ARENA_RESERVE_BYTES`], at most `elastic_cap` bytes (`CROW_GLM_ARENA_ELASTIC_GB`).
/// Returns (elastic chunks, slots per layer).
pub fn decode_hot_per_layer(plan: &crate::manager::TierPlan, scratch: u64, borrowed: bool, record: u64, moe_layers: usize, elastic_cap: u64) -> (u64, f64) {
    let left = plan.vram_ceiling.saturating_sub(plan.fixed_bytes + plan.hot_bytes()) + if borrowed { scratch } else { 0 };
    let cb = plan.hot as u64 * record;
    let n = if cb == 0 { 0 } else { left.saturating_sub(ARENA_RESERVE_BYTES).min(elastic_cap) / cb };
    (n, plan.hot as f64 + (n * plan.hot as u64) as f64 / moe_layers as f64)
}

/// `CROW_GLM_ARENA` elastic: the prompt phase borrows its scratch (`Glm5Run::prefill_with`) when
/// the arena is global with an elastic part (`CROW_GLM_ARENA_ELASTIC_GB` above 0) and the prompt
/// chunk is above the decode calls' rows; `get` reads the environment. A malformed value reads as
/// no borrow (`ExpertTiers::new` refuses it by name).
pub fn prompt_borrow_from_env(get: &dyn Fn(&str) -> Option<String>, chunk: usize, decode_t: usize) -> bool {
    chunk > decode_t && arena_kind(get(ARENA_ENV).as_deref()) == Ok(ArenaKind::Global) && arena_config(get).is_ok_and(|c| c.elastic_bytes > 0)
}

pub fn prompt_calls(n: usize, chunk: usize) -> Vec<(usize, usize)> {
    let c = chunk.max(1);
    let mut v: Vec<(usize, usize)> = (0..n / c).map(|i| (i * c, c)).collect();
    if n % c > 0 {
        v.push((n / c * c, n % c));
    }
    v
}

/// #186: the prompt chunk of the glm5_next path: `CROW_CHUNK` (`geo::chunk_from_env`), else 1 =
/// every prompt row one decode call (the path of record until the lever is measured)
pub fn prompt_chunk_from_env() -> usize {
    crate::geo::chunk_from_env().unwrap_or(1)
}

/// #186: the #159 plan of a glm5_next run of up to `rows` rows over a cache of `context` rows on
/// `vram_total` and `pinned_budget`, the expert record `record_bytes`: the prompt chunk
/// `Glm5Run::load` takes (`CROW_CHUNK`, at most `rows`) booked by `plan_glm5_next_chunk`. The one
/// plan of `glm5_run` and of serve's boot (`glm5_engine::serve_plan`), so both book the same memory.
pub fn plan_for_rows(g: &Glm5Geo, context: usize, rows: usize, vram_total: u64, pinned_budget: u64, record_bytes: u64) -> Result<(crate::manager::Glm5States, crate::manager::TierInput, TierPlan), String> {
    let chunk = prompt_chunk_from_env().clamp(1, rows.max(1));
    // #149 path B: the stager's pinned blocks come off the pinned budget (as #186's prefill landing)
    let env = |k: &str| std::env::var(k).ok();
    let stager = stager_on(env(STAGER_ENV).as_deref(), env(glm5_flags::ENV_FLAGS).as_deref(), env(CPU_LANE_ENV).as_deref())?;
    let pinned_budget = if stager { pinned_budget.saturating_sub(stager_pinned_bytes(g.moe_layers(), g.experts, g.topk, record_bytes)) } else { pinned_budget };
    // CROW_GLM_PREFETCH's store and CROW_GLM_LA's host-mapped embedding table, the same way
    let pinned_budget = pinned_budget.saturating_sub(decode_switch_pinned_bytes(g, &Switches::from_env(), record_bytes));
    crate::manager::plan_glm5_next_chunk(g, context, vram_total, pinned_budget, crate::geo::GLM5_NEXT_DENSE_BYTES, record_bytes, crate::gen::pf_tg(), crate::gen::pf_async_on(), chunk)
}

/// the pinned bytes the decode switches `sw` allocate besides the stager: the prefetch store
/// (`glm5_flags::prefetch_pinned_bytes`) and the lookahead's host-mapped embedding table
/// (`glm5_flags::feed_pinned_bytes`)
pub fn decode_switch_pinned_bytes(g: &Glm5Geo, sw: &Switches, record_bytes: u64) -> u64 {
    let pf = if sw.prefetch { glm5_flags::prefetch_pinned_bytes(g.topk, record_bytes) } else { 0 };
    let la = if sw.la { glm5_flags::feed_pinned_bytes(g) } else { 0 };
    pf + la
}

/// #186: the staging slots of a prompt call: the #176 prefill set the plan books
/// (`Stability::stage_slots(..).prefill`: `PF_TG`, doubled with `CROW_PF_ASYNC`)
pub fn prefill_stage_slots(topk: usize) -> usize {
    crate::geo::Stability::of(crate::geo::Family::Glm5Next).stage_slots(topk, crate::gen::pf_tg(), crate::gen::pf_async_on()).prefill
}

// ---------------------------------------------------------------- the global arena (CROW_GLM_ARENA=global)

/// `layer` (default) | `global`
pub const ARENA_ENV: &str = "CROW_GLM_ARENA";

/// How the VRAM and pinned slots of the plan are shared by the MoE layers.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ArenaKind {
    /// `vram` + `pinned` slots per MoE layer under the #175 cache (the path of record)
    #[default]
    Layer,
    /// [`GlobalArena`]: the same slots in one VRAM pool (CLOCK) and one pinned pool (LRU), shared
    /// by every MoE layer, no quota per layer
    Global,
}

/// Parse `CROW_GLM_ARENA`: unset, empty or `layer` = [`ArenaKind::Layer`]; `global` =
/// [`ArenaKind::Global`]; anything else refused by name.
pub fn arena_kind(v: Option<&str>) -> Result<ArenaKind, String> {
    match v.map(str::trim) {
        None | Some("") | Some("layer") => Ok(ArenaKind::Layer),
        Some("global") => Ok(ArenaKind::Global),
        Some(x) => Err(format!("{ARENA_ENV}={x:?}: accepted layer (default), global")),
    }
}

/// whether a call admits its misses into the global arena's VRAM: a call of at most `admit_max`
/// routed picks (rows x top-k, sybil's `GLM53_EC_ADMIT_MAX`, [`ArenaConfig::admit_max`]), never a
/// prompt call (prefill reads its misses where they lie, the decode working set survives)
pub fn arena_admits(picks: usize, prompt: bool, admit_max: usize) -> bool {
    !prompt && picks <= admit_max
}

/// sybil's `GLM53_EC_ADMIT_MAX` (default 64): admission only for calls of at most this many picks
pub const ARENA_ADMIT_MAX_ENV: &str = "CROW_GLM_ARENA_ADMIT_MAX";
/// sybil's `GLM53_NV_NOADMIT`: `1` = NVMe picks are never admitted into VRAM; `0` (default)
pub const ARENA_NOADMIT_ENV: &str = "CROW_GLM_ARENA_NOADMIT";
/// sybil's `GLM53_EC_WARM`: a JSON file of routing scores per MoE layer; the arena starts filled
pub const ARENA_WARM_ENV: &str = "CROW_GLM_ARENA_WARM";
/// sybil's `GLM53_NV_VRING`: VRAM victims written back through a ring of N VRAM slots and a copy
/// stream (default 24; 0 = the D2H on the call's stream)
pub const ARENA_VRING_ENV: &str = "CROW_GLM_ARENA_VRING";
/// sybil's `GLM53_EC_ELASTIC_GB`: an elastic VRAM part of the arena beyond the plan, from free VRAM
pub const ARENA_ELASTIC_ENV: &str = "CROW_GLM_ARENA_ELASTIC_GB";
/// sybil's `GLM53_EC_STAGE_GB`: two staging buffers of this size for large prompt calls
pub const ARENA_STAGE_ENV: &str = "CROW_GLM_ARENA_STAGE_GB";
/// sybil's `GLM53_EC_STAGE_MIN`: a prompt call of at least this many picks runs staged (default 512)
pub const ARENA_STAGE_MIN_ENV: &str = "CROW_GLM_ARENA_STAGE_MIN";
/// the free VRAM the elastic part leaves (sybil's `GLM53_EC_RESERVE_GB` default 2.5)
pub const ARENA_RESERVE_BYTES: u64 = 5 << 29;
const GIB: f64 = (1u64 << 30) as f64;

/// The switches of the global arena (all read with [`arena_config`]; only under `CROW_GLM_ARENA=global`).
#[derive(Clone, Debug, PartialEq)]
pub struct ArenaConfig {
    pub admit_max: usize,
    pub noadmit: bool,
    pub warm: Option<String>,
    pub vring: usize,
    pub elastic_bytes: u64,
    pub stage_bytes: u64,
    pub stage_min: usize,
}

impl Default for ArenaConfig {
    fn default() -> ArenaConfig {
        ArenaConfig { admit_max: expert_cache::ADMIT_MAX, noadmit: false, warm: None, vring: 24, elastic_bytes: 0, stage_bytes: 0, stage_min: 512 }
    }
}

/// Parse the arena switches from `get` (the environment): unset or empty = the default of
/// [`ArenaConfig::default`]; a malformed value is refused by name.
pub fn arena_config(get: &dyn Fn(&str) -> Option<String>) -> Result<ArenaConfig, String> {
    let mut c = ArenaConfig::default();
    let val = |k: &str| get(k).map(|v| v.trim().to_string()).filter(|v| !v.is_empty());
    let int = |k: &str, lo: usize, hi: usize| -> Result<Option<usize>, String> {
        match val(k) {
            None => Ok(None),
            Some(v) => match v.parse::<usize>() {
                Ok(n) if (lo..=hi).contains(&n) => Ok(Some(n)),
                _ => Err(format!("{k}={v:?}: a whole number in {lo}..={hi}")),
            },
        }
    };
    let gb = |k: &str| -> Result<Option<u64>, String> {
        match val(k) {
            None => Ok(None),
            Some(v) => match v.parse::<f64>() {
                Ok(x) if x.is_finite() && (0.0..=1024.0).contains(&x) => Ok(Some((x * GIB) as u64)),
                _ => Err(format!("{k}={v:?}: GiB in 0..=1024")),
            },
        }
    };
    if let Some(n) = int(ARENA_ADMIT_MAX_ENV, 0, 1024)? {
        c.admit_max = n;
    }
    c.noadmit = match val(ARENA_NOADMIT_ENV).as_deref() {
        None | Some("0") => false,
        Some("1") => true,
        Some(v) => return Err(format!("{ARENA_NOADMIT_ENV}={v:?}: accepted 0 (default), 1")),
    };
    c.warm = val(ARENA_WARM_ENV);
    if let Some(n) = int(ARENA_VRING_ENV, 0, 256)? {
        c.vring = n;
    }
    if let Some(b) = gb(ARENA_ELASTIC_ENV)? {
        c.elastic_bytes = b;
    }
    if let Some(b) = gb(ARENA_STAGE_ENV)? {
        c.stage_bytes = b;
    }
    if let Some(n) = int(ARENA_STAGE_MIN_ENV, 1, 1 << 20)? {
        c.stage_min = n;
    }
    Ok(c)
}

/// The routing scores of a warm-start file (sybil's `GLM53_EC_WARM`, `data/stats_own_dec.json`
/// format): a JSON object whose keys end in the decoder layer index (`"3"`, or
/// `"model.language_model.layers.3.mlp"`), each an array of `experts` scores (keys starting with
/// `_` are notes), or a JSON array of one such array per MoE layer. Returns the scores per MoE layer (empty = not warmed).
pub fn parse_warm(text: &str, layers: usize, first_moe: usize, experts: usize) -> Result<Vec<Vec<f64>>, String> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| format!("{ARENA_WARM_ENV}: not JSON: {e}"))?;
    let row = |x: &serde_json::Value, what: &str| -> Result<Vec<f64>, String> {
        let a = x.as_array().ok_or_else(|| format!("{ARENA_WARM_ENV}: {what} is no array"))?;
        if a.len() != experts {
            return Err(format!("{ARENA_WARM_ENV}: {what} holds {} scores, {experts} experts", a.len()));
        }
        a.iter().map(|s| s.as_f64().ok_or_else(|| format!("{ARENA_WARM_ENV}: {what} holds a non-number"))).collect()
    };
    let mut out = vec![Vec::new(); layers];
    match &v {
        serde_json::Value::Array(a) => {
            if a.len() != layers {
                return Err(format!("{ARENA_WARM_ENV}: {} layers, {layers} MoE layers", a.len()));
            }
            for (l, x) in a.iter().enumerate() {
                out[l] = row(x, &format!("MoE layer {l}"))?;
            }
        }
        serde_json::Value::Object(m) => {
            for (k, x) in m.iter().filter(|(k, _)| !k.starts_with('_')) {
                let digits: String = k.split(|c: char| !c.is_ascii_digit()).filter(|d| !d.is_empty()).last().unwrap_or("").to_string();
                let dl: usize = digits.parse().map_err(|_| format!("{ARENA_WARM_ENV}: key {k:?} names no layer"))?;
                let l = dl.checked_sub(first_moe).filter(|&l| l < layers).ok_or_else(|| format!("{ARENA_WARM_ENV}: key {k:?} is no MoE layer"))?;
                out[l] = row(x, k)?;
            }
        }
        _ => return Err(format!("{ARENA_WARM_ENV}: a JSON object or array")),
    }
    Ok(out)
}


/// Where one record of the global arena lies. The slots are global: VRAM slot `v` is slot
/// `v % vram` of the VRAM allocation of MoE layer `v / vram` (`vram` = the plan's slots per
/// layer), pinned slot `q` likewise; the allocations are the per-layer path's, unchanged.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Place {
    Vram(u32),
    Ram(u32),
    Nvme,
}

impl Place {
    fn tier(self) -> Tier {
        match self {
            Place::Vram(_) => Tier::Vram,
            Place::Ram(_) => Tier::Pinned,
            Place::Nvme => Tier::Nvme,
        }
    }
}

/// host counts of the global arena since construction (or [`GlobalArena::reset`])
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ArenaStats {
    /// misses admitted into VRAM
    pub admitted: u64,
    /// VRAM victims written back into the pinned tier
    pub write_backs: u64,
    /// VRAM victims dropped to the NVMe (no pinned slot free of the call's own experts)
    pub wb_dropped: u64,
    /// pinned LRU victims dropped to the NVMe
    pub ram_evictions: u64,
    /// calls whose admissions stopped because every VRAM slot was pinned by the call
    pub no_victim: u64,
}

/// sybil-solutions/glm53-flash-offload 6769b27 (`glm53/expert_cache.py` `ec_step_k`,
/// `kernels/nv2/nv2_host.cpp` RAM LRU), on the host, over every MoE layer of the store:
///
/// - **VRAM**: ONE ring of `layers x vram` slots shared by all MoE layers, CLOCK: a hit sets the
///   slot's reference bit and pins it for the call (epoch); an admitted miss takes the slot the
///   hand stops at (a clear bit, never a slot pinned in this call; at most 2 x slots steps, else
///   the call's remaining misses are not admitted) and enters with its bit set.
/// - **Admission**: only for calls of at most 64 picks ([`arena_admits`]); a prompt call never
///   admits.
/// - **Pinned (RAM)**: ONE LRU of `layers x pinned` slots, exclusive with VRAM: an admission frees
///   the expert's pinned copy, a VRAM victim is written back into it as the most recent; a miss
///   that is not admitted lands there from the NVMe (or is staged when no slot is free of the
///   call's own experts). The LRU victim is the oldest expert not routed in the call.
/// - **NVMe**: everything else.
///
/// [`GlobalArena::step`] decides one call and returns the net change of every expert it moved;
/// [`serve_global`] executes it through the [`Mover`] of the per-layer path.
#[derive(Clone, Debug)]
pub struct GlobalArena {
    pub layers: usize,
    pub experts: usize,
    /// key = MoE layer x experts + expert
    place: Vec<Place>,
    vowner: Vec<u32>,
    refb: Vec<bool>,
    /// the epoch (call) that pinned the slot
    vpin: Vec<u64>,
    hand: usize,
    epoch: u64,
    rowner: Vec<u32>,
    /// the pinned LRU: a doubly linked list over the pinned slots, `oldest` .. `newest`
    prev: Vec<u32>,
    next: Vec<u32>,
    oldest: u32,
    newest: u32,
    rfree: Vec<u32>,
    /// the epoch in which a key was routed (never a pinned LRU victim in that call)
    prot: Vec<u64>,
    /// the epoch in which a key's place before the call was recorded
    seen: Vec<u64>,
    touched: Vec<(u32, Place)>,
    pin_stay: bool,
    /// sybil's `NV_NOADMIT`: an NVMe pick is never admitted into VRAM (it lands in pinned)
    noadmit: bool,
    /// bumped by every call that may move an expert (staged prefetches are valid for one generation)
    gen: u64,
    counters: Vec<[u64; 3]>,
    pub stats: ArenaStats,
}

/// `vpin` of a disabled VRAM slot (the write-back ring, an elastic chunk handed back): never a victim
const DISABLED: u64 = u64::MAX;

impl GlobalArena {
    /// An empty arena: every expert on the NVMe, `vram` VRAM slots and `ram` pinned slots in all.
    pub fn new(layers: usize, experts: usize, vram: usize, ram: usize) -> Result<GlobalArena, String> {
        let keys = layers * experts;
        if keys == 0 || keys >= NONE as usize || vram >= NONE as usize || ram >= NONE as usize {
            return Err(format!("global arena: {layers} layers x {experts} experts, {vram} VRAM / {ram} pinned slots is no arena"));
        }
        Ok(GlobalArena {
            layers,
            experts,
            place: vec![Place::Nvme; keys],
            vowner: vec![NONE; vram],
            refb: vec![false; vram],
            vpin: vec![0; vram],
            hand: 0,
            epoch: 0,
            rowner: vec![NONE; ram],
            prev: vec![NONE; ram],
            next: vec![NONE; ram],
            oldest: NONE,
            newest: NONE,
            // popped from the back: slot 0 first
            rfree: (0..ram as u32).rev().collect(),
            prot: vec![0; keys],
            seen: vec![0; keys],
            touched: Vec::new(),
            pin_stay: false,
            noadmit: false,
            gen: 0,
            counters: vec![[0; 3]; layers],
            stats: ArenaStats::default(),
        })
    }

    /// empty again (as [`GlobalArena::new`]); the switches and the disabled slots are kept
    pub fn reset(&mut self) {
        let (stay, noadmit, gen) = (self.pin_stay, self.noadmit, self.gen);
        let off: Vec<bool> = self.vpin.iter().map(|&p| p == DISABLED).collect();
        *self = GlobalArena::new(self.layers, self.experts, self.vowner.len(), self.rowner.len()).expect("the same shape");
        (self.pin_stay, self.noadmit, self.gen) = (stay, noadmit, gen + 1);
        for (p, off) in self.vpin.iter_mut().zip(off) {
            if off {
                *p = DISABLED;
            }
        }
    }

    /// sybil's `NV_NOADMIT`: NVMe picks are never admitted into VRAM (they land in pinned and
    /// are read zero-copy); pinned hits keep the admission rule
    pub fn set_noadmit(&mut self, on: bool) {
        self.noadmit = on;
    }

    /// the placement generation: changes whenever a call or a slot switch may have moved an expert
    pub fn generation(&self) -> u64 {
        self.gen
    }

    /// VRAM slots a victim may be taken from (not disabled)
    pub fn enabled_vram(&self) -> usize {
        self.vpin.iter().filter(|&&p| p != DISABLED).count()
    }

    /// `n` more VRAM slots (an elastic chunk), empty and enabled, after the existing ones
    pub fn add_slots(&mut self, n: usize) {
        let m = self.vowner.len() + n;
        self.vowner.resize(m, NONE);
        self.refb.resize(m, false);
        self.vpin.resize(m, 0);
        self.gen += 1;
    }

    /// Disable the VRAM slots `r` (never victims from now on): their experts go to pinned as the
    /// most recent (a write-back) or, without a free-able pinned slot, to the NVMe. Returns the net
    /// changes as [`GlobalArena::step`] does.
    pub fn disable(&mut self, r: std::ops::Range<usize>) -> Vec<(u32, Place, Place)> {
        self.epoch += 1;
        self.gen += 1;
        self.touched.clear();
        for s in r {
            self.vpin[s] = DISABLED;
            self.refb[s] = false;
            let v = std::mem::replace(&mut self.vowner[s], NONE);
            if v != NONE {
                let v = v as usize;
                self.record(v);
                self.place[v] = Place::Nvme;
                if self.ram_insert(v).is_some() {
                    self.stats.write_backs += 1;
                } else {
                    self.stats.wb_dropped += 1;
                }
            }
        }
        let mut ch: Vec<(u32, Place, Place)> = self.touched.iter().filter(|&&(k, b)| self.place[k as usize] != b).map(|&(k, b)| (k, b, self.place[k as usize])).collect();
        ch.sort_unstable_by_key(|c| c.0);
        ch
    }

    /// enable the (empty) VRAM slots `r` again
    pub fn enable(&mut self, r: std::ops::Range<usize>) {
        for s in r {
            debug_assert_eq!(self.vowner[s], NONE, "an enabled slot comes back empty");
            self.vpin[s] = 0;
            self.refb[s] = false;
        }
        self.gen += 1;
    }

    /// A staged call (sybil's `_ec_hits`): the routed experts of layer `l` are counted and the VRAM
    /// hits get their reference bit; nothing is admitted, nothing moves. Returns each id's place.
    pub fn mark(&mut self, l: usize, ids: &[u32]) -> Vec<Place> {
        ids.iter()
            .map(|&e| {
                let p = self.place(l, e);
                match p {
                    Place::Vram(s) => {
                        self.refb[s as usize] = true;
                        self.counters[l][0] += 1;
                    }
                    Place::Ram(_) => self.counters[l][1] += 1,
                    Place::Nvme => self.counters[l][2] += 1,
                }
                p
            })
            .collect()
    }

    /// sybil's `warm`: per MoE layer the top `enabled VRAM / layers` experts by score for VRAM,
    /// the next `pinned / layers` for pinned (descending score, ties the lower id; a layer
    /// without scores is not warmed). Returns `(vram ids, pinned ids)` per layer, ascending.
    pub fn warm_plan(&self, scores: &[Vec<f64>]) -> (Vec<Vec<u32>>, Vec<Vec<u32>>) {
        let (k, r) = (self.enabled_vram() / self.layers, self.rowner.len() / self.layers);
        let mut vr = Vec::with_capacity(self.layers);
        let mut pr = Vec::with_capacity(self.layers);
        for l in 0..self.layers {
            let sc = scores.get(l).filter(|s| s.len() == self.experts);
            let Some(sc) = sc else {
                vr.push(Vec::new());
                pr.push(Vec::new());
                continue;
            };
            let mut ord: Vec<u32> = (0..self.experts as u32).collect();
            ord.sort_by(|&a, &b| sc[b as usize].partial_cmp(&sc[a as usize]).unwrap_or(std::cmp::Ordering::Equal).then(a.cmp(&b)));
            let k = k.min(ord.len());
            let r = r.min(ord.len() - k);
            let mut v = ord[..k].to_vec();
            let mut p = ord[k..k + r].to_vec();
            v.sort_unstable();
            p.sort_unstable();
            vr.push(v);
            pr.push(p);
        }
        (vr, pr)
    }

    /// counters and stats to 0 (after a warm start)
    pub fn clear_counters(&mut self) {
        for c in &mut self.counters {
            *c = [0; 3];
        }
        self.stats = ArenaStats::default();
    }

    /// #188 `CROW_GLM_PINNED=zerocopy`: a pinned hit stays in pinned (its LRU place is
    /// refreshed) instead of being admitted into VRAM; NVMe misses keep the admission rule
    pub fn set_pin_stay(&mut self, on: bool) {
        self.pin_stay = on;
    }

    pub fn vram_slots(&self) -> usize {
        self.vowner.len()
    }

    pub fn ram_slots(&self) -> usize {
        self.rowner.len()
    }

    pub fn place(&self, l: usize, e: u32) -> Place {
        assert!(l < self.layers && (e as usize) < self.experts, "global arena: layer {l} expert {e} outside the shape");
        self.place[l * self.experts + e as usize]
    }

    /// `[vram, pinned, nvme]` per MoE layer: the tier each routed access was served from
    pub fn counters(&self) -> &[[u64; 3]] {
        &self.counters
    }

    /// keep the place of `k` before this call (once per call)
    fn record(&mut self, k: usize) {
        if self.seen[k] != self.epoch {
            self.seen[k] = self.epoch;
            self.touched.push((k as u32, self.place[k]));
        }
    }

    fn ram_unlink(&mut self, q: u32) {
        let (p, n) = (self.prev[q as usize], self.next[q as usize]);
        if p != NONE {
            self.next[p as usize] = n;
        } else {
            self.oldest = n;
        }
        if n != NONE {
            self.prev[n as usize] = p;
        } else {
            self.newest = p;
        }
        self.prev[q as usize] = NONE;
        self.next[q as usize] = NONE;
    }

    fn ram_push_newest(&mut self, q: u32) {
        self.prev[q as usize] = self.newest;
        self.next[q as usize] = NONE;
        if self.newest != NONE {
            self.next[self.newest as usize] = q;
        } else {
            self.oldest = q;
        }
        self.newest = q;
    }

    fn ram_touch(&mut self, q: u32) {
        if self.newest != q {
            self.ram_unlink(q);
            self.ram_push_newest(q);
        }
    }

    /// `k` (on the NVMe now) into the pinned tier as the most recent: a free slot, else the
    /// oldest expert not routed in this call drops to the NVMe. `None` = no such slot.
    fn ram_insert(&mut self, k: usize) -> Option<u32> {
        let q = match self.rfree.pop() {
            Some(q) => q,
            None => {
                let mut x = self.oldest;
                while x != NONE && self.prot[self.rowner[x as usize] as usize] == self.epoch {
                    x = self.next[x as usize];
                }
                if x == NONE {
                    return None;
                }
                let o = self.rowner[x as usize] as usize;
                self.record(o);
                self.place[o] = Place::Nvme;
                self.ram_unlink(x);
                self.stats.ram_evictions += 1;
                x
            }
        };
        self.rowner[q as usize] = k as u32;
        self.ram_push_newest(q);
        self.place[k] = Place::Ram(q);
        Some(q)
    }

    /// CLOCK: the VRAM slot the hand stops at (skipping slots pinned in this call, clearing set
    /// reference bits), at most 2 x slots steps
    fn victim(&mut self) -> Option<usize> {
        let n = self.vowner.len();
        for _ in 0..2 * n {
            let s = self.hand;
            self.hand = if s + 1 == n { 0 } else { s + 1 };
            if self.vpin[s] == self.epoch || self.vpin[s] == DISABLED {
                continue;
            }
            if self.refb[s] {
                self.refb[s] = false;
                continue;
            }
            return Some(s);
        }
        None
    }

    /// One call of MoE layer `l` with the distinct routed experts `ids` (ascending); `admit` =
    /// [`arena_admits`]. Returns `(key, before, after)` of every expert whose place changed,
    /// ascending key (key = `l x experts + expert`; a write-back or a pinned eviction may be of
    /// any layer, an admission or an NVMe landing only of `l`).
    pub fn step(&mut self, l: usize, ids: &[u32], admit: bool) -> Vec<(u32, Place, Place)> {
        assert!(l < self.layers, "global arena: layer {l} outside 0..{}", self.layers);
        self.epoch += 1;
        self.gen += 1;
        let ep = self.epoch;
        self.touched.clear();
        let base = l * self.experts;
        let mut misses = Vec::with_capacity(ids.len());
        for &e in ids {
            assert!((e as usize) < self.experts, "global arena: expert {e} outside 0..{}", self.experts);
            let k = base + e as usize;
            self.prot[k] = ep;
            match self.place[k] {
                Place::Vram(s) => {
                    self.refb[s as usize] = true;
                    self.vpin[s as usize] = ep;
                    self.counters[l][0] += 1;
                }
                Place::Ram(_) => {
                    self.counters[l][1] += 1;
                    misses.push(k);
                }
                Place::Nvme => {
                    self.counters[l][2] += 1;
                    misses.push(k);
                }
            }
        }
        if admit && !self.vowner.is_empty() {
            for &k in &misses {
                if (self.pin_stay && matches!(self.place[k], Place::Ram(_))) || (self.noadmit && self.place[k] == Place::Nvme) {
                    continue;
                }
                let Some(s) = self.victim() else {
                    self.stats.no_victim += 1;
                    break;
                };
                let v = self.vowner[s];
                self.record(k);
                // exclusive: the admitted expert's pinned copy goes (its slot is free for the write-back)
                if let Place::Ram(q) = self.place[k] {
                    self.ram_unlink(q);
                    self.rowner[q as usize] = NONE;
                    self.rfree.push(q);
                }
                self.vowner[s] = k as u32;
                self.place[k] = Place::Vram(s as u32);
                self.refb[s] = true;
                self.vpin[s] = ep;
                self.stats.admitted += 1;
                if v != NONE {
                    let v = v as usize;
                    self.record(v);
                    self.place[v] = Place::Nvme;
                    if self.ram_insert(v).is_some() {
                        self.stats.write_backs += 1;
                    } else {
                        self.stats.wb_dropped += 1;
                    }
                }
            }
        }
        for &k in &misses {
            match self.place[k] {
                Place::Vram(_) => {}
                Place::Ram(q) => self.ram_touch(q),
                Place::Nvme => {
                    self.record(k);
                    self.ram_insert(k);
                }
            }
        }
        let mut ch: Vec<(u32, Place, Place)> = self.touched.iter().filter(|&&(k, b)| self.place[k as usize] != b).map(|&(k, b)| (k, b, self.place[k as usize])).collect();
        ch.sort_unstable_by_key(|c| c.0);
        ch
    }

    /// the invariants: every expert in at most one slot, the slot owners and the places agree,
    /// the pinned LRU list holds exactly the pinned experts, the free list the rest
    pub fn check(&self) -> Result<(), String> {
        for (s, &k) in self.vowner.iter().enumerate() {
            if k != NONE && self.place[k as usize] != Place::Vram(s as u32) {
                return Err(format!("VRAM slot {s} owner {k} is at {:?}", self.place[k as usize]));
            }
        }
        for (q, &k) in self.rowner.iter().enumerate() {
            if k != NONE && self.place[k as usize] != Place::Ram(q as u32) {
                return Err(format!("pinned slot {q} owner {k} is at {:?}", self.place[k as usize]));
            }
        }
        for (k, p) in self.place.iter().enumerate() {
            match *p {
                Place::Vram(s) if self.vowner[s as usize] != k as u32 => return Err(format!("key {k} at VRAM {s}, owner {}", self.vowner[s as usize])),
                Place::Ram(q) if self.rowner[q as usize] != k as u32 => return Err(format!("key {k} at pinned {q}, owner {}", self.rowner[q as usize])),
                _ => {}
            }
        }
        let (mut n, mut x, mut last) = (0usize, self.oldest, NONE);
        while x != NONE {
            if self.rowner[x as usize] == NONE || self.prev[x as usize] != last || n > self.rowner.len() {
                return Err(format!("pinned LRU broken at slot {x}"));
            }
            n += 1;
            last = x;
            x = self.next[x as usize];
        }
        let res = self.rowner.iter().filter(|&&k| k != NONE).count();
        if last != self.newest || n != res || res + self.rfree.len() != self.rowner.len() {
            return Err(format!("pinned LRU holds {n}, {res} resident, {} free of {}", self.rfree.len(), self.rowner.len()));
        }
        Ok(())
    }
}

/// One call of MoE layer `l` through the global arena: `ids` the distinct selected experts
/// (ascending), `admit` = [`arena_admits`]. The arena decides ([`GlobalArena::step`]); the moves
/// run in [`serve`]'s phases over the global slots, so no slot is overwritten before it is read:
/// (A) every expert entering VRAM (from pinned or the NVMe) and every selected expert left on the
/// NVMe into a staging slot; barrier; (B) every expert entering pinned (a VRAM victim by D2H, an
/// NVMe miss read straight into its slot); (C) the VRAM entrants from staging into their slots.
/// Returns where each selected record is (global slots).
pub fn serve_global(a: &mut GlobalArena, l: usize, ids: &[u32], admit: bool, stage_cap: usize, m: &mut dyn Mover) -> Result<Served, String> {
    let n = a.experts;
    let before: Vec<Place> = ids.iter().map(|&e| a.place(l, e)).collect();
    let changes = a.step(l, ids, admit);
    let after: Vec<Place> = ids.iter().map(|&e| a.place(l, e)).collect();
    let own = |k: u32| -> Result<u32, String> {
        let (kl, e) = (k as usize / n, k as usize % n);
        if kl != l {
            return Err(format!("global arena: a call of layer {l} reads expert {e} of layer {kl} from the NVMe"));
        }
        Ok(e as u32)
    };
    // the VRAM entrants (all selected, ascending), then the selected experts left on the NVMe
    let mut staged: Vec<(u32, Place)> = changes.iter().filter(|c| matches!(c.2, Place::Vram(_)) && !matches!(c.1, Place::Vram(_))).map(|c| (c.0, c.1)).collect();
    let entrants = staged.len();
    staged.extend(ids.iter().zip(before.iter().zip(&after)).filter(|(_, (_, a))| **a == Place::Nvme).map(|(&e, (&b, _))| ((l * n) as u32 + e, b)));
    if staged.len() > stage_cap {
        return Err(format!("expert tiers: layer {l} stages {} records in one call, {stage_cap} staging slots", staged.len()));
    }
    let mut out = Served::default();
    out.moves.visits = ids.len() as u64;
    for &(_, b, af) in &changes {
        let mv = &mut out.moves;
        match (b.tier(), af.tier()) {
            (Tier::Nvme, Tier::Vram) => mv.n2v += 1,
            (Tier::Pinned, Tier::Vram) => mv.p2v += 1,
            (Tier::Nvme, Tier::Pinned) => mv.n2p += 1,
            (Tier::Vram, Tier::Pinned) => mv.v2p += 1,
            (Tier::Vram, Tier::Nvme) => mv.v2n += 1,
            (Tier::Pinned, Tier::Nvme) => mv.p2n += 1,
            _ => {}
        }
    }
    // phase A
    let mut from_nvme = Vec::new();
    for (s, &(k, b)) in staged.iter().enumerate() {
        match b {
            Place::Vram(v) => {
                m.vram_to_stage(v, s as u32);
                out.moves.vram_to_stage += 1;
            }
            Place::Ram(q) => {
                m.pinned_to_stage(q, s as u32);
                out.moves.pinned_to_stage += 1;
            }
            Place::Nvme => from_nvme.push((own(k)?, Dst::Landing(s as u32))),
        }
    }
    for c in from_nvme.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
    }
    for &(_, d) in &from_nvme {
        if let Dst::Landing(i) = d {
            m.landing_to_stage(i);
            out.moves.landing_to_stage += 1;
        }
    }
    m.barrier();
    // phase B: a pinned slot an entrant takes was left by an expert staged in A or dropped
    let mut to_pinned = Vec::new();
    for &(k, b, af) in &changes {
        let Place::Ram(q) = af else { continue };
        match b {
            Place::Vram(v) => {
                m.vram_to_pinned(v, q);
                out.moves.vram_to_pinned += 1;
            }
            Place::Nvme => to_pinned.push((own(k)?, Dst::Pinned(q))),
            Place::Ram(_) => return Err(format!("global arena: key {k} moved between pinned slots")),
        }
    }
    for c in to_pinned.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
    }
    // phase C: a VRAM slot an entrant takes was left by a victim read in B or dropped
    for (s, &(k, _)) in staged.iter().enumerate().take(entrants) {
        let Place::Vram(v) = a.place[k as usize] else { unreachable!("an entrant is in VRAM") };
        m.stage_to_vram(s as u32, v);
        out.moves.stage_to_vram += 1;
    }
    out.nvme_reads = from_nvme.len() + to_pinned.len();
    out.moves.nvme_to_landing = from_nvme.len() as u64;
    out.moves.nvme_to_pinned = to_pinned.len() as u64;
    out.moves.zero_copy = after.iter().filter(|p| matches!(p, Place::Ram(_))).count() as u64;
    out.locs = ids
        .iter()
        .zip(&after)
        .map(|(&e, &p)| {
            let loc = match p {
                Place::Vram(v) => Loc::Vram(v),
                Place::Ram(q) => Loc::Pinned(q),
                Place::Nvme => Loc::Stage(staged.iter().position(|x| x.0 as usize == l * n + e as usize).expect("a selected NVMe expert is staged") as u32),
            };
            (e, loc)
        })
        .collect();
    Ok(out)
}

/// the records one [`serve_global`] of `ids` in layer `l` would stage; the arena is left as it is
pub fn staged_count_global(a: &GlobalArena, l: usize, ids: &[u32], admit: bool) -> usize {
    let mut c = a.clone();
    let ch = c.step(l, ids, admit);
    ch.iter().filter(|x| matches!(x.2, Place::Vram(_)) && !matches!(x.1, Place::Vram(_))).count() + ids.iter().filter(|&&e| c.place(l, e) == Place::Nvme).count()
}

/// [`fitting_rows`] for the global arena (a prompt call: no admission)
pub fn fitting_rows_global(a: &GlobalArena, l: usize, sel: &[i32], k: usize, cap: usize) -> Result<usize, String> {
    let t = sel.len() / k;
    if t == 0 || sel.len() != t * k {
        return Err(format!("expert tiers: layer {l}: a selection of {} ids is no [rows][{k}]", sel.len()));
    }
    let fits = |r: usize| -> Result<bool, String> { Ok(staged_count_global(a, l, &distinct_ids(&sel[..r * k], a.experts)?, false) <= cap) };
    if fits(t)? {
        return Ok(t);
    }
    let mut r = 1usize << (usize::BITS - 1 - t.leading_zeros());
    if r == t {
        r /= 2;
    }
    while r >= 1 {
        if fits(r)? {
            return Ok(r);
        }
        r /= 2;
    }
    let one = staged_count_global(a, l, &distinct_ids(&sel[..k], a.experts)?, false);
    Err(format!("expert tiers: layer {l}: one row stages {one} records, {cap} staging slots"))
}

/// [`serve_chunk`] for the global arena: a prompt call's `[t][k]` selection in row sub-batches,
/// each one [`serve_global`] without admission
#[allow(clippy::too_many_arguments)]
pub fn serve_chunk_global(
    a: &mut GlobalArena,
    l: usize,
    sel: &[i32],
    k: usize,
    cap: usize,
    m: &mut dyn Mover,
    each: &mut dyn FnMut(usize, usize, &Served) -> Result<(), String>,
) -> Result<usize, String> {
    let t = sel.len() / k;
    let (mut r0, mut batches) = (0, 0);
    while r0 < t {
        if r0 > 0 {
            m.barrier();
        }
        let rows = fitting_rows_global(a, l, &sel[r0 * k..], k, cap)?;
        let ids = distinct_ids(&sel[r0 * k..(r0 + rows) * k], a.experts)?;
        let served = serve_global(a, l, &ids, false, cap, m)?;
        each(r0, rows, &served)?;
        batches += 1;
        r0 += rows;
    }
    Ok(batches)
}

/// a mover whose VRAM and pinned bases can be pointed at another allocation, and the stream its
/// copies run on
trait Rebase<'a> {
    fn set_vram(&mut self, base: Dev);
    fn set_pinned(&mut self, p: Option<&'a Pinned>);
    fn stream(&self) -> sys::CUstream;
}

impl<'a> Rebase<'a> for GpuMover<'a> {
    fn set_vram(&mut self, base: Dev) {
        self.vram = base;
    }
    fn set_pinned(&mut self, p: Option<&'a Pinned>) {
        self.pinned = p;
    }
    fn stream(&self) -> sys::CUstream {
        cuda::cur_stream()
    }
}

impl<'a> Rebase<'a> for StagerMover<'a> {
    fn set_vram(&mut self, base: Dev) {
        self.vram = base;
    }
    fn set_pinned(&mut self, p: Option<&'a Pinned>) {
        self.pinned = p;
    }
    fn stream(&self) -> sys::CUstream {
        self.s
    }
}

/// The host book of the VRAM write-back ring (sybil's `vfree` / `kvr`): ring entry `r` holds the
/// pinned slot its D2H writes (`NONE` = free). `done(r)` says whether entry `r`'s D2H has landed.
#[derive(Clone, Debug)]
struct RingBook {
    q: Vec<u32>,
    next: usize,
}

impl RingBook {
    fn new(n: usize) -> RingBook {
        RingBook { q: vec![NONE; n], next: 0 }
    }

    /// the entry for the next write-back: the first free or landed one from `next`; else the
    /// entry at `next`, whose D2H the caller must wait for first (`true`)
    fn take(&mut self, done: &mut dyn FnMut(usize) -> bool) -> (usize, bool) {
        let n = self.q.len();
        for i in 0..n {
            let r = (self.next + i) % n;
            if self.q[r] == NONE || done(r) {
                self.q[r] = NONE;
                self.next = (r + 1) % n;
                return (r, false);
            }
        }
        let r = self.next;
        self.next = (r + 1) % n;
        (r, true)
    }

    /// the entries whose D2H into pinned slot `q` may still run (landed ones are freed)
    fn pending(&mut self, q: u32, done: &mut dyn FnMut(usize) -> bool) -> Vec<usize> {
        let mut v = Vec::new();
        for r in 0..self.q.len() {
            if self.q[r] == q {
                if done(r) {
                    self.q[r] = NONE;
                } else {
                    v.push(r);
                }
            }
        }
        v
    }
}

/// sybil's `GLM53_NV_VRING` on the device: `slots` VRAM slots of the arena (disabled for the
/// CLOCK) as a ring. A VRAM victim is copied D2D into a ring entry on the call's stream (its slot
/// is free for the entrant right after), then D2H into its pinned slot on the ring's own
/// non-blocking stream (a copy engine), off the call's critical path. Every later reader of that
/// pinned slot waits for the D2H: a queued copy or the kernels through an event on their stream,
/// an NVMe write into the slot on the host.
struct WbRing {
    book: RingBook,
    slots: Vec<u32>,
    d2d: Vec<sys::CUevent>,
    wb: Vec<sys::CUevent>,
    stream: sys::CUstream,
    /// write-backs through the ring, and those that waited for a busy entry
    issued: u64,
    waited: u64,
}

/// an event that never ran or has completed
fn event_done(e: sys::CUevent) -> bool {
    unsafe { sys::cuEventQuery(e) == sys::CUresult::CUDA_SUCCESS }
}

impl WbRing {
    /// # Safety
    /// A CUDA context is current.
    unsafe fn new(slots: Vec<u32>) -> WbRing {
        let n = slots.len();
        WbRing {
            book: RingBook::new(n),
            d2d: (0..n).map(|_| cuda::event_create()).collect(),
            wb: (0..n).map(|_| cuda::event_create()).collect(),
            slots,
            stream: cuda::stream_create_non_blocking(),
            issued: 0,
            waited: 0,
        }
    }

    /// every write-back landed; the book empty
    unsafe fn sync_all(&mut self) {
        cuda::stream_sync(self.stream);
        self.book.q.fill(NONE);
    }

    unsafe fn free(&mut self) {
        self.sync_all();
        for e in self.d2d.drain(..).chain(self.wb.drain(..)) {
            cuda::event_destroy(e);
        }
        cuda::stream_destroy(self.stream);
    }
}

/// The global arena's slots over the VRAM chunks and the per-layer pinned allocations (sybil's
/// `cbase[s / spc] + s % spc`): global VRAM slot `v` -> slot `v % vpl` of `vram[v / vpl]`, pinned
/// slot `q` -> `q % ppl` of `pinned[q / ppl]`; every move is the inner mover's, on its base for
/// that slot, except a VRAM -> pinned write-back with the ring ([`WbRing`]).
struct ChunkMover<'a, 'r, M> {
    inner: M,
    vram: &'a [Dev],
    pinned: &'a [Pinned],
    vpl: usize,
    ppl: usize,
    rb: u64,
    ring: Option<&'r mut WbRing>,
}

impl<'a, M: Mover + Rebase<'a>> ChunkMover<'a, '_, M> {
    fn v(&mut self, v: u32) -> u32 {
        self.inner.set_vram(self.vram[v as usize / self.vpl]);
        (v as usize % self.vpl) as u32
    }
    fn q(&mut self, q: u32) -> u32 {
        self.inner.set_pinned(Some(&self.pinned[q as usize / self.ppl]));
        (q as usize % self.ppl) as u32
    }
    fn vram_addr(&self, v: u32) -> Dev {
        self.vram[v as usize / self.vpl] + (v as usize % self.vpl) as u64 * self.rb
    }
    fn pinned_host(&self, q: u32) -> *mut u8 {
        unsafe { (self.pinned[q as usize / self.ppl].host as *mut u8).add((q as usize % self.ppl) * self.rb as usize) }
    }
    /// the ring entries still writing pinned slot `q`
    fn pending(&mut self, q: u32) -> Vec<usize> {
        match self.ring.as_deref_mut() {
            Some(WbRing { book, wb, .. }) => book.pending(q, &mut |r| event_done(wb[r])),
            None => Vec::new(),
        }
    }
    /// the mover's stream waits for every write-back into `q`
    fn wait_gpu(&mut self, q: u32) {
        let s = self.inner.stream();
        for r in self.pending(q) {
            let e = self.ring.as_ref().expect("pending entries come from the ring").wb[r];
            unsafe { cuda::stream_wait_event(s, e) };
        }
    }
    /// the host reads pinned slot `q` (the CPU lane): every write-back into it landed
    fn wait_host(&mut self, q: u32) {
        for r in self.pending(q) {
            let e = self.ring.as_ref().expect("pending entries come from the ring").wb[r];
            unsafe { cuda::ck(sys::cuEventSynchronize(e)) };
        }
    }

    /// the kernels of this call read the pinned slots of `locs`: their stream waits for the
    /// write-backs into them
    fn settle_locs(&mut self, locs: &[(u32, Loc)]) {
        if self.ring.is_none() {
            return;
        }
        for &(_, loc) in locs {
            if let Loc::Pinned(q) = loc {
                self.wait_gpu(q);
            }
        }
    }
}

impl<'a, M: Mover + Rebase<'a>> Mover for ChunkMover<'a, '_, M> {
    fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
        let ppl = self.ppl;
        let chunk = |d: Dst| match d {
            Dst::Landing(_) => None,
            Dst::Pinned(q) => Some(q as usize / ppl),
        };
        // a host write into a pinned slot waits for a write-back still landing there
        for &(_, d) in jobs {
            if let Dst::Pinned(q) = d {
                for r in self.pending(q) {
                    let e = self.ring.as_ref().expect("pending entries come from the ring").wb[r];
                    unsafe { cuda::ck(sys::cuEventSynchronize(e)) };
                }
            }
        }
        let (mut bytes, mut i) = (0, 0);
        while i < jobs.len() {
            let c = chunk(jobs[i].1);
            let mut local = Vec::new();
            while i < jobs.len() && chunk(jobs[i].1) == c {
                let (e, d) = jobs[i];
                local.push((e, if let Dst::Pinned(q) = d { Dst::Pinned((q as usize % ppl) as u32) } else { d }));
                i += 1;
            }
            if let Some(c) = c {
                self.inner.set_pinned(Some(&self.pinned[c]));
            }
            bytes += self.inner.nvme(&local)?;
        }
        Ok(bytes)
    }
    fn landing_to_stage(&mut self, i: u32) {
        self.inner.landing_to_stage(i);
    }
    fn pinned_to_stage(&mut self, q: u32, s: u32) {
        self.wait_gpu(q);
        let q = self.q(q);
        self.inner.pinned_to_stage(q, s);
    }
    fn vram_to_stage(&mut self, v: u32, s: u32) {
        let v = self.v(v);
        self.inner.vram_to_stage(v, s);
    }
    fn barrier(&mut self) {
        self.inner.barrier();
    }
    fn vram_to_pinned(&mut self, v: u32, q: u32) {
        let (src, dst, rb, s) = (self.vram_addr(v), self.pinned_host(q), self.rb as usize, self.inner.stream());
        let Some(ring) = self.ring.as_deref_mut() else {
            let (v, q) = (self.v(v), self.q(q));
            self.inner.vram_to_pinned(v, q);
            return;
        };
        let WbRing { book, wb, .. } = ring;
        let (r, wait) = book.take(&mut |r| event_done(wb[r]));
        let ra = self.vram[ring.slots[r] as usize / self.vpl] + (ring.slots[r] as usize % self.vpl) as u64 * self.rb;
        unsafe {
            if wait {
                // the entry's previous D2H must have read it before the D2D overwrites it
                cuda::stream_wait_event(s, ring.wb[r]);
                ring.waited += 1;
            }
            cuda::memcpy_async_on(ra, src, rb, s);
            cuda::event_record(ring.d2d[r], s);
            cuda::stream_wait_event(ring.stream, ring.d2d[r]);
            cuda::ck(sys::cuMemcpyDtoHAsync_v2(dst as *mut _, ra, rb, ring.stream));
            cuda::event_record(ring.wb[r], ring.stream);
            cuda::stream_query(ring.stream);
        }
        ring.book.q[r] = q;
        ring.issued += 1;
    }
    fn stage_to_vram(&mut self, s: u32, v: u32) {
        let v = self.v(v);
        self.inner.stage_to_vram(s, v);
    }
}

/// the device address of a [`Loc`] of the global arena
fn arena_addr(vram: &[Dev], pinned: &[Pinned], vpl: usize, ppl: usize, stage: Dev, rb: u64, loc: Loc) -> Dev {
    match loc {
        Loc::Vram(v) => vram[v as usize / vpl] + (v as usize % vpl) as u64 * rb,
        Loc::Pinned(q) => pinned[q as usize / ppl].dev + (q as usize % ppl) as u64 * rb,
        Loc::Stage(s) => stage + s as u64 * rb,
    }
}

/// the moves of [`GlobalArena::disable`]'s changes: every VRAM expert that went to pinned is
/// written back (D2H); one that went to the NVMe is dropped
fn write_back(changes: &[(u32, Place, Place)], m: &mut dyn Mover) -> Result<u64, String> {
    let mut n = 0;
    for &(k, b, a) in changes {
        match (b, a) {
            (Place::Vram(v), Place::Ram(q)) => {
                m.vram_to_pinned(v, q);
                n += 1;
            }
            (Place::Vram(_), Place::Nvme) | (Place::Ram(_), Place::Nvme) => {}
            _ => return Err(format!("global arena: disabling slots moved key {k} from {b:?} to {a:?}")),
        }
    }
    Ok(n)
}

/// the elastic part of the global arena (sybil's `GLM53_EC_ELASTIC_GB`), since construction
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ElasticStats {
    /// elastic chunks allocated at construction (`vram` slots each)
    pub chunks: usize,
    /// hand-backs (staged prompt calls, or free VRAM below the reserve) and regrowths
    pub enter: u64,
    pub exit: u64,
    /// chunks that could not be allocated again (their slots stay disabled until the next try)
    pub realloc_fail: u64,
    /// experts written back to pinned at a hand-back
    pub write_backs: u64,
}

/// the staging buffers of large prompt calls (sybil's `GLM53_EC_STAGE_GB` / `GLM53_EC_STAGE_MIN`)
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StageStats {
    /// staged calls (prompt calls of at least `stage_min` picks)
    pub calls: u64,
    /// layers staged one layer ahead on the copy stream, and those whose prefetch was used
    pub prefetched: u64,
    pub prefetch_used: u64,
    /// calls staged synchronously (no valid prefetch) and calls that fell back to sub-batches
    pub restaged: u64,
    pub fallbacks: u64,
    /// records copied into a buffer from their pinned slot, and read from the NVMe
    pub from_pinned: u64,
    pub from_nvme: u64,
}

/// two staging buffers of `nst` records and the copy stream that fills them one layer ahead
struct StageSet {
    bufs: [Dev; 2],
    nst: usize,
    /// allocated at construction (no elastic part), else for one staged forward
    permanent: bool,
    stream: sys::CUstream,
    ready: [sys::CUevent; 2],
    used: [sys::CUevent; 2],
    /// the experts in each buffer, its MoE layer (`usize::MAX` none) and the placement generation
    content: [Vec<u32>; 2],
    layer: [usize; 2],
    gen: [u64; 2],
    in_forward: bool,
    stats: StageStats,
}

/// The global arena on the device: the host policy, its switches, the VRAM chunks its slots live
/// in, the write-back ring, the elastic part and the staging buffers.
struct ArenaDev {
    a: GlobalArena,
    cfg: ArenaConfig,
    /// VRAM chunks of `vram` (per layer) slots: the per-layer allocations, then the elastic ones
    /// (0 = handed back)
    chunks: Vec<Dev>,
    base_chunks: usize,
    ring: Option<WbRing>,
    stage: Option<StageSet>,
    elastic: ElasticStats,
}

impl ArenaDev {
    fn new(a: GlobalArena, cfg: ArenaConfig) -> ArenaDev {
        ArenaDev { a, cfg, chunks: Vec::new(), base_chunks: 0, ring: None, stage: None, elastic: ElasticStats::default() }
    }

    fn reset(&mut self) {
        // pinned slots still landing a write-back are free in the new arena: let them land first
        if let Some(r) = self.ring.as_mut() {
            unsafe { r.sync_all() };
        }
        self.a.reset();
    }

    /// the elastic chunks (index into `chunks`)
    fn flex(&self) -> std::ops::Range<usize> {
        self.base_chunks..self.chunks.len()
    }
}

impl ExpertTiers {
    /// how the slots are shared (`CROW_GLM_ARENA` as read at construction)
    pub fn arena_kind(&self) -> ArenaKind {
        if self.arena.is_some() {
            ArenaKind::Global
        } else {
            ArenaKind::Layer
        }
    }

    /// the global arena's policy (`None` on the per-layer path)
    pub fn arena(&self) -> Option<&GlobalArena> {
        self.arena.as_ref().map(|d| &d.a)
    }

    /// `CROW_GLM_ARENA` elastic: (elastic chunks allocated now, chunks of the elastic part, VRAM
    /// slots per chunk); `None` without the global arena
    pub fn elastic_live(&self) -> Option<(usize, usize, usize)> {
        let d = self.arena.as_ref()?;
        Some((d.flex().filter(|&c| d.chunks[c] != 0).count(), d.flex().len(), self.sizes.vram))
    }

    /// `CROW_GLM_ARENA=global`: the decode phase begins on the host thread: a staged forward still
    /// open ends (its copy stream drained, per-forward buffers freed) and the elastic part grows
    /// back. `table_global` does the same at a decode call, but under `CROW_GLM_CONTROLLER` that
    /// call runs on the controller thread while the device waits for its reply, where a stream
    /// sync would wait for the device's own wait; so the host calls this before it enqueues a
    /// controlled row, and after a prompt phase. A no-op without the global arena.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading a staging buffer is pending.
    pub unsafe fn decode_ready(&mut self) {
        if self.arena.is_none() {
            return;
        }
        self.stage_end();
        if self.arena.as_ref().is_some_and(|d| d.flex().any(|c| d.chunks[c] == 0)) {
            self.elastic_exit();
        }
    }

    /// `CROW_GLM_ARENA` elastic: hand the elastic chunks back before the prompt phase borrows
    /// their memory (`Glm5Run::prefill_with`); a staged forward still open ends first. The next
    /// decode call grows them back (`table_global`). A no-op without the global arena.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading the arena is pending.
    pub unsafe fn elastic_hand_back(&mut self) -> Result<(), String> {
        if self.arena.is_none() {
            return Ok(());
        }
        self.stage_end();
        self.elastic_enter()
    }

    /// the global arena's switches (`None` on the per-layer path)
    pub fn arena_config(&self) -> Option<&ArenaConfig> {
        self.arena.as_ref().map(|d| &d.cfg)
    }

    /// the write-back ring: entries, write-backs through it, write-backs that waited for an entry
    pub fn arena_ring_stats(&self) -> Option<(usize, u64, u64)> {
        self.arena.as_ref().and_then(|d| d.ring.as_ref()).map(|r| (r.slots.len(), r.issued, r.waited))
    }

    pub fn arena_elastic_stats(&self) -> Option<ElasticStats> {
        self.arena.as_ref().map(|d| d.elastic)
    }

    /// the staging buffers' counters and their size in records (`None` = no staging)
    pub fn arena_stage_stats(&self) -> Option<(StageStats, usize)> {
        self.arena.as_ref().and_then(|d| d.stage.as_ref()).map(|s| (s.stats, s.nst))
    }

    /// `[vram, pinned, nvme]` per MoE layer: the tier each access was served from (the #175
    /// cache's counters, or the global arena's)
    pub fn tier_counters(&self) -> &[[u64; 3]] {
        match &self.arena {
            Some(d) => d.a.counters(),
            None => self.cache.counters(),
        }
    }

    /// VRAM the global arena holds beyond the per-layer allocations (elastic chunks, staging)
    fn arena_extra_vram_bytes(&self) -> u64 {
        let Some(d) = self.arena.as_ref() else { return 0 };
        let flex = d.chunks[d.base_chunks..].iter().filter(|&&c| c != 0).count() as u64 * self.sizes.vram as u64 * self.rb;
        let stage = d.stage.as_ref().map_or(0, |s| if s.bufs[0] != 0 { 2 * s.nst as u64 * self.rb } else { 0 });
        flex + stage
    }

    /// The device side of `CROW_GLM_ARENA=global`, after the per-layer allocations: the ring's
    /// slots carved from the end of the arena (off when it holds fewer than 2 x N slots or there
    /// is no pinned tier to write back into), the
    /// elastic chunks from free VRAM above [`ARENA_RESERVE_BYTES`], the staging buffers (at once
    /// without an elastic part, else per staged forward), the warm start.
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn arena_boot(&mut self) -> Result<(), String> {
        let (rb, vpl, nl, topk) = (self.rb, self.sizes.vram, self.slots.len(), self.topk);
        let d = self.arena.as_mut().expect("arena_boot without the global arena");
        d.chunks = if vpl > 0 { self.vram.clone() } else { Vec::new() };
        d.base_chunks = d.chunks.len();
        d.a.set_noadmit(d.cfg.noadmit);
        let base = nl * vpl;
        if d.cfg.vring > 0 && base >= 2 * d.cfg.vring && d.a.ram_slots() > 0 {
            let r = base - d.cfg.vring..base;
            let ch = d.a.disable(r.clone());
            debug_assert!(ch.is_empty(), "an empty arena moves nothing");
            d.ring = Some(WbRing::new(r.map(|s| s as u32).collect()));
        }
        if d.cfg.elastic_bytes > 0 && vpl > 0 {
            let cb = vpl as u64 * rb;
            let want = d.cfg.elastic_bytes.div_ceil(cb);
            for _ in 0..want {
                if cuda::free_vram_bytes() < cb + ARENA_RESERVE_BYTES {
                    break;
                }
                match cuda::try_alloc_zeroed("glm5 elastic VRAM expert slots", cb as usize) {
                    Ok(c) => {
                        d.chunks.push(c);
                        d.a.add_slots(vpl);
                        d.elastic.chunks += 1;
                    }
                    Err(_) => break,
                }
            }
        }
        if d.cfg.stage_bytes > 0 {
            let nst = (d.cfg.stage_bytes / rb) as usize;
            if nst < topk {
                return Err(format!("{ARENA_STAGE_ENV}: {:.2} GiB holds {nst} records, less than one row's top-{topk}", d.cfg.stage_bytes as f64 / GIB));
            }
            let permanent = d.elastic.chunks == 0;
            let mut bufs = [0; 2];
            if permanent {
                for b in &mut bufs {
                    *b = cuda::try_alloc_zeroed("glm5 arena staging buffer", nst * rb as usize).map_err(|e| format!("{ARENA_STAGE_ENV}: the staging buffers do not fit: {e:?}"))?;
                }
            }
            d.stage = Some(StageSet {
                bufs,
                nst,
                permanent,
                stream: cuda::stream_create_non_blocking(),
                ready: [cuda::event_create(), cuda::event_create()],
                used: [cuda::event_create(), cuda::event_create()],
                content: [Vec::new(), Vec::new()],
                layer: [usize::MAX; 2],
                gen: [0; 2],
                in_forward: false,
                stats: StageStats::default(),
            });
        }
        if let Some(path) = d.cfg.warm.clone() {
            let text = std::fs::read_to_string(&path).map_err(|e| format!("{ARENA_WARM_ENV}={path}: {e}"))?;
            let scores = parse_warm(&text, nl, self.first_moe, self.cache.experts)?;
            self.arena_warm(&scores)?;
        }
        Ok(())
    }

    /// sybil's `warm`: fill the arena from per-layer scores ([`GlobalArena::warm_plan`]) through
    /// the admission path (VRAM, in groups of the staging slots) and NVMe landings (pinned); the
    /// counters of the arena and of the store restart at 0 afterwards.
    ///
    /// # Safety
    /// A CUDA context is current; nothing reads the store.
    unsafe fn arena_warm(&mut self, scores: &[Vec<f64>]) -> Result<(), String> {
        let (rb, vpl, ppl, stage) = (self.rb, self.sizes.vram, self.sizes.pinned, self.stage);
        let d = self.arena.as_mut().expect("arena_warm without the global arena");
        let (vr, pr) = d.a.warm_plan(scores);
        for (l, (v, p)) in vr.iter().zip(&pr).enumerate() {
            let inner = GpuMover { vram: 0, pinned: None, stage, landing: self.landing.p, rb, src: &self.src, recs: &self.records[l] };
            let mut m = ChunkMover { inner, vram: &d.chunks, pinned: &self.pinned, vpl, ppl, rb, ring: None };
            for g in v.chunks(self.stage_cap.max(1)) {
                serve_global(&mut d.a, l, g, true, self.stage_cap, &mut m)?;
                m.barrier();
            }
            for g in p.chunks(MAX_IN_FLIGHT) {
                serve_global(&mut d.a, l, g, false, self.stage_cap, &mut m)?;
            }
        }
        cuda::sync();
        d.a.check()?;
        d.a.clear_counters();
        self.nvme_reads = 0;
        self.nvme_bytes = 0;
        self.moves.iter_mut().for_each(|m| *m = Moves::default());
        Ok(())
    }

    /// Hand the elastic chunks back: their experts written back to pinned (or dropped), their
    /// slots disabled, their memory freed.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading the arena is pending.
    unsafe fn elastic_enter(&mut self) -> Result<(), String> {
        let (rb, vpl, ppl, stage) = (self.rb, self.sizes.vram, self.sizes.pinned, self.stage);
        let d = self.arena.as_mut().expect("elastic_enter without the global arena");
        let live: Vec<usize> = d.flex().filter(|&c| d.chunks[c] != 0).collect();
        if live.is_empty() {
            return Ok(());
        }
        cuda::sync();
        if let Some(r) = d.ring.as_mut() {
            r.sync_all();
        }
        let mut ch = Vec::new();
        for &c in &live {
            ch.extend(d.a.disable(c * vpl..(c + 1) * vpl));
        }
        let inner = GpuMover { vram: 0, pinned: None, stage, landing: self.landing.p, rb, src: &self.src, recs: &self.records[0] };
        let mut m = ChunkMover { inner, vram: &d.chunks, pinned: &self.pinned, vpl, ppl, rb, ring: None };
        d.elastic.write_backs += write_back(&ch, &mut m)?;
        cuda::sync();
        for c in live {
            cuda::free_dev(&mut d.chunks[c]);
            d.chunks[c] = 0;
        }
        d.elastic.enter += 1;
        Ok(())
    }

    /// Grow the elastic part back: every handed-back chunk that fits above the reserve is
    /// allocated again and its (empty) slots enabled.
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn elastic_exit(&mut self) {
        let (rb, vpl) = (self.rb, self.sizes.vram);
        let d = self.arena.as_mut().expect("elastic_exit without the global arena");
        let down: Vec<usize> = d.flex().filter(|&c| d.chunks[c] == 0).collect();
        if down.is_empty() {
            return;
        }
        let cb = vpl as u64 * rb;
        for c in down {
            let ok = cuda::free_vram_bytes() >= cb + ARENA_RESERVE_BYTES;
            match ok.then(|| cuda::try_alloc_zeroed("glm5 elastic VRAM expert slots", cb as usize)) {
                Some(Ok(p)) => {
                    d.chunks[c] = p;
                    d.a.enable(c * vpl..(c + 1) * vpl);
                }
                _ => d.elastic.realloc_fail += 1,
            }
        }
        d.elastic.exit += 1;
    }

    /// end a staged forward: the copy stream drained, per-forward buffers freed, the elastic part
    /// grown back
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading a staging buffer is pending.
    unsafe fn stage_end(&mut self) {
        let Some(d) = self.arena.as_mut() else { return };
        let Some(st) = d.stage.as_mut() else { return };
        if !st.in_forward {
            return;
        }
        cuda::stream_sync(st.stream);
        st.layer = [usize::MAX; 2];
        st.in_forward = false;
        if !st.permanent {
            cuda::sync();
            for b in &mut st.bufs {
                cuda::free_dev(b);
                *b = 0;
            }
        }
        self.elastic_exit();
    }

    /// [`ExpertTiers::table_for`] through the global arena (decode-sized calls admit), with the
    /// synchronous mover or the stager
    ///
    /// # Safety
    /// As [`ExpertTiers::table_for`].
    unsafe fn table_global(&mut self, l: usize, sel: &[i32], ids: &[u32], reply: Option<(Dev, u64)>) -> Result<(Dev, Served), String> {
        // a decode call ends a staged forward; the elastic part grows back when it can
        self.stage_end();
        if l == 0 && self.arena.as_ref().is_some_and(|d| d.flex().any(|c| d.chunks[c] == 0)) {
            self.elastic_exit();
        }
        let (rb, experts, vpl, ppl, stage) = (self.rb, self.cache.experts, self.sizes.vram, self.sizes.pinned, self.stage);
        let d = self.arena.as_mut().expect("table_global without the global arena");
        let admit = arena_admits(sel.len(), false, d.cfg.admit_max);
        d.a.set_pin_stay(self.pinned_use.stay);
        let served = match self.stager.as_mut() {
            None => {
                let inner = GpuMover { vram: 0, pinned: None, stage, landing: self.landing.p, rb, src: &self.src, recs: &self.records[l] };
                let mut m = ChunkMover { inner, vram: &d.chunks, pinned: &self.pinned, vpl, ppl, rb, ring: d.ring.as_mut() };
                let served = match self.prefetch.as_mut() {
                    Some(pf) => serve_global(&mut d.a, l, ids, admit, self.stage_cap, &mut glm5_flags::PrefetchMover::new(&mut m, pf, &self.src, l, stage, None))?,
                    None => serve_global(&mut d.a, l, ids, admit, self.stage_cap, &mut m)?,
                };
                m.settle_locs(&served.locs);
                // the CPU lane reads pinned slots on the host: every write-back into them landed
                if self.pinned_use.cpu_lane {
                    for &(_, loc) in &served.locs {
                        if let Loc::Pinned(q) = loc {
                            m.wait_host(q);
                        }
                    }
                }
                let mut table = vec![0u64; experts];
                for &(e, loc) in &served.locs {
                    table[e as usize] = arena_addr(&d.chunks, &self.pinned, vpl, ppl, stage, rb, loc);
                }
                cuda::to_u64_into(self.tables[l], &table);
                (served, None)
            }
            Some(st) => {
                st.settle(&self.src)?;
                st.seq += 1;
                st.stats.calls += 1;
                let row = l * experts;
                let inner = StagerMover {
                    s: st.stream,
                    vram: 0,
                    pinned: None,
                    stage,
                    landing: st.landing.host as *mut u8,
                    rb,
                    src: &self.src,
                    recs: &self.records[l],
                    landed_host: (st.landed.host as *mut u64).add(row),
                    landed_dev: st.landed.dev + (row * 8) as u64,
                    seq: st.seq,
                    read_pinned: Vec::new(),
                    pending: &mut st.pending,
                    stats: &mut st.stats,
                };
                let mut m = ChunkMover { inner, vram: &d.chunks, pinned: &self.pinned, vpl, ppl, rb, ring: d.ring.as_mut() };
                let served = match self.prefetch.as_mut() {
                    Some(pf) => serve_global(&mut d.a, l, ids, admit, self.stage_cap, &mut glm5_flags::PrefetchMover::new(&mut m, pf, &self.src, l, stage, Some(st.stream)))?,
                    None => serve_global(&mut d.a, l, ids, admit, self.stage_cap, &mut m)?,
                };
                m.settle_locs(&served.locs);
                if self.pinned_use.cpu_lane {
                    for &(_, loc) in &served.locs {
                        if let Loc::Pinned(q) = loc {
                            m.wait_host(q);
                        }
                    }
                }
                // as `table_staged`: the table into this layer's pinned row, up on the stager
                let trow = (st.tables.host as *mut u64).add(row);
                let host = std::slice::from_raw_parts_mut(trow, experts);
                host.fill(0);
                for &(e, loc) in &served.locs {
                    host[e as usize] = arena_addr(&d.chunks, &self.pinned, vpl, ppl, stage, rb, loc);
                }
                (served, Some((trow, st.stream, st.event)))
            }
        };
        let (mut served, staged) = served;
        // the CPU lane: the host addresses of the arena's pinned slots
        let lane = {
            let pinned = &self.pinned;
            let tv: Vec<u64> = match staged {
                Some((trow, _, _)) => std::slice::from_raw_parts(trow, experts).to_vec(),
                None => {
                    let d = self.arena.as_ref().expect("table_global without the global arena");
                    let mut t = vec![0u64; experts];
                    for &(e, loc) in &served.locs {
                        t[e as usize] = arena_addr(&d.chunks, pinned, vpl, ppl, stage, rb, loc);
                    }
                    t
                }
            };
            self.lane_plan(l, sel, &mut served, &|e| tv[e as usize], &|q| (pinned[q as usize / ppl].host as *const u8).add((q as usize % ppl) * rb as usize))
        };
        match staged {
            Some((trow, stream, event)) => self.finish_staged(l, lane, served.locs.len(), trow, stream, event, reply)?,
            None => {
                let post = lane.map(|(combos, _)| crate::glm5_moe::lane::Call { table: self.tables[l], combos, clock: self.lane_clock.clone(), ready: None });
                crate::glm5_moe::lane::post(post);
            }
        }
        self.count_heat(l, sel);
        if let Some(pf) = self.prefetch.as_mut() {
            let a = &self.arena.as_ref().expect("table_global without the global arena").a;
            glm5_flags::prefetch_hinted(pf, &self.src, &self.records, experts, &|l, e| a.place(l, e) == Place::Nvme, self.first_moe, self.first_moe + l)?;
        }
        self.nvme_reads += served.nvme_reads as u64;
        self.nvme_bytes += served.nvme_bytes;
        self.moves[l].add(&served.moves);
        self.routing_syncs += 1;
        self.sub_batches += 1;
        Ok((self.tables[l], served))
    }

    /// [`ExpertTiers::tables_for_chunk`] through the global arena (a prompt call never admits):
    /// staged ([`ExpertTiers::stage_call`]) at `stage_min` picks or more with staging on, else in
    /// row sub-batches through the prefill staging set as on the per-layer path
    ///
    /// # Safety
    /// As [`ExpertTiers::tables_for_chunk`].
    unsafe fn tables_for_chunk_global(&mut self, l: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>) -> Result<(), String> {
        if self.pf_cap == 0 {
            self.alloc_prefill_stage(prefill_stage_slots(self.topk))?;
        }
        if self.arena_landing.p.is_null() {
            self.arena_landing = Landing::new(self.pf_cap * self.rb as usize);
        }
        self.settle()?;
        crate::glm5_moe::lane::post(None);
        let d = self.arena.as_ref().expect("tables_for_chunk_global without the global arena");
        if d.stage.is_some() && sel.len() >= d.cfg.stage_min {
            if self.stage_call(l, sel, run)? {
                return Ok(());
            }
        } else {
            self.stage_end();
            // shrink with free VRAM: once per prompt call (its first MoE layer)
            if l == 0 && self.arena.as_ref().is_some_and(|d| d.flex().any(|c| d.chunks[c] != 0)) && cuda::free_vram_bytes() < ARENA_RESERVE_BYTES {
                self.elastic_enter()?;
            }
        }
        let (rb, k, experts, vpl, ppl) = (self.rb, self.topk, self.cache.experts, self.sizes.vram, self.sizes.pinned);
        let (stage, table_dev) = (self.pf_stage, self.tables[l]);
        let d = self.arena.as_mut().expect("tables_for_chunk_global without the global arena");
        d.a.set_pin_stay(self.pinned_use.stay);
        // prompt calls admit nothing, so write no ring entry: let the pending ones land first
        if let Some(r) = d.ring.as_mut() {
            r.sync_all();
        }
        let (vram, pinned) = (&d.chunks, &self.pinned);
        let inner = GpuMover { vram: 0, pinned: None, stage, landing: self.arena_landing.p, rb, src: &self.src, recs: &self.records[l] };
        let mut m = ChunkMover { inner, vram, pinned, vpl, ppl, rb, ring: None };
        let (mut reads, mut bytes, mut moves) = (0u64, 0u64, Moves::default());
        let mut each = |r0: usize, rows: usize, served: &Served| -> Result<(), String> {
            let mut table = vec![0u64; experts];
            for &(e, loc) in &served.locs {
                table[e as usize] = arena_addr(vram, pinned, vpl, ppl, stage, rb, loc);
            }
            cuda::to_u64_into(table_dev, &table);
            reads += served.nvme_reads as u64;
            bytes += served.nvme_bytes;
            moves.add(&served.moves);
            run(r0, rows, table_dev)
        };
        let r = serve_chunk_global(&mut d.a, l, sel, k, self.pf_cap, &mut m, &mut each);
        if r.is_ok() {
            self.count_heat(l, sel);
        }
        self.nvme_reads += reads;
        self.nvme_bytes += bytes;
        self.moves[l].add(&moves);
        self.routing_syncs += 1;
        self.sub_batches += *r.as_ref().unwrap_or(&0) as u64;
        r.map(|_| ())
    }

    /// Copy the records `ids` of MoE layer `l` into staging buffer `b` on the copy stream (behind
    /// the buffer's last reader): from their pinned slot, or read from the NVMe through the
    /// prefill landing. Counted as layer `l`'s moves. Every id is in pinned or on the NVMe.
    ///
    /// # Safety
    /// A CUDA context is current; the buffers are allocated; no write-back is pending.
    unsafe fn stage_issue(&mut self, l: usize, b: usize, ids: Vec<u32>) -> Result<(), String> {
        let (rb, ppl) = (self.rb, self.sizes.pinned);
        let d = self.arena.as_mut().expect("stage_issue without the global arena");
        let st = d.stage.as_mut().expect("stage_issue without staging");
        cuda::stream_wait_event(st.stream, st.used[b]);
        let buf = st.bufs[b];
        let mut nv = Vec::new();
        let mut mv = Moves::default();
        for (i, &e) in ids.iter().enumerate() {
            match d.a.place(l, e) {
                Place::Ram(q) => {
                    let host = (self.pinned[q as usize / ppl].host as *const u8).add((q as usize % ppl) * rb as usize);
                    cuda::ck(sys::cuMemcpyHtoDAsync_v2(buf + i as u64 * rb, host as *const _, rb as usize, st.stream));
                    mv.pinned_to_stage += 1;
                    st.stats.from_pinned += 1;
                }
                Place::Nvme => nv.push((e, i)),
                Place::Vram(_) => return Err(format!("global arena: staging layer {l} expert {e}, which is in VRAM")),
            }
        }
        let cap = self.pf_cap.clamp(1, MAX_IN_FLIGHT);
        for g in nv.chunks(cap) {
            let jobs: Vec<(ExpertRecord, RecordDst)> =
                g.iter().enumerate().map(|(j, &(e, _))| (self.records[l][e as usize], RecordDst { gu: self.arena_landing.p.add(j * rb as usize), dn: std::ptr::null_mut() })).collect();
            let t = self.src.fetch(&jobs)?;
            self.nvme_bytes += self.src.wait(t)?.bytes;
            for (j, &(_, i)) in g.iter().enumerate() {
                // pageable source: the call returns once the bytes are taken, so the landing is free again
                cuda::ck(sys::cuMemcpyHtoDAsync_v2(buf + i as u64 * rb, self.arena_landing.p.add(j * rb as usize) as *const _, rb as usize, st.stream));
            }
            mv.nvme_to_landing += g.len() as u64;
            mv.landing_to_stage += g.len() as u64;
            st.stats.from_nvme += g.len() as u64;
        }
        self.nvme_reads += nv.len() as u64;
        cuda::event_record(st.ready[b], st.stream);
        cuda::stream_query(st.stream);
        st.content[b] = ids;
        st.layer[b] = l;
        st.gen[b] = d.a.generation();
        self.moves[l].add(&mv);
        Ok(())
    }

    /// sybil's staged prefill: one prompt call of MoE layer `l` with at least `stage_min` picks.
    /// The VRAM hits are read in place (reference bit only, nothing admitted, nothing moves);
    /// every other selected record is read from staging buffer `l % 2`, filled one layer ahead
    /// (every non-VRAM expert of the layer, up to the buffer) or now when that prefetch is stale
    /// or misses a selected id. One `run` for all rows; then the next MoE layer is prefetched into
    /// the other buffer while this one computes. `false` = more selected records than a buffer
    /// holds: the caller serves the call in sub-batches.
    ///
    /// # Safety
    /// As [`ExpertTiers::tables_for_chunk`].
    unsafe fn stage_call(&mut self, l: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>) -> Result<bool, String> {
        let (experts, rb, vpl, ppl, nl) = (self.cache.experts, self.rb, self.sizes.vram, self.sizes.pinned, self.slots.len());
        let ids = distinct_ids(sel, experts)?;
        let nst = self.arena.as_ref().and_then(|d| d.stage.as_ref()).expect("stage_call without staging").nst;
        // a staged forward begins: per-forward buffers come from the elastic part's memory (its
        // hand-back moves VRAM experts, so the staged set is taken after it)
        let begin = !self.arena.as_ref().and_then(|d| d.stage.as_ref()).expect("staging").in_forward;
        if begin {
            if let Some(r) = self.arena.as_mut().and_then(|d| d.ring.as_mut()) {
                r.sync_all();
            }
            let permanent = self.arena.as_ref().and_then(|d| d.stage.as_ref()).expect("staging").permanent;
            if !permanent {
                self.elastic_enter()?;
                let st = self.arena.as_mut().and_then(|d| d.stage.as_mut()).expect("staging");
                for i in 0..2 {
                    st.bufs[i] = cuda::try_alloc_zeroed("glm5 arena staging buffer", nst * rb as usize).map_err(|e| format!("{ARENA_STAGE_ENV}: a staging buffer does not fit after the elastic hand-back: {e:?}"))?;
                }
            }
            self.arena.as_mut().and_then(|d| d.stage.as_mut()).expect("staging").in_forward = true;
        }
        let need: Vec<u32> = {
            let d = self.arena.as_ref().expect("stage_call without the global arena");
            ids.iter().copied().filter(|&e| !matches!(d.a.place(l, e), Place::Vram(_))).collect()
        };
        if need.len() > nst {
            let st = self.arena.as_mut().and_then(|d| d.stage.as_mut()).expect("staging");
            st.stats.fallbacks += 1;
            return Ok(false);
        }
        let b = l % 2;
        let fresh = {
            let d = self.arena.as_ref().expect("arena");
            let st = d.stage.as_ref().expect("staging");
            st.layer[b] == l && st.gen[b] == d.a.generation() && need.iter().all(|e| st.content[b].contains(e))
        };
        {
            let st = self.arena.as_mut().and_then(|d| d.stage.as_mut()).expect("staging");
            st.stats.calls += 1;
            if fresh {
                st.stats.prefetch_used += 1;
            } else {
                st.stats.restaged += 1;
            }
        }
        if !fresh {
            self.stage_issue(l, b, need.clone())?;
        }
        let d = self.arena.as_mut().expect("arena");
        let places = d.a.mark(l, &ids);
        let st = d.stage.as_ref().expect("staging");
        cuda::stream_wait_event(cuda::cur_stream(), st.ready[b]);
        let mut table = vec![0u64; experts];
        let mv = Moves { visits: ids.len() as u64, ..Moves::default() };
        for (&e, &p) in ids.iter().zip(&places) {
            table[e as usize] = match p {
                Place::Vram(v) => arena_addr(&d.chunks, &self.pinned, vpl, ppl, 0, rb, Loc::Vram(v)),
                _ => {
                    st.bufs[b] + st.content[b].iter().position(|&x| x == e).expect("a selected staged id is in the buffer") as u64 * rb
                }
            };
        }
        cuda::to_u64_into(self.tables[l], &table);
        let rows = sel.len() / self.topk;
        run(0, rows, self.tables[l])?;
        cuda::event_record(st.used[b], cuda::cur_stream());
        self.moves[l].add(&mv);
        self.routing_syncs += 1;
        self.sub_batches += 1;
        // the next MoE layer into the other buffer while this one computes
        if l + 1 < nl {
            let d = self.arena.as_ref().expect("arena");
            let next: Vec<u32> = (0..experts as u32).filter(|&e| !matches!(d.a.place(l + 1, e), Place::Vram(_))).take(nst).collect();
            self.stage_issue(l + 1, (l + 1) % 2, next)?;
            self.arena.as_mut().and_then(|d| d.stage.as_mut()).expect("staging").stats.prefetched += 1;
        }
        Ok(true)
    }

    /// free the global arena's own device memory, streams and events
    ///
    /// # Safety
    /// No launch reading the store is pending.
    unsafe fn free_arena(&mut self) {
        let Some(d) = self.arena.as_mut() else { return };
        if let Some(mut r) = d.ring.take() {
            r.free();
        }
        if let Some(mut st) = d.stage.take() {
            cuda::stream_sync(st.stream);
            for b in &mut st.bufs {
                if *b != 0 {
                    cuda::free_dev(b);
                }
            }
            for e in st.ready.into_iter().chain(st.used) {
                cuda::event_destroy(e);
            }
            cuda::stream_destroy(st.stream);
        }
        let flex = d.flex();
        for c in flex {
            if d.chunks[c] != 0 {
                cuda::free_dev(&mut d.chunks[c]);
            }
        }
        d.chunks.truncate(d.base_chunks);
    }
}

#[cfg(test)]
mod arena_tests {
    //! `CROW_GLM_ARENA=global`: the host policy ([`GlobalArena`]), its execution ([`serve_global`])
    //! on a memory twin, the switches, the ring book, and a replay on recorded routing (ignored).
    use super::*;
    use crate::expert_cache::Policy;

    /// the xorshift64 trace of `docs/expert-cache.md` (`expert_cache` tests, the same generator in
    /// Python): `k` distinct ascending ids per token per layer, a hot set drifting every 200 tokens
    fn trace(tokens: usize, layers: usize, experts: u64, k: usize) -> Vec<Vec<Vec<u32>>> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15;
        (0..tokens)
            .map(|t| {
                let base = (t as u64 / 200) * 16;
                (0..layers as u64)
                    .map(|l| {
                        let mut got: Vec<u32> = Vec::with_capacity(k);
                        while got.len() < k {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let e = if (x >> 32) % 10 < 7 { (base + l * 7 + x % 48) % experts } else { (x >> 8) % experts } as u32;
                            if !got.contains(&e) {
                                got.push(e);
                            }
                        }
                        got.sort_unstable();
                        got
                    })
                    .collect()
            })
            .collect()
    }

    fn keys(a: &GlobalArena, p: impl Fn(Place) -> bool) -> Vec<(usize, u32)> {
        (0..a.layers).flat_map(|l| (0..a.experts as u32).map(move |e| (l, e))).filter(|&(l, e)| p(a.place(l, e))).collect()
    }

    #[test]
    fn the_arena_switch_defaults_to_the_per_layer_path() {
        assert_eq!(arena_kind(None), Ok(ArenaKind::Layer));
        assert_eq!(arena_kind(Some("")), Ok(ArenaKind::Layer));
        assert_eq!(arena_kind(Some("layer")), Ok(ArenaKind::Layer));
        assert_eq!(arena_kind(Some(" global ")), Ok(ArenaKind::Global));
        for bad in ["Global", "1", "lru"] {
            assert!(arena_kind(Some(bad)).unwrap_err().contains(ARENA_ENV), "{bad}");
        }
        let none = |_: &str| None;
        assert_eq!(arena_config(&none).unwrap(), ArenaConfig::default());
        assert_eq!(ArenaConfig::default(), ArenaConfig { admit_max: 64, noadmit: false, warm: None, vring: 24, elastic_bytes: 0, stage_bytes: 0, stage_min: 512 });
        let set = |k: &str| match k {
            ARENA_ADMIT_MAX_ENV => Some("16".to_string()),
            ARENA_NOADMIT_ENV => Some("1".to_string()),
            ARENA_WARM_ENV => Some("w.json".to_string()),
            ARENA_VRING_ENV => Some("0".to_string()),
            ARENA_ELASTIC_ENV => Some("10".to_string()),
            ARENA_STAGE_ENV => Some("2.5".to_string()),
            ARENA_STAGE_MIN_ENV => Some("256".to_string()),
            _ => None,
        };
        let c = arena_config(&set).unwrap();
        assert_eq!(c, ArenaConfig { admit_max: 16, noadmit: true, warm: Some("w.json".into()), vring: 0, elastic_bytes: 10 << 30, stage_bytes: 5 << 29, stage_min: 256 });
        for (k, v) in [(ARENA_ADMIT_MAX_ENV, "x"), (ARENA_NOADMIT_ENV, "yes"), (ARENA_VRING_ENV, "-1"), (ARENA_ELASTIC_ENV, "nan"), (ARENA_STAGE_MIN_ENV, "0")] {
            let one = |q: &str| (q == k).then(|| v.to_string());
            assert!(arena_config(&one).unwrap_err().contains(k), "{k}={v}");
        }
    }

    #[test]
    fn admission_is_for_decode_sized_calls_only() {
        assert!(arena_admits(8, false, 64));
        assert!(arena_admits(64, false, 64));
        assert!(!arena_admits(65, false, 64));
        assert!(!arena_admits(8, true, 64), "a prompt call never admits");
        assert!(!arena_admits(1, false, 0));
    }

    /// no quota per layer: one layer may hold every VRAM slot, the next layer's miss takes one
    #[test]
    fn one_layer_can_hold_every_vram_slot() {
        let mut a = GlobalArena::new(3, 16, 4, 0).unwrap();
        a.step(0, &[0, 1, 2, 3], true);
        assert_eq!(keys(&a, |p| matches!(p, Place::Vram(_))), vec![(0, 0), (0, 1), (0, 2), (0, 3)]);
        let ch = a.step(1, &[5], true);
        // every bit set: the hand clears all four and takes slot 0 (expert 0 of layer 0)
        assert_eq!(ch, vec![(0, Place::Vram(0), Place::Nvme), (16 + 5, Place::Nvme, Place::Vram(0))]);
        a.check().unwrap();
    }

    /// a slot pinned by the call (a hit or an admission of the same call) is never a victim; with
    /// none left the call's remaining misses are not admitted
    #[test]
    fn clock_never_evicts_a_slot_the_call_uses() {
        let mut a = GlobalArena::new(1, 16, 2, 4).unwrap();
        a.step(0, &[1, 2], true);
        let ch = a.step(0, &[1, 3, 4], true);
        assert_eq!(a.place(0, 1), Place::Vram(0), "the hit stays");
        assert_eq!(a.place(0, 3), Place::Vram(1), "3 takes 2's slot");
        assert_eq!(a.place(0, 2), Place::Ram(0), "the victim is written back to pinned");
        assert_eq!(a.place(0, 4), Place::Ram(1), "no victim left: 4 lands in pinned, not admitted");
        assert_eq!(a.stats.no_victim, 1);
        assert_eq!(ch.len(), 3);
        a.check().unwrap();
    }

    /// exclusive tiers: an admission frees the pinned copy, the VRAM victim is written back into
    /// pinned as the most recent; the pinned LRU victim is the oldest expert the call does not route
    #[test]
    fn tiers_are_exclusive_and_victims_are_written_back() {
        let mut a = GlobalArena::new(2, 8, 1, 2).unwrap();
        a.step(0, &[0], true);
        a.step(0, &[1], true);
        assert_eq!((a.place(0, 0), a.place(0, 1)), (Place::Ram(0), Place::Vram(0)));
        a.step(0, &[0], true);
        assert_eq!((a.place(0, 0), a.place(0, 1)), (Place::Vram(0), Place::Ram(0)), "0 promoted, 1 written back into the freed slot");
        a.step(1, &[3], true);
        a.step(1, &[4], true);
        // pinned holds 2: (0,1) is the oldest, written back slots are newest
        let ram = keys(&a, |p| matches!(p, Place::Ram(_)));
        assert_eq!(ram.len(), 2);
        assert!(!ram.contains(&(0, 1)), "the oldest pinned expert dropped to the NVMe: {ram:?}");
        assert_eq!(a.stats.ram_evictions, 1);
        a.check().unwrap();
        // protected: a call's own pinned expert is never the LRU victim
        let mut b = GlobalArena::new(1, 8, 0, 1).unwrap();
        b.step(0, &[1], false);
        b.step(0, &[1, 2], false);
        assert_eq!((b.place(0, 1), b.place(0, 2)), (Place::Ram(0), Place::Nvme), "2 cannot evict the call's own 1");
    }

    /// a prompt call (no admission) reads its misses where they lie: NVMe misses land in pinned,
    /// VRAM is untouched
    #[test]
    fn a_call_without_admission_moves_nothing_into_vram() {
        let mut a = GlobalArena::new(1, 16, 4, 8).unwrap();
        a.step(0, &[0, 1], true);
        let ch = a.step(0, &[0, 1, 2, 3, 4], false);
        assert!(ch.iter().all(|c| c.2 != Place::Vram(2) && !matches!(c.2, Place::Vram(_))), "{ch:?}");
        assert_eq!(keys(&a, |p| matches!(p, Place::Vram(_))), vec![(0, 0), (0, 1)]);
        assert_eq!(keys(&a, |p| matches!(p, Place::Ram(_))), vec![(0, 2), (0, 3), (0, 4)]);
    }

    /// `NV_NOADMIT`: NVMe picks land in pinned, pinned hits are still admitted;
    /// `zerocopy` (pin stay): pinned hits stay, NVMe picks are admitted
    #[test]
    fn noadmit_and_pinned_stays() {
        let mut a = GlobalArena::new(1, 16, 4, 8).unwrap();
        a.set_noadmit(true);
        a.step(0, &[5], true);
        assert_eq!(a.place(0, 5), Place::Ram(0));
        a.step(0, &[5], true);
        assert_eq!(a.place(0, 5), Place::Vram(0), "a pinned hit is admitted");
        let mut b = GlobalArena::new(1, 16, 4, 8).unwrap();
        b.step(0, &[5], false);
        b.set_pin_stay(true);
        b.step(0, &[5, 6], true);
        assert_eq!((b.place(0, 5), b.place(0, 6)), (Place::Ram(0), Place::Vram(0)));
    }

    /// disabled slots (ring, a handed-back elastic chunk) are never victims; disabling writes the
    /// owners back; enabling brings the slots back empty; reset keeps them disabled
    #[test]
    fn disabled_slots_are_never_used() {
        let mut a = GlobalArena::new(1, 32, 4, 8).unwrap();
        a.step(0, &[0, 1, 2, 3], true);
        let ch = a.disable(2..4);
        assert_eq!(ch, vec![(2, Place::Vram(2), Place::Ram(0)), (3, Place::Vram(3), Place::Ram(1))]);
        assert_eq!(a.enabled_vram(), 2);
        for t in 0..20u32 {
            a.step(0, &[4 + t % 8], true);
            assert!(keys(&a, |p| matches!(p, Place::Vram(2) | Place::Vram(3))).is_empty());
        }
        a.check().unwrap();
        a.reset();
        assert_eq!(a.enabled_vram(), 2, "reset keeps the disabled slots");
        a.enable(2..4);
        a.add_slots(2);
        assert_eq!((a.enabled_vram(), a.vram_slots()), (6, 6));
        a.step(0, &[0, 1, 2, 3, 4, 5], true);
        assert_eq!(keys(&a, |p| matches!(p, Place::Vram(_))).len(), 6);
        a.check().unwrap();
    }

    /// The VRAM hits of the global CLOCK arena are sybil's `ec_step_k` as `tools/glm_tier_sim.py`
    /// `dyn_run(.., "clock", "global", 64)` replays it (its pinned tier does not change VRAM):
    /// 600 tokens of the shared trace, 42 x 288 experts, top-8; the figures are the sim's.
    #[test]
    fn vram_hits_equal_glm_tier_sim_global_clock() {
        let tr = trace(600, 42, 288, 8);
        assert_eq!(&tr[0][0], &[6, 9, 31, 36, 42, 44, 45, 66], "the generator drifted from the Python one");
        for (cv, cp, hits, admitted) in [(25 * 42, 83 * 42, 46_635u64, 154_965u64), (46 * 42, 124 * 42, 88_748, 112_852), (8 * 42, 0, 16_712, 184_888)] {
            let mut a = GlobalArena::new(42, 288, cv, cp).unwrap();
            for tok in &tr {
                for (l, ids) in tok.iter().enumerate() {
                    a.step(l, ids, arena_admits(ids.len(), false, 64));
                }
            }
            let v: u64 = a.counters().iter().map(|c| c[0]).sum();
            assert_eq!((v, a.stats.admitted), (hits, admitted), "V {cv} P {cp}");
            a.check().unwrap();
        }
    }

    #[test]
    fn warm_plan_and_warm_files() {
        let a = GlobalArena::new(2, 6, 4, 4).unwrap();
        let s = vec![vec![0.0, 5.0, 1.0, 5.0, 3.0, 2.0], vec![]];
        let (v, p) = a.warm_plan(&s);
        assert_eq!(v, vec![vec![1, 3], vec![]], "top 2 by score, ties the lower id");
        assert_eq!(p, vec![vec![4, 5], vec![]]);
        let obj = r#"{"_source": "notes", "model.language_model.layers.3.mlp": [1,2,3], "4": [3,2,1]}"#;
        assert_eq!(parse_warm(obj, 2, 3, 3).unwrap(), vec![vec![1.0, 2.0, 3.0], vec![3.0, 2.0, 1.0]]);
        assert_eq!(parse_warm("[[1,2,3],[0,0,1]]", 2, 3, 3).unwrap()[1], vec![0.0, 0.0, 1.0]);
        for bad in ["[[1,2,3]]", r#"{"9": [1,2,3]}"#, r#"{"3": [1,2]}"#, "nope"] {
            assert!(parse_warm(bad, 2, 3, 3).unwrap_err().contains(ARENA_WARM_ENV), "{bad}");
        }
    }

    #[test]
    fn the_ring_book_reuses_landed_entries_and_waits_for_busy_ones() {
        let mut b = RingBook::new(2);
        let mut none = |_: usize| false;
        assert_eq!(b.take(&mut none), (0, false));
        b.q[0] = 7;
        assert_eq!(b.take(&mut none), (1, false));
        b.q[1] = 9;
        assert_eq!(b.take(&mut none), (0, true), "both busy: the next one, after its D2H");
        b.q[0] = 11;
        assert_eq!(b.pending(9, &mut none), vec![1]);
        assert_eq!(b.pending(9, &mut |r| r == 1), Vec::<usize>::new(), "landed: freed");
        assert_eq!(b.q[1], NONE);
        assert_eq!(b.take(&mut |_| false), (1, false));
    }

    /// The host twin of the device (as `tests::Sim`, over global slots): every slot holds the key
    /// (`layer x experts + expert`) of the record in it; the container "reads" a record as its key.
    struct GSim {
        experts: usize,
        layer: usize,
        vram: Vec<u32>,
        pinned: Vec<u32>,
        stage: Vec<u32>,
        landing: Vec<u32>,
        pending_reads_pinned: Vec<u32>,
        ops: Moves,
    }

    const EMPTY: u32 = u32::MAX - 1;

    impl GSim {
        fn new(experts: usize, vram: usize, pinned: usize, stage: usize) -> GSim {
            GSim { experts, layer: 0, vram: vec![EMPTY; vram], pinned: vec![EMPTY; pinned], stage: vec![EMPTY; stage], landing: vec![EMPTY; stage], pending_reads_pinned: Vec::new(), ops: Moves::default() }
        }
    }

    impl Mover for GSim {
        fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
            assert!(jobs.len() <= MAX_IN_FLIGHT);
            for &(e, d) in jobs {
                let key = (self.layer * self.experts) as u32 + e;
                match d {
                    Dst::Landing(i) => {
                        self.landing[i as usize] = key;
                        self.ops.nvme_to_landing += 1;
                    }
                    Dst::Pinned(q) => {
                        assert!(!self.pending_reads_pinned.contains(&q), "NVMe wrote pinned slot {q} while a queued copy still reads it");
                        self.pinned[q as usize] = key;
                        self.ops.nvme_to_pinned += 1;
                    }
                }
            }
            Ok(jobs.len() as u64)
        }
        fn landing_to_stage(&mut self, i: u32) {
            self.ops.landing_to_stage += 1;
            self.stage[i as usize] = self.landing[i as usize];
        }
        fn pinned_to_stage(&mut self, q: u32, s: u32) {
            self.ops.pinned_to_stage += 1;
            self.pending_reads_pinned.push(q);
            self.stage[s as usize] = self.pinned[q as usize];
        }
        fn vram_to_stage(&mut self, v: u32, s: u32) {
            self.ops.vram_to_stage += 1;
            self.stage[s as usize] = self.vram[v as usize];
        }
        fn barrier(&mut self) {
            self.pending_reads_pinned.clear();
        }
        fn vram_to_pinned(&mut self, v: u32, q: u32) {
            self.ops.vram_to_pinned += 1;
            self.pinned[q as usize] = self.vram[v as usize];
        }
        fn stage_to_vram(&mut self, s: u32, v: u32) {
            self.ops.stage_to_vram += 1;
            self.vram[v as usize] = self.stage[s as usize];
        }
    }

    fn holds(sim: &GSim, l: usize, e: u32, loc: Loc) -> bool {
        let key = (l * sim.experts) as u32 + e;
        key == match loc {
            Loc::Vram(v) => sim.vram[v as usize],
            Loc::Pinned(q) => sim.pinned[q as usize],
            Loc::Stage(s) => sim.stage[s as usize],
        }
    }

    /// Every call of a trace over three layers, at capacities from all-NVMe to roomy, with
    /// admission on and off, `zerocopy`, NOADMIT and slots disabled and enabled on the way: every
    /// selected id is read from a slot that holds its record, the locations come back for exactly
    /// the selected ids in the order the per-layer path returns them, the twin saw the moves
    /// `serve_global` counts, and the arena's invariants hold.
    #[test]
    fn every_location_holds_its_record() {
        let (nl, ex, k) = (3usize, 24u64, 6usize);
        let tr = trace(160, nl, ex, k);
        for (v, p) in [(0, 0), (0, 9), (3, 0), (6, 6), (9, 30), (24, 12)] {
            for variant in 0..4 {
                let mut a = GlobalArena::new(nl, ex as usize, v, p).unwrap();
                a.set_pin_stay(variant == 1);
                a.set_noadmit(variant == 2);
                let mut sim = GSim::new(ex as usize, v, p, 2 * k);
                let mut cache = ExpertCache::new(Policy::Lru, Scope::PerLayer, nl, ex as usize, v / nl, p / nl).unwrap();
                let mut slots: Vec<LayerSlots> = (0..nl).map(|_| LayerSlots::new(ex as usize, TierSizes { vram: v / nl, pinned: p / nl })).collect();
                let mut psim = tests_sim_free::PSim::default();
                for (t, tok) in tr.iter().enumerate() {
                    if variant == 3 && v >= 6 && t == 60 {
                        let ch = a.disable(v - 3..v);
                        write_back(&ch, &mut sim).unwrap();
                        sim.barrier();
                    }
                    if variant == 3 && v >= 6 && t == 120 {
                        a.enable(v - 3..v);
                    }
                    for (l, ids) in tok.iter().enumerate() {
                        let admit = arena_admits(ids.len() * if t % 7 == 3 { 20 } else { 1 }, false, 64);
                        sim.layer = l;
                        let before = sim.ops;
                        let served = serve_global(&mut a, l, ids, admit, 2 * k, &mut sim).unwrap();
                        sim.barrier();
                        for &(e, loc) in &served.locs {
                            assert!(holds(&sim, l, e, loc), "V {v} P {p} variant {variant} token {t} layer {l}: expert {e} at {loc:?}");
                        }
                        let ops = sim.ops.since(&before);
                        let m = served.moves;
                        assert_eq!(
                            (ops.nvme_to_landing, ops.nvme_to_pinned, ops.landing_to_stage, ops.pinned_to_stage, ops.vram_to_stage, ops.vram_to_pinned, ops.stage_to_vram),
                            (m.nvme_to_landing, m.nvme_to_pinned, m.landing_to_stage, m.pinned_to_stage, m.vram_to_stage, m.vram_to_pinned, m.stage_to_vram),
                            "V {v} P {p} variant {variant} token {t} layer {l}: moves"
                        );
                        // ids identical: the per-layer path serves the same ids in the same order
                        let per = serve(&mut cache, l, &mut slots[l], ids, 2 * k, &mut psim).unwrap();
                        assert_eq!(per.locs.iter().map(|x| x.0).collect::<Vec<_>>(), served.locs.iter().map(|x| x.0).collect::<Vec<_>>());
                        assert_eq!(served.locs.iter().map(|x| x.0).collect::<Vec<_>>(), *ids);
                    }
                    a.check().unwrap_or_else(|e| panic!("V {v} P {p} variant {variant} token {t}: {e}"));
                }
            }
        }
    }

    mod tests_sim_free {
        use super::super::*;
        /// a mover that records nothing (the per-layer path's ids only)
        #[derive(Default)]
        pub struct PSim;
        impl Mover for PSim {
            fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
                Ok(jobs.len() as u64)
            }
            fn landing_to_stage(&mut self, _: u32) {}
            fn pinned_to_stage(&mut self, _: u32, _: u32) {}
            fn vram_to_stage(&mut self, _: u32, _: u32) {}
            fn barrier(&mut self) {}
            fn vram_to_pinned(&mut self, _: u32, _: u32) {}
            fn stage_to_vram(&mut self, _: u32, _: u32) {}
        }
    }

    /// a prompt call through the global arena: row sub-batches that fit the staging slots, every
    /// row's ids read from slots that hold them, nothing admitted
    #[test]
    fn a_prompt_call_is_served_in_sub_batches_without_admission() {
        let (ex, k) = (32usize, 4usize);
        let mut a = GlobalArena::new(2, ex, 4, 0).unwrap();
        a.step(1, &[0, 1, 2, 3], true);
        let mut sim = GSim::new(ex, 4, 0, 8);
        sim.layer = 1;
        // warm the twin's VRAM like the arena (the step above moved no bytes)
        for e in 0..4u32 {
            if let Place::Vram(v) = a.place(1, e) {
                sim.vram[v as usize] = ex as u32 + e;
            }
        }
        let sel: Vec<i32> = (0..8).flat_map(|r| (0..k as i32).map(move |j| (r * 3 + j * 5) % ex as i32)).collect();
        let mut rows_seen = Vec::new();
        let mut each = |r0: usize, rows: usize, s: &Served| -> Result<(), String> {
            rows_seen.push((r0, rows));
            assert_eq!(s.moves.n2v + s.moves.p2v, 0, "no admission in a prompt call");
            Ok(())
        };
        let n = serve_chunk_global(&mut a, 1, &sel, k, 8, &mut sim, &mut each).unwrap();
        assert!(n > 1, "the 8 rows do not fit 8 staging slots at once");
        assert_eq!(rows_seen.iter().map(|x| x.1).sum::<usize>(), 8);
        assert_eq!(keys(&a, |p| matches!(p, Place::Vram(_))), vec![(1, 0), (1, 1), (1, 2), (1, 3)]);
        assert!(fitting_rows_global(&a, 1, &sel[..3], k, 8).unwrap_err().contains("no [rows]"));
    }

    /// the chunk mover addresses global slot `g` as slot `g % per` of chunk `g / per`
    #[test]
    fn the_chunk_mover_maps_global_slots_onto_the_chunks() {
        #[derive(Default)]
        struct Rec {
            vram: Dev,
            pinned: u64,
            log: Vec<(&'static str, u64, u32, u64, u32)>,
        }
        impl Mover for Rec {
            fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
                for &(_, d) in jobs {
                    if let Dst::Pinned(q) = d {
                        self.log.push(("nvme", 0, 0, self.pinned, q));
                    }
                }
                Ok(0)
            }
            fn landing_to_stage(&mut self, _: u32) {}
            fn pinned_to_stage(&mut self, q: u32, _: u32) {
                self.log.push(("p2s", 0, 0, self.pinned, q));
            }
            fn vram_to_stage(&mut self, v: u32, _: u32) {
                self.log.push(("v2s", self.vram, v, 0, 0));
            }
            fn barrier(&mut self) {}
            fn vram_to_pinned(&mut self, v: u32, q: u32) {
                self.log.push(("v2p", self.vram, v, self.pinned, q));
            }
            fn stage_to_vram(&mut self, _: u32, v: u32) {
                self.log.push(("s2v", self.vram, v, 0, 0));
            }
        }
        impl<'a> Rebase<'a> for Rec {
            fn set_vram(&mut self, base: Dev) {
                self.vram = base;
            }
            fn set_pinned(&mut self, p: Option<&'a Pinned>) {
                self.pinned = p.map_or(0, |p| p.dev);
            }
            fn stream(&self) -> sys::CUstream {
                std::ptr::null_mut()
            }
        }
        let pinned: Vec<Pinned> = (0..3).map(|i| { let mut p: Pinned = unsafe { std::mem::zeroed() }; p.dev = 1000 * (i + 1); p }).collect();
        let vram = [10u64, 20, 30];
        let mut m = ChunkMover { inner: Rec::default(), vram: &vram, pinned: &pinned, vpl: 4, ppl: 5, rb: 1, ring: None };
        m.vram_to_pinned(9, 7);
        m.stage_to_vram(0, 4);
        m.pinned_to_stage(14, 0);
        m.nvme(&[(1, Dst::Pinned(0)), (2, Dst::Pinned(12)), (3, Dst::Pinned(13))]).unwrap();
        assert_eq!(
            m.inner.log,
            vec![("v2p", 30, 1, 2000, 2), ("s2v", 20, 0, 0, 0), ("p2s", 0, 0, 3000, 4), ("nvme", 0, 0, 1000, 0), ("nvme", 0, 0, 3000, 2), ("nvme", 0, 0, 3000, 3)]
        );
        std::mem::forget(pinned);
    }

    /// The replay of recorded routing (the held-out file of `tools/glm_tier_sim.py`, exported as
    /// `routes.u16` `[n][42][8]` and `mask.u8` `[n]`, 1 = generated, into `$ARENA_REPLAY_DIR`):
    /// token by token from an empty cache, the per-layer LRU of the path of record against the
    /// global arena at the same slots (`$ARENA_REPLAY_SLOTS` = V:P per layer, default 46:124),
    /// counted over the generated positions. `$ARENA_REPLAY_WARM` = a warm-start file for the arena.
    #[test]
    #[ignore = "needs exported routing: ARENA_REPLAY_DIR=<dir> cargo test --release --lib arena_replay -- --ignored --nocapture"]
    fn arena_replay_on_recorded_routing() {
        let dir = std::env::var("ARENA_REPLAY_DIR").expect("ARENA_REPLAY_DIR");
        let r = std::fs::read(format!("{dir}/routes.u16")).unwrap();
        let mask = std::fs::read(format!("{dir}/mask.u8")).unwrap();
        let (nl, ex, k) = (42usize, 288usize, 8usize);
        let n = mask.len();
        assert_eq!(r.len(), n * nl * k * 2);
        let id = |t: usize, l: usize, j: usize| u16::from_le_bytes([r[((t * nl + l) * k + j) * 2], r[((t * nl + l) * k + j) * 2 + 1]]) as u32;
        let slots = std::env::var("ARENA_REPLAY_SLOTS").unwrap_or("46:124".into());
        let (v, p) = slots.split_once(':').map(|(a, b)| (a.parse::<usize>().unwrap(), b.parse::<usize>().unwrap())).unwrap();
        let warm = std::env::var("ARENA_REPLAY_WARM").ok().map(|f| parse_warm(&std::fs::read_to_string(f).unwrap(), nl, 3, ex).unwrap());
        let gen = mask.iter().filter(|&&m| m == 1).count() as f64;
        let visits = gen * (nl * k) as f64;
        let report = |name: &str, c: [u64; 3]| {
            println!(
                "{name:<40} VRAM hit {:6.2} %  VRAM hits/token {:6.1}  pinned/token {:6.1}  NVMe reads/token {:6.2}",
                100.0 * c[0] as f64 / visits,
                c[0] as f64 / gen,
                c[1] as f64 / gen,
                c[2] as f64 / gen
            );
        };
        println!("replay: {n} positions, {gen} generated, V {v} P {p} per layer");
        for stay in [false, true] {
            let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, nl, ex, v, p).unwrap();
            c.set_pinned_stays(stay);
            let mut tot = [0u64; 3];
            for t in 0..n {
                let before: [u64; 3] = c.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                for l in 0..nl {
                    let ids: Vec<u32> = (0..k).map(|j| id(t, l, j)).collect();
                    c.observe_token(l, &ids);
                }
                if mask[t] == 1 {
                    let after: [u64; 3] = c.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                    (0..3).for_each(|i| tot[i] += after[i] - before[i]);
                }
            }
            report(&format!("per-layer LRU {}", if stay { "zerocopy" } else { "promote" }), tot);
        }
        for (stay, noadmit, vring) in [(false, false, 0), (false, false, 24), (true, false, 24), (false, true, 24)] {
            let mut a = GlobalArena::new(nl, ex, nl * v, nl * p).unwrap();
            if vring > 0 {
                a.disable(nl * v - vring..nl * v);
            }
            a.set_pin_stay(stay);
            a.set_noadmit(noadmit);
            if let Some(w) = &warm {
                let (vr, pr) = a.warm_plan(w);
                for l in 0..nl {
                    a.step(l, &vr[l], true);
                    a.step(l, &pr[l], false);
                }
                a.clear_counters();
            }
            let mut tot = [0u64; 3];
            for t in 0..n {
                let before: [u64; 3] = a.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                for l in 0..nl {
                    let ids: Vec<u32> = (0..k).map(|j| id(t, l, j)).collect();
                    a.step(l, &ids, arena_admits(k, false, 64));
                }
                if mask[t] == 1 {
                    let after: [u64; 3] = a.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                    (0..3).for_each(|i| tot[i] += after[i] - before[i]);
                }
            }
            a.check().unwrap();
            report(&format!("global CLOCK{}{}{} ring {vring}", if stay { " zerocopy" } else { "" }, if noadmit { " noadmit" } else { "" }, if warm.is_some() { " warm" } else { "" }), tot);
        }
    }
}

#[cfg(test)]
mod arena_gpu_tests {
    //! `CROW_GLM_ARENA=global` on the device, on a synthetic container of three MoE layers x 16
    //! experts (455 MB in the temp dir, about 1 GB VRAM). `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_arena_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;

    const REC: u64 = 9_474_048;

    struct Synth {
        dir: std::path::PathBuf,
        path: String,
    }

    impl Drop for Synth {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    /// `layers` MoE layers (decoder layers 3..) of `experts` MUL1 records each, random bytes
    fn synth(layers: u32, experts: u32) -> Synth {
        use std::io::Write;
        let dir = std::env::temp_dir().join(format!("crow-glm5-arena-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-arena.cnq");
        let lead = 4084u64;
        let mut tensors = vec![serde_json::json!({ "name": "model.language_model.layers.3.mlp.gate.weight", "section": "text",
            "dtype": "bf16", "offset": 0, "n_values": lead / 2, "shape": [2, lead / 4] })];
        for l in 0..layers {
            for e in 0..experts {
                for (k, p) in ["gate", "up", "down"].into_iter().enumerate() {
                    tensors.push(serde_json::json!({ "name": crate::nvme_source::glm5_expert_tensor_name(3 + l, e, p), "section": "text", "dtype": "mul1",
                        "offset": lead + (l * experts + e) as u64 * REC + k as u64 * (REC / 3), "n_values": 2048u64 * 4096, "shape": [2048, 4096] }));
                }
            }
        }
        let tail = lead + (layers * experts) as u64 * REC;
        tensors.push(serde_json::json!({ "name": "model.language_model.norm.weight", "section": "text", "dtype": "bf16", "offset": tail, "n_values": 2048, "shape": [2048] }));
        let sha = |s: &str| crate::cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "blob_offset": 12, "recipe": "synthetic-glm5-arena",
            "model": { "family": "Glm5Next", "model_type": "glm5_next_text", "config_json": "{}", "config_json_sha256": sha("{}"),
                "generation_config_json": "{}", "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-glm5-arena", "revision": "arena", "shards": [] }, "geo": {} },
            "tensors": tensors
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        let mut x = 0x0A7E_4A5E_1D2C_3B4Fu64;
        let mut chunk = vec![0u8; 1 << 20];
        let mut left = tail + 4096;
        while left > 0 {
            for w in chunk.chunks_exact_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.copy_from_slice(&x.to_le_bytes());
            }
            let n = left.min(chunk.len() as u64) as usize;
            f.write_all(&chunk[..n]).unwrap();
            left -= n as u64;
        }
        f.write_all(&ib).unwrap();
        f.write_all(&(ib.len() as u64).to_le_bytes()).unwrap();
        f.flush().unwrap();
        Synth { dir, path: path.to_str().unwrap().to_string() }
    }

    /// set the arena's variables for one scope, the previous values back on drop
    struct Env(Vec<(String, Option<String>)>);

    impl Env {
        fn set(kv: &[(&str, &str)]) -> Env {
            let names = [ARENA_ENV, ARENA_ADMIT_MAX_ENV, ARENA_NOADMIT_ENV, ARENA_WARM_ENV, ARENA_VRING_ENV, ARENA_ELASTIC_ENV, ARENA_STAGE_ENV, ARENA_STAGE_MIN_ENV];
            let old = names.iter().map(|n| (n.to_string(), std::env::var(n).ok())).collect();
            for n in names {
                std::env::remove_var(n);
            }
            for (k, v) in kv {
                std::env::set_var(k, v);
            }
            Env(old)
        }
    }

    impl Drop for Env {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    /// the xorshift routing of `tests::trace` (drifting hot set), `k` distinct ids per layer
    fn routing(tokens: usize, layers: usize, experts: u64, k: usize, seed: u64) -> Vec<Vec<Vec<u32>>> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed;
        (0..tokens)
            .map(|t| {
                let base = (t as u64 / 10) * 3;
                (0..layers as u64)
                    .map(|l| {
                        let mut got: Vec<u32> = Vec::with_capacity(k);
                        while got.len() < k {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let e = if (x >> 32) % 10 < 7 { (base + l * 5 + x % 9) % experts } else { (x >> 8) % experts } as u32;
                            if !got.contains(&e) {
                                got.push(e);
                            }
                        }
                        got
                    })
                    .collect()
            })
            .collect()
    }

    /// Every decode call (one row; every third call two rows, still admitting) and every prompt
    /// call (sub-batches, and staged with the staging buffers and the elastic part) of a routing
    /// trace over three MoE layers, under the arena's switches: each selected id's table entry
    /// holds the bytes of its layer's record (`Cnq::read_range`), every other entry is 0.
    #[test]
    #[ignore = "needs the GPU (about 1 GB VRAM, a 455 MB synthetic container in the temp dir): cargo test --release --lib glm5_arena_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_arena_gpu_every_table_entry_holds_its_record() {
        let (nl, ex) = (3usize, 16u32);
        let s = synth(nl as u32, ex);
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk) = (3 + nl, 3, ex as usize, 8);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let rb = spec.bytes as usize;
        let want: Vec<Vec<Vec<u8>>> =
            (0..nl).map(|l| (0..ex).map(|e| cnq.read_range(&cnq.find(&crate::nvme_source::glm5_expert_tensor_name(3 + l as u32, e, "gate"), "text").clone(), 0, rb)).collect()).collect();
        let tr = routing(20, nl, ex as u64, 8, 0xA4E);
        let warm = s.dir.join("warm.json");
        std::fs::write(&warm, serde_json::to_string(&(0..nl).map(|l| (0..ex).map(|e| ((e as usize * 7 + l) % 16) as f64).collect::<Vec<_>>()).collect::<Vec<_>>()).unwrap()).unwrap();
        let warm = warm.to_str().unwrap().to_string();
        let stage_gb = format!("{}", 12.0 * REC as f64 / (1u64 << 30) as f64);
        let elastic_gb = format!("{}", 2.0 * 3.0 * REC as f64 / (1u64 << 30) as f64);
        let check = |what: &str, l: usize, tb: Dev, ids: &[u32]| unsafe {
            cuda::sync();
            let table = cuda::dtoh_u64(tb, ex as usize);
            for e in 0..ex {
                assert_eq!(table[e as usize] != 0, ids.contains(&e), "{what}: layer {l} table entry of expert {e}");
                if ids.contains(&e) {
                    let got: Vec<u8> = cuda::dtoh_t(table[e as usize], rb);
                    assert!(got == want[l][e as usize], "{what}: layer {l} expert {e}: the bytes differ from read_range");
                }
            }
        };
        unsafe {
            let _ctx = cuda::Ctx::init();
            let cases: Vec<(&str, Vec<(&str, &str)>, bool)> = vec![
                ("plain", vec![(ARENA_VRING_ENV, "0")], false),
                ("ring", vec![(ARENA_VRING_ENV, "2")], false),
                ("ring+stager", vec![(ARENA_VRING_ENV, "2")], true),
                ("noadmit+admit16", vec![(ARENA_VRING_ENV, "2"), (ARENA_NOADMIT_ENV, "1"), (ARENA_ADMIT_MAX_ENV, "16")], false),
                ("warm", vec![(ARENA_VRING_ENV, "2"), (ARENA_WARM_ENV, &warm)], false),
                ("stage", vec![(ARENA_VRING_ENV, "2"), (ARENA_STAGE_ENV, &stage_gb), (ARENA_STAGE_MIN_ENV, "16")], false),
                ("stage+elastic", vec![(ARENA_VRING_ENV, "2"), (ARENA_STAGE_ENV, &stage_gb), (ARENA_STAGE_MIN_ENV, "16"), (ARENA_ELASTIC_ENV, &elastic_gb)], false),
            ];
            for (v, p) in [(0, 0), (2, 3), (3, 8), (6, 0)] {
                for (name, kv, stager) in &cases {
                    if v == 0 && *name != "plain" && !name.starts_with("stage") {
                        continue;
                    }
                    let mut kv = kv.clone();
                    kv.push((ARENA_ENV, "global"));
                    let _env = Env::set(&kv);
                    let what = format!("{name} V {v} P {p}");
                    let sizes = TierSizes { vram: v, pinned: p };
                    let mut t = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, 16).unwrap();
                    assert_eq!(t.arena_kind(), ArenaKind::Global);
                    if *stager {
                        t.set_stager(true).unwrap();
                    }
                    if *name == "warm" && v > 0 {
                        let a = t.arena().unwrap();
                        assert!((0..nl).all(|l| (0..ex).any(|e| matches!(a.place(l, e), Place::Vram(_)))), "{what}: every layer warmed into VRAM");
                    }
                    for (i, tok) in tr.iter().enumerate() {
                        for (l, ids) in tok.iter().enumerate() {
                            // every third call two rows (16 picks), the second row the first reversed
                            let mut sel: Vec<i32> = ids.iter().map(|&e| e as i32).collect();
                            if i % 3 == 2 {
                                sel.extend(ids.iter().rev().map(|&e| e as i32));
                            }
                            let (tb, served) = t.table_for(3 + l, &sel).unwrap();
                            assert_eq!(served.moves.visits as usize, ids.len(), "{what}: visits");
                            check(&what, l, tb, ids);
                        }
                        // every tenth token a prompt call of 4 rows per layer (32 picks: staged at 16)
                        if i % 10 == 9 {
                            for l in 0..nl {
                                let rows: Vec<&Vec<u32>> = (0..4).map(|r| &tr[i - r][l]).collect();
                                let sel: Vec<i32> = rows.iter().flat_map(|r| r.iter().map(|&e| e as i32)).collect();
                                let mut calls = Vec::new();
                                t.tables_for_chunk(3 + l, &sel, &mut |r0, n, tb| {
                                    let mut ids: Vec<u32> = sel[r0 * 8..(r0 + n) * 8].iter().map(|&e| e as u32).collect();
                                    ids.sort_unstable();
                                    ids.dedup();
                                    check(&what, l, tb, &ids);
                                    calls.push(n);
                                    Ok(())
                                })
                                .unwrap();
                                assert_eq!(calls.iter().sum::<usize>(), 4, "{what}: every row served");
                            }
                        }
                    }
                    let a = t.arena().unwrap();
                    a.check().unwrap();
                    let tot: [u64; 3] = t.tier_counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                    eprintln!(
                        "glm5_arena synthetic {what}: accesses v/p/n {tot:?}, NVMe reads {}, ring {:?}, elastic {:?}, stage {:?}",
                        t.nvme_reads,
                        t.arena_ring_stats(),
                        t.arena_elastic_stats(),
                        t.arena_stage_stats()
                    );
                    if name.starts_with("stage") && v >= 6 {
                        let st = t.arena_stage_stats().unwrap().0;
                        assert!(st.calls > 0 && st.prefetch_used > 0, "{what}: staged calls ran, some on their prefetch: {st:?}");
                    }
                    t.reset_cache().unwrap();
                    let sel: Vec<i32> = tr[0][0].iter().map(|&e| e as i32).collect();
                    let (tb, served) = t.table_for(3, &sel).unwrap();
                    assert_eq!(served.nvme_reads, tr[0][0].len(), "{what}: a reset arena reads the first selection from NVMe");
                    check(&what, 0, tb, &tr[0][0]);
                    t.free();
                }
            }
        }
        drop(cnq);
    }
}

// ---------------------------------------------------------------- the device store

/// a 4096-aligned host buffer (the NVMe landing of the staging slots; pageable, not pinned, so
/// it does not count against `HOST_PINNED_CAP`)
struct Landing {
    p: *mut u8,
    layout: Option<std::alloc::Layout>,
}

impl Landing {
    fn new(bytes: usize) -> Landing {
        if bytes == 0 {
            return Landing { p: std::ptr::null_mut(), layout: None };
        }
        let layout = std::alloc::Layout::from_size_align(bytes, crate::nvme_source::ALIGN as usize).expect("landing layout");
        let p = unsafe { std::alloc::alloc(layout) };
        assert!(!p.is_null(), "expert tiers: the {bytes} B landing buffer was refused");
        Landing { p, layout: Some(layout) }
    }
}

impl Drop for Landing {
    fn drop(&mut self) {
        if let Some(l) = self.layout.take() {
            unsafe { std::alloc::dealloc(self.p, l) };
        }
    }
}

/// what one token cost in the tiers
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TierTick {
    pub nvme_reads: u64,
    pub nvme_bytes: u64,
}

/// The three tiers of every MoE layer on the device: cache, slots, the VRAM and pinned arenas,
/// the record tables, the staging slots and the NVMe reader.
pub struct ExpertTiers {
    pub sizes: TierSizes,
    pub cache: ExpertCache,
    pub first_moe: usize,
    pub rb: u64,
    pub stage_cap: usize,
    slots: Vec<LayerSlots>,
    /// `CROW_GLM_ARENA=global`: the global arena over the same slots (`None` = per layer)
    arena: Option<ArenaDev>,
    vram: Vec<Dev>,
    pinned: Vec<Pinned>,
    tables: Vec<Dev>,
    stage: Dev,
    landing: Landing,
    records: Vec<Vec<ExpertRecord>>,
    src: NvmeSource,
    /// since construction
    pub nvme_reads: u64,
    pub nvme_bytes: u64,
    /// #187: the [`Moves`] of every MoE layer since construction (host counters)
    pub moves: Vec<Moves>,
    /// #186: MoE calls served since construction (one routing sync each) and the [`serve`]
    /// sub-batches they took (a decode call one, a prompt call [`serve_chunk`]'s count)
    pub routing_syncs: u64,
    pub sub_batches: u64,
    /// #186: the prefill staging set of prompt calls (0 slots until [`ExpertTiers::alloc_prefill_stage`])
    pf_cap: usize,
    pf_stage: Dev,
    /// the prompt calls' pinned landing ring and copy stream ([`PrefillMover`])
    pf_ring: Option<PfRing>,
    /// `CROW_GLM_ARENA=global`: the pageable NVMe landing of its prompt calls (`pf_cap` records,
    /// allocated with their first call); the per-layer path lands in `pf_ring` instead
    arena_landing: Landing,
    /// #188: `CROW_GLM_PINNED` / `CROW_GLM_CPU_LANE` as read at construction
    pub pinned_use: PinnedUse,
    /// the pinned arenas are write-combined (`CROW_PINNED_ALLOC`)
    pinned_wc: bool,
    topk: usize,
    lane_clock: std::sync::Arc<crate::glm5_moe::lane::Clock>,
    /// the routed-expert geometry (the CPU lane under the controller builds its experts from it)
    lane_geo: MoeGeo,
    /// `CROW_GLM_CONTROLLER` with the CPU lane: the lane's device side (lane and stager on)
    dev_lane: Option<glm5_flags::DevLane>,
    /// `CROW_GLM_CPU_LANE=split` as read at construction: the cost model of [`plan_split`]
    /// (`None` = `1` or off: the lane, if on, takes every pinned id)
    pub split: Option<SplitCost>,
    /// `[moe layer][expert]` selections so far (decode and prompt calls), the split's heat:
    /// colder pinned ids go to the CPU first
    heat: Vec<u32>,
    /// #149 path B (`CROW_GLM_STAGER=1`): the stager stream and its pinned sources (`None` = off)
    stager: Option<Stager>,
    /// `CROW_GLM_PREFETCH=1`: the pinned store of the next layer's guessed records (`None` = off)
    prefetch: Option<glm5_flags::Prefetch>,
}

struct GpuMover<'a> {
    vram: Dev,
    pinned: Option<&'a Pinned>,
    stage: Dev,
    landing: *mut u8,
    rb: u64,
    src: &'a NvmeSource,
    recs: &'a [ExpertRecord],
}

impl Mover for GpuMover<'_> {
    fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
        let rb = self.rb as usize;
        let mut v = Vec::with_capacity(jobs.len());
        for &(e, d) in jobs {
            let gu = match d {
                Dst::Landing(i) => unsafe { self.landing.add(i as usize * rb) },
                Dst::Pinned(q) => unsafe { (self.pinned.expect("a pinned destination without a pinned arena").host as *mut u8).add(q as usize * rb) },
            };
            v.push((self.recs[e as usize], RecordDst { gu, dn: std::ptr::null_mut() }));
        }
        // SAFETY: the destinations are this store's landing buffer and pinned arena, `rb` bytes
        // each, and nothing reads them until `wait` returned (phase A/B order of `serve`)
        let t = unsafe { self.src.fetch(&v) }?;
        Ok(self.src.wait(t)?.bytes)
    }
    fn landing_to_stage(&mut self, i: u32) {
        let rb = self.rb as usize;
        unsafe { cuda::upload_into(self.stage + i as u64 * self.rb, std::slice::from_raw_parts(self.landing.add(i as usize * rb), rb)) };
    }
    fn pinned_to_stage(&mut self, q: u32, s: u32) {
        let p = self.pinned.expect("a pinned source without a pinned arena");
        unsafe { cuda::upload_from_pinned(self.stage + s as u64 * self.rb, (p.host as *const u8).add(q as usize * self.rb as usize) as *const _, self.rb as usize) };
    }
    fn vram_to_stage(&mut self, v: u32, s: u32) {
        unsafe { cuda::d2d_async(self.stage + s as u64 * self.rb, self.vram + v as u64 * self.rb, self.rb as usize) };
    }
    fn barrier(&mut self) {
        unsafe { cuda::sync() };
    }
    fn vram_to_pinned(&mut self, v: u32, q: u32) {
        let p = self.pinned.expect("a pinned destination without a pinned arena");
        unsafe {
            cuda::ck(sys::cuMemcpyDtoHAsync_v2(
                (p.host as *mut u8).add(q as usize * self.rb as usize) as *mut _,
                self.vram + v as u64 * self.rb,
                self.rb as usize,
                cuda::cur_stream(),
            ))
        };
    }
    fn stage_to_vram(&mut self, s: u32, v: u32) {
        unsafe { cuda::d2d_async(self.vram + v as u64 * self.rb, self.stage + s as u64 * self.rb, self.rb as usize) };
    }
}

/// The copy machinery of prompt calls (template: glm53-flash-offload's prefill NVMe reads land in
/// pinned RAM and the copy engine moves them): a [`crate::manager::GLM5_PREFILL_RING`]-record
/// pinned NVMe landing ring, a non-blocking copy stream, one event per ring record (its last
/// H2D), per staging half the event "the kernels of its last sub-batch are done", and the event
/// "the current sub-batch's copies are done". Allocated with the prefill staging set.
struct PfRing {
    ring: Pinned,
    ring_ev: Vec<sys::CUevent>,
    cs: sys::CUstream,
    free_ev: [sys::CUevent; 2],
    ready_ev: sys::CUevent,
    cursor: usize,
}

impl PfRing {
    unsafe fn new(rb: u64) -> PfRing {
        let n = crate::manager::GLM5_PREFILL_RING;
        PfRing {
            ring: Pinned::alloc(n * rb as usize),
            ring_ev: (0..n).map(|_| cuda::event_create()).collect(),
            cs: cuda::stream_create_non_blocking(),
            free_ev: [cuda::event_create(), cuda::event_create()],
            ready_ev: cuda::event_create(),
            cursor: 0,
        }
    }

    unsafe fn free(&mut self) {
        cuda::stream_sync(self.cs);
        for &e in self.ring_ev.iter().chain(self.free_ev.iter()).chain([&self.ready_ev]) {
            cuda::event_destroy(e);
        }
        cuda::stream_destroy(self.cs);
        self.ring.free();
    }
}

/// The [`Mover`] of prompt calls ([`serve_prefill`] issues only `nvme`, `landing_to_stage` and
/// `pinned_to_stage`): every H2D is a `cuMemcpyHtoDAsync` from pinned memory on the copy stream,
/// so the copies of sub-batch `j` run while the compute stream runs sub-batch `j - 1`. Sub-batch
/// `j` uses staging half `j % halves`; its copies wait until the kernels of the last sub-batch on
/// that half are done, and the compute stream waits for its copies. An NVMe read lands in the
/// next ring record once that record's previous H2D finished (the host waits on its event).
struct PrefillMover<'a> {
    pinned: Option<&'a Pinned>,
    stage: Dev,
    half: usize,
    halves: usize,
    base: Dev,
    rb: u64,
    src: &'a NvmeSource,
    recs: &'a [ExpertRecord],
    pf: &'a mut PfRing,
    /// staging slot of the current sub-batch -> the ring record its NVMe read landed in
    landed: Vec<(u32, usize)>,
}

impl Mover for PrefillMover<'_> {
    fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
        let (rb, n) = (self.rb as usize, self.pf.ring_ev.len());
        assert!(jobs.len() <= n, "prefill ring: {} reads, {n} ring records", jobs.len());
        let mut v = Vec::with_capacity(jobs.len());
        for &(e, d) in jobs {
            let Dst::Landing(i) = d else { panic!("a prompt call reads no record into a pinned slot") };
            let r = self.pf.cursor;
            self.pf.cursor = (r + 1) % n;
            // the record's previous H2D is done before the read overwrites it
            unsafe { cuda::ck(sys::cuEventSynchronize(self.pf.ring_ev[r])) };
            self.landed.retain(|&(s, _)| s != i);
            self.landed.push((i, r));
            let gu = unsafe { (self.pf.ring.host as *mut u8).add(r * rb) };
            v.push((self.recs[e as usize], RecordDst { gu, dn: std::ptr::null_mut() }));
        }
        // SAFETY: the destinations are ring records no queued copy reads any more
        let t = unsafe { self.src.fetch(&v) }?;
        Ok(self.src.wait(t)?.bytes)
    }
    fn landing_to_stage(&mut self, i: u32) {
        let r = self.landed.iter().find(|&&(s, _)| s == i).expect("a landing is staged after its read").1;
        unsafe {
            let src = (self.pf.ring.host as *const u8).add(r * self.rb as usize);
            cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.base + i as u64 * self.rb, src as *const _, self.rb as usize, self.pf.cs));
            cuda::event_record(self.pf.ring_ev[r], self.pf.cs);
        }
    }
    fn pinned_to_stage(&mut self, q: u32, s: u32) {
        let p = self.pinned.expect("a pinned source without a pinned arena");
        unsafe {
            let src = (p.host as *const u8).add(q as usize * self.rb as usize);
            cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.base + s as u64 * self.rb, src as *const _, self.rb as usize, self.pf.cs));
        }
    }
    fn vram_to_stage(&mut self, _: u32, _: u32) {
        unreachable!("a prompt call stages no VRAM record")
    }
    fn barrier(&mut self) {
        unreachable!("a prompt call has no barrier")
    }
    fn vram_to_pinned(&mut self, _: u32, _: u32) {
        unreachable!("a prompt call admits nothing")
    }
    fn stage_to_vram(&mut self, _: u32, _: u32) {
        unreachable!("a prompt call admits nothing")
    }
    fn begin_batch(&mut self, j: usize) {
        let h = j % self.halves;
        self.base = self.stage + (h * self.half) as u64 * self.rb;
        self.landed.clear();
        // the copies into this half wait for the kernels of its last sub-batch
        unsafe { cuda::stream_wait_event(self.pf.cs, self.pf.free_ev[h]) };
    }
    fn end_batch(&mut self) {
        unsafe {
            cuda::event_record(self.pf.ready_ev, self.pf.cs);
            // WDDM: submit the copy stream's batch now
            cuda::stream_query(self.pf.cs);
            cuda::stream_wait_event(cuda::cur_stream(), self.pf.ready_ev);
        }
    }
}

impl ExpertTiers {
    /// Allocate the tiers of every MoE layer of `g` (cache empty, G1d: no seed) and open the
    /// container `path` for unbuffered reads with `readers` threads (1 = PREREG amendment 5).
    /// `stage_cap` staging slots in VRAM (and as many landing records in pageable RAM): the
    /// selected experts of one call (`topk` for a decode row).
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(cnq: &Cnq, path: &str, g: &Glm5Geo, moe: &MoeGeo, sizes: TierSizes, readers: usize, stage_cap: usize) -> Result<ExpertTiers, String> {
        let policy = expert_cache::policy_for(expert_cache::GLM5_NEXT_FAMILY, std::env::var(expert_cache::ENV).ok().as_deref(), false)?
            .ok_or_else(|| format!("{}=off: the glm5_next three-tier path reads every expert through the cache; ask for --vram-slots 0 --pinned-slots 0 for an all-NVMe run", expert_cache::ENV))?;
        let layers: Vec<u32> = (g.dense_prefix..g.layers).map(|l| l as u32).collect();
        let nl = layers.len();
        let pu = pinned_use(std::env::var(PINNED_ENV).ok().as_deref(), std::env::var(CPU_LANE_ENV).ok().as_deref())?;
        let pinned_wc = sizes.pinned > 0 && cuda::pin_alloc_mode() == cuda::PinAlloc::Wc;
        lane_on_wc(pu, pinned_wc)?;
        let lane_threads = lane_threads(std::env::var(LANE_THREADS_ENV).ok().as_deref(), std::env::var(CPU_LANE_ENV).ok().as_deref())?;
        crate::glm5_moe::lane::THREADS.store(lane_threads, std::sync::atomic::Ordering::Relaxed);
        let cache = ExpertCache::new(policy, Scope::PerLayer, nl, g.experts, sizes.vram, sizes.pinned)?;
        let arena = match arena_kind(std::env::var(ARENA_ENV).ok().as_deref())? {
            ArenaKind::Layer => None,
            ArenaKind::Global => Some(ArenaDev::new(GlobalArena::new(nl, g.experts, nl * sizes.vram, nl * sizes.pinned)?, arena_config(&|k| std::env::var(k).ok())?)),
        };
        let records = ExpertRecord::glm5_table(cnq, &moe.record, &layers, g.experts as u32)?;
        let mut cfg = NvmeConfig::new(path);
        cfg.readers = readers;
        let src = NvmeSource::open(&cfg)?;
        let rb = moe.record.bytes;
        let mut vram = Vec::with_capacity(nl);
        let mut pinned = Vec::with_capacity(nl);
        let mut tables = Vec::with_capacity(nl);
        for _ in 0..nl {
            vram.push(if sizes.vram > 0 { cuda::alloc_named("glm5 VRAM expert slots", sizes.vram * rb as usize) } else { 0 });
            if sizes.pinned > 0 {
                pinned.push(Pinned::alloc_cold(sizes.pinned * rb as usize));
            }
            tables.push(cuda::alloc_named("glm5 expert record table", g.experts * 8));
        }
        let mut t = ExpertTiers {
            sizes,
            cache,
            first_moe: g.dense_prefix,
            rb,
            stage_cap,
            slots: (0..nl).map(|_| LayerSlots::new(g.experts, sizes)).collect(),
            arena,
            vram,
            pinned,
            tables,
            stage: cuda::alloc_named("glm5 expert staging slots", stage_cap * rb as usize),
            landing: Landing::new(stage_cap * rb as usize),
            records,
            src,
            nvme_reads: 0,
            nvme_bytes: 0,
            moves: vec![Moves::default(); nl],
            routing_syncs: 0,
            sub_batches: 0,
            pf_cap: 0,
            pf_stage: 0,
            pf_ring: None,
            arena_landing: Landing::new(0),
            pinned_use: PinnedUse::default(),
            pinned_wc,
            topk: g.topk,
            lane_clock: Default::default(),
            lane_geo: *moe,
            dev_lane: None,
            split: lane_split(std::env::var(CPU_LANE_ENV).ok().as_deref()).then(|| SplitCost::for_threads(lane_threads)),
            heat: vec![0; nl * g.experts],
            stager: None,
            prefetch: None,
        };
        t.set_pinned_use(pu)?;
        if t.arena.is_some() {
            t.arena_boot()?;
        }
        let env = |k: &str| std::env::var(k).ok();
        if stager_on(env(STAGER_ENV).as_deref(), env(glm5_flags::ENV_FLAGS).as_deref(), env(CPU_LANE_ENV).as_deref())? {
            t.set_stager(true)?;
        }
        let sw = Switches::from_env();
        sw.check()?;
        if sw.controller && t.stager.is_none() {
            return Err(format!("{}=1 needs {STAGER_ENV}=1: the controller serves the layers through the stager", glm5_flags::ENV_CONTROLLER));
        }
        if sw.prefetch {
            t.set_prefetch(true);
        }
        Ok(t)
    }

    /// #188: how selected pinned experts are read from now on ([`PinnedUse`]; `new` sets it
    /// from `CROW_GLM_PINNED` / `CROW_GLM_CPU_LANE`). The CPU lane is refused on a
    /// write-combined pinned arena. Changes no slot and no record.
    pub fn set_pinned_use(&mut self, u: PinnedUse) -> Result<(), String> {
        lane_on_wc(u, self.pinned_wc)?;
        self.cache.set_pinned_stays(u.stay);
        self.pinned_use = u;
        // SAFETY: construction and the callers hold a current context
        unsafe { self.sync_dev_lane() };
        Ok(())
    }

    /// the pinned bytes this store holds
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned.iter().map(|p| p.bytes as u64).sum::<u64>() + self.stager.as_ref().map_or(0, |st| st.pinned_bytes()) + self.prefetch.as_ref().map_or(0, |p| p.pinned_bytes())
    }

    /// the VRAM bytes this store holds (arenas, staging, tables)
    pub fn vram_bytes(&self) -> u64 {
        let nl = self.slots.len() as u64;
        nl * (self.sizes.vram as u64 * self.rb + self.cache.experts as u64 * 8) + (self.stage_cap + self.pf_cap) as u64 * self.rb + self.arena_extra_vram_bytes()
    }

    /// The record table of decoder layer `layer` for the selection `sel` (`[t][topk]` i32):
    /// the cache observes the call, the records move (see [`serve`]), the device table points
    /// each selected id at its record; every other entry is 0.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading this layer's slots or table is pending.
    pub unsafe fn table_for(&mut self, layer: usize, sel: &[i32]) -> Result<(Dev, Served), String> {
        let l = layer.checked_sub(self.first_moe).filter(|&l| l < self.slots.len()).ok_or_else(|| format!("expert tiers: layer {layer} is no MoE layer"))?;
        let ids = distinct_ids(sel, self.cache.experts)?;
        if self.arena.is_some() {
            return self.table_global(l, sel, &ids, None);
        }
        if self.stager.is_some() {
            return self.table_staged(l, sel, &ids, None);
        }
        let rb = self.rb;
        let mut m = GpuMover {
            vram: self.vram[l],
            pinned: self.pinned.get(l),
            stage: self.stage,
            landing: self.landing.p,
            rb,
            src: &self.src,
            recs: &self.records[l],
        };
        let mut served = match self.prefetch.as_mut() {
            Some(pf) => serve(&mut self.cache, l, &mut self.slots[l], &ids, self.stage_cap, &mut glm5_flags::PrefetchMover::new(&mut m, pf, &self.src, l, self.stage, None))?,
            None => serve(&mut self.cache, l, &mut self.slots[l], &ids, self.stage_cap, &mut m)?,
        };
        let mut table = vec![0u64; self.cache.experts];
        for &(e, loc) in &served.locs {
            table[e as usize] = match loc {
                Loc::Vram(v) => self.vram[l] + v as u64 * rb,
                Loc::Pinned(q) => self.pinned[l].dev + q as u64 * rb,
                Loc::Stage(s) => self.stage + s as u64 * rb,
            };
        }
        cuda::to_u64_into(self.tables[l], &table);
        // #188: a decode call's pinned ids go to the CPU lane (`GpuMoePlan::experts` takes the
        // post); their table entries stay valid pinned bases. Anything else clears the post.
        // `split`: only the ids `plan_split` gives the CPU (a call it gives none posts nothing)
        // CROW_GLM_MAX_BATCH: a batched step's rows too (one pool run per row)
        let host = self.pinned.get(l).map_or(std::ptr::null(), |p| p.host as *const u8);
        let post = self
            .lane_plan(l, sel, &mut served, &|e| table[e as usize], &|q| host.add(q as usize * rb as usize))
            .map(|(combos, _)| crate::glm5_moe::lane::Call { table: self.tables[l], combos, clock: self.lane_clock.clone(), ready: None });
        crate::glm5_moe::lane::post(post);
        self.count_heat(l, sel);
        if let Some(pf) = self.prefetch.as_mut() {
            let cache = &self.cache;
            glm5_flags::prefetch_hinted(pf, &self.src, &self.records, cache.experts, &|l, e| cache.tier(l, e) == Tier::Nvme, self.first_moe, layer)?;
        }
        self.nvme_reads += served.nvme_reads as u64;
        self.nvme_bytes += served.nvme_bytes;
        self.moves[l].add(&served.moves);
        self.routing_syncs += 1;
        self.sub_batches += 1;
        Ok((self.tables[l], served))
    }

    /// #186: allocate the prefill staging set: `slots` VRAM staging slots and as many pageable
    /// landing records ([`prefill_stage_slots`], the set the plan books), unless it is there.
    /// Prompt calls ([`ExpertTiers::tables_for_chunk`]) stage through it; decode calls keep the
    /// `stage_cap` set.
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn alloc_prefill_stage(&mut self, slots: usize) -> Result<(), String> {
        if self.pf_cap > 0 {
            return Ok(());
        }
        if slots < self.topk {
            return Err(format!("expert tiers: a prefill staging set of {slots} slots holds less than one row's top-{}", self.topk));
        }
        self.pf_stage = cuda::alloc_named("glm5 expert prefill staging slots", slots * self.rb as usize);
        self.pf_ring = Some(PfRing::new(self.rb));
        self.pf_cap = slots;
        Ok(())
    }

    /// #186: the slots of the prefill staging set (0 before the first prompt call)
    pub fn prefill_cap(&self) -> usize {
        self.pf_cap
    }

    /// #186: the experts of one prompt call of decoder layer `layer` (`sel` = `[t][topk]` i32):
    /// [`serve_chunk_prefill`] through the prefill staging set (allocated here on first use; no
    /// admission, the cache's tiers stay as decode left them), the device table rewritten per row
    /// sub-batch; `run(row0, rows, table)` queues that sub-batch's experts. One routing sync for
    /// the call. The staging copies go through [`PrefillMover`] (pinned landing ring, copy
    /// stream, two staging halves), so a sub-batch's NVMe reads and H2D run while the previous
    /// one computes.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading this layer's slots, its table or the prefill
    /// staging set is pending; `run` queues on the current stream.
    pub unsafe fn tables_for_chunk(&mut self, layer: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>) -> Result<(), String> {
        let l = layer.checked_sub(self.first_moe).filter(|&l| l < self.slots.len()).ok_or_else(|| format!("expert tiers: layer {layer} is no MoE layer"))?;
        if self.arena.is_some() {
            return self.tables_for_chunk_global(l, sel, run);
        }
        if self.pf_cap == 0 {
            self.alloc_prefill_stage(prefill_stage_slots(self.topk))?;
        }
        // #149 path B: the stager's reads of earlier decode calls (landed long ago: this call's
        // routing synchronized the stream past their experts) report their errors here
        self.settle()?;
        // a CPU-lane post belongs to a decode call: none for this call's tables
        crate::glm5_moe::lane::post(None);
        let (rb, k, experts) = (self.rb, self.topk, self.cache.experts);
        let (vram, pin_dev, stage, table_dev) = (self.vram[l], self.pinned.get(l).map_or(0, |p| p.dev), self.pf_stage, self.tables[l]);
        // two staging halves when each holds a row's top-k: sub-batch j fills half j % 2
        // while the kernels of sub-batch j - 1 read the other
        let halves = if self.pf_cap >= 2 * k { 2 } else { 1 };
        let half = self.pf_cap / halves;
        let pf = self.pf_ring.as_mut().expect("the prefill ring is allocated with the staging set");
        let free_ev = pf.free_ev;
        let mut m = PrefillMover { pinned: self.pinned.get(l), stage, half, halves, base: stage, rb, src: &self.src, recs: &self.records[l], pf, landed: Vec::new() };
        let (mut reads, mut bytes, mut moves, mut j) = (0u64, 0u64, Moves::default(), 0usize);
        let mut each = |r0: usize, rows: usize, served: &Served| -> Result<(), String> {
            let h = j % halves;
            let base = stage + (h * half) as u64 * rb;
            let mut table = vec![0u64; experts];
            for &(e, loc) in &served.locs {
                table[e as usize] = match loc {
                    Loc::Vram(v) => vram + v as u64 * rb,
                    Loc::Pinned(q) => pin_dev + q as u64 * rb,
                    Loc::Stage(s) => base + s as u64 * rb,
                };
            }
            cuda::to_u64_into(table_dev, &table);
            reads += served.nvme_reads as u64;
            bytes += served.nvme_bytes;
            moves.add(&served.moves);
            let r = run(r0, rows, table_dev);
            // the half is free again once these kernels are done
            cuda::event_record(free_ev[h], cuda::cur_stream());
            j += 1;
            r
        };
        let r = serve_chunk_prefill(&mut self.cache, l, &self.slots[l], sel, k, half, &mut m, &mut each);
        if r.is_ok() {
            self.count_heat(l, sel);
        }
        self.nvme_reads += reads;
        self.nvme_bytes += bytes;
        self.moves[l].add(&moves);
        self.routing_syncs += 1;
        self.sub_batches += *r.as_ref().unwrap_or(&0) as u64;
        r.map(|_| ())
    }

    /// the split's heat: every selection of `sel` (`[t][topk]` i32, ids checked by the caller)
    /// counts once for its expert of MoE layer `l`
    fn count_heat(&mut self, l: usize, sel: &[i32]) {
        let e = self.cache.experts;
        for &id in sel {
            let h = &mut self.heat[l * e + id as usize];
            *h = h.saturating_add(1);
        }
    }

    /// #190: the device record table of every MoE layer (the buffers `table_for` rewrites)
    pub fn tables(&self) -> &[Dev] {
        &self.tables
    }

    /// #188: the CPU lane's clock (wall time of its pool runs, experts, runs) since construction,
    /// shared: a report callback reads it while `generate` holds the tiers
    pub fn cpu_lane_clock(&self) -> std::sync::Arc<crate::glm5_moe::lane::Clock> {
        self.lane_clock.clone()
    }

    /// #187 (`glm5_run --cold`): empty the cache again, as after [`ExpertTiers::new`] (see
    /// [`reset_cache`]). The arenas stay allocated; no allocation, no GPU call. `nvme_reads`,
    /// `nvme_bytes` and `moves` keep counting; the cache's own counters restart at 0.
    pub fn reset_cache(&mut self) -> Result<(), String> {
        self.settle()?;
        if let Some(a) = self.arena.as_mut() {
            a.reset();
        }
        if let Some(pf) = self.prefetch.as_mut() {
            pf.forget(&self.src)?;
        }
        reset_cache(&mut self.cache, &mut self.slots, self.sizes)
    }

    /// # Safety
    /// No launch reading the store is pending.
    pub unsafe fn free(&mut self) {
        cuda::sync();
        self.free_arena();
        if let Some(mut st) = self.stager.take() {
            st.free();
        }
        if let Some(mut d) = self.dev_lane.take() {
            d.free();
        }
        if let Some(mut pf) = self.prefetch.take() {
            let _ = pf.free(&self.src);
        }
        for d in self.vram.iter_mut().chain(self.tables.iter_mut()) {
            cuda::free_dev(d);
        }
        for p in self.pinned.iter_mut() {
            p.free();
        }
        cuda::free_dev(&mut self.stage);
        if let Some(mut r) = self.pf_ring.take() {
            r.free();
        }
        cuda::free_dev(&mut self.pf_stage);
        self.pf_cap = 0;
    }
}

// ---------------------------------------------------------------- #149 path B: the stager

/// `CU_STREAM_WAIT_VALUE_GEQ`: the cyclic greater-or-equal (the flags are monotonic sequences)
const WAIT_GEQ: u32 = 0;

/// #149 path B, the landed half (`CROW_GLM_STAGER=1`): what [`ExpertTiers::table_for`] moves
/// records with instead of the synchronous [`GpuMover`].
///
/// - **One stager stream** (non-blocking): every copy of [`serve`]'s phases A-C and the record
///   table's upload are queued there, in `serve`'s order; nothing synchronizes it per call.
/// - **Persistent pinned sources**: the NVMe landing of the staging slots is pinned (the H2D is a
///   true async DMA, no pageable bounce), and each MoE layer's record table is written into its
///   own row of a pinned `[layers][experts]` u64 block and uploaded from there.
/// - **Landed flags**: one u64 per (MoE layer, expert) in mapped pinned memory. An NVMe read is
///   issued with [`NvmeSource::fetch_landed`]; the reader raises the record's flag to the call's
///   sequence number after the bytes are in. The stager stream waits on it
///   (`cuStreamWaitValue64_v2`, GEQ) before any copy behind it; the host does not wait.
/// - **The compute stream** waits for the stager's batch through an event (`cuEventRecord` on the
///   stager, `cuStreamWaitEvent` on the current stream) before the layer's experts. The driver
///   API says ordering through a stream memop "is not visible to CUDA" and that CUDA tasks it
///   orders should also have that order expressed with CUDA-visible dependencies such as events
///   (CUDA Driver API, Stream Memory Operations, `cuStreamWaitValue64`), so the memop waits stay on
///   the stager (host-raised flags only) and the stream-to-stream order is an event.
///
/// Why no host wait is needed per call (the argument the synchronous path's syncs made): the host
/// enters a call of MoE layer l only after the router flag of l (`CROW_GLM_FLAGS`), and the
/// compute stream is in order, so every earlier call's experts have run, and with them (they
/// waited on its event) every earlier stager batch, its landed flags and its reads. So the
/// staging slots, the landing buffer, the layer's VRAM / pinned slots, its device table and its
/// pinned table row are free to be rewritten. The one exception is inside a call: an NVMe read
/// straight into a pinned slot that this call's phase A still copies from (the slot of an expert
/// promoted to VRAM): the host synchronizes the stager stream first (`gate_syncs`). Under LRU that
/// needs an NVMe miss that loses VRAM to another pick of the same call, i.e. more picks than VRAM
/// slots (the synthetic V 3 / top-8 test: 15 of 50 calls); with the plan's 46 VRAM slots an older
/// victim always exists (derived, not measured).
struct Stager {
    stream: sys::CUstream,
    event: sys::CUevent,
    /// `stage_cap` records: the NVMe landing of the decode staging slots
    landing: Pinned,
    /// `[MoE layers][experts]` u64: the host side of every layer's record table
    tables: Pinned,
    /// `[MoE layers][experts]` u64: the landed flags (sequence of the call that read the record)
    landed: Pinned,
    /// the last call's sequence number (0 = none yet)
    seq: u64,
    /// the NVMe tickets of earlier calls, drained at the next call, [`ExpertTiers::settle`] or free
    pending: Vec<crate::nvme_source::Ticket>,
    stats: StagerStats,
}

/// #149 path B: host-side counts of the stager since it was turned on
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StagerStats {
    /// decode calls served through the stager
    pub calls: u64,
    /// NVMe records read with a landed flag (the stager stream waits on each)
    pub landed_reads: u64,
    /// host syncs of the stager stream before an NVMe read into a pinned slot phase A still reads
    pub gate_syncs: u64,
    /// #202: calls answered (the reply or the compute stream's event queued) and the NVMe
    /// records in flight at each answer, summed (`inflight_at_answer / answers` = records in
    /// flight per layer)
    pub answers: u64,
    pub inflight_at_answer: u64,
}

impl Stager {
    /// # Safety
    /// A CUDA context is current.
    unsafe fn new(layers: usize, experts: usize, stage_cap: usize, rb: u64) -> Result<Stager, String> {
        let mut dev: sys::CUdevice = 0;
        cuda::ck(sys::cuCtxGetDevice(&mut dev));
        let mut ok = 0i32;
        cuda::ck(sys::cuDeviceGetAttribute(&mut ok, sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS, dev));
        if ok == 0 {
            return Err(format!(
                "{STAGER_ENV}=1: this device has no 64-bit stream memory operations (CU_DEVICE_ATTRIBUTE_CAN_USE_64_BIT_STREAM_MEM_OPS = 0); the landed flags need cuStreamWaitValue64"
            ));
        }
        let words = (layers * experts * 8).next_multiple_of(4096);
        let st = Stager {
            stream: cuda::stream_create_non_blocking(),
            event: cuda::event_create(),
            landing: Pinned::alloc(stage_cap * rb as usize),
            tables: Pinned::alloc(words),
            landed: Pinned::alloc(words),
            seq: 0,
            pending: Vec::new(),
            stats: StagerStats::default(),
        };
        std::ptr::write_bytes(st.tables.host as *mut u8, 0, st.tables.bytes);
        std::ptr::write_bytes(st.landed.host as *mut u8, 0, st.landed.bytes);
        assert_eq!(st.pinned_bytes(), stager_pinned_bytes(layers, experts, stage_cap, rb), "the stager allocates what the plan books");
        Ok(st)
    }

    fn pinned_bytes(&self) -> u64 {
        (self.landing.bytes + self.tables.bytes + self.landed.bytes) as u64
    }

    /// every pending read's report; the first error by name
    fn settle(&mut self, src: &NvmeSource) -> Result<(), String> {
        let mut err = None;
        for t in self.pending.drain(..) {
            if let Err(e) = src.wait(t) {
                err.get_or_insert(format!("{STAGER_ENV}: an NVMe read of an earlier call: {e}"));
            }
        }
        err.map_or(Ok(()), Err)
    }

    /// # Safety
    /// A CUDA context is current; nothing on the compute stream waits on this stager any more.
    unsafe fn free(&mut self) {
        cuda::stream_sync(self.stream);
        // the reads are done (the stager waited on their flags); dropping a ticket drains it
        self.pending.clear();
        cuda::event_destroy(self.event);
        cuda::stream_destroy(self.stream);
        self.landing.free();
        self.tables.free();
        self.landed.free();
    }
}

/// [`serve`]'s mover on the stager stream (see [`Stager`])
struct StagerMover<'a> {
    s: sys::CUstream,
    vram: Dev,
    pinned: Option<&'a Pinned>,
    stage: Dev,
    landing: *mut u8,
    rb: u64,
    src: &'a NvmeSource,
    recs: &'a [ExpertRecord],
    /// this layer's row of the landed flags: host and device address
    landed_host: *mut u64,
    landed_dev: Dev,
    seq: u64,
    /// pinned slots this call's queued copies read (phase A), not yet known to be done
    read_pinned: Vec<u32>,
    pending: &'a mut Vec<crate::nvme_source::Ticket>,
    stats: &'a mut StagerStats,
}

impl Mover for StagerMover<'_> {
    fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
        // a reader writes a pinned slot from the host: not while a queued copy still reads it
        if jobs.iter().any(|(_, d)| matches!(d, Dst::Pinned(q) if self.read_pinned.contains(q))) {
            unsafe { cuda::stream_sync(self.s) };
            self.read_pinned.clear();
            self.stats.gate_syncs += 1;
        }
        let rb = self.rb as usize;
        let mut v = Vec::with_capacity(jobs.len());
        let mut landed = Vec::with_capacity(jobs.len());
        let mut bytes = 0u64;
        for &(e, d) in jobs {
            let gu = match d {
                Dst::Landing(i) => unsafe { self.landing.add(i as usize * rb) },
                Dst::Pinned(q) => unsafe { (self.pinned.expect("a pinned destination without a pinned arena").host as *mut u8).add(q as usize * rb) },
            };
            let dst = RecordDst { gu, dn: std::ptr::null_mut() };
            let rec = self.recs[e as usize];
            bytes += rec.parts(&dst).iter().map(|p| p.1.len as u64).sum::<u64>();
            v.push((rec, dst));
            landed.push(crate::nvme_source::Landed { flag: unsafe { self.landed_host.add(e as usize) }, value: self.seq });
        }
        // SAFETY: the destinations are the stager's pinned landing and this layer's pinned arena,
        // `rb` bytes each, read only by stager copies queued behind the flags below; the flags
        // are this layer's row, each written by one read per call
        let t = unsafe { self.src.fetch_landed(&v, &landed) }?;
        self.pending.push(t);
        for &(e, _) in jobs {
            unsafe { cuda::ck(sys::cuStreamWaitValue64_v2(self.s, self.landed_dev + e as u64 * 8, self.seq, WAIT_GEQ)) };
        }
        self.stats.landed_reads += jobs.len() as u64;
        Ok(bytes)
    }
    fn landing_to_stage(&mut self, i: u32) {
        let rb = self.rb as usize;
        unsafe { cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.stage + i as u64 * self.rb, self.landing.add(i as usize * rb) as *const _, rb, self.s)) };
    }
    fn pinned_to_stage(&mut self, q: u32, s: u32) {
        let p = self.pinned.expect("a pinned source without a pinned arena");
        unsafe { cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.stage + s as u64 * self.rb, (p.host as *const u8).add(q as usize * self.rb as usize) as *const _, self.rb as usize, self.s)) };
        self.read_pinned.push(q);
    }
    fn vram_to_stage(&mut self, v: u32, s: u32) {
        unsafe { cuda::ck(sys::cuMemcpyDtoDAsync_v2(self.stage + s as u64 * self.rb, self.vram + v as u64 * self.rb, self.rb as usize, self.s)) };
    }
    /// the stager stream orders phase B's copies behind phase A's; a host write into a pinned
    /// slot waits in [`StagerMover::nvme`] (the gate)
    fn barrier(&mut self) {}
    fn vram_to_pinned(&mut self, v: u32, q: u32) {
        let p = self.pinned.expect("a pinned destination without a pinned arena");
        unsafe { cuda::ck(sys::cuMemcpyDtoHAsync_v2((p.host as *mut u8).add(q as usize * self.rb as usize) as *mut _, self.vram + v as u64 * self.rb, self.rb as usize, self.s)) };
    }
    fn stage_to_vram(&mut self, s: u32, v: u32) {
        unsafe { cuda::ck(sys::cuMemcpyDtoDAsync_v2(self.vram + v as u64 * self.rb, self.stage + s as u64 * self.rb, self.rb as usize, self.s)) };
    }
}

impl ExpertTiers {
    /// #149 path B: the stager on or off (`new` takes it from `CROW_GLM_STAGER`, [`stager_on`]).
    /// On allocates the stager stream, its event and its pinned blocks (landing `stage_cap`
    /// records, the table rows and the landed flags, `[MoE layers][experts]` u64 each); off frees
    /// them. Refused by name on a device without 64-bit stream memops and with the CPU lane.
    /// The cache and the slots stay as they are.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading this store is pending.
    pub unsafe fn set_stager(&mut self, on: bool) -> Result<(), String> {
        match (on, self.stager.is_some()) {
            (true, false) => {
                self.stager = Some(Stager::new(self.slots.len(), self.cache.experts, self.stage_cap, self.rb)?);
            }
            (false, true) => {
                cuda::sync();
                let r = self.settle();
                if let Some(mut st) = self.stager.take() {
                    st.free();
                }
                r?;
            }
            _ => {}
        }
        self.sync_dev_lane();
        Ok(())
    }

    /// the CPU lane under the controller ([`glm5_flags::DevLane`]) exists exactly while the lane
    /// and the stager are on (the controller serves through the stager)
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading the lane is pending.
    unsafe fn sync_dev_lane(&mut self) {
        let want = self.pinned_use.cpu_lane && self.stager.is_some();
        match (want, self.dev_lane.is_some()) {
            (true, false) => self.dev_lane = Some(glm5_flags::DevLane::new(self.topk, self.lane_geo.hidden, self.rb)),
            (false, true) => {
                cuda::sync();
                if let Some(mut d) = self.dev_lane.take() {
                    d.free();
                }
            }
            _ => {}
        }
    }

    /// `CROW_GLM_CONTROLLER` with the CPU lane: the device addresses of a controlled row (`None`
    /// without the lane)
    pub fn dev_lane(&self) -> Option<glm5_flags::DevLaneDev> {
        self.dev_lane.as_ref().map(|d| d.dev())
    }

    /// `CROW_GLM_CONTROLLER` with the CPU lane: a fresh controller counts from 0 again
    pub fn dev_lane_reset(&self) {
        if let Some(d) = self.dev_lane.as_ref() {
            d.reset();
        }
    }

    /// after a failed controller job: every lane wait of the device passes
    pub fn dev_lane_release(&self) {
        if let Some(d) = self.dev_lane.as_ref() {
            d.release();
        }
    }

    /// The CPU lane of one decode call of MoE layer `l` (`sel` = `[rows][topk]`, the served
    /// locations, `table` the record bases by expert, `host(q)` the host address of pinned slot
    /// `q`): which pinned ids the CPU computes (every one; `split`: those `plan_split` gives it,
    /// from the layer's heat) and the combos. `None` with the lane off, more rows than a batched
    /// step holds, or no pinned pick; the counters move the CPU's ids from zero-copy to the lane.
    fn lane_plan(&self, l: usize, sel: &[i32], served: &mut Served, table: &dyn Fn(u32) -> u64, host: &dyn Fn(u32) -> *const u8) -> Option<(Vec<crate::glm5_moe::lane::Combo>, Vec<u32>)> {
        let k = self.topk;
        if !self.pinned_use.cpu_lane || sel.is_empty() || sel.len() % k != 0 || sel.len() / k > MAX_BATCH || served.moves.zero_copy == 0 {
            return None;
        }
        let e = self.cache.experts;
        let heat = &self.heat[l * e..(l + 1) * e];
        let pinned: Vec<u32> = served.locs.iter().filter(|x| matches!(x.1, Loc::Pinned(_))).map(|x| x.0).collect();
        let cpu = match self.split {
            Some(cost) => {
                let hits = served.locs.len() - pinned.len();
                let ram: Vec<(u32, u32)> = pinned.iter().map(|&x| (x, heat[x as usize])).collect();
                plan_split(&cost, hits, &ram)
            }
            None => pinned,
        };
        let (combos, n) = lane_combos_where(sel, &served.locs, |e, _| table(e), |q| host(q), |e| cpu.contains(&e));
        let nd = cpu.len() as u64;
        served.moves.cpu_lane = nd;
        served.moves.zero_copy -= nd;
        (n > 0).then_some((combos, cpu))
    }

    /// #149 path B: the stager is on
    pub fn stager_on(&self) -> bool {
        self.stager.is_some()
    }

    /// #149 path B: the stager's counters (`None` when it is off)
    pub fn stager_stats(&self) -> Option<StagerStats> {
        self.stager.as_ref().map(|st| st.stats)
    }

    /// #202: the NVMe source's records asked for since construction and those in flight now
    pub fn nvme_io(&self) -> (u64, u64) {
        (self.src.records_submitted(), self.src.records_in_flight())
    }

    /// `CROW_GLM_PREFETCH`: the store of guessed records on or off (`new` takes it from the
    /// environment). On allocates 2 x top-k pinned records (not booked by the plan); off waits for
    /// its reads and frees it. The cache and the slots stay as they are.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading this store is pending.
    pub unsafe fn set_prefetch(&mut self, on: bool) {
        match (on, self.prefetch.is_some()) {
            (true, false) => self.prefetch = Some(glm5_flags::Prefetch::new(self.topk, self.rb)),
            (false, true) => {
                cuda::sync();
                if let Some(mut pf) = self.prefetch.take() {
                    let _ = pf.free(&self.src);
                }
            }
            _ => {}
        }
    }

    /// `CROW_GLM_PREFETCH`: the store's counters (`None` when it is off)
    pub fn prefetch_stats(&self) -> Option<glm5_flags::PrefetchStats> {
        self.prefetch.as_ref().map(|p| p.stats)
    }

    /// `CROW_GLM_PREFETCH`: the store's counters as of its last call, shared, so a report
    /// callback reads them while a run holds the store (`None` when it is off)
    pub fn prefetch_clock(&self) -> Option<std::sync::Arc<std::sync::Mutex<glm5_flags::PrefetchStats>>> {
        self.prefetch.as_ref().map(|p| p.clock())
    }

    /// `CROW_GLM_CONTROLLER`: [`ExpertTiers::table_for`] of decoder layer `layer` through the
    /// stager, the record table in place, and `q` written to the reply word `reply` behind the
    /// layer's moves on the stager stream (instead of the compute stream's event wait).
    ///
    /// # Safety
    /// As [`ExpertTiers::table_for`]; the device published this call's request (so every earlier
    /// call's experts ran); `reply` is a mapped u64 the device waits on.
    pub unsafe fn table_reply(&mut self, layer: usize, sel: &[i32], reply: Dev, q: u64) -> Result<(), String> {
        let l = layer.checked_sub(self.first_moe).filter(|&l| l < self.slots.len()).ok_or_else(|| format!("expert tiers: layer {layer} is no MoE layer"))?;
        if self.stager.is_none() {
            return Err(format!("{}=1 needs {STAGER_ENV}=1", glm5_flags::ENV_CONTROLLER));
        }
        let ids = distinct_ids(sel, self.cache.experts)?;
        if self.arena.is_some() {
            return self.table_global(l, sel, &ids, Some((reply, q))).map(|_| ());
        }
        self.table_staged(l, sel, &ids, Some((reply, q))).map(|_| ())
    }

    /// `CROW_GLM_CONTROLLER`: wait for the stager stream (after a failure in a controller job)
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn stager_idle(&self) {
        if let Some(st) = self.stager.as_ref() {
            cuda::stream_sync(st.stream);
        }
    }

    /// #149 path B: the outcome of every NVMe read the stager issued and has not reported yet
    /// (each raised its flag, also on a failure). Blocks only while such a read still runs, which
    /// after a stream sync past the last call's experts is none. A no-op with the stager off.
    pub fn settle(&mut self) -> Result<(), String> {
        match self.stager.as_mut() {
            Some(st) => st.settle(&self.src),
            None => Ok(()),
        }
    }

    /// The tail of a stager call (per-layer and global): the record table (its pinned row `trow`)
    /// up on the stager stream behind the moves, then either the compute stream's event wait and
    /// the CPU lane's post (`ready` = the stager's event), or, under the controller, the reply
    /// word behind the batch and the CPU lane served here on the controller thread
    /// ([`glm5_flags::DevLane::serve`]: the CPU's table entries point at its zeroed record, its
    /// rows are computed once the stager's moves are done, the lane flag goes to the request's
    /// number, also with no CPU combo).
    ///
    /// # Safety
    /// As [`ExpertTiers::table_staged`]; `trow` is layer `l`'s pinned table row.
    #[allow(clippy::too_many_arguments)]
    unsafe fn finish_staged(
        &mut self,
        l: usize,
        lane: Option<(Vec<crate::glm5_moe::lane::Combo>, Vec<u32>)>,
        _served: usize,
        trow: *mut u64,
        stream: sys::CUstream,
        event: sys::CUevent,
        reply: Option<(Dev, u64)>,
    ) -> Result<(), String> {
        let experts = self.cache.experts;
        let dev_lane = reply.is_some() && self.dev_lane.is_some();
        if dev_lane {
            if let Some((_, cpu)) = lane.as_ref() {
                let dummy = self.dev_lane.as_ref().expect("dev lane").dummy;
                for &e in cpu {
                    *trow.add(e as usize) = dummy;
                }
            }
        }
        cuda::ck(sys::cuMemcpyHtoDAsync_v2(self.tables[l], trow as *const _, experts * 8, stream));
        if let Some(st) = self.stager.as_mut() {
            st.stats.answers += 1;
            st.stats.inflight_at_answer += self.src.records_in_flight();
        }
        match reply {
            None => {
                cuda::event_record(event, stream);
                // WDDM: submit the stager's batch now (it would otherwise wait for a later query or sync)
                cuda::stream_query(stream);
                cuda::stream_wait_event(cuda::cur_stream(), event);
                let post = lane.map(|(combos, _)| crate::glm5_moe::lane::Call { table: self.tables[l], combos, clock: self.lane_clock.clone(), ready: Some(event as u64) });
                crate::glm5_moe::lane::post(post);
            }
            // CROW_GLM_CONTROLLER: the reply word behind the batch; the compute stream's device
            // wait for it was queued long before
            Some((w, q)) => {
                cuda::ck(sys::cuStreamWriteValue64_v2(stream, w, q, 0));
                cuda::stream_query(stream);
                crate::glm5_moe::lane::post(None);
                if dev_lane {
                    let cpu: Vec<(usize, *const u8)> = match lane.as_ref() {
                        Some((combos, _)) => {
                            // the CPU reads its records once the moves that put them there ran
                            cuda::event_record(event, stream);
                            cuda::ck(sys::cuEventSynchronize(event));
                            combos.iter().enumerate().filter_map(|(c, x)| if let crate::glm5_moe::lane::Combo::Cpu(r) = *x { Some((c, r)) } else { None }).collect()
                        }
                        None => Vec::new(),
                    };
                    self.dev_lane.as_ref().expect("dev lane").serve(q, &cpu, &self.lane_geo, &self.lane_clock);
                }
            }
        }
        Ok(())
    }

    /// [`ExpertTiers::table_for`] through the stager: the earlier calls' reads report, then
    /// [`serve`] with a [`StagerMover`], the table into its pinned row and up on the stager, the
    /// stager's event recorded and waited on by the current stream. The host waits for nothing
    /// here except the gate of [`StagerMover::nvme`].
    ///
    /// # Safety
    /// As [`ExpertTiers::table_for`]; the host has seen this layer's router finish (its flag).
    unsafe fn table_staged(&mut self, l: usize, sel: &[i32], ids: &[u32], reply: Option<(Dev, u64)>) -> Result<(Dev, Served), String> {
        let rb = self.rb;
        let experts = self.cache.experts;
        let st = self.stager.as_mut().expect("table_staged without the stager");
        st.settle(&self.src)?;
        st.seq += 1;
        st.stats.calls += 1;
        let row = l * experts;
        let mut m = StagerMover {
            s: st.stream,
            vram: self.vram[l],
            pinned: self.pinned.get(l),
            stage: self.stage,
            landing: st.landing.host as *mut u8,
            rb,
            src: &self.src,
            recs: &self.records[l],
            landed_host: (st.landed.host as *mut u64).add(row),
            landed_dev: st.landed.dev + (row * 8) as u64,
            seq: st.seq,
            read_pinned: Vec::new(),
            pending: &mut st.pending,
            stats: &mut st.stats,
        };
        let mut served = match self.prefetch.as_mut() {
            Some(pf) => serve(&mut self.cache, l, &mut self.slots[l], ids, self.stage_cap, &mut glm5_flags::PrefetchMover::new(&mut m, pf, &self.src, l, self.stage, Some(st.stream)))?,
            None => serve(&mut self.cache, l, &mut self.slots[l], ids, self.stage_cap, &mut m)?,
        };
        let (stream, event, trow) = (st.stream, st.event, (st.tables.host as *mut u64).add(row));
        // the table into this layer's pinned row (its last upload ran before this layer's
        // previous experts), then up on the stager behind the moves
        let host = std::slice::from_raw_parts_mut(trow, experts);
        host.fill(0);
        for &(e, loc) in &served.locs {
            host[e as usize] = match loc {
                Loc::Vram(v) => self.vram[l] + v as u64 * rb,
                Loc::Pinned(q) => self.pinned[l].dev + q as u64 * rb,
                Loc::Stage(s) => self.stage + s as u64 * rb,
            };
        }
        // the CPU lane: its pinned records are read after this call's moves (the stager's event)
        let pin_host = self.pinned.get(l).map_or(std::ptr::null(), |p| p.host as *const u8);
        let lane = {
            let tv: Vec<u64> = host.to_vec();
            self.lane_plan(l, sel, &mut served, &|e| tv[e as usize], &|q| pin_host.add(q as usize * rb as usize))
        };
        self.finish_staged(l, lane, served.locs.len(), trow, stream, event, reply)?;
        self.count_heat(l, sel);
        if let Some(pf) = self.prefetch.as_mut() {
            let cache = &self.cache;
            glm5_flags::prefetch_hinted(pf, &self.src, &self.records, cache.experts, &|l, e| cache.tier(l, e) == Tier::Nvme, self.first_moe, self.first_moe + l)?;
        }
        self.nvme_reads += served.nvme_reads as u64;
        self.nvme_bytes += served.nvme_bytes;
        self.moves[l].add(&served.moves);
        self.routing_syncs += 1;
        self.sub_batches += 1;
        Ok((self.tables[l], served))
    }
}

// ---------------------------------------------------------------- the container and the run

/// a glm5_next container opened and checked the way `decode glmgolden` checks it
pub struct Opened {
    pub cnq: Cnq,
    pub path: String,
    pub g: Glm5Geo,
    pub moe: MoeGeo,
    pub spec: ExpertRecordSpec,
    pub records: usize,
    pub constants: usize,
}

/// Open `path` and refuse it by name unless it is an index v2 whose config passes the glm5_next
/// family row, with MUL1 expert records (the checks of `decode glmgolden`).
pub fn open_container(path: &str) -> Result<Opened, String> {
    use crate::meta::{v2_config_label, ModelMeta};
    let cnq = Cnq::open_checked(path).map_err(|e| format!("{path}: {e}"))?;
    let config = cnq.config_json().map(str::to_string).ok_or_else(|| format!("{path}: no index v2 config"))?;
    let generation = cnq.generation_config_json().map(str::to_string);
    let gen_label = v2_config_label(path, "generation_config_json");
    let meta = ModelMeta::from_config_texts(&config, &v2_config_label(path, "config_json"), generation.as_deref().map(|t| (t, gen_label.as_str())))?;
    let bad = meta.verify();
    if !bad.is_empty() {
        return Err(format!("{} constants differ from the glm5_next family row: {}", bad.len(), bad.iter().map(|c| c.line()).collect::<Vec<_>>().join("; ")));
    }
    let g = meta.glm5_geo()?;
    let (spec, records) = crate::nvme_source::glm5_record_of_container(path)?;
    let moe = MoeGeo::new(&g, spec)?;
    Ok(Opened { cnq, path: path.to_string(), g, moe, spec, records, constants: meta.checks().len() })
}

/// one row of a run, as the report callback sees it
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TokenReport {
    pub pos: usize,
    /// the row was a prompt row
    pub prompt: bool,
    /// the greedy id the head chose after this row (`None` for prompt rows before the last)
    pub next: Option<i64>,
    pub secs: f64,
    pub nvme_reads: u64,
    pub nvme_bytes: u64,
    /// `[vram, pinned, nvme]` accesses of this row per MoE layer
    pub tiers: Vec<[u64; 3]>,
    /// #187: the [`Moves`] of this row per MoE layer (host counters)
    pub moves: Vec<Moves>,
    /// #186: the prompt rows this report covers (a prompt call of `CROW_CHUNK` rows reports once,
    /// at its last row `pos`); 1 for a decode row
    pub rows: usize,
    /// #186: MoE calls of this report (one routing sync each) and their [`serve`] sub-batches
    pub routing_syncs: u64,
    pub sub_batches: u64,
}

/// what a run generated
#[derive(Clone, Debug, Default, PartialEq)]
pub struct Generated {
    pub ids: Vec<i64>,
    /// the head's logits before each generated id (when asked to keep them)
    pub logits: Vec<Vec<f32>>,
}

/// The glm5_next model with every layer's dense part in VRAM, run row by row.
pub struct Glm5Run {
    pub g: Glm5Geo,
    pub moe: MoeGeo,
    pub cap: usize,
    /// #186: prompt rows per prompt call (1 = every prompt row one decode call); at most the
    /// pass's `max_t` (`CROW_CHUNK` at `load`, [`Glm5Run::set_prompt_chunk`])
    prompt_chunk: usize,
    /// `CROW_GLM_ARENA=global` with `CROW_GLM_ARENA_ELASTIC_GB` and a prompt chunk above the
    /// decode calls: the pass and the residual hold `decode_t` rows; a prompt phase borrows
    /// `prompt_t` rows of scratch from the elastic part and gives it back after the prompt
    /// ([`Glm5Run::prefill_with`]); `None` = the scratch of `max_t` rows stays allocated
    borrow: Option<(usize, usize)>,
    pub load: LoadReport,
    pass: Glm5Pass,
    layers: Vec<LayerW>,
    kda: Vec<Option<KdaState>>,
    mla: Vec<Option<MlaCache>>,
    head: Head,
    hw: HeadW,
    x: Dev,
    normed: Dev,
    logits: Dev,
    next: Dev,
    /// #149 path B / #189: the decode switches and what they hold (all `None` when off)
    sw: Switches,
    sw_kernels: Option<glm5_flags::Kernels>,
    feed: Option<Feed>,
    readback: Option<Readback>,
    /// #190 (`CROW_GLM_GRAPH=1`): the row's and the head's captured graphs (`None` when off)
    graph: Option<glm5_graph::RowGraphs>,
    /// #192 (`CROW_GLM_MTP`): the speculative decode's block and state (`None` when off)
    spec: Option<Box<crate::glm5_mtp::Spec>>,
    /// `CROW_GLM_CONTROLLER`: the controller thread (`None` when off)
    worker: Option<glm5_flags::Worker<TokenReport>>,
    /// `CROW_GLM_LA`: the row `decode_la` enqueued ahead, and the KDA states of its start
    ahead: Option<Ahead>,
    kda_bak: Vec<(Dev, Dev)>,
    /// `CROW_GLM_MAX_BATCH`: the sequence slots ([`Glm5Run::set_slots`]); empty = the one
    /// sequence of `kda` / `mla` / `logits`. Slot `cur`'s state is in those fields and
    /// `seqs[cur]` is an empty placeholder; every other slot's state is parked in `seqs`.
    seqs: Vec<SeqSlot>,
    cur: usize,
    /// `CROW_GLM_MAX_BATCH`: the head buffers of a batched decode step (`None` with one slot)
    bat: Option<BatchHead>,
}

/// `CROW_GLM_MAX_BATCH`: one parked sequence: a KDA state per KDA layer, an MLA cache per DSA
/// layer, the logits row of its last head
#[derive(Default)]
struct SeqSlot {
    kda: Vec<Option<KdaState>>,
    mla: Vec<Option<MlaCache>>,
    logits: Dev,
}

/// `CROW_GLM_MAX_BATCH`: `normed [B][hidden]`, `logits [B][vocab]`, `ids [B]` i32
struct BatchHead {
    normed: Dev,
    logits: Dev,
    ids: Dev,
}

/// #189: a row whose greedy id the host has not read yet (`CROW_GLM_LOOKAHEAD`)
struct Pending {
    pos: usize,
    prompt: bool,
    t0: std::time::Instant,
    base: RowBase,
}

/// the store's counters at a row's start; a row's report is the difference at its end
struct RowBase {
    counters: Vec<[u64; 3]>,
    moves: Vec<Moves>,
    nvme_reads: u64,
    nvme_bytes: u64,
    routing_syncs: u64,
    sub_batches: u64,
}

impl RowBase {
    fn of(t: &ExpertTiers) -> RowBase {
        RowBase {
            counters: t.tier_counters().to_vec(),
            moves: t.moves.clone(),
            nvme_reads: t.nvme_reads,
            nvme_bytes: t.nvme_bytes,
            routing_syncs: t.routing_syncs,
            sub_batches: t.sub_batches,
        }
    }

    fn report(&self, t: &ExpertTiers, pos: usize, prompt: bool, next: Option<i64>, t0: std::time::Instant) -> TokenReport {
        TokenReport {
            pos,
            prompt,
            next,
            secs: t0.elapsed().as_secs_f64(),
            nvme_reads: t.nvme_reads - self.nvme_reads,
            nvme_bytes: t.nvme_bytes - self.nvme_bytes,
            tiers: t.tier_counters().iter().zip(&self.counters).map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]]).collect(),
            moves: t.moves.iter().zip(&self.moves).map(|(a, b)| a.since(b)).collect(),
            rows: 1,
            routing_syncs: t.routing_syncs - self.routing_syncs,
            sub_batches: t.sub_batches - self.sub_batches,
        }
    }
}

impl Glm5Run {
    /// Load the dense part of every layer and the head (`cnq`), one KDA state per KDA layer, one
    /// MLA cache of `cap` tokens per DSA layer.
    ///
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn load(cnq: &mut Cnq, g: &Glm5Geo, moe: &MoeGeo, cap: usize, log: &mut dyn FnMut(&str)) -> Glm5Run {
        let t0 = std::time::Instant::now();
        let mut load = LoadReport::default();
        let mut layers = Vec::with_capacity(g.layers);
        for l in 0..g.layers {
            layers.push(gm::load_layer_without_experts(cnq, g, moe, l, &mut load));
        }
        let hw = gm::load_head(cnq, g, &mut load);
        log(&format!(
            "[glm5_run] {} layers + head loaded in {:.1} s without routed experts: {} B to VRAM, {} NVFP4 scale bytes 0x7F -> 0x7E, kv_b to BF16 {} of {} values inexact",
            g.layers,
            t0.elapsed().as_secs_f64(),
            load.bytes,
            load.sanitized,
            load.kv_b_inexact,
            load.kv_b_values
        ));
        let (kd, md) = (KdaDims::of(g), MlaDims::of(g));
        let kda = (0..g.layers).map(|l| (gm::attn_kind(g, l) == AttnKind::Kda).then(|| KdaState::alloc(&kd))).collect();
        let mla = (0..g.layers).map(|l| (gm::attn_kind(g, l) == AttnKind::Mla).then(|| MlaCache::new(&md, cap))).collect();
        let row = g.hc_streams * g.hidden;
        // #186: the pass and the residual hold a prompt call of CROW_CHUNK rows (1 = row by row)
        let chunk = prompt_chunk_from_env().clamp(1, cap.max(1));
        // #192: the pass and the residual also hold a verify call of 1 + CROW_GLM_MTP rows
        let max_t = chunk.max((1 + crate::glm5_mtp::draft_rows_from_env().unwrap_or(0)).min(cap.max(1)));
        // CROW_GLM_MAX_BATCH: and a batched decode step of one row per sequence slot
        let max_t = max_t.max(max_batch_from_env().unwrap_or(1).min(cap.max(1)));
        // CROW_GLM_ARENA elastic: the prompt scratch is borrowed per prompt phase
        let decode_t = (1 + crate::glm5_mtp::draft_rows_from_env().unwrap_or(0)).max(max_batch_from_env().unwrap_or(1)).min(cap.max(1));
        let borrow = prompt_borrow_from_env(&|k| std::env::var(k).ok(), chunk, decode_t).then_some((decode_t, max_t));
        let held_t = borrow.map_or(max_t, |b| b.0);
        if let Some((d, p)) = borrow {
            log(&format!("[glm5_run] prompt scratch of {p} rows borrowed from the elastic arena per prompt phase; {d} rows held"));
        }
        let max_t = held_t;
        let mut run = Glm5Run {
            g: *g,
            moe: *moe,
            cap,
            prompt_chunk: chunk,
            borrow,
            load,
            pass: Glm5Pass::new(g, *moe, max_t, cap),
            layers,
            kda,
            mla,
            head: Head::new(gm::head_geo(g)),
            hw,
            x: cuda::alloc_named("glm5_run residual", max_t * row * 4),
            normed: cuda::alloc_named("glm5_run normed", g.hidden * 4),
            logits: cuda::alloc_named("glm5_run logits", g.vocab * 4),
            next: cuda::alloc_named("glm5_run greedy id", 4),
            sw: Switches::default(),
            sw_kernels: None,
            feed: None,
            readback: None,
            graph: None,
            spec: None,
            worker: None,
            ahead: None,
            kda_bak: Vec::new(),
            seqs: Vec::new(),
            cur: 0,
            bat: None,
        };
        let sw = Switches::from_env();
        if sw != Switches::default() {
            run.set_switches(cnq, sw);
            log(&format!("[glm5_run] decode switches: {}{}", sw.label(), run.feed.as_ref().map_or(String::new(), |f| format!("; lookahead embedding table {} B in VRAM (generate only)", f.bytes))));
        }
        if glm5_graph::on_from_env() {
            run.set_graph(true);
            log(&format!("[glm5_run] {} on: decode rows as piecewise CUDA graphs, one segment per MoE router + 1, the head one graph", glm5_graph::ENV));
        }
        run
    }

    /// #190: `CROW_GLM_GRAPH` on or off (`load` takes it from the environment); off frees the
    /// graphs and the capture stream.
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this run is pending.
    pub unsafe fn set_graph(&mut self, on: bool) {
        match (on, self.graph.is_some()) {
            (true, false) => self.graph = Some(glm5_graph::RowGraphs::new()),
            (false, true) => {
                if let Some(mut gr) = self.graph.take() {
                    gr.free();
                }
            }
            _ => {}
        }
    }

    /// #190: the graphs and their counters, when `CROW_GLM_GRAPH` is on
    pub fn graphs(&self) -> Option<&glm5_graph::RowGraphs> {
        self.graph.as_ref()
    }

    /// `CROW_GLM_PREFETCH`: the guesses' counters (`None` without the guess)
    pub fn guess_stats(&self) -> Option<glm5_flags::GuessStats> {
        let mut g = self.pass.routed.as_ref().filter(|r| r.predicting()).map(|r| r.guess)?;
        if let Some(c) = self.pass.ctl.as_ref() {
            if let Ok(s) = c.score.lock() {
                (g.compared, g.picks, g.hits) = (g.compared + s.stats.compared, g.picks + s.stats.picks, g.hits + s.stats.hits);
            }
        }
        Some(g)
    }

    /// routings read through the router's flag (`CROW_GLM_FLAGS`; 0 without the publisher)
    pub fn flag_calls(&self) -> u64 {
        self.pass.routed.as_ref().map_or(0, |r| r.calls)
    }

    /// the decode switches in force
    pub fn switches(&self) -> Switches {
        self.sw
    }

    /// #149 path B / #189: put the decode switches in force (`load` takes them from the
    /// environment): `flags` gives the pass its router-ids publisher, `lookahead` loads the
    /// embedding table into VRAM and the pinned readback. Turning a switch off frees what it held.
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this run is pending.
    pub unsafe fn set_switches(&mut self, cnq: &mut Cnq, sw: Switches) {
        cuda::sync();
        // CROW_GLM_CONTROLLER: the thread and the ring go (a row ahead is dropped unrestored:
        // the switches change between sequences)
        if let Some(mut w) = self.worker.take() {
            w.free();
        }
        self.ahead = None;
        if let Some(mut c) = self.pass.ctl.take() {
            c.free();
        }
        if let Some(r) = self.pass.routed.as_mut() {
            r.free();
        }
        self.pass.routed = None;
        if let Some(f) = self.feed.as_mut() {
            f.free();
        }
        self.feed = None;
        if let Some(r) = self.readback.as_mut() {
            r.free();
        }
        self.readback = None;
        if let Some(k) = self.sw_kernels.as_mut() {
            k.free();
        }
        self.sw_kernels = None;
        self.sw = sw;
        self.pass.overlap = false;
        if sw == Switches::default() {
            return;
        }
        let k = glm5_flags::Kernels::new(&self.g);
        if sw.flags {
            // the decode calls' ids (one row); #186 prompt calls and #192 verify calls of more rows
            // read theirs after a stream sync
            let mut r = Routed::new(&k, self.moe.topk);
            if sw.prefetch {
                r.predict_on(&self.layers, &self.moe, sw.pf_side);
            }
            self.pass.routed = Some(r);
        }
        self.pass.overlap = sw.flags && sw.overlap;
        if sw.lookahead {
            self.feed = Some(Feed::load(&k, cnq, &self.g));
            self.readback = Some(Readback::new(self.g.vocab));
        }
        if sw.controller && sw.flags {
            self.pass.ctl = Some(glm5_flags::Ctl::new(&k, self.moe.topk, self.g.layers));
            self.worker = Some(glm5_flags::Worker::new());
            if sw.la && !sw.lookahead {
                // the reference's prep_model: the gather reads a host-mapped table
                self.feed = Some(Feed::load_mapped(&k, cnq, &self.g));
                self.readback = Some(Readback::new(self.g.vocab));
            }
        }
        self.sw_kernels = Some(k);
    }

    /// Greedy: feed `prompt`, then generate `n` ids, every row one decode call through all
    /// layers with the experts from `tiers`. `report` sees every row, in order. Starts a new
    /// sequence (every KDA state zeroed). Switches off, every row is [`Glm5Run::row`].
    ///
    /// #189 (`CROW_GLM_LOOKAHEAD`): a row with a head that is not the last queues the next row
    /// before the host reads its id. The next row's input is gathered on the GPU from the greedy
    /// id ([`Feed::gather`]); the id (and the logits) come back through pinned memory and are read
    /// at the next row's first MoE layer, after its routing sync or flag. That row's report fires
    /// then, so its `secs` end there (after the next row's dense prefix and first router).
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn generate(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], n: usize, keep_logits: bool, report: &mut dyn FnMut(&TokenReport)) -> Result<Generated, String> {
        if prompt.is_empty() || n == 0 {
            return Err("glm5_run: an empty prompt or nothing to generate".into());
        }
        let rows = prompt.len() + n - 1;
        if rows > self.cap {
            return Err(format!("glm5_run: {} prompt + {n} generated ids need {rows} rows, the caches hold {}", prompt.len(), self.cap));
        }
        for s in self.kda.iter().flatten() {
            s.reset();
        }
        if self.spec.is_some() {
            return self.generate_spec(cnq, tiers, prompt, n, keep_logits, report);
        }
        if self.sw.la && self.pass.ctl.is_some() {
            return self.generate_la(cnq, tiers, prompt, n, keep_logits, report);
        }
        let mut out = Generated::default();
        // #186: with a prompt chunk above 1 the prompt rows run as prompt calls; the rows from
        // `start` on (the generated ones) run below as before
        let start = if self.prompt_chunk > 1 {
            let id = self.prefill(cnq, tiers, prompt, 0, report)?;
            if keep_logits {
                out.logits.push(cuda::dtoh(self.logits, self.g.vocab));
            }
            out.ids.push(id);
            prompt.len()
        } else {
            0
        };
        if self.feed.is_none() {
            let pn = prompt.len();
            for pos in start..rows {
                let t0 = std::time::Instant::now();
                let base = RowBase::of(tiers);
                let tok = if pos < pn { prompt[pos] } else { out.ids[pos - pn] };
                let next = self.row(cnq, tiers, tok, pos, pos + 1 >= pn)?;
                if let Some(id) = next {
                    if keep_logits {
                        out.logits.push(cuda::dtoh(self.logits, self.g.vocab));
                    }
                    out.ids.push(id);
                }
                report(&base.report(tiers, pos, pos < pn, next, t0));
            }
            return Ok(out);
        }
        let rb = self.readback.take().expect("glm5_run: lookahead without its readback");
        let r = self.generate_ahead(cnq, tiers, prompt, start, rows, keep_logits, &rb, &mut out, report);
        self.readback = Some(rb);
        r.map(|_| out)
    }

    /// [`Glm5Run::generate`] with the lookahead (#189)
    #[allow(clippy::too_many_arguments)]
    unsafe fn generate_ahead(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], start: usize, rows: usize, keep_logits: bool, rb: &Readback, out: &mut Generated, report: &mut dyn FnMut(&TokenReport)) -> Result<(), String> {
        let (pn, vocab) = (prompt.len(), self.g.vocab);
        // the host reads a pending row's id: only after a sync point that follows its readback
        let finish = |p: Pending, t: &ExpertTiers, out: &mut Generated, report: &mut dyn FnMut(&TokenReport)| -> Result<(), String> {
            let id = rb.id() as i64;
            if !(0..vocab as i64).contains(&id) {
                return Err(format!("glm5_run: row {}: the greedy id {id} is outside the vocab of {vocab}", p.pos));
            }
            if keep_logits {
                out.logits.push(rb.logits());
            }
            out.ids.push(id);
            report(&p.base.report(t, p.pos, p.prompt, Some(id), p.t0));
            Ok(())
        };
        let mut pending: Option<Pending> = None;
        for pos in start..rows {
            let t0 = std::time::Instant::now();
            let base = RowBase::of(tiers);
            if pos < pn {
                self.embed(cnq, prompt[pos]);
            } else {
                // the previous row had the head and is pending: its greedy id is in `next`
                self.feed.as_ref().expect("glm5_run: lookahead without its feed").gather(self.next, self.x);
            }
            {
                let mut first = |t: &ExpertTiers| -> Result<(), String> {
                    match pending.take() {
                        Some(p) => finish(p, t, out, report),
                        None => Ok(()),
                    }
                };
                self.layers(tiers, pos, &mut first)?;
            }
            // no MoE layer read the routing in this row: read the pending id after a sync
            if let Some(p) = pending.take() {
                cuda::sync();
                finish(p, tiers, out, report)?;
            }
            if pos + 1 < pn {
                cuda::sync();
                report(&base.report(tiers, pos, true, None, t0));
                continue;
            }
            self.head_row()?;
            rb.enqueue(self.next, keep_logits.then_some(self.logits));
            let p = Pending { pos, prompt: pos < pn, t0, base };
            if pos + 1 < rows {
                // WDDM: submit the head, so it runs while the host queues the next row
                cuda::stream_query(cuda::cur_stream());
                pending = Some(p);
            } else {
                cuda::sync();
                tiers.settle()?;
                finish(p, tiers, out, report)?;
            }
        }
        Ok(())
    }

    /// the embedding row of `tok` (read from the container on the host) into every stream of `x`
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn embed(&mut self, cnq: &mut Cnq, tok: i64) {
        let (g, h) = (self.g, self.g.hidden);
        let e = gm::embed_rows(cnq, &g, &[tok]);
        cuda::to_f32_into(self.x, &gm::trunk_input(&e, h, g.hc_streams));
    }

    /// Every layer of row `pos` on `x`, the experts from `tiers`. `first` sees the store once, in
    /// the row's first MoE layer after its routing reached the host and before the store moves
    /// anything (the lookahead reads the previous row's id there).
    ///
    /// #190: with `CROW_GLM_GRAPH` the row is replayed from its captured segments, or captured
    /// from [`Glm5Run::layers_eager`] when there are none for this position's key.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    unsafe fn layers(&mut self, tiers: &mut ExpertTiers, pos: usize, first: &mut dyn FnMut(&ExpertTiers) -> Result<(), String>) -> Result<(), String> {
        let Some(gr) = self.graph.as_mut() else {
            return self.layers_eager(tiers, pos, first);
        };
        if tiers.pinned_use.cpu_lane {
            return Err(format!("{}=1 and {CPU_LANE_ENV}=1: the CPU lane computes experts on the host inside the post-router segment; turn one of them off", glm5_graph::ENV));
        }
        gr.stage(self.pass.mla_st(), pos);
        if moe_layers(&self.g) == 0 {
            // the pinned scalars are rewritten next row: no router wait orders that after this copy
            cuda::sync();
        }
        let key = glm5_graph::Key { score_grid: crate::glm5_mla::score_grid(pos, 1), tables: tiers.tables().to_vec() };
        if gr.promote(&key) {
            return self.replay(tiers, pos, first);
        }
        let stream = gr.stream;
        self.pass.ensure_plans(1);
        glm5_graph::begin_row(stream);
        if let Err(e) = self.layers_eager(tiers, pos, first) {
            glm5_graph::abort_row();
            return Err(e);
        }
        let c = glm5_graph::end_row().map_err(|e| format!("glm5_run: row {pos}: {e}"))?;
        self.graph.as_mut().expect("glm5_run: the graphs went away during a capture").keep(key, c);
        Ok(())
    }

    /// #190: one row from the captured segments: segment 0, then per MoE layer its router ids to
    /// the host, `first` (once), `table_for`, the next segment
    ///
    /// # Safety
    /// As [`Glm5Run::layers`]; the row's graphs are ready for this position's key and `st` is staged.
    unsafe fn replay(&mut self, tiers: &mut ExpertTiers, pos: usize, first: &mut dyn FnMut(&ExpertTiers) -> Result<(), String>) -> Result<(), String> {
        let gr = self.graph.as_mut().expect("glm5_run: replay without graphs");
        gr.replays += 1;
        let c = gr.current().expect("glm5_run: replay without a captured row");
        glm5_graph::launch(c, 0);
        for (i, s) in c.seams.iter().enumerate() {
            let ids = gm::router_ids(self.pass.routed.as_mut(), s.ids, s.n, s.layer).map_err(|e| format!("glm5_run: row {pos} layer {}: {e}", s.layer))?;
            if i == 0 {
                first(&*tiers).map_err(|e| format!("glm5_run: row {pos} layer {}: {e}", s.layer))?;
            }
            let (tb, _) = tiers.table_for(s.layer, &ids).map_err(|e| format!("glm5_run: row {pos} layer {}: {e}", s.layer))?;
            if tb != s.table {
                return Err(format!("glm5_run: row {pos} layer {}: {}: the record table moved from {:#x} to {tb:#x} after the capture", s.layer, glm5_graph::ENV, s.table));
            }
            glm5_graph::launch(c, i + 1);
        }
        Ok(())
    }

    /// #190: the head on `x` (greedy id into `next`); with `CROW_GLM_GRAPH` one graph, captured
    /// on its first use
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn head_row(&mut self) -> Result<(), String> {
        let Some(gr) = self.graph.as_mut() else {
            gm::run_head(&self.pass.kn, &self.head, &self.hw, self.x, self.normed, self.logits, self.next, 1);
            return Ok(());
        };
        if let Some(c) = gr.head.as_ref() {
            glm5_graph::launch(c, 0);
            return Ok(());
        }
        glm5_graph::begin_row(gr.stream);
        gm::run_head(&self.pass.kn, &self.head, &self.hw, self.x, self.normed, self.logits, self.next, 1);
        gr.head = Some(glm5_graph::end_row().map_err(|e| format!("glm5_run: head: {e}"))?);
        gr.head_captures += 1;
        Ok(())
    }

    /// [`Glm5Run::layers`], every launch eager (with a capture open, into its segments).
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    unsafe fn layers_eager(&mut self, tiers: &mut ExpertTiers, pos: usize, first: &mut dyn FnMut(&ExpertTiers) -> Result<(), String>) -> Result<(), String> {
        let mut seen = false;
        for l in 0..self.g.layers {
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            let mut hook = |layer: usize, sel: &[i32]| -> Result<Dev, String> {
                if !seen {
                    seen = true;
                    first(&*tiers)?;
                }
                tiers.table_for(layer, sel).map(|(tb, _)| tb)
            };
            let r = self.pass.call_with_experts(&self.layers[l], self.x, pos, 1, true, &mut hook);
            // the layer's own state goes back even when the call failed
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            r.map_err(|e| format!("glm5_run: row {pos} layer {l}: {e}"))?;
        }
        Ok(())
    }

    /// # Safety
    /// No launch of this run is pending.
    pub unsafe fn free(&mut self) {
        cuda::sync();
        for lw in self.layers.iter_mut() {
            lw.free();
        }
        for s in self.kda.iter_mut().flatten() {
            s.free();
        }
        for c in self.mla.iter_mut().flatten() {
            c.free();
        }
        self.pass.free();
        self.head.free();
        for d in [&mut self.hw.norm, &mut self.hw.lm, &mut self.x, &mut self.normed, &mut self.logits, &mut self.next] {
            cuda::free_dev(d);
        }
        if let Some(f) = self.feed.as_mut() {
            f.free();
        }
        if let Some(r) = self.readback.as_mut() {
            r.free();
        }
        if let Some(k) = self.sw_kernels.as_mut() {
            k.free();
        }
        if let Some(mut gr) = self.graph.take() {
            gr.free();
        }
        if let Some(mut sp) = self.spec.take() {
            sp.free();
        }
        if let Some(mut w) = self.worker.take() {
            w.free();
        }
        if let Some(mut c) = self.pass.ctl.take() {
            c.free();
        }
        for (a, b) in self.kda_bak.iter_mut() {
            cuda::free_dev(a);
            cuda::free_dev(b);
        }
        self.kda_bak.clear();
        // CROW_GLM_MAX_BATCH: the parked sequences and the batch head
        self.free_slots();
    }
}

/// the number of MoE layers a run serves through the tiers (`layers - dense_prefix`)
pub fn moe_layers(g: &Glm5Geo) -> usize {
    (0..g.layers).filter(|&l| gm::ffn_kind(g, l) == FfnKind::Moe).count()
}

/// the fixed prompt of the cache-size test and the default smoke: the tokenizer golden
/// `sys_user_default` (system + user message, generation prompt; 31 ids, rendered by
/// transformers 5.16.1 from rev `eb9eb208`)
pub fn fixed_prompt() -> Vec<i64> {
    const GOLDENS: &str = include_str!("../tests/fixtures/GLM-5.3-Flash/tokenizer-goldens.json");
    let v: serde_json::Value = serde_json::from_str(GOLDENS).expect("tokenizer goldens");
    let r = v["render"].as_array().expect("render cases").iter().find(|c| c["name"] == "sys_user_default").expect("sys_user_default");
    r["ids"].as_array().expect("ids").iter().map(|x| x.as_i64().expect("an id")).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::expert_cache::Policy;

    /// The host twin of the device store: every slot holds the id of the record in it (`EMPTY`
    /// none), the landing buffer and staging slots likewise; the container "reads" an expert as
    /// its id. A copy moves the id, so a slot overwritten before it is read shows up as a wrong
    /// id where the record is looked up.
    struct Sim {
        vram: Vec<u32>,
        pinned: Vec<u32>,
        stage: Vec<u32>,
        landing: Vec<u32>,
        /// copies queued since the last barrier: the sim executes them at once, so it records
        /// every slot a queued copy READ, and a host-side NVMe write into such a slot before the
        /// barrier is the race the GPU would have
        pending_reads_pinned: Vec<u32>,
        reads: usize,
        /// #187: the mover calls the twin saw, counted independently of `serve`'s own counts
        ops: Moves,
    }

    const EMPTY: u32 = u32::MAX - 1;

    impl Sim {
        fn new(s: TierSizes, stage: usize) -> Sim {
            Sim { vram: vec![EMPTY; s.vram], pinned: vec![EMPTY; s.pinned], stage: vec![EMPTY; stage], landing: vec![EMPTY; stage], pending_reads_pinned: Vec::new(), reads: 0, ops: Moves::default() }
        }
    }

    impl Mover for Sim {
        fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
            assert!(jobs.len() <= MAX_IN_FLIGHT);
            for &(e, d) in jobs {
                match d {
                    Dst::Landing(i) => {
                        self.landing[i as usize] = e;
                        self.ops.nvme_to_landing += 1;
                    }
                    Dst::Pinned(q) => {
                        assert!(!self.pending_reads_pinned.contains(&q), "NVMe wrote pinned slot {q} while a queued copy still reads it (no barrier)");
                        self.pinned[q as usize] = e;
                        self.ops.nvme_to_pinned += 1;
                    }
                }
                self.reads += 1;
            }
            Ok(jobs.len() as u64 * 100)
        }
        fn landing_to_stage(&mut self, i: u32) {
            self.ops.landing_to_stage += 1;
            self.stage[i as usize] = self.landing[i as usize];
        }
        fn pinned_to_stage(&mut self, q: u32, s: u32) {
            self.ops.pinned_to_stage += 1;
            self.pending_reads_pinned.push(q);
            self.stage[s as usize] = self.pinned[q as usize];
        }
        fn vram_to_stage(&mut self, v: u32, s: u32) {
            self.ops.vram_to_stage += 1;
            self.stage[s as usize] = self.vram[v as usize];
        }
        fn barrier(&mut self) {
            self.pending_reads_pinned.clear();
        }
        fn vram_to_pinned(&mut self, v: u32, q: u32) {
            self.ops.vram_to_pinned += 1;
            self.pinned[q as usize] = self.vram[v as usize];
        }
        fn stage_to_vram(&mut self, s: u32, v: u32) {
            self.ops.stage_to_vram += 1;
            self.vram[v as usize] = self.stage[s as usize];
        }
    }

    /// the xorshift routing of `expert_cache`'s tests: `k` distinct ids per token per layer with
    /// a drifting hot set
    pub(super) fn trace(tokens: usize, layers: usize, experts: u64, k: usize, seed: u64) -> Vec<Vec<Vec<u32>>> {
        let mut x: u64 = 0x9E37_79B9_7F4A_7C15 ^ seed;
        (0..tokens)
            .map(|t| {
                let base = (t as u64 / 50) * 16;
                (0..layers as u64)
                    .map(|l| {
                        let mut got: Vec<u32> = Vec::with_capacity(k);
                        while got.len() < k {
                            x ^= x << 13;
                            x ^= x >> 7;
                            x ^= x << 17;
                            let e = if (x >> 32) % 10 < 7 { (base + l * 7 + x % 24) % experts } else { (x >> 8) % experts } as u32;
                            if !got.contains(&e) {
                                got.push(e);
                            }
                        }
                        got.sort_unstable();
                        got
                    })
                    .collect()
            })
            .collect()
    }

    /// after a call: every selected id's location holds that id, every cached expert's slot
    /// holds it, and the call read from NVMe exactly the selected records no tier held at its
    /// start (`nvme_at_start`). The cache's NVMe counter can be higher: below V + P = top-k an
    /// expert evicted earlier in the same call counts as NVMe-served, and `serve` stages it from
    /// its old slot instead of reading it again.
    #[allow(clippy::too_many_arguments)]
    fn check(c: &ExpertCache, l: usize, slots: &LayerSlots, sim: &Sim, served: &Served, ids: &[u32], nvme_at_start: usize, nvme_before: u64) {
        assert_eq!(served.locs.iter().map(|x| x.0).collect::<Vec<_>>(), ids);
        for &(e, loc) in &served.locs {
            let got = match loc {
                Loc::Vram(v) => sim.vram[v as usize],
                Loc::Pinned(q) => sim.pinned[q as usize],
                Loc::Stage(s) => sim.stage[s as usize],
            };
            assert_eq!(got, e, "layer {l}: expert {e} at {loc:?} holds {got}");
        }
        for e in 0..c.experts as u32 {
            match c.tier(l, e) {
                Tier::Vram => assert_eq!(sim.vram[slots.vram_of[e as usize] as usize], e, "VRAM slot of {e}"),
                Tier::Pinned => assert_eq!(sim.pinned[slots.pin_of[e as usize] as usize], e, "pinned slot of {e}"),
                Tier::Nvme => assert!(slots.vram_of[e as usize] == NONE && slots.pin_of[e as usize] == NONE, "NVMe expert {e} keeps a slot"),
            }
        }
        assert_eq!(served.nvme_reads, nvme_at_start, "NVMe reads = selected records in no tier at the start of the call");
        assert!(served.nvme_reads as u64 <= c.counters()[l][2] - nvme_before, "more NVMe reads than NVMe-served accesses");
    }

    /// Every policy, capacities from all-NVMe (0 + 0) through VRAM-only, pinned-only, V + P below
    /// top-8 (records evicted within the call) up to every expert cached: the moves of `serve`
    /// leave every selected record where the table points and every cached record in its slot,
    /// read a record from NVMe exactly when no tier held it at the start of the call (never more
    /// often than the cache's NVMe counter: the policy may evict a later id of the call before
    /// its access, whose record is then staged from its old slot), and never let an NVMe write
    /// hit a pinned slot a queued copy still reads.
    #[test]
    fn the_moves_put_every_record_where_the_table_points() {
        let (layers, experts, k) = (3, 64, 8);
        let tr = trace(150, layers, experts as u64, k, 0x175);
        let policies = [Policy::Lru, Policy::Clock { admit: None }, Policy::Clock { admit: Some(2) }, Policy::Lfu { decay: 0.7 }];
        for p in policies {
            for (v, pin) in [(0, 0), (0, 8), (8, 0), (1, 7), (2, 3), (3, 12), (16, 40), (24, 40)] {
                let sizes = TierSizes { vram: v, pinned: pin };
                let mut c = ExpertCache::new(p, Scope::PerLayer, layers, experts, v, pin).unwrap();
                let mut slots: Vec<LayerSlots> = (0..layers).map(|_| LayerSlots::new(experts, sizes)).collect();
                let mut sims: Vec<Sim> = (0..layers).map(|_| Sim::new(sizes, k)).collect();
                for tok in &tr {
                    for (l, ids) in tok.iter().enumerate() {
                        let nv = c.counters()[l][2];
                        let cold = ids.iter().filter(|&&e| c.tier(l, e) == Tier::Nvme).count();
                        let s = serve(&mut c, l, &mut slots[l], ids, k, &mut sims[l]).unwrap_or_else(|e| panic!("{p:?} V {v} P {pin}: {e}"));
                        check(&c, l, &slots[l], &sims[l], &s, ids, cold, nv);
                        assert_eq!(s.nvme_bytes, s.nvme_reads as u64 * 100);
                    }
                }
                let reads: usize = sims.iter().map(|s| s.reads).sum();
                let nvme: u64 = c.counters().iter().map(|x| x[2]).sum();
                assert!(reads as u64 <= nvme, "{p:?} V {v} P {pin}");
                if v + pin == 0 {
                    assert_eq!(reads, 150 * layers * k, "all-NVMe reads every selected record");
                }
            }
        }
    }

    /// LRU V 2 + P 1 by hand, one layer, one id per token: 1 2 1 3 2 4 1 (the sequence of
    /// `expert_cache::tests::lru_hand_sequence`). The record of the promoted pinned expert goes
    /// to VRAM through staging, the VRAM victim lands in the pinned slot it left.
    #[test]
    fn lru_by_hand_three_way_exchange() {
        let sizes = TierSizes { vram: 2, pinned: 1 };
        let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, 1, 16, 2, 1).unwrap();
        let mut slots = LayerSlots::new(16, sizes);
        let mut sim = Sim::new(sizes, 8);
        let mut locs = Vec::new();
        for e in [1u32, 2, 1, 3, 2, 4, 1] {
            let s = serve(&mut c, 0, &mut slots, &[e], 8, &mut sim).unwrap();
            locs.push((s.locs[0].1, s.nvme_reads));
        }
        assert_eq!(c.counters()[0], [1, 1, 5]);
        // 1 N -> V0, 2 N -> V1, 1 V0, 3 N: victim 2 to pinned, 3 into V1; 2 pinned hit: victim 1
        // into the pinned slot 2 left, 2 into V0; 4 N: victim 3 (V1) drops 1 from pinned; 1 N
        assert_eq!(
            locs,
            vec![(Loc::Vram(0), 1), (Loc::Vram(1), 1), (Loc::Vram(0), 0), (Loc::Vram(1), 1), (Loc::Vram(0), 0), (Loc::Vram(1), 1), (Loc::Vram(0), 1)]
        );
        assert_eq!((sim.vram.clone(), sim.pinned.clone()), (vec![1, 4], vec![2]));
    }

    #[test]
    fn the_sizes_are_the_plans_and_a_larger_ask_is_refused_by_name() {
        use crate::geo::{GLM5_NEXT_DENSE_BYTES, HOST_PINNED_CAP};
        let g = Glm5Geo::GLM_5_3_FLASH;
        // the RTX 5090 of record (32,607 MiB), 200,000 tokens, the 3-bit record, 46 GiB pinned
        let (_, _, plan) = crate::manager::plan_glm5_next(&g, 200_000, 32_607 << 20, HOST_PINNED_CAP, GLM5_NEXT_DENSE_BYTES, 9_474_048, 64, true).unwrap();
        assert_eq!((plan.hot, plan.pinned, plan.nvme), (50, 124, 114));
        assert_eq!(tier_sizes(&plan, None, None), Ok(TierSizes { vram: 50, pinned: 124 }));
        assert_eq!(tier_sizes(&plan, Some(1), Some(7)), Ok(TierSizes { vram: 1, pinned: 7 }));
        assert_eq!(tier_sizes(&plan, Some(0), Some(0)), Ok(TierSizes { vram: 0, pinned: 0 }));
        let e = tier_sizes(&plan, Some(51), None).unwrap_err();
        assert!(e.starts_with("--vram-slots 51 per MoE layer is above the #159 plan's 50"), "{e}");
        let e = tier_sizes(&plan, None, Some(125)).unwrap_err();
        assert!(e.starts_with("--pinned-slots 125 per MoE layer is above the #159 plan's 124"), "{e}");
        // 124 pinned records per MoE layer fit the 46 GiB cap
        assert!(124 * moe_layers(&g) as u64 * 9_474_048 <= HOST_PINNED_CAP);
        assert_eq!(moe_layers(&g), 42);
    }

    #[test]
    fn a_selection_outside_the_experts_is_refused_and_ids_are_distinct_ascending() {
        assert_eq!(distinct_ids(&[7, 3, 3, 250, 0, 7], 288), Ok(vec![0, 3, 7, 250]));
        assert!(distinct_ids(&[288], 288).unwrap_err().contains("id 288, outside 0..288"));
        assert!(distinct_ids(&[-1], 288).is_err());
        let p = fixed_prompt();
        assert_eq!(&p[..3], &[154822, 154824, 154826]);
    }

    /// #187: the host counters of `serve`, against what the twin saw and the cache's tiers, under
    /// every policy and capacity of the trace test. Per call: visits = distinct ids = the cache's
    /// `vram + pinned + nvme` accesses; every mover count = the twin's calls of that kind; NVMe
    /// reads = `Served::nvme_reads`; the six transitions = the before/after tier diff counted
    /// here; zero-copy = the ids served from their pinned slot. All-NVMe: visits = NVMe reads,
    /// no promotion, no eviction.
    #[test]
    fn the_move_counters_equal_the_twins_calls_and_the_tier_diff() {
        let (layers, experts, k) = (3, 64, 8);
        let tr = trace(150, layers, experts as u64, k, 0x187);
        let policies = [Policy::Lru, Policy::Clock { admit: None }, Policy::Clock { admit: Some(2) }, Policy::Lfu { decay: 0.7 }];
        for p in policies {
            for (v, pin) in [(0, 0), (0, 8), (8, 0), (1, 7), (2, 3), (3, 12), (16, 40), (24, 40)] {
                let sizes = TierSizes { vram: v, pinned: pin };
                let mut c = ExpertCache::new(p, Scope::PerLayer, layers, experts, v, pin).unwrap();
                let mut slots: Vec<LayerSlots> = (0..layers).map(|_| LayerSlots::new(experts, sizes)).collect();
                let mut sims: Vec<Sim> = (0..layers).map(|_| Sim::new(sizes, k)).collect();
                let mut total = Moves::default();
                for tok in &tr {
                    for (l, ids) in tok.iter().enumerate() {
                        let before: Vec<Tier> = (0..experts as u32).map(|e| c.tier(l, e)).collect();
                        let c0 = c.counters()[l];
                        let ops0 = sims[l].ops;
                        let s = serve(&mut c, l, &mut slots[l], ids, k, &mut sims[l]).unwrap();
                        let after: Vec<Tier> = (0..experts as u32).map(|e| c.tier(l, e)).collect();
                        let m = s.moves;
                        let what = format!("{p:?} V {v} P {pin} layer {l}");
                        let acc: u64 = (0..3).map(|i| c.counters()[l][i] - c0[i]).sum();
                        assert_eq!((m.visits, acc), (ids.len() as u64, ids.len() as u64), "{what}: visits");
                        let ops = sims[l].ops.since(&ops0);
                        assert_eq!(
                            [m.nvme_to_landing, m.nvme_to_pinned, m.landing_to_stage, m.pinned_to_stage, m.vram_to_stage, m.vram_to_pinned, m.stage_to_vram],
                            [ops.nvme_to_landing, ops.nvme_to_pinned, ops.landing_to_stage, ops.pinned_to_stage, ops.vram_to_stage, ops.vram_to_pinned, ops.stage_to_vram],
                            "{what}: mover calls"
                        );
                        assert_eq!(m.nvme_reads(), s.nvme_reads as u64, "{what}: NVMe reads");
                        let tr = |a: Tier, b: Tier| before.iter().zip(&after).filter(|(x, y)| **x == a && **y == b).count() as u64;
                        assert_eq!(
                            [m.n2v, m.p2v, m.n2p, m.v2p, m.v2n, m.p2n],
                            [
                                tr(Tier::Nvme, Tier::Vram),
                                tr(Tier::Pinned, Tier::Vram),
                                tr(Tier::Nvme, Tier::Pinned),
                                tr(Tier::Vram, Tier::Pinned),
                                tr(Tier::Vram, Tier::Nvme),
                                tr(Tier::Pinned, Tier::Nvme)
                            ],
                            "{what}: transitions"
                        );
                        assert_eq!(m.zero_copy, s.locs.iter().filter(|x| matches!(x.1, Loc::Pinned(_))).count() as u64, "{what}: zero-copy");
                        total.add(&m);
                    }
                }
                assert_eq!(total.visits, (150 * layers * k) as u64);
                if v + pin == 0 {
                    assert_eq!((total.nvme_reads(), total.promotions(), total.evictions()), (total.visits, 0, 0), "{p:?}: all-NVMe");
                }
            }
        }
    }

    /// #187 (`glm5_run --cold`): after `reset_cache` the same trace moves exactly as the first
    /// pass did (call by call); without it the second pass starts warm and reads no more from
    /// NVMe than the first (fewer at every capacity that holds anything).
    #[test]
    fn reset_cache_replays_the_cold_pass_and_warm_reads_less() {
        let (layers, experts, k) = (3, 64, 8);
        let tr = trace(120, layers, experts as u64, k, 0x188);
        for (v, pin) in [(0, 0), (1, 7), (8, 0), (3, 12), (16, 40)] {
            let sizes = TierSizes { vram: v, pinned: pin };
            let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, layers, experts, v, pin).unwrap();
            let mut slots: Vec<LayerSlots> = (0..layers).map(|_| LayerSlots::new(experts, sizes)).collect();
            let mut sims: Vec<Sim> = (0..layers).map(|_| Sim::new(sizes, k)).collect();
            let mut pass = |c: &mut ExpertCache, slots: &mut Vec<LayerSlots>| -> Vec<Served> {
                let mut out = Vec::new();
                for tok in &tr {
                    for (l, ids) in tok.iter().enumerate() {
                        let nv = c.counters()[l][2];
                        let cold = ids.iter().filter(|&&e| c.tier(l, e) == Tier::Nvme).count();
                        let s = serve(c, l, &mut slots[l], ids, k, &mut sims[l]).unwrap();
                        check(c, l, &slots[l], &sims[l], &s, ids, cold, nv);
                        out.push(s);
                    }
                }
                out
            };
            let first = pass(&mut c, &mut slots);
            let warm = pass(&mut c, &mut slots);
            reset_cache(&mut c, &mut slots, sizes).unwrap();
            assert!((0..layers).all(|l| (0..experts as u32).all(|e| c.tier(l, e) == Tier::Nvme)), "V {v} P {pin}: reset leaves a cached expert");
            assert!(c.counters().iter().all(|x| *x == [0, 0, 0]));
            let cold = pass(&mut c, &mut slots);
            let reads = |x: &[Served]| x.iter().map(|s| s.nvme_reads).sum::<usize>();
            assert_eq!(cold.iter().map(|s| (s.locs.clone(), s.moves)).collect::<Vec<_>>(), first.iter().map(|s| (s.locs.clone(), s.moves)).collect::<Vec<_>>(), "V {v} P {pin}: cold pass");
            assert!(reads(&warm) <= reads(&first), "V {v} P {pin}");
            if v + pin > 0 {
                assert!(reads(&warm) < reads(&first), "V {v} P {pin}: a warm pass reads less");
            }
        }
    }

    /// more selected-and-uncached records than staging slots is refused by name, not overrun
    #[test]
    fn staging_overflow_is_refused_by_name() {
        let sizes = TierSizes { vram: 0, pinned: 0 };
        let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, 1, 16, 0, 0).unwrap();
        let mut slots = LayerSlots::new(16, sizes);
        let mut sim = Sim::new(sizes, 8);
        let e = serve(&mut c, 0, &mut slots, &[0, 1, 2, 3, 4, 5, 6, 7, 8], 8, &mut sim).unwrap_err();
        assert!(e.contains("stages 9 records in one call, 8 staging slots"), "{e}");
    }

    /// #186: a Mover over a shared twin, so a sub-batch callback of [`serve_chunk`] can look at
    /// the slots while the chunk is being served
    struct SharedSim<'a>(&'a std::cell::RefCell<Sim>);

    impl Mover for SharedSim<'_> {
        fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
            self.0.borrow_mut().nvme(jobs)
        }
        fn landing_to_stage(&mut self, i: u32) {
            self.0.borrow_mut().landing_to_stage(i)
        }
        fn pinned_to_stage(&mut self, q: u32, s: u32) {
            self.0.borrow_mut().pinned_to_stage(q, s)
        }
        fn vram_to_stage(&mut self, v: u32, s: u32) {
            self.0.borrow_mut().vram_to_stage(v, s)
        }
        fn barrier(&mut self) {
            let mut sim = self.0.borrow_mut();
            sim.barrier();
            sim.ops.visits += 1; // counts the barriers (no serve counts visits on the twin)
        }
        fn vram_to_pinned(&mut self, v: u32, q: u32) {
            self.0.borrow_mut().vram_to_pinned(v, q)
        }
        fn stage_to_vram(&mut self, s: u32, v: u32) {
            self.0.borrow_mut().stage_to_vram(s, v)
        }
    }

    /// #186 (Expected result 5): one prompt call of 512 rows x top-8 over 288 experts, 3 MoE
    /// layers, the 128-slot prefill staging set, at V 0 + P 0 (every record staged) and the
    /// #159 plan's V 50 + P 124: the call is served in row sub-batches, each one `serve` whose
    /// staged records fit the 128 slots; the sub-batches cover the 512 rows once, in order; after
    /// each, every (row, expert) of its rows finds its record where the table points, and every
    /// cached record sits in its slot; the mover waits between sub-batches. At V 0 + P 0 the call
    /// needs more than one sub-batch (one `serve` of the whole call is refused: the defect).
    #[test]
    fn a_prompt_call_is_served_in_sub_batches_that_fit_the_prefill_set() {
        let (layers, experts, k, rows, cap) = (3, 288, 8, 512, 128);
        let tr = trace(rows, layers, experts as u64, k, 0x186);
        for (v, pin) in [(0, 0), (50, 124)] {
            let sizes = TierSizes { vram: v, pinned: pin };
            let mut c = ExpertCache::new(Policy::Lru, Scope::PerLayer, layers, experts, v, pin).unwrap();
            let mut slots: Vec<LayerSlots> = (0..layers).map(|_| LayerSlots::new(experts, sizes)).collect();
            for l in 0..layers {
                // the call's selection in pick order: [rows][k] i32
                let sel: Vec<i32> = tr.iter().flat_map(|tok| tok[l].iter().rev().map(|&e| e as i32)).collect();
                let sim = std::cell::RefCell::new(Sim::new(sizes, cap));
                let mut m = SharedSim(&sim);
                let mut seen: Vec<(usize, usize)> = Vec::new();
                let (cache_ptr, slots_ptr) = (&c as *const ExpertCache, &slots[l] as *const LayerSlots);
                let mut each = |r0: usize, n: usize, served: &Served| -> Result<(), String> {
                    let staged = served.moves.vram_to_stage + served.moves.pinned_to_stage + served.moves.nvme_to_landing;
                    assert!(staged as usize <= cap, "V {v} P {pin} layer {l}: rows {r0}..{} stage {staged} records", r0 + n);
                    let ids = distinct_ids(&sel[r0 * k..(r0 + n) * k], experts).unwrap();
                    assert_eq!(served.locs.iter().map(|x| x.0).collect::<Vec<_>>(), ids, "every id of the sub-batch has a location");
                    let sim = sim.borrow();
                    for &(e, loc) in &served.locs {
                        let got = match loc {
                            Loc::Vram(q) => sim.vram[q as usize],
                            Loc::Pinned(q) => sim.pinned[q as usize],
                            Loc::Stage(q) => sim.stage[q as usize],
                        };
                        assert_eq!(got, e, "V {v} P {pin} layer {l} rows {r0}..{}: expert {e} at {loc:?} holds {got}", r0 + n);
                    }
                    // SAFETY: read-only looks at the cache and the slots `serve` just left; no
                    // mutable borrow is used while this callback runs
                    let (cc, ss) = unsafe { (&*cache_ptr, &*slots_ptr) };
                    for e in 0..experts as u32 {
                        match cc.tier(l, e) {
                            Tier::Vram => assert_eq!(sim.vram[ss.vram_of[e as usize] as usize], e),
                            Tier::Pinned => assert_eq!(sim.pinned[ss.pin_of[e as usize] as usize], e),
                            Tier::Nvme => {}
                        }
                    }
                    seen.push((r0, n));
                    Ok(())
                };
                let batches = serve_chunk(&mut c, l, &mut slots[l], &sel, k, cap, &mut m, &mut each).unwrap_or_else(|e| panic!("V {v} P {pin} layer {l}: {e}"));
                assert_eq!(batches, seen.len());
                let mut at = 0;
                for &(r0, n) in &seen {
                    assert_eq!(r0, at, "sub-batches in row order, no gap");
                    assert!(n == rows - r0 || n.is_power_of_two(), "a sub-batch of {n} rows is the rest or a power of two");
                    at += n;
                }
                assert_eq!(at, rows, "every row served once");
                // every serve has its own barrier (after phase A); serve_chunk adds one before every later sub-batch
                assert_eq!(sim.borrow().ops.visits as usize, 2 * batches - 1, "one barrier before every later sub-batch");
                if v + pin == 0 {
                    assert!(batches > 1, "layer {l}: V 0 + P 0 must split the call (512 rows x top-8 > 128 slots)");
                }
                eprintln!("glm5_tiers #186 V {v} P {pin} layer {l}: 512 rows in {batches} sub-batches {:?}", seen.iter().map(|x| x.1).collect::<Vec<_>>());
            }
        }
    }

    /// Prefill never admits (glm53-flash-offload): a cache warmed by 40 decode calls (LRU, CLOCK,
    /// LFU; V 6 + P 10 of 64, 8 decode staging slots), then a prompt call of 48 rows x top-8
    /// through `serve_chunk_prefill` with 12 prefill slots, once as token rows and once as the
    /// expert-major pseudo-rows of `glm5_moe::ExpertMajor`. After every sub-batch each served
    /// location holds its record (VRAM hits in their slot, every other record in a staging slot)
    /// and every cached record sits in its slot; after the call every expert's tier and slot are
    /// those decode left, NVMe was read only for records no tier held, nothing went to pinned or
    /// VRAM, and no barrier was issued between sub-batches. The VRAM hits were marked (the LRU
    /// recency of a hit rises). The admitting `serve_chunk` on the same state moves tiers.
    #[test]
    fn a_prompt_call_admits_nothing_and_stages_the_rest() {
        let (e, k, l, rows, cap) = (64usize, 8usize, 0usize, 48usize, 12usize);
        let sizes = TierSizes { vram: 6, pinned: 10 };
        let sel_rows: Vec<i32> = trace(rows, 1, e as u64, k, 0x1a7).into_iter().flat_map(|t| t[0].iter().map(|&x| x as i32).collect::<Vec<_>>()).collect();
        let sel_em = crate::glm5_moe::ExpertMajor::new(&sel_rows, k, e, crate::glm5_moe::GROUP_ROWS).unwrap().sel();
        for policy in [Policy::Lru, Policy::Clock { admit: None }, Policy::Lfu { decay: 0.7 }] {
            for (name, sel) in [("token rows", &sel_rows), ("pseudo-rows", &sel_em)] {
                let mut c = ExpertCache::new(policy, Scope::PerLayer, 1, e, sizes.vram, sizes.pinned).unwrap();
                let mut slots = LayerSlots::new(e, sizes);
                let sim = std::cell::RefCell::new(Sim::new(sizes, cap));
                for tok in trace(40, 1, e as u64, k, 0x1a8) {
                    serve(&mut c, l, &mut slots, &tok[0], cap, &mut SharedSim(&sim)).unwrap();
                }
                let tiers0: Vec<Tier> = (0..e as u32).map(|x| c.tier(l, x)).collect();
                let (slots0, ops0) = (slots.clone(), sim.borrow().ops);
                let hit = (0..e as u32).find(|&x| tiers0[x as usize] == Tier::Vram && sel.contains(&(x as i32))).expect("a VRAM hit in the prompt");
                let other = (0..e as u32).find(|&x| tiers0[x as usize] == Tier::Vram && !sel.contains(&(x as i32)));
                let mut visits = vec![0u32; e];
                let (cache_ptr, slots_ptr) = (&c as *const ExpertCache, &slots as *const LayerSlots);
                let mut each = |r0: usize, n: usize, served: &Served| -> Result<(), String> {
                    let s = sim.borrow();
                    for &(x, loc) in &served.locs {
                        visits[x as usize] += 1;
                        let got = match loc {
                            Loc::Vram(q) => s.vram[q as usize],
                            Loc::Pinned(q) => s.pinned[q as usize],
                            Loc::Stage(q) => s.stage[q as usize],
                        };
                        assert_eq!(got, x, "{policy:?} {name} rows {r0}..{}: expert {x} at {loc:?} holds {got}", r0 + n);
                        assert_eq!(matches!(loc, Loc::Vram(_)), tiers0[x as usize] == Tier::Vram, "{policy:?} {name}: expert {x} at {loc:?}");
                    }
                    // SAFETY: read-only looks at the cache and the slots between sub-batches
                    let (cc, ss) = unsafe { (&*cache_ptr, &*slots_ptr) };
                    for x in 0..e as u32 {
                        match cc.tier(l, x) {
                            Tier::Vram => assert_eq!(s.vram[ss.vram_of[x as usize] as usize], x),
                            Tier::Pinned => assert_eq!(s.pinned[ss.pin_of[x as usize] as usize], x),
                            Tier::Nvme => {}
                        }
                    }
                    Ok(())
                };
                let batches = serve_chunk_prefill(&mut c, l, &slots, sel, k, cap, &mut SharedSim(&sim), &mut each).unwrap_or_else(|err| panic!("{policy:?} {name}: {err}"));
                assert_eq!((0..e as u32).map(|x| c.tier(l, x)).collect::<Vec<_>>(), tiers0, "{policy:?} {name}: the prompt call moved tiers");
                assert_eq!(slots, slots0, "{policy:?} {name}: the prompt call moved slots");
                let d = sim.borrow().ops.since(&ops0);
                let distinct = distinct_ids(sel, e).unwrap();
                let nvme_ids = distinct.iter().filter(|&&x| tiers0[x as usize] == Tier::Nvme).count() as u64;
                assert_eq!((d.visits, d.vram_to_pinned, d.stage_to_vram, d.nvme_to_pinned, d.vram_to_stage), (0, 0, 0, 0, 0), "{policy:?} {name}: no barrier, no admission moves");
                if name == "pseudo-rows" {
                    assert!(distinct.iter().all(|&x| visits[x as usize] == 1), "{policy:?}: every expert served once");
                    assert_eq!(d.nvme_to_landing, nvme_ids, "{policy:?}: NVMe read once per record no tier held");
                }
                if let (Policy::Lru, Some(other)) = (policy, other) {
                    assert_eq!(c.keep_order(l, hit, other), std::cmp::Ordering::Greater, "the VRAM hit {hit} was marked, the untouched {other} not");
                }
                eprintln!("glm5_tiers prefill {policy:?} {name}: {batches} sub-batches, {} NVMe reads, {} pinned stagings, max visits {}", d.nvme_to_landing, d.pinned_to_stage, visits.iter().max().unwrap());
                // the contrast: the admitting serve_chunk on the same state moves tiers
                let mut c2 = c.clone();
                let mut slots2 = slots.clone();
                serve_chunk(&mut c2, l, &mut slots2, sel, k, cap, &mut SharedSim(&sim), &mut |_, _, _| Ok(())).unwrap();
                assert_ne!((0..e as u32).map(|x| c2.tier(l, x)).collect::<Vec<_>>(), tiers0, "{policy:?} {name}: serve_chunk admits");
            }
        }
    }

    /// #186: the prompt calls cover the prompt once, in order: full chunks, then the remainder as
    /// one call (`ceil(n / chunk)` calls, one routing sync per MoE layer each); chunk 1 is one row
    /// per call. The booked FFN plan sizes: every power of two below the chunk, and the chunk
    #[test]
    fn the_prompt_calls_cover_the_prompt_in_full_chunks_and_one_rest() {
        for chunk in [1usize, 2, 3, 16, 32, 512] {
            for n in [1usize, 2, 7, 31, 32, 33, 86, 600] {
                let calls = prompt_calls(n, chunk);
                let mut at = 0;
                for &(r0, t) in &calls {
                    assert_eq!(r0, at);
                    assert!((1..=chunk).contains(&t));
                    at += t;
                }
                assert_eq!(at, n);
                assert_eq!(calls.len(), n.div_ceil(chunk));
            }
        }
        assert_eq!(prompt_calls(31, 16), vec![(0, 16), (16, 15)]);
        assert_eq!(prompt_calls(31, 32), vec![(0, 31)]);
        assert_eq!(prompt_calls(86, 32), vec![(0, 32), (32, 32), (64, 22)]);
        assert_eq!(crate::manager::glm5_prompt_call_sizes(1), Vec::<usize>::new());
        assert_eq!(crate::manager::glm5_prompt_call_sizes(32), vec![2, 4, 8, 16, 32]);
        assert_eq!(crate::manager::glm5_prompt_call_sizes(24), vec![2, 4, 8, 16, 24]);
    }

    /// #186: `fitting_rows` takes every row when they fit, else the largest power of two that
    /// does, and refuses one row that does not fit by name
    #[test]
    fn fitting_rows_takes_all_or_the_largest_power_of_two() {
        let c = ExpertCache::new(Policy::Lru, Scope::PerLayer, 1, 64, 0, 0).unwrap();
        // 10 rows of 2 picks, all distinct: r rows stage 2r records
        let sel: Vec<i32> = (0..20).collect();
        assert_eq!(fitting_rows(&c, 0, &sel, 2, 20), Ok(10));
        assert_eq!(fitting_rows(&c, 0, &sel, 2, 19), Ok(8));
        assert_eq!(fitting_rows(&c, 0, &sel, 2, 15), Ok(4));
        assert_eq!(fitting_rows(&c, 0, &sel, 2, 2), Ok(1));
        let e = fitting_rows(&c, 0, &sel, 2, 1).unwrap_err();
        assert!(e.contains("one row stages 2 records, 1 staging slots"), "{e}");
        // the same two ids in every row: any number of rows stages 2
        let same: Vec<i32> = (0..10).flat_map(|_| [5, 9]).collect();
        assert_eq!(fitting_rows(&c, 0, &same, 2, 2), Ok(10));
    }

    /// #188 `CROW_GLM_PINNED=zerocopy`: under every policy and capacity of the trace test, with
    /// pinned hits staying in pinned:
    /// - one id per call (no other access of the call can move it): no promotion pinned -> VRAM,
    ///   no H2D of a pinned record, every id that sat in pinned is served from its slot;
    /// - top-8 per call: an id the call's earlier misses push out of pinned before its own access
    ///   is an NVMe access by the policy's rule (it may enter VRAM, staged from its old slot):
    ///   every pinned -> VRAM transition is such a selected id, every pinned record staged is a
    ///   selected id that left pinned, the moves still put every record where the table points
    ///   (`check`), and at P >= top-k there are fewer pinned -> VRAM transitions over the trace
    ///   than under `promote` (below that the in-call churn dominates either way).
    ///
    /// Under `promote` the one-id traces do promote pinned hits (so the test can fail), and
    /// `reset_cache` keeps the option.
    #[test]
    fn zerocopy_pinned_hits_stay_and_are_read_in_place() {
        let (layers, experts) = (3, 64);
        let policies = [Policy::Lru, Policy::Clock { admit: None }, Policy::Clock { admit: Some(2) }, Policy::Lfu { decay: 0.7 }];
        for k in [1usize, 8] {
            let tr = trace(150, layers, experts as u64, k, 0x188);
            for p in policies {
                for (v, pin) in [(0, 0), (0, 8), (8, 0), (1, 7), (2, 3), (3, 12), (16, 40), (24, 40)] {
                    let sizes = TierSizes { vram: v, pinned: pin };
                    let mut promoted = 0u64;
                    for stay in [false, true] {
                        let mut c = ExpertCache::new(p, Scope::PerLayer, layers, experts, v, pin).unwrap();
                        c.set_pinned_stays(stay);
                        let mut slots: Vec<LayerSlots> = (0..layers).map(|_| LayerSlots::new(experts, sizes)).collect();
                        let mut sims: Vec<Sim> = (0..layers).map(|_| Sim::new(sizes, k)).collect();
                        let mut total = Moves::default();
                        for tok in &tr {
                            for (l, ids) in tok.iter().enumerate() {
                                let what = format!("k {k} {p:?} V {v} P {pin} stay {stay} layer {l}");
                                let was_pinned: Vec<u32> = ids.iter().copied().filter(|&e| c.tier(l, e) == Tier::Pinned).collect();
                                let (nv, hits0) = (c.counters()[l][2], c.counters()[l][1]);
                                let cold = ids.iter().filter(|&&e| c.tier(l, e) == Tier::Nvme).count();
                                let s = serve(&mut c, l, &mut slots[l], ids, k, &mut sims[l]).unwrap_or_else(|e| panic!("{what}: {e}"));
                                check(&c, l, &slots[l], &sims[l], &s, ids, cold, nv);
                                assert_eq!(s.moves.zero_copy, s.locs.iter().filter(|x| matches!(x.1, Loc::Pinned(_))).count() as u64, "{what}: zero-copy");
                                if stay {
                                    let hits = c.counters()[l][1] - hits0;
                                    let left: u64 = was_pinned.iter().filter(|&&e| c.tier(l, e) != Tier::Pinned).count() as u64;
                                    let up: u64 = was_pinned.iter().filter(|&&e| c.tier(l, e) == Tier::Vram).count() as u64;
                                    assert_eq!(s.moves.p2v, up, "{what}: an unselected pinned expert entered VRAM");
                                    assert_eq!(s.moves.pinned_to_stage, left, "{what}: H2D of pinned records = selected pinned ids that left pinned");
                                    if k == 1 {
                                        assert_eq!((s.moves.p2v, s.moves.pinned_to_stage, hits), (0, 0, was_pinned.len() as u64), "{what}");
                                        for &e in &was_pinned {
                                            assert!(matches!(s.locs.iter().find(|x| x.0 == e).unwrap().1, Loc::Pinned(_)), "{what}: pinned hit {e} not read in place");
                                        }
                                    }
                                }
                                total.add(&s.moves);
                            }
                        }
                        if stay {
                            if v > 0 && pin >= k && promoted > 0 {
                                assert!(total.p2v < promoted, "k {k} {p:?} V {v} P {pin}: {} pinned -> VRAM under zerocopy, {promoted} under promote", total.p2v);
                            }
                            reset_cache(&mut c, &mut slots, sizes).unwrap();
                            assert!(c.pinned_stays(), "k {k} {p:?} V {v} P {pin}: reset_cache dropped the option");
                        } else {
                            promoted = total.p2v;
                        }
                    }
                    if k == 1 && v > 0 && pin > 0 && p != (Policy::Clock { admit: Some(2) }) {
                        assert!(promoted > 0, "{p:?} V {v} P {pin}: promote never promoted a pinned hit (the test would prove nothing)");
                    }
                }
            }
        }
    }

    /// #188: the two switches. Unset = today (promote, no lane); `zerocopy` = stay; the lane
    /// implies stay; lane + an explicit `promote`, and any other value, refused by name; the
    /// lane refused on a write-combined pinned arena.
    #[test]
    fn the_pinned_switches_parse_and_refuse_by_name() {
        assert_eq!(pinned_use(None, None), Ok(PinnedUse::default()));
        assert_eq!(pinned_use(Some("promote"), Some("0")), Ok(PinnedUse::default()));
        assert_eq!(pinned_use(Some(""), Some("")), Ok(PinnedUse::default()));
        assert_eq!(pinned_use(Some("zerocopy"), None), Ok(PinnedUse { stay: true, cpu_lane: false }));
        assert_eq!(pinned_use(None, Some("1")), Ok(PinnedUse { stay: true, cpu_lane: true }));
        assert_eq!(pinned_use(Some("zerocopy"), Some("1")), Ok(PinnedUse { stay: true, cpu_lane: true }));
        let e = pinned_use(Some("promote"), Some("1")).unwrap_err();
        assert!(e.starts_with("CROW_GLM_CPU_LANE=1 reads the selected pinned experts where they lie"), "{e}");
        assert!(pinned_use(Some("zero-copy"), None).unwrap_err().starts_with("CROW_GLM_PINNED=\"zero-copy\""));
        assert!(pinned_use(None, Some("yes")).unwrap_err().starts_with("CROW_GLM_CPU_LANE=\"yes\""));
        assert!(lane_on_wc(PinnedUse { stay: true, cpu_lane: true }, true).unwrap_err().ends_with("set CROW_PINNED_ALLOC=host"));
        assert_eq!(lane_on_wc(PinnedUse { stay: true, cpu_lane: false }, true), Ok(()));
        assert_eq!(lane_on_wc(PinnedUse { stay: true, cpu_lane: true }, false), Ok(()));
    }

    /// #188: the lane's combos follow the pick order of the selection; a pinned location goes to
    /// the CPU with its host record, every other to the GPU with its table entry.
    #[test]
    fn lane_combos_follow_the_pick_order() {
        use crate::glm5_moe::lane::Combo;
        let sel = [7, 3, 250, 0];
        let locs = [(0u32, Loc::Vram(2)), (3, Loc::Pinned(5)), (7, Loc::Stage(1)), (250, Loc::Pinned(0))];
        let host = 0x1000 as *const u8;
        let (c, n) = lane_combos(&sel, &locs, |e, _| 100 + e as u64, |q| host.wrapping_add(q as usize * 10));
        assert_eq!(n, 2);
        assert_eq!(c, vec![Combo::Gpu(107), Combo::Cpu(host.wrapping_add(50)), Combo::Cpu(host), Combo::Gpu(100)]);
    }

    /// A synthetic glm5_next container (index v2, `glm5_next_text`): `experts` MUL1 records of
    /// 9,474,048 B for layer 3, each on a 4096-B file offset, random bytes. Removed on drop.
    pub(super) struct SynthGlm {
        dir: std::path::PathBuf,
        pub(super) path: String,
    }

    impl Drop for SynthGlm {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    pub(super) fn synth_glm(experts: u32) -> SynthGlm {
        use std::io::Write;
        const REC: u64 = 9_474_048;
        let dir = std::env::temp_dir().join(format!("crow-glm5-tiers-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("synth-glm.cnq");
        let lead = 4084u64; // 12 + 4084: the first record on a sector
        let mut tensors = vec![serde_json::json!({ "name": "model.language_model.layers.3.mlp.gate.weight", "section": "text",
            "dtype": "bf16", "offset": 0, "n_values": lead / 2, "shape": [2, lead / 4] })];
        for e in 0..experts {
            for (k, p) in ["gate", "up", "down"].into_iter().enumerate() {
                tensors.push(serde_json::json!({ "name": crate::nvme_source::glm5_expert_tensor_name(3, e, p), "section": "text", "dtype": "mul1",
                    "offset": lead + e as u64 * REC + k as u64 * (REC / 3), "n_values": 2048u64 * 4096, "shape": [2048, 4096] }));
            }
        }
        let tail = lead + experts as u64 * REC;
        tensors.push(serde_json::json!({ "name": "model.language_model.norm.weight", "section": "text", "dtype": "bf16", "offset": tail, "n_values": 2048, "shape": [2048] }));
        let sha = |s: &str| crate::cnq::sha256_hex(s.as_bytes());
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "blob_offset": 12, "recipe": "synthetic-glm5-tiers",
            "model": { "family": "Glm5Next", "model_type": "glm5_next_text", "config_json": "{}", "config_json_sha256": sha("{}"),
                "generation_config_json": "{}", "generation_config_json_sha256": sha("{}"),
                "source": { "repo": "crow-nest/synthetic-glm5-tiers", "revision": "175", "shards": [] }, "geo": {} },
            "tensors": tensors
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut f = std::io::BufWriter::with_capacity(1 << 20, std::fs::File::create(&path).unwrap());
        f.write_all(b"CNQ1\0\0\0\0\0\0\0\0").unwrap();
        let mut x = 0x1755_F491_4F6C_DD1Du64;
        let mut chunk = vec![0u8; 1 << 20];
        let mut left = tail + 4096;
        while left > 0 {
            for w in chunk.chunks_exact_mut(8) {
                x ^= x << 13;
                x ^= x >> 7;
                x ^= x << 17;
                w.copy_from_slice(&x.to_le_bytes());
            }
            let n = left.min(chunk.len() as u64) as usize;
            f.write_all(&chunk[..n]).unwrap();
            left -= n as u64;
        }
        f.write_all(&ib).unwrap();
        f.write_all(&(ib.len() as u64).to_le_bytes()).unwrap();
        f.flush().unwrap();
        SynthGlm { dir, path: path.to_str().unwrap().to_string() }
    }

    /// The device store on a synthetic container (one MoE layer, 16 experts of 9,474,048 B, 151
    /// MB): for every call of a routing trace, under every capacity from all-NVMe to all-cached,
    /// the bytes at each selected id's table entry (VRAM slot, pinned slot through its UVA
    /// address, or staging slot) are `Cnq::read_range` of that expert's record, and every other
    /// entry is 0. Proves the device moves (NVMe into landing and pinned, H2D, D2H, D2D) and the
    /// table, not the model. 54 s on this machine (2026-10-09), most of it NVMe reads and compares.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_every_table_entry_holds_its_record() {
        let s = synth_glm(16);
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk) = (4, 3, 16, 8);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let rb = spec.bytes as usize;
        let want: Vec<Vec<u8>> = (0..16).map(|e| cnq.read_range(&cnq.find(&crate::nvme_source::glm5_expert_tensor_name(3, e, "gate"), "text").clone(), 0, rb)).collect();
        let tr = trace(24, 1, 16, 8, 0x149);
        unsafe {
            let _ctx = cuda::Ctx::init();
            for (v, p) in [(0, 0), (0, 8), (1, 7), (3, 4), (6, 0), (4, 12), (16, 0)] {
                let sizes = TierSizes { vram: v, pinned: p };
                let mut t = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, 8).unwrap();
                for tok in &tr {
                    let sel: Vec<i32> = tok[0].iter().rev().map(|&e| e as i32).collect();
                    let (tb, served) = t.table_for(3, &sel).unwrap();
                    assert_eq!(served.moves.visits as usize, tok[0].len(), "V {v} P {p}: visits");
                    assert_eq!(served.moves.nvme_reads() as usize, served.nvme_reads, "V {v} P {p}: NVMe reads");
                    cuda::sync();
                    let table = cuda::dtoh_u64(tb, 16);
                    for e in 0..16u32 {
                        let picked = tok[0].contains(&e);
                        assert_eq!(table[e as usize] != 0, picked, "V {v} P {p}: table entry of expert {e}");
                        if picked {
                            let got: Vec<u8> = cuda::dtoh_t(table[e as usize], rb);
                            assert!(got == want[e as usize], "V {v} P {p}: expert {e} at {:?}: the bytes differ from read_range", served.locs.iter().find(|x| x.0 == e).unwrap().1);
                        }
                    }
                }
                let tot: [u64; 3] = t.cache.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                eprintln!("glm5_tiers synthetic V {v} P {p}: accesses {tot:?}, NVMe reads {}", t.nvme_reads);
                assert_eq!(t.moves[0].visits, tot.iter().sum::<u64>(), "V {v} P {p}: visits = accesses");
                assert_eq!(t.moves[0].nvme_reads(), t.nvme_reads, "V {v} P {p}: NVMe reads");
                // #187 --cold: after the reset the first call reads every selected record again
                t.reset_cache().unwrap();
                let sel: Vec<i32> = tr[0][0].iter().map(|&e| e as i32).collect();
                let (tb, served) = t.table_for(3, &sel).unwrap();
                cuda::sync();
                assert_eq!(served.nvme_reads, tr[0][0].len(), "V {v} P {p}: a reset cache reads the first selection from NVMe");
                let table = cuda::dtoh_u64(tb, 16);
                for &e in &tr[0][0] {
                    let got: Vec<u8> = cuda::dtoh_t(table[e as usize], rb);
                    assert!(got == want[e as usize], "V {v} P {p}: after reset_cache expert {e} differs from read_range");
                }
                t.free();
            }
        }
        drop(cnq);
    }

    /// #188 on the synthetic container (as `glm5_tiers_gpu_every_table_entry_holds_its_record`,
    /// with `CROW_PINNED_ALLOC=host` for this test so the CPU lane may read the arena): under
    /// `promote`, `zerocopy` and the CPU lane, at four capacities, every selected id's table
    /// entry holds its record; `zerocopy` serves at least as many visits zero-copy as `promote`
    /// and promotes fewer pinned hits; the lane posts the call's combos in pick order, every CPU
    /// combo a host pointer to the record's bytes, every GPU combo the id's table entry, and
    /// counts them as `cpu_lane` (zero-copy 0); `promote` and `zerocopy` post nothing.
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_zerocopy_and_cpu_lane_tables_hold_their_records() {
        use crate::glm5_moe::lane::{self, Combo};
        let s = synth_glm(16);
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk) = (4, 3, 16, 8);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let rb = spec.bytes as usize;
        let want: Vec<Vec<u8>> = (0..16).map(|e| cnq.read_range(&cnq.find(&crate::nvme_source::glm5_expert_tensor_name(3, e, "gate"), "text").clone(), 0, rb)).collect();
        let tr = trace(24, 1, 16, 8, 0x188);
        let old = std::env::var("CROW_PINNED_ALLOC").ok();
        std::env::set_var("CROW_PINNED_ALLOC", "host");
        unsafe {
            let _ctx = cuda::Ctx::init();
            for (v, p) in [(1, 7), (3, 4), (4, 12), (0, 16)] {
                let sizes = TierSizes { vram: v, pinned: p };
                let mut tot = Vec::new();
                for (name, pu) in [("promote", PinnedUse::default()), ("zerocopy", PinnedUse { stay: true, cpu_lane: false }), ("lane", PinnedUse { stay: true, cpu_lane: true })] {
                    let mut t = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, 8).unwrap();
                    t.set_pinned_use(pu).unwrap();
                    for tok in &tr {
                        let sel: Vec<i32> = tok[0].iter().rev().map(|&e| e as i32).collect();
                        let (tb, served) = t.table_for(3, &sel).unwrap();
                        let what = format!("{name} V {v} P {p}");
                        let pinned_locs = served.locs.iter().filter(|x| matches!(x.1, Loc::Pinned(_))).count() as u64;
                        let posted = lane::take(tb, 1, 8);
                        cuda::sync();
                        let table = cuda::dtoh_u64(tb, 16);
                        for &e in &tok[0] {
                            let got: Vec<u8> = cuda::dtoh_t(table[e as usize], rb);
                            assert!(got == want[e as usize], "{what}: expert {e}: the table entry's bytes differ from read_range");
                        }
                        if pu.cpu_lane {
                            assert_eq!((served.moves.cpu_lane, served.moves.zero_copy), (pinned_locs, 0), "{what}: lane counters");
                            match posted {
                                None => assert_eq!(pinned_locs, 0, "{what}: pinned ids but no lane post"),
                                Some(call) => {
                                    assert_eq!(call.combos.len(), 8);
                                    for (c, combo) in call.combos.iter().enumerate() {
                                        let e = sel[c] as usize;
                                        match *combo {
                                            Combo::Cpu(ptr) => assert!(std::slice::from_raw_parts(ptr, rb) == &want[e][..], "{what}: CPU combo {c} (expert {e}) reads other bytes"),
                                            Combo::Gpu(base) => assert_eq!(base, table[e], "{what}: GPU combo {c} (expert {e})"),
                                        }
                                    }
                                    assert_eq!(call.combos.iter().filter(|x| matches!(x, Combo::Cpu(_))).count() as u64, pinned_locs, "{what}: CPU combos");
                                }
                            }
                        } else {
                            assert!(posted.is_none(), "{what}: a post without the lane");
                            assert_eq!((served.moves.cpu_lane, served.moves.zero_copy), (0, pinned_locs), "{what}: counters");
                        }
                    }
                    let m = t.moves[0];
                    eprintln!("glm5_tiers #188 {name} V {v} P {p}: zero-copy {} cpu_lane {} p2v {} pinned_to_stage {} vram_to_pinned {} NVMe reads {}", m.zero_copy, m.cpu_lane, m.p2v, m.pinned_to_stage, m.vram_to_pinned, m.nvme_reads());
                    tot.push(m);
                    t.free();
                }
                let (pr, zc, ln) = (tot[0], tot[1], tot[2]);
                assert_eq!((zc.zero_copy, zc.p2v, zc.pinned_to_stage), (ln.cpu_lane, ln.p2v, ln.pinned_to_stage), "V {v} P {p}: the lane moves as zerocopy");
                if v > 0 && p >= 8 {
                    assert!(zc.zero_copy >= pr.zero_copy && zc.p2v < pr.p2v, "V {v} P {p}: zerocopy {zc:?} vs promote {pr:?}");
                }
            }
        }
        match old {
            Some(o) => std::env::set_var("CROW_PINNED_ALLOC", o),
            None => std::env::remove_var("CROW_PINNED_ALLOC"),
        }
        drop(cnq);
    }

    /// Plan step 16 abort criterion (A9): the cache size is invisible in the output. The real
    /// 3-bit container, the fixed prompt, 6 greedy ids, three cache sizes: the #159 plan's
    /// (this machine's free VRAM and pinned budget), V 1 + P 7 (every access moves records,
    /// three-way exchanges, pinned zero-copy) and V 0 + P 0 (every record read from NVMe into
    /// staging). Same ids and byte-identical logits at every generated position.
    #[test]
    #[ignore = "needs the GPU and the real 3-bit container (dense part ~6 GB VRAM, the plan's tiers up to 46 GiB pinned): cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_cache_size_is_invisible_in_the_logits() {
        let path = std::env::var("CROW_CNQ").unwrap_or_else(|_| crate::geo::from_engine_dir(gm::GLM5_MUL1K3_CNQ));
        let mut o = open_container(&path).unwrap();
        let prompt = fixed_prompt();
        let n = 6;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let free = cuda::free_vram_bytes();
            let budget = crate::manager::derive_host_pinned_budget(crate::geo::HOST_PINNED_CAP, &mut |s| eprintln!("{s}"));
            let (_, _, plan) = crate::manager::plan_glm5_next(&o.g, o.g.context_floor, free, budget, crate::geo::GLM5_NEXT_DENSE_BYTES, o.spec.bytes, crate::gen::pf_tg(), crate::gen::pf_async_on()).unwrap();
            let mut run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            let mut outs = Vec::new();
            for (v, p) in [(None, None), (Some(1), Some(7)), (Some(0), Some(0))] {
                let sizes = tier_sizes(&plan, v, p).unwrap();
                let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, 1, o.g.topk).unwrap();
                let t0 = std::time::Instant::now();
                let gen = run.generate(&mut o.cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap();
                let tot: [u64; 3] = tiers.cache.counters().iter().fold([0; 3], |a, x| [a[0] + x[0], a[1] + x[1], a[2] + x[2]]);
                eprintln!(
                    "glm5_tiers V {} P {}: ids {:?}, accesses [vram {} pinned {} nvme {}], NVMe reads {} ({:.2} GB), {:.1} s",
                    sizes.vram, sizes.pinned, gen.ids, tot[0], tot[1], tot[2], tiers.nvme_reads, tiers.nvme_bytes as f64 / 1e9, t0.elapsed().as_secs_f64()
                );
                tiers.free();
                outs.push((sizes, gen));
            }
            run.free();
            let (s0, g0) = &outs[0];
            for (s, gx) in &outs[1..] {
                assert_eq!(gx.ids, g0.ids, "token ids differ: V {} P {} vs V {} P {}", s.vram, s.pinned, s0.vram, s0.pinned);
                for (i, (a, b)) in gx.logits.iter().zip(&g0.logits).enumerate() {
                    let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    assert_eq!(diff, 0, "generated position {i}: {diff} logits differ in bits, V {} P {} vs V {} P {}", s.vram, s.pinned, s0.vram, s0.pinned);
                }
            }
        }
    }

    /// #187 (`glm5_run --reps`, `--cold`): repetitions on one loaded model are invisible in the
    /// output. The real 3-bit container, the fixed prompt, 6 greedy ids at the #159 plan's tiers:
    /// rep 1 (cold), rep 2 (warm), `reset_cache`, rep 3 (cold). Ids and logits bit-identical
    /// across the three; rep 3 moves row by row exactly as rep 1 (the same per-layer `Moves`).
    #[test]
    #[ignore = "needs the GPU and the real 3-bit container (as glm5_tiers_gpu_cache_size_is_invisible_in_the_logits): cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_reps_and_reset_are_invisible_in_the_logits() {
        let path = std::env::var("CROW_CNQ").unwrap_or_else(|_| crate::geo::from_engine_dir(gm::GLM5_MUL1K3_CNQ));
        let mut o = open_container(&path).unwrap();
        let prompt = fixed_prompt();
        let n = 6;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let free = cuda::free_vram_bytes();
            let budget = crate::manager::derive_host_pinned_budget(crate::geo::HOST_PINNED_CAP, &mut |s| eprintln!("{s}"));
            let (_, _, plan) = crate::manager::plan_glm5_next(&o.g, o.g.context_floor, free, budget, crate::geo::GLM5_NEXT_DENSE_BYTES, o.spec.bytes, crate::gen::pf_tg(), crate::gen::pf_async_on()).unwrap();
            let mut run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, prompt.len() + n, &mut |s| eprintln!("{s}"));
            let sizes = tier_sizes(&plan, None, None).unwrap();
            let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, 1, o.g.topk).unwrap();
            let mut reps = Vec::new();
            for rep in 0..3 {
                if rep == 2 {
                    tiers.reset_cache().unwrap();
                }
                let mut rows: Vec<Vec<Moves>> = Vec::new();
                let gen = run.generate(&mut o.cnq, &mut tiers, &prompt, n, true, &mut |r| rows.push(r.moves.clone())).unwrap();
                let reads: u64 = rows.iter().flatten().map(|m| m.nvme_reads()).sum();
                eprintln!("glm5_tiers rep {}: ids {:?}, NVMe reads {reads}", rep + 1, gen.ids);
                reps.push((gen, rows));
            }
            tiers.free();
            run.free();
            let (g0, r0) = &reps[0];
            for (i, (gx, _)) in reps.iter().enumerate().skip(1) {
                assert_eq!(gx.ids, g0.ids, "rep {} ids differ from rep 1", i + 1);
                for (j, (a, b)) in gx.logits.iter().zip(&g0.logits).enumerate() {
                    let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                    assert_eq!(diff, 0, "rep {} generated position {j}: {diff} logits differ in bits", i + 1);
                }
            }
            assert_eq!(&reps[2].1, r0, "rep 3 after reset_cache moves differently from rep 1");
        }
    }
}

// ---------------------------------------------------------------- #185 part 2: serve's row door

/// #185 part 2: the row door `glm5_engine::Glm5Device` drives for `bin/serve` (prefill a chunk
/// row by row, decode one step, the logits row, the KDA states its prefix cache snapshots).
/// [`Glm5Run::generate`] runs its rows through [`Glm5Run::row`] when the lookahead is off (#189:
/// one body); the GPU test `glm5_engine::tests::glm5_engine_gpu_serve_rows_are_glm5_run_rows`
/// holds serve's rows equal to `generate`'s.
impl Glm5Run {
    /// One row: `tok` at position `pos` through every layer with the experts from `tiers`; with
    /// `head` the head runs and its greedy id comes back (its logits stay in
    /// [`Glm5Run::logits_dev`] until the next head). The KDA states and MLA caches advance as in
    /// `generate`; nothing is reset here. `CROW_GLM_FLAGS` applies (through the pass);
    /// `CROW_GLM_LOOKAHEAD` does not (the caller needs the id before the next row).
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn row(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String> {
        if pos >= self.cap {
            return Err(format!("glm5_run: row {pos} is outside the caches of {} rows", self.cap));
        }
        if !(0..self.g.vocab as i64).contains(&tok) {
            return Err(format!("glm5_run: token id {tok} outside the vocab of {}", self.g.vocab));
        }
        self.settle_ahead(tiers)?;
        if self.pass.ctl.is_some() {
            return self.row_ctl(cnq, tiers, tok, pos, head);
        }
        self.embed(cnq, tok);
        self.layers(tiers, pos, &mut |_| Ok(()))?;
        if !head {
            cuda::sync();
            tiers.settle()?;
            return Ok(None);
        }
        self.head_row()?;
        cuda::sync();
        tiers.settle()?;
        Ok(Some(cuda::dtoh_i32(self.next, 1)[0] as i64))
    }

    /// the `[vocab]` f32 logits of the last head
    pub fn logits_dev(&self) -> Dev {
        self.logits
    }

    /// the KDA state of every KDA layer, in layer order (what a prefix snapshot copies)
    pub fn kda_states(&self) -> impl Iterator<Item = &KdaState> {
        self.kda.iter().flatten()
    }
}


// ---------------------------------------------------------------- CROW_GLM_CONTROLLER / CROW_GLM_LA

/// `CROW_GLM_LA`: the row enqueued ahead by [`Glm5Run::decode_la`]: its position and the id it
/// runs on
#[derive(Clone, Copy, Debug)]
struct Ahead {
    pos: usize,
    tok: i64,
}

/// `CROW_GLM_CONTROLLER`: the controller thread's job of one row: its `n` requests read from the
/// ring in order, each served through the stager with the reply written behind the layer's moves
/// (`ExpertTiers::table_reply`), the next layer's guess handed to the prefetch first. The row's
/// report (the store's counters around the job; clock, position and id are the caller's). On a
/// failure the stager stream drains and every device wait is released, so the row runs out.
fn ctl_job(t: &mut ExpertTiers, mut rd: glm5_flags::RingReader, n: usize, score: std::sync::Arc<std::sync::Mutex<glm5_flags::GuessScore>>) -> Result<TokenReport, String> {
    let base = RowBase::of(t);
    for _ in 0..n {
        let rq = match rd.next() {
            Ok(r) => r,
            Err(e) => {
                // SAFETY: the stager stream of this store, on the thread that queues on it
                unsafe { t.stager_idle() };
                rd.release_all();
                t.dev_lane_release();
                return Err(e);
            }
        };
        if let Ok(mut s) = score.lock() {
            s.see(rq.layer, &rq.ids, rq.guess.as_ref());
        }
        glm5_flags::post_hint(rq.guess.clone());
        // SAFETY: the host side of the protocol: the main thread does not touch the store while
        // a job runs; the device published this layer's request after the experts of its
        // previous call ran (see `Stager`)
        if let Err(e) = unsafe { t.table_reply(rq.layer, &rq.ids, rd.reply_dev, rq.seq) } {
            unsafe { t.stager_idle() };
            rd.release_all();
            t.dev_lane_release();
            return Err(format!("{}: layer {}: {e}", glm5_flags::ENV_CONTROLLER, rq.layer));
        }
    }
    Ok(base.report(t, 0, false, None, std::time::Instant::now()))
}

impl Glm5Run {
    /// `CROW_GLM_CONTROLLER`: what a controlled row needs, refused by name
    fn ctl_check(&self, tiers: &ExpertTiers) -> Result<(), String> {
        let c = glm5_flags::ENV_CONTROLLER;
        if self.graph.is_some() {
            return Err(format!("{c}=1 and {}=1: a controlled row has no host hand-off to cut its graphs at; turn one of them off", glm5_graph::ENV));
        }
        if !tiers.stager_on() {
            return Err(format!("{c}=1 needs {STAGER_ENV}=1: the controller serves the layers through the stager"));
        }
        // the CPU lane under the controller: its device side (`DevLane`, made with lane + stager)
        if tiers.pinned_use.cpu_lane && tiers.dev_lane().is_none() {
            return Err(format!("{c}=1 and {CPU_LANE_ENV}: the lane's device side is missing (ExpertTiers::set_stager)"));
        }
        if self.spec.is_some() {
            return Err(format!("{c}=1 and {}: the verify calls hand their routing to the host", crate::glm5_mtp::MTP_ENV));
        }
        match (self.pass.ctl.as_ref(), self.worker.as_ref()) {
            (Some(ctl), Some(_)) if !ctl.poisoned => Ok(()),
            (Some(_), Some(_)) => Err(format!("{c}: an earlier controlled row failed; put the switches in force again")),
            _ => Err(format!("{c}: the controller is not set up (Glm5Run::set_switches)")),
        }
    }

    /// Queue row `pos` (its input already in `x`) through the controller: the row's job to the
    /// controller thread, then every layer's launches; no host wait. A failure here abandons
    /// the job (it releases the device's waits) and drains the row.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model and is not touched by the
    /// caller until the job's result was taken.
    unsafe fn enqueue_ctl(&mut self, tiers: &mut ExpertTiers, pos: usize) -> Result<(), String> {
        let (g, first) = (self.g, tiers.first_moe);
        let n = moe_layers(&g);
        // the arena's decode phase begins here, not on the controller thread (`decode_ready`)
        tiers.decode_ready();
        let ctl = self.pass.ctl.as_mut().expect("glm5_run: a controlled row without the controller");
        ctl.tables = (0..g.layers).map(|l| l.checked_sub(first).and_then(|i| tiers.tables().get(i).copied()).unwrap_or(0)).collect();
        // the CPU lane: a fresh controller counts its requests from 1 again, so does the lane flag
        if ctl.fresh() {
            tiers.dev_lane_reset();
        }
        ctl.lane = tiers.dev_lane();
        let rd = ctl.reader(n);
        let score = ctl.score.clone();
        let tp = glm5_flags::SendPtr(tiers as *mut ExpertTiers);
        self.worker.as_mut().expect("glm5_run: a controlled row without the controller thread").send(Box::new(move || {
            let tp = tp;
            // SAFETY: see `ctl_job`; the caller keeps off the store until this job's result
            ctl_job(unsafe { &mut *tp.0 }, rd, n, score)
        }));
        let ctl = self.pass.ctl.as_mut().expect("controller");
        ctl.active = true;
        let r = self.layers_ctl(pos);
        let ctl = self.pass.ctl.as_mut().expect("controller");
        ctl.active = false;
        if r.is_err() {
            ctl.cancel();
            ctl.poisoned = true;
            self.drain_ctl();
        }
        r
    }

    /// every layer of row `pos` with the controller active (the expert hook is never asked)
    unsafe fn layers_ctl(&mut self, pos: usize) -> Result<(), String> {
        for l in 0..self.g.layers {
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            let mut hook = |layer: usize, _: &[i32]| -> Result<Dev, String> { Err(format!("layer {layer}: a controlled row asked the host for its experts")) };
            let r = self.pass.call_with_experts(&self.layers[l], self.x, pos, 1, true, &mut hook);
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            r.map_err(|e| format!("glm5_run: row {pos} layer {l}: {e}"))?;
        }
        Ok(())
    }

    /// the oldest job's result, checked against the device's timeout word; a failure poisons
    /// the controller
    fn ctl_result(&mut self, r: Result<TokenReport, String>) -> Result<TokenReport, String> {
        let ctl = self.pass.ctl.as_mut().expect("controller");
        let q = ctl.timed_out();
        if r.is_err() || q != 0 {
            ctl.poisoned = true;
        }
        let r = r?;
        if q != 0 {
            return Err(format!(
                "{}: the device's wait for request {q} gave up after {} ms (bounded under the WDDM TDR); that layer's experts read a stale table",
                glm5_flags::ENV_CONTROLLER,
                glm5_flags::CTL_WAIT_NS / 1_000_000
            ));
        }
        Ok(r)
    }

    /// after a row's launches (and its head): the stream drained, the row's job result
    unsafe fn finish_ctl(&mut self) -> Result<TokenReport, String> {
        cuda::sync();
        let r = self.worker.as_mut().expect("controller thread").wait();
        self.ctl_result(r)
    }

    /// after a failure: the stream drained and every pending job's result dropped
    unsafe fn drain_ctl(&mut self) {
        cuda::sync();
        if let Some(w) = self.worker.as_mut() {
            while w.pending > 0 {
                let _ = w.wait();
            }
        }
        self.ahead = None;
    }

    /// [`Glm5Run::row`] through the controller: the row enqueued whole, the head, one stream
    /// sync at the end
    unsafe fn row_ctl(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String> {
        self.ctl_check(tiers)?;
        self.embed(cnq, tok);
        self.enqueue_ctl(tiers, pos)?;
        if head {
            if let Err(e) = self.head_row() {
                self.drain_ctl();
                return Err(e);
            }
        }
        self.finish_ctl()?;
        tiers.settle()?;
        Ok(head.then(|| cuda::dtoh_i32(self.next, 1)[0] as i64))
    }

    /// `CROW_GLM_LA`: the KDA states (state and conv window of every KDA layer) into their
    /// backups, queued
    unsafe fn backup_kda(&mut self) {
        let kd = KdaDims::of(&self.g);
        let (sb, cb) = (kd.state_floats() * 4, kd.conv_floats() * 4);
        if self.kda_bak.is_empty() {
            self.kda_bak = self.kda.iter().flatten().map(|_| (cuda::alloc_named("glm5 LA KDA state backup", sb), cuda::alloc_named("glm5 LA KDA conv backup", cb))).collect();
        }
        let s = cuda::cur_stream();
        for (k, b) in self.kda.iter().flatten().zip(&self.kda_bak) {
            cuda::ck(sys::cuMemcpyDtoDAsync_v2(b.0, k.s, sb, s));
            cuda::ck(sys::cuMemcpyDtoDAsync_v2(b.1, k.conv, cb, s));
        }
    }

    /// `CROW_GLM_LA`: the backups into the KDA states, queued
    unsafe fn restore_kda(&mut self) {
        let kd = KdaDims::of(&self.g);
        let (sb, cb) = (kd.state_floats() * 4, kd.conv_floats() * 4);
        let s = cuda::cur_stream();
        for (k, b) in self.kda.iter().flatten().zip(&self.kda_bak) {
            cuda::ck(sys::cuMemcpyDtoDAsync_v2(k.s, b.0, sb, s));
            cuda::ck(sys::cuMemcpyDtoDAsync_v2(k.conv, b.1, cb, s));
        }
    }

    /// `CROW_GLM_LA`, serve's door: decode row `pos` on `tok` and its head; the greedy id. Before
    /// the host reads it, row `pos + 1` is enqueued on the device's id (the KDA states of its start
    /// kept). When the next call asks for that row with that id, its launches are already queued;
    /// any other call first drops it ([`Glm5Run::settle_ahead`]). Not with MTP.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn decode_la(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, tok: i64, pos: usize) -> Result<i64, String> {
        if pos >= self.cap {
            return Err(format!("glm5_run: row {pos} is outside the caches of {} rows", self.cap));
        }
        if !(0..self.g.vocab as i64).contains(&tok) {
            return Err(format!("glm5_run: token id {tok} outside the vocab of {}", self.g.vocab));
        }
        self.ctl_check(tiers)?;
        if self.feed.is_none() || self.readback.is_none() {
            return Err(format!("{}: the lookahead is not set up (Glm5Run::set_switches)", glm5_flags::ENV_LA));
        }
        match self.ahead.take() {
            Some(a) if a.pos == pos && a.tok == tok => {}
            Some(_) => {
                self.drop_ahead(tiers)?;
                self.embed(cnq, tok);
                self.enqueue_ctl(tiers, pos)?;
            }
            None => {
                self.embed(cnq, tok);
                self.enqueue_ctl(tiers, pos)?;
            }
        }
        if let Err(e) = self.head_row() {
            self.drain_ctl();
            return Err(e);
        }
        self.readback.as_ref().expect("readback").enqueue_marked(self.next, None);
        let ahead = pos + 1 < self.cap;
        if ahead {
            self.backup_kda();
            self.feed.as_ref().expect("feed").gather(self.next, self.x);
            self.enqueue_ctl(tiers, pos + 1)?;
        }
        let rb = self.readback.as_ref().expect("readback");
        rb.wait_marked();
        let id = rb.id() as i64;
        let r = self.worker.as_mut().expect("controller thread").wait();
        if let Err(e) = self.ctl_result(r) {
            self.drain_ctl();
            return Err(e);
        }
        if !(0..self.g.vocab as i64).contains(&id) {
            self.drain_ctl();
            return Err(format!("glm5_run: row {pos}: the greedy id {id} is outside the vocab of {}", self.g.vocab));
        }
        if ahead {
            self.ahead = Some(Ahead { pos: pos + 1, tok: id });
        }
        Ok(id)
    }

    /// `CROW_GLM_LA`: the row enqueued ahead runs out (its job served), then the KDA states of its
    /// start come back; its MLA rows are left to be overwritten
    unsafe fn drop_ahead(&mut self, tiers: &mut ExpertTiers) -> Result<(), String> {
        cuda::sync();
        let r = self.worker.as_mut().expect("controller thread").wait();
        let r = self.ctl_result(r);
        self.restore_kda();
        cuda::sync();
        r?;
        tiers.settle()
    }

    /// `CROW_GLM_LA`: drop the row enqueued ahead, if any (a no-op otherwise). Everything that
    /// reads or replaces the sequence's state calls this first.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn settle_ahead(&mut self, tiers: &mut ExpertTiers) -> Result<(), String> {
        match self.ahead.take() {
            Some(_) => self.drop_ahead(tiers),
            None => Ok(()),
        }
    }

    /// `CROW_GLM_LA`: a row is enqueued ahead
    pub fn has_ahead(&self) -> bool {
        self.ahead.is_some()
    }

    /// [`Glm5Run::generate`] with `CROW_GLM_LA` (the reference's `iterate_gen_la`): every row
    /// through the controller; after a head that is not the last row's, the next row is enqueued
    /// on the device's id before the host waits for that id (an event behind its readback). No
    /// row is launched past the last one, so nothing is dropped. Reports as without it.
    unsafe fn generate_la(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], n: usize, keep_logits: bool, report: &mut dyn FnMut(&TokenReport)) -> Result<Generated, String> {
        self.ctl_check(tiers)?;
        if self.feed.is_none() || self.readback.is_none() {
            return Err(format!("{}: the lookahead is not set up (Glm5Run::set_switches)", glm5_flags::ENV_LA));
        }
        let rows = prompt.len() + n - 1;
        let mut out = Generated::default();
        let start = if self.prompt_chunk > 1 {
            let id = self.prefill(cnq, tiers, prompt, 0, report)?;
            if keep_logits {
                out.logits.push(cuda::dtoh(self.logits, self.g.vocab));
            }
            out.ids.push(id);
            prompt.len()
        } else {
            0
        };
        let r = self.la_rows(cnq, tiers, prompt, start, rows, keep_logits, &mut out, report);
        if r.is_err() {
            self.drain_ctl();
        }
        r.map(|_| out)
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn la_rows(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], start: usize, rows: usize, keep_logits: bool, out: &mut Generated, report: &mut dyn FnMut(&TokenReport)) -> Result<(), String> {
        let (pn, vocab) = (prompt.len(), self.g.vocab);
        let mut ahead: Option<std::time::Instant> = None;
        for pos in start..rows {
            let t0 = match ahead.take() {
                Some(t) => t,
                None => {
                    let t = std::time::Instant::now();
                    let tok = if pos < pn { prompt[pos] } else { *out.ids.last().expect("a decode row after an id") };
                    self.embed(cnq, tok);
                    self.enqueue_ctl(tiers, pos)?;
                    t
                }
            };
            if pos + 1 < pn {
                let mut rep = self.finish_ctl()?;
                (rep.pos, rep.prompt, rep.next, rep.secs) = (pos, true, None, t0.elapsed().as_secs_f64());
                report(&rep);
                continue;
            }
            self.head_row()?;
            self.readback.as_ref().expect("readback").enqueue_marked(self.next, keep_logits.then_some(self.logits));
            if pos + 1 < rows {
                self.feed.as_ref().expect("feed").gather(self.next, self.x);
                let t1 = std::time::Instant::now();
                self.enqueue_ctl(tiers, pos + 1)?;
                ahead = Some(t1);
            }
            let rb = self.readback.as_ref().expect("readback");
            rb.wait_marked();
            let id = rb.id() as i64;
            let logits = keep_logits.then(|| rb.logits());
            let r = self.worker.as_mut().expect("controller thread").wait();
            let mut rep = self.ctl_result(r)?;
            if !(0..vocab as i64).contains(&id) {
                return Err(format!("glm5_run: row {pos}: the greedy id {id} is outside the vocab of {vocab}"));
            }
            if let Some(l) = logits {
                out.logits.push(l);
            }
            out.ids.push(id);
            (rep.pos, rep.prompt, rep.next, rep.secs) = (pos, pos < pn, Some(id), t0.elapsed().as_secs_f64());
            report(&rep);
        }
        cuda::sync();
        tiers.settle()
    }
}

// ---------------------------------------------------------------- #186: the prompt phase in calls

/// the per-call hook of [`Glm5Run::prefill_with`]: `(run, container, r0, t)`
pub(crate) type PromptRows<'a> = dyn FnMut(&mut Glm5Run, &mut Cnq, usize, usize) -> Result<(), String> + 'a;

/// #186: the prompt phase of the glm5_next path in prompt calls of up to `prompt_chunk` rows.
impl Glm5Run {
    /// prompt rows per prompt call (1 = every prompt row one decode call)
    pub fn prompt_chunk(&self) -> usize {
        self.prompt_chunk
    }

    /// Prompt rows per prompt call from now on, 1 ..= the `max_t` the pass was built with at
    /// `load` (`CROW_CHUNK`); a larger ask is refused by name.
    pub fn set_prompt_chunk(&mut self, chunk: usize) -> Result<(), String> {
        let most = self.borrow.map_or(self.pass.max_t, |b| b.1);
        if chunk == 0 || chunk > most {
            return Err(format!("glm5_run: a prompt chunk of {chunk} rows, the pass holds calls of 1 ..= {most} rows (CROW_CHUNK at load)"));
        }
        self.prompt_chunk = chunk;
        Ok(())
    }

    /// The prompt rows `ids` at `pos0 ..` through every layer with the experts from `tiers`, the
    /// head on the last: its greedy id (the logits stay in [`Glm5Run::logits_dev`]). Prompt chunk
    /// 1: every row is [`Glm5Run::row`], reported one by one. Above 1: the rows run in the calls
    /// of [`prompt_calls`], each call one prompt call per layer (KDA's chunk path, MLA's multi-row
    /// path) with one routing sync per MoE layer and the selection served through
    /// [`ExpertTiers::tables_for_chunk`]; one report per call (`rows` = its rows, `pos` = its last
    /// row). The KDA states and MLA caches continue from what the rows before `pos0` left; nothing
    /// is reset here.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn prefill(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, ids: &[i64], pos0: usize, report: &mut dyn FnMut(&TokenReport)) -> Result<i64, String> {
        self.prefill_with(cnq, tiers, ids, pos0, report, &mut |_, _, _, _| Ok(()))
    }

    /// [`Glm5Run::prefill`], and after every prompt call (prompt chunk 1: every row) `rows`
    /// sees the run, the container and the call's rows `(r0, t)` (indices into `ids`): their
    /// final residuals are the first `t` rows of the residual `x` (`[t][streams][H]`), queued,
    /// before the head of the last call runs (#192: the MTP block takes each row's head-norm row).
    ///
    /// # Safety
    /// As [`Glm5Run::prefill`]; `rows` only queues reads of `x` and its own buffers.
    pub(crate) unsafe fn prefill_with(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, ids: &[i64], pos0: usize, report: &mut dyn FnMut(&TokenReport), rows: &mut PromptRows) -> Result<i64, String> {
        if ids.is_empty() || pos0 + ids.len() > self.cap {
            return Err(format!("glm5_run: a prompt of {} rows at {pos0}, the caches hold {}", ids.len(), self.cap));
        }
        if let Some(bad) = ids.iter().find(|&&t| !(0..self.g.vocab as i64).contains(&t)) {
            return Err(format!("glm5_run: token id {bad} outside the vocab of {}", self.g.vocab));
        }
        let n = ids.len();
        if self.prompt_chunk <= 1 {
            let mut last = None;
            for (i, &tok) in ids.iter().enumerate() {
                let t0 = std::time::Instant::now();
                let base = RowBase::of(tiers);
                last = self.row(cnq, tiers, tok, pos0 + i, i + 1 == n)?;
                rows(self, cnq, i, 1)?;
                report(&base.report(tiers, pos0 + i, true, last, t0));
            }
            return last.ok_or_else(|| "glm5_run: the last prompt row gave no id".to_string());
        }
        // CROW_GLM_ARENA elastic: borrow the prompt scratch, give it back whatever happens
        let borrowed = self.borrow.is_some_and(|b| self.prompt_chunk > b.0);
        if borrowed {
            self.borrow_scratch(tiers)?;
        }
        let r = self.prompt_calls_with(cnq, tiers, ids, pos0, report, rows);
        if borrowed {
            self.return_scratch();
        }
        // the staged forward ends and the elastic part grows back here, on the host thread
        cuda::sync();
        tiers.decode_ready();
        r
    }

    /// `CROW_GLM_ARENA` elastic: the elastic chunks handed back, the pass and the residual at the
    /// prompt's rows; captured row graphs dropped (they hold the old buffers)
    ///
    /// # Safety
    /// As [`Glm5Run::prefill`].
    unsafe fn borrow_scratch(&mut self, tiers: &mut ExpertTiers) -> Result<(), String> {
        let (_, p) = self.borrow.expect("borrow_scratch without the borrow");
        tiers.elastic_hand_back()?;
        self.resize_rows(p);
        Ok(())
    }

    /// the prompt's scratch back to the decode rows; the elastic part grows back at the next
    /// decode call
    ///
    /// # Safety
    /// As [`Glm5Run::prefill`].
    unsafe fn return_scratch(&mut self) {
        let (d, _) = self.borrow.expect("return_scratch without the borrow");
        self.resize_rows(d);
    }

    unsafe fn resize_rows(&mut self, t: usize) {
        if self.pass.max_t == t {
            return;
        }
        self.pass.set_max_t(t);
        cuda::free_dev(&mut self.x);
        self.x = cuda::alloc_named("glm5_run residual", t * self.g.hc_streams * self.g.hidden * 4);
        if self.graph.is_some() {
            self.set_graph(false);
            self.set_graph(true);
        }
    }

    /// rows the pass holds now (`CROW_GLM_ARENA` elastic: the decode rows between prompts)
    pub fn rows_held(&self) -> usize {
        self.pass.max_t
    }

    /// the prompt calls of [`Glm5Run::prefill_with`] above prompt chunk 1
    ///
    /// # Safety
    /// As [`Glm5Run::prefill_with`].
    unsafe fn prompt_calls_with(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, ids: &[i64], pos0: usize, report: &mut dyn FnMut(&TokenReport), rows: &mut PromptRows) -> Result<i64, String> {
        let n = ids.len();
        let (g, h) = (self.g, self.g.hidden);
        let row = g.hc_streams * h;
        let mut next = None;
        for (r0, t) in prompt_calls(n, self.prompt_chunk) {
            let t0 = std::time::Instant::now();
            let base = RowBase::of(tiers);
            let e = gm::embed_rows(cnq, &g, &ids[r0..r0 + t]);
            cuda::to_f32_into(self.x, &gm::trunk_input(&e, h, g.hc_streams));
            self.layers_chunk(tiers, pos0 + r0, t)?;
            rows(self, cnq, r0, t)?;
            if r0 + t == n {
                // the head on the call's last row only
                gm::run_head(&self.pass.kn, &self.head, &self.hw, self.x + ((t - 1) * row * 4) as u64, self.normed, self.logits, self.next, 1);
                cuda::sync();
                let id = cuda::dtoh_i32(self.next, 1)[0] as i64;
                if !(0..g.vocab as i64).contains(&id) {
                    return Err(format!("glm5_run: row {}: the greedy id {id} is outside the vocab of {}", pos0 + n - 1, g.vocab));
                }
                next = Some(id);
            } else {
                cuda::sync();
            }
            let mut r = base.report(tiers, pos0 + r0 + t - 1, true, next, t0);
            r.rows = t;
            report(&r);
        }
        next.ok_or_else(|| "glm5_run: the last prompt call gave no id".to_string())
    }

    /// Every layer of the prompt call `pos0 .. pos0 + t` on the first `t` rows of `x`
    /// (`Glm5Pass::call_with_expert_batches`), the experts from `tiers` per row sub-batch.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model; `t <= max_t`.
    unsafe fn layers_chunk(&mut self, tiers: &mut ExpertTiers, pos0: usize, t: usize) -> Result<(), String> {
        for l in 0..self.g.layers {
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            let mut hook = |layer: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>| tiers.tables_for_chunk(layer, sel, run);
            let r = self.pass.call_with_expert_batches(&self.layers[l], self.x, pos0, t, &mut hook);
            // the layer's own state goes back even when the call failed
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            r.map_err(|e| format!("glm5_run: prompt rows {pos0}..{} layer {l}: {e}", pos0 + t))?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests_186 {
    //! #186 on the GPU with synthetic weights (no container of record): the planner's prompt-chunk
    //! bytes against the allocations they book, and a synthetic glm5_next model (the layer shapes
    //! of record, 4 layers) run with its prompt in prompt calls against the row-by-row path, three
    //! tier sizes (one forcing row sub-batches) and a layer-at-a-time chain with every record in
    //! VRAM. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_tiers_gpu_186 -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::geo::{ExpertCodec, ExpertRecordSpec};
    use crate::glm5_flags::tests::synth_model;
    use crate::glm5_moe::{GpuFfnPlan, GpuMoeGroupedPlan};

    /// `manager::glm5_chunk_scratch_bytes` is what a pass of `max_t = chunk`, its FFN plans of
    /// every booked call size and the residual's extra rows register as engine allocations, above
    /// a one-row pass: within 64 KiB (the parameter arrays the formula leaves out), GLM-5.3-Flash
    /// shapes, chunks 2 / 16 / 32, caches of 4,096 and 200,000 rows.
    #[test]
    #[ignore = "needs the GPU (up to about 1 GB VRAM): cargo test --release --lib glm5_tiers_gpu_186 -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_186_chunk_bytes_are_what_the_prompt_phase_allocates() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let moe = MoeGeo::new(&g, ExpertRecordSpec::new(ExpertCodec::Mul1, crate::cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
        let row = g.hc_streams * g.hidden;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mk = crate::kernels::mul1::Kernels::new();
            for cap in [4096usize, 200_000] {
                let held = |chunk: usize| -> u64 {
                    let before = cuda::live_dev().1;
                    let mut pass = Glm5Pass::new(&g, moe, chunk, cap);
                    let mut x = cuda::alloc_named("test residual", chunk * row * 4);
                    // the prompt call's one expert-major MoE plan of `chunk` rows, the dense FFN per call size
                    let mut plans: Vec<GpuMoeGroupedPlan> = if chunk > 1 { vec![GpuMoeGroupedPlan::new(&moe, chunk, &mk)] } else { Vec::new() };
                    let mut dense: Vec<GpuFfnPlan> = Vec::new();
                    for t in crate::manager::glm5_prompt_call_sizes(chunk) {
                        dense.push(GpuFfnPlan::new(g.hidden, g.dense_inter, t, g.swiglu_limit as f32));
                    }
                    let b = cuda::live_dev().1 - before;
                    for p in plans.iter_mut() {
                        p.free();
                    }
                    for p in dense.iter_mut() {
                        p.free();
                    }
                    cuda::free_dev(&mut x);
                    pass.free();
                    b
                };
                let one = held(1);
                for chunk in [2usize, 16, 32] {
                    let got = held(chunk) - one;
                    let want = crate::manager::glm5_chunk_scratch_bytes(&g, chunk, cap);
                    eprintln!("glm5_tiers #186 cap {cap} chunk {chunk}: allocated {got} B above a one-row pass, the plan books {want} B");
                    assert!(got.abs_diff(want) <= 64 << 10, "cap {cap} chunk {chunk}: allocated {got} B, booked {want} B");
                }
            }
        }
    }

    /// KL(p || q) in nats of two logit rows (f64 softmax)
    fn kl(p: &[f32], q: &[f32]) -> f64 {
        let ls = |v: &[f32]| -> Vec<f64> {
            let m = v.iter().fold(f64::NEG_INFINITY, |a, &x| a.max(x as f64));
            let z: f64 = v.iter().map(|&x| (x as f64 - m).exp()).sum();
            v.iter().map(|&x| x as f64 - m - z.ln()).collect()
        };
        let (a, b) = (ls(p), ls(q));
        a.iter().zip(&b).map(|(x, y)| x.exp() * (x - y)).sum()
    }

    fn bits_differ(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<usize> {
        a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect()
    }

    /// A synthetic glm5_next model (glm5_flags' `synth_model`: layers 0-2 KDA + dense SwiGLU, layer
    /// 3 MLA/DSA + MoE with 16 MUL1 experts, top-8, vocab 2048), a 37-id prompt, 5 greedy ids:
    /// - chunk 1 on a pass built for chunk 16 gives the bits of a pass built for chunk 1 (the
    ///   row-by-row path is unchanged by a larger pass);
    /// - chunk 16 (prompt calls 16 + 16 + 5) at V 3 + P 4, at V 0 + P 0 with a prefill
    ///   staging set of 8 slots (every call split into row sub-batches) and at V 16 + P 0: the
    ///   same ids and bit-identical logits (tiers and sub-batches invisible); one report per
    ///   prompt call with its rows, one routing sync per MoE layer per call, more sub-batches
    ///   than syncs only where the staging set forces them; the decode switches (flags +
    ///   lookahead, CUDA graphs) after the chunked prompt give the same bits;
    /// - a layer-at-a-time chain with every record in VRAM (`load_layer`, `Glm5Pass::call`), the
    ///   same call split, teacher-forced on the chunked run's ids: bit-identical head logits on
    ///   every generated row (the chunked tiered path computes what the plain pass computes);
    /// - chunk 16 vs chunk 1: the same ids, mean KL(row by row || chunked) over the generated rows
    ///   <= 0.073 (the KDA chunk path and MLA's split count sum in another order: no bit identity).
    #[test]
    #[ignore = "needs the GPU (about 1.5 GB VRAM, a 0.9 GB synthetic container in the temp dir): cargo test --release --lib glm5_tiers_gpu_186 -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_186_prompt_calls_are_the_rows_tiers_and_chain() {
        const REC: u64 = 9_474_048;
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (4, 3, 16, 8, 2048);
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt: Vec<i64> = (0..37).map(|i| (i * 131 + 7) % 2048).collect();
        let (n, chunk) = (5usize, 16usize);
        let cap = prompt.len() + n;
        let old = std::env::var("CROW_CHUNK").ok();
        let gen_with = |run: &mut Glm5Run, cnq: &mut Cnq, sizes: TierSizes, pf: Option<usize>| -> (Generated, Vec<TokenReport>) {
            unsafe {
                let mut tiers = ExpertTiers::new(cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
                if let Some(slots) = pf {
                    tiers.alloc_prefill_stage(slots).unwrap();
                }
                let mut reps = Vec::new();
                let out = run.generate(cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                tiers.free();
                (out, reps)
            }
        };
        unsafe {
            let _ctx = cuda::Ctx::init();
            // the path of record: a pass of one row
            std::env::remove_var("CROW_CHUNK");
            let mut run1 = Glm5Run::load(&mut cnq, &g, &moe, cap, &mut |s| eprintln!("{s}"));
            assert_eq!(run1.prompt_chunk(), 1);
            let (a0, _) = gen_with(&mut run1, &mut cnq, TierSizes { vram: 3, pinned: 4 }, None);
            run1.free();
            // a pass of `chunk` rows
            std::env::set_var("CROW_CHUNK", chunk.to_string());
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, cap, &mut |s| eprintln!("{s}"));
            match &old {
                Some(o) => std::env::set_var("CROW_CHUNK", o),
                None => std::env::remove_var("CROW_CHUNK"),
            }
            assert_eq!(run.prompt_chunk(), chunk);
            assert!(run.set_prompt_chunk(chunk + 1).is_err());
            run.set_prompt_chunk(1).unwrap();
            let (a, ra) = gen_with(&mut run, &mut cnq, TierSizes { vram: 3, pinned: 4 }, None);
            run.set_prompt_chunk(chunk).unwrap();
            let arms = [
                ("V 3 + P 4", TierSizes { vram: 3, pinned: 4 }, None),
                ("V 0 + P 0, 8 prefill slots", TierSizes { vram: 0, pinned: 0 }, Some(8)),
                ("V 16 + P 0", TierSizes { vram: 16, pinned: 0 }, None),
                // two staging halves of 8 slots: the sub-batches alternate halves on the copy stream
                ("V 0 + P 0, 16 prefill slots", TierSizes { vram: 0, pinned: 0 }, Some(16)),
            ];
            let mut chunked = Vec::new();
            for (name, sizes, pf) in arms {
                let (out, reps) = gen_with(&mut run, &mut cnq, sizes, pf);
                eprintln!(
                    "glm5_tiers #186 chunk {chunk} {name}: ids {:?}, prompt reports (rows, syncs, sub-batches) {:?}",
                    out.ids,
                    reps.iter().filter(|r| r.prompt).map(|r| (r.rows, r.routing_syncs, r.sub_batches)).collect::<Vec<_>>()
                );
                chunked.push((name, out, reps));
            }
            // the decode switches after a chunked prompt: flags + lookahead, then the graphs
            run.set_switches(&mut cnq, Switches { flags: true, lookahead: true, ..Switches::default() });
            let (sw, _) = gen_with(&mut run, &mut cnq, TierSizes { vram: 3, pinned: 4 }, None);
            run.set_switches(&mut cnq, Switches::default());
            run.set_graph(true);
            let (gr, _) = gen_with(&mut run, &mut cnq, TierSizes { vram: 3, pinned: 4 }, None);
            run.set_graph(false);
            // the chain: every layer with its records in VRAM, one layer at a time, the same calls
            let ids: Vec<i64> = prompt.iter().copied().chain(chunked[0].1.ids[..n - 1].iter().copied()).collect();
            let rows = ids.len();
            let mut calls: Vec<(usize, usize, bool)> = prompt_calls(prompt.len(), chunk).into_iter().map(|(r0, t)| (r0, t, false)).collect();
            calls.extend((prompt.len()..rows).map(|r| (r, 1, true)));
            let row = g.hc_streams * g.hidden;
            let mut pass = Glm5Pass::new(&g, moe, chunk, cap);
            let mut xd = cuda::alloc_named("chain residual", rows * row * 4);
            cuda::to_f32_into(xd, &gm::trunk_input(&gm::embed_rows(&mut cnq, &g, &ids), g.hidden, g.hc_streams));
            let mut rep = LoadReport::default();
            for l in 0..g.layers {
                let mut lw = gm::load_layer(&mut cnq, &g, &moe, l, &mut rep);
                pass.begin_layer();
                for &(r0, t, decode) in &calls {
                    pass.call(&lw, xd + (r0 * row * 4) as u64, r0, t, decode);
                }
                cuda::sync();
                lw.free();
            }
            let mut head = Head::new(gm::head_geo(&g));
            let mut hw = gm::load_head(&mut cnq, &g, &mut rep);
            let (mut normed, mut logits, mut next) = (cuda::alloc_named("chain normed", g.hidden * 4), cuda::alloc_named("chain logits", g.vocab * 4), cuda::alloc_named("chain id", 4));
            let mut chain = Vec::new();
            for r in prompt.len() - 1..rows {
                gm::run_head(&pass.kn, &head, &hw, xd + (r * row * 4) as u64, normed, logits, next, 1);
                cuda::sync();
                chain.push(cuda::dtoh(logits, g.vocab));
            }
            for d in [&mut xd, &mut normed, &mut logits, &mut next, &mut hw.norm, &mut hw.lm] {
                cuda::free_dev(d);
            }
            head.free();
            pass.free();
            run.free();
            drop(cnq);
            // the row-by-row path is unchanged by the larger pass
            assert_eq!(a.ids, a0.ids, "chunk 1 on a pass of {chunk}: ids");
            assert!(bits_differ(&a.logits, &a0.logits).iter().all(|&d| d == 0), "chunk 1 on a pass of {chunk}: logits differ in bits");
            assert_eq!(ra.iter().filter(|r| r.prompt).count(), prompt.len(), "chunk 1 reports every prompt row");
            // tiers and sub-batches are invisible in a prompt call
            let (_, b, rb) = &chunked[0];
            let finite = b.logits.iter().flatten().filter(|v| v.is_finite()).count();
            assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
            for (name, x) in chunked[1..].iter().map(|c| (c.0, &c.1)).chain([("flags + lookahead", &sw), ("CROW_GLM_GRAPH", &gr)]) {
                assert_eq!(x.ids, b.ids, "{name}: ids");
                let d = bits_differ(&x.logits, &b.logits);
                assert!(d.iter().all(|&v| v == 0), "{name}: logits differ in bits per generated position {d:?}");
            }
            // reports: one per prompt call, its rows, one routing sync per MoE layer per call
            let want: Vec<usize> = prompt_calls(prompt.len(), chunk).iter().map(|c| c.1).collect();
            for (name, _, reps) in &chunked {
                let pre: Vec<&TokenReport> = reps.iter().filter(|r| r.prompt).collect();
                assert_eq!(pre.iter().map(|r| r.rows).collect::<Vec<_>>(), want, "{name}: prompt reports");
                assert!(pre.iter().all(|r| r.routing_syncs == 1), "{name}: one MoE layer, one routing sync per call");
                assert_eq!(pre.last().unwrap().next, Some(b.ids[0]), "{name}: the last prompt call gives the first id");
                assert_eq!(reps.iter().filter(|r| !r.prompt).count(), n - 1, "{name}: decode rows");
            }
            let sub = |reps: &[TokenReport]| reps.iter().filter(|r| r.prompt).map(|r| r.sub_batches).sum::<u64>();
            assert_eq!(sub(rb), want.len() as u64, "V 3 + P 4 with 128 prefill slots: no call splits");
            assert!(sub(&chunked[1].2) > want.len() as u64, "8 prefill slots must split the calls into row sub-batches");
            assert!(sub(&chunked[3].2) > want.len() as u64, "two halves of 8 slots must split the calls into sub-batches");
            // the chain computes the same bits
            let d = bits_differ(&b.logits, &chain);
            assert!(d.iter().all(|&v| v == 0), "chunked tiered run vs layer-at-a-time chain: logits differ in bits per generated position {d:?}");
            // chunked vs row by row: close, not bit-identical
            let kls: Vec<f64> = a.logits.iter().zip(&b.logits).map(|(p, q)| kl(p, q)).collect();
            let maxabs = a.logits.iter().zip(&b.logits).map(|(p, q)| p.iter().zip(q).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max)).fold(0f32, f32::max);
            eprintln!("glm5_tiers #186 chunk {chunk} vs row by row: ids {:?} vs {:?}, KL per generated row {kls:?}, max abs logit difference {maxabs:.3e}", b.ids, a.ids);
            assert_eq!(b.ids, a.ids, "chunk {chunk} vs row by row: ids");
            assert!(kls.iter().sum::<f64>() / kls.len() as f64 <= 0.073, "mean KL above 0.073");
        }
    }
}

// ---------------------------------------------------------------- #192: MTP speculative decode

/// #192 (`CROW_GLM_MTP=N`, plan step 23): greedy decoding with the MTP block's drafts, verified
/// by one multi-row trunk call per step. Lossless by construction: every verify row's logits are
/// bit for bit the one-row decode path's (`Glm5Pass::call_verify_with_experts`), so the emitted
/// ids are the ids of `generate` with the switch off; MTP changes only how many rows one trunk
/// call carries.
///
/// Per step at trunk position `P` (last emitted id `t_P`):
/// 1. the drafts `d_{P+1} ..` were made at the end of the previous step (or the prompt): the
///    block's row `P - 1` = (embed(t_P), h_{P-1}) through the trunk's lm_head, chained on its own
///    `shared_head.norm` row for `N > 1`;
/// 2. ONE verify call: rows `P ..= P + k` with inputs `[t_P, d_{P+1} .. d_{P+k}]`, every layer,
///    the experts of all rows staged once (one `table_for` per MoE layer), the head over the rows;
/// 3. accepted: the longest prefix with `d_{P+j} == ` the trunk's greedy id of row `P + j - 1`;
///    the step emits those ids and the trunk's own next id (`a` ids, `1 ..= k + 1`);
/// 4. rollback for `a <= k`: every KDA state back to its snapshot after row `P + a - 1` (D2D,
///    taken during the verify); MLA / DSA rows and the block's cache rows of rejected positions
///    stay and are rewritten before any later row reads them (absolute positions);
/// 5. the block's catch-up over the accepted rows `P .. P + a - 1` (true ids, the trunk's
///    head-norm rows), whose last row drafts the next step.
///
/// The prompt rows run as in `generate` (`Glm5Run::row`, graphs if `CROW_GLM_GRAPH`); each also
/// leaves its head-norm row for the block, which catches up over the prompt in calls of
/// [`crate::glm5_mtp::MTP_CHUNK`] rows (their time is in the prompt rows' reports). The verify
/// and the block's calls run uncaptured. `CROW_GLM_LOOKAHEAD` and `CROW_CHUNK` (#186) do not apply;
/// the verify is not the #186 prompt call (KDA prompt path, MLA multi-row: other bits). The CPU lane
/// (`CROW_GLM_CPU_LANE=1`) is refused (its experts have other bits than the verify's GPU path).
/// #192: the block's window over prompt rows: `k` rows of `Spec::h` / `Spec::e` filled, for
/// the block's cache positions `start ..`
struct BlockWindow {
    start: usize,
    k: usize,
}

impl Glm5Run {
    /// Put the speculative decode in force: `n` drafts per step with `block` (`n = 0` or no block:
    /// off, the state freed). The pass must have been built for `1 + n` rows (`load` sizes it
    /// from `CROW_GLM_MTP`).
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this run is pending.
    pub unsafe fn set_mtp(&mut self, n: usize, block: Option<crate::glm5_mtp::MtpBlock>) -> Result<(), String> {
        if let Some(mut sp) = self.spec.take() {
            sp.free();
        }
        let Some(mut block) = block else { return Ok(()) };
        if n == 0 {
            block.free();
            return Ok(());
        }
        if self.pass.max_t < 1 + n {
            block.free();
            return Err(format!("{}={n}: the model was loaded for {} verify rows; set {} before Glm5Run::load", crate::glm5_mtp::MTP_ENV, self.pass.max_t, crate::glm5_mtp::MTP_ENV));
        }
        let kda: Vec<bool> = self.kda.iter().map(Option::is_some).collect();
        self.spec = Some(Box::new(crate::glm5_mtp::Spec::new(&self.g, self.moe, self.cap, n, block, &kda)));
        Ok(())
    }

    /// `CROW_GLM_MTP=N` (N > 0): load the MTP block from `cnq` (its 288 MUL1 records, section
    /// `mtp`) and the overlay (`CROW_GLM_MTP_OVERLAY`, default
    /// `converter/GLM-5.3-Flash-MTP-overlay.cnq`), then [`Glm5Run::set_mtp`]. Returns N (0: off,
    /// nothing loaded).
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this run is pending.
    pub unsafe fn mtp_from_env(&mut self, cnq: &mut Cnq, log: &mut dyn FnMut(&str)) -> Result<usize, String> {
        use crate::glm5_mtp as mtp;
        let n = mtp::draft_rows_from_env()?;
        if n == 0 {
            return Ok(0);
        }
        let path = std::env::var(mtp::OVERLAY_ENV).ok().filter(|p| !p.trim().is_empty()).unwrap_or_else(|| crate::geo::from_engine_dir(mtp::GLM5_MTP_OVERLAY_CNQ));
        let t0 = std::time::Instant::now();
        cuda::sync();
        let free0 = cuda::free_vram_bytes();
        let mut ov = Cnq::open_checked(&path).map_err(|e| format!("{} {path}: {e}", mtp::OVERLAY_ENV))?;
        let block = mtp::load_mtp(cnq, &mut ov, &self.g, &self.moe)?;
        let (bytes, sanitized, inexact) = (block.bytes, block.sanitized, block.kv_b_inexact);
        self.set_mtp(n, Some(block))?;
        cuda::sync();
        let used = free0.saturating_sub(cuda::free_vram_bytes());
        let snap = self.spec.as_ref().map_or(0, |s| s.snapshot_bytes());
        log(&format!(
            "[glm5_run] {}={n}: MTP block loaded in {:.1} s from the container + {path}: {bytes} B to VRAM, {sanitized} NVFP4 scale bytes 0x7F -> 0x7E, kv_b {inexact} values inexact; KDA snapshot slots {n} x {snap} B; VRAM used by MTP {used} B (derived beforehand {} B)",
            mtp::MTP_ENV,
            t0.elapsed().as_secs_f64(),
            mtp::spec_vram_bytes(&self.g, &self.moe, self.cap, n)
        ));
        Ok(n)
    }

    /// the drafts per step in force (0 = off)
    pub fn mtp_drafts(&self) -> usize {
        self.spec.as_ref().map_or(0, |s| s.n)
    }

    /// the speculative decode's counters of the last `generate` (None when off)
    pub fn mtp_stats(&self) -> Option<&crate::glm5_mtp::SpecStats> {
        self.spec.as_ref().map(|s| &s.stats)
    }

    /// test hook: override every draft (`(index of the generated id it guesses, draft) -> draft`)
    #[cfg(test)]
    pub(crate) fn set_mtp_hook(&mut self, hook: Option<crate::glm5_mtp::DraftHook>) {
        if let Some(s) = self.spec.as_mut() {
            s.hook = hook;
        }
    }

    /// test probe: called after every KDA rollback with the last valid row and the states' bytes
    #[cfg(test)]
    pub(crate) fn set_mtp_probe(&mut self, probe: Option<Box<dyn FnMut(usize, &[Vec<u8>])>>) {
        if let Some(s) = self.spec.as_mut() {
            s.probe = probe;
        }
    }

    /// the MLA cache of every DSA layer, in layer order
    pub fn mla_caches(&self) -> impl Iterator<Item = &MlaCache> {
        self.mla.iter().flatten()
    }

    /// [`Glm5Run::generate`] with the speculative decode (the state taken out for the run)
    ///
    /// # Safety
    /// As [`Glm5Run::generate`].
    unsafe fn generate_spec(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], n: usize, keep_logits: bool, report: &mut dyn FnMut(&TokenReport)) -> Result<Generated, String> {
        let mut sp = self.spec.take().expect("glm5_run: generate_spec without its state");
        let r = self.spec_run(&mut sp, cnq, tiers, prompt, n, keep_logits, report);
        self.spec = Some(sp);
        r
    }

    /// the refusals of the speculative decode, by name: the CPU lane (its experts have other bits
    /// than the verify's GPU kernels) and too few staging slots for a verify of `1 + N` rows
    pub(crate) fn spec_check(nd: usize, tiers: &ExpertTiers, topk: usize) -> Result<(), String> {
        use crate::glm5_mtp::MTP_ENV;
        if tiers.pinned_use.cpu_lane {
            return Err(format!("{MTP_ENV}={nd} and {CPU_LANE_ENV}=1: the CPU lane's experts have other bits than the verify's GPU kernels, so the ids could differ from the run without MTP; turn one of them off"));
        }
        if tiers.stage_cap < (1 + nd) * topk {
            return Err(format!("{MTP_ENV}={nd}: the tiers hold {} staging slots, a verify of {} rows may stage {}; build ExpertTiers with stage_cap (1 + {nd}) x top-k", tiers.stage_cap, 1 + nd, (1 + nd) * topk));
        }
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    unsafe fn spec_run(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, tiers: &mut ExpertTiers, prompt: &[i64], n: usize, keep_logits: bool, report: &mut dyn FnMut(&TokenReport)) -> Result<Generated, String> {
        let (g, h, v, nd, cap) = (self.g, self.g.hidden, self.g.vocab, sp.n, self.cap);
        Self::spec_check(nd, tiers, g.topk)?;
        sp.reset_stats();
        sp.ahead = None;
        let pn = prompt.len();
        let row_at = |b: Dev, r: usize, w: usize| b + (r * w * 4) as u64;
        // the drafts of the step whose first verify row is `pos`, with `emitted` ids out
        let k_of = |emitted: usize, pos: usize| nd.min(n.saturating_sub(emitted + 1)).min(cap.saturating_sub(pos + 1));
        let use_mtp = k_of(1, pn) > 0;
        let mut out = Generated::default();
        // the prompt rows (prompt calls of `CROW_CHUNK` rows, or one by one); after each call its
        // rows' head-norm rows go to the block's windows, the windows of the row path (MTP_CHUNK
        // rows from position 0); the window of the last row waits for the first id
        let mut w = BlockWindow { start: 0, k: 0 };
        let first = {
            let mut rows = |run: &mut Glm5Run, cnq: &mut Cnq, r0: usize, t: usize| -> Result<(), String> {
                if use_mtp {
                    run.spec_window_rows(sp, cnq, prompt, r0, t, &mut w, false);
                }
                Ok(())
            };
            self.prefill_with(cnq, tiers, prompt, 0, report, &mut rows)?
        };
        if keep_logits {
            out.logits.push(cuda::dtoh(self.logits, v));
        }
        out.ids.push(first);
        let mut drafts: Vec<i64> = Vec::new();
        if use_mtp {
            cuda::to_f32_into(row_at(sp.e, w.k - 1, h), &gm::embed_rows(cnq, &g, &[first]));
            self.spec_block(sp, sp.h, w.start, w.k);
            drafts = self.spec_drafts(sp, cnq, w.k - 1, pn - 1, k_of(1, pn), 1)?;
        }
        let mut pos = pn;
        let mut last = first;
        while out.ids.len() < n {
            let t0 = std::time::Instant::now();
            let base = RowBase::of(tiers);
            let (ids, a) = self.spec_verify(sp, cnq, tiers, pos, last, &drafts, true)?;
            for (j, &id) in ids.iter().take(a).enumerate() {
                if keep_logits {
                    out.logits.push(cuda::dtoh(row_at(sp.logits, j, v), v));
                }
                out.ids.push(id);
            }
            let p0 = pos;
            pos += a;
            last = ids[a - 1];
            let kn = k_of(out.ids.len(), pos);
            drafts = Vec::new();
            if out.ids.len() < n && kn > 0 {
                // the block over the accepted rows: (embed(id), the trunk's head-norm row) at p0 ..
                cuda::to_f32_into(sp.e, &gm::embed_rows(cnq, &g, &ids[..a]));
                self.spec_block(sp, sp.normed, p0, a);
                drafts = self.spec_drafts(sp, cnq, a - 1, pos - 1, kn, out.ids.len())?;
            }
            cuda::sync();
            // one report per emitted id: the step's counters on its first id, its clock shared
            let r = base.report(tiers, p0, false, Some(ids[0]), t0);
            let secs = r.secs / a as f64;
            for (j, &id) in ids.iter().take(a).enumerate() {
                let mut rj = if j == 0 {
                    r.clone()
                } else {
                    TokenReport { tiers: vec![[0; 3]; r.tiers.len()], moves: vec![Moves::default(); r.moves.len()], ..TokenReport::default() }
                };
                (rj.pos, rj.prompt, rj.next, rj.secs) = (p0 + j, false, Some(id), secs);
                report(&rj);
            }
        }
        Ok(out)
    }

    /// The block over `t` rows at cache positions `p0 ..`: embeddings in `Spec::e`, head-norm
    /// rows at `h`; counted.
    ///
    /// # Safety
    /// A CUDA context is current; `h` and `Spec::e` hold `t` rows.
    unsafe fn spec_block(&self, sp: &mut crate::glm5_mtp::Spec, h: Dev, p0: usize, t: usize) {
        sp.block.call(&mut sp.mp, &self.pass.kn, &sp.mk, sp.e, h, p0, t, false, None);
        sp.stats.mtp_rows += t as u64;
    }

    /// A prompt call's rows `r0 .. r0 + t` of `ids` (the first `t` rows of `x`) into the block's
    /// window: their head-norm rows into `Spec::h`, the next ids' embeddings into `Spec::e`, the
    /// block over every full window of `MTP_CHUNK` rows. `pend` (serve): the last row of `ids`
    /// leaves its head-norm row in `Spec::hp` instead, its next id is unknown. Without `pend`
    /// (generate) the last row stays in the window and the window waits for its embedding.
    ///
    /// # Safety
    /// A CUDA context is current; `x` holds the call's rows.
    #[allow(clippy::too_many_arguments)]
    unsafe fn spec_window_rows(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, ids: &[i64], r0: usize, t: usize, w: &mut BlockWindow, pend: bool) {
        use crate::glm5_mtp::MTP_CHUNK;
        let (g, h) = (self.g, self.g.hidden);
        let (row, n) = (g.hc_streams * h, ids.len());
        let x0 = self.x;
        let x_at = |i: usize| x0 + ((i - r0) * row * 4) as u64;
        let mut i = r0;
        while i < r0 + t {
            if pend && i + 1 == n {
                self.head.stream_mean_rms(x_at(i), self.hw.norm, sp.hp, 1);
                i += 1;
                continue;
            }
            let lim = if pend { n - 1 } else { n };
            let piece = (r0 + t).min(lim).min(i + MTP_CHUNK - w.k) - i;
            // one block per row: a row's head-norm row does not depend on the launch's rows
            self.head.stream_mean_rms(x_at(i), self.hw.norm, sp.h + (w.k * h * 4) as u64, piece);
            let known = (i + piece).min(n - 1).saturating_sub(i);
            if known > 0 {
                cuda::to_f32_into(sp.e + (w.k * h * 4) as u64, &gm::embed_rows(cnq, &g, &ids[i + 1..i + 1 + known]));
            }
            w.k += piece;
            i += piece;
            if w.k == MTP_CHUNK && (pend || i < n) {
                self.spec_block(sp, sp.h, w.start, w.k);
                w.start += w.k;
                w.k = 0;
            }
        }
    }

    /// One verify step at `pos`: `last` and `drafts` through every layer
    /// (`Glm5Pass::call_verify_with_experts`, a KDA snapshot after every row but the last), the
    /// head over every row; `a` = 1 + the drafts that equal the trunk's greedy ids. `greedy`
    /// (generate): the step emits `a` ids and the KDA states go back to after row `pos + a - 1`
    /// when a draft was rejected. Without it (serve) every row stays; the caller's ids decide
    /// (`Glm5Run::spec_decode`). Counted. Returns the verify's greedy ids (`1 + drafts`) and `a`.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model; `pos + 1 + drafts <= cap`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn spec_verify(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, tiers: &mut ExpertTiers, pos: usize, last: i64, drafts: &[i64], greedy: bool) -> Result<(Vec<i64>, usize), String> {
        let (g, h, v) = (self.g, self.g.hidden, self.g.vocab);
        let k = drafts.len();
        let t = 1 + k;
        let toks: Vec<i64> = std::iter::once(last).chain(drafts.iter().copied()).collect();
        cuda::to_f32_into(sp.x, &gm::trunk_input(&gm::embed_rows(cnq, &g, &toks), h, g.hc_streams));
        for l in 0..g.layers {
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            let mut hook = |layer: usize, sel: &[i32]| -> Result<Dev, String> { tiers.table_for(layer, sel).map(|(tb, _)| tb) };
            let r = self.pass.call_verify_with_experts(&self.layers[l], sp.x, pos, t, &sp.snaps[l], &mut hook);
            // the layer's own state goes back even when the call failed
            if let Some(s) = self.kda[l].as_mut() {
                self.pass.swap_kda_state(s);
            }
            if let Some(c) = self.mla[l].as_mut() {
                self.pass.swap_mla_cache(c);
            }
            r.map_err(|e| format!("glm5_run: verify rows {pos}..{} layer {l}: {e}", pos + t))?;
        }
        gm::run_head(&self.pass.kn, &self.head, &self.hw, sp.x, sp.normed, sp.logits, sp.ids, t);
        cuda::sync();
        let ids: Vec<i64> = cuda::dtoh_i32(sp.ids, t).into_iter().map(i64::from).collect();
        if let Some(&bad) = ids.iter().find(|&&id| !(0..v as i64).contains(&id)) {
            return Err(format!("glm5_run: verify rows {pos}..{}: the greedy id {bad} is outside the vocab of {v}", pos + t));
        }
        let mut a = 1;
        while a <= k && drafts[a - 1] == ids[a - 1] {
            a += 1;
        }
        let n_kda = self.kda.iter().flatten().count() as u64;
        let slot = sp.snapshot_bytes();
        let st = &mut sp.stats;
        st.steps += 1;
        st.drafts += k as u64;
        st.verify_rows += t as u64;
        st.kda_snapshots += k as u64 * n_kda;
        st.kda_snapshot_bytes += k as u64 * slot;
        if !greedy {
            // serve: the first id now, the others when the caller reaches them
            st.tokens += 1;
            return Ok((ids, a));
        }
        st.tokens += a as u64;
        st.accepted += (a - 1) as u64;
        st.hist[a - 1] += 1;
        if a < t {
            // a rejected draft: every KDA state back to its state after row pos + a - 1
            self.spec_restore(sp, a - 1);
            #[cfg(test)]
            if let Some(pr) = sp.probe.as_mut() {
                cuda::sync();
                let kd = KdaDims::of(&g);
                let b: Vec<Vec<u8>> = self.kda.iter().flatten().flat_map(|s| [cuda::dtoh_t::<u8>(s.s, kd.state_floats() * 4), cuda::dtoh_t::<u8>(s.conv, kd.conv_floats() * 4)]).collect();
                pr(pos + a - 1, &b);
            }
        }
        Ok((ids, a))
    }

    /// every KDA state from snapshot slot `j` (its state after verify row `j`); counted
    ///
    /// # Safety
    /// A CUDA context is current; the last verify wrote slot `j`.
    unsafe fn spec_restore(&mut self, sp: &mut crate::glm5_mtp::Spec, j: usize) {
        for (s, snaps) in self.kda.iter().zip(&sp.snaps) {
            if let Some(s) = s {
                s.copy_from(&snaps[j]);
            }
        }
        let slot = sp.snapshot_bytes();
        let st = &mut sp.stats;
        st.kda_restore_steps += 1;
        st.kda_restores += self.kda.iter().flatten().count() as u64;
        st.kda_restore_bytes += slot;
    }

    // ------------------------------------------------ serve (#192): the speculative decode one id at a time

    /// serve: the prompt rows `ids` at `pos0 ..` (`CROW_CHUNK` prompt calls or row by row, as
    /// [`Glm5Run::prefill`]) with the block: first the sequence is brought level at `pos0` with
    /// `ids[0]` as its next id ([`Glm5Run::spec_settle`]), then the block runs over every prompt
    /// row whose next id is in `ids` (windows of `MTP_CHUNK` rows from `pos0`); the last row's
    /// head-norm row waits in `Spec::hp`. Returns the greedy id of the last row (its logits in
    /// [`Glm5Run::logits_dev`]).
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model; the speculative decode is on.
    pub unsafe fn spec_prefill(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, ids: &[i64], pos0: usize) -> Result<i64, String> {
        let mut sp = self.spec.take().ok_or_else(|| "glm5_run: spec_prefill without CROW_GLM_MTP".to_string())?;
        let r = self.spec_prefill_in(&mut sp, cnq, tiers, ids, pos0);
        self.spec = Some(sp);
        r
    }

    unsafe fn spec_prefill_in(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, tiers: &mut ExpertTiers, ids: &[i64], pos0: usize) -> Result<i64, String> {
        if ids.is_empty() || pos0 + ids.len() > self.cap {
            return Err(format!("glm5_run: a prompt of {} rows at {pos0}, the caches hold {}", ids.len(), self.cap));
        }
        if let Some(bad) = ids.iter().find(|&&t| !(0..self.g.vocab as i64).contains(&t)) {
            return Err(format!("glm5_run: token id {bad} outside the vocab of {}", self.g.vocab));
        }
        Self::spec_check(sp.n, tiers, self.g.topk)?;
        self.spec_settle(sp, cnq, ids[0], pos0)?;
        let mut w = BlockWindow { start: pos0, k: 0 };
        let id = {
            let mut rows = |run: &mut Glm5Run, cnq: &mut Cnq, r0: usize, t: usize| -> Result<(), String> {
                run.spec_window_rows(sp, cnq, ids, r0, t, &mut w, true);
                Ok(())
            };
            self.prefill_with(cnq, tiers, ids, pos0, &mut |_| {}, &mut rows)?
        };
        if w.k > 0 {
            self.spec_block(sp, sp.h, w.start, w.k);
        }
        cuda::sync();
        Ok(id)
    }

    /// serve: one decode step, `tok` at `pos` (the caller's sequence holds `pos` rows): the
    /// greedy id after it, its logits row in [`Glm5Run::logits_dev`] (the verify's row, the bits
    /// of the one-row decode). While the last verify's rows lie ahead (`Spec::ahead`) and `tok`
    /// is the draft the verify fed at `pos`, that row's id and logits row are handed out without
    /// a launch: a greedy caller keeps the drafts that equal the argmax, a sampling caller the
    /// drafts its draws hit (speculative sampling with a point-mass draft: kept with probability
    /// p(draft), otherwise the draw is from p without the draft). Otherwise the sequence is
    /// brought level ([`Glm5Run::spec_settle`]: the KDA states back to after row `pos - 1`, the
    /// block over the rows whose next id is now known), up to `N` drafts are chained from the
    /// block, and one verify runs.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model; the speculative decode is on.
    pub unsafe fn spec_decode(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, tok: i64, pos: usize) -> Result<i64, String> {
        let mut sp = self.spec.take().ok_or_else(|| "glm5_run: spec_decode without CROW_GLM_MTP".to_string())?;
        let r = self.spec_decode_in(&mut sp, cnq, tiers, tok, pos);
        self.spec = Some(sp);
        r
    }

    unsafe fn spec_decode_in(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, tiers: &mut ExpertTiers, tok: i64, pos: usize) -> Result<i64, String> {
        let v = self.g.vocab;
        if pos >= self.cap {
            return Err(format!("glm5_run: row {pos} is outside the caches of {} rows", self.cap));
        }
        if !(0..v as i64).contains(&tok) {
            return Err(format!("glm5_run: token id {tok} outside the vocab of {v}"));
        }
        if let Some(ah) = sp.ahead.as_mut() {
            let c = ah.taken;
            if pos == ah.p0 + c && c < ah.fed.len() && tok == ah.fed[c] {
                // the trunk already holds row `pos` with `tok` (the draft the verify fed there):
                // its logits row and greedy id, no launch
                ah.taken += 1;
                sp.stats.tokens += 1;
                sp.stats.accepted += 1;
                cuda::d2d_async(self.logits, sp.logits + (c * v * 4) as u64, v * 4);
                cuda::sync();
                return Ok(ah.ids[c]);
            }
        }
        Self::spec_check(sp.n, tiers, self.g.topk)?;
        let row = self.spec_settle(sp, cnq, tok, pos)?;
        let k = if row.is_some() { sp.n.min(self.cap.saturating_sub(pos + 1)) } else { 0 };
        let drafts = match row {
            Some(r) if k > 0 => self.spec_drafts(sp, cnq, r, pos - 1, k, pos + 1)?,
            _ => Vec::new(),
        };
        let (ids, _) = self.spec_verify(sp, cnq, tiers, pos, tok, &drafts, false)?;
        let fed = std::iter::once(tok).chain(drafts.iter().copied()).collect();
        let first = ids[0];
        sp.ahead = Some(crate::glm5_mtp::Ahead { p0: pos, fed, ids, taken: 1 });
        cuda::d2d_async(self.logits, sp.logits, v * 4);
        cuda::sync();
        Ok(first)
    }

    /// Bring the sequence level at `pos` with `tok` as the id at `pos`: with verify rows ahead,
    /// the KDA states back to after row `pos - 1` (snapshot slot `taken - 1`) unless the trunk
    /// holds exactly `pos` rows, and the block over the verify's rows `p0 .. pos` (their next ids:
    /// the drafts the caller reached, then `tok`); with none ahead, the block over row `pos - 1`
    /// (the pending head-norm row `Spec::hp`, next id `tok`). The MLA / DSA rows `>= pos` of the trunk and the
    /// block are overwritten before any call reads them. Returns the block's output row of row
    /// `pos - 1` in `MtpPass::normed` (`None` at `pos` 0).
    ///
    /// # Safety
    /// A CUDA context is current.
    unsafe fn spec_settle(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, tok: i64, pos: usize) -> Result<Option<usize>, String> {
        let g = self.g;
        match sp.ahead.take() {
            Some(ah) => {
                let c = ah.taken;
                if pos != ah.p0 + c {
                    return Err(format!("glm5_run: a row at {pos}, the sequence holds {} rows (MTP verify at {} handed out {c} ids)", ah.p0 + c, ah.p0));
                }
                sp.stats.hist[c - 1] += 1;
                if c < ah.fed.len() {
                    // the caller's id at `pos` is not the draft the verify fed there
                    self.spec_restore(sp, c - 1);
                }
                let e: Vec<i64> = ah.fed[1..c].iter().copied().chain(std::iter::once(tok)).collect();
                cuda::to_f32_into(sp.e, &gm::embed_rows(cnq, &g, &e));
                self.spec_block(sp, sp.normed, ah.p0, c);
                Ok(Some(c - 1))
            }
            None if pos > 0 => {
                cuda::to_f32_into(sp.e, &gm::embed_rows(cnq, &g, &[tok]));
                self.spec_block(sp, sp.hp, pos - 1, 1);
                Ok(Some(0))
            }
            None => Ok(None),
        }
    }

    /// serve: f32 values the prefix snapshot copies besides the KDA states (the pending
    /// head-norm row `Spec::hp`; 0 when off)
    pub fn spec_state_floats(&self) -> usize {
        if self.spec.is_some() {
            self.g.hidden
        } else {
            0
        }
    }

    /// serve: the pending head-norm row the prefix snapshot copies (`None` when off). The engine
    /// snapshots after a prefill only, when no verified ids are ahead.
    pub fn spec_pending_row(&self) -> Option<Dev> {
        self.spec.as_ref().map(|sp| {
            assert!(sp.ahead.is_none(), "glm5_run: a prefix snapshot while MTP verified ids are ahead of the sequence (the engine snapshots after a prefill only)");
            sp.hp
        })
    }

    /// serve: the sequence was reset or rolled back to a snapshot; nothing is ahead of it
    pub fn spec_level(&mut self) {
        if let Some(sp) = self.spec.as_mut() {
            sp.ahead = None;
        }
    }

    /// test probe: the block's MLA latent and indexer rows `0 .. rows` and its pending
    /// head-norm row (bytes; empty when off)
    #[cfg(test)]
    pub(crate) unsafe fn mtp_state_bytes(&self, rows: usize) -> Vec<Vec<u8>> {
        let Some(sp) = self.spec.as_ref() else { return Vec::new() };
        cuda::sync();
        let md = MlaDims::of(&self.g);
        let c = &sp.mp.mla_c;
        vec![
            cuda::dtoh_t::<u8>(c.latent, rows * md.latent_bytes_per_token() as usize),
            cuda::dtoh_t::<u8>(c.index, rows * md.indexer_bytes_per_token() as usize),
            cuda::dtoh_t::<u8>(sp.hp, self.g.hidden * 4),
        ]
    }

    /// serve: the counters since the last call, then zero (`None` when off)
    pub fn mtp_take_stats(&mut self) -> Option<crate::glm5_mtp::SpecStats> {
        self.spec.as_mut().map(|sp| {
            let s = sp.stats.clone();
            sp.reset_stats();
            s
        })
    }

    /// `k` drafts from the block's `normed` row `row` (its cache position `pos`): the trunk's
    /// lm_head and greedy id, then for `k > 1` the block again on (embed(draft), its own normed
    /// row) at the next position. `gen` = the index of the generated id the first draft guesses
    /// (`generate`), the position of the id it guesses (serve); the test hook's argument.
    ///
    /// # Safety
    /// A CUDA context is current; the block's call that wrote `row` is queued.
    unsafe fn spec_drafts(&mut self, sp: &mut crate::glm5_mtp::Spec, cnq: &mut Cnq, row: usize, pos: usize, k: usize, gen: usize) -> Result<Vec<i64>, String> {
        let (g, h, v) = (self.g, self.g.hidden, self.g.vocab);
        let mut d: Vec<i64> = Vec::with_capacity(k);
        let mut src = sp.mp.normed + (row * h * 4) as u64;
        for j in 0..k {
            if j > 0 {
                cuda::d2d_async(sp.h, src, h * 4);
                cuda::to_f32_into(sp.e, &gm::embed_rows(cnq, &g, &[d[j - 1]]));
                sp.block.call(&mut sp.mp, &self.pass.kn, &sp.mk, sp.e, sp.h, pos + j, 1, false, None);
                sp.stats.mtp_rows += 1;
                src = sp.mp.normed;
            }
            self.head.lm_head(&self.pass.kn.k, self.hw.lm, src, sp.dlogits, 1);
            self.head.argmax(&self.pass.kn.k, sp.dlogits, sp.did, 1);
            cuda::sync();
            let mut id = cuda::dtoh_i32(sp.did, 1)[0] as i64;
            if let Some(hk) = sp.hook.as_mut() {
                id = hk(gen + j, id);
            }
            if !(0..v as i64).contains(&id) {
                return Err(format!("glm5_run: MTP draft {id} is outside the vocab of {v}"));
            }
            d.push(id);
        }
        Ok(d)
    }
}

#[cfg(test)]
mod spec_tests {
    //! #192 on the GPU: the synthetic 8-layer glm5_next model of the `glm5_graph` tests (layers
    //! 0-2 KDA + dense, 3-7 MoE with 16 MUL1 experts, top-8, DSA at 3 and 7, vocab 2048) and a
    //! synthetic MTP block. `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::glm5_flags::tests::synth_model;
    use crate::glm5_flags::Switches;
    use crate::glm5_mtp::{self as mtp, SpecStats};
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    const REC: u64 = 9_474_048;

    fn geo() -> Glm5Geo {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (8, 3, 16, 8, 2048);
        g
    }

    fn hash(bs: &[Vec<u8>]) -> u64 {
        let mut x = 0xcbf2_9ce4_8422_2325u64;
        for b in bs {
            for c in b.chunks_exact(8) {
                x = (x ^ u64::from_le_bytes(c.try_into().unwrap())).wrapping_mul(0x0000_0100_0000_01b3);
            }
        }
        x
    }

    /// every KDA state's S and conv bytes, then every MLA cache's latent and indexer rows
    /// `0 ..= last`
    unsafe fn states(run: &Glm5Run, g: &Glm5Geo, last: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>) {
        cuda::sync();
        let (kd, md) = (KdaDims::of(g), MlaDims::of(g));
        let kda = run.kda_states().flat_map(|s| [cuda::dtoh_t::<u8>(s.s, kd.state_floats() * 4), cuda::dtoh_t::<u8>(s.conv, kd.conv_floats() * 4)]).collect();
        let rows = last + 1;
        let mla = run
            .mla_caches()
            .flat_map(|c| [cuda::dtoh_t::<u8>(c.latent, rows * md.latent_bytes_per_token() as usize), cuda::dtoh_t::<u8>(c.index, rows * md.indexer_bytes_per_token() as usize)])
            .collect();
        (kda, mla)
    }

    /// The counters a draft pattern must give: `ok(i)` = the draft that guesses generated id `i`
    /// is right. `(stats, last valid row of every rollback)`
    fn simulate(n_draft: usize, pn: usize, n: usize, cap: usize, ok: &dyn Fn(usize) -> bool) -> (SpecStats, Vec<usize>) {
        let mut s = SpecStats { n: n_draft, hist: vec![0; n_draft + 1], ..SpecStats::default() };
        let mut restores = Vec::new();
        let (mut emitted, mut pos) = (1usize, pn);
        while emitted < n {
            let k = n_draft.min(n - emitted - 1).min(cap - pos - 1);
            let mut a = 1;
            while a <= k && ok(emitted + a - 1) {
                a += 1;
            }
            s.steps += 1;
            s.tokens += a as u64;
            s.drafts += k as u64;
            s.accepted += (a - 1) as u64;
            s.verify_rows += (k + 1) as u64;
            s.hist[a - 1] += 1;
            if a < k + 1 {
                s.kda_restore_steps += 1;
                restores.push(pos + a - 1);
            }
            emitted += a;
            pos += a;
        }
        (s, restores)
    }

    /// Lossless and rollback, `CROW_GLM_MTP` on the synthetic model: a 5-id prompt and 70 greedy
    /// ids, V 3 + P 4 tiers (every kind of move), a fresh store and a fresh block per arm. Arms:
    /// N = 1 / 2 / 3 with drafts forced right or wrong by a pattern (the draft hook), N = 2 with
    /// every draft wrong, N = 1 with every draft right under `CROW_GLM_FLAGS` + `CROW_GLM_GRAPH`,
    /// and N = 2 with the synthetic block's own drafts. Every arm gives the ids and every logit's
    /// bits of the run without MTP; after every rollback the KDA states (S, conv) equal, bit for
    /// bit, the states the one-row path has after the same row (`Glm5Run::row`, hashed per row);
    /// at the end every KDA state and every MLA cache row `0 ..= last` equal the run without
    /// MTP; the counters equal the host simulation of the pattern.
    #[test]
    #[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_spec_gpu_is_lossless() {
        let g = geo();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt = [3i64, 17, 101, 999, 5];
        let (pn, n) = (prompt.len(), 70usize);
        let cap = pn + n;
        let last = pn + n - 2;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let old = std::env::var(mtp::MTP_ENV).ok();
        unsafe {
            let _ctx = cuda::Ctx::init();
            // the pass holds verify calls of 1 + 3 rows
            std::env::set_var(mtp::MTP_ENV, "3");
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, cap, &mut |s| eprintln!("{s}"));
            match &old {
                Some(o) => std::env::set_var(mtp::MTP_ENV, o),
                None => std::env::remove_var(mtp::MTP_ENV),
            }
            // the run without MTP, and the one-row door's KDA states after every row
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            let base = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |_| {}).unwrap();
            let (kda0, mla0) = states(&run, &g, last);
            tiers.free();
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            run.kda_states().for_each(|k| k.reset());
            let mut row_hash: HashMap<usize, u64> = HashMap::new();
            let mut door = Vec::new();
            let mut tok = 0i64;
            for pos in 0..=last {
                let input = if pos < pn { prompt[pos] } else { tok };
                if let Some(id) = run.row(&mut cnq, &mut tiers, input, pos, pos + 1 >= pn).unwrap() {
                    tok = id;
                    door.push(id);
                }
                row_hash.insert(pos, hash(&states(&run, &g, 0).0));
            }
            tiers.free();
            assert_eq!(door, base.ids, "the one-row door gives generate's ids");
            let finite = base.logits.iter().flatten().filter(|v| v.is_finite()).count();
            assert_eq!(finite, n * g.vocab, "the synthetic model must stay finite for the comparison to mean something");
            let gold = base.ids.clone();
            let vocab = g.vocab as i64;
            type Pat = fn(usize) -> bool;
            let arms: Vec<(&str, usize, Option<Pat>, bool)> = vec![
                ("N 1, right unless i % 3 == 2", 1, Some(|i| i % 3 != 2), false),
                ("N 2, right unless i % 4 == 1", 2, Some(|i| i % 4 != 1), false),
                ("N 3, right unless i % 5 == 3", 3, Some(|i| i % 5 != 3), false),
                ("N 2, every draft wrong", 2, Some(|_| false), false),
                ("N 1, every draft right, flags + graph", 1, Some(|_| true), true),
                ("N 2, the block's own drafts", 2, None, false),
            ];
            for (seed, (name, nd, pat, sw)) in arms.into_iter().enumerate() {
                let block = mtp::synthetic_block(&g, &moe, 0x0192_0000 + seed as u64);
                run.set_mtp(nd, Some(block)).unwrap();
                if let Some(p) = pat {
                    let gold = gold.clone();
                    run.set_mtp_hook(Some(Box::new(move |i, _| if p(i) { gold[i] } else { (gold[i] + 1) % vocab })));
                }
                let seen: Arc<Mutex<Vec<(usize, bool)>>> = Arc::default();
                let rh = row_hash.clone();
                let seen2 = seen.clone();
                run.set_mtp_probe(Some(Box::new(move |row, b| seen2.lock().unwrap().push((row, rh.get(&row) == Some(&hash(b)))))));
                if sw {
                    run.set_graph(true);
                    run.set_switches(&mut cnq, Switches { flags: true, lookahead: false, ..Switches::default() });
                }
                let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, (1 + nd) * g.topk).unwrap();
                let mut reps: Vec<TokenReport> = Vec::new();
                let out = run.generate(&mut cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                tiers.free();
                if sw {
                    run.set_graph(false);
                    run.set_switches(&mut cnq, Switches::default());
                }
                let st = run.mtp_stats().unwrap().clone();
                eprintln!("glm5_mtp spec {name}: {st:?}");
                assert_eq!(out.ids, base.ids, "{name}: ids");
                let diff: Vec<usize> = out.logits.iter().zip(&base.logits).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect();
                assert!(out.logits.len() == n && diff.iter().all(|&d| d == 0), "{name}: logits differ in bits per generated id {diff:?}");
                let (kda1, mla1) = states(&run, &g, last);
                assert!(kda1 == kda0, "{name}: the KDA states at the end differ from the run without MTP");
                assert!(mla1 == mla0, "{name}: the MLA cache rows 0..={last} differ from the run without MTP");
                // one report per row and id; the decode reports carry the emitted ids in order
                assert_eq!(reps.iter().map(|r| r.pos).collect::<Vec<_>>(), (0..=last).collect::<Vec<_>>(), "{name}: report positions");
                assert_eq!(reps.iter().filter_map(|r| r.next).collect::<Vec<_>>(), base.ids, "{name}: report ids");
                let seen = seen.lock().unwrap().clone();
                assert!(seen.iter().all(|x| x.1), "{name}: KDA states after a rollback differ from the one-row path at rows {:?}", seen.iter().filter(|x| !x.1).map(|x| x.0).collect::<Vec<_>>());
                assert_eq!(seen.len() as u64, st.kda_restore_steps, "{name}: probe calls = rollback steps");
                assert_eq!((st.tokens, st.verify_rows, st.hist.iter().sum::<u64>()), ((n - 1) as u64, st.steps + st.drafts, st.steps), "{name}: counter identities");
                assert_eq!(st.kda_snapshots, st.drafts * 6, "{name}: one snapshot per KDA layer (6) per draft row");
                match pat {
                    Some(p) => {
                        let (want, restores) = simulate(nd, pn, n, cap, &|i| p(i));
                        assert_eq!(
                            (st.steps, st.tokens, st.drafts, st.accepted, st.verify_rows, st.hist.clone(), st.kda_restore_steps),
                            (want.steps, want.tokens, want.drafts, want.accepted, want.verify_rows, want.hist.clone(), want.kda_restore_steps),
                            "{name}: counters vs the simulation"
                        );
                        assert_eq!(seen.iter().map(|x| x.0).collect::<Vec<_>>(), restores, "{name}: rollback rows");
                        assert!(want.drafts > 0, "{name}: the arm must draft");
                    }
                    None => assert!(st.drafts > 0 && st.accepted <= st.drafts, "{name}: {st:?}"),
                }
            }
            run.set_mtp(0, None).unwrap();
            run.free();
        }
        drop(cnq);
    }

    /// The block's cache rows `0 .. rows` of `run` (latent, indexer)
    unsafe fn block_rows(run: &Glm5Run, rows: usize) -> Vec<Vec<u8>> {
        let mut b = run.mtp_state_bytes(rows);
        b.truncate(2);
        b
    }

    /// The reference of the block over a prompt: the trunk's prompt calls of `chunk` rows with MTP
    /// off (`prefill_with`, every call's rows' head-norm rows read back), then the block (a fresh
    /// pass, the arm's seed) over windows of `MTP_CHUNK` rows from position 0 on (head-norm row,
    /// embedding of the next id; the last row's next id is the greedy id). Its cache rows
    /// `0 .. prompt.len()`.
    #[allow(clippy::too_many_arguments)]
    unsafe fn block_reference(run: &mut Glm5Run, cnq: &mut Cnq, path: &str, g: &Glm5Geo, moe: &MoeGeo, prompt: &[i64], chunk: usize, seed: u64) -> Vec<Vec<u8>> {
        let (h, pn) = (g.hidden, prompt.len());
        run.set_prompt_chunk(chunk).unwrap();
        run.kda_states().for_each(|k| k.reset());
        let mut tiers = ExpertTiers::new(cnq, path, g, moe, TierSizes { vram: 3, pinned: 4 }, 1, 4 * g.topk).unwrap();
        let mut hn = cuda::alloc_named("test head-norm rows", pn * h * 4);
        let row = g.hc_streams * h;
        let first = {
            let mut rows = |r: &mut Glm5Run, _: &mut Cnq, r0: usize, t: usize| -> Result<(), String> {
                for i in 0..t {
                    r.head.stream_mean_rms(r.x + (i * row * 4) as u64, r.hw.norm, hn + ((r0 + i) * h * 4) as u64, 1);
                }
                Ok(())
            };
            run.prefill_with(cnq, &mut tiers, prompt, 0, &mut |_| {}, &mut rows).unwrap()
        };
        tiers.free();
        let next: Vec<i64> = prompt[1..].iter().copied().chain(std::iter::once(first)).collect();
        let mut e = cuda::alloc_named("test embeddings", pn * h * 4);
        cuda::to_f32_into(e, &gm::embed_rows(cnq, g, &next));
        let mut block = mtp::synthetic_block(g, moe, seed);
        let mut mp = mtp::MtpPass::new(g, *moe, mtp::MTP_CHUNK, run.cap);
        let mut mk = mtp::MtpKernels::new();
        let mut p0 = 0;
        while p0 < pn {
            let t = mtp::MTP_CHUNK.min(pn - p0);
            block.call(&mut mp, &run.pass.kn, &mk, e + (p0 * h * 4) as u64, hn + (p0 * h * 4) as u64, p0, t, false, None);
            p0 += t;
        }
        cuda::sync();
        let md = MlaDims::of(g);
        let out = vec![cuda::dtoh_t::<u8>(mp.mla_c.latent, pn * md.latent_bytes_per_token() as usize), cuda::dtoh_t::<u8>(mp.mla_c.index, pn * md.indexer_bytes_per_token() as usize)];
        mp.free();
        block.free();
        mk.module.unload();
        cuda::free_dev(&mut hn);
        cuda::free_dev(&mut e);
        out
    }

    /// #192 gap 1: `CROW_GLM_MTP` with `CROW_CHUNK`. A 37-id prompt, 24 greedy ids, V 3 + P 4,
    /// a pass of 16 rows. Arms: N = 3 (drafts forced right unless i % 5 == 3) at prompt chunk
    /// 1 / 16, N = 2 (the block's own drafts) at chunk 12 (calls that cross the block's windows).
    /// Each chunked arm gives the ids and every logit's bits of MTP off at the same chunk and
    /// the ids of row by row; its prompt reports are the prompt calls' rows; the block's cache
    /// rows over the prompt equal, bit for bit, the block run in the row path's windows on the
    /// same chunked trunk's rows; the forced arms' counters equal the host simulation.
    #[test]
    #[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_spec_gpu_chunked_prompt_is_lossless() {
        let g = geo();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let prompt: Vec<i64> = (0..37).map(|i| (i * 131 + 7) % 2048).collect();
        let (pn, n) = (prompt.len(), 24usize);
        let cap = pn + n;
        let sizes = TierSizes { vram: 3, pinned: 4 };
        let (old_mtp, old_chunk) = (std::env::var(mtp::MTP_ENV).ok(), std::env::var("CROW_CHUNK").ok());
        let gen_with = |run: &mut Glm5Run, cnq: &mut Cnq, nd: usize| -> (Generated, Vec<TokenReport>) {
            unsafe {
                let mut tiers = ExpertTiers::new(cnq, &s.path, &g, &moe, sizes, 1, (1 + nd) * g.topk).unwrap();
                let mut reps = Vec::new();
                let out = run.generate(cnq, &mut tiers, &prompt, n, true, &mut |r| reps.push(r.clone())).unwrap();
                tiers.free();
                (out, reps)
            }
        };
        let bits = |a: &Generated, b: &Generated| -> Vec<usize> { a.logits.iter().zip(&b.logits).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect() };
        unsafe {
            let _ctx = cuda::Ctx::init();
            std::env::set_var(mtp::MTP_ENV, "3");
            std::env::set_var("CROW_CHUNK", "16");
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, cap, &mut |s| eprintln!("{s}"));
            for (k, o) in [(mtp::MTP_ENV, &old_mtp), ("CROW_CHUNK", &old_chunk)] {
                match o {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
            assert_eq!(run.prompt_chunk(), 16);
            // MTP off: row by row, chunk 16, chunk 12
            let mut base = HashMap::new();
            for c in [1usize, 16, 12] {
                run.set_prompt_chunk(c).unwrap();
                base.insert(c, gen_with(&mut run, &mut cnq, 0).0);
            }
            assert_eq!(base[&16].ids, base[&1].ids, "the synthetic run must give the row path's ids at chunk 16 for the comparison");
            let gold = base[&1].ids.clone();
            let vocab = g.vocab as i64;
            let pat: fn(usize) -> bool = |i| i % 5 != 3;
            type Arm = (&'static str, usize, usize, Option<fn(usize) -> bool>, u64);
            let arms: [Arm; 3] = [("N 3 forced, chunk 1", 3, 1, Some(pat), 0x0192_1001), ("N 3 forced, chunk 16", 3, 16, Some(pat), 0x0192_1001), ("N 2 own drafts, chunk 12", 2, 12, None, 0x0192_1002)];
            let mut stats = Vec::new();
            for (name, nd, chunk, p, seed) in arms {
                run.set_mtp(nd, Some(mtp::synthetic_block(&g, &moe, seed))).unwrap();
                if let Some(p) = p {
                    let gold = gold.clone();
                    run.set_mtp_hook(Some(Box::new(move |i, _| if p(i) { gold[i] } else { (gold[i] + 1) % vocab })));
                }
                run.set_prompt_chunk(chunk).unwrap();
                let (out, reps) = gen_with(&mut run, &mut cnq, nd);
                let st = run.mtp_stats().unwrap().clone();
                let prompt_rows: Vec<usize> = reps.iter().filter(|r| r.prompt).map(|r| r.rows).collect();
                eprintln!("glm5_mtp chunked {name}: prompt reports {prompt_rows:?}, {st:?}");
                let want = &base[&chunk];
                assert_eq!(out.ids, gold, "{name}: ids vs row by row without MTP");
                let d = bits(&out, want);
                assert!(out.logits.len() == n && d.iter().all(|&x| x == 0), "{name}: logits differ in bits from MTP off at chunk {chunk}: {d:?}");
                assert_eq!(prompt_rows, prompt_calls(pn, chunk).iter().map(|c| c.1).collect::<Vec<_>>(), "{name}: one prompt report per prompt call");
                let got = block_rows(&run, pn);
                if let Some(p) = p {
                    let (want_st, _) = simulate(nd, pn, n, cap, &|i| p(i));
                    assert_eq!((st.steps, st.tokens, st.drafts, st.accepted, st.hist.clone()), (want_st.steps, want_st.tokens, want_st.drafts, want_st.accepted, want_st.hist.clone()), "{name}: counters vs the simulation");
                } else {
                    assert!(st.drafts > 0, "{name}: {st:?}");
                }
                stats.push((name, got, seed, chunk));
                run.set_mtp(0, None).unwrap();
            }
            // the block over the prompt: the row path's windows on the same trunk rows, bit for bit
            for (name, got, seed, chunk) in &stats {
                let want = block_reference(&mut run, &mut cnq, &s.path, &g, &moe, &prompt, *chunk, *seed);
                assert!(got == &want, "{name}: the block's cache rows 0..{pn} differ from its windows over the trunk's rows at chunk {chunk}");
            }
            // informational: chunked trunk rows vs the row path's (#186: not bit-identical)
            let a: Vec<f32> = stats[0].1[0].chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            let b: Vec<f32> = stats[1].1[0].chunks_exact(2).map(|c| f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)).collect();
            let maxd = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).fold(0f32, f32::max);
            eprintln!("glm5_mtp chunked: block latent rows over the prompt, chunk 16 vs chunk 1: {} of {} BF16 values differ in bits, max abs {maxd:e}", a.iter().zip(&b).filter(|(x, y)| x.to_bits() != y.to_bits()).count(), a.len());
            run.free();
        }
        drop(cnq);
    }

    /// The CPU lane and too few staging slots are refused by name before any row runs; a model
    /// loaded for one row refuses `set_mtp(2, ..)`.
    #[test]
    #[ignore = "needs the GPU (about 3 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_mtp_spec_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_spec_gpu_refusals() {
        let g = geo();
        let s = synth_model(&g, REC);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let old_alloc = std::env::var("CROW_PINNED_ALLOC").ok();
        std::env::set_var("CROW_PINNED_ALLOC", "host");
        unsafe {
            let _ctx = cuda::Ctx::init();
            std::env::remove_var(mtp::MTP_ENV);
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, 16, &mut |s| eprintln!("{s}"));
            let e = run.set_mtp(2, Some(mtp::synthetic_block(&g, &moe, 1))).unwrap_err();
            assert!(e.starts_with("CROW_GLM_MTP=2: the model was loaded for 1 verify rows"), "{e}");
            run.free();
            std::env::set_var(mtp::MTP_ENV, "1");
            let mut run = Glm5Run::load(&mut cnq, &g, &moe, 16, &mut |s| eprintln!("{s}"));
            std::env::remove_var(mtp::MTP_ENV);
            run.set_mtp(1, Some(mtp::synthetic_block(&g, &moe, 2))).unwrap();
            let sizes = TierSizes { vram: 3, pinned: 4 };
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, g.topk).unwrap();
            let e = run.generate(&mut cnq, &mut tiers, &[3, 4], 4, false, &mut |_| {}).unwrap_err();
            assert!(e.starts_with("CROW_GLM_MTP=1: the tiers hold 8 staging slots, a verify of 2 rows may stage 16"), "{e}");
            tiers.free();
            let mut tiers = ExpertTiers::new(&cnq, &s.path, &g, &moe, sizes, 1, 2 * g.topk).unwrap();
            tiers.set_pinned_use(PinnedUse { stay: true, cpu_lane: true }).unwrap();
            let e = run.generate(&mut cnq, &mut tiers, &[3, 4], 4, false, &mut |_| {}).unwrap_err();
            assert!(e.starts_with("CROW_GLM_MTP=1 and CROW_GLM_CPU_LANE=1"), "{e}");
            tiers.free();
            run.set_mtp(0, None).unwrap();
            assert_eq!(run.mtp_drafts(), 0);
            run.free();
        }
        match old_alloc {
            Some(o) => std::env::set_var("CROW_PINNED_ALLOC", o),
            None => std::env::remove_var("CROW_PINNED_ALLOC"),
        }
        drop(cnq);
    }
}

// ---------------------------------------------------------------- the split planner's tests

#[cfg(test)]
mod split_tests {
    //! `CROW_GLM_CPU_LANE=split`: the switch, the planner (host), and on the GPU (`#[ignore]`) the
    //! split's posts on the synthetic container and the cost model's calibration bench.
    use super::*;
    use crate::cpu_mul1::{self, testkit, Mul1Expert, Path};
    use crate::geo::{ExpertCodec, ExpertRecordSpec};
    use crate::glm5_moe::{lane, GpuFfnWeights, GpuMoePlan, GpuMoeWeights, GpuNvfp4};
    use crate::kernels;
    use std::sync::Arc;

    // ------------------------------------------------------------ host

    /// `split` turns the lane on (implies zerocopy, like `1`) and is the only value that asks
    /// for the planner; with an explicit `promote`, with the stager, and any other value refused
    /// by name.
    #[test]
    fn the_split_switch_parses_and_refuses_by_name() {
        assert_eq!(pinned_use(None, Some("split")), Ok(PinnedUse { stay: true, cpu_lane: true }));
        assert_eq!(pinned_use(Some("zerocopy"), Some(" split ")), Ok(PinnedUse { stay: true, cpu_lane: true }));
        assert!(lane_split(Some("split")) && lane_split(Some(" split")));
        assert!(!lane_split(Some("1")) && !lane_split(Some("0")) && !lane_split(None));
        let e = pinned_use(Some("promote"), Some("split")).unwrap_err();
        assert!(e.starts_with("CROW_GLM_CPU_LANE=split reads the selected pinned experts where they lie"), "{e}");
        assert!(pinned_use(None, Some("splitt")).unwrap_err().ends_with("accepted 0 (default), 1, split"));
        assert_eq!(stager_on(Some("1"), Some("1"), Some("split")), Ok(true), "the split lane runs with the stager");
    }

    /// the layer time of `nc` CPU ids out of `n` pinned ids, by the planner's own accounting
    fn makespan(c: &SplitCost, hits: usize, n: usize, nc: usize) -> f64 {
        let g = c.g0 + c.thit * hits as f64 + c.tzc * (n - nc) as f64;
        let cpu = if n > 0 { c.ca + c.cb * nc as f64 } else { 0.0 };
        g.max(cpu)
    }

    /// The greedy split is the min-max: over 20,000 random cost models, VRAM hits 0..8 and
    /// pinned ids 0..8 (caps 1..32), the plan's layer time equals the least over every count of
    /// CPU ids the cap allows (identical ids, so the count is the whole choice).
    #[test]
    fn plan_split_is_the_min_max_of_every_split() {
        let mut rng = cpu_mul1::testkit::Rng(0x5917);
        let mut u = |lo: f64, hi: f64| lo + (rng.next() % 1_000_000) as f64 / 1e6 * (hi - lo);
        for i in 0..20_000 {
            let c = SplitCost { g0: u(0.0, 0.3), thit: u(0.0, 0.05), tzc: u(0.05, 0.6), ca: u(0.0, 0.5), cb: u(0.05, 1.2), maxcpu: 1 + i % 32 };
            let (hits, n) = (i % 9, (i / 9) % 9);
            let ram: Vec<(u32, u32)> = (0..n as u32).map(|e| (e * 3, e % 3)).collect();
            let got = plan_split(&c, hits, &ram).len();
            assert!(got <= c.maxcpu.min(n));
            let best = (0..=c.maxcpu.min(n)).map(|nc| makespan(&c, hits, n, nc)).fold(f64::MAX, f64::min);
            assert!(makespan(&c, hits, n, got) <= best + 1e-12, "{c:?} hits {hits} n {n}: plan {got} CPU ids, {} ms vs best {best} ms", makespan(&c, hits, n, got));
        }
    }

    /// The ids are considered colder first (lower heat, ties to the lower id), so the coldest
    /// goes to the CPU when any does; at most `maxcpu`; no pinned id, no CPU id.
    #[test]
    fn plan_split_sends_colder_ids_first_up_to_maxcpu() {
        let c = SplitCost { g0: 0.1, thit: 0.01, tzc: 0.25, ca: 0.05, cb: 0.3, maxcpu: 32 };
        let ram = [(5u32, 9u32), (7, 1), (2, 1), (9, 4), (11, 30)];
        // considered 2, 7, 9, 5, 11 (heat 1, 1, 4, 9, 30); each to the lane that keeps the max
        // lower: CPU, GPU, GPU, CPU, GPU (CPU 0.05 + 2 x 0.3 = 0.65, GPU 0.1 + 3 x 0.25 = 0.85)
        assert_eq!(plan_split(&c, 0, &ram), vec![2, 5]);
        assert_eq!(plan_split(&SplitCost { maxcpu: 1, ..c }, 0, &ram), vec![2]);
        assert_eq!(plan_split(&SplitCost { maxcpu: 0, ..c }, 0, &ram), Vec::<u32>::new());
        assert_eq!(plan_split(&c, 6, &[]), Vec::<u32>::new());
        // a CPU much faster than PCIe takes every pinned id, coldest first
        assert_eq!(plan_split(&SplitCost { cb: 0.01, ..c }, 0, &ram), vec![2, 7, 9, 5, 11]);
        // a CPU much slower takes none
        assert_eq!(plan_split(&SplitCost { cb: 5.0, ..c }, 0, &ram), Vec::<u32>::new());
    }

    /// The calibrated models plan the splits the bench measured best (confirmation run
    /// 2026-10-10, CPU ids for V 0 / 2 / 4 / 6 of 8 picks): 8 threads 3 / 2 / 1 / 1, 12 threads
    /// 3 / 2 / 2 / 1, 16 and 20 threads 4 / 3 / 2 / 1; the split's default is the 20-thread model.
    #[test]
    fn the_calibrated_cost_model_plans_the_measured_best_splits() {
        let want = [(8usize, [3usize, 2, 1, 1]), (12, [3, 2, 2, 1]), (16, [4, 3, 2, 1]), (20, [4, 3, 2, 1])];
        for (threads, best) in want {
            let c = SplitCost::for_threads(threads);
            for (hits, nc) in [0usize, 2, 4, 6].into_iter().zip(best) {
                let ram: Vec<(u32, u32)> = (0..(8 - hits) as u32).map(|e| (e, 0)).collect();
                assert_eq!(plan_split(&c, hits, &ram).len(), nc, "{threads} threads V {hits}");
            }
        }
        assert_eq!(SplitCost::RTX5090_285K, SplitCost::for_threads(SPLIT_LANE_THREADS));
        assert_eq!(SPLIT_LANE_THREADS, 20);
        assert_eq!(SplitCost::for_threads(1), SplitCost::for_threads(8));
        assert_eq!(SplitCost::for_threads(14), SplitCost::for_threads(12));
        assert_eq!(SplitCost::for_threads(17), SplitCost::for_threads(16));
        assert_eq!(SplitCost::for_threads(24), SplitCost::for_threads(20));
    }

    /// `CROW_GLM_LANE_THREADS`: unset keeps the #188 lane's 8 threads, and gives `split` its
    /// measured default (20); a whole number 1..=256 is taken as is; anything else refused by name.
    #[test]
    fn the_lane_threads_switch_parses_and_refuses_by_name() {
        assert_eq!(lane_threads(None, None), Ok(8));
        assert_eq!(lane_threads(None, Some("1")), Ok(8));
        assert_eq!(lane_threads(Some(""), Some("split")), Ok(20));
        assert_eq!(lane_threads(Some(" 12 "), Some("split")), Ok(12));
        assert_eq!(lane_threads(Some("16"), Some("1")), Ok(16));
        assert_eq!(lane_threads(Some("256"), None), Ok(256));
        for bad in ["0", "257", "-4", "twenty", "1.5"] {
            let e = lane_threads(Some(bad), Some("split")).unwrap_err();
            assert!(e.starts_with(&format!("CROW_GLM_LANE_THREADS={bad:?}: accepted a whole number")), "{e}");
        }
        assert_eq!(crate::glm5_moe::lane::threads(), crate::glm5_moe::LANE_THREADS, "nothing stored: the lane keeps 8");
    }

    /// `lane_combos_where`: a pinned id the plan refuses stays a GPU combo with its table entry;
    /// `lane_combos` (`1`) still sends every pinned id to the CPU.
    #[test]
    fn lane_combos_where_keeps_refused_pinned_ids_on_the_gpu() {
        use crate::glm5_moe::lane::Combo;
        let sel = [7, 3, 250, 0, 12];
        let locs = [(0u32, Loc::Vram(2)), (3, Loc::Pinned(5)), (7, Loc::Stage(1)), (12, Loc::Pinned(1)), (250, Loc::Pinned(0))];
        let host = 0x1000 as *const u8;
        let gpu = |e: u32, _| 0x9000 + e as u64;
        let cpu = |q: u32| host.wrapping_add(q as usize * 16);
        let (c, n) = lane_combos_where(&sel, &locs, gpu, cpu, |e| e == 250 || e == 12);
        assert_eq!(n, 2);
        assert_eq!(c, vec![Combo::Gpu(0x9007), Combo::Gpu(0x9003), Combo::Cpu(host), Combo::Gpu(0x9000), Combo::Cpu(host.wrapping_add(16))]);
        let (c, n) = lane_combos(&sel, &locs, gpu, cpu);
        assert_eq!(n, 3);
        assert_eq!(c[1], Combo::Cpu(host.wrapping_add(80)));
    }

    // ------------------------------------------------------------ GPU

    /// `split` on the synthetic container (as `glm5_tiers_gpu_zerocopy_and_cpu_lane_tables_hold_
    /// their_records`, `CROW_PINNED_ALLOC=host`): at four capacities, over a routing trace, every
    /// decode call posts exactly the CPU combos `plan_split` gives for the call's VRAM hits and
    /// pinned ids at the heat before the call (none: no post), each a host pointer to the
    /// record's bytes, every other combo the id's table entry; `cpu_lane` + `zero_copy` = the
    /// pinned visits; the heat counts every selection. A cost model with a free CPU takes every
    /// pinned id (the `1` lane's posts).
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_tiers_gpu_split -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_split_posts_the_planned_combos() {
        use lane::Combo;
        let s = super::tests::synth_glm(16);
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk) = (4, 3, 16, 8);
        let (spec, _) = crate::nvme_source::glm5_record_of_container(&s.path).unwrap();
        let moe = MoeGeo::new(&g, spec).unwrap();
        let mut cnq = Cnq::open_checked(&s.path).unwrap();
        let rb = spec.bytes as usize;
        let want: Vec<Vec<u8>> = (0..16).map(|e| cnq.read_range(&cnq.find(&crate::nvme_source::glm5_expert_tensor_name(3, e, "gate"), "text").clone(), 0, rb)).collect();
        let tr = super::tests::trace(24, 1, 16, 8, 0x5917);
        let old = std::env::var("CROW_PINNED_ALLOC").ok();
        std::env::set_var("CROW_PINNED_ALLOC", "host");
        unsafe {
            let _ctx = cuda::Ctx::init();
            for (v, p) in [(1, 7), (3, 4), (4, 12), (0, 16)] {
                for (name, cost) in [("calibrated", SplitCost::RTX5090_285K), ("free CPU", SplitCost { cb: 1e-6, ..SplitCost::RTX5090_285K })] {
                    let what0 = format!("{name} V {v} P {p}");
                    let mut t = ExpertTiers::new(&cnq, &s.path, &g, &moe, TierSizes { vram: v, pinned: p }, 1, 8).unwrap();
                    t.set_pinned_use(PinnedUse { stay: true, cpu_lane: true }).unwrap();
                    t.split = Some(cost);
                    let (mut on_cpu, mut on_gpu) = (0u64, 0u64);
                    for (i, tok) in tr.iter().enumerate() {
                        let what = format!("{what0} call {i}");
                        let sel: Vec<i32> = tok[0].iter().rev().map(|&e| e as i32).collect();
                        let heat = t.heat.clone();
                        let (tb, served) = t.table_for(3, &sel).unwrap();
                        let posted = lane::take(tb, 1, 8);
                        let pinned: Vec<(u32, u32)> = served.locs.iter().filter(|x| matches!(x.1, Loc::Pinned(_))).map(|x| (x.0, heat[x.0 as usize])).collect();
                        let plan = plan_split(&cost, served.locs.len() - pinned.len(), &pinned);
                        if name == "free CPU" {
                            assert_eq!(plan.len(), pinned.len(), "{what}: a free CPU takes every pinned id");
                        }
                        assert_eq!((served.moves.cpu_lane, served.moves.zero_copy), (plan.len() as u64, (pinned.len() - plan.len()) as u64), "{what}: counters");
                        for &e in &tok[0] {
                            assert_eq!(t.heat[e as usize], heat[e as usize] + 1, "{what}: heat of {e}");
                        }
                        cuda::sync();
                        let table = cuda::dtoh_u64(tb, 16);
                        match posted {
                            None => assert!(plan.is_empty(), "{what}: planned CPU ids {plan:?} but no post"),
                            Some(call) => {
                                for (c, combo) in call.combos.iter().enumerate() {
                                    let e = sel[c] as usize;
                                    match *combo {
                                        Combo::Cpu(ptr) => {
                                            assert!(plan.contains(&(e as u32)), "{what}: CPU combo {c} (expert {e}) not planned");
                                            assert!(std::slice::from_raw_parts(ptr, rb) == &want[e][..], "{what}: CPU combo {c} (expert {e}) reads other bytes");
                                        }
                                        Combo::Gpu(base) => {
                                            assert!(!plan.contains(&(e as u32)), "{what}: planned expert {e} on the GPU");
                                            assert_eq!(base, table[e], "{what}: GPU combo {c} (expert {e})");
                                        }
                                    }
                                }
                                assert_eq!(call.combos.iter().filter(|x| matches!(x, Combo::Cpu(_))).count(), plan.len(), "{what}: CPU combos");
                            }
                        }
                        on_cpu += plan.len() as u64;
                        on_gpu += (pinned.len() - plan.len()) as u64;
                    }
                    assert_eq!((t.moves[0].cpu_lane, t.moves[0].zero_copy), (on_cpu, on_gpu), "{what0}: summed counters");
                    eprintln!("glm5_tiers split {what0}: pinned visits {} CPU {on_cpu} zero-copy {on_gpu}", on_cpu + on_gpu);
                    t.free();
                }
            }
        }
        match old {
            Some(o) => std::env::set_var("CROW_PINNED_ALLOC", o),
            None => std::env::remove_var("CROW_PINNED_ALLOC"),
        }
        drop(cnq);
    }

    // ------------------------------------------------------------ bench helpers

    fn median(mut v: Vec<f64>) -> f64 {
        v.sort_by(f64::total_cmp);
        v[v.len() / 2]
    }

    /// least squares `y = a + b x`
    fn line(pts: &[(f64, f64)]) -> (f64, f64) {
        let m = pts.len() as f64;
        let (sx, sy) = (pts.iter().map(|p| p.0).sum::<f64>(), pts.iter().map(|p| p.1).sum::<f64>());
        let (sxx, sxy) = (pts.iter().map(|p| p.0 * p.0).sum::<f64>(), pts.iter().map(|p| p.0 * p.1).sum::<f64>());
        let b = (m * sxy - sx * sy) / (m * sxx - sx * sx);
        ((sy - b * sx) / m, b)
    }

    /// the layer time the model gives a split: `k + max(gpu, cpu)` (no CPU lane: `k + gpu`)
    fn model(c: &SplitCost, v: usize, nc: usize, k: usize) -> f64 {
        let g = c.g0 + c.thit * v as f64 + c.tzc * (k - v - nc) as f64;
        let cpu = if nc > 0 { c.ca + c.cb * nc as f64 } else { 0.0 };
        g.max(cpu)
    }

    /// Calibration bench of [`SplitCost`], not a gate. One synthetic GLM MoE layer (zero router
    /// and bias: the picks are experts 0..7; zero shared expert), 32 distinct 3-bit records in
    /// cacheable pinned RAM (303 MB) and 16 in VRAM, rotated call by call so neither the L3 nor
    /// the GPU's 96 MB L2 holds them. (1) Solo GPU: `mul1::FfnPlan` over n = 1..8 records from
    /// VRAM and from pinned RAM (zero-copy), lines `a + thit n`, `a + tzc n`. (2) Solo CPU: the
    /// lane's `experts_ffn` (clamped SwiGLU) over n = 1..8 pinned records. (3) The real lane
    /// (`GpuMoePlan::run` with a `lane::post`, (2) and (3) at lane threads 8, 12, 16, 20 through
    /// `lane::THREADS`): every split of V VRAM picks (0, 2, 4, 6)
    /// and nc CPU picks (the rest zero-copy), wall time per layer call (route + experts + shared
    /// + combine + sync); fit of `k + max(g0 + thit V + tzc Z, ca + cb nc)` with `thit` from (1)
    /// over a grid of (g0, tzc, ca, cb), and per V the best nc measured vs planned by the fit and
    /// by the shipped [`SplitCost::for_threads`]. (4) The same
    /// split emulated with the CPU at 8, 16, 20 threads (FfnPlan zero-copy + `experts_ffn`
    /// concurrently): what a larger `LANE_THREADS` would give. Median of 25 calls per point.
    #[test]
    #[ignore = "bench, needs the GPU (about 0.4 GB VRAM, 0.3 GB pinned): cargo test --release --lib glm5_tiers_gpu_split_cost_bench -- --ignored --nocapture --test-threads 1"]
    fn glm5_tiers_gpu_split_cost_bench() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let module = cuda::compile(&kernels::KernelGeo::flash_next().source());
            let kn = kernels::Kernels::new(&module, false);
            let mk = kernels::mul1::Kernels::new();
            let gk = kernels::glm5_moe::Kernels::new();
            let g = MoeGeo::new(&Glm5Geo::GLM_5_3_FLASH, ExpertRecordSpec::new(ExpertCodec::Mul1, cpu_mul1::GLM_RECORD_BYTES_K3 as u64).unwrap()).unwrap();
            let (h, k, rb, inter) = (g.hidden, g.topk, g.record.bytes as usize, g.expert_inter);
            let c = &testkit::glm_cases()[0];
            let base = testkit::record(c);
            let tb = 3 * cpu_mul1::Mul1Matrix::trellis_bytes(c.hidden, c.inter, c.bitrate);
            let rec = |i: u32| {
                let mut r = base.clone();
                r[..tb].iter_mut().for_each(|b| *b = b.rotate_left(i % 8) ^ ((i / 8) as u8).wrapping_mul(37));
                r
            };
            const NP: usize = 32;
            const NV: usize = 16;
            let mut pinned = cuda::Pinned::alloc(NP * rb);
            for i in 0..NP {
                pinned.write_bytes(i * rb, &rec(i as u32));
            }
            let mut vram = cuda::alloc_zeroed(NV * rb);
            for i in 0..NV {
                cuda::upload_into(vram + (i * rb) as u64, &rec(100 + i as u32));
            }
            let (pdev, phost) = (pinned.dev, pinned.host as *const u8);
            let mut rng = testkit::Rng(0x5917);
            let x: Vec<f32> = (0..8 * h).map(|_| rng.f(0.17)).collect();
            let act = |a: f32, b: f32| crate::glm5_moe::swiglu_clamp(a, b, 10.0);
            const REPS: usize = 28;
            const WARM: usize = 3;
            eprintln!("glm5_tiers split bench: {} B per record, {NP} pinned + {NV} VRAM records", rb);

            // (1) solo GPU
            let (mut xd, mut yd, mut ptrs) = (cuda::to_f32_dev(&x), cuda::alloc_zeroed(8 * h * 4), cuda::alloc_zeroed(64));
            let mut plans: Vec<kernels::mul1::FfnPlan> = (1..=8).map(|n| kernels::mul1::FfnPlan::new(h, inter, 3, false, n, 1)).collect();
            let mut at = 0usize;
            let mut gpu_solo = |from_vram: bool| -> Vec<(f64, f64)> {
                let mut ms = vec![Vec::new(); 8];
                for rep in 0..REPS + WARM {
                    for n in 1..=8usize {
                        let bases: Vec<u64> = (0..n)
                            .map(|j| {
                                at += 1;
                                if from_vram { vram + ((at + j) % NV * rb) as u64 } else { pdev + ((at + j) % NP * rb) as u64 }
                            })
                            .collect();
                        cuda::to_u64_into(ptrs, &bases);
                        cuda::sync();
                        let t0 = std::time::Instant::now();
                        plans[n - 1].run(&mk, ptrs, xd, yd);
                        cuda::sync();
                        if rep >= WARM {
                            ms[n - 1].push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                }
                ms.into_iter().enumerate().map(|(i, v)| ((i + 1) as f64, median(v))).collect()
            };
            let (pv, pz) = (gpu_solo(true), gpu_solo(false));
            let ((av, thit), (az, tzc_solo)) = (line(&pv), line(&pz));
            let row = |p: &[(f64, f64)]| p.iter().map(|&(n, t)| format!("{n:.0}:{t:.3}")).collect::<Vec<_>>().join(" ");
            eprintln!("  (1) GPU VRAM    ms per call {}; fit {av:.3} + {thit:.4} n", row(&pv));
            eprintln!("  (1) GPU pinned  ms per call {}; fit {az:.3} + {tzc_solo:.4} n ({:.1} GB/s)", row(&pz), rb as f64 / tzc_solo / 1e6);

            // (2) solo CPU at the lane's thread count
            let ex = |slot: usize| Mul1Expert::from_record(std::slice::from_raw_parts(phost.add(slot % NP * rb), rb), h, inter, g.bitrate).unwrap();
            let mut ys = vec![0f32; 8 * h];
            let mut cpu_solo = |threads: usize| -> Vec<(f64, f64)> {
                let mut at = 1000usize;
                let mut ms = vec![Vec::new(); 8];
                for rep in 0..REPS + WARM {
                    for n in 1..=8usize {
                        let es: Vec<Mul1Expert> = (0..n).map(|j| ex(at + j)).collect();
                        at += n;
                        let t0 = std::time::Instant::now();
                        cpu_mul1::experts_ffn(&es, &x[..h], &mut ys[..n * h], &act, threads, Path::Auto);
                        if rep >= WARM {
                            ms[n - 1].push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                }
                ms.into_iter().enumerate().map(|(i, v)| ((i + 1) as f64, median(v))).collect()
            };
            // (3) the real lane
            let shared = |rows: usize, cols: usize| GpuNvfp4 { w: cuda::alloc_zeroed(crate::cpu_nvfp4::Nvfp4Matrix::byte_len(rows, cols)), gs: cuda::to_f32_dev(&[0.3]), rows, cols };
            let w = GpuMoeWeights {
                router: cuda::alloc_zeroed(g.experts * h * 2),
                bias: cuda::alloc_zeroed(g.experts * 4),
                shared: GpuFfnWeights { gate: shared(g.shared_inter, h), up: shared(g.shared_inter, h), down: shared(h, g.shared_inter) },
            };
            let mut plan = GpuMoePlan::new(&g, 1);
            let mut table = cuda::alloc_zeroed(g.experts * 8);
            let clock = Arc::new(lane::Clock::default());
            let splits: Vec<(usize, usize)> = [0usize, 2, 4, 6].iter().flat_map(|&v| (0..=k - v).map(move |nc| (v, nc))).collect();
            let mut ac = 0.0;
            let mut cost = SplitCost::RTX5090_285K;
            for &threads in &[8usize, 12, 16, 20] {
            lane::THREADS.store(threads, std::sync::atomic::Ordering::Relaxed);
            let pc = cpu_solo(threads);
            let cb_solo;
            (ac, cb_solo) = line(&pc);
            eprintln!("  (2) CPU {threads:2} thr  ms per call {}; fit {ac:.3} + {cb_solo:.4} n", row(&pc));
            let mut ms = vec![Vec::new(); splits.len()];
            for rep in 0..REPS + WARM {
                for (i, &(v, nc)) in splits.iter().enumerate() {
                    // picks 0..nc on the CPU, nc..nc + v from VRAM, the rest zero-copy
                    let mut tab = vec![0u64; g.experts];
                    let mut combos = Vec::with_capacity(k);
                    for (e, t) in tab.iter_mut().enumerate().take(k) {
                        at += 1;
                        let (ps, vs) = (at % NP, at % NV);
                        *t = if (nc..nc + v).contains(&e) { vram + (vs * rb) as u64 } else { pdev + (ps * rb) as u64 };
                        combos.push(if e < nc { lane::Combo::Cpu(phost.add(ps * rb)) } else { lane::Combo::Gpu(*t) });
                    }
                    cuda::to_u64_into(table, &tab);
                    lane::post((nc > 0).then(|| lane::Call { table, combos, clock: clock.clone(), ready: None }));
                    cuda::sync();
                    let t0 = std::time::Instant::now();
                    plan.run(&kn, &mk, &gk, &w, table, xd, yd);
                    cuda::sync();
                    if rep >= WARM {
                        ms[i].push(t0.elapsed().as_secs_f64() * 1e3);
                    }
                }
            }
            assert_eq!(cuda::dtoh_i32(plan.ids, k), (0..k as i32).collect::<Vec<_>>(), "the zero router picks experts 0..7");
            let meas: Vec<f64> = ms.into_iter().map(median).collect();
            // fit: thit from (1); grid over g0, tzc, ca, cb; k = mean residual
            let mut best = (f64::MAX, SplitCost { g0: 0.0, thit, tzc: 0.0, ca: 0.0, cb: 0.0, maxcpu: 32 }, 0.0);
            for gi in 0..=40 {
                for ti in 0..=40 {
                    for ci in 0..=40 {
                        for bi in 0..=40 {
                            let cost = SplitCost { g0: gi as f64 * 0.01, thit, tzc: tzc_solo * (0.6 + ti as f64 * 0.025), ca: ci as f64 * 0.01, cb: cb_solo * (0.6 + bi as f64 * 0.025), maxcpu: 32 };
                            let r: Vec<f64> = splits.iter().zip(&meas).map(|(&(v, nc), &t)| t - model(&cost, v, nc, k)).collect();
                            let kk = r.iter().sum::<f64>() / r.len() as f64;
                            let sse = r.iter().map(|x| (x - kk) * (x - kk)).sum::<f64>();
                            if sse < best.0 {
                                best = (sse, cost, kk);
                            }
                        }
                    }
                }
            }
            let (sse, fit, kk) = best;
            cost = fit;
            eprintln!(
                "  (3) {threads:2} thr lane fit: g0 {:.3} thit {:.4} tzc {:.4} ca {:.3} cb {:.4} (+ common {kk:.3}), rms residual {:.3} ms",
                cost.g0,
                cost.thit,
                cost.tzc,
                cost.ca,
                cost.cb,
                (sse / splits.len() as f64).sqrt()
            );
            for &v in &[0usize, 2, 4, 6] {
                let pts: Vec<(usize, f64)> = splits.iter().zip(&meas).filter(|x| x.0 .0 == v).map(|(&(_, nc), &t)| (nc, t)).collect();
                let best_nc = pts.iter().min_by(|a, b| a.1.total_cmp(&b.1)).unwrap().0;
                let ram: Vec<(u32, u32)> = (0..(k - v) as u32).map(|e| (e, 0)).collect();
                let planned = plan_split(&cost, v, &ram).len();
                let shipped = plan_split(&SplitCost::for_threads(threads), v, &ram).len();
                let t_of = |nc: usize| pts.iter().find(|p| p.0 == nc).unwrap().1;
                eprintln!(
                    "  (3) {threads:2} thr V {v}: ms by nc {}; best nc {best_nc} ({:.3} ms), fit plans {planned} ({:.3} ms), shipped plans {shipped} ({:.3} ms), all-GPU {:.3}, all-CPU {:.3}",
                    pts.iter().map(|&(nc, t)| format!("{nc}:{t:.3}")).collect::<Vec<_>>().join(" "),
                    t_of(best_nc),
                    t_of(planned),
                    t_of(shipped),
                    t_of(0),
                    t_of(k - v)
                );
            }
            }
            lane::THREADS.store(0, std::sync::atomic::Ordering::Relaxed);

            // (4) emulated split with more CPU threads (V 0)
            for threads in [8usize, 16, 20] {
                let mut ms = vec![Vec::new(); k + 1];
                for rep in 0..REPS + WARM {
                    for nc in 0..=k {
                        let ng = k - nc;
                        let bases: Vec<u64> = (0..ng).map(|j| pdev + ((at + j) % NP * rb) as u64).collect();
                        at += ng;
                        let es: Vec<Mul1Expert> = (0..nc).map(|j| ex(at + j)).collect();
                        at += nc;
                        if ng > 0 {
                            cuda::to_u64_into(ptrs, &bases);
                        }
                        cuda::sync();
                        let t0 = std::time::Instant::now();
                        if ng > 0 {
                            plans[ng - 1].run(&mk, ptrs, xd, yd);
                            let _ = cudarc::driver::sys::cuStreamQuery(cuda::cur_stream());
                        }
                        if nc > 0 {
                            cpu_mul1::experts_ffn(&es, &x[..h], &mut ys[..nc * h], &act, threads, Path::Auto);
                        }
                        cuda::sync();
                        if rep >= WARM {
                            ms[nc].push(t0.elapsed().as_secs_f64() * 1e3);
                        }
                    }
                }
                let m: Vec<f64> = ms.into_iter().map(median).collect();
                let (bi, bt) = m.iter().enumerate().min_by(|a, b| a.1.total_cmp(b.1)).unwrap();
                eprintln!(
                    "  (4) {threads:2} CPU threads, V 0: ms by nc {}; best nc {bi} ({bt:.3} ms) vs all-GPU {:.3} ms ({:+.1} %)",
                    m.iter().enumerate().map(|(nc, t)| format!("{nc}:{t:.3}")).collect::<Vec<_>>().join(" "),
                    m[0],
                    (bt / m[0] - 1.0) * 100.0
                );
            }

            // (5) the lane's hand-off of x as `experts_lane` does it: D2H of the [hidden] row into
            // pinned RAM, an event, the host waits on the event (from an idle stream)
            {
                use cudarc::driver::sys;
                let mut buf = cuda::Pinned::alloc(h * 4);
                let ev = cuda::event_create();
                let s = cuda::cur_stream();
                let mut us = Vec::new();
                for rep in 0..210 {
                    cuda::sync();
                    let t0 = std::time::Instant::now();
                    cuda::ck(sys::cuMemcpyDtoHAsync_v2(buf.host, xd, h * 4, s));
                    cuda::event_record(ev, s);
                    let _ = sys::cuStreamQuery(s);
                    cuda::ck(sys::cuEventSynchronize(ev));
                    if rep >= 10 {
                        us.push(t0.elapsed().as_secs_f64() * 1e6);
                    }
                }
                eprintln!("  (5) x hand-off (16 KB D2H + event wait): median {:.1} us; lane fixed cost ca {:.3} ms vs solo CPU run {ac:.3} ms", median(us), cost.ca);
                cuda::event_destroy(ev);
                buf.free();
            }

            plan.free();
            for p in plans.iter_mut() {
                p.free();
            }
            for mut d in [w.router, w.bias, w.shared.gate.w, w.shared.gate.gs, w.shared.up.w, w.shared.up.gs, w.shared.down.w, w.shared.down.gs] {
                cuda::free_dev(&mut d);
            }
            for d in [&mut vram, &mut table, &mut xd, &mut yd, &mut ptrs] {
                cuda::free_dev(d);
            }
            pinned.free();
        }
    }
}

// ---------------------------------------------------------------- CROW_GLM_MAX_BATCH: sequence slots

/// `CROW_GLM_MAX_BATCH=N` (serve): the sequences the model holds at once and one batched decode
/// step carries (unset = 1: the one sequence, the path of record)
pub const MAX_BATCH_ENV: &str = "CROW_GLM_MAX_BATCH";
/// the most sequence slots (the template's `--max-batch-size 8`)
pub const MAX_BATCH: usize = 8;

/// `CROW_GLM_MAX_BATCH`'s value: unset or empty = 1; `1 ..= MAX_BATCH`; anything else by name
pub fn max_batch_of(v: Option<&str>) -> Result<usize, String> {
    match v.map(str::trim).filter(|s| !s.is_empty()) {
        None => Ok(1),
        Some(s) => match s.parse::<usize>() {
            Ok(n) if (1..=MAX_BATCH).contains(&n) => Ok(n),
            _ => Err(format!("{MAX_BATCH_ENV}={s}: the sequences one decode step carries, 1 ..= {MAX_BATCH} (unset = 1, one sequence)")),
        },
    }
}

/// [`max_batch_of`] on the environment
pub fn max_batch_from_env() -> Result<usize, String> {
    max_batch_of(std::env::var(MAX_BATCH_ENV).ok().as_deref())
}

/// Device bytes `n` sequence slots add to the one sequence of `Glm5Run::load` (derived, as
/// `glm5_mtp::spec_vram_bytes`): per further slot a KDA state per KDA layer, an MLA cache of
/// `cap` rows per DSA layer and a logits row; the batch head (`[n]` normed, logits and ids
/// rows); the residual, collapsed and sublayer rows of an `n`-row call; the MoE plan of an
/// `n`-row call. 0 for `n <= 1`.
pub fn batch_vram_bytes(g: &Glm5Geo, cap: usize, n: usize) -> u64 {
    if n <= 1 {
        return 0;
    }
    let (kd, md) = (KdaDims::of(g), MlaDims::of(g));
    let kda = (0..g.layers).filter(|&l| gm::attn_kind(g, l) == AttnKind::Kda).count() as u64;
    let mla = (0..g.layers).filter(|&l| gm::attn_kind(g, l) == AttnKind::Mla).count() as u64;
    let (h, v, b) = (g.hidden as u64, g.vocab as u64, n as u64);
    let slot = kda * ((kd.state_floats() + kd.conv_floats()) * 4) as u64 + mla * MlaCache::bytes(&md, cap) + 4 * v;
    let head = 4 * b * (h + v + 1);
    let rows = 4 * b * (g.hc_streams as u64 * h + 2 * h);
    let plan = 4 * b * g.topk as u64 * (2 * h + 3 * g.expert_inter as u64) * 2;
    (b - 1) * slot + head + rows + plan
}

/// `CROW_GLM_MAX_BATCH` (serve): sequence slots and the batched decode step. Slot `cur` is the
/// sequence every one-sequence path of the run acts on (`row`, `prefill`, the KDA states and
/// MLA caches, the logits row); [`Glm5Run::use_slot`] parks it and brings another in by
/// swapping the device pointers (no copy). [`Glm5Run::decode_batch`] runs one decode row of
/// several slots in one trunk pass, each row the bits of that slot's one-row decode.
impl Glm5Run {
    /// `n` sequence slots from now on (1 = the one sequence; the others' states are freed and
    /// slot 0 stays). Refused by name: more than the pass's `max_t` rows (`CROW_GLM_MAX_BATCH`
    /// sizes it at `load`), MTP (`CROW_GLM_MTP`: its block and drafts are one sequence's) and
    /// the graphs (`CROW_GLM_GRAPH`: a captured row holds slot 0's state pointers).
    ///
    /// # Safety
    /// A CUDA context is current; no launch of this run is pending.
    pub unsafe fn set_slots(&mut self, n: usize) -> Result<(), String> {
        if n == 0 || n > MAX_BATCH {
            return Err(format!("{MAX_BATCH_ENV}={n}: 1 ..= {MAX_BATCH} sequence slots"));
        }
        self.use_slot(0)?;
        cuda::sync();
        self.free_slots();
        if n == 1 {
            return Ok(());
        }
        if self.spec.is_some() {
            return Err(format!("{MAX_BATCH_ENV}={n} and {}={}: the MTP block, its cache and drafts are one sequence's; turn one of them off", crate::glm5_mtp::MTP_ENV, self.mtp_drafts()));
        }
        if self.graph.is_some() {
            return Err(format!("{MAX_BATCH_ENV}={n} and {}=1: a captured row holds slot 0's KDA states and MLA caches; turn one of them off", glm5_graph::ENV));
        }
        if n > self.pass.max_t {
            return Err(format!("{MAX_BATCH_ENV}={n}: the model was loaded for calls of {} rows; set {MAX_BATCH_ENV} before Glm5Run::load", self.pass.max_t));
        }
        let (g, kd, md) = (self.g, KdaDims::of(&self.g), MlaDims::of(&self.g));
        // slot 0 is the run's own sequence: its entry is the placeholder while it is current
        self.seqs.push(SeqSlot::default());
        for _ in 1..n {
            let kda = self.kda.iter().map(|k| k.as_ref().map(|_| KdaState::alloc(&kd))).collect();
            let mla = self.mla.iter().map(|c| c.as_ref().map(|_| MlaCache::new(&md, self.cap))).collect();
            let logits = cuda::alloc_named("glm5_run slot logits", g.vocab * 4);
            self.seqs.push(SeqSlot { kda, mla, logits });
        }
        self.bat = Some(BatchHead {
            normed: cuda::alloc_named("glm5_run batch normed", n * g.hidden * 4),
            logits: cuda::alloc_named("glm5_run batch logits", n * g.vocab * 4),
            ids: cuda::alloc_named("glm5_run batch greedy ids", n * 4),
        });
        Ok(())
    }

    /// the sequence slots (1 without `CROW_GLM_MAX_BATCH`)
    pub fn slots(&self) -> usize {
        self.seqs.len().max(1)
    }

    /// the slot every one-sequence path acts on
    pub fn slot(&self) -> usize {
        self.cur
    }

    /// Make slot `s` the one every one-sequence path acts on: the current slot's KDA states,
    /// MLA caches and logits row are parked, `s`'s come in (pointer swaps, no copy, no launch).
    pub fn use_slot(&mut self, s: usize) -> Result<(), String> {
        if s >= self.slots() {
            return Err(format!("glm5_run: sequence slot {s}, the run holds {} ({MAX_BATCH_ENV})", self.slots()));
        }
        if s == self.cur {
            return Ok(());
        }
        if self.ahead.is_some() {
            return Err(format!("glm5_run: sequence slot {s} while a row of slot {} is ahead ({}); settle_ahead first", self.cur, glm5_flags::ENV_LA));
        }
        let c = self.cur;
        for i in [c, s] {
            let p = &mut self.seqs[i];
            std::mem::swap(&mut self.kda, &mut p.kda);
            std::mem::swap(&mut self.mla, &mut p.mla);
            std::mem::swap(&mut self.logits, &mut p.logits);
        }
        self.cur = s;
        Ok(())
    }

    /// the refusals of a batched step of `b` rows, by name: too few staging slots. The CPU lane
    /// runs every row's CPU combos in one pool run per row (`GpuMoePlan::experts_lane`), so a
    /// row's bits are its solo row's when the same experts go to the CPU.
    pub fn batch_check(b: usize, tiers: &ExpertTiers, topk: usize) -> Result<(), String> {
        if b <= 1 {
            return Ok(());
        }
        if tiers.stage_cap < b * topk {
            return Err(format!("{MAX_BATCH_ENV}={b}: the tiers hold {} staging slots, a step of {b} rows may stage {}; build ExpertTiers with stage_cap {b} x top-k", tiers.stage_cap, b * topk));
        }
        Ok(())
    }

    /// One decode row per entry `(slot, tok, pos)`, every slot at most once: `tok` at `pos` of
    /// that slot's sequence through every layer with the experts from `tiers` (ONE trunk pass,
    /// `Glm5Pass::call_slots_with_experts`), the head over the rows; the greedy id of each row,
    /// and each slot's logits row (read it after [`Glm5Run::use_slot`] through
    /// [`Glm5Run::logits_dev`]). Every row is the bits of that slot's one-row decode
    /// ([`Glm5Run::row`]); one row IS that call. The current slot does not change for more rows.
    ///
    /// # Safety
    /// A CUDA context is current; `tiers` belongs to this model.
    pub unsafe fn decode_batch(&mut self, cnq: &mut Cnq, tiers: &mut ExpertTiers, rows: &[(usize, i64, usize)]) -> Result<Vec<i64>, String> {
        let (g, h, v) = (self.g, self.g.hidden, self.g.vocab);
        let b = rows.len();
        if b == 0 {
            return Err("glm5_run: a batched decode step of no rows".into());
        }
        for (i, &(s, tok, pos)) in rows.iter().enumerate() {
            if s >= self.slots() {
                return Err(format!("glm5_run: sequence slot {s}, the run holds {} ({MAX_BATCH_ENV})", self.slots()));
            }
            if rows[..i].iter().any(|r| r.0 == s) {
                return Err(format!("glm5_run: sequence slot {s} twice in one decode step"));
            }
            if pos >= self.cap {
                return Err(format!("glm5_run: row {pos} of slot {s} is outside the caches of {} rows", self.cap));
            }
            if !(0..v as i64).contains(&tok) {
                return Err(format!("glm5_run: token id {tok} outside the vocab of {v}"));
            }
        }
        if b == 1 {
            let (s, tok, pos) = rows[0];
            self.use_slot(s)?;
            let id = self.row(cnq, tiers, tok, pos, true)?.ok_or_else(|| format!("glm5_run: the head row at {pos} gave no id"))?;
            return Ok(vec![id]);
        }
        if b > self.pass.max_t {
            return Err(format!("glm5_run: a decode step of {b} rows, the pass holds calls of {} ({MAX_BATCH_ENV} at load)", self.pass.max_t));
        }
        if self.spec.is_some() || self.graph.is_some() {
            return Err(format!("glm5_run: a batched decode step with {} or {} on", crate::glm5_mtp::MTP_ENV, glm5_graph::ENV));
        }
        Self::batch_check(b, tiers, g.topk)?;
        let (bn, bl, bi) = match self.bat.as_ref() {
            Some(bh) => (bh.normed, bh.logits, bh.ids),
            None => return Err(format!("glm5_run: a batched decode step without sequence slots ({MAX_BATCH_ENV})")),
        };
        let toks: Vec<i64> = rows.iter().map(|r| r.1).collect();
        let pos: Vec<usize> = rows.iter().map(|r| r.2).collect();
        cuda::to_f32_into(self.x, &gm::trunk_input(&gm::embed_rows(cnq, &g, &toks), h, g.hc_streams));
        {
            let Glm5Run { pass, layers, kda, mla, seqs, cur, x, .. } = self;
            let (cur, x) = (*cur, *x);
            for l in 0..g.layers {
                let ks: Vec<&KdaState> = match kda[l] {
                    Some(_) => rows.iter().map(|r| if r.0 == cur { kda[l].as_ref() } else { seqs[r.0].kda[l].as_ref() }.expect("glm5_run: a slot without its KDA state")).collect(),
                    None => Vec::new(),
                };
                let ms: Vec<&MlaCache> = match mla[l] {
                    Some(_) => rows.iter().map(|r| if r.0 == cur { mla[l].as_ref() } else { seqs[r.0].mla[l].as_ref() }.expect("glm5_run: a slot without its MLA cache")).collect(),
                    None => Vec::new(),
                };
                let mut hook = |layer: usize, sel: &[i32]| -> Result<Dev, String> { tiers.table_for(layer, sel).map(|(tb, _)| tb) };
                pass.call_slots_with_experts(&layers[l], x, &pos, &ks, &ms, &mut hook).map_err(|e| format!("glm5_run: batch rows {pos:?} layer {l}: {e}"))?;
            }
        }
        gm::run_head(&self.pass.kn, &self.head, &self.hw, self.x, bn, bl, bi, b);
        cuda::sync();
        let ids: Vec<i64> = cuda::dtoh_i32(bi, b).into_iter().map(i64::from).collect();
        if let Some(&bad) = ids.iter().find(|&&id| !(0..v as i64).contains(&id)) {
            return Err(format!("glm5_run: batch rows {pos:?}: the greedy id {bad} is outside the vocab of {v}"));
        }
        // every slot's logits row of its last head
        for (i, r) in rows.iter().enumerate() {
            let dst = if r.0 == self.cur { self.logits } else { self.seqs[r.0].logits };
            cuda::d2d_async(dst, bl + (i * v * 4) as u64, v * 4);
        }
        cuda::sync();
        tiers.settle()?;
        Ok(ids)
    }

    /// free every parked slot and the batch head (slot `cur`'s state is in the run's own fields
    /// and stays, as slot 0)
    ///
    /// # Safety
    /// No launch of this run is pending.
    unsafe fn free_slots(&mut self) {
        let cur = self.cur;
        for (i, sl) in self.seqs.iter_mut().enumerate() {
            if i == cur {
                continue;
            }
            for s in sl.kda.iter_mut().flatten() {
                s.free();
            }
            for c in sl.mla.iter_mut().flatten() {
                c.free();
            }
            cuda::free_dev(&mut sl.logits);
        }
        self.seqs.clear();
        self.cur = 0;
        if let Some(mut bh) = self.bat.take() {
            for d in [&mut bh.normed, &mut bh.logits, &mut bh.ids] {
                cuda::free_dev(d);
            }
        }
    }
}

#[cfg(test)]
mod batch_env_tests {
    use super::*;

    #[test]
    fn max_batch_is_one_unset_and_refused_out_of_range_by_name() {
        assert_eq!(max_batch_of(None), Ok(1));
        assert_eq!(max_batch_of(Some(" ")), Ok(1));
        assert_eq!(max_batch_of(Some("1")), Ok(1));
        assert_eq!(max_batch_of(Some("8")), Ok(8));
        for bad in ["0", "9", "-1", "four"] {
            let e = max_batch_of(Some(bad)).unwrap_err();
            assert!(e.contains(MAX_BATCH_ENV) && e.contains(bad), "{e}");
        }
    }

    #[test]
    fn batch_vram_is_the_further_slots_and_nothing_for_one() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        assert_eq!(batch_vram_bytes(&g, 4096, 1), 0);
        let (b2, b3) = (batch_vram_bytes(&g, 4096, 2), batch_vram_bytes(&g, 4096, 3));
        // one more slot: at least its caches of 4096 rows
        let md = MlaDims::of(&g);
        assert!(b3 - b2 > MlaCache::bytes(&md, 4096), "{b2} {b3}");
        // the caches scale with the rows
        assert!(batch_vram_bytes(&g, 8192, 2) > b2);
    }
}
