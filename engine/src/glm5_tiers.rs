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
//!   decode call's pinned ids on the CPU (`glm5_moe::lane`, posted by [`ExpertTiers::table_for`]).
//!   Both off by default: the exchange rule above, bit for bit.
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
/// `1` = the CPU lane; unset / `0` = off (default)
pub const CPU_LANE_ENV: &str = "CROW_GLM_CPU_LANE";

/// #188: what happens to a selected expert the cache holds in pinned. Default (both false): it
/// is promoted into VRAM (staged H2D, its VRAM victim back to pinned by D2H), today's rule.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct PinnedUse {
    /// `CROW_GLM_PINNED=zerocopy` (or implied by the CPU lane): a pinned hit stays in pinned
    /// ([`ExpertCache::set_pinned_stays`]); the GPU kernels read it zero-copy from its slot
    pub stay: bool,
    /// `CROW_GLM_CPU_LANE=1`: in a decode call the selected ids in pinned are computed by the
    /// CPU from their slot (`glm5_moe::lane`), the others on the GPU
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
        Some("1") => true,
        Some(v) => return Err(format!("{CPU_LANE_ENV}={v:?}: accepted 0 (default), 1")),
    };
    if cpu_lane && promote_explicit == Some(true) {
        return Err(format!("{CPU_LANE_ENV}=1 reads the selected pinned experts where they lie, {PINNED_ENV}=promote moves them to VRAM first: unset {PINNED_ENV} or set it to zerocopy"));
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
    use crate::glm5_moe::lane::Combo;
    let mut n = 0;
    let combos = sel
        .iter()
        .map(|&e| {
            let loc = locs.iter().find(|x| x.0 == e as u32).expect("every selected id has a location").1;
            match loc {
                Loc::Pinned(q) => {
                    n += 1;
                    Combo::Cpu(cpu_host(q))
                }
                _ => Combo::Gpu(gpu_base(e as u32, loc)),
            }
        })
        .collect();
    (combos, n)
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
    let t = sel.len() / k;
    if t == 0 || sel.len() != t * k {
        return Err(format!("expert tiers: layer {l}: a selection of {} ids is no [rows][{k}]", sel.len()));
    }
    let fits = |r: usize| -> Result<bool, String> { Ok(staged_count(cache, l, &distinct_ids(&sel[..r * k], cache.experts)?) <= cap) };
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
    let one = staged_count(cache, l, &distinct_ids(&sel[..k], cache.experts)?);
    Err(format!("expert tiers: layer {l}: one row stages {one} records, {cap} staging slots"))
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

/// #186: the calls of a prompt phase of `n` rows at up to `chunk` rows per call, `(first row,
/// rows)`: full chunks, then the remainder as one call (its FFN runs in the plan's sizes,
/// `Glm5Pass::call_with_expert_batches`). Chunk 1: one row each.
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

/// #186: the staging slots of a prompt call: the #176 prefill set the plan books
/// (`Stability::stage_slots(..).prefill`: `PF_TG`, doubled with `CROW_PF_ASYNC`)
pub fn prefill_stage_slots(topk: usize) -> usize {
    crate::geo::Stability::of(crate::geo::Family::Glm5Next).stage_slots(topk, crate::gen::pf_tg(), crate::gen::pf_async_on()).prefill
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
    pf_landing: Landing,
    /// #188: `CROW_GLM_PINNED` / `CROW_GLM_CPU_LANE` as read at construction
    pub pinned_use: PinnedUse,
    /// the pinned arenas are write-combined (`CROW_PINNED_ALLOC`)
    pinned_wc: bool,
    topk: usize,
    lane_clock: std::sync::Arc<crate::glm5_moe::lane::Clock>,
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
        let cache = ExpertCache::new(policy, Scope::PerLayer, nl, g.experts, sizes.vram, sizes.pinned)?;
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
            pf_landing: Landing::new(0),
            pinned_use: PinnedUse::default(),
            pinned_wc,
            topk: g.topk,
            lane_clock: Default::default(),
        };
        t.set_pinned_use(pu)?;
        Ok(t)
    }

    /// #188: how selected pinned experts are read from now on ([`PinnedUse`]; `new` sets it
    /// from `CROW_GLM_PINNED` / `CROW_GLM_CPU_LANE`). The CPU lane is refused on a
    /// write-combined pinned arena. Changes no slot and no record.
    pub fn set_pinned_use(&mut self, u: PinnedUse) -> Result<(), String> {
        lane_on_wc(u, self.pinned_wc)?;
        self.cache.set_pinned_stays(u.stay);
        self.pinned_use = u;
        Ok(())
    }

    /// the pinned bytes this store holds
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned.iter().map(|p| p.bytes as u64).sum()
    }

    /// the VRAM bytes this store holds (arenas, staging, tables)
    pub fn vram_bytes(&self) -> u64 {
        let nl = self.slots.len() as u64;
        nl * (self.sizes.vram as u64 * self.rb + self.cache.experts as u64 * 8) + (self.stage_cap + self.pf_cap) as u64 * self.rb
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
        let mut served = serve(&mut self.cache, l, &mut self.slots[l], &ids, self.stage_cap, &mut m)?;
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
        let mut post = None;
        if self.pinned_use.cpu_lane && sel.len() == self.topk && served.moves.zero_copy > 0 {
            let host = self.pinned[l].host as *const u8;
            let (combos, n) = lane_combos(sel, &served.locs, |e, _| table[e as usize], |q| host.add(q as usize * rb as usize));
            served.moves.cpu_lane = n as u64;
            served.moves.zero_copy -= n as u64;
            post = Some(crate::glm5_moe::lane::Call { table: self.tables[l], combos, clock: self.lane_clock.clone() });
        }
        crate::glm5_moe::lane::post(post);
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
        self.pf_landing = Landing::new(slots * self.rb as usize);
        self.pf_cap = slots;
        Ok(())
    }

    /// #186: the slots of the prefill staging set (0 before the first prompt call)
    pub fn prefill_cap(&self) -> usize {
        self.pf_cap
    }

    /// #186: the experts of one prompt call of decoder layer `layer` (`sel` = `[t][topk]` i32):
    /// [`serve_chunk`] through the prefill staging set (allocated here on first use), the device
    /// table rewritten per row sub-batch; `run(row0, rows, table)` queues that sub-batch's
    /// experts. One routing sync for the call; a stream sync before every later sub-batch.
    ///
    /// # Safety
    /// A CUDA context is current; no launch reading this layer's slots, its table or the prefill
    /// staging set is pending; `run` queues on the current stream.
    pub unsafe fn tables_for_chunk(&mut self, layer: usize, sel: &[i32], run: &mut dyn FnMut(usize, usize, Dev) -> Result<(), String>) -> Result<(), String> {
        let l = layer.checked_sub(self.first_moe).filter(|&l| l < self.slots.len()).ok_or_else(|| format!("expert tiers: layer {layer} is no MoE layer"))?;
        if self.pf_cap == 0 {
            self.alloc_prefill_stage(prefill_stage_slots(self.topk))?;
        }
        // a CPU-lane post belongs to a decode call: none for this call's tables
        crate::glm5_moe::lane::post(None);
        let (rb, k, experts) = (self.rb, self.topk, self.cache.experts);
        let (vram, pin_dev, stage, table_dev) = (self.vram[l], self.pinned.get(l).map_or(0, |p| p.dev), self.pf_stage, self.tables[l]);
        let mut m = GpuMover { vram, pinned: self.pinned.get(l), stage, landing: self.pf_landing.p, rb, src: &self.src, recs: &self.records[l] };
        let (mut reads, mut bytes, mut moves) = (0u64, 0u64, Moves::default());
        let mut each = |r0: usize, rows: usize, served: &Served| -> Result<(), String> {
            let mut table = vec![0u64; experts];
            for &(e, loc) in &served.locs {
                table[e as usize] = match loc {
                    Loc::Vram(v) => vram + v as u64 * rb,
                    Loc::Pinned(q) => pin_dev + q as u64 * rb,
                    Loc::Stage(s) => stage + s as u64 * rb,
                };
            }
            cuda::to_u64_into(table_dev, &table);
            reads += served.nvme_reads as u64;
            bytes += served.nvme_bytes;
            moves.add(&served.moves);
            run(r0, rows, table_dev)
        };
        let r = serve_chunk(&mut self.cache, l, &mut self.slots[l], sel, k, self.pf_cap, &mut m, &mut each);
        self.nvme_reads += reads;
        self.nvme_bytes += bytes;
        self.moves[l].add(&moves);
        self.routing_syncs += 1;
        self.sub_batches += *r.as_ref().unwrap_or(&0) as u64;
        r.map(|_| ())
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
        reset_cache(&mut self.cache, &mut self.slots, self.sizes)
    }

    /// # Safety
    /// No launch reading the store is pending.
    pub unsafe fn free(&mut self) {
        cuda::sync();
        for d in self.vram.iter_mut().chain(self.tables.iter_mut()) {
            cuda::free_dev(d);
        }
        for p in self.pinned.iter_mut() {
            p.free();
        }
        cuda::free_dev(&mut self.stage);
        cuda::free_dev(&mut self.pf_stage);
        self.pf_cap = 0;
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
            counters: t.cache.counters().to_vec(),
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
            tiers: t.cache.counters().iter().zip(&self.counters).map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]]).collect(),
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
        let mut run = Glm5Run {
            g: *g,
            moe: *moe,
            cap,
            prompt_chunk: chunk,
            load,
            pass: Glm5Pass::new(g, *moe, chunk, cap),
            layers,
            kda,
            mla,
            head: Head::new(gm::head_geo(g)),
            hw,
            x: cuda::alloc_named("glm5_run residual", chunk * row * 4),
            normed: cuda::alloc_named("glm5_run normed", g.hidden * 4),
            logits: cuda::alloc_named("glm5_run logits", g.vocab * 4),
            next: cuda::alloc_named("glm5_run greedy id", 4),
            sw: Switches::default(),
            sw_kernels: None,
            feed: None,
            readback: None,
            graph: None,
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
        if sw == Switches::default() {
            return;
        }
        let k = glm5_flags::Kernels::new(&self.g);
        if sw.flags {
            // the decode calls' ids (one row); #186 prompt calls read theirs after a stream sync
            self.pass.routed = Some(Routed::new(&k, self.moe.topk));
        }
        if sw.lookahead {
            self.feed = Some(Feed::load(&k, cnq, &self.g));
            self.readback = Some(Readback::new(self.g.vocab));
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
    fn trace(tokens: usize, layers: usize, experts: u64, k: usize, seed: u64) -> Vec<Vec<Vec<u32>>> {
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
    struct SynthGlm {
        dir: std::path::PathBuf,
        path: String,
    }

    impl Drop for SynthGlm {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.dir);
        }
    }

    fn synth_glm(experts: u32) -> SynthGlm {
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
        self.embed(cnq, tok);
        self.layers(tiers, pos, &mut |_| Ok(()))?;
        if !head {
            cuda::sync();
            return Ok(None);
        }
        self.head_row()?;
        cuda::sync();
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

// ---------------------------------------------------------------- #186: the prompt phase in calls

/// #186: the prompt phase of the glm5_next path in prompt calls of up to `prompt_chunk` rows.
impl Glm5Run {
    /// prompt rows per prompt call (1 = every prompt row one decode call)
    pub fn prompt_chunk(&self) -> usize {
        self.prompt_chunk
    }

    /// Prompt rows per prompt call from now on, 1 ..= the `max_t` the pass was built with at
    /// `load` (`CROW_CHUNK`); a larger ask is refused by name.
    pub fn set_prompt_chunk(&mut self, chunk: usize) -> Result<(), String> {
        if chunk == 0 || chunk > self.pass.max_t {
            return Err(format!("glm5_run: a prompt chunk of {chunk} rows, the pass holds calls of 1 ..= {} rows (CROW_CHUNK at load)", self.pass.max_t));
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
                report(&base.report(tiers, pos0 + i, true, last, t0));
            }
            return last.ok_or_else(|| "glm5_run: the last prompt row gave no id".to_string());
        }
        let (g, h) = (self.g, self.g.hidden);
        let row = g.hc_streams * h;
        let mut next = None;
        for (r0, t) in prompt_calls(n, self.prompt_chunk) {
            let t0 = std::time::Instant::now();
            let base = RowBase::of(tiers);
            let e = gm::embed_rows(cnq, &g, &ids[r0..r0 + t]);
            cuda::to_f32_into(self.x, &gm::trunk_input(&e, h, g.hc_streams));
            self.layers_chunk(tiers, pos0 + r0, t)?;
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
    use crate::glm5_moe::{GpuFfnPlan, GpuMoePlan};

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
            for cap in [4096usize, 200_000] {
                let held = |chunk: usize| -> u64 {
                    let before = cuda::live_dev().1;
                    let mut pass = Glm5Pass::new(&g, moe, chunk, cap);
                    let mut x = cuda::alloc_named("test residual", chunk * row * 4);
                    let mut plans: Vec<GpuMoePlan> = Vec::new();
                    let mut dense: Vec<GpuFfnPlan> = Vec::new();
                    for t in crate::manager::glm5_prompt_call_sizes(chunk) {
                        plans.push(GpuMoePlan::new(&moe, t));
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
            run.set_switches(&mut cnq, Switches { flags: true, lookahead: true });
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
