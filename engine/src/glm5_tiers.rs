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
//! - **NVMe** (#149): [`NvmeSource`], one handle per reader, `FILE_FLAG_NO_BUFFERING`, 1 reader by
//!   default (B = 6.994 GB/s, PREREG amendment 5); every record (9,474,048 B at 3 bit) is located
//!   once at setup ([`ExpertRecord::glm5_table`]).
//! - **Driver** ([`Glm5Run`]): the dense part of every layer stays in VRAM
//!   (`glm5_model::load_layer_without_experts`), one KDA state per KDA layer and one MLA cache
//!   per DSA layer are swapped into the `Glm5Pass` around the layer's call, every row (prompt and
//!   generated) runs as one decode call (no chunked prefill in this path), greedy head.
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
    // staged: entering VRAM (any id), then the selected ids the policy left on NVMe
    let mut staged: Vec<u32> = (0..n as u32).filter(|&e| after[e as usize] == Tier::Vram && before[e as usize] != Tier::Vram).collect();
    staged.extend(ids.iter().copied().filter(|&e| after[e as usize] == Tier::Nvme));
    if staged.len() > stage_cap {
        return Err(format!("expert tiers: layer {l} stages {} records in one call, {stage_cap} staging slots", staged.len()));
    }
    let mut out = Served::default();
    // phase A: into staging, from pinned / VRAM (queued) or the container (via the landing buffer)
    let mut from_nvme = Vec::new();
    for (s, &e) in staged.iter().enumerate() {
        match before[e as usize] {
            Tier::Vram => m.vram_to_stage(slots.vram_of[e as usize], s as u32),
            Tier::Pinned => m.pinned_to_stage(slots.pin_of[e as usize], s as u32),
            Tier::Nvme => from_nvme.push((e, Dst::Landing(s as u32))),
        }
    }
    for c in from_nvme.chunks(MAX_IN_FLIGHT) {
        out.nvme_bytes += m.nvme(c)?;
    }
    for &(_, d) in &from_nvme {
        if let Dst::Landing(i) = d {
            m.landing_to_stage(i);
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
                Tier::Vram => m.vram_to_pinned(slots.vram_of[e], q),
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
        }
    }
    out.nvme_reads = from_nvme.len() + to_pinned.len();
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
        Ok(ExpertTiers {
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
        })
    }

    /// the pinned bytes this store holds
    pub fn pinned_bytes(&self) -> u64 {
        self.pinned.iter().map(|p| p.bytes as u64).sum()
    }

    /// the VRAM bytes this store holds (arenas, staging, tables)
    pub fn vram_bytes(&self) -> u64 {
        let nl = self.slots.len() as u64;
        nl * (self.sizes.vram as u64 * self.rb + self.cache.experts as u64 * 8) + self.stage_cap as u64 * self.rb
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
        let served = serve(&mut self.cache, l, &mut self.slots[l], &ids, self.stage_cap, &mut m)?;
        let mut table = vec![0u64; self.cache.experts];
        for &(e, loc) in &served.locs {
            table[e as usize] = match loc {
                Loc::Vram(v) => self.vram[l] + v as u64 * rb,
                Loc::Pinned(q) => self.pinned[l].dev + q as u64 * rb,
                Loc::Stage(s) => self.stage + s as u64 * rb,
            };
        }
        cuda::to_u64_into(self.tables[l], &table);
        self.nvme_reads += served.nvme_reads as u64;
        self.nvme_bytes += served.nvme_bytes;
        Ok((self.tables[l], served))
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
        Glm5Run {
            g: *g,
            moe: *moe,
            cap,
            load,
            pass: Glm5Pass::new(g, *moe, 1, cap),
            layers,
            kda,
            mla,
            head: Head::new(gm::head_geo(g)),
            hw,
            x: cuda::alloc_named("glm5_run residual", row * 4),
            normed: cuda::alloc_named("glm5_run normed", g.hidden * 4),
            logits: cuda::alloc_named("glm5_run logits", g.vocab * 4),
            next: cuda::alloc_named("glm5_run greedy id", 4),
        }
    }

    /// Greedy: feed `prompt`, then generate `n` ids, every row one decode call through all
    /// layers with the experts from `tiers`. `report` sees every row. Starts a new sequence
    /// (every KDA state zeroed).
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
        let (g, h) = (self.g, self.g.hidden);
        let mut out = Generated::default();
        for pos in 0..rows {
            let t0 = std::time::Instant::now();
            let tok = if pos < prompt.len() { prompt[pos] } else { out.ids[pos - prompt.len()] };
            let e = gm::embed_rows(cnq, &g, &[tok]);
            cuda::to_f32_into(self.x, &gm::trunk_input(&e, h, g.hc_streams));
            let c0: Vec<[u64; 3]> = tiers.cache.counters().to_vec();
            let mut tick = TierTick::default();
            for l in 0..g.layers {
                if let Some(s) = self.kda[l].as_mut() {
                    self.pass.swap_kda_state(s);
                }
                if let Some(c) = self.mla[l].as_mut() {
                    self.pass.swap_mla_cache(c);
                }
                let mut hook = |layer: usize, sel: &[i32]| -> Result<Dev, String> {
                    let (tb, s) = tiers.table_for(layer, sel)?;
                    tick.nvme_reads += s.nvme_reads as u64;
                    tick.nvme_bytes += s.nvme_bytes;
                    Ok(tb)
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
            let mut next = None;
            if pos + 1 >= prompt.len() {
                gm::run_head(&self.pass.kn, &self.head, &self.hw, self.x, self.normed, self.logits, self.next, 1);
                cuda::sync();
                if keep_logits {
                    out.logits.push(cuda::dtoh(self.logits, g.vocab));
                }
                let id = cuda::dtoh_i32(self.next, 1)[0] as i64;
                out.ids.push(id);
                next = Some(id);
            } else {
                cuda::sync();
            }
            let tiers_row: Vec<[u64; 3]> = tiers.cache.counters().iter().zip(&c0).map(|(a, b)| [a[0] - b[0], a[1] - b[1], a[2] - b[2]]).collect();
            report(&TokenReport { pos, prompt: pos < prompt.len(), next, secs: t0.elapsed().as_secs_f64(), nvme_reads: tick.nvme_reads, nvme_bytes: tick.nvme_bytes, tiers: tiers_row });
        }
        Ok(out)
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
    }
}

/// the number of MoE layers a run serves through the tiers (`layers - dense_prefix`)
pub fn moe_layers(g: &Glm5Geo) -> usize {
    (0..g.layers).filter(|&l| gm::ffn_kind(g, l) == FfnKind::Moe).count()
}

/// the fixed prompt of the cache-size test and the default smoke: the tokenizer golden
/// `sys_user_default` (system + user message, generation prompt; 38 ids, rendered by
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
    }

    const EMPTY: u32 = u32::MAX - 1;

    impl Sim {
        fn new(s: TierSizes, stage: usize) -> Sim {
            Sim { vram: vec![EMPTY; s.vram], pinned: vec![EMPTY; s.pinned], stage: vec![EMPTY; stage], landing: vec![EMPTY; stage], pending_reads_pinned: Vec::new(), reads: 0 }
        }
    }

    impl Mover for Sim {
        fn nvme(&mut self, jobs: &[(u32, Dst)]) -> Result<u64, String> {
            assert!(jobs.len() <= MAX_IN_FLIGHT);
            for &(e, d) in jobs {
                match d {
                    Dst::Landing(i) => self.landing[i as usize] = e,
                    Dst::Pinned(q) => {
                        assert!(!self.pending_reads_pinned.contains(&q), "NVMe wrote pinned slot {q} while a queued copy still reads it (no barrier)");
                        self.pinned[q as usize] = e;
                    }
                }
                self.reads += 1;
            }
            Ok(jobs.len() as u64 * 100)
        }
        fn landing_to_stage(&mut self, i: u32) {
            self.stage[i as usize] = self.landing[i as usize];
        }
        fn pinned_to_stage(&mut self, q: u32, s: u32) {
            self.pending_reads_pinned.push(q);
            self.stage[s as usize] = self.pinned[q as usize];
        }
        fn vram_to_stage(&mut self, v: u32, s: u32) {
            self.stage[s as usize] = self.vram[v as usize];
        }
        fn barrier(&mut self) {
            self.pending_reads_pinned.clear();
        }
        fn vram_to_pinned(&mut self, v: u32, q: u32) {
            self.pinned[q as usize] = self.vram[v as usize];
        }
        fn stage_to_vram(&mut self, s: u32, v: u32) {
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
                t.free();
            }
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
}
