//! #185 (GLM-5.3-Flash plan steps 20/22): the glm5_next engine behind `bin/serve`.
//!
//! `serve` dispatches by the container's family before it maps the container
//! (`bin/serve.rs`, `engine_kind`): Flash-Next and the 27B boot `gen::Engine` exactly as
//! before, and `glm5_next` boots this one through `boot::open_glm5`. Part 1 wired the request
//! layer (template variables, the 400s, the GLM tool markup, the reasoning filter, the stop
//! ids); part 2 (this file) is the generator under it.
//!
//! - [`Glm5Device`]: the model on the card. `glm5_tiers::Glm5Run` (every layer's dense part in
//!   VRAM, one KDA state per KDA layer, one MLA + DSA cache of `n_ctx` rows per DSA layer) and
//!   `glm5_tiers::ExpertTiers` (the routed experts in VRAM / pinned / NVMe, sized by the #159
//!   plan at this card's free VRAM and the derived pinned budget). Its boot prints the plan
//!   table as `[budget]` lines.
//! - [`Rows`]: what the engine needs of a model, one row at a time. [`Rows::prefill_chunk`]
//!   takes a chunk of ids and runs them row by row; #186 (chunked prefill inside `glm5_tiers`)
//!   overrides it for [`Glm5Device`] without changing `serve`.
//! - [`Glm5Engine`]: the sequence over a [`Rows`]: the held ids (`history`), prefill a chunk,
//!   decode one step, the logits row, reset, and the one-conversation prefix cache of `serve`
//!   (#31 A9, #36 M2b: one snapshot after each prompt, a warm request rolls back to it; #100:
//!   an identical re-request prefills nothing).
//!
//! The prefix snapshot of glm5_next (`docs/glm5-kda.md`, `docs/glm5-mla.md`): the KDA recurrent
//! state and its conv window of every KDA layer are COPIED (34 x (4 MiB + 288 KiB) f32 = 152,633,344 B
//! of pageable host RAM, allocated at the first snapshot, plus the 619,520 B logits row). The MLA
//! latent rows and the DSA indexer rows are position-indexed and every call reads rows
//! `0 .. pos0 + t` only (the pooled keys are recomputed from the rows, `glm5_mla` module doc), so a
//! prefix of `p` rows is a TRUNCATION: rows `>= p` left by the answer are overwritten before any
//! call reads them. A cold start writes rows from 0, so it drops the snapshot.
//!
//! #192 (`CROW_GLM_MTP=N`): [`Rows::decode`] of the device is the speculative decode
//! (`Glm5Run::spec_decode`): one verify step emits up to `1 + N` ids, handed out one per
//! [`Glm5Engine::decode_step`] while the fed id is the verified one (they stream at once); a fed id
//! that differs (a sampled draw, a forced id, a grammar redraw) rolls the KDA states back to that
//! row (the verify's snapshot slot). Every emitted id's logits row is the verify's row, the bits of
//! the one-row decode, so the host sampler draws as without MTP. The block's MLA cache truncates
//! like the trunk's; its one pending row (the last trunk row's head-norm row, whose next id is not
//! known yet) is part of the snapshot (`state_floats` + H).

use crate::cuda;
use crate::geo::{Family, Glm5Geo, HOST_PINNED_CAP};
use crate::glm5_kda::KdaDims;
use crate::glm5_template;
use crate::glm5_tiers::{self as gt, ExpertTiers, Glm5Run, Opened};
use crate::manager::{derive_host_pinned_budget, glm5_plan_table, Glm5States, TierInput, TierPlan};
use crate::meta::ModelMeta;
use crate::toolcall::Markup;

/// NVMe readers of serve's expert tier by default (PREREG amendment 5: one reader,
/// B = 6.994 GB/s); `CROW_GLM_NVME_READERS` asks for another count (#185, `boot::glm5_tier_ask`)
pub const READERS: usize = 1;

/// Is `meta` a GLM-5.3-Flash config? `serve`'s dispatch sends only `Glm5Next` here; anything
/// else is refused by name.
pub fn check_family(cnq_path: &str, meta: &ModelMeta) -> Result<(), String> {
    if meta.family != Family::Glm5Next {
        return Err(format!(
            "[glm5] {cnq_path}: family {:?} ({}) is not glm5_next - Glm5Engine boots GLM-5.3-Flash only",
            meta.family, meta.config_path
        ));
    }
    Ok(())
}

/// #186: serve's boot plan: `glm5_tiers::plan_for_rows` over the whole context, so the prompt
/// chunk `CROW_CHUNK` that `Glm5Run::load` builds for is booked as in `glm5_run`
pub fn serve_plan(g: &Glm5Geo, context: usize, vram_total: u64, pinned_budget: u64, record_bytes: u64) -> Result<(Glm5States, TierInput, TierPlan), String> {
    gt::plan_for_rows(g, context, context, vram_total, pinned_budget, record_bytes)
}

/// What [`Glm5Engine`] needs of a model, one row at a time.
pub trait Rows {
    /// rows the caches hold (the context the boot allocated)
    fn n_ctx(&self) -> usize;
    /// the head's rows
    fn vocab(&self) -> usize;
    /// decoder layers (`crow_layers` on the wire)
    fn layers(&self) -> usize;
    /// f32 values of the recurrent state a prefix snapshot copies
    fn state_floats(&self) -> usize;
    /// One row: `tok` at `pos`; with `head`, the greedy id of the head (its logits row stays
    /// readable through [`Rows::logits`]).
    ///
    /// # Safety
    /// The model's device state is this thread's (a CUDA context is current).
    unsafe fn row(&mut self, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String>;
    /// The rows `ids` at `pos0 ..`, the head on the last: its greedy id. #186 replaces this for
    /// the device with a chunked prefill; the default is one row at a time.
    ///
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn prefill_chunk(&mut self, ids: &[i64], pos0: usize) -> Result<i64, String> {
        let mut last = None;
        for (i, &tok) in ids.iter().enumerate() {
            last = self.row(tok, pos0 + i, i + 1 == ids.len())?;
        }
        last.ok_or_else(|| "glm5: an empty prefill chunk".to_string())
    }
    /// One decode step: `tok` at `pos`, the greedy id after it (its logits row through
    /// [`Rows::logits`]). The default is [`Rows::row`] with the head; #192: the device's
    /// speculative decode overrides it.
    ///
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn decode(&mut self, tok: i64, pos: usize) -> Result<i64, String> {
        self.row(tok, pos, true)?.ok_or_else(|| format!("glm5: the head row at {pos} gave no id"))
    }
    /// #192: draft tokens per verify step (`CROW_GLM_MTP`; 0 = off)
    fn mtp_drafts(&self) -> usize {
        0
    }
    /// #192: the MTP counters since the last call as one line, then zero (`None` when off)
    fn mtp_report(&mut self) -> Option<String> {
        None
    }
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn logits(&self) -> Vec<f32>;
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn set_logits(&mut self, row: &[f32]);
    /// copy the recurrent state into `into` (`state_floats` values)
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn save_state(&self, into: &mut [f32]);
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn load_state(&mut self, from: &[f32]);
    /// back to the start of a sequence
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn zero_state(&mut self);
    /// cumulative `[vram, pinned, nvme]` expert accesses over every MoE layer, and the NVMe
    /// bytes read, since boot (never reset)
    fn counters(&self) -> ([u64; 3], u64);

    /// `CROW_GLM_LA`: drop a row enqueued ahead (the state back to the last returned row);
    /// everything that reads or replaces the sequence's state calls this first
    ///
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn settle_ahead(&mut self) -> Result<(), String> {
        Ok(())
    }
    /// `CROW_GLM_MAX_BATCH`: the sequence slots the model holds (1 = the one sequence)
    fn slots(&self) -> usize {
        1
    }
    /// `CROW_GLM_MAX_BATCH`: make slot `s` the sequence every other call of this trait acts on
    /// (its recurrent state, its rows, its logits row)
    ///
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn use_slot(&mut self, s: usize) -> Result<(), String> {
        if s == 0 {
            Ok(())
        } else {
            Err(format!("glm5: sequence slot {s}, the model holds one sequence ({})", gt::MAX_BATCH_ENV))
        }
    }
    /// `CROW_GLM_MAX_BATCH`: one decode row per entry `(slot, tok, pos)`, every slot at most
    /// once: the greedy id after each, each slot's logits row readable after
    /// [`Rows::use_slot`]. Every row is the bits of that slot's [`Rows::decode`] without MTP.
    /// The default is that call, slot after slot; the device runs the rows in one trunk pass.
    /// The current slot afterwards is not specified: the caller selects the one it needs.
    ///
    /// # Safety
    /// As [`Rows::row`].
    unsafe fn decode_batch(&mut self, rows: &[(usize, i64, usize)]) -> Result<Vec<i64>, String> {
        let mut ids = Vec::with_capacity(rows.len());
        for &(s, tok, pos) in rows {
            self.use_slot(s)?;
            ids.push(self.decode(tok, pos)?);
        }
        Ok(ids)
    }
}

// ---------------------------------------------------------------- the device

/// The glm5_next model on the card: every layer's dense part, the head, the KDA states and MLA
/// caches of `n_ctx` rows (`Glm5Run`), the routed experts in three tiers (`ExpertTiers`).
pub struct Glm5Device {
    pub o: Opened,
    pub run: Glm5Run,
    pub tiers: ExpertTiers,
    pub plan: TierPlan,
    pub n_ctx: usize,
    kd: KdaDims,
}

impl Glm5Device {
    /// Plan and load: the #159 plan at this card's free VRAM, `context` rows and the derived
    /// pinned budget (its table goes to `log` as `[budget]` lines), every layer without its
    /// routed experts, the head, the tiers at the plan's sizes. `log` sees every boot line.
    ///
    /// # Safety
    /// A CUDA context is current (`boot::open_glm5`) and outlives the returned value.
    pub unsafe fn load(mut o: Opened, context: usize, log: &mut dyn FnMut(&str)) -> Result<Glm5Device, String> {
        let free = cuda::free_vram_bytes();
        let budget = derive_host_pinned_budget(HOST_PINNED_CAP, &mut |s| log(s));
        // #192: CROW_GLM_MTP=N; the block, its cache of `context` rows and the KDA snapshot slots
        // come off the plan's free VRAM, as in glm5_run
        let drafts = crate::glm5_mtp::draft_rows_from_env()?;
        let mtp_reserved = if drafts > 0 { crate::glm5_mtp::spec_vram_bytes(&o.g, &o.moe, context, drafts) } else { 0 };
        if drafts > 0 {
            log(&format!("[budget] {}={drafts}: the #159 plan runs at free VRAM minus {mtp_reserved} B (the MTP block, its cache, {drafts} KDA snapshot slots; derived)", crate::glm5_mtp::MTP_ENV));
        }
        // CROW_GLM_MAX_BATCH=N: the further sequence slots and the N-row step come off the plan's
        // free VRAM the same way
        let batch = gt::max_batch_from_env()?;
        let batch_reserved = gt::batch_vram_bytes(&o.g, context, batch);
        if batch > 1 {
            log(&format!("[budget] {}={batch}: the #159 plan runs at free VRAM minus {batch_reserved} B ({} further sequence slots of {context} rows, the {batch}-row step; derived)", gt::MAX_BATCH_ENV, batch - 1));
        }
        let (states, input, plan) = serve_plan(&o.g, context, free.saturating_sub(mtp_reserved + batch_reserved), budget, o.spec.bytes)?;
        let sources = [
            ("dense", "GLM5_NEXT_DENSE_BYTES".to_string()),
            ("expert", format!("{}: {} records, codec {}", o.path, o.records, o.spec.codec.dtype())),
            ("vram", "free VRAM at boot".to_string()),
            ("pinned", "derived budget: min(HOST_PINNED_CAP, free RAM - CROW_RAM_MARGIN_GB)".to_string()),
        ];
        for l in glm5_plan_table(&o.g, &states, &input, &plan, &sources).lines() {
            log(&format!("[budget] {l}"));
        }
        // #185: CROW_GLM_VRAM_SLOTS / CROW_GLM_PINNED_SLOTS / CROW_GLM_NVME_READERS (glm5_run's
        // --vram-slots / --pinned-slots / --readers); unset = the plan and READERS
        let ask = crate::boot::glm5_tier_ask_from_env()?;
        let (sizes, readers) = ask.resolve(&plan, READERS)?;
        let mut run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, context, log);
        let mtp_n = match run.mtp_from_env(&mut o.cnq, log) {
            Ok(n) => n,
            Err(e) => {
                run.free();
                return Err(e);
            }
        };
        // CROW_GLM_MAX_BATCH: the sequence slots (refused by name with MTP or the graphs)
        if let Err(e) = run.set_slots(batch) {
            run.free();
            return Err(e);
        }
        // #192: a verify call stages the experts of 1 + N rows at once; the CPU lane is refused
        // (CROW_GLM_MAX_BATCH: a batched step those of N rows)
        let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, readers, (1 + mtp_n).max(batch) * o.g.topk)?;
        if mtp_n > 0 {
            if let Err(e) = Glm5Run::spec_check(mtp_n, &tiers, o.g.topk) {
                tiers.free();
                run.free();
                return Err(e);
            }
        }
        if let Err(e) = Glm5Run::batch_check(batch, &tiers, o.g.topk) {
            tiers.free();
            run.free();
            return Err(e);
        }
        if batch > 1 {
            log(&format!("[glm5_run] {}={batch}: {batch} sequence slots, a decode step carries one row of each active sequence in one trunk pass", gt::MAX_BATCH_ENV));
        }
        let gib = |b: u64| b as f64 / (1u64 << 30) as f64;
        log(&format!(
            "[budget] glm5_next tiers per MoE layer: VRAM {} / pinned {} / NVMe {} x {} MoE layers (plan {} / {} / {}; {}); {:.2} GiB VRAM (slots, {} staging, tables), {:.2} GiB pinned; free VRAM now {:.2} GiB; policy {:?}, cache empty at start, NVMe readers {readers}",
            sizes.vram,
            sizes.pinned,
            o.g.experts - sizes.vram - sizes.pinned,
            gt::moe_layers(&o.g),
            plan.hot,
            plan.pinned,
            plan.nvme,
            ask.sources(),
            gib(tiers.vram_bytes()),
            tiers.stage_cap,
            gib(tiers.pinned_bytes()),
            gib(cuda::free_vram_bytes()),
            tiers.cache.policy
        ));
        let kd = KdaDims::of(&o.g);
        Ok(Glm5Device { o, run, tiers, plan, n_ctx: context, kd })
    }

    /// f32 values of one KDA layer's state and conv window
    fn one_state(&self) -> (usize, usize) {
        (self.kd.state_floats(), self.kd.conv_floats())
    }
}

impl Rows for Glm5Device {
    fn n_ctx(&self) -> usize {
        self.n_ctx
    }
    fn vocab(&self) -> usize {
        self.o.g.vocab
    }
    fn layers(&self) -> usize {
        self.o.g.layers
    }
    fn state_floats(&self) -> usize {
        let (s, c) = self.one_state();
        // #192: + the MTP block's pending head-norm row
        self.run.kda_states().count() * (s + c) + self.run.spec_state_floats()
    }
    unsafe fn row(&mut self, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String> {
        self.run.row(&mut self.o.cnq, &mut self.tiers, tok, pos, head)
    }
    unsafe fn prefill_chunk(&mut self, ids: &[i64], pos0: usize) -> Result<i64, String> {
        // #186: `CROW_CHUNK` > 1 runs the chunk as prompt calls; 1 (unset) is the row loop of
        // the trait's default, row for row. #192: with MTP the block follows the prompt rows
        if self.run.mtp_drafts() > 0 {
            return self.run.spec_prefill(&mut self.o.cnq, &mut self.tiers, ids, pos0);
        }
        self.run.prefill(&mut self.o.cnq, &mut self.tiers, ids, pos0, &mut |_| {})
    }
    unsafe fn decode(&mut self, tok: i64, pos: usize) -> Result<i64, String> {
        if self.run.mtp_drafts() > 0 {
            return self.run.spec_decode(&mut self.o.cnq, &mut self.tiers, tok, pos);
        }
        // CROW_GLM_LA: the next row goes ahead on the device's id before this one is read
        if self.run.switches().la {
            return self.run.decode_la(&mut self.o.cnq, &mut self.tiers, tok, pos);
        }
        self.row(tok, pos, true)?.ok_or_else(|| format!("glm5: the head row at {pos} gave no id"))
    }
    fn mtp_drafts(&self) -> usize {
        self.run.mtp_drafts()
    }
    fn mtp_report(&mut self) -> Option<String> {
        self.run.mtp_take_stats().map(|s| s.summary())
    }
    unsafe fn logits(&self) -> Vec<f32> {
        // CROW_GLM_LA: a row may be running ahead; the stream first (a blocking pageable copy
        // issued while a controlled row waits on the device faulted, WDDM, 2026-10-10)
        if self.run.has_ahead() {
            cuda::sync();
        }
        cuda::dtoh(self.run.logits_dev(), self.o.g.vocab)
    }
    unsafe fn set_logits(&mut self, row: &[f32]) {
        cuda::to_f32_into(self.run.logits_dev(), row);
        cuda::sync();
    }
    unsafe fn save_state(&self, into: &mut [f32]) {
        let (s, c) = self.one_state();
        cuda::sync();
        let mut at = 0;
        for k in self.run.kda_states() {
            for (src, n) in [(k.s, s), (k.conv, c)] {
                // blocking: `into` is complete when this returns
                cuda::ck(cudarc::driver::sys::cuMemcpyDtoH_v2(into[at..at + n].as_mut_ptr() as *mut std::ffi::c_void, src, n * 4));
                at += n;
            }
        }
        // #192: the MTP block's pending head-norm row
        if let Some(hp) = self.run.spec_pending_row() {
            let n = self.o.g.hidden;
            cuda::ck(cudarc::driver::sys::cuMemcpyDtoH_v2(into[at..at + n].as_mut_ptr() as *mut std::ffi::c_void, hp, n * 4));
        }
    }
    unsafe fn load_state(&mut self, from: &[f32]) {
        let (s, c) = self.one_state();
        cuda::sync();
        let mut at = 0;
        for k in self.run.kda_states() {
            for (dst, n) in [(k.s, s), (k.conv, c)] {
                cuda::to_f32_into(dst, &from[at..at + n]);
                at += n;
            }
        }
        // #192: nothing is ahead of a restored sequence; its pending head-norm row comes back
        self.run.spec_level();
        if let Some(hp) = self.run.spec_pending_row() {
            cuda::to_f32_into(hp, &from[at..at + self.o.g.hidden]);
        }
        cuda::sync();
    }
    unsafe fn zero_state(&mut self) {
        for k in self.run.kda_states() {
            k.reset();
        }
        self.run.spec_level();
    }
    unsafe fn settle_ahead(&mut self) -> Result<(), String> {
        self.run.settle_ahead(&mut self.tiers)
    }
    fn counters(&self) -> ([u64; 3], u64) {
        // the tier each access was served from on either path (the global arena keeps its own)
        let a = self.tiers.tier_counters().iter().fold([0u64; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
        (a, self.tiers.nvme_bytes)
    }
    fn slots(&self) -> usize {
        self.run.slots()
    }
    unsafe fn use_slot(&mut self, s: usize) -> Result<(), String> {
        // CROW_GLM_LA: a row ahead belongs to the current slot's KDA states (and their backup);
        // it is dropped before another slot's states come in
        if s != self.run.slot() {
            self.run.settle_ahead(&mut self.tiers)?;
        }
        self.run.use_slot(s)
    }
    unsafe fn decode_batch(&mut self, rows: &[(usize, i64, usize)]) -> Result<Vec<i64>, String> {
        self.run.settle_ahead(&mut self.tiers)?;
        self.run.decode_batch(&mut self.o.cnq, &mut self.tiers, rows)
    }
}

impl Drop for Glm5Device {
    fn drop(&mut self) {
        // SAFETY: the context `boot::open_glm5` made is still current (the caller drops the
        // engine before it), and no launch is pending between requests
        unsafe {
            let _ = self.run.settle_ahead(&mut self.tiers);
            self.tiers.free();
            self.run.free();
        }
    }
}

// ---------------------------------------------------------------- the sequence and its prefix cache

/// The one prompt snapshot of the held conversation (#36 M2b: one per request, after the
/// prompt; #100: with its logits row and greedy id, so an identical re-request prefills nothing).
#[derive(Default)]
struct Snapshot {
    /// the position it holds; `None` = empty
    pos: Option<usize>,
    greedy: i64,
    state: Vec<f32>,
    logits: Vec<f32>,
}

/// The glm5_next sequence `serve` drives: the held ids, prefill / decode / logits / reset, and
/// the prefix cache (one snapshot after each prompt).
pub struct Glm5Engine<R: Rows = Glm5Device> {
    rows: R,
    /// the ids whose rows the state holds (prompt and fed generated ids); `pos` = its length
    history: Vec<i64>,
    cache_on: bool,
    snap: Snapshot,
    /// `CROW_GLM_MAX_BATCH`: the current sequence slot (`history` and `snap` are its books) and
    /// the parked books of every slot (`parked[slot]` is empty while `slot` is current)
    slot: usize,
    parked: Vec<SlotBooks>,
}

/// `CROW_GLM_MAX_BATCH`: a parked slot's held ids and prompt snapshot
#[derive(Default)]
struct SlotBooks {
    history: Vec<i64>,
    snap: Snapshot,
}

impl<R: Rows> Glm5Engine<R> {
    /// `cache_on` is `CROW_PREFIX_CACHE != 0` in `serve`; off, every request is a cold start
    pub fn new(rows: R, cache_on: bool) -> Glm5Engine<R> {
        let parked = (0..rows.slots()).map(|_| SlotBooks::default()).collect();
        Glm5Engine { rows, history: Vec::new(), cache_on, snap: Snapshot::default(), slot: 0, parked }
    }

    pub fn rows(&self) -> &R {
        &self.rows
    }

    pub fn rows_mut(&mut self) -> &mut R {
        &mut self.rows
    }

    /// the context the boot allocated
    pub fn n_ctx(&self) -> usize {
        self.rows.n_ctx()
    }

    /// the vocabulary size of the loaded head (`Glm5Geo::vocab`)
    pub fn vocab(&self) -> usize {
        self.rows.vocab()
    }

    pub fn layers(&self) -> usize {
        self.rows.layers()
    }

    /// the three end-of-turn ids of GLM-5.3-Flash (`generation_config.json` `eos_token_id`)
    pub fn stop_ids(&self) -> &'static [u32] {
        &glm5_template::EOS_IDS
    }

    /// the tool-call markup this family's template writes
    pub fn markup(&self) -> Markup {
        Markup::Glm
    }

    /// the held conversation: every id whose row is in the state
    pub fn history(&self) -> &[i64] {
        &self.history
    }

    pub fn pos(&self) -> usize {
        self.history.len()
    }

    pub fn cache_enabled(&self) -> bool {
        self.cache_on
    }

    /// the position of the prompt snapshot, `None` while none is held
    pub fn snapshot_pos(&self) -> Option<usize> {
        self.snap.pos
    }

    /// host bytes one snapshot takes (state + logits row)
    pub fn snapshot_bytes(&self) -> usize {
        (self.rows.state_floats() + self.rows.vocab()) * 4
    }

    /// cumulative `[vram, pinned, nvme]` expert accesses and NVMe bytes since boot
    pub fn counters(&self) -> ([u64; 3], u64) {
        self.rows.counters()
    }

    /// The detection rule of `serve` (spec 7.4, #100) for the one snapshot: `L` is the common
    /// id prefix of the held conversation and `prompt`; the snapshot is reused when it sits at
    /// or below `L` and below the prompt's length, or AT it (its logits row stands in).
    pub fn decide(&self, prompt: &[i64]) -> crate::cache::Decision {
        if !self.cache_on {
            return crate::cache::Decision { l: 0, reuse: None, parked: false };
        }
        let l = crate::cache::common_prefix_len(&self.history, prompt);
        let reuse = crate::cache::reuse_slot_with_logits(&[self.snap.pos], &[self.snap.pos.is_some()], l, prompt.len());
        crate::cache::Decision { l, reuse, parked: false }
    }

    /// Back to the snapshot at `p`: the KDA states come back, the held ids are cut to `p`, the
    /// MLA / DSA rows `>= p` are left to be overwritten. Returns the wall in ms.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn rollback(&mut self, p: usize) -> Result<f64, String> {
        self.rows.settle_ahead()?;
        if self.snap.pos != Some(p) || p > self.history.len() {
            return Err(format!("glm5: no snapshot at {p} (held {:?}, history {})", self.snap.pos, self.history.len()));
        }
        let t = std::time::Instant::now();
        self.rows.load_state(&self.snap.state);
        self.history.truncate(p);
        Ok(t.elapsed().as_secs_f64() * 1e3)
    }

    /// #100: the snapshot's logits row back into the head's buffer; its greedy id
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn restore_logits(&mut self) -> i64 {
        self.rows.set_logits(&self.snap.logits);
        self.snap.greedy
    }

    /// A cold start: every KDA state to zero, nothing held. The snapshot goes too: the next
    /// prefill writes MLA rows from 0, below its position. Returns the wall in ms.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn reset(&mut self) -> f64 {
        let t = std::time::Instant::now();
        // a failed drop leaves nothing to keep: the state is zeroed next
        let _ = self.rows.settle_ahead();
        self.rows.zero_state();
        self.history.clear();
        self.snap.pos = None;
        t.elapsed().as_secs_f64() * 1e3
    }

    /// Prefill `chunk` at the held position; the greedy id of its last row. A failure resets
    /// the engine (the state is part-advanced), so the next request starts cold.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn prefill(&mut self, chunk: &[i64]) -> Result<i64, String> {
        let pos0 = self.history.len();
        if chunk.is_empty() || pos0 + chunk.len() > self.rows.n_ctx() {
            return Err(format!("glm5: prefill of {} ids at {pos0} with n_ctx {}", chunk.len(), self.rows.n_ctx()));
        }
        if let Err(e) = self.rows.settle_ahead() {
            self.reset();
            return Err(e);
        }
        match self.rows.prefill_chunk(chunk, pos0) {
            Ok(id) => {
                self.history.extend_from_slice(chunk);
                Ok(id)
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    /// Feed `id` (the id the last step chose) at the held position; the next greedy id. A
    /// failure resets the engine.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn decode_step(&mut self, id: i64) -> Result<i64, String> {
        let pos = self.history.len();
        if pos >= self.rows.n_ctx() {
            return Err(format!("glm5: decode at {pos}, n_ctx {}", self.rows.n_ctx()));
        }
        // #192: with MTP on the device, a verified id comes back without a launch
        match self.rows.decode(id, pos) {
            Ok(next) => {
                self.history.push(id);
                Ok(next)
            }
            Err(e) => {
                self.reset();
                Err(e)
            }
        }
    }

    /// the logits row of the last head (host copy)
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn logits(&self) -> Vec<f32> {
        self.rows.logits()
    }

    /// The prompt snapshot at the held position, with the head's logits row and `greedy`, the
    /// id the head chose there. Nothing with the cache off. Returns the wall in ms.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn snapshot(&mut self, greedy: i64) -> f64 {
        if !self.cache_on {
            return 0.0;
        }
        let t = std::time::Instant::now();
        if self.rows.settle_ahead().is_err() {
            return 0.0;
        }
        let n = self.rows.state_floats();
        if self.snap.state.len() != n {
            self.snap.state = vec![0.0; n];
        }
        self.rows.save_state(&mut self.snap.state);
        self.snap.logits = self.rows.logits();
        self.snap.greedy = greedy;
        self.snap.pos = Some(self.history.len());
        t.elapsed().as_secs_f64() * 1e3
    }
}

/// `CROW_GLM_MAX_BATCH`: several sequences, each with its own books (held ids, prompt
/// snapshot) and device state (the model's slot), and the batched decode step. Every call of
/// the one-sequence API above acts on the selected slot ([`Glm5Engine::select`]).
impl<R: Rows> Glm5Engine<R> {
    /// the sequence slots (1 without `CROW_GLM_MAX_BATCH`)
    pub fn slots(&self) -> usize {
        self.parked.len().max(1)
    }

    /// the selected slot
    pub fn slot(&self) -> usize {
        self.slot
    }

    /// Select slot `s`: its held ids, its snapshot and its device state are what every call of
    /// the one-sequence API acts on from now on (the model's slot follows).
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn select(&mut self, s: usize) -> Result<(), String> {
        if s >= self.slots() {
            return Err(format!("glm5: sequence slot {s}, the engine holds {} ({})", self.slots(), gt::MAX_BATCH_ENV));
        }
        self.rows.use_slot(s)?;
        if s == self.slot {
            return Ok(());
        }
        for i in [self.slot, s] {
            let p = &mut self.parked[i];
            std::mem::swap(&mut self.history, &mut p.history);
            std::mem::swap(&mut self.snap, &mut p.snap);
        }
        self.slot = s;
        Ok(())
    }

    fn books(&self, s: usize) -> (&[i64], &Snapshot) {
        if s == self.slot {
            (&self.history, &self.snap)
        } else {
            (&self.parked[s].history, &self.parked[s].snap)
        }
    }

    /// slot `s`'s held ids
    pub fn history_of(&self, s: usize) -> &[i64] {
        self.books(s).0
    }

    /// the position of slot `s`'s prompt snapshot, `None` while it holds none
    pub fn snapshot_pos_of(&self, s: usize) -> Option<usize> {
        self.books(s).1.pos
    }

    /// [`Glm5Engine::decide`] on slot `s`'s books (serve picks the slot whose snapshot serves
    /// the prompt best)
    pub fn decide_in(&self, s: usize, prompt: &[i64]) -> crate::cache::Decision {
        if !self.cache_on {
            return crate::cache::Decision { l: 0, reuse: None, parked: false };
        }
        let (history, snap) = self.books(s);
        let l = crate::cache::common_prefix_len(history, prompt);
        let reuse = crate::cache::reuse_slot_with_logits(&[snap.pos], &[snap.pos.is_some()], l, prompt.len());
        crate::cache::Decision { l, reuse, parked: false }
    }

    /// One decode step of several sequences: per `(slot, id)` (every slot at most once) `id`
    /// is fed at that slot's held position, all rows in one [`Rows::decode_batch`]; the next
    /// greedy id of each, each slot's logits row readable after [`Glm5Engine::select`]. Each
    /// slot's ids and logits are those of its own [`Glm5Engine::decode_step`]s. A failure resets
    /// every slot of the step (their states are part-advanced), so their next requests start
    /// cold. The selected slot stays selected on success.
    ///
    /// # Safety
    /// As [`Rows::row`].
    pub unsafe fn decode_batch(&mut self, steps: &[(usize, i64)]) -> Result<Vec<i64>, String> {
        let n_ctx = self.rows.n_ctx();
        let mut rows = Vec::with_capacity(steps.len());
        for (i, &(s, id)) in steps.iter().enumerate() {
            if s >= self.slots() {
                return Err(format!("glm5: sequence slot {s}, the engine holds {} ({})", self.slots(), gt::MAX_BATCH_ENV));
            }
            if steps[..i].iter().any(|x| x.0 == s) {
                return Err(format!("glm5: sequence slot {s} twice in one decode step"));
            }
            let pos = self.history_of(s).len();
            if pos >= n_ctx {
                return Err(format!("glm5: decode at {pos} of slot {s}, n_ctx {n_ctx}"));
            }
            rows.push((s, id, pos));
        }
        let r = self.rows.decode_batch(&rows);
        let back = self.rows.use_slot(self.slot);
        match r.and_then(|ids| back.map(|_| ids)) {
            Ok(next) => {
                for &(s, id, _) in &rows {
                    if s == self.slot {
                        self.history.push(id);
                    } else {
                        self.parked[s].history.push(id);
                    }
                }
                Ok(next)
            }
            Err(e) => {
                for &(s, _, _) in &rows {
                    if self.select(s).is_ok() {
                        self.reset();
                    }
                }
                Err(e)
            }
        }
    }
}

#[cfg(test)]
mod tests_186_plan {
    //! #186: serve's boot books the prompt chunk as `glm5_run` does
    use super::*;

    /// `CROW_CHUNK=32`, RTX 5090, 200,000 rows, the 3-bit record: serve's plan is `glm5_run`'s
    /// (`plan_for_rows`) and books chunk 32. No other lib test reads `CROW_CHUNK`.
    #[test]
    fn serve_books_the_prompt_chunk_as_glm5_run() {
        let g = Glm5Geo::GLM_5_3_FLASH;
        let (card, rec) = (32_607u64 << 20, 9_474_048u64);
        let old = std::env::var("CROW_CHUNK").ok();
        std::env::set_var("CROW_CHUNK", "32");
        let serve = serve_plan(&g, 200_000, card, HOST_PINNED_CAP, rec);
        let run = gt::plan_for_rows(&g, 200_000, 200_000, card, HOST_PINNED_CAP, rec);
        match old {
            Some(o) => std::env::set_var("CROW_CHUNK", o),
            None => std::env::remove_var("CROW_CHUNK"),
        }
        let ((ss, si, sp), (rs, ri, rp)) = (serve.unwrap(), run.unwrap());
        assert_eq!((si.chunk, si.chunk_scratch_bytes), (32, crate::manager::glm5_chunk_scratch_bytes(&g, 32, 200_000)), "serve books chunk 32");
        assert_eq!((ss.total(), si.chunk, si.chunk_scratch_bytes, si.host_pinned_budget, sp), (rs.total(), ri.chunk, ri.chunk_scratch_bytes, ri.host_pinned_budget, rp));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::glm5_tiers as gt;

    /// The host twin of [`Rows`] (`fake::FakeRows`), the model `serve`'s tests drive too.
    use super::fake::FakeRows;

    fn eng() -> Glm5Engine<FakeRows> {
        Glm5Engine::new(FakeRows::new(256, 64, (10..40).collect()), true)
    }

    /// greedy: prefill `prompt` (in the chunks given), then `n - 1` decode steps; the ids
    unsafe fn greedy(e: &mut Glm5Engine<FakeRows>, chunks: &[&[i64]], n: usize) -> Vec<i64> {
        let mut next = 0;
        for c in chunks {
            next = e.prefill(c).unwrap();
        }
        let mut out = vec![next];
        for _ in 1..n {
            next = e.decode_step(next).unwrap();
            out.push(next);
        }
        out
    }

    /// #186's contract: a prefill in chunks is the prefill of the whole (same ids, same logits,
    /// same held ids), and only the last row of each chunk runs the head
    #[test]
    fn a_prefill_in_chunks_is_the_prefill_of_the_whole() {
        let p: Vec<i64> = (0..23).map(|i| (i * 7 % 50) as i64).collect();
        unsafe {
            let mut a = eng();
            let ga = greedy(&mut a, &[&p], 12);
            let mut b = eng();
            let gb = greedy(&mut b, &[&p[..5], &p[5..6], &p[6..]], 12);
            assert_eq!(ga, gb);
            assert_eq!(a.logits(), b.logits());
            assert_eq!(a.history(), b.history());
            assert_eq!((a.rows().heads_run, b.rows().heads_run), (12, 14));
        }
    }

    /// #185 Expected result 3 on the host twin: a warm turn rolled back to the prompt snapshot
    /// produces the ids of a cold re-prefill of the whole conversation, with P > 0
    #[test]
    fn a_warm_turn_is_a_cold_reprefill() {
        let p1: Vec<i64> = (0..31).map(|i| (i * 13 % 60) as i64).collect();
        unsafe {
            let mut w = eng();
            assert_eq!(w.decide(&p1).reuse, None);
            w.reset();
            let first = w.prefill(&p1).unwrap();
            w.snapshot(first);
            let mut out = vec![first];
            for _ in 0..9 {
                out.push(w.decode_step(*out.last().unwrap()).unwrap());
            }
            // turn 2 extends turn 1 (its answer re-rendered differently: the rollback lands on
            // the prompt snapshot, the rule of #31 A9)
            let mut p2 = p1.clone();
            p2.extend([3, 5, 8, 13, 21]);
            let d = w.decide(&p2);
            assert_eq!((d.l, d.reuse), (31, Some((0, 31))));
            w.rollback(31).unwrap();
            let warm = greedy(&mut w, &[&p2[31..]], 16);
            let mut c = eng();
            let cold = greedy(&mut c, &[&p2], 16);
            assert_eq!(warm, cold, "warm ids are the cold re-prefill's");
            assert_eq!(w.history(), c.history());
            assert_eq!(w.logits(), c.logits());
        }
    }

    /// #100: an identical re-request prefills nothing: the snapshot's logits row and greedy id
    /// stand in, and the decode continues as after the prefill
    #[test]
    fn an_identical_request_prefills_nothing() {
        let p: Vec<i64> = (0..9).map(|i| i as i64 + 2).collect();
        unsafe {
            let mut e = eng();
            let first = e.prefill(&p).unwrap();
            let row = e.logits();
            e.snapshot(first);
            let mut a = vec![first];
            for _ in 0..5 {
                a.push(e.decode_step(*a.last().unwrap()).unwrap());
            }
            assert_eq!(e.decide(&p).reuse, Some((0, 9)));
            e.rollback(9).unwrap();
            let rows = e.rows().rows_run;
            assert_eq!(e.restore_logits(), first);
            assert_eq!(e.logits(), row);
            assert_eq!(e.rows().rows_run, rows, "nothing prefilled");
            let mut b = vec![first];
            for _ in 0..5 {
                b.push(e.decode_step(*b.last().unwrap()).unwrap());
            }
            assert_eq!(a, b);
        }
    }

    /// a cold start drops the snapshot (its MLA rows are overwritten from 0); a failed row
    /// resets the engine, so the next request is cold; the cache off never reuses
    #[test]
    fn a_cold_start_or_a_failed_row_drops_the_snapshot() {
        let p: Vec<i64> = (0..12).map(|i| i as i64 + 1).collect();
        unsafe {
            let mut e = eng();
            let f = e.prefill(&p).unwrap();
            e.snapshot(f);
            assert_eq!(e.decide(&p).reuse, Some((0, 12)));
            e.reset();
            assert_eq!((e.snapshot_pos(), e.decide(&p).reuse), (None, None));
            let f = e.prefill(&p).unwrap();
            e.snapshot(f);
            e.rows_mut().fail_at = Some(13);
            let next = e.decode_step(f).unwrap();
            let err = e.decode_step(next).unwrap_err();
            assert!(err.contains("row 13 failed"), "{err}");
            assert_eq!((e.history().len(), e.snapshot_pos()), (0, None));
            assert!(e.rollback(12).is_err());
            let mut off = Glm5Engine::new(FakeRows::new(64, 64, (10..40).collect()), false);
            let f = off.prefill(&p).unwrap();
            off.snapshot(f);
            assert_eq!((off.snapshot_pos(), off.decide(&p).reuse), (None, None));
        }
    }

    /// the prompt and the answer must fit the caches; an over-long prefill is refused by name
    #[test]
    fn a_prefill_past_n_ctx_is_refused() {
        unsafe {
            let mut e = Glm5Engine::new(FakeRows::new(8, 64, (10..40).collect()), true);
            let err = e.prefill(&[1; 9]).unwrap_err();
            assert!(err.contains("prefill of 9 ids at 0 with n_ctx 8"), "{err}");
        }
    }

    /// the checkpoint's own config beside the repository root (tests run from engine/);
    /// `None` when this machine has not downloaded it (`models/` is not in git)
    fn meta(dir: &str) -> Option<ModelMeta> {
        let c = format!("../models/{dir}/config.json");
        if !std::path::Path::new(&c).is_file() {
            eprintln!("no {c} on this machine - skipped");
            return None;
        }
        let g = format!("../models/{dir}/generation_config.json");
        let g = std::path::Path::new(&g).is_file().then_some(g);
        Some(ModelMeta::from_config_files(&c, g.as_deref()).expect("config parses"))
    }

    /// a Qwen config is never booted as glm5_next; GLM-5.3-Flash's is
    #[test]
    fn only_a_glm5_next_config_is_a_glm5_engine() {
        if let Some(m) = meta("Qwen3.8-27B") {
            let e = check_family("converter/Qwen3.8-27B-CNQ4.5.cnq", &m).unwrap_err();
            assert!(e.contains("is not glm5_next"), "{e}");
        }
        if let Some(m) = meta("GLM-5.3-Flash-original") {
            assert_eq!(check_family("converter/GLM-5.3-Flash-MUL1K3.cnq", &m), Ok(()));
        }
    }

    /// #185 Expected results 2 and 3 on the card: the real 3-bit container at the boot's
    /// context, the fixed prompt. (2) The serve rows (`Glm5Engine` over `Glm5Device`: prefill a
    /// chunk, decode steps) give `Glm5Run::generate`'s ids and byte-identical logits. (3) A warm
    /// turn rolled back to the prompt snapshot gives the ids of a cold re-prefill.
    #[test]
    #[ignore = "needs the GPU and the real 3-bit container (the #159 plan's tiers, up to 46 GiB pinned): cargo test --release --lib glm5_engine_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_engine_gpu_serve_rows_are_glm5_run_rows() {
        let path = std::env::var("CROW_CNQ").unwrap_or_else(|_| crate::geo::from_engine_dir(crate::glm5_model::GLM5_MUL1K3_CNQ));
        let prompt = gt::fixed_prompt();
        let n = 16;
        unsafe {
            let (o, _ctx, context) = crate::boot::open_glm5(&path).unwrap();
            let mut dev = Glm5Device::load(o, context, &mut |s| eprintln!("{s}")).unwrap();
            let want = dev.run.generate(&mut dev.o.cnq, &mut dev.tiers, &prompt, n, true, &mut |_| {}).unwrap();
            let mut e = Glm5Engine::new(dev, true);
            e.reset();
            let mut next = e.prefill(&prompt).unwrap();
            e.snapshot(next);
            let (mut ids, mut logits) = (vec![next], vec![e.logits()]);
            for _ in 1..n {
                next = e.decode_step(next).unwrap();
                ids.push(next);
                logits.push(e.logits());
            }
            assert_eq!(ids, want.ids, "serve rows vs glm5_run");
            for (i, (a, b)) in logits.iter().zip(&want.logits).enumerate() {
                let diff = a.iter().zip(b).filter(|(x, y)| x.to_bits() != y.to_bits()).count();
                assert_eq!(diff, 0, "generated position {i}: {diff} logits differ in bits");
            }
            let mut p2 = prompt.clone();
            p2.extend_from_slice(&ids[..8]);
            p2.extend_from_slice(&prompt[prompt.len() - 4..]);
            assert_eq!(e.decide(&p2).reuse, Some((0, prompt.len())));
            e.rollback(prompt.len()).unwrap();
            let mut w = vec![e.prefill(&p2[prompt.len()..]).unwrap()];
            for _ in 1..8 {
                w.push(e.decode_step(*w.last().unwrap()).unwrap());
            }
            e.reset();
            let mut c = vec![e.prefill(&p2).unwrap()];
            for _ in 1..8 {
                c.push(e.decode_step(*c.last().unwrap()).unwrap());
            }
            eprintln!("glm5_engine gpu: ids {ids:?}; warm {w:?}; cold {c:?}");
            assert_eq!(w, c, "warm turn vs cold re-prefill");
            drop(e);
        }
    }
}

#[cfg(test)]
mod tests_192_serve {
    //! #192 part 2 on the GPU: `Glm5Engine` over a `Glm5Device` of the synthetic 8-layer glm5_next
    //! model of the `glm5_flags` / `glm5_graph` tests (layers 0-2 KDA + dense, 3-7 MoE with 16
    //! MUL1 experts, DSA at 3 and 7, vocab 2048), V 3 + P 4 tiers, and a synthetic MTP block.
    //! `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_mtp_serve_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::cnq::Cnq;
    use crate::glm5_flags::tests::synth_model;
    use crate::glm5_moe::MoeGeo;
    use crate::glm5_mtp as mtp;
    use crate::glm5_tiers::TierSizes;
    use std::collections::HashMap;
    use std::sync::{Arc, Mutex};

    const REC: u64 = 9_474_048;

    fn geo() -> Glm5Geo {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (8, 3, 16, 8, 2048);
        g
    }

    /// a device of the synthetic model at `cap` rows: `nd` drafts per step (0 = off) with the
    /// synthetic block of `seed`
    unsafe fn device(path: &str, g: &Glm5Geo, nd: usize, cap: usize, seed: u64) -> Glm5Device {
        let (spec, _) = crate::nvme_source::glm5_record_of_container(path).unwrap();
        let moe = MoeGeo::new(g, spec).unwrap();
        let mut cnq = Cnq::open_checked(path).unwrap();
        let old = std::env::var(mtp::MTP_ENV).ok();
        std::env::set_var(mtp::MTP_ENV, nd.to_string());
        let mut run = Glm5Run::load(&mut cnq, g, &moe, cap, &mut |s| eprintln!("{s}"));
        match old {
            Some(o) => std::env::set_var(mtp::MTP_ENV, o),
            None => std::env::remove_var(mtp::MTP_ENV),
        }
        if nd > 0 {
            run.set_mtp(nd, Some(mtp::synthetic_block(g, &moe, seed))).unwrap();
        }
        let tiers = ExpertTiers::new(&cnq, path, g, &moe, TierSizes { vram: 3, pinned: 4 }, 1, (1 + nd) * g.topk).unwrap();
        let o = Opened { cnq, path: path.to_string(), g: *g, moe, spec, records: 0, constants: 0 };
        let plan = TierPlan { vram_ceiling: 0, fixed_bytes: 0, unit_bytes: 0, hot: 3, pinned: 4, nvme: g.experts - 7 };
        Glm5Device { o, run, tiers, plan, n_ctx: cap, kd: KdaDims::of(g) }
    }

    /// how the caller picks the id it feeds back from the logits row (as `glm_generate` does)
    #[derive(Clone, Copy, Debug, PartialEq)]
    enum Policy {
        /// the head's argmax
        Greedy,
        /// a draw at temperature 0.7: Gumbel-max with noise seeded by (position, id)
        Sampled,
        /// the argmax, but every fifth id forced to another one (an injected / redrawn id)
        Forced,
    }

    fn mix(a: u64) -> u64 {
        let mut x = a.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 31;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^ (x >> 29)
    }

    fn pick(p: Policy, i: usize, pos: usize, row: &[f32], next: i64) -> i64 {
        match p {
            Policy::Greedy => next,
            Policy::Forced if i % 5 == 4 => (next * 7 + 3) % row.len() as i64,
            Policy::Forced => next,
            Policy::Sampled => {
                let mut best = (f64::NEG_INFINITY, 0usize);
                for (k, &l) in row.iter().enumerate() {
                    let u = ((mix(((pos as u64) << 20) ^ k as u64) >> 11) as f64 + 0.5) / (1u64 << 53) as f64;
                    let v = l as f64 / 0.7 - (-u.ln()).ln();
                    if v > best.0 {
                        best = (v, k);
                    }
                }
                best.1 as i64
            }
        }
    }

    /// One request as `glm_generate` drives the engine: the decision, rollback or reset, the
    /// prefill (or the snapshot's row when the whole prompt is held), the snapshot, then `n` ids
    /// picked from the logits row and fed back. (fed ids, their logits rows, cached rows)
    unsafe fn turn<R: Rows>(e: &mut Glm5Engine<R>, prompt: &[i64], n: usize, p: Policy) -> (Vec<i64>, Vec<Vec<f32>>, usize) {
        let d = e.decide(prompt);
        let cached = match d.reuse {
            Some((_, q)) => {
                e.rollback(q).unwrap();
                q
            }
            None => {
                e.reset();
                0
            }
        };
        let mut next = if cached == prompt.len() { e.restore_logits() } else { e.prefill(&prompt[cached..]).unwrap() };
        e.snapshot(next);
        let (mut ids, mut rows) = (Vec::new(), Vec::new());
        for i in 0..n {
            let row = e.logits();
            let x = pick(p, i, e.pos(), &row, next);
            ids.push(x);
            rows.push(row);
            if i + 1 == n {
                break;
            }
            next = e.decode_step(x).unwrap();
        }
        (ids, rows, cached)
    }

    fn bits(a: &[Vec<f32>], b: &[Vec<f32>]) -> Vec<usize> {
        a.iter().zip(b).map(|(x, y)| x.iter().zip(y).filter(|(p, q)| p.to_bits() != q.to_bits()).count()).collect()
    }

    /// position -> the id fed there, of a turn
    fn transcript(prompt: &[i64], fed: &[i64]) -> HashMap<usize, i64> {
        prompt.iter().chain(fed).copied().enumerate().collect()
    }

    /// Serve with MTP gives the ids and logits rows of serve without it, for greedy, sampled
    /// (T 0.7) and forced ids, over four requests: T1 cold, T2 warm (rolled back to T1's prompt
    /// snapshot, a 11-id suffix prefilled), T3 the same prompt again (nothing prefilled, the
    /// snapshot's row), T4 T2's prompt cold. Engines: MTP off; N = 3 with the drafts taken from
    /// MTP off's transcript (the id fed at that position), wrong when the position % 4 == 1;
    /// N = 2 with the block's own drafts. A warm turn equals the cold re-prefill (ids and bits).
    #[test]
    #[ignore = "needs the GPU (about 6 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_mtp_serve_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_serve_gpu_turns_are_the_ids_without_mtp() {
        let g = geo();
        let s = synth_model(&g, REC);
        let p1: Vec<i64> = (0..21).map(|i| (i * 97 + 11) % 2048).collect();
        let n = 20usize;
        let cap = 21 + 11 + n + 4;
        let vocab = g.vocab as i64;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let gold: Arc<Mutex<HashMap<usize, i64>>> = Arc::default();
            let mut e0 = Glm5Engine::new(device(&s.path, &g, 0, cap, 0), true);
            let mut e1 = Glm5Engine::new(device(&s.path, &g, 3, cap, 0x0192_2001), true);
            let mut e2 = Glm5Engine::new(device(&s.path, &g, 2, cap, 0x0192_2002), true);
            assert_eq!((e0.rows().mtp_drafts(), e1.rows().mtp_drafts(), e2.rows().mtp_drafts()), (0, 3, 2));
            assert_eq!(e1.snapshot_bytes(), e0.snapshot_bytes() + g.hidden * 4, "the snapshot holds the block's pending head-norm row");
            let gh = gold.clone();
            e1.rows_mut().run.set_mtp_hook(Some(Box::new(move |pos, d| match gh.lock().unwrap().get(&pos) {
                Some(&t) if pos % 4 != 1 => t,
                Some(&t) => (t + 1) % vocab,
                None => d,
            })));
            for p in [Policy::Greedy, Policy::Sampled, Policy::Forced] {
                // MTP off: the four requests
                let t1 = turn(&mut e0, &p1, n, p);
                let mut p2 = p1.clone();
                p2.extend_from_slice(&t1.0[..6]);
                p2.extend([5, 77, 1024, 3, 2000]);
                let t2 = turn(&mut e0, &p2, n, p);
                let t3 = turn(&mut e0, &p2, n, p);
                e0.reset();
                let t4 = turn(&mut e0, &p2, n, p);
                assert_eq!((t1.2, t2.2, t3.2, t4.2), (0, 21, p2.len(), 0), "{p:?}: cached rows per request");
                // the sampled and forced ids leave the argmax (else the arm would be greedy)
                let off: usize = [&t1, &t2, &t3, &t4].iter().map(|t| t.0.iter().zip(&t.1).filter(|(&x, r)| x as usize != r.iter().enumerate().fold((0, f32::NEG_INFINITY), |b, (k, &v)| if v > b.1 { (k, v) } else { b }).0).count()).sum();
                eprintln!("glm5_mtp serve {p:?}: {off} of {} fed ids are not the argmax of their row", 4 * n);
                assert_eq!(off == 0, p == Policy::Greedy, "{p:?}: {off} ids off the argmax");
                assert_eq!(t2.0, t4.0, "{p:?}: MTP off: warm vs cold ids");
                assert!(bits(&t2.1, &t4.1).iter().all(|&d| d == 0), "{p:?}: MTP off: warm vs cold logits");
                let want = [(&p1, &t1), (&p2, &t2), (&p2, &t3), (&p2, &t4)];
                for (name, e) in [("N 3 transcript drafts", &mut e1), ("N 2 own drafts", &mut e2)] {
                    let _ = e.rows_mut().mtp_report();
                    for (k, (prompt, w)) in want.iter().enumerate() {
                        *gold.lock().unwrap() = transcript(prompt, &w.0);
                        if k == 3 {
                            e.reset();
                        }
                        let got = turn(e, prompt, n, p);
                        assert_eq!(got.2, w.2, "{name} {p:?} T{}: cached rows", k + 1);
                        assert_eq!(got.0, w.0, "{name} {p:?} T{}: ids vs MTP off", k + 1);
                        let d = bits(&got.1, &w.1);
                        assert!(d.iter().all(|&x| x == 0), "{name} {p:?} T{}: logits rows differ in bits {d:?}", k + 1);
                    }
                    let st = e.rows_mut().run.mtp_take_stats().unwrap();
                    eprintln!("glm5_mtp serve {name} {p:?}: {}", st.summary());
                    assert!(st.steps > 0 && st.drafts > 0, "{name} {p:?}: the verify ran: {st:?}");
                    if name.starts_with("N 3") {
                        assert!(st.accepted > 0 && st.kda_restore_steps > 0, "{name} {p:?}: accepted drafts and rollbacks: {st:?}");
                        if p == Policy::Greedy {
                            // 4 requests x 19 decode steps; accepted ids come back without a verify
                            assert!(st.steps < 4 * 19, "{name} {p:?}: {st:?}");
                        }
                    }
                }
            }
            drop((e0, e1, e2));
        }
    }

    /// every KDA state, the trunk's MLA rows `0 .. rows`, the block's rows `0 .. rows - 1` and
    /// its pending row
    unsafe fn seq_state(d: &Glm5Device, rows: usize) -> (Vec<Vec<u8>>, Vec<Vec<u8>>, Vec<Vec<u8>>) {
        cuda::sync();
        let (kd, md) = (KdaDims::of(&d.o.g), crate::glm5_mla::MlaDims::of(&d.o.g));
        let kda = d.run.kda_states().flat_map(|s| [cuda::dtoh_t::<u8>(s.s, kd.state_floats() * 4), cuda::dtoh_t::<u8>(s.conv, kd.conv_floats() * 4)]).collect();
        let mla = d.run.mla_caches().flat_map(|c| [cuda::dtoh_t::<u8>(c.latent, rows * md.latent_bytes_per_token() as usize), cuda::dtoh_t::<u8>(c.index, rows * md.indexer_bytes_per_token() as usize)]).collect();
        (kda, mla, d.run.mtp_state_bytes(rows - 1))
    }

    /// The prefix cache with MTP: T1 (21-id prompt, 9 greedy decode steps whose drafts are all
    /// right, so verified ids are still ahead when the request ends, then a 4-id prefill that
    /// moves the block's pending row), then T2 rolls back to the prompt snapshot and prefills an
    /// 11-id suffix. Right after that prefill, every KDA state,
    /// the trunk's MLA rows, the block's cache rows and its pending row equal, bit for bit, a
    /// sequence that prefilled the same two pieces without the answer in between; the next 12
    /// greedy ids equal it and a cold prefill of the whole prompt.
    #[test]
    #[ignore = "needs the GPU (about 6 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_mtp_serve_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mtp_serve_gpu_rollback_restores_the_block() {
        let g = geo();
        let s = synth_model(&g, REC);
        let p1: Vec<i64> = (0..21).map(|i| (i * 53 + 29) % 2048).collect();
        let cap = 64;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut e0 = Glm5Engine::new(device(&s.path, &g, 0, cap, 0), true);
            let mut e = Glm5Engine::new(device(&s.path, &g, 3, cap, 0x0192_2003), true);
            let t1 = turn(&mut e0, &p1, 10, Policy::Greedy);
            let mut p2 = p1.clone();
            p2.extend_from_slice(&t1.0[..6]);
            p2.extend([9, 99, 999, 1999, 1]);
            let gold: Arc<Mutex<HashMap<usize, i64>>> = Arc::new(Mutex::new(transcript(&p1, &t1.0)));
            let gh = gold.clone();
            e.rows_mut().run.set_mtp_hook(Some(Box::new(move |pos, d| gh.lock().unwrap().get(&pos).copied().unwrap_or(d))));
            let w1 = turn(&mut e, &p1, 10, Policy::Greedy);
            assert_eq!(w1.0, t1.0, "T1 ids vs MTP off");
            let st = e.rows_mut().run.mtp_take_stats().unwrap();
            assert!(st.accepted > 0, "T1 must accept drafts: {st:?}");
            // a prefill after the snapshot moves the pending row (the engine allows it;
            // `glm_generate` prefills once per request, before its snapshot)
            e.prefill(&[7, 70, 700, 1700]).unwrap();
            // T2 warm: rollback to the snapshot at 21, the suffix prefilled
            assert_eq!(e.decide(&p2).reuse, Some((0, 21)));
            e.rollback(21).unwrap();
            let first_w = e.prefill(&p2[21..]).unwrap();
            let warm = seq_state(e.rows(), p2.len());
            let mut ids_w = vec![first_w];
            for _ in 0..12 {
                ids_w.push(e.decode_step(*ids_w.last().unwrap()).unwrap());
            }
            // the same two prefills without the answer in between
            e.reset();
            e.prefill(&p1).unwrap();
            let first_r = e.prefill(&p2[21..]).unwrap();
            let refs = seq_state(e.rows(), p2.len());
            assert!(warm.0 == refs.0, "the KDA states after the warm prefill differ");
            assert!(warm.1 == refs.1, "the trunk's MLA rows after the warm prefill differ");
            assert!(warm.2[2] == refs.2[2], "the block's pending head-norm row after the warm prefill differs");
            assert!(warm.2[..2] == refs.2[..2], "the block's cache rows 0..{} after the warm prefill differ", p2.len() - 1);
            let mut ids_r = vec![first_r];
            for _ in 0..12 {
                ids_r.push(e.decode_step(*ids_r.last().unwrap()).unwrap());
            }
            assert_eq!(ids_w, ids_r, "warm vs the two prefills: ids");
            // cold: the whole prompt at once (other block windows)
            e.reset();
            let mut ids_c = vec![e.prefill(&p2).unwrap()];
            let cold = seq_state(e.rows(), p2.len());
            for _ in 0..12 {
                ids_c.push(e.decode_step(*ids_c.last().unwrap()).unwrap());
            }
            assert_eq!(ids_w, ids_c, "warm vs cold: ids");
            assert!(warm.0 == cold.0 && warm.1 == cold.1, "warm vs cold: the trunk's states");
            eprintln!(
                "glm5_mtp serve rollback: block rows warm vs cold (windows from 0 vs 0+21): latent equal {}, indexer equal {}, pending row equal {}",
                warm.2[0] == cold.2[0],
                warm.2[1] == cold.2[1],
                warm.2[2] == cold.2[2]
            );
            drop((e0, e));
        }
    }
}

/// The host twin of [`Rows`]: the cache shapes of glm5_next in miniature, for the engine's
/// tests and `serve`'s (a bin cannot see a `cfg(test)` item of the library, so this is public
/// and tiny; nothing outside the tests constructs it).
pub mod fake {
    use super::Rows;

    /// A recurrent state (a running 64-bit mix of every row fed, split over two f32 bit
    /// patterns: the KDA part) and one position-indexed row per token (the MLA part); a head
    /// row's logits are a function of both over `cands` (every other id is -1e30). A stale row
    /// below `pos` or a state that was not brought back changes the logits.
    pub struct FakeRows {
        pub n_ctx: usize,
        pub vocab: usize,
        pub cands: Vec<u32>,
        state: u64,
        mla: Vec<i64>,
        logits: Vec<f32>,
        /// a row at this position fails (an NVMe read error, say)
        pub fail_at: Option<usize>,
        /// an allocation of a row at this position fails (`cuda::AllocFailed::raise`: inside a
        /// request scope the `AllocFailed` panic `serve` answers with a 503)
        pub alloc_fail_at: Option<usize>,
        /// rows run, heads run
        pub rows_run: usize,
        pub heads_run: usize,
        /// `CROW_GLM_MAX_BATCH`: the parked sequences (state, rows, logits; the current slot's
        /// entry is empty), the current slot, and the rows of every `decode_batch` call
        parked: Vec<(u64, Vec<i64>, Vec<f32>)>,
        cur: usize,
        pub batches: Vec<usize>,
    }

    impl FakeRows {
        pub fn new(n_ctx: usize, vocab: usize, cands: Vec<u32>) -> FakeRows {
            FakeRows::with_slots(n_ctx, vocab, cands, 1)
        }

        /// `slots` sequences (`CROW_GLM_MAX_BATCH`)
        pub fn with_slots(n_ctx: usize, vocab: usize, cands: Vec<u32>, slots: usize) -> FakeRows {
            let parked = (0..slots).map(|i| if i == 0 { (0, Vec::new(), Vec::new()) } else { (0, vec![-1; n_ctx], vec![0.0; vocab]) }).collect();
            FakeRows { n_ctx, vocab, cands, state: 0, mla: vec![-1; n_ctx], logits: vec![0.0; vocab], fail_at: None, alloc_fail_at: None, rows_run: 0, heads_run: 0, parked, cur: 0, batches: Vec::new() }
        }
    }

    fn mix(a: u64, b: u64) -> u64 {
        let mut x = a ^ b.wrapping_mul(0x9E37_79B9_7F4A_7C15);
        x ^= x >> 29;
        x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
        x ^ (x >> 32)
    }

    impl Rows for FakeRows {
        fn n_ctx(&self) -> usize {
            self.n_ctx
        }
        fn vocab(&self) -> usize {
            self.vocab
        }
        fn layers(&self) -> usize {
            45
        }
        fn state_floats(&self) -> usize {
            2
        }
        unsafe fn row(&mut self, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String> {
            if self.fail_at == Some(pos) {
                return Err(format!("fake: row {pos} failed"));
            }
            if self.alloc_fail_at == Some(pos) {
                crate::cuda::AllocFailed { what: format!("fake: the buffer of row {pos}"), bytes: 1 << 20, free: 0, result: "CUDA_ERROR_OUT_OF_MEMORY".into() }.raise();
            }
            self.rows_run += 1;
            self.mla[pos] = tok;
            self.state = mix(self.state, tok as u64 ^ ((pos as u64) << 40));
            if !head {
                return Ok(None);
            }
            self.heads_run += 1;
            // the attention part: every row 0..=pos, position-weighted
            let att = self.mla[..=pos].iter().enumerate().fold(0u64, |a, (i, &t)| mix(a, (t as u64) ^ ((i as u64) << 32)));
            let mut best = (f32::NEG_INFINITY, 0usize);
            self.logits.iter_mut().for_each(|l| *l = -1e30);
            for (k, &c) in self.cands.iter().enumerate() {
                let v = (mix(self.state ^ att, k as u64) >> 40) as f32 / 1024.0;
                self.logits[c as usize] = v;
                if v > best.0 {
                    best = (v, c as usize);
                }
            }
            Ok(Some(best.1 as i64))
        }
        unsafe fn logits(&self) -> Vec<f32> {
            self.logits.clone()
        }
        unsafe fn set_logits(&mut self, row: &[f32]) {
            self.logits.copy_from_slice(row);
        }
        unsafe fn save_state(&self, into: &mut [f32]) {
            into[0] = f32::from_bits(self.state as u32);
            into[1] = f32::from_bits((self.state >> 32) as u32);
        }
        unsafe fn load_state(&mut self, from: &[f32]) {
            self.state = from[0].to_bits() as u64 | ((from[1].to_bits() as u64) << 32);
        }
        unsafe fn zero_state(&mut self) {
            self.state = 0;
        }
        fn counters(&self) -> ([u64; 3], u64) {
            ([self.rows_run as u64 * 8, 0, 0], 0)
        }
        fn slots(&self) -> usize {
            self.parked.len().max(1)
        }
        unsafe fn use_slot(&mut self, s: usize) -> Result<(), String> {
            if s >= self.slots() {
                return Err(format!("fake: sequence slot {s} of {}", self.slots()));
            }
            if s == self.cur {
                return Ok(());
            }
            for i in [self.cur, s] {
                let p = &mut self.parked[i];
                std::mem::swap(&mut self.state, &mut p.0);
                std::mem::swap(&mut self.mla, &mut p.1);
                std::mem::swap(&mut self.logits, &mut p.2);
            }
            self.cur = s;
            Ok(())
        }
        unsafe fn decode_batch(&mut self, rows: &[(usize, i64, usize)]) -> Result<Vec<i64>, String> {
            self.batches.push(rows.len());
            let mut ids = Vec::with_capacity(rows.len());
            for &(s, tok, pos) in rows {
                self.use_slot(s)?;
                ids.push(self.decode(tok, pos)?);
            }
            Ok(ids)
        }
    }
}

#[cfg(test)]
mod tests_batch {
    //! `CROW_GLM_MAX_BATCH` on the host twin: sequence slots and the batched decode step
    use super::fake::FakeRows;
    use super::*;

    const CANDS: std::ops::Range<u32> = 10..40;

    /// the id fed after `next` at generated index `i`: the argmax, every fifth one forced to
    /// another id (a sampled draw / an injected id)
    fn feed(i: usize, next: i64) -> i64 {
        if i % 5 == 4 {
            10 + (next * 7 + 3) % 30
        } else {
            next
        }
    }

    /// one sequence alone: prefill `prompt`, then `n` ids fed back; (fed ids, logits rows)
    unsafe fn solo(prompt: &[i64], n: usize) -> (Vec<i64>, Vec<Vec<f32>>) {
        let mut e = Glm5Engine::new(FakeRows::new(256, 64, CANDS.collect()), true);
        let mut next = e.prefill(prompt).unwrap();
        let (mut ids, mut rows) = (Vec::new(), vec![e.logits()]);
        for i in 0..n {
            let x = feed(i, next);
            ids.push(x);
            next = e.decode_step(x).unwrap();
            rows.push(e.logits());
        }
        (ids, rows)
    }

    /// Three sequences in three slots, joining at steps 0, 2 and 5 and running different
    /// lengths: every slot's fed ids, held ids and logits rows (after its prompt and after every
    /// step) are its solo run's; the steps carried 1, 2 and 3 rows.
    #[test]
    fn batched_slots_are_their_solo_sequences() {
        let prompts: Vec<Vec<i64>> = vec![(0..9).map(|i| 10 + i * 3 % 30).collect(), (0..14).map(|i| 11 + (i * 7) % 29).collect(), vec![12, 13, 14]];
        let (join, len) = ([0usize, 2, 5], [12usize, 9, 10]);
        unsafe {
            let want: Vec<_> = (0..3).map(|k| solo(&prompts[k], len[k])).collect();
            let mut e = Glm5Engine::new(FakeRows::with_slots(256, 64, CANDS.collect(), 3), true);
            assert_eq!(e.slots(), 3);
            let mut next = [0i64; 3];
            let mut got: Vec<(Vec<i64>, Vec<Vec<f32>>)> = vec![Default::default(); 3];
            for step in 0..20 {
                for k in 0..3 {
                    if join[k] == step {
                        e.select(k).unwrap();
                        e.reset();
                        next[k] = e.prefill(&prompts[k]).unwrap();
                        e.snapshot(next[k]);
                        got[k].1.push(e.logits());
                    }
                }
                let active: Vec<usize> = (0..3).filter(|&k| join[k] <= step && got[k].0.len() < len[k]).collect();
                if active.is_empty() {
                    continue;
                }
                let steps: Vec<(usize, i64)> = active.iter().map(|&k| (k, feed(got[k].0.len(), next[k]))).collect();
                let ids = e.decode_batch(&steps).unwrap();
                for (j, &(k, x)) in steps.iter().enumerate() {
                    got[k].0.push(x);
                    next[k] = ids[j];
                    e.select(k).unwrap();
                    got[k].1.push(e.logits());
                }
            }
            for k in 0..3 {
                assert_eq!(got[k].0, want[k].0, "slot {k}: fed ids vs solo");
                assert!(got[k].1.iter().zip(&want[k].1).all(|(a, b)| a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())), "slot {k}: logits rows vs solo");
                assert_eq!(got[k].1.len(), want[k].1.len());
                let held: Vec<i64> = prompts[k].iter().chain(&got[k].0).copied().collect();
                assert_eq!(e.history_of(k), held.as_slice(), "slot {k}: held ids");
            }
            let b = &e.rows().batches;
            assert!(b.contains(&1) && b.contains(&2) && b.contains(&3), "step sizes {b:?}");
        }
    }

    /// Each slot keeps its own prompt snapshot: a request that extends slot 1's prompt is warm
    /// in slot 1 only, and a turn there equals the cold re-prefill
    #[test]
    fn every_slot_keeps_its_own_prefix_cache() {
        let (p0, p1): (Vec<i64>, Vec<i64>) = ((10..20).collect(), (20..35).collect());
        unsafe {
            let mut e = Glm5Engine::new(FakeRows::with_slots(256, 64, CANDS.collect(), 2), true);
            for (k, p) in [(0, &p0), (1, &p1)] {
                e.select(k).unwrap();
                let n = e.prefill(p).unwrap();
                e.snapshot(n);
                e.decode_batch(&[(k, n)]).unwrap();
            }
            let mut p2 = p1.clone();
            p2.extend([11, 12, 13]);
            assert_eq!(e.decide_in(0, &p2).reuse, None);
            assert_eq!(e.decide_in(1, &p2).reuse.map(|r| r.1), Some(p1.len()));
            assert_eq!((e.snapshot_pos_of(0), e.snapshot_pos_of(1)), (Some(p0.len()), Some(p1.len())));
            e.select(1).unwrap();
            e.rollback(p1.len()).unwrap();
            let warm = e.prefill(&p2[p1.len()..]).unwrap();
            let mut cold = Glm5Engine::new(FakeRows::new(256, 64, CANDS.collect()), true);
            assert_eq!(warm, cold.prefill(&p2).unwrap());
            assert_eq!(e.logits(), cold.logits());
            // slot 0 untouched
            assert_eq!(e.history_of(0).len(), p0.len() + 1);
        }
    }

    /// A failed step resets the slots it carried (their next requests are cold); a slot outside
    /// the step keeps its sequence. Refusals: a slot twice, a slot the engine does not hold.
    #[test]
    fn a_failed_batch_step_resets_its_slots_only() {
        unsafe {
            let mut e = Glm5Engine::new(FakeRows::with_slots(256, 64, CANDS.collect(), 3), true);
            let mut next = [0i64; 3];
            for k in 0..3 {
                e.select(k).unwrap();
                next[k] = e.prefill(&[10 + k as i64, 20, 30, 15]).unwrap();
                e.snapshot(next[k]);
            }
            assert!(e.decode_batch(&[(0, next[0]), (0, next[0])]).unwrap_err().contains("twice"));
            assert!(e.decode_batch(&[(3, next[0])]).is_err());
            e.rows_mut().fail_at = Some(4);
            let err = e.decode_batch(&[(0, next[0]), (1, next[1])]).unwrap_err();
            assert!(err.contains("failed"), "{err}");
            assert_eq!((e.history_of(0).len(), e.snapshot_pos_of(0)), (0, None));
            assert_eq!((e.history_of(1).len(), e.snapshot_pos_of(1)), (0, None));
            assert_eq!((e.history_of(2).len(), e.snapshot_pos_of(2)), (4, Some(4)));
            // one slot without CROW_GLM_MAX_BATCH
            let mut one = Glm5Engine::new(FakeRows::new(256, 64, CANDS.collect()), true);
            assert_eq!(one.slots(), 1);
            assert!(one.select(1).is_err());
        }
    }
}

#[cfg(test)]
mod tests_batch_gpu {
    //! `CROW_GLM_MAX_BATCH` on the GPU: `Glm5Engine` over a `Glm5Device` of the synthetic
    //! 8-layer glm5_next model of `tests_192_serve` (layers 0-2 KDA + dense, 3-7 MoE with 16
    //! MUL1 experts, DSA at 3 and 7, vocab 2048), V 3 + P 4 tiers. `#[ignore]`: CI has no GPU.
    //! Run with `cargo test --release --lib glm5_batch_gpu -- --ignored --nocapture --test-threads 1`.
    use super::*;
    use crate::cnq::Cnq;
    use crate::glm5_flags::tests::synth_model;
    use crate::glm5_moe::MoeGeo;
    use crate::glm5_tiers::TierSizes;

    const REC: u64 = 9_474_048;

    fn geo() -> Glm5Geo {
        let mut g = Glm5Geo::GLM_5_3_FLASH;
        (g.layers, g.dense_prefix, g.experts, g.topk, g.vocab) = (8, 3, 16, 8, 2048);
        g
    }

    /// a device of the synthetic model at `cap` rows with `slots` sequence slots
    unsafe fn device(path: &str, g: &Glm5Geo, cap: usize, slots: usize) -> Glm5Device {
        device_sized(path, g, cap, slots, TierSizes { vram: 3, pinned: 4 })
    }

    /// [`device`] with the tier sizes per layer
    unsafe fn device_sized(path: &str, g: &Glm5Geo, cap: usize, slots: usize, sizes: TierSizes) -> Glm5Device {
        let (spec, _) = crate::nvme_source::glm5_record_of_container(path).unwrap();
        let moe = MoeGeo::new(g, spec).unwrap();
        let mut cnq = Cnq::open_checked(path).unwrap();
        let old = std::env::var(gt::MAX_BATCH_ENV).ok();
        std::env::set_var(gt::MAX_BATCH_ENV, slots.to_string());
        let mut run = Glm5Run::load(&mut cnq, g, &moe, cap, &mut |s| eprintln!("{s}"));
        match old {
            Some(o) => std::env::set_var(gt::MAX_BATCH_ENV, o),
            None => std::env::remove_var(gt::MAX_BATCH_ENV),
        }
        run.set_slots(slots).unwrap();
        let tiers = ExpertTiers::new(&cnq, path, g, &moe, sizes, 1, slots * g.topk).unwrap();
        let o = Opened { cnq, path: path.to_string(), g: *g, moe, spec, records: 0, constants: 0 };
        let plan = TierPlan { vram_ceiling: 0, fixed_bytes: 0, unit_bytes: 0, hot: sizes.vram, pinned: sizes.pinned, nvme: g.experts - sizes.vram - sizes.pinned };
        Glm5Device { o, run, tiers, plan, n_ctx: cap, kd: KdaDims::of(g) }
    }

    fn feed(i: usize, next: i64) -> i64 {
        if i % 5 == 4 {
            (next * 7 + 3) % 2048
        } else {
            next
        }
    }

    fn same_bits(a: &[f32], b: &[f32]) -> bool {
        a.len() == b.len() && a.iter().zip(b).all(|(x, y)| x.to_bits() == y.to_bits())
    }

    /// Four sequences in four slots, joining at steps 0, 1, 3 and 6 with prompts of 5 .. 23
    /// ids, 14 fed ids each (every fifth forced): every slot's greedy ids and logits rows (after
    /// its prompt and after every step) are, bit for bit, those of the same sequence alone on a
    /// one-slot device; the steps carried up to 4 rows in one trunk pass.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_batch_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_batch_gpu_every_slot_is_its_solo_sequence() {
        batch_is_solo(&[]);
    }

    /// Cross-wiring 2: the CPU lane in a batched step (one pool run per row). With every expert
    /// in pinned (V 0 + P 16 of 16) `CROW_GLM_CPU_LANE=1` gives every pick to the CPU in the solo
    /// runs and in the batch, so every slot's ids and logits rows are its solo lane run's bits;
    /// the batch's steps ran rows of several slots through the lane.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_batch_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_batch_gpu_every_slot_is_its_solo_sequence_with_the_cpu_lane() {
        let lane = [("CROW_GLM_CPU_LANE", "1"), ("CROW_PINNED_ALLOC", "host")];
        let _all = EnvGuard::set(&lane);
        batch_is_solo_sized(&[], TierSizes { vram: 0, pinned: 16 }, true);
    }

    /// The batch test with the decode switches of the integration on the four-slot device only
    /// (the solo runs stay on the default path): flags + stager + controller + LA + prefetch +
    /// overlap. A solo step of the batch goes through the lookahead (`decode_la`), a wider one
    /// through `decode_batch`; the row ahead must be dropped before another slot comes in.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_batch_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_batch_gpu_every_slot_is_its_solo_sequence_under_the_controller_and_la() {
        batch_is_solo(&[
            ("CROW_GLM_FLAGS", "1"),
            ("CROW_GLM_STAGER", "1"),
            ("CROW_GLM_CONTROLLER", "1"),
            ("CROW_GLM_LA", "1"),
            ("CROW_GLM_PREFETCH", "1"),
            ("CROW_GLM_SHARED_OVERLAP", "1"),
        ]);
    }

    /// The batch test under the integration's full template arm (`glm5_int_tests::full_arm_env`,
    /// `CROW_GLM_MAX_BATCH` as the device sets it), solo runs on the default path; both at prompt
    /// chunk 12 (`CROW_CHUNK`, whose bits differ from row by row by design, #186): every slot's
    /// ids and logits rows bit-identical to its solo default run.
    #[test]
    #[ignore = "needs the GPU (about 4 GB VRAM, a 2.3 GB synthetic container in the temp dir): cargo test --release --lib glm5_batch_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_batch_gpu_every_slot_is_its_solo_sequence_under_the_full_arm() {
        let dir = std::env::temp_dir().join(format!("crow-batch-full-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let warm = crate::glm5_int_tests::synth_warm(&dir);
        let full: Vec<(&str, String)> = crate::glm5_int_tests::full_arm_env(&warm).into_iter().filter(|(k, _)| !matches!(*k, "CROW_CHUNK" | "CROW_GLM_MAX_BATCH")).collect();
        let together: Vec<(&str, &str)> = full.iter().map(|(k, v)| (*k, v.as_str())).collect();
        let _chunk = EnvGuard::set(&[("CROW_CHUNK", "12")]);
        batch_is_solo(&together);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// set `kv` for the life of the guard, the old values back on drop
    pub(crate) struct EnvGuard(Vec<(String, Option<String>)>);

    impl EnvGuard {
        pub(crate) fn set(kv: &[(&str, &str)]) -> EnvGuard {
            let old = kv.iter().map(|(k, _)| (k.to_string(), std::env::var(k).ok())).collect();
            for (k, v) in kv {
                std::env::set_var(k, v);
            }
            EnvGuard(old)
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            for (k, v) in &self.0 {
                match v {
                    Some(v) => std::env::set_var(k, v),
                    None => std::env::remove_var(k),
                }
            }
        }
    }

    /// the body of the batch tests: `together` is set while the four-slot device is built and runs
    pub(crate) fn batch_is_solo(together: &[(&str, &str)]) {
        batch_is_solo_sized(together, TierSizes { vram: 3, pinned: 4 }, false);
    }

    /// [`batch_is_solo`] with the tier sizes; `lane`: the four-slot device's CPU lane must have
    /// run batched steps
    pub(crate) fn batch_is_solo_sized(together: &[(&str, &str)], sizes: TierSizes, lane: bool) {
        let g = geo();
        let s = synth_model(&g, REC);
        let prompts: Vec<Vec<i64>> = [5usize, 23, 11, 17].iter().enumerate().map(|(k, &n)| (0..n as i64).map(|i| (i * (31 + 2 * k as i64) + 7 * k as i64 + 1) % 2048).collect()).collect();
        let (join, n) = ([0usize, 1, 3, 6], 14usize);
        let cap = 23 + n + 4;
        unsafe {
            let _ctx = cuda::Ctx::init();
            // alone
            let mut e0 = Glm5Engine::new(device_sized(&s.path, &g, cap, 1, sizes), true);
            let mut want: Vec<(Vec<i64>, Vec<Vec<f32>>)> = Vec::new();
            for p in &prompts {
                e0.reset();
                let mut next = e0.prefill(p).unwrap();
                let (mut ids, mut rows) = (Vec::new(), vec![e0.logits()]);
                for i in 0..n {
                    let x = feed(i, next);
                    ids.push(x);
                    next = e0.decode_step(x).unwrap();
                    rows.push(e0.logits());
                }
                want.push((ids, rows));
            }
            drop(e0);
            // together
            let _env = EnvGuard::set(together);
            let mut e = Glm5Engine::new(device_sized(&s.path, &g, cap, 4, sizes), true);
            assert_eq!(e.slots(), 4);
            let mut next = [0i64; 4];
            let mut got: Vec<(Vec<i64>, Vec<Vec<f32>>)> = vec![Default::default(); 4];
            let mut widest = 0;
            for step in 0..(6 + n) {
                for k in 0..4 {
                    if join[k] == step {
                        e.select(k).unwrap();
                        e.reset();
                        next[k] = e.prefill(&prompts[k]).unwrap();
                        got[k].1.push(e.logits());
                    }
                }
                let active: Vec<usize> = (0..4).filter(|&k| join[k] <= step && got[k].0.len() < n).collect();
                if active.is_empty() {
                    continue;
                }
                widest = widest.max(active.len());
                let steps: Vec<(usize, i64)> = active.iter().map(|&k| (k, feed(got[k].0.len(), next[k]))).collect();
                let ids = e.decode_batch(&steps).unwrap();
                for (j, &(k, x)) in steps.iter().enumerate() {
                    got[k].0.push(x);
                    next[k] = ids[j];
                    e.select(k).unwrap();
                    got[k].1.push(e.logits());
                }
            }
            assert_eq!(widest, 4, "a step carried every slot");
            if lane {
                let (_, experts, runs) = e.rows().tiers.cpu_lane_clock().read();
                eprintln!("glm5_batch lane: {experts} CPU experts in {runs} pool runs");
                assert!(experts > 0, "the CPU lane computed nothing");
            }
            for k in 0..4 {
                assert_eq!(got[k].0, want[k].0, "slot {k}: ids vs alone");
                let bad: Vec<usize> = got[k].1.iter().zip(&want[k].1).enumerate().filter(|(_, (a, b))| !same_bits(a, b)).map(|(i, _)| i).collect();
                assert!(bad.is_empty() && got[k].1.len() == want[k].1.len(), "slot {k}: logits rows {bad:?} differ in bits from the solo run");
                let held: Vec<i64> = prompts[k].iter().chain(&got[k].0).copied().collect();
                assert_eq!(e.history_of(k), held.as_slice(), "slot {k}: held ids");
            }
            eprintln!("glm5_batch {together:?}: 4 slots, {n} ids each, ids and logits rows bit-identical to the solo runs");
            drop(e);
        }
    }
}
