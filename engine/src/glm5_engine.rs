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
        let (states, input, plan) = serve_plan(&o.g, context, free.saturating_sub(mtp_reserved), budget, o.spec.bytes)?;
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
        // #192: a verify call stages the experts of 1 + N rows at once; the CPU lane is refused
        let mut tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, readers, (1 + mtp_n) * o.g.topk)?;
        if mtp_n > 0 {
            if let Err(e) = Glm5Run::spec_check(mtp_n, &tiers, o.g.topk) {
                tiers.free();
                run.free();
                return Err(e);
            }
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
        self.row(tok, pos, true)?.ok_or_else(|| format!("glm5: the head row at {pos} gave no id"))
    }
    fn mtp_drafts(&self) -> usize {
        self.run.mtp_drafts()
    }
    fn mtp_report(&mut self) -> Option<String> {
        self.run.mtp_take_stats().map(|s| s.summary())
    }
    unsafe fn logits(&self) -> Vec<f32> {
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
    fn counters(&self) -> ([u64; 3], u64) {
        let a = self.tiers.cache.counters().iter().fold([0u64; 3], |a, c| [a[0] + c[0], a[1] + c[1], a[2] + c[2]]);
        (a, self.tiers.nvme_bytes)
    }
}

impl Drop for Glm5Device {
    fn drop(&mut self) {
        // SAFETY: the context `boot::open_glm5` made is still current (the caller drops the
        // engine before it), and no launch is pending between requests
        unsafe {
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
}

impl<R: Rows> Glm5Engine<R> {
    /// `cache_on` is `CROW_PREFIX_CACHE != 0` in `serve`; off, every request is a cold start
    pub fn new(rows: R, cache_on: bool) -> Glm5Engine<R> {
        Glm5Engine { rows, history: Vec::new(), cache_on, snap: Snapshot::default() }
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
        /// rows run, heads run
        pub rows_run: usize,
        pub heads_run: usize,
    }

    impl FakeRows {
        pub fn new(n_ctx: usize, vocab: usize, cands: Vec<u32>) -> FakeRows {
            FakeRows { n_ctx, vocab, cands, state: 0, mla: vec![-1; n_ctx], logits: vec![0.0; vocab], fail_at: None, rows_run: 0, heads_run: 0 }
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
    }
}
