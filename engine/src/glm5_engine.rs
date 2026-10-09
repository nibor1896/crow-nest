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

use crate::cuda;
use crate::geo::{Family, GLM5_NEXT_DENSE_BYTES, HOST_PINNED_CAP};
use crate::glm5_kda::KdaDims;
use crate::glm5_template;
use crate::glm5_tiers::{self as gt, ExpertTiers, Glm5Run, Opened};
use crate::manager::{derive_host_pinned_budget, glm5_plan_table, plan_glm5_next, TierPlan};
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
        let (states, input, plan) = plan_glm5_next(&o.g, context, free, budget, GLM5_NEXT_DENSE_BYTES, o.spec.bytes, crate::gen::pf_tg(), crate::gen::pf_async_on())?;
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
        let run = Glm5Run::load(&mut o.cnq, &o.g, &o.moe, context, log);
        let tiers = ExpertTiers::new(&o.cnq, &o.path, &o.g, &o.moe, sizes, readers, o.g.topk)?;
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
        self.run.kda_states().count() * (s + c)
    }
    unsafe fn row(&mut self, tok: i64, pos: usize, head: bool) -> Result<Option<i64>, String> {
        self.run.row(&mut self.o.cnq, &mut self.tiers, tok, pos, head)
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
        cuda::sync();
    }
    unsafe fn zero_state(&mut self) {
        for k in self.run.kda_states() {
            k.reset();
        }
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
        match self.rows.row(id, pos, true) {
            Ok(Some(next)) => {
                self.history.push(id);
                Ok(next)
            }
            Ok(None) => unreachable!("a head row returns its id"),
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
