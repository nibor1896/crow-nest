//! #31 A9 - prefix cache: KV, QSA ring, GDN and the PLE conv held across turns (spec section 7).
//!
//! Purpose:
//!
//! - Crow resends the whole transcript every turn (`crow_core.py:4672-4700`).
//! - Without this, every turn pays a full prefill: 16k = 24.13 s wall (measured, spec 7.1).
//! - The invariant is spec 7.5: the cache never changes the output.
//! - Where the correct state cannot be proven present, the answer is recompute.
//!
//! Detection (spec 7.4), ids only, never text:
//!
//! | symbol | value |
//! |---|---|
//! | `L` | longest common prefix length of the request ids and `Engine::history` |
//! | `L` (#114) | cut to the first row of the first image whose span (start, len, content hash) is not held identically, `common_prefix_len_mm` |
//! | `P` | `max { S_pos : S_pos <= L and S_pos < request length }` over the held snapshots |
//! | `P` (#100) | a snapshot that holds its LOGITS ROW may also sit AT the request length |
//! | cold | no such `S_pos` exists; `Engine::reset_to_zero` runs and the slot is dropped |
//!
//! - Only a PREFILL CLEAN snapshot is a reuse candidate; see the section below.
//!
//! - `Engine::history` covers prompt AND generated ids (`gen.rs:2698`, `gen.rs:2945`).
//! - `S_pos < request length` is a guard, not a spec change: `prefill` of an empty
//!   slice has no last position to return a greedy id from.
//! - #100: the guard made an IDENTICAL re-request (Crow #217's same-prefix retry) roll
//!   back to an older snapshot, or go cold. A snapshot therefore also keeps the last
//!   position's logits row (`s.logits` row 0, V f32 = 993,280 B) and its greedy id, the
//!   only two things `prefill` leaves behind besides the state (`gen.rs:3973-3979`).
//!   A request whose ids equal such a snapshot's prefix rolls back onto it, the row is
//!   uploaded back, and NO token is prefilled: the first draw reads the same row the
//!   first prefill wrote (`arm_sampler`, `draw_biased`, the #91 logprobs readback).
//! - llama.cpp and vLLM recompute the last token instead (`n_past--`, "need to evaluate
//!   at least 1 token"; vLLM `max_cache_hit_length = num_tokens - 1`); for recurrent
//!   state llama.cpp needs a checkpoint 4 tokens early for that (its PR #20288). The row
//!   is cheaper here: ONE held conversation, and the state is copied at that instant.
//! - A slot filled from a slot file (#32 A10) carries no row and keeps the old guard.
//!
//! What is snapshotted, and why (spec 7.6):
//!
//! | buffer | shape | evidence | why it needs a snapshot |
//! |---|---|---|---|
//! | `ThreeStates::gdn_s` | `[36][48*128*128]` f32 | `manager.rs:60`, `kernels.rs:987-1012` | `delta_rule_persist` folds each token into S in place; S holds no position |
//! | `ThreeStates::gdn_conv` | `[36][10240*3]` f32 | `manager.rs:61`, `kernels.rs:920-928` | `conv_state_update` shifts a 3 wide window; no position in it |
//! | `Ple::state` | `[10240][9]` f32 | `gen.rs:204`, `kernels.rs:2839-2867` | dilated conv over nine history rows; same argument (spec 7.3, fifth state) |
//! | `ThreeStates::qsa_keys` | `[12][ring][128]` f32 | `manager.rs:56`, `kernels.rs:1759-1769` | ring row = `pos % ring`; block `P/4` needs the up to 3 raw rows below P |
//!
//! What needs NO snapshot, and why:
//!
//! | buffer | why |
//! |---|---|
//! | `ThreeStates::kv_buf` | row address is the absolute slot `pos`, RoPE applied at that position (`manager.rs:238-249`, `manager.rs:305-311`); rows `>= P` are unreachable, `ncb` caps the scan at `(pos+1)/4` |
//! | `ThreeStates::qsa_pooled` | absolute block index `pos/4`, same argument, same `ncb` bound (`gen.rs:1647-1661`) |
//! | `Ple` row cache (`cache`, `gs`, `slot_map`) | content addressed by n-gram id, carries no position (`gen.rs:1013-1040`) |
//! | weights, hot set, `Params` | position independent or uploaded per chunk (`gen.rs:2494-2510`) |
//!
//! Host side restored with the buffers:
//!
//! | field | value | evidence |
//! |---|---|---|
//! | `Engine::pos` | `S_pos` | `prefill` reads it as `pos_base` (`gen.rs:2459`) |
//! | `Engine::done_blocks` | the value captured with the buffers | QSA pooled cursor (`gen.rs:1647-1650`) |
//! | `Engine::history` | truncated to `S_pos` | PLE n-gram prefix reads `history` (`gen.rs:2467-2477`) |
//! | `Engine::route_log` | cleared | grows per token under `CROW_ROUTE_DUMP` (`gen.rs:2894`) |
//!
//! Ordering, and why it is not negotiable (A4, `reset.rs` module doc, "The active stream"):
//!
//! - `rollback` calls `Engine::drop_decode_graph` BEFORE the resumed prefill, the same
//!   teardown the cold path `reset_to_zero` runs, from the same single definition.
//! - `reset.rs` holds the measurement and the reason; it is not restated here.
//! - The first `decode_step` of the request re-creates and re-captures (`gen.rs:2740`, `gen.rs:2836`).
//! - `prefill` zeroes S only when `self.pos == 0` (`gen.rs:2433`, `gen.rs:2495`), which is
//!   exactly the cold start, so a restored `pos > 0` leaves the restored S alone.
//!
//! When a snapshot is taken (M2b, robin 2026-09-10, #36: after the prompt, unconditional):
//!
//! | slot | point | position |
//! |---|---|---|
//! | `SLOT_PROMPT` (slot 0) | after the prefill of this turn's prompt | rendered prompt length |
//! | slots 1..`SLOTS` | the previous turns' prompts, largest position first | their rendered lengths |
//!
//! Slot bookkeeping (#100, #101), the invariant: every held slot names a PREFIX of
//! `Engine::history`, and every row below it is a prefill row.
//!
//! - #101: `rollback` onto `P` forgets every slot ABOVE `P`: the prefill and decode after
//!   it rewrite the KV rows `P..`, so those slots name a branch the engine no longer holds
//!   (seen live twice on 2026-09-22 before this rule; llama.cpp: "erase any checkpoints
//!   with pos_max > pos_next"). The cold start forgets all of them (`invalidate`), unless
//!   it PARKS them (#118, the section below).
//! - So the held slots are nested prefixes, and the newest snapshot is the largest one.
//! - #100: a snapshot at a position a slot already holds RE-USES that slot (the same ids
//!   over prefill rows); before, `rotate_right` pushed duplicates that evicted the shared
//!   prefix after three identical requests. llama.cpp: "replace an existing checkpoint at
//!   the same n_tokens instead of appending a duplicate".
//! - Otherwise an empty slot is taken, else the one with the SMALLEST position (the
//!   oldest turn boundary of the nested chain). The chosen slot moves to slot 0.
//!
//! - Holding the last few prompt snapshots is what turns a history edit that diverges
//!   below the newest position from a full cold prefill into a rollback to the previous turn.
//! - The after-answer snapshot of M1 (spec 7.6 point 2) is DROPPED, see the section below.
//! - The last generated id is never fed back, so it is not in `history` and not in `pos`.
//! - ONE held conversation per process (M1): a request that shares no prefix replaces it,
//!   and since #118 may PARK it (below).
//!
//! Parking a conversation for a short unrelated request (#118, 2026-09-25):
//!
//! - A snapshot holds the recurrent state, not the KV rows: those sit in the ONE KV buffer,
//!   addressed by position. A cold request writes rows `0..n`, so before #118 every snapshot
//!   of the held conversation named rows it no longer had, and `invalidate` dropped them.
//!   Measured 2026-09-25: Crow's judge (4,659 prompt ids, 4,947 rows written) cost the next
//!   main turn a cold 118,282-token prefill, 139.5 s; 37 such pairs that day, 4,590 s.
//! - The fix copies only the rows the side request can overwrite. On a cold request
//!   `plan_cold` PARKS the held conversation when its newest snapshot `T` survives it:
//!   KV rows `0..R` and pooled blocks `0..R/4+1` go device to host (`R = min(T, cap)`,
//!   `CROW_PREFIX_PARK_ROWS`, default 8192), `history` and its image spans are kept, and
//!   its slots stay, marked `parked`. A conversation parked earlier is kept instead when
//!   it saves more; else everything is dropped as before.
//! - A parked slot at `P` stays usable while `min(dirty, P) <= R` (`parked_usable`),
//!   `dirty` = the highest row written since the park (`rolled_back` counts the rows a
//!   request wrote before its rollback shortens `history`). `decide` runs the prefix rule
//!   against the parked history too, and the one that reuses more wins; `unpark` then
//!   uploads the rows back and `rollback` proceeds as for any warm request.
//! - Why the state is the one of record: rows `0..R` and blocks `0..R/4+1` come back byte
//!   for byte, rows `R..P` were never written (`dirty <= R`), rows `>= P` are unreachable,
//!   and the recurrent state is the snapshot's. The extra block covers the decode graph's
//!   write to block `done_blocks` ("garbage lands beyond done_blocks", `gen.rs`).
//! - The side request's own snapshot takes an empty or live slot first, then the smallest
//!   parked one, never the parked conversation's last one (`claim`).
//! - llama.cpp saves the WHOLE sequence state to host RAM for the same case (`--cache-ram`,
//!   `server_slot::prompt_save`); vLLM and SGLang keep other prefixes in paged or radix KV.
//!   Here the rest of the KV stays in VRAM, so only the overwritten rows are copied.
//!
//! Memory (spec 7.7, chunk 2048, ring 2052):
//!
//! ```text
//! GDN S    36 * 48 * 128 * 128 * 4 = 113,246,208 B
//! GDN conv 36 * 10240 * 3      * 4 =   4,423,680 B
//! PLE conv      10240 * 9      * 4 =     368,640 B
//! QSA ring 12 * 2052 * 128     * 4 =  12,607,488 B
//! total per slot                   = 130,646,016 B = 124.60 MiB
//! process total (SLOTS = 3)        = 391,938,048 B = 373.80 MiB
//! #100 logits row  248320      * 4 =     993,280 B per slot, 2,979,840 B for SLOTS = 3
//! ```
//!
//! - The logits row is NOT part of `Shape::snapshot_bytes`: that number is the slot
//!   file's `state_bytes` (#32 A10), and the file does not carry the row.
//! - #118 park stash (`park_host_bytes`), pageable host RAM, allocated at the FIRST park:
//!
//! ```text
//! KV      8192 * 12 * 2 * 2 * 256 * 1 (fp8) = 100,663,296 B
//! pooled  12 * 2049 * 128 * 4              =  12,589,056 B
//! total at the default cap, fp8 KV         = 113,252,352 B = 108.0 MiB (bf16 KV: 204.0 MiB)
//! ```
//!
//! - M1 held two slots (261,292,032 B); M2 (robin, 2026-09-10, #36) dropped the
//!   after-answer slot; M3 keeps `SLOTS` prompt snapshots (newest in slot 0) so a
//!   history edit that diverges below the newest still rolls back to an older turn.
//! - Pageable host RAM, allocated once at process start, reused per snapshot.
//! - Not pinned (the pinned tier is capped by `geo::HOST_PINNED_CAP` and derived at boot).
//! - Not VRAM (the loader already clamps N against it, `manager.rs:122-170`).
//! - Snapshots are in-process state, never a file: the shape is only valid for the load
//!   that produced it (spec 7.6, `manager.rs:31`).
//!
//! Prefill clean, and why only such a snapshot may be reused (measured 2026-09-10):
//!
//! - Spec 7.3 holds that a KV row is reusable because it is addressed by the absolute
//!   `pos` and "nothing in the row depends on the request that wrote it".
//! - Measured: a row DOES depend on WHICH CODE PATH wrote it.
//! - `prefill` writes a position through `attn_prompt` over a chunk of `t` tokens.
//! - `decode_step` writes the same position one token at a time through the decode path.
//! - The two are not bit equal, and the pooled QSA block over those rows is not either.
//!
//! | run | reuse point | rows below it written by | ids vs a fresh process |
//! |---|---|---|---|
//! | A9 gate part 2, turn 2 | 16127 (after the answer) | prefill AND `decode_step` | equal over 35 ids, 5 runs |
//! | A9 gate part 3, turn 3 | 16159 (after a prompt) | prefill AND `decode_step` | DIFFERENT, first at id 43 |
//! | A9 probe, turn 4 | 16064 (after a prompt) | prefill only | equal over 49 ids |
//! | A9 control, turn 3 | none, cold, chunk cut changed | prefill only | equal, so the cut is not it |
//!
//! - Evidence: `decode_out/srv-a9.log`, `-control.log`, `-chunkcut.log`, `-probe.log`.
//! - Spec 7.5 is binding: where the correct state cannot be PROVEN present, recompute.
//! - Consequence: a snapshot is a reuse candidate only while every row below its position
//!   was written by `prefill`. That property is called PREFILL CLEAN here.
//!
//! Why the point 1 snapshot is always prefill clean, by induction:
//!
//! - Base: a cold request prefills from position 0, so rows `0..prompt_len` are prefill rows.
//! - Step: a warm request rolls back only to a prefill clean `P`, so rows `0..P` are prefill
//!   rows, and its own prefill writes `P..prompt_len` as prefill rows.
//! - The re-rendered previous answer sits inside `P..prompt_len`, so those positions are
//!   REWRITTEN by prefill; the `decode_step` rows at them are gone.
//! - Therefore rows `0..prompt_len` are prefill rows whenever point 1 is taken.
//!
//! Why the after-answer snapshot was DROPPED (M2b, robin 2026-09-10, #36):
//!
//! - It was taken after the answer, and `decode_step` wrote every row of that answer.
//! - Those rows are never bit equal to the prefill rows at the same positions (#31 A9).
//! - So it was never offered to `reuse_slot`: taken on every request, consumed on none.
//! - Measured price of holding it (#31 A9, spec 7.7):
//!
//! | quantity | value |
//! |---|---|
//! | pageable host RAM per process | 130,646,016 B = 124.60 MiB |
//! | DtoH per request, warm | 14.5 ms |
//! | DtoH on the first request of a process | 35 to 64 ms |
//!
//! - Unchanged by the drop: the previous answer is re-prefilled, 63 tokens at the A9
//!   operating point (pos 16,127 minus the prefill clean `P` 16,064,
//!   `decode_out/srv-a9.log:42-43`).
//! - Those 63 sit inside a 95 token warm prefill of 404.3 ms at 234.95 tok/s
//!   (`decode_out/srv-a9.log:29`). The answer's own share of those ms is NOT measured.
//!
//! The one exception to "in-process state, never a file" (#32 A10, spec 7.6):
//!
//! - `engine/src/slot.rs` writes the `SLOT_PROMPT` slot to a file and reads it back.
//! - It is the deliberate exception, so the FILE carries the whole load shape.
//! - A restore refuses any shape mismatch instead of loading state that is only shaped right.
//! - Only `SLOT_PROMPT` is ever written: it is the one slot, and it is prefill clean.
//! - A restore names it through `set_prompt_slot`, so the next request lands on the
//!   ordinary warm path of this file, not on a second one.
//!
//! Off switch:
//!
//! | variable | effect |
//! |---|---|
//! | `CROW_PREFIX_CACHE=0` | no slots are allocated, every request is a cold start |
//! | | `/slots/0` then refuses save and restore: there is no slot to write or fill |
//! | `CROW_PREFIX_PARK_ROWS=0` | #118: no park; a cold start drops every snapshot (the pre-#118 rule) |

use crate::cuda;
use crate::gen::Engine;
use crate::geo::{GD, GDN_CONV, GDN_VHEADS, V};
use cudarc::driver::sys;

/// slot of the snapshot taken after the prompt prefill (spec 7.6, point 1)
pub const SLOT_PROMPT: usize = 0;
/// prompt snapshots held per conversation, newest in slot 0, older behind it. Keeping
/// the last few turn prompts lets a history edit that diverges below the newest
/// position roll back to an older prefill clean point (partial reuse) instead of
/// paying the whole prefill the way a single snapshot does.
pub const SLOTS: usize = 3;

/// - #118: the most KV rows a PARKED conversation keeps in host RAM (`CROW_PREFIX_PARK_ROWS`)
/// - 8192 rows cost 113,252,352 B = 108.0 MiB at fp8 KV (12,288 B KV per row plus 2,049
///   pooled QSA blocks of 6,144 B), 204.0 MiB at bf16 (`park_host_bytes`); the 37 side
///   requests of 2026-09-25 wrote at most 4,947 rows (engine.log, the judge of Crow #266)
pub const PARK_ROWS_DEFAULT: usize = 8192;

/// - #118: `CROW_PREFIX_PARK_ROWS`, read once per process: the row cap of a park
/// - `0` turns parking off (every cold start drops every snapshot, the pre-#118 rule);
///   unset or not a number gives `PARK_ROWS_DEFAULT`
pub fn park_rows_cap() -> usize {
    std::env::var("CROW_PREFIX_PARK_ROWS")
        .ok()
        .and_then(|v| v.trim().parse().ok())
        .unwrap_or(PARK_ROWS_DEFAULT)
}

/// - #118: rows `0..park_rows(top, cap)` of KV and pooled QSA are copied to host RAM when
///   a conversation whose newest snapshot sits at `top` is parked
/// - never more than the snapshot needs (`top`), never more than the cap
pub fn park_rows(top: usize, cap: usize) -> usize {
    top.min(cap)
}

/// - #118: pooled QSA blocks a park of `rows` rows copies: `rows / 4 + 1`, so the block that
///   HOLDS row `rows` is in it too. The decode graph writes block `done_blocks` on every
///   step (`gen.rs`, "garbage lands beyond done_blocks"), which is `dirty / 4` for a
///   request that ended at `dirty`; with `dirty <= rows` that block is `<= rows / 4`
/// - capped at the `ceil(n_ctx / 4)` blocks `qsa_pooled` holds (`manager.rs`)
pub fn park_blocks(rows: usize, n_ctx: usize) -> usize {
    (rows / 4 + 1).min(n_ctx.div_ceil(4))
}

/// - #118: host RAM of a park of `rows` rows: the KV rows of every attention layer (K and
///   V, `NKV` heads of `AHD` values at `kv_value_bytes`) plus `park_blocks` pooled QSA
///   blocks of `QSA_HIDD` f32 per attention layer
pub fn park_host_bytes(rows: usize, n_ctx: usize, attn_layers: usize, kv_value_bytes: usize) -> usize {
    rows * attn_layers * 2 * crate::geo::NKV * crate::geo::AHD * kv_value_bytes
        + attn_layers * park_blocks(rows, n_ctx) * crate::geo::QSA_HIDD * 4
}

/// - #118: may a parked snapshot at `p` be rolled back onto after other requests wrote
///   KV rows `0..dirty`? Its state needs rows `0..p`; rows `0..min(dirty, p)` were
///   overwritten, and the park put back rows `0..rows`
/// - true iff `min(dirty, p) <= rows`
pub fn parked_usable(p: usize, dirty: usize, rows: usize) -> bool {
    p.min(dirty) <= rows
}

/// #118: what a cold start does with the held snapshots (`PrefixCache::plan_cold`)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ColdPlan {
    /// the held conversation is parked: its snapshots stay, rows `0..rows` go to host RAM
    Park { rows: usize },
    /// a conversation parked earlier stays parked; the held one (a side request) is dropped
    Keep,
    /// every snapshot is dropped (the pre-#118 cold start)
    Drop,
}

/// f32 slots of one GDN layer's recurrent state S, `[48][128][128]` (`manager.rs:60`)
const GDN_S_STATE: usize = GDN_VHEADS * GD * GD;
/// f32 slots of one GDN layer's causal conv state, `[10240][3]` (`manager.rs:61`)
const GDN_CONV_STATE: usize = GDN_CONV * 3;
/// f32 slots of the PLE dilated conv state, `[10240][9]` (`gen.rs:204`)
const PLE_STATE: usize = GDN_CONV * 9;

// ------------------------------------------------------------------ pure rules

/// - length of the longest common prefix of two id lists (spec 7.4)
/// - ids only, never text, never a hash of the rendered prompt
pub fn common_prefix_len(a: &[i64], b: &[i64]) -> usize {
    a.iter().zip(b.iter()).take_while(|(x, y)| x == y).count()
}

/// - #114: `common_prefix_len`, image aware: the prompt ids of an image are
///   `IMAGE_PAD` repeated per visual token, so two DIFFERENT images of the same grid
///   are the same id run. The prefix therefore ends at the START of the first image
///   whose span (`start`, `len`, content `hash`) is not held identically on both sides.
/// - `a_imgs` / `b_imgs` are the image spans of `a` / `b` (`vit::ImageSpan`); a span on
///   one side with no equal span at the same start on the other side is a difference,
///   whichever side it is on (a slot file restores ids without spans, so an image the
///   held side cannot name never matches)
/// - no spans on either side: exactly `common_prefix_len` (text-only requests, and the
///   `CROW_VIT=0` placeholder whose lone `IMAGE_PAD` is an ordinary token)
/// - llama.cpp `server_tokens::get_common_prefix` (tools/server/server-common.cpp):
///   media chunks are equal only if chunk id (image hash) AND token count are equal,
///   else the prefix is the chunk's first index
pub fn common_prefix_len_mm(
    a: &[i64],
    a_imgs: &[crate::vit::ImageSpan],
    b: &[i64],
    b_imgs: &[crate::vit::ImageSpan],
) -> usize {
    let l = common_prefix_len(a, b);
    [(a_imgs, b_imgs), (b_imgs, a_imgs)]
        .iter()
        .flat_map(|(mine, theirs)| mine.iter().filter(move |sp| !theirs.contains(sp)))
        .map(|sp| sp.start)
        .filter(|&s| s < l)
        .fold(l, usize::min)
}

/// - `P = max { S_pos in positions : S_pos <= l and S_pos < new_len }` (spec 7.4, 7.6)
/// - `None` = no snapshot at or below `l`, so the request is a cold start
/// - returns `(slot index, S_pos)`; a tie on `S_pos` takes the lower index
pub fn reuse_slot(positions: &[Option<usize>], l: usize, new_len: usize) -> Option<(usize, usize)> {
    reuse_slot_with_logits(positions, &[], l, new_len)
}

/// - #100: `reuse_slot`, except that a slot whose `logits[i]` is true may also sit AT
///   `new_len`: its stored logits row stands in for the prefill of the last position
/// - `logits` shorter than `positions` counts as false for the missing slots
pub fn reuse_slot_with_logits(
    positions: &[Option<usize>],
    logits: &[bool],
    l: usize,
    new_len: usize,
) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    for (i, p) in positions.iter().enumerate() {
        let Some(p) = *p else { continue };
        let exact = logits.get(i).copied().unwrap_or(false);
        if p > l || p > new_len || (p == new_len && !exact) {
            continue;
        }
        match best {
            Some((_, b)) if b >= p => {}
            _ => best = Some((i, p)),
        }
    }
    best
}

/// what the detection rule decided for ONE request
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Decision {
    /// longest common id prefix with the held conversation
    pub l: usize,
    /// `Some((slot, P))` = roll back to that slot; `None` = cold start
    pub reuse: Option<(usize, usize)>,
    /// #118: `reuse` names a slot of the PARKED conversation, and `l` is the prefix
    /// against its history; `PrefixCache::unpark` runs before the rollback
    pub parked: bool,
}

impl Decision {
    /// tokens reused from the held state (`cached_tokens` on the wire)
    pub fn cached_n(&self) -> usize {
        self.reuse.map(|(_, p)| p).unwrap_or(0)
    }
}

// ------------------------------------------------------- the host side buffers

/// the buffer shape of ONE engine load (spec 7.6: never valid for another load)
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Shape {
    /// GDN layers, `ThreeStates::gdn_s.len()`
    pub gdn_layers: usize,
    /// attention layers, `ThreeStates::qsa_keys.len()`
    pub attn_layers: usize,
    /// f32 per QSA layer, `qsa_ring_rows * 128`
    pub qsa_ring_len: usize,
}

impl Shape {
    /// read off the loaded states, never assumed
    pub fn of(eng: &Engine) -> Shape {
        Shape {
            gdn_layers: eng.st.gdn_s.len(),
            attn_layers: eng.st.qsa_keys.len(),
            qsa_ring_len: eng.st.qsa_ring_rows * crate::geo::QSA_HIDD,
        }
    }

    /// bytes one snapshot costs in host RAM (spec 7.7)
    pub fn snapshot_bytes(&self) -> usize {
        4 * (self.gdn_layers * (GDN_S_STATE + GDN_CONV_STATE)
            + PLE_STATE
            + self.attn_layers * self.qsa_ring_len)
    }
}

/// one held position: the four recurrent buffers plus the host triple that names it
struct Snapshot {
    /// position these buffers belong to; `None` = the slot is empty
    pos: Option<usize>,
    /// every KV and pooled QSA row below `pos` was written by `prefill`, never by
    /// `decode_step` (see the module doc); only such a slot may be rolled back onto
    prefill_clean: bool,
    /// `Engine::done_blocks` as it stood when the buffers were copied
    done_blocks: usize,
    /// `[gdn_layers][48*128*128]` f32
    gdn_s: Vec<Vec<f32>>,
    /// `[gdn_layers][10240*3]` f32
    gdn_conv: Vec<Vec<f32>>,
    /// `[10240*9]` f32
    ple_state: Vec<f32>,
    /// `[attn_layers][ring*128]` f32
    qsa_ring: Vec<Vec<f32>>,
    /// #100: `s.logits` row 0 as the prefill of `pos` left it, `[V]` f32
    logits: Vec<f32>,
    /// #100: the greedy id `prefill` returned for that row; `None` = no row held (an
    /// empty slot, or one filled from a slot file), and then `pos` must stay below the
    /// request length
    greedy: Option<usize>,
    /// #118: the slot belongs to the PARKED conversation, not to `Engine::history`
    parked: bool,
}

/// #118: a conversation set aside while an unrelated request uses the one KV buffer
struct Parked {
    /// its `Engine::history` when it was parked
    history: Vec<i64>,
    /// its `Engine::history_images` when it was parked
    images: Vec<crate::vit::ImageSpan>,
    /// KV rows `0..rows` and pooled blocks `0..blocks` are held in `PrefixCache::park_kv`
    /// and `park_pooled`, byte for byte as they stood when it was parked
    rows: usize,
    blocks: usize,
    /// the highest KV row the requests since the park have written (their end `pos`)
    dirty: usize,
}

/// - `vec![0f32; n]`, with every page of it faulted in before it is returned
/// - TASK I: `vec![0f32; n]` is a `calloc`, so its pages are the shared zero page
///   until something WRITES them. The first `snapshot` into a fresh slot therefore
///   paid one minor fault per 4 KiB inside the request: measured on the 130 MB slot
///   of a 262k-context load, 45.5 / 43.7 / 38.4 ms for the first three snapshots of
///   a process against 8.8 ms once the pages are there. The slots are allocated at
///   boot, where no request is waiting, so the faults belong here.
/// - the store is `write_volatile` because the value written is the value already
///   there, and nothing else may read it: a plain store would be dead code
fn faulted(n: usize) -> Vec<f32> {
    let mut v = vec![0f32; n];
    // one store per 4 KiB page = one f32 every 1024
    for i in (0..n).step_by(1024) {
        unsafe { std::ptr::write_volatile(v.as_mut_ptr().add(i), 0f32) };
    }
    v
}

impl Snapshot {
    /// allocate once; every later snapshot writes into these buffers
    fn new(shape: &Shape, vocab: usize) -> Snapshot {
        Snapshot {
            pos: None,
            prefill_clean: false,
            done_blocks: 0,
            gdn_s: (0..shape.gdn_layers).map(|_| faulted(GDN_S_STATE)).collect(),
            gdn_conv: (0..shape.gdn_layers).map(|_| faulted(GDN_CONV_STATE)).collect(),
            ple_state: faulted(PLE_STATE),
            qsa_ring: (0..shape.attn_layers)
                .map(|_| faulted(shape.qsa_ring_len))
                .collect(),
            logits: faulted(vocab),
            greedy: None,
            parked: false,
        }
    }
}

/// - device to host copy of `dst.len()` f32
/// - `cuMemcpyDtoH_v2` blocks the host, so `dst` is complete when this returns
/// - the caller syncs first, so nothing in flight still writes `src`
///
/// # Safety
///
/// - a CUDA context must be current and `src` must hold at least `dst.len()` f32
unsafe fn dtoh_into(dst: &mut [f32], src: cuda::CUdeviceptr) {
    cuda::ck(sys::cuMemcpyDtoH_v2(
        dst.as_mut_ptr() as *mut std::ffi::c_void,
        src,
        dst.len() * 4,
    ));
}

// ----------------------------------------------------------------- the cache

/// the ONE held conversation of this process (M1) with its `SLOTS` prompt snapshots, plus
/// at most ONE parked conversation (#118)
pub struct PrefixCache {
    /// `CROW_PREFIX_CACHE=0` turns every request into a cold start
    enabled: bool,
    shape: Shape,
    slots: Vec<Snapshot>,
    /// #118: `CROW_PREFIX_PARK_ROWS`, 0 = never park
    park_cap: usize,
    /// #118: the parked conversation, if any; its snapshots are the slots with `parked`
    park: Option<Parked>,
    /// #118: KV rows `0..rows` of the parked conversation, in `slot::kv_row_order`,
    /// allocated at the first park and reused (pageable host RAM, like the slots)
    park_kv: Vec<u8>,
    /// #118: pooled QSA blocks `0..blocks` of the parked conversation, per attention layer
    park_pooled: Vec<u8>,
}

impl PrefixCache {
    /// - allocates the one slot up front when the cache is on (spec 7.7)
    /// - reads the shape off the loaded states, so it can never outlive its load
    pub fn new(eng: &Engine) -> PrefixCache {
        let enabled = std::env::var("CROW_PREFIX_CACHE").as_deref() != Ok("0");
        let shape = Shape::of(eng);
        let slots = if enabled {
            (0..SLOTS).map(|_| Snapshot::new(&shape, V)).collect()
        } else {
            Vec::new()
        };
        PrefixCache {
            enabled,
            shape,
            slots,
            park_cap: park_rows_cap(),
            park: None,
            park_kv: Vec::new(),
            park_pooled: Vec::new(),
        }
    }

    /// #118: the row cap of a park in force for this process (0 = parking off)
    pub fn park_cap(&self) -> usize {
        self.park_cap
    }

    /// on or off for this process
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// the shape the slots were allocated for
    pub fn shape(&self) -> Shape {
        self.shape
    }

    /// the position each slot holds, `None` for an empty slot
    pub fn positions(&self) -> Vec<Option<usize>> {
        self.slots.iter().map(|s| s.pos).collect()
    }

    /// - the position of each slot that may be ROLLED BACK ONTO, `None` for the others
    /// - a slot that is not prefill clean is hidden here, never in `positions`
    /// - #118: so is a slot of the parked conversation (`parked_positions` names those)
    pub fn reuse_candidates(&self) -> Vec<Option<usize>> {
        self.slots
            .iter()
            .map(|s| if s.prefill_clean && !s.parked { s.pos } else { None })
            .collect()
    }

    /// #118: the position of each slot of the PARKED conversation, `None` for the others
    pub fn parked_positions(&self) -> Vec<Option<usize>> {
        self.slots.iter().map(|s| if s.parked { s.pos } else { None }).collect()
    }

    /// #118: `(history length, rows held, rows written since)` of the parked conversation
    pub fn parked(&self) -> Option<(usize, usize, usize)> {
        self.park.as_ref().map(|pk| (pk.history.len(), pk.rows, pk.dirty))
    }

    /// #118: the newest (largest) position among the held, prefill clean LIVE slots
    fn live_top(&self) -> Option<usize> {
        self.reuse_candidates().into_iter().flatten().max()
    }

    /// - #118: the parked slots that may still be rolled back onto once the requests since
    ///   the park have written KV rows `0..dirty` (`parked_usable`)
    /// - `None` for every slot while nothing is parked
    fn parked_candidates(&self, dirty: usize) -> Vec<Option<usize>> {
        let Some(pk) = self.park.as_ref() else { return vec![None; self.slots.len()] };
        self.slots
            .iter()
            .map(|s| match s.pos {
                Some(p) if s.parked && s.prefill_clean && parked_usable(p, dirty, pk.rows) => Some(p),
                _ => None,
            })
            .collect()
    }

    /// - the whole detection rule of spec 7.4 for ONE request
    /// - `history` is `Engine::history`, `ids` the rendered request
    pub fn decide(&self, history: &[i64], ids: &[i64]) -> Decision {
        self.decide_for(history, ids, true)
    }

    /// - `decide`, with the #100 zero-prefill reuse allowed or not
    /// - `allow_exact == false` keeps the old guard `S_pos < request length` for every
    ///   slot: a request that needs its prompt's own prefill (the `CROW_VIT_DUMP` logits
    ///   collection) must get at least one token prefilled
    pub fn decide_for(&self, history: &[i64], ids: &[i64], allow_exact: bool) -> Decision {
        self.decide_mm(history, &[], ids, &[], allow_exact)
    }

    /// - #114: `decide_for` with the image spans of both sides: `L` is
    ///   `common_prefix_len_mm`, so a request whose image differs from the held one
    ///   at the same place gets no snapshot at or past that image's first row
    /// - `held_imgs` is `Engine::history_images`, `imgs` the request plan's spans
    pub fn decide_mm(
        &self,
        history: &[i64],
        held_imgs: &[crate::vit::ImageSpan],
        ids: &[i64],
        imgs: &[crate::vit::ImageSpan],
        allow_exact: bool,
    ) -> Decision {
        if !self.enabled {
            return Decision { l: 0, reuse: None, parked: false };
        }
        let l = common_prefix_len_mm(history, held_imgs, ids, imgs);
        let logits = if allow_exact { self.logits_held() } else { Vec::new() };
        let live = Decision {
            l,
            reuse: reuse_slot_with_logits(&self.reuse_candidates(), &logits, l, ids.len()),
            parked: false,
        };
        // #118: the same rule against the PARKED conversation; `history.len()` is where the
        // requests since the park have written up to. The one that reuses more wins.
        let Some(pk) = self.park.as_ref() else { return live };
        let lp = common_prefix_len_mm(&pk.history, &pk.images, ids, imgs);
        let cands = self.parked_candidates(pk.dirty.max(history.len()));
        match reuse_slot_with_logits(&cands, &logits, lp, ids.len()) {
            Some((slot, p)) if p > live.cached_n() => Decision { l: lp, reuse: Some((slot, p)), parked: true },
            _ => live,
        }
    }

    /// #100: per slot, whether it holds a logits row it may stand in with
    pub fn logits_held(&self) -> Vec<bool> {
        self.slots.iter().map(|s| s.prefill_clean && s.greedy.is_some()).collect()
    }

    /// - `Some((pos, done_blocks))` of the PROMPT slot while it is a reuse candidate
    /// - `None` when it is empty; that slot is never anything but prefill clean
    /// - #32 A10: this is the position a slot file holds, and `n_saved` on the wire
    pub fn prompt_slot(&self) -> Option<(usize, usize)> {
        let s = self.slots.get(SLOT_PROMPT)?;
        // #118: a parked slot names another conversation than `Engine::history`
        if !s.prefill_clean || s.parked {
            return None;
        }
        s.pos.map(|p| (p, s.done_blocks))
    }

    /// - the four recurrent buffers of the PROMPT slot, in the ONE order a slot file uses
    /// - empty when no slot is allocated (`CROW_PREFIX_CACHE=0`)
    /// - #32 A10: `engine/src/slot.rs` is the only reader, and it writes them in this order
    pub fn prompt_state_blocks(&self) -> Vec<&[f32]> {
        let Some(s) = self.slots.get(SLOT_PROMPT) else { return Vec::new() };
        let mut v: Vec<&[f32]> = Vec::with_capacity(2 * s.gdn_s.len() + s.qsa_ring.len() + 1);
        v.extend(s.gdn_s.iter().map(|b| b.as_slice()));
        v.extend(s.gdn_conv.iter().map(|b| b.as_slice()));
        v.push(s.ple_state.as_slice());
        v.extend(s.qsa_ring.iter().map(|b| b.as_slice()));
        v
    }

    /// the same blocks, in the same order, to be filled from a slot file
    pub fn prompt_state_blocks_mut(&mut self) -> Vec<&mut [f32]> {
        let Some(s) = self.slots.get_mut(SLOT_PROMPT) else { return Vec::new() };
        let Snapshot { gdn_s, gdn_conv, ple_state, qsa_ring, .. } = s;
        let mut v: Vec<&mut [f32]> = Vec::with_capacity(2 * gdn_s.len() + qsa_ring.len() + 1);
        v.extend(gdn_s.iter_mut().map(|b| b.as_mut_slice()));
        v.extend(gdn_conv.iter_mut().map(|b| b.as_mut_slice()));
        v.push(ple_state.as_mut_slice());
        v.extend(qsa_ring.iter_mut().map(|b| b.as_mut_slice()));
        v
    }

    /// - name the PROMPT slot after its buffers were filled from a slot file (#32 A10)
    /// - prefill clean by construction: a slot file only ever holds a prefill clean position
    /// - it is the ONLY slot (M2b, #36): no second slot can claim an unwritten position
    pub fn set_prompt_slot(&mut self, pos: usize, done_blocks: usize) {
        if let Some(s) = self.slots.get_mut(SLOT_PROMPT) {
            s.pos = Some(pos);
            s.prefill_clean = true;
            s.done_blocks = done_blocks;
            // #100: a slot file carries no logits row
            s.greedy = None;
        }
    }

    /// - every slot forgets its position, and the parked conversation is dropped (#118);
    ///   the buffers stay allocated
    /// - the cold start of `CROW_PREFIX_PARK_ROWS=0`, a slot file restore and a dropped
    ///   request call it, so no slot can name a discarded history
    pub fn invalidate(&mut self) {
        for s in self.slots.iter_mut() {
            s.pos = None;
            s.prefill_clean = false;
            s.greedy = None;
            s.parked = false;
        }
        self.park = None;
    }

    /// #118: the live conversation has written KV rows `0..written`; a parked one counts them
    fn note_written(&mut self, written: usize) {
        if let Some(pk) = self.park.as_mut() {
            pk.dirty = pk.dirty.max(written);
        }
    }

    /// #118: forget the LIVE slots only; the parked conversation keeps its snapshots
    fn drop_live(&mut self) {
        for s in self.slots.iter_mut().filter(|s| !s.parked) {
            s.pos = None;
            s.prefill_clean = false;
            s.greedy = None;
        }
    }

    /// - #118: what a COLD start does with the held snapshots, host side only
    /// - `live_len` is `Engine::history().len()` (the rows the held conversation wrote),
    ///   `new_len` the prompt of the cold request (it writes at least rows `0..new_len`)
    /// - the held conversation is parked when its newest snapshot survives the new request
    ///   (`parked_usable` at `dirty = new_len`) and it is at least as large as what is
    ///   parked already (the one that saves more prefill is kept)
    /// - else a conversation parked earlier stays, if the new request leaves it usable
    /// - else everything is dropped, the pre-#118 cold start
    pub fn plan_cold(&self, live_len: usize, new_len: usize) -> ColdPlan {
        if !self.enabled || self.park_cap == 0 {
            return ColdPlan::Drop;
        }
        let live = self
            .live_top()
            .filter(|&t| parked_usable(t, new_len, park_rows(t, self.park_cap)));
        let kept = self.park.as_ref().and_then(|pk| {
            let dirty = pk.dirty.max(live_len).max(new_len);
            self.parked_candidates(dirty).into_iter().flatten().max()
        });
        match (live, kept) {
            (Some(t), k) if k.is_none_or(|k| t >= k) => ColdPlan::Park { rows: park_rows(t, self.park_cap) },
            (_, Some(_)) => ColdPlan::Keep,
            _ => ColdPlan::Drop,
        }
    }

    /// - #118: the host bookkeeping of a cold start after `plan_cold`; `history` and
    ///   `images` are the held conversation's, and `Park` takes them
    /// - `Park`: an older parked conversation is dropped, every prefill clean live slot
    ///   becomes a parked slot, `dirty` starts at 0 (nothing has overwritten a row yet)
    /// - `Keep`: the live slots are dropped, and the rows the held conversation wrote are
    ///   added to `dirty`
    /// - the stash bytes themselves are copied by `cold_start`, not here
    fn apply_cold(
        &mut self,
        plan: ColdPlan,
        history: Vec<i64>,
        images: Vec<crate::vit::ImageSpan>,
        blocks: usize,
    ) {
        match plan {
            ColdPlan::Drop => self.invalidate(),
            ColdPlan::Keep => {
                let written = history.len();
                self.drop_live();
                if let Some(pk) = self.park.as_mut() {
                    pk.dirty = pk.dirty.max(written);
                }
            }
            ColdPlan::Park { rows } => {
                for s in self.slots.iter_mut() {
                    if s.parked || !s.prefill_clean {
                        s.pos = None;
                        s.prefill_clean = false;
                        s.greedy = None;
                        s.parked = false;
                    } else if s.pos.is_some() {
                        s.parked = true;
                    }
                }
                self.park = Some(Parked { history, images, rows, blocks, dirty: 0 });
            }
        }
    }

    /// - #118: the COLD start of a request of `new_len` prompt ids: park the held
    ///   conversation (or keep the one parked earlier, or drop all, `plan_cold`), then
    ///   `Engine::reset_to_zero`
    /// - a park copies KV rows `0..rows` and pooled blocks `0..blocks` device to host,
    ///   BEFORE the reset and the new prefill write them
    /// - returns the plan and the wall of the copy in ms (0.0 without a park)
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn cold_start(&mut self, eng: &mut Engine, new_len: usize) -> (ColdPlan, f64) {
        let plan = self.plan_cold(eng.history.len(), new_len);
        let mut ms = 0.0;
        let mut blocks = 0;
        if let ColdPlan::Park { rows } = plan {
            let t0 = std::time::Instant::now();
            blocks = park_blocks(rows, eng.st.context);
            // whatever the last request left in flight must land before the copies read it
            cuda::sync();
            let row_bytes = crate::geo::AHD * eng.st.kv.byte_per_value();
            let n = rows * row_bytes;
            let groups: Vec<_> = crate::slot::kv_row_order(eng.st.qsa_pooled.len()).collect();
            self.park_kv.resize(groups.len() * n, 0);
            for (g, (layer, is_k, kvh)) in groups.into_iter().enumerate() {
                crate::slot::dtoh_bytes(&mut self.park_kv[g * n..(g + 1) * n], eng.st.kv_row_ptr(layer, is_k, kvh, 0));
            }
            let pb = blocks * crate::geo::QSA_HIDD * 4;
            self.park_pooled.resize(eng.st.qsa_pooled.len() * pb, 0);
            for (layer, &src) in eng.st.qsa_pooled.iter().enumerate() {
                crate::slot::dtoh_bytes(&mut self.park_pooled[layer * pb..(layer + 1) * pb], src);
            }
            ms = t0.elapsed().as_secs_f64() * 1e3;
        }
        // `reset_to_zero` clears both; `Park` keeps them for the unpark
        let history = std::mem::take(&mut eng.history);
        let images = std::mem::take(&mut eng.history_images);
        self.apply_cold(plan, history, images, blocks);
        eng.reset_to_zero();
        (plan, ms)
    }

    /// - #118: host bookkeeping of an unpark: the live slots are dropped, the parked ones
    ///   become the live ones, and the parked history is handed back to be put into
    ///   `Engine::history`
    fn adopt_parked(&mut self) -> Option<Parked> {
        let pk = self.park.take()?;
        self.drop_live();
        for s in self.slots.iter_mut() {
            s.parked = false;
        }
        Some(pk)
    }

    /// - #118: bring the parked conversation back: KV rows `0..rows` and pooled blocks
    ///   `0..blocks` host to device, `Engine::history` and its image spans put back, its
    ///   snapshots made the live ones. `rollback` onto the slot `decide` picked follows
    /// - the decode graph and the capture stream are dropped FIRST (A4), as in `rollback`
    /// - returns the wall in ms, 0.0 when nothing is parked
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn unpark(&mut self, eng: &mut Engine) -> f64 {
        let t0 = std::time::Instant::now();
        let Some(pk) = self.adopt_parked() else { return 0.0 };
        cuda::sync();
        eng.drop_decode_graph();
        let row_bytes = crate::geo::AHD * eng.st.kv.byte_per_value();
        let n = pk.rows * row_bytes;
        for (g, (layer, is_k, kvh)) in crate::slot::kv_row_order(eng.st.qsa_pooled.len()).enumerate() {
            cuda::upload_into(eng.st.kv_row_ptr(layer, is_k, kvh, 0), &self.park_kv[g * n..(g + 1) * n]);
        }
        let pb = pk.blocks * crate::geo::QSA_HIDD * 4;
        for (layer, &dst) in eng.st.qsa_pooled.iter().enumerate() {
            cuda::upload_into(dst, &self.park_pooled[layer * pb..(layer + 1) * pb]);
        }
        cuda::sync();
        eng.history = pk.history;
        eng.history_images = pk.images;
        t0.elapsed().as_secs_f64() * 1e3
    }

    /// - copy the four recurrent buffers device to host into `SLOT_PROMPT` (slot 0)
    /// - `prefill_clean` says whether every row below `Engine::pos` is a prefill row;
    ///   `false` keeps the slot out of `reuse_candidates` (see the module doc)
    /// - BEFORE the copy `claim` picks the slot and moves it to slot 0 (#100): the slot that
    ///   already holds `Engine::pos`, else an empty one, else the smallest position. It is
    ///   overwritten in place, so no buffer is reallocated. A history edit that diverges
    ///   below the newest position then still finds an earlier turn's snapshot further down.
    /// - `greedy` is what `prefill` returned; with it the logits row `s.logits` row 0 is
    ///   copied too (#100), so a later identical request needs no prefill at all
    /// - returns the wall of the copy in ms (spec 7.9 asks for it)
    /// - a disabled cache does nothing and returns 0.0
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn snapshot(&mut self, eng: &Engine, prefill_clean: bool, greedy: usize) -> f64 {
        if !self.enabled || self.slots.is_empty() {
            return 0.0;
        }
        let t0 = std::time::Instant::now();
        // whatever the last launch left in flight must land before the copy reads it
        cuda::sync();
        let slot = self.claim(eng.pos);
        let s = &mut self.slots[slot];
        for (i, buf) in s.gdn_s.iter_mut().enumerate() {
            dtoh_into(buf, eng.st.gdn_s[i]);
        }
        for (i, buf) in s.gdn_conv.iter_mut().enumerate() {
            dtoh_into(buf, eng.st.gdn_conv[i]);
        }
        dtoh_into(&mut s.ple_state, eng.ple.state);
        for (i, buf) in s.qsa_ring.iter_mut().enumerate() {
            dtoh_into(buf, eng.st.qsa_keys[i]);
        }
        dtoh_into(&mut s.logits, eng.logits());
        self.name(slot, eng.pos, prefill_clean, eng.done_blocks, Some(greedy));
        t0.elapsed().as_secs_f64() * 1e3
    }

    /// - host bookkeeping of a snapshot at `pos`: which slot the copy goes into (#100)
    /// - the slot already holding `pos` (no duplicate), else an empty slot, else the one
    ///   with the smallest position; it is moved to `SLOT_PROMPT`, the others keep their order
    /// - with the #101 invariant every held slot is a prefix of `history`, so a slot at
    ///   `pos` holds the same ids over prefill rows, and the smallest is the oldest turn
    /// - #118: a parked slot is never `same` (it names another history). It is taken only
    ///   when no empty and no live slot is left, the smallest parked one, and never the
    ///   parked conversation's LAST slot while another exists; taking the last one drops
    ///   the park
    fn claim(&mut self, pos: usize) -> usize {
        let n = self.slots.len();
        let same = self.slots.iter().position(|s| !s.parked && s.pos == Some(pos));
        let empty = || self.slots.iter().position(|s| s.pos.is_none());
        let smallest_of = |parked: bool| {
            (0..n)
                .filter(|&i| self.slots[i].parked == parked && self.slots[i].pos.is_some())
                .min_by_key(|&i| self.slots[i].pos.unwrap_or(0))
        };
        let parked_n = self.slots.iter().filter(|s| s.parked).count();
        let spare_parked = || if parked_n > 1 { smallest_of(true) } else { None };
        let i = same
            .or_else(empty)
            .or_else(|| smallest_of(false))
            .or_else(spare_parked)
            .or_else(|| smallest_of(true))
            .unwrap_or(SLOT_PROMPT);
        self.slots[i].parked = false;
        if !self.slots.iter().any(|s| s.parked) {
            self.park = None;
        }
        self.slots[..=i].rotate_right(1);
        SLOT_PROMPT
    }

    /// host bookkeeping of a snapshot: name the slot the copy went into
    fn name(&mut self, slot: usize, pos: usize, prefill_clean: bool, done_blocks: usize, greedy: Option<usize>) {
        let s = &mut self.slots[slot];
        s.pos = Some(pos);
        s.prefill_clean = prefill_clean;
        s.done_blocks = done_blocks;
        s.greedy = greedy;
    }

    /// - host bookkeeping of a rollback onto `slot` (#101)
    /// - every slot ABOVE its position is forgotten: the prefill and the decode that
    ///   follow rewrite the KV rows from there, so those slots name a discarded branch
    /// - #118: `written` is `Engine::history().len()` BEFORE the rollback: the held
    ///   conversation wrote KV rows `0..written`, and a parked one counts them
    ///   (`note_written`) also when this rollback cuts `history` shorter than that
    fn rolled_back(&mut self, slot: usize, written: usize) {
        self.note_written(written);
        let Some(p) = self.slots[slot].pos else { return };
        // #118: a parked slot names another history, which this rollback does not touch
        for s in self.slots.iter_mut().filter(|s| !s.parked) {
            if s.pos.is_some_and(|q| q > p) {
                s.pos = None;
                s.prefill_clean = false;
                s.greedy = None;
            }
        }
    }

    /// - #100: the zero-prefill path. Upload the logits row `slot` holds into `s.logits`
    ///   row 0 and return its greedy id: the state `prefill` of the last prompt position
    ///   would have left for the first draw
    /// - call right after `rollback(eng, slot)`, instead of `prefill`
    /// - panics on a slot without a row: `decide` only picks one at the request length
    ///   when `logits_held` said it has one
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    pub unsafe fn restore_logits(&self, eng: &Engine, slot: usize) -> usize {
        let s = &self.slots[slot];
        let greedy = s.greedy.expect("zero-prefill reuse of a slot without a logits row");
        cuda::to_f32_into(eng.logits(), &s.logits);
        cuda::sync();
        greedy
    }

    /// - roll the engine back to the position `slot` holds (spec 7.6)
    /// - the decode graph and the capture stream are dropped FIRST (see the module doc)
    /// - returns the wall of the whole rollback in ms
    /// - panics on an empty slot: `decide` is what picks the slot, and it only picks a full one
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn rollback(&mut self, eng: &mut Engine, slot: usize) -> f64 {
        let t0 = std::time::Instant::now();
        let s = &self.slots[slot];
        let pos = s.pos.expect("rollback into an empty snapshot slot");
        assert!(s.prefill_clean, "rollback onto a snapshot that is not prefill clean");

        // whatever the last request left in flight must land before anything below
        cuda::sync();

        // spec 7.2 remedy, measured in A4 and defined once in `reset.rs`: the resumed
        // prefill uploads its per chunk scalars from temporaries, and those uploads only
        // sync on the legacy stream. Must stay BEFORE the uploads and the prefill.
        eng.drop_decode_graph();

        for (i, buf) in s.gdn_s.iter().enumerate() {
            cuda::to_f32_into(eng.st.gdn_s[i], buf);
        }
        for (i, buf) in s.gdn_conv.iter().enumerate() {
            cuda::to_f32_into(eng.st.gdn_conv[i], buf);
        }
        cuda::to_f32_into(eng.ple.state, &s.ple_state);
        for (i, buf) in s.qsa_ring.iter().enumerate() {
            cuda::to_f32_into(eng.st.qsa_keys[i], buf);
        }

        eng.pos = pos;
        eng.done_blocks = s.done_blocks;
        // #118: the rows the held conversation wrote, before `history` is cut to `pos`
        let written = eng.history.len();
        eng.history.truncate(pos);
        eng.truncate_history_images(pos);
        eng.route_log.clear();
        self.rolled_back(slot, written);

        // the uploads read the slot's vectors; sync before the caller may touch them
        cuda::sync();
        t0.elapsed().as_secs_f64() * 1e3
    }
}

#[cfg(test)]
impl PrefixCache {
    /// - a cache with host slots only, for the rules that need no CUDA
    /// - `enabled == false` reproduces `CROW_PREFIX_CACHE=0`: no slot is allocated
    fn for_shape(shape: Shape, enabled: bool) -> PrefixCache {
        let slots = if enabled {
            (0..SLOTS).map(|_| Snapshot::new(&shape, 4)).collect()
        } else {
            Vec::new()
        };
        PrefixCache {
            enabled,
            shape,
            slots,
            park_cap: PARK_ROWS_DEFAULT,
            park: None,
            park_kv: Vec::new(),
            park_pooled: Vec::new(),
        }
    }

    /// - name a slot's position and its prefill clean flag without a device copy
    fn set_slot(&mut self, slot: usize, pos: usize, prefill_clean: bool) {
        self.slots[slot].pos = Some(pos);
        self.slots[slot].prefill_clean = prefill_clean;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// the shape of no real load: four f32 per QSA layer, so a slot costs nothing
    fn tiny() -> Shape {
        Shape { gdn_layers: 1, attn_layers: 1, qsa_ring_len: 4 }
    }

    #[test]
    fn common_prefix_of_two_empty_lists_is_zero() {
        assert_eq!(common_prefix_len(&[], &[]), 0);
    }

    #[test]
    fn common_prefix_of_equal_lists_is_the_length() {
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3]), 3);
    }

    #[test]
    fn common_prefix_stops_at_the_first_difference() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4], &[1, 2, 9, 4]), 2);
    }

    #[test]
    fn common_prefix_of_a_shorter_prefix_is_the_shorter_length() {
        assert_eq!(common_prefix_len(&[1, 2, 3, 4, 5], &[1, 2, 3]), 3);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[1, 2, 3, 4, 5]), 3);
    }

    #[test]
    fn common_prefix_is_zero_when_the_first_id_differs() {
        assert_eq!(common_prefix_len(&[7, 2, 3], &[8, 2, 3]), 0);
    }

    #[test]
    fn common_prefix_with_one_empty_list_is_zero() {
        assert_eq!(common_prefix_len(&[], &[1, 2, 3]), 0);
        assert_eq!(common_prefix_len(&[1, 2, 3], &[]), 0);
    }

    #[test]
    fn no_snapshot_at_all_is_a_cold_start() {
        assert_eq!(reuse_slot(&[None, None], 100, 200), None);
    }

    #[test]
    fn every_snapshot_above_l_is_a_cold_start() {
        assert_eq!(reuse_slot(&[Some(120), Some(180)], 100, 200), None);
    }

    #[test]
    fn the_newest_snapshot_at_or_below_l_wins() {
        assert_eq!(reuse_slot(&[Some(40), Some(90)], 100, 200), Some((1, 90)));
        assert_eq!(reuse_slot(&[Some(90), Some(40)], 100, 200), Some((0, 90)));
    }

    #[test]
    fn a_snapshot_exactly_at_l_is_taken() {
        assert_eq!(reuse_slot(&[Some(40), Some(100)], 100, 200), Some((1, 100)));
    }

    #[test]
    fn a_snapshot_above_l_is_skipped_for_the_one_below() {
        // the edited-answer case of spec 7.6: point 2 sits above L, point 1 below it
        assert_eq!(reuse_slot(&[Some(60), Some(150)], 100, 200), Some((0, 60)));
    }

    #[test]
    fn a_tie_on_the_position_takes_the_lower_slot() {
        assert_eq!(reuse_slot(&[Some(80), Some(80)], 100, 200), Some((0, 80)));
    }

    #[test]
    fn a_snapshot_at_the_new_length_is_refused() {
        // `prefill` needs at least one token: it returns the last position's greedy id
        assert_eq!(reuse_slot(&[Some(200)], 250, 200), None);
        assert_eq!(reuse_slot(&[Some(60), Some(200)], 250, 200), Some((0, 60)));
    }

    /// #100: a slot that holds its logits row may sit AT the new length; one without may not
    #[test]
    fn a_snapshot_at_the_new_length_is_taken_when_it_holds_its_logits_row() {
        assert_eq!(reuse_slot_with_logits(&[Some(200)], &[true], 250, 200), Some((0, 200)));
        assert_eq!(reuse_slot_with_logits(&[Some(200)], &[false], 250, 200), None);
        // the row never lets a snapshot ABOVE the request length through
        assert_eq!(reuse_slot_with_logits(&[Some(201)], &[true], 250, 200), None);
        // nor one above L
        assert_eq!(reuse_slot_with_logits(&[Some(200)], &[true], 150, 200), None);
        // the exact slot beats an older one below it
        assert_eq!(reuse_slot_with_logits(&[Some(60), Some(200)], &[true, true], 250, 200), Some((1, 200)));
        // a missing flag counts as no row
        assert_eq!(reuse_slot_with_logits(&[Some(60), Some(200)], &[true], 250, 200), Some((0, 60)));
    }

    #[test]
    fn a_snapshot_one_below_the_new_length_is_taken() {
        assert_eq!(reuse_slot(&[Some(199)], 250, 200), Some((0, 199)));
    }

    #[test]
    fn an_empty_slot_list_is_a_cold_start() {
        assert_eq!(reuse_slot(&[], 100, 200), None);
    }

    /// the normal Crow turn of spec 7.6: the new ids extend the held transcript
    #[test]
    fn a_continued_turn_reuses_the_answer_snapshot() {
        let held: Vec<i64> = (0..100).collect();
        let mut new = held.clone();
        new.extend(200..210);
        let l = common_prefix_len(&held, &new);
        assert_eq!(l, 100);
        assert_eq!(reuse_slot(&[Some(60), Some(100)], l, new.len()), Some((1, 100)));
    }

    /// the edited-answer case of spec 7.6: divergence inside the last answer
    #[test]
    fn an_edited_answer_falls_back_to_the_prompt_snapshot() {
        let held: Vec<i64> = (0..100).collect();
        let mut new: Vec<i64> = (0..70).collect();
        new.extend(900..930);
        let l = common_prefix_len(&held, &new);
        assert_eq!(l, 70);
        assert_eq!(reuse_slot(&[Some(60), Some(100)], l, new.len()), Some((0, 60)));
    }

    /// the edited-prompt case of spec 7.6: below point 1, still warm, never cold
    #[test]
    fn an_edited_prompt_falls_back_below_both_points_of_this_turn() {
        // point 1 of turn 2 at 60, point 2 of turn 2 at 100; the user rewrote the turn 2
        // prompt, so L lands at 45, below both. Nothing at or under 45 is held: cold.
        assert_eq!(reuse_slot(&[Some(60), Some(100)], 45, 200), None);
    }

    #[test]
    fn a_decision_reports_zero_cached_tokens_for_a_cold_start() {
        let d = Decision { l: 12, reuse: None, parked: false };
        assert_eq!(d.cached_n(), 0);
    }

    #[test]
    fn a_decision_reports_p_as_the_cached_tokens() {
        let d = Decision { l: 120, reuse: Some((1, 100)), parked: false };
        assert_eq!(d.cached_n(), 100);
    }

    /// #31 A9, measured 2026-09-10: a slot that is not prefill clean is hidden from the rule
    #[test]
    fn a_slot_that_is_not_prefill_clean_is_no_reuse_candidate() {
        // `reuse_candidates` passes `None` for such a slot, so the rule cannot pick it
        assert_eq!(reuse_slot(&[Some(60), None], 100, 200), Some((0, 60)));
        assert_eq!(reuse_slot(&[None, None], 100, 200), None);
        // and it is the point-2 slot that is hidden: the answer's rows came from decode
        assert_eq!(reuse_slot(&[Some(16_064), None], 16_127, 16_159), Some((0, 16_064)));
    }

    /// the A9 gate turn 2, as the guard makes it: point 1, not point 2
    #[test]
    fn a_continued_turn_reuses_the_prompt_snapshot_when_point_two_is_hidden() {
        let held: Vec<i64> = (0..16_127).collect();
        let mut new: Vec<i64> = (0..16_127).collect();
        new.extend(90_000..90_032);
        let l = common_prefix_len(&held, &new);
        assert_eq!(l, 16_127);
        // without the guard the rule would take 16_127; with it, 16_064
        assert_eq!(reuse_slot(&[Some(16_064), Some(16_127)], l, new.len()), Some((1, 16_127)));
        assert_eq!(reuse_slot(&[Some(16_064), None], l, new.len()), Some((0, 16_064)));
        // still far above the gate's 95 % line
        assert!(16_064.0 >= 0.95 * new.len() as f64);
    }

    /// spec 7.7, the operating point of this process: chunk 2048 gives ring 2052
    #[test]
    fn the_snapshot_size_is_the_spec_table_row() {
        let shape = Shape { gdn_layers: 36, attn_layers: 12, qsa_ring_len: 2052 * 128 };
        assert_eq!(shape.snapshot_bytes(), 130_646_016);
    }

    /// spec 7.7, the `Config` default row of the same table
    #[test]
    fn the_snapshot_size_at_chunk_512_is_the_other_spec_row() {
        let shape = Shape { gdn_layers: 36, attn_layers: 12, qsa_ring_len: 516 * 128 };
        assert_eq!(shape.snapshot_bytes(), 121_208_832);
    }

    /// `CROW_PREFIX_CACHE=0`: `decide` returns before it computes anything (`cache.rs:328`)
    #[test]
    fn a_disabled_cache_decides_cold_for_every_request() {
        let c = PrefixCache::for_shape(tiny(), false);
        assert!(!c.enabled());
        let held: Vec<i64> = (0..100).collect();
        let mut new = held.clone();
        new.extend(200..210);
        let d = c.decide(&held, &new);
        assert_eq!(d, Decision { l: 0, reuse: None, parked: false });
        assert_eq!(d.cached_n(), 0);
        // and no slot exists to be reported
        assert!(c.positions().is_empty());
        assert!(c.reuse_candidates().is_empty());
    }

    /// an enabled cache with its one slot empty is a cold start, and says so in both lists
    #[test]
    fn a_fresh_cache_holds_no_position() {
        let c = PrefixCache::for_shape(tiny(), true);
        assert!(c.enabled());
        assert_eq!(c.positions(), vec![None, None, None]);
        assert_eq!(c.reuse_candidates(), vec![None, None, None]);
        assert_eq!(c.decide(&[1, 2, 3], &[1, 2, 3, 4]).reuse, None);
    }

    /// `positions` reports the slot, `reuse_candidates` hides it when it is not clean
    #[test]
    fn reuse_candidates_hides_the_slot_that_is_not_prefill_clean() {
        let held: Vec<i64> = (0..100).collect();
        let mut new = held.clone();
        new.extend(200..210);
        // prefill clean: reported in both lists, and the decision rolls back onto it
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, true);
        assert_eq!(c.positions(), vec![Some(60), None, None]);
        assert_eq!(c.reuse_candidates(), vec![Some(60), None, None]);
        let d = c.decide(&held, &new);
        assert_eq!(d.l, 100);
        assert_eq!(d.reuse, Some((SLOT_PROMPT, 60)));
        assert_eq!(d.cached_n(), 60);
        // NOT prefill clean: `positions` still reports it, `reuse_candidates` hides it,
        // and with only empty slots left to fall back on the request is cold
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, false);
        assert_eq!(c.positions(), vec![Some(60), None, None]);
        assert_eq!(c.reuse_candidates(), vec![None, None, None]);
        let d = c.decide(&held, &new);
        assert_eq!(d.l, 100);
        assert_eq!(d.reuse, None);
        assert_eq!(d.cached_n(), 0);
    }

    /// M2b (one slot, #36) + M3: SLOTS prompt snapshots per process, newest in slot 0, so
    /// a history edit that diverges below the newest position still finds the previous turn's
    #[test]
    fn the_process_holds_three_prompt_slots() {
        assert_eq!(SLOTS, 3);
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, true);
        assert_eq!(c.positions(), vec![Some(60), None, None]);
        assert_eq!(c.reuse_candidates(), vec![Some(60), None, None]);
        assert_eq!(c.reuse_candidates().iter().filter(|p| p.is_some()).count(), 1);
    }

    /// the edit case from the live serve log: three held prompts (newest 87746, previous
    /// 86359, older 86254), the edit diverges at 87389, so the newest is out of reach and
    /// the previous turn is reused instead of resetting to a full cold prefill
    #[test]
    fn an_edit_below_the_newest_snapshot_reuses_the_previous_turn() {
        assert_eq!(
            reuse_slot(&[Some(87_746), Some(86_359), Some(86_254)], 87_389, 87_510),
            Some((1, 86_359))
        );
    }

    /// `invalidate` is what a cold start runs: every slot forgets position AND flag
    #[test]
    fn invalidate_clears_the_position_and_the_flag() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, true);
        assert_eq!(c.reuse_candidates(), vec![Some(60), None, None]);
        c.invalidate();
        assert_eq!(c.positions(), vec![None, None, None]);
        assert_eq!(c.reuse_candidates(), vec![None, None, None]);
        // and a request that would have been warm is now cold
        assert_eq!(
            c.decide(
                &(0..100).collect::<Vec<i64>>(),
                &(0..110).collect::<Vec<i64>>()
            )
            .reuse,
            None
        );
    }

    // ------------------------------------------------ #100: the serve request, host side

    /// - one `chat_generate` of `serve`, host side only: `decide`, then the rollback's or
    ///   the cold start's bookkeeping, the prefill's `history`, the point 1 snapshot's
    ///   bookkeeping, and the answer's ids appended to `history` as `decode_step` does
    /// - returns the `cached` count the `[chat]` line and the wire report
    fn serve_request(c: &mut PrefixCache, history: &mut Vec<i64>, prompt: &[i64], answer: &[i64]) -> usize {
        let d = c.decide(history, prompt);
        let cached = match d.reuse {
            Some((slot, p)) => {
                let written = history.len();
                history.truncate(p);
                c.rolled_back(slot, written);
                p
            }
            None => {
                history.clear();
                c.invalidate();
                0
            }
        };
        history.extend_from_slice(&prompt[cached..]);
        let slot = c.claim(history.len());
        c.name(slot, history.len(), true, history.len() / 4, Some(7));
        history.extend_from_slice(answer);
        cached
    }

    /// the #91 replay probe of 2026-09-22 (`decode_out/corruption-replay/baseline.log`):
    /// three prompts of one session, each a prefix of the next, sent 8 times each in that
    /// order (point major), every answer different (fresh seed)
    fn replay_probe(c: &mut PrefixCache) -> Vec<Vec<usize>> {
        let lens = [6_738usize, 24_410, 39_273];
        let session: Vec<i64> = (0..lens[2] as i64).collect();
        let mut history = Vec::new();
        let mut out = Vec::new();
        for (k, &n) in lens.iter().enumerate() {
            let mut row = Vec::new();
            for seed in 0..8i64 {
                let answer: Vec<i64> = (0..20).map(|j| 1_000_000 + 1_000 * (10 * k as i64 + seed) + j).collect();
                row.push(serve_request(c, &mut history, &session[..n], &answer));
            }
            out.push(row);
        }
        out
    }

    /// #100: an identical re-request reuses the whole prompt, and eight repeats of one
    /// prompt never evict the snapshot of the prefix the next prompt shares
    #[test]
    fn the_replay_probe_reuses_every_identical_prompt_and_keeps_the_shared_prefix() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let got = replay_probe(&mut c);
        assert_eq!(
            got,
            vec![
                vec![0, 6_738, 6_738, 6_738, 6_738, 6_738, 6_738, 6_738],
                vec![6_738, 24_410, 24_410, 24_410, 24_410, 24_410, 24_410, 24_410],
                vec![24_410, 39_273, 39_273, 39_273, 39_273, 39_273, 39_273, 39_273],
            ]
        );
        // and after the run the three prompt ends are held, newest first, no duplicate
        assert_eq!(c.positions(), vec![Some(39_273), Some(24_410), Some(6_738)]);
    }

    /// #101: a rollback onto an OLDER snapshot rewrites every KV row above it, so the
    /// newer snapshots above that point name a history the engine no longer holds and
    /// must never be rolled back onto again
    #[test]
    fn a_rollback_forgets_every_snapshot_above_its_point() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut history: Vec<i64> = Vec::new();
        // three turns of one conversation: prompt ends at 100, 200, 300
        let conv: Vec<i64> = (0..400).collect();
        for n in [100usize, 200, 300] {
            serve_request(&mut c, &mut history, &conv[..n], &[]);
        }
        assert_eq!(c.positions(), vec![Some(300), Some(200), Some(100)]);
        // an edit at 250 rolls back onto 200 and prefills a DIFFERENT tail to 280
        let mut edited: Vec<i64> = conv[..250].to_vec();
        edited.extend(5_000..5_030);
        assert_eq!(serve_request(&mut c, &mut history, &edited, &(6_000..6_040).collect::<Vec<i64>>()), 200);
        // the snapshot at 300 held the recurrent state of the OLD tokens 250..300
        assert!(!c.positions().contains(&Some(300)), "stale snapshot kept: {:?}", c.positions());
        // the next turn extends the edited branch past 300: it must land on 280, not 300
        let mut next = history.clone();
        next.extend(7_000..7_050);
        let d = c.decide(&history, &next);
        assert_eq!(d.reuse.map(|(_, p)| p), Some(280));
    }

    /// #100: a snapshot at a position already held re-uses that slot, moved to the front
    #[test]
    fn a_snapshot_at_a_held_position_replaces_that_slot_instead_of_a_duplicate() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        for (i, p) in [300usize, 200, 100].into_iter().enumerate() {
            c.set_slot(i, p, true);
        }
        assert_eq!(c.claim(200), SLOT_PROMPT);
        c.name(SLOT_PROMPT, 200, true, 50, Some(1));
        assert_eq!(c.positions(), vec![Some(200), Some(300), Some(100)]);
    }

    /// #100: a new position takes an empty slot first, else the smallest position
    #[test]
    fn a_new_snapshot_takes_an_empty_slot_else_the_smallest_position() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(0, 100, true);
        c.claim(200);
        c.name(SLOT_PROMPT, 200, true, 50, Some(1));
        assert_eq!(c.positions(), vec![Some(200), Some(100), None]);
        c.claim(300);
        c.name(SLOT_PROMPT, 300, true, 75, Some(1));
        assert_eq!(c.positions(), vec![Some(300), Some(200), Some(100)]);
        c.claim(400);
        c.name(SLOT_PROMPT, 400, true, 100, Some(1));
        assert_eq!(c.positions(), vec![Some(400), Some(300), Some(200)]);
    }

    /// #100 + #32 A10: a slot filled from a slot file has no logits row, so an identical
    /// request after a restore keeps the old guard; `set_slot` models the same (no row)
    #[test]
    fn a_slot_without_a_logits_row_is_not_reused_at_the_request_length() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_prompt_slot(100, 25);
        assert_eq!(c.logits_held(), vec![false, false, false]);
        let ids: Vec<i64> = (0..100).collect();
        assert_eq!(c.decide(&ids, &ids).reuse, None);
        // with a row it is taken, and `decide_for(.., false)` refuses it again
        c.name(SLOT_PROMPT, 100, true, 25, Some(3));
        assert_eq!(c.decide(&ids, &ids).reuse, Some((SLOT_PROMPT, 100)));
        assert_eq!(c.decide_for(&ids, &ids, false).reuse, None);
        // a cold start forgets the row with the position
        c.invalidate();
        assert_eq!(c.logits_held(), vec![false, false, false]);
    }

    /// #100, the Crow #217 retry: the same prompt again after an answer, with the
    /// previous turns held below it; nothing is prefilled and nothing is evicted
    #[test]
    fn an_identical_retry_is_served_from_the_prompt_snapshot_and_evicts_nothing() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut history = Vec::new();
        let conv: Vec<i64> = (0..1_000).collect();
        for n in [300usize, 600, 900] {
            serve_request(&mut c, &mut history, &conv[..n], &[9_000, 9_001]);
        }
        for retry in 0..10i64 {
            let cached = serve_request(&mut c, &mut history, &conv[..900], &[9_100 + retry]);
            assert_eq!(cached, 900, "retry {retry}");
        }
        assert_eq!(c.positions(), vec![Some(900), Some(600), Some(300)]);
    }

    // ------------------------------------------------ #114: image-aware prefix

    use crate::vit::{ImageSpan, IMAGE_PAD};

    /// the colour-probe shape of 2026-09-24: 10 text ids, one image of `n` visual tokens
    /// (all `IMAGE_PAD`), then 13 text ids of the question
    fn probe_prompt(n: usize) -> Vec<i64> {
        let mut v: Vec<i64> = (0..10).collect();
        v.extend(std::iter::repeat(IMAGE_PAD).take(n));
        v.extend(100..113);
        v
    }

    /// a span whose 32-byte identity is `hash` spread over its first eight bytes
    fn img(start: usize, len: usize, hash: u64) -> ImageSpan {
        let mut key = [0u8; 32];
        key[..8].copy_from_slice(&hash.to_le_bytes());
        ImageSpan { start, len, hash: key }
    }

    /// the defect: same ids, DIFFERENT image. The held prompt snapshot (with its logits
    /// row) sat at the request length and was taken, `prefill 0 of 1047` - the new image
    /// was never spliced. `L` must stop at the image's first row, so only a snapshot at or
    /// below it is a candidate
    #[test]
    fn same_ids_with_a_different_image_are_not_reused_past_the_image_start() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let ids = probe_prompt(1024);
        c.set_slot(1, 8, true);
        c.name(SLOT_PROMPT, ids.len(), true, ids.len() / 4, Some(7));
        let d = c.decide_mm(&ids, &[img(10, 1024, 0xAAAA)], &ids, &[img(10, 1024, 0xBBBB)], true);
        assert_eq!(d.l, 10, "L past the first row of an image the engine does not hold");
        assert_eq!(d.reuse, Some((1, 8)), "the only snapshot below the image is the reuse point");
        // without a snapshot below the image it is a cold start
        c.invalidate();
        c.name(SLOT_PROMPT, ids.len(), true, ids.len() / 4, Some(7));
        let d = c.decide_mm(&ids, &[img(10, 1024, 0xAAAA)], &ids, &[img(10, 1024, 0xBBBB)], true);
        assert_eq!(d.reuse, None);
    }

    /// the same image re-sent (an agent loop resends its whole history): full reuse,
    /// including the #100 zero-prefill path at the request length
    #[test]
    fn the_same_image_again_is_reused_in_full() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let ids = probe_prompt(1024);
        c.name(SLOT_PROMPT, ids.len(), true, ids.len() / 4, Some(7));
        let span = [img(10, 1024, 0xAAAA)];
        let d = c.decide_mm(&ids, &span, &ids, &span, true);
        assert_eq!(d.l, ids.len());
        assert_eq!(d.reuse, Some((SLOT_PROMPT, ids.len())));
        // and a next turn that appends after the image rolls back onto the whole prompt
        let mut next = ids.clone();
        next.extend([500, 501, 502]);
        let d = c.decide_mm(&ids, &span, &next, &span, true);
        assert_eq!(d.reuse, Some((SLOT_PROMPT, ids.len())));
    }

    /// a text-only exchange has no spans: `L` is the id prefix, as before #114
    #[test]
    fn text_only_requests_decide_exactly_as_before() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let held: Vec<i64> = (0..200).collect();
        let mut new = held.clone();
        new.extend([900, 901]);
        c.set_slot(0, 200, true);
        c.set_slot(1, 120, true);
        assert_eq!(c.decide_mm(&held, &[], &new, &[], true), c.decide(&held, &new));
        let edited: Vec<i64> = (0..150).chain(700..760).collect();
        assert_eq!(c.decide_mm(&held, &[], &edited, &[], true), c.decide(&held, &edited));
        assert_eq!(common_prefix_len_mm(&held, &[], &edited, &[]), common_prefix_len(&held, &edited));
    }

    /// an image AFTER the difference, or an equal image before a later different one:
    /// the prefix ends at the first image that differs, not at the first image
    #[test]
    fn the_prefix_ends_at_the_first_image_that_differs() {
        // text 0..5, image A at 5 (4 rows), text 9..12, image at 12 (4 rows), text 16..20
        let mut ids: Vec<i64> = (0..5).collect();
        ids.extend([IMAGE_PAD; 4]);
        ids.extend(50..53);
        ids.extend([IMAGE_PAD; 4]);
        ids.extend(60..64);
        let held = [img(5, 4, 1), img(12, 4, 2)];
        assert_eq!(common_prefix_len_mm(&ids, &held, &ids, &[img(5, 4, 1), img(12, 4, 3)]), 12);
        assert_eq!(common_prefix_len_mm(&ids, &held, &ids, &[img(5, 4, 9), img(12, 4, 2)]), 5);
        assert_eq!(common_prefix_len_mm(&ids, &held, &ids, &held), ids.len());
        // an id difference below both images wins over them
        let mut edited = ids.clone();
        edited[3] = 999;
        assert_eq!(common_prefix_len_mm(&ids, &held, &edited, &held), 3);
    }

    /// a held image the request does not name (a slot file restores ids without spans,
    /// or a text-only request carries the same pads) and the reverse: both are a
    /// difference at that image's start
    #[test]
    fn an_image_only_one_side_can_name_is_a_difference() {
        let ids = probe_prompt(16);
        let span = [img(10, 16, 0xAAAA)];
        assert_eq!(common_prefix_len_mm(&ids, &[], &ids, &span), 10);
        assert_eq!(common_prefix_len_mm(&ids, &span, &ids, &[]), 10);
        // the same hash with another token count (another grid) is another image
        assert_eq!(common_prefix_len_mm(&ids, &span, &ids, &[img(10, 12, 0xAAAA)]), 10);
    }

    /// a disabled cache still decides cold, spans or not
    #[test]
    fn a_disabled_cache_decides_cold_with_images_too() {
        let c = PrefixCache::for_shape(tiny(), false);
        let ids = probe_prompt(16);
        let span = [img(10, 16, 1)];
        assert_eq!(c.decide_mm(&ids, &span, &ids, &span, true), Decision { l: 0, reuse: None, parked: false });
    }

    // ------------------------------------------------ #118: a side request parks the conversation

    /// the n_ctx of the 2026-09-25 operating point (`[load] ... 200000 x fp8_e4m3`)
    const N_CTX: usize = 200_000;

    /// - the device side of `serve` as far as #118 needs it: `kv[i]` is the id whose KV row
    ///   sits at row `i` (the rows are absolutely addressed), `stash` the host copy of a park
    /// - every conversation of a test uses its own id range, so a row another conversation
    ///   wrote is seen as a wrong id
    struct Dev {
        history: Vec<i64>,
        kv: Vec<i64>,
        stash: Vec<i64>,
    }

    impl Dev {
        fn new() -> Dev {
            Dev { history: Vec::new(), kv: vec![-1; N_CTX], stash: Vec::new() }
        }
    }

    /// - one `chat_generate` of `serve`, host side, with the #118 park: `decide`, then the
    ///   unpark and rollback (warm) or `plan_cold` + the stash copy + `apply_cold` (cold,
    ///   `PrefixCache::cold_start`), the prefill's rows, the point 1 snapshot, the answer
    /// - on every warm request it checks the state it resumes from: every KV row below `P`
    ///   holds the id `history` names there (the row was not overwritten by another
    ///   conversation, or was put back by the unpark)
    /// - returns `(cached, the decision came from the parked conversation)`
    fn serve_kv(c: &mut PrefixCache, dev: &mut Dev, prompt: &[i64], answer: &[i64]) -> (usize, bool) {
        let d = c.decide(&dev.history, prompt);
        let cached = match d.reuse {
            Some((slot, p)) => {
                if d.parked {
                    let (_, rows, _) = c.parked().expect("a parked decision without a park");
                    let pk = c.adopt_parked().expect("a parked decision without a park");
                    dev.kv[..rows].copy_from_slice(&dev.stash[..rows]);
                    dev.history = pk.history;
                }
                // `rollback`, host side
                let written = dev.history.len();
                dev.history.truncate(p);
                c.rolled_back(slot, written);
                assert_eq!(&dev.kv[..p], &dev.history[..p], "rollback onto {p} over rows another request wrote");
                p
            }
            None => {
                let plan = c.plan_cold(dev.history.len(), prompt.len());
                if let ColdPlan::Park { rows } = plan {
                    dev.stash = dev.kv[..rows].to_vec();
                }
                let blocks = match plan {
                    ColdPlan::Park { rows } => park_blocks(rows, N_CTX),
                    _ => 0,
                };
                let h = std::mem::take(&mut dev.history);
                c.apply_cold(plan, h, Vec::new(), blocks);
                0
            }
        };
        for (i, &id) in prompt.iter().enumerate().skip(cached) {
            dev.kv[i] = id;
        }
        dev.history.extend_from_slice(&prompt[cached..]);
        let slot = c.claim(dev.history.len());
        c.name(slot, dev.history.len(), true, dev.history.len() / 4, Some(7));
        for &id in answer {
            dev.kv[dev.history.len()] = id;
            dev.history.push(id);
        }
        (cached, d.parked)
    }

    /// the main Crow conversation of 2026-09-25 17:30 (engine.log): three turns that leave
    /// the snapshots `[116900, 116202, 115944]`, the last answer 711 ids long
    fn main_conversation(c: &mut PrefixCache, dev: &mut Dev) -> Vec<i64> {
        let main: Vec<i64> = (0..130_000).collect();
        for (k, n) in [115_944usize, 116_202, 116_900].into_iter().enumerate() {
            let len = if k == 2 { 711 } else { 40 };
            let answer: Vec<i64> = (0..len).map(|j| 1_000_000 + 10_000 * k as i64 + j).collect();
            serve_kv(c, dev, &main[..n], &answer);
        }
        assert_eq!(c.positions(), vec![Some(116_900), Some(116_202), Some(115_944)]);
        assert_eq!(dev.history.len(), 117_611, "held 117611 before the judge (engine.log 17:31:14)");
        main
    }

    /// the judge request of 17:31:14: 4,659 prompt ids of an unrelated context, 288 generated
    fn judge(round: i64) -> (Vec<i64>, Vec<i64>) {
        let base = 50_000_000 + 1_000_000 * round;
        ((base..base + 4_659).collect(), (0..288).map(|j| 90_000_000 + j).collect())
    }

    /// #118, the measured defect: `[116900, 116202, 115944]` -> judge 4,659 (COLD) ->
    /// the next main request of 118,282 ids. Before the fix the judge's cold start
    /// dropped every snapshot (`[Some(4659), None, None]`) and the main request
    /// prefilled 118,282 of 118,282 ids (139.5 s); now it resumes from 116,900
    #[test]
    fn a_short_side_request_keeps_the_main_conversation_warm_from_its_newest_snapshot() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut dev = Dev::new();
        let main = main_conversation(&mut c, &mut dev);
        let (jp, ja) = judge(0);
        assert_eq!(serve_kv(&mut c, &mut dev, &jp, &ja), (0, false), "the judge itself is cold");
        let (cached, parked) = serve_kv(&mut c, &mut dev, &main[..118_282], &[7; 30]);
        assert_eq!((cached, parked), (116_900, true), "main must be WARM from 116900, prefill 1382");
        // the judge's snapshot had taken the smallest parked slot (115944); the park is
        // consumed by the unpark, the newest two main snapshots are the live ones again
        assert!(c.parked().is_none());
        assert_eq!(c.positions(), vec![Some(118_282), Some(116_900), Some(116_202)]);
    }

    /// #118: the lighthouse step 5 pattern: judge rounds between main turns, many times
    #[test]
    fn every_judge_round_leaves_the_next_main_turn_warm() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut dev = Dev::new();
        let main = main_conversation(&mut c, &mut dev);
        let mut n = 116_900;
        for round in 0..8i64 {
            let (jp, ja) = judge(round);
            assert_eq!(serve_kv(&mut c, &mut dev, &jp, &ja).0, 0);
            let before = n;
            n += 1_000;
            let answer: Vec<i64> = (0..200).map(|j| 2_000_000 + 1_000 * round + j).collect();
            assert_eq!(serve_kv(&mut c, &mut dev, &main[..n], &answer), (before, true), "round {round}");
        }
    }

    /// #118: two side requests in a row, the second one warm on the first one's prefix,
    /// then the main turn: the park survives both
    #[test]
    fn two_side_requests_in_a_row_keep_the_park() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut dev = Dev::new();
        let main = main_conversation(&mut c, &mut dev);
        let (jp, ja) = judge(0);
        serve_kv(&mut c, &mut dev, &jp, &ja);
        // the same judge context again with a longer tail: warm on its own snapshot
        let mut jp2 = jp.clone();
        jp2.extend(60_000_000..60_000_100);
        assert_eq!(serve_kv(&mut c, &mut dev, &jp2, &ja), (4_659, false));
        // an unrelated third one: cold, and the park is KEPT (it saves more than the judge)
        let (jp3, ja3) = judge(5);
        assert_eq!(c.plan_cold(dev.history.len(), jp3.len()), ColdPlan::Keep);
        assert_eq!(serve_kv(&mut c, &mut dev, &jp3, &ja3), (0, false));
        assert_eq!(serve_kv(&mut c, &mut dev, &main[..118_282], &[]), (116_900, true));
    }

    /// #118: the judge's cold start parks the three main snapshots, and the judge's own
    /// snapshot takes the smallest of them, never the newest
    #[test]
    fn the_judge_parks_the_main_snapshots_and_evicts_only_the_oldest() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut dev = Dev::new();
        main_conversation(&mut c, &mut dev);
        assert_eq!(c.plan_cold(117_611, 4_659), ColdPlan::Park { rows: 8_192 });
        let (jp, ja) = judge(0);
        serve_kv(&mut c, &mut dev, &jp, &ja);
        assert_eq!(c.parked_positions(), vec![None, Some(116_900), Some(116_202)]);
        assert_eq!(c.positions(), vec![Some(4_659), Some(116_900), Some(116_202)]);
        assert_eq!(c.reuse_candidates(), vec![Some(4_659), None, None]);
        assert_eq!(c.parked(), Some((117_611, 8_192, 0)));
    }

    /// #118, the guard: a side request that writes past the park's rows makes every parked
    /// snapshot above them unusable, also when a later rollback cuts `history` shorter than
    /// the rows it wrote. Cap 100 rows, main snapshots at 1,000
    #[test]
    fn a_side_conversation_that_wrote_past_the_park_leaves_the_main_one_cold() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.park_cap = 100;
        let mut dev = Dev::new();
        let main: Vec<i64> = (0..2_000).collect();
        serve_kv(&mut c, &mut dev, &main[..1_000], &[1_000_001; 5]);
        let side: Vec<i64> = (5_000_000..5_000_060).collect();
        assert_eq!(c.plan_cold(1_005, 60), ColdPlan::Park { rows: 100 });
        // 60 prompt + 50 answer ids: rows 0..110 written, 10 past the park
        serve_kv(&mut c, &mut dev, &side, &[5_900_000; 50]);
        // warm on the side snapshot at 60: `history` is cut back to 60, then 70 long
        let mut side2 = side.clone();
        side2.extend(6_000_000..6_000_010);
        assert_eq!(serve_kv(&mut c, &mut dev, &side2, &[]), (60, false));
        assert_eq!(dev.history.len(), 70);
        // rows 100..110 hold side ids: the main turn must NOT roll back onto 1,000
        assert_eq!(serve_kv(&mut c, &mut dev, &main[..1_200], &[]), (0, false));
    }

    /// #118: a cold request at least as long as the cap (a context rollover of the main
    /// conversation, 71,116 ids at 14:54:29) leaves nothing a park could keep: the
    /// pre-#118 cold start, every snapshot dropped
    #[test]
    fn a_cold_request_longer_than_the_park_drops_every_snapshot() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        let mut dev = Dev::new();
        main_conversation(&mut c, &mut dev);
        assert_eq!(c.plan_cold(117_611, 71_116), ColdPlan::Drop);
        let other: Vec<i64> = (70_000_000..70_071_116).collect();
        serve_kv(&mut c, &mut dev, &other, &[]);
        assert_eq!(c.positions(), vec![Some(71_116), None, None]);
        assert!(c.parked().is_none());
    }

    /// #118: `CROW_PREFIX_PARK_ROWS=0` is the pre-#118 cold start
    #[test]
    fn a_park_cap_of_zero_never_parks() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.park_cap = 0;
        let mut dev = Dev::new();
        let main = main_conversation(&mut c, &mut dev);
        let (jp, ja) = judge(0);
        serve_kv(&mut c, &mut dev, &jp, &ja);
        assert_eq!(c.positions(), vec![Some(4_659), None, None], "the 2026-09-25 17:31:27 state");
        assert_eq!(serve_kv(&mut c, &mut dev, &main[..118_282], &[]), (0, false));
    }

    /// #118: the parked conversation's LAST snapshot is never evicted while a live or an
    /// empty slot can take the new one
    #[test]
    fn a_snapshot_never_takes_the_last_parked_slot() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(0, 500, true);
        c.apply_cold(ColdPlan::Park { rows: 500 }, (0..500).collect(), Vec::new(), 126);
        assert_eq!(c.parked_positions(), vec![Some(500), None, None]);
        for p in [10usize, 20, 30, 40] {
            c.claim(p);
            c.name(SLOT_PROMPT, p, true, p / 4, Some(1));
        }
        assert_eq!(c.parked_positions().into_iter().flatten().collect::<Vec<_>>(), vec![500]);
        assert!(c.parked().is_some());
    }

    /// #118: the host RAM of a park, the numbers the docs and the boot line state
    #[test]
    fn the_park_costs_108_mib_at_the_default_cap_with_fp8_kv() {
        assert_eq!(park_blocks(8_192, N_CTX), 2_049);
        assert_eq!(park_host_bytes(8_192, N_CTX, 12, 1), 113_252_352);
        assert_eq!(park_host_bytes(8_192, N_CTX, 12, 2), 213_915_648);
        // the judge of 17:31 wrote 4,947 rows, the largest of the 37 side requests that day
        assert!(parked_usable(116_900, 4_947, park_rows(116_900, PARK_ROWS_DEFAULT)));
        assert!(!parked_usable(116_900, 8_193, park_rows(116_900, PARK_ROWS_DEFAULT)));
        // the pooled block count never runs past `qsa_pooled` (ceil(n_ctx / 4) blocks)
        assert_eq!(park_blocks(N_CTX, N_CTX), 50_000);
    }
}
