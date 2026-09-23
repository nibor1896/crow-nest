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
//!   with pos_max > pos_next"). The cold start forgets all of them (`invalidate`).
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
//! - ONE held conversation per process (M1): a request that shares no prefix replaces it.
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

/// the ONE held conversation of this process (M1), with its ONE snapshot (M2b, #36)
pub struct PrefixCache {
    /// `CROW_PREFIX_CACHE=0` turns every request into a cold start
    enabled: bool,
    shape: Shape,
    slots: Vec<Snapshot>,
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
        PrefixCache { enabled, shape, slots }
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
    pub fn reuse_candidates(&self) -> Vec<Option<usize>> {
        self.slots
            .iter()
            .map(|s| if s.prefill_clean { s.pos } else { None })
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
        if !self.enabled {
            return Decision { l: 0, reuse: None };
        }
        let l = common_prefix_len(history, ids);
        let logits = if allow_exact { self.logits_held() } else { Vec::new() };
        Decision { l, reuse: reuse_slot_with_logits(&self.reuse_candidates(), &logits, l, ids.len()) }
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
        if !s.prefill_clean {
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

    /// - the slot forgets its position; the buffers stay allocated
    /// - called with every cold start, so no slot can name a discarded history
    pub fn invalidate(&mut self) {
        for s in self.slots.iter_mut() {
            s.pos = None;
            s.prefill_clean = false;
            s.greedy = None;
        }
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
    fn claim(&mut self, pos: usize) -> usize {
        let same = self.slots.iter().position(|s| s.pos == Some(pos));
        let empty = || self.slots.iter().position(|s| s.pos.is_none());
        let smallest = || {
            (0..self.slots.len())
                .min_by_key(|&i| self.slots[i].pos.unwrap_or(0))
                .unwrap_or(SLOT_PROMPT)
        };
        let i = same.or_else(empty).unwrap_or_else(smallest);
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
    fn rolled_back(&mut self, slot: usize) {
        let Some(p) = self.slots[slot].pos else { return };
        for s in self.slots.iter_mut() {
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
        eng.history.truncate(pos);
        eng.route_log.clear();
        self.rolled_back(slot);

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
        PrefixCache { enabled, shape, slots }
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
        let d = Decision { l: 12, reuse: None };
        assert_eq!(d.cached_n(), 0);
    }

    #[test]
    fn a_decision_reports_p_as_the_cached_tokens() {
        let d = Decision { l: 120, reuse: Some((1, 100)) };
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
        assert_eq!(d, Decision { l: 0, reuse: None });
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
                history.truncate(p);
                c.rolled_back(slot);
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
}
