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
//! | cold | no such `S_pos` exists; `Engine::reset_to_zero` runs and both slots are dropped |
//!
//! - Only a PREFILL CLEAN snapshot is a reuse candidate; see the section below.
//!
//! - `Engine::history` covers prompt AND generated ids (`gen.rs:2698`, `gen.rs:2945`).
//! - `S_pos < request length` is a guard, not a spec change: `prefill` of an empty
//!   slice has no last position to return a greedy id from.
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
//! When a snapshot is taken (M1, robin 2026-09-09: BOTH unconditional):
//!
//! | slot | point | position |
//! |---|---|---|
//! | `SLOT_PROMPT` | after the prefill of this turn's prompt | rendered prompt length |
//! | `SLOT_ANSWER` | after the last `decode_step` of this turn | prompt length + generated - 1 |
//!
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
//! two slots                        = 261,292,032 B = 249.19 MiB
//! ```
//!
//! - Pageable host RAM, allocated once at process start, reused per snapshot.
//! - Not pinned (the pinned tier is budgeted at 46 GiB, `geo.rs:110`).
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
//! Why the point 2 snapshot is NOT prefill clean:
//!
//! - It is taken after the answer, and `decode_step` wrote every row of that answer.
//! - It is still TAKEN (M1: both snapshots unconditional) and it is still reported.
//! - It is not offered to `reuse_slot`, so no request can roll back onto a decode row.
//! - Cost of not using it: the previous answer is re-prefilled, 63 tokens at the A9
//!   operating point, 0.03 s of the 22.9 s a cold turn pays.
//!
//! The one exception to "in-process state, never a file" (#32 A10, spec 7.6):
//!
//! - `engine/src/slot.rs` writes the `SLOT_PROMPT` slot to a file and reads it back.
//! - It is the deliberate exception, so the FILE carries the whole load shape.
//! - A restore refuses any shape mismatch instead of loading state that is only shaped right.
//! - Only `SLOT_PROMPT` is ever written: it is the one prefill clean position (see above).
//! - A restore names it through `set_prompt_slot` and CLEARS `SLOT_ANSWER`, so the next
//!   request lands on the ordinary warm path of this file, not on a second one.
//!
//! Off switch:
//!
//! | variable | effect |
//! |---|---|
//! | `CROW_PREFIX_CACHE=0` | no slots are allocated, every request is a cold start |
//! | | `/slots/0` then refuses save and restore: there is no slot to write or fill |

use crate::cuda;
use crate::gen::Engine;
use crate::geo::{GD, GDN_CONV, GDN_VHEADS};
use cudarc::driver::sys;

/// slot of the snapshot taken after the prompt prefill (spec 7.6, point 1)
pub const SLOT_PROMPT: usize = 0;
/// slot of the snapshot taken after the generated answer (spec 7.6, point 2)
pub const SLOT_ANSWER: usize = 1;
/// slots held per conversation
pub const SLOTS: usize = 2;

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
    let mut best: Option<(usize, usize)> = None;
    for (i, p) in positions.iter().enumerate() {
        let Some(p) = *p else { continue };
        if p > l || p >= new_len {
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
}

impl Snapshot {
    /// allocate once; every later snapshot writes into these buffers
    fn new(shape: &Shape) -> Snapshot {
        Snapshot {
            pos: None,
            prefill_clean: false,
            done_blocks: 0,
            gdn_s: (0..shape.gdn_layers).map(|_| vec![0f32; GDN_S_STATE]).collect(),
            gdn_conv: (0..shape.gdn_layers).map(|_| vec![0f32; GDN_CONV_STATE]).collect(),
            ple_state: vec![0f32; PLE_STATE],
            qsa_ring: (0..shape.attn_layers)
                .map(|_| vec![0f32; shape.qsa_ring_len])
                .collect(),
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

/// the ONE held conversation of this process (M1), with its two snapshots
pub struct PrefixCache {
    /// `CROW_PREFIX_CACHE=0` turns every request into a cold start
    enabled: bool,
    shape: Shape,
    slots: Vec<Snapshot>,
}

impl PrefixCache {
    /// - allocates both slots up front when the cache is on (spec 7.7)
    /// - reads the shape off the loaded states, so it can never outlive its load
    pub fn new(eng: &Engine) -> PrefixCache {
        let enabled = std::env::var("CROW_PREFIX_CACHE").as_deref() != Ok("0");
        let shape = Shape::of(eng);
        let slots = if enabled {
            (0..SLOTS).map(|_| Snapshot::new(&shape)).collect()
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
        if !self.enabled {
            return Decision { l: 0, reuse: None };
        }
        let l = common_prefix_len(history, ids);
        Decision { l, reuse: reuse_slot(&self.reuse_candidates(), l, ids.len()) }
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
    /// - the ANSWER slot is CLEARED: nothing may claim a position this process never wrote
    pub fn set_prompt_slot(&mut self, pos: usize, done_blocks: usize) {
        if let Some(s) = self.slots.get_mut(SLOT_PROMPT) {
            s.pos = Some(pos);
            s.prefill_clean = true;
            s.done_blocks = done_blocks;
        }
        if let Some(s) = self.slots.get_mut(SLOT_ANSWER) {
            s.pos = None;
            s.prefill_clean = false;
        }
    }

    /// - both slots forget their position; the buffers stay allocated
    /// - called with every cold start, so no slot can name a discarded history
    pub fn invalidate(&mut self) {
        for s in self.slots.iter_mut() {
            s.pos = None;
            s.prefill_clean = false;
        }
    }

    /// - copy the four recurrent buffers device to host into `slot`
    /// - `prefill_clean` says whether every row below `Engine::pos` is a prefill row;
    ///   `false` keeps the slot out of `reuse_candidates` (see the module doc)
    /// - returns the wall of the copy in ms (spec 7.9 asks for it)
    /// - a disabled cache does nothing and returns 0.0
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn snapshot(&mut self, eng: &Engine, slot: usize, prefill_clean: bool) -> f64 {
        if !self.enabled || slot >= self.slots.len() {
            return 0.0;
        }
        let t0 = std::time::Instant::now();
        // whatever the last launch left in flight must land before the copy reads it
        cuda::sync();
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
        s.pos = Some(eng.pos);
        s.prefill_clean = prefill_clean;
        s.done_blocks = eng.done_blocks;
        t0.elapsed().as_secs_f64() * 1e3
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
            (0..SLOTS).map(|_| Snapshot::new(&shape)).collect()
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

    /// an enabled cache with two empty slots is a cold start, and says so in both lists
    #[test]
    fn a_fresh_cache_holds_no_position() {
        let c = PrefixCache::for_shape(tiny(), true);
        assert!(c.enabled());
        assert_eq!(c.positions(), vec![None, None]);
        assert_eq!(c.reuse_candidates(), vec![None, None]);
        assert_eq!(c.decide(&[1, 2, 3], &[1, 2, 3, 4]).reuse, None);
    }

    /// `positions` reports every slot, `reuse_candidates` hides the ones that are not clean
    #[test]
    fn reuse_candidates_hides_the_slot_that_is_not_prefill_clean() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, true);
        c.set_slot(SLOT_ANSWER, 100, false);
        assert_eq!(c.positions(), vec![Some(60), Some(100)]);
        assert_eq!(c.reuse_candidates(), vec![Some(60), None]);
        // so the decision lands on point 1 even though point 2 sits at L
        let held: Vec<i64> = (0..100).collect();
        let mut new = held.clone();
        new.extend(200..210);
        let d = c.decide(&held, &new);
        assert_eq!(d.l, 100);
        assert_eq!(d.reuse, Some((SLOT_PROMPT, 60)));
        assert_eq!(d.cached_n(), 60);
    }

    /// `invalidate` is what a cold start runs: both slots forget position AND flag
    #[test]
    fn invalidate_clears_both_positions_and_both_flags() {
        let mut c = PrefixCache::for_shape(tiny(), true);
        c.set_slot(SLOT_PROMPT, 60, true);
        c.set_slot(SLOT_ANSWER, 100, true);
        assert_eq!(c.reuse_candidates(), vec![Some(60), Some(100)]);
        c.invalidate();
        assert_eq!(c.positions(), vec![None, None]);
        assert_eq!(c.reuse_candidates(), vec![None, None]);
        // and a request that would have been warm is now cold
        assert_eq!(c.decide(&(0..100).collect::<Vec<i64>>(), &(0..110).collect::<Vec<i64>>()).reuse, None);
    }
}
