//! #26 A4 - reset the engine to position 0 between requests (spec section 7).
//!
//! Purpose:
//!
//! - One `serve` process answers more than one request, without a prefix cache.
//! - After `reset_to_zero` the next `prefill` must equal a fresh process, id for id.
//! - #31 A9 did NOT replace this: the prefix cache (`engine/src/cache.rs`) calls it as its
//!   COLD path, whenever no snapshot sits at or below the longest common id prefix.
//! - The warm path of A9 restores instead of zeroing, and keeps this file's stream ordering.
//!
//! What is reset, and why:
//!
//! | field | evidence | why |
//! |---|---|---|
//! | `Engine::pos` | `gen.rs:2697`, `gen.rs:2944` | only grows; `prefill` reads it as `pos_base` |
//! | `Engine::history` | `gen.rs:2698`, `gen.rs:2945` | PLE n-gram prefix `history[pos_base-2..pos_base]` (`gen.rs:2550`) |
//! | `Engine::done_blocks` | `gen.rs:2696`, `gen.rs:2947` | QSA pooled block cursor (`gen.rs:1648-1660`) |
//! | `Ple::state` | `gen.rs:902`, `kernels.rs:2839` | `[10240][9]` dilated conv left state, carried chunk to chunk |
//! | `ThreeStates::gdn_conv` | `manager.rs:47`, `kernels.rs:902` | `[36][10240][3]` causal conv left state, carried chunk to chunk |
//! | `Engine::route_log` | `gen.rs:2894` | grows per token under `CROW_ROUTE_DUMP`, never shrinks |
//! | `Engine::graph_exec` | `gen.rs:2913` | see the stream section below (spec 7.2 remedy) |
//! | `Engine::cap_stream` | `gen.rs:2741` | see the stream section below |
//! | the ACTIVE stream | `cuda.rs:16`, `gen.rs:2742` | see the stream section below |
//!
//! The active stream, measured 2026-09-09 (this is why the graph is dropped):
//!
//! - `decode_step` creates the capture stream ONCE and leaves it active (`gen.rs:2740-2743`).
//! - `launch_v` and `upload_into` both read that active stream (`gen.rs:1194`, `cuda.rs:373`).
//! - `upload_into` skips its sync on any stream but the legacy one (`cuda.rs:381-386`).
//! - `prefill` uploads its per chunk scalars from TEMPORARIES (`gen.rs:2493-2510`).
//! - So a second `prefill` in the same process read freed host memory for those scalars.
//! - Measured: same prompt, greedy, request 1 gave id 18622, request 2 gave id 17.
//! - Dropping the graph and the stream restores the legacy stream for the next `prefill`.
//! - `decode_step` then re-creates the stream and re-captures (`gen.rs:2740`, `gen.rs:2836`).
//! - Cost: one eager decode step plus one graph instantiate per request.
//! - Not measured against keeping the graph: keeping it is what broke the ids.
//! - A4 gate, 18 token prompt, 29 generated: decode 1170 to 1203 ms over four runs.
//!
//! What needs no reset, and why:
//!
//! | field | why |
//! |---|---|
//! | `ThreeStates::gdn_s` | `delta_rule_persist{,_r}` zeroes S when `init_p=1` (`kernels.rs:996`, `kernels.rs:1067`) |
//! | | `prefill` sets `init=1` exactly for the first chunk when `pos == 0` (`gen.rs:2433`, `gen.rs:2495`) |
//! | `ThreeStates::kv_buf` | written at `pos_base` by `store_kv`; every read is bounded by `sel`/`sel_n` from `ncb` |
//! | `ThreeStates::qsa_keys` | ring row = `pos % ring`; `pool4_cache` reads only rows this chunk appended |
//! | `ThreeStates::qsa_pooled` | block `b` is written before it is scored; `ncb` caps the scan at `(pos+1)/4` |
//! | `Ple` row cache (`cache`, `gs`, `gs_host`, `slot_map`, `batch_tag`, `batch`) | content addressed by n-gram id; a hit is byte identical to a fill |
//! | `Ple::req`, `Ple::miss` | hit rate counters, read by no kernel |
//! | `Engine::sel_counts`, `adapt_base`, `adapt_ema` | `serve` never calls `adapt_tick` or `trickle_tick`; the hot set is the loaded one |
//! | `Engine::scalar_stage`, `embed_buf`, `sb_pack` | per step staging, overwritten before every replay |
//! | `Engine::stage`, `pf_*`, `pa_*` | per launch MoE plan and prefetch ring, rebuilt per chunk |
//! | `Engine::p` (`Params`) | every position dependent scalar is uploaded per chunk (`gen.rs:2494-2510`) and per step |
//! | `Engine::dev_sampler` | A6 enables it per sampled request (`enable_dev_sampler`, `gen.rs:2957`) and PARKS it for a greedy request |
//! | | parking moves the `DevSampler` out of the engine into `Srv::parked_sampler` (`serve.rs:1027`, `serve.rs:1044`, `serve.rs:1274`) |
//! | | deliberately not reset: the parking mechanism owns the field and reuses the same device buffers |
//!
//! Cost and ordering:
//!
//! - Zeroing is a host upload of `36 * 10240 * 3 + 10240 * 9` f32 = 4.8 MB.
//! - `cuda::sync` before, so nothing in flight still writes the buffers.
//! - `cuda::sync` after, because the upload is async and reads host memory.
//! - The graph teardown mirrors `impl Drop for Engine` (`gen.rs:3307-3317`).

use crate::cuda;
use crate::gen::Engine;
use crate::geo::GDN_CONV;

/// f32 slots of one GDN layer's causal conv state, `[10240][3]` (`manager.rs:47`)
const GDN_CONV_STATE: usize = GDN_CONV * 3;
/// f32 slots of the PLE dilated conv state, `[10240][9]` (`gen.rs:902`)
const PLE_STATE: usize = GDN_CONV * 9;

impl Engine {
    /// - position, history, block cursor and both recurrent conv states back to load state
    /// - the decode graph and its capture stream are dropped, the legacy stream made active
    /// - the next `prefill` sees `pos == 0` and therefore zeroes the GDN state S itself
    /// - weights, hot set, PLE row cache, KV and QSA buffers are left alone (see the module table)
    ///
    /// # Safety
    ///
    /// - a CUDA context must be current, as for every other engine call
    /// - no kernel of this engine may be in flight on another thread
    pub unsafe fn reset_to_zero(&mut self) {
        // whatever the last request left in flight must land before anything below
        cuda::sync();

        // spec 7.2: drop the captured decode graph and its stream, so the next
        // prefill runs on the legacy stream its temporaries-as-upload-source needs
        if self.graph_exec != 0 {
            cuda::graph_exec_destroy(self.graph_exec as cudarc::driver::sys::CUgraphExec);
            self.graph_exec = 0;
        }
        if self.cap_stream != 0 {
            cuda::set_stream(0);
            cuda::stream_destroy(self.cap_stream as cudarc::driver::sys::CUstream);
            self.cap_stream = 0;
        } else {
            cuda::set_stream(0);
        }

        self.pos = 0;
        self.history.clear();
        self.done_blocks = 0;
        self.route_log.clear();

        let z_conv = vec![0f32; GDN_CONV_STATE];
        for i in 0..self.st.gdn_conv.len() {
            cuda::to_f32_into(self.st.gdn_conv[i], &z_conv);
        }
        let z_ple = vec![0f32; PLE_STATE];
        cuda::to_f32_into(self.ple.state, &z_ple);

        // the uploads read `z_conv` / `z_ple`, which die with this frame
        cuda::sync();
    }
}
