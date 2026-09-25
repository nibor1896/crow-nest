//! crow-nest engine — residency scheduler (#8), three-state manager (#9),
//! end-to-end production decode (#11). Kernel math is the probe-verified set
//! (probes p5–p16); see probes/p5_STATUS.md for the evidence chain.
//!
//! # The module map (2026-09-17, `log` added 2026-09-18 with #13)
//!
//! Six layers, no cycles: a module may use the layers above it and never the
//! ones below. The same graph is drawn in `docs/diagrams.md` (diagram 7) and
//! explained module by module in `docs/architecture.md` section 8; the logging
//! module has its own section 9.
//!
//! L0 — leaves, no in-crate dependency:
//!
//! - [`log`]: the logging facade (#13) — `tracing` with one target per component,
//!   the rotating gzipping file writer, the stderr mirror whose lines are the
//!   `eprintln!` lines of record byte for byte, the `CROW_LOG` filter, the boot
//!   report as one JSON line and the per-request routing line. It is a leaf, and
//!   it is the one module EVERY other module depends on: every line the engine
//!   says goes through it.
//! - [`cuda`]: the CUDA driver-API facade — context, NVRTC compile, module load,
//!   device alloc/copy/free, streams and graphs, pinned host memory, and the
//!   `/proc/meminfo` reading the pinned budget is derived from.
//! - [`cnq`]: the CNQ container reader — trailer index, whole-file mapping, every
//!   weight read, the FP4/FP8 host twins, and the `fadvise` purge that keeps the
//!   load from leaving a page-cache trail.
//! - [`geo`]: model geometry and `Config` — the constants every other module
//!   derives from, the chunk and adapt policies, `env_parse`.
//! - [`tokenizer`]: the in-engine HF tokenizer and chat template (`bin/serve` only).
//! - [`toolcall`]: the streaming `<tool_call>` parser (`bin/serve` only).
//! - [`toolgrammar`]: the lazy tool-call grammar and its vocabulary trie (`bin/serve` only).
//! - [`stopstr`]: the OpenAI stop-string hold (`bin/serve` only, #86).
//!
//! L1 — on the leaves:
//!
//! - [`kernels`] → cuda: `KERNEL_SRC` (the frozen CUDA source), the kernel table,
//!   `launch_v` / `launch_sync` and the per-kernel profile; `define_u32` reads the
//!   four `#define`s the Rust twins are asserted against at every load.
//! - [`manager`] → cuda, geo: the three-state allocator and the two-sided planner
//!   clamp; `derive_host_pinned_budget` and the RAM margin.
//! - [`sample`] → geo: the host sampler reference, the sampler profile, `EOS_IDS`.
//! - [`weights`] → cnq, cuda: the container tensor → device loaders and the NVFP4
//!   pair `Fp4` (no launch policy).
//! - [`boot`] → cnq, cuda, geo: `open_model` — container, CUDA context and the
//!   starting `Config`, in the order the engine-loading bins need them.
//!
//! L2 and above:
//!
//! - [`residency`] → cnq, cuda, geo, kernels, manager: the hot expert set in VRAM,
//!   the pinned cold tier, the swaps and the three-phase stream trickle.
//! - [`vit`] → cnq, cuda, geo, kernels, weights: the visual tower, image
//!   preprocessing, the interleaved-mrope tables and the prefill splice plan.
//! - [`gen`] → cnq, cuda, geo, kernels, manager, residency, sample, vit, weights:
//!   `Engine` — the boot (`Engine::load`), every layer primitive, `prefill` and
//!   `decode_step`, the device sampler, adaptation and the trickle.
//! - [`cache`] → cuda, gen, geo: the prefix cache (spec section 7).
//! - [`reset`] → cuda, gen, geo: the teardown and the zero state, as inherent
//!   `impl Engine` methods — the only foreign `impl` on `Engine` in the crate, and
//!   the reason `impl Drop for Engine` in `gen` can call `drop_decode_graph`
//!   without a `use` edge that any import graph would show.
//! - [`slot`] → cache, cuda, gen, geo: the slot file behind `POST /slots/0`.
//! - [`lend`] → cuda, gen, vit: #117 VRAM lending, the state machine, the parked queue and the tier-1 plan.
//!
//! The bins (`engine/src/bin`) are separate crates: `pub(crate)` is a hard wall to
//! them, so `Engine`'s API surface is what they can reach (architecture 8.3).

// #13: the logging facade - `tracing`, the rotating file, the stderr mirror the
// tools parse, the boot line and the per-request routing line. L0: it depends on
// no other module of this crate, and every other module depends on it.
pub mod log;
pub mod cuda;
pub mod cnq;
pub mod geo;
pub mod kernels;
pub mod manager;
pub mod residency;
// the container tensor -> device loaders and the NVFP4 pair (no launch policy):
// what `gen` and `vit` both need, so neither has to reach into the other
pub mod weights;
pub mod gen;
pub mod gguf;
// the boot sequence the engine-loading bins share: container, CUDA context, config
pub mod boot;
// #94 phase 1: the metadata gate — the checkpoint's config.json parsed at boot
// and every formula constant asserted equal to the pinned value BEFORE the
// container is mapped (zero numeric change; mismatch = loud named panic). It
// reads the pins of geo and sample, so it sits above them; boot is its caller.
pub mod meta;
pub mod sample;
pub mod tokenizer;
// #29 A7 review: the tool-call parser of `bin/serve.rs`, extracted so it is unit tested
// in the library and `serve.rs` keeps only the chunk builders and the call sites
pub mod toolcall;
// #93: the lazy tool-call grammar (llama.cpp qwen3_coder semantics) and the
// vocabulary trie its token masks walk; `bin/serve.rs` owns the redraw
pub mod toolgrammar;
// #86: the OpenAI stop-string filter of `bin/serve.rs`, the same tail-hold
// `toolcall::find_marker` gives `<tool_call>`, on arbitrary strings
pub mod stopstr;
pub mod reset;
// #31 A9: the prefix cache (spec section 7) - snapshot, rollback and the id prefix rule
pub mod cache;
// #32 A10: the slot file behind POST /slots/0?action=save|restore (spec section 7)
pub mod slot;
// #VIT: the visual tower (the container "vit" section), Crow image decoding and
// preprocessing, the interleaved-mrope tables and the prefill splice plan
pub mod vit;
// #117: VRAM lending to a co-resident GPU client - the state machine, the parked
// request queue and the tier-1 plan (pure); the VMM half lives in `cuda`
pub mod lend;
// TASK D (2026-09-17): the cuTile Rust pilot, behind the default-off `cutile-pilot`
// feature. An evaluation artefact for docs/cuda-rust-evaluation.md, not a production
// path; nothing in the engine calls it.
#[cfg(feature = "cutile-pilot")]
pub mod cutile_pilot;
