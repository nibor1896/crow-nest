//! crow-nest engine — residency scheduler (#8), three-state manager (#9),
//! end-to-end production decode (#11). Kernel math is the probe-verified set
//! (probes p5–p16); see probes/p5_STATUS.md for the evidence chain.

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
// the boot sequence the engine-loading bins share: container, CUDA context, config
pub mod boot;
pub mod sample;
pub mod tokenizer;
// #29 A7 review: the tool-call parser of `bin/serve.rs`, extracted so it is unit tested
// in the library and `serve.rs` keeps only the chunk builders and the call sites
pub mod toolcall;
pub mod reset;
// #31 A9: the prefix cache (spec section 7) - snapshot, rollback and the id prefix rule
pub mod cache;
// #32 A10: the slot file behind POST /slots/0?action=save|restore (spec section 7)
pub mod slot;
// #VIT: the visual tower (the container "vit" section), Crow image decoding and
// preprocessing, the interleaved-mrope tables and the prefill splice plan
pub mod vit;
// TASK D (2026-09-17): the cuTile Rust pilot, behind the default-off `cutile-pilot`
// feature. An evaluation artefact for docs/cuda-rust-evaluation.md, not a production
// path; nothing in the engine calls it.
#[cfg(feature = "cutile-pilot")]
pub mod cutile_pilot;
