//! crow-nest engine — residency scheduler (#8), three-state manager (#9),
//! end-to-end production decode (#11). Kernel math is the probe-verified set
//! (probes p5–p16); see probes/p5_STATUS.md for the evidence chain.

pub mod cuda;
pub mod cnq;
pub mod geo;
pub mod kernels;
pub mod manager;
pub mod residency;
pub mod gen;
pub mod sample;
