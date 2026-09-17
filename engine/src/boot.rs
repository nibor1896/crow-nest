//! One front door for the bins that load an engine: the container, the CUDA
//! context and the starting `Config`, opened in the order they have to be.

use crate::cnq::Cnq;
use crate::cuda;
use crate::geo::{Config, CONTEXT_FLOOR};

/// `CROW_CNQ` / `CROW_HOTSETS` (else the given defaults), the mapped container, a current CUDA context, the config at `CONTEXT_FLOOR`.
///
/// The RETURNED ORDER is the drop order: bound as `let (mut cnq, _ctx, mut cfg, cnq_path, sidecar) = open_model(..)`
/// the bindings drop in reverse, so the `Engine` loaded below them dies first, then the context, then the container
/// mapping — the order all three bins wrote by hand. A `#[must_use]` guard cannot order anything, so it would say less.
///
/// # Safety
///
/// - creates the process's CUDA primary context, so no kernel may have run yet
/// - the caller keeps `_ctx` alive for as long as any device allocation lives
pub unsafe fn open_model(
    cnq_default: String,
    sidecar_default: String,
) -> (Cnq, cuda::Ctx, Config, String, String) {
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or(cnq_default);
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or(sidecar_default);
    let cnq = Cnq::open(&cnq_path);
    let ctx = cuda::Ctx::init();
    let cfg = Config { context: CONTEXT_FLOOR, ..Config::default() };
    (cnq, ctx, cfg, cnq_path, sidecar)
}
