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
    let mut cnq = Cnq::open(&cnq_path);
    // #77 CROW_CNQ_OVERLAY: a second CNQ1 container opened BESIDE the base one, holding the
    // dense text tensors as bf16. A tensor it names shadows the base tensor of the same name
    // and section for every reader in the engine. Unset - the default - attaches nothing and
    // the engine is byte-identical to a build without this block. One door for all three
    // bins: `decode`, `parity` and `serve` all come through here.
    if let Ok(ov_path) = std::env::var("CROW_CNQ_OVERLAY") {
        if !ov_path.is_empty() {
            match cnq.attach_overlay(&ov_path) {
                Ok(r) => {
                    println!(
                        "[overlay] {} — {} tensors shadowed, {} values, {:.2} GB bf16 (base {:.2} GB nvfp4), source {}, built {}",
                        r.path,
                        r.tensors,
                        r.values,
                        r.bytes as f64 / 1e9,
                        (r.values as f64 * 4.5 / 8.0) / 1e9,
                        r.source,
                        r.built
                    );
                    for (kind, count, values) in &r.per_kind {
                        println!("[overlay]   {count:>3} x {kind}  ({values} values)");
                    }
                }
                // loud and named, at the front door: a mismatch that reached a kernel would
                // be a wrong-size GEMV nobody could read out of a logit dump
                Err(why) => panic!("[overlay] refused: {why}"),
            }
        }
    }
    let ctx = cuda::Ctx::init();
    let cfg = Config { context: CONTEXT_FLOOR, ..Config::default() };
    (cnq, ctx, cfg, cnq_path, sidecar)
}
