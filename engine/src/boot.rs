//! One front door for the bins that load an engine: the container, the CUDA
//! context and the starting `Config`, opened in the order they have to be.

use crate::cnq::Cnq;
use crate::cuda;
use crate::geo::{Config, KvDtype, CONTEXT_FLOOR};
use crate::meta;

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
    // #102: CROW_KV is read HERE, once, for all three bins (it used to be read by
    // `decode parity` only, so `serve` booted FP8 under CROW_KV=bf16 and said nothing).
    // A bad value dies before the container is mapped and before any CUDA work.
    let kv = match KvDtype::from_env_value(std::env::var("CROW_KV").ok().as_deref()) {
        Ok(kv) => kv,
        Err(why) => panic!("[boot] refused: {why}"),
    };
    warn_unknown_crow_env();
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or(cnq_default);
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or(sidecar_default);
    // #94 phase 1 — the metadata gate, FIRST: the checkpoint's config.json is
    // parsed and every formula constant asserted equal to the pinned value
    // before the container is mapped and the CUDA context created, so a
    // mismatched checkpoint dies at the front door instead of computing
    // quietly wrong numbers (the llama.cpp get_key discipline). Zero numeric
    // change on the checkpoint of record; `None` is the selftest package (no
    // models/ dir beside the container), which continues after a WARN line.
    let _meta = meta::assert_pinned(&cnq_path);
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
                    // #79 made the line say WHICH overlay this is. `dense-bf16` trades bytes
                    // for precision and the planner sees the difference; `expert-nvfp4` is the
                    // same format at the same byte length, so "base" and "overlay" must read
                    // identical or something about the wiring is wrong.
                    println!(
                        "[overlay] {} — kind {}, {} tensors shadowed, {} values, {:.2} GB (base {:.2} GB), source {}, built {}",
                        r.path,
                        r.kind,
                        r.tensors,
                        r.values,
                        r.bytes as f64 / 1e9,
                        r.base_bytes as f64 / 1e9,
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
    let mut cfg = Config { context: CONTEXT_FLOOR, ..Config::default() };
    if let Some(kv) = kv {
        cfg.kv = kv;
    }
    tracing::info!(target: "boot", "[boot] kv cache dtype {} ({})", cfg.kv.name(),
        if kv.is_some() { "CROW_KV" } else { "default, CROW_KV unset" });
    (cnq, ctx, cfg, cnq_path, sidecar)
}

/// The variable table of record, compiled in: `docs/env.md` is kept equal to the
/// set of names in the sources by `tools/check_env_docs.py`, so its rows ARE the
/// names this engine reads.
const ENV_DOC: &str = include_str!("../../docs/env.md");

/// Every `CROW_*` name that owns a row in `docs/env.md` (the same row rule as
/// `tools/check_env_docs.py`: first cell is exactly one backticked name).
pub fn documented_crow_names() -> std::collections::BTreeSet<&'static str> {
    ENV_DOC
        .lines()
        .filter_map(|l| {
            let rest = l.trim_start().strip_prefix('|')?.trim_start().strip_prefix('`')?;
            let (name, tail) = rest.split_once('`')?;
            let ok = name.starts_with("CROW_")
                && name.len() > 5
                && name.bytes().all(|b| b.is_ascii_uppercase() || b.is_ascii_digit() || b == b'_')
                && tail.trim_start().starts_with('|');
            ok.then_some(name)
        })
        .collect()
}

/// The `CROW_*` names among `present` that the engine has no row for, sorted.
/// Pure, so the rule is tested without touching the process environment.
pub fn unknown_crow_names<'a>(
    present: impl IntoIterator<Item = &'a str>,
    known: &std::collections::BTreeSet<&str>,
) -> Vec<&'a str> {
    let mut v: Vec<&str> = present
        .into_iter()
        .filter(|n| n.starts_with("CROW_") && !known.contains(n))
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

/// #102: one WARN per `CROW_*` variable in the environment that no engine code
/// reads, so a misspelt or wrong-binary switch (the MEAS-0923 `CROW_KV` arm) is
/// on the boot log instead of silently ignored. Names only, never values: a
/// client secret (Crow's search API key) may be in the caller's shell. Warn, never
/// refuse: Crow and the tools/ scripts share the prefix.
fn warn_unknown_crow_env() {
    let known = documented_crow_names();
    let present: Vec<String> = std::env::vars_os()
        .filter_map(|(k, _)| k.into_string().ok())
        .collect();
    for name in unknown_crow_names(present.iter().map(|s| s.as_str()), &known) {
        tracing::warn!(target: "boot",
            "[boot] {name} is set but no engine code reads it (not in docs/env.md) - ignored");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_env_doc_rows_are_the_names_the_engine_reads() {
        let known = documented_crow_names();
        // the doc's own count line says 107 on e379361; the guard keeps it equal to the code
        assert!(known.len() >= 100, "only {} rows parsed from docs/env.md", known.len());
        for n in ["CROW_KV", "CROW_CNQ", "CROW_HOTSETS", "CROW_PINNED_BUDGET_GB", "CROW_GRAPH"] {
            assert!(known.contains(n), "{n} missing from the parsed doc rows");
        }
        assert!(!known.iter().any(|n| !n.starts_with("CROW_") || n.len() <= 5));
    }

    #[test]
    fn unknown_crow_names_are_reported_once_sorted_and_only_with_the_prefix() {
        let known = documented_crow_names();
        // the two unknown names are built at runtime: a literal here would be a
        // code name without a doc row for tools/check_env_docs.py
        let typo = format!("{}_KV_DTYPE", "CROW");
        let client = format!("{}_SEARCH_KEY", "CROW");
        let present = ["PATH", "CROW_KV", typo.as_str(), client.as_str(), typo.as_str(), "crow_kv", "CROW_GRAPH"];
        assert_eq!(unknown_crow_names(present, &known), vec![typo.as_str(), client.as_str()]);
        assert!(unknown_crow_names(["CROW_KV", "CROW_CNQ", "HOME"], &known).is_empty());
    }
}
