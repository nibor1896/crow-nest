//! One front door for the bins that load an engine: the container, the CUDA
//! context and the starting `Config`, opened in the order they have to be.

use crate::cnq::{Cnq, IndexPeek};
use crate::cuda;
use crate::geo::{Config, Ffn, Geo, KvDtype};
use crate::meta;

/// `CROW_CNQ` / `CROW_HOTSETS` (else the given defaults), the mapped container, a current CUDA context, the config at
/// the model's context floor, and the model's runtime `Geo` (Crow #300 C3: the engine loaded from these owns it).
/// The hot-set sidecar is `None` for a family without routed experts (C5, [`hot_set_sidecar`]).
///
/// The RETURNED ORDER is the drop order: bound as `let (mut cnq, _ctx, mut cfg, cnq_path, sidecar, geo) = open_model(..)`
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
) -> (Cnq, cuda::Ctx, Config, String, Option<String>, Geo) {
    // #102: CROW_KV is read HERE, once, for all three bins (it used to be read by
    // `decode parity` only, so `serve` booted FP8 under CROW_KV=bf16 and said nothing).
    // A bad value dies before the container is mapped and before any CUDA work.
    let kv = match KvDtype::from_env_value(std::env::var("CROW_KV").ok().as_deref()) {
        Ok(kv) => kv,
        Err(why) => panic!("[boot] refused: {why}"),
    };
    warn_unknown_crow_env();
    let cnq_path = std::env::var("CROW_CNQ").unwrap_or(cnq_default);
    // #94 phase 1 — the metadata gate, FIRST: the checkpoint's config.json is
    // parsed and every formula constant asserted equal to the pinned value
    // before the container is mapped and the CUDA context created, so a
    // mismatched checkpoint dies at the front door instead of computing
    // quietly wrong numbers (the llama.cpp get_key discipline). Zero numeric
    // change on the checkpoint of record; `None` is the selftest package (no
    // models/ dir beside the container), which continues after a WARN line.
    // Crow #300 C1/C2: the same door detects the model family, refuses unknown
    // config keys by name, and derives the runtime `Geo`, asserted equal to
    // `Geo::FLASH_NEXT` (a dense checkpoint prints its geometry and dies here).
    // C3: the `Geo` is handed back to the caller, which gives it to `Engine::load`;
    // the engine owns it from there (the way llama.cpp's `llama_model` owns its
    // `hparams`) and every host site reads its numbers from it.
    // C5: the same door refuses a family at its first unbuilt block (`Geo::built`).
    let geo = model_geo(&cnq_path);
    // C5: the hot-set sidecar belongs to the MoE arm; a family without routed
    // experts reads none, so CROW_HOTSETS is not required there (a set one is named
    // and ignored)
    let hotsets_env = std::env::var("CROW_HOTSETS").ok();
    if hotsets_env.is_some() && !matches!(geo.ffn, Ffn::Moe { .. }) {
        tracing::warn!(target: "boot",
            "[boot] CROW_HOTSETS is set but family {:?} has no routed experts (Ffn::Dense) - ignored", geo.family);
    }
    let sidecar = hot_set_sidecar(&geo, hotsets_env, sidecar_default);
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
    // Crow #300 phase 2: CROW_CONTEXT raises the context above the family's floor, up to the
    // checkpoint's max_position_embeddings; a bad value stops the boot here, by name
    let context = context_from_env(std::env::var("CROW_CONTEXT").ok().as_deref(), geo.context_floor, geo.context_max)
        .unwrap_or_else(|why| panic!("[boot] refused: {why}"));
    let ctx = cuda::Ctx::init();
    // Crow #300 phase 2: unset CROW_KV takes the family's default (Flash-Next FP8, the dense
    // family BF16, `Family::default_kv`)
    let cfg = Config { context, kv: kv.unwrap_or(geo.family.default_kv()), ..Config::default() };
    tracing::info!(target: "boot", "[boot] kv cache dtype {} ({})", cfg.kv.name(),
        if kv.is_some() { "CROW_KV".to_string() } else { format!("default of family {:?}, CROW_KV unset", geo.family) });
    (cnq, ctx, cfg, cnq_path, sidecar, geo)
}

/// Crow #300 phase 2: the context the boot allocates. Unset (or empty) is the family's floor
/// (`Geo::context_floor`: Flash-Next 200,000, the dense family 100,000); a value must be an
/// integer in `floor..=max` (`Geo::context_max`, the checkpoint's max_position_embeddings).
pub fn context_from_env(v: Option<&str>, floor: usize, max: usize) -> Result<usize, String> {
    let Some(v) = v.map(str::trim).filter(|v| !v.is_empty()) else { return Ok(floor) };
    let n: usize = v.parse().map_err(|_| format!("CROW_CONTEXT {v:?} is not a whole number of tokens"))?;
    if n < floor || n > max {
        return Err(format!("CROW_CONTEXT {n} is outside {floor}..={max} (the family's context floor .. max_position_embeddings)"));
    }
    Ok(n)
}

/// The runtime `Geo` of the checkpoint beside `cnq_path`, through the #94 / Crow #300
/// metadata gate (`meta::assert_pinned`: a refused config panics here, by name).
/// `open_model` calls it; so do the probe bins that map a container without the
/// front door. A container with no config.json beside it (the selftest package) gets
/// `Geo::FLASH_NEXT`, the pins of record, after the gate's WARN line - the same
/// "pins stand unchecked" behavior as before C3.
///
/// Crow #300 C5: then the family check (`Geo::built`): a `Geo` with an arm this
/// engine has not built panics here by the name of its first such block, still
/// before the container is mapped and before any CUDA call.
///
/// Crow #300 C7: the container's index trailer is read FIRST (`Cnq::peek_index`: no
/// mapping). An index v2 carries its model's config and the gate judges that; the
/// index v1 container of record keeps the pre-C7 path (`CROW_MODEL_DIR`, else
/// `models/`). One `[boot] container index` line says which, before the `meta:` line.
pub fn model_geo(cnq_path: &str) -> Geo {
    let peek = Cnq::peek_index(cnq_path).unwrap_or_else(|why| panic!("[cnq] refused: {cnq_path}: {why}"));
    let model_dir = std::env::var("CROW_MODEL_DIR").ok();
    tracing::info!(target: "boot", "{}", index_line(&peek, model_dir.as_deref()));
    let geo = match meta::assert_pinned(cnq_path, &peek, model_dir.as_deref()) {
        Some((_meta, geo)) => geo,
        None => Geo::FLASH_NEXT,
    };
    if let Err(why) = geo.built() {
        tracing::error!(target: "boot", "[boot] {why}");
        panic!("[boot] refused: {why}");
    }
    geo
}

/// `model_geo` without the process: the peeked index and `CROW_MODEL_DIR` in, the
/// `Geo` or the refusal text out. The same three steps in the same order (the
/// metadata gate, `Geo::FLASH_NEXT` when an index v1 finds no config, `Geo::built`),
/// so the tests drive the boot door on synthetic containers with no environment,
/// no panic and no CUDA.
pub fn geo_for(cnq_path: &str, peek: &IndexPeek, model_dir: Option<&str>) -> Result<Geo, String> {
    let geo = match meta::gate(cnq_path, peek, model_dir) {
        Ok(Some((_meta, geo))) => geo,
        Ok(None) => Geo::FLASH_NEXT,
        Err(meta::GateRefusal::Unreadable(why) | meta::GateRefusal::Refused(why)) => return Err(why),
    };
    geo.built()?;
    Ok(geo)
}

/// The boot line that names where the config comes from (Crow #300 C7).
pub fn index_line(peek: &IndexPeek, model_dir: Option<&str>) -> String {
    match peek.model() {
        None => format!(
            "[boot] container index v1 (the Flash-Next CNQ4.5-M container of record, index sha256 {}…): config from {}",
            &crate::cnq::CNQ45M_INDEX_SHA256[..12],
            match model_dir {
                Some(d) => format!("CROW_MODEL_DIR={d}"),
                None => "models/ beside the container (CROW_MODEL_DIR unset)".to_string(),
            }
        ),
        Some(m) => format!(
            "[boot] container index v2 (family {}, recipe {}, {} @ {}): config from its model block (config.json sha256 {}…){}",
            m.family,
            m.recipe,
            m.source_repo,
            m.source_revision,
            &crate::cnq::sha256_hex(m.config_json.as_bytes())[..12],
            if model_dir.is_some() { ", CROW_MODEL_DIR sha-checked against it" } else { "" }
        ),
    }
}

/// Crow #300 C5: the hot-set sidecar a boot of `geo` reads. The hot sets, the
/// residency planner and the expert slabs are the `Ffn::Moe` arm, so a MoE family
/// reads `CROW_HOTSETS` (else the caller's default) and a family without routed
/// experts reads none: `None`, whatever the variable says. Pure, so the rule is
/// tested without touching the process environment.
pub fn hot_set_sidecar(geo: &Geo, env: Option<String>, default: String) -> Option<String> {
    match geo.ffn {
        Ffn::Moe { .. } => Some(env.unwrap_or(default)),
        Ffn::Dense { .. } => None,
    }
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

    /// C5: CROW_HOTSETS is read for a MoE family only; a dense family needs no
    /// hot-set sidecar and ignores a set variable
    #[test]
    fn the_hot_set_sidecar_is_required_for_a_moe_family_only() {
        let (fnx, dense) = (Geo::FLASH_NEXT, crate::meta::dense_fixture_geo());
        let d = || "decode_out/default.json".to_string();
        assert_eq!(hot_set_sidecar(&fnx, None, d()), Some(d()));
        assert_eq!(hot_set_sidecar(&fnx, Some("x.json".into()), d()), Some("x.json".to_string()));
        assert_eq!(hot_set_sidecar(&dense, None, d()), None);
        assert_eq!(hot_set_sidecar(&dense, Some("x.json".into()), d()), None);
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

/// Crow #300 C7: the boot door on containers. The containers are synthetic CNQ1 files
/// written here (a trailer with the tensor table and, for v2, the model block; no payload is
/// read before `Cnq::open`), the configs are the tracked fixtures of the two families.
#[cfg(test)]
mod tests_300_c7 {
    use super::{geo_for, index_line};
    use crate::cnq::{sha256_hex, Cnq, IndexKind, IndexPeek, TensorInfo};
    use crate::geo::{Family, Geo};
    use crate::meta::{self, GateRefusal};

    fn fixture(dir: &str, f: &str) -> String {
        format!("{}/tests/fixtures/{dir}/{f}", env!("CARGO_MANIFEST_DIR"))
    }
    const FN: &str = "Qwen3.8-Flash-Next";
    const DENSE: &str = "Qwen3.8-27B";

    fn read(dir: &str, f: &str) -> String {
        std::fs::read_to_string(fixture(dir, f)).unwrap()
    }

    /// the tensor facts `container_mismatch` reads: the embedding and one tensor per layer
    fn tensors(hidden: u64, layers: usize) -> Vec<(String, Vec<u64>)> {
        let mut v = vec![("model.language_model.embed_tokens.weight".to_string(), vec![248_320, hidden])];
        v.extend((0..layers).map(|l| (format!("model.language_model.layers.{l}.input_layernorm.weight"), vec![hidden])));
        v
    }

    fn info((name, shape): &(String, Vec<u64>)) -> TensorInfo {
        TensorInfo {
            name: name.clone(),
            section: "text".into(),
            dtype: "bf16".into(),
            offset: 0,
            n_values: shape.iter().product(),
            global_scale: 1.0,
            shape: shape.clone(),
            overlay: false,
        }
    }

    /// a CNQ1 file with an index v2 trailer: `config` / `generation` as the model block's
    /// verbatim strings (with their sha256), `family` as the converter would have recorded it
    fn v2(tag: &str, config: &str, generation: &str, family: &str, model_type: &str, t: &[(String, Vec<u64>)]) -> String {
        let tensors: Vec<serde_json::Value> = t
            .iter()
            .map(|(n, s)| serde_json::json!({"name": n, "section": "text", "dtype": "bf16", "offset": 0,
                "n_values": s.iter().product::<u64>(), "shape": s}))
            .collect();
        let index = serde_json::json!({
            "format": "crow-nest-quant", "format_version": 2, "recipe": "c7-synthetic", "scales": "ceil",
            "blob_offset": 12, "sections": {}, "tensors": tensors,
            "model": {"family": family, "model_type": model_type,
                "config_json": config, "config_json_sha256": sha256_hex(config.as_bytes()),
                "generation_config_json": generation, "generation_config_json_sha256": sha256_hex(generation.as_bytes()),
                "geo": {}, "source": {"repo": "crow-nest/c7-synthetic", "revision": "c7", "shards": []}}
        });
        let ib = serde_json::to_vec(&index).unwrap();
        let mut f = b"CNQ1\0\0\0\0\0\0\0\0".to_vec();
        f.extend_from_slice(&ib);
        f.extend_from_slice(&(ib.len() as u64).to_le_bytes());
        let dir = std::env::temp_dir().join(format!("crow-c7-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join(format!("{tag}.cnq"));
        std::fs::write(&p, f).unwrap();
        p.to_string_lossy().into_owned()
    }

    fn flash_next_v2(tag: &str) -> String {
        v2(tag, &read(FN, "config.json"), &read(FN, "generation_config.json"), "FlashNext", "qwen4_exp_text", &tensors(2560, 48))
    }

    /// Requirement 1: an index v2 container boots its metadata gate from its own `model`
    /// block, with no `CROW_MODEL_DIR` and no `models/` beside it.
    #[test]
    fn a_v2_container_boots_its_metadata_gate_from_its_own_model_block() {
        let path = flash_next_v2("own-block");
        let peek = Cnq::peek_index(&path).unwrap();
        let (meta, geo) = meta::gate(&path, &peek, None).unwrap().expect("the v2 config was not read");
        assert_eq!(meta.config_path, meta::v2_config_label(&path, "config_json"));
        assert_eq!(meta.generation_config_path.as_deref(), Some(meta::v2_config_label(&path, "generation_config_json").as_str()));
        assert_eq!(meta.checks().len(), 21);
        assert!(meta.verify().is_empty());
        assert_eq!(geo, Geo::FLASH_NEXT);
        assert_eq!(geo_for(&path, &peek, None).unwrap(), Geo::FLASH_NEXT);
        let line = index_line(&peek, None);
        assert!(line.starts_with("[boot] container index v2 (family FlashNext, recipe c7-synthetic"), "{line}");
        assert!(line.contains("config from its model block (config.json sha256 889658f2508e"), "{line}");
    }

    /// Phase 1 acceptance: "a container with the other model's config dies at boot with a
    /// named mismatch table". Each config passes its own family row, so the table is what
    /// stops it: the container's facts against the config's, one row per difference.
    #[test]
    fn a_container_with_the_other_models_config_dies_with_a_named_mismatch_table() {
        let refusal = |path: &str, peek: &IndexPeek, dir: Option<&str>| match meta::gate(path, peek, dir) {
            Err(GateRefusal::Refused(why)) => why,
            other => panic!("expected the mismatch table, got {:?}", other.map(|o| o.map(|(_, g)| g.family))),
        };
        // a dense container whose whole model block is Flash-Next's
        let path = v2("fn-block-in-dense", &read(FN, "config.json"), &read(FN, "generation_config.json"), "FlashNext", "qwen4_exp_text", &tensors(5120, 64));
        let why = refusal(&path, &Cnq::peek_index(&path).unwrap(), None);
        assert!(why.starts_with("[meta] 2 of 3 container facts differ from the config"), "{why}");
        assert!(why.contains("  embed_tokens [vocab, hidden]: container [248320, 5120], config [248320, 2560]"), "{why}");
        assert!(why.contains("  text layers: container 64, config 48"), "{why}");
        assert!(why.contains(&format!("[meta]   config: {}", meta::v2_config_label(&path, "config_json"))), "{why}");
        assert_eq!(geo_for(&path, &Cnq::peek_index(&path).unwrap(), None).unwrap_err(), why);
        // the same, with the converter's family record left as it was
        let path = v2("fn-config-in-dense", &read(FN, "config.json"), &read(FN, "generation_config.json"), "Qwen35Dense", "qwen3_5_text", &tensors(5120, 64));
        let why = refusal(&path, &Cnq::peek_index(&path).unwrap(), None);
        assert!(why.starts_with("[meta] 3 of 3 container facts differ"), "{why}");
        assert!(why.contains("  family: container Qwen35Dense (index v2), config FlashNext"), "{why}");
        // the index v1 container of record with the 27B's config through CROW_MODEL_DIR: the
        // table, not the later `Residual::Plain` refusal of a dense Geo
        let v1 = IndexPeek { kind: IndexKind::V1OfRecord, tensors: tensors(2560, 48).iter().map(info).collect() };
        let dense_dir = fixture(DENSE, "");
        let why = refusal("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", &v1, Some(&dense_dir));
        assert!(why.starts_with("[meta] 3 of 3 container facts differ"), "{why}");
        assert!(why.contains("  family: container FlashNext (index v1), config Qwen35Dense"), "{why}");
        assert!(why.contains("  embed_tokens [vocab, hidden]: container [248320, 2560], config [248320, 5120]"), "{why}");
        assert!(why.contains("  text layers: container 48, config 64"), "{why}");
    }

    /// Crow #300 phase 2: CROW_CONTEXT - unset or empty is the floor, a value in floor..=max is
    /// taken, anything else refuses by name
    #[test]
    fn crow_context_takes_a_value_between_the_floor_and_the_max() {
        use crate::boot::context_from_env;
        assert_eq!(context_from_env(None, 100_000, 262_144), Ok(100_000));
        assert_eq!(context_from_env(Some(" "), 100_000, 262_144), Ok(100_000));
        assert_eq!(context_from_env(Some("131072"), 100_000, 262_144), Ok(131_072));
        assert!(context_from_env(Some("99999"), 100_000, 262_144).unwrap_err().contains("outside 100000..=262144"));
        assert!(context_from_env(Some("262145"), 100_000, 262_144).unwrap_err().contains("outside"));
        assert!(context_from_env(Some("128k"), 100_000, 262_144).unwrap_err().contains("not a whole number"));
    }

    /// A dense index v2 container (the 27B's config, 27B-shaped tensors) passes the gate and
    /// `Geo::built` (Crow #300 phase 2 built its arms). `geo_for` is the whole boot door and
    /// touches no CUDA.
    #[test]
    fn a_dense_v2_container_passes_the_boot_door() {
        let path = v2("dense", &read(DENSE, "config.json"), &read(DENSE, "generation_config.json"), "Qwen35Dense", "qwen3_5_text", &tensors(5120, 64));
        let peek = Cnq::peek_index(&path).unwrap();
        let (meta, geo) = meta::gate(&path, &peek, None).unwrap().unwrap();
        assert_eq!((meta.family, geo.hidden, geo.layers), (Family::Qwen35Dense, 5120, 64));
        let g = geo_for(&path, &peek, None).unwrap();
        assert_eq!((g.family, g.residual, g.attn), (Family::Qwen35Dense, crate::geo::Residual::Plain, crate::geo::Attn::Full));
    }

    /// `CROW_MODEL_DIR` beside an index v2 container is a sha-checked cross-check: the same
    /// config passes (and the container's copy is still the one read), another model's config
    /// is refused by name with both hashes.
    #[test]
    fn crow_model_dir_is_a_sha_checked_cross_check_for_a_v2_container() {
        let path = flash_next_v2("override");
        let peek = Cnq::peek_index(&path).unwrap();
        let why = match meta::gate(&path, &peek, Some(&fixture(DENSE, ""))) {
            Err(GateRefusal::Refused(why)) => why,
            other => panic!("expected the CROW_MODEL_DIR refusal, got {:?}", other.map(|o| o.is_some())),
        };
        assert!(why.starts_with("[meta] CROW_MODEL_DIR "), "{why}");
        assert!(why.contains("config.json sha256 191e0af2") && why.contains("model.config_json sha256 889658f2508e"), "{why}");
        let (meta, _) = meta::gate(&path, &peek, Some(&fixture(FN, ""))).unwrap().unwrap();
        assert_eq!(meta.config_path, meta::v2_config_label(&path, "config_json"));
        assert!(index_line(&peek, Some("x")).ends_with(", CROW_MODEL_DIR sha-checked against it"));
        assert!(matches!(meta::gate(&path, &peek, Some("")), Err(GateRefusal::Refused(w)) if w == "CROW_MODEL_DIR is set but empty"));
        assert!(matches!(meta::gate(&path, &peek, Some("/nonexistent-c7")), Err(GateRefusal::Refused(w)) if w.contains("config.json")));
    }

    /// The index v1 container of record keeps the pre-C7 path: the config comes from
    /// `CROW_MODEL_DIR` (the caller's value, not the process environment).
    #[test]
    fn the_v1_container_of_record_reads_crow_model_dir() {
        let v1 = IndexPeek { kind: IndexKind::V1OfRecord, tensors: tensors(2560, 48).iter().map(info).collect() };
        let dir = fixture(FN, "");
        let (meta, geo) = meta::gate("converter/x.cnq", &v1, Some(&dir)).unwrap().expect("CROW_MODEL_DIR was not read");
        assert_eq!(meta.config_path, format!("{dir}config.json"));
        assert_eq!(geo, Geo::FLASH_NEXT);
        let line = index_line(&v1, Some(&dir));
        assert!(line.starts_with("[boot] container index v1 (the Flash-Next CNQ4.5-M container of record, index sha256 a21afc43203d…): config from CROW_MODEL_DIR="), "{line}");
    }
}
