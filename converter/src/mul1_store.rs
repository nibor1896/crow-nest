//! crow-nest #182 (GLM-5.3-Flash plan step 12): the MUL1 store the conversion reads its routed-
//! expert records from (`converter --experts-mul1 <store>`).
//!
//! The store is written by `tools/glm_mul1_quantize.py quantize` (exllamav3's `quantize_exl3`, mul1
//! codebook, K = 3, Hessians from the layerwise runner's MoE inputs):
//!
//! ```text
//! <store>/store.json            {"format": "crow-nest mul1 store", "version": 1, "k": 3, "hidden", "inter",
//!                                "record_bytes", "quantizer", "calibration", ...}
//! <store>/journal.jsonl         one line per finished expert, appended after its file is synced:
//!                                {"layer", "expert", "file", "sha256", ...}
//! <store>/L<ll>/E<eee>.safetensors   gate/up/down `.trellis` (I16 [k/16, n/16, 16 K]), `.suh` (F16 [k]),
//!                                `.svh` (F16 [n]); gate/up [hidden, inter], down [inter, hidden]
//! <store>/L<ll>/E<eee>.done     written by the converter once the record is journalled in its container
//! ```
//!
//! The record bytes are built here, by `mul1::write_record`, so the layout lives in one place. A
//! file whose sha256 differs from its journal line, a tensor of another shape or dtype, or a
//! store whose record size is not `RecordLayout::size` of its shapes is refused by name. The store
//! never deletes anything; `tools/glm_mul1_quantize.py quantize --prune-consumed` removes record
//! files that carry a `.done` (derived bytes that are in the container by then).

use crate::mul1::{Bitrate, Linear, RecordLayout};
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub const STORE_FILE: &str = "store.json";
pub const JOURNAL_FILE: &str = "journal.jsonl";
pub const FORMAT: &str = "crow-nest mul1 store";
/// The bitrate of every routed expert of the 3-bit GLM container (robin, 2026-10-08; #181).
pub const K: f64 = 3.0;

/// `L<ll>/E<eee>.safetensors`, relative to the store.
pub fn rel_path(layer: u64, expert: u64) -> String {
    format!("L{layer:02}/E{expert:03}.safetensors")
}

pub struct Store {
    pub dir: PathBuf,
    pub layout: RecordLayout,
    /// `store.json` as parsed, and the sha256 of its bytes (the conversion journal names it)
    pub head: serde_json::Value,
    pub head_sha256: String,
    /// (layer, expert) -> (file relative to the store, sha256 of the file)
    recs: BTreeMap<(u64, u64), (String, String)>,
}

impl Store {
    /// Open a store for experts of `hidden` x `inter` (the config's `hidden_size` and
    /// `moe_intermediate_size`) at K = 3.
    pub fn open(dir: &Path, hidden: usize, inter: usize) -> Result<Store, String> {
        let p = dir.join(STORE_FILE);
        let raw = std::fs::read(&p).map_err(|e| format!("{}: {e} (tools/glm_mul1_quantize.py quantize writes it)", p.display()))?;
        let head: serde_json::Value = serde_json::from_slice(&raw).map_err(|e| format!("{}: {e}", p.display()))?;
        if head["format"] != FORMAT || head["version"] != 1 {
            return Err(format!("{}: format {} version {}, expected \"{FORMAT}\" version 1", p.display(), head["format"], head["version"]));
        }
        if head["k"].as_f64() != Some(K) {
            return Err(format!("{}: K {}, the 3-bit container takes K = {K} only", p.display(), head["k"]));
        }
        if head["hidden"].as_u64() != Some(hidden as u64) || head["inter"].as_u64() != Some(inter as u64) {
            return Err(format!(
                "{}: experts [hidden {}, inter {}], config.json says [hidden {hidden}, inter {inter}] - a store of another model",
                p.display(),
                head["hidden"],
                head["inter"]
            ));
        }
        let layout = RecordLayout::new(hidden, inter, Bitrate::from_k(K)?)?;
        if head["record_bytes"].as_u64() != Some(layout.size) {
            return Err(format!(
                "{}: record_bytes {}, a MUL1 K = 3 record of [hidden {hidden}, inter {inter}] is {} B (mul1::RecordLayout)",
                p.display(),
                head["record_bytes"],
                layout.size
            ));
        }
        let mut s = Store { dir: dir.to_path_buf(), layout, head, head_sha256: crate::recipe::sha256_hex(&raw), recs: BTreeMap::new() };
        s.refresh()?;
        Ok(s)
    }

    /// Re-read the journal (append-only); a torn last line (a writer killed mid-line) is ignored.
    pub fn refresh(&mut self) -> Result<(), String> {
        let p = self.dir.join(JOURNAL_FILE);
        let text = match std::fs::read(&p) {
            Ok(t) => t,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(()),
            Err(e) => return Err(format!("{}: {e}", p.display())),
        };
        for line in text.split(|b| *b == b'\n') {
            let Ok(v) = serde_json::from_slice::<serde_json::Value>(line) else { continue };
            if let (Some(l), Some(e), Some(f), Some(sha)) = (v["layer"].as_u64(), v["expert"].as_u64(), v["file"].as_str(), v["sha256"].as_str()) {
                self.recs.insert((l, e), (f.to_string(), sha.to_string()));
            }
        }
        Ok(())
    }

    pub fn has(&self, layer: u64, expert: u64) -> bool {
        self.recs.contains_key(&(layer, expert))
    }

    /// The record bytes of one expert (`layout.size`), from its file, checked against the journal.
    pub fn record(&self, layer: u64, expert: u64) -> Result<Vec<u8>, String> {
        let (f, want) = self.recs.get(&(layer, expert)).ok_or(format!("MUL1 store {}: no record for layer {layer} expert {expert} in {JOURNAL_FILE}", self.dir.display()))?;
        let p = self.dir.join(f);
        let raw = std::fs::read(&p).map_err(|e| format!("{}: {e} (journalled, so the file was removed after the quantizer wrote it)", p.display()))?;
        let got = crate::recipe::sha256_hex(&raw);
        if &got != want {
            return Err(format!("{}: sha256 {got}, the store journal recorded {want} - not converting this record", p.display()));
        }
        let mats = parse_expert(&raw, &self.layout).map_err(|e| format!("{}: {e}", p.display()))?;
        crate::mul1::write_record(&self.layout, [&mats[0], &mats[1], &mats[2]])
    }

    /// `<store>/L<ll>/E<eee>.done`: the record is in the container `out` and journalled there.
    pub fn mark_done(&self, layer: u64, expert: u64, out: &Path) -> Result<(), String> {
        let p = self.dir.join(rel_path(layer, expert)).with_extension("done");
        let body = serde_json::json!({ "layer": layer, "expert": expert, "out": out.display().to_string() });
        std::fs::write(&p, serde_json::to_vec(&body).unwrap()).map_err(|e| format!("{}: {e}", p.display()))
    }
}

/// gate, up, down of one expert file: safetensors with `<proj>.trellis` (I16), `<proj>.suh`,
/// `<proj>.svh` (F16) at the layout's shapes; any other tensor, dtype or shape is refused.
pub fn parse_expert(raw: &[u8], layout: &RecordLayout) -> Result<[Linear; 3], String> {
    if raw.len() < 8 {
        return Err("not a safetensors file (shorter than 8 B)".into());
    }
    let hlen = u64::from_le_bytes(raw[..8].try_into().unwrap()) as usize;
    if 8 + hlen > raw.len() {
        return Err(format!("safetensors header of {hlen} B does not fit a {} B file", raw.len()));
    }
    let header: serde_json::Value = serde_json::from_slice(&raw[8..8 + hlen]).map_err(|e| format!("safetensors header: {e}"))?;
    let data = &raw[8 + hlen..];
    let names = header.as_object().map(|o| o.keys().filter(|k| *k != "__metadata__").count()).unwrap_or(0);
    if names != 9 {
        return Err(format!("{names} tensors, an expert file holds 9 (gate/up/down .trellis .suh .svh)"));
    }
    let wpt = layout.bitrate.words_per_tile();
    let get = |name: &str, dtype: &str, shape: Vec<usize>| -> Result<Vec<u16>, String> {
        let t = &header[name];
        if t.is_null() {
            return Err(format!("no tensor {name}"));
        }
        let got: Vec<usize> = t["shape"].as_array().map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect()).unwrap_or_default();
        if t["dtype"] != dtype || got != shape {
            return Err(format!("{name}: {} {got:?}, expected {dtype} {shape:?} (MUL1 K = {} for [hidden {}, inter {}])", t["dtype"], layout.bitrate.k(), layout.hidden, layout.inter));
        }
        let (Some(a), Some(b)) = (t["data_offsets"][0].as_u64(), t["data_offsets"][1].as_u64()) else {
            return Err(format!("{name}: no data_offsets"));
        };
        let n: usize = shape.iter().product();
        if b < a || (b - a) as usize != 2 * n || b as usize > data.len() {
            return Err(format!("{name}: data_offsets [{a}, {b}) do not hold {n} two-byte values"));
        }
        Ok(data[a as usize..b as usize].chunks_exact(2).map(|c| u16::from_le_bytes([c[0], c[1]])).collect())
    };
    let mut mats = Vec::with_capacity(3);
    for (p, (k, n)) in ["gate", "up", "down"].iter().zip(layout.shapes()) {
        mats.push(Linear {
            k,
            n,
            trellis: get(&format!("{p}.trellis"), "I16", vec![k / 16, n / 16, wpt])?,
            suh: get(&format!("{p}.suh"), "F16", vec![k])?,
            svh: get(&format!("{p}.svh"), "F16", vec![n])?,
        });
    }
    let [g, u, d]: [Linear; 3] = mats.try_into().map_err(|_| "three linears".to_string())?;
    Ok([g, u, d])
}

/// Free bytes for the caller on the volume of `dir` (the disk check of a MUL1 conversion).
#[cfg(windows)]
pub fn free_bytes(dir: &Path) -> Result<u64, String> {
    use std::os::windows::ffi::OsStrExt;
    #[link(name = "kernel32")]
    extern "system" {
        fn GetDiskFreeSpaceExW(dir: *const u16, avail: *mut u64, total: *mut u64, free: *mut u64) -> i32;
    }
    let w: Vec<u16> = dir.as_os_str().encode_wide().chain(std::iter::once(0)).collect();
    let (mut avail, mut total, mut free) = (0u64, 0u64, 0u64);
    // SAFETY: a NUL-terminated wide path and three valid out pointers, as the API documents
    let ok = unsafe { GetDiskFreeSpaceExW(w.as_ptr(), &mut avail, &mut total, &mut free) };
    if ok == 0 {
        return Err(format!("{}: GetDiskFreeSpaceExW failed: {}", dir.display(), std::io::Error::last_os_error()));
    }
    Ok(avail)
}

/// Free bytes for the caller on the volume of `dir`: POSIX `df -Pk` (available 1024-blocks).
#[cfg(not(windows))]
pub fn free_bytes(dir: &Path) -> Result<u64, String> {
    let out = std::process::Command::new("df").arg("-Pk").arg(dir).output().map_err(|e| format!("df -Pk {}: {e}", dir.display()))?;
    let text = String::from_utf8_lossy(&out.stdout);
    let avail = text.lines().nth(1).and_then(|l| l.split_whitespace().nth(3)).and_then(|s| s.parse::<u64>().ok());
    avail.map(|k| k * 1024).ok_or(format!("df -Pk {}: no available-blocks column in {text:?}", dir.display()))
}
