//! `converter requant-check <dense.safetensors> <out.cnq>` — #76, THE PROOF.
//!
//! The fetch tool (`tools/fetch-dense-originals.py`) pulls the BF16 originals of the dense
//! text path back out of the Hugging Face repo by HTTP range. Nothing about a download is
//! self-evident: the range arithmetic could be off, a shard could have been re-uploaded, the
//! revision could be the wrong one. This subcommand settles it with the only evidence this
//! repository accepts — the bytes it already ships.
//!
//! For every tensor in the fetched file it runs the SAME `quantize_nvfp4` the conversion ran
//! (not a copy of it, the function itself) and compares the 36-byte NVFP4 blocks and the f32
//! global scale against what `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` stores under that name. Byte
//! for byte identical means two things at once: the download is the original, and the
//! converter is CHARACTERIZED — that code path, on that input, still emits what it emitted on
//! 2026-09-18, which is what step 3 of the requant series will be measured against.
//!
//! It writes nothing. It reads the container and the fetched file and exits non-zero on the
//! first sign that they disagree.
//!
//! The scale mode defaults to `mse`, the mode CNQ4.5-M was built with (`docs/model-card.md`,
//! Provenance); `--scales ceil` before the subcommand overrides it.

use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use crate::{bytes_to_f32, elem_size_of, quantize_nvfp4, read_safetensors_header, ScalesMode, MAGIC};

pub const HELP: &str = "usage: converter [--scales ceil|mse] requant-check <originals.safetensors> <container.cnq> [--experts] [--threads N] [--limit N]\n  re-quantizes every tensor of the fetched file and compares it with the container's own bytes\n  --experts  the fetched file holds ROUTED EXPERTS of one or more layers (#79) instead of the dense path";

/// One tensor as the container's index trailer describes it.
struct IndexEntry {
    dtype: String,
    section: String,
    name: String,
    offset: u64,
    len: u64,
    n_values: usize,
    global_scale: Option<f32>,
}

/// The container's JSON index: it is a TRAILER — the last 8 bytes are its u64 length.
fn read_container_index(path: &Path) -> std::io::Result<Vec<IndexEntry>> {
    let mut f = std::fs::File::open(path)?;
    let mut magic = [0u8; 4];
    f.read_exact(&mut magic)?;
    if &magic != MAGIC {
        return Err(std::io::Error::other(format!(
            "{}: magic is {magic:?}, not CNQ1",
            path.display()
        )));
    }
    let trailer = f.seek(SeekFrom::End(-8))?; // = file size - 8, where the u64 length sits
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf)?;
    let index_len = u64::from_le_bytes(len_buf);
    f.seek(SeekFrom::Start(trailer - index_len))?;
    let mut buf = vec![0u8; index_len as usize];
    f.read_exact(&mut buf)?;
    let index: serde_json::Value = serde_json::from_slice(&buf)?;
    let blob_offset = index["blob_offset"].as_u64().unwrap_or(12);
    if blob_offset != 12 {
        return Err(std::io::Error::other(format!(
            "{}: blob_offset is {blob_offset}, this reader knows 12",
            path.display()
        )));
    }
    let mut out = Vec::new();
    for t in index["tensors"].as_array().into_iter().flatten() {
        out.push(IndexEntry {
            dtype: t["dtype"].as_str().unwrap_or("").to_string(),
            section: t["section"].as_str().unwrap_or("").to_string(),
            name: t["name"].as_str().unwrap_or("").to_string(),
            offset: t["offset"].as_u64().unwrap_or(0),
            len: t["len"].as_u64().unwrap_or(0),
            n_values: t["n_values"].as_u64().unwrap_or(0) as usize,
            // serde_json keeps an f32 as the f64 it widens to exactly, and the shortest
            // round-tripping decimal of that f64 narrows back to the same f32
            global_scale: t["global_scale"].as_f64().map(|v| v as f32),
        });
    }
    Ok(out)
}

/// The #76 selection rule, derived a SECOND time — here from the container's index rather
/// than from the sidecar the Python tool reads. Two independent derivations of one list.
fn is_dense_text(e: &IndexEntry) -> bool {
    e.section == "text" && e.dtype == "nvfp4" && !e.name.contains(".mlp.experts.")
}

/// #79: the same rule with the sign flipped — the routed experts, under `--experts`.
fn is_expert_text(e: &IndexEntry) -> bool {
    e.section == "text" && e.dtype == "nvfp4" && e.name.contains(".mlp.experts.")
}

/// the layer a tensor name belongs to, so `--experts` can narrow the coverage rule to the
/// layers the fetched file actually carries
fn layer_of(name: &str) -> Option<usize> {
    let rest = name.strip_prefix("model.language_model.layers.")?;
    let dot = rest.find('.')?;
    rest[..dot].parse::<usize>().ok()
}

/// What one tensor's comparison found.
struct Verdict {
    order: usize,
    name: String,
    n_values: usize,
    blocks: usize,
    diff_blocks: usize,
    first_diff_block: Option<usize>,
    first_diff_kind: &'static str,
    scale_ok: bool,
    got_scale: f32,
    want_scale: f32,
    note: Option<String>,
}

impl Verdict {
    fn identical(&self) -> bool {
        self.diff_blocks == 0 && self.scale_ok && self.note.is_none()
    }
}

/// Compare two block streams: (differing blocks, first differing block, what differs in it).
/// A block is 36 B — 4 ue4m3 sub-block scale bytes then 32 B of packed E2M1 nibbles — so a
/// disagreement can be named: the scale choice, or the values under it.
fn compare_blocks(got: &[u8], want: &[u8]) -> (usize, Option<usize>, &'static str) {
    let n = got.len() / 36;
    let mut diff = 0usize;
    let mut first = None;
    let mut kind = "";
    for b in 0..n {
        let g = &got[b * 36..b * 36 + 36];
        let w = &want[b * 36..b * 36 + 36];
        if g != w {
            diff += 1;
            if first.is_none() {
                first = Some(b);
                kind = match (g[..4] != w[..4], g[4..] != w[4..]) {
                    (true, true) => "scales+nibbles",
                    (true, false) => "scales",
                    _ => "nibbles",
                };
            }
        }
    }
    (diff, first, kind)
}

pub fn run(args: &[String], mode: ScalesMode, mode_explicit: bool) -> i32 {
    let mut positional: Vec<&str> = Vec::new();
    let mut threads: usize = 0;
    let mut limit: usize = 0;
    let mut experts = false;
    let mut it = args.iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--threads" => threads = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--limit" => limit = it.next().and_then(|v| v.parse().ok()).unwrap_or(0),
            "--experts" => experts = true,
            s if s.starts_with("--") => {
                eprintln!("unknown flag {s}\n{HELP}");
                return 2;
            }
            s => positional.push(s),
        }
    }
    if positional.len() != 2 {
        eprintln!("{HELP}");
        return 2;
    }
    let dense_path = Path::new(positional[0]);
    let cnq_path = Path::new(positional[1]);
    let mode = if mode_explicit { mode } else { ScalesMode::Mse };
    let mode_str = if mode == ScalesMode::Mse { "mse" } else { "ceil" };

    let index = match read_container_index(cnq_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("container index: {e}");
            return 2;
        }
    };
    let (header, data_start) = match read_safetensors_header(dense_path) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("{}: {e}", dense_path.display());
            return 2;
        }
    };

    // the tensors the fetched file carries, in its own (name) order
    let mut fetched: Vec<(String, String, u64, u64, usize)> = Vec::new(); // name, dtype, begin, end, n
    for (name, info) in header.as_object().into_iter().flatten() {
        if name == "__metadata__" {
            continue;
        }
        let dt = info["dtype"].as_str().unwrap_or("").to_string();
        let shape: Vec<usize> = info["shape"]
            .as_array()
            .map(|a| a.iter().map(|v| v.as_u64().unwrap_or(0) as usize).collect())
            .unwrap_or_default();
        let n: usize = shape.iter().product();
        let begin = info["data_offsets"][0].as_u64().unwrap_or(0);
        let end = info["data_offsets"][1].as_u64().unwrap_or(0);
        fetched.push((name.clone(), dt, data_start + begin, data_start + end, n));
    }
    fetched.sort_by(|a, b| a.0.cmp(&b.0));
    if limit > 0 {
        fetched.truncate(limit);
    }

    // #79: `--experts` proves the FETCHED ROUTED EXPERTS the same way, and against the same
    // container. The fetched file then holds one layer, so the coverage rule below is narrowed
    // to the layers it carries - everything else about the check is the same code.
    let fetched_layers: std::collections::BTreeSet<Option<usize>> =
        fetched.iter().map(|t| layer_of(&t.0)).collect();
    let want_names: Vec<&IndexEntry> = if experts {
        index
            .iter()
            .filter(|e| is_expert_text(e) && fetched_layers.contains(&layer_of(&e.name)))
            .collect()
    } else {
        index.iter().filter(|e| is_dense_text(e)).collect()
    };
    println!(
        "container {}: {} tensors, {} of them {} nvfp4",
        cnq_path.display(),
        index.len(),
        want_names.len(),
        if experts { "routed-expert text (in the fetched layers)" } else { "dense text" }
    );
    println!(
        "fetched   {}: {} tensors, {} values, scales {mode_str}",
        dense_path.display(),
        fetched.len(),
        fetched.iter().map(|t| t.4).sum::<usize>()
    );

    // coverage, derived from the container itself: nothing the rule names may be absent
    let have: std::collections::BTreeSet<&str> = fetched.iter().map(|t| t.0.as_str()).collect();
    let missing: Vec<&str> = want_names
        .iter()
        .map(|e| e.name.as_str())
        .filter(|n| !have.contains(n))
        .collect();
    if limit == 0 && !missing.is_empty() {
        println!("MISSING {} selected tensors from the fetched file:", missing.len());
        for n in missing.iter().take(10) {
            println!("  {n}");
        }
        return 1;
    }

    let by_name: std::collections::BTreeMap<&str, &IndexEntry> =
        index.iter().map(|e| (e.name.as_str(), e)).collect();

    // One dense tensor is at most 500 MB of f32; one routed-expert tensor is 6.7 GB of f32
    // plus its blocks and the container's, about 10 GB per worker - so `--experts` defaults to
    // two workers instead of eight and `--threads` overrides both.
    let n_threads = if threads > 0 {
        threads
    } else if experts {
        2
    } else {
        std::thread::available_parallelism().map(|v| v.get()).unwrap_or(4).min(8)
    };
    let next = AtomicUsize::new(0);
    let verdicts: Mutex<Vec<Verdict>> = Mutex::new(Vec::with_capacity(fetched.len()));
    let t0 = std::time::Instant::now();
    let done = AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..n_threads {
            scope.spawn(|| {
                let mut dense_f = std::fs::File::open(dense_path).expect("open dense file");
                let mut cnq_f = std::fs::File::open(cnq_path).expect("open container");
                loop {
                    let i = next.fetch_add(1, Ordering::SeqCst);
                    if i >= fetched.len() {
                        break;
                    }
                    let (name, dt, begin, end, n) = &fetched[i];
                    let mut v = Verdict {
                        order: i,
                        name: name.clone(),
                        n_values: *n,
                        blocks: 0,
                        diff_blocks: 0,
                        first_diff_block: None,
                        first_diff_kind: "",
                        scale_ok: false,
                        got_scale: 0.0,
                        want_scale: 0.0,
                        note: None,
                    };
                    let entry = by_name.get(name.as_str());
                    let Some(entry) = entry else {
                        v.note = Some("not in the container index".into());
                        verdicts.lock().unwrap().push(v);
                        continue;
                    };
                    if entry.dtype != "nvfp4" {
                        v.note = Some(format!("the container stores it as {}", entry.dtype));
                        verdicts.lock().unwrap().push(v);
                        continue;
                    }
                    if entry.n_values != *n {
                        v.note = Some(format!(
                            "{} values fetched, the container says {}",
                            n, entry.n_values
                        ));
                        verdicts.lock().unwrap().push(v);
                        continue;
                    }
                    if elem_size_of(dt).is_none() || dt == "I64" {
                        // `bytes_to_f32` knows F32, BF16 and F16; I64 is integer metadata
                        v.note = Some(format!("dtype {dt} is not a float this reader converts"));
                        verdicts.lock().unwrap().push(v);
                        continue;
                    }
                    if n % 64 != 0 {
                        v.note = Some(format!("{n} values is not a multiple of 64"));
                        verdicts.lock().unwrap().push(v);
                        continue;
                    }

                    let mut raw = vec![0u8; (end - begin) as usize];
                    dense_f.seek(SeekFrom::Start(*begin)).expect("seek dense");
                    dense_f.read_exact(&mut raw).expect("read dense tensor");
                    let values = bytes_to_f32(&raw, dt);
                    drop(raw);
                    // the conversion's own function, the conversion's own mode
                    let (blocks, global, _stats, _sse) = quantize_nvfp4(&values, mode);
                    drop(values);

                    v.blocks = blocks.len() / 36;
                    v.got_scale = global;
                    v.want_scale = entry.global_scale.unwrap_or(f32::NAN);
                    v.scale_ok = entry.global_scale.map(|s| s.to_bits() == global.to_bits()).unwrap_or(false);

                    if blocks.len() as u64 != entry.len {
                        v.note = Some(format!(
                            "{} B produced, the container stores {} B",
                            blocks.len(),
                            entry.len
                        ));
                        verdicts.lock().unwrap().push(v);
                        continue;
                    }
                    let mut want = vec![0u8; entry.len as usize];
                    cnq_f.seek(SeekFrom::Start(12 + entry.offset)).expect("seek container");
                    cnq_f.read_exact(&mut want).expect("read container tensor");
                    let (diff, first, kind) = compare_blocks(&blocks, &want);
                    v.diff_blocks = diff;
                    v.first_diff_block = first;
                    v.first_diff_kind = kind;
                    verdicts.lock().unwrap().push(v);

                    let d = done.fetch_add(1, Ordering::SeqCst) + 1;
                    if d.is_multiple_of(50) {
                        eprintln!(
                            "  [{d}/{}] {:.0} s",
                            fetched.len(),
                            t0.elapsed().as_secs_f64()
                        );
                    }
                }
            });
        }
    });

    let mut verdicts = verdicts.into_inner().unwrap();
    verdicts.sort_by_key(|v| v.order);

    let mut identical = 0usize;
    for v in &verdicts {
        if v.identical() {
            identical += 1;
            println!(
                "OK    {}  {} values, {} blocks, global {:e}",
                v.name, v.n_values, v.blocks, v.got_scale
            );
        } else if let Some(note) = &v.note {
            println!("DIFF  {}  {}", v.name, note);
        } else {
            println!(
                "DIFF  {}  {} of {} blocks differ, first block {} ({}), global {} ({:e} vs {:e})",
                v.name,
                v.diff_blocks,
                v.blocks,
                v.first_diff_block.map(|b| b as i64).unwrap_or(-1),
                v.first_diff_kind,
                if v.scale_ok { "equal" } else { "DIFFERENT" },
                v.got_scale,
                v.want_scale
            );
        }
    }

    let values: usize = verdicts.iter().map(|v| v.n_values).sum();
    println!(
        "requant-check: {identical} of {} tensors byte-identical including the global scale \
({} values, scales {mode_str}, {} threads, {:.0} s)",
        verdicts.len(),
        values,
        n_threads,
        t0.elapsed().as_secs_f64()
    );
    if identical == verdicts.len() && !verdicts.is_empty() {
        0
    } else {
        1
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_stream_that_matches_reports_no_difference() {
        let a: Vec<u8> = (0..72u32).map(|x| x as u8).collect();
        assert_eq!(compare_blocks(&a, &a), (0, None, ""));
    }

    #[test]
    fn the_first_differing_block_is_named_and_the_rest_counted() {
        let a: Vec<u8> = (0..36 * 4u32).map(|x| x as u8).collect();
        let mut b = a.clone();
        b[36 + 10] ^= 0xFF; // block 1, a nibble byte
        b[36 * 3 + 2] ^= 0xFF; // block 3, a scale byte
        let (diff, first, kind) = compare_blocks(&a, &b);
        assert_eq!((diff, first), (2, Some(1)));
        assert_eq!(kind, "nibbles");
    }

    #[test]
    fn a_scale_byte_and_a_nibble_byte_are_told_apart() {
        let a = vec![7u8; 36];
        let mut b = a.clone();
        b[1] = 8;
        assert_eq!(compare_blocks(&a, &b).2, "scales");
        let mut c = a.clone();
        c[0] = 8;
        c[35] = 9;
        assert_eq!(compare_blocks(&a, &c).2, "scales+nibbles");
    }

    #[test]
    fn the_selection_rule_is_the_one_the_fetch_tool_derives() {
        let mk = |name: &str, section: &str, dtype: &str| IndexEntry {
            dtype: dtype.into(),
            section: section.into(),
            name: name.into(),
            offset: 0,
            len: 0,
            n_values: 0,
            global_scale: None,
        };
        assert!(is_dense_text(&mk("m.layers.0.linear_attn.out_proj.weight", "text", "nvfp4")));
        assert!(!is_dense_text(&mk("m.layers.0.mlp.experts.down_proj", "text", "nvfp4")));
        assert!(!is_dense_text(&mk("m.layers.0.input_layernorm.weight", "text", "bf16")));
        assert!(!is_dense_text(&mk("m.visual.blocks.0.attn.qkv.weight", "vit", "nvfp4")));
        assert!(!is_dense_text(&mk("m.layers.1.ple.ple_embedding.ngram_embedding.shard_0.weight", "ple", "nvfp4")));
    }

    #[test]
    fn an_f32_global_scale_survives_the_index_json_exactly() {
        // the comparison is on the bits, so this is the property it rests on: an f32 written
        // into the index as JSON and read back as f64 narrows to the SAME f32
        for bits in [0x3f80_0000u32, 0x3456_789au32, 0x0000_0001u32, 0x7f7f_ffffu32, 0x3727_c5acu32] {
            let s = f32::from_bits(bits);
            let json = serde_json::to_string(&serde_json::Value::from(s)).unwrap();
            let back: serde_json::Value = serde_json::from_str(&json).unwrap();
            assert_eq!(back.as_f64().unwrap() as f32, s, "round trip broke for {s:e}");
        }
    }
}
