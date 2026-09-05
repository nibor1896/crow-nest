//! Hybrid-container experiment (A-P4, 2026-09-04): is a low-bit tier
//! acceptable when it covers ONLY the rarest experts?
//!
//! The 2- and 3-bit cold tiers (coldtier.rs) failed the trace probe with
//! every cold expert low-bit. Here a COPY of the container gets only the K
//! rarest routed experts per layer (by measured routing counts) re-encoded
//! through the 3-bit codebook + scale ladder and written back as ordinary
//! 36-byte NVFP4 blocks - the engine math is then identical to what a mixed
//! pinned tier would compute, with zero engine changes. The source container
//! is opened read-only and never modified; the copy is a separate file.
//!
//! Monotone extension: re-running with a larger K on the same output patches
//! only the additional experts (a manifest next to the output records what
//! was patched).
//!
//! usage: hybrid <src.cnq> <out.cnq> <K> --counts <routestats.json>... [--levels 0.5,1.5,3,6] [--threads 12]
use crow_nest_engine::cnq::{self, Cnq};
use crow_nest_engine::geo::*;
use std::io::{Read, Seek, SeekFrom, Write};
use std::os::windows::fs::FileExt;

fn mag_index(m: f32) -> u32 {
    for (i, v) in [0.0f32, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0].iter().enumerate() {
        if (*v - m).abs() < 1e-6 {
            return i as u32;
        }
    }
    panic!("codebook magnitude {m} is not an e2m1 level");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: hybrid <src.cnq> <out.cnq> <K> --counts <routestats.json>... [--levels 0.5,1.5,3,6] [--threads 12]");
        std::process::exit(2);
    }
    let src_path = args[1].clone();
    let out_path = args[2].clone();
    let k: usize = args[3].parse().unwrap();
    let mut levels: Vec<f32> = vec![0.5, 1.5, 3.0, 6.0];
    let mut threads = 12usize;
    let mut count_files: Vec<String> = Vec::new();
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--levels" => { levels = args[i + 1].split(',').map(|v| v.parse().unwrap()).collect(); i += 2; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            "--counts" => { count_files.push(args[i + 1].clone()); i += 2; }
            other => panic!("unknown arg {other}"),
        }
    }
    assert!(!count_files.is_empty(), "need at least one --counts file");
    // codebook: symmetric e2m1 subset
    let mut cb_val: Vec<f32> = Vec::new();
    let mut cb_nib: Vec<u32> = Vec::new();
    for &m in &levels { cb_val.push(m); cb_nib.push(mag_index(m)); }
    for &m in &levels { cb_val.push(-m); cb_nib.push(mag_index(m) | 8); }

    // ---- routing counts [48][512], summed over the given files ----
    let mut counts = vec![vec![0u64; E]; LAYERS];
    for cf in &count_files {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(cf).unwrap()).unwrap();
        let arr = if v.get("prefill_counts").is_some() { &v["prefill_counts"] } else { &v };
        let arr = arr.as_array().expect("counts: [48][512] array");
        assert_eq!(arr.len(), LAYERS);
        for l in 0..LAYERS {
            let row = arr[l].as_array().unwrap();
            assert_eq!(row.len(), E);
            for e in 0..E { counts[l][e] += row[e].as_u64().unwrap(); }
        }
    }
    // rarest K per layer: ascending count, ties by ascending id
    let mut rare: Vec<Vec<u32>> = Vec::with_capacity(LAYERS);
    for l in 0..LAYERS {
        let mut ord: Vec<u32> = (0..E as u32).collect();
        ord.sort_by(|&a, &b| counts[l][a as usize].cmp(&counts[l][b as usize]).then(a.cmp(&b)));
        ord.truncate(k);
        rare.push(ord);
    }
    let tot_sel: u64 = counts.iter().map(|c| c.iter().sum::<u64>()).sum();
    let rare_sel: u64 = (0..LAYERS).map(|l| rare[l].iter().map(|&e| counts[l][e as usize]).sum::<u64>()).sum();
    eprintln!("hybrid: K={k} rarest experts per layer carry {:.3} % of the {} counted selections",
        100.0 * rare_sel as f64 / tot_sel.max(1) as f64, tot_sel);

    // ---- output: copy the source once, then patch in place ----
    let manifest_path = format!("{out_path}.hybrid.json");
    let mut already: Vec<Vec<u32>> = vec![Vec::new(); LAYERS];
    if std::path::Path::new(&out_path).exists() {
        let m: serde_json::Value = serde_json::from_slice(&std::fs::read(&manifest_path).expect("output exists but no manifest")).unwrap();
        assert_eq!(m["source"].as_str().unwrap(), src_path, "manifest source differs");
        assert_eq!(m["levels"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap() as f32).collect::<Vec<_>>(), levels, "manifest codebook differs");
        for l in 0..LAYERS {
            already[l] = m["patched"][l].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        }
        eprintln!("hybrid: extending {out_path} ({} experts already patched in layer 0)", already[0].len());
    } else {
        let t0 = std::time::Instant::now();
        let mut fi = std::fs::File::open(&src_path).unwrap();
        let mut fo = std::fs::File::create(&out_path).unwrap();
        let total = fi.metadata().unwrap().len();
        let mut buf = vec![0u8; 64 << 20];
        let mut done = 0u64;
        let mut last = 0u64;
        loop {
            let n = fi.read(&mut buf).unwrap();
            if n == 0 { break; }
            fo.write_all(&buf[..n]).unwrap();
            done += n as u64;
            if done - last >= (8u64 << 30) {
                eprintln!("hybrid: copied {:.1} / {:.1} GB ({:.0} s)", done as f64 / 1e9, total as f64 / 1e9, t0.elapsed().as_secs_f64());
                last = done;
            }
        }
        fo.flush().unwrap();
        drop(fo);
        eprintln!("hybrid: copy done in {:.0} s", t0.elapsed().as_secs_f64());
    }
    for l in 0..LAYERS {
        for e in &already[l] {
            assert!(rare[l].contains(e), "layer {l}: expert {e} was patched before but is not among the K={k} rarest now - not monotone, refusing");
        }
    }

    let idx = Cnq::open(&src_path);
    let blob = idx.blob_offset;
    let mut layers = Vec::new();
    for l in 0..LAYERS {
        let gu = idx.find(&format!("model.language_model.layers.{l}.mlp.experts.gate_up_proj"), "text").clone();
        let dn = idx.find(&format!("model.language_model.layers.{l}.mlp.experts.down_proj"), "text").clone();
        layers.push((blob + gu.offset, gu.global_scale, blob + dn.offset, dn.global_scale));
    }
    let gu_blocks = (2 * INTER * H) / 64;
    let dn_blocks = (H * INTER) / 64;

    // ---- worker: patch one layer's new rare experts in the output file ----
    let work = |l: usize| -> (f64, f64, u64, usize) {
        let (gu_off, gu_gs, dn_off, dn_gs) = layers[l];
        let mut fi = std::fs::File::open(&src_path).unwrap();
        let fo = std::fs::OpenOptions::new().write(true).open(&out_path).unwrap();
        let mut sse_new = 0f64;
        let mut sse_ref = 0f64;
        let mut n_vals = 0u64;
        let mut n_patched = 0usize;
        let mut blk = [0f32; 64];
        for &e in &rare[l] {
            if already[l].contains(&e) { continue; }
            for (off, gs, nblk) in [(gu_off, gu_gs, gu_blocks), (dn_off, dn_gs, dn_blocks)] {
                let per_expert = nblk * 36;
                let mut raw = vec![0u8; per_expert];
                fi.seek(SeekFrom::Start(off + (e as usize * per_expert) as u64)).unwrap();
                fi.read_exact(&mut raw).unwrap();
                let mut out = raw.clone();
                for b in 0..nblk {
                    let src = &raw[b * 36..(b + 1) * 36];
                    cnq::dequant_block(src, gs, &mut blk);
                    let dst = &mut out[b * 36..(b + 1) * 36];
                    for sb in 0..4 {
                        let vals = &blk[sb * 16..(sb + 1) * 16];
                        let orig = src[sb] as i32;
                        let mut best = (f64::INFINITY, orig as u32, [0u32; 16]);
                        for d in -4i32..=4 {
                            let byte = orig + d;
                            if byte < 1 || byte > 0x7E { continue; }
                            let s = cnq::ue4m3(byte as u32) * gs;
                            let mut sse = 0f64;
                            let mut cd = [0u32; 16];
                            for (j, &v) in vals.iter().enumerate() {
                                let mut bk = 0usize;
                                let mut be = f32::INFINITY;
                                for (kk, &c) in cb_val.iter().enumerate() {
                                    let err = (v - c * s).abs();
                                    if err < be { be = err; bk = kk; }
                                }
                                cd[j] = bk as u32;
                                sse += (be as f64) * (be as f64);
                            }
                            if sse < best.0 { best = (sse, byte as u32, cd); }
                        }
                        dst[sb] = best.1 as u8;
                        // expanded block: codebook nibbles, LSB-first pairs
                        for j in 0..16 {
                            let idx = sb * 16 + j;
                            let nib = cb_nib[best.2[j] as usize] as u8;
                            let bi = 4 + (idx >> 1);
                            if idx & 1 == 1 { dst[bi] = (dst[bi] & 0x0F) | (nib << 4); } else { dst[bi] = (dst[bi] & 0xF0) | nib; }
                        }
                        sse_new += best.0;
                        for &v in vals { sse_ref += (v as f64) * (v as f64); }
                        n_vals += 16;
                    }
                }
                fo.seek_write(&out, off + (e as usize * per_expert) as u64).unwrap();
            }
            n_patched += 1;
        }
        (sse_new, sse_ref, n_vals, n_patched)
    };

    let t0 = std::time::Instant::now();
    let mut tot_new = 0f64;
    let mut tot_ref = 0f64;
    let mut tot_n = 0u64;
    let mut tot_p = 0usize;
    let mut next = 0usize;
    while next < LAYERS {
        let batch: Vec<usize> = (next..(next + threads).min(LAYERS)).collect();
        let results: Vec<(usize, (f64, f64, u64, usize))> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch.iter().map(|&l| { let w = &work; sc.spawn(move || (l, w(l))) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (l, (sn, sr, n, p)) in results {
            tot_new += sn; tot_ref += sr; tot_n += n; tot_p += p;
            eprintln!("hybrid: layer {l:2} patched {p} experts ({:.0} s), rel_rms so far {:.4}", t0.elapsed().as_secs_f64(), (tot_new / tot_ref.max(1e-30)).sqrt());
        }
        next += threads;
    }
    let manifest = serde_json::json!({
        "format": "crow-nest-hybrid", "version": 1, "source": src_path, "k": k, "levels": levels,
        "codebook_nibbles": cb_nib, "counts_files": count_files,
        "rare_share_percent": 100.0 * rare_sel as f64 / tot_sel.max(1) as f64,
        "patched": rare,
    });
    std::fs::write(&manifest_path, serde_json::to_string_pretty(&manifest).unwrap()).unwrap();
    eprintln!("hybrid: {tot_p} newly patched experts, relative RMS error vs FP4 {:.4} over {} values, manifest {manifest_path}",
        (tot_new / tot_ref.max(1e-30)).sqrt(), tot_n);
}
