//! Low-bit COLD tier builder (proposal A-P2, 2026-09-04).
//!
//! Re-encodes every routed expert of the container into a compact record that
//! the staging kernels expand back into the 36-byte NVFP4 block layout in VRAM.
//! The container itself is never touched: the output is a separate file next
//! to it (`<cnq>.cold<bits>.bin`), consumed via `CROW_COLD_TIER=<file>`.
//!
//! Record per 64-value block: 4 ue4m3 sub-block scale bytes (re-optimized on
//! the same ladder) + 64 codes of `bits` bits (LSB-first, packed) = 4 + 8*bits
//! bytes (2-bit: 20 B = 2.5 bpw, 3-bit: 28 B = 3.5 bpw). Codes index a
//! symmetric codebook of e2m1 NIBBLES, so the expanded block is a valid NVFP4
//! block and every downstream kernel stays unchanged.
//!
//! Per 16-value sub-block the tool searches the scale ladder around the
//! container's scale (delta -4..+4 steps) for the minimum SSE against the
//! container's dequantized FP4 values (the only reference available here) and
//! codes each value to the nearest codebook level.
//!
//! usage: coldtier <cnq> <out.bin> [--bits 2|3] [--levels 1,3] [--threads N]
use crow_nest_engine::cnq::{self, Cnq};
use crow_nest_engine::geo::*;
use std::io::{Read, Seek, SeekFrom, Write};

/// nibble of a signed e2m1 level given its magnitude index
fn nib(mag_idx: u32, neg: bool) -> u32 {
    mag_idx | if neg { 8 } else { 0 }
}

fn main() {
    // #13: the logging subscriber of this process. Every library line this bin
    // triggers (`[prefill]`, `[load]`, `[budget]`, `[ple]`, ...) is a `tracing`
    // event now, so without this call they go nowhere. The guard drains the two
    // writer threads when `main` returns; an `exit` below calls `shutdown` first.
    let _log = crow_nest_engine::log::init();
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: coldtier <cnq> <out.bin> [--bits 2|3] [--levels 1,3] [--threads N]");
        crow_nest_engine::log::shutdown();
        std::process::exit(2);
    }
    let cnq_path = args[1].clone();
    let out_path = args[2].clone();
    let mut bits = 2u32;
    let mut levels: Vec<f32> = vec![1.0, 3.0];
    let mut threads = 12usize;
    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--bits" => { bits = args[i + 1].parse().unwrap(); i += 2; }
            "--levels" => { levels = args[i + 1].split(',').map(|v| v.parse().unwrap()).collect(); i += 2; }
            "--threads" => { threads = args[i + 1].parse().unwrap(); i += 2; }
            other => panic!("unknown arg {other}"),
        }
    }
    let n_codes = 1usize << bits;
    assert_eq!(levels.len() * 2, n_codes, "codebook needs {} magnitudes for {bits} bits", n_codes / 2);
    // codebook: codes 0..n/2-1 = +levels ascending, n/2..n-1 = -levels ascending
    let mut cb_val: Vec<f32> = Vec::new();
    let mut cb_nib: Vec<u32> = Vec::new();
    for &m in &levels { cb_val.push(m); cb_nib.push(nib(cnq::mag_index(m), false)); }
    for &m in &levels { cb_val.push(-m); cb_nib.push(nib(cnq::mag_index(m), true)); }
    let rec_bytes = 4 + 8 * bits as usize;

    let cnq_idx = Cnq::open(&cnq_path);
    let blob = cnq_idx.blob_offset;
    // per layer: (gu offset, gu bytes, gu gs, dn offset, dn bytes, dn gs)
    let mut layers = Vec::new();
    for l in 0..LAYERS {
        let gu = cnq_idx.find(&format!("model.language_model.layers.{l}.mlp.experts.gate_up_proj"), "text").clone();
        let dn = cnq_idx.find(&format!("model.language_model.layers.{l}.mlp.experts.down_proj"), "text").clone();
        layers.push((blob + gu.offset, Cnq::byte_len(&gu), gu.global_scale, blob + dn.offset, Cnq::byte_len(&dn), dn.global_scale));
    }
    let gu_blocks = (2 * INTER * H) / 64;
    let dn_blocks = (H * INTER) / 64;
    let gu_rec = gu_blocks * rec_bytes;
    let dn_rec = dn_blocks * rec_bytes;
    let per_layer_out = E * (gu_rec + dn_rec);
    eprintln!("coldtier: {bits}-bit, codebook {:?}, record {rec_bytes} B/64 values, per expert {} B (was {} B), output {:.1} GB",
        levels, gu_rec + dn_rec, (gu_blocks + dn_blocks) * 36, (per_layer_out * LAYERS) as f64 / 1e9);

    // ---- worker: one layer -> Vec<u8> of E*(gu_rec+dn_rec) bytes, plus SSE stats
    let work = |l: usize| -> (Vec<u8>, f64, f64, u64) {
        let (gu_off, gu_len, gu_gs, dn_off, dn_len, dn_gs) = layers[l];
        let mut f = std::fs::File::open(&cnq_path).unwrap();
        let mut out = vec![0u8; per_layer_out];
        let mut sse_new = 0f64;
        let mut sse_ref = 0f64; // energy of the reference values (for a relative number)
        let mut n_vals = 0u64;
        let mut blk = [0f32; 64];
        for (which, off, len, gs, nblk) in [(0usize, gu_off, gu_len, gu_gs, gu_blocks), (1, dn_off, dn_len, dn_gs, dn_blocks)] {
            let per_expert_in = nblk * 36;
            assert_eq!(per_expert_in as u64 * E as u64, len);
            let mut raw = vec![0u8; per_expert_in];
            for e in 0..E {
                f.seek(SeekFrom::Start(off + (e * per_expert_in) as u64)).unwrap();
                f.read_exact(&mut raw).unwrap();
                let obase = e * (gu_rec + dn_rec) + if which == 0 { 0 } else { gu_rec };
                for b in 0..nblk {
                    let src = &raw[b * 36..(b + 1) * 36];
                    cnq::dequant_block(src, gs, &mut blk);
                    let dst = &mut out[obase + b * rec_bytes..obase + (b + 1) * rec_bytes];
                    let mut codes = [0u32; 64];
                    for sb in 0..4 {
                        let vals = &blk[sb * 16..(sb + 1) * 16];
                        let best = cnq::best_scale_and_codes(vals, src[sb] as i32, gs, &cb_val);
                        dst[sb] = best.1 as u8;
                        codes[sb * 16..(sb + 1) * 16].copy_from_slice(&best.2);
                        sse_new += best.0;
                        for &v in vals { sse_ref += (v as f64) * (v as f64); }
                        n_vals += 16;
                    }
                    // pack codes LSB-first
                    let mut acc: u64 = 0;
                    let mut nbits = 0u32;
                    let mut o = 4usize;
                    for &c in &codes {
                        acc |= (c as u64) << nbits;
                        nbits += bits;
                        while nbits >= 8 {
                            dst[o] = (acc & 0xFF) as u8;
                            o += 1;
                            acc >>= 8;
                            nbits -= 8;
                        }
                    }
                    assert_eq!(o, rec_bytes);
                }
            }
        }
        (out, sse_new, sse_ref, n_vals)
    };

    // ---- run layers in parallel, write in order
    let t0 = std::time::Instant::now();
    let mut outf = std::fs::File::create(&out_path).unwrap();
    // header: json + u64 length at the END (same trailer idea as CNQ)
    let header = serde_json::json!({
        "format": "crow-nest-coldtier", "version": 1, "bits": bits, "levels": levels,
        "codebook_nibbles": cb_nib, "record_bytes": rec_bytes,
        "gu_record_bytes": gu_rec, "dn_record_bytes": dn_rec, "layers": LAYERS, "experts": E,
        "layer_bytes": per_layer_out, "source": cnq_path,
    });
    let mut done = 0usize;
    let mut tot_new = 0f64;
    let mut tot_ref = 0f64;
    let mut tot_n = 0u64;
    let mut next = 0usize;
    while next < LAYERS {
        let batch: Vec<usize> = (next..(next + threads).min(LAYERS)).collect();
        let results: Vec<(usize, (Vec<u8>, f64, f64, u64))> = std::thread::scope(|sc| {
            let hs: Vec<_> = batch.iter().map(|&l| { let w = &work; sc.spawn(move || (l, w(l))) }).collect();
            hs.into_iter().map(|h| h.join().unwrap()).collect()
        });
        for (l, (bytes, sn, sr, n)) in results {
            assert_eq!(l, done);
            outf.write_all(&bytes).unwrap();
            tot_new += sn; tot_ref += sr; tot_n += n;
            done += 1;
            eprintln!("coldtier: layer {l:2} done ({:.0} s), rel_rms so far {:.4}", t0.elapsed().as_secs_f64(), (tot_new / tot_ref).sqrt());
        }
        next += threads;
    }
    let hb = serde_json::to_vec(&header).unwrap();
    outf.write_all(&hb).unwrap();
    outf.write_all(&(hb.len() as u64).to_le_bytes()).unwrap();
    outf.flush().unwrap();
    eprintln!("coldtier: wrote {out_path} in {:.0} s — relative RMS error vs the FP4 values {:.4} over {} values",
        t0.elapsed().as_secs_f64(), (tot_new / tot_ref).sqrt(), tot_n);
}
