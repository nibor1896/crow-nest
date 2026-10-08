//! crow-nest #148 (GLM-5.3-Flash step 10, PREREG gate G2): the decoder draft for the staging path.
//!
//! One routed expert block (gate, up, down = 14,155,776 B of NVFP4: 393,216 blocks of 36 B, 4 ue4m3
//! scale bytes then 32 bytes of two E2M1 nibbles) is read from a container, coded with ONE static
//! order-0 table per block exactly as the converter sizes it (`converter/src/entropy.rs`: canonical
//! Huffman, package-merge length limit 12 bits for the 16 codes and 15 bits for the 256 scale bytes,
//! two streams), and decoded back on the CPU into a staging buffer in the written 36-byte layout.
//! Every decode is compared with the original bytes (lossless, bit for bit).
//!
//! Decoder: LSB-first bit streams, unaligned 8-byte peeks; a 4096-entry pair table emits one packed
//! code byte (two nibbles) per lookup when both codes fit 12 bits, else two single lookups; a
//! 32768-entry table for scale bytes. Framing: the block is cut into segments of 1024 NVFP4 blocks
//! (36,864 B out); per segment two u32 bit offsets (codes, scales) let T threads decode
//! independent segments. Framing bytes are reported apart from the G2 size (which has none).
//!
//! "Cold visit": before every timed decode a 512 MiB buffer is written to evict the caches (L3 of the
//! measuring machine: 36 MiB), so the coded input comes from DRAM as after an NVMe read; the output
//! buffer is allocated and touched once (a pinned staging buffer is pre-faulted). Thread spawn and
//! join are inside the timing (std::thread::scope per decode, no pool): conservative.
//!
//!   cargo run --release -- <container.cnq> <offset> <len> [--reps 9] [--threads 1,2,4,8,16,24] [--json out]
//!
//! The package-merge below is a copy of `converter/src/entropy.rs` `huffman_lengths` (the step-10
//! file set allows no converter change); the sizes it yields are checked against
//! `tools/glm_entropy_report.py census --block L:E` in `docs/glm-entropy.md`.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom, Write};
use std::time::Instant;

const CODE_LEN_LIMIT: u32 = 12;
const SCALE_LEN_LIMIT: u32 = 15;
const TABLE_BYTES: usize = 8 + 128;
const SEG_BLOCKS: usize = 1024;
const EVICT_BYTES: usize = 512 << 20;

/// Optimal length-limited prefix code lengths by package-merge (copy of the converter's).
fn huffman_lengths(h: &[u64], limit: u32) -> Vec<u32> {
    let used: Vec<usize> = (0..h.len()).filter(|&i| h[i] > 0).collect();
    let mut len = vec![0u32; h.len()];
    match used.len() {
        0 => return len,
        1 => {
            len[used[0]] = 1;
            return len;
        }
        n => assert!((1u64 << limit) >= n as u64),
    }
    let mut leaves: Vec<(u64, usize)> = used.iter().map(|&i| (h[i], i)).collect();
    leaves.sort();
    let n = leaves.len();
    let leaf_items: Vec<(u64, Vec<u16>)> = leaves
        .iter()
        .enumerate()
        .map(|(k, (w, _))| {
            let mut v = vec![0u16; n];
            v[k] = 1;
            (*w, v)
        })
        .collect();
    let mut list = leaf_items.clone();
    for _ in 1..limit {
        let packages: Vec<(u64, Vec<u16>)> = list
            .chunks_exact(2)
            .map(|p| (p[0].0 + p[1].0, p[0].1.iter().zip(&p[1].1).map(|(a, b)| a + b).collect()))
            .collect();
        let mut merged = Vec::with_capacity(leaf_items.len() + packages.len());
        let (mut a, mut b) = (0, 0);
        while a < leaf_items.len() || b < packages.len() {
            if b >= packages.len() || (a < leaf_items.len() && leaf_items[a].0 <= packages[b].0) {
                merged.push(leaf_items[a].clone());
                a += 1;
            } else {
                merged.push(packages[b].clone());
                b += 1;
            }
        }
        list = merged;
    }
    for item in &list[..2 * n - 2] {
        for (k, c) in item.1.iter().enumerate() {
            len[leaves[k].1] += *c as u32;
        }
    }
    len
}

/// Canonical codes (DEFLATE order: by length, then symbol), bit-reversed for LSB-first emission.
fn canonical_lsb(len: &[u32]) -> Vec<u32> {
    let mut order: Vec<usize> = (0..len.len()).filter(|&i| len[i] > 0).collect();
    order.sort_by_key(|&i| (len[i], i));
    let mut codes = vec![0u32; len.len()];
    let (mut code, mut prev) = (0u32, 0u32);
    for (k, &i) in order.iter().enumerate() {
        if k > 0 {
            code = (code + 1) << (len[i] - prev);
        }
        prev = len[i];
        codes[i] = code.reverse_bits() >> (32 - len[i]);
    }
    codes
}

struct BitWriter {
    buf: Vec<u8>,
    acc: u64,
    n: u32,
    bits: u64,
}

impl BitWriter {
    fn new() -> Self {
        BitWriter { buf: Vec::new(), acc: 0, n: 0, bits: 0 }
    }
    fn put(&mut self, code: u32, len: u32) {
        self.acc |= (code as u64) << self.n;
        self.n += len;
        self.bits += len as u64;
        while self.n >= 8 {
            self.buf.push(self.acc as u8);
            self.acc >>= 8;
            self.n -= 8;
        }
    }
    fn finish(mut self) -> Vec<u8> {
        if self.n > 0 {
            self.buf.push(self.acc as u8);
        }
        self
            .buf
            .extend_from_slice(&[0u8; 16]); // peek padding, not counted in the coded size
        self.buf
    }
}

struct Coded {
    codes: Vec<u8>,
    scales: Vec<u8>,
    code_bytes: usize,
    scale_bytes: usize,
    seg_code_bit: Vec<u32>,
    seg_scale_bit: Vec<u32>,
    code_len: Vec<u32>,
    scale_len: Vec<u32>,
}

fn encode(raw: &[u8]) -> Coded {
    let mut hc = [0u64; 16];
    let mut hs = [0u64; 256];
    for b in raw.chunks_exact(36) {
        for s in &b[..4] {
            hs[*s as usize] += 1;
        }
        for p in &b[4..] {
            hc[(p & 0xF) as usize] += 1;
            hc[(p >> 4) as usize] += 1;
        }
    }
    let code_len = huffman_lengths(&hc, CODE_LEN_LIMIT);
    let scale_len = huffman_lengths(&hs, SCALE_LEN_LIMIT);
    let cc = canonical_lsb(&code_len);
    let cs = canonical_lsb(&scale_len);
    let (mut wc, mut ws) = (BitWriter::new(), BitWriter::new());
    let (mut seg_c, mut seg_s) = (Vec::new(), Vec::new());
    for (k, b) in raw.chunks_exact(36).enumerate() {
        if k % SEG_BLOCKS == 0 {
            seg_c.push(u32::try_from(wc.bits).expect("code stream < 4 Gbit"));
            seg_s.push(u32::try_from(ws.bits).expect("scale stream < 4 Gbit"));
        }
        for s in &b[..4] {
            ws.put(cs[*s as usize], scale_len[*s as usize]);
        }
        for p in &b[4..] {
            let (lo, hi) = ((p & 0xF) as usize, (p >> 4) as usize);
            wc.put(cc[lo], code_len[lo]);
            wc.put(cc[hi], code_len[hi]);
        }
    }
    let code_bytes = wc.bits.div_ceil(8) as usize;
    let scale_bytes = ws.bits.div_ceil(8) as usize;
    Coded { codes: wc.finish(), scales: ws.finish(), code_bytes, scale_bytes, seg_code_bit: seg_c, seg_scale_bit: seg_s, code_len, scale_len }
}

struct Tables {
    /// codes, 12-bit index: bit 12 set = pair (bits 0-7 the packed byte, 8-11 total length);
    /// else bits 0-3 the first nibble, bits 8-11 its length
    pair: Vec<u16>,
    /// codes, 12-bit index: bits 0-3 symbol, 4-7 length
    single: Vec<u8>,
    /// scales, 15-bit index: bits 0-7 symbol, 8-11 length
    scale: Vec<u16>,
}

fn build_tables(c: &Coded) -> Tables {
    let cc = canonical_lsb(&c.code_len);
    let cs = canonical_lsb(&c.scale_len);
    let mut single = vec![0u8; 1 << CODE_LEN_LIMIT];
    for s in 0..16 {
        let l = c.code_len[s];
        if l == 0 {
            continue;
        }
        for k in 0..(1u32 << (CODE_LEN_LIMIT - l)) {
            single[(cc[s] | (k << l)) as usize] = s as u8 | (l as u8) << 4;
        }
    }
    let mut pair = vec![0u16; 1 << CODE_LEN_LIMIT];
    for idx in 0..(1usize << CODE_LEN_LIMIT) {
        let e1 = single[idx];
        let (s1, l1) = ((e1 & 0xF) as u16, (e1 >> 4) as u32);
        let e2 = single[idx >> l1];
        let (s2, l2) = ((e2 & 0xF) as u16, (e2 >> 4) as u32);
        pair[idx] = if l1 + l2 <= CODE_LEN_LIMIT {
            (1 << 12) | ((l1 + l2) as u16) << 8 | s1 | s2 << 4
        } else {
            (l1 as u16) << 8 | s1
        };
    }
    let mut scale = vec![0u16; 1 << SCALE_LEN_LIMIT];
    for s in 0..256 {
        let l = c.scale_len[s];
        if l == 0 {
            continue;
        }
        for k in 0..(1u32 << (SCALE_LEN_LIMIT - l)) {
            scale[(cs[s] | (k << l)) as usize] = s as u16 | (l as u16) << 8;
        }
    }
    Tables { pair, single, scale }
}

#[inline(always)]
fn peek(data: &[u8], pos: u64) -> u64 {
    let i = (pos >> 3) as usize;
    let w = u64::from_le_bytes(data[i..i + 8].try_into().unwrap());
    w >> (pos & 7)
}

/// Decode segments [s0, s1) into `out` (their 36-byte blocks, contiguous).
fn decode_segments(c: &Coded, t: &Tables, s0: usize, s1: usize, n_blocks: usize, out: &mut [u8]) {
    let mut o = 0usize;
    for seg in s0..s1 {
        let mut pc = c.seg_code_bit[seg] as u64;
        let mut ps = c.seg_scale_bit[seg] as u64;
        let b_end = ((seg + 1) * SEG_BLOCKS).min(n_blocks);
        for _ in seg * SEG_BLOCKS..b_end {
            let blk = &mut out[o..o + 36];
            // 4 scale bytes, two per peek (2 x 15 <= 57 bits)
            for h in 0..2 {
                let mut w = peek(&c.scales, ps);
                let e = t.scale[(w & 0x7FFF) as usize];
                blk[2 * h] = e as u8;
                let l = (e >> 8) as u64;
                w >>= l;
                let e2 = t.scale[(w & 0x7FFF) as usize];
                blk[2 * h + 1] = e2 as u8;
                ps += l + (e2 >> 8) as u64;
            }
            // 32 code bytes, two per peek (2 x 24 <= 57 bits)
            for q in 0..16 {
                let mut w = peek(&c.codes, pc);
                let mut used = 0u64;
                for r in 0..2 {
                    let e = t.pair[(w & 0xFFF) as usize];
                    let (byte, l) = if e & (1 << 12) != 0 {
                        (e as u8, ((e >> 8) & 0xF) as u64)
                    } else {
                        let (s1, l1) = (e as u8 & 0xF, ((e >> 8) & 0xF) as u64);
                        let e2 = t.single[((w >> l1) & 0xFFF) as usize];
                        (s1 | (e2 & 0xF) << 4, l1 + (e2 >> 4) as u64)
                    };
                    blk[4 + 2 * q + r] = byte;
                    w >>= l;
                    used += l;
                }
                pc += used;
            }
            o += 36;
        }
    }
}

fn decode(c: &Coded, t: &Tables, n_blocks: usize, out: &mut [u8], threads: usize) {
    let n_seg = c.seg_code_bit.len();
    if threads <= 1 {
        decode_segments(c, t, 0, n_seg, n_blocks, out);
        return;
    }
    let per = n_seg.div_ceil(threads);
    std::thread::scope(|s| {
        let mut rest = out;
        let mut s0 = 0;
        while s0 < n_seg {
            let s1 = (s0 + per).min(n_seg);
            let bytes = ((s1 * SEG_BLOCKS).min(n_blocks) - s0 * SEG_BLOCKS) * 36;
            let (mine, tail) = rest.split_at_mut(bytes);
            rest = tail;
            s.spawn(move || decode_segments(c, t, s0, s1, n_blocks, mine));
            s0 = s1;
        }
    });
}

fn evict(buf: &mut [u8], rep: usize) {
    for chunk in buf.chunks_mut(4096) {
        chunk[0] = rep as u8;
        chunk[2048] = rep as u8;
    }
    std::hint::black_box(&buf[0]);
}

fn median(v: &mut [f64]) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 { v[n / 2] } else { (v[n / 2 - 1] + v[n / 2]) / 2.0 }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 4 {
        eprintln!("usage: glm-huff-draft <container.cnq> <offset> <len> [--reps N] [--threads 1,2,4] [--json out]");
        std::process::exit(2);
    }
    let (path, off, len): (&str, u64, usize) = (&args[1], args[2].parse().unwrap(), args[3].parse().unwrap());
    let (mut reps, mut threads, mut json): (usize, Vec<usize>, Option<String>) = (9, vec![1, 2, 4, 8, 16, 24], None);
    let mut i = 4;
    while i < args.len() {
        match args[i].as_str() {
            "--reps" => reps = args[i + 1].parse().unwrap(),
            "--threads" => threads = args[i + 1].split(',').map(|x| x.parse().unwrap()).collect(),
            "--json" => json = Some(args[i + 1].clone()),
            a => panic!("unknown argument {a}"),
        }
        i += 2;
    }
    assert!(len % 36 == 0 && reps >= 5);
    let mut raw = vec![0u8; len];
    let mut f = File::open(path).expect("open container");
    f.seek(SeekFrom::Start(off)).unwrap();
    f.read_exact(&mut raw).unwrap();
    let n_blocks = len / 36;

    let t0 = Instant::now();
    let c = encode(&raw);
    let enc_ms = t0.elapsed().as_secs_f64() * 1e3;
    let t = build_tables(&c);
    let framing = c.seg_code_bit.len() * 8;
    let coded = c.code_bytes + c.scale_bytes + TABLE_BYTES;
    println!(
        "block {} @ {} len {}: codes {} B, scales {} B, tables {} B -> coded {} B, saving {:.3} % (framing {} B apart, {} segments); encode {:.1} ms (not on the read path)",
        path, off, len, c.code_bytes, c.scale_bytes, TABLE_BYTES, coded, 100.0 * (1.0 - coded as f64 / len as f64), framing, c.seg_code_bit.len(), enc_ms
    );

    let mut out = vec![0u8; len];
    let mut ev = vec![0u8; EVICT_BYTES];
    // warm-up and correctness once
    decode(&c, &t, n_blocks, &mut out, 1);
    assert!(out == raw, "decode is not lossless");

    let mut rows = Vec::new();
    // reference: a cold memcpy of the raw block (the CPU copy floor of an uncoded staging)
    let mut cp = Vec::new();
    for r in 0..reps {
        evict(&mut ev, r);
        let s = Instant::now();
        out.copy_from_slice(&raw);
        std::hint::black_box(&out);
        cp.push(s.elapsed().as_secs_f64() * 1e3);
    }
    let cp_med = median(&mut cp.clone());
    println!("memcpy raw block, cold, 1 thread: median {:.3} ms over {} reps ({:.3} .. {:.3})", cp_med, reps, cp.iter().cloned().fold(f64::INFINITY, f64::min), cp.iter().cloned().fold(0.0, f64::max));
    for &th in &threads {
        let mut ms = Vec::new();
        for r in 0..reps {
            out.fill(0);
            evict(&mut ev, r);
            let s = Instant::now();
            decode(&c, &t, n_blocks, &mut out, th);
            ms.push(s.elapsed().as_secs_f64() * 1e3);
            assert!(out == raw, "decode is not lossless (threads {th}, rep {r})");
        }
        let med = median(&mut ms.clone());
        let lo = ms.iter().cloned().fold(f64::INFINITY, f64::min);
        let hi = ms.iter().cloned().fold(0.0, f64::max);
        println!(
            "decode cold, {:>2} threads: median {:.3} ms ({:.3} .. {:.3}, spread {:.3}), {:.2} GB/s out, lossless {}/{}",
            th, med, lo, hi, hi / lo, len as f64 / med / 1e6, reps, reps
        );
        rows.push((th, med, lo, hi, ms));
    }
    if let Some(p) = json {
        let mut s = String::new();
        s += &format!(
            "{{\"container\": {:?}, \"offset\": {}, \"len\": {}, \"segments\": {}, \"seg_blocks\": {}, \"framing_bytes\": {}, \"code_bytes\": {}, \"scale_bytes\": {}, \"table_bytes\": {}, \"coded_bytes\": {}, \"encode_ms\": {:.3}, \"reps\": {}, \"memcpy_cold_ms\": {{\"median\": {:.4}, \"reps\": {:?}}}, \"decode\": [",
            path, off, len, c.seg_code_bit.len(), SEG_BLOCKS, framing, c.code_bytes, c.scale_bytes, TABLE_BYTES, coded, enc_ms, reps, cp_med, cp
        );
        for (k, (th, med, lo, hi, ms)) in rows.iter().enumerate() {
            s += &format!(
                "{}{{\"threads\": {}, \"median_ms\": {:.4}, \"min_ms\": {:.4}, \"max_ms\": {:.4}, \"gbps_out\": {:.4}, \"lossless\": true, \"reps_ms\": {:?}}}",
                if k > 0 { ", " } else { "" }, th, med, lo, hi, len as f64 / med / 1e6, ms
            );
        }
        s += "]}\n";
        File::create(&p).unwrap().write_all(s.as_bytes()).unwrap();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic(n_blocks: usize) -> Vec<u8> {
        // skewed scales and codes, deterministic LCG
        let mut x: u64 = 0x9E3779B97F4A7C15;
        let mut v = Vec::with_capacity(n_blocks * 36);
        for _ in 0..n_blocks {
            for _ in 0..4 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                v.push(0x60 + ((x >> 60) as u8 & 7) * ((x >> 59) as u8 & 1));
            }
            for _ in 0..32 {
                x = x.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                let a = ((x >> 33) % 16) as u8;
                let b = ((x >> 45) % 7) as u8;
                v.push(a | b << 4);
            }
        }
        v
    }

    #[test]
    fn round_trip_is_lossless_for_every_thread_count() {
        let raw = synthetic(SEG_BLOCKS * 3 + 17);
        let c = encode(&raw);
        let t = build_tables(&c);
        for th in [1, 2, 3, 8] {
            let mut out = vec![0u8; raw.len()];
            decode(&c, &t, raw.len() / 36, &mut out, th);
            assert_eq!(out, raw, "threads {th}");
        }
    }

    #[test]
    fn coded_size_is_the_package_merge_cost() {
        let raw = synthetic(500);
        let c = encode(&raw);
        let mut hc = [0u64; 16];
        let mut hs = [0u64; 256];
        for b in raw.chunks_exact(36) {
            b[..4].iter().for_each(|s| hs[*s as usize] += 1);
            b[4..].iter().for_each(|p| {
                hc[(p & 0xF) as usize] += 1;
                hc[(p >> 4) as usize] += 1;
            });
        }
        let bits = |h: &[u64], l: &[u32]| h.iter().zip(l).map(|(c, l)| c * *l as u64).sum::<u64>();
        assert_eq!(c.code_bytes as u64, bits(&hc, &c.code_len).div_ceil(8));
        assert_eq!(c.scale_bytes as u64, bits(&hs, &c.scale_len).div_ceil(8));
        // Kraft: a complete prefix code on both streams
        for (l, lim) in [(&c.code_len, CODE_LEN_LIMIT), (&c.scale_len, SCALE_LEN_LIMIT)] {
            let k: f64 = l.iter().filter(|x| **x > 0).map(|x| 2f64.powi(-(*x as i32))).sum();
            assert!(k <= 1.0 + 1e-12 && l.iter().all(|x| *x <= lim));
        }
    }
}
