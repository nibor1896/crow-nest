//! crow-nest #155 (GLM-5.3-Flash step 5, PREREG gate G2): code histograms of the WRITTEN NVFP4
//! bytes and the exact size a static order-0 entropy coder reaches on them, tables included.
//!
//! - **Histograms** are counted from the packed 36-byte blocks the converter writes (4 ue4m3
//!   scale bytes, then 32 bytes of two E2M1 nibbles each), never from the f32 input. A value that
//!   `--scales mse` clips is still one written nibble, so `sum(h_codes) == n_values` and
//!   `sum(h_scales) == n_values / 16` hold by construction. Codes are the 16 nibble values
//!   including the sign bit (+0 and -0 are two codes, both are written).
//! - **Exact coder size**: canonical Huffman over the histogram, length-limited by package-merge
//!   (Larmore & Hirschberg 1990: optimal under the limit), 12 bits for the 16 codes and 15 bits
//!   for the 256 scale bytes so a table decoder can index it. Size = ceil(sum count * length / 8)
//!   per stream plus the code-length table (4 bits per symbol: 8 B for codes, 128 B for scales).
//!   Codes and scale bytes are two streams per table. A stream with one distinct symbol gets
//!   length 1 (a conventional canonical code; a zero-bit stream would need a side channel).
//! - **Order-0 entropy** H = sum -c log2(c / N) is printed beside it: the bound of any static
//!   order-0 coder (an ANS coder could close at most the gap Huffman -> H; it is reported, not
//!   built). A context coder may do better and costs more decode time (PREREG G2).

use std::collections::BTreeMap;

pub const CODE_LEN_LIMIT: u32 = 12;
pub const SCALE_LEN_LIMIT: u32 = 15;
/// 16 symbols x 4-bit code length
pub const CODE_TABLE_BYTES: u64 = 8;
/// 256 symbols x 4-bit code length
pub const SCALE_TABLE_BYTES: u64 = 128;

#[derive(Clone, PartialEq, Eq, Debug)]
pub struct Hist {
    pub codes: [u64; 16],
    pub scales: [u64; 256],
}

impl Default for Hist {
    fn default() -> Self {
        Hist { codes: [0; 16], scales: [0; 256] }
    }
}

impl Hist {
    /// Count the written bytes of an NVFP4 tensor (36 B per 64 values).
    pub fn of_blocks(blocks: &[u8]) -> Hist {
        assert!(blocks.len() % 36 == 0, "NVFP4 blocks are 36 bytes");
        let mut h = Hist::default();
        for b in blocks.chunks_exact(36) {
            for s in &b[..4] {
                h.scales[*s as usize] += 1;
            }
            for p in &b[4..] {
                h.codes[(p & 0xF) as usize] += 1;
                h.codes[(p >> 4) as usize] += 1;
            }
        }
        h
    }

    pub fn add(&mut self, o: &Hist) {
        for (a, b) in self.codes.iter_mut().zip(o.codes.iter()) {
            *a += b;
        }
        for (a, b) in self.scales.iter_mut().zip(o.scales.iter()) {
            *a += b;
        }
    }

    pub fn n_codes(&self) -> u64 {
        self.codes.iter().sum()
    }

    pub fn n_scales(&self) -> u64 {
        self.scales.iter().sum()
    }

    /// The NVFP4 bytes this histogram describes: half a byte per code, one per scale.
    pub fn raw_bytes(&self) -> u64 {
        self.n_codes() / 2 + self.n_scales()
    }

    pub fn to_json(&self) -> (serde_json::Value, serde_json::Value) {
        (serde_json::json!(self.codes.to_vec()), serde_json::json!(self.scales.to_vec()))
    }

    pub fn from_json(codes: &serde_json::Value, scales: &serde_json::Value) -> Option<Hist> {
        let c = codes.as_array()?;
        let s = scales.as_array()?;
        if c.len() != 16 || s.len() != 256 {
            return None;
        }
        let mut h = Hist::default();
        for (d, v) in h.codes.iter_mut().zip(c) {
            *d = v.as_u64()?;
        }
        for (d, v) in h.scales.iter_mut().zip(s) {
            *d = v.as_u64()?;
        }
        Some(h)
    }
}

/// Order-0 entropy of a histogram in bits (sum over symbols of -c log2(c / N)).
pub fn entropy_bits(h: &[u64]) -> f64 {
    let n: u64 = h.iter().sum();
    if n == 0 {
        return 0.0;
    }
    let nf = n as f64;
    h.iter().filter(|c| **c > 0).map(|&c| -(c as f64) * (c as f64 / nf).log2()).sum()
}

/// Optimal length-limited prefix code lengths by package-merge. Symbols with count 0 get
/// length 0; a lone symbol gets length 1.
pub fn huffman_lengths(h: &[u64], limit: u32) -> Vec<u32> {
    let used: Vec<usize> = (0..h.len()).filter(|&i| h[i] > 0).collect();
    let mut len = vec![0u32; h.len()];
    match used.len() {
        0 => return len,
        1 => {
            len[used[0]] = 1;
            return len;
        }
        n => assert!((1u64 << limit) >= n as u64, "{n} symbols do not fit a {limit}-bit limit"),
    }
    // leaves sorted by (count, symbol): deterministic
    let mut leaves: Vec<(u64, usize)> = used.iter().map(|&i| (h[i], i)).collect();
    leaves.sort();
    let n = leaves.len();
    // an item = (weight, how often each leaf occurs in it)
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
        // merge leaves and packages by weight; a leaf goes before a package of equal weight
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

/// Bits of one stream coded with `len`.
pub fn coded_bits(h: &[u64], len: &[u32]) -> u64 {
    h.iter().zip(len).map(|(c, l)| c * *l as u64).sum()
}

/// One table over one histogram: what a static order-0 coder stores.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Coded {
    pub raw_bytes: u64,
    /// both streams, each rounded up to whole bytes, plus both code-length tables
    pub coded_bytes: u64,
    pub table_bytes: u64,
    /// order-0 entropy of both streams in bytes, no tables, no rounding
    pub entropy_bytes: f64,
}

impl Coded {
    pub fn of(h: &Hist) -> Coded {
        let lc = huffman_lengths(&h.codes, CODE_LEN_LIMIT);
        let ls = huffman_lengths(&h.scales, SCALE_LEN_LIMIT);
        let table_bytes = CODE_TABLE_BYTES + SCALE_TABLE_BYTES;
        Coded {
            raw_bytes: h.raw_bytes(),
            coded_bytes: coded_bits(&h.codes, &lc).div_ceil(8) + coded_bits(&h.scales, &ls).div_ceil(8) + table_bytes,
            table_bytes,
            entropy_bytes: (entropy_bits(&h.codes) + entropy_bits(&h.scales)) / 8.0,
        }
    }

    pub fn add(&mut self, o: &Coded) {
        self.raw_bytes += o.raw_bytes;
        self.coded_bytes += o.coded_bytes;
        self.table_bytes += o.table_bytes;
        self.entropy_bytes += o.entropy_bytes;
    }

    /// 1 - coded / raw (the G2 saving, tables included)
    pub fn saving(&self) -> f64 {
        if self.raw_bytes == 0 { 0.0 } else { 1.0 - self.coded_bytes as f64 / self.raw_bytes as f64 }
    }

    pub fn saving_entropy(&self) -> f64 {
        if self.raw_bytes == 0 { 0.0 } else { 1.0 - self.entropy_bytes / self.raw_bytes as f64 }
    }
}

/// One NVFP4 tensor's histogram with what it is.
#[derive(Clone, Debug)]
pub struct TensorHist {
    pub class: String,
    pub layer: Option<u64>,
    pub expert: Option<u64>,
    pub hist: Hist,
}

/// The `code_summary` records of a conversion (sidecar lines), from the per-tensor histograms:
/// - `scope: "class"`: one table per class;
/// - `scope: "layer_class"`: one table per (layer, class);
/// - `scope: "tensor_tables"`: one table per tensor, summed per class;
/// - `scope: "expert_blocks"`: one table per routed expert block (the projections of one expert
///   of one layer together), summed over all blocks — the G2 metric — with the min / max block
///   saving, and the same per layer (`scope: "expert_blocks_layer"`).
pub fn code_summary(tensors: &[TensorHist]) -> Vec<serde_json::Value> {
    let rec = |scope: &str, c: &Coded, extra: serde_json::Value| {
        let mut v = serde_json::json!({
            "record": "code_summary", "scope": scope,
            "raw_bytes": c.raw_bytes, "coded_bytes": c.coded_bytes, "table_bytes": c.table_bytes,
            "entropy_bytes": c.entropy_bytes, "saving": c.saving(), "saving_entropy": c.saving_entropy(),
        });
        for (k, x) in extra.as_object().unwrap() {
            v[k] = x.clone();
        }
        v
    };
    let mut out = Vec::new();
    let mut by_class: BTreeMap<&str, (usize, Hist, Coded)> = BTreeMap::new();
    let mut by_layer_class: BTreeMap<(u64, &str), (usize, Hist)> = BTreeMap::new();
    let mut blocks: BTreeMap<(u64, u64), Hist> = BTreeMap::new();
    for t in tensors {
        let e = by_class.entry(t.class.as_str()).or_insert((0, Hist::default(), Coded::default()));
        e.0 += 1;
        e.1.add(&t.hist);
        e.2.add(&Coded::of(&t.hist));
        if let Some(l) = t.layer {
            let e = by_layer_class.entry((l, t.class.as_str())).or_insert((0, Hist::default()));
            e.0 += 1;
            e.1.add(&t.hist);
            if let Some(x) = t.expert {
                blocks.entry((l, x)).or_default().add(&t.hist);
            }
        }
    }
    for (class, (n, h, per_tensor)) in &by_class {
        out.push(rec("class", &Coded::of(h), serde_json::json!({ "class": class, "tensors": n, "n": h.n_codes() })));
        out.push(rec("tensor_tables", per_tensor, serde_json::json!({ "class": class, "tensors": n })));
    }
    for ((layer, class), (n, h)) in &by_layer_class {
        out.push(rec("layer_class", &Coded::of(h), serde_json::json!({ "class": class, "layer": layer, "tensors": n })));
    }
    if !blocks.is_empty() {
        let mut all = Coded::default();
        let mut per_layer: BTreeMap<u64, (usize, Coded, f64, f64)> = BTreeMap::new();
        let (mut lo, mut hi) = (f64::INFINITY, f64::NEG_INFINITY);
        for ((layer, _), h) in &blocks {
            let c = Coded::of(h);
            all.add(&c);
            lo = lo.min(c.saving());
            hi = hi.max(c.saving());
            let e = per_layer.entry(*layer).or_insert((0, Coded::default(), f64::INFINITY, f64::NEG_INFINITY));
            e.0 += 1;
            e.1.add(&c);
            e.2 = e.2.min(c.saving());
            e.3 = e.3.max(c.saving());
        }
        out.push(rec("expert_blocks", &all, serde_json::json!({ "blocks": blocks.len(), "saving_min": lo, "saving_max": hi })));
        for (layer, (n, c, lo, hi)) in &per_layer {
            out.push(rec("expert_blocks_layer", c, serde_json::json!({ "layer": layer, "blocks": n, "saving_min": lo, "saving_max": hi })));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// {1: 50 %, 2: 25 %, 3: 25 %}: Huffman lengths 1, 2, 2 = 1.5 bits per symbol, which is
    /// also the entropy; the table bytes come on top.
    #[test]
    fn huffman_on_a_hand_computable_histogram() {
        let mut h = [0u64; 16];
        h[1] = 512;
        h[2] = 256;
        h[3] = 256;
        let l = huffman_lengths(&h, CODE_LEN_LIMIT);
        assert_eq!((l[1], l[2], l[3]), (1, 2, 2));
        assert_eq!(l.iter().filter(|x| **x > 0).count(), 3);
        assert_eq!(coded_bits(&h, &l), 1536); // 1024 symbols x 1.5 bits
        assert!((entropy_bits(&h) - 1536.0).abs() < 1e-9);
        let mut hist = Hist::default();
        hist.codes = h;
        hist.scales[0x38] = 64; // one scale byte value: length 1
        let c = Coded::of(&hist);
        assert_eq!(c.raw_bytes, 1024 / 2 + 64);
        assert_eq!(c.coded_bytes, 1536 / 8 + 64 / 8 + 8 + 128);
        assert_eq!(c.table_bytes, 136);
    }

    /// The length limit binds: a Fibonacci histogram wants a 15-deep code; limited to 4 bits it
    /// must still be a prefix code (Kraft sum <= 1) and no worse than the flat 4-bit code.
    #[test]
    fn package_merge_respects_the_limit_and_kraft() {
        let mut h = [0u64; 16];
        let (mut a, mut b) = (1u64, 1u64);
        for x in h.iter_mut() {
            *x = a;
            (a, b) = (b, a + b);
        }
        let free = huffman_lengths(&h, 15);
        assert!(*free.iter().max().unwrap() > 4);
        let l = huffman_lengths(&h, 4);
        assert!(l.iter().all(|x| *x <= 4 && *x >= 1));
        let kraft: f64 = l.iter().map(|x| 2f64.powi(-(*x as i32))).sum();
        assert!(kraft <= 1.0 + 1e-12, "{kraft}");
        assert_eq!(coded_bits(&h, &l), 4 * h.iter().sum::<u64>(), "16 symbols at 4 bits: the flat code is optimal");
        // unlimited: Kraft equality and at most 1 bit/symbol above entropy
        let kraft: f64 = free.iter().map(|x| 2f64.powi(-(*x as i32))).sum();
        assert!((kraft - 1.0).abs() < 1e-12);
        let n: u64 = h.iter().sum();
        assert!((coded_bits(&h, &free) as f64) < entropy_bits(&h) + n as f64);
    }

    #[test]
    fn histograms_count_the_written_bytes() {
        let mut b = vec![0u8; 72];
        b[0] = 0x40;
        b[4] = 0xF1; // low nibble 1, high nibble 15
        let h = Hist::of_blocks(&b);
        assert_eq!(h.n_codes(), 128);
        assert_eq!(h.n_scales(), 8);
        assert_eq!((h.codes[1], h.codes[15], h.codes[0]), (1, 1, 126));
        assert_eq!((h.scales[0x40], h.scales[0]), (1, 7));
        let (c, s) = h.to_json();
        assert_eq!(Hist::from_json(&c, &s).unwrap(), h);
    }

    #[test]
    fn the_expert_block_summary_has_one_table_per_block() {
        let mut h = Hist::default();
        h.codes[0] = 64;
        h.scales[1] = 4;
        let t = |class: &str, l, x| TensorHist { class: class.into(), layer: Some(l), expert: x, hist: h.clone() };
        let v = code_summary(&[t("expert_gate", 3, Some(0)), t("expert_up", 3, Some(0)), t("expert_down", 3, Some(0)), t("expert_gate", 3, Some(1)), t("attn_mla", 3, None)]);
        let eb = v.iter().find(|r| r["scope"] == "expert_blocks").unwrap();
        assert_eq!(eb["blocks"], 2);
        // block 0: 192 codes at 1 bit = 24 B + 12 scales at 1 bit = 2 B + 136 B tables; block 1: 8 + 1 + 136
        assert_eq!(eb["coded_bytes"], 24 + 2 + 136 + 8 + 1 + 136);
        assert_eq!(eb["raw_bytes"], 3 * 36 + 36);
        assert!(v.iter().any(|r| r["scope"] == "class" && r["class"] == "attn_mla"));
        assert!(v.iter().any(|r| r["scope"] == "layer_class" && r["layer"] == 3 && r["class"] == "expert_gate" && r["tensors"] == 2));
    }
}
