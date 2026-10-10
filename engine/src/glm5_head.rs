//! #165 (GLM-5.3-Flash plan step 13e): the glm5_next head — the final mean over the residual
//! streams, the final RMSNorm and the untied lm_head.
//!
//! After its last decoder layer GLM-5.3-Flash holds `hc_mult` = 4 residual streams per token.
//! The head (transformers 5.16.1 `modeling_glm5_next.py`, docs/glm5-next-recipe.md section 3
//! rows 4-6) is
//!
//! ```text
//! h      = mean over the 4 streams       Glm5NextTextHyperHead (:298-302), unweighted
//! normed = weight * (h * rsqrt(mean(h^2) + 1e-5))   Glm5NextTextRMSNorm (:66-80, :1421), NOT (1 + w)
//! logits = normed @ lm_head^T            [154880, 4096] BF16, untied (:2075, :2179-2181)
//! ```
//!
//! The mean and the norm are one kernel (`kernels_glm5_head.cu`, `glm5_stream_mean_rms`, its own
//! NVRTC module behind [`crate::kernels::glm5_head`]). The lm_head and the greedy pick reuse the
//! engine's own kernels from `KERNEL_SRC`: `gemv_bf16_w` (the BF16 lm_head GEMV `lm_head_row`
//! launches, `gen.rs`) and `argmax_k`. Nothing in the engine calls this module yet: the lead
//! wires the `StreamMeanRms` arm into `gen.rs` (docs/glm5-head.md, "Integration").
//!
//! The host twin ([`stream_mean_rms_ref`], [`logit_ref`]) is the f64 reference the tests hold
//! the kernel and the oracle goldens (`engine/tests/fixtures/glm5/head`, written by
//! `oracle/export_glm5_head_golden.py`) against.

use crate::cuda;
use crate::geo::Glm5Geo;
use crate::kernels::{glm5_head, launch_v, Kernels};
use cudarc::driver::sys::CUdeviceptr;

/// GLM's final RMSNorm multiplies by `weight`, not `1 + weight` (`Glm5NextTextRMSNorm.forward`,
/// `modeling_glm5_next.py:80`; the Qwen families' zero-centred gamma is `Geo::norm_one_plus_w`).
/// `Glm5Geo` has no such field because the family has one form; this pin is it.
pub const GLM5_NORM_ONE_PLUS_W: bool = false;

/// the head's numbers, read from the family row
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HeadGeo {
    pub hidden: usize,
    /// residual streams per token row (`hc_mult`)
    pub streams: usize,
    pub vocab: usize,
    /// the final norm's eps (`rms_norm_eps`)
    pub eps: f32,
    /// the norm weight applied as `1 + w`
    pub one_plus_w: bool,
}

impl HeadGeo {
    /// panics by name on a tied checkpoint: the head reads `lm_head.weight`, never the embedding
    pub fn of(g: &Glm5Geo) -> HeadGeo {
        assert!(!g.tie_word_embeddings, "glm5_head: tie_word_embeddings is true, the head reads an untied lm_head (#165)");
        HeadGeo { hidden: g.hidden, streams: g.hc_streams, vocab: g.vocab, eps: g.rms_eps as f32, one_plus_w: GLM5_NORM_ONE_PLUS_W }
    }

    /// bytes of the BF16 lm_head `[vocab][hidden]`
    pub fn lm_head_bytes(&self) -> u64 {
        self.vocab as u64 * self.hidden as u64 * 2
    }
}

/// f64 host twin of `glm5_stream_mean_rms`: `x [rows][S][H]` -> `[rows][H]`
pub fn stream_mean_rms_ref(x: &[f32], w: &[f32], g: &HeadGeo) -> Vec<f64> {
    let (h, s) = (g.hidden, g.streams);
    assert_eq!(w.len(), h);
    assert_eq!(x.len() % (s * h), 0, "glm5_head: x is not [rows][{s}][{h}]");
    let mut out = Vec::with_capacity(x.len() / s);
    for xr in x.chunks(s * h) {
        let m: Vec<f64> = (0..h).map(|d| (0..s).map(|k| xr[k * h + d] as f64).sum::<f64>() / s as f64).collect();
        let r = 1.0 / (m.iter().map(|v| v * v).sum::<f64>() / h as f64 + g.eps as f64).sqrt();
        out.extend(m.iter().zip(w).map(|(&v, &wd)| (if g.one_plus_w { 1.0 + wd as f64 } else { wd as f64 }) * (v * r)));
    }
    out
}

/// the BF16 bit pattern as f32 (exact widening)
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

/// f64 host twin of one logit: `normed . lm_head[row]` with the row in BF16
pub fn logit_ref(normed: &[f64], lm_row: &[u16]) -> f64 {
    assert_eq!(normed.len(), lm_row.len());
    normed.iter().zip(lm_row).map(|(&a, &b)| a * bf16_to_f32(b) as f64).sum()
}

/// the device head: `glm5_stream_mean_rms` and the per-shape parameter buffers of the three
/// launches. Launches queue on the current stream (graph capture safe: no sync, no upload).
pub struct Head {
    pub geo: HeadGeo,
    pub kn: glm5_head::Kernels,
    /// {H, S, one_plus_w, eps bits} for `glm5_stream_mean_rms`
    prm: CUdeviceptr,
    /// {H} = the GEMV's k, {V} = its rows and `argmax_k`'s n
    prm_h: CUdeviceptr,
    prm_v: CUdeviceptr,
}

impl Head {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(geo: HeadGeo) -> Head {
        // gemv_bf16_w: each lane reads 8 BF16 per step over a 256-wide stride (uint4 loads)
        assert!(geo.hidden > 0 && geo.hidden % 256 == 0, "glm5_head: hidden {} is not a multiple of 256 (gemv_bf16_w)", geo.hidden);
        assert!(geo.streams > 0 && geo.vocab > 0);
        let i = |v: usize| i32::try_from(v).expect("glm5_head: parameter beyond i32");
        Head {
            geo,
            kn: glm5_head::Kernels::new(),
            prm: cuda::to_i32_dev(&[i(geo.hidden), i(geo.streams), geo.one_plus_w as i32, geo.eps.to_bits() as i32]),
            prm_h: cuda::to_i32_dev(&[i(geo.hidden)]),
            prm_v: cuda::to_i32_dev(&[i(geo.vocab)]),
        }
    }

    /// queue `out [rows][H] = norm(mean over the streams of x [rows][S][H])`
    ///
    /// # Safety
    /// `x`, `w` (f32 `[H]`) and `out` are device buffers of these shapes.
    pub unsafe fn stream_mean_rms(&self, x: CUdeviceptr, w: CUdeviceptr, out: CUdeviceptr, rows: usize) {
        launch_v(self.kn.stream_mean_rms, rows as u32, 1, 1, 256, &[x, w, out, self.prm]);
    }

    /// queue `logits [rows][V] = normed [rows][H] . lm_head^T`: the engine's BF16 lm_head GEMV
    /// (`gemv_bf16_w`, the kernel `lm_head_row` launches), batched over rows by its grid y
    ///
    /// # Safety
    /// `k` is the engine's kernel table; `lm` is the raw BF16 `[V][H]`, `normed` f32 `[rows][H]`,
    /// `logits` f32 `[rows][V]`, all device buffers.
    pub unsafe fn lm_head(&self, k: &Kernels, lm: CUdeviceptr, normed: CUdeviceptr, logits: CUdeviceptr, rows: usize) {
        launch_v(k.f("gemv_bf16_w"), self.geo.vocab.div_ceil(8) as u32, rows as u32, 1, 256, &[lm, normed, logits, self.prm_h, self.prm_v]);
    }

    /// queue the greedy id of every logits row into `ids [rows]` i32 (`argmax_k`, one launch per row)
    ///
    /// # Safety
    /// As [`Head::lm_head`]; `ids` holds `rows` i32.
    pub unsafe fn argmax(&self, k: &Kernels, logits: CUdeviceptr, ids: CUdeviceptr, rows: usize) {
        for r in 0..rows {
            launch_v(k.f("argmax_k"), 1, 1, 1, 1024, &[logits + (r * self.geo.vocab * 4) as u64, ids + (r * 4) as u64, self.prm_v]);
        }
    }

    /// queue the whole head: stream mean + norm into `normed`, the lm_head into `logits`
    ///
    /// # Safety
    /// As [`Head::stream_mean_rms`] and [`Head::lm_head`].
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn run(&self, k: &Kernels, x: CUdeviceptr, w: CUdeviceptr, lm: CUdeviceptr, normed: CUdeviceptr, logits: CUdeviceptr, rows: usize) {
        self.stream_mean_rms(x, w, normed, rows);
        self.lm_head(k, lm, normed, logits, rows);
    }

    /// # Safety
    /// No launch of this head is pending.
    pub unsafe fn free(&mut self) {
        for d in [&mut self.prm, &mut self.prm_h, &mut self.prm_v] {
            cuda::free_dev(d);
        }
        self.kn.module.unload();
    }
}

/// the synthetic weights of the goldens (`oracle/export_glm5_head_golden.py`): a counter hash
/// whose values are exact in BF16, so the 1.27 GB lm_head is generated, not stored
pub mod testkit {
    /// splitmix64 of counter `i` in stream `seed` (wrapping u64, the oracle's formula)
    pub fn splitmix64(seed: u64, i: u64) -> u64 {
        let mut z = seed.wrapping_add(i.wrapping_add(1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// `lm_head[r][c] = ((h >> 56) - 128) * 2^-9`, `h = splitmix64(seed, r * hidden + c)`, as BF16 bits
    pub fn lm_head_bf16(seed: u64, i: u64) -> u16 {
        let k = (splitmix64(seed, i) >> 56) as i32 - 128;
        let v = k as f32 * (1.0 / 512.0);
        debug_assert_eq!(v.to_bits() & 0xFFFF, 0);
        (v.to_bits() >> 16) as u16
    }

    /// `norm_w[d] = 0.5 + (h >> 57) * 2^-7`, `h = splitmix64(seed, d)`
    pub fn norm_weight(seed: u64, h: usize) -> Vec<f32> {
        (0..h as u64).map(|d| 0.5 + (splitmix64(seed, d) >> 57) as f32 * (1.0 / 128.0)).collect()
    }

    /// rows `r0..r1` of the lm_head, BF16 `[r1 - r0][hidden]`
    pub fn lm_head_rows(seed: u64, hidden: usize, r0: usize, r1: usize) -> Vec<u16> {
        (r0 * hidden..r1 * hidden).map(|i| lm_head_bf16(seed, i as u64)).collect()
    }

    pub fn cosine(a: &[f64], b: &[f64]) -> f64 {
        let dot: f64 = a.iter().zip(b).map(|(x, y)| x * y).sum();
        let na: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let nb: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (na * nb)
    }

    pub fn max_abs(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f64::max)
    }

    /// the golden fixture `engine/tests/fixtures/glm5/head`
    pub struct Golden {
        pub man: serde_json::Value,
        pub anchors: usize,
        pub hidden: usize,
        pub streams: usize,
        pub vocab: usize,
        pub x: Vec<f32>,
        pub normed: Vec<f32>,
        pub logits: Vec<f32>,
    }

    pub fn dir() -> std::path::PathBuf {
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/glm5/head")
    }

    fn f32s(name: &str, n: usize) -> Vec<f32> {
        let b = std::fs::read(dir().join(name)).unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(b.len(), n * 4, "{name}: {} bytes, expected {}", b.len(), n * 4);
        b.chunks_exact(4).map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])).collect()
    }

    pub fn golden() -> Golden {
        let man: serde_json::Value = serde_json::from_slice(&std::fs::read(dir().join("manifest.json")).unwrap()).unwrap();
        let u = |k: &str| man[k].as_u64().unwrap_or_else(|| panic!("manifest: {k}")) as usize;
        let (anchors, hidden, streams, vocab) = (u("anchors"), u("hidden"), u("hc_mult"), u("vocab"));
        Golden {
            x: f32s("x.f32", anchors * streams * hidden),
            normed: f32s("normed.f32", anchors * hidden),
            logits: f32s("logits.f32", anchors * vocab),
            man,
            anchors,
            hidden,
            streams,
            vocab,
        }
    }

    pub fn seed(g: &Golden, k: &str) -> u64 {
        g.man[k].as_u64().unwrap_or_else(|| panic!("manifest: {k}"))
    }
}

#[cfg(test)]
mod tests {
    use super::testkit::*;
    use super::*;

    /// G3 head part (`docs/glm5-head.md`): cosine per anchor, fixed before any result
    const COS_MIN: f64 = 0.9999;

    fn geo() -> HeadGeo {
        HeadGeo::of(&Glm5Geo::GLM_5_3_FLASH)
    }

    #[test]
    fn head_geo_is_the_family_row() {
        let g = geo();
        assert_eq!(g, HeadGeo { hidden: 4096, streams: 4, vocab: 154_880, eps: 1e-5, one_plus_w: false });
        assert_eq!(g.lm_head_bytes(), 1_268_776_960, "154,880 x 4096 x 2 B (#165 Evidence)");
    }

    #[test]
    #[should_panic(expected = "untied lm_head")]
    fn a_tied_checkpoint_is_refused_by_name() {
        HeadGeo::of(&Glm5Geo { tie_word_embeddings: true, ..Glm5Geo::GLM_5_3_FLASH });
    }

    /// exact small cases: the mean (not the sum, not stream 0), eps added inside the root,
    /// `w * x_hat` (not `(1 + w) * x_hat`)
    #[test]
    fn stream_mean_rms_ref_is_mean_then_weighted_rms() {
        let g = HeadGeo { hidden: 4, streams: 4, vocab: 1, eps: 0.0, one_plus_w: false };
        // streams of a row: means [1, -1, 3, -3] (sum would be x4, stream 0 is [4, 0, 0, 0])
        #[rustfmt::skip]
        let x = [4.0, 0.0, 0.0, 0.0,   0.0, -4.0, 6.0, 0.0,   0.0, 0.0, 6.0, -6.0,   0.0, 0.0, 0.0, -6.0f32];
        let w = [1.0, 2.0, 0.5, 0.25f32];
        // mean(m^2) = (1 + 1 + 9 + 9) / 4 = 5
        let r = 1.0 / 5f64.sqrt();
        let want = [r, -2.0 * r, 1.5 * r, -0.75 * r];
        let got = stream_mean_rms_ref(&x, &w, &g);
        for (a, b) in got.iter().zip(want) {
            assert!((a - b).abs() < 1e-15, "{got:?} vs {want:?}");
        }
        let ge = HeadGeo { eps: 1e-5, ..g };
        let re = 1.0 / (5.0 + 1e-5f32 as f64).sqrt();
        assert!((stream_mean_rms_ref(&x, &w, &ge)[0] - re).abs() < 1e-15);
        let g1 = HeadGeo { one_plus_w: true, ..g };
        assert!((stream_mean_rms_ref(&x, &w, &g1)[1] - (-3.0 * r)).abs() < 1e-15);
    }

    /// the Rust generator is the oracle's, bit for bit (reference value: splitmix64's first
    /// output for seed 0 is 0xE220A8397B1DCDAF)
    #[test]
    fn the_weight_generator_is_the_oracles() {
        assert_eq!(splitmix64(0, 0), 0xE220_A839_7B1D_CDAF);
        let gd = golden();
        for s in gd.man["hash_samples"].as_array().unwrap() {
            let (sd, i) = (s["seed"].as_u64().unwrap(), s["i"].as_u64().unwrap());
            assert_eq!(splitmix64(sd, i).to_string(), s["u64"].as_str().unwrap(), "seed {sd} i {i}");
        }
        let lm = seed(&gd, "seed_lm");
        for s in gd.man["lm_head_samples"].as_array().unwrap() {
            let (r, c) = (s["r"].as_u64().unwrap(), s["c"].as_u64().unwrap());
            let v = bf16_to_f32(lm_head_bf16(lm, r * gd.hidden as u64 + c));
            assert_eq!(v as f64, s["v"].as_f64().unwrap(), "lm_head[{r}][{c}]");
        }
        let w = norm_weight(seed(&gd, "seed_norm"), gd.hidden);
        for s in gd.man["norm_weight_samples"].as_array().unwrap() {
            let d = s["d"].as_u64().unwrap() as usize;
            assert_eq!(w[d] as f64, s["v"].as_f64().unwrap(), "norm_w[{d}]");
        }
    }

    /// the fixture is the real shape and the family row's numbers
    #[test]
    fn the_golden_has_the_real_shapes() {
        let gd = golden();
        let g = geo();
        assert_eq!((gd.hidden, gd.streams, gd.vocab), (g.hidden, g.streams, g.vocab));
        assert_eq!(gd.man["rms_norm_eps"].as_f64().unwrap() as f32, g.eps);
        assert_eq!(gd.man["transformers"].as_str().unwrap(), "5.16.1");
    }

    /// G3 on the host: the f64 twin against HF's modules per anchor; the normed hidden in full,
    /// the logits on every 97th vocab row, the last row and the golden's top-1 rows (the full
    /// vocab runs on the GPU test)
    #[test]
    fn the_host_head_matches_the_oracle_golden() {
        let gd = golden();
        let g = geo();
        let w = norm_weight(seed(&gd, "seed_norm"), g.hidden);
        let normed = stream_mean_rms_ref(&gd.x, &w, &g);
        let top1: Vec<usize> = gd.man["top1"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let mut rows: Vec<usize> = (0..g.vocab).step_by(97).chain([g.vocab - 1]).chain(top1.iter().copied()).collect();
        rows.sort_unstable();
        rows.dedup();
        let lm = seed(&gd, "seed_lm");
        let lm_rows: Vec<Vec<u16>> = rows.iter().map(|&r| lm_head_rows(lm, g.hidden, r, r + 1)).collect();
        for a in 0..gd.anchors {
            let n = &normed[a * g.hidden..(a + 1) * g.hidden];
            let want: Vec<f64> = gd.normed[a * g.hidden..(a + 1) * g.hidden].iter().map(|&v| v as f64).collect();
            let (cn, mn) = (cosine(n, &want), max_abs(n, &want));
            let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
            let got_l: Vec<f64> = lm_rows.iter().map(|row| logit_ref(n, row)).collect();
            let want_l: Vec<f64> = rows.iter().map(|&r| gd.logits[a * g.vocab + r] as f64).collect();
            let (cl, ml) = (cosine(&got_l, &want_l), max_abs(&got_l, &want_l));
            println!("anchor {a}: normed cos {cn:.9} max_abs {mn:.2e}; logits ({} rows) cos {cl:.9} max_abs {ml:.2e}", rows.len());
            assert!(cn >= COS_MIN && cl >= COS_MIN, "anchor {a}: cosine normed {cn} logits {cl} < {COS_MIN}");
            // cosine is scale-blind; the norm's scale (eps, mean vs sum) is held by max_abs
            assert!(mn <= 1e-5 * scale, "anchor {a}: normed max_abs {mn:.3e} > 1e-5 x {scale:.3e}");
            assert!(ml <= 1e-3, "anchor {a}: logits max_abs {ml:.3e}");
            let i = rows.binary_search(&top1[a]).unwrap();
            assert!((got_l[i] - want_l[i]).abs() <= 1e-3, "anchor {a}: top-1 row {}", top1[a]);
        }
    }

    /// the kernel source compiles alone (NVRTC, host only, no GPU) with every entry `NAMES` lists
    #[test]
    fn glm5_head_source_compiles_with_every_entry() {
        let ptx = crate::kernels::tests_300_c4::ptx(crate::kernels::GLM5_HEAD_SRC);
        let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), glm5_head::NAMES.len(), "{names:?}");
        for n in glm5_head::NAMES {
            assert!(names.iter().any(|e| e == n), "{n} missing from {names:?}");
        }
    }
}

#[cfg(test)]
mod tests_gpu {
    //! #165 on the GPU: the kernel against the host twin, and the whole head (kernel +
    //! `gemv_bf16_w` + `argmax_k`) against the oracle goldens at the real shapes and the full
    //! vocab (a 1.27 GB synthetic lm_head in VRAM). `#[ignore]`: CI has no GPU. Run with
    //! `cargo test --release --lib glm5_head_gpu -- --ignored --nocapture --test-threads 1`.
    use super::testkit::*;
    use super::*;
    use crate::kernels::KernelGeo;
    use crate::sample::Rng;

    fn fill(n: usize, seed: u64, scale: f32, offset: f32) -> Vec<f32> {
        let mut r = Rng::new(seed);
        (0..n).map(|_| ((r.next_f64() * 2.0 - 1.0) as f32) * scale + offset).collect()
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_head_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_head_gpu_kernel_matches_the_host_twin() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            for (hidden, streams, rows, eps, one_plus_w) in
                [(4096, 4, 5, 1e-5f32, false), (4096, 4, 3, 1e-5, true), (256, 4, 7, 1e-6, false), (512, 3, 2, 1e-5, false)]
            {
                let g = HeadGeo { hidden, streams, vocab: 8, eps, one_plus_w };
                let mut head = Head::new(g);
                let mut x = fill(rows * streams * hidden, 0x165 + hidden as u64, 2.0, 0.3);
                // row 0 tiny: mean(m^2) ~ eps, so eps and the mean's scale are visible
                for v in &mut x[..streams * hidden] {
                    *v *= 2e-3;
                }
                let w = fill(hidden, 0x1650, 0.5, 1.0);
                let want = stream_mean_rms_ref(&x, &w, &g);
                let (mut xd, mut wd, mut od) = (cuda::to_f32_dev(&x), cuda::to_f32_dev(&w), cuda::alloc_zeroed(rows * hidden * 4));
                head.stream_mean_rms(xd, wd, od, rows);
                cuda::sync();
                let got = cuda::dtoh(od, rows * hidden);
                let rel = got.iter().zip(&want).map(|(&a, &b)| (a as f64 - b).abs() / b.abs().max(1e-3)).fold(0.0, f64::max);
                let scale = want.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                let ma = got.iter().zip(&want).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
                println!("H {hidden} S {streams} rows {rows} eps {eps:e} 1+w {one_plus_w}: max_abs {ma:.3e} (scale {scale:.3e}), max rel {rel:.3e}");
                assert!(ma <= 1e-5 * scale, "max_abs {ma:.3e} > 1e-5 x {scale:.3e}");
                for d in [&mut xd, &mut wd, &mut od] {
                    cuda::free_dev(d);
                }
                head.free();
            }
        }
    }

    /// G3 head part: fed with the golden's input, the engine's normed hidden and logits reach
    /// cosine >= 0.9999 per anchor against HF's modules, and the greedy id is the golden's top-1
    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_head_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_head_gpu_matches_the_oracle_golden() {
        let gd = golden();
        let g = HeadGeo::of(&Glm5Geo::GLM_5_3_FLASH);
        assert_eq!((gd.hidden, gd.streams, gd.vocab), (g.hidden, g.streams, g.vocab));
        let w = norm_weight(seed(&gd, "seed_norm"), g.hidden);
        let t0 = std::time::Instant::now();
        let lm = lm_head_rows(seed(&gd, "seed_lm"), g.hidden, 0, g.vocab);
        println!("synthetic lm_head {} B generated in {:.1} s", lm.len() * 2, t0.elapsed().as_secs_f64());
        let top1: Vec<i32> = gd.man["top1"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap() as i32).collect();
        unsafe {
            let _ctx = cuda::Ctx::init();
            let mut module = cuda::compile(&KernelGeo::flash_next().source());
            let k = Kernels::new(&module, false);
            let mut head = Head::new(g);
            let n = gd.anchors;
            let mut xd = cuda::to_f32_dev(&gd.x);
            let mut wd = cuda::to_f32_dev(&w);
            let mut lmd = cuda::upload_dev(std::slice::from_raw_parts(lm.as_ptr() as *const u8, lm.len() * 2));
            drop(lm);
            let mut nd = cuda::alloc_zeroed(n * g.hidden * 4);
            let mut ld = cuda::alloc_zeroed(n * g.vocab * 4);
            let mut idd = cuda::alloc_zeroed(n * 4);
            head.run(&k, xd, wd, lmd, nd, ld, n);
            head.argmax(&k, ld, idd, n);
            cuda::sync();
            let normed = cuda::dtoh(nd, n * g.hidden);
            let logits = cuda::dtoh(ld, n * g.vocab);
            let ids = cuda::dtoh_i32(idd, n);
            let to64 = |v: &[f32]| v.iter().map(|&x| x as f64).collect::<Vec<f64>>();
            for a in 0..n {
                let (h0, h1, v0, v1) = (a * g.hidden, (a + 1) * g.hidden, a * g.vocab, (a + 1) * g.vocab);
                let (gn, wn) = (to64(&normed[h0..h1]), to64(&gd.normed[h0..h1]));
                let (gl, wl) = (to64(&logits[v0..v1]), to64(&gd.logits[v0..v1]));
                let (cn, cl) = (cosine(&gn, &wn), cosine(&gl, &wl));
                let (mn, ml) = (max_abs(&gn, &wn), max_abs(&gl, &wl));
                println!("anchor {a}: normed cos {cn:.9} max_abs {mn:.2e}; logits cos {cl:.9} max_abs {ml:.2e}; top-1 {} (golden {})", ids[a], top1[a]);
                assert!(cn >= 0.9999 && cl >= 0.9999, "anchor {a}: cosine normed {cn} logits {cl}");
                // cosine is scale-blind; the norm's scale (eps, mean vs sum) is held by max_abs
                let scale = wn.iter().fold(0.0f64, |m, v| m.max(v.abs()));
                assert!(mn <= 1e-5 * scale && ml <= 1e-3, "anchor {a}: max_abs normed {mn:.3e} (scale {scale:.3e}) logits {ml:.3e}");
                assert_eq!(ids[a], top1[a], "anchor {a}: greedy id");
            }
            for d in [&mut xd, &mut wd, &mut lmd, &mut nd, &mut ld, &mut idd] {
                cuda::free_dev(d);
            }
            head.free();
            module.unload();
        }
    }
}
