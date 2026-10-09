//! #161 (GLM-5.3-Flash plan step 14, mHC part): the manifold-constrained hyper-connection (mHC)
//! residual of every glm5_next decoder layer — two sites per layer (`attn_hc`, `ffn_hc`), four
//! residual streams, a Sinkhorn-projected 4x4 stream mixer.
//!
//! The math is HF's `Glm5NextTextHyperConnection.forward` (transformers 5.16.1,
//! `modeling_glm5_next.py:267-295`) and the decoder layer's stream mix (`:1316-1318`), written out
//! in `docs/glm5-next-recipe.md` section 5; the engine side is `docs/glm5-mhc.md`. Per row t, with
//! `X` the four streams `[4][H]` (H 4096, 16,384 values):
//!
//! 1. `r = rsqrt(mean(X²) + 1e-5)` (unweighted RMSNorm, eps = `rms_norm_eps`, not `hc_eps`)
//! 2. `m = fn · (r X)`, 24 values = 4 pre + 4 post + 16 comb (`fn` `[24][4H]` BF16)
//! 3. `pre = σ(m_pre·scale0 + base[0:4]) + 1e-6`, `post = 2·σ(m_post·scale1 + base[4:8])`
//! 4. `comb = softmax_i(m_comb[j][i]·scale2 + base[8 + 4j + i]) + 1e-6`, one column normalisation,
//!    then 19 (`hc_sinkhorn_iters − 1`) rounds of row then column normalisation, every divisor + 1e-6
//! 5. `collapsed = Σ_s pre[s]·X[s]` (the sublayer input, before `input_layernorm` /
//!    `post_attention_layernorm`)
//! 6. after the sublayer output `y`: `X'[i] = post[i]·y + Σ_j comb[j][i]·X[j]`
//!
//! Two halves: the CPU twin (the functions below, f32 per element, f64 for the two 4H-long sums)
//! and the GPU half (`Kernels`, `Plan`) on `kernels_glm5_mhc.cu`, its own NVRTC module. Nothing in
//! the engine calls either yet: the layer driver integrates it (`Residual::Mhc`, plan step 11/14).
//!
//! The decoder layer of HF's BF16 model casts `post` and `comb` to BF16 and mixes in BF16; the
//! reference runner (`oracle/glm5_layerwise.py`, #158) runs f32, and so does this module.

use crate::cuda;
use crate::kernels::launch_v;
use cudarc::driver::sys::{CUdeviceptr, CUfunction};

/// residual streams (`hc_mult`)
pub const HC: usize = 4;
/// mixing logits per row: (2 + hc_mult) · hc_mult = 4 pre + 4 post + 16 comb
pub const MIX: usize = (2 + HC) * HC;
/// `hc_eps`
pub const HC_EPS: f32 = 1e-6;
/// `rms_norm_eps`, the eps of the unweighted RMSNorm in front of `fn`
pub const RMS_EPS: f32 = 1e-5;
/// `hc_sinkhorn_iters`: one column normalisation after the softmax, then 19 row + column rounds
pub const SINKHORN_ITERS: usize = 20;
/// threads per block of both kernels (`MHC_THREADS`)
pub const THREADS: usize = 256;

/// The three mHC outputs of one row besides `collapsed`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Coeffs {
    /// stream collapse weights, `σ(·) + 1e-6`
    pub pre: [f32; HC],
    /// sublayer-output placement, `2·σ(·)`, range [0, 2]
    pub post: [f32; HC],
    /// `comb[j][i]`: row j = source stream, column i = destination stream (doubly stochastic)
    pub comb: [[f32; HC]; HC],
}

pub fn bf16_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

fn sigmoid(v: f32) -> f32 {
    1.0 / (1.0 + (-v).exp())
}

/// Steps 1-2: the 24 mixing logits of one row. `x` `[4][h]`, `fn_` `[24][4h]` BF16 bits.
pub fn logits(x: &[f32], fn_: &[u16]) -> [f32; MIX] {
    let n = x.len();
    assert!(n > 0 && n % HC == 0 && fn_.len() == MIX * n, "mhc: x {} / fn {} do not fit [4][h] / [24][4h]", n, fn_.len());
    let ss: f64 = x.iter().map(|&v| v as f64 * v as f64).sum();
    let r = 1.0 / ((ss as f32 / n as f32) + RMS_EPS).sqrt();
    let xn: Vec<f32> = x.iter().map(|&v| v * r).collect();
    let mut m = [0f32; MIX];
    for (k, mk) in m.iter_mut().enumerate() {
        let row = &fn_[k * n..(k + 1) * n];
        *mk = row.iter().zip(&xn).map(|(&w, &v)| bf16_to_f32(w) as f64 * v as f64).sum::<f64>() as f32;
    }
    m
}

/// Steps 3-4: `pre`, `post`, `comb` from the logits, f32 in HF's order of operations.
pub fn coeffs(m: &[f32; MIX], base: &[f32; MIX], scale: &[f32; 3]) -> Coeffs {
    let mut c = Coeffs { pre: [0.0; HC], post: [0.0; HC], comb: [[0.0; HC]; HC] };
    for s in 0..HC {
        c.pre[s] = sigmoid(m[s] * scale[0] + base[s]) + HC_EPS;
        c.post[s] = 2.0 * sigmoid(m[HC + s] * scale[1] + base[HC + s]);
    }
    for j in 0..HC {
        let mut l = [0f32; HC];
        for i in 0..HC {
            let q = 2 * HC + j * HC + i;
            l[i] = m[q] * scale[2] + base[q];
        }
        let mx = l.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let mut sum = 0f32;
        for v in l.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        for i in 0..HC {
            c.comb[j][i] = l[i] / sum + HC_EPS;
        }
    }
    sinkhorn(&mut c.comb);
    c
}

/// The column normalisation, then `SINKHORN_ITERS − 1` rounds of rows then columns, each divisor
/// `+ HC_EPS` (`modeling_glm5_next.py:287-290`): a fixed count, not "to convergence".
fn sinkhorn(c: &mut [[f32; HC]; HC]) {
    let cols = |c: &mut [[f32; HC]; HC]| {
        for i in 0..HC {
            let s = (0..HC).fold(0f32, |a, j| a + c[j][i]) + HC_EPS;
            for row in c.iter_mut() {
                row[i] /= s;
            }
        }
    };
    cols(c);
    for _ in 1..SINKHORN_ITERS {
        for row in c.iter_mut() {
            let s = row.iter().fold(0f32, |a, &v| a + v) + HC_EPS;
            for v in row.iter_mut() {
                *v /= s;
            }
        }
        cols(c);
    }
}

/// Step 5: `out[d] = Σ_s pre[s]·x[s][d]`; `x` `[4][h]`, `out` `[h]`.
pub fn collapse(x: &[f32], pre: &[f32; HC], out: &mut [f32]) {
    let h = out.len();
    assert_eq!(x.len(), HC * h);
    for (d, o) in out.iter_mut().enumerate() {
        *o = (0..HC).fold(0f32, |a, s| a + pre[s] * x[s * h + d]);
    }
}

/// Step 6: `out[i][d] = post[i]·y[d] + Σ_j comb[j][i]·x[j][d]`; `x`, `out` `[4][h]`, `y` `[h]`.
pub fn expand(x: &[f32], y: &[f32], c: &Coeffs, out: &mut [f32]) {
    let h = y.len();
    assert!(x.len() == HC * h && out.len() == HC * h);
    for d in 0..h {
        for i in 0..HC {
            let mix = (0..HC).fold(0f32, |a, j| a + c.comb[j][i] * x[j * h + d]);
            out[i * h + d] = c.post[i] * y[d] + mix;
        }
    }
}

/// Steps 1-5 for `t` rows: `x` `[t][4][h]` -> one `Coeffs` per row and `collapsed` `[t][h]`.
pub fn site(x: &[f32], fn_: &[u16], base: &[f32; MIX], scale: &[f32; 3], h: usize) -> (Vec<Coeffs>, Vec<f32>) {
    assert_eq!(x.len() % (HC * h), 0);
    let t = x.len() / (HC * h);
    let mut collapsed = vec![0f32; t * h];
    let cs = (0..t)
        .map(|r| {
            let xr = &x[r * HC * h..(r + 1) * HC * h];
            let c = coeffs(&logits(xr, fn_), base, scale);
            collapse(xr, &c.pre, &mut collapsed[r * h..(r + 1) * h]);
            c
        })
        .collect();
    (cs, collapsed)
}

// ---------------- the GPU half (`kernels_glm5_mhc.cu`) ----------------

/// every entry of `kernels::GLM5_MHC_SRC`
pub const NAMES: &[&str] = &["glm5_mhc_coeffs", "glm5_mhc_expand", "glm5_mhc_mix"];

/// the compiled module and its entries
pub struct Kernels {
    pub module: cuda::Module,
    expand: CUfunction,
    /// #191: steps 1-5 over (24, T) blocks, the last block of a row runs 3-5 and the collapse
    mix: CUfunction,
}

impl Kernels {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new() -> Kernels {
        let module = cuda::compile(crate::kernels::GLM5_MHC_SRC);
        Kernels {
            expand: module.get("glm5_mhc_expand"),
            mix: module.get("glm5_mhc_mix"),
            module,
        }
    }
}

/// The device weights of one site (`layers.L.hc_{attn,ffn}_{fn,base,scale}`): `fn_` `[24][4h]`
/// BF16, `base` `[24]` f32, `scale` `[3]` f32 (the container stores base and scale as F32).
#[derive(Clone, Copy, Debug)]
pub struct SiteDev {
    pub fn_: CUdeviceptr,
    pub base: CUdeviceptr,
    pub scale: CUdeviceptr,
}

impl SiteDev {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn upload(fn_: &[u16], base: &[f32; MIX], scale: &[f32; 3]) -> SiteDev {
        SiteDev { fn_: cuda::to_dev(fn_), base: cuda::to_f32_dev(base), scale: cuda::to_f32_dev(scale) }
    }

    /// # Safety
    /// No launch reading these weights is pending.
    pub unsafe fn free(&mut self) {
        for d in [&mut self.fn_, &mut self.base, &mut self.scale] {
            cuda::free_dev(d);
        }
    }
}

/// The launch parameters and the coefficient scratch of one width `h` for up to `max_tokens`
/// rows. One plan serves both sites of every layer: `coeffs` writes `pre`, `post`, `comb`, the
/// sublayer runs, `expand` reads them, then the next site's `coeffs` overwrites them.
pub struct Plan {
    pub h: usize,
    pub max_tokens: usize,
    prm: CUdeviceptr,
    /// `[T][24]` f32
    pub logits: CUdeviceptr,
    /// `[T][4]` f32
    pub pre: CUdeviceptr,
    /// `[T][4]` f32
    pub post: CUdeviceptr,
    /// `[T][4][4]` f32, `[j][i]` = source j, destination i
    pub comb: CUdeviceptr,
    /// #191: `[T]` u32, zero between launches: the blocks of a row that finished `glm5_mhc_mix`
    done: CUdeviceptr,
}

impl Plan {
    /// # Safety
    /// A CUDA context is current.
    pub unsafe fn new(h: usize, max_tokens: usize) -> Plan {
        assert!(h > 0 && max_tokens > 0, "mhc: h {h}, max_tokens {max_tokens}");
        let h32 = i32::try_from(HC * h).map(|_| h as i32).expect("mhc: 4h beyond i32");
        Plan {
            h,
            max_tokens,
            prm: cuda::to_i32_dev(&[h32]),
            logits: cuda::alloc_zeroed(max_tokens * MIX * 4),
            pre: cuda::alloc_zeroed(max_tokens * HC * 4),
            post: cuda::alloc_zeroed(max_tokens * HC * 4),
            comb: cuda::alloc_zeroed(max_tokens * HC * HC * 4),
            done: cuda::alloc_zeroed(max_tokens * 4),
        }
    }

    /// Queue steps 1-5 for `t` rows on the current stream: `x` `[t][4][h]` f32 ->
    /// `collapsed` `[t][h]` f32; `logits`, `pre`, `post`, `comb` stay in the plan. One launch
    /// (#191: `glm5_mhc_mix` over (24, t) blocks, bit-identical to the record `glm5_mhc_coeffs`,
    /// which ran one block per row).
    ///
    /// # Safety
    /// `x`, `collapsed` are device buffers of those shapes; `w` is live.
    pub unsafe fn coeffs(&self, kn: &Kernels, w: &SiteDev, x: CUdeviceptr, collapsed: CUdeviceptr, t: usize) {
        assert!((1..=self.max_tokens).contains(&t), "mhc: {t} rows (1..={})", self.max_tokens);
        launch_v(kn.mix, MIX as u32, t as u32, 1, THREADS as u32, &[
            x, w.fn_, w.base, w.scale, self.logits, self.pre, self.post, self.comb, collapsed, self.done, self.prm]);
    }

    /// Queue step 6 for `t` rows with the coefficients of the last `coeffs`: `x` `[t][4][h]` (the
    /// residual the coefficients were computed from), `y` `[t][h]` the sublayer output ->
    /// `out` `[t][4][h]`. `out == x` updates the streams in place. One launch.
    ///
    /// # Safety
    /// `x`, `y`, `out` are device buffers of those shapes.
    pub unsafe fn expand(&self, kn: &Kernels, x: CUdeviceptr, y: CUdeviceptr, out: CUdeviceptr, t: usize) {
        assert!((1..=self.max_tokens).contains(&t), "mhc: {t} rows (1..={})", self.max_tokens);
        launch_v(kn.expand, self.h.div_ceil(THREADS) as u32, t as u32, 1, THREADS as u32, &[
            x, y, self.post, self.comb, out, self.prm]);
    }

    /// # Safety
    /// No launch of this plan is pending.
    pub unsafe fn free(&mut self) {
        for d in [&mut self.prm, &mut self.logits, &mut self.pre, &mut self.post, &mut self.comb, &mut self.done] {
            cuda::free_dev(d);
        }
    }
}

/// Fixture readers and the comparison metrics shared by the CPU and the GPU tests.
#[cfg(test)]
pub(crate) mod testkit {
    use std::path::PathBuf;

    pub(crate) fn dir() -> PathBuf {
        PathBuf::from(concat!(env!("CARGO_MANIFEST_DIR"), "/tests/fixtures/glm5/mhc"))
    }

    pub(crate) fn raw(name: &str) -> Vec<u8> {
        std::fs::read(dir().join("real").join(name)).unwrap_or_else(|e| panic!("{name}: {e}"))
    }

    pub(crate) fn f32s(name: &str) -> Vec<f32> {
        raw(name).chunks_exact(4).map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]])).collect()
    }

    pub(crate) fn u16s(name: &str) -> Vec<u16> {
        raw(name).chunks_exact(2).map(|b| u16::from_le_bytes([b[0], b[1]])).collect()
    }

    pub(crate) fn manifest() -> serde_json::Value {
        serde_json::from_slice(&raw("manifest.json")).expect("manifest.json")
    }

    /// the real-shape fixture: (T, H, fn, base, scale, x, y)
    pub(crate) struct Real {
        pub t: usize,
        pub h: usize,
        pub fn_: Vec<u16>,
        pub base: [f32; super::MIX],
        pub scale: [f32; 3],
        pub x: Vec<f32>,
        pub y: Vec<f32>,
    }

    pub(crate) fn real() -> Real {
        let m = manifest();
        Real {
            t: m["T"].as_u64().unwrap() as usize,
            h: m["hidden_size"].as_u64().unwrap() as usize,
            fn_: u16s("fn.bf16"),
            base: f32s("base.f32").try_into().unwrap(),
            scale: f32s("scale.f32").try_into().unwrap(),
            x: f32s("x.f32"),
            y: f32s("y.f32"),
        }
    }

    pub(crate) struct Case {
        pub name: String,
        pub h: usize,
        pub t: usize,
        pub v: serde_json::Value,
    }

    impl Case {
        pub(crate) fn get(&self, k: &str) -> Vec<f32> {
            self.v[k].as_array().unwrap_or_else(|| panic!("{}: no {k}", self.name)).iter().map(|x| x.as_f64().unwrap() as f32).collect()
        }
        pub(crate) fn fn_bf16(&self) -> Vec<u16> {
            self.get("fn")
                .iter()
                .map(|&v| {
                    assert_eq!(v.to_bits() & 0xffff, 0, "{}: fn value {v} is not BF16", self.name);
                    (v.to_bits() >> 16) as u16
                })
                .collect()
        }
    }

    pub(crate) fn tiny() -> Vec<Case> {
        let v: serde_json::Value = serde_json::from_slice(&std::fs::read(dir().join("tiny.json")).expect("tiny.json")).unwrap();
        v["cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Case {
                name: c["name"].as_str().unwrap().to_string(),
                h: c["hidden"].as_u64().unwrap() as usize,
                t: c["T"].as_u64().unwrap() as usize,
                v: c.clone(),
            })
            .collect()
    }

    pub(crate) fn cosine(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        let (mut ab, mut aa, mut bb) = (0f64, 0f64, 0f64);
        for (&x, &y) in a.iter().zip(b) {
            ab += x as f64 * y as f64;
            aa += x as f64 * x as f64;
            bb += y as f64 * y as f64;
        }
        ab / (aa.sqrt() * bb.sqrt())
    }

    pub(crate) fn max_abs(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs()).fold(0.0, f64::max)
    }

    /// max over elements of |a − b| / (1 + |b|): the exact-op measure (values up to ~30 here)
    pub(crate) fn max_rel1(a: &[f32], b: &[f32]) -> f64 {
        assert_eq!(a.len(), b.len());
        a.iter().zip(b).map(|(&x, &y)| (x as f64 - y as f64).abs() / (1.0 + (y as f64).abs())).fold(0.0, f64::max)
    }

    pub(crate) fn flat(c: &super::Coeffs) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
        (c.pre.to_vec(), c.post.to_vec(), c.comb.iter().flatten().copied().collect())
    }
}

#[cfg(test)]
mod tests {
    //! #161: the CPU twin against HF's module (goldens of `oracle/export_glm5_mhc_golden.py`,
    //! synthetic weights): exact-op on tiny shapes, gate G3 (cosine >= 0.9999 per row) on the real
    //! block shapes, and the kernel source compiling with every entry (host NVRTC, no GPU).
    use super::testkit::*;
    use super::*;

    /// exact-op bound: f32 per element, only the summation order and libm's exp differ from torch
    const EXACT: f64 = 4e-6;
    /// gate G3 (plan, "Tore"): per-layer cosine against the golden
    const G3: f64 = 0.9999;

    #[test]
    fn glm5_mhc_fixture_matches_its_manifest() {
        use sha2::{Digest, Sha256};
        let m = manifest();
        assert_eq!(m["hc_mult"].as_u64(), Some(HC as u64));
        assert_eq!(m["hc_sinkhorn_iters"].as_u64(), Some(SINKHORN_ITERS as u64));
        assert_eq!(m["hc_eps"].as_f64().map(|v| v as f32), Some(HC_EPS));
        assert_eq!(m["rms_norm_eps"].as_f64().map(|v| v as f32), Some(RMS_EPS));
        let files = m["files"].as_object().unwrap();
        assert_eq!(files.len(), 11, "{:?}", files.keys().collect::<Vec<_>>());
        for (name, f) in files {
            let d: String = Sha256::digest(raw(name)).iter().map(|b| format!("{b:02x}")).collect();
            assert_eq!(d, f["sha256"].as_str().unwrap(), "{name}");
        }
    }

    #[test]
    fn glm5_mhc_tiny_cases_are_hf_exact_ops() {
        for c in tiny() {
            let (fn_, base, scale) = (c.fn_bf16(), c.get("base"), c.get("scale"));
            let (base, scale): ([f32; MIX], [f32; 3]) = (base.try_into().unwrap(), scale.try_into().unwrap());
            let (x, y, gl) = (c.get("x"), c.get("y"), c.get("logits"));
            let n = HC * c.h;
            for r in 0..c.t {
                let xr = &x[r * n..(r + 1) * n];
                let m = logits(xr, &fn_);
                let e = max_rel1(&m, &gl[r * MIX..(r + 1) * MIX]);
                assert!(e <= EXACT, "{} row {r}: logits {e:.2e}", c.name);
                // the coefficient tail from the golden logits: the Sinkhorn twin alone
                let g: [f32; MIX] = gl[r * MIX..(r + 1) * MIX].try_into().unwrap();
                let co = coeffs(&g, &base, &scale);
                let (pre, post, comb) = flat(&co);
                for (what, got, k) in [("pre", &pre, HC), ("post", &post, HC), ("comb", &comb, HC * HC)] {
                    let want = c.get(what);
                    let e = max_rel1(got, &want[r * k..(r + 1) * k]);
                    assert!(e <= EXACT, "{} row {r}: {what} {e:.2e} ({got:?} vs {:?})", c.name, &want[r * k..(r + 1) * k]);
                }
                let mut col = vec![0f32; c.h];
                collapse(xr, &co.pre, &mut col);
                let e = max_rel1(&col, &c.get("collapsed")[r * c.h..(r + 1) * c.h]);
                assert!(e <= EXACT, "{} row {r}: collapsed {e:.2e}", c.name);
                let mut out = vec![0f32; n];
                expand(xr, &y[r * c.h..(r + 1) * c.h], &co, &mut out);
                let e = max_rel1(&out, &c.get("expanded")[r * n..(r + 1) * n]);
                assert!(e <= EXACT, "{} row {r}: expanded {e:.2e}", c.name);
            }
        }
    }

    #[test]
    fn glm5_mhc_real_shapes_meet_g3() {
        let f = real();
        let (cs, col) = site(&f.x, &f.fn_, &f.base, &f.scale, f.h);
        let (gpre, gpost, gcomb) = (f32s("pre.f32"), f32s("post.f32"), f32s("comb.f32"));
        let (gcol, gexp) = (f32s("collapsed.f32"), f32s("expanded.f32"));
        let n = HC * f.h;
        let mut worst = 1f64;
        for (r, c) in cs.iter().enumerate() {
            let (pre, post, comb) = flat(c);
            let ma = [
                max_abs(&pre, &gpre[r * HC..(r + 1) * HC]),
                max_abs(&post, &gpost[r * HC..(r + 1) * HC]),
                max_abs(&comb, &gcomb[r * HC * HC..(r + 1) * HC * HC]),
            ];
            let mut out = vec![0f32; n];
            expand(&f.x[r * n..(r + 1) * n], &f.y[r * f.h..(r + 1) * f.h], c, &mut out);
            let cc = cosine(&col[r * f.h..(r + 1) * f.h], &gcol[r * f.h..(r + 1) * f.h]);
            let ce = cosine(&out, &gexp[r * n..(r + 1) * n]);
            eprintln!("mhc real row {r}: 1-cos collapsed {:.2e} expanded {:.2e}, max_abs pre {:.2e} post {:.2e} comb {:.2e}", 1.0 - cc, 1.0 - ce, ma[0], ma[1], ma[2]);
            assert!(cc >= G3 && ce >= G3, "row {r}: cosine collapsed {cc} expanded {ce} < {G3}");
            // the coefficients carry the bound of their own inputs: 24 dot products of 16,384 terms
            assert!(ma.iter().all(|&e| e <= 1e-5), "row {r}: pre/post/comb max_abs {ma:?}");
            worst = worst.min(cc).min(ce);
        }
        eprintln!("mhc real: worst 1-cos {:.2e} over {} rows", 1.0 - worst, cs.len());
    }

    #[test]
    fn glm5_mhc_source_compiles_with_every_entry() {
        let src = crate::kernels::GLM5_MHC_SRC;
        let ptx = crate::kernels::tests_300_c4::ptx(src);
        let names: Vec<String> = crate::kernels::tests_300_c4::entries(&ptx).into_iter().map(|(n, _)| n).collect();
        assert_eq!(names.len(), NAMES.len(), "{names:?}");
        for n in NAMES {
            assert!(names.iter().any(|m| m == n), "{n} missing from {names:?}");
        }
        let def = |name: &str| -> String {
            let pat = format!("#define {name} ");
            let i = src.find(&pat).unwrap_or_else(|| panic!("no #define {name}")) + pat.len();
            src[i..].split_whitespace().next().unwrap().to_string()
        };
        assert_eq!(def("MHC_HC"), HC.to_string());
        assert_eq!(def("MHC_MIX"), MIX.to_string());
        assert_eq!(def("MHC_THREADS"), THREADS.to_string());
        assert_eq!(def("MHC_SINKHORN_ITERS"), SINKHORN_ITERS.to_string());
        assert_eq!(def("MHC_HC_EPS").trim_end_matches('f').parse::<f32>().unwrap(), HC_EPS);
        assert_eq!(def("MHC_RMS_EPS").trim_end_matches('f').parse::<f32>().unwrap(), RMS_EPS);
    }
}

#[cfg(test)]
mod tests_gpu {
    //! #161 on the GPU (RTX 5090, sm_120): the two kernels against the same goldens as the CPU
    //! twin, and the in-place expand bit-identical to the out-of-place one. `#[ignore]`: CI has no
    //! GPU. Run with `cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1`.
    use super::testkit::*;
    use super::*;

    struct Out {
        logits: Vec<f32>,
        pre: Vec<f32>,
        post: Vec<f32>,
        comb: Vec<f32>,
        collapsed: Vec<f32>,
        expanded: Vec<f32>,
        in_place: Vec<f32>,
    }

    unsafe fn run(kn: &Kernels, fn_: &[u16], base: &[f32; MIX], scale: &[f32; 3], x: &[f32], y: &[f32], h: usize, t: usize) -> Out {
        let mut w = SiteDev::upload(fn_, base, scale);
        let mut plan = Plan::new(h, t);
        let (mut xd, mut yd) = (cuda::to_f32_dev(x), cuda::to_f32_dev(y));
        let (mut cd, mut od) = (cuda::alloc_zeroed(t * h * 4), cuda::alloc_zeroed(t * HC * h * 4));
        plan.coeffs(kn, &w, xd, cd, t);
        plan.expand(kn, xd, yd, od, t);
        plan.expand(kn, xd, yd, xd, t);
        cuda::sync();
        let o = Out {
            logits: cuda::dtoh(plan.logits, t * MIX),
            pre: cuda::dtoh(plan.pre, t * HC),
            post: cuda::dtoh(plan.post, t * HC),
            comb: cuda::dtoh(plan.comb, t * HC * HC),
            collapsed: cuda::dtoh(cd, t * h),
            expanded: cuda::dtoh(od, t * HC * h),
            in_place: cuda::dtoh(xd, t * HC * h),
        };
        for d in [&mut xd, &mut yd, &mut cd, &mut od] {
            cuda::free_dev(d);
        }
        plan.free();
        w.free();
        o
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mhc_gpu_tiny_cases_are_hf_exact_ops() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Kernels::new();
            for c in tiny() {
                let (base, scale): ([f32; MIX], [f32; 3]) = (c.get("base").try_into().unwrap(), c.get("scale").try_into().unwrap());
                let o = run(&kn, &c.fn_bf16(), &base, &scale, &c.get("x"), &c.get("y"), c.h, c.t);
                for (what, got) in [("logits", &o.logits), ("pre", &o.pre), ("post", &o.post), ("comb", &o.comb),
                    ("collapsed", &o.collapsed), ("expanded", &o.expanded)] {
                    let e = max_rel1(got, &c.get(what));
                    assert!(e <= 4e-6, "{}: {what} {e:.2e}", c.name);
                }
                assert_eq!(bits(&o.in_place), bits(&o.expanded), "{}: in-place expand differs", c.name);
            }
        }
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mhc_gpu_real_shapes_meet_g3() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Kernels::new();
            let f = real();
            let o = run(&kn, &f.fn_, &f.base, &f.scale, &f.x, &f.y, f.h, f.t);
            assert_eq!(bits(&o.in_place), bits(&o.expanded), "in-place expand differs");
            let (gcol, gexp) = (f32s("collapsed.f32"), f32s("expanded.f32"));
            let (h, n) = (f.h, HC * f.h);
            let ma = [
                max_abs(&o.logits, &f32s("logits.f32")),
                max_abs(&o.pre, &f32s("pre.f32")),
                max_abs(&o.post, &f32s("post.f32")),
                max_abs(&o.comb, &f32s("comb.f32")),
            ];
            eprintln!("mhc gpu real: max_abs logits {:.2e} pre {:.2e} post {:.2e} comb {:.2e}", ma[0], ma[1], ma[2], ma[3]);
            assert!(ma[1..].iter().all(|&e| e <= 1e-5), "pre/post/comb max_abs {ma:?}");
            // GPU vs the CPU twin
            let (cs, ccol) = site(&f.x, &f.fn_, &f.base, &f.scale, h);
            for r in 0..f.t {
                let cc = cosine(&o.collapsed[r * h..(r + 1) * h], &gcol[r * h..(r + 1) * h]);
                let ce = cosine(&o.expanded[r * n..(r + 1) * n], &gexp[r * n..(r + 1) * n]);
                let (pre, post, comb) = flat(&cs[r]);
                let tw = [
                    max_abs(&pre, &o.pre[r * HC..(r + 1) * HC]),
                    max_abs(&post, &o.post[r * HC..(r + 1) * HC]),
                    max_abs(&comb, &o.comb[r * HC * HC..(r + 1) * HC * HC]),
                    max_abs(&ccol[r * h..(r + 1) * h], &o.collapsed[r * h..(r + 1) * h]),
                ];
                eprintln!("mhc gpu real row {r}: 1-cos collapsed {:.2e} expanded {:.2e}; vs CPU twin max_abs pre/post/comb/collapsed {tw:?}", 1.0 - cc, 1.0 - ce);
                assert!(cc >= 0.9999 && ce >= 0.9999, "row {r}: cosine collapsed {cc} expanded {ce}");
                assert!(tw[..3].iter().all(|&e| e <= 1e-5), "row {r}: GPU vs CPU twin {tw:?}");
            }
        }
    }

    /// the record launch of steps 1-5: `glm5_mhc_coeffs`, one block per row (the path before
    /// #191); its own scratch so it can run next to [`Plan::coeffs`]
    unsafe fn record_coeffs(kn: &Kernels, w: &SiteDev, plan: &Plan, x: CUdeviceptr, collapsed: CUdeviceptr, t: usize, out: &[CUdeviceptr; 4]) {
        let f = kn.module.get("glm5_mhc_coeffs");
        launch_v(f, t as u32, 1, 1, THREADS as u32, &[x, w.fn_, w.base, w.scale, out[0], out[1], out[2], out[3], collapsed, plan.prm]);
    }

    fn rnd(n: usize, seed: &mut u64, a: f32) -> Vec<f32> {
        (0..n)
            .map(|i| {
                *seed ^= *seed << 13;
                *seed ^= *seed >> 7;
                *seed ^= *seed << 17;
                if i % 89 == 7 { -0.0 } else { ((*seed >> 40) as f32 / (1u64 << 24) as f32 * 2.0 - 1.0) * a }
            })
            .collect()
    }

    fn bf16s(v: &[f32]) -> Vec<u16> {
        v.iter().map(|x| (x.to_bits() >> 16) as u16).collect()
    }

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mhc_gpu_coeffs_are_bit_identical_to_the_record_kernel() {
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Kernels::new();
            let f = real();
            let mut seed = 0x9e37_79b9_7f4a_7c15u64;
            // the real-shape fixture, then random sites and streams at the GLM width and two small ones
            let mut cases: Vec<(String, usize, usize, Vec<u16>, [f32; MIX], [f32; 3], Vec<f32>)> =
                vec![("real".into(), f.h, f.t, f.fn_.clone(), f.base, f.scale, f.x.clone())];
            for (h, t) in [(4096usize, 1usize), (4096, 5), (64, 3), (8, 2)] {
                let fn_ = bf16s(&rnd(MIX * HC * h, &mut seed, 0.05));
                let base: [f32; MIX] = rnd(MIX, &mut seed, 1.0).try_into().unwrap();
                let scale: [f32; 3] = rnd(3, &mut seed, 2.0).try_into().unwrap();
                cases.push((format!("random h {h} t {t}"), h, t, fn_, base, scale, rnd(t * HC * h, &mut seed, 3.0)));
            }
            for (name, h, t, fn_, base, scale, x) in &cases {
                let (h, t) = (*h, *t);
                let mut w = SiteDev::upload(fn_, base, scale);
                let mut plan = Plan::new(h, t);
                let mut xd = cuda::to_f32_dev(x);
                let nan_h = vec![f32::NAN; t * h];
                let (mut ca, mut cb) = (cuda::to_f32_dev(&nan_h), cuda::to_f32_dev(&nan_h));
                let mut rec = [MIX, HC, HC, HC * HC].map(|k| cuda::to_f32_dev(&vec![f32::NAN; t * k]));
                record_coeffs(&kn, &w, &plan, xd, ca, t, &rec);
                let got = [plan.logits, plan.pre, plan.post, plan.comb];
                // three calls on one plan, the outputs NaN-filled before each: the last-block
                // counter must be back at 0 after every call
                for rep in 0..3 {
                    for (i, d) in got.iter().enumerate() {
                        cuda::to_f32_into(*d, &vec![f32::NAN; t * [MIX, HC, HC, HC * HC][i]]);
                    }
                    cuda::to_f32_into(cb, &nan_h);
                    plan.coeffs(&kn, &w, xd, cb, t);
                    cuda::sync();
                    for (i, what) in ["logits", "pre", "post", "comb"].iter().enumerate() {
                        let n = t * [MIX, HC, HC, HC * HC][i];
                        assert_eq!(bits(&cuda::dtoh(got[i], n)), bits(&cuda::dtoh(rec[i], n)), "{name} call {rep}: {what} differs from the record kernel");
                    }
                    assert_eq!(bits(&cuda::dtoh(cb, t * h)), bits(&cuda::dtoh(ca, t * h)), "{name} call {rep}: collapsed differs from the record kernel");
                }
                for d in rec.iter_mut().chain([&mut xd, &mut ca, &mut cb]) {
                    cuda::free_dev(d);
                }
                plan.free();
                w.free();
            }
        }
    }

    /// #191 acceptance bound, fixed in the ticket before the after-measurement: steps 1-5
    /// of one site for one decode row at the GLM width, VRAM-cold `fn` (us)
    const COEFFS_US: f64 = 20.0;

    #[test]
    #[ignore = "needs the GPU: cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1"]
    fn glm5_mhc_gpu_coeffs_time_per_site() {
        use cudarc::driver::sys;
        unsafe {
            let _ctx = cuda::Ctx::init();
            let kn = Kernels::new();
            let (h, t) = (4096usize, 1usize);
            let mut seed = 0x51ce_5eedu64;
            let fn_ = bf16s(&rnd(MIX * HC * h, &mut seed, 0.05));
            let base: [f32; MIX] = rnd(MIX, &mut seed, 1.0).try_into().unwrap();
            let scale: [f32; 3] = rnd(3, &mut seed, 2.0).try_into().unwrap();
            // >= 256 MiB of distinct `fn` copies: every timed launch reads its site from VRAM
            let copies = (256usize << 20).div_ceil(fn_.len() * 2);
            let mut sites: Vec<SiteDev> = (0..copies).map(|_| SiteDev::upload(&fn_, &base, &scale)).collect();
            let mut plan = Plan::new(h, t);
            let mut xd = cuda::to_f32_dev(&rnd(t * HC * h, &mut seed, 3.0));
            let mut cd = cuda::alloc_zeroed(t * h * 4);
            let mut rec = [MIX, HC, HC, HC * HC].map(|k| cuda::alloc_zeroed(t * k * 4));
            let time = |f: &mut dyn FnMut(usize)| -> f64 {
                f(0);
                cuda::sync();
                let mk = || {
                    let mut e: sys::CUevent = std::ptr::null_mut();
                    cuda::ck(sys::cuEventCreate(&mut e, 0));
                    e
                };
                let (a, b) = (mk(), mk());
                cuda::event_record(a, cuda::cur_stream());
                let n = 4 * copies;
                for i in 0..n {
                    f(i);
                }
                cuda::event_record(b, cuda::cur_stream());
                cuda::sync();
                let mut ms = 0f32;
                cuda::ck(sys::cuEventElapsedTime_v2(&mut ms, a, b));
                cuda::event_destroy(a);
                cuda::event_destroy(b);
                ms as f64 * 1e3 / n as f64
            };
            let r = time(&mut |i| record_coeffs(&kn, &sites[i % copies], &plan, xd, cd, t, &rec));
            let p = time(&mut |i| plan.coeffs(&kn, &sites[i % copies], xd, cd, t));
            let bytes = (MIX * HC * h * 2 + HC * h * 4) as f64;
            eprintln!("mhc coeffs h {h} t {t} ({copies} fn copies): record {r:.1} us ({:.0} GB/s), plan {p:.1} us ({:.0} GB/s); x 90 sites per row: {:.2} -> {:.2} ms", bytes / r / 1e3, bytes / p / 1e3, 90.0 * r / 1e3, 90.0 * p / 1e3);
            for d in rec.iter_mut().chain([&mut xd, &mut cd]) {
                cuda::free_dev(d);
            }
            for s in sites.iter_mut() {
                s.free();
            }
            plan.free();
            assert!(p <= COEFFS_US, "the plan's coeffs take {p:.1} us per site (bound {COEFFS_US} us)");
        }
    }
}
