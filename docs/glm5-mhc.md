# GLM-5.3-Flash mHC residual

crow-nest #161 (plan step 14, mHC part) computes the manifold-constrained hyper-connection (mHC)
of a glm5_next decoder layer: four residual streams, a collapse into the sublayer input, and a
Sinkhorn-projected 4x4 mixer that writes the sublayer output back into the streams. The math is
HF's `Glm5NextTextHyperConnection.forward` (transformers 5.16.1, `modeling_glm5_next.py:267-295`)
and the decoder layer's mix (`:1316-1318`); `docs/glm5-next-recipe.md` sections 4 and 5 list it
step by step.

- **GPU** (sm_120): `engine/src/kernels_glm5_mhc.cu`, its own NVRTC module
  (`kernels::GLM5_MHC_SRC`), host side `glm5_mhc::{Kernels, SiteDev, Plan}`.
- **CPU twin**: `glm5_mhc::{logits, coeffs, collapse, expand, site}`, the reference the GPU is
  tested against.

**Status:** kernels, CPU twin and tests. The layer driver, the loader of
`layers.L.hc_{attn,ffn}_{fn,base,scale}` and the `decode glmgolden` harness are `glm5_model`
(#161, [glm5-model.md](glm5-model.md)); its GPU run awaits robin's Go.

## 1. Commands

```
cd engine
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_mhc                     # 4 tests, no GPU
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_mhc_gpu -- --ignored --nocapture --test-threads 1
```

The GPU tests are `#[ignore]` (CI has no GPU). Regenerate the goldens (CPU, 2 torch threads,
seconds; never needed unless the reference changes):

```
ORACLE_THREADS=2 .venv-oracle/Scripts/python.exe -I oracle/export_glm5_mhc_golden.py
```

## 2. API for the layer driver

| call | does |
|---|---|
| `Kernels::new()` | compiles `GLM5_MHC_SRC` (3 entries: `glm5_mhc_mix`, `glm5_mhc_expand`, and `glm5_mhc_coeffs`, the record the #191 test compares against) |
| `SiteDev::upload(fn_, base, scale)` | one site's weights: `fn` `[24][4H]` BF16 bits, `base` `[24]` f32, `scale` `[3]` f32 (the container stores base and scale as F32, `oracle/glm5_common.py` `CNQ_F32`) |
| `Plan::new(h, max_tokens)` | parameter buffer and coefficient scratch: `logits` `[T][24]`, `pre`/`post` `[T][4]`, `comb` `[T][4][4]` |
| `plan.coeffs(kn, w, x, collapsed, t)` | one launch (#191): `glm5_mhc_mix` grid (24, T), one mix row per block (each block re-derives the RMS factor); the last block of a row (a per-row counter in the plan) runs steps 3-5 on 16 lanes of warp 0 and the collapse on warps 1-7. Steps 1-5 below, bit-identical to the one-block-per-row record `glm5_mhc_coeffs`; `x` `[t][4][H]` f32 -> `collapsed` `[t][H]` f32 |
| `plan.expand(kn, x, y, out, t)` | one launch, grid (H/256, T): step 6; `out == x` updates the streams in place |

One plan serves both sites of every layer, in the decoder order: `coeffs(attn_hc)` ->
`input_layernorm` -> attention -> `expand` -> `coeffs(ffn_hc)` -> `post_attention_layernorm` -> FFN
-> `expand`. Layer 0's input is the embedding copied into all four streams (`modeling :1477`); the
final collapse is the unweighted stream mean (plan step 13e, not here). No host round trip: the
Sinkhorn loop runs on the GPU, so both launches can sit in a captured decode graph.

## 3. Numerics

Per row, all f32 (`docs/glm5-next-recipe.md` section 5):

1. `r = rsqrt(mean(X²) + 1e-5)` over the 16,384 values (eps = `rms_norm_eps`, not `hc_eps`)
2. `m = fn · (r X)`: normalise first, then the 24 dot products, HF's order
3. `pre = σ(m·s0 + b) + 1e-6`, `post = 2·σ(m·s1 + b)`
4. `comb = softmax_i(m·s2 + b) + 1e-6`, one column normalisation, then 19 rounds of row then column
   normalisation, every divisor `+ 1e-6`: a fixed 20 steps as HF, not "to convergence"
5. `collapsed = Σ_s pre[s] X[s]`
6. `X'[i] = post[i] y + Σ_j comb[j][i] X[j]` (row j = source stream, column i = destination)

The kernel writes every add, multiply and divide as an `__f*_rn` intrinsic (no fma contraction
outside the dot products and the sum of squares), accurate `expf`; the coefficient tail runs on one
thread per row. The sum of squares and the dot products are block reductions, so their summation
order differs from torch's; the CPU twin sums those two in f64.

The decoder layer of HF's BF16 model casts `post` and `comb` to BF16 and mixes in BF16
(`modeling :1316-1318`). The layerwise reference runner runs f32 (#158), so this module mixes in
f32; a BF16 residual stream would need the cast added to `glm5_mhc_expand`.

## 4. Tests and goldens

Goldens: `engine/tests/fixtures/glm5/mhc/` (1.6 MB), written by `oracle/export_glm5_mhc_golden.py`
from HF's own module with the checkpoint config (rev `eb9eb208`: hc_mult 4, hc_eps 1e-6,
hc_sinkhorn_iters 20, rms_norm_eps 1e-5) and SYNTHETIC BF16 weights, torch f32 on the CPU.

- `real/`: the real block shapes (H 4096, `fn` `[24][16384]`), 5 rows; row 0 has four identical
  streams (the layer-0 broadcast), rows 1-4 unequal stream magnitudes and a massive-activation
  channel. `manifest.json` holds shapes and sha256.
- `tiny.json`: three H 8 cases with a mild, sharp and extreme comb scale (0.5 / 4 / 12), exact f32
  values in JSON.

| test | checks | measured 2026-10-09 (RTX 5090) |
|---|---|---|
| `glm5_mhc_fixture_matches_its_manifest` | constants and sha256 of every `real/` file | |
| `glm5_mhc_tiny_cases_are_hf_exact_ops` | CPU logits, Sinkhorn tail from the golden logits, collapse, expand: \|a−b\|/(1+\|b\|) ≤ 4e-6 | pass |
| `glm5_mhc_real_shapes_meet_g3` | gate G3: cosine ≥ 0.9999 per row on `collapsed` and `expanded`; `pre`/`post`/`comb` max_abs ≤ 1e-5 | 1−cos ≤ 3.8e-14; max_abs ≤ 6.0e-7 |
| `glm5_mhc_source_compiles_with_every_entry` | NVRTC (host only), every entry, `#define`s = Rust constants | |
| `glm5_mhc_gpu_tiny_cases_are_hf_exact_ops` | the kernels on `tiny.json`, the same 4e-6; in-place expand = out-of-place bit for bit | pass |
| `glm5_mhc_gpu_real_shapes_meet_g3` | the kernels on `real/`: G3 per row, coefficients ≤ 1e-5, GPU vs CPU twin ≤ 1e-5 | 1−cos ≤ 4.1e-14; max_abs pre/post/comb ≤ 4.2e-7, logits 1.9e-6 |

Each test was run red once against a mutated implementation (one Sinkhorn round fewer, the column
normalisation dropped, `comb` transposed in the mix, `hc_eps` in the RMSNorm; on the CPU twin and,
except the column normalisation, on the kernel).

## 5. Open

- Speed (#191): the record `glm5_mhc_coeffs` ran one block per row (one SM per site at decode,
  90 sites per token): 177 us per site in the 2026-10-09 `glm5_run` profile, 16 ms per token.
  `coeffs` now spreads it over (24, T) blocks with the record's per-thread order (no split-K, so
  the bits stay). Micro-bench numbers: #191 and `glm5_mhc_gpu_coeffs_time_per_site`.
- G3 on the real model (every layer, both sites, against the layerwise runner's goldens) is the
  `decode glmgolden` table ([glm5-model.md](glm5-model.md)); not run yet. `pre` cannot be compared:
  HF's hyper-connection does not return it.
