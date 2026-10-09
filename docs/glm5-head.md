# glm5_next head: stream mean, final norm, lm_head

crow-nest #165 (GLM-5.3-Flash plan step 13e, the head part of gate G3). After its 45th decoder layer
GLM-5.3-Flash holds 4 residual streams per token. The head turns them into logits:

| # | operation | HF (transformers 5.16.1, `modeling_glm5_next.py`) |
|---|---|---|
| 1 | `h = mean over the 4 streams`, unweighted | `Glm5NextTextHyperHead.forward` `:298-302`, called at `:1493` |
| 2 | `normed = weight * (h * rsqrt(mean(h²) + 1e-5))`, f32, weight not `1 + weight` | `Glm5NextTextRMSNorm` `:66-80`, `:1421` |
| 3 | `logits = normed @ lm_head^T`, `[154880, 4096]` BF16, untied | `:2075`, `:2179-2181` |

The recipe rows are [glm5-next-recipe.md](glm5-next-recipe.md) section 3, rows 4-6.

**Status:** kernel, host side and tests. Nothing in the engine calls them; `gen.rs` gets the
`StreamMeanRms` arm when the lead integrates the GLM arms (13a-13e).

## 1. Code

- `engine/src/kernels_glm5_head.cu`: `glm5_stream_mean_rms`, steps 1 and 2 in one kernel, grid
  (rows), block 256. Its own NVRTC module (the `kernels_mul1.cu` pattern: one more `.entry` in
  `KERNEL_SRC` would break the PTX of record). The scalars `{H, S, one_plus_w, eps bits}` sit in one
  device int buffer. The mean is a sum followed by one IEEE division, the same as HF's `mean`.
- `engine/src/kernels.rs`: `GLM5_HEAD_SRC` and the loader `kernels::glm5_head::Kernels`.
- `engine/src/glm5_head.rs`:
  - `HeadGeo::of(&Glm5Geo)` reads hidden 4096, streams 4, vocab 154,880 and eps 1e-5 from the family
    row. It refuses a tied checkpoint by name. `GLM5_NORM_ONE_PLUS_W = false` pins the norm form,
    because `Glm5Geo` has no `norm_one_plus_w` field (the family has only one form).
  - `Head` has these methods: `new`, `stream_mean_rms(x, w, out, rows)`,
    `lm_head(k, lm, normed, logits, rows)`, `argmax(k, logits, ids, rows)`, `run` (mean + norm +
    lm_head) and `free`. Step 3 reuses the engine's own kernels from `KERNEL_SRC`: `gemv_bf16_w`
    (the BF16 GEMV that `lm_head_row` launches) batched over rows by its grid y, and `argmax_k`.
    Every launch is queued on the current stream with no sync and no upload, so it is safe inside
    graph capture.
  - The host twin is f64: `stream_mean_rms_ref` and `logit_ref`. `testkit` holds the weight
    generator and the fixture reader.
- `oracle/export_glm5_head_golden.py`: writes the goldens with HF's own two modules. It runs on
  `.venv-oracle` with torch threads 2 and takes about 45 s.

## 2. Goldens

`engine/tests/fixtures/glm5/head/` (2.8 MB): `x.f32` `[4][4][4096]`, `normed.f32` `[4][4096]`,
`logits.f32` `[4][154880]`, `manifest.json` (versions, seeds, sha256, top-1 per anchor).

The weights are **synthetic** at the **real shapes**: hidden 4096, 4 streams, eps 1e-5 and the full
vocab of 154,880 (it is not reduced). The weights are not stored. Python and Rust generate them from
the same counter hash (splitmix64), integer-exact and exact in BF16:
`lm_head[r][c] = ((h >> 56) - 128) * 2^-9` and `norm_w[d] = 0.5 + (h >> 57) * 2^-7`. The Rust
generator is checked bit for bit against the samples in the manifest. The four anchors:

| anchor | input | what it catches |
|---|---|---|
| 0 | streams `a_s * randn + b_s`, distinct per stream | the mean is no single stream |
| 1 | anchor 0 x 300 | scale invariance of the norm |
| 2 | tiny, `mean(m²)` ≈ 1.9e-6 < eps | eps 1e-5 vs 1e-6, mean vs sum |
| 3 | stream 3 x 40, streams 0-2 opposite in sign | a skewed mean |

The golden's own f32 noise against the same math in f64 is at most 7.4e-7 on `normed` and 1.9e-5 on
the logits (manifest `golden_vs_f64_max_abs`).

## 3. Acceptance and result

The thresholds were fixed before the first run: cosine ≥ 0.9999 per anchor on `normed` and on the
logits (G3), top-1 equal to the golden's, and max_abs ≤ 1e-5 x max|golden| on `normed` and ≤ 1e-3 on
the logits. Cosine cannot see scale, so max_abs is what holds the norm's scale (eps, mean vs sum).

Measured 2026-10-09 on the RTX 5090 at worktree `g165-head`, n = 4 anchors:

| check | normed cos (min) | normed max_abs | logits cos (min) | logits max_abs | top-1 |
|---|---|---|---|---|---|
| host f64 twin vs golden (logits on 1,601 vocab rows) | 1.000000000 | 7.4e-7 | 1.000000000 | 9.7e-6 | equal on its rows |
| GPU kernel + `gemv_bf16_w` + `argmax_k` vs golden, full vocab | 1.000000000 | 4.8e-7 | 1.000000000 | 1.7e-5 | 4 / 4 |

Against the host twin over 4 shapes (H 4096/256/512, S 4/3, eps 1e-5/1e-6, `1 + w` on and off), the
kernel stays within max_abs ≤ 1.1e-6 at a scale up to 7.1.

Five deliberate defects each turn a test red (failure line of anchor 0):

| defect | test | failure |
|---|---|---|
| kernel sums the streams, no division | GPU golden | normed max_abs 6.1e-5 (bound 4.2e-5) |
| kernel applies `1 + w` | GPU golden | normed cosine 0.99027 |
| kernel eps 1e-6 | GPU golden | normed max_abs 5.8e-5 |
| host twin drops eps | host golden | normed max_abs 6.5e-5 |
| host twin takes stream 0 | host golden | normed cosine 0.566 |

Not done here: the head fed with the step-7 runner's real layer-44 output on the real weights
(`decode glmgolden`, #161), and the engine's own layer-44 output once 13a-13d pass. Both need the
harness and the container.

## 4. Commands

```
cd engine
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_head                    # 7 host tests (one NVRTC compile)
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_head_gpu -- --ignored --nocapture --test-threads 1
../.venv-oracle/Scripts/python.exe -I ../oracle/export_glm5_head_golden.py   # rewrites the goldens
```

The GPU tests need about 1.4 GB of VRAM (the synthetic lm_head is 1,268,776,960 B) and take 2 s.

## 5. Integration (for `gen.rs`)

- A `FinalNorm::StreamMeanRms` arm next to `HcMixer` and `Rms`, at the prefill head (`gen.rs` around
  `:5059-5076`) and at the decode step (around `:5867-5870`):
  `head.stream_mean_rms(<stream state [t][4][4096]>, <norm weight f32>, s.mixed_final, rows)`.
  After that, the existing `lm_head_row(i)` and `argmax_k` run unchanged, with `n_vocab` = 154,880.
- At the load, `lm_head.weight` is read raw as BF16 into `w.lm_head`, as the `HcMixer` arm does, and
  `model.language_model.norm.weight` goes through `load_f32`. The GEMV's k is `p.n2560`, which
  holds `d.h` (`gen.rs:2659`), so it is 4096 once the GLM dims are set.
- `HeadGeo::of(&Glm5Geo::GLM_5_3_FLASH)` gives the numbers. A runtime `Geo` for glm5_next does not
  exist yet (`meta::glm5_not_built`).

## 6. Open

- The input side of the stream trunk is the embedding copied into the 4 streams
  (`modeling_glm5_next.py:1477`, recipe section 3 row 2). It is not in #165's scope and not in this
  module. #161 cites it in its Evidence, but no step names it as its own.
- The real model runs in BF16, where HF rounds the norm output to BF16 before the weight. The engine
  and the step-7 runner are f32. That difference belongs to the whole-model gate (step 15), not here.
