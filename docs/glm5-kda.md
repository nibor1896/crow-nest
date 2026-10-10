# GLM-5.3-Flash: the KDA sub-block on the GPU

`engine/src/glm5_kda.rs` is the KDA part of step 13b of the GLM-5.3-Flash plan (crow-nest #162,
parent nibor1896/Crow#362). It computes `a = KDA(h)` of `docs/glm5-next-recipe.md` section 6 for
one of the 34 KDA layers: from the input-layernormed rows `h` `[T][4096]` to the `o_proj` output
`[T][4096]`, with the per-sequence state carried across prompt calls and decode steps.

**Wired (#161, 2026-10-09):** `glm5_model` calls it in the decoder order and `decode glmgolden` runs it against the layerwise goldens; see [glm5-model.md](glm5-model.md). No GPU run of the wired path yet.

## 1. Kernels: reused and new

| step (recipe section 6) | kernel | source |
|---|---|---|
| 2: q, k, v projections (one stacked `[24576][4096]` BF16 matrix) | `gemm_bf16_dense` (prompt), `gemv_bf16_w` (decode) | `KERNEL_SRC` |
| 3: causal depthwise conv k 4 + SiLU, window of the last 3 pre-conv rows | `transpose_rt`, `conv_silu`, `conv_state_update` (prompt), `conv_step` (decode) | `KERNEL_SRC` |
| 4: split into q, k, v `[T][64][128]` | `split_qkv` (prompt); decode reads the conv output in place | `KERNEL_SRC` |
| 5-6: `g = -5 sigmoid(exp(A_log[h]) (f_b(f_a x) + dt_bias))` per key channel, `beta = sigmoid(b_proj x)` | `kda_gate` | `kernels_glm5_kda.cu` |
| 7: l2norm of q and k (eps 1e-6), q x 128^-0.5 | `l2norm_repeat` (64 key heads = 64 value heads, no repeat) | `KERNEL_SRC` |
| 8: `S <- diag(exp g_t) S; delta = beta (v - S^T k); S <- S + k delta^T; o = S^T q` | `kda_persist_r` (prompt), `kda_step_r` (decode) | `kernels_glm5_kda.cu` |
| 10-11: `gate = g_b(g_a x)`, gated RMSNorm per head (eps 1e-5, weight, sigmoid) | `gemm_bf16_dense` / `gemv_bf16_w`, `rmsnorm_gated` | `KERNEL_SRC` |
| 12: `o_proj` | `gemm_bf16_dense` / `gemv_bf16_w` | `KERNEL_SRC` |

The reused kernels are `KERNEL_SRC` compiled at `glm5_kda::kernel_geo` (GDN key heads = value heads
= 64, head dims 128, conv channels 24576, `CN_EPS` 1e-5, sigmoid gate). Fields glm5_next has no GDN
counterpart for keep their Flash-Next values: compile-only, never launched by this module. The
three new kernels are their own NVRTC module (`kernels::GLM5_KDA_SRC`, loader
`kernels::glm5_kda::Kernels`), like `kernels_mul1.cu`: one more entry in `KERNEL_SRC` would break
the PTX of record (`tests_300_c4`).

`kda_persist_r` / `kda_step_r` are `delta_rule_persist_r` / `delta_rule_step_r` with one change:
the decay is per key channel (`S[dk][:] *= exp(g[dk])`), where GDN multiplies the whole head by
`exp(g[head])`. Operation order and the `__fmul_rn` / `__fmaf_rn` intrinsics are the same, so
`kda_persist_r` over n tokens and n `kda_step_r` calls give identical bits (tested).

The prompt recurrence runs token by token, as GDN's does. HF's prompt path is the chunked form
(chunk 64, `chunk_kimi_delta_attention`); both compute the same recurrence and differ only in f32
summation order. A chunked (WY) prompt kernel is deferred until a measured prefill number asks
for it (ticket #162, alternatives).

## 2. API for the integrator

```rust
let kk = glm5_kda::KdaKernels::new(&Glm5Geo::GLM_5_3_FLASH);   // compiles both modules once
let w  = glm5_kda::KdaWeights::upload(&kk.d, &host_weights);   // one per KDA layer
let st = glm5_kda::KdaState::alloc(&kk.d);                     // one per KDA layer per sequence
st.reset();                                                     // zero = start of a sequence
let sc = glm5_kda::KdaScratch::alloc(&kk.d, max_rows_per_call); // shared by all KDA layers
glm5_kda::prompt(&kk, &w, &st, &sc, x, t, out);                 // t <= max_rows, continues st
glm5_kda::step(&kk, &w, &st, &sc, x, out);                      // one decode row
```

- `x` and `out` are f32 device rows `[t][4096]`; launches queue on the current stream.
- `KdaHostWeights` takes the checkpoint tensors in module layout: BF16 bit patterns for the
  projections, f32 for `conv1d` (`[24576][4]`, q|k|v order), `dt_bias`, `A_log`, `o_norm`.
- State per layer per sequence: `s` `[64][128][128]` f32 = 4 MiB (`Glm5Geo::kda_state_bytes`),
  `conv` `[24576][3]` f32 = 288 KiB (`Glm5Geo::kda_conv_bytes`), the oldest row first.
- Any split of a prompt into calls gives the one-call result; measured bit-identical for calls of
  40 + 2 + 54 rows against one call of 96.
- #161: `prompt_with` / `step_with` are `prompt` / `step` launch for launch, with q|k|v and o_proj
  handed to a closure (`KdaProj`): the container stores them NVFP4, `glm5_model` runs them on
  `glm5_gemv_fp4` (#191, bit-identical to `gemv_fp4_bs` / `gemv_fp4_b`; q|k|v as one
  `glm5_gemv_fp4_x3` launch; `w.qkv`, `w.o_proj`
  unused, 0). `KdaKernels::with_base` takes the
  one shared glm5 `KERNEL_SRC` module instead of compiling its own.

## 3. Goldens and tests

`oracle/export_glm5_kda_golden.py` writes `engine/tests/fixtures/glm5/kda/` (2.4 MB): one HF
`Glm5NextTextLinearAttention` (transformers 5.16.1, eager, f32, CPU, 2 threads) built from the real
config (`engine/tests/fixtures/GLM-5.3-Flash/config.json`, layer 0) on synthetic weights; 96 prompt
rows in one call, then 8 decode rows one by one against a `DynamicCache`. Stored: the output of all
104 rows, the conv window after the prompt, the recurrent state of heads 0, 21, 42, 63 after the
prompt and after the last row, and the Frobenius norm of all 64 heads at both points.

The weights are not stored. Weights and input rows come from a counter-based generator
(splitmix64, `glm5_kda::synth`) both sides implement; every value is `q * 2^-p` with an integer `q`
in [-128, 127], exact in BF16 and f32. `dt_bias` lies in [-6, 2), so the decay ranges from
`exp(-5)` per token to almost 1 within one head. The manifest holds the sha256 of every generated
tensor; the Rust generator matches all 14.

| test | GPU | checks |
|---|---|---|
| `kernels::tests_glm5_kda_src::glm5_kda_source_compiles_with_every_entry` | no | the module compiles (NVRTC) with its 3 entries, `KDA_D` 128 |
| `glm5_kda::tests::kda_dims_are_the_family_row` | no | widths, state bytes = `Glm5Geo`, the `CN_*` defines of `kernel_geo` |
| `glm5_kda::tests::kda_synth_generator_is_the_exporters_bit_for_bit` | no | 14 sha256 against the manifest |
| `glm5_kda::tests::kda_reused_kernels_compile_at_the_kda_geometry` | no | the 9 reused entries compile at the KDA geometry |
| `glm5_kda::tests_gpu::kda_gpu_gate_and_recurrence_match_the_f64_reference` | yes | `kda_gate` and 6 tokens of the recurrence (non-zero start state) against f64; persist = step bits |
| `glm5_kda::tests_gpu::kda_gpu_layer_matches_the_hf_golden` | yes | G3: prompt in one call + 8 decode steps |
| `glm5_kda::tests_gpu::kda_gpu_chunked_prompt_matches_the_hf_golden` | yes | G3 with the prompt in calls of 40 + 2 + 54, and against the engine's one-call run |

```
cargo test --release --lib glm5_kda                                    # host tests
cargo test --release --lib glm5_kda::tests_gpu -- --ignored --nocapture --test-threads 1
```

The G3 criterion is that of `runs/glm53-flash/PREREG.md` (per layer: cosine >= 0.9999, max
|deviation| reported). Prompt rows and decode rows are two anchors. Each anchor must pass flattened
and in every single row. The states must reach cosine >= 0.9999 and a per-head norm within 1e-3.

## 4. Measured (2026-10-09, RTX 5090, worktree `g162-kda` on crow-nest `3f72673`, n = 1 run each)

| anchor | cosine (flattened) | worst row | max abs | golden RMS |
|---|---|---|---|---|
| prompt rows 0-95, one call | 1.000000000 | 1.000000000 | 9.4e-6 | 0.100 (all rows) |
| decode rows 96-103 | 1.000000000 | 1.000000000 | 4.8e-7 | |
| prompt in calls 40 + 2 + 54 | 1.000000000 | 1.000000000 | 9.4e-6 | |

- Decode `1 - cosine` by step: 5.0e-13 to 3.9e-13, falling. No drift over the 8 steps.
- State after the prompt: heads 0/21/42/63 max abs 3.1e-7; the norm of every head within 6.9e-7
  of the golden. After the last row: 1.0e-7 and 9.2e-7. Conv window: identical (max abs 0).
- `kda_gate` against f64: g max abs 6.7e-7, beta 7.2e-8. Recurrence over 6 tokens: o max abs 2.0e-8.
- HF's own prompt split (calls 40 + 2 + 54 against one call): out max abs 3.1e-7, state 1.1e-6
  (manifest `hf_split`).

Red without the change (mutations, each reverted): the GDN form of the decay (one scalar per head)
gives cosine 0.608 on the prompt and 0.719 on the f64 recurrence test. Leaving out
`conv_state_update` gives 0.962 for the split prompt and 0.851 for the decode rows. Restarting S on
every prompt call gives 0.900 for the split prompt.

Limits: synthetic weights only. The real KDA tensors (NVFP4 in the committed recipe,
PREREG "Recipe as committed") and the 34 real layers are judged by `glmgolden` on the container
once the layer loop exists. No speed was measured.
