# GLM-5.3-Flash: MLA latent cache and DSA indexer

Plan step 14, the MLA/DSA part (crow-nest #163, parent nibor1896/Crow#362). This page says how the
engine computes the 11 MLA + DSA layers of glm5_next, what state it keeps, how the module is called
and what was measured. What the layer computes is [glm5-next-recipe.md](glm5-next-recipe.md),
sections 7 and 8. The code is `engine/src/glm5_mla.rs` and its own NVRTC module
`engine/src/kernels_glm5_mla.cu`.

**Wired (#161, 2026-10-09):** `glm5_model` calls it in the decoder order and `decode glmgolden` runs it against the layerwise goldens; see [glm5-model.md](glm5-model.md). No GPU run of the wired path yet.

## 1. What it computes

One call takes `t` rows `x` (the output of `input_layernorm`, `[t][4096]` f32) at the absolute positions
`pos0 .. pos0 + t` and returns the attention sub-block output `[t][4096]`. Per row at position `p`:

| # | step | kernel |
|---|---|---|
| 1 | `q_resid = rms(q_a x) * w`, `q = q_b q_resid` `[64][256]` | `gm_gemm`, `gm_rmsnorm` |
| 2 | `c = rms(kv_a x) * w` → latent cache row `p`, BF16 `[512]` | `gm_gemm`, `gm_latent_store` |
| 3 | `[wk x \| gate x \| weights_proj x]` → indexer cache row `p` = `[LayerNorm(wk x) \| gate x \| 1]`, BF16 `[257]` | `gm_gemm`, `gm_idx_store` |
| 4 | `iq = wq_b q_resid` `[32][128]`; per complete pool `P` (rows `4P..4P+3`, `P < (p+1)/4`): `pk[c] = Σ_j softmax_j(gate_j[c] + ape[j][c]) · key_j[c]`, `score = Σ_h (w_h · 32^-0.5) · relu(iq_h · pk · 128^-0.5)` | `gm_gemm`, `gm_idx_scores` |
| 5 | the best `2048 / 4 = 512` pools (all of them while there are at most 512), expanded to their 4 rows, plus the tail rows `4·⌊(p+1)/4⌋ .. p`: at most 2051 rows | `gm_sel_prep`, `qsa_select_fast` |
| 6 | `q~_h = W_k,hᵀ q_h` `[512]` (`W_k,h` = `kv_b` rows `h·512 .. h·512+256`) | `gm_absorb` |
| 7 | `u_h = Σ_{s∈sel} softmax_s(q~_h · c_s · 256^-0.5) · c_s`, split-K with online softmax, then the merge | `gm_attn`, `gm_attn_merge` |
| 8 | `o_h = W_v,h u_h` `[256]` (`W_v,h` = `kv_b` rows `h·512+256 .. (h+1)·512`), `y = o_proj o` | `gm_out_v`, `gm_gemm` |

Steps 6 to 8 are the absorbed form of HF's expanded K/V attention (DeepSeek-V2, arXiv:2405.04434,
§2.1.2-2.1.3; with no RoPE part all of K absorbs). The host test
`absorbed_attention_equals_the_expanded_form` shows the two orders agree to 1e-12 in f64.

The selection is `qsa_select_fast` of `KERNEL_SRC`, unchanged: it already pools 4 rows from position 0,
breaks exact score ties to the lowest pool index and appends the tail. HF uses `torch.topk`, whose order
among equal values is unspecified (https://pytorch.org/docs/stable/generated/torch.topk.html), so
selections are compared as sets and exact ties at the 512th/513th pool are counted
(`glm5_layerwise._watch_index_ties`). The bitmap of `qsa_select_fast` holds 65,536 pools: a cache of at
most 262,144 tokens (`glm5_mla::MAX_POOLS`); the 200k boot floor fits, `context_max` 1,048,576 does not.

## 2. State per sequence and layer

| cache | row | bytes per token | 200,000 tokens × 11 layers |
|---|---|---|---|
| latent `c` | 512 BF16 | 1,024 | 2.25 GB |
| indexer, HF layout `[key \| gate \| valid]` | 257 BF16 | 514 | 1.13 GB |

Both are indexed by absolute position (`MlaCache`, `cap` rows). The byte counts are the #159 planner's
(`Glm5Geo::latent_bytes_per_token`, `indexer_bytes_per_token`); the test
`the_glm_5_3_flash_layer_has_the_planned_shapes_and_cache_bytes` holds them equal. Pooled keys are not
cached: `gm_idx_scores` recomputes them from the indexer rows on every call, as HF does
(`modeling_glm5_next.py:899-972`). The cost of the BF16 indexer row is in section 4.

## 3. The module for the integrator

```rust
let d = glm5_mla::MlaDims::of(&Glm5Geo::GLM_5_3_FLASH);
let kn = glm5_mla::MlaKernels::new(d, main_module.get("qsa_select_fast"));
let cache = glm5_mla::MlaCache::new(&d, cap);            // one per DSA layer and sequence
let mut s = glm5_mla::MlaScratch::new(&d, max_t, cap);   // shared by the layers
s.forward(&kn, &weights, &cache, x, y, pos0, t);         // or the stages below
```

- `MlaWeights`: device pointers. Matrices BF16 `[out][in]`; the three x-side indexer projections stacked
  as `idx_x = [wk; index_kpool_compress_gate; weights_proj]` `[288][4096]`; norm weights, the LayerNorm
  bias and `ape` `[4][128]` f32.
- `MlaScratch::begin(pos0, t)` writes the call into a device word pair every kernel reads; the stages
  `query`, `store_latent`, `store_index`, `select`, `attend` and `linear` queue on the current stream.
  `forward` runs them all with `gm_gemm` projections. A caller with other weight codecs (the container
  plans NVFP4 for q_a, q_b, kv_a, kv_b and o_proj, `converter/src/recipe.rs`) runs its own projections
  into `qa`, `q`, `kva`, `ip`, `iq` and from `o`, and keeps the stages.
- `kv_b` is read by `gm_absorb` and `gm_out_v` as BF16, whatever the container holds (open question 2).
- #161: `forward_with` is `forward` stage for stage with q_a, q_b, kv_a and o_proj handed to a
  closure (`MlaProj`); `glm5_model` runs them on `glm5_gemv_fp4` (#191, bit-identical to
  `gemv_fp4_b`). `MlaKernels::rmsnorm_rows` exposes
  `gm_rmsnorm` for the decoder's two layernorms.
- Scratch: `[max_t][cap/4]` f32 pool scores dominate for long caches (512 rows × 50,000 pools = 100 MB);
  a prefill chunk bounds `max_t`.
- Graph capture: the per-call position is device data, but the grid of `gm_idx_scores` grows with the
  pool count and the attention split count follows `t`. A captured decode graph needs a fixed grid
  (not built here).

## 4. Evidence

Golden: `oracle/export_glm5_mla_golden.py` (HF `Glm5NextTextAttention`, transformers 5.16.1, torch
2.13.0+cpu, eager, f32, 2 threads, layer 3 of the real-shape mini config of `glm5_layerwise`), synthetic
weights at the real block shapes from a counter-based generator that `glm5_mla::synth` reproduces bit
for bit (every weight BF16-representable), T = 2200 prompt rows in calls of 512, then D = 8 decode rows.
Rows `p >= 2051` are sparse (513 to 552 complete pools, 512 kept). Files in
`engine/tests/fixtures/glm5/mla/` (1.2 MB): outputs and latents at 21 anchor rows, the selected pools of
the 157 sparse rows, the pool scores of the 10 sparse anchors, the generator probes. Two goldens on the
same weights and input:

- `bf16kv`: HF with the engine's cache precision (latent and indexer row rounded to BF16 where HF stores
  them; nothing else changes);
- `f32`: HF as the oracle runs it.

Engine run, RTX 5090 (driver 616.56), 2026-10-09, branch `g163-mla`: prompt in calls of 384 rows (not the
golden's 512), then the 8 decode rows, `cargo test --release --lib glm5_mla::tests_gpu -- --ignored`:

| check | result | gate |
|---|---|---|
| anchor cosine vs `bf16kv`, 16 prompt + 5 decode anchors | worst 0.999999996 (row 1023); max \|d\| ≤ 2.6e-5 at row RMS 0.058-0.98 | ≥ 0.9999 (G3) |
| selection, 2208 rows | 2051 dense rows = the causal set; 157 sparse rows = the golden's 512 pools, 0 differ, 0 tie rows | identical |
| small shapes vs the f64 host reference (4 heads, latent 64, budget 4 pools, 46 rows) | every selection identical, max \|d\| ≤ 1e-4 × row RMS | |
| anchor cosine vs `f32` | worst 0.999452 (row 2206) | not gated |
| selection vs `f32` | 10 of 157 sparse rows differ by one pool | not gated |

**The cost of the BF16 indexer row** (oracle side, same weights and input, `bf16kv` against `f32`,
2026-10-09): rounding only the latent to BF16 leaves every selection unchanged and every row at cosine
≥ 0.9999972; rounding only the indexer row to BF16 swaps one pool at the boundary in 10 of 157 sparse
rows (2120, 2123, 2136, 2137, 2153, 2161, 2182, 2186, 2191, 2206), and those rows fall to cosine
0.998888 (row 2120) against HF in f32. With the HF layout at BF16 the engine therefore meets G3 against
HF at the cache's own precision, not against the f32 golden (open question 1).

Each new test was run red: mean pooling instead of the gate softmax (host pool test red; GPU: all 157
sparse rows differ), the tail dropped (selection and absorbed-vs-expanded tests red), `W_v` in place of
`W_k` in the absorption (host test red; GPU: G3 cosine 0.088), another generator constant (probe test
red); the first build of the source failed the NVRTC test (`INFINITY` undefined).

## 5. Open questions (robin decides)

1. **Indexer row precision.** Keep the HF layout at BF16 (514 B per token per layer, the #159 plan) and
   gate against HF at BF16 cache precision, as now; or store the indexer row in f32 (1,028 B, 2.26 GB at
   200k × 11 layers); or cache the pooled key once per completed pool in f32 (128 B per token plus at most
   3 pending rows, the "minimal" row of the recipe's section 13, exact against the f32 golden because a
   completed pool's key never changes). The last is smaller and exact but changes the #159 layout.
2. **kv_b codec.** The kernels read `kv_b` as BF16 (16.8 M values per layer, 33.6 MB; 369 MB for 11
   layers); the converter recipe writes it NVFP4 from a BF16 source. Either the converter keeps it BF16,
   or the boot decodes NVFP4 into a BF16 buffer (a second rounding unless the decoded values are exact in
   BF16), or the kernels learn NVFP4. #161 took the second (scope comment on #161): decoded once at load,
   the inexact-value count printed, the BF16 bytes booked by the #159 planner.

## 6. Not done here

Wiring into `gen.rs` / `boot.rs`, the KDA layers (#162), mHC (#161), MoE (#164), the head (#165),
shared top-k layers (`indexer_types` "shared", absent from rev `eb9eb208`), MTP's indexer sharing
(step 21), FP8 KV, speed and a decode graph. The ticket's env-gated expanded-K/V reference mode in the
engine was replaced by the host test of section 1 (same proof, no product switch).
