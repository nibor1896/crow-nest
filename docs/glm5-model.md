# GLM-5.3-Flash: the layer path and the `glmgolden` harness

crow-nest #161–#165 (GLM-5.3-Flash plan steps 13a–13e, the G3 half of each ticket; scope in the
comment on #161). `engine/src/glm5_model.rs` calls the five block modules in the decoder order and
loads the real 3-bit container one layer at a time; `decode glmgolden` runs it against the
layerwise runner's goldens. The `Engine` / `Geo` path of Flash-Next and the 27B does not see it
(gate R, #174); the GLM boot still refuses (`meta::glm5_not_built`: #175, #149 / plan step 14,
plan step 20).

**Status (2026-10-09):** built and host-tested. No GPU run: the real-layer G3 table waits for
robin's Go. The goldens are `models/GLM-5.3-Flash-step06/ref-fp8/` (#156, FP8 originals, layers
0–3, `--capture-subblocks`).

## 1. The layer

Per call of `t` rows (`Glm5Pass::call`), `docs/glm5-next-recipe.md` sections 3–4:

| # | step | module |
|---|---|---|
| 1 | `attn_hc` coefficients and collapse, `[t][4][4096]` -> `[t][4096]` | `glm5_mhc::Plan::coeffs` |
| 2 | `input_layernorm`, `w * (x * rsqrt(mean(x²) + 1e-5))` in place | `MlaKernels::rmsnorm_rows` (`gm_rmsnorm`) |
| 3 | KDA (34 layers): prompt calls `prompt_with`, decode rows `step_with`; MLA + DSA (11 layers): `forward_with` | `glm5_kda`, `glm5_mla` |
| 4 | expand into the streams, in place | `glm5_mhc::Plan::expand` |
| 5 | `ffn_hc` coefficients and collapse | `glm5_mhc` |
| 6 | `post_attention_layernorm` | `gm_rmsnorm` |
| 7 | dense FFN (layers 0–2) or router + 8 of 288 MUL1 experts + shared expert | `GpuFfnPlan`, `GpuMoePlan` |
| 8 | expand | `glm5_mhc` |

The trunk input is the embedding copied into the 4 streams (`trunk_input`, HF
`modeling_glm5_next.py:1477`); the head is `glm5_head` (`run_head`). mHC runs exactly
`hc_sinkhorn_iters` = 20 Sinkhorn rounds, HF's count, not "to convergence": HF's own `comb` row
sums are off by up to 8.5e-2 after them (#156 goldens).

## 2. Codec gap

The container stores these as NVFP4 (`converter/src/recipe.rs`, PREREG "Recipe as committed"):

| tensor | engine |
|---|---|
| KDA `q/k/v_proj`, `o_proj` | `gemv_fp4_bs` (q, k, v strided into the `[t][24576]` row) / `gemv_fp4_b`, through `glm5_kda::prompt_with` / `step_with` (`KdaProj`) |
| KDA `q/k/v_conv1d` | decoded once to f32 `[24576][4]` (HF holds the conv in f32) |
| MLA `q_a`, `q_b`, `kv_a_proj_with_mqa`, `o_proj` | `gemv_fp4_b` through `MlaScratch::forward_with` (`MlaProj`) |
| MLA `kv_b_proj` | decoded once to BF16 (round to nearest even); the load line prints how many values BF16 does not hold exactly |

Every NVFP4 scale byte 0x7F is rewritten to 0x7E before use (`residency::sanitize_sf_slab`, the
rule of `gen::load_pw_x`, #177); the load line prints the count. The BF16 keeps (KDA gates, indexer,
router, `hc_*_fn`) run as stored; norms, biases, `ape`, `A_log`, `dt_bias`, `hc_*_base/scale` are
f32 on the device. `kv_b` at BF16 is 369,098,752 B for 11 layers, 265,289,728 B more than its
NVFP4 bytes inside the dense part: the #159 planner books that difference
(`Glm5Geo::kv_b_decode_bytes`, the `kv_b at BF16` line of `states --plan`). At the 3-bit record
(9,474,048 B, RTX 5090, 200,000 tokens) the plan moves from N 51 / P 124 / NVMe 113 to N 50 /
P 124 / NVMe 114; at the 4.5-bit record it stays N 32 / P 83 / NVMe 173. Planner numbers, not
measured.

`glm5_model::check_plan` holds every tensor of a layer (`layer_tensors`, `model_tensors`) against
the index before a byte is read: present, a dtype its take accepts, the planned shape; a miss is
refused by name. The test runs it against the real container's index, cut to layers 0–4
(`engine/tests/fixtures/glm5/model/index-l0-4.json`).

## 3. One compile, one layer in VRAM

- `Glm5Kernels::new` compiles `KERNEL_SRC` once at `glm5_kda::kernel_geo` (`p2` off). The KDA GDN
  kernels (`KdaKernels::with_base`), the engine kernel table of the FFN and the head (`gemv_fp4_b`,
  `gemv_fp4_bs`, `gemv_bf16_b`, `gemv_bf16_w`, `argmax_k`) and MLA's `qsa_select_fast` all come from
  it. Its entry set equals the Flash-Next compile's (host test, NVRTC).
- `load_layer` reads one layer: the planned tensors, and for a MoE layer its 288 MUL1 records
  (9,474,048 B each, 2.73 GB) from the gate tensor's offset into one VRAM buffer behind the `[288]`
  record table `GpuMoePlan::run` reads. `LayerW::free` releases all of it after the layer. No
  expert cache, no pinned or NVMe tier.

## 4. `decode glmgolden`

```
cd engine
CARGO_BUILD_JOBS=4 cargo build --release --bin decode
target/release/decode glmgolden ../models/GLM-5.3-Flash-step06/ref-fp8 [--chain] [--layers A:B] [--cnq PATH]
```

- Container: `--cnq`, else `CROW_CNQ`, else `converter/GLM-5.3-Flash-MUL1K3.cnq`. Refused before
  CUDA unless it is an index v2 whose config passes the glm5_next family row (38 checks) with MUL1
  expert records.
- Calls: the runner's (`prompt_chunk` rows per prompt call, 0 = one call; each decode row alone).
- **Golden-fed** (default): layer k reads `l<k-1>-output.f32` (layer 0: `embed.f32` in 4 streams),
  its FFN site the golden `l<k>-ffn_hc-in.f32`, so each site is judged on the golden's input. Per
  layer, site (`attn`, `ffn`) and row group (prompt, decode): `collapsed`, the sub-layer `out` and the
  `expanded` streams get cosine, max_abs, rel_rms and the G3 verdict (cosine >= 0.9999, a NaN fails);
  `post` and `comb` get max_abs; `pre` is not compared (HF's hyper-connection does not return it, so
  the runner cannot capture it). MoE layers print the routing top-8 overlap against
  `l<k>-routing-ids.i32`, DSA layers the selection overlap against `l<k>-dsa-topk.i32` (sets per row).
  Exit 1 when any G3 row fails.
- **`--chain`**: layer 0 reads the container's own embedding rows (printed against `embed.f32`), each
  later layer the engine's output: drift over depth, the same table, not gated, exit 0.
- Both modes print `1 - cos` of the layer output by depth and flag a monotone rise (#161's failure
  mode). When the last layer runs and the golden has `logits-anchor-<p>.f32`, the head's logits are
  compared per anchor, with the greedy id.
- The taps synchronize after every stage: the seconds printed are no speed figure.

## 5. Tests (host, no GPU)

`cargo test --release --lib glm5_model` (8): the layer schedule, the plan against the real index,
refusals by name, the load decodes (NVFP4 -> f32 / BF16 with the inexact count, the 0x7F rewrite),
the trunk input, the shared compile, the manifest and call split, the metrics and overlaps.
`manager::tests_159_glm_plan::the_glm_plan_books_kv_b_decoded_to_bf16` holds the planner line.

## 6. Not verified

- Any GPU run of the path: the per-layer G3 table on layers 0–3 (golden-fed and `--chain`) awaits
  robin's Go. The goldens come from the FP8 originals, the engine reads the 3-bit container, so the
  table holds the quantisation error as well (step 6 measured container vs FP8 at cosine
  0.99308–0.99806 for layers 0–3 on the 4.5-bit container).
- Layers 4–44 and the logits: no golden yet.
- Speed, graph capture, the boot (#175, #149, step 14), MTP (step 21), vision (step 20).
