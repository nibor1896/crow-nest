# glm5_next FFN: sigmoid router, clamped MUL1 experts, shared expert, dense FFN

crow-nest #164 (GLM-5.3-Flash plan step 13d / 15, gate G3). `engine/src/glm5_moe.rs` computes the
FFN sub-block of a glm5_next layer: the router with the selection-only `e_score_correction_bias`
over 288 experts, top-8, the routed experts as MUL1 records (#181) with the SwiGLU clamp, the
shared expert and the dense FFN of layers 0-2 (NVFP4). GPU kernels: `engine/src/kernels_glm5_moe.cu`,
its own NVRTC module (`kernels::GLM5_MOE_SRC`, `kernels::glm5_moe::Kernels`).

**Status:** module and tests only. Nothing in the engine calls it; `gen.rs` / `boot.rs` wire it
with the other glm5_next arms (#161-#165). The host lane is checked against the HF oracle; the GPU
lane's tests are written and compile, **not run yet** (section 5).

## 1. Commands

```
cd engine
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_moe                    # 9 tests, no GPU
CARGO_BUILD_JOBS=2 cargo test --release --lib glm5_moe_gpu -- --ignored --nocapture --test-threads 1
```

Regenerate the goldens (only when the synthetic set or the reference changes):

```
cd engine
GLM5_MOE_DUMP=<scratch dir> cargo test --release --lib glm5_moe_write_oracle_inputs -- --ignored --nocapture
cd ..
ORACLE_THREADS=2 .venv-oracle/Scripts/python -I oracle/export_glm5_moe_golden.py --inputs <scratch dir>
```

The dump is ~3.5 GB (30 decoded experts, the dense matrices in f32); the goldens it produces are
118 KB in `engine/tests/fixtures/glm5/moe/`. Delete the dump afterwards.

## 2. Math of record

transformers 5.16.1 `modeling_glm5_next.py`, `docs/glm5-next-recipe.md` sections 9-10:

| step | rule | HF |
|---|---|---|
| logits | `x W_r^T` in f32, W_r BF16 `[288][4096]` | `M:160` |
| choice | `s = sigmoid(logits)`, `c = s + e_score_correction_bias` (f32); the group stage is a no-op (`n_group` = `topk_group` = 1) | `M:161-176` |
| select | top-8 of `c`; ties to the lowest expert (torch gives no order, so routing is compared as a set) | `M:177` |
| weights | `w = s[ids]` (unbiased), `w / (sum w + 1e-20) * 2.5` | `M:178-182` |
| SwiGLU | `h = silu(min(g, 10)) * clamp(u, -10, 10)` in every FFN; a NaN stays NaN | `M:98-104`, `M:137-142` |
| MoE | `y = sum_k w_k down_k(h_k) + shared(x)`, shared width 2048, no gate | `M:200-207` |
| dense | the same MLP at width 12,288, layers 0-2 | `M:84-104` |

## 3. API (for the integrator)

| item | what it is |
|---|---|
| `MoeGeo::new(&Glm5Geo, ExpertRecordSpec)` | the FFN geometry; refuses a non-MUL1 record and a size no MUL1 bitrate gives; K from the record bytes (9,474,048 B = K 3) |
| `route(geo, &RouterWeights, x) -> Routing` | host router (f64-accumulated logits, then the kernel's selection `route_from_logits`) |
| `Routing { ids, weights }` | `[T][8]` in pick order; `ids_of(t)` for `ExpertCache::observe_token`, `needed()` = the records the layer needs (the NVMe fetch list once the cache says which are cold) |
| `moe_cpu(geo, &MoeLayerCpu, &dyn ExpertRecords, x, y, threads, path)` | CPU lane: routed experts on `cpu_mul1::gemv`, shared on `cpu_nvfp4::gemv`, clamp between |
| `ExpertRecords::record(e) -> &[u8]` | CPU hook: pinned tier, RAM copy, or a host buffer an NVMe fetch (`nvme_source::ColdSource`, #149) filled |
| `ffn_nvfp4_cpu(&ExpertBlock, limit, x, y, ..)` | dense layers 0-2 and the shared expert on the CPU |
| `GpuMoePlan::new(geo, tokens)`, `run(kn, mk, gk, &GpuMoeWeights, table, x, y)` | GPU lane, 13 launches on the current stream, no host sync (graph-capturable) |
| `table` | device `[288] u64` of record bases, one per expert: a VRAM slot or a pinned-host UVA address (the residency / #175 cache hook; the kernels do not know which) |
| `GpuFfnPlan::new(hidden, inter, tokens, limit)`, `run(kn, gk, &GpuFfnWeights, x, y)` | dense layers 0-2 (and the shared expert inside `GpuMoePlan`) on `gemv_fp4_b` |
| `GpuMoePlan::read_routing()` | the last routing (syncs), for the counters and the routing dumps |

`kn` is the engine's main `kernels::Kernels` (it needs `gemv_bf16_b`, `gemv_fp4_b`), `mk` the
`kernels::mul1::Kernels`, `gk` the `kernels::glm5_moe::Kernels`. NVFP4 scale bytes reach
`gemv_fp4_b` raw (0x7F would read as 480): hand it sanitized bytes, as the residency does for
experts. The GPU lane runs one MUL1 slot per (token, k) combo (`tokens * 8` slots of one token);
a prefill grouping by expert is a speed question, not part of this step.

## 4. Acceptance on synthetic weights (G3, plan step 15)

Synthetic set (`glm5_moe.rs` tests, seed `0x01646105`): router BF16 uniform +-0.55, bias +-0.09,
activations +-0.17; routed experts = the #181 synthetic MUL1 records (K 3, seed `0x1640 + 9 e`),
shared and dense NVFP4 with scale bytes 0x30-0x3F and global scale 0.3. The clamp is exercised:
6.9 % of the routed gate outputs exceed 10, 13.6 % of the up outputs exceed 10 in magnitude.
The oracle is HF's own `Glm5NextTextTopkRouter`, `Glm5NextTextMoE` and `Glm5NextTextMLP` (eager,
f32, torch 2.13.0+cpu, 2 threads) on the same weights decoded to f32 (MUL1 by the #181 decoder).
The Rust tests re-check the sha256 of every encoded input and record against the golden's manifest.

Measured 2026-10-09 (host lane, Windows, release build):

| check | rows | result | bound |
|---|---|---|---|
| routed id set vs oracle | 64 router + 4 MoE | 68 / 68 equal; smallest 8th/9th choice gap 3.59e-5, 0 ties | equal sets |
| routing weights | 68 x 8 | max abs 5.96e-8 | 1e-6 |
| MoE layer output (MUL1 experts + shared) | 4 | 1 - cosine 4.99e-13 .. 5.46e-13, max abs 6.3e-3 at rms 1.4e3 | cosine >= 0.9999 |
| dense FFN output (12,288) | 2 | 1 - cosine 3.14e-13 .. 3.20e-13, max abs 8.3e-3 at rms 2.7e3 | cosine >= 0.9999 |

Without the clamp the same rows give cosine 0.956 (MoE) and 0.971 (dense).

## 5. Not verified

- The GPU tests (`glm5_moe_gpu_router_is_the_host_selection`, `glm5_moe_gpu_layer_matches_the_oracle`:
  router kernel = host selection incl. ties and NaN rows; clamp kernel = `swiglu_clamp` bit for bit;
  MoE from VRAM and pinned RAM equal bits and cosine >= 0.9999 against the same goldens; dense
  likewise) were not run on 2026-10-09: the RTX 5090 was at 97 % for a running conversion. The
  NVRTC compile of `GLM5_MOE_SRC` is checked on the host.
- Speed: not measured (no benchmark in this step).
- Real weights, the full container, the per-layer `glmgolden` table of the ticket: plan step 9 /
  13a, not this step.
