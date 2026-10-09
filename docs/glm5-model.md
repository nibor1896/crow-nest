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

**Since #175 / #149 (2026-10-09):** `engine/src/glm5_tiers.rs` and `bin/glm5_run` run all 45 layers
token by token with the routed experts in VRAM, pinned RAM and the container (NVMe), section 6.
Host- and synthetic-GPU-tested; no run on the real container yet (the lead's smoke).

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
| KDA `q/k/v_proj`, `o_proj` | `glm5_gemv_fp4` (q, k, v strided into the `[t][24576]` row), through `glm5_kda::prompt_with` / `step_with` (`KdaProj`) |
| KDA `q/k/v_conv1d` | decoded once to f32 `[24576][4]` (HF holds the conv in f32) |
| MLA `q_a`, `q_b`, `kv_a_proj_with_mqa`, `o_proj` | `glm5_gemv_fp4` through `MlaScratch::forward_with` (`MlaProj`) |
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
  kernels (`KdaKernels::with_base`), the engine kernel table of the router and the head
  (`gemv_bf16_b`, `gemv_bf16_w`, `argmax_k`) and MLA's `qsa_select_fast` all come from it. Its
  entry set equals the Flash-Next compile's (host test, NVRTC).
- #191: every NVFP4 projection of the path (KDA, MLA, dense FFN, shared expert) runs on
  `glm5_gemv_fp4` from the glm5 FFN module (`kernels_glm5_moe.cu`), not on the engine's
  `gemv_fp4_b` / `gemv_fp4_bs`: the same per-thread FMA chains and reduction tree, so the outputs
  are bit-identical (GPU test `glm5_dense_gpu_fp4_gemv_is_bit_identical_to_the_record_kernels`),
  without the engine kernel's local-memory decode table and with several rows per block.
  `KERNEL_SRC` (Flash-Next, 27B) is unchanged. Per-shape times: #191 and
  `glm5_dense_gpu_fp4_gemv_reads_the_weights_at_vram_rate`.
- `load_layer` reads one layer: the planned tensors, and for a MoE layer its 288 MUL1 records
  (9,474,048 B each, 2.73 GB) from the gate tensor's offset into one VRAM buffer behind the `[288]`
  record table `GpuMoePlan::run` reads. `LayerW::free` releases all of it after the layer. No
  expert cache, no pinned or NVMe tier on this path (`glm5_tiers`, section 6, has them).

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
- **Head of every row** (#165): when the golden carries the runner's `--capture-head` files
  (`head-mean`, `head-norm`, `head-logits.f32`), the head (`run_head`: stream mean + final norm,
  BF16 lm_head GEMV, argmax) runs over all N rows. Per row group: the final norm and the logits get
  cosine, max_abs, rel_rms and the G3 verdict; `top-1` counts the rows whose greedy id equals the
  golden's (first index of the maximum) and lists the others with the golden logit gap. The stream
  mean is fused into the norm kernel, so its line is the host mean of the head's input against
  `head-mean.f32`, reported, not judged (golden-fed it checks the golden's own consistency).
- The taps synchronize after every stage: the seconds printed are no speed figure.

## 5. Tests (host, no GPU)

`cargo test --release --lib glm5_model` (8): the layer schedule, the plan against the real index,
refusals by name, the load decodes (NVFP4 -> f32 / BF16 with the inexact count, the 0x7F rewrite),
the trunk input, the shared compile, the manifest and call split, the metrics and overlaps.
`manager::tests_159_glm_plan::the_glm_plan_books_kv_b_decoded_to_bf16` holds the planner line.

### G3 on all 45 layers and the head (2026-10-09, RTX 5090)

Goldens `models/GLM-5.3-Flash-step06/ref-mul1-all` (runner on the same 3-bit container, all 45 layers,
`--capture-subblocks --capture-head`, record `runs/glm53-flash/step06/golden-mul1-all.json`):

```
target/release/decode glmgolden <crow-nest>/models/GLM-5.3-Flash-step06/ref-mul1-all --cnq <crow-nest>/converter/GLM-5.3-Flash-MUL1K3.cnq [--chain]
```

- **Golden-fed** (`models/glm-g3-all.txt`, rc 0, 1 m 43 s): `glmgolden: ALL PASS (0 of 549 G3 rows failed)`
  (45 layers × 2 sites × 3 roles × 2 row groups, head norm and logits × 2 groups, 5 anchors). Every KDA
  row group (dense and MoE) prints cosine 1.000000000. The only row groups below are the DSA layers'
  `attn out` and `attn expanded` (22 each); the lowest is l31 attn out, decode rows, 0.999987255
  (worst row 89 0.999985368, max_abs 4.61e-3), next l31 prompt 0.999988084 and l35 prompt 0.999990027;
  attributed, not isolated, to the engine's MLA `kv_b` decoded to BF16 (section 2; the golden keeps
  f32). Routing top-8 overlap
  1.0000 (90 / 90 rows) on all 42 MoE layers, DSA selection 1.0000 (90 / 90) on all 11 DSA layers.
  1-cos of the layer output 4.6e-14 … 6.0e-13 at every depth. Head: norm cosine 1.000000000
  (max_abs 2.86e-6), logits 1.000000000 (max_abs 7.63e-6 prompt, 4.77e-6 decode), top-1 90 / 90.
- **`--chain`** (`models/glm-g3-all-chain.txt`, reported, not gated): 1-cos of the layer output
  rises from 1e-11 (l0–l2) to 4.1e-6 (l3), 1.3e-4 (l8), 2.7e-3 (l18), peaks at 3.83e-2 (l32) and is
  1.26e-2 at l44. Routing overlap falls from 0.9972 (l3) to 0.8833 (l34, 32 / 90 rows the same set);
  DSA selection stays 90 / 90 on every DSA layer. Head: logits cosine 0.99316 (prompt) / 0.99226
  (decode), top-1 83 / 90 (the 7 misses are prompt rows, golden logit gaps 0.022–0.179); the 5
  anchors' greedy ids all equal the golden's.

## 6. Three tiers, token by token (`glm5_tiers`, `glm5_run`)

#175 + #149, plan steps 16–17. glm5_next only: nothing on the `Engine` / `Geo` path of Flash-Next
or the 27B constructs or calls it, and no default, flag or manifest of those families changed (gate
R, #174). The layer math is section 1 unchanged; what changes is where an expert record is read.

- **Resident model** (`Glm5Run::load`): `load_layer_without_experts` for all 45 layers (the dense
  part, `FfnW::Moe` with no records and no table) and the head. One `KdaState` per KDA layer and
  one `MlaCache` per DSA layer; `Glm5Pass::swap_kda_state` / `swap_mla_cache` put them in around
  the layer's call. Every row, prompt and generated, is one decode call (`t = 1`, KDA's recurrent
  step); no chunked prefill here. Greedy head after the last prompt row.
- **Sizes** (`tier_sizes`): VRAM and pinned slots per MoE layer are the #159 plan's `hot` and
  `pinned` at the measured free VRAM, `CROW_CONTEXT` (floor 200,000) and the derived pinned budget
  (`HOST_PINNED_CAP` 46 GiB or less: free RAM − `CROW_RAM_MARGIN_GB`). RTX 5090 of record: 50 +
  124, 114 on NVMe; 124 × 42 × 9,474,048 B = 45.95 GiB pinned, 50 × 42 records = 19.9 GB VRAM.
  `--vram-slots` / `--pinned-slots` may ask for less; more is refused by name.
- **Policy**: `ExpertCache` per MoE layer (G1d's arena `layer`), LRU (the policy G1d chose, `LRU s
  0.00 P 0`: no seed, the cache starts empty); `CROW_EXPERT_CACHE` picks another policy, `off` is
  refused (ask for 0 + 0 instead). Counters `[vram, pinned, nvme]` per MoE layer.
- **One MoE call** (`Glm5Pass::call_with_experts`): `GpuMoePlan::route` (router GEMV + top-8),
  host sync, the 8 ids to `ExpertTiers::table_for`, then `GpuMoePlan::experts` (gather, MUL1
  gate/up/act/down, shared expert, combine). `route` + `experts` are exactly the launches of `run`.
  With `CROW_GLM_FLAGS=1` the ids come through mapped memory instead of the sync (section 6.2).
- **The moves** (`serve`): the cache observes the ids (one tick, ascending order like
  `glm_tier_sim`); then, per tier change, in three phases so no slot is overwritten before it is
  read: (A) every expert entering VRAM and every selected expert the policy leaves on NVMe goes to
  a VRAM staging slot (H2D from its pinned slot, D2D from its old VRAM slot, or read from the
  container through `nvme_source` into a 4096-aligned pageable landing buffer, then H2D);
  barrier; (B) experts entering pinned take a freed pinned slot (D2H from their VRAM slot, or read
  from the container straight into the slot); (C) VRAM entrants go from staging into freed VRAM
  slots. The `[288]` table points each selected id at its VRAM slot, its pinned slot (UVA, read
  zero-copy by the MUL1 kernels) or its staging slot; every other entry is 0.
- **NVMe**: `NvmeSource`, one handle per reader, `FILE_FLAG_NO_BUFFERING` + IOCP, 1 reader
  (`--readers`, PREREG amendment 5), records located once (`ExpertRecord::glm5_table`). The
  reader backend is `CROW_NVME_BACKEND`: `iocp` (unset, default; one completion port per reader)
  or `ioring` (Windows 11 I/O ring per reader, refused by name where the system has none, never a
  silent fall-back; same bytes, same checks, `nvme_source::both_backends_read_records_byte_identical_to_a_plain_read`).
  `ioring` with 2 readers is the candidate default, pending an engine A/B (#149). A record is
  read when no tier held it at the start of the call; the cache's NVMe counter can be higher (LRU
  may evict a later id of the same call before its turn; its record is staged from its old slot).
- **Memory beyond the plan**: 8 staging records in VRAM (75.8 MB; the plan books 160) and 8 landing
  records in pageable RAM (75.8 MB, not pinned).
- **How a selected pinned expert is read** (#188, two switches, both off by default; off = the
  path above, bit for bit):
  - `CROW_GLM_PINNED=promote` (default): the policy's exchange rule; under LRU every pinned hit
    enters VRAM (staged H2D) and its VRAM victim goes back to pinned (D2H). #187 baseline, warm
    decode: 1.228 GB H2D + 1.228 GB D2H per token, zero-copy 0.
  - `CROW_GLM_PINNED=zerocopy`: `ExpertCache::set_pinned_stays(true)`: a pinned hit refreshes its
    recency / score / reference bit and stays in pinned; the table points the MUL1 kernels at its
    pinned slot (zero-copy, the same bits as from VRAM). NVMe misses keep the policy's rule: they
    enter VRAM (from the landing buffer), the VRAM victim goes to pinned (D2H), the pinned victim
    to NVMe. Moves left: n2v, n2p, v2p, v2n, p2n; p2v and `pinned_to_stage` only for a selected id
    that earlier misses of the same call pushed out of pinned before its own access (an NVMe access
    by the policy's rule, staged from its old pinned slot; never with one id per call).
  - `CROW_GLM_CPU_LANE=1` (implies `zerocopy`; `CROW_GLM_PINNED=promote` with it is refused by
    name; refused by name on a write-combined pinned tier, i.e. it needs `CROW_PINNED_ALLOC=host`
    on Windows): in a decode call (`t = 1`) every selected id in pinned is computed on the CPU.
    `ExpertTiers::table_for` posts the call's combos (pick order: GPU record base, or CPU host
    record) to `glm5_moe::lane`; `GpuMoePlan::experts` of the same thread takes the post for its
    table: x to the host (async + event), gather, the GPU combos compacted into the first slots
    (`mul1::GemvPlan::run_slots`), gate / up / act / down over those slots, the shared expert;
    then, while the GPU runs them, ALL of the layer's CPU experts in one pool run
    (`cpu_mul1::experts_ffn` with `swiglu_clamp`, 8 threads, the GPU host thread as worker 0)
    straight from their pinned slots; then the compact GPU rows to their combo rows (D2D), one H2D
    per CPU row into `ye`, and the unchanged `glm5_moe_combine` (`w_k ye_k` in pick order +
    shared). The expert hook and its call site in `glm5_model.rs` are unchanged. Prompt calls
    (`t > 1`) and NVMe misses stay on the GPU.
  - **Bits.** `zerocopy`: identical to `promote` (VRAM and pinned reads give the same bits). CPU
    lane: GPU combos and the combine's order are the GPU path's; a CPU combo's row is
    `expert_ffn_mul1_cpu`'s, which differs from the GPU MUL1 kernels' (another f32 order in the
    GEMVs, `docs/mul1-gemv.md` section 3, and the host `exp` in the clamp). Synthetic GLM layer
    (`glm5_moe_gpu_cpu_lane_rows_are_the_cpu_and_gpu_experts`, 2026-10-09): 1 to 8 of 8 combos on
    the CPU, against the GPU-only output max abs 1.6e-3 to 5.2e-3 at rms 1.41e3, 1 - cosine
    3.3e-14 to 2.8e-13; against the oracle 1 - cosine 2.9e-13 to 4.8e-13 (GPU-only 2.75e-13). Not
    bit-identical, so the lane stays switch-only; G3 on the real container is not measured.
  - **Why `host` pinned for the lane** (`glm5_moe_gpu_lane_wc_bench`, 2026-10-09, Core Ultra 9
    285K, one synthetic 3-bit record, 8 threads, median of 15): CPU FFN from write-combined pinned
    32.44 ms per expert (0.29 GB/s), from cacheable pinned 0.440 ms (21.52 GB/s), from the heap
    0.418 ms (22.65 GB/s). CUDA's `cuMemHostAlloc` documents WC memory as not readable efficiently
    by most CPUs. The GPU's read rate from a cacheable pinned tier on this Windows box is not
    measured here (the 2026-09-04 WDDM figure favoured WC, `docs/env.md` `CROW_PINNED_ALLOC`).
  - **Counters** (#187 `Moves`): `cpu_lane` = selected ids the CPU computed (not counted in
    `zero_copy`); `glm5_run` prints `CPU lane <experts> experts <s> s` per token on the counters
    lines and writes `cpu_lane_per_token` / `cpu_lane_s_per_token` (wall time of the lane's pool
    runs, `ExpertTiers::cpu_lane_clock`, read per row in the report callback) and `pinned_use` to
    the JSON.

```
cd engine
CARGO_BUILD_JOBS=4 cargo build --release --bin glm5_run
target/release/glm5_run -n 16 [--cnq PATH] [--ids a,b,c | --prompt TEXT --tokenizer tokenizer.json | --prompt-ids PATH]
                        [--prompt-tokens N] [--reps N] [--cold] [--json PATH]
                        [--vram-slots N] [--pinned-slots N] [--readers N]
```

Without `--ids` / `--prompt` / `--prompt-ids` the prompt is the tokenizer golden `sys_user_default`
(31 ids). Per row: position, phase, rep, greedy id, seconds, NVMe reads (count, MB), the summed and
the per-layer `v/p/n` accesses; then per rep the ids (the text with `--tokenizer`), the prefill and
decode lines and the summary over the reps (section 6.1). The times are those of this synchronous
path: every MoE layer syncs for its routing (switches off; section 6.2 for the two decode switches).

### 6.1 Measuring with `glm5_run` (#187)

The lever A/B instrument: prefill and decode timed and counted apart on one loaded model. It is not
G5 (#151 judges speed only in Crow's window); it gives the baseline a lever (#186 prefill chunks,
router prefetch, reader count) is compared against. No flag changes what `generate` computes: the
counters are host integers in `serve`, the clocks sit behind syncs the path already has.

| flag | effect |
|---|---|
| `--reps N` | `generate` N times on the loaded model (default 1). Rep 1 starts cold (cache empty); later reps start with the cache the previous rep left (warm) |
| `--cold` | empty the cache before every rep (`ExpertTiers::reset_cache`: new `ExpertCache`, slot maps reset, arenas kept, no allocation) |
| `--prompt-tokens N` | the base prompt (golden, `--ids`, `--prompt` or `--prompt-ids`) repeated cyclically and cut to N ids: synthetic, deterministic, chat markers repeat |
| `--prompt-ids PATH` | ids from a file, separated by commas, blanks or newlines (`[ ]` ignored) |
| `--json PATH` | everything below per rep, phase and MoE layer, plus args, `CROW_*` env, machine state, setup times; rewritten after every rep |

**Phases.** Prefill = the prompt rows; the last prompt row runs the head and yields the first id.
Decode = every later row, one generated id each (`-n N` gives N - 1 decode rows).

**Clocks** (no sync added):
- Row seconds = `TokenReport::secs`: `Instant` at the row's start in `Glm5Run::generate` (before
  the embedding), read after the row's closing `cuda::sync()` and the greedy id's `dtoh`. They
  include the 42 routing syncs and the tier moves (NVMe reads are synchronous inside `serve`); they
  exclude the host counter diff and the report callback.
- TTFT and the phase wall times are read in the bin on entry of the report callback (after that
  sync), from an `Instant` taken right before the `generate` call. TTFT = the callback of the last
  prompt row; decode wall = the last row's callback - TTFT. Both include the per-row print of the
  rows before (one line per row).
- Prefill tok/s = prompt ids / TTFT. Decode: tok/s per token = 1 / row seconds, median over the
  decode rows with min / max; the phase rate = decode rows / decode wall; latency p50 / p99 by
  nearest rank over the row seconds.
- Load: `open` (container index and checks), `load` (dense part + head), `tiers` (arenas, pinned
  allocation, record table) printed apart on the `glm5_run setup:` line; the rep wall is the
  `generate` call.

**Counters per phase, per token** (host, `glm5_tiers::Moves`, counted where `serve` issues the
moves): visits (distinct selected ids, 336 per decode token); hits per tier `[vram, pinned, nvme]`
(the cache's counters) and their rates; r = NVMe reads per token (records no tier held at the start
of the call; PREREG-dyn's bar r_hi <= 18.4 at 3.05 bpw is on a bootstrap CI, so no verdict is
printed) and NVMe GB per token; m = NVMe-served visits / visits (can exceed r / visits when LRU
evicts a later id of the same call, section 6); H2D = landing and pinned records copied to staging;
zero-copy = selected ids the kernels read from their pinned slot; host DRAM -> GPU = H2D +
zero-copy; D2H = VRAM records demoted into pinned; promotions (NVMe -> VRAM, pinned -> VRAM, NVMe
-> pinned) and evictions (VRAM -> pinned, VRAM -> NVMe, pinned -> NVMe). Bytes = records x the
record size (9,474,048 B at 3 bit); for zero-copy this assumes a decode visit reads the whole
record (not measured). Prefetch: not built; `issued`, `used`, `wasted` print 0 and `demand misses
uncovered` = r (without prefetch every demand miss is uncovered); the fields are in place for the
router-prefetch lever. Per MoE layer: in the JSON; the text gives the lowest and highest layer by
NVMe reads per token and by VRAM hit rate.

**Machine state** after setup and after every rep: VRAM used / free (`cuMemGetInfo`), the process's
working set, peak working set and private commit (Windows `K32GetProcessMemoryInfo`; unix `VmRSS`,
`VmHWM`), free RAM and system commit (`GlobalMemoryStatusEx`), the store's pinned bytes; once:
`git rev-parse HEAD` of the checkout (`-dirty` with tracked changes; the binary's path and mtime in
the JSON), container path and size, tier sizes, readers, policy.

**Summary over the reps:** median and spread max / min of TTFT, prefill tok/s, decode tok/s
(median per rep), decode phase rate, decode p50 / p99, rep wall and decode r; the spread rule
<= 1.15 of the gates (`runs/glm53-flash/PREREG.md`) is named per metric (`within` / `above for
...`), not a gate verdict; whether the ids are identical across reps; cold / warm per rep.

Stable grep prefixes: `glm5_run row`, `glm5_run setup:`, `glm5_run machine after`, `glm5_run rep
K/N cache cold|warm` + `ids` / `prefill:` / `decode:` / `prefill counters` / `decode counters` /
`prefill layers` / `decode layers` / `wall`, `glm5_run summary`, `glm5_run json`. Shape (values
from the unit test's synthetic rows, not a measurement):

```
glm5_run rep 1/3 cache cold prefill: 3 tok, TTFT 1.030 s, 2.91 tok/s, row latency p50 0.2500 s p99 0.5000 s
glm5_run rep 1/3 cache cold decode: 2 tok, 7.50 tok/s median over tokens (min 5.00, max 10.00), wall 0.320 s = 6.25 tok/s, latency p50 0.1000 s p99 0.2000 s
glm5_run rep 1/3 cache cold decode counters per token: visits 16.0, hits vram 25.0 % pinned 62.5 % nvme 12.5 %, r 2.00 NVMe reads (0.000 GB), m 0.1250, H2D 0.019 GB, zero-copy 0.038 GB, host DRAM->GPU 0.057 GB, D2H 0.000 GB, promotions 2.00, evictions 2.00, prefetch none (issued 0, used 0, wasted 0, demand misses uncovered 2.00)
glm5_run rep 1/3 cache cold decode layers: NVMe reads/tok min l3 1.00 median 1.00 max l3 1.00; VRAM hit min l3 25.0 % max l3 25.0 %
glm5_run summary over 2 reps (median, spread max/min): ttft_s 1.0300 (spread 1.000); prefill_tok_s 2.9126 (spread 1.000); decode_tok_s_median 7.5000 (spread 1.000); ...
glm5_run summary spread rule <= 1.15: every metric within; ids identical across reps: yes; cache per rep: cold warm
```

Baseline recipe (the lead's, after robin's go for the GPU run): `target/release/glm5_run -n 64
--reps 3 --json ../runs/glm53-flash/glm5-run-baseline.json`, then the same with `--cold`. No
figure from this tool exists yet.

**Tests.** Host: `the_moves_put_every_record_where_the_table_points` (a twin of the device store
whose slots hold expert ids: LRU, CLOCK, CLOCK admit 2, LFU 0.7 at V/P 0/0, 0/8, 8/0, 1/7, 2/3,
3/12, 16/40, 24/40 over 150 tokens × 3 layers: every selected id where its entry points, every
cached record in its slot, NVMe reads exactly the records no tier held, no NVMe write into a pinned
slot a queued copy still reads), `lru_by_hand_three_way_exchange`, the sizes against the plan,
staging overflow and out-of-range ids refused by name; `nvme_source::the_record_table_is_locate_glm5_record_by_record`.
#187: `the_move_counters_equal_the_twins_calls_and_the_tier_diff` (the same policies and capacities:
per call visits = distinct ids = the cache's accesses, every mover count = the twin's calls, the six
transitions = the before/after tier diff, zero-copy = the pinned-served ids; all-NVMe: visits = r,
no promotion or eviction) and `reset_cache_replays_the_cold_pass_and_warm_reads_less`. #188:
`zerocopy_pinned_hits_stay_and_are_read_in_place` (the same policies and capacities, one id and
top-8 per call: one id per call never promotes a pinned hit and never stages a pinned record; at
top-8 every pinned -> VRAM transition and every staged pinned record is a selected id earlier misses
of the call pushed out; at P >= top-k fewer pinned -> VRAM transitions than `promote`; `reset_cache`
keeps the option), `the_pinned_switches_parse_and_refuse_by_name`, `lane_combos_follow_the_pick_order`; the bin's
own tests (`cargo test --release --bin glm5_run`: phase split, TTFT, percentiles, spread, summary,
prompt length, ids file, the working-set query).
GPU (`cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1`):
`glm5_tiers_gpu_every_table_entry_holds_its_record` (synthetic container, 16 records, 7
capacities: the bytes at every entry are `read_range` of the record; 54 s, passed 2026-10-09; since
#187 also the device path's visits and NVMe reads per call, and after `reset_cache` the first
selection read from NVMe again with the right bytes; 51 s, passed 2026-10-09),
#188 `glm5_tiers_gpu_zerocopy_and_cpu_lane_tables_hold_their_records` (synthetic container,
`CROW_PINNED_ALLOC=host` for the test, `promote` / `zerocopy` / lane at V/P 1/7, 3/4, 4/12, 0/16:
every entry holds its record; the lane posts the combos in pick order, CPU pointers to the record's
bytes, counted as `cpu_lane`; at 4/12 `promote` 58 p2v + 58 pinned records staged + 60 D2H over
24 calls, `zerocopy` 0 + 0 + 8; 61 s with the test above, passed 2026-10-09),
`glm5_tiers_gpu_cache_size_is_invisible_in_the_logits` (the real container, the fixed prompt, 6
ids at the plan's V/P, 1 + 7 and 0 + 0: ids and logits bit-identical; plan step 16 abort
criterion, `docs/architecture.md` A9; not run yet) and
`glm5_tiers_gpu_reps_and_reset_are_invisible_in_the_logits` (#187: the real container at the
plan's V/P, rep 1 cold, rep 2 warm, `reset_cache`, rep 3: ids and logits bit-identical, rep 3's
per-row moves = rep 1's; not run yet).

### 6.2 Decode switches (#149 path B, #189)

Two switches of `Glm5Run` (`glm5_flags`), both default off; off, `generate` runs every row through
`Glm5Run::row` (one row body) and the path is call for call the one before them. `glm5_run` prints
`[glm5_run] decode switches: ...` at load when one is on.

- **`CROW_GLM_FLAGS=1`** (#149 path B, the routing half): after `route`, a one-block kernel copies
  the 8 selected ids into mapped pinned host memory, `__threadfence_system`, then raises a 64-bit
  sequence flag there (the device-side publication of `docs/architecture.md` 3.2; a kernel-side
  mapped write, no memop on the legacy stream); the stream is submitted (`cuStreamQuery`). The host
  spins on the flag (30 s, then refused by name) instead of `cuStreamSynchronize` plus a blocking
  pageable `cuMemcpyDtoH`. **What the host still waits for:** the flag, i.e. every launch up to and
  including the router (the GPU work the stream sync waited for), and inside
  `ExpertTiers::table_for` its own waits: the phase-A barrier (a stream sync), the NVMe reads, the
  synchronous landing and table uploads. Applies to `generate` and to `row` (serve).
  **Not built: the "landed" half** (the GPU waiting on a host-raised flag before `experts`, so the
  host could queue ahead of the staging): `serve`'s mover queues its copies on the current stream and
  syncs it, and the table goes up from a transient host vector, so experts queued ahead would run
  before their copies or deadlock the barrier. It needs the mover on its own stream with persistent
  pinned sources (`serve` / `GpuMover`, not this change).
- **`CROW_GLM_LOOKAHEAD=1`** (#189, `generate` only): a row with a head that is not the last queues
  the next row before the host reads its id. The greedy id stays on the GPU; `Feed::gather` writes
  its embedding row (BF16 widened exactly, as `embed_rows` + `trunk_input`) into the four streams;
  the id (and the logits when kept) go to pinned memory by an async copy and are read at the next
  row's first MoE layer, after its routing sync or flag. The embedding table goes to VRAM at load
  (154,880 x 4096 BF16 = 1,268,776,960 B; the #159 plan does not book it: `glm5_run` plans before the
  load, so the tier sizes do not change, free VRAM after setup is that much lower). Clock: a decode
  row's `secs` and its report callback end when its id is read, inside the next row (after that
  row's dense prefix and first router); the decode phase rate from the callback times stays
  comparable, the per-row latency does not. `row` (serve) does not look ahead (serve needs token k
  for its end-of-turn check before step k+1).
- **PUBFAST** (the reference's `GLM53_NV_PUBFAST`): not built, no variable. The reference's warp-ballot publish
  (sybil-solutions/glm-flash-lite `kernels/nv2/nv2_dev.cu` `nv_pub_fast_k`) replaces two serial
  thread-0 scans over the experts inside its publish kernel; here the publish kernel copies the raw
  top-8 and the host dedups 8 values (`distinct_ids`), so there is no scan to replace.

Tests: host `glm5_flags::tests::only_1_turns_a_switch_on`. GPU (`cargo test --release --lib
glm5_flags_gpu -- --ignored --nocapture --test-threads 1`, 17 s, passed 2026-10-09):
`glm5_flags_gpu_the_feed_is_the_host_feed_bit_for_bit` (64-id table with NaN, +-Inf, -0,
denormals), `glm5_flags_gpu_routed_ids_arrive_with_their_flag` (a kernel holding the stream 0 /
0.3 / 2 ms before it writes the ids, 300 calls), `glm5_flags_gpu_the_switches_are_invisible_in_ids_logits_and_reports`
(a synthetic 4-layer glm5_next container with the real layer shapes, layers 0-2 KDA + dense, layer 3
DSA + MoE with 16 MUL1 records, vocab 2048, 5 + 6 ids, V 3 + P 4: flags, lookahead and both give
the switch-off ids, logits bit for bit and row reports except the clock; `row` with the flags too).
Each was shown red against a broken mechanism: the host not waiting for the flag (ids `[0, ..]`,
then `CUDA_ERROR_ILLEGAL_ADDRESS`), the id read before a sync point (ids shifted by one row), the
gather writing one stream (12,285 of 16,384 values differ).

### 6.3 Decode rows as CUDA graphs (`CROW_GLM_GRAPH`, #190)

Default off; off, the row is the one before it call for call (`glm5_graph`'s hooks are no-ops).
On, `Glm5Run` captures a row's launches into piecewise CUDA graphs and replays them; `glm5_run`
prints `[glm5_run] CROW_GLM_GRAPH on: ...` at load. Applies to `generate` (with or without the two
switches of 6.2) and to `row` (serve).

- **Segments**, cut at the only host hand-off of a row, the MoE router: segment 0 = every layer
  before the first MoE layer and that layer up to `route`; segment m = MoE layer m-1's `experts` +
  expand and every layer up to MoE layer m's `route`; the last = the last `experts` + expand. The
  head is one graph (3 kernels). GLM-5.3-Flash: 42 MoE layers, so 43 row graphs + the head.
- **What still needs the host per MoE layer** (unchanged): the router ids (`glm5_model::router_ids`:
  stream sync + copy, or the 6.2 flag) and `ExpertTiers::table_for` (staging, its syncs, the table
  upload). The embedding upload per row stays eager too.
- **Capture**: the row's eager body runs with a capture open on a non-blocking stream
  (thread-local capture mode); at each router `call_inner` closes the segment, which is checked
  (kernel nodes only, else refused by name), instantiated and launched on the legacy stream; the
  hand-off runs eagerly; the next segment opens. Replay launches the same segments around the same
  hand-off. Graphs run on the legacy stream, so the eager copies and syncs stay ordered with them.
- **Per-row inputs**: the position reaches the kernels through the MLA scalars `[pos, 1]`, written
  once per row from pinned memory (`RowGraphs::stage`; `MlaScratch::begin` skips its own upload
  inside a capture). The one position-dependent launch shape, the `idx_scores` grid
  (`glm5_mla::score_grid`, 0 below position 3, +1 every 128 positions), and the record tables'
  addresses key the captured row. Up to 8 captured rows are kept, least recently used out
  (`glm5_graph::KEYS`): a key seen before replays, only a new key captures (a new sequence over
  positions already seen: none). Cost: 4 kept rows of 266 kernel nodes (synthetic model) dropped
  free VRAM by 4 MiB against 0 with one kept row; a GLM-5.3-Flash row has about 6x the nodes; host
  memory not measured. The t = 1 FFN plans are made before the first capture (`Glm5Pass::ensure_plans`).
- **Refused**: with `CROW_GLM_CPU_LANE=1` (its host work sits inside a segment); a replayed seam
  whose table moved.

Counts (nsys `cuda_api_sum`, 2026-10-09, RTX 5090, #190 commit, test `glm5_graph_gpu_profile_rows`:
the synthetic 8-layer model below, every record in VRAM, 8 steady decode rows at positions 8-15
under `--capture-range=cudaProfilerApi`, per row):

| arm | `cuLaunchKernel` | `cuGraphLaunch` | `cuStreamSynchronize` | `cuMemcpyHtoDAsync` | `cuMemcpyDtoH` |
|---|---|---|---|---|---|
| off | 269 | 0 | 19 | 8 | 6 |
| `CROW_GLM_GRAPH=1` | 0 | 7 | 17 | 7 | 6 |
| `CROW_GLM_GRAPH=1` + `CROW_GLM_FLAGS=1` | 5 (publish) | 7 | 12 | 7 | 1 |

The 2 syncs fewer are the 2 DSA layers' scalar uploads; the 6 row graphs + the head replace 266 + 3
kernel launches. No speed figure: synthetic weights, 8 rows.

Tests (`cargo test --release --lib glm5_graph -- --ignored --nocapture --test-threads 1`): host
`only_1_turns_the_graph_switch_on`, `the_score_grid_is_select_s_launch_shape`; GPU
`glm5_graph_gpu_the_graphs_are_invisible_in_ids_logits_and_reports` (a synthetic 8-layer glm5_next
container with the real layer shapes: layers 0-2 KDA + dense, 3-7 MoE with 16 MUL1 records each, DSA
at 3 and 7, vocab 2048; 5 + 6 ids, V 3 + P 4; the graph alone, with the flags, with both 6.2
switches: the switch-off ids, logits bit for bit and row reports except the clock; `row` too; 2
captures and 8 replays over the 10 rows). Shown red against: the per-row position staging removed
(ids `[1931, 2010, ..]` instead of `[998, 1709, ..]`), the MLA upload left inside the capture
(refused: "segment 0 captured a graph node of type 1"), the score-grid key removed (1 capture, 9
replays, 264 kernel nodes instead of 266; the ids stay equal there: 10 tokens are far below the
selection's 2,051 (`sel_max`), so the pool scores cannot exclude any; inferred, not traced).
`glm5_graph_gpu_seen_keys_replay_without_recapture`: two sequences of 5 + 266 ids on one store
(rows 0-269, score grid 0 to 3): 4 captures in the first, none in the second, ids and logits bit
for bit as switch-off; red with one kept row (8 captures, 532 replays instead of 4, 536).

## 7. Not verified

- Whether the `--chain` drift (routing flips compounding f32 rounding over depth) is the size HF's own
  f32 model shows under a 1e-7 perturbation: no such reference run exists.
- The goldens hold the 3-bit weights; the quantisation error against FP8 is measured for layers 0–3
  only (runner doc section 7). `pre` is not compared. MTP (layer 45, step 21), vision (step 20).
- Speed, graph capture, the boot (#175, #149, step 14): the taps synchronize.
- `glm5_run` and the cache-size logits test on the real container (section 6): built, not run.
- `glm5_run`'s timings and counters on the real container and the reps/reset logits test (section
  6.1, #187): built, not run; no baseline figure exists.
- The decode switches (section 6.2) on the real container: built, not run; no speed figure.
- `CROW_GLM_GRAPH` (section 6.3) on the real container: not run (ids, launches per row, speed);
  a position past 2,051 tokens, where a stale `idx_scores` grid would change the selection: not run.
- #188 `CROW_GLM_PINNED=zerocopy` and `CROW_GLM_CPU_LANE=1` on the real container: not run (speed,
  ids, G3 cosine); how far the CPU lane overlaps the GPU's experts, the idle pool workers' spin
  against the GPU host thread, and the GPU's zero-copy rate from a cacheable pinned tier on Windows:
  not measured.
