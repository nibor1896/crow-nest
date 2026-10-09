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
path: every MoE layer syncs for its routing.

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
no promotion or eviction) and `reset_cache_replays_the_cold_pass_and_warm_reads_less`; the bin's
own tests (`cargo test --release --bin glm5_run`: phase split, TTFT, percentiles, spread, summary,
prompt length, ids file, the working-set query).
GPU (`cargo test --release --lib glm5_tiers_gpu -- --ignored --nocapture --test-threads 1`):
`glm5_tiers_gpu_every_table_entry_holds_its_record` (synthetic container, 16 records, 7
capacities: the bytes at every entry are `read_range` of the record; 54 s, passed 2026-10-09; since
#187 also the device path's visits and NVMe reads per call, and after `reset_cache` the first
selection read from NVMe again with the right bytes; 51 s, passed 2026-10-09),
`glm5_tiers_gpu_cache_size_is_invisible_in_the_logits` (the real container, the fixed prompt, 6
ids at the plan's V/P, 1 + 7 and 0 + 0: ids and logits bit-identical; plan step 16 abort
criterion, `docs/architecture.md` A9; not run yet) and
`glm5_tiers_gpu_reps_and_reset_are_invisible_in_the_logits` (#187: the real container at the
plan's V/P, rep 1 cold, rep 2 warm, `reset_cache`, rep 3: ids and logits bit-identical, rep 3's
per-row moves = rep 1's; not run yet).

## 7. Not verified

- Whether the `--chain` drift (routing flips compounding f32 rounding over depth) is the size HF's own
  f32 model shows under a 1e-7 perturbation: no such reference run exists.
- The goldens hold the 3-bit weights; the quantisation error against FP8 is measured for layers 0–3
  only (runner doc section 7). `pre` is not compared. MTP (layer 45, step 21), vision (step 20).
- Speed, graph capture, the boot (#175, #149, step 14): the taps synchronize.
- `glm5_run` and the cache-size logits test on the real container (section 6): built, not run.
- `glm5_run`'s timings and counters on the real container and the reps/reset logits test (section
  6.1, #187): built, not run; no baseline figure exists.
