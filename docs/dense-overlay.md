# The dense path in BF16, as an overlay container

Step 3 of the requant series (issue #77), after the characterization harness of #75 and the
BF16 originals of #76. It is the first step that changes what the engine can COMPUTE, so it is
built the way #75 and #76 were read: the default path stays byte-identical, every new path gets
its own control, and the control runs before any quality claim.

- `docs/quality-probe.md` — what answer quality is measured with, and the A1 / B1 baseline.
- `docs/dense-originals.md` — where the BF16 originals came from and the proof that they are
  the originals.

## 1. What this is

`CNQ4.5-M` puts RTN NVFP4 on all 17 dense text kinds — 495 tensors, 2,583,306,240 values, 2.9 %
of the model's values and 3 % of its bytes. The working Unsloth GGUF of this architecture keeps
exactly those tensors at 5.5 to 32 bit and saves only on routed experts; NVIDIA's hybrid NVFP4
recipe keeps attention, the Mamba layers feeding it and every conv1d in BF16. #75 measured what
CNQ4.5-M pays for not doing that: 16.40 German non-words per 1000 against llama-server's 8.21,
and exact literals broken.

Rebuilding those 495 tensors at BF16 does not need a second 104.73 GB container. It needs an
OVERLAY: a container in the same CNQ1 format holding only those tensors as `dtype bf16`, opened
BESIDE the base container, whose tensors shadow the base tensors of the same name and section.
5.17 GB on disk, one environment variable, and without that variable nothing about the engine
moves.

## 2. The converter side

```
converter dense-overlay --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
                        --out  converter/dense-bf16-originals.cnq \
                        --from-originals models/Qwen3.8-Flash-Next-original/dense/dense.safetensors

converter dense-overlay --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
                        --out  converter/dense-bf16-control.cnq \
                        --from-container converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
```

Additive, exactly as `requant-check` is: the word `dense-overlay` is taken off the front of the
argument list and the conversion path below never sees it. `requant-check` still reports 495 of
495 byte-identical.

| flag | what it writes |
|---|---|
| `--from-originals <f.safetensors>` | the BF16 originals of #76, copied byte for byte — no arithmetic at all |
| `--from-container <base.cnq>` | the CONTROL: the base container's own NVFP4 blocks, dequantized by the same `e2m1 * ue4m3 * global` walk `cnq::dequant_block` runs, rounded to bf16 nearest-even |
| `--kinds a,b,c` | a subset of the 17 kinds (default all); an unknown kind is refused with the list the base carries, an empty selection is refused rather than written |

The kind of a tensor is its name without the `model.language_model.layers.<N>.` prefix and
without `.weight`. The engine derives it the same way (`cnq::kind_of`), and both derivations are
under test, because a kind name that silently matched nothing would be measured as "the
originals change nothing".

The index trailer carries an `overlay` block — base file name, base byte size, source, kinds,
counts, build date — and the engine refuses an overlay built over a different base by the first
two fields.

**What the control costs, and it is reported rather than hidden**: 2,132,462,512 of
2,583,306,240 values (82.55 %) do not fit bf16 exactly. An NVFP4 value is
`e2m1 (3 bits) x ue4m3 (4 bits) x an f32 global scale`, which carries more mantissa than bf16's
eight bits. The residue is about 2^-9 relative, against NVFP4's own ~2^-4 — the control is about
thirty times closer to the FP4 weights than the FP4 weights are to the originals, which is what
makes it a control and not a second experiment. It is not zero, and section 4 is what settles
whether it matters.

Both files, 2026-09-18, on this machine: 495 tensors, 2,583,306,240 values, 5.17 GB payload,
3 s from the originals and 24 s from the container.

## 3. The engine side

`CROW_CNQ_OVERLAY=<path>`, read in `boot::open_model`, so `decode`, `parity` and `serve` all get
it through the one front door. `docs/env.md` carries the row.

**One lookup.** `Cnq::attach_overlay` opens the second container and `Cnq::find` asks it FIRST.
A tensor the overlay names shadows the base tensor for every reader in the engine; no loader, no
kernel and no call site has a second lookup. `read_bytes` / `read_range` switch file on
`TensorInfo::overlay` and hand the pages back with `POSIX_FADV_DONTNEED`, the same discipline the
base load keeps.

**The refusals are at the front door**, named, before a byte is loaded: no `overlay` block in the
trailer, a base file name or byte size the overlay was not built over, a tensor the base does not
carry, the same name in another section, a different value count, a different shape, a dtype that
is not `bf16`, or a payload that runs past the file. Nothing about a mismatch can reach a kernel,
where it would be a silently wrong-size GEMV.

**The boot log** names the overlay, its source and build date, the tensors shadowed per kind and
the bytes they cost against the NVFP4 they replace.

### 3.1 The seventeen kinds, and what runs instead

Fourteen of the seventeen were `weights::Fp4` and are `gen::PW` now. Every FP4 launch stayed
VERBATIM — it moved into a closure that `PW::or_bf16` runs when the weight is NVFP4. When it is
bf16 the same call takes the BF16 GEMV/GEMM off the f32 activation the naive FP4 GEMV reads:
`gemv_bf16_w` at t = 1, `gemm_bf16_dense` at t >= 8. Those are the kernels `q_proj`, `k_proj`
and the hyper-connection low-rank have used since the 2026-09-03 keep-set amendment, so the
BF16 path is not new code — only new callers. The one new kernel is `gemv_bf16_bs`,
`gemv_bf16_b` with the explicit y row stride the shared expert's gate|up pair needs: the BF16
twin of `gemv_fp4_bs`.

The other three kinds needed nothing but a loader branch. The two conv1d kinds were already
DEQUANTIZED to f32 at load (`weights::dequant_fp4_dev`) because `conv_silu` / `conv_step` /
`ple_conv` read f32; under the overlay they are WIDENED from bf16 instead, and the kernel is
the same kernel.

| kind | decode launch site | prefill launch site | today | under the overlay | slower |
|---|---|---|---|---|---|
| `linear_attn.in_proj_qkv` | `gdn_step`: grouped `gemv_fp4_mma_g32` slot 0 | `gdn_prompt`: `launch_mma_d` -> `gemm_fp4_dense` | grouped FP4 MMA over the shared `xq_m` | group broken open; `gemv_bf16_w` / `gemm_bf16_dense` off the f32 `mixed` row | yes |
| `linear_attn.in_proj_z` | grouped slot 1 (or `gemv_fp4_mma_d32` under `CROW_GDN_SPLIT_Z`) | `launch_mma_d` | as above | as above | yes |
| `linear_attn.in_proj_b` | grouped slot 2 / `gemv_fp4_mma_d` | `launch_mma_d`, grid 1 | as above | as above | yes |
| `linear_attn.in_proj_a` | grouped slot 3 / `gemv_fp4_mma_d` | `launch_mma_d`, grid 1 | as above | as above | yes |
| `linear_attn.out_proj` | `gemv_fp4_mma_d32` over `xq_v` | `launch_mma_d` | FP4 MMA over the quantized `gnorm` | `gemv_bf16_w` / `gemm_bf16_dense` off the f32 `gnorm` | yes |
| `linear_attn.conv1d` | `conv_step` | `conv_silu` | f32 on the card, DEQUANTIZED at load | f32 on the card, WIDENED at load — same kernel, same launch | no |
| `self_attn.v_proj` | `attn_step`: `gemv_fp4_mma_d` over `xq_m` | `attn_prompt`: `launch_mma_d` | FP4 MMA | `gemv_bf16_w` / `gemm_bf16_dense` off f32 `mixed` | yes |
| `self_attn.o_proj` | `attn_step`: `gemv_fp4_mma_d` over `xq_v` | `attn_prompt`: `launch_mma_d` | FP4 MMA | as above, off f32 `agated` | yes |
| `self_attn.indexer.index_qk_proj` | `attn_step`: `gemv_fp4_mma_d` over `xq_m` | `attn_prompt`: `launch_mma_d` | FP4 MMA | as above, off f32 `mixed` | yes |
| `attn_hyper_connection.block_inject_weight` | `hc_run`: fused `hc_down_inj` (the inject GEMV IS its body) | `hc_run`: `gemv_fp4_b1k` / `gemv_fp4_b` | 19f fusion: down + silu + inject + sig2 in one launch | fusion off; `gemv_bf16_w` (4 rows) plus the separate `silu_div4` / `sigmoid_el` / `sig2_div4` of record | yes |
| `mlp_hyper_connection.block_inject_weight` | as above | as above | as above | as above | yes |
| `mlp.shared_expert.gate_proj` | `moe_run`: fused `sh_gate_up_q` | `launch_mma_d` / `gemv_fp4_bs` | 19h fusion: gate|up in one launch with the silu epilogue | fusion off; the new `gemv_bf16_bs` into the same `[t][1280]` buffer plus `silu_mul640[_q]` | yes |
| `mlp.shared_expert.up_proj` | as above | as above | as above | as above | yes |
| `mlp.shared_expert.down_proj` | `moe_run`: fused `gemv_fp4_mma_dg` (down + `gate_shared` at the store) | `launch_mma_d` / `gemv_fp4_b` | 19h fusion | fusion off; `gemv_bf16_w` / `gemm_bf16_dense` plus the separate `gemv_b` and `gate_shared` | yes |
| `ple.key_proj` | `ple_step_kernels`: `launch_mma_d` over `xq_e` | `ple_run`: `launch_mma_d` | FP4 MMA | `gemv_bf16_w` / `gemm_bf16_dense` off the f32 `emb` row | yes |
| `ple.value_proj` | as above | as above | as above | as above | yes |
| `ple.conv1d` | `ple_conv_step` | `ple_conv` | f32 on the card, DEQUANTIZED at load | f32 on the card, WIDENED at load — same kernel | no |

Every fused or grouped launch is GUARDED, not removed: `PW::is_bf16()` decides, and with no
overlay every guard is false and the fused kernel runs exactly as it did. The BF16 path is
unfused on purpose. #77 buys correctness; speed is a later ticket.

## 4. The control experiment

Nothing about quality is claimed until the wiring is proven, and the control overlay is what
proves it: it carries no new information, so an engine that computes something else under it has
a wiring difference and not a weight difference.

### 4.1 The decisive reading: the package self-test

`decode selftest` compares the engine's own layer 0 and layer 3 outputs against goldens produced
from the UNQUANTIZED originals (`selftest/manifest.json`, F5 of #64 and #69). It is the one
reference-based instrument on this machine, and it answers both questions at once.

| arm | layer 0 `max_abs` | layer 0 `rel_L2` | layer 3 `max_abs` | layer 3 `rel_L2` |
|---|---|---|---|---|
| no overlay (NVFP4 of record) | 9.184837e-2 | 1.3671e-2 | 4.473233e-1 | 1.2821e-1 |
| CONTROL overlay (bf16 dequant of the same NVFP4) | 9.214520e-2 | 1.3648e-2 | 4.511342e-1 | 1.2821e-1 |
| ORIGINALS overlay (the BF16 originals) | 1.079398e-2 | 1.5586e-3 | 1.092313e-1 | 2.8969e-2 |

All three PASS both gates (0.125 and 0.625). Two things follow, and they are the whole point of
this section:

- **The wiring is right.** The control lands on top of the FP4 arm — `rel_L2` 1.3648e-2 against
  1.3671e-2 at layer 0 and 1.2821e-1 against 1.2821e-1 at layer 3. The BF16 path computes what
  the FP4 path computes when it is handed the same numbers, through different kernels, with the
  fusions off.
- **The originals are a real improvement in the only reference-based reading available**: 8.8x
  closer at layer 0 (`rel_L2` 1.3671e-2 -> 1.5586e-3) and 4.4x closer at layer 3
  (1.2821e-1 -> 2.8969e-2). That is the dense quantization error being removed, measured against
  the unquantized model.

### 4.2 The parity forms, and why a logit delta is the weaker reading

`decode parity` on the 8-row form of record, 12 rows (8 teacher-forced, 4 autoregressive),
against the no-overlay run of the same binary:

| arm | teacher-forced rows 0..7 | autoregressive rows 8..11 | `decode run` 32 greedy ids |
|---|---|---|---|
| CONTROL overlay | `max_abs` 3.04, `mean_abs` 0.267, argmax 8 of 8 | `max_abs` 31.1, `mean_abs` 1.070, argmax 1 of 4 | differ from the ids of record at position 1 |
| ORIGINALS overlay | `max_abs` 6.81, `mean_abs` 0.546, argmax 8 of 8 | `max_abs` 4.15, `mean_abs` 0.504, argmax 4 of 4 | IDENTICAL to the 32 ids of record |

The teacher-forced half is the honest half: every argmax agrees on both arms. The
autoregressive half is not a second measurement — once one argmax flips, the next rows are a
different continuation, and the numbers there measure that, not the arithmetic.

The originals arm reproducing the 32 greedy ids of record while the control arm does not is
luck, not evidence in either direction; the ablation below is what shows why.

### 4.3 Why the logit delta is as large as it is

A perturbation of 2^-9 on the dense weights producing a mean logit delta of 0.27 over 248,320
vocabulary entries looks wrong until it is measured per kind. A CONTROL overlay was built for
each of the 17 kinds alone (`--kinds`, which is what that flag is for) and each run compared
with the no-overlay run on the teacher-forced rows:

| kind alone (control) | values | `max_abs` | `mean_abs` | argmax 0..7 |
|---|---|---|---|---|
| `self_attn.indexer.index_qk_proj` | 19,660,800 | 0.0 | 0.0 | 8 of 8 |
| `linear_attn.in_proj_a` | 4,423,680 | 1.34 | 0.109 | 8 of 8 |
| `self_attn.o_proj` | 188,743,680 | 1.47 | 0.121 | 8 of 8 |
| `linear_attn.in_proj_qkv` | 943,718,400 | 1.88 | 0.153 | 8 of 8 |
| `mlp.shared_expert.gate_proj` | 78,643,200 | 2.62 | 0.177 | 8 of 8 |
| `self_attn.v_proj` | 15,728,640 | 2.70 | 0.200 | 8 of 8 |
| `ple.key_proj` | 26,214,400 | 2.62 | 0.230 | 8 of 8 |
| `ple.value_proj` | 6,553,600 | 3.41 | 0.259 | 8 of 8 |
| `linear_attn.in_proj_b` | 4,423,680 | 2.79 | 0.259 | 8 of 8 |
| `ple.conv1d` | 40,960 | 3.10 | 0.261 | 8 of 8 |
| `mlp.shared_expert.up_proj` | 78,643,200 | 3.66 | 0.272 | 8 of 8 |
| `linear_attn.out_proj` | 566,231,040 | 3.75 | 0.273 | 8 of 8 |
| `linear_attn.conv1d` | 1,474,560 | 3.93 | 0.282 | 8 of 8 |
| `mlp.shared_expert.down_proj` | 78,643,200 | 3.63 | 0.284 | 8 of 8 |
| `linear_attn.in_proj_z` | 566,231,040 | 3.93 | 0.284 | 8 of 8 |
| `mlp_hyper_connection.block_inject_weight` | 1,966,080 | 3.89 | 0.295 | 8 of 8 |
| `attn_hyper_connection.block_inject_weight` | 1,966,080 | 3.89 | 0.302 | 8 of 8 |
| **all 17 together** | 2,583,306,240 | 3.04 | 0.267 | 8 of 8 |

Read it: the delta does NOT scale with the number of values, and seventeen kinds together give
the same number as almost any one of them alone. `ple.conv1d` is ONE tensor of 40,960 values in
layer 1 and reaches 0.261; `linear_attn.in_proj_qkv` is 943,718,400 values over 36 layers and
reaches 0.153. This is saturation: a perturbation anywhere near the front of a 48-layer residual
stream decorrelates the logits to the same typical distance, and the level of that distance is a
property of the model, not of the perturbation. It is the same phenomenon `docs/architecture.md`
already records for the NVRTC/driver JIT, where the Windows and Linux 512-row logits drift from
row 23 while all 517 ids stay identical.

`self_attn.indexer.index_qk_proj` at exactly 0.0 is not a dead path: the indexer's output feeds
BLOCK SELECTION only, at 8 tokens there are 2 complete blocks against a `block_topk` of 512, the
selection is dense by construction, and the same blocks selected means bit-identical logits.

A second control rules the NVFP4 ACTIVATION cascade out as the source. With `CROW_MMA_DENSE=0`
on BOTH arms the dense FP4 projections take `gemv_fp4_b` / `gemv_fp4` on the f32 activation and
quantize no activation at all; the control's teacher-forced delta is then `max_abs` 3.50,
`mean_abs` 0.256 — the same as with the cascade on (3.04 / 0.267). So the difference is the
2^-9 weight rounding of the control, amplified, and not the cascade.

## 5. The measurement with the originals

### 5.1 It does not fit the default operating point, and that is a result

The dense set grows from 6.63 GiB to 9.99 GiB, +3.36 GiB, and the residency planner DOES see it:
it derives the dense size from measured free VRAM after the load (`gen.rs`, `dense_measured`),
not from a constant, and it clamps N. But the clamp is two-sided (`docs/architecture.md` 2.1):
VRAM lowers N, and a lower N RAISES the pinned cold tier, because every expert that is not hot
is pinned. 27 hot experts fewer per layer cost about 3.4 GiB more pinned host RAM, and the
pinned budget of record — the 46 GiB cap of `geo.rs` — has only about 1.6 GiB of headroom.

So at the default operating point the engine REFUSES, by name and at boot, exactly as it is
supposed to:

```
refusing config: no hot-set size fits BOTH the VRAM budget (free 19.22 GiB) and the host
pinned budget (46.0 GiB) — shrink the chunk/scratch, the PLE cache, or the keep-set (spec 2.1)
```

Every measurement below therefore runs with `CROW_PINNED_BUDGET_GB=50` (the switch that exists
for exactly this: pinning the budget for a measurement). This machine reports 60.6 GiB free for
pinning, so 50 GiB is inside it, but it is NOT the operating point of record and nothing here
should be quoted as if it were.

### 5.2 `[budget]`, with and without

`tools/serve-linux.sh --port 8099`, n_ctx 200,000, chunk 2048. The script passes every `CROW_*`
of the caller through its `systemd-run` scope, `CROW_CNQ_OVERLAY` included.

| `[budget]` | no overlay (A1, 2026-09-18) | originals overlay (C1) |
|---|---|---|
| free VRAM at start of the plan | 22.63 GiB | 19.22 GiB |
| dense resident | 6.63 GiB | 9.99 GiB |
| hot experts per layer, planned N | 160 -> 155 | 160 -> 128 |
| hot experts per layer, resident | 148 | 121 |
| host pinned budget | 46.00 GiB (configured cap) | 50.00 GiB (`CROW_PINNED_BUDGET_GB`) |
| VRAM used after load | 30.80 GiB | 30.88 GiB |
| free VRAM after load | 0.57 GiB | 0.50 GiB |

### 5.3 Speed

`decode run decode_out/srv-a5-t1read-ids.json 256`, the 16,064-id t1-read form the README's
operating point of record uses, adjacent runs, same session, same binary:

| | no overlay | originals overlay |
|---|---|---|
| planned N | 160 -> 148 (141 resident) | 160 -> 121 (114 resident) |
| prefill, 16,064 ids | 16.33 s = 984 tok/s | 22.48 s = 715 tok/s |
| decode, 256 steps, context 16,320 | 23.68 ms/token = **42.2 tok/s** | 27.30 ms/token = **36.6 tok/s** |
| cold experts per token | 210.7 | 232.8 |

-13 % decode and -27 % prefill, and BOTH costs are in the table twice: the unfused BF16 kernels
read 3.6x the weight bytes of the NVFP4 ones, and the smaller hot set streams 22 more cold
experts per token over PCIe. The two are not separated here.

### 5.4 The quality probe

`tools/quality-probe.py`, the #75 harness, same prompt set (version 1), same seeds
(1201/1202/1203), same row, thinking off, 36 generations per arm. `C1-crow-dense-bf16` is the
originals overlay, `C0-crow-control` the dequantized control, `A1-crow` and `B1-llama` the
baselines of `docs/quality-probe.md`. A1 and A3 are the SAME arm on two seed lists, and the
difference between them is the noise floor this table has to be read against.

| reading (loanwords removed) | A1 `serve` FP4 | A3 same arm, other seeds | C0 control overlay | C1 originals overlay | B1 llama-server |
|---|---|---|---|---|---|
| **DE long prose, 4 tasks** | **16.40** | **15.76** | **15.81** | **14.61** | **8.21** |
| DE pooled, all German tasks | 16.05 | 16.79 | 15.54 | 14.53 | 8.49 |
| DE agentic plans, 2 tasks | 14.75 | 20.72 | 20.43 | 16.22 | 11.61 |
| DE literal task | 14.66 | 20.52 | 8.71 | 12.41 | 6.89 |
| EN long prose, 2 tasks | 6.58 | 10.24 | 7.62 | 5.65 | 9.16 |
| literal occurrence share | 0.996 (0.966-1.000) | 0.959 (0.793-1.000) | 0.955 (0.667-1.000) | 0.978 (0.833-1.000) | 1.000 |
| JSON: valid document / shape ok | 5 / 4 of 6 | 4 / 4 of 6 | 4 / 4 of 6 | 4 / 4 of 6 | 6 / 6 of 6 |
| distinct-word ratio | 0.535 (0.400-0.748) | 0.524 (0.400-0.805) | 0.532 (0.400-0.701) | 0.536 (0.400-0.695) | 0.546 (0.455-0.900) |
| longest immediate repeat run | 5 | 5 | 5 | 5 | 2 |
| decode through `serve`, mean | 62.55 tok/s | 62.90 | 51.37 (45.07-55.09) | 51.17 (45.50-55.25) | 47.58 |

The per-generation spread of the DE long-prose reading is in the same range on every crow arm
(A1 8.69 to 33.73, C1 7.49 to 29.59) and the pooled figure is flags over words, not a mean of
means, so a long answer weighs what it is worth.

**The noise floor (#75 section 6).** `serve` is byte-reproducible at a fixed seed, so run-to-run
noise at the same seeds is ZERO; the only noise a probe run carries is seed and prompt variance,
and A1 against A3 measures it at **0.64 per 1000 on the DE long-prose reading**. The English
reading, the literal share and the six JSON generations are all too noisy at three seeds to
decide anything on their own — that has not changed.

**C0 is the control, and it behaves like one.** The control overlay reads 15.81 against A1's
16.40 — a difference of 0.59, BELOW the 0.64 seed noise, on an arm that shares C1's kernel path
exactly (same unfused BF16 GEMVs, same fusions off, same N = 128, same +3.36 GiB dense). So the
kernel path costs nothing the probe can see, and whatever C1 shows is the WEIGHTS.

**The reading.** C1 improves the German long-prose non-word rate from 16.40 to 14.61 against A1
(**-1.79 per 1000, about 2.8x the seed noise**) and from 15.81 to 14.61 against its own control
(**-1.20, about 1.9x**). Both are real and both are small. The gap to llama-server is 8.19 per
1000; the whole dense path in BF16 closes **15 to 22 % of it**. Everything else about the arm
moves inside its own noise: the literal share is 0.996 -> 0.978 against A1 but 0.955 -> 0.978
against C0, English 6.58 -> 5.65 against an English seed noise of 3.66, JSON 4 of 6 either way.

The KIND of breakage is unchanged, which is the more important half of the reading. Ten of the
109 flagged German prose occurrences of C1, verbatim with their context, drawn at random
(seed 175) after loanwords were removed:

- `...nutzen **Architekturs** eine mehrstufige Speicherhierarchie...` (`de-prose-speicher`, 1201)
- `...das Plakat als statisches, museales Schild **wirkung**, ohne dass...` (`de-prose-plakat`, 1201)
- `...schließt sich eine**adsor** **Feinstfiltration** über Sandfilter an...` (`de-prose-wasserwerk`, 1202)
- `...unterliegt jedoch einer der ständigen **Bewachungrung**, denn auf dem Weg vom Hochbehälter...` (`de-prose-wasserwerk`, 1202)
- `...Materialabweichungen der **Buchblocke** und den Vorlieben des Kunden...` (`de-prose-buchbinderei`, 1203)
- `...die Erwartungshaltung des **Betrachers** steuert...` (`de-prose-plakat`, 1202)
- `...die Formensprache der ausgewählten Bildmotive **anleihen**...` (`de-prose-plakat`, 1203)
- `...Die Werkstoffe waren **naturgewachsenen** Ursprungs...` (`de-prose-buchbinderei`, 1202)
- `...unterwarf sich den Gesetzen der **Marktgängigkeit**...` (`de-prose-buchbinderei`, 1202, a hunspell false positive)
- `...eine progressive **Festigkeitsminderung** durch Carbonatisierung...` (`de-prose-wasserwerk`, 1203, a hunspell false positive)

Truncated stems (`wirkung`, `adsor`), invented compounds (`Bewachungrung`), a wrong plural
(`Buchblocke`, `Architekturs`) and a dropped letter (`Betrachers`) — the exact catalogue #75
recorded for arm A, at a slightly lower rate. Arm B's errors are still of a different kind
entirely.

### 5.5 The live flow

Through Crow's real sender (`~/.local/share/crow/venv/bin/python`, `crow_core.stream_reply` from
`~/Projects/crow/cli`) against the originals overlay on port 8099, plus the image path through
`tools/vit-colorprobe.py`. `tools/serve-linux.sh` reported `2 CROW_* passed through`, which is
`CROW_CNQ_OVERLAY` and `CROW_PINNED_BUDGET_GB` reaching the engine inside the scope.

| turn | what happened |
|---|---|
| text, no `reasoning_effort` | the model reached for a tool instead of answering (`web_search`, 59 B of arguments, valid JSON) — Crow's own TOOLS block travels on every turn, so this is ordinary |
| text, `reasoning_effort="high"` | 4,487 characters of `reasoning_content` and 819 of content, German, on topic. `[chat] thinking on (request): reasoning_effort xhigh, asked as "high"; the generation prompt ends in <think> and the reasoning filter starts Inside (#74)` — 1,440 tokens generated, 205 content chunks against 1,233 reasoning chunks, 1 think tag stripped, decode 49.2 tok/s |
| tool round | `read_file` with `{"path": "...README.md", "start_line": 1, "end_line": 5}`, valid JSON |
| big `write_file` | ONE call, 5,785 B of arguments, valid JSON, a complete Python module in `content` |
| image | `tools/vit-colorprobe.py` 11 of 11 correct (seven solid colours, the split image, both bars, the circle) |

The `[chat]` line of the `write_file` turn: prompt 5,511 tok (5,394 cached, 117 prefilled),
generated 1,686 tok, prefill 223.0 tok/s, decode 47.5 tok/s, finish `tool_calls`, 1 tool call,
no repeat warning. The prefix cache works under the overlay — 5,394 of 5,511 prompt tokens were
cached on the fourth turn. The `[vit]` and `[budget]` lines are the ones quoted above.

### 5.6 How to re-run all of it

```
cd converter && cargo build --release && cd ../engine && cargo build --release
./converter/target/release/converter dense-overlay --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
    --out converter/dense-bf16-originals.cnq \
    --from-originals models/Qwen3.8-Flash-Next-original/dense/dense.safetensors
./converter/target/release/converter dense-overlay --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
    --out converter/dense-bf16-control.cnq --from-container converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq

tools/gate-linux.sh                                   # no overlay: ALL GREEN at the values of record
CROW_CNQ=... CROW_HOTSETS=... CROW_CNQ_OVERLAY=$PWD/converter/dense-bf16-control.cnq \
    CROW_PINNED_BUDGET_GB=48 decode selftest selftest # the wiring proof of 4.1

CROW_CNQ_OVERLAY=$PWD/converter/dense-bf16-originals.cnq CROW_PINNED_BUDGET_GB=50 \
    tools/serve-linux.sh --port 8099 &
tools/quality-probe.py --label C1-crow-dense-bf16 --base-url http://127.0.0.1:8099
tools/quality-probe.py --compare A1-crow C1-crow-dense-bf16
```

There is deliberately NO gate item for any of this. Every arm needs a 5.17 GB file that is not
in the repository and cannot be, and the gate's job is the DEFAULT path — where the values of
record are unchanged and already green. The cheap part that IS kept is the ten-row refusal table
of `Cnq::attach_overlay`, three pure tests in `cnq.rs` with no GPU, no container and no overlay
file (`tools/gate-linux.sh` TESTS 228 -> 231).

## 6. The honest reading

The BF16 dense path does NOT remove the symptom. It reduces it: German long-prose non-words
fall from 16.40 to 14.61 per 1000 (2.8x the seed noise against A1, 1.9x against its own control
C0), and the 8.19 gap to the Unsloth GGUF arm is 6.40 afterwards. The broken words are the same broken words at a slightly lower
rate. Against the unquantized goldens the dense path itself became 8.8x more accurate at layer 0
— so the quantization error this ticket removed was real, it was removed, and removing it bought
about a fifth of the symptom.

That points the next step away from the dense path: 97 % of this container's bytes are routed
experts at RTN NVFP4, and the Unsloth GGUF that answers cleanly quantizes those too, but with
importance-weighted scales rather than round-to-nearest. Engine numerics are the other open
suspect the control did NOT rule out: the control only proves that the BF16 path computes what
the FP4 path computes, not that either is what the model should compute.

## 7. What is NOT decided

- **Which kinds are needed.** 4.3 ablates per kind for WIRING, not for quality; the probe was
  run on all 17 together. The per-kind quality question is a later ticket, and `--kinds` is
  already the tool for it.
- **BF16 or FP8.** FP8 E4M3 would halve the 3.36 GiB and cost about 12 hot experts instead of
  27. Nothing here measures it.
- **Shipping a baked container.** The overlay is a measurement instrument. A shipped mixed
  container would put these tensors in the base file and drop the second open entirely.
- **The unfused cost.** Every BF16 site is deliberately unfused. A fused BF16 chain (a bf16
  `hc_down_inj`, a bf16 `sh_gate_up`) is a speed ticket, not a correctness one.

## 8. The default that lasted five hours (2026-09-23, #91)

**What happened.** On 2026-09-23 `tools/serve-linux.sh` briefly made this overlay the default,
then reverted it:

- `0924406` (13:57) loaded `converter/dense-bf16-originals.cnq` unless `CROW_CNQ_OVERLAY` was
  set. It used the pinned settings the measured dense arm booted with: budget 50 GiB, alloc
  `wc`, margin 1.
- `0254ed6` (18:43) reverted it. The script now boots the bare container with the default
  pinned budget and allocation.

**The script today.** The overlay is opt-in: `CROW_CNQ_OVERLAY=<file>` loads it.
`CROW_CNQ_OVERLAY=none` is accepted, and the script unsets it. An empty value also means no
overlay, because `boot.rs` ignores an empty value. The Linux default for pinned allocation is
now `CROW_PINNED_ALLOC=register` (#103, `e46090c`). `wc` and `host` stay selectable, and
`CROW_PINNED_WC` is no longer read. The pinned budget of §5.1 and §5.2 (the 46 GiB cap, and
`CROW_PINNED_BUDGET_GB=50` for the overlay arms) is unchanged by this. The runs in this document
used the allocation of their day, and none was re-run under `register`.

**What the overlay measured before the fix.** Multi-site probe, 2026-09-23: 23 corrupt tool-call
sites (9 with fresh context), teacher-forced, prompts 18k to 103k tokens. Hot-set sidecar
`hotsets-M-longctx2100-n160.json`, KV fp8_e4m3, build before the PLE fix. Per arm: corrupt wins,
mean margin, correct top-1.

| arm | corrupt wins /23 | fresh /9 | mean margin | correct top-1 |
|---|---|---|---|---|
| bare (`70a69e3`, pinned 46) | 15 | 6 | -1.67 | 8 |
| placebo (attn-v-out control overlay, pinned 47) | 15 | - | -2.01 | 8 |
| dense (originals overlay, pinned 50 WC) | 12 | 3 | -0.23 | 11 |
| dense + `CROW_KV=bf16` (pinned 52) | 12 | 4 | -0.10 | 9 |

- **Short-prompt decode** (probe `speed`, 256 tokens, median of 3, before the fix): bare
  66.6 tok/s, dense overlay 55.8 tok/s.
- **Group bisection** (14-site subset, before the fix): each dense kind group was enabled alone
  (attn v/o, GDN in-proj, GDN out-proj, HC, indexer, PLE+HC, PLE, rest, shared experts). Each
  had 8 to 11 of 14 sites corrupt. No single group reproduces the dense arm.

**Most of that benefit was masking the PLE read bug.** The cause of the corruption was the PLE
n-gram row read at the wrong container offset. It is fixed in `85a48e7`
(`docs/numerics-diff.md` §7.2). With the fix and the same overlay, the `plefix` arm has 4/23
corrupt wins, 0/9 fresh, mean margin +8.40 and correct top-1 17. llama.cpp UD-Q2_K_XL has 4/23.
The dense arm's 15 -> 12 is small next to 12 -> 4. **The bare container with the fix has NO
multi-site number:** that boot (`plefix-bare`, 18:41) panicked with `CUDA_ERROR_INVALID_CONTEXT`
(`cuda.rs:252`). So what the overlay adds on top of the fix is not measured.

**Why it was reverted.** It cost the operating point. At pinned 50 GiB WC, robin's serve was
OOM-killed at 18:38 CEST, with the desktop left about 14 GiB. The hot set was 128 instead of 156,
and that run read prefill 500 to 600 tok/s and decode 33 to 37 tok/s. Section 7's open questions
(which kinds are needed, BF16 or FP8) stand. They are now to be asked on the fixed engine, with
paired oracle-KLD as the acceptance rule (`docs/improve-loop.md`).
