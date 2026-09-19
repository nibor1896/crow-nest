# The routed experts, re-quantized with an importance matrix

Step 5 of the requant series (issue #79), after the characterization harness of #75, the BF16
originals of #76, the dense overlay of #77 and the f32 oracle of #78. It is the first step that
touches the 97 % of this container's bytes nobody has touched yet.

- `docs/dense-originals.md` — where the BF16 originals come from and the proof that they are
  the originals. #79 uses the same tool and the same proof, pointed at the routed experts.
- `docs/dense-overlay.md` — the overlay mechanism, and the discipline every arm here follows:
  the default path stays byte-identical, the control runs before any quality claim.
- `docs/oracle-kld.md` — the instrument every number in section 7 is read with, and the
  paired noise floor of 0.025 the stop rule is decided against.

**The answer, up front.** It does not work. Three importance- and search-based re-quantizations
of 8 of 48 layers' routed experts all lower the weight-space error — one of them by 20 % on the
very quantity an importance matrix is supposed to buy — and all three land the model FURTHER
from its own f32 oracle, by 1 to 4 times the paired noise floor, with sign tests down to
6.5e-22. The stop rule is negative and the remaining 40 layers are not worth fetching.
Section 8 is the verdict in full; section 7.5 is the part worth keeping.

## 1. Why the experts, and why an importance matrix

`CNQ4.5-M` is 104.73 GB, and 97 % of it is `mlp.experts.gate_up_proj` and
`mlp.experts.down_proj` — 48 layers x 512 experts, RTN NVFP4 with plain-MSE sub-block scales.
#78 measured what the whole container costs against the f32 oracle (mean KLD 0.461 on 607 rows,
82.4 % same top-1, which is llama.cpp's `q2_K` band, not its `q4_K_M` band) and ruled out the
dense weights, the FP8 KV cache, the NVFP4 activation cascade and run-to-run noise as the main
source. The experts were what was left.

The lever is not more bits. It is WHICH error the scale search is allowed to make. NVFP4's block
runs along the row: 64 values, four sub-blocks of 16, one `ue4m3` scale per sub-block. Inside one
sub-block every one of the 16 values belongs to a DIFFERENT input column of the same output row.
A scale chosen by the unweighted squared error of those 16 weights spends its resolution evenly
over columns the model never excites and columns it lives on. An importance matrix is the
measurement that tells them apart.

## 2. The importance matrix, and where it comes from

| item | value |
|---|---|
| file | `models/unsloth-imatrix/imatrix_unsloth.gguf` (580,038,720 B) |
| sha256 | `a5863123db1ca458727e738955bef7bfc199520aa2bee3a30142a1aff9254154` |
| source | `unsloth/Qwen3.8-Flash-Next-GGUF` on Hugging Face, file `imatrix_unsloth.gguf_file` |
| the repo's record | LFS `sha256` `a5863123db1ca458…`, size 580,038,720 B — the same bytes, byte for byte |
| repo revision | `38bb39ee97821de2c9009abb7e93950eec396e66`, last modified 2026-09-02 |
| licence | the repo's card says `license: other`, `license_name: qwen-community-1.0`, `license_link: LICENSE` — the base model's licence, not a separate one for the matrix |
| what it is | GGUF v3, 1852 F32 tensors, kv `general.type imatrix`, `imatrix.chunk_count 45`, `imatrix.chunk_size 18432`, `imatrix.datasets ["unsloth_calibration_Qwen3.8-Flash-Next.txt"]` |

It is not a model and it holds no weights. It is what `llama-imatrix` accumulated while 45
chunks of 18,432 tokens of Unsloth's own calibration text ran through the BF16 model: per layer
`N`,

```
blk.N.ffn_gate_exps.weight.in_sum2  [2560, 512]   blk.N.ffn_gate_exps.weight.counts  [1, 512]
blk.N.ffn_up_exps.weight.in_sum2    [2560, 512]   blk.N.ffn_up_exps.weight.counts    [1, 512]
blk.N.ffn_down_exps.weight.in_sum2  [ 640, 512]   blk.N.ffn_down_exps.weight.counts  [1, 512]
```

GGUF lists dimensions innermost first, so `ne[0]` is the INPUT column index and `ne[1] = 512` the
expert: element `[e][j]` sits at `e * ne[0] + j`. The semantics are read out of
`tools/imatrix/imatrix.cpp`, not guessed — `in_sum2[e][j] += x[j] * x[j]` for every token routed
to expert `e`, `counts[e]++` for every such routing — and the weight the quantizer gets is
`tools/quantize/quantize.cpp`'s own GGUF branch:

```
qw[e][j] = counts[e] > 0 ? in_sum2[e][j] / counts[e] : 1
```

the mean squared activation of input column `j` for expert `e`, and a FLAT 1 for an expert the
calibration never routed to. That fallback is llama.cpp's and it is kept deliberately: MoEQuant
(arXiv 2505.03804) names rare experts as the weak spot of every calibration-based method, and a
row of zeros would let a rare expert's weights round anywhere at all.

**Rare experts, counted.** Over all 48 layers there are 24,576 (layer, expert) slots. **8** have
`counts == 0` and take the flat fallback; **17** more have `0 < counts < 100` against a mean of
16,518. In the eight pilot layers: layer 1 has the one zero-count expert, layer 37 has one with
`counts = 92`, and the other six layers have neither. `counts` is identical for `gate`, `up` and
`down` within a layer, which is what routing implies.

**`gate` and `up` carry the SAME matrix, and that is checked rather than assumed.** This
architecture stores gate and up FUSED: `mlp.experts.gate_up_proj` is `[512, 1280, 2560]`, 1280
output rows per expert that are gate and up together, and one imatrix row per expert has to cover
both. It does, because both read the same activation vector, so `imatrix.cpp` accumulates the same
`x[j]^2` into both entries. `blk.N.ffn_gate_exps.weight.in_sum2` and
`blk.N.ffn_up_exps.weight.in_sum2` are BITWISE identical in **48 of 48 layers**, and
`converter/src/imatrix.rs` REFUSES the fused tensor if they ever stop being. So the gate/up split
inside the 1280 rows never has to be known.

### 2.1 The expert axis is aligned, and here is the evidence

The imatrix is indexed by the GGUF's expert number and the container by the checkpoint's. If a
conversion had permuted them, every weight would be weighted by the wrong expert's activations and
the result would be a slow, plausible-looking way of measuring nothing.

There is an independent routing measurement on this machine that never went through llama.cpp:
`decode_out/hotsets-M-longctx2100-n160.json`, the top-160 experts per layer by frequency from
`decode`'s own warm-up over 2,100 real tokens, in CONTAINER expert ids. Overlap it with the
imatrix's own top-160 by `counts`:

| | value |
|---|---|
| chance overlap (hypergeometric, 160 of 512 twice) | 50.0 ± 4.87 per layer |
| measured, mean over 48 layers | **74.4** of 160 |
| layers above chance | 46 of 48 |
| z of the mean over 48 layers | **34.8** |
| the eight pilot layers | 118, 75, 46, 55, 66, 73, 87, 86 |

Two different calibration corpora — Unsloth's text and a 2,100-token long-context id list — agree
on which experts this model uses far beyond chance. A permuted expert axis would land exactly at
50. Layer 13 at 46 and layer 25 at 66 show how much of the per-layer spread is corpus and not
alignment; the aggregate is what decides.

The check needs no tool of its own: `converter imatrix-show <file> blk.N.ffn_gate_exps.weight.counts`
prints the counts (section 5 shows it agreeing with a stdlib Python GGUF reader on dims, data
offsets, first and last values and the sum), and the hot-set sidecar is plain JSON.

### 2.2 How much structure is there to exploit

The weighted rule can only move a scale when the 16 columns of one sub-block disagree about their
importance. Median ratio of the largest to the smallest importance weight inside a 16-column
sub-block, sampled over 14 experts per tensor:

| layer | `gate_up` median / p90 / max | `down` median / p90 / max |
|---|---|---|
| 1 | 14.49 / 56.88 / 1769.61 | 10.75 / 46.98 / 1222.39 |
| 13 | 4.70 / 9.33 / 65.16 | 4.35 / 11.31 / 66.35 |
| 25 | 3.99 / 9.27 / 35.89 | 3.83 / 8.57 / 31.80 |
| 43 | 3.48 / 9.28 / 56.88 | 4.82 / 11.26 / 70.49 |

So there IS structure, it is largest at the front of the model, and it is a factor of three to
fifteen — not a factor of a thousand. Section 7 reads the result against this.

## 3. The originals: fetching the routed experts

`tools/fetch-dense-originals.py` gained `--experts`, the mirror of #76's rule derived from the
same sidecar: `section == "text"`, `dtype == "nvfp4"`, `.mlp.experts.` IN the name, and the layer
in the list. One safetensors file per layer under
`models/Qwen3.8-Flash-Next-original/experts/layer-NN.safetensors`, resumable and idempotent
exactly as the dense file is. The dense path did not move: the same derivation, the same metadata
string, and a second dense run still re-verifies 495 tensors, fetches 0 and makes 0 requests.

**Pilot layers 1, 7, 13, 19, 25, 31, 37, 43** — eight of 48, evenly spread, one inside the first
eight layers and one inside the last eight, as the ticket asks.

One real change to the fetch loop, and the experts forced it: a coalesced span is now cut into
requests of at most `--chunk-bytes` (128 MiB). One expert `gate_up_proj` is 3,355,443,200 B, and a
single request for it would buffer all of it and make one timeout cost all of it. Proved on the
line before it was used: three dense tensors fetched at a 400,000 B chunk size are byte-identical
to the #76 fetch of record.

### 3.1 The fetch of record, 2026-09-19

| item | value |
|---|---|
| layers | 1, 7, 13, 19, 25, 31, 37, 43 |
| tensors | 16 — `mlp.experts.gate_up_proj` `[512, 1280, 2560]` (3,355,443,200 B) and `mlp.experts.down_proj` `[512, 2560, 640]` (1,677,721,600 B) per layer |
| values | 20,132,659,200 — 22.5 % of the container's routed-expert values, 16.7 % of the model's bytes |
| bytes | 40,265,318,400 B of tensor payload |
| wall time | 02:01:21 to 03:09 local, 67 minutes; the longest single worker ran 4,035 s |
| how | EIGHT processes, one per layer, because one HTTP connection to this host is capped near 1.6 MB/s. Measured before the run: 4 parallel streams gave 6.4 MB/s aggregate against 1.4 MB/s for one. The eight together held **13 MB/s**, a mean of **10.0 MB/s** over the whole 67 minutes |
| requests | 363 (16 shard headers, 347 data ranges of at most 128 MiB) |
| retries | 45 — 40 `TimeoutError`, 2 `URLError`, 3 `IncompleteRead`, each retried after 2.0 s and then served |
| HTTP 429 | **0** |
| output | `models/Qwen3.8-Flash-Next-original/experts/layer-NN.safetensors`, 5,033,165,296 to 5,033,165,304 B each (a 496 to 504 B header plus the payload), 37.5 GiB in total |
| manifest | `layer-NN.manifest.json`, one record per tensor: source shard, absolute byte range, `sha256`, dtype, shape, output offset, plus the run's counters |

The 45 retries are the reason the chunking mattered. At 128 MiB a timeout costs 128 MiB; at
one request per tensor it would have cost 3.36 GB, forty-five times.


## 4. The proof that they are the originals

`converter requant-check --experts`, the #76 proof pointed at a fetched expert layer. It
derives the list a SECOND time from the container's index trailer — `section == "text"`,
`dtype == "nvfp4"`, `.mlp.experts.` in the name, narrowed to the layers the fetched file
carries — runs the SAME `quantize_nvfp4` at the same `--scales mse` the container was built
with, and compares the 36-byte blocks AND the `f32` global scale on its bits.

```
for L in 01 07 13 19 25 31 37 43; do
  converter requant-check models/Qwen3.8-Flash-Next-original/experts/layer-$L.safetensors \
                          converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq --experts
done
```

Result of 2026-09-19, at commit d542eca, one run per layer, 35 to 37 s each on two workers
(one routed-expert tensor is 6.7 GB of `f32`, so `--experts` defaults to two workers instead of
eight):

```
requant-check: 2 of 2 tensors byte-identical including the global scale (2516582400 values, scales mse, 2 threads, 36 s)
```

**16 of 16 tensors, 20,132,659,200 values, byte-identical including the global scale.** Eight
`OK` pairs, no `DIFF` line, exit code 0 every time. The fetched bytes ARE the tensors
`CNQ4.5-M` was built from, and the caveats of `docs/dense-originals.md` 4 apply here unchanged:
this is a checksum with arithmetic in it, it says nothing about quality, and it cannot see a
difference the same NVFP4 rounding would have collapsed anyway.

The dense path is unaffected: `converter requant-check` without `--experts` still reports
`495 of 495 tensors byte-identical including the global scale`.


## 5. The converter: four rules on one grid

`converter expert-overlay` is additive in exactly the way `requant-check` (#76) and
`dense-overlay` (#77) are: the word `expert-overlay` is taken off the front of the argument list
and the conversion path below never sees it. `converter <model-dir> <out.cnq>` reads and writes
what it always did, and `requant-check` still reports 495 of 495.

```
converter expert-overlay --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
                         --out  converter/expert-<rule>.cnq \
                         --originals models/Qwen3.8-Flash-Next-original/experts \
                         --layers 1,7,13,19,25,31,37,43 \
                         --rule mse|imatrix|imatrix46 \
                         [--imatrix models/unsloth-imatrix/imatrix_unsloth.gguf] \
                         [--report decode_out/79/weights-<rule>.json]
```

`--rule` is NOT optional and has no default. The control and the experiments must never be
confused for one another, so the tool refuses to guess which one was meant.

### 5.1 What stays fixed

The overlay is NVFP4 of the same shape, the same byte length and the same global-scale
convention as the base container's, so nothing downstream can tell the two apart by geometry:

- the GLOBAL scale is still `max over sub-blocks(max|x| / 6) / 448`, computed from the same
  values by the same walk. A better convention may exist; it is not this ticket's to change,
  because moving it would move the one number `gs_dev` hands every expert kernel.
- a 64-value block is still four `ue4m3` sub-block scale bytes plus 32 B of packed E2M1
  nibbles, 36 B, rows contiguous.
- `1280 * 2560` and `2560 * 640` and `2560` and `640` are all multiples of 64, so a 16-wide
  sub-block belongs to ONE expert and ONE output row and therefore to 16 consecutive input
  columns. `converter/src/expert_overlay.rs` asserts it out of the shape instead of assuming it,
  and refuses a geometry where it would not hold.

### 5.2 The four rules

**(a) `mse` — the control.** The conversion's own `--scales mse`: a moment-matched seed, two
fixed-shape least-squares refinements, then the explicit SSE of the two `ue4m3` ladder steps
bracketing the analytic optimum plus the policy ceiling step. It calls `encode_subblock_mse`,
the conversion's own function, not a copy. A unit test asserts that the new code path is
BYTE-IDENTICAL to `quantize_nvfp4` over 1, 3 and 8 threads, and a second one that the thread
count is invisible in the output. That is what makes it a control and not a second experiment.

**(b) `imatrix` — llama.cpp's form.** From `ggml/src/ggml-quants.c`, where both
`quantize_row_q4_K_impl` and `quantize_row_iq4_nl_impl` build

```
sigma2    = 2 * sum(x^2) / super_block_size
weight[l] = qw[l] * sqrtf(sigma2 + x[l]*x[l])
```

and then minimize the WEIGHTED squared error over the scale. `iq4_nl` is the closer twin,
because it searches a scale for a FIXED non-uniform 16-level grid — the same shape of problem
E2M1 poses: it takes `d = sumqx/sumq2`, the weighted least-squares scale for the current levels,
tries neighbours, and keeps the best by `sumqx^2/sumq2`, which is exactly the smallest weighted
SSE. Three differences, and they are named rather than hidden:

- the super block is the 64-value NVFP4 block, the unit four sub-block scales share, so
  `sigma2 = 2 * sum_64(x^2) / 64`;
- the scale is not a free float but one of 128 `ue4m3` ladder steps times the tensor's global
  scale, so the analytic optimum is BRACKETED by its two neighbouring steps and scored against
  them and against the ceiling step — the same three-candidate shape `encode_subblock_mse`
  already has, with weights;
- `qw` is the per-expert importance row of section 2, which is why the expert index and the
  column index both have to be right.

There is no NVFP4 prior art to copy. llama.cpp's own `quantize_nvfp4` takes a `quant_weights`
argument and IGNORES it. What is taken from llama.cpp is the method.

**(c) `imatrix46` — (b) plus four-over-six.** NVIDIA's Nemotron 3 Ultra recipe (arXiv 2512.02010)
found that plain weight-MSE scales gave no consistent downstream gain over max, and picked
instead a per-block choice between putting the block maximum at the top of the grid (6) or at 4,
by reconstruction error. Here that is one more pair of ladder candidates, `max|x| / 4`, scored by
the same weighted SSE — so it is only ever chosen when it wins, and (c) can never be worse than
(b) on (b)'s own objective. A unit test asserts that.

**(d) `mse46` — the second control, added after (c) was first run.** Four-over-six candidates
with `w[l] = 1`: the same candidate set as (c), no importance matrix anywhere in it. It exists
because (c)'s first weight-space numbers read as a large win for the imatrix until (d) was run
beside them, and without it "(c) beats (a)" and "the importance matrix works" are the same
sentence. They are not (section 7.1).

MR-GPTQ (arXiv 2509.23202) reports about one point over RTN for NVFP4, but with a full-Hessian
GPTQ. The imatrix here is DIAGONAL — one number per input column, no cross-column term — so that
method is not available and is not claimed.

### 5.3 What the report measures

`--report` writes one JSON record per tensor, and the same numbers go to stdout as a table. All
of them are read back off the bytes that were WRITTEN, by dequantizing them again:

| column | what it is |
|---|---|
| `mse` | `sum(err^2) / n`, unweighted |
| `imatrix-mse` | `sum(qw_j * err_j^2) / sum(qw_j)` with the PLAIN importance weight, not the `sqrt(sigma2 + x^2)` factor of the search. It is the first-order proxy for the squared error the rounding puts on the layer's OUTPUT: `E[(dw . x)^2] = sum_j dw_j^2 E[x_j^2]` for independent columns. This is the quantity (b) is supposed to lower. It is computed for EVERY rule, the control included, whenever `--imatrix` is given — otherwise the control's column would be a different quantity with the same name (commit d542eca) |
| `clipped%` | share of values whose magnitude ran past `6 x sub-block scale` and saturated |
| `scales%` | share of sub-blocks whose chosen ladder byte DIFFERS from what rule (a) would have chosen — how much the importance matrix actually moved |
| `diff blocks` | 36-byte blocks that differ from the base container's. For rule (a) it must be 0 |


### 5.4 The GGUF reader, checked against a second reader

A header parser that is wrong by one field reads plausible numbers out of the wrong place, so
`converter imatrix-show` exists to be compared with an independent one. Against a stdlib Python
GGUF parser written for the purpose, on the real 580 MB file:

| | `converter imatrix-show` | the Python reader |
|---|---|---|
| header | GGUF v3, 1852 tensors, 4 kv, alignment 32, data_start 122496, 580038720 B | the same |
| `blk.1.ffn_gate_exps.weight.in_sum2` | dims [2560, 512], offset 13414496, n 1310720 | the same |
| its first 8 values | 1955.4269 1537.1399 3920.0618 4663.9893 2029.3301 4385.1533 5106.9272 5105.018 | the same |
| its last 3 | 477.1185 71.90224 772.93317 | the same |
| its sum | 3.362472463e9 | 3.36247246e9 |
| `blk.1.ffn_down_exps.weight.in_sum2` | dims [640, 512], offset 12099136, sum 2.149483608e7 | the same |
| `blk.43.ffn_gate_exps.weight.counts` | dims [1, 512], offset 465713152, sum 8.45728e6 | the same |

Plus eight pure unit tests that build GGUF bytes by hand: the header of a hand-built file, a
prefix too short to hold it (which has to say so rather than guess), a file that is not GGUF v3,
the values coming back, the `sums / counts` rule with its zero-count fallback, the refusal when
gate and up disagree, a negative `in_sum2`, and a non-F32 tensor read as floats.

## 6. The engine: an overlay the same kernels read

`CROW_CNQ_OVERLAY` is reused unchanged; there is no new environment variable. What #77 built —
`Cnq::attach_overlay`, one lookup in `Cnq::find`, `read_bytes` / `read_range` switching file on
`TensorInfo::overlay` — already covers the experts, because every expert byte in this engine is
read through exactly those two calls:

- `residency::expert_slab_info` takes the two tensors through `cnq.find`, so the per-expert slab
  byte counts AND the per-layer global scales that land in `gs_dev` come from the overlay;
- the one ascending sweep in `Residency::build` reads every expert id with
  `cnq.read_range(&gt, id * gu_bytes, gu_bytes)` and sends it to its VRAM hot slot or its pinned
  cold slot. Both destinations are filled from that one read;
- everything after the load — the staging kernel, the zero-copy cold path, `swap_in`,
  `swap_in_bundled` and the stream-side trickle — moves bytes that are already in the pinned
  tier or in VRAM. They never touch the container again, so they carry the overlay's bytes
  without knowing it exists;
- `Cnq::abs_offset`, the one call that would have been wrong (it returns a BASE-container
  offset), is the PLE row prefetch and nothing else, and it carries a `debug_assert!(!t.overlay)`
  since #77.

So the engine change is a widened rule and two refusals, not a new path.

### 6.1 The refusal table, widened

`overlay_refusal` used to end every tensor with "a dense overlay stores bf16 only". It now reads
the dtype AFTER the name, section, value-count and shape checks, and the rule is:

| overlay dtype | rule |
|---|---|
| `bf16` | the dense path of #77. The byte length CHANGES (2 B per value against 36 B per 64) and every reader takes that from `byte_len`. **A routed expert may not come this way** — `residency` cuts per-expert slabs out of `byte_len / 512` and hands them to kernels that index 36 B per 64 values, so a bf16 expert would be a silently wrong-size slab, not a refusal |
| `nvfp4` | the routed experts of #79. The base tensor must also be `nvfp4`, `byte_len` must match to the byte, and the global scale must be finite and positive — a zero or a NaN there would make every block of the tensor zero or NaN and nothing downstream looks at it again |
| anything else | refused by name |

`residency.rs` adds one more refusal that is not about the file: `CROW_COLD_TIER` together with
an expert overlay. A low-bit cold tier is a SECOND file the pinned slabs are filled from, so under
both at once only the hot experts would carry the new bytes and the arm would be half an arm.
Named at boot rather than measured.

The `[overlay]` boot line now says which kind it is and prints the base bytes beside the overlay
bytes, because for an expert overlay those two must read IDENTICAL:

```
[overlay] <path> — kind expert-nvfp4, 16 tensors shadowed, 20132659200 values,
          11.32 GB (base 11.32 GB), source originals:<dir> rule:imatrix, built <date>
```


## 7. The measurement

### 7.1 Weight space: all four rules lower the error they were asked to lower

Every number below is read back off the bytes that were written, over all 16 tensors and
20,132,659,200 values. `imatrix-MSE` is `sum(qw * err^2) / sum(qw)` with the SAME importance
weights in all four rows — that is what commit d542eca fixed, and without it the control's
column would have been a different quantity with the same name.

| rule | plain MSE | vs (a) | importance-weighted MSE | vs (a) | clipped | sub-block scales moved | 36-B blocks differing from the base |
|---|---|---|---|---|---|---|---|
| (a) `mse`, the control | 1.2825e-6 | — | 1.2909e-6 | — | 2.164 % | 0 % | **0 of 314,572,800** |
| (d) `mse46` | 1.0923e-6 | **-14.83 %** | 1.0962e-6 | -15.08 % | 2.914 % | 48.98 % | 293,244,199 |
| (b) `imatrix` | 1.2736e-6 | -0.69 % | 1.1756e-6 | -8.93 % | 4.254 % | 44.08 % | 283,519,038 |
| (c) `imatrix46` | 1.1365e-6 | -11.38 % | **1.0361e-6** | **-19.74 %** | 2.768 % | 57.56 % | 304,309,355 |

Per layer and tensor, plain MSE / importance-weighted MSE, both x 1e-6:

| layer | tensor | (a) `mse` | (d) `mse46` | (b) `imatrix` | (c) `imatrix46` |
|---|---|---|---|---|---|
| 1 | `gate_up_proj` | 1.3116 / 1.3586 | 1.1180 / 1.1441 | 1.3280 / 1.1837 | 1.1968 / 1.0349 |
| 1 | `down_proj` | 1.2384 / 1.2009 | 1.0535 / 1.0186 | 1.2756 / 1.0301 | 1.1430 / 0.9007 |
| 7 | `gate_up_proj` | 1.3024 / 1.3060 | 1.1097 / 1.1114 | 1.2872 / 1.2092 | 1.1485 / 1.0687 |
| 7 | `down_proj` | 1.2296 / 1.2304 | 1.0479 / 1.0478 | 1.2309 / 1.0994 | 1.0993 / 0.9684 |
| 13 | `gate_up_proj` | 1.2463 / 1.2622 | 1.0621 / 1.0711 | 1.2372 / 1.1581 | 1.1037 / 1.0201 |
| 13 | `down_proj` | 1.2122 / 1.2101 | 1.0309 / 1.0282 | 1.2019 / 1.1016 | 1.0680 / 0.9689 |
| 19 | `gate_up_proj` | 1.2593 / 1.2636 | 1.0710 / 1.0738 | 1.2340 / 1.1950 | 1.0963 / 1.0555 |
| 19 | `down_proj` | 1.2188 / 1.2098 | 1.0378 / 1.0309 | 1.2012 / 1.1264 | 1.0691 / 0.9955 |
| 25 | `gate_up_proj` | 1.2360 / 1.2452 | 1.0536 / 1.0587 | 1.2133 / 1.1642 | 1.0846 / 1.0306 |
| 25 | `down_proj` | 1.1909 / 1.1909 | 1.0133 / 1.0123 | 1.1760 / 1.0562 | 1.0454 / 0.9307 |
| 31 | `gate_up_proj` | 1.3103 / 1.3478 | 1.1175 / 1.1386 | 1.2954 / 1.2186 | 1.1585 / 1.0733 |
| 31 | `down_proj` | 1.2519 / 1.2428 | 1.0647 / 1.0578 | 1.2476 / 1.1154 | 1.1089 / 0.9816 |
| 37 | `gate_up_proj` | 1.3761 / 1.3855 | 1.1707 / 1.1759 | 1.3553 / 1.2934 | 1.2062 / 1.1419 |
| 37 | `down_proj` | 1.3404 / 1.3263 | 1.1399 / 1.1311 | 1.3437 / 1.1757 | 1.2009 / 1.0360 |
| 43 | `gate_up_proj` | 1.3473 / 1.3582 | 1.1499 / 1.1563 | 1.3333 / 1.2512 | 1.1870 / 1.1054 |
| 43 | `down_proj` | 1.3185 / 1.3154 | 1.1213 / 1.1213 | 1.3233 / 1.1637 | 1.1794 / 1.0234 |

Read it: every rule wins on its own objective and loses on the other. (b) buys 8.93 % of the
importance-weighted error and gives back nothing worth naming on the plain one. (d), which
never sees the importance matrix, buys 14.83 % of the plain error. (c) has the best weighted
number of all four, and **(d) already has 15.08 % of the 19.74 %** — so three quarters of (c)'s
weighted gain is the WIDER CANDIDATE SET and about one quarter is the importance matrix. That
is why (d) exists, and it is the first thing a reader of this table would otherwise get wrong.

**Rare experts.** The pilot's eight layers carry exactly TWO zero-count expert slots — expert
116 of layer 1, in both its tensors — which take llama.cpp's flat fallback, and TWO tiny-count
ones, expert 361 of layer 37 at `counts = 92`. Over all 48 layers it is 8 zero-count and 17
tiny-count slots of 24,576. They are not what decides anything here.

**One confound checked and cleared.** `residency::sanitize_sf_slab` rewrites any `ue4m3` scale
byte `0x7F` to `0x7E` before a slab reaches the card, because the `mxf4nvf4` tensor-core
instruction makes a NaN of `0x7F`. A wider scale search could produce more of them, and the
engine would then silently compute something the converter did not write. Counted on layer 19's
`gate_up_proj`, 104,857,600 scale bytes: **0** in the base container, **0** under `imatrix`,
**4** under `mse46`. Not a confound.

### 7.2 The control: the wiring, proven before any quality claim

`converter expert-overlay --rule mse` wrote an overlay that is **16 of 16 tensors
byte-identical to the base container including the global scale**, and the engine's boot log
says it was attached and read:

```
[overlay] .../expert-mse-control.cnq — kind expert-nvfp4, 16 tensors shadowed, 20132659200 values,
          11.32 GB (base 11.32 GB), source originals:... rule:mse, built 2026-09-19T01:13:11Z
[overlay]     8 x mlp.experts.down_proj  (6710886400 values)
[overlay]     8 x mlp.experts.gate_up_proj  (13421772800 values)
```

`decode parity` under it, against the `none` dumps of #78:

| set | `none` | control arm `x-ctl` |
|---|---|---|
| 607 rows | `79a4d2b682b4db9e0038f3a32093ed8380ec10cc984bd636927304935546ac75` | **the same sha256** |
| 298 rows | `eee41832c6ccd28efb67d13bce7b4c060f65b69817e45b58bffefc697c169acd` | **the same sha256** |

Byte-identical, both sets, and `oracle-kld.py --paired x-ctl,none` reports
`0.000000 +- 0.000000, 0/0 rows`. 11.32 GB of expert weights were read out of a different file,
through a different descriptor, and the engine computed the same 607 x 248,320 floats. The
wiring is right, and the arms below are weights and not plumbing.

### 7.3 VRAM and speed do not move, which is what "same format" has to mean

`[budget]`, adjacent runs of `decode run decode_out/srv-a5-t1read-ids.json 256`, the 16,064-id
t1-read form:

| `[budget]` | no overlay | `imatrix` overlay |
|---|---|---|
| VRAM total / free at start | 31.37 GiB / 21.77 GiB | 31.37 GiB / 21.77 GiB |
| states+dense+hot measured | 2900.0 MiB (planned 21681.0 MiB, N=148) | 2900.0 MiB (planned 21681.0 MiB, N=148) |
| free VRAM after load | 0.55 GiB | 0.55 GiB |
| VRAM used after load | 30.82 GiB (141 hot experts x 48) | 30.82 GiB (141 hot experts x 48) |
| host pinned budget | 46.00 GiB (the configured cap, unchanged) | 46.00 GiB |
| prefill, 16,064 ids | 16.41 s (979 tok/s) | 16.19 s (992 tok/s) |
| decode, 256 steps, context 16,320 | 23.94 ms/token = **41.8 tok/s** | 23.90 ms/token = **41.8 tok/s** |
| cold experts per timed token | 210.7 | 211.8 |

Identical plan, identical residency, identical speed. `CROW_PINNED_BUDGET_GB` was NOT needed —
unlike #77's dense overlay, which cost 3.36 GiB of VRAM and 27 hot experts per layer, this
overlay costs nothing at all, because it is the same format at the same byte length.

### 7.4 The oracle: all three rules land FURTHER from the f32 model

`tools/oracle-kld.py`, the #78 instrument, `KL(P_f32 || Q_arm)` in nats. The 607-row set
(`decode_out/oracle-t2b`, `--rows 0:607 --prompt-rows 596`):

```
== all rows
arm                     rows  same top-1 as ref    mean KLD                   median      p90      p99    p99.9       max
none                     607   82.37 +- 1.55 %   0.460689 +- 0.048404   0.05074   1.1886   5.7673  10.6430   11.5598
x-ctl                    607   82.37 +- 1.55 %   0.460689 +- 0.048404   0.05074   1.1886   5.7673  10.6430   11.5598
x-im                     607   80.72 +- 1.60 %   0.485422 +- 0.051018   0.06589   1.1499   6.5209  11.1323   12.6017
x-im46                   607   80.07 +- 1.62 %   0.564517 +- 0.057488   0.07305   1.3874   7.7907  12.3880   15.0544
x-mse46                  607   81.05 +- 1.59 %   0.505436 +- 0.053519   0.05628   1.3086   6.5732  11.6726   11.8054

== paired, per row: KLD(A) - KLD(B) on the SAME row
A - B                           rows mean difference              median    A worse  sign test p
x-im - none                      607  0.024733 +-  0.010709   0.000689   389/607      3.77e-12
x-im46 - none                    607  0.103828 +-  0.018096   0.004404   421/607      6.53e-22
x-mse46 - none                   607  0.044747 +-  0.017562   0.000498   355/607      3.33e-05
x-im46 - x-mse46                 607  0.059081 +-  0.017824   0.003374   420/607      1.48e-21
x-ctl - none                     607  0.000000 +-  0.000000   0.000000     0/0               1
```

The 298-row set (`decode_out/oracle-tf298`, `--rows 0:298 --prompt-rows 235`):

```
== all rows
arm                     rows  same top-1 as ref    mean KLD                   median      p90      p99    p99.9       max
none                     298   82.89 +- 2.19 %   0.343108 +- 0.050790   0.07171   0.8179   3.5276   9.2108    9.5268
x-ctl                    298   82.89 +- 2.19 %   0.343108 +- 0.050790   0.07171   0.8179   3.5276   9.2108    9.5268
x-im                     298   80.87 +- 2.28 %   0.354425 +- 0.050034   0.06566   0.9024   2.7663   8.9716    9.2644
x-im46                   298   83.56 +- 2.15 %   0.342197 +- 0.050799   0.07067   0.8823   3.0418   9.0944    9.2274
x-mse46                  298   80.54 +- 2.30 %   0.375856 +- 0.053727   0.07814   0.9389   3.9523   9.0986    9.1989

== paired, per row: KLD(A) - KLD(B) on the SAME row
A - B                           rows mean difference              median    A worse  sign test p
x-im - none                      298  0.011317 +-  0.009359   0.000000   150/298         0.954
x-im46 - none                    298 -0.000912 +-  0.007758  -0.000043   135/298         0.118
x-mse46 - none                   298  0.032748 +-  0.013496  -0.000000   148/298         0.954
x-im46 - x-mse46                 298 -0.033659 +-  0.014783  -0.000006   139/298         0.271
x-ctl - none                     298  0.000000 +-  0.000000   0.000000     0/0               1
```

**The 607-row set decides and the 298-row set decides nothing.** On 607 rows every paired
difference is POSITIVE — the arm is further from f32 — by 1 to 4 times #78's 0.025 paired floor,
with sign tests of 3.8e-12, 6.5e-22 and 3.3e-5 on 607 correlated rows. On 298 rows no sign test
is below 0.118 and every mean is inside its own standard error; that is the same insensitivity
#78 recorded for that set, and nothing there contradicts the 607-row reading — it simply cannot
see it.

Same top-1 agreement falls with the same ordering on 607 rows: 82.37 % -> 81.05 (d) -> 80.72 (b)
-> 80.07 (c), all of them 1.3 to 2.3 points below `none` against an instrument noise of about
1.5 points.

### 7.5 The inversion, and the one thing that orders the arms

This is the result, and it is worth stating flatly: **the weight-space error and the distance to
the f32 model moved in OPPOSITE directions.**

| arm | plain MSE vs (a) | importance-weighted MSE vs (a) | sub-block scales moved | paired KLD vs `none`, 607 rows |
|---|---|---|---|---|
| (b) `imatrix` | -0.69 % | -8.93 % | 44.08 % | **+0.0247** |
| (d) `mse46` | -14.83 % | -15.08 % | 48.98 % | **+0.0447** |
| (c) `imatrix46` | -11.38 % | **-19.74 %** | 57.56 % | **+0.1038** |

The arm with the BEST weight-space numbers in both columns, (c), is the arm furthest from the
model. Neither MSE column orders the three; the share of sub-block scales that were MOVED away
from the container's own choice orders them exactly, 44 / 49 / 58 % against +0.025 / +0.045 /
+0.104. Three points are three points and that is a correlation, not a law — but it is the only
column that does order them, and it says the penalty tracks how MUCH the encoding was changed
rather than which way its average error went.

Why that can happen at all: `sum_j dw_j^2 E[x_j^2]` is a FIRST-ORDER, diagonal, per-tensor proxy.
It prices every column independently, it knows nothing about the correlations between columns
that a real activation has, it is blind to the router — an error in an expert the calibration
rarely picked but this prompt does pick costs the same as one in a common expert — and it is
computed against Unsloth's calibration corpus rather than the 607 rows the oracle uses. NVIDIA's
Nemotron 3 Ultra report (arXiv 2512.02010) already found that plain weight-MSE scales gave no
consistent downstream gain over max-based ones; measured here against the f32 model rather than
a downstream benchmark, the sign is not merely absent, it is negative.

The clipping share does NOT explain it either, and it was the obvious suspect: (b) clips the most
(4.25 % against the base's 2.16 %) and is the LEAST damaged arm; (c) clips 2.77 % and is the most
damaged.


## 8. The stop rule

**The stop rule is negative, and not in the way it anticipated.**

Neither (b) nor (c) moves the paired mean KLD on the 607-row set in the right direction. Both
move it in the WRONG direction, by clearly more than the 0.025 floor, with sign tests that leave
no room: `imatrix` is +0.0247 +- 0.0107 and worse on 389 of 607 rows (p = 3.8e-12), `imatrix46`
is +0.1038 +- 0.0181 and worse on 421 of 607 (p = 6.5e-22). The second control, `mse46`, which
uses no importance matrix at all, is +0.0447 +- 0.0176 and worse on 355 of 607 (p = 3.3e-5). The
298-row set can decide nothing about any of them. Same top-1 falls 1.3 to 2.3 points on 607 rows
for all three. So: **the routed experts' rounding is not where the remaining distance lives, and
the remaining 40 layers are NOT worth fetching.** The series does not continue down this road.

What the pilot bought instead is a harder fact than "no effect". Re-rounding 22.5 % of the
container's expert values onto the same NVFP4 grid, with a rule that provably lowers both the
plain and the importance-weighted weight-space error, makes this model measurably FURTHER from
its own f32 oracle. If the effect is roughly linear in the number of layers — and nothing here
establishes that — the whole-model version of these rules would sit about +0.15 (b) to +0.62 (c)
from `none` on the 607-row set, which is a worse quant than `UD-Q2_K_XL`. The instrument that
would have been used to tune such a rule, weight-space MSE, points the wrong way on this
architecture, and any future attempt at better expert scales has to be measured against the f32
oracle from the first iteration rather than against an error term.

There is therefore no unattended procedure to hand over for the remaining 40 layers, and no disk
estimate to give, because the answer to "should they be fetched" is no. For the record, had it
been yes: 40 layers is 201,326,592,000 B = 201.33 GB of BF16 originals and about 113 GB per
overlay, which does not fit the 94 GB this machine had free — it would have had to run layer by
layer, fetching, quantizing into a whole-model overlay and deleting each layer's originals, at
the 10.0 MB/s measured here, about 5.6 hours of download.

**What is left.** #78 named the two remaining suspects and this ticket removes one of them.
Untouched: the NVFP4 PLE table, and the attention sub-block's residual (2.9 % `rel_L2` at layer 3
with every weight in BF16, `docs/dense-overlay.md` 4.1) — which is the one that no weight change
has been able to explain, and which points at the engine's own numerics rather than at any
quantization.


## 9. How to re-run all of it

```
# 1. the originals (about 1 to 2 hours for eight layers on this line; resumable, idempotent)
tools/fetch-dense-originals.py --experts 1,7,13,19,25,31,37,43 --dry-run   # the plan, no network
tools/fetch-dense-originals.py --experts 1,7,13,19,25,31,37,43            # one file per layer
python tools/test_fetch_dense_originals.py                                 # 44 unit tests, no network

# 2. the proof, per layer (about a minute each, no GPU, two workers, ~10 GB resident)
cd converter && cargo build --release && cd ..
for L in 01 07 13 19 25 31 37 43; do
  ./converter/target/release/converter requant-check \
      models/Qwen3.8-Flash-Next-original/experts/layer-$L.safetensors \
      converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq --experts
done

# 3. the overlays (one at a time; each is about 11.3 GB on disk)
./converter/target/release/converter expert-overlay \
    --base converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq --out converter/expert-mse-control.cnq \
    --originals models/Qwen3.8-Flash-Next-original/experts --layers 1,7,13,19,25,31,37,43 \
    --rule mse --report decode_out/79/weights-mse.json
#   --rule mse46 / imatrix / imatrix46 are the other three; --imatrix is required for the
#   two weighted ones and is read by all four, because it scores every rule's report

# 4. one arm, both id sets (ONE engine on the card at a time)
systemd-run --user --scope --slice=session.slice --quiet -p MemorySwapMax=0 \
    -p MemoryHigh=56G -p MemoryMax=58G \
    env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
        CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json CROW_GRAPH=1 CROW_MMA=1 \
        LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
        CROW_CNQ_OVERLAY=$PWD/converter/expert-imatrix.cnq \
    engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json \
        decode_out/oracle-t2b/x-im
#   and the same with decode_out/oracle-tf298/tf298-ids.json -> decode_out/oracle-tf298/x-im
#   the four arms of record are x-ctl (rule mse), x-mse46, x-im and x-im46

# 5. the reading (no GPU)
tools/oracle-kld.py --ref decode_out/oracle-t2b/ref-logits.f32 --rows 0:607 --prompt-rows 596 \
    --arm none=decode_out/oracle-t2b/none/gpu-logits.f32 \
    --arm x-ctl=decode_out/oracle-t2b/x-ctl/gpu-logits.f32 \
    --arm x-im=decode_out/oracle-t2b/x-im/gpu-logits.f32 \
    --arm x-im46=decode_out/oracle-t2b/x-im46/gpu-logits.f32 \
    --arm x-mse46=decode_out/oracle-t2b/x-mse46/gpu-logits.f32 \
    --paired x-im,none --paired x-im46,none --paired x-mse46,none --paired x-ctl,none \
    --json decode_out/79/kld-t2b.json
#   the 298-row set is --rows 0:298 --prompt-rows 235 against decode_out/oracle-tf298/

tools/gate-linux.sh          # no overlay: ALL GREEN at the values of record
```

There is deliberately NO gate item for any of this, for the reason `docs/dense-overlay.md` 5.6
gives: every arm needs files that are not in the repository and cannot be — a 104.73 GB
container, 40.27 GB of originals, an 11.32 GB overlay, a 580 MB importance matrix and 900 MB of
reference logits. What the gate carries is the cheap part: the pure unit tests of the GGUF
reader, of the four rules and of the widened refusal table.

