# Step 6 — partial container layers 0–3 and layer goldens (crow-nest #156), 2026-10-08

Plan step 6 of the GLM-5.3-Flash series, PREREG `runs/glm53-flash/PREREG.md` (blob sha256 375de4d8…).
Machine: Windows 11, 63.38 GiB RAM, CPU only (torch 2.13.0+cpu, transformers 5.16.1, 16 threads), crow-nest
`e1d6b7e` + branch `glm-step6`. Large outputs live under `models/GLM-5.3-Flash-step06/` (git-ignored); this
directory keeps the small records.

## Inputs

- Shards 01, 02, 03, 17, 31, 32, 62 of `zai-org/GLM-5.3-Flash` rev `eb9eb208`, 33,403,348,264 B, each
  `VERIFIED` (size + lfs.sha256) by `fetch.log` (run 01:33:54–02:17:28 CEST, `run end: rc 0`).
- Ids: `ids.json` (sha256 `56e2037e…29bb7`), 90 tokens = 86 prompt rows + 4 decode rows, from the text in
  `ids-source.json` with the checkpoint's `tokenizer.json` (no special tokens). Not a PREREG anchor: G3's
  anchors are not fixed yet (PREREG G3, "dated amendment before the first parity row").

## Partial container (`convert.log.txt`)

```
converter --scales mse --source-repo zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a \
  --headers models/GLM-5.3-Flash-original/headers --layers 0-3 --with-embed-head \
  models/GLM-5.3-Flash-original models/GLM-5.3-Flash-step06/GLM-5.3-Flash-CNQ4.5-L0-3.cnq
```

| item | value |
|---|---|
| output | `GLM-5.3-Flash-CNQ4.5-L0-3.cnq`, 7,220,714,871 B, sha256 `1d22f669…901df2`; sidecar sha256 `d8f8f37d…e9cf35` |
| tensors | 972 written (902 NVFP4, 70 BF16/F32) of 37,534; 72,149 weight_map names filtered, 2,107 omitted by the recipe (vision, MTP), 0 missing |
| payload | 7.22 GB, 5,140 B alignment zeros, 244 s (rc 0) |
| recipe | `cnq4.5-glm5-next` as committed, `--scales mse` |

## Gate 0 — FP8 → NVFP4 error per tensor (`weight-check.json`, sidecar)

`oracle/glm5_weight_check.py`, 92 s. Every container tensor against the FP8 originals:

- BF16/F32 keeps: 70 / 70 equal to the originals, exactly.
- FP8 sources: 880 / 880 tensors with `glm5_common.fp8_dequant` == transformers `Fp8Dequantize._dequantize_one`, bit for bit.
- Converter f32 vs oracle f32 (indirect): `mean((W_cnq − W_fp8)²)` recomputed from the oracle's FP8 decode equals the
  sidecar's `mse` (computed by the converter against its own FP8 decode) on all 902 NVFP4 tensors, worst relative
  difference 3.2e-13. A different scale order or block size on either side would show here; the converter's f32 itself
  is not dumped, so this is not a bit-for-bit proof.

| class | tensors | rel RMS error median [min, max] | MSE / MSE(ceil) | clipped (share) | old-bound violations |
|---|---|---|---|---|---|
| attn_kda (q/k/v/o_proj, q/k/v_conv1d, layers 0–2) | 21 | 0.0884 [0.0862, 0.0913] | 0.8005 | 7,626,948 (1.893 %) | 21,492 |
| attn_mla (layer 3) | 5 | 0.0860 [0.0853, 0.0886] | 0.7870 | 2,213,766 (1.885 %) | 10,016 |
| dense_mlp (layers 0–2) | 9 | 0.0850 [0.0840, 0.0883] | 0.7737 | 8,326,365 (1.838 %) | 43,159 |
| expert_down (layer 3) | 288 | 0.0841 [0.0835, 0.0859] | 0.7788 | 41,023,590 (1.698 %) | 61,814 |
| expert_gate | 288 | 0.0842 [0.0836, 0.0873] | 0.7819 | 41,801,340 (1.730 %) | 147,355 |
| expert_up | 288 | 0.0844 [0.0838, 0.0860] | 0.7794 | 42,496,632 (1.759 %) | 165,917 |
| shared_expert | 3 | 0.0875 [0.0866, 0.0876] | 0.7864 | 485,726 (1.930 %) | 1,569 |
| **all (section text)** | **902** | | **0.7810** | **143,974,367** | **451,322** |

The old relative bound (`max_rel ≤ 1.08` per element) is void under `--scales mse` by design (clipping allowed,
`converter/README.md` "Scale policy"); 901 of 902 NVFP4 tensors carry violations of it, every one named in
`weight-check.json` → `gate0_violations`. Under `mse` gate 0 is the MSE report: MSE 2.80e-6 vs 3.58e-6 with ceiling
scales on the same weights (ratio 0.781). No tensor was refused, no NaN.

## Layer goldens, both back ends (`golden-*.manifest.json`, `compare-fp8-vs-cnq.json`)

```
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights fp8-originals models/GLM-5.3-Flash-original \
  --ids runs/glm53-flash/step06/ids.json --decode 4 --out models/GLM-5.3-Flash-step06/ref-fp8 --layers 0:4
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights container models/GLM-5.3-Flash-step06/GLM-5.3-Flash-CNQ4.5-L0-3.cnq \
  --ids runs/glm53-flash/step06/ids.json --decode 4 --out models/GLM-5.3-Flash-step06/ref-cnq --layers 0:4
.venv-oracle/Scripts/python.exe -I oracle/glm5_compare.py models/GLM-5.3-Flash-step06/ref-fp8 models/GLM-5.3-Flash-step06/ref-cnq
```

| back end | wall | layer 3 load / compute | peak RSS (layer 3) |
|---|---|---|---|
| A: FP8 originals | 19.6 s | 7.8 s / 0.69 s | 28.13 GiB after load, peak working set 28.16 GiB |
| B: container via `converter dequant` | 68.2 s | 52.1 s / 0.60 s | 28.00 GiB after load, peak working set 28.06 GiB |

Quantisation error, container vs FP8 originals (each back end's own chain; layer k of B reads B's layer k−1;
all 4 streams × 90 rows × 4096 flattened). Reported, not gated (PREREG G3 robustness: "reported separately").

| layer | kind | cosine | max \|Δ\| (prompt / decode rows) | rel RMS | output RMS (A) |
|---|---|---|---|---|---|
| 0 | KDA + dense | 0.99307936 | 6.99e-3 (6.99e-3 / 5.25e-3) | 0.1177 | 5.86e-3 |
| 1 | KDA + dense | 0.99402281 | 6.04e-3 (6.04e-3 / 4.55e-3) | 0.1104 | 5.40e-3 |
| 2 | KDA + dense | 0.99805944 | 2.33e-2 (2.33e-2 / 1.62e-2) | 0.0644 | 3.08e-2 |
| 3 | DSA + MoE | 0.99788674 | 1.98e-1 (1.98e-1 / 4.42e-2) | 0.0714 | 6.83e-2 |

Layer 3: routed top-8 sets per token overlap 0.911 on average (min 0.625), 40 of 90 rows identical; DSA indexer
selections identical on 90 / 90 rows (T = 86 < 2052: every pool is selected, the layer is dense causal attention).
Golden sha256 per file: the two manifests (embed.f32 identical on both sides: the embedding is a BF16 keep).

## G3 for step 6: not answered

PREREG G3 per layer compares the ENGINE's layer output with the golden on the container-dequantised weights
(cosine ≥ 0.9999). The engine has no glm5_next layer math yet (kernels are plan step 13, after G1), so that half
cannot run; the goldens of back B above are its reference when it can. The cosines in the table are the
quantisation error (container vs FP8), not the G3 metric.

## Regenerated 2026-10-09 (FP8 back end, with sub-block captures)

The goldens above were deleted on 2026-10-08 in a disk cleanup, together with the partial container. Back end A was
run again from the re-downloaded FP8 originals (62/62 shards verified by `tools/fetch-glm.py`, weights identity
`f47bd154…bfc90`, index and config sha256 as in the first run), at commit `24fd820`, with `--capture-subblocks`:

```
ORACLE_THREADS=16 .venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights fp8-originals models/GLM-5.3-Flash-original \
  --ids runs/glm53-flash/step06/ids.json --decode 4 --out models/GLM-5.3-Flash-step06/ref-fp8 --layers 0:4 --capture-subblocks
```

rc 0, 19.8 s wall, layer 3 load / compute 8.81 / 0.58 s, RSS after the layer-3 load 28.13 GiB (peak working set
28.18 GiB). 48 files: the 8 of the first run, each byte-identical to it (sha256 of `golden-fp8.manifest.json`), plus
40 sub-block files (10 per layer). Record with every file's sha256, the weights identity and the command:
`golden-fp8-regen.json` (a copy lies in the golden dir as `evidence.json`). Back end B (container) was not regenerated.

## 3-bit goldens on the MUL1 container (2026-10-09)

G3 compares the engine on the 3-bit container against goldens computed on the same dequantized weights, so back end
B was run on `converter/GLM-5.3-Flash-MUL1K3.cnq` (124,591,634,567 B, sha256 `9ce11456…0213e`, index sha256
`56741348…61612`; routed experts MUL1 K=3, 12,384 records, store sha256 `95c76f56…f9fef`; dense part NVFP4
`cnq4.5-glm5-next --scales mse`). `converter dequant` decodes the MUL1 records since `eb1d92d` (#181 decoder, then
`diag(suh) H W_hat H diag(svh) / 128` in f64, one f32 rounding). Same ids and flags as the FP8 run of record:

```
ORACLE_THREADS=16 .venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights container converter/GLM-5.3-Flash-MUL1K3.cnq \
  --ids runs/glm53-flash/step06/ids.json --decode 4 --out models/GLM-5.3-Flash-step06/ref-mul1 --layers 0:4 --capture-subblocks
```

rc 0, 68 s wall, layer 3 load / compute 50.56 / 0.58 s (layers 0-2 about 2.0 s load each), RSS after the layer-3 load
28.84 GiB (peak working set 28.88 GiB; the `converter dequant` child processes not included). 48 files, manifest sha256
`31b7966f…38b77`; `post*y + comb^T*x` recomputed in f64 equals the expanded output to 1.1e-7, `attn_hc-in` of l0 is the
embedding broadcast and of l1..3 the previous output, bit-exact. Identical to the FP8 goldens: `embed.f32` and the
four l0 attn hyper-connection files before the first quantized matmul. Record with every file's sha256:
`golden-mul1.json` (a copy lies in the golden dir as `evidence.json`).

Quantisation error, 3-bit container vs FP8 originals (`oracle/glm5_compare.py ref-fp8 ref-mul1`, each run's own chain,
all 4 streams x 90 rows x 4096). Reported, not gated:

| layer | kind | cosine | max \|Δ\| (prompt / decode rows) | rel RMS |
|---|---|---|---|---|
| 0 | KDA + dense | 0.99307566 | 6.98e-3 (6.98e-3 / 5.25e-3) | 0.1178 |
| 1 | KDA + dense | 0.99402332 | 6.05e-3 (6.05e-3 / 4.55e-3) | 0.1104 |
| 2 | KDA + dense | 0.99805874 | 2.33e-2 (2.33e-2 / 1.62e-2) | 0.0644 |
| 3 | DSA + MoE | 0.99776613 | 1.78e-1 (1.78e-1 / 4.44e-2) | 0.0759 |

Layer 3: routed top-8 overlap 0.910 on average (min 0.625), 39 of 90 rows the same set; DSA selections identical on
90 / 90 rows. G3 itself (`decode glmgolden models/GLM-5.3-Flash-step06/ref-mul1`) needs the GPU and was not run.

## All 45 layers and the head on the MUL1 container (2026-10-09)

```
ORACLE_THREADS=16 .venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights container converter/GLM-5.3-Flash-MUL1K3.cnq   --ids runs/glm53-flash/step06/ids.json --decode 4 --out models/GLM-5.3-Flash-step06/ref-mul1-all --capture-subblocks --capture-head
```

rc 0, 2290 s wall, RSS after load ≤ 28.12 GiB (peak working set 28.93 GiB), 599 files (1.13 GB), manifest sha256
`420494b1…a82a0`; layers 0–3 byte-identical to `ref-mul1`. The head files (`head-mean`, `head-norm`, `head-logits`,
#165) hold every row. Record: `golden-mul1-all.json`. G3 on it: `docs/glm5-model.md` section 4 (golden-fed ALL PASS,
0 of 549 rows failed).

## MTP golden on the 3-bit container (2026-10-09, #182)

The MTP block (checkpoint layer 45) on the same 90 ids: `oracle/glm5_mtp.py` (formula of record and sources in
`docs/glm5-mtp.md`) over the trunk dir `models/GLM-5.3-Flash-step06/ref-mul1-all` (the full 45-layer pass on the
3-bit container with `--capture-head`, #165: `head-norm.f32` is the trunk's post-final-norm row, `head-logits.f32` its
logits). The block's 288 experts come from the container (section `mtp`, MUL1 K=3, identity Hessian); its other 25
tensors are not in the container and come from the FP8 originals.

```
ORACLE_THREADS=16 .venv-oracle/Scripts/python.exe -I oracle/glm5_mtp.py run --weights container converter/GLM-5.3-Flash-MUL1K3.cnq \
  --fallback fp8-originals models/GLM-5.3-Flash-original --trunk models/GLM-5.3-Flash-step06/ref-mul1-all \
  --out models/GLM-5.3-Flash-step06/ref-mul1-mtp
```

rc 0, 89 s wall (load 62.1 s, lm_head 6.1 s, five variants 4.7 s), RSS after the load 28.28 GiB, peak working set
33.0 GiB. 89 draft rows (row i = embed(ids[i+1]) with trunk row i, drafts ids[i+2]), 14 files, manifest sha256
`eaffa03a…ffc13d`; record with every file's sha256: `golden-mtp.json` (copy in the golden dir as `evidence.json`).

Draft top-1 = the trunk's own top-1 for the same id (first acceptance indication, reported, not gated; 89 rows, one
draft step):

| pairing | all | prompt rows | decode rows |
|---|---|---|---|
| SGLang (primary: row 0 = (t_1, h_0)) | 55 / 89 = 0.618 | 55 / 85 | 0 / 4 |
| vLLM (row 0 embedding zeroed) | 57 / 89 = 0.640 | 57 / 85 | 0 / 4 |
| llama.cpp (leading (t_0, 0) row) | 56 / 89 = 0.629 | 56 / 85 | 0 / 4 |
| DeepSeek-V3 paper order [h; e] | 0 / 89 | 0 / 85 | 0 / 4 |
| trunk state before the final norm | 52 / 89 = 0.584 | 52 / 85 | 0 / 4 |

The paper's written order gives no agreement at all; the embedding-first order of the code bases is the trained one.
For scale: the draft matches the text's next id on 33 of 88 rows, the trunk's own pick also on 33 of 88. Elsewhere
(other hardware, quant and text, not comparable): adrienbrault 1.756 tokens per step at depth 1, Xxianna 89 %
acceptance (vault note `glm-5-3-flash-recherche-0xsero-bauart-landschaft-und-3-bit-format.md`).
