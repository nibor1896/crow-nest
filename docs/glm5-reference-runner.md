# GLM-5.3-Flash: the layerwise HF reference runner

`oracle/glm5_layerwise.py` is step 7 of the GLM-5.3-Flash plan (crow-nest #158, parent
nibor1896/Crow#362). It produces the reference that later steps compare against: the layer
goldens of steps 6 and 13a–13e, the routing data of step 8 and the logits of the whole-model
gate of step 15. It changes nothing in `engine/`, the converter or any container.

GLM-5.3-Flash has 321,323,031,390 parameters (HF API, rev `eb9eb208`). In BF16 that is about
640 GB; this machine has 63.38 GiB of RAM. `transformers` 5.16.1 cannot hold
`Glm5NextForConditionalGeneration` here. The runner holds one decoder layer at a time.

## 1. What it does

For each layer `l` of the text model (layers 0–44; layer 45, the MTP block, is not part of the HF
text model and is not run):

1. build `Glm5NextTextDecoderLayer(config, l)` on the meta device;
2. fill it from the weights with `load_state_dict(strict=True, assign=True)`: a missing, extra or
   misshaped tensor is an error;
3. run the prompt rows in one call, then each decode row alone, against a `DynamicCache` that only
   this layer uses (KDA conv and recurrent state, MLA K/V, indexer keys);
4. write the layer output, the 4-stream residual `[N][4][4096]`, to disk;
5. free the layer and read the state back from disk as the input of layer `l+1`.

After the last layer it computes `lm_head(norm(mean over the 4 streams))` at the anchor rows.
That is the call sequence of `Glm5NextTextModel.forward` (`modeling_glm5_next.py:1477-1493`) and of
the head (`:2181`), with the same HF modules. The only difference is that the residual goes through
a file between layers.

Numerics: f32 everywhere, eager attention (f32 softmax), and the plain per-expert loop of
`Glm5NextTextExperts.forward` (`_experts_implementation = "eager"`). HF's full model would pick
`grouped_mm` when it can dispatch it, and `from_pretrained` would pick `sdpa` attention. Section 4
shows what each choice changes. Threads: `ORACLE_THREADS` (default 16).

## 2. Commands

```
# the reference over the FP8 originals (downloaded to models/GLM-5.3-Flash-original/)
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run \
    --weights fp8-originals models/GLM-5.3-Flash-original \
    --ids ids.json --decode 4 --out runs/glm53-flash/ref-<name> \
    [--layers 0:4] [--anchors 95,96,97,98,99] [--state-dtype f32|bf16]

# the CNQ container back end: a stub until the glm5_next converter lands (exit 2, says why)
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights container <file.cnq> ...

# the proof of the runner (section 4)
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py selftest [--shapes small|real] [--hf-experts eager|grouped_mm]

# the tests (about 30 s)
.venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v
```

- `--ids`: a JSON list of token ids. The last `--decode D` of them are decode rows: teacher-forced,
  one call each, against the cache the prompt left. Rows `0..T-1` are the prompt.
- `--layers A:B` runs layers `A..B-1`. For `A > 0` the input is `l<A-1>-output.*` in `--out`, and the
  existing `manifest.json` is extended. It must have the same ids, `T`, `D`, state dtype and weights
  (index sha256), or the run stops. Step 6 needs layers 0–3 only (`--layers 0:4`). A long run that
  stopped can go on from the last layer on disk.
- `--anchors`: the rows that get logits. Default: the last prompt row and every decode row. Logits
  are written only when the run reaches the last layer.
- `--state-dtype bf16` writes the hand-over state in BF16 (16 KiB per token per layer instead of
  32 KiB). Every later layer then reads a rounded input, so that run is no f32 reference. It is
  meant for long routing runs (step 8). Every golden of record is f32.

## 3. Weights

`glm5_common.WeightSource` reads tensors under their CHECKPOINT names and returns f32.

**FP8 originals** (`--weights fp8-originals <dir>`: `config.json`, `model.safetensors.index.json`,
the shards). A tensor `X.weight` that has a sibling `X.weight_scale_inv` is E4M3. Its value is
`e4m3 * scale_inv[i // 128, j // 128]`, with one f32 scale per 128×128 block (`weight_block_size`
of rev `eb9eb208`, DeepSeek-V3 report arXiv:2412.19437 §3.3.2). Every other tensor (BF16, F32) is
widened exactly. FP8 is decided per tensor from the index, not from a name list. In the index of
rev `eb9eb208` (HF web view, 2026-10-08) these carry a scale: the dense MLP, the shared and routed
experts, and the DSA `q_a_proj`, `q_b_proj`, `kv_a_proj_with_mqa` and `o_proj`. These do not: the
KDA projections, `kv_b_proj`, the indexer, the router, the norms, `hc_*`, `embed_tokens` and
`lm_head`.

A dimension that is not a multiple of 128 ends in a partial block (the DeepSeek-V3 `weight_dequant`
rule). HF's `Fp8Dequantize` instead takes the block size from the grid (`rows // scale_rows`). The
two rules agree whenever a dimension is a multiple of 128 or has a single block. That holds for every
FP8 tensor of glm5_next. `test_partial_blocks_follow_the_128_grid` pins the 128 rule.

Shards are opened per tensor and never cached, so no mmap stays resident. The routed experts of a
layer are dequantized one expert at a time into one preallocated `[E, 2I, H]` / `[E, H, I]` tensor.
For 288 experts that is 19.3 GB + 9.7 GB in f32. This size is computed, not measured.

**Checkpoint names → module names** (`conversion_mapping.py:535-579`, applied in reverse by
`glm5_common.ckpt_recipe`):

| checkpoint (under `model.language_model.layers.<l>.`) | module |
|---|---|
| `self_attn.{f_a_proj,f_b_proj}.weight`, `self_attn.{dt_bias,A_log}` | `self_attn.forget_gate.*` |
| `hc_{attn,ffn}_{fn,base,scale}` | `{attn,ffn}_hc.{fn,base,scale}` |
| `mlp.experts.<e>.{gate,up}_proj.weight` | `mlp.experts.gate_up_proj` `[E, 2I, H]` (stack over e, gate rows then up rows) |
| `mlp.experts.<e>.down_proj.weight` | `mlp.experts.down_proj` `[E, H, I]` |
| `self_attn.{q,k,v}_conv1d.weight` | `self_attn.conv1d.weight` (concatenated on dim 0) |
| everything else | the same name |

`embed_tokens.weight` and `norm.weight` sit under `model.language_model.`, `lm_head.weight` at the
top. The embedding is read row by row for the ids used, and `lm_head` in 16,384-row chunks.

**CNQ container** (`--weights container <file.cnq>`): a stub. The glm5_next converter has not
landed, so the container layout of its FP8 and per-expert records is not defined. The run exits
with code 2 and says so. The back end gets its reader together with the converter.

## 4. The proof (abort criterion of plan step 7)

The criterion: the runner agrees with HF's full model class to 1e-5 (f32, absolute) per layer and
at the logits. This is checked on a SYNTHETIC mini config. It proves the runner, not the model:
the real 321 G weights are not on disk.

The mini model has 8 layers: dense KDA at 0–2, DSA at 3 and 7, KDA+MoE at 4–6. That is the real
schedule cut to 8 layers, covering every block family. It has 16 routed experts with top-8 kept,
4 mHC streams with 20 Sinkhorn iterations, and `index_kpool` 4. `--shapes real` uses the real
per-block shapes: hidden 4096, 64 heads × 256, `kv_lora_rank` 512, `q_lora_rank` 1536, KDA 64 × 128,
conv 4, `moe_intermediate_size` 2048, `intermediate_size` 12288, indexer 32 × 128, vocab 154,880.
`--shapes small` keeps the structure at small widths, and its FP8 tensors still span several 128×128
blocks. The seed is 20261008. The HF init leaves norms at 1, `A_log`, the router bias and the k-pool
`ape` at 0, and the k-pool gate at 1. Those tensors get random values, so a swapped or dropped
tensor shows up. The embeddings get std 1 instead of 0.02, so the absolute bound is strict against
a residual RMS of about 1.

How the proof is built:

1. The mini model is written as a checkpoint in the ORIGINAL naming and format: per-expert tensors,
   `hc_*`, `q/k/v_conv1d`, E4M3 + `weight_scale_inv` for the FP8 set above. KDA `b_proj` is also FP8
   here, as a synthetic-only one-block case. The rest is BF16, and `A_log`, `dt_bias`,
   `e_score_correction_bias` and the conv weights are F32.
2. **HF side.** HF's `Fp8Dequantize` dequantizes every FP8 tensor. `from_pretrained` loads the
   result (original naming) through HF's own renames and merges, with eager attention. Missing keys
   may only be `model.visual.*` (the vision tower is not written); unexpected keys must be 0. The
   prompt and then the teacher-forced decode rows run against one `DynamicCache`. Hooks record every
   decoder-layer output, the router top-8 and the indexer selection. (HF's FP8 quantizer path itself
   needs `accelerate`, which the venv does not have.)
3. **Runner side.** `glm5_layerwise.run_layerwise` over the FP8 checkpoint, with its own name table
   and its own FP8 unpack.
4. The comparison is per layer: max |Δ| on the prompt rows and on the decode rows, top-8 ids per
   token (sorted), their weights, the DSA top-k per row, and the logits at all `T + D` rows. It runs
   at `index_topk` 32 (8 pools of 4 at T = 96, so the indexer really drops pools: 28.1 selected
   tokens per row on average) and at 2048 (everything visible: 50.5).

T = 96 prompt rows, D = 4 decode rows. So the KDA prompt runs two 64-row chunks, and decode takes
the recurrent path.

**Result, 2026-10-08, CPU, torch 2.13.0+cpu, transformers 5.16.1, 16 threads:**

| run | layers 0–7: max \|Δ\| (prompt / decode) | routing ids / weights | DSA top-k | logits max \|Δ\| | time |
|---|---|---|---|---|---|
| small, HF experts eager, topk 32 and 2048 | 0 / 0 (bit-identical) | identical / 0 | identical | 7.7e-7 | 6 s |
| small, HF experts `grouped_mm`, topk 32 | ≤ 9.5e-7 / ≤ 7.2e-7 | identical / ≤ 8.9e-8 | identical | 6.6e-7 | 6 s |
| small, HF experts `grouped_mm`, topk 2048 | ≤ 1.2e-6 / ≤ 7.2e-7 | identical / ≤ 8.9e-8 | identical | 7.7e-7 | 6 s |
| **real shapes**, HF experts eager, topk 32 and 2048 | 0 / 0 (bit-identical) | identical / 0 | identical | 3.8e-6 / 4.3e-6 | 104 s, peak RSS 27.5 GiB |

With the same kernels on both sides, every layer is bit-identical. The logits differ (< 1e-6 at
hidden 256, up to 4.3e-6 at hidden 4096 with logits of the scale the residual RMS of ~5 gives) only
because the runner computes `lm_head` in 16,384-row chunks. HF's `grouped_mm` experts kernel moves the
MoE layers by about 1e-6. That is a property of the kernel, not of the runner.

The proof also fails when it should (`test_glm5_layerwise.py`):
- A hand-over that collapses the 4 mHC streams into their mean fails from layer 1 on (max |Δ| > 1e-3).
  Layer 0 still passes, because it reads the embeddings.
- Decode rows run without the layer's cache leave the prompt rows exact. The decode rows of layer 0
  then fail (max |Δ| > 1e-3).

## 5. Output files

All files are raw little-endian, row-major, with no header. `manifest.json` holds each file's shape,
dtype and sha256. It also holds the versions, threads, attention and experts implementation, the
weight provenance (sha256 of index and config, tensor and FP8 counts), the ids, `T`, `D`, anchors,
the state dtype, per-layer load and compute seconds and RSS, and `complete`.
`N = T + D` rows; `hc` = 4; `H` = 4096.

| file | shape | content |
|---|---|---|
| `embed.f32` | `[N][H]` | token embeddings, the input of layer 0 (expanded to 4 streams) |
| `l<k>-output.f32` | `[N][hc][H]` | output of decoder layer k = the state handed to layer k+1 (`.bf16` with `--state-dtype bf16`) |
| `l<k>-routing-ids.i32` | `[N][8]` | MoE layers only: the 8 routed expert ids per token, ascending |
| `l<k>-routing-weights.f32` | `[N][8]` | their weights in the same order: sigmoid scores normalized over the 8, × `routed_scaling_factor` (what the experts are scaled with) |
| `l<k>-dsa-topk.i32` | `[N][W]` | DSA layers only: the indexer's selected token positions per row (whole pools in score order, then the incomplete tail pool), `-1` = empty, `W = index_topk + index_kpool - 1` (2051 on rev `eb9eb208`) |
| `logits-anchor-<p>.f32` | `[V]` | logits of row p (V = 154,880) |

**Routing dump (step 8).** The routing is binary, not JSON, because of its size. At 45 layers × 8
ids per token, JSON would be tens of bytes per id. The binary form is 64 bytes per token and MoE
layer (32 for ids, 32 for weights). Read it like this:

```python
import json, numpy as np
man = json.load(open(f"{out}/manifest.json"))
ids = np.fromfile(f"{out}/l5-routing-ids.i32", "<i4").reshape(man["files"]["l5-routing-ids.i32"]["shape"])
w   = np.fromfile(f"{out}/l5-routing-weights.f32", "<f4").reshape(ids.shape)
```

The router's choice uses `sigmoid(logits) + e_score_correction_bias` (`n_group` = `topk_group` = 1,
so there is no group mask). The weights come from the sigmoid scores without the bias. The order
inside HF's top-8 (`sorted=False`) is not defined, so the dump sorts the ids and moves the weights
with them.

## 6. Limits

- Nothing has run on the real weights yet: they are not on disk. The run time and peak RSS of one
  real layer on this CPU are **not measured**. The f32 size of one MoE layer's experts (29 GB) is
  computed.
- `indexer_types` `"shared"` (cross-layer top-k reuse) is not supported: the runner raises. The
  config of rev `eb9eb208` sets `"full"` on all 45 layers. Supporting it would mean feeding the
  previous layer's `l<k>-dsa-topk.i32` back into `prev_topk_indices`.
- Not run: the MTP block (layer 45, plan step 21) and the vision tower (step 20). No GPU path: no
  CUDA build of torch is installed.
- Batch 1, no padding: the attention mask is all ones, as the HF text model builds it when none is
  given.
