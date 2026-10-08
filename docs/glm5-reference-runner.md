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
3. run the prompt rows in calls of `--prompt-chunk` rows (default 512; 0 = one call), then each
   decode row alone, against a `DynamicCache` that only this layer uses (KDA conv and recurrent state,
   MLA K/V, indexer keys). Its DSA slot appends each call's K/V and indexer keys in place into the
   N rows it allocates on the first call (`_AppendIndexedLayer`, crow-nest #147); HF's
   `DynamicIndexedLayer` would `torch.cat` the whole history on every call (`cache_utils.py:144-145`,
   `:350`). Attention is causal, so a row sees rows `0..itself` in every split;
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
    [--layers 0:4] [--anchors 95,96,97,98,99] [--state-dtype f32|bf16] \
    [--prompt-chunk 512] [--delete-states-behind]

# the CNQ container back end (crow-nest #156): weights as `converter dequant` decodes them;
# a partial container (converter --layers 0-3 --with-embed-head) runs only the layers it holds
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py run --weights container <file.cnq> \
    --ids ids.json --decode 4 --out <dir> --layers 0:4

# per-layer agreement of two runs over the same ids (cosine, max |d|, routing and DSA overlap)
.venv-oracle/Scripts/python.exe -I oracle/glm5_compare.py <run A> <run B> [--json out.json]

# every container tensor against the FP8 originals + the gate-0 summary of its sidecar
.venv-oracle/Scripts/python.exe -I oracle/glm5_weight_check.py <file.cnq> models/GLM-5.3-Flash-original --json out.json

# the proof of the runner (section 4); --prompt-chunk C also proves the chunked prompt
.venv-oracle/Scripts/python.exe -I oracle/glm5_layerwise.py selftest [--shapes small|real] [--hf-experts eager|grouped_mm] \
    [--prompt-chunk 40] [--T 96] [--D 4]

# step 8: the five routing passes of PREREG amendment 1 (section 8)
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py [--dry-run]

# the tests (about 1 min; the container suite needs the converter binary, or CROW_CONVERTER)
.venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v
.venv-oracle/Scripts/python.exe -I tools/test_glm_route_passes.py
```

- **Why `.venv-oracle`.** The runner imports `transformers.models.glm5_next` and subclasses
  `transformers.cache_utils.DynamicIndexedLayer` for its in-place DSA slot (`glm5_layerwise.py:58-59`,
  `:101`). Both are in the venv's transformers 5.16.1, the oracle of record. The system Python's
  transformers (5.5.4 on the owner's machine, 2026-10-08) has neither. Under the system Python the
  runner and everything that imports it fail at import: `tools/test_glm_route_passes.py` runs 8 tests
  with 1 error (`ImportError: cannot import name 'DynamicIndexedLayer'`), against 10 OK in the venv.

- `--ids`: a JSON list of token ids. The last `--decode D` of them are decode rows: teacher-forced,
  one call each, against the cache the prompt left. Rows `0..T-1` are the prompt.
- `--layers A:B` runs layers `A..B-1`. For `A > 0` the input is `l<A-1>-output.*` in `--out`, and the
  existing `manifest.json` is extended. It must have the same ids, `T`, `D`, state dtype and weights
  (index sha256), or the run stops. Step 6 needs layers 0–3 only (`--layers 0:4`). A long run that
  stopped can go on from the last layer on disk. `--layers 45:` (A = the number of layers) runs no
  layer and computes only the logits from `l44-output.*`: a pass that stopped after its last layer.
- `--prompt-chunk C` (default 512): the prompt rows go through each layer in calls of C rows against
  the layer's cache; `0` runs them in one call. A 32,768-row prompt in one call cannot run here: eager
  DSA attention would hold about 833 GB of scores (PREREG amendment 1), and KDA's chunk form about
  64 GiB of decay masks (64 heads × 512 blocks × 64 × 64 × 128 × 4 B, computed). In 512-row calls the
  largest attention call holds about 12 GiB. The manifest records C and the seconds of every prompt
  call per layer (`prompt_call_s`).
- `--delete-states-behind`: once layer k's state is written and recorded in the manifest, layer
  k−1's state is deleted, and the last one after the logits (manifest `deleted_states`). A pass keeps
  at most two states on disk and can still resume from the last one. Routing, DSA top-k, `embed.f32`,
  logits and the manifest stay.
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
For 288 experts that is 19.3 GB + 9.7 GB in f32 (computed); a real layer 3 peaked at 28.1 GiB RSS
after load (section 7).

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

**CNQ container** (`--weights container <file.cnq>`, crow-nest #156): `glm5_common.ContainerSource`.
The container keeps the checkpoint names, so the name table above serves both back ends. The index
trailer is read for names, shapes and `config.json` (carried verbatim in `model.config_json`); the
values come from `converter dequant <file.cnq> --names -` (`converter/src/dequant.rs`), one converter
process per module, streamed into the same preallocated tensors as the FP8 path (`load_plan` fixes
the order for both). NVFP4 is decoded by `nvfp4_scale` / `nvfp4_value`, the arithmetic gate 0 (the
sidecar) measures the written encoding with; BF16 keeps are widened exactly, F32 carries are read as
stored. So the reference sees exactly what the converter wrote, decoded by the converter's own code.
The binary is `converter/target/release/converter[.exe]` (or `CROW_CONVERTER`); without it the run
exits 2. A partial container (`partial` block in the index) refuses `--layers` outside the layers
it holds, before anything runs (exit 2).

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

**The chunked prompt (crow-nest #147).** `selftest --prompt-chunk C` runs the runner a second time with
the prompt in calls of C rows. It compares that run with HF's full model (one prompt call) and with
the runner's own one-call run. The rule:
- every routing id and every DSA selection (as a set per row) is identical;
- every value agrees to `CHUNK_TOL` = 1e-4 absolute.

Another call split is another f32 summation order: the matmul row counts change, and KDA's 64-row
blocks are cut in other places. It is not another computation. The 1e-5 bound above holds for the
same call sequence on both sides, where the layers are bit-identical. The row LAYOUT of
`l<k>-dsa-topk.i32` depends on the split: `select_k` = min(`index_topk` / kpool, pools in the cache)
sets where the tail starts. The set does not depend on the split, and the attention mask is built
from the set.

| run (2026-10-08, CPU, 16 threads, the converter running beside it) | layers 0–7 max \|Δ\| vs one call (prompt / decode) | routing ids | DSA sets | logits max \|Δ\| | time, peak working set |
|---|---|---|---|---|---|
| real shapes, T 96 in calls of 40, D 4, topk 32 and 2048 | ≤ 4.3e-5 / ≤ 5.2e-5 (residual RMS 2.9–5.3) | identical | identical, 0 boundary ties | ≤ 1.5e-5 | 118.7 s, 25.16 GiB |
| real shapes, T 600 in calls of 512 (512 + 88), D 4, topk 32 and 2048 | ≤ 5.6e-5 / ≤ 4.7e-5 | identical | identical, 0 boundary ties | ≤ 1.3e-5 | 154.9 s, 29.33 GiB |
| small shapes, T 96 in calls of 40 | ≤ 1.4e-6 / ≤ 9.5e-7 | identical | identical | ≤ 7.2e-7 | 11 s |

Both real-shape rows read `selftest real: PASS`. The chunked run matches HF's full model just as
closely as it matches the one-call run.

**Exact ties at the DSA selection boundary.** On the small shapes, calls of 1, 7 and 64 rows select a
different token set on one row of layer 7 (row 123 of 148). Calls of 40 and 100 rows select identical
sets everywhere. On row 123, the one-call run's index scores at ranks k and k+1 are both exactly 0.0.
The indexer applies a ReLU per head, and with the 4 indexer heads of the small shapes a pool's score
is often exactly zero. `torch.topk` breaks exact ties in an order that depends on the tensor's shape,
so HF itself picks differently under another split. The routing ids stayed identical. The runner
records such rows per DSA layer (`dsa_tie_rows_n` and the first 1,000 rows in `dsa_tie_rows`).
`test_every_chunk_size_routes_as_one_call` holds the rule: every layer before the first DSA layer
that differs is within tolerance with identical ids, and every row that differs there is a recorded
tie. The real shapes (32 indexer heads) had 0 tie rows in both runs.

The proof also fails when it should (`test_glm5_layerwise.py`):
- A hand-over that collapses the 4 mHC streams into their mean fails from layer 1 on (max |Δ| > 1e-3).
  Layer 0 still passes, because it reads the embeddings.
- Decode rows run without the layer's cache leave the prompt rows exact. The decode rows of layer 0
  then fail (max |Δ| > 1e-3).
- A DSA cache slot that drops its history on every call (`_AppendIndexedLayer.update` restarting at
  row 0) fails the chunked proof at the first DSA layer (max |Δ| > 1e-3). The KDA layers 0–2 still pass.
- With HF's own `DynamicIndexedLayer` in place of `_AppendIndexedLayer`, the K storage grows on every
  call: 12 sizes over 12 calls instead of one buffer of N rows
  (`test_the_dsa_cache_is_allocated_once`).

## 5. Output files

All files are raw little-endian, row-major, with no header. `manifest.json` holds each file's shape,
dtype and sha256. It also holds the versions, threads, attention and experts implementation, the
weight provenance (sha256 of index and config, tensor and FP8 counts), the ids, `T`, `D`, anchors,
the state dtype, per-layer load and compute seconds and RSS, and `complete`. Since #147 it also
holds `prompt_chunk`, `delete_states_behind` and `deleted_states`, and per layer `prompt_call_s` and
(DSA layers) `dsa_tie_rows_n` / `dsa_tie_rows`.
`N = T + D` rows; `hc` = 4; `H` = 4096.

| file | shape | content |
|---|---|---|
| `embed.f32` | `[N][H]` | token embeddings, the input of layer 0 (expanded to 4 streams) |
| `l<k>-output.f32` | `[N][hc][H]` | output of decoder layer k = the state handed to layer k+1 (`.bf16` with `--state-dtype bf16`) |
| `l<k>-routing-ids.i32` | `[N][8]` | MoE layers only: the 8 routed expert ids per token, ascending |
| `l<k>-routing-weights.f32` | `[N][8]` | their weights in the same order: sigmoid scores normalized over the 8, × `routed_scaling_factor` (what the experts are scaled with) |
| `l<k>-dsa-topk.i32` | `[N][W]` | DSA layers only: the indexer's selected token positions per row (whole pools in score order, then the incomplete tail pool), `-1` = empty, `W = index_topk + index_kpool - 1` (2051 on rev `eb9eb208`). The layout depends on the prompt split, the set does not (section 4) |
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

- On the real weights only layers 0–3 have run (step 6, section 7). Layers 4–44, the logits and the
  run time of a whole pass are **not measured**.
- The chunked prompt is proven on the synthetic mini config (16 experts, 8 layers, T ≤ 604). On a
  real 32,768-token file, rows whose DSA selection boundary is an exact tie can select another set than
  a one-call or row-wise run would. The pass counts them (`dsa_tie_rows_n`).
- `indexer_types` `"shared"` (cross-layer top-k reuse) is not supported: the runner raises. The
  config of rev `eb9eb208` sets `"full"` on all 45 layers. Supporting it would mean feeding the
  previous layer's `l<k>-dsa-topk.i32` back into `prev_topk_indices`.
- Not run: the MTP block (layer 45, plan step 21) and the vision tower (step 20). No GPU path: no
  CUDA build of torch is installed.
- Batch 1, no padding: the attention mask is all ones, as the HF text model builds it when none is
  given.

## 7. Step 6 on the real weights (crow-nest #156, 2026-10-08)

Layers 0–3 over 90 fixed ids (86 prompt + 4 decode rows, `runs/glm53-flash/step06/ids.json`), both back ends,
CPU, 16 threads. Full record: `runs/glm53-flash/step06/README.md`.

| back end | wall | layer 3 load / compute | RSS after layer-3 load |
|---|---|---|---|
| FP8 originals | 19.6 s | 7.8 s / 0.69 s | 28.13 GiB |
| partial container (`converter dequant`) | 68.2 s | 52.1 s / 0.60 s | 28.00 GiB |

Container vs FP8 (the quantisation error, reported, not gated): cosine 0.99308 / 0.99402 / 0.99806 / 0.99789 for
layers 0 / 1 / 2 / 3, max |Δ| 7.0e-3 / 6.0e-3 / 2.3e-2 / 0.198; layer 3 routing overlap 0.911 (40 / 90 rows with the
same top-8), DSA selection identical on 90 / 90 rows. The engine side of G3 needs the glm5_next kernels (plan step 13).

## 8. Step 8: the routing passes (crow-nest #147)

`tools/glm_route_passes.py` runs the five corpus files of PREREG amendment 1 one after another
through this runner on the full container. Each pass covers 32,768 tokens; the files hold exactly
the routed prefix. It writes the routing dumps that `tools/glm_tier_sim.py sim` reads.

```
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py --dry-run   # checks + the five commands, runs nothing
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py             # the passes
```

Defaults: `--container converter/GLM-5.3-Flash-CNQ4.5.cnq`, `--corpus decode_out/glm-step8/corpus`,
`--runs decode_out/glm-step8/runs`, `--prompt-chunk 512`, and `ORACLE_THREADS` from the environment
(16 if unset). `--only <name>,...` limits the run to some files.

**Before anything runs** (exit 2 with the reason):
- **Container complete.** No `<cnq>.journal.jsonl` lies beside it; the converter deletes its journal
  only after writing the index trailer (`converter/src/main.rs:2042-2046`). The file has the CNQ1
  magic and an index trailer that parses: index v2, recipe `cnq4.5-glm5-next`, not partial, source
  rev `eb9eb208`. Every layer 0..L−1, `embed_tokens`, `norm` and `lm_head` are in the index.
- **Corpus matches amendment 1.** `corpus.json` and each file's ids and mask have the amendment's
  sha256 (the table is pinned in the tool; the tests check it against `PREREG.md`). Each ids file
  holds 32,768 ids.
- **Run dirs.** No run dir may hold another container's or another file's run.

**Per file**, in the amendment's order (held-out first), the amendment's command plus the two new
options:

```
oracle/glm5_layerwise.py run --weights container <cnq> --ids <corpus>/<name>-ids.json --anchors 32767 \
    --state-dtype bf16 --prompt-chunk 512 --delete-states-behind --out <runs>/<name>
```

- **Resumable per file.** A pass complete over every layer is skipped. An interrupted pass continues
  with `--layers k+1:` from the last recorded state `l<k>-output.bf16`. A pass with no state to
  continue from starts again at layer 0.
- **Checked after each pass** with the sim's own self-test (routing sha256 against the manifest,
  `[N][8]`, ids 0..287 ascending) and its source check (CNQ, not partial, complete).
- **Logged.** The runner's output goes to `<runs>/<name>/runner.log`. One JSON line per pass, failed
  passes included, goes to `<runs>/passes.jsonl`: start, wall seconds, rc, the layer it resumed from,
  load and compute seconds, peak working set, commit, threads. That line is the measurement-book row.

**Disk per pass:** at most two BF16 states (2 GiB) plus `embed.f32` (0.5 GiB) while it runs. Kept
afterwards: the routing, about 88 MB (42 layers × 32,768 × 8 × 4 B, ids and weights); the DSA top-k,
2.95 GB (11 layers × 32,768 × 2051 × 4 B); the embedding, logits and manifest. Without
`--delete-states-behind`, the states alone would be 45 GiB per pass.

**Runtime per pass (derived, not measured; the first pass measures it):**

| part | seconds | basis |
|---|---|---|
| load, 42 MoE + 3 dense layers | 2,194 | 52.1 s / 2.0 s per layer through `converter dequant` (step 6, layer 3, measured) |
| KDA layers, 3 dense + 31 MoE, 64 calls of 512 | 1,832 | 0.76 s (dense) / 0.85 s (MoE) per 512-row call, synthetic real shapes with 16 experts, 16 threads (functional, 2026-10-08); KDA's cost per call does not grow with depth |
| DSA layers, 11 × 64 calls, base | 260 | 0.37 s per 512-row call at depth ≤ 512 (functional, same run) |
| DSA attention growth to 32,768 | 774 | 65,536 FLOP × T² / 2 per layer at an assumed 0.5 TFLOP/s (amendment 1) |
| 288 instead of 16 experts | 1,227 | 27.4 GB more f32 expert weights (288 − 16 experts × 100.7 MB) read per call and MoE layer, 42 × 64 calls, at an assumed 60 GB/s |
| states, dumps, sha256 | ≈ 225 | ≈ 5 s per layer, assumed |
| **per pass** | **≈ 6,510 (≈ 1.8 h)** | five passes ≈ 9 h |

Amendment 1 derived 1.4 h; the difference is KDA's chunk form, which is slower per call than the
0.5 TFLOP/s it assumed. **RAM per pass (derived):** about 48 GiB of 63.38, as in amendment 1. That is
28 GiB of f32 weights for a MoE layer, 2 + 2 GiB of f32 state in and out, 4 GiB of K/V allocated once,
and up to about 12 GiB in the last 512-row attention call. HF's cache would also have needed a second
4 GiB copy at each `torch.cat`; the in-place slot removes it. Run no other heavy job beside a pass.
