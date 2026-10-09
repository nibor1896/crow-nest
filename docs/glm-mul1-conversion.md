# GLM-5.3-Flash: the 3-bit MUL1 container

crow-nest #182, plan steps 11/12 (root #169). The routed experts of GLM-5.3-Flash become MUL1 trellis
records at K = 3 (exllamav3's codebook, codec #181), quantized by exllamav3's `quantize_exl3` from
the FP8 originals (`zai-org/GLM-5.3-Flash` @ `eb9eb208`) with Hessians from the layerwise runner's MoE
inputs. The dense part stays on `cnq4.5-glm5-next`, byte for byte. Nothing on this page has run on
the real shards yet; every runtime is derived.

## Data flow

| stage | interpreter | reads | writes | resumes from |
|---|---|---|---|---|
| `tools/glm_mul1_quantize.py capture` | `.venv-oracle` (CPU, transformers 5.16.1) | FP8 originals, the four calibration files | `<work>/L<ll>/moe-in.bf16` [131,072][4096], `ids.i32` [131,072][8], `capture.json`; hand-over states `<work>/states/<file>/l<k>-output.bf16` | `<work>/states/progress.json` (last layer with states for every file) |
| `tools/glm_mul1_quantize.py quantize` | `.venv-exl3` (GPU, exllamav3 1.6.0, torch 2.13.0+cu132) | FP8 originals (expert weights), the captures | `<store>/L<ll>/E<eee>.safetensors`, `<store>/journal.jsonl`, `<store>/store.json` | `journal.jsonl` (one synced line per expert) |
| `converter --experts-mul1 <store>` | converter | FP8 originals (dense part), the store | the container, its journal and sidecar, `<store>/L<ll>/E<eee>.done` | `<out>.cnq.journal.jsonl` (as every conversion) |

- **The originals are only read.** A `--work` or `--store` inside the FP8 directory is refused. Deleted
  are only derived files: hand-over states behind the newest layer, a layer's capture once its 288
  records are in the store (`--keep-capture` keeps it), and with `--prune-consumed <container>` the
  record files whose `.done` names that container.
- **capture** runs every calibration file through each layer with the runner's own `run_layer`
  (`oracle/glm5_layerwise.py:200`), one layer load for all four files, and dumps the input of
  `layer.mlp` (the `post_attention_layernorm` output) by a forward pre-hook, in BF16. The MTP block
  (layer 45) has no forward in the runner (`docs/glm5-next-recipe.md` O1) and is not captured.
  `--max-ahead 2`: at most two captured layers wait for the quantizer.
- **quantize**, per expert: gate/up/down dequantized from FP8 (`oracle/glm5_common.py` `WeightSource.get`);
  `H_gu = X^T X` over all 131,072 rows of the layer (exllamav3 calibrates every expert on every token
  for this architecture: `architecture/glm5_next.py:348` `calibration_all_experts = True`);
  `H_down = A^T A`, `A = silu(min(X Wg^T, 10)) * clamp(X Wu^T, -10, 10)` (recipe section 10, the
  FP8-dequantized weights); `quantize_exl3(W^T, H, {K: 3, mul1, seed, sigma_reg 0.025, apply_out_scales
  auto})`, gate and up on one H as exllamav3 does for Q/K/V. Seed `(layer * 288 + expert) * 3 + {0, 1, 2}`.
  `--hessian-tokens routed` takes only the rows routed to the expert (about 3,640 on average, below the
  4096-wide input; heavily damped). The MTP layer's experts get an identity Hessian (uncalibrated LDLQ),
  recorded per journal line as `"hessian": "identity"`.
- **converter** (`converter/src/mul1_store.rs`, `src/main.rs` `mul1_*`): the dense part through the
  unchanged `encode_tensor`, then one record per expert in (layer, expert) order, built by
  `mul1::write_record` from the expert file after its sha256 matched the store journal. `--mul1-wait`
  waits for `store.json` (the quantizer writes it once the capture's `calibration.json` exists) and for records
  still being quantized; without it a missing `store.json` or record is refused before a byte is written.
  `quantize --wait` likewise waits for `calibration.json` and for each layer's `capture.json`, polling every
  30 s with no timeout, so all three terminals of (b) can be started at once; without `--wait` a missing
  capture is refused.

## Calibration set

PREREG amendment 1's four `cal` files (`decode_out/glm-step8/corpus`, sha256 pinned in
`tools/glm_route_passes.py:73-84`): `omarchy-0915a`, `lenis-0830`, `ctx7-0830`, `zetalab-0829`, each
32,768 tokens, **n = 131,072 tokens** (25.6 % of exllamav3's default 250 x 2048 = 512,000,
`conversion/convert_model.py:275-276`). The held-out `todo-1006` is refused. The capture records the
files, their sha256 and the FP8 identity in `<work>/calibration.json`; the store copies it into
`store.json` and the container into its `expert_codec` block.

## Container

| item | value |
|---|---|
| experts | 42 MoE layers (3-44, section `text`) + the MTP block (45, section `mtp`, optional to load) = 43 x 288 = 12,384 records; the rest of the MTP block stays omitted as in `cnq4.5-glm5-next` |
| record | 9,474,048 B = 2313 x 4096: `[gate.trellis][up.trellis][down.trellis][gate.suh][gate.svh][up.suh][up.svh][down.suh][down.svh]`, trellis 3,145,728 B each (`converter/src/mul1.rs`) |
| order | the dense part first (the 4.5 recipe's units, same bytes), then the records, each on a 4096-B file offset |
| index entries | dtype `mul1`; gate at the record start, up at + 3,145,728, down at + 6,291,456 (`RecordLayout::tensor_offsets`); `len` 3,145,728 / 3,145,728 / 3,182,592, so "next offset - gate offset" is the record (`engine/src/nvme_source.rs:196`); `mul1: {k, record_offset, record_bytes}` |
| index top level | `recipe` stays `cnq4.5-glm5-next` (the dense rule); new `expert_codec`: dtype, K, hidden, inter, trellis and record bytes, records, layout, the store's sha256, quantizer and calibration |
| size | non-expert part 7,250,323,716 B + 12,384 x 9,474,048 B = 124.58 GB = 116.0 GiB (derived from `converter plan` 2026-10-08) |
| decode | `converter dequant` and so the oracle's `--weights container` decode `mul1` to the original-basis weight `diag(suh) H W_hat H diag(svh) / 128` in f64, one f32 rounding (#156, 2026-10-09; `converter/README.md`) |

## Disk

Only drive C: exists (2026-10-09); about 168 GiB will be free after the FP8 download. A full store
beside a full container would need 225 GiB, so the full run streams: the converter writes each record
as the quantizer journals it (`--mul1-wait`), and the quantizer deletes the records the container holds
(`--prune-consumed`).

| run | peak (derived) | refused below |
|---|---|---|
| (a) layers 0-3 | container 5.5 GiB + hand-over states 8 GiB + one capture 1 GiB + 288 records 2.5 GiB = 17.0 GiB | that + 16 GiB |
| (b) full | container 116.0 GiB + states 8 GiB + two captures 2 GiB + 576 records of backlog 5.1 GiB = 131.1 GiB | that + 16 GiB |

Each stage checks its own share before it starts: `capture` (states + `--max-ahead` captures),
`quantize` (the records it will write, or 576 of backlog with `--prune-consumed`, re-checked before each
expert), the converter (the container bytes still to write; `--disk-reserve-gib N`, default 16). All
refuse with exit 2 and the numbers. `python -I tools/glm_mul1_quantize.py plan` prints the whole verdict.

## Runtime (derived, not measured)

| part | (a) layers 0-3 | (b) full | basis |
|---|---|---|---|
| capture | 0.4 h | 4.9 h | one 32,768-token runner pass on the FP8 originals ≈ 4,650 s (`docs/glm5-reference-runner.md` 8, derived), compute ≈ 96 s per layer per file, x 4 files; one layer load per layer (7.8 s FP8 MoE layer, step 6, measured once) |
| quantize | 0.1 h | 5.0 h | 1.04 s per expert in `quantize_exl3` (measured 2026-10-09: 0.373 + 0.353 + 0.311 s, RTX 5090, synthetic weights and Hessian, n = 2 reps, warm) + 0.4 s FP8 read, H_down, write (assumed) |
| convert | 0.03 h | 0.1 h | step 6: 244 s for 7.22 GB at `--scales mse` (measured once); records at 1 GB/s with two sha256 passes (assumed) |
| wall | 0.6 h, one after the other | about 5 h, the three side by side (quantize trails capture by at most 2 layers) | |

## Commands

Build the converter first (`cd converter && cargo build --release`). The FP8 originals must be verified
by `tools/fetch-glm.py` (all 62 shards with their `.verified` marker); `capture` and `quantize` refuse
otherwise. Ask robin before either run (GPU and multi-hour CPU jobs).

```
python -I tools/glm_mul1_quantize.py plan        # the verdict and the commands below, nothing runs

# (a) step 11: the partial container of layers 0-3, one after the other
.venv-oracle/Scripts/python.exe -I tools/glm_mul1_quantize.py capture --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work --layers 0-3
.venv-exl3/Scripts/python.exe -I tools/glm_mul1_quantize.py quantize --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work --store decode_out/glm-mul1/store --layers 0-3
converter/target/release/converter.exe --scales mse --source-repo zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --experts-mul1 decode_out/glm-mul1/store --layers 0-3 --with-embed-head models/GLM-5.3-Flash-original converter/GLM-5.3-Flash-MUL1K3-L0-3.cnq

# (b) the full container: three terminals side by side
.venv-oracle/Scripts/python.exe -I tools/glm_mul1_quantize.py capture --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work
.venv-exl3/Scripts/python.exe -I tools/glm_mul1_quantize.py quantize --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-mul1/work --store decode_out/glm-mul1/store --wait --prune-consumed converter/GLM-5.3-Flash-MUL1K3.cnq
converter/target/release/converter.exe --scales mse --source-repo zai-org/GLM-5.3-Flash --revision eb9eb208eb0d988989d07a6a12d0fdeb5f52574a --experts-mul1 decode_out/glm-mul1/store --mul1-wait models/GLM-5.3-Flash-original converter/GLM-5.3-Flash-MUL1K3.cnq
```

- (b) continues from (a): the capture resumes at layer 4 from (a)'s states, the quantizer skips layer
  3's journalled experts, the converter consumes them from the store.
- Each command is resumable: run it again after a kill. `--scales mse` is the PREREG scale policy of
  the 4.5 recipe (`runs/glm53-flash/PREREG.md`), so the dense bytes equal the 4.5 container's.
- RAM (derived): capture ≈ 48 GiB of 63.38 as a runner pass, quantize a few GiB, the converter small.

## Tests

```
cd converter && cargo test --release                                   # 100 passed, 1 ignored (2026-10-09)
python -I tools/test_glm_mul1_quantize.py                              # pure: 10 passed, 4 skipped
.venv-oracle/Scripts/python.exe -I tools/test_glm_mul1_quantize.py     # + capture vs HF: 12 passed, 2 skipped
.venv-exl3/Scripts/python.exe -I tools/test_glm_mul1_quantize.py       # + quantize on the GPU and the converter: 12 passed, 2 skipped
```

- Converter (miniature: hidden 128, inter 256, layer 3 experts 0-1 and the MTP expert 0): record layout
  and alignment, index entries under the engine's rule, dense part identical to `cnq4.5-glm5-next`,
  kill-and-resume byte identity at six points, `--mul1-wait` (also started before `store.json` exists), `--layers 3`, and the refusals (missing
  record, sha256 mismatch, other shapes / record size / K, wrong dtype, `--consume`, disk).
- Tool: capture equals HF's full model's MoE input and routing on every row of the runner's synthetic
  small checkpoint; quantize writes exllamav3's tensors, resumes without duplicates, uses the identity
  Hessian for layer 45, and the built converter accepts its store.

## Open

- MTP experts are uncalibrated (identity Hessian) until a forward exists (O1, step 21); exllamav3
  quantizes MTP at 4 bits by default (`convert_model.py:268`), the container keeps one record size.
- Activations come from the FP8 trunk, not from already-quantized layers as in exllamav3's own
  converter; exllamav3's `convert.py` on the whole model (it ships `glm5_next`) is the comparison arm if
  step 11's numbers disappoint.
