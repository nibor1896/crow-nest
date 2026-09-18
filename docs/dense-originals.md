# The BF16 originals of the dense path, and the proof that they are the originals

Step 2 of the requant series (issue #76), after the characterization harness of #75. It changes
NOTHING about what the engine computes or what the converter emits: it obtains 5.17 GB of
original weights that were no longer on this machine, and it proves them against the 104.73 GB
of NVFP4 this repository already ships.

Why they are needed: `CNQ4.5-M` puts RTN NVFP4 on the whole dense path of `qwen4exp`, while the
Unsloth GGUF that answers cleanly at the same row keeps exactly those tensors high, and NVIDIA's
hybrid NVFP4 recipe does the same. Deciding that with numbers needs the BF16 originals - for the
reference-based readings #75 named as a later step (mean KLD, top-1 agreement) and for any
higher-precision rebuild. The 360 GB of `Qwen/Qwen3.8-Flash-Next` are on neither disk anymore.

## 1. What was fetched, and how the list was derived

The list is DERIVED twice, never typed. `tools/fetch-dense-originals.py` derives it from the
container's own sidecar `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq.sidecar.jsonl`: every line
with `"section": "text"` and `"dtype": "nvfp4"` whose name does not contain `.mlp.experts.`. The
four `record: "section_summary"` lines carry no `name` and are skipped. `converter requant-check`
derives the same list a second time from the container's index trailer and refuses if the fetched
file is missing one of them.

That is 495 tensors, 2,583,306,240 values, 5,166,612,480 B at BF16 - 2.9 % of the model's
values. The routed experts, 97 % of the bytes, were not touched.

| kind (per layer unless the count says otherwise) | tensors | values | bytes BF16 |
|---|---|---|---|
| `linear_attn.in_proj_qkv` | 36 | 943,718,400 | 1,887,436,800 |
| `linear_attn.in_proj_z` | 36 | 566,231,040 | 1,132,462,080 |
| `linear_attn.out_proj` | 36 | 566,231,040 | 1,132,462,080 |
| `self_attn.o_proj` | 12 | 188,743,680 | 377,487,360 |
| `mlp.shared_expert.down_proj` | 48 | 78,643,200 | 157,286,400 |
| `mlp.shared_expert.gate_proj` | 48 | 78,643,200 | 157,286,400 |
| `mlp.shared_expert.up_proj` | 48 | 78,643,200 | 157,286,400 |
| `ple.key_proj` (layer 1 only) | 1 | 26,214,400 | 52,428,800 |
| `self_attn.indexer.index_qk_proj` | 12 | 19,660,800 | 39,321,600 |
| `self_attn.v_proj` | 12 | 15,728,640 | 31,457,280 |
| `ple.value_proj` (layer 1 only) | 1 | 6,553,600 | 13,107,200 |
| `linear_attn.in_proj_a` | 36 | 4,423,680 | 8,847,360 |
| `linear_attn.in_proj_b` | 36 | 4,423,680 | 8,847,360 |
| `attn_hyper_connection.block_inject_weight` | 48 | 1,966,080 | 3,932,160 |
| `mlp_hyper_connection.block_inject_weight` | 48 | 1,966,080 | 3,932,160 |
| `linear_attn.conv1d` | 36 | 1,474,560 | 2,949,120 |
| `ple.conv1d` (layer 1 only) | 1 | 40,960 | 81,920 |
| total | 495 | 2,583,306,240 | 5,166,612,480 |

- Seventeen kinds, and they are the seventeen the ticket expected. The counts are what the
  container holds, not an architecture claim read from elsewhere: the three `ple.*` kinds appear
  once each and only in layer 1, `self_attn.*` in twelve layers, `linear_attn.*` in the other 36,
  and the two hyper-connection injects in all 48.
- The dtype of the originals is `BF16` for all 495, read out of the shard headers, not assumed.
  The tool refuses a tensor whose shard header disagrees with the container about the value count
  or the byte length.

## 2. Where they came from

| item | value |
|---|---|
| repository | `Qwen/Qwen3.8-Flash-Next` on Hugging Face, public, not gated |
| revision | `de4b8e4d43b917e7706784d8bb445c9af86a3540` - the revision of record of `docs/model-card.md` |
| URL form | `https://huggingface.co/<repo>/resolve/<revision>/model-000NN-of-00131.safetensors` |
| access | unauthenticated `urllib`, no token, no `huggingface_hub`, no venv |
| method | HTTP range requests; nothing but the wanted bytes and 51 shard headers crossed the line |

safetensors is what makes this cheap: the first 8 bytes of a shard are a little-endian `u64`
header length, the JSON header that follows names every tensor with its `dtype`, `shape` and
`data_offsets`, and those offsets are relative to the END of the header. One tensor is one range
request. The tool reads each shard header once (cached under `.shard-headers/`), then asks for
the tensor ranges, coalescing two of them into one request when the gap between them is at most
2 MiB.

## 3. The fetch of record, 2026-09-18

| item | value |
|---|---|
| shards touched | 51 of 131 |
| requests | 211 (51 headers, 158 data ranges) |
| bytes | 5,166,612,480 B of tensor payload |
| wall time | 825.7 s |
| mean rate | 6.3 MB/s over the whole run; the per-shard rates were 9.5 to 10.5 MB/s except where noted below |
| retries | 2 (both a `TimeoutError` on one range, retried after 2.0 s and then served), HTTP 429: 0 |
| output | `models/Qwen3.8-Flash-Next-original/dense/dense.safetensors`, 5,166,682,768 B (a 70,288 B header plus the payload) |
| manifest | `models/Qwen3.8-Flash-Next-original/dense/manifest.json`, one record per tensor |

- `models/` is gitignored; none of this is in the repository, and the manifest is what a later
  run or a later machine reproduces it from.
- The manifest names, per tensor: source shard, absolute byte range in that shard, byte count,
  `sha256` of the fetched bytes, dtype, shape and the offset it landed at in the output file,
  plus the repository, the revision and the run's own counters.
- The output is ONE valid safetensors file: original tensor names, original `BF16` dtype,
  contiguous `data_offsets` in name order with no holes and no overlap, `__metadata__` carrying
  `format: pt` and the source revision. Checked against the format spec and read back by the
  converter's own reader; there is no `safetensors` package on this machine to check it against
  the reference implementation.

## 4. The proof

A download is not evidence of itself. The proof is `converter requant-check`, an additive
read-only subcommand of the converter (`converter/src/requant_check.rs`, issue #76):

```
converter requant-check models/Qwen3.8-Flash-Next-original/dense/dense.safetensors \
                        converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
```

For every tensor of the fetched file it runs the SAME `quantize_nvfp4` the conversion ran - the
function itself, not a copy - with the same `--scales mse` the container was built with, and
compares two things against the container:

- the NVFP4 block bytes, 36 B per 64 values: 4 `ue4m3` sub-block scale bytes then 32 B of packed
  `E2M1` nibbles. Per tensor it reports how many blocks differ, the first one that does, and
  whether its scale bytes or its nibbles are what disagree.
- the `f32` global scale, compared on its bits.

Result of 2026-09-18, at commit 6874e13:

```
container converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq: 1658 tensors, 495 of them dense text nvfp4
fetched   models/Qwen3.8-Flash-Next-original/dense/dense.safetensors: 495 tensors, 2583306240 values, scales mse
OK    model.language_model.layers.0.attn_hyper_connection.block_inject_weight.weight  40960 values, 640 blocks, global 1.15530835e-4
...
OK    model.language_model.layers.9.mlp_hyper_connection.block_inject_weight.weight  40960 values, 640 blocks, global 1.1625744e-4
requant-check: 495 of 495 tensors byte-identical including the global scale (2583306240 values, scales mse, 8 threads, 7 s)
```

495 `OK` lines, no `DIFF` line, exit code `0`. Every one of the 2,583,306,240 values reproduced
the container's NVFP4 bytes and every one of the 495 `f32` global scales matched on its bits.
7 s of wall time over 8 threads, 944 MiB peak resident, no GPU.

### What this proves

- The fetched bytes ARE the tensors `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` was built from: 495 of 495
  reproduce its stored bytes exactly. The revision is the right one, the range arithmetic is
  right, and the file on disk is intact.
- The converter is CHARACTERIZED on that path: `quantize_nvfp4` under `--scales mse`, rebuilt
  today from today's toolchain, still emits what it emitted when the container was written. The
  container's own sidecar has been in this repository unchanged since its first commit, 7ba3ed6
  of 2026-09-05, so the conversion is at least that old; no commit in this tree dates the run
  itself. That is the baseline step 3 will be measured against - a byte that moves there will be
  a change somebody made, not drift.

### What this does NOT prove

- Nothing about quality. No KLD, no top-1 agreement, no answer is measured here. This is a
  checksum with arithmetic in it.
- Nothing about the routed experts, the ViT, the MTP head or the PLE embedding shards: they were
  not fetched and are not compared.
- It cannot see a difference in the originals that NVFP4 would have collapsed anyway. The
  comparison is on the QUANTIZED bytes, so a corrupted element whose value still rounds to the
  same `E2M1` level under the same sub-block scale passes. A negative control on synthetic
  weights showed both sides of this: a flipped mantissa bit in a small element next to a large
  outlier was invisible, a flipped exponent bit was reported as `1 of 128 blocks differ, first
  block 0 (nibbles)`. What the check is strong against is the failure this ticket had to rule
  out - wrong offsets, a wrong revision, a truncated or shifted download - which moves whole
  blocks, not one nibble.
- It says nothing about determinism across machines. Both sides here were produced on this
  machine; the container's own run was too.

## 5. How to re-run it

```
tools/fetch-dense-originals.py --dry-run        # the plan: shards, bytes per shard, total
tools/fetch-dense-originals.py                  # fetch what is missing, verify what is there
python tools/test_fetch_dense_originals.py      # 35 unit tests, no network

cd converter && cargo build --release
./converter/target/release/converter requant-check \
    models/Qwen3.8-Flash-Next-original/dense/dense.safetensors \
    converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
```

- The fetch is idempotent: a second run fetches nothing, re-reads all 5.17 GB from disk and
  re-verifies every `sha256` against the manifest. `--skip-verify` trusts the manifest instead.
- It is resumable per tensor: the manifest is rewritten after every shard and a tensor enters it
  only after its bytes are on disk and hashed, so an interrupted run loses at most the tensor it
  was inside.
- `--dry-run` touches no network at all. Everything else it needs - the selection, the shard map,
  the byte counts - it reads from the sidecar and `model.safetensors.index.json`.
- Neither tool needs a GPU. `requant-check` uses one thread per core up to 8 and reached 944 MiB
  peak resident in total on 2026-09-18; nothing else on this machine needs to be idle for it.
- Both are polite: sequential requests, `Retry-After` honoured in both its forms, a response that
  is not a `206` of exactly the asked length refused rather than retried into a whole shard.
