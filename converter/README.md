# crow-nest converter

| item | value |
|---|---|
| what it does | streams the original safetensors shard by shard and writes one CNQ container with an index v2 (Crow #300 C6, 2026-09-26; v1 before) |
| quantization | NVFP4 per ggml geometry, round to nearest, calibration free |
| what it never does | hold the model in RAM, judge its own output, or read a GGUF |
| spec | `../docs/architecture.md` section 1, approved 2026-09-02 |
| platform | Linux and Windows; the crate is pure Rust with no CUDA and no platform code, and it was untouched by the Linux port of 2026-09-17 (issue #15) |
| module comment of record | `src/main.rs:1-65` |

## Usage

```
usage: converter [--scales ceil|mse] --source-repo <org/name> [--revision <sha>] <model-dir | file.safetensors> <out.cnq>
  writes an index v2 container: config.json + generation_config.json verbatim, the family's recipe, source repo/revision/shard sha256
  (--revision defaults to the Hugging Face cache in the model dir; Crow #300 C6)
  --scales ceil  ceiling sub-block scales: stored >= raw always, max_rel <= 1.0 (default)
  --scales mse   per-sub-block SSE-minimizing scales: clipping allowed, quality via MSE report
       converter [--scales ceil|mse] requant-check <dense.safetensors> <container.cnq>
  re-quantizes fetched originals and compares them with the container's own bytes (#76)
       converter plan [--source-repo <org/name>] [--revision <sha>] <model-dir | file.safetensors>
  the dry run: family, recipe, per-tensor dtype/section table, GPU / host byte totals (Crow #300 C6)
```

- The usage text above is the `HELP` constant verbatim (`src/main.rs:503`).
- Input is a directory with `model.safetensors.index.json`, or a single `.safetensors` file.
- Two positional arguments are required; anything else exits 2.
- `config.json` and `generation_config.json` must sit in the model directory (or beside the single file): the index v2 carries them, and `text_config.model_type` picks the recipe. `--source-repo` is required; a missing one exits 2 before anything is written.
- The usage block above omits the overlay and `imatrix-show` lines of `HELP`.
- `requant-check` is the read-only subcommand of issue #76; it is described below.

## Container layout

| range | content | source |
|---|---|---|
| `[0..4)` | magic `CNQ1` | `src/main.rs:53`, spec section 1 approved 2026-09-02 |
| `[4..12)` | reserved, zeros | `src/main.rs:54`, spec section 1 approved 2026-09-02 |
| `[12..)` | payload blob, streamed | `src/main.rs:55`, spec section 1 approved 2026-09-02 |
| `[end-8-index_len .. end-8)` | index JSON, UTF-8 | `src/main.rs:56`, spec section 1 approved 2026-09-02 |
| `[end-8 .. end)` | `u64` little endian `index_len` | `src/main.rs:57`, spec section 1 approved 2026-09-02 |

- The index is a TRAILER, not a header: the payload streams to disk without knowing the index size up front (`src/main.rs:5-7`).
- The old index-first layout needed the whole blob in RAM (`src/main.rs:6-7`).
- Tensor offsets in the index are relative to blob start, which is `12` (`src/main.rs:58`).

## NVFP4 geometry

| quantity | value | source |
|---|---|---|
| values per block | 64 | `src/main.rs:9`, spec section 1 approved 2026-09-02 |
| sub-block scales per block | 4, `ue4m3`, one per 16-wide k-block | `src/main.rs:9-10`, spec section 1 approved 2026-09-02 |
| packed `E2M1` payload per block | 32 B | `src/main.rs:10` |
| stored bytes per block | 36 B | `src/main.rs:10` |
| effective width | 4.5 bpw | `src/main.rs:10` |
| second level | one global `f32` scale per tensor | `src/main.rs:11` |
| calibration | none, round to nearest | `src/main.rs:11` |

## Index v2 and the per-model recipe (Crow #300 C6, 2026-09-26)

| item | value |
|---|---|
| index keys added | `format_version: 2` (replaces `version: 1`), `recipe`, `scales`, `model` |
| `model` block | `config_json` and `generation_config_json` verbatim as strings, each with its hash in `<name>_sha256`; `family`, `model_type`, `geo`; `source.repo`, `source.revision`, `source.shards[]` (`file`, `size`, `sha256`, `sha256_from`: `hf-lfs` or `computed`) |
| recipe rows | `cnq4.5-flash-next` (`qwen4_exp_text`): the keep set below, verbatim. `cnq4.5-qwen35-dense` (`qwen3_5_text`): the phase 2 recipe, a whitelist |
| new dtype | `f32`: the dense row's `A_log`, widened exactly from BF16 |
| dry run | `converter plan <model-dir>`: headers and configs only |
| source | `src/recipe.rs`, `src/main.rs` `index_v2`; full table in `../docs/architecture.md` 1.7 |

- The Flash-Next row is proved against all 1658 tensors of the CNQ4.5-M index (`tests/fixtures/cnq45m-index.tsv`, extracted 2026-09-26).
- The engine accepts a v1 index only for the CNQ4.5-M container of record (`../engine/src/cnq.rs`, `CNQ45M_INDEX_SHA256`).

## BF16 keep set (the Flash-Next row)

- Approved as spec 1.2 on 2026-09-02, source `src/main.rs:13-17`.

- Embeddings and `lm_head`.
- The router GEMM.
- `shared_expert_gate`.
- All norms.
- Every 1-D tensor: biases, `A_log`, `dt_bias`, gates. Reason: under 0.1 % of the bytes, and a 1-D recurrence parameter must not go through a GEMM quant path.
- Any tensor whose length is not a multiple of `64` (`src/main.rs:17`).

## Section marks

- Decided 2026-09-02, source `src/main.rs:19-21`.

| mark | content | load |
|---|---|---|
| `text` | the default section | always |
| `ple` | `ngram_embedding`, NVFP4, block exchangeable to FP8 | always |
| `vit` | `model.visual`, carried in the container | optional |
| `mtp` | carried in the container | optional |

## Verification sidecar

- Source `src/main.rs:23-26`.

- File name: `<out>.cnq.sidecar.jsonl`.
- One JSON line per tensor with max and mean error against the sub-block scales.
- Computed during quantization by dequantizing in place, so it costs no second pass.
- It is gate 0 of the measurement ladder.
- Exit code 1 on any bound violation, in the mode that has a bound.
- The converter never judges its own output beyond that bound; FP8-KV and PLE-NVFP4 quality are oracle comparisons (`../docs/architecture.md:119-120`, and section 5 for the gates).

## Scale policy `--scales`

| policy | sub-block scale | bound | exit gate | source |
|---|---|---|---|---|
| `ceil` (default) | smallest `ue4m3` ladder step at or above the sub-block max | stored is always at or above raw, so values never clamp; per-element error is bounded by one `E2M1` half-gap; `max_rel <= 1.0` | exit 1 on any violation | `src/main.rs:29-31`, spec section 1 approved 2026-09-02 |
| `mse` | the `ue4m3` ladder step minimizing the summed squared error of the 16 `e2m1` values, by analytic pre-selection plus local refinement | clipping is DELIBERATE: elements above six times the scale are cut, so `max_rel <= 1.0` does NOT hold and no fixed relative bound holds | disabled; quality is reported instead | `src/main.rs:32-41`, spec section 1 approved 2026-09-02 |

- The `mse` mode is the one that voids the bound.
- Read that row before quoting a `max_rel` number from an `mse` container.

| under `mse`, every NVFP4 sidecar line gains | meaning | source |
|---|---|---|
| `mse` | the written encoding | `src/main.rs:41-42` |
| `mse_ceil` | the same weights re-encoded with ceiling scales | `src/main.rs:42-43` |
| `mse_ratio` | the two above, as a ratio | `src/main.rs:43` |
| `max_abs_clipped` | a COUNT of clipped elements, those above six times the scale; the name is per report spec | `src/main.rs:43-44` |
| one `record: "section_summary"` line per NVFP4 section | aggregated numbers for `text`, `vit`, `ple`, `mtp` | `src/main.rs:45-46` |

- The GLOBAL tensor scale stays max-based in BOTH modes, so ladder utilization is unchanged; only the sub-block scale choice differs (`src/main.rs:47-48`).
- Cost of `mse`: about two to three times the per-value quantization work of `ceil` (`src/main.rs:49-50`).

## `requant-check` — the proof of the fetched originals (issue #76)

```
converter requant-check <dense.safetensors> <container.cnq> [--threads N] [--limit N]
```

- Added 2026-09-18 as an ADDITIVE read-only subcommand. It writes nothing, and a run without the word `requant-check` parses, reads and writes exactly what it did before: the conversion path is untouched (`src/main.rs:60-65`).
- It reads a safetensors file of ORIGINAL tensors — the one `../tools/fetch-dense-originals.py` fetches back by HTTP range — runs every tensor through the same `quantize_nvfp4` the conversion runs, and compares the produced NVFP4 blocks and the `f32` global scale against what the container stores under that name.
- The comparison is on bytes: per tensor it reports how many `36` B blocks differ, the first one that does, and whether its four scale bytes or its packed nibbles are what disagree. The global scale is compared on its bits.
- Scale mode: `mse` by default, because that is the mode `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` was built with (`../docs/model-card.md`, Provenance). `--scales ceil` before the subcommand overrides it.
- It derives the tensor list a second time, from the container's own index (`section` `text`, `dtype` `nvfp4`, name without `.mlp.experts.`), and refuses if the fetched file is missing one of them. The fetch tool derives the same list from the sidecar — two independent derivations of one list.
- Exit code: `0` only if every tensor is identical including its global scale, `1` otherwise, `2` on a malformed argument or an unreadable file.
- What it proves and what it does not: see `../docs/dense-originals.md`.

## Tests

```
cd converter
cargo test --release
```

- Seven `#[test]` functions in `src/main.rs`, counted 2026-09-17, plus five in `src/requant_check.rs` added on 2026-09-18 (issue #76), so `cargo test --release` in this crate reads 12 passed, 0 failed on 2026-09-18.
- 2026-09-26 (Crow #300 C6): `cargo test` reads 57 passed, 0 failed (46 before C6; the eleven new ones cover the recipe rows, the index v2 round trip against `../engine/tests/fixtures/synthetic-v2/`, provenance and the per-family layer-rule arms).
- They are not part of the engine's count: `cd engine && cargo test --release` reads 165 passed, 0 failed on 2026-09-17 and covers `crow_nest_engine` and `bin/serve` only. `tools/gate-linux.sh` pins THAT count, not this one, so a test added here moves no gate value.
