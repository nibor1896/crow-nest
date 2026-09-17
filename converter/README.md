# crow-nest converter

| item | value |
|---|---|
| what it does | streams the original safetensors shard by shard and writes one CNQ v1 container |
| quantization | NVFP4 per ggml geometry, round to nearest, calibration free |
| what it never does | hold the model in RAM, judge its own output, or read a GGUF |
| spec | `../docs/architecture.md` section 1, approved 2026-09-02 |
| platform | Linux and Windows; the crate is pure Rust with no CUDA and no platform code, and it was untouched by the Linux port of 2026-09-17 (issue #15) |
| module comment of record | `src/main.rs:1-59` |

## Usage

```
converter [--scales ceil|mse] <model-dir | file.safetensors> <out.cnq>
  --scales ceil  ceiling sub-block scales: stored >= raw always, max_rel <= 1.0 (default)
  --scales mse   per-sub-block SSE-minimizing scales: clipping allowed, quality via MSE report
```

- The usage text above is the `HELP` constant verbatim (`src/main.rs:494`).
- Input is a directory with `model.safetensors.index.json`, or a single `.safetensors` file.
- Two positional arguments are required; anything else exits 2.

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

## BF16 keep set

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

## Tests

```
cd converter
cargo test --release
```

- Seven `#[test]` functions in `src/main.rs`, counted 2026-09-17. They are not part of the engine's count: `cd engine && cargo test --release` reads 165 passed, 0 failed on the same day and covers `crow_nest_engine` and `bin/serve` only.
