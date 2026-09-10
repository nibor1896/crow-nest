# crow-nest engine

| item | value |
|---|---|
| crate | `crow_nest_engine`, Rust, thin CUDA kernels through NVRTC and cudarc |
| target | NVIDIA Blackwell, `compute_120a` (plain `sm_120` is rejected by ptxas) |
| spec | `../docs/architecture.md`, sections 0 to 7 |
| environment variables | `../docs/env.md`, one row per `CROW_*` name with its `file:line` |

## Binaries

- One row per `[[bin]]` entry in `Cargo.toml`.

| binary | purpose | kind |
|---|---|---|
| `serve` | the HTTP face of the engine, spec section 7 | product |
| `decode` | one end-to-end run, parity dumps, layer check | product |
| `parity` | the standing ten-task harness, A/B against llama.cpp | product |
| `states` | acceptance demo of the three-state memory manager (issue #9) | probe |
| `residency` | acceptance demo of the residency scheduler (issue #8) | probe |
| `plecheck` | isolated `Ple::load` hang reproduction | probe |
| `kcheck` | NVRTC compile of the consolidated kernel source plus FP8 validation | probe |
| `mma_probe` | scale-fragment layout pin for `mma.sync ... mxf4nvf4.block_scale` | probe |
| `mma_gate` | gate for the FP4 tensor-core path against the reference GEMV | probe |
| `mma_probe2` | ue4m3 scale-byte edge decode on the MMA hardware | probe |
| `sf_scan` | scan container expert scale bytes for the NaN encoding `0x7F` | probe |
| `graph_probe` | CUDA-graph node cost on this machine (WDDM) | probe |
| `pcie_probe` | cold-expert staging bandwidth over PCIe | probe |
| `pin_probe` | what limits pinned host allocation | probe |
| `pin_leak` | device memory per pinned host allocation (issue #18) | probe |
| `coldtier` | low-bit COLD tier builder | probe |
| `hybrid` | hybrid-container experiment | probe |

## Build

```
cd engine
cargo build --release --bin serve
```

### Target directory rule, for every build next to a measurement chain

- Build into a target directory of its own, never under a running chain.
- `cargo build --release --target-dir target_gate --bin decode` is the shape.
- A rebuild of the same source produces a different sha, so it is ungated until parity runs again.
- Delete the extra target directory when the task that made it is done.
- `.gitignore` ignores every `target*` directory (issue #45).

## serve

```
serve [--port <n>] [--slot-save-path <dir>]
```

- Start it from the repository root: container and hot-set paths are repository relative.
- Default port 8099, bind address `127.0.0.1` (`src/bin/serve.rs:445`).
- Container default `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` (`src/bin/serve.rs:446`), `CROW_CNQ` overrides it.
- Hot-set default `decode_out/hotsets-M-longctx2100-n160.json` (`src/bin/serve.rs:447`), `CROW_HOTSETS` overrides it.
- Prompt chunk pinned at `2048` for the whole process (`src/bin/serve.rs:448-449`).
- Blocking, one request at a time, no async runtime; a second connection waits in the accept queue.
- One engine per machine: `Engine::load` takes `engine/.engine.lock` before anything is pinned.
- `--slot-save-path` must name a directory that already exists; a typo exits 2 before the engine loads.

### Endpoints

| method and path | answer | anchor |
|---|---|---|
| `GET /health` | `{"status":"ok"}` | `src/bin/serve.rs:495` |
| `GET /props` | the operating point as a JSON document, field names mirror llama-server | `src/bin/serve.rs:496` |
| `POST /v1/chat/completions` | SSE `chat.completion.chunk` frames, or one `chat.completion` document when `stream` is `false` | `src/bin/serve.rs:497` |
| `GET /slots` | the one slot of this process as an array of one | `src/bin/serve.rs:498` |
| `POST /slots/0?action=save\|restore` | writes or reads the slot file; `400` without `--slot-save-path` | `src/bin/serve.rs:499` |
| anything else | `404` with a JSON body | `src/bin/serve.rs:855` |

### Request fields read by `POST /v1/chat/completions`

| field | behaviour | anchor |
|---|---|---|
| `messages` | required, non-empty array, every entry needs a string `role` | `src/bin/serve.rs:128` |
| `stream` | `true` streams frames, `false` or absent answers one document | `src/bin/serve.rs:129` |
| `max_tokens` | default `1024`, capped at `32768`, clamped to `n_ctx` minus prompt ids | `src/bin/serve.rs:130` |
| `model` | echoed, default `crow-nest` | `src/bin/serve.rs:131` |
| `temperature` | absent, `null` or at most zero is greedy; above zero samples | `src/bin/serve.rs:133` |
| `top_p`, `top_k`, `presence_penalty`, `seed` | read only when `temperature` is above zero; `top_k` clamped to `64` by the device sampler (`SAMPLE_MAXK`, `engine/src/kernels.rs:2942`) | `src/bin/serve.rs:134-137` |
| `min_p` | accepted and ignored, the device sampler has none (issue #28) | `src/bin/serve.rs:138` |
| `tools` | OpenAI function tools, rendered into the chat template | `src/bin/serve.rs:139` |
| `stream_options.include_usage`, `timings_per_token` | add `usage` and `timings` to the final chunk | `src/bin/serve.rs:140-141` |

- `serve` reads none of the `CROW_SAMPLE`, `CROW_TEMP`, `CROW_TOP_P`, `CROW_TOP_K`, `CROW_PRESENCE`, `CROW_SEED` variables; the sampler comes from the request (`src/bin/serve.rs:196-197`).
- Those variables keep working for `decode` and `parity`.

### Subcommand `serve tokenize`

```
serve tokenize --chat --file <prompts.json> --out <ids.json>
serve tokenize --chat --text "<text>"
serve tokenize --raw  --text "<text>"
```

- No CUDA context, no `.engine.lock`, no GPU (`src/bin/serve.rs:93`, `:108`).
- Exit codes: 0 written or printed, 2 usage, 3 tokenizer load failed, 4 IO or encode.

## decode

```
decode parity <ids.json> <out_dir>
decode run <ids.json> <gen> <out_dir>
decode longctx <prompt_ids> <fill> <gen>
decode layercheck
```

- `decode parity` dumps every position's logits as `<out_dir>/gpu-logits.f32` plus a greedy trace.
- `decode run` prefills and then takes `<gen>` decode steps with per-step timing; this is the engine side of the standing series.
- `decode layercheck` compares layer 0 against the golden.

## parity

```
parity <run> <prompts.json> [llama-url]
parity <phase> <run_index> <crow|llama> <prompts.json> [llama-url] [outprefix]
```

- `run` is interleaved and needs both engines resident at once.
- `phase` runs one arm over the full rotated order; that is the form this machine uses, because of its RAM and VRAM budget.
- Example of record: `parity phase 0 crow decode_out/ten-tasks.json e3`.

## Parity gate

1. Build the candidate into its own target directory, then record `sha1sum` of both `decode.exe` binaries.
2. Export the operating point, with `CROW_ADAPT`, `CROW_SAMPLE`, `CROW_CHUNK` and `CROW_CHUNK_AUTO` unset:

```
CROW_GRAPH=1
CROW_MMA=1
CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json
```

3. Run reference and candidate over both id files, each into its own output directory:

```
decode parity decode_out/parity-ids.json  <dir>
decode parity decode_out/real512-ids.json <dir>
```

4. Compare with `cmp -s <ref>/gpu-logits.f32 <new>/gpu-logits.f32`; every pair must be byte-identical, and the `real512` form runs twice because an eight-token test alone hid a barrier race until 2026-09-04.
5. Delete the output directories afterwards; one `real512` pair is 512,532,480 B per side.

| id file | logit rows | verdict of record | date |
|---|---|---|---|
| `decode_out/parity-ids.json` | 12 | identical against `d211ab52ad2b` | 2026-09-10 |
| `decode_out/real512-ids.json` | 516 | identical against `d211ab52ad2b` | 2026-09-10 |
| `decode_out/t3-debug-1024-ids.json` | 1028 | identical against `d211ab52ad2b` (issue #33) | 2026-09-10 |

- The third row is the largest chunk form in production; it has been a gate since 2026-09-05 (issue #22).
- It is run with `CROW_CHUNK=1024` set, twice, the same way.
- Reference build of record: `d211ab52ad2b` (issue #43).

## Machine rules

- One engine process per machine, never two GPU jobs.
- Stop a run by PID, never by process name: a name filter matches the shell that started it.
- Wait for more than 50.5 GiB free host RAM before a load (rule since 2026-09-10, issue #38).
- A tok/s number is only quoted next to a decode run from the same session (issue #38).

## Tests

```
cd engine
cargo test --release
```
