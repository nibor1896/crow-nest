# crow-nest engine

| item | value |
|---|---|
| crate | `crow_nest_engine`, Rust, thin CUDA kernels through NVRTC and cudarc |
| target | NVIDIA Blackwell, `compute_120a` (plain `sm_120` is rejected by ptxas) |
| platform | Linux and Windows; the Linux port landed 2026-09-17 (issue #15) and the Windows path is byte-for-byte the old code under `#[cfg(windows)]` |
| spec | `../docs/architecture.md`, sections 0 to 9 (8 is the code map, 9 is logging) |
| environment variables | `../docs/env.md`, one row per `CROW_*` name with its `file:line` |
| logs | `tracing`, one rotating gzipped file plus the stderr mirror; `CROW_LOG` sets the level per component without a rebuild (issue #13, 2026-09-18; spec 9) |

## Binaries

- One row per `[[bin]]` entry in `Cargo.toml`. Eighteen entries, counted 2026-09-17.
- The product binaries are named `serve`, `decode` and `parity` on Linux and `serve.exe`, `decode.exe` and `parity.exe` on Windows; every command below prints the Linux name.
- Every binary that opens the container defaults to `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` and lets `CROW_CNQ` override it; `plecheck` and `states` were the last two to be brought over (issue #60, 2026-09-18).

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
| `qsa_probe` | the parallel QSA selection against the single-block form, row by row | probe |
| `router_probe` | the `CROW_ROUTER_GEMM` and `CROW_PF_GEMM_B` router forms against the reference | probe |
| `sf_scan` | scan container expert scale bytes for the NaN encoding `0x7F` | probe |
| `graph_probe` | CUDA-graph node cost on this machine (measured on Windows/WDDM; not re-run on Linux) | probe |
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

- The same command builds on both platforms. Stable Rust; `libc 0.2` is the only dependency the Linux port added (issue #15, 2026-09-17). Nothing links CUDA at build time (`cudarc` `dynamic-loading`).

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
- On Linux, start it through `../tools/serve-linux.sh`: it puts the process in a transient scope (`systemd-run --user --scope --slice=session.slice`, `MemorySwapMax=0`, `MemoryHigh=MemTotal-8G`, `MemoryMax=MemTotal-6G`) and sets `LD_LIBRARY_PATH` from `CUDA_LIB` (default `~/.local/share/crow/cuda/lib`, never the `lib/stubs` sibling).
- Default port 8099, bind address `127.0.0.1` (`src/bin/serve.rs:491`, `src/bin/serve.rs:2983`).
- Container default `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` (`src/geo.rs:75`), `CROW_CNQ` overrides it.
- Hot-set default `decode_out/hotsets-M-longctx2100-n160.json` (`src/geo.rs:76`), `CROW_HOTSETS` overrides it. The file is one JSON object with a `sets` array of 48 rows of expert ids; a row that is not as long as the run's N is padded or truncated and named in the load log, and a file that is not a hot set is refused by name (`src/residency.rs` `sidecar_sets`, issue #49, 2026-09-18).
- Prompt chunk pinned at `2048` for the whole process, derived from `geo::TRICKLE_CHUNK_THRESHOLD` (`src/bin/serve.rs:497`, `src/geo.rs:57`).
- Blocking, one request at a time, no async runtime; a second connection waits in the accept queue.
- One engine per machine: `Engine::load` takes `engine/.engine.lock` before anything is pinned. On Linux the host-memory budget is derived at boot as `min(46 GiB cap, free_for_pin - CROW_RAM_MARGIN_GB)` and a second live CUDA process (detected by a foreign holder of `/dev/nvidia-uvm`) drops that basis to `MemAvailable` (`src/manager.rs:55`, issue #15).
- `--slot-save-path` must name a directory that already exists; a typo exits 2 before the engine loads.
- The planner reserves the vision path's VRAM before it chooses N: `[budget] vit reserve 277.3 MB (tower scratch 228.5 + mrope span 48.8)` at `n_ctx` 200,000, measured 2026-09-17 (`src/vit.rs:108`, `CROW_VIT_RESERVE_MB`). No per-request VRAM is allocated for images.

### Endpoints

| method and path | answer | anchor |
|---|---|---|
| `GET /health` | `{"status":"ok"}` | `src/bin/serve.rs:543` |
| `GET /props` | the operating point as a JSON document, field names mirror llama-server | `src/bin/serve.rs:544` |
| `POST /v1/chat/completions` | SSE `chat.completion.chunk` frames, or one `chat.completion` document when `stream` is `false` | `src/bin/serve.rs:545` |
| `GET /slots` | the one slot of this process as an array of one | `src/bin/serve.rs:546` |
| `POST /slots/0?action=save\|restore` | writes or reads the slot file; `400` without `--slot-save-path` | `src/bin/serve.rs:547` |
| anything else | `404` with a JSON body | `src/bin/serve.rs:905` |

### Request fields read by `POST /v1/chat/completions`

- The body is parsed by `parse_chat` (`src/bin/serve.rs:1033`); unknown fields are accepted and ignored, as llama-server does.

| field | behaviour | anchor |
|---|---|---|
| `messages` | required, non-empty array, every entry needs a string `role` | `src/bin/serve.rs:977` |
| `content` parts | a string, `null`, or a list of `text` and `image_url` blocks; an `image_url` block without `image_url.url` is a `400` that names the message index | `src/bin/serve.rs:997-1006` |
| `stream` | `true` streams frames, `false` or absent answers one document | `src/bin/serve.rs:1035` |
| `max_tokens` | default `1024`, capped at `32768`, clamped to `n_ctx` minus prompt ids | `src/bin/serve.rs:1041` |
| `model` | echoed, default `crow-nest` | `src/bin/serve.rs:1058` |
| `stream_options.include_usage`, `timings_per_token` | add `usage` and `timings` to the final chunk | `src/bin/serve.rs:1065-1071` |
| `temperature` | absent, `null` or at most zero is greedy; above zero samples | `src/bin/serve.rs:1076` |
| `top_p`, `top_k`, `presence_penalty`, `seed` | read only when `temperature` is above zero; `top_k` clamped to `64` by the device sampler (`SAMPLE_MAXK`, `src/kernels.rs:3994`) | `src/bin/serve.rs:1077-1088` |
| `min_p` | accepted and ignored, the device sampler has none (issue #28) | `src/bin/serve.rs:1079` |
| `tools` | OpenAI function tools, rendered into the chat template | `src/bin/serve.rs:1013` |

- `serve` reads none of the `CROW_SAMPLE`, `CROW_TEMP`, `CROW_TOP_P`, `CROW_TOP_K`, `CROW_PRESENCE`, `CROW_SEED` variables; the sampler comes from the request.
- Those variables keep working for `decode` and `parity`.
- A message shape the chat template cannot render is refused BEFORE the render with a body that names the message index and the field (`check_messages` `src/bin/serve.rs:1564`, `check_content` `:1601`, `check_tool_call` `:1640`); a `function.arguments` that is not a mapping is rewritten instead of refused (`normalize_messages` `src/bin/serve.rs:1481`): a string that parses to an object becomes that object, `null` becomes `{}`, anything else becomes `{"_raw": "<verbatim>"}`. Every rewrite logs one `[chat] normalised` line (TASK J, 2026-09-17).
- The `[chat] sampling on the device` line names the SOURCE of every value (issue #68, 2026-09-18): `temperature 1 (request) top_p 0.95 (request) top_k 20 (data sheet) presence_penalty 1.5 (data sheet) seed 0 (data sheet)`, plus one line saying the presence-penalty set is cleared per request and holds this answer's generated tokens only. The live goal-mode session was read off the old line as "the client sent presence_penalty 1.5"; the client has no such field. The defaults are unchanged (issue #28) and nothing on the wire moves.
- No `<think>` or `</think>` ever leaves as `content` (issue #67, 2026-09-18): a leading `<think>...</think>` block goes out as `reasoning_content` instead and every bare `</think>` is dropped, on the stream and on the document alike (`ThinkFilter`, `send_emits`); a tag split across deltas is held back until it resolves. The same two shapes are stripped off a STORED assistant `content` before the render (`strip_stored_think` in `normalize_messages`), so a client that already stored one cannot poison its next turns. It changes what is streamed, never what is sampled: the generated ids are untouched.
- A `stream:false` generation ends when its client is gone (issue #54, 2026-09-18): the document path writes nothing until the answer is complete, so it cannot learn of a gone client from a failed write the way the stream does, and it asks instead - one `poll(POLLRDHUP)` on the request socket between two decode steps, no byte read and no byte written, 0.10 us per probe (`ClientProbe`, `CollectSink::still_there`). A client that only half-closed its write side is NOT gone and still gets its whole document; the reason and the step go to one `[chat]` line and the `[serve]` line says `200 OK (client gone)`. Linux only (`POLLRDHUP`), inert elsewhere, and the SSE path is unchanged on every platform. `../tools/replay-toolcalls.py --gone-client` is the live test.
- A CUDA allocation refused inside a request raises `cuda::AllocFailed`, which `guarded` (`src/bin/serve.rs:2678`) catches: it frees what was taken, resets the engine and answers `503` with a body naming the allocation, its byte count and the free VRAM. It is the only `503` this server answers; any other panic still ends the process (TASK K, 2026-09-17).

### Subcommand `serve tokenize`

```
serve tokenize --chat --file <prompts.json> --out <ids.json>
serve tokenize --chat --text "<text>"
serve tokenize --raw  --text "<text>"
```

- No CUDA context, no `.engine.lock`, no GPU (`src/bin/serve.rs:710`, dispatched at `:2870`).
- Exit codes: 0 written or printed, 2 usage, 3 tokenizer load failed, 4 IO or encode.

## decode

```
decode parity <ids.json> <out_dir>
decode run <ids.json> <gen> <out_dir>
decode longctx <prompt_ids> <fill> <gen>
decode layercheck
decode layercheck3
decode selftest [<golden_dir>]
```

- `decode parity` dumps every position's logits as `<out_dir>/gpu-logits.f32` plus a greedy trace.
- `decode run` prefills and then takes `<gen>` decode steps with per-step timing; this is the engine side of the standing series.
- `decode layercheck` compares layer 0 against the golden in `../oracle/golden/` and must be run from `engine/`, because both paths are relative to that directory (`tools/perf_loop.sh` is the form of record).
- `decode layercheck3` does the same for the self_attn sub-block of layer 3, the first full-attention layer, against `../oracle/golden/layer3-attn-*.f32`: it runs the production `attn_prompt` on the golden `[8][2560]` `mixed` input and prints `max_abs`, `rel_L2`, `corr`, the `|err|` percentiles the layer-3 gate is read off, and the stepwise (one token at a time) repeat that pins batched == stepped. The Linux reading on the `-M` container is `max_abs` 0.4473, `rel_L2` 0.1282, `corr` 0.99179, 2026-09-18. Until issue #69 that day this mode returned an IDENTICALLY ZERO output, because handing `mixed` over from the host skips `hc_run` and with it the fused `mix_streams_q` cascade into `xq_m` that the FP4 v and indexer projections read; the mode now says so out loud if it ever happens again, and `decode selftest` fails such a check before it consults the gate.
- `decode selftest` is the same comparison against the golden set the QUANT PACKAGE ships (F5, issue #64, 2026-09-18): the directory is an argument and both model paths come from `CROW_CNQ` / `CROW_HOTSETS`, so the mode runs from any working directory and reads no `oracle/golden/`, no `models/` and no oracle venv. `<golden_dir>/manifest.json` names the checks, their shapes and their gates; the mode prints one `max_abs` line per layer, then `PASS n of n checks`, and its EXIT CODE is the verdict (0 or 1) — the only mode of this bin whose exit code says anything. The golden set of record is `../selftest/`, and one line per run names whether a `models/` directory sat beside it. `../tools/selftest.sh` is the wrapper that adds the `test ! -d models` control and the `sha256sum -c` of the golden set; the Linux readings on the `-M` container are `max_abs` 9.184837e-2 against the 0.125 gate for the whole of layer 0 and `max_abs` 4.473233e-1 against the 0.625 gate for the layer-3 attention sub-block, PASS 2 of 2, 2026-09-18 (the second check is issue #69 of that day). An engine output that is identically zero fails as `output is identically zero` whatever the gate says.

## parity

```
parity <run> <prompts.json> [llama-url]
parity <phase> <run_index> <crow|llama> <prompts.json> [llama-url] [outprefix]
```

- `run` is interleaved and needs both engines resident at once.
- Both python children of the oracle venv (`tools/tokenize_ids.py --chat` per task, `tools/detokenize_ids.py` per phase) go through one bounded retry since issue #65 (2026-09-18): three attempts with a 2 s and then a 5 s pause, so a single child that dies costs the TASK a second attempt instead of costing the ten-task phase a re-run (~15 min). It stays fail-closed — after the third attempt the phase fails and records nothing for that task — and a retry that succeeded lands in the record as `oracle_retries` on that task's row, with the attempt count, the exit code and the captured stderr. The diagnosis leads with the EXIT CODE, because the three occurrences of #65 had an empty stderr and the old call site printed stderr only (`../docs/architecture.md` 8.9).
- `phase` runs one arm over the full rotated order; that is the form this machine uses, because of its RAM and VRAM budget.
- Example of record: `parity phase 0 crow decode_out/ten-tasks.json e3`.

## Parity gate

- On Linux the whole gate is one script: `../tools/gate-linux.sh [outdir]` from the repository root. It runs the three parity forms, `decode run` over 32 ids, `cargo test --release`, clippy and the doc guards against the values of record, prints GREEN or RED per item and exits non-zero on any RED. Nine items; all nine green at commit `8ff2055` on 2026-09-17. Every expected value carries its provenance in the script header. The steps below are the same gate by hand, and the form Windows uses.

- The three opt-in decode levers of 2026-09-18 are DEFAULT OFF and each was proven under this gate with its flag ON, against the same three values: `CROW_ATTN_LUT=1` (issue #61), `CROW_STAGE_PAR=1` (issue #19) and `CROW_GDN_SPLIT_Z=1` (issue #71, the z slab out of the grouped GDN input launch). For a lever on the DECODE path the form that carries the proof is the P8 teacher-forced one: the 8- and 512-row forms are prefill-only, so they cannot see a decode-path change at all. The sparse decode regime is covered separately by `decode run` on the t1-read ids (`decode_out/71/` for #71: the sha256 `56305eee11d6` of record in 12 of 12 runs of both arms).

1. Build the candidate into its own target directory, then record `sha1sum` of both `decode` binaries (`decode.exe` on Windows).
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

| id file | logit rows | Windows verdict of record | Linux value of record | date |
|---|---|---|---|---|
| `decode_out/parity-ids.json` | 12 | identical against `d211ab52ad2b` | `bceba6ff772431de…a122a2`, 11,919,360 B, byte-identical to Windows | Windows 2026-09-10, Linux 2026-09-17 |
| `decode_out/real512-ids.json` | 516 | identical against `d211ab52ad2b` | `8387234709271515…`, 512,532,480 B | Windows 2026-09-10, Linux 2026-09-17 |
| `decode_out/real512-ids.json`, teacher-forced | 516 | not a Windows form | `3bb3e69edf90…`, 512,532,480 B, run with `CROW_GRAPH=0 CROW_PARITY_PREFILL=8` | Linux 2026-09-17 |
| `decode_out/t3-debug-1024-ids.json` | 1028 | identical against `d211ab52ad2b` (issue #33) | `117dd8d9d8dc…`, 1,021,091,840 B, at 740 tok/s cold | Windows 2026-09-10, Linux 2026-09-17 |

- The 1024-row form is the largest chunk form in production; it has been a gate since 2026-09-05 (issue #22).
- It is run with `CROW_CHUNK=1024` set, twice, the same way. It is NOT in `gate-linux.sh`, because it costs a full long-prompt run.
- Reference build of record on Windows: `d211ab52ad2b` (issue #43). The Linux values come from commit `9f12429` (8, 512, P8) and 2026-09-17 (1024); the 512-row and 1024-row bytes differ between the platforms because the NVRTC and driver JIT differ, not because of the port (`../docs/architecture.md` section 8.7).
- The teacher-forced form puts the DECODE path under the parity contract, which the other three do not.

## Logs

- Every line the engine says is a `tracing` event with a per-component target (issue #13,
  2026-09-18). Two sinks, both behind a non-blocking writer thread: `engine.log` in
  `$XDG_STATE_HOME/crow/logs` (Linux default; `%LOCALAPPDATA%\crow\logs` on Windows) with the
  machine form `<ISO-8601-UTC>  <LEVEL> <target>: <message>` (2026-09-18), and the stderr mirror,
  which prints the MESSAGE ONLY and is therefore byte-identical to the `eprintln!` lines every
  tool and chain log of this repository was written against.
- `CROW_LOG` is `RUST_LOG` syntax, default `info`. `CROW_LOG=info,chat=debug` brings back the full
  `[chat] ids [...]` list of every answer (DEBUG since #13, because at INFO it made a redirected
  log grow without bound); `CROW_LOG=info,decode=trace` turns on the per-token decode forensics of
  `gen::decode_step`; `CROW_LOG=info,routing=debug` adds one line per prefill chunk. A filter this
  build cannot parse installs the default and says so — it never silences the process.
- `CROW_LOG_DIR`, `CROW_LOG_ROTATE_MB` and `CROW_LOG_KEEP` are the file knobs (defaults `64` MiB,
  decimals accepted, and `8` kept archives, both since 2026-09-18); the file also rotates on the
  UTC day boundary and every rotated file is gzipped. All four rows are in `../docs/env.md`.
- Two structured JSON lines sit next to the human ones: one `boot` line per process with the
  operating point (context, N residency, KV dtype, kernel path, cold-path policy per layer) and
  one `routing` line per request with the counters — expert selections and how many were cold, the
  residency hit rate, layers with cold work, bytes streamed, PLE rows and fills, trickle swaps.
- Nothing here is on the numeric path: the gate's `sha256` values and its ids of record are the
  same at `info` and at `trace` (measured 2026-09-18 over five decode runs, spec 9.4).

## Machine rules

- One engine process per machine, never two GPU jobs. On Linux the engine enforces it: `.engine.lock`, plus the `/dev/nvidia-uvm` scan that drops the pinned budget when another CUDA process is alive.
- Stop a run by PID, never by process name: a name filter matches the shell that started it.
- On Windows, wait for more than 50.5 GiB free host RAM before a load (rule since 2026-09-10, issue #38). On Linux that manual rule is replaced by the derived budget of issue #15: `free_for_pin` is `MemTotal` minus what cannot be reclaimed, not `MemAvailable`, because about 45 GiB sits in the driver's pinned-page pool after an exit and is served straight back to the next allocation (measured 2026-09-17: 60.76 GiB free for pinning against `MemAvailable` 10.89 GiB).
- Keep the container off a compressed mount. The PLE section is read as 108-byte rows at random offsets on the critical path of every token; on btrfs set `chattr +m` BEFORE the file is written and check with `filefrag -v <container> | grep -c encoded` (0 is what you want), because `chattr +m` plus a defragment on an already compressed file is a no-op (measured 2026-09-17).
- A tok/s number is only quoted next to a decode run from the same session (issue #38). On Linux a throughput number also needs a defined page-cache state: the exit purge cools the container, so a cold-form reading belongs to the previous process unless a purging run precedes it (measured 2026-09-17).
- `../tools/drift-chain.sh <label> <order>` is the chain form behind that rule, and the Linux answer to issue #38: alternating `serve` and `decode run` arms over the t1-read prompt of record (16,064 ids, greedy, 256 tokens, 255 timed steps), one fresh process per run, a machine block before every start, the `routing` and boot JSON lines plus the generated-ids sha256 after every run. Measured 2026-09-18 over two chains, `S D S D S D S D` and `S S S S`: the serve within-arm spread is 1.0056 and 1.0009 and the adjacent `decode run` arm 1.0012, one ids sha256 per arm in all twelve runs — the run-position drift of that issue is absent on this box. The record, the machine blocks and a recommendation on each of the issue's consequence rules are in `../docs/measurement-coverage.md`.

## Tests

```
cd engine
cargo test --release
```

- 202 passed, 0 failed on 2026-09-18 (113 lib + 78 serve + 6 parity + 5 decode; 165 on 2026-09-17, plus the six of issue #67, the three of issue #68, the three of issue #49, the two of issue #60, the four of issue #54, the four of issue #65, the three of issue #64, the ten of issue #13 and the two of issue #69 — the shipped self-test manifest parsed into its two checks and the zero-output refusal, both in `src/bin/decode.rs`). This line said 200 until issue #71 (2026-09-18) brought it back to what the gate has enforced since #69: `../tools/gate-linux.sh` has `TESTS=202`, and the two counts are the same measurement. Also, `cargo clippy --release --all-targets` reports 1,421 warnings, counted as `grep -cE '^warning: '` — one FEWER than the 1,422 of record, because the `redundant reference in eprintln! argument` warning at `gen.rs:2890` no longer exists: that line is a `tracing` event now (issue #13, 2026-09-18). The 154 converted sites and the new `log.rs` add no warning of their own. Both counts are enforced by `../tools/gate-linux.sh`.
- The ten tokenizer tests need `../models/` and are skipped without it.

```
python3 ../tools/check_env_docs.py
python3 ../tools/check_readme_dates.py
```

- The two doc guards need no GPU and no model: `check_env_docs` reads `code 89, doc 89` and exits 0 (82 = 82 before the four `CROW_LOG*` rows of issue #13, 86 = 86 before the `CROW_ATTN_LUT` row of issue #61, 87 = 87 before the `CROW_STAGE_PAR` row of issue #19 and 88 = 88 before the `CROW_GDN_SPLIT_Z` row of issue #71, all 2026-09-18), `check_readme_dates` reports 0 offenders.
