# crow-nest

| item | value |
|---|---|
| product | inference engine for one model on one GPU, own quantization, own container, thin CUDA kernels in Rust |
| model | Qwen3.8-Flash-Next as CNQ4.5-M (NVFP4, 4.5 bpw), converted from the original safetensors |
| platform | Windows, NVIDIA Blackwell (`sm_120`), CUDA only |
| license | code Apache-2.0 (`LICENSE`); the model files carry the Qwen Community License 1.0 |

## What this is

- An engine that loads one CNQ container and serves it over HTTP to one client.
- The client is the Crow repository (`nibor1896/Crow`), which talks to `serve` the way it talks to llama-server.
- Three product binaries: `serve` (HTTP), `decode` (single run and parity dumps), `parity` (the standing ten-task harness).
- Fourteen further binaries under `engine/src/bin` are probes and gates, not product surface (`engine/README.md`).
- Non-goals from the decision record: no training, no multi-user, no multi-GPU, no arbitrary architectures, no GGUF input, no CPU-only mode.

## Platform

- v0.1.0 is Windows only.
- CUDA on Windows, MSVC toolchain, PowerShell measurement chains.
- The FP4 path needs the arch-specific target `compute_120a`; plain `sm_120` is rejected by ptxas (`docs/system-landscape.md:29-30`).
- Blackwell only: the Ampere and Ada fallback stage is not planned (issue #12).
- Linux is unverified and is issue #15; no Linux number is quoted anywhere in this repository.

## Requirements

| item | value | source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090, `sm_120`, 170 SMs (measured 2026-09-02), 32,607 MiB VRAM | `docs/system-landscape.md:12` |
| host RAM | 64 GB; a chain waits for more than 50.5 GiB free before it starts an engine (rule since 2026-09-10, issue #38) | `docs/system-landscape.md:14` |
| OS | Windows | `docs/system-landscape.md:15` |
| CUDA toolkit | CUDA 13.3 (nvcc, NVRTC, ptxas); `nvrtc64_133_0.dll` needs the toolkit bin directory on `PATH` | `docs/system-landscape.md:22` |
| Rust | Rust 1.97.0, cargo 1.97.0 | `docs/system-landscape.md:23` |
| cudarc | 0.19.9, features `cuda-13030`, `dynamic-loading`, `nvrtc` | `docs/system-landscape.md:24` |
| container | `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, 104,727,179,972 B, not in the repository, produced by `converter` | `engine/src/bin/serve.rs:446` |
| hot set | `decode_out/hotsets-M-longctx2100-n160.json`, tracked in the repository | `engine/src/bin/serve.rs:447` |

## Start

### Build

```
cd engine
cargo build --release --bin serve
```

### Run, from the repository root

```
engine/target/release/serve.exe --port 8099
```

### Check the endpoint after the load

```
curl.exe -s http://127.0.0.1:8099/health
{"status":"ok"}
```

### Ask one question, from a POSIX shell (Git Bash, WSL)

```
curl -s -X POST http://127.0.0.1:8099/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"stream":false}'
```

### Ask one question, from PowerShell

```
$body = '{"messages":[{"role":"user","content":"Say hello in five words."}],"max_tokens":32,"stream":false}'
Invoke-RestMethod -Uri http://127.0.0.1:8099/v1/chat/completions -Method Post -ContentType application/json -Body ([Text.Encoding]::UTF8.GetBytes($body)) | ConvertTo-Json -Depth 6
```

- Two blocks, because a JSON body on one command line does not survive both quoting rules.
- PowerShell 5.1 mangles `curl.exe ... -d "{\"...\"}"`; that form is not printed here.

| item | value |
|---|---|
| smoke run of the blocks above | verified 2026-09-11, RTX 5090, commit 17a76ed, both shells, 2 of 2 |
| build of `serve` | 16.5 s from an empty target directory, measured 2026-09-11 |
| load to `/health` ok | 82.6 s, measured 2026-09-11 |
| chat answer | HTTP 200, `finish_reason` `stop`, content "Hello there, my dear friend.", measured 2026-09-11 |
| ids of that answer | identical in 3 of 3 answers over 2 starts on 2026-09-11 (5 requests: 3 HTTP 200, 2 malformed curl forms rejected), greedy |

- The server binds `127.0.0.1` and defaults to port 8099 (`engine/src/bin/serve.rs:445`).
- It must be started from the repository root: container and hot-set paths are repository relative.
- One engine per machine: `Engine::load` takes `engine/.engine.lock`, a second `serve` exits non zero.
- It is blocking: one request at a time, a second connection waits in the accept queue, no `503`.

## What is measured

- The two arms run different weights: crow-nest CNQ4.5-M (NVFP4, 4.5 bpw); llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL (GGUF, 2.4 bpw). Every comparison names both.

| metric | crow-nest | llama.cpp | delta | machine | date | source |
|---|---|---|---|---|---|---|
| ten-task quality, greedy | 2 Pass / 5 Partial / 3 Fail of 10 | 2 Pass / 6 Partial / 2 Fail of 10 | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| decode, engine arm, ten tasks | 35.5 to 46.8 tok/s | 44.4 to 48.2 tok/s | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| prefill, engine arm, ten tasks | 110 to 706 tok/s | 265 to 846 tok/s | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| decode through `serve`, before the hot-set tick | 23.13 to 25.57 tok/s against 33.06 to 34.50 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| decode through `serve`, with the hot-set tick | 26.43 to 26.77 tok/s against 32.82 to 32.98 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| decode through `serve`, tick plus the `CROW_ADAPT_WINDOW` default | 31.97 to 32.32 tok/s against 32.71 to 33.09 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| run-to-run drift of a `serve` rate | 26 % (run 1 30.45, run 6 22.42 tok/s, identical ids) | n/a | n/a | RTX 5090 | 2026-09-10 | issue #38 |
| cold prefill of a 16,064 id prompt on `serve` | 21.6 to 22.0 s | n/a | n/a | RTX 5090 | 2026-09-10 | `docs/architecture.md:426` |
| warm turn after the prefix cache | 404 ms for 95 of 16,159 ids, 99.41 % of the prompt reused | n/a | n/a | RTX 5090 | 2026-09-10 | issue #31, `docs/architecture.md:427` |
| ten-task quality, sampled, six seeds | gate met in 1 of 6 seeds; 4 Pass / 37 Partial / 19 Fail of 60 answers | n/a | n/a | RTX 5090 | 2026-09-10 | issue #44 |
| parity, 8 and 512 logit rows | byte-identical against the installed build `d211ab52ad2b` | n/a | n/a | RTX 5090 | 2026-09-10 | issue #43 |

- Delta is n/a on every row: each value above is a range or a Pass/Partial/Fail distribution, not a single point number with a sourced ratio, so no ratio is computed here.
- A `serve` rate is still not an engine rate: `serve` ticks the stream trickle since 2026-09-11 (issue #37), and the two rows above are the same six-run form before and after that change.
- The tick's ranking signal is `CROW_ADAPT_WINDOW`, and `serve` sets it to `1` when unset since 2026-09-11 (issue #37, `engine/src/bin/serve.rs:2313`); `CROW_ADAPT_WINDOW=0` restores the cumulative ranking.
- With that default the gap to the adjacent `decode run` is -3.39 %, -1.87 % and -2.03 % over three pairs, and `serve` moves 140.0 cold experts per token against 137.3 (issue #37, measured 2026-09-11).
- Identity holds across all three chains: 18 of 18 runs produce the same 255 ids (issue #37, measured 2026-09-11).
- No tok/s number is quoted without an adjacent decode run in the same session (rule from issue #38, 2026-09-10); the #38 row above is the serve-to-serve drift pair itself, not a rate claim.
- The sampler default is decided (issue #55, 2026-09-11): the request decides, no `temperature` is greedy, `temperature > 0` samples with the data-sheet defaults (`engine/src/bin/serve.rs:1041`); the six-seed and the greedy rows above are the measured basis.

### Targets

- Targets are goals the engine aims at, never pass or fail gates (`docs/architecture.md:18-19`, `:57-58`).

| target | value | measurement point | set |
|---|---|---|---|
| context floor | at least 200,000 tokens, configuration ceiling 262,144 with FP8-KV | loader refuses a smaller configuration | 2026-09-01 |
| decode | at least 42 tok/s | batch 1, session filled to the 200k floor, after a discarded warm-up | 2026-09-02 |
| prefill | at least 972 tok/s | the 32k operating point, completed-prompt average | 2026-09-02 |
| latency | minimal TTFT and inter-token latency | measured per task, not yet a gate | 2026-09-01 |

## Repository layout

| path | content |
|---|---|
| `engine/` | the engine crate, the CUDA kernels and the 17 binaries counted 2026-09-11 (`engine/README.md`) |
| `converter/` | the streaming safetensors to CNQ quantizer (`converter/README.md`) |
| `docs/` | `architecture.md` (the spec, sections 0 to 7), `system-landscape.md`, `env.md`, the ten-task material |
| `tools/` | Python guards and harness helpers, no GPU needed |
| `decode_out/` | gate inputs only; measurement records are ignored (`.gitignore`) |
| `oracle/` | the layer-wise reference against the unquantized originals |
| `probes/` | the hardware probes the plan stands on |
| `models/`, `converter/*.cnq` | model weights and containers, ignored, never committed |

## Environment

- Every `CROW_*` variable has a row in `docs/env.md` with its `file:line`.
- The README repeats none of them: `docs/env.md` is the single list.
- `tools/check_env_docs.py` fails when a variable exists in the sources without a row, or a row without a variable.
- `serve` reads none of the sampler variables; it builds the sampler from the request (`engine/src/bin/serve.rs:196-197`).

## Quality gates

| gate | rule | command |
|---|---|---|
| parity, short and long form | every logit row byte-identical against the reference build, run twice, last green 2026-09-10 | `engine/README.md`, section "Parity gate" |
| parity, 1024 rows | the largest chunk form in production is the one gated (since 2026-09-05, issue #22) | `CROW_CHUNK=1024 decode parity decode_out/t3-debug-1024-ids.json <dir>` |
| layercheck | `max_abs` at or below 0.125 on the layer 0 golden (gate hard since 2026-09-04) | `decode layercheck` |
| ten-task quality | no degeneration, Pass at least the reference minus one, Fail at most the reference | `parity phase 0 <crow or llama> decode_out/ten-tasks.json <outprefix>` |
| env documentation | code set equals doc set | `python tools/check_env_docs.py` |
| oracle transport | the harness sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` on the Python oracle process, not the shell, since 2026-09-11 (issue #34) | `parity phase 0 crow decode_out/c1-tasks/t4-prose.json <prefix>` in a shell without both variables |
| record header | a `parity` record names the sampler that produced it: greedy says greedy, a sampled run carries the profile and the seed, since 2026-09-11 (issue #53) | `meta.operating_point` of the written record |
| README numbers | no number without a date, a unit or an identifier | `python tools/check_readme_dates.py` |
| CI, GitHub Actions | four jobs on windows-latest: build, test, clippy (non-blocking), doc guards; engine tests 72 of 80 lib and 55 of 57 serve, 10 tokenizer tests skipped for want of `../models/`, measured 2026-09-11 | `.github/workflows/ci.yml` |

## License

| what | license | where |
|---|---|---|
| the code in this repository | Apache License 2.0 | `LICENSE` |
| the model weights and every container derived from them | Qwen Community License 1.0 | with the model files under `models/`, and in the model card |

- Apache-2.0 was chosen over MIT because this repository ships its own CUDA kernels and its own container format, and Apache-2.0 carries an express patent grant.

## Status

| item | value |
|---|---|
| version | v0.1.0, tagged 2026-09-11 on `592d05d` (`git ls-remote --tags origin`); the perf stage after the tag (issue #1, 2026-09-11) is unreleased work on `release-v0.1` |
| history | one branch `release-v0.1`, pushed to `origin` (`github.com/nibor1896/crow-nest`, private) with tag `v0.1.0` = `592d05d`, 2026-09-11 |
| scope | one model, one GPU, one client, Windows |
| open, throughput | prefill gap to the target, issue #10 |
| `serve` rate | within 5 % of the adjacent `decode run` since 2026-09-11, three pairs, issue #37 |
| open, measurement discipline | run-position drift of a `serve` rate, issue #38 |
| open, platform | Linux environment unverified, issue #15 |
| open, logging | engine logging stage not started, issue #13 |
