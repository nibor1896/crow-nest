# crow-nest

<a href="https://github.com/nibor1896/crow-nest/releases"><img src="https://img.shields.io/github/v/release/nibor1896/crow-nest?style=flat-square&logo=github&logoColor=ffffff&labelColor=000000" alt="release"></a>
<a href="https://github.com/nibor1896/crow-nest/actions"><img src="https://img.shields.io/github/actions/workflow/status/nibor1896/crow-nest/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=ffffff&labelColor=000000" alt="ci"></a>
<a href="engine/"><img src="https://img.shields.io/badge/Rust-1.98-555555?style=flat-square&logo=rust&logoColor=ffffff&labelColor=000000" alt="rust"></a>
<a href="docs/architecture.md"><img src="https://img.shields.io/badge/CUDA-13.3%20%C2%B7%20Blackwell%20sm__120-555555?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=000000" alt="cuda"></a>
<a href="README.md#run-on-linux-from-the-repository-root"><img src="https://img.shields.io/badge/Linux-x86__64-555555?style=flat-square&logo=linux&logoColor=ffffff&labelColor=000000" alt="linux"></a>
<a href="README.md"><img src="https://img.shields.io/badge/Windows-x64-555555?style=flat-square&labelColor=000000" alt="windows"></a>
<a href="https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M"><img src="https://img.shields.io/badge/model-Qwen3.8--Flash--Next--CNQ4.5--M-555555?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=000000" alt="model on hugging face"></a>
<a href="https://github.com/nibor1896/Crow"><img src="https://img.shields.io/badge/client-Crow-555555?style=flat-square&logo=github&logoColor=ffffff&labelColor=000000" alt="crow"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square&labelColor=000000" alt="license"></a>

| item | value |
|---|---|
| product | inference engine for one model on one GPU, own quantization, own container, thin CUDA kernels in Rust |
| model | Qwen3.8-Flash-Next as CNQ4.5-M (NVFP4, 4.5 bpw), converted from the original safetensors; the container carries the visual tower (`vit` section) and serve answers image requests (`CROW_VIT`, default on) |
| platform | Linux and Windows, NVIDIA Blackwell (`sm_120`), CUDA only |
| license | code Apache-2.0 (`LICENSE`); the model files carry the Qwen Community License 1.0 |

## What this is

- An engine that loads one CNQ container and serves it over HTTP to one client.
- The client is the Crow repository (`nibor1896/Crow`), which talks to `serve` the way it talks to llama-server.
- Three product binaries: `serve` (HTTP), `decode` (single run and parity dumps), `parity` (the standing ten-task harness).
- The engine sees (#VIT, 2026-09-14): Crow's `/image`, drag-and-drop and `read_image` work against `serve` — the container's visual tower loads by default, the f32 tower matches the oracle at cos 1.000000, and the text path stays byte-identical (`decode_out/srv-vit.log`). That verification ran ONE image per process, which is every image it could see: from the second image of a process on, the tower's result was read back before it was finished and the model answered for the PREVIOUS image (issue #73, found by robin and fixed 2026-09-18, commit `bc9cd9b`). Multi-image requests and second-image requests are correct since that commit; the end-to-end guards are `tools/vit-colorprobe.py`, `tools/vit-lag.py` and `tools/vit-imgprobe.py`, run by hand — no image runs inside `tools/gate-linux.sh` (2026-09-18).
- Sixteen further binaries under `engine/src/bin` are probes and gates, not product surface (counted 2026-09-17, `engine/README.md`).
- Non-goals from the decision record: no training, no multi-user, no multi-GPU, no arbitrary architectures, no GGUF input, no CPU-only mode.

## Platform

- Linux and Windows. The Linux port landed on 2026-09-17 (issue #15) and is measured: library and every binary build, `cargo test --release` green, and the 8-row parity form byte-identical to the Windows reference.
- CUDA on Linux (stable Rust from rustup, the `libc` crate, the CUDA runtime directory on `LD_LIBRARY_PATH`, bash measurement chains) and CUDA on Windows (MSVC toolchain, PowerShell measurement chains).
- The FP4 path needs the arch-specific target `compute_120a`; plain `sm_120` is rejected by ptxas (`docs/system-landscape.md:29-30`).
- Blackwell only: the Ampere and Ada fallback stage is not planned (issue #12).
- The numeric contract is per platform, because the NVRTC and the driver JIT differ (Windows NVRTC 13.3.73 with driver 616.56, Linux NVRTC 13.3.33 with driver 610.57, both measured 2026-09-17): the 8-row parity form is byte-identical on both, and the 512-row and 1024-row forms have their own Linux values of record. The drift never flipped an argmax on the 512-row form; over a long generation it does (`docs/architecture.md` section 8.7).

## Requirements

| item | value | source |
|---|---|---|
| GPU | NVIDIA GeForce RTX 5090, `sm_120`, 170 SMs (measured 2026-09-02), 32,607 MiB VRAM | `docs/system-landscape.md:12` |
| host RAM | 64 GB on the Windows box: a chain waits for more than 50.5 GiB free before it starts an engine (the Windows gate, rule since 2026-09-10, issue #38). 62.17 GiB on the Linux box, where that gate does not apply and stays replaced by the derived pinned budget the engine prints on its `[budget]` boot line (issue #15, 2026-09-17; robin kept the replacement on 2026-09-18) | `docs/system-landscape.md:14` |
| OS | Windows, or Linux since 2026-09-17 (measured on Arch Linux, kernel 7.2.3-arch1-3) | `docs/system-landscape.md:15`, second environment block |
| CUDA toolkit | CUDA 13.3 (nvcc, NVRTC, ptxas); on Windows `nvrtc64_133_0.dll` needs the toolkit bin directory on `PATH`, on Linux the runtime directory needs to be on `LD_LIBRARY_PATH` (never the `lib/stubs` sibling) | `docs/system-landscape.md:22` |
| Rust | Rust 1.97.0, cargo 1.97.0 on the Windows box; rustc 1.98.1 from rustup stable on the Linux box | `docs/system-landscape.md:23` |
| cudarc | 0.19.9, features `cuda-13030`, `dynamic-loading`, `nvrtc` | `docs/system-landscape.md:24` |
| container | `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, 104,727,179,972 B, not in the repository, produced by `converter` | `engine/src/geo.rs:71` |
| hot set | `decode_out/hotsets-M-longctx2100-n160.json`, tracked in the repository | `engine/src/geo.rs:72` |

## Start

### Build

```
cd engine
cargo build --release --bin serve
```

- The same command builds on Linux, on stable Rust from rustup; the only platform dependency the port added is the `libc` crate (issue #15, 2026-09-17). Nothing links CUDA at build time (`cudarc` `dynamic-loading`), so a machine without a toolkit still builds.

### Run on Windows, from the repository root

```
engine/target/release/serve.exe --port 8099
```

### Run on Linux, from the repository root

```
# through the launcher: it sets LD_LIBRARY_PATH from CUDA_LIB and bounds the memory itself
tools/serve-linux.sh --port 8099

# directly, without the scope: the CUDA runtime has to be on LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib
engine/target/release/serve --port 8099
```

- The cold expert tier is pinned host memory: unevictable, unswappable, and on this box most
  of the machine. The kernel's only reclaim target left is the page cache, and `systemd-oomd`
  fires on memory pressure rather than on exhaustion, so a machine-wide spike takes the
  desktop down with the engine (issue #15).
- The launcher puts `serve` in a transient scope (`systemd-run --user --scope
  --slice=session.slice -p MemorySwapMax=0 -p MemoryHigh=<MemTotal-8G> -p
  MemoryMax=<MemTotal-6G>`), computing the limits from `/proc/meminfo` `MemTotal`, so the
  pressure is bounded to one cgroup and the desktop keeps a floor. It is the same shape Crow
  uses for `llama-server` on Linux.
- `CUDA_LIB` names the CUDA runtime directory (default `~/.local/share/crow/cuda/lib`, never
  the `lib/stubs` sibling); every `CROW_*` variable of the caller and every argument are
  passed through. `docs/env.md` has the host-memory rows the launcher bounds:
  `CROW_RAM_MARGIN_GB`, `CROW_PINNED_BUDGET_GB`.
- **Keep the container off a compressed mount** (2026-09-17). The PLE section is read as 108-byte
  rows at random offsets, on the critical path of every token, so a filesystem that decompresses a
  whole extent per 4 KiB read is the wrong home for it. On btrfs, check with
  `filefrag -v <container> | grep -c encoded` (0 is what you want) and give the file `chattr +m`
  BEFORE it is written — `chattr +m` plus `btrfs filesystem defragment` on an already compressed
  file is a no-op, only a full rewrite clears the extents. Same rule for a `compress-force` mount
  or a compressed ZFS dataset.

### Check the branch on Linux, from the repository root

```
tools/gate-linux.sh
```

- It runs the three parity forms, the short generated-id run, the tests, clippy and the two doc
  guards against the Linux values of record, prints GREEN or RED per item and exits non-zero on
  any RED. Every expected value carries its provenance in the script header.
- All nine items green at commit 8ff2055 on 2026-09-17, with `cargo test` at 165 (`decode_out/final/GATES.md` section 2); the `487128d` follow-up re-ran build, tests and both doc guards. Run again through 2026-09-18 at eight commits of that day, all nine green each time and the three doc guards with it, the last GPU run at `bc9cd9b` (`decode_out/gate61`, `gate61g`, `gate62`, `gate69`, `gate68b`, `gate71`, `gate13-final`); the host-side counts the script pins are `TESTS=221` and `CLIPPY=1421` on 2026-09-18.
- Engine runs are sequential on purpose: the RAM gate refuses a second engine while the first
  one holds the pinned tier.

### Check the endpoint after the load

```
curl -s http://127.0.0.1:8099/health        # Linux
curl.exe -s http://127.0.0.1:8099/health    # Windows
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

- The server binds `127.0.0.1` and defaults to port 8099 (`engine/src/bin/serve.rs:454`).
- A request without `max_tokens` gets 8192 since 2026-09-18 (1024 before it, issue-less, commit `8bad310`): Crow sends no budget on the local path, and a `write_file` carrying a whole file ended in `finish length` before the model had written its `path` parameter. The 32768 cap and the clamp to `n_ctx` minus the prompt ids are unchanged.
- It must be started from the repository root: container and hot-set paths are repository relative.
- One engine per machine: `Engine::load` takes `engine/.engine.lock`, a second `serve` exits non zero.
- It is blocking: one request at a time, a second connection waits in the accept queue. The one `503` it answers is a CUDA allocation refused inside a request: the body names the allocation, its byte count and the free VRAM, the request is dropped and the engine stays up (measured 2026-09-17, issue-less, commit 8ff2055). It answered one live on 2026-09-18 — `[vit] scratch allocation refused after 23 buffer(s)`, free VRAM 35.7 MiB — which is the symptom issue #72 fixed that day; the engine stayed up through it.
- The planner HOLDS the vision path's VRAM at boot — `[budget] vit reserve 277.3 MB (tower scratch 228.5 + mrope span 48.8)` at `n_ctx` 200,000, measured 2026-09-17 — so an image request allocates no per-request VRAM at all and the image count per request is bounded by the context, not by VRAM. Until issue #72 (fixed 2026-09-18, commit `74970b5`) the reserve was only PLANNED: the scratch stayed lazy, everything allocated after the plan spent the slack, and robin's first image request of that day found 35.7 MiB free and got the named 503. `Engine::load` allocates the tower scratch and the two mrope span tables before the budget verify now, and a `[budget] post-plan allocations held at boot` line names every one of them with its side of the bus (277.6 MB of VRAM, plus the 256.0 MB image cache that is host RAM and never on the card) against a free-VRAM floor of 0.25 GiB. Live at the full operating point on 2026-09-18: free VRAM 551 MiB after load, 544.1 MiB at both an image request and a four-image request, `engine live allocs` unchanged at 2243 across them.

## What is measured

- The two arms run different weights: crow-nest CNQ4.5-M (NVFP4, 4.5 bpw); llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL (GGUF, 2.4 bpw). Every comparison names both.

| metric | crow-nest | llama.cpp | delta | machine | date | source |
|---|---|---|---|---|---|---|
| ten-task quality, greedy | 2 Pass / 5 Partial / 3 Fail of 10 | 2 Pass / 6 Partial / 2 Fail of 10 | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| decode, engine arm, ten tasks | 35.5 to 46.8 tok/s | 44.4 to 48.2 tok/s | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| prefill, engine arm, ten tasks | 110 to 706 tok/s | 265 to 846 tok/s | n/a | RTX 5090 | 2026-09-10 | issue #11 (comment); `nibor1896/Crow` issue #192 (comment) |
| prefill on Linux, the value of record, t1-read 16,064 ids, `decode run` at 128 tokens, two runs with identical id traces | 16.60 s = 968 tok/s and 16.66 s = 964 tok/s | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | commit 1032bc5 (the PLE prefill floor); unchanged at 4004e66 (16.49 and 16.50 s) |
| decode on Linux, the same two runs, context 16,192 | 36.8 and 36.7 tok/s (mean 27.19 and 27.22 ms per token) | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | commit 1032bc5; 4004e66 reads 27.09 ms = 36.9 tok/s on the same prompt |
| cold prefill on Linux, the 1024-row parity form, page cache cold | 740 tok/s (739.6 and 740.6 at 1032bc5, 740.2 and 742.8 at 4004e66, 740.9 at 8ff2055), sha256 `117dd8d9d8dc` and 1,021,091,840 B on every run | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | commits 1032bc5, 4004e66, 8ff2055 |
| warm short turn through `serve` on Linux, the six-turn replay (3,296-token cached prefix, 39 to 101 new ids per turn, greedy, `CROW_CHUNK=2048`), mean over turns 1 to 6 | prefill 228.3 ms, time to first token 247.4 ms | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | commit 4004e66 (267.1 and 296.2 ms at 1032bc5) |
| prefill on Linux, the same prompt and form BEFORE the PLE prefill floor, kept as history | 25.38 s = 633 tok/s and 25.44 s = 631 tok/s | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | `decode_out/final/GATES.md` item 9, commit 0667e0b |
| decode on Linux, those same two runs, context 16,192, kept as history | 36.8 and 36.9 tok/s (mean 27.17 and 27.12 ms per token) | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | `decode_out/final/GATES.md` item 9, commit 0667e0b |
| decode on Windows, the same prompt, the `final4` t1-read record (its harness counts the first token in, so the two readings are within noise of each other) | 36.80 tok/s, prefill 492.15 tok/s | n/a | n/a | RTX 5090, Windows | 2026-09-05 | `decode_out/final4-t1-read-run0-crow.json`, commit ce65176 |
| load on Linux, the 8-row parity form, before and after the ordered cold-tier sweep of issue #15 | 44 s to 25 s | n/a | n/a | RTX 5090, Arch Linux | 2026-09-17 | commit 0c9feb5; the 1024-row form read 71 s on the pre-fix build against 23 s on HEAD, `decode_out/final/GATES.md` section 4 |
| prefill, t1-read 16,064 ids, the #10c dense variant B pair (opt-in `CROW_PF_GEMM_B=1`, no default flip; W + 3 adjacent pairs, the two clean pairs -2.270 and -2.330 s = -11.0 and -11.2 percent; the default-of-record B plateau 20.723 / 20.728 s, B1 18.491 s a documented whole-run outlier) | 18.44 s = 871 tok/s with the switch on | 20.73 s = 775 tok/s, the default of record (switch off) | -2.30 s = -11.1 percent mean of the two clean pairs | RTX 5090 | 2026-09-14 | issue #10, `decode_out/srv-10c.log` |
| decode through `serve`, before the hot-set tick | 23.13 to 25.57 tok/s against 33.06 to 34.50 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| decode through `serve`, with the hot-set tick | 26.43 to 26.77 tok/s against 32.82 to 32.98 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| decode through `serve`, tick plus the `CROW_ADAPT_WINDOW` default | 31.97 to 32.32 tok/s against 32.71 to 33.09 tok/s for `decode run`, three adjacent pairs | n/a | n/a | RTX 5090 | 2026-09-11 | issue #37 |
| run-to-run drift of a `serve` rate | 26 % (run 1 30.45, run 6 22.42 tok/s, identical ids) | n/a | n/a | RTX 5090 | 2026-09-10 | issue #38 |
| run-to-run drift of a `serve` rate on Linux, three chains, one fresh process per run, ids identical per arm | absent: serve within-arm spread 1.0056 over 4 runs (mean 49.82 tok/s) and 1.0009 over 4 consecutive fresh loads (mean 49.86), the adjacent `decode run` arm 1.0012 (mean 39.83); rerun at the #61g default the same day, serve 1.0038 over 4 runs (mean 53.32) against the adjacent `decode run` arm's 1.0010 (mean 42.56); no chain monotonic in run index | n/a | n/a | RTX 5090, Arch Linux | 2026-09-18 | issue #38, `docs/measurement-coverage.md`, `tools/drift-chain.sh`, logs under `decode_out/38/` |
| `serve` at the engine default of record, the chain form the #38 rule requires (4 counted serve runs, one fresh process each, an adjacent `decode run` arm in the same chain) | 53.32 tok/s mean, spread 1.0038, next to `decode run` 42.56 tok/s mean, spread 1.0010, ids `e7c17e064ea2` and `56305eee11d6` 4 of 4 each | n/a | the two arms are different operating points (N 149 / 142, chunk 2048 / 4096, trickle on / off), so the gap is not a delta | RTX 5090, Arch Linux | 2026-09-18 | issue #38 at HEAD `6c87054`, `decode_out/38/c3-sdsd-61g/chain.log`, `docs/architecture.md` 4.6.1 |
| cold prefill of a 16,064 id prompt on `serve` | 21.6 to 22.0 s | n/a | n/a | RTX 5090 | 2026-09-10 | `docs/architecture.md:426` |
| warm turn after the prefix cache | 404 ms for 95 of 16,159 ids, 99.41 % of the prompt reused | n/a | n/a | RTX 5090 | 2026-09-10 | issue #31, `docs/architecture.md:427` |
| ten-task quality, sampled, six seeds | gate met in 1 of 6 seeds; 4 Pass / 37 Partial / 19 Fail of 60 answers | n/a | n/a | RTX 5090 | 2026-09-10 | issue #44 |
| parity, 8 and 512 logit rows | byte-identical against the installed build `d211ab52ad2b` | n/a | n/a | RTX 5090 | 2026-09-10 | issue #43 |
| decode operating point, t1-read 16,064 ids, 255 timed steps, engine default | 23.52 ms per token = 42.5 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.06 x the llama.cpp ms per token | RTX 5090, Arch Linux | crow-nest 2026-09-18, llama.cpp 2026-09-11 | issue #61, 61g, `decode_out/61g/` and `decode_out/srv-59b.log` |
| same operating point, the previous default until the #61g `CROW_ATTN_LUT` flip (the #62e 32-row GDN slab default, measured five days earlier on a different id stream) | 22.18 ms per token = 45.1 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 0.996 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-13, llama.cpp 2026-09-11 | issue #62, `decode_out/srv-62e.log` and `decode_out/srv-59b.log` |
| same operating point, the previous default until the #62e 32-row GDN slab flip (the #19i all-fused default: QSA par + splits 8, fused hc chain, fused shared chain, grouped GDN projections, cascade) | 22.43 ms per token = 44.6 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.01 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-13, llama.cpp 2026-09-11 | issues #19 and #62, `decode_out/srv-19i.log` and `decode_out/srv-59b.log` |
| same operating point, the previous default until the #19i all-fused flip (the #19/#62 combined fusion default: QSA par + splits 8, fused hc chain, grouped GDN projections, shared chain unfused) | 22.80 ms per token = 43.9 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.02 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-13, llama.cpp 2026-09-11 | issues #19 and #62, `decode_out/srv-19g.log` and `decode_out/srv-59b.log` |
| same operating point, the previous default until the #19/#62 combined fusion flip (QSA par + splits 8, unfused hc chain, per-slab GDN projections) | 23.94 ms per token = 41.8 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.08 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-12, llama.cpp 2026-09-11 | issue #61, `decode_out/srv-61b.log` and `decode_out/srv-59b.log` |
| same operating point, the #61d splits-32 experiment, ROLLED BACK by #61e after the quality verdict | 22.85 ms per token = 43.8 tok/s | 24.13 ms per token = 41.4 tok/s with `CROW_ATTN_SPLITS=8` | 22.27 ms per token = 44.9 tok/s | RTX 5090 | crow-nest 2026-09-13, llama.cpp 2026-09-11 | issue #61, `decode_out/srv-61d.log` and `decode_out/srv-59b.log` |
| same operating point, the previous default before the QSA selection flip | 24.78 ms per token = 40.3 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.11 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-12, llama.cpp 2026-09-11 | issue #63, `decode_out/srv-63c.log` and `decode_out/srv-59b.log` |
| same operating point, the previous default before the trickle issue point moved | 26.42 ms per token = 37.9 tok/s | 22.27 ms per token = 44.9 tok/s | crow-nest 1.19 x the llama.cpp ms per token | RTX 5090 | crow-nest 2026-09-12, llama.cpp 2026-09-11 | issue #19, `decode_out/srv-19e.log` and `decode_out/srv-59b.log` |

- Delta is n/a on every row but the last seven: those values are ranges or Pass/Partial/Fail distributions, not single point numbers with a sourced ratio, so no ratio is computed for them.
- The seven decode operating point rows are the single-point pairs, so they carry a ratio; each names its own crow-nest arm and the llama.cpp arm in its own column.
- The first of them is the engine default of record (issue #61, 61g, 2026-09-18): `CROW_ATTN_LUT` — the shared e4m3 table in the split decode attention kernel — is the DEFAULT since this day, so the decode attention launch is `attn_sel_split_l` with no env set; the flip's confirmation runs (W + 3N, one fresh process per run, no adjacent B arm because the no-env arm now IS the lever) measured 23.5193 ms per token = 42.52 tok/s (N1 23.5571 / N2 23.4800 / N3 23.5207, W 23.5697) and the one `CROW_ATTN_LUT=0` fallback run reproduced the arm the flip left behind at 25.2044 = 39.68, ids `56305eee11d6` in 5 of 5 runs; through `serve` the figure of record at this default is the #38 drift-chain form of the same day (2026-09-18, four counted serve runs, one fresh process each, `decode_out/38/c3-sdsd-61g/chain.log`): **53.32 tok/s mean, within-arm spread 1.0038**, next to the adjacent `decode run` arm's **42.56 tok/s mean, spread 1.0010**, ids `e7c17e064ea2` and `56305eee11d6` 4 of 4 each — and the flip's own one-run serve A/B reads 53.31 against 49.88 tok/s with the fallback on the same ids. The lever's own adjacent-pair reading is the 61f one, -1.5866 ms = -6.28 percent in 3 of 3 pairs, and it is byte-identical by construction: `tools/gate-linux.sh` is ALL GREEN at the NEW default (`decode_out/gate61g`). This row is NOT an adjacent pair with the 22.18 row below it — that one was measured on 2026-09-13 on a different generated id stream (`5098f885ab3a` against `56305eee11d6`), and the same-day pre-flip arm of this machine reads 25.18 to 25.30 ms (issues #19, #61 and #71).
- The second of them is the previous default until the #61g flip (issue #62, 62e, 2026-09-13): the resumed #62d lever puts the three big GDN decode projections at 32 rows per block (grouped input 515 blocks, per-slab fallback 320 + 192, out projection 80; every non-GDN user keeps the 64-row geometry), on top of the all-fused default with NO new env; the resumed-chain pairs measured 22.1761 ms per token = 45.1 tok/s against the 19i-state B arm's 22.6231 = 44.2, -0.4470 ms = -1.98 percent in 3 of 3 pairs, ids `5098f885ab3a` in 7 of 7 runs incl. the warm-up; byte identity held on 9 of 9 parity forms at the 61b sha256 values of record incl. the 16,056-row PXFUSE form under the PROTOCOL v2 VRAM headroom gate (`decode_out/srv-62e.log`); this is the first no-env mean under the 22.27 llama.cpp row.
- The second of them is the previous default (issues #19 and #62, #19i, 2026-09-13; the engine default until #62e): the decode hc chain AND the shared-expert chain run fused and the GDN input projections grouped, all by default with no env, on top of the QSA par + splits-8 selection; the #19i confirmation runs (the W + 3N form, no adjacent B arm post-flip because `CROW_QFUSE=0` would also disable the NVFP4 cascade) measured 22.43 ms per token = 44.6 tok/s against the #19g combined default's 22.80 = 43.9, ids `5098f885ab3a` identical in 4 of 4 runs incl. the warm-up; the lever's own adjacent-pair reading is the #19h one, -0.2286 ms = -1.00 percent in 3 of 3 pairs (`decode_out/srv-19h.log`).
- The second of them is the previous default (issues #19 and #62, #19g, 2026-09-13; the engine default until #19i): the same selection with the hc chain fused and the GDN input projections grouped but the shared chain unfused; its pair arm measured 22.80 ms per token = 43.9 tok/s against the previous default's 23.94 = 41.8, an honest gain of -1.14 ms = -4.8 percent, ids `5098f885ab3a` identical in 7 of 7 runs of both arms; the same-chain double fallback `CROW_QFUSE=0 CROW_GDN_FUSE_IN=0` measured 26.86 ms per token, but it also disables the NVFP4 cascade (the documented `CROW_QFUSE` overload), so it is not comparable to the previous default.
- The second of them is the previous default (issue #61 61b, 2026-09-12; default from 61b until #19g): the decode QSA top-k runs as `qsa_select_par` and the decode attention runs 8 splits; `CROW_QSA_PAR=0` restores the single-block form and measured 24.87 ms per token = 40.2 tok/s in the same chain on RTX 5090, 2026-09-12.
- The second of them is the ROLLED-BACK #61d splits-32 experiment (2026-09-13): it measured 22.85 ms per token = 43.8 tok/s against 24.13 = 41.4 with `CROW_ATTN_SPLITS=8` in the same chain, but the ten-task quality bar was NOT held at 32 (0/7/3 against the record 2/5/3), so the default went back to 8 under the improvement-loop quality rule; `16` and `32` stay measurement only.
- The third of them was the engine default from 2026-09-12 (issue #63) until the QSA flip: the stream trickle's copies are issued after the graph launch; `CROW_TRICKLE_DEFER=0` restores the previous order and measured 26.41 ms per token = 37.9 tok/s in its chain on RTX 5090, 2026-09-12.
- The staging kernel `stage_cold_ca` is the engine default since 2026-09-12 (issue #19) and is the kernel of both rows; `CROW_STAGE_KERNEL=1` restores the previous kernel and measured 29.68 ms per token = 33.7 tok/s on RTX 5090, 2026-09-11.
- A `serve` rate is still not an engine rate: `serve` ticks the stream trickle since 2026-09-11 (issue #37), and the two rows above are the same six-run form before and after that change.
- The tick's ranking signal is `CROW_ADAPT_WINDOW`, and `serve` sets it to `1` when unset since 2026-09-11 (issue #37, `engine/src/bin/serve.rs:2313`); `CROW_ADAPT_WINDOW=0` restores the cumulative ranking.
- With that default the gap to the adjacent `decode run` is -3.39 %, -1.87 % and -2.03 % over three pairs, and `serve` moves 140.0 cold experts per token against 137.3 (issue #37, measured 2026-09-11).
- Identity holds across all three chains: 18 of 18 runs produce the same 255 ids (issue #37, measured 2026-09-11).
- A `serve` tok/s is quoted only next to an adjacent `decode run` measured in the same chain (rule from issue #38, 2026-09-10; kept on Linux too, robin 2026-09-18); the #38 rows above are the serve-to-serve drift pairs themselves, not rate claims. The reason since 2026-09-18 is the operating point and no longer the drift, which the Linux chains found absent: `serve` runs N 149 hot experts per layer, prompt chunk 2048 and the stream trickle ticking, `decode run` N 142, chunk 4096 and no tick, so a lone serve rate invites a comparison the configuration does not support.
- A `serve` decode number enters `docs/architecture.md` on Linux only in the drift-chain form (issue #38, relaxed by robin 2026-09-18): at least 3 counted serve runs in one chain, one fresh process per run, the generated-ids sha256 identical across them, an adjacent `decode run` arm in the same chain, and the figure quoted as its arm mean with its max-over-min spread beside that decode arm's mean and spread. Anything else is refused; on Windows the older bar stands until the M2a form is rerun there (2026-09-10).
- The sampler default is decided (issue #55, 2026-09-11): the request decides, no `temperature` is greedy, `temperature > 0` samples with the data-sheet defaults (`engine/src/bin/serve.rs:1041`); the six-seed and the greedy rows above are the measured basis.

### Targets

- Targets are goals the engine aims at, never pass or fail gates (`docs/architecture.md:18-19`, `:57-58`).
- Measured against the prefill target on Linux, 2026-09-17: 968 tok/s on the 16,064 id t1-read form (`decode run`, commit 1032bc5). The target names the 32k operating point, which has no Linux reading, so this is not the target met.
- Measured against the decode target on Linux, 2026-09-18: 42.5 tok/s (23.52 ms per token) at context 16,320 on the same form, at the `CROW_ATTN_LUT` default of issue #61 (61g, commit `6c87054`). The target names a session filled to the 200,000 token floor, which has no Linux reading, so this is not the target met either.

| target | value | measurement point | set |
|---|---|---|---|
| context floor | at least 200,000 tokens, configuration ceiling 262,144 with FP8-KV | loader refuses a smaller configuration | 2026-09-01 |
| decode | at least 42 tok/s | batch 1, session filled to the 200k floor, after a discarded warm-up | 2026-09-02 |
| prefill | at least 972 tok/s | the 32k operating point, completed-prompt average | 2026-09-02 |
| latency | minimal TTFT and inter-token latency | measured per task, not yet a gate | 2026-09-01 |

## Repository layout

| path | content |
|---|---|
| `engine/` | the engine crate, the CUDA kernels and the 19 binaries counted 2026-09-17 (`engine/README.md`) |
| `converter/` | the streaming safetensors to CNQ quantizer (`converter/README.md`) |
| `docs/` | `architecture.md` (the spec, sections 0 to 9, the code map added 2026-09-17, the package self-test 8.10 and the logging section 9 added 2026-09-18), `model-card.md` (the Hugging Face model card of record, tracked here since 2026-09-18 and uploaded verbatim as that repo's `README.md`), `system-landscape.md`, `env.md`, `diagrams.md`, `long-context-goalmode.md` (the measurement record of the 170k goal-mode session, 2026-09-18), `measurement-coverage.md` (what is measured and what is not, with the run-position drift chains of 2026-09-18), `cuda-rust-evaluation.md` (the CUDA Rust / cuTile evaluation and its pilot, 2026-09-17), `quality-probe.md` (what the answer-quality probe measures, both engines' baseline of 2026-09-18, the noise floor and what it cannot decide, issue #75), the ten-task material |
| `tools/` | Python guards and harness helpers, no GPU needed: the three doc guards (`check_env_docs.py`, `check_readme_dates.py`, `check_model_card_dates.py`, the last tracked since 2026-09-18), the launcher and the gates (`serve-linux.sh`, `gate-linux.sh`, `selftest.sh`, `drift-chain.sh`), the replays (`replay-toolcalls.py`, `replay-session.py`) and the probes (`longctx-gate.py`, `quality-probe.py` with its prompt set and its unit tests since 2026-09-18, `vit-colorprobe.py`, `vit-lag.py`, `vit-imgprobe.py`). The gate, the self-test wrapper, the drift chain and the session replay all start an engine; nothing else here does |
| `decode_out/` | gate inputs and the four quality-probe baseline runs of 2026-09-18 (`quality-probe/`, issue #75); every other measurement record is ignored (`.gitignore`) |
| `oracle/` | the layer-wise reference against the unquantized originals; its output (`oracle/golden/`) is ignored |
| `selftest/` | the package self-test's golden set, tracked: the layer-0 pair (two `f32` arrays of 327,680 B each, F5, issue #64, 2026-09-18), the layer-3 attention pair (81,920 B each, issue #69, 2026-09-18) and the manifest that gates them, 825,424 B in all. These are the files the quant package ships beside the container, and `tools/selftest.sh` is what runs them |
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
| layercheck | `max_abs` at or below 0.125 on the layer 0 golden (gate hard since 2026-09-04); the Linux reading is 9.184837e-2 on the `-M` container, 2026-09-18 | `decode layercheck` |
| layercheck3 | `max_abs` at or below 0.625 and `corr` above 0.99 on the layer-3 attention golden (gate set 2026-09-18, issue #69); the Linux reading is 0.4473 at `corr` 0.99179 on the `-M` container, 2026-09-18 | `decode layercheck3` |
| package self-test, no originals | the two goldens above and their two bounds, from a package directory that holds no `models/` — the check a downloader can run. `PASS 2 of 2`, `ALL GREEN`, from a hard-linked package copy outside the repository on 2026-09-18 at `max_abs` 9.184837e-2 / `rel_L2` 1.3671e-2 for layer 0 and 4.473233e-1 / 1.2821e-1 for the layer-3 attention sub-block, NaN 0; an output that is identically zero fails whatever the gate says; the `test ! -d models` control refused the same copy the moment a `models/` directory existed, before the engine was started (issues #64 and #69) | `tools/selftest.sh <package-dir>` |
| ten-task quality | no degeneration, Pass at least the reference minus one, Fail at most the reference | `parity phase 0 <crow or llama> decode_out/ten-tasks.json <outprefix>` |
| env documentation | code set equals doc set | `python tools/check_env_docs.py` |
| oracle transport | the harness sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` on the Python oracle process, not the shell, since 2026-09-11 (issue #34) | `parity phase 0 crow decode_out/c1-tasks/t4-prose.json <prefix>` in a shell without both variables |
| record header | a `parity` record names the sampler that produced it: greedy says greedy, a sampled run carries the profile and the seed, since 2026-09-11 (issue #53) | `meta.operating_point` of the written record |
| README numbers | no number without a date, a unit or an identifier | `python tools/check_readme_dates.py` |
| model card numbers | the same rule on the Hugging Face card, with the YAML frontmatter skipped: 173 number lines over `docs/model-card.md` and the local `hf-package/README.md` when it exists — 22 in code blocks, 44 in the frontmatter, 110 dated, 63 exempt, 0 offenders on 2026-09-18, negative control exits 1. The guard was written in F4 and never committed, so every Linux gate run printed it as skipped until 2026-09-18 | `python tools/check_model_card_dates.py` |
| CI, GitHub Actions | four jobs on ubuntu-latest: build, test, clippy (non-blocking), doc guards. The runner moved from windows-latest with the Linux port on 2026-09-17; the counts of record are still the local Windows proof of 2026-09-11 (engine tests 72 of 80 lib and 55 of 57 serve, 10 tokenizer tests skipped for want of `../models/`), because no run of this workflow is recorded in this repository yet | `.github/workflows/ci.yml` |
| Linux parity gate | the three parity forms, `decode run 32`, tests, clippy and the three doc guards against the Linux values of record; GREEN or RED per item, non-zero exit on any RED; all nine items green at commit 8ff2055 on 2026-09-17 (tests 165, clippy 1422), and green at eight commits of 2026-09-18, the last of them `bc9cd9b`. The counts it pins at this release are `TESTS=221` (128 lib + 82 serve + 6 parity + 5 decode) and `CLIPPY=1421`, both measured 2026-09-18 | `tools/gate-linux.sh` |
| measurement chain, `serve` against `decode run` | the issue #38 form: arms alternating as adjacent pairs, one fresh process per run, a machine block before every engine start, the generated-ids sha256 per run as the stop condition, and the within-arm max-over-min spread as the figure of record. It is also the form a `serve` rate must carry to enter `docs/architecture.md` on Linux (issue #38, robin 2026-09-18). Three chains on 2026-09-18: serve 1.0056, 1.0009 and 1.0038, `decode run` 1.0012 and 1.0010 | `tools/drift-chain.sh <label> SDSDSDSD` |
| tool-call session, against a running `serve` | a Crow-shaped tool loop: the engine's own streamed `tool_calls` are fed back as the history, verbatim, so a turn that poisons the history shows up as the 400 it caused; exit non-zero when any round is refused (TASK J, 2026-09-17). `--think` is the reasoning-filter form of issue #67 and `--gone-client` the disconnect form of issue #54, both 2026-09-18 | `tools/replay-toolcalls.py --poison --refusals` |
| long-context quality, against a running `serve` | the first long-context reading this engine has: five questions whose answers are checkable against the stored session, scored Pass or Fail, at the prompt length the session actually reached. 5 of 5 Pass, greedy, 2026-09-18 (issue #68). It scores ANSWERS, not bytes: a quality gate is not a parity gate | `tools/longctx-gate.py` |

## License

| what | license | where |
|---|---|---|
| the code in this repository | Apache License 2.0 | `LICENSE` |
| the model weights and every container derived from them | Qwen Community License 1.0 | with the model files under `models/`, and in the model card |

- Apache-2.0 was chosen over MIT because this repository ships its own CUDA kernels and its own container format, and Apache-2.0 carries an express patent grant.

## Status

| item | value |
|---|---|
| version | v0.3.1, this release, on `8bad310`, 2026-09-18. The tags before it: `v0.1.0` = `592d05d` (2026-09-11), `v0.2.0` = `42e2b67` and `v0.2.1` = `61b8dc6` (both 2026-09-14), `v0.3.0` = `b0102c0` (2026-09-17). The crate version field in `engine/Cargo.toml` stays `0.1.0` and has never been bumped: `CHANGELOG.md` is the release record, not the manifest |
| history | branch `main`, pushed to `origin` (`github.com/nibor1896/crow-nest`, private); the day's twenty-six commits `667b68b` to `8bad310` all landed 2026-09-18, and the fourteen before them `9f12429` to `487128d` on 2026-09-17. `release-v0.1` and `linux-refactor` stay on the remote as history |
| scope | one model, one GPU, one client; Linux and Windows |
| open, throughput | prefill gap to the target, issue #10; its second probe is measured and still opt-in (2026-09-18, RED greedy on the degeneration clause, GREEN sampled, prefill −1.750 s = −10.81 percent on the 16k form). What remains of a warm turn is PCIe staging of the cold experts the chunk routes to: 9,431 MB and 192.8 ms of a 256.4 ms chunk, measured 2026-09-17; the decode side of that row is issue #19, measured 2026-09-18 at 587.8 MB per decode token over a link running at 51.1 of the 51.6 GB/s this box can reach |
| open, the decode step | the three rows of profile #59 are decomposed per kernel on Linux (2026-09-18): attention 3.595 ms per decode token (issue #61), GDN 1.954 ms (issue #62) and cold-expert staging 11.468 ms (issue #19), of a 30.05 ms profiled step. One lever of each is in the tree, `CROW_ATTN_LUT` as the default since that day; the levers not taken are written down with their numbers in `docs/architecture.md` 4.6 to 4.8 |
| `serve` rate | within 5 % of the adjacent `decode run` since 2026-09-11, three pairs, issue #37; the form of record on Linux is the drift chain of issue #38 (2026-09-18) |
| closed, measurement discipline | issue #38: the run-position drift of a `serve` rate is ABSENT on this box (three chains, 2026-09-18, spread 1.0056, 1.0009 and 1.0038), and robin decided the issue's three consequence rules the same day — the Windows 50.5 GiB gate stays a Windows rule, the adjacent-`decode run` rule is kept for the operating point and not for the drift, and a `serve` number enters the spec on Linux in the drift-chain form. The ticket stays open until the M2a form is rerun on Windows |
| platform, issue #15 | the port, the host-memory fix and the Linux values of record landed 2026-09-17; what is still owed is one reference ten-task run on the pre-refactor build after a reboot (`decode_out/final/GATES.md` section 6) |
| closed, the `</think>` filter | issue #67, fixed 2026-09-18: `serve` strips a leading `<think>...</think>` block and every bare `</think>` from the streamed content (the block leaves as `reasoning_content`), and the normaliser strips a stored one out of the history before the render; the generated ids are untouched |
| open, long-context goal mode | a 273-turn goal-mode session at 178,779 of 200,000 tokens degenerates: the tag on 67 turns, then the model echoes the client's goal nudge, then a single repeated token; the engine errored on none of its 318 completions, issue #68 (2026-09-17), artefacts under `decode_out/sessions/2026-09-17-goalmode/`. Measured 2026-09-18 (`docs/long-context-goalmode.md`): the sampler is not the cause (the presence penalty covers this answer's generated tokens only, per request) and neither is the tag — the replayed session reproduces the echo and the single-token answer with the filter active, and greedy is the worst arm; the same history at 120,924 ids answers normally under every row, and the greedy flip sits between 158,639 and 163,401 ids after the length-matched control of the same day narrowed it. The 105 identical goal nudges are not the cause either: de-duplicated, the same history is 160,589 ids and degenerates on none of three runs, while at matched length WITH the nudges kept it degenerates on none either — what they contribute is LENGTH, about 84 ids each and 4 percent of the budget (2026-09-18). The engine now REPORTS the shape no single request can see: `repeat_of`, `repeat_run` and `single_token` on the `routing` line, one WARN at three identical answers in a row, no sampling change. The quality gate `tools/longctx-gate.py` is the engine's first long-context reading, 5 of 5 Pass greedy on 2026-09-18 |
| open, measurement | the job-ring round trip of `docs/measurement-handoff.md` is still a WDDM number and owes its Linux retest (2026-09-17) |
| logging | issue #13, landed 2026-09-18: every line the engine says is a `tracing` event with a per-component target, mirrored to stderr byte for byte and written to one rotating gzipped file, with `CROW_LOG` setting the level per component without a rebuild and two JSON lines beside the human ones (the operating point at boot, the counters per request). The ticket stays open on one item: no Windows process has written a rotating log yet |
| closed, the ragged hot-set sidecar | issue #49, 2026-09-18: a sidecar row shorter or longer than the run's N is padded or truncated per row and NAMED in the load log, and a file that is not a hot set is refused by name, where `residency.rs` used to assert |
| closed, the `parity` record header | issue #60, 2026-09-18: a record names the sampler of the arm that produced it, so the llama arm stays greedy under `CROW_SAMPLE`, and `plecheck` and `states` read `CROW_CNQ` like every other bin that opens the container |
| closed, the gone client | issue #54, 2026-09-18: a `stream:false` generation ends when its client is gone — one `poll(POLLRDHUP)` between two decode steps, 0.10 us per probe, nothing on the wire; a client that only half-closed still gets its whole document |
| closed, the quant package | issue #64, 2026-09-18: the package verifies itself with the engine alone. `selftest/` ships the goldens (825,424 B beside a 105 GB container), `decode selftest` makes its exit code the verdict, and `tools/selftest.sh` refuses to run in a tree that HAS the originals |
| closed, the all-zero `layercheck3` | issue #69, 2026-09-18: the layer-3 sub-block check drove a dead path and reported the golden's own numbers back as a measurement; it runs the live attention path again (`max_abs` 0.4473, `corr` 0.99179), the package self-test gained that second check, and an identically zero output is a named FAIL before any gate |
| closed, the GDN input launch | issue #71, 2026-09-18: `CROW_GDN_SPLIT_Z` takes the z slab out of the grouped GDN input launch, bit-identical, −0.19 ms per decode token. DEFAULT OFF, opt-in |
| closed, the vit reserve | issue #72, 2026-09-18: the 277.3 MB reserve is HELD at boot instead of planned and then spent, with a `[budget]` ledger of every post-plan allocation and a free-VRAM floor of 0.25 GiB. An image request allocates no VRAM at all — free VRAM 544.1 MiB at both image requests of the live check |
| closed, the visual path | issue #73, 2026-09-18: every image but the first of a process was read as the PREVIOUS one, because the tower's result was copied back before the work was done. `Vit::run` synchronizes now; text ids untouched |
| open, the harness's oracle children | issue #65: a child that dies once costs the TASK one more attempt since 2026-09-18, not the ten-task phase a re-run, and the diagnosis leads with the exit code. WHICH death the three occurrences of 2026-09-13 died is not recoverable from what was recorded |
| open, the living diagrams | issue #14: the eight diagrams render `8bad310` and each says what it was read against (2026-09-18). It is a standing duty, not a stage |
