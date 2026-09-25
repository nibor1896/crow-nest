# Getting started

Full build, run and check instructions, moved out of the README on 2026-09-23. The README keeps the short form.

## What this is

- An engine that loads one CNQ container and serves it over HTTP to one client.
- The client is the Crow repository (`nibor1896/Crow`), which talks to `serve` the way it talks to llama-server.
- Three product binaries: `serve` (HTTP), `decode` (single run and parity dumps), `parity` (the standing ten-task harness).
- The engine sees (#VIT, 2026-09-14): Crow's `/image`, drag-and-drop and `read_image` work against `serve` — the container's visual tower loads by default, the f32 tower matches the oracle at cos 1.000000, and the text path stays byte-identical (`decode_out/srv-vit.log`). That verification ran ONE image per process, which is every image it could see: from the second image of a process on, the tower's result was read back before it was finished and the model answered for the PREVIOUS image (issue #73, found by robin and fixed 2026-09-18, commit `bc9cd9b`). Multi-image requests and second-image requests are correct since that commit; the end-to-end guards are `tools/vit-colorprobe.py`, `tools/vit-lag.py` and `tools/vit-imgprobe.py`, run by hand — no image runs inside `tools/gate-linux.sh` (2026-09-18). Since v0.5.0 (2026-09-24) the tower runs llama.cpp's F16 projector `mmproj-F16.gguf` when `CROW_VIT_MMPROJ` finds one (#108, +611 MiB of VRAM; the container's NVFP4 section otherwise), every image gets 1,024 to 1,280 visual tokens (#107, `CROW_VIT_MIN_TOKENS` / `CROW_VIT_MAX_TOKENS`), and a different image of the same size no longer reuses the previous image's cached state (#114).
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
| hot set | `decode_out/hotsets-M-crow0924-n160.json`, tracked in the repository (since 2026-09-24, [hot-set calibration](hotset-calibration.md)) | `engine/src/geo.rs` `DEFAULT_HOTSETS` |

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
  passed through, except secrets (`CROW_*KEY`, `CROW_*TOKEN`, `CROW_*SECRET` belong to the client
  and are dropped). `docs/env.md` has the host-memory rows the launcher bounds:
  `CROW_RAM_MARGIN_GB`, `CROW_PINNED_BUDGET_GB`, `CROW_PINNED_ALLOC`.
- The launcher's operating point is the BARE container since 2026-09-23 (`0254ed6`): no overlay,
  the engine's default pinned budget and allocation (`register` on Linux, issue #103). An overlay
  is opt-in with `CROW_CNQ_OVERLAY=<file>`; `CROW_CNQ_OVERLAY=none` and an empty value mean none.
  The dense BF16 overlay was the launcher default for five hours that day (`0924406`) and was
  reverted: at pinned 50 GiB write-combined robin's `serve` was OOM-killed (2026-09-23, 18:38 CEST),
  and the overlay shrank the hot set to 128 slots instead of 156 in that run (2026-09-23).
  `CROW_KV=bf16` is read by `serve` since issue #102 and needs a larger `CROW_PINNED_BUDGET_GB` (`docs/env.md`).
- **Keep the container off a compressed mount** (2026-09-17). The PLE section is read as rows of
  160 values (up to four 36-byte NVFP4 blocks of the flat stream since `85a48e7`, 2026-09-23) at random offsets, on the critical path of every token, so a filesystem that decompresses a
  whole extent per 4 KiB read is the wrong home for it. On btrfs, check with
  `filefrag -v <container> | grep -c encoded` (0 is what you want) and give the file `chattr +m`
  BEFORE it is written — `chattr +m` plus `btrfs filesystem defragment` on an already compressed
  file is a no-op, only a full rewrite clears the extents. Same rule for a `compress-force` mount
  or a compressed ZFS dataset.

### Check the branch on Linux, from the repository root

```
tools/gate-linux.sh
```

- It runs the three parity forms, the short generated-id run, the tests, clippy and the three doc
  guards against the Linux values of record, prints GREEN or RED per item and exits non-zero on
  any RED. Every expected value carries its provenance in the script header.
- All nine items green at commit 8ff2055 on 2026-09-17, with `cargo test` at 165 (`decode_out/final/GATES.md` section 2); the `487128d` follow-up re-ran build, tests and both doc guards. Run again through 2026-09-18 at eight commits of that day, all nine green each time and the three doc guards with it, the last GPU run at `bc9cd9b` (`decode_out/gate61`, `gate61g`, `gate62`, `gate69`, `gate68b`, `gate71`, `gate13-final`); the host-side counts the script pins are `TESTS=221` and `CLIPPY=1421` on 2026-09-18.
- On 2026-09-23 the pinned count is `TESTS=358` (237 lib + 110 serve + 6 parity + 5 decode, measured
  that day on `release-2026-09-23`). The parity shas and the run-32 ids of record are the values of
  the pre-2026-09-23 engine: the activation pre-scale (`488a840`) and the PLE fix (`85a48e7`) change
  the numerics on purpose, so those four items are expected RED until a GPU gate run records new
  values. That run has not happened yet.
- On 2026-09-24 (v0.5.0, `9c9fd51`) the pinned count is `TESTS=387` (259 lib + 117 serve + 6 parity + 5 decode, 3 lib
  tests ignored: GPU only). `CLIPPY=1505` is not re-counted. The four numeric items are still expected RED.
- Engine runs are sequential on purpose: before every engine start the gate asks the engine's own
  `ramcheck --need` (`free_for_pin`, issue #103, `tools/pin-room.sh`) whether the pinned tier fits.

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
- A request without `max_tokens` gets 8192 since 2026-09-18 (1024 before it, issue-less, commit `8bad310`): a `write_file` carrying a whole file ended in `finish length` before the model had written its `path` parameter. The 32768 cap and the clamp to `n_ctx` minus the prompt ids are unchanged. Crow sends `max_tokens` 16384 on every request (read 2026-09-24, #112).
- `max_completion_tokens` is the same budget since 2026-09-24 (#112); both fields with different values are a 400.
- A request without `temperature` samples at the model card row of its thinking mode (thinking 1.0 / 0.95 / 20 / 0, non-thinking 0.7 / 0.8 / 20 / 0 for `temperature` / `top_p` / `top_k` / `min_p`), never greedy, since 2026-09-24 (#111). Greedy is `temperature` 0 sent explicitly. The full table of request defaults is in `docs/env.md`.
- The request body cap is 100 MiB since 2026-09-24 (#113; 16 MiB before), llama-server's cap: a body over it is a 413.
- It must be started from the repository root: container and hot-set paths are repository relative.
- One engine per machine: `Engine::load` takes `engine/.engine.lock`, a second `serve` exits non zero.
- It is blocking: one request at a time, a second connection waits in the accept queue. The one `503` it answers is a CUDA allocation refused inside a request: the body names the allocation, its byte count and the free VRAM, the request is dropped and the engine stays up (measured 2026-09-17, issue-less, commit 8ff2055). It answered one live on 2026-09-18 — `[vit] scratch allocation refused after 23 buffer(s)`, free VRAM 35.7 MiB — which is the symptom issue #72 fixed that day; the engine stayed up through it.
- The planner HOLDS the vision path's VRAM at boot — `[budget] vit reserve 277.3 MB (tower scratch 228.5 + mrope span 48.8)` at `n_ctx` 200,000, measured 2026-09-17; 334.5 MiB since the 1,280-token cap of #107 (tower scratch 285.6 MiB, 2026-09-24) — so an image request allocates no per-request VRAM at all and the image count per request is bounded by the context, not by VRAM. Until issue #72 (fixed 2026-09-18, commit `74970b5`) the reserve was only PLANNED: the scratch stayed lazy, everything allocated after the plan spent the slack, and robin's first image request of that day found 35.7 MiB free and got the named 503. `Engine::load` allocates the tower scratch and the two mrope span tables before the budget verify now, and a `[budget] post-plan allocations held at boot` line names every one of them with its side of the bus (277.6 MB of VRAM, plus the 256.0 MB image cache that is host RAM and never on the card) against a free-VRAM floor of 0.25 GiB. Live at the full operating point on 2026-09-18: free VRAM 551 MiB after load, 544.1 MiB at both an image request and a four-image request, `engine live allocs` unchanged at 2243 across them.
- The planner also keeps a **render reserve** FREE for a co-resident GPU client, Crow's `render_page` first (issue #110, 2026-09-25): `CROW_RENDER_RESERVE_MB`, default 1024, `0` = off, one `[budget] render reserve 1024.0 MB ... costs 8.1 hot-set units` line, and the free-VRAM floor after load becomes 0.25 + 1.00 GiB. It costs about 8 hot experts per layer; the decode cost is not measured yet. With it off, serve left 73-185 MiB free and every Crow capture fell back to SwiftShader (2026-09-23/24).
