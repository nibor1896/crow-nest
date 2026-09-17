# crow-nest — system landscape and environment (2026-09-02)

Every measurement in this repo refers to this page. A number without this environment is
not comparable — methodology #159 (interleaved, same session, one variable, resolution
stated) starts here. Owned by issue crow-nest #4; the spec references this page from
section 0.

## Machine

| component | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 — sm_120 (compute capability 12.0), 170 SMs, 32,607 MiB VRAM |
| driver | 616.56 (KMD), UMD CUDA 13.4 |
| host RAM | 64 GB, dual-channel desktop platform |
| OS | Windows — this is the Windows box; the Linux box has its own block below (2026-09-17) |
| storage | NVMe (carries the PLE window and the cold-expert tier, decision 2026-09-02) |

## Toolchain (verified by the probes, 2026-09-01)

| component | value |
|---|---|
| CUDA toolkit | 13.3, V13.3.73 (nvcc, NVRTC, ptxas) |
| Rust | 1.97.0 (2026-06-30), cargo 1.97.0 |
| cudarc | 0.19.9, features `cuda-13030` + `dynamic-loading` + `nvrtc` (repo now `chelsea0x3b/cudarc`) |
| runtime DLLs | `nvrtc64_133_0.dll` needs the toolkit bin dir on PATH; `nvcuda.dll` comes from the driver (System32) |

## Hardware facts pinned by probe 2 (`dev/crow-nest/probes/RESULTS.md`)

- FP4 path: `mma.sync.aligned…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64…ue4m3`
  — **requires the arch-specific target `compute_120a`**; plain `sm_120` is rejected by
  ptxas 13.3 ("Feature '.kind::mxf4nvf4' not supported"). Evidence: `p2_emitted.ptx`.
- Fragment layouts pinned (delta exactly 0.0): LSB-first nibbles in both operands
  (element j at bit 4j), scale operand = one u32 per matrix, byte i = ue4m3 scale of the
  16-wide k-sub-block i; a/b/D fragment geometry as in RESULTS.md.
- cudarc chain works on Windows end to end: context, memcpy, NVRTC compile, module load,
  launch, verified result.

## Second machine — the Linux box (added 2026-09-17, issue #15)

The Linux port, the host-memory fix, the three refactor cuts, the two prefill floors and the
three live-session fixes of 2026-09-17 (branch `main`, `9f12429` to `487128d`) were all built and
measured here. Every Linux number in this repository names this box.

| component | value |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 — `sm_120`, 32,607 MiB VRAM |
| driver | 610.57.04 |
| CPU | Intel Core Ultra 9 285K |
| host RAM | 62.17 GiB (`MemTotal` 65,188,584 kB) |
| swap | zram, zstd, 62.2 GiB backing, with `vm.swappiness=150` (the Arch/omarchy default) |
| OS | Arch Linux, kernel 7.2.3-arch1-3 |
| userspace OOM | `systemd-oomd` active, watching `app.slice` on memory PRESSURE |
| CUDA | runtime 13.3.1 unpacked at `~/.local/share/crow/cuda` (`lib` on `LD_LIBRARY_PATH`, never `lib/stubs`); NVRTC 13.3.33, ptxas V13.3.73, cuBLAS 13.6 |
| Rust | rustc 1.98.1 (2026-09-01) from rustup stable, the Linux toolchain of record |
| container storage | btrfs; the `-M` container lives on a NOCOMPRESS path — `chattr +m` set BEFORE the file is written, verified with `filefrag -v <container> \| grep -c encoded` = 0 encoded extents, 2,817 extents total, sha256 unchanged by the rewrite |

Read on 2026-09-17 from `/proc/meminfo`, `/proc/sys/vm/swappiness`, `zramctl`, `uname -r`,
`nvidia-smi`, `ptxas --version`, `rustc --version`, `filefrag` and `lsattr` on this machine.

### The two facts that shaped the code

1. **The driver's pinned-page pool survives the process and is invisible to `MemAvailable`.**
   After any engine exits, about 45 GiB stays in the NVIDIA driver's pinned-page pool. It sits
   in no `/proc/meminfo` class, so `MemAvailable` cannot see it, yet it is reclaimable under
   pressure and the next `cuMemHostAlloc` is served straight out of it. A `MemAvailable`-based
   RAM gate therefore refused every second engine start. `cuda::free_physical_ram_parts` counts
   what cannot be reclaimed instead — `MemTotal - (AnonPages + Shmem + SUnreclaim + KernelStack
   + PageTables + Percpu)`. Measured 2026-09-17 with the pool present: free for pinning
   60.76 GiB against `MemAvailable` 10.89 GiB (`0c9feb5`).
2. **The desktop compositor holds `/dev/nvidia0`, which is why `/dev/nvidia-uvm` is the
   CUDA-process test.** That pool is only ours to count while no other CUDA process is alive, so
   the engine scans `/proc/<pid>/fd` for another process holding the device node. Measured
   2026-09-17: Hyprland, quickshell, Xwayland and every GTK/GL client hold `/dev/nvidia0`,
   `/dev/nvidiactl` and `/dev/nvidia-modeset` permanently and own no pinned pool — testing for
   those refused the engine's own operating point. `/dev/nvidia-uvm` is opened by every CUDA
   context and by no graphics client, so that is the node the check reads (`bb9d2ca`).

- Consequence for every measurement on this box: engine runs go through
  `tools/serve-linux.sh` or the same `systemd-run --user --scope --slice=session.slice`
  form (`MemorySwapMax=0`, `MemoryHigh=MemTotal-8G`, `MemoryMax=MemTotal-6G`), one engine at a
  time, because the RAM gate refuses a second engine while the first holds the pinned tier.
- The Windows-vs-Linux logit drift (Windows NVRTC 13.3.73 + driver 616.56 against Linux NVRTC
  13.3.33 + driver 610.57) is the toolchain, not the port: `docs/architecture.md` section 8.7
  carries the four values of record and the evidence.

## Model (research strands 2026-09-01, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`)

- Qwen3.8-Flash-Next: 131 safetensors, 359,999,963,128 B (~360 GB BF16), ~180 B params,
  1,658 tensors.
- Sparse-addressed share: PLE 102.4 GB (128 × `[2,500,012,160]`, tensors on `layers.1`
  while config says `ple_layer_ids: [2]`) + experts 241.6 GB (48 layers, fused
  `gate_up_proj [512,1280,2560]` + `down_proj [512,2560,640]`) = 95.55 % of the bytes.
- NVFP4 footprint at 4.5 bpw ≈ 101 GB on disk (decision: PLE block exchangeable).

## Baseline and goals

- Baseline: llama.cpp on this machine — 41 tok/s decode endstand, 67.1 % synchronization
  at 62 CPU↔GPU handoffs per token, RAM bus at 33.5 %, a swapped barrier costs 4–6 ms
  (Crow #159/#186).
- Goals (robin, 2026-09-01/02): ≥ 200k context (ceiling 262,144), ≥ 42 tok/s decode,
  ≥ 972 tok/s prefill, minimal latency (TTFT, inter-token). Every number is reported
  with its operating point named.
- Decisions 2026-09-02 (crow-nest #2, closed): cold path A with C designed in from day 1 ·
  ViT/MTP later, carried in the format · PLE NVFP4 with exchangeable block · default
  262,144 context with FP8-KV (slider 200k–262,144) · Ampere/Ada fallback as a later
  stage — **crow-nest runs Blackwell-only until that stage**.

## Open on this page

- The Linux box is above since 2026-09-17; the probes of 2026-09-01/02 were never re-run there,
  so the fragment-layout and handoff findings on this page remain Windows measurements. The
  engine's own Linux evidence is the parity battery (`docs/architecture.md` section 8.7,
  `tools/gate-linux.sh`), not a probe.
- `docs/measurement-handoff.md` still owes its Linux retest: the job-ring round-trip numbers
  there are WDDM numbers and no Linux figure replaces them.
- The parity battery reads ALL GREEN at `8ff2055` with `cargo test` at 165 and clippy at 1,422
  (2026-09-17). The ten-task gate has no Linux value of record: today's run is 1 of 10
  byte-identical to the Windows `final4` records and 9 flip on near-ties, and the run that would
  close it needs a reboot (`decode_out/final/GATES.md` section 6).
- Any second machine/OS is appended with its own probe evidence before its first
  measurement is quoted.
