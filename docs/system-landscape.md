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
| OS | Windows (Linux is a target system per constant 1 — not yet exercised by any probe) |
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

- Linux environment: unverified (both probes ran on Windows); entries land here when a
  Linux probe runs.
- Any second machine/OS is appended with its own probe evidence before its first
  measurement is quoted.
