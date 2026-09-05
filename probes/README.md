# crow-nest probes — step (a) before any architecture decision

Epic: Crow #188, decision record 2026-09-01. These are the two local probes the record
demands before any architecture is chosen; no engine code lives here.

## p1_cudarc_context — the driver/toolkit layer on Windows

`CudaContext::new(0)` with cudarc 0.19.9 (`cuda-13030`, `dynamic-loading`, `nvrtc`) on
Windows, RTX 5090, driver 616.56, CUDA 13.3 toolkit: context, compute-capability query,
host<->device round trip, NVRTC compile (compute_120), module load, launch, verified
result. This is the whole chain the engine's driver layer would sit on.

## p2_nvfp4_mma — one mma.sync…mxf4nvf4 on the 5090

The instruction is copied verbatim from llama.cpp `ggml/src/ggml-cuda/mma.cuh`
(`mma_block_scaled_fp4`, GGML_TYPE_NVFP4 branch), independently confirmed by cubecl
`crates/cubecl-cpp/src/cuda/mma/manual.rs` (arch >= 120 && < 130: E2M1 x E2M1, k64,
scales E4M3, scales_factor 4). One warp, one m16n8k64 tile, one mma.sync; the host packs
fragments, the kernel computes, the result is compared against a CPU reference decoded
from the same packed bits. All values are dyadic, so the expected delta on success is
exactly 0.0.

Findings and the config sweep are documented in `RESULTS.md`.

## Run

PowerShell:

```powershell
$env:PATH = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v13.3\bin;$env:PATH"
cargo run --release --bin p1_cudarc_context
cargo run --release --bin p2_nvfp4_mma
```

The PATH entry is needed so cudarc's dynamic loading finds `nvrtc64_133_0.dll`
(`nvcuda.dll` comes from the driver, in System32).
