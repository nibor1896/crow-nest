# Probe results — Crow #188, step (a), 2026-09-01

Machine: RTX 5090 (sm_120, CC 12.0, 170 SMs, 32 GiB), driver 616.56 (UMD CUDA 13.4),
CUDA toolkit 13.3, Rust 1.97.0, Windows.

## Probe 1 — cudarc on Windows: PASS

`CudaContext::new(0)` with cudarc 0.19.9, features `cuda-13030` + `dynamic-loading` +
`nvrtc`:

- context creation works on Windows (the strand-4 caveat listed only `cargo check` CI
  evidence; all historic Windows failures were runtime DLL loads — none occurred here)
- compute capability 12.0, 170 SMs reported via `CUdevice_attribute`
- host<->device round trip of 1024 f32 exact
- NVRTC 13.3 compiles a trivial kernel (arch `compute_120`), driver JIT accepts it,
  launch produces the correct vector sum

## Probe 2 — one mma.sync…mxf4nvf4 via NVRTC on the 5090: PASS, and it computes

Kernel: one warp, one m16n8k64 tile, exactly one
`mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`
(copied verbatim from llama.cpp `mma.cuh`), NVRTC-compiled, compared against a CPU
reference decoded from the same packed fragments. All values dyadic -> expected delta on
success is exactly 0.0.

### New finding 1: the instruction needs the arch-specific target `sm_120a`

- ptxas 13.3 rejects the PTX on the plain target: *"Feature '.kind::mxf4nvf4' not
  supported on .target 'sm_120'"* — for `mma with block scale`, `.block_scale`,
  `.scale_vec::4X` alike. With `.target sm_120a` the same PTX assembles cleanly
  (5,936-byte cubin).
- NVRTC must therefore be called with `--gpu-architecture=compute_120a`; the driver JIT
  (UMD 13.4) then loads the emitted PTX without complaint.
- This matches llama.cpp, which builds Blackwell FP4 as `120a-real` (their CMakeLists
  comment: "120 == Blackwell, needs CUDA v12.8, FP4 tensor cores").
- Consequence for strand 4's model: "sm_120 computes via mma.sync" is necessary but not
  sufficient — the engine's kernel path must target `120a`, and `120a` cubins/PTX are
  arch-specific (no forward portability, same caveat family as sm_100a).

### New finding 2: layouts pinned exactly, delta 0.0000 across the sweep

Config sweep (aperiodic test data; see "methodology" below):

| config | packing / scales | result |
|---|---|---|
| e0 | LSB-first nibbles both operands, sf A=[1,2,.5,1] B=[1,1,1,2] byte i -> k-block i | **MATCH, delta 0.0000** |
| e1 | all scales 1.0 | MATCH, delta 0.0000 |
| e2 | A nibbles packed MSB-first (reference models LSB hardware read) | MATCH, delta 0.0000 |
| e3 | B nibbles packed MSB-first (same modelling) | MATCH, delta 0.0000 |
| e4 | scale bytes reversed: A=[1,.5,2,1] B=[2,1,1,1], byte i -> k-block i | MATCH, delta 0.0000 |
| e5 | A[0][1] sign flipped (+1.0 -> -1.0), e0 reference | no match — delta exactly 12.0 = 2 * 1.0 * B[1][0]=4, the exact expected effect |

Confirmed for the engine:

- fragment layout H1 as hypothesized from the PTX ISA m16n8k64 e2m1 tables (per lane
  `g = lane >> 2`, `t = lane & 3`: a0 = A[g][8t..8t+7], a1 = A[g+8][8t..8t+7],
  a2 = A[g][32+8t..], a3 = A[g+8][32+8t..]; b0 = B[8t..][g], b1 = B[32+8t..][g];
  D written as c0 = D[g][2t], c1 = D[g][2t+1], c2/c3 rows g+8) — all 128 outputs match
- e2m1 nibble order is LSB-first within each register (element j at bit 4j), pinned from
  both packing directions (e0 + e2/e3)
- scale operand = one u32 per matrix; byte i (LSB-first) is the ue4m3 scale for the
  16-wide k-sub-block i, independently for A and B (e0 vs e4 change the output exactly
  as the byte-wise products predict)
- accumulation is exact for dyadic inputs — 0.0000 delta in every matching config
- the D fragment write order used here reproduces the 16x8 output matrix correctly

### Methodology notes (why the first two runs lied)

1. The first data generator was linear in k with period 16 (`a[m][k+16]==a[m][k]`,
   `b[k+16][n]==b[k][n]`), so all four k-sub-block sums were equal per output cell and
   every scale pattern summed to the same total — four contradictory "results" were all
   correct arithmetic over degenerate data. Fixed with `(m+3)*k` / `(k+3)*(n+3)` mod
   terms.
2. Flipping the sign bit of an element that decodes to 0.0 changes nothing — the
   transport-sanity flip must target a nonzero element.
3. Permuting A and B identically along k cannot change the matmul (index relabeling),
   so nibble order is testable only per operand. (Corollary seen live: e2 and e3 —
   A-permuted vs B-permuted — produce bit-identical GPU outputs, as substitution
   predicts.)

## What this unblocks

- Step (b) of the record: 2-3 architecture options with trade-offs and a recommendation.
- The Rust chain from strand 4 stands on this machine, not just on paper: cudarc
  (driver, streams, NVRTC, module load, launch) + NVRTC-emitted sm_120a kernels +
  block-scaled FP4 tensor cores on the 5090.
- The kernel path must carry the `120a` target constraint as a first-class fact
  (Ampere/Ada fallback remains a separate, INT4-style path — unchanged).

## Reproduce

See README.md. `p2_emitted.ptx` is the NVRTC output of the last run (kept as evidence:
`.target sm_120a`).
