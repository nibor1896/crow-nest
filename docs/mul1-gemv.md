# MUL1 trellis expert GEMV and FFN

crow-nest #180 (plan step 10, child of #169) computes a routed GLM-5.3-Flash expert straight from
its CNQ MUL1 record (#181, `converter/src/mul1.rs`): `y = down(silu(gate(x)) * up(x))`, without a
dequant pass, on two paths:

- **GPU** (sm_120): `engine/src/kernels_mul1.cu`, host side `kernels::mul1` (`MUL1_SRC`,
  `Kernels`, `GemvPlan`, `FfnPlan`). One pointer table per launch: a record base is a VRAM slot or
  a pinned-host UVA address, the kernels do not know which (the `gemv_fp4_ptrb` pattern).
- **CPU**: `engine/src/cpu_mul1.rs`, AVX2 with a scalar fallback, API and threading of
  `cpu_nvfp4` (#173).

**Status:** kernels and tests only. Nothing in the engine calls them. The GPU launch sites (plan
step 16), the CPU lane with its hit/miss split and a persistent thread pool (plan step 19) and gate
R (#174) are open. The int8-activation fast path of exllamav3's CPU kernel is out of scope (lossy,
see section 3).

## 1. Commands

```
cd engine
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_mul1                       # 7 tests, no GPU
CARGO_BUILD_JOBS=4 cargo test --release --lib tests_mul1_src                 # 1 test, NVRTC only
CARGO_BUILD_JOBS=4 cargo test --release --lib mul1_gpu -- --ignored --nocapture --test-threads 1 --skip bench
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_mul1_bench -- --ignored --nocapture
CARGO_BUILD_JOBS=4 cargo test --release --lib mul1_gpu_bench -- --ignored --nocapture
```

The GPU tests are `#[ignore]` (CI has no GPU) and need the RTX 5090. The CPU tests read the #181
fixtures from `converter/tests/fixtures` at compile time.

## 2. Format and math

- Record (#181): `[gate.trellis][up.trellis][down.trellis] [gate.suh][gate.svh] [up.suh][up.svh]
  [down.suh][down.svh] [zeros to a multiple of 4096]`. gate and up are `[k = hidden, n = inter]`,
  down `[inter, hidden]`. One GLM expert at K = 3: 3 x 3,145,728 B + 36,864 B = 9,474,048 B.
- Trellis: `(k/16)(n/16)` tiles of `16 K` u16 words, tile `kb * n/16 + nb` = rows `16 kb..`,
  columns `16 nb..`. Position `8 L + j` is the 16-bit window ending at bit `end_bit(8 L + j)` of the
  tile's circular MSB-first stream of LE u32 words, at row `2 (L % 4) + [0, 1, 8, 9][j % 4]`,
  column `L / 4 + 8 (j / 4)`.
- Weight: `w = rne16(s * k_inv - 3.453125)`, `s = bytesum(state * 0x83DCD12D)`,
  `k_inv = fp16(0x1eee)`. `s * k_inv - 3.453125` is exact in f32; one RNE rounding to fp16 is
  `__hfma`'s one rounding, so both paths give the codec's fp16 value for every state (tested
  exhaustively on CPU and GPU).
- Transform: `y = x W`, `W = diag(suh) H W_hat H diag(svh) / 128`, H the 128-wide Sylvester
  Hadamard (exllamav3 `get_weight_tensor`, `exl3.py:239-249` at `151539c7`). Both paths compute
  `xh = H (x * suh)`, `y' = xh W_hat` (decoding in the inner loop), `y = (H y') / 128 * svh`.
  The single `/ 128` (exact) replaces exllamav3's two `* 0.088388347648` (inexact).

| path | step | what |
|---|---|---|
| GPU | `mul1_had_in` | grid (k/128, T, E), one warp per 128-block: x * suh, FWHT |
| GPU | `mul1_gemv` | grid (n/128, S, E), 256 threads: warp = one tile column over k/S, lane = trellis lane; tile words staged through warp-private shared memory (one coalesced read per byte, also over PCIe) |
| GPU | `mul1_had_out` | sums the S partials left to right, FWHT, / 128, * svh |
| GPU | `mul1_act_had_in` | gate and up finished, `silu(g) * u` (also stored in `FfnPlan::h`), down's x * suh and FWHT |
| CPU | `gemv` / `expert_ffn` | input transform on the caller, output columns split over `threads` in units of 4 tile columns, k-major walk with prefetch; the FFN has two barriers (gate/up, then worker 0 finishes and prepares down, then down) |

## 3. Order of operations and the bound

Amended on #180 on 2026-10-09, before the first comparison
(https://github.com/nibor1896/crow-nest/issues/180#issuecomment-6070827734).

**CPU** (`cpu_mul1::rounding_steps`): x*suh 1, FWHT 7, per output 8 lane accumulators per tile
column in sybil's layout (`ft_core.h:218-246`), each a chain of k/4 products and adds (no FMA),
folded `(a0 + a1) + (a2 + a3)` 2, FWHT 7, * 2^-7 exact, * svh 1: **n = k/4 + 18** (1042 at k 4096,
530 at k 2048). AVX2 and scalar run exactly these operations: **AVX2 == scalar bit for bit for
every thread count**.

**GPU** (`kernels::mul1::rounding_steps`): x*suh 1, FWHT 7, per lane an fmaf chain of k/(4S), two
xor-shuffle adds 2, S partials S-1, FWHT 7, * 2^-7 exact, * svh 1: **n = k/(4S) + S + 17**, S the
largest power of two <= 16 dividing k/16 (GLM: S = 16, n = 97 at k 4096, 65 at k 2048). Every add
and multiply outside the fmaf chain is `__fadd_rn` / `__fmul_rn`, which NVRTC never contracts.

**Bound** (`|y_hat - y_ref|` per output; `gamma(n) = n u / (1 - n u)`, u = 2^-24, `gamma64` with
u = 2^-53 for the f64 reference's own rounding):

- ticket form (the acceptance): `gamma(n) * sum_i |W_ij x_i| + gamma64(k + 20) * M_j`, W the
  original-basis weight in f64 from the #181 decoder;
- rigorous form (asserted beside it): `(gamma(n) + gamma64(k + 20)) * M_j`, M_j the |.|-sum over
  every product path of the factorized form. The kernels evaluate that form, so Higham's theorem
  bounds them by M, not by `sum |W x|`;
- GPU vs CPU: `(gamma(n_gpu) + gamma(n_cpu)) * sum_i |W_ij x_i|`;
- FFN, per path with its own g_hat, u_hat, h_hat: `gamma(n_d) sum_j |Wd_jr h_hat_j| + gamma64 M_d
  + sum_j |Wd_jr| dh_j`, `dh_j = 1.1 Eg_j |u_hat_j| + |silu(g_j)| Eu_j + gamma(9) |h_hat_j|`,
  `Eg = gamma(n_gu) A_g + gamma64 M_g`; 1.1 >= sup |silu'|; gamma(9) covers the f32 activation,
  assumed within gamma(8) of the exact value (CPU: swept over g in [-40, 40], worst 1.88e-7 vs
  4.77e-7; GPU: CUDA's documented expf error of 2 ulp, not swept).

Activations stay f32. exllamav3's default CPU path (`moe_mul1.cpp:35-45`) quantizes them to int8
with one scale per input row (about 0.9 % output RMS in its author's words); that arm would need its
own acceptance (#166 template), not this bound.

## 4. Tests

Data: the 4 exllamav3 quantizer experts of #181 (K = 3, 2, 4, 3.5, real suh/svh, up to
[384, 256]) and the 3 GLM-shaped experts of the #181 synthetic set (`synth:11` K = 3, `synth:23`
K = 2, `synth:37` K = 3.5; [4096, 2048] / [2048, 4096]) with synthetic suh/svh (fp16, random sign,
magnitude 0.5..1.5). T 1, 2, 4; x uniform in [-2, 2). The reference decoder is a port of #181's,
held to the exllamav3 `reconstruct` digests #181 holds.

| test | asserts | worst measured (2026-10-09) |
|---|---|---|
| `cpu_mul1_codec_port_is_the_181_decoder` | the reference port gives all 21 exllamav3 digests | - |
| `cpu_mul1_weights_are_the_codec_values` | scalar weight == codec for all 65,536 states; AVX2 lane decode == codec tile, every bitrate 1..8, 1.5, 2.5, 3.5 | - |
| `cpu_mul1_shapes_are_checked` | GLM record 9,474,048 B, size checks, n constants, f64 FWHT == explicit Sylvester matrix | - |
| `cpu_mul1_gemv_holds_the_bound` | 7 experts x gate/up/down x T 1, 2, 4, scalar and Auto | 0.026 x ticket bound, 2.0e-4 x rigorous |
| `cpu_mul1_avx2_equals_scalar_bits` | T 1, 2, 3, 5, 8 x threads 1, 3, 8, 16 on the 4 quantizer experts; FFN of a full GLM expert | bit-identical |
| `cpu_mul1_activation_within_8_roundings` | f32 activation within gamma(8) | 1.88e-7 (gamma(8) = 4.77e-7) |
| `cpu_mul1_expert_ffn_matches_reference` | fused == staged bits, down stage bound, chain bound, every thread count and path | 0.0013 x chain bound |
| `tests_mul1_src::mul1_source_compiles_with_every_entry` | `MUL1_SRC` compiles (NVRTC, compute_120a) with its 5 entries; GPU n constants; record offsets | - |
| `mul1_gpu_decode_is_the_codec` (GPU) | GPU weight == codec for all 65,536 states | - |
| `mul1_gpu_gemv_holds_the_bound_in_vram_and_pinned` (GPU) | VRAM == pinned bits; ticket and rigorous bound; GPU vs CPU | 0.039 x bound, 3.3e-4 x rigorous; GPU vs CPU 0.018 x summed bound |
| `mul1_gpu_ffn_matches_reference` (GPU) | VRAM == pinned; fused down == public GEMV on the FFN's own h; down stage and chain bound; GPU vs CPU; 2 slots in one launch == 1 slot | 0.0021 x chain bound, down stage 0.063 x bound |

Every test was run once with its piece removed and failed (#180 implementation comment).

## 5. Micro-benchmark

2026-10-08 23:15 UTC (2026-10-09 local), release build, Core Ultra 9 285K (24 threads), DDR5-5600,
RTX 5090 (PCIe link width 16), Windows, pinned memory `Pinned::alloc_cold` (WC on Windows).
**Alongside a running download, not a clean measurement; one block, under a minute.** GLM-shaped
K = 3 records (synthetic trellis), rotated over 8 (CPU, 76 MB) or 16 (GPU, 152 MB) distinct records.

CPU (`cpu_mul1_bench`, AVX2, FFN, record bytes / wall time, median of n 24):

| T | 1 thread | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| 1 | 1.29 GB/s (7.34 ms) | 4.13 | **6.91** (1.37 ms) | 5.65 | 6.16 |
| 4 | 0.95 GB/s (10.0 ms) | 2.80 | **4.11** (2.30 ms) | 3.56 | 3.84 |

GPU (`mul1_gpu_bench`, 64 calls queued back to back per round, one sync, median of 5 rounds
after one warm-up, min..max):

| lane | kernel | T 1 | T 4 |
|---|---|---|---|
| VRAM | GEMV gate (3,145,728 B trellis) | 110.4 GB/s, 28.5 us (84.3..120.1) | 63.7 GB/s, 49.4 us (43.7..154.4) |
| VRAM | FFN (9,474,048 B record) | 132.2 GB/s, 71.7 us (125.5..135.4) | 106.4 GB/s, 89.1 us (87.2..126.1) |
| pinned | GEMV gate | 7.1 GB/s, 440 us (6.8..7.4) | 5.7 GB/s, 552 us (4.6..6.7) |
| pinned | FFN | 8.1 GB/s, 1171 us (7.6..8.2) | 6.1 GB/s, 1553 us (6.0..7.1) |

Reading (not analysed further here): both GPU lanes are far below their links (the pinned stage
pattern measured 51.6 GB/s, `cuda.rs` `alloc_registered` doc; VRAM peak ~1.8 TB/s), and the CPU
peaks at 8 threads, below the 27.96 GB/s of the NVFP4 FFN (docs/cpu-nvfp4.md section 6), so the
first kernels look bound by decode work and memory-level parallelism, not by DRAM or PCIe
(unverified attribution). By #180's failure mode these numbers stand in the ticket before plan
step 16 or 19 is built. The clean table of #180 item 5 (no download, n >= 20, boot-to-boot spread,
compared with #170 and plan step 5) is open.
