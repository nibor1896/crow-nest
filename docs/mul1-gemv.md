# MUL1 trellis expert GEMV and FFN

crow-nest #180 (plan step 10, child of #169) computes a routed GLM-5.3-Flash expert straight from
its CNQ MUL1 record (#181, `converter/src/mul1.rs`): `y = down(silu(gate(x)) * up(x))`, without a
dequant pass, on two paths:

- **GPU** (sm_120): `engine/src/kernels_mul1.cu`, host side `kernels::mul1` (`MUL1_SRC`,
  `Kernels`, `GemvPlan`, `FfnPlan`). One pointer table per launch: a record base is a VRAM slot or
  a pinned-host UVA address, the kernels do not know which (the `gemv_fp4_ptrb` pattern).
- **CPU**: `engine/src/cpu_mul1.rs`, AVX2 (+ FMA) with a scalar fallback, the API of
  `cpu_nvfp4` (#173), its own persistent worker pool (#183 C1).

**Status:** kernels and tests; `glm5_moe` (#164) calls `cpu_mul1::gemv`, nothing else calls them.
The GPU launch sites (plan step 16), the CPU lane with its hit/miss split (plan step 19) and gate
R (#174) are open. The int8-activation fast path of exllamav3's CPU kernel is out of scope (lossy,
see section 3).

## 1. Commands

```
cd engine
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_mul1                       # 11 tests, no GPU
CARGO_BUILD_JOBS=4 cargo test --release --lib tests_mul1_src                 # 1 test, NVRTC only
CARGO_BUILD_JOBS=4 cargo test --release --lib mul1_gpu -- --ignored --nocapture --test-threads 1 --skip bench
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_mul1_bench -- --ignored --nocapture
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_nvfp4_bench -- --ignored --nocapture   # the C1 comparison
CARGO_BUILD_JOBS=4 cargo test --release --lib mul1_gpu_bench -- --ignored --nocapture
CARGO_BUILD_JOBS=4 cargo test --release --lib mul1_gpu_read_bench -- --ignored --nocapture
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
| GPU | `mul1_gemv` | grid (n/128, S, E), 256 threads: warp = one tile column over k/S, lane = trellis lane; the block loads the contiguous run of its 8 tiles per tile row (32 n32 bytes, whole 128-byte lines) with 16-byte loads into a two-deep shared ring, the next stage in flight while the warps compute (#183) |
| GPU | `mul1_gemv_warp` | the first kernel of #180, reference arm only (`Kernels::use_warp_gemv`): each warp loads its own tile, 96 B per warp request at K = 3; same f32 order (`mul1_tile_fma`), same bits |
| GPU | `mul1_had_out` | sums the S partials left to right, FWHT, / 128, * svh |
| GPU | `mul1_act_had_in` | gate and up finished, `silu(g) * u` (also stored in `FfnPlan::h`), down's x * suh and FWHT |
| CPU | `gemv` / `expert_ffn` | one run of the persistent pool (`pool`: workers spin ~2 ms after a job, then park; the caller is worker 0; a worker that has not started when worker 0 is done is skipped, not waited for): input transform per 128-block of x, then tile columns in guided chunks of 4, 2, 1 columns from an atomic counter (`chunks`: the last chunk a slow core takes is one column), the worker that completes a 128-column block finishes it (Hadamard out; in the FFN also the activation and down's input transform), later phases wait for completed work, never for workers. K-major walk with prefetch of every line 6 tile rows ahead; lane states by one byte shuffle + shift + mask from a division-free window (`Lane`: one 16-byte load, or `alignr` of the first and last 16 bytes for the lane that wraps), weight via `vpmaddwd` x 8192 + the bits of 1024.0 and one exact FMA; at K = 3 the 32 lanes of a tile are unrolled over compile-time constants (`K3_LANES`, `unit_k3`). Reference arms, test only: `Impl::V2` (#183) and `Impl::V1` (#180) |

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
| `cpu_mul1_decode_and_schedule_equal_v1_bits` (#183) | shuffle decoder == first decoder for every lane, bitrates 1..8, 1.5, 2.5, 3.5; the production path == `Impl::V1`: GEMV T 1, 2, 3, 5, 8 x threads 1, 3, 8, 16 on 9 experts, GLM FFN T 1, 4 x threads 1, 8, 16, 24 | bit-identical |
| `cpu_mul1_c1_kernel_equals_v2_bits` (#183 C1) | `weights_fast` == `weights` for all 65,536 states; `decode8_fast` == `decode8` for every lane, bitrates 1..8, 1.5, 2.5, 3.5, 64 random tiles each; `unit_k3` == `unit_fast` (GLM gate, T 1..4); the production path == `Impl::V2`: GEMV T 1, 2, 3, 5, 8 x threads 1, 3, 8, 16, 24 on 9 experts, GLM gate / down T 1, 4, GLM FFN T 1, 2, 4, 5 x threads 1, 2, 8, 16, 24 | bit-identical |
| `cpu_mul1_scales_and_chunk_plan` (#183 C1) | bit-built `f16_to_f32` == the former `powi` formula for all 65,536 inputs; the chunk plan covers every column once, in aligned chunks of 4 / 2 / 1 inside one 128-block, ending in one-column chunks | - |
| `cpu_mul1_pool_runs_all_work_once` (#183 C1) | 300 pool runs (n 2, 3, 8, 16, 24): all work done before `run` returns, job(0) once, no worker twice or at or past n; two threads calling at once; a nested call; a worker's panic raised in the caller, the pool usable after | - |
| `tests_mul1_src::mul1_source_compiles_with_every_entry` | `MUL1_SRC` compiles (NVRTC, compute_120a) with its 6 entries; GPU n constants; record offsets | - |
| `mul1_gpu_decode_is_the_codec` (GPU) | GPU weight == codec for all 65,536 states | - |
| `mul1_gpu_gemv_holds_the_bound_in_vram_and_pinned` (GPU) | VRAM == pinned bits; ticket and rigorous bound; GPU vs CPU | 0.039 x bound, 3.3e-4 x rigorous; GPU vs CPU 0.018 x summed bound |
| `mul1_gpu_gemv_block_equals_warp_bits` (GPU, #183) | `mul1_gemv` == `mul1_gemv_warp`: 12 bitrate cases x gate/up/down x T 1, 3, 8, two slots, VRAM / pinned / trellis base 4 B off 16-byte alignment; GLM FFN from pinned RAM | bit-identical, 324 GEMV cases + 1 FFN |
| `mul1_gpu_ffn_matches_reference` (GPU) | VRAM == pinned; fused down == public GEMV on the FFN's own h; down stage and chain bound; GPU vs CPU; 2 slots in one launch == 1 slot | 0.0021 x chain bound, down stage 0.063 x bound |

Every test was run once with its piece removed and failed (#180 implementation comment; the #183 C1
tests with a deliberate fault each, #183 C1 implementation comment).

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

#183 (2026-10-08 23:41 UTC, the same machine, release, **alongside a running download, not
clean**; one binary, the old and new kernels as alternating arms; GPU 5 rounds x 64 calls per arm,
CPU median of 24 calls per arm; NVFP4 `cpu_nvfp4_bench` right after in the same session):

| lane | T | first kernel (#180) | #183 | ratio |
|---|---|---|---|---|
| pinned GEMV gate | 1 | 8.0 GB/s | **36.2** | 4.5 x |
| pinned FFN | 1 | 9.4 GB/s | **35.9** | 3.8 x |
| pinned FFN | 4 | 9.1 GB/s | 31.5 | 3.5 x |
| VRAM FFN | 1 / 4 | 143.7 / 144.8 GB/s | 150.8 / 152.8 | 1.05 x |
| CPU FFN, 1 thread | 1 | 1.27 GB/s (7.445 ms) | 1.69 (5.593 ms) | 1.33 x |
| CPU FFN, 8 threads | 1 | 7.07 GB/s (1.339 ms) | **9.12** (1.039 ms) | 1.29 x |
| CPU FFN, 16 / 24 threads | 1 | 5.79 / 5.94 GB/s | 9.68 / 9.42 | 1.67 / 1.59 x |
| CPU FFN, 8 / 24 threads | 4 | 4.42 / 3.90 GB/s | 4.96 / 5.24 | 1.12 / 1.34 x |

Causes (#183): from pinned RAM the first kernel's load shape alone (`rd_warp96`, one warp per tile,
96 B per request) reads 7.8-8.0 GB/s, the same bytes as 128-byte-aligned block runs (`rd_block16`)
39.9-41.5 GB/s, a plain 16 B sweep 28.9-34.9 GB/s on this Windows box (`mul1_gpu_read_bench`;
the Linux device-issued ceiling of record is 51.6 GB/s); the pinned FFN now runs at 70 % of 51.6 and
~90 % of the best local read shape. On the CPU the lane decode is ~94 % of one thread (decode alone
1.66 of 1.76 ms per gate GEMV), and the static split made 16 threads slower than 8.

**CPU target not reached at `13bea90`** (superseded by the C1 follow-up below): #183 asked for the MUL1 FFN at >= 50 % of the NVFP4 FFN's GB/s at the same
thread count (8 threads: >= 14.01 of 28.02 GB/s); measured 9.12 (33 %). Both formats run the same
25.2 M weights, so the target needs MUL1 within 1.34 x NVFP4's time per expert; one thread takes
5.593 ms against NVFP4's 2.104 (2.66 x), and even perfect scaling over 8 P-cores (0.70 ms, 13.5 GB/s)
stays under the target. Starting 8 scoped threads and joining them costs 171 us per call on this
Windows box (measured with empty work), for both formats; a persistent pool is plan step 19.

### #183 C1 follow-up: the CPU target met

2026-10-09 15:56 UTC, the same machine, release, one binary, two sessions (S1 15:56:04, S2 15:56:14);
total CPU load 0.8-3.3 % before and after each (`typeperf`), no other heavy process. `old` =
`Impl::V2` (the #183 kernel as at `bd52328`), timed after the pool's workers parked; `new` alternates
with it and so starts from parked workers every call; `new warm` = 24 calls back to back; median of 24
per arm; `cpu_nvfp4_bench` right after each session. GB/s of each format's own bytes, S1 / S2:

| T | threads | old (#183) | new | new warm | NVFP4 | new / NVFP4 |
|---|---|---|---|---|---|---|
| 1 | 1 | 1.76 / 1.76 (5.382 ms) | 3.02 / 3.02 (3.138 ms) | 3.00 / 3.00 | 6.96 / 7.05 | 43 / 43 % |
| 1 | 4 | 5.65 / 5.65 | 11.57 / 11.61 | 11.79 / 11.79 | 20.87 / 20.99 | 55 / 55 % |
| 1 | **8** | 9.66 / 9.71 (0.980 ms) | **19.74 / 21.23** (0.480 / 0.446 ms) | 22.57 / 22.79 | 27.92 / 28.40 (0.507 / 0.498 ms) | **71 / 75 %** |
| 1 | 16 | 9.75 / 9.69 | 26.31 / 26.66 | 29.35 / 29.51 | 22.09 / 21.94 | 119 / 122 % |
| 1 | 24 | 9.39 / 9.52 | 31.92 / 32.48 | 35.55 / 34.26 | 17.06 / 17.79 | 187 / 183 % |
| 4 | 1 | 1.20 / 1.20 (7.902 ms) | 1.89 / 1.89 (5.015 ms) | 1.88 / 1.88 | 4.79 / 4.72 | 39 / 40 % |
| 4 | 4 | 3.60 / 3.67 | 7.27 / 7.30 | 7.37 / 7.40 | 13.93 / 14.26 | 52 / 51 % |
| 4 | 8 | 5.68 / 5.78 | 14.20 / 14.29 | 14.32 / 13.54 | 21.62 / 21.55 | 66 / 66 % |
| 4 | 16 | 5.90 / 5.85 | 18.31 / 18.34 | 18.62 / 18.46 | 17.32 / 17.68 | 106 / 104 % |
| 4 | 24 | 5.77 / 6.07 | 22.18 / 22.00 | 22.45 / 22.45 | 15.24 / 15.68 | 146 / 140 % |

**C1** (CPU FFN, T 1, 8 threads, `new` >= 0.5 x the NVFP4 FFN of the same session): 19.74 >= 13.96
(S1) and 21.23 >= 14.20 GB/s (S2): **met**. What changed, and what each step was measured to remove
(the step measurements ran beside other jobs, not clean; they explain, the table above judges):

- the lane window without a division (the #183 decoder took the wrapping lane's words modulo
  `n32`, 2 of 32 lanes per tile at K = 3) and, at K = 3, the 32 lanes unrolled over compile-time
  constants: one-thread gate GEMV 1.27-1.29 ms (table-driven) -> 1.03-1.05 ms (unrolled);
- the weight from `vpmaddwd` x 8192 + the bits of 1024.0 and one exact FMA: one-thread FFN 3.36 ->
  3.14 ms (two runs each, back to back, ~1-2 % load; both bit-identical); an AVX2 CPU without FMA
  keeps the #183 unit;
- the input transform: 78-94 us serial per FFN at `bd52328` (`f16_to_f32` called `powi` per scale)
  -> bit-built conversion, and the transform runs per 128-block inside the pool;
- zeroed per-call buffers (~20 us per FFN) -> a per-thread scratch vector;
- fixed 4-column units: with 2-5 of 8 workers on slow cores (half the units each) every phase ended
  with a ~100 us tail -> guided chunks of 4, 2, 1 columns (tail 5-20 us);
- 8 scoped threads per call (171 us, #183) -> the persistent pool.

Not measured: a re-run by the lead on a machine known to be quiet, Linux, E-cores alone, the CPU
time the idle workers spin away after a call (up to `pool::SPINS` pauses each). Earlier sessions
the same day under 5-8 % load (15:45 UTC, the same kernel before the no-FMA fallback and the
test-only scratch poisoning were added) gave 20.63 / 20.46 GB/s at 8 threads against NVFP4 23.98 / 24.55, and a warm-pool
median of 0.761 / 1.060 ms at T 4, 24 threads (max 2.2-2.3 ms, the cold arm 0.44 ms; spinning
workers on every core), not seen in the quiet sessions above.
