# CPU NVFP4 expert FFN

`engine/src/cpu_nvfp4.rs` (crow-nest #173, child of #169) computes one routed expert on the CPU,
straight from its NVFP4 bytes: `y = down(silu(gate(x)) * up(x))`, AVX2 with a scalar fallback,
multi-threaded over output rows. It is lever 2 of the GLM measurement book, section F ("CPU on
miss"): an expert in the pinned RAM tier is computed where it lies instead of crossing PCIe.

**Status:** kernel and tests only. Nothing in the engine calls it. The wiring (hit/miss split, the
per-layer GPU-CPU hand-off, a thread pool) belongs to the glm5_next port, #159 and plan steps
14-15; the DDR5 bandwidth it shares with the pinned tier is #170.

## 1. Commands

```
cd engine
CARGO_BUILD_JOBS=4 cargo test --lib cpu_nvfp4                     # 7 tests, no GPU
CARGO_BUILD_JOBS=4 cargo test --release --lib cpu_nvfp4_bench -- --ignored --nocapture
```

## 2. API

| item | what |
|---|---|
| `Nvfp4Matrix::new(bytes, rows, cols, global_scale)` | one NVFP4 matrix `[rows, cols]`, `cols` a multiple of 64, `bytes.len() == rows * cols / 64 * 36` |
| `ExpertBlock::from_unit(unit, hidden, inter, [gs_gate, gs_up, gs_down])` | one GLM expert unit as the converter writes it (`write_units`): gate `[I, H]` \| up `[I, H]` \| down `[H, I]`, each with its own global scale |
| `ExpertBlock::new(gate, up, down)` | the same from three matrices (also Flash-Next's fused `gate_up`: rows `0..I` and `I..2I`, one global scale) |
| `gemv(m, x, y, threads, path)` | `y[t][r] = sum_k W[r][k] x[t][k]`, `x` `[T][cols]`, `y` `[T][rows]` |
| `expert_ffn(e, x, y, threads, path)` | the FFN for `x`, `y` `[T][hidden]`; decode is T 1, small batches tile 4 tokens per pass over the weights |
| `Path` | `Auto` (AVX2 if the CPU has it), `Scalar`, `Avx2` (panics without AVX2) |

`threads` counts the caller's thread; workers split rows, never a sum. The FFN runs in one
`std::thread::scope` with one `Barrier` between the gate/up phase and the down phase.

## 3. Layout and scale rule

The decode is the engine's, bit for bit:

- block of 36 B: 4 ue4m3 sub-block scales, then 32 B of E2M1 codes; value `idx` in byte `4 + idx/2`,
  low nibble for even `idx` (`kernels.rs` `gemv_fp4`, converter `dequant_nvfp4`);
- E2M1 values and `ue4m3` are `cnq::e2m1` / `cnq::ue4m3` (the test asserts the table equal);
- sub-block scale = `ue4m3(byte) * global_scale` in f32, after the expert-slab rule of
  `residency::sanitize_sf_slab`: 0x7F (480 in the scalar device decode, NaN on the MMA path) reads
  as 0x7E (448). The kernel applies the rule itself, so raw container bytes and sanitized slabs give
  the same bits; bit 7 of a scale byte is ignored, as by the device decoder;
- activation `gate / (1 + exp(-gate)) * up`, the formula of `silu_mul640`.

## 4. Order of operations and error bound

Per output row and token: 8 lane accumulators; for each 16-value sub-block lane l takes values 2l
and 2l+1, `p = w[2l]·x[2l] + w[2l+1]·x[2l+1]`, `acc[l] = acc[l] + p·s`; the result is
`acc[0] + ... + acc[7]` left to right. Every operation rounds on its own (no FMA). Both paths run
exactly these operations, so **AVX2 == scalar bit for bit, for every thread count**. The order is
not the GPU's tree reduction, so the CPU result is not bit-identical to `gemv_fp4`; it is within:

`|y - y_exact| <= gamma(n) · sum_k |w_k x_k|`, `n = K/16 + 10`, `gamma(n) = n·u / (1 - n·u)`, `u = 2^-24`

(K 4096: 1.59e-5; K 2048: 8.2e-6), with `y_exact` the f64 dot product over the engine's f32 scales.

## 5. Tests (`cargo test --lib cpu_nvfp4`, names without the `cpu_nvfp4_` prefix)

| test | checks |
|---|---|
| `tables_are_the_engine_decoders` | E2M1 table = `cnq::e2m1`, scale LUT = `cnq::ue4m3` with 0x7F at 448, lane order |
| `shapes_are_checked` | wrong lengths and shapes are errors; a GLM unit splits into 3 × 4,718,592 B |
| `gemv_holds_the_fp32_bound` | shapes `[37, 256]`, `[16, 2048]`, `[9, 4096]`, T 1/2/3/5/8, scalar and AVX2, against f64 |
| `avx2_equals_scalar_bits` | every scale byte 0..255, T 1/2/3/5/8, threads 1/3/8 |
| `scale_rule_is_the_slab_rule` | within the bound of the 448 reference and outside it of the 480 one; raw == sanitized slab |
| `expert_ffn_matches_reference` | fused FFN == gemv + activation + gemv bit for bit; down stage in the bound; end to end within 1e-4 · sum &#124;w h&#124; of an all-f64 reference; threads and paths bit-identical |
| `full_glm_unit` | one 14,155,776 B unit: AVX2 == scalar, 64 gate rows in the bound at K 4096 |

The f64 reference sanitizes its rows with `residency::sanitize_sf_slab` itself, not with this
module's rule; with the rule removed, 5 of the 7 tests fail (#173 implementation comment).

## 6. Micro-benchmark

`cpu_nvfp4_bench` (`#[ignore]`, release): one FFN per call, rotating over 8 distinct GLM units
(113 MB, more than the 36 MB L3), 48 calls per cell, median. 2026-10-08, Intel Core Ultra 9 285K,
2 × 32 GB DDR5-5600, Windows 11, AVX2 path, threads spawned per call (no pool), **alongside a
running download, not a clean measurement**:

| T | 1 thread | 4 | 8 | 16 | 24 |
|---|---|---|---|---|---|
| 1 | 2.092 ms · 6.77 GB/s | 0.699 · 20.25 | **0.506 · 27.96** | 0.637 · 22.24 | 0.827 · 17.12 |
| 4 | 3.065 ms · 4.62 GB/s | 1.008 · 14.04 | 0.652 · 21.70 | 0.809 · 17.49 | 0.930 · 15.22 |

GB/s = 14,155,776 B of expert bytes per median call. Above 8 threads the per-call thread start costs
more than it brings; a persistent pool is part of the wiring, not of this kernel.
