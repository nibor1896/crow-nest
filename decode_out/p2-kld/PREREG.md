# PREREG: what the dense 27B's crow-nest quantization and engine cost against the BF16 model (Crow #300 phase 2)

Written 2026-09-26 before any row of this campaign is computed. Only crow-nest is measured (robin: "Wir vermessen
Crow Nest, nicht llama.cpp").

## Rows (existing, verified id sets; Qwen3.8 tokenizer, byte-identical to Flash-Next's)
- short code: `decode_out/oracle-tf298` (rows 0..297) and `decode_out/oracle-t2b` (rows 0..606) = 905 rows
- English prose: `decode_out/oracle-en`, 5 books x 512 rows = 2,560 rows
- long context (engine arms only, see Q4): `decode_out/oracle-longctx/longctx-170k-a{1000,2564,50000,100000,158000,178553}-ids.json`,
  the last 64 rows of each (`CROW_PARITY_TAIL=64`)

## Arms
- REF: `oracle/ref_qwen35_logits.py --weights bf16` (the BF16 originals, f32 math, eager) — the reference
- CNQ: the same with `--weights cnq` (our container dequantized with the engine's load rule) — the pure weight quantization
- R1: the same with attention q/k/v/o and GDN in_proj_qkv / in_proj_z / out_proj as FP8 e4m3 with one scale per tensor
  (amax / 448, taken from the BF16 originals; SIMULATION of the precision split of nvidia/Qwen3.8-27B-NVFP4, NOT its
  calibrated NVFP4), everything else as CNQ
- E1: `decode parity`, production (`CROW_MMA=1 CROW_GRAPH=1`, FP8 KV)
- E2: the same with `CROW_KV=bf16`

## Metric
`tools/oracle-kld.py` (llama.cpp's definitions, docs/oracle-kld.md 1): KLD(REF || arm) per row, mean with standard error,
median, p90, p99, same top-1; paired per-row differences between arms with their standard error.

## Questions and decision rules (fixed now)
- Q1 (is our quant bad?): CNQ mean KLD over short + prose. Context, not pass/fail: published 4-bit 27B means are
  0.0227 (Unsloth Qwen3.6-27B 4-bit), 0.0325 (its NVFP4), 0.0028 (8-bit) — different model and unknown corpus.
  Reading: CNQ <= 0.03 = "in the range of published 4-bit quants".
- Q2 (does the NVIDIA precision split pay?): paired mean(KLD_CNQ - KLD_R1). Worth a recipe change only if the gain is
  >= 25 % of CNQ's mean AND > 2 standard errors; its cost (+~4 GB VRAM, slower decode, new FP8 kernels) is weighed
  after, with robin.
- Q3 (does the engine add error?): paired mean(KLD_E2 - KLD_CNQ). Flag if > 10 % of CNQ's mean.
- Q4 (FP8 KV): paired mean(KLD_E1 - KLD_E2) on short + prose; at long context KL(E2 || E1) per anchor (no f32
  reference there). FP8 KV stays the default if at every anchor mean KL(E2 || E1) <= 0.25 x CNQ's mean KLD.
