# RESULTS: MTP speculative decoding for the dense 27B, against decode_out/p2-mtp/PREREG.md

2026-09-27, RTX 5090, crow-nest `22bea05`, container `Qwen3.8-27B-CNQ4.5.cnq` (DIAG scales, sha256 `ba014aeb…`),
BF16 KV, `CROW_MMA=1 CROW_GRAPH=1 CROW_MTP=1`, kmax 3, adaptive k. `decode mtpspec` runs plain decoding and the
serve-shaped path (`spec_step` x n, `spec_finish`, `decode_step` x 16) in one process, one after the other.

## C1 (MTP forward vs f32 reference) - FAIL by the letter on one row, attributed
- apollo + the 27B's 128 greedy tokens, 190 rows: argmax 190/190, mean KL 7.0e-6.
- tf298, 297 rows, `CROW_MMA=1`: 295/297, one flip at a reference top-2 margin >= 0.01 (row 7, 0.0133), mean KL 4.7e-5.
- tf298, `CROW_MMA=0`: 0 judged flips, mean KL 2.5e-5. The flip comes from the main model's NVFP4 activation
  quantization of the prompt path under `CROW_MMA=1`, not from the head.

## C2 (output identical to plain decoding) - PASS (greedy); sampled by a stronger check than registered
- Greedy: byte-identical on the 6 prompts x 512 tokens (apollo, tf298 prompt, t2b prompt, dracula, moby, 8k Crow
  session window), in the batched and the serve-shaped path, and in all S runs below (continuation after
  `spec_finish` included).
- Sampled: the registered check was a chi-square test over 20,000 draws on 3 contexts. It was NOT run. Instead:
  each emitted token is exactly one draw (the verify rows are drawn lazily), so with the same seed the sampled
  output is the plain sampled output byte for byte - observed for the non-thinking row (seeds 1, 2, 3, 7) and the
  thinking row (seeds 1, 2, 3, 7), 512 tokens each, and in `serve` (greedy and sampled answers identical with and
  without MTP). Identity of the draws implies identity of the distribution; this is a deviation from the
  registered method, stated here.

## S (speed; plain / MTP tok/s, n = 3)
| check | runs | ratios | threshold | verdict |
|---|---|---|---|---|
| S1 greedy, apollo, 512 tokens | 73.2/120.2, 72.2/120.3, 72.3/120.0 | 1.64, 1.67, 1.66 | >= 1.5 | PASS |
| S2 greedy, 30k prompt, 256 tokens | 59.0/73.1, 58.9/72.1, 58.9/71.9 | 1.24, 1.22, 1.22 | >= 0.95 | PASS |
| S3 non-thinking row (0.7/0.8/20/0/1.5), apollo, seeds 1-3 | 72.5/103.2, 71.9/87.7, 71.9/95.4 | 1.42, 1.22, 1.33 | >= 1.2 | PASS |
| S3 thinking row (1.0/0.95/20/0/0), apollo, seeds 1-3 | 72.2/109.4, 71.8/100.4, 71.9/99.6 | 1.52, 1.40, 1.39 | >= 1.2 | PASS |
Tokens per pass: S1 3.32-3.38, S2 2.66-2.80, S3 2.14-2.87. Reference on this card: llama.cpp 27B `UD-Q4_K_XL`
66.5 plain, 123.05 with its MTP.

## V (VRAM at context 100,000) - FAIL on the added bytes, PASS on the lend
- `serve` load: 23.85 GiB used with MTP, 22.07 without: +1.78 GiB > 1.7 GiB. FAIL by 0.08 GiB.
- Lend: 2,300 MiB lendable, about 1.4 GiB of it the MTP weights / slots / scratch (>= 1.2 GiB: PASS); the
  weights come back from host copies in 47 ms; the same request after the return identical (C2) at 85.4 tok/s vs
  86.5 before (-1.3 %, within +- 5 %: PASS).
