# PREREG: calibrated NVFP4 scales for the dense 27B (Crow #300 phase 2, step "Local-Hessian")

Written 2026-09-26 before any calibration statistic, scale or KLD row of this step exists. robin approved the runs
("Ja, darfst die Läufe starten, go"). Research: Crow #300 comment of 2026-09-26 (NVIDIA Model-Optimizer
`local_hessian_calibrate`, PR #2417; nvidia/Qwen3.8-27B-NVFP4; arXiv 2512.02010, 2509.23202, 2606.07618).

## What changes
Only the choice of the ue4m3 sub-block scale byte of every NVFP4 tensor of `converter/Qwen3.8-27B-CNQ4.5.cnq`
(MLP gate/up/down, GDN in_proj_qkv/in_proj_z/out_proj, attention q/k/v/o, lm_head). Unchanged: the global scale
(max over sub-blocks of max|x|/6, divided by 448), E2M1 round-to-nearest (the converter's `quant_dequant`), the
container layout, every non-NVFP4 tensor. Same bytes on the card, same VRAM, same kernels.

## Arms (weights from the BF16 originals, f32 math, `oracle/ref_qwen35_logits.py`)
- REF: `--weights bf16`
- CNQ: `--weights cnq` (the container of record, `--scales mse`: moment seed + 2 Lloyd steps + 3 candidates)
- X126: every sub-block searches all 126 finite ue4m3 codes (1..0x7E) for the smallest plain weight SSE (control:
  isolates search completeness)
- DIAG: the same search, error weighted by the diagonal of the input statistics (sum x_i^2 per input column;
  the imatrix form)
- LH: the same search, error e^T H_b e with H_b = sum over calibration tokens of x_b x_b^T, the 16x16 matrix of
  the sub-block's 16 input columns (NVIDIA's Local-Hessian objective), shared by all output rows
A sub-block whose 16 weights are all 0 keeps byte 0. Ties keep the smaller byte.

## Calibration data (held out from every evaluation row; fixed now)
Token sequences, Qwen3.8 tokenizer, deterministic window choice (numpy seed 0):
- agent traffic: the Crow sessions under `decode_out/sessions/` (2026-09-17-goalmode, 2026-09-22-diorama-rollover,
  2026-09-23-diorama), rendered through the 27B chat template: 96 windows of 2,048 tokens + 4 windows of 8,192
- code: crow-nest `engine/src`, `converter/src` (Rust) and Crow `cli/*.py` (Python): 64 windows of 2,048
- notes/docs: the Obsidian vault markdown (mostly German) 32 windows, crow-nest `docs/*.md` (English) 32 windows
Total 224 x 2,048 + 4 x 8,192 = 491,520 tokens. Excluded by rule: `tools/corpora/*` (the five books),
`decode_out/oracle-*` id sets, the libghost long-context generator. A source that turns out unusable is replaced
by a dated amendment written before any statistic is computed.
Statistics are taken on the CNQ container's own forward (f32, GPU, layer by layer), accumulated in f64.

## Evaluation rows (the campaign's short + prose sets, decode_out/p2-kld/PREREG.md)
`oracle-tf298` rows 0..297, `oracle-t2b` rows 0..606, `oracle-en` 5 books x 513 rows = 3,470 rows.
Metric: `tools/oracle-kld.py`, KLD(REF || arm), mean +- SE, median, p99, same top-1; paired per-row differences.

## Checks before any KLD is trusted
1. The GPU oracle against the CPU oracle, CNQ weights, set tf298: mean KL(CPU || GPU) <= 1e-5 and same argmax on
   every row (else the GPU port is wrong and nothing below is run on it).
2. The Python quantizer, fed the container's own scale bytes, reproduces the container's dequantized values
   bit-exactly on at least one tensor of each group (else the simulation does not simulate our encoder).

## Decision rule (fixed now)
- An arm is a WIN if its paired gain mean(KLD_CNQ - KLD_arm) over the 3,470 rows is > 2 SE AND >= 10 % of CNQ's
  mean, and neither the code rows (905) nor the prose rows (2,565) get worse by more than 2 SE.
- Among winning arms the one with the largest gain is built as a real container; if two differ by < 2 SE of their
  paired difference, the simpler one is built (order X126 < DIAG < LH).
- No win: the container of record stays; the result is reported, nothing is built.
- The real container (if built) must then match its simulation: engine BF16-KV parity against the arm's f32
  reference, paired mean(E2 - arm) <= 10 % of the arm's mean (the Q3 rule), p2golden ALL PASS.
