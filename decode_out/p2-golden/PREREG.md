# PREREG: dense 27B sub-block goldens (Crow #300 phase 2)

Written 2026-09-26 before the first engine-vs-golden run. Goldens: oracle/golden/qwen35-27b/manifest.json
(HF qwen3_5 modules, f32 eager, weights = the CNQ4.5 container dequantized, seed 20260926, T=40 prompt + 4 decode rows).

Metric per golden and per row group (prompt rows 0..39, decode rows 40..43 separately):
rel_rms = ||engine - golden_cnq||_2 / ||golden_cnq||_2, plus max_abs (reported).

Pass thresholds (engine math may add at most 1/10 of the quantization error the manifest measures, `cnq_vs_bf16.rel_rms`):

| golden | quant mark | threshold |
|---|---|---|
| l0-input-layernorm | 0 | rel_rms <= 1e-5 |
| l0-mlp | 0.0404 | rel_rms <= 4.0e-3 |
| l0-gdn (prompt, decode) | 0.1597 | rel_rms <= 1.6e-2 |
| l3-attn (prompt, decode), CROW_KV=bf16 | 0.1565 | rel_rms <= 1.57e-2 |

l3-attn under the default FP8 KV is reported as a measurement, not judged (FP8 KV is the production choice of record, a deliberate precision trade).
Engine: default switches (CROW_MMA / CROW_GRAPH as the binary defaults them), one run each, RTX 5090.
