# PREREG: long-context parity of the dense 27B against the f32 reference (Crow #300 phase 2)

Written 2026-09-26 before the first run. Prompt: the first 8,192 tokens of `docs/architecture.md` (Qwen3.8 tokenizer).
Engine: `decode parity` with `CROW_PARITY_TAIL=64` (prefill 8,128 tokens in 2,048-token chunks without logits, then the
last 64 prompt ids teacher-forced through `decode_step`, then 4 free greedy decode steps = 68 logit rows at positions
8,128..8,195), `CROW_MMA=1 CROW_GRAPH=1`, FP8 KV (production). Reference: `oracle/ref_qwen35_logits.py --weights cnq`
(the container dequantized with the engine's load rule, f32, the 64 HF `Qwen3_5DecoderLayer`s, eager attention).

Pass (all three):
1. argmax equal at every row whose reference top-2 margin is >= 0.5 nats;
2. mean KL(ref || engine) over the 68 rows <= 1e-3 (the 67-row short-context run: 1.29e-4);
3. no NaN.

Reported, not judged: worst |dlogit|, max KL, the same numbers with `CROW_KV=bf16`, and the old prefill attention
(`CROW_P2_FA=0`) against the new tiled kernel.
