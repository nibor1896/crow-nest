# PREREG: MTP speculative decoding for the dense 27B (crow-nest #95, Crow #300 phase 2)

Written 2026-09-27 before any MTP code or measurement. Design of record: crow-nest #95 comment of 2026-09-26
(chain draft, adaptive k in {0..3}, per-position GDN state slots, M-token verify, llama.cpp acceptance rule,
presence penalty per position, CUDA graphs per k). robin's decisions: BF16 MTP head (option A, 2026-09-27); while
an image is generated the VRAM lend (#117) frees the MTP weights and GDN slots, the KV stays.

## Forward of record (vLLM `qwen3_5_mtp.py` forward, llama.cpp `qwen35.cpp` graph_mtp)
draft for position p+1: x = fc(cat[pre_fc_norm_embedding(embed(t_{p+1})), pre_fc_norm_hidden(h_p)]) with h_p the main
model's hidden after `model.norm`; x -> `mtp.layers.0` (gated full attention + dense SwiGLU, its own KV cache,
RoPE at position p, the vLLM convention) -> `mtp.norm` -> the shared lm_head. Step j > 1 feeds the previous MTP
output (after `mtp.norm`) as h. All norms (1 + w).

## C1: the MTP forward (step 1)
- Reference: `oracle/ref_qwen35_mtp.py`, f32, the container's weights (CNQ, BF16 MTP), teacher-forced over a
  sequence: MTP logits at every position p (pair h_p, t_{p+1}).
- Engine: the same rows through the engine's MTP forward, BF16 KV.
- PASS: argmax equal on every row whose reference top-2 margin is >= 0.01, and mean KL(ref || engine) <= 1e-3,
  on the 63-token parity prompt and on tf298 (298 rows).
- Reported, not gated: the teacher-forced greedy acceptance of the head (MTP argmax at p == main argmax at p+1)
  on tf298, t2b and the five prose sets.

## C2: the correctness contract (steps 2-4)
- Greedy (`CROW_SAMPLE` off, the decode bin): the generated token ids with MTP on are BYTE-IDENTICAL to MTP off,
  512 tokens each, on 6 prompts: the parity prompt, tf298's prompt, t2b's prompt, `ids-apollo`, one prose window,
  one 8k window of a Crow session. The verify pass is built batch-invariant (every row computed with the decode
  path's own reduction order) so that this can hold; any divergence is a FAIL, not noise.
- Sampled (Crow's rows): the llama.cpp acceptance rule is identical in distribution to plain sampling; checked by
  the per-position token-frequency test of 20,000 draws on 3 fixed contexts (chi-square p >= 0.01 for each).

## S: speed (RTX 5090, `CROW_MMA=1 CROW_GRAPH=1`, BF16 KV, the new DIAG container; alternating MTP off/on, n = 3)
- S1 greedy, `ids-apollo` (context ~191), 512 tokens: tok/s >= 1.5 x MTP off (73.5 -> >= 110).
- S2 greedy, 30k-token prompt, 256 tokens: tok/s >= 0.95 x MTP off (adaptive k may not lose).
- S3 Crow's non-thinking row (0.7 / 0.8 / 20 / 0 / presence 1.5) and thinking row (1.0 / 0.95 / 20 / 0 / 0) on
  `ids-apollo`: tok/s >= 1.2 x MTP off; acceptance per position reported.
- Reported beside every number: llama.cpp 27B `UD-Q4_K_XL` on this card, 66.5 tok/s plain, 123.05 with MTP.

## V: VRAM (the Qwen-Image room, context 100,000)
- MTP adds <= 1.7 GiB after load (weights 0.79 + MTP KV 0.38 + GDN slots 0.42 GiB computed).
- The lend frees >= 1.2 GiB of it on request; after the return decode tok/s within +- 5 % of before, and the
  greedy contract C2 still holds on the parity prompt.
