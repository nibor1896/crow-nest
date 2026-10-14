## GPU acceptance (2026-10-11, robin's GPU go), per the amendment above (issuecomment-6109018083)

**Conditions:**
- Software: local main `2a18fbc`, RTX 5090, Windows 11, `runs/glm53-flash/quick/arm4.env`, `serve --port 8099`, `CROW_CONTEXT` unset (200,000), MTP off. Device sampler (#215).
- Isolation: GPU lock held, nothing else on the GPU, Crow closed.
- Boots: a fresh boot per replay (21 s to `/health`). Arms alternated off, on, off, on, off, on, where on = `CROW_GLM_PREFIX_ANCHOR=1`.
- Requests per replay:
  - t1 request 1 (seed 1408639702).
  - t2 request 1 (seed 1145815877).
  - The third chat (t2's system message with the working area `C:\Users\robin\Desktop\g5-run1\t3-accept`, t1's user message, seed 1408639702, `logprobs` + `top_logprobs` 20).
  - All with Crow's tools and sampling fields, `max_tokens` 64.
- Raw: `runs/glm53-flash/accept-20261011/217/` (untracked: `rows.jsonl`, serve logs, `analysis.txt`). Scripts: `replays.py`, `bodies.py`.

**Per replay (prefill_ms; `[cache]`):**

| replay | t1 r1 (6,438, COLD) | t2 r1 (6,456) | third r1 (6,438) |
|---|---|---|---|
| off r1 | 6,975.4 | 6,833.3, COLD L 6043, 6,456 rows | 6,894.3, COLD L 6043, 6,438 rows |
| on r2 | 6,772.6; anchor at 6376 (first `<|user|>`) | 6,894.4, COLD L 6043; anchor at 6043 (L of a cold prompt) | **5,499.9**, WARM L 6043, `anchor Some(6043) (P)`, **395 rows** |
| off r3 | 6,720.4 | 6,825.2, COLD | 6,893.0, COLD, 6,438 rows |
| on r4 | 6,792.6; anchor 6376 | 6,885.1, COLD; anchor 6043 | **5,483.7**, WARM P 6043, **395 rows** |
| off r5 | 6,703.8 | 6,833.5, COLD | 6,874.7, COLD, 6,438 rows |
| on r6 | 6,953.0; anchor 6376 | 6,861.8, COLD; anchor 6043 | **6,876.6**, WARM P 6043, **395 rows** |

**Criteria as written:**
- **A1: PASS, 3/3.** The third chat is WARM in every after replay: prefilled 395 ≤ (6,438 − 6,043) + 64 = 459. Its `[cache]` line names the anchor: `anchor Some(6043) (P)`. t2 request 1 stays COLD in both arms, as the amendment predicted, and anchors at 6,043.
- **A2: PASS.** The third chat's median `prefill_ms` is **5,499.9 after vs 6,893.0 before**, saving **1.39 s per new chat** (estimate was 0.6-4.8 s).
  - The 395-row call still costs 5.5-6.9 s. Its `[glm5] prompt phase` lines:
    - r2/r4: `unfit 42`, 182 sub-batches, NVMe ≈ 4,315 reads (40.9 GB), host wait ≈ 4.94 s.
    - r6: `staged calls 42`, NVMe 5,610 reads (53.1 GB), 6,876.6 ms.
  - So the remaining time is #216's small-call cost.
- **A3: PASS.** t1 request 1's median is 6,792.6 after vs 6,720.4 before = 1.011 (≤ 1.03). Anchored prefill 6,722-6,908 ms; anchor DtoH inside it.
- **A4:**
  - GPU tests: PASS. `kda_gpu_a_cut_scan_is_the_scan` gave 0 output rows differing in bits (cut at 45 of 96). `glm5_prefix_anchor_gpu` passed 2 tests: the anchored prefill equals the plain prefill bit for bit, and from the anchor vs cold the cosine is 1.000000000 with the same top-1 (`gputests/217a.log`, `217b.log`).
  - Replay part (amendment: `top_logprobs` 20 at the first generated position, cosine over the shared entries' probabilities): **cosine 0.9997391 between the arms, FAIL (< 0.9999)**. Top-1 is the same (`The`, logprob −0.0637 off vs −0.0444 on).
  - The row is identical within each arm across all 3 boots, so the difference is the WARM path's own. It is a partial view (18 shared entries of the top-20).
- **A5: PASS.** Boot line: `1 snapshot after each prompt + 1 anchor of 152633344 B (CROW_GLM_PREFIX_ANCHOR, #217)`. `CROW_RAM_MARGIN_GB` is unchanged.

No threshold was changed.
