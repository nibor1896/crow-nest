## GPU acceptance, Step 0 (2026-10-11, robin's GPU go): the A2 rule fires, the A/B was not run

**Conditions:**
- Software: local main `2a18fbc` (`CROW_GLM_SMALL_PROMPT` off), RTX 5090, Windows 11, `runs/glm53-flash/quick/arm4.env`, `serve --port 8099`, `CROW_CONTEXT` unset (200,000), MTP off. Device sampler (#215).
- Isolation: GPU lock held, nothing else on the GPU, Crow closed. A fresh boot (21.3 s to `/health`).
- Requests: G5 t1 requests 1-3, rebuilt byte for byte from Crow's t1 session file (27,159 / 36,479 / 36,867 B, as in the G5 log), seeds 1408639702, 559453182, 772167202, `max_tokens` 16384.
- Raw: `runs/glm53-flash/accept-20261011/216/` (untracked: `rows.jsonl`, `serve-off-r1.log`).
- Ignored GPU test `glm5_int_gpu_the_small_prompt_route_keeps_the_ids`: **passed**. With the route on: no hand-back, no staged forward, 8 guesses. Ids equal to off, logits cosine min 1.0000000 (`gputests/216.log`).

**The three `[glm5] prompt phase` lines (switch off):**

| row | rows at pos | prefill_ms | `[cache]` | prompt phase |
|---|---|---|---|---|
| 1 | 6,438 at 0 | 6,708.0 | COLD | NVMe **5,314** reads (50.3 GB); `unfit 42, NVPF calls 42`; hand-back 147.8 ms (14 chunks); calls 6,511.4 ms |
| 2 | 2,343 at 6,438 | 6,324.3 | WARM L 6517, P 6438 | NVMe **5,203** reads (49.3 GB); `unfit 42, NVPF calls 42`; lend-back 32.0 ms (128 records); hand-back 114.0 ms (10 chunks); calls 6,142.6 ms |
| 3 | 86 at 8,781 | **6,816.9** | WARM L 8797 (held 8797), P 8781 | NVMe **5,573** reads (52.8 GB); **`staged calls 42, unfit 0`**, 42 sub-batches; lend-back 32.1 ms (128 records); hand-back 54.9 ms (5 chunks, 740 written back); calls 6,641.4 ms; decode_ready 86.0 ms |

Decode: row 1 generated 881 tokens at 14.2 tok/s, row 2 16 at 11.3, row 3 265 at 13.8. The sampled answers are not reproducible run to run (see #215).

**A2 rule:** row 3's NVMe reads × 9,474,048 B / 7.2 GB/s = 5,573 × 9,474,048 B = 52,798,869,504 B / 7.2 GB/s = **7,333 ms ≥ 1,400 ms**.
- The rule fires: A1 (≤ 1,600 ms) is out of reach without fewer bytes on the default route.
- As the rule requires, **the after arm was not run**. The ticket needs a dated re-scope before any A/B.

**What Step 0 shows (facts for the re-scope):**
- In this replay, row 3's 86-row call opened a **staged forward on all 42 MoE layers** and read 5,573 NVMe records. That is 259 more than the 6,438-row cold prompt.
  - 86 rows × top-8 = 688 picks ≥ `stage_min` 512. In G5 the same call had fit = 0 (sub-batch path, 2,699.0 ms). Here fit was > 0, and the call cost 6,816.9 ms.
- The same staged re-stream appeared elsewhere in this block on small calls with the switch off:
  - #219's 65-row LIVE call: `staged calls 42`, NVMe 5,573-5,610 reads, 6,821-9,716 ms.
  - #217's 395-row anchored call: `staged calls 42` in 1 of 3 replays (5,610 reads, 6,876.6 ms). The other 2 ran unfit, with 182 sub-batches, ≈ 4,315 reads and a 4.94 s host wait.
- The small route of this ticket (switch on) takes no staged forward. That is the GPU test above. Its speed on the real container is **not measured**: the A2 rule stops the A/B before it. The combined run of this block (all switches on vs off) records the route's numbers per row, report only.
