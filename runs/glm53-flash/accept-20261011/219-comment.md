## GPU acceptance (2026-10-11, robin's GPU go)

**Conditions:**
- Software: local main `2a18fbc`, RTX 5090, Windows 11, `runs/glm53-flash/quick/arm4.env`, `serve --port 8099`, `CROW_CONTEXT` unset (200,000), MTP off.
- Isolation: GPU lock held, nothing else on the GPU, Crow closed.
- Boots: a fresh boot per replay (21 s to `/health`). Arms alternated off, on, off, on, off, on, where on = `CROW_GLM_ANSWER_REUSE=1`.
- Request 1: t2's first request, rebuilt byte for byte from Crow's `session.json` msgs 0-1, Crow's 31 tools, G5's sampling fields, seed 1145815877.
  - Teacher-forced with `crow_force_ids` = 7,547 ids: the canonical ids of msg 2's answer from the offline render (7,545), with the id at answer offset 6,669 (`'XL'` 36345) replaced by `'X'` 55 + `'L'` 43, then `<|observation|>` 154829.
  - Context of the split: `' "'` `'IV'` **`'XL'`** `'CD'` `'M'`.
- Request 2: msgs 0-3 with `temperature` 0, `logprobs: true`, `top_logprobs` 20 and `crow_force_ids: []` (to get `crow_id` per generated id), `max_tokens` 4096.
- Raw: `runs/glm53-flash/accept-20261011/219/` (untracked: `rows.jsonl`, serve logs, `analysis.txt`, `force219.json`). Scripts: `replays.py`, `force219.py`, `bodies.py`, `render.py`.
- An earlier start of this series was stopped after its first request on robin's change of plan. Its data is kept in `219/aborted-1/` and is not counted.

**Request 2 per replay:**

| replay | `[cache]` | prompt / cached / prefilled | prefill_ms | prompt phase (NVMe reads, staged calls) | generated, finish |
|---|---|---|---|---|---|
| off r1 | WARM L 13125 (held 14002), P 6456 | 14,066 / 6,456 / 7,610 | 7,455.6 | 5,314, 0 | 1,891, tool_calls |
| on r2 | realigned 13125..14001 → held 13125..14002 (3,028 B); **LIVE L 14002** | 14,067 / 14,002 / **65** | 7,713.4 | 5,610, **42** | 1,939, tool_calls |
| off r3 | WARM L 13125 (held 14002), P 6456 | 14,066 / 6,456 / 7,610 | 7,401.4 | 5,314, 0 | 1,938, tool_calls |
| on r4 | realigned, LIVE L 14002 | 14,067 / 14,002 / **65** | 6,821.3 | 5,610, 42 | 1,938, tool_calls |
| off r5 | WARM L 13125 (held 14002), P 6456 | 14,066 / 6,456 / 7,610 | 7,385.0 | 5,314, 0 | 1,938, tool_calls |
| on r6 | realigned, LIVE L 14002 | 14,067 / 14,002 / **65** | 9,716.4 | 5,573, 42 | 1,971, tool_calls |

Request 1 was the same in all six replays: COLD, 6,456 rows, prefill 6,660-6,841 ms, 7,546 generated, 68.2-69.0 ms/token.

**Criteria as written:**
- **B1: PASS, 3/3.**
  - After arm: F = 65 ≤ (14,067 − 14,002) + 64 = 129, on every replay. The `[cache]` line reads `LIVE` and names the realigned span: `realigned: ids 13125..14001 of the prompt replaced by the held ids 13125..14002 (same 3028 bytes, non-canonical split), walk 0.031-0.046 ms`.
  - Before arm: L 13,125 lies inside the answer, which is G5's row-5 shape (L 13125, prefilled 7,610). The replay is valid.
- **B2: FAIL as written.**
  - Same in all 9 off×on pairs: `finish_reason` (`tool_calls`) and the content text (empty; 84 reasoning chars).
  - Not the same: the tool calls. Every replay made 4 calls with the same names: `goal_step`, `write_file`, `write_file`, `run_command`. The arguments of `goal_step` and `run_command` are byte-equal in all 6 replays. The two `write_file` contents (code) differ: 4 and 6 distinct texts over 6 replays. The first differing generated id is at 107-755.
  - Within one arm the greedy answers also differ (off r1 vs off r3: first id diff 135; on r2 vs on r4: 107). As in #215, free decoding is not reproducible run to run, so byte-equal calls cannot be expected between arms either.
  - Prompt-end row, read from `top_logprobs` 20 at the first generated position (the raw distribution; a partial view of the row, cosine over the shared entries' probabilities):
    - Top-1 is the same in every pair (`Now`, 7039).
    - Cosine vs the off arm: on r2 **0.998829** (< 0.9999, FAIL), on r4 0.999995 and on r6 0.999966 (pass).
    - The off arm's row is identical on all 3 boots (cosine 1.0, the same logprob −0.0732491), because it is written by the prefill. The on arm's row follows rows written by `decode_step` and varies boot to boot: top-1 logprob −0.1207 / −0.0761 / −0.0818.
- **B3: FAIL.** The median `prefill_ms` of request 2 is **7,713.4 after vs 7,401.4 before** (+0.31 s). G5 was 7,927.7.
  - Cause, from the `[glm5] prompt phase` line: the 65-row LIVE call opened a **staged forward on all 42 MoE layers**: `staged calls 42, unfit 0`, NVMe 5,573-5,610 reads ≈ 53 GB. That costs as much as the 7,610-row prompt: 5,314 reads, `NVPF calls 42`, `staged calls 0`.
  - 65 rows × top-8 = 520 picks ≥ `stage_min` 512. This is #216 Evidence 3's edge (a staged forward re-streams every non-VRAM expert), here with fit > 0.
  - So (A)+(B) cut the rows from 7,610 to 65, but not the seconds while `CROW_GLM_SMALL_PROMPT` is off. #216's route covers this call (65 < 1,024 rows). The combined run of this block measures the two together.
- **B4:** host tests, green in the implementation run, not re-run here. The ignored GPU test `glm5_answer_reuse_gpu` passed (2 tests). LIVE vs cold prefill gives cosine 1.000000000 and the same top-1, with greedy ids equal over 8. Bits: 2,041 of 2,048 logits differ at chunk 12, 0 of 2,048 row by row (`gputests/219.log`).
- **B5: PASS.** Boot line: `[serve] CROW_GLM_ANSWER_REUSE=1 (#219): ... (LIVE, 0 B host RAM, no rollback) ...`. `CROW_RAM_MARGIN_GB` is unchanged.

No threshold was changed.
