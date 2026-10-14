## Step 1 measured (2026-10-11, RTX 5090, Windows 11, under `C:\Users\robin\dev\.gpu-lock`)

**Result: none of the three prompt-phase switches closes the long-prompt decode gap. The gap follows NVMe demand reads per decode token, about 2.5 ms per read. The 4,096-token synthetic prompt decodes as fast as the short one.**

Setup: `glm5_run` from `crow-nest-wt-int/engine/target/release` (mtime 2026-10-10 23:14; engine source = `36b0d9d` apart from one test line. The `crow-nest` copy is from 2026-10-09 and lacks the arm3 switches, so it was not used). Env: `arm2.env` + `CROW_GLM_RT2=1` + `arm3.env`, applied as in `quick/bundle.sh`. `--vram-slots 37 -n 256 --reps 3` (rep 1 cold, reps 2–3 warm). n = 2 processes per arm, arm order alternated (forward, then reversed). The real-text arm uses the 3,985 ids of serve request 2 (`docs/glm5-model.md[0:12000]` + question, GLM chat template, thinking closed; the count equals serve's `prompt_n`). Raw data: `runs/glm53-flash/split-20261011/v2/<arm>-r<n>/decode.json` (+ `series.log`).

Decode over 255 tokens, wall tok/s (ms/token), process 1 / process 2. Cold = rep 1, warm = mean of reps 2–3:
- **short** (57 ids): cold 18.01 / 17.93 (55.5 / 55.8 ms), warm 19.9 / 19.7. NVMe/token cold 9.5. ms tokens 1–16 / 17–48 / 49–256: 82 / 64 / 52.
- **long synthetic** (`--prompt-tokens 4096`, 31-id base): cold 18.69 / 18.56 (53.5 / 53.9 ms), warm 21.7 / 22.2. NVMe/token cold 8.7. 85 / 55 / 51 ms.
- **long real** (3,985 ids, baseline): cold 13.21 / 13.14 (75.7 / 76.1 ms), warm 15.26 / 15.32. NVMe/token cold 17.6, warm 13.6. 94 / 66 / 76 ms.
- long real + `CROW_GLM_ARENA_LAZY_REFILL=0`: cold 12.03 / 13.17, warm 11.7 (one rep at 9.64 with tokens 1–48 at 170 ms) / 15.2. NVMe/token 17.6. `refilled` 498, `refill_nvme` 0, TTFT unchanged (6.7 s).
- long real + `CROW_GLM_ARENA_PROMPT_COUNT=0`: cold 9.71 / 11.39, warm 14.8 / 15.2. NVMe/token 17.7 (unchanged), so the cold slowdown is not NVMe.
- long real + `CROW_GLM_ARENA_STAGE_LEND=0`: cold 10.84 / 11.51, warm 13.4 / 13.4. NVMe/token 18.3 cold, 14.5 warm (up by ~0.7–0.9), and 100–170 ms spikes in tokens 1–48.

Arena per rep (cold, all long arms): `enter` 1, `write_backs` 336, `regrown` 9–10, `lazy_slots` 461–498 (0 with `LAZY_REFILL=0`), `lend_write_backs` 0 (128 in warm reps), `refill_nvme` 0. Short: `write_backs` 156, `regrown` 5, `lazy_slots` 313. Long synthetic has the same hand-back counts as long real (336 / 10 / 498) and decodes at the short speed.

Earlier series the same morning (cold, no `--vram-slots`, n = 2, `runs/glm53-flash/split-20261011/A*-r*`), long = first 4,096 ids of `quick/prefill8192.ids` (held-out todo-1006, real text): short 17.37 / 17.62, long 11.81 / 11.92, `LAZY_REFILL=0` 11.98 / 11.96, `PROMPT_COUNT=0` 12.15 / 12.03, `STAGE_LEND=0` 11.74 / 11.81 tok/s wall. NVMe/token 9.8 (short) vs 21.8 (long), tokens 49–256 at 54.5 vs 75 ms.

Reading:
- `LAZY_REFILL`, `PROMPT_COUNT` and `STAGE_LEND` explain about 0 % of the gap. All moves are within ±2 % (series 1) or downward (v2). `STAGE_LEND=1` helps (off: +0.7 NVMe/token, slower).
- Across arms, tokens 49–256 cost 52 → 76 ms when NVMe reads per token go 8.6 → 18.2. That is about 2.5 ms per read, which is one 9.47 MB record at SSD speed. The prompt-phase hand-back is identical in the synthetic arm, and that arm stays fast. So the gap goes with the decode routing of the real-text continuation (VRAM hits 172 vs 209–228 per token), not with prompt length or the borrow/lend/lazy/prompt-count path.
- Not separated: the decode routing of a short prompt with the same document-style content. The synthetic arm's repetitive continuation may route narrowly. The control that would settle content vs placement is the req2 question alone, or a short slice of the same doc.
- Odd: switch-on runs show large run-to-run spread at equal NVMe reads (e.g. `PROMPT_COUNT=0` cold 103 ms vs 88 ms/token at 17.7 reads). Single-run differences under ~15 % on the long real arm are not claims.

Options B/C (#213) are not supported by this split. The step-0 trigger for C2 (`PROMPT_COUNT=0` ≥ 3 % better) is not met.
