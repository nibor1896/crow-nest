# Crow #245 — reasoning budget 1024 vs 2048 on the 27B, Windows — criteria fixed before any round

Written 2026-09-30 13:25 CEST (11:25 UTC), before the replay session is on this machine and
before any request of this series was sent. Seen so far: nothing of this series. What is known
beforehand is MEAS-0923 (Linux, Qwen3.8-Flash-Next CNQ4.5-M, budget 1024 only, Crow #245
Evidence) and the 27B's short-prompt decode speed under Windows (79.2 / 79.6 / 83.6 tok/s,
n = 3, 2026-09-30).

## Question
Does raising `reasoning_budget_tokens` from 1024 to 2048 on the 27B at Crow's operating point
cost tool calls, correctness or too much time? robin's choice 2026-09-30: grid 1024 and 2048
(the ticket's 4096 / 16384 only if 2048 wins, see Follow-on).

## This is a NEW operating point, not a repeat of MEAS-0923
| | MEAS-0923 (the ticket's cause) | this series |
|---|---|---|
| model | Qwen3.8-Flash-Next CNQ4.5-M | Qwen3.8-27B CNQ4.5 (`converter/Qwen3.8-27B-CNQ4.5.cnq`) |
| machine | Linux | Windows 11, RTX 5090, 63 GB RAM |
| engine | crow-nest branch meas-0923 (`e379361`) | crow-nest `main` `27874fa` (contains `e379361`), `engine/target/release/serve.exe`, n_ctx 65,536, `--slot-save-path decode_out/session`, no CROW_* env (tool grammar ON by default, `CROW_TOOL_GRAMMAR`) |
| Crow body | Crow of 2026-09-22 | Crow v2.8.3 installed, `%LOCALAPPDATA%\Crow\cli\crow_core.py` sha256 prefix `ca5e8e761da7` |
K=69 seed 2 of MEAS-0923 cannot be reproduced here; its prompt is the same, the model is not.

## Data
- Replay session: the 2026-09-22 diorama `session.json` of the Linux machine, sha256
  `559bb1ed8e17ec4beecfc9ed2aba33f2538e1b53df446921160019441b3729a8` (the probe preset's
  pin), copied to `decode_out/meas-245/session-0922.json`. A copy with another sha256 is not used.
- Head: `tools/corpora/91-replay-diorama-0922-head.txt` (sha256 `bd462012b971…`, the preset's).
- Points: K = 26, 69, 71, 109, 139. A point is IN when messages[K-1] is a user or tool turn and
  its prompt fits: prompt_tokens + max_tokens 16,384 <= n_ctx 65,536, i.e. prompt <= 49,152,
  read from the warm-up request of that point (below) before any counted round. An excluded
  point is named in the record with its prompt_tokens.

## Instrument
`tools/corruption-replay-probe.py` with Crow #245's change (uncommitted at writing time, lands
with this file): `--served-name auto` (the body today's window builds: /props name
`Qwen3.8-27B-CNQ4.5.cnq` -> manifest entry `qwen38-27b-cnq`: temperature 1.0, top_p 0.95,
min_p 0.0, top_k 20, presence 0.0, reasoning_effort `high` (serve: xhigh), budget 1024 +
Crow's REASONING_BUDGET_MESSAGE, max_tokens 16,384, streamed); per round `reasoning_chunks`,
`budget_closed`, and for rounds without a call the full `content` + reasoning tail.
`--home /home/nibor1896`. Probe tests on this machine: 16 / 19, the 3 red are the
filesystem-confirmed `digit_near_miss` cases (POSIX paths), red before the change as well.

## Arms (one variable)
- A: `reasoning_budget_tokens` 1024 (the manifest value, sent as is)
- B: `reasoning_budget_tokens` 2048 (`--sampling '{"reasoning_budget_tokens": 2048}'`)
Everything else byte-identical, max_tokens 16,384 in both (>= 2048 + 8192, the ticket's room).

## Design
- Per point: one warm-up request (arm A body, max_tokens 1, not counted) so every counted round
  starts on a warm prefix cache; then seeds 0..7 and one greedy round (`temperature` 0) per arm.
- Arms alternate within a point, ABBA by seed (even seed A first, odd seed B first; greedy A
  first), so drift in machine state falls on both arms.
- One engine only: before the series `Get-Process serve` = the one `serve.exe`, no
  `llama-server`, no `sd-server`; nothing else on the GPU. Checked again after the series.
- A failed round (HTTP error, timeout) is recorded as failed and is not re-run.
- Runner: `decode_out/meas-245/run_series.py`; every probe JSON and the engine.log slice
  (`%LOCALAPPDATA%\Crow\logs\engine.log`, UTC) of the series are kept in `decode_out/meas-245/`.

## Metrics, per arm and per point (counts with n)
1. rounds without a tool call (`n_calls` 0)
2. corrupt calls (json_invalid, placeholder, control_char, home_mismatch; `digit_near_miss`
   cannot be confirmed on Windows and stays info — stated limit)
3. schema-wrong calls (unknown_tool, unknown_arg, missing_required, type_mismatch)
4. budget closes (`budget_closed`), cross-checked against serve's
   `[chat] reasoning budget N spent` lines
5. reasoning tokens = `reasoning_chunks`, cross-checked against serve's `reasoning chunks N`
6. wall clock per round (`seconds`, warm cache), median and range

## Decision rule (from the ticket, narrowed to two values)
2048 replaces 1024 in the `qwen38-27b-cnq` entry only if ALL hold, summed over the points that are IN:
- a. rounds without a call (B) <= rounds without a call (A)
- b. corrupt calls (B) <= corrupt calls (A)
- c. median wall clock per round (B) <= 1.5 x median (A)
Otherwise 1024 stays, with this series as its `_reasoning_budget_status` evidence.
Schema-wrong calls and budget closes are reported, not judged.

## Stated limits, before the result
- n = 9 rounds per point and arm (45 per arm at most). A difference of one or two no-call rounds
  is not a rate; the rule says "not worse", it does not claim "better". Wilson 95 % intervals
  are reported beside every count.
- One session (one task: the diorama), one day, one machine, one model.
- The result speaks for the 27B entry. Flash-Next's 1024 (`flash-next-cnq45-m`,
  `flash-next-q2-k-xl`) is not decided by it.

## Follow-on
If 2048 passes the rule, 4096 is measured the same way against 2048 — written down as an
amendment to this file before its first round. If it fails, the series ends.
