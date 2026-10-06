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

---

# AMENDMENT 1 — 2026-09-30 ~13:30 CEST (committed d477684, 13:31), before any round on the new data

Cause: the 2026-09-22 diorama session (sha256 `559bb1ed…`) exists only on the Linux
installation of this dual-boot machine; robin will not boot Linux for it (2026-09-30). No
request of the series has been sent on any data. Seen so far: only the plumbing selftest
(`selftest/`, K=4 of the session below, seed 0, one round per arm: both 1 call, 41 reasoning
chunks, no close, 1.7 / 1.6 s) — K=4 is excluded from selection for that reason.

## Data (replaces "Data" above)
- Replay session: `decode_out/meas-245/session-0915.json`, a byte copy of
  `%LOCALAPPDATA%\Crow\session\rollover-20260915-092436.json`, sha256
  `6ee99879f3982c78e5a5e9118ef50e57834d89201d6393bec15c76fc0f46a834`: robin's Windows work
  session of 2026-09-15 after a rollover, 274 messages, 122 assistant turns with tool calls.
  Chosen by rule, not by content: the most recent Windows session file larger than 100 kB.
  Head: the session's own messages[0] (no head file). History written by another model than
  the 27B — as in MEAS-0923's replay.
- Nothing is known about where the budget binds in it, so the points are SELECTED by a
  screening pass whose rounds are never counted.

## Screening (new, before the counted rounds)
1. Candidates: every K with messages[K-1] a user or tool turn and messages[K] an assistant turn
   with tool calls, K >= 5, ascending.
2. Per candidate: a warm-up (arm A body, max_tokens 1) gives prompt_tokens; the first candidate
   with prompt > 49,152 ends the list (prompts grow with K).
3. Per fitting candidate: ONE screening round, arm A (1024), seed 100 (not a counted seed).
4. Points = the candidates whose screening round has `budget_closed` True. More than 5: take 5
   spread evenly over them in K order (indices round(i x (n-1) / 4), i = 0..4). 1 to 5: all.
   0: the result is "1024 does not bind on the 27B in this session"; no counted rounds, 1024
   stays with that as its evidence, and the ticket gets that sentence with the screening table.
5. Screening rounds are reported (count, closes, no-call) but never enter the decision.
Selection is by arm A's own close at a seed outside the counted ones, so it favours points where
1024 binds — the question the ticket asks — without looking at any arm B round.

## Grader on this session (stated limit)
The paths are Windows paths (`C:\Users\robin\…`). The probe's path checks read POSIX paths
only, so `home_mismatch` and `digit_near_miss` cannot fire here; "corrupt" is json_invalid,
placeholder and control_char only. `--home C:/Users/robin`.

Design, arms, metrics, decision rule, limits and follow-on above are unchanged.

---

# AMENDMENT 2 — 2026-09-30, the closing sentence at 1024, before any round of it

Cause: robin, 2026-09-30, after the 1024/2048 result (1024 stays; its one no-call round,
K=45 seed 0, followed the cut + sentence A with a written "final answer" and no call): "mach
satz B test bei 1024". The ticket's second question. Seen so far of this question: that one
round and MEAS-0923's K=69 seed 2 — nothing with sentence B.

## Arms (one variable: the sentence; budget 1024 in both)
- A: Crow's REASONING_BUDGET_MESSAGE, sent as is:
  `"\n\nThat is enough analysis. I will now write the final answer for the user.\n"`
- B: the ticket's sentence B, via `--sampling '{"reasoning_budget_message": ...}'`:
  `"\n\nThat is enough analysis. I will now act on it.\n"`

## Data and points
Same session (`session-0915.json`, sha256 `6ee99879…`), same two points the 1024/2048 screening
selected: K=10 (19,172 prompt tokens, 1024 closed 6 of 9 there) and K=45 (41,254, 9 of 9). No
new screening.

## Design
Seeds 0..31 + greedy per arm and point (33 x 2 x 2 = 132 counted rounds), ABBA by seed as
before, a warm-up per point, `serve.exe` alone. Arm A at seeds 0..7 + greedy repeats the
1024/2048 series' A rounds with the same body: serve is seed-deterministic (crow-nest
docs/acceptance/issue-87.md), so the repeats are a determinism check (reported: how many of
the 18 match the earlier round's finish, call count and reasoning chunks) and are counted like
every other round. Output: `decode_out/meas-245/sentence/`.

## Metrics
As above; the one that decides is rounds without a tool call. Rounds where the budget did NOT
close carry no sentence and are the same prompt and seed in both arms — so the comparison that
matters is among the CLOSED rounds; both are reported (all rounds, and closed rounds only).

## Decision rule
B is proposed as Crow's REASONING_BUDGET_MESSAGE only if ALL hold, summed over both points:
- a. no-call(B) < no-call(A) — strictly fewer (the ticket: "change the sentence only if B
  reduces no-call rounds at 1024")
- b. corrupt(B) <= corrupt(A) and schema-wrong(B) <= schema-wrong(A)
- c. median s/round(B) <= 1.5 x median(A)
Otherwise sentence A stays. Even if B passes, the change is PROPOSED, not made: the sentence is
one constant for every model and every turn (text answers too, where #176 chose A to stop
mid-word cuts, 2 of 9 -> 0 of 6), and this series measures the 27B's tool turns only. A
one-sided Fisher exact p on no-call among closed rounds is reported beside the counts, not used
as a gate.

## Stated limits, before the result
- Base rate from the 1024/2048 series: 1 no-call in 15 closed 1024 rounds. At ~50 closed
  rounds per arm, A expects roughly 3; a difference of one or two rounds is not a rate.
- One session, two turns, one model, one day; the effect on answers without tools is not
  measured here.

---

# AMENDMENT 3 — 2026-09-30, sentence B on turns that need no tool, before any round of it

Cause: Amendment 2's rule passed (B: 0 of 49 cut rounds without a call, A: 4 of 49), and B is
only PROPOSED because the sentence is one constant for every turn. #176 chose sentence A for
turns that end in a written answer (mid-word starts 2 of 9 -> 0 of 6 without/with it,
2026-08-31, llama arm). robin, 2026-09-30: "mach die kurze prüfung ohne tools". Seen so far on
this question: nothing.

## Data
The 12 prompts of crow-nest `tools/quality-probe-prompts.json` (6 prose de/en, 2 literal-format,
2 JSON, 2 agent-style questions), each as a two-message conversation: messages[0] = the head
Crow sent in `session-0915.json` (its messages[0], unchanged), messages[1] = the prompt's `user`
text. The prompt's own `system` line is NOT sent (a Crow user types only the question). Crow's
tools travel as always (`--served-name auto`); nothing forbids a call.

## Design (paired by determinism)
- Budget 1024 in both arms; arms = Amendment 2's A and B sentences.
- Seeds 0, 1, 2 per prompt (no greedy). Arm A runs first for every (prompt, seed).
- Arm B runs ONLY where arm A's round closed at the budget: where A did not close, the sentence
  was never injected, so B's request would generate the same ids (determinism 18 of 18,
  Amendment 2) and adds nothing. Uncut A rounds are reported as uncut, not as pairs.
- `serve.exe` alone, as before. Output `decode_out/meas-245/notools/`.

## Metrics, on the CUT pairs
1. empty answer: no call and content without a non-space character
2. mid-word start: the answer's first non-space character is a lowercase letter or one of
   `, . ; : ) ]` (a word or sentence the think block was in the middle of) — mechanical; the
   first 80 characters of every answer are listed for the eye as well
3. `finish length` (the answer ran into max_tokens 16,384)
4. a tool call instead of a written answer, per arm; and the discordant pairs: B called where
   A wrote the answer, A called where B wrote it
5. wall clock per round

## Decision rule (all must hold, summed over the cut pairs)
- a. empty(B) <= empty(A)
- b. mid-word(B) <= mid-word(A)
- c. finish length(B) <= finish length(A)
- d. pairs "B called, A wrote the answer" <= pairs "A called, B wrote it" + 1
- e. median s/round(B) <= 1.5 x median(A)
All hold -> "no measured harm on turns without tools"; the proposal from Amendment 2 stands and
robin decides. Any fails -> reported with the pairs; B is not proposed as the global constant.
0 cut pairs -> "1024 did not cut on these prompts": no statement about B on text turns.

## Stated limits, before the result
12 prompts x 3 seeds, one model, Crow's head from one session; "no measured harm" is not
"no harm". The prompts' own quality metrics (nonword, repetition, foreign) are not scored here.

## Seen before this amendment was committed (disclosed, rule above NOT changed)
The runner's selftest, written after the rule and before the commit: head of ANOTHER session
(rollover-20260915-081727), prompt de-prose-plakat, seed 0, one pair. A cut and wrote the answer
("Ein Plakat, das eine Ausstellung …", 56.1 s); B cut and called `write_file` (46.4 s). Not
counted (different head, selftest directory). It is the harm criterion d exists for; the +1
allowance in d was written before it was seen and stays.

---

# AMENDMENT 4 — 2026-10-06, Flash-Next's 1024, before any round of it

Cause: every result above is the 27B's and says so ("Flash-Next's 1024 is not decided by this
series"). The 2026-10-06 handover names it as measurement c of the Flash-Next operating point;
robin chose it on 2026-10-06 ("Weiter mit c, dann b"). Seen so far on Flash-Next with this
session: nothing.

## Operating point
- Qwen3.8-Flash-Next CNQ4.5-M via `%LOCALAPPDATA%\Crow\bin\serve.exe` (crow-nest build
  `298f7cd`, code-equal on Windows to v0.9.3), the installed container and sidecar crow0924,
  `CROW_RAM_MARGIN_GB` unset (1 GiB default), n_ctx 200,000 (serve's default for this model),
  tool grammar on, robin's Windows machine (RTX 5090), `serve.exe` the only GPU process.
- The body the installed Crow builds (`%LOCALAPPDATA%\Crow\cli\crow_core.py`, sha256 in
  plan.json), `--served-name auto` -> manifest entry `flash-next-cnq45-m`
  (reasoning_budget 1024, reasoning_fixed high). That crow_core carries Crow `12b88fb`'s rule:
  the budget sentence follows the last message (after a tool result B, otherwise A).

## Data and design
- Same session as Amendment 1 (`session-0915.json`, sha256 pinned above), same arms
  (A1024 = manifest 1024, B2048 = `reasoning_budget_tokens` 2048), same seeds 0..7 + greedy,
  ABBA by seed, a warm-up per point.
- Screening: the SAME 17 candidate turns the 27B screened (K = 10 ... 47), one A1024 round at
  seed 100 each; the points = the turns the budget closed, at most 5, spread as in Amendment 1.
  The list is cut at K 47 (not at Flash-Next's larger window) so the two models are screened on
  the same turns and the series stays under ~2 h. Zero closes -> "1024 does not bind on
  Flash-Next in this session", no counted rounds.
- `max_tokens` 16,384 as Crow sends it; a point is IN when prompt + 16,384 <= 200,000.
- Output `decode_out/meas-245/flashnext/`.

## Metrics and decision rule
Unchanged from the PREREG and Amendment 1 (no-call, corrupt, schema-wrong, budget closes,
reasoning chunks, s/round). 2048 replaces 1024 in `flash-next-cnq45-m` only if, summed over
the points: no-call(2048) <= no-call(1024), corrupt(2048) <= corrupt(1024), median
s/round(2048) <= 1.5 x median(1024). Otherwise 1024 stays, with this series as its
`_reasoning_budget_status`. Changing the manifest is robin's decision either way.

## Stated limits, before the result
n = 9 per point and arm; one session (robin's 27B-era work of 2026-09-15, replayed on
Flash-Next), one day, one machine; Windows paths, so home_mismatch / digit_near_miss cannot
fire. The installed Crow (3.2.4) differs from v2.8.4 in more than the sentence rule; its
crow_core sha256 is recorded.
