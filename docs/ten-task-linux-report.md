# The ten-task Linux record — grading report (#80, finished under #87 phase 1)

Measured 2026-09-19, graded 2026-09-20. Grader: ZCode subagent, pointwise, blind first.
Worksheet: `docs/ten-task-grades-worksheet.json` (every requirement, every answer, one-line
justification each). Summary doc: `docs/ten-task-linux.md` §0 and §4.

## 1. Provenance and seal

- Grading input: `decode_out/ten-task/blind/answers.md` — 60 answers under opaque 8-hex ids
  (10 tasks × 6), written by `tools/ten-task-run.py blind --labels A-crow,B-llama`.
- Seal verified BEFORE grading and re-verified after opening:

  ```
  1214d0f3bdcce72f7fa0029af6c7463565db8ca19fa9bb16c5ffd5da58e7813b  blind/idmap.json
  ```

  matches `blind/idmap.json.sha256` exactly (expected prefix `1214d0f3bdcce72f`).
- Data completeness: A-crow 30/30 records, B-llama 30/30, A-crow-32k 30/30, zero request
  errors; `blind/answers.md` holds exactly the 60 answers of the two 16k arms (38 empty).
  `run-B32.log` is 0 bytes — the aborted B-llama-32k control (see §8).

## 2. Protocol as executed

1. Seal verified (above).
2. All 60 blind answers graded pointwise against `checklist.json` — 43 requirements over 10
   tasks, `met` / `partly` / `not` each with a one-line justification quoting the answer.
   The id map was NOT opened during grading. Mechanical checks (compile/run harness, number
   matchers, language, word counts) were re-run by the grader on the blind file alone, so no
   arm information entered the grading step.
3. Grades fixed and written to `docs/ten-task-grades-worksheet.json` (blind section).
4. Only then: the A-crow-32k arm (not part of the sealed set — it was generated after the
   seal) graded against the same rubric, arm identity known by construction, in the
   worksheet's `supplement_A-crow-32k` section.
5. Only then: `idmap.json` opened (hash unchanged), ids joined back to arms, aggregates
   computed.
6. Verdicts: `pass` = every requirement met (share_met 1.0); `fail` = nothing met (all 38+9
   empty answers land here by rule); `partial` = between.

Re-grade consistency (protocol §3.2.4): the runner's 20 % sample (`blind/regrade.md`, 12
answers, fresh order) reproduced 12/12 frozen verdicts and share_met values. Caveat: same
grader, same session — see §8.

## 3. Totals

| arm | n | Pass | Partial | Fail | empty answers | finish `length` | finish `stop` | share_met mean |
|---|---|---|---|---|---|---|---|---|
| A-crow @16384 (blind) | 30 | 2 | 7 | 21 | 21 | 24 | 6 | 0.221 |
| B-llama @16384 (blind) | 30 | 9 | 3 | 18 | 17 | 19 | 11 | 0.388 |
| A-crow @32768 (supplement) | 30 | 7 | 14 | 9 | 9 | 10 | 20 | 0.606 |

- B-llama's 18 fails = 17 empty + 1 non-empty (`t1b-read-lang` seed 2103: a 111-character
  fragment cut off inside its first heading, `finish length` at exactly 16 384 completion
  tokens).
- A-crow's 21 fails are ALL empty answers; its 6 `finish stop` generations are the only ones
  that produced full answers.

English-only (nine tasks, 27 per arm; the German `t4-prose` is reported separately, robin
2026-09-19):

| arm | English Pass | Wilson 95 % | English share_met | mean answer tok | mean thinking tok | mean wall s |
|---|---|---|---|---|---|---|
| A-crow @16384 | 1/27 | 0.0066–0.1828 | 0.148 | 213 | 14 914 | 282 |
| B-llama @16384 | 6/27 | 0.1061–0.4076 | 0.320 | 638 | 13 265 | 306 |
| A-crow @32768 | 6/27 | 0.1061–0.4076 | 0.576 | 1 227 | 22 208 | 439 |

German `t4-prose` (3 per arm): B-llama `pass/pass/pass`; A-crow `partial(0.875)/partial
(0.75)/pass`; A-crow-32k the same three verdicts — the texts are byte-identical to the 16k
arm (§6).

## 4. Per-task × per-arm verdicts

`P` pass, `p` partial (share_met in parentheses), `F` fail with content, `E` empty fail.
Seeds in order 2101 / 2102 / 2103.

| task | A-crow @16k | B-llama @16k | A-crow @32k |
|---|---|---|---|
| t1-read | p(0.6) p(0.6) E | p(0.8) E P(1.0) | p(0.6) p(0.7) p(0.8) |
| t1b-read-lang | p(0.92) p(0.67) E | p(0.92) p(0.92) F(0.0) | p(0.92) p(0.92) p(0.92) |
| t2-write | p(0.22) E E | E E E | P(1.0) p(0.94) E |
| t2b-write-refactor | E E E | E E E | E p(0.9) p(0.7) |
| t3-debug | E E E | E E E | E E E |
| t3b-debug-syn | E E E | E E E | P P P |
| t4-prose (German) | p(0.88) p(0.75) P | P P P | p(0.88) p(0.75) P |
| t5-agent | P E E | E P P | P p(0.88) p(0.88) |
| t6-reason | E E E | E E E | E P(1.0) p(0.4) |
| t6b-reason-multi | E E E | P P P | E E E |

Task-level reading:

- **t6b-reason-multi** — whenever an answer appeared it was arithmetically perfect: B-llama
  3/3 full pass (all five steps exact, including the 728.25 MiB margin). The crow arm never
  produced one at either budget: all six crow cells are empty (thinking consumed 16k AND 32k).
- **t3b-debug-syn** — zero answers from both engines at 16k; the crow 32k arm then produced
  three full passes (host→device release/acquire and the ring-ABA family, concrete
  interleavings, correct `==`/wrap analysis).
- **t3-debug** — zero answers everywhere, all budgets, all seeds, both engines. The hardest
  budget-eater of the set.
- **t4-prose** — the only task where both engines always answered; B-llama 3/3 flawless
  German within 400 words with correct numbers; the crow arm's two partials miss rubric
  clauses (driver-spill argument; the 312.44 MiB per-slot constancy framing), not facts.
- **t5-agent** — every answer that existed passed except the two crow-32k one-liners that mix
  CMake configure and build options in one invocation (verified against cmake 4.4.3: `cmake
  -S . -B x … --build --target t` and `… --target t --parallel` both error "Unknown
  argument"; the two-step form exits 0) — plausible-looking, non-running command lines.
- **t1/t1b** — reading-comprehension answers are long (5–9 k chars) and good
  (share_met 0.67–0.92) but almost never finished inside 16 384 tokens; the visible crow 16k
  texts are the exact prefixes of the 32k texts.

## 5. Notable failure modes

1. **Budget exhaustion into thinking (the dominant mode).** 38/60 blind answers empty; four
   whole task classes answerless at 16k on BOTH engines. This is what "thinking `high`, no
   budget, `max_tokens` 16384" does at temperature 1.0: the engine thinks until the cap and
   never opens the answer block.
2. **Unbounded think loops that 32k does not fix.** 9 crow cells empty at 32 768 too
   (`t3-debug` ×3, `t6b` ×3, `t2w`-2103, `t2b`-2101, `t6r`-2101; thinking 50k–128k chars).
   No token budget helps these — they need the #81 reasoning-budget mechanism (a cap that
   closes the think block).
3. **One premature stop.** A-crow `t1-read` 2101 ends mid-sentence ("The file size is") with
   `finish_reason stop` at 14 298 completion tokens — the model emitted its stop token early,
   at BOTH budgets with byte-identical text. One case in 90 generations; B-llama zero.
4. **Token-level corruption in long crow answers (32k arm).** `uint644_t` / `uint332_t`
   (twice/three times), stray digits inside prose ("fit.9727 Then", "refactoring.8763"), a
   degenerate `#define push_back push_back` argued for in several paragraphs. Both t2b
   answers therefore do not compile although structurally complete. Same family as
   `docs/quality-probe.md`'s non-word finding for CNQ4.5-M.
5. **Confident but invalid tool syntax.** The two t5 crow-32k answers build CMake one-liners
   that do not parse (verified, see §4) while explicitly listing correct-sounding assumptions
   — hallucinated command syntax under a correctness-aware framing.
6. **Runner false negatives the blind grading caught.** The mechanical number matcher does
   not recognise LaTeX `{,}` separators; two t6b answers with every value correct were marked
   as missing four/six values. Corrected by reading; the runner was left untouched.

## 6. Seed determinism across `max_tokens` (crow arm)

For all 30 (task, seed) cells, comparing A-crow@16384 with A-crow@32768:

- 15 cells byte-identical (every cell where the 16k generation had already ended — including
  the empty-because-length ones that stay empty at 32k),
- 15 cells where the 16k text is an exact character-level prefix of the 32k text
  (16k `finish length` → 32k `finish stop`, e.g. `t1-read` 2102: 586 → 9 572 chars),
- 0 divergences.

`serve` therefore does not perturb sampling when `max_tokens` changes: the same seed walks the
same token stream. Consequences: (a) the 16k-vs-32k quality gap is pure budget truncation;
(b) re-running a seed reproduces the same sample — the three-seed band is exactly three
samples, and replication adds no information; (c) as a side effect, byte-comparing the 32k
arm against the blind file identifies the crow blind ids — this comparison was made only
AFTER the blind grades were frozen (worksheet `_order` note) and is disclosed here as a
protocol deviation without grade impact. Cross-engine check: zero identical non-empty
answers between A-crow and B-llama.

## 7. The five most important findings

1. At the real operating row with thinking `high` and NO budget, the ten-task record is
   mostly a measurement of thinking length: 63 % of blind answers empty, both engines, and
   the 16k "B-llama wins" (9 vs 2 passes) is largely "B-llama thinks shorter" (13 265 vs
   14 914 mean thinking tokens at the same cap).
2. Give the crow engine double the budget and it draws level on passes (7; English 6/27 =
   B-llama's 6/27) with the highest share_met of the record (0.606) — the engine's answers
   are not the 16k problem, the budget was.
3. 9 crow generations never answer at ANY budget — the unbounded think-loop class that
   motivates #81's reasoning cap; `t3-debug` and `t6b-reason-multi` are empty on all seeds at
   32 768 (up to 128k thinking chars).
4. serve is seed-deterministic across `max_tokens` (30/30 identical-or-prefix, zero
   divergence) — a re-run is the same sample; only new seeds widen the record.
5. Quality signals independent of budget: one premature mid-sentence stop in 90; token-level
   corruption (`uint644_t`, embedded digits) in the crow arm's long answers, absent from
   B-llama's; B-llama flawless 3/3 on the German prose task vs crow 1/3 at equal budget.

## 8. Limits and data gaps

- **B-llama @32768 is a missing control** — aborted by the user ("sehen reicht"), no records
  (`run-B32.log` empty). The budget-sensitivity conclusion rests on the crow arm's own
  16k→32k comparison plus B-llama's 16k numbers, not on a symmetric design. Documented as a
  gap, never as a result.
- The sealed blind set covers only the two 16k arms; the 32k arm was graded non-blind
  (necessarily — it postdates the seal) and is labelled as a supplement everywhere.
- One grader, one session; the 12/12 re-grade agreement cannot exclude memory contamination.
  Pointwise only, no pairwise judging (position bias avoided by design, not measured).
- Three seeds per cell; overlapping Wilson intervals for the two 16k arms (0.7–18.3 % vs
  10.6–40.8 %) — "B-llama ahead at 16k" is a point estimate, not a separated result.
- The mechanical matcher's LaTeX `{,}` blind spot (two false negatives, hand-corrected).
- Protocol deviation disclosed: the 32k-vs-blind byte-identity comparison (§6) partially
  de-anonymised crow blind ids after the grades were frozen; worksheet section `_order`
  records the ordering guarantee.

## 9. Where everything is

| artefact | path |
|---|---|
| per-answer, per-requirement grades | `docs/ten-task-grades-worksheet.json` |
| this report | `docs/ten-task-linux-report.md` |
| summary + limits in the record doc | `docs/ten-task-linux.md` §0, §4 |
| graded blind answers / seal | `decode_out/ten-task/blind/answers.md`, `blind/idmap.json(.sha256)` |
| re-grade sample | `decode_out/ten-task/blind/regrade.md` |
| rubric | `decode_out/ten-task/checklist.json` |
| raw generations | `decode_out/ten-task/{A-crow,B-llama,A-crow-32k}/records.jsonl`, `run.json`, `tokens.json` |
