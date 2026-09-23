# Acceptance protocol — issue #87, phase 1 (blind grading of the Linux ten-task record)

- Date: 2026-09-20 (night). Executed by: fleet subagent (grading), orchestrator (verification below).
- Scope of phase 1: seal verification, pointwise blind grading of the #80 runs, unsealing, aggregation, report + doc blocks, tracking the runner files. Phase 2 (a fresh record at the full operating row with identical sampler fields on both arms, after #83/#84 land) is NOT part of this acceptance.

## What was delivered

| artifact | what it is |
|---|---|
| `docs/ten-task-grades-worksheet.json` | 90 graded answers (60 blind + 30 supplement), per-requirement `met/partly/not` + one-line justifications, `_order` records the (post-freeze) de-anonymisation check |
| `docs/ten-task-linux-report.md` | full report: tables, failure-mode analysis, instrument limits |
| `docs/ten-task-linux.md` | RESULTS and LIMITS blocks filled (were empty placeholders since 2026-09-19) |
| `tools/ten-task-run.py`, `tools/test_ten_task_run.py` | the record's runner files, now tracked (36 tests green) |
| `CHANGELOG.md`, `README.md` | Measured entry + measured-table row, this date |

## Headline numbers (n = 30 per arm)

| arm | Pass/Partial/Fail | empty | finish length/stop | share_met | English pass (of 27) |
|---|---|---|---|---|---|
| A-crow @16384 | 2/7/21 | 21 | 24/6 | 0.221 | 1/27 |
| B-llama @16384 | 9/3/18 | 17 | 19/11 | 0.388 | 6/27 |
| A-crow @32768 | 7/14/9 | 9 | 10/20 | 0.606 | 6/27 |

B-llama-32k: aborted by robin before any answer — missing control, in LIMITS, never a result.

Load-bearing findings: (1) the 16k deficit is budget exhaustion into thinking (38/60 empty; llama thinks shorter, 13,265 vs 14,914 mean tokens); (2) at 32k crow draws level (English 6/27 both, share_met 0.606 highest); (3) 9 crow cells empty even at 32k = the unbounded think-loop class that #81's `reasoning_budget_tokens` (landed 2026-09-20, after these runs) addresses; (4) serve is seed-deterministic across `max_tokens` — a re-run is the same sample; (5) budget-independent signals: one premature stop in 90; token-level corruption in long crow answers (`uint644_t`, `uint332_t`, degenerate `#define`); llama German prose 3/3 vs crow 1/3.

## Orchestrator verification (done 2026-09-20, all checks passed)

1. **Seal re-verified independently**: `sha256sum decode_out/ten-task/blind/idmap.json` = `1214d0f3bdcce72f7fa0029af6c7463565db8ca19fa9bb16c5ffd5da58e7813b`, byte-exact match with `blind/idmap.json.sha256` (graded-before-unsealed discipline corroborated by the agent's frozen-grades-first protocol and the worksheet `_order` note).
2. **Counts cross-checked three ways**: agent report = RESULTS block table = worksheet JSON (60 blind = 11 pass/10 partial/39 fail = 2+9 / 7+3 / 21+18; supplement 30).
3. **Change discipline**: `git status` shows the agent created exactly the three agreed doc files; engine sources untouched by this unit (other modifications in the tree belong to the parallel #83/#84, #89, #94 units).
4. **Runner tests**: `python3 tools/test_ten_task_run.py` → 36 tests, OK.
5. **Doc guards**: `tools/check_readme_dates.py` 0 offenders after the README row (full gate re-run happens at wave-1 close).
6. **Honesty checks**: missing control documented, not invented; overlapping Wilson intervals disclosed; single-grader caveat disclosed; LaTeX matcher false-negatives hand-corrected and disclosed.

## Test cases for robin's live acceptance (phase 1)

| # | what to check | how | expected |
|---|---|---|---|
| 1 | the seal still holds | `cd /home/nibor1896/Projects/crow-nest && sha256sum decode_out/ten-task/blind/idmap.json && cat decode_out/ten-task/blind/idmap.json.sha256` | identical hashes |
| 2 | the worksheet parses and counts | `python3 -c "import json;w=json.load(open('docs/ten-task-grades-worksheet.json'));print(len(w['blind']),len(w['supplement_A-crow-32k']))"` | `60 30` |
| 3 | spot-check 3 grades against the raw answers | open the worksheet entry for any 3 cells, read the raw answer file it names, check the per-requirement justification | justification matches the text |
| 4 | the doc blocks are filled | `grep -A6 '<!-- RESULTS -->' docs/ten-task-linux.md` | the graded outcome table |
| 5 | the runner tests pass | `python3 tools/test_ten_task_run.py` | 36 tests OK |
| 6 | the record's determinism claim (any cell) | re-run one 16k cell's exact request against serve (same seed/row) | byte-identical answer (finding 4) |

## Remainder of #87 (phase 2 — NOT accepted here)

After #83 (min_p) and #84 (penalties) land: Crow sends the complete identical sampler row to BOTH arms (incl. `min_p`, `reasoning_budget_tokens`), new record at the real operating point, multi-seed, blind grading round 2. Blocked on: engine build of wave 1 + llama GGUF shard 1 (currently incomplete on disk).
