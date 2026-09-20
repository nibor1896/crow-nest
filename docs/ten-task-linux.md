# The Linux ten-task record: both engines at the real operating row

Issue #80, measured 2026-09-19 on RTX 5090 / Arch Linux. It changes nothing in the engine, the
converter or the container: it adds `tools/ten-task-run.py`, the records under
`decode_out/ten-task/` and this page.

- `docs/ten-tasks.md` — the fixation of the ten prompts, 2026-09-03.
- `docs/ten-task-expected.md` — the core requirements this record is graded against
  (sections 2.1 to 2.10, Rev2 and Rev3), and the documented prompt contradictions W1 to W8.
- `docs/quality-probe.md` — the reference-free answer-quality probe whose sampling row,
  request plumbing and metrics this runner imports rather than copies.
- `docs/oracle-kld.md` — the reference-based reading of the same two quants against the f32
  oracle, and the finding this page is the answer-side complement of.
- `docs/architecture.md` 5.4 — why the old greedy ten-task record cannot be used here.

<!-- RESULTS -->

Graded 2026-09-20 (issue #87, phase 1). Seal verified before and after: `1214d0f3bdcce72f…`.
All 90 answers (60 blind + the 30 of the A-crow-32k arm) were graded pointwise against
`checklist.json`; the worksheet is `docs/ten-task-grades-worksheet.json`, the full report
`docs/ten-task-linux-report.md`.

## 0. The graded outcome

| arm | n | Pass | Partial | Fail | Fail of it empty | finish `length` | finish `stop` | share_met mean |
|---|---|---|---|---|---|---|---|---|
| A-crow @16384 | 30 | 2 | 7 | 21 | 21 | 24 | 6 | 0.221 |
| B-llama @16384 | 30 | 9 | 3 | 18 | 17 | 19 | 11 | 0.388 |
| A-crow @32768 (non-blind supplement) | 30 | 7 | 14 | 9 | 9 | 10 | 20 | 0.606 |

English-only (nine tasks, 27 generations per arm; `t4-prose` reported separately):

| arm | Pass | Wilson 95 % | share_met mean | mean answer tok | mean thinking tok |
|---|---|---|---|---|---|
| A-crow @16384 | 1/27 | 0.7–18.3 % | 0.148 | 213 | 14 914 |
| B-llama @16384 | 6/27 | 10.6–40.8 % | 0.320 | 638 | 13 265 |
| A-crow @32768 | 6/27 | 10.6–40.8 % | 0.576 | 1 227 | 22 208 |

German `t4-prose`: B-llama 3/3 full pass; A-crow 2 partial + 1 pass — at BOTH budgets the same
three texts (the 32k arm reproduces the 16k arm byte-identically here).

Per-task verdicts (2101 / 2102 / 2103; `P` pass, `p` partial, `F` fail with content, `E` empty
fail):

| task | A-crow 16k | B-llama 16k | A-crow 32k |
|---|---|---|---|
| t1-read | p p E | p E P | p p p |
| t1b-read-lang | p p E | p p F | p p p |
| t2-write | p E E | E E E | P p E |
| t2b-write-refactor | E E E | E E E | E P P |
| t3-debug | E E E | E E E | E E E |
| t3b-debug-syn | E E E | E E E | P P P |
| t4-prose | p p P | P P P | p p P |
| t5-agent | P E E | E P P | P p p |
| t6-reason | E E E | E E E | E P p |
| t6b-reason-multi | E E E | P P P | E E E |

What the numbers say:

1. **The operating row without a thinking budget produces mostly empty answers — on both
   engines.** 38 of 60 blind answers (63 %) are empty: A-crow 21/30, B-llama 17/30. Four task
   classes (`t2b`, `t3`, `t3b`, `t6-reason`) produced zero answers from BOTH engines at 16384,
   all seeds. The failure is "the whole budget goes into thinking", not wrong answers.
2. **At 16384 B-llama wins every aggregate** (9 vs 2 passes, share_met 0.388 vs 0.221, English
   6/27 vs 1/27) — but this mostly measures thinking length, not answer quality.
3. **At 32768 the crow arm reaches B-llama's pass count** (7 vs 9; English 6/27 = 6/27) with
   share_met 0.606, and 9 cells stay empty even there (`t3-debug` and `t6b` all seeds, plus
   `t2w`-2103, `t2b`-2101, `t6r`-2101): unbounded think loops that no token budget fixes —
   the exact class the #81 reasoning budget exists for.
4. **serve is seed-deterministic across `max_tokens`:** for all 30 (task, seed) cells the
   A-crow@16384 answer is byte-identical to (15) or an exact prefix of (15) the A-crow@32768
   answer; zero divergences. The 16k deficits are therefore purely budget truncations, and a
   re-run of the same seed is the same sample, not a new one.
5. **Answer-quality findings independent of budget:** exactly one premature stop (A-crow
   `t1-read` 2101, both budgets: `finish stop` mid-sentence at 14 298 completion tokens);
   token-level corruption in long crow technical answers (`uint644_t`, `uint332_t`, stray
   digits inside words, a degenerate `#define push_back push_back`) — the decode-quality
   signal `docs/quality-probe.md` already measured; B-llama's German prose is flawless 3/3.

## 1. Why a new record was needed

The ten-task record the repository carried is a **Windows greedy realization**, and
`docs/architecture.md` 5.4 measured that this platform does not reproduce it: with no flag set
at all, nine of the ten answers differ from the Windows record `final4`, the cause being the
Windows-vs-Linux logit drift of section 8.7 flipping near-ties over a 1024-token generation.
Worse, **a greedy re-run is the same sample, not a second one** — 5.4's control and switch each
reproduced byte-identically, ten of ten, degeneration included. One greedy draw per task
therefore discriminates nothing, and #40 had already asked for a seed band.

The quant series that ran before this (#75 to #79) settled the reference-based half of the
question: against the f32 oracle, CNQ4.5-M is at least as close to the original as the Unsloth
UD-Q2_K_XL quant on llama-server (82.4 % same top-1 and mean KLD 0.461 against 79.6 % and
0.557 on 607 rows, `docs/oracle-kld.md`), while on `docs/quality-probe.md`'s German prose it
writes twice the non-words. German is dropped as a target (robin, 2026-09-19). What was left
unmeasured is the thing the product is for: **the ANSWERS, on the English workload, at the row
Crow really sends, with thinking on.**

## 2. The conditions

Identical on both arms, and as Crow really runs:

| | value | where it comes from |
|---|---|---|
| prompts | `decode_out/ten-tasks.json`, the frozen Rev2 set | `docs/ten-tasks.md` |
| tasks | 10 — nine English, `t4-prose` German and reported separately | robin, 2026-09-19 |
| seeds | 2101, 2102, 2103 — 3 per task, 30 generations per arm | #80 |
| sampling | temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0 | `tools/quality-probe.py` `ROW`, imported |
| thinking | top-level `"reasoning_effort": "high"` | #74 |
| reasoning budget | NONE sent to either arm | see below |
| `max_tokens` | 16384 on every task and both arms | see below |
| prompt cache | `cache_prompt: false` on the wire | `probes/p5_STATUS.md` |
| conversation | one `user` message, no system prompt, no tools, fresh single turn | #80 |
| context | both servers boot at 200k | `/props` on both |

**`high` is one step on two engines, not one word.** On `serve` it maps to the template's
`xhigh` (`serve.rs` `map_reasoning_effort`, #74, `docs/architecture.md` 7.11.20). On
llama-server the unsloth template renders `high` byte-identically to its own default — Crow's
manifest measured the grouping through `/apply-template` and records it as
`reasoning_groups [["off","high"],["low"],["medium"],["none"]]`, and the same template's error
message names `xhigh` as that default. Both arms therefore think at the template's top step.

**No reasoning budget on either side.** `serve` has none at all. llama-server's 1024-token cap
is a CLIENT field of Crow's manifest (`flash-next-q2-k-xl`, `reasoning_budget: 1024`); the
server command line `start-server.py` builds carries no `--reasoning-budget`, which was
verified by building the command through `crow_core.server_command` before the arm was started.
Both arms think freely.

**`max_tokens` 16384, everywhere.** The frozen budgets in the prompt file (1024 to 1536) were
fixed for thinking OFF (`docs/ten-task-expected.md` Rev2) and cannot hold thinking plus answer.
`serve` caps `max_tokens` at 32768 and clamps it to the free context, so 16384 arrives
unclamped on every prompt here, the two long ones included. **A generation that ends with
`finish_reason length` is recorded as a finding and is never retried.**

**One engine on the card at a time.** Before every start, `ps -C serve,decode,parity,llama-server`
was empty and the card held under 1 GiB. Each arm got one throwaway generation after its start,
which pays the cold prefill and is not part of the record (`docs/ten-tasks.md`, the cold-prefill
rule).

### The two arms

| arm | server | model | port | start |
|---|---|---|---|---|
| A | `serve`, this engine, default operating point, no overlay | `Qwen3.8-Flash-Next-CNQ4.5-M` | 8099 | `tools/serve-linux.sh --port 8099` |
| B | llama-server | `Qwen3.8-Flash-Next-UD-Q2_K_XL` | 8083 | `~/.local/share/crow/venv/bin/python ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl` |

Both were stopped with SIGINT to the real server pid.

## 3. The grading protocol

Two halves, and only the first one needs no judge.

### 3.1 The mechanical checks

Computed by `tools/ten-task-run.py` for every generation on both arms, from the stored text,
and unit-tested without a server or a GPU (`tools/test_ten_task_run.py`):

| task | check |
|---|---|
| `t2-write` | the C++ compiles (`g++ -std=c++17 -fsyntax-only`); when it carries a `main()` it is built and RUN, so the answer's own `assert()` tests decide |
| `t2b-write-refactor` | the C++ compiles (`-fsyntax-only`; the answer is a function, not a program) |
| `t6b-reason-multi` | the five exact intermediate values of `docs/ten-task-expected.md` 2.10 |
| `t6-reason` | the three worked answers 3 / 4 / -1, extracted from the query's own line, plus the algorithmic markers that can be matched at all |
| `t4-prose` | the 400-word limit the prompt itself states |
| every task | answer language = prompt language (Rev3 core requirement 0), foreign script, repetition |

Numbers are matched with their thousands separators optional, so `6144`, `6,144` and `6.144`
all count, while `16144` and `61440` do not. The compiler harness prepends a prelude of standard
includes, so a correct function is not failed for the includes a prompt did not ask for, and
falls back to the largest single block when an answer prints an alternative that would redefine
its own symbols. A check can see a right value; it cannot see a wrong value written beside it —
that is what the blind grading is for.

### 3.2 The blind, pointwise grading

Pairwise LLM judging carries a strong position bias and flips on about 14 % of re-runs
(arXiv 2602.02219, arXiv 2606.13685), and pointwise scores are noisy too. So:

1. `tools/ten-task-run.py blind` writes every answer into one shuffled file under an **opaque
   8-hex id**, with its task id and nothing else: no arm, no seed, no timing, no model name, no
   finish reason. The id map goes to a separate file and is **sealed** — written, hashed, and
   not opened again until the grades exist.
2. Each answer is graded ALONE against the checklist of its task
   (`decode_out/ten-task/checklist.json`, the core requirements of
   `docs/ten-task-expected.md` 2.1 to 2.10). Every requirement is `met`, `partly` or `not`, with
   a one-line justification quoting the answer. Never two answers side by side.
3. A **full pass** is every core requirement of the task met. `share_met` is the mean over the
   task's requirements with `met` 1.0, `partly` 0.5, `not` 0.0.
4. A random 20 % is re-graded a second time in a FRESH shuffled order, to measure the grader's
   own consistency, and that consistency is reported.
5. Only then is the seal opened and the ids joined back to their arms.

`t1b-read-lang` is graded against **Rev2**: question 2 is the `gemv_fp4_bs` stride contract, not
the withdrawn `__nanosleep` question of W1. Its norm answer is derived from the frozen prompt
source itself:

> `y[(size_t)t * ys + row] = red2[0];` with `row = blockIdx.x`, `t = blockIdx.y` and
> `ys = *y_stride_p`, and the comment above the kernel: "same as `gemv_fp4_b` but with an
> EXPLICIT y row stride — used for the shared expert's gate|up pair writing into ONE `[t][1280]`
> buffer (p13 layout, the contract `silu_mul640` reads). The plain `gemv_fp4_b` keeps the
> `[t][rows]` compact layout used by every other call site."

The grading artefacts are kept: the shuffled file, the sealed map and its hash, the re-grade
file and the per-answer checklist results.

## 4. What the record cannot decide

<!-- LIMITS -->

- **B-llama @32768 is a missing control.** The run was aborted by the user before any record
  was written (`run-B32.log` is empty); there is no llama-side measurement of the budget
  effect. The claim "the 16k gap is mostly budget" therefore rests on the crow arm's own
  16k→32k comparison plus B-llama's 16k numbers, not on a symmetric 2×2 design. It is
  documented here as a gap, never as a result.
- **The sealed blind set covers only the two 16384 arms.** `blind/answers.md` was written
  before the 32k run existed, so the A-crow-32k arm was graded non-blind (arm identity known
  by construction) against the same checklist, and is reported everywhere as a separately
  labelled supplement.
- **One grader, one session.** The 20 % re-grade sample (12 answers) reproduced 12/12 verdicts
  and share_met values, but the second pass happened in the same session that had already read
  the answers; the protocol's "fresh shuffled order" was followed mechanically, the grader was
  not amnesiac. Pointwise only — no pairwise judging was done, by design.
- **Small bands.** Three seeds per cell; the Wilson intervals of the two 16k arms overlap
  (0.7–18.3 % vs 10.6–40.8 % English pass), so "B-llama ahead at 16k" is the best point
  estimate, not a separated result. Seed determinism (finding 4) means a re-run adds no
  variance information; only new seeds do.
- **The mechanical matcher has a LaTeX blind spot.** Thousands separators written `{,}`
  (`1{,}610{,}612{,}736`) are not recognised, which marked two arithmetically perfect
  `t6b` answers as missing values; the blind grading corrected both by reading. The runner
  itself was not modified in this pass.
- **Token economics are comparable only within an arm's own budget**: mean answer/thinking
  tokens of the 32k arm are counted from a 32 768-token ceiling, the 16k arms from 16 384, and
  no pairwise position-bias control exists for the pointwise grades.

## 5. How to re-run it

```
# one engine on the card at a time; ps -C serve,decode,parity,llama-server empty, card < 1 GiB

tools/serve-linux.sh --port 8099 > decode_out/ten-task/serve-A.log 2>&1 &
tools/ten-task-run.py run --label A-crow --base-url http://127.0.0.1:8099
kill -INT $(ps -C serve -o pid=)

~/.local/share/crow/venv/bin/python ~/.local/share/crow/tools/start-server.py flash-next-q2-k-xl \
    > decode_out/ten-task/llama-B.log 2>&1 &
tools/ten-task-run.py run --label B-llama --base-url http://127.0.0.1:8083
kill -INT $(ps -C llama-server -o pid=)

# no GPU from here on
tools/ten-task-run.py tokens --label A-crow
tools/ten-task-run.py tokens --label B-llama
tools/ten-task-run.py blind  --labels A-crow,B-llama      # writes the SEALED id map
#   ... grade decode_out/ten-task/blind/answers.md against decode_out/ten-task/checklist.json ...
tools/ten-task-run.py report --labels A-crow,B-llama --grades decode_out/ten-task/blind/grades.json
python tools/test_ten_task_run.py
```

## 6. Where the records are

`decode_out/ten-task/`:

| path | content |
|---|---|
| `A-crow/records.jsonl`, `B-llama/records.jsonl` | one JSON object per generation: task, seed, the full `content`, the full `reasoning_content`, finish reason, token counts, timings, the sampling row that was sent, the server's `/props` model name, and the mechanical checks |
| `A-crow/run.json`, `B-llama/run.json` | the run headers: repo commit, endpoint, props, the row, what each server carries for thinking |
| `A-crow/tokens.json`, `B-llama/tokens.json` | the exact answer- and thinking-token counts from the reference tokenizer |
| `checklist.json` | the pointwise checklist the blind grading read |
| `blind/answers.md`, `blind/regrade.md` | the shuffled answer files, opaque ids, no arm label |
| `blind/idmap.json`, `blind/idmap.json.sha256` | the seal |
| `blind/grades.json`, `blind/regrades.json` | the per-answer checklist results |
| `report.json` | the joined tables of section 7 |

The server logs beside them are not tracked.
