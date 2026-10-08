# GLM-5.3-Flash: tier simulation and gate G1

`tools/glm_tier_sim.py` is step 8 of the GLM-5.3-Flash plan (crow-nest #147, parent nibor1896/Crow#362). It
answers gate G1 of `runs/glm53-flash/PREREG.md`: what share m of the routed expert visits per token
would fall on the NVMe with a VRAM hot set, a pinned tier and an NVMe window, and whether that share
and the NVMe rate B carry 40 tok/s. It reads the routing that the layerwise runner writes
(`docs/glm5-reference-runner.md` §5). It changes nothing in `engine/`, the converter or a container.

No GLM routing exists yet. The corpus is fixed (PREREG amendment 1, 2026-10-08); the routing passes need
the full container (step 9). The current step-3 run has no valid B, so `sim` prints
`G1 not answered` until a step-3 run of record exists.

## 1. Commands

```
# corpus: Crow sessions through GLM's own chat template and tokenizer -> <name>-ids.json, <name>-mask.json, corpus.json
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py corpus --model models/GLM-5.3-Flash-original \
    --out decode_out/glm-step8/corpus --cap 32768 --held <name> --file <name> <task> <session.json> [--file ...]

# routing: the five runner passes of PREREG amendment 1 on the full container, resumable per file
# (docs/glm5-reference-runner.md section 8; --dry-run checks and prints the five commands)
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py [--dry-run]

# simulation, every report row, and the G1 verdict fields
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py sim --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs --step3 runs/glm53-flash/step03/<run>.json [--windows 0,8,16,32] [--json out.json]

# the tests (no GPU, no weights): the sim about 7 s, the route passes about 45 s
.venv-oracle/Scripts/python.exe -I tools/test_glm_tier_sim.py
.venv-oracle/Scripts/python.exe -I tools/test_glm_route_passes.py
```

Step 8 runs under `.venv-oracle` (transformers 5.16.1). The system Python's transformers (5.5.4 on the
owner's machine, 2026-10-08) lacks `transformers.cache_utils.DynamicIndexedLayer`, which the runner's
in-place DSA cache subclasses: `tools/test_glm_route_passes.py` fails there in `setUpClass` of its
end-to-end test (8 tests run, 1 error), and `corpus` needs GLM's tokenizer from the same venv.
`tools/test_glm_tier_sim.py` alone passes under either (26 tests, OK under both).

`corpus` uses `messages` and `render` of `tools/session_ids.py` and swaps in GLM's tokenizer, so the
generated spans are cut the same way as in the Flash-Next calibration (`docs/hotset-calibration.md`).
`--cap N` routes the first N tokens of each file. Because the model is causal, a prefix routes exactly
as it does inside the full file. The ids decode to private transcripts, so they stay in the
git-ignored `decode_out/`. `corpus.json` records each file's source sha256, the full and routed token
counts, the generated counts and the sha256 of the ids and mask.

## 2. The policy

The shape is from `config.json` (rev `eb9eb208`): 42 MoE layers (3–44), 288 routed experts, top 8, so
336 visits per token.

| tier | per layer | which experts |
|---|---|---|
| VRAM hot set | N = 25 … 37 | ranks 0..N−1 |
| pinned RAM | 83 | ranks N..N+82 |
| NVMe | the rest | ranks ≥ N+83 |
| NVMe window | W slots | LRU over the NVMe-tier experts, in token order over every position of the file |

- **Rank.** An expert's rank is set by its routed count over the generated positions of the
  calibration files, in frequency order with ties going to the lower id. This is the `G` rule of #106.
- **Never the held-out.** The held-out file never enters the rank. `cut()` refuses a calibration list
  that contains it, and `check_corpus()` refuses a corpus in which it shares a task, a source or its
  ids with a calibration file.
- **Window.** The policy is LRU (Eliseev & Mazur, arXiv:2312.17238 §3.1). Belady's MIN with bypass
  over the same W is printed as the ceiling no replacement policy can beat. It is never the policy.
  LRU is a stack algorithm, so its NVMe reads never grow with W. m at W = 0 therefore bounds every W,
  and that is why the verdict is judged at W = 0 (amendment 1).
- **m and p.** m = NVMe reads / 336 per token, and p = pinned visits / 336. Both are primary on the
  generated positions of the held-out file.

## 3. What `sim` prints

- **Self-test.** Every routing file's sha256 equals the runner manifest's record. Its shape is [N][8].
  Its ids lie in 0..287 and each row is eight distinct ids in ascending order. The runner's ids equal
  the corpus file's ids.
- **m at N = 25 … 37, W = 0.** For each N: VRAM / pinned / NVMe shares, m and p with a 95 % block
  bootstrap CI, and m on the prefill (non-generated) positions and on all positions for contrast. The
  bootstrap uses non-overlapping 1,000-position blocks of consecutive generated positions, 2,000
  resamples and seed 20261008. It is the form of `tools/hotset-eval.py:80-86`, and a tail shorter than
  a block is dropped as there.
- **Windows.** At N ∈ {25, 31, 37} and W ∈ `--windows`: LRU m with CI, and MIN as the ceiling.
- **Per layer and depth.** m per layer at N = 25, and the depth buckets 0–32k / 32k–100k / 100k+
  (no CI under two blocks).
- **Own-count ceiling.** The held-out cut on its own counts. It is never a verdict.
- **Leave-one-out.** For each file: cut on all the others, score its generated positions, at
  N ∈ {25, 31, 37}.
- **Pinned traffic.** p × 190.3 GB/s at 40 tok/s, set against the staging ceilings 31.5 GB/s
  (`stage_cold` kernel, default) and 55.0 GB/s (copy engine, opt-in). Also the per-token floor
  p × 4.756 GB / R + m × 4.756 GB / B. These are reported and have no threshold.
- **G1.** At N = 25 and W = 0 the gate holds when the m CI upper bound ≤ m* = B / 190.3 and
  39.9 × B ≥ 150 tok/s. The result is `G1 passed`, `G1 failed: <which>` or `G1 not answered: <why>`.

## 4. When the answer is "not answered"

- **No valid B.** B is taken only from a step-3 run of record (`tools/nvme_read_rate.py --json`). The
  run must be valid (not VOID), and its best reader count, the one with the best median, must hold the
  1.15 spread. That spread is re-computed here from the raw repetitions, so a `b_for_g1` written into
  a run with a wider spread is refused. The run of 2026-10-08 00:18 UTC has the best count at 2 readers
  with spread 1.232, so it gives no B.
- **Wrong routing source.** The routing does not come from the dequantised full container: the runner
  ran on the FP8 originals or on a partial container (`--layers`), or the pass is incomplete. Such a
  number is plausibility only (PREREG G1).
- **Too few positions.** The held-out has fewer than two blocks of generated positions, so there is no CI.

Refusals exit with code 2 and print the reason: a corpus guard, a self-test failure, or ids that are
not the corpus file's.

## 5. Tests

`tools/test_glm_tier_sim.py` has 26 tests on synthetic dumps in the runner's own layout, each with a
known answer:

- shares of exactly 0.25 / 0.25 / 0.5 at every N;
- generated positions alone set the rank;
- LRU 4 reads and MIN 3 reads on A B A C B with W = 2;
- LRU is monotone in W, and MIN ≤ LRU;
- bootstrap blocks behave as specified;
- every self-test refusal fires;
- an end-to-end run with a window and leave-one-out.

The guards are red without the fix and green with it (checked 2026-10-08 by removing each hunk):

| guard | removed hunk | failing tests without it |
|---|---|---|
| held-out never in the cut | `cut()`, the held-out check | `test_cut_refuses_held_in_calibration` |
| held-out task differs | `check_corpus()`, the task check | `test_corpus_refuses_same_task_or_same_file`, `test_cli_exit_code_on_refusal` |
| spread re-check of B | `b_from_step3()`, the 1.15 check | `test_tampered_b_with_bad_spread_is_refused`, `test_known_m_and_not_answered_without_b` |
| no B means not answered | `g1_verdict()`, the B-missing branch | `test_verdict_not_answered_without_b`, `test_known_m_and_not_answered_without_b` |

There was also a dry run on synthetic random routing at corpus size (5 × 32,768 positions,
42 × 288 × 8). It took 33 s on Windows for every report row, the windows and the MIN ceilings
included. It checks the tool only; no number from it describes GLM.

## 6. Limits

- **Not run on GLM routing.** The routing passes wait for the full container (step 9). The runner
  takes the prompt in 512-row calls against an in-place DSA cache (`--prompt-chunk`, crow-nest #147);
  `tools/glm_route_passes.py` drives the five passes. About 1.8 h per pass, about 9 h for the corpus
  (derived, not measured; the first pass measures it), against about 16 h per pass for the runner
  before the chunk option, which had to feed 28,672 of the 32,768 rows one at a time while HF's
  `DynamicCache` concatenated the expanded MLA K/V on every row. Amendment 1 derived 1.4 h; the
  difference is KDA's chunk form (`docs/glm5-reference-runner.md` section 8).
- **Teacher-forced order.** The routing follows a teacher-forced order, not free generation, and other
  models wrote the sessions. Step 14 measures m in the engine (G4); this tool does not assume it.
- **Shape.** The tool reads the runner's per-layer files. The engine's `CROW_ROUTE_DUMP_PREFILL`
  format and the Flash-Next tools (`hotset-eval.py`, `coverage-curve.py`, 48 × 512 × 10) are
  unchanged; they are not needed for G1.
