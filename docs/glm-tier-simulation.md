# GLM-5.3-Flash: tier simulation and gate G1

`tools/glm_tier_sim.py` is step 8 of the GLM-5.3-Flash plan (crow-nest #147, parent nibor1896/Crow#362). It
answers gate G1 of `runs/glm53-flash/PREREG.md`: what share m of the routed expert visits per token
would fall on the NVMe with a VRAM hot set, a pinned tier and an NVMe window, and whether that share
and the NVMe rate B carry 40 tok/s. It reads the routing that the layerwise runner writes
(`docs/glm5-reference-runner.md` §5). It changes nothing in `engine/`, the converter or a container.

No GLM routing exists yet. The corpus is fixed (PREREG amendment 1, 2026-10-08); the routing passes need
the full container (step 9). Step 3's statistic gives no B on this machine; PREREG amendment 5 (robin,
2026-10-08, #146) fixes B = 6.994 GB/s, the 1-reader median of run `20261008T001819Z`, passed as `--readers 1`.

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
    --runs decode_out/glm-step8/runs --step3 runs/glm53-flash/step03/<run>.json [--readers 1] [--windows 0,8,16,32] \n    [--json out.json]
# G1 under amendment 5: --step3 runs/glm53-flash/step03/20261008T001819Z.json --readers 1

# dynamic expert-cache policies (#178, section 7)
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py dyn --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs (--slots V:P[,...] | --vram 25.6GB --pinned 46GiB) [--bpw 4.5,3.05,3.5] \
    [--arena layer|global] [--policies lru,clock,lfu] [--admit-max 64] [--prefetch none,oracle,0.5,0.7,0.9] \
    [--depths 1,2,3] [--pf-budget N] [--step3 <run>.json --readers 1] [--rates rates.json] [--json out.json]

# the tests (no GPU, no weights): the sim about 18 s, the route passes about 45 s
.venv-oracle/Scripts/python.exe -I tools/test_glm_tier_sim.py
.venv-oracle/Scripts/python.exe -I tools/test_glm_route_passes.py
```

Step 8 runs under `.venv-oracle` (transformers 5.16.1). The system Python's transformers (5.5.4 on the
owner's machine, 2026-10-08) lacks `transformers.cache_utils.DynamicIndexedLayer`, which the runner's
in-place DSA cache subclasses: `tools/test_glm_route_passes.py` fails there in `setUpClass` of its
end-to-end test (8 tests run, 1 error), and `corpus` needs GLM's tokenizer from the same venv.
`tools/test_glm_tier_sim.py` alone passes under either (41 tests, OK under both on 2026-10-08, about 18 s).

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
- **Pinned traffic.** p × 190.3 GB/s at 40 tok/s, set against the staging ceilings 51.6 GB/s
  (`stage_cold_ca` kernel, default; the device-issued ceiling, `docs/architecture.md:1414`) and 54.6 GB/s (copy
  engine, opt-in). The old `stage_cold` kernel (31.5 GB/s) is the `CROW_STAGE_KERNEL=1` fallback. Also the per-token floor
  p × 4.756 GB / R + m × 4.756 GB / B. These are reported and have no threshold.
- **G1.** At N = 25 and W = 0 the gate holds when the m CI upper bound ≤ m* = B / 190.3 and
  39.9 × B ≥ 150 tok/s. The result is `G1 passed`, `G1 failed: <which>` or `G1 not answered: <why>`.

## 4. When the answer is "not answered"

- **No valid B.** B is taken only from a step-3 run of record (`tools/nvme_read_rate.py --json`). The
  run must be valid (not VOID), and its best reader count, the one with the best median, must hold the
  1.15 spread. That spread is re-computed here from the raw repetitions, so a `b_for_g1` written into
  a run with a wider spread is refused. The run of 2026-10-08 00:18 UTC has the best count at 2 readers
  with spread 1.232, so it gives no B. With `--readers N` (the count a PREREG amendment fixes; amendment 5:
  1) B is that count's median instead, under the same validity and 1.15 spread checks; that run gives
  6.994 GB/s at 1 reader, spread 1.003.
- **Wrong routing source.** The routing does not come from the dequantised full container: the runner
  ran on the FP8 originals or on a partial container (`--layers`), or the pass is incomplete. Such a
  number is plausibility only (PREREG G1).
- **Too few positions.** The held-out has fewer than two blocks of generated positions, so there is no CI.

Refusals exit with code 2 and print the reason: a corpus guard, a self-test failure, or ids that are
not the corpus file's.

## 5. Tests

`tools/test_glm_tier_sim.py` has 41 tests on synthetic dumps in the runner's own layout, each with a
known answer (the 13 of `dyn` are in section 7):

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

## 7. Dynamic expert-cache policies (`dyn`, #178)

`dyn` is step 2 of the dynamic-tier plan (#169). It replays the held-out file's routing, in token order over
every position, through an elastic cache instead of the static cut, and scores the generated positions as
`sim` does. It reuses `sim`'s corpus loading, held-out guard, routing self-test and B rule. It decides nothing:
the G1d verdict is step 3 and needs its own PREREG (step 1).

**The model.**

| element | rule |
|---|---|
| arena | `--arena layer`: one cache per MoE layer; `--arena global`: one cache for all layers (sybil's VRAM arena) |
| tiers | VRAM (V slots) and pinned RAM (P slots), exclusive; the rest on the NVMe |
| VRAM hit | no transfer |
| pinned hit | admitted to VRAM (1 PCIe copy) when admission is on, else read zero-copy (1 PCIe read) |
| NVMe read | admitted to VRAM, or landed in pinned and read zero-copy |
| victims | a VRAM victim moves to pinned (1 PCIe write-back); a pinned victim is dropped (its record stays on the NVMe) |
| admission | on unless `--admit-max N` and a step has more than N picks (sybil's `GLM53_EC_ADMIT_MAX`, default 64) |
| capacity | `--vram`, `--pinned` (B, KB, MB, GB, KiB, MiB, GiB) / expert bytes, per layer also / 42; or `--slots V:P` |
| expert bytes | 4.5 bpw 14,155,776 (CNQ record); 3.05 bpw 9,474,048 (sybil record, external); 3.5 bpw 10,871,858 (the 3.05 record x 3.5 / 3.05, derived) |

Example: 46 GiB pinned is 83 slots per layer at 4.5 bpw and 124 at 3.05 bpw; 25.6 GB of VRAM is 64 per layer
at 3.05 bpw.

**Policies.**
- **LRU** handles the picks one at a time, in order. It is exactly `lru_reads` with every visit on the
  NVMe tier and W = V + P, for every split of V and P. That holds because an exclusive two-level LRU with
  promotion and demotion acts as one LRU of the summed size (Mattson et al. 1970). So `--slots C:0`
  reproduces the scratch numbers of #169.
- **CLOCK** follows sybil-solutions/glm53-flash-offload `glm53/expert_cache.py` `ec_step_k` (`df0b439`). A
  hit sets the ref bit. An insert sweeps from the hand, skips slots that hold one of the step's picks, clears
  set bits and takes the first clear slot. The new entry gets its bit set. Empty slots are taken first.
- **LFU** decays exponentially. The score is the sum of 2^-(age / half-life) over an expert's accesses, with
  `--lfu-halflife` in tokens (default 64, an unmeasured choice). The victim is the lowest score, never one of
  the step's picks.
- **Belady MIN with bypass** at V + P slots is the ceiling, never a policy. A policy with fewer NVMe reads
  than MIN over the whole file is refused as a simulator defect (exit 2).

sybil's 64 is a batch gate, not a per-token budget. It switches admission off for steps with more picks
(prefill). In this one-token-per-step replay a step has 8 picks, so any `--admit-max` of 8 or more leaves
admission on.

**Prefetch.** `--prefetch oracle` or a precision p. After layer j, the experts predicted for layer j + d
(`--depths`, crossing into the next token after the last layer) are read from the NVMe into pinned, at most
`--pf-budget` reads per token. A predictor of precision p keeps each true expert with probability p and
otherwise names a wrong expert of that layer (seed 20261008). Wrong reads cost NVMe bytes and pinned slots.
This follows Eliseev & Mazur, arXiv:2312.17238 §3.2, and sybil's layer-ahead prefetch. A config without
pinned slots prints no prefetch rows.

**What it prints**, per bpw and capacity, on the generated positions:
- MIN m;
- per policy and prefetch setting:
  - m with the 95 % block bootstrap CI. m counts every NVMe read, prefetch reads included;
  - m_demand, the stalling reads;
  - VRAM and pinned hit shares;
  - PCIe = zero-copy + admissions + write-backs;
  - DRAM = NVMe data landed + every PCIe transfer;
  - useful and wasted prefetches;
  - per-stage ceilings: NVMe at B, PCIe at R_PCIe, DRAM at R_DRAM;
  - the binding ceiling (perfect overlap) and the serial bound (no overlap).

B comes from the step-3 JSON exactly as in `sim` (`--step3`, `--readers`). R_PCIe and R_DRAM come from
`--rates`, a JSON file `{"R_pcie_gbps": x, "R_dram_gbps": y, "source": "..."}`. A missing rate prints "-",
never a default.

**Tests** (in `tools/test_glm_tier_sim.py`, known answers, red without the change, 2026-10-08):

| test | known answer |
|---|---|
| CLOCK hand sequence | A B C B A D C B E D, 3 slots: reads 1 1 1 0 0 1 0 0 1 0 (LRU 1 1 1 0 0 1 1 1 1 1) |
| CLOCK step picks | no victim outside the step's picks: the entry is not stored |
| LRU = window, capacity 0 = static | `dyn` LRU at V:P equals `lru_reads` at W = V + P; at 0:0 every visit is an NVMe read (the W = 0 path) |
| MIN bounds every policy | LRU, CLOCK, LFU, both arenas, with and without prefetch and admission gate |
| MIN guard | a policy below MIN is refused |
| precision 1.0 = oracle | identical counters for LRU, CLOCK (global), LFU |
| oracle prefetch | per layer, CLOCK and LFU: only the first d layers of token 0 stall, nothing wasted; budget 0 = no prefetch |
| uniform routing | LRU m = (1/8) sum_{k=0..7} (288 - C) / (288 - k) = 0.6327 at C 108; CLOCK and LFU (step picks protected) m = (288 - C) / 288 = 0.625; both within 0.004 |
| capacity from bytes | 83 / 124 slots per layer for 46 GiB at 4.5 / 3.05 bpw |
| PCIe accounting | zero-copy + admissions = visits - VRAM hits; a hand trace of admissions and write-backs |
| cost model, rates file | ceilings and serial bound; a bad rate is refused |
| CLI | an end-to-end run with JSON; refusals exit 2 |

Removing a hunk turns its test red (checked 2026-10-08):

| removed hunk | failing test |
|---|---|
| CLOCK ref bit on insert | CLOCK hand sequence |
| CLOCK step-pick protection | CLOCK step picks |
| LFU step-pick protection | uniform routing |
| precision draws | precision 1.0 = oracle |
| write-back count | PCIe hand trace |
| prefetch reads counted | MIN bounds every policy |
| LRU demotion to pinned | LRU = window |

**Speed** on synthetic uniform routing at full shape (4,096 positions, Windows, 2026-10-08), per config:
LRU 0.7 s, CLOCK 1.0 s, LFU 2.8 s, LRU with prefetch 1.8 s, MIN 0.5 s (layer) and 0.9 s (global).
For a 32,768-position file that is about 8x (derived, not measured).

**Reproduction, pending.** The routing dumps `decode_out/glm-step8/runs/` were deleted on 2026-10-08 and will be
regenerated. Once they are back, this command must give LRU m 0.278 / 0.185 / 0.060 and MIN 0.122 / 0.072 / 0.019
on `todo-1006`, within +-0.5 pp (plan step 2). Otherwise #178 records why:

```
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py dyn --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs --slots 108:0,145:0,216:0 --policies lru \
    --step3 runs/glm53-flash/step03/20261008T001819Z.json --readers 1 --json <out.json>
```

**Not modelled:** prefetch from pinned into VRAM, batched prefill steps, kernel time, and the latency a
demand read stalls for (only bytes).
