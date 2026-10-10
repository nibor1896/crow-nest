# GLM-5.3-Flash: tier simulation and gate G1

`tools/glm_tier_sim.py` is step 8 of the GLM-5.3-Flash plan (crow-nest #147, parent nibor1896/Crow#362). It
answers gate G1 of `runs/glm53-flash/PREREG.md`: what share m of the routed expert visits per token
would fall on the NVMe with a VRAM hot set, a pinned tier and an NVMe window, and whether that share
and the NVMe rate B carry 40 tok/s. It reads the routing that the layerwise runner writes
(`docs/glm5-reference-runner.md` §5). It changes nothing in `engine/`, the converter or a container.

The corpus is fixed (PREREG amendment 1, 2026-10-08). G1 was judged on routing from the 4.5-bit container
(`runs/glm53-flash/step08/20261008-g1.md`, G1 failed). That container and its dumps were deleted on 2026-10-08.
The routing for G1d comes from the FP8 originals instead (`runs/glm53-flash/PREREG-dyn.md` amendment 1, #179);
no such dumps exist yet. Step 3's statistic gives no B on this machine; PREREG amendment 5 (robin,
2026-10-08, #146) fixes B = 6.994 GB/s, the 1-reader median of run `20261008T001819Z`, passed as `--readers 1`.

## 1. Commands

```
# corpus: Crow sessions through GLM's own chat template and tokenizer -> <name>-ids.json, <name>-mask.json, corpus.json
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py corpus --model models/GLM-5.3-Flash-original \
    --out decode_out/glm-step8/corpus --cap 32768 --held <name> --file <name> <task> <session.json> [--file ...]

# routing: the five runner passes of PREREG amendment 1, resumable per file (docs/glm5-reference-runner.md
# section 8; --dry-run checks and prints the five commands). G1d: the FP8 originals, verified by
# tools/fetch-glm.py (#179, PREREG-dyn amendment 1); each pass dir gets weights.json, the weights' identity.
# --container was G1's source; that container is deleted. PREREG-dyn amendment 2: the four calibration files'
# routing comes from the MUL1 conversion's capture (section 7), so only the held-out needs a pass (--files).
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py --fp8 models/GLM-5.3-Flash-original [--dry-run]
.venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py --fp8 models/GLM-5.3-Flash-original --files todo-1006

# simulation, every report row, and the G1 verdict fields
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py sim --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs --step3 runs/glm53-flash/step03/<run>.json [--readers 1] [--windows 0,8,16,32] \
    [--json out.json]
# G1 under amendment 5: --step3 runs/glm53-flash/step03/20261008T001819Z.json --readers 1

# dynamic expert-cache policies (#178, section 7)
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py dyn --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs [--capture decode_out/glm-step8/capture-ids] \
    (--slots V:P[,...] | --vram 25.6GB --pinned 46GiB) [--bpw 4.5,3.05,3.5] \
    [--arena layer|global] [--policies lru,clock,lfu] [--admit-max 64] [--prefetch none,oracle,0.5,0.7,0.9] \
    [--depths 1,2,3] [--pf-budget N] [--step3 <run>.json --readers 1] [--rates rates.json] [--json out.json]

# the judged G1d policy and verdict of PREREG-dyn (#178, section 8); --cells all for every capacity scenario
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py g1d --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs --capture decode_out/glm-step8/capture-ids \
    --step3 runs/glm53-flash/step03/20261008T001819Z.json --readers 1 [--cells 3.05:46:V25] [--jobs 16] [--json out.json]

# the tests (no GPU, no weights): the sim about 19 s, the route passes about 80 s (2026-10-09)
.venv-oracle/Scripts/python.exe -I tools/test_glm_tier_sim.py
.venv-oracle/Scripts/python.exe -I tools/test_glm_route_passes.py
```

Step 8 runs under `.venv-oracle` (transformers 5.16.1). The system Python's transformers (5.5.4 on the
owner's machine, 2026-10-08) lacks `transformers.cache_utils.DynamicIndexedLayer`, which the runner's
in-place DSA cache subclasses: `tools/test_glm_route_passes.py` fails there in `setUpClass` of its
end-to-end test (14 tests run, 1 error; under `.venv-oracle` 18 tests, OK, 80 s; both 2026-10-09), and `corpus`
needs GLM's tokenizer from the same venv. `tools/test_glm_tier_sim.py` alone passes under either (53 tests, OK
under both on 2026-10-09, about 19 s).

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
  number is plausibility only (PREREG G1). This is `sim`'s rule for the G1 record; `dyn` judges its source by
  PREREG-dyn amendment 1 instead (section 7).
- **Too few positions.** The held-out has fewer than two blocks of generated positions, so there is no CI.

Refusals exit with code 2 and print the reason: a corpus guard, a self-test failure, or ids that are
not the corpus file's.

## 5. Tests

`tools/test_glm_tier_sim.py` has 43 tests on synthetic dumps in the runner's own layout, each with a
known answer (the 15 of `dyn` are in section 7):

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
the G1d verdict is step 3 under `runs/glm53-flash/PREREG-dyn.md`.

**Routing source (PREREG-dyn amendment 1, 2026-10-09).** G1d's routing comes from the FP8 originals of
`zai-org/GLM-5.3-Flash` at `eb9eb208`, run by `tools/glm_route_passes.py --fp8` (#179). `dyn` checks each pass dir
(`g1d_source_reasons`, check 2 of the amendment):
- the runner manifest's weights are `fp8`, not partial, and the pass is complete over every layer;
- the pass ran with `--state-dtype bf16 --prompt-chunk 512`;
- `weights.json` is kind `fp8-originals` at that revision with 62 shards;
- its `identity_sha256` matches its content, and its index sha256 is the manifest's.

Across the run: all five dirs carry one identity, and every `passes.jsonl` row of a corpus file carries it, with
an ok row per file. Any failed check prints `NOT the G1d source` with the reason and lands in `source_reasons` of
the JSON; the identity sha256 is printed and stored as `weights_identity_sha256`. `sim` keeps G1's rule: FP8 routing
stays "plausibility only" there and G1 "not answered" on it.

**Calibration routing from the conversion's capture (PREREG-dyn amendment 2, 2026-10-09).** The MUL1 conversion
(#182) runs the four calibration files through the same `run_layer` on the same FP8 originals. Per MoE layer it
writes the router's top-8 ids as `ids.i32`, int32 [131,072][8]: the files' 32,768 rows one after another, in the
amendment-1 order. Beside it, `capture.json` names the files and rows, the FP8 identity and `ids_sha256`. A copier
keeps both per layer in `decode_out/glm-step8/capture-ids/L<ll>/` before `quantize` prunes them. `--capture <dir>`
(in `dyn` and `sim`) takes the calibration files from there; `--runs` then needs only the held-out pass dir.
Refused by name (exit 2), per MoE layer 3..44:
- `L<ll>/ids.i32` or `capture.json` missing (all missing layers named);
- `capture.json` of another layer;
- its `identity` is not the `identity_sha256` of the held-out pass dir's `weights.json` (no `weights.json`: refused);
- its files are not the corpus' calibration files in corpus order;
- per-file rows are not the length of each file's ids, the total is not their sum, or top-k is not 8;
- the sha256 of `ids.i32` is not `ids_sha256`, or its size is not rows x 8;
- the routing self-test of `load_runner`: ids 0..287, eight distinct ascending per row.

Each file gets its slice in that order. Generated and prompt positions come from `<name>-mask.json`, as for pass
dirs. Under `dyn` the G1d checks then cover the held-out pass dir and its `passes.jsonl` row. The JSON names the
source in `calibration_routing`, with the 42 per-layer sha256. Under `sim` (check 4.2 below) the capture adds a
"plausibility only" reason, so G1 stays "not answered". Layer 3's ids were pruned before the copier started; it is
recomputed with `tools/glm_mul1_quantize.py capture --layers 3` in a separate work dir (amendment 2).

`dyn` scores only the held-out: LRU, CLOCK, LFU, prefetch and MIN. The judged policy, SEED+LRU, SEED+LRU+IDPF,
the calibration grid choice and the verdict row are `g1d` (section 8).

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
| G1d source | FP8 routing with `weights.json` and `passes.jsonl`: no source reason in `dyn`, while `sim` still says "plausibility only" and G1 "not answered" |
| G1d source refusals | CNQ weights, prompt chunk, index sha256, a tampered or foreign `weights.json`, a missing one, two identities, a missing book, a book row with another identity |
| capture accepted (amendment 2) | calibration files of different lengths only in a capture dir: each Run equals its slice, mask from the corpus, no source reason in `dyn`, `calibration_routing` with 42 sha256; `sim --capture` runs as plausibility only |
| capture refusals (amendment 2) | by layer name: foreign identity, file order, per-file rows, total rows, ids sha256, a missing layer dir, a missing `ids.i32`, another layer's `capture.json`, a non-ascending row, no held-out `weights.json`; the CLI exits 2 |

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
| `dyn` on the G1d rule (2026-10-09) | G1d source |
| identity content check | G1d source refusals |
| `passes.jsonl` check | G1d source refusals |
| `load_corpus(capture=...)` (2026-10-09) | capture accepted, capture refusals (TypeError) |
| each capture check alone: identity, order, rows, sha256, missing layer, layer field | its capture refusal case |

**Speed** on synthetic uniform routing at full shape (4,096 positions, Windows, 2026-10-08), per config:
LRU 0.7 s, CLOCK 1.0 s, LFU 2.8 s, LRU with prefetch 1.8 s, MIN 0.5 s (layer) and 0.9 s (global).
For a 32,768-position file that is about 8x (derived, not measured).

**Side-by-side control, pending.** This replaces the old +-0.5 pp reproduction (PREREG-dyn amendment 1, #178
comment of 2026-10-09). The scratch numbers came from 4.5-container routing; FP8 routing differs from it (step 6,
layer 3: top-8 overlap 0.911), so a tolerance would test the quantisation, not the tool. Once the FP8 dumps exist,
and before the first G1d row:
1. the self-test on all five dumps (required);
2. `sim` at N 25, W 0: held-out m next to G1's 0.5217, with dm;
3. `dyn` LRU and MIN at 108 / 122 / 130 / 145 / 216 slots on `todo-1006`, next to the scratch numbers (LRU 0.278 /
   0.240 / 0.220 / 0.185 / 0.060, MIN 0.122 / 0.100 / 0.090 / 0.072 / 0.019), with the difference per cell.

These are descriptive, have no threshold, and can neither pass nor fail G1d.

```
.venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py dyn --corpus decode_out/glm-step8/corpus/corpus.json \
    --runs decode_out/glm-step8/runs --capture decode_out/glm-step8/capture-ids \
    --slots 108:0,122:0,130:0,145:0,216:0 --policies lru \
    --step3 runs/glm53-flash/step03/20261008T001819Z.json --readers 1 --json <out.json>
```

Item 2 runs `sim` with the same `--runs` and `--capture`. Without `--capture`, both need all five pass dirs.

**Not modelled:** prefetch from pinned into VRAM, batched prefill steps, kernel time, and the latency a
demand read stalls for (only bytes).

## 8. The judged G1d policy and verdict (`g1d`, #178)

`g1d` builds what `runs/glm53-flash/PREREG-dyn.md` fixes for the verdict. The source is the same as `dyn`'s: G1d checks,
with `--capture` from amendment 2. The details the PREREG leaves open are read as in its amendment 3.

**Per cell** (bpw, R GiB, V):
- C from bytes: floor(floor((V + R) / S) / 42), with the PREREG's S (4.5: 14,155,776; 3.5: 11,010,048;
  3.05: 9,474,048).
- Selection on the four calibration files, over all 30 grid points: s {0, 0.25, 0.5} x (P 0, or P {4, 8, 16} x
  d {1, 2, 4}).
  - Each file is replayed alone from an empty per-layer arena.
  - Score = mean over files of reads per generated token.
  - Ties: simpler class, then smaller s, P, d.
- Leave-one-out folds: the point is chosen on three files, with seed and table fitted on them, and scored on the
  fourth.
- The chosen point goes to `<json>.choices.jsonl` before the held-out is replayed.
- The held-out row with that one point: r with the 95 % block bootstrap CI, per-layer r, depth buckets, the first
  1,000 generated positions against the rest, the reset variant, prefetch reads issued and used, and Belady MIN at C.
- The verdict: `G1d passed / failed / not answered` for the primary cell (3.05, 46, V25: C 161, bar r_hi <= 18.455).
  Every other cell is a `scenario (no gate role)` under the same rule (4.5: 12.351, 3.5: 15.880).
  - Not answered: without B (`--step3 ... --readers 1`), without a CI (fewer than two blocks), or on any source
    reason.
  - Every report carries the PREREG's status line: the bar, grid and cells await robin's confirmation.

**Policies** (per layer):
- The seed is floor(s x C) experts by calibration count and is never evicted.
- The LRU part is C - seed - P.
- IDPF: before layer m's visit, the P highest-scoring non-resident experts are read into a buffer, one read each. The
  scores sum T[m - d][a, .] over the same token's layer m - d ids. Demanded buffered experts join the LRU part; the
  rest are dropped.

**Runtime (derived, not measured on real routing).** 4,096 synthetic positions at full shape, C 161, Windows,
2026-10-09, with the MUL1 conversion running beside it: 0.6 s for the three P 0 points and 16.4 s for nine IDPF
points of one d. A 32,768-position file is about 8x that, ≈ 400 s for all 30 points. One cell replays 4 files under
all points, then 4 folds x 3 files, then 4 left-out files under one point, then the held-out twice: about 1.8 h in
one process. `--jobs 16` runs the 16 (file, d) groups of a step side by side, ≈ 12 min. Cells with the same C share
the selection; `--cells all` has 18 distinct C.

**Tests** (`TestG1dModel`, `TestG1dCli`, red without the change, 2026-10-09):

| test | known answer |
|---|---|
| capacity and bar | every C of the PREREG table; bars 18.45 / 12.35 / 15.88; B / 40 = 174.84 MB |
| grid and tie order | 30 points, LRU first; equal scores pick the simpler class, then smaller s, P, d; floor(s x C) 40 / 80 at 161 |
| SEED+LRU hand trace | 0 1 2 0 3 1 2: LRU 3 reads 1 1 1 0 1 1 1, seed {0} + LRU 2 reads 0 1 1 0 1 1 1; the seed survives 5 misses |
| prefetch hand trace | perfect keys 1 read per token, all used; wrong keys 2 reads per token; unused buffer dropped; resident experts never proposed; used ones join the LRU part |
| IDPF table | counts over generated positions only; scores and lower-id ties; no prefetch for layers below d; seed + P > C refused |
| selection | score = mean of per-file replays; folds use the other files' statistics; held-out routing changes nothing; held-out in the calibration refused; only generated positions scored; `--jobs` = one process |
| verdict | passed at r_hi 18.400 <= 18.455, failed at 18.5, not answered without B, CI or on a source reason; a scenario cell is never "G1d" |
| CLI | capture corpus end to end: C 161, 30 points, 2 folds, the choice written before the held-out replay, held-out r = unseeded picks of token 0 / 2,000, passed |

Each rule removed alone turns a test red (checked 2026-10-09): tie order, prefetch reads counted, buffer dropped,
seed never evicted, resident not proposed, floor(s x C), table on generated positions, fold statistics, held-out
guard, source layer, bar comparison, and the choice written first.

## 9. The engine's arena against alternatives (`arena`, #188)

`arena` replays decode routing through a host copy of the engine's `GlobalArena` (`engine/src/glm5_tiers.rs`, 28f459a:
one CLOCK ring of VRAM slots and one pinned tier for all MoE layers, exclusive, pinned hits stay under
`CROW_GLM_PINNED=zerocopy`, NVMe misses admitted into VRAM, the VRAM victim written back as the newest pinned entry,
the oldest pinned entry dropped, warm start from `CROW_GLM_ARENA_WARM`) and through the alternatives:

| policy | rule |
|---|---|
| `today` | the engine as above |
| `promote` | `today` with pinned hits promoted into VRAM (`CROW_GLM_PINNED=promote`) |
| `lfu` | `today` with the pinned victim = the lowest decayed count (half-life `--halflife` decode tokens) |
| `tinylfu` | `lfu`, and an NVMe miss scoring no higher than that victim is read for the call only (bypass) |
| `noadmit` | NVMe misses land in pinned (`CROW_GLM_ARENA_NOADMIT=1`) |
| `freq` | the frequency tiers of `CROW_GLM_ARENA_FREQ=1`: pinned victim = lowest score; a routed expert enters VRAM only into a free slot or above (1 + `--margin`) x the lowest VRAM score, that expert written back; other NVMe misses are read into pinned |
| `lru` | one LRU over VRAM + pinned (the exclusive two-tier LRU with promotion), NVMe reads only |
| `min` | Belady's MIN with bypass over VRAM + pinned from the same warm set: the ceiling, guesses included |

Scores start at `--prior-tokens` (default 0.5 x half-life / ln 2) x each expert's visits per token in the warm counts.
Workloads: the held-out file of the corpus (generated positions decode, the rest are prompt calls that only set
reference bits), and a run's own routing written by `CROW_GLM_ROUTE_LOG` with `--boot base:ring:chunks:chunk:handback`
(the elastic boot: ring slots disabled, warm, the handed-back chunks written back; a sixth field `:1` regrows them
at the prompt's end and refills them from the experts the hand-back wrote out, then the warm scores, as
`CROW_GLM_ARENA_REGROW=1` does). With `--decode-log` the `today`
replay is compared per token and layer with the run's `tiers v/p/n` rows. Per policy: NVMe demand and speculative
reads per token (no guesses in the replay: speculative 0), H2D records (VRAM entries), promotions (pinned -> VRAM),
D2H write-backs, VRAM hits, pinned-served visits, zero-copy GB (pinned-served x the measured zero-copy share 0.352 x
9,474,048 B) and a critical-path cost per token from the decode profile of #202 (1.7 ms per demand read; zero-copy and
promotions at 27.05 ms per 0.833 GB). Run of record: `runs/glm53-flash/cache-sim-20261010/`. Tests: `TestArena` in
`tools/test_glm_tier_sim.py` (the engine's own CLOCK figures on its xorshift trace, hand traces of `lfu` and `freq`,
the route-log check, MIN below every policy).
