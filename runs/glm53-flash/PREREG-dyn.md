# PREREG-dyn: gate G1d, dynamic expert residency for GLM-5.3-Flash on crow-nest (crow-nest #172, root #169)

Written 2026-10-08 ~21:50 CEST on crow-nest `e6f901a` (branch `c1-s1-prereg`), before any G1d row exists. No simulation of the dynamic policies below has been run in this repo; no measurement, no GPU, no engine run was made for this file. **Every threshold, grid and tolerance in this file is "proposed, awaiting robin's confirmation"** (list in "Awaiting confirmation"). Until he confirms or changes them, no G1d row may be taken.

Relation to `runs/glm53-flash/PREREG.md` (G1, amendments 1-5): that file is read in full and stays untouched. G1 (static hot set N 25, W 0) failed on 2026-10-08 (`step08/20261008-g1.md`: m 0.5217, 95 % CI [0.4948, 0.5500], m* 0.0368) and its verdict is not reinterpreted. G1d is a new gate on a different residency model; it is not a re-judgement of G1. Inherited unchanged from that file: corpus and held-out (amendment 1), the CI method, B = 6.994 GB/s at 1 reader (amendment 5), the source and recipe of the container, `HOST_PINNED_CAP` 46 GiB (`engine/src/geo.rs:42`) and `CROW_RAM_MARGIN_GB` 1 (`engine/src/manager.rs:140`), the rule that no default, manifest or flag changes before G7.

## Seen while writing

- The G1 run: `runs/glm53-flash/step08/20261008-g1.md` and `g1-sim-20261008.json` (static cut, NVMe window LRU W 8 / 16 / 32: m 0.4065 / 0.3456 / 0.2693, MIN at W 32 0.1479).
- A scratch run, not committed, over `tools/glm_tier_sim.py` at `e6f901a` (`lru_reads`, `min_reads` reused as a pure dynamic cache per layer, no static cut): held-out `todo-1006`, generated positions, routing of the (now deleted) 2026-10-08 dumps, capacities C of 4.5-bit experts per layer 108 / 122 / 130 / 145 / 216: **LRU m 0.278 / 0.240 / 0.220 / 0.185 / 0.060, Belady MIN 0.122 / 0.100 / 0.090 / 0.072 / 0.019.** Which RAM budget each of 130 / 145 / 216 belongs to was not recorded; under the capacity formula below 108 and 122 are 46 and 54 GiB at 4.5 bpw. No CI, no leave-one-out, no prefetch, no calibration-chosen parameters were seen.
- What this does to the bar: in reads per token (x 336 visits) LRU is 93.4 / 80.6 / 73.9 / 62.2 / 20.2 and MIN 41.0 / 33.6 / 30.2 / 24.2 / 6.4, against a proposed 4.5-bit bar of 12.3 reads. The bar is derived from B and 40 tok/s only (below), not from these numbers; it is not set so that a policy passes. Among the seen cells only a ceiling (MIN at C 216, i.e. ~120 GiB of arena) is under it.
- The G1 corpus and held-out were chosen before the G1 routing existed (amendment 1); G1d reuses them and so has seen their G1 routing. The held-out `todo-1006` therefore has been seen under a static policy and, in the scratch run above, under LRU. This is stated as a limit: the held-out is no longer blind for LRU at the five capacities above. It stays blind for every parameter the calibration chooses (see "Parameter selection").

## Data

- Corpus and held-out exactly as PREREG.md amendment 1: `decode_out/glm-step8/corpus/`, `corpus.json` sha256 `8fb9560f43a58ed5a1791ed2c458130a0febf4236f39ef882c12e665faae5f26`, held-out `todo-1006` (26,599 generated of 32,768 routed positions, 26 blocks of 1,000), calibration `omarchy-0915a`, `lenis-0830`, `ctx7-0830`, `zetalab-0829`. Each file's ids sha256 and mask sha256 are the amendment-1 table's. The held-out is never used to choose a parameter, a policy or a grid point.
- **Data note.** The routing dumps `decode_out/glm-step8/runs/` (the five passes behind the G1 record) were deleted on 2026-10-08. `corpus.json` and the ten ids/mask files were restored from a shadow copy; on 2026-10-08 ~21:50 CEST `sha256sum` of `corpus.json` on the main checkout returned `8fb9560f...faae5f26` and the first 16 hex digits of the five mask files equal the amendment-1 table (`c306f565542c1f7d`, `5398ae6c1691bdfa`, `01b384512eb3620a`, `b84763b3335b0899`, `7cf87ef8191f33f1`). The ids files are checked in full by `tools/glm_route_passes.py` before a pass (it pins `CORPUS_JSON_SHA256` and the per-file ids and mask sha256 of amendment 1). The dumps will be regenerated from that same corpus; this is a separate step and is not done by this file.
- **Regeneration check, all of it before the first G1d row** (any failure: G1d is "not answered", not passed and not failed):
  1. `corpus.json` sha256 is `8fb9560f...faae5f26` and every ids/mask file matches amendment 1 (the pin of `glm_route_passes.py`).
  2. Each run dir's `manifest.json`: `ids` equal to the corpus ids, 32,768 tokens, weights = the container `converter/GLM-5.3-Flash-CNQ4.5.cnq` (sha256 `0684a8230264afa9f7387b2d02a164b4822d722b753a92113d18ea23f8e89c82`) with `index_json_sha256` `598eea2252411f8039da56d5f4acd8f4bbb042ffc3e83842356adfb37896c399`, complete over every layer (`source_reasons` checks the layer range), not partial, `--state-dtype bf16 --prompt-chunk 512` as in `passes.jsonl` (`tools/glm_tier_sim.py:126` `source_reasons` empty).
  3. The self-test of `glm_tier_sim.load_runner` (`:93`) passes for all five: every routing file's sha256 equals the manifest's, shape [N][8], ids in 0..287, rows strictly ascending; and the pass row in `passes.jsonl` has `rc` 0, `ok` true, no problems.
  4. Reproduction control against the G1 record: `glm_tier_sim.py sim` over the regenerated dumps at N 25, W 0 reproduces the held-out m of `g1-sim-20261008.json` (0.5217210770544543, same seed 20261008) within |dm| <= 0.005 (proposed; about a fifth of the CI half-width 0.0276). Equality of each routing file's sha256 with the lost runs is reported per file but cannot be required: the lost manifests were not kept in the repo, only the per-pass rows. If dm is outside the tolerance, the cause is named before any G1d row.
  5. The check and its outputs are recorded in the measurement book (artifact `MpmBA5NTAguy8EyHcRgoV4`) with date and commit.

## Cache model (fixed here)

- Arena: C expert slots per MoE layer (per-layer arena, judged) or 42 x C slots shared by all layers (global arena, reported). Experts are visited in token order over every position of the file, prompt and generated, as in G1; layers in ascending order within a token; the eight ids of a (token, layer) in ascending id order as the dumps hold them. State starts empty at position 0 of each file; the metric counts the generated positions only. Variant reported, not judged: state reset to empty at the first generated position.
- Capacity from bytes. Expert size S: 4.5 bpw NVFP4 14,155,776 B (3 x 4096 x 2048 weights, 36 B per 64 values, `PREREG.md` "Fixed for the whole series"); 3.05 bpw 9,474,048 B (the plan's figure; it equals 3 bit x 25,165,824 weights / 8 + 36,864 B, whose breakdown is not re-derived here); 3.5 bpw nominal 11,010,048 B (3.5 x 25,165,824 / 8, side data of a concrete format not added). Slots = floor((V + R) / S) over all layers, divided by 42 for the per-layer arena. R = pinned arena in GiB (46 = `HOST_PINNED_CAP`; 50 and 54 are scenarios that would need a different cap and are not a recommendation; nothing is changed). V = bytes of the VRAM expert share: primary V25 = 25 x 42 x 14,155,776 = 14,863,564,800 B (the low end of G1's 25-37, a byte budget held fixed across bpw); reported V37 = 21,998,075,904 B.

  | bpw | S (B) | R 46 GiB: C per layer / global slots | 50 GiB | 54 GiB | VRAM part of C (V25) |
  |---|---|---|---|---|---|
  | 4.5 | 14,155,776 | 108 / 4,539 | 115 / 4,842 | 122 / 5,146 | 25 |
  | 3.5 | 11,010,048 | 138 / 5,836 | 148 / 6,226 | 157 / 6,616 | 32 |
  | 3.05 | 9,474,048 | 161 / 6,782 | 172 / 7,235 | 183 / 7,688 | 37 |

  (Arithmetic, not measured. V37 gives 120 / 127 / 134 at 4.5, 154 / 163 / 172 at 3.5, 179 / 190 / 201 at 3.05.) **bpw 3.05 and 3.5 are capacity scenarios of the simulation only**: the routing is the same, no such container exists, and building one would be a second quantisation of the 4.5-bit container, which is not part of this file and is robin's decision. The one container that exists is 4.5 bpw.
- Tiers inside the arena (for PCIe/DRAM accounting only, never for the read count): the VRAM part is the C_V most recently used slots of the arena's recency order, the rest is pinned RAM.
- Policies "the engine can build": a policy that uses only the arena state, the ids of the current and earlier positions (and, for prefetch, of lower layers of the current token), and parameters fixed before the run (calibration statistics included). Router-logit or hidden-state prefetch is not scorable: the dumps hold routing ids only. Policies:
  - **LRU** per layer (and global).
  - **SEED+LRU**: a fraction s of the C slots holds, per layer, the experts of highest routed count over the generated positions of the calibration files (rank rule of `glm_tier_sim.py:256` `cut`, ties lower id) and never evicts; the rest is LRU.
  - **SEED+LRU+IDPF**: as SEED+LRU, plus id-only prefetch: for layer l at distance d (layers l+d), a table fitted on the calibration files (conditional counts of layer l+d ids given the layer-l ids of the same token) proposes the P highest-scoring experts not resident; each is one NVMe read when issued; it sits in a prefetch buffer of P slots taken from C; it joins the LRU part when demanded and is dropped after the layer-l+d visit if not. Wasted prefetches count as reads.
  - Not judged, reported: **MIN** (Belady with bypass, per layer and global; offline, needs the future) and **oracle prefetch** (every demanded expert known one layer ahead at zero wasted reads: it changes stall time, not bytes, so its byte count is the MIN/policy miss count and its stall is 0).
- Reads: one NVMe read = one expert block of size S that is not resident when needed or is prefetched. r = reads per token, over the 336 visits.

## Parameter selection (calibration only)

- Grid, fixed here (proposed): s in {0, 0.25, 0.5}; P in {0, 4, 8, 16}; d in {1, 2, 4} (P = 0 has no d). Policy classes LRU, SEED+LRU, SEED+LRU+IDPF.
- Score of a grid point on a set of files = mean over those files of reads per generated token, each file replayed alone from an empty arena at the capacity of the cell. Chosen point = lowest score on the four calibration files; ties go to the simpler policy, then the smaller s, then the smaller P, then the smaller d.
- Leave-one-out: for each calibration file the point is chosen on the other three and scored on the one left out; the four folds are reported with the point each chose and whether it equals the full-calibration choice. This is the robustness check; the judged point is the full-calibration choice.
- The chosen point, per cell, is written to the book before the held-out is scored for that cell. The held-out is scored with that one point; other points on the held-out may be reported as extra rows, never as a pick.

## Judged metric and thresholds (proposed, awaiting robin's confirmation)

- Metric: r_hi x S, where r_hi is the upper bound of the 95 % CI of r on the generated positions of `todo-1006`: block bootstrap, non-overlapping 1,000-token blocks over the generated positions, 2,000 resamples, seed 20261008 (`glm_tier_sim.py:344` `boot_ci`, `tools/hotset-eval.py:80-86`, Kuensch 1989). n = 26,599 positions, 26 blocks; fewer than two blocks gives no CI and no pass.
- Threshold: **r_hi x S <= B / 40 tok/s** = 174.84 MB per token with B = 6.994 GB/s (1 reader, amendment 5), for the policy chosen on the calibration (not MIN, not an oracle), per-layer arena, V25.
  - 4.5 bpw (the container that exists), R 46 GiB (C 108): **r_hi <= 12.35, stated as 12.3 reads per token** = 3.7 % of 336 visits. This cell decides G1d for the existing container.
  - 3.05 bpw, R 46 GiB (C 161): **r_hi <= 18.45, stated as 18.4 reads per token** = 5.5 % of 336 visits. Recorded as the verdict of the capacity scenario "if a 3.05 container existed"; it does not by itself make the gate pass.
  - 3.5 bpw: 15.88 reads (4.7 %), reported with the same rule, no gate role.
  - Reference, not a threshold: with the 2-reader median B 9.765 GB/s (spread 1.232 > 1.15, unconfirmed, not usable) the 4.5-bit bar would be 17.2 reads (5.1 %).
- What a pass means: the NVMe term alone fits 25 ms per token with full overlap. It is a necessary condition for 40 tok/s, not evidence of 40 tok/s; G5 owns the speed. The prefill line of G1 (39.9 x B >= 150 tok/s) is unchanged and already holds (279 tok/s); it is not re-judged here.
- Verdict values: passed / failed / not answered. A failed G1d stays failed; the series stops there and goes to robin (abort rules of PREREG.md). A pass allows the engine steps of root #169; it does not touch G2-G7.

## Reported without threshold

All per cell (bpw 3.05 / 3.5 / 4.5 x R 46 / 50 / 54 GiB x V25 / V37), with the same CI method where a mean is shown, every grid point and every fold, not only the chosen one:
- Belady-MIN reads per token, per-layer and global; LRU, SEED+LRU, SEED+LRU+IDPF; per-layer vs global arena for each.
- Ceiling per stage, cumulative, each as r, B / (r x S) tok/s and the ratio to the bar: static cut of G1 (0.5217 x 336 reads), LRU, SEED+LRU, + IDPF, MIN, oracle prefetch.
- Prefetch: reads issued, used, wasted, and demand misses not covered per token (the uncovered ones stall). Derived, not measured: one expert read takes S / B = 2.02 ms (4.5 bpw) or 1.35 ms (3.05 bpw) at 6.994 GB/s, against 0.56 ms per layer of the 25 ms token at 40 tok/s over 45 layers, so a prefetch one layer ahead cannot hide its own read; d is in the grid for that reason.
- PCIe bytes per token on the staging path = S x (arena hits in the pinned part + reads), DRAM bytes per token = S x (pinned hits + 2 x reads: written by the read, read again for the upload); a CPU lane for pinned experts is not assumed. Time floors at R_pcie 51.6 GB/s (`stage_cold_ca`, `docs/architecture.md:1414`, see erratum in `step08/20261008-g1.md`) and 54.6 GB/s: serial sum and overlapped max of the NVMe, PCIe terms.
- Depth buckets 0-32k / 32-100k (as G1, `tools/hotset-eval.py:159`), the first 1,000 generated positions vs the rest, the reset-at-first-generated-position variant, per-layer r, per-file r on every file (calibration files too, each as its own cold-started replay).

## Failure modes

- A G1d row timestamped before the commit of this file is invalid.
- A policy parameter, policy or grid point chosen with the held-out (including after seeing a held-out row to pick among variants) makes that cell "not answered".
- A failed self-test, a manifest not matching the corpus or the container, or a failed reproduction control (check 4) makes G1d "not answered" until the cause is named; it is never passed or failed on such data.
- A simulator that does not implement the cache model above as written (token order, tie order, prefetch accounting, capacity rule) gives "not answered"; a change to the model, grid, B, S, V, R or tolerance after the first G1d row needs a new PREREG.
- The G1 verdict, `PREREG.md` and its amendments are not edited by any G1d step.
- Limits stated now: one held-out task, one machine, one day; the sessions were written by other models and GLM reads them teacher-forced; the routed prefixes are session starts (32,768 tokens); the dumps are BF16-hand-over routing (covered by the engine re-measure of G4); a block bootstrap over 26 blocks is wide; id-only prefetch is weaker than router-based prefetch, which this data cannot score.

## Awaiting confirmation (robin)

1. The bar B / 40 tok/s with the upper CI bound, 12.3 reads at 4.5 bpw (decides the existing container) and 18.4 reads at 3.05 bpw (scenario only).
2. Primary cell: per-layer arena, R 46 GiB (`HOST_PINNED_CAP` unchanged), V25.
3. Whether 3.05 / 3.5 bpw stay simulation-only capacity scenarios (as written) or whether a 3.05 container is a topic at all.
4. The policy grid (s, P, d) and the tie order.
5. The reproduction tolerance |dm| <= 0.005 for the regenerated dumps.

---

Amendments (dated, below this line, written before the result they judge):

## Amendment 1 (2026-10-09 ~00:10 CEST, crow-nest `9207b32` + #179, before any G1d row and before any routing pass on the FP8 originals)

**Routing source = the FP8 originals, not the 4.5-bit container.** robin decided on 2026-10-08: the offline verdict G1d is taken; the routing passes run on the FP8 originals of `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a` (re-downloaded into `models/GLM-5.3-Flash-original/`, every file hash-verified by `tools/fetch-glm.py`); no 4.5-bit container is rebuilt; the experts will be 3.05 bpw (MUL1 trellis, plan step 9), the dense part stays on the 4.5 recipe.

- **Why.** The container `converter/GLM-5.3-Flash-CNQ4.5.cnq` (sha256 `0684a823...`) and the step-8 dumps were deleted on 2026-10-08. The layerwise runner reads the FP8 originals directly (`--weights fp8-originals`, its default path); on layers 0-3 it took 19.6 s against 68.2 s from the container (step 6, 90 ids, 16 threads, `docs/glm5-reference-runner.md` section 7). The format the engine will run (3.05-bit experts) is made from the FP8 originals, so the 4.5-bit container's routing is no longer the routing of any planned build.
- **Driver.** `tools/glm_route_passes.py --fp8 models/GLM-5.3-Flash-original` (#179). It refuses a directory where `hf-revision.json` is not that revision's record with 62 shards, or where `config.json`, `model.safetensors.index.json` or any of the 62 shards lacks a `.verified` marker equal to that record and of that revision. It writes the weights' identity (kind, revision, index and config sha256, the 62 verified shard sha256, and the sha256 of that record) to `weights.json` in every pass dir and to each `passes.jsonl` row. The corpus rule is unchanged (amendment-1 sha256 of `PREREG.md`, `corpus.json` `8fb9560f...faae5f26`).
- **Check 2 is replaced** for G1d: each run dir's `manifest.json` has the corpus ids, 32,768 tokens, weights kind `fp8` (`weights.weights` starts with `fp8`), `index_json_sha256` equal to the sha256 of the verified `model.safetensors.index.json`, complete over every layer, `--state-dtype bf16 --prompt-chunk 512`; `weights.json` is the same in all five dirs and its `identity_sha256` equals `weights_identity_sha256` in each pass row. The identity sha256 is written into the measurement book with the first pass row. `glm_tier_sim.source_reasons` (`tools/glm_tier_sim.py:133`) calls non-CNQ routing "plausibility only"; for G1d that one reason is replaced by this check (the tool change is #178's); every other reason it gives still makes G1d "not answered".
- **Check 4 (reproduction control) can no longer be a 1:1 check and is replaced.** The G1 record (held-out m 0.5217, `g1-sim-20261008.json`) and the scratch LRU / MIN numbers above came from 4.5-container routing. FP8 routing differs from it: on step 6, layer 3, 90 rows, the top-8 sets agreed on 40 rows, overlap 0.911. A tolerance on m would test the quantisation, not the regeneration. In its place, all before the first G1d row, none with a threshold, none able to pass or fail G1d:
  1. the self-test of check 3 on all five new dumps (unchanged, and still required);
  2. `glm_tier_sim.py sim` at N 25, W 0, seed 20261008 on the new dumps: its held-out m reported side by side with 0.5217, with dm;
  3. `glm_tier_sim.py dyn` LRU and Belady MIN at capacities 108 / 122 / 130 / 145 / 216 on `todo-1006`: reported side by side with the scratch numbers (LRU 0.278 / 0.240 / 0.220 / 0.185 / 0.060, MIN 0.122 / 0.100 / 0.090 / 0.072 / 0.019), with the difference per cell.
  These rows go to the book, marked "FP8 routing vs 4.5-container routing, descriptive". Awaiting-confirmation item 5 (the tolerance |dm| <= 0.005) falls away.
- **Primary bpw for the verdict = 3.05.** The deciding cell becomes bpw 3.05, R 46 GiB, V25, per-layer arena, policy chosen on the calibration: **r_hi <= 18.4 reads per token** (same rule and bar B / 40 tok/s as above, still proposed and awaiting robin's confirmation as item 1). Record size S = **9,474,048 B (plan figure, to be confirmed by step 9)**; no step-9 ticket gave a measured size when this was written. If step 9 confirms another size before the first G1d row, C is recomputed with it by the capacity formula above and the bar r_hi <= B / (40 x S) restated in the book before that row; after the first G1d row the size is fixed. The 4.5 bpw cell (r_hi <= 12.3) and 3.5 bpw are reported with the same rule, without a gate role. The sentence above that 3.05 would be "a second quantisation of the 4.5-bit container" no longer applies: the 3.05-bit experts are made from the FP8 originals.
- **Limits added.** The dumps are FP8-weight routing (dequantized to f32, BF16 hand-over); the engine will route through 3.05-bit experts and a 4.5-recipe dense part, whose routing differs from both and is not measured (covered by the engine re-measure of G4). The held-out's G1 and scratch-LRU exposure stated above was on container routing; it stays a stated limit, and nothing chosen by the calibration has seen the held-out.
