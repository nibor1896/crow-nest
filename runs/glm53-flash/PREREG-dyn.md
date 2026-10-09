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


## Amendment 2 (2026-10-09 ~08:30 CEST, crow-nest `3f72673` + #178/#179 capture route, before any G1d row and before any routing pass of record)

**Calibration routing from the MUL1 conversion's capture; one #179 pass for the held-out.** The routing of the four calibration files is not computed again by four runner passes. It is taken from the capture of the running MUL1 conversion (#182). The held-out `todo-1006` gets one pass on the FP8 originals: `tools/glm_route_passes.py --fp8 models/GLM-5.3-Flash-original --files todo-1006`.

- **What the capture is.** `tools/glm_mul1_quantize.py capture` runs `omarchy-0915a`, `lenis-0830`, `ctx7-0830` and `zetalab-0829` (amendment-1 ids, checked against the pinned sha256 by `glm_route_passes.check_corpus`) layer by layer. It uses `oracle/glm5_layerwise.py` `run_layer`, the function the runner's `run` calls for every layer, with all 32,768 rows as prompt, `--prompt-chunk 512` and BF16 hand-over states (the pass options of amendment 1). It reads the FP8 originals that `glm_route_passes.fp8_identity` verifies (identity `f47bd1541fed05c6a27abcc8dc17023fb83ee1a52c02f9f5c46adbcace7bfc90` in the conversion's `calibration.json` and every `capture.json`). Per MoE layer it writes `routing[0]` of `run_layer` as `ids.i32`, int32 [131,072][8], ascending. That is the array a pass writes as `l<k>-routing-ids.i32`, here with the four files' rows one after another in the order above. The only difference in procedure is the loop order: one layer for all four files instead of one file through all layers.
- **Kept.** `quantize` deletes a layer's `ids.i32` after its records, so a copier saves `ids.i32` and `capture.json` per layer into `decode_out/glm-step8/capture-ids/L<ll>/`, from layer 4 on.
- **Layer 3** was pruned before the copier started. Only the conversion's `L03/capture.json` is left, with `ids_sha256` `4b46267eb5658e21d5e97ab55a5577de6192af2a4e4193206e29ccdfc320a23a`. Layer 3 is recomputed by the same tool in a separate work dir: `capture --fp8 models/GLM-5.3-Flash-original --work decode_out/glm-step8/capture-l3 --layers 3 --max-ahead 0`. It advances the four files through layers 0-3 and captures layer 3; `capture` never deletes `ids.i32`. Its `ids.i32` and `capture.json` are copied to `capture-ids/L03/`. Its `ids_sha256` is written into the book next to `4b46267e...`. Equality is expected, but it is not a check. A difference is recorded with the share of rows whose top-8 sets agree, and the recomputed layer is used if it passes the checks below.
- **Check rules** (`tools/glm_tier_sim.py dyn --capture decode_out/glm-step8/capture-ids`). Any failure refuses the run by name, exit 2, and no row is taken. For every MoE layer 3..44:
  1. `L<ll>/ids.i32` and `L<ll>/capture.json` exist (every missing layer named), and `capture.json` names layer `ll`;
  2. its `identity` equals the `identity_sha256` of the held-out pass dir's `weights.json`, which check 2 of amendment 1 ties to the verified FP8 originals and to the `passes.jsonl` row;
  3. its `files` are the corpus' calibration files in corpus order (`omarchy-0915a`, `lenis-0830`, `ctx7-0830`, `zetalab-0829`);
  4. each file's `rows` equals the length of its ids file (32,768), the total equals their sum (131,072), and `top_k` is 8;
  5. the sha256 of `ids.i32` equals `ids_sha256`, and the file holds rows x 8 ids;
  6. the self-test of check 3: ids in 0..287, eight distinct ascending ids per row.
  Each file's routing is its slice in that order. Generated and prompt positions come from `decode_out/glm-step8/corpus/<name>-mask.json`, as for pass dirs, and each calibration ids file must match `corpus.json`.
- **Amendment 1 adjusted, nothing else.** Check 2 covers the held-out pass dir. "`weights.json` is the same in all five dirs" becomes "every layer's capture identity equals the held-out's `weights.json` identity", and only the held-out needs an ok row in `passes.jsonl`. Check 3 for the calibration files is rule 6 above. In check 4, item 1 is the held-out self-test plus these rules. Item 2 runs `glm_tier_sim.py sim --capture`, which adds a "plausibility only" reason, so G1 stays "not answered". Item 3 runs `dyn --capture`. Corpus, held-out, cache model, grid, bar, B, S, V, R and the verdict rules are unchanged.
- **Why.** The capture is the same routing computation on the same weights, and reusing it saves four of the five passes: about 5.2 h of CPU (4 x 4,650 s, derived in `docs/glm5-reference-runner.md` section 8). What remains is the held-out pass (about 1.3 h, derived) and the layer-3 recompute, which took 20.5 min in the conversion (04:55:15 to 05:15:46).
- **Limit added.** The calibration routing and the held-out routing come from two runs of the same code. If the CPU's f32 reductions are not bitwise reproducible across runs, near-tied router scores could pick a different top-8 set in a few rows. The layer-3 comparison above measures that once.


## Amendment 3 (2026-10-09 ~08:50 CEST, crow-nest `4b706be` + the `g1d` command of #178, before any G1d row)

**How `tools/glm_tier_sim.py g1d` reads the points this file leaves open.** Nothing here changes a threshold, the bar, the grid, the tie order, B, S, V, R, the primary cell or a verdict rule; those stay as written and still await robin's confirmation. Each reading below is fixed now, before any G1d row, and the tool's known-answer tests pin it.

1. **Capacity.** C = floor(floor((V + R) / S) / 42), with S of this file (3.5 bpw: 11,010,048 B; not `dyn`'s derived 10,871,858). This reproduces every entry of the capacity table (108 / 115 / 122, 138 / 148 / 157, 161 / 172 / 183 at V25; 120 / 127 / 134, 154 / 163 / 172, 179 / 190 / 201 at V37).
2. **Split of C.** Seed slots = floor(s x C), for example 40 and 80 at C 161. Prefetch buffer = P. LRU part = C - seed - P. A point whose LRU part would be negative does not fit C and is skipped. This cannot happen at C >= 108 (at most 54 + 16 = 70 slots).
3. **Seed.** The floor(s x C) experts of lowest rank under `cut`'s rule (routed count over the generated positions of the statistics files, ties lower id), per layer.
4. **IDPF.**
   - The table T[l][a, b] counts the generated positions of the statistics files with a among layer l's ids and b among layer l+d's ids of the same token. The score of expert b for layer l+d is the sum of T[l][a, b] over the eight layer-l ids a.
   - The P experts with the highest scores among those not resident (seed or LRU part) are read, ties lower id: min(P, non-resident count) reads, each counted at that token.
   - Only MoE layers are sources, so layers 3 .. 2+d get no prefetch.
   - In the per-layer arena, the proposal for layer m is made from layer m's state before its visit of that token. That equals issuing it after layer m-d's visit.
   - A demanded buffered expert joins the LRU part as most recent. The rest of the buffer is dropped after layer m's visit.
5. **Demand.** A visit to an expert that is neither seeded, in the LRU part nor buffered is one read and joins the LRU part as most recent (with an LRU part of 0 it is not kept). The eight ids of a (token, layer) are visited in ascending order.
6. **Score and choice.** A file's score is its reads per generated position, replayed alone from an empty arena at position 0. A point's score is the unweighted mean over the files. Equal means are equal floats, and ties follow this file's order: simpler class (LRU < SEED+LRU < SEED+LRU+IDPF), then smaller s, P, d. LRU is s 0, P 0; SEED+LRU is s > 0, P 0; any P > 0 is SEED+LRU+IDPF.
7. **Statistics per choice.**
   - The full choice fits seed and table on all four calibration files, and the held-out is scored with those.
   - Each leave-one-out fold fits them on the other three, chooses on those three and scores the left-out file with that point and those statistics.
8. **The bar** is evaluated as r_hi <= B / (40 x S) without rounding: 18.455 reads at 3.05 bpw, 12.351 at 4.5 and 15.880 at 3.5, with B 6.9936611328 GB/s. The 18.4 / 12.3 / 15.88 above are its roundings. r_hi is the upper bound of `boot_ci`.
9. **The book.** Before the held-out is replayed for a cell, the chosen point is appended to `<json>.choices.jsonl` with its UTC time. That line goes to the book.
10. **Built or not.**
    - Built, all reported without threshold: every grid point on every calibration file, every fold, the held-out row with the chosen point, Belady MIN at C, per-layer r, the depth buckets, the first 1,000 generated positions against the rest, and the reset-at-first-generated variant.
    - Not built, all reported-only: SEED+LRU and IDPF in the global arena, the oracle-prefetch stage row, the stage table against the static cut, and PCIe / DRAM bytes for these policies.
    - None of them decides the gate.

## Amendment 4 (2026-10-09, before any G1d row): robin's confirmation

robin, 2026-10-09 (chat, translated): "I think the thresholds fit, as you said." Confirmed therefore, unchanged:
1. The bar r_hi x S <= B / 40 tok/s on the upper 95 % CI bound; the deciding cell is 3.05 bpw (S = 9,474,048 B, confirmed by #181): **r_hi <= 18.4 NVMe reads per token** (5.5 % of 336 visits, B 6.994 GB/s, 1 reader). 4.5 and 3.5 bpw are reported without a gate role.
2. Primary cell: per-layer arena, R 46 GiB (`HOST_PINNED_CAP` unchanged), V25, C 161.
3. The policy grid s {0, 0.25, 0.5} x P {0, 4, 8, 16} x d {1, 2, 4} and the tie order as written.
Items 3 and 5 of the original list were settled by amendment 1. Also decided by robin on 2026-10-09: the DSA indexer cache stays in the HF / zai-org layout (257 values, BF16, 514 B per token and layer, #163); the f32 / llama.cpp layouts are not used.

## Amendment 5 (2026-10-09, before any G1d row): G1d is diagnostic, not a stop

robin, 2026-10-09 (chat, translated): "G1d is no exclusion criterion for me while not all levers are in; for me the acceptance is at the end, not at an intermediate step that runs without optimisation." Therefore: the G1d verdict (bar and cell as in amendment 4) is still computed and recorded as pass / fail with its number, and it still chooses the cache policy for plan step 16; but a fail does **not** stop the series. The series runs on with all levers (dynamic cache, NVMe tier, prefetch, CPU lane, MTP); acceptance is at the end through G3–G6 (`runs/glm53-flash/PREREG.md`), G5 decode >= 40 tok/s in the Crow window unchanged. Thresholds are not changed by this amendment.
