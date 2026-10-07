# PREREG: the GLM-5.3-Flash series on crow-nest, gates G1–G7 (crow-nest #145, root nibor1896/Crow#362)

Written 2026-10-08 ~01:25 CEST on crow-nest `166bc7a`, before any row of this series exists (no download, no disk read test, no routing dump, no engine run). Thresholds: robin, 2026-10-08 (G5 decode ≥ 40 tok/s, target 60 not a gate; all others as the plan proposed them). Plan of record: vault `Crow/00_Notes/glm-5-3-flash-laeuft-nur-auf-crow-nest-der-container-bleibt-bei-4-5-bit-mit-gestufter-residenz.md`, sections "Schritte", "Tore", "Folgen".

## Fixed for the whole series

- Source: `zai-org/GLM-5.3-Flash`, revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, 62 shards = 328,337,455,672 B. Every shard checked by size (`siblings[].size`) and `lfs.sha256` from the HF API before it is converted; a mismatching shard is deleted and fetched again, never converted. Fetched from the original source only, never mirrored.
- Recipe: CNQ4.5 (NVFP4) for routed and shared experts and every other tensor not in the keep set; keep set in BF16 as the plan's step 4. Scale policy `--scales mse`, passed explicitly (converter default is `ceil`, `converter/src/main.rs:29`); `mse` is the policy Flash-Next CNQ4.5-M was built with (`docs/dense-originals.md:107`). `diag` is not used: it needs activation statistics from a reference that does not exist before step 7, and it would make the GLM recipe differ from the Flash-Next comparison arm in a second variable.
- Unchanged parameters: `HOST_PINNED_CAP` 46 GiB (`engine/src/geo.rs:42`), `CROW_RAM_MARGIN_GB` 1 (`engine/src/manager.rs:140`). No default, manifest (`stack.json`, `operating-point.json`) or flag is changed by any step before G7.
- No llama.cpp measurement: no speed row, no quality run, no llama row. llama.cpp source is read-only reference.
- Comparison arms: Flash-Next (running operating point) and the dense 27B, only as arms of G5/G6, never as input to a step.
- Every row, false starts and aborted runs included, goes to the Flash-Next measurement book (artifact `EcRQo5otLHnPqPUJXtZ9Xr`) with date, commit, env, context depth, RAM/VRAM.
- A row timestamped before the commit of this file is invalid.
- Derived constants (from `config.json`, not measured): expert block 14,155,776 B (3 × 4096 × 2048 weights, 36 B per 64 values); 336 expert visits per token = 4,756,340,736 B if all from disk; a 4096-token prefill chunk reads ≈ 102.7 GB at ~60 % non-resident experts.

## Abort rules

- A failed gate stays failed; its verdict is recorded with the number and never reinterpreted. The series stops at a failed gate. Going on is robin's decision and only along that gate's row in "Consequences".
- A fix of an engine defect after a G3 failure allows a new G3 run under the unchanged criterion on anchors not used before (the `decode_out/q8-kld2/PREREG.md` form); the first verdict stays in the book.
- Under 5 tok/s decode (G5 window median) the quality series (step 17) is not run; robin decides.
- A gate whose inputs are void (see each gate) is "not answered", not passed.

## Amendments

Dated, appended below the line at the end, written before the result they judge, with the reason and what had been seen when written (the `decode_out/hotset-0924/PREREG.md` form). No threshold changes after the first row of the gate it governs. Thresholds change only by a new PREREG.

---

## Step 3 — NVMe read rate B (input to G1)

- Question: sustained read rate of expert-sized blocks from this machine's NVMe.
- Metric: B in GB/s (10⁹ B/s) per reader count; sequential rate reported beside it, no threshold.
- Data: the GLM shards that step 6 needs anyway; no file of another model.
- Method: blocks of 14,155,776 B at 4096-aligned random offsets, `FILE_FLAG_NO_BUFFERING` (unbuffered, no page cache), one handle per reader, 1, 2 and 4 readers.
- n: ≥ 3 repetitions per reader count.
- Statistic: median per reader count. B for G1 = the median at the best reader count measured; that reader count is binding for step 14.
- Robustness: spread max/min over the repetitions ≤ 1.15. Above it the number is named as cache or driver noise and does not enter G1; G1 stays "not answered" until a B within 1.15 exists.
- Void: a run with a second process reading or writing the disk.

## G1 (step 8) — does the tiered residency carry?

- Question: is the share of expert visits that fall to the NVMe small enough for 40 tok/s, and does B allow the prefill line?
- Metric: m = NVMe expert visits / all routed expert visits per token, offline tier simulation over teacher-forced top-8 routing in token order: static hot set N per layer (VRAM), pinned tier 83 per layer, NVMe window with the planned replacement. Primary on generated positions (assistant reasoning, text, tool calls) of the held-out file.
- Routing source: step 7 runner on the dequantised full container (step 9). A number from the partial container is plausibility only and does not count.
- Data: calibration transcripts and one held-out file, named with sha256 in a dated amendment before any routing dump of step 8 exists. The held-out is a different task than every calibration file and is never used for cutting the hot set.
- n: all generated positions of the held-out file (count reported).
- CI: 95 % block bootstrap over the token sequence, non-overlapping 1,000-token blocks, 2,000 resamples (`tools/hotset-eval.py:24,80-86`, Künsch 1989).
- Robustness: leave-one-out over the calibration plus held-out files (cut on all but one, score on the one left out), every fold reported; depth buckets 0–32k, 32k–100k, 100k+ (`tools/hotset-eval.py:159`), reported; m at N = 25 … 37, reported.
- Threshold (all must hold), judged at N = 25 hot experts per layer (low end of the derived 25–37):
  - upper bound of the 95 % CI of m ≤ m* = B / 190.3 GB/s (40 tok/s × 4.7563 GB per token);
  - prefill bound 39.9 × B ≥ 150 tok/s, i.e. B ≥ 3.76 GB/s (the G5 cold-prefill line).
- Consequence: pass → steps 11 ff. Fail (m > m* or 39.9 × B < 150) → the port is not built further (steps 11 ff.); finding to the book; options to robin: more RAM (128 GiB, bandwidth with four DIMMs unmeasured) or end.

## G2 (step 10) — does lossless entropy coding pay?

- Question: does a static order-0 entropy coder per expert block shrink the expert bytes enough to build option B?
- Metric: saving = 1 − (coded bytes incl. code tables per block) / (raw NVFP4 bytes), summed over all 12,096 expert blocks; codes (16 classes) and scale bytes (256 classes) both counted, from the converter histogram written during step 9.
- Data: the full container (census, not a sample).
- n: 12,096 expert blocks, histogram sum = value count (step 5 test).
- CI: none; census. Order-0 limit stated: a context coder may save more and costs more decode time.
- Robustness: saving per class (gate/up/down, shared) and per layer reported; spread min–max reported.
- Threshold: saving ≥ 8 % including tables AND a decoder draft for the staging path names the decode time per cold visit.
- Consequence: pass → conditional step 19. Fail → option B stays unbuilt.

## G3 (steps 6, 15) — does the engine compute correctly?

- Question: does the engine's math match the HF reference on the same weights, layer by layer and for the whole model?
- Per layer (steps 6, 13):
  - Metric: cosine of the layer output (all hidden streams, all rows of the anchor, flattened) engine vs. golden of step 7 on the container-dequantised weights; max abs deviation reported.
  - Data: the anchors below; layers 0 and 3 in step 6, every layer in step 13.
  - Threshold: cosine ≥ 0.9999 per layer per anchor.
- Whole model (step 15):
  - Metric: mean KL(reference ‖ engine), natural log, over the anchor's scored rows; top-1 agreement.
  - Noise control (mandatory): the same engine with another rounding order (another kernel path) against itself, noise = mean KL(E ‖ E′) per anchor.
  - Data: ≥ 6 anchors at distinct context depths, ids and sha256 fixed in a dated amendment before the first parity row; no anchor reused from an earlier run.
  - n: ≥ 6 anchors; rows per anchor fixed in the same amendment.
  - CI: none; the noise arm is the control (q8-kld2 form).
  - Threshold: every anchor KL ≤ max(0.073, 3 × noise), and top-1 ≥ 98 % pooled over all scored rows (per anchor reported). One failing anchor fails G3. rc ≠ 0 or NaN in a dump fails its anchor.
  - Robustness: the quantisation error against the FP8 originals is reported separately, never mixed into the pass.
- Consequence: fail → cause named per layer; engine defects fixed; gate unchanged (abort rules).

## G4 (step 14) — does it boot and hold?

- Question: does the NVMe tier boot at 200k context and survive a cold long turn within VRAM and commit?
- Metric: boot success with one cold 31,979-token turn; VRAM after the turn (`nvidia-smi`); peak commit charge; engine-measured m over that turn.
- Data: the 31,979-token turn (synthetic row of step 16); hot sets from step 8.
- n: 10 boots.
- CI: none for boots (10/10 binary); the engine m is judged against the step-8 95 % CI.
- Threshold (all): 10/10 boots; VRAM < 31.9 GiB after the turn; commit under the limit (RAM + page file 103.38 GiB, pinned and VRAM counted 1:1); engine m inside the step-8 95 % CI.
- Robustness: engine m above the CI means the teacher-forced assumption was wrong; it is a measured value and G1 is re-judged with it, not silently adjusted. VRAM ≥ 31.9 GiB is not an operating point.

## G5 (step 16) — fast enough?

- Question: is decode on Crow's real path at least as fast as the running Flash-Next point?
- Metric: decode tok/s per turn in the Crow window (`python cli\crow_gui.py --base-url http://127.0.0.1:8099/v1` against `serve`), no manifest or default change; cold prefill tok/s at ~32k; wall clock per task; each row with depth, `prompt_n`, RAM/VRAM, m.
- Data: ≥ 5 real agent turns at 7–10k context in the window; synthetic row: cold 31,979-token turn over 3 boots (never carries alone); Flash-Next on the same tasks the same day, order Flash-Next → GLM → Flash-Next.
- n: ≥ 5 window turns; 3 boots.
- Statistic: median.
- Threshold (all):
  - decode median ≥ 40 tok/s in the window (target 60, not a gate);
  - cold prefill at ~32k ≥ 150 tok/s;
  - wall clock per task ≤ 2.0 × Flash-Next the same day;
  - spread max/min ≤ 1.15 over the 3 boots.
- Robustness: a number that holds only in a script and not in the window is "no operating point" (STAGE_PAR: +6.7 % in A/B, halved in Crow).
- Consequence: fail with G6 passed → levers one at a time through this same gate: entropy coding (step 19), MTP (step 21); no threshold change without a new PREREG. Decode median < 5 tok/s → step 17 not run.

## G6 (step 17) — good enough?

- Question: is GLM-5.3-Flash not measurably worse than Flash-Next on the ten-task run, blind?
- Method: `tools/ten-task-run.py` (`run`, `blind`, `tokens`, `report`) per arm against `serve`, one model on the card at a time; one wire body for all arms (temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence_penalty 0.0, `reasoning_effort high`, `max_tokens` 16384); order Flash-Next → GLM → Flash-Next the same day; the 27B as a further arm, reported, no threshold.
- n: ≥ 3 runs per arm; median and range.
- Metrics and thresholds (all):
  - Q1 median correct tasks GLM ≥ Flash-Next;
  - Q2 tokens per answer GLM ≤ 1.5 × Flash-Next;
  - Q3 `finish_reason length` count GLM ≤ Flash-Next (`length` counts as undecided, not correct);
  - Q4 control arm: the same engine with deliberately degraded expert selection (4 instead of 8 active experts, built only for this arm) breaks at least one of Q1–Q3. If it does not, G6 is "not answered".
- Robustness: scores by robin blind; tasks 5, 7, 8 (agent, long context, synthesis) reported separately. "Tool calls arrive" is not a quality gate.
- Pass means "not measurably worse", not "better".
- Consequence: fail or "not answered" → no local point; GLM stays a remote spot.

## G7 (step 18) — does it become an operating point?

- Question: fourth operating point or not.
- Criterion: robin's decision on the recorded verdicts G1–G6 (each "passed / failed / not answered" with its number).
- Consequence: all passed → proposal to robin: step 22 (product wiring; the ~180 GB upload is its own go).

---

Amendments (dated, below this line):
