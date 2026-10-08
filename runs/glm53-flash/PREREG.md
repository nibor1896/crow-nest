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

## Amendment 1 — 2026-10-08 ~03:10 CEST: the G1 corpus, the window rule and the dump plan (crow-nest #147)

Written on crow-nest `f4f54e4` (branch `glm-step8`) before any routing dump of step 8 exists. Seen when written: no GLM routing over any corpus file. GLM data seen so far: the step-3 rows (`runs/glm53-flash/step03/20261008T001819Z.md`, no valid B) and the step-6 goldens of layers 0–3 over 90 fixed ids that are not from this corpus (`runs/glm53-flash/step06/ids-source.json`; their layer-3 routing was read only as the FP8-vs-CNQ overlap 0.911). Seen for the corpus: token and generated-span counts with GLM's tokenizer and template (below), and a dry run of `tools/glm_tier_sim.py sim` on synthetic random routing of the same sizes (not GLM; none of its numbers is a GLM number). Reason: G1's data clause ("named with sha256 in a dated amendment before any routing dump") and the open terms of its metric (window, positions). No threshold is touched.

**Corpus.** Crow agent sessions in the form of the 2026-09-24 Flash-Next calibration (`decode_out/hotset-0924/PREREG.md`: session JSON, rendered with `tools/session_ids.py`'s `messages` and `render`, generated spans as there). The 2026-09-23 diorama archives of that calibration are not on this Windows machine; these are robin's own sessions on it. Rendered by `tools/glm_tier_sim.py corpus` through GLM's own template and tokenizer (`models/GLM-5.3-Flash-original`, rev `eb9eb208`: `tokenizer.json` sha256 `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d`, `chat_template.jinja` sha256 `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5`), thinking on (template default), no tool schemas; every assistant span matched (0 skipped in all five files). Each file is routed over its first 32,768 tokens: a token prefix of the full render, so every included position routes as in the full file (the cap follows from the runner's memory and disk, see the dump plan). Session files under `C:\Users\robin\AppData\Local\`. The ids and masks go to `decode_out/glm-step8/corpus/`, which is git-ignored because the ids decode to private transcripts; `corpus.json` there has sha256 `8fb9560f43a58ed5a1791ed2c458130a0febf4236f39ef882c12e665faae5f26`.

| name | role | task | source (under `AppData\Local\`) | bytes | source sha256 | GLM tokens full / routed | generated full / routed | ids sha256 (routed) | mask sha256 |
|---|---|---|---|---|---|---|---|---|---|
| `todo-1006` | **held-out** | coding: build a Python todo CLI in goal mode (2026-10-06) | `Crow\session\chat-20261006-165103.json` | 202,922 | `2139720d6e14c14c0d61898ece31267180ddc9459025d8d3b68beca865cc0b2c` | 46,330 / 32,768 | 36,799 / **26,599** | `0c30a34d5d212fc9f072f803dd964a6b45f56335e157247bfe7fb09874f359e3` | `c306f565542c1f7d6c42605a25cea37439711a08e17d514f60e5c9a0aa5c2674` |
| `omarchy-0915a` | calibration | ops: Linux dual-boot install troubleshooting (2026-09-15) | `Crow.old\session\rollover-20260915-081727.json` | 638,510 | `dab5d51e86b8d57addf540f7dc9476348e3cc919dd2bc5a7b49ac16e79c4ba13` | 164,494 / 32,768 | 114,885 / 5,441 | `083567463f108b8a562d2d398013b24164573bbfe4808d0a5f9043f83c7e7380` | `5398ae6c1691bdfa921ce285e4ec4a8aee4ad1b683b7ff87d33f60935ab7325b` |
| `lenis-0830` | calibration | web: build a website with delegated research (2026-08-30) | `Crow.old\session\archiv\chat-20260830-131505.json` | 350,409 | `366cd62eddfd10871aeeddef93fb497721d3ea381519fab7601562b3d1b72b62` | 100,176 / 32,768 | 73,244 / 7,136 | `7f411cf6a2a44bbd90714ca123201c4a4bd875849b3fdae25ddf3444c0e83085` | `01b384512eb3620a0df2ba56b705f26b6337a18971833a4fe54c3c90a374b037` |
| `ctx7-0830` | calibration | research: library docs into the knowledge base (2026-08-30) | `Crow.old\session\archiv\chat-20260830-174113.json` | 334,811 | `cad962fe6a34cc1bc4adacca1401d3ee7bf06b079360f86e1158405793eb1740` | 85,033 / 32,768 | 46,271 / 10,450 | `d73e494ed333be8af6b5fc1c07d9da82d873676b453a49cbb467e234c2790044` | `b84763b3335b0899cc00cefd666f8bcfcc010109096549811ce34137e9cd4960` |
| `zetalab-0829` | calibration | numerics: zeta-zero search lab, after a rollover (2026-08-29) | `Crow.old\session\archiv\chat-20260829-110038.json` | 566,407 | `ea719d142cad122a0f49f396cb90d9f4b7c1c3d292815257486df575e27ea263` | 175,714 / 32,768 | 126,977 / 15,166 | `4e96d0d09f84c070c4f914dfcd3d0580693d8782e540db74cce162f258953bf7` | `7cf87ef8191f33f105b9fde090d1e85e51933853e172426abbeb277f09fa0a51` |

- The held-out has a task no calibration file has and is never used for the cut: `glm_tier_sim.py` refuses a corpus where it shares a task, a source or its ids with a calibration file, and refuses a cut that contains it. Calibration: 38,193 generated positions over four tasks.
- Limits stated now: the sessions were written by other models (Flash-Next, Qwen) and GLM reads them teacher-forced; the routed prefixes are the start of each session, where tool output dominates (5,441 to 15,166 generated of 32,768 in the calibration files); one held-out task on one machine, so the held-out m may be optimistic or pessimistic for other work by an unknown amount.
- n for G1 = the 26,599 generated positions of `todo-1006` (26 blocks of 1,000). Depth buckets: with the cap, 0–32k holds all but the last 768 positions of a file, 32k–100k at most 768, 100k+ none; reported as they fall, without a CI under two blocks.

**Window rule.** NVMe window = per MoE layer, W slots of LRU over the NVMe-tier experts, in token order over every position of the file (Eliseev & Mazur, arXiv:2312.17238 §3.1). The plan's "NVMe-Fenster mit der vorgesehenen Ersetzung" names no policy; ticket #147 names LRU. Belady's MIN with bypass over the same W is reported as the ceiling, never as the policy. Neither this PREREG nor the planner (step 11, unbuilt) fixes W, so the G1 verdict is judged at W = 0. LRU is a stack algorithm (Mattson, Gecsei, Slutz, Traiger, IBM Systems Journal 9(2), 1970): its NVMe reads never grow with W, so a pass at W = 0 holds for every W. W ∈ {8, 16, 32} are reported. A fail at W = 0 beside a pass at some W > 0 is recorded as the fail and goes to robin with that row; it is not a pass.

**Dump plan (derived from step 6 and the HF code, not measured).** One runner pass per file, five passes, one at a time, no engine and no other heavy job beside it, `--weights container <full container> --ids decode_out/glm-step8/corpus/<name>-ids.json --anchors 32767 --state-dtype bf16 --out decode_out/glm-step8/runs/<name>`. `sim` runs once, after all five passes exist.
- Positions: all 32,768 of each file (the window and the prefill contrast need every position); G1 reads the generated ones.
- The runner as built cannot run 32,768 prompt rows in its one prompt call: eager attention in the 11 DSA layers holds f32 [64][q][kv] scores in about three live copies plus the mask, ≈ 776 B × q × kv: 13.0 GB at q = kv = 4,096, 833 GB at 32,768.
- As built, the only form is 4,096 prompt rows plus `--decode 28672` teacher-forced single-row calls (causal: the same routing up to rounding order). Per pass: load 42 × 52.1 s + 3 × 2.0 s ≈ 36.6 min (every MoE layer assumed to load like step 6's layer 3); prompt call ≈ 31 GFLOP per token at an assumed 0.5 TFLOP/s f32 → ≈ 4 min; each single row reads 9 f32 experts per MoE layer (905,969,664 B) at an assumed 60 GB/s → 0.63 s per row → ≈ 5.1 h; and in each DSA layer HF's `DynamicCache` appends the expanded MLA K/V (64 heads × 512 values × 4 B = 128 KiB per token) with `torch.cat` (`transformers/cache_utils.py:144-145`), so every row copies the whole cache and attention reads it once more: ≈ 1.04e14 B per DSA layer ≈ 58 min, ≈ 10.6 h over 11 layers. ≈ 16 h per pass, ≈ 82 h for the corpus: not practical.
- Planned form: a prompt-chunk option in the runner (prompt rows in calls of 512 against the layer cache; not built, requested in #147; step-7 file set). Per pass ≈ 36.6 min load + ≈ 34 min compute (31 GFLOP per token, assumed 0.5 TFLOP/s) + ≈ 13 min eager DSA attention (65,536 FLOP × T² / 2 per DSA layer) + ≈ 5 s of cache copies per DSA layer ≈ 1.4 h, ≈ 7 h for the corpus. The routing is the same in either form up to rounding order. Step 6's 0.60 s for 90 rows of layer 3 is too small a batch to calibrate any of these rates; the first pass measures them.
- RAM per pass: 28.0 GiB RSS after a MoE layer's load (step 6, layer 3) + 2 × 2 GiB of f32 hand-over state ([32,768][4][4096], in and out) + 4 GiB of expanded MLA K/V in a DSA layer at 32,768 tokens + ≈ 13.0 GB (12.1 GiB) in the largest attention call ≈ 48 GiB of 63.38 GiB.
- Disk per pass: the runner keeps every layer's hand-over state, 45 × 1 GiB in BF16 (`--state-dtype bf16`, the runner's mode for long routing runs; f32 would be 90 GiB). Kept after a pass: the routing files (≈ 88 MB per file) and the manifest. BF16 hand-over rounds every layer's input against an f32 reference; its effect on routing is not measured here and is covered by the engine re-measure of step 14 (G4).
- Every pass, also an aborted one, is a row in the Flash-Next measurement book.

### Amendment 2 — 2026-10-08 ~07:30 CEST (administrative, no criterion changed)

robin asked on 2026-10-08 morning for a GLM-5.3-Flash measurement book of its own. From now on every row of this series goes to https://claude.ai/artifact/MpmBA5NTAguy8EyHcRgoV4 ("GLM-5.3-Flash Messbuch") instead of the Flash-Next book named in "Fixed for the whole series" (`EcRQo5otLHnPqPUJXtZ9Xr`); the 14 rows written there on 2026-10-08 were moved. No threshold, metric, data set or method changes.
