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

### Amendment 3 — 2026-10-08 ~09:20 CEST (owner confirmations, before any G1 routing row)

robin confirmed on 2026-10-08, before any routing dump or G1 row exists:
1. **Recipe as committed.** KDA q/k/v/o projections, KDA short conv and MLA `kv_a`/`kv_b` stay NVFP4 in the container (the plan's step-4 keep set, "everything else NVFP4"), although the checkpoint keeps them BF16 and llama.cpp leaves them unquantized. A BF16 variant for exactly these tensors stays an option: fetched by HTTP range from the originals (~6.6 GB) and loaded as a dense BF16 overlay (`CROW_CNQ_OVERLAY`, #77) instead of a re-conversion; it is measured only if G3 or G6 point at these tensors, as its own arm through the same gates.
2. **G1 as amendment 1 set it:** judged at N = 25 static hot experts per layer and NVMe window W = 0 (strict case); W > 0 is reported, not judged.
No threshold changes.

### Amendment 4 — 2026-10-08 ~09:43 CEST: data for the step-3 re-run (crow-nest #146)

Written on crow-nest `e476bd6` before the re-run of step 3 exists. Seen when written: the first step-3 run (`runs/glm53-flash/step03/20261008T001819Z.md`: valid, medians 6.994 / 9.765 / 7.811 GB/s at 1 / 2 / 4 readers, spreads 1.003 / 1.232 / 1.430, so no B for G1), and the finished step-9 container (#157: 62/62 shards verified, index v2, sha256 below). Reason: step 3's data clause names "the GLM shards that step 6 needs anyway"; those seven shards were converted and deleted by the staged conversion (`models/GLM-5.3-Flash-original/stage.log`, all 62 shards deleted by 09:38:36). Fetching them again (33.4 GB) would read a file no later step reads; the container is the same model and the file step 14's NVMe tier reads.

**Data for the re-run.** `converter/GLM-5.3-Flash-CNQ4.5.cnq`, 178,490,593,097 B, sha256 `0684a8230264afa9f7387b2d02a164b4822d722b753a92113d18ea23f8e89c82`. It holds 12,096 routed-expert blocks of 14,155,776 B (gate + up + down back to back), each starting at an absolute file offset that is a multiple of 4096 (0 exceptions, read from the index trailer), 171,228,266,496 B = 95.93 % of the file; the rest is the dense part and the index. `tools/nvme_read_rate.py` takes no offset list: it draws 4096-aligned uniform random offsets over the whole file, which is the PREREG method as written ("4096-aligned random offsets"), so about 96 % of reads start inside the expert region and most span two neighbouring expert blocks instead of exactly one. No code is changed for this run.

**Unchanged:** block 14,155,776 B, `FILE_FLAG_NO_BUFFERING`, one handle per reader, readers 1, 2, 4, ≥ 3 interleaved repetitions, median per reader count, spread max/min ≤ 1.15, B for G1 = the median at the best reader count (binding for step 14, no fallback to another count), the void rule.

**Conditions recorded, not criteria:** no download, converter or runner on the disk; 10 consecutive minutes with ≤ 4 MB/s read and ≤ 4 MB/s written on the physical disk (its own counters, 1-minute windows) before the run; drive temperature (`IOCTL_STORAGE_QUERY_PROPERTY`, `StorageDeviceTemperatureProperty`, no admin) before and after. One run; a VOID run is repeated once. The first run stays in the book as recorded.

### Amendment 5 — 2026-10-08 ~10:25 CEST: B for G1 by owner decision (crow-nest #146)

Written on crow-nest `ed18b8d` before any G1 routing row exists (step 8 not started). Seen when written: both step-3 runs. `20261008T001819Z` (valid): medians 6.994 / 9.765 / 7.811 GB/s at 1 / 2 / 4 readers, spreads 1.003 / 1.232 / 1.430. `20261008T075202Z` (VOID, foreign reads 4.8 > 4.0 MB/s): medians 8.556 / 7.963 / 8.014 GB/s, spreads 1.003 / 1.613 / 1.401, drive 59 → 79 °C. Neither run gives a B under step 3's statistic, because the best reader count's spread exceeds 1.15 each time; 1 reader is flat in both runs, 2 and 4 readers fall within the run (throttling suspected, not measured). robin (2026-10-08): the disk is never idle in practice, and its peak rates are known.

**Decision (robin, 2026-10-08):** B for G1 = **6.994 GB/s**, the 1-reader median of the valid run `20261008T001819Z` (spread 1.003 ≤ 1.15). The reader count binding for step 14 is 1. For G1 this replaces step 3's "median at the best reader count" by "the 1-reader median of the valid run"; the robustness limit 1.15 and every G1 threshold are unchanged. Derived as in that run's table: m* = B / 190.3 = 3.68 %, prefill bound 39.9 × B = 279 tok/s. No further step-3 run is made for G1.

### Amendment 6 — 2026-10-11: no Flash-Next arm in G5 and G6 (crow-nest #151, #152)

Written 2026-10-11 on crow-nest `36b0d9d` (branch `g56-prereg`), before any G5 or G6 row exists. Seen when written: no Crow-window turn and no ten-task run of GLM-5.3-Flash; the GLM speed rows so far are `glm5_run` and `serve` script rows (GLM measurement book, table A; `docs/measurements.md`), which G5 does not count. No Flash-Next row was taken for this series. Nothing below touches G1–G4, G1d (`PREREG-dyn.md`) or any recorded result.

**Reason.** robin, 2026-10-10: GLM-5.3-Flash is accepted like a first model, no Flash-Next comparison.

**Form.** This removes the criteria whose only reference was the Flash-Next arm, on the owner's word and before the first row of either gate (the form of amendment 5, and of robin's 2026-10-08 decision on #152 to adapt G6 by amendment). It sets no new threshold and changes no remaining one.

**What changes.**
1. **"Fixed for the whole series", line "Comparison arms":** Flash-Next is no arm of G5 or G6. The dense 27B stays what G6 made it: a further arm, reported, no threshold, never in the verdict; it depended on no Flash-Next row and is kept. The sentence on `--scales` that names "the Flash-Next comparison arm" stays as the record of why the recipe was chosen.
2. **G5.** Question: is decode on Crow's real path fast enough? Data: ≥ 5 real agent turns at 7–10k context in the window and the synthetic cold 31,979-token turn over 3 boots, GLM only; no Flash-Next turns and no order Flash-Next → GLM → Flash-Next. Thresholds that stand unchanged (all must hold): decode median ≥ 40 tok/s in the window (target 60, not a gate); cold prefill at ~32k ≥ 150 tok/s; spread max/min ≤ 1.15 over the 3 boots. **Removed:** "wall clock per task ≤ 2.0 × Flash-Next the same day". Wall clock per task is still recorded per turn, without a threshold. Metric, robustness, consequence and the < 5 tok/s rule are unchanged.
3. **G6.** Question: is GLM-5.3-Flash good enough on the ten-task run, graded blind? Method, wire body, n ≥ 3 runs, blind grading by robin and the separate report of tasks 5, 7, 8 are unchanged; there is no Flash-Next arm and no order Flash-Next → GLM → Flash-Next. **Removed:** Q1 (median correct ≥ Flash-Next), Q2 (tokens per answer ≤ 1.5 × Flash-Next), Q3 (`length` count ≤ Flash-Next); each was defined only against Flash-Next. Q4's control arm (same engine, 4 instead of 8 active experts) stays in the series; its condition "breaks at least one of Q1–Q3" has no Q1–Q3 left and is open (below).
   What G6 compares against from now on, as far as the existing documents already fix it without Flash-Next:
   - each GLM answer against its task's own core requirements, graded pass / partial / fail by the rules of `docs/ten-task-expected.md` §1–2, final acceptance per task by robin (§5); `finish_reason length` stays undecided as written above;
   - robin's rule of 2026-10-08 (#152, #166): a failed task is judged by its kind of error, and a gross error disqualifies however many tasks pass. **Fixed now: zero S1** (#166 severity class S1) in the GLM arm;
   - the 4-of-8 control arm and the 27B arm (reported), as above.
   Until the open points below are answered by a further amendment, G6 can be judged failed (an S1) but not passed, and no G6 row is taken (#152: the #166 adaptation is due before step 17 starts).

**Open for robin (before the first G6 row):**
1. G6 bar for correct tasks: with Q1 gone, what minimum of correct tasks (median over the runs) counts as good enough, if any?
2. Tokens per answer and `length` aborts had Flash-Next as their reference: drop them, set absolute bars, or make the 27B (today reported only) their reference?
3. Control arm: what must the 4-of-8 arm break for G6 to count as answered (an S1, or the bar of point 1)?
4. #166 adaptation: k, the mild control, the process-error factor over the A/A arm, and #166's open S1 question (does an S1 still disqualify if the A/A arm shows the same S1 on the same task?).
5. G5 wall clock per task: no threshold, or an absolute bar per task?

### Amendment 7 — 2026-10-11: answers to amendment 6's open points (crow-nest #151, #152, #166)

Written 2026-10-11 on crow-nest `16d24c8` (branch `g56-amend7`), before any G5 or G6 row exists. Seen when written: what amendment 6 saw; since then no Crow-window turn, no ten-task run, no control-arm and no 27B row of this series. Form as amendment 6: owner's word, before the first row of either gate. A point marked **lead's decision** was set by the lead, under robin's guidance where he gave one; robin can overrule it by a further amendment before the first row of its gate.

**G5** (all else as amendment 6 left it):
1. **Decode bar 40 → 23.7 tok/s**, median over the ≥ 5 window turns in the Crow window. robin, 2026-10-11: 23.7 is, in his view, the most this hardware gives, and below it is no longer really a fail criterion. A median below 23.7 is therefore recorded as "below the bar" with its number; it is not a G5 failure under "Abort rules" and does not by itself stop the series (the < 5 tok/s rule for step 17 stands). Cold prefill at ~32k ≥ 150 tok/s and spread max/min ≤ 1.15 over 3 boots are unchanged and still fail G5.
2. **Wall clock per task:** recorded per turn, no threshold (robin confirmed, 2026-10-11).

**G6.** The dense 27B becomes G6's reference, "for now" (robin, 2026-10-11); this replaces amendment 6's "reported, no threshold, never in the verdict" for the 27B. Arms: GLM (8 of 8), 27B, 27B A/A, GLM 6 of 8, GLM 4 of 8, each n ≥ 3 runs with the one wire body. G6 passes only if all of 3–6 hold, and is judged only if 7 holds:
3. **Correct tasks:** GLM's median of correct tasks over its runs ≥ the 27B's median over its runs. robin, 2026-10-11: orient on normal other references; if too many tasks are wrong, it does not ship. **Lead's decision** under that guidance: the reference is the 27B arm, the only other model in the series.
4. **Tokens per answer and `length`** (reference the 27B, robin): tokens per answer GLM ≤ 1.5 × 27B; `finish_reason length` count GLM ≤ 27B (`length` stays undecided, not correct); the form Q2/Q3 had against Flash-Next. Unit: each arm's own `usage.completion_tokens` (the tokenizers differ), answer characters reported beside, as #152 "Proposed fix" 1 proposed.
5. **Zero S1** in the GLM arm (amendment 6, unchanged; see the S1 point below).
6. **Process errors:** GLM's process-error rate ≤ 1.5 × the A/A arm's (#166 item 9; factor below).
7. **Sensitivity (4-of-8 arm), lead's decision** (robin gave no view): G6 counts as answered only if robin's blind grades score the 4-of-8 arm worse than the GLM 8-of-8 arm, i.e. a lower median of correct tasks or at least one S1. If not, the grading is not sensitive enough and G6 is "not answered", whatever 3–6 show.

**#166 adaptation** (robin, 2026-10-11: yes). #166's defaults are proposals "fixed per series in its PREREG"; fixed here:
- **k = 3** generations per task and arm in each run: #166 item 4's default, which is the harness's three seeds (`tools/ten-task-run.py:128`, 2101–2103).
- **A/A arm:** #166 item 1, "the reference again with other seeds": the 27B once more, same wire body and n, seeds 2104–2106 (**lead's decision** on the seeds: the next three after the default, fixed before any run). It is the noise floor for point 6; its correct-task, token and `length` figures are reported beside the 27B's.
- **Mild control:** 6 of 8 active experts (`CROW_MOE_TOPK=6`, the setting the 4-of-8 arm uses, #152 "Proposed fix" 2), #166 item 1's first example. Reported, not in pass/fail; per #166 item 10, if it does not score worse than GLM 8-of-8 by the test of point 7, the verdict is recorded as resolving only collapses.
- **Process-error factor 1.5, lead's decision** (#166 names no value): the factor this series already uses for tokens per answer, so no new number enters. Process errors are #166 item 5's as far as the single-turn harness shows them (unclosed think tags, repeated loops; it sends no tools); `length` and tokens count under point 4 only. Rate = generations with ≥ 1 process error / all generations of the arm.
- **An S1 the reference also shows, lead's decision** (#166 leaves it open): a GLM S1 still disqualifies when the 27B or the A/A arm makes the same S1 on the same task; the row notes it. Reason: robin's rule of 2026-10-08 (a gross error disqualifies however many tasks pass) has no such exception.
- Not adopted: the rest of #166 (stage 1 KLD, which #166 skips across tokenizers; 20–40 Crow tasks with tools; the CI/margin rule, which point 3 replaces). G6's method stays as amendment 6 left it.

Amendment 6's open points 1–5 are closed by this amendment; G5 and G6 rows may be taken.

### Amendment 8 — 2026-10-11: the G5 turn set, the per-turn record and the run order (crow-nest #151)

Written 2026-10-11 on crow-nest `4bcda60` (branch `g5-tasks`), before any G5 row exists. Seen when written: what amendment 7 saw; since then no Crow-window turn and no G5 boot. GLM `serve` rows seen: the G4 series (`runs/glm53-flash/g3g4-20261011/`, local, not in git: 10 boots, one cold turn of `prompt-33k.txt` each, GLM `prompt_n` 30,986, arm3.env) and that day's serve session (unrelated requests, all COLD; #151 comment of 2026-10-11). Reason: #151 "Proposed fix" 1 asks for the ≥ 5 agent tasks with text, target folder and sha256 fixed in this file, which G5's data clause names but does not fix; the #151 comment of 2026-10-11 adds the prefix-cache record (robin promised its live check publicly). No threshold changes; G5's bars are amendment 7's. Points marked **lead's decision** can be overruled by robin by a further amendment before the first G5 row.

**Language:** English, the language robin types his own Crow tasks in on this machine (the 2026-10-06 sessions under `%LOCALAPPDATA%\Crow\session\`).

**Sources checked for reuse; none fits (empty folder, several tool rounds, 7–10k):**
- G6's ten tasks (`docs/ten-task-expected.md`, `docs/ten-task-prompts-crowlab.json`): not used. G6 is graded blind by robin; typing its prompts into the window would show him GLM's answers to them before the blind grading. They are also single-turn prompts that the harness sends without tools.
- The 2026-10-06 todo CLI (robin's typed text in `chat-20261006-165103.json`; the task of the Flash-Next live run `crow-lab/runs/fn-crow-task-win-20261006`): it runs far past the band (Flash-Next: 52 requests, final context 58,519 tokens), and it is G1's held-out file. Not used. Its form (build in "this folder", test, fix until green) is the model for the texts below, and t2 takes its clause "Run the tests and fix every failure until everything is green." verbatim.
- Crow `docs/acceptance/issue-207.md` (robin's live tests, 2026-09-21): Tests 1 and 2 search an existing repository, so the folder cannot be empty; Test 1b is a 500 MB output bomb, not a task. Not used.
- `chat-20261006-170158.json`, "how many files are in this folder + subfolder?": one tool round in an empty folder. Not used.
- The four calibration tasks of amendment 1: not used, because arm4.env's `CROW_GLM_ARENA_WARM` is cut from their routing (`arena-warm/glm53-cal4-decode-counts.json`, `_source`), and a task from them would flatter decode.

The five texts are therefore new, written for this amendment. None shares a task with G6, the G1 corpus or the arena warm set.

**Tasks.** Typed in this order, each in a new chat in its own fresh, empty folder `C:\Users\robin\Desktop\g5-run<k>\<id>` (k = 1 for the first session; a repeated session takes the next k and never reuses a folder). The texts name no path, so the hashes do not depend on k. sha256 is taken over the exact text between the fences, UTF-8, no trailing newline.

1. `t1-wordfreq`: word-frequency script on a sample file. 291 B, sha256 `3d7116ca612053248535aa51ad427c1dee390f5d3575850b0e1148204db7a32a`
   ```text
   Create a sample text file in this folder with about ten lines of English prose. Then write a Python script wordfreq.py that prints the ten most frequent words of the text file given as its argument, case-insensitive and ignoring punctuation. Run it on the sample file and show me the output.
   ```
2. `t2-roman`: Roman numerals with tests, fix until green. 299 B, sha256 `760946570c4363a97021603df30c852f33d16e8b43c8a7765a14151fba691c3c`
   ```text
   Write roman.py in this folder with a function to_roman(n) for 1 to 3999 and a function from_roman(s) that reverses it. Add unittest tests in test_roman.py with at least ten cases, including a round trip over every number from 1 to 3999. Run the tests and fix every failure until everything is green.
   ```
3. `t3-inventory`: CSV data and a report script. 258 B, sha256 `31834d40c99ebd064b8499fb397814ea33a6a08d1ba6c1bf5dcafa616b581c06`
   ```text
   Create inventory.csv in this folder with twelve rows of name, quantity and unit price. Then write report.py that reads it and prints the total stock value and the three items with the highest value (quantity times unit price). Run it and tell me the numbers.
   ```
4. `t4-units`: command-line unit converter with an error path. 308 B, sha256 `238052922297f4b9e46936e7c176070888a8e2d6157ca12b00079c96b5f1a2a5`
   ```text
   Write units.py in this folder, a command-line converter between metres, kilometres, feet and miles (units m, km, ft, mi), used like this: python units.py 5 km mi. An unknown unit must print a clear error and exit with code 1. Try three conversions and one unknown unit, and show me what it printed each time.
   ```
5. `t5-stats`: hand-written statistics checked against the standard library. 319 B, sha256 `ac02f0a2682e92c15267c8b927a693ea3d153f3124b9144f90d7ab39ccd0f39f`
   ```text
   Write mystats.py in this folder with mean, median and sample standard deviation, implemented by hand without the statistics module. Then write check.py that compares all three against Python's statistics module on five random lists and prints for each whether they agree within 1e-9. Run check.py and report the result.
   ```

**What a turn is and what counts (lead's decision, the reading of G5's "turns at 7–10k context").**
- A row is one request Crow sends to `serve`, i.e. one round. A task (one typed line) is one G5 turn. Its rounds are the requests from the typed line to the end of Crow's turn; Crow's turn line ("N rounds") gives N. Any other request (Crow's memory review, a digest, a title) is a side row: recorded and labelled, never counted.
- A row's depth is its rendered prompt, `P` in `[chat] prompt P tok`. A row is in band at 7,000 ≤ P ≤ 10,000.
- A task's decode tok/s = Σ `predicted_n` / Σ `predicted_ms` over its in-band rows. This is Crow's turn-line formula (`cli/crow_core.py:7412-7417`, Crow `d2a7d0b`) restricted to the band. The G5 decode median is the median over the tasks with ≥ 1 in-band row. G5 needs ≥ 5 such tasks; with fewer, the window part of G5 is "not answered" (abort rules), and new tasks come only by a further amendment before their run. Reported beside it, not judged: Crow's turn line over all rounds of each task, and the median over all in-band rows.
- Expected depth, derived and not measured. Flash-Next's first Crow round on 2026-10-06 was 7,259 tokens (`crow-lab/runs/fn-crow-task-win-20261006/serve-8099.log`, Crow 3.2.4). GLM's tokenizer counts `prompt-33k.txt` 3.1 % shorter (30,986 vs 31,979). So a first round near 7.0k, which may fall just under the band; the later rounds carry it. Rows above 10,000 are recorded and not counted.

**Per-row record** (#151 "Proposed fix" 3 and its 2026-10-11 comment). Sources: serve's stderr log of the session and its routing JSON lines.
- Task id, round index, side-row flag, end time (routing JSON `ts`).
- `prompt_n` = P, the whole rendered prompt (the routing JSON's `prompt_n`; `usage.prompt_tokens` on the wire). `cached_n` = C (routing JSON `cached_n` = `usage.prompt_tokens_details.cached_tokens`). Prefilled F = P − C, which is what the wire's `timings.prompt_n` carries (`engine/src/bin/serve.rs:432`). The two `prompt_n` are named apart here so they are never mixed. Also C / P.
- The `[cache]` line verbatim: WARM/COLD, L, held, P, snapshots, `prefill n of m tok`, reset ms.
- Prefill tok/s = F / prefill ms, from the `[chat]` line. On a WARM row this is a small-batch figure (`serve.rs:453-454`) and is never compared with the cold ~32k figure.
- Decode tok/s = `predicted_n` / `predicted_ms` (`[chat]` line, `timings.predicted_per_second`); `predicted_n`, finish, tool calls, malformed.
- Wall clock per row = reset + prefill ms + decode ms from the `[chat]` line. Wall clock per task = Crow's turn line `waited`, with its model/tools split, beside the sum of its row walls.
- m = `cold` / `selections` of the routing JSON line; `bytes_streamed` as logged. RAM/VRAM: `nvidia-smi` memory.used, Windows commit charge and available physical memory at the row's end, from a read-only sampler (period ≤ 5 s, started before t1, the G4 `CommitSampler` form).
- The request as sent: serve's `[chat] sampling on the device` and `thinking` lines (which value came from the request, which from the data sheet). The row states that Crow sends GLM the floor, since GLM has no manifest entry (#151 Evidence).
- Per task, reported with no threshold (the prefix-cache live check): rounds, in-band rows, WARM rows among rounds 2..N, and their mean C / P. Expected: rounds 2..N WARM. A COLD row among them is named with its `[cache]` line.

**Run order: one day, one GPU lock window** (`C:\Users\robin\dev\.gpu-lock`, held from `mkdir` to `rmdir`, as `runs/glm53-flash/quick/bundle.sh`).
1. Take the lock. No other GPU job and no other heavy disk job.
2. One `serve` boot for all five tasks:
   - binary built from crow-nest main at or after `732369c` (`CROW_GLM_KV_LAZY` needs it);
   - `CROW_CNQ` = `converter\GLM-5.3-Flash-MUL1K3.cnq`, `CROW_TOKENIZER` = `models\GLM-5.3-Flash-original\tokenizer.json`;
   - environment `runs/glm53-flash/quick/arm4.env` and nothing else;
   - port 8099, `CROW_CONTEXT` unset (floor 200,000);
   - stderr to `runs/glm53-flash/g5-<date>/serve-tasks.log`.
3. t1 → t5 in the Crow window (robin's steps below).
4. Close Crow and stop `serve` (taskkill while idle, #126; VRAM back under 3 GB).
5. Synthetic row: 3 boots, one after another, each with one cold turn of `crow-lab/runs/2026-08-28-flash-next/prompt-33k.txt`:
   - prompt: sha256 `62141cb4f43e4151c41bfbc0446821c42ec59509ffae2da8c5c47a9bca1bfcde`; 31,979 tokens with Flash-Next's tokenizer, 30,986 with GLM's in G4;
   - setup: the same binary, container and arm4.env;
   - request: G4's (`g4_run.py boots`: temperature 0, `max_tokens` 64, `reasoning_budget_tokens` 0, streamed with timings);
   - recorded per boot: GLM `prompt_n`, prefill tok/s, 64-token decode tok/s, VRAM after the turn, commit peak, m.
   - G5's cold prefill = the median of the 3. The 3-boot spread max/min ≤ 1.15 is read on that prefill figure (lead's decision; the boots measure it). The 64-token decode is reported.
6. Release the lock (`rmdir`).

A serve crash or a void session is a row. The session is then repeated in a new lock window with folders `g5-run<k+1>`; the first one stays in the book.

**robin's steps.**
1. Before you start: the lead has created the five empty folders `C:\Users\robin\Desktop\g5-run1\t1-wordfreq` … `t5-stats`, `serve` answers `/health`, and the sampler runs.
2. In `C:\Users\robin\dev\Crow`, start Crow: `python cli\crow_gui.py --base-url http://127.0.0.1:8099/v1`. Leave the release level at `auto` (the default; nothing asks, so no approval wait enters the wall clock). Leave the model/reasoning chip as Crow offers it for this server, the same for all five tasks.
3. For each task, in order t1 → t5:
   1. Right-click the empty space of the chat rail, choose new project and pick the task's folder, then open a new chat in it. Before you type, check that the composer's working-directory chip shows that folder.
   2. Paste the task text from its code block above and press Enter.
   3. Type nothing else in that chat: no follow-up and no answer to a question. If the turn ends on a question, the task ends there.
   4. Wait for the turn line (`N rounds | … tok @ X tok/s | … | waited …`) and for 30 s of quiet in serve's log, since Crow's memory review can still send a side request. Note the turn line or take a screenshot.
   5. Go to the next task with a new project and a new chat. Never reuse a chat.
4. After t5, close Crow and tell the lead. The lead stops `serve` and runs the 3 synthetic boots, with nobody at the window.

After the session, the lead copies the five session files (`%LOCALAPPDATA%\Crow\session\chat-*.json`). For each task, the lead checks the first user message (trailing whitespace stripped) against its sha256. A task whose text differs is recorded and not counted.

### Amendment 9 — 2026-10-13: the G4 verdict as robin accepted it (crow-nest #149)

Written 2026-10-13 on crow-nest `8e4ee79` (branch `claude/serene-cannon-n57rq2`), after the G4 series and robin's acceptance; it records a verdict and changes no threshold, metric, data set or method. Seen when written: the G4 series (#149 comment of 2026-10-11), robin's acceptance (#149 comment of 2026-10-12, issue closed) and what amendment 8 saw; no G5 verdict (#151) and no G6 row (#152). Reason: the abort rules record every verdict with its number, and G7 judges recorded verdicts; until now the G4 verdict stood only in #149, `docs/measurements.md` and `docs/status.md`.

**Series.** 2026-10-11 08:04–08:14 CEST, robin's GPU go, `.gpu-lock` held. crow-nest local main `298092b` (engine tree = `36b0d9d` v0.11.0), `engine/target/release/serve.exe` built 07:51 from that tree. Container `converter/GLM-5.3-Flash-MUL1K3.cnq`, env `runs/glm53-flash/quick/arm3.env` (33 variables), `serve --port 8099`, `[serve] n_ctx 200000`, tiers per MoE layer VRAM 37 / pinned 118 / NVMe 133. Each boot: one streamed greedy turn of `crow-lab/runs/2026-08-28-flash-next/prompt-33k.txt` (sha256 `62141cb4…bcde`, GLM `prompt_n` 30,986), `max_tokens` 64, thinking closed; commit charge sampled every 0.5 s from serve start to the end of the turn. n = 10. Run dir `runs/glm53-flash/g3g4-20261011/` (local, not in git: `g4-boots.{json,out}`, `boot01..10.log`).

**Per clause (G4 threshold, all must hold):**
- 10/10 boots: **met**, 10/10 with `/health` and a finished turn.
- VRAM < 31.9 GiB after the turn: **met**, 29,472 MiB = 28.78 GiB every boot.
- commit under 103.38 GiB: **met**, peak 91.999–92.102 GiB (system-wide, other processes included).
- engine m inside the step-8 95 % CI [0.4948, 0.5500] (`step08/20261008-g1.md`, N 25, W 0): **not met**, m = 0.3343–0.3344 (21,342–21,350 cold of 63,848 selections), below the CI. The robustness line covers only m above the CI. Stated, not judged (#149): the engine m is over the prompt-33k prefill (30,986 prompt rows + 64 decode rows) with VRAM 37 / pinned 118 and the global arena; the CI is over generated positions of `todo-1006` at N 25.

**Verdict: accepted by robin, 2026-10-12** (#149). By the letter of the threshold G4 is not passed on its m clause; #149 left how that clause counts to robin. His decision, #149 acceptance comment of 2026-10-12, verbatim: "robin's live check on 2026-10-12: accepted (G4 and his Crow sessions on GLM-5.3-Flash)." Its conditions line names the boot, VRAM and commit clauses ("G4 2026-10-11, 10/10 boots at 200k, VRAM 28.78 GiB (< 31.9), commit peak 92.1 of 103.38 GiB") and gives no separate rule for m. Build of that live check: installed crow-nest v0.12.0 serve in robin's Crow (`%LOCALAPPDATA%\Crow\bin`), arm4.env, 200k context. For G7 the verdict is carried as written here, "accepted by robin, m clause not met by the letter (0.3343 below [0.4948, 0.5500])"; it is not rewritten as "passed".

**Administrative, no criterion changed.** G7's consequence line says "the ~180 GB upload". That size is the 4.5-bit container `converter/GLM-5.3-Flash-CNQ4.5.cnq` (178,490,593,097 B, #144), deleted 2026-10-08 ~21:20 and not rebuilt (#144, #179). The container of record is `converter/GLM-5.3-Flash-MUL1K3.cnq`, 124,591,634,567 B (sha256 `9ce11456c68f80926e1fe8289888a3571309a1f8558bc73d1578f0db2b30213e`, #156), the one G4 ran on. The G7 text above is not edited.
