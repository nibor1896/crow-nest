# issue #88 — KV cache precision A/B (e4m3-LUT vs bf16 vs non-LUT)

Instrument: `tools/oracle-kld.py` against `decode_out/oracle-t2b/` (ref 611 rows,
analysis rows 0:607, prompt-rows 596). Arms collected with
`engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json <out>`
from the repo root, every engine run through `flock /tmp/crow-gpu.lock`.
Paired threshold of the instrument (docs/oracle-kld.md §6): 0.025 mean KLD / 1.5 pp top-1.

Machine state at start: commit e27f005, decode binary 2026-09-20 08:28 (fresher than
every lib source; only serve.rs/ctx_reset_probe.rs — other agent's, not in this binary —
are newer). GPU 903 MiB in use, no engine processes.

Standard form (A matrix) — the collection form of the arms of record (docs/oracle-kld.md §5):
prefill all 607 ids (rows 0..606) + 4 greedy decode steps (rows 607..610, excluded from analysis).
CROW_GRAPH=1 CROW_MMA=1, cgroup MemoryHigh=56G/MemoryMax=58G per docs/oracle-kld.md §8.

## A1 — default (fp8 KV + LUT decode), baseline re-measure

    flock /tmp/crow-gpu.lock -c "systemd-run --user --scope --slice=session.slice --quiet \
        -p MemorySwapMax=0 -p MemoryHigh=56G -p MemoryMax=58G \
        env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
            CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json CROW_GRAPH=1 CROW_MMA=1 \
            LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
        engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json \
            decode_out/kv-ab/a1-none" > decode_out/kv-ab/a1-none.log 2>&1

env: CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json
     CROW_GRAPH=1 CROW_MMA=1  (CROW_KV unset = fp8_e4m3, CROW_ATTN_LUT unset = attn_sel_split_l)

## A2 — fp8 KV + LUT off (non-LUT decode path)

same as A1 plus CROW_ATTN_LUT=0  (decode attention kernel attn_sel_split, the pre-61f form)

## operational note (between-run recovery)

A2's first two attempts refused config at manager.rs:203: after A1's exit the
~46 GiB pinned tier sits in the driver's LAZY pool (#15; HANDOFF-2026-09-20-pm
'Ballon-Beweis': the pool IS returned under pressure, not a leak) and MemAvailable
reads ~8 GiB. Because a chromium process holds a CUDA context, the loader counts
free_for_pin AS MemAvailable (manager.rs derive_host_pinned_budget, #15 follow-up).
Recovery between runs: `python3 decode_out/kv-ab/balloon.py` (anonymous balloon,
stop at 3 GiB MemAvailable) — reproduced the handoff: 48 GiB balloon, pool returned,
MemAvailable 50.6 GiB. Applied before every subsequent arm; state noted per run.

## A3 — bf16 KV (CROW_KV=bf16), LUT on

first attempt refused: VRAM free at start 22.95 GiB (record day: 23.22) — a live
chromium GPU process costs ~270 MB, and with the bf16 KV's +2344 MB the two-sided
clamp finds no N (VRAM wants N<=141, host cold tier at 46 GiB wants N>=142).
Same form as the doc §8 overlay arms: CROW_PINNED_BUDGET_GB=48 frees the host side
(memavail 51 GiB; 48+3 margin passes the pre-pin gate). Placement note: kvbf16 of
record ran N=142; A3 may clamp lower — hot/cold placement is expected numerics-
neutral, TESTED by byte-compare A3 vs the kvbf16 arm of record.

## A4 — bf16 KV + LUT off (CROW_KV=bf16 CROW_ATTN_LUT=0 CROW_PINNED_BUDGET_GB=48)

## A-matrix byte-parity results (2026-09-20)

| dump | sha256 (first 16) | N | note |
|---|---|---|---|
| a1-none            | 79a4d2b682b4db9e | 158 | BYTE-IDENTICAL to none of record (79a4d2b682b4) — baseline reproduces across days |
| a2-lut0            | 79a4d2b682b4db9e | 158 | BYTE-IDENTICAL to A1 — CROW_ATTN_LUT=0 == default on ALL 611 rows (incl. the 4 decode rows) |
| a3-kvbf16          | 04ba4ee86ed4e609 | 138 | BYTE-IDENTICAL to kvbf16 of record (04ba4ee86ed4, N=142) — hot/cold placement is numerics-neutral |
| a4-kvbf16-lut0     | 04ba4ee86ed4e609 | 139 | BYTE-IDENTICAL to A3 — LUT parity holds on the bf16 path too |

Consequences: on the standard form the A matrix is exactly the #78 kvbf16 pair
of record ({fp8 79a4d2b6, bf16 04ba4ee8}); the LUT lever is BYTE-IDENTICAL by
measurement, not only by construction. The KLD side of A is the #78 table; it is
re-run below for this protocol. The LUT question is only measurable on rows that
run the DECODE path, so the D matrix follows (CROW_GRAPH=0 CROW_PARITY_PREFILL=8
— the p8tf gate form; rows 0..7 prefill, 8..606 teacher-forced decode steps,
607..610 greedy; analysis rows 0:607 as before).

## D1 — decode-path baseline (CROW_GRAPH=0 CROW_PARITY_PREFILL=8, fp8 KV, LUT on)

## D2 — decode-path, LUT off (+ CROW_ATTN_LUT=0)

(recovery note: the pool needed a deeper sustained balloon this time — 54 GiB before
it released; balloon.py's quick stop-at-3 form was not enough. memavail 52.9 GiB.)

## D3 — decode-path, bf16 KV (+ CROW_KV=bf16 CROW_PINNED_BUDGET_GB=48)

## D4 — decode-path, bf16 KV + LUT off (+ CROW_KV=bf16 CROW_ATTN_LUT=0 CROW_PINNED_BUDGET_GB=48)

## D-matrix byte-parity results

| dump | sha256 (first 16) | N | note |
|---|---|---|---|
| d1-tf-none         | 542850166419454b | 160 | decode-path baseline (603 decode rows: 8..610) |
| d2-tf-lut0         | 542850166419454b | 160 | BYTE-IDENTICAL to D1 — the LUT lever is bit-identical ON THE DECODE PATH |
| d3-tf-kvbf16       | 87ccfd3de2ee5ee3 | 143 | bf16 decode path (differs from D1, as it must) |
| d4-tf-kvbf16-lut0  | 87ccfd3de2ee5ee3 | 143 | BYTE-IDENTICAL to D3 — LUT parity on the bf16 decode path |

## KLD analysis (tools/oracle-kld.py; rows 0:607, prompt-rows 596; no GPU)

## KLD results (rows 0:607) — tables in kld-a.json / kld-d.json, full text in this log's runs

A matrix (standard form):  a1 = a2 (byte-identical) 0.460689;  a3 = a4 0.472301;
  paired a3-a1 = +0.011612 +- 0.014840 (287/607, sign 0.194) — reproduces the #78
  kvbf16 row exactly. Inside the 0.025 floor, wrong direction.
D matrix (decode path):    d1 = d2 0.476149;  d3 = d4 0.465485;
  paired d3-d1 = -0.010664 +- 0.018694 (240/607 worse, sign 2.86e-07) — right
  direction, INSIDE the 0.025 floor (the sign test on 607 correlated rows is the
  instrument's own §6 caveat). LUT pairs are 0.000000 exactly.
  context: d1 - a1 (decode-path vs prefill-path form) = +0.015460 +- 0.011911.

## L matrix — layercheck3 (the layer-3 attn sub-block vs the f32 golden; cwd engine/)

The issue's expected-result names the 2.9 % rel_L2 layer-3 residual (with every
weight BF16, docs/dense-overlay.md 4.1). The KV attribution probe:
  L1 plain FP4 + fp8 KV | L2 plain FP4 + bf16 KV
  L3 originals overlay + fp8 KV | L4 originals overlay + bf16 KV
overlay runs: CROW_CNQ_OVERLAY=converter/dense-bf16-originals.cnq (absolute),
CROW_PINNED_BUDGET_GB=49 (doc §8 uses 50; today's post-balloon memavail 52.9 GiB
makes 49 the value that passes the pre-pin gate; N placement proven neutral).

## L1 — layercheck3, plain FP4, default fp8 KV (the p16 mark: rel_L2 0.165, max_abs 0.582)

L1 result: rel_L2=0.1282 max_abs=0.4473 corr=0.99179 (stepwise identical) — the
plain-FP4 mark today, better than the p16 mark (0.165/0.582) recorded before later fixes.

## L2 — layercheck3, plain FP4, CROW_KV=bf16

L2 result: rel_L2=0.1282 max_abs=0.4473 corr=0.99179 — IDENTICAL to L1 at print
precision: at plain FP4 the weight error dominates; the KV dtype moves nothing.

## L3 — layercheck3, originals overlay (all dense weights BF16), fp8 KV
## L4 — layercheck3, originals overlay, CROW_KV=bf16

(recovery note 2: the pool sometimes refuses shallow passes; the reliable form is
a deep pass that HOLDS ~30 s at ~1 GiB MemAvailable — sustained pressure, then it
releases. balloon.py carries this form now. 2026-09-20 late: recovered to 54.1 GiB.)

L3 result (overlay originals, fp8 KV): rel_L2=0.0290 max_abs=0.1092 corr=0.99958 —
reproduces the 2.9 % residual of record (docs/expert-requant.md §8 / dense-overlay 4.1).
L4 result (overlay originals, bf16 KV): rel_L2=0.0290 max_abs=0.1092 corr=0.99958 —
IDENTICAL at print precision. The layer-3 residual is NOT the KV store/decode path.
L1/L2 likewise identical at plain FP4 (0.1282).

## VERDICT (Phase 1)

No arm clears the 0.025 paired-KLD / 1.5 pp top-1 threshold:
  LUT off vs on: 0.000000 exactly (byte-identical dumps, both forms, both KV dtypes)
  bf16 KV vs fp8: +0.0116 ± 0.0148 (standard form, wrong direction)
                  -0.0107 ± 0.0187 (decode path, right direction, under floor)
  layer-3 golden: identical at print precision in all four L runs.
The KV path is measured innocent on every instrument this form offers.
Phase 2 (per-block e4m3 scales / bf16-KV planner option) NOT implemented — the
issue's own condition ("If fp8 moves KLD") is not met. The 100k-200k sparse
regime remains unreachable by this instrument (needs #T8) — the honest limit.
