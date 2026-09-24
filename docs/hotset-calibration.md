# Hot-set calibration

How the hot-set manifest `serve` loads is cut and how a new one is accepted.
Issue #106. Criteria of the 2026-09-24 recalibration were written before the first result:
`decode_out/hotset-0924/PREREG.md` (sha256 `6d8561ae…` after amendment 1, 08:22 CEST).

## Procedure

| step | command (repository root) | output |
|---|---|---|
| 1. corpus | `.venv-oracle/bin/python tools/session_ids.py <crow-session.json> <name>-ids.json <name>-mask.json` | the ids `serve` prefilled (the model's own chat template, reasoning and tool calls kept) and the GENERATED spans (assistant reasoning, text, tool calls) |
| 2. routing | `ROUTE_DUMP=1 tools/hotset-calibrate.sh <dir> <name>-ids.json...` | per file: `decode routestats <ids> <out> 0` with `CROW_ROUTE_DUMP_PREFILL` → `<name>-routes.bin` (every position's routed ids) and `<name>-counts.json` (the engine's own counters) |
| 3. score + cut | `.venv-oracle/bin/python tools/hotset-eval.py [--write <sidecar> <variant>] <dir> <mask-dir> <held-out> <cal>...` | self-test (dump == counters), held-out hit rates with bootstrap CI, leave-one-out, depth buckets, the sidecar |
| 4. speed | `tools/hotset-speed.sh <dir> <held-out-ids.json> <old-sidecar> <new-sidecar>` | `decode run <ids> 64`, 3 runs per sidecar, alternating |

- One engine at a time; every run in the `systemd-run` scope of `tools/gate-linux.sh`.
- The generated positions stand in for decode: a causal model routes a token the same way whether it is
  generated or read back. Prefill routing over all positions is flatter than decode routing
  (ProMoE; llama.cpp RFC #24528), so the cut and the primary metric use generated positions only.
- `CROW_ROUTE_DUMP_PREFILL` syncs the device once per chunk-layer; measured cost 0 (525 tok/s prefill with and without, 2026-09-24).
- Per-layer budgets do not apply: `residency::sidecar_sets` brings every row to the planner's one N (#49).

## 2026-09-24 result — `hotsets-M-crow0924-n160.json`

Corpus: robin's three 2026-09-23 diorama rollover archives (174,149 / 171,748 / 167,867 tokens;
100,263 / 112,721 / 100,571 generated). Held-out: that run's `session.json` (121,955 tokens, 77,271 generated).
Engine: crow-nest main 7641a0b (PLE fix 85a48e7), -M container, RTX 5090, Linux.

| sidecar | held-out hit, generated positions, N 160 | all positions | diff vs current, 95 % CI (block bootstrap, 1,000 tokens, 2,000 resamples) |
|---|---|---|---|
| `longctx2100` (previous default) | 0.401 | 0.416 | — |
| `tentasks` | 0.426 | 0.496 | +0.025 [+0.014, +0.038] |
| P (prefill counts, all positions) | 0.698 | 0.685 | +0.297 [+0.285, +0.310] |
| **G (generated positions) — chosen** | **0.723** | 0.681 | **+0.322 [+0.310, +0.335]** |
| G + 0.1·P | 0.722 | 0.682 | +0.321 |
| G + 0.3·P | 0.720 | 0.684 | +0.319 |
| held-out's own top-160 (ceiling) | 0.791 | 0.683 | +0.390 |

Leave-one-out (cut on three files, score the fourth), generated positions, N 160:

| held out | previous | P | G | G+0.3P |
|---|---|---|---|---|
| session | 0.401 | 0.698 | 0.723 | 0.720 |
| rollover-195925 | 0.349 | 0.654 | 0.621 | 0.632 |
| rollover-210418 | 0.383 | 0.705 | 0.713 | 0.713 |
| rollover-225111 | 0.392 | 0.724 | 0.741 | 0.740 |

Decode, `decode run <held-out> 64`, context 122,019, planner N 148, 3 runs each (alternating):

| sidecar | tok/s | mean ms/token | cold experts/token |
|---|---|---|---|
| previous | 32.5 / 32.4 / 32.5 | 30.79 / 30.86 / 30.76 | 217.7 |
| new | 35.6 / 35.8 / 35.8 | 28.06 / 27.92 / 27.92 | 180.5 |

All three criteria of the PREREG hold (A' +0.322, CI low +0.310; B' 4 of 4 folds; C ranges do not overlap).

## Limits

- One task (a WebGL diorama goal run), one day, one model: the held-out score is optimistic for other work.
- The decode gain (+10 %) is far below what the hit-rate gain suggests: the live N is 148, and the
  engine's own per-prompt adaptation already recovers part of a poor manifest.
- The gates of record keep `hotsets-M-longctx2100-n160.json` (`tools/gate-linux.sh` pins it), because their
  greedy ids depend on the manifest.
- `hf-package/` still ships the previous manifest; the Hugging Face upload is a separate step.
