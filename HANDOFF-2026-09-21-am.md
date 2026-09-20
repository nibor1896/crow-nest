# HANDOFF 2026-09-21 (morning) — the quality fleet's first night

Written by the orchestrator at ~01:00, robin asleep. Everything local, nothing pushed.

## What happened (short)

The 2026-09-20 quality re-triage (issues #83–#96, created from the llama.cpp gap analysis) was executed as a six-unit fleet overnight. **Wave 1 is complete: all six units accepted, verified and committed locally** — commits `d8b66a0` (#87 grading), `08ae3c8` (#89 numerics diff), `5662fea` (#94 metadata gate), `192c099` (#83 min_p + #84 penalties), `22a274f` (#88 KV A/B), `74d6a3e` (#90 long-context instrument), plus this gate/handoff commit.

| unit | headline result |
|---|---|
| #87 phase 1 | Linux record graded blind (seal held); 16k deficit is thinking budget (38/60 empty), crow draws level at 32k (share_met 0.606) |
| #89 | engine numerics ACQUITTED: 17 MATCH / 0 MISMATCH / 3 UNVERIFIABLE vs HF + llama.cpp; #28068 class not present; 2 probes PASS on the 5090 |
| #94 phase 1 | metadata gate: 20 constants asserted vs config.json at the front door, zero numeric change, 13 tests |
| #83+#84 | min_p + llama.cpp windowed penalties, host/device bit-twins, 265/0 tests, goldens byte-identical, defaults neutral |
| #88 | KV A/B: no arm clears the 0.025 floor; LUT decode bit-identical incl. the decode path; layer-3 residual NOT the KV path |
| #90 | long-context instrument: verified 178k ids (5 probe lengths exact vs the #68 gate record), 6-anchor plan incl. sparse-QSA boundary 2564, resumable f32 oracle (self-test 2.1e-07), oracle-kld sparse+curves (55+9 tests), English Gutenberg corpus; paired-baseline is the standing mode |

Acceptance protocols with live test cases: `docs/acceptance/issue-{87,88,89,90,94}.md` (+ `issue-83.md`, `issue-84.md`). CHANGELOG has Measured/Added entries per unit; README has the record row + docs row.

## The machine state (READ THIS FIRST)

**The nvidia_uvm pinned pool is in the HARD-leak state (#82 class): ~40 GiB held with no owner, balloon cannot reclaim (3 rounds to 9.8 GiB MemAvailable), only a reboot reclaims it.** Cause: the night's engine cycles (kv-ab matrix + longctx arm retries + three gate runs). Consequences and the checklist:

1. **Reboot the machine.** After it:
2. `bash tools/gate-linux.sh` — expect **ALL GREEN**. Tonight's evidence: parity8 GREEN at FULL tier with the exact sha of record `bceba6ff7724` (the byte-identity proof through every wave-1 engine change); cargo test 265/0 GREEN; clippy 1480 GREEN (counters bumped 234→265 / 1459→1480, justified in the gate header — same pre-existing lint classes, no new class); all three doc guards GREEN. parity512 / p8tf / run32 could not run full-tier after the pool died mid-gate — the gate now pool-recovers between GPU items (`pool_recover`, the kv-ab balloon, timeout-capped).
3. Relaunch the #90 engine arm: `nohup tools/oracle_longctx_engine_arm.sh > decode_out/oracle-longctx/engine-arm.log 2>&1 & echo $! > /tmp/fleet-monitor/90-oracle.pids` (idempotent; anchors 1000 + 2564; its RAM ladder did 0/4 runs tonight purely because of the squeeze/leak).
4. Optional, for the llama arm of #90 and the #87 phase-2 record: the llama GGUF **shard 1 is broken/incomplete** (10 MB on disk) — re-download needed (robin's call, ~25 GB).

## Open threads (next waves, nothing running)

- Wave 2 candidates: #85 DRY, #86 stop strings + logit_bias, #92 optional samplers, #96 YaRN, the #89 leftover probes P1b/P2/P3, the F1 radix fix (own issue).
- Wave 3: #93 grammar, #91 quant layer rules (now the PRIME suspect for the layer-3 residual after the #89+#88 acquittals), #95 MTP speculative, #87 phase 2 (fresh record at the real row with the new sampler surface — needs the reboot + llama shard).
- The #90 anchor depth >2564 needs a `decode parity --rows` write mode (documented in `docs/oracle-longctx.md`).
- The 30-min fleet monitor cron (`/tmp/fleet-monitor/`) can be deleted once robin is awake: all agents DONE.

## Live-acceptance short list for robin (per unit, full versions in docs/acceptance/)

- #83: chat request with `min_p 0.01` → provenance line honors it; disabled → byte-identical to a pre-#83 build; `CROW_SAMPLE_HOST=1` A/B parity (after reboot).
- #84: `repeat_penalty 1.1 / frequency_penalty 0.1 / penalty_last_n 64` → window-armed line; neutral defaults → old goldens.
- #94: boot shows `meta: 20 constants verified`; doctored `CROW_MODEL_DIR` panics by named table.
- #87/#88/#89/#90: see the docs — one-command re-runs each.

No push happened. `main` is 7 commits ahead of origin + robin's own #82 commits below them.
