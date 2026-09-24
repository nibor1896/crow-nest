# Hot-set recalibration 2026-09-24 — criteria fixed before any new set exists

Written 2026-09-24 ~08:20 CEST, before the new sidecar was cut and before the three
calibration runs finished. Seen so far: only the BASELINE on the held-out file (below).

## Data
- Calibration: the three 2026-09-23 diorama rollover archives, rendered by
  `tools/session_ids.py` (174,149 + 171,748 + 167,867 tokens).
- Held-out: `session.json` of the same run (121,955 tokens). Never used for cutting.
  Caveat stated up front: same task, same day, same model — the held-out score is
  optimistic for other workloads.

## Baseline (already measured, prefill routing, N 160)
| sidecar | held-out hit |
|---|---|
| uniform 160/512 | 0.312 |
| hotsets-M-longctx2100-n160 (current default) | 0.416 |
| hotsets-M-tentasks-n160 | 0.496 |
| in-sample oracle on the held-out itself | 0.714 (ceiling, not attainable) |

## Primary metric
Held-out prefill hit rate = routed choices landing in the set / all routed choices,
at N 160 and at the engine's live N, from `routestats` counts (`tools/hotset-from-counts.py score`).

## Secondary metrics (one GPU run each per sidecar, same held-out prompt)
1. Decode-route hit rate over 256 decode tokens after prefilling the held-out prompt
   (`decode routestats <held-out> <out> 256`, decode routes scored offline).
2. Decode tok/s: `decode run` on the held-out prompt, 64 tokens, 3 repetitions per sidecar,
   same binary, same env, engines strictly one at a time.

## Acceptance (all must hold, else the default stays and the result is reported as is)
- A. Primary: new ≥ 0.600 at N 160 (≥ +0.18 over the current default, ≥ +0.10 over tentasks).
- B. Secondary 1: new decode-route hit rate > current default's by ≥ 0.05.
- C. Secondary 2: new mean decode tok/s > current default's mean, and new min > current max
  (ranges do not overlap). If they overlap: "no measurable decode gain", default not changed
  on tok/s grounds alone.

## Method choice
Default method: summed prefill selection counts over the three archives, top-N per layer,
frequency order (the `decode warmup` rule). Alternatives from the SOTA research (e.g.
decode-weighted counts, per-layer N) may be evaluated ONLY on the same held-out with the
same metrics; the pick is the best primary score among those that also pass B and C.
Every variant tried is reported, not only the winner.

---

# AMENDMENT 1 — 2026-09-24 ~08:35 CEST, still before any new set exists

Cause: the SOTA research (scratchpad research-hotset.md) finds prefill routing near-flat and
decode routing skewed (ProMoE; DuoServe-MoE §2.2; llama.cpp RFC #24528: "prompt routing is far
flatter"; arXiv 2604.09780 §5.2). Crow's live workload is decode-bound, so the PRIMARY metric
above measured the wrong phase. No new set had been cut or scored when this was written; the
only numbers seen are the baseline table above (prefill hit, old sidecars).

New instrument: `CROW_ROUTE_DUMP_PREFILL` (engine, env-gated) dumps every prefill position's
routed ids; `tools/session_ids.py` writes the GENERATED positions of each file (assistant
reasoning, text, tool calls = what decode produced; causal model, so their routing is decode's).
Generated positions: 100,263 / 112,721 / 100,571 (archives), 77,271 (held-out session.json).

## Primary metric (replaces the one above)
Generated-position hit rate on the held-out session.json at N 160 and at the engine's live N,
with a 95 % block-bootstrap CI (1,000-token blocks, 2,000 resamples).

## Acceptance (replaces A and B; C unchanged)
- A'. new − current default ≥ +0.10 on the primary metric, and the bootstrap 95 % CI of that
  difference has a lower bound > +0.05.
- B'. Leave-one-out robustness: cut on any three of the four files, score on the fourth; the new
  method beats the current default in 4 of 4 folds.
- C. unchanged (decode tok/s, 3 runs, ranges do not overlap).
Prefill hit rate (the old primary) is reported, no threshold. Hit rate by context depth
(0–32k, 32–100k, 100k+) is reported, no threshold.

## Variants (all reported; pick = best primary among those passing A', B', C)
- P: prefill counts, all positions (the `decode warmup` rule).
- G: generated-position counts only.
- G+λP: generated counts + λ · prefill counts, λ ∈ {0.1, 0.3}.
- G-budget: G with the 48 × 160 slots distributed across layers greedily by marginal held-in
  gain, per-layer floor 128, cap 200 (only if the engine accepts ragged per-layer N — to be
  checked; otherwise reported as not applicable).
