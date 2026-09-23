# Acceptance protocol — issue #92 (optional sampler tier: top-nσ, XTC, typical_p, mirostat v2)

- Date: 2026-09-21. Implemented by fleet subagent (completed posthumously by the orchestrator), verified by the orchestrator.
- Spec: `gh issue view 92` — the llama.cpp default-chain members crow lacked, all OFF by default, chain positions per the issue.

## What was built

- `Sampler` grows `top_n_sigma` (0.0 NEUTRAL), `typical_p` (1.0), `xtc_probability` (0.0), `xtc_threshold` (0.1, >0.5 refused at parse — llama.cpp disables XTC there), `mirostat` (u8: 0 off, 2 = v2, 1 refused with a named 400 — v1 unimplemented; v2 with `temperature ≤ 0` refused, its surprise distribution IS the temperature softmax), `mirostat_tau` (5.0), `mirostat_eta` (0.1).
- top-nσ: mean/std of the candidate logits, mask below `max − n·std` — no softmax, no sort (the paper's point).
- typical_p: entropy-target |−log p − H| inclusion until mass, per arXiv:2202.00666.
- XTC: with `xtc_probability` per token, removes the leading run above the threshold (min_keep honored); carries its OWN RNG stream — the main xorshift64* draw sequence is unperturbed (llama.cpp's independence rule, pinned by test).
- mirostat v2: truncates candidates with surprise `−log2(p) > mu`, renormalizes, samples, updates `mu −= eta·(surprise − tau)`; mu is per-request state.
- serve: all fields parsed (strict types), `TierSent` provenance flags, the conditional tier line; env rows `CROW_TOP_N_SIGMA/TYPICAL_P/XTC_PROB/XTC_THR/MIROSTAT/MIROSTAT_TAU/MIROSTAT_ETA`.

## Spec-adherence checklist

| issue requirement | status |
|---|---|
| all four OFF by default, goldens byte-identical | ✓ (`dry_and_tier_defaults_are_neutral_and_goldens_stay_byte_identical` covers the tier) |
| top-nσ no-softmax form | ✓ |
| XTC independent RNG stream | ✓ |
| mirostat v2 mu update | ✓ |
| chain positions per issue | ✓ |
| serve fields + provenance | ✓ (tier line prints only when non-neutral) |

## robin's live-acceptance cases

| # | check | how | expected |
|---|---|---|---|
| 1 | tier honored | request with `"top_n_sigma": 2.0` or `"mirostat": 2, "mirostat_tau": 5` | tier line names the armed members |
| 2 | XTC refusal rule | `"xtc_threshold": 0.7` | 400 naming the rule |
| 3 | mirostat v1 refused | `"mirostat": 1` | 400 naming v1 unimplemented |
| 4 | neutral defaults | request without tier fields | byte-identical to pre-#92 build |
| 5 | reproducibility with XTC on | same seed twice with `xtc_probability: 0.3` | identical streams (XTC RNG independent, not perturbing the draw order) |

## Incident note

Same as issue-85.md — orchestrator completed the serve wiring and fixture fixes posthumously; suite green at acceptance (lib 193/0, serve 101/0).
