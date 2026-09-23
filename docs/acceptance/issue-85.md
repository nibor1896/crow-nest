# Acceptance protocol — issue #85 (DRY sampler)

- Date: 2026-09-21. Implemented by fleet subagent (completed posthumously by the orchestrator after the OOM incident), verified by the orchestrator.
- Spec: `gh issue view 85` — llama.cpp DRY (PR #9702, KoboldCpp #982), host-first route (issue option a).

## What was built

- `Sampler` grows `dry_multiplier` (0.0 NEUTRAL), `dry_base` (1.75, forced ≥ 1.0 — llama-server's own fix), `dry_allowed_length` (2), `dry_last_n` (64, 0 disables). Host-only path per the issue; state rides the #84 window machinery.
- The penalty: reverse Z-algorithm over the window finds, per candidate, the longest suffix repeat length L; `logit -= multiplier · base^(L − allowed)`, exponent clamped at `88.7228391/ln(base)` (llama.cpp's float-max guard). Sequence breakers `{"\n", ":", "\"", "*"}` bound the scan and exempt the token that begins one — llama.cpp defaults.
- serve: `dry_multiplier/base/allowed_length/last_n` request fields (strict types, 400 named), carried on the conditional tier line only when non-neutral.
- Env rows: `CROW_DRY_*` in `docs/env.md` (read only when `CROW_SAMPLE=1`).

## Spec-adherence checklist

| issue requirement | status |
|---|---|
| exact llama.cpp math incl. exponent clamp | ✓ (test `dry_exponent_clamps_at_float_max_log_over_ln_base`) |
| breakers bound scan + own-token exemption | ✓ (`dry_sequence_breakers_bound_the_scan_and_exempt_their_own_token`) |
| armed only with multiplier>0 ∧ base≥1 ∧ window>0 | ✓ |
| penalizes the EXTENDING token, not flat repeats | ✓ (`dry_penalizes_the_token_that_would_extend_a_repeat`, naive-reference match `dry_matches_the_naive_llama_reference_over_random_windows`) |
| disabled = byte-identical goldens | ✓ (`dry_and_tier_defaults_are_neutral_and_goldens_stay_byte_identical`) |
| host-first, device port deferred by measurement | ✓ (issue option a) |

## robin's live-acceptance cases

| # | check | how | expected |
|---|---|---|---|
| 1 | field honored | chat request with `"dry_multiplier": 0.8, "dry_last_n": 64` | tier line names DRY armed with the values |
| 2 | neutral default | same request without DRY fields | answers byte-identical to pre-#85 build |
| 3 | the #68 shape | long session/replay with DRY armed | phrase-echo suppressed vs unarmed replay |
| 4 | type errors | `"dry_base": "x"` | 400 naming dry_base |

## Incident note (honest)

This unit's fleet run died in the 2026-09-21 OOM incident (see issue-96's note and the stopstr cut() tombstone); the orchestrator completed the serve parse wiring (`int_field`, tier gating, mirostat/xtc parse rules), fixed the test fixtures, and verified the suite (lib 193/0, serve 101/0 at acceptance).
