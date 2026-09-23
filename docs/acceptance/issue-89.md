# Acceptance protocol — issue #89 (GDN/QSA numerics diff vs HF reference and llama.cpp)

- Date: 2026-09-20 (night). Executed by: fleet subagent (diff + probes), orchestrator (verification below).
- Issue spec required: line-by-line three-way diff table, probe plan, no existing engine code changed, honest verdicts. All delivered; issue stays OPEN for probes P1b/P2/P3 (planned).

## What was delivered

| artifact | what it is |
|---|---|
| `docs/numerics-diff.md` | 24-row table: crow-nest (formula + file:line) \| HF transformers 5.16.1 (`models/qwen4_exp/modeling_qwen4_exp.py`, from the oracle venv) \| llama.cpp (`qwen4exp.cpp`, `qwen3next.cpp`, `models.h`) \| verdict; + UNVERIFIABLE section, probe plan, 17-row config-pin cross-check, findings, conclusion |
| `engine/src/bin/qsa_tie_probe.rs` | probe P1 — QSA tie-break determinism (planted bit-identical tie groups straddling rank 512) |
| `engine/src/bin/attn_path_probe.rs` | probe P4 — prefill-vs-decode QSA attention row equality at n=2051 |
| `engine/Cargo.toml` | two `[[bin]]` registrations (only these lines staged; the concurrent `ctx_reset_probe` registration belongs to robin's #82 work and stays unstaged) |

## Verdict

**17 MATCH / 0 MISMATCH / 3 UNVERIFIABLE.** The engine-numerics suspect (expert-requant.md §8) is acquitted on every line compared. Highlights:

- The llama.cpp #28068 bug class (GDN norm max instead of rsqrt) is NOT present — `l2norm_repeat` (kernels.rs:1589-1602) is exactly HF's rsqrt form; llama.cpp's post-fix form is algebraically identical.
- Attention scale 0.0625 applied pre-max-subtraction in all three engines (positive scale commutes — MATCH).
- QSA pipeline order identical three-way (pool raw → RMSNorm(128) → RoPE at block start; whole blocks + unconditional tail).
- crow ≥ HF precision at every compared site (HF does elementwise bf16, crow stays f32).
- Probes: tie-break deterministic lowest-index on all three selectors (7/7); prefill-vs-decode divergence ≤ 1.1e-6 rel_L2 — cannot explain the 2.9 % layer-3 residual.

Open (issue stays open): U1/P1b torch-side tie order, P2 GDN recurrent-vs-chunked prefill floor (the one remaining formula-level risk), P3 RoPE ULP. New defect F1 filed as its own issue (`qsa_select` radix K>ncb, env-gated).

## Orchestrator verification (all checks passed)

1. **Deliverable existence + structure**: 24 table rows with verdict column; tallies match (17/0/3).
2. **Change discipline**: `git status` — exactly 3 new files + 2 `[[bin]]` Cargo.toml blocks; no existing engine source touched (verified via `git diff` inspection).
3. **Probe runs corroborated**: `engine/target/debug/{qsa_tie_probe,attn_path_probe}` binaries built 2026-09-20 23:23 (mtime), matching the reported run window; both probes' pass criteria documented with inputs, grids and commands in the doc.
4. **Cargo.toml partial staging**: `git apply --cached` of an 8-line patch — robin's #82 `ctx_reset_probe` registration NOT staged with this unit.
5. **Source quotes spot-checked**: rows 2, 8, 14, 16 formula quotes match the file:line sites named.
6. **Honesty**: unverifiable items carry reasons + severity; F1 disclosed rather than silently fixed; reduction-order differences declared as bit-noise, not matches.

## Test cases for robin's live acceptance

| # | what to check | how (from `engine/`) | expected |
|---|---|---|---|
| 1 | probe P1 — tie-break determinism | `LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib cargo run --bin qsa_tie_probe` | PASS line, 7/7 cases, all three selectors byte-identical to reference |
| 2 | probe P4 — attn path equality | `LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib cargo run --bin attn_path_probe` | PASS, rel_L2 ≤ ~1.1e-6 across split factors 1/2/8 |
| 3 | the verdict table | open `docs/numerics-diff.md` §1 | 24 rows, verdict column, 17 MATCH / 0 MISMATCH / 3 UNVERIFIABLE in §1 tally |
| 4 | spot-check one formula against the code | e.g. row 2: `sed -n '1589,1602p' engine/src/kernels.rs` vs the quoted rsqrt form | identical |
| 5 | the config pin table | `docs/numerics-diff.md` §4 | all 17 rows agree with `models/Qwen3.8-Flash-Next-original/config.json` |

## Consequences for the fleet

- The layer-3 residual attribution shifts to weight quantization (FP4/BF16 keeps) unless P2 measures a real GDN-prefill floor — feeds #88/#91 prioritization.
- U1 (sparse-regime tie-break) feeds #90's long-context instrument design (the dense-regime oracle rows cannot see it).
- F1 got its own issue; fix goes through the env-gated → bit-parity promotion path.
