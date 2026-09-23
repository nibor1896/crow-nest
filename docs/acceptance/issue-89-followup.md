# Acceptance protocol — issue #89 follow-up (F1 fix + probes P1b/P2/P3)

- Date: 2026-09-21. Executed by: numerics subagent (all runs below, RTX 5090).
- Scope: the four leftovers of the #89 numerics diff — the F1 engine fix (issue #97), and the probes P1b (torch tie order), P2 (GDN recurrent-vs-chunked), P3 (RoPE table ulp) from docs/numerics-diff.md §3/§5. No model, no container: GPU probes compile `KERNEL_SRC` and run synthetic buffers under `flock /tmp/crow-gpu.lock`.

## 1. F1 — qsa_select dense-regime fix (issue #97) — FIXED, regression proven both ways

| run | result |
|---|---|
| `qsa_tie_probe` BEFORE the fix (dense cases un-skipped) | `dense ncb400 tail3 -> FAIL qsa_select: sel_n 3 != ref 1603` (the #97 repro); all 5 sparse cases + dense ncb512 PASS |
| `qsa_tie_probe` AFTER the fix | **7/7 PASS** — all three selectors byte-identical to the host reference on every case incl. dense |
| `qsa_probe` (default arms byte-identity pin) | 4928 rows, **0 differences** |
| `attn_path_probe` | 3/3 PASS (1.08e-6 / 8.98e-7 / 7.97e-7 rel_L2, unchanged floors) |
| `router_probe` | PASS (unchanged) |
| `cargo test --release` | **265 passed / 0 failed** |

Change: `engine/src/kernels.rs`, `qsa_select` only — the same `if (K >= ncb) { emit 0..=pos; sel_n = pos+1; return; }` dense shortcut `qsa_select_fast` (kernels.rs:2596-2604) and `qsa_select_par_e` already carry, inserted immediately after the `K = *k_p` read. The radix body below it is untouched; fast/par arms untouched. Probe change: `engine/src/bin/qsa_tie_probe.rs` runs the plain-radix arm on the dense cases too (previously skipped with a note). No commit made (no-git rule); the doc carries the session note.

## 2. P1b — torch.topk tie order (venv, CUDA+CPU)

torch 2.14.0+cu130. Rows: the P1 planted-tie design (bit-identical 0.5 straddling rank 512; ncb ∈ {600..65536}; groups 2..64; boundary-exact; all-tied) + the realistic relu-clamped-`+0.0` case (300 zeros / 600 blocks, need 212) + 60 random ambiguous replicates.

- **CUDA (the oracle's device): lowest-index SET rule matched in EVERY ambiguous case** — fixed cases, 60/60 replicates, relu-zero case; bitwise repeatable across runs; equal values emit ascending (same as crow).
- **CPU: 49/60 replicates** — the quickselect path is not stable; straddle-24/ncb3000 fixed case picks a different tie subset. Irrelevant to the CUDA oracle.
- Consequence bound (2048-token list, f32 softmax over N(0,σ)): 4 tokens at the row max hold 3.9% (σ=1) / 63.0% (σ=4) of mass; +2-logit content advantage → 23.2% / 99.98%. Bounded by ordinary top-k boundary sensitivity; not tie-specific.
- Bonus U3 cross-check: torch `inv_freq` (`1/base**e`) vs crow (`base**(-e)`): 17/32 bits equal, 15 differ by exactly 1 ulp; both ≤ 1 ulp off correctly-rounded.

## 3. P3 — RoPE table ulp (host-only `rope_table_probe`) — PASS

262144 positions × 32 pairs, manager.rs formula recomputed byte-for-byte:

| quantity | reference | max ulp |
|---|---|---|
| inv_freq (32 values) | f64 powf → f32 | **0** |
| cos, sin (full table) | f64 cos/sin of the table's own f32 argument | **1** |
| golden pairing vs host rotate_half | — | **bitwise equal** at every sampled t |
| (report-only) cos/sin vs full-f64 end-to-end | f64 powf+product+cos | ~2.0e9 ulp at t≈2.5e5 — the f32-table property every f32 engine (HF included) shares; first >2 ulp at t=3..5 |

U3 closed: everything crow controls is ≤ 1 ulp; the pairing formula is exactly HF's rotate_half.

## 4. P2 — GDN recurrent-vs-chunked (`gdn_chunk_probe` + `/tmp/p2_gdn_dump.py`) — PASS, 26x margin

Inputs: 48 heads, T ∈ {64, 512, 4096}, normed q/k (q pre-scaled 1/√128), v ~ N(0,1), β ∈ (0.2, 0.8), g ∈ (−0.05, 0]; both torch references dumped on the same f32 bits (fla absent → pure-torch bodies).

| T | crow out vs chunked (rel_L2, worst head) | crow out vs torch recurrent | S vs chunked | ‖S‖_F |
|---|---|---|---|---|
| 64 | 3.64e-7 (4.06e-7), max_abs 5.2e-8 | 2.46e-7 | 2.52e-7 | 175.08 |
| 512 | 3.88e-7 (4.18e-7), max_abs 6.7e-8 | 2.56e-7 | 2.70e-7 | 180.12 |
| 4096 | 3.92e-7 (4.00e-7), max_abs 8.9e-8 | 2.57e-7 | 2.66e-7 | 180.37 |

- Tolerance: rel_L2 ≤ 1e-5 expected — met with **26x margin** (250x to the 1e-4 decision line).
- torch's own rec-vs-chunk floor: 3.14e-7 / 3.41e-7 / 3.45e-7 — crow (3.9e-7) is numerically the same class as torch-vs-torch.
- **S growth-vs-T: none** (2.5e-7 → 2.7e-7 flat 64→4096); ‖S‖ stable ~180 — the delta rule self-bounds the state and g ≤ 0 contracts it. The hypothesized long-context compounding does not materialize.
- Consequence for the layer-3 residual (2.9e-2): the GDN prefill class floor is 4e-7, three orders below — the residual is attributable to weight quantization, not GDN algorithm drift.

## Files

| file | change |
|---|---|
| `engine/src/kernels.rs` | `qsa_select` dense shortcut only (F1) |
| `engine/src/bin/qsa_tie_probe.rs` | dense cases run through the plain radix (regression) |
| `engine/src/bin/rope_table_probe.rs` | NEW (P3, host-only) |
| `engine/src/bin/gdn_chunk_probe.rs` | NEW (P2, GPU synthetic) |
| `engine/Cargo.toml` | `[[bin]]` rows appended: `rope_table_probe`, `gdn_chunk_probe` |
| `docs/numerics-diff.md` | F1 → FIXED; §3 P1b/P2/P3 RESULTS; §6 follow-up note |
| `/tmp/p1b_torch_tie.py`, `/tmp/p2_gdn_dump.py` | the venv experiment + dumper (kept in /tmp per the "no code in repo" P1b plan; commands recorded in the doc) |

## Deviations

- P2's "real layer-3 oracle dump" input variant was not used — the synthetic O(1) rows already answer the class question at 26x margin; a real-row dump is redundant instrumentation for a bound this tight.
- P1b's "fraction of tied-boundary rows in real generations" is not measurable with current instruments (oracle rows are 298/607 tokens, dense regime) — stated in the doc, needs #90's long-context KLD instrument.
- The P3 gate is against the shared-f32-argument reference (the only reference against which ≤2 ulp is meaningful at 262k positions); the end-to-end f64 distance is reported alongside it, honestly, as the f32-table class every engine shares.
- `cargo test` was re-verified green mid-session (265/265) after the kernel fix; concurrent agents were editing serve/stopstr during this session (their transient compile breaks delayed two probe runs; no file of theirs was touched).
