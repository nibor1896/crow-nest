# crow-nest — expert coverage curve (issue #3, 2026-09-02)

Source: Crow's own routing logs — 15 `print_locality` blocks from 15 server runs in
`dev/crow-lab/runs/` (ladder, locality, parity, tasks series, all 2026-09-01, model
UD-Q2_K_XL quant, 48 expert layers × 512 experts, 10 routed + 1 shared). Read-only;
analysis script: `tools/coverage-curve.py`.

## Measured (per block: fraction of experts covering 50/80/95 % of routed selections)

| statistic | 50 % covered by | 80 % covered by | 95 % covered by | Gini |
|---|---|---|---|---|
| min | 5.2 % (27 experts) | 16.1 % (82) | 31.9 % (163) | 0.761 |
| median | 7.3 % (37) | 20.0 % (100) | 36.4 % (186) | ~0.773 |
| max | 7.8 % (40) | 21.1 % (108) | 40.0 % (205) | 0.811 |

Task-type spread is real and matters: reasoning/writing tasks are more concentrated
(95 % by 163–174 experts), debugging more diffuse (95 % by 205). The ladder runs over
generic traffic cluster tightly (95 % by 183–196).

## Curve

Least-squares through the three median points: **c(N) = −0.5162 + 0.2823 · ln(N)**
(capped 0.995). This is an interpolation of measured points, not a model.

| N/layer | VRAM (GB) | coverage c(N) | P(all 10 resident) = c¹⁰ | cold bytes/token | serial H2D* | fits budget |
|---|---|---|---|---|---|---|
| 32 | 4.2 | 46.2 % | 0.0 % | 713 MB | 28.5 ms | yes |
| 64 | 8.5 | 65.8 % | 1.5 % | 453 MB | 18.1 ms | yes |
| 96 | 12.7 | 77.2 % | 7.5 % | 302 MB | 12.1 ms | yes |
| 128 | 17.0 | 85.3 % | 20.5 % | 194 MB | 7.8 ms | yes |
| **160** | **21.2** | **91.6 %** | **41.7 %** | **111 MB** | **4.4 ms** | **yes** |
| 176 | 23.3 | 94.3 % | 55.8 % | 75 MB | 3.0 ms | yes |
| 192 | 25.4 | 96.8 % | 72.1 % | 43 MB | 1.7 ms | no |
| 256 | 33.9 | 99.5 % | 95.1 % | 7 MB | 0.3 ms | no |

\* bandwidth input (~25 GB/s pinned H2D), serialized — not a throughput prediction.
Budget: 23.5 GB for experts = 32 GiB minus dense 4.5 GB, KV FP8 ~3.0 GiB, GDN/QSA
~0.25, PLE rows ~1.0, activation/graph pools ~2.0 → max 177 experts/layer.
**Correction 2026-09-02 (after section 1 approval):** the approved BF16 keep-set
(embeddings, lm_head, router, gates, norms stay BF16) raises the dense VRAM footprint
by ~1.8 GB — expert budget ~20.9–22.5 GB depending on pool sizing → hard ceiling
~158–169 experts/layer. Default 160 stands; the loader auto-clamps N to the measured
fit at load time (section 2 of the spec).

## Recommendation to the spec (memory layout section)

- Hot-set default **160 experts/layer** (91.6 % coverage, 21.2 GB, ~4 GB total VRAM
  headroom left), **configuration ceiling 176** (94.3 %, 23.3 GB). 192+ does not fit
  next to the 262k/FP8-KV decision.
- Expected per token at N=160: ~39 of 48 layers issue at least one cold job
  (independence estimate, c¹⁰); the streaming path (cold path A) carries them —
  ~111 MB/token, serialized ≈ 4.4 ms, overlappable by design.
- The per-layer policy field (#7) decides per layer by bandwidth; the engine's own
  telemetry refines this curve in operation — the numbers here are design inputs with
  named assumptions, not promises.

## Assumptions (binding for anyone quoting these numbers)

1. Coverage measured through the UD-Q2_K_XL quant on Crow traffic, aggregated over 48
   layers; per-layer top-N residency typically achieves ≥ this global curve.
2. P(all 10 resident) = c¹⁰ assumes independence; correlation within a layer lowers it,
   per-layer selection raises coverage. Refine from engine telemetry.
3. The logs carry per-request aggregates only (no per-token per-layer selections) — a
   per-token dump would be a new measurement, not needed for the hot-set default.
4. The original strand-1 quote (Gini 0.778, 50 % on 7.3 %) sits inside this range —
   consistent.
