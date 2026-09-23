# Improve-Loop crow-nest (kanonische Definition, 2026-09-03)

**Ziel (robin):** maximale tok/s für prefill + decode bei minimalster Latenz und
minimalster System-Auslastung — Qualität bleibt über llama.cpp-Niveau, das
System wird nicht mit Last zugebombt. **Projektmethodik: besser als llama.cpp**
(42 tok/s decode ist Unterkante, nicht Ziel).

## Der Loop

1. **Measure** — `tools/perf_loop.sh <label>`: build → layercheck-Gate (≤ 0,125)
   → 12-Position-Argmax (≥ 7/12) → decode ms/Token → prefill-Rate (2100er
   Prompt). Eine Zeile in `decode_out/perf_history.tsv`.
2. **Review** — Gates grün? Wenn nein: Änderung zurückrollen, kein weiterer
   Schritt auf kaputter Basis.
3. **Research (zwei Beine, IMMER beide):**
   - intern: Breakdown, welcher Kernel/Pfad jetzt die meiste Zeit frisst;
   - extern: **SOTA-Check gegen die Regeln unten + Web-Recherche nur für den
     konkreten Schritt** (llama.cpp / vLLM / TRT-LLM / FlashInfer / Colfax /
     DeepGEMM — wie lösen die genau dieses Problem?).
4. **Improve** — den nächsten Hebel umsetzen, hinter env-Flag (variabel
   gehalten: alte Pfade bleiben als Fallback), Engine-Finetuning (Tile-/Chunk-
   /Warp-Parameter) solange Gates grün.
5. zurück zu 1. **Loop-Ende:** eine ganze Runde ohne messbaren Gewinn (< 5 %)
   = Decke erreicht → finale Zahl an robin.

## SOTA-Regeln (feste Checkliste vor JEDER Verbesserung — aus
docs/sota-research-2026-09-03.md, mit Quellen dort)

- [ ] **R1 Tensor-Cores:** Der Schritt trägt vom `mma.sync m16n8k64`-MMA
  (tcgen05 existiert auf sm_120a nicht — m16n8k64 block-scaled ist das
  Maximum der Hardware). Kein neuer Pfad als Skalar-Loop.
- [ ] **R2 Launch-Zahl:** Wird die Launch-Count reduziert (5–20 µs/Launch auf
  WDDM; auf Linux nicht nachgemessen, `docs/measurement-handoff.md` schuldet
  den Retest)? Zielrichtung: CUDA-Graphs (capture-once + Parameter-Patching) statt
  nur je-Op-Fusion; statische Buffer halten (graph-kompatible Formen).
- [ ] **R3 Quant-Fusion:** Aktivierungs-Quantisierung fusioniert in den
  Vorgänger-Op (Norm/Activation gibt FP4 + Blockskalen direkt aus;
  `QuantizedActivation`-Reuse statt separater Quant-Launch + HBM-Roundtrip).
- [ ] **R4 Prefill als GEMM:** Prefill läuft als tiled GEMM-Pipeline (TMA,
  Multi-Stage-Buffering, 8 MMA-Warps, 128er-Tiles — Colfax ~60 % FP4-Peak
  als Referenz), nie als GEMV-Loop über Token.
- [ ] **R5 MoE-Dispatch:** batch-1 → Gruppierung + Vektorpfad (top-10-GEMV);
  batched → moe_align_block_size (sortieren + BLOCK_M padden). NIE dense-
  override bei 512 Experten (0,18× bei batch-1, TRT-LLM gemessen).
- [ ] **R6 Prefetch:** n-gram-/PLE-Lookups des nächsten Chunks werden
  asynchron vor dem Chunk vorbereitet (Prefetch-Pipeline statt synchroner
  Row-Fills).
- [ ] **R7 Mess-Ehrlichkeit:** jede Zahl nennt Container, CROW_MMA, Sidecar-
  Herkunft, Chunk-Größe, Budgets; llama-/SOTA-Zahlen werden nie mit Crow-
  Zahlen in eine Tabelle relationalisiert (nur als separater Kontext).
- [ ] **R8 Qualitätsgates:** layercheck ≤ 0,125 · Argmax ≥ 7/12 · Traces ohne
  neue Degeneration · Rel-zahlen gegen Ceiling-Referenz ausgewiesen. Rot =
  Stufe zurück, nie weiter.

## Abarbeitungs-Reihenfolge (Stand 2026-09-03)

1. Dense-GEMV-MMA (~16 Callsites) — 🔄 läuft
2. Launch-Fusion + CUDA-Graphs-PoC (R2/R3 kombinieren) — Warteschlange
3. Prefill-GEMM-Pipeline nach Colfax (R4) + Chunk/Sync/PLE-Batching — Queue
4. MMA-Headroom (ld.128, 8 Warps, Experten-Gruppierung im Prefill) — Queue
5. Sparse-Prefill (t7-Pfad: Indexer + Selektion über 13k Kontext) — Queue
6. Sidecar-Re-Warm-up mit echtem Crow-Traffic + async n-gram-Prefetch (R6) — Queue

## Der Linux-Gate (Ergänzung 2026-09-17, issue #15)

The loop above is the performance loop and is unchanged. What changed on 2026-09-17 is where
step 2 (**Review — Gates grün?**) gets its answer on Linux: `tools/gate-linux.sh [outdir]`,
from the repository root, runs the parity forms 8 / 512 / P8 teacher-forced, the short
generated-id run, `cargo test --release`, clippy and the doc guards against the Linux
values of record, prints GREEN or RED per item and exits non-zero on any RED. Nine items;
all nine green at commit `8ff2055` on 2026-09-17.

- Scope: it is an identity gate, not a performance gate. It answers "do the bytes still match",
  which is the precondition of step 2; it measures no tok/s and replaces no `perf_loop.sh` run.
- The values the script checks are `bceba6ff7724…` (8 rows, identical to the Windows reference),
  `8387234709271515…` (512 rows) and `3bb3e69edf90…` (P8 teacher-forced), plus the 32 ids of
  record. They are hard-coded with their provenance; a value there moves only when a new
  reference run establishes a new record, and the commit that moves it says so. R8 applies
  unchanged: RED = eine Stufe zurück.
- The host-side counts the script enforces are `TESTS=221` (128 lib + 82 serve + 6 parity +
  5 decode at `8bad310`, v0.3.1, 2026-09-18; every step of that day's rise from 165 carries its
  issue and its reason in the script header) and `CLIPPY=1421` (the
  `--all-targets` form, counted as
  `grep -cE '^warning: '`; 1422 until `#13` on 2026-09-18 turned the one
  `redundant reference in eprintln! argument` site into a `tracing` event). The guard loop names
  three scripts — `check_env_docs` (exit 0, `code 89, doc 89` since the `CROW_GDN_SPLIT_Z` row of #71, 2026-09-18),
  `check_readme_dates` (0 offenders) and `check_model_card_dates` (0 offenders, in this tree and
  running since 2026-09-18).
- The 1024-row form is a Linux value of record too (`117dd8d9d8dc…`, established 2026-09-17) but
  is NOT in the script: it costs a full long-prompt run. Run it by hand before a change that
  touches the chunk regimes (`docs/architecture.md` 8.7).
- The Linux and Windows values differ on the 512-row and 1024-row forms because the NVRTC and
  driver JIT differ — measured, documented, and not a lever (`docs/architecture.md` 8.7).
- Machine and method for every Linux number: the second environment block of
  `docs/system-landscape.md`, one engine at a time, inside the memory-bounded scope of
  `tools/serve-linux.sh` (the RAM gate refuses a second engine while the first holds the tier).

## The quantization acceptance rule (Ergänzung 2026-09-20, issue #91)

A HARD RULE for every container or precision change, written down because #79 earned it: **no
container/precision change is accepted on weight-space evidence.** Not plain MSE, not
importance-weighted MSE, not any error term — #79 measured three requant rules that all
LOWERED the weight-space error (best −19.74 % importance-weighted) and all moved the model
FURTHER from its f32 oracle (+0.025…+0.104, sign tests to 6.5e-22;
`docs/expert-requant.md` §7.5/§8: "the instrument that would have been used to tune such a
rule, weight-space MSE, points the wrong way on this architecture").

Acceptance = oracle-KLD on the #90 corpora (short + long-context + German) against the
`none` baseline, beyond the 0.025 paired threshold. Weight-space numbers may be REPORTED for
the record, never used to accept, reject, tune or select. Phase-1 artifacts of #91:
`docs/acceptance/issue-91-phase1.md`.

**Lesson of 2026-09-23 (#91): the per-layer reference diff comes before precision arms.**

- **What happened.** The dense BF16 overlay improved the symptom metric. On the multi-site
  corruption probe it went from 15 to 12 of 23 corrupt sites, and it was briefly made the
  serve default (`0924406`, reverted `0254ed6`).
- **What it was hiding.** The cause was an engine wiring bug: the PLE n-gram rows were read
  at the wrong container offset (`85a48e7`). The overlay was partly masking that bug. With the
  fix the probe reads 4/23, the same as llama.cpp.
- **What found the bug.** A per-layer diff against a reference engine
  (`tools/layerdiff/`). Layer 0 matched, and layer 1 split at cos 0.2. None of the precision
  arms found it.
- **The rule.** When a symptom points at quantization, first diff the engine per layer against
  a reference engine at a failing site. Run precision or overlay arms only after the first
  divergent layer is explained. An arm that moves a symptom metric is not evidence about the
  cause. Acceptance of any precision change stays paired oracle-KLD, as above.
