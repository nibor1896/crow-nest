# SOTA-Research: FP4-Inferenz auf sm_120a / WDDM (2026-09-03, SubAgent-Recherche, alle Aussagen mit Quellen im Original-Report)

## TL;DR — die 3 größten tok/s-Hebel für unser Setup

1. **CUDA Graphs** (capture-once + Parameter-Patching, statische Buffers, graph-kompatibles masked-MoE-Layout à la DeepGEMM): auf WDDM der größte Einzelfaktor für Batch-1-Decode-Latenz. WDDM-Launches kosten 5–20 µs (Linux ~3–5 µs); bei ~15–30 Launches/Layer sind wir bei 75–600 µs reinem CPU-Overhead pro Layer. llama.cpp: +5–18 % auf Linux gemessen, default für batch-1, Strategie = `cudaGraphExecUpdate` + Parameter-Patching, NIE re-capturen. **Keine veröffentlichten WDDM-Graph-Benchmarks existieren — wir müssen selbst messen.** HAGS: A/B-Test, kann konkurrierende Kernel-Streams reduzieren (konsistent mit unserer p9-HGS-Probe: negativ).
2. **Prefill-MMA-Pipeline nach der Colfax-Rezeptur** (RTX Pro Blackwell sm12x): TMA für A/B/SFA/SFB in EINER Pipeline, 8 MMA-Warps, Multi-Stage-Double-Buffering, CTA-Tile 128×128×128, CLC/PDL/Warpgroup-Register-Re-Allocation → ~60 % des FP4-Peaks dokumentiert. **tcgen05 existiert auf sm_120a NICHT** (nur sm_100a/101a/103a) — unser `mma.sync m16n8k64` ist die korrekte hardware-native Wahl, unser E2M1+ue4m3-Format ist exakt hardware-nativ (SFA 16×4 / SFB 4×8, Hardware-Replikation fixiert).
3. **Aktivierungsquant in den Vorgänger-Op fusionieren** (Norm/Activation gibt direkt FP4 + Blockskalen aus; `QuantizedActivation`-Reuse-Muster aus vLLM/CUTLASS; globale Scales offline kalibrieren, online nur 16er-Blockskalen): spart pro GEMM einen Launch + einen HBM-Roundtrip — auf WDDM doppelt wert.

## MoE-Tricks (bestätigt + widerlegt)

- Batch-1 decode: unsere top-10-Gruppierung + Vektor-GEMV ist korrekt (TRT-LLM „MoE as dense GEMM" ist bei batch-1 0,18× — Vermeidung bestätigt; Sweet Spot dort 64–208 Tokens). Für batched Prefill: `moe_align_block_size`-Pfad (sortieren + auf BLOCK_M padden, vLLM/SGLang).
- llama.cpp MMVQ bei kleinen Batches, CrossOver runtime-tunbar — deckt sich mit unserer MMA/naiv-Umschaltung (CROW_MMA).
- TRT-LLM hält separate Low-Latency-MoE-Kernels (grouped GEMM + fused finalize/SwiGLU) vom Throughput-Pfad getrennt.

## Prefill/Decode-Trennung

- llama.cpp dispatcht per Batchgröße (≤32 → MMVQ, sonst GEMM-Pfade), CrossOver tunbar.
- DeepGEMM: contiguous grouped GEMM (prefill) vs masked grouped GEMM (decode, graph-kompatibel, statische Formen) — der sauberste Weg, unseren MoE-Pfad in CUDA Graphs zu bekommen.
- ROCm-Split-K als Decode-Latenzmuster (K-Slices, Bandbreite ausschöpfen).

## Embedding-Tabellen (102 GB PLE)

- Unsere 3-Tier-Struktur (GPU-Cache → pinned RAM → NVMe) = Industriestandard (HugeCTR HPS, NVIDIA Merlin).
- Konkreter Verbesserungspunkt: **n-gram-Lookups des nächsten Prefill-Chunks asynchron vorziehen** (Prefetch-Pipeline, Meta/Kakao-Referenz).
- `cuco::static_map` / HierarchicalKV als GPU-Hash-Index mit Cache-Semantik für variable Row-Indizes.

## Nicht dokumentiert (keine Raterei)

- WDDM-CUDA-Graph-Performancezahlen (eigenes Benchmark nötig — erster Kandidat nach dem Dense-Umbau).
- tcgen05-vs-mma.sync auf sm_120: nicht vergleichbar, tcgen05 existiert dort nicht.
- llama.cpp Fusionszahl Ops-per-MoE-Layer.

## Eingearbeitet in die Warteschlange

- **NEU in der Queue: CUDA-Graphs** (nach dem Dense-MMA-Umbau; braucht statische Buffer + masked-MoE-Layout → hängt mit der Launch-Fusion zusammen, kombiniert umsetzen).
- ③ Prefill-Kette erweitert: Colfax-Pipeline-Elemente (TMA, 8 Warps, Multi-Stage) statt nur Chunk-Größen-Tuning.
- ⑥ Sidecar/PLE ergänzt: asynchroner n-gram-Prefetch des nächsten Chunks.
