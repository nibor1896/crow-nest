# crow-nest — architecture diagrams

Living renderings of the approved spec (`architecture.md`, sections 1–6). A diagram
contradicting the spec is a bug in the diagram. Owner: issue #14. Updated with every
stage acceptance.

## 1 · System overview — from originals to served tokens

```mermaid
flowchart LR
    subgraph offline["offline · once per model"]
        ORIG["original safetensors\n131 shards · 360 GB BF16"] --> CONV["converter (Rust)\nRTN → NVFP4 · streaming"]
        CONV --> CNQ["CNQ4.5 container\n~101 GB · 4.5 bpw\nsections: text / ple / vit / mtp\nsidecar: per-tensor errors"]
    end
    subgraph runtime["runtime · per session"]
        CNQ --> LOAD["loader\nbudget check · auto-clamp N\ncold-path policy per layer"]
        LOAD --> ENG["engine (Rust + CUDA sm_120a)"]
        ENG -->|"tokens"| API["server\nOpenAI-style chat\n(np=1 · Crow)"]
    end
    HARNESS["parity harness (#6)\n#159 methodology"] -.interleaved A/B.-> ENG
    BASE["llama.cpp baseline"] -.same machine.-> HARNESS
```

## 2 · Decode path — one token, the handoff-free shape

```mermaid
flowchart TB
    EMB["embedding row → residual stream 10240
(PLE host prep: n-gram ids, row cache, slot upload)"] --> L0["layer 0 (GDN + MoE)"]
    L0 --> PLE["PLE at the top of layer 1
gather · key/value proj · gate · dilated conv (state 9)
h += gated + silu(conv) — same place as prefill and reference"]
    PLE --> HC["hyper-connections
per layer: mix → sub-block → inject"]
    HC --> LOOP["layers 1..47 · 36 GDN + 12 attention
 interleaved per layer_types"]
    LOOP -->|each MoE layer| RT["router (BF16)\n2560 × 512 · top-10 + shared"]
    RT --> RES{"all routed experts\nresident?"}
    RES -->|"resident: VRAM pointers"| CMP["GPU computes\nNVFP4 GEMMs (sm_120a)"]
    RES -->|"cold: pinned HOST pointers (UVA zero-copy, ~23 GB/s)"| CMP
    CMP --> COMB["shared_expert_gate combine"] --> HC
    STAGER["stager (control plane only):\nresidency swaps · NVMe tier · telemetry"] -.off critical path.-> CMP
    CMP --> OUT["lm_head (BF16) → logits row"] --> ARG["argmax_k (greedy gate discipline)"]
    OUT -.CROW_SAMPLE=1.-> SMP["host sampler (#20)
temp 0.7 · top_p 0.8 · top_k 20 · presence 1.5
seeded xorshift"] --> TOK["token · CROW_STOP_EOS=1 stops at 248046 / 248044"]
    ARG --> TOK
```

## 3 · VRAM layout at the default operating point (262k ctx, FP8-KV)

```mermaid
pie showData title 32 GiB budget
    "hot experts (N≈160/layer)" : 21.2
    "dense resident (NVFP4 + BF16 keeps)" : 6.0
    "KV cache FP8" : 3.2
    "PLE hot-row cache (128 MB default since 2026-09-05)" : 0.13
    "GDN + QSA state" : 0.3
    "activations + graph pools" : 1.8
    "headroom" : 2.0
```

## 4 · Converter pipeline (stage 1 = RTN, calibration-free)

```mermaid
flowchart LR
    IDX["model.safetensors.index.json\nweight_map"] --> WALK["131-shard walk\nshard in RAM, never the model"]
    WALK --> KEEP{"keep-set?"}
    KEEP -->|"norm · 1-D · embed · router · gates"| BF16["BF16 pass-through"]
    KEEP -->|"I64 metadata (PLE index tables)"| I64["raw carry · dtype i64"]
    KEEP -->|"else · 64-aligned"| Q["RTN → NVFP4\n64/block · 4 ue4m3 · ceiling scales\nglobal f32 per tensor"]
    Q --> SC["sidecar: max/mean err per tensor\ngate 0: violations = 0"]
    BF16 --> OUT["CNQ4.5.cnq\nmagic + streamed blob + index trailer"]
    I64 --> OUT
    SC --> OUT
```

## Changelog

- 2026-09-02: first cut — rendered from approved spec sections 1–6 + decisions (#2). Stage-1 converter reflects the implemented converter; runtime boxes are the spec's design, not yet code.
- 2026-09-02 (later): cold path amended to zero-copy direct read (spec 3.4 amended after the #8 pre-study) — ring/stager moved off the decode critical path to control plane.
- 2026-09-05: decode path redrawn after #11 — the PLE step now sits at the top of layer 1 (it ran before layer 0 in `decode_step` until 2026-09-05 07:00, which is what degenerated the answers); sampler and EOS stop (#20) added as the opt-in tail; PLE cache default 128 MB (#16). Section 3 stays the 2026-09-02 estimate with the PLE slice corrected; the measured load line on 2026-09-05 read "VRAM used 31.21 GiB (dense 7.02 GiB + hot experts 160 × 48 × 2.64 MB + states)".
