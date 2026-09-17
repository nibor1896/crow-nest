# crow-nest architecture diagrams

Living renderings of the approved spec (`architecture.md`, sections 1 to 8). A diagram
contradicting the spec is a bug in the diagram. Owner: issue #14. Updated with every
stage acceptance. Diagrams 1 to 6 are current as of 2026-09-12, commit 885bb27 (the #61b
docs commit), re-read against the tree on 2026-09-17; diagram 7 was added on 2026-09-17 in
commit c1a68cd and re-verified at 487128d, branch `main`.

## 1 · System overview, from originals to served tokens

```mermaid
flowchart LR
    subgraph offline["offline · once per model"]
        ORIG["original safetensors\n131 shards · 360 GB BF16"] --> CONV["converter (Rust)\nRTN → NVFP4 · streaming"]
        CONV --> CNQ["CNQ4.5 container\n~101 GB · 4.5 bpw\nsections: text / ple / vit / mtp\nsidecar: per-tensor errors"]
    end
    subgraph runtime["runtime · per session"]
        CNQ --> LOAD["loader\nbudget check · auto-clamp N\ncold-path policy per layer"]
        LOAD --> ENG["engine (Rust + CUDA sm_120a)\none CUDA graph replay per decode token"]
        ENG -->|"tokens"| API["serve (Rust)\nOpenAI-style chat, np = 1\nGET /health · GET /props · GET /slots\nPOST /v1/chat/completions · POST /slots/0\nprefix cache A9 · slot file A10"]
    end
    HARNESS["parity harness (#6)\n#159 methodology"] -.interleaved A/B.-> ENG
    BASE["llama.cpp baseline"] -.same machine.-> HARNESS
```

Status: reflects the server as built in stage A (#24 to #32) on top of the engine defaults
of 2026-09-12, commit 885bb27.

## 2 · Decode path at the current defaults

```mermaid
flowchart TB
    GRAPH["one CUDA graph replay per decode_step\nscalar refreshes · table flip · ev_commit before the launch\none boot line each names the staging kernel, the trickle order, the selection"]
    TRICKLE["stream trickle, deferred since #63c\ncopies parked in trickle_tick are issued right after the launch\nand overlap the replay (trickle_drain_after_launch)\nCROW_TRICKLE_DEFER=0 restores the eager issue before the launch"]
    GRAPH --> EMB["embedding row → residual stream 10240\n(PLE host prep: n-gram ids, row cache, slot upload)"]
    TRICKLE -.overlaps the replay.-> GRAPH
    EMB --> L0["layer 0 (GDN + MoE)"]
    L0 --> PLE["PLE at the top of layer 1\ngather · key/value proj · gate · dilated conv (state 9)\nh += gated + silu(conv), same place as prefill and reference"]
    PLE --> HC["hyper-connections\nper layer: mix → sub-block → inject"]
    HC --> LOOP["layers 1..47 · 36 GDN + 12 attention\ninterleaved per layer_types"]
    LOOP -->|each MoE layer| RT["router_top10 (BF16)\n2560 × 512 · top-10 + shared expert"]
    RT --> STG["stage_cold_ca, default since #19e, inside the graph\npersistent grid, CROW_STAGE_BLOCKS=40 × 256 threads\npulls every cold combo of the layer into its VRAM staging slot\ncp.async 4 KB tiles · rewrites the combo pointer tables\nCROW_STAGE_KERNEL=1 falls back to stage_cold"]
    STG --> GEMM["expert GEMMs read VRAM pointers only\nhot slab slot or staging slot · NVFP4 GEMMs (sm_120a)"]
    GEMM --> COMB["shared_expert_gate combine"] --> HC
    LOOP -->|each attention layer| QSC["qsa_scores_par over the QSA ring and pooled cache"]
    QSC --> SEL{"QSA top-k, same list in the same order either way"}
    SEL -->|"default, CROW_QSA_PAR unset (#61b)"| PARS["qsa_select_par_h, CROW_QSA_PAR_BLOCKS=32 blocks, 12-bit histogram\nthen qsa_select_par_e, one block, 1024 threads\nthreshold refine · ascending emit"]
    SEL -->|"fallback, CROW_QSA_PAR=0"| FAST["qsa_select_fast on one block\nthe prefill selection keeps this form"]
    PARS --> ATT["attn_sel_split over the selected list\ngrid NQ × 1 × CROW_ATTN_SPLITS=8\nthe splits knob 4/8/16/32 is measurement only (#61a)"]
    FAST --> ATT
    ATT --> MERG["attn_merge reads the device scalar n_splits"] --> HC
    STAGER["stager (control plane only):\nresidency swaps · NVMe tier · telemetry"] -.off the critical path.-> GEMM
    LOOP --> OUT["lm_head (BF16) → logits row"] --> ARG["argmax_k (greedy gate discipline)"]
    OUT -.CROW_SAMPLE=1.-> SMP["host sampler (#20)\ntemp 0.7 · top_p 0.8 · top_k 20 · presence 1.5\nseeded xorshift"] --> TOK["token · CROW_STOP_EOS=1 stops at 248046 / 248044"]
    ARG --> TOK
```

Status: reflects commit 885bb27, 2026-09-12, the three defaults of the week: stage_cold_ca
(#19e), the deferred trickle (#63c), qsa_select_par with the CROW_QSA_PAR=0 fallback and
attn_sel_split at 8 splits (#61a and #61b, the split count is measurement only). Measured
operating point: 23.94 ms per token = 41.8 tok/s against 24.87 with the fallback
(`decode_out/srv-61b.log`, architecture 3.4).

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

Status: unchanged from the section 2.1 estimate of 2026-09-02 (PLE slice corrected
2026-09-05 per #16); no stage since moved a slice, checked 2026-09-12. The measured load
line of 2026-09-05 read VRAM used 31.21 GiB.

## 4 · Residency: hot set VRAM, pinned cold tier, zero-copy read

```mermaid
flowchart TB
    NVME["NVMe tier · the CNQ4.5 container"]
    STAGER["stager (control plane only)\nresidency swaps · NVMe tier reads · telemetry\noff the decode critical path"]
    NVME --> STAGER
    subgraph host["host RAM · 64 GiB"]
        COLD["cold tier, PINNED: 43 to 47 GiB\nexpert slabs as UVA pinned host pointers"]
    end
    subgraph vram["VRAM · 32 GiB budget"]
        HOT["hot set: about 160 expert slabs per layer · 21.2 GiB"]
        STAGE["staging slots: one per cold combo of this step"]
    end
    GEMV["routed expert GEMVs read VRAM pointers only:\nhot slab slot or this step's staging slot"]
    STAGER -->|"tier transitions and refresh"| COLD
    STAGER -.residency swaps.-> HOT
    COLD -->|"stage_cold_ca inside the graph (#19e)\ncp.async 4 KB tiles, measured 47.78 GB/s\nagainst 32.45 GB/s for stage_cold (#19d)"| STAGE
    COLD -.->|"CROW_STAGE=0: direct zero-copy\ninside the GEMM, ~23 GB/s"| GEMV
    STAGE --> GEMV
    HOT --> GEMV
```

Status: reflects architecture 3.4 as amended (zero-copy direct read is the cold-path
primary) with the staging default of #19e (commit 7afb4f8), 2026-09-12; staging numbers
from `decode_out/srv-19d.log` (RTX 5090, 338 MB per token).

## 5 · Serve: endpoints, prefix cache (A9), slot save and restore (A10)

```mermaid
flowchart TB
    CLI["Crow CLI (crow_core.py)\nand any OpenAI-style client"]
    SRV["serve on 127.0.0.1:8099, blocking, one request at a time\nGET /health · GET /props · GET /slots\nPOST /v1/chat/completions: SSE stream, or one document\nwith stream false (#39 B3a) · POST /slots/0 action save or restore"]
    CLI --> SRV
    SRV --> LCP{"longest common id prefix L\nof the request ids vs the held history\nids only, never text (#31 A9)"}
    LCP --> P["reuse point P: the largest snapshot position at or below L\nwhose rows are all PREFILL CLEAN\nnothing is erased: pos moves back to P,\nrows at or above P are stale but unreachable"]
    P --> ROLL["rollback: GDN state, conv state and QSA ring restored\nfrom the snapshot at P; P = 0 is the cold start,\none 16k prefill of 21.6 to 22.0 s"]
    ROLL --> PF["prefill P to prompt_len\nrewrites the re-rendered previous answer as prefill rows"]
    PF --> S2["snapshot at the prompt end (prefill clean)"]
    S2 --> DEC["decode_step loop, the diagram above"]
    DEC --> WIRE["streamed chunks, usage and timings on the final chunk\ncached_tokens = P, always present"]
    SRV --> SLOT["slot file, across processes (#32 A10)\nKV rows 0..pos + pooled blocks 0..floor(pos/4)\nmeasured 352,843,384 B at 16k\nneeds --slot-save-path, else both actions answer 400\ncontract: n_saved = n_restored"]
    SLOT -.->|"restore into a fresh process"| ROLL
```

Status: reflects architecture section 7 as built (A9 #31, A10 #32, the dropped
after-answer snapshot M2b #36, the stream false document #39 B3a), 2026-09-12; the warm
turn at 16k reused 99.41 percent of the prompt in 404 ms (README measured table, #31).

## 6 · Converter pipeline (stage 1 = RTN, calibration-free)

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

Status: unchanged since the first cut of 2026-09-02 (#2); no converter stage has landed
since, verified 2026-09-12.

## 7 · Module dependency graph of the engine crate (2026-09-17)

```mermaid
graph LR
  subgraph L0[leaves]; cuda[cuda.rs]; cnq[cnq.rs]; geo[geo.rs]; tokenizer[tokenizer.rs]; toolcall[toolcall.rs]; end
  subgraph L1[on the leaves]; kernels[kernels.rs: kernel table + launch_v + kprof]; manager[manager.rs]; sample[sample.rs]; weights[weights.rs: tensor loaders + Fp4]; boot[boot.rs]; end
  residency[residency.rs]; vit[vit.rs]; gen[gen.rs]; cache[cache.rs]; reset[reset.rs]; slot[slot.rs]
  kernels --> cuda; manager --> cuda & geo; sample --> geo; weights --> cnq & cuda; boot --> cnq & cuda & geo
  residency --> cnq & cuda & geo & kernels & manager; vit --> cnq & cuda & geo & kernels & weights
  gen --> cnq & cuda & geo & kernels & manager & residency & sample & vit & weights
  cache --> cuda & gen & geo; reset --> cuda & gen & geo; slot --> cache & cuda & gen & geo
```

Status: the `use crate::` edges of `engine/src` at commit 487128d, 2026-09-17, regenerated from
the tree; acyclic since bb9d2ca broke `gen <-> residency` and `gen <-> vit`. The six engine
commits after c1a68cd (f8f75c0, 1032bc5, 4004e66, e2b9845, 8ff2055, 487128d) changed `serve.rs`,
`cnq.rs`, `cuda.rs`, `gen.rs`, `geo.rs`, `toolcall.rs`, `vit.rs` and `cache.rs` without moving a
single module edge. Not drawn: the
feature-gated `cutile_pilot.rs` (`cuda`, `kernels`; nothing calls it), and the one edge no import
graph shows — `impl Drop for Engine` in `gen.rs` calls `Engine::drop_decode_graph`, an inherent
method defined in `reset.rs`. Module by module: `architecture.md` section 8.

## Changelog

- 2026-09-02: first cut — rendered from approved spec sections 1–6 + decisions (#2). Stage-1 converter reflects the implemented converter; runtime boxes are the spec's design, not yet code.
- 2026-09-02 (later): cold path amended to zero-copy direct read (spec 3.4 amended after the #8 pre-study) — ring/stager moved off the decode critical path to control plane.
- 2026-09-05: decode path redrawn after #11 — the PLE step now sits at the top of layer 1 (it ran before layer 0 in `decode_step` until 2026-09-05 07:00, which is what degenerated the answers); sampler and EOS stop (#20) added as the opt-in tail; PLE cache default 128 MB (#16). Section 3 stays the 2026-09-02 estimate with the PLE slice corrected; the measured load line on 2026-09-05 read "VRAM used 31.21 GiB (dense 7.02 GiB + hot experts 160 × 48 × 2.64 MB + states)".
- 2026-09-12: post #61b pass. Decode path redrawn for the three default flips: the staging kernel `stage_cold_ca` (#19e, engine commit e256004), the deferred trickle (#63c, engine commit 095a1c8) and the parallel QSA selection `qsa_select_par` with the `CROW_QSA_PAR=0` fallback plus the `CROW_ATTN_SPLITS` measurement knob (61a and 61b, engine commit 9696b13). The resident-or-cold decision at the GEMMs is gone: the staging kernel hands the GEMMs VRAM pointers in every case, so zero-copy direct read survives only behind `CROW_STAGE=0`. New section 4, the residency picture (hot set VRAM, pinned cold tier, zero-copy read), and new section 5, the serve picture (endpoints, prefix cache A9, slot save and restore A10). The system overview server box now names the endpoints. Pie and converter unchanged; every diagram carries a status line.
- 2026-09-17: new section 7, the module dependency graph of the engine crate, after the three refactor cuts of branch `linux-refactor` (74c79f2, bb9d2ca, 7ddd296). It is the first diagram in this file that renders the CODE rather than the spec, and it is generated from the `use crate::` edges, so a module move that is not reflected here is a stale diagram. Both module cycles the pre-refactor tree carried (`gen <-> residency`, `gen <-> vit`) are gone: `launch_v`/`launch_sync` moved into `kernels.rs` and the tensor loaders plus `Fp4` into the new `weights.rs`, and `boot.rs` (the shared container/context/config front door) joined the second layer. Diagrams 1 to 6 were re-read against the tree on 2026-09-17 and none of them contradicts it: the decode path, the residency picture, the serve picture and the converter pipeline are unchanged by a refactor that moved no launch, no kernel and no byte of `KERNEL_SRC`.
- 2026-09-17 (later, `487128d`): the graph re-generated after the six engine commits that followed `c1a68cd` and found unchanged; the status lines carry `487128d` and branch `main`. Diagrams 1 to 6 keep their 2026-09-12 Windows numbers, which are dated and machine-named; the Linux values of record that now sit beside them are in `README.md` and `CHANGELOG.md` (16k prefill 968 / 964 tok/s, decode 27.19 ms = 36.8 tok/s, the six-turn serve replay at 228.3 ms of prefill and 247.4 ms to first token, the 1024-row parity form at 740 tok/s). Not redrawn: the image path of `8ff2055` and `487128d` (the planner's vit reserve, the named 503, and the removal of the per-request splice buffer), which belongs in the serve picture of section 5 and is written in `architecture.md` 7.13.
