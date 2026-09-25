# crow-nest architecture diagrams

Living renderings of the approved spec (`architecture.md`, sections 1 to 9). A diagram
contradicting the spec is a bug in the diagram. Owner: issue #14. Updated with every stage
acceptance. All eight render the tree at `8bad310` (v0.3.1), 2026-09-18, branch `main`, and
each says what it was read against: diagrams 1 to 5 and the new diagram 8 were redrawn during
the day's `cea9406` pass over v0.3.0 (`9f12429`..`487128d`) and the first seven v0.3.1 commits,
and 2, 3, 5 and 8 moved again at the release pass for the commits that followed
(`6c87054`, `74970b5`, `b70310a`, `cf6a135`, `bc9cd9b`); diagram 6 (the converter) and diagram 7
(the module graph) were re-read against the same tree and are unchanged, and each says why.
Every box names the section of `architecture.md` it renders. Corrected on 2026-09-23 against
branch `release-2026-09-23` (`ebcd780`): diagrams 1, 2, 4 and 5 for the launcher's overlay
default, the `CROW_QFUSE` default, the flat PLE row read, the registered cold tier and the #100
zero-prefill reuse (changelog entry of that date); 3, 6, 7 and 8 were not re-read.

## 1 · System overview, from originals to served tokens

```mermaid
flowchart LR
    subgraph offline["offline · once per model"]
        ORIG["original safetensors\n131 shards · 360 GB BF16"] --> CONV["converter (Rust)\nRTN → NVFP4 · streaming"]
        CONV --> CNQ["CNQ4.5 container\n~101 GB · 4.5 bpw\nsections: text / ple / vit / mtp\nsidecar: per-tensor errors"]
        ORAC["oracle · torch f32 over the UNQUANTIZED originals"] --> GOLD["selftest/ · the layer-0 golden\n2 × 327,680 B + manifest.json = 658,998 B\nmanifest carries max_abs_gate 0.125"]
    end
    subgraph runtime["runtime · per session"]
        CNQ --> LOAD["loader · boot::open_model then Engine::load\nhost pinned budget DERIVED at boot\nvision reserve subtracted before N is chosen\nrender reserve granted best-effort (#110, default 0)\ntwo-sided clamp of N · cold-path policy per layer"]
        LOAD --> ENG["engine (Rust + CUDA sm_120a)\nWindows, and Linux since 2026-09-17 (#15)\none CUDA graph replay per decode token"]
        ENG -->|"tokens"| API["serve (Rust)\nOpenAI-style chat, np = 1\nGET /health · GET /props · GET /slots\nPOST /v1/chat/completions · POST /slots/0\nprefix cache A9 · slot file A10"]
    end
    SCOPE["tools/serve-linux.sh\nsystemd-run --user --scope, MemorySwapMax=0,\nMemoryHigh / MemoryMax from /proc/meminfo\nbare container: no overlay unless CROW_CNQ_OVERLAY is set (0254ed6)"] -.bounds the process.-> API
    CNQ -.ships together.-> PKG["the quant package: container + sidecar +\nhot sets + selftest/ + docs/model-card.md"]
    GOLD -.ships together.-> PKG
    PKG --> SELF["decode selftest · the gate a downloader can run\nexit code is the verdict (diagram 8)"]
    HARNESS["parity harness (#6)\n#159 methodology · the oracle children are retried"] -.interleaved A/B.-> ENG
    BASE["llama.cpp baseline"] -.same machine.-> HARNESS
```

Status: renders `cea9406`, 2026-09-18. New since the 885bb27 cut: the Linux port and the
bounded launcher (`#15`, 8.8 point 6), the derived host pinned budget and the vision reserve in
the loader box (2.1, 8.8 point 1, 7.13), the quant package and its travelling golden set
(8.10), and the retry around the harness's oracle children (8.9). Spec: 1.1 to 1.3 for the
offline arm, 2.1 and 8.4 for the loader, 7.11.1 to 7.11.7 for the endpoints, 5.1 and 8.9 for
the harness, 8.10 for the package and the self-test.

## 2 · Decode path at the current defaults

```mermaid
flowchart TB
    GRAPH["one CUDA graph replay per decode_step\nscalar refreshes · table flip · ev_commit before the launch\none boot line each names the staging kernel, the trickle order, the selection"]
    TRICKLE["stream trickle, deferred since #63c\ncopies parked in trickle_tick are issued right after the launch\nand overlap the replay (trickle_drain_after_launch)\nthe hot-set re-cut behind it ticks every 8 decode tokens\n(geo::TRICKLE_EVERY, TASK I 2026-09-17; 16 under #17)\nCROW_TRICKLE_DEFER=0 restores the eager issue before the launch"]
    GRAPH --> EMB["embedding row → residual stream 10240\nPLE host prep: n-gram ids, row cache, slot upload\nrow misses are ONE batch on the cnq::Warm reader pool\n(pread, 16 threads): about 0.5 ms of host time per step (TASK H)\nrow r read from the flat block stream at value 160·r,\nrepacked to the 108 B cache row (85a48e7, 2026-09-23)"]
    TRICKLE -.overlaps the replay.-> GRAPH
    EMB --> L0["layer 0 (GDN + MoE)"]
    L0 --> PLE["PLE at the top of layer 1\ngather · key/value proj · gate · dilated conv (state 9)\nh += gated + silu(conv), same place as prefill and reference"]
    PLE --> HC["hyper-connections\nper layer: mix → sub-block → inject\nunfused 8-launch chain + pre-scaled NVFP4 cascade by default\nsince 2026-09-23 (488a840); CROW_QFUSE=1 fuses hc + shared chains"]
    HC --> LOOP["layers 1..47 · 36 GDN + 12 attention\ninterleaved per layer_types"]
    LOOP -->|each MoE layer| RT["router_top10 (BF16)\n2560 × 512 · top-10 + shared expert"]
    RT --> STG["stage_cold_ca, default since #19e, inside the graph\npersistent grid, CROW_STAGE_BLOCKS=40 × 256 threads\npulls every cold combo of the layer into its VRAM staging slot\ncp.async 4 KB tiles · rewrites the combo pointer tables\nCROW_STAGE_KERNEL=1 falls back to stage_cold"]
    STG --> GEMM["expert GEMMs read VRAM pointers only\nhot slab slot or staging slot · NVFP4 GEMMs (sm_120a)"]
    GEMM --> COMB["shared_expert_gate combine"] --> HC
    LOOP -->|each attention layer| QSC["qsa_scores_par over the QSA ring and pooled cache"]
    QSC --> SEL{"QSA top-k, same list in the same order either way"}
    SEL -->|"default, CROW_QSA_PAR unset (#61b)"| PARS["qsa_select_par_h, CROW_QSA_PAR_BLOCKS=32 blocks, 12-bit histogram\nthen qsa_select_par_e, one block, 1024 threads\nthreshold refine · ascending emit"]
    SEL -->|"fallback, CROW_QSA_PAR=0"| FAST["qsa_select_fast on one block\nthe prefill selection keeps this form"]
    PARS --> ATT["attn_sel_split_l over the selected list, default since #61g\ngrid NQ × 1 × CROW_ATTN_SPLITS=8\nthe e4m3 KV bytes come out of a 256-entry shared table\nCROW_ATTN_LUT=0 is the pre-#61f attn_sel_split, bit-identical\nthe splits knob 4/8/16/32 is measurement only (#61a)"]
    FAST --> ATT
    ATT --> MERG["attn_merge reads the device scalar n_splits"] --> HC
    STAGER["stager (control plane only):\nresidency swaps · NVMe tier · telemetry"] -.off the critical path.-> GEMM
    LOOP --> OUT["lm_head (BF16) → logits row"] --> ARG["argmax_k (greedy gate discipline)"]
    OUT -.CROW_SAMPLE=1.-> SMP["host sampler (#20)\ntemp 0.7 · top_p 0.8 · top_k 20 · presence 1.5\nseeded xorshift · the presence set is the tokens THIS answer\ngenerated, never the prompt (7.11.17)"] --> TOK["token · CROW_STOP_EOS=1 stops at 248046 / 248044"]
    ARG --> TOK
```

Status: renders `8bad310`, 2026-09-18. The three defaults of 885bb27 are untouched —
`stage_cold_ca` (#19e), the deferred trickle (#63c), `qsa_select_par` with the
`CROW_QSA_PAR=0` fallback and 8 splits (#61a, #61b) — and one default moved on 2026-09-18:
the split decode attention kernel is `attn_sel_split_l` since `#61g` (`6c87054`), which reads
its e4m3 KV bytes out of a shared 256-entry table and is bit-identical to the
`CROW_ATTN_LUT=0` fallback by construction. The two levers the same day added are NOT drawn,
because both are DEFAULT OFF: `CROW_STAGE_PAR` (`#19`) would fork `stage_cold_ca` onto a side
stream and `CROW_GDN_SPLIT_Z` (`#71`) would split the grouped GDN input launch. Two boxes moved in
v0.3.0: the PLE row misses of a step are now one batched `pread` fetch on the reader pool
instead of one mapping fault per row (7.14 parts 1 and 2), and the adaptation behind the
trickle ticks every 8 decode tokens instead of 16 (7.14 part 4, `docs/env.md`
`CROW_ADAPT_EVERY`). The sampler box names the penalty scope measured for `#68` (7.11.17): the
set is the tokens this answer generated and never the prompt, so it is drawn inside the step and
not over the history. Windows operating point of record, 2026-09-12: 23.94 ms per token =
41.8 tok/s against 24.87 with the fallback (`decode_out/srv-61b.log`, architecture 3.4); the
Linux figure of record is 23.52 ms = 42.5 tok/s at the `#61g` default (`CHANGELOG.md`,
2026-09-18; it was 27.19 ms = 36.8 tok/s on 2026-09-17, before that flip). Both figures are
the ALL-FUSED `CROW_QFUSE` default of 2026-09-13 to 2026-09-23 and the pre-`85a48e7` PLE
read; the 2026-09-23 default (unfused chains, pre-scaled cascade, flat PLE rows) has no decode
figure of record yet (not measured). Spec: 3.4, 3.5,
4.2, 4.6, 7.14, 7.11.17.

## 3 · VRAM layout at the default operating point (262k ctx, FP8-KV)

```mermaid
pie showData title 32 GiB budget
    "hot experts (N≈155/layer, the 2.1 target)" : 21.2
    "dense resident (NVFP4 + BF16 keeps)" : 6.0
    "KV cache FP8" : 3.2
    "PLE hot-row cache (128 MB default since 2026-09-05)" : 0.13
    "GDN + QSA state" : 0.3
    "activations + graph pools" : 1.8
    "vision reserve + device sampler, HELD at boot since #72 (277.6 MB)" : 0.28
    "headroom" : 1.72
```

#110 (2026-09-25): `CROW_RENDER_RESERVE_MB` can keep VRAM FREE for another GPU client, Crow's
`render_page` first. Since the #110 follow-up it defaults to 0 and is granted best-effort (a
granted reserve is taken from the hot experts, 132,710,400 B per unit); the 1536 MiB default of the
first commit refused the boot on the 46 GiB pinned cap. Crow borrows VRAM per render (#117).

Status: renders `8bad310`, 2026-09-18 — the section 2.1 estimate of 2026-09-02 with the PLE
slice corrected 2026-09-05 (#16), plus the one slice v0.3.0 added: the vision reserve of TASK K
(2.1 last paragraph, 7.13) — the cap-sized tower scratch (228.5 MiB) and the interleaved-mrope
span tables (48.8 MiB at `n_ctx` 200,000), 277.3 MB together, so an image request can never find
the card full. It comes out of the headroom slice, which is what it was living in. **The slice is
an allocation since `#72` (`74970b5`, 2026-09-18), not a plan**: until that commit the reserve
only lowered N and the scratch stayed lazy, so everything taken AFTER the plan spent the slack
and the first image request of a session found 35.7 MiB free. `Engine::load` holds the scratch,
the two span tables and the device sampler's five buffers (0.27 MB, which is why the slice reads
277.6 MB now) before the budget verify, and a `[budget]` ledger names every one of them against a
0.25 GiB floor. Measured cost at the serve operating point (2026-09-17, same free-at-start
22.86 GiB, `n_ctx` 200,000, chunk 2048): N 157 → 155, VRAM used 30.83 → 30.58 GiB; holding it
moved N by nothing (2026-09-18, N 155 and 148 of 155 before and after, free VRAM 551 MiB after
load); `CROW_VIT=0` reserves nothing. The measured load line of 2026-09-05 read VRAM used
31.21 GiB. The HOST side of the same planner loop — the derived pinned budget — is diagram 4.
Spec: 2.1, 2.5, 7.13, 8.4 step 8.

## 4 · Residency and load: derived budget, pinned cold tier, page-cache discipline

```mermaid
flowchart TB
    subgraph hostmem["the host-memory model (8.8)"]
        RAM["cuda::free_physical_ram_parts\nfree_for_pin = MemTotal − (AnonPages + Shmem + SUnreclaim\n+ KernelStack + PageTables + Percpu) − driver_live\nthe driver's pool is reclaimable, so it counts as free;\ndriver memory live processes map does not (#103, 2026-09-23)"]
        BUD["manager::derive_host_pinned_budget\nmin(46 GiB cap, free_for_pin − CROW_RAM_MARGIN_GB)\none [budget] boot line, before anything is pinned"]
        RAM --> BUD
    end
    BUD --> PLAN["manager::ThreeStates::allocate\ntwo-sided clamp: VRAM lowers N, the pinned budget raises it\npending = LAUNCH_SLACK + ring reserve + vision reserve\n+ granted render reserve (#110, default 0, never allocated)\nrefuses only when no N satisfies both sides"]
    SIDE["hot-set sidecar · residency::sidecar_sets (#49)\none JSON object, a sets array of 48 rows;\nevery row adapted to THIS run's N per row — short rows padded\nwith the lowest unused ids, long ones truncated, each row logged;\nnot 48 rows, a bad id, an id twice: refused by name"] --> BUILD
    NVME["NVMe tier · the CNQ4.5 container\nread through Cnq::read_range"] --> BUILD
    PLAN --> BUILD["Residency::build\nRAM gate, hot slabs in VRAM, pinned cold tier,\nONE ascending sweep per expert tensor"]
    BUILD -->|"Cnq::fadvise_consumed drops the pages behind the\ncursor in 64 MiB batches, every section but ple"| PC["the load leaves no page-cache trail\nmax Cached 33.04 → 5.14 GiB, 2026-09-17\nCROW_CNQ_PURGE=0 disables it"]
    subgraph host["host RAM"]
        COLD["cold tier, PINNED to the derived budget\nexpert slabs as UVA pinned host pointers\nLinux default: anon mmap + cuMemHostRegister\n(CROW_PINNED_ALLOC=register, #103); wc on Windows"]
    end
    subgraph vram["VRAM · 32 GiB budget"]
        HOT["hot set: N expert slabs per layer\n7 slots spare for the trickle"]
        STAGE["staging slots: one per cold combo of this step"]
    end
    BUILD --> COLD
    BUILD --> HOT
    STAGER["stager (control plane only)\nplan_swaps · swap_in · the three-phase stream trickle\noff the decode critical path"]
    STAGER -.->|"re-cut every 8 decode tokens (TASK I);\na swap moves bytes between tiers and changes no id"| HOT
    COLD -->|"stage_cold_ca inside the graph (#19e)\ncp.async 4 KB tiles, measured 47.78 GB/s\nagainst 32.45 GB/s for stage_cold (#19d)"| STAGE
    COLD -.->|"CROW_STAGE=0: direct zero-copy\ninside the GEMM, ~23 GB/s"| GEMV
    STAGE --> GEMV["routed expert GEMVs read VRAM pointers only:\nhot slab slot or this step's staging slot"]
    HOT --> GEMV
    subgraph plerow["the PLE row path (7.14)"]
        MISS["Ple::ensure_rows misses: row r = value 160·r of the\nflat block stream, up to 4 blocks (144 B) per row,\nrepacked to 108 B (ple_row_span / ple_row_pad, 85a48e7)\nscattered over the 26.8 GiB ple section"]
        RUNS["cnq::page_runs coalesces and de-duplicates the pages"]
        WARM["Cnq::warm: the process's one reader pool,\nCROW_PLE_FETCH=16 threads, pread on a second\ndescriptor with POSIX_FADV_RANDOM, two queues so a\nbackground prefetch never blocks the current chunk"]
        MISS --> RUNS --> WARM
    end
    NVME --> MISS
    WARM --> PLEC["PLE hot-row cache in VRAM\nabout 0.5 ms of host time per decode step"]
    EXITP["Cnq::drop: on unix the exit purge steps AROUND the ple\nbyte range, so the rows the next process wants stay warm"]
    PLEC -.-> EXITP
```

Status: renders `cea9406`, 2026-09-18. The decode-side half is unchanged from the #19e default
of 2026-09-12 (staging numbers from `decode_out/srv-19d.log`, RTX 5090, 338 MB per token);
everything above it is v0.3.0 and v0.3.1. New boxes: the derived pinned budget with its
`/dev/nvidia-uvm` rule (8.8 points 1 to 3, 2.1), the vision reserve inside the planner's
`pending` bytes (8.4 step 8), the single ascending sweep with `fadvise_consumed` (8.8 point 4),
the `ple`-sparing exit purge (8.8 point 5), the batched PLE row fetch (7.14 parts 1 and 2), the
per-row sidecar rule of `#49` (2.2, `0adbe6a`), and the trickle's 8-token re-cut (7.14 part 4).
Corrected 2026-09-23: the `/dev/nvidia-uvm` fallback to `MemAvailable` is gone (`free_for_pin`
subtracts `driver_live`, 8.8 point 3), the Linux cold tier is registered anonymous memory
(8.8 point 7), and the PLE miss box reads the flat-stream span instead of `row*108` (2.4); the
TASK H figures in the PLE boxes were measured with the old offsets and not re-measured.
Spec: 2.1, 2.2, 2.3, 2.4, 3.4 as amended, 7.14, 8.8.

## 5 · Serve: one chat request end to end (in, reuse, out)

```mermaid
flowchart TB
    CLI["Crow CLI (crow_core.py)\nand any OpenAI-style client"]
    SRV["serve on 127.0.0.1:8099, blocking, one request at a time\nGET /health · GET /props · GET /slots\nPOST /v1/chat/completions: SSE stream, or one document\nwith stream false (#39 B3a) · POST /slots/0 action save or restore\nGET /v1/crow/vram · POST /v1/crow/vram/lend, /return (#117):\nstateless scratch VRAM lent while idle, remapped at the same addresses"]
    CLI --> SRV
    SRV --> GUARD["guarded(chat_route) arms cuda::RequestScope\na CUDA allocation refused INSIDE a request is a named 503\nnaming the allocation, its bytes and the free VRAM;\nthe engine is reset and keeps serving. Any other panic still ends the process"]
    subgraph inbound["on the way in (8.5 steps 2 to 4)"]
        PARSE["parse_chat: messages, tools, sampling fields, image blocks"]
        NORM["normalize_messages\ntool turns rendered as a tool_response block\narguments: a JSON string becomes the MAPPING the template needs (7.11.14)\nstrip_stored_think: a stored leading or trailing reasoning tag goes,\nthe text stays, every other shape byte-identical (7.11.16)\none [chat] normalised line per strip"]
        CHK{"check_messages, BEFORE the render"}
        ENC["tk.encode_chat: minijinja render + HF encode"]
        PARSE --> NORM --> CHK
        CHK -->|"renderable"| ENC
    end
    GUARD --> PARSE
    CHK -->|"any other shape"| R400["400 JSON naming the message index and the field"]
    ENC --> IMG["image branch, when the request carries images and the tower is loaded\nbuild_vision_plan: LRU hit, else prep_image → Vit::run\nthe plan holds HOST rows only (7.13)"]
    IMG --> CLAMP["clamped_max_tokens FIRST, then begin_vision\nso the interleaved-mrope span tables are at most n_ctx rows\ndefault 8192 since 8bad310, cap 32768"]
    CLAMP --> LCP{"longest common id prefix L\nof the request ids vs the held history\nids only, never text (#31 A9)"}
    LCP --> P["reuse point P: the largest snapshot position at or below L\nwhose rows are all PREFILL CLEAN\nnothing is erased: pos moves back to P,\nrows at or above P are stale but unreachable\n#100: an identical re-send reuses a slot holding its\nlast logits row, 0 prefill; #101: a rollback forgets\nthe slots above P (2026-09-23, host tests only)"]
    LCP -.->|"no reuse, a request shorter than the held conversation (#118)"| PARK["park: KV rows 0..min(T, CROW_PREFIX_PARK_ROWS = 8192)\nand the pooled QSA blocks under them go to HOST RAM (108.0 MiB at fp8 KV),\nhistory and snapshots kept; then the cold start"]
    PARK -.->|"the next main turn: unpark, rows back byte for byte"| ROLL
    P --> ROLL["rollback: GDN state, conv state and QSA ring restored\nfrom the snapshot at P; P = 0 is the cold start,\none 16k prefill of 21.6 to 22.0 s"]
    ROLL --> PF["prefill P to prompt_len\nrewrites the re-rendered previous answer as prefill rows"]
    PF --> S2["snapshot at the prompt end (prefill clean)"]
    S2 --> ARM["the sampler prologue: arm_sampler for a sampled request,\npark_sampler for a greedy one\ntwo [chat] lines name the SOURCE of every sampling value\n(request or data sheet) and the penalty scope:\ncleared for this request, generated tokens only (7.11.17)"]
    ARM --> DEC["decode_step loop, the diagram above"]
    subgraph outbound["on the way out (7.11.13, 7.11.16, 7.11.18)"]
        TS["ToolStream: the tool-open token arms the parser;\nEmit::Args fragments are never rewritten, with ONE exception since cf6a135:\na declared array or object that failed to parse gets one minimal repair pass\nand is kept only if it then parses AS THAT TYPE, else the string goes out with a warn;\na malformed call carries its raw markup as content"]
        TF["ThinkFilter on Emit::Content only\nLead: a block the model opens is owned\nInside: its text leaves as delta.reasoning_content\nBody: a stray closing tag is DROPPED, a held prefix flushes at the end\nso no byte of an answer is lost, on either request form"]
        SINK{"ChatSink, the one split between the two request forms"}
        TS --> TF --> SINK
    end
    DEC --> TS
    SINK -->|"SseSink"| WIRE["streamed chunks, usage and timings on the final chunk\ncached_tokens = P, always present\nstill_there = the failed flush of the first frame"]
    SINK -->|"CollectSink"| DOCU["one chat.completion document: content, tool_calls\nand reasoning_content only when something was stripped\nusage and timings ALWAYS present"]
    SINK -.->|"when the answer is complete, either form"| REP["the cross-turn repeat counter (#68, 7.11.19)\nFNV-1a over the GENERATED ids into a ring of the last 8 answers\nrepeat_of · repeat_run · single_token on the routing line,\nthe [chat] line only when they are not the healthy values\nWARN at three identical answers in a row: no 4xx, no brake,\nthe sampler and the wire untouched. Per PROCESS, by design"]
    DOCU --> PROBE{"CollectSink::still_there(step), every step (PROBE_EVERY = 1)\npoll(POLLRDHUP), timeout 0, nothing on the wire\nagainst the BASELINE probe taken when the sink was built"}
    PROBE -->|"EOF already there: a client that half-closed"| DOCU2["it gets its whole document"]
    PROBE -->|"EOF appears later: gone"| STOPG["the generation ENDS, the slot is free;\nfinish abort (#99), status 200 OK (client gone)"]
    SRV --> SLOT["slot file, across processes (#32 A10)\nKV rows 0..pos + pooled blocks 0..floor(pos/4)\nmeasured 352,843,384 B at 16k\nneeds --slot-save-path, else both actions answer 400\ncontract: n_saved = n_restored"]
    SLOT -.->|"restore into a fresh process"| ROLL
```

Status: renders `8bad310`, 2026-09-18. The reuse chain and the slot file are the 2026-09-12
picture (A9 #31, A10 #32, the dropped after-answer snapshot M2b #36, the stream false document
#39 B3a); the warm turn at 16k reused 99.41 percent of the prompt in 404 ms (README measured
table, #31). Added since: the `guarded` / `RequestScope` 503 and the clamp-before-`begin_vision`
order (7.11.8 last row, 7.13, TASK K), the normaliser's two jobs (7.11.14 for `arguments`,
7.11.16 end 2 for a stored reasoning tag, `667b68b`), `check_messages` as the 400 before the
render (7.11.8), the `ThinkFilter` between the tool parser and the sink (7.11.16 end 1,
`667b68b`), the sink split with `CollectSink`'s `ClientProbe` (7.11.13, 7.11.18, `20bc121`) and
the sampling-provenance lines (7.11.17, `f14e557`). Added at the release pass, for the three
commits of that afternoon: the cross-turn repeat counter as a dotted box off the sink
(7.11.19, `b70310a`) — it reads the answer and changes nothing, which is why it hangs off the
path instead of sitting in it; the one exception the tool stream now makes to "fragments are
never rewritten" (`cf6a135`); and `clamped_max_tokens`, whose default is 8192 since `8bad310`
and no longer 1024. Spec: 7.3 to 7.8 for the reuse chain, 7.11.3
to 7.11.8, 7.11.13, 7.11.16 to 7.11.19, 7.13, 8.5.

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

Status: unchanged since the first cut of 2026-09-02 (#2); no converter stage has landed since,
verified against the tree at `cea9406` on 2026-09-18 — `converter/src` was last touched before
v0.3.0, and the two v0.3.1 container-side facts (the `ple` range of the exit purge, the row
fetch) are reader-side, not writer-side. The per-tensor sidecar this diagram writes is NOT the
hot-set sidecar of diagram 4, and 2.2 refuses it by name. Spec: 1.2, 1.3, 1.5.

## 7 · Module dependency graph of the engine crate

```mermaid
graph LR
  subgraph L0[leaves]; log["log.rs: tracing facade + EnvFilter\nRotatingFile (size + day + keep N + gzip)\nboot line, routing line"]; cuda[cuda.rs]; cnq[cnq.rs]; geo[geo.rs]; tokenizer[tokenizer.rs]; toolcall[toolcall.rs]; end
  subgraph L1[on the leaves]; kernels[kernels.rs: kernel table + launch_v + kprof]; manager[manager.rs]; sample[sample.rs]; weights[weights.rs: tensor loaders + Fp4]; boot[boot.rs]; end
  residency[residency.rs]; vit[vit.rs]; gen[gen.rs]; cache[cache.rs]; reset[reset.rs]; slot[slot.rs]
  kernels --> cuda; manager --> cuda & geo; sample --> geo; weights --> cnq & cuda; boot --> cnq & cuda & geo
  residency --> cnq & cuda & geo & kernels & manager; vit --> cnq & cuda & geo & kernels & weights
  gen --> cnq & cuda & geo & kernels & manager & residency & sample & vit & weights
  cache --> cuda & gen & geo; reset --> cuda & gen & geo; slot --> cache & cuda & gen & geo
  gen -. "tracing event, no use edge" .-> log
```

Status: **one box added, `log.rs` (#13, 2026-09-18).** It is an L0 leaf — it uses no module of
this crate — and it is the module every other one reaches, because 152 of the 153 `eprintln!` sites
of `engine/src` are `tracing` events now. Only ONE dotted edge is drawn (`gen` → `log`) and it is
dotted on purpose: a `tracing::info!` call travels through the crate-global subscriber, not
through a `use`, so twelve solid arrows into one leaf would say something the import graph does
not say and would make the picture unreadable. Same class of invisible edge as the
`gen` → `reset` one named below. `log.rs` has its own spec section, 9. Otherwise unchanged: the
`use crate::` edges of `engine/src` were regenerated from the tree at
`cea9406`, 2026-09-18, and again at `8bad310` (v0.3.1) at the release pass, and are edge for edge
the graph added at `c1a68cd` and re-verified at
`487128d`: acyclic since `bb9d2ca` broke `gen <-> residency` and `gen <-> vit`, with
`weights.rs` and `boot.rs` in the second layer. The seven commits after `487128d` touched two
library files — `sample.rs` (`f14e557`, the two penalty-scope tests) and `residency.rs`
(`0adbe6a`, `sidecar_sets`) — and no module moved; everything else of v0.3.1 landed in the bins,
which this graph does not draw: `ThinkFilter`, `ClientProbe` and the sampling provenance in
`bin/serve.rs`, `oracle_child` in `bin/parity.rs`, the `selftest` mode in `bin/decode.rs`, and
the `CROW_CNQ` reads of `bin/plecheck.rs` and `bin/states.rs`. Not drawn: the feature-gated
`cutile_pilot.rs` (`cuda`, `kernels`; nothing calls it), and the one edge no import graph shows
— `impl Drop for Engine` in `gen.rs` calls `Engine::drop_decode_graph`, an inherent method
defined in `reset.rs`. Module by module: `architecture.md` section 8.1 and 8.2.

## 8 · Gates, guards and the self-test the package carries

```mermaid
flowchart TB
    subgraph gate["tools/gate-linux.sh [outdir] · nine items, GREEN or RED each, non-zero exit on any RED (8.7)"]
        P8["parity 8 rows · bceba6ff7724…\nthe one form identical to the Windows reference"]
        P512["parity 512 rows · 8387234709271515…"]
        PTF["P8 teacher-forced · 3bb3e69edf90…\nprefill 8 ids, the other 504 through decode_step"]
        D32["decode run over 32 ids"]
        TST["cargo test --release · TESTS 358\n237 lib + 110 serve + 6 parity + 5 decode"]
        CLP["clippy --all-targets · CLIPPY 1505"]
        G1["check_env_docs · code 107, doc 107"]
        G2["check_readme_dates · 0 offenders"]
        G3["check_model_card_dates · 0 offenders,\nrunning since 2026-09-18 (it was never committed before)"]
    end
    HAND["parity 1024 rows · 117dd8d9d8dc…\nnot in the script: a full long-prompt run, checked by hand"]
    subgraph pkg["the gate a downloader can run (8.10)"]
        WRAP["tools/selftest.sh [package-dir] [--with-originals] [--full]"]
        CTRL{"test ! -d models in the package directory"}
        SHA["the golden set's own sha256: the SHA256SUMS lines\nof selftest/ checked with sha256sum -c, no read of the container"]
        RUN["decode selftest [golden_dir]\nreads CROW_CNQ / CROW_HOTSETS, so it runs from any cwd;\nmanifest checks: decoder_layer and attn_subblock,\nshapes and max_abs_gate required per check,\na golden whose byte length misses its shape is REFUSED"]
        VERD["max_abs 9.184837e-2 against the 0.125 gate, NaN 0,\nthe same six digits in both arms;\nthe EXIT CODE is the verdict"]
        WRAP --> CTRL
        CTRL -->|"models/ present without --with-originals"| REF["RED, exit 1 — the engine is never started"]
        CTRL -->|"absent, or --with-originals"| SHA --> RUN --> VERD
    end
    subgraph harn["the ten-task parity harness (8.9)"]
        PH["a phase: ten tasks, interleaved A/B"]
        OC["oracle_child(program, args, stdin, what, backoff)\nthree attempts, 2 s then 5 s between them\nthe diagnosis LEADS with the exit code (exit_code_name)\na partial write or an empty stdout fails the attempt"]
        CH["tokenize_ids.py --chat, once per task\ndetokenize_ids.py, once per phase\nspawned BEFORE open_model, out of the worst host-memory window"]
        REC["the record names the retry: oracle_retries on the task row,\nthe keys appear only when something was retried"]
        FC["all three attempts fail: fail-closed panic with the whole\nattempt table, nothing recorded for that task"]
        PH --> OC --> CH
        OC --> REC
        OC --> FC
    end
    subgraph rep["the replays that reproduce a live bug (7.11.16, 7.11.17, 7.11.18)"]
        RT["tools/replay-toolcalls.py\n--think: no reasoning tag may reach the content\n--gone-client: the drop and the half-close, one run\n--poison / --refusals: the arguments contract"]
        RS["tools/replay-session.py: robin's session at 168,928 ids"]
        LG["tools/longctx-gate.py: the first long-context quality gate\n5 of 5 Pass, greedy, 2026-09-18"]
    end
    gate --> VOR["the numeric contract of record, per platform\nbyte-identical logits or the commit does not land"]
    HAND --> VOR
    pkg --> VOR
    harn --> TEN["the ten-task gate: 10 of 10 identical id streams"]
    rep --> LIVE["the live shapes: a filtered content path, a freed slot,\nand a degeneration that is context length, not the sampler"]
```

Status: new on 2026-09-18, renders `8bad310` — its two host-side boxes carry the values the
script pins at the release commit (`TESTS=221`, `check_env_docs` 89 = 89); updated 2026-09-23 on `release-2026-09-23` to
`TESTS=358` (measured that day), `CLIPPY=1505` (the pinned value, not re-counted) and `check_env_docs` 107 = 107. The three
parity shas in the boxes are the values of the pre-2026-09-23 engine: `488a840` and `85a48e7` change the numerics on purpose,
and no new values are recorded yet. It draws what section 5.2 asks for and what
v0.3.0 and v0.3.1 built around it: the nine-item Linux gate and the two host-side values it
pins (8.7), the package self-test with its `models/` control and its sha256 (8.10, `#64`), the
bounded retry around the harness's two oracle children (8.9, `#65`), and the three replay
commands that reproduce the live bugs of `#67`, `#54` and `#68` (7.11.16, 7.11.18, 7.11.17).
Every number in it carries its section; the 1024-row form sits outside the script by design.
Spec: 5.1, 5.2, 5.3, 7.11.16 to 7.11.18, 8.7, 8.9, 8.10.

## Changelog

- 2026-09-02: first cut — rendered from approved spec sections 1–6 + decisions (#2). Stage-1 converter reflects the implemented converter; runtime boxes are the spec's design, not yet code.
- 2026-09-02 (later): cold path amended to zero-copy direct read (spec 3.4 amended after the #8 pre-study) — ring/stager moved off the decode critical path to control plane.
- 2026-09-05: decode path redrawn after #11 — the PLE step now sits at the top of layer 1 (it ran before layer 0 in `decode_step` until 2026-09-05 07:00, which is what degenerated the answers); sampler and EOS stop (#20) added as the opt-in tail; PLE cache default 128 MB (#16). Section 3 stays the 2026-09-02 estimate with the PLE slice corrected; the measured load line on 2026-09-05 read "VRAM used 31.21 GiB (dense 7.02 GiB + hot experts 160 × 48 × 2.64 MB + states)".
- 2026-09-12: post #61b pass. Decode path redrawn for the three default flips: the staging kernel `stage_cold_ca` (#19e, engine commit e256004), the deferred trickle (#63c, engine commit 095a1c8) and the parallel QSA selection `qsa_select_par` with the `CROW_QSA_PAR=0` fallback plus the `CROW_ATTN_SPLITS` measurement knob (61a and 61b, engine commit 9696b13). The resident-or-cold decision at the GEMMs is gone: the staging kernel hands the GEMMs VRAM pointers in every case, so zero-copy direct read survives only behind `CROW_STAGE=0`. New section 4, the residency picture (hot set VRAM, pinned cold tier, zero-copy read), and new section 5, the serve picture (endpoints, prefix cache A9, slot save and restore A10). The system overview server box now names the endpoints. Pie and converter unchanged; every diagram carries a status line.
- 2026-09-17: new section 7, the module dependency graph of the engine crate, after the three refactor cuts of branch `linux-refactor` (74c79f2, bb9d2ca, 7ddd296). It is the first diagram in this file that renders the CODE rather than the spec, and it is generated from the `use crate::` edges, so a module move that is not reflected here is a stale diagram. Both module cycles the pre-refactor tree carried (`gen <-> residency`, `gen <-> vit`) are gone: `launch_v`/`launch_sync` moved into `kernels.rs` and the tensor loaders plus `Fp4` into the new `weights.rs`, and `boot.rs` (the shared container/context/config front door) joined the second layer. Diagrams 1 to 6 were re-read against the tree on 2026-09-17 and none of them contradicts it: the decode path, the residency picture, the serve picture and the converter pipeline are unchanged by a refactor that moved no launch, no kernel and no byte of `KERNEL_SRC`.
- 2026-09-17 (later, `487128d`): the graph re-generated after the six engine commits that followed `c1a68cd` and found unchanged; the status lines carry `487128d` and branch `main`. Diagrams 1 to 6 keep their 2026-09-12 Windows numbers, which are dated and machine-named; the Linux values of record that now sit beside them are in `README.md` and `CHANGELOG.md` (16k prefill 968 / 964 tok/s, decode 27.19 ms = 36.8 tok/s, the six-turn serve replay at 228.3 ms of prefill and 247.4 ms to first token, the 1024-row parity form at 740 tok/s). Not redrawn: the image path of `8ff2055` and `487128d` (the planner's vit reserve, the named 503, and the removal of the per-request splice buffer), which belongs in the serve picture of section 5 and is written in `architecture.md` 7.13.
- 2026-09-18, after `#13` (engine logging): **diagram 7 gained a box**, `log.rs`, the new L0 leaf that every module reaches through the subscriber instead of through a `use` — drawn as one dotted edge from `gen` with the reason in the status line, and specified in `architecture.md` section 9. **Diagram 8** carries the two host-side values the gate pins, so its boxes moved with them: `TESTS 200` (113 lib + 78 serve + 6 parity + 3 decode), `CLIPPY 1421` and `check_env_docs · code 86, doc 86`. No other diagram changed: logging adds no stage to the decode path, no slice to the VRAM pie and no step to the request path, and the two structured lines it adds (`boot`, `routing`) are outputs of boxes that are already drawn. Diagrams 7 and 8 were both render-checked locally with `@mermaid-js/mermaid-cli` 11.17.0.
- 2026-09-18, the #14 audit of v0.3.0 and v0.3.1 (`cea9406`): the eight commits that landed since the 2026-09-12 pass were read against every diagram, and the one debt the entry above names is paid. Per diagram:
  - **1, system overview: redrawn.** The loader box says the host pinned budget is DERIVED and the vision reserve is subtracted before N is chosen (2.1, 8.8 point 1, 7.13); the engine box says Linux since 2026-09-17 (#15) and the bounded launcher `tools/serve-linux.sh` is drawn as what wraps the process (8.8 point 6); the quant package and the travelling `selftest/` golden set are new boxes (8.10), and the harness box names the retried oracle children (8.9).
  - **2, decode path: redrawn, two boxes.** The PLE prep names the batched `cnq::Warm` row fetch instead of one mapping fault per row (7.14 parts 1 and 2, `1032bc5`), and the trickle box names the 8-token re-cut that replaced #17's 16 (7.14 part 4, `4004e66`). The sampler box gained the penalty scope measured for `#68` — the tokens this answer generated, never the prompt, cleared per request (7.11.17, `f14e557`). The three defaults of #19e, #63c and #61a/#61b are untouched, so the rest of the path is the 885bb27 drawing.
  - **3, VRAM pie: redrawn, one slice.** The vision reserve of TASK K (277.3 MB: tower scratch 228.5 MiB + mrope span tables 48.8 MiB) is planned VRAM now and comes out of the headroom slice it used to live in (2.1 last paragraph, 7.13); the reserve cost N 157 → 155 at the serve operating point.
  - **4, residency: redrawn and renamed** to "Residency and load", because v0.3.0 put a host-memory model in front of it: the derived budget with the `/dev/nvidia-uvm` fallback (8.8 points 1 to 3), the two-sided clamp with the reserve in `pending` (8.4 step 8), the one ascending sweep with `fadvise_consumed` and the flat page cache (8.8 point 4), the `ple`-sparing exit purge (8.8 point 5), the per-row hot-set sidecar rule of `#49` (2.2, `0adbe6a`) and the PLE row path — `page_runs`, `Cnq::warm`, 16 reader threads, two queues (7.14).
  - **5, serve: redrawn as the whole request path** (in, reuse, out), which is where the #67, #54, #68 and TASK K work lands: `normalize_messages` with both its jobs (7.11.14, 7.11.16 end 2), `check_messages` as the 400 before the render, the `ThinkFilter` between the tool parser and the sink with its three states (7.11.16 end 1), the sink split and `CollectSink`'s `ClientProbe(POLLRDHUP)` with its baseline (7.11.13, 7.11.18), the two sampling-provenance lines (7.11.17), and `guarded` + `RequestScope` answering a request-scoped allocation failure with a named 503 (7.11.8, 7.13). The prefix-cache chain and the slot file are unchanged.
  - **6, converter: unchanged.** No converter stage has landed since 2026-09-02; `converter/src` was not touched in v0.3.0 or v0.3.1, and the two container facts that did move (the exit purge's `ple` range, the row fetch) are reader-side. The status line now says which sidecar this one is, because diagram 4 draws the other.
  - **7, module graph: unchanged, re-generated.** The `use crate::` edges of `engine/src` at `cea9406` are edge for edge those of `487128d`: only `sample.rs` and `residency.rs` were touched in the library, and every other v0.3.1 change landed in a bin, which this graph does not draw. The status line now names those bins so the next reader does not look for `ThinkFilter` in a module.
  - **8, gates, guards and the self-test: NEW.** The verification side had no picture at all, and three commits of v0.3.1 built one: `tools/gate-linux.sh`'s nine items and the two host-side values they pin (8.7), `decode selftest` with `tools/selftest.sh`, the `test ! -d models` refusal and the sha256 of the shipped golden (8.10, `#64`), `oracle_child`'s bounded retry with the exit code leading the diagnosis (8.9, `#65`), and the three replay commands that reproduce `#67`, `#54` and `#68` (7.11.16, 7.11.18, 7.11.17). The 1024-row parity form is drawn outside the script, where 8.7 puts it.
- 2026-09-18, the v0.3.1 release pass (`8bad310`): the seventeen commits that landed after the `#13` pass (`788fb64`) were read against all eight pictures, and four moved. **2, decode path:** the split decode attention kernel is `attn_sel_split_l` since `#61g` (`6c87054`) and the box says so, with `CROW_ATTN_LUT=0` named as the bit-identical fallback; the Linux figure in the status line is 23.52 ms = 42.5 tok/s now, and the two levers of the same day (`CROW_STAGE_PAR` `#19`, `CROW_GDN_SPLIT_Z` `#71`) are deliberately NOT drawn, because both are DEFAULT OFF and a picture of a default is a picture of what runs. **3, VRAM pie:** the vision slice is HELD at boot since `#72` (`74970b5`) and reads 277.6 MB, the device sampler's five buffers included, against a `[budget]` ledger and a 0.25 GiB free-VRAM floor — the slice was planned-only before, which is exactly what let the first image request of a session find 35.7 MiB free. **5, request path:** the cross-turn repeat counter of `#68` (`b70310a`) hangs off the sink as a dotted box, because it reads the answer and changes nothing; `ToolStream` names the one exception it now makes to "fragments are never rewritten" (`cf6a135`); and the clamp box carries the 8192 default of `8bad310`. **8, gates:** `TESTS 221` and `check_env_docs` 89 = 89, the values the script pins at this commit. Diagrams 1, 4, 6 and 7 were re-read and are unchanged. The module graph in particular: the whole `crate::` edge set of `engine/src` at `8bad310` is the one of `788fb64`, with exactly one addition, `manager → gen::sampler_bytes` inside `#[cfg(test)] mod tests_72` — test code, not a layering edge, and the picture draws the layering. Everything the seventeen commits added to the library landed inside modules that were already drawn (`vit::arm_scratch`, `manager::PostPlan`, `toolcall::repair_json`, `gen`'s two levers), and the rest landed in the bins, which this graph does not draw. All eight were rendered locally with `@mermaid-js/mermaid-cli` 11.17.0 before the release commit (#14).
- 2026-09-23, branch `release-2026-09-23` (`ebcd780`): four diagrams corrected where they stated a default or the PLE layout that the branch changed. **1:** the launcher box says the bare container is the default (`0254ed6`; the dense-overlay default of `0924406` lasted from 13:57 to 18:43 that day). **2:** the PLE prep box names the flat-stream row read (`85a48e7`, architecture 2.4) and the hyper-connection box the unfused default of the activation-floor fix (`488a840`, `CROW_QFUSE=1` opt-in); the status line says the Linux decode figure is the all-fused, pre-fix one and that the new default is not measured. **4:** the `/dev/nvidia-uvm` decision and its `MemAvailable` fallback are removed (`free_for_pin` subtracts `driver_live`, #103), the cold tier names `CROW_PINNED_ALLOC=register`, the PLE miss box the flat-stream span. **5:** the reuse box names the #100 zero-prefill reuse and the #101 rollback rule. Not drawn: the #102 `CROW_KV` read at `boot::open_model`, the `492f137` grammar relaxations (architecture 7.11.23), the GDN/PLE window shift of `c4d37ca` (architecture 7.3). Diagrams 3, 6, 7 and 8 were not re-read in this pass.
- 2026-09-25, #110: **1, system overview:** the loader box names the render reserve. **3, VRAM pie:** one new slice, the render reserve (1.5 GiB, kept free for a co-resident renderer, `CROW_RENDER_RESERVE_MB`), taken from the hot-expert slice (21.2 -> 19.7, arithmetic, not a measured boot). **4, residency and load:** the planner box names the render reserve in `pending`. The other diagrams do not show the planner's reserves.
- 2026-09-25, #110 follow-up: the render reserve defaults to 0 and is granted best-effort. **1** and **4** name the grant, **3** drops the 1.5 GiB slice and restores the hot-expert slice (19.7 -> 21.2, arithmetic).
- 2026-09-25, v0.6.0 (#117, #118): **5, serve:** the server box names the three `/v1/crow/vram` endpoints of #117; a new PARK node between the prefix decision and the rollback draws #118, the held conversation parked in host RAM for a shorter unrelated request and restored by the next main turn. The other diagrams do not change.
