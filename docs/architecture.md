# crow-nest engine spec

**Status: fully approved by robin on 2026-09-02 (all six sections, issue crow-nest #4).**

Approved section by section (issue crow-nest #4; a section is truth only after robin's
approval is recorded there). Environment for every number: `docs/system-landscape.md`.
Methodology: Crow #159. No north star in the record's sense — misses are measured
results that re-cut stages.

---

## Section 0 — goals and operating points (APPROVED by robin 2026-09-02, with the prefill provenance integrated)

### 0.1 Status of the goals

Product requirements from robin (2026-09-01/02, epic #1). They are targets the engine
aims at and the harness reports against — the record's discipline stands: a miss is a
measured result that re-cuts a stage (timebox guard), never a project failure. No goal
is a pass/fail gate.

### 0.2 Context

- **Floor 200,000 tokens** — every shipped configuration must hold ≥ 200k; the engine
  refuses a configuration that cannot.
- **Default 262,144** (architecture ceiling) with **FP8-KV**; slider [200k, 262,144].
- The oracle checks FP8-KV against the BF16 reference; if it goes red, the fallback
  operating point is 200k + BF16-KV (~4.6 GiB KV).
- State at default: KV ~3.0 GiB (FP8) / ~6.0 GiB (BF16), GDN ~0.12 GB (f32, fixed),
  QSA per its budget of 2,048 tokens.

### 0.3 Throughput targets, with their measurement points

- **Decode ≥ 42 tok/s** — measured at batch 1 (`-np 1`), session filled to the **200k
  floor** (not the ceiling), after a discarded warm-up request; reported against the
  interleaved llama.cpp baseline on this machine (41 tok/s endstand, same session).
- **Prefill ≥ 972 tok/s**, binding at the **32k operating point**, measured as the
  **completed-prompt average** (prompt tokens ÷ prompt time). Instantaneous rates are
  reported alongside but never substitute for the average.
  - Provenance (grounded 2026-09-02): 972 first appears as the baseline's instantaneous
    rate at 26 % of a ~31.7k-token prefill (`crow-lab/runs/2026-08-30-levers-159/
    control-03`: n_tokens 8,245, 972.24 tok/s). The completed-32k baseline average is
    ~750 tok/s (ladder: 750.75 at 30,777 tokens). The target is therefore honestly a
    ~+29 % stretch over baseline at 32k — a goal, not a gate.
- Every throughput number names: context fill, prompt length, chunk size, quant of both
  engines under test.

### 0.4 Latency metrics

- **TTFT** (time to first token) at the 200k floor with a named prompt length; target
  set by robin against crow-nest's first measured prefill (no invented number now).
- **Mean inter-token latency** is implied by the decode goal: 42 tok/s ⇔ ~23.8 ms.
- "Minimal latency" as a design constraint is already decided and needs no number: no
  host in the hot loop, no kernel launch for synchronization (job ring, stream memops).

### 0.5 Measurement discipline (binding)

Interleaved A/B in the same session, one variable at a time, resolution stated,
environment per `docs/system-landscape.md`. Performance is a measurement, not a
condition.

---

## Section 1 — model format and streaming converter (APPROVED by robin 2026-09-02: BF16 keep-set as proposed; 551 GB free NVMe confirmed)

> **Official name (robin, 2026-09-02): the format and quant family is CNQ; the base
> variant is CNQ4.5** (container magic `CNQ1`, 4.5 bpw NVFP4). Planned follow-ups inherit
> the scheme: CNQ4.5-C (per-expert calibrated scales, stage 2), CNQ4.5-P8 (PLE block
> exchanged to FP8 if the oracle goes red). KV dtype stays runtime config, not part of
> the name. First artifact: `Qwen3.8-Flash-Next-CNQ4.5.cnq`.

### 1.1 Input

The original safetensors: 131 shards, 359,999,963,128 B, revision
`de4b8e4d43b917e7706784d8bb445c9af86a3540`, read with mmap streaming — the model is
never held in RAM. Disk requirement for a conversion: input 360 GB + output ~101 GB
(plus a verification sidecar) — **~470 GB free NVMe**.

### 1.2 Output format (own container, no GGUF)

- **Container**: own binary, shard-aligned, JSON header with the tensor index
  (name, logical dtype, shape, block format, file offset) — conceptually safetensors'
  header, not its code; the container is ours.
- **Quantized weights**: NVFP4 blocks per ggml geometry — 64 values per block, 32 B
  packed E2M1 + 4 ue4m3 scales (one per 16-wide sub-block) = 36 B / 64 values = 4.5 bpw,
  **plus one global f32 scale per tensor** (two-level scaling; probe-pinned hardware
  facts apply, the global scale is a converter convention to be validated numerically in
  the first oracle run).
- **Kept in BF16** (proposal): embeddings and `lm_head` (untied, 2 × 1.27 GB BF16 —
  2.5 GB of 360, no reason to risk them), **router GEMM** (2560 × 512 per layer) and
  `shared_expert_gate`, all norms (`hc_norm` et al.). Routing stability and norm
  precision are cheap to keep.
- **Everything else** (GDN, attention, hyper-connection low-rank, experts, shared
  expert): NVFP4.
- **PLE block**: NVFP4, **exchangeable** — the header tags it as a self-contained
  section so an FP8 swap (oracle fallback) needs no format change.
- **ViT + MTP**: carried in the format, tagged optional-to-load (decision 2026-09-02).
- **KV dtype is runtime config** (FP8 E4M3 default), not file content.

### 1.3 Stage 1 — calibration-free RTN

Shard-streaming on read AND write; each tensor quantized on the fly, per-tensor global
scale computed before emitting its blocks; output written incrementally so at most one
working copy exists beyond input and output.

### 1.4 Stage 2 — per-expert calibration (separate ticket, later)

Re-emission of scales only (format unchanged) from Crow's routing statistics —
ModelOpt's `--calib_all_experts` idea with real traffic instead of synthetic samples.

### 1.5 Acceptance

- Round-trip decode of sample tensors matches BF16 reference within documented error;
  per-tensor max/mean deltas recorded in a verification sidecar next to the output.
- The oracle (#6) consumes the **unquantized originals** — this stage's output is never
  the quality reference.

### 1.6 Resolved (robin, 2026-09-02)

- BF16 keep-set approved as proposed (embeddings, lm_head, router GEMM, gates, norms).
- Disk: 551 GB free NVMe — conversion runs fit comfortably (~470 GB needed).
- Budget consequence carried into section 2: the BF16 keep-set adds ~1.8 GB dense
  VRAM; the hard hot-set ceiling drops from ~177 to ~169 experts/layer (default 160
  unchanged, loader auto-clamps).

---

## Section 2 — memory layout: residency, PLE window, three-state manager (APPROVED by robin 2026-09-02)

### 2.1 VRAM budget at the default operating point (262k context, FP8-KV)

| item | GB | note |
|---|---|---|
| dense resident | ~6.0 | 12.5 GB BF16 → NVFP4 (~3.5) + BF16 keep-set 2.5 (embeddings, lm_head, router, gates, norms); ViT/MTP carried in file, NOT loaded at this stage |
| hot experts | N × 0.132 | 2.76 MB per expert per layer × 48 layers; default N=160 → 21.2 GB |
| KV cache | ~3.2 | FP8 E4M3, 12 KiB/token × 262,144; BF16 would be 6.4 |
| GDN state + QSA | ~0.3 | GDN f32 36 × ~3.1 MB ≈ 0.11; QSA per budget 2048 |
| PLE hot-row cache | 1.0–2.0 | window cache, sized by telemetry (2.2) |
| activations + graph pools | 1.5–2.0 | decode graphs are small; measured at first integration |
| **total non-expert** | **~13.0–13.5** | of 34.4 (32 GiB) |
| **expert budget** | **~20.9–21.4** | → hard ceiling **~158–162** experts/layer with honest pool sizing |

Loader rule (binding): N=160 is the **target**; the loader verifies the full budget with
measured overheads at load time and auto-clamps N if the sum exceeds VRAM — it refuses
configurations that fall below the 200k context floor, never silently degrades context.

### 2.2 Expert residency (per layer, data-driven)

- Residency sets are **per-layer top-N by selection frequency** — per-layer ranking, not
  the global curve (#3; per-layer selection achieves ≥ the global coverage).
- Warm-up: first runs accumulate per-layer selection counts (every routed choice counts,
  hit or miss — the print_locality discipline), promote the top-N per layer, and persist
  the sets in a sidecar next to the model file; refreshable by config or command.
- Routing-gated skip: a layer whose 10 routed experts are all resident has **no cold
  job at all** — the handoff count responds to this, measured per token (#7 acceptance).

### 2.3 Cold path (variant A; stager built for C from day 1)

- GPU publishes cold jobs (expert IDs per layer) into the pinned job ring via
  `cuStreamWriteValue32`; the stager gathers weights (RAM tier) into pinned staging
  blocks and streams H2D; the GPU waits on per-job flags via `cuStreamWaitValue32`.
- No host in the hot loop, no kernel launch for synchronization, no full-device barrier.
- The stager's source is an interface from day 1: RAM tier now, NVMe tier (variant C) is
  a second backend of the same interface (decision 2026-09-02). "Compute on CPU"
  (variant B) stays a policy option behind the same interface, unbuilt.
- Per-layer policy field chooses the cold path by bandwidth, not hardcoded.

### 2.4 PLE (one layer, 128 tables of [2,500,012, 160])

- On disk NVFP4 (~28.8 GB), accessed as an **NVMe-mmap window** with a VRAM hot-row
  cache (default 1.0–2.0 GB, sized by hit-rate telemetry).
- Row indices computed **host-side per token** first (llama.cpp's proven
  `set_input` pattern — small H2D of indices); GPU-side indexing is a later refinement,
  not a stage gate.
- Row format NVFP4, block-exchangeable to FP8 (section 1); `ple_layer_ids: [2]` vs
  tensors on `layers.1` — the converter resolves the index once and records the
  resolution in the header.
- Row granularity on NVMe: a row is ~108 B NVFP4; reads happen at page granularity, so
  the cache design assumes read amplification (~37 rows per 4 KB page) and lets the
  hot-row cache absorb it — hit rate is measured, never assumed.

### 2.5 Three-state manager

- **KV**: 12 full-attention layers, 2 KV heads × head_dim 256, FP8 E4M3 default, BF16
  fallback via config; allocation at load for the configured context; resize only
  across full reloads (no mid-session shrink in stage 1).
- **GDN recurrent state**: 36 layers, f32 (16 K / 48 V heads at 128, conv kernel 4) —
  fixed size, context-independent.
- **QSA indexer cache**: budget 2,048 tokens, compress ratio 4 — third state kind,
  allocated alongside KV (`llama-memory-hybrid-idx` as the shape reference).
- Hyper-connections: `hc_norm` state is part of the weights, the 10240-wide residual
  stream is activation, not state.
- The manager owns all three kinds; stages never allocate around it (no leaks across
  stages, one accounting).

### 2.6 Acceptance

- Budget table verified by the loader on this machine (measured, not estimated, pool
  sizes); auto-clamp demonstrated by forcing N=192.
- Warm-up promotion demonstrated: coverage of the persisted sets vs the #3 curve.
- Cold-job path end to end on synthetic traffic before the first real decode.

---

## Section 3 — scheduler: job ring, routing-gated skip, cold-path policy (APPROVED by robin 2026-09-02)

### 3.1 The constraint the scheduler exists for

The borrowed engine's fixed term is 67.1 % synchronization at 62 CPU↔GPU handoffs per
token; a swapped barrier costs 4–6 ms (Crow #186). Binding design constraints: **no host
in the hot loop, no kernel launch for synchronization, no full-device barrier between
layers.** The decode loop per token: embeddings + PLE gather → 48 layers (36 GDN, 12
full attention, interleaved per `layer_types`) under hyper-connections → per MoE layer:
router on GPU → top-10 + shared expert → resident experts computed immediately → cold
experts via the job ring → `shared_expert_gate` combine.

### 3.2 Job ring (the handoff)

- Pinned host-memory ring, job descriptors 64-byte aligned (exl3 `moe_handoff.h`
  pattern): layer id, job kind, cold expert ids (≤ 10), sequence number, flag slots.
- The **descriptor is written by device-side mapped writes** from a small GPU kernel
  (the router result never leaves the GPU to produce it); publication of the flag goes
  through `cuStreamWriteValue32`. The stager thread busy-polls the pinned ring (no sync
  primitives), gathers cold expert weights from the RAM tier into pinned staging blocks,
  copies H2D on a dedicated copy stream, and raises the completion flag; the GPU's cold
  expert compute is enqueued behind `cuStreamWaitValue32` on that flag.
- **Overlap is the design goal, not alternation**: resident-expert compute and
  subsequent dense work proceed while the stager services cold jobs.
- Ring sizing: ≥ 256 slots (≤ 48 jobs/token worst case), sequence numbers against
  wraparound; a dead stager fails the engine loudly — no silent stall, ever.

### 3.3 Routing-gated skip

- Residency bitmaps (one per layer, from section 2.2's sets) live on the GPU; the skip
  decision is derived on-GPU from routing vs bitmap. A layer whose 10 routed experts are
  all resident publishes **no job**; a layer with k cold experts publishes exactly one
  job carrying those k ids.

### 3.4 Cold-path policy (AMENDED 2026-09-02, robin approved: zero-copy replaces stream-weights as primary)

Measurement history that drove the amendment: per-layer ring round trips measured
~3.1–3.4 ms on WDDM (28 cold layers/token → dead); speculative cross-token prefetch
measured 0.2–0.7 % prefetch gain (dead — the hot set already IS the temporal
structure); zero-copy direct read measured 21.6–25.1 GB/s (probe 3). HGS probed: no
benefit for driver-API handoffs (4.7 vs 3.1 ms).

- **Cold path A′ "zero-copy direct read" (primary)**: expert GEMM kernels take VRAM
  pointers for resident experts and PINNED HOST pointers for cold ones — the GPU pulls
  weights over PCIe inside the GEMM. No copy, no submission, no stall, no speculation.
  Cost: ~111 MB/token at ~23 GB/s ≈ 4.8 ms (N=160) / ~3.3 ms (N=176), spread across
  layers inside the GEMMs. Pinned cold tier ≈ 43–47 GB of 64 GB host RAM.
- **The job ring + stager remain for the control plane**: residency-set swaps,
  NVMe-tier staging (variant C), refresh, telemetry — off the decode critical path.
  Variant B (CPU compute) stays reserved; variant A (stream-weights) is retired from
  the primary path (kept behind the ring interface for tier transitions).
- Policy is still per layer, chosen at load, fixed per session.

### 3.5 PLE in the loop

- PLE row indices are computed host-side per token (section 2.4) — the one host touch
  per token outside the ring: pure index math, a few KB H2D, off the critical ring.

### 3.6 Acceptance

- Ring round-trip microbenchmark on Windows (resolution stated) against the 4–6 ms
  barrier cost from #186 — Windows FIRST (strand 2: exl3 memops unexercised on Windows).
- Synthetic 48-layer MoE pass: measured skips, cold jobs, bytes per token.
- A timestamped trace of a decode step shows **zero host wakes per layer** — the 62
  handoffs must collapse to ring jobs only.

---

## Section 4 — kernel path: sm_120a FP4 (APPROVED by robin 2026-09-02)

### 4.1 Target and probe-pinned facts

- Compile target **`compute_120a`** — plain `sm_120` is rejected by ptxas (probe 2);
  llama.cpp builds `120a-real` for the same reason. NVRTC at load time (probe 1/2
  chain), no nvcc dependency at runtime.
- Instruction:
  `mma.sync.aligned…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`.
- Fragment layouts as pinned by probe 2 (LSB-first nibbles at bit 4j in both operands;
  scale operand = one u32 per matrix, byte i = ue4m3 scale of the 16-wide k-sub-block;
  a/b/D geometry per RESULTS.md) — the engine's kernels reuse these layouts verbatim,
  and the probe's CPU reference (delta exactly 0 on dyadic inputs) is the numerics test.

### 4.2 Kernel families the model needs

1. **Expert GEMMs, fused**: `gate_up [512,1280,2560]`, `down [512,2560,640]`, batched
   over the ≤ 10 routed (+ 1 shared) per token. Our format keeps per-expert blocks
   contiguous, so the active set is directly addressable — no used-expert copy stage in
   the graph.
2. **Dense paths**: hidden 2560 ↔ residual stream 10240 (hyper-connections), attention
   QKVO, GDN projections (`in_proj_qkvz`, `out_proj`), `conv1d`, PLE gather + conv.
3. **Router**: BF16 GEMM 2560 × 512 (kept BF16 per section 1).
4. **Attention** (12 layers): 24 heads × head_dim 256, GQA 2 KV heads, partial rotary
   0.25, mrope interleaved [11,11,10] — compute stays BF16/FP8 (flash-attention style);
   FP4 is a weight format, not an attention math format. QSA indexer (budget 2048) as
   its own small kernel family.
5. **GDN** (36 layers): gated delta-net with sigmoid output gate — recurrent/chunked
   custom kernel family; reading templates: mistral.rs `qwen3_next.rs`, the transformers
   reference (oracle-side). SSM state f32.

### 4.3 Kernel hygiene

- Thin kernels (constant 4): one module per family, NVRTC-compiled at load, shared
  block-scaled-MMA core.
- Numerics gate in this order: kernel vs probe CPU reference → layer vs oracle (#6).
- No tok/s anywhere in kernel code or comments — measurement goes through the harness.

### 4.4 Ampere/Ada fallback

Own stage after this path works (decision 2026-09-02, amended): separate INT4/Q4 kernel
family without block-scale MMA, dispatch per architecture at load, own ticket and
timebox (created with spec section 6). Until then the engine runs Blackwell-only.

### 4.5 Acceptance

- All families pass numerics (probe reference, then oracle per layer) on this machine.
- `compute_120a` load verified on Windows (probe chain); Linux verification is part of
  the fallback stage's environment work, before any Linux number is quoted.

---

## Section 5 — correctness: oracle, parity gates, ten-task gate (APPROVED by robin 2026-09-02)

### 5.1 Oracles

- **Layer-wise oracle from the unquantized originals**: the transformers 5.8.x
  `qwen4_exp` reference, one layer at a time from the safetensors (one layer fits
  64 GB RAM; the model does not), no `trust_remote_code` needed (reference lives
  upstream). Early acceptance item: verify the reference's kernel imports
  (`fla`, `causal_conv1d`, triton) have CPU fallbacks so a layer runs oracle-side
  without a GPU path — strand 1 flagged this as unverified; it gates the oracle.
- **QSA-under-budget density oracle**: below `indexer_top_k + compress_ratio − 1`
  cached tokens, QSA is dense by construction — bit-identical on BF16/F32 (PR #27742:
  max logit delta 0.0 over 2,051 rows). A free correctness gate for the whole attention
  path.
- **`test-llama-archs` is not the oracle** (no PLE tensors, insensitive to the GDN QKV
  segmentation — plumbing only).
- **Quantization quality**: converter emits a per-tensor delta sidecar (section 1.5);
  FP8-KV vs BF16 and PLE-NVFP4 vs BF16 are oracle comparisons with pre-agreed fallbacks
  (sections 0.2, 1.2) — the converter never judges its own output.

### 5.2 Gates per stage

- **Parity gate**: every stage is measured against llama.cpp on this machine (#159:
  interleaved, same session, one variable, resolution stated) **and** against its own
  previous stage. Every number carries its operating point (section 0).
- **Ten-task gate**: before anything ships into Crow, the fixed ten-task series runs
  identically on both engines and the answers are compared for correctness (not speed).
  The concrete ten tasks are fixed from Crow's real workload before stage #11 starts;
  crow-lab's 2026-09-01-tasks series (t1–t6) is the seed set.

### 5.3 Honesty rules (binding)

Numbers only through the harness; no foreign tok/s transferred onto this machine; every
claim has an object (a file, a run, a commit); provenance is recorded like the 972 note
in section 0.3.

---

## Section 6 — stages and timeboxes (APPROVED by robin 2026-09-02 — boxes are guards, not deadlines)

### 6.1 Sequencing

Three parallel tracks at the start, then the long pole:

| track | stage (ticket) | proposed box | parallel with |
|---|---|---|---|
| A | #5 streaming converter (format, RTN stage 1) | 2 weeks | B, C |
| B | #6 oracle + parity harness | 2 weeks | A, C |
| C | #7 job ring + stream memops, Windows first | 1 week | A, B |
| D | #10 FP4 kernels (five families) | **5 weeks** | runs after C, alongside E, F |
| E | #8 residency scheduler + cold path | 3 weeks | after C, alongside D |
| F | #9 three-state memory manager | 2 weeks | after B, alongside D |
| G | #11 first end-to-end decode + standing parity series | 2 weeks | after D, E, F |

Critical path ≈ **12–14 weeks** to the first measured decode. The Ampere/Ada fallback
stage gets its ticket with this section's approval (section 4.4) and runs after #10.

### 6.2 The cut rule (constant 5)

A stage that misses its box is **re-cut** — split into smaller stages with a new box —
never terminated, and the project carries no end date. A box is a guard that forces the
re-cut conversation, nothing else.

### 6.3 Definition of done

Per stage: the acceptance section of its ticket, plus the parity gate (5.2) with the
operating points of section 0. "Done" is recorded on the ticket, board follows.

---

## Section 7: server and prefix cache (PROPOSED 2026-09-09, awaiting robin's approval. Sections 0-6 unchanged)

### 7.1 What the server is, and the one question this section answers

**Given:**

- The Crow client sends the **whole chat history on every turn**.
- Without a prefix cache every turn pays the full prefill again.
- The plan's reference point is a **16k prefill = 24.13 s wall (measured)**.
- Provenance: measured 2026-09-06 by chain F43 (t1-read 16,064 tokens, chunk 2048).
- Provenance: 24.19 s on 2026-09-09 by chain F49 on the installed build d211ab52ad2b.
- The logs live in the session scratchpad and the vault, not in this repository.
- The server ("serve") is a **blocking single-request binary** (robin's decision, not
  reopened here).
- **One engine process per machine** (robin's decision, not reopened here).

**Decided before the plumbing:**

1. Which of the four engine states survives a request boundary.
2. How a prefix divergence is detected.
3. What happens when one occurs.

**The failure mode this section exists to prevent:**

- Rule: a recurrent state must never be silently carried along.
- Reason: it produces answers that look right and come from the wrong state.
- Consequence: the A9 gate (ids bit-identical to a cold run, twice) is built against
  exactly that.
- Consequence: reaching A9 with a wrong design costs the whole stage.

### 7.2 What the engine does today (read from the code, 2026-09-09)

**No reset path:**

- The engine has **no reset path**.
- `Engine::pos`, `Engine::history` and `Engine::done_blocks` are initialised once at load
  (`gen.rs:774-776`).
- From then on they only grow: in `prefill` (`gen.rs:2696-2698`) and in `decode_step`
  (`gen.rs:2944-2947`).
- Every binary today builds a fresh `Engine` per process and runs exactly one prefill from
  position 0 (`bin/decode.rs:78-83`, `bin/decode.rs:150-153`).
- "Start over" is a process restart.
- A server has to introduce the concept of *setting the position back*.
- That concept is what the rest of this section defines.

**Two structural facts decide everything else.**

**Fact 1: absolute positions are baked into the states.**

- RoPE is applied from the `[context][32]` cos/sin tables at the absolute position before
  the KV store (`manager.rs:238-249`).
- RoPE for prefill: `gen.rs:1587-1588`.
- RoPE for decode: `gen.rs:1741-1744`.
- The KV row address is `slot = pos` (`manager.rs:305-311`, `kernels.rs:1190-1207`).
- The pooled QSA block index is `pos / 4` (`gen.rs:1647-1661`).
- Consequence: a cached state is reusable **as a prefix only**.
- Consequence: a fragment cannot be reused at a different offset.
- Consequence: there is no block-reuse / paged-attention story here without recomputing
  RoPE.

**Fact 2: the recurrent states carry no position at all.**

- `delta_rule_persist` (`kernels.rs:987-1012`) folds each token into `S` in place.
- `delta_rule_step` (`kernels.rs:1025-1048`) folds each token into `S` in place.
- `conv_state_update` (`kernels.rs:920-928`) shifts a 3-wide window.
- `conv_step` (`kernels.rs:1013-1024`) shifts a 3-wide window.
- The `init` flag that zeroes `S` is set only for the first chunk of the first prefill
  (`gen.rs:2433`, `gen.rs:2583`).
- Consequence: **a recurrent state cannot be rewound to an earlier position unless it was
  saved there.**

**A third fact, and an untested assumption.**

- The decode CUDA graph is captured **once per process**:
  `let capturing = graph && self.graph_exec == 0` (`gen.rs:2833`).
- It is instantiated at `gen.rs:2913`.
- Every token after that replays the instantiated graph.
- A server therefore runs its **second** prefill against an already-instantiated decode
  graph.
- That is a state **no binary has reached yet**, because every binary today prefills
  exactly once.
- By design the graph is position-agnostic: it reads its position from device scalars that
  the per-token scalar upload refreshes before each replay.
- `rope_p` deliberately takes `p.pos_base` as a device pointer rather than a host-computed
  table offset (`gen.rs:1733-1735`).
- Reason: a host offset would be baked into the capture and replay the capture's token
  position forever (`gen.rs:1733-1735`).
- That is the intent recorded in the code, **not a measurement**.
- **A2 and A9 must verify it.**

**Checks for A2 and A9:**

1. After a rollback and a second prefill, the replayed decode graph must still decode from
   the new position.
2. A9's bit-identity against a cold run is what would catch it if it does not.
3. If it does not hold, the remedy is to drop `graph_exec` on rollback and recapture.

### 7.3 The four states

| state | reusable across requests | prefix divergence detection | behaviour on divergence | cost of a cache miss at 16k |
|---|---|---|---|---|
| **KV cache**: 12 attention layers, `[12][2][2][context][256]`, FP8 E4M3 default (`geo.rs:23-25`, `manager.rs:43`, `manager.rs:200`) | **Yes, as it stands.** A row is addressed by the absolute slot `pos` and its RoPE was applied at that absolute position before the store, so rows `0..L` stay valid for any request whose ids agree on `0..L`. Nothing in the row depends on the request that wrote it. | Host-side only, from the id lists (7.4). The cache is never probed: there is no key in a KV row to compare against. | **Nothing is erased.** `pos` is set back to P. Rows `>= P` are stale but unreachable: the selector only scans blocks below `ncb = (pos+1)/4` (`gen.rs:2503-2506`, `gen.rs:2746`). The new suffix overwrites them as it is prefilled. | Full miss (P = 0) = one 16k prefill = **24.13 s wall** (measured, the plan's reference). A partial miss of n tokens is `n/16384 x 24.13 s` as a **linear estimate only** (unmeasured). Prefill is not linear in n: the chunk policy (`geo.rs:138-152`) and the one-cold-tier-pass-per-chunk cost (`gen.rs:2444-2448`) both bend it. |
| **QSA indexer state**: raw-key **ring** `[12][ring][128]` f32 with `row = pos % ring`, `ring = min(ceil4(prompt_chunk + 4), context)` (`manager.rs:37-41`, `manager.rs:58`, `kernels.rs:1759-1769`). Plus the **full-length pooled cache** `[12][ceil(context/4)][128]` f32 indexed by the absolute block `pos/4` (`manager.rs:46`, `manager.rs:60`, `kernels.rs:1747-1758`) | **Pooled cache: yes**, same argument as KV (absolute block index, RoPE at the absolute block position). **Ring: conditionally.** The ring is modular and holds only the last `ring` positions. Its only reader is `pool4_cache`, which for a resume at P needs the `P mod 4` rows of the still-incomplete block. Those are live iff the held run advanced fewer than `ring - 3` positions past P. Made unconditional by snapshotting the ring (7.6) or by rounding P down to a multiple of 4. | Host-side only (7.4). | Set `done_blocks = P/4` and restore the ring from the snapshot. Pooled blocks `>= P/4` are stale but unreachable by the same `ncb` bound. Block `P/4` is re-pooled by the resumed prefill **before** any query scores it (pooling precedes scoring inside `attn_prompt`: `gen.rs:1647-1661` then `gen.rs:1680-1683`). | No separate cost. The ring and the pooled blocks of the diverged suffix are rebuilt inside the same prefill pass that rebuilds KV. They add no pass of their own. Their share of the 24.13 s is **unmeasured** (`CROW_KPROF=1` would produce a per-kernel breakdown; none is recorded). |
| **GDN recurrent state**: 36 layers, `S[48][128][128]` f32 + `conv[10240][3]` f32, **112.22 MiB**, fixed and context-independent (`geo.rs:24`, `geo.rs:12-15`, `manager.rs:47-48`, `manager.rs:226-237`) | **No, not as the engine stands.** The state holds no position. It is the fold of every token seen so far. After the held run reached L there is no `S` at any P < L anywhere in the process, and there is no reset path. It is reusable **exactly at P = L**, and for any P < L **only from a snapshot taken at P** (7.6). | Host-side only, and this is the point: the state itself **cannot be probed**. Nothing in `S` says which ids produced it. If the id comparison is wrong, nothing downstream notices. | Restore `S` and `conv` from the newest snapshot at a position `S_pos <= L`, then re-prefill from `S_pos`. With **no** snapshot at or below L, the only correct move is a **cold start** (`S_pos = 0`): the KV and pooled rows that are still valid must be thrown away with it, because a KV prefix without the matching GDN state is precisely the silent-wrong-answer case. | **This is the state that sets the price.** The suffix to re-prefill starts at the last snapshot, not at the divergence point: extra cost = `(L - S_pos)` tokens of prefill on top of the diverged suffix. No snapshot at all = the full **24.13 s** at 16k. Its own share of a prefill is **unmeasured**. |
| **PLE row cache**: hot rows of the 128 n-gram shards, `n_slots = cache_bytes / 112`, default 128 MB = **1,198,372 slots** (`geo.rs:72`, `geo.rs:108`, `gen.rs:899-902`) | **Yes, unconditionally.** It is **content-addressed**, not position-addressed: `slot = ngram_row_id % n_slots` with `slot_map[slot]` holding the id (`gen.rs:1013-1040`), and a slot's content is a verbatim copy of a container row. It carries no position and does not depend on which request filled it. | **Not needed.** Divergence cannot invalidate it: a slot either already holds the row a token asks for, or is refilled from the container. | **Nothing.** The cache survives every divergence, every request, and every rollback. Rows filled by a discarded prefix stay useful. | A PLE miss is a container row read, **not** a prefill. It never forces recomputation. Not measured in seconds anywhere in the repo; the measured quantity is the **miss rate** (#16, 2026-09-05: 128 MB costs +0.2 % misses against 1 GB and frees ~7 hot-set units, `geo.rs:108`). |

**A fifth state hides inside the fourth row.**

- "PLE" in the plan means the *row cache*.
- The PLE layer also owns a **recurrent conv state** `Ple::state`, `[10240][9]` f32 =
  368,640 B (`gen.rs:902`).
- The conv is dilated (`src = t + k*3 - 9`, `kernels.rs:2839-2856`), so it needs nine
  history rows.
- `ple_state_update` (`kernels.rs:2857-2867`) refreshes it per prefill chunk.
- `ple_conv_step` (`kernels.rs:2868-2879`) shifts it per decode token.
- It behaves exactly like the GDN conv window and **must be snapshotted with it**.
- The #11 finding of 2026-09-05 (`gen.rs:2268-2273`) records what a wrong row in this path
  costs.
- Recorded: decode rows drifted 5-15 logit units from the prefill rows over the same
  context.
- Recorded: the same comparison with PLE off agreed within 1.8.

### 7.4 Detection: the longest common id prefix

**What the server holds:**

- Per cached conversation, the exact id sequence the engine consumed.
- That is `Engine::history` (`gen.rs:359`).
- It is appended in prefill (`gen.rs:2698`) and in decode (`gen.rs:2945`).
- Consequence: it covers prompt tokens **and** generated tokens.
- Consequence: it is exactly the transcript Crow will resend.

> **Detection rule.**
>
> - `L = ` length of the longest common prefix of the new request's ids and the held ids.
> - Compare ids, nothing else.
> - Never text, never a hash of the rendered prompt, never the tokenizer's input string.
> - The reuse point is `P = max { snapshot position S_pos : S_pos <= L }`.
> - The tokens `P..` of the new request are prefilled.

**Why ids and not text:**

- Rule: comparing ids and not text is not a stylistic choice.
- Reason: the states are built from ids, so ids are the only thing whose equality implies
  state equality.
- Consequence: two id lists that differ at position i produce different states from
  position i on, whatever the text looked like.

### 7.5 The invariant (binding)

> **The cache never changes the output.**
>
> - For any request, the ids produced with a warm cache are **bit-identical** to the ids
>   produced by a cold engine on the same request.

**What the invariant is:**

- This is the A9 gate: bit-identical ids against a cold run, **twice**.
- Twice, because once is not evidence, per the engine's own race-hunting rule.
- The invariant is what makes every "behaviour on divergence" cell above conservative.
- Rule: where the correct state cannot be proven present, the answer is *recompute*, never
  *carry on*.

**Two conditions the invariant depends on, both outside the four states.**

**Condition 1: greedy only, or a reseeded RNG.**

- The device sampler's xorshift state (`DevSampler::rng`, `gen.rs:2955-2969`) advances per
  token.
- It lives for the engine's lifetime.
- Consequence: under sampling, a second request on a warm engine draws from a different
  RNG position than a cold engine would.
- Consequence: "bit-identical ids" is only a meaningful gate at greedy/argmax, or with the
  sampler reseeded per request.
- Rule: A9 must run greedy.

**Condition 2: residency stays numerically invisible.**

- A cold expert is read zero-copy through the pointer table (`residency.rs:1-16`).
- Consequence: hot-set adaptation between requests changes *where* an expert is read from,
  not *what* is read (`residency.rs:1-16`).
- This holds for the default tier only.
- `CROW_COLD_TIER` (`residency.rs:231`) installs a **low-bit** cold tier.
- The low-bit cold tier is lossy and would break bit-identity between two runs with
  different hot sets.
- Rule: the server must not enable it while A9 is the gate.

### 7.6 Snapshot and rollback (GDN, PLE conv, QSA ring)

**What needs no snapshot:**

- Rule: KV and the pooled QSA cache need **no** snapshot.
- Reason: they are absolutely addressed, append-only, and a stale row past `pos` is never
  read (7.3).
- Consequence: only the states that fold history into a fixed-size buffer do.

**Snapshot = a device-to-host copy of:**

1. the 36 GDN `S` buffers,
2. the 36 GDN `conv` buffers,
3. the PLE conv state,
4. the QSA raw-key ring,
5. plus the host-side triple `(pos, done_blocks, history[..pos])` that names the position
   it belongs to.

**Why the ring is in the snapshot:**

- Rule: the ring is in the snapshot so that the reuse point may be **any** position, not
  only a multiple of 4.
- Consequence without it: P must be rounded down to `4 * floor(P/4)` and the ring's live
  window checked.

**Do not confuse the ring with the selection budget:**

- The `ring` here is the **raw-key ring**.
- It has nothing to do with the QSA **selection budget**.
- The "budget 2,048 tokens" of §2.1 is `QSA_BLOCK_TOPK` 512 selected blocks x compression
  ratio 4 = 2,048 tokens a *query* may attend to (`geo.rs:41-42`).
- The ring is a `ceil4(prompt_chunk + 4)`-row scratch buffer of raw indexer keys awaiting
  pooling.
- The two numbers are unrelated and only happen to sit close together at chunk 2048.

**When to snapshot.** Two points per turn, both natural and both cheap relative to a
prefill:

1. after the prefill of a turn's prompt (`pos = prompt length`), and
2. after that turn's generated answer (`pos = end of turn`).

**Point 2 is the normal case:**

- Point 2 is the one Crow normally hits.
- The next turn's ids extend the previous turn's transcript.
- Consequence: `L = ` the previous end and `P = L`.
- Consequence: no rollback, no recompute.
- Consequence: the new prompt suffix is simply prefilled onto the held state.

**Point 1 earns its keep in the regenerate / edited-answer case:**

- The prompt's ids are unchanged and the divergence sits in the *answer*.
- Consequence: `L >= ` the end of that prompt.
- Consequence: the point-1 snapshot is the newest one at or below `L`, and the rollback
  lands on it.
- Consequence: **the prompt's prefill is spared** and only the answer is recomputed.

**Point 1 does not cover the edited-prompt case:**

- If the user rewrites their last message, the common prefix ends at the **start** of that
  prompt.
- That is below the point-1 snapshot, which is taken at prompt *end*.
- Consequence: `P` is the previous turn's post-answer snapshot, still a warm resume, not a
  cold start.
- A cold start only happens when **no** snapshot at or below `L` exists at all.
- With any earlier snapshot held, the fallback is never cold.
- The rule is unchanged in both cases and needs no special-casing:
  `P = max { S_pos : S_pos <= L }`.

**Rollback:**

1. Restore the four buffers with host-to-device copies.
2. Set `pos = S_pos`.
3. Set `done_blocks = S_pos / 4`.
4. Truncate `history` to `S_pos`.
5. Call `prefill` with the new ids from `S_pos` on.

- Note: `prefill`'s `init` flag zeroes `S` only when `self.pos == 0` (`gen.rs:2433`).
- That is exactly the cold-start case.
- Consequence: the restored path must leave it at 0.
- The current signature already does the right thing once `pos` is set.

**What must NOT be reused:**

- Rule: never reuse a snapshot from a different engine load.
- Reason: the chunk size fixes the ring rows, the scratch, and the clamped hot-set N
  (`manager.rs:31`, `manager.rs:108-170`).
- Consequence: a snapshot's shape is only valid for the process that produced it.
- Consequence: snapshots are in-process state, not a file format.

### 7.7 Memory cost of a snapshot

- All four buffers are exactly sized by the geometry constants.
- Nothing here is estimated.

```
snapshot_bytes =
    GDN_LAYERS * GDN_VHEADS * GD * GD * 4   (GDN S)     = 36 * 48 * 128 * 128 * 4 = 113,246,208 B = 108.00 MiB
  + GDN_LAYERS * GDN_CONV   * 3      * 4   (GDN conv)  = 36 * 10240 * 3 * 4       =   4,423,680 B =   4.22 MiB
  + GDN_CONV   * 9          * 4            (PLE conv)  = 10240 * 9 * 4            =     368,640 B =   0.35 MiB
  + ATTN_LAYERS * ring * QSA_HIDD * 4      (QSA ring)  = 12 * ring * 128 * 4
```

with `ring = ceil4(prompt_chunk + 4).min(context)` (`manager.rs:37-41`):

| prompt_chunk | ring | QSA ring bytes | snapshot total |
|---|---|---|---|
| 512 (default `Config`) | 516 | 3,170,304 B = 3.02 MiB | **121,208,832 B = 115.60 MiB** |
| 2048 (long-prompt policy) | 2052 | 12,607,488 B = 12.02 MiB | **130,646,016 B = 124.60 MiB** |

**Totals for two snapshots per held conversation (7.6):**

- **231.19 MiB** at chunk 512.
- **249.19 MiB** at chunk 2048.

**Where snapshots live:**

- Rule: snapshots belong in **pageable host RAM**.
- Not VRAM: VRAM is already the binding budget, and the loader clamps N against it
  (`manager.rs:122-170`).
- Not the pinned tier: it is budgeted at 46 GiB against a measured ~48.5 GiB host ceiling
  (`geo.rs:110`).

**Unmeasured:**

- The **wall time of one snapshot copy is unmeasured**.
- It is the first thing to measure in the plumbing task.
- 115.60 MiB down and up per rollback is small against 24.13 s, but it sits inside the
  request.

**The formula holds for the default QSA layout only.**

- `CROW_QSA_FULL=1` (`manager.rs:37-41`) sets `ring = context`.
- That is the full-length raw-key layout from before 2026-09-05 (`manager.rs:20-21`),
  which is still supported.
- The ring term is then `12 * 262,144 * 128 * 4 = 1,610,612,736 B` (about **1.5 GiB**) at
  the default context of 262,144 (`geo.rs:105`).
- That is roughly **13x the whole snapshot**, and it would dominate every number in this
  section.
- Rule: `CROW_QSA_FULL=1` is therefore **out of scope for serve**.
- Consequence: the server runs the default layout, and every size above assumes it.

### 7.8 What a cache miss actually costs

- One number is measured and carries the section: **a full 16k prefill = 24.13 s wall**.
- It is also the honest ceiling for every row of the table.
- Reason: the four states are produced by **one interleaved 48-layer pass**.
- Reason: there is no way to rebuild KV without also rebuilding the GDN fold.
- Reason: there is no way to rebuild the GDN fold cheaply on its own.

Therefore:

- the cost of a miss is the cost of **re-prefilling from `P` to the end of the new
  prompt**, once, for all four states together;
- the **per-state fraction is unmeasured** and is not invented here. `CROW_KPROF=1`
  (`gen.rs:58-83`) produces a per-kernel breakdown and would settle it;
- the practical lever is **P**, and P is set by the GDN snapshot policy, not by KV. A
  design that caches KV and forgets GDN has a cache that is always cold.

### 7.9 Acceptance

- Section 7 approved by robin, as sections 0-6 were on 2026-09-02.
- The A9 gate stands as written: **ids bit-identical to a cold run, twice**, greedy, with
  the default cold tier.
- The first plumbing task measures, and records with its operating point:
  1. snapshot copy time (DtoH and HtoD),
  2. the warm-turn time for a 16k transcript whose next turn appends a short prompt,
  3. the rollback time when the divergence lands inside the last turn.

### 7.10 Open for robin (M1)

1. **The prefill chunk must be pinned at load.**
   - `apply_chunk_policy` runs *before* `Engine::load` and sizes the chunk from the one
     prompt it is about to serve (`bin/decode.rs:147`).
   - The chunk then fixes the QSA ring rows, the scratch, and the clamped N
     (`manager.rs:31`, `manager.rs:108-170`).
   - A server takes prompts of every length against one load, so it must pick **one** chunk
     for the process lifetime.
   - Candidate: chunk 2048 (the long-context operating point, N ~147, stream trickle),
     `geo.rs:130-176`.
   - Candidate: chunk 512 (N ~157, better on short prompts), `geo.rs:130-176`.
   - Note: the source disagrees with itself about the chunk-2048 hot set.
   - `geo.rs:157` says **N 147**; `geo.rs:132` says **140** for the same operating point.
   - Neither is picked here: settling it is a measurement, not a doc edit.
   - **Question: which one is the serve default?**
2. **How many conversations are held, and how many snapshots each?**
   - At 115.60 MiB per snapshot and 2 per conversation, four held conversations is ~0.9 GiB
     of host RAM.
   - That is against the 47 GiB `free_wait` threshold the chains already run into.
3. **Is the post-answer snapshot (point 2 in 7.6) taken unconditionally**, i.e. does every
   answer pay one DtoH of 115.60 MiB even when the conversation is never continued?
4. **Concurrency.**
   - Blocking single-request means a second request waits.
   - Queue or reject?
5. **Sampling.**
   - A9 runs greedy (7.5).
   - Is sampling out of scope for stage A, or does the server reseed `DevSampler::rng`
     per request so a warm engine and a cold engine agree?
