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

- Amended 2026-09-11 (robin, chat, recorded on issue #1): an end state below llama.cpp on this
  machine is not acceptable; the performance stage (#37, #38, a per-token kernel profile against
  llama.cpp, #19, #10) runs before the quant upload (F4 to F6), and the comparison is quoted on one
  prompt with both quantizations named (CNQ4.5-M 4.5 bpw against UD-Q2_K_XL 2.4 bpw).

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
- **Decode staging as built (`CROW_STAGE`, default on)**: after `router_top10` the
  staging kernel pulls every COLD combo of the layer into a VRAM staging slot and
  rewrites the combo pointer tables; hot combos keep their slab pointer. The kernel is
  `stage_cold_ca` since #19e (2026-09-12); `CROW_STAGE_KERNEL=1` restores `stage_cold`,
  which reads with coalesced 16-byte loads and measured 10.248 ms/token at 33.0 GB/s over
  338 MB/token on 2026-09-11 (#19a).
- **`CROW_STAGE_DMA=1` (measurement only, #19b, default off)**: the same staging done by
  the COPY ENGINE, one `cuMemcpyDtoDAsync` per cold combo and matrix issued from the host
  from the mapped pinned pointer. The routed pointers exist on the host only after
  `router_top10` of that layer, so the switch forces the decode graph off and costs one
  host sync per layer. Not an operating default.
- **`stage_cold_ca` (the DEFAULT since #19e, 2026-09-12; `CROW_STAGE_KERNEL=1` falls
  back)**: the same staging done by `stage_cold_ca` (`kernels.rs:2263`), a PERSISTENT
  grid of `CROW_STAGE_BLOCKS` x 256 threads (default 40) launched from `gen.rs:2051`.
  Work item = (matrix, combo); the owning block `item % gridDim.x` rewrites the pointer
  table entry, and the 4 KB tiles of a cold item are split over all blocks, each tile
  read with `cp.async.cg.shared.global` 16 B per thread into a two-deep shared double
  buffer and stored coalesced to VRAM. Same inputs plus the combo count `t * TOPK` by
  value, same outputs, still inside the decode graph.
- **Requirement of `stage_cold_ca`**: both staged slab byte counts must be exact
  multiples of 4096, because the kernel carries no tail tile. `gen.rs:683` asserts it at
  load and `gen.rs:2048` again at the launch site; both panic messages name
  `CROW_STAGE_KERNEL=1` as the fallback. This container: `gate_up` 1843200 B = 450 tiles,
  `down` 921600 B = 225 tiles.
- **Boot line**: every engine process prints one `[stage]` line naming the kernel it will
  run, its grid and the slab byte counts (`gen.rs:697` for `stage_cold_ca`, `gen.rs:700`
  for `stage_cold`), so every log says which kernel produced it.

| Switch | Kernel | Grid | Read shape | Mode |
|---|---|---|---|---|
| unset (default) | `stage_cold_ca` | `CROW_STAGE_BLOCKS` x 1 x 1, 256 threads | `cp.async.cg.shared.global` 16 B per thread into 4 KB shared tiles, 2-deep, slab bytes must be 4 KB multiples | operating |
| `CROW_STAGE_KERNEL=1` | `stage_cold` | `t * TOPK` x 2 x `CROW_STAGE_SPLIT`, 256 threads | 4 x 16 B `uint4` loads in flight per thread | operating fallback |
| `CROW_STAGE_DMA=1` | none, copy engine | host-issued `cuMemcpyDtoDAsync` per cold combo | copy engine, decode graph off | measurement |

- The decode operating point that decided the default: #19e, 2026-09-12.
- The two arms run different weights: crow-nest CNQ4.5-M (NVFP4, 4.5 bpw); llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL (GGUF, 2.4 bpw).
- A tok/s figure is quoted only next to its adjacent arm in the same chain (#38).

| shape metric | crow-nest, default `stage_cold_ca` | crow-nest, `CROW_STAGE_KERNEL=1` | llama.cpp | machine | date | source |
|---|---|---|---|---|---|---|
| decode, t1-read 16,064 ids, 255 timed steps | 26.46 ms per token = 37.8 tok/s | 29.68 ms per token = 33.7 tok/s | 22.27 ms per token = 44.9 tok/s | RTX 5090 | 2026-09-11 | `decode_out/srv-19d.log`, `decode_out/srv-59b.log` |
| staging row of that step, nsys, 338 MB per token | 7.07 ms per token at 47.78 GB/s | 10.42 ms per token at 32.45 GB/s | n/a | RTX 5090 | 2026-09-11 | `decode_out/srv-19d.log` |

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

## Section 7: server and prefix cache (approved by robin 2026-09-09 at M1. Sections 0-6 unchanged)

**Status of this section:**

| item | value |
|---|---|
| proposed | 2026-09-09, task A1, issue #23 |
| approved | 2026-09-09 by robin, M1 decision comment on issue #1 |
| corrected against the built server | 2026-09-10, task A11, issue #33 |
| evidence for the corrections | closing comments of #24 to #32, gate logs `decode_out/srv-a*.log` |
| corrected rows | 7.2 graph assumption, 7.3 KV row claim, 7.6 point 2 and the never-cold claim, 7.7 to 7.9 numbers |
| added after the build | 7.11 endpoint contract as built, 7.12 stage A gate table |
| sections 0 to 6 | unchanged, approved 2026-09-02 |

### 7.1 What the server is, and the one question this section answers

**Given:**

- The Crow client sends the **whole chat history on every turn** (`crow_core.py:4672-4700`).
- Without a prefix cache every turn pays the full prefill again.
- The plan's reference point is a **16k prefill = 24.13 s wall (measured)**.
- Provenance: measured 2026-09-06 by chain F43 (t1-read 16,064 tokens, chunk 2048).
- Provenance: 24.19 s on 2026-09-09 by chain F49 on the installed build d211ab52ad2b.
- The logs live in the session scratchpad and the vault, not in this repository.
- The server ("serve") is a **blocking single-request binary** (robin's decision, not
  reopened here).
- **One engine process per machine** (robin's decision, not reopened here).

**Measured on the serve build of 2026-09-10 (A9, #31, `decode_out/srv-a9.log`):**

| quantity | value | note |
|---|---|---|
| cold 16k prefill, turn 1, 16,064 ids | **21.6 to 22.0 s** | the same work the 24.13 s reference names, on the serve binary |
| warm turn prefill, 95 of 16,159 ids | **404 ms** | 99.41 % of the prompt reused |
| rollback into the last turn (HtoD) | **11.98 to 13.12 ms** | 7.9 acceptance point 3 |

- Rule: 24.13 s stays the plan's reference number, with the provenance above.
- Rule: 21.6 to 22.0 s is the number of THIS build and is the one 7.8 costs are read against.
- Neither number is a correction of the other: different binary, same operating point.

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
  (`gen.rs:788-790`).
- From then on they only grow: in `prefill` (`gen.rs:2804-2806`) and in `decode_step`
  (`gen.rs:3052-3055`).
- Every binary before serve built a fresh `Engine` per process and ran exactly one prefill
  from position 0 (`bin/decode.rs:78-83`, `bin/decode.rs:150-153`).
- "Start over" was a process restart.
- A server has to introduce the concept of *setting the position back*.
- Built as `Engine::reset_to_zero` (`engine/src/reset.rs`, #26 A4) and, warm, as
  `PrefixCache::rollback` (`engine/src/cache.rs:452`, #31 A9).

**Two structural facts decide everything else.**

**Fact 1: absolute positions are baked into the states.**

- RoPE is applied from the `[context][32]` cos/sin tables at the absolute position before
  the KV store (`manager.rs:238-249`).
- RoPE for prefill: `gen.rs:1608-1609`.
- RoPE for decode: `gen.rs:1762-1765`.
- The KV row address is `slot = pos` (`manager.rs:305-311`, `kernels.rs:1190-1207`).
- The pooled QSA block index is `pos / 4` (`gen.rs:1668-1682`).
- Consequence: a cached state is reusable **as a prefix only**.
- Consequence: a fragment cannot be reused at a different offset.
- Consequence: there is no block-reuse / paged-attention story here without recomputing
  RoPE.

**Fact 2: the recurrent states carry no position at all.**

- `delta_rule_persist` (`kernels.rs:987-1012`) folds each token into `S` in place.
- `delta_rule_step` (`kernels.rs:1025-1045`) folds each token into `S` in place.
- `conv_state_update` (`kernels.rs:920-928`) shifts a 3-wide window.
- `conv_step` (`kernels.rs:1013-1024`) shifts a 3-wide window.
- The `init` flag that zeroes `S` is set only for the first chunk of the first prefill
  (`gen.rs:2541`, `gen.rs:2691`).
- Consequence: **a recurrent state cannot be rewound to an earlier position unless it was
  saved there.**

**A third fact, and the assumption A4 measured FALSE.**

- The decode CUDA graph is captured **once per process**:
  `let capturing = graph && self.graph_exec == 0` (`gen.rs:2941`).
- It is instantiated at `gen.rs:3021`.
- Every token after that replays the instantiated graph.
- A server therefore runs its **second** prefill against an already-instantiated decode
  graph.
- The proposal assumed that graph is position-agnostic, because `rope_p` takes
  `p.pos_base` as a device pointer rather than a host-computed table offset
  (`gen.rs:1754-1756`).
- **Measured 2026-09-09 (A4, issue #26 (comment), `reset.rs` module doc): the assumption does not hold.**

| step | what the code does | evidence |
|---|---|---|
| 1 | `decode_step` creates the capture stream ONCE and leaves it ACTIVE | `gen.rs:2848-2851` |
| 2 | `launch_v` and `upload_into` both read that active stream | `gen.rs:1215`, `cuda.rs:377` (`cur_stream`) |
| 3 | `upload_into` skips its sync on any stream but the legacy one | `cuda.rs:384-386` |
| 4 | `prefill` uploads its per chunk scalars and the embedding block from TEMPORARIES | `gen.rs:2601-2618`, `gen.rs:2625-2634` |
| 5 | so a second `prefill` posted async HtoD copies whose host source had already died | measured |
| result | same prompt, greedy: request 1 gave id 18622, request 2 gave id 17 | issue #26 (comment) |

**The remedy, exactly the one this section named as the fallback:**

- `Engine::drop_decode_graph` (`engine/src/reset.rs`) destroys `graph_exec`, destroys
  `cap_stream` and puts the legacy stream (0) back, mirroring `impl Drop for Engine`
  (`gen.rs:3417-3427`).
- It is the ONE definition of that teardown.
- The cold path `Engine::reset_to_zero` calls it before any prefill.
- The warm path `PrefixCache::rollback` (`cache.rs:452`) calls it before any prefill.
- `slot::restore` (`slot.rs:577`) calls it before its uploads.
- `decode_step` then re-creates the stream and re-captures on the first step of the next
  request (`gen.rs:2848`, `gen.rs:2944`).
- Rule: **the decode graph is dropped and recaptured per request.** It is not carried.
- Cost: one eager decode step plus one graph instantiate per request.
- Not measured against keeping the graph: keeping it is what broke the ids.
- A4 gate, 18 token prompt, 29 generated: decode **1154.2 to 1213.2 ms** over the seven
  logged runs (`decode_out/srv-a4.stderr.log:199`, `:207`, `:215`,
  `decode_out/srv-a4-fresh.stderr.log:143`, `:158`,
  `decode_out/srv-a4-fix.stderr.log:146`, `:154`).

### 7.3 The four states

| state | reusable across requests | prefix divergence detection | behaviour on divergence | cost of a cache miss at 16k |
|---|---|---|---|---|
| **KV cache**: 12 attention layers, `[12][2][2][context][256]`, FP8 E4M3 default (`geo.rs:23-25`, `manager.rs:43`, `manager.rs:200`) | **Yes, for rows PREFILL wrote.** A row is addressed by the absolute slot `pos` and its RoPE was applied at that absolute position before the store, so rows `0..L` stay valid for any request whose ids agree on `0..L`. **Corrected 2026-09-10 (A9, #31): a row DOES depend on the code path that wrote it.** `decode_step` rows are not bit-equal to `prefill` rows at the same position. A reuse point is valid only while every row below it was written by `prefill` (PREFILL CLEAN, `engine/src/cache.rs`). | Host-side only, from the id lists (7.4). The cache is never probed: there is no key in a KV row to compare against. **Plus the prefill-clean flag per slot**, which is state the server keeps, not something read out of a row. | **Nothing is erased.** `pos` is set back to P. Rows `>= P` are stale but unreachable: the selector only scans blocks below `ncb = (pos+1)/4` (`gen.rs:2611-2614`, `gen.rs:2854`). The new suffix overwrites them as it is prefilled. | Full miss (P = 0) = one 16k prefill = **21.6 to 22.0 s** on the serve build (A9, #31); **24.13 s** is the plan's reference on the installed decode build. A partial miss of n tokens is `n/16384 x` that as a **linear estimate only** (unmeasured). Prefill is not linear in n: the chunk policy (`geo.rs:138-152`) and the one-cold-tier-pass-per-chunk cost (`gen.rs:2552-2556`) both bend it. |
| **QSA indexer state**: raw-key **ring** `[12][ring][128]` f32 with `row = pos % ring`, `ring = min(ceil4(prompt_chunk + 4), context)` (`manager.rs:37-41`, `manager.rs:58`, `kernels.rs:1759-1769`). Plus the **full-length pooled cache** `[12][ceil(context/4)][128]` f32 indexed by the absolute block `pos/4` (`manager.rs:46`, `manager.rs:60`, `kernels.rs:1747-1758`) | **Pooled cache: yes, under the same prefill-clean rule as KV** (A9, #31: a pooled block over decode-written rows is not bit-equal either). **Ring: conditionally.** The ring is modular and holds only the last `ring` positions. Its only reader is `pool4_cache`, which for a resume at P needs the `P mod 4` rows of the still-incomplete block. Those are live iff the held run advanced fewer than `ring - 3` positions past P. Made unconditional by snapshotting the ring (7.6) or by rounding P down to a multiple of 4. | Host-side only (7.4). | Set `done_blocks = P/4` and restore the ring from the snapshot. Pooled blocks `>= P/4` are stale but unreachable by the same `ncb` bound. Block `P/4` is re-pooled by the resumed prefill **before** any query scores it (pooling precedes scoring inside `attn_prompt`: `gen.rs:1668-1682` then `gen.rs:1701-1704`). | No separate cost. The ring and the pooled blocks of the diverged suffix are rebuilt inside the same prefill pass that rebuilds KV. They add no pass of their own. Their share of the prefill is **unmeasured** (`CROW_KPROF=1` would produce a per-kernel breakdown; none is recorded). |
| **GDN recurrent state**: 36 layers, `S[48][128][128]` f32 + `conv[10240][3]` f32, **112.22 MiB**, fixed and context-independent (`geo.rs:24`, `geo.rs:12-15`, `manager.rs:47-48`, `manager.rs:226-237`) | **No, not without a snapshot.** The state holds no position. It is the fold of every token seen so far. After the held run reached L there is no `S` at any P < L anywhere in the process. It is reusable **exactly at P = L**, and for any P < L **only from a snapshot taken at P** (7.6). | Host-side only, and this is the point: the state itself **cannot be probed**. Nothing in `S` says which ids produced it. If the id comparison is wrong, nothing downstream notices. | Restore `S` and `conv` from the newest prefill-clean snapshot at a position `S_pos <= L`, then re-prefill from `S_pos`. With **no** such snapshot, the only correct move is a **cold start** (`S_pos = 0`): the KV and pooled rows that are still valid must be thrown away with it, because a KV prefix without the matching GDN state is precisely the silent-wrong-answer case. | **This is the state that sets the price.** The suffix to re-prefill starts at the last snapshot, not at the divergence point: extra cost = `(L - S_pos)` tokens of prefill on top of the diverged suffix. No snapshot at all = the full **21.6 to 22.0 s** at 16k on this build. Its own share of a prefill is **unmeasured**. |
| **PLE row cache**: hot rows of the 128 n-gram shards, `n_slots = cache_bytes / 112`, default 128 MB = **1,198,372 slots** (`geo.rs:72`, `geo.rs:108`, `gen.rs:913-916`) | **Yes, unconditionally.** It is **content-addressed**, not position-addressed: `slot = ngram_row_id % n_slots` with `slot_map[slot]` holding the id (`gen.rs:1027-1054`), and a slot's content is a verbatim copy of a container row. It carries no position and does not depend on which request filled it. | **Not needed.** Divergence cannot invalidate it: a slot either already holds the row a token asks for, or is refilled from the container. | **Nothing.** The cache survives every divergence, every request, and every rollback. Rows filled by a discarded prefix stay useful. | A PLE miss is a container row read, **not** a prefill. It never forces recomputation. Not measured in seconds anywhere in the repo; the measured quantity is the **miss rate** (#16, 2026-09-05: 128 MB costs +0.2 % misses against 1 GB and frees ~7 hot-set units, `geo.rs:108`). |

**The prefill-clean rule (A9, #31, measured 2026-09-10, binding):**

| run | reuse point | rows below it written by | ids vs a fresh process |
|---|---|---|---|
| A9 gate part 2, turn 2 | 16127 (after the answer) | prefill AND `decode_step` | equal over 35 ids, 5 runs |
| A9 gate part 3, turn 3 | 16159 (after a prompt) | prefill AND `decode_step` | DIFFERENT, first at generated id 43 |
| A9 probe, turn 4 | 16064 (after a prompt) | prefill only | equal over 49 ids |
| A9 control, turn 3 | none, cold, chunk cut changed | prefill only | equal, so the chunk cut is not the cause |

- Evidence: `decode_out/srv-a9.log` and its appendix, `-control.log`, `-chunkcut.log`, `-probe.log`.
- Rule: a snapshot is a reuse candidate only while every row below its position was written
  by `prefill`. That property is called **PREFILL CLEAN**.
- Reason: 7.5 is binding, and a decode-written row cannot be PROVEN to be the row a cold run
  would have.
- Consequence: the after-answer snapshot was never offered to `reuse_slot`, so M2b DROPPED
  it (robin 2026-09-10, #36; `cache.rs:336`, `cache.rs:182`, 7.6).
- Induction that point 1 is always prefill clean: a cold request prefills `0..prompt_len`; a
  warm request rolls back only to a prefill-clean `P` and prefills `P..prompt_len`, which
  REWRITES the re-rendered previous answer as prefill rows.

**KV and the snapshot, within a process and across a restart:**

| scope | what carries KV | evidence |
|---|---|---|
| within one process | nothing: KV rows and pooled blocks stay in VRAM, absolutely addressed, and rows `>= P` are unreachable | 7.6, `cache.rs` module doc |
| across a restart | the slot file: KV rows `0..pos` and pooled blocks `0..floor(pos/4)` | `engine/src/slot.rs` module doc, #32 A10 |

- Rule: the pooled range is `floor(pos/4)`, not `ceil`.
- Reason: block `ceil(pos/4)` is written only by `decode_step` at that position
  (`gen.rs:1668`, `gen.rs:2804`), so a saved `ceil` block would be a decode row.
- A restore refuses `done_blocks != pos / 4` before the first device upload (`slot.rs`).

**A fifth state hides inside the fourth row.**

- "PLE" in the plan means the *row cache*.
- The PLE layer also owns a **recurrent conv state** `Ple::state`, `[10240][9]` f32 =
  368,640 B (`gen.rs:916`).
- The conv is dilated (`src = t + k*3 - 9`, `kernels.rs:2839-2856`), so it needs nine
  history rows.
- `ple_state_update` (`kernels.rs:2857-2867`) refreshes it per prefill chunk.
- `ple_conv_step` (`kernels.rs:2868-2879`) shifts it per decode token.
- It behaves exactly like the GDN conv window and **must be snapshotted with it**.
- The #11 finding of 2026-09-05 (`gen.rs:2376-2381`) records what a wrong row in this path
  costs.
- Recorded: decode rows drifted 5-15 logit units from the prefill rows over the same
  context.
- Recorded: the same comparison with PLE off agreed within 1.8.

### 7.4 Detection: the longest common id prefix

**What the server holds:**

- Per cached conversation, the exact id sequence the engine consumed.
- That is `Engine::history` (`gen.rs:359`).
- It is appended in prefill (`gen.rs:2806`) and in decode (`gen.rs:3053`).
- Consequence: it covers prompt tokens **and** generated tokens.
- Consequence: it is exactly the transcript Crow will resend.

> **Detection rule.**
>
> - `L = ` length of the longest common prefix of the new request's ids and the held ids.
> - Compare ids, nothing else.
> - Never text, never a hash of the rendered prompt, never the tokenizer's input string.
> - The reuse point is `P = max { snapshot position S_pos : S_pos <= L }`.
> - The tokens `P..` of the new request are prefilled.

**As built (`engine/src/cache.rs`, #31 A9):**

| symbol | implementation | anchor |
|---|---|---|
| `L` | `common_prefix_len(history, ids)` | `cache.rs:175` |
| `P` | `reuse_slot(reuse_candidates, l, new_len)` | `cache.rs:182`, `cache.rs:336` |
| candidates | only PREFILL CLEAN slots (7.3) | `cache.rs:336` |
| extra guard | `S_pos < request length` | `cache.rs:182` |
| decision | `PrefixCache::decide` | `cache.rs:345` |

- Reason for the extra guard: `prefill` of an empty slice has no last position to return a
  greedy id from. It is a guard, not a change of the rule.

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
- Measured 2026-09-10: this rule is what removed the after-answer snapshot from the reuse
  candidates (7.3, A9 #31).

**Two conditions the invariant depends on, both outside the four states.**

**Condition 1: greedy only, or a reseeded RNG. BOTH hold as built (A6, #28).**

- The device sampler's xorshift state (`DevSampler::rng`, `gen.rs:3063-3077`) advances per
  token.
- It lives for the engine's lifetime.
- Consequence: under sampling, a second request on a warm engine would draw from a
  different RNG position than a cold engine.
- Built: `Engine::enable_dev_sampler` runs for EVERY sampled request and uploads
  `Rng::new(seed)`, so request k starts cold (M1 decision, robin 2026-09-09).
- Built: a greedy request PARKS the sampler in `Srv::parked_sampler`, so it cannot sample
  silently (`serve.rs:2065` the field, `serve.rs:1746` park, `serve.rs:1729` hand back).
- Measured: seed 7 warm equals seed 7 cold, 2 of 2; seed 8 differs; greedy between two
  sampled requests still identical (`decode_out/srv-a6.log`, `decode_out/srv-a6-fix.log`).
- Rule: A9 runs greedy (M1 decision).

**Condition 2: residency stays numerically invisible.**

- A cold expert is read zero-copy through the pointer table (`residency.rs:1-16`).
- Consequence: hot-set adaptation between requests changes *where* an expert is read from,
  not *what* is read (`residency.rs:1-16`).
- This holds for the default tier only.
- `CROW_COLD_TIER` (`residency.rs:231`) installs a **low-bit** cold tier.
- The low-bit cold tier is lossy and would break bit-identity between two runs with
  different hot sets.
- Rule: the server must not enable it while A9 is the gate.
- As built (#37): `serve` TICKS the stream trickle once per `decode_step`
  (`bin/serve.rs:1841-1843`), the mirror of `bin/decode.rs:224-231`.
- Condition 2 is what allows it: the tick moves where an expert is read from, not what.
- `adapt_tick`, the post-prefill re-cut of `CROW_ADAPT=1`, is still never called by `serve`.
- The tick also runs for sampled requests, not only greedy: intended, harmless for identity
  since logits are untouched, not measured (every #37 chain ran greedy).
- The two preconditions `trickle_tick` asserts are read once at start
  (`bin/serve.rs:1653-1655`), so a `CROW_COLD_TIER` process logs a line instead of panicking.
- Fix round 1 of #37: `serve` sets `CROW_ADAPT_WINDOW=1` when it is unset, next to `CROW_GRAPH`
  and `CROW_MMA` (`bin/serve.rs:2313`), so the tick ranks swaps by the decayed selections since
  the last tick instead of the prefill-dominated cumulative count.
- Condition 2 covers that too: the ranking signal picks WHICH expert moves, not what is read.

**Where the trickle's copies are issued (#63b, 2026-09-12).**

| switch | issue point of the side-stream copies | mode |
|---|---|---|
| unset (default) | inside `trickle_tick`, BEFORE the token's graph launch (`gen.rs:3355-3365`) | operating |
| `CROW_TRICKLE_DEFER=1` | inside `decode_step`, AFTER the graph launch and before the end-of-step sync (`gen.rs:3131`, `Engine::trickle_drain_after_launch` at `gen.rs:3435`) | measurement |

- 63a measured the default form: 2.5968 ms per token of copies, class b (before the graph) 19,364 of 19,364, class a 0.
- The switch moves only the HOST issue order; `event_record(ev_commit)` and the table flip stay before the launch.
- Condition 2 covers the switch: it changes WHEN an expert moves, not what is read.

### 7.6 Snapshot and rollback (GDN, PLE conv, QSA ring)

**What needs no snapshot:**

- Rule: KV and the pooled QSA cache need **no** snapshot within a process.
- Reason: they are absolutely addressed, append-only, and a stale row past `pos` is never
  read (7.3).
- Consequence: only the states that fold history into a fixed-size buffer do.
- Exception, across a restart: the slot file carries them (7.3 table, `slot.rs`).

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
- The "budget 2,048 tokens" of section 2.1 is `QSA_BLOCK_TOPK` 512 selected blocks x
  compression ratio 4 = 2,048 tokens a *query* may attend to (`geo.rs:40-41`).
- The ring is a `ceil4(prompt_chunk + 4)`-row scratch buffer of raw indexer keys awaiting
  pooling.
- The two numbers are unrelated and only happen to sit close together at chunk 2048.

**When to snapshot. ONE point per turn, taken unconditionally (M2b, robin 2026-09-10, #36):**

| slot | point | position | reuse candidate |
|---|---|---|---|
| `SLOT_PROMPT` (`cache.rs:158`) | after the prefill of this turn's prompt | rendered prompt length | **yes**, it is prefill clean (7.3) |

- `SLOTS = 1` (`cache.rs:162`): one slot per process, and it is the prompt slot.
- The last generated id is never fed back, so it is not in `history` and not in `pos`.
- ONE held conversation per process (M1): a request that shares no prefix replaces it.
- The M1 after-answer point is gone from `chat_stream`; the block below says why.

**The after-answer snapshot was never the reuse case, and is DROPPED (M2b, #36):**

- The proposal said it is the normal case and spares the answer's prefill.
- Measured 2026-09-10 (A9, #31): it was taken, reported and never consumed
  (`prefill_clean` guard, `cache.rs:336`).
- Decision M2 option b (robin, 2026-09-10, #1 comment): drop it. Built in #36.
- Unchanged by the drop: the reuse behaviour, the `[cache]` stderr lines, `GET /slots`,
  the slot file format of 7.3 and `slot.rs`.
- Gone with it, per process: **130,646,016 B = 124.60 MiB** of pageable host RAM and one
  **14.5 ms** DtoH per request (**35 to 64 ms** on the first request of a process).
- Consequence, before and after the drop alike: every turn re-prefills the previous answer.
- Measured cost: **63 to 95 tokens** at the 16k operating point, still **99.41 % cached**
  (16,064 of 16,159), prefill **404 ms** (`decode_out/srv-a9.log`).
- The 63 is the answer alone: the M1 after-answer position 16,127 minus the prefill clean
  `P` 16,064 (`decode_out/srv-a9.log:42-43`).
- The 404 ms times all 95 re-prefilled tokens at 234.95 tok/s (`decode_out/srv-a9.log:29`);
  the answer's own share of those ms is **unmeasured**.
- For scale, the cold turn in the same log prefilled 16,064 ids in **21.63 s** of a
  **24.33 s** wall (`decode_out/srv-a9.log:21`, `:24`).
- The rollback lands on the prompt snapshot whether the divergence sits in the answer or
  not; that is the rule working, not a special case.
- The re-rendered assistant message need not reproduce the generated ids exactly, which is a
  second reason the after-answer position rarely matched anyway.

**Point 1 covers the regenerate / edited-answer case:**

- The prompt's ids are unchanged and the divergence sits in the *answer*.
- Consequence: `L >= ` the end of that prompt.
- Consequence: the point-1 snapshot is the newest candidate at or below `L`, and the
  rollback lands on it.
- Consequence: **the prompt's prefill is spared** and only the answer is recomputed.

**Point 1 does not cover the edited-prompt case, and the fallback CAN be cold. Corrected 2026-09-10 (A9, #31):**

- If the user rewrites their last message, the common prefix ends at the **start** of that
  prompt.
- That is below the point-1 snapshot, which is taken at prompt *end*.
- The proposal said: with any earlier snapshot held, the fallback is never cold.
- Measured: **false for this implementation.** There is exactly ONE usable slot, the current
  prompt end: one conversation is held, and since M2b (#36) there is one slot to hold it.
- Consequence: an edit below that position is a **cold prefill**, 21.6 to 22.0 s at 16k.
- The rule itself is unchanged and needs no special-casing:
  `P = max { prefill-clean S_pos : S_pos <= L }`.

**M2 DECISION (robin, 2026-09-10, #1 comment 5612308881): option b, built in #36.**

| option | effect | cost | decided |
|---|---|---|---|
| a: keep as built (M1) | the after-answer point is taken and never used | 124.60 MiB of host RAM and one DtoH of 14.5 ms per request, for nothing | no |
| b: drop the after-answer point | one slot, same reuse behaviour | saves that RAM and that copy | **YES** |
| c: reassign slot 2 to the PREVIOUS turn's after-prompt point | two prefill-clean positions, an edited last prompt stays warm | same RAM, same copies, more bookkeeping | no |

- A third slot is the same question with one more buffer; not taken either.
- Source: A9 review, issue #31, the plan's M2 list, and issue #36 for the build.
- Gate of the build: `decode_out/srv-a9b.log` (7.12 row M2b).

**Rollback (`PrefixCache::rollback`, `cache.rs:452`):**

1. `cuda::sync`, then `Engine::drop_decode_graph` (7.2, `reset.rs`).
2. Restore the four buffers with host-to-device copies.
3. Set `pos = S_pos`.
4. Set `done_blocks = S_pos / 4`.
5. Truncate `history` to `S_pos`, clear `route_log`.
6. Call `prefill` with the new ids from `S_pos` on.

- Note: `prefill`'s `init` flag zeroes `S` only when `self.pos == 0` (`gen.rs:2541`).
- That is exactly the cold-start case.
- Consequence: the restored path must leave it at 0.
- The current signature already does the right thing once `pos` is set.
- Measured HtoD: **11.98 to 13.12 ms** (A9, #31, `decode_out/srv-a9.log:43`, `:46`,
  `:142`, `:145`).

**What must NOT be reused:**

- Rule: never reuse a snapshot from a different engine load.
- Reason: the chunk size fixes the ring rows, the scratch, and the clamped hot-set N
  (`manager.rs:31`, `manager.rs:108-170`).
- Consequence: a snapshot's shape is only valid for the process that produced it.
- Consequence: snapshots are in-process state, not a file format.

**The ONE deliberate exception: the slot file (#32 A10, `engine/src/slot.rs`):**

- Rule: `POST /slots/0?action=save|restore` writes the `SLOT_PROMPT` slot to a file and
  reads it back, because Crow saves at exit and restores at the next start
  (`crow_core.py:2458`, `:2688`).
- Rule: because it IS a file, the file carries the whole load shape in its header, and a
  restore refuses any mismatch **before the first device write**.
- Rule: only `SLOT_PROMPT` is ever written. Since M2b (#36) it is the only slot, and it is
  the one prefill-clean position (7.3).

| header field | what it pins |
|---|---|
| `magic` `CROWSLT\x01` | the file kind |
| `format_version` (1) | payload order and header layout |
| `load_id` | fnv1a-64 of the container path mixed with `n_hot`; catches the obvious model swap, NOT a content hash |
| `n_ctx` | the context of the load |
| `prompt_chunk` | 2048 as pinned (M1) |
| `qsa_ring_rows` | `ring`, which follows from the chunk |
| `gdn_layers`, `attn_layers` | the geometry |
| `kv_groups` (`attn_layers * 2 * NKV`) | the KV fan-out |
| `kv_row_bytes` | `AHD * bytes per KV value`, so the KV dtype by size |
| `pooled_row_bytes` (`QSA_HIDD * 4`) | the pooled block row |
| `state_bytes` | `Shape::snapshot_bytes`, the four recurrent buffers |
| `pos` | the saved prefill-clean position, and `n_saved` on the wire |
| `done_blocks` | `pos / 4`, refused when it differs |
| `history_len` | must equal `pos` |

- Header size: 120 bytes, magic plus fourteen little-endian u64 (`slot.rs`).
- Payload order: `gdn_s`, `gdn_conv`, `ple_state`, `qsa_ring`, `kv` rows `0..pos`, pooled
  blocks `0..done_blocks`, `history[..pos]`.
- Save is atomic: sibling `.part-<pid>`, fsync, `fs::rename` over the target.
- Restore fills `SLOT_PROMPT` and sets `pos`, `done_blocks`, `history`.
- Consequence: the next chat request is an ordinary warm turn of 7.4, `L >= pos`, `P = pos`.
- Measured at 16k (A10, #32, `decode_out/srv-a10-fix.log`): file **352,843,384 B**,
  `n_saved` = `n_restored` = **16,064**, save **173 ms** (108 ms before the fsync was added),
  restore **137 ms** in the same process and **225 to 235 ms** into a fresh one.

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
| 2048 (the serve default, M1) | 2052 | 12,607,488 B = 12.02 MiB | **130,646,016 B = 124.60 MiB** |

**Totals per held conversation, ONE snapshot since M2b (robin 2026-09-10, #36):**

| build | chunk 512 | chunk 2048 (the serve default) |
|---|---|---|
| M1, two snapshots | 231.19 MiB | 249.19 MiB = 261,292,032 B |
| M2b, one snapshot | **115.60 MiB** | **124.60 MiB = 130,646,016 B** |
| saved by M2b | 115.60 MiB | **124.60 MiB = 130,646,016 B** |

**Confirmed against the build (A9, #31, `decode_out/srv-a9.log`, `cache.rs:235`):**

| quantity | value | note |
|---|---|---|
| snapshot bytes at chunk 2048 | **130,646,016 B** | exactly the table row above |
| DtoH per snapshot, warm | **14.5 ms** | steady state |
| DtoH per snapshot, first request of a process | **35 to 64 ms** | first-touch of the host pages |
| HtoD per rollback | **11.98 to 13.12 ms** | 7.9 acceptance point 1 and 3 |

**Where snapshots live:**

- Rule: snapshots belong in **pageable host RAM**.
- Not VRAM: VRAM is already the binding budget, and the loader clamps N against it
  (`manager.rs:122-170`).
- Not the pinned tier: it is budgeted at 46 GiB against a measured ~48.5 GiB host ceiling
  (`geo.rs:110`).
- Allocated once at process start and reused per snapshot (`cache.rs:306`, `cache.rs:310`).

**The formula holds for the default QSA layout only.**

- `CROW_QSA_FULL=1` (`manager.rs:37-41`) sets `ring = context`.
- That is the full-length raw-key layout from before 2026-09-05 (`manager.rs:20-21`),
  which is still supported.
- The ring term is then `12 * 262,144 * 128 * 4 = 1,610,612,736 B` (about **1.5 GiB**) at
  the default context of 262,144 (`geo.rs:105`).
- That is roughly **12.3x one snapshot** (1,610,612,736 / 130,646,016 B), and it would
  dominate every number in this section.
- Rule: `CROW_QSA_FULL=1` is therefore **out of scope for serve**.
- Consequence: the server runs the default layout, and every size above assumes it.

### 7.8 What a cache miss actually costs

- One number carries the section: **a full 16k prefill**.
- Plan reference: **24.13 s wall** (chain F43, 2026-09-06; 24.19 s by chain F49 on the
  installed build d211ab52ad2b).
- This build: **21.6 to 22.0 s** (A9, #31, `decode_out/srv-a9.log`, cold turn 1, 16,064 ids).
- It is also the honest ceiling for every row of the 7.3 table.
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

**What the built server actually pays per turn at 16k (A9, #31):**

| turn | prompt ids | cached | prefilled | prefill wall |
|---|---|---|---|---|
| cold turn 1 | 16,064 | 0 | 16,064 | 21.6 to 22.0 s |
| warm turn 2 | 16,159 | 16,064 (99.41 %) | 95 (0.59 %) | 404 ms |
| rollback into the last turn | n/a | n/a | n/a | 11.98 to 13.12 ms HtoD |

- `prompt_ms` on the wire is the `Engine::prefill` call only: it excludes the rollback, the
  reset, the snapshots and the tokenizer (`serve.rs` module doc).
- Consequence: a low `prompt_per_second` on a warm turn is a small-batch effect, not a
  regression.

### 7.9 Acceptance

- Section 7 approved by robin at M1 on 2026-09-09, as sections 0-6 were on 2026-09-02.
- The A9 gate stands as written: **ids bit-identical to a cold run, twice**, greedy, with
  the default cold tier.
- The three measurements this section asked for, delivered by #31 A9:

| # | measurement | result | source |
|---|---|---|---|
| 1 | snapshot copy time, DtoH and HtoD | DtoH 14.5 ms warm, 35 to 64 ms on the first request; HtoD 11.98 to 13.12 ms | `decode_out/srv-a9.log` |
| 2 | warm-turn time for a 16k transcript whose next turn appends a short prompt | prefill 404 ms, 16,064 of 16,159 ids cached (99.41 %) | `decode_out/srv-a9.log` |
| 3 | rollback time when the divergence lands inside the last turn | 11.98 to 13.12 ms, `L` 16,172, `P` 16,159 | `decode_out/srv-a9.log` |

- The A9 identity gate itself: part 1 cached 99.41 %, 2 of 2; part 2 identity 2 of 2, one
  sha over 5 runs; part 3 rollback PASS 2 of 2 (`decode_out/srv-a9.log`,
  `decode_out/srv-a9-fix.log`, same three shas).

### 7.10 M1 decisions (answered by robin 2026-09-09, issue #1)

| # | question of the proposal | decision | where it lives in the build |
|---|---|---|---|
| 1 | which prefill chunk is pinned for the process | **2048**, pinned at load; `geo::apply_chunk_policy` is NOT applied | `serve.rs:449` (`SERVE_CHUNK`), `serve.rs` module doc |
| 2 | how many conversations are held, how many snapshots each | **ONE** conversation, **two** snapshots (249.19 MiB at chunk 2048) | `cache.rs:162` (`SLOTS`), `cache.rs` module doc |
| 3 | is the post-answer snapshot taken unconditionally | **yes**, both points unconditional | `cache.rs` module doc, 7.6 |
| 4 | concurrency: queue or reject | **a second request waits** in the accept queue, no 503 | `serve.rs` module doc, blocking `TcpListener` |
| 5 | sampling: out of scope, or reseed per request | **reseeded per request**; the A9 identity gate runs **greedy** | `Engine::enable_dev_sampler`, `serve.rs:1043` (`sampler_from`) |

- Rows 2 and 3 are SUPERSEDED by M2 option b (robin 2026-09-10, #36): **ONE** snapshot per
  process, **124.60 MiB** at chunk 2048, the after-answer point dropped. See 7.6 and 7.7.
- Row 5 confirmed by C2 (robin 2026-09-11, #55): the default stays request driven, greedy
  without `temperature`, sampled above 0; the ten-task gate under sampling is met in 1 of 6
  seeds (#44), under greedy in 0 of 1 (#11). See 7.12.

**Note carried from the proposal, still open as a measurement, not a doc edit:**

- `geo.rs:157` says hot set **N 147** for chunk 2048; `geo.rs:132` says **140** for the same
  operating point.
- The source disagrees with itself; settling it is a measurement.
- The serve gates ran with the sidecar `decode_out/hotsets-M-longctx2100-n160.json`, so this
  disagreement did not decide anything in stage A.

### 7.11 The endpoint contract as built (stage A, #24 to #32)

**Rule for this section: every row names the crow-nest anchor AND the `crow_core.py` reader.**

- Crow's readers were read on 2026-09-09 from `C:\Users\robin\dev\Crow\cli\crow_core.py`
  (worktrees under `.claude/worktrees/` excluded).
- Line numbers are of that reading.

**7.11.1 `GET /health`**

| item | as built | crow-nest anchor | Crow reader |
|---|---|---|---|
| route | `GET /health`, query and trailing slash dropped | `serve.rs:495`, `serve.rs:772` | `crow_core.py:14801` (`health_url`) |
| body | `{"status":"ok"}` | `serve.rs:2188` | `crow_core.py:14814` (`check_endpoint`) |

**7.11.2 `GET /props`**

| field | as built | crow-nest anchor | Crow reader |
|---|---|---|---|
| `model_path` | the container path of this load | `serve.rs:787` (`props_json`) | `crow_core.py:1408` (`server_model_path`) |
| `model` | the container file stem, `.cnq` stripped | `serve.rs:780` (`model_name`) | `crow_core.py:14884` (`fetch_model_name`) |
| `n_ctx` | `Engine::st.context`, read back from the load, 200,000 (`CONTEXT_FLOOR`) | `serve.rs:787` | `crow_core.py:14837` (`fetch_n_ctx`) |
| `default_generation_settings.n_ctx` | the same number | `serve.rs:787` | `crow_core.py:14837` |
| `modalities.vision` | `false` | `serve.rs:787` | `crow_core.py:1429` (`refuse_images`) |
| `prompt_chunk` | 2048 (M1) | `serve.rs:787` | no reader in Crow, informational |
| `build` | `crow-nest-engine 0.1.0` | `serve.rs:787` | no reader in Crow, informational |

**7.11.3 `POST /v1/chat/completions`, request body**

| field | as built | crow-nest anchor | Crow writer |
|---|---|---|---|
| `messages` | required, non empty, every entry needs a string `role` | `serve.rs:918` (`parse_chat`) | `crow_core.py:4672-4700` |
| `model` | echoed into every chunk, default `crow-nest` | `serve.rs:918`, `serve.rs:1144` | `crow_core.py:4672-4700` |
| `stream` | `true` streams `chat.completion.chunk` frames; `false` or absent answers ONE `chat.completion` document (#39 B3a, 7.11.13) | `serve.rs:918`, `serve.rs:1570` | `crow_core.py:4672-4700` (always `true`), `:2971` (digest path, no `stream` field) |
| `stream_options.include_usage` | `true` puts `usage` on the final chunk | `serve.rs:918`, `serve.rs:1222` | `crow_core.py:4672-4700` |
| `timings_per_token` | `true` puts `timings` on the final chunk | `serve.rs:918`, `serve.rs:1222` | `crow_core.py:4672-4700` |
| `max_tokens` | default 1024, capped at 32768, clamped to `n_ctx - prompt ids` | `serve.rs:1253` (`clamped_max_tokens`) | `crow_core.py:4672-4700` |
| `temperature` | absent, `null` or `<= 0` is GREEDY; `> 0` samples | `serve.rs:1043` (`sampler_from`) | `crow_core.py:4672-4700` |
| `top_p` | nucleus mass, default 0.8 (data sheet), read only when `temperature > 0` | `serve.rs:461`, `serve.rs:1043` | `crow_core.py:4672-4700` |
| `top_k` | default 20 (data sheet), clamped to 64 by the device sampler (`SAMPLE_MAXK`, `engine/src/kernels.rs:2942`), read only when `temperature > 0` | `serve.rs:463`, `serve.rs:1043` | not sent by Crow |
| `presence_penalty` | default 1.5 (data sheet), read only when `temperature > 0` | `serve.rs:465`, `serve.rs:1043` | not sent by Crow |
| `seed` | RNG seed of THIS request, default 0, reseeded per request (M1) | `serve.rs:467`, `serve.rs:1043` | not sent by Crow |
| `min_p` | **ACCEPTED AND IGNORED**, one stderr line per request | `serve.rs:1751` (the stderr line); `sampler_from` (`serve.rs:1043`) carries no `min_p`; `serve.rs` module doc | `crow_core.py:4672-4700` (0.01 at Crow's operating point) |
| `tools` | rendered as the template variable `tools` | `serve.rs:918`, `tokenizer::render_chat` | `crow_core.py:4672-4700`, `TOOLS` (25 builtin at `crow_core.py:579-838`, frozen at `:846`, plus the `mcp.json` tools added at import, `:841`) |
| `chat_template_kwargs.enable_thinking` | template variable, default false | `serve.rs:918` | `crow_core.py:2970` (digest path) |
| `messages[].role = "tool"` | `content` rendered as `<tool_response>...</tool_response>` | `serve.rs:1284` (`normalize_messages`) | `crow_core.py` tool turns |
| `messages[].tool_calls[].function.arguments` | a JSON STRING from Crow is parsed into the MAPPING the template needs | `serve.rs:1284` | `crow_core.py:3564` |
| `tool_call_id` | carried, never read; this template pairs by order | `serve.rs:1284` | `crow_core.py` tool turns |

**7.11.4 `POST /v1/chat/completions`, the stream**

| order | line as built | crow-nest anchor | Crow reader |
|---|---|---|---|
| 1 | `delta:{"role":"assistant"}`, `finish_reason` null | `serve.rs:1168` (`chunk_role`) | `crow_core.py:4831-4877` |
| 2..n | `delta:{"content":"..."}` , one per emitted piece | `serve.rs:1173` (`chunk_content`) | `crow_core.py:4831-4877` (`delta.content`) |
| n+1 | `delta:{}` plus `finish_reason`, optionally `usage` and `timings` | `serve.rs:1222` (`chunk_finish`) | `crow_core.py:4831-4877`, `:4999-5018` |
| n+2 | `data: [DONE]` | `serve.rs:471` (`SSE_DONE`) | `crow_core.py:4035` |
| framing | `data: <compact json>` plus a blank line, one flush per frame | `serve.rs:1244` (`sse_frame`) | `crow_core.py:4831-4877` |
| headers | `text/event-stream`, `no-cache`, `Connection: close`, no `Content-Length` | `serve.rs:1580` (`chat_stream`) | `crow_core.py:4821` (the train) |
| `finish_reason` | `stop` (EOS), `length` (budget), `tool_calls` (a call was closed) | `serve.rs:1580` | `crow_core.py:4831-4877` |

- One exception to "one token, one frame": a content token whose tail is a prefix of
  `<tool_call>` is HELD until the next token resolves it (`engine/src/toolcall.rs`).
- The concatenated content is the same either way; only the frame boundary moves.

**7.11.5 `usage` and `timings` on the final chunk**

| object | field | as built | crow-nest anchor | Crow reader |
|---|---|---|---|---|
| `usage` | `prompt_tokens` | rendered prompt ids, cached part included | `serve.rs:1111` (`usage_json`) | `crow_core.py:4831-4877` |
| `usage` | `completion_tokens` | generated ids, **the prefill token included**: it is `t.predicted_n` itself (`serve.rs:1112`), the same value as `timings.predicted_n` | `serve.rs:1111` | `crow_core.py:4831-4877` |
| `usage` | `total_tokens` | `prompt_tokens + completion_tokens` | `serve.rs:1111` | `crow_core.py:4831-4877` |
| `usage` | `prompt_tokens_details.cached_tokens` | `P`, ALWAYS present as an integer | `serve.rs:1111` | `crow_core.py:4831-4877`, fallback at `:14923` |
| `timings` | `prompt_n` | `prompt_tokens - cached_tokens`, the ids actually prefilled | `serve.rs:1124` (`timings_json`) | `crow_core.py:4999-5018` |
| `timings` | `prompt_ms` | wall of the `Engine::prefill` call only | `serve.rs:1124` | `crow_core.py:4999-5018` |
| `timings` | `prompt_per_second` | `prompt_n / prompt_ms * 1000` | `serve.rs:1083` (`per_second`) | `crow_core.py:4999-5018` |
| `timings` | `prompt_per_token_ms` | `prompt_ms / prompt_n` | `serve.rs:1092` | no reader in Crow |
| `timings` | `predicted_n` | generated ids, **the prefill token included** (llama-server convention); the same value as `usage.completion_tokens` (`serve.rs:1112`, `serve.rs:1128`) | `serve.rs:1124` | `crow_core.py:4999-5018` |
| `timings` | `predicted_ms` | wall of the decode loop, first `decode_step` to the last | `serve.rs:1124` | `crow_core.py:4999-5018` |
| `timings` | `predicted_per_second` | `predicted_n / predicted_ms * 1000` | `serve.rs:1083` | `crow_core.py:4999-5018` |
| `timings` | `predicted_per_token_ms` | `predicted_ms / predicted_n` | `serve.rs:1092` | no reader in Crow |
| `timings` | `cache_n` | `P`, the same number as `cached_tokens` | `serve.rs:1124` | Crow's measuring tools |
| `timings` | `crow_expert_selections` | u64, cumulative, `atomicAdd(&counters[0], 10ull)` per token per layer | `kernels.rs:2181`, launched `gen.rs:1972-1974` | Crow #54 rule, tools |
| `timings` | `crow_expert_cold` | u64, cumulative, `atomicAdd(&counters[1], __popc(s_cold))` | `kernels.rs:2182` | Crow #54 rule, tools |
| `timings` | `crow_ple_rows` | u64, cumulative, PLE rows requested | `gen.rs:1055` | Crow #54 rule, tools |
| `timings` | `crow_ple_misses` | u64, cumulative, PLE rows filled from the container | `gen.rs:1056` | Crow #54 rule, tools |
| `timings` | `crow_layers` | int, `geo::LAYERS` = 48, the divisor | `serve.rs:1124` | Crow #54 rule, tools |

- Rule: the five `crow_*` counters are **cumulative per process and never reset**, in any
  place, per request or otherwise (the Crow #54 rule).
- Reason: a request-local value is the DIFFERENCE of two consecutive blocks; a reset would
  break that for every reader at once.
- They are read by `Engine::drain_counters` (`gen.rs:3343`, `residency.rs:638`), a
  `dtoh_u64` of 48 x 2 u64 = 768 bytes; "drain" READS, it does not zero.
- Measured cost of that read: **0.026 ms** per request (A8, #30, `decode_out/srv-a8.log`).
- Two decode rates on purpose: stderr prints `(gen - 1) / decode_ms * 1000`, the wire prints
  `gen / decode_ms * 1000`. Same `decode_ms`, different numerator, because `predicted_n`
  counts the prefill token and Crow's reader expects that ratio.
- Every rate is 0.0 when its ms is 0, negative or not finite; no NaN reaches the wire.

**7.11.6 `tool_calls` on the wire (#29 A7)**

| order | `delta` as built | crow-nest anchor | Crow reader |
|---|---|---|---|
| 1 | `{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"...","arguments":""}}]}` | `serve.rs:1180` (`chunk_tool_open`) | `crow_core.py:4864-4877`, `:4869-4874` |
| 2..n | `{"tool_calls":[{"index":0,"function":{"arguments":"<fragment>"}}]}` | `serve.rs:1202` (`chunk_tool_args`) | `crow_core.py:4864-4877` |
| last | `{}` with `finish_reason":"tool_calls"` | `serve.rs:1222` | `crow_core.py:4831-4877` |

- `id` and `name` ride on chunk 1 ONLY; Crow overwrites them only on a truthy value, so a
  later chunk can never erase them.
- The concatenation of every `arguments` fragment of one index is the arguments JSON text.
- A second call gets `index` 1, a third `index` 2; each has its own `call_<index>`.
- The model's markup is XML-like, NOT the JSON form:

```text
<tool_call>
<function=read_file>
<parameter=path>
C:/x/y.md
</parameter>
</function>
</tool_call>
```

- `<tool_call>` is added token 248058 and is matched by TOKEN ID; `</tool_call>` is 248059
  and is matched by TEXT (the ids at `toolcall.rs:27`, the two constants at
  `toolcall.rs:52-55`, the arming call `arm()` at `toolcall.rs:264`).
- The OpenAI `arguments` object is BUILT from the parameter blocks by declared schema type
  (`tool_param_types` at `toolcall.rs:141`, `value()` at `toolcall.rs:374`; the piece feed
  that drives them is `feed()` at `toolcall.rs:280`).
- EOS after `</function>` CLOSES the call: `finish_reason` `tool_calls`, trailing markup
  dropped and counted, never replayed as content (`toolcall.rs:293`).
- Malformed markup (no `</function>`, no name): the RAW markup goes out as `delta.content`,
  `finish_reason` stays `stop` or `length`, `arguments` stays unterminated on purpose so
  Crow's `json.loads` fails rather than running half a command.
- `crow_core.TOOLS` is **25 builtin declarations** (`crow_core.py:579-838`, frozen as
  `BUILTIN_TOOLS` at `:846`) plus whatever `mcp.json` adds at import (`:841`, grown at
  `Crow/cli/crow_core.py:9159`, reset at `Crow/cli/crow_core.py:9154`).
- Measured on this machine on 2026-09-09: **31** declarations, 25 builtin plus 6 MCP
  (`decode_out/srv-a7-tools.json`). The plan said seven.
- The rendered tools block is byte-identical to the Python oracle at **322 ids**
  (A3 #25, A7 #29).

**7.11.7 `GET /slots` and `POST /slots/0`**

| item | as built | crow-nest anchor | Crow reader |
|---|---|---|---|
| `GET /slots` | `[{"id":0,"n_ctx":...,"n_prompt_tokens":...,"is_processing":false}]` | `serve.rs:812` (`slots_json`) | `tools/measure-slot-restart.ps1:87`, `tools/probe-slot-persistence.py:152` |
| `n_prompt_tokens` | the held PREFILL CLEAN position, 0 while none is held | `serve.rs:812` | the same two tools, element 0 |
| `POST /slots/0?action=save` | body `{"filename": "<bare name>"}` | `serve.rs:845` (`slot_filename`), `serve.rs:2075` (`slot_route`) | `crow_core.py:2458` |
| save answer | `id_slot`, `filename`, **`n_saved`**, `n_written`, `timings.save_ms` | `serve.rs:822` (`slot_saved_json`) | `crow_core.py:2458` reads `n_saved` |
| `POST /slots/0?action=restore` | body `{"filename": "<bare name>"}` | `serve.rs:845`, `serve.rs:2075` | `crow_core.py:2688` |
| restore answer | `id_slot`, `filename`, **`n_restored`**, `n_read`, `timings.restore_ms` | `serve.rs:833` (`slot_restored_json`) | `crow_core.py:2688` reads `n_restored` |
| the contract | `n_saved == n_restored`; Crow withdraws the warm-cache claim when they differ | `slot.rs`, `serve.rs` module doc | `crow_core.py:2694` |
| `--slot-save-path <existing dir>` | required for both actions; a typo exits **2 at boot** | `serve.rs:577` (`check_slot_save_path`) | not read by Crow |
| without `--slot-save-path` | both actions answer **400**, as llama-server refuses them | `serve.rs:2075` | not read by Crow |
| `filename` | bare name only, allowlist `[A-Za-z0-9._-]`, Windows device names refused | `slot.rs:342` (`sanitize_filename`) | not read by Crow |

- Only `n_saved` and `n_restored` are contractual; Crow reads nothing else of these bodies.
- Both numbers are the PREFILL CLEAN position, so `SLOT_PROMPT`, never the answer.
- Save is atomic (temp, fsync, rename); restore refuses every shape, content and size
  mismatch BEFORE the first engine write (7.6, `slot.rs`).

**7.11.8 Refusals and status codes**

| case | answer | crow-nest anchor |
|---|---|---|
| unknown route, or a wrong method on a known path | 404 JSON naming every route this server answers | `serve.rs:855` (`not_found_json`), `serve.rs:495` (`route`) |
| garbage request line | 400 JSON `{"error":"bad request"}` | `serve.rs:1971` (`read_head_from`) |
| malformed or repeated `Content-Length` | 400 JSON | `serve.rs:1971` |
| head (request line plus headers) over 64 KiB | 431 JSON, then close | `serve.rs:453`, `serve.rs:1971` |
| body over 16 MiB | 413 JSON, then close | `serve.rs:455`, `serve.rs:1971` |
| `Transfer-Encoding: chunked` | 501 JSON | `serve.rs:1971` |
| `stream: false` or absent | **200, one `chat.completion` document** (501 until #39) | `serve.rs:1570` (`chat_route` branch), `serve.rs:1606` (`chat_document`) |
| prompt ids `>= n_ctx` | 413 before any GPU work | `serve.rs:1253` (`clamped_max_tokens`) |
| `/slots/0` save with no prefill-clean position held | 409 | `serve.rs:2075`, `slot.rs` |
| `/slots/0` bad filename, missing file, shape or content mismatch | 400, engine untouched | `slot.rs:342`, `slot.rs:288` (`check_content`) |
| a second `serve` process | non-zero exit on `engine/.engine.lock` | `serve.rs` module doc, `Engine::load` |
| read or write timeout (10 s per connection) | one stderr line, that connection closed, accept loop continues | `serve.rs:451` |
| a client that sent nothing | closed silently, no response | `serve.rs:1971` |

- Measured (A2, #24): `engine/.engine.lock` is held for the process life and is **left
  behind by a `Stop-Process` kill**; it must be removed by hand before the next engine run.

**7.11.9 What is deliberately NOT built**

| item | state | reason |
|---|---|---|
| `min_p` | parsed, ignored, logged once per request | the device sampler `sample_k` implements top_k, top_p and presence only; adding it is a kernel change. **Open for robin at M2** (#28) |
| `/tokenize` | not built | no caller in the client; the mentions in Crow's `CHANGELOG.md:1172`, `:1768` are measurement prose |
| `/v1/models`, `/v1/messages` | not built | those are the REMOTE providers in Crow (`crow_core.py:13593`, `:13649`, `:13669`, `:3600-3610`), not the local server |
| `/completion` | not built | appears only in Crow's log-parser test fixtures |
| `/apply-template` | not built | only Crow's probes call it (`tools/probe_reasoning_levels.py:92`, `tools/check_chat_template.py:19`) |
| `delta.reasoning_content` | never emitted | `enable_thinking` is false on this path; Crow reads the key if present (`crow_core.py:4831-4877`) |
| stream trickle in serve | ticked once per `decode_step` (#37) | `bin/serve.rs:1841-1843`, the mirror of `bin/decode.rs:224-231`; drained after the last step; one `[serve]` line at start says whether this process ticks, and the `[chat]` line carries `crow_trickle_swaps` per request |
| the trickle's ranking signal in serve | `CROW_ADAPT_WINDOW=1` by default (#37 fix round 1) | `bin/serve.rs:2313` sets it when unset, the same loop as `CROW_GRAPH` and `CROW_MMA`; an explicit `CROW_ADAPT_WINDOW=0` restores the cumulative ranking |
| `adapt_tick` in serve | never called | the post-prefill re-cut of `CROW_ADAPT=1` stays a harness path; callers are `bin/decode.rs:230` and `bin/parity.rs:216` |
| `CROW_QSA_FULL=1` | out of scope | 7.7, the ring would dominate every size |
| `CROW_COLD_TIER` low-bit tier | must stay off | 7.5 condition 2 |

- Every `CROW_*` variable of the engine has one row in `docs/env.md`; `tools/check_env_docs.py` guards code against doc.

- `stream: false` left this table with #39 (B3a): the probe-suite and Crow's rollover
  digest send no `stream` field, and both are served now. See 7.11.13.

**7.11.10 Two measured traps for a Crow-driven test (stage B)**

| trap | evidence | consequence |
|---|---|---|
| Crow looks for the server binary at `<install>/bin/llama-server.exe` and builds a llama-server command line | `crow_core.py:1222`, `:1234` | Crow cannot BOOT crow-nest without a change to Crow; `--base-url` is enough for a test |
| Crow's overbooking guard looks for processes `Name like 'llama-server%'` | `crow_core.py:1309` (`_PROCESS_QUERY`) | a running crow-nest server is INVISIBLE to it; no llama-server may run during stage B, and that is checked before every run, not assumed |

**7.11.11 The tokenizer and the template behind every chat request (#25 A3)**

| item | as built | evidence |
|---|---|---|
| tokenizer | in-engine (`crow_nest_engine::tokenizer`), no Python process is started | A3 #25 |
| template | minijinja, the model's own `tokenizer_config.json` chat template | A3 #25 |
| gate | ids identical to the Python oracle on **10 of 10** prompts, plus a 6 of 6 docs file | `decode_out/srv-a3-tok.log`, `srv-a3-rust-ids.json`, `srv-a3-oracle-ids.json` |
| tools render | byte-identical to the oracle at **322 ids**, but ONLY with `preserve_order` on serde_json AND on minijinja | A3 #25 |
| warm-up | the tokenizer loads right after argument parsing, BEFORE `cuda::Ctx::init`; failure exits 3 in a second | `serve.rs` module doc |
| subcommand | `serve tokenize --chat\|--raw` runs without CUDA and without `engine/.engine.lock` | `serve.rs` module doc |
| harness trap | the Python oracle decodes STDIN as cp1252 unless `PYTHONIOENCODING=utf-8` is set; 4 of 10 prompts are affected when it is bare | issue #34 |

**7.11.12 Open items for M2 (not decided here)**

| # | item | state | source |
|---|---|---|---|
| 1 | `min_p` in the device sampler | accepted and ignored; kernel change or drop it from Crow's profile | #28 |
| 2 | serve decode rate | serve reaches **18 to 24 tok/s** across the A4 to A6 gates. The **35.7 tok/s** #35 compares against is `decode run` on the **t3-debug** prompt (3,769 prompt ids; the named file records `decode_tok_s` **36.19**, `decode_out/final4-t3-debug-run0-crow.json`, harness of 2026-09-05/06), NOT the 16k t1-read prompt: on that prompt the A5 `decode run` control measures 17.1 tok/s over one timed decode token (58.50 ms, `decode_out/srv-a5-decoderun.log:151-154`). Prefill on the 16k prompt is equal, 662.33 (serve) vs 664.08 (`decode run`) tok/s, -0.26 % (`decode_out/srv-a5.log:144-145`). **Cause unmeasured**, and the two rates are not on one prompt | #35, `decode_out/final4-t3-debug-run0-crow.json`, `srv-a5.log`, `srv-a5-decoderun.log` |
| 3 | the after-answer snapshot slot | DECIDED 2026-09-10: option b, dropped; built in #36, gate `decode_out/srv-a9b.log` | #31, #36, 7.6 |

- Rule: items 1 and 2 are not decided in this document; item 3 is, and 7.6 carries it.

**7.11.13 `POST /v1/chat/completions`, the non-streaming document (#39 B3a)**

- Sent to every caller that omits `stream`, or sets it to `false`.
- Two such callers exist in Crow: the probe-suite (`tools/probe-suite.py:604-639`, `ask_model`)
  and the rollover digest (`cli/crow_core.py:2960-2990`).
- Until #39 both got a 501; that is why `stream: false` left the 7.11.9 table.

| field | as built | crow-nest anchor | reader |
|---|---|---|---|
| `id` | `chatcmpl-<created>-<seq>`, the same form the stream uses | `serve.rs:1477` (`completion_json`), `serve.rs:1648` (`chat_generate`) | not read by either caller |
| `object` | `chat.completion` | `serve.rs:1477` | not read by either caller |
| `created` | unix seconds of this request | `serve.rs:1477` | not read by either caller |
| `model` | echoed from the request, default `crow-nest` | `serve.rs:1477` | not read by either caller |
| `choices[0].index` | `0`, one choice per request | `serve.rs:1477` | not read by either caller |
| `choices[0].message.role` | `assistant` | `serve.rs:1477` | not read by either caller |
| `choices[0].message.content` | ALWAYS a string: every content delta of the stream, concatenated; empty when a tool call was the whole answer | `serve.rs:1397` (`CollectSink`), `serve.rs:1477` | `probe-suite.py:679`, `crow_core.py:2984` |
| `choices[0].message.tool_calls` | present ONLY when the parser closed a call: `[{id, type "function", function{name, arguments}}]`, `arguments` a JSON STRING | `serve.rs:1388` (`CallBuf`), `serve.rs:1477` | neither caller reads it |
| `choices[0].message.reasoning_content` | NEVER present, as on the stream (`enable_thinking` is false, `tokenizer.rs:18-19`) | `serve.rs:1477` | `probe-suite.py:680` reads it when present |
| `choices[0].finish_reason` | `stop`, `length` or `tool_calls`, the stream's rules unchanged | `serve.rs:1648` | `probe-suite.py:678` |
| `usage` | `usage_json`, the object of the final stream chunk, ALWAYS present | `serve.rs:1109` (`usage_json`), `serve.rs:1477` | `probe-suite.py:681-683` (`completion_tokens`) |
| `timings` | `timings_json`, the object of the final stream chunk, ALWAYS present | `serve.rs:1122` (`timings_json`), `serve.rs:1477` | neither caller reads it |
| headers | `application/json`, `Content-Length`, `Connection: close`, as on every other JSON route | `serve.rs:1606` (`chat_document`), `serve.rs:1947` (`respond`) | `urllib.request` in both callers |

**The decisions #39 took, and why:**

| decision | choice | reason |
|---|---|---|
| `usage` and `timings` on the document | ALWAYS, neither flag consulted | ONE document shape; the probe-suite reads `usage.completion_tokens` while sending neither `stream_options` nor `timings_per_token`; llama-server also carries `timings` on the non-streaming document |
| `stream_options` on a non-streaming request | accepted, ignored | it names a stream that does not exist |
| `message.content` when a tool call is the whole answer | the empty STRING, never `null` | both readers use `or ""`, so either works; a string keeps one type on the wire |
| `tool_calls[]` entry shape | OpenAI non-streaming: `id`, `type`, `function`, no `index` | `index` is a stream reassembly field, meaningless on a document |
| the generation | ONE loop, two sinks | a second loop would let the two request forms drift apart |

**The sink split (the refactor #39 made, `serve.rs`):**

| element | where it lives | shared by both request forms |
|---|---|---|
| prefix cache decide, rollback or reset, snapshot point 1 | `chat_generate` (`serve.rs:1648`) | yes |
| prefill, device sampler arm or park, decode loop, stop rules | `chat_generate` | yes |
| detokenizer and the hold back of an incomplete character | `chat_generate`, `next_delta` (`serve.rs:1266`) | yes |
| tool-call parser `ToolStream` and the `finish_reason` rules | `chat_generate` | yes |
| engine counters and the `Timing` block | `chat_generate` | yes |
| the three `[chat]` stderr lines, `[chat] ids` included | `chat_generate` | yes |
| the per delta side effect | `ChatSink::on_emit` (`serve.rs:1327`) | no, this is the split |
| the wire form | `SseSink` writes frames (`serve.rs:1346`), `CollectSink` fills two strings (`serve.rs:1397`) | no |
| the route switch | `chat_route` (`serve.rs:1532`), branch at `serve.rs:1570` | no |

- `chat_stream` (`serve.rs:1580`) writes the SSE head, then runs `chat_generate` with `SseSink`.
- `chat_document` (`serve.rs:1606`) runs `chat_generate` with `CollectSink`, then answers
  `completion_json`.
- Wire proof of the refactor: the raw SSE bytes of one streaming request are identical
  before and after, `id`, `created` and the wall-clock `timings` numbers excepted
  (`decode_out/srv-b3a.log`, WIRE DIFF).

### 7.12 The stage A gate table (what was measured, and where the artefact is)

**Rule: a gate without a log artefact does not count.**

| task | issue | what its gate measured | artefact under `decode_out/` |
|---|---|---|---|
| A1 | #23 | this section written, reviewed, approved | none (documentation) |
| A2 | #24 | `/health` and `/props` through Crow's own readers 4 of 4, lock 1 of 1, 404/400/413/431/timeout | `srv-a2.log`, `srv-a2.stderr.log`, `srv-a2-fix.stderr.log` |
| A3 | #25 | in-engine tokenizer vs the Python oracle, ids identical 10 of 10 prompts, tools block 322 ids byte-identical | `srv-a3-tok.log`, `srv-a3-rust-ids.json`, `srv-a3-oracle-ids.json` |
| A4 | #26 | SSE stream through `crow_core.stream_reply`, two identical requests in one process give the fresh-process ids (4 of 4, t3-debug 2 of 2) | `srv-a4-gateA.log`, `srv-a4-gateB-procA.log`, `srv-a4-gateB-procB.log`, `srv-a4-fix.log`, `srv-a4-parity.log` |
| A5 | #27 | the eight `usage` and `timings` fields, totals 16,072 = 16,064 + 8, control serve 662.33 vs `decode run` 664.08 tok/s | `srv-a5.log`, `srv-a5-decoderun.log` |
| A6 | #28 | greedy identity 1 of 1, seed 7 warm equals cold 2 of 2, seed 8 differs, `finish_reason` 10 of 10 | `srv-a6.log`, `srv-a6-fix.log` |
| A7 | #29 | a real tool call in 24 fragments with `finish_reason` `tool_calls`, control without tools `stop`, identity 64 of 64 | `srv-a7.log`, `srv-a7-fix.log` |
| A8 | #30 | the five `crow_*` counters monotone 10 of 10, delta 2 to 3 = +22,560, fresh process starts at 0, read 0.026 ms | `srv-a8.log` |
| A9 | #31 | prefix cache: cached 99.41 %, warm ids bit-identical to cold 2 of 2, rollback 2 of 2, snapshot 130,646,016 B | `srv-a9.log`, `srv-a9-fix.log`, `srv-a9-control.log`, `srv-a9-chunkcut.log`, `srv-a9-probe.log` |
| A10 | #32 | slot file save and restore across processes, 30 of 30 then 22 of 22, `n_saved` = `n_restored` = 16,064, six refusals 4xx with a working chat after | `srv-a10.log`, `srv-a10-fix.log`, `srv-a10-smoke.log` |
| A11 | #33 | this section corrected, `cargo test` in engine and converter, parity 8 / 512 twice / 1024 against the installed build | `srv-a11.log` |
| M2b | #36 | after-answer snapshot dropped: A9 gate re-run in full, cached 99.41 % 2 of 2, warm turn 2 ids `b6eaffb2` 2 of 2, rollback `addbbcb5` 2 of 2, every run equals the A9 reference 10 of 10, serve `PrivateMemorySize64` 139,309,056 B lower after turn 1, 0 point-2 snapshot lines, A10 restore into a fresh process 6 of 6 | `srv-a9b.log` |
| B3a | #39 | `stream:false` answers ONE `chat.completion` document: probe-suite request form 15 of 15 field checks, greedy identity stream vs document 2 of 2 on a one token answer and 2 of 2 on a longer one, Crow digest form 200 with content, the A7 tool call equal in `function.name` and `arguments` with `finish_reason` `tool_calls`, A9 gate part 2 warm turn 2 `b6eaffb2` 1 of 1 at 99.41 % cached, the raw SSE bytes of one streaming request identical before and after the refactor 1 of 1 | `srv-b3a.log` |
| B4 | #11, Crow #192 | ten tasks arm-phased greedy, crow ids == final4 10 of 10, quality result per engine in the "decided" table below (this section), one reader Rev3 | `srv-b4.log`, the ten `final4-*-run0-crow.json` |
| C1 | #40 | sampling seeds 3 and 4, gate 0 seed 2 ids == smpv2 1,536 tokens, smp3 0/6/4, smp4 2/6/2 by the C1 reader | `srv-c1.log` (untracked since E3, named on #40) |
| C2 | #44 | sampling seeds 5 and 6 plus one reader over six series, 60 answers, gate met in 1 of 6 seeds (smpv1 1/6/3, smpv2 1/6/3, smp3 0/6/4, smp4 1/7/2, smp5 0/7/3, smp6 1/5/4), 4 / 37 / 19 of 60, degeneration 0 of 60, reviewer re-judged 60 of 60 with 55 agreeing | `srv-c2.log`, `srv-c2-reader.log`, `srv-c2-review.log` (untracked since E3, named on #44) |
| 19e | #19 | the decode staging kernel default flipped to `stage_cold_ca`: parity 7 of 7 forms byte-identical against `d211ab52ad2b` including the fallback `CROW_STAGE_KERNEL=1`, ten-task greedy ids == `final4` 10 of 10, A9 parts 1 to 3 PASS with reference shas 10 of 10, A10 smoke 6 of 6, three adjacent pairs 26.42 against 29.68 ms per token (mean, RTX 5090, 2026-09-12), ids sha `5098f885ab3a` in 7 of 7 runs | `srv-19e.log` |

**The ten-task gate on the server path, decided (C2, robin 2026-09-11, #55):**

| item | value |
|---|---|
| gate rule | pass >= 1 of 10, fail <= 2 of 10, no degeneration (#11) |
| sampled result | gate met in 1 of 6 seeds, 4 Pass / 37 Partial / 19 Fail of 60, degeneration 0 of 60 (`decode_out/srv-c2-reader.log:301-312`, #44) |
| greedy result | gate met in 0 of 1 arms, 2 Pass / 5 Partial / 3 Fail of 10 (#11, Crow #192) |
| reference llama.cpp | 2 Pass / 6 Partial / 2 Fail of 10 on UD-Q2_K_XL greedy, gate met (#11, Crow #192) |
| decision | the default stays as built (#28 A6), the request decides: `sampler_from` at `engine/src/bin/serve.rs:1041` |
| request without `temperature` | greedy, the A4 path; `null` or `<= 0` is the same path (`engine/src/bin/serve.rs:133`) |
| request with `temperature > 0` | samples; absent fields take the data sheet `top_p` 0.8, `top_k` 20, `presence_penalty` 1.5, `seed` 0 reseeded per request (`engine/src/bin/serve.rs:134-137`) |
| what a Crow turn gets | sampled at `temperature` 1.0, `top_p` 0.95, `min_p` 0.01 accepted and ignored (`Crow cli/crow_core.py:472`, #28) |
| unmeasured | the ten-task gate at Crow's own profile (temperature 1.0, top_p 0.95); the six series ran temp 0.7, top_p 0.8, top_k 20, presence 1.5 (`engine/src/sample.rs:76-81`) |
| t2b-write-refactor | Fail under greedy after 21 ids (#11), Partial on 6 of 6 sampled seeds (`decode_out/srv-c2-reader.log:214`, #44) |

**The server-path unit tests the A11 gate names (existing since A2 to A9):**

| category | test | file:line |
|---|---|---|
| request parsing | `the_two_stream_flags_parse_out_of_the_body_crow_sends` | `engine/src/bin/serve.rs:3012` |
| request parsing | `tools_and_tool_turns_parse_out_of_the_body_crow_sends` | `engine/src/bin/serve.rs:3155` |
| SSE framing | `an_sse_frame_is_one_data_line_and_a_blank_line` | `engine/src/bin/serve.rs:3041` |
| prefix length determination | `common_prefix_stops_at_the_first_difference` | `engine/src/cache.rs:526` |
| prefix length determination | `the_newest_snapshot_at_or_below_l_wins` | `engine/src/cache.rs:558` |
| one slot per process | `the_process_holds_one_slot_and_one_reuse_candidate` | `engine/src/cache.rs:732` |

- Counts at commit 9054592: engine lib **79 of 79**, `bin/serve` **52 of 52**, every other
  binary 0 tests, doc-tests 0; converter **7 of 7**.
- Counts after M2b (#36): engine lib **80 of 80** (the new one-slot test), `bin/serve`
  **52 of 52**, every other binary 0 tests, doc-tests 0; converter untouched
  (`decode_out/srv-a9b.log:473`, `:488`).
- CI (E7, #50, 2026-09-11) runs engine lib **72 of 80** and `bin/serve` **55 of 57** on the
  windows-latest runner; the gap is 10 tokenizer tests that need `../models/`, not present on a
  fresh clone. The full counts above hold locally, where the models directory exists.

**The unit tests #39 added (`engine/src/bin/serve.rs`):**

| category | test | file:line |
|---|---|---|
| document builder | `the_non_streaming_document_carries_every_field_the_probe_suite_reads` | `engine/src/bin/serve.rs:3575` |
| document builder | `the_non_streaming_document_carries_the_tool_calls_the_parser_closed` | `engine/src/bin/serve.rs:3614` |
| sink equivalence | `the_collector_and_the_sse_sink_see_the_same_delta_sequence` | `engine/src/bin/serve.rs:3639` |
| request parsing | `the_two_callers_that_send_no_stream_field_parse_as_non_streaming` | `engine/src/bin/serve.rs:3705` |
| collector | `the_collector_holds_one_buffer_per_tool_call_index` | `engine/src/bin/serve.rs:3733` |

- Counts after B3a (#39): engine lib **80 of 80**, `bin/serve` **57 of 57** (the five new
  tests above), every other binary 0 tests, doc-tests 0; converter untouched. The warning
  set is byte-identical to the adb34b6 baseline `decode_out/srv-m2b-orch-tests.txt`
  (`decode_out/srv-b3a-tests.txt`).
- Commands: `cd engine && cargo test --release --target-dir target_srv`, and
  `cd converter && cargo test --release`.

**The parity rule after every rebuild (falls out of trap F44):**

| rule | reason |
|---|---|
| a rebuild of the same source has a different sha and is therefore **ungated** | the sha is what a gate names, not the source |
| parity runs **8, 512 and 1024** ids against the installed `engine/target/release/decode.exe` | three prompt lengths cross the chunk and selection regimes |
| the **512 run twice**, and the two candidate runs compared against each other | one green run does not prove the absence of a race |
| the candidate binary is built into its own target directory (`engine/target_srv/`) | never build under a running chain; `engine/target/release` is the reference and is not touched |
| the comparison is `cmp` of `gpu-logits.f32`, byte for byte | rows are `ids + 4`: 12, 516, 1028 |
| environment of both arms | `CROW_GRAPH=1`, `CROW_MMA=1`, `CROW_CNQ`, `CROW_HOTSETS` set, `CROW_ADAPT` unset |
| a green identity gate is separate | greedy ids of a warm run against a cold run, twice |

- Rule: the server work must not move the decode path, and "nothing in the kernels changed"
  is not evidence of that.
- Reason: the PLE bug of epic #1 stayed hidden for two days behind exactly that sentence.
