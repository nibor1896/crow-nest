# p5 status — RESOLVED (2026-09-02, v2 clean rewrite)

The v1 divergence does not exist in the clean rewrite. All three stages PASS:
- A (all combos expert 0, VRAM): max_abs 5.96e-8 — float-exact
- B (experts 0/4 alternating, VRAM + pinned): max_abs 9.2e-3
- C (8 tokens × full random routing, mixed residency): max_abs 1.465e-2
(all vs independent CPU dequant chain; bound 0.05; NaN=0)

v1 root cause: the probe itself — 15+ debug-patch rounds left shadowed buffer
declarations, duplicated launch blocks and stale debug paths. The v1 "GPU matches
EXPERT 2 instead of 3" reading was an artifact of that code state. Lesson recorded:
when a probe needs more than ~3 debugging rounds, rewrite it minimal instead of
patching.

## v2 timings (microbenchmarks, first-correct-then-fast)

- chain per layer (3 launches, 80 combos, 6/10 experts cold over PCIe): 5.8 ms
  (single expert) → 10.6 ms (full random routing) — ×48 layers ≈ 276–511 ms/token
  naive. ~40–100× over the per-layer budget: the #10 perf phase (tiling, fused
  launches, vectorized loads) has clear, measured headroom targets.

## What v2 proves for the engine

- expert FFN chain (gate_up → silu×up → down) numerics on sm_120a vs CPU dequant ✓
- mixed residency through per-combo device pointer tables — VRAM and pinned-host
  weights interchangeable, residency invisible to the kernel ✓
- zero-copy cold path: cold experts computed directly over PCIe from pinned RAM ✓
- WDDM rules in code: async+sync uploads only; scalars via device buffers ✓

---

# p6/p7/p8 (GDN + attention + full layer-0 GPU ports) — RESOLVED 2026-09-02

All three gates PASS against the transformers 5.16.1 reference goldens, real
BF16→f32 weights, T=8, sync-per-launch probe pipeline (microbenchmark only —
not an engine number):

- **p6 GDN sub-block** (`p6_gdn.rs` vs `oracle/golden/layer0-gdn-*.f32`):
  max_abs 3.576e-7, NaN=0, tol 5e-3. Two math gaps fixed while wiring: the
  q-scale 1/√128 and the `S *= exp(g)` decay were missing from the drafted
  kernels (`g_t` was a dead variable). `norm.weight` is [128] — one shared
  weight across all 48 v-heads (checkpoint shape, not [6144]).
- **p7 attention sub-block** (`p7_attention.rs`, layer 3, dense regime — QSA
  indexer selects all 8 visible tokens by construction at T=8): max_abs
  2.027e-6, NaN=0. Ported: q_proj 24×512 (query+gate chunk), RMSNorm with
  (1+w) delta weight, partial rotary 64 dims rotate_half-style (interleaved
  mrope degenerates to plain rope for text-only), GQA 12:1, scaling 1/√256,
  eager softmax f32, sigmoid output gate before o_proj. Golden exported by
  `oracle/export_attention_golden.py` (9 tensors, provenance in manifest).
- **p8 full layer-0** (`p8_full_layer.rs` vs `layer0-golden-output.f32`):
  max_abs 4.768e-7, NaN=0. Complete decoder layer: both hyper-connection
  blocks (grouped RMSNorm 4×2560 with (1+w), low-rank mix 10240→320→10240,
  mean over 4 streams, inject 2·sigmoid(·/4)), GDN chain, MoE — router GEMV +
  softmax f32 + top-10 on host (probe shortcut; engine moves it on-GPU),
  all 512 experts f32 in VRAM (10 GB, 3.7 s upload), shared expert.
  Composition debugged with staged dumps vs a torch re-run
  (`oracle/ref_layer0_stages.py`, `ref_layer0_moe.py`): everything matched to
  1e-6 except the shared expert — root cause was a one-character bug
  (`g/(1+e^-g)` = g·sigmoid(g) instead of `1/(1+e^-g)`) in gate_shared.
  Reference recomposition reproduces the golden exactly (max_abs 0.0).

## What this pins for the engine

- Kernel numerics for GDN, dense attention (dense regime) and the full layer
  composition are reference-verified on sm_120a — the f32 parity gate that
  section 4.3 demands BEFORE quantization is green for these families.
- p5 (expert FFN chain) + p8 close the MoE path end to end.
- Remaining kernel families for #10: FP4 versions (the f32 chain is the
  reference), QSA indexer, PLE gather, flash-style attention beyond the
  dense regime.

## Session lessons (both sessions)

- Past ~10 debug-patch rounds, rewrite beats patching (p5 v1→v2).
- When a composition gate fails, stage-wise dumps vs a torch re-run find the
  diverging stage in one pass — no kernel-level guessing (p8: 3 dumps →
  pinpointed gate_shared).
- Elementwise kernels must guard against the FULL flat buffer length, not the
  per-row length (p8 first run silently processed only token 0).

---

# p9 (job ring + stream memops, Windows first) — RESOLVED 2026-09-02

`p9_job_ring.rs`. Spec 3.2 mechanics + 3.6 acceptance, strand-2 questions answered
on driver 616.56 / WDDM:

**The memops verdict is two-tier (this corrected an interim misread):**
- 32-bit non-_v2 variants (`cuStreamWriteValue32/WaitValue32`, loaded from
  nvcuda.dll via libloading — cudarc only emits memop bindings for cuda-11.x
  features): host-mapped target → `CUDA_ERROR_NOT_SUPPORTED`; device target →
  ACCESS VIOLATION crash inside the driver. DO NOT USE on WDDM.
- **64-bit `_v2` variants (`cuStreamWriteValue64_v2`/`cuStreamWaitValue64_v2`,
  exposed by cudarc, same as handoff-bench): fully functional on WDDM.**
- Publication GPU→host: kernel-side mapped write into pinned DEVICEMAP ring +
  `__threadfence_system()` works and is how the descriptor lands (stream memops
  onto host-mapped memory are rejected, so the spec's "publication via
  cuStreamWriteValue32" wording needs the WDDM amendment: publish by kernel
  mapped write, release via the 64-v2 memops).

**Round-trip microbenchmark** (sync per rep; host Instant; per rep: params H2D,
producer kernel, publish kernel, then release + consumer + sync):
- baseline single launch+sync: p50 7.2 µs
- memops release (`WaitValue64_v2`, GEQ, device flag): **signaling p50 28.3 µs,
  p99 340 µs; with 1 MiB staged H2D p50 87.5 µs** — the copy mostly overlaps
- poll-consumer fallback (host volatile write into WRITE-COMBINED pinned done
  buffer, consumer kernel polls with `__nanosleep` backoff): p50 1710 µs —
  works everywhere but 61× slower; keep as portability fallback
- Reference: swapped barrier 4–6 ms (#186); retired variant-A ring 3.1–3.4 ms.

This ring is the CONTROL PLANE (residency swaps, tier transitions, telemetry);
the decode hot loop stays on zero-copy direct reads (spec 3.4, A′). Engine
consequence: use the 64-v2 memops for job completion, kernel-side mapped writes
for descriptor publication, and never touch the 32-bit non-_v2 entry points.

---

# p10 (full layer-0 in PRODUCTION FP4) + p11 (GDN decode step) — DONE 2026-09-02

**p10 (`p10_fp4_layer.rs`)** — the complete p8 pipeline, but every NVFP4 tensor
loads straight from the real CNQ4.5 container and dequantizes on the fly in the
GEMV kernels (p2 layouts, p4/p5 pattern); BF16 keeps load as f32; experts sit in
VRAM as FP4 slabs (1.35 GB total instead of 10 GB f32 — pointer offsets per
expert). Stage A: gemv_fp4 vs CPU dequant of the same bytes → rel 1.15e-7
(transport+numerics exact). Stage B is a MEASUREMENT vs the f32 golden:
max_abs 0.125, mean_abs 1.21e-2, rel_L2 1.73e-2, NaN=0 — the honest 4.5-bpw
RTN layer-level delta; quality judgment belongs to the ten-task gate (#11).
Converter container notes: section field separates `text` from the `mtp`
copy of layers.0; bf16 keeps carry dtype "bf16"; rows are contiguous
(offset deltas = bpr*36 exactly).

**p11 (`p11_gdn_step.rs`)** — the GDN decode step with PERSISTENT state, the
kernel shape the generation loop needs. Two phase design:
- prompt: batched p6 chain with `delta_rule_persist` (state lives in global
  memory, thread d owns column d, no shared memory, no atomics),
- stepping: the same tokens one at a time from zero state through
  `delta_rule_step` + `conv_step` (state shift + silu).
Gates vs new step goldens (`oracle/export_gdn_step_golden.py`; stepping ==
batched golden at 4.8e-7): per-step y 4.8e-7, core 8.4e-9, S stepping 1.8e-7,
S batched-persist 1.9e-7, conv_state 1.9e-6. PASS.

****p12 (`p12_attn_step.rs`)** — attention DECODE step with persistent KV cache
(mirror of p11 for the 12 full-attention layers). Per token: projections,
RMSNorm (1+w), rotary at position t (cos/sin table pointer offset — no kernel
change), K/V appended by plain DtoD row copies into [2 kv-heads][slots][256]
caches, `attn_step` dense attention over cache[0..=t] (GQA 12:1), gate, o_proj.
Steps 0..7 each match the p7 golden rows: max_abs 2.0e-6, NaN=0. PASS.
Cache slot sizing (T_max) is a probe constant — real sizing is #9's manager.

**Status toward #11:** all layer math is now reference-verified in BOTH shapes
(batched prompt + single-token decode with persistent state), at f32 and (p10)
at production FP4 precision for the GDN-layer path. Remaining before the first
generation loop: attention-layer FP4 (mechanical, p10 pattern), QSA indexer,
PLE gather, embeddings/lm_head/sampling wiring, then residency + scheduler.

Debug lesson (p10 prep):** when transcribing reference math, test-side
einsum index slips are silent — `"hkd,hd->hd"` contracts the state's k-axis
with the vector's d-axis (nonsense math, valid shapes). The GPU kernel was
right all along; the torch re-derivation was wrong. Cross-check hand-written
transcriptions against the library reference implementation
(torch_recurrent_gated_delta_rule) before debugging the kernel.

**Session lesson:** an access violation while "reading a result" was a device
pointer dereferenced as host memory; and a driver-level crash can masquerade as
a hang two steps later. Checkpoint-print around every suspicious call, never
`from_raw_parts` a CUdeviceptr, and read the handoff-bench history BEFORE
declaring a mechanism dead — the 64-v2 variants had already been proven there.

---

# p13 (f32 generation loop — first end-to-end model forward) — RESOLVED 2026-09-02

`p13_gen_loop.rs` + `oracle/ref_gen_logits.py`. The whole text model runs on
GPU: embedding (BF16-keep, host rows) → 48 decoder layers dispatched by
`layer_types` (36× GDN via p11 kernels, 12× attention via p12 kernels, every
4th; p8 HC+MoE composition per layer) → `hyper_connection_mixer` (the
model-level GatedResidual, use_combine=False — there is NO final norm in
qwen4_exp; the mixer collapses 4×2560 → 2560 directly before lm_head) →
lm_head (untied, BF16-keep) → greedy argmax.

**Gate: PASS.** Prompt = tokenizer("The quick brown fox jumps over the lazy")
= [760, 3841, 13477, 37550, 33075, 888, 279, 15217]; 8 prompt positions
batched + 4 decode steps with persistent state. GPU logits [12][248320] vs
the full 48-layer f32 reference forward on CPU (transformers 5.16.1, eager,
real weights, layer-by-layer streaming): **max_abs ≤ 2.74e-5 across all 12
positions** (tol 5e-3 — ~180× margin), **greedy argmax match 12/12**, NaN=0.
Worst ref top-2 margin was 0.0167 (pos 10) — argmax safe by ~3 orders.
Trace: 5388, 13, 561, 6561 (+ candidate 17629 after the last processed
position). Artifacts: `probes/p13debug/{gen-sequence.json, gpu-logits.f32,
ref-logits.f32}`, logs `probes/p13_run.log`, `probes/p13_gate.log`.

**f32 composition is stable across 48 layers**: sub-block gates were
1e-7..2e-6; end-to-end through the full model (HC + MoE + GDN/attention +
mixer + lm_head) it accumulates to only 2.7e-5 on logits.

**Probe shortcuts (documented, both sides consistent):**
- PLE (layers.1) skipped on BOTH sides — gated separately with the PLE gather.
- QSA dense by construction (T=12 < 2048).
- Router softmax/top-10 on host (p8 pattern).
- **Layer weights are streamed per pass** from the safetensors shards:
  raw BF16 HtoD → on-GPU `bf16_to_f32` kernel (exact bit shift — cheaper and
  simpler than host conversion; a 10 GB f32 model layer never fits 32 GiB
  alongside anything else, so all 48 layers stream, ~4 s/layer ≈ 186 s/pass,
  5 passes ≈ 16 min total). This is a numerics probe, NOT the engine's
  residency design (fp4 + per-layer hot sets replace it, #8/#9).
- Timings are stream-UI numbers, no engine claim.

**Kernel-shape additions over p11/p12:** `attn_step` generalized (cache
stride via device i32 `tmax` param instead of hardcoded 8), `delta_rule_persist`
feeds the exact state buffers the decode steps consume (36 GDN S-states +
conv-states, 12 KV caches [2][12][256]), model-level mixer shares the
rms_group/mix machinery with the layer-internal HC blocks.

---

# Next (session 5): QSA indexer → PLE gather (layers.1, config `ple_layer_ids:
# [2]` is 1-based) → attention-layer FP4 (p10 pattern) → #8/#9.

---

# p14 (QSA indexer kernel family, sparse regime) — RESOLVED 2026-09-02

`p14_qsa_indexer.rs` + `oracle/export_qsa_golden.py` (+ `ref_qsa_stages.py`
comparator). The Qwen4ExpTextQSAIndexer (layer 3, real weights) runs on GPU in
the genuinely SPARSE regime: T=2560 → 640 complete blocks, block_topk 512 →
128 blocks dropped per query (below the 2048-token budget the indexer is dense
by construction — that free gate was p13's).

**Math ported** (modeling L611-717): index_qk_proj 2560→(4+1)·128 → q RMSNorm
(1+w) per head → partial rotary 64 dims (theta 1e7) at position t; per block
b: pooled = rotary(k_layernorm(mean(raw_keys[4b..4b+4])), pos 4b) — raw keys
are UNNORMALIZED, the norm hits the pooled mean; score[b] = Σ_h relu(Σ_d
q[t,h,d]·pooled[b,d])/√128 (relu per head BEFORE summing); topk(min(512,ncb))
in score order + tail tokens always selected; width ≤ 2051 = budget + ratio −1.

**Gate: PASS.** Block scores gpu vs golden max_abs 3.695e-5 (tol 5e-3).
Selection, tie-aware classification per query: 1276 identical lists, 1265
order-only swaps (same set — mask-identical, scatter order is irrelevant),
19 tie-swaps across the topk boundary with score gaps ≤ 1e-3 (f32 noise,
score max_abs 3.7e-5), **0 hard failures**. GPU kernels: 9.1 ms for the whole
T=2560 index pass (sync-per-launch probe number).

**Debug lesson (cost 3 rounds — do not repeat):** the "contiguous slice"
assumption failed TWICE: k columns (offset 512) and q columns (offset 0) live
STRIDED inside the [T][640] qk matrix, but pool4/rms128/rope64 first read them
as compact [T][128]/[T][512] arrays. Rule: any kernel reading a slice of a
strided matrix takes an explicit stride parameter (or the slice is copied
compact first). Found by the p8 method — stage-wise dumps vs torch re-run
(`ref_qsa_stages.py`): qk green, q-normed 5.6 red → stride, no guessing.

**Reference-side findings (export script):** the real class returns only the
additive mask (0 / finfo.min — NOT -inf; `> -inf` matches everything!). The
index-level golden is built by a torch.topk rebuild of the reference loop,
ASSERTED set-identical to the class mask for all 2560 queries — the rebuild is
pinned to the real path before serving as the order-carrying golden.

Probe shortcut: top-k on host from GPU scores (torch.topk order semantics); a
GPU topk primitive comes with the #10 perf phase. Everything else — proj,
norms, rotary, pooling, scores — is on GPU.

---

# Next: PLE gather (layers.1) → attention-layer FP4 (p10 pattern) → #8/#9.

---

# p15 (PLE gather + gating + dilated conv) — RESOLVED 2026-09-02

`p15_ple.rs` + `oracle/export_ple_golden.py`. The PLELayer of layer index 1
(config `ple_layer_ids [2]`, 1-based), real weights, f32, input = the p13
12-token sequence, no cache (n-gram history EOS-filled).

**Split of work (engine design, spec 3.5):** HOST does the n-gram index math —
shift-right-ignore-EOS history, per-position multiplier XOR (wrapping i64 =
torch int64 bit pattern; remainder = rem_euclid), mod by the checkpoint's
stored prime head vocab sizes + offsets (the three I64 tables — nothing about
the hashing is re-derived). GPU does: row gather from a compact resident cache
(the 102 GB table never loads; host remaps ids → cache slots = the hot-row
pattern), key/value projections (2560→10240 / 2560→2560), 3 grouped RMSNorms
(1+w), gate = Σ(key·query)/√2560 then √|g|·sign (clamp 1e-6), sigmoid
broadcast over the 4 streams, norm_conv, dilated depthwise conv (kernel 4,
dilation 3, left state (k−1)·dil = 9) + silu, final add onto the HC stream.

**Gate: PASS.** ngram ids host (Rust) vs golden: **0/192 mismatches (exact)**.
Embeddings gpu vs golden: **bit-exact** (same BF16→f32 bytes). Layer output
max_abs **1.192e-7** (tol 5e-3), NaN=0.

**Facts pinned:** PLE shards are [2500012, 160] each, 128 of them = 320,001,536
rows = padded total vocab (Σ16 primes ≈ 320.0M after 20M base); row id → shard
= divmod 2,500,012. key_proj [10240, 2560], value_proj [2560, 2560], conv1d
[10240, 1, 4]. Reading many mmap'd 2.7 GB safetensors files via safe_open dies
SILENTLY on Windows after ~70 handles — the export reads rows with raw file
seeks (header-offset + row×320 B) instead; the Rust probe does the same.

**Debug lesson (cost 1 round):** a kernel writing a [·][2560] row from 256
threads needs the stride loop — the first gate_apply version wrote only the
first 256 columns (90 % of the buffer silently zero, ratio-0 pattern). Same
family as the p8 full-flat guard rule: block-stride loops or guards must cover
the FULL row.

---

# Remaining before #11: attention-layer FP4 (mechanical, p10 pattern) → #8/#9.
Session handoff: p13 ✓ p14 ✓ p15 ✓ — generation loop, QSA indexer, PLE all
reference-verified; artifacts in probes/p13debug, p14debug, p15debug + logs.

---

# p16 (attention sub-block at PRODUCTION FP4) — RESOLVED 2026-09-02

`p16_fp4_attention.rs`. The p10 pattern applied to the 12 full-attention
layers: layer 3's q_proj/k_proj/v_proj/o_proj read straight from the CNQ4.5
container (section `text`, NVFP4, dequant on the fly in the GEMVs; q_norm/
k_norm BF16 keeps → f32), activations f32, both run shapes.

**Gates (green):** Stage A gemv_fp4 vs CPU dequant of the same bytes:
max_abs 2.86e-6, rel 2.0e-4 (near-zero values excluded from the ratio).
Stage B' on-the-fly FP4 chain vs pre-dequantized f32 chain: **max_abs
1.07e-6** — the FP4 GEMVs are equivalent to a plain f32 GEMM of the same
dequantized weights.

**Measurement (not a gate):** FP4 vs the f32 golden: **rel_L2 0.165,
max_abs 0.582, mean 8.6e-2, NaN=0** (batched T=8 AND all 8 decode steps
identical). That is ~10× the p10 whole-layer GDN delta (rel_L2 1.7e-2) —
**the attention projections are the most NVFP4-RTN-sensitive spot measured so
far** (rope + softmax sharpen the q/k error). Consequence carried to #11: if
the ten-task gate flags attention quality, the lever is CNQ4.5-C calibration
(stage 2, scales only) or promoting q/k to BF16 keeps (format already
supports per-tensor keeps). No decision now — measurement recorded.

**Debug lessons (3 rounds):** (1) the classic arg-swap — fp4_gemv_row passed
(y, gs) where the kernel reads (gs, y): GPU output was all zeros because the
kernel read gs=0 from the zeroed y buffer; caught immediately by the stage-A
self-check, which is exactly its job. (2) A "max rel" gate explodes on
near-zero CPU references — always report abs alongside rel. (3) An
ILLEGAL_ADDRESS in the f32 reference chain was the dequant buffers sized
k_dim×4 instead of rows×k_dim (gemv_f32 read rows beyond a 1-row buffer) —
buffer sizing must derive from the FULL tensor shape, not the row length.

#10 scoreboard: attention FP4 ✓ (measured) — all kernel families have
production-precision evidence. Next: #8 residency scheduler + #9 three-state
manager, then #11.


---

# ENGINE (dev/crow-nest/engine) — #8 residency + #9 three-state manager + #11 first production decode — Session 6, 2026-09-03

The engine crate assembles the probe-verified kernel families into the real
system: `engine/` (cudarc 0.19.9, NVRTC compute_120a, raw sys API). Bins:
`states` (#9 demo), `residency` (#8 demo), `decode` (parity/run), `parity`
(llama.cpp A/B skeleton), `kcheck` (fast NVRTC + FP8 validation).

## kcheck (fast gates, both PASS)

- All 49 kernels compile via NVRTC at `compute_120a` and resolve.
- FP8 E4M3: device enc/dec vs the Rust twin (engine/cnq.rs) bit-exact over
  17,122 sampled values (sweep −620..620 step 0.1, all exponent/mantissa
  lattice points, random bit patterns, boundary specials); decode delta 0.
  Encoder = RNE + satfinite at 448 (boundary 464), subnormals step 2^-9.

## #9 three-state manager (`manager.rs`, bin `states`)

- State kinds: KV (12 layers × 2 kv-heads × 256 × ctx × {FP8=1 B | BF16=2 B}),
  QSA indexer key cache (12 × ctx × 128 f32 — FULL-LENGTH, see finding below),
  QSA pooled-block cache (12 × ceil(ctx/4) × 128 f32), GDN S-state
  (36 × 48×128×128 f32) + conv state (36 × 10240×3 f32), RoPE tables.
- **Design finding (reference-pinned):** the QSA "budget 2048" is the SELECTION
  budget (512 blocks × 4 tokens); the indexer key cache itself is full-length
  [ctx][128] per layer (transformers `StaticIndexedLayer.update_indexer`
  appends in place). At 262k that is 1.5 GiB — the spec's ~0.3 GB assumption
  was wrong; honest budget table:
  | point | states | dense (index) | hot 160 | total / 32 GB |
  |---|---|---|---|---|
  | 262k FP8 | 4.86 GiB | 3.20 GiB | 19.8 GiB | 27.9 GiB |
  | 262k BF16 | 7.88 GiB | 3.20 GiB | 19.8 GiB | 30.9 GiB |
  | 200k FP8 | 3.70 GiB | 3.20 GiB | 19.8 GiB | 26.7 GiB |
  | 200k BF16 | 5.72 GiB | 3.20 GiB | 19.8 GiB | 28.5 GiB |
- Loader rule VERIFIED: budget check runs against measured `cuMemGetInfo`
  deltas; forcing N=192 auto-clamps to 174; context < 200k REFUSES (spec 2.1/2.6).
  Loader never clamps context.
- **Two-sided N clamp (new, measured):** the pinned cold tier has a HOST
  ceiling (~48.5 GB observed on this machine; 64 GB RAM). The loader now
  clamps N DOWN for VRAM and UP for the host pinned budget (default 46 GiB
  config, `host_pinned_budget`); if the feasible window is empty it refuses
  (spec 2.1: never silently degrade context).

## #8 residency scheduler (`residency.rs`, bin `residency`) — all PASS

- Warm-up → per-expert GPU selection counters (`router_top10` counts every
  routed choice, hit or miss, spec 2.2) → top-N per layer → sidecar
  `<cnq>.hotsets.json` persisted next to the container → reload from sidecar.
- Hot experts in VRAM slabs (160 × 48 × 2.64 MB = 20.25 GiB), cold experts in
  pinned host (~44.6 GiB) read ZERO-COPY inside the FP4 GEMV via device
  pointer tables (`gemv_fp4_ptrb`) — residency invisible to the kernel (p5).
- Routing-gated skip on GPU: per-layer u32 bitmap tested inside `router_top10`;
  per-layer u64 counters [selections, cold] drained by the control plane
  between tokens (job ring's control-plane role, spec 3.4; hot path has NO
  host sync per layer — structural, decode loop queues all kernels then syncs
  once at argmax).
- Held-out measurement (demo traffic, word-hash tokenization — real Crow
  traffic warm-up is the standing-series step): 480 selections/token exact,
  cold experts 65–92/token, cold bytes ≈ 171–243 MB/token ≈ 7–10 ms at
  21.6–25.1 GB/s, 26–40 of 48 layers fully resident (no cold work at all),
  hot-set coverage 96.1 % on held-out routing.
- Decode step wall-clock ≈ 95 ms (naive GEMV kernels, sync-per-launch debug
  off, single-token submission) — the #10 perf phase (MMA, fused MoE) owns
  the gap to 42 tok/s; no scheduler-side stall is on the critical path.

## #11 first end-to-end PRODUCTION decode (`gen.rs`, bin `decode`)

- Full graph at production precision: embeddings (BF16 keep, host rows),
  PLE at layer 1 (host n-gram index math per spec 3.5 + direct-mapped NVFP4
  row cache, 112 B/slot incl. per-slot global scale), 36 GDN + 12 attention
  layers (FP4 weights, QSA indexer with full-length key cache + incremental
  block pooling + exact GPU top-k block selection), hyper-connections,
  router top-10 on GPU with residency pointers, MoE with shared expert +
  zero-copy cold experts, model-level mixer, untied BF16 lm_head, greedy
  argmax. FP8 E4M3 KV cache with on-store cast; BF16-KV is a config switch.
- New kernels this stage: exact GPU top-k (radix refine on ordered f32 keys,
  tie-safe lowest-index fill, deterministic ascending token list), FP8
  cast/decode, pointer-table batched FP4 GEMV, list-based attention
  (`attn_sel`), GPU router (softmax+top10+bitmap+pointers+counters), PLE
  NVFP4 row gather, cross-chunk conv states (GDN + PLE), GPU transpose/split
  (removes p13's host roundtrips).
- First full run (8-token p13 prompt + 4 decode steps, 12 positions × 248320
  logits): NaN=0, no faults, greedy trace continues plausibly. Oracle f32
  comparison (with PLE reference per p15) = the standing gate.
- Latency at this stage: ~95–107 ms/token (see #8 note — perf is #10's phase).

## Debug lessons (session 6 — all found by the p8 stage-dump method)

1. **Shared-expert layout:** `silu_mul640` assumes gate|up in ONE [t][1280]
   buffer (p13 layout). Separate [t][640] gate/up buffers made the kernel read
   out of bounds from token 4 on → NaN rows 4-7 → NaN through the whole model.
   Fix: combined sh12 buffer, gate at offset 0, up at offset INTER.
2. **KV cache offset:** `layer_cache_ptrs` must index by ATTENTION index
   (ai = layer/4), not raw layer id — layers ≥ 15 read/wrote up to 1.4 GB past
   the KV buffer (silent corruption, then faults). With the raw id, layers
   7..11 also aliased OTHER layers' caches (wrong-but-finite → silent nonsense).
3. **qsa_select smem bitmap** must be zeroed per launch (stale bits = phantom
   selected blocks → attention reads far future tokens).
4. **Warp-uniform popc must not be warp-reduced** — all 32 lanes scanning the
   same words then `__shfl_down`-reducing multiplies the count by 32
   (selection counts 32× → tail-offset corruption).
5. **PLE n-gram rows:** the p15 host index math returns prefix+chunk rows —
   slice to the chunk rows BEFORE uploading (buffer overflow otherwise) and
   the slots must correspond to chunk positions.
6. **Budget double-count:** `cuMemGetInfo` free-at-load already excludes the
   resident dense weights — plan only counts NOT-yet-resident bytes.
7. **Pinned host ceiling:** ~48.5 GB observed; the cold tier must be part of
   the loader's feasibility check, not an afterthought (ALREADY_MAPPED/OOM
   at 49.6 GB pinned).
8. **Two processes, one RAM:** pinned cold tier + oracle python (f32 layers)
   collide — sequence GPU loads and oracle runs, or shrink the pinned budget.

## Open for #11

- Oracle end-to-end logit gate (running), ten-task gate vs llama.cpp
  (docs/ten-tasks.md fixation; harness skeleton in `parity`),
  long-context QSA-sparse decode run, 200k-floor standing series.

## Gate closure (same session, later): argmax 0/12 — root cause isolated, engine verified correct

- Oracle end-to-end gate (f32 reference WITH PLE, `ref_engine_logits.py`):
  argmax 0/12, logit delta 9–25. Bisectors: PLE off / BF16-KV → no change.
- `layercheck` + stage dumps localized the error; harness bug found first
  (debug entry missed the `nt_low` refresh — silu computed 1 element), then:
- **KHC container** (robin GO ①+②: BF16 keeps for HC down/up + attention q/k,
  converter amendment 2026-09-03; 843 nvfp4 / 815 keeps, 104.73 GB, 0
  violations): HC stage now EXACT (mixed-a 1.9e-6, low 1.5e-5 vs torch),
  GDN 0.103 (FP4-plausible), **MoE 0.964** — the routed FP4 experts inject
  ~O(1) noise per layer; 48 additive layers ≈ the observed ~20 logit units.
- End-to-end on KHC: unchanged (0/12) — as expected, the experts dominate.
- Engine-side fixes verified en route: shared-expert layout, KV offsets by
  attention index, qsa_select bitmap/popc, PLE row slicing, budget
  double-count, two-sided N clamp, sidecar N adaptation.
- Decision menu posted on #11: CNQ4.5-C / shared-expert keeps / error-budget
  study / empirical ten-task first (recommended — llama.cpp baseline is
  ~2-bit experts, the product gate is ten-task, not f32 parity).

---

# SESSION 7 (2026-09-03) — ten-task harness complete + THE MEASURED RESULT: both CNQ4.5 containers collapse greedy generation

## Harness (engine/src/bin/parity.rs) — complete and exercised

- Real HTTP client (the skeleton posted the body without headers), answer
  capture (crow: greedy ids → batch detokenize via `tools/detokenize_ids.py`),
  and a new `phase` mode: arm-phased execution over the full rotated order.
  Reason: the engine's pinned tier (~44.6 GB RAM, ~29 GB VRAM) and the
  llama-server model (~50 GB page cache, ~12 GB VRAM) cannot coexist on 64 GB
  / 32 GB — the loader would clamp/refuse. The `run` (interleaved) mode stays
  for a machine that fits both. Rotation / first-mover swap / cold-prefill
  discard rules implemented exactly as fixed (robin 2026-09-03).
- Series file `decode_out/ten-tasks.json`: 6 hash-pinned seeds + t7–t10
  authored per the fixed classes (`decode_out/ten-tasks-authored-t7-t10.json`;
  t7 embeds the full `engine/src/kernels.rs` as >8k-token material). Robin can
  replace any of the four and re-run. Reasoning-aware budgets 640–896.
- llama arm served via **CHAT endpoint with `enable_thinking:false`** — raw
  `/completion` makes this chat-trained model emit EOS mid-think (measured:
  4/5 long tasks returned empty content); chat is the baseline's pinned
  operating mode (`--jinja`). Answer-quality rule for evaluation: answer =
  text after the last `</think>` (no think text appeared under the flag).
- Costly lesson (documented on #11): re-running the same prompts in the same
  server session hits llama.cpp's prompt cache (prompt eval = 4 tokens) and
  contaminated one answer — one series per server session, fresh start.

## Befund A — the ten-task gate verdict: FAIL for CNQ4.5-BASE and KHC (measured)

64-step greedy trace, demo prompt ("The quick brown fox jumps over the lazy"):

| model | trace |
|---|---|
| f32 oracle (p13) | `5388, 13, 561, 6561, …` plausible |
| CNQ4.5 BASE | `82, 19, 48652, 70, 12, 636, 12, 70, …` period-3 loop from ~token 5 |
| CNQ4.5 KHC | `11, 77, 9828, 26, 17, 25, 17, 17, 15, 15, 15, …` |

Ten-task smoke (t6/t2, 256 tokens): crow answers are repetition debris
("h h h h…"); llama.cpp UD-Q2_K_XL under the identical greedy condition stays
coherent (its 64-step trace: " dog. Write the above sentence…", 32.8 tok/s).

Consistent with the session-6 oracle: argmax 0/12 from position 1, logit
delta 9–25 units, MoE-stage error 0.964/layer adding to ~20 units over 48
layers. Unquantized f32 pipeline verified plausibility only to position 12 —
a hypothetical multi-step state bug beyond position 12 was never excluded;
the decisive falsifier (f32 GPU reference over 64 steps or CPU oracle over
20–30 positions) is proposed on #11, second opinion requested by robin.

## Befund B — llama.cpp reference side complete: 10/10 usable answers

Run 0 (unrotated order), chat mode, temp 0: all ten tasks return full,
plausible answers (decode 31–36.5 tok/s, prefill 204–777 tok/s, 200k floor,
-np 1). First quality reading: t6-reason strong (persistent segment tree +
last-occurrence transform — the canonical solution); t1-read structured with
a hedge; t3-debug plausible against its harness-encoding scenario. Detailed
acceptance per task is robin's call (parity-expected notes not yet written).
Speed side-by-side: crow ~11 tok/s prefill / 7–9 tok/s decode (naive GEMVs,
#10's phase) — quality is the blocker, speed is planned.

## Befund C — data-sheet sampling facts (robin's check, honored)

README + generation_config.json of the checkpoint: thinking mode temp 1.0 /
top_p 0.95 / top_k 20; instruct mode temp 0.7 / top_p 0.80 / top_k 20 +
presence_penalty 1.5 ("to reduce endless repetition"); do_sample true.
Crow's engine has NO sampling layer — pure argmax (gate discipline, both
arms greedy; the baseline stays coherent under it, so the comparison is
valid and greedy is the sharpest degeneration detector). PRODUCTION gap:
sampling layer with the data-sheet profiles + EOS stop (248046/248044; the
engine currently always runs its full max_tokens) = new ticket, not part of
the gate.

## Langkontext-Mechanik (QSA sparse, >2048): PASS

`decode run longctx_ids.json 12` (2100 real prompt ids, BASE): prefill
through the sparse QSA regime (selection budget exceeded) in 202.9 s
(10 tok/s), 12 decode steps at 138 ms (7.2 tok/s) at context 2112 — no
faults, no NaN, KV slots and indexer pooling behave past the budget. Trace
degenerates as expected (BASE). Note: the per-token cold-expert telemetry
reads 40260 in prefill mode — counter semantics across chunks need a look
(engine-side bookkeeping, not a correctness gate).

## Where this leaves the decision (robin's call, data on #11)

The measured quality position: CNQ4.5 RTN collapses generation on both
containers while 2.4-bpw *calibrated* GGUF stays coherent — calibration beats
bit width on this model. Strongest lever: **CNQ4.5-C** (stage-2 calibrated
scales, format and engine unchanged) with a fast iteration loop (12-position
argmax gate + 64-step greedy trace as the metric, minutes per attempt), then
ten-task re-run once traces stay plausible; error-budget keeps study as the
second lever if C alone is insufficient. Second opinion (fable5.1) requested
by robin on the quant-vs-state-bug question before spending the conversion
run — status + problem documented in the #11 comment (2026-09-03).

---

# SESSION 7 continuation (2026-09-03, same day): second opinion VERIFIED — the "quant" was FOUR ENGINE BUGS; fixed and re-measured

## Fable 5.1's findings, all four verified in code and fixed (robin GO)

| # | bug | place | fix |
|---|-----|-------|-----|
| C | `gemv_fp4_ptrb` addressed blocks without the row offset (`w + b*36`), so every output row was row 0's dot product — routed experts carried no signal | kernels.rs:199 | `rowp = w + (size_t)row * bpr * 36` (the p10-verified `gemv_fp4` pattern) |
| D | `acc_combo` did `y += rw*x` from 10 blocks on the same addresses without atomics — lost updates, per-warp variance | kernels.rs acc_combo | deterministic fixed-order reduction per token in ONE launch (`for j<10` sequential accumulate) |
| B | shared expert gate/up written with stride 640 into one buffer while `silu_mul640` reads the `[t][1280]` p13 layout — token 0 right, tokens 1-3 got up-as-gate, tail zeros; decode (t=1) unaffected — why decode looked fine and prefill didn't | gen.rs:1147 | new `gemv_fp4_bs` kernel with explicit y-stride (p14 lesson: never derive a slice stride from the grid); plain `gemv_fp4_b` untouched (14 other call sites stay `[t][rows]`) |
| A | `gs_dev` held layer 0's two global scales but `moe_run` used it for all 48 layers (sidecar ratio range 0.42..1.77 vs layer 0); the slab assert checked only BYTES, never scales | residency.rs:78/236 | `gs_dev` is now `[LAYERS][2]` f32, filled per layer; `moe_run` indexes `l*8` / `l*8+4` |
| — | sidecar writer sorted sets by ID before persisting → later N-truncate (two-sided clamp) kept the LOWEST ids hot, not the hottest (KHC N=160→148 today: 192.8 cold experts/token vs 133.8 on the truncated run) | residency.rs:131 | writer keeps frequency order; extend path no longer sorts |

## Layer-0 regression gate (the cheapest falsifier, p10 as the reference)

`layercheck` max_abs vs the p10 golden (0.125 = genuine FP4 RTN noise):

| state | BASE | KHC |
|---|---|---|
| before fixes | 1.189 | 1.18 |
| +C | 1.052 | — |
| +C+D | 0.993 | — |
| +C+D+B+A (final) | **0.1249** | **0.1206** |

The whole pre-fix surplus was the four bugs. Layer 0 now reproduces the
probe-crate FP4 result — the engine crate is, at layer 0, equivalent to the
p10-verified numerics.

## End-to-end after the fixes (12-position teacher-forced + 64-step greedy)

- teacher-forced argmax vs f32 reference (with PLE): BASE 3/12 → **was 0/12
  before**; KHC 6/12 → 5/12 on the final build. Misses are small-margin flips
  (0.15–3.1) plus EOS-attraction from position 8 on — additive RTN noise, not
  signal loss (the pre-fix distribution had entropy near the maximum; that
  signature is gone).
- 64-step greedy trace: **BASE is coherent again** — it recites the prompt
  ("The quick brown fox jumps over the lazy" → 5388 " dog"), emits EOS, then
  fragments with visible repetition-neigbourhood noise, but no period-loop
  collapse (`…, 561, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 13,
  248046, …`). KHC still loops (`271, 262, 220` period) even with the full
  N=160 sidecar — the KHC keeps (attention q/k BF16, layers 3+, never
  individually verified in the engine crate; HC keeps are layer-0-verified)
  are now the prime suspect, NOT the hot-set set (verified by sidecar swap).
- fable's framing confirmed: the KHC keeps were an answer to these bugs, not
  to quant noise. BASE is the reference container again; KHC needs either a
  q/k-BF16-path diagnosis (attention-layer engine check vs `oracle/golden/
  layer3-attn-*`) or removal.

## Consequence for the decision queue

1. Quant-CNQ4.5-C stays plausible but is NOT yet justified as the next spend:
   first the KHC question must be settled (q/k-keep path verified or dropped),
   because a KHC-specific bug would poison any quality comparison.
2. The ten-task gate was re-measured on BASE (crow arm run 0, this session):
   results in `decode_out/parity-run0-crow.json` + crow-run0.log (see #11
   comment for the verdict).

## Process notes

- The four bugs were found by a second opinion (fable 5.1, systematic
  debugging + fake-quant per-stage correlation/entropy metrics) and
  confirmed by code inspection before fixing — the "quant" reading had been
  sustained by absolute thresholds (0.964/layer at small magnitude) and by
  the layercheck 1.18-vs-p10-0.125 signal being read as quant instead of as
  the bug it was (more BF16 keeps, same error = cannot be quant).
- Stage-gate thresholds should be RELATIVE (rel_L2 + corr vs a fake-quant
  reference), never absolute — adopted as a rule.

---

# SESSION 8 (2026-09-03 evening): CNQ4.5-M container + MMA kernel — the two quality/speed levers, measured

Parallel subagent wave (5 agents): converter MSE mode, KHC q/k diagnosis +
sidecar regeneration, evaluation protocol, ~4-bit GGUF sourcing (stopped —
robin downloads himself), MMA perf sprint.

## CNQ4.5-M (weight-MSE scales) — converter, 87 min full run

`converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`: 1658 tensors (843 nvfp4 / 815
keeps), 104.73 GB, `--scales mse`. Subblock scales = MSE-optimal (analytic
pre-selection + 3 explicit SSE evals; the ceiling candidate stays in the
candidate set, so MSE_neu ≤ MSE_ceil per subblock BY CONSTRUCTION). clipped elements are the MECHANISM, and the honest numbers (sidecar
`max_abs_clipped`, not the old rel-bound counter that read 9,178,206):
text 2,680,721,843 (2.17 %), ple 1,084,508,500 (2.12 %), mtp 55.9 M,
vit 10.1 M — max_rel_err per tensor 1.59 (not 1.00); per-subblock MSE ratio
0.778–0.787. Interpretation warning (fable gate): weight-MSE is not the
model loss — clipping removes the largest weights of a sub-block, which
carry disproportionate information; whether M beats BASE on QUALITY is
exactly what the BASE-control ladder measures. Shard-1 ratio 0.787 (vit) / 0.792 (text). Name is deliberately -M
(weight-MSE), -C stays reserved for the spec's stage-2 traffic calibration.
Known follow-up: the ue4m3 encoder can still emit 0x7F bytes (hardware NaN,
see below) — engine-side sanitize covers it; converter should saturate at
0x7E in a future run.

## MMA kernel (`gemv_fp4_mma`) — 10.9× decode / 53.7× prefill on the routed MoE GEMV path

- p2-proven instruction (`m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3`,
  scale_vec::4X), A-operand = 16 weight rows straight from the 36-byte
  container blocks (no repacking), B-operand = quantized activation, expert
  pointers from the residency table (VRAM or pinned UVA — zero-copy cold
  path unchanged). Env `CROW_MMA=1` (default off, naive path stays as
  fallback/reference).
- NEW pinned SF fragment maps (differential probe, dyadic, delta exactly 0):
  sf_a lane L carries row (L>>2)+8·(L&3), only L&3<2 counts; sf_b lane L
  carries column L>>2, only L&3==0.
- Activations: `quant_x_fp4` with 3-level residual cascade per sub-block
  (each level ~10× error reduction: 1-level 10 % → 2-level 1.0 % → 3-level
  0.12 % rel_L2; +1 MMA per k-block, weight fragments reused).
- **Container finding: occasional ue4m3 scale byte 0x7F — the HARDWARE
  decodes 0x7F as NaN (E4M3 NaN code; 0x7E=448 is the real maximum). The
  naive kernel silently decoded it as 448 — a software/hardware decode
  mismatch that never showed until the MMA path. Layer 9 down contains
  exactly 1 such byte.** Fix: `sanitize_sf_slab` in residency.rs clamps
  0x7F→0x7E at slab materialization (all 4 copy sites, logged); the
  activation quantizer saturates at 0x7E too.
- Numerics: dyadic tile check delta exactly 0.0; quant kernel vs Rust twin
  byte-exact; MMA vs CPU-reference of the quantized chain rel_L2 8.3e-7;
  layercheck BASE 0.1249 unchanged; 12-position parity 10/12 identical,
  2 improved (position 10 now matches the f32 reference argmax).
- Perf: routed MoE GEMV 0.653→0.060 ms/layer (t=1) and 1283.7→23.9 ms/layer
  (t=2048); end-to-end decode 127.8→96.3 ms/token (1.33×) — the remaining
  ~96 ms are the dense FP4 GEMVs (next #10 lever), launch fusion, and the
  bandwidth-bound cold path (~7–10 ms/token).

## Container × kernel matrix (12-position argmax vs f32, teacher-forced)

| stand | argmax | layercheck |
|---|---|---|
| BASE pre-fix | 0/12 | 1.189 |
| BASE post-fix | 3/12 | 0.1249 |
| KHC post-fix | 5–6/12 | 0.1206 |
| **CNQ4.5-M + MMA** | **7/12** | **0.0915 — better than the p10 mark** |

KHC loop closed as container-specific attractor (bit-identical across three
sidecar configurations; q/k-BF16 path verified CLEAN via new `layercheck3`
vs the p7 goldens: KHC rel_L2 0.1430 BETTER than BASE 0.1683, both shapes).
Whether KHC or M (or M-KHC) rides the series: the ten-task run itself
decides with 10 diverse prompts instead of one chaotic trace.

## Next (waiting on fable gate, then the series)

1. llama UD-Q2_K_XL Rev2 re-run (budgets 1024–1536; the old 10/10 was under
   clipped budgets), ~10 min.
2. crow CNQ4.5-M + MMA, run 0 + run 1 (first-mover swap), incremental
   reports, ~2.5–3 h each until the dense GEMV MMA lands.
3. llama UD-IQ4_XS (4.25 bpw) — quant-vs-quant arm, robin downloading.
4. Evaluation per docs/ten-task-expected.md (rev2 noted; protocol found two
   agent errors: t7 question-2 false premise, 728.25 MiB margin — fixed).
Series start is FROZEN until robin's fable gate returns.

---

# SESSION 7 final: ten-task gate RE-MEASURED on post-fix BASE — model speaks, degenerates at token level; CNQ4.5-C is now the data-backed next measure

## Crow arm, run 0 (BASE, rotated order, budgets 640–896, temp 0)

`decode_out/parity-run0-crow.json` (+ crow-run0.log). Wall 2 h 47 min; prefill
6.0–6.5 tok/s, decode 4.0–6.5 tok/s (naive kernels — the #10 perf numbers,
longest task t7/t1b-read-lang: 54 min for ~13.5k prompt tokens through the
QSA-sparse regime, the first measured long-prompt prefill data point).

**Answer quality (vs the 10/10 llama reference):**
- 10/10 tasks produce on-topic answers — vs 0/10 repetition debris pre-fix.
- 2/10 start at reference level and stay usable for most of the budget:
  t1b-read-lang (names the right FP4 GEMV kernels, quotes real code) and
  t6-reason (last-occurrence insight for k-th-distinct — the canonical
  direction, matching llama).
- The other 8 start correctly and then degenerate at the TOKEN level:
  doubled words, number loops ("CP1252222222…"), chat-template artifacts
  (assistant/user tags), wrong arithmetic in step chains (t6b: 12*2*256
  computed as 25,048). No task runs clean end to end.
- Gate verdict: **FAIL on quality, PASS on diagnosability** — the failure
  mode is now uniform, additive, and container/quant-attributable, not a
  mystery: uncalibrated RTN leaves the coarse distribution right and the
  fine distribution broken.

## Speed measurements (documented SEPARATELY — not a comparison; robin correction 2026-09-03)

These are measurements of two DIFFERENT systems (different hardware
assignment, different quant, different offload architecture). They are NOT
comparable and must not be put in relation; the crow numbers are the
engine-internal baseline that #10 improves against.

- **llama.cpp UD-Q2_K_XL** (llama-server, hybrid: 40/512 experts per layer on
  GPU via `-ncmoe 40`, remaining experts CPU-dequantized on 24 threads,
  2.4-bpw GGUF, KV q8_0, -c 200000 -b/-ub 4096): prefill 204–777 tok/s,
  decode 31–36.5 tok/s. Context of use: the ten-task CORRECTNESS reference
  arm — its speed says nothing about the crow engine's speed.
- **crow-nest engine, BASE CNQ4.5** (GPU-only incl. zero-copy cold path over
  pinned host RAM, 4.5 bpw NVFP4 + BF16 keeps, naive scalar GEMVs, single-
  chunk prefill, FP8-KV): prefill 6.0–6.5 tok/s, decode 4.0–6.5 tok/s.
  Context of use: engine-internal baseline for the #10 perf phase (MMA
  kernels, launch fusion, coalesced loads). Longest prefill data point:
  t1b-read-lang, ~13.5k prompt tokens through the QSA-sparse regime in
  54 min wall — first long-prompt sparse-prefill measurement.

## Decision queue (robin's call)

1. **CNQ4.5-C (stage-2 calibrated scales)** — now data-backed: kernel bugs
   fixed (layercheck at p10 level), remaining error is uniform RTN noise,
   and calibration is exactly the lever for that. Converter re-emits scales
   only; the 12-position argmax + 64-step trace loop (minutes) iterates it.
2. **KHC keeps**: post-fix measurement says BASE > KHC — the q/k-BF16 path
   (attention layers 3+, never individually verified in the engine crate) is
   the suspect for the KHC loop. Either verify it against
   `oracle/golden/layer3-attn-*` or drop the keeps.
3. **#10 perf**: long-prompt prefill in the sparse QSA regime measured at
   6 tok/s (54 min for one task) — the prefill kernels join the perf phase
   with a concrete target.
4. Harness follow-ups: incremental report writes (a phase crash currently
   loses the whole arm), re-warm-up to regenerate frequency-ordered sidecars.


## CUDA Graphs (CROW_GRAPH=1) — implementiert, ein Runtime-Bug offen

- Stream-Capture des 48-Layer-Decode-Blocks (launch über einen globalen
  Current-Stream; Skalar-Refreshes aus PINNED Staging — async HtoD von
  pinned ohne Sync; Stack-Quellen sterben auf dem NON-BLOCKING-Stream, das
  war der erste Invalid-Value).
- Graph-Funktionen via libloading aus nvcuda.dll (cudarc 0.19.9 bindet die
  Graph-Einstiegspunkte nur für CUDA 11.4–11.8, nicht cuda-13030).
- `cuGraphInstantiateWithFlags` verlangt 3 Argumente (exec*, graph, flags) —
  der erste Versuch nutzte die alte 5-Argument-Signatur (UB).
- STAND: Build grün; Capture + Instantiate + Replay laufen (2588 Launches
  SUCCESS im 64-Step-Lauf), der Token-End-Sync gibt danach
  CUDA_ERROR_INVALID_VALUE — Verdacht: invalidiertes Capture (Poisoned
  Stream) oder ein DtoD/rope64-Detail; Doku-Quellen:
  docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__STREAM.html (Capture-
  Regeln), NVIDIA-Forum „capture across separate streams".
  Nächste Iteration: EndCapture-Status prüfen (cuStreamGetCaptureInfo),
  d2d_block-Grid/Limits prüfen, ggf. CUDA-Graphs nur für die Layer-Kette
  ohne QSA-Pool-Zweig.
- Der Non-Graph-Pfad läuft stabil (143 ms/Token M+MMA, Traces ok,
  layercheck 0,0918) — die Messungen fahren non-graph, bis der Graph-Modus
  sauber ist. Engine bleibt variabel: env CROW_GRAPH=1.
