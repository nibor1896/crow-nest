# crow-nest — job-ring handoff measurement (issue #7, 2026-09-02)

Benchmark: `handoff-bench/` (Rust, cudarc 0.19.9 sys API, NVRTC, target `compute_120a`).
One warp publishes a cold-expert job via device-side mapped writes + system atomics; a
host stager thread polls the pinned ring, "gathers" a 2.76 MB expert block (pinned→pinned),
ships it H2D on a dedicated copy stream; the GPU gates the consumer on a 64-bit
`cuStreamWaitValue64_v2`. Sequence numbers u64, monotonic. Windows first, as spec section
3 demands. Machine: `docs/system-landscape.md`.

## Measured (RTX 5090, driver 616.56, Windows, WDDM)

| measurement | value | n |
|---|---|---|
| raw memop: `cuStreamWriteValue64_v2` + `cuStreamWaitValue64_v2` (pre-set) | **43 µs** per pair | 2000 |
| full handoff round trip, serialized (publish → stager → H2D → consume) | **mean 3.06–3.15 ms**, p99 ≈ 3.2, min ≈ 3.0 | 200 |
| full handoff, pipelined (200 back-to-back, stager overlapping) | **3.37–3.47 ms/handoff** | 200 |
| busy-poll event sync instead of `cuStreamSynchronize` | no change (3.06 ms) | 200 |

## Findings

1. **The memops themselves are cheap (43 µs) — 100× under the 4–6 ms barrier** (Crow
   #186). The primitive is not the problem.
2. **The full round trip is ~3 ms — dominated by WDDM submission batching, not by the
   memops and not by host wait-yield** (busy-poll made no difference). Every submission
   (kernel launch, DMA, memop) sits behind the WDDM command-batcher; a per-token ring
   round trip therefore costs ~3 ms in this naive per-layer shape — only ~1.5–2× better
   than the borrowed engine's barrier.
3. **Architecture consequence (changes the spec's assumption, needs robin's review):**
   per-layer ring round trips are TOO EXPENSIVE on Windows/WDDM. The scheduler must
   **batch** handoffs — one round trip per token (or per N layers), not per layer —
   and/or Hardware-Accelerated GPU Scheduling (HGS) must be evaluated, which shrinks
   WDDM batching latency. Linux retest is mandatory before any Linux number (the same
   shape is expected to be far cheaper there — exl3's pattern is Linux-proven).
4. **Completion-path finding:** the design path "GPU raises the completion flag via
   memop on the copy stream, ordered after the H2D" did NOT hold reliably on WDDM —
   consumers observed torn payloads (deterministic per mode, reproducible). The
   WDDM-safe default is the exl3-style host-store completion: the stager waits for its
   OWN copy stream (`cuStreamSynchronize(s_copy)`, off the GPU critical path) and then
   raises the flag with a plain host store + fence. The memop path stays behind a
   feature flag (`completion_memop`) for the Linux retest.
5. **Verification status:** payload integrity is EXACT on the synchronized path
   (sync H2D + consumer: kernel xor == host xor == 0xe3865729, n=690,000 ✓). In the
   async loop the consumer's xor is deterministic-per-mode but ≠ reference — read
   timing/visibility of async H2D against a concurrently running kernel on WDDM;
   open item for stage #8 (double-buffered payload slots + acquire fences).

## Debugging findings preserved (each cost a round, each is a rule now)

- Device pointers are NOT host-dereferenceable (crash: `read_volatile` on a
  `CUdeviceptr`) — results cross to the host via DtoH or mapped memory only.
- A per-thread partial result needs a warp reduction (`__shfl_down_sync`) before thread
  0 reports — otherwise the result is thread 0's stripe only (silent, deterministic,
  wrong).
- `cuMemcpyDtoH_v2` on the legacy null stream does NOT serialize against
  NON_BLOCKING streams — torn reads observed; use stream-ordered copies + sync.
- Mapped-memory region sizing: the kernel's result region must cover every field it
  writes; a 16-byte region with 32-byte writes clobbered the adjacent probe and source
  regions (the source of several phantom "wrong data" states).
- The first H2D of a stale-zeroed job flag fires at stager startup — job slots must be
  initialized to a sentinel (0xFFFFFFFF), not zero.

## What this means for the plan

- Stage #7's mechanism question is answered: the ring works on Windows, end to end,
  with the stager fully off the GPU critical path.
- The cost question got a number nobody hoped for: WDDM batching is the wall at
  per-layer granularity. This feeds the scheduler design (#8) directly: batch handoffs,
  double-buffer, and measure HGS. The parity harness (#6) will quantify all of it
  against llama.cpp on the same stack.
