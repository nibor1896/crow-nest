# #202 / #188: where a decode MoE layer waits (ARM2, profile-20261010h)

Source: `runs/glm53-flash/profile-20261010h/decode.sqlite` (nsys, CUPTI trace, no OS-runtime
or NVTX rows), ARM2 env, commit 7d71819, 63 cold decode tokens after a 57-id prompt. All
numbers below are read from that trace (one run); "model" marks the few that are derived.
Scripts: the session's scratchpad `layers2.py` / `nv.py` (sqlite only, not in the repo).

Clock: one host thread issues every CUDA call (tid 32184). The compute stream is s7, the
router guess (PREFETCH_SIDE) s13, the stager stream s16, the write-back ring s14.

## One MoE layer, in order (ARM2)

| step | what happens | who waits on what |
|---|---|---|
| 1 | `gemv_bf16_b` + `glm5_router_sig_topk` on s7; the guess of l+1 on s13 | - |
| 2 | `glm5_publish_pred` (s7) waits for s13's event, copies ids + guess to mapped memory, raises the flag | GPU: s7 waits for the side guess |
| 3 | host spins on the mapped flag (`Routed::wait`, no CUDA call), then `table_for` -> `table_global` (stager): arena step, NVMe reads issued, landed-flag waits + copies + table H2D queued on s16, event recorded, **s7 waits on that event** | host: planning |
| 4 | `experts_lane`: D2H of x (s7, queued **behind** the stager-event wait), event, gather, ptrs H2D, gate/up/act/down over the GPU slots (10 launches) | - |
| 5 | host `cuEventSynchronize(x event)` | host: x, i.e. the whole stager batch, i.e. **every NVMe landing of the layer** |
| 6 | host `cuEventSynchronize(stager event)` | already done after 5 |
| 7 | CPU lane: the CPU combos (`cpu_mul1::experts_ffn`) | GPU runs its experts meanwhile |
| 8 | D2D of each GPU row, H2D of each CPU row, combine, expand, next layer's attention launches | GPU waits for the lane if it is longer |

## Per-layer timeline, median microseconds (2,634 layers, 41.8 per token)

Columns: `flag` router end -> publish end; `plan` publish end -> x D2H call (host plan);
`launch` the expert launches; `stager` GPU wait from the x D2H call to its run (the stager
event); `idle pre` publish end -> first expert kernel; `sync1` the host's x-event wait;
`lane` CPU-lane run; `gpu exp` first to last expert kernel; `lane gap` GPU idle between the last
GPU expert and the lane's rows; `tail` uploads + combine; `attn idle` GPU gaps from the combine
to the next router.

| class (layers/token) | span | GPU idle | flag | plan | launch | stager | idle pre | sync1 | lane | gpu exp | lane gap | tail | attn idle |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A no landing wait (22.9) | 1259 | 327 | 61 | 47 | 90 | 37 | 112 | 6 | 625 | 765 | 3 | 27 | 80 |
| B one landing wait (11.8) | 2845 | 2066 | 61 | 79 | 74 | 1718 | 1858 | 1657 | 677 | 757 | 3 | 29 | 77 |
| C two (4.5) | 4357 | 3702 | 61 | 97 | 74 | 3263 | 3452 | 3225 | 668 | 740 | 4 | 33 | 76 |
| D three or more (2.6) | 6737 | 5846 | 61 | 122 | 73 | 5330 | 5476 | 5276 | 675 | 750 | 16 | 46 | 69 |

"Landing wait" = a `cuStreamWaitValue64` the plan queued on the stager stream (an NVMe demand
read, or a join of a guessed read still in flight).

## Per token (ms; 103.8 ms per token, GPU kernels busy 38.9 ms)

| where the GPU idles | ms / token |
|---|---|
| s7 waiting on the stager event = NVMe landings (classes B-D 46.5, A 0.9) | 47.4 |
| router end -> flag (the side guess's join + publish kernel), 61 us x 41.8 | 2.7 |
| host plan after the flag, 54 us median | 2.9 |
| expert launches before the first expert kernel runs | ~2 |
| lane longer than the GPU experts (lane gap) | 2.8 |
| uploads + combine after the lane (8 copy calls ~7 us each + combine) | 2.2 |
| gaps inside attention / dense segments (launch bound) | 4.6 |

Host side: `cuEventSynchronize` 45.8 ms / token over 94.7 calls = step 5 (45.7 ms, 41.8 calls)
+ step 6 (0.03 ms) + 11 MLA scalar waits (~1 us each). So the 45.8 ms of host event sync is the
NVMe landing, seen through the x copy that was queued behind the stager event; the routing
readback itself is already a polled mapped word (CROW_GLM_FLAGS) and costs ~5 us.

## The NVMe landings

First landing copy after the flag, median: 1 read 1,391 us, 2 reads 3,009, 3 reads 4,379,
5 reads 5,994 (743 staging layers). The reads land one after another at ~1.4 ms per 9.47 MB
record (~6.8 GB/s): the drive's bandwidth, single reader (`readers: 1`, no pool).
Records read per token (decode.json): 16.4 demand + 45.9 guessed (used 24.5, wasted 21.3,
joins 14.2) = 62.3 x 9.47 MB = 590 MB = ~87 ms of drive time per 103.8 ms token. The drive is
~84 % busy; demand reads queue behind guessed reads in the one reader FIFO.

## What that means for the fix (model, not measured)

- Starting the CPU lane and the GPU experts on resident records right after the flag, and only
  the landing experts after the stager event, overlaps up to sum(min(stager wait, max(lane, GPU
  experts))) = 13.4 ms / token; folding the 8 row copies into the combine ~2-3 ms. Model:
  ~88 ms / token.
- 1,700 launches / token are ~40 per layer, not per expert: the experts already run as one
  launch per matrix over all slots (had_in / gemv / had_out x 3 + act); attention, KDA and the
  dense layers are the other ~28.
- 50 ms / token needs the NVMe bytes per token cut: at ~6.8 GB/s, 50 ms is ~36 records per
  token including guesses (now 62.3). The wasted guesses alone are 21.3 records = ~30 ms of drive.

## The fix: `CROW_GLM_RT2=1` (default off; ARM2 env, not the controller)

- `glm5_model::call_inner`: the MoE input row goes to the CPU lane's host buffer (D2H + event)
  ahead of the router's publish, so the host has x once it saw the flag (`GpuMoePlan::lane_x_early`).
- `ExpertTiers::table_global` (global arena + stager, one row, no controller): the stager's
  copies record the device addresses they write (`StagerMover::written`); an expert is late when
  its record is a staging slot, an address the batch writes, a pinned slot a write-back still
  lands in, or a read not landed yet (flag below its value). The host no longer waits for
  write-backs (`wait_host`): such picks are late. The CPU lane takes only non-late pinned picks,
  so it never waits. The compute stream is not made to wait for the stager event here; the call
  is posted with the event and the late mask (`lane::Rt2`).
- `GpuMoePlan::experts_rt2`: gather (through a zeroed table), one H2D of slot bases + row
  addresses, gate/up/act/down over the early slots, event, `cuStreamWaitEvent(stager)`, the same
  over the late slots, shared expert; then the lane (x polled with `cuEventQuery`, no
  `cuEventSynchronize`), then one `glm5_moe_combine_rows` that reads GPU rows from `ye` and CPU
  rows from mapped host memory (replaces ~8 D2D/H2D calls + combine). The compute stream still
  waits for every stager batch before the layer's combine.
- Host syncs removed per MoE layer under RT2: 2 x `cuEventSynchronize` (x, stager) and the
  write-back `cuEventSynchronize` of `wait_host`.

Tests (synthetic / unit, this worktree):
- `glm5_moe_gpu_rt2_lane_and_resident_experts_run_before_the_stager_event`: stager event held
  by a `cuStreamWaitValue64`; `run` returns with the lane done and the layer not finished, the
  resident GPU experts complete under the hold, `y` = `experts_lane`'s bits for 6 CPU/late masks;
  the lane started while the GPU ran its resident experts in 3 of 3 runs that had both; a held
  memop wait does not slow the resident experts (1.303 ms free vs 1.301 ms held, median of 9).
- `glm5_int_gpu_rt2_is_the_arm_without_the_controller_within_the_accuracy_bar`: ARM2-like env,
  300-id prompt + 6 ids, V3 P4 and V0 P16: without the lane RT2 = arm bit for bit (same NVMe
  reads); with the lane split same ids, cosine min 1.0000000 (V0 P16 bit-identical; V3 P4 the
  lane took 27 instead of 58 experts: picks still landing go to the GPU's late pass).
- `cargo test --release --lib`: 584 passed, 130 ignored.

Not measured on the real container (the lead measures). Expected from the trace (model):
<= 13.4 ms / token overlap + ~2-3 ms from the combine; the drive time (~87 ms / token at 62.3
records) stays the floor until fewer records are read.
