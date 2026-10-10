# GLM-5.3-Flash: stability policy (lever 6)

Lever 6 of the GLM measurement book, section F (root #169): keep the card below the edge where WDDM
starts paging, size decode staging apart from prefill, and keep the E4M3 NaN code out of the container.
Tickets: #176 (engine, VRAM headroom and decode staging), #177 (converter, scale byte 0x7F).
Flash-Next and the 27B are unchanged by all of it.

## Why

Crow, 2026-08-11, RTX 5090, fresh server per run (`docs/archive/README-v0.5.1-deepseek.md:532-541` of Crow):

| free VRAM | prefill, median | spread |
|---:|---:|---:|
| 593 MiB | 15.28 tok/s (3.83 to 33.29) | 8.69x |
| 2,059 MiB | 112.69 tok/s | 1.013x |

Near the card limit the Windows driver moves allocations to system memory without an error (NVIDIA,
"System Memory Fallback", driver 536.40 and later). The engine's own reserves are 512 MiB
(`manager::SAFETY`) and 256 MiB (`manager::POST_PLAN_FLOOR`).

## The policy (`engine/src/geo.rs`, `Stability`)

| | Flash-Next, 27B (`OF_RECORD`) | glm5_next (`GLM5_NEXT`) |
|---|---|---|
| VRAM kept free on top of the reserves | 0 | 2 GiB (`GLM5_NEXT_VRAM_HEADROOM`) |
| VRAM cap | the card | 31.9 GiB = 34,252,364,185 B (`GLM5_NEXT_VRAM_CAP`, G4 of `runs/glm53-flash/PREREG.md`) |
| plan ceiling on a 32,607 MiB card | 34,190,917,632 B | 32,043,433,984 B |
| cold staging | one set: max(2 x topk, PF_TG x (1 + async)) = 128 slots | decode 4 x 8 = 32 slots, prefill 128 slots apart; one expert record per slot (`StageSlots::bytes`): 160 x 14,155,776 = 2,264,924,160 B at NVFP4, 160 x 9,474,048 = 1,515,847,680 B at the plan's 3.05-bpw MUL1 record (decode set 452,984,832 / 303,169,536 B) |

- `Stability::of(Family)` gives `OF_RECORD` for Flash-Next and the 27B and `GLM5_NEXT` for `Family::Glm5Next`
  (#159); `Stability::for_model_type("glm5_next_text")` gives the same. `manager::planner_pending_for` adds
  `planner_reserve(total)` to the planner's pending bytes (0 B for `OF_RECORD`); the glm5_next plan
  (`states --plan`) books it, #149 (stager) allocates the separate prefill set.
- `gen.rs` sizes `Stage::max` through `stage_slots(..).held()`; for `OF_RECORD` that is the old expression bit
  for bit. A policy with decode slots apart is refused there until the GLM arm allocates the prefill set.
- On GLM the shared formula would hold 128 x 14,155,776 B = 1.8 GB of staging through every decode step.
- The slot bytes are the container's routed-expert record (`geo::ExpertRecordSpec`, from the container index or
  `states --plan --expert-bytes`), never the NVFP4 constant: a 3-bit container stages 9,474,048-B slots (#159).

## The converter cap (`converter/src/recipe.rs`, `Family::scale_byte_max`)

A `cnq4.5-glm5-next` container stops every ue4m3 sub-block scale at 0x7E (448). 0x7F is the E4M3 NaN code
(arXiv:2209.05433, Table 1); the mxf4nvf4 MMA instruction reads it as NaN. The cap is applied before the
E2M1 codes are chosen (`quantize_nvfp4_cap`). On the GLM miniature of the unit tests, `--scales mse` wrote
231 such bytes into one 128 x 128 tensor before the cap. The engine still rewrites 0x7F at load
(`residency::sanitize_sf_slab`, `gen.rs` `load_pw_x`) for containers written before #177. Check a new
container: `h_scales[127]` is 0 on every NVFP4 line of its sidecar.

## The 2-hour soak (a measurement, not run yet)

Runs once a GLM container boots (#149, #159). Every run, also a false start, goes into the GLM measurement book.

1. Fresh `serve` on the GLM container, nothing else on the GPU but the desktop. Record
   `nvidia-smi --query-gpu=memory.used,memory.total --format=csv` and the `[budget]` and headroom boot lines.
2. Crow drives a fixed loop of the ten-task prompts (`docs/ten-tasks.md`) for 120 minutes, greedy.
3. Log per request decode and prefill tok/s; `nvidia-smi -l 10` for `memory.used`; the process's shared GPU
   memory (`Get-Counter '\GPU Process Memory(*)\Shared Usage'`).
4. Pass: decode spread (max/min, same task) <= 1.15; free VRAM never below 2 GiB; shared GPU memory flat
   (growth <= 64 MiB); no request below 0.5x the median.

## Tests

```
cd engine && CARGO_BUILD_JOBS=4 cargo test --release --lib tests_176      # 3 tests, no GPU
cd converter && CARGO_BUILD_JOBS=4 cargo test --release -- 0x7 the_glm_rule  # 3 tests
```
