# Environment variables of the crow-nest engine

## Scope and rule

| Item | Value |
|---|---|
| Variables in this table | 74 |
| Distinct `CROW_[A-Z0-9_]+` tokens in the code | 74 in `engine/src`, 0 in `converter/src` |
| Measured | 2026-09-10, task E5, issue #46, parent #1 |
| Repository state | branch `release-v0.1`, HEAD `0d1cc0d` plus the `#61d` commit, 2026-09-13 |
| Guard | `tools/check_env_docs.py` (code list minus doc list must be empty, both ways) |
| Rule | a variable not in this table does not exist |

Command that produced the list:

```
grep -a -r -o -h -E 'CROW_[A-Z0-9_]+' engine/src converter/src | sort -u
```

- `-a` is required: `engine/src/cuda.rs` holds non-UTF-8 bytes and `grep` treats it as binary without it.
- `tools/check_env_docs.py` reads the same two directories with `errors='replace'` and the same regex.
- Every row names a `file:line` in the engine sources. A row without a read site or a comment site is not written.

## How to read the columns

| Column | Meaning |
|---|---|
| `Name` | the exact variable name |
| `Read at` | `file:line` of the read (`std::env::var`, `var_os`, or the helper that reads it) |
| `Values / default` | accepted values and the value in force when the variable is unset |
| `Effect` | one clause, taken from the read site |
| `Mode` | `operating`, `measurement`, `diagnostic`, or `deprecated or comment-only` |
| `Notes` | parity relevance, the gate that pinned the default, warning text |

Mode meanings:

- `operating`: part of the gated configuration or a supported operating switch.
- `measurement`: harness and benchmark control, not read by `serve`.
- `diagnostic`: changes output or timing on purpose; never in a gate.
- `deprecated or comment-only`: no read site, mentioned in a source comment only.

Helpers used by the read sites:

| Helper | Site | Semantics |
|---|---|---|
| `env_on(name)` | `engine/src/gen.rs:1138` | true unless the value is exactly `0`; default on |
| `num(k)` | `engine/src/geo.rs:168` | parse to `usize`, `None` when unset or unparsable |
| `f(k, d)` / `u(k, d)` | `engine/src/sample.rs:73`, `:74` | parse to `f32` / `usize`, fall back to `d` |

- Scope: `engine/src` and `converter/src` only; shell variables of Crow (for example `CROW_TAVILY_KEY`) have no engine read site and no row.
- The `--diag` cargo feature alternative of the E5 gate was not taken; no engine change before the tag.
- The four diagnostic values stay reachable in the default binary and carry a warning row (section Diagnostic values).

## Container and residency (13 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_CNQ` | `engine/src/bin/decode.rs:40` | path; default `../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` in `decode`, `residency` and `sf_scan`, `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` in `parity`, `DEFAULT_CNQ` in `serve` (`bin/serve.rs:2320`, constant at `bin/serve.rs:454`) | selects the container file | operating | also read at `bin/parity.rs:158`, `bin/parity.rs:371`, `bin/residency.rs:46`, `bin/sf_scan.rs:9`; all five default sites name `-M`, `decode`, `parity` and `serve` since `#51`, the two generator bins since `#52` (2026-09-11); `bin/plecheck.rs:5` and `bin/states.rs:45` hard-code the old container and read no variable; chain value `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` |
| `CROW_HOTSETS` | `engine/src/bin/decode.rs:45` | path; default `../decode_out/hotsets-M-longctx2100-n160.json` in `decode`, `decode_out/hotsets-M-longctx2100-n160.json` in `parity`, `DEFAULT_HOTSETS` in `serve` (`bin/serve.rs:2321`, constant at `bin/serve.rs:455`) | overrides the hot set sidecar | operating | also `bin/parity.rs:164`; `residency` reads `CROW_HOTSETS_OUT` instead, because its sidecar path is an output (`#52`); the default is a literal since `#51` (2026-09-11), no longer `<cnq>.hotsets.json`, because the sidecar next to the `-M` container is ragged (`#49`); chain value `decode_out/hotsets-M-longctx2100-n160.json` |
| `CROW_HOTSETS_OUT` | `engine/src/bin/residency.rs:50` | path; default `../decode_out/residency-warmup.hotsets.json` | names the sidecar file the `residency` warm-up WRITES, and reloads from on the next start | measurement | `#52` (2026-09-11): never `<container>.hotsets.json`, because the ragged sidecar of `#49` lives there and must stay byte-unchanged; read by `bin/residency.rs` only, `decode`, `parity` and `serve` read `CROW_HOTSETS` |
| `CROW_COLD_TIER` | `engine/src/residency.rs:231` | path to `<cnq>.cold<bits>.bin`; default unset (exact NVFP4 tier) | installs the low-bit cold tier built by `bin/coldtier.rs` | operating | record sizes read from the header at `gen.rs:699`; `docs/architecture.md:1163` keeps it off for `serve` (7.5 condition 2) |
| `CROW_COLD_FULL` | `engine/src/gen.rs:706` | `1`, `0`; default: full tier when `E * unit <= cfg.host_pinned_budget` | forces the full tier (every expert pinned) or cold-only | operating | full tier is what enables the prompt-adaptive hot set |
| `CROW_RAM_MARGIN_GB` | `engine/src/residency.rs:249` | integer GiB; default `3` | free physical RAM that must remain after pinning the cold tier | operating | below the margin `residency.rs:255` panics before anything is pinned |
| `CROW_PINNED_WC` | `engine/src/residency.rs:275` | `0` restores cacheable pinned memory; default write-combined | allocation type of the pinned cold slabs | operating | measured 2026-09-04 (`pcie_probe`): WC 47.6 GB/s vs cacheable 24 GB/s |
| `CROW_MMAP` | `engine/src/cnq.rs:91` | `0` disables; default on | maps the container read-only, row reads become page-cache memcpys | operating | doc comment `cnq.rs:29`: about 1 us on a hit instead of a seek plus read pair |
| `CROW_CNQ_PURGE` | `engine/src/cnq.rs:46` | `0` keeps the cache; default purge | on drop, re-opens the container unbuffered to purge its cached pages | operating | rationale `cnq.rs:38-42`, 2026-09-06: 3.3 GB of tier reads stayed in the system cache and `residency.rs` then refused to pin |
| `CROW_QSA_FULL` | `engine/src/manager.rs:37` | `1` sets `ring = context`; default `ceil4(prompt_chunk + 4)` capped by context | size of the raw indexer-key ring per attention layer | measurement | pre-2026-09-05 layout; `docs/architecture.md:1162` marks it out of scope for `serve` |
| `CROW_KV` | `engine/src/bin/decode.rs:62` | `bf16`; default the container KV dtype | parity ladder switch for the KV cache dtype | measurement | read only in `bin/decode.rs`, not by `serve` |
| `CROW_PLE` | `engine/src/bin/decode.rs:65` | `off`; default on | parity ladder switch that disables the PLE stage | measurement | read only in `bin/decode.rs`, not by `serve` |
| `CROW_PLE_CACHE_MB` | `engine/src/gen.rs:639` | integer MiB; default `cfg.ple_cache_bytes` (128 MB in `serve`, `bin/serve.rs:47`) | size of the PLE hot-row cache | operating | VRAM diet knob, 2026-09-05; C1 allowlist keeps it unset (issue #11, comment 5621121049, C1 result) |
| `CROW_PLE_PREFETCH` | `engine/src/gen.rs:2571` | `0` disables; default on | warms the next chunk's PLE rows on a helper thread | operating | no-op without a file mapping (`CROW_MMAP=0`) |

## Prefill and chunking (13 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_CHUNK` | `engine/src/geo.rs:139` | integer tokens; default: policy by prompt length | authoritative prefill chunk size | operating | also `bin/decode.rs:58`, `bin/decode.rs:387`; `serve` pins 2048 for the process and does not call the policy (`bin/serve.rs:41-43`) |
| `CROW_CHUNK_AUTO` | `engine/src/geo.rs:143` | `1`, `0`; default `1` when `CROW_CHUNK` is unset, `0` when it is set | applies the length policy on top of an explicit chunk, cap `max(CROW_CHUNK, 2048)` | operating | default since 2026-09-05; gated per `geo.rs:136-137`: chunk 1024 and 2048 deterministic since #22 |
| `CROW_CHUNK_BALANCE` | `engine/src/gen.rs:2556` | `1` enables; default off | cuts the prompt into equal chunks instead of full chunks plus a tail | measurement | measured 2026-09-04: 3 x 700 gives 382 tok/s and loses to 1024+1024+52 at 414 tok/s; opt-in only |
| `CROW_PF_ASYNC` | `engine/src/gen.rs:504` | `0`, `1`, `2`, `3`, `4`; default `2` | staging path of the prefill cold experts | operating | default since 2026-09-09 (#10, robin's call), gated: parity 8/512/1024 plus ten tasks ids equal to final4 without the env; values `3` and `4` are diagnostics, see the warning table |
| `CROW_PF_DMA` | `engine/src/gen.rs:487` | `1` enables; default off | copy-engine prefetch ring of one layer's cold slab during prefill | measurement | measured 2026-09-04: with the exact NVFP4 tier the ring makes the plan infeasible; `gen.rs:727` disables it for the run when the ring does not fit |
| `CROW_PF_GEMM` | `engine/src/gen.rs:492` | `0` disables; default on | expert-grouped tile GEMM for prefill-sized MoE batches | operating | `gen.rs:1980`: a low-bit cold tier needs this staging path |
| `CROW_PF_GEMM_B` | `engine/src/gen.rs:522` | `1` enables; default off | prefill dense GEMM variant B (#10c): 32-token tiles + 16-byte vectorised weight fragments, `gemm_fp4_dense_b` (`kernels.rs:573`) and `gemm_bf16_dense_b` (`kernels.rs:1134`), selected inside `launch_mma_d` (`gen.rs:1256`) and the `launch_bf16_dense` helper (`gen.rs:1277`) that `PW::launch_gemv` and the `CROW_ROUTER_GEMM` site share | measurement | #10C, 2026-09-14: per-token math identical to the 8-token forms (same per-row k chain, levels, KS reduce); bit identity proven, parity 29 of 29 GREEN incl. the switch-ON 8 / 512 x2 / 1024 forms at the 61b sha256 values of record and the teacher-forced 16,064-row PX form `f217e1c55926` under the > 26 GiB VRAM headroom gate (`decode_out/srv-10c.log`, RTX 5090); KPROF dense group 2.00x (568.9 vs 284.7 ms per step over the profile run); ten tasks 10 of 10 equal final4 with no env; pairs (t1-read 16,064 ids, W + 3 adjacent pairs): N 18.437 s = 871.3 tok/s against the default-of-record B plateau 20.726 s (the two clean pairs -2.270 and -2.330 s = -11.0 and -11.2 percent; B1 18.491 s is a documented whole-run outlier); distinct from `CROW_PF_GEMM`, deliberately not overloaded; one `[pf-gemm-b]` boot line per process names the form (`gen.rs:785`) |
| `CROW_PF_TG` | `engine/src/gen.rs:464` | integer, clamped to `[8, 512]`; default `PF_TG` = 64 | tiles per staged group | operating | default 64 since 2026-09-09 (#10, gated on ten tasks with `CROW_PF_ASYNC=2`; 32 before); costs `CROW_PF_TG` x 2.76 MB of staging slots |
| `CROW_STAGE` | `engine/src/gen.rs:519` | `0` selects direct zero-copy; default on | stages cold experts into VRAM with coalesced PCIe reads before the routed GEMVs | operating | decode path only (`t * TOPK <= stage.max`) |
| `CROW_STAGE_SPLIT` | `engine/src/gen.rs:512` | `1`, `2`, `4`, `8`, `16`, `32`, `64`; other values fall back; default `8` | blocks per expert in the `stage_cold` staging copies; KERNEL 1 ONLY: `stage_cold_ca`, the default since #19e, does not read it, and the load assert that does is gated on the kernel 1 branch (`engine/src/gen.rs:687`) | measurement | measured 2026-09-04 with the WC tier: 16/32/64 keep more PCIe requests in flight; dead for the default kernel since 2026-09-12 (#19e, fix C4 of #19d) |
| `CROW_STAGE_DMA` | `engine/src/gen.rs:530` | `1` enables; default off | stages the cold combos of a layer with one `cuMemcpyDtoDAsync` per combo and matrix (copy engine, mapped host pointer) instead of the `stage_cold` kernel; forces `CROW_GRAPH` off | measurement | #19b, 2026-09-11; the routed pointers are known on the host only after `router_top10`, so the decode graph cannot hold the copies; one host sync per layer |
| `CROW_STAGE_KERNEL` | `engine/src/gen.rs:542` | `1` selects the old `stage_cold`; unset or any other value selects `stage_cold_ca`; default `2` | shape of the decode staging copy: `stage_cold_ca` is a persistent grid that pulls each cold combo through `cp.async.cg.shared.global` into 4 KB shared tiles and stores them coalesced to VRAM; the kernel has NO tail tile, so it requires both staged slab byte counts to be exact multiples of 4096, asserted at load (`engine/src/gen.rs:683`) and again at the launch site (`engine/src/gen.rs:2048`), both panic messages naming this switch | operating | default `2` since 2026-09-12 (#19e); measured 2026-09-11 on the #59 profile arm (t1-read 16,064 ids, 255 timed steps, RTX 5090): 26.46 ms per decode token against 29.68 for `stage_cold`, staging row 7.07 ms at 47.78 GB/s against 10.42 ms at 32.45 GB/s (`decode_out/srv-19d.log`); every engine process names its kernel in one `[stage]` line (`engine/src/gen.rs:697`) |
| `CROW_STAGE_BLOCKS` | `engine/src/gen.rs:552` | integer, accepted `8` to `512`, other values fall back; default `40` | blocks of the persistent `stage_cold_ca` grid (block 256) | operating | default `40` since 2026-09-12 (#19e), read whenever `stage_cold_ca` runs, which is every run without `CROW_STAGE_KERNEL=1`; 19d measured 2026-09-11: 40 and 80 tied inside their own spreads, 20 worse by 0.1532 ms per decode token |

## Attention and kernels (16 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_ATTN_R` | `engine/src/gen.rs:1161` | `0`, `1`, `2`, `3`, `4`, `5`, `8`, `9`; default unset = `attn_sel_s8l` | selects the decode attention kernel (`gen.rs:1162`) | operating | default since 2026-09-06 late (#10, robin's call), gated by parity 8/512/1024 against the previous build plus ten tasks; `8` and `9` are diagnostics, see the warning table |
| `CROW_ATTN_SB` | `engine/src/gen.rs:1172` | `0` restores the full-chunk buffer; default on (`ATTN_SB` = 512) | prompt attention in sub-batches of 512 tokens | operating | #16, 2026-09-05: shrinks the QSA score buffer from 512 MB at chunk 2048; bit-identical by construction |
| `CROW_ATTN_SPLIT` | `engine/src/gen.rs:1151` | `0` disables; default on | decode attention as 8 partials plus merge, QSA scores warp per block | operating | both forms are graph-static |
| `CROW_ATTN_SPLITS` | `engine/src/gen.rs:1302` | `4`, `8`, `16`, `32`; anything else and unset take 8 (`ATTN_SPLITS`, DEFAULT 8, restored by #61e 2026-09-13) | split count of the decode attention: `grid.z` of `attn_sel_split` (`gen.rs:2070`) and the device scalar `p.n_splits` read by `attn_merge` (`gen.rs:2074`) | operating | DEFAULT 8 SINCE #61E, 2026-09-13: the #61d flip to 32 (robin's performance-over-ids ruling of 2026-09-12, kept of record) was ROLLED BACK one day later under the improvement-loop quality rule - the ten-task quality bar is NOT held at 32, judged 0 Pass / 7 Partial / 3 Fail against the crow record 2 / 5 / 3 at 8 and the llama reference 2 / 6 / 2 (`.superpowers/sdd/task-61d-quality-report.md`); the rollback is engine commit `fdc00c4`, the const plus its `[attn]` boot line and nothing else; `16` and `32` stay MEASUREMENT ONLY; the 61d numbers stay the measurement of record for the knob: the adjacent pair of `decode_out/srv-61d.log` (RTX 5090, 2026-09-13, t1-read 16,064 ids, 255 timed steps), 22.8545 against 24.1325 ms per token with `8`, -1.278 ms = -5.3 percent at 32.3 x the fallback spread, B ids `5098f885ab3a` 3 of 3, N ids `c65969f7793a` 3 of 3 (the 61a/61c S32 value); the 61c sweep ordered 32 < 16 < 8 in every dataset (B 24.8433 / S16 23.8484 / S32 23.4001 ms per token); a different split count changes the merge order of the flash-decoding partials, so the last bits of the logits may move: `16` and `32` change the generated ids (first differing index 45 and 48 of 256 on t1-read, `decode_out/srv-61a.log`); every engine process prints one `[attn]` boot line naming the split count it runs and the restored default (`gen.rs:741`); the partial buffers are sized for `ATTN_SPLITS_MAX` = 32 (`gen.rs:1593-1594`, VRAM plus 0.55 MB); the value is read once and the decode graph captures it |
| `CROW_BF16_GEMM` | `engine/src/gen.rs:120` | `0` selects the warp GEMV; default on | 8-token bf16 tile GEMM for prefill-sized batches | operating | applies only to `PW::Bf16` weights at `t >= 8` |
| `CROW_BF16_W` | `engine/src/gen.rs:1142` | `0` selects `gemv_bf16_b` / `gemv_bf16`; default on | selects `gemv_bf16_w` (8 rows per block) for bf16 GEMVs and the LM head | operating | step-2 kernel switch; LM head site `gen.rs:2408` |
| `CROW_DENSE_GEMM` | `engine/src/gen.rs:1118` | `0` selects the per-token MMA GEMV; default on | 8-token tile GEMM for dense FP4 projections at `t >= 8` | operating | reached only when `CROW_MMA=1` (`launch_mma_d`) |
| `CROW_GDN_REG` | `engine/src/gen.rs:475` | `0` off, `p` prefill scan only, `s` decode step only, anything else both; default both | register delta-rule scan instead of the global-memory scan | operating | `0` is the bit-identical fallback per `gen.rs:470` |
| `CROW_GDN_FUSE_IN` | `engine/src/gen.rs:1309` | `0` selects the four per-slab launches (the fallback of record); unset or any other value runs the grouped form; DEFAULT ON SINCE #19G, 2026-09-13 | GDN decode input projections as ONE `gemv_fp4_mma_g` launch (`kernels.rs:555`, registered `kernels.rs:3601`): the qkv 10240 + z 6144 + b 48 + a 48 rows of `gdn_step` in one grouped launch, 258 blocks x `mma_bx()`, one shared quantized row `xq_m`, per-slab global scales KEPT, launch site `gen.rs:1805` behind `gdn_fuse_in_on()` (else the four `gemv_fp4_mma_d` launches of record); capture-time only, the decode graph records the branch once per process, `CROW_GRAPH=0` takes the same branch per launch; one `[gdn]` boot line per process names the form it runs (`gen.rs:749`) | operating | #62B, 2026-09-13, opt-in: bit identity proven at logit level, P8FUSE (504 teacher-forced rows) and PXFUSE (16,056 rows) byte-identical with the switch ON against fresh reference runs, switch-OFF parity 8 of 8 against `d211ab52ad2b`, ten-task splits-32 baseline 10 of 10 (`decode_out/srv-62b.log`, RTX 5090); opt-in pairs (t1-read 16,064 ids, 255 timed steps, ids `c65969f7793a` in 7 of 7 runs): N 22.1704 vs B 22.2627 ms per token, the two clean pairs -0.33 ms against an N spread of 0.0437 ms (17x tighter than B); DEFAULT SINCE #19G, 2026-09-13 (`0` = the fallback of record): the COMBINED parity pass ran both defaults at once with no env - six short forms + the 16,056-row PXBOTH long form byte-identical to `d211ab52ad2b` reproducing the 61b sha256 values of record, double-fallback 8 rows identical, ten tasks 10 of 10 equal BOTH splits-8 records (`decode_out/srv-19g.log`, RTX 5090); the 19g pairs (ids `5098f885ab3a` 7 of 7): the combined default 22.7999 ms per token = 43.9 tok/s against the previous default of record 23.94 = 41.8 tok/s, honest gain -1.14 ms = -4.8 percent (the pair-chain basis: 19f pairs -0.98 ms + 62b clean pairs -0.33 ms); the same-chain double-fallback arm 26.8559 ms per token is the cascade-off artifact of the `CROW_QFUSE=0` overload, not the previous default |
| `CROW_GRAPH` | `engine/src/gen.rs:1103` | `1` enables; default off in the library, `1` in `serve` (`bin/serve.rs:2313`) | captures the per-token kernel sequence once and replays it | operating | fable gate 2026-09-03 (WDDM launch overhead); `serve` sets it only when unset, so `CROW_GRAPH=0` still wins |
| `CROW_INJ_1K` | `engine/src/gen.rs:1143` | `0` selects `gemv_fp4_b`; default on (`gemv_fp4_b1k`, 1024 threads) | kernel of the block-inject GEMV | operating | `gen.rs:1442-1445`: the MMA tile is about 10x slower here, documented skip |
| `CROW_MMA` | `engine/src/gen.rs:1091` | `1` enables; default off in the library, `1` in `serve` (`bin/serve.rs:2313`) | tensor-core FP4 paths for routed and dense GEMVs | operating | read once through a `OnceLock`, so `serve` must set it before `cuda::Ctx::init` (`bin/serve.rs:2301-2312`) |
| `CROW_MMA_DENSE` | `engine/src/gen.rs:1178` | `0` disables; default follows `CROW_MMA` | dense-GEMV MMA switch, routed MoE MMA stays on | measurement | A/B knob for the dense #10 paths |
| `CROW_MMA_KS` | `engine/src/gen.rs:1133` | `1` to `4`; other values fall back; default `4` | MMA k-split factor, block = `128 * KS` threads | operating | `KS=1` is the original single-slice kernel, bit-identical (`gen.rs:1129`) |
| `CROW_QFUSE` | `engine/src/gen.rs:1271` (cascade `qfuse_on`), `engine/src/gen.rs:1289` (hc fusion `hc_fuse_on`), `engine/src/gen.rs:1327` (shared fusion `sh_fuse_on`) | value table: unset or any value but `0` = cascade ON + hc chain FUSED + shared chain FUSED (the default of record since #19i, 2026-09-13 — the completed launch-fusion default set); `1` = the same all-fused state (the #19h opt-in value, kept accepted, now redundant); `0` = all OFF (separate `quant_x_fp4` launches + the unfused 8-launch hc chain + the unfused 6-launch shared chain) | producers emit the NVFP4 activation cascade, no separate `quant_x_fp4` launch; the hc-chain meaning is the 19f fusion, default since #19g (`hc_down_inj` `kernels.rs:871`, `gemv_bf16_ws` `kernels.rs:947`, launch sites `gen.rs:1695`/`gen.rs:2768`); the shared-expert decode chain fusion (#19h, 2026-09-13, proven opt-in `1`) fuses on the SAME default since #19i: `sh_gate_up_q` (`kernels.rs:3047`) is ONE launch replacing the two shared gate|up `gemv_fp4_mma_d` GEMVs plus `silu_mul640_q` (`kernels.rs:3016`): the mma_d body runs twice verbatim (same lane split, same k walk, same ks-split reduce) over the gate and up slabs against the same `xq_gu` row, the silu_mul640_q math applies warp-wide on the finished accumulator pairs keeping the standalone 16-consecutive-j warp layout so the quant16_store shuffle groups are unchanged, writes `sh2` + `xq_s` (the `sh12` write is dead), and `gemv_fp4_mma_dg` (`kernels.rs:3143`) is the down `gemv_fp4_mma_d` with the `gate_shared` (`kernels.rs:1106`) epilogue at the store: `moe_out = sigmoid(sgv) * down`, ASSIGN, still the FIRST writer of `moe_out` (no memset, graph-capturable); the `sgv` `gemv_b` hoists before the down launch (reads `mixed_m` only); 6 -> 3 launches per layer x 48 layers = 144 launches removed, decode `t < 8` + the mma/dense path only (prefill `gemm_fp4_dense` and the `gemv_fp4_bs` fallback keep the separate launches verbatim), launch site `gen.rs:2359`, the `[hc]` boot line names all three states (`gen.rs:759`) | operating | cascade bit-identical per `gen.rs:1271`; the shared fusion bit-identical by construction (epilogue folding only: every op applies elementwise, or warp-wide with the standalone quant layout, on a finished accumulator, each k walk in the separate-launch order, the merged grid keeps every row's dot product), so the default flip changes only the predicate; the 19h opt-in proofs (`decode_out/srv-19h.log`, 2026-09-13, RTX 5090): parity 12 of 12 GREEN (OFF 8 rows byte-identical to the 19g-committed binary `42c6419f2cd6` with the sha256 of record `bceba6ff7724`, the ON 8-row form byte-identical to the OFF dump with the same sha, the P8FUSE form (504 teacher-forced t=1 decode rows, `CROW_GRAPH=0 CROW_PARITY_PREFILL=8`) no-env vs `CROW_QFUSE=1` byte-identical to each other and BOTH carrying the teacher-forced sha256 of record `b7f6419203b4`), pairs ids `5098f885ab3a` 7 of 7 runs, -0.2286 ms per token 3 of 3 pairs, inside the 19c-pre rank-2 estimate band 0.20 to 0.30 ms; the #19i DEFAULT verification (`decode_out/srv-19i.log`, repair HEAD `be7bb60`, build `5c7919276203`, 2026-09-13, RTX 5090): the trimmed 19g battery (no PXBOTH) with NO env — 8 rows `bceba6ff7724`, 512 `14c8628acbec` x3, 1024 `b2e87b2bf99a`, P8 `b7f6419203b4` — 20 of 20 subchecks GREEN, every no-env form byte-identical to the reference `d211ab52ad2b` at the 61b sha256 values of record, plus the `0` fallback 8 rows at the absolute `bceba6ff7724` with the unfused/`0` boot lines (the 19g INFO basis: the cascade-off dump equals the cascade-on dump); smoke run exit 0 with the fused boot lines; ten tasks 10 of 10 equal the standing splits-8 record (t19h = t19g, cross-checked in-log); pairs, the coordinator-approved W + 3N form (NO adjacent B arm post-flip: `0` would also kill the cascade, the 19g artifact): ids `5098f885ab3a` 4 of 4 incl. W, the all-fused default 22.4266 ms per token = 44.6 tok/s (N1 22.4336 / N2 22.4179 / N3 22.4284, spread 0.0157 ms), hard gate < 22.6999 MET, vs the 19g combined default 22.7999 = -0.3733 ms = -1.64 percent and vs the 19h N arm 22.5584 = -0.1318 ms = -0.58 percent; DOCUMENTED DEBT: one variable still carries THREE meanings (coordinator ruling 2026-09-13, speed over naming); a later rename to distinct switches costs one rebuild plus the standard gate forms |
| `CROW_QSA_FAST` | `engine/src/gen.rs:1141` | `0` selects `qsa_select`; default on (`qsa_select_fast`) | dense-regime shortcut in the QSA selector | operating | kernel sites `gen.rs:1708`, `gen.rs:1834`; comment `kernels.rs:1956` |
| `CROW_QSA_PAR` | `engine/src/gen.rs:1302` | `0` selects `qsa_select_fast`; unset or any other value runs the parallel form; default on since #61b | decode QSA top-k as `qsa_select_par_h` (G blocks, 12-bit histogram, `kernels.rs:2177`) plus `qsa_select_par_e` (one block of 1024 threads, threshold refine and ascending emit, `kernels.rs:2195`) instead of `qsa_select_fast` on one block (`gen.rs:1997-2016`) | operating | DEFAULT ON SINCE #61B, 2026-09-12; every engine process prints one `[qsa]` boot line naming the selection it runs (`gen.rs:731`); basis: the adjacent pair of `decode_out/srv-61b.log` (RTX 5090, 2026-09-12, t1-read 16,064 ids, 255 timed steps), 23.9411 against 24.8735 ms per token with `0`, -0.9324 ms = -3.75 percent at 19.2 x the fallback spread, ids `5098f885ab3a` in 6 of 6 runs, parity 8 of 8 forms against `d211ab52ad2b` including the new radix-path PX form; the nsys selection row fell 1.1101 to 0.0927 ms per token (`decode_out/srv-61a.log`); decode path only, the prefill launch at `gen.rs:1871` keeps `qsa_select_fast`; `0` is the fallback of record |
| `CROW_QSA_PAR_BLOCKS` | `engine/src/gen.rs:1309` | integer, default 32, clamped to 4 .. 256 | block count G of `qsa_select_par_h` | operating | the operating value is 32 since #61b, the measured form of record (61a: 16 and 32 tie inside the fallback spread, `decode_out/srv-61a.log`); the histogram is order free, so the count never moves a bit of the selection list |
| `CROW_ROUTER_GEMM` | `engine/src/gen.rs:1963` | `1` enables; default off | bf16 tensor-core GEMM on the exact bf16 router twin at `t >= 8` | measurement | `gen.rs:1964-1965`: summation order differs from `gemv_b`, not bit-identical, env-gated until validated |

## Adaptation and hot set (9 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_ADAPT` | `engine/src/bin/decode.rs:94` | `1` enables; default off | re-cuts the hot set once after prefill | measurement | also `bin/decode.rs:155`, `bin/parity.rs:178`; harness only, `serve` never calls `adapt_tick` (#37, `docs/architecture.md:1181`); the gate chains set `=1` |
| `CROW_ADAPT_DECAY` | `engine/src/gen.rs:3159` | float; default `0.5` | decay of the selection window used by the adaptation tick | measurement and `serve` | also `gen.rs:3180`; read only when `CROW_ADAPT_WINDOW=1`, which `serve` now sets itself, so this path is the `serve` default since #37 fix round 1 |
| `CROW_ADAPT_EVERY` | `engine/src/geo.rs:169` | integer; default `0` = no tick | re-cut interval in decode tokens | measurement | overridden by the long-context policy when `CROW_ADAPT_STREAM` is unset and chunk >= 2048 (`geo.rs:174`) |
| `CROW_ADAPT_MAX` | `engine/src/geo.rs:169` | integer; default `8` | maximum swaps per layer per tick | measurement | same policy override as `CROW_ADAPT_EVERY` |
| `CROW_ADAPT_MAX0` | `engine/src/bin/decode.rs:96` | integer; default `0` = unbounded | caps the post-prefill swaps per layer (#21) | measurement | also `bin/decode.rs:158`, `bin/parity.rs:179` |
| `CROW_ADAPT_SPARE` | `engine/src/geo.rs:169` | integer; default `0`, or `1` with `CROW_ADAPT_STREAM=1` | spare hot slots held for the trickle | measurement | policy value is `7` at chunk >= 2048 (`geo.rs:174`) |
| `CROW_ADAPT_STREAM` | `engine/src/geo.rs:170` | `1` manual stream trickle, `0` manual compute-stream swaps; default: policy by chunk | selects the adaptation mode and switches every knob to manual | measurement | #17, 2026-09-05, measured on the ten-task series: trickle gains 0.1 to 0.8 tok/s at chunk 2048 and loses 0.2 to 1.0 tok/s at chunk 512 (`geo.rs:154-159`); `gen.rs:3237` asserts spare slots exist |
| `CROW_ADAPT_WINDOW` | `engine/src/gen.rs:3156` | `1` enables; default off in the library, `1` in `serve` (`bin/serve.rs:2313`) | ranks by a decayed count of selections since the last tick | measurement and `serve` (`serve` sets it to `1` when unset) | also `gen.rs:3174`; `gen.rs:3151`: swaps are the same exact three-way exchange, numerics untouched; #37 made `serve` tick the trickle, and its fix round 1 made `serve` set this variable when unset, so an explicit `CROW_ADAPT_WINDOW=0` still restores the CUMULATIVE ranking; read per tick, not through a `OnceLock` |
| `CROW_SWAP_BUNDLE` | `engine/src/gen.rs:1098` | `1` enables; default off | exchanges all pairs of a tick in one launch per layer (#17) | measurement | comment site `gen.rs:3195` |
| `CROW_TRICKLE_DEFER` | `engine/src/gen.rs:1223` | `0` selects the eager order; unset or any other value defers; default 1 | the stream trickle's side-stream copies are parked in `trickle_tick` and issued in `decode_step` right AFTER the graph launch (`Engine::trickle_drain_after_launch`, `gen.rs:3449`), so they overlap the replay instead of holding the one async copy engine while the launch waits behind the scalar refreshes; `0` restores the pre-#63c order, the issue inside `trickle_tick` before the launch (`gen.rs:3412-3422`) | operating | DEFAULT SINCE #63c, 2026-09-12; every engine process prints one `[trickle]` boot line naming the order it runs (`gen.rs:721`); 63b measured the pair on the #59 profile arm, 24.80 against 26.42 ms per decode token, and 63a measured the eager form at 2.5968 ms per token of copies, all in class b (before the graph), 0 inside a graph span (`decode_out/srv-19d-crow-NK.sqlite`); the table flip and `event_record(ev_commit)` stay before the launch, so the side stream's queue order is the same in both settings; under `CROW_PROFILE=1` the deferred copy issue (about 0.24 ms per token) is counted in the `head` bucket, not in `tail` |

## Sampler (8 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_SAMPLE` | `engine/src/sample.rs:70` | `1` enables; default off = greedy argmax | builds the data-sheet sampler from the environment | measurement | `sample.rs:9`: greedy stays the gate discipline; `serve` builds one sampler per request and does NOT read env (`sample.rs:57-61`) |
| `CROW_SAMPLE_HOST` | `engine/src/sample.rs:44` | `1` keeps the host path; default device sampler | logits readback plus host top-k instead of `sample_k` behind `argmax_k` | measurement | reference path per `sample.rs:14` |
| `CROW_TEMP` | `engine/src/sample.rs:76` | float; default `0.7` | sampler temperature | measurement | data-sheet instruct profile; read only when `CROW_SAMPLE=1` |
| `CROW_TOP_P` | `engine/src/sample.rs:77` | float; default `0.8` | nucleus threshold | measurement | read only when `CROW_SAMPLE=1` |
| `CROW_TOP_K` | `engine/src/sample.rs:78` | integer; default `20`, clamped to 64 by the device sampler (`SAMPLE_MAXK`, `engine/src/kernels.rs:2942`) | top-k cut | measurement | read only when `CROW_SAMPLE=1` |
| `CROW_PRESENCE` | `engine/src/sample.rs:79` | float; default `1.5` | presence penalty over the tokens of this answer | measurement | read only when `CROW_SAMPLE=1` |
| `CROW_SEED` | `engine/src/sample.rs:80` | integer; default `0` | sampler seed, also the RNG start state (`sample.rs:81`) | measurement | `sample.rs:53`: the record names the seed, so a sampled answer is reproducible |
| `CROW_STOP_EOS` | `engine/src/sample.rs:192` | `1` enables; default off | ends a harness run at EOS | measurement | `bin/serve.rs:194`: `sample::EOS_IDS` stops both server modes and this variable is NOT read there |

## Server (1 row)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_PREFIX_CACHE` | `engine/src/cache.rs:309` | `0` disables; default on | allocates the one held-conversation slot (M1, spec 7.7) | operating | `0` makes every request a cold start and prints `L n/a` (`bin/serve.rs:386`, `bin/serve.rs:397`, `bin/serve.rs:2324`); error texts at `slot.rs:493`, `slot.rs:585` |

## Tokenizer (2 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_TOKENIZER` | `engine/src/tokenizer.rs:359` | path; default `DEFAULT_TOKENIZER` = `models/Qwen3.8-Flash-Next-original/tokenizer.json` | HF tokenizer file, repository relative | operating | table at `tokenizer.rs:13` |
| `CROW_TOKENIZER_CONFIG` | `engine/src/tokenizer.rs:360` | path; default the sibling `tokenizer_config.json` of the tokenizer | source of the `chat_template` field | operating | table at `tokenizer.rs:14` |

## Dumps and profiling (6 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_PROFILE` | `engine/src/gen.rs:56` | any value; default off | per-section CPU microseconds, printed as ms per step | measurement | also `bin/decode.rs:274`; report format at `gen.rs:52` |
| `CROW_KPROF` | `engine/src/kernels.rs:3146` | any value; default off | per-kernel GPU time, sync before and after each launch | measurement | the number is kernel time plus one launch latency (`kernels.rs:3138-3140`); referenced by `docs/architecture.md:536`, `:911` |
| `CROW_DUMP_H` | `engine/src/gen.rs:2251` | directory path; default off | writes `.f32` stage dumps for the determinism bisect | diagnostic | also `gen.rs:2263`, `gen.rs:2307`, `gen.rs:2680`, `gen.rs:2766`, `gen.rs:2795`; forces `cuda::sync()` at every dump point, so timings taken with it are not serving numbers |
| `CROW_DROP_DBG` | `engine/src/cuda.rs:260` | `1` enables; default off | prints free VRAM and live allocation counts at named points | diagnostic | #18 leak hunt |
| `CROW_GRAPH_DBG` | `engine/src/gen.rs:2860` | any value; default off | prints graph capture progress markers | diagnostic | effective only while `CROW_GRAPH=1`; markers at `gen.rs:2922` to `gen.rs:3047` |
| `CROW_ROUTE_DUMP` | `engine/src/gen.rs:2861` | any value; default off | logs the routed expert ids per token per layer into `Engine::route_log` | diagnostic | non-graph decode only; `bin/decode.rs:313` sets it for `routestats` together with `CROW_GRAPH=0`; the log grows per token and never shrinks (`reset.rs:20`, `cache.rs:49`) |

## Tooling and process (2 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_LOCK` | `engine/src/gen.rs:3368` | `0` disables, any other non-empty value is a path; default `engine/.engine.lock` | one engine per machine, refuses the start before anything is pinned | operating | 2026-09-05: two engines pin 2 x 45 GiB and froze the 64 GB host twice on 2026-09-04 (`gen.rs:3361-3363`); refusal text `gen.rs:3390` |
| `CROW_PARITY_PREFILL` | `engine/src/bin/decode.rs:71` | integer, clamped to `[1, ids.len()]`; default `ids.len()` | prefills only `ids[..n]` and feeds the rest teacher-forced | measurement | #11, 2026-09-05: decode-path rows against prefill-path rows under the same context |

## Diagnostic values with wrong output by design

| Value | Read at | What it does | Warning |
|---|---|---|---|
| `CROW_ATTN_R=8` | `engine/src/gen.rs:1161`, named `attn_sel_d8` at `gen.rs:1162` | attention without the K dot product (`gen.rs:1157`) | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_ATTN_R=9` | `engine/src/gen.rs:1161`, named `attn_sel_d9` at `gen.rs:1162` | attention without the V loop (`gen.rs:1157`) | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_PF_ASYNC=3` | `engine/src/gen.rs:504`, honoured in the kernels at `engine/src/kernels.rs:2322` and `kernels.rs:2778` | the stage kernel returns before it copies and the host issues no copy either, so the tile GEMMs read stale staging slots | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_PF_ASYNC=4` | `engine/src/gen.rs:504`, branch at `engine/src/gen.rs:2145` | copy-only floor: the copy engine stages but no tile GEMM, SiLU or quant kernel runs | WARNING: wrong output by design, diagnostic only, never in a gate |

- These four values are reachable in the default release binary.
- The `--diag` cargo feature alternative of the E5 gate was not taken, so no parity run proves them unreachable.
- Any measurement carrying one of these four values is void as an answer-quality or parity result.

## Reference operating point

- Environment of B4 part 3 (the F49 chain), `decode_out/srv-b4.log:216`
- Same names on issue #11: comments 5548346118, 5549858612, 5550904582 (F49 chains, with `CROW_PLE_CACHE_MB=512`) and 5621121049 (C1, without it)

```
CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json
CROW_GRAPH=1 CROW_MMA=1 CROW_ADAPT=1
CROW_CHUNK=2048 CROW_CHUNK_AUTO=1
CROW_ADAPT_EVERY=32 CROW_ADAPT_MAX=8 CROW_ADAPT_MAX0=0
CROW_ADAPT_WINDOW=1 CROW_ADAPT_DECAY=0.5
```

Unset in that chain, listed so the absence is on the record:

```
CROW_PLE_CACHE_MB CROW_ADAPT_STREAM CROW_ADAPT_SPARE
CROW_SAMPLE CROW_SAMPLE_HOST CROW_TEMP CROW_TOP_P CROW_TOP_K CROW_PRESENCE CROW_SEED
CROW_PF_ASYNC CROW_PF_TG CROW_STAGE_SPLIT CROW_STAGE_DMA CROW_CNQ_PURGE CROW_KPROF CROW_PROFILE
```

- `CROW_SAMPLE` unset means greedy argmax (`bin/parity.rs:194`, `sample.rs:70`), and the record header of `parity` says so since `#53` (`bin/parity.rs:287`).
- The C1 seed series adds `CROW_SAMPLE=1` and `CROW_SEED=2|3|4` to the same block.
- The chain sets `CROW_ADAPT=1`, `CROW_CHUNK_AUTO=1` and `CROW_ADAPT_WINDOW=1`; all three are off by default in the library. This is a deliberate deviation, not a documentation error.
- `CROW_ADAPT_WINDOW=1` is no longer a deviation for `serve`: `serve` sets it itself when it is unset (#37 fix round 1).
- The chain sets `CROW_GRAPH=1` and `CROW_MMA=1` explicitly; `serve` sets those two and `CROW_ADAPT_WINDOW` itself when they are unset (`bin/serve.rs:2313`).
- The chain leaves `CROW_PF_ASYNC` unset, so the default `2` is in force.

## Checking this file

```
python tools/check_env_docs.py
python tools/check_env_docs.py --list
python tools/check_env_docs.py --doc <path to a copy>
```

| Exit code | Meaning |
|---|---|
| `0` | code list and doc list are equal |
| `1` | at least one difference; both directions are printed |

- E7 will run `python tools/check_env_docs.py` in CI without a GPU.
- Adding a `CROW_*` name to `engine/src` or `converter/src` without a row here fails that job.
