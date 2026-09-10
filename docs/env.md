# Environment variables of the crow-nest engine

## Scope and rule

| Item | Value |
|---|---|
| Variables in this table | 65 |
| Distinct `CROW_[A-Z0-9_]+` tokens in the code | 65 in `engine/src`, 0 in `converter/src` |
| Measured | 2026-09-10, task E5, issue #46, parent #1 |
| Repository state | branch `release-v0.1`, HEAD `1d08bcb`, 2026-09-11 |
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
| `env_on(name)` | `engine/src/gen.rs:1117` | true unless the value is exactly `0`; default on |
| `num(k)` | `engine/src/geo.rs:168` | parse to `usize`, `None` when unset or unparsable |
| `f(k, d)` / `u(k, d)` | `engine/src/sample.rs:73`, `:74` | parse to `f32` / `usize`, fall back to `d` |

- Scope: `engine/src` and `converter/src` only; shell variables of Crow (for example `CROW_TAVILY_KEY`) have no engine read site and no row.
- The `--diag` cargo feature alternative of the E5 gate was not taken; no engine change before the tag.
- The four diagnostic values stay reachable in the default binary and carry a warning row (section Diagnostic values).

## Container and residency (13 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_CNQ` | `engine/src/bin/decode.rs:40` | path; default `../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` in `decode`, `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` in `parity`, `DEFAULT_CNQ` in `serve` (`bin/serve.rs:2270`, constant at `bin/serve.rs:446`) | selects the container file | operating | also read at `bin/parity.rs:150`, `bin/parity.rs:341`, `bin/residency.rs:45`, `bin/sf_scan.rs:7`; `decode`, `parity` and `serve` share the `-M` default since `#51` (2026-09-11); the two generator bins keep `../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq`; chain value `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` |
| `CROW_HOTSETS` | `engine/src/bin/decode.rs:45` | path; default `../decode_out/hotsets-M-longctx2100-n160.json` in `decode`, `decode_out/hotsets-M-longctx2100-n160.json` in `parity`, `DEFAULT_HOTSETS` in `serve` (`bin/serve.rs:2271`, constant at `bin/serve.rs:447`) | overrides the hot set sidecar | operating | also `bin/parity.rs:156`; the default is a literal since `#51` (2026-09-11), no longer `<cnq>.hotsets.json`, because the sidecar next to the `-M` container is ragged (`#49`); chain value `decode_out/hotsets-M-longctx2100-n160.json` |
| `CROW_COLD_TIER` | `engine/src/residency.rs:231` | path to `<cnq>.cold<bits>.bin`; default unset (exact NVFP4 tier) | installs the low-bit cold tier built by `bin/coldtier.rs` | operating | record sizes read from the header at `gen.rs:685`; `docs/architecture.md:1163` keeps it off for `serve` (7.5 condition 2) |
| `CROW_COLD_FULL` | `engine/src/gen.rs:692` | `1`, `0`; default: full tier when `E * unit <= cfg.host_pinned_budget` | forces the full tier (every expert pinned) or cold-only | operating | full tier is what enables the prompt-adaptive hot set |
| `CROW_RAM_MARGIN_GB` | `engine/src/residency.rs:249` | integer GiB; default `3` | free physical RAM that must remain after pinning the cold tier | operating | below the margin `residency.rs:255` panics before anything is pinned |
| `CROW_PINNED_WC` | `engine/src/residency.rs:275` | `0` restores cacheable pinned memory; default write-combined | allocation type of the pinned cold slabs | operating | measured 2026-09-04 (`pcie_probe`): WC 47.6 GB/s vs cacheable 24 GB/s |
| `CROW_MMAP` | `engine/src/cnq.rs:91` | `0` disables; default on | maps the container read-only, row reads become page-cache memcpys | operating | doc comment `cnq.rs:29`: about 1 us on a hit instead of a seek plus read pair |
| `CROW_CNQ_PURGE` | `engine/src/cnq.rs:46` | `0` keeps the cache; default purge | on drop, re-opens the container unbuffered to purge its cached pages | operating | rationale `cnq.rs:38-42`, 2026-09-06: 3.3 GB of tier reads stayed in the system cache and `residency.rs` then refused to pin |
| `CROW_QSA_FULL` | `engine/src/manager.rs:37` | `1` sets `ring = context`; default `ceil4(prompt_chunk + 4)` capped by context | size of the raw indexer-key ring per attention layer | measurement | pre-2026-09-05 layout; `docs/architecture.md:1162` marks it out of scope for `serve` |
| `CROW_KV` | `engine/src/bin/decode.rs:62` | `bf16`; default the container KV dtype | parity ladder switch for the KV cache dtype | measurement | read only in `bin/decode.rs`, not by `serve` |
| `CROW_PLE` | `engine/src/bin/decode.rs:65` | `off`; default on | parity ladder switch that disables the PLE stage | measurement | read only in `bin/decode.rs`, not by `serve` |
| `CROW_PLE_CACHE_MB` | `engine/src/gen.rs:626` | integer MiB; default `cfg.ple_cache_bytes` (128 MB in `serve`, `bin/serve.rs:47`) | size of the PLE hot-row cache | operating | VRAM diet knob, 2026-09-05; C1 allowlist keeps it unset (issue #11, comment 5621121049, C1 result) |
| `CROW_PLE_PREFETCH` | `engine/src/gen.rs:2463` | `0` disables; default on | warms the next chunk's PLE rows on a helper thread | operating | no-op without a file mapping (`CROW_MMAP=0`) |

## Prefill and chunking (9 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_CHUNK` | `engine/src/geo.rs:139` | integer tokens; default: policy by prompt length | authoritative prefill chunk size | operating | also `bin/decode.rs:58`, `bin/decode.rs:387`; `serve` pins 2048 for the process and does not call the policy (`bin/serve.rs:41-43`) |
| `CROW_CHUNK_AUTO` | `engine/src/geo.rs:143` | `1`, `0`; default `1` when `CROW_CHUNK` is unset, `0` when it is set | applies the length policy on top of an explicit chunk, cap `max(CROW_CHUNK, 2048)` | operating | default since 2026-09-05; gated per `geo.rs:136-137`: chunk 1024 and 2048 deterministic since #22 |
| `CROW_CHUNK_BALANCE` | `engine/src/gen.rs:2448` | `1` enables; default off | cuts the prompt into equal chunks instead of full chunks plus a tail | measurement | measured 2026-09-04: 3 x 700 gives 382 tok/s and loses to 1024+1024+52 at 414 tok/s; opt-in only |
| `CROW_PF_ASYNC` | `engine/src/gen.rs:502` | `0`, `1`, `2`, `3`, `4`; default `2` | staging path of the prefill cold experts | operating | default since 2026-09-09 (#10, robin's call), gated: parity 8/512/1024 plus ten tasks ids equal to final4 without the env; values `3` and `4` are diagnostics, see the warning table |
| `CROW_PF_DMA` | `engine/src/gen.rs:485` | `1` enables; default off | copy-engine prefetch ring of one layer's cold slab during prefill | measurement | measured 2026-09-04: with the exact NVFP4 tier the ring makes the plan infeasible; `gen.rs:713` disables it for the run when the ring does not fit |
| `CROW_PF_GEMM` | `engine/src/gen.rs:490` | `0` disables; default on | expert-grouped tile GEMM for prefill-sized MoE batches | operating | `gen.rs:1874`: a low-bit cold tier needs this staging path |
| `CROW_PF_TG` | `engine/src/gen.rs:462` | integer, clamped to `[8, 512]`; default `PF_TG` = 64 | tiles per staged group | operating | default 64 since 2026-09-09 (#10, gated on ten tasks with `CROW_PF_ASYNC=2`; 32 before); costs `CROW_PF_TG` x 2.76 MB of staging slots |
| `CROW_STAGE` | `engine/src/gen.rs:517` | `0` selects direct zero-copy; default on | stages cold experts into VRAM with coalesced PCIe reads before the routed GEMVs | operating | decode path only (`t * TOPK <= stage.max`) |
| `CROW_STAGE_SPLIT` | `engine/src/gen.rs:510` | `1`, `2`, `4`, `8`, `16`, `32`, `64`; other values fall back; default `8` | blocks per expert in the staging copies | measurement | measured 2026-09-04 with the WC tier: 16/32/64 keep more PCIe requests in flight |

## Attention and kernels (15 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_ATTN_R` | `engine/src/gen.rs:1140` | `0`, `1`, `2`, `3`, `4`, `5`, `8`, `9`; default unset = `attn_sel_s8l` | selects the decode attention kernel (`gen.rs:1141`) | operating | default since 2026-09-06 late (#10, robin's call), gated by parity 8/512/1024 against the previous build plus ten tasks; `8` and `9` are diagnostics, see the warning table |
| `CROW_ATTN_SB` | `engine/src/gen.rs:1151` | `0` restores the full-chunk buffer; default on (`ATTN_SB` = 512) | prompt attention in sub-batches of 512 tokens | operating | #16, 2026-09-05: shrinks the QSA score buffer from 512 MB at chunk 2048; bit-identical by construction |
| `CROW_ATTN_SPLIT` | `engine/src/gen.rs:1130` | `0` disables; default on | decode attention as 8 partials plus merge, QSA scores warp per block | operating | both forms are graph-static |
| `CROW_BF16_GEMM` | `engine/src/gen.rs:120` | `0` selects the warp GEMV; default on | 8-token bf16 tile GEMM for prefill-sized batches | operating | applies only to `PW::Bf16` weights at `t >= 8` |
| `CROW_BF16_W` | `engine/src/gen.rs:1121` | `0` selects `gemv_bf16_b` / `gemv_bf16`; default on | selects `gemv_bf16_w` (8 rows per block) for bf16 GEMVs and the LM head | operating | step-2 kernel switch; LM head site `gen.rs:2300` |
| `CROW_DENSE_GEMM` | `engine/src/gen.rs:1097` | `0` selects the per-token MMA GEMV; default on | 8-token tile GEMM for dense FP4 projections at `t >= 8` | operating | reached only when `CROW_MMA=1` (`launch_mma_d`) |
| `CROW_GDN_REG` | `engine/src/gen.rs:473` | `0` off, `p` prefill scan only, `s` decode step only, anything else both; default both | register delta-rule scan instead of the global-memory scan | operating | `0` is the bit-identical fallback per `gen.rs:468` |
| `CROW_GRAPH` | `engine/src/gen.rs:1088` | `1` enables; default off in the library, `1` in `serve` (`bin/serve.rs:2263`) | captures the per-token kernel sequence once and replays it | operating | fable gate 2026-09-03 (WDDM launch overhead); `serve` sets it only when unset, so `CROW_GRAPH=0` still wins |
| `CROW_INJ_1K` | `engine/src/gen.rs:1122` | `0` selects `gemv_fp4_b`; default on (`gemv_fp4_b1k`, 1024 threads) | kernel of the block-inject GEMV | operating | `gen.rs:1421-1424`: the MMA tile is about 10x slower here, documented skip |
| `CROW_MMA` | `engine/src/gen.rs:1077` | `1` enables; default off in the library, `1` in `serve` (`bin/serve.rs:2263`) | tensor-core FP4 paths for routed and dense GEMVs | operating | read once through a `OnceLock`, so `serve` must set it before `cuda::Ctx::init` (`bin/serve.rs:2257-2260`) |
| `CROW_MMA_DENSE` | `engine/src/gen.rs:1157` | `0` disables; default follows `CROW_MMA` | dense-GEMV MMA switch, routed MoE MMA stays on | measurement | A/B knob for the dense #10 paths |
| `CROW_MMA_KS` | `engine/src/gen.rs:1112` | `1` to `4`; other values fall back; default `4` | MMA k-split factor, block = `128 * KS` threads | operating | `KS=1` is the original single-slice kernel, bit-identical (`gen.rs:1108`) |
| `CROW_QFUSE` | `engine/src/gen.rs:1126` | `0` disables; default on | producers emit the NVFP4 activation cascade, no separate `quant_x_fp4` launch | operating | bit-identical per `gen.rs:1125` |
| `CROW_QSA_FAST` | `engine/src/gen.rs:1120` | `0` selects `qsa_select`; default on (`qsa_select_fast`) | dense-regime shortcut in the QSA selector | operating | kernel sites `gen.rs:1687`, `gen.rs:1813`; comment `kernels.rs:1956` |
| `CROW_ROUTER_GEMM` | `engine/src/gen.rs:1857` | `1` enables; default off | bf16 tensor-core GEMM on the exact bf16 router twin at `t >= 8` | measurement | `gen.rs:1858-1859`: summation order differs from `gemv_b`, not bit-identical, env-gated until validated |

## Adaptation and hot set (9 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_ADAPT` | `engine/src/bin/decode.rs:94` | `1` enables; default off | re-cuts the hot set once after prefill | measurement | also `bin/decode.rs:155`, `bin/parity.rs:168`; harness only, `serve` never ticks adaptation (`docs/architecture.md:1161`); the gate chains set `=1` |
| `CROW_ADAPT_DECAY` | `engine/src/gen.rs:3051` | float; default `0.5` | decay of the selection window used by the adaptation tick | measurement | also `gen.rs:3072`; read only when `CROW_ADAPT_WINDOW=1` |
| `CROW_ADAPT_EVERY` | `engine/src/geo.rs:169` | integer; default `0` = no tick | re-cut interval in decode tokens | measurement | overridden by the long-context policy when `CROW_ADAPT_STREAM` is unset and chunk >= 2048 (`geo.rs:174`) |
| `CROW_ADAPT_MAX` | `engine/src/geo.rs:169` | integer; default `8` | maximum swaps per layer per tick | measurement | same policy override as `CROW_ADAPT_EVERY` |
| `CROW_ADAPT_MAX0` | `engine/src/bin/decode.rs:96` | integer; default `0` = unbounded | caps the post-prefill swaps per layer (#21) | measurement | also `bin/decode.rs:158`, `bin/parity.rs:169` |
| `CROW_ADAPT_SPARE` | `engine/src/geo.rs:169` | integer; default `0`, or `1` with `CROW_ADAPT_STREAM=1` | spare hot slots held for the trickle | measurement | policy value is `7` at chunk >= 2048 (`geo.rs:174`) |
| `CROW_ADAPT_STREAM` | `engine/src/geo.rs:170` | `1` manual stream trickle, `0` manual compute-stream swaps; default: policy by chunk | selects the adaptation mode and switches every knob to manual | measurement | #17, 2026-09-05, measured on the ten-task series: trickle gains 0.1 to 0.8 tok/s at chunk 2048 and loses 0.2 to 1.0 tok/s at chunk 512 (`geo.rs:154-159`); `gen.rs:3129` asserts spare slots exist |
| `CROW_ADAPT_WINDOW` | `engine/src/gen.rs:3048` | `1` enables; default off (cumulative re-cut) | ranks by a decayed count of selections since the last tick | measurement | also `gen.rs:3066`; `gen.rs:3043`: swaps are the same exact three-way exchange, numerics untouched |
| `CROW_SWAP_BUNDLE` | `engine/src/gen.rs:1084` | `1` enables; default off | exchanges all pairs of a tick in one launch per layer (#17) | measurement | comment site `gen.rs:3087` |

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
| `CROW_DUMP_H` | `engine/src/gen.rs:2143` | directory path; default off | writes `.f32` stage dumps for the determinism bisect | diagnostic | also `gen.rs:2155`, `gen.rs:2199`, `gen.rs:2572`, `gen.rs:2658`, `gen.rs:2687`; forces `cuda::sync()` at every dump point, so timings taken with it are not serving numbers |
| `CROW_DROP_DBG` | `engine/src/cuda.rs:260` | `1` enables; default off | prints free VRAM and live allocation counts at named points | diagnostic | #18 leak hunt |
| `CROW_GRAPH_DBG` | `engine/src/gen.rs:2752` | any value; default off | prints graph capture progress markers | diagnostic | effective only while `CROW_GRAPH=1`; markers at `gen.rs:2814` to `gen.rs:2939` |
| `CROW_ROUTE_DUMP` | `engine/src/gen.rs:2753` | any value; default off | logs the routed expert ids per token per layer into `Engine::route_log` | diagnostic | non-graph decode only; `bin/decode.rs:313` sets it for `routestats` together with `CROW_GRAPH=0`; the log grows per token and never shrinks (`reset.rs:20`, `cache.rs:49`) |

## Tooling and process (2 rows)

| Name | Read at | Values / default | Effect | Mode | Notes |
|---|---|---|---|---|---|
| `CROW_LOCK` | `engine/src/gen.rs:3260` | `0` disables, any other non-empty value is a path; default `engine/.engine.lock` | one engine per machine, refuses the start before anything is pinned | operating | 2026-09-05: two engines pin 2 x 45 GiB and froze the 64 GB host twice on 2026-09-04 (`gen.rs:3253-3255`); refusal text `gen.rs:3282` |
| `CROW_PARITY_PREFILL` | `engine/src/bin/decode.rs:71` | integer, clamped to `[1, ids.len()]`; default `ids.len()` | prefills only `ids[..n]` and feeds the rest teacher-forced | measurement | #11, 2026-09-05: decode-path rows against prefill-path rows under the same context |

## Diagnostic values with wrong output by design

| Value | Read at | What it does | Warning |
|---|---|---|---|
| `CROW_ATTN_R=8` | `engine/src/gen.rs:1140`, named `attn_sel_d8` at `gen.rs:1141` | attention without the K dot product (`gen.rs:1136`) | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_ATTN_R=9` | `engine/src/gen.rs:1140`, named `attn_sel_d9` at `gen.rs:1141` | attention without the V loop (`gen.rs:1136`) | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_PF_ASYNC=3` | `engine/src/gen.rs:502`, honoured in the kernels at `engine/src/kernels.rs:2322` and `kernels.rs:2778` | the stage kernel returns before it copies and the host issues no copy either, so the tile GEMMs read stale staging slots | WARNING: wrong output by design, diagnostic only, never in a gate |
| `CROW_PF_ASYNC=4` | `engine/src/gen.rs:502`, branch at `engine/src/gen.rs:2037` | copy-only floor: the copy engine stages but no tile GEMM, SiLU or quant kernel runs | WARNING: wrong output by design, diagnostic only, never in a gate |

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
CROW_PF_ASYNC CROW_PF_TG CROW_STAGE_SPLIT CROW_CNQ_PURGE CROW_KPROF CROW_PROFILE
```

- `CROW_SAMPLE` unset means greedy argmax (`bin/parity.rs:184`, `sample.rs:70`).
- The C1 seed series adds `CROW_SAMPLE=1` and `CROW_SEED=2|3|4` to the same block.
- The chain sets `CROW_ADAPT=1`, `CROW_CHUNK_AUTO=1` and `CROW_ADAPT_WINDOW=1`; all three are off by default in the code. This is a deliberate deviation, not a documentation error.
- The chain sets `CROW_GRAPH=1` and `CROW_MMA=1` explicitly; `serve` sets the same two itself when they are unset (`bin/serve.rs:2263`).
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
