# Residency replay: `tools/residency_replay.py`

`tools/residency_replay.py` is step 1 of issue #245 ("residency: Flash-Next hot set hits less than a dynamic
cache"). It replays per-position Flash-Next routing offline through candidate residency policies. For each
policy it reports how many routed experts of the generated tokens would have been in VRAM. Step 2, building the
winner in `engine/src/residency.rs`, is gated on this table, and the table does not exist yet: the dumps it
reads are not on this machine's Windows drive (section 3). The tool runs without a GPU and without the model.

```text
python -I tools/residency_replay.py <dump-dir> <mask-dir> <held-out> [<cal>...] [--budgets 144,151,170] \
    [--policies static,today,lru,lfu:0.5,lfu:0.99,freq,trickle-lru,trickle-lfu:0.5] [--prefill skip,batch] \
    [--sets decode_out/hotsets-M-crow0924-n160.json] [--halflife 64] [--margin 1] [--no-reprefill] \
    [--per-span] [--json out.json]
python -I tools/test_residency_replay.py
```

It needs numpy. The test file also imports `tools/glm_tier_sim.py`, which needs numpy too. Both run under the
system Python 3.13.3 / numpy 2.4.6 and under `.venv-oracle`.

## 1. Input

The input is the same as `tools/hotset-eval.py` takes ([hot-set calibration](hotset-calibration.md)):

| file | written by | read as |
|---|---|---|
| `<dump-dir>/<name>-routes.bin` | `CROW_ROUTE_DUMP_PREFILL` during `decode routestats` | the routed ids of every position, `[positions][48][10]`, through `hotset-eval.py`'s own `load_routes` |
| `<dump-dir>/<name>-counts.json` | `decode routestats` (`prefill_counts`) | self-test: the dump's counts must equal the engine's counters, else the run stops |
| `<mask-dir>/<name>-mask.json` | `tools/session_ids.py` | the generated spans, one per assistant message |
| `--sets` | a hot-set sidecar, default `decode_out/hotsets-M-crow0924-n160.json` | the hot set every policy starts from |

Positional `<cal>` names are optional calibration files. They are self-tested like the held-out file. Their
generated-position counts give the `freq` prior. Without them `freq` starts every score at 0, and the run says
so.

## 2. What it computes

**Requests.** Each span of the mask is one request's decode, and the prompt run before it is that request's
prefill. Serve re-prefills the previous answer: request k prefills from the start of span k-1. The evidence is
the engine log of 2026-10-14 (section 4.3). For requests 2 to 10 of that session, the prompt runs of the
rendered session equal the logged `prompt_n - cached_n` token for token: 501, 466, 767, 612, 794, 535, 535, 720
and 411. With `--no-reprefill`, a request prefills from the end of the previous span instead.

**Budgets.** A budget is the number of VRAM expert slots per layer. At 151, today's operating point, that is
144 logical slots + 7 trickle spares (engine.log `operating_point`, 2026-10-12 to 2026-10-14). The start set is
the `--sets` rows brought to N by `residency::sidecar_sets`' rule: truncate to the first N ids, or pad with the
lowest unused ids. crow0924 holds 160 per layer, so at 170 its rows are padded with 10 ids, as the engine
would pad them.

**Policies.** Every policy keeps one pool per layer.

| policy | slots at budget B | rule | source |
|---|---|---|---|
| `static` | B | the start set, never changes | the hot-set sidecar |
| `today` | B - 7 + 7 spares | serve's stream trickle. Tick before decode step i for i > 0; plan when i % 8 == 0. The plan ranks by the decayed window of `CROW_ADAPT_WINDOW=1` (decay 0.5, x1024, truncated), and its first call only sets the base. `plan_swaps` makes at most 7 swaps per layer: the hottest absent experts against the coldest residents, residents winning ties. The incoming expert is resident from the next tick (commit A); the evicted one stays until the tick after (commit B). `trickle_drain` runs at the end of each request. The window counts every prefilled and decoded position | `engine/src/geo.rs:368`, `gen.rs` `trickle_tick` / `decay_window` / `window_layer`, `residency.rs` `plan_swaps`, `serve.rs:4807-4957` |
| `lru` | B | `expert_cache.rs` `Policy::Lru` per token (`observe_token`): the token's ids ascending, one at a time. A miss is admitted and evicts the least recent; that can be a later id of the same token | `expert_cache.rs` `Pool::ranked` |
| `lfu:D` | B | `Policy::Lfu{decay: D}` per token: a tick multiplies every score by D, an access adds 1, and a miss enters when its score is not below the lowest VRAM score (ties go to the more recent access) | `expert_cache.rs`; D 0.5 is `LFU_DECAY_DEFAULT` |
| `freq` | B | GLM-5.3-Flash's frequency tiers per layer. A score is the decayed count of decode routings, half-life `--halflife` tokens, starting at the prior. A miss enters VRAM only above (1 + `--margin`) x the lowest VRAM score not routed in the same call | `tools/glm_tier_sim.py` `FreqTiers` (#188), [GLM tier simulation](glm-tier-simulation.md) section 9 |
| `trickle-lru`, `trickle-lfu:D` | B - 7 + 7 spares | the same caches driven through the trickle the way `Residency::plan_swaps_cached` drives them. The cache observes the cumulative counts every 8 tokens. Incoming experts are its VRAM set's absent ones, most valuable first; out are the residents it does not hold, least valuable first; at most 7 per layer. This is #245 step 2 as the issue words it, with the `CROW_ADAPT_WINDOW` refusal resolved | `residency.rs` `plan_swaps_cached`, `attach_cache` |

`--prefill` decides what a prompt run does to `lru`, `lfu` and `freq`:

- `skip`: the run moves nothing. This is GLM's freq rule, where prompt calls admit nothing.
- `batch`: one `observe_counts` per layer over the run. This is `expert_cache.rs`' rule for cumulative counts, and
  it applies to `lru` and `lfu` only.
- `token`: every prompt position is fed in as a decode token but not scored.

`today` and `trickle-*` see the prompt runs through the cumulative selection counts, as the engine does.

**Output, per budget and policy:**

- the hit rate on the generated positions, which is the primary metric of #106;
- the cold experts per generated token: (1 - hit) x 480;
- the moves per generated token: admissions for the per-token caches, swaps started for the trickle policies;
- the difference against `static`, with the 95 % block-bootstrap CI (`hotset-eval.py`'s `boot_ci`: 1,000-token
  blocks, 2,000 resamples, seed 20260924, needs at least 2 blocks);
- the difference against `today`.

After each budget comes a verdict line, and the last line is the #245 go rule. The rule is GO when a candidate
(any policy but `static` and `today`) beats `static` by at least 0.10. The verdict line also gives the candidate's
absolute hit rate against 0.82, which is crow0924's 0.72 of 2026-09-24 + 0.10, and its margin over `today`.
`--per-span` adds per-request figures for `today`: prompt positions, generated positions, swaps started, cold
prompt choices, cold generated choices and the hit rate. `--json` writes everything, `per_span` included.

**Runtime.** The full default (8 policies, 2 prefill modes, 3 budgets) took 88 s on a synthetic 20,000-position
dump with 12,304 generated positions (system Python, 2026-10-18). Linear in the positions, that is about 9
minutes for the 2026-09-24 held-out (77,271 generated) and about 35 s for the 2026-10-14 rollover (4,955).
`lfu` and `trickle-lfu` are the slowest rows, at about 0.4 ms per generated token.

**Tests.** `tools/test_residency_replay.py`, 24 tests, 0.5 s:

- a hand-computed trace for every policy;
- the budget boundary (truncate at the row length, pad above it);
- the trickle's commit timing, and its window over prompt runs with and without the re-prefill;
- LRU and LFU against a literal port of `expert_cache.rs`, with the per-tick multiply, the linear victim search,
  `observe_token` and `observe_counts`;
- `freq` against `glm_tier_sim.FreqTiers` at one layer, where its single pool is that layer's pool;
- `plan_freq` against a literal port of `plan_swaps`, in-flight exclusions included;
- a round trip of the engine's dump layout;
- the command line on synthetic dumps, one GO case and one NO GO case.

## 3. Data on this machine (2026-10-18)

| what | where | usable for the replay |
|---|---|---|
| per-position dumps `*-routes.bin`, masks `*-mask.json`, the 2026-09-23 rollover archives and the 2026-09-24 held-out `session.json` | none under `C:\Users\robin`, searched to depth 9 and 10, nor in the WSL home | the table needs them |
| `decode_out/routes-tentasks-all.json` (main checkout, 132,354 B, 2026-09-04) | `decode_routes` of 1 decode token, aggregate `prefill_counts`, `hot_sets` at N 148, a 2-id `trace` | no: no per-position order |
| `decode_out/srv-a4-routes.log` (2,664 B) | an HTTP route smoke log (`/health`, `/props`) | no: not expert routing |
| `%LOCALAPPDATA%\Crow\session\crow-session.bin` | a slot save, magic `CROWSLT` | no |
| `%LOCALAPPDATA%\Crow\session\session.json` | 3 messages, model `claude-opus-5-5` | no: not a Flash-Next session |
| `%LOCALAPPDATA%\Crow\session\cleared\*.json` | tool-result archives (`results`), not message lists | no |
| `%LOCALAPPDATA%\Crow\session\rollover-20261014-205358.json` | 28 messages (10 assistant, 16 tool). It was served by the Flash-Next process that booted 2026-10-14T18:44:14Z with `residency_n` 144, `residency_stride` 151 and sidecar crow0924 (engine log `engine-20261015-062916-000.log.gz`) | yes, after a dump (section 4.2); held out from crow0924, which was cut on 2026-09-23 data |
| disk 0 partition 7: GPT type `0fc63daf-…` (Linux filesystem), 363.8 GB, no drive letter | not readable from Windows | probably: the 2026-09-24 dumps were made on Linux (`hotset-calibrate.sh` runs under `systemd-run`, `eval.txt` came from `.venv-oracle/bin/python`) and kept under `decode_out/hotset-0924/` (`.gitignore` since `4876b2c`). Not verified |

The rollover file was rendered on the CPU on 2026-10-18 with `tools/session_ids.py` in `.venv-oracle`
(transformers 5.16.1): 6,661 tokens and 4,955 generated positions in 10 spans, 0 skipped. The spans are
[818, 1252), [1319, 1657), [1785, 2477), [2552, 3077), [3164, 3761), [3958, 4408), [4493, 4998), [5028, 5574),
[5748, 6114) and [6159, 6661).

## 4. Producing the dumps

A dump loads the model on the GPU. It is not run without robin's go.

### 4.1 First, the Linux install

Boot the Arch install and look in its crow-nest checkout for `decode_out/hotset-0924/`. Look for
`session-routes.bin`, `rollover-20260923-195925-routes.bin`, `rollover-20260923-210418-routes.bin` and
`rollover-20260923-225111-routes.bin`, each with its `-counts.json` and `-mask.json`. These are the names in
`decode_out/hotset-0924/eval.txt`. If they are there, no GPU run is needed:

```text
.venv-oracle/bin/python -I tools/residency_replay.py decode_out/hotset-0924 decode_out/hotset-0924 session \
    rollover-20260923-195925 rollover-20260923-210418 rollover-20260923-225111 --json decode_out/hotset-0924/replay-245.json
```

Here `session` is the held-out of #106: 121,955 positions, 77,271 generated, 77 bootstrap blocks. The cross
check is the `static` row: with `--budgets 160` it must read 0.723, the `G` row of `eval.txt`.

### 4.2 On Windows: the 2026-10-14 Flash-Next session

Run this in PowerShell from the main checkout root, `C:\Users\robin\dev\crow-nest`. That root holds
`engine\target\release\decode.exe` (2026-10-14 10:33; the binary contains `CROW_ROUTE_DUMP_PREFILL`), the -M
container, `models\Qwen3.8-Flash-Next-original` and `.venv-oracle`. NVRTC comes from CUDA v13.3 `bin\x64` on
`PATH`.

Preconditions: no other engine is running (Crow's own serve is stopped, one engine per machine), and more than
50.5 GiB of RAM is free (the Windows rule of [getting started](getting-started.md)).

```text
$d = "decode_out\residency-245"; $n = "rollover-20261014-205358"
New-Item -ItemType Directory -Force $d | Out-Null
.venv-oracle\Scripts\python.exe -I tools\session_ids.py "$env:LOCALAPPDATA\Crow\session\$n.json" "$d\$n-ids.json" "$d\$n-mask.json"
$env:CROW_CNQ = "converter\Qwen3.8-Flash-Next-CNQ4.5-M.cnq"
$env:CROW_HOTSETS = "decode_out\hotsets-M-crow0924-n160.json"
$env:CROW_MMA = "1"
$env:CROW_ROUTE_DUMP_PREFILL = "$d\$n-routes.bin"
engine\target\release\decode.exe routestats "$d\$n-ids.json" "$d\$n-counts.json" 0 *> "$d\$n.log"
Remove-Item Env:CROW_ROUTE_DUMP_PREFILL, Env:CROW_CNQ, Env:CROW_HOTSETS, Env:CROW_MMA
python -I tools\residency_replay.py $d $d $n --per-span --json "$d\replay-245.json"
```

This is the command of `tools/hotset-calibrate.sh` without its Linux `systemd-run` scope. What the engine does
with it:

- `decode routestats <ids> <out> 0` (`engine/src/bin/decode.rs:335`) sets `CROW_ROUTE_DUMP=1` and `CROW_GRAPH=0`,
  prefills the ids and writes `prefill_counts`. The trailing 0 means no decode step runs.
- The prefill chunk is `Config::default`'s 512, because `routestats` does not call `apply_chunk_policy`.
- `route_dump_prefill` (`engine/src/gen.rs:6880`, called from `moe_run` for every chunk with more than one
  position) appends one record per chunk and layer: u32 layer, u32 t, then t x 10 u16 ids.
- It creates the file with `File::create`, so an existing dump at that path is overwritten.
- It syncs the device once per chunk-layer.
- A chunk of one position is not dumped (`t > 1`). A prompt whose last chunk is one position long would lose
  that position, and the self-test would refuse the dump. The rollover's last chunk holds 5 positions; the
  2026-09-24 files end in 99, 69, 228 and 443.
- The hot set does not change the routing, because residency is numerically invisible on the exact tier.

**Size.** 6,661 positions x 48 x 10 x 2 B + 14 chunks x 48 x 8 B = 6,399,936 B (6.4 MB). The counts file
holds 24,576 integers, about 0.1 MB.

**Time.**

- The boot pins the 45 GiB cold tier: 104 s for serve on 2026-10-14, from the `[budget]` line at 18:42:29.95Z to
  `operating_point` at 18:44:14.36Z.
- The prefill of 6,661 tokens takes 13 s at the 525 tok/s of the 2026-09-24 Linux calibration (dump on and off).
  At the 124 tok/s of that Windows session's first request it takes 54 s.
- The tokenization took 16 s.
- Together that is 2 to 3 minutes, and one engine start.

The replay itself takes about 35 s.

**Statistical weight.** 4,955 generated positions are 4 bootstrap blocks. The CI is printed, but it is weak,
and a go on this file alone is provisional. The 2026-09-24 held-out (section 4.1) has 77 blocks.

**If the 2026-09-23 sessions are found but not their dumps,** run the same commands once per file. The four
files are 635,719 positions, which comes to 610 MB of dumps. The time is 4 boots of about 2 minutes, plus
20 minutes (at 525 tok/s) to 85 minutes (at 124 tok/s) of prefill.

### 4.3 Checking `today` against the engine's own counters

The process that served the rollover logged one `routing` event and one `[chat] trickle` line per request. Those
lines are in `%LOCALAPPDATA%\Crow\logs\engine-20261015-062916-000.log.gz`, 2026-10-14T18:49:32Z to 18:53:16Z.
`--per-span` prints the same quantities for `today`, and the 151 rows are the ones to compare:

| request | prefilled | generated | trickle swaps | selections | cold | hit |
|---|---|---|---|---|---|---|
| 1 | 7,239 | 432 | 13,966 | 3,682,080 | 1,991,443 | 0.459 |
| 2 | 501 | 336 | 12,623 | 401,760 | 89,808 | 0.776 |
| 3 | 466 | 690 | 25,394 | 554,880 | 129,971 | 0.766 |
| 4 | 767 | 523 | 19,390 | 619,200 | 168,575 | 0.728 |
| 5 | 612 | 595 | 20,786 | 579,360 | 142,338 | 0.754 |
| 6 | 794 | 448 | 14,449 | 596,160 | 133,289 | 0.776 |
| 7 | 535 | 503 | 17,115 | 498,240 | 143,694 | 0.712 |
| 8 | 535 | 544 | 18,867 | 517,920 | 168,468 | 0.675 |
| 9 | 720 | 364 | 12,947 | 520,320 | 123,100 | 0.763 |
| 10 | 411 | 500 | 18,295 | 437,280 | 89,688 | 0.795 |

How the two line up:

- selections = (prefilled + generated) x 480, so the logged cold and hit mix prefill and decode.
- A mask span is the generated count + 2: the closing `<|im_end|>` and newline are never decoded.
- Request 1 prefilled 7,239 tokens, but the render has 818 before span 1. The tool schemas (6,421 tokens) are not
  in the session file, so request 1 and the window of its first ticks cannot match.
- From request 2 on, the swaps per request should be close to the logged ones but not equal. The dump routes the
  rendered session, not the live decode, and the trickle's state carries request 1's difference forward.

A gap of more than a few percent in requests 2 to 10 means the emulation is wrong somewhere. Find it before the
`today` row is used.

## 5. Limits

- **The routing is a prefill of the rendered session**, as in #106. It is not the live decode, and the tool
  schemas are missing from the render.
- **Per-token `lru`, `lfu` and `freq` assume a miss can be kept in VRAM when it is used.** One way would be to keep
  the staged copy of `stage_cold_ca`. `residency.rs` cannot do that today; only the trickle moves experts.
  `trickle-*` is what the existing swap path can execute.
- **LFU is held in the growing-increment form of `FreqTiers`:** an access adds the increment, and a tick multiplies
  the increment by 1 / D. That is the engine's order in exact arithmetic and the same floats for D 0.5 and 1.0.
  For other D, the rounding of near-equal scores can differ from the engine's per-tick multiply.
- **`freq` is per layer.** The GLM arena is one pool for all layers; `residency.rs` is per layer. Without
  calibration files it has no prior.
- **No MTP.** Flash-Next loads no MTP head (`docs/env.md` `CROW_MTP`), so there is one decode step per
  generated position.
- **The engine's own `today` is assumed to start each session in a fresh process,** with the window base set at
  the first plan tick, as after a boot.
