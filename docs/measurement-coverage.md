# crow-nest — expert coverage curve (issue #3, 2026-09-02)

Source: Crow's own routing logs — 15 `print_locality` blocks from 15 server runs in
`dev/crow-lab/runs/` (ladder, locality, parity, tasks series, all 2026-09-01, model
UD-Q2_K_XL quant, 48 expert layers × 512 experts, 10 routed + 1 shared). Read-only;
analysis script: `tools/coverage-curve.py`.

## Measured (per block: fraction of experts covering 50/80/95 % of routed selections)

| statistic | 50 % covered by | 80 % covered by | 95 % covered by | Gini |
|---|---|---|---|---|
| min | 5.2 % (27 experts) | 16.1 % (82) | 31.9 % (163) | 0.761 |
| median | 7.3 % (37) | 20.0 % (100) | 36.4 % (186) | ~0.773 |
| max | 7.8 % (40) | 21.1 % (108) | 40.0 % (205) | 0.811 |

Task-type spread is real and matters: reasoning/writing tasks are more concentrated
(95 % by 163–174 experts), debugging more diffuse (95 % by 205). The ladder runs over
generic traffic cluster tightly (95 % by 183–196).

## Curve

Least-squares through the three median points: **c(N) = −0.5162 + 0.2823 · ln(N)**
(capped 0.995). This is an interpolation of measured points, not a model.

| N/layer | VRAM (GB) | coverage c(N) | P(all 10 resident) = c¹⁰ | cold bytes/token | serial H2D* | fits budget |
|---|---|---|---|---|---|---|
| 32 | 4.2 | 46.2 % | 0.0 % | 713 MB | 28.5 ms | yes |
| 64 | 8.5 | 65.8 % | 1.5 % | 453 MB | 18.1 ms | yes |
| 96 | 12.7 | 77.2 % | 7.5 % | 302 MB | 12.1 ms | yes |
| 128 | 17.0 | 85.3 % | 20.5 % | 194 MB | 7.8 ms | yes |
| **160** | **21.2** | **91.6 %** | **41.7 %** | **111 MB** | **4.4 ms** | **yes** |
| 176 | 23.3 | 94.3 % | 55.8 % | 75 MB | 3.0 ms | yes |
| 192 | 25.4 | 96.8 % | 72.1 % | 43 MB | 1.7 ms | no |
| 256 | 33.9 | 99.5 % | 95.1 % | 7 MB | 0.3 ms | no |

\* bandwidth input (~25 GB/s pinned H2D), serialized — not a throughput prediction.
Budget: 23.5 GB for experts = 32 GiB minus dense 4.5 GB, KV FP8 ~3.0 GiB, GDN/QSA
~0.25, PLE rows ~1.0, activation/graph pools ~2.0 → max 177 experts/layer.
**Correction 2026-09-02 (after section 1 approval):** the approved BF16 keep-set
(embeddings, lm_head, router, gates, norms stay BF16) raises the dense VRAM footprint
by ~1.8 GB — expert budget ~20.9–22.5 GB depending on pool sizing → hard ceiling
~158–169 experts/layer. Default 160 stands; the loader auto-clamps N to the measured
fit at load time (section 2 of the spec).

## Recommendation to the spec (memory layout section)

- Hot-set default **160 experts/layer** (91.6 % coverage, 21.2 GB, ~4 GB total VRAM
  headroom left), **configuration ceiling 176** (94.3 %, 23.3 GB). 192+ does not fit
  next to the 262k/FP8-KV decision.
- Expected per token at N=160: ~39 of 48 layers issue at least one cold job
  (independence estimate, c¹⁰); the streaming path (cold path A) carries them —
  ~111 MB/token, serialized ≈ 4.4 ms, overlappable by design.
- The per-layer policy field (#7) decides per layer by bandwidth; the engine's own
  telemetry refines this curve in operation — the numbers here are design inputs with
  named assumptions, not promises.

## Assumptions (binding for anyone quoting these numbers)

1. Coverage measured through the UD-Q2_K_XL quant on Crow traffic, aggregated over 48
   layers; per-layer top-N residency typically achieves ≥ this global curve.
2. P(all 10 resident) = c¹⁰ assumes independence; correlation within a layer lowers it,
   per-layer selection raises coverage. Refine from engine telemetry.
3. The logs carry per-request aggregates only (no per-token per-layer selections) — a
   per-token dump would be a new measurement, not needed for the hot-set default.
4. The original strand-1 quote (Gini 0.778, 50 % on 7.3 %) sits inside this range —
   consistent.

## 2026-09-18 — what a `serve` rate does with run position on this Linux box (issue #38)

Why this page carries it: neither of the two measurement pages of this repository held the
chains of record before today — the chains live in `CHANGELOG.md`, `docs/architecture.md` and
the issue threads — and this is the page whose own subject the chain measures. Everything above
is a design input derived from Crow's routing logs: cold experts per token, cold bytes per
token, the serialized H2D those bytes imply. The chain below is the first time the engine's own
per-request counters on this machine are written down beside that curve, and the hypothesis it
was run against is a cold-bytes hypothesis.

**The question.** On the Windows box, 2026-09-10, two `serve` runs of the SAME request in one
chain read 30.45 and 22.42 tok/s over 255 timed decode steps — same binary, bit-identical
generated ids, identical engine counters, 26.4 % apart (`decode_out/srv-m2a.log:824`, issue
crow-nest #38). Four later chains on that box (#37, 2026-09-11) spread only 1.004 to 1.105 and
did not reproduce it. The unmeasured hypothesis of the issue: the drift scales with cold-tier
copy volume, so it is a host or PCIe effect and not a GPU clock effect. The issue's own first
step (0 EUR): one chain that runs the serve arm several times back to back, one fresh process
per run, recording enough per run to say whether the drift is monotonic in run index, a
first-run effect, or absent.

**The form.** `tools/drift-chain.sh` (new with this section) is that chain on Linux. Two arms,
alternating as adjacent pairs the way the #37 chains ran them: **S** = `serve` through
`tools/serve-linux.sh` on port 8099 answering ONE `POST /v1/chat/completions`, the t1-read
prompt as a single user message, `max_tokens` 256, `temperature` 0, `stream` false; **D** =
`decode run decode_out/srv-a5-t1read-ids.json 256`, the arm #37 called D1. Both arms generate
256 tokens and time 255 steps. No warmup run is discarded: the question IS whether run 1
differs from run 4. One fresh process per run, `pgrep` clean and `engine/.engine.lock` absent
before every start, every `serve` stopped by PID and the GPU back under 900 MiB before the next
load. The prompt is the 16,064 ids of record in both arms — serve renders the chat template
itself, and `serve tokenize --chat` on the same text reproduces
`decode_out/srv-a5-t1read-ids.json` id for id (16,064 of 16,064, 2026-09-18).

Chain 1, order S D S D S D S D, 2026-09-18 04:12–04:18 UTC, HEAD `788fb64`,
`decode_out/38/c1-sdsd/chain.log` (81 lines, table at `:53`, spreads at `:72`):

| run | arm | tok/s | predicted_ms | cold/token processed | cold selections, this request | bytes streamed | PLE rows / fills | free-for-pin GiB | sm MHz before the load | generated ids sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S | 49.636 | 5137.392 | 275.42 | 4,494,905 of 7,833,120 | 12,427,513,344,000 | 261,104 / 107,227 | 60.24 | 360 | `e7c17e064ea2…` |
| 2 | D | 39.861 | 6397.227 | ~283.44 | ~4,625,751 of 7,833,600 | ~12,789,276,364,800 | 261,120 / 107,179 | 60.23 | 2,692 | `56305eee11d6…` |
| 3 | S | 49.891 | 5111.193 | 275.42 | 4,494,905 of 7,833,120 | 12,427,513,344,000 | 261,104 / 107,227 | 60.23 | 2,917 | `e7c17e064ea2…` |
| 4 | D | 39.824 | 6403.183 | ~283.44 | ~4,625,751 of 7,833,600 | ~12,789,276,364,800 | 261,120 / 107,179 | 60.22 | 2,917 | `56305eee11d6…` |
| 5 | S | 49.845 | 5115.855 | 275.42 | 4,494,905 of 7,833,120 | 12,427,513,344,000 | 261,104 / 107,227 | 60.24 | 2,917 | `e7c17e064ea2…` |
| 6 | D | 39.813 | 6404.906 | ~283.44 | ~4,625,751 of 7,833,600 | ~12,789,276,364,800 | 261,120 / 107,179 | 60.25 | 2,400 | `56305eee11d6…` |
| 7 | S | 49.914 | 5108.743 | 275.42 | 4,494,905 of 7,833,120 | 12,427,513,344,000 | 261,104 / 107,227 | 60.25 | 2,902 | `e7c17e064ea2…` |
| 8 | D | 39.823 | 6403.280 | ~283.44 | ~4,625,751 of 7,833,600 | ~12,789,276,364,800 | 261,120 / 107,179 | 60.26 | 2,377 | `56305eee11d6…` |

Arm S's counters are one `routing` JSON line per request and are exact. Arm D's are composed
from the two per-token lines `decode run` prints (`cold experts during prefill+warm-up: 284.6
per token of 480 selections (16065 tokens)` and `cold experts per timed decode token: 210.4 of
480`, `ple rows during prefill+warm-up: 257040 requested, 104652 misses` plus `16.0 / 9.91` per
timed token), so the D totals carry the rounding of those prints and are marked `~`. `bytes
streamed` is the routing line's own definition, cold selections x 2,764,800 B per expert: it
over-counts prefill, where one chunk stages an expert once for many tokens. Every run of an arm
carries the identical figure, which is the point of the column.

Chain 2, order S S S S — four consecutive fresh serve loads, the form the issue asked for,
2026-09-18 04:19–04:22 UTC, `decode_out/38/c2-ssss/chain.log` (56 lines, table at `:36`):

| run | arm | tok/s | predicted_ms | cold/token processed | cold selections | PLE rows / fills | free-for-pin GiB | sm MHz before the load | generated ids sha256 |
|---|---|---|---|---|---|---|---|---|---|
| 1 | S | 49.856 | 5114.761 | 275.42 | 4,494,905 | 261,104 / 107,227 | 60.26 | 360 | `e7c17e064ea2…` |
| 2 | S | 49.851 | 5115.237 | 275.42 | 4,494,905 | 261,104 / 107,227 | 60.28 | 2,917 | `e7c17e064ea2…` |
| 3 | S | 49.896 | 5110.604 | 275.42 | 4,494,905 | 261,104 / 107,227 | 60.29 | 1,260 | `e7c17e064ea2…` |
| 4 | S | 49.851 | 5115.291 | 275.42 | 4,494,905 | 261,104 / 107,227 | 60.30 | 2,595 | `e7c17e064ea2…` |

### Within-arm spread, max over min (the form of the #37 comment table)

| chain | arm | runs | positions | min tok/s | max tok/s | mean tok/s | spread | run 1 against its arm mean |
|---|---|---|---|---|---|---|---|---|
| 1 | serve, `stream:false`, 255 timed steps | 4 | 1, 3, 5, 7 | 49.636 | 49.914 | 49.8215 | **1.0056** | −0.372 % (the slowest) |
| 1 | `decode run` D1 | 4 | 2, 4, 6, 8 | 39.813 | 39.861 | 39.8304 | **1.0012** | +0.077 % (the fastest) |
| 2 | serve, four consecutive fresh loads | 4 | 1, 2, 3, 4 | 49.851 | 49.896 | 49.8635 | **1.0009** | −0.015 % |

The `predicted_ms` spreads are the same figures read on the wall clock instead of the rate:
1.00561, 1.00120 and 1.00092. Neither arm is monotonic in run index in either chain — the
successive deltas of chain 1's serve arm are +0.255, −0.046, +0.069 tok/s and of chain 2's
−0.005, +0.045, −0.045.

### What the numbers say

- **The drift is ABSENT on this machine, and it is neither monotonic in run index nor a
  first-run effect.** The largest spread over 8 counted serve runs in two chains is 1.0056 =
  0.56 %, which is 47x smaller than the 26.4 % of `srv-m2a.log` and 19x smaller than the
  largest #37 spread (1.105). Four consecutive fresh serve loads (chain 2) read 49.851 to
  49.896 tok/s: the first-run effect the issue asks about does not exist here at a resolution
  of 0.09 %.
- **The one run that could be read as a first-run effect points the other way.** Chain 1's
  first serve run is its SLOWEST (−0.372 % against the arm mean), the direction M2a had
  backwards, and chain 2 does not reproduce even that: its run 1 sits −0.015 % off the mean.
  Chain 1's first `decode run` is the fastest of its arm by +0.077 %, the "first counted run
  fast" bias of the 38a analysis (−0.17 to −1.73 % there) in miniature.
- **The cold-bytes hypothesis cannot be tested here, because there is no drift left to scale
  with volume — and the converse bound is now measured.** Cold selections, bytes streamed, PLE
  rows and PLE fills are identical to the digit in every run of an arm (serve 4,494,905 cold of
  7,833,120 selections, 261,104 PLE rows and 107,227 fills in 8 of 8 serve runs across both
  chains; `decode run` 210.4 cold experts per timed decode token and 257,040 / 104,652 PLE rows
  during prefill in 4 of 4), while the
  rate moves by at most 0.56 %. With the copy volume held exactly fixed, run position buys at
  most 0.56 % on this box.
- **The cross-arm direction is consistent with a copy-bound decode and carries no weight for
  #38.** The arm with the smaller cold total is the faster arm (serve 4,494,905 cold and
  49.82 tok/s against `decode run` 4,625,751 and 39.83), but the two arms are not one variable
  apart: serve pins `prompt_chunk` 2048 and got N = 149 hot experts per layer with a 48.17 GB
  pinned cold tier, `decode run` let the policy pick 4096 and got N = 142 with 49.10 GB, and
  serve ticks the stream trickle on every decode step with the `CROW_ADAPT_WINDOW` ranking
  (8,911 swaps per request, #37) where this `decode run` arm does not. The prefill of the same
  16,064 ids separates them the same way and just as stably: serve 18.20 to 18.37 s = 874 to
  883 tok/s over its 8 runs, `decode run` 16.13 to 16.22 s = 991 to 996 tok/s over its 4. That is
  an arm difference, not a finding.
- **Against a clock or power explanation, more strongly than #37 could put it.** #37 read
  180 MHz sm at every machine block. Here the pre-load state varied by 8x in clock and 3x in
  power across the twelve starts — 360 MHz / 35.6 W / 31 °C at the first start of each chain
  against 1,260 to 2,917 MHz, 93.6 to 112.4 W and 44 to 53 °C at the other ten — and the rate
  of both arms held inside 0.6 % through all of it.
- **This chain does the same work the M2a serve arm did.** Expert selections 7,833,120 and PLE
  rows 261,104 per request are the M2a figures to the digit (issue #38 table, 2026-09-10); only
  the cold share differs, 4,494,905 against 4,655,928 (−3.5 %), because the hot set is not the
  same one (N = 149 with the #37 trickle tick). The generated ids differ from M2a's
  `1224f4bb6524…` because the Linux NVRTC and driver drift the logits over a long generation,
  which is `docs/architecture.md` section 8.7 and not this issue.
- **What this says about the Windows M2a outlier: nothing directly.** Different OS, driver
  (610.57.04 against 616.56), host platform and a much-changed engine. It refutes "this engine
  has an intrinsic run-position drift in its serve path" for this machine at `788fb64`; it
  cannot retire the M2a datum, which needs the same form rerun on the Windows box (the retiring
  series of the 38a comment).

### The ids, in full

- serve arm, 256 generated ids, 8 of 8 runs over both chains:
  `e7c17e064ea2a1f82e447747a08e25562597687fd77f63a8db697fe1fb67edda`
  (sha256 of the id list as `,`-joined decimals, `decode_out/38/*/r*-ids.json`).
- `decode run` arm, its own 256-id trace, 4 of 4:
  `56305eee11d6b9df1eae8fc266739453295cd98397bb63e6e66c1cdf2787ded3`. The two arms are offset by
  one token by construction — `decode run` discards a warm-up step whose token it keeps
  generating from — and the chain checks the overlap instead: `serve[1:] == decode[:255]` is TRUE
  (`decode_out/38/c1-sdsd/chain.log:80`). So the two arms walk the same greedy trajectory and the
  spreads above are spreads of timing, not of work.

### Machine blocks, twelve starts

Every one of the twelve engine starts recorded UTC, free-for-pinning as the engine derives it,
`MemAvailable`, `Cached`, and the GPU's memory, sm clock, power and temperature BEFORE the load
(`decode_out/38/c1-sdsd/machine.jsonl`, `decode_out/38/c2-ssss/machine.jsonl`). Free for
pinning read 60.22 to 60.30 GiB at the pre-start block and 60.12 to 60.22 GiB on the engine's
own `[budget]` line inside the load; `MemAvailable` read 9.42 to 11.58 GiB over the same
twelve; the `Cached` page cache 3.48 to 5.66 GiB; the GPU 652 to 658 MiB, i.e. the desktop
alone. The derived pinned budget was 46.00 GiB from the configured cap at all twelve starts and
the RAM guard never fired. **A `MemAvailable`-based "> 50.5 GiB free" gate would have refused
every one of the twelve** — which is the v0.3.0 host-memory model (`docs/architecture.md` 8.8
point 2) measured again, twelve times, as a by-product of this chain.

Artefacts, all gitignored, 480 KiB in total and no payload among them: per chain a `chain.log`
(the run-by-run transcript, the table and the spreads), `machine.jsonl` (one machine block per
start), `rows.jsonl` (one parsed row per run), `request.json` (the body every serve run posted),
and per run `r<i>-serve.log` or `r<i>-decode.log` (the engine's own stderr mirror, 181 and 163
lines, carrying the boot JSON line, the `[budget]` lines and the `routing` JSON line),
`r<i>-ids.json`, plus `r<i>-resp.json` for a serve run and `r<i>-run.json` for a `decode run`.
The stdout of the two chain processes is `decode_out/38/c1-sdsd.out` and
`decode_out/38/c2-ssss.out`.

### Consequence rules — recommendation, robin decides

1. **"Chains that load the engine wait for more than 50.5 GiB free": keep it replaced on
   Linux.** The derived budget of `#15`/v0.3.0 is the gate here and it held twelve times; the
   Windows rule stays as written in `engine/README.md`.
2. **"A serve tok/s is quoted only next to an adjacent `decode run`": KEEP, on Linux too.** Not
   because of the drift — that is gone — but because of what this chain shows about the arms:
   `serve` and `decode run` are not the same operating point (N 149 against 142, chunk 2048
   against 4096, the trickle tick on against off), so a serve rate quoted alone invites a
   comparison the configuration does not support. `tools/drift-chain.sh` makes the adjacent run
   one character of the order string.
3. **"No serve decode number enters `docs/architecture.md`": RELAX on Linux, conditionally.**
   Proposal: a serve rate may enter the docs when it comes from a drift-chain form — at least
   3 counted serve runs in one chain, one fresh process per run, the generated-ids sha256
   identical across them, the adjacent `decode run` arm in the same chain — and is quoted with
   its arm mean and its max-over-min spread (38a rules 4, 7 and 12). The number this chain would
   put forward is *serve 49.82 tok/s mean over 4 runs, spread 1.0056, next to `decode run`
   39.83 tok/s mean, spread 1.0012, t1-read 16,064 ids, 255 timed steps, 2026-09-18*. The bar
   stays in force for Windows until the M2a form is rerun there. Nothing was added to
   `docs/architecture.md` in the commit that carries this section: the rule stands until it is
   lifted.
4. **New, if the drift is ever quoted again: it is answered per machine.** This section bounds
   one box, one GPU, one day, one prompt shape and one HEAD. It bounds no other.
