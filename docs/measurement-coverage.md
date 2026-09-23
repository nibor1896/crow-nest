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

### Consequence rules — the recommendation of the morning (robin decided them the same day; the decisions are the section below)

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


## 2026-09-18, later the same day — robin's decision on the three rules, and the chain rerun at the new default (issue #38)

Robin read the recommendation above and took all three, with one of them changed from the shape it
had at 04:00. The rule texts below are the ones of record; they are written the same way in
`docs/architecture.md` 0.5, `engine/README.md` "Machine rules" and `README.md`.

1. **The host-RAM gate before an engine start is per OS.** *Windows: the 50.5 GiB gate* — a chain
   waits for more than 50.5 GiB free host RAM before it loads the engine (rule since 2026-09-10).
   *Linux: the derived budget, read off the `[budget]` boot line* — the 50.5 GiB gate does not
   apply here and stays replaced by the pinned budget the engine derives at boot
   (`docs/architecture.md` 8.8 point 2, issue #15). Nothing in the code changes: the twenty
   engine starts of the three chains of this day all passed on the derived budget, and a
   `MemAvailable` reading of the Windows gate would have refused every one of them.
2. **A `serve` tok/s is quoted only next to an adjacent `decode run` measured in the same
   chain** — KEPT, on Linux too, with a NEW reason. It is no longer the drift, which is absent
   on this box (0.56 % over the eight counted serve runs of the morning, 0.38 % over the four
   below). It is the operating point: the two arms are not the same one. `serve` pins
   `prompt_chunk` 2048, gets N = 149 hot experts per layer and ticks the stream trickle on every
   decode step (8,911 swaps per request); `decode run` lets the policy pick 4096, gets N = 142
   and does not tick. A lone serve rate therefore invites a comparison the configuration does not
   support, and the adjacent `decode run` is the anchor. `tools/drift-chain.sh` makes that
   adjacent run one character of the order string.
3. **"No serve decode number enters `docs/architecture.md`" — RELAXED on Linux, to the
   drift-chain form.** A serve rate may enter that document when it carries all of: at least
   3 counted serve runs inside one chain, one fresh process per run, the generated-ids sha256
   identical across those runs, an adjacent `decode run` arm in the SAME chain, and the figure
   quoted as its arm mean with its max-over-min spread beside that decode arm's mean and spread,
   naming the chain's log (38a rules 4, 7 and 12). A serve number without that form is refused.
   On Windows the bar stays as written until the M2a form is rerun there.

**The number that satisfies rule 3, at the CURRENT default.** The 49.82 / 1.0056 of the morning
was measured at the pre-`#61g` default; `CROW_ATTN_LUT` became the default at HEAD `6c87054`
later the same day (issue #61, 61g), which moves both arms. So the chain was rerun in the same
`S D S D S D S D` form at that HEAD, `CROW_STAGE_PAR`, `CROW_GDN_SPLIT_Z` and `CROW_ATTN_LUT` all
unset — the new default — 2026-09-18 09:58–10:05 UTC,
`decode_out/38/c3-sdsd-61g/chain.log` (81 lines, table at `:53`, spreads at `:72`, the cross-arm
check at `:80`), stdout `decode_out/38/c3-sdsd-61g.out`.

| run | arm | tok/s | predicted_ms | mean ms per token | cold/token processed | cold selections, this request | PLE rows / fills | free-for-pin GiB | sm MHz before the load | generated ids sha256 |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | S | 53.282 | 4785.844 | 18.768 | 275.42 | 4,494,905 of 7,833,120 | 261,104 / 107,227 | 60.22 | 360 | `e7c17e064ea2…` |
| 2 | D | 42.583 | 5988.369 | 23.4838 | ~283.44 | ~4,625,751 of 7,833,600 | 261,120 / 107,179 | 60.20 | 2,917 | `56305eee11d6…` |
| 3 | S | 53.412 | 4774.224 | 18.722 | 275.42 | 4,494,905 of 7,833,120 | 261,104 / 107,227 | 60.21 | 2,872 | `e7c17e064ea2…` |
| 4 | D | 42.542 | 5994.145 | 23.5065 | ~283.44 | ~4,625,751 of 7,833,600 | 261,120 / 107,179 | 60.21 | 2,062 | `56305eee11d6…` |
| 5 | S | 53.209 | 4792.393 | 18.794 | 275.42 | 4,494,905 of 7,833,120 | 261,104 / 107,227 | 60.21 | 2,902 | `e7c17e064ea2…` |
| 6 | D | 42.581 | 5988.591 | 23.4847 | ~283.44 | ~4,625,751 of 7,833,600 | 261,120 / 107,179 | 60.20 | 2,902 | `56305eee11d6…` |
| 7 | S | 53.375 | 4777.537 | 18.735 | 275.42 | 4,494,905 of 7,833,120 | 261,104 / 107,227 | 60.20 | 2,685 | `e7c17e064ea2…` |
| 8 | D | 42.544 | 5993.756 | 23.5049 | ~283.44 | ~4,625,751 of 7,833,600 | 261,120 / 107,179 | 60.21 | 2,505 | `56305eee11d6…` |

### Within-arm spread, max over min (chain 3, the same form as the two above)

| chain | arm | runs | positions | min tok/s | max tok/s | mean tok/s | spread | run 1 against its arm mean |
|---|---|---|---|---|---|---|---|---|
| 3 | serve, `stream:false`, 255 timed steps | 4 | 1, 3, 5, 7 | 53.209 | 53.412 | **53.3195** | **1.00382** | −0.070 % |
| 3 | `decode run` D1 | 4 | 2, 4, 6, 8 | 42.542 | 42.583 | **42.5623** | **1.00096** | +0.049 % |

The `predicted_ms` spreads are the same figures on the wall clock: serve 1.00381 (mean 4,782.499 ms)
and `decode run` 1.00096 (mean 5,991.215 ms). The `decode run` arm's mean ms per token is 23.4950
with p50 22.778 to 22.884, which reproduces the `#61g` confirmation runs of the same day (N mean
23.5193 ms, `docs/architecture.md` 4.6.1) to 0.10 %. Neither arm is monotonic in run index: the
serve deltas are +0.130, −0.203, +0.166 tok/s and the `decode run` deltas −0.041, +0.039, −0.037.

### What chain 3 adds

- **The drift is still absent at the new default, and now at a smaller spread than the morning
  chain's.** Serve 1.00382 over 4 counted runs against 1.0056 at 04:12, `decode run` 1.00096
  against 1.0012. That is 69x under the 26.4 % of `decode_out/srv-m2a.log:824` (2026-09-10,
  Windows) and 27x under the widest `#37` chain (1.105, 2026-09-11).
- **The work is the same work, digit for digit, as the morning chain's.** Serve: 4,494,905 cold of
  7,833,120 expert selections, 261,104 PLE rows, 107,227 fills, 8,911 trickle swaps in 4 of 4
  runs, N = 149, `prompt_chunk` 2048, 48.17 GB pinned. `decode run`: 210.4 cold experts per timed
  decode token, N = 142, chunk 4096, 49.10 GB pinned, context 16,320. The ids are the shas of
  record in 4 of 4 runs per arm and `serve[1:] == decode[:255]` is TRUE, so `#61g` moved the rate
  and not one generated id through either arm.
- **What the flip is worth at the operating point, on both arms of one chain.** Against the
  04:12 chain at the pre-flip default: serve 53.3195 against 49.8215 tok/s = **+7.02 %**
  (4,782.499 against 5,118.296 ms predicted, −335.8 ms = −6.56 %) and `decode run` 42.5623
  against 39.8304 = **+6.86 %** (5,991.215 against 6,402.149 ms, −410.9 ms = −6.42 %). Those two
  chains are 6 hours apart on one box at one machine state, which rule 11 of the 38a discipline
  calls a cross-chain comparison; the lever's own adjacent-pair reading stays the 61f one
  (−6.28 %, `docs/architecture.md` 4.6.1).
- **Prefill is untouched by the flip and just as stable.** serve 18.411 to 18.423 s = 871.9 to
  872.5 tok/s over its 4 runs (874 to 883 at 04:12), `decode run` 16.25 to 16.27 s = 987.1 to
  988.6 tok/s over its 4 (16.13 to 16.22 s = 991 to 996 at 04:12).
- **The machine blocks say what the morning's said.** Eight starts, free for pinning 60.20 to
  60.22 GiB at the pre-start block and 60.10 to 60.17 GiB on the engine's own `[budget]` line,
  the derived pinned budget 46.00 GiB at all eight and the guard never fired; `MemAvailable`
  9.39 to 9.64 GiB — the figure the Windows gate would have read, and refused all eight on;
  `Cached` 3.13 to 4.95 GiB; the GPU 630 MiB (the desktop alone), sm clock 360 MHz at the first
  start and 2,062 to 2,917 MHz at the other seven, 35.9 to 112.4 W, 33 to 54 °C, and both rates
  held inside 0.4 % through all of it.

### The figure of record entered into `docs/architecture.md`

Under rule 3 the serve number that enters section 4.6.1, next to its adjacent arm, is:

> **`serve` 53.32 tok/s, mean of 4 counted runs, within-arm spread 1.0038**, next to the adjacent
> **`decode run` arm's 42.56 tok/s mean, spread 1.0010**, one fresh process per run, generated ids
> `e7c17e064ea2` 4 of 4 and `56305eee11d6` 4 of 4, t1-read 16,064 ids, 256 generated tokens,
> 255 timed steps, HEAD `6c87054`, RTX 5090 / Arch Linux, 2026-09-18,
> `decode_out/38/c3-sdsd-61g/chain.log`.

It replaces the single serve reading of the `#61g` flip (53.31 tok/s, one run) as the serve figure
of record; that one-run pair stays in 4.6.1 as the lever's own `CROW_ATTN_LUT` A/B, which is what
it measures. Artefacts as for the two chains above, 296 KiB, all gitignored.

## 2026-09-23 — instruments for token corruption (issue #91)

The sections above measure expert coverage and serve-rate drift. The corruption work of #91
added three facts about instruments to this page: one instrument was retired, and two replaced
it. Each is listed with what it covers and what it does not.

### Retired: the K=2 pass/fail replay ladder (2026-09-22)

- **What it was.** The replay preset diorama-0922, run as 8 seeded rounds plus 1 greedy round
  per point, counting corrupt tool calls per arm. The rows are in
  `decode_out/corruption-arms-replay-diorama-0922/TABLE.md`.
- **Why it was retired** (#91 comment 2026-09-22T20:38Z). The placebo overlay changes only the
  kernel path: the values are byte-identical, run through BF16 instead of NVFP4. At the first
  command token (#15) that placebo flipped a 0.039-nat tie, after which the two paths generate
  different text. So "K=2 greedy clean/corrupt" measured that fork and nothing about the error
  mechanism. The ladder is retired as an instrument at n = 8, and its arm "fixes" prove nothing.
- **What replaced it.** Continuous, teacher-forced logprob margins. The first was the single
  site #65 (`tools/teacher-forced-91.sh`, #91 comment 2026-09-23T03:38Z). It was then extended
  to the multi-site probe below, because one site cannot rank fixes.

### Multi-site teacher-forced probe (`tools/multisite-corruption-probe.py`, runner `tools/multisite-0923.sh`)

- **What it covers.** 23 corrupt tool-call sites from robin's 2026-09-23 diorama session, in
  the site set `tools/corpora/91-multisite-0923.json`. 9 sites are "fresh" (no earlier corrupt
  spelling in the context) and 14 are "contaminated". Prompts run 18k to 103k tokens, so the
  probe reaches the sparse-attention long-context regime that the oracle-KLD rows (298 / 607
  tokens) cannot reach.
- **How it measures.** Per site it teacher-forces the model's own tokens up to and including
  the corrupt token, through `serve` (`crow_force_ids`, `top_logprobs` 20, greedy). It reads
  lp(correct) and lp(corrupt) there. Per arm it reports the corrupt-win count, the mean margin
  lp(correct) - lp(corrupt), and correct top-1. Optional extras: `freerun` (sampled generation,
  counting digit errors) and `speed` (256 short-prompt decode tokens, median of 3). The runner
  boots one serve per arm on port 8111, behind `ramcheck --need`, with logs under `$OUT`.
- **Its measurement of record** is the 2026-09-23 arm table in `docs/numerics-diff.md` §7 and
  `docs/dense-overlay.md` §8:

  | arm | corrupt wins |
  |---|---|
  | bare, before the fix | 15/23 |
  | dense overlay | 12/23 |
  | activation pre-scale | 9/23 |
  | PLE fix, with overlay | 4/23 |
  | llama.cpp UD-Q2_K_XL | 4/23 |
- **What it does not cover:**
  - The forced ids are the stored text re-tokenized with the model's tokenizer, not the live
    ids, which were not logged. That is an approximation.
  - All sites come from one session, and it scores only sites that were already corrupt. It
    cannot find new corruption classes, and it says nothing about general answer quality or
    distance to the f32 model.
  - The free-run counts are n = 1 per point and not significant (2026-09-23).
  - It needs the untracked, sha-pinned session snapshots under
    `decode_out/sessions/2026-09-23-diorama/`.

### Per-layer diff against llama.cpp (`tools/layerdiff/`, README there)

- **What it covers.** At one position of a corrupt site, it compares crow-nest and llama.cpp
  UD-Q2_K_XL (`ldump.cpp`) layer by layer: residual stream, sub-block outputs, the layer-1 PLE
  contribution, per-stream cos, experts in common, and the final logit margin (`compare.py`).
  `cnq_ple.py` / `flat.py` decode PLE rows straight from the CNQ container and the GGUF. This
  instrument located the PLE row-offset bug (`85a48e7`). At mat44-a149 and N33-a131 on
  2026-09-23:
  - layer 0 matched (cos ≥ 0.9985)
  - layer 1 split (residual cos 0.23 / 0.21)
  - with the flat read, the layer-1 residual cos was 0.997 to 0.999
- **What it does not cover:**
  - The reference is llama.cpp's own 2.4 bpw quantization, not BF16. Agreement means "the same
    function up to two different weight quantizations", not "both are right". The gap left at
    layer 47 (cos 0.91 / 0.92) is not attributed.
  - The crow-nest dump side (`decode sitedump`) is a local instrumentation patch that is not on
    this branch. It lives in the measurement worktree
    (`decode_out/meas-0923/layerdiff/cn-dump-instrumentation+ple-flat.patch`).
  - It was run at 2 sites, one position each, with `CROW_GRAPH=0`.
