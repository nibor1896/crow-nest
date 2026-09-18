# crow-nest — the 300-turn goal-mode session at 170k context, measured (issue #68, 2026-09-18)

What this page is: the measurement record of issue `#68` — robin's 40-minute Crow goal-mode
session of 2026-09-17 (574 messages, 273 assistant turns, the last prompt 178,779 tokens of
200,000) replayed through `serve` at `main` after the `#67` reasoning filter landed, plus the
first long-context quality reading this engine has ever had. It answers one question: of the three
candidate causes the ticket named — the engine, the Crow client, the model/quant — which one
produces the two late stages of the degeneration, the verbatim echo of Crow's goal nudge and the
single-token answer.

Machine: the second environment block of `docs/system-landscape.md` (RTX 5090, driver 610.57.04,
Arch Linux 7.2.3-arch1-3, CUDA 13.3.1). `serve` started with `tools/serve-linux.sh --port 8099`
at commit `667b68b` plus the `#68` working tree, one engine process at a time. Artefacts of the
runs: `decode_out/68/` (gitignored; `serve.log` plus one JSON record per run).

## 0. The answer in four lines

1. **The engine's sampler is not the cause.** `presence_penalty` is applied over the tokens
   generated in THIS request, once per distinct token, and the set is cleared per request — the
   HF/vLLM semantics the model card's `presence_penalty` is written in. No bug, no fix; the scope
   is now pinned by tests and documented in `docs/architecture.md` 7.11.17.
2. **`#67` was not the cause either.** With the reasoning filter active and every stored
   `</think>` stripped out of the history, the echo and the single-token answer **reproduce** at
   168,928 tokens of context on the first and second turn.
3. **Length is the enabling condition, the history is the cause.** The same session answers
   normally at 120,924 ids under every row and degenerates at 163,401 to 168,928 under greedy and
   under Crow's row alike (the live first single-token answer came at prompt 172,599). But a CLEAN
   agentic history of **178,553** ids — 167 files read through 167 tool calls — answers 4 of 5
   probes of the new quality gate with **no** degeneration. So the engine's decode at the live
   context size is not broken; robin's session is what is broken at that size.
4. **Crow did not send the mixed sampling row it was blamed for.** Crow has no
   `presence_penalty` field at all; the 1.5 in the live `[chat]` line is `serve`'s own default.
   What Crow owns is the shape of the session: 105 identical nudges, no loop detection, no cap on
   turns per step.
5. **The stage-3 shape is 48 one-token answers, not one runaway generation** — each request
   correct on its own, `finish stop` after a single `3`, the client concatenating them. Nothing
   inside one request can see that, which is where loop detection has to live.

## 1. The penalty scope — what the engine applies over what (part A1)

| question | answer | code |
|---|---|---|
| which token set carries `presence_penalty`? | the tokens THIS request generated, nothing else | `gen.rs:3641` `enable_dev_sampler` uploads a zeroed `mask[V]`, `kernels.rs sample_k` sets `mask[tok] = 1` for the token it just drew |
| is the prompt in the set? | no. The mask is zeroed after `prefill` and before the first `decode_step`; the first token is drawn from the prefill's last logits row with an empty mask (`arm_sampler` = `enable_dev_sampler` + `sample_last`) | `gen.rs:443`, `bin/serve.rs:2558` |
| a last-n window (llama.cpp's `repeat_last_n`)? | none. There is no window and no decay; a token stays penalized for the rest of the answer | `kernels.rs sample_k` |
| presence or frequency? | presence. The set is a `u8` mask, so it cannot count; a token drawn ten times is penalized once, by `presence_penalty` | `kernels.rs:4148` `mask[tok] = 1`, host twin `sample.rs:66` |
| across the 300 turns of a prefix-cached session? | reset per request. `arm_sampler` runs for EVERY sampled request, after `prefill`, and re-uploads the zeroed mask plus `Rng::new(seed)` — whatever the prefix cache reused, turn k starts with an empty set | `bin/serve.rs:2550-2562` |
| a greedy request? | no penalty at all: the sampler is parked out of the engine and the head ends in `argmax_k`, which is why the parity ids cannot move | `bin/serve.rs:2567`, `gen.rs:448` |
| the reference this is measured against | the model card's two rows as this project recorded them (`probes/p5_STATUS.md:539-545`, robin's check): thinking temp 1.0 / top_p 0.95 / top_k 20; non-thinking temp 0.7 / top_p 0.80 / top_k 20 + `presence_penalty` 1.5 "to reduce endless repetition". A card `presence_penalty` is the HF/vLLM knob, i.e. the penalty over the tokens generated in this response — which is exactly what the engine does. llama.cpp's `repeat_penalty` is a DIFFERENT knob (multiplicative, windowed over prompt plus generation) and the engine does not implement it, by design | `probes/p5_STATUS.md`, `sample.rs:1-14` |

**Verdict: not a bug.** The engine's scope matches the documented contract. Two consequences worth
writing down anyway:

- `presence_penalty 1.5` cannot stop a loop that spans TURNS. The live stage-3 answers are one
  token long; the set is empty when each of them is drawn, so no penalty value would have changed
  them. A repetition brake across turns is not in the card's contract and is not in the engine —
  it belongs to the client (loop detection) or to a new, deliberately chosen knob.
- Tests pin the two halves that could silently drift: `sample.rs`
  `the_presence_penalty_is_applied_once_per_distinct_token` (presence, not frequency — the two
  readings pick different tokens, so the test cannot pass under a count-scaled penalty) and
  `the_penalty_set_is_this_answers_tokens_only` (a fresh `Sampler` penalizes nothing, and
  `Sampler::new` is what the server builds per request).

## 2. The sampling row the live session really ran (part A1, second half)

The `[chat]` line of the live log reads
`temperature 1 top_p 0.95 top_k 20 presence_penalty 1.5 seed 0`, and `#68` quoted it as "sampling
as sent by Crow". Measured against Crow's own source (`~/.local/share/crow/cli/crow_core.py`,
build of 2026-09-16):

| field | who sent it | evidence |
|---|---|---|
| `temperature` 1.0 | Crow | `crow_core.py:419` `TEMPERATURE = 1.0`, `sampling_for()` at `:496` |
| `top_p` 0.95 | Crow | `crow_core.py:425`, and `manifests/operating-point.json` chose 1.0 / 0.95 for DeepSeek-V4-Flash-0731 |
| `min_p` 0.01 | Crow, accepted and ignored by the device sampler | `crow_core.py:430`, `[chat] min_p 0.01 accepted and ignored` |
| `top_k` 20 | **the engine** (`DEFAULT_TOP_K`); the manifest entry that would have sent 20 is keyed on `flash-next-q2-k-xl`, not on this container's model name | `bin/serve.rs:511`, `manifests/operating-point.json` |
| `presence_penalty` 1.5 | **the engine** (`DEFAULT_PRESENCE`). The string `presence_penalty` does not occur anywhere in Crow's source; its wire list is `SAMPLING_FIELDS = ("temperature", "top_p", "min_p", "top_k")` | `bin/serve.rs:513`, `crow_core.py:716` |
| `seed` 0 | **the engine** (`DEFAULT_SEED`, fixed so a warm process draws what a cold one draws) | `bin/serve.rs:515` |

So the live operating point mixed the card's two rows, but the engine supplied the second half of
the mix. The fix on this side is honesty in the log, not a changed default: since `#68` the line
names the source of every value —

```text
[chat] sampling on the device: temperature 1 (request) top_p 0.95 (request) top_k 20 (data sheet) presence_penalty 1.5 (data sheet) seed 0 (data sheet)
[chat] presence penalty set: cleared for this request, generated tokens only (#68)
```

— pinned by `bin/serve.rs::the_sampling_line_says_which_values_the_request_carried`.

## 3. The replay (part A2)

Tool: `tools/replay-session.py` (new with `#68`). It cuts robin's stored conversation after a user
turn at a chosen context size, sends it whole the way Crow does, feeds the model's own tool calls
back with the live session's own tool results, re-sends the goal nudge as the next user turn, and
records per round: prompt tokens (cached / prefilled), the generated text, an echo verdict
(32-character shingle overlap with the nudge plus the live marker string), a degeneration verdict
(a repeated unit inside one answer, or an answer of at most two tokens), and both rates out of
`timings`. `tools` is Crow's own 25-function `TOOLS` block, exec-extracted from `crow_core.py`
(14,541 B of JSON).

Images: the 17 `image_url` parts of the stored history are dropped by default, which is why the
whole 574-message history renders as **168,928** tokens here against **178,779** live — the
difference is the ~11k visual tokens of the 17 images (874 to 1,000 each, `[vit-cache]` lines of
the live log). Every context figure below is the server's own `prompt_tokens`.

### 3.1 The runs

| run | cut | context (prompt tok) | sampling row | rounds | echo of the nudge | single-token answer | decode tok/s | prefill |
|---|---|---|---|---|---|---|---|---|
| A | msg 305 | 120,924 | greedy | 3 | 0 of 3 | 0 of 3 | 39.3 / 46.6 / 47.5 | 859 tok/s cold (144 s) |
| B | msg 305 | 120,924 | card 0.7 / 0.8 / 20 / 1.5 | 3 | 0 of 3 | 0 of 3 | 43.5 / 48.6 / 46.7 | 884 tok/s cold |
| C | msg 305 | 120,924 | Crow 1.0 / 0.95 / 20 / 1.5 seed 0 | 3 | 0 of 3 | 0 of 3 | 40.8 / 44.9 / 48.5 | 884 tok/s cold |
| D | msg 573, the live tail | **168,928** | Crow 1.0 / 0.95 / 20 / 1.5 seed 0 | 3 | **1 of 3** (round 0, verbatim) | **2 of 3** (rounds 1, 2: `3`) | 40.3 / 44.9 / 42.3 | 861 tok/s, 120,924 cached |
| E | msg 573, the live tail | **168,928** | **greedy** | 3 | 0 of 3 | **3 of 3** (`3`, 0.54 s per turn) | 44.5 / 42.7 / 43.0 | 888 tok/s cold (190 s) |

The onset sweep, greedy, two turns per point, ascending so each prefill reuses the one below it:

| cut | context (prompt tok) | answer of round 0 | verdict |
|---|---|---|---|
| msg 323 | 124,090 | 169 tok, a `render_page` call, on task | healthy |
| msg 342 | 146,483 | 98 tok, a `read_image` call, on task | healthy |
| msg 365 | 153,755 | 270 tok, a `run_command` call, on task | healthy |
| msg 483 | **163,401** | **1 tok, `3`, `finish stop`** | degenerate |
| msg 573 | 168,928 | 1 tok, `3`, `finish stop` (run E) | degenerate |

**The greedy flip on this history is between 153,755 and 163,401 prompt tokens.** The live session
(Crow's row, 17 images in the context) held out to prompt 172,599 before its first single-token
answer, which is the same band once the ~11k visual tokens are accounted for.

### 3.2 What the runs say

- At ~121k the session is HEALTHY under all three rows: three sensible turns each, one tool call
  per turn, 39 to 48 tok/s, no echo, no short answer. The prompt is the same 120,924 ids in A, B
  and C, so the only difference between them is the sampling row.
- At 168,928 under Crow's row the live shape is back on the first two turns: round 0 answers
  `3 of 5 steps done. Let me finish Step 4 with one deterministic pass …` — the nudge's own opening
  words, the live marker string — and rounds 1 and 2 answer the single token `3` with
  `finish stop`, in 0.70 s and 0.54 s.
- **Under greedy at 168,928 it is worse: all three rounds are the single token `3`.** Greedy has no
  sampler armed at all (`argmax_k`, no penalty, no draw), so the degeneration is not a sampling
  artefact and no penalty scope could have prevented it. The sampled rows escape the attractor for
  one turn; greedy never does.
- **The `#67` tags are not needed for any of this.** The replay runs against the fixed build: every
  stored `</think>` of the artefact is stripped before the render, and the count is the ticket's own
  number — **67** `[chat] normalised` lines on every full-history request (36 on every ~121k one),
  against the 67 of 273 assistant turns that carry the tag — and nothing the engine streams can
  carry a tag — `think tags stripped 0` on 32 of the 33 requests of this session, and on the 33rd
  (run B round 0, the card row, a 404-token answer) the MODEL emitted one itself mid-answer and the
  filter took it out: `[chat] reasoning filter: 1 <think>/</think> tag(s) stripped from the content
  (#67); the generated ids are untouched`, with `reasoning chunks 0`. That is `#67` working on a
  fresh generation rather than on stored text, measured here by accident. So
  the doubled closing tag was the first stage of the live session, not the cause of the other two —
  which is exactly what step 1 of the ticket's order was there to establish.
- Prefill at this size is 834 to 907 tok/s cold on this machine (2026-09-18), i.e. 2.4 minutes for
  120,924 ids and 3.2 minutes for 168,928; decode holds 37 to 48 tok/s at 124k to 169k of context.
- One economy finding for anyone planning replays: **an identical re-send pays a full cold
  prefill.** The snapshot taken after a request's prompt sits at `S_pos` = that prompt's length,
  and the reuse rule needs `S_pos < request length` (`cache.rs`, spec 7.4), so a second request
  with the SAME ids finds `P = 0` (`[cache] COLD L 120924 (held 122091), P 0, snapshots [121880,
  121492, 120924]`). A request that EXTENDS the history is warm; a repeat of the same one is not.

### 3.3 What the live log says about stage 3, and where the report needs correcting

The report describes the end state as "the digit 3 written non-stop". The artefact is more
specific, and the difference matters for the fix:

| reading | value | evidence |
|---|---|---|
| answers in the session | 293 `[chat] ids` lines | `serve.log` |
| answers that are the single token `3` (id 18) with `finish stop` | **48** | `serve.log`, first at prompt 172,599 tokens (`:6139`), last at 177,233 |
| answers of at most 3 ids | 59 | same |
| answers after the first single-token one | 56, of which 48 are that one token | same |
| assistant turns that repeat the nudge text | 9 of 273 (`session.json` messages 388 to 456) | `crow-session/session.json` |

So the loop was never one runaway generation: the model answered `3`, emitted EOS, Crow nudged
again, and the client's transcript concatenated 48 one-token answers. Nothing in the engine can
see that pattern — each request is correct on its own, `finish_reason` `stop`, one token, 0.5 s —
which is exactly why loop detection belongs to the client.

## 4. The long-context quality gate (part A3)

Tool: `tools/longctx-gate.py` (new with `#68`). One agentic session shape — a system turn, then
one `read_file` tool call plus its result per synthetic source file of a generated `libghost`
crate, then five probe turns on the same session, greedy. The material is generated from the file
index alone, so the gate is byte-stable and cannot drift with the tree. Scored the way
`docs/ten-tasks.md` scores: a probe is a Pass when every required fact of its recorded expectation
is in the answer AND the answer does not degenerate (`docs/ten-task-expected.md` §1: degeneration
is a Fail on its own).

The five probes: the constant planted at ~5 % depth, at ~50 %, at ~95 % (value plus the file that
declares it), the three-value arithmetic across those depths (`5137 + 9281 − 4409 = 10009`), and
the one function whose body contradicts its own doc comment (`checksum_rows`, `-=` where the doc
says sum). Recorded expectation, so this can become a standing gate: **>= 4 of 5 Pass and 0
degenerate**.

| run | target | context (prompt tok) | files of material | Pass | degenerate | decode tok/s |
|---|---|---|---|---|---|---|
| 100k | 100,000 | 104,433 | 98 | **5 of 5** | 0 | 35.7 to 42.2 |
| 170k | 170,000 | **178,553** | 167 | **4 of 5** | 0 | 37.5 to 40.9 |

Per probe at 104,433 tokens (greedy, 2026-09-18), the answers as recorded in
`decode_out/68/longctx-100k.json`:

| probe | depth | answer | verdict |
|---|---|---|---|
| 1 | ~5 % (`f004.rs`) | `RING_FLOOR is 5137, declared in libghost/src/f004.rs.` | Pass |
| 2 | ~50 % (`f049.rs`) | `TILE_SPAN is 9281, declared in libghost/src/f049.rs.` | Pass |
| 3 | ~95 % (`f093.rs`) | `PURGE_MARK is 4409, declared in libghost/src/f093.rs.` | Pass |
| 4 | all three | `5137 + 9281 − 4409 = 10009.` | Pass |
| 5 | ~70 % (`f068.rs`) | `The function is checksum_rows in libghost/src/f068.rs. Its doc says it returns the mean of the sum of every row, but the loop uses acc -= *r (subtraction) instead of acc += *r (addition).` | Pass |

At 178,553 tokens — the live session's own context size, 226 ids under it — the same five probes
give **4 of 5 Pass, 0 degenerate**, which meets the recorded expectation. The one Fail is worth
quoting, because it is what a long-context quality gate is for:

| probe | answer at 178,553 tokens | verdict |
|---|---|---|
| 1 | `` `RING_FLOOR` is `5137`, declared in `libghost/src/f008.rs`. `` | Pass |
| 2 | `` `TILE_SPAN` is `9281`, declared in `libghost/src/f083.rs`. `` | Pass |
| 3 | `` `PURGE_MARK` is `4409`, declared in `libghost/src/f158.rs`. `` | Pass |
| 4 | `5137 + 9281 − 4409 = 10009.` | Pass |
| 5 | "The function is `checksum_rows` in `libghost/src/f166.rs`. Its doc comment says it returns the mean of the rows … but the loop uses `acc -= *r` …" | **Fail**: the function, the operator and the reasoning are right, the FILE is `f116.rs` and the answer says `f166.rs` — two digits transposed |

**This is the finding that reframes the whole ticket.** A CLEAN agentic history of 178,553 tokens —
167 files read through 167 tool calls, 669 messages — answers four of five probes perfectly and
degenerates on none. The engine's decode at the live context size is not broken. What breaks is
robin's session at 163k to 169k: 300 turns of churn, 105 byte-identical nudges, contradictory
half-finished tool output. Length is the enabling condition; the content of those tokens is the
cause.


**What this gate does and does not say.** It says the engine's decode answers depth-spread,
multi-step and debug questions correctly over an agentic history of that size, greedy, with no
degeneration — the first such reading this engine has. The ten-task gate's own longest prompt is
t1-read at 60,290 characters and t1b-read-lang at 53,889 (`decode_out/ten-tasks.json`, read
2026-09-18), i.e. about 20k ids, so everything above that was unmeasured until this page. It does NOT say anything about logits: there
is no parity form above 1,024 rows and no oracle to compare one against at 100k. And it is one
session shape, not ten tasks: the material is synthetic, so a Pass here is not a Pass on robin's
workload. What is missing for a full ten-task-style gate at long context is the ten task classes
themselves rewritten for a 100k context with recorded expectations — a series, not one command.

## 5. Cause separation (what is the engine's, what is Crow's, what is the model's)

| stage | cause | evidence | owner |
|---|---|---|---|
| 1. `</think>` on 67 of 273 turns | the engine: no reasoning parser, the tag streamed as content, re-fed by the client, rendered verbatim into a doubled closing tag | `#67`, fixed in `667b68b`; `docs/architecture.md` 7.11.16 | crow-nest, DONE |
| 2. the nudge echoed verbatim | the model on THIS history at length: reproduced at 168,928 ids with the tags gone under Crow's row (run D round 0), absent at 120,924 under the same row, and absent at 178,553 ids of clean agentic history | this page §3, §4 | model + Crow's session shape (105 identical nudges) |
| 3. the single-token answer | the same, and under GREEDY too (run E, 3 of 3) — so not a sampler effect at all; the penalty set is empty at the first token of every answer and greedy arms no sampler. Greedy flip between 153,755 and 163,401 ids; live onset 172,599 | this page §1, §3 | model / quant on this history |
| the mixed sampling row | HALF the engine's: `presence_penalty` and `seed` are `serve` defaults, Crow never sent them. Crow's half is that its global row is the card's THINKING row for a model rendered with `enable_thinking false` | this page §2 | both; the engine now names the source of every value, Crow needs one row per model |
| 105 identical nudges, no loop detection, no per-step cap | Crow | `session.json`: 105 of 114 user turns identical, the last 10 messages are consecutive nudges with no assistant turn between them | Crow (separate issue) |

## 6. Open questions

1. **What in the history flips it?** The band is bracketed (healthy at 153,755, degenerate at
   163,401, both greedy on robin's history) and a clean history of 178,553 ids does not flip at
   all — so it is not length alone. The candidates the artefact offers are the 105 byte-identical
   nudges, the contradictory tool output of 300 churning turns, and the 17 images; none of them is
   isolated here. The cheap next measurement is the same replay with the nudges de-duplicated.
2. **Is it the quant?** CNQ4.5-M at 4.5 bpw has never been compared against a higher-bit container
   or against llama.cpp at 100k+ context. The ten-task gate runs at 16k. Nothing here separates
   "this model at 170k" from "this quant at 170k".
3. **Is the QSA regime at long context numerically clean?** The parity gate proves bytes at 8, 512
   and 1024 rows; there is no parity form at 100k+ and no oracle to compare one against. A quality
   gate is not a numeric gate — §4 measures answers, not logits.
4. **Does the engine owe the client a brake?** A cross-turn repetition signal (e.g. "this answer
   is the k-th identical answer of this session" in the `[chat]` line, or a documented
   `repeat_penalty` knob) is a deliberate feature decision for robin, not a bug fix.
5. **The image path at long context** is unmeasured here: the replay is text-only, and the live
   session carried 17 images (~11k visual tokens) in every late request.

## 7. How to reproduce, in order

```bash
# one engine at a time: no serve/decode/parity alive, engine/.engine.lock absent
tools/serve-linux.sh --port 8099 > decode_out/68/serve.log 2>&1 &

# the three rows over the same ~121k prefix, then the live tail twice
tools/replay-session.py --target-tokens 120000 --row greedy --turns 3 --out decode_out/68/a.json
tools/replay-session.py --target-tokens 120000 --row card   --turns 3 --out decode_out/68/b.json
tools/replay-session.py --target-tokens 120000 --row crow   --turns 3 --out decode_out/68/c.json
tools/replay-session.py --cut-index 573 --row crow   --turns 3 --out decode_out/68/d.json
tools/replay-session.py --cut-index 573 --row greedy --turns 3 --out decode_out/68/e.json

# the onset sweep, ascending so every prefill reuses the one below it
for t in 130000 140000 150000 160000; do
  tools/replay-session.py --target-tokens $t --row greedy --turns 2 --out decode_out/68/onset-$t.json
done

# the quality gate at both sizes (exit 0 = >= 4 of 5 Pass and 0 degenerate)
tools/longctx-gate.py --target-tokens 100000 --out decode_out/68/longctx-100k.json
tools/longctx-gate.py --target-tokens 170000 --out decode_out/68/longctx-170k.json

# stop by PID, then remove the lock a kill leaves behind, then the byte gate
kill <serve pid>; rm -f engine/.engine.lock
tools/gate-linux.sh decode_out/gate-68
```

- Wall clock on this machine (2026-09-18): a cold prefill is 2.4 minutes at 121k and 3.3 minutes at
  178k, the sweep points after the first are 12 to 30 seconds each, a gate run at one size is the
  prefill plus about 30 seconds of probes, and `tools/gate-linux.sh` is about 6 minutes.
- The whole series above is about 35 minutes of GPU time. The 2 hours the first plan feared came
  from assuming the live per-turn rate (138 to 266 tok/s on 100-to-300-token prefills) applies to a
  120k prefill; it does not — that rate is the small-batch floor of a warm turn, not the cold rate.
