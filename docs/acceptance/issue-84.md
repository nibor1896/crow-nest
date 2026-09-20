# Acceptance protocol — issue #84 (llama.cpp-style windowed repetition + frequency penalties)

- Date: 2026-09-20. Executed by: fleet subagent (implementation + unit tests). Live/GPU acceptance: robin.
- Issue: the llama.cpp `penalties` sampler — sliding window `ring_buffer prev(penalty_last_n)` (default 64) spanning PROMPT + GENERATED, incremental `token_count`, asymmetric repeat (`l>0 ? l/repeat : l*repeat` — the CTRL-paper fix), then `l -= c·freq + (c>0)·present`; applied in GREEDY and SAMPLED paths alike; defaults neutral so no existing row moves implicitly.

## What changed (file:line, all lines post-change)

| file | where | what |
|---|---|---|
| `engine/src/sample.rs:77-85` | `Sampler` fields | `repeat_penalty: f32` (1.0), `frequency_penalty: f32` (0.0), `penalty_last_n: usize` (64) |
| `engine/src/sample.rs:96-104` | `win` / `win_counts` state | the window: `VecDeque<u32>` of the last `penalty_last_n` accepted ids (prompt tail first) + incremental count map; a token evicted from the ring decrements by exactly one |
| `engine/src/sample.rs:136-138` | `from_env` | `CROW_REPEAT` / `CROW_FREQ` / `CROW_LASTN` knobs (docs/env.md rows added), `CROW_LASTN` clamped to `SAMPLE_RING_MAX` |
| `engine/src/sample.rs:192` | `win_armed` | `last_n > 0 && (repeat != 1.0 \|\| freq > 0)` — llama.cpp's "repeat 1.0 or window 0 disables" |
| `engine/src/sample.rs:204` | `win_pen` | the per-candidate penalty: the WHOLE block gated on `c > 0` (llama.cpp gates on a `token_count` hit), then asymmetric div/mul, then `l -= c*freq + presence` |
| `engine/src/sample.rs:223` | `win_push` | ring push + evict/decrement |
| `engine/src/sample.rs:241` | `observe` | now ALSO accepts into the window (the HF `seen` set keeps its #68 role while the window is not armed) |
| `engine/src/sample.rs:250` | `observe_prompt` | seeds the window with the prompt's last `penalty_last_n` ids; `seen` stays EMPTY (the #68 pin) |
| `engine/src/sample.rs:271-316` | `sample` | penalties FIRST on the raw logits, in BOTH branches: greedy (penalized argmax — greedy had no penalty before #84) and sampled (before top-k), armed window → llama.cpp form, not armed → the HF presence subtraction, byte-identical to before |
| `engine/src/kernels.rs:4108-4110` | `sample_topk_part` | `counts` param; the same gated penalty branch on the raw logits, exact op order of the host's `win_pen`; not armed → the untouched #68 mask branch |
| `engine/src/kernels.rs:4184-4187` | `sample_k` | `counts` + `ring` params; after the draw, thread 0 runs the llama.cpp accept: full ring → evict `ids[head]` and decrement its count, else fill+1; write `ids[head] = tok`, `counts[tok] += 1`, advance head |
| `engine/src/gen.rs:667-669, 1923` | `DevSampler` | `counts` [V] u16 + `ring` i32 {head, fill, ids[1024]}; `SAMPLE_RING_MAX = 1024` (a `KERNEL_SRC` #define twin, added to `assert_kernel_defines`) |
| `engine/src/gen.rs:4228-4260` | `enable_dev_sampler` | per-request seeding, like the mask: params [6..9] = repeat/freq/last_n (clamped), `counts[V]` u16 from the sampler's window counts, ring {head=fill%lastn, fill, ids} from its prompt-tail ids |
| `engine/src/gen.rs:1898` | `sampler_bytes()` | ledger mirror: +2V (u16 counts) + 4104 B ring → 781,876 B |
| `engine/src/bin/serve.rs:1329-1350` | `parse_chat` | `repeat_penalty`/`frequency_penalty` (num_field, strict on type) + `penalty_last_n` (integer, clamped to 1024); `SamplingSent` carries all three |
| `engine/src/bin/serve.rs:1505-1526` | `sampler_from` | greedy + armed penalties now BUILDS a sampler (`temperature <= 0`, device draw `ci[0]` = penalized argmax); plain greedy, and greedy with only top_p/min_p/seed, is still `None` |
| `engine/src/bin/serve.rs:3147-3153` | `chat_generate` | `sampler_from` + `observe_prompt(ids)` BEFORE `arm_sampler` — the window spans prompt + generated on host and device alike, whatever the prefix cache reused |
| `engine/src/bin/serve.rs:3160-3185` | provenance lines | `sampling_line` gains repeat/freq/last_n with (request)/(data sheet) tags; a greedy+penalties request logs `greedy with windowed penalties (#84)`; the #68 presence line is replaced BY an armed-window line naming the semantics switch |
| `engine/src/manager.rs:498,517` | ledger pins | sampler VRAM pin moved 281,132 → 781,876 (the issue's device twin IS VRAM); the `[budget]` line pin 277.6 → 278.1 MB |
| `docs/env.md` | Sampler table | `CROW_REPEAT` / `CROW_FREQ` / `CROW_LASTN` rows (12 rows now), summary list updated |

## Spec-adherence checklist (issue text → implementation)

| #84 requirement | where it lives |
|---|---|
| "add `repeat_penalty: f32`, `frequency_penalty: f32`, `penalty_last_n: usize` to `Sampler`" | `sample.rs:77-85` |
| "Sliding window ring_buffer(penalty_last_n) (default 64), spanning prompt + generated tokens; incremental token_count — a dropped old token decrements its count exactly" | host `win`/`win_counts` (`win_push`), device `ring`/`counts` (`sample_k` accept), seeded per request from the prompt tail by `observe_prompt` + `enable_dev_sampler` |
| "logit /= penalty_repeat if logit > 0 else logit *= penalty_repeat — the asymmetric form" | `win_pen` / `sample_topk_part`, gated on count > 0 exactly as llama.cpp gates the block on a `token_count` hit |
| "logit -= c·penalty_freq + (c>0)·penalty_present" | same sites: inside the gate, `v -= c*fq + presence` (the c>0 test is the gate itself) |
| "penalties sampler FIRST in the default chain, applied to raw logits; runs in greedy and sampled paths alike" | host `sample` applies penalties before argmax / before top-k; serve arms the device sampler for greedy+penalties too (`sampler_from`), the device draw at temp<=0 is `ci[0]`, the penalized argmax |
| "Defaults: penalty_last_n=64, penalty_repeat=1.0, penalty_freq=0" (ours keeps presence 1.5 as the data sheet) | `Sampler::new` / serve defaults; `win_armed` false at defaults |
| "penalty_repeat=1.0 (or window 0) disables the whole sampler" | `win_armed`: `last_n > 0 && (repeat != 1.0 \|\| freq > 0)` |
| "EOS is NOT special-cased" | no special case anywhere; suppression stays a future logit-bias mechanism |
| "Device twin: u16 counts[V] + ring buffer in global memory, mirrored per request like the existing presence mask; zeroed per request" | `DevSampler.counts/ring`, seeded-or-zeroed per request in `enable_dev_sampler`, advanced by `sample_k` |
| "Defaults neutral … so no existing row changes implicitly" | pinned by `windowed_defaults_are_neutral_and_goldens_stay_byte_identical`: both pre-#83/#84 goldens byte-identical with the new fields at defaults; the #68 presence tests and the #83 min_p-disabled golden all still pass unchanged |
| "Bit-parity gate vs CROW_SAMPLE_HOST=1 as usual" | unit level: `the_device_mirror_draws_what_the_host_draws_windowed` — a line-for-line host mirror of the CUDA (window accept included) agrees token-for-token with `Sampler::sample` over armed profiles, prompt-seeded, sampled AND armed-greedy. The GPU run is robin's live gate (case 4 below) |
| "thread the last-last_n ids into the sampler state at request build" | `chat_generate` calls `observe_prompt(ids)` before `arm_sampler` |

## Unit tests (all green; `cargo test` = 264 passed / 0 failed across 24 targets)

| test | pins |
|---|---|
| `sample::tests::windowed_defaults_are_neutral_and_goldens_stay_byte_identical` | defaults (1.0 / 0.0 / 64) do not arm; window 0 disables; the sampled AND greedy goldens of the pre-#83/#84 code are byte-identical |
| `sample::tests::repeat_penalty_is_asymmetric_div_for_positive_mul_for_negative` | positive divided, negative multiplied (CTRL fix); an UNCOUNTED token is never divided |
| `sample::tests::frequency_penalty_scales_with_the_window_count` | once per occurrence (count 2 vs count 4 flips the argmax) |
| `sample::tests::presence_joins_the_window_form_when_armed` | while armed: prompt tokens (count>0) get `(c>0)*presence`, count-0 tokens untouched; while not armed: presence stays HF and the prompt is invisible |
| `sample::tests::the_window_spans_the_prompt_tail_and_evicts_exactly` | prompt tail seeds the window; an evicted id loses its count; a drawn id enters it |
| `sample::tests::greedy_applies_the_windowed_penalties_too` | the greedy path is penalized when armed, plain argmax when not |
| `sample::tests::the_device_mirror_draws_what_the_host_draws_windowed` | host == device-mirror, 4 armed profiles x 12 sequential draws incl. armed GREEDY, prompt-seeded both sides |
| `serve::tests::penalty_fields_parse_with_neutral_defaults_and_plumb_into_the_sampler` | defaults neutral, request values win, tags on the `[chat]` line, null = absent, wrong type = 400 |
| `serve::tests::penalty_last_n_clamps_to_the_device_ring` | > 1024 clamps to SAMPLE_RING_MAX (the line names the effective value) |
| `serve::tests::greedy_with_penalties_arms_a_sampler_plain_greedy_does_not` | greedy + repeat or freq arms; window 0 or neutral knobs does not; top_p/min_p/seed alone never does |

## Test cases for robin's LIVE acceptance

```
flock /tmp/crow-gpu.lock -c 'cargo run --release --bin serve -- --port 8099' 2>serve.log &
```

| # | what to check | how | expected |
|---|---|---|---|
| 1 | penalties echoed with provenance | `curl -s http://127.0.0.1:8099/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"temperature":1.0,"top_p":0.95,"min_p":0.01,"repeat_penalty":1.05,"frequency_penalty":0.3}'`; `grep -E "sampling on the device\|penalty window" serve.log` | the line carries `repeat_penalty 1.05 (request) frequency_penalty 0.3 (request) penalty_last_n 64 (data sheet)`, then `penalty window armed: last 64 ids, prompt tail + generated (#84)` |
| 2 | defaults are invisible | same curl WITHOUT the penalty fields | `repeat_penalty 1 (data sheet) frequency_penalty 0 (data sheet)` and the #68 presence line, NOT the window line; a request identical to a pre-#84 row answers byte-identically |
| 3 | greedy penalties work (the #68 echo-loop counterweight) | ask the same factoid THREE times in one session with `"temperature":0,"repeat_penalty":1.1`; then again without the fields | with the fields: the third answer's `[chat]` line says `greedy with windowed penalties (#84)` and the model avoids verbatim repeats of the earlier answers; without them, `greedy (temperature absent or <= 0)` as always |
| 4 | host/device bit parity (the gate of the issue) | one golden prompt sampled with the device sampler, then with `CROW_SAMPLE_HOST=1` on the host path, both with `repeat_penalty=1.05` armed | identical token streams |
| 5 | the #68 long-session brake | a goal-mode-style long session with `repeat_penalty`/`frequency_penalty` sent | `repeat_of`/`repeat_run` WARN incidence drops; no single-token answer runs |
| 6 | unit suite | `cd engine && cargo test` | 264 passed / 0 failed |

## Deviations from the issue text (with justification)

1. **The whole per-candidate block is gated on `c > 0`.** The issue's formula line ("per candidate with window count c: logit /= repeat …; logit -= c·freq + (c>0)·present") is ambiguous about tokens with c = 0; llama.cpp's source applies the ENTIRE block (repeat div/mul included) only to candidates found in `token_count`. Since the issue says "the exact div/mul asymmetric form … matching llama.cpp semantics" and the reference is the source, both twins gate on count > 0. A token outside the window keeps its raw logit.
2. **`presence_penalty` joins the window only while armed; it does not arm it.** The issue's point 4 says "present stays the existing knob until migrated … so no existing row changes implicitly", but its formula uses `penalty_present` over the window. Resolution: the VALUE stays the existing `presence_penalty` knob (there is no new field); while the window is armed (repeat != 1.0 or freq > 0) that value rides the llama.cpp form `(c>0)*presence` and the HF subtraction is not read; while it is not armed (the defaults, and any request that names only `presence_penalty`), presence is exactly what #68 pinned. Making presence itself arm the window would have armed it by default (the data sheet's 1.5) and moved every existing row — the exact thing point 4 forbids. `penalty_last_n = 0` disables everything, as in llama.cpp.
3. **Greedy arming means the sampler node runs for greedy+penalties requests.** A plain greedy request still parks the sampler and runs `argmax_k` (byte-identical); only a request that names repeat/freq arms, and then `sample_k` with `temp <= 0` returns `ci[0]` — the penalized argmax, the same order of operations the host greedy branch runs.
4. **`penalty_last_n` is clamped to 1024** (the device ring's depth, `SAMPLE_RING_MAX`), silently but VISIBLY — the `[chat]` line prints the effective value. llama-server rejects values above context; a clamp keeps the engine's lenient-on-range discipline (top_k's) and the clamp is on the record.
5. **`manager.rs` ledger pins moved** (sampler 281,132 → 781,876 B; `[budget]` line 277.6 → 278.1 MB). The counts buffer (2·V) and the ring are device VRAM the issue's point 3 mandates; the pins mirror `sampler_bytes()` by design. Same class of edit as the #83 one, confirmed with the orchestrator. No other manager.rs line touched.
6. **Crow client work (issue point 5) is out of scope here** — this unit is engine-side; the Crow row moves in the Crow repo once both arms accept the fields.

---

## Orchestrator verification (appended 2026-09-20, all checks passed)

1. Full suite re-run by orchestrator: **265 passed, 0 failed**.
2. Formula conformance vs llama.cpp `penalties` (src/llama-sampler.cpp): verified by direct code read — candidate not in window untouched (c==0), asymmetric `l > 0 ? l/repeat : l*repeat`, then `l -= c*freq + presence` (sample.rs `win_pen`); ring eviction decrements exactly; window spans prompt tail + generated (`observe_prompt` + per-accept push); window 0 disables the whole pass; greedy applies the armed penalties (serve arms `ci[0]` penalized argmax).
3. Semantic-fork decision reviewed and endorsed: `presence_penalty` rides the window ONLY while armed and does NOT arm it — otherwise the data-sheet default 1.5 would have armed-by-default and moved every row (violating the issue's own neutrality requirement); unarmed presence keeps the #68 HF-`seen` form, pinned by test.
4. The manager ledger incident (291080748 vs ...728) resolved as tasked: pins updated to the new sampler-state footprint (281,132 -> 781,876 B device twin; budget line 277.6 -> 278.1 MB) — the VRAM cost of counts[V] u16 + 1024-ring, documented in env.md rows.
5. Footprint audit: ownership map held (manager.rs ledger pins = coordinator-approved class); env.md rows present, checker OK.
6. GPU-level parity deferred to live acceptance / wave-1 gate as with #83 (mirror proof exists: armed windows incl. armed-greedy, host==device).

## robin's live-acceptance short list

| # | check | how | expected |
|---|---|---|---|
| 1 | fields parsed strictly | request with `repeat_penalty: 1.1, frequency_penalty: 0.1, penalty_last_n: 64` | provenance line `penalty window armed`; type errors are 400s |
| 2 | neutral defaults | any existing row, no new fields | byte-identical answers to pre-#84 build |
| 3 | greedy + armed penalties | `temperature: 0` + armed window | penalized argmax (not plain argmax) |
| 4 | the #68 shape | long goal-mode session or replay with armed window | echo/single-token loop suppressed vs the same replay unarmed |
| 5 | clamp visible | `penalty_last_n: 99999` | clamped to 1024, named on the provenance line |
