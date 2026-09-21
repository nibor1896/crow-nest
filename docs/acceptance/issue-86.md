# Acceptance protocol — issue #86 (OpenAI stop strings + logit_bias)

- Date: 2026-09-20. Executed by: fleet subagent (implementation + unit tests). Live/GPU acceptance: robin.
- Issue: finish the OpenAI completion contract — `stop: [string]` (generation ends BEFORE the sequence is emitted; partial-prefix buffering so no half of a stop string ever reaches the wire) and `logit_bias: {token: f32}` (additive on the raw logits, first, outside the chain; −INFINITY = hard mask). Provenance: llama.cpp server (`server-schema.cpp:475-487`, `server-context.cpp:572`); `ignore_eos` is `logit_bias_eog`, i.e. EOS suppression at logit level.

## What changed (file:line, all lines post-change)

| file | where | what |
|---|---|---|
| `engine/src/stopstr.rs` (NEW) | whole module | `StopStrings`: the tail-hold of `toolcall.rs` on arbitrary strings. `new` (drops empties, sorts LONGEST FIRST), `push` (emit-safe prefix out, hold the could-still-match tail), `flush` (a never-completed hold is text and leaves), `hit`/`matched`/`matched_at`/`dropped` (the log line's numbers). Module doc carries the match-policy table |
| `engine/src/toolcall.rs:279-283` | `find_marker` | `pub(crate)` only — `stopstr` runs the same hold; the tool-call behaviour of this copy is byte-identical, pinned by every existing test of that module |
| `engine/src/lib.rs:92` | module map | `pub mod stopstr;` registered (L0, `bin/serve` only, beside `toolcall`) |
| `engine/src/bin/serve.rs:1130-1135` | `ChatReq` | `stop: Vec<String>` (body order), `logit_bias: Vec<(usize, f32)>` (sorted by token id) |
| `engine/src/bin/serve.rs:1429-1453` | `parse_chat` | `stop`: array of strings (a bare string is the one-stop OpenAI form), empty entries dropped, no cap; non-string entry → 400 `stop[i] is not a string` |
| `engine/src/bin/serve.rs:1455-1491` | `parse_chat` | `logit_bias`: object token-id-key → number; key not an id → 400; key ≥ `geo::V` → 400 naming the vocabulary range; value not a number → 400 naming the key; `-1e39`/`1e39` are ∓inf once f32 (strict JSON cannot carry the `-Infinity` literal — the body refuses it, pinned by test) |
| `engine/src/bin/serve.rs:1527-1668` | `sampler_from` + `apply_logit_bias` | bias arms a sampler even in greedy (the biased argmax, host route); greedy bias-only takes the data-sheet presence 1.5 OUT unless the body sent it. `apply_logit_bias`: additive, an infinite bias (either sign) SETS the logit — a mask stays a mask, `-inf + x` never becomes NaN |
| `engine/src/bin/serve.rs:1670-1680` | `draw_biased` | one host draw: `cuda::dtoh(eng.logits(), V)` → `apply_logit_bias` FIRST → `Sampler::sample` → `observe` (the bookkeeping the device node does for itself) |
| `engine/src/bin/serve.rs:2832-2864` | `send_emits` | `stops` gate: after a hit the rest of the batch (content AND tool fragments) is swallowed; the stop filter runs per piece |
| `engine/src/bin/serve.rs:2866-2896` | `send_split` | the CONTENT half passes the stop filter LAST (after the think split); the reasoning half is never stop-scanned; a fully-held piece sends no frame and counts nowhere |
| `engine/src/bin/serve.rs:3339-3375` | `chat_generate` arming | `(Some(s), biased)` arm: parks the device sampler, draws the first token on the host, logs `[chat] logit_bias: N entries (request), drawing on the HOST sampler (#86) ... the device mask-path twin is the follow-up` (plus `sampling_line` when sampled, `greedy with logit_bias` when greedy) |
| `engine/src/bin/serve.rs:3547-3556` | decode loop | `decode_step` then, when biased, the host re-draw replaces the argmax id (`draw_biased`, ~0.3 ms readback per token, the route's price) |
| `engine/src/bin/serve.rs:3422-3426` | `chat_generate` filter | `StopStrings::new(&req.stop)` beside `ToolStream` and `ThinkFilter` — it never touches an id |
| `engine/src/bin/serve.rs:3648-3670` | loop + tail | stop hit inside the loop → `finish = "stop"`, break (above the budget counter and the probe — the answer is over); the tail path feeds the last pieces through the same gates, then `stops.flush()` releases a never-completed hold as text; a stop completing in the FINAL tail flips `length` → `stop` (the sequence IS in the generated text); one `[chat] stopped on a stop string (#86): "<stop>" at content byte N, M byte(s) swallowed` line |
| `engine/src/bin/serve.rs:3709` | finish precedence | `ts.closed() > 0 && !stops.hit()`: a stop hit keeps `stop` over `tool_calls` — the call's fragments were swallowed with everything after the stop, so `tool_calls` would promise a call the client never received |

## Spec-adherence checklist (issue text → implementation)

| #86 requirement | where it lives |
|---|---|
| "`serve.rs` `parse_chat`: accept `stop: [string]` and `logit_bias: {token: f32}`; document both in the field table" | `parse_chat` sites above; the module field table of `serve.rs` carries both rows; the "Rendering and generation", "Stream shape" and sampling sections document the behaviour |
| "stop handling server-context.cpp:572" (earliest occurrence ends generation, sequence not emitted) | `StopStrings::push`: earliest byte position across the stop list; the stop and everything after it swallowed |
| "partial-prefix buffering required so no half of a stop string ever reaches the wire" | the hold: `find_marker`'s safe prefix; `stopstr` tests pin "no emission ever contains a stop" at every piece size, byte-by-byte included |
| "incremental detokenizer + hold-buffer identical in spirit to the existing tool-call tail-hold — reuse that mechanism" | the SAME `toolcall::find_marker` does the scan (`pub(crate)`, behaviour byte-identical); no logic lives twice |
| "on match, emit nothing and finish with `finish=stop`" | loop break sets `finish = "stop"`; the tail-completion flip covers a stop that completes on the last token |
| stop string spans a chunk boundary | held across any number of pieces (`a_stop_string_split_across_pieces_is_held_not_emitted`: "E", "N", "D") |
| one stop contains another — longest-match-first, documented | `new` sorts longest-first; `find_marker` keeps the first slice entry on position ties → the longest is NAMED. Wire bytes are identical either way (generation ends at the match) — pinned by `when_one_stop_contains_another_the_longest_is_named` |
| interaction with ThinkFilter documented (stop applies to the CONTENT channel after think-splitting) | `send_split` runs the stop filter on `Split.content` only; `stopstr` module doc "Channel contract"; `stop_strings_apply_to_the_content_after_the_think_split` pins that an "END" inside the think block does NOT stop the answer |
| "logit_bias: additive map applied first (host sampler)" | `draw_biased`: row read back → `apply_logit_bias` → `Sampler::sample` (llama.cpp order: the map is outside/before the chain) |
| "−INFINITY means hard-masking like EOS" | an infinite bias SETS the logit (both signs); mask never becomes NaN; `-1e39` on the wire is −inf as f32 |
| "device path via the existing per-request mask/count buffers or host fallback" — host route taken | biased requests draw on the host; the device twin (bias through the sampler's mask/count buffers) is named as the follow-up in the code comment, the `[chat]` line and the module doc |
| defaults empty = behavior byte-identical | empty `stop`/`logit_bias` → no filter, no sampler, no held byte; `without_stop_and_logit_bias_the_stream_is_the_pre_86_bytes` pins the stream byte-for-byte; every pre-#86 serve test unchanged (265 baseline green) |
| "Generalize `sample::EOS_IDS` into a per-request stop set" (foundation for `ignore_eos`) | not re-plumbed in `sample.rs` (owned by the #84/#87 line of work); `ignore_eos` rides `logit_bias` instead: masking the EOS ids is exactly `logit_bias_eog` — demonstrated by `a_bias_is_additive_and_minus_infinity_masks` (both EOS ids masked → the draw steps aside) — no new plumbing needed |

## Unit tests (all green; counts in the final report)

| test | pins |
|---|---|
| `stopstr::a_stop_string_ends_the_answer_and_never_reaches_the_wire` | basic: everything before the stop leaves, the stop and the tail are swallowed, frame for frame |
| `stopstr::a_stop_string_split_across_pieces_is_held_not_emitted` | "E"+"N"+"D": held pieces emit nothing; byte-by-byte and whole-piece splits give the same wire bytes |
| `stopstr::when_one_stop_contains_another_the_longest_is_named` | ENDMARK over END at the same position; wire bytes identical either way |
| `stopstr::the_earliest_stop_position_wins_across_different_stops` | earliest byte position across the list |
| `stopstr::a_stop_string_at_byte_zero_empties_the_answer` | the empty-content finish: nothing emitted, the match still lands |
| `stopstr::an_absent_stop_list_is_a_byte_identical_passthrough` | the default: verbatim passthrough, nothing held, nothing dropped |
| `stopstr::a_partial_prefix_that_never_completes_leaves_at_flush` | no byte of the answer is lost to the hold |
| `stopstr::empty_entries_are_dropped_and_duplicates_are_harmless` | the parse contract the filter also defends |
| `stopstr::non_ascii_stop_strings_hold_on_character_boundaries` | →, é, 🦆!: the hold cuts on char boundaries only |
| `stopstr::every_split_gives_the_earliest_stop_and_never_leaks_or_loses_a_byte` | the sweep: nested/overlapping/adjacent hazards × piece sizes 1..1000 — earliest stop, no leak, no loss, always |
| `serve::stop_parses_in_both_wire_forms_and_refuses_garbage_with_named_reasons` | array + bare-string forms, empty entries dropped, absent/null = pre-#86, 400s name the field and the entry |
| `serve::logit_bias_parses_token_ids_strictly_and_sorts_by_token` | id keys sorted, ∓inf via `±1e39`, out-of-vocabulary key refused by name, non-number refused by key, `-Infinity` literal refused by the body parse |
| `serve::a_bias_is_additive_and_minus_infinity_masks` | additive; promote flips the argmax; −inf masks; a mask stays a mask (never NaN); masking BOTH `EOS_IDS` steps aside (the `ignore_eos` shape) |
| `serve::a_biased_request_draws_on_the_host_even_in_greedy` | greedy+bias → Some sampler, biased argmax, presence 0 unless sent; sampled+bias keeps the #28/#68 contract; greedy+penalties+bias keeps the #84 presence rule; plain greedy still `None` |
| `serve::stop_strings_apply_to_the_content_after_the_think_split` | "END" inside the think block does not stop; the same letters in content do; counters name frames written |
| `serve::a_stop_hit_swallows_the_tool_fragments_that_followed_it` | a call whose markup came after the stop never reaches the wire (why `finish` keeps `stop` over `tool_calls`) |
| `serve::without_stop_and_logit_bias_the_stream_is_the_pre_86_bytes` | defaults: the stream gate is the think filter alone, byte for byte |
| toolcall regression | every existing `toolcall::tests` / `arguments_contract` test unchanged and green (`find_marker` visibility only) |

## Test cases for robin's LIVE acceptance

```
flock /tmp/crow-gpu.lock -c 'cargo run --release --bin serve -- --port 8099' 2>serve.log &
```

| # | what to check | how | expected |
|---|---|---|---|
| 1 | stop string ends the answer | `curl -s http://127.0.0.1:8099/v1/chat/completions -d '{"messages":[{"role":"user","content":"count to twenty, digits separated by spaces"}],"max_tokens":512,"stop":[" 7"]}'` (append `,"stream":true` for the SSE form) | the answer stops right before ` 7`: no frame's content ends in a partial ` 7` prefix, the last content frame ends at the byte before the stop, `finish_reason` is `"stop"`, and serve.log carries `[chat] stopped on a stop string (#86): " 7" at content byte N, M byte(s) swallowed` |
| 2 | stop spanning a chunk boundary | same request with `"stop":["seventeen"]` (a word the tokenizer will split) | identical content cut, whatever the token split — the hold is byte-based |
| 3 | the empty-content finish | `"stop": ["1"]` on the count prompt (the answer starts with `1`) | `content` is `""`, `finish_reason` `"stop"`, no content frame on the stream |
| 4 | reasoning is never stop-scanned | a thinking request (`"chat_template_kwargs":{"enable_thinking":true}`) whose reasoning names a word that is also the stop | the stop fires in the CONTENT only; `reasoning_content` frames stream through untouched |
| 5 | defaults invisible | the same curl without `stop`/`logit_bias`, twice: once on this build, once on the pre-#86 build | byte-identical answers (the golden check) |
| 6 | logit_bias bans a token (greedy) | `"temperature":0,"logit_bias":{"<id of the token the answer starts with>":-1e39}` (ids from `serve tokenize --raw --text ...`) | the first token changes; serve.log carries `[chat] logit_bias: 1 entries (request), drawing on the HOST sampler (#86)` and `greedy with logit_bias (#86): the biased argmax, presence_penalty 0 (data sheet)` |
| 7 | logit_bias promotes a token | `"logit_bias":{"<id>":10}` | the promoted token dominates the answer |
| 8 | ignore_eos shape | `"logit_bias":{"248046":-1e39,"248044":-1e39}` on a short-prompt request | the model does NOT end at `<|im_end|>`; the answer runs to `max_tokens` (`finish_reason` `"length"`) — the `logit_bias_eog` mechanism |
| 9 | strict parse | `"stop": [1]`, `"logit_bias": {"x": 1}`, `"logit_bias": {"248320": 1}`, `"logit_bias": {"5": "high"}` | 400s: `stop[0] is not a string`, `logit_bias key "x" is not a token id`, `... is not a token id of this model's vocabulary (0..248320)`, `logit_bias[5] is not a number` |
| 10 | unit suite | `cd engine && cargo test` | all green (see final report for the count) |

## Deviations from the issue text (with justification)

1. **The device twin of `logit_bias` is not built; biased requests take the host route, per the issue's own alternative.** The issue: "device path via the existing per-request mask/count buffers or host fallback". The device `sample_k` node has no bias input, and threading ±inf additive biases through the u8 mask path is its own issue; the host route (`cuda::dtoh` readback → `apply_logit_bias` → `Sampler::sample`) is the chain's bit-for-bit reference (`CROW_SAMPLE_HOST=1`'s road), costs ~0.3 ms/token, and is documented as the decision in the code comment, the `[chat]` line and the module doc. The mask/count-buffer twin is named as the follow-up.
2. **`-Infinity` cannot ride strict JSON, so the mask has a wire form.** serde_json refuses the `-Infinity` literal at the BODY level (pinned by test). The mask is "any bias that is −inf once cast to f32" — any magnitude over `f32::MAX`, e.g. `-1e39`. llama-server's nlohmann accepts the literal; that is the one client-visible difference, and `-100` (the OpenAI-conventional strong ban) works everywhere as an additive, non-absolute ban.
3. **`sample::EOS_IDS` was NOT generalized into a per-request stop set** (issue point 4). `sample.rs` is owned by the parallel #84-line work; re-plumbing `EOS_IDS` there would collide with it. The goal of point 4 — `ignore_eos` without new plumbing — is reachable through `logit_bias` today: masking the EOS ids is exactly llama-server's `logit_bias_eog` (demonstrated in unit test). The generalization can land with the device twin.
4. **A greedy bias-only request drops the data-sheet `presence_penalty` 1.5 (unless sent).** The host route runs `Sampler::sample`, whose greedy branch would subtract presence over the answer's own tokens — a penalty the request never asked for. The #84 greedy-armed path keeps its rule (the request named penalties there); here presence is 0 unless the body sent it, and the `[chat]` line says so.
5. **A stop hit overrides `tool_calls` on `finish_reason`.** If the stop matched before the call's markup, the call fragments were swallowed — `tool_calls` would promise a call the client never received. EOS keeps the old precedence (its stop is atomic, nothing is swallowed).
6. **No count cap on `stop`** (OpenAI documents up to 4, llama-server up to 8). This server caps nothing else about the profile either; the scan is O(text × stops) with the same early-out `find_marker` always had.
7. **Empty `stop` entries are dropped, not refused** — llama-server drops them; refusing `["", "END"]` would break a real client over a no-op.
