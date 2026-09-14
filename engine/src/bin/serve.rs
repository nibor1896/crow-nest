//! #24 serve: the HTTP face of the engine (spec section 7).
//!
//! Endpoints:
//!
//! - `GET /health` answers `{"status":"ok"}`.
//! - `GET /props` answers the operating point as a JSON document.
//! - `/props` field names mirror llama-server, so Crow's readers work unchanged.
//! - Crow readers served: `check_endpoint`, `fetch_n_ctx`, `fetch_model_name`.
//! - Crow readers served: `refuse_images`, `server_model_path`.
//! - `POST /v1/chat/completions` streams the answer as SSE (#26 A4).
//! - `GET /slots` answers the one slot of this process as an array of one (#32 A10).
//! - `POST /slots/0?action=save|restore` writes or reads the slot file (#32 A10).
//! - Crow readers served: `save_session` reads `n_saved`, `load_session` reads `n_restored`.
//! - Anything else answers 404 with a JSON body.
//!
//! Connection handling:
//!
//! - Read and write timeout of 10 s on every accepted connection.
//! - A timeout or read error logs one stderr line and closes that connection.
//! - The accept loop continues after any single connection failed.
//! - Head (request line plus headers) capped at 64 KiB total.
//! - The reader itself is bounded, it never buffers past that cap.
//! - Over the head cap: 431 JSON, then close.
//! - `Content-Length` is parsed case insensitively.
//! - A malformed or repeated `Content-Length` answers 400 JSON.
//! - `Transfer-Encoding: chunked` answers 501 JSON, A3 may implement it.
//! - The body is read and discarded before the response is written.
//! - Body capped at 16 MiB; over the cap: 413 JSON, then close.
//! - A bodied POST to an unknown route therefore gets the 404 JSON, not a reset.
//! - A garbage request line answers 400 JSON with `{"error":"bad request"}`.
//! - A client that sent nothing is closed silently, without a response.
//! - The write half is shut down after the response is flushed.
//! - The parsed body is kept as `Vec<u8>` for A3 and A4; A2 routes ignore it.
//!
//! Operating point (M1 decisions, robin 2026-09-09):
//!
//! - Container default `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`.
//! - `CROW_CNQ` overrides the container path.
//! - Hot sets default `decode_out/hotsets-M-longctx2100-n160.json`.
//! - `CROW_HOTSETS` overrides the hot set sidecar.
//! - Prompt chunk pinned at 2048 for the whole process.
//! - The per prompt policy `geo::apply_chunk_policy` is NOT applied.
//! - Same result as `CROW_CHUNK=2048 CROW_CHUNK_AUTO=0` before `Engine::load`.
//! - Context `CONTEXT_FLOOR` (200,000), the operating point of `decode run`.
//! - `n_ctx` is read back from the loaded states (`Engine::st.context`).
//! - Process wide defaults since d36353a need no env variable.
//! - Those defaults: `CROW_PF_ASYNC=2`, PF_TG 64, PLE 128 MB, attn_sel_s8l.
//!
//! Environment defaults set by `serve` itself (#26 review, #37 fix round 1):
//!
//! | variable | behaviour |
//! |---|---|
//! | `CROW_GRAPH`, `CROW_MMA` | default 1 in serve (the gated configuration); env overrides |
//! | `CROW_ADAPT_WINDOW` | default 1 in serve (the trickle's ranking signal); env overrides |
//!
//! - All three are set in `main` BEFORE `cuda::Ctx::init` and before the first kernel call.
//! - `gen.rs` reads graph and mma once through a `OnceLock`, so the order is the whole contract.
//! - `CROW_ADAPT_WINDOW` is read per tick (`gen.rs:3048`, `gen.rs:3066`), always after this point.
//! - Only an unset variable is set; an explicit `CROW_GRAPH=0` still turns graphs off.
//! - One stderr line per variable carries the effective value.
//!
//! Hot set adaptation (#26 review, #37):
//!
//! - `apply_adapt_policy` prints a `[policy] ... every N ...` line at start.
//! - #37: `serve` ticks the STREAM TRICKLE once per `decode_step`, the mirror of `decode.rs:224-231`.
//! - `adapt_tick` (the post-prefill re-cut of `CROW_ADAPT=1`) is still NOT called by `serve`.
//! - The tick runs only when the policy asked for the stream trickle with `every > 0`.
//! - #37 fix round 1: `serve` sets `CROW_ADAPT_WINDOW=1` when unset, so the tick ranks swaps
//!   by the decayed selections since the last tick, not by the prefill-dominated cumulative count.
//! - The three variables `serve` sets when unset: `CROW_GRAPH`, `CROW_MMA`, `CROW_ADAPT_WINDOW`.
//! - Two `trickle_tick` preconditions (`gen.rs:3128-3129`) are checked ONCE at start, not per token.
//! - One stderr line after the policy line says whether this process ticks and why.
//! - Per request the swaps go to the `[chat]` line as `crow_trickle_swaps`; the wire is untouched.
//! - The trickle is drained after the last `decode_step`, so no copy crosses a request boundary.
//!
//! Concurrency (constant 4: one binary, no runtime):
//!
//! - `std::net::TcpListener`, blocking, one request at a time, no async stack.
//! - A second connection waits in the accept queue; no 503.
//!
//! One engine per machine:
//!
//! - `Engine::load` takes `engine/.engine.lock` before anything is pinned.
//! - A second `serve` therefore exits non zero with the lock message.
//! - The engine loads BEFORE the socket binds, so the lock speaks first.
//!
//! Unsafe surface:
//!
//! - `cuda::Ctx::init` and `Engine::load` are the only unsafe calls.
//! - Argument parsing, the socket loop and the routes are safe code.
//!
//! Run:
//!
//! - `serve [--port <n>] [--slot-save-path <dir>]` from the repository root (repo relative paths).
//! - Default port 8099, bind address 127.0.0.1.
//! - Without `--slot-save-path` both `/slots/0` actions answer 400, as llama-server refuses them.
//! - One stderr line at start says which directory is in use, or that none is.
//! - `--slot-save-path` must name a directory that already exists; serve never creates one.
//! - A typo exits 2 before the engine is loaded, so it costs a second, not a conversation.
//!
//! Subcommand `tokenize` (#25 A3, no engine, no GPU, no lock):
//!
//! | invocation | effect |
//! |---|---|
//! | `serve tokenize --chat --file <prompts.json> --out <ids.json>` | every prompt as one user message |
//! | `serve tokenize --chat --text "<text>"` | one prompt, id list on stdout |
//! | `serve tokenize --raw --text "<text>"` | `add_special_tokens=false`, id list on stdout |
//!
//! - `--chat` is `add_generation_prompt=true`, `enable_thinking=false`, as `tokenize_ids.py --chat`.
//! - `<prompts.json>` is read as `{id: text}` or as `[{"id": ..., "text": ...}]`.
//! - `<ids.json>` is written as `{task_id: [ids]}`.
//! - Per task the id count goes to stderr, so the gate log carries the ten lengths.
//! - The subcommand returns before `cuda::Ctx::init`, so no CUDA context and no `.engine.lock`.
//! - No Python process is started; `crow_nest_engine::tokenizer` is the whole path.
//!
//! Exit codes of `serve tokenize`:
//!
//! | code | meaning |
//! |---|---|
//! | 0 | ids written (`--file`) or printed (`--text`) |
//! | 2 | usage: bad, missing, doubled or conflicting arguments |
//! | 3 | tokenizer load failed (`tokenizer.json` or `tokenizer_config.json`) |
//! | 4 | IO or encode: prompt file unreadable, not JSON, wrong shape, encode failed, output unwritable |
//!
//! Start of `serve` (no subcommand):
//!
//! - The tokenizer is warmed up right after argument parsing, BEFORE `cuda::Ctx::init`.
//! - A missing or broken tokenizer therefore fails in a second, not after the engine load.
//! - Warm-up failure prints the error plus both paths and exits 3.
//! - Warm-up success logs the two loaded paths, one stderr line each.
//!
//! `POST /v1/chat/completions` (#26 A4), request body:
//!
//! | field | A4 behaviour |
//! |---|---|
//! | `messages` | required, non empty array, every entry needs a string `role` |
//! | `stream` | `true` streams `chat.completion.chunk` frames; `false` or absent answers ONE `chat.completion` document (#39 B3a) |
//! | `max_tokens` | default 1024, capped at 32768 |
//! | `model` | echoed into every chunk, default `crow-nest` |
//! | `chat_template_kwargs.enable_thinking` | template variable, default false |
//! | `temperature` | absent, `null` or `<= 0` is GREEDY (the A4 path); `> 0` samples (#28 A6) |
//! | `top_p` | nucleus mass, default 0.8 (data sheet); read only when `temperature > 0` |
//! | `top_k` | candidates kept, default 20 (data sheet); read only when `temperature > 0` |
//! | `presence_penalty` | default 1.5 (data sheet); read only when `temperature > 0` |
//! | `seed` | RNG seed of THIS request, default 0; a warm process draws what a cold one draws |
//! | `min_p` | ACCEPTED AND IGNORED, the device sampler has no min_p (#28, open for robin) |
//! | `tools` | array of OpenAI function tools, RENDERED as the template variable `tools` (#29 A7) |
//! | `stream_options.include_usage` | `true` puts `usage` on the final chunk (#27 A5) |
//! | `timings_per_token` | `true` puts `timings` on the final chunk (#27 A5) |
//!
//! - Both flags are read leniently: anything that is not JSON `true` counts as off.
//! - Neither flag changes generation; they only add two objects to the last chunk.
//!
//! Tool turns in the request (#29 A7):
//!
//! | message | fields read by the template | note |
//! |---|---|---|
//! | `role: "tool"` | `content` | rendered as `<tool_response>...</tool_response>` inside a user turn |
//! | `role: "assistant"` | `tool_calls[].function.name`, `.arguments` | rendered as the markup below |
//!
//! - `tool_call_id` is CARRIED and never read: this template pairs by order, not by id.
//! - Crow sends `arguments` as a JSON STRING (`crow_core.py:3564`); the template needs a MAPPING.
//! - Measured 2026-09-09 against the Python oracle: the string form raises
//!   `TypeError: Can only get item pairs from a mapping`, so it is not a render at all.
//! - `normalize_messages` therefore parses the string into the object it encodes.
//! - A string that is not a JSON object is left alone, so the render fails loudly with a 400.
//!
//! The markup THIS model uses for a tool call (chat template, measured, not assumed):
//!
//! ```text
//! <tool_call>
//! <function=read_file>
//! <parameter=path>
//! C:/x/y.md
//! </parameter>
//! </function>
//! </tool_call>
//! ```
//!
//! - It is NOT the `<tool_call>{"name": ..., "arguments": {...}}</tool_call>` JSON form.
//! - `<tool_call>` is added token 248058, `</tool_call>` is 248059, both `special: false`.
//! - The other four markers are ordinary text and split across tokens.
//! - The OpenAI arguments JSON object is BUILT from the parameter blocks; see `ToolStream`.
//!
//! Rendering and generation:
//!
//! - `tokenizer::render_chat(messages, tools, add_generation_prompt=true, enable_thinking)`.
//! - Greedy decode: `Engine::prefill` gives the first id, `Engine::decode_step` the rest.
//! - Stops on `sample::EOS_IDS` (`finish_reason` `stop`) or at `max_tokens` (`length`).
//! - `prompt ids >= n_ctx` answers 413 before any GPU work.
//! - Otherwise `max_tokens` is CLAMPED to `n_ctx - prompt ids` and to 32768.
//! - A clamp logs one stderr line and the request is served, not refused.
//!
//! Sampling (#28 A6), what the server path does per request:
//!
//! | `temperature` | first id | rest of the ids | device sampler |
//! |---|---|---|---|
//! | absent, `null`, `<= 0` | `Engine::prefill` (argmax) | `Engine::decode_step` (argmax) | TAKEN OUT of the engine |
//! | `> 0` | `Engine::sample_last` | `decode_step` behind the `sample_k` node | ARMED before the first step |
//!
//! - Greedy is the A4 path unchanged: same calls, same order, no sampler node in the graph.
//! - `sample::EOS_IDS` stops BOTH modes; `CROW_STOP_EOS` is the harness opt-in and is NOT read here.
//! - The sampler is built from the REQUEST, never from the environment.
//! - `CROW_SAMPLE`, `CROW_TEMP`, `CROW_TOP_P`, `CROW_TOP_K`, `CROW_PRESENCE`, `CROW_SEED`
//!   keep working for `decode` and `parity`; `serve` reads none of them.
//! - `Sampler::new(seed)` carries the data-sheet defaults, the request overwrites what it sends.
//! - Absent fields when `temperature > 0`: top_p 0.8, top_k 20, presence_penalty 1.5, seed 0.
//!
//! Per request reseed (M1, robin 2026-09-09):
//!
//! - `Engine::enable_dev_sampler` runs for EVERY sampled request, after `prefill`.
//! - It uploads `Rng::new(seed)` and clears the presence mask, so request k starts cold.
//! - Therefore request 1 of a fresh process and request 5 of a warm one draw the same ids.
//! - The decode graph is dropped before the prefill of every request, cold by
//!   `reset_to_zero` and warm by `PrefixCache::rollback`, both through
//!   `Engine::drop_decode_graph`, so the first `decode_step` re-captures.
//! - Arming BEFORE that first step is what puts the `sample_k` node INTO the new graph.
//! - Armed after it, the sampler would run as an eager launch per replay: correct, slower.
//!
//! Greedy after a sampled request (why the sampler is taken out, not left armed):
//!
//! - `decode_step` samples whenever `Engine::dev_sampler` is `Some` (`gen.rs:2905-2909`).
//! - `reset_to_zero` does NOT clear that field (`reset.rs`, the "needs no reset" table).
//! - So a greedy request after a sampled one would silently sample.
//! - `Srv::parked_sampler` holds the `DevSampler` while a greedy request runs.
//! - The next sampled request hands the same device buffers back, so nothing is reallocated.
//!
//! `min_p` (open decision for robin, #28):
//!
//! - Crow's operating point is temperature 1.0, top_p 0.95, min_p 0.01 (`crow_core.py`).
//! - The device sampler (`kernels.rs sample_k`) implements top_k, top_p and presence only.
//! - A6 does not touch kernels, so `min_p` is parsed, ignored, and logged once per request.
//! - The log line names it: `min_p accepted and ignored (device sampler has no min_p; #28)`.
//! - Consequence: an answer at Crow's operating point has NO min_p floor under the nucleus.
//! - Options for robin: add min_p to `sample_k` (kernel change), or drop it from the profile.
//!
//! Stream shape (llama-server / OpenAI, `crow_core.py:4831-4877`):
//!
//! | order | line |
//! |---|---|
//! | 1 | `data: {... "choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}` |
//! | 2..n | `data: {... "delta":{"content":"..."},"finish_reason":null}` |
//! | n+1 | `data: {... "delta":{},"finish_reason":"stop"[, "usage":{...}][, "timings":{...}]}` |
//! | n+2 | `data: [DONE]` |
//!
//! - Headers: `Content-Type: text/event-stream`, `Cache-Control: no-cache`, `Connection: close`.
//! - No `Content-Length` and no chunked framing: the body is close delimited.
//! - Every frame is flushed on its own; one token never waits for the next, with one
//!   exception (#29 A7): a content token whose tail is a prefix of `<tool_call>` is HELD
//!   until the next token resolves it, so no half marker can reach the wire.
//! - The concatenated content is the same either way; only the frame boundary moves.
//!
//! Tool calls on the wire (#29 A7, `crow_core.py:4864-4877` is the reader):
//!
//! | order | `delta` |
//! |---|---|
//! | 1 | `{"tool_calls":[{"index":0,"id":"call_0","type":"function","function":{"name":"read_file","arguments":""}}]}` |
//! | 2..n | `{"tool_calls":[{"index":0,"function":{"arguments":"<fragment>"}}]}` |
//! | last | `{}` with `finish_reason":"tool_calls"` |
//!
//! - `id` and `name` ride on chunk 1 ONLY; no later chunk carries either key.
//! - Crow overwrites `id` and `name` only on a TRUTHY value, so this can never erase them.
//! - The concatenation of every `arguments` fragment of one index is the arguments JSON text.
//! - A second call gets `index` 1, a third `index` 2; each has its own `call_<index>`.
//! - `finish_reason` is `tool_calls` when at least one call was CLOSED.
//! - `<tool_call>` is matched by TOKEN ID (248058, `ToolStream::arm`), `</tool_call>` by TEXT.
//! - Text before the first `<tool_call>` streams as `delta.content`, unchanged from A4.
//! - Text after it is DROPPED and counted in one stderr line (the template forbids it).
//! - EOS or `max_tokens` after `</function>` but before `</tool_call>`: the call is COMPLETE,
//!   so it is CLOSED, `finish_reason` is `tool_calls`, and the trailing markup is DROPPED.
//! - That case never replays the raw markup as content: the client gets the call, once.
//! - Malformed markup (no `</function>`, no name): the RAW markup goes out as `delta.content`,
//!   `finish_reason` stays `stop` or `length`, one stderr line names it, nothing panics.
//! - A malformed call leaves its `arguments` unterminated on purpose: Crow's `json.loads`
//!   then fails and the tool is not run, which is the safe end of a truncated call.
//! - The parser itself lives in `crow_nest_engine::toolcall` (#29 review); this file keeps
//!   the chunk builders (`chunk_tool_open`, `chunk_tool_args`) and the call sites.
//!
//! `usage` and `timings` on the final chunk (#27 A5):
//!
//! - Both ride on the SAME chunk that carries `finish_reason`, before `data: [DONE]`.
//! - `usage` is written when `stream_options.include_usage` is `true`, else omitted.
//! - `timings` is written when `timings_per_token` is `true`, else omitted.
//! - Neither flag set: the final chunk is exactly the A4 chunk, byte for byte.
//! - Field names and shape mirror llama-server, the server Crow was written against.
//!
//! | object | field | type | value |
//! |---|---|---|---|
//! | `usage` | `prompt_tokens` | int | rendered prompt ids, the cached part included |
//! | `usage` | `completion_tokens` | int | generated ids |
//! | `usage` | `total_tokens` | int | `prompt_tokens + completion_tokens` |
//! | `usage` | `prompt_tokens_details.cached_tokens` | int | `P`, prompt ids reused from the prefix cache (#31 A9) |
//! | `timings` | `prompt_n` | int | prompt ids actually PREFILLED, `prompt_tokens - cached_tokens` (#31 A9) |
//! | `timings` | `prompt_ms` | float | wall of the `Engine::prefill` call |
//! | `timings` | `prompt_per_second` | float | `prompt_n / prompt_ms * 1000` |
//! | `timings` | `prompt_per_token_ms` | float | `prompt_ms / prompt_n` |
//! | `timings` | `predicted_n` | int | generated ids |
//! | `timings` | `predicted_ms` | float | wall of the decode loop, first `decode_step` to the last |
//! | `timings` | `predicted_per_second` | float | `predicted_n / predicted_ms * 1000` |
//! | `timings` | `predicted_per_token_ms` | float | `predicted_ms / predicted_n` |
//! | `timings` | `cache_n` | int | `P`, the same number as `cached_tokens` (#31 A9) |
//! | `timings` | `crow_expert_selections` | u64 | see the A8 table below |
//! | `timings` | `crow_expert_cold` | u64 | see the A8 table below |
//! | `timings` | `crow_ple_rows` | u64 | see the A8 table below |
//! | `timings` | `crow_ple_misses` | u64 | see the A8 table below |
//! | `timings` | `crow_layers` | int | see the A8 table below |
//!
//! - `cached_tokens` and `cache_n` are always PRESENT as an integer, never missing: a
//!   missing `cached_tokens` makes Crow fall back to `prompt_n` (`crow_core.py:14923`),
//!   which counts something else.
//! - A cold request has `cached_tokens` 0, and then every number is the A5 number.
//! - `prompt_tokens` is the whole rendered prompt either way, so `total_tokens` still is
//!   `prompt_tokens + completion_tokens` and Crow's context accounting is unchanged.
//! - `prompt_ms` and `prompt_per_second` belong to `prompt_n`, the PREFILLED tokens: a warm
//!   turn prefills few tokens, so a low `prompt_per_second` there is a small-batch effect.
//! - `prompt_ms` excludes the rollback, `Engine::reset_to_zero`, the snapshots and the
//!   tokenizer, so it is the same window `decode run` prints as `prefill done in X s`.
//! - `predicted_n` counts the token `prefill` returned, `predicted_ms` starts at the first
//!   `decode_step`; llama-server has the same offset and Crow's reader expects it.
//! - Every rate is 0.0 when its ms is 0, negative or not finite; no NaN can reach the wire
//!   (`serde_json` turns a non finite float into `null`).
//!
//! Two decode rates, on purpose (#27 doc):
//!
//! | place | formula | why |
//! |---|---|---|
//! | stderr `[chat] ... decode X ms, Y tok/s` | `(gen - 1) / decode_ms * 1000` | honest decode rate: `decode_ms` times the `decode_step` calls only, and the token `prefill` returned cost none of them |
//! | wire `timings.predicted_per_second` | `gen / decode_ms * 1000` | llama-server convention: `predicted_n` counts the prefill token too, and Crow's reader expects that ratio |
//!
//! - Same `decode_ms` in both, different numerator; the wire number is the higher one.
//! - The gap is one token, so it shrinks with the answer length (1 of 1024 = 0.1 %).
//! - Do not "fix" one to match the other: the log would lie, or Crow's reader would.
//!
//! Engine counters in the `timings` block (#30 A8, the Crow #54 rule):
//!
//! | key | type | unit | meaning | reset | incremented by |
//! |---|---|---|---|---|---|
//! | `crow_expert_selections` | u64 | selections | routed expert selections, SUMMED over the 48 layers (10 per token per layer) | never, cumulative since process start | `kernels.rs:2181` `router_top10` `atomicAdd(&counters[0], 10ull)`, launched at `gen.rs:1866-1868` |
//! | `crow_expert_cold` | u64 | selections | of those, the ones that hit a COLD (non resident) expert, summed over the 48 layers | never, cumulative since process start | `kernels.rs:2182` `router_top10` `atomicAdd(&counters[1], __popc(s_cold))`, same launch |
//! | `crow_ple_rows` | u64 | rows | PLE embedding rows requested (#16 hit-rate denominator) | never, cumulative since process start | `gen.rs:1041` `self.req += ngids.len()` |
//! | `crow_ple_misses` | u64 | rows | PLE rows that had to be filled from the container (row cache misses) | never, cumulative since process start | `gen.rs:1042` `self.miss += fill_rows.len()` |
//! | `crow_layers` | int | layers | `geo::LAYERS` (48), the divisor for a per-layer figure | constant | not a counter |
//!
//! - CUMULATIVE means exactly what Crow #54 means: NO reset exists, not per request, not anywhere.
//! - Request-local values are the DIFFERENCE of two consecutive blocks; the server never subtracts.
//! - A per-request reset would put two readers at odds over one state and silently break that
//!   difference, which is what Crow's tools are built on. That is why none is built in.
//! - The read site is `chat_stream`, after the last `decode_step` and before the final chunk.
//! - `Engine::drain_counters` (`gen.rs:3235`, `residency.rs:638`) is `cuda::dtoh_u64` of
//!   48 x 2 u64 = 768 bytes. It READS; it does NOT zero the device block. "drain" is the
//!   control-plane name, not a reset.
//! - `Ple::req` and `Ple::miss` (`gen.rs:202-203`) are host u64 that only ever grow.
//! - Cost per request: one 768 byte device to host copy, logged as `counter read X ms`.
//! - The same five numbers go to stderr as `[chat] counters (cumulative, never reset): ...`.
//! - They are written ONLY into `timings`, so they appear only when `timings_per_token` is true.
//!   Without the flag the final chunk is still exactly the A4 chunk.
//!
//! Counters that exist in the engine and are NOT in the block (names are not invented):
//!
//! | counter | where | why not |
//! |---|---|---|
//! | `Engine::sel_counts` `[48][512]` u64 | `gen.rs:362`, same `router_top10` launch | per-expert warm-up bookkeeping, 24576 values; a block is not a histogram |
//! | `Trickle::swaps` | `gen.rs:425` | cumulative over the process; #37 logs the REQUEST-LOCAL count as `crow_trickle_swaps` on the `[chat]` line instead |
//! | `Stage::n_tiles` | `gen.rs:445` | per launch value, ZEROED by `moe_plan` every layer; not cumulative |
//!
//! - There is NO cumulative staging counter (bytes staged, tiles staged) in the engine today.
//! - `decode run` derives `cold experts/token` from two `drain_counters` blocks
//!   (`bin/decode.rs:260-264`), the same difference this block hands to a client.
//!
//! Incremental detokenization:
//!
//! - The accumulated generated ids are decoded after every token.
//! - Only the new byte suffix is sent, so the text arrives exactly once.
//! - A tail that is not a whole character is HELD BACK, not sent.
//! - The byte level decoder renders an incomplete UTF-8 sequence as U+FFFD.
//! - No chunk therefore carries a replacement character from a split token.
//! - Assumed of `decode`: prefix stability across one more id; a violation is lossy, never a panic.
//! - Cost: one decode of the whole answer per token, microseconds against ms of GPU.
//!
//! One conversation at a time (M1), with the prefix cache (#31 A9, spec section 7):
//!
//! - ONE held conversation per process; a request that shares no prefix replaces it.
//! - `L` = longest common id prefix of the request and `Engine::history`, ids only.
//! - `P` = the newest snapshot position at or below `L`, and below the request length.
//! - `P` found: `PrefixCache::rollback` restores the state, `prefill` gets `ids[P..]`.
//! - No such snapshot: `Engine::reset_to_zero`, the slot dropped, the whole prompt prefilled.
//! - ONE snapshot per request, unconditional (M2b, robin 2026-09-10, #36): after the prompt.
//! - The after-answer snapshot of M1 is DROPPED: `decode_step` rows are not bit equal to
//!   prefill rows at the same positions (#31 A9 gate part 3, measured), so it was never a
//!   reuse candidate, and spec 7.5 says recompute where the state cannot be proven present.
//! - What is copied, what is not, the induction and the evidence: `engine/src/cache.rs`.
//! - The reset field list and its evidence: `engine/src/reset.rs`.
//! - A write error aborts the generation loop; the next request rolls back or resets first.
//! - `CROW_PREFIX_CACHE=0` allocates no slot and makes every request a cold start.
//!
//! Cache lines on stderr, one set per request (spec 7.9 asks for these numbers):
//!
//! | line | carries |
//! |---|---|
//! | `[cache] WARM\|COLD L .., P .., snapshots [..], reusable [..], prefill n of m tok, reset X ms` | the decision and the HtoD wall |
//! | `[cache] snapshot point 1 (after prompt) at pos .., DtoH X ms` | point 1 of spec 7.6 |
//!
//! - `reset X ms` is the ONE number the `[chat]` line also calls `reset`: the rollback of a
//!   warm request or the `reset_to_zero` of a cold one, whichever ran.
//! - With `CROW_PREFIX_CACHE=0` the first line prints `L n/a`, because `decide` returns
//!   before it computes `L`, and the snapshot line is not printed at all.
//!
//! - One stderr line per request: prompt tokens (cached and prefilled), generated tokens,
//!   prefill ms, decode ms.
//! - One stderr line per request with the generated ids, for the A4 and A9 identity gates.
//!
//! What the second turn of a real Crow conversation actually hits (#31 A9, measured):
//!
//! - The transcript is re-rendered through the chat template every turn.
//! - The re-rendered assistant message need not reproduce the generated ids exactly.
//! - The rollback lands on the prompt snapshot in either case; it is the only slot.
//! - That is the rule working, not a special case: the prompt prefill is still spared.
//!
//! The slot file across processes (#32 A10, `crow_core.py:2458`, `:2688`):
//!
//! | subject | the ONE place it is written down |
//! |---|---|
//! | the file layout, the payload order, every refusal, the save and restore ordering | `engine/src/slot.rs` module doc |
//! | the three answer documents and their fields | `slots_json`, `slot_saved_json`, `slot_restored_json` below |
//! | the `[slot]` stderr lines | the `eprintln!` calls in `slot_route` below |
//!
//! What only THIS file can say, because it is the wire and not the format:
//!
//! - Only `n_saved` and `n_restored` are contractual; Crow reads nothing else of these bodies.
//! - `n_saved` and `n_restored` are the PREFILL CLEAN position, so `SLOT_PROMPT`, never the answer.
//! - Crow withdraws the warm-cache claim when `n_restored` disagrees with the saved `n_saved`,
//!   so equality of the two numbers is the contract, not the status code (`crow_core.py:2694`).
//! - `n_prompt_tokens` of `GET /slots` is the same number, 0 while nothing is held.
//! - `GET /slots` is read by Crow's tools only (`tools/measure-slot-restart.ps1:87`,
//!   `tools/probe-slot-persistence.py:152`), which take element 0 of the array.
//! - A restore fills `SLOT_PROMPT` and sets `pos`, `done_blocks`, `history`.
//! - So the NEXT chat request is an ordinary A9 warm turn: `L >= pos`, `P = pos`, one rollback.
//! - There is no second warm path; the tested one is the only one.
//! - A refusal answers 4xx with a JSON error body and leaves the engine exactly as it was.
//! - `--slot-save-path` must name an EXISTING directory; a typo refuses the BOOT, not the save.

use crow_nest_engine::cache::{PrefixCache, SLOTS, SLOT_PROMPT};
use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::gen::{DevSampler, Engine};
use crow_nest_engine::geo::{apply_adapt_policy, Adapt, Config, CONTEXT_FLOOR, LAYERS};
use crow_nest_engine::sample::{Sampler, EOS_IDS};
use crow_nest_engine::slot;
use crow_nest_engine::toolcall::{Emit, ToolStream, TOOL_OPEN};
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PORT: u16 = 8099;
const DEFAULT_CNQ: &str = "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq";
const DEFAULT_HOTSETS: &str = "decode_out/hotsets-M-longctx2100-n160.json";
/// pinned for the process (M1): every request prefills at chunk 4096 (the
/// #10b stage-1 diet lifted the planner wall; 2048 halved the serve prefill)
const SERVE_CHUNK: usize = 4096;
/// read and write timeout per connection, so one stalled client cannot hold the loop
const IO_TIMEOUT_SECS: u64 = 10;
/// request line plus headers, 64 KiB total; over it the answer is 431
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// declared `Content-Length`, 16 MiB; over it the answer is 413
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// `max_tokens` when the request carries none
const DEFAULT_MAX_TOKENS: usize = 1024;
/// ceiling for `max_tokens`, so one request cannot hold the process forever
const MAX_MAX_TOKENS: usize = 32768;
/// #28: `top_p` when a sampled request carries none (data sheet, `generation_config.json`)
const DEFAULT_TOP_P: f32 = 0.8;
/// #28: `top_k` when a sampled request carries none (data sheet)
const DEFAULT_TOP_K: usize = 20;
/// #28: `presence_penalty` when a sampled request carries none (data sheet)
const DEFAULT_PRESENCE: f32 = 1.5;
/// #28: RNG seed when the request carries none; fixed, so warm equals cold (M1)
const DEFAULT_SEED: u64 = 0;
/// the last line of every stream
const SSE_DONE: &str = "data: [DONE]

";

// ---------------------------------------------------------------- pure parts

/// the route a `(method, path)` pair dispatches to
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Route {
    /// `GET /health`
    Health,
    /// `GET /props`
    Props,
    /// `POST /v1/chat/completions` (#26 A4)
    Chat,
    /// `GET /slots` (#32 A10), the one slot of this process as an array of one
    Slots,
    /// `POST /slots/0?action=save|restore` (#32 A10)
    Slot0,
    /// everything else, a wrong method on a known path included
    NotFound,
}

/// the whole dispatch table, pure so the test can drive it
fn route(method: &str, path: &str) -> Route {
    match (method, path) {
        ("GET", "/health") => Route::Health,
        ("GET", "/props") => Route::Props,
        ("POST", "/v1/chat/completions") => Route::Chat,
        ("GET", "/slots") => Route::Slots,
        ("POST", "/slots/0") => Route::Slot0,
        _ => Route::NotFound,
    }
}

/// what one connection sent, as far as the head reader could tell
#[derive(Debug, PartialEq, Eq)]
enum Head {
    /// the client closed without sending a byte; no response is written
    Empty,
    /// there was a request line, but it is not HTTP; the answer is 400
    Bad,
    /// the head passed `MAX_HEAD_BYTES`; the answer is 431
    HeadTooLarge,
    /// the declared body passed `MAX_BODY_BYTES`; the answer is 413
    BodyTooLarge(usize),
    /// the head asks for chunked transfer encoding; the answer is 501
    Chunked,
    /// a well formed request; `body` is drained and kept for A3 and A4
    Req { method: String, target: String, body: Vec<u8> },
}

/// what the command line of a `serve` run says
#[derive(Debug, Clone, PartialEq, Eq)]
struct ServeArgs {
    /// `--port <n>`, default `DEFAULT_PORT`
    port: u16,
    /// #32 A10: `--slot-save-path <dir>`; without it `/slots/0` refuses both actions
    slot_save_path: Option<String>,
}

/// - the whole command line of a `serve` run (no subcommand), argv[0] skipped
/// - `--flag <value>` and `--flag=<value>` are the same thing
/// - an unknown flag, a missing value and an empty value are usage errors, never a default
/// - `Err` carries the message for the operator
fn parse_args(args: &[String]) -> Result<ServeArgs, String> {
    const USAGE: &str = "usage: serve [--port <n>] [--slot-save-path <dir>]";
    let mut i = 1;
    let mut out = ServeArgs { port: DEFAULT_PORT, slot_save_path: None };
    while i < args.len() {
        let a = args[i].as_str();
        let (flag, inline) = match a.split_once('=') {
            Some((f, v)) => (f, Some(v.to_string())),
            None => (a, None),
        };
        let val = match inline {
            Some(v) => v,
            None => {
                i += 1;
                args.get(i).cloned().ok_or_else(|| format!("{flag} needs a value ({USAGE})"))?
            }
        };
        match flag {
            "--port" => out.port = val.parse::<u16>().map_err(|_| format!("bad port {val:?}"))?,
            "--slot-save-path" => {
                if val.is_empty() {
                    return Err(format!("--slot-save-path needs a directory ({USAGE})"));
                }
                out.slot_save_path = Some(val);
            }
            _ => return Err(format!("unknown argument {a:?} ({USAGE})")),
        }
        i += 1;
    }
    Ok(out)
}

/// - the port half of `parse_args`, the view the #26 A4 argument test drives
#[cfg(test)]
fn parse_port(args: &[String]) -> Result<u16, String> {
    parse_args(args).map(|a| a.port)
}

/// - `--slot-save-path <dir>` must name an EXISTING directory, checked at boot (#32 review)
/// - never created here: llama-server's convention is a directory the operator already made
/// - `Err` carries the stderr line, so a typo costs a second, not a whole conversation
fn check_slot_save_path(dir: &str) -> Result<(), String> {
    if std::path::Path::new(dir).is_dir() {
        return Ok(());
    }
    Err(format!(
        "--slot-save-path {dir:?} is not a directory; create it first (llama-server takes an \
         existing directory too, and this server never creates one)"
    ))
}

// ------------------------------------------------- tokenize subcommand (A3)

/// what `serve tokenize ...` was asked to do
#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    /// every prompt of `file` through the chat template, ids written to `out`
    File { chat: bool, file: String, out: String },
    /// one text, id list on stdout
    Text { chat: bool, text: String },
}

/// - argument vector AFTER the `tokenize` word
/// - exactly one of `--chat` and `--raw`
/// - `--file` needs `--out`; `--file` and `--text` are exclusive
fn parse_tokenize(rest: &[String]) -> Result<Tok, String> {
    const USAGE: &str = "usage: serve tokenize (--chat|--raw) (--file <prompts.json> --out <ids.json> | --text <text>)";
    let mut chat: Option<bool> = None;
    let (mut file, mut out, mut text) = (None, None, None);
    let mut i = 0;
    while i < rest.len() {
        let a = rest[i].as_str();
        let slot: &mut Option<String> = match a {
            "--chat" | "--raw" => {
                if chat.replace(a == "--chat").is_some() {
                    return Err(format!("mode given twice ({USAGE})"));
                }
                i += 1;
                continue;
            }
            "--file" => &mut file,
            "--out" => &mut out,
            "--text" => &mut text,
            other => return Err(format!("unknown argument {other:?} ({USAGE})")),
        };
        i += 1;
        let v = rest.get(i).ok_or_else(|| format!("{a} needs a value ({USAGE})"))?;
        if slot.replace(v.clone()).is_some() {
            return Err(format!("{a} given twice ({USAGE})"));
        }
        i += 1;
    }
    let chat = chat.ok_or_else(|| format!("one of --chat and --raw is required ({USAGE})"))?;
    match (file, out, text) {
        (Some(_), _, Some(_)) => Err(format!("--file and --text are exclusive ({USAGE})")),
        (Some(f), Some(o), None) => Ok(Tok::File { chat, file: f, out: o }),
        (Some(_), None, None) => Err(format!("--file needs --out ({USAGE})")),
        (None, _, Some(t)) => Ok(Tok::Text { chat, text: t }),
        (None, _, None) => Err(format!("one of --file and --text is required ({USAGE})")),
    }
}

/// - `{id: text}` and `[{"id": ..., "text": ...}]` both give the same pairs
/// - array shape: the array order
/// - object shape: the order the keys stand in the file (`serde_json` feature `preserve_order`)
/// - without that feature the object shape would come back sorted by key
fn prompts_from_json(v: &serde_json::Value) -> Result<Vec<(String, String)>, String> {
    if let Some(a) = v.as_array() {
        let mut out = Vec::with_capacity(a.len());
        for (i, p) in a.iter().enumerate() {
            let id = p["id"].as_str().ok_or_else(|| format!("entry {i} has no string id"))?;
            let text = p["text"].as_str().ok_or_else(|| format!("entry {id} has no string text"))?;
            out.push((id.to_string(), text.to_string()));
        }
        return Ok(out);
    }
    if let Some(m) = v.as_object() {
        let mut out = Vec::with_capacity(m.len());
        for (k, t) in m {
            let text = t.as_str().ok_or_else(|| format!("entry {k} is not a string"))?;
            out.push((k.clone(), text.to_string()));
        }
        return Ok(out);
    }
    Err("prompts file is neither an object nor an array".to_string())
}

/// `serve tokenize ...`, the gate arm; exit code is the return value
fn tokenize_main(rest: &[String]) -> i32 {
    let cmd = match parse_tokenize(rest) {
        Ok(c) => c,
        Err(e) => {
            eprintln!("[tokenize] {e}");
            return 2;
        }
    };
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => {
            eprintln!("[tokenize] {e}");
            return 3;
        }
    };
    let (tp, cp) = tk.paths();
    eprintln!("[tokenize] tokenizer {tp}");
    eprintln!("[tokenize] chat template {cp}");

    let ids_of = |chat: bool, text: &str| -> Result<Vec<u32>, String> {
        if chat {
            tk.encode_chat_user(text)
        } else {
            tk.encode_raw(text)
        }
    };

    match cmd {
        Tok::Text { chat, text } => match ids_of(chat, &text) {
            Ok(ids) => {
                println!("{}", serde_json::json!(ids));
                eprintln!("[tokenize] {} tokens", ids.len());
                0
            }
            Err(e) => {
                eprintln!("[tokenize] {e}");
                4
            }
        },
        Tok::File { chat, file, out } => {
            let raw = match std::fs::read(&file) {
                Ok(r) => r,
                Err(e) => {
                    eprintln!("[tokenize] cannot read {file}: {e}");
                    return 4;
                }
            };
            let doc: serde_json::Value = match serde_json::from_slice(&raw) {
                Ok(d) => d,
                Err(e) => {
                    eprintln!("[tokenize] {file} is not JSON: {e}");
                    return 4;
                }
            };
            let prompts = match prompts_from_json(&doc) {
                Ok(p) => p,
                Err(e) => {
                    eprintln!("[tokenize] {file}: {e}");
                    return 4;
                }
            };
            let mut map = serde_json::Map::new();
            for (id, text) in &prompts {
                match ids_of(chat, text) {
                    Ok(ids) => {
                        eprintln!("[tokenize] {id} chars {} tokens {}", text.chars().count(), ids.len());
                        map.insert(id.clone(), serde_json::json!(ids));
                    }
                    Err(e) => {
                        eprintln!("[tokenize] {id}: {e}");
                        return 4;
                    }
                }
            }
            let text = serde_json::Value::Object(map).to_string();
            if let Err(e) = std::fs::write(&out, text) {
                eprintln!("[tokenize] cannot write {out}: {e}");
                return 4;
            }
            eprintln!("[tokenize] {} prompts -> {out}", prompts.len());
            0
        }
    }
}

/// method + request target of an HTTP request line, `None` when it is not one.
fn parse_request_line(line: &str) -> Option<(String, String)> {
    let mut it = line.trim_end_matches(['\r', '\n']).split_whitespace();
    let method = it.next()?.to_string();
    let target = it.next()?.to_string();
    let version = it.next()?;
    if !version.starts_with("HTTP/") || !target.starts_with('/') {
        return None;
    }
    Some((method, target))
}

/// value of one header line when it carries `name`, matched case insensitively
fn header_value<'a>(line: &'a str, name: &str) -> Option<&'a str> {
    let (k, v) = line.split_once(':')?;
    if k.trim().eq_ignore_ascii_case(name) {
        Some(v.trim())
    } else {
        None
    }
}

/// request target without query string or fragment
fn route_path(target: &str) -> &str {
    let cut = target.find(['?', '#']).unwrap_or(target.len());
    let p = &target[..cut];
    // "/props/" and "/props" are the same route
    if p.len() > 1 { p.trim_end_matches('/') } else { p }
}

/// container file stem, the name Crow shows (`fetch_model_name`)
fn model_name(model_path: &str) -> String {
    let base = model_path.replace('\\', "/");
    let base = base.rsplit('/').next().unwrap_or(&base);
    base.strip_suffix(".cnq").unwrap_or(base).to_string()
}

/// the `/props` document; `n_ctx` comes from the loaded engine, not a constant.
/// #VIT: `modalities.vision` reports the switch, so Crow's `refuse_images`
/// (crow_core.py:1429) sends or refuses for real — `true` with the tower
/// loaded (CROW_VIT unset/1), `false` with CROW_VIT=0 (BLIND_SERVER_HINT).
fn props_json(model_path: &str, n_ctx: usize, prompt_chunk: usize, vision: bool) -> serde_json::Value {
    serde_json::json!({
        "model_path": model_path,
        "model": model_name(model_path),
        "n_ctx": n_ctx,
        "default_generation_settings": { "n_ctx": n_ctx },
        "modalities": { "vision": vision },
        "prompt_chunk": prompt_chunk,
        "build": "crow-nest-engine 0.1.0",
    })
}

/// - value of one query parameter of a request target, `None` when it is absent
/// - #32 A10: Crow names the action in the query (`crow_core.py:2458`), not in the body
/// - no percent decoding: `save` and `restore` are the only values this server reads
fn query_param<'a>(target: &'a str, key: &str) -> Option<&'a str> {
    let q = &target[target.find('?')? + 1..];
    let q = &q[..q.find('#').unwrap_or(q.len())];
    q.split('&').find_map(|pair| pair.strip_prefix(key)?.strip_prefix('='))
}

/// - `GET /slots` (#32 A10), the shape Crow's measuring tools read
/// - one element, because this process holds ONE conversation (M1)
/// - `n_prompt_tokens` is the held PREFILL CLEAN position, 0 when none is held
fn slots_json(n_ctx: usize, n_prompt_tokens: usize) -> serde_json::Value {
    serde_json::json!([{
        "id": 0,
        "n_ctx": n_ctx,
        "n_prompt_tokens": n_prompt_tokens,
        "is_processing": false,
    }])
}

/// the answer of `POST /slots/0?action=save`; only `n_saved` is contractual
fn slot_saved_json(filename: &str, s: &crow_nest_engine::slot::Saved) -> serde_json::Value {
    serde_json::json!({
        "id_slot": 0,
        "filename": filename,
        "n_saved": s.n_saved,
        "n_written": s.n_written,
        "timings": { "save_ms": s.ms },
    })
}

/// the answer of `POST /slots/0?action=restore`; only `n_restored` is contractual
fn slot_restored_json(filename: &str, r: &crow_nest_engine::slot::Restored) -> serde_json::Value {
    serde_json::json!({
        "id_slot": 0,
        "filename": filename,
        "n_restored": r.n_restored,
        "n_read": r.n_read,
        "timings": { "restore_ms": r.ms },
    })
}

/// - the `filename` of a `/slots/0` body, sanitized to a bare file name
/// - `Err` carries the message the 400 body shows
fn slot_filename(body: &[u8]) -> Result<String, String> {
    let doc: serde_json::Value = serde_json::from_slice(body)
        .map_err(|e| format!("the /slots/0 body is not a JSON object: {e}"))?;
    let name = doc
        .get("filename")
        .and_then(|v| v.as_str())
        .ok_or_else(|| "the /slots/0 body needs a string \"filename\"".to_string())?;
    slot::sanitize_filename(name).map(|s| s.to_string())
}

fn not_found_json(path: &str) -> serde_json::Value {
    serde_json::json!({
        "error": { "code": 404, "message": format!("no route {path} (serve answers GET /health, GET /props, POST /v1/chat/completions, GET /slots, POST /slots/0)") }
    })
}

/// flat error document for 400, 413, 431 and 501
fn error_json(msg: &str) -> serde_json::Value {
    serde_json::json!({ "error": msg })
}

// --------------------------------------------- chat completions (#26 A4)

/// the fields of a `POST /v1/chat/completions` body that A4 acts on
#[derive(Debug, Clone, PartialEq)]
struct ChatReq {
    /// echoed into every chunk's `model`
    model: String,
    /// the OpenAI message array, handed to `render_chat` unchanged
    messages: serde_json::Value,
    /// #29 A7: the OpenAI function tool array, a template variable; `None` renders no tool block
    tools: Option<serde_json::Value>,
    /// `false` (or absent) answers 501 in A4
    stream: bool,
    /// generation budget, `DEFAULT_MAX_TOKENS` when absent
    max_tokens: usize,
    /// `chat_template_kwargs.enable_thinking`, a template variable
    enable_thinking: bool,
    /// `stream_options.include_usage`: `usage` on the final chunk (#27 A5)
    include_usage: bool,
    /// `timings_per_token`: `timings` on the final chunk (#27 A5)
    timings_per_token: bool,
    /// #28: `<= 0` (absent included) is greedy, `> 0` samples
    temperature: f32,
    /// #28: nucleus mass, `DEFAULT_TOP_P` when absent
    top_p: f32,
    /// #28: candidates kept, `DEFAULT_TOP_K` when absent
    top_k: usize,
    /// #28: presence penalty, `DEFAULT_PRESENCE` when absent
    presence_penalty: f32,
    /// #28: RNG seed of this request, `DEFAULT_SEED` when absent
    seed: u64,
    /// #28: parsed, ignored, logged; the device sampler has no min_p
    min_p: f32,
    /// #VIT: the `image_url` data URLs of the content blocks, in message order.
    /// This is the exact wire form Crow sends (crow_core.py `image_part`):
    /// `{"type":"image_url","image_url":{"url":"data:<mime>;base64,..."}}`.
    images: Vec<String>,
}

/// - a number field of the sampling profile: absent or `null` gives `d`, a non number is a 400
/// - `top_k` and `seed` have their own readers below, they are not floats
fn num_field(obj: &serde_json::Map<String, serde_json::Value>, key: &str, d: f32) -> Result<f32, String> {
    match obj.get(key) {
        None | Some(serde_json::Value::Null) => Ok(d),
        Some(v) => v
            .as_f64()
            .map(|x| x as f32)
            .ok_or_else(|| format!("{key} is not a number")),
    }
}

/// - the request body, as Crow sends it (`crow_core.py:4672-4700`)
/// - unknown fields are accepted and ignored, as llama-server does
/// - `tools` falls under that rule in A4; `min_p` under it in A6
/// - the sampling fields (#28) are STRICT on type and lenient on absence
/// - `Err` carries the message for the 400 body
fn parse_chat(body: &[u8]) -> Result<ChatReq, String> {
    let doc: serde_json::Value =
        serde_json::from_slice(body).map_err(|e| format!("body is not JSON: {e}"))?;
    let obj = doc
        .as_object()
        .ok_or_else(|| "body is not a JSON object".to_string())?;
    let messages = obj
        .get("messages")
        .ok_or_else(|| "no messages in the body".to_string())?;
    let arr = messages
        .as_array()
        .ok_or_else(|| "messages is not an array".to_string())?;
    if arr.is_empty() {
        return Err("messages is empty".to_string());
    }
    for (i, m) in arr.iter().enumerate() {
        if !m.get("role").map(|r| r.is_string()).unwrap_or(false) {
            return Err(format!("message {i} has no string role"));
        }
    }
    // #VIT: collect the image content blocks in message order. The template
    // renders each of them as `<|vision_start|><|image_pad|><|vision_end|>`,
    // so the rendered ids carry the pads in exactly this order.
    let mut images = Vec::new();
    for (i, m) in arr.iter().enumerate() {
        if let Some(items) = m.get("content").and_then(|c| c.as_array()) {
            for item in items {
                let is_img = item.get("type").and_then(|t| t.as_str()) == Some("image_url")
                    || item.get("image_url").is_some();
                if !is_img {
                    continue;
                }
                let url = item
                    .get("image_url")
                    .and_then(|u| u.get("url"))
                    .and_then(|u| u.as_str())
                    .ok_or_else(|| format!("message {i} carries an image_url block without image_url.url"))?;
                images.push(url.to_string());
            }
        }
    }
    // #29 A7: `tools` is a template variable now. Absent and `null` render the A4 prompt;
    // an empty array renders the same, because the template's `if tools` is falsy on it.
    let tools = match obj.get("tools") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let a = v
                .as_array()
                .ok_or_else(|| "tools is not an array".to_string())?;
            for (i, t) in a.iter().enumerate() {
                if !t.get("function").map(|f| f.is_object()).unwrap_or(false) {
                    return Err(format!("tool {i} has no function object"));
                }
                if !t
                    .get("function")
                    .and_then(|f| f.get("name"))
                    .map(|n| n.is_string())
                    .unwrap_or(false)
                {
                    return Err(format!("tool {i} has no string function.name"));
                }
            }
            Some(v.clone())
        }
    };
    let stream = match obj.get("stream") {
        None | Some(serde_json::Value::Null) => false,
        Some(v) => v
            .as_bool()
            .ok_or_else(|| "stream is not a boolean".to_string())?,
    };
    let max_tokens = match obj.get("max_tokens") {
        None | Some(serde_json::Value::Null) => DEFAULT_MAX_TOKENS,
        Some(v) => {
            let n = v
                .as_u64()
                .ok_or_else(|| "max_tokens is not a positive integer".to_string())?;
            if n == 0 {
                return Err("max_tokens is 0".to_string());
            }
            (n as usize).min(MAX_MAX_TOKENS)
        }
    };
    let enable_thinking = obj
        .get("chat_template_kwargs")
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let model = obj
        .get("model")
        .and_then(|v| v.as_str())
        .unwrap_or("crow-nest")
        .to_string();
    // #27 A5: lenient on purpose, a wrong type is "off", never a 400. Crow always sends both
    // as `true` (`crow_core.py:4684-4688`); every other client just gets the A4 stream.
    let include_usage = obj
        .get("stream_options")
        .and_then(|o| o.get("include_usage"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let timings_per_token = obj
        .get("timings_per_token")
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    // #28 A6: the sampling profile. Absent stays the data sheet, a wrong type is a 400 (the
    // caller asked for a draw and must know it did not get the one it named).
    let temperature = num_field(obj, "temperature", 0.0)?;
    let top_p = num_field(obj, "top_p", DEFAULT_TOP_P)?;
    let presence_penalty = num_field(obj, "presence_penalty", DEFAULT_PRESENCE)?;
    let min_p = num_field(obj, "min_p", 0.0)?;
    let top_k = match obj.get("top_k") {
        None | Some(serde_json::Value::Null) => DEFAULT_TOP_K,
        Some(v) => v
            .as_u64()
            .ok_or_else(|| "top_k is not a non negative integer".to_string())? as usize,
    };
    // llama-server takes -1 as "pick a seed"; M1 wants determinism, so a negative seed is
    // used as its unsigned bit pattern and nothing here ever draws a seed of its own
    let seed = match obj.get("seed") {
        None | Some(serde_json::Value::Null) => DEFAULT_SEED,
        Some(v) => v
            .as_u64()
            .or_else(|| v.as_i64().map(|x| x as u64))
            .ok_or_else(|| "seed is not an integer".to_string())?,
    };
    Ok(ChatReq {
        model,
        messages: messages.clone(),
        tools,
        stream,
        max_tokens,
        enable_thinking,
        include_usage,
        timings_per_token,
        temperature,
        top_p,
        top_k,
        presence_penalty,
        seed,
        min_p,
        images,
    })
}

/// - decode one `data:<mime>;base64,<payload>` URL into (mime, bytes)
/// - Crow sends exactly this shape (crow_core.py `image_part`); anything else is a 400
fn decode_data_url(url: &str) -> Result<(&str, Vec<u8>), String> {
    let rest = url
        .strip_prefix("data:")
        .ok_or_else(|| "image_url.url is not a data: URL (Crow sends data URLs only)".to_string())?;
    let (mime, payload) = rest
        .split_once(";base64,")
        .ok_or_else(|| "image_url.url carries no ;base64, payload".to_string())?;
    const B64: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut table = [255u8; 256];
    for (i, &c) in B64.iter().enumerate() {
        table[c as usize] = i as u8;
    }
    let mut out = Vec::with_capacity(payload.len() / 4 * 3);
    let mut acc: u32 = 0;
    let mut nbits = 0u32;
    for ch in payload.bytes() {
        match ch {
            b'\r' | b'\n' | b' ' => continue,
            b'=' => break,
            _ => {
                let v = table[ch as usize];
                if v == 255 {
                    return Err("image_url.url payload is not valid base64".into());
                }
                acc = (acc << 6) | v as u32;
                nbits += 6;
                if nbits >= 8 {
                    nbits -= 8;
                    out.push((acc >> nbits) as u8);
                }
            }
        }
    }
    if out.is_empty() {
        return Err("image_url.url payload decodes to zero bytes".into());
    }
    Ok((mime, out))
}

/// - the `Sampler` this request asks for, or `None` for the greedy A4 path
/// - `None` is the whole greedy contract: no sampler is built, none is armed
/// - every field the request left out comes from `Sampler::new` (data sheet)
fn sampler_from(req: &ChatReq) -> Option<Sampler> {
    if !(req.temperature > 0.0) {
        return None;
    }
    let mut s = Sampler::new(req.seed);
    s.temperature = req.temperature;
    s.top_p = req.top_p;
    s.top_k = req.top_k;
    s.presence_penalty = req.presence_penalty;
    Some(s)
}

/// - what one served request counted and how long each phase took
/// - the only input of the `usage` and `timings` builders, so the test drives them directly
/// - `prompt_ms` is the `Engine::prefill` wall, `predicted_ms` the decode loop wall
/// - the four `*_total` fields are ENGINE counters, CUMULATIVE since process start (#30 A8)
#[derive(Debug, Clone, Copy, PartialEq)]
struct Timing {
    /// prompt ids actually PREFILLED this request, `rendered prompt - cached_n` (#31 A9)
    prompt_n: usize,
    /// #31 A9: prompt ids reused from the held state, the reuse point `P` of spec 7.4
    cached_n: usize,
    /// generated ids, the token `prefill` returned included
    predicted_n: usize,
    /// wall of the prefill call, in ms
    prompt_ms: f64,
    /// wall from the first `decode_step` to the last, in ms
    predicted_ms: f64,
    /// #30 A8: routed expert selections, summed over the 48 layers, cumulative
    selections_total: u64,
    /// #30 A8: selections that hit a COLD (non resident) expert, cumulative
    cold_total: u64,
    /// #30 A8: PLE embedding rows requested, cumulative
    ple_rows_total: u64,
    /// #30 A8: PLE rows filled from the container (row cache misses), cumulative
    ple_miss_total: u64,
}

/// - tokens per second out of a count and a wall time in ms
/// - 0.0 for a wall of 0, a negative wall or a non finite wall, so no NaN reaches the wire
fn per_second(n: usize, ms: f64) -> f64 {
    if !(ms > 0.0) || !ms.is_finite() {
        return 0.0;
    }
    n as f64 * 1000.0 / ms
}

/// - ms per token out of a wall time in ms and a count
/// - 0.0 for a count of 0 or a non finite wall, the mirror of `per_second`
fn per_token_ms(n: usize, ms: f64) -> f64 {
    if n == 0 || !ms.is_finite() {
        return 0.0;
    }
    ms / n as f64
}

/// microsecond resolution, so the gate log carries a number a person can read
fn round3(x: f64) -> f64 {
    if !x.is_finite() {
        return 0.0;
    }
    (x * 1e3).round() / 1e3
}

/// - the OpenAI `usage` object, as llama-server sends it
/// - `prompt_tokens` is the WHOLE rendered prompt, the cached part included
/// - `cached_tokens` is `P`, the tokens the prefix cache reused (#31 A9, spec 7.4)
/// - a cold request carries `cached_n` 0, so this is the A5 object unchanged
fn usage_json(t: &Timing) -> serde_json::Value {
    serde_json::json!({
        "prompt_tokens": t.cached_n + t.prompt_n,
        "completion_tokens": t.predicted_n,
        "total_tokens": t.cached_n + t.prompt_n + t.predicted_n,
        "prompt_tokens_details": { "cached_tokens": t.cached_n },
    })
}

/// - the llama.cpp `timings` object (`crow_core.py:4999-5018` reads six of these fields)
/// - `prompt_n` is what was PREFILLED, `cache_n` what was reused from the cache (#31 A9)
/// - the `crow_` keys are the engine counters (#30 A8): u64, CUMULATIVE, never reset
/// - the prefix keeps them out of llama-server's key space, so no reader collides
fn timings_json(t: &Timing) -> serde_json::Value {
    serde_json::json!({
        "prompt_n": t.prompt_n,
        "prompt_ms": round3(t.prompt_ms),
        "prompt_per_second": round3(per_second(t.prompt_n, t.prompt_ms)),
        "prompt_per_token_ms": round3(per_token_ms(t.prompt_n, t.prompt_ms)),
        "predicted_n": t.predicted_n,
        "predicted_ms": round3(t.predicted_ms),
        "predicted_per_second": round3(per_second(t.predicted_n, t.predicted_ms)),
        "predicted_per_token_ms": round3(per_token_ms(t.predicted_n, t.predicted_ms)),
        "cache_n": t.cached_n,
        "crow_expert_selections": t.selections_total,
        "crow_expert_cold": t.cold_total,
        "crow_ple_rows": t.ple_rows_total,
        "crow_ple_misses": t.ple_miss_total,
        "crow_layers": LAYERS,
    })
}

/// one `chat.completion.chunk`, the only object shape this endpoint streams
fn chunk(
    id: &str,
    created: u64,
    model: &str,
    delta: serde_json::Value,
    finish: Option<&str>,
) -> serde_json::Value {
    serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "delta": delta,
            "finish_reason": match finish {
                Some(r) => serde_json::Value::String(r.to_string()),
                None => serde_json::Value::Null,
            },
        }],
    })
}

/// first chunk: the role, no content (`crow_core.py:4877` reads content only)
fn chunk_role(id: &str, created: u64, model: &str) -> serde_json::Value {
    chunk(id, created, model, serde_json::json!({ "role": "assistant" }), None)
}

/// a content delta chunk, one per emitted text piece
fn chunk_content(id: &str, created: u64, model: &str, text: &str) -> serde_json::Value {
    chunk(id, created, model, serde_json::json!({ "content": text }), None)
}

/// - #29 A7: the FIRST fragment of one tool call, the only one carrying `id` and `name`
/// - `arguments` is the empty string here, as llama-server and OpenAI send it
/// - Crow keeps `id` and `name` because it tests them for truth (`crow_core.py:4869-4874`)
fn chunk_tool_open(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    call_id: &str,
    name: &str,
) -> serde_json::Value {
    let delta = serde_json::json!({
        "tool_calls": [{
            "index": index,
            "id": call_id,
            "type": "function",
            "function": { "name": name, "arguments": "" },
        }],
    });
    chunk(id, created, model, delta, None)
}

/// - #29 A7: one `arguments` fragment of the call at `index`
/// - NO `id` and NO `name`: an empty string here would erase what the open chunk said
/// - the concatenation of every fragment of one index is the arguments JSON object text
fn chunk_tool_args(
    id: &str,
    created: u64,
    model: &str,
    index: usize,
    args: &str,
) -> serde_json::Value {
    let delta = serde_json::json!({
        "tool_calls": [{
            "index": index,
            "function": { "arguments": args },
        }],
    });
    chunk(id, created, model, delta, None)
}

/// - last chunk before `[DONE]`: empty delta, the finish reason
/// - `usage` rides along when `include_usage`, `timings` when `timings_per_token` (#27 A5)
/// - neither flag: the object is exactly the A4 chunk, no empty placeholders
/// - pure: the whole final chunk contract is one function the test can drive
fn chunk_finish(
    id: &str,
    created: u64,
    model: &str,
    reason: &str,
    t: &Timing,
    include_usage: bool,
    timings_per_token: bool,
) -> serde_json::Value {
    let mut doc = chunk(id, created, model, serde_json::json!({}), Some(reason));
    if let Some(obj) = doc.as_object_mut() {
        if include_usage {
            obj.insert("usage".to_string(), usage_json(t));
        }
        if timings_per_token {
            obj.insert("timings".to_string(), timings_json(t));
        }
    }
    doc
}

/// one SSE event: `data: <compact json>` plus the blank line that ends it
fn sse_frame(doc: &serde_json::Value) -> String {
    format!("data: {doc}\n\n")
}

/// - the generation budget one request may actually spend
/// - `None` means the prompt alone does not fit `n_ctx`: the answer is 413
/// - `Some(n)`: `max_tokens`, clamped to the free context and to `MAX_MAX_TOKENS`
/// - a budget that does not fit is clamped, never refused (llama-server does the same)
/// - pure: no engine, no socket, so the test drives the arithmetic directly
fn clamped_max_tokens(prompt_ids: usize, max_tokens: usize, n_ctx: usize) -> Option<usize> {
    if prompt_ids >= n_ctx {
        return None;
    }
    Some(max_tokens.min(n_ctx - prompt_ids).min(MAX_MAX_TOKENS))
}

/// - `full` is `decode` over EVERY generated id so far
/// - `emitted` is how many BYTES of `full` already left as content
/// - `None` holds the delta back until the tail is a whole character
/// - the byte level decoder renders an incomplete UTF-8 sequence as U+FFFD
/// - so no chunk this returns ends in a replacement character
/// - ASSUMED of `decode`: prefix stability, `decode(ids[..k])` is a byte prefix of `decode(ids[..k+1])`
/// - a violation is LOSSY, never a panic: the changed bytes below `emitted` are never re-sent,
///   and the `is_char_boundary` guard keeps the slice legal for any input
fn next_delta(full: &str, emitted: usize) -> Option<&str> {
    if full.len() <= emitted || !full.is_char_boundary(emitted) {
        return None;
    }
    if full.ends_with(char::REPLACEMENT_CHARACTER) {
        return None;
    }
    Some(&full[emitted..])
}

/// - #29 A7: Crow sends `tool_calls[].function.arguments` as a JSON STRING (`crow_core.py:3564`)
/// - the chat template iterates it with `|items`, which needs a MAPPING
/// - measured 2026-09-09: the Python oracle raises `TypeError: Can only get item pairs from a
///   mapping` on the string form, so the object form is the only one that renders at all
/// - this converts the string to the object it encodes and leaves everything else untouched
/// - a string that is not a JSON object stays as it is, so the render fails loudly (400)
fn normalize_messages(messages: &serde_json::Value) -> serde_json::Value {
    let mut doc = messages.clone();
    let arr = match doc.as_array_mut() {
        Some(a) => a,
        None => return doc,
    };
    for m in arr.iter_mut() {
        let calls = match m.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
            Some(c) => c,
            None => continue,
        };
        for c in calls.iter_mut() {
            let args = match c.get_mut("function").and_then(|f| f.get_mut("arguments")) {
                Some(a) => a,
                None => continue,
            };
            let parsed = match args.as_str() {
                Some(s) => serde_json::from_str::<serde_json::Value>(s).ok(),
                None => None,
            };
            if let Some(v) = parsed {
                if v.is_object() {
                    *args = v;
                }
            }
        }
    }
    doc
}

/// write one SSE frame and flush it; `false` means the client is gone
fn sse_send<W: Write>(w: &mut W, text: &str) -> bool {
    match w.write_all(text.as_bytes()).and_then(|_| w.flush()) {
        Ok(()) => true,
        Err(e) => {
            eprintln!("[chat] write failed, aborting the generation: {e}");
            false
        }
    }
}

/// - #39 B3a: where the per token side effects of ONE generation go
/// - the SSE writer is one implementation, the collector behind the non streaming document
///   is the other; `chat_generate` stays the only generation loop in this file
/// - every method returns `false` for "the client is gone, stop the loop"
trait ChatSink {
    /// before the first token: the role delta of the stream, nothing for a document
    fn open(&mut self, id: &str, created: u64, model: &str) -> bool;
    /// one parser fragment, in arrival order
    fn on_emit(&mut self, id: &str, created: u64, model: &str, e: &Emit) -> bool;
    /// after the last token: the final chunk plus `[DONE]`, nothing for a document
    fn on_finish(
        &mut self,
        id: &str,
        created: u64,
        model: &str,
        finish: &str,
        t: &Timing,
        include_usage: bool,
        timings_per_token: bool,
    ) -> bool;
}

/// the A4/A5 sink: one flushed SSE frame per piece, over any writer
struct SseSink<W: Write> {
    w: W,
}

impl<W: Write> SseSink<W> {
    fn new(w: W) -> Self {
        SseSink { w }
    }
}

impl<W: Write> ChatSink for SseSink<W> {
    fn open(&mut self, id: &str, created: u64, model: &str) -> bool {
        sse_send(&mut self.w, &sse_frame(&chunk_role(id, created, model)))
    }
    fn on_emit(&mut self, id: &str, created: u64, model: &str, e: &Emit) -> bool {
        let doc = match e {
            Emit::Content(t) => chunk_content(id, created, model, t),
            Emit::Call { index, id: call_id, name } => {
                chunk_tool_open(id, created, model, *index, call_id, name)
            }
            Emit::Args { index, text } => chunk_tool_args(id, created, model, *index, text),
        };
        sse_send(&mut self.w, &sse_frame(&doc))
    }
    fn on_finish(
        &mut self,
        id: &str,
        created: u64,
        model: &str,
        finish: &str,
        t: &Timing,
        include_usage: bool,
        timings_per_token: bool,
    ) -> bool {
        let last = chunk_finish(id, created, model, finish, t, include_usage, timings_per_token);
        sse_send(&mut self.w, &sse_frame(&last)) && sse_send(&mut self.w, SSE_DONE)
    }
}

/// - #39 B3a: what the fragments of ONE tool call add up to
/// - `arguments` is the concatenation of every `Emit::Args` of that index, a JSON text
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CallBuf {
    id: String,
    name: String,
    arguments: String,
}

/// - #39 B3a: the sink of a `stream:false` request, the same deltas into strings
/// - nothing is written to the socket here; the document leaves after the loop
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct CollectSink {
    /// every `Emit::Content` in order, the `message.content` of the document
    content: String,
    /// one entry per tool call index, in the order the parser opened them
    calls: Vec<CallBuf>,
}

impl ChatSink for CollectSink {
    fn open(&mut self, _id: &str, _created: u64, _model: &str) -> bool {
        true
    }
    fn on_emit(&mut self, _id: &str, _created: u64, _model: &str, e: &Emit) -> bool {
        match e {
            Emit::Content(t) => self.content.push_str(t),
            Emit::Call { index, id: call_id, name } => {
                if *index == self.calls.len() {
                    self.calls.push(CallBuf::default());
                }
                if let Some(c) = self.calls.get_mut(*index) {
                    c.id = call_id.clone();
                    c.name = name.clone();
                }
            }
            Emit::Args { index, text } => {
                if let Some(c) = self.calls.get_mut(*index) {
                    c.arguments.push_str(text);
                }
            }
        }
        true
    }
    fn on_finish(
        &mut self,
        _id: &str,
        _created: u64,
        _model: &str,
        _finish: &str,
        _t: &Timing,
        _include_usage: bool,
        _timings_per_token: bool,
    ) -> bool {
        true
    }
}

/// - #29 A7: one sink call per parser fragment, in order
/// - the chunk counters of the `[chat]` line are counted HERE, so both sinks count alike
/// - `false` means the client is gone and the generation loop must stop
fn send_emits(
    sink: &mut dyn ChatSink,
    id: &str,
    created: u64,
    model: &str,
    pieces: &[Emit],
    content_chunks: &mut usize,
    tool_chunks: &mut usize,
) -> bool {
    for e in pieces {
        match e {
            Emit::Content(_) => *content_chunks += 1,
            Emit::Call { .. } | Emit::Args { .. } => *tool_chunks += 1,
        }
        if !sink.on_emit(id, created, model, e) {
            return false;
        }
    }
    true
}

/// - #39 B3a: the ONE `chat.completion` document a `stream:false` request answers
/// - `usage` and `timings` are `usage_json` and `timings_json`, the objects of the final
///   stream chunk, and both are ALWAYS present: one document shape, and the probe-suite
///   reads `usage.completion_tokens` (`probe-suite.py:681-683`) while sending neither
///   `stream_options` nor `timings_per_token`
/// - `content` is always a string, empty when the answer was a tool call alone
/// - `tool_calls` appears only when the parser closed at least one call, in the OpenAI
///   non streaming shape (`id`, `type`, `function`), `arguments` a JSON STRING
/// - `reasoning_content` is NOT sent: the template renders `enable_thinking false`, so the
///   think block is empty (`tokenizer.rs:18-19`)
/// - pure: the whole document contract is one function the test drives directly
fn completion_json(
    id: &str,
    created: u64,
    model: &str,
    content: &str,
    calls: &[CallBuf],
    finish: &str,
    t: &Timing,
) -> serde_json::Value {
    let mut message = serde_json::json!({ "role": "assistant", "content": content });
    if !calls.is_empty() {
        let arr: Vec<serde_json::Value> = calls
            .iter()
            .map(|c| {
                serde_json::json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": c.arguments },
                })
            })
            .collect();
        if let Some(obj) = message.as_object_mut() {
            obj.insert("tool_calls".to_string(), serde_json::Value::Array(arr));
        }
    }
    serde_json::json!({
        "id": id,
        "object": "chat.completion",
        "created": created,
        "model": model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": usage_json(t),
        "timings": timings_json(t),
    })
}

/// a JSON response plus its status, so the caller can log one label
fn respond_json(
    stream: &mut TcpStream,
    status: &'static str,
    doc: &serde_json::Value,
) -> &'static str {
    if let Err(e) = respond(stream, status, &doc.to_string()) {
        eprintln!("[serve] response write failed: {e}");
    }
    status
}

/// - `POST /v1/chat/completions`
/// - every rejection happens BEFORE the first stream byte, as a JSON response
/// - the return value is the status for the access log line
fn chat_route(stream: &mut TcpStream, srv: &mut Srv, body: &[u8]) -> &'static str {
    let mut req = match parse_chat(body) {
        Ok(r) => r,
        Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
    };
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };
    // #29 A7: `tools` is rendered, and an assistant turn's `arguments` string is turned into
    // the object the template's `|items` needs. Both are the request as Crow sends it.
    let msgs = normalize_messages(&req.messages);
    let ids = match tk.encode_chat(&msgs, req.tools.as_ref(), true, req.enable_thinking) {
        Ok(v) => v,
        Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
    };
    if ids.is_empty() {
        return respond_json(stream, "400 Bad Request", &error_json("the rendered prompt is empty"));
    }
    // #VIT: run the visual tower for the request's images, expand every
    // image_pad into its visual tokens, arm the mrope span tables. With
    // CROW_VIT=0 the engine holds no tower and the request falls through as
    // the placeholder of record (the single image_pad rides as a token).
    let ids: Vec<u32> = if !req.images.is_empty() && srv.eng.vit.is_some() {
        eprintln!("[vit-chat] {} image(s) in request, decoding data URLs ...", req.images.len());
        let mut bytes = Vec::with_capacity(req.images.len());
        for (i, url) in req.images.iter().enumerate() {
            match decode_data_url(url) {
                Ok((_mime, raw)) => bytes.push(raw),
                Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&format!("image {i}: {e}"))),
            }
        }
        let vit_t0 = std::time::Instant::now();
        let plan = match unsafe { srv.eng.vit.as_mut().unwrap().build_plan(&srv.eng.k, &ids, &bytes) } {
            Ok(p) => p,
            Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
        };
        let vit_ms = vit_t0.elapsed().as_secs_f64() * 1e3;
        eprintln!(
            "[vit-chat] {} image(s), {} visual token(s), grids {:?}, mrope delta {}, vision {} ms (decode + preprocess + tower)",
            req.images.len(),
            plan.n_visual,
            plan.grids,
            plan.delta,
            vit_ms
        );
        let expanded = plan.ids.clone();
        unsafe { srv.eng.begin_vision(plan, expanded.len() + req.max_tokens) };
        expanded
    } else {
        ids
    };
    // #26 review: the prompt alone is the only 413 case; a budget that does not fit is CLAMPED,
    // not refused (the old combined check was dead, `max_tokens` is capped at 32768 first)
    let budget = match clamped_max_tokens(ids.len(), req.max_tokens, srv.n_ctx) {
        Some(n) => n,
        None => {
            let m = format!("prompt {} tokens over n_ctx {}", ids.len(), srv.n_ctx);
            return respond_json(stream, "413 Payload Too Large", &error_json(&m));
        }
    };
    if budget != req.max_tokens {
        eprintln!(
            "[chat] max_tokens {} clamped to {} (prompt {}, n_ctx {})",
            req.max_tokens,
            budget,
            ids.len(),
            srv.n_ctx
        );
        req.max_tokens = budget;
    }
    if req.stream {
        chat_stream(stream, srv, &req, &ids)
    } else {
        chat_document(stream, srv, &req, &ids)
    }
}

/// - #39 B3a: `stream:true`, the A4/A5 wire, unchanged byte for byte: the SSE head, the
///   role chunk, one flushed frame per emitted delta, the final chunk, `[DONE]`
/// - the generation itself is `chat_generate`, the ONE loop both request forms run
fn chat_stream(stream: &mut TcpStream, srv: &mut Srv, req: &ChatReq, ids: &[u32]) -> &'static str {
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };

    const HEAD: &str = "HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-cache\r\nConnection: close\r\n\r\n";
    if !sse_send(stream, HEAD) {
        return "200 OK (client gone)";
    }
    let mut sink = SseSink::new(stream);
    let out = chat_generate(srv, req, ids, tk, &mut sink);
    if out.aborted {
        "200 OK (client gone)"
    } else {
        "200 OK (text/event-stream)"
    }
}

/// - #39 B3a: `stream:false`, or no `stream` field at all, the form the probe-suite
///   (`probe-suite.py:624-639`) and Crow's rollover digest (`crow_core.py:2960-2990`) send
/// - the SAME generation as the stream, collected instead of written, then ONE
///   `chat.completion` document with `Content-Length` and `Connection: close`, like every
///   other JSON route of this server
/// - nothing reaches the socket before the document, so a mid generation failure is still
///   a JSON answer, not a half written stream
fn chat_document(stream: &mut TcpStream, srv: &mut Srv, req: &ChatReq, ids: &[u32]) -> &'static str {
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };
    let mut sink = CollectSink::default();
    let out = chat_generate(srv, req, ids, tk, &mut sink);
    let doc = completion_json(
        &out.id,
        out.created,
        &req.model,
        &sink.content,
        &sink.calls,
        out.finish,
        &out.timing,
    );
    respond_json(stream, "200 OK", &doc)
}

/// - #39 B3a: what one shared generation produced, whatever its sink did with it
/// - the sink holds the text; this holds what BOTH request forms still need afterwards
struct GenOut {
    /// the `chatcmpl-...` id of this request: every chunk's `id`, and the document's
    id: String,
    /// the unix second of this request, the `created` of both forms
    created: u64,
    /// `stop`, `length` or `tool_calls`
    finish: &'static str,
    /// the counts and walls behind `usage` and `timings`
    timing: Timing,
    /// a sink call refused: the client is gone and the loop stopped early
    aborted: bool,
}

/// - #37: the two preconditions `Engine::trickle_tick` ASSERTS (`gen.rs:3128-3129`)
/// - exact NVFP4 tier only: a `CROW_COLD_TIER` process has no three-way exchange
/// - spare hot slots must exist (`stride > n`), or there is nothing to copy into
/// - read once at start and once per request, never inside the token loop
/// - a misconfigured process therefore logs one line instead of panicking mid request
fn trickle_ready(eng: &Engine) -> bool {
    eng.res.lb.is_none() && eng.res.stride > eng.res.n
}

/// - #39 B3a: THE generation loop of this server, the only one. `stream:true` runs it with
///   `SseSink`, `stream:false` with `CollectSink`; prefix cache, sampler, stop rules,
///   tool-call parser, snapshot, counters and the three `[chat]` stderr lines are shared.
/// - prefill, then decode, one sink call per emitted delta
/// - #31 A9: the prefix cache decides FIRST (spec 7.4); a warm request rolls back to `P`
///   and prefills `ids[P..]`, a cold one runs `reset_to_zero` and prefills everything
/// - ONE snapshot per request, unconditional (M2b, #36): after the prompt
/// - a sink refusal breaks the loop; the next request rolls back or resets before any prefill
fn chat_generate(
    srv: &mut Srv,
    req: &ChatReq,
    ids: &[u32],
    tk: &'static crow_nest_engine::tokenizer::ChatTokenizer,
    sink: &mut dyn ChatSink,
) -> GenOut {
    srv.seq += 1;
    let created = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let id = format!("chatcmpl-{created}-{}", srv.seq);
    let model = req.model.clone();

    let prompt: Vec<i64> = ids.iter().map(|&v| v as i64).collect();
    // #VIT: a text-only request never runs behind stale mrope tables — a 413
    // or an error after the plan armed them leaves them set, so clear here.
    if req.images.is_empty() {
        srv.eng.end_vision();
    }
    // #31 A9: the detection rule of spec 7.4, host side, IDS ONLY. `Engine::history` is the
    // held conversation: prompt ids AND generated ids (`gen.rs:2698`, `gen.rs:2945`).
    let plan = srv.cache.decide(&srv.eng.history, &prompt);
    let held = srv.eng.history.len();
    let cache_on = srv.cache.enabled();
    let snaps = srv.cache.positions();
    let reusable = srv.cache.reuse_candidates();
    // unsafe: engine kernels; the CUDA context and engine/.engine.lock are this process's
    let t_reset = Instant::now();
    let cached_n = match plan.reuse {
        // warm: restore the four recurrent buffers, put pos, done_blocks and history back
        Some((slot, p)) => {
            unsafe { srv.cache.rollback(srv.eng, slot) };
            p
        }
        // cold: no snapshot at or below L, so the whole state goes back to 0 (A4). The
        // slot is dropped with it: its position names a history this process discards.
        None => {
            unsafe { srv.eng.reset_to_zero() };
            srv.cache.invalidate();
            0
        }
    };
    let reset_ms = t_reset.elapsed().as_secs_f64() * 1e3;
    let prefilled = prompt.len() - cached_n;
    // `reset_ms` is the ONE name for this number: the rollback of a warm request or the
    // `reset_to_zero` of a cold one. The `[chat]` line below calls it `reset` as well.
    // With the cache off `decide` returns before it computes `L`, so no number is claimed.
    eprintln!(
        "[cache] {} L {} (held {}), P {cached_n}, snapshots {:?}, reusable {:?}, prefill {prefilled} of {} tok, reset {reset_ms:.3} ms",
        if plan.reuse.is_some() { "WARM" } else { "COLD" },
        if cache_on { plan.l.to_string() } else { "n/a".to_string() },
        held,
        snaps,
        reusable,
        prompt.len()
    );
    // #27 A5: the timed window is the prefill CALL alone, the same window `decode run` prints
    // as `prefill done in X s`; the rollback, the reset and the tokenizer are outside it.
    // #VIT: under CROW_VIT_DUMP=dir the prefill also collects every prompt logit
    // row (the oracle compare path) and the patch inputs are written before it.
    let dump_dir = std::env::var("CROW_VIT_DUMP").ok().filter(|_| !req.images.is_empty());
    let mut collect = dump_dir.as_ref().map(|_| Vec::new());
    let t_pre = Instant::now();
    let mut next = unsafe {
        match collect.as_mut() {
            Some(c) => srv.eng.prefill(srv.cnq, &prompt[cached_n..], Some(c)),
            None => srv.eng.prefill(srv.cnq, &prompt[cached_n..], None),
        }
    };
    let prefill_ms = t_pre.elapsed().as_secs_f64() * 1e3;
    if let Some(dir) = dump_dir.as_ref() {
        unsafe { write_vit_dump(srv, dir, &prompt, collect.as_deref(), prefill_ms) };
    }
    // #31 A9: spec 7.6 point 1, after the prefill of this turn's prompt, unconditional (M1).
    // Before the first `decode_step`, so no capture stream is live and no graph exists yet.
    // prefill clean: the prefill above started at 0 or at a prefill clean `P`, so every
    // KV and pooled QSA row below `pos` is a prefill row (`cache.rs`, the induction)
    let snap1_ms = unsafe { srv.cache.snapshot(srv.eng, SLOT_PROMPT, true) };
    // a disabled cache copies nothing, so it reports nothing either
    if cache_on {
        eprintln!(
            "[cache] snapshot point 1 (after prompt) at pos {}, DtoH {snap1_ms:.3} ms",
            srv.eng.pos
        );
    }

    // #28 A6: arm or disarm the device sampler for THIS request. After `prefill` and before the
    // first `decode_step`: the decode graph was dropped before that prefill, cold by
    // `reset_to_zero` and warm by `PrefixCache::rollback`, so that step re-captures it
    // and the `sample_k` node goes in with it. `enable_dev_sampler` re-uploads `Rng::new(seed)`
    // and clears the presence mask on every call, which is the per request reseed (M1).
    let sampler = sampler_from(req);
    match &sampler {
        Some(s) => {
            // the device buffers a greedy request parked come back here, nothing is reallocated
            if srv.eng.dev_sampler.is_none() {
                srv.eng.dev_sampler = srv.parked_sampler.take();
            }
            // unsafe: device uploads and one eager sampler launch, as parity.rs:194-205 does
            unsafe {
                srv.eng.enable_dev_sampler(s);
                // the first id is drawn from the prefill's last logits row, not taken from argmax
                next = srv.eng.sample_last();
            }
            eprintln!(
                "[chat] sampling on the device: temperature {} top_p {} top_k {} presence_penalty {} seed {}",
                s.temperature, s.top_p, s.top_k, s.presence_penalty, s.seed
            );
        }
        None => {
            // greedy is the A4 path: `decode_step` samples whenever `dev_sampler` is Some
            // (gen.rs:2905-2909) and `reset_to_zero` does not clear it, so it is taken out here
            if srv.eng.dev_sampler.is_some() {
                srv.parked_sampler = srv.eng.dev_sampler.take();
            }
            eprintln!("[chat] greedy (temperature absent or <= 0)");
        }
    }
    if req.min_p != 0.0 {
        eprintln!(
            "[chat] min_p {} accepted and ignored (device sampler has no min_p; #28)",
            req.min_p
        );
    }

    // #29 A7: the tool-call parser of THIS request. `tool_open` is the id the decode loop
    // arms it with; without that id the literal text `<tool_call>` stays content.
    let mut ts = ToolStream::new(req.tools.as_ref());
    let tool_open = tk.token_id(TOOL_OPEN);
    let mut tool_chunks = 0usize;

    let mut aborted = !sink.open(&id, created, &model);

    // #27 A5: the decode window opens at the FIRST `decode_step` and closes when the last one
    // returns, so the detokenize and the sink call of token 1 are not counted as decode
    let mut t_dec: Option<Instant> = None;
    let mut out: Vec<u32> = Vec::with_capacity(req.max_tokens);
    // bytes of the accumulated decode that already left as content
    let mut emitted = 0usize;
    let mut content_chunks = 0usize;
    let mut finish = "length";
    let mut decode_ms = 0.0f64;
    // #37: the stream trickle, one tick per `decode_step`, the mirror of `decode.rs:224-231`.
    // `cfg.adapt` is what `apply_adapt_policy` (geo.rs:167-176) gave this process: with
    // `CROW_ADAPT_STREAM` unset and chunk 2048 that is stream / 7 spare / every 16 / max 7.
    // `decode.rs` ticks for `i in 1..gen`, that is before every `decode_step` EXCEPT the
    // first; loop index `i` here names the same token, so the guard is the same `i > 0`.
    let Adapt { stream: adapt_stream, every: adapt_every, max: adapt_max, .. } = srv.eng.cfg.adapt;
    let tick_trickle = adapt_stream && adapt_every > 0 && trickle_ready(srv.eng);
    let mut trickle_swaps = 0usize;
    if !aborted {
        for i in 0..req.max_tokens {
            if EOS_IDS.contains(&next) {
                finish = "stop";
                break;
            }
            out.push(next as u32);
            // #29 A7: the ID is what opens a tool call, never the text (spec of the task)
            if Some(next as u32) == tool_open {
                ts.arm();
            }
            let full = match tk.decode(&out) {
                Ok(t) => t,
                Err(e) => {
                    eprintln!("[chat] detokenize failed: {e}");
                    String::new()
                }
            };
            if let Some(delta) = next_delta(&full, emitted) {
                emitted = full.len();
                let pieces = ts.feed(delta);
                if !send_emits(
                    sink,
                    &id,
                    created,
                    &model,
                    &pieces,
                    &mut content_chunks,
                    &mut tool_chunks,
                ) {
                    aborted = true;
                    break;
                }
            }
            // the budget is spent: no decode_step whose token nobody reads
            if i + 1 == req.max_tokens {
                break;
            }
            // #37: the tick sits INSIDE the decode window, as it does in `decode.rs`, so
            // `predicted_ms` carries the host bookkeeping of the trickle it pays for.
            // unsafe: side-stream copies plus hot-set table flips (`gen.rs:3127-3197`)
            if tick_trickle && i > 0 {
                trickle_swaps += unsafe { srv.eng.trickle_tick(i % adapt_every == 0, adapt_max) };
            }
            let t = *t_dec.get_or_insert_with(Instant::now);
            next = unsafe { srv.eng.decode_step(srv.cnq, next as i64) };
            decode_ms = t.elapsed().as_secs_f64() * 1e3;
        }
    }

    // #37: finish the trickle of THIS request, as `decode.rs:253` does after its loop.
    // Outside the decode window on purpose: `decode run` does not charge the drain to a
    // token either. No copy is left in flight, so the next request prefills on a settled
    // hot set and the A9 warm turn sees the residency the `[chat]` line reported.
    // unsafe: two empty ticks plus a stream sync (`gen.rs:3202-3216`)
    if tick_trickle {
        let drained = unsafe { srv.eng.trickle_drain() };
        eprintln!(
            "[chat] trickle {trickle_swaps} swaps started this request (every {adapt_every}, \
             max {adapt_max}/layer, {drained} since process start)"
        );
    }

    // the tail the hold back kept (a never completed sequence, or a real U+FFFD), plus the
    // fragments the tool-call parser still owes (#29 A7)
    let mut malformed = false;
    if !aborted {
        let full = tk.decode(&out).unwrap_or_default();
        let mut pieces = if full.len() > emitted && full.is_char_boundary(emitted) {
            ts.feed(&full[emitted..])
        } else {
            Vec::new()
        };
        malformed = ts.finish(&mut pieces);
        if !send_emits(
            sink,
            &id,
            created,
            &model,
            &pieces,
            &mut content_chunks,
            &mut tool_chunks,
        ) {
            aborted = true;
        }
    }
    // #29 A7: a closed call answers `tool_calls`; a malformed one keeps `stop` / `length`.
    // `ToolStream::finish` closes a call whose `</function>` arrived, so EOS in the tail is
    // a complete call, not a malformed one (#29 review).
    if malformed {
        eprintln!(
            "[chat] MALFORMED tool call: no </function> or no name before the end; \
             the raw markup went out as content, finish stays {finish}"
        );
    } else if ts.closed() > 0 {
        finish = "tool_calls";
    }
    if ts.dropped() > 0 {
        eprintln!(
            "[chat] {} byte(s) dropped after the first tool call started: text the template \
             forbids, or markup after the last `</function>`",
            ts.dropped()
        );
    }
    // #30 A8: the engine counters, read AFTER the last decode step and BEFORE the final chunk.
    // cumulative, never reset (Crow #54 rule); request-local = difference of two blocks.
    // `Engine::drain_counters` is a plain `cuda::dtoh_u64` of 48 x 2 u64 (residency.rs:638-641):
    // it READS the device block, it does not zero it. `Ple::req` / `Ple::miss` are host u64 that
    // only ever grow (gen.rs:1041-1042). Nothing here writes device or host state.
    let t_ctr = Instant::now();
    let blocks = unsafe { srv.eng.drain_counters() };
    let (selections_total, cold_total) = blocks
        .iter()
        .fold((0u64, 0u64), |a, c| (a.0 + c[0], a.1 + c[1]));
    let (ple_rows_total, ple_miss_total) = (srv.eng.ple.req, srv.eng.ple.miss);
    let counters_ms = t_ctr.elapsed().as_secs_f64() * 1e3;

    let gen = out.len();
    let timing = Timing {
        prompt_n: prefilled,
        cached_n,
        predicted_n: gen,
        prompt_ms: prefill_ms,
        predicted_ms: decode_ms,
        selections_total,
        cold_total,
        ple_rows_total,
        ple_miss_total,
    };
    if !aborted {
        let _ = sink.on_finish(
            &id,
            created,
            &model,
            finish,
            &timing,
            req.include_usage,
            req.timings_per_token,
        );
    }

    // #36 M2b (robin 2026-09-10): the after-answer snapshot of M1 is GONE from here.
    // `decode_step` wrote every row of the answer, those rows are not bit equal to the
    // prefill rows at the same positions (#31 A9), so that slot was taken on every request
    // and consumed on none. Dropping it saves 130,646,016 B of pageable host RAM per
    // process and one 14.5 ms DtoH per request. The reuse behaviour is unchanged.

    // #27 doc: the tok/s below is (gen - 1) / decode_ms, the wire's
    // `timings.predicted_per_second` is gen / decode_ms. Both are correct for what they name:
    // - `decode_ms` is the wall of the `decode_step` calls only (gen - 1 of them).
    // - The log therefore divides by gen - 1: the honest decode rate, prefill token excluded.
    // - The wire follows llama-server, where `predicted_n` counts the prefill token as well.
    // - Making the two equal would either mislabel the log or break Crow's reader.
    // #39 B3a fix round 1: `usage`/`timings` below are the EFFECTIVE flags, not the request's.
    // `completion_json` puts both on the document unconditionally (`stream:false`), so the
    // stream flags would under-report there; the stream path still logs the request flags,
    // byte identical to before this fix.
    let (log_usage, log_timings) = if req.stream {
        (req.include_usage, req.timings_per_token)
    } else {
        (true, true)
    };
    eprintln!(
        "[chat] prompt {} tok ({cached_n} cached, {prefilled} prefilled), generated {gen} tok, prefill {prefill_ms:.1} ms ({:.1} tok/s), reset {reset_ms:.1} ms, decode {decode_ms:.1} ms, {:.1} tok/s, finish {finish}, content chunks {content_chunks}, tool chunks {tool_chunks}, tool calls {}, usage {}, timings {}, crow_trickle_swaps {trickle_swaps}{}",
        ids.len(),
        per_second(prefilled, prefill_ms),
        (gen.saturating_sub(1)) as f64 * 1000.0 / decode_ms.max(1e-9),
        ts.closed(),
        log_usage,
        log_timings,
        if aborted { ", client gone" } else { "" }
    );
    // #30 A8: the same numbers the `timings` block carries, cumulative since process start
    eprintln!(
        "[chat] counters (cumulative, never reset): expert selections {selections_total}, \
         expert cold {cold_total}, ple rows {ple_rows_total}, ple misses {ple_miss_total}, \
         layers {LAYERS}, counter read {counters_ms:.3} ms"
    );
    eprintln!("[chat] ids {out:?}");
    // #VIT: the image request is done — back to the load-time rope tables.
    // The next request re-arms them from its own plan. The dump gains the
    // generated ids, so the oracle can compare its own greedy continuation.
    if let Some(dir) = dump_dir.as_ref() {
        let seq_path = format!("{dir}/vit-gen-sequence.json");
        if let Ok(mut doc) = std::fs::read_to_string(&seq_path) {
            if let Ok(mut v) = serde_json::from_str::<serde_json::Value>(&doc) {
                v["generated"] = serde_json::json!(out);
                doc = v.to_string();
                let _ = std::fs::write(&seq_path, doc);
            }
        }
    }
    srv.eng.end_vision();
    GenOut {
        id,
        created,
        finish,
        timing,
        aborted,
    }
}

/// #VIT: the `CROW_VIT_DUMP=<dir>` artifacts of one image request, the input of
/// both oracle comparisons:
/// - `vit-gen-sequence.json`: the exact prompt ids (expanded), the cached
///   prefix offset, the grids, the visual-row map and the mrope delta
/// - `gpu-logits.f32`: one f32 [V] row per processed prompt position
/// The per-image patch inputs and the tower outputs are written by
/// `Vit::build_plan` under the same variable.
unsafe fn write_vit_dump(
    srv: &Srv,
    dir: &str,
    prompt: &[i64],
    collect: Option<&[Vec<f32>]>,
    prefill_ms: f64,
) {
    let _ = std::fs::create_dir_all(dir);
    let (map, grids, delta, n_visual) = match srv.eng.vit_plan.as_ref() {
        Some(p) => (p.map.clone(), p.grids.clone(), p.delta, p.n_visual),
        None => (Vec::new(), Vec::new(), 0i64, 0usize),
    };
    let rows = collect.map(|c| c.len()).unwrap_or(0);
    let seq = serde_json::json!({
        "rows": rows,
        "prompt_len": prompt.len(),
        "cached_n": prompt.len() - rows,
        "ids": prompt,
        "visual_map": map,
        "grids": grids,
        "n_visual": n_visual,
        "prefill_ms": prefill_ms,
    });
    let _ = std::fs::write(format!("{dir}/vit-gen-sequence.json"), seq.to_string());
    if let Some(c) = collect {
        let mut flat = Vec::with_capacity(c.len() * crow_nest_engine::geo::V);
        for row in c {
            flat.extend_from_slice(row);
        }
        f32_to_file(&format!("{dir}/gpu-logits.f32"), &flat);
    }
}

/// raw f32 little-endian file write, failures logged not raised
fn f32_to_file(path: &str, v: &[f32]) {
    match std::fs::File::create(path) {
        Ok(mut f) => {
            use std::io::Write;
            let bytes = unsafe { std::slice::from_raw_parts(v.as_ptr() as *const u8, v.len() * 4) };
            let _ = f.write_all(bytes);
        }
        Err(e) => eprintln!("[vit-dump] write {path} failed: {e}"),
    }
}

// ---------------------------------------------------------------- the socket

/// one response, HTTP/1.1 with Content-Length; the connection closes after it
fn respond(stream: &mut TcpStream, status: &str, body: &str) -> std::io::Result<()> {
    let head = format!(
        "HTTP/1.1 {status}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
        body.len()
    );
    stream.write_all(head.as_bytes())?;
    stream.write_all(body.as_bytes())?;
    stream.flush()
}

/// - reads one line into `buf`, at most `cap` bytes from `r`
/// - the cap is what keeps an endless line from growing the buffer
/// - returns the bytes taken; `buf` ends in a newline on a complete line
fn read_line_capped<R: BufRead>(r: &mut R, cap: usize, buf: &mut Vec<u8>) -> std::io::Result<usize> {
    r.by_ref().take(cap as u64).read_until(b'\n', buf)
}

/// - request line, headers and body of one connection, over any `BufRead`
/// - the read is bounded: never more than `MAX_HEAD_BYTES` are buffered
/// - a head that spends the cap without a newline gives `Head::HeadTooLarge`
/// - `Content-Length` must be one well formed value, else `Head::Bad`
/// - `Transfer-Encoding: chunked` gives `Head::Chunked`
/// - the body is drained after the blank line
/// - draining first is what lets a bodied POST see the 404 instead of a reset
fn read_head_from<R: BufRead>(r: &mut R) -> std::io::Result<Head> {
    let mut used = 0usize;

    let mut first = Vec::new();
    let n = read_line_capped(r, MAX_HEAD_BYTES, &mut first)?;
    if n == 0 {
        return Ok(Head::Empty);
    }
    used += n;
    if !first.ends_with(b"\n") && used >= MAX_HEAD_BYTES {
        return Ok(Head::HeadTooLarge);
    }
    let first = String::from_utf8_lossy(&first).into_owned();

    let mut content_length: Option<usize> = None;
    let mut bad_length = false;
    let mut chunked = false;
    loop {
        let budget = MAX_HEAD_BYTES - used;
        if budget == 0 {
            return Ok(Head::HeadTooLarge);
        }
        let mut raw = Vec::new();
        let n = read_line_capped(r, budget, &mut raw)?;
        if n == 0 {
            break;
        }
        used += n;
        if !raw.ends_with(b"\n") && used >= MAX_HEAD_BYTES {
            return Ok(Head::HeadTooLarge);
        }
        let line = String::from_utf8_lossy(&raw).into_owned();
        if line.trim().is_empty() {
            break;
        }
        if let Some(v) = header_value(&line, "content-length") {
            match (content_length, v.parse::<usize>()) {
                (None, Ok(len)) => content_length = Some(len),
                // a second Content-Length, or one that does not parse, is a 400
                _ => bad_length = true,
            }
        }
        if let Some(v) = header_value(&line, "transfer-encoding") {
            if v.split(',').any(|c| c.trim().eq_ignore_ascii_case("chunked")) {
                chunked = true;
            }
        }
    }

    // chunked first: there is no declared length to trust or to drain
    if chunked {
        return Ok(Head::Chunked);
    }
    if bad_length {
        return Ok(Head::Bad);
    }

    let content_length = content_length.unwrap_or(0);
    if content_length > MAX_BODY_BYTES {
        return Ok(Head::BodyTooLarge(content_length));
    }
    let mut body = vec![0u8; content_length];
    if content_length > 0 {
        r.read_exact(&mut body)?;
    }

    match parse_request_line(&first) {
        Some((method, target)) => Ok(Head::Req { method, target, body }),
        None => Ok(Head::Bad),
    }
}

/// - `read_head_from` over one connection, buffered
fn read_head(stream: &TcpStream) -> std::io::Result<Head> {
    let mut r = BufReader::new(stream);
    read_head_from(&mut r)
}

/// what a connection needs beyond the socket: the engine, its container, the static props
struct Srv<'a> {
    /// the one loaded engine of this process; every chat request resets it first
    eng: &'a mut Engine,
    /// the container handle `prefill` fills PLE rows from
    cnq: &'a mut Cnq,
    /// `/props` `model_path`
    model_path: &'a str,
    /// `/props` `n_ctx`, read back from the loaded states
    n_ctx: usize,
    /// `/props` `prompt_chunk`, pinned for the process
    prompt_chunk: usize,
    /// per process request counter, the tail of every chunk `id`
    seq: u64,
    /// #28: the device sampler while a GREEDY request runs, so its node stays out of the
    /// capture; the next sampled request takes it back and reuses the same device buffers
    parked_sampler: Option<DevSampler>,
    /// #31 A9: the ONE held conversation of this process and its ONE snapshot (M2b, #36)
    cache: PrefixCache,
    /// #32 A10: `--slot-save-path <dir>`; `None` makes `/slots/0` refuse both actions
    slot_save_path: Option<String>,
}

/// - `POST /slots/0?action=save|restore` (#32 A10)
/// - every refusal is a 4xx with a JSON error body and leaves the engine untouched
/// - the return value is the status and the document `serve_one` writes
fn slot_route(srv: &mut Srv, target: &str, body: &[u8]) -> (&'static str, serde_json::Value) {
    let refuse = |msg: String| {
        eprintln!("[slot] refused: {msg}");
        ("400 Bad Request", error_json(&msg))
    };
    let Some(action) = query_param(target, "action") else {
        return refuse("no action in the query string (save or restore)".to_string());
    };
    let name = match slot_filename(body) {
        Ok(n) => n,
        Err(e) => return refuse(e),
    };
    let Some(dir) = srv.slot_save_path.clone() else {
        return refuse("this server was started without --slot-save-path".to_string());
    };
    let path = std::path::Path::new(&dir).join(&name);
    match action {
        "save" => {
            // 409, and the ONE case that earns it: the request is well formed and the
            // process simply holds no prefill clean position yet. The status belongs
            // here, at the HTTP layer, so `slot::save` can stay a plain Result<_, String>.
            if srv.cache.prompt_slot().is_none() {
                let m = "no prefill clean state is held; run one chat request first".to_string();
                eprintln!("[slot] refused: {m}");
                return ("409 Conflict", error_json(&m));
            }
            // unsafe: device to host copies only; the engine state is not written
            match unsafe { slot::save(srv.eng, &srv.cache, srv.model_path, &path) } {
                Ok(s) => {
                    eprintln!(
                        "[slot] save {name:?}: n_saved {}, {} B, {:.1} ms -> {}",
                        s.n_saved,
                        s.n_written,
                        s.ms,
                        path.display()
                    );
                    ("200 OK", slot_saved_json(&name, &s))
                }
                Err(e) => refuse(e),
            }
        }
        // unsafe: host to device uploads plus the A4 graph teardown, as PrefixCache::rollback
        "restore" => {
            match unsafe { slot::restore(srv.eng, &mut srv.cache, srv.model_path, &path) } {
                Ok(r) => {
                    eprintln!(
                        "[slot] restore {name:?}: n_restored {}, {} B, {:.1} ms <- {}",
                        r.n_restored,
                        r.n_read,
                        r.ms,
                        path.display()
                    );
                    ("200 OK", slot_restored_json(&name, &r))
                }
                Err(e) => refuse(e),
            }
        }
        other => refuse(format!("unknown action {other:?} (save or restore)")),
    }
}

fn serve_one(stream: &mut TcpStream, srv: &mut Srv) {
    let t = Duration::from_secs(IO_TIMEOUT_SECS);
    if let Err(e) = stream.set_read_timeout(Some(t)) {
        eprintln!("[serve] no read timeout on this connection, closing: {e}");
        return;
    }
    if let Err(e) = stream.set_write_timeout(Some(t)) {
        eprintln!("[serve] no write timeout on this connection, closing: {e}");
        return;
    }

    let head = match read_head(stream) {
        Ok(h) => h,
        Err(e) => {
            eprintln!("[serve] read failed or timed out after {IO_TIMEOUT_SECS}s, closing: {e}");
            return;
        }
    };

    let (label, status, doc) = match head {
        Head::Empty => return,
        Head::Bad => (
            "<no HTTP request line>".to_string(),
            "400 Bad Request",
            error_json("bad request"),
        ),
        Head::HeadTooLarge => (
            format!("<head over {MAX_HEAD_BYTES} bytes>"),
            "431 Request Header Fields Too Large",
            error_json("request header fields too large"),
        ),
        Head::BodyTooLarge(n) => (
            format!("<Content-Length {n} over {MAX_BODY_BYTES} bytes>"),
            "413 Payload Too Large",
            error_json("payload too large"),
        ),
        Head::Chunked => (
            "<Transfer-Encoding: chunked>".to_string(),
            "501 Not Implemented",
            error_json("chunked transfer encoding not supported"),
        ),
        Head::Req { method, target, body } => {
            let path = route_path(&target).to_string();
            let label = format!("{method} {target} (body {} bytes)", body.len());
            match route(&method, &path) {
                // the chat route writes its own response: SSE, or a JSON error
                Route::Chat => {
                    let status = chat_route(stream, srv, &body);
                    eprintln!("[serve] {label} -> {status}");
                    let _ = stream.shutdown(Shutdown::Write);
                    return;
                }
                Route::Health => (label, "200 OK", serde_json::json!({ "status": "ok" })),
                Route::Props => (
                    label,
                    "200 OK",
                    props_json(srv.model_path, srv.n_ctx, srv.prompt_chunk, srv.eng.vit.is_some()),
                ),
                Route::Slots => (
                    label,
                    "200 OK",
                    slots_json(srv.n_ctx, srv.cache.prompt_slot().map(|(p, _)| p).unwrap_or(0)),
                ),
                Route::Slot0 => {
                    let (status, doc) = slot_route(srv, &target, &body);
                    (label, status, doc)
                }
                Route::NotFound => (label, "404 Not Found", not_found_json(&path)),
            }
        }
    };

    let text = doc.to_string();
    eprintln!("[serve] {label} -> {status}");
    if let Err(e) = respond(stream, status, &text) {
        eprintln!("[serve] response write failed: {e}");
    }
    // half close, so the client reads EOF instead of a reset
    let _ = stream.shutdown(Shutdown::Write);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    // A3: the tokenize arm returns HERE, before the CUDA context and before Engine::load
    // takes engine/.engine.lock; it never starts a Python process
    if args.get(1).map(|s| s.as_str()) == Some("tokenize") {
        std::process::exit(tokenize_main(&args[2..]));
    }
    let cli = match parse_args(&args) {
        Ok(a) => a,
        Err(e) => {
            eprintln!("[serve] {e}");
            std::process::exit(2);
        }
    };
    // #32 review: a typo'd --slot-save-path used to be discovered at the FIRST save, after a
    // whole conversation had been prefilled. It costs one stat call to find it here.
    if let Some(d) = &cli.slot_save_path {
        if let Err(e) = check_slot_save_path(d) {
            eprintln!("[serve] {e}");
            std::process::exit(2);
        }
    }

    // #25 A3: warm up the tokenizer BEFORE the CUDA context and before Engine::load.
    // A missing or broken tokenizer must fail in a second, not after the engine is pinned.
    let (tok_path, tok_cfg) = crow_nest_engine::tokenizer::default_paths();
    match crow_nest_engine::tokenizer::global() {
        Ok(tk) => {
            let (tp, cp) = tk.paths();
            eprintln!("[serve] tokenizer {tp}");
            eprintln!("[serve] chat template {cp}");
        }
        Err(e) => {
            eprintln!("[serve] {e}");
            eprintln!("[serve] tokenizer {tok_path}");
            eprintln!("[serve] chat template {tok_cfg}");
            std::process::exit(3);
        }
    }

    // #26 review: the gated configuration is CROW_GRAPH=1 and CROW_MMA=1 (every A2..A4 gate and
    // the engine's ten-task gates ran with both on). gen.rs reads each one ONCE through a
    // OnceLock (`mma_on` gen.rs:1075, `graph_on` gen.rs:1086, `dense_mma_on` gen.rs:1156), so
    // setting them here, before cuda::Ctx::init and before any kernel call, is the whole switch.
    // Only an UNSET variable is set, so an explicit value still overrides and diagnostics stay
    // possible. serve is single threaded at this point: no other thread can read the environment.
    //
    // #37 fix round 1: CROW_ADAPT_WINDOW joins them. It is NOT a OnceLock read; gen.rs reads it
    // per tick (`window_counts` gen.rs:3048, `adapt_tick` gen.rs:3066), both of which can only
    // run inside a request, long after this point. Unset it ranks the trickle's swaps by
    // `drain_sel_counts`, the count CUMULATIVE since process start, which a 16,064 token prefill
    // dominates; the measured cost was -19.2 % against the adjacent `decode run` D1 (#37 block B).
    for key in ["CROW_GRAPH", "CROW_MMA", "CROW_ADAPT_WINDOW"] {
        if std::env::var_os(key).is_none() {
            std::env::set_var(key, "1");
        }
        eprintln!("[serve] {key}={}", std::env::var(key).unwrap_or_default());
    }

    let cnq_path = std::env::var("CROW_CNQ").unwrap_or_else(|_| DEFAULT_CNQ.into());
    let sidecar = std::env::var("CROW_HOTSETS").unwrap_or_else(|_| DEFAULT_HOTSETS.into());
    let mut cnq = Cnq::open(&cnq_path);

    // unsafe: creates the CUDA context; it must outlive every device allocation
    let _ctx = unsafe { crow_nest_engine::cuda::Ctx::init() };

    let mut cfg = Config::default();
    cfg.context = CONTEXT_FLOOR;
    // M1: chunk pinned for the process, no per prompt policy
    cfg.prompt_chunk = SERVE_CHUNK;
    apply_adapt_policy(&mut cfg);

    // unsafe: pins device and host memory; takes engine/.engine.lock, a second serve dies here
    let (eng, _rep) = unsafe {
        Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| eprintln!("[load] {m}"))
    };
    // `Cnq::open` handed the container to the loader; the engine keeps a raw
    // pointer to it (`Engine::cnq`), and prefill needs it back as `&mut`
    let n_ctx = eng.st.context;
    let prompt_chunk = eng.cfg.prompt_chunk;

    eprintln!("[serve] container {cnq_path}");
    eprintln!("[serve] hotsets {sidecar}");
    eprintln!("[serve] n_ctx {n_ctx}");
    eprintln!("[serve] prompt_chunk {prompt_chunk}");

    // #37: the "[policy] ..." line above is what apply_adapt_policy CHOSE; this line is what
    // serve DOES with it. The stream trickle is ticked once per decode_step, the mirror of
    // decode.rs:224-231. adapt_tick, the post-prefill re-cut of CROW_ADAPT=1, stays uncalled.
    let ad = eng.cfg.adapt;
    if ad.stream && ad.every > 0 && trickle_ready(&eng) {
        eprintln!(
            "[serve] #37 stream trickle ticked once per decode_step: every {}, max {}/layer, {} spare hot slot(s)",
            ad.every, ad.max, ad.spare
        );
    } else {
        eprintln!(
            "[serve] #37 stream trickle NOT ticked: stream {}, every {}, spare slots {}, exact NVFP4 tier {}",
            ad.stream,
            ad.every,
            eng.res.stride.saturating_sub(eng.res.n),
            eng.res.lb.is_none()
        );
    }
    eprintln!("[serve] adapt_tick (the CROW_ADAPT=1 hot-set re-cut) is never called by serve");

    // #32 A10: without --slot-save-path, POST /slots/0 refuses save and restore
    match &cli.slot_save_path {
        Some(d) => eprintln!("[serve] slot save path {d}"),
        None => eprintln!("[serve] no --slot-save-path, so POST /slots/0 refuses save and restore"),
    }

    let addr = format!("127.0.0.1:{}", cli.port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            eprintln!("[serve] cannot bind {addr}: {e}");
            std::process::exit(3);
        }
    };
    eprintln!("[serve] listening on http://{addr} (blocking, one request at a time)");

    let mut eng = eng;
    // #31 A9: the snapshot slot is allocated here, once, from the loaded shape (spec 7.7).
    // #36 M2b: SLOTS is 1, and the line below reads it instead of naming a count of its own.
    let cache = PrefixCache::new(&eng);
    eprintln!(
        "[serve] prefix cache {}, {} B per snapshot, {} snapshot(s), QSA ring rows {}",
        if cache.enabled() { "on" } else { "off (CROW_PREFIX_CACHE=0)" },
        cache.shape().snapshot_bytes(),
        SLOTS,
        eng.st.qsa_ring_rows
    );
    let mut srv = Srv {
        eng: &mut eng,
        cnq: &mut cnq,
        model_path: &cnq_path,
        n_ctx,
        prompt_chunk,
        seq: 0,
        parked_sampler: None,
        cache,
        slot_save_path: cli.slot_save_path.clone(),
    };
    for conn in listener.incoming() {
        match conn {
            Ok(mut s) => serve_one(&mut s, &mut srv),
            Err(e) => eprintln!("[serve] accept failed: {e}"),
        }
    }
    // #28: a parked device sampler goes back into the engine, so `Engine::drop` frees its
    // buffers (the accept loop above only ends on a listener error)
    if srv.eng.dev_sampler.is_none() {
        srv.eng.dev_sampler = srv.parked_sampler.take();
    }
    drop(srv);
    drop(eng);
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn request_line_is_method_and_target() {
        assert_eq!(
            parse_request_line("GET /props HTTP/1.1\r\n"),
            Some(("GET".into(), "/props".into()))
        );
        assert_eq!(
            parse_request_line("POST /v1/chat/completions HTTP/1.1"),
            Some(("POST".into(), "/v1/chat/completions".into()))
        );
        assert_eq!(parse_request_line(""), None);
        assert_eq!(parse_request_line("GET /props"), None);
        assert_eq!(parse_request_line("GARBAGE\r\n"), None);
        // a proxy style absolute target is not a route this server answers
        assert_eq!(parse_request_line("GET http://x/props HTTP/1.1"), None);
    }

    #[test]
    fn route_drops_query_and_trailing_slash() {
        assert_eq!(route_path("/props"), "/props");
        assert_eq!(route_path("/props?x=1"), "/props");
        assert_eq!(route_path("/health/"), "/health");
        assert_eq!(route_path("/health#frag"), "/health");
        assert_eq!(route_path("/"), "/");
    }

    #[test]
    fn dispatch_is_method_and_path() {
        // the request line goes through the same two helpers serve_one uses
        let d = |line: &str| {
            let (m, t) = parse_request_line(line).expect("request line parses");
            route(&m, route_path(&t))
        };
        assert_eq!(d("GET /health HTTP/1.1\r\n"), Route::Health);
        assert_eq!(d("GET /health/ HTTP/1.1\r\n"), Route::Health);
        assert_eq!(d("GET /props HTTP/1.1\r\n"), Route::Props);
        assert_eq!(d("GET /props?x=1 HTTP/1.1\r\n"), Route::Props);
        // unknown path
        assert_eq!(d("GET /nope HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("GET /v1/models HTTP/1.1\r\n"), Route::NotFound);
        // non GET method, on a known path and on a future one
        assert_eq!(d("POST /health HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("HEAD /props HTTP/1.1\r\n"), Route::NotFound);
        assert_eq!(d("POST /v1/chat/completions HTTP/1.1\r\n"), Route::Chat);
        assert_eq!(d("POST /v1/chat/completions?x=1 HTTP/1.1\r\n"), Route::Chat);
        // the chat path answers POST only
        assert_eq!(d("GET /v1/chat/completions HTTP/1.1\r\n"), Route::NotFound);
    }

    #[test]
    fn content_length_header_is_case_insensitive() {
        assert_eq!(header_value("Content-Length: 7\r\n", "content-length"), Some("7"));
        assert_eq!(header_value("content-length:7", "content-length"), Some("7"));
        assert_eq!(header_value("CONTENT-LENGTH:  42  \r\n", "content-length"), Some("42"));
        assert_eq!(header_value("Host: 127.0.0.1\r\n", "content-length"), None);
        assert_eq!(header_value("no colon here", "content-length"), None);
    }

    #[test]
    fn props_carries_the_fields_crow_reads() {
        let doc = props_json("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", 200_000, 2048, false);
        // server_model_path (crow_core.py:1408)
        assert_eq!(doc["model_path"], "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq");
        // fetch_model_name (crow_core.py:14884)
        assert_eq!(doc["model"], "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert!(doc["model"].as_str().unwrap().contains("Flash-Next"));
        // fetch_n_ctx (crow_core.py:14837): settings first, bare key as fallback
        assert_eq!(doc["default_generation_settings"]["n_ctx"], 200_000);
        assert_eq!(doc["n_ctx"], 200_000);
        assert!(doc["n_ctx"].as_u64().unwrap() >= 200_000);
        // refuse_images (crow_core.py:1429): false = BLIND_SERVER_HINT, not an error
        assert_eq!(doc["modalities"]["vision"], serde_json::Value::Bool(false));
        assert_eq!(doc["prompt_chunk"], 2048);
        // #VIT: with the tower loaded the same document reports vision, so
        // Crow's refuse_images lets /image and read_image through
        let doc = props_json("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq", 200_000, 2048, true);
        assert_eq!(doc["modalities"]["vision"], serde_json::Value::Bool(true));
    }

    #[test]
    fn image_blocks_are_collected_in_message_order() {
        // exactly the Crow wire form (crow_core.py user_content + image_part)
        let body = serde_json::json!({
            "messages": [
                {"role": "user", "content": [
                    {"type": "text", "text": "what is in this picture?"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/png;base64,aGVsbG8="}},
                    {"type": "text", "text": "and this one?"},
                    {"type": "image_url",
                     "image_url": {"url": "data:image/jpeg;base64,d29ybGQ="}}
                ]}
            ]
        });
        let req = parse_chat(body.to_string().as_bytes()).unwrap();
        assert_eq!(req.images.len(), 2);
        assert!(req.images[0].starts_with("data:image/png;base64,"));
        assert!(req.images[1].starts_with("data:image/jpeg;base64,"));
        // a plain string content carries no images (the bare-string contract)
        let plain = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert!(plain.images.is_empty());
    }

    #[test]
    fn data_urls_decode_to_the_original_bytes() {
        // "hello world", encoded and split across padding
        let (mime, raw) = decode_data_url("data:image/png;base64,aGVsbG8gd29ybGQ=").unwrap();
        assert_eq!(mime, "image/png");
        assert_eq!(raw, b"hello world");
        assert!(decode_data_url("https://x/y.png").is_err());
        assert!(decode_data_url("data:image/png;base64,!!!!").is_err());
        assert!(decode_data_url("data:image/png;base64,").is_err());
    }

    #[test]
    fn smart_resize_follows_the_hf_rules() {
        // exact HF smart_resize cases (factor 32, min 65536, max 16777216)
        use crow_nest_engine::vit as vv;
        let r = |h: u64, w: u64| vv::smart_resize_for_test(h, w).unwrap();
        // round to the factor, inside the window: untouched grid
        assert_eq!(r(448, 448), (448, 448));
        assert_eq!(r(384, 512), (384, 512));
        // 383x511 rounds DOWN half... 383/32 = 11.97 -> 12*32 = 384; 511/32 = 15.97 -> 16*32 = 512
        assert_eq!(r(383, 511), (384, 512));
        // banker's rounding: 496/32 = 15.5 exactly -> rounds to 16 (even)
        assert_eq!(r(496, 448), (512, 448));
        // upscale to min_pixels: 160x224 = 35840 < 65536 -> beta = sqrt(65536/35840)
        let (h, w) = r(160, 224);
        assert!(h * w >= 65536 && h % 32 == 0 && w % 32 == 0);
        // small images below the factor are refused
        assert!(vv::smart_resize_for_test(16, 1024).is_err());
    }

    #[test]
    fn model_name_is_the_container_stem() {
        assert_eq!(model_name("converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq"), "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert_eq!(model_name("C:\\x\\converter\\Qwen3.8-Flash-Next-CNQ4.5-M.cnq"), "Qwen3.8-Flash-Next-CNQ4.5-M");
        assert_eq!(model_name("plain"), "plain");
    }

    #[test]
    fn port_flag_beats_the_default() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_port(&v(&["serve"])), Ok(DEFAULT_PORT));
        assert_eq!(parse_port(&v(&["serve", "--port", "8099"])), Ok(8099));
        assert_eq!(parse_port(&v(&["serve", "--port=1234"])), Ok(1234));
        assert!(parse_port(&v(&["serve", "--port"])).is_err());
        assert!(parse_port(&v(&["serve", "--port", "no"])).is_err());
        assert!(parse_port(&v(&["serve", "-p", "1"])).is_err());
    }

    #[test]
    fn tokenize_arguments_parse_into_one_job() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(
            parse_tokenize(&v(&["--chat", "--file", "p.json", "--out", "i.json"])),
            Ok(Tok::File { chat: true, file: "p.json".into(), out: "i.json".into() })
        );
        assert_eq!(
            parse_tokenize(&v(&["--raw", "--text", "hi"])),
            Ok(Tok::Text { chat: false, text: "hi".into() })
        );
        assert_eq!(
            parse_tokenize(&v(&["--chat", "--text", "hi"])),
            Ok(Tok::Text { chat: true, text: "hi".into() })
        );
        // no mode, both modes, missing --out, both inputs, unknown flag, missing value
        assert!(parse_tokenize(&v(&["--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--raw", "--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--file", "p.json"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--file", "p.json", "--text", "hi"])).is_err());
        assert!(parse_tokenize(&v(&["--chat"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--text"])).is_err());
        assert!(parse_tokenize(&v(&["--chat", "--nope", "x"])).is_err());
    }

    #[test]
    fn both_prompt_file_shapes_give_the_same_pairs() {
        // docs/ten-task-prompts-crowlab.json is {id: text}; with serde_json's preserve_order
        // the object comes back in FILE order, not sorted by key (t2 stands first here)
        let obj = serde_json::json!({ "t2-write": "b", "t1-read": "a" });
        assert_eq!(
            prompts_from_json(&obj).unwrap(),
            vec![("t2-write".to_string(), "b".to_string()), ("t1-read".to_string(), "a".to_string())]
        );
        // the same shape parsed from bytes, so this pins the parser, not just the json! macro
        let parsed: serde_json::Value =
            serde_json::from_str(r#"{"t2-write": "b", "t1-read": "a"}"#).unwrap();
        assert_eq!(
            prompts_from_json(&parsed).unwrap(),
            vec![("t2-write".to_string(), "b".to_string()), ("t1-read".to_string(), "a".to_string())]
        );
        // decode_out/ten-tasks.json is the parity harness array
        let arr = serde_json::json!([
            { "id": "t1-read", "text": "a", "max_tokens": 1024 },
            { "id": "t2-write", "text": "b", "max_tokens": 1280 }
        ]);
        assert_eq!(
            prompts_from_json(&arr).unwrap(),
            vec![("t1-read".to_string(), "a".to_string()), ("t2-write".to_string(), "b".to_string())]
        );
        assert!(prompts_from_json(&serde_json::json!([{ "text": "a" }])).is_err());
        assert!(prompts_from_json(&serde_json::json!({ "t": 1 })).is_err());
        assert!(prompts_from_json(&serde_json::json!("nope")).is_err());
    }

    #[test]
    fn unknown_route_is_a_404_document() {
        let doc = not_found_json("/v1/models");
        assert_eq!(doc["error"]["code"], 404);
        assert!(doc["error"]["message"].as_str().unwrap().contains("/v1/models"));
    }

    /// review finding: the 404 body listed three of the five routes this server answers
    #[test]
    fn the_404_body_lists_every_route_this_server_answers() {
        let msg = not_found_json("/v1/models")["error"]["message"].as_str().unwrap().to_string();
        for route in [
            "GET /health",
            "GET /props",
            "POST /v1/chat/completions",
            "GET /slots",
            "POST /slots/0",
        ] {
            assert!(msg.contains(route), "the 404 body does not name {route:?}: {msg}");
        }
    }

    #[test]
    fn flat_error_documents_name_the_reason() {
        assert_eq!(error_json("bad request")["error"], "bad request");
        assert_eq!(error_json("payload too large")["error"], "payload too large");
    }
    // ------------------------------------------------ chat completions (#26 A4)

    #[test]
    fn chat_body_parses_messages_max_tokens_and_stream() {
        // the body Crow sends (crow_core.py:4672-4700), trimmed to what A4 reads
        let raw = br#"{"model":"crow-nest",
            "messages":[{"role":"user","content":"Say the word ready."}],
            "temperature":0,"top_p":0.95,"min_p":0.01,"stream":true,
            "stream_options":{"include_usage":true},"timings_per_token":true,
            "max_tokens":16}"#;
        let r = parse_chat(raw).expect("the body parses");
        assert_eq!(r.model, "crow-nest");
        assert_eq!(r.max_tokens, 16);
        assert!(r.stream);
        assert!(!r.enable_thinking);
        assert_eq!(r.messages[0]["content"], "Say the word ready.");

        // defaults: no stream, no max_tokens, no model
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(r.max_tokens, DEFAULT_MAX_TOKENS);
        assert!(!r.stream);
        assert_eq!(r.model, "crow-nest");

        // the digest path sends chat_template_kwargs (crow_core.py:2969)
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "chat_template_kwargs":{"enable_thinking":true}}"#,
        )
        .unwrap();
        assert!(r.enable_thinking);

        // max_tokens is capped, never trusted
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":999999}"#)
            .unwrap();
        assert_eq!(r.max_tokens, MAX_MAX_TOKENS);

        // tools are accepted and ignored; top_k is read since #28
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "tools":[{"type":"function","function":{"name":"read"}}],"top_k":20}"#,
        )
        .unwrap();
        assert_eq!(r.max_tokens, DEFAULT_MAX_TOKENS);

        // rejections
        assert!(parse_chat(b"not json").is_err());
        assert!(parse_chat(b"[]").is_err());
        assert!(parse_chat(br#"{"messages":[]}"#).is_err());
        assert!(parse_chat(br#"{"model":"x"}"#).is_err());
        assert!(parse_chat(br#"{"messages":"hi"}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"content":"hi"}]}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"stream":"yes"}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":0}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"max_tokens":-1}"#).is_err());
    }

    // ------------------------------------------------------- sampling (#28 A6)

    #[test]
    fn the_sampling_fields_parse_with_the_data_sheet_defaults() {
        // Crow's operating point, whole: every field present
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":1.0,"top_p":0.95,"min_p":0.01,"top_k":40,
                 "presence_penalty":0.5,"seed":7}"#,
        )
        .unwrap();
        assert_eq!(r.temperature, 1.0);
        assert_eq!(r.top_p, 0.95);
        assert_eq!(r.min_p, 0.01);
        assert_eq!(r.top_k, 40);
        assert_eq!(r.presence_penalty, 0.5);
        assert_eq!(r.seed, 7);

        // nothing present: the data sheet, a fixed seed, and greedy
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}]}"#).unwrap();
        assert_eq!(r.temperature, 0.0);
        assert_eq!(r.top_p, DEFAULT_TOP_P);
        assert_eq!(r.top_k, DEFAULT_TOP_K);
        assert_eq!(r.presence_penalty, DEFAULT_PRESENCE);
        assert_eq!(r.seed, DEFAULT_SEED);
        assert_eq!(r.min_p, 0.0);

        // an explicit null is an absent field, not a 400
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":null,"top_p":null,"top_k":null,"seed":null,"min_p":null}"#,
        )
        .unwrap();
        assert_eq!(r.top_p, DEFAULT_TOP_P);
        assert_eq!(r.top_k, DEFAULT_TOP_K);
        assert_eq!(r.seed, DEFAULT_SEED);

        // llama-server's "pick a seed" is taken as a value, never as a draw (M1: determinism)
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"seed":-1}"#).unwrap();
        assert_eq!(r.seed, u64::MAX);

        // a wrong type is a 400: the caller named a profile it would not have got
        let bad = [
            &br#"{"messages":[{"role":"user","content":"hi"}],"temperature":"hot"}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"top_p":"wide"}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"top_k":"many"}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"top_k":-3}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"seed":"lucky"}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"min_p":"low"}"#[..],
            &br#"{"messages":[{"role":"user","content":"hi"}],"presence_penalty":"some"}"#[..],
        ];
        for b in bad {
            assert!(parse_chat(b).is_err(), "expected a 400 for {}", String::from_utf8_lossy(b));
        }
    }

    #[test]
    fn greedy_is_absent_zero_and_negative_temperature() {
        let mk = |t: &str| {
            let body = format!(
                r#"{{"messages":[{{"role":"user","content":"hi"}}]{t}}}"#
            );
            parse_chat(body.as_bytes()).unwrap()
        };
        // greedy: the A4 path, no sampler is built at all
        assert!(sampler_from(&mk("")).is_none());
        assert!(sampler_from(&mk(r#","temperature":0"#)).is_none());
        assert!(sampler_from(&mk(r#","temperature":0.0"#)).is_none());
        assert!(sampler_from(&mk(r#","temperature":null"#)).is_none());
        assert!(sampler_from(&mk(r#","temperature":-1.0"#)).is_none());
        // greedy stays greedy even when the rest of the profile is sent
        assert!(sampler_from(&mk(r#","temperature":0,"top_p":0.95,"min_p":0.01,"seed":7"#)).is_none());
        // any positive temperature samples
        assert!(sampler_from(&mk(r#","temperature":0.0001"#)).is_some());
        assert!(sampler_from(&mk(r#","temperature":1.0"#)).is_some());
    }

    #[test]
    fn the_sampler_carries_the_request_seed_and_the_data_sheet_rest() {
        // only temperature and seed sent: everything else is the data sheet
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],"temperature":1.0,"seed":7}"#,
        )
        .unwrap();
        let s = sampler_from(&r).expect("temperature 1.0 samples");
        assert_eq!(s.seed, 7);
        assert_eq!(s.temperature, 1.0);
        assert_eq!(s.top_p, DEFAULT_TOP_P);
        assert_eq!(s.top_k, DEFAULT_TOP_K);
        assert_eq!(s.presence_penalty, DEFAULT_PRESENCE);
        // the RNG state is the seed's, so two requests with the same seed upload the same state
        assert_eq!(s.rng.state(), Sampler::new(7).rng.state());
        assert_ne!(s.rng.state(), Sampler::new(8).rng.state());

        // what the request does send wins over the data sheet
        let r = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":1.0,"top_p":0.95,"top_k":40,"presence_penalty":0.0,"seed":1}"#,
        )
        .unwrap();
        let s = sampler_from(&r).unwrap();
        assert_eq!((s.top_p, s.top_k, s.presence_penalty, s.seed), (0.95, 40, 0.0, 1));
    }

    #[test]
    fn the_budget_is_clamped_to_the_free_context_and_413_only_without_one() {
        // the normal case: the whole budget fits, nothing is touched
        assert_eq!(clamped_max_tokens(18, 64, 200_000), Some(64));
        // the prompt eats most of the context: the budget shrinks to what is left
        assert_eq!(clamped_max_tokens(199_990, 64, 200_000), Some(10));
        assert_eq!(clamped_max_tokens(1, 1024, 2), Some(1));
        // the 32768 cap still binds when the free context is larger
        assert_eq!(clamped_max_tokens(18, 999_999, 200_000), Some(MAX_MAX_TOKENS));
        assert_eq!(clamped_max_tokens(18, MAX_MAX_TOKENS, 200_000), Some(MAX_MAX_TOKENS));
        // exactly one token of room, and exactly none
        assert_eq!(clamped_max_tokens(199_999, 64, 200_000), Some(1));
        assert_eq!(clamped_max_tokens(200_000, 64, 200_000), None);
        assert_eq!(clamped_max_tokens(200_001, 1, 200_000), None);
        // the clamp is never 0 when it returns Some: a served request always has a budget
        for prompt in [0usize, 1, 17, 199_999] {
            let n = clamped_max_tokens(prompt, DEFAULT_MAX_TOKENS, 200_000).unwrap();
            assert!(n >= 1, "prompt {prompt} got budget {n}");
            assert!(prompt + n <= 200_000);
        }
    }

    #[test]
    fn a_chunk_carries_exactly_what_crow_reads() {
        // crow_core.py:4846 finish_reason, :4877 delta.content
        let role = chunk_role("chatcmpl-1", 1_757_000_000, "crow-nest");
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["id"], "chatcmpl-1");
        assert_eq!(role["created"], 1_757_000_000u64);
        assert_eq!(role["model"], "crow-nest");
        assert_eq!(role["choices"][0]["index"], 0);
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], serde_json::Value::Null);
        assert!(role["choices"][0]["delta"].get("content").is_none());

        let c = chunk_content("chatcmpl-1", 1, "crow-nest", " ready");
        assert_eq!(c["choices"][0]["delta"]["content"], " ready");
        assert_eq!(c["choices"][0]["finish_reason"], serde_json::Value::Null);

        // the last chunk: empty delta, a finish reason, and only ONE of them exists
        let t = T0;
        let f = chunk_finish("chatcmpl-1", 1, "crow-nest", "stop", &t, false, false);
        assert_eq!(f["choices"][0]["finish_reason"], "stop");
        assert_eq!(f["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(
            chunk_finish("i", 1, "m", "length", &t, false, false)["choices"][0]["finish_reason"],
            "length"
        );
    }

    /// the numbers of one served request, so every A5 test speaks about the same turn
    const T0: Timing = Timing {
        prompt_n: 16_064,
        cached_n: 0,
        predicted_n: 8,
        prompt_ms: 24_500.0,
        predicted_ms: 320.0,
        selections_total: 7_710_720,
        cold_total: 2_534_400,
        ple_rows_total: 160_640,
        ple_miss_total: 12_811,
    };

    #[test]
    fn without_the_two_flags_the_final_chunk_is_the_a4_chunk() {
        let f = chunk_finish("id", 7, "m", "stop", &T0, false, false);
        assert!(f.get("usage").is_none());
        assert!(f.get("timings").is_none());
        // nothing else moved either: the object is exactly what A4 sent
        assert_eq!(
            f,
            chunk("id", 7, "m", serde_json::json!({}), Some("stop")),
        );
        // one flag at a time carries one object at a time
        let u = chunk_finish("id", 7, "m", "stop", &T0, true, false);
        assert!(u.get("usage").is_some());
        assert!(u.get("timings").is_none());
        let t = chunk_finish("id", 7, "m", "stop", &T0, false, true);
        assert!(t.get("usage").is_none());
        assert!(t.get("timings").is_some());
    }

    #[test]
    fn the_final_chunk_carries_the_eight_fields_crow_reads() {
        // crow_core.py:4838-4845 (usage) and :4999-5018 (timings)
        let f = chunk_finish("id", 7, "m", "stop", &T0, true, true);
        // the finish reason did not move: Crow reads it off the SAME chunk
        assert_eq!(f["choices"][0]["finish_reason"], "stop");

        let u = &f["usage"];
        assert_eq!(u["prompt_tokens"].as_u64(), Some(16_064));
        assert_eq!(u["completion_tokens"].as_u64(), Some(8));
        assert_eq!(u["total_tokens"].as_u64(), Some(16_072));
        // the equality the gate checks, from the object itself, not from the inputs
        assert_eq!(
            u["total_tokens"].as_u64().unwrap(),
            u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
        );
        // PRESENT as 0, never missing: a missing one makes Crow fall back to prompt_n
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"].as_i64(), Some(0));

        let g = &f["timings"];
        assert_eq!(g["prompt_n"].as_u64(), Some(16_064));
        assert_eq!(g["predicted_n"].as_u64(), Some(8));
        assert_eq!(g["cache_n"].as_i64(), Some(0));
        // ms and rates are floats, counts are ints
        for k in ["prompt_ms", "predicted_ms", "prompt_per_second", "predicted_per_second",
                  "prompt_per_token_ms", "predicted_per_token_ms"] {
            assert!(g[k].is_f64(), "{k} is not a float: {}", g[k]);
        }
        for k in ["prompt_n", "predicted_n", "cache_n"] {
            assert!(g[k].is_i64() || g[k].is_u64(), "{k} is not an int: {}", g[k]);
        }
        assert_eq!(g["prompt_ms"].as_f64(), Some(24_500.0));
        assert_eq!(g["predicted_ms"].as_f64(), Some(320.0));
        assert_eq!(g["prompt_per_second"].as_f64(), Some(round3(16_064.0 * 1000.0 / 24_500.0)));
        assert_eq!(g["predicted_per_second"].as_f64(), Some(25.0));
        assert_eq!(g["predicted_per_token_ms"].as_f64(), Some(40.0));
    }

    /// #31 A9: the SAME turn served warm. 16,064 prompt ids, 15,000 of them reused.
    const T_WARM: Timing = Timing {
        prompt_n: 1_064,
        cached_n: 15_000,
        predicted_n: 8,
        prompt_ms: 1_700.0,
        predicted_ms: 320.0,
        selections_total: 7_710_720,
        cold_total: 2_534_400,
        ple_rows_total: 160_640,
        ple_miss_total: 12_811,
    };

    /// #31 A9: `prompt_tokens` stays the WHOLE prompt, `cached_tokens` is P, `prompt_n` the rest
    #[test]
    fn a_warm_turn_splits_the_prompt_into_cached_and_prefilled() {
        let f = chunk_finish("id", 7, "m", "stop", &T_WARM, true, true);
        let u = &f["usage"];
        // the same 16,064 token prompt as the cold turn: the client's accounting cannot move
        assert_eq!(u["prompt_tokens"].as_u64(), Some(16_064));
        assert_eq!(u["prompt_tokens"].as_u64(), usage_json(&T0)["prompt_tokens"].as_u64());
        assert_eq!(u["prompt_tokens_details"]["cached_tokens"].as_u64(), Some(15_000));
        assert_eq!(u["completion_tokens"].as_u64(), Some(8));
        assert_eq!(u["total_tokens"].as_u64(), Some(16_072));
        assert_eq!(
            u["total_tokens"].as_u64().unwrap(),
            u["prompt_tokens"].as_u64().unwrap() + u["completion_tokens"].as_u64().unwrap()
        );
        let g = &f["timings"];
        // what was PREFILLED, not what was rendered: the A9 gate reads exactly this pair
        assert_eq!(g["prompt_n"].as_u64(), Some(1_064));
        assert_eq!(g["cache_n"].as_u64(), Some(15_000));
        assert_eq!(
            g["cache_n"].as_u64(),
            u["prompt_tokens_details"]["cached_tokens"].as_u64()
        );
        assert_eq!(
            g["prompt_n"].as_u64().unwrap() + g["cache_n"].as_u64().unwrap(),
            u["prompt_tokens"].as_u64().unwrap()
        );
        // the gate thresholds, computed from the wire object alone
        let p = u["prompt_tokens"].as_u64().unwrap() as f64;
        assert!(u["prompt_tokens_details"]["cached_tokens"].as_u64().unwrap() as f64 >= 0.93 * p);
        assert!(g["prompt_n"].as_u64().unwrap() as f64 <= 0.07 * p);
        // the rate belongs to the PREFILLED tokens, so it is 1064 over 1700 ms
        assert_eq!(g["prompt_per_second"].as_f64(), Some(round3(1_064.0 * 1000.0 / 1_700.0)));
        // nothing crept in and nothing left: still the nine A5 keys plus the five A8 ones
        assert_eq!(g.as_object().unwrap().len(), 14, "{g}");
    }

    /// #31 A9: both keys are integers on every path; a missing one changes what Crow counts
    #[test]
    fn the_cache_fields_are_present_as_integers_warm_and_cold() {
        for t in [&T0, &T_WARM] {
            let f = chunk_finish("id", 7, "m", "stop", t, true, true);
            let c = &f["usage"]["prompt_tokens_details"]["cached_tokens"];
            let n = &f["timings"]["cache_n"];
            assert!(c.is_u64() || c.is_i64(), "cached_tokens is not an int: {c}");
            assert!(n.is_u64() || n.is_i64(), "cache_n is not an int: {n}");
            assert!(!c.is_f64() && !n.is_f64(), "a float reached the wire: {c} {n}");
            assert_eq!(c.as_u64(), Some(t.cached_n as u64));
            assert_eq!(n.as_u64(), Some(t.cached_n as u64));
        }
    }

    /// #31 A9: a cold turn is byte for byte the A5 object, so nothing regressed
    #[test]
    fn a_cold_turn_is_the_a5_usage_object_unchanged() {
        let u = usage_json(&T0);
        assert_eq!(
            u,
            serde_json::json!({
                "prompt_tokens": 16_064,
                "completion_tokens": 8,
                "total_tokens": 16_072,
                "prompt_tokens_details": { "cached_tokens": 0 },
            })
        );
    }

    #[test]
    fn the_rates_are_n_over_ms_and_never_a_nan() {
        assert_eq!(per_second(16_064, 24_500.0), 16_064.0 * 1000.0 / 24_500.0);
        assert_eq!(per_second(8, 320.0), 25.0);
        assert_eq!(per_token_ms(8, 320.0), 40.0);
        assert_eq!(per_token_ms(16_064, 24_500.0), 24_500.0 / 16_064.0);
        // the degenerate turns: an aborted request, a request that generated nothing
        assert_eq!(per_second(0, 0.0), 0.0);
        assert_eq!(per_second(8, 0.0), 0.0);
        assert_eq!(per_second(8, -1.0), 0.0);
        assert_eq!(per_second(8, f64::NAN), 0.0);
        assert_eq!(per_second(8, f64::INFINITY), 0.0);
        assert_eq!(per_token_ms(0, 320.0), 0.0);
        assert_eq!(per_token_ms(8, f64::NAN), 0.0);
        assert_eq!(round3(f64::NAN), 0.0);
        assert_eq!(round3(1.23456), 1.235);
        // a zero wall must still leave a NUMBER on the wire, serde turns NaN into null
        let z = Timing {
            prompt_n: 0,
            cached_n: 0,
            predicted_n: 0,
            prompt_ms: 0.0,
            predicted_ms: 0.0,
            selections_total: 0,
            cold_total: 0,
            ple_rows_total: 0,
            ple_miss_total: 0,
        };
        let f = chunk_finish("id", 7, "m", "stop", &z, true, true);
        for k in ["prompt_ms", "prompt_per_second", "predicted_per_second", "predicted_per_token_ms"] {
            assert_eq!(f["timings"][k].as_f64(), Some(0.0), "{k} is {}", f["timings"][k]);
        }
        assert_eq!(f["usage"]["total_tokens"].as_u64(), Some(0));
        assert!(!f["timings"].to_string().contains("null"), "{}", f["timings"]);
    }

    /// #30 A8: the five keys, their exact names, and u64 (never a float)
    #[test]
    fn the_timings_block_carries_the_engine_counters_as_u64() {
        let g = &chunk_finish("id", 7, "m", "stop", &T0, true, true)["timings"];
        // the names are the contract: a renamed key silently breaks every difference reader
        assert_eq!(g["crow_expert_selections"].as_u64(), Some(7_710_720));
        assert_eq!(g["crow_expert_cold"].as_u64(), Some(2_534_400));
        assert_eq!(g["crow_ple_rows"].as_u64(), Some(160_640));
        assert_eq!(g["crow_ple_misses"].as_u64(), Some(12_811));
        assert_eq!(g["crow_layers"].as_u64(), Some(LAYERS as u64));
        // counts, not rates: a float here would round the atomicAdd totals away
        for k in ["crow_expert_selections", "crow_expert_cold", "crow_ple_rows",
                  "crow_ple_misses", "crow_layers"] {
            assert!(g[k].is_u64(), "{k} is not a u64: {}", g[k]);
            assert!(!g[k].is_f64(), "{k} came out as a float: {}", g[k]);
        }
        // the counters carry no llama-server key name, so no reader of the six A5 fields collides
        for k in ["prompt_n", "prompt_ms", "prompt_per_second", "prompt_per_token_ms",
                  "predicted_n", "predicted_ms", "predicted_per_second",
                  "predicted_per_token_ms", "cache_n"] {
            assert!(g[k].is_number(), "A5 key {k} moved: {}", g[k]);
        }
        // exactly fourteen keys: the nine of A5 plus the five of A8, nothing crept in
        assert_eq!(g.as_object().unwrap().len(), 14, "{g}");
    }

    /// #30 A8: cumulative means the builder never subtracts and never resets
    #[test]
    fn the_counters_are_passed_through_unchanged_and_only_live_in_timings() {
        // no flag at all: the counters are NOT on the chunk, the A4 shape is untouched
        let f = chunk_finish("id", 7, "m", "stop", &T0, false, false);
        assert!(!f.to_string().contains("crow_expert"), "{f}");
        // include_usage alone: `usage` carries none of them either
        let u = chunk_finish("id", 7, "m", "stop", &T0, true, false);
        assert!(u.get("timings").is_none());
        assert!(!u.to_string().contains("crow_"), "{u}");
        // timings on: the value on the wire is the value the engine read, byte for byte
        let g = &chunk_finish("id", 7, "m", "stop", &T0, false, true)["timings"];
        assert_eq!(g["crow_expert_selections"].as_u64(), Some(T0.selections_total));
        assert_eq!(g["crow_expert_cold"].as_u64(), Some(T0.cold_total));
        assert_eq!(g["crow_ple_rows"].as_u64(), Some(T0.ple_rows_total));
        assert_eq!(g["crow_ple_misses"].as_u64(), Some(T0.ple_miss_total));
        // two consecutive blocks: the request-local value is their difference, computed by
        // the READER. Same Timing twice = a difference of 0, never a reset to 0.
        let mut t1 = T0;
        t1.selections_total += 30_720;
        t1.cold_total += 9_920;
        let a = timings_json(&T0);
        let b = timings_json(&t1);
        assert_eq!(
            b["crow_expert_selections"].as_u64().unwrap()
                - a["crow_expert_selections"].as_u64().unwrap(),
            30_720
        );
        assert_eq!(
            b["crow_expert_cold"].as_u64().unwrap() - a["crow_expert_cold"].as_u64().unwrap(),
            9_920
        );
        // a counter that did not move gives 0, which is a valid difference, not a missing key
        assert_eq!(
            b["crow_ple_rows"].as_u64().unwrap() - a["crow_ple_rows"].as_u64().unwrap(),
            0
        );
        // 48 layers: the divisor a reader needs to turn selections into per-layer per-token
        assert_eq!(b["crow_layers"].as_u64(), Some(48));
    }

    /// #30 A8: a u64 near the top of the range survives serde and the reader's parse
    #[test]
    fn a_large_counter_stays_exact_on_the_wire() {
        let mut t = T0;
        t.selections_total = u64::MAX;
        t.ple_rows_total = 9_007_199_254_740_993; // 2^53 + 1, the first f64 cannot hold
        let text = timings_json(&t).to_string();
        let back: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(back["crow_expert_selections"].as_u64(), Some(u64::MAX));
        assert_eq!(back["crow_ple_rows"].as_u64(), Some(9_007_199_254_740_993));
        assert!(text.contains("18446744073709551615"), "{text}");
    }

    #[test]
    fn the_two_stream_flags_parse_out_of_the_body_crow_sends() {
        // the body Crow builds (crow_core.py:4672-4700), trimmed to the A5 fields
        let r = parse_chat(
            br#"{"model":"crow-nest","messages":[{"role":"user","content":"hi"}],
                 "stream":true,"stream_options":{"include_usage":true},"timings_per_token":true}"#,
        )
        .unwrap();
        assert!(r.include_usage);
        assert!(r.timings_per_token);

        // absent means off, and that is the A4 stream
        let r = parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"stream":true}"#).unwrap();
        assert!(!r.include_usage);
        assert!(!r.timings_per_token);

        // explicit false, an empty stream_options, a null, a wrong type: all off, never a 400
        for body in [
            &br#"{"messages":[{"role":"user","content":"x"}],"stream_options":{"include_usage":false},"timings_per_token":false}"#[..],
            &br#"{"messages":[{"role":"user","content":"x"}],"stream_options":{},"timings_per_token":null}"#[..],
            &br#"{"messages":[{"role":"user","content":"x"}],"stream_options":null,"timings_per_token":"yes"}"#[..],
            &br#"{"messages":[{"role":"user","content":"x"}],"stream_options":"nope","timings_per_token":1}"#[..],
        ] {
            let r = parse_chat(body).unwrap();
            assert!(!r.include_usage, "include_usage true for {}", String::from_utf8_lossy(body));
            assert!(!r.timings_per_token, "timings_per_token true for {}", String::from_utf8_lossy(body));
        }
    }

    #[test]
    fn an_sse_frame_is_one_data_line_and_a_blank_line() {
        let f = sse_frame(&chunk_content("id", 7, "m", "hi"));
        assert!(f.starts_with("data: {"));
        assert!(f.ends_with("\n\n"));
        // exactly one event: one `data:` line, then the terminator
        assert_eq!(f.matches("data: ").count(), 1);
        assert_eq!(f.trim_end_matches('\n').matches('\n').count(), 0);
        // a newline inside the content is escaped by the JSON writer, never raw
        let f = sse_frame(&chunk_content("id", 7, "m", "a\nb"));
        assert!(f.contains(r#""content":"a\nb""#));
        assert_eq!(f.trim_end_matches('\n').matches('\n').count(), 0);
        // the closing line of every stream
        assert_eq!(SSE_DONE, "data: [DONE]\n\n");
    }

    #[test]
    fn an_incomplete_utf8_tail_is_held_back() {
        // "Gr" = 2 bytes, "u umlaut" = 2 bytes: a token split inside it decodes
        // to a replacement character until the second byte arrives
        assert_eq!(next_delta("Gr\u{FFFD}", 0), None);
        assert_eq!(next_delta("Gr\u{00FC}", 0), Some("Gr\u{00FC}"));
        // the next token continues the same string; only the new suffix goes out
        assert_eq!(next_delta("Gr\u{00FC}\u{00DF}e", 4), Some("\u{00DF}e"));
        // nothing new, nothing sent
        assert_eq!(next_delta("Gr\u{00FC}", 4), None);
        // never split a character, even if the caller asks for it
        assert_eq!(next_delta("Gr\u{00FC}", 3), None);
    }

    #[test]
    fn a_split_multibyte_token_never_reaches_a_chunk() {
        // the real tokenizer, no GPU: an emoji is one 4 byte character whose
        // byte level tokens can end mid sequence
        // tests run from engine/, the model lives at the repository root
        let t = "../models/Qwen3.8-Flash-Next-original/tokenizer.json";
        let tk = crow_nest_engine::tokenizer::ChatTokenizer::load(
            t,
            &t.replace("tokenizer.json", "tokenizer_config.json"),
        )
        .expect("tokenizer loads");
        let ids = tk.encode_raw("\u{1F985}\u{1F985}").expect("encode");
        assert!(ids.len() >= 2, "the emoji pair must be more than one token: {ids:?}");
        let mut emitted = 0usize;
        let mut sent = String::new();
        let mut held = 0usize;
        for n in 1..=ids.len() {
            let full = tk.decode(&ids[..n]).expect("decode");
            match next_delta(&full, emitted) {
                Some(d) => {
                    assert!(
                        !d.contains(char::REPLACEMENT_CHARACTER),
                        "chunk {d:?} carries U+FFFD"
                    );
                    sent.push_str(d);
                    emitted = full.len();
                }
                None => held += 1,
            }
        }
        // at least one token was held back, and the text still arrived whole
        assert!(held > 0, "no token was held back, the split was never exercised");
        assert_eq!(sent, "\u{1F985}\u{1F985}");
    }

    // ------------------------------------------------ tool calls (#29 A7)

    /// the declaration shape Crow's `_fn` builds (`crow_core.py:569-574`), two parameters
    /// of two different declared types so both value paths are exercised
    fn a7_tools() -> serde_json::Value {
        serde_json::json!([{
            "type": "function",
            "function": {
                "name": "read_file",
                "description": "Read a UTF-8 text file.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "path": { "type": "string", "description": "Path to the file." },
                        "start_line": { "type": "integer", "description": "First line, 1-based." }
                    },
                    "required": ["path"]
                }
            }
        }])
    }

    #[test]
    fn a_tool_chunk_carries_exactly_what_crow_reassembles() {
        let open = chunk_tool_open("chatcmpl-1", 7, "crow-nest", 0, "call_0", "read_file");
        let c = &open["choices"][0];
        assert_eq!(c["index"], 0);
        assert_eq!(c["finish_reason"], serde_json::Value::Null);
        let call = &c["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "read_file");
        assert_eq!(call["function"]["arguments"], "");

        let frag = chunk_tool_args("chatcmpl-1", 7, "crow-nest", 0, "{\"path\":\"");
        let call = &frag["choices"][0]["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["function"]["arguments"], "{\"path\":\"");
        // Crow overwrites id and name on a truthy value only; neither key may be here at all
        assert!(call.get("id").is_none(), "an arguments fragment must not carry id");
        assert!(
            call["function"].get("name").is_none(),
            "an arguments fragment must not carry name"
        );
        // and no content key, so the reader never mixes the two paths
        assert!(frag["choices"][0]["delta"].get("content").is_none());
    }

    #[test]
    fn tools_and_tool_turns_parse_out_of_the_body_crow_sends() {
        // the whole shape of a second round: tools, an assistant turn with a call, a result
        let body = br##"{
            "model": "crow-nest",
            "messages": [
                {"role": "user", "content": "Read a.md"},
                {"role": "assistant", "content": "",
                 "tool_calls": [{"id": "call_0", "type": "function",
                                 "function": {"name": "read_file",
                                              "arguments": "{\"path\": \"a.md\"}"}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "# Title"}
            ],
            "tools": [{"type": "function",
                       "function": {"name": "read_file",
                                    "parameters": {"type": "object",
                                                   "properties": {"path": {"type": "string"}}}}}],
            "stream": true, "temperature": 0
        }"##;
        let r = parse_chat(body).expect("the body parses");
        let tools = r.tools.as_ref().expect("tools survive");
        assert_eq!(tools.as_array().map(|a| a.len()), Some(1));
        assert_eq!(tools[0]["function"]["name"], "read_file");
        assert_eq!(r.messages.as_array().map(|a| a.len()), Some(3));
        assert_eq!(r.messages[2]["role"], "tool");
        assert_eq!(r.messages[2]["tool_call_id"], "call_0");

        // absent and null give None, an empty array survives (the template reads it as falsy)
        let none = parse_chat(br#"{"messages":[{"role":"user","content":"x"}],"stream":true}"#)
            .expect("no tools");
        assert_eq!(none.tools, None);
        let null =
            parse_chat(br#"{"messages":[{"role":"user","content":"x"}],"tools":null}"#).unwrap();
        assert_eq!(null.tools, None);
        let empty =
            parse_chat(br#"{"messages":[{"role":"user","content":"x"}],"tools":[]}"#).unwrap();
        assert_eq!(empty.tools, Some(serde_json::json!([])));

        // a wrong shape is a 400, never a silent render without tools
        for bad in [
            &br#"{"messages":[{"role":"user","content":"x"}],"tools":{}}"#[..],
            &br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"type":"function"}]}"#[..],
            &br#"{"messages":[{"role":"user","content":"x"}],"tools":[{"function":{}}]}"#[..],
        ] {
            assert!(parse_chat(bad).is_err(), "accepted {}", String::from_utf8_lossy(bad));
        }
    }

    #[test]
    fn an_arguments_string_becomes_the_object_the_template_needs() {
        let msgs = serde_json::json!([
            {"role": "user", "content": "Read a.md"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_0", "type": "function",
                 "function": {"name": "read_file", "arguments": "{\"path\": \"a.md\"}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_0", "content": "# Title"}
        ]);
        let n = normalize_messages(&msgs);
        assert_eq!(n[1]["tool_calls"][0]["function"]["arguments"], serde_json::json!({"path": "a.md"}));
        // everything else is untouched, byte for byte
        assert_eq!(n[0], msgs[0]);
        assert_eq!(n[2], msgs[2]);
        assert_eq!(n[1]["tool_calls"][0]["id"], "call_0");

        // an object stays an object, and a string that is not a JSON object stays a string,
        // so the render fails loudly instead of inventing arguments
        let already = serde_json::json!([{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": {"a": 1}}}]}]);
        assert_eq!(normalize_messages(&already), already);
        let junk = serde_json::json!([{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "not json"}}]}]);
        assert_eq!(normalize_messages(&junk), junk);
        let scalar = serde_json::json!([{"role": "assistant", "tool_calls": [
            {"function": {"name": "f", "arguments": "3"}}]}]);
        assert_eq!(normalize_messages(&scalar), scalar);
    }

    /// Provenance of `ORACLE_RENDER`: run once on 2026-09-09 with
    ///   `.venv-oracle/Scripts/python.exe` (CPython 3.13.3), `PYTHONIOENCODING=utf-8`,
    ///   transformers 5.16.1,
    ///   `AutoTokenizer.from_pretrained("models/Qwen3.8-Flash-Next-original")`
    ///   `.apply_chat_template(HIST, tools=TOOLS, add_generation_prompt=True,`
    ///   `    tokenize=False, enable_thinking=False)`
    ///   with `TOOLS` = `a7_tools()` and `HIST` the three messages below, the assistant
    ///   turn carrying `arguments` as the OBJECT `{"path": "a.md", "start_line": 1}`.
    /// Measured in the same run: the same call with `arguments` as the STRING
    ///   `"{\"path\": \"a.md\", \"start_line\": 1}"` raises
    ///   `TypeError: Can only get item pairs from a mapping`, because the template
    ///   iterates `tool_call.arguments|items`. That is why `normalize_messages` exists.
    #[test]
    fn a_history_with_a_tool_turn_renders_byte_identical_to_the_oracle() {
        let t = "../models/Qwen3.8-Flash-Next-original/tokenizer.json";
        let tk = crow_nest_engine::tokenizer::ChatTokenizer::load(
            t,
            &t.replace("tokenizer.json", "tokenizer_config.json"),
        )
        .expect("tokenizer loads");
        const ORACLE_RENDER: &str = concat!(
            "<|im_start|>system\n",
            "# Tools\n",
            "\n",
            "You have access to the following functions:\n",
            "\n",
            "<tools>\n",
            "{\"type\": \"function\", \"function\": {\"name\": \"read_file\", \"description\": \"Read a UTF-8 text file.\", \"parameters\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\", \"description\": \"Path to the file.\"}, \"start_line\": {\"type\": \"integer\", \"description\": \"First line, 1-based.\"}}, \"required\": [\"path\"]}}}\n",
            "</tools>\n",
            "\n",
            "If you choose to call a function ONLY reply in the following format with NO suffix:\n",
            "\n",
            "<tool_call>\n",
            "<function=example_function_name>\n",
            "<parameter=example_parameter_1>\n",
            "value_1\n",
            "</parameter>\n",
            "<parameter=example_parameter_2>\n",
            "This is the value for the second parameter\n",
            "that can span\n",
            "multiple lines\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call>\n",
            "\n",
            "<IMPORTANT>\n",
            "Reminder:\n",
            "- Function calls MUST follow the specified format: an inner <function=...></function> block must be nested within <tool_call></tool_call> XML tags\n",
            "- Required parameters MUST be specified\n",
            "- You may provide optional reasoning for your function call in natural language BEFORE the function call, but NOT after\n",
            "- If there is no function call available, answer the question like normal with your current knowledge and do not tell the user about function calls\n",
            "</IMPORTANT><|im_end|>\n",
            "<|im_start|>user\n",
            "Read the first lines of a.md and tell me its title.<|im_end|>\n",
            "<|im_start|>assistant\n",
            "<think>\n",
            "\n",
            "</think>\n",
            "\n",
            "<tool_call>\n",
            "<function=read_file>\n",
            "<parameter=path>\n",
            "a.md\n",
            "</parameter>\n",
            "<parameter=start_line>\n",
            "1\n",
            "</parameter>\n",
            "</function>\n",
            "</tool_call><|im_end|>\n",
            "<|im_start|>user\n",
            "<tool_response>\n",
            "# Title\n",
            "body\n",
            "</tool_response><|im_end|>\n",
            "<|im_start|>assistant\n",
            "<think>\n",
            "\n",
            "</think>\n",
            "\n",
        );
        // the history exactly as Crow builds it (`crow_core.py:3552-3570`): arguments is a STRING
        let msgs = serde_json::json!([
            {"role": "user", "content": "Read the first lines of a.md and tell me its title."},
            {"role": "assistant", "content": "", "tool_calls": [{
                "id": "call_0", "type": "function",
                "function": {"name": "read_file",
                             "arguments": "{\"path\": \"a.md\", \"start_line\": 1}"}
            }]},
            {"role": "tool", "tool_call_id": "call_0", "content": "# Title\nbody"}
        ]);
        let tools = a7_tools();
        let s = tk
            .render_chat(&normalize_messages(&msgs), Some(&tools), true, false)
            .expect("the tool history renders");
        assert_eq!(s, ORACLE_RENDER);
        // without the conversion the template cannot iterate the arguments at all
        let raw = tk.render_chat(&msgs, Some(&tools), true, false);
        assert!(raw.is_err(), "the string form must not render silently: {raw:?}");
    }

    #[test]
    fn a_request_line_over_the_cap_stops_the_reader_at_the_cap() {
        let mut raw = b"GET /".to_vec();
        raw.resize(70_000, b'a');
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::HeadTooLarge);
        // bounded: the cap was never passed, the 70,000 byte line was never buffered
        assert!(c.position() <= MAX_HEAD_BYTES as u64);
        assert_eq!(c.position(), MAX_HEAD_BYTES as u64);
    }

    #[test]
    fn header_lines_over_the_cap_stop_the_reader_at_the_cap() {
        let mut raw = b"GET /health HTTP/1.1\r\n".to_vec();
        while raw.len() < MAX_HEAD_BYTES + 4096 {
            raw.extend_from_slice(b"X-Pad: 0123456789012345678901234567890123456789\r\n");
        }
        raw.extend_from_slice(b"\r\n");
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::HeadTooLarge);
        assert!(c.position() <= MAX_HEAD_BYTES as u64);
    }

    #[test]
    fn a_plain_get_parses_out_of_a_cursor() {
        let mut c = Cursor::new(b"GET /health HTTP/1.1\r\nHost: x\r\n\r\n".to_vec());
        assert_eq!(
            read_head_from(&mut c).unwrap(),
            Head::Req { method: "GET".into(), target: "/health".into(), body: Vec::new() }
        );
    }

    #[test]
    fn a_declared_body_is_read_whole() {
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: 5\r\n\r\nhello".to_vec());
        match read_head_from(&mut c).unwrap() {
            Head::Req { method, target, body } => {
                assert_eq!(method, "POST");
                assert_eq!(target, "/x");
                assert_eq!(body, b"hello".to_vec());
            }
            other => panic!("expected a request, got {other:?}"),
        }
    }

    #[test]
    fn an_empty_stream_gets_no_response() {
        let mut c = Cursor::new(Vec::new());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Empty);
    }

    #[test]
    fn a_garbage_request_line_is_a_bad_request() {
        let mut c = Cursor::new(b"NOT HTTP AT ALL\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn a_malformed_content_length_is_a_bad_request() {
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: abc\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\nContent-Length: -1\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn a_repeated_content_length_is_a_bad_request() {
        let raw = b"POST /x HTTP/1.1\r\nContent-Length: 5\r\ncontent-length: 5\r\n\r\nhello".to_vec();
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Bad);
    }

    #[test]
    fn chunked_transfer_encoding_is_rejected() {
        let raw = b"POST /x HTTP/1.1\r\nTransfer-Encoding: chunked\r\n\r\n0\r\n\r\n".to_vec();
        let mut c = Cursor::new(raw);
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Chunked);
        // case insensitive, and inside a coding list
        let mut c = Cursor::new(b"POST /x HTTP/1.1\r\ntransfer-encoding: gzip, Chunked\r\n\r\n".to_vec());
        assert_eq!(read_head_from(&mut c).unwrap(), Head::Chunked);
        assert_eq!(
            error_json("chunked transfer encoding not supported")["error"],
            "chunked transfer encoding not supported"
        );
    }
    // ------------------------------------------------- #32 A10, /slots and /slots/0

    #[test]
    fn the_slot_routes_dispatch_on_method_and_path() {
        let d = |line: &str| {
            let (m, t) = parse_request_line(line).expect("request line parses");
            route(&m, route_path(&t))
        };
        assert_eq!(d("GET /slots HTTP/1.1
"), Route::Slots);
        assert_eq!(d("GET /slots/ HTTP/1.1
"), Route::Slots);
        assert_eq!(d("POST /slots/0?action=save HTTP/1.1
"), Route::Slot0);
        assert_eq!(d("POST /slots/0?action=restore HTTP/1.1
"), Route::Slot0);
        // the methods are not interchangeable, and no other slot id exists
        assert_eq!(d("POST /slots HTTP/1.1
"), Route::NotFound);
        assert_eq!(d("GET /slots/0 HTTP/1.1
"), Route::NotFound);
        assert_eq!(d("POST /slots/1?action=save HTTP/1.1
"), Route::NotFound);
    }

    #[test]
    fn the_action_is_read_out_of_the_query_string() {
        assert_eq!(query_param("/slots/0?action=save", "action"), Some("save"));
        assert_eq!(query_param("/slots/0?action=restore", "action"), Some("restore"));
        // more than one parameter, in either order, and a fragment after it
        assert_eq!(query_param("/slots/0?id=0&action=save", "action"), Some("save"));
        assert_eq!(query_param("/slots/0?action=save&id=0", "action"), Some("save"));
        assert_eq!(query_param("/slots/0?action=save#x", "action"), Some("save"));
        // absent, empty and a prefix that only looks like the key
        assert_eq!(query_param("/slots/0", "action"), None);
        assert_eq!(query_param("/slots/0?", "action"), None);
        assert_eq!(query_param("/slots/0?actionx=save", "action"), None);
        assert_eq!(query_param("/slots/0?xaction=save", "action"), None);
        assert_eq!(query_param("/slots/0?action=", "action"), Some(""));
    }

    /// `tools/measure-slot-restart.ps1:87` and `tools/probe-slot-persistence.py:152` read
    /// element 0 of this array and take `n_prompt_tokens` out of it
    #[test]
    fn the_slots_document_is_an_array_of_exactly_one_slot() {
        let doc = slots_json(200_000, 16_064);
        let arr = doc.as_array().expect("an array");
        assert_eq!(arr.len(), 1);
        assert_eq!(arr[0]["id"], 0);
        assert_eq!(arr[0]["n_ctx"], 200_000);
        assert_eq!(arr[0]["n_prompt_tokens"], 16_064);
        assert_eq!(arr[0]["is_processing"], serde_json::Value::Bool(false));
        // an empty slot says 0, it does not omit the key
        let empty = slots_json(200_000, 0);
        assert_eq!(empty[0]["n_prompt_tokens"], 0);
    }

    /// `crow_core.py:2458` reads `n_saved` and nothing else
    #[test]
    fn a_save_answer_carries_n_saved_as_an_integer() {
        let s = crow_nest_engine::slot::Saved { n_saved: 16_064, n_written: 352_861_048, ms: 1234.5 };
        let doc = slot_saved_json("crow-slot.bin", &s);
        assert_eq!(doc["id_slot"], 0);
        assert_eq!(doc["filename"], "crow-slot.bin");
        assert_eq!(doc["n_saved"].as_u64(), Some(16_064));
        assert!(doc["n_saved"].is_u64(), "n_saved must be an integer, not a float");
        assert_eq!(doc["n_written"].as_u64(), Some(352_861_048));
        assert_eq!(doc["timings"]["save_ms"].as_f64(), Some(1234.5));
    }

    /// `crow_core.py:2688` reads `n_restored` and compares it with the saved `n_saved`
    #[test]
    fn a_restore_answer_carries_n_restored_as_an_integer() {
        let r = crow_nest_engine::slot::Restored { n_restored: 16_064, n_read: 352_861_048, ms: 987.6 };
        let doc = slot_restored_json("crow-slot.bin", &r);
        assert_eq!(doc["id_slot"], 0);
        assert_eq!(doc["filename"], "crow-slot.bin");
        assert_eq!(doc["n_restored"].as_u64(), Some(16_064));
        assert!(doc["n_restored"].is_u64(), "n_restored must be an integer, not a float");
        assert_eq!(doc["n_read"].as_u64(), Some(352_861_048));
        assert_eq!(doc["timings"]["restore_ms"].as_f64(), Some(987.6));
    }

    #[test]
    fn the_filename_is_read_from_the_body_and_sanitized() {
        assert_eq!(slot_filename(br#"{"filename":"crow-session.bin"}"#), Ok("crow-session.bin".to_string()));
        // a path separator never becomes a path (crow-nest owns the directory, not the client)
        assert!(slot_filename(br#"{"filename":"../x.bin"}"#).is_err());
        assert!(slot_filename(br#"{"filename":"a/b.bin"}"#).is_err());
        assert!(slot_filename(br#"{"filename":"a\b.bin"}"#).is_err());
        // absent, wrong type, empty and a body that is not JSON at all
        assert!(slot_filename(br#"{}"#).is_err());
        assert!(slot_filename(br#"{"filename":7}"#).is_err());
        assert!(slot_filename(br#"{"filename":""}"#).is_err());
        assert!(slot_filename(b"not json").is_err());
        assert!(slot_filename(b"").is_err());
    }

    #[test]
    fn the_slot_directory_comes_from_the_command_line() {
        let v = |a: &[&str]| a.iter().map(|s| s.to_string()).collect::<Vec<_>>();
        assert_eq!(parse_args(&v(&["serve"])).map(|a| a.slot_save_path), Ok(None));
        assert_eq!(
            parse_args(&v(&["serve", "--slot-save-path", "decode_out"])).map(|a| a.slot_save_path),
            Ok(Some("decode_out".to_string()))
        );
        assert_eq!(
            parse_args(&v(&["serve", "--slot-save-path=decode_out"])).map(|a| a.slot_save_path),
            Ok(Some("decode_out".to_string()))
        );
        // both flags together, in either order, and the port still parses
        let a = parse_args(&v(&["serve", "--port", "8099", "--slot-save-path", "d"])).unwrap();
        assert_eq!((a.port, a.slot_save_path), (8099, Some("d".to_string())));
        let b = parse_args(&v(&["serve", "--slot-save-path", "d", "--port=1234"])).unwrap();
        assert_eq!((b.port, b.slot_save_path), (1234, Some("d".to_string())));
        // a flag without its value is a usage error, not a silent default
        assert!(parse_args(&v(&["serve", "--slot-save-path"])).is_err());
        assert!(parse_args(&v(&["serve", "--slot-save-path="])).is_err());
    }

    /// review finding: a typo'd `--slot-save-path` must not be discovered at the first save
    #[test]
    fn a_slot_save_path_that_is_no_directory_is_refused_at_boot() {
        // this source file exists and is NOT a directory
        let file = format!("{}/src/bin/serve.rs", env!("CARGO_MANIFEST_DIR"));
        let e = check_slot_save_path(&file).unwrap_err();
        assert!(e.contains("not a directory"), "{e}");
        // the message names the path the operator typed (debug quoted, so backslashes double)
        assert!(e.contains("serve.rs"), "{e}");

        let missing = format!("{}/no-such-slot-dir-4711", env!("CARGO_MANIFEST_DIR"));
        let e = check_slot_save_path(&missing).unwrap_err();
        assert!(e.contains("not a directory"), "{e}");
        // the boot check never creates the directory
        assert!(!std::path::Path::new(&missing).exists());

        // an existing directory passes
        assert_eq!(check_slot_save_path(env!("CARGO_MANIFEST_DIR")), Ok(()));
    }

    // ------------------------------------------- #39 B3a: the non streaming document

    /// a `Timing` whose every field is a different number, so a document test can name it
    fn b3a_timing() -> Timing {
        Timing {
            prompt_n: 7,
            cached_n: 3,
            predicted_n: 5,
            prompt_ms: 20.0,
            predicted_ms: 100.0,
            selections_total: 11,
            cold_total: 2,
            ple_rows_total: 9,
            ple_miss_total: 4,
        }
    }

    #[test]
    fn the_non_streaming_document_carries_every_field_the_probe_suite_reads() {
        // probe-suite.py:677-683 reads choices[0].finish_reason, .message.content,
        // .message.reasoning_content and usage.completion_tokens
        let t = b3a_timing();
        let d = completion_json("chatcmpl-1700-2", 1700, "crow", "hello", &[], "stop", &t);
        assert_eq!(d["id"], "chatcmpl-1700-2");
        assert_eq!(d["object"], "chat.completion");
        assert_eq!(d["created"], 1700);
        assert_eq!(d["model"], "crow");
        let c = &d["choices"][0];
        assert_eq!(c["index"], 0);
        assert_eq!(c["message"]["role"], "assistant");
        assert_eq!(c["message"]["content"], "hello");
        assert_eq!(c["finish_reason"], "stop");
        // ONE document shape: `usage` and `timings` are the objects of the final stream chunk,
        // unconditionally, because the probe-suite sends no `stream_options` and still reads
        // `usage.completion_tokens`
        assert_eq!(d["usage"], usage_json(&t));
        assert_eq!(d["timings"], timings_json(&t));
        assert_eq!(d["usage"]["completion_tokens"], 5);
        assert_eq!(d["usage"]["prompt_tokens"], 10);
        assert_eq!(d["timings"]["cache_n"], 3);
        // no tool call and no reasoning block on a plain answer
        assert!(c["message"].get("tool_calls").is_none());
        assert!(c["message"].get("reasoning_content").is_none());
        // exactly the top level keys of the contract, in order (preserve_order is on)
        let keys: Vec<&str> = d.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(
            keys,
            vec!["id", "object", "created", "model", "choices", "usage", "timings"]
        );
        let ck: Vec<&str> = c.as_object().unwrap().keys().map(|s| s.as_str()).collect();
        assert_eq!(ck, vec!["index", "message", "finish_reason"]);
        // `length` is the other reason gate part 1 accepts
        let l = completion_json("x", 1, "crow", "hi", &[], "length", &t);
        assert_eq!(l["choices"][0]["finish_reason"], "length");
    }

    #[test]
    fn the_non_streaming_document_carries_the_tool_calls_the_parser_closed() {
        // the A7 call, in the OpenAI non streaming shape: `id`, `type`, `function`
        let t = b3a_timing();
        let calls = vec![CallBuf {
            id: "call_0".to_string(),
            name: "read_file".to_string(),
            arguments: "{\"path\":\"a.md\",\"start_line\":1}".to_string(),
        }];
        let d = completion_json("id1", 5, "crow", "", &calls, "tool_calls", &t);
        let m = &d["choices"][0]["message"];
        assert_eq!(m["role"], "assistant");
        assert_eq!(m["content"], "");
        assert_eq!(m["tool_calls"][0]["id"], "call_0");
        assert_eq!(m["tool_calls"][0]["type"], "function");
        assert_eq!(m["tool_calls"][0]["function"]["name"], "read_file");
        assert_eq!(
            m["tool_calls"][0]["function"]["arguments"],
            "{\"path\":\"a.md\",\"start_line\":1}"
        );
        assert_eq!(d["choices"][0]["finish_reason"], "tool_calls");
        // `arguments` stays a JSON STRING, as OpenAI and llama-server send it
        assert!(m["tool_calls"][0]["function"]["arguments"].is_string());
    }

    #[test]
    fn the_collector_and_the_sse_sink_see_the_same_delta_sequence() {
        let pieces = vec![
            Emit::Content("Hi".to_string()),
            Emit::Content(" there".to_string()),
            Emit::Call {
                index: 0,
                id: "call_0".to_string(),
                name: "read_file".to_string(),
            },
            Emit::Args {
                index: 0,
                text: "{\"path\":\"".to_string(),
            },
            Emit::Args {
                index: 0,
                text: "a.md\"}".to_string(),
            },
        ];
        let mut buf: Vec<u8> = Vec::new();
        let mut sse = SseSink::new(&mut buf);
        let (mut sc, mut st) = (0usize, 0usize);
        assert!(send_emits(&mut sse, "id1", 5, "crow", &pieces, &mut sc, &mut st));
        let mut col = CollectSink::default();
        let (mut cc, mut ct) = (0usize, 0usize);
        assert!(send_emits(&mut col, "id1", 5, "crow", &pieces, &mut cc, &mut ct));
        // the same loop counts the same chunks for both sinks
        assert_eq!((sc, st), (cc, ct));
        assert_eq!((sc, st), (2, 3));

        // rebuild the answer out of the raw SSE frames the way Crow does, then compare
        let text = String::from_utf8(buf).unwrap();
        let mut content = String::new();
        let mut calls: Vec<CallBuf> = Vec::new();
        for frame in text.split("\n\n").filter(|f| !f.is_empty()) {
            let d: serde_json::Value =
                serde_json::from_str(frame.trim_start_matches("data: ")).unwrap();
            let delta = &d["choices"][0]["delta"];
            if let Some(s) = delta.get("content").and_then(|v| v.as_str()) {
                content.push_str(s);
            }
            if let Some(tc) = delta.get("tool_calls").and_then(|v| v.as_array()) {
                let i = tc[0]["index"].as_u64().unwrap() as usize;
                if i == calls.len() {
                    calls.push(CallBuf::default());
                }
                if let Some(v) = tc[0].get("id").and_then(|v| v.as_str()) {
                    calls[i].id = v.to_string();
                }
                let f = &tc[0]["function"];
                if let Some(v) = f.get("name").and_then(|v| v.as_str()) {
                    calls[i].name = v.to_string();
                }
                if let Some(v) = f.get("arguments").and_then(|v| v.as_str()) {
                    calls[i].arguments.push_str(v);
                }
            }
        }
        assert_eq!(content, col.content);
        assert_eq!(calls, col.calls);
        assert_eq!(col.content, "Hi there");
        assert_eq!(col.calls.len(), 1);
        assert_eq!(col.calls[0].name, "read_file");
        assert_eq!(col.calls[0].arguments, "{\"path\":\"a.md\"}");
    }

    #[test]
    fn the_two_callers_that_send_no_stream_field_parse_as_non_streaming() {
        // probe-suite.py:624-630, verbatim shape: no `stream` key at all
        let r = parse_chat(
            br#"{"model":"crow","messages":[{"role":"user","content":"hi"}],
                 "max_tokens":4096,"temperature":0.6,"seed":1234}"#,
        )
        .unwrap();
        assert!(!r.stream);
        assert_eq!(r.model, "crow");
        assert_eq!(r.max_tokens, 4096);
        assert_eq!(r.seed, 1234);
        // crow_core.py:2960-2976, the rollover digest: no stream, enable_thinking false
        let d = parse_chat(
            br#"{"model":"crow","messages":[{"role":"user","content":"hi"}],
                 "max_tokens":400,"chat_template_kwargs":{"enable_thinking":false}}"#,
        )
        .unwrap();
        assert!(!d.stream);
        assert!(!d.enable_thinking);
        // an explicit false is the same request
        let f = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],"stream":false}"#,
        )
        .unwrap();
        assert!(!f.stream);
    }

    #[test]
    fn the_collector_holds_one_buffer_per_tool_call_index() {
        let mut col = CollectSink::default();
        let (mut c, mut t) = (0usize, 0usize);
        let pieces = vec![
            Emit::Call {
                index: 0,
                id: "call_0".to_string(),
                name: "a".to_string(),
            },
            Emit::Args {
                index: 0,
                text: "{}".to_string(),
            },
            Emit::Call {
                index: 1,
                id: "call_1".to_string(),
                name: "b".to_string(),
            },
            Emit::Args {
                index: 1,
                text: "{\"x\":".to_string(),
            },
            Emit::Args {
                index: 1,
                text: "1}".to_string(),
            },
        ];
        assert!(send_emits(&mut col, "i", 1, "m", &pieces, &mut c, &mut t));
        assert_eq!(col.calls.len(), 2);
        assert_eq!(col.calls[0].arguments, "{}");
        assert_eq!(col.calls[1].name, "b");
        assert_eq!(col.calls[1].arguments, "{\"x\":1}");
        assert_eq!((c, t), (0, 5));
    }
}
