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
//! - #54: a `stream:false` generation is ended when its client is gone.
//! - The probe is one `poll(POLLRDHUP)` on the request socket between two decode steps.
//! - Linux only (`POLLRDHUP`); elsewhere the probe is inert and that path is as it was.
//! - A client that only half-closed its write side still gets its document (7.11.18).
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
//! | `max_tokens` | default 8192 (1024 until 2026-09-18), capped at 32768 |
//! | `model` | echoed into every chunk, default `crow-nest` |
//! | `chat_template_kwargs.enable_thinking` | template variable, default false |
//! | `reasoning_effort` | #74: top level OR `chat_template_kwargs`; `none` and an absent field are the render of record, `low` / `medium` pass through, `high` and `xhigh` both render `xhigh`, anything else is a 400 |
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
//! - Crow sends `arguments` as a JSON STRING (`crow_core.py:5068`); the template needs a MAPPING.
//! - Measured 2026-09-09 against the Python oracle: the string form raises
//!   `TypeError: Can only get item pairs from a mapping`, so it is not a render at all.
//! - `normalize_messages` therefore parses the string into the object it encodes.
//!
//! What reaches `tool_call.arguments|items` (`chat_template.jinja:136`), TASK J 2026-09-17:
//!
//! | `arguments` as it arrives | what the render sees | note |
//! |---|---|---|
//! | an object | unchanged | the template's own form |
//! | a string that parses to an object | that object | Crow's form, the A7 path, bytes unchanged |
//! | the EMPTY string | unchanged | the template's `arguments != ''` guard skips it |
//! | any other string | `{"_raw": "<the string, verbatim>"}` | the model still sees what the turn asked for |
//! | JSON `null` | `{}` | null means there are no arguments |
//! | any other value | `{"_raw": "<its compact JSON>"}` | same reason as the string row |
//!
//! - NOTHING that is not a mapping reaches that line any more. Before TASK J a string that
//!   was not a JSON object was left alone and the render failed with a 400 - and because
//!   Crow stores the assistant turn verbatim (`crow_core.py:3756-3761`), re-sends the whole
//!   history every turn (`:3783-3785`, `:4866`) and cannot drop a message (`:13485-13487`),
//!   that one 400 repeated for every later turn of the session. Measured live on 2026-09-17:
//!   a tool call truncated at `max_tokens` left `{"path":"` in the history and the next three
//!   turns all answered `400 chat template render failed: invalid operation: cannot convert
//!   value into pairs (in chat:136)` (`tools/replay-toolcalls.py`, `decode_out/taskj/`).
//! - The engine end is `toolcall.rs`: an abandoned call now closes its own `arguments` object
//!   and marks it `_truncated`, so the engine cannot produce that string in the first place.
//! - Every rewrite writes one `[chat] normalised: message <i> tool_call <j> ...` stderr line
//!   with the first 200 bytes of the offending value.
//!
//! Messages a 400 names (TASK J): `check_messages` runs on the normalized messages before the
//! render, and every refusal body names the MESSAGE INDEX and the FIELD - the bare
//! `chat template render failed: ... (in chat:NNN)` did not. The table of what renders and
//! what is refused is on `check_messages`; the shapes covered are a `content` that is not a
//! string, a list of parts or null, a `tool_calls` that is not an array, a tool call whose
//! `function` or `function.name` is wrong, a role the template does not render and a system
//! message that is not first. `reasoning_content` of any type renders (`chat:112` ignores a
//! non-string). Both the check and a render that still fails log `message_digest`: one stderr
//! line per message with its role, its content kind and every tool call's name and the first
//! 200 bytes of its `arguments`.
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
//! - `tokenizer::render_chat_effort(messages, tools, add_generation_prompt=true,
//!   enable_thinking, reasoning_effort)` - four template variables, no string surgery.
//! - #74: `reasoning_effort` is DEFINED only for a request that thinks, so a request that
//!   names no level renders the ids of record and every parity and gate value stands.
//! - #74: with thinking on the generation prompt ends in `<think>\n`, so `ThinkFilter` starts
//!   `Inside` for that request - otherwise the reasoning would leave as `content`.
//! - #81: `reasoning_budget_tokens` caps the THINKING, not the answer. Absent, null or
//!   negative is the behaviour of every release before #81; `0` closes the block before it
//!   opens; `n > 0` closes it once `n` reasoning tokens were generated - the sampled
//!   continuation is REPLACED by `reasoning_budget_message` (when sent; Crow's
//!   REASONING_BUDGET_MESSAGE, #176: a bare close starts answers mid-word), the
//!   `</think>` token and a `\n\n`, all appended as generated ids so the model's own
//!   context contains the close and it answers in what is left of `max_tokens`. The same
//!   mechanism llama-server ships as `--reasoning-budget` (its PRs #13771/#17750) and
//!   Qwen's docs name `thinking_budget`; without the field the ids stay byte-identical.
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
//! - TASK J: a malformed or truncated call CLOSES its `arguments` object and marks it
//!   `{"..., "_truncated":true}`, so the concatenation of the fragments of every index is
//!   always a parseable JSON object. It used to be left unterminated so `json.loads` would
//!   fail; that string is what Crow stored and re-sent until the session died. The marker
//!   keeps the safety: no tool declares `_truncated`, so half a command is not runnable.
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
//! The cross-turn repeat counter (#68, 2026-09-18), pure observability:
//!
//! | name | meaning | where |
//! |---|---|---|
//! | `repeat_of` | how many answers back the most recent IDENTICAL answer of this process is; 0 = none in the ring | `routing` line only |
//! | `repeat_run` | how many identical answers in a row ended with this one; 1 = none | `routing` line, and `, repeat run N` on the `[chat]` line when N > 1 |
//! | `single_token` | the answer is exactly ONE generated id and the model ended it itself (`finish stop`) | `routing` line, and `, single-token answer` on the `[chat]` line when true |
//!
//! - The state is a ring of the last `REPEAT_RING` = 8 answer HASHES (`RepeatRing`), FNV-1a
//!   over the GENERATED IDS - what the model produced, not what the detokenizer made of it.
//! - `repeat_run` is not capped by the ring: the live `#68` session's 48 identical answers
//!   would be reported as 48. The ring bounds `repeat_of` only.
//! - At `LOOP_WARN_AT` = 3 one WARN line goes to target `chat`
//!   (`the client is looping: N identical answers in a row`), also for three consecutive
//!   single-token answers. It is a line and nothing else: no 4xx, no brake, no sampling
//!   change, no wire field. Loop detection across turns belongs to the client; this server
//!   only says what it sees, because nothing INSIDE one request can see it
//!   (`docs/long-context-goalmode.md` 3.3: each of those 48 requests was correct on its own).
//! - Always on, no env flag: it costs one hash pass over at most `max_tokens` ids per
//!   request, after the last `decode_step`, and eight `u64` of state.
//! - SCOPE: per PROCESS, not per session. `serve` holds one conversation and there is no
//!   session id on the wire; and a COLD prefill is NOT a new conversation - an identical
//!   re-send is cold by construction (its snapshot sits at its own prompt length, spec 7.4),
//!   which is exactly the case this counter exists to see. So the ring lives as long as the
//!   process does and a restart is what clears it.
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
//! | the `[slot]` stderr lines | the `tracing` events on target `slot` in `slot_route` below (#13: `eprintln!` until 2026-09-18, same text) |
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

use crow_nest_engine::cache::{PrefixCache, SLOTS};
use crow_nest_engine::boot;
use crow_nest_engine::cnq::Cnq;
use crow_nest_engine::gen::{DevSampler, Engine};
use crow_nest_engine::geo::{apply_adapt_policy, DEFAULT_CNQ, DEFAULT_HOTSETS, LAYERS, TRICKLE_CHUNK_THRESHOLD};
use crow_nest_engine::sample::{Sampler, EOS_IDS};
use crow_nest_engine::slot;
use crow_nest_engine::toolcall::{Emit, ToolStream, TOOL_OPEN};
use std::collections::VecDeque;
use std::io::{BufRead, BufReader, Read, Write};
use std::net::{Shutdown, TcpListener, TcpStream};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const DEFAULT_PORT: u16 = 8099;
/// pinned for the process (M1): every request prefills at chunk 2048. The
/// 4096 experiment of 2026-09-14 collapsed the live serve prefill (~100 tok/s)
/// - the trickle/adapt policy is tuned for 2048 and no serve-form measurement
/// backed the raise; reverted same day. It DERIVES from the policy threshold
/// it is tuned against: serve's chunk is what arms the stream trickle.
const SERVE_CHUNK: usize = TRICKLE_CHUNK_THRESHOLD;
/// read and write timeout per connection, so one stalled client cannot hold the loop
const IO_TIMEOUT_SECS: u64 = 10;
/// request line plus headers, 64 KiB total; over it the answer is 431
const MAX_HEAD_BYTES: usize = 64 * 1024;
/// declared `Content-Length`, 16 MiB; over it the answer is 413
const MAX_BODY_BYTES: usize = 16 * 1024 * 1024;
/// `max_tokens` when the request carries none
// 8192 since 2026-09-18: Crow sends no max_tokens on the local path, and at 1024 a
// write_file that carries a whole SVG ends in `finish length` before the model has
// written the `path` parameter (robin's session, 13:49 UTC). An agentic client needs
// room for one file per call; the 32768 cap and the n_ctx clamp are unchanged.
const DEFAULT_MAX_TOKENS: usize = 8192;
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
/// #54: how many decode steps of the `stream:false` path ONE gone-client probe covers.
/// 1 means every step, and that is what the measurement bought: one probe is a single
/// `poll` on one descriptor, no byte read and no byte written, 0.10 us mean and 0.14 us
/// worst over 1,000,000 calls (2026-09-18, this machine) against a 13.6 ms token on the
/// live serve (27 ms at the 16k reading of record) - one 136,000th of the budget at worst.
/// A cadence of k would buy nothing measurable and would pay for it with k-1 further steps
/// of a generation nobody is reading (7.11.18).
const PROBE_EVERY: usize = 1;

/// #68: which sampling fields the REQUEST carried, so the `[chat]` line can say per value
/// whether it came from the client or from the data sheet `serve` fills in.
///
/// Why it exists: the live goal-mode session of `#68` was read off that line as "Crow sent
/// `presence_penalty 1.5`", and Crow has no such field at all - its wire list is
/// `SAMPLING_FIELDS = ("temperature", "top_p", "min_p", "top_k")` (`crow_core.py:716`,
/// build of 2026-09-16) and 1.5 is `DEFAULT_PRESENCE` here. A line that cannot be read
/// that way costs four words; a wrong attribution cost an issue.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct SamplingSent {
    top_p: bool,
    top_k: bool,
    presence_penalty: bool,
    seed: bool,
}

impl SamplingSent {
    /// what the `[chat]` line writes behind one value
    fn tag(sent: bool) -> &'static str {
        if sent { "request" } else { "data sheet" }
    }
}

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
            tracing::error!(target: "tokenize", "[tokenize] {e}");
            return 2;
        }
    };
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => {
            tracing::error!(target: "tokenize", "[tokenize] {e}");
            return 3;
        }
    };
    let (tp, cp) = tk.paths();
    tracing::info!(target: "tokenize", "[tokenize] tokenizer {tp}");
    tracing::info!(target: "tokenize", "[tokenize] chat template {cp}");

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
                tracing::info!(target: "tokenize", "[tokenize] {} tokens", ids.len());
                0
            }
            Err(e) => {
                tracing::error!(target: "tokenize", "[tokenize] {e}");
                4
            }
        },
        Tok::File { chat, file, out } => {
            let raw = match std::fs::read(&file) {
                Ok(r) => r,
                Err(e) => {
                    tracing::error!(target: "tokenize", "[tokenize] cannot read {file}: {e}");
                    return 4;
                }
            };
            let doc: serde_json::Value = match serde_json::from_slice(&raw) {
                Ok(d) => d,
                Err(e) => {
                    tracing::error!(target: "tokenize", "[tokenize] {file} is not JSON: {e}");
                    return 4;
                }
            };
            let prompts = match prompts_from_json(&doc) {
                Ok(p) => p,
                Err(e) => {
                    tracing::error!(target: "tokenize", "[tokenize] {file}: {e}");
                    return 4;
                }
            };
            let mut map = serde_json::Map::new();
            for (id, text) in &prompts {
                match ids_of(chat, text) {
                    Ok(ids) => {
                        tracing::info!(target: "tokenize", "[tokenize] {id} chars {} tokens {}", text.chars().count(), ids.len());
                        map.insert(id.clone(), serde_json::json!(ids));
                    }
                    Err(e) => {
                        tracing::error!(target: "tokenize", "[tokenize] {id}: {e}");
                        return 4;
                    }
                }
            }
            let text = serde_json::Value::Object(map).to_string();
            if let Err(e) = std::fs::write(&out, text) {
                tracing::error!(target: "tokenize", "[tokenize] cannot write {out}: {e}");
                return 4;
            }
            tracing::info!(target: "tokenize", "[tokenize] {} prompts -> {out}", prompts.len());
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
    /// the RESOLVED thinking switch, a template variable (#74: `reasoning_effort` resolves it
    /// too, `chat_template_kwargs.enable_thinking` is still the direct door)
    enable_thinking: bool,
    /// #74: the word THIS template gets as `reasoning_effort`, `None` leaves the variable
    /// undefined; never `Some` while `enable_thinking` is false, so a non-thinking request
    /// renders the bytes it rendered before #74
    reasoning_effort: Option<&'static str>,
    /// #74: the word the BODY carried, for the `[chat]` line; `None` when it named none
    reasoning_asked: Option<String>,
    /// #74: the body named `chat_template_kwargs.enable_thinking`; read by the `[chat]` line
    /// only, so the `(request)` tag is true for the second door as well
    kwargs_thinking_sent: bool,
    /// #81: the thinking budget in TOKENS. `None` (absent, null or negative) is the
    /// behaviour of every release before #81: reasoning runs until the model closes the
    /// block itself or `max_tokens` is spent. `Some(0)` closes the block before the first
    /// reasoning token; `Some(n)` closes it once `n` reasoning tokens were generated --
    /// the mechanism llama-server owns as `--reasoning-budget` (PR #13771/#17750) and Crow
    /// has sent since #176, so the two arms of the same client can run the same protocol.
    reasoning_budget: Option<usize>,
    /// #81: the wrap-up text injected as the LAST reasoning tokens before the forced
    /// close. Crow sends `REASONING_BUDGET_MESSAGE` beside the budget because a bare
    /// force-close was measured to start answers mid-word (2 of 9 capped answers, #176);
    /// with the message 0 of 6. `None` closes without one.
    reasoning_budget_message: Option<String>,
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
    /// #68: which of `top_p`, `top_k`, `presence_penalty`, `seed` the body carried; read by
    /// the `[chat]` line only, never by the sampler
    sampling_sent: SamplingSent,
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

/// #74: the words a client may name, in the order the 400 body lists them
const REASONING_WORDS: &str = "none, low, medium, high, xhigh";

/// - #74: the wire word a client sends -> the word THIS model's template accepts.
/// - The two vocabularies are not the same one and never were. Crow's ladder for these
///   weights on llama-server is `none / low / medium / high` (its manifest entry
///   `flash-next-q2-k-xl`, measured #160), and this template
///   (`models/Qwen3.8-Flash-Next-original/chat_template.jinja:46-53`) accepts
///   `xhigh`, `medium`, `low`, defaults to `xhigh` and RAISES on anything else.
///
/// | sent | template `reasoning_effort` | `enable_thinking` | why this and not something else |
/// |---|---|---|---|
/// | absent | undefined | false, the default of record | the prompt of every parity value, byte for byte |
/// | `none` | undefined | false | the meaning llama-server gives the TOP-LEVEL field: `none` sets `enable_thinking = false` and DROPS the key (`server-common.cpp:1323`), which is exactly the render above |
/// | `low` | `low` | true | the template's own word |
/// | `medium` | `medium` | true | the template's own word |
/// | `high` | `xhigh` | true | this template has no `high`. On the unsloth template that serves the SAME weights under llama-server, `high` renders byte-identically to the unset key, and the unset key is `xhigh` (Crow's manifest, `flash-next-q2-k-xl` `reasoning_groups` `["off", "high"]`, measured 2026-08-30 #160). So `high -> xhigh` is the step llama-server already gives the word, not a promotion to a dearer one |
/// | `xhigh` | `xhigh` | true | the template's own top word, reachable under its own name |
/// | anything else | - | - | 400, naming the five words. `max`, `minimal` and an explicit `off` are fatal on llama-server too (#160), so a silent downgrade would hide a client bug that the reference engine reports |
///
/// - `Ok(None)` is `none`: thinking OFF, and no variable, which is the record render.
/// - Matching is exact and lower case, as llama-server matches it; `High` is a 400.
fn map_reasoning_effort(word: &str) -> Result<Option<&'static str>, String> {
    match word {
        "none" => Ok(None),
        "low" => Ok(Some("low")),
        "medium" => Ok(Some("medium")),
        "high" | "xhigh" => Ok(Some("xhigh")),
        other => Err(format!(
            "reasoning_effort \"{other}\" is not one of {REASONING_WORDS} (this model's \
             template accepts xhigh, medium and low; high is its xhigh and none turns \
             thinking off)"
        )),
    }
}

/// - #74: the ONE word the request `[chat]` line carries for this request's thinking, and the
///   same word Crow's manifest calls the step: `off`, or the template word that was rendered.
/// - `off` is not a level, it is the absence of one - the prompt then carries the CLOSED empty
///   think block and nothing the model writes is reasoning.
/// - thinking through the kwargs door alone leaves the variable undefined and the template
///   takes its own default, which is `xhigh`; the word is the same either way, and which door
///   asked for it is what the line below says.
fn thinking_tag(req: &ChatReq) -> &'static str {
    match (req.enable_thinking, req.reasoning_effort) {
        (false, _) => "off",
        (true, Some(w)) => w,
        (true, None) => "xhigh",
    }
}

/// - #74: the provenance line of thinking, in the shape the sampling line of #68 established:
///   the VALUE, then `(request)` or `(data sheet)` for where it came from.
/// - `(request)` means the body named a door - the top-level `reasoning_effort`, or
///   `chat_template_kwargs`; `(data sheet)` means it named neither and this file decided.
/// - The word the client SENT is quoted when it is not the word the template got, because
///   `high -> xhigh` is the one place the two vocabularies differ and a line that hid it would
///   make a step look like a step it is not.
/// - pure: the test drives it on parsed bodies, no engine and no socket.
fn thinking_line(req: &ChatReq) -> String {
    let tag = SamplingSent::tag(req.reasoning_asked.is_some() || req.kwargs_thinking_sent);
    let asked = match req.reasoning_asked.as_deref() {
        Some(w) if Some(w) != req.reasoning_effort => format!(", asked as \"{w}\""),
        _ => String::new(),
    };
    if req.enable_thinking {
        format!(
            "[chat] thinking on ({tag}): reasoning_effort {}{asked}; the generation prompt \
             ends in <think> and the reasoning filter starts Inside (#74)",
            thinking_tag(req)
        )
    } else {
        format!(
            "[chat] thinking off ({tag}){asked}; the generation prompt carries the closed \
             empty think block, the render of record (#74)"
        )
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
    // #74: the two doors, and which one wins. `chat_template_kwargs.enable_thinking` is the
    // direct template variable and was the only one `serve` read until today; the TOP-LEVEL
    // `reasoning_effort` is the door llama-server owns and the one Crow has sent since its
    // #176. Both are read here, the top-level field first, because that is the order
    // llama-server resolves them in: it writes its value INTO the kwargs.
    let kwargs = obj.get("chat_template_kwargs");
    let kw_thinking = kwargs
        .and_then(|k| k.get("enable_thinking"))
        .and_then(|v| v.as_bool());
    let asked = match obj.get("reasoning_effort") {
        None | Some(serde_json::Value::Null) => kwargs
            .and_then(|k| k.get("reasoning_effort"))
            .filter(|v| !v.is_null()),
        Some(v) => Some(v),
    };
    let reasoning_asked = match asked {
        None => None,
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| format!("reasoning_effort is not a string (one of {REASONING_WORDS})"))?
                .to_string(),
        ),
    };
    // a word the engine does not have is a 400 BEFORE any GPU work, never a silent downgrade
    let mapped = match reasoning_asked.as_deref() {
        Some(w) => Some(map_reasoning_effort(w)?),
        None => None,
    };
    // `enable_thinking` stays the last word when the body spells it out: that is what
    // llama-server does with the pair (it sets the kwarg and never overwrites an explicit
    // false), and it keeps the digest path of `crow_core.py:2970` exactly as it was.
    let (enable_thinking, reasoning_effort) = match mapped {
        None => (kw_thinking.unwrap_or(false), None),
        Some(None) => (false, None),
        Some(word) => (kw_thinking.unwrap_or(true), word),
    };
    // a request that does not think never carries the variable, so its prompt is the prompt
    // of record whatever word the body named
    let reasoning_effort = if enable_thinking { reasoning_effort } else { None };
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
    // #68: a field counts as SENT when the body carries it as a non-null value - the same
    // condition every reader above treats as "absent", so the tag cannot disagree with the value
    let sent = |k: &str| !matches!(obj.get(k), None | Some(serde_json::Value::Null));
    let sampling_sent = SamplingSent {
        top_p: sent("top_p"),
        top_k: sent("top_k"),
        presence_penalty: sent("presence_penalty"),
        seed: sent("seed"),
    };
    // #81: the thinking budget, in llama-server's integer dialect: absent, null or negative
    // is unrestricted (the behaviour of every release before #81), 0 closes the block
    // before the first reasoning token, n > 0 caps the reasoning at n tokens. The message
    // is Crow's REASONING_BUDGET_MESSAGE and travels beside the cap for exactly the
    // measured reason recorded in #176: a bare force-close starts answers mid-word.
    let reasoning_budget = match obj.get("reasoning_budget_tokens") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => {
            let n = v
                .as_i64()
                .ok_or_else(|| "reasoning_budget_tokens is not an integer".to_string())?;
            if n < 0 {
                None
            } else {
                Some(n as usize)
            }
        }
    };
    let reasoning_budget_message = match obj.get("reasoning_budget_message") {
        None | Some(serde_json::Value::Null) => None,
        Some(v) => Some(
            v.as_str()
                .ok_or_else(|| "reasoning_budget_message is not a string".to_string())?
                .to_string(),
        ),
    };
    Ok(ChatReq {
        model,
        messages: messages.clone(),
        tools,
        stream,
        max_tokens,
        enable_thinking,
        reasoning_effort,
        reasoning_asked,
        kwargs_thinking_sent: kw_thinking.is_some(),
        reasoning_budget,
        reasoning_budget_message,
        include_usage,
        timings_per_token,
        temperature,
        top_p,
        top_k,
        presence_penalty,
        seed,
        sampling_sent,
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

/// the three fields EVERY chunk and the document of one response repeat
#[derive(Clone, Copy)]
struct ChunkCtx<'a> {
    id: &'a str,
    created: u64,
    model: &'a str,
}

impl<'a> ChunkCtx<'a> {
    fn new(id: &'a str, created: u64, model: &'a str) -> ChunkCtx<'a> {
        ChunkCtx { id, created, model }
    }
}

/// what the LAST chunk needs beyond the ChunkCtx (#27 A5: usage / timings are flags)
struct FinishArgs<'a> {
    finish: &'a str,
    t: &'a Timing,
    include_usage: bool,
    timings_per_token: bool,
}

/// one `chat.completion.chunk`, the only object shape this endpoint streams
fn chunk(c: &ChunkCtx, delta: serde_json::Value, finish: Option<&str>) -> serde_json::Value {
    serde_json::json!({
        "id": c.id,
        "object": "chat.completion.chunk",
        "created": c.created,
        "model": c.model,
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
fn chunk_role(c: &ChunkCtx) -> serde_json::Value {
    chunk(c, serde_json::json!({ "role": "assistant" }), None)
}

/// a content delta chunk, one per emitted text piece
fn chunk_content(c: &ChunkCtx, text: &str) -> serde_json::Value {
    chunk(c, serde_json::json!({ "content": text }), None)
}

/// - #67: one `delta.reasoning_content` piece, the text of a `<think>` block the model
///   opened itself. Crow reads this key (`crow_core.py:5045`), shows it behind
///   `--show-reasoning` and stores it as `message.reasoning_content` (`:3755`) - which is
///   the field THIS template renders inside the assistant turn's think block, so the round
///   trip is the template's own form and not a second copy of the answer.
/// - Absent from every stream that stripped nothing: no empty frame is ever sent.
fn chunk_reasoning(c: &ChunkCtx, text: &str) -> serde_json::Value {
    chunk(c, serde_json::json!({ "reasoning_content": text }), None)
}

/// - #29 A7: the FIRST fragment of one tool call, the only one carrying `id` and `name`
/// - `arguments` is the empty string here, as llama-server and OpenAI send it
/// - Crow keeps `id` and `name` because it tests them for truth (`crow_core.py:4869-4874`)
fn chunk_tool_open(c: &ChunkCtx, index: usize, call_id: &str, name: &str) -> serde_json::Value {
    let delta = serde_json::json!({
        "tool_calls": [{
            "index": index,
            "id": call_id,
            "type": "function",
            "function": { "name": name, "arguments": "" },
        }],
    });
    chunk(c, delta, None)
}

/// - #29 A7: one `arguments` fragment of the call at `index`
/// - NO `id` and NO `name`: an empty string here would erase what the open chunk said
/// - the concatenation of every fragment of one index is the arguments JSON object text
fn chunk_tool_args(c: &ChunkCtx, index: usize, args: &str) -> serde_json::Value {
    let delta = serde_json::json!({
        "tool_calls": [{
            "index": index,
            "function": { "arguments": args },
        }],
    });
    chunk(c, delta, None)
}

/// - last chunk before `[DONE]`: empty delta, the finish reason
/// - `usage` rides along when `include_usage`, `timings` when `timings_per_token` (#27 A5)
/// - neither flag: the object is exactly the A4 chunk, no empty placeholders
/// - pure: the whole final chunk contract is one function the test can drive
fn chunk_finish(c: &ChunkCtx, a: &FinishArgs) -> serde_json::Value {
    let mut doc = chunk(c, serde_json::json!({}), Some(a.finish));
    if let Some(obj) = doc.as_object_mut() {
        if a.include_usage {
            obj.insert("usage".to_string(), usage_json(a.t));
        }
        if a.timings_per_token {
            obj.insert("timings".to_string(), timings_json(a.t));
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

/// #67: `<think>`, the tag the model opens a reasoning block with. Ordinary TEXT, not an
/// added token, so it arrives split across decode steps like any other markup.
const THINK_OPEN: &str = "<think>";
/// #67: `</think>`. The prompt already carries a CLOSED empty block (`enable_thinking` is
/// false, `tokenizer.rs:18-19`), so every one of these in the generation is a stray.
const THINK_CLOSE: &str = "</think>";

/// #67: what one fed piece splits into - `delta.content` and `delta.reasoning_content`
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct Split {
    content: String,
    reasoning: String,
}

/// #67: where the filter stands in the text of ONE generation
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
enum Think {
    /// nothing but whitespace has been emitted as content yet: a `<think>` HERE opens a block
    #[default]
    Lead,
    /// inside a block the model opened itself: the text is `reasoning_content`, never content
    Inside,
    /// content is flowing: `<think>` is ordinary text, `</think>` is a stray and is dropped
    Body,
}

/// - #67: the reasoning filter of the generation path, the one llama.cpp's chat parser is
///   (`reasoning_format`): a leading `<think>...</think>` block becomes `reasoning_content`
///   and a bare `</think>` is DROPPED, so neither tag ever leaves as `content`.
/// - Why it has to exist: the model emits a stray `</think>` after a long paste, Crow stores
///   the turn verbatim and re-sends the whole history every turn (`crow_core.py:3756-3761`,
///   `:3783-3785`), and THIS template renders a stored `content` verbatim inside the assistant
///   turn's own think block (7.11.16) - so one stray tag teaches the model to emit one, every
///   turn, for the rest of the session.
/// - It is OFF the numeric path: it changes what is STREAMED, never what is sampled. The
///   generated ids, the `[chat] ids` line and every gate value are untouched by construction -
///   nothing here feeds back into `decode_step`.
///
/// | state | `<think>` | `</think>` | anything else |
/// |---|---|---|---|
/// | `Lead` (only whitespace emitted so far) | opens the block, the whitespace before it is dropped | dropped, the whitespace before it is dropped | whitespace is HELD, the first real character flushes it and opens `Body` |
/// | `Inside` | ordinary reasoning text | closes the block, back to `Lead` so the `\n\n` after it is trimmed | `reasoning_content` |
/// | `Body` | ordinary content (a leading block is the only one this filter owns) | DROPPED, the stray of #67 | `content` |
///
/// - Token boundaries: a tag arrives across several deltas, so a tail that is a PREFIX of a
///   candidate tag is held back until the next piece resolves it (`held`). A prefix that never
///   completes leaves as content at `flush()`, so no byte is ever lost.
/// - `<` that opens no candidate is emitted immediately; only a real prefix is held.
/// - pure: no engine, no socket, so the tests drive it at every split point directly.
#[derive(Debug, Clone, Default)]
struct ThinkFilter {
    /// the state above; `Lead` at the first byte of every generation
    state: Think,
    /// a tail that is a proper prefix of a candidate tag, waiting for the next piece
    held: String,
    /// whitespace seen in `Lead`, flushed by the first real character
    lead: String,
    /// that whitespace follows a tag this filter dropped, so it is dropped with it
    drop_lead: bool,
    /// how many tags were stripped, for the one `[chat]` line
    stripped: usize,
}

impl ThinkFilter {
    fn new() -> Self {
        ThinkFilter::default()
    }

    /// - #74: the filter of a THINKING request, and the whole reason `Lead` was not enough.
    /// - With `enable_thinking` true the generation prompt already ENDS in `<think>\n`
    ///   (`chat_template.jinja:167`), so the model is inside the block at its first token and
    ///   will never open one. A filter that started in `Lead` would stream the reasoning as
    ///   `content` and then DROP the model's own `</think>` as a stray - the answer would
    ///   carry the thinking, and Crow would store it and re-send it every turn (7.11.16).
    /// - It starts where the PROMPT put it, so the first `</think>` closes the block, the
    ///   `\n\n` after it is trimmed, and the answer starts at the first real character.
    fn inside() -> Self {
        ThinkFilter { state: Think::Inside, ..ThinkFilter::default() }
    }

    /// the filter THIS request needs: `enable_thinking` decides where the text starts
    fn for_request(enable_thinking: bool) -> Self {
        if enable_thinking {
            ThinkFilter::inside()
        } else {
            ThinkFilter::new()
        }
    }

    /// #81: is the block still open? The budget counter asks this of every generated
    /// token AFTER the filter has seen its text, so a token that closes the block itself
    /// is the model's own close and never a counted reasoning token.
    fn is_inside(&self) -> bool {
        matches!(self.state, Think::Inside)
    }

    /// how many `<think>` / `</think>` tags this filter kept off the wire
    fn stripped(&self) -> usize {
        self.stripped
    }

    /// the tags that mean something in the CURRENT state; both start with `<`
    fn candidates(&self) -> &'static [&'static str] {
        match self.state {
            Think::Lead => &[THINK_OPEN, THINK_CLOSE],
            Think::Inside | Think::Body => &[THINK_CLOSE],
        }
    }

    /// one decoded piece in, its content and reasoning halves out
    fn push(&mut self, piece: &str) -> Split {
        let mut out = Split::default();
        let mut buf = std::mem::take(&mut self.held);
        buf.push_str(piece);
        let mut i = 0usize;
        while i < buf.len() {
            if buf.as_bytes()[i] == b'<' {
                let rest = &buf[i..];
                let mut hit: Option<&'static str> = None;
                let mut partial = false;
                for tag in self.candidates() {
                    if rest.len() >= tag.len() {
                        if rest.starts_with(tag) {
                            hit = Some(tag);
                            break;
                        }
                    } else if tag.starts_with(rest) {
                        partial = true;
                    }
                }
                if let Some(tag) = hit {
                    self.take_tag(tag);
                    i += tag.len();
                    continue;
                }
                if partial {
                    // the piece ends inside a candidate tag: hold it, the next one decides
                    self.held.push_str(rest);
                    return out;
                }
            }
            let ch = match buf[i..].chars().next() {
                Some(c) => c,
                None => break,
            };
            let n = ch.len_utf8();
            self.take_char(ch, &buf[i..i + n], &mut out);
            i += n;
        }
        out
    }

    /// end of generation: a held prefix never completed, so it is text after all
    fn flush(&mut self) -> Split {
        let mut out = Split::default();
        let held = std::mem::take(&mut self.held);
        for ch in held.chars() {
            let mut b = [0u8; 4];
            let s = ch.encode_utf8(&mut b).to_string();
            self.take_char(ch, &s, &mut out);
        }
        if !self.lead.is_empty() && !self.drop_lead {
            out.content.push_str(&self.lead);
        }
        self.lead.clear();
        out
    }

    /// a complete tag: it never reaches the wire, it only moves the state
    fn take_tag(&mut self, tag: &str) {
        self.stripped += 1;
        self.lead.clear();
        if tag == THINK_OPEN {
            // reachable in `Lead` only: the block the model opened itself
            self.state = Think::Inside;
        } else {
            // `</think>`: the close of that block, or the stray this filter exists for.
            // `Body` stays `Body` - the answer has started, whitespace is content.
            if self.state != Think::Body {
                self.state = Think::Lead;
            }
            self.drop_lead = true;
        }
    }

    /// one ordinary character, into the half the state names
    fn take_char(&mut self, ch: char, s: &str, out: &mut Split) {
        match self.state {
            Think::Inside => out.reasoning.push_str(s),
            Think::Lead => {
                if ch.is_whitespace() {
                    self.lead.push_str(s);
                } else {
                    if !self.drop_lead {
                        out.content.push_str(&self.lead);
                    }
                    self.lead.clear();
                    self.drop_lead = false;
                    out.content.push_str(s);
                    self.state = Think::Body;
                }
            }
            Think::Body => out.content.push_str(s),
        }
    }
}

/// - #67: the SAME two tags, stripped off a STORED assistant `content` on the way IN
/// - `Some(clean)` when something was stripped, `None` when the string is left alone: a
///   history without a tag must render byte-identical to what it rendered before this fix
/// - a leading `<think>...</think>` block (and the whitespace around it) goes, because the
///   template puts the stored content INSIDE its own think block and a nested pair is what
///   the model imitates; a TRAILING `</think>` goes, because that is the shape Crow stored
/// - what is NOT touched: a `</think>` in the middle of a turn (it can be quoted text), a
///   `<think>` that never closes, and `reasoning_content` - the template's own field for this
fn strip_stored_think(content: &str) -> Option<String> {
    let mut s = content;
    let mut changed = false;
    if let Some(rest) = s.trim_start().strip_prefix(THINK_OPEN) {
        if let Some(i) = rest.find(THINK_CLOSE) {
            s = rest[i + THINK_CLOSE.len()..].trim_start();
            changed = true;
        }
    }
    while let Some(head) = s.trim_end().strip_suffix(THINK_CLOSE) {
        s = head.trim_end();
        changed = true;
    }
    if changed {
        Some(s.to_string())
    } else {
        None
    }
}

/// TASK J: the key a non-mapping `arguments` is carried into the render under
const RAW_KEY: &str = "_raw";

/// - the JSON kind of a value, for a refusal or a log line that names a field
fn kind_of(v: &serde_json::Value) -> &'static str {
    match v {
        serde_json::Value::Null => "null",
        serde_json::Value::Bool(_) => "a boolean",
        serde_json::Value::Number(_) => "a number",
        serde_json::Value::String(_) => "a string",
        serde_json::Value::Array(_) => "an array",
        serde_json::Value::Object(_) => "an object",
    }
}

/// - TASK K: the bytes AROUND the place `serde_json` gave up, so the `[chat]
///   normalised` line carries the defect itself and not only its coordinates
/// - `serde_json::Error` gives a 1-based line and column; this walks the string
///   once to turn that pair into a byte index, then shows 60 bytes either side
///   with a `<HERE>` marker, on char boundaries
/// - an error with no position (line 0) adds nothing
fn error_window(s: &str, e: &serde_json::Error) -> String {
    let (line, col) = (e.line(), e.column());
    if line == 0 {
        return String::new();
    }
    let mut at = 0usize;
    let mut l = 1usize;
    for (i, c) in s.char_indices() {
        if l == line {
            at = i + col.saturating_sub(1).min(s.len() - i);
            break;
        }
        if c == '\n' {
            l += 1;
            at = i + c.len_utf8();
        }
    }
    let mut at = at.min(s.len());
    while at > 0 && !s.is_char_boundary(at) {
        at -= 1;
    }
    let mut lo = at.saturating_sub(60);
    while lo > 0 && !s.is_char_boundary(lo) {
        lo -= 1;
    }
    let mut hi = (at + 60).min(s.len());
    while hi < s.len() && !s.is_char_boundary(hi) {
        hi += 1;
    }
    format!(" (byte {at} of {}: {:?} <HERE> {:?})", s.len(), &s[lo..at], &s[at..hi])
}

/// - the first 200 bytes of a value, on a char boundary, for the diagnostic lines
/// - a string is shown as its own text, anything else as its compact JSON
fn head200(v: &serde_json::Value) -> String {
    let s = match v.as_str() {
        Some(s) => s.to_string(),
        None => v.to_string(),
    };
    let mut n = s.len().min(200);
    while n > 0 && !s.is_char_boundary(n) {
        n -= 1;
    }
    if n == s.len() {
        s
    } else {
        format!("{}... ({} B total)", &s[..n], s.len())
    }
}

/// - #29 A7: Crow sends `tool_calls[].function.arguments` as a JSON STRING (`crow_core.py:5068`,
///   stored verbatim at `:3756-3761`, re-sent whole every turn at `:3783-3785`, `:4866`)
/// - the chat template iterates it with `|items` (`chat_template.jinja:136`), which needs a MAPPING
/// - measured 2026-09-09: the Python oracle raises `TypeError: Can only get item pairs from a
///   mapping` on the string form; minijinja raises `cannot convert value into pairs (in chat:136)`
/// - TASK J (2026-09-17): NOTHING that is not a mapping reaches that line any more. One
///   unterminated `arguments` string used to kill every later turn of a session: Crow stores it,
///   re-sends it, gets a 400, and has no way to drop the message again (`crow_core.py:13485-13487`)
///
/// | `arguments` as it arrives | what the render sees | why |
/// |---|---|---|
/// | an object | unchanged | the template's own form |
/// | a string that parses to an object | that object | Crow's form, the A7 path, bytes unchanged |
/// | the EMPTY string | unchanged | the template's `arguments != ''` guard skips it; it is what the engine's open chunk sends for a call with no fragments |
/// | any other string (truncated JSON, prose, a JSON array / number / bool / string) | `{"_raw": "<the string, verbatim>"}` | the model must still SEE what the previous turn asked for; `{}` would rewrite history silently, and the leading underscore says it is not a declared parameter |
/// | JSON `null` | `{}` | null means there are no arguments; a `_raw` of `"null"` would invent content that was never sent |
/// | any other value (array, number, bool) | `{"_raw": "<its compact JSON>"}` | same reason as the string row |
///
/// - Every rewrite returns a note naming the message index, the tool_call index and the first
///   200 bytes of the offending value; `chat_route` puts each on one stderr line.
/// - `function` may be absent: the template then reads `name` / `arguments` off the call itself
///   (`tool_call.function is defined`), and so does this.
///
/// - #67 (2026-09-18): the SECOND end of the reasoning filter, the same idea one field over.
///   A stored assistant `content` that carries `<think>` / `</think>` is stripped by
///   `strip_stored_think` BEFORE the render. Measured against this template, not assumed:
///   the assistant branch renders `'<think>\n' + reasoning_content|trim + '\n</think>\n\n' +
///   content` (`chat_template.jinja`, the `preserve_thinking` branch), so a stored `</think>`
///   does NOT cut the turn - it is rendered VERBATIM inside the template's own think block,
///   and the turn the model reads back carries two closing tags. That nested pair is what it
///   imitates. The client cannot repair its own history (`crow_core.py:13485-13487`), so the
///   engine repairs it on the way in.
///
/// | stored assistant `content` | what the render sees | why |
/// |---|---|---|
/// | no tag | unchanged, byte for byte | the A7 oracle renders are untouched |
/// | ends with `</think>` (whitespace allowed after it) | the tag and the whitespace around it go, the text stays | the shape Crow stored from a pre-fix stream |
/// | starts with `<think>...</think>` | the whole block goes, the answer after it stays | the template puts its own think block around this; a nested pair is what the model imitates |
/// | a `</think>` in the MIDDLE, or a `<think>` that never closes | unchanged | it can be quoted text, and neither shape nests |
/// | `reasoning_content` | never touched | it is the template's own field for prior reasoning, and this is where the stream now puts it |
fn normalize_messages(messages: &serde_json::Value) -> (serde_json::Value, Vec<String>) {
    let mut doc = messages.clone();
    let mut notes = Vec::new();
    let arr = match doc.as_array_mut() {
        Some(a) => a,
        None => return (doc, notes),
    };
    for (mi, m) in arr.iter_mut().enumerate() {
        // #67: the stored assistant turn, before anything else looks at it
        if m.get("role").and_then(|r| r.as_str()) == Some("assistant") {
            if let Some(c) = m.get_mut("content") {
                let stripped = c.as_str().and_then(strip_stored_think);
                if let Some(clean) = stripped {
                    notes.push(format!(
                        "message {mi} assistant content carried a <think>/</think> tag this \
                         template renders verbatim inside its own think block (#67); stripped \
                         to {} B: {}",
                        clean.len(),
                        head200(c)
                    ));
                    *c = serde_json::Value::String(clean);
                }
            }
        }
        let calls = match m.get_mut("tool_calls").and_then(|v| v.as_array_mut()) {
            Some(c) => c,
            None => continue,
        };
        for (ci, call) in calls.iter_mut().enumerate() {
            // the template uses `tool_call.function` when it is defined, else the call itself
            let target = if call.get("function").is_some() {
                call.get_mut("function")
            } else {
                Some(call)
            };
            let args = match target.and_then(|t| t.get_mut("arguments")) {
                Some(a) => a,
                None => continue,
            };
            // the two forms that already render, in the order they arrive: the object the
            // template wants, and the empty string its own guard skips
            if args.is_object() || args.as_str() == Some("") {
                continue;
            }
            // TASK K: WHY the string is not a mapping is the one thing the old note
            // did not say. `serde_json`'s own message plus the byte window it stopped
            // in turns the next case into a fix instead of another bisect.
            let mut why = String::new();
            if let Some(s) = args.as_str() {
                match serde_json::from_str::<serde_json::Value>(s) {
                    Ok(v) if v.is_object() => {
                        *args = v;
                        continue;
                    }
                    Ok(v) => why = format!("; it parses, as {}", kind_of(&v)),
                    Err(e) => why = format!("; serde_json: {e}{}", error_window(s, &e)),
                }
            }
            let (fixed, how) = if args.is_null() {
                (serde_json::json!({}), "{}".to_string())
            } else {
                let raw = match args.as_str() {
                    Some(s) => s.to_string(),
                    None => args.to_string(),
                };
                (serde_json::json!({ RAW_KEY: raw }), format!("{{\"{RAW_KEY}\": ...}}"))
            };
            notes.push(format!(
                "message {mi} tool_call {ci} function.arguments is {} the template cannot \
                 iterate{why}, rendered as {how}: {}",
                kind_of(args),
                head200(args)
            ));
            *args = fixed;
        }
    }
    (doc, notes)
}

/// - TASK J: the message shapes THIS chat template cannot render, named by index and field
/// - run on the NORMALIZED messages, so `arguments` is already a mapping the template can iterate
/// - every rule was MEASURED against `models/Qwen3.8-Flash-Next-original/chat_template.jinja`;
///   a shape that renders is never refused here, so no request that worked before is refused now
/// - `Err` is the 400 body: it names the message index and the field, which the bare
///   `chat template render failed: ... (in chat:NNN)` did not
///
/// | shape | template | answer |
/// |---|---|---|
/// | a role other than system / user / assistant / tool | `raise_exception` at `chat:160` | 400, names the index and the role |
/// | a system message that is not first | `raise_exception` at `chat:106` | 400, names the index |
/// | `content` an object, a number or a boolean | `raise_exception` at `chat:39` | 400, names the index |
/// | `content` a list part that is not an object | containment check or `chat:33` | 400, names the index and the part |
/// | `content` a list part with no `text`, `image`, `image_url` or `video` | `raise_exception` at `chat:33` | 400, names the index and the part |
/// | `content` a STRING, `null`, absent, or a list of text / image_url parts | renders | served |
/// | `tool_calls` absent or `null` | the `if` is falsy | served |
/// | `tool_calls` not an array | SILENTLY DROPPED (`is iterable and is not mapping`) | 400: a dropped call is a history that no longer says what happened |
/// | a `tool_calls` entry that is not an object, or whose `function` is not an object | `+` on undefined at `chat:128` | 400, names the index and the entry |
/// | `function.name` not a string | `+` on undefined / none / number at `chat:128-133` | 400, names the index and the entry |
/// | `reasoning_content` of any type | a non-string is ignored (`chat:112`) | served |
fn check_messages(messages: &serde_json::Value) -> Result<(), String> {
    let arr = match messages.as_array() {
        Some(a) => a,
        None => return Ok(()),
    };
    for (i, m) in arr.iter().enumerate() {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("");
        if !matches!(role, "system" | "user" | "assistant" | "tool") {
            return Err(format!(
                "message {i} has role {role:?}; the template renders only system, user, assistant and tool"
            ));
        }
        if role == "system" && i > 0 {
            return Err(format!(
                "message {i} is a system message; the template requires the system message first"
            ));
        }
        check_content(i, m.get("content"))?;
        match m.get("tool_calls") {
            None | Some(serde_json::Value::Null) => {}
            Some(serde_json::Value::Array(cs)) => {
                for (j, c) in cs.iter().enumerate() {
                    check_tool_call(i, j, c)?;
                }
            }
            Some(other) => {
                return Err(format!(
                    "message {i} tool_calls is {}, the template needs an array",
                    kind_of(other)
                ))
            }
        }
    }
    Ok(())
}

/// the `content` rules of the table on `check_messages`, for one message
fn check_content(i: usize, content: Option<&serde_json::Value>) -> Result<(), String> {
    let parts = match content {
        None | Some(serde_json::Value::Null) | Some(serde_json::Value::String(_)) => return Ok(()),
        Some(serde_json::Value::Array(a)) => a,
        Some(other) => {
            return Err(format!(
                "message {i} content is {}, the template renders a string, a list of parts or null",
                kind_of(other)
            ))
        }
    };
    for (j, part) in parts.iter().enumerate() {
        let obj = match part.as_object() {
            Some(o) => o,
            None => {
                return Err(format!(
                    "message {i} content part {j} is {}, the template renders text and image_url blocks only",
                    kind_of(part)
                ))
            }
        };
        let ty = obj.get("type").and_then(|t| t.as_str()).unwrap_or("");
        let known = obj.contains_key("text")
            || obj.contains_key("image")
            || obj.contains_key("image_url")
            || obj.contains_key("video")
            || ty == "image"
            || ty == "video";
        if !known {
            return Err(format!(
                "message {i} content part {j} is neither a text nor an image_url block: {}",
                head200(part)
            ));
        }
    }
    Ok(())
}

/// the `tool_calls` entry rules of the table on `check_messages`, for one entry
fn check_tool_call(i: usize, j: usize, call: &serde_json::Value) -> Result<(), String> {
    if !call.is_object() {
        return Err(format!("message {i} tool_call {j} is {}, not an object", kind_of(call)));
    }
    let target = match call.get("function") {
        None => call,
        Some(f) if f.is_object() => f,
        Some(other) => {
            return Err(format!(
                "message {i} tool_call {j} function is {}, not an object",
                kind_of(other)
            ))
        }
    };
    match target.get("name") {
        Some(serde_json::Value::String(_)) => Ok(()),
        other => Err(format!(
            "message {i} tool_call {j} has no string function.name (it is {}); the template \
             renders `<function=` + the name",
            other.map(kind_of).unwrap_or("absent")
        )),
    }
}

/// - TASK J: one line per message for the stderr dump that goes with every messages 400
/// - names the index, the role, the content kind and, per tool call, the name and the first
///   200 bytes of `arguments`, so a 400 is diagnosable from the log alone
/// - the RAW messages go in here, not the normalized ones: the log has to show what arrived
fn message_digest(messages: &serde_json::Value) -> Vec<String> {
    let arr = match messages.as_array() {
        Some(a) => a,
        None => return vec![format!("messages is {}", kind_of(messages))],
    };
    let mut out = Vec::with_capacity(arr.len());
    for (i, m) in arr.iter().enumerate() {
        let role = m.get("role").and_then(|r| r.as_str()).unwrap_or("<no role>");
        let content = match m.get("content") {
            None => "absent".to_string(),
            Some(serde_json::Value::String(s)) => format!("string({} B)", s.len()),
            Some(serde_json::Value::Array(a)) => format!("list({} part(s))", a.len()),
            Some(other) => kind_of(other).to_string(),
        };
        let mut line = format!("message {i} role={role} content={content}");
        match m.get("tool_calls") {
            None => {}
            Some(serde_json::Value::Array(cs)) => {
                line.push_str(&format!(" tool_calls={}", cs.len()));
                for (j, c) in cs.iter().enumerate() {
                    let t = match c.get("function") {
                        Some(f) if f.is_object() => f,
                        _ => c,
                    };
                    let name = t.get("name").map(head200).unwrap_or_else(|| "<absent>".to_string());
                    let args = t.get("arguments");
                    line.push_str(&format!(
                        " [{j} name={name:?} arguments={} {:?}]",
                        args.map(kind_of).unwrap_or("absent"),
                        args.map(head200).unwrap_or_default()
                    ));
                }
            }
            Some(other) => line.push_str(&format!(" tool_calls={} (not an array)", kind_of(other))),
        }
        out.push(line);
    }
    out
}

/// - TASK J: the one place a messages 400 is logged, so both refusal paths log alike
/// - the reason first, then `message_digest` of the RAW messages, one line each
fn log_messages_400(reason: &str, raw: &serde_json::Value) {
    tracing::warn!(target: "chat", "[chat] 400 {reason}");
    for l in message_digest(raw) {
        tracing::warn!(target: "chat", "[chat]   {l}");
    }
}

/// write one SSE frame and flush it; `false` means the client is gone
fn sse_send<W: Write>(w: &mut W, text: &str) -> bool {
    match w.write_all(text.as_bytes()).and_then(|_| w.flush()) {
        Ok(()) => true,
        Err(e) => {
            tracing::warn!(target: "chat", "[chat] write failed, aborting the generation: {e}");
            false
        }
    }
}

// ------------------------------------------------------- #54 the gone-client probe

/// the four `revents` bits this probe reads, by their Linux values. They are named here and
/// not taken from `libc` so the decision below is pure, testable and compiled on every
/// platform; `the_probe_reads_the_kernels_own_revents_bits` asserts them against `libc` on
/// Linux, which is the only platform that polls.
const PROBE_ERR: i16 = 0x008; // POLLERR
const PROBE_HUP: i16 = 0x010; // POLLHUP
const PROBE_NVAL: i16 = 0x020; // POLLNVAL
const PROBE_RDHUP: i16 = 0x2000; // POLLRDHUP (Linux's own bit)

/// - #54: what ONE probe of the request socket found
/// - `Open`: nothing to report. Unread request bytes are NOT a report - this asks about the
///   peer's READ side, never about what it still has to say, so `POLLIN` is not requested.
/// - `Eof`: the peer sent FIN. `close()` and `shutdown(Write)` are the SAME wire event and no
///   kernel can tell them apart (measured 2026-09-18: both give `POLLIN|POLLRDHUP` and
///   `recv(MSG_PEEK) == 0`, and the TCP state is `CLOSE_WAIT` either way), so this on its own
///   is not "the client is gone" - see `gone_reason`.
/// - `Hup`: the peer RESET the connection, or the descriptor is unusable (`POLLERR`, `POLLHUP`,
///   `POLLNVAL`). Unambiguous at any time, and the only thing a write would have discovered.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Peer {
    Open,
    Eof,
    Hup,
}

/// - #54: pure: what `poll`'s return value and the `revents` of ONE descriptor mean
/// - `rc < 0` is `EINTR` or a bad argument, `rc == 0` is "no bit set": both are `Open`. A
///   signal that interrupts the probe must not end a generation.
fn peer_from_poll(rc: i32, revents: i16) -> Peer {
    if rc <= 0 {
        return Peer::Open;
    }
    if revents & (PROBE_ERR | PROBE_HUP | PROBE_NVAL) != 0 {
        Peer::Hup
    } else if revents & PROBE_RDHUP != 0 {
        Peer::Eof
    } else {
        Peer::Open
    }
}

/// - #54: pure: `true` when step `step` of the generation carries a probe
/// - `every == 1` (`PROBE_EVERY`) is every step; `every == 0` would disable the probe
fn probe_due(step: usize, every: usize) -> bool {
    every > 0 && step.is_multiple_of(every)
}

/// - #54: pure: the decision, and the reason the `[chat]` line names
/// - `base_eof` is what the FIRST probe of this request found, taken before the prefill.
/// - A reset is gone at any time: nothing else can produce `POLLERR`/`POLLHUP` here.
/// - An EOF that appears DURING the generation is gone: the client whose read side was open
///   when this request started has closed the connection while it was our turn to speak.
/// - An EOF that was ALREADY there is NOT gone: a client that half-closed its write side
///   after sending the request is waiting for the answer, and since the kernel cannot tell
///   that client from one that left, it gets its answer. (No client of record does it - curl
///   and Python `requests` keep the read side open until the response arrives - but a
///   `shutdown(Write)` client is legal HTTP and must not lose its document.)
fn gone_reason(base_eof: bool, p: Peer) -> Option<&'static str> {
    match p {
        Peer::Open => None,
        Peer::Hup => Some("the peer reset the connection (POLLERR/POLLHUP/POLLNVAL)"),
        Peer::Eof if !base_eof => {
            Some("the peer closed the connection mid-generation (POLLRDHUP; its read side was open at the first probe)")
        }
        Peer::Eof => None,
    }
}

/// - #54: ONE probe of `fd`: `poll` with timeout 0. No byte is read, no byte is written, and
///   nothing goes on the wire - a zero-byte write sends no segment, so it could not tell a
///   closed peer from a live one anyway.
/// - `events` asks for `POLLRDHUP` alone; `POLLERR`, `POLLHUP` and `POLLNVAL` are reported
///   whether they were asked for or not, and `POLLIN` would fire on unread request bytes.
#[cfg(target_os = "linux")]
fn poll_peer(fd: i32) -> Peer {
    let mut p = libc::pollfd { fd, events: PROBE_RDHUP, revents: 0 };
    // unsafe: one `poll` on one descriptor this process owns, timeout 0, no allocation
    let rc = unsafe { libc::poll(&mut p, 1, 0) };
    peer_from_poll(rc, p.revents)
}

/// #54: every other platform has no `POLLRDHUP`, so the document path there is what it was
/// before this fix: the generation runs its budget out. The Windows build compiles, and the
/// SSE path detects a gone client by its failed flush on every platform.
#[cfg(not(target_os = "linux"))]
fn poll_peer(_fd: i32) -> Peer {
    Peer::Open
}

/// #54: the descriptor of the request socket, borrowed for the length of the request.
/// `None` on a platform that cannot probe, which makes the whole watch inert.
#[cfg(target_os = "linux")]
fn socket_fd(stream: &TcpStream) -> Option<i32> {
    use std::os::unix::io::AsRawFd;
    Some(stream.as_raw_fd())
}

#[cfg(not(target_os = "linux"))]
fn socket_fd(_stream: &TcpStream) -> Option<i32> {
    None
}

/// - #54: the gone-client watch of ONE `stream:false` request
/// - Why it exists: the SSE path learns that its client left from the next failed flush
///   (`sse_send`), and the document path writes NOTHING until the generation is over. So a
///   client that disconnected after a `stream:false` POST with `max_tokens: 512` held the one
///   slot of this process for the whole budget - the one asymmetry between the two sinks.
/// - What it holds: the descriptor (borrowed - the `TcpStream` of `serve_one` owns it for the
///   whole request and closes it), and what the FIRST probe found.
/// - `Default` is the no-socket form: `fd` `None`, so `gone` is always `None` and every unit
///   test and every non-Linux build runs the loop exactly as it ran before #54.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct ClientProbe {
    fd: Option<i32>,
    /// the baseline: `true` when the peer had already closed its write side before the first
    /// decode step, which is a client that finished SENDING, not one that left
    base_eof: bool,
}

impl ClientProbe {
    /// - #54: the watch of one request, with its baseline taken HERE
    /// - the baseline is taken before the prefill and after the body, the head, the template
    ///   render and the tokenizer: a half-closing client's FIN travels the loopback in
    ///   microseconds and those steps cost milliseconds, so a `shutdown(Write)` sent with the
    ///   request is in long before this, while a client that dies during the prefill is not
    ///   (its EOF appears at a later probe and is therefore gone, as it should be)
    fn new(fd: Option<i32>) -> Self {
        let base_eof = matches!(fd.map(poll_peer), Some(Peer::Eof));
        if base_eof {
            tracing::info!(target: "chat",
                "[chat] the client had already closed its write side when this request started \
                 (POLLRDHUP at the first probe): a client that finished SENDING, not one that \
                 left - the document is generated and answered as before (#54)"
            );
        }
        ClientProbe { fd, base_eof }
    }

    /// one probe at step `step`: `Some(reason)` when this client is gone and the loop must stop
    fn gone(&mut self, step: usize) -> Option<&'static str> {
        let fd = self.fd?;
        if !probe_due(step, PROBE_EVERY) {
            return None;
        }
        gone_reason(self.base_eof, poll_peer(fd))
    }
}

/// - #39 B3a: where the per token side effects of ONE generation go
/// - the SSE writer is one implementation, the collector behind the non streaming document
///   is the other; `chat_generate` stays the only generation loop in this file
/// - every method returns `false` for "the client is gone, stop the loop"
trait ChatSink {
    /// before the first token: the role delta of the stream, nothing for a document
    fn open(&mut self, c: &ChunkCtx) -> bool;
    /// one parser fragment, in arrival order
    fn on_emit(&mut self, c: &ChunkCtx, e: &Emit) -> bool;
    /// #67: one `reasoning_content` piece the think filter took out of the content
    fn on_reasoning(&mut self, c: &ChunkCtx, text: &str) -> bool;
    /// after the last token: the final chunk plus `[DONE]`, nothing for a document
    fn on_finish(&mut self, c: &ChunkCtx, a: &FinishArgs) -> bool;
    /// - #54: between two decode steps: `false` means the client is gone and the loop stops
    /// - the default is `true`, and that is the SSE path: a sink that writes and flushes per
    ///   token learns of a gone client from its next failed flush, which is what it has always
    ///   done. The document sink writes nothing until the generation is over, so it overrides
    ///   this and asks the socket (`ClientProbe`, 7.11.18).
    fn still_there(&mut self, _step: usize) -> bool {
        true
    }
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
    fn open(&mut self, c: &ChunkCtx) -> bool {
        sse_send(&mut self.w, &sse_frame(&chunk_role(c)))
    }
    fn on_emit(&mut self, c: &ChunkCtx, e: &Emit) -> bool {
        let doc = match e {
            Emit::Content(t) => chunk_content(c, t),
            Emit::Call { index, id: call_id, name } => chunk_tool_open(c, *index, call_id, name),
            Emit::Args { index, text } => chunk_tool_args(c, *index, text),
        };
        sse_send(&mut self.w, &sse_frame(&doc))
    }
    fn on_reasoning(&mut self, c: &ChunkCtx, text: &str) -> bool {
        sse_send(&mut self.w, &sse_frame(&chunk_reasoning(c, text)))
    }
    fn on_finish(&mut self, c: &ChunkCtx, a: &FinishArgs) -> bool {
        sse_send(&mut self.w, &sse_frame(&chunk_finish(c, a))) && sse_send(&mut self.w, SSE_DONE)
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
    /// #67: every reasoning piece in order, the `message.reasoning_content` of the document
    reasoning: String,
    /// one entry per tool call index, in the order the parser opened them
    calls: Vec<CallBuf>,
    /// #54: the gone-client watch of this request. `Default` holds no descriptor and never
    /// fires, which is what every unit test of this sink gets.
    probe: ClientProbe,
}

impl CollectSink {
    /// #54: the document sink of a LIVE request, watching the socket the one document will go
    /// out on. The baseline probe of `ClientProbe::new` is taken here, before the prefill.
    fn watching(stream: &TcpStream) -> Self {
        CollectSink { probe: ClientProbe::new(socket_fd(stream)), ..Default::default() }
    }
}

impl ChatSink for CollectSink {
    fn open(&mut self, _c: &ChunkCtx) -> bool {
        true
    }
    fn on_emit(&mut self, _c: &ChunkCtx, e: &Emit) -> bool {
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
    fn on_reasoning(&mut self, _c: &ChunkCtx, text: &str) -> bool {
        self.reasoning.push_str(text);
        true
    }
    fn on_finish(&mut self, _c: &ChunkCtx, _a: &FinishArgs) -> bool {
        true
    }
    /// #54: the one sink that has to ASK. One `poll` per step, and one loud line when it fires.
    fn still_there(&mut self, step: usize) -> bool {
        match self.probe.gone(step) {
            None => true,
            Some(why) => {
                tracing::info!(target: "chat",
                    "[chat] the client is gone at step {step}: {why} - ending the generation, \
                     the slot is free for the next request (#54)"
                );
                false
            }
        }
    }
}

/// - #29 A7: one sink call per parser fragment, in order
/// - the chunk counters of the `[chat]` line are counted HERE, so both sinks count alike
/// - `false` means the client is gone and the generation loop must stop
/// - TASK K: the `Emit::Args` fragments of one request, concatenated per call index
/// - the same accumulation Crow does (`crow_core.py:5069`), so what is checked here is
///   exactly what the client will store and re-send
fn accumulate_args(pieces: &[Emit], acc: &mut Vec<String>) {
    for e in pieces {
        if let Emit::Args { index, text } = e {
            while acc.len() <= *index {
                acc.push(String::new());
            }
            acc[*index].push_str(text);
        }
    }
}

/// - #67: the reasoning filter runs HERE, between the tool-call parser and the sink, on
///   `Emit::Content` alone. Arguments fragments are never touched: a `</think>` inside a
///   tool parameter value is that value's business, and the `arguments` contract of 7.11.14
///   must stay byte-identical.
/// - The malformed-tool-call path carries its raw markup as `Emit::Content`, so it is
///   filtered like any other content - the tag never leaves as content on any path.
/// - A piece that is entirely held back or entirely stripped sends NO frame, and is counted
///   in neither column: the counters name frames written, which is what they always named.
fn send_emits(
    sink: &mut dyn ChatSink,
    c: &ChunkCtx,
    pieces: &[Emit],
    think: &mut ThinkFilter,
    counts: &mut Chunks,
) -> bool {
    for e in pieces {
        match e {
            Emit::Content(t) => {
                let split = think.push(t);
                if !send_split(sink, c, &split, counts) {
                    return false;
                }
            }
            Emit::Call { .. } | Emit::Args { .. } => {
                counts.tool += 1;
                if !sink.on_emit(c, e) {
                    return false;
                }
            }
        }
    }
    true
}

/// #67: the two halves of one filtered piece, in wire order: reasoning first, then content
fn send_split(sink: &mut dyn ChatSink, c: &ChunkCtx, split: &Split, counts: &mut Chunks) -> bool {
    if !split.reasoning.is_empty() {
        counts.reasoning += 1;
        if !sink.on_reasoning(c, &split.reasoning) {
            return false;
        }
    }
    if !split.content.is_empty() {
        counts.content += 1;
        if !sink.on_emit(c, &Emit::Content(split.content.clone())) {
            return false;
        }
    }
    true
}

/// the frames one generation wrote, per kind; the `[chat]` line prints all three
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct Chunks {
    content: usize,
    reasoning: usize,
    tool: usize,
}

/// - #39 B3a: the ONE `chat.completion` document a `stream:false` request answers
/// - `usage` and `timings` are `usage_json` and `timings_json`, the objects of the final
///   stream chunk, and both are ALWAYS present: one document shape, and the probe-suite
///   reads `usage.completion_tokens` (`probe-suite.py:681-683`) while sending neither
///   `stream_options` nor `timings_per_token`
/// - `content` is always a string, empty when the answer was a tool call alone
/// - `tool_calls` appears only when the parser closed at least one call, in the OpenAI
///   non streaming shape (`id`, `type`, `function`), `arguments` a JSON STRING
/// - #67: `reasoning_content` is present ONLY when the think filter stripped a block the
///   model opened itself. The template renders `enable_thinking false`, so that is rare and
///   the ordinary document is the document it always was, field for field
///   (`probe-suite.py:680` reads the key when it is there)
/// - pure: the whole document contract is one function the test drives directly
fn completion_json(
    c: &ChunkCtx,
    content: &str,
    reasoning: &str,
    calls: &[CallBuf],
    finish: &str,
    t: &Timing,
) -> serde_json::Value {
    let mut message = serde_json::json!({ "role": "assistant", "content": content });
    if !reasoning.is_empty() {
        if let Some(obj) = message.as_object_mut() {
            obj.insert("reasoning_content".to_string(), serde_json::json!(reasoning));
        }
    }
    if !calls.is_empty() {
        let arr: Vec<serde_json::Value> = calls
            .iter()
            .map(|c| {
                // TASK K: the producer end of the `arguments` contract. The parser's
                // invariant says this string is a JSON object for every input, and
                // `toolcall::tests` proves it over every markup shape and every piece
                // split; this is the last line of defence for the one form the engine
                // can still repair before it leaves - the `stream:false` document.
                // A string that is not an object is replaced by `{"_raw": ...}`, the
                // same shape `normalize_messages` uses on the way back in, so a client
                // that stores and re-sends the turn cannot poison its own history.
                let (arguments, note) = args_object_or_raw(&c.arguments);
                if let Some(n) = note {
                    tracing::info!(target: "chat", "[chat] tool_call {} arguments repaired before the document: {n}", c.id);
                }
                serde_json::json!({
                    "id": c.id,
                    "type": "function",
                    "function": { "name": c.name, "arguments": arguments },
                })
            })
            .collect();
        if let Some(obj) = message.as_object_mut() {
            obj.insert("tool_calls".to_string(), serde_json::Value::Array(arr));
        }
    }
    serde_json::json!({
        "id": c.id,
        "object": "chat.completion",
        "created": c.created,
        "model": c.model,
        "choices": [{
            "index": 0,
            "message": message,
            "finish_reason": finish,
        }],
        "usage": usage_json(t),
        "timings": timings_json(t),
    })
}

/// - TASK K: the `arguments` string of one finished call, checked at the SOURCE
/// - `Ok` shape (a JSON object): returned verbatim, no note — the byte-identical path
/// - anything else: `{"_raw": "<verbatim>"}` plus a note that carries `serde_json`'s
///   own message and the byte window it stopped in
/// - the empty string is what a call with no `arguments` fragment carries and is
///   left alone: `""` is the form llama-server and OpenAI send and the template's
///   own guard skips it
/// - pure, so the test drives it without a socket or an engine
fn args_object_or_raw(args: &str) -> (String, Option<String>) {
    if args.is_empty() {
        return (args.to_string(), None);
    }
    match serde_json::from_str::<serde_json::Value>(args) {
        Ok(serde_json::Value::Object(_)) => (args.to_string(), None),
        Ok(v) => (
            serde_json::json!({ RAW_KEY: args }).to_string(),
            Some(format!("it parses, as {}, not an object: {}", kind_of(&v), head200(&serde_json::json!(args)))),
        ),
        Err(e) => (
            serde_json::json!({ RAW_KEY: args }).to_string(),
            Some(format!(
                "serde_json: {e}{}: {}",
                error_window(args, &e),
                head200(&serde_json::json!(args))
            )),
        ),
    }
}

/// a JSON response plus its status, so the caller can log one label
fn respond_json(
    stream: &mut TcpStream,
    status: &'static str,
    doc: &serde_json::Value,
) -> &'static str {
    if let Err(e) = respond(stream, status, &doc.to_string()) {
        tracing::warn!(target: "serve", "[serve] response write failed: {e}");
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
    // TASK J: `normalize_messages` now leaves NO non-mapping for that `|items`, and every
    // rewrite it made goes to stderr; `check_messages` refuses what the template cannot
    // render with a message that names the index and the field, before any render.
    let (msgs, notes) = normalize_messages(&req.messages);
    for n in &notes {
        tracing::info!(target: "chat", "[chat] normalised: {n}");
    }
    if let Err(e) = check_messages(&msgs) {
        log_messages_400(&e, &req.messages);
        return respond_json(stream, "400 Bad Request", &error_json(&e));
    }
    // #74: `reasoning_effort` rides as the fourth template variable. It is `None` for every
    // request that does not think, so the ids of a request that names neither door are the
    // ids of record.
    let ids = match tk.encode_chat_effort(
        &msgs,
        req.tools.as_ref(),
        true,
        req.enable_thinking,
        req.reasoning_effort,
    ) {
        Ok(v) => v,
        Err(e) => {
            // a render that still fails is a shape `check_messages` does not know: the dump
            // is what turns the next one into a fix instead of another bisect
            log_messages_400(&e, &req.messages);
            return respond_json(stream, "400 Bad Request", &error_json(&e));
        }
    };
    if ids.is_empty() {
        return respond_json(stream, "400 Bad Request", &error_json("the rendered prompt is empty"));
    }
    // #VIT: run the visual tower for the request's images, expand every
    // image_pad into its visual tokens, arm the mrope span tables. With
    // CROW_VIT=0 the engine holds no tower and the request falls through as
    // the placeholder of record (the single image_pad rides as a token).
    let (ids, vision_plan): (Vec<u32>, Option<crow_nest_engine::vit::VisionPlan>) = if !req.images.is_empty() && srv.eng.has_vision() {
        tracing::info!(target: "vit", "[vit-chat] {} image(s) in request, decoding data URLs ...", req.images.len());
        let mut bytes = Vec::with_capacity(req.images.len());
        for (i, url) in req.images.iter().enumerate() {
            match decode_data_url(url) {
                Ok((_mime, raw)) => bytes.push(raw),
                Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&format!("image {i}: {e}"))),
            }
        }
        let vit_t0 = std::time::Instant::now();
        let plan = match unsafe { srv.eng.build_vision_plan(&ids, &bytes) } {
            Ok(p) => p,
            Err(e) => return respond_json(stream, "400 Bad Request", &error_json(&e)),
        };
        let vit_ms = vit_t0.elapsed().as_secs_f64() * 1e3;
        // TASK K: free VRAM and the engine's live allocation count ride this line.
        // The 2026-09-17 panic was an out-of-memory in exactly this window and the
        // log said nothing about how much room the card still had; a session that
        // walks this number down is now visible turn by turn.
        let (live_n, live_b) = crow_nest_engine::cuda::live_dev();
        tracing::info!(target: "vit",
            "[vit-chat] {} image(s), {} visual token(s), grids {:?}, mrope delta {}, vision {} ms (decode + preprocess + tower), free VRAM {:.1} MiB, engine live allocs {live_n} = {:.1} MiB",
            req.images.len(),
            plan.n_visual,
            plan.grids,
            plan.delta,
            vit_ms,
            unsafe { crow_nest_engine::cuda::free_vram_bytes() } as f64 / (1u64 << 20) as f64,
            live_b as f64 / (1u64 << 20) as f64
        );
        let expanded = plan.ids.clone();
        (expanded, Some(plan))
    } else {
        (ids, None)
    };
    // #26 review: the prompt alone is the only 413 case; a budget that does not fit is CLAMPED,
    // not refused (the old combined check was dead, `max_tokens` is capped at 32768 first)
    //
    // TASK K: this now runs BEFORE `begin_vision`, not after. The mrope span tables are
    // `span = prompt + budget` rows, and with the RAW `max_tokens` that span was bounded
    // only by the 32,768 cap and by whatever prompt length the body carried - a request
    // that the next line refuses with 413 still allocated its tables first. With the
    // clamped budget the span is at most `n_ctx`, which is exactly what
    // `vit::reserve_bytes` sets aside, and a refused request allocates nothing (the plan
    // drops here and frees its splice buffer). The table CONTENT of a served request is
    // unchanged: only rows past the budget disappear, and no kernel ever read those.
    let budget = match clamped_max_tokens(ids.len(), req.max_tokens, srv.n_ctx) {
        Some(n) => n,
        None => {
            let m = format!("prompt {} tokens over n_ctx {}", ids.len(), srv.n_ctx);
            return respond_json(stream, "413 Payload Too Large", &error_json(&m));
        }
    };
    if budget != req.max_tokens {
        tracing::info!(target: "chat",
            "[chat] max_tokens {} clamped to {} (prompt {}, n_ctx {})",
            req.max_tokens,
            budget,
            ids.len(),
            srv.n_ctx
        );
        req.max_tokens = budget;
    }
    if let Some(plan) = vision_plan {
        unsafe { srv.eng.begin_vision(plan, ids.len() + req.max_tokens) };
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
    srv.stream_head_sent = true;
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
/// - #54: because nothing reaches the socket before the document, this path cannot learn of a
///   gone client from a failed write the way the stream does - so `CollectSink` watches the
///   socket between the decode steps (`ClientProbe`, 7.11.18) and the generation ends when the
///   client is gone instead of running the whole `max_tokens` budget out
fn chat_document(stream: &mut TcpStream, srv: &mut Srv, req: &ChatReq, ids: &[u32]) -> &'static str {
    let tk = match crow_nest_engine::tokenizer::global() {
        Ok(t) => t,
        Err(e) => return respond_json(stream, "500 Internal Server Error", &error_json(e)),
    };
    // #54: the sink watches the request socket from here on - see `ClientProbe`
    let mut sink = CollectSink::watching(stream);
    let out = chat_generate(srv, req, ids, tk, &mut sink);
    let doc = completion_json(
        &ChunkCtx::new(&out.id, out.created, &req.model),
        &sink.content,
        &sink.reasoning,
        &sink.calls,
        out.finish,
        &out.timing,
    );
    // #54: the document is written even for a client that is gone - one document shape, and
    // the write either lands in a socket nobody reads or fails with one `[serve]` line. The
    // STATUS says which it was, the way `chat_stream` has always said it.
    let status = respond_json(stream, "200 OK", &doc);
    if out.aborted {
        "200 OK (client gone)"
    } else {
        status
    }
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
    /// a sink call refused, or the #54 probe found the client gone: the loop stopped early
    aborted: bool,
}

/// - #37: the two preconditions `Engine::trickle_tick` ASSERTS (`gen.rs:3128-3129`)
/// - exact NVFP4 tier only: a `CROW_COLD_TIER` process has no three-way exchange
/// - spare hot slots must exist (`stride > n`), or there is nothing to copy into
/// - read once at start and once per request, never inside the token loop
/// - a misconfigured process therefore logs one line instead of panicking mid request
fn trickle_ready(eng: &Engine) -> bool {
    eng.residency().lb.is_none() && eng.residency().stride > eng.residency().n
}

/// - #39 B3a: THE generation loop of this server, the only one. `stream:true` runs it with
///   `SseSink`, `stream:false` with `CollectSink`; prefix cache, sampler, stop rules,
///   tool-call parser, snapshot, counters and the three `[chat]` stderr lines are shared.
/// - prefill, then decode, one sink call per emitted delta
/// - #31 A9: the prefix cache decides FIRST (spec 7.4); a warm request rolls back to `P`
///   and prefills `ids[P..]`, a cold one runs `reset_to_zero` and prefills everything
/// - ONE snapshot per request, unconditional (M2b, #36): after the prompt
/// - a sink refusal breaks the loop; the next request rolls back or resets before any prefill
/// - #54: one `ChatSink::still_there` probe between two decode steps, which is how the
///   `stream:false` path ends a generation whose client is gone (the stream path answers
///   `true` there and learns the same thing from its next failed flush)
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
    let plan = srv.cache.decide(srv.eng.history(), &prompt);
    let held = srv.eng.history().len();
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
    tracing::info!(target: "cache",
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
    let snap1_ms = unsafe { srv.cache.snapshot(srv.eng, true) };
    // a disabled cache copies nothing, so it reports nothing either
    if cache_on {
        tracing::info!(target: "cache",
            "[cache] snapshot point 1 (after prompt) at pos {}, DtoH {snap1_ms:.3} ms",
            srv.eng.pos()
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
            srv.eng.unpark_sampler(&mut srv.parked_sampler);
            // unsafe: device uploads and one eager sampler launch, as parity does
            next = unsafe { srv.eng.arm_sampler(s) };
            // #68: every value says where it came from. `temperature` is the only one that is
            // always the request's own - without it this branch is not taken at all.
            let sent = req.sampling_sent;
            tracing::info!(target: "chat",
                "[chat] sampling on the device: temperature {} (request) top_p {} ({}) top_k {} ({}) presence_penalty {} ({}) seed {} ({})",
                s.temperature,
                s.top_p, SamplingSent::tag(sent.top_p),
                s.top_k, SamplingSent::tag(sent.top_k),
                s.presence_penalty, SamplingSent::tag(sent.presence_penalty),
                s.seed, SamplingSent::tag(sent.seed)
            );
            // the penalty set of THIS request: `arm_sampler` clears the device mask and reloads
            // `Rng::new(seed)`, so no token of the prompt and no token of an earlier turn is in
            // it, whatever the prefix cache reused (7.11.17)
            tracing::info!(target: "chat",
                "[chat] presence penalty set: cleared for this request, generated tokens only (#68)"
            );
        }
        None => {
            // greedy is the A4 path: `decode_step` samples whenever `dev_sampler` is Some
            // (gen.rs:2905-2909) and `reset_to_zero` does not clear it, so it is taken out here
            srv.eng.park_sampler(&mut srv.parked_sampler);
            tracing::info!(target: "chat", "[chat] greedy (temperature absent or <= 0)");
        }
    }
    if req.min_p != 0.0 {
        tracing::info!(target: "chat",
            "[chat] min_p {} accepted and ignored (device sampler has no min_p; #28)",
            req.min_p
        );
    }
    // #74: one line per request that says whether it thought, at which level, and whether the
    // value came from the body - the same provenance shape the two sampling lines above carry
    tracing::info!(target: "chat", "{}", thinking_line(req));

    // #29 A7: the tool-call parser of THIS request. `tool_open` is the id the decode loop
    // arms it with; without that id the literal text `<tool_call>` stays content.
    let mut ts = ToolStream::new(req.tools.as_ref());
    let tool_open = tk.token_id(TOOL_OPEN);
    // #67: the reasoning filter of THIS request. It sees `Emit::Content` only, it holds a
    // partial `</think` back across deltas, and it never touches an id: the loop below
    // samples and pushes the same ids it pushed before this filter existed.
    let mut think = ThinkFilter::for_request(req.enable_thinking);
    let mut counts = Chunks::default();

    // the id/created/model triple of this response, once
    let cx = ChunkCtx::new(&id, created, &model);
    let mut aborted = !sink.open(&cx);

    // #27 A5: the decode window opens at the FIRST `decode_step` and closes when the last one
    // returns, so the detokenize and the sink call of token 1 are not counted as decode
    let mut t_dec: Option<Instant> = None;
    let mut out: Vec<u32> = Vec::with_capacity(req.max_tokens);
    // bytes of the accumulated decode that already left as content
    let mut emitted = 0usize;
    let mut finish = "length";
    // TASK K: what each call's `arguments` fragments add up to, per index, so the
    // contract can be checked where it is produced (see the loop after the flush)
    let mut args_acc: Vec<String> = Vec::new();
    let mut decode_ms = 0.0f64;
    // #37: the stream trickle, one tick per `decode_step`, the mirror of `decode.rs:224-231`.
    // `cfg.adapt` is what `apply_adapt_policy` (geo.rs:167-176) gave this process: with
    // `CROW_ADAPT_STREAM` unset and chunk 2048 that is stream / 7 spare / every 8 / max 7.
    // `decode.rs` ticks for `i in 1..gen`, that is before every `decode_step` EXCEPT the
    // first; loop index `i` here names the same token, so the guard is the same `i > 0`.
    let (adapt_stream, adapt_every, adapt_max) = srv.eng.cfg.adapt.knobs();
    let tick_trickle = adapt_stream && adapt_every > 0 && trickle_ready(srv.eng);
    let mut trickle_swaps = 0usize;
    // #81: THE THINKING BUDGET, and the shape it takes here. The mechanism llama-server
    // ships as `--reasoning-budget` (its PRs #13771/#17750) and Qwen's docs describe as
    // `thinking_budget`: when the cap of reasoning tokens is spent, the close of the think
    // block is FORCED -- the sampled continuation is replaced by the wrap-up message (when
    // the request sent one), the `</think>` token and a `\n\n`, all appended as GENERATED
    // ids, so the model's own context contains the close it never wrote and it continues
    // with the answer. What is spent of `max_tokens` by the injection belongs to the
    // budget the same way a sampled token does.
    //
    // THE TOKENS ARE REPLACED, NOT MASKED: vLLM's ThinkingPlugin masks every logit except
    // the close token; this loop achieves the same sequence by discarding the one sampled
    // id at the top of the next iteration and substituting its own. One id of sampling is
    // wasted per close, once per request, and no sampler internals change hands.
    //
    // OFF BY DEFAULT: no `reasoning_budget_tokens` in the body leaves this whole block
    // inert and the generated ids byte-identical to every release before #81 -- the same
    // rule #74 gave `reasoning_effort`.
    //
    // `Some(0)` closes before the first reasoning token, armed here rather than in the
    // loop, so a budget of zero is "do not think" rather than "think one token".
    let think_budget = if req.enable_thinking {
        req.reasoning_budget
    } else {
        None // a non-thinking request has no block to close
    };
    let mut think_tokens = 0usize;
    let mut inject: VecDeque<usize> = VecDeque::new();
    let mut injection_built = false;
    let build_injection = || -> VecDeque<usize> {
        // the message as raw BPE, then the single close id, then a paragraph break: the
        // message lands INSIDE the block as its last reasoning text, exactly where Crow's
        // #176 measurement wants it, and the `\n\n` after the close is what `Lead` trims
        // so the answer starts at its first real character.
        let mut ids: Vec<usize> = req
            .reasoning_budget_message
            .as_deref()
            .and_then(|m| tk.encode_raw(m).ok())
            .unwrap_or_default()
            .into_iter()
            .map(|id| id as usize)
            .collect();
        if let Some(close) = tk.token_id(THINK_CLOSE) {
            ids.push(close as usize);
        }
        if let Ok(tail) = tk.encode_raw("\n\n") {
            ids.extend(tail.into_iter().map(|id| id as usize));
        }
        VecDeque::from(ids)
    };
    if think_budget == Some(0) {
        inject = build_injection();
        injection_built = true;
        tracing::info!(target: "chat",
            "[chat] reasoning budget 0 (request): closing the think block before it opens");
    }
    if !aborted {
        for i in 0..req.max_tokens {
            // #81: a pending injection REPLACES the token the sampler produced - this one
            // line is the whole force. It sits above the EOS check because an injected
            // close is never EOS, and above `out.push`, so the replaced id simply never
            // exists anywhere.
            if !inject.is_empty() {
                next = inject.pop_front().expect("checked non-empty");
            }
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
                    tracing::warn!(target: "chat", "[chat] detokenize failed: {e}");
                    String::new()
                }
            };
            if let Some(delta) = next_delta(&full, emitted) {
                emitted = full.len();
                let pieces = ts.feed(delta);
                accumulate_args(&pieces, &mut args_acc);
                if !send_emits(sink, &cx, &pieces, &mut think, &mut counts) {
                    aborted = true;
                    break;
                }
            }
            // #81: COUNT AFTER THE FILTER SAW THE TOKEN. A token that closed the block
            // left `Inside`, is the model's own close and counts for nothing; everything
            // the block still holds is reasoning, whatever its text. Arming happens on
            // the token that SPENDS the budget, the injection starts with the next one.
            if !injection_built {
                if let Some(cap) = think_budget {
                    if think.is_inside() {
                        think_tokens += 1;
                        if think_tokens >= cap {
                            inject = build_injection();
                            injection_built = true;
                            tracing::info!(target: "chat",
                                "[chat] reasoning budget {} spent after {} thinking tokens \
                                 (request): closing the think block",
                                cap, think_tokens);
                        }
                    }
                }
            }
            // the budget is spent: no decode_step whose token nobody reads
            if i + 1 == req.max_tokens {
                break;
            }
            // #54: BETWEEN two steps, ask whether the client is still there. The SSE sink
            // answers `true` here and learns it from its next failed flush instead; the
            // document sink probes the socket, because it writes nothing until the end and a
            // gone client would otherwise hold this slot for the whole `max_tokens` budget.
            // One `poll`, 0.10 us, per step (`PROBE_EVERY`, 7.11.18).
            // #82: a shutdown signal ends the generation THROUGH the normal
            // abort path, so the slot, the sink and the loop bookkeeping all
            // close the way a gone client closes them
            if SHUTTING_DOWN.load(std::sync::atomic::Ordering::SeqCst)
                || !sink.still_there(i) {
                aborted = true;
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
        tracing::info!(target: "chat",
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
        accumulate_args(&pieces, &mut args_acc);
        if !send_emits(sink, &cx, &pieces, &mut think, &mut counts) {
            aborted = true;
        }
    }
    // #67: what the filter is still holding back. A prefix of `</think` that never completed
    // is TEXT, and it leaves here, so no byte of the answer is lost to the filter.
    if !aborted {
        let tail = think.flush();
        if !send_split(sink, &cx, &tail, &mut counts) {
            aborted = true;
        }
    }
    if think.stripped() > 0 {
        tracing::info!(target: "chat",
            "[chat] reasoning filter: {} <think>/</think> tag(s) stripped from the content \
             (#67); the generated ids are untouched{}",
            think.stripped(),
            // #74: a thinking request OWES one of them - the close of the block its prompt
            // opened. A non-thinking request owes none, and every one is the stray of #67.
            if req.enable_thinking {
                ", and this request was thinking, so one of them is its own </think>"
            } else {
                ""
            }
        );
    }
    // TASK K: the invariant, checked where it is PRODUCED. `toolcall` guarantees that the
    // concatenation of one call's `Emit::Args` is a parseable JSON object for every input
    // (`toolcall::tests`, every markup shape at every piece size), and this says so out loud
    // for the request that just ran. A stream cannot be repaired - the fragments are already
    // on the wire - so a violation is a named, loud line with `serde_json`'s own message and
    // the byte window, instead of a `{"_raw": ...}` two turns later in someone else's history.
    for (i, a) in args_acc.iter().enumerate() {
        if let (_, Some(note)) = args_object_or_raw(a) {
            tracing::error!(target: "chat", "[chat] BUG: the arguments of tool call {i} are not a JSON object - {note}");
        }
    }
    // #29 A7: a closed call answers `tool_calls`; a malformed one keeps `stop` / `length`.
    // `ToolStream::finish` closes a call whose `</function>` arrived, so EOS in the tail is
    // a complete call, not a malformed one (#29 review).
    if malformed {
        tracing::warn!(target: "chat",
            "[chat] MALFORMED tool call: no </function> or no name before the end; \
             the raw markup went out as content, finish stays {finish}"
        );
    } else if ts.closed() > 0 {
        finish = "tool_calls";
    }
    if ts.dropped() > 0 {
        tracing::info!(target: "chat",
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
    let (ple_rows_total, ple_miss_total) = (srv.eng.ple().req, srv.eng.ple().miss);
    let counters_ms = t_ctr.elapsed().as_secs_f64() * 1e3;

    let gen = out.len();
    // #68: the cross-turn repeat counter, after the last decode step and off the numeric
    // path. `finish` is final here (`tool_calls` is decided above), and `out` holds exactly
    // the ids this request generated - EOS is never pushed, so a one-id answer that stopped
    // by itself is the live single-token shape.
    let rep = srv.repeats.observe(&out, single_token_answer(gen, finish));
    if let Some(w) = loop_warning(&rep) {
        tracing::warn!(target: "chat", "{w}");
    }
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
            &cx,
            &FinishArgs {
                finish,
                t: &timing,
                include_usage: req.include_usage,
                timings_per_token: req.timings_per_token,
            },
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
    tracing::info!(target: "chat",
        "[chat] prompt {} tok ({cached_n} cached, {prefilled} prefilled), generated {gen} tok, prefill {prefill_ms:.1} ms ({:.1} tok/s), reset {reset_ms:.1} ms, decode {decode_ms:.1} ms, {:.1} tok/s, finish {finish}, content chunks {}, reasoning chunks {}, thinking {}, tool chunks {}, think tags stripped {}, tool calls {}, usage {}, timings {}, crow_trickle_swaps {trickle_swaps}{}{}",
        ids.len(),
        per_second(prefilled, prefill_ms),
        (gen.saturating_sub(1)) as f64 * 1000.0 / decode_ms.max(1e-9),
        counts.content,
        counts.reasoning,
        thinking_tag(req),
        counts.tool,
        think.stripped(),
        ts.closed(),
        log_usage,
        log_timings,
        if aborted { ", client gone" } else { "" },
        repeat_note(&rep)
    );
    // #30 A8: the same numbers the `timings` block carries, cumulative since process start
    tracing::info!(target: "chat",
        "[chat] counters (cumulative, never reset): expert selections {selections_total}, \
         expert cold {cold_total}, ple rows {ple_rows_total}, ple misses {ple_miss_total}, \
         layers {LAYERS}, counter read {counters_ms:.3} ms"
    );
    // #13: ONE structured routing line per request, target `routing`, at INFO.
    // Same source as the `timings` block and the two lines above - the counters
    // are cumulative, so the request-local numbers are the difference against
    // the block the previous request left in `srv.prev_counters`.
    let prev = std::mem::replace(&mut srv.prev_counters, blocks.clone());
    let (mut d_sel, mut d_cold, mut layers_cold) = (0u64, 0u64, 0usize);
    for (l, c) in blocks.iter().enumerate() {
        let b = prev.get(l).copied().unwrap_or([0, 0]);
        d_sel += c[0].saturating_sub(b[0]);
        let dc = c[1].saturating_sub(b[1]);
        d_cold += dc;
        if dc > 0 {
            layers_cold += 1;
        }
    }
    let d_rows = ple_rows_total.saturating_sub(srv.prev_ple.0);
    let d_fills = ple_miss_total.saturating_sub(srv.prev_ple.1);
    srv.prev_ple = (ple_rows_total, ple_miss_total);
    let expert_bytes = srv.eng.residency().gu_bytes + srv.eng.residency().dn_bytes;
    crow_nest_engine::log::routing(&crow_nest_engine::log::Routing {
        seq: srv.seq,
        route: if req.stream { "chat_stream" } else { "chat_document" }.to_string(),
        finish: finish.to_string(),
        prompt_n: ids.len(),
        cached_n,
        predicted_n: gen,
        prompt_ms: prefill_ms,
        predicted_ms: decode_ms,
        tok_s: (gen.saturating_sub(1)) as f64 * 1000.0 / decode_ms.max(1e-9),
        selections: d_sel,
        cold: d_cold,
        layers_cold,
        bytes_streamed: d_cold * expert_bytes,
        ple_rows: d_rows,
        ple_fills: d_fills,
        trickle_swaps,
        counters_ms,
        repeat_of: rep.repeat_of,
        repeat_run: rep.repeat_run,
        single_token: rep.single_token,
    });
    // #13: the full id list of every answer is DEBUG now (`CROW_LOG=info,chat=debug`).
    // At INFO it made a redirected stderr and the log file grow with every answer.
    tracing::debug!(target: "chat", "[chat] ids {out:?}");
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
    let (map, grids, _delta, n_visual) = match srv.eng.vision_plan() {
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
        crow_nest_engine::vit::f32_file(&format!("{dir}/gpu-logits.f32"), &flat);
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

/// #68 (2026-09-18): how many answer hashes the cross-turn counter keeps. Eight is what a
/// human reads a `[chat]` line against; the RUN is counted separately and is not capped by it.
const REPEAT_RING: usize = 8;
/// #68: the WARN threshold, a constant and deliberately not an env knob. Three identical
/// answers in a row is the live shape (48 of them at the end of the goal-mode session), two
/// is a client that asked the same thing twice.
const LOOP_WARN_AT: usize = 3;

/// #68: what the cross-turn counter says about ONE completed answer. Pure data.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct RepeatStats {
    /// how many answers back the most recent identical answer is; 0 = none in the ring
    repeat_of: usize,
    /// how many identical answers in a row ended with this one; 1 = none
    repeat_run: usize,
    /// exactly one generated id, and the model ended the answer itself
    single_token: bool,
    /// how many single-token answers in a row ended with this one; 0 = this one is not one
    single_run: usize,
}

/// #68: the last [`REPEAT_RING`] answers of THIS PROCESS, as hashes of their generated ids.
///
/// Why the ids and not the text: they are what the model produced, they are what the
/// `[chat] ids` line prints, and two answers with the same ids are the same answer whatever
/// the detokenizer, the tool-call parser or the reasoning filter make of them downstream.
///
/// Why per process: `serve` holds ONE conversation (`PrefixCache`, #31 A9) and the wire
/// carries no session id, so there is nothing else to key on - and a cold prefill is not a
/// new conversation, since an identical re-send is cold by construction (spec 7.4).
///
/// It is read by the log and by nothing else. No sampler, no finish reason, no status code
/// and no wire field depends on any of it.
#[derive(Default)]
struct RepeatRing {
    /// the hashes of the last [`REPEAT_RING`] answers, oldest first
    seen: std::collections::VecDeque<u64>,
    /// the previous answer's hash, and the run of identical answers that ended with it
    last: Option<u64>,
    run: usize,
    /// single-token answers in a row
    single_run: usize,
}

/// #68: FNV-1a 64 over the generated ids. Not a cryptographic hash and it does not need to
/// be: a collision costs one wrong `repeat run` on a log line and nothing else.
fn answer_hash(ids: &[u32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for id in ids {
        for b in id.to_le_bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

impl RepeatRing {
    /// One completed answer in, its counters out. Pure apart from the ring it advances.
    fn observe(&mut self, ids: &[u32], single_token: bool) -> RepeatStats {
        let h = answer_hash(ids);
        // the newest entry is the LAST one, so the distance counts from the back
        let repeat_of = self.seen.iter().rev().position(|&p| p == h).map_or(0, |i| i + 1);
        let repeat_run = if self.last == Some(h) { self.run + 1 } else { 1 };
        self.single_run = if single_token { self.single_run + 1 } else { 0 };
        self.last = Some(h);
        self.run = repeat_run;
        if self.seen.len() == REPEAT_RING {
            self.seen.pop_front();
        }
        self.seen.push_back(h);
        RepeatStats { repeat_of, repeat_run, single_token, single_run: self.single_run }
    }
}

/// #68: true when the answer is exactly one generated id AND the model ended it itself.
/// A one-token answer that ran into the client's own `max_tokens` budget is `finish length`
/// and says nothing about the model, so it is not the live shape and is not counted as one.
fn single_token_answer(gen: usize, finish: &str) -> bool {
    gen == 1 && finish == "stop"
}

/// #68: what the `[chat]` summary line appends for this answer. EMPTY for a healthy one, so
/// a healthy session's line is byte-identical to the line every tool of this repo greps.
fn repeat_note(rep: &RepeatStats) -> String {
    let mut s = String::new();
    if rep.repeat_run > 1 {
        s.push_str(&format!(", repeat run {}", rep.repeat_run));
    }
    if rep.single_token {
        s.push_str(", single-token answer");
    }
    s
}

/// #68: the ONE WARN a request may owe, or `None`. Pure, so the threshold is a test and not
/// a guess. Never two lines: a run of identical single-token answers is already reported as
/// the run of identical answers that it is.
fn loop_warning(rep: &RepeatStats) -> Option<String> {
    let what = if rep.repeat_run >= LOOP_WARN_AT {
        format!("{} identical answers in a row", rep.repeat_run)
    } else if rep.single_run >= LOOP_WARN_AT {
        format!("{} single-token answers in a row", rep.single_run)
    } else {
        return None;
    };
    Some(format!(
        "[chat] the client is looping: {what} (#68) - an observation, nothing about this \
         generation was changed: no brake, no refusal, the sampler is untouched"
    ))
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
    /// TASK K: true once the SSE head of the request in flight left the socket.
    /// A CUDA allocation failure after that cannot be answered with an HTTP
    /// status any more, so `guarded` sends an SSE error frame instead.
    stream_head_sent: bool,
    /// #13: the device counter block (`[48][2]` selections/cold) and the two PLE
    /// counters as the PREVIOUS request left them. Every counter in this engine is
    /// cumulative and never reset (the Crow #54 rule), so the routing line of one
    /// request is the difference of two blocks - held here, never on the device.
    prev_counters: Vec<[u64; 2]>,
    prev_ple: (u64, u64),
    /// #68: the cross-turn repeat counter's ring, per process (see the module doc). The one
    /// piece of state this server keeps about what the MODEL said, and only the log reads it.
    repeats: RepeatRing,
}

/// - TASK K: one request may not take the server down with it
/// - a CUDA allocation that fails anywhere inside a request (the image path, the
///   mrope tables, a state buffer) raises `cuda::AllocFailed` instead of the bare
///   panic of record, because the `RequestScope` below is armed; this catches
///   exactly that payload, frees nothing itself (every allocation site frees what
///   it took before it raises, and every RAII buffer drops in the unwind), puts
///   the engine back to its zero state and answers 503 with a body that NAMES the
///   allocation and its byte count
/// - anything else that panics is re-raised unchanged: a bug is still a crash,
///   and only the out-of-VRAM case is a served error
fn guarded<F>(stream: &mut TcpStream, srv: &mut Srv, f: F) -> &'static str
where
    F: FnOnce(&mut TcpStream, &mut Srv) -> &'static str,
{
    srv.stream_head_sent = false;
    let caught = {
        let _scope = crow_nest_engine::cuda::RequestScope::new();
        std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| f(stream, srv)))
    };
    let payload = match caught {
        Ok(status) => return status,
        Err(p) => p,
    };
    let failed = match payload.downcast::<crow_nest_engine::cuda::AllocFailed>() {
        Ok(af) => *af,
        Err(p) => std::panic::resume_unwind(p),
    };
    tracing::info!(target: "serve",
        "[serve] the request was dropped: {} - the engine stays up, the next request is served",
        failed.message()
    );
    // back to a state the NEXT request can prefill from: no armed vision plan, no
    // held conversation, no captured decode graph.
    srv.eng.end_vision();
    srv.cache.invalidate();
    unsafe { srv.eng.reset_to_zero() };
    let body = error_json(&format!(
        "{}. The request was dropped and the engine is up; retry with fewer or smaller images, \
         a shorter prompt, or restart with a larger CROW_VIT_RESERVE_MB",
        failed.message()
    ));
    if srv.stream_head_sent {
        // the 200 head is already on the wire: the only thing the client can still
        // read is a frame, so the error rides one and the stream ends properly
        let _ = sse_send(stream, &sse_frame(&serde_json::json!({ "error": body["error"] })));
        let _ = sse_send(stream, SSE_DONE);
        "503 (as an SSE error frame, the head was already sent)"
    } else {
        respond_json(stream, "503 Service Unavailable", &body)
    }
}

/// - `POST /slots/0?action=save|restore` (#32 A10)
/// - every refusal is a 4xx with a JSON error body and leaves the engine untouched
/// - the return value is the status and the document `serve_one` writes
fn slot_route(srv: &mut Srv, target: &str, body: &[u8]) -> (&'static str, serde_json::Value) {
    let refuse = |msg: String| {
        tracing::warn!(target: "slot", "[slot] refused: {msg}");
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
                tracing::warn!(target: "slot", "[slot] refused: {m}");
                return ("409 Conflict", error_json(&m));
            }
            // unsafe: device to host copies only; the engine state is not written
            match unsafe { slot::save(srv.eng, &srv.cache, srv.model_path, &path) } {
                Ok(s) => {
                    tracing::info!(target: "slot",
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
                    tracing::info!(target: "slot",
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
        tracing::warn!(target: "serve", "[serve] no read timeout on this connection, closing: {e}");
        return;
    }
    if let Err(e) = stream.set_write_timeout(Some(t)) {
        tracing::warn!(target: "serve", "[serve] no write timeout on this connection, closing: {e}");
        return;
    }

    let head = match read_head(stream) {
        Ok(h) => h,
        Err(e) => {
            tracing::warn!(target: "serve", "[serve] read failed or timed out after {IO_TIMEOUT_SECS}s, closing: {e}");
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
                    let status = guarded(stream, srv, |stream, srv| chat_route(stream, srv, &body));
                    tracing::info!(target: "serve", "[serve] {label} -> {status}");
                    let _ = stream.shutdown(Shutdown::Write);
                    return;
                }
                Route::Health => (label, "200 OK", serde_json::json!({ "status": "ok" })),
                Route::Props => (
                    label,
                    "200 OK",
                    props_json(srv.model_path, srv.n_ctx, srv.prompt_chunk, srv.eng.has_vision()),
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
    tracing::info!(target: "serve", "[serve] {label} -> {status}");
    if let Err(e) = respond(stream, status, &text) {
        tracing::warn!(target: "serve", "[serve] response write failed: {e}");
    }
    // half close, so the client reads EOF instead of a reset
    let _ = stream.shutdown(Shutdown::Write);
}

// #82: THE GRACEFUL SHUTDOWN STATE. SIGTERM's default action terminates without
// unwinding, so no `Drop` runs -- and an engine that dies with its pinned cold
// tier allocated leaves ~40 GiB held by `nvidia_uvm` until the next reboot
// (three measurements on this machine, see cuda::ctx_hard_reset). The watcher
// thread below catches SIGINT/SIGTERM/SIGHUP/SIGQUIT INSTEAD of the default
// action, asks the accept loop to end, and lets `main` return so every `Drop`
// and the context reset run. The signals are blocked BEFORE any thread exists
// (the log writer included), because one unblocked thread is enough for the
// kernel to kill the process outright.
//
// SIGHUP/SIGQUIT joined on 2026-09-20, fourth leak of the series: the day's
// fix-run serve (built WITH this shutdown) was torn down with its systemd
// scope at 11:58 - no SIGTERM/SIGINT trace, and the tier leaked. A scope or
// terminal going away hangs up the foreground process group, so SIGHUP is the
// realistic end of a serve someone started in a terminal; SIGQUIT follows the
// same rule. SIGKILL stays out on purpose: it is uncatchable, and "kill -9 the
// engine" remains the one way to leak (documented, not fixed - a watchdog that
// TERMs first is the answer there, not this file).
static SHUTTING_DOWN: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
static LISTENER_FD: std::sync::atomic::AtomicI32 = std::sync::atomic::AtomicI32::new(-1);

fn install_shutdown_watch() {
    unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGHUP);
        libc::sigaddset(&mut set, libc::SIGQUIT);
        libc::sigprocmask(libc::SIG_BLOCK, &set, std::ptr::null_mut());
    }
    std::thread::spawn(|| unsafe {
        let mut set: libc::sigset_t = std::mem::zeroed();
        libc::sigemptyset(&mut set);
        libc::sigaddset(&mut set, libc::SIGINT);
        libc::sigaddset(&mut set, libc::SIGTERM);
        libc::sigaddset(&mut set, libc::SIGHUP);
        libc::sigaddset(&mut set, libc::SIGQUIT);
        let mut sig: libc::c_int = 0;
        while libc::sigwait(&set, &mut sig) == 0 {
            SHUTTING_DOWN.store(true, std::sync::atomic::Ordering::SeqCst);
            // eprintln, not tracing: the subscriber may not exist yet when an
            // early signal lands, and this line is the one an operator waits for
            eprintln!("[serve] signal {sig}: shutting down - the pinned tier is being freed before exit (#82)");
            let fd = LISTENER_FD.load(std::sync::atomic::Ordering::SeqCst);
            if fd >= 0 {
                libc::shutdown(fd, libc::SHUT_RDWR);
            }
            // no exit here on purpose: main returns, so Engine, Residency and
            // the Ctx all Drop, and the caller of this process gets its RAM back
        }
    });
}

fn main() {
    // #13: the subscriber, before the first line this process says. The guard keeps
    // the two writer threads alive for the whole process and drains them when `main`
    // returns; every `std::process::exit` below calls `log::shutdown()` first, because
    // `exit` runs no destructor and a lost `[serve] cannot bind ...` is the one line
    // an operator needs.
    // #82: FIRST, before the log spawns its writer threads - see the block above
    install_shutdown_watch();
    let _log = crow_nest_engine::log::init();
    let args: Vec<String> = std::env::args().collect();
    // A3: the tokenize arm returns HERE, before the CUDA context and before Engine::load
    // takes engine/.engine.lock; it never starts a Python process
    if args.get(1).map(|s| s.as_str()) == Some("tokenize") {
        let rc = tokenize_main(&args[2..]);
        crow_nest_engine::log::shutdown();
        std::process::exit(rc);
    }
    // TASK K: an out-of-VRAM inside a request raises `cuda::AllocFailed`, which `guarded`
    // catches and answers with a 503. Its payload is not a string, so the DEFAULT hook
    // prints `panicked at ...: Box<dyn Any>` on the way past - a line that says nothing and
    // reads like a crash in a log where the server is fine. `AllocFailed::raise` has already
    // printed the `[alloc]` line with the name, the byte count and the free VRAM, so this
    // hook drops that one payload and leaves every other panic exactly as it was.
    let previous_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        if info.payload().downcast_ref::<crow_nest_engine::cuda::AllocFailed>().is_some() {
            return;
        }
        previous_hook(info);
    }));
    let cli = match parse_args(&args) {
        Ok(a) => a,
        Err(e) => {
            tracing::error!(target: "serve", "[serve] {e}");
            crow_nest_engine::log::shutdown();
            std::process::exit(2);
        }
    };
    // #32 review: a typo'd --slot-save-path used to be discovered at the FIRST save, after a
    // whole conversation had been prefilled. It costs one stat call to find it here.
    if let Some(d) = &cli.slot_save_path {
        if let Err(e) = check_slot_save_path(d) {
            tracing::error!(target: "serve", "[serve] {e}");
            crow_nest_engine::log::shutdown();
            std::process::exit(2);
        }
    }

    // #25 A3: warm up the tokenizer BEFORE the CUDA context and before Engine::load.
    // A missing or broken tokenizer must fail in a second, not after the engine is pinned.
    let (tok_path, tok_cfg) = crow_nest_engine::tokenizer::default_paths();
    match crow_nest_engine::tokenizer::global() {
        Ok(tk) => {
            let (tp, cp) = tk.paths();
            tracing::info!(target: "serve", "[serve] tokenizer {tp}");
            tracing::info!(target: "serve", "[serve] chat template {cp}");
        }
        Err(e) => {
            tracing::error!(target: "serve", "[serve] {e}");
            tracing::info!(target: "serve", "[serve] tokenizer {tok_path}");
            tracing::info!(target: "serve", "[serve] chat template {tok_cfg}");
            crow_nest_engine::log::shutdown();
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
        tracing::info!(target: "serve", "[serve] {key}={}", std::env::var(key).unwrap_or_default());
    }

    // unsafe: creates the CUDA context; it must outlive every device allocation
    let (mut cnq, _ctx, mut cfg, cnq_path, sidecar) = unsafe {
        boot::open_model(DEFAULT_CNQ.into(), DEFAULT_HOTSETS.into())
    };
    // M1: chunk pinned for the process, no per prompt policy
    cfg.prompt_chunk = SERVE_CHUNK;
    apply_adapt_policy(&mut cfg);

    // unsafe: pins device and host memory; takes engine/.engine.lock, a second serve dies here
    let eng = unsafe {
        Engine::load(&mut cnq, cfg, None, &sidecar, false, &mut |m| tracing::info!(target: "load", "[load] {m}"))
    };
    let n_ctx = eng.n_ctx();
    let prompt_chunk = eng.cfg.prompt_chunk;
    // #82: a signal that arrived DURING the load has no listener to wake; this
    // check is its exit, through the same drops the normal end takes
    if SHUTTING_DOWN.load(std::sync::atomic::Ordering::SeqCst) {
        crow_nest_engine::log::shutdown();
        unsafe { crow_nest_engine::cuda::ctx_hard_reset(); }
        return;
    }

    tracing::info!(target: "serve", "[serve] container {cnq_path}");
    tracing::info!(target: "serve", "[serve] hotsets {sidecar}");
    tracing::info!(target: "serve", "[serve] n_ctx {n_ctx}");
    tracing::info!(target: "serve", "[serve] prompt_chunk {prompt_chunk}");

    // #37: the "[policy] ..." line above is what apply_adapt_policy CHOSE; this line is what
    // serve DOES with it. The stream trickle is ticked once per decode_step, the mirror of
    // decode.rs:224-231. adapt_tick, the post-prefill re-cut of CROW_ADAPT=1, stays uncalled.
    let ad = eng.cfg.adapt;
    if ad.stream && ad.every > 0 && trickle_ready(&eng) {
        tracing::info!(target: "serve",
            "[serve] #37 stream trickle ticked once per decode_step: every {}, max {}/layer, {} spare hot slot(s)",
            ad.every, ad.max, ad.spare
        );
    } else {
        tracing::info!(target: "serve",
            "[serve] #37 stream trickle NOT ticked: stream {}, every {}, spare slots {}, exact NVFP4 tier {}",
            ad.stream,
            ad.every,
            eng.residency().stride.saturating_sub(eng.residency().n),
            eng.residency().lb.is_none()
        );
    }
    tracing::info!(target: "serve", "[serve] adapt_tick (the CROW_ADAPT=1 hot-set re-cut) is never called by serve");

    // #32 A10: without --slot-save-path, POST /slots/0 refuses save and restore
    match &cli.slot_save_path {
        Some(d) => tracing::info!(target: "serve", "[serve] slot save path {d}"),
        None => tracing::info!(target: "serve", "[serve] no --slot-save-path, so POST /slots/0 refuses save and restore"),
    }

    let addr = format!("127.0.0.1:{}", cli.port);
    let listener = match TcpListener::bind(&addr) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!(target: "serve", "[serve] cannot bind {addr}: {e}");
            crow_nest_engine::log::shutdown();
            std::process::exit(3);
        }
    };
    // #82: the watcher shuts this socket down to break the accept below
    use std::os::fd::AsRawFd;
    LISTENER_FD.store(listener.as_raw_fd(), std::sync::atomic::Ordering::SeqCst);
    tracing::info!(target: "serve", "[serve] listening on http://{addr} (blocking, one request at a time)");

    let mut eng = eng;
    // #31 A9: the snapshot slot is allocated here, once, from the loaded shape (spec 7.7).
    // #36 M2b: SLOTS is 1, and the line below reads it instead of naming a count of its own.
    let cache = PrefixCache::new(&eng);
    tracing::info!(target: "serve",
        "[serve] prefix cache {}, {} B per snapshot, {} snapshot(s) in HOST RAM (#72: never VRAM, see cache.rs), QSA ring rows {}",
        if cache.enabled() { "on" } else { "off (CROW_PREFIX_CACHE=0)" },
        cache.shape().snapshot_bytes(),
        SLOTS,
        eng.qsa_ring_rows()
    );
    // #13: the boot report as ONE structured line, target `boot`, at INFO, in
    // addition to the eight human lines above - the operating point every later
    // number references (context, N residency, KV dtype, kernel path, cold-path
    // policy per layer). It is emitted LAST of the boot lines, because
    // `PrefixCache::new` is the last fact it carries.
    let cold_policy = if ad.stream && ad.every > 0 && trickle_ready(&eng) {
        format!(
            "zero-copy read from the pinned tier; stream trickle every {} decode token(s), max {}/layer, {} spare hot slot(s)",
            ad.every, ad.max, ad.spare
        )
    } else {
        "zero-copy read from the pinned tier; no trickle, no re-cut (serve never calls adapt_tick)".to_string()
    };
    crow_nest_engine::log::boot(&eng.boot_point(
        "serve",
        &cnq_path,
        &sidecar,
        &cold_policy,
        cache.enabled(),
    ));
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
        stream_head_sent: false,
        prev_counters: Vec::new(),
        prev_ple: (0, 0),
        repeats: RepeatRing::default(),
    };
    for conn in listener.incoming() {
        match conn {
            Ok(mut s) => serve_one(&mut s, &mut srv),
            Err(e) => {
                if SHUTTING_DOWN.load(std::sync::atomic::Ordering::SeqCst) {
                    break; // #82: the watcher shut the listener - end through the drops
                }
                tracing::warn!(target: "serve", "[serve] accept failed: {e}");
            }
        }
    }
    // #28: a parked device sampler goes back into the engine, so `Engine::drop` frees its
    // buffers (the accept loop above only ends on a listener error)
    srv.eng.unpark_sampler(&mut srv.parked_sampler);
    drop(srv);
    drop(eng);
    // #82: last CUDA act of this process - see cuda::ctx_hard_reset
    unsafe { crow_nest_engine::cuda::ctx_hard_reset(); }
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
        let doc = props_json(DEFAULT_CNQ, 200_000, 2048, false);
        // server_model_path (crow_core.py:1408)
        assert_eq!(doc["model_path"], DEFAULT_CNQ);
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
        let doc = props_json(DEFAULT_CNQ, 200_000, 2048, true);
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
        assert_eq!(model_name(DEFAULT_CNQ), "Qwen3.8-Flash-Next-CNQ4.5-M");
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

    // ------------------------------------------------ reasoning budget (#81)

    const BUDGET_BODY: &str =
        r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":"#;

    #[test]
    fn the_reasoning_budget_parses_in_llama_servers_integer_dialect() {
        // absent, null and negative are unrestricted - the behaviour of every release
        // before #81, and the shape a request that names no cap must keep
        for body in [
            r#"{"messages":[{"role":"user","content":"hi"}]}"#.to_string(),
            r#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":null}"#.to_string(),
            BUDGET_BODY.to_string() + "-1}",
        ] {
            let r = parse_chat(body.as_bytes()).unwrap();
            assert_eq!(r.reasoning_budget, None, "{body}");
        }
        // 0 closes before the first reasoning token, n > 0 caps at n
        assert_eq!(
            parse_chat((BUDGET_BODY.to_string() + "0}").as_bytes())
                .unwrap()
                .reasoning_budget,
            Some(0)
        );
        assert_eq!(
            parse_chat((BUDGET_BODY.to_string() + "1024}").as_bytes())
                .unwrap()
                .reasoning_budget,
            Some(1024)
        );
        // the message travels beside the cap, and only as a string
        assert_eq!(
            parse_chat(
                (BUDGET_BODY.to_string()
                    + r#"5,"reasoning_budget_message":"wrap up"}"#)
                    .as_bytes()
            )
            .unwrap()
            .reasoning_budget_message
            .as_deref(),
            Some("wrap up")
        );
        // rejections, in this parse's own words
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":"lots"}"#).is_err());
        assert!(parse_chat(br#"{"messages":[{"role":"user","content":"hi"}],"reasoning_budget_tokens":5,"reasoning_budget_message":7}"#).is_err());
    }

    #[test]
    fn is_inside_counts_only_what_the_block_still_holds() {
        // #81: the counter asks the filter AFTER the token's text was seen, so the token
        // that closes the block is the model's own close and never a counted token.
        let mut f = ThinkFilter::inside();
        assert!(f.is_inside());
        let s = f.push("reasoning text");
        assert!(!s.content.is_empty() || !s.reasoning.is_empty());
        assert!(f.is_inside(), "reasoning text keeps the block open");
        f.push("</think>");
        assert!(!f.is_inside(), "the model's own close counts for nothing");
        // and the injection's close takes the same road: message inside, close flips,
        // the paragraph break after it is trimmed by Lead
        let mut g = ThinkFilter::inside();
        let s = g.push("\n\nThat is enough analysis.</think>\n\n");
        assert!(!s.reasoning.is_empty(), "the wrap-up lands in the block");
        assert!(!g.is_inside());
        assert!(s.content.is_empty(), "Lead trims the break after the close");
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

    // #68 (2026-09-18): the `[chat]` line has to say per value whether the client named it.
    // The live goal-mode session was read as "Crow sent presence_penalty 1.5"; Crow's wire list
    // is `("temperature", "top_p", "min_p", "top_k")` (`crow_core.py:716`) and 1.5 is
    // `DEFAULT_PRESENCE`, so the request that produced the degeneration carried the model card's
    // THINKING temperature row and got the NON-thinking penalty from this file.
    #[test]
    fn the_sampling_line_says_which_values_the_request_carried() {
        // what Crow really sends: temperature, top_p, min_p - and nothing else
        let crow = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":1.0,"top_p":0.95,"min_p":0.01}"#,
        )
        .unwrap();
        assert_eq!(
            crow.sampling_sent,
            SamplingSent { top_p: true, top_k: false, presence_penalty: false, seed: false }
        );
        // the values behind the two flags that are false are this file's, not the client's
        assert_eq!(crow.presence_penalty, DEFAULT_PRESENCE);
        assert_eq!(crow.seed, DEFAULT_SEED);
        assert_eq!(crow.top_k, DEFAULT_TOP_K);

        // a body that names every field
        let full = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":0.7,"top_p":0.8,"top_k":20,"presence_penalty":1.5,"seed":3}"#,
        )
        .unwrap();
        assert_eq!(
            full.sampling_sent,
            SamplingSent { top_p: true, top_k: true, presence_penalty: true, seed: true }
        );

        // an explicit null is an absent field here too, so the tag never contradicts the value
        let nulls = parse_chat(
            br#"{"messages":[{"role":"user","content":"hi"}],
                 "temperature":1.0,"top_p":null,"top_k":null,"presence_penalty":null,"seed":null}"#,
        )
        .unwrap();
        assert_eq!(nulls.sampling_sent, SamplingSent::default());
        assert_eq!(SamplingSent::tag(true), "request");
        assert_eq!(SamplingSent::tag(false), "data sheet");
    }

    // #68 (2026-09-18): the cross-turn repeat counter. Four tests, all pure - the ring is
    // host state and the whole point of it is that nothing on the numeric path can see it.
    // The shape they pin is the live one: `docs/long-context-goalmode.md` 3.3, 48 of the 293
    // answers of robin's goal-mode session were the single id 18 (`3`) with `finish stop`.

    #[test]
    fn the_repeat_ring_counts_the_run_and_the_distance_back() {
        let mut r = RepeatRing::default();
        // a fresh ring repeats nothing
        let a1 = r.observe(&[1, 2, 3], false);
        assert_eq!((a1.repeat_of, a1.repeat_run), (0, 1));
        // the same answer again: one back, a run of two
        let a2 = r.observe(&[1, 2, 3], false);
        assert_eq!((a2.repeat_of, a2.repeat_run), (1, 2));
        let a3 = r.observe(&[1, 2, 3], false);
        assert_eq!((a3.repeat_of, a3.repeat_run), (1, 3));
        // a different answer breaks the RUN but is still a first sighting
        let b = r.observe(&[9], false);
        assert_eq!((b.repeat_of, b.repeat_run), (0, 1));
        // the old answer comes back: seen two answers back, but the RUN starts over at 1,
        // because a run is CONSECUTIVE and the run of three ended when `b` arrived
        let a4 = r.observe(&[1, 2, 3], false);
        assert_eq!((a4.repeat_of, a4.repeat_run), (2, 1));
        // the run is not capped by the ring: 48 identical answers report 48
        let mut long = RepeatRing::default();
        let mut last = RepeatStats::default();
        for _ in 0..48 {
            last = long.observe(&[18], false);
        }
        assert_eq!(last.repeat_run, 48);
        assert_eq!(last.repeat_of, 1);
    }

    #[test]
    fn the_repeat_ring_sees_the_generated_ids_and_only_the_last_eight() {
        // the ids are what is hashed: order matters, length matters, the text does not exist
        assert_ne!(answer_hash(&[1, 2]), answer_hash(&[2, 1]));
        assert_ne!(answer_hash(&[1, 2]), answer_hash(&[1, 2, 2]));
        assert_eq!(answer_hash(&[7, 8, 9]), answer_hash(&[7, 8, 9]));
        // an answer older than REPEAT_RING is out of the ring and reports no repeat
        let mut r = RepeatRing::default();
        r.observe(&[42], false);
        for i in 0..REPEAT_RING as u32 {
            r.observe(&[100 + i], false);
        }
        assert_eq!(r.observe(&[42], false).repeat_of, 0, "8 answers back is out of the ring");
        // one inside the window is still seen
        let mut r2 = RepeatRing::default();
        r2.observe(&[42], false);
        for i in 0..(REPEAT_RING as u32 - 1) {
            r2.observe(&[100 + i], false);
        }
        assert_eq!(r2.observe(&[42], false).repeat_of, REPEAT_RING);
    }

    #[test]
    fn a_single_token_answer_is_one_id_the_model_ended_itself() {
        // the live shape: one id, EOS
        assert!(single_token_answer(1, "stop"));
        // the client's own budget is not the model stopping
        assert!(!single_token_answer(1, "length"));
        // and neither is a one-id answer that opened a tool call
        assert!(!single_token_answer(1, "tool_calls"));
        assert!(!single_token_answer(0, "stop"));
        assert!(!single_token_answer(2, "stop"));
        // the streak counts consecutive ones and is cleared by any other answer
        let mut r = RepeatRing::default();
        assert_eq!(r.observe(&[18], true).single_run, 1);
        assert_eq!(r.observe(&[19], true).single_run, 2);
        let back = r.observe(&[1, 2, 3], false);
        assert_eq!(back.single_run, 0);
        assert!(!back.single_token);
        assert_eq!(r.observe(&[18], true).single_run, 1);
    }

    #[test]
    fn the_loop_warning_and_the_chat_line_suffix_fire_at_three() {
        let mk = |repeat_run, single_token, single_run| RepeatStats {
            repeat_of: usize::from(repeat_run > 1),
            repeat_run,
            single_token,
            single_run,
        };
        // a healthy answer adds NOTHING to the `[chat]` line and owes no WARN: the line every
        // tool of this repo greps is byte-identical to the line of record
        let healthy = mk(1, false, 0);
        assert_eq!(repeat_note(&healthy), "");
        assert!(loop_warning(&healthy).is_none());
        // two in a row is visible but not yet a loop
        assert_eq!(repeat_note(&mk(2, false, 0)), ", repeat run 2");
        assert!(loop_warning(&mk(2, false, 0)).is_none());
        // three is the threshold, and the line names the count
        let w = loop_warning(&mk(LOOP_WARN_AT, false, 0)).expect("a WARN at three");
        assert!(w.contains("the client is looping: 3 identical answers in a row"), "{w}");
        assert!(!w.contains('\n'), "one line");
        // three single-token answers that are NOT identical warn on their own
        let w2 = loop_warning(&mk(1, true, 3)).expect("a WARN at three single-token answers");
        assert!(w2.contains("3 single-token answers in a row"), "{w2}");
        // both at once is still ONE line, and it is the identical-answers one
        let w3 = loop_warning(&mk(4, true, 4)).expect("a WARN");
        assert!(w3.contains("4 identical answers in a row"), "{w3}");
        assert_eq!(repeat_note(&mk(4, true, 4)), ", repeat run 4, single-token answer");
        assert_eq!(repeat_note(&mk(1, true, 1)), ", single-token answer");
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
        let role = chunk_role(&ChunkCtx::new("chatcmpl-1", 1_757_000_000, "crow-nest"));
        assert_eq!(role["object"], "chat.completion.chunk");
        assert_eq!(role["id"], "chatcmpl-1");
        assert_eq!(role["created"], 1_757_000_000u64);
        assert_eq!(role["model"], "crow-nest");
        assert_eq!(role["choices"][0]["index"], 0);
        assert_eq!(role["choices"][0]["delta"]["role"], "assistant");
        assert_eq!(role["choices"][0]["finish_reason"], serde_json::Value::Null);
        assert!(role["choices"][0]["delta"].get("content").is_none());

        let c = chunk_content(&ChunkCtx::new("chatcmpl-1", 1, "crow-nest"), " ready");
        assert_eq!(c["choices"][0]["delta"]["content"], " ready");
        assert_eq!(c["choices"][0]["finish_reason"], serde_json::Value::Null);

        // the last chunk: empty delta, a finish reason, and only ONE of them exists
        let t = T0;
        let f = chunk_finish(&ChunkCtx::new("chatcmpl-1", 1, "crow-nest"), &FinishArgs { finish: "stop", t: &t, include_usage: false, timings_per_token: false });
        assert_eq!(f["choices"][0]["finish_reason"], "stop");
        assert_eq!(f["choices"][0]["delta"], serde_json::json!({}));
        assert_eq!(
            chunk_finish(&ChunkCtx::new("i", 1, "m"), &FinishArgs { finish: "length", t: &t, include_usage: false, timings_per_token: false })["choices"][0]["finish_reason"],
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
        let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: false, timings_per_token: false });
        assert!(f.get("usage").is_none());
        assert!(f.get("timings").is_none());
        // nothing else moved either: the object is exactly what A4 sent
        assert_eq!(
            f,
            chunk(&ChunkCtx::new("id", 7, "m"), serde_json::json!({}), Some("stop")),
        );
        // one flag at a time carries one object at a time
        let u = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: true, timings_per_token: false });
        assert!(u.get("usage").is_some());
        assert!(u.get("timings").is_none());
        let t = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: false, timings_per_token: true });
        assert!(t.get("usage").is_none());
        assert!(t.get("timings").is_some());
    }

    #[test]
    fn the_final_chunk_carries_the_eight_fields_crow_reads() {
        // crow_core.py:4838-4845 (usage) and :4999-5018 (timings)
        let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: true, timings_per_token: true });
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
    const T_WARM: Timing = Timing { prompt_n: 1_064, cached_n: 15_000, prompt_ms: 1_700.0, ..T0 };

    /// #31 A9: `prompt_tokens` stays the WHOLE prompt, `cached_tokens` is P, `prompt_n` the rest
    #[test]
    fn a_warm_turn_splits_the_prompt_into_cached_and_prefilled() {
        let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T_WARM, include_usage: true, timings_per_token: true });
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
            let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &t, include_usage: true, timings_per_token: true });
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
        let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &z, include_usage: true, timings_per_token: true });
        for k in ["prompt_ms", "prompt_per_second", "predicted_per_second", "predicted_per_token_ms"] {
            assert_eq!(f["timings"][k].as_f64(), Some(0.0), "{k} is {}", f["timings"][k]);
        }
        assert_eq!(f["usage"]["total_tokens"].as_u64(), Some(0));
        assert!(!f["timings"].to_string().contains("null"), "{}", f["timings"]);
    }

    /// #30 A8: the five keys, their exact names, and u64 (never a float)
    #[test]
    fn the_timings_block_carries_the_engine_counters_as_u64() {
        let g = &chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: true, timings_per_token: true })["timings"];
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
        let f = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: false, timings_per_token: false });
        assert!(!f.to_string().contains("crow_expert"), "{f}");
        // include_usage alone: `usage` carries none of them either
        let u = chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: true, timings_per_token: false });
        assert!(u.get("timings").is_none());
        assert!(!u.to_string().contains("crow_"), "{u}");
        // timings on: the value on the wire is the value the engine read, byte for byte
        let g = &chunk_finish(&ChunkCtx::new("id", 7, "m"), &FinishArgs { finish: "stop", t: &T0, include_usage: false, timings_per_token: true })["timings"];
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
        let f = sse_frame(&chunk_content(&ChunkCtx::new("id", 7, "m"), "hi"));
        assert!(f.starts_with("data: {"));
        assert!(f.ends_with("\n\n"));
        // exactly one event: one `data:` line, then the terminator
        assert_eq!(f.matches("data: ").count(), 1);
        assert_eq!(f.trim_end_matches('\n').matches('\n').count(), 0);
        // a newline inside the content is escaped by the JSON writer, never raw
        let f = sse_frame(&chunk_content(&ChunkCtx::new("id", 7, "m"), "a\nb"));
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
        let tk = tk();
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

    #[test]
    fn a_tool_chunk_carries_exactly_what_crow_reassembles() {
        let open = chunk_tool_open(&ChunkCtx::new("chatcmpl-1", 7, "crow-nest"), 0, "call_0", "read_file");
        let c = &open["choices"][0];
        assert_eq!(c["index"], 0);
        assert_eq!(c["finish_reason"], serde_json::Value::Null);
        let call = &c["delta"]["tool_calls"][0];
        assert_eq!(call["index"], 0);
        assert_eq!(call["id"], "call_0");
        assert_eq!(call["type"], "function");
        assert_eq!(call["function"]["name"], "read_file");
        assert_eq!(call["function"]["arguments"], "");

        let frag = chunk_tool_args(&ChunkCtx::new("chatcmpl-1", 7, "crow-nest"), 0, "{\"path\":\"");
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
        let (n, notes) = normalize_messages(&msgs);
        assert_eq!(n[1]["tool_calls"][0]["function"]["arguments"], serde_json::json!({"path": "a.md"}));
        assert!(notes.is_empty(), "the A7 path is not a rewrite: {notes:?}");
        // everything else is untouched, byte for byte
        assert_eq!(n[0], msgs[0]);
        assert_eq!(n[2], msgs[2]);
        assert_eq!(n[1]["tool_calls"][0]["id"], "call_0");

        // an object stays an object, and so does the empty string: the template's own
        // `arguments != ''` guard skips that one, and it is what the engine's open chunk sends
        for same in [
            serde_json::json!([{"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": {"a": 1}}}]}]),
            serde_json::json!([{"role": "assistant", "tool_calls": [
                {"function": {"name": "f", "arguments": ""}}]}]),
            serde_json::json!([{"role": "assistant", "tool_calls": [
                {"function": {"name": "f"}}]}]),
        ] {
            let (n, notes) = normalize_messages(&same);
            assert_eq!(n, same, "rewritten: {same}");
            assert!(notes.is_empty(), "{notes:?}");
        }
    }

    /// TASK J: the shape that killed robin's session. Every `arguments` value that is not a
    /// mapping becomes one, the note names the message and the tool_call index, and the
    /// value the model sent is still in the render.
    #[test]
    fn no_arguments_shape_reaches_the_template_as_a_non_mapping() {
        // the truncated call the engine used to emit, and every other shape Crow can store:
        // `crow_core.py:5068` concatenates the fragments into a string and never parses it
        let shapes: Vec<(serde_json::Value, serde_json::Value)> = vec![
            // (as it arrives, what the render must see)
            (serde_json::json!("{\"path\": \"C:/x/y.m"), serde_json::json!({RAW_KEY: "{\"path\": \"C:/x/y.m"})),
            (serde_json::json!("not json"), serde_json::json!({RAW_KEY: "not json"})),
            (serde_json::json!("{'path': 'a.md'}"), serde_json::json!({RAW_KEY: "{'path': 'a.md'}"})),
            (serde_json::json!("[1, 2]"), serde_json::json!({RAW_KEY: "[1, 2]"})),
            (serde_json::json!("3"), serde_json::json!({RAW_KEY: "3"})),
            (serde_json::json!("null"), serde_json::json!({RAW_KEY: "null"})),
            (serde_json::json!("\"{\\\"path\\\": \\\"a.md\\\"}\""), serde_json::json!({RAW_KEY: "\"{\\\"path\\\": \\\"a.md\\\"}\""})),
            (serde_json::json!(null), serde_json::json!({})),
            (serde_json::json!([1, 2]), serde_json::json!({RAW_KEY: "[1,2]"})),
            (serde_json::json!(7), serde_json::json!({RAW_KEY: "7"})),
            (serde_json::json!(true), serde_json::json!({RAW_KEY: "true"})),
        ];
        let tk = tk();
        let tools = crow_nest_engine::toolcall::a7_tools_fixture();
        for (arrived, wanted) in shapes {
            let msgs = serde_json::json!([
                {"role": "user", "content": "Read a.md"},
                {"role": "assistant", "content": "", "tool_calls": [
                    {"id": "call_0", "type": "function",
                     "function": {"name": "read_file", "arguments": arrived}}]},
                {"role": "tool", "tool_call_id": "call_0", "content": "# Title"}
            ]);
            // before the fix this was the 400 of record, at chat:136
            let raw = tk.render_chat(&msgs, Some(&tools), true, false);
            if arrived.as_str() == Some("") {
                assert!(raw.is_ok());
            } else {
                let e = raw.expect_err("the un-normalized form must not render");
                assert!(e.contains("cannot convert value into pairs (in chat:136)"), "{e}");
            }
            let (n, notes) = normalize_messages(&msgs);
            assert_eq!(n[1]["tool_calls"][0]["function"]["arguments"], wanted, "arrived {arrived}");
            assert_eq!(notes.len(), 1, "one note per rewrite: {notes:?}");
            assert!(notes[0].starts_with("message 1 tool_call 0 function.arguments is"), "{}", notes[0]);
            assert!(check_messages(&n).is_ok(), "arrived {arrived}");
            let s = tk
                .render_chat(&n, Some(&tools), true, false)
                .unwrap_or_else(|e| panic!("arrived {arrived} still does not render: {e}"));
            // the model still sees what the previous turn asked for
            if let Some(raw_text) = wanted.get(RAW_KEY).and_then(|v| v.as_str()) {
                assert!(s.contains(&format!("<parameter={RAW_KEY}>")), "{s}");
                assert!(s.contains(raw_text), "the value the turn sent is gone: {s}");
            }
        }
    }

    /// TASK J: the note a rewrite writes to stderr names the message, the tool_call and the
    /// first 200 bytes of the value, so the log alone says which history entry was broken.
    #[test]
    fn a_rewrite_note_names_the_message_the_call_and_the_first_200_bytes() {
        let long = "x".repeat(500);
        let msgs = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "", "tool_calls": [
                {"function": {"name": "a", "arguments": "{\"p\": 1}"}},
                {"function": {"name": "b", "arguments": long}}]},
        ]);
        let (_, notes) = normalize_messages(&msgs);
        assert_eq!(notes.len(), 1, "{notes:?}");
        let n = &notes[0];
        assert!(n.starts_with("message 1 tool_call 1 function.arguments is a string"), "{n}");
        assert!(n.contains("(500 B total)"), "{n}");
        assert!(n.contains(&"x".repeat(200)), "{n}");
        assert!(!n.contains(&"x".repeat(201)), "the log line is capped at 200 bytes: {n}");
        // TASK K: and WHY it is not a mapping - serde_json's own message plus the byte
        // window it stopped in, which is what robin's 3,112-byte case did not carry
        assert!(n.contains("serde_json:"), "the decoder's message is missing: {n}");
        assert!(n.contains("<HERE>"), "the byte window is missing: {n}");
        // the digest carries the same locator, one line per message
        let d = message_digest(&msgs);
        assert_eq!(d.len(), 2);
        assert!(d[0].starts_with("message 0 role=user content=string(2 B)"), "{}", d[0]);
        assert!(d[1].contains("tool_calls=2"), "{}", d[1]);
        assert!(d[1].contains("[1 name=\"b\" arguments=a string"), "{}", d[1]);
    }

    /// TASK J: the neighbouring hazards of `chat:136`. Each one either RENDERS or is refused
    /// by `check_messages` with a 400 that names the message index and the field - never the
    /// bare `chat template render failed` the log could not act on.
    #[test]
    fn every_neighbouring_template_hazard_renders_or_is_named() {
        let tk = tk();
        let tools = crow_nest_engine::toolcall::a7_tools_fixture();
        let call = serde_json::json!({"function": {"name": "f", "arguments": {"a": 1}}});
        // renders: the shapes Crow really sends (`crow_core.py:3555-3566`, `:13739-13741`,
        // `:3754-3755`) plus the ones the template guards itself
        let renders: Vec<(&str, serde_json::Value)> = vec![
            ("a text-only content list", serde_json::json!([{"role": "user", "content": [{"type": "text", "text": "hi"}]}])),
            ("a text plus image_url list", serde_json::json!([{"role": "user", "content": [
                {"type": "text", "text": "hi"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA"}}]}])),
            ("an empty content list", serde_json::json!([{"role": "user", "content": []}])),
            ("a null content", serde_json::json!([{"role": "user", "content": null}])),
            ("an absent content", serde_json::json!([{"role": "user"}])),
            ("a tool result carrying an image", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "tool_calls": [call.clone()]},
                {"role": "tool", "content": [{"type": "text", "text": "r"}, {"type": "image_url", "image_url": {"url": "u"}}]}])),
            ("a null tool result", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "tool_calls": [call.clone()]},
                {"role": "tool", "content": null}])),
            ("reasoning_content, a string", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "reasoning_content": "thought"}])),
            ("reasoning_content, not a string", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "reasoning_content": {"a": 1}}])),
            ("a null tool_calls", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "tool_calls": null}])),
            ("an empty tool_calls", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "x", "tool_calls": []}])),
            ("a call with no function key", serde_json::json!([{"role": "user", "content": "hi"},
                {"role": "assistant", "content": "", "tool_calls": [{"name": "f", "arguments": {"a": 1}}]},
                {"role": "tool", "content": "r"}])),
        ];
        for (label, msgs) in renders {
            let (n, _) = normalize_messages(&msgs);
            check_messages(&n).unwrap_or_else(|e| panic!("{label} must render, refused: {e}"));
            tk.render_chat(&n, Some(&tools), true, false)
                .unwrap_or_else(|e| panic!("{label} must render: {e}"));
        }
        // refused, each with the index and the field in the body
        let refused: Vec<(serde_json::Value, &str)> = vec![
            (serde_json::json!([{"role": "user", "content": {"text": "hi"}}]), "message 0 content is an object"),
            (serde_json::json!([{"role": "user", "content": 5}]), "message 0 content is a number"),
            (serde_json::json!([{"role": "user", "content": ["hi"]}]), "message 0 content part 0 is a string"),
            (serde_json::json!([{"role": "user", "content": [null]}]), "message 0 content part 0 is null"),
            (serde_json::json!([{"role": "user", "content": [{"type": "text"}]}]), "message 0 content part 0 is neither a text nor an image_url block"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": "abc"}]), "message 1 tool_calls is a string"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": {"a": 1}}]), "message 1 tool_calls is an object"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": [5]}]), "message 1 tool_call 0 is a number, not an object"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": [{"function": "f"}]}]), "message 1 tool_call 0 function is a string, not an object"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": [{"function": null}]}]), "message 1 tool_call 0 function is null, not an object"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": [{"function": {"arguments": {}}}]}]), "message 1 tool_call 0 has no string function.name"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "assistant", "content": "x", "tool_calls": [{"function": {"name": 5}}]}]), "message 1 tool_call 0 has no string function.name"),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "weird", "content": "x"}]), "message 1 has role \"weird\""),
            (serde_json::json!([{"role": "user", "content": "hi"}, {"role": "system", "content": "x"}]), "message 1 is a system message"),
        ];
        for (msgs, wanted) in refused {
            let (n, _) = normalize_messages(&msgs);
            let e = check_messages(&n).expect_err(&format!("{msgs} was served"));
            assert!(e.contains(wanted), "body {e:?} does not name {wanted:?}");
            // and the template agrees: it could not have rendered it either
            assert!(tk.render_chat(&n, Some(&tools), true, false).is_err()
                    || msgs[1].get("tool_calls").map(|t| !t.is_array()).unwrap_or(false),
                    "refused a shape the template renders: {msgs}");
        }
    }

    /// Provenance of `ORACLE_RENDER`: run once on 2026-09-09 with
    ///   `.venv-oracle/Scripts/python.exe` (CPython 3.13.3), `PYTHONIOENCODING=utf-8`,
    ///   transformers 5.16.1,
    ///   `AutoTokenizer.from_pretrained("models/Qwen3.8-Flash-Next-original")`
    ///   `.apply_chat_template(HIST, tools=TOOLS, add_generation_prompt=True,`
    ///   `    tokenize=False, enable_thinking=False)`
    ///   with `TOOLS` = `crow_nest_engine::toolcall::a7_tools_fixture()` and `HIST` the three messages below, the assistant
    ///   turn carrying `arguments` as the OBJECT `{"path": "a.md", "start_line": 1}`.
    /// Measured in the same run: the same call with `arguments` as the STRING
    ///   `"{\"path\": \"a.md\", \"start_line\": 1}"` raises
    ///   `TypeError: Can only get item pairs from a mapping`, because the template
    ///   iterates `tool_call.arguments|items`. That is why `normalize_messages` exists.
    #[test]
    fn a_history_with_a_tool_turn_renders_byte_identical_to_the_oracle() {
        let tk = tk();
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
        let tools = crow_nest_engine::toolcall::a7_tools_fixture();
        let s = tk
            .render_chat(&normalize_messages(&msgs).0, Some(&tools), true, false)
            .expect("the tool history renders");
        assert_eq!(s, ORACLE_RENDER);
        // without the conversion the template cannot iterate the arguments at all
        let raw = tk.render_chat(&msgs, Some(&tools), true, false);
        assert!(raw.is_err(), "the string form must not render silently: {raw:?}");
    }

    // ---- #67: the reasoning filter, both ends (2026-09-18) ----

    /// feed `text` through the filter at EVERY piece size from 1 to its length, plus whole,
    /// and return the concatenated `(content, reasoning, tags stripped)` of each run once -
    /// every split must give the same answer, which is the whole point of the hold back
    fn filtered(text: &str) -> (String, String, usize) {
        filtered_from(false, text)
    }

    /// #74: the same sweep with the start state of a request that DOES think - the prompt
    /// opened the block, so the filter starts `Inside` and nothing will open one
    fn filtered_from(enable_thinking: bool, text: &str) -> (String, String, usize) {
        let chars: Vec<char> = text.chars().collect();
        let mut first: Option<(String, String, usize)> = None;
        for step in 1..=chars.len().max(1) {
            let mut f = ThinkFilter::for_request(enable_thinking);
            let (mut content, mut reasoning) = (String::new(), String::new());
            for piece in chars.chunks(step) {
                let piece: String = piece.iter().collect();
                let out = f.push(&piece);
                content.push_str(&out.content);
                reasoning.push_str(&out.reasoning);
            }
            let tail = f.flush();
            content.push_str(&tail.content);
            reasoning.push_str(&tail.reasoning);
            let got = (content, reasoning, f.stripped());
            match &first {
                None => first = Some(got),
                Some(want) => assert_eq!(&got, want, "piece size {step} differs on {text:?}"),
            }
        }
        first.unwrap_or_default()
    }

    /// no tag at all: every byte comes back, in order, whatever the deltas were
    #[test]
    fn a_stream_without_a_reasoning_tag_is_a_byte_identical_passthrough() {
        for text in [
            "Hello.",
            "Let me test Chromium full-page screenshot of a tiny local page first.",
            // the characters that make the scanner look twice, and never a tag
            "a < b and c <= d",
            "<thinker>, <thin, <, <<, </thin, </thinking>",
            // leading and trailing whitespace of an untouched answer survives
            "\n\n  indented\n",
            "if (a<b) { return \"<think\"; }",
            // non ASCII around the scanner's `<` fast path
            "Gr\u{fc}\u{df}e \u{1f985} < \u{e4}\u{f6}\u{fc}",
        ] {
            let (content, reasoning, stripped) = filtered(text);
            assert_eq!(content, text, "content changed for {text:?}");
            assert_eq!(reasoning, "", "reasoning invented for {text:?}");
            assert_eq!(stripped, 0, "a tag was counted in {text:?}");
        }
    }

    /// the shape of #67: the stray closing tag at the very end of a turn, and the tag
    /// arriving across several deltas - `filtered` drives every split point of both
    #[test]
    fn a_stray_closing_tag_never_reaches_the_content_at_any_split() {
        // robin's line of record, tag and all
        let (content, reasoning, stripped) = filtered(
            "Let me test Chromium full-page screenshot of a tiny local page first:\n</think>",
        );
        assert_eq!(
            content,
            "Let me test Chromium full-page screenshot of a tiny local page first:\n"
        );
        assert_eq!(reasoning, "");
        assert_eq!(stripped, 1);
        assert!(!content.contains(THINK_CLOSE));
        // mid-answer, and twice
        let (content, _, stripped) = filtered("a</think>b</think>c");
        assert_eq!(content, "abc");
        assert_eq!(stripped, 2);
        // the whole answer IS the tag
        assert_eq!(filtered("</think>"), (String::new(), String::new(), 1));
        // a prefix that never completes is TEXT, not a tag: no byte is lost to the filter
        assert_eq!(filtered("done</think").0, "done</think");
        assert_eq!(filtered("done</think").2, 0);
        assert_eq!(filtered("done<").0, "done<");
    }

    /// a block the model opened itself becomes `reasoning_content`, never `content`
    #[test]
    fn a_leading_think_block_is_reasoning_content_and_never_content() {
        let (content, reasoning, stripped) =
            filtered("<think>\nfirst I check the path\n</think>\n\nThe file is empty.");
        assert_eq!(content, "The file is empty.");
        assert_eq!(reasoning, "\nfirst I check the path\n");
        assert_eq!(stripped, 2);
        // whitespace before the block goes with it, and so does the `\n\n` after it
        assert_eq!(filtered("\n<think>x</think>\n\nA").0, "A");
        // a block that never closes stays reasoning: the model opened it, nothing closed it
        let (content, reasoning, _) = filtered("<think>still thinking");
        assert_eq!(content, "");
        assert_eq!(reasoning, "still thinking");
        // `<think>` in the MIDDLE of an answer is ordinary text; only a leading block is owned
        assert_eq!(filtered("see <think> below").0, "see <think> below");
    }

    /// the filter sits on `Emit::Content` alone: an `arguments` fragment is never rewritten
    #[test]
    fn the_filter_leaves_every_tool_call_fragment_alone() {
        let pieces = vec![
            Emit::Content("答え</think>".to_string()),
            Emit::Call {
                index: 0,
                id: "call_0".to_string(),
                name: "write_file".to_string(),
            },
            // a value that CONTAINS the tag: the arguments contract of 7.11.14 is bytes
            Emit::Args {
                index: 0,
                text: "{\"content\":\"</think>\"}".to_string(),
            },
        ];
        let mut col = CollectSink::default();
        let mut think = ThinkFilter::new();
        let mut counts = Chunks::default();
        assert!(send_emits(&mut col, &ChunkCtx::new("i", 1, "m"), &pieces, &mut think, &mut counts));
        assert_eq!(col.content, "答え");
        assert_eq!(col.calls[0].arguments, "{\"content\":\"</think>\"}");
        assert_eq!(think.stripped(), 1);
        assert_eq!((counts.content, counts.reasoning, counts.tool), (1, 0, 2));
        // and the reasoning half reaches both sinks as its own frame
        let mut buf: Vec<u8> = Vec::new();
        let mut sse = SseSink::new(&mut buf);
        let mut f2 = ThinkFilter::new();
        let mut c2 = Chunks::default();
        assert!(send_emits(
            &mut sse,
            &ChunkCtx::new("i", 1, "m"),
            &[Emit::Content("<think>why</think>then".to_string())],
            &mut f2,
            &mut c2
        ));
        let text = String::from_utf8(buf).expect("utf8 frames");
        assert!(text.contains(r#""reasoning_content":"why""#), "{text}");
        assert!(text.contains(r#""content":"then""#), "{text}");
        assert!(!text.contains(THINK_CLOSE), "{text}");
        assert_eq!((c2.content, c2.reasoning), (1, 1));
    }

    /// - #67 second end, against the REAL template: what it actually does with a stored
    ///   `</think>`, and what the normaliser makes of it
    /// - the issue expected `content.split('</think>')[-1]` (the turn would render EMPTY).
    ///   This template has no such rule: it renders `reasoning_content` into its own think
    ///   block and the stored `content` VERBATIM after it, so the text is never lost and the
    ///   turn carries TWO closing tags - which is the shape the model imitates.
    #[test]
    fn a_history_whose_last_assistant_turn_ends_with_the_tag_keeps_its_text() {
        let tk = tk();
        let answer = "Let me test the screenshot at width 1280:\n</think>";
        let msgs = serde_json::json!([
            {"role": "user", "content": "Take a screenshot."},
            {"role": "assistant", "content": answer},
            {"role": "user", "content": "And now?"}
        ]);
        // 1) the template does NOT cut the turn at the tag - the text is all there
        let raw = tk.render_chat(&msgs, None, true, false).expect("the raw history renders");
        assert!(raw.contains("Let me test the screenshot at width 1280:"), "{raw}");
        // 2) and the assistant turn carries the stray INSIDE the template's own block
        const POISONED: &str = concat!(
            "<|im_start|>assistant\n",
            "<think>\n",
            "\n",
            "</think>\n",
            "\n",
            "Let me test the screenshot at width 1280:\n",
            "</think><|im_end|>\n",
        );
        assert!(raw.contains(POISONED), "{raw}");
        assert_eq!(raw.matches(THINK_CLOSE).count(), 3, "{raw}");
        // 3) after the normaliser: the text is kept, the stray is gone, and the only closing
        //    tags left are the template's own two (the stored turn's, and the header's)
        let (clean, notes) = normalize_messages(&msgs);
        let s = tk.render_chat(&clean, None, true, false).expect("the clean history renders");
        assert!(s.contains("Let me test the screenshot at width 1280:"), "{s}");
        assert!(s.contains(concat!(
            "<|im_start|>assistant\n",
            "<think>\n",
            "\n",
            "</think>\n",
            "\n",
            "Let me test the screenshot at width 1280:<|im_end|>\n",
        )), "{s}");
        assert_eq!(s.matches(THINK_CLOSE).count(), 2, "{s}");
        assert_eq!(notes.len(), 1, "{notes:?}");
        assert!(notes[0].starts_with("message 1 assistant content carried a <think>"), "{notes:?}");
        // the ids of the clean history are the ids of the same history written by hand
        let hand = serde_json::json!([
            {"role": "user", "content": "Take a screenshot."},
            {"role": "assistant", "content": "Let me test the screenshot at width 1280:"},
            {"role": "user", "content": "And now?"}
        ]);
        assert_eq!(
            tk.encode_chat(&clean, None, true, false).expect("clean ids"),
            tk.encode_chat(&hand, None, true, false).expect("hand ids")
        );
    }

    /// the normaliser strips the two shapes that NEST and leaves every other one alone
    #[test]
    fn the_normaliser_strips_only_the_two_shapes_that_nest() {
        // stripped
        assert_eq!(strip_stored_think("answer\n</think>").as_deref(), Some("answer"));
        assert_eq!(strip_stored_think("answer</think>\n\n").as_deref(), Some("answer"));
        assert_eq!(strip_stored_think("</think>").as_deref(), Some(""));
        assert_eq!(strip_stored_think("a</think>\n</think>").as_deref(), Some("a"));
        assert_eq!(
            strip_stored_think("<think>\nplan\n</think>\n\nanswer").as_deref(),
            Some("answer")
        );
        assert_eq!(strip_stored_think("<think>plan</think>done</think>").as_deref(), Some("done"));
        // left alone, byte for byte: `None` is the contract, not an equal string
        for untouched in [
            "",
            "a plain answer",
            "the tag </think> in the middle of a sentence",
            "<think> that never closes",
            "</think> at the start, text after it",
        ] {
            assert_eq!(strip_stored_think(untouched), None, "{untouched:?} was rewritten");
        }
        // only the assistant role, only a string content, and `reasoning_content` untouched
        let msgs = serde_json::json!([
            {"role": "user", "content": "keep </think> here"},
            {"role": "assistant", "content": "kept\n</think>", "reasoning_content": "r</think>"},
            {"role": "tool", "tool_call_id": "call_0", "content": "result </think>"}
        ]);
        let (out, notes) = normalize_messages(&msgs);
        assert_eq!(out[0]["content"], "keep </think> here");
        assert_eq!(out[1]["content"], "kept");
        assert_eq!(out[1]["reasoning_content"], "r</think>");
        assert_eq!(out[2]["content"], "result </think>");
        assert_eq!(notes.len(), 1);
        // a history without the tag comes back identical, object for object
        let clean = serde_json::json!([
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "there"}
        ]);
        assert_eq!(normalize_messages(&clean), (clean.clone(), Vec::new()));
    }

    // ---- #74: thinking, the two doors, the mapping and the filter's start (2026-09-18) ----

    /// The whole wire contract of `reasoning_effort` on one screen: which door was read,
    /// what the template gets, what stays off, and which word is a 400. Every row of the
    /// table on `map_reasoning_effort` is here.
    #[test]
    fn the_two_doors_of_reasoning_effort_resolve_to_the_mapping_of_record() {
        let body = |extra: &str| -> Vec<u8> {
            format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]{extra}}}")
                .into_bytes()
        };
        // what the client sent -> (enable_thinking, the template word)
        let rows: [(&str, bool, Option<&str>); 12] = [
            // nothing at all: the default of record, both doors silent
            ("", false, None),
            // the TOP-LEVEL door, the one Crow has used since its #176
            (",\"reasoning_effort\":\"none\"", false, None),
            (",\"reasoning_effort\":\"low\"", true, Some("low")),
            (",\"reasoning_effort\":\"medium\"", true, Some("medium")),
            (",\"reasoning_effort\":\"high\"", true, Some("xhigh")),
            (",\"reasoning_effort\":\"xhigh\"", true, Some("xhigh")),
            // an explicit null is an absent field, as everywhere else in this parser
            (",\"reasoning_effort\":null", false, None),
            // the KWARGS door, the one this file has always read
            (",\"chat_template_kwargs\":{\"enable_thinking\":true}", true, None),
            (",\"chat_template_kwargs\":{\"enable_thinking\":false}", false, None),
            (",\"chat_template_kwargs\":{\"reasoning_effort\":\"medium\"}", true, Some("medium")),
            // both doors: the top-level one is read first, as llama-server reads it
            (
                ",\"reasoning_effort\":\"low\",\"chat_template_kwargs\":{\"reasoning_effort\":\"medium\"}",
                true,
                Some("low"),
            ),
            // an explicit `enable_thinking: false` still wins - it is the direct variable,
            // and llama-server's own `none` is the only thing that overrules a level
            (
                ",\"reasoning_effort\":\"high\",\"chat_template_kwargs\":{\"enable_thinking\":false}",
                false,
                None,
            ),
        ];
        for (extra, thinking, effort) in rows {
            let r = parse_chat(&body(extra)).unwrap_or_else(|e| panic!("{extra:?}: {e}"));
            assert_eq!(r.enable_thinking, thinking, "enable_thinking for {extra:?}");
            assert_eq!(r.reasoning_effort, effort, "reasoning_effort for {extra:?}");
            // the variable is NEVER defined while thinking is off, so a non-thinking
            // request renders the prompt of record whatever word it named
            assert!(thinking || r.reasoning_effort.is_none(), "{extra:?}");
        }
        // a word this engine does not have is a 400 that NAMES the five, before any GPU work
        for bad in ["max", "minimal", "off", "High", "XHIGH", "", "none ", "xhigh2"] {
            let raw = body(&format!(",\"reasoning_effort\":\"{bad}\""));
            let e = parse_chat(&raw).expect_err("{bad} must be a 400");
            assert!(e.starts_with("reasoning_effort "), "{bad}: {e}");
            assert!(e.contains(REASONING_WORDS), "{bad}: {e}");
        }
        // a non-string is a 400 too, with the same list
        let e = parse_chat(&body(",\"reasoning_effort\":3")).expect_err("a number is a 400");
        assert!(e.contains(REASONING_WORDS), "{e}");
        // the mapping itself, word for word
        assert_eq!(map_reasoning_effort("none"), Ok(None));
        assert_eq!(map_reasoning_effort("low"), Ok(Some("low")));
        assert_eq!(map_reasoning_effort("medium"), Ok(Some("medium")));
        assert_eq!(map_reasoning_effort("high"), Ok(Some("xhigh")));
        assert_eq!(map_reasoning_effort("xhigh"), Ok(Some("xhigh")));
    }

    /// The line the whole issue rests on: a request that names NEITHER door renders the ids
    /// of record, byte for byte. Every parity value, every gate value and every prefix-cache
    /// hit of a running session depend on this, so it is asserted against the frozen oracle
    /// ids and not against a second call.
    #[test]
    fn a_request_that_names_no_level_renders_the_ids_of_record() {
        let tk = tk();
        let msgs = crow_nest_engine::tokenizer::user_message("Hello");
        // the oracle of `tokenizer.rs`: transformers 5.16.1, enable_thinking=False
        const ORACLE: [u32; 13] =
            [248045, 846, 198, 9419, 248046, 198, 248045, 74455, 198, 248068, 271, 248069, 271];
        let plain = parse_chat(br#"{"messages":[{"role":"user","content":"Hello"}]}"#).unwrap();
        let ids = tk
            .encode_chat_effort(&msgs, None, true, plain.enable_thinking, plain.reasoning_effort)
            .unwrap();
        assert_eq!(ids, ORACLE.to_vec());
        // `none` is the same prompt: that is what makes it the word for "do not think"
        let none = parse_chat(
            br#"{"messages":[{"role":"user","content":"Hello"}],"reasoning_effort":"none"}"#,
        )
        .unwrap();
        assert_eq!(
            tk.encode_chat_effort(&msgs, None, true, none.enable_thinking, none.reasoning_effort)
                .unwrap(),
            ORACLE.to_vec()
        );
        // a level does move the prompt, and it ends INSIDE the block, not after a closed one
        for (word, len_differs) in [("low", true), ("medium", true), ("high", true)] {
            let raw = format!(
                "{{\"messages\":[{{\"role\":\"user\",\"content\":\"Hello\"}}],\"reasoning_effort\":\"{word}\"}}"
            );
            let r = parse_chat(raw.as_bytes()).unwrap();
            let s = tk
                .render_chat_effort(&msgs, None, true, r.enable_thinking, r.reasoning_effort)
                .unwrap();
            assert!(s.ends_with("<|im_start|>assistant\n<think>\n"), "{word}: {s}");
            assert!(!s.contains("</think>"), "{word}: {s}");
            let ids = tk
                .encode_chat_effort(&msgs, None, true, r.enable_thinking, r.reasoning_effort)
                .unwrap();
            assert_eq!(ids != ORACLE.to_vec(), len_differs, "{word}");
        }
    }

    /// The `[chat]` line has to say whether the request thought, at which level, and whether
    /// the body asked for it - the provenance shape #68 gave the sampling line.
    #[test]
    fn the_chat_line_says_whether_the_request_thought_and_who_asked() {
        let parse = |extra: &str| -> ChatReq {
            let raw =
                format!("{{\"messages\":[{{\"role\":\"user\",\"content\":\"hi\"}}]{extra}}}");
            parse_chat(raw.as_bytes()).unwrap()
        };
        // nothing asked: off, and this file decided it
        let d = parse("");
        assert_eq!(thinking_tag(&d), "off");
        assert!(thinking_line(&d).starts_with("[chat] thinking off (data sheet);"), "{}", thinking_line(&d));
        // `none`: still off, but the CLIENT chose it
        let n = parse(",\"reasoning_effort\":\"none\"");
        assert_eq!(thinking_tag(&n), "off");
        let l = thinking_line(&n);
        assert!(l.starts_with("[chat] thinking off (request)"), "{l}");
        assert!(l.contains("asked as \"none\""), "{l}");
        // `high`: on, and the line says BOTH words, because they are not the same word
        let h = parse(",\"reasoning_effort\":\"high\"");
        assert_eq!(thinking_tag(&h), "xhigh");
        let l = thinking_line(&h);
        assert!(l.starts_with("[chat] thinking on (request): reasoning_effort xhigh, asked as \"high\""), "{l}");
        assert!(l.contains("the reasoning filter starts Inside"), "{l}");
        // a word the template owns is not repeated back
        let m = parse(",\"reasoning_effort\":\"medium\"");
        assert_eq!(thinking_tag(&m), "medium");
        assert!(!thinking_line(&m).contains("asked as"), "{}", thinking_line(&m));
        // the kwargs door alone: on, from the request, and the template's own default
        let k = parse(",\"chat_template_kwargs\":{\"enable_thinking\":true}");
        assert_eq!(thinking_tag(&k), "xhigh");
        assert!(thinking_line(&k).starts_with("[chat] thinking on (request): reasoning_effort xhigh;"), "{}", thinking_line(&k));
        assert_eq!(SamplingSent::tag(true), "request");
    }

    /// #74's half of the reasoning filter: with thinking on, the PROMPT opened the block, so
    /// the filter starts `Inside`. Started in `Lead` the reasoning would go out as `content`
    /// and the model's own `</think>` would be dropped as a stray - the answer would carry the
    /// thinking, and Crow would store and re-send it every turn (7.11.16).
    #[test]
    fn a_thinking_stream_starts_inside_the_block_the_prompt_opened() {
        // what a thinking generation really looks like: no opening tag, it is in the prompt
        let text = "The user asks for a colour.
Red is #FF0000.
</think>

The hex is #FF0000.";
        let (content, reasoning, stripped) = filtered_from(true, text);
        assert_eq!(content, "The hex is #FF0000.");
        assert_eq!(reasoning, "The user asks for a colour.
Red is #FF0000.
");
        assert_eq!(stripped, 1);
        assert!(!content.contains(THINK_OPEN) && !content.contains(THINK_CLOSE));
        assert!(!reasoning.contains(THINK_CLOSE));
        // the SAME bytes through the filter of a non-thinking request: this is the bug
        let (bad_content, bad_reasoning, _) = filtered_from(false, text);
        assert!(bad_content.starts_with("The user asks for a colour."), "{bad_content}");
        assert_eq!(bad_reasoning, "");
        // a second `</think>` later in the answer is still the stray of #67
        let (content, _, stripped) = filtered_from(true, "t
</think>

A</think>B");
        assert_eq!(content, "AB");
        assert_eq!(stripped, 2);
        // a `<think>` INSIDE the reasoning is ordinary reasoning text, not a second block
        let (content, reasoning, stripped) =
            filtered_from(true, "I will write <think> here
</think>

Done.");
        assert_eq!(content, "Done.");
        assert_eq!(reasoning, "I will write <think> here
");
        assert_eq!(stripped, 1);
    }

    /// `max_tokens` covers thinking PLUS answer, so a budget that runs out mid-thought is an
    /// ordinary `finish length` - and nothing half-open may reach the wire. Both request
    /// forms, because the document is the same filter with a different sink.
    #[test]
    fn a_thinking_request_that_hits_the_cap_ends_without_a_half_open_tag() {
        // the block never closed: every byte is reasoning, the answer is empty
        let (content, reasoning, stripped) = filtered_from(true, "still weighing the two option");
        assert_eq!(content, "");
        assert_eq!(reasoning, "still weighing the two option");
        assert_eq!(stripped, 0);
        // the budget ended INSIDE a tag the filter was holding back: it is text, and it
        // leaves as reasoning, so no byte is lost and no tag is invented
        let (content, reasoning, stripped) = filtered_from(true, "nearly done
</thin");
        assert_eq!(content, "");
        assert_eq!(reasoning, "nearly done
</thin");
        assert_eq!(stripped, 0);
        // the `stream:false` document of that same request: reasoning in its own field,
        // `content` an empty string, `finish_reason` length
        let t = b3a_timing();
        let d = completion_json(
            &ChunkCtx::new("chatcmpl-74", 74, "crow-nest"),
            "",
            &reasoning,
            &[],
            "length",
            &t,
        );
        let m = &d["choices"][0]["message"];
        assert_eq!(m["content"], "");
        assert_eq!(m["reasoning_content"], "nearly done
</thin");
        assert_eq!(d["choices"][0]["finish_reason"], "length");
        assert!(!d.to_string().contains("</think>"));
        // and the ordinary thinking document: answer in `content`, thought beside it
        let d = completion_json(
            &ChunkCtx::new("chatcmpl-74", 74, "crow-nest"),
            "The hex is #FF0000.",
            "Red is #FF0000.
",
            &[],
            "stop",
            &t,
        );
        let m = &d["choices"][0]["message"];
        assert_eq!(m["content"], "The hex is #FF0000.");
        assert_eq!(m["reasoning_content"], "Red is #FF0000.
");
    }

    /// The history side, through the REAL template: a stored assistant turn carrying
    /// `reasoning_content` renders into the template's OWN think block, and the next
    /// generation prompt still ends where the filter expects it to.
    #[test]
    fn a_history_with_reasoning_content_renders_into_the_templates_own_think_block() {
        let tk = tk();
        let msgs = serde_json::json!([
            {"role": "user", "content": "What colour?"},
            {"role": "assistant", "content": "Red is #FF0000.",
             "reasoning_content": "The user asks for a colour."},
            {"role": "user", "content": "And blue?"}
        ]);
        let (n, notes) = normalize_messages(&msgs);
        // nothing to repair: `reasoning_content` is the template's own field (7.11.16)
        assert!(notes.is_empty(), "{notes:?}");
        assert_eq!(n, msgs);
        let s = tk.render_chat_effort(&n, None, true, true, Some("xhigh")).unwrap();
        assert!(
            s.contains("<|im_start|>assistant
<think>
The user asks for a colour.
</think>

Red is #FF0000.<|im_end|>"),
            "{s}"
        );
        // the stored thought is NOT repeated into the answer, and the answer is not in the block
        assert_eq!(s.matches("The user asks for a colour.").count(), 1, "{s}");
        // the generation prompt still opens the block the filter starts inside
        assert!(s.ends_with("<|im_start|>assistant
<think>
"), "{s}");
        assert!(s.contains("Reasoning effort is set to xhigh."), "{s}");
        // the same history without thinking renders the closed empty block, as it always did
        let off = tk.render_chat(&n, None, true, false).unwrap();
        assert!(off.ends_with("<|im_start|>assistant
<think>

</think>

"), "{off}");
        assert!(!off.contains("Reasoning effort is set to"), "{off}");
        // and the assistant turn of the history keeps its own think block either way
        assert!(off.contains("<think>
The user asks for a colour.
</think>

Red is #FF0000."), "{off}");
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
    /// tests run from engine/, the model lives at the repository root
    fn tk() -> crow_nest_engine::tokenizer::ChatTokenizer {
        let t = format!("../{}", crow_nest_engine::tokenizer::DEFAULT_TOKENIZER);
        crow_nest_engine::tokenizer::ChatTokenizer::load(
            &t,
            &crow_nest_engine::tokenizer::sibling_config(&t),
        )
        .expect("tokenizer loads")
    }

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
        let d = completion_json(&ChunkCtx::new("chatcmpl-1700-2", 1700, "crow"), "hello", "", &[], "stop", &t);
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
        let l = completion_json(&ChunkCtx::new("x", 1, "crow"), "hi", "", &[], "length", &t);
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
        let d = completion_json(&ChunkCtx::new("id1", 5, "crow"), "", "", &calls, "tool_calls", &t);
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
        let (mut sf, mut cf) = (ThinkFilter::new(), ThinkFilter::new());
        let (mut sc, mut cc) = (Chunks::default(), Chunks::default());
        assert!(send_emits(&mut sse, &ChunkCtx::new("id1", 5, "crow"), &pieces, &mut sf, &mut sc));
        let mut col = CollectSink::default();
        assert!(send_emits(&mut col, &ChunkCtx::new("id1", 5, "crow"), &pieces, &mut cf, &mut cc));
        // the same loop counts the same chunks for both sinks
        assert_eq!(sc, cc);
        assert_eq!((sc.content, sc.reasoning, sc.tool), (2, 0, 3));

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
        let mut think = ThinkFilter::new();
        let mut counts = Chunks::default();
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
        assert!(send_emits(&mut col, &ChunkCtx::new("i", 1, "m"), &pieces, &mut think, &mut counts));
        assert_eq!(col.calls.len(), 2);
        assert_eq!(col.calls[0].arguments, "{}");
        assert_eq!(col.calls[1].name, "b");
        assert_eq!(col.calls[1].arguments, "{\"x\":1}");
        assert_eq!((counts.content, counts.reasoning, counts.tool), (0, 0, 5));
    }

    // ---- TASK K: the producer end of the `arguments` contract, and the diagnostics ----

    /// the shape the parser always produces passes through byte for byte
    #[test]
    fn a_json_object_argument_string_is_left_alone() {
        for good in [
            r#"{}"#,
            r#"{"path":"/etc/hostname"}"#,
            r#"{"path":"/x","content":"<!doctype html>\n<p class=\"a\">\u0009</p>"}"#,
            r#"{"path":"/etc/host","_truncated":true}"#,
            "",
        ] {
            let (out, note) = args_object_or_raw(good);
            assert_eq!(out, good, "a valid arguments string was rewritten: {good}");
            assert!(note.is_none(), "a valid arguments string produced a note: {note:?}");
        }
    }

    /// anything else becomes `{"_raw": ...}` and the note carries serde_json's own words
    /// plus the byte window - the two things robin's 2026-09-17 line did not have
    #[test]
    fn a_broken_argument_string_becomes_raw_and_the_note_names_the_defect() {
        // the history entry of record, cut in the middle of a value
        let (out, note) = args_object_or_raw(r#"{"path":"/etc/host"#);
        let doc: serde_json::Value = serde_json::from_str(&out).expect("the repair is JSON");
        assert_eq!(doc[RAW_KEY], r#"{"path":"/etc/host"#);
        let note = note.expect("a broken string must produce a note");
        assert!(note.contains("serde_json:"), "the decoder's message is missing: {note}");
        assert!(note.contains("EOF while parsing"), "the decoder's reason is missing: {note}");
        assert!(note.contains("<HERE>"), "the byte window is missing: {note}");

        // the same shape with an HTML content parameter, cut mid-escape
        let cut = r#"{"path":"/home/x/sheet.html","content":"<!doctype html>\n<html lang=\"en\">\"#;
        let (_, note) = args_object_or_raw(cut);
        let note = note.expect("a cut escape must produce a note");
        assert!(note.contains(&format!("of {}", cut.len())), "the window must name the total length: {note}");
        assert!(note.contains("<HERE>"), "the window must mark the cut: {note}");

        // a string that parses but is not a mapping: the template cannot iterate it either
        let (out, note) = args_object_or_raw("[1,2]");
        let doc: serde_json::Value = serde_json::from_str(&out).expect("the repair is JSON");
        assert_eq!(doc[RAW_KEY], "[1,2]");
        assert!(note.expect("a note").contains("it parses, as an array"));
    }

    /// the window is a byte range around the failure, always on char boundaries
    #[test]
    fn the_error_window_points_at_the_failing_byte() {
        let s = "{\"a\":\"\u{e9}\u{1f426}\", \"b\" 1}";
        let e = serde_json::from_str::<serde_json::Value>(s).expect_err("this is not JSON");
        let w = error_window(s, &e);
        assert!(w.contains("<HERE>"), "no marker: {w}");
        assert!(w.contains(&format!("of {}", s.len())), "no total length: {w}");
        // the marked byte is where serde stopped: the window's left half ends with what
        // came before it, so the defect itself is the first thing after `<HERE>`
        assert!(w.contains(r#"<HERE> "1}""#), "the window must start at the offending byte: {w}");
    }

    /// #54: the pure decision table of the gone-client probe, plus the `revents` bits it
    /// reads. Every line of `gone_reason` and every branch of `peer_from_poll` is here, and
    /// on Linux the four constants are asserted against the kernel's own values - they are
    /// written out so the decision compiles and is tested on a platform without `poll`.
    #[test]
    fn the_probe_reads_the_kernels_own_revents_bits() {
        #[cfg(target_os = "linux")]
        {
            assert_eq!(PROBE_ERR, libc::POLLERR, "POLLERR");
            assert_eq!(PROBE_HUP, libc::POLLHUP, "POLLHUP");
            assert_eq!(PROBE_NVAL, libc::POLLNVAL, "POLLNVAL");
            assert_eq!(PROBE_RDHUP, libc::POLLRDHUP, "POLLRDHUP");
        }
        // nothing set, and the two non-results of the syscall itself: a timeout (0) and an
        // interrupted call (-1, EINTR). A signal may not end a generation.
        assert_eq!(peer_from_poll(0, 0), Peer::Open);
        assert_eq!(peer_from_poll(-1, 0), Peer::Open);
        assert_eq!(peer_from_poll(-1, PROBE_RDHUP), Peer::Open);
        // unread request bytes are not a report about the peer's read side
        assert_eq!(peer_from_poll(1, 0x001), Peer::Open);
        // EOF, and a reset; a reset outranks the EOF that comes with it
        assert_eq!(peer_from_poll(1, PROBE_RDHUP), Peer::Eof);
        assert_eq!(peer_from_poll(1, 0x001 | PROBE_RDHUP), Peer::Eof);
        assert_eq!(peer_from_poll(1, PROBE_ERR), Peer::Hup);
        assert_eq!(peer_from_poll(1, PROBE_HUP), Peer::Hup);
        assert_eq!(peer_from_poll(1, PROBE_NVAL), Peer::Hup);
        assert_eq!(peer_from_poll(1, 0x001 | PROBE_ERR | PROBE_HUP | PROBE_RDHUP), Peer::Hup);

        // the decision: a reset is gone whatever the baseline was
        for base in [false, true] {
            let why = gone_reason(base, Peer::Hup).expect("a reset is always gone");
            assert!(why.contains("reset"), "{why}");
            assert_eq!(gone_reason(base, Peer::Open), None);
        }
        // an EOF that appears during the generation is gone, and the reason names the bit
        let why = gone_reason(false, Peer::Eof).expect("an EOF that was not there before is gone");
        assert!(why.contains("POLLRDHUP"), "{why}");
        assert!(why.contains("mid-generation"), "{why}");
        // an EOF that was ALREADY there is a client that finished sending: it gets its answer
        assert_eq!(gone_reason(true, Peer::Eof), None);
    }

    /// #54: the cadence. `PROBE_EVERY` is 1 - every step - because one probe is 0.10 us
    /// against a ~27 ms token; the rule is written for any k so raising the constant needs
    /// no second reading of it, and k = 0 is the off switch.
    #[test]
    fn the_probe_cadence_is_every_k_steps() {
        assert_eq!(PROBE_EVERY, 1, "the constant of record");
        // every step, which is what this server runs
        assert!((0..8).all(|i| probe_due(i, PROBE_EVERY)));
        // every eighth step, first the one that opens the generation
        assert_eq!(
            (0..17).filter(|&i| probe_due(i, 8)).collect::<Vec<_>>(),
            vec![0, 8, 16]
        );
        assert_eq!((0..7).filter(|&i| probe_due(i, 4)).collect::<Vec<_>>(), vec![0, 4]);
        // k = 0 probes never; a 512-step budget with k = 8 costs 64 probes, that is 6.4 us
        assert!(!probe_due(0, 0) && !probe_due(7, 0));
        assert_eq!((0..512).filter(|&i| probe_due(i, 8)).count(), 64);
    }

    /// #54: the probe against a REAL loopback socket, the three shapes that matter. No
    /// engine, no GPU: a `TcpListener` on port 0 and one client thread.
    ///
    /// The point of the third case is the one the design turns on: a client that closed the
    /// connection and a client that only shut its WRITE side down are the same wire event,
    /// so the probe reports `Eof` for both and only the baseline tells them apart.
    #[test]
    fn the_probe_sees_a_closed_peer_and_not_a_live_one() {
        use std::sync::mpsc;

        // wait for the FIN to arrive: on the loopback it is there at once, but a bounded
        // retry keeps this test off the scheduler's mercy
        fn wait_eof(fd: Option<i32>) -> Peer {
            for _ in 0..200 {
                let p = fd.map(poll_peer).unwrap_or(Peer::Open);
                if p != Peer::Open {
                    return p;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            Peer::Open
        }

        let l = TcpListener::bind("127.0.0.1:0").expect("bind");
        let port = l.local_addr().expect("addr").port();
        let (tx, rx) = mpsc::channel::<&'static str>();
        let client = std::thread::spawn(move || {
            let mut c = TcpStream::connect(("127.0.0.1", port)).expect("connect");
            c.write_all(b"POST /v1/chat/completions\r\n\r\n").expect("write");
            // 1) still there
            assert_eq!(rx.recv().expect("step 1"), "half-close");
            // 2) the write side only, and the read side kept open, waiting for the document
            c.shutdown(Shutdown::Write).expect("shutdown write");
            assert_eq!(rx.recv().expect("step 2"), "close");
            // 3) gone
            drop(c);
        });
        let (srv, _) = l.accept().expect("accept");
        let fd = socket_fd(&srv);

        // 1) a live client with unread bytes in the socket is OPEN, and a probe at any step
        //    of any cadence says so
        let mut probe = ClientProbe::new(fd);
        assert!(!probe.base_eof, "a live client is not an EOF baseline");
        for step in 0..4 {
            assert_eq!(probe.gone(step), None, "step {step} of a live client");
        }

        // 2) the half-closed client: the probe reports EOF, and the decision keeps generating
        //    only because the BASELINE of that request saw the same EOF
        tx.send("half-close").expect("tell the client");
        let p = wait_eof(fd);
        assert_eq!(p, Peer::Eof, "a shutdown(Write) client must read as an EOF");
        assert!(probe.gone(0).is_some(), "an EOF that was not there at the baseline is gone");
        let mut half = ClientProbe::new(fd);
        assert!(half.base_eof, "the baseline of a request that starts half-closed");
        for step in 0..4 {
            assert_eq!(half.gone(step), None, "a half-closed client still gets its document");
        }

        // 3) the real thing: the client closes the whole connection. The wire event is the
        //    one of case 2, which is why the baseline is the whole decision.
        tx.send("close").expect("tell the client");
        client.join().expect("the client thread");
        assert_eq!(wait_eof(fd), Peer::Eof, "a closed peer reads as an EOF too");
        let mut gone = ClientProbe::new(fd);
        assert!(gone.base_eof, "and it is indistinguishable at the baseline");
        assert_eq!(gone.gone(0), None, "which is exactly why the baseline latches");
        // the request that was already running is the case #54 is about, and it stops
        let why = probe.gone(1).expect("the client of THIS request is gone");
        assert!(why.contains("POLLRDHUP"), "{why}");
    }

    /// #54: the sink side. A `CollectSink` with no socket behind it - the shape every other
    /// test of this file builds - never reports a gone client, so nothing about the document
    /// path changed for a client that stays. The stream sink answers `true` by the trait's
    /// default: its detector is the failed flush, and that is unchanged.
    #[test]
    fn a_document_sink_without_a_socket_never_stops_the_loop() {
        let mut col = CollectSink::default();
        assert_eq!(col.probe, ClientProbe::default());
        for step in 0..512 {
            assert!(col.still_there(step), "step {step}");
        }
        let mut buf: Vec<u8> = Vec::new();
        let mut sse = SseSink::new(&mut buf);
        for step in 0..8 {
            assert!(sse.still_there(step), "the stream sink keeps the default at step {step}");
        }
        assert!(buf.is_empty(), "the probe may not put a byte on the wire");
    }

    /// the accumulator is Crow's own: one string per call index, in fragment order
    #[test]
    fn the_arguments_accumulator_is_per_call_index() {
        let mut acc = Vec::new();
        accumulate_args(&[
            Emit::Call { index: 0, id: "call_0".into(), name: "a".into() },
            Emit::Args { index: 0, text: "{\"p\":".into() },
            Emit::Content("ignored".into()),
        ], &mut acc);
        accumulate_args(&[
            Emit::Args { index: 1, text: "{\"q\":2}".into() },
            Emit::Args { index: 0, text: "1}".into() },
        ], &mut acc);
        assert_eq!(acc, vec!["{\"p\":1}".to_string(), "{\"q\":2}".to_string()]);
        for a in &acc {
            assert!(serde_json::from_str::<serde_json::Value>(a).unwrap().is_object());
        }
    }
}
