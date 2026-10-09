# GLM-5.3-Flash: tokenizer, chat template and tool-call markup

Step 12 of the GLM-5.3-Flash plan (crow-nest #160, parent Crow #362). It makes the prompt side
of the engine able to serve GLM-5.3-Flash: the tokenizer and the chat template byte-identical
to transformers, the reasoning words of this template, and GLM's tool-call markup as a parser
and as a grammar. The Qwen family (Flash-Next, the 27B) keeps its bytes: every existing
tokenizer, parser, grammar and `serve` test is unchanged and green.

- `engine/src/glm5_template.rs` — the family test, the reasoning words, the stop ids, the goldens replay.
- `engine/src/tokenizer.rs` — template loading, the `tojson` keywords, `render_chat_with`, `resolve_tokenizer`.
- `engine/src/toolcall.rs` — `Markup::Glm` in `ToolStream`.
- `engine/src/toolgrammar.rs` — the GLM frame (`ToolGrammar::build_markup`).
- `oracle/export_glm5_tokenizer_goldens.py` → `engine/tests/fixtures/GLM-5.3-Flash/tokenizer-goldens.json`.

#185 part 1 wired the request side into `bin/serve.rs`; see [Wired into serve](#wired-into-serve-185-part-1).
The glm5_next engine behind it (`glm5_engine::Glm5Engine`) still refuses to boot until the expert tiers
of #175/#149 exist (#185 part 2).

## Inputs

`zai-org/GLM-5.3-Flash`, rev `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, in
`models/GLM-5.3-Flash-original/` (size and sha256 checked against the `.verified` records; the
golden script refuses a file that differs).

| file | bytes | sha256 |
|---|---|---|
| `tokenizer.json` | 20,217,442 | `19e773648cb4e65de8660ea6365e10acca112d42a854923df93db4a6f333a82d` |
| `tokenizer_config.json` | 761 | `98b1271574f41abf89427ae2dda030d94dc9478f0edc5a8bd240db213c6fd5fc` |
| `chat_template.jinja` | 10,950 | `0c4099f3382d6c92700dfb99725025360966fd73032f0ecf32377c0d9e6309c5` |

`tokenizer_config.json` has no `chat_template` field. `ChatTokenizer::load` now reads
`chat_template.jinja` beside the config when it exists, and the config field otherwise. This is
transformers 5.16.1's order: it reads the file and replaces the field with it
(`tokenization_utils_base.py:1785-1798`). Both Qwen directories carry both sources, byte-equal
(8,952 B each, compared 2026-10-08), so their renders do not move.

`resolve_tokenizer` also tries `models/<model>-original/tokenizer.json`, the directory the
original download lands in. A container whose model is not a Qwen model (`Qwen*`) and has no
tokenizer of its own does NOT fall back to Flash-Next's tokenizer any more. It gets its own
missing path back, and the load refuses it by name (`cannot load models/GLM-5.3-Flash/tokenizer.json`).

## Token ids

All of them are added tokens.

| token | id | special | role |
|---|---|---|---|
| `<\|endoftext\|>` `<\|user\|>` `<\|observation\|>` | 154820 154827 154829 | yes | the three `eos_token_id` of `generation_config.json` |
| `[gMASK]` `<sop>` | 154822 154824 | yes | first two ids of every prompt |
| `<\|system\|>` `<\|assistant\|>` | 154826 154828 | yes | role headers |
| `<think>` `</think>` | 154841 154842 | no | reasoning block |
| `<tool_call>` `</tool_call>` | 154843 154844 | no | call frame; `<tool_call>` arms parser and grammar by id |
| `<arg_key>` `</arg_key>` `<arg_value>` `</arg_value>` | 154847–154850 | no | argument frame |
| `<tool_response>` `</tool_response>` | 154845 154846 | no | tool results |

The six tool markers are not special. So `decode(.., skip_special_tokens=true)` keeps them, and
the grammar's vocabulary trie carries them as text. The parser and the grammar see the bytes the
model writes. `vocab_size` in `config.json` is 154,880; ids 154,856 and up have no bytes.

## The template

| input | rendered |
|---|---|
| always | `[gMASK]<sop>` |
| `reasoning_effort` `low` / `high` | `<\|system\|>Reasoning Effort: Low` / `High` |
| `reasoning_effort` undefined or any other value | `<\|system\|>Reasoning Effort: Max` |
| `clear_thinking` (default false) | false: every past assistant turn keeps its `<think>..</think>`; true: only turns after the last user message |
| `add_generation_prompt` | `<\|assistant\|><think>`, with no newline |
| `enable_thinking` | not read |
| a past assistant turn without reasoning | `<\|assistant\|><think></think>` |
| tool calls | `<tool_call>NAME<arg_key>K</arg_key><arg_value>V</arg_value>…</tool_call>`, tight; a string `V` raw, any other `V` as `tojson(ensure_ascii=False)` |
| tool results | one `<\|observation\|>`, then `<tool_response>…</tool_response>` per result, sorted into the order of the calls when every id matches |

**"Off" cannot be expressed.** No variable closes the think block of the generation prompt, and
`effective_reasoning_effort` is never none. Every one of the 15 golden renders with a generation
prompt ends in `<|assistant|><think>` (ids 154828 154841). So does a render with
`enable_thinking: false`, and so does one with `reasoning_effort: "none"`. Closing the block after
the render would be string surgery, which #25 rules out.

### Reasoning words (`glm5_template::reasoning_level`)

The same system and user message, with a generation prompt. The sha256 is over the UTF-8 render.
The engine's render has the same hash.

| request `reasoning_effort` | template variable | renders | sha256 |
|---|---|---|---|
| absent | undefined | Max | `a28b9e93db2ccc8fb3c5cb1095cdef88ded6bd38d26a10582b1bc99d48b885b1` |
| `max` | undefined | Max | `a28b9e93…b885b1` |
| `low` | `low` | Low | `05a7e44d1453c37c60293d62f707356e91a558edde0645972695b8a10836a218` |
| `high`, `xhigh` | `high` | High | `7c349ba45c32be4266bacc4ed862f64b090f6869af934353f7fa326295815e03` |
| `medium` | refused by name | (the template would render Max) | `a28b9e93…b885b1` |
| `none` | refused by name | (off cannot be expressed) | `a28b9e93…b885b1` |
| anything else | refused by name | | |

Qwen's mapping in `serve.rs` (`none` → off, `medium` passes, `high` → `xhigh`) stays as it is.

## Jinja parity

The engine renders with minijinja in the environment of `tokenizer.rs`. #160 added two things:

- `tojson` takes keywords as transformers' own filter does (`chat_template_utils.py:481-484`):
  - `ensure_ascii` (False: raw; True: `\uXXXX` in lower-case hex, surrogate pairs above U+FFFF);
  - `indent=None`.
- `indent` with any other value, `separators`, `sort_keys` and unknown keywords are errors that name the
  keyword. Before #160 any keyword was a render error ("too many arguments"), so every GLM render with
  tools failed.
- `render_chat_with(messages, tools, add_generation_prompt, vars)` passes arbitrary template variables
  (`reasoning_effort`, `clear_thinking`), as `apply_chat_template(**kwargs)` does.

Each construct the GLM template uses is checked alone against transformers' render of the same
source (`jinja_features` in the goldens, 16 cases):

- `capitalize`
- `namespace` with `loop.index0`
- `range` with `break`
- `loop.first`
- `split()[0]` and `[-1]`
- `strip`, `rstrip('\n')`, `lstrip`
- `.0` on a list and on a string, and `is defined` after it
- `items()`, `in` on a mapping, `tojson(ensure_ascii=…)`
- `set` inside `if` inside `for`: a new scope per iteration, so the reasoning of one turn does not leak into the next
- a macro's output compared with `==`
- the type tests
- `~` and `length`
- the conditional expression
- `trim_blocks` / `lstrip_blocks`

Known difference, not used by any template of record: a boolean printed raw is `True` in jinja2
and `true` in minijinja. The feature cases avoid it on purpose.

## Tool-call markup

### Parser (`ToolStream::with_markup(tools, Markup::Glm)`)

`ToolStream::new` stays Qwen. The GLM states sit in the same machine and share its fragments,
`arm()`, abandon path, `Malformed` records and `dropped` count:

| state | reads up to | on | then |
|---|---|---|---|
| `GName` | `<arg_key>` or `</tool_call>` | `<arg_key>` | `Emit::Call`, then `GKey` |
| | | `</tool_call>` | `Emit::Call` and `{}` (a call with no arguments) |
| `GKey` | `</arg_key>` | | `GAwaitVal` |
| `GAwaitVal` | `<arg_value>` | `<arg_value>` | `{"k":` (plus `"` for a declared string), then `GVal` |
| | | `</tool_call>` | malformed `bad-param-name` (a key without a value) |
| `GVal` | the FIRST `</arg_value>` | | the value is closed, then `GAfter` |
| `GAfter` | `<arg_key>` or `</tool_call>` | | `GKey`, or `}` and the call closes |

- A string value is kept verbatim, newlines included. Qwen's separator newlines do not exist here.
- A value of any other declared type is trimmed and parsed as JSON. It falls back to a string with
  the existing warning.
- An undeclared parameter is JSON when it parses, else a string. This is Qwen's rule, and it matches
  the template: non-strings are rendered as `tojson`.
- Names and keys are trimmed and must not contain `<`, `>` or a line break, else `bad-name` /
  `bad-param-name`.
- GLM has no `</function>`. Only `</tool_call>` completes a call, so the end of generation anywhere
  inside a call is `end-in-call`, with `_truncated` arguments.
- llama.cpp's GLM-4.7-Flash parser cases (`tests/test-chat.cpp:4258-4370` at `11924d4c1`, read, not
  run) give the same result here:
  - one call;
  - two parallel calls;
  - `{"arg1": 1}` from `<arg_value>1</arg_value>` for an integer schema;
  - `<tool_call>name</tool_call>` → `{}`.

**A value that contains argument markup** (the plan's failure mode):

- `<arg_value>`, `<arg_key>` and `</arg_key>` inside a value are value bytes. A golden renders
  `"Krähe <arg_value> und </arg_key> bleiben Text"` in Python, and the parser gives the same
  string back.
- A value ends at the first `</arg_value>`. Without the grammar, the rest of what the model meant
  as value is skipped up to the next marker, and the call still closes as JSON.
- With the grammar, `</arg_value>` cannot be written inside a string value.

### Grammar (`ToolGrammar::build_markup(.., Markup::Glm)`)

`ToolGrammar::build` stays Qwen. The GLM frame is

```text
<tool_call> NAME (<arg_key> P </arg_key><arg_value> VALUE </arg_value>)* </tool_call>
```

- It keeps the schema rules of the Qwen grammar:
  - NAME is a declared tool;
  - each P is a declared parameter, written once;
  - VALUE follows llama.cpp's `resolves_to_string` rules (raw string, raw enum, typed JSON).
- The frame has no newlines, because the template renders it tight.
- A string VALUE may not contain `</arg_value>` or `</tool_call>`.
- A JSON VALUE may be followed by `[ \t]*`.
- After `</tool_call>` come the existing `Between` and EOS rules. EOS is any of the three stop ids,
  and `<|observation|>` ends a tool turn.
- NAME and P may not contain `<`; `build_markup` refuses such a tool by name.
- The guard (`</arg_value>` / `</tool_call>`) has the same length and the same `</` prefix as
  Qwen's, so `guard_step` serves both.

## Goldens

```text
.venv-oracle/Scripts/python.exe -I oracle/export_glm5_tokenizer_goldens.py
```

The oracle is transformers 5.16.1, tokenizers 0.23.1, jinja2 3.1.6 and CPython 3.13.3. The golden
file records every version and the sha256 of every input. It holds:

- **10 encode cases**: German (umlauts, ß, ẞ), code, emoji (ZWJ and skin tone), Chinese, the role
  tokens with `[gMASK]<sop>`, a GLM tool transcript, marker text with spaces, whitespace edges,
  numbers, and the empty string. Each case has the ids and both decodes.
- **16 render cases**: the reasoning words; tools (`strict` and `defer_loading` keys); one assistant
  call with string, number, bool, list and object arguments; two parallel calls with two
  `<tool_response>` blocks re-sorted by id; tool results without ids; reasoning before and after
  the last user message; `clear_thinking: true`; inline `<think>` in the content followed by a turn
  without reasoning; content parts with an image. Each case has the render, its sha256 and its ids.
- **16 Jinja feature cases**.

## Tests

```text
cd engine
cargo test --lib -- tokenizer:: toolcall:: toolgrammar:: glm5_template:: stopstr::
cargo test --bin serve
```

Results on 2026-10-08, Windows, CPU only:

- the lib modules: 94 passed. 66 of these existed before; 28 are new: `tokenizer` 5, `toolcall::glm_tests` 11,
  `toolgrammar::glm_tests` 7, `glm5_template` 5.
- `serve`: 120 passed, unchanged.

The GLM tokenizer tests skip with a line when `models/GLM-5.3-Flash-original/` is missing. The
parser and grammar tests do not need the files.

## Wired into serve (#185 part 1)

The five items #160 left for `bin/serve.rs` are wired (2026-10-09). `serve` chooses the request
family once at boot and every GLM request follows it:

1. **Family.** `serve` reads the container's family before the CUDA context (`container_meta`,
   `engine_kind`): Flash-Next and the 27B boot `Engine` as before, glm5_next boots
   `glm5_engine::Glm5Engine`. The request family is `glm5_template::markup_of(tk)` on the loaded
   tokenizer and must equal the engine's (`EngineKind::markup`), else the boot is refused by name.
   It rides on every request as `ChatReq::markup`.
2. **Render.** A GLM request renders with
   `tk.encode_chat_with(messages, tools, true, &glm5_template::template_vars(level, clear_thinking))`
   (`render_ids`). `level` is `glm5_template::reasoning_level` of the request's `reasoning_effort`
   (top level, else `chat_template_kwargs`); `clear_thinking` is `chat_template_kwargs.clear_thinking`
   (a boolean, else a 400). 400s, by name and before any render: `none`, `medium` and every unknown
   word, a non-string `reasoning_effort` (naming `low, high, xhigh, max`), and
   `chat_template_kwargs.enable_thinking: false`, because off cannot be expressed. `enable_thinking: true`
   is accepted; it is what renders anyway.
   - The #67 strip of a stored `<think>…</think>` in an assistant `content` is the Qwen template's
     repair and is NOT applied to GLM: GLM's template splits that block itself (line 146) and
     renders it as the turn's reasoning. The `arguments` repairs apply to both families.
3. **Tools.** `ToolStream::with_markup(tools, req.markup)` (`tool_stream`) and
   `ToolGrammar::build_markup(.., req.markup)` (`tool_gate`). `<tool_call>` arms both by its id
   (154843), found by text as for Qwen.
4. **Reasoning filter.** A GLM request has `enable_thinking` true, so `ThinkFilter::for_request`
   starts `Inside`. The `[chat]` line names the rendered level (`low`, `high`, `max`), never `off`.
5. **Stop ids.** The loop and the tool trie take their stop ids from the engine through the
   `ServeEngine` seam: `Geo::eos_ids` for Flash-Next and the 27B, `glm5_template::EOS_IDS` for
   `Glm5Engine`. `sample::EOS_IDS` keeps its type and value.

Tests (`cargo test --release --bin serve`, CPU only, 2026-10-09; the GLM ones need
`models/GLM-5.3-Flash-original/`):

- `a_glm_request_renders_the_golden_ids_of_160`: 12 golden renders (5 with tools) go through
  serve's parse, normaliser, message check and `render_ids` and give transformers' ids; the three
  goldens the template cannot honour (`medium`, `none`, `enable_thinking: false`) are 400s.
- `glm_thinking_off_and_unknown_levels_are_400s_by_name`, and the same bodies on the Qwen family
  keep their meaning of record.
- `glm_tool_markup_streams_as_openai_tool_calls_deltas`: reasoning, `</think>` and one GLM call
  (real tokenizer ids) through `tool_stream`, `admit_id`, the think filter, `send_emits` and the SSE
  sink: one `tool_calls` entry `read_file` with arguments `{"path":"README.md"}`, the reasoning as
  `reasoning_content`, no content, the GLM grammar refuses no id, `<|observation|>` stops.
- `the_dispatch_keeps_flash_next_and_the_27b_on_their_path`.

What part 2 adds: the `Glm5Engine` body on the generator of #175/#149 (boot at 200k with the #159
plan, prefill, decode step, logits, snapshot / restore, reset), the request loop on it, and the
prefix-cache snapshots (KDA recurrent + conv states; MLA latent and DSA indexer rows truncate).
Not decided here: GLM's sampling card row (a GLM request takes `card_row(true)`, whose
temperature 1.0 and top_p 0.95 equal GLM's `generation_config.json`; its top_k 20 is Qwen's card).

## Known limitations

- `tojson(indent=n)` for n > none is not implemented. It is an error by name; no template of record uses it.
- A boolean printed raw renders `true`, where jinja2 renders `True`. The GLM template prints none.
- Not measured live: no GLM model is booted (step 14, #185 part 2). Robin's live check: in Crow, with tools on,
  ask for a file read. The GUI should show one `read_file` call with the right `path`, and the turn
  continues after the tool result.
