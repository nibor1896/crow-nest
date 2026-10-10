# Export the GLM-5.3-Flash tokenizer and chat-template goldens - crow-nest #160 (plan step 12).
#
# The oracle of record is transformers 5.16.1 in .venv-oracle, the same reference
# `tools/tokenize_ids.py` is for the Qwen family (`engine/src/tokenizer.rs:3-27`):
#
#   encode  tok(text, add_special_tokens=False)["input_ids"]
#   decode  tok.decode(ids, skip_special_tokens=True)  and  =False
#   render  tok.apply_chat_template(messages, tools=..., add_generation_prompt=...,
#                                   tokenize=False, **template_vars)
#   ids     the same call with tokenize=True; asserted equal to encode(render)
#
# Inputs: models/GLM-5.3-Flash-original/{tokenizer.json, tokenizer_config.json,
# chat_template.jinja} of zai-org/GLM-5.3-Flash rev eb9eb208eb0d988989d07a6a12d0fdeb5f52574a
# (size + sha256 checked against the `.verified` records the download wrote; the
# script refuses a file whose hash differs). transformers reads chat_template.jinja
# beside the config (tokenization_utils_base.py:1785-1798); the config has no template.
#
# Output: engine/tests/fixtures/GLM-5.3-Flash/tokenizer-goldens.json - versions, input
# hashes, the encode cases, the render cases (render text, its sha256, its ids). The
# engine test `glm5_template::tests` replays every case through the Rust tokenizer.
#
# Run (reads downloaded files, so isolated mode):
#   .venv-oracle/Scripts/python.exe -I oracle/export_glm5_tokenizer_goldens.py
# CPU only, a few seconds. Deterministic: no clock, no randomness in the output.

import hashlib
import json
import os
import sys

import jinja2
import tokenizers
import transformers
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(ROOT, "models", "GLM-5.3-Flash-original")
OUT = os.path.join(ROOT, "engine", "tests", "fixtures", "GLM-5.3-Flash", "tokenizer-goldens.json")
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
FILES = ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja"]


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def inputs():
    out = {}
    for f in FILES:
        p = os.path.join(MODEL, f)
        rec = json.load(open(p + ".verified", encoding="utf-8"))
        got = sha256_file(p)
        size = os.path.getsize(p)
        if rec["sha256"] != got or rec["size"] != size or rec["revision"] != REVISION:
            sys.exit(f"{f}: sha256/size/revision differ from {f}.verified - refusing")
        out[f] = {"size": size, "sha256": got}
    return out


# ------------------------------------------------------------------ encode cases
# German (umlauts, sharp s, capital sharp s), code, emoji incl. a ZWJ sequence and a
# skin tone, Chinese, the role tokens with [gMASK]<sop>, a tool transcript in GLM's
# markup, the literal marker texts with spaces around them, whitespace edges, numbers.
ENCODE = [
    ("german", "Gr\u00fc\u00dfe aus M\u00fcnchen: \u00c4pfel, \u00d6l, \u00dcberma\u00df \u2013 Stra\u00dfe \u1e9e, \u00e4\u00f6\u00fc\u00df."),
    ("code", "fn main() {\n    let v: Vec<u32> = (1..=3).collect();\n    println!(\"{:?}\", v);\n}\n"),
    ("emoji", "Adler \U0001F985 und Kr\u00e4he \U0001F426\u200d\u2b1b \U0001F44D\U0001F3FD!"),
    ("chinese", "\u4f60\u597d\uff0c\u4e16\u754c\u3002\u5de5\u5177\u8c03\u7528"),
    ("roles", "[gMASK]<sop><|system|>Sys<|user|>Hallo<|assistant|><think>kurz</think>Antwort<|observation|><tool_response>ok</tool_response><|endoftext|>"),
    ("tool_transcript", "<tool_call>read_file<arg_key>path</arg_key><arg_value>C:/Users/robin/dev/Gr\u00fc\u00dfe.md</arg_value><arg_key>start_line</arg_key><arg_value>1</arg_value></tool_call>"),
    ("marker_text", "a <arg_value> b </arg_value> c</arg_key>d <tool_call >"),
    ("whitespace", "  leading\n\n\ttabs   trailing  \r\n\r\n end"),
    ("numbers", "1234567 3.14159 -42 0x1F 1e-9"),
    ("empty", ""),
]

# ------------------------------------------------------------------ render cases
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Liest eine UTF-8-Datei (Gr\u00f6\u00dfe egal).",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Pfad zur Datei."},
                    "start_line": {"type": "integer", "description": "Erste Zeile, ab 1."},
                },
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "configure",
            "description": "All value types.",
            "parameters": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "zoll": {"type": "number"},
                    "aktiv": {"type": "boolean"},
                    "tags": {"type": "array", "items": {"type": "string"}},
                    "opts": {"type": "object", "properties": {"tiefe": {"type": "integer"}}},
                },
                "required": ["name"],
                "additionalProperties": False,
            },
            "strict": True,
        },
    },
    {
        "type": "function",
        "function": {"name": "hidden_tool", "description": "deferred", "parameters": {"type": "object"}, "defer_loading": True},
    },
]

SYS = {"role": "system", "content": "Du bist ein hilfreicher Assistent."}
USER = {"role": "user", "content": "Wie gro\u00df ist der Monitor? \U0001F985"}

CALL_ALL_TYPES = {
    "id": "call_0",
    "type": "function",
    "function": {
        "name": "configure",
        "arguments": {
            "name": "Kr\u00e4he <arg_value> und </arg_key> bleiben Text",
            "zoll": 27.5,
            "aktiv": True,
            "tags": ["a", "\u00fc"],
            "opts": {"tiefe": 2},
        },
    },
}
CALL_A = {"id": "call_a", "type": "function", "function": {"name": "read_file", "arguments": {"path": "a.md"}}}
CALL_B = {"id": "call_b", "type": "function", "function": {"name": "read_file", "arguments": {"path": "b.md", "start_line": 3}}}

RENDER = [
    # reasoning levels: undefined -> max, low, high, anything else -> max
    ("sys_user_default", [SYS, USER], None, True, {}),
    ("sys_user_low", [SYS, USER], None, True, {"reasoning_effort": "low"}),
    ("sys_user_high", [SYS, USER], None, True, {"reasoning_effort": "high"}),
    ("sys_user_medium", [SYS, USER], None, True, {"reasoning_effort": "medium"}),
    ("sys_user_none", [SYS, USER], None, True, {"reasoning_effort": "none"}),
    ("sys_user_max", [SYS, USER], None, True, {"reasoning_effort": "max"}),
    # the variable GLM-4.7 had; this template does not read it (asserted equal to default)
    ("sys_user_enable_thinking_false", [SYS, USER], None, True, {"enable_thinking": False}),
    ("user_no_generation_prompt", [USER], None, False, {}),
    ("user_with_tools", [USER], TOOLS, True, {}),
    ("assistant_call_all_types", [USER, {"role": "assistant", "content": "", "tool_calls": [CALL_ALL_TYPES]}], TOOLS, True, {}),
    (
        "two_parallel_calls_two_responses",
        [
            USER,
            {"role": "assistant", "content": "Ich lese beide.", "tool_calls": [CALL_A, CALL_B]},
            {"role": "tool", "tool_call_id": "call_b", "content": "B-Inhalt"},
            {"role": "tool", "tool_call_id": "call_a", "content": "A-Inhalt \u00e4"},
        ],
        TOOLS,
        True,
        {},
    ),
    (
        "tool_responses_without_ids",
        [
            USER,
            {"role": "assistant", "content": "", "tool_calls": [CALL_A, CALL_B]},
            {"role": "tool", "content": "eins"},
            {"role": "tool", "content": "zwei"},
        ],
        TOOLS,
        True,
        {},
    ),
    (
        "reasoning_before_and_after_last_user",
        [
            SYS,
            {"role": "user", "content": "Erste Frage"},
            {"role": "assistant", "reasoning_content": "alt\nzwei Zeilen", "content": "Erste Antwort"},
            {"role": "user", "content": "Zweite Frage"},
            {"role": "assistant", "reasoning_content": "neu", "content": "", "tool_calls": [CALL_A]},
            {"role": "tool", "tool_call_id": "call_a", "content": "ok"},
        ],
        TOOLS,
        True,
        {},
    ),
    (
        "reasoning_clear_thinking_true",
        [
            {"role": "user", "content": "Erste Frage"},
            {"role": "assistant", "reasoning_content": "alt", "content": "Erste Antwort"},
            {"role": "user", "content": "Zweite Frage"},
            {"role": "assistant", "reasoning_content": "neu", "content": "Zweite Antwort"},
        ],
        None,
        True,
        {"clear_thinking": True},
    ),
    (
        "inline_think_then_turn_without_reasoning",
        [
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "<think>innen</think>\n\nsichtbar"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "nur Text"},
            {"role": "user", "content": "q3"},
        ],
        None,
        True,
        {"reasoning_effort": "high"},
    ),
    (
        "content_parts",
        [
            {"role": "user", "content": [{"type": "text", "text": "Was ist "}, {"type": "image_url", "image_url": {"url": "x"}}, {"type": "text", "text": "das?"}]},
        ],
        None,
        True,
        {"reasoning_effort": "low"},
    ),
]


# ------------------------------------------------------------------ jinja features
# Every construct GLM-5.3-Flash's template uses, alone, rendered by transformers' own
# environment (`_compile_jinja_template`, chat_template_utils.py:420-495). The engine test
# `tokenizer::tests::glm_jinja_features_match_transformers` renders the same source in the
# minijinja environment of `tokenizer.rs`. A boolean is never printed raw here: jinja2
# prints `True`, minijinja `true`, and no template of record prints one.
FEATURES = [
    ("capitalize", "{{ 'max' | capitalize }}|{{ 'low' | capitalize }}|{{ 'hIGH wORLD' | capitalize }}", {}),
    ("namespace_loop_index0", "{% set ns = namespace(last=-1) %}{% for m in xs %}{% if m == 'u' %}{% set ns.last = loop.index0 %}{% endif %}{% endfor %}{{ ns.last }}", {"xs": ["s", "u", "a", "u", "t"]}),
    ("range_break", "{% for i in range(2, 9) %}{% if i > 4 %}{% break %}{% endif %}{{ i }},{% endfor %}", {}),
    ("loop_first", "{% for m in xs %}{% if loop.first or xs[loop.index0 - 1] != m %}[{% endif %}{{ m }}{% endfor %}", {"xs": ["t", "t", "a", "t"]}),
    ("split_index", "{{ c.split('</think>')[0].split('<think>')[-1] }}|{{ c.split('</think>')[-1] }}", {"c": "<think>r\n</think>\n\nx"}),
    ("strip_family", "[{{ s.strip() }}][{{ s.rstrip('\\n') }}][{{ s.lstrip('\\n') }}][{{ s.strip() | length }}]", {"s": "\n a \n"}),
    ("dot_zero", "{{ l.0.output }}|{{ s.0 }}|{% if s.0.output is defined %}D{% else %}U{% endif %}|{% if l.0.output is defined %}D{% else %}U{% endif %}|{% if e and e.0.output is defined %}D{% else %}U{% endif %}", {"l": [{"output": "o"}], "s": "Hallo", "e": []}),
    ("items_in_tojson", "{% for k, v in d.items() %}{% if k != 'strict' %}\"{{ k }}\": {{ v | tojson(ensure_ascii=False) }};{% endif %}{% endfor %}{% if 'name' in d %}N{% endif %}{% if 'function' in d %}F{% endif %}", {"d": {"name": "größe", "strict": True, "n": [1, "ü", None, 2.5, {"a": False}]}}),
    ("tojson_ensure_ascii_true", "{{ v | tojson(ensure_ascii=True) }}", {"v": {"k": "Grüße \U0001F985", "n": [1]}}),
    ("tojson_indent_none", "{{ v | tojson(indent=None) }}|{{ v | tojson }}", {"v": {"k": "ü"}}),
    ("set_in_if_in_for", "{% for m in xs %}{% if m == 1 %}{% set r = 'x' %}{% endif %}{% if r is defined %}D{% else %}U{% endif %};{% endfor %}", {"xs": [1, 2]}),
    ("macro_output", "{% macro id_of(o) %}{% if o.tool_call_id %}{{ o.tool_call_id }}{% elif o.id %}{{ o.id }}{% endif %}{% endmacro %}{% if id_of(a) == 'c1' %}A{% endif %}{% if id_of(b) == '' %}B{% endif %}{% if id_of(b) %}T{% else %}F{% endif %}{% if id_of(c) == 'c3' %}C{% endif %}", {"a": {"id": "c1"}, "b": {}, "c": {"tool_call_id": "c3", "id": "x"}}),
    ("type_tests", "{% if x is string %}S{% endif %}{% if d is mapping %}M{% endif %}{% if l is iterable and l is not mapping %}I{% endif %}{% if n is none %}N{% endif %}{% if d is not string %}D{% endif %}", {"x": "s", "d": {}, "l": [1], "n": None}),
    ("tilde_length", "{{ 'a' ~ 1 ~ 'b' }}|{{ xs | length }}|{{ (xs | length) - 1 }}", {"xs": [1, 2, 3]}),
    ("conditional_expr", "{% for v in vs %}<{{ v | tojson(ensure_ascii=False) if v is not string else v }}>{% endfor %}", {"vs": [5, "sü", True, [1, "ä"], {"k": None}]}),
    ("trim_lstrip_blocks", "{%- if t -%}\n  <a>\n{%- endif -%}\n{% if t %}\n  x\n{% endif %}\ny", {"t": True}),
]


def main():
    ins = inputs()
    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = []
    for name, text in ENCODE:
        ids = list(tok(text, add_special_tokens=False)["input_ids"])
        enc.append(
            {
                "name": name,
                "text": text,
                "ids": ids,
                "decode_skip_special": tok.decode(ids, skip_special_tokens=True),
                "decode_keep_special": tok.decode(ids, skip_special_tokens=False),
            }
        )
    ren = []
    for name, messages, tools, agp, tvars in RENDER:
        s = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=agp, tokenize=False, **tvars)
        out = tok.apply_chat_template(messages, tools=tools, add_generation_prompt=agp, tokenize=True, **tvars)
        ids = list(out["input_ids"]) if hasattr(out, "keys") else list(out)
        if ids != list(tok(s, add_special_tokens=False)["input_ids"]):
            sys.exit(f"{name}: apply_chat_template(tokenize=True) != encode(render)")
        ren.append(
            {
                "name": name,
                "messages": messages,
                "tools": tools,
                "add_generation_prompt": agp,
                "template_vars": tvars,
                "render": s,
                "sha256": hashlib.sha256(s.encode("utf-8")).hexdigest(),
                "ids": ids,
            }
        )
    from transformers.utils.chat_template_utils import _compile_jinja_template

    feats = []
    for name, src, ctx in FEATURES:
        feats.append({"name": name, "template": src, "context": ctx, "render": _compile_jinja_template(src).render(**ctx)})
    doc = {
        "provenance": {
            "script": "oracle/export_glm5_tokenizer_goldens.py",
            "repo": "zai-org/GLM-5.3-Flash",
            "revision": REVISION,
            "inputs": ins,
            "python": sys.version.split()[0],
            "transformers": transformers.__version__,
            "tokenizers": tokenizers.__version__,
            "jinja2": jinja2.__version__,
        },
        "special_ids": {t: tok.convert_tokens_to_ids(t) for t in [
            "<|endoftext|>", "[gMASK]", "<sop>", "<|system|>", "<|user|>", "<|assistant|>", "<|observation|>",
            "<think>", "</think>", "<tool_call>", "</tool_call>", "<arg_key>", "</arg_key>", "<arg_value>", "</arg_value>",
            "<tool_response>", "</tool_response>",
        ]},
        "encode": enc,
        "render": ren,
        "jinja_features": feats,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8", newline="\n") as f:
        json.dump(doc, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print(f"wrote {OUT}: {len(enc)} encode, {len(ren)} render, {len(feats)} jinja feature cases")
    for r in ren:
        print(f"  {r['name']:42s} {len(r['ids']):5d} ids  sha256 {r['sha256']}")


if __name__ == "__main__":
    main()
