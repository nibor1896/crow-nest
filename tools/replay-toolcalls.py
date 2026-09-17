#!/usr/bin/env python3
"""TASK J: replay a Crow goal-mode tool loop against `serve` - the engine's OWN streamed
`tool_calls` are fed back as the assistant turn, verbatim, the way `crow_core.py` stores and
re-sends them, so a poisoned history shows up as the 400 storm it caused live.

Usage, from anywhere:  tools/replay-toolcalls.py [--port 8099] [--rounds 4] [--first-max-tokens 0]

What it reproduces (all line numbers are `~/.local/share/crow/cli/crow_core.py`):

- `:5060-5076`  the SSE reader: `index` picks a slot, `id`/`name` only on a truthy value,
                `arguments` accumulated by plain string concatenation, never parsed.
- `:3756-3761`  the assistant turn is stored with `arguments` as THAT raw string.
- `:3783-3785`, `:4866`  the whole history is re-sent, unchanged, every turn.
- `:13485-13487`  a 400 ends the turn and NOTHING is dropped, so the poison is permanent.

Round 0 is TRUNCATED on purpose - that is the shape that killed robin's 79-round session. With
`--first-max-tokens 0` (the default) the budget is found first, by sending the same first turn at
a rising budget until the answer is a tool call the model did not finish; those probes send no
history, so they cannot poison the loop that follows. Before the TASK J fix round 0 leaves an
unterminated `arguments` string in the history and every later round answers
`400 {"error":"chat template render failed: invalid operation: cannot convert value into pairs
(in chat:136)"}`. After it, `arguments` is always a parseable JSON object (a truncated one
carries `"_truncated": true`) and the session runs.

`--write-file` (TASK K) runs the same loop with a `write_file` tool whose `content` parameter
carries a multi-line HTML body - quotes, backslashes, tabs and a `<script>` block - which is the
shape that produced robin's 3,112-byte `arguments` string the template could not iterate. Every
round prints how the accumulated `arguments` reads back and, when it is not JSON, the decoder's
own error text plus the offending byte window; the raw string is written to
`--dump-dir` so the next case is diagnosable.

`--poison` splices a HANDCRAFTED broken turn into the history after round 0 - the exact shape
the live 400 carried, `arguments` left at `{"path":"/etc/host` - so the second end of the fix
(`serve::normalize_messages`) is exercised even on a build whose model finishes every call.
`--refusals` sends four message shapes the template cannot render and prints the 400 bodies;
each one must name the message index and the field.

Exit 0 when every round answered 200, 1 when any round was refused, 2 on a usage error.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

TOOLS = [{
    "type": "function",
    "function": {
        "name": "read_file",
        "description": "Read a UTF-8 text file.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path to the file."},
                "start_line": {"type": "integer", "description": "First line, 1-based."},
            },
            "required": ["path"],
        },
    },
}]

WRITE_TOOLS = [{
    "type": "function",
    "function": {
        "name": "write_file",
        "description": "Write a UTF-8 text file, creating parent directories.",
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Absolute path of the file."},
                "content": {"type": "string", "description": "The whole file content."},
            },
            "required": ["path", "content"],
        },
    },
}]

# TASK K: every prompt asks for a body that carries the characters JSON has to escape -
# double quotes, backslashes, tabs, newlines - inside one `content` parameter.
WRITE_PROMPTS = [
    'Write the file /tmp/crow-readme-sheet.html. It must be a complete HTML5 document with '
    '<!doctype html>, <html lang="en" data-theme="dark">, a <head> with <meta charset="utf-8"> '
    'and a <title>, a <style> block, a <script> block containing the regular expression '
    '/a\\b"c/g and a console.log with a tab escape, and a <body> with one <p>. Use the '
    'write_file tool.',
    'Write /tmp/crow-notes.md: a markdown note with a fenced code block containing a Windows '
    'path C:\\Users\\robin\\a.txt, a quoted string "hello", and a tab-indented line. '
    'Use the write_file tool.',
    'Write /tmp/crow-config.json: a JSON config with a "path" key holding an escaped Windows '
    'path and a "note" key holding a sentence with a quoted word. Use the write_file tool.',
]

PROMPTS = [
    "Read the file /etc/hostname, lines from 1, and tell me what is in it.",
    "Now read /etc/os-release from line 1 and name the distribution.",
    "Read /proc/version from line 1 and name the kernel.",
    "Read /etc/shells from line 1 and list the shells.",
    "Read /etc/hosts from line 1 and name the loopback entry.",
]


def stream_round(base, messages, max_tokens, tools=None):
    """one POST /v1/chat/completions, the reader of crow_core.py:5040-5127.

    returns (finish_reason, content, [{id, name, arguments}]) or raises HTTPError."""
    body = {
        "model": "crow-nest",
        "messages": messages,
        "tools": tools if tools is not None else TOOLS,
        "stream": True,
        "stream_options": {"include_usage": True},
        "timings_per_token": True,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    calls = {}
    content = ""
    finish = None
    with urllib.request.urlopen(req, timeout=1800) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            frame = json.loads(payload)
            choice = frame["choices"][0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
            for call in delta.get("tool_calls") or []:                 # crow_core.py:5060
                idx = call.get("index", 0)
                slot = calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]               # crow_core.py:5069
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return finish, content, [calls[i] for i in sorted(calls)]


def call_state(call):
    """how one accumulated call reads back: `crow_core.py:12798-12803` is the same test."""
    try:
        args = json.loads(call["arguments"] or "{}")
    except json.JSONDecodeError as exc:
        # TASK K: the decoder's own text and the byte window it stopped in - without them a
        # failure of this kind is a `_raw` line and nothing else
        raw = call["arguments"]
        lo = max(0, exc.pos - 60)
        return (f"NOT JSON [{exc.msg} at byte {exc.pos} of {len(raw)}] "
                f"...{raw[lo:exc.pos + 60]!r}...")
    if not isinstance(args, dict):
        return type(args).__name__
    return "object, _truncated" if args.get("_truncated") else "object"


def find_truncating_budget(base, prompt, lo, hi, step):
    """the smallest budget that cuts a tool call in half, or None.

    Every probe sends the ONE user turn and nothing else, so no probe can poison the replay."""
    for budget in range(lo, hi + 1, step):
        try:
            finish, _, calls = stream_round(base, [{"role": "user", "content": prompt}], budget)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            print(f"probe max_tokens={budget}: HTTP {exc.code} -> {detail}")
            return None
        state = ", ".join(call_state(c) for c in calls) or "no call"
        print(f"probe max_tokens={budget}: finish={finish}, {len(calls)} call(s), {state}")
        if finish == "length" and calls and any(call_state(c) != "object" for c in calls):
            return budget
    return None


# the shapes the template cannot render; each 400 body must name the message index and field
REFUSALS = [
    ("content is an object",
     [{"role": "user", "content": {"text": "hi"}}], "message 0 content is an object"),
    ("a content part that is not a block",
     [{"role": "user", "content": ["hi"]}], "message 0 content part 0 is a string"),
    ("tool_calls is not an array",
     [{"role": "user", "content": "hi"},
      {"role": "assistant", "content": "x", "tool_calls": "abc"}], "message 1 tool_calls is a string"),
    ("a tool call with no name",
     [{"role": "user", "content": "hi"},
      {"role": "assistant", "content": "x", "tool_calls": [{"function": {"arguments": {}}}]}],
     "message 1 tool_call 0 has no string function.name"),
]


def refusals(base):
    """every unrenderable shape answers a 400 that names the message index; 0 when all do."""
    bad = 0
    for label, messages, wanted in REFUSALS:
        try:
            stream_round(base, messages, 8)
            print(f"refusal {label!r}: SERVED, expected a 400")
            bad += 1
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", "replace")[:500]
            ok = exc.code == 400 and wanted in body
            print(f"refusal {label!r}: HTTP {exc.code} {'names it' if ok else 'DOES NOT NAME IT'} -> {body}")
            bad += 0 if ok else 1
    print(f"replay-toolcalls.py: {len(REFUSALS)} refusal shape(s), {bad} wrong")
    return 1 if bad else 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--max-tokens", type=int, default=96)
    ap.add_argument("--first-max-tokens", type=int, default=0,
                    help="budget of round 0; 0 finds the one that cuts the tool call in half")
    ap.add_argument("--probe-from", type=int, default=20)
    ap.add_argument("--probe-to", type=int, default=64)
    ap.add_argument("--probe-step", type=int, default=2)
    ap.add_argument("--poison", action="store_true",
                    help="splice the handcrafted unterminated tool call into the history")
    ap.add_argument("--write-file", action="store_true",
                    help="TASK K: run the loop with the write_file tool and HTML content")
    ap.add_argument("--dump-dir", default=None,
                    help="write every accumulated arguments string to this directory")
    ap.add_argument("--refusals", action="store_true",
                    help="also send four unrenderable shapes and print the 400 bodies")
    a = ap.parse_args()
    if a.rounds < 1:
        print("replay-toolcalls.py: --rounds must be at least 1", file=sys.stderr)
        return 2
    base = f"http://127.0.0.1:{a.port}"

    tools = WRITE_TOOLS if a.write_file else TOOLS
    prompts = WRITE_PROMPTS if a.write_file else PROMPTS
    if a.dump_dir:
        os.makedirs(a.dump_dir, exist_ok=True)

    first = a.first_max_tokens
    if first <= 0 and a.write_file:
        # TASK K: the write_file loop is not about truncation - every round gets the full budget
        first = a.max_tokens
    if first <= 0:
        first = find_truncating_budget(base, PROMPTS[0], a.probe_from, a.probe_to, a.probe_step)
        if first is None:
            print("replay-toolcalls.py: no budget in the probe range cut a tool call in half; "
                  "widen --probe-from/--probe-to", file=sys.stderr)
            return 2
        print(f"replay-toolcalls.py: round 0 runs at max_tokens={first}, the truncating budget")

    messages = []
    refused = 0
    broken = 0
    for rnd in range(a.rounds):
        messages.append({"role": "user", "content": prompts[rnd % len(prompts)]})
        budget = first if rnd == 0 else a.max_tokens
        size = len(json.dumps({"messages": messages}).encode())
        try:
            finish, content, calls = stream_round(base, messages, budget, tools)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]       # crow_core.py:4182
            print(f"round {rnd}: HTTP {exc.code}, body {size} B -> {detail}")
            refused += 1
            continue                                                   # crow_core.py:13485-13487
        note = [f"call {i} {c['name']!r} arguments={call_state(c)} {c['arguments'][:120]!r}"
                for i, c in enumerate(calls)]
        for i, c in enumerate(calls):
            if a.dump_dir:
                with open(os.path.join(a.dump_dir, f"round{rnd}-call{i}.args"), "w") as fh:
                    fh.write(c["arguments"])
            if call_state(c).startswith("NOT JSON"):
                broken += 1
        print(f"round {rnd}: 200, body {size} B, finish={finish}, "
              f"content {len(content)} B, {len(calls)} call(s)")
        for n in note:
            print(f"          {n}")
        # the history exactly as Crow stores it: `arguments` is the raw string (crow_core.py:3756)
        if calls:
            messages.append({
                "role": "assistant",
                "content": content,
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["arguments"]}}
                    for c in calls
                ],
            })
            for c in calls:
                messages.append({"role": "tool", "tool_call_id": c["id"],
                                 "content": "crow-nest-replay: the tool result of this call"})
        else:
            messages.append({"role": "assistant", "content": content})
        if a.poison and rnd == 0:
            # the history entry of record: Crow stored this string verbatim and re-sent it
            poison = '{"path":"/etc/host'
            messages.append({"role": "assistant", "content": "", "tool_calls": [
                {"id": "call_poison", "type": "function",
                 "function": {"name": "read_file", "arguments": poison}}]})
            messages.append({"role": "tool", "tool_call_id": "call_poison",
                             "content": "error: arguments were not valid JSON"})
            print(f"          spliced the poisoned turn: arguments={poison!r}")

    print(f"replay-toolcalls.py: {a.rounds} round(s), {refused} refused, "
          f"{broken} call(s) whose arguments were not a JSON object")
    if a.refusals:
        refused_ok = refusals(base)
        if refused_ok != 0:
            return refused_ok
    return 1 if (refused or broken) else 0


if __name__ == "__main__":
    sys.exit(main())
