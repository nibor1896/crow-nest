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

`--think` (#67) is the shape robin hit live: a ~3 KB code paste in the first user turn, then
three ordinary turns, with the whole history re-sent every turn the way Crow does it. Before
the fix the model answers the paste with a stray `</think>`, the client stores it, the chat
template renders it VERBATIM inside the assistant turn's own think block, and every later turn
ends with the tag. This mode asserts that NO streamed `content` of any round carries `</think>`
or `<think>`, and it prints what each round's `reasoning_content` carried, if anything. It
sends no `tools`: the tags are a content-path bug, not a tool-call one.

`--gone-client` (#54) is the live test of the `stream:false` document path: one POST with
`stream: false` and `max_tokens 512` on a RAW socket, the connection dropped with a plain
`close()` after `--drop-after` seconds, then the NEXT request timed from the drop. Before the
fix that request waited for the whole 512-token budget - measured 7.463 s on 2026-09-18, with
`generated 512 tok` in the log - because `CollectSink` could not notice a gone client; after it
the generation ends ONE decode step after the drop and the next request is served 0.172 s
later. The mode also sends a `shutdown(SHUT_WR)` client that keeps its read side open, which
is the SAME wire event and must still get its whole document, and it first runs `--cost-rounds`
timed document rounds plus one streaming round, so the per-token cost of the probe and the
identity of the streamed answer can be read off the same command on two builds.

Exit 0 when every round answered 200, 1 when any round was refused, 2 on a usage error.
"""
import argparse
import hashlib
import json
import os
import socket
import sys
import time
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

# #67: the trigger of record - a large code block pasted into the chat, right before the
# first stray `</think>` of robin's session. ~3 KB, the shape a `run_command` heredoc comes
# back as: long, dense, full of braces and quotes, and nothing to do with reasoning.
_STAGE = """fn stage_%d(rows: &mut [f32], scale: f32, bias: f32) -> f32 {
    let mut acc = 0.0f32;
    for (j, r) in rows.iter_mut().enumerate() {
        let w = if j %% 3 == 0 { scale } else { scale * 0.5 };
        *r = r.mul_add(w, bias);
        acc += *r;
    }
    acc / (rows.len() as f32).max(1.0)
}"""

CODE_PASTE = (
    "Here is the module I am working on. Read it, then answer my questions.\n\n```rust\n"
    + "\n\n".join(_STAGE % i for i in range(10))
    + "\n```\n\nWhat does stage_7 do differently from stage_8?"
)

# #67: the paste, then three ordinary turns - the shape that degraded live
THINK_PROMPTS = [
    CODE_PASTE,
    "Now name the one line that decides the weight w.",
    "And what happens when rows is empty?",
    "Summarise all three answers in one sentence.",
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
    reasoning = ""
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
            if delta.get("reasoning_content"):                         # crow_core.py:5045
                reasoning += delta["reasoning_content"]
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
    return finish, content, [calls[i] for i in sorted(calls)], reasoning


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
            finish, _, calls, _r = stream_round(base, [{"role": "user", "content": prompt}], budget)
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


def think_round(base, max_tokens, rounds):
    """#67: the ~3 KB code paste, then three turns, history re-sent whole every turn.

    No `tools` and no truncation: this is the CONTENT path. A round is bad when its streamed
    content carries `<think>` or `</think>` - which is what the client would store and re-send,
    and what the chat template then renders verbatim inside the assistant turn's own think
    block. Returns the number of bad rounds (0 is what the fix must produce)."""
    messages = []
    bad = 0
    for rnd in range(rounds):
        prompt = THINK_PROMPTS[rnd % len(THINK_PROMPTS)]
        messages.append({"role": "user", "content": prompt})
        size = len(json.dumps({"messages": messages}).encode())
        try:
            finish, content, _calls, reasoning = stream_round(
                base, messages, max_tokens, tools=[])
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:500]
            print(f"think round {rnd}: HTTP {exc.code}, body {size} B -> {detail}")
            return rounds - rnd
        tags = [t for t in ("</think>", "<think>") if t in content]
        if tags:
            bad += 1
        print(f"think round {rnd}: 200, prompt {len(prompt)} B, body {size} B, "
              f"finish={finish}, content {len(content)} B, reasoning {len(reasoning)} B, "
              f"tags in content {tags if tags else 'none'}")
        print(f"          content tail {content[-90:]!r}")
        if reasoning:
            print(f"          reasoning head {reasoning[:90]!r}")
        # the history exactly as Crow stores it (`crow_core.py:3754-3761`): the turn verbatim,
        # `reasoning_content` alongside it when the stream carried one
        turn = {"role": "assistant", "content": content}
        if reasoning:
            turn["reasoning_content"] = reasoning
        messages.append(turn)
    print(f"replay-toolcalls.py: {rounds} think round(s), {bad} with a reasoning tag in content")
    return bad


# #54: a prompt that runs a 512-token budget out, so a client that leaves mid-generation
# would otherwise hold the one slot of the process for the whole budget
GONE_PROMPT = ("Write a detailed essay of at least 600 words about the history of the "
               "printing press, from Gutenberg to the rotary press.")


def document_round(base, prompt, max_tokens, timeout=1800):
    """one POST with `stream: false` - the 7.11.13 document. Returns the parsed document."""
    body = {
        "model": "crow-nest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    req = urllib.request.Request(
        base + "/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def raw_request(host, port, prompt, max_tokens):
    """#54: one whole `stream:false` POST as bytes, head and body, for a RAW socket."""
    body = json.dumps({
        "model": "crow-nest",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "max_tokens": max_tokens,
        "temperature": 0,
    }).encode()
    head = (f"POST /v1/chat/completions HTTP/1.1\r\nHost: {host}:{port}\r\n"
            f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n"
            f"Connection: close\r\n\r\n").encode()
    return head + body


def raw_post_then_drop(host, port, prompt, max_tokens, drop_after):
    """#54: send a whole `stream:false` POST, then DROP the connection mid-generation.

    A plain `close()`, never `shutdown(SHUT_WR)`: the two are the same wire event (FIN) and
    the server may only treat the first as a gone client - which is what the baseline probe of
    `ClientProbe` decides. Returns the seconds the socket was open."""
    s = socket.create_connection((host, port), timeout=30)
    t0 = time.monotonic()
    s.sendall(raw_request(host, port, prompt, max_tokens))
    time.sleep(drop_after)
    s.close()                                  # the drop: FIN, no byte read, no half close
    return time.monotonic() - t0


def raw_post_half_close(host, port, prompt, max_tokens, timeout=120):
    """#54: send a whole POST, `shutdown(SHUT_WR)`, then WAIT for the document.

    The case the probe may not get wrong: this client is at EOF on the server's read side, the
    same wire event as the drop above, and it is still waiting for its answer. Returns
    (status line, the parsed document) - a truncated or missing document is the failure."""
    s = socket.create_connection((host, port), timeout=timeout)
    s.sendall(raw_request(host, port, prompt, max_tokens))
    s.shutdown(socket.SHUT_WR)                 # the write side only; the read side stays open
    chunks = []
    while True:
        part = s.recv(65536)
        if not part:
            break
        chunks.append(part)
    s.close()
    raw = b"".join(chunks)
    status = raw.split(b"\r\n", 1)[0].decode("utf-8", "replace")
    head_end = raw.find(b"\r\n\r\n")
    doc = None
    if head_end >= 0:
        try:
            doc = json.loads(raw[head_end + 4:].decode("utf-8", "replace"))
        except json.JSONDecodeError:
            doc = None
    return status, doc


def gone_client(base, host, port, max_tokens, drop_after, bound, cost_rounds, serve_log):
    """#54: the live test - the probe's cost, the streamed answer's identity, then the drop.

    Returns 0 when the next request was served within `bound` seconds of the drop."""
    bad = 0

    # 1) the per-token cost of the probe, off the server's own `timings` block, and the
    #    streamed answer of the same prompt, so two builds can be compared value for value
    rates = []
    for rnd in range(cost_rounds):
        doc = document_round(base, GONE_PROMPT, 64)
        t = doc["timings"]
        rates.append((t["predicted_per_second"], t["predicted_per_token_ms"]))
        print(f"cost round {rnd}: {t['predicted_n']} tok, {t['predicted_per_second']} tok/s, "
              f"{t['predicted_per_token_ms']} ms/token, prefill {t['prompt_per_second']} tok/s")
    if rates:
        mean_s = sum(r[0] for r in rates) / len(rates)
        mean_ms = sum(r[1] for r in rates) / len(rates)
        print(f"cost: mean {mean_s:.3f} tok/s, {mean_ms:.3f} ms/token over {len(rates)} round(s)")
    finish, content, _calls, _r = stream_round(base, [{"role": "user", "content": GONE_PROMPT}],
                                               64, tools=[])
    digest = hashlib.sha256(content.encode()).hexdigest()
    print(f"stream identity: finish={finish}, content {len(content)} B, sha256 {digest[:16]}")

    # 2) the drop, and the wait for the NEXT request. `serve` is one request at a time, so the
    #    time from the drop to that answer IS the time the slot stayed busy.
    open_s = raw_post_then_drop(host, port, GONE_PROMPT, max_tokens, drop_after)
    t_drop = time.monotonic()
    print(f"drop: the socket carried a stream:false POST with max_tokens={max_tokens} and was "
          f"closed after {open_s:.3f} s")
    doc = document_round(base, "Say the single word: ready.", 8)
    waited = time.monotonic() - t_drop
    served = doc["usage"]["completion_tokens"]
    print(f"next request: served {served} tok {waited:.3f} s after the drop "
          f"(bound {bound:.3f} s)")
    if waited > bound:
        print(f"gone-client: the slot stayed busy for {waited:.3f} s, over the {bound:.3f} s bound")
        bad += 1

    # 3) the client the probe may NOT end: half-closed, and waiting for its document. The
    #    same prompt as the drop, so the budget is what ends it and 32 of 32 tokens is the
    #    proof that no probe cut it short.
    status, doc = raw_post_half_close(host, port, GONE_PROMPT, 32)
    got = (doc or {}).get("usage", {}).get("completion_tokens")
    finish = ((doc or {}).get("choices") or [{}])[0].get("finish_reason")
    text = ((doc or {}).get("choices") or [{}])[0].get("message", {}).get("content", "")
    print(f"half close: shutdown(SHUT_WR) then wait -> {status}, {got} tok, finish={finish}, "
          f"content {len(text)} B")
    if "200" not in status or got != 32 or finish != "length":
        print("half close: the document is missing or was cut short - a client that only "
              "closed its WRITE side must still be served")
        bad += 1

    # 4) what the server itself logged, when the log is at hand
    if serve_log:
        try:
            with open(serve_log, encoding="utf-8", errors="replace") as fh:
                lines = fh.read().splitlines()
        except OSError as exc:
            print(f"gone-client: --serve-log {serve_log}: {exc}")
            return 1
        gone = [l for l in lines if "the client is gone at step" in l]
        summary = [l for l in lines if l.startswith("[chat] prompt ") and "client gone" in l]
        for l in gone[-1:] + summary[-1:]:
            print(f"          {l}")
        if not gone:
            print("gone-client: the log carries no `[chat] the client is gone at step` line")
            bad += 1
        elif summary:
            # the bound in STEPS: the line names the step the probe fired at, and the summary
            # names how many tokens the generation produced before it stopped
            step = int(gone[-1].split("at step ")[1].split(":")[0])
            tok = int(summary[-1].split("generated ")[1].split(" tok")[0])
            print(f"gone-client: the probe fired at step {step}, the generation produced "
                  f"{tok} token(s) - {tok - step} step(s) between the drop and the stop")
            if tok - step > 2:
                print("gone-client: more than 2 steps between the probe and the stop")
                bad += 1
    print(f"replay-toolcalls.py: gone-client, {bad} finding(s)")
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
    ap.add_argument("--think", action="store_true",
                    help="#67: a ~3 KB code paste then three turns; no </think> in any content")
    ap.add_argument("--refusals", action="store_true",
                    help="also send four unrenderable shapes and print the 400 bodies")
    ap.add_argument("--gone-client", action="store_true",
                    help="#54: drop a stream:false connection mid-generation and time the slot")
    ap.add_argument("--drop-after", type=float, default=1.0,
                    help="#54: seconds the dropped connection stays open (default 1)")
    ap.add_argument("--gone-max-tokens", type=int, default=512,
                    help="#54: max_tokens of the dropped request (default 512)")
    ap.add_argument("--bound", type=float, default=5.0,
                    help="#54: seconds the next request may wait after the drop (default 5)")
    ap.add_argument("--cost-rounds", type=int, default=3,
                    help="#54: timed document rounds before the drop, for the probe cost")
    ap.add_argument("--serve-log", default=None,
                    help="#54: the serve stderr log, checked for the `[chat]` gone-client line")
    a = ap.parse_args()
    if a.rounds < 1:
        print("replay-toolcalls.py: --rounds must be at least 1", file=sys.stderr)
        return 2
    base = f"http://127.0.0.1:{a.port}"

    if a.gone_client:
        # #54: its own path - a raw socket, a drop, and the wait for the next request
        return gone_client(base, "127.0.0.1", a.port, a.gone_max_tokens, a.drop_after,
                           a.bound, a.cost_rounds, a.serve_log)

    if a.think:
        # #67: its own loop - no tools, no truncating budget, the content path alone. The stray
        # tag appears at the END of a turn, so a turn has to be allowed to finish: the tool
        # loop's default of 96 would cut most answers off before the interesting byte.
        budget = a.max_tokens if a.max_tokens != 96 else 384
        return 1 if think_round(base, budget, a.rounds) else 0

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
            finish, content, calls, _r = stream_round(base, messages, budget, tools)
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
