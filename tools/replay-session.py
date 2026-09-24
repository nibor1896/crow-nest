#!/usr/bin/env python3
"""#68: replay robin's stored goal-mode session against a running `serve`, at a chosen
context size and with a chosen sampling row, and say whether the two late stages of the
degeneration reproduce - the verbatim echo of Crow's goal nudge and the single-token loop.

Usage, from anywhere:
  tools/replay-session.py --target-tokens 120000 --row greedy  --turns 3
  tools/replay-session.py --target-tokens 120000 --row card    --turns 3
  tools/replay-session.py --target-tokens 120000 --row crow    --turns 3
  tools/replay-session.py --cut-index 573        --row crow    --turns 3   # the live tail

What it sends (all line numbers are `~/.local/share/crow/cli/crow_core.py`):

- The stored conversation of `decode_out/sessions/2026-09-17-goalmode/crow-session/session.json`
  (574 messages, the artefact of `#68`), cut after a USER turn so the engine has to answer, and
  re-sent WHOLE on every round - which is what Crow does (`:3783-3785`, `:4866`).
- `tools` as Crow builds it: the module-level `TOOLS` constant is exec-extracted out of
  `crow_core.py` (25 functions, ~14.5 KB of JSON). Without that file a `--no-tools` style
  fallback list is built from the tool names the session actually called, and the record says so.
- The model's own answer is stored the way Crow stores it (`:3754-3761`): `content` verbatim,
  `tool_calls` with `arguments` as the accumulated raw string, `reasoning_content` alongside when
  the stream carried one (#67).
- A tool call is answered with the LAST result the live session gave that tool (so the replay
  feeds back the artefact's own strings, not invented ones), then the goal nudge of the cut point
  is sent again as the next user turn - which is what the live session did on 105 of its 114 user
  turns and is the material the model started to parrot.

The three sampling rows (`--row`):

| row | temperature | top_p | top_k | presence_penalty | seed | what it is |
|---|---|---|---|---|---|---|
| `greedy` | 0 (sent) | - | - | - | - | the A4 greedy path, the gate discipline; `temperature 0` is SENT - since crow-nest #111 an absent temperature samples at the card row |
| `card` | 0.7 | 0.8 | 20 | 1.5 | 0 | the model card's NON-thinking row (= the engine's defaults) |
| `crow` | 1.0 | 0.95 | 20 | 1.5 | 0 | what Crow sent live (thinking-row temperature, non-thinking penalty) |

De-duplicated nudges (`--dedup-nudges`, #68 open question 1, 2026-09-18):

The stored history has **114 user turns of 9 distinct texts**, and **105 of them are goal-mode
nudges**: `[Goal mode. 3 of 5 steps done. Next is step 4: ...]` 102 times byte for byte, plus
`2 of 5` twice and `1 of 5` once. The flag collapses CONSECUTIVE identical user turns - a user
message whose text is byte-identical to the previous USER message (assistant and tool messages in
between are kept, so the churn is untouched) is dropped. That keeps the first nudge after any
different user message and drops the byte-identical repeats that only re-nudge; the runs it
collapses are 2, 3, 27, 60 and 12 turns long.

- **99 of the 114 user turns go, 15 remain** (6 of them goal nudges), and the stored message list
  goes 574 -> 475. Nothing else is touched: all 273 assistant turns, all 186 tool results and the
  system turn stay exactly where they were, so the ONLY variable this flag moves is the repetition.
- The cut is taken on the STORED indices first and the de-duplication is applied to that prefix, so
  `--cut-index 483` still names the message it names. When the de-duplication removes the cut
  point's own nudge (it does, at every cut inside a run), ONE copy of it is appended back, so the
  request still ends on the user turn the model has to answer.
- Rendered length (this machine, 2026-09-18, images stripped, greedy, the server's own
  `prompt_tokens`): the full de-duplicated history is **160,589 ids** against 168,928 with the
  nudges kept, and the `--cut-index 483` point is **158,717** against 163,401 - about 84 ids per
  dropped nudge. 16 user turns are sent for the full history and 14 at cut 483 (the re-appended
  copy included). `docs/long-context-goalmode.md` section 8 has the runs and the answer.
- The per-round probe is unchanged: after each answered round the replay appends the model's turn,
  its tool results and ONE nudge, exactly as in runs A to E. The flag de-duplicates the STORED
  conversation, not the probe.

Images: every `image_url` part of the stored history is dropped by default (`--images strip`),
so the replay is text-only and one cold prefill does not also pay 17 tower passes. The record
names the token difference this makes; `--images keep` sends them.

Per round the record carries: prompt tokens (cached / prefilled), the generated text, the echo
verdict, the degeneration verdict, and both rates out of `timings`. Exit 0 when every round
answered 200, 1 when any round was refused, 2 on a usage error.
"""
import argparse
import collections
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

SESSION = "decode_out/sessions/2026-09-17-goalmode/crow-session/session.json"
CROW_CORE = os.path.expanduser("~/.local/share/crow/cli/crow_core.py")

ROWS = {
    # name: (temperature, top_p, top_k, presence_penalty, seed) - 0 temperature = greedy,
    # sent explicitly: since crow-nest #111 an ABSENT temperature samples at the card row
    "greedy": (0, None, None, None, None),
    "card": (0.7, 0.8, 20, 1.5, 0),
    "crow": (1.0, 0.95, 20, 1.5, 0),
}


def crow_tools(path):
    """Crow's own `TOOLS` constant, exec-extracted; (tools, provenance).

    The slice from `def _fn(` to the end of `TOOLS = [...]` is self-contained apart from a
    handful of module constants that only appear INSIDE description strings; they are stubbed,
    so a description can read `0` where Crow interpolates a timeout. That is a few bytes of
    prose in a 14.5 KB block and it is recorded here rather than papered over."""
    try:
        src = open(path, encoding="utf-8").read()
    except OSError as exc:
        return None, f"crow_core.py unreadable ({exc})"
    try:
        i = src.index("def _fn(name, description, properties, required):")
        j = src.index("\nTOOLS = [", i)
        k = src.index("\n]\n", j)
    except ValueError:
        return None, "crow_core.py has no TOOLS block in the expected shape"
    seg = src[i:k + 3]
    ns = {n: 0 for n in set(re.findall(r"\b([A-Z][A-Z0-9_]{2,})\b", seg))}
    ns.pop("TOOLS", None)
    try:
        exec(seg, ns)                                        # noqa: S102 - Crow's own literal
        tools = ns["TOOLS"]
    except Exception as exc:                                  # noqa: BLE001
        return None, f"TOOLS block did not evaluate ({exc})"
    return tools, f"crow_core.py TOOLS, {len(tools)} functions, {len(json.dumps(tools))} B"


def fallback_tools(names):
    """one function per tool name the session called - the shape, not Crow's wording"""
    return [{"type": "function",
             "function": {"name": n, "description": f"The {n} tool of Crow.",
                          "parameters": {"type": "object",
                                         "properties": {"arg": {"type": "string"}},
                                         "required": []}}}
            for n in names]


def strip_images(msg):
    """the stored message with every `image_url` part dropped, text parts kept"""
    c = msg.get("content")
    if not isinstance(c, list):
        return msg
    keep = [p for p in c if p.get("type") == "text"]
    out = dict(msg)
    out["content"] = "".join(p.get("text", "") for p in keep)
    return out


def text_chars(msg):
    c = msg.get("content")
    n = 0
    if isinstance(c, str):
        n += len(c)
    elif isinstance(c, list):
        for p in c:
            if p.get("type") == "text":
                n += len(p.get("text", ""))
    for tc in msg.get("tool_calls") or []:
        fn = tc.get("function") or {}
        n += len(fn.get("name", "")) + len(fn.get("arguments", ""))
    return n


def build_prefix(messages, target, cut_index, ratio):
    """the history up to `target` estimated tokens, cut after a USER turn.

    `ratio` is EFFECTIVE chars per rendered token, measured on this artefact 2026-09-18:
    433,889 text chars tokenize to 157,345 ids (`serve tokenize --chat`), and the render adds the
    per-message headers plus Crow's 14.5 KB tools block, about 10k ids over the whole history -
    so 433,889 / ~167k = 2.60. The cut is an ESTIMATE; the record carries `prompt_tokens` from
    the server, which is the truth."""
    if cut_index is not None:
        end = cut_index + 1
        while end > 0 and messages[end - 1].get("role") != "user":
            end -= 1
        return messages[:end], end - 1
    tot = 0
    best = None
    for i, m in enumerate(messages):
        tot += text_chars(m)
        if m.get("role") == "user" and tot / ratio <= target:
            best = i
    if best is None:
        raise SystemExit("replay-session.py: no user turn under the target")
    return messages[:best + 1], best


def user_text(msg):
    """the text of one message, the way the render sees it (a list content is joined)"""
    c = msg.get("content")
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(p.get("text", "") for p in c if p.get("type") == "text")
    return ""


def dedup_user_turns(messages):
    """#68: drop every user turn that is byte-identical to the PREVIOUS user turn.

    Consecutive is meant over the USER subsequence: the assistant turns and tool results between
    two nudges are kept, so the session's churn, its half-finished tool output and its length
    profile are the only things left that could flip the model. Returns (messages, dropped)."""
    out, prev, dropped = [], None, 0
    for m in messages:
        if m.get("role") == "user":
            t = user_text(m)
            if t == prev:
                dropped += 1
                continue
            prev = t
        out.append(m)
    return out, dropped


def last_tool_results(messages):
    """name -> the last result the live session gave that tool, as a string"""
    out = {}
    for i, m in enumerate(messages):
        for tc in m.get("tool_calls") or []:
            name = (tc.get("function") or {}).get("name", "")
            cid = tc.get("id")
            for nxt in messages[i + 1:i + 4]:
                if nxt.get("role") == "tool" and nxt.get("tool_call_id") == cid:
                    c = nxt.get("content")
                    if isinstance(c, str):
                        out[name] = c
                    elif isinstance(c, list):
                        out[name] = "".join(p.get("text", "") for p in c if p.get("type") == "text")
                    break
    return out


def shingles(text, n=32):
    t = " ".join(text.split())
    return {t[i:i + n] for i in range(0, max(0, len(t) - n + 1))}


def echo_verdict(answer, nudge):
    """how much of the nudge the answer repeats, by 32-char shingle overlap of the normalised
    texts. The live echo turns repeat the nudge nearly whole, so anything over 0.5 is the
    stage-2 shape; the marker string of the live session is checked by name as well."""
    a, n = shingles(answer), shingles(nudge)
    frac = (len(a & n) / len(a)) if a else 0.0
    return {"shingle_frac": round(frac, 3),
            "marker": "3 of 5 steps done" in answer,
            "echo": bool(a and (frac >= 0.5 or "3 of 5 steps done" in answer))}


def degeneration_verdict(answer, completion_tokens=None):
    """the stage-3 shape, in its two live forms.

    Form A, inside one answer: the longest run of ONE repeated unit (1 to 8 characters,
    whitespace normalised out of the comparison) over half the answer with at least 20 repeats,
    or one whitespace-separated word over half the words.

    Form B, ACROSS answers, which is the form robin's session actually took: the whole answer is
    one or two tokens and the turn ends with `finish stop`. 48 of the 293 answers of the live
    session are the single token `3` (id 18) with `finish stop`, from prompt 172,599 tokens on
    (`decode_out/sessions/2026-09-17-goalmode/serve.log:6139`); the "digit written non-stop" of
    the report is the client concatenating 48 one-token answers, not one runaway generation.
    Either form is a Fail."""
    t = answer.strip()
    best = (0, "")
    for u in range(1, 9):
        i = 0
        while i + u <= len(t):
            unit = t[i:i + u]
            reps = 1
            while t[i + reps * u: i + (reps + 1) * u] == unit:
                reps += 1
            if reps * u > best[0] and reps >= 3:
                best = (reps * u, unit)
            i += 1
    words = t.split()
    top = collections.Counter(words).most_common(1)
    word_frac = (top[0][1] / len(words)) if words else 0.0
    run_frac = (best[0] / len(t)) if t else 0.0
    inside = bool((run_frac >= 0.5 and best[0] >= 20) or (word_frac >= 0.5 and len(words) >= 20))
    single = bool(completion_tokens is not None and completion_tokens <= 2)
    return {"longest_repeat_unit": best[1][:16], "repeat_chars": best[0],
            "repeat_frac": round(run_frac, 3),
            "top_word": (top[0][0][:16] if top else ""), "top_word_frac": round(word_frac, 3),
            "single_token_answer": single, "repeat_inside_answer": inside,
            "degenerate": bool(inside or single)}


def stream_round(base, messages, tools, row, max_tokens, timeout):
    """one POST /v1/chat/completions, the reader of `crow_core.py:5040-5127` plus `timings`"""
    temperature, top_p, top_k, presence, seed = row
    body = {"model": "crow-nest", "messages": messages, "tools": tools, "stream": True,
            "stream_options": {"include_usage": True}, "timings_per_token": True,
            "max_tokens": max_tokens}
    if temperature == 0:
        body["temperature"] = 0     # #111: greedy is a sent temperature 0, never an absent one
    else:
        body.update({"temperature": temperature, "top_p": top_p, "top_k": top_k,
                     "presence_penalty": presence, "seed": seed})
    req = urllib.request.Request(base + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    calls, content, reasoning, finish = {}, "", "", None
    usage, timings = {}, {}
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for line in r:
            line = line.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            frame = json.loads(payload)
            usage = frame.get("usage") or usage
            timings = frame.get("timings") or timings
            choice = (frame.get("choices") or [{}])[0]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content += delta["content"]
            if delta.get("reasoning_content"):
                reasoning += delta["reasoning_content"]
            for call in delta.get("tool_calls") or []:
                slot = calls.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                if call.get("id"):
                    slot["id"] = call["id"]
                fn = call.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]
            if choice.get("finish_reason"):
                finish = choice["finish_reason"]
    return {"finish": finish, "content": content, "reasoning": reasoning,
            "calls": [calls[i] for i in sorted(calls)], "usage": usage, "timings": timings,
            "wall_s": round(time.time() - t0, 2)}


def main():
    sys.stdout.reconfigure(line_buffering=True)   # a 3-minute prefill must not sit in a buffer
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--session", default=SESSION)
    ap.add_argument("--target-tokens", type=int, default=120000)
    ap.add_argument("--cut-index", type=int, default=None,
                    help="cut after this message index instead of the token target")
    ap.add_argument("--ratio", type=float, default=2.60,
                    help="effective chars per rendered token of this artefact (measured)")
    ap.add_argument("--row", choices=sorted(ROWS), default="greedy")
    ap.add_argument("--turns", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--images", choices=("strip", "keep"), default="strip")
    ap.add_argument("--dedup-nudges", action="store_true",
                    help="collapse consecutive identical user turns of the STORED history "
                         "(#68: 105 goal nudges -> 6, 114 user turns -> 15)")
    ap.add_argument("--timeout", type=int, default=3600)
    ap.add_argument("--tools-from", default=CROW_CORE)
    ap.add_argument("--out", default=None, help="write the record as JSON here")
    a = ap.parse_args()

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    path = a.session if os.path.isabs(a.session) else os.path.join(root, a.session)
    with open(path, encoding="utf-8") as fh:
        session = json.load(fh)
    stored = session["messages"]
    results = last_tool_results(stored)
    messages = [strip_images(m) for m in stored] if a.images == "strip" else stored

    prefix, cut = build_prefix(messages, a.target_tokens, a.cut_index, a.ratio)
    nudge = prefix[-1].get("content")
    if not isinstance(nudge, str):
        nudge = user_text(prefix[-1])
    dropped = 0
    if a.dedup_nudges:
        before = len(prefix)
        prefix, dropped = dedup_user_turns(prefix)
        # the cut point's own nudge is the last of a run at every cut inside one, so the
        # de-duplication takes it with the rest; ONE copy goes back, because the request has
        # to end on the user turn the model answers
        if prefix[-1].get("role") != "user":
            prefix = prefix + [{"role": "user", "content": nudge}]
        print(f"replay-session.py: --dedup-nudges dropped {dropped} of the {before} messages "
              f"({sum(1 for m in prefix if m.get('role') == 'user')} user turns left of "
              f"{sum(1 for m in messages[:cut + 1] if m.get('role') == 'user')})")
    tools, provenance = crow_tools(a.tools_from)
    if tools is None:
        names = sorted({(tc.get("function") or {}).get("name", "")
                        for m in stored for tc in (m.get("tool_calls") or [])} - {""})
        tools = fallback_tools(names)
        provenance = f"fallback, {len(tools)} names out of the session ({provenance})"

    base = f"http://127.0.0.1:{a.port}"
    row = ROWS[a.row]
    record = {"session": a.session, "images": a.images, "cut_index": cut,
              "dedup_nudges": bool(a.dedup_nudges), "user_turns_dropped": dropped,
              "user_turns_sent": sum(1 for m in prefix if m.get("role") == "user"),
              "cut_role": prefix[-1].get("role"), "messages_sent": len(prefix),
              "row": a.row, "sampling": {"temperature": row[0], "top_p": row[1], "top_k": row[2],
                                         "presence_penalty": row[3], "seed": row[4]},
              "max_tokens": a.max_tokens, "tools": provenance, "turns": []}
    print(f"replay-session.py: cut after message {cut} ({prefix[-1].get('role')}), "
          f"{len(prefix)} messages, images {a.images}, row {a.row} {record['sampling']}")
    print(f"replay-session.py: tools = {provenance}")
    print(f"replay-session.py: nudge tail {nudge[-120:]!r}")

    history = list(prefix)
    bad = 0
    for rnd in range(a.turns):
        body_b = len(json.dumps({"messages": history, "tools": tools}).encode())
        try:
            r = stream_round(base, history, tools, row, a.max_tokens, a.timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", "replace")[:600]
            print(f"round {rnd}: HTTP {exc.code}, body {body_b} B -> {detail}")
            record["turns"].append({"round": rnd, "http": exc.code, "error": detail})
            bad += 1
            break
        u, t = r["usage"], r["timings"]
        echo = echo_verdict(r["content"], nudge)
        deg = degeneration_verdict(r["content"], u.get("completion_tokens"))
        turn = {"round": rnd, "http": 200, "body_bytes": body_b, "finish": r["finish"],
                "prompt_tokens": u.get("prompt_tokens"),
                "cached_tokens": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
                "prefilled": t.get("prompt_n"), "prompt_per_second": t.get("prompt_per_second"),
                "completion_tokens": u.get("completion_tokens"),
                "decode_per_second": t.get("predicted_per_second"),
                "wall_s": r["wall_s"], "content": r["content"], "reasoning": r["reasoning"],
                "tool_calls": r["calls"], "echo": echo, "degeneration": deg,
                "think_tag_in_content": [x for x in ("</think>", "<think>") if x in r["content"]]}
        record["turns"].append(turn)
        print(f"round {rnd}: 200 prompt {turn['prompt_tokens']} tok "
              f"({turn['cached_tokens']} cached, {turn['prefilled']} prefilled at "
              f"{turn['prompt_per_second']} tok/s), generated {turn['completion_tokens']} tok at "
              f"{turn['decode_per_second']} tok/s, finish {r['finish']}, "
              f"{len(r['calls'])} call(s) {[c['name'] for c in r['calls']]}, wall {r['wall_s']}s")
        print(f"        echo {echo}  degeneration {deg}")
        print(f"        content {r['content'][:300]!r}")
        if len(r["content"]) > 300:
            print(f"        ... tail {r['content'][-160:]!r}")

        # the history the way Crow writes it back (`crow_core.py:3754-3761`)
        turn_msg = {"role": "assistant", "content": r["content"]}
        if r["reasoning"]:
            turn_msg["reasoning_content"] = r["reasoning"]
        if r["calls"]:
            turn_msg["tool_calls"] = [{"id": c["id"] or f"call_{i}", "type": "function",
                                       "function": {"name": c["name"], "arguments": c["arguments"]}}
                                      for i, c in enumerate(r["calls"])]
        history.append(turn_msg)
        for i, c in enumerate(r["calls"]):
            history.append({"role": "tool",
                            "content": results.get(c["name"], '{"ok": true}'),
                            "tool_call_id": c["id"] or f"call_{i}"})
        history.append({"role": "user", "content": nudge})

    echoes = sum(1 for t in record["turns"] if t.get("echo", {}).get("echo"))
    degs = sum(1 for t in record["turns"] if t.get("degeneration", {}).get("degenerate"))
    record["summary"] = {"rounds": len(record["turns"]), "refused": bad,
                         "echo_rounds": echoes, "degenerate_rounds": degs}
    print(f"replay-session.py: {len(record['turns'])} round(s), {bad} refused, "
          f"{echoes} echoing the nudge, {degs} degenerate")
    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(record, fh, ensure_ascii=False, indent=1)
        print(f"replay-session.py: record written to {a.out}")
    return 1 if bad else 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
