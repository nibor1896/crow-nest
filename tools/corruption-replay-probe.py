#!/usr/bin/env python3
"""#91 corruption probe, REPLAY -- the instrument where the damage actually happened.

WHY: both copy probes sit on their floor (short 984 tokens: 1/320; 100k tokens: 2/320,
the same two dropped lines under greedy), yet the worst corruption of 2026-09-22 came at
6,738 prompt tokens, the first answer after a Crow rollover: session.json message [2] ran
`run_command` with cwd `/home/nibor11896/three-staging` (11896 for 1896) -> [3] ENOENT.
Same session: [26] a `delegate` with the literal arguments `parameter_placeholder` /
`context_placeholder`, [69] a `write_file` path `"/\\n/home/nibor11896/..."`. The class is
not "copy a hex literal"; it is "write a tool-call ARGUMENT the context spells correctly".
This probe re-asks exactly that question: messages[0:K] of the stored session, the body
Crow builds for it, N seeds, and every returned tool call graded.

WHAT IS SENT, and where each byte comes from:
  - messages   session.json messages[0:K]. Crow's save_session writes
               `conversation.payload()`, the same list stream_reply sends, so the file IS
               the wire history (crow_core.py save_session / stream_reply) -- EXCEPT the head,
               which the save re-pins: --head-file (a preset names its own) replaces
               messages[0].content with the head the requests carried.
  - the body   built BY CROW: crow_core.stream_reply() is called with a Conversation holding
               messages[0:K] and a `_post_stream` stand-in that captures the dict and raises
               -- tools, sampling (sampling_for(model): temperature 1.0, top_p 0.95,
               min_p 0.01, top_k 20, presence_penalty 0.0 for flash-next-cnq45-m), max_tokens
               (MAX_TOKENS 16384), stream + stream_options + timings_per_token, and
               `model` = --wire-model (default DEFAULT_MODEL "crow", what the window sends).
               No second copy of Crow's request lives in this file.
  - added      `seed` (Crow sends none; serve logged `seed 0 (data sheet)` on every request
               of the session, so --seed0 0 makes round 0 the live request's seed). With
               --no-stream also `stream: false`. Nothing else is changed.
  - --tools-json FILE replaces body["tools"] with a captured array (provenance recorded).

FIDELITY IS MEASURED, NOT ASSUMED, and for the preset it is exact. session.json's messages[0]
is the head at SAVE time (17:54), not the one the requests carried: the post-rollover head
(prompt_head(include_status=True), crow_core #210) was base + SKILLS + the goal block WITH the
cut's marks (steps 1-8 [done], 9 [running]) -- no MEMORY yet, it was written at [75] -- and
the saved head is base + MEMORY + SKILLS. The preset therefore sends that head
(tools/corpora/91-replay-diorama-0922-head.txt: rollover-20260922-171255.json's head + "\n\n" +
goal_block(the goal_set of that file, those marks, include_status=True)). Second, the window
sends `model: "crow"` (its --model default) and resolves sampling from the display name, so no
reasoning-budget fields travel. With both, checked offline 2026-09-22: the body is
byte-for-byte serve's `body N bytes` on ALL 325 streamed requests of the session (delta 0;
[9] "[open]" instead of "[running]" leaves exactly 3 B), the reference tokenizer
(.venv-oracle) renders 6738 tokens at K=2 = serve's prompt_tokens, and the longest common
prefix with the pre-cut request is 4092 = serve's `[cache] COLD L 4092`. Before the fix the
rebuilt request was 155 tokens / 520 B short. Every point records body_bytes_delta_vs_live,
every round prompt_tokens_delta_vs_live.

GRADING (per returned tool call; BFCL-style AST checks plus the #91 classes):
  errors (count toward calls_with_error):
    json_invalid      `arguments` does not parse, or is not a JSON object
    unknown_tool      name not declared in the request's tools
    unknown_arg       argument name not in the tool's declared properties (`_truncated`,
                      serve's own marker for a cut call, is exempt and reported as truncated)
    missing_required  a declared required argument absent
    type_mismatch     declared integer/number/boolean/array/object, value of another type
    placeholder       a value like `parameter_placeholder`, `<path>`, `{{x}}`
    control_char      a path-like argument (path/cwd/...) carrying \\n, \\r, \\t or NUL
    home_mismatch     an absolute /home/<user> path whose user is not --home's
    digit_near_miss   a digit-bearing PATH component absent from the context but ONE edit
                      (insert/delete/substitute/transpose) from one present in it --
                      `nibor11896` vs `nibor1896` -- AND the corrected prefix exists on this
                      machine while the produced one does not (filesystem truth); without
                      that confirmation it is info `digit_near_miss_unconfirmed`
  info (recorded, never counted):
    token_near_miss   the same test on any other >=5-char token with >=3 digits (queries,
                      commands, file content). INFO, because on the stored session it fires on
                      legitimate text: `0.185` in a web_search query (context had `0.85`),
                      `1.15t` in a GLSL body -- a version number is not a copy error.
    path_not_in_context, fs_missing (first missing ancestor on this machine), numeric_string,
    truncated (finish length / `_truncated`), no_tool_call (the answer was text only),
    markup_in_content (`<tool_call>` / `<function=` in content: serve's MALFORMED path)

The stored live answer messages[K] is graded with the same grader as round `live` (never in
the totals): a grader that does not flag the live 11896 is a broken grader, and the record
shows it before any engine number is read.

A call is CORRUPT when it carries json_invalid, placeholder, control_char, home_mismatch or
digit_near_miss, and SCHEMA-WRONG when it carries any of the other errors; both counts are in
every record. Summary fields mirror the copy probes so corruption-arms.sh's table reads it
unchanged: lines_total = tool calls graded, lines_with_error = CORRUPT calls,
hex_char_errors = digit_near_miss count, missing = rounds without a tool call,
line_error_rate = the ratio; schema_calls_with_error and error_kinds sit beside them.

LOGPROBS (--top-logprobs N, 2..20; serve #91, architecture 7.11.22): every round asks for
`logprobs: true, top_logprobs: N` -- the RAW model distribution at every generated id,
tool-call markup and arguments included, `token`/`bytes` the raw token text. For every
graded call with an error the probe finds the corrupt span in the concatenated token bytes
(home_mismatch: `/home/<produced user>` vs `/home/<--home user>`; digit_near_miss: the
produced component vs the context's; control_char: the value vs the value without control
characters; placeholder: the value, no correct form), locates the FIRST DIVERGING TOKEN (the
token holding the first byte where produced and correct differ), and prints the tokens around
it with their top-N alternatives and the MARGIN = logprob[chosen] - logprob[best alternative]
(positive: the chosen id was the raw argmax by that many nats; negative: it was not) plus the
margin to the alternative that spells the CORRECT continuation, when one is in the top N.
Every round, erroneous or not, also records the NARROWEST margins inside tool calls, so a
clean round (the placebo arm) shows the same position's margin. All of it lands in the JSON
(`rounds_detail[].logprobs`); the full entry list is not stored.

Usage: corruption-replay-probe.py --preset diorama-0922 --session SNAPSHOT [--rounds 8]
           [--port 8099] [--seed0 0] [--label arm] [--json OUT]
       corruption-replay-probe.py --session S --at K [--at K2 ...] [...]
Options: --top-logprobs N, --no-stream, --sampling JSON (merged over Crow's), --tools-json FILE, --head-file FILE,
         --wire-model NAME,
         --crow-core PATH (default the INSTALLED ~/.local/share/crow/cli/crow_core.py, the
         file the live GUI ran), --home DIR (default ~), --base-url URL (default
         http://127.0.0.1:<port>/v1, Crow's local endpoint), --live-only (grade the stored
         answers, send nothing).
"""
import argparse
import copy
import hashlib
import importlib.util
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

CROW_CORE = os.path.expanduser("~/.local/share/crow/cli/crow_core.py")

# The replay points of record. live_* are serve's own numbers for the SAME request
# (~/.local/state/crow/logs/engine.log, 2026-09-22, the `[chat] prompt` line and the
# `POST ... (body N bytes)` line of the answer that became messages[at]).
PRESETS = {
    "diorama-0922": {
        "session": "~/.local/state/crow/session/session.json",
        "session_sha256": "559bb1ed8e17ec4beecfc9ed2aba33f2538e1b53df446921160019441b3729a8",
        # the head the window SENT, not the one session.json kept (see FIDELITY above)
        "head_file": "corpora/91-replay-diorama-0922-head.txt",
        "head_sha256": "bd462012b971412cf3804d6e04fac08d0e880f94572a311065bdb36a94eddae6",
        "points": [
            {"at": 2, "live_prompt_tokens": 6738, "live_body_bytes": 25539,
             "live_generated": 119,
             "why": "first answer after the rollover: run_command cwd /home/nibor11896/three-staging"},
            {"at": 26, "live_prompt_tokens": 24410, "live_body_bytes": 76183,
             "live_generated": 250,
             "why": "4 parallel calls, the 4th a delegate with parameter_placeholder/context_placeholder"},
            {"at": 69, "live_prompt_tokens": 39273, "live_body_bytes": 128812,
             "live_generated": 1186,
             "why": "write_file path \"/\\n/home/nibor11896/.local/state/crow/...\""},
        ],
    },
}

PATH_KEYS = {"path", "cwd", "dir", "directory", "file", "filename", "root", "target", "dest",
             "destination", "src", "source", "output", "outfile"}
ABS_PATH = re.compile(r"(?<![\w.~:/-])/(?:[\w.@+~-]+/?)+")
DIGIT_TOKEN = re.compile(r"[A-Za-z0-9_.+-]*\d[A-Za-z0-9_.+-]*")
PLACEHOLDER = re.compile(r"(?i)(\b\w*_placeholder\b|\bplaceholder_\w*|^<[^<>\n]{1,40}>$|\{\{[^{}\n]{0,40}\}\}"
                         r"|^(todo|tbd|placeholder|\.\.\.|xxx)$)")
MARKUP = re.compile(r"<tool_call>|</tool_call>|<function=|</function>")
CONTROL = re.compile(r"[\n\r\t\x00]")
# the #91 class: damage to what the model was writing. The rest (unknown_tool, unknown_arg,
# missing_required, type_mismatch) is the schema class -- `old_string`/`new_string` for
# edit_file's `old`/`new` is 22 of the stored session's 302 calls, a habit the model brings
# from other agents' schemas, not a mangled character. Both are counted, separately.
CORRUPTION = {"json_invalid", "placeholder", "control_char", "home_mismatch", "digit_near_miss"}
JSON_TYPES = {"integer": int, "number": (int, float), "boolean": bool,
              "array": list, "object": dict, "string": str}


# ------------------------------------------------------------------ crow's request

def load_crow(path):
    spec = importlib.util.spec_from_file_location("crow_core_replay", path)
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.dirname(path))       # crow_core imports crow_platform beside it
    spec.loader.exec_module(mod)
    return mod


class _Captured(Exception):
    pass


def crow_body(crow, messages, model, wire_model, base_url):
    """The body crow_core.stream_reply builds for this history -- captured, never sent.

    TWO MODEL NAMES, as in the window (crow_gui.py): sampling is resolved from the model the
    server reports (`sampling_for(self._model)`, the display name the session stores), while
    the request's `model` field is the endpoint's (`provider_endpoint(..., args.model)`, whose
    default is DEFAULT_MODEL "crow"). The reasoning budget is resolved inside stream_reply from
    the WIRE name, so "crow" sends no budget fields -- which is what the live bodies measure."""
    got = {}

    def capture(url, body, api_key, timeout, extra=None):
        got["url"], got["body"] = url, body
        raise _Captured()

    saved = crow._post_stream
    crow._post_stream = capture
    try:
        conv = crow.Conversation()
        conv._messages = copy.deepcopy(messages)
        s = crow.sampling_for(model)
        try:
            crow.stream_reply(conv, base_url=base_url, model=wire_model, api_key="",
                              temperature=s["temperature"], top_p=s["top_p"],
                              min_p=s["min_p"], top_k=s.get("top_k"),
                              presence_penalty=s.get("presence_penalty"), timeout=10)
        except _Captured:
            pass
    finally:
        crow._post_stream = saved
    if "body" not in got:
        sys.exit("crow_core.stream_reply returned without reaching _post_stream - body not captured")
    return got["url"], got["body"]


def post(url, body, timeout=3600):
    """(content, tool_calls, finish, usage, logprobs). Streamed like Crow: `index` picks a slot,
    id/name only on a truthy value, `arguments` concatenated raw (crow_core.stream_reply).
    `logprobs` is the concatenation of every `choices[0].logprobs.content` (#91), [] when the
    body did not ask for them."""
    req = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        if not body.get("stream"):
            ans = json.loads(r.read())
            ch = ans["choices"][0]
            msg = ch.get("message") or {}
            calls = [{"id": c.get("id") or "", "name": (c.get("function") or {}).get("name") or "",
                      "arguments": (c.get("function") or {}).get("arguments") or ""}
                     for c in msg.get("tool_calls") or []]
            lps = list(((ch.get("logprobs") or {}).get("content")) or [])
            return msg.get("content") or "", calls, ch.get("finish_reason"), ans.get("usage") or {}, lps
        content, slots, finish, usage, lps = [], {}, None, {}, []
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            try:
                chunk = json.loads(line[6:])
            except json.JSONDecodeError:
                continue
            if chunk.get("usage"):
                usage = chunk["usage"]
            for ch in chunk.get("choices") or []:
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
                lps += ((ch.get("logprobs") or {}).get("content")) or []
                delta = ch.get("delta") or {}
                for call in delta.get("tool_calls") or []:
                    slot = slots.setdefault(call.get("index", 0), {"id": "", "name": "", "arguments": ""})
                    if call.get("id"):
                        slot["id"] = call["id"]
                    fn = call.get("function") or {}
                    if fn.get("name"):
                        slot["name"] = fn["name"]
                    if fn.get("arguments"):
                        slot["arguments"] += fn["arguments"]
                if delta.get("content"):
                    content.append(delta["content"])
        return "".join(content), [slots[i] for i in sorted(slots)], finish, usage, lps


# ------------------------------------------------------------------ logprobs (#91)

def _alts(e):
    return [a for a in e.get("top_logprobs") or [] if bytes(a.get("bytes") or []) != bytes(e.get("bytes") or [])]


def margin(e):
    """logprob[chosen] - logprob[best alternative that is not the chosen id], or None when the
    top list names no other id. By bytes: two ids never share their bytes in this vocabulary
    except for an added token and its spelling, which is the same text either way."""
    alts = _alts(e)
    return round(e["logprob"] - max(a["logprob"] for a in alts), 6) if alts else None


def _tok(e):
    return bytes(e.get("bytes") or []).decode("utf-8", "replace")


def token_view(e, i):
    return {"i": i, "token": _tok(e), "logprob": round(e["logprob"], 6), "margin": margin(e),
            "top": [[bytes(a.get("bytes") or []).decode("utf-8", "replace"), round(a["logprob"], 6)]
                    for a in e.get("top_logprobs") or []]}


def token_offsets(entries):
    """(the concatenated bytes, [start byte of each entry])"""
    starts, buf = [], bytearray()
    for e in entries:
        starts.append(len(buf))
        buf += bytes(e.get("bytes") or [])
    return bytes(buf), starts


def error_pair(e, home):
    """(produced, correct or None) of one grader error, the strings whose first difference is
    the first diverging token; None when the error names no span (json_invalid, schema)."""
    k = e["kind"]
    if k == "home_mismatch":
        return "/home/" + e["user"], "/home/" + e["want"]
    if k == "digit_near_miss":
        return e["got"], e["context"]
    if k == "control_char":
        return e["value"], CONTROL.sub("", e["value"])
    if k == "placeholder":
        return e["value"], None
    return None


def divergence(entries, produced, correct, window=6):
    """The first diverging token of `produced` (vs `correct`) in the generated token bytes:
    {first_diverging_token, margin, correct_alt, window[...]} or None when the span is not
    in the generated text. The LAST occurrence is taken: a path the model echoes in its
    reasoning before the call is not the call."""
    text, starts = token_offsets(entries)
    pb = produced.encode("utf-8")
    at = text.rfind(pb)
    if at < 0 or not pb:
        return None
    cp = 0
    if correct is not None:
        cb = correct.encode("utf-8")
        while cp < min(len(pb), len(cb)) and pb[cp] == cb[cp]:
            cp += 1
    # a produced span that is a strict prefix of the correct one diverges right after it
    div = min(at + cp, len(text) - 1)
    j = max(i for i, s in enumerate(starts) if s <= div)
    e = entries[j]
    out = {"produced": produced, "correct": correct, "byte": div, "token_index": j,
           "first_diverging_token": token_view(e, j), "margin": margin(e),
           "window": [token_view(entries[i], i) for i in range(max(0, j - window), min(len(entries), j + window + 1))]}
    if correct is not None:
        # what the correct text continues with FROM THIS TOKEN'S START
        want = text[starts[j]:div] + correct.encode("utf-8")[cp:]
        ok = [a for a in _alts(e) if bytes(a.get("bytes") or []) and want.startswith(bytes(a["bytes"]))]
        if ok:
            best = max(ok, key=lambda a: a["logprob"])
            out["correct_alt"] = {"token": bytes(best["bytes"]).decode("utf-8", "replace"),
                                  "logprob": round(best["logprob"], 6),
                                  "margin_vs_correct": round(e["logprob"] - best["logprob"], 6)}
        else:
            out["correct_alt"] = None
    return out


def tool_spans(entries):
    """Indices of the entries between `<tool_call>` and `</tool_call>` (markup included)."""
    inside, idx = False, []
    for i, e in enumerate(entries):
        t = _tok(e)
        if "<tool_call>" in t:
            inside = True
        if inside:
            idx.append(i)
        if "</tool_call>" in t:
            inside = False
    return idx


def logprob_report(entries, graded, home, narrowest=8):
    """What a round's logprobs say: one `divergence` per errored call's spannable error, and
    the `narrowest` smallest |margin| tokens inside tool calls (every round, clean or not)."""
    spans = []
    for ci, g in enumerate(graded["calls"]):
        for e in g["errors"]:
            pair = error_pair(e, home)
            if not pair:
                continue
            d = divergence(entries, *pair)
            spans.append(dict(d or {"produced": pair[0], "correct": pair[1], "not_found": True},
                              call=ci, name=g["name"], kind=e["kind"]))
    cand = [(abs(margin(entries[i])), i) for i in tool_spans(entries) if margin(entries[i]) is not None]
    near = [token_view(entries[i], i) for _, i in sorted(cand)[:narrowest]]
    return {"n_tokens": len(entries), "spans": spans, "narrowest_in_tool_calls": near}


def print_report(rep, k, seed, out=sys.stderr):
    for s in rep["spans"]:
        if s.get("not_found"):
            print("  logprobs: call %d %s (%s): %r not in the generated tokens" % (
                s["call"], s["name"], s["kind"], s["produced"]), file=out)
            continue
        f = s["first_diverging_token"]
        ca = s.get("correct_alt")
        print("  logprobs at %d seed %d: call %d %s (%s) %r vs %r: first diverging token #%d %r "
              "logprob %.4f, margin %s%s" % (
                  k, seed, s["call"], s["name"], s["kind"], s["produced"], s["correct"],
                  f["i"], f["token"], f["logprob"],
                  "n/a" if s["margin"] is None else "%+.4f" % s["margin"],
                  (", correct %r logprob %.4f (margin vs correct %+.4f)" % (
                      ca["token"], ca["logprob"], ca["margin_vs_correct"])) if ca else
                  (", correct continuation not in the top list" if s["correct"] is not None else "")),
              file=out)
        for t in s["window"]:
            print("    %s#%-4d %-18r %9.4f  margin %-9s top %s" % (
                ">" if t["i"] == f["i"] else " ", t["i"], t["token"], t["logprob"],
                "n/a" if t["margin"] is None else "%+.4f" % t["margin"],
                ", ".join("%r %.3f" % (a, lp) for a, lp in t["top"])), file=out)
    if rep["narrowest_in_tool_calls"]:
        print("  logprobs at %d seed %d: narrowest margins in tool calls: %s" % (
            k, seed, ", ".join("#%d %r %+.4f" % (t["i"], t["token"], t["margin"])
                               for t in rep["narrowest_in_tool_calls"][:5])), file=out)


# ------------------------------------------------------------------ the grader

def _strings(v):
    if isinstance(v, str):
        yield v
    elif isinstance(v, dict):
        for x in v.values():
            yield from _strings(x)
    elif isinstance(v, list):
        for x in v:
            yield from _strings(x)


def context_index(messages):
    """What the model could have copied from: every absolute path, path component and
    digit-bearing token in messages[0:K] -- contents and earlier tool-call arguments."""
    texts = []
    for m in messages:
        c = m.get("content")
        if isinstance(c, str):
            texts.append(c)
        elif isinstance(c, list):
            texts += [p.get("text") or "" for p in c if isinstance(p, dict)]
        for tc in m.get("tool_calls") or []:
            texts.append((tc.get("function") or {}).get("arguments") or "")
    paths, comps, tokens = set(), set(), set()
    for t in texts:
        for p in ABS_PATH.findall(t):
            p = p.rstrip("/.")
            paths.add(p)
            comps.update(x for x in p.split("/") if x)
        tokens.update(DIGIT_TOKEN.findall(t))
    return {"paths": paths, "components": comps, "tokens": tokens}


def edit1(a, b):
    """The one-edit class of a vs b (a = produced, b = context), or None when further apart."""
    if a == b:
        return None
    la, lb = len(a), len(b)
    if la == lb + 1:
        for i in range(la):
            if a[:i] + a[i + 1:] == b:
                return "insertion"
        return None
    if la + 1 == lb:
        for i in range(lb):
            if b[:i] + b[i + 1:] == a:
                return "deletion"
        return None
    if la != lb:
        return None
    diff = [i for i in range(la) if a[i] != b[i]]
    if len(diff) == 1:
        return "substitution"
    if len(diff) == 2 and diff[1] == diff[0] + 1 and a[diff[0]] == b[diff[1]] and a[diff[1]] == b[diff[0]]:
        return "transposition"
    return None


def near_miss(tok, pool):
    if tok in pool:
        return None
    for ref in pool:
        if abs(len(ref) - len(tok)) <= 1:
            kind = edit1(tok, ref)
            if kind:
                return {"got": tok, "context": ref, "kind": kind}
    return None


def fs_missing(path):
    """First ancestor of `path` that does not exist on this machine, or None."""
    cur = ""
    for part in [x for x in path.split("/") if x]:
        cur += "/" + part
        if not os.path.lexists(cur):
            return cur
    return None


def grade_call(call, tools, ctx, home):
    """One tool call -> {"name", "errors": [...], "info": [...]}."""
    errors, info = [], []
    name, raw = call.get("name") or "", call.get("arguments") or ""
    decl = {t["function"]["name"]: t["function"].get("parameters") or {} for t in tools}
    if name not in decl:
        errors.append({"kind": "unknown_tool", "name": name})
    params = decl.get(name) or {}
    props, required = params.get("properties") or {}, params.get("required") or []
    try:
        args = json.loads(raw) if raw.strip() else {}
    except json.JSONDecodeError as exc:
        errors.append({"kind": "json_invalid", "error": str(exc), "raw_tail": raw[-120:]})
        return {"name": name, "errors": errors, "info": info}
    if not isinstance(args, dict):
        errors.append({"kind": "json_invalid", "error": "arguments is %s, not an object" % type(args).__name__})
        return {"name": name, "errors": errors, "info": info}
    if args.get("_truncated"):
        info.append({"kind": "truncated"})
    if name in decl:
        for k in args:
            if k not in props and k != "_truncated":
                errors.append({"kind": "unknown_arg", "arg": k})
        if not args.get("_truncated"):
            for k in required:
                if k not in args:
                    errors.append({"kind": "missing_required", "arg": k})
        for k, v in args.items():
            want = (props.get(k) or {}).get("type")
            if want in JSON_TYPES and want != "string":
                ok = isinstance(v, JSON_TYPES[want]) and not (want in ("integer", "number") and isinstance(v, bool))
                if not ok:
                    if want in ("integer", "number") and isinstance(v, str) and re.fullmatch(r"-?\d+(\.\d+)?", v.strip()):
                        info.append({"kind": "numeric_string", "arg": k, "value": v})
                    else:
                        errors.append({"kind": "type_mismatch", "arg": k, "want": want,
                                       "got": type(v).__name__})
    user = os.path.basename(home.rstrip("/"))
    for k, v in args.items():
        for s in _strings(v):
            m = PLACEHOLDER.search(s.strip())
            if m:
                errors.append({"kind": "placeholder", "arg": k, "value": m.group(0)})
            if k.lower() in PATH_KEYS and CONTROL.search(s):
                errors.append({"kind": "control_char", "arg": k, "value": s[:120]})
            cands = ABS_PATH.findall(s)
            if k.lower() in PATH_KEYS and s.strip().startswith("/"):
                cands.append(CONTROL.sub("", s.strip()))
            seen_paths = set()
            for p in cands:
                p = re.sub(r"/+", "/", p).rstrip("/.")
                if not p or p in seen_paths:
                    continue
                seen_paths.add(p)
                parts = [x for x in p.split("/") if x]
                if len(parts) >= 2 and parts[0] == "home" and parts[1] != user:
                    errors.append({"kind": "home_mismatch", "arg": k, "path": p,
                                   "user": parts[1], "want": user,
                                   "edit": edit1(parts[1], user)})
                for i, comp in enumerate(parts):
                    if re.search(r"\d", comp):
                        nm = near_miss(comp, ctx["components"])
                        if nm:
                            # FILESYSTEM TRUTH decides: the corrected prefix exists and the
                            # produced one does not. `/tmp/shot13` next to a context
                            # `/tmp/shot12` is a counter, not a copy error (23 of 24
                            # candidates on the stored session were of that kind).
                            got = "/" + "/".join(parts[:i + 1])
                            fixed = "/" + "/".join(parts[:i] + [nm["context"]])
                            hit = dict(nm, arg=k, path=p, edit=nm["kind"], fixed_prefix=fixed)
                            if os.path.lexists(fixed) and not os.path.lexists(got):
                                errors.append(dict(hit, kind="digit_near_miss"))
                            else:
                                info.append(dict(hit, kind="digit_near_miss_unconfirmed"))
                if p not in ctx["paths"] and not any(p.startswith(c + "/") for c in ctx["paths"]):
                    info.append({"kind": "path_not_in_context", "arg": k, "path": p})
                miss = fs_missing(p)
                if miss:
                    info.append({"kind": "fs_missing", "arg": k, "path": p, "first_missing": miss})
            if k.lower() not in PATH_KEYS:
                inpath = {c for p in seen_paths for c in p.split("/")}
                for tok in set(DIGIT_TOKEN.findall(s)):
                    if len(tok) >= 5 and sum(ch.isdigit() for ch in tok) >= 3 and tok not in inpath:
                        nm = near_miss(tok, ctx["tokens"])
                        if nm:
                            info.append(dict(nm, kind="token_near_miss", arg=k, edit=nm["kind"]))
    # one error per (kind, arg, got/path): a path seen as `path` and inside a command counts once
    uniq, keys = [], set()
    for e in errors:
        key = (e["kind"], e.get("arg"), e.get("got") or e.get("path") or e.get("value"))
        if key not in keys:
            keys.add(key)
            uniq.append(e)
    return {"name": name, "errors": uniq, "info": info}


def grade_answer(content, calls, finish, tools, ctx, home):
    graded = [grade_call(c, tools, ctx, home) for c in calls]
    notes = []
    if not calls:
        notes.append({"kind": "no_tool_call"})
    if finish == "length":
        notes.append({"kind": "truncated", "finish": finish})
    if MARKUP.search(content or ""):
        notes.append({"kind": "markup_in_content"})
    for g in graded:
        kinds = {e["kind"] for e in g["errors"]}
        g["corrupt"] = bool(kinds & CORRUPTION)
        g["schema"] = bool(kinds - CORRUPTION)
    return {"calls": graded, "n_calls": len(calls),
            "calls_with_error": sum(1 for g in graded if g["corrupt"]),
            "calls_with_schema_error": sum(1 for g in graded if g["schema"]),
            "digit_near_miss": sum(1 for g in graded for e in g["errors"] if e["kind"] == "digit_near_miss"),
            "notes": notes}


def stored_answer(msg):
    calls = [{"id": tc.get("id") or "", "name": (tc.get("function") or {}).get("name") or "",
              "arguments": (tc.get("function") or {}).get("arguments") or ""}
             for tc in msg.get("tool_calls") or []]
    return msg.get("content") or "", calls


def _short(calls):
    return [{"name": c["name"], "arguments": c["arguments"][:400] +
             ("...(%d chars)" % len(c["arguments"]) if len(c["arguments"]) > 400 else "")}
            for c in calls]


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--session", help="Crow session.json (a SNAPSHOT: Crow rewrites the live file on exit)")
    ap.add_argument("--at", type=int, action="append", default=[],
                    help="replay messages[0:K] (repeatable); the model answers message K")
    ap.add_argument("--preset", choices=sorted(PRESETS))
    ap.add_argument("--rounds", type=int, default=8, help="seeds per replay point")
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--seed0", type=int, default=0,
                    help="first seed; 0 = serve's data-sheet seed, what the live request ran with")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--sampling", default="{}", help="JSON merged over the body Crow built")
    ap.add_argument("--tools-json", default=None, help="a captured tools array to send instead of Crow's")
    ap.add_argument("--crow-core", default=CROW_CORE)
    ap.add_argument("--head-file", default=None,
                    help="system text to send as messages[0] instead of the stored one (a preset names its own)")
    ap.add_argument("--wire-model", default=None,
                    help="the request's `model` field (default: crow_core.DEFAULT_MODEL, the window's --model default)")
    ap.add_argument("--home", default=os.path.expanduser("~"))
    ap.add_argument("--no-stream", action="store_true")
    ap.add_argument("--top-logprobs", type=int, default=None, metavar="N",
                    help="ask serve for logprobs + N alternatives (2..20, #91) and report the margin "
                         "at the first diverging token of every corrupt call")
    ap.add_argument("--live-only", action="store_true", help="grade the stored answers only, send nothing")
    ap.add_argument("--allow-session-drift", action="store_true",
                    help="run a preset against a session whose sha256 differs from the preset's")
    ap.add_argument("--json", default="-")
    args = ap.parse_args()
    if args.top_logprobs is not None and not 2 <= args.top_logprobs <= 20:
        sys.exit("--top-logprobs N: 2..20 (a margin needs one alternative beside the chosen id)")

    preset = PRESETS.get(args.preset) if args.preset else None
    session = os.path.expanduser(args.session or (preset or {}).get("session") or "")
    if not session:
        sys.exit("--session (or --preset) is required")
    with open(session, "rb") as fh:
        blob = fh.read()
    sha = hashlib.sha256(blob).hexdigest()
    if preset and sha != preset["session_sha256"] and not args.allow_session_drift:
        sys.exit("session %s has sha256 %s, the preset was measured on %s - replay a snapshot "
                 "of the preset's file (or --allow-session-drift)" % (session, sha[:16],
                                                                       preset["session_sha256"][:16]))
    doc = json.loads(blob)
    messages, model = doc["messages"], doc.get("model") or ""
    head_file = args.head_file or ((preset or {}).get("head_file") and
                                   os.path.join(os.path.dirname(os.path.abspath(__file__)), preset["head_file"]))
    head_sha = None
    if head_file:
        with open(head_file, encoding="utf-8") as fh:
            head = fh.read()
        head_sha = hashlib.sha256(head.encode("utf-8")).hexdigest()
        if preset and not args.head_file and head_sha != preset["head_sha256"]:
            sys.exit("head file %s has sha256 %s, the preset pins %s" % (head_file, head_sha[:16],
                                                                        preset["head_sha256"][:16]))
        if messages[0].get("role") != "system":
            sys.exit("--head-file: the session has no system message to replace")
        messages = [dict(messages[0], content=head)] + messages[1:]
    points = [dict(p) for p in preset["points"]] if preset else []
    points += [{"at": k} for k in args.at if k not in {p["at"] for p in points}]
    if not points:
        sys.exit("no replay point: give --at K or --preset")
    for p in points:
        k = p["at"]
        if not 0 < k < len(messages) + 1 or messages[k - 1]["role"] == "assistant":
            sys.exit("--at %d: messages[0:%d] must end on a user or tool turn (session has %d)" % (
                k, k, len(messages)))

    crow = load_crow(args.crow_core)
    wire_model = args.wire_model or crow.DEFAULT_MODEL
    base_url = args.base_url or "http://127.0.0.1:%d/v1" % args.port
    tools_override = None
    if args.tools_json:
        with open(args.tools_json) as fh:
            tools_override = json.load(fh)
    extra = json.loads(args.sampling)
    with open(args.crow_core, "rb") as fh:
        crow_sha = hashlib.sha256(fh.read()).hexdigest()
    info = {"session": session, "session_sha256": sha, "model": model, "wire_model": wire_model,
            "head_file": head_file, "head_sha256": head_sha, "home": args.home,
            "crow_core": args.crow_core,
            "crow_core_sha256": crow_sha,
            "tools_source": ("--tools-json %s" % args.tools_json) if tools_override is not None
            else "crow_core.TOOLS via stream_reply", "stream": not args.no_stream,
            "sampling_override": extra, "top_logprobs": args.top_logprobs}

    rounds, per_point = [], []
    for p in points:
        k = p["at"]
        hist = messages[:k]
        url, body = crow_body(crow, hist, model, wire_model, base_url)
        if tools_override is not None:
            body["tools"] = tools_override
        body.update(extra)
        if args.no_stream:
            body["stream"] = False
            body.pop("stream_options", None)
        if args.top_logprobs is not None:
            body["logprobs"] = True
            body["top_logprobs"] = args.top_logprobs
        ctx = context_index(hist)
        tools = body.get("tools") or []
        nbytes = len(json.dumps(body).encode("utf-8"))
        p.update({"body_bytes": nbytes, "url": url,
                  "body_bytes_delta_vs_live": nbytes - p["live_body_bytes"] if p.get("live_body_bytes") else None,
                  "tools": len(tools),
                  "tools_sha256": hashlib.sha256(json.dumps(tools, sort_keys=True).encode()).hexdigest()[:16],
                  "sampling": {f: body.get(f) for f in ("temperature", "top_p", "min_p", "top_k",
                                                        "presence_penalty", "max_tokens")}})
        if k < len(messages) and messages[k]["role"] == "assistant":
            c, calls = stored_answer(messages[k])
            live = grade_answer(c, calls, None, tools, ctx, args.home)
            live["reply"] = _short(calls)
            p["live_graded"] = live
            print("point %d live answer: %d call(s), %d corrupt %s" % (
                k, live["n_calls"], live["calls_with_error"],
                sorted({e["kind"] for g in live["calls"] for e in g["errors"]})), file=sys.stderr)
        per_point.append(p)
        if args.live_only:
            continue
        for r in range(args.rounds):
            seed = args.seed0 + r
            b = dict(body, seed=seed)
            t0 = time.time()
            try:
                content, calls, finish, usage, lps = post(url, b)
            except urllib.error.HTTPError as exc:
                rounds.append({"at": k, "seed": seed, "error": "HTTP %d: %s" % (exc.code, exc.read()[:300])})
                continue
            except Exception as exc:                      # a failed round is a failed round
                rounds.append({"at": k, "seed": seed, "error": repr(exc)})
                continue
            g = grade_answer(content, calls, finish, tools, ctx, args.home)
            ptok = usage.get("prompt_tokens")
            g.update({"at": k, "seed": seed, "seconds": round(time.time() - t0, 1), "finish": finish,
                      "prompt_tokens": ptok,
                      "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
                      "completion_tokens": usage.get("completion_tokens"),
                      "prompt_tokens_delta_vs_live": (ptok - p["live_prompt_tokens"])
                      if ptok is not None and p.get("live_prompt_tokens") else None,
                      "content_chars": len(content), "reply": _short(calls)})
            if args.top_logprobs is not None:
                g["logprobs"] = logprob_report(lps, g, args.home)
            rounds.append(g)
            print("at %d seed %d: %d call(s), %d corrupt %s, prompt %s tok (%s cached), %.0f s" % (
                k, seed, g["n_calls"], g["calls_with_error"],
                sorted({e["kind"] for c in g["calls"] for e in c["errors"]}), ptok,
                g["cached_tokens"], g["seconds"]), file=sys.stderr)
            if "logprobs" in g:
                print_report(g["logprobs"], k, seed)

    ok = [x for x in rounds if "error" not in x]
    kinds = {}
    for x in ok:
        for c in x["calls"]:
            for e in c["errors"]:
                kinds[e["kind"]] = kinds.get(e["kind"], 0) + 1
    for p in per_point:
        pr = [x for x in ok if x["at"] == p["at"]]
        p.update({"rounds_ok": len(pr), "calls": sum(x["n_calls"] for x in pr),
                  "calls_with_error": sum(x["calls_with_error"] for x in pr),
                  "calls_with_schema_error": sum(x["calls_with_schema_error"] for x in pr),
                  "rounds_with_error": sum(1 for x in pr if x["calls_with_error"])})
    calls_total = sum(x["n_calls"] for x in ok)
    bad = sum(x["calls_with_error"] for x in ok)
    expected = 0 if args.live_only else args.rounds * len(points)
    summary = {
        "label": args.label, "rounds": expected, "rounds_ok": len(ok),
        "complete": bool(expected) and len(ok) == expected,
        "lines_total": calls_total, "lines_with_error": bad,
        "hex_char_errors": sum(x["digit_near_miss"] for x in ok),
        "missing": sum(1 for x in ok if not x["n_calls"]),
        "line_error_rate": round(bad / calls_total, 4) if calls_total else None,
        "char_error_rate": None,
        "rounds_with_error": sum(1 for x in ok if x["calls_with_error"]),
        "schema_calls_with_error": sum(x["calls_with_schema_error"] for x in ok),
        "error_kinds": kinds,
        **info,
        "points": per_point,
        "rounds_detail": rounds,
    }
    out = json.dumps(summary, indent=1)
    if args.json == "-":
        print(out)
    else:
        with open(args.json, "w") as fh:
            fh.write(out + "\n")
    print("%s replay @ %s: %d/%d calls corrupt (line_error_rate=%s), %d digit near-miss, "
          "%d schema-wrong, %d rounds without a call, kinds %s (%d rounds)" % (
              args.label, ",".join(str(p["at"]) for p in points), bad, calls_total,
              summary["line_error_rate"], summary["hex_char_errors"],
              summary["schema_calls_with_error"], summary["missing"], kinds, len(ok)), file=sys.stderr if args.json == "-" else sys.stdout)


if __name__ == "__main__":
    sys.exit(main())
