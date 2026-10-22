#!/usr/bin/env python3
"""The head anchor (`CROW_PREFIX_ANCHOR=1`), live against a running `serve`: a new chat with the same
system prompt rolls back onto the anchor instead of going cold, and answers as without it.

  python -I tools/head_anchor_check.py --port 8199 --out run.json [--expect other.json]
                                       [--doc docs/env.md] [--chars 24000] [--tokens 64]

Two chats, greedy, thinking off, one long system prompt (the first `--chars` characters of `--doc`):
  A  system + "Name the first variable in this table." (cold: with the switch on, the anchor is
     taken at the first user turn during this prefill)
  B  the same system + another question, a NEW chat (with the switch on: WARM from the anchor,
     `cached_tokens` = the anchor position; without it: COLD)
Writes both answers, B's `cached_tokens` and both prefill walls to `--out`. With `--expect FILE`
(the output of a run against a serve WITHOUT the switch, booted the same way), exits 1 unless both
answers are the same as there. Standard library only.
"""

import argparse
import json
import sys
import time
import urllib.request


def ask(url, system, question, tokens):
    body = {"messages": [{"role": "system", "content": system}, {"role": "user", "content": question}],
            "max_tokens": tokens, "temperature": 0, "stream": False,
            "chat_template_kwargs": {"enable_thinking": False}}
    started = time.monotonic()
    request = urllib.request.Request(url, data=json.dumps(body).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as resp:
        doc = json.load(resp)
    return {"content": doc["choices"][0]["message"].get("content"),
            "prompt_tokens": doc["usage"]["prompt_tokens"],
            "cached_tokens": doc["usage"]["prompt_tokens_details"]["cached_tokens"],
            "prompt_ms": doc.get("timings", {}).get("prompt_ms"),
            "wall_s": round(time.monotonic() - started, 2)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--expect", default=None)
    ap.add_argument("--doc", default="docs/env.md")
    ap.add_argument("--chars", type=int, default=24000)
    ap.add_argument("--tokens", type=int, default=64)
    a = ap.parse_args()
    url = "http://127.0.0.1:%d/v1/chat/completions" % a.port
    with open(a.doc, encoding="utf-8") as fh:
        system = "You answer questions about this reference.\n\n" + fh.read()[:a.chars]
    run = {"A": ask(url, system, "Name the first variable in this table.", a.tokens),
           "B": ask(url, system, "Which variable turns the prefix cache off? One sentence.", a.tokens)}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(run, fh, indent=1)
    for k in "AB":
        r = run[k]
        print("%s  prompt %d tok, cached %d, prompt_ms %s, wall %.2f s: %r"
              % (k, r["prompt_tokens"], r["cached_tokens"], r["prompt_ms"], r["wall_s"], (r["content"] or "")[:80]))
    if a.expect:
        with open(a.expect, encoding="utf-8") as fh:
            ref = json.load(fh)
        same = all(run[k]["content"] == ref[k]["content"] for k in "AB")
        print("answers: %s" % ("the same as %s" % a.expect if same else "DIFFERENT from %s" % a.expect))
        return 0 if same else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
