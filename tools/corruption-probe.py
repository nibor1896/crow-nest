#!/usr/bin/env python3
"""#91 corruption probe -- the digit-fidelity instrument.

MEASURES THE CLASS THAT KILLED EVERY LONG SESSION SO FAR: trailing-digit and
token-level damage in exact-literal reproduction (`df5fe00` for `df5fe09`,
`[0,1,2,0,2,33]` for `[0,1,2,0,2,3]`, `1..05`, `0x46566b,,`). The subject copies
seeded 6-hex-digit literals back; every character that differs is one count.
An arm that lowers the rate is an arm that removes the failure at its source.

Usage: corruption-probe.py [--rounds 8] [--port 8099] [--seed0 1] [--json OUT]
One JSON line per round to OUT (or stdout), then a summary object.
"""
import argparse
import json
import random
import re
import sys
import time
import urllib.request

API = "http://127.0.0.1:%d/v1/chat/completions"
LINES = 40          # literals per round; 8 rounds x 40 = 320 per arm
HEXCHARS = "0123456789abcdef"

# the pinned operating point of both flash-next arms (manifest, 2026-09-21)
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.01}


def literals(rng):
    out = []
    for i in range(LINES):
        hexv = "".join(rng.choice(HEXCHARS) for _ in range(6))
        out.append("const id_%03d = 0x%s; // slot %03d" % (i, hexv, i))
    return out


def prompt(lines):
    return (
        "Reproduce the block below EXACTLY, inside one fenced code block, "
        "nothing else -- no thinking, no commentary. Every identifier, every "
        "hex digit, every comma and semicolon must match character for "
        "character. Copy, do not re-derive.\n\n```\n" + "\n".join(lines) + "\n```"
    )


def call(port, seed, text, max_tokens=1600):
    body = {
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "seed": seed,
        "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "minimal"},
        **SAMPLING,
    }
    req = urllib.request.Request(
        API % port, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        return json.loads(r.read())


def grade(reply, want):
    m = re.search(r"```[a-z]*\n(.*?)```", reply, re.S)
    got = m.group(1) if m else reply
    got_pairs = dict(re.findall(r"const (id_\d+) = 0x([0-9a-fA-F]+);", got))
    char_err = miss = extra = 0
    bad_lines = 0
    for i, line in enumerate(want):
        name = "id_%03d" % i
        hexv = re.search(r"0x([0-9a-f]+)", line).group(1)
        if name not in got_pairs:
            miss += 1
            bad_lines += 1
            continue
        g = got_pairs[name].lower()
        if g != hexv:
            bad_lines += 1
            for a, b in zip(hexv, g):
                char_err += a != b
            char_err += abs(len(hexv) - len(g))
    extra = max(0, len(got_pairs) - LINES)
    return {"lines": LINES, "missing": miss, "extra": extra,
            "lines_with_error": bad_lines, "hex_char_errors": char_err}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--label", default="arm")
    ap.add_argument("--json", default="-")
    args = ap.parse_args()

    rounds = []
    for r in range(args.rounds):
        seed = args.seed0 + r
        want = literals(random.Random(seed))
        t0 = time.time()
        try:
            ans = call(args.port, seed, prompt(want))
            reply = ans["choices"][0]["message"]["content"] or ""
        except Exception as exc:                      # a failed round is a failed round
            rounds.append({"seed": seed, "error": repr(exc)})
            continue
        g = grade(reply, want)
        g.update({"seed": seed, "seconds": round(time.time() - t0, 1),
                  "reply_chars": len(reply)})
        rounds.append(g)

    ok = [x for x in rounds if "error" not in x]
    tot_lines = sum(x["lines"] for x in ok)
    summary = {
        "label": args.label, "rounds": args.rounds, "rounds_ok": len(ok),
        "lines_total": tot_lines,
        "lines_with_error": sum(x["lines_with_error"] for x in ok),
        "hex_char_errors": sum(x["hex_char_errors"] for x in ok),
        "missing": sum(x["missing"] for x in ok),
        "line_error_rate": round(sum(x["lines_with_error"] for x in ok) / tot_lines, 4) if tot_lines else None,
        "char_error_rate": round(sum(x["hex_char_errors"] for x in ok) / (tot_lines * 6), 6) if tot_lines else None,
        "rounds_detail": rounds,
    }
    out = json.dumps(summary, indent=1)
    if args.json == "-":
        print(out)
    else:
        with open(args.json, "w") as fh:
            fh.write(out + "\n")
        print("%s: line_error_rate=%s char_error_rate=%s (%d rounds, %d lines)" % (
            args.label, summary["line_error_rate"], summary["char_error_rate"],
            len(ok), tot_lines))


if __name__ == "__main__":
    sys.exit(main())
