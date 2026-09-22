#!/usr/bin/env python3
"""#91 corruption probe, LONG CONTEXT -- the same instrument at session depth.

WHY: the short probe (corruption-probe.py, 984 prompt tokens) put all five arms
on its floor (2026-09-21: 1/320 = 0.0031 for baseline, placebo and three BF16
arms, the same seed-2 event everywhere). The corruption of record appears from
~100k tokens. This probe asks for the SAME copy of the SAME seeded literals and
grades it with the SAME grade() (imported, one definition) -- the only thing
that changes is how much session sits in front of it.

Shape of one run:
  1. calibrate: a 400-line filler sample, max_tokens 1 -> tokens per line from
     usage.prompt_tokens (no tokenizer on the python side, serve counts).
  2. warm-up: [user: filler] -> the one big prefill; serve snapshots after the
     prompt (#31 A9), so the rounds CAN resume there instead of re-prefilling.
     Whether they do is reported per round (cached_tokens), never assumed.
  3. rounds: [user: filler, assistant: ack, user: copy task], seed = seed0 + r.

The filler is seeded session noise (gate lines, shas, tool results, code) --
digit-rich on purpose, but never `const id_NNN = 0x...`, so the task stays
unambiguous and the grader cannot pick up a filler line.

--position end    literals in the final turn, behind the filler (default; the
                  pure depth effect, same filler every round -> cacheable)
--position start  literals at the TOP of the filler turn, the task at the end
                  (copy across the whole distance; nothing cacheable)
--ctx-tokens 0    no filler at all: byte-for-byte the short probe's request,
                  the control that must reproduce its numbers.

Usage: corruption-probe-long.py --ctx-tokens 100000 [--rounds 8] [--port 8099]
           [--seed0 1] [--filler-seed 91] [--position end|start] [--json OUT]
The summary object carries the short probe's fields (corruption-arms.sh's
table reads it unchanged) plus the context readings.
"""
import argparse
import importlib.util
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request

_here = os.path.dirname(os.path.abspath(__file__))
_spec = importlib.util.spec_from_file_location(
    "corruption_probe", os.path.join(_here, "corruption-probe.py"))
short = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(short)

N_CTX = 200_000          # serve's context (slot.rs); prompt ids >= n_ctx is a 413
REPLY_BUDGET = 1600      # the short probe's max_tokens
ACK = "OK"
CAL_LINES = 400

WORDS = ("gate parity tier pinned pool balloon serve prefill decode overlay tensor "
         "layer expert router shared down proj attn ring snapshot rollback budget "
         "margin scope cgroup driver refcount planner residency trickle swap hot cold "
         "bf16 cnq container sha manifest clippy test doc guard item arm probe").split()
FILES = ("engine/src/gen.rs engine/src/manager.rs engine/src/residency.rs "
         "engine/src/cache.rs engine/src/bin/serve.rs converter/layer_rule_overlay.rs "
         "tools/gate-linux.sh tools/serve-linux.sh crow_core.py").split()


def _hex(rng, n):
    return "".join(rng.choice("0123456789abcdef") for _ in range(n))


def filler_line(rng):
    k = rng.randrange(6)
    if k == 0:
        return "[%02d:%02d:%02d] gate item %s%d: sha %s ok, %d tests, %d failed, %.1f tok/s" % (
            rng.randrange(24), rng.randrange(60), rng.randrange(60), rng.choice(WORDS),
            rng.randrange(1, 513), _hex(rng, 12), rng.randrange(200, 400), rng.randrange(3),
            rng.uniform(40, 90))
    if k == 1:
        a = rng.randrange(1, 4000)
        return "tool_result: read %s lines %d-%d (%d lines)" % (
            rng.choice(FILES), a, a + rng.randrange(10, 80), rng.randrange(10, 80))
    if k == 2:
        n = rng.randrange(48)
        return "fn layer_%d_%s(x: &Tensor) -> Tensor { x.matmul(&w_%d).add(&b_%d) } // rev %s" % (
            n, rng.choice(WORDS), n, n, _hex(rng, 7))
    if k == 3:
        return "commit %s %s: %s" % (_hex(rng, 7), rng.choice(WORDS),
                                     " ".join(rng.choice(WORDS) for _ in range(rng.randrange(6, 14))))
    if k == 4:
        return "MemAvailable %d GiB after %d s, pass %d, budget %.2f GiB, offset 0x%s, ids [%s]" % (
            rng.randrange(3, 58), rng.randrange(600), rng.randrange(1, 7), rng.uniform(8, 47),
            _hex(rng, 8), ",".join(str(rng.randrange(0, 40)) for _ in range(6)))
    return " ".join(rng.choice(WORDS) for _ in range(rng.randrange(10, 22))).capitalize() + "."


def filler(seed, n_lines):
    rng = random.Random(seed)
    return "\n".join(filler_line(rng) for _ in range(n_lines))


def post(port, messages, seed, max_tokens):
    body = {
        "messages": messages, "max_tokens": max_tokens, "seed": seed, "stream": False,
        "chat_template_kwargs": {"reasoning_effort": "none"},   # see corruption-probe.py
        **short.SAMPLING,
    }
    req = urllib.request.Request(
        short.API % port, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    # a 100k+ prefill is minutes, not the short probe's seconds
    with urllib.request.urlopen(req, timeout=3600) as r:
        return json.loads(r.read())


def usage_of(ans):
    u = ans.get("usage") or {}
    return (u.get("prompt_tokens"),
            (u.get("prompt_tokens_details") or {}).get("cached_tokens"))


SESSION_HEAD = ("Session log so far, for context. Read it; do not act on it. "
                "Reply with exactly: %s\n\n" % ACK)
START_TASK = ("Reproduce the block between BEGIN-LITERALS and END-LITERALS from the top of "
              "this conversation EXACTLY, inside one fenced code block, nothing else -- no "
              "thinking, no commentary. Every identifier, every hex digit, every comma and "
              "semicolon must match character for character. Copy, do not re-derive.")


def build(position, fill, want):
    if not fill:                                   # --ctx-tokens 0: the short probe's request
        return [{"role": "user", "content": short.prompt(want)}]
    if position == "end":
        first, last = SESSION_HEAD + fill, short.prompt(want)
    else:
        first = (SESSION_HEAD + "BEGIN-LITERALS\n" + "\n".join(want) + "\nEND-LITERALS\n\n" + fill)
        last = START_TASK
    return [{"role": "user", "content": first},
            {"role": "assistant", "content": ACK},
            {"role": "user", "content": last}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx-tokens", type=int, required=True,
                    help="target prompt tokens in front of the task (0 = the short probe's request)")
    ap.add_argument("--rounds", type=int, default=8)
    ap.add_argument("--port", type=int, default=8099)
    ap.add_argument("--seed0", type=int, default=1)
    ap.add_argument("--filler-seed", type=int, default=91)
    ap.add_argument("--position", choices=("end", "start"), default="end")
    ap.add_argument("--label", default="arm")
    ap.add_argument("--sampling", default="{}",
                    help='JSON merged over the short probe\'s SAMPLING, e.g. \'{"presence_penalty": 0}\' '
                         '(what serve fills in from the data sheet when absent: presence_penalty 1.5)')
    ap.add_argument("--json", default="-")
    args = ap.parse_args()
    short.SAMPLING.update(json.loads(args.sampling))   # post() reads it through short.SAMPLING

    # room for the literal block (~1k tokens), the reply and the template
    ceiling = N_CTX - REPLY_BUDGET - 3000
    if args.ctx_tokens > ceiling:
        sys.exit("--ctx-tokens %d over the usable context (%d of n_ctx %d)" % (
            args.ctx_tokens, ceiling, N_CTX))

    info = {"ctx_tokens_target": args.ctx_tokens, "position": args.position,
            "filler_seed": args.filler_seed, "sampling": dict(short.SAMPLING)}
    fill = ""
    if args.ctx_tokens > 0:
        cal = post(args.port, [{"role": "user", "content": filler(args.filler_seed, CAL_LINES)}],
                   args.seed0, 1)
        cal_tok, _ = usage_of(cal)
        if not cal_tok:
            sys.exit("calibration: serve returned no usage.prompt_tokens - cannot size the filler")
        per_line = cal_tok / CAL_LINES
        n_lines = max(1, int(args.ctx_tokens / per_line))
        fill = filler(args.filler_seed, n_lines)
        info.update({"tokens_per_filler_line": round(per_line, 2), "filler_lines": n_lines})
        print("calibration: %.2f tok/line -> %d filler lines for ~%d tokens" % (
            per_line, n_lines, args.ctx_tokens), file=sys.stderr)
        if args.position == "end":
            t0 = time.time()
            try:
                warm = post(args.port, build("end", fill, [])[:1], args.seed0, 4)
                info["warmup"] = {"prompt_tokens": usage_of(warm)[0],
                                  "seconds": round(time.time() - t0, 1)}
            except Exception as exc:               # the rounds still run, they just prefill cold
                info["warmup"] = {"error": repr(exc)}
            print("warm-up: %s" % info["warmup"], file=sys.stderr)

    rounds = []
    for r in range(args.rounds):
        seed = args.seed0 + r
        want = short.literals(random.Random(seed))
        t0 = time.time()
        try:
            ans = post(args.port, build(args.position, fill, want), seed, REPLY_BUDGET)
            reply = ans["choices"][0]["message"]["content"] or ""
        except urllib.error.HTTPError as exc:
            rounds.append({"seed": seed, "error": "HTTP %d: %s" % (exc.code, exc.read()[:300])})
            continue
        except Exception as exc:                      # a failed round is a failed round
            rounds.append({"seed": seed, "error": repr(exc)})
            continue
        g = short.grade(reply, want)
        ptok, cached = usage_of(ans)
        g.update({"seed": seed, "seconds": round(time.time() - t0, 1), "reply_chars": len(reply),
                  "prompt_tokens": ptok, "cached_tokens": cached,
                  "finish": ans["choices"][0].get("finish_reason")})
        rounds.append(g)
        print("round seed %d: %d/%d bad lines, %d hex errs, prompt %s tok (%s cached), %.0f s" % (
            seed, g["lines_with_error"], g["lines"], g["hex_char_errors"], ptok, cached,
            g["seconds"]), file=sys.stderr)

    ok = [x for x in rounds if "error" not in x]
    tot_lines = sum(x["lines"] for x in ok)
    ptoks = [x["prompt_tokens"] for x in ok if x.get("prompt_tokens")]
    summary = {
        "label": args.label, "rounds": args.rounds, "rounds_ok": len(ok),
        "lines_total": tot_lines,
        "lines_with_error": sum(x["lines_with_error"] for x in ok),
        "hex_char_errors": sum(x["hex_char_errors"] for x in ok),
        "missing": sum(x["missing"] for x in ok),
        "line_error_rate": round(sum(x["lines_with_error"] for x in ok) / tot_lines, 4) if tot_lines else None,
        "char_error_rate": round(sum(x["hex_char_errors"] for x in ok) / (tot_lines * 6), 6) if tot_lines else None,
        "prompt_tokens_mean": round(sum(ptoks) / len(ptoks)) if ptoks else None,
        **info,
        "rounds_detail": rounds,
    }
    out = json.dumps(summary, indent=1)
    if args.json == "-":
        print(out)
    else:
        with open(args.json, "w") as fh:
            fh.write(out + "\n")
        print("%s @ %s prompt tok (%s): line_error_rate=%s char_error_rate=%s (%d rounds, %d lines)" % (
            args.label, summary["prompt_tokens_mean"], args.position, summary["line_error_rate"],
            summary["char_error_rate"], len(ok), tot_lines))


if __name__ == "__main__":
    sys.exit(main())
