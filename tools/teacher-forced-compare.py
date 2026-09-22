#!/usr/bin/env python3
"""#91 / #91: compare TEACHER-FORCED logprob dumps position by position.

  tools/teacher-forced-compare.py --ref bare=DUMP.json --arm dense=DUMP2.json [--arm ...] \
      [--at 65 --want 8 --got 1] [--top 5] [--json OUT]

A dump is what `tools/corruption-replay-probe.py --dump-lp FILE` writes: every round's full
logprob entry list with serve's `crow_id` on each entry and alternative. The REFERENCE dump
fixes the sequence (normally the free greedy run of the bare container, whose ids are the
engine's own argmax path); every ARM dump must be that same id sequence forced
(`--force-ids REF`), and the tool refuses a pair whose ids differ at any compared position.

What it prints, from the top-N alternatives serve returned (top_logprobs <= 20: this is a
sparse view, never a full-distribution KLD):

  * at position --at (default 65, the K=2 digit): logprob of the --want and --got tokens
    (matched by token text) and the top-N, per dump - a token outside the returned top-N is
    reported as `< min(top)` (its bound), never as a number;
  * the CONTINUOUS measure over every compared position: logprob of the forced id in each
    dump, its difference to the reference, and the top-1 id per dump (a flip is marked).

Exit 0 when every arm's forced ids equal the reference's; 2 on a mismatch or bad input.
"""

import argparse
import json
import sys


def load(spec):
    label, _, path = spec.partition("=")
    if not path:
        sys.exit("expected LABEL=FILE, got %r" % spec)
    with open(path) as fh:
        doc = json.load(fh)
    rnd = doc["rounds"][0]
    return {"label": label, "path": path, "ids": rnd["ids"], "entries": rnd["entries"],
            "forced": rnd.get("forced", 0), "prompt_tokens": rnd.get("prompt_tokens"),
            "cached_tokens": rnd.get("cached_tokens")}


def tok_lp(entry, text):
    """(logprob, exact) of the token spelled `text` at this position: the chosen id or an
    alternative; not in the list -> (min of the list, False), the upper bound"""
    if entry.get("token") == text:
        return entry["logprob"], True
    alts = entry.get("top_logprobs") or []
    for a in alts:
        if a.get("token") == text:
            return a["logprob"], True
    return (min(a["logprob"] for a in alts) if alts else None), False


def top_n(entry, n):
    return [(a.get("token"), a.get("crow_id"), a["logprob"]) for a in (entry.get("top_logprobs") or [])[:n]]


def fmt_lp(v):
    lp, exact = v
    if lp is None:
        return "n/a"
    return "%.4f" % lp if exact else "<%.3f" % lp


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="LABEL=DUMP, the sequence of record")
    ap.add_argument("--arm", action="append", default=[], help="LABEL=DUMP, forced on the ref ids")
    ap.add_argument("--at", type=int, default=65)
    ap.add_argument("--want", default="8", help="the correct token text at --at")
    ap.add_argument("--got", default="1", help="the produced token text at --at")
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    ref = load(args.ref)
    arms = [load(a) for a in args.arm]
    dumps = [ref] + arms
    if any(i is None for i in ref["ids"]):
        print("ref %s has entries without crow_id - dump it with --dump-lp" % ref["path"], file=sys.stderr)
        return 2
    bad = 0
    for a in arms:
        n = min(len(a["ids"]), len(ref["ids"]))
        diff = [i for i in range(min(n, a["forced"] or n)) if a["ids"][i] != ref["ids"][i]]
        if diff:
            print("arm %s: forced ids differ from the ref at %s - not the same sequence"
                  % (a["label"], diff[:5]), file=sys.stderr)
            bad += 1
    if bad:
        return 2

    out = {"at": args.at, "want": args.want, "got": args.got, "dumps": [], "positions": []}
    print("prompt tokens: %s" % ", ".join("%s %s (%s cached)" % (d["label"], d["prompt_tokens"],
                                                               d["cached_tokens"]) for d in dumps))
    print("\n== position #%d: logprob(%r) vs logprob(%r), top-%d" % (args.at, args.want, args.got, args.top))
    for d in dumps:
        if args.at >= len(d["entries"]):
            print("  %-10s no entry #%d (%d entries)" % (d["label"], args.at, len(d["entries"])))
            continue
        e = d["entries"][args.at]
        w, g = tok_lp(e, args.want), tok_lp(e, args.got)
        margin = (g[0] - w[0]) if (w[1] and g[1]) else None
        print("  %-10s lp(%s) %-9s lp(%s) %-9s margin(got-want) %s  top: %s" % (
            d["label"], args.want, fmt_lp(w), args.got, fmt_lp(g),
            "%+.4f" % margin if margin is not None else "n/a",
            ", ".join("%r %.3f" % (t, lp) for t, _, lp in top_n(e, args.top))))
        out["dumps"].append({"label": d["label"], "path": d["path"], "lp_want": w[0], "want_exact": w[1],
                             "lp_got": g[0], "got_exact": g[1], "margin_got_minus_want": margin,
                             "top": top_n(e, args.top)})

    n = min(len(d["entries"]) for d in dumps)
    print("\n== continuous: logprob of the forced (ref) id per position; delta = arm - %s" % ref["label"])
    hdr = "  #    token              %10s" % ref["label"] + "".join(
        "  %10s %8s" % (a["label"], "delta") for a in arms)
    print(hdr)
    sums = {a["label"]: [] for a in arms}
    for i in range(n):
        rid = ref["ids"][i]
        r = ref["entries"][i]
        row = {"i": i, "id": rid, "token": r.get("token"), ref["label"]: r["logprob"],
               "top1": {d["label"]: (d["entries"][i].get("top_logprobs") or [{}])[0].get("crow_id") for d in dumps}}
        line = "  %-4d %-18r %10.4f" % (i, r.get("token")[:16], r["logprob"])
        for a in arms:
            ae = a["entries"][i]
            if a["ids"][i] != rid:          # past the forced part the arm ran free
                line += "  %10s %8s" % ("(free)", "")
                continue
            dlt = ae["logprob"] - r["logprob"]
            sums[a["label"]].append(dlt)
            row[a["label"]] = ae["logprob"]
            flip = row["top1"][a["label"]] != row["top1"][ref["label"]]
            line += "  %10.4f %+8.4f%s" % (ae["logprob"], dlt, "*" if flip else " ")
        out["positions"].append(row)
        print(line)
    print("  (* = the arm's top-1 differs from the ref's at that position)")
    for a in arms:
        s = sums[a["label"]]
        if s:
            print("  %s: %d positions, mean delta %+.4f nats, mean |delta| %.4f, max |delta| %.4f"
                  % (a["label"], len(s), sum(s) / len(s), sum(abs(x) for x in s) / len(s),
                     max(abs(x) for x in s)))
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
