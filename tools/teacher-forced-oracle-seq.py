#!/usr/bin/env python3
"""#91 / #91: the ORACLE side of the teacher-forced K=2 measurement.

  prep  (system python, stdlib only):
    tools/teacher-forced-oracle-seq.py --serve-log OUT/serve-bare.log --dump OUT/bare.lp.json \
        --out OUT/oracle
      -> OUT/oracle/gen-sequence.json in the form oracle/ref_engine_logits.py reads
         (CROW_PARITY_DIR): all_ids = serve's OWN prompt ids (the `[chat] prompt ids` debug
         line of a `crow_force_ids` request, CROW_LOG=info,chat=debug) + the bare greedy
         completion ids of the dump; rows = len(all_ids). Refuses when the log's generated
         ids (`[chat] ids`) differ from the dump's, or the prompt is not the dump's
         prompt_tokens long.

  read  (.venv-oracle python: numpy + tokenizers):
    .venv-oracle/bin/python tools/teacher-forced-oracle-seq.py --read OUT/oracle \
        [--top 20] [--label oracle]
      -> OUT/oracle/oracle.lp.json, the reference rows turned into the dump form
         (per completion position j: row prompt_len-1+j, log-softmax in f64, the forced id's
         logprob and the top-N with crow_id/token/bytes), which
         tools/teacher-forced-compare.py takes as one more --arm.

The reference itself (CPU f32, the ORIGINAL checkpoint) is oracle/ref_engine_logits.py:
    CROW_PARITY_DIR=OUT/oracle .venv-oracle/bin/python oracle/ref_engine_logits.py
It needs the 131 original shards under models/Qwen3.8-Flash-Next-original/ (check with
`oracle/ref_longctx_logits.py check-weights`).
"""

import argparse
import json
import os
import re
import sys


def last_list(log_text, tag):
    """the id list of the FIRST line carrying `[chat] <tag> [...]`"""
    m = re.search(r"\[chat\] %s \[([0-9, ]*)\]" % re.escape(tag), log_text)
    if not m:
        return None
    body = m.group(1).strip()
    return [int(x) for x in body.split(",")] if body else []


def prep(args):
    with open(args.serve_log, encoding="utf-8", errors="replace") as fh:
        log = fh.read()
    prompt = last_list(log, "prompt ids")
    gen = last_list(log, "ids")
    if prompt is None:
        sys.exit("%s has no `[chat] prompt ids` line - boot serve with CROW_LOG=info,chat=debug "
                 "and send crow_force_ids (the probe's --dump-lp does)" % args.serve_log)
    with open(args.dump) as fh:
        rnd = json.load(fh)["rounds"][0]
    comp = rnd["ids"]
    if any(i is None for i in comp):
        sys.exit("%s has entries without crow_id" % args.dump)
    if rnd.get("prompt_tokens") is not None and rnd["prompt_tokens"] != len(prompt):
        sys.exit("prompt ids %d != the dump's prompt_tokens %d" % (len(prompt), rnd["prompt_tokens"]))
    if gen is not None and gen != comp:
        sys.exit("the log's generated ids differ from the dump's ids - not the same request")
    n = len(comp) if args.n is None else args.n
    all_ids = prompt + comp[:n]
    os.makedirs(args.out, exist_ok=True)
    seq = {"rows": len(all_ids), "prompt_len": len(prompt), "completion_forced": n,
           "all_ids": all_ids, "source_log": os.path.abspath(args.serve_log),
           "source_dump": os.path.abspath(args.dump),
           "note": "#91: serve's prompt ids + the bare greedy completion; "
                   "row prompt_len-1+j is the distribution of completion token #j"}
    with open(os.path.join(args.out, "gen-sequence.json"), "w") as fh:
        json.dump(seq, fh)
    print("gen-sequence.json: %d prompt + %d completion ids = %d rows -> %s"
          % (len(prompt), n, len(all_ids), args.out))
    return 0


def read(args):
    import numpy as np
    from tokenizers import Tokenizer
    d = args.read
    with open(os.path.join(d, "gen-sequence.json")) as fh:
        seq = json.load(fh)
    T, P = seq["rows"], seq["prompt_len"]
    ids = seq["all_ids"]
    path = os.path.join(d, "ref-logits.f32")
    V = os.path.getsize(path) // 4 // T
    ref = np.memmap(path, dtype=np.float32, mode="r", shape=(T, V))
    here = os.path.dirname(os.path.abspath(__file__))
    tok = Tokenizer.from_file(os.path.join(here, "..", "models", "Qwen3.8-Flash-Next-original",
                                           "tokenizer.json"))

    def one(i, lp):
        t = tok.decode([int(i)], skip_special_tokens=False)
        return {"token": t, "logprob": float(lp), "bytes": list(t.encode("utf-8")), "crow_id": int(i)}

    entries = []
    for j in range(T - P):
        row = ref[P - 1 + j].astype(np.float64)
        m = row.max()
        lz = m + np.log(np.exp(row - m).sum())
        lps = row - lz
        chosen = ids[P + j]
        top = np.argsort(-lps, kind="stable")[:args.top]
        e = one(chosen, lps[chosen])
        e["top_logprobs"] = [one(i, lps[i]) for i in top]
        entries.append(e)
    out = os.path.join(d, "%s.lp.json" % args.label)
    with open(out, "w") as fh:
        json.dump({"label": args.label, "rounds": [{"ids": [e["crow_id"] for e in entries],
                                                   "entries": entries, "forced": len(entries),
                                                   "prompt_tokens": P, "cached_tokens": None}]}, fh)
    print("%d oracle positions -> %s" % (len(entries), out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-log")
    ap.add_argument("--dump")
    ap.add_argument("--out")
    ap.add_argument("--n", type=int, default=None, help="completion ids to include (default all)")
    ap.add_argument("--read", metavar="DIR", help="turn DIR/ref-logits.f32 into a dump")
    ap.add_argument("--top", type=int, default=20)
    ap.add_argument("--label", default="oracle")
    args = ap.parse_args(argv)
    if args.read:
        return read(args)
    if not (args.serve_log and args.dump and args.out):
        ap.error("prep needs --serve-log, --dump and --out (or --read DIR)")
    return prep(args)


if __name__ == "__main__":
    sys.exit(main())
