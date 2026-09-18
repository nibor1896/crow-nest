#!/usr/bin/env python3
"""#78: the per-position next-token distribution of a GGUF, out of llama-server, for the
EXACT token id sequence a crow-nest reference was teacher-forced on.

    # the tokenizer check first - it is what decides whether the llama-perplexity route exists
    tools/llama-row-probs.py --base-url http://127.0.0.1:8083 \
        --ids decode_out/oracle-tf298/tf298-ids.json --round-trip

    # then the collection: one request per row, the prefix grown by one token each time
    tools/llama-row-probs.py --base-url http://127.0.0.1:8083 \
        --ids decode_out/oracle-tf298/tf298-ids.json --rows 0:298 \
        --out decode_out/oracle-tf298/llama-q2kxl/gpu-logits.f32

stdlib only. It talks to a running llama-server and to nothing else.

WHY THIS SHAPE. llama.cpp has no teacher forcing: nothing in it will take a token sequence and
hand back the distribution at every position. What `/completion` WILL do is take a prompt as an
ARRAY OF TOKEN IDS - `tokenize_input_subprompt`, `tools/server/server-common.cpp:972`, which
uses the ids verbatim, adds no BOS and applies no chat template - and return, for each
GENERATED token, the top `n_probs` of the distribution it was drawn from. So the distribution
at row r is one request whose prompt is ids[0..r] and whose `n_predict` is 1. With
`cache_prompt` the slot keeps the previous prompt's KV, so a row costs one decode step and not
one prefill.

WHY THE PROBABILITIES ARE THE MODEL'S AND NOT THE SAMPLER'S. With `post_sampling_probs` false -
the default, and set explicitly here - the server takes the PRE-SAMPLING logits and runs
`get_token_probabilities` (`tools/server/server-common.cpp:1501`): a plain softmax over the
WHOLE vocabulary, then a partial sort. Temperature, top-k, top-p and min-p never touch them.

WHY THE WHOLE VOCABULARY AND NOT A TOP N. `n_probs` carries no cap in the request schema
(`tools/server/server-schema.cpp:178`), and asking for all 248,320 entries costs 0.47 s and
25 MB per row on this machine against 0.07 s for a top 2,000 - so there is no reason to accept
a truncated estimator and its lower bound. This tool writes the FULL row, as natural
logarithms of the probabilities, into the same flat `rows x vocab` f32 little-endian file
`decode parity` writes, and `tools/oracle-kld.py` then reads the llama arm exactly as it reads
a crow-nest arm. A log-probability row and a logit row are the same thing to that tool: it
subtracts the row's own log-sum-exp, which also repairs the 1.0001 that llama.cpp's f32 softmax
accumulates over a quarter of a million terms.

WHAT THIS TOOL DOES NOT DO: it does not verify that the server serves the model you think it
does. The sidecar JSON beside the dump records `/props` verbatim so that the file says which
build and which GGUF produced it.
"""

import argparse
import array
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request


def post(base_url, path, payload, timeout=600):
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(base_url.rstrip("/") + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def get(base_url, path, timeout=60):
    with urllib.request.urlopen(base_url.rstrip("/") + path, timeout=timeout) as fh:
        return json.loads(fh.read().decode("utf-8"))


def round_trip(base_url, ids):
    """Detokenize the ids and tokenize the text back. Returns a dict of findings.

    This is the check that decides whether `llama-perplexity` could be used instead: that
    tool reads a TEXT file and tokenizes it with `common_tokenize(ctx, prompt, true)` -
    add_special TRUE and parse_special FALSE by the default of `common/common.h:1050`. If a
    sequence of ids does not survive detokenize -> tokenize under those settings, then no
    text file exists that would make llama-perplexity read these rows, and the route is
    invalid rather than inconvenient.
    """
    text = post(base_url, "/detokenize", {"tokens": ids})["content"]
    out = {"n_ids": len(ids), "text_chars": len(text), "text_head": text[:200]}
    for label, body in (
            ("parse_special=true,  add_special=false", {"content": text, "parse_special": True,
                                                        "add_special": False}),
            ("parse_special=false, add_special=false", {"content": text, "parse_special": False,
                                                        "add_special": False}),
            ("parse_special=false, add_special=true",  {"content": text, "parse_special": False,
                                                        "add_special": True}),
            ("parse_special=true,  add_special=true",  {"content": text, "parse_special": True,
                                                        "add_special": True}),
    ):
        back = post(base_url, "/tokenize", body)["tokens"]
        first = None
        for i in range(min(len(back), len(ids))):
            if back[i] != ids[i]:
                first = i
                break
        if first is None and len(back) != len(ids):
            first = min(len(back), len(ids))
        out[label] = {"n": len(back), "equal": back == ids, "first_difference": first,
                      "got_around_difference": back[max(0, (first or 0) - 2):(first or 0) + 3],
                      "want_around_difference": ids[max(0, (first or 0) - 2):(first or 0) + 3]}
    return out


def row_request(ids, row, n_probs, cache_prompt=True):
    """The /completion body for row `row`: the prefix ids[0..row] and one step off its end."""
    return {
        "prompt": ids[:row + 1],
        "n_predict": 1,
        "n_probs": n_probs,
        "post_sampling_probs": False,
        "temperature": 0.0,
        "top_k": 0,
        "top_p": 1.0,
        "min_p": 0.0,
        "cache_prompt": cache_prompt,
        "stream": False,
    }


SENTINEL = -1.0e30


def row_vector(resp, vocab):
    """Scatter one /completion response into a vocab-wide array of natural log-probabilities.

    Every id has to arrive: a slot left at the sentinel would be a log-probability of zero,
    which is a probability of ONE, and that is the one mistake this file must not carry.
    """
    probs = resp.get("completion_probabilities")
    if not probs:
        raise SystemExit("the server returned no completion_probabilities - is n_probs > 0?")
    top = probs[0]["top_logprobs"]
    if len(top) != vocab:
        raise SystemExit("the server returned %d entries, not the %d of the vocabulary - "
                         "n_probs was capped or the vocabulary is not what was assumed"
                         % (len(top), vocab))
    out = array.array("f", [SENTINEL]) * vocab
    mass = 0.0
    for e in top:
        lp = float(e["logprob"]) if "logprob" in e else math.log(max(float(e["prob"]), 1e-300))
        out[int(e["id"])] = lp
        mass += math.exp(lp)
    for i, v in enumerate(out):
        if v <= SENTINEL / 2:
            raise SystemExit("token id %d never arrived in the response" % i)
    return out, mass


def parse_range(text, total):
    if text is None:
        return (0, total)
    a, b = text.split(":", 1)
    first = int(a) if a.strip() else 0
    last = int(b) if b.strip() else total
    if not (0 <= first < last <= total):
        raise SystemExit("--rows %r is not inside 0:%d" % (text, total))
    return (first, last)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8083")
    ap.add_argument("--ids", required=True, help="a JSON array of token ids")
    ap.add_argument("--rows", default=None, help="A:B, half open")
    ap.add_argument("--vocab", type=int, default=248320,
                    help="the served vocabulary; every row is asked for in full")
    ap.add_argument("--out", default=None,
                    help="the rows x vocab f32 dump of natural log-probabilities")
    ap.add_argument("--no-cache-prompt", dest="cache_prompt", action="store_false",
                    help="prefill the whole prefix for every row instead of reusing the slot's "
                         "KV - the form that makes a row a pure function of its prefix, and the "
                         "one this measurement uses; the cached form is a control, not the arm")
    ap.add_argument("--round-trip", action="store_true",
                    help="only the tokenizer round-trip check, then stop")
    args = ap.parse_args(argv)

    with open(args.ids, encoding="utf-8") as fh:
        ids = json.load(fh)
    if not isinstance(ids, list) or not all(isinstance(i, int) for i in ids):
        raise SystemExit("%s is not a flat JSON array of token ids" % args.ids)

    props = get(args.base_url, "/props")
    print("server   %s" % args.base_url)
    print("build    %s" % props.get("build_info", props.get("build", "?")))
    print("model    %s" % props.get("model_path", props.get("default_generation_settings", {})
                                    .get("model", "?")))
    print("n_ctx    %s" % props.get("default_generation_settings", {}).get("n_ctx", "?"))
    print("")

    if args.round_trip:
        rt = round_trip(args.base_url, ids)
        print("ids                 %d" % rt["n_ids"])
        print("detokenized         %d characters" % rt["text_chars"])
        print("text head           %r" % rt["text_head"])
        for k, v in rt.items():
            if not isinstance(v, dict):
                continue
            print("%-40s %s  (%d tokens back, first difference at %s)"
                  % (k, "ROUND TRIPS" if v["equal"] else "DOES NOT ROUND TRIP",
                     v["n"], v["first_difference"]))
            if not v["equal"]:
                print("%-40s   want %s" % ("", v["want_around_difference"]))
                print("%-40s   got  %s" % ("", v["got_around_difference"]))
        if args.out:
            with open(args.out, "w", encoding="utf-8") as fh:
                json.dump({"props": props, "ids": args.ids, "round_trip": rt}, fh, indent=1)
        return 0

    first, last = parse_range(args.rows, len(ids))
    if first != 0:
        raise SystemExit("--rows has to start at 0: tools/oracle-kld.py reads a dump by "
                         "ABSOLUTE row index, so row 0 of the file must be row 0 of the run")
    if not args.out:
        raise SystemExit("--out is required for a collection run")
    vocab = args.vocab
    stride = vocab * 4
    side = args.out + ".meta.json"

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    have = os.path.getsize(args.out) // stride if os.path.exists(args.out) else 0
    if have * stride != (os.path.getsize(args.out) if os.path.exists(args.out) else 0):
        raise SystemExit("%s is not a whole number of rows - delete it and start again" % args.out)
    meta = []
    if have and os.path.exists(side):
        with open(side, encoding="utf-8") as fh:
            meta = json.load(fh)["rows"][:have]
    if have >= last:
        print("%s already carries %d rows" % (args.out, have))
        return 0
    if have:
        print("resuming at row %d of %d" % (have, last))

    t0 = time.time()
    warm = -1
    with open(args.out, "r+b" if have else "wb") as sink:
        sink.seek(have * stride)
        for row in range(have, last):
            if args.cache_prompt and row != warm + 1 and row > 0:
                # a resumed run starts in the middle: prime the slot with the prefix once,
                # cheaply, so the expensive request below is a single decode step
                post(args.base_url, "/completion", row_request(ids, row - 1, 1))
            resp = post(args.base_url, "/completion",
                        row_request(ids, row, vocab, args.cache_prompt))
            vec, mass = row_vector(resp, vocab)
            vec.tofile(sink)
            sink.flush()
            warm = row
            timings = resp.get("timings", {})
            meta.append({"row": row, "mass": mass,
                         "prompt_n": resp.get("tokens_evaluated", timings.get("prompt_n")),
                         "cached": resp.get("tokens_cached"),
                         "top1": int(resp["completion_probabilities"][0]["id"])})
            with open(side, "w", encoding="utf-8") as fh:
                json.dump({"props": props, "ids_file": args.ids, "vocab": vocab,
                           "first_row": first, "cache_prompt": args.cache_prompt,
                           "rows": meta}, fh)
            if row % 25 == 0 or row == last - 1:
                print("row %4d/%4d  mass %.7f  prompt_n %s  cached %s  %.1f s"
                      % (row, last - 1, mass, meta[-1]["prompt_n"], meta[-1]["cached"],
                         time.time() - t0), flush=True)

    masses = sorted(m["mass"] for m in meta)
    print("")
    print("%d rows x %d -> %s (%d bytes)" % (last, vocab, args.out, os.path.getsize(args.out)))
    print("llama.cpp's own f32 softmax mass over the whole vocabulary: min %.7f median %.7f "
          "max %.7f  (tools/oracle-kld.py renormalises every row)"
          % (masses[0], masses[len(masses) // 2], masses[-1]))
    print("wall %.1f s" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
