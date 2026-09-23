#!/usr/bin/env python3
"""#90: the long-context oracle form's INPUT side - the clean 178,553-id history of
issue #68 rebuilt deterministically, plus the sampled teacher-forced row plan.

    # rebuild + verify the ids against the recorded gate run (reads nothing but the
    # repo; deterministic, idempotent)
    tools/oracle_longctx_rows.py ids --out decode_out/oracle-longctx

    # the row plan: contiguous 64-row blocks ending at each depth anchor
    tools/oracle_longctx_rows.py plan --out decode_out/oracle-longctx

PROVENANCE, spelled out. `docs/long-context-goalmode.md` 4 records a CLEAN agentic
history of 178,553 prompt tokens - 167 synthetic `libghost` files read through 167
tool calls, 669 messages - that answers 4 of 5 probes with no degeneration. That
history is not stored anywhere as ids; it is a pure function of

  * `tools/longctx-gate.py::build_material(167)` / `build_session` (byte-stable by
    design, "generated from the file index alone"), and
  * the chat template of `models/Qwen3.8-Flash-Next-original` with
    `enable_thinking=False`, `add_generation_prompt=True` - the same token stream
    `tools/tokenize_ids.py --chat` produces and the parity harness treats as
    authoritative,

so this tool rebuilds it and VERIFIES the rebuild against the only independent
record: the per-probe `prompt_tokens` of `decode_out/68/longctx-170k.json`
(178553 / 178608 / 178664 / 178728 / 178798). Probe n's prompt contains the
recorded ANSWERS of probes 1..n-1, so all five counts check the template, the
tokenizer and the material generators at once. A count that does not match is a
hard error - the ids are never written on trust.

THE ROW PLAN. Row r is the next-token distribution conditioned on ids[0..r]
inclusive; teacher-forced rows over a sequence of length T are r = 0..T-2. The
plan samples CONTIGUOUS blocks of `--block` rows ending at each depth anchor
(default 1000 / 50000 / 100000 / 158000 / 178553; the last is T itself and clamps
to the final row T-2). Contiguous - not scattered - because both arms read a block
as one growing prefix: `decode parity`/`llama-row-probs` walk rows in order, and
the paired estimator of `tools/oracle-kld.py` wants neighbour rows on the same
prefix. The JSON this writes is also a `--row-groups` file for `oracle-kld.py`:
one group per depth.

The per-anchor PREFIX ids files exist for the crow arm: `decode parity` reads an
ids file whole, so anchor d's run gets ids[0..d] as its own file.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GATE_RUN = os.path.join(REPO, "decode_out", "68", "longctx-170k.json")
GATE_TOOL = os.path.join(REPO, "tools", "longctx-gate.py")
MODEL = os.path.join(REPO, "models", "Qwen3.8-Flash-Next-original")

N_FILES = 167            # the recorded 170k gate run: 178,553 prompt tokens
# the depths of record: 2564 is the sparse-QSA boundary (first rows past the
# 2048-token indexer budget); the rest are the issue's own anchors
DEPTHS = [1000, 2564, 50000, 100000, 158000, 178553]


def load_gate():
    spec = importlib.util.spec_from_file_location("longctx_gate", GATE_TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def build_probe_sequences(tok):
    """(descriptions, [ids per probe prompt]) using the gate's own generators and
    the recorded answers of decode_out/68/longctx-170k.json."""
    gate = load_gate()
    run = json.load(open(GATE_RUN, encoding="utf-8"))
    files, where = gate.build_material(run["n_files"])
    if run["n_files"] != N_FILES:
        raise SystemExit("the recorded run has n_files=%d, this tool pins %d - read the "
                         "record before changing anything" % (run["n_files"], N_FILES))
    msgs = gate.build_session(files)
    for m in msgs:
        for tc in m.get("tool_calls") or []:
            # the chat template iterates arguments as a mapping; the gate stores the
            # OpenAI wire form (a JSON string) - parse it the way every renderer does
            tc["function"]["arguments"] = json.loads(tc["function"]["arguments"])
    out = []
    base = list(msgs)
    for row in run["probes"]:
        question = {"role": "user", "content": row["question"]}
        prompt = base + [question]
        ids = tok.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True,
                                      enable_thinking=False)
        if hasattr(ids, "keys") and "input_ids" in ids:
            ids = list(ids["input_ids"])
        out.append({"probe": row["probe"], "recorded_prompt_tokens": row["prompt_tokens"],
                    "ids": [int(i) for i in ids], "verdict": row["verdict"]})
        # the next probe's prompt grows by this probe's question and RECORDED answer
        base = base + [question, {"role": "assistant", "content": row["answer"]}]
    return out, run


def cmd_ids(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    seqs, run = build_probe_sequences(tok)

    bad = [(s["probe"], len(s["ids"]), s["recorded_prompt_tokens"]) for s in seqs
           if len(s["ids"]) != s["recorded_prompt_tokens"]]
    if bad:
        for probe, got, want in bad:
            print("probe %d: rebuilt %d ids, the gate recorded %d - MISMATCH"
                  % (probe, got, want), file=sys.stderr)
        raise SystemExit("the rebuild does not reproduce the recorded gate run; nothing written")

    os.makedirs(args.out, exist_ok=True)
    manifest = {"what": "the clean #68 agentic history as token ids, rebuilt and verified",
                "gate_run": os.path.relpath(GATE_RUN, REPO),
                "n_files": run["n_files"], "material_chars": run["material_chars"],
                "history_messages": run["history_messages"],
                "generator": "tools/longctx-gate.py (build_material/build_session/probes)",
                "template": "models/Qwen3.8-Flash-Next-original chat_template.jinja, "
                            "enable_thinking=False, add_generation_prompt=True",
                "tokenizer": "transformers AutoTokenizer of the original model",
                "verified_against": "per-probe prompt_tokens of the gate run (all five)",
                "files": []}
    for s in seqs:
        name = "longctx-170k-p%d-ids.json" % s["probe"]
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(s["ids"], fh)
        manifest["files"].append({"file": name, "probe": s["probe"], "ids": len(s["ids"]),
                                  "recorded_prompt_tokens": s["recorded_prompt_tokens"],
                                  "gate_verdict": s["verdict"],
                                  "sha256": sha256_file(path)})
        print("probe %d: %7d ids  == recorded prompt_tokens  %s"
              % (s["probe"], len(s["ids"]), name))
    with open(os.path.join(args.out, "ids-manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    print("verified: every rebuilt prompt is the recorded length; manifest written")
    return 0


def cmd_plan(args):
    ids_path = os.path.join(args.out, "longctx-170k-p1-ids.json")
    if not os.path.exists(ids_path):
        raise SystemExit("%s is missing - run the `ids` subcommand first" % ids_path)
    with open(ids_path, encoding="utf-8") as fh:
        ids = json.load(fh)
    t = len(ids)
    last_row = t - 2                     # row T-2 predicts ids[T-1]; there is no row T-1

    groups = []
    for anchor in args.depths:
        end = min(anchor, last_row)
        start = max(0, end - args.block + 1)
        rows = list(range(start, end + 1))
        groups.append({"name": "d%d" % anchor, "anchor": anchor,
                       "prefix_tokens": end + 1, "rows": rows})
        prefix_name = "longctx-170k-a%d-ids.json" % anchor
        with open(os.path.join(args.out, prefix_name), "w", encoding="utf-8") as fh:
            json.dump(ids[:end + 1], fh)
        print("anchor %6d: rows %d..%d (%d rows, prefix %d ids) -> %s"
              % (anchor, rows[0], rows[-1], len(rows), end + 1, prefix_name))

    plan = {"what": "teacher-forced row plan over the clean 178,553-id history (#90)",
            "ids_file": os.path.basename(ids_path), "ids": t, "block": args.block,
            "depths": args.depths,
            "row_definition": "row r is the distribution conditioned on ids[0..r] "
                              "inclusive, predicting ids[r+1]; valid rows 0..T-2",
            "groups": groups}
    with open(os.path.join(args.out, "row-plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    print("plan written; it is also a --row-groups file for tools/oracle-kld.py")
    return 0


def cmd_subset(args):
    """dense dump -> sparse dump: the form oracle-kld.py's sparse reader and
    ref_longctx_logits.py's writer share (`<path>` + `<path>.rows.json`)."""
    import array

    rows = []
    if args.from_plan:
        with open(args.from_plan, encoding="utf-8") as fh:
            doc = json.load(fh)
        for g in doc["groups"]:
            rows += [int(r) for r in g["rows"]]
    else:
        rows = [int(r) for r in json.load(open(args.rows_file, encoding="utf-8"))]
    rows = sorted(set(rows))
    if not rows:
        raise SystemExit("no rows to keep")

    vocab = args.vocab
    stride = vocab * 4
    total = os.path.getsize(args.source) // stride
    if rows[-1] >= total:
        raise SystemExit("row %d is beyond the %d rows of %s" % (rows[-1], total, args.source))
    done = 0
    if os.path.exists(args.out) and os.path.exists(args.out + ".rows.json"):
        with open(args.out + ".rows.json", encoding="utf-8") as fh:
            done = len(json.load(fh)["rows"])
        print("resuming after %d rows already written" % done)
    with open(args.source, "rb") as src, open(args.out, "r+b" if done else "wb") as dst:
        dst.seek(done * stride)
        for r in rows[done:]:
            src.seek(r * stride)
            a = array.array("f")
            a.fromfile(src, vocab)
            if sys.byteorder != "little":
                a.byteswap()
            a.tofile(dst)
    with open(args.out + ".rows.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": rows, "source": os.path.relpath(args.source, REPO),
                   "vocab": vocab, "note": "sparse subset written by "
                   "tools/oracle_longctx_rows.py subset"}, fh, indent=1)
    print("%d rows -> %s (+ .rows.json)" % (len(rows), args.out))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_ids = sub.add_parser("ids", help="rebuild + verify the 178,553-id history and the "
                                       "five probe prompts")
    p_ids.add_argument("--out", default=os.path.join(REPO, "decode_out", "oracle-longctx"))
    p_ids.add_argument("--model", default=MODEL)
    p_ids.set_defaults(fn=cmd_ids)

    p_plan = sub.add_parser("plan", help="write the sampled row plan + per-anchor prefix "
                                         "ids files")
    p_plan.add_argument("--out", default=os.path.join(REPO, "decode_out", "oracle-longctx"))
    p_plan.add_argument("--block", type=int, default=64,
                        help="rows per depth block (default 64)")
    p_plan.add_argument("--depths", type=int, nargs="+", default=DEPTHS,
                        help="depth anchors in tokens (default 1000 2564 50000 100000 "
                             "158000 178553)")
    p_plan.set_defaults(fn=cmd_plan)

    p_sub = sub.add_parser("subset", help="carve selected rows out of a dense dump into the "
                                          "sparse form (f32 + .rows.json)")
    p_sub.add_argument("--source", required=True, help="a dense rows x vocab f32 dump")
    p_sub.add_argument("--out", required=True, help="the sparse f32 output path")
    p_sub.add_argument("--rows-file", default=None, help="a JSON array of absolute row ids")
    p_sub.add_argument("--from-plan", default=None,
                       help="a row-plan / row-groups JSON with a \"groups\" list")
    p_sub.add_argument("--vocab", type=int, default=248320)
    p_sub.set_defaults(fn=cmd_subset)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
