#!/usr/bin/env python3
"""#90: an ENGLISH long-prose corpus for the oracle-KLD instrument (scope change
2026-09-20: English only - the German-prose component of the issue was dropped).

The German non-word symptom had no logits-space counterpart because no corpus fed
the instrument; this closes that gap in English: public-domain long prose, fetched
once from Project Gutenberg, stripped of the boilerplate, tokenized with the
ORIGINAL model tokenizer, and emitted as teacher-forced id rows in the same dump
format the instrument already reads.

    # fetch + strip the corpus texts (idempotent; skips files already present)
    tools/oracle_longctx_corpus.py fetch --corpus tools/corpora

    # tokenize + emit the oracle id sets (idempotent)
    tools/oracle_longctx_corpus.py emit --corpus tools/corpora --out decode_out/oracle-en

FORM. This is a CORPUS, not a chat prompt: raw text, `add_special_tokens=False`,
no chat template - the form `llama-perplexity` would read a text file in, and the
form that makes every row a plain continuation row. From each text one contiguous
slice of `--rows + 1` ids is taken, centered on the book's midpoint (steady-state
narrative prose, not the opening the model has seen a thousand times), so the set
carries exactly `--rows` teacher-forced rows. The row groups file written beside
the ids is a `--row-groups` input for `tools/oracle-kld.py`: one group per text,
which is the per-corpus summary of #90.

Everything is deterministic given the tokenizer: same texts, same slice, same ids.
`SOURCES.json` records url, bytes and sha256 of every raw download plus the exact
strip rule, so a rerun after a Project Gutenberg re-upload can be DIFFED rather
than trusted.
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.request

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.path.join(REPO, "models", "Qwen3.8-Flash-Next-original")

# public-domain English long prose, chosen for length and variety of register:
# epic novel, social novel, gothic, epistolary horror, vernacular first-person
TEXTS = [
    {"id": 2701, "name": "moby-dick", "author": "Herman Melville", "title": "Moby Dick"},
    {"id": 1342, "name": "pride-prejudice", "author": "Jane Austen",
     "title": "Pride and Prejudice"},
    {"id": 84, "name": "frankenstein", "author": "Mary Shelley", "title": "Frankenstein"},
    {"id": 345, "name": "dracula", "author": "Bram Stoker", "title": "Dracula"},
    {"id": 76, "name": "huckleberry-finn", "author": "Mark Twain",
     "title": "Adventures of Huckleberry Finn"},
]

START_MARKERS = ["*** START OF", "*END*THE SMALL PRINT", "***START OF"]
END_MARKERS = ["*** END OF", "***END OF", "End of the Project Gutenberg"]


def sha256_bytes(b):
    return hashlib.sha256(b).hexdigest()


def strip_boilerplate(raw):
    """Cut the Project Gutenberg header/footer. Returns (text, note)."""
    text = raw.decode("utf-8", "replace")
    lo = 0
    for marker in START_MARKERS:
        i = text.find(marker)
        if i >= 0:
            eol = text.find("\n", i)
            lo = eol + 1 if eol >= 0 else len(text)
            break
    hi = len(text)
    for marker in END_MARKERS:
        i = text.find(marker, lo)
        if i >= 0:
            hi = i
            break
    note = ("stripped header (%d B) and footer (%d B)" % (lo, len(text) - hi)
            if (lo > 0 or hi < len(text)) else "no markers found - kept whole")
    return text[lo:hi], note


def cmd_fetch(args):
    os.makedirs(args.corpus, exist_ok=True)
    sources_path = os.path.join(args.corpus, "SOURCES.json")
    sources = {"what": "public-domain English long prose for the #90 oracle corpus",
               "fetched_utc_note": "see per-file sha256; refetch and diff rather than trust",
               "license": "Project Gutenberg / public domain in the US",
               "strip_rule": "text between the first '*** START OF' line and the first "
                             "'*** END OF' marker, inclusive cut, utf-8 with replacement",
               "texts": []}
    for t in TEXTS:
        raw_path = os.path.join(args.corpus, "%s.raw.txt" % t["name"])
        txt_path = os.path.join(args.corpus, "%s.txt" % t["name"])
        if not os.path.exists(txt_path):
            url = "https://www.gutenberg.org/ebooks/%d.txt.utf-8" % t["id"]
            print("fetch %s -> %s" % (url, raw_path))
            req = urllib.request.Request(url, headers={"User-Agent": "curl/8.0"})
            with urllib.request.urlopen(req, timeout=120) as fh:
                raw = fh.read()
            if len(raw) < 100_000:
                raise SystemExit("%s came back with %d bytes - not a book; nothing written"
                                 % (url, len(raw)))
            with open(raw_path, "wb") as fh:
                fh.write(raw)
            text, note = strip_boilerplate(raw)
            if len(text) < 100_000:
                raise SystemExit("%s stripped to %d chars - strip rule failed" % (url, len(text)))
            with open(txt_path, "w", encoding="utf-8") as fh:
                fh.write(text)
        else:
            raw = open(raw_path, "rb").read()
            text = open(txt_path, encoding="utf-8").read()
            note = "already present - not refetched"
        sources["texts"].append({"gutenberg_id": t["id"], "name": t["name"],
                                 "author": t["author"], "title": t["title"],
                                 "raw_bytes": len(raw), "raw_sha256": sha256_bytes(raw),
                                 "stripped_chars": len(text), "strip_note": note,
                                 "url": "https://www.gutenberg.org/ebooks/%d" % t["id"]})
        print("  %-18s %8d chars  %s" % (t["name"], len(text), note))
    with open(sources_path, "w", encoding="utf-8") as fh:
        json.dump(sources, fh, indent=1)
    print("SOURCES.json written")
    return 0


def cmd_emit(args):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    os.makedirs(args.out, exist_ok=True)
    manifest = {"what": "English long-prose oracle id sets (#90, English-only scope)",
                "tokenizer": "models/Qwen3.8-Flash-Next-original, add_special_tokens=False, "
                             "no chat template - plain corpus continuation rows",
                "slice_rule": "one contiguous slice of rows+1 ids centered on the book's "
                              "midpoint; rows 0..rows-1 are teacher-forced on it",
                "rows": args.rows, "sets": []}
    groups = {"what": "row groups (one per corpus text) for oracle-kld --row-groups",
              "groups": []}
    for t in TEXTS:
        txt_path = os.path.join(args.corpus, "%s.txt" % t["name"])
        if not os.path.exists(txt_path):
            raise SystemExit("%s missing - run `fetch` first" % txt_path)
        text = open(txt_path, encoding="utf-8").read()
        ids = tok(text, add_special_tokens=False)["input_ids"]
        if len(ids) < args.rows + 1:
            raise SystemExit("%s has %d ids, fewer than rows+1=%d" % (t["name"], len(ids),
                                                                      args.rows + 1))
        mid = len(ids) // 2
        start = mid - (args.rows + 1) // 2
        ids = [int(i) for i in ids[start:start + args.rows + 1]]
        name = "en-prose-%s-ids.json" % t["name"]
        path = os.path.join(args.out, name)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(ids, fh)
        rows = list(range(args.rows))
        manifest["sets"].append({"file": name, "text": t["name"], "book_ids_start": start,
                                 "ids": len(ids), "rows": args.rows})
        groups["groups"].append({"name": t["name"], "ids_file": name, "rows": rows})
        print("%-18s %6d book ids -> slice %d..%d (%d rows) %s"
              % (t["name"], 2 * mid, start, start + len(ids) - 1, args.rows, name))
    with open(os.path.join(args.out, "en-manifest.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=1)
    with open(os.path.join(args.out, "en-row-groups.json"), "w", encoding="utf-8") as fh:
        json.dump(groups, fh, indent=1)
    print("manifest + row-groups written; oracle rows are produced from these ids by "
          "oracle/ref_longctx_logits.py once the f32 checkpoint exists again")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("fetch", help="download + strip the Gutenberg texts (idempotent)")
    p.add_argument("--corpus", default=os.path.join(REPO, "tools", "corpora"))
    p.set_defaults(fn=cmd_fetch)

    p = sub.add_parser("emit", help="tokenize the texts into oracle id sets (idempotent)")
    p.add_argument("--corpus", default=os.path.join(REPO, "tools", "corpora"))
    p.add_argument("--out", default=os.path.join(REPO, "decode_out", "oracle-en"))
    p.add_argument("--model", default=MODEL)
    p.add_argument("--rows", type=int, default=512, help="teacher-forced rows per text")
    p.set_defaults(fn=cmd_emit)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
