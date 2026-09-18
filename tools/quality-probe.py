#!/usr/bin/env python3
"""#75: the quality probe - a re-runnable characterization of ANSWER QUALITY at the real
operating row, against any OpenAI-compatible `/v1/chat/completions` endpoint.

Usage, from anywhere:
  tools/quality-probe.py --label A1-crow  --base-url http://127.0.0.1:8099
  tools/quality-probe.py --label B1-llama --base-url http://127.0.0.1:8083
  tools/quality-probe.py --label A3-crow  --seeds 7,8,9
  tools/quality-probe.py --compare A1-crow B1-llama
  python tools/test_quality_probe.py          # the metric functions, no server, no GPU

Why this exists (step 1 of the requant work, issue #75): robin's poster goal misspells from
the first sentence on this engine - "Erschd kuck ich mir alles an", "Kunstgewerkds",
"Jahrundert" - and llama-server with the Unsloth UD-Q2_K_XL quant does not, on the same
sampling row. Before anything about the quant moves, the present behaviour has to be pinned
by a tool that can be run again afterwards (Feathers: characterization before change). This
is that tool. It changes nothing in the engine, the converter or the quant.

WHAT IT MEASURES, AND WHAT IT CANNOT
------------------------------------
Every metric here is REFERENCE-FREE: it reads one answer and says something about the answer
itself. None of them compares a distribution to a reference model, so none of them can say
"this quant is N bits worse". The literature's reference-based readings - mean KL divergence
and top-1 agreement against the BF16 original, teacher-forced (arXiv 2407.09141, llama.cpp
discussion 4110), and Unsloth's greedy-continuation agreement over 32 tokens - need the BF16
weights, which are not on this machine. That is a later step and is named as such in
`docs/quality-probe.md`.

THE ROW
-------
The REAL operating row of the client, not a measurement row: temperature 1.0, top_p 0.95,
top_k 20, presence_penalty 0.0, min_p 0.0, `max_tokens` 2600, and a fixed seed list. Both
servers read `seed`. Thinking is OFF for the baseline, and the two servers are switched off
differently, which is why `--engine` exists:

| engine | thinking off | thinking on |
|---|---|---|
| `crow` (`serve`) | the field is ABSENT - #74: an absent `reasoning_effort` is the render of record | top-level `reasoning_effort`, `low` / `medium` / `high` / `xhigh` |
| `llama` (llama-server) | top-level `"reasoning_effort": "none"` - the server intercepts it and sets `enable_thinking = false` (`server-common.cpp:1323`) | top-level `reasoning_effort`, `low` / `medium` / `high` |

With `--reasoning-effort` the run repeats with thinking on. Only `content` is ever scored;
`reasoning_content` is stored in the record and never measured.

THE METRICS
-----------
| metric | what it counts | the failure it is built for |
|---|---|---|
| `nonword` | words the hunspell dictionary does not know, per 1000 checked words, German and English separately | "Kunstgewerkds", "Jahrundert", "Erschd" |
| `literal` | share of literals from the prompt reproduced EXACTLY, plus every near-miss at edit distance 1 or 2 | "#DFFFE0" where the prompt said "#D7FFE0" |
| `json` | the answer parses, and has the shape the prompt demanded | a JSON answer that is prose, or misses a key |
| `repetition` | longest immediately repeated n-gram run, and distinct-word ratio | the degeneration of #68 |
| `foreign` | CJK, Cyrillic, Arabic, Hebrew, Devanagari characters in a German or English answer | a script slip |
| `length` | words, characters, completion tokens, finish reason | an answer that stopped early |

What is NOT counted as a word: anything inside a fenced code block or an inline code span,
any URL, and any whitespace-delimited token carrying a digit or one of `_ / \\ # @ < > { } = |`
- so hex codes, versions, paths and identifiers never reach the dictionary. Hyphenated
compounds are split and each part is checked, single letters and ALL-CAPS tokens are dropped.
Hunspell's known weakness on German is compounds and proper names, and the run record keeps
every flagged word so the false-positive share can be hand-checked rather than assumed.

THE DICTIONARIES
----------------
`hunspell` is installed on this machine and NO dictionary is. The probe fetches `de_DE_frami`
and `en_US` from the LibreOffice dictionaries repository into `~/.cache/crow-nest/dict/`
(override with `--dict-dir`) and records the URL and the sha256 of every file it used in the
run record. They are never committed.

THE OUTPUT
----------
`decode_out/quality-probe/<label>/`:

- `records.jsonl` - one JSON object per generation: prompt id, seed, the full text, every
  metric, the server's `/props` model name, the sampling that was sent, the timings.
- `run.json` - the run header: date, repo commit, endpoint, engine, props, dictionary
  provenance, the prompt-set version, and the aggregate of every metric.
- `summary.md` - the table a human reads.

`--compare A B` prints the two labels side by side with the per-prompt differences.

Exit code is 0 when the run completed, 1 when a request failed, 2 on a usage error.
"""

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PROMPTS = REPO / "tools" / "quality-probe-prompts.json"
OUT_ROOT = REPO / "decode_out" / "quality-probe"
DEFAULT_DICT_DIR = Path.home() / ".cache" / "crow-nest" / "dict"

# The LibreOffice dictionaries repository, the same files LibreOffice itself ships.
DICT_BASE = "https://raw.githubusercontent.com/LibreOffice/dictionaries/master"
DICTS = {
    "de": {"stem": "de_DE_frami", "files": {"de_DE_frami.aff": "de/de_DE_frami.aff",
                                            "de_DE_frami.dic": "de/de_DE_frami.dic"}},
    "en": {"stem": "en_US", "files": {"en_US.aff": "en/en_US.aff",
                                      "en_US.dic": "en/en_US.dic"}},
}

# the real operating row, the one the client sends
ROW = {"temperature": 1.0, "top_p": 0.95, "top_k": 20, "presence_penalty": 0.0, "min_p": 0.0}
DEFAULT_SEEDS = (1201, 1202, 1203)


# ---------------------------------------------------------------- pure metrics

FENCE_RE = re.compile(r"```.*?(?:```|\Z)", re.S)
INLINE_CODE_RE = re.compile(r"`[^`\n]*`")
URL_RE = re.compile(r"(?:https?://|www\.)\S+")
# a token is "code shaped" when it carries a digit or one of these characters
CODEY_RE = re.compile(r"[0-9_/\\#@<>{}=|]")
HYPHENS = "-‐‑‒–—―−"
LETTER_WORD_RE = re.compile(r"[^\W\d_]+(?:'[^\W\d_]+)*", re.UNICODE)


def strip_uncheckable(text):
    """Everything the dictionary must never see: code, links, identifiers, numbers.

    Fenced blocks and inline code spans go first (a code span is code whatever is in it),
    then URLs, then every whitespace-delimited token that carries a digit or one of
    ``_ / \\ # @ < > { } = |``. What is left is the prose a speller may judge.
    """
    text = FENCE_RE.sub(" ", text)
    text = INLINE_CODE_RE.sub(" ", text)
    text = URL_RE.sub(" ", text)
    kept = [t for t in text.split() if not CODEY_RE.search(t)]
    return " ".join(kept)


def checkable_words(text):
    """The words of ``text`` a spell checker is allowed to judge, in order.

    Hyphenated compounds are split into their parts - "Jugendstil-Ornamentik" is two words
    the dictionary knows and one it does not. Single letters go (they are "z. B." debris)
    and so do ALL-CAPS tokens of two or more letters, which are acronyms and not spelling.
    """
    cleaned = strip_uncheckable(text)
    for h in HYPHENS:
        cleaned = cleaned.replace(h, " ")
    cleaned = cleaned.replace("’", "'")
    out = []
    for w in LETTER_WORD_RE.findall(cleaned):
        if len(w) < 2:
            continue
        if w.isupper():
            continue
        out.append(w)
    return out


def edit_distance(a, b, cap=None):
    """Levenshtein distance, with an optional early exit above ``cap``."""
    if a == b:
        return 0
    if cap is not None and abs(len(a) - len(b)) > cap:
        return cap + 1
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        if cap is not None and min(cur) > cap:
            return cap + 1
        prev = cur
    return prev[-1]


CANDIDATE_RE = re.compile(r"[#0-9A-Za-z][0-9A-Za-z._:/\\-]*")


def literal_report(text, literals):
    """Exact-literal fidelity: how often each demanded literal came back byte for byte.

    ``literals`` is a list of ``{"text": ..., "min": n}``. A literal counts as reproduced
    when it appears at least ``min`` times EXACTLY - case included, because "#d7ffe0" is a
    different string on the wire than "#D7FFE0".

    A NEAR-MISS is a candidate token of the answer that is not the literal and whose
    case-folded edit distance to it is at most 2. Case is folded for the distance and not
    for the count: "#d7ffe0" is six edits from "#D7FFE0" case-sensitively, which would hide
    the most readable failure of all behind the cap, so it is reported at distance 0 with
    ``case_only`` set. Everything else is the "#DFFFE0 instead of #D7FFE0" shape.
    """
    wanted = [dict(l) for l in literals]
    cands = {}
    for c in CANDIDATE_RE.findall(text):
        c = c.rstrip(".,;:)")
        if c:
            cands[c] = cands.get(c, 0) + 1
    per = []
    near = []
    for spec in wanted:
        lit, need = spec["text"], int(spec.get("min", 1))
        exact = text.count(lit)
        per.append({"literal": lit, "required": need, "exact": exact, "ok": exact >= need})
        folded = lit.lower()
        for c, n in cands.items():
            if c == lit:
                continue
            d = edit_distance(c.lower(), folded, cap=2)
            if d <= 2:
                near.append({"literal": lit, "seen": c, "distance": d,
                             "case_only": d == 0, "count": n})
    need_total = sum(p["required"] for p in per)
    got_total = sum(min(p["exact"], p["required"]) for p in per)
    near.sort(key=lambda m: (m["distance"], -m["count"], m["seen"]))
    return {
        "literals_total": len(per),
        "literals_ok": sum(1 for p in per if p["ok"]),
        "literals_share": (sum(1 for p in per if p["ok"]) / len(per)) if per else None,
        "occurrence_share": (got_total / need_total) if need_total else None,
        "per_literal": per,
        "near_misses": near,
    }


FENCED_JSON_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def extract_json(text):
    """(value, how) for the first JSON document in ``text``, or (None, why-not).

    Three doors, in this order: the whole answer parses; a fenced block parses; the first
    balanced ``{...}`` or ``[...]`` parses. ``how`` records which door it came through, so
    a summary can separate "wrote JSON" from "wrote JSON inside prose it was told not to
    write".
    """
    s = text.strip()
    if not s:
        return None, "empty"
    try:
        return json.loads(s), "bare"
    except ValueError:
        pass
    m = FENCED_JSON_RE.search(text)
    if m:
        try:
            return json.loads(m.group(1).strip()), "fenced"
        except ValueError:
            pass
    for opener, closer in (("{", "}"), ("[", "]")):
        start = text.find(opener)
        while start != -1:
            depth, in_str, esc = 0, False, False
            for i in range(start, len(text)):
                ch = text[i]
                if in_str:
                    if esc:
                        esc = False
                    elif ch == "\\":
                        esc = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == opener:
                    depth += 1
                elif ch == closer:
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[start:i + 1]), "embedded"
                        except ValueError:
                            break
            start = text.find(opener, start + 1)
    return None, "no parse"


def _type_ok(value, name):
    if name == "string":
        return isinstance(value, str)
    if name == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if name == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if name == "boolean":
        return isinstance(value, bool)
    if name == "integer_or_null":
        return value is None or (isinstance(value, int) and not isinstance(value, bool))
    if name == "array":
        return isinstance(value, list)
    if name == "object":
        return isinstance(value, dict)
    return False


def check_shape(value, spec, path="$"):
    """Every way ``value`` fails ``spec``, as a list of sentences. Empty means it fits.

    The spec language is the small one the prompt file needs and nothing more: a type name
    (``string``, ``integer``, ``number``, ``boolean``, ``integer_or_null``), an object with
    ``required`` (name -> spec), or an array with ``length`` or ``min_items`` and ``items``.
    """
    problems = []
    if isinstance(spec, str):
        if not _type_ok(value, spec):
            problems.append(f"{path}: expected {spec}, got {type(value).__name__}")
        return problems
    kind = spec.get("type")
    if kind == "object":
        if not isinstance(value, dict):
            return [f"{path}: expected object, got {type(value).__name__}"]
        for key, sub in (spec.get("required") or {}).items():
            if key not in value:
                problems.append(f"{path}.{key}: missing")
            else:
                problems += check_shape(value[key], sub, f"{path}.{key}")
        if spec.get("no_extra_keys"):
            for key in value:
                if key not in (spec.get("required") or {}):
                    problems.append(f"{path}.{key}: unexpected key")
        return problems
    if kind == "array":
        if not isinstance(value, list):
            return [f"{path}: expected array, got {type(value).__name__}"]
        if "length" in spec and len(value) != spec["length"]:
            problems.append(f"{path}: expected {spec['length']} items, got {len(value)}")
        if "min_items" in spec and len(value) < spec["min_items"]:
            problems.append(f"{path}: expected at least {spec['min_items']} items, got {len(value)}")
        items = spec.get("items")
        if items is not None:
            for i, v in enumerate(value):
                problems += check_shape(v, items, f"{path}[{i}]")
        return problems
    if kind is None:
        return [f"{path}: broken shape spec {spec!r}"]
    return check_shape(value, kind, path)


def repetition(words, max_n=8):
    """Longest IMMEDIATELY repeated n-gram run, and the distinct-word ratio.

    "a b a b a b" is one 2-gram repeated 3 times. Immediate repetition is the shape the
    long-context degeneration of #68 takes, and it is the one a distinct-word ratio alone
    cannot separate from an ordinary text that reuses a term.
    """
    n_words = len(words)
    if n_words == 0:
        return {"words": 0, "distinct_word_ratio": None, "longest_repeat_n": 0,
                "longest_repeat_count": 0, "longest_repeat_text": ""}
    best = (1, 1, words[0])
    for n in range(1, min(max_n, n_words) + 1):
        i = 0
        while i + n <= n_words:
            gram = words[i:i + n]
            count = 1
            j = i + n
            while j + n <= n_words and words[j:j + n] == gram:
                count += 1
                j += n
            if count > 1 and (count, n) > (best[1], best[0]):
                best = (n, count, " ".join(gram))
            i = max(i + 1, j - n + 1) if count > 1 else i + 1
    return {
        "words": n_words,
        "distinct_word_ratio": len(set(w.lower() for w in words)) / n_words,
        "longest_repeat_n": best[0],
        "longest_repeat_count": best[1],
        "longest_repeat_text": best[2],
    }


SCRIPTS = (
    ("cjk", ((0x3400, 0x4DBF), (0x4E00, 0x9FFF), (0xF900, 0xFAFF),
             (0x3040, 0x30FF), (0x31F0, 0x31FF), (0xAC00, 0xD7AF))),
    ("cyrillic", ((0x0400, 0x04FF), (0x0500, 0x052F))),
    ("arabic", ((0x0600, 0x06FF), (0x0750, 0x077F))),
    ("hebrew", ((0x0590, 0x05FF),)),
    ("devanagari", ((0x0900, 0x097F),)),
    ("greek", ((0x0370, 0x03FF),)),
)
# greek letters are ordinary in a technical text (alpha, beta), so they are counted apart
FOREIGN_SCRIPTS = tuple(name for name, _ in SCRIPTS if name != "greek")


def foreign_script(text, context=24):
    """Characters from a script a German or English answer has no business carrying."""
    counts = {}
    samples = []
    for i, ch in enumerate(text):
        o = ord(ch)
        if o < 0x0370:
            continue
        for name, ranges in SCRIPTS:
            if any(lo <= o <= hi for lo, hi in ranges):
                counts[name] = counts.get(name, 0) + 1
                if len(samples) < 12:
                    a, b = max(0, i - context), min(len(text), i + context)
                    samples.append({"script": name, "char": ch,
                                    "context": text[a:b].replace("\n", " ")})
                break
    return {
        "counts": counts,
        "foreign_chars": sum(counts.get(n, 0) for n in FOREIGN_SCRIPTS),
        "greek_chars": counts.get("greek", 0),
        "samples": samples,
    }


def word_contexts(text, words, context=45, per_word=1):
    """Verbatim context for each flagged word, so a human can hand-check the flags."""
    out = []
    for w in words:
        pat = re.compile(r"(?<![^\W\d_])" + re.escape(w) + r"(?![^\W\d_])")
        hits = 0
        for m in pat.finditer(text):
            a, b = max(0, m.start() - context), min(len(text), m.end() + context)
            out.append({"word": w, "context": text[a:b].replace("\n", " ")})
            hits += 1
            if hits >= per_word:
                break
    return out


# ------------------------------------------------------------ hunspell (impure)

def hunspell_flag(words, dict_stem, binary="hunspell"):
    """The subset of ``words`` the dictionary at ``dict_stem`` does not know.

    One word per line, so hunspell's own tokenizer never gets to split anything: the
    tokenization is this file's and is under unit test. Returns a set.
    """
    uniq = sorted(set(words))
    if not uniq:
        return set()
    proc = subprocess.run(
        [binary, "-i", "UTF-8", "-d", str(dict_stem), "-l"],
        input="\n".join(uniq) + "\n", capture_output=True, text=True, check=False)
    if proc.returncode != 0 and not proc.stdout:
        raise RuntimeError(f"hunspell failed ({proc.returncode}): {proc.stderr.strip()}")
    flagged = {line.strip() for line in proc.stdout.splitlines() if line.strip()}
    return flagged & set(uniq)


def nonword_report(text, lang, dict_dir, binary="hunspell"):
    """Non-word rate per 1000 checked words, plus every flagged word and its context."""
    words = checkable_words(text)
    stem = dict_dir / DICTS[lang]["stem"]
    flagged = hunspell_flag(words, stem, binary)
    occurrences = [w for w in words if w in flagged]
    counts = {}
    for w in occurrences:
        counts[w] = counts.get(w, 0) + 1
    ordered = sorted(counts, key=lambda w: (-counts[w], w))
    return {
        "lang": lang,
        "checked_words": len(words),
        "flagged_occurrences": len(occurrences),
        "flagged_unique": len(flagged),
        "rate_per_1000": (1000.0 * len(occurrences) / len(words)) if words else None,
        "flagged": [{"word": w, "count": counts[w]} for w in ordered],
        "contexts": word_contexts(text, ordered[:12]),
    }


# --------------------------------------------------------------- dictionaries

def ensure_dicts(dict_dir, fetch=True):
    """The dictionaries, fetched if missing, with URL and sha256 of every file used."""
    dict_dir = Path(dict_dir)
    dict_dir.mkdir(parents=True, exist_ok=True)
    prov = {}
    for lang, spec in DICTS.items():
        files = {}
        for name, rel in spec["files"].items():
            path = dict_dir / name
            url = f"{DICT_BASE}/{rel}"
            if not path.exists():
                if not fetch:
                    raise SystemExit(f"quality-probe: {path} is missing and --no-fetch-dicts is set")
                with urllib.request.urlopen(url, timeout=120) as r:
                    path.write_bytes(r.read())
                print(f"quality-probe: fetched {name} from {url}", file=sys.stderr)
            files[name] = {
                "url": url,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "bytes": path.stat().st_size,
            }
        prov[lang] = {"stem": str(dict_dir / spec["stem"]), "files": files}
    return prov


# --------------------------------------------------------------------- the run

def http_json(url, body=None, timeout=900):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data,
                                 headers={"Content-Type": "application/json"} if data else {})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8"))


def detect_engine(props):
    build = str(props.get("build", ""))
    if build.startswith("crow-nest"):
        return "crow"
    if "build_info" in props or "chat_template" in props:
        return "llama"
    return "unknown"


def build_body(prompt, seed, args, engine):
    body = {
        "model": args.model,
        "messages": [{"role": "system", "content": prompt["system"]},
                     {"role": "user", "content": prompt["user"]}],
        "stream": False,
        "max_tokens": args.max_tokens,
        "seed": seed,
    }
    body.update(ROW)
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    elif engine == "llama":
        # the door llama-server owns: `none` sets enable_thinking = false and drops the key
        body["reasoning_effort"] = "none"
    # crow with thinking off sends NO reasoning_effort: an absent field is the render of record
    return body


def score(text, prompt, dict_dir, binary="hunspell"):
    metrics = {}
    want = set(prompt.get("metrics") or [])
    words = checkable_words(text)
    if "nonword" in want:
        metrics["nonword"] = nonword_report(text, prompt["lang"], dict_dir, binary)
    if "literal" in want and prompt.get("literals"):
        metrics["literal"] = literal_report(text, prompt["literals"])
    if "json" in want:
        value, how = extract_json(text)
        problems = check_shape(value, prompt["json_shape"]) if (
            value is not None and prompt.get("json_shape")) else None
        metrics["json"] = {
            "parsed": value is not None,
            "how": how,
            "shape_ok": (problems == []) if problems is not None else False,
            "problems": problems if problems is not None else ["did not parse"],
        }
    if "repetition" in want:
        metrics["repetition"] = repetition(words)
    if "foreign" in want:
        metrics["foreign"] = foreign_script(text)
    metrics["length"] = {
        "chars": len(text),
        "words": len(words),
        "raw_words": len(text.split()),
        "lines": text.count("\n") + 1 if text else 0,
    }
    return metrics


def run(args):
    doc = json.loads(PROMPTS.read_text(encoding="utf-8"))
    prompts = doc["prompts"]
    if args.only:
        wanted = {p.strip() for p in args.only.split(",") if p.strip()}
        prompts = [p for p in prompts if p["id"] in wanted]
        if not prompts:
            raise SystemExit(f"quality-probe: no prompt matches --only {args.only}")
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]

    props = http_json(args.base_url.rstrip("/") + "/props", timeout=30)
    engine = args.engine or detect_engine(props)
    prov = ensure_dicts(args.dict_dir, fetch=not args.no_fetch_dicts)

    outdir = OUT_ROOT / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    records_path = outdir / "records.jsonl"
    records = []
    started = time.time()
    failures = 0

    with records_path.open("w", encoding="utf-8") as fh:
        for prompt in prompts:
            for seed in seeds:
                body = build_body(prompt, seed, args, engine)
                t0 = time.time()
                err = None
                try:
                    resp = http_json(args.base_url.rstrip("/") + "/v1/chat/completions",
                                     body, timeout=args.timeout)
                except urllib.error.HTTPError as exc:
                    err = f"HTTP {exc.code}: {exc.read().decode('utf-8', 'replace')[:400]}"
                    resp = None
                except Exception as exc:                      # noqa: BLE001 - reported, not raised
                    err = f"{type(exc).__name__}: {exc}"
                    resp = None
                wall = time.time() - t0
                if resp is None:
                    failures += 1
                    rec = {"prompt_id": prompt["id"], "seed": seed, "error": err,
                           "wall_s": round(wall, 3)}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    records.append(rec)
                    print(f"  {prompt['id']} seed {seed}: FAILED {err}", file=sys.stderr)
                    continue
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                text = msg.get("content") or ""
                rec = {
                    "prompt_id": prompt["id"],
                    "lang": prompt["lang"],
                    "kind": prompt["kind"],
                    "seed": seed,
                    "label": args.label,
                    "engine": engine,
                    "model": props.get("model"),
                    "model_path": props.get("model_path"),
                    "sampling": body,
                    "finish_reason": choice.get("finish_reason"),
                    "usage": resp.get("usage"),
                    "timings": resp.get("timings"),
                    "wall_s": round(wall, 3),
                    "content": text,
                    "reasoning_content": msg.get("reasoning_content"),
                    "metrics": score(text, prompt, Path(args.dict_dir), args.hunspell),
                }
                # the messages are already in the prompt file; the record keeps the body
                # WITHOUT them so a records.jsonl stays readable
                rec["sampling"] = {k: v for k, v in body.items() if k != "messages"}
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                nw = rec["metrics"].get("nonword")
                note = f"nonword/1000 {nw['rate_per_1000']:.2f}" if nw and nw["rate_per_1000"] is not None else ""
                lit = rec["metrics"].get("literal")
                if lit:
                    note += f" literals {lit['literals_ok']}/{lit['literals_total']}"
                js = rec["metrics"].get("json")
                if js:
                    note += f" json {'ok' if js['shape_ok'] else 'BAD'}"
                print(f"  {prompt['id']} seed {seed}: {rec['metrics']['length']['words']} words, "
                      f"{wall:.1f} s, {rec['finish_reason']}, {note}")

    header = {
        "label": args.label,
        "date": time.strftime("%Y-%m-%d"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "wall_s": round(time.time() - started, 1),
        "repo_commit": git_commit(),
        "prompt_set_version": doc.get("version"),
        "prompt_ids": [p["id"] for p in prompts],
        "seeds": seeds,
        "base_url": args.base_url,
        "engine": engine,
        "props": props,
        "row": dict(ROW, max_tokens=args.max_tokens),
        "reasoning_effort": args.reasoning_effort or ("none (llama)" if engine == "llama" else "absent (crow)"),
        "hunspell": hunspell_version(args.hunspell),
        "dictionaries": prov,
        "failures": failures,
        "aggregate": aggregate(records),
    }
    (outdir / "run.json").write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    (outdir / "summary.md").write_text(summary_md(header, records), encoding="utf-8")
    print(f"\nquality-probe: {len(records)} records, {failures} failed -> {outdir}")
    return 1 if failures else 0


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:                                          # noqa: BLE001
        return None


def hunspell_version(binary="hunspell"):
    try:
        out = subprocess.run([binary, "-vv"], capture_output=True, text=True, check=False)
        return (out.stdout or out.stderr).splitlines()[0].strip()
    except Exception:                                          # noqa: BLE001
        return None


# ------------------------------------------------------------------ aggregate

def _spread(values):
    """mean, min, max of a list of numbers - a mean without its spread is not a reading."""
    vals = [v for v in values if v is not None]
    if not vals:
        return None
    return {"n": len(vals), "mean": statistics.fmean(vals), "min": min(vals), "max": max(vals),
            "sd": statistics.stdev(vals) if len(vals) > 1 else 0.0}


def aggregate(records):
    ok = [r for r in records if "metrics" in r]
    out = {"generations": len(records), "scored": len(ok)}
    for lang in ("de", "en"):
        rates, words, flags = [], 0, 0
        for r in ok:
            nw = r["metrics"].get("nonword")
            if nw and nw["lang"] == lang and nw["rate_per_1000"] is not None:
                rates.append(nw["rate_per_1000"])
                words += nw["checked_words"]
                flags += nw["flagged_occurrences"]
        out[f"nonword_{lang}"] = {
            "per_generation": _spread(rates),
            "pooled_rate_per_1000": (1000.0 * flags / words) if words else None,
            "checked_words": words,
            "flagged_occurrences": flags,
        }
    lit = [r["metrics"]["literal"] for r in ok if "literal" in r["metrics"]]
    out["literal"] = {
        "generations": len(lit),
        "literals_share": _spread([m["literals_share"] for m in lit]),
        "occurrence_share": _spread([m["occurrence_share"] for m in lit]),
        "near_miss_kinds": sum(len(m["near_misses"]) for m in lit),
    }
    js = [r["metrics"]["json"] for r in ok if "json" in r["metrics"]]
    out["json"] = {
        "generations": len(js),
        "parsed": sum(1 for m in js if m["parsed"]),
        "shape_ok": sum(1 for m in js if m["shape_ok"]),
        "bare": sum(1 for m in js if m["how"] == "bare"),
    }
    rep = [r["metrics"]["repetition"] for r in ok if "repetition" in r["metrics"]]
    out["repetition"] = {
        "distinct_word_ratio": _spread([m["distinct_word_ratio"] for m in rep]),
        "longest_repeat_count": _spread([float(m["longest_repeat_count"]) for m in rep]),
        "max_repeat_count": max([m["longest_repeat_count"] for m in rep], default=None),
    }
    frn = [r["metrics"]["foreign"] for r in ok if "foreign" in r["metrics"]]
    out["foreign"] = {
        "generations": len(frn),
        "with_foreign_chars": sum(1 for m in frn if m["foreign_chars"] > 0),
        "foreign_chars": sum(m["foreign_chars"] for m in frn),
        "greek_chars": sum(m["greek_chars"] for m in frn),
    }
    out["length"] = {
        "words": _spread([float(r["metrics"]["length"]["words"]) for r in ok]),
        "finish_length": sum(1 for r in ok if r.get("finish_reason") == "length"),
        "wall_s": _spread([r.get("wall_s") for r in ok]),
    }
    return out


def _fmt(sp, digits=2):
    if not sp:
        return "n/a"
    return f"{sp['mean']:.{digits}f} ({sp['min']:.{digits}f} to {sp['max']:.{digits}f})"


def summary_md(header, records):
    ok = [r for r in records if "metrics" in r]
    agg = header["aggregate"]
    L = []
    L.append(f"# quality probe - {header['label']}")
    L.append("")
    L.append(f"- date {header['date']}, repo commit `{header['repo_commit']}`, "
             f"prompt set version {header['prompt_set_version']}")
    L.append(f"- endpoint `{header['base_url']}`, engine `{header['engine']}`, "
             f"model `{header['props'].get('model')}`")
    L.append(f"- row temperature {ROW['temperature']}, top_p {ROW['top_p']}, top_k {ROW['top_k']}, "
             f"presence_penalty {ROW['presence_penalty']}, min_p {ROW['min_p']}, "
             f"max_tokens {header['row']['max_tokens']}")
    L.append(f"- thinking: {header['reasoning_effort']}; seeds {header['seeds']}; "
             f"{agg['generations']} generations, {header['failures']} failed, "
             f"{header['wall_s']:.0f} s wall")
    L.append("")
    L.append("## the arm in one table")
    L.append("")
    L.append("| metric | value |")
    L.append("|---|---|")
    L.append(f"| non-word rate DE per 1000 words, per generation | {_fmt(agg['nonword_de']['per_generation'])} |")
    L.append(f"| non-word rate DE, pooled over {agg['nonword_de']['checked_words']} words | "
             f"{agg['nonword_de']['pooled_rate_per_1000']:.2f} |"
             if agg["nonword_de"]["pooled_rate_per_1000"] is not None else "| non-word rate DE pooled | n/a |")
    L.append(f"| non-word rate EN per 1000 words, per generation | {_fmt(agg['nonword_en']['per_generation'])} |")
    L.append(f"| non-word rate EN, pooled over {agg['nonword_en']['checked_words']} words | "
             f"{agg['nonword_en']['pooled_rate_per_1000']:.2f} |"
             if agg["nonword_en"]["pooled_rate_per_1000"] is not None else "| non-word rate EN pooled | n/a |")
    L.append(f"| exact literals reproduced (share of literals) | {_fmt(agg['literal']['literals_share'], 3)} |")
    L.append(f"| exact literals reproduced (share of demanded occurrences) | {_fmt(agg['literal']['occurrence_share'], 3)} |")
    L.append(f"| near-miss literal kinds seen | {agg['literal']['near_miss_kinds']} |")
    L.append(f"| JSON parsed / shape ok | {agg['json']['parsed']} / {agg['json']['shape_ok']} of {agg['json']['generations']} |")
    L.append(f"| distinct-word ratio | {_fmt(agg['repetition']['distinct_word_ratio'], 3)} |")
    L.append(f"| longest immediate repeat run | {_fmt(agg['repetition']['longest_repeat_count'], 1)}, max {agg['repetition']['max_repeat_count']} |")
    L.append(f"| generations with foreign-script characters | {agg['foreign']['with_foreign_chars']} of {agg['foreign']['generations']} ({agg['foreign']['foreign_chars']} chars) |")
    L.append(f"| words per answer | {_fmt(agg['length']['words'], 0)} |")
    L.append(f"| answers stopped at max_tokens | {agg['length']['finish_length']} of {agg['scored']} |")
    L.append("")
    L.append("## per prompt")
    L.append("")
    L.append("| prompt | lang | seed | words | finish | non-word/1000 | flagged | literals | json | longest repeat | foreign |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in records:
        if "metrics" not in r:
            L.append(f"| {r['prompt_id']} | | {r['seed']} | | ERROR | | | | | | |")
            continue
        m = r["metrics"]
        nw = m.get("nonword")
        lit = m.get("literal")
        js = m.get("json")
        rep = m.get("repetition")
        frn = m.get("foreign")
        L.append("| {id} | {lang} | {seed} | {words} | {fin} | {rate} | {flag} | {lit} | {js} | {rep} | {frn} |".format(
            id=r["prompt_id"], lang=r["lang"], seed=r["seed"], words=m["length"]["words"],
            fin=r.get("finish_reason"),
            rate=f"{nw['rate_per_1000']:.2f}" if nw and nw["rate_per_1000"] is not None else "-",
            flag=f"{nw['flagged_occurrences']}/{nw['checked_words']}" if nw else "-",
            lit=f"{lit['literals_ok']}/{lit['literals_total']} ({lit['occurrence_share']:.2f})" if lit else "-",
            js=("ok" if js["shape_ok"] else ("parse-only" if js["parsed"] else "no")) if js else "-",
            rep=f"{rep['longest_repeat_count']}x{rep['longest_repeat_n']}" if rep else "-",
            frn=frn["foreign_chars"] if frn else "-"))
    L.append("")
    L.append("## flagged words, with context")
    L.append("")
    seen = {}
    for r in ok:
        nw = r["metrics"].get("nonword")
        if not nw:
            continue
        for c in nw["contexts"]:
            key = (r["lang"], c["word"])
            if key in seen:
                continue
            seen[key] = (r["prompt_id"], r["seed"], c["context"])
    if not seen:
        L.append("None.")
    else:
        L.append("| lang | word | prompt | seed | context |")
        L.append("|---|---|---|---|---|")
        for (lang, word), (pid, seed, ctx) in sorted(seen.items()):
            L.append(f"| {lang} | `{word}` | {pid} | {seed} | {ctx.replace('|', ' ')} |")
    L.append("")
    L.append("## near-miss literals")
    L.append("")
    rows = []
    for r in ok:
        lit = r["metrics"].get("literal")
        if not lit:
            continue
        for nm in lit["near_misses"]:
            rows.append((r["prompt_id"], r["seed"], nm["literal"], nm["seen"], nm["distance"], nm["count"]))
    if not rows:
        L.append("None: every literal that appeared at all appeared exactly.")
    else:
        L.append("| prompt | seed | demanded | seen | distance | count |")
        L.append("|---|---|---|---|---|---|")
        for row in rows:
            L.append("| {} | {} | `{}` | `{}` | {} | {} |".format(*row))
    L.append("")
    return "\n".join(L) + "\n"


# -------------------------------------------------------------------- compare

def load_label(label):
    d = OUT_ROOT / label
    header = json.loads((d / "run.json").read_text(encoding="utf-8"))
    recs = [json.loads(line) for line in (d / "records.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    return header, recs


def compare(label_a, label_b, out=None):
    ha, ra = load_label(label_a)
    hb, rb = load_label(label_b)
    idx_a = {(r["prompt_id"], r["seed"]): r for r in ra if "metrics" in r}
    idx_b = {(r["prompt_id"], r["seed"]): r for r in rb if "metrics" in r}
    keys = [k for k in idx_a if k in idx_b]
    L = []
    L.append(f"# quality probe - {label_a} against {label_b}")
    L.append("")
    L.append(f"- A `{label_a}`: engine `{ha['engine']}`, model `{ha['props'].get('model')}`, "
             f"{ha['date']}, commit `{ha['repo_commit']}`")
    L.append(f"- B `{label_b}`: engine `{hb['engine']}`, model `{hb['props'].get('model')}`, "
             f"{hb['date']}, commit `{hb['repo_commit']}`")
    L.append(f"- {len(keys)} generation pairs matched by (prompt, seed)")
    L.append("")
    aa, ab = ha["aggregate"], hb["aggregate"]
    L.append("| metric | A | B | B - A |")
    L.append("|---|---|---|---|")

    def row(name, va, vb, digits=2):
        if va is None or vb is None:
            L.append(f"| {name} | {va} | {vb} | n/a |")
        else:
            L.append(f"| {name} | {va:.{digits}f} | {vb:.{digits}f} | {vb - va:+.{digits}f} |")

    row("non-word rate DE per 1000 (pooled)", aa["nonword_de"]["pooled_rate_per_1000"],
        ab["nonword_de"]["pooled_rate_per_1000"])
    row("non-word rate EN per 1000 (pooled)", aa["nonword_en"]["pooled_rate_per_1000"],
        ab["nonword_en"]["pooled_rate_per_1000"])
    row("literal occurrence share", (aa["literal"]["occurrence_share"] or {}).get("mean"),
        (ab["literal"]["occurrence_share"] or {}).get("mean"), 3)
    row("distinct-word ratio", (aa["repetition"]["distinct_word_ratio"] or {}).get("mean"),
        (ab["repetition"]["distinct_word_ratio"] or {}).get("mean"), 3)
    L.append(f"| JSON shape ok | {aa['json']['shape_ok']}/{aa['json']['generations']} | "
             f"{ab['json']['shape_ok']}/{ab['json']['generations']} | |")
    L.append(f"| generations with foreign script | {aa['foreign']['with_foreign_chars']} | "
             f"{ab['foreign']['with_foreign_chars']} | |")
    row("words per answer", (aa["length"]["words"] or {}).get("mean"),
        (ab["length"]["words"] or {}).get("mean"), 0)
    L.append("")
    L.append("## per prompt and seed")
    L.append("")
    L.append("| prompt | seed | non-word/1000 A | B | diff | literals A | B | words A | B |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for k in sorted(keys):
        a, b = idx_a[k], idx_b[k]
        na, nb = a["metrics"].get("nonword"), b["metrics"].get("nonword")
        la, lb = a["metrics"].get("literal"), b["metrics"].get("literal")
        ra_, rb_ = (na["rate_per_1000"] if na else None), (nb["rate_per_1000"] if nb else None)
        diff = f"{rb_ - ra_:+.2f}" if (ra_ is not None and rb_ is not None) else "-"
        L.append("| {p} | {s} | {a} | {b} | {d} | {la} | {lb} | {wa} | {wb} |".format(
            p=k[0], s=k[1],
            a=f"{ra_:.2f}" if ra_ is not None else "-", b=f"{rb_:.2f}" if rb_ is not None else "-",
            d=diff,
            la=f"{la['occurrence_share']:.2f}" if la else "-",
            lb=f"{lb['occurrence_share']:.2f}" if lb else "-",
            wa=a["metrics"]["length"]["words"], wb=b["metrics"]["length"]["words"]))
    text = "\n".join(L) + "\n"
    print(text)
    if out:
        Path(out).write_text(text, encoding="utf-8")
        print(f"quality-probe: wrote {out}", file=sys.stderr)
    return 0


# ------------------------------------------------------------------------ cli

def main(argv):
    ap = argparse.ArgumentParser(description="the #75 quality probe")
    ap.add_argument("--label", help="run label; output goes to decode_out/quality-probe/<label>/")
    ap.add_argument("--base-url", default="http://127.0.0.1:8099")
    ap.add_argument("--engine", choices=["crow", "llama"],
                    help="how thinking is switched off; auto-detected from /props when absent")
    ap.add_argument("--model", default="crow-nest", help="the `model` field of the request")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--max-tokens", type=int, default=2600)
    ap.add_argument("--timeout", type=float, default=900.0)
    ap.add_argument("--reasoning-effort", help="repeat with thinking ON at this level")
    ap.add_argument("--only", help="comma-separated prompt ids, for a subset run")
    ap.add_argument("--dict-dir", default=str(DEFAULT_DICT_DIR))
    ap.add_argument("--no-fetch-dicts", action="store_true")
    ap.add_argument("--hunspell", default="hunspell")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    ap.add_argument("--out", help="with --compare: also write the table to this file")
    args = ap.parse_args(argv[1:])

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.out)
    if not args.label:
        ap.error("--label is required for a run")
    if shutil.which(args.hunspell) is None:
        raise SystemExit(f"quality-probe: {args.hunspell} is not on PATH")
    return run(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
