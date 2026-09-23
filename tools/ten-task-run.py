#!/usr/bin/env python3
"""#80: the ten-task runner - the seed band at the REAL operating row, against any
OpenAI-compatible `/v1/chat/completions` endpoint.

Usage, from anywhere:
  tools/ten-task-run.py run   --label A-crow  --base-url http://127.0.0.1:8099
  tools/ten-task-run.py run   --label B-llama --base-url http://127.0.0.1:8083
  tools/ten-task-run.py tokens --label A-crow            # exact answer/thinking token split
  tools/ten-task-run.py blind --labels A-crow,B-llama    # the shuffled file + the SEALED id map
  tools/ten-task-run.py mech  --label A-crow             # re-run the mechanical checks only
  tools/ten-task-run.py report --labels A-crow,B-llama --grades <grades.json>
  python tools/test_ten_task_run.py                      # the pure functions, no server, no GPU

WHY THIS EXISTS
---------------
The ten-task record of `docs/ten-tasks.md` is a WINDOWS GREEDY realization, and
`docs/architecture.md` 5.4 measured that Linux does not reproduce it: nine of ten answers
differ with no flag set at all, and a greedy re-run is the SAME sample, not a second one.
One greedy draw therefore cannot discriminate anything. #80 asks the question the product is
actually for: at the row Crow really sends (temperature 1.0, top_p 0.95, top_k 20,
min_p 0.0, presence_penalty 0.0) with thinking ON, are the ANSWERS of the two engines equally
good on the nine ENGLISH tasks? This tool produces the generations and the checks that need
no judge; the judgement itself is a separate, BLIND step this tool only prepares and joins.

WHAT IT IS NOT
--------------
It is not a gate and it decides nothing on its own. It writes a record. The grading rubric
lives in `docs/ten-task-expected.md` and the verdict lives in `docs/ten-task-linux.md`.

THE ROW, AND WHY BOTH ARMS GET THE SAME WIRE BODY
-------------------------------------------------
The sampling row is `tools/quality-probe.py`'s `ROW`, imported rather than copied. Thinking is
switched on with a TOP-LEVEL `"reasoning_effort": "high"`, which is the same STEP on both
engines and not the same word:

| engine | what `high` does |
|---|---|
| `serve` | maps to the template's `xhigh` (#74, `docs/architecture.md` 7.11.20) |
| llama-server | the unsloth template renders `high` byte-identically to its default (Crow manifest `flash-next-q2-k-xl`, `reasoning_groups [["off","high"],["low"],["medium"],["none"]]`) |

NO reasoning budget is sent to either arm, so both think freely: `serve` has none at all, and
llama-server only gets one when Crow sends it (manifest `reasoning_budget: 1024`, a CLIENT
field; the server line of `start-server.py` carries no `--reasoning-budget`).

`max_tokens` is 16384 for EVERY task on both arms - it has to cover thinking PLUS answer, and
the frozen budgets in `decode_out/ten-tasks.json` (1024 to 1536) were fixed for thinking OFF.
`serve` caps `max_tokens` at 32768 and clamps it to the free context (`serve.rs`
`clamped_max_tokens`), so 16384 arrives unclamped on every prompt here. A generation that ends
with `finish_reason length` is RECORDED AS A FINDING and never retried.

`cache_prompt: false` goes to BOTH arms: llama-server reads it (the prompt cache contaminated
an answer once - `probes/p5_STATUS.md`, "one series per server session"), `serve` ignores every
field it does not know (`serve.rs` `parse_chat`), so the wire body is identical on both arms
except for nothing at all. `serve`'s own prefix cache may reuse the PROMPT prefix; what must
differ per seed is the sampled continuation, and `blind`/`report` check exactly that by
comparing the three seeds' texts of every (task, arm) cell.

Each generation is its own fresh single-turn conversation: one `user` message carrying the
frozen task text, no system prompt, no tools.

THE MECHANICAL CHECKS
---------------------
These need no judge and are computed for every generation, on both arms, from the stored text:

| task | check |
|---|---|
| `t2-write` | the C++ compiles (`g++ -std=c++17 -fsyntax-only`), and when it has a `main()` it is built and RUN, so the answer's own `assert()` tests are the verdict |
| `t2b-write-refactor` | the C++ compiles (`-fsyntax-only`; the answer is a function, not a program) |
| `t6b-reason-multi` | the five exact intermediate numbers of `docs/ten-task-expected.md` 2.10 |
| `t6-reason` | the three worked answers 3 / 4 / -1 and the algorithmic markers that can be matched at all |
| `t4-prose` | the 400-word limit of the prompt |
| all | answer language = prompt language (Rev3 core requirement 0), foreign script, repetition |

Numbers are matched with their thousands separators optional and in any of `. , _` or a space,
so `6144`, `6,144` and `6.144` all count.

THE OUTPUT
----------
`decode_out/ten-task/<label>/`:

- `records.jsonl` - one JSON object per generation: task id, seed, the full `content`, the full
  `reasoning_content`, finish reason, token counts, timings, the sampling row that was sent,
  the server's `/props` model name, and the mechanical checks.
- `run.json` - the run header: date, repo commit, endpoint, props, the row, what the server
  said about thinking, and the aggregate.
- `tokens.json` - written by `tokens`: the exact answer-token count per record from the
  reference tokenizer of the oracle venv, and the thinking tokens derived from it.

`decode_out/ten-task/blind/`:

- `answers.md` - every answer under an OPAQUE id in a shuffled order, with its task id and
  NOTHING else: no arm, no seed, no timing, no model name.
- `idmap.json` - the seal. Written by `blind`, NOT read again until the grades exist.
- `regrade.md` - a second shuffled file over a 20 % sample, for the grader's self-consistency.

Exit code is 0 when the run completed, 1 when a request failed, 2 on a usage error.
"""

import argparse
import hashlib
import importlib.util
import json
import random
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
TOOLS = REPO / "tools"
TASKS = REPO / "decode_out" / "ten-tasks.json"
OUT_ROOT = REPO / "decode_out" / "ten-task"
ORACLE_PY = REPO / ".venv-oracle" / "bin" / "python"

# The probe is the same repo and the same style, and its plumbing and its reference-free
# metrics are IMPORTED, never copied: one scorer, one place to correct it.
_SPEC = importlib.util.spec_from_file_location("quality_probe", TOOLS / "quality-probe.py")
qp = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(qp)

ROW = qp.ROW                       # temperature 1.0, top_p 0.95, top_k 20, min_p 0.0, presence 0.0
DEFAULT_SEEDS = (2101, 2102, 2103)
DEFAULT_MAX_TOKENS = 16384
DEFAULT_EFFORT = "high"
GERMAN_TASK = "t4-prose"           # the one German prompt; reported separately, never in the verdict


# ------------------------------------------------------------------ pure: text shapes

FENCE_RE = re.compile(r"```([A-Za-z0-9_+#.-]*)[ \t]*\r?\n(.*?)(?:```|\Z)", re.S)
CPP_LANGS = {"", "cpp", "c++", "cc", "cxx", "c", "h", "hpp", "hh", "cu", "cuda", "text"}


def fenced_blocks(text):
    """Every fenced block as (language tag lowercased, body). An unclosed fence runs to the end
    of the text - that is what a truncated answer leaves behind, and it is still code."""
    return [(m.group(1).lower(), m.group(2)) for m in FENCE_RE.finditer(text or "")]


# One separator character, optional: `6144`, `6,144`, `6.144`, `6 144` and `6_144` are the same
# number written five ways, and a grader that only knows one of them measures its own regex.
_SEP = r"[ \t  .,_'’]?"


def grouped_pattern(n):
    """The regex body that matches the integer `n` with optional thousands separators."""
    s = str(int(n))
    if len(s) <= 3:
        return re.escape(s)
    head = len(s) % 3 or 3
    parts = [s[:head]] + [s[i:i + 3] for i in range(head, len(s), 3)]
    return _SEP.join(re.escape(p) for p in parts)


def int_present(text, n):
    """Is the integer `n` written anywhere in `text`, separators optional?"""
    pat = r"(?<![0-9])" + grouped_pattern(n) + r"(?![0-9])"
    return re.search(pat, text or "") is not None


def decimal_present(text, whole, frac):
    """`728.25` and `728,25` are the same number; `7282.5` and `1728.25` are not."""
    pat = (r"(?<![0-9])" + grouped_pattern(whole) + r"[.,]" + re.escape(str(frac))
           + r"(?![0-9])")
    return re.search(pat, text or "") is not None


def percent_present(text, n):
    pat = r"(?<![0-9])" + re.escape(str(n)) + r"\s*(?:%|percent|per cent|Prozent)"
    return re.search(pat, text or "", re.I) is not None


# ------------------------------------------------------------------ pure: language

# Rev3 core requirement 0: answer language = prompt language. Nine prompts are English, the
# tenth (t4-prose) is German. Function words decide it; content words are the same in a
# technical answer either way.
DE_WORDS = {
    "der", "die", "das", "und", "ist", "nicht", "ein", "eine", "einen", "einem", "einer",
    "den", "dem", "des", "mit", "auf", "fuer", "für", "von", "im", "zu", "zur", "zum",
    "wird", "werden", "sich", "auch", "als", "aber", "dass", "wenn", "man", "es", "bei",
    "nach", "über", "ueber", "diese", "dieser", "dieses", "kann", "muss", "nur", "oder",
    "sind", "wie", "um", "vor", "dann", "weil", "damit", "durch", "sie", "wir", "noch",
    "schon", "wäre", "waere", "haelt", "hält", "beim", "ohne", "gegen", "ihre", "sein",
}
EN_WORDS = {
    "the", "and", "is", "of", "to", "in", "that", "it", "for", "on", "with", "as", "are",
    "this", "be", "by", "not", "or", "from", "at", "which", "but", "can", "will", "if",
    "we", "you", "there", "has", "have", "was", "were", "an", "when", "would", "should",
    "because", "into", "than", "then", "its", "they", "what", "how", "so", "does", "do",
}
_WORD_RE = re.compile(r"[A-Za-zÄÖÜäöüß]+")


def detect_lang(text):
    """`de`, `en` or `unknown`, from function words in the PROSE only (code is stripped by the
    probe's own `strip_uncheckable`). Returns the counts too, so a close call is visible."""
    prose = qp.strip_uncheckable(text or "")
    words = [w.lower() for w in _WORD_RE.findall(prose)]
    de = sum(1 for w in words if w in DE_WORDS)
    en = sum(1 for w in words if w in EN_WORDS)
    if de == 0 and en == 0:
        lang = "unknown"
    elif de > en:
        lang = "de"
    elif en > de:
        lang = "en"
    else:
        lang = "unknown"
    return {"lang": lang, "de_hits": de, "en_hits": en, "words": len(words)}


def word_count(text):
    """The prompt's own unit for t4-prose: whitespace-separated words of the whole answer."""
    return len((text or "").split())


# ------------------------------------------------------------------ pure: task checks

# docs/ten-task-expected.md 2.10 - every one of the five intermediate values must appear, and
# a single wrong one makes the core requirement wrong. This finds the RIGHT ones; it cannot
# see a wrong one that is also written down, which is what the blind grading is for.
T6B_STEPS = {
    "step1_6144": ("int", 6144),
    "step2_1610612736": ("int", 1610612736),
    "step2_1_5_gib": ("dec", (1, 5)),
    "step3_524288": ("int", 524288),
    "step3_budget_3221225472": ("int", 3221225472),
    "step4_12288": ("int", 12288),
    "step4_overhead_100pct": ("pct", 100),
    "step5_2457600000": ("int", 2457600000),
    "step5_margin_763625472": ("int", 763625472),
    "step5_margin_728_25": ("dec", (728, 25)),
}


def t6b_numbers(text):
    out = {}
    for name, (kind, val) in T6B_STEPS.items():
        if kind == "int":
            out[name] = int_present(text, val)
        elif kind == "dec":
            out[name] = decimal_present(text, val[0], val[1])
        else:
            out[name] = percent_present(text, val)
    out["verdict_pass_uppercase"] = re.search(r"\bPASS\b", text or "") is not None
    out["verdict_pass_any_case"] = re.search(r"\bpass(?:es|ed)?\b", text or "", re.I) is not None
    # the margin may be given either way round; step 5 counts when one of the two is there
    out["step5_margin_either"] = out["step5_margin_763625472"] or out["step5_margin_728_25"]
    required = ["step1_6144", "step2_1610612736", "step2_1_5_gib", "step3_524288",
                "step4_12288", "step4_overhead_100pct", "step5_2457600000",
                "step5_margin_either"]
    out["all_five_steps_exact"] = all(out[k] for k in required)
    out["missing"] = [k for k in required if not out[k]]
    return out


# docs/ten-task-expected.md 2.6 - the three worked queries and the markers of the intended
# algorithm. The answers are EXTRACTED, not merely searched for, so `3` somewhere in the prose
# cannot pass for the answer to query 1.
_Q_PATTERNS = {
    "q1_1_5_2": (r"\(?\s*1\s*,\s*5\s*,\s*2\s*\)?", 3),
    "q2_2_7_3": (r"\(?\s*2\s*,\s*7\s*,\s*3\s*\)?", 4),
    "q3_4_4_2": (r"\(?\s*4\s*,\s*4\s*,\s*2\s*\)?", -1),
}
_ANSWER_AFTER = re.compile(r"(?:=|->|→|⇒|=>|\banswer\b|\bresult\b|\breturns?\b|\byields?\b|"
                           r"\bgives?\b|\bis\b|\bequals\b|\boutput\b)"
                           r"[^0-9\-\n]{0,40}(-?\d+)", re.I)


def t6_answers(text):
    """For each of the three queries: the number the answer states for it, or None.

    The window is the query's own line plus the two lines after it, which is where a worked
    case puts its result; the LAST stated value in that window wins, because a derivation
    lists intermediates before it concludes."""
    text = text or ""
    lines = text.splitlines()
    out = {}
    anyq = [re.compile(p) for p, _ in _Q_PATTERNS.values()]
    for name, (pat, want) in _Q_PATTERNS.items():
        rx = re.compile(pat)
        found = None
        for i, line in enumerate(lines):
            if not rx.search(line):
                continue
            # the window stops at the NEXT query: query 2's result is not query 1's
            window = [line]
            for j in range(i + 1, min(i + 3, len(lines))):
                if any(q.search(lines[j]) for q in anyq):
                    break
                window.append(lines[j])
            hits = _ANSWER_AFTER.findall("\n".join(window))
            if hits:
                found = int(hits[-1])
        out[name] = {"stated": found, "expected": want, "ok": found == want}
    out["all_three_ok"] = all(out[n]["ok"] for n in _Q_PATTERNS)
    return out


def t6_markers(text):
    t = text or ""
    return {
        "last_occurrence": bool(re.search(r"last occurrence|last[_ ]occ|latest occurrence|"
                                          r"most recent occurrence|prev\s*\[|previous occurrence",
                                          t, re.I)),
        "persistent_or_bit": bool(re.search(r"persistent segment tree|persistent seg|wavelet|"
                                            r"\bBIT\b|fenwick|merge sort tree", t, re.I)),
        "offline_by_r": bool(re.search(r"offline|sort(?:ed)? (?:the )?quer(?:y|ies) by r|"
                                       r"sweep|by increasing r", t, re.I)),
        "complexity_stated": bool(re.search(r"O\s*\(\s*\(?\s*[nq].{0,20}\blog", t, re.I)),
        "minus_one_path": bool(re.search(r"-\s?1", t)),
    }


# ------------------------------------------------------------------ pure-ish: the compiler

CPP_PRELUDE = "\n".join("#include <%s>" % h for h in (
    "cstddef", "cstdint", "cstring", "cstdio", "cstdlib", "cassert", "string", "vector",
    "list", "unordered_map", "map", "optional", "stdexcept", "utility", "algorithm",
    "iostream", "memory", "functional", "initializer_list")) + "\n"


def cpp_variants(text):
    """The translation units to try, in order. `all` is every C++ block of the answer in the
    order it wrote them - which is what "a header plus a main()" means. `largest` is the single
    biggest block, the fallback for an answer that also prints an alternative or a usage
    snippet and would otherwise redefine its own symbols."""
    blocks = [b for (lang, b) in fenced_blocks(text) if lang in CPP_LANGS and b.strip()]
    if not blocks:
        return []
    out = [("all", CPP_PRELUDE + "\n".join(blocks))]
    largest = max(blocks, key=len)
    if len(blocks) > 1:
        out.append(("largest", CPP_PRELUDE + largest))
    return out


def compile_check(text, cxx="g++", std="c++17", run_binary=False, timeout=120, workdir=None):
    """`-fsyntax-only` on the answer's own code, and - when it carries a `main()` and
    `run_binary` is set - a real build and run, so the answer's `assert()` tests decide.

    Never raises: a missing compiler, a timeout and a compile error are all RESULTS."""
    variants = cpp_variants(text)
    res = {"blocks": len(fenced_blocks(text)), "variants_tried": [], "compiles": False,
           "variant": None, "ran": None, "exit_code": None, "stderr_head": None}
    if not variants:
        res["stderr_head"] = "no fenced C++ block in the answer"
        return res
    tmp = Path(tempfile.mkdtemp(prefix="tentask-cpp-", dir=str(workdir) if workdir else None))
    try:
        _compile_variants(variants, res, tmp, cxx, std, run_binary, timeout)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return res


def _compile_variants(variants, res, tmp, cxx, std, run_binary, timeout):
    for name, src in variants:
        res["variants_tried"].append(name)
        path = tmp / ("%s.cpp" % name)
        path.write_text(src, encoding="utf-8")
        cmd = [cxx, "-std=" + std, "-w", "-fsyntax-only", str(path)]
        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except FileNotFoundError:
            res["stderr_head"] = "%s is not on PATH" % cxx
            return res
        except subprocess.TimeoutExpired:
            res["stderr_head"] = "compiler timed out after %g s" % timeout
            continue
        if proc.returncode != 0:
            res["stderr_head"] = proc.stderr.strip()[:600]
            continue
        res["compiles"] = True
        res["variant"] = name
        res["stderr_head"] = None
        if run_binary and re.search(r"\bint\s+main\s*\(", src):
            exe = tmp / ("%s.bin" % name)
            try:
                build = subprocess.run([cxx, "-std=" + std, "-w", "-o", str(exe), str(path)],
                                       capture_output=True, text=True, timeout=timeout)
            except subprocess.TimeoutExpired:
                res["ran"] = False
                res["stderr_head"] = "link timed out"
                break
            if build.returncode != 0:
                res["ran"] = False
                res["stderr_head"] = build.stderr.strip()[:600]
                break
            try:
                out = subprocess.run([str(exe)], capture_output=True, text=True, timeout=20)
                res["ran"] = True
                res["exit_code"] = out.returncode
                if out.returncode != 0:
                    res["stderr_head"] = (out.stderr or out.stdout).strip()[:600]
            except subprocess.TimeoutExpired:
                res["ran"] = True
                res["exit_code"] = "timeout"
                res["stderr_head"] = "the answer's own tests did not terminate in 20 s"
        break


def mechanical(task_id, content, cxx="g++", workdir=None):
    """Every check that needs no judge, for one answer."""
    expected = "de" if task_id == GERMAN_TASK else "en"
    lang = detect_lang(content)
    m = {
        "expected_lang": expected,
        "language": lang,
        "language_ok": lang["lang"] == expected,
        "foreign_script": qp.foreign_script(content or ""),
        "repetition": qp.repetition(qp.checkable_words(content or "")),
        "raw_words": word_count(content),
        "chars": len(content or ""),
    }
    if task_id == "t4-prose":
        m["word_limit"] = {"limit": 400, "words": m["raw_words"],
                           "within_400": m["raw_words"] <= 400,
                           "within_440_tolerance": m["raw_words"] <= 440}
    if task_id == "t2-write":
        m["cpp"] = compile_check(content, cxx=cxx, run_binary=True, workdir=workdir)
    if task_id == "t2b-write-refactor":
        m["cpp"] = compile_check(content, cxx=cxx, run_binary=False, workdir=workdir)
    if task_id == "t6b-reason-multi":
        m["numbers"] = t6b_numbers(content)
    if task_id == "t6-reason":
        m["worked_case"] = t6_answers(content)
        m["markers"] = t6_markers(content)
    return m


# ------------------------------------------------------------------ the run

def load_tasks(only=None):
    tasks = json.loads(TASKS.read_text(encoding="utf-8"))
    if only:
        wanted = {t.strip() for t in only.split(",") if t.strip()}
        tasks = [t for t in tasks if t["id"] in wanted]
        if not tasks:
            raise SystemExit("ten-task-run: no task matches --only %s" % only)
    return tasks


def build_body(task, seed, args):
    body = {
        "model": args.model,
        "messages": [{"role": "user", "content": task["text"]}],
        "stream": False,
        "max_tokens": args.max_tokens,
        "seed": seed,
        # llama-server reads it; `serve` ignores every unknown field, so the wire body is the
        # same on both arms and no prompt cache can carry an answer between two seeds
        "cache_prompt": False,
    }
    body.update(ROW)
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    return body


def thinking_note(engine, effort, props):
    """What the server says about thinking, for the record."""
    note = {"sent": effort, "engine": engine}
    if engine == "crow":
        note["maps_to"] = "xhigh (serve.rs map_reasoning_effort, #74)"
    elif engine == "llama":
        note["maps_to"] = ("the unsloth template renders `high` identically to its default "
                           "(manifest reasoning_groups [[off,high],[low],[medium],[none]])")
    note["chat_template_sha256"] = hashlib.sha256(
        (props.get("chat_template") or "").encode("utf-8")).hexdigest()[:16] if props.get(
        "chat_template") else None
    return note


def run(args):
    tasks = load_tasks(args.only)
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    base = args.base_url.rstrip("/")
    props = qp.http_json(base + "/props", timeout=60)
    engine = args.engine or qp.detect_engine(props)
    outdir = OUT_ROOT / args.label
    outdir.mkdir(parents=True, exist_ok=True)
    records_path = outdir / "records.jsonl"
    records, failures = [], 0
    started = time.time()

    with records_path.open("w", encoding="utf-8") as fh:
        for task in tasks:
            for seed in seeds:
                body = build_body(task, seed, args)
                t0 = time.time()
                err, resp = None, None
                try:
                    resp = qp.http_json(base + "/v1/chat/completions", body, timeout=args.timeout)
                except urllib.error.HTTPError as exc:
                    err = "HTTP %d: %s" % (exc.code, exc.read().decode("utf-8", "replace")[:400])
                except Exception as exc:                       # noqa: BLE001 - reported, not raised
                    err = "%s: %s" % (type(exc).__name__, exc)
                wall = time.time() - t0
                if resp is None:
                    failures += 1
                    rec = {"task": task["id"], "seed": seed, "label": args.label,
                           "error": err, "wall_s": round(wall, 3)}
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                    records.append(rec)
                    print("  %s seed %d: FAILED %s" % (task["id"], seed, err),
                          file=sys.stderr, flush=True)
                    continue
                choice = (resp.get("choices") or [{}])[0]
                msg = choice.get("message") or {}
                content = msg.get("content") or ""
                rec = {
                    "task": task["id"],
                    "seed": seed,
                    "label": args.label,
                    "engine": engine,
                    "model": qp.props_model(props),
                    "model_path": props.get("model_path"),
                    "sampling": {k: v for k, v in body.items() if k != "messages"},
                    "prompt_chars": len(task["text"]),
                    "finish_reason": choice.get("finish_reason"),
                    "usage": resp.get("usage"),
                    "timings": resp.get("timings"),
                    "wall_s": round(wall, 3),
                    "content": content,
                    "reasoning_content": msg.get("reasoning_content"),
                    "content_chars": len(content),
                    "reasoning_chars": len(msg.get("reasoning_content") or ""),
                    "mechanical": mechanical(task["id"], content, cxx=args.cxx),
                }
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                records.append(rec)
                usage = rec["usage"] or {}
                print("  %-18s seed %d: %6.1f s, finish %-6s, completion %5s tok, "
                      "think %6d ch, answer %6d ch" %
                      (task["id"], seed, wall, rec["finish_reason"],
                       usage.get("completion_tokens"), rec["reasoning_chars"],
                       rec["content_chars"]), flush=True)

    header = {
        "label": args.label,
        "issue": 80,
        "date": time.strftime("%Y-%m-%d"),
        "started": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(started)),
        "wall_s": round(time.time() - started, 1),
        "repo_commit": qp.git_commit(),
        "base_url": args.base_url,
        "engine": engine,
        "props": props,
        "tasks": [t["id"] for t in tasks],
        "seeds": seeds,
        "row": dict(ROW, max_tokens=args.max_tokens, cache_prompt=False),
        "thinking": thinking_note(engine, args.reasoning_effort, props),
        "reasoning_budget_sent": None,
        "failures": failures,
        "finish_length": [(r["task"], r["seed"]) for r in records
                          if r.get("finish_reason") == "length"],
        "cxx": compiler_version(args.cxx),
    }
    (outdir / "run.json").write_text(json.dumps(header, ensure_ascii=False, indent=2) + "\n",
                                     encoding="utf-8")
    print("\nten-task-run: %d records, %d failed -> %s" % (len(records), failures, outdir))
    return 1 if failures else 0


def compiler_version(cxx="g++"):
    try:
        out = subprocess.run([cxx, "--version"], capture_output=True, text=True, check=False)
        return out.stdout.splitlines()[0].strip()
    except Exception:                                          # noqa: BLE001
        return None


# ------------------------------------------------------------------ the token split

def load_records(label):
    path = OUT_ROOT / label / "records.jsonl"
    if not path.exists():
        raise SystemExit("ten-task-run: no records at %s" % path)
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def tokens(args):
    """The exact answer-token count from the REFERENCE tokenizer, one interpreter call for the
    whole label. Both arms serve the same base model, so one tokenizer counts both arms in the
    same unit; thinking tokens are what the server's own `completion_tokens` has left over."""
    if not ORACLE_PY.exists():
        raise SystemExit("ten-task-run: the oracle venv is not at %s" % ORACLE_PY)
    recs = load_records(args.label)
    texts = [r.get("content") or "" for r in recs]
    think = [r.get("reasoning_content") or "" for r in recs]
    script = (
        "import json,sys\n"
        "from transformers import AutoTokenizer\n"
        "tok = AutoTokenizer.from_pretrained('models/Qwen3.8-Flash-Next-original')\n"
        "d = json.load(sys.stdin)\n"
        "out = {k: [len(tok(t, add_special_tokens=False)['input_ids']) for t in v]\n"
        "       for k, v in d.items()}\n"
        "sys.stdout.write('@@' + json.dumps(out))\n"
    )
    proc = subprocess.run([str(ORACLE_PY), "-c", script], cwd=str(REPO),
                          input=json.dumps({"content": texts, "reasoning": think}),
                          capture_output=True, text=True, timeout=1800)
    if proc.returncode != 0 or "@@" not in proc.stdout:
        raise SystemExit("ten-task-run: the tokenizer failed:\n" + proc.stderr[-2000:])
    counts = json.loads(proc.stdout.split("@@", 1)[1])
    out = []
    for i, r in enumerate(recs):
        total = ((r.get("usage") or {}).get("completion_tokens"))
        answer = counts["content"][i]
        reason = counts["reasoning"][i]
        out.append({
            "task": r["task"], "seed": r["seed"],
            "completion_tokens_server": total,
            "answer_tokens": answer,
            "reasoning_tokens": reason,
            # what the server counted minus what the two texts tokenize to: the template's own
            # control tokens between the thinking block and the answer
            "unaccounted": (total - answer - reason) if isinstance(total, int) else None,
        })
    path = OUT_ROOT / args.label / "tokens.json"
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print("ten-task-run: %d token rows -> %s" % (len(out), path))
    return 0


# ------------------------------------------------------------------ the blind file

def opaque_id(label, task, seed, salt):
    h = hashlib.sha256(("%s|%s|%d|%s" % (label, task, seed, salt)).encode("utf-8")).hexdigest()
    return h[:8].upper()


def blind(args):
    """The shuffled file with opaque ids and NO arm label, and the sealed id map beside it.

    The task id STAYS - a checklist is per task and a grader cannot work without it - and
    everything that names the arm goes: label, engine, model, seed, timings, finish reason."""
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    salt = args.salt or hashlib.sha256(str(time.time()).encode()).hexdigest()[:16]
    entries = []
    for label in labels:
        for r in load_records(label):
            if r.get("error"):
                continue
            entries.append({
                "id": opaque_id(label, r["task"], r["seed"], salt),
                "task": r["task"],
                "content": r.get("content") or "",
                "_label": label, "_seed": r["seed"],
            })
    rng = random.Random(args.shuffle_seed)
    rng.shuffle(entries)

    outdir = OUT_ROOT / "blind"
    outdir.mkdir(parents=True, exist_ok=True)
    body = ["# Blind answer file - #80", "",
            "%d answers, shuffled. The task id is the only thing that identifies an answer;"
            % len(entries),
            "there is no arm, no seed, no timing and no model name in this file.", ""]
    for e in entries:
        body += ["", "---", "", "## %s  (task: %s)" % (e["id"], e["task"]), "",
                 "```text", e["content"] if e["content"].strip() else "(empty answer)", "```", ""]
    (outdir / args.out).write_text("\n".join(body), encoding="utf-8")

    idmap = {e["id"]: {"label": e["_label"], "task": e["task"], "seed": e["_seed"]}
             for e in entries}
    mappath = outdir / args.map_out
    mappath.write_text(json.dumps(idmap, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    seal = hashlib.sha256(mappath.read_bytes()).hexdigest()
    (outdir / (args.map_out + ".sha256")).write_text(seal + "\n", encoding="utf-8")
    print("ten-task-run: %d answers -> %s" % (len(entries), outdir / args.out))
    print("ten-task-run: the seal is %s (%s)" % (seal[:16], mappath))
    # the sample for the grader's own consistency, drawn from the ids alone
    rng2 = random.Random(args.shuffle_seed + 1)
    sample = rng2.sample([e["id"] for e in entries],
                         max(1, round(len(entries) * args.regrade_share)))
    order = list(sample)
    rng2.shuffle(order)
    by_id = {e["id"]: e for e in entries}
    body2 = ["# Re-grade file - #80", "",
             "%d answers (%.0f %% of the set), a FRESH shuffled order, same opaque ids."
             % (len(order), 100 * args.regrade_share), ""]
    for i in order:
        e = by_id[i]
        body2 += ["", "---", "", "## %s  (task: %s)" % (e["id"], e["task"]), "",
                  "```text", e["content"] if e["content"].strip() else "(empty answer)",
                  "```", ""]
    (outdir / args.regrade_out).write_text("\n".join(body2), encoding="utf-8")
    print("ten-task-run: %d re-grade answers -> %s" % (len(order), outdir / args.regrade_out))
    return 0


# ------------------------------------------------------------------ the report

def _mean(xs):
    xs = [x for x in xs if x is not None]
    return statistics.fmean(xs) if xs else None


def wilson(k, n, z=1.959963984540054):
    """The Wilson score interval - an exact-enough interval for a pass count out of n, and it
    does not collapse to a point at 0 or n the way the normal approximation does."""
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / d
    half = (z / d) * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5)
    return (max(0.0, centre - half), min(1.0, centre + half))


def seed_diversity(records):
    """Did a new seed really re-sample? Per (task, label) cell: how many DISTINCT answers the
    three seeds produced, and how many distinct THINKING blocks.

    The thinking column is the one that answers the question. A cell whose three answers are
    the same string because all three are EMPTY - the whole budget went into thinking - says
    nothing about the sampler; three distinct thinking blocks say the seed re-sampled."""
    cells = {}
    for r in records:
        if r.get("error"):
            continue
        cells.setdefault((r["label"], r["task"]), []).append(
            (r.get("content") or "", r.get("reasoning_content") or ""))
    return {"%s/%s" % k: {"n": len(v),
                          "distinct_answers": len({a for a, _ in v}),
                          "distinct_thinking": len({t for _, t in v}),
                          "empty_answers": sum(1 for a, _ in v if not a)}
            for k, v in sorted(cells.items())}


def report(args):
    labels = [l.strip() for l in args.labels.split(",") if l.strip()]
    grades = json.loads(Path(args.grades).read_text(encoding="utf-8")) if args.grades else {}
    idmap_path = OUT_ROOT / "blind" / args.map_out
    idmap = json.loads(idmap_path.read_text(encoding="utf-8")) if idmap_path.exists() else {}
    allrecs, toks = [], {}
    for label in labels:
        allrecs += load_records(label)
        tp = OUT_ROOT / label / "tokens.json"
        if tp.exists():
            for row in json.loads(tp.read_text(encoding="utf-8")):
                toks[(label, row["task"], row["seed"])] = row

    # join the blind grades back to their arm
    joined = {}
    for oid, g in grades.items():
        who = idmap.get(oid)
        if who:
            joined[(who["label"], who["task"], who["seed"])] = g

    out = {"labels": labels, "seed_diversity": seed_diversity(allrecs), "per_task": {},
           "totals": {}}
    tasks = [t["id"] for t in load_tasks(None)]
    for task in tasks:
        row = {}
        for label in labels:
            rs = [r for r in allrecs if r["label"] == label and r["task"] == task]
            gs = [joined.get((label, task, r["seed"])) for r in rs]
            met = [g["share_met"] for g in gs if g and "share_met" in g]
            passes = sum(1 for g in gs if g and g.get("verdict") == "pass")
            tk = [toks.get((label, task, r["seed"])) for r in rs]
            row[label] = {
                "n": len(rs),
                "passes": passes,
                "share_met_mean": _mean(met),
                "finish": sorted({r.get("finish_reason") for r in rs}),
                "finish_length": sum(1 for r in rs if r.get("finish_reason") == "length"),
                "answer_tokens_mean": _mean([t["answer_tokens"] for t in tk if t]),
                "reasoning_tokens_mean": _mean([t["reasoning_tokens"] for t in tk if t]),
                "wall_s_mean": _mean([r.get("wall_s") for r in rs]),
            }
        out["per_task"][task] = row

    english = [t for t in tasks if t != GERMAN_TASK]
    for label in labels:
        k = sum(out["per_task"][t][label]["passes"] for t in english)
        n = sum(out["per_task"][t][label]["n"] for t in english)
        lo, hi = wilson(k, n)
        shares = [out["per_task"][t][label]["share_met_mean"] for t in english]
        out["totals"][label] = {
            "english_passes": k, "english_n": n,
            "english_pass_rate": (k / n) if n else None,
            "wilson95": [round(lo, 4), round(hi, 4)],
            "english_share_met_mean": _mean(shares),
            "german": out["per_task"][GERMAN_TASK][label],
        }
    txt = json.dumps(out, indent=2, ensure_ascii=False) + "\n"
    if args.out:
        Path(args.out).write_text(txt, encoding="utf-8")
        print("ten-task-run: report -> %s" % args.out)
    else:
        sys.stdout.write(txt)
    return 0


def mech(args):
    """Re-run the mechanical checks over a stored label, without touching the GPU."""
    recs = load_records(args.label)
    for r in recs:
        if r.get("error"):
            continue
        r["mechanical"] = mechanical(r["task"], r.get("content") or "", cxx=args.cxx)
    path = OUT_ROOT / args.label / "records.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for r in recs:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("ten-task-run: mechanical checks rewritten for %d records in %s" % (len(recs), path))
    return 0


# ------------------------------------------------------------------ cli

def main(argv):
    ap = argparse.ArgumentParser(description="#80 the ten-task runner")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="generate one arm")
    r.add_argument("--label", required=True)
    r.add_argument("--base-url", default="http://127.0.0.1:8099")
    r.add_argument("--engine", choices=["crow", "llama"])
    r.add_argument("--model", default="crow-nest")
    r.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    r.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    r.add_argument("--reasoning-effort", default=DEFAULT_EFFORT)
    r.add_argument("--timeout", type=float, default=3600.0)
    r.add_argument("--only")
    r.add_argument("--cxx", default="g++")
    r.set_defaults(fn=run)

    t = sub.add_parser("tokens", help="exact answer/thinking token split, no GPU")
    t.add_argument("--label", required=True)
    t.set_defaults(fn=tokens)

    m = sub.add_parser("mech", help="re-run the mechanical checks over a stored label")
    m.add_argument("--label", required=True)
    m.add_argument("--cxx", default="g++")
    m.set_defaults(fn=mech)

    b = sub.add_parser("blind", help="the shuffled answer file and the sealed id map")
    b.add_argument("--labels", required=True)
    b.add_argument("--out", default="answers.md")
    b.add_argument("--map-out", default="idmap.json")
    b.add_argument("--regrade-out", default="regrade.md")
    b.add_argument("--regrade-share", type=float, default=0.20)
    b.add_argument("--shuffle-seed", type=int, default=8080)
    b.add_argument("--salt")
    b.set_defaults(fn=blind)

    p = sub.add_parser("report", help="join the grades back to the arms")
    p.add_argument("--labels", required=True)
    p.add_argument("--grades")
    p.add_argument("--map-out", default="idmap.json")
    p.add_argument("--out")
    p.set_defaults(fn=report)

    args = ap.parse_args(argv[1:])
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
