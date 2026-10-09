#!/usr/bin/env python3
"""#174: regression gate R - proves that an engine change left Flash-Next and the 27B unchanged.

    python tools/regression_gate.py before  --session S --root <tree before the change>
    python tools/regression_gate.py after   --session S --root <tree after the change>
    python tools/regression_gate.py compare --session S

R1  `cargo test --release` in engine/ and converter/: exit code and libtest counts, before and after.
R2  byte-identical token ids and logits, greedy, 512 tokens, per model, through the `decode` bin:
    `parity` (all logits rows), `parity` teacher-forced (the decode path), `run` (512 greedy ids),
    and for the 27B (MTP on, the default) `mtpspec` (C2 lines, draft counters) and `mtpgolden`
    (the MTP head's logits). The forms are those of tools/gate-linux.sh and crow-nest #95 C2.
R3  decode ms/token, warm-up + 3 adjacent A/B pairs (before binary, after binary) interleaved with
    3 A/A pairs (before, before) in the same session; PASS iff |mean(N) - mean(B)| <= the A/A window
    (max - min of the six A/A runs). Measured by `after`, which holds both binaries.
R4  `compare` writes r4-protocol.md, Crow's real path, filled in by a human.

Red means: report the cause. Nothing in this script changes a default or a parameter to make a
check green, and nothing may be changed for that purpose (docs/regression-gate.md).

Results: JSON under runs/regression-gate/<session>/ (--out); raw dumps (logits, ~0.5 GB per item)
under the git-ignored decode_out/regression-gate/<session>/ (--raw). Stdlib only; Windows and Linux.
Every inherited CROW_* variable is stripped from the child environment and only the configured ones
are set, so a stray CROW_MTP=0 cannot turn the 27B's MTP head off unnoticed.
"""

import argparse
import datetime
import hashlib
import json
import math
import os
import platform
import re
import shutil
import struct
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
VOCAB = 248320
EXE = "decode.exe" if os.name == "nt" else "decode"
CRATES = ("engine", "converter")

# The env of record (tools/gate-linux.sh scope_run); path-valued entries are resolved against
# --data-root. CROW_MTP is NOT set: MTP on is the 27B's default (engine/src/gen.rs mtp_default).
FLASH_ENV = {
    "CROW_CNQ": "converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq",
    "CROW_HOTSETS": "decode_out/hotsets-M-longctx2100-n160.json",
    "CROW_GRAPH": "1",
    "CROW_MMA": "1",
}
DENSE_ENV = {
    "CROW_CNQ": "converter/Qwen3.8-27B-CNQ4.5.cnq",
    "CROW_GRAPH": "1",
    "CROW_MMA": "1",
}
PATH_ENV = ("CROW_CNQ", "CROW_HOTSETS")
IDS512 = "decode_out/real512-ids.json"
IDS8 = "decode_out/parity-ids.json"
TF_ENV = {"CROW_GRAPH": "0", "CROW_PARITY_PREFILL": "8"}

R2_COMMON = [
    {"name": "logits512", "mode": "parity", "ids": IDS512},
    {"name": "logits512-tf", "mode": "parity", "ids": IDS512, "env": TF_ENV},
    {"name": "greedy512", "mode": "run", "ids": IDS8, "gen": 512},
]
DEFAULT_CONFIG = {
    "vocab": VOCAB,
    "models": {
        "flash-next": {
            "env": FLASH_ENV,
            "r2": R2_COMMON,
            "r3": [{"name": "run512", "mode": "run", "ids": IDS8, "gen": 512}],
        },
        "qwen27b": {
            "env": DENSE_ENV,
            "r2": R2_COMMON + [
                {"name": "mtp512", "mode": "mtpspec", "ids": IDS8, "n": 512, "k": 3},
                {"name": "mtp-logits", "mode": "mtpgolden", "seq_from": "logits512"},
            ],
            "r3": [
                {"name": "run512", "mode": "run", "ids": IDS8, "gen": 512},
                {"name": "mtp512-step", "mode": "mtpspec", "ids": IDS8, "n": 512, "k": 3},
            ],
        },
    },
}

GREEN, RED, SKIPPED = "GREEN", "RED", "SKIPPED"


# ---------------------------------------------------------------- pure comparison logic

LIBTEST_RE = re.compile(
    r"^test result: (ok|FAILED)\. (\d+) passed; (\d+) failed; (\d+) ignored; (\d+) measured; (\d+) filtered out",
    re.M,
)


def parse_cargo_test(text, exit_code):
    """Sum libtest's per-binary summary lines (library/test/src/formatters/pretty.rs write_run_finish)."""
    passed = failed = ignored = 0
    binaries = 0
    failed_binaries = 0
    for m in LIBTEST_RE.finditer(text):
        binaries += 1
        passed += int(m.group(2))
        failed += int(m.group(3))
        ignored += int(m.group(4))
        failed_binaries += m.group(1) == "FAILED"
    return {
        "exit": exit_code,
        "passed": passed,
        "failed": failed,
        "ignored": ignored,
        "binaries": binaries,
        "failed_binaries": failed_binaries,
    }


def r1_side_causes(label, rec):
    causes = []
    if rec is None:
        return [f"{label}: not run"]
    if rec["binaries"] == 0:
        causes.append(f"{label}: no libtest summary line (build failed?), exit {rec['exit']}")
    if rec["exit"] != 0:
        causes.append(f"{label}: cargo test exit {rec['exit']}")
    if rec["failed"] > 0 or rec["failed_binaries"] > 0:
        causes.append(f"{label}: {rec['failed']} failed in {rec['failed_binaries']} test binaries")
    return causes


def compare_r1(before, after):
    """One crate: both sides clean, and the after side passes at least as many tests as before."""
    causes = r1_side_causes("before", before) + r1_side_causes("after", after)
    if before and after and after["passed"] < before["passed"]:
        causes.append(f"fewer tests pass after the change: {before['passed']} -> {after['passed']}")
    summary = ", ".join(
        f"{label} " + (f"{r['passed']} passed {r['failed']} failed" if r else "not run")
        for label, r in (("before", before), ("after", after))
    )
    return {"verdict": RED if causes else GREEN, "causes": causes, "summary": summary}


def compare_ids(a, b):
    if a is None or b is None:
        return {"identical": False, "cause": "ids missing on one side"}
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return {"identical": False, "index": i, "before": a[i], "after": b[i],
                    "cause": f"first differing id at index {i}: {a[i]} -> {b[i]}"}
    if len(a) != len(b):
        return {"identical": False, "index": n,
                "cause": f"id count differs: {len(a)} -> {len(b)}"}
    return {"identical": True, "count": n}


def first_diff_bytes(path_a, path_b, chunk=1 << 20):
    """Offset of the first differing byte, or None if the files are byte-identical."""
    with open(path_a, "rb") as fa, open(path_b, "rb") as fb:
        base = 0
        while True:
            ca, cb = fa.read(chunk), fb.read(chunk)
            if ca != cb:
                n = min(len(ca), len(cb))
                for i in range(n):
                    if ca[i] != cb[i]:
                        return base + i
                return base + n  # one file is a prefix of the other
            if not ca:
                return None
            base += len(ca)


def _f32_at(path, word_off):
    with open(path, "rb") as f:
        f.seek(word_off)
        w = f.read(4)
    return struct.unpack("<f", w)[0] if len(w) == 4 else None


def compare_logits(before, after, vocab):
    """Two logits dumps [rows][vocab] f32: sha + size first, then the first differing byte."""
    if not before or not after:
        return {"identical": False, "cause": "logits dump missing on one side"}
    if before["sha256"] == after["sha256"] and before["bytes"] == after["bytes"]:
        return {"identical": True, "sha256": before["sha256"], "bytes": before["bytes"]}
    res = {"identical": False, "sha256": [before["sha256"], after["sha256"]],
           "bytes": [before["bytes"], after["bytes"]]}
    pa, pb = before.get("path"), after.get("path")
    if pa and pb and os.path.isfile(pa) and os.path.isfile(pb):
        off = first_diff_bytes(pa, pb)
        if off is None:  # the sha disagreed with the bytes: the record is stale
            res["cause"] = "sha256 differs but the files on disk are identical (stale record)"
            return res
        word = off - off % 4
        row, col = divmod(word // 4, vocab)
        res.update({"offset": off, "row": row, "col": col,
                    "before_f32": _f32_at(pa, word), "after_f32": _f32_at(pb, word)})
        res["cause"] = (f"first differing byte at offset {off}: row {row}, column {col}, "
                        f"{res['before_f32']!r} -> {res['after_f32']!r}")
        if before["bytes"] != after["bytes"]:
            res["cause"] += f"; sizes {before['bytes']} -> {after['bytes']} B"
    else:
        res["cause"] = (f"sha256 {before['sha256'][:12]} -> {after['sha256'][:12]} "
                        f"({before['bytes']} -> {after['bytes']} B); raw dumps not on disk for the offset")
    return res


MTP_C2_RE = re.compile(r"^(mtpspec(?:-batched|-step)?): C2 [^\n]*?: (true|false)\b", re.M)
MTP_DRAFT_RE = re.compile(r"^(mtpspec(?:-batched|-step)?): (passes \d+, [^\n]*)$", re.M)
MTP_RATE_RE = re.compile(r", ([\d.]+) tok/s \(plain ([\d.]+)\)\s*$")
MTP_TRACE_RE = re.compile(r"^trace-plain: (\[[^\n]*\])", re.M)


def parse_mtpspec(text):
    """`decode mtpspec` stdout -> plain ids, the three C2 verdicts, draft counters, step rate."""
    out = {"c2": {}, "draft": {}, "ids": None, "step_tok_s": None, "plain_tok_s": None}
    m = MTP_TRACE_RE.search(text)
    if m:
        out["ids"] = json.loads(m.group(1))
    for m in MTP_C2_RE.finditer(text):
        out["c2"][m.group(1)] = m.group(2) == "true"
    for m in MTP_DRAFT_RE.finditer(text):
        line = m.group(2)
        r = MTP_RATE_RE.search(line)
        if r:
            if m.group(1) == "mtpspec-step":
                out["step_tok_s"] = float(r.group(1))
                out["plain_tok_s"] = float(r.group(2))
            line = line[: r.start()]  # the rate is timing, not a draft counter
        out["draft"][m.group(1)] = line
    return out


C2_PATHS = ("mtpspec", "mtpspec-batched", "mtpspec-step")


def compare_mtp(before, after):
    causes = []
    for label, rec in (("before", before), ("after", after)):
        for p in C2_PATHS:
            v = rec["c2"].get(p)
            if v is not True:
                causes.append(f"{label}: {p} C2 line {'missing' if v is None else 'false'}")
    ids = compare_ids(before.get("ids"), after.get("ids"))
    if not ids["identical"]:
        causes.append("trace-plain: " + ids["cause"])
    for p in C2_PATHS:
        a, b = before["draft"].get(p), after["draft"].get(p)
        if a != b:
            causes.append(f"{p} draft counters differ (the MTP head's drafts moved): {a!r} -> {b!r}")
    return causes


def compare_r2_item(before, after, vocab):
    if before is None or after is None:
        return {"verdict": RED, "causes": ["item missing on one side"]}
    causes = []
    for label, rec in (("before", before), ("after", after)):
        if rec.get("error"):
            causes.append(f"{label}: {rec['error']}")
    if causes:
        return {"verdict": RED, "causes": causes}
    if "logits" in before or "logits" in after:
        lg = compare_logits(before.get("logits"), after.get("logits"), vocab)
        if not lg["identical"]:
            causes.append("logits: " + lg["cause"])
    if "mtp" in before or "mtp" in after:
        if not before.get("mtp") or not after.get("mtp"):
            causes.append("mtpspec output missing on one side")
        else:
            causes += compare_mtp(before["mtp"], after["mtp"])
    elif "ids" in before or "ids" in after:
        ids = compare_ids(before.get("ids"), after.get("ids"))
        if not ids["identical"]:
            causes.append("ids: " + ids["cause"])
    return {"verdict": RED if causes else GREEN, "causes": causes}


def schedule(pairs):
    """Warm-up (before binary, discarded), then A/B and A/A pairs interleaved: AB1 AA1 AB2 AA2 ..."""
    seq = [("warmup", 0, "B")]
    for i in range(1, pairs + 1):
        seq += [("ab", i, "B"), ("ab", i, "N"), ("aa", i, "B"), ("aa", i, "B2")]
    return seq


def r3_verdict(runs):
    """runs: [{block, pair, arm, ms, ids_sha}] -> verdict against the A/A window of the same session."""
    causes = []
    bad = [r for r in runs if r.get("error")]
    for r in bad:
        causes.append(f"{r['block']}{r['pair']} {r['arm']}: {r['error']}")
    good = [r for r in runs if not r.get("error")]
    ab_b = [r["ms"] for r in good if r["block"] == "ab" and r["arm"] == "B"]
    ab_n = [r["ms"] for r in good if r["block"] == "ab" and r["arm"] == "N"]
    aa = [r["ms"] for r in good if r["block"] == "aa"]
    res = {"causes": causes}
    if not ab_b or not ab_n or len(aa) < 2:
        causes.append("not enough runs for a verdict")
        res["verdict"] = RED
        return res
    window = max(aa) - min(aa)
    mean_b = sum(ab_b) / len(ab_b)
    mean_n = sum(ab_n) / len(ab_n)
    delta = mean_n - mean_b
    pair_deltas = []
    for i in sorted({r["pair"] for r in good if r["block"] == "ab"}):
        b = [r["ms"] for r in good if r["block"] == "ab" and r["pair"] == i and r["arm"] == "B"]
        n = [r["ms"] for r in good if r["block"] == "ab" and r["pair"] == i and r["arm"] == "N"]
        if b and n:
            pair_deltas.append(n[0] - b[0])
    res.update({"aa_window_ms": window, "mean_b_ms": mean_b, "mean_n_ms": mean_n,
                "delta_ms": delta, "pair_deltas_ms": pair_deltas,
                "delta_in_windows": (abs(delta) / window) if window > 0 else (math.inf if delta else 0.0)})
    if abs(delta) > window:
        side = "slower" if delta > 0 else "faster"
        causes.append(f"after build {side} by {abs(delta):.4f} ms/token, outside the A/A window "
                      f"{window:.4f} ms (B {mean_b:.4f}, N {mean_n:.4f})")
    shas = {r.get("ids_sha") for r in good}
    if len(shas) > 1:
        causes.append(f"generated ids differ between timed runs ({len(shas)} distinct traces)")
    res["verdict"] = RED if causes else GREEN
    return res


def r4_verdict(text):
    m = re.search(r"^R4 verdict:\s*(PENDING|PASS|FAIL)\b", text or "", re.M)
    return m.group(1) if m else "PENDING"


def overall(parts):
    """parts: list of verdict strings for R1-R3 plus the R4 status -> (overall, exit code)."""
    r123, r4 = parts[:-1], parts[-1]
    if RED in r123 or r4 == "FAIL":
        return RED, 1
    if SKIPPED in r123:
        return "INCOMPLETE", 3
    if r4 != "PASS":
        return "R1-R3 GREEN, R4 PENDING", 3
    return "ALL GREEN", 0


def compare_sessions(before, after, vocab, r4_text=None):
    res = {"r1": {}, "r2": {}, "r3": {}}
    verdicts = []
    # R1
    r1b, r1a = before.get("r1"), after.get("r1")
    if r1b is None or r1a is None:
        res["r1"] = {"verdict": SKIPPED, "causes": ["R1 not run on " + ("before" if r1b is None else "after")]}
        verdicts.append(SKIPPED)
    else:
        for crate in CRATES:
            res["r1"][crate] = compare_r1(r1b.get(crate), r1a.get(crate))
            verdicts.append(res["r1"][crate]["verdict"])
    # R2
    r2b, r2a = before.get("r2"), after.get("r2")
    if r2b is None or r2a is None:
        res["r2"] = {"verdict": SKIPPED, "causes": ["R2 not run on one side"]}
        verdicts.append(SKIPPED)
    else:
        for model in sorted(set(r2b) | set(r2a)):
            items_b, items_a = r2b.get(model, {}), r2a.get(model, {})
            res["r2"][model] = {}
            for item in list(dict.fromkeys(list(items_b) + list(items_a))):
                v = compare_r2_item(items_b.get(item), items_a.get(item), vocab)
                res["r2"][model][item] = v
                verdicts.append(v["verdict"])
    # R3 (measured by `after`)
    r3 = after.get("r3")
    if r3 is None:
        res["r3"] = {"verdict": SKIPPED, "causes": ["R3 not run"]}
        verdicts.append(SKIPPED)
    else:
        for model, metrics in r3.items():
            res["r3"][model] = {}
            for metric, rec in metrics.items():
                v = r3_verdict(rec.get("runs", []))
                res["r3"][model][metric] = v
                verdicts.append(v["verdict"])
    res["r4"] = {"verdict": r4_verdict(r4_text)}
    res["overall"], res["exit"] = overall(verdicts + [res["r4"]["verdict"]])
    return res


def report_lines(res):
    lines = []

    def line(v, name, detail):
        lines.append(f"{v:<8} {name:<34} {detail}")

    r1 = res["r1"]
    if "verdict" in r1:
        line(r1["verdict"], "R1", "; ".join(r1["causes"]))
    else:
        for crate, v in r1.items():
            line(v["verdict"], f"R1 cargo test {crate}", "; ".join(v["causes"]) or v["summary"])
    r2 = res["r2"]
    if "verdict" in r2:
        line(r2["verdict"], "R2", "; ".join(r2["causes"]))
    else:
        for model, items in r2.items():
            for item, v in items.items():
                line(v["verdict"], f"R2 {model}/{item}", "; ".join(v["causes"]) or "byte-identical")
    r3 = res["r3"]
    if "verdict" in r3:
        line(r3["verdict"], "R3", "; ".join(r3["causes"]))
    else:
        for model, metrics in r3.items():
            for metric, v in metrics.items():
                detail = "; ".join(v["causes"])
                if not detail and "delta_ms" in v:
                    detail = (f"delta {v['delta_ms']:+.4f} ms within A/A window {v['aa_window_ms']:.4f} ms "
                              f"(B {v['mean_b_ms']:.4f}, N {v['mean_n_ms']:.4f})")
                line(v["verdict"], f"R3 {model}/{metric}", detail)
    line(res["r4"]["verdict"], "R4 Crow's real path", "filled in by a human: r4-protocol.md")
    lines.append(f"== regression gate: {res['overall']}")
    return lines


# ---------------------------------------------------------------- R4 template

def r4_template(session, before, after, models, crow_repo):
    def git(rec):
        g = (rec or {}).get("git", {})
        head = g.get("head") or "?"  # None when --root is no git checkout (git_info)
        return f"`{head[:12]}`{' (dirty)' if g.get('dirty') else ''}"

    out = [
        f"# R4: Crow's real path - session `{session}`",
        "",
        f"before {git(before)}, after {git(after)}. Filled in by a human (crow-nest #174, docs/regression-gate.md).",
        "Run the AFTER build. One engine at a time; stop serve before the next model.",
        "",
        "R4 verdict: PENDING",
        "",
        "Set the line above to `R4 verdict: PASS` or `R4 verdict: FAIL` when done; `compare` reads it.",
        "",
    ]
    for name in models:
        out += [
            f"## {name}",
            "",
            "1. Start serve with this model's container (`CROW_CNQ`, see docs/getting-started.md), port 8099.",
            "2. `python cli\\crow_gui.py --base-url http://127.0.0.1:8099/v1` from the Crow checkout.",
            "3. At least 3 turns; judge each answer as you would in daily use (tool calls, thinking, stop).",
            "",
            "| turn | prompt (short) | answer usable (y/n) | tool calls ok (y/n/-) | notes |",
            "|---|---|---|---|---|",
            "| 1 | | | | |",
            "| 2 | | | | |",
            "| 3 | | | | |",
            "",
        ]
    out += [
        "## Crow manifests unchanged",
        "",
        f"`git -C {crow_repo} diff --stat -- manifests/` must print nothing. Output:",
        "",
        "```",
        "",
        "```",
        "",
    ]
    return "\n".join(out)


# ---------------------------------------------------------------- running things

def die(msg):
    """A setup error (missing input, failed build): exit 2, distinct from a RED verdict (1)."""
    print(msg, file=sys.stderr)
    sys.exit(2)


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds")


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def file_rec(path):
    return {"path": str(path), "sha256": sha256_file(path), "bytes": os.path.getsize(path)}


def ids_sha(ids):
    return hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:16] if ids is not None else None


def clean_env(extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith("CROW_")}
    env.update(extra)
    return env


def model_env(model_cfg, item, data_root):
    env = dict(model_cfg.get("env", {}))
    env.update(item.get("env", {}))
    for k in PATH_ENV:
        if k in env and not os.path.isabs(env[k]):
            env[k] = str((data_root / env[k]).resolve())
    return env


def git_info(root):
    def g(*a):
        p = subprocess.run(["git", "-C", str(root), *a], capture_output=True, text=True)
        return p.stdout.strip() if p.returncode == 0 else None
    return {"head": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def run_logged(cmd, cwd, env, log, timeout=None):
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    try:
        p = subprocess.run(cmd, cwd=str(cwd), env=env, capture_output=True, text=True,
                           encoding="utf-8", errors="replace", timeout=timeout)
        rc, out = p.returncode, p.stdout + ("\n[stderr]\n" + p.stderr if p.stderr else "")
    except subprocess.TimeoutExpired as e:
        rc, out = -9, f"timeout after {timeout} s\n{e.stdout or ''}"
    except OSError as e:
        rc, out = -1, f"could not start {cmd[0]}: {e}"
    log.write_text(f"$ {' '.join(map(str, cmd))}\n# cwd {cwd}\n{out}", encoding="utf-8")
    return rc, out, time.time() - t0


def gpu_used_mib():
    try:
        p = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        return int(p.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.TimeoutExpired):
        return None


def engines_alive():
    names = {"decode", "serve", "parity", "llama-server"}
    try:
        if os.name == "nt":
            p = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, timeout=30)
            procs = [l.split(",")[0].strip('"').lower().removesuffix(".exe") for l in p.stdout.splitlines() if l]
        else:
            p = subprocess.run(["ps", "-eo", "comm="], capture_output=True, text=True, timeout=30)
            procs = [l.strip() for l in p.stdout.splitlines()]
    except (OSError, subprocess.TimeoutExpired):
        return []
    return sorted({x for x in procs if x in names})


def free_ram_gib():
    """Windows only (the 50.5 GiB chain gate of docs/architecture.md 0.5 rule 1); None elsewhere."""
    if os.name != "nt":
        return None
    import ctypes

    class MS(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]
    ms = MS()
    ms.dwLength = ctypes.sizeof(MS)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
    return ms.ullAvailPhys / 2**30


def precheck(args):
    """One engine at a time, never on a busy GPU (tools/gate-linux.sh precheck); None = go."""
    alive = engines_alive()
    if alive:
        return f"precheck refused: an engine is alive ({', '.join(alive)})"
    used = gpu_used_mib()
    if used is not None and used >= args.gpu_used_max_mib:
        return f"precheck refused: GPU holds {used} MiB (>= {args.gpu_used_max_mib})"
    if args.ram_gate_gib > 0:
        t_end = time.time() + args.ram_wait_s
        while True:
            free = free_ram_gib()
            if free is None or free > args.ram_gate_gib:
                break
            if time.time() > t_end:
                return f"precheck refused: {free:.1f} GiB free host RAM <= {args.ram_gate_gib} after {args.ram_wait_s} s"
            time.sleep(5)
    return None


def decode_once(binary, item, model_cfg, data_root, work, log, args):
    """One fresh `decode` process for one item -> record (ids / logits / mtp / ms) or {"error"}."""
    env = clean_env(model_env(model_cfg, item, data_root))
    ids_path = str((data_root / item["ids"]).resolve()) if "ids" in item else None
    mode = item["mode"]
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    run_json = data_root / "decode_out" / "run.json"
    if mode == "parity":
        cmd = [binary, "parity", ids_path, str(work)]
    elif mode == "run":
        if run_json.exists():
            run_json.unlink()  # `decode run` writes here whatever its arguments say; never read a stale one
        cmd = [binary, "run", ids_path, str(item["gen"])]
    elif mode == "mtpspec":
        cmd = [binary, "mtpspec", ids_path, str(item["n"]), str(item["k"])]
    elif mode == "mtpgolden":
        src = work.parent / item["seq_from"] / "gen-sequence.json"
        if not src.exists():
            return {"error": f"mtpgolden needs {item['seq_from']}'s gen-sequence.json, not found"}
        shutil.copy2(src, work / "gen-sequence.json")
        cmd = [binary, "mtpgolden", str(work)]
    else:
        return {"error": f"unknown mode {mode}"}
    why = precheck(args)
    if why:
        return {"error": why}
    rc, out, wall = run_logged([str(c) for c in cmd], data_root, env, log, args.timeout_s)
    rec = {"cmd": [str(c) for c in cmd[1:]], "env": {k: v for k, v in env.items() if k.startswith("CROW_")},
           "exit": rc, "wall_s": round(wall, 1), "log": str(log)}
    if rc != 0:
        rec["error"] = f"decode exit {rc}, see {log}"
        return rec
    try:
        if mode == "parity":
            seq = json.loads((work / "gen-sequence.json").read_text(encoding="utf-8"))
            rec["logits"] = file_rec(work / "gpu-logits.f32")
            rec["ids"] = seq["all_ids"]
            rec["rows"], rec["nan"] = seq["rows"], seq["nan"]
        elif mode == "run":
            rj = json.loads(run_json.read_text(encoding="utf-8"))
            shutil.copy2(run_json, work / "run.json")
            rec["ids"], rec["ms"] = rj["trace"], rj["mean_ms"]
        elif mode == "mtpspec":
            m = parse_mtpspec(out)
            rec["mtp"] = m
            rec["ids"] = m["ids"]
            if m["step_tok_s"]:
                rec["ms"] = 1000.0 / m["step_tok_s"]
        elif mode == "mtpgolden":
            rec["logits"] = file_rec(work / "mtp-gpu-logits.f32")
    except (OSError, KeyError, ValueError) as e:
        rec["error"] = f"output missing or unreadable: {e}"
    return rec


def selected_models(cfg, args):
    names = list(cfg["models"])
    if args.models:
        want = [m.strip() for m in args.models.split(",") if m.strip()]
        unknown = [m for m in want if m not in cfg["models"]]
        if unknown:
            die(f"regression_gate: unknown model(s) {unknown}; known: {names}")
        names = want
    return names


def do_r1(root, raw, record):
    record["r1"] = {}
    for crate in CRATES:
        log = raw / f"r1-cargo-test-{crate}.log"
        rc, out, wall = run_logged(["cargo", "test", "--release"], root / crate, clean_env({}), log)
        rec = parse_cargo_test(out, rc)
        rec.update({"wall_s": round(wall, 1), "log": str(log)})
        record["r1"][crate] = rec
        print(f"  R1 {crate}: exit {rc}, {rec['passed']} passed, {rec['failed']} failed ({rec['binaries']} binaries)")


def build_and_snapshot(root, raw, record, no_build):
    if not no_build:
        log = raw / "build-decode.log"
        rc, _, _ = run_logged(["cargo", "build", "--release", "--bin", "decode"], root / "engine", clean_env({}), log)
        if rc != 0:
            die(f"regression_gate: cargo build failed (exit {rc}), see {log}")
    src = root / "engine" / "target" / "release" / EXE
    if not src.exists():
        die(f"regression_gate: no {src}; build it or drop --no-build")
    dst = raw / "bin" / EXE
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    record["binary"] = {"source": str(src), **file_rec(dst)}
    return dst


def do_r2(binary, cfg, models, data_root, raw, record, args):
    record["r2"] = {}
    for name in models:
        mc = cfg["models"][name]
        record["r2"][name] = {}
        for item in mc["r2"]:
            work = raw / "r2" / name / item["name"]
            log = raw / "r2" / name / f"{item['name']}.log"
            rec = decode_once(binary, item, mc, data_root, work, log, args)
            record["r2"][name][item["name"]] = rec
            print(f"  R2 {name}/{item['name']}: {rec.get('error') or 'done'}")


def do_r3(bin_b, bin_n, cfg, models, data_root, raw, record, args):
    record["r3"] = {}
    for name in models:
        mc = cfg["models"][name]
        record["r3"][name] = {}
        for metric in mc.get("r3", []):
            runs = []
            for block, pair, arm in schedule(args.pairs):
                binary = bin_n if arm == "N" else bin_b
                tag = f"{block}{pair}-{arm}"
                work = raw / "r3" / name / metric["name"] / tag
                log = raw / "r3" / name / metric["name"] / f"{tag}.log"
                rec = decode_once(binary, metric, mc, data_root, work, log, args)
                run = {"block": block, "pair": pair, "arm": arm, "binary": str(binary),
                       "ms": rec.get("ms"), "ids_sha": ids_sha(rec.get("ids"))}
                if rec.get("error") or rec.get("ms") is None:
                    run["error"] = rec.get("error") or "no timing in the output"
                runs.append(run)
                shown = run.get("error") or f"{run['ms']:.4f} ms/token"
                print(f"  R3 {name}/{metric['name']} {tag}: {shown}")
            record["r3"][name][metric["name"]] = {"form": f"warm-up + {args.pairs} A/B + {args.pairs} A/A pairs",
                                                  "runs": [r for r in runs if r["block"] != "warmup"],
                                                  "warmup": [r for r in runs if r["block"] == "warmup"]}


def load_config(path):
    if not path:
        return DEFAULT_CONFIG
    return json.loads(Path(path).read_text(encoding="utf-8"))


def side_cmd(args, side):
    cfg = load_config(args.config)
    root = Path(args.root).resolve()
    data_root = Path(args.data_root).resolve()
    out = Path(args.out).resolve() / args.session
    raw = Path(args.raw).resolve() / args.session / side
    out.mkdir(parents=True, exist_ok=True)
    raw.mkdir(parents=True, exist_ok=True)
    skip = {s.strip().lower() for s in (args.skip or "").split(",") if s.strip()}
    models = selected_models(cfg, args)
    record = {"schema": 1, "side": side, "session": args.session, "started": now(),
              "root": str(root), "data_root": str(data_root), "git": git_info(root),
              "host": {"platform": platform.platform(), "python": sys.version.split()[0]},
              "models": models, "skipped": sorted(skip)}
    print(f"== regression gate {side}: {root} at {record['git']['head']} -> {out}")
    if "r1" not in skip:
        do_r1(root, raw, record)
    binary = build_and_snapshot(root, raw, record, args.no_build)
    if "r2" not in skip:
        do_r2(binary, cfg, models, data_root, raw, record, args)
    if side == "after" and "r3" not in skip:
        before_path = Path(args.out).resolve() / (args.before_session or args.session) / "before.json"
        if not before_path.exists():
            die(f"regression_gate: R3 needs {before_path} (run `before` first or pass --skip r3)")
        bin_b = Path(json.loads(before_path.read_text(encoding="utf-8"))["binary"]["path"])
        if not bin_b.exists():
            die(f"regression_gate: the before binary {bin_b} is gone; rerun `before`")
        record["r3_binaries"] = {"B": file_rec(bin_b), "N": record["binary"]}
        do_r3(bin_b, binary, cfg, models, data_root, raw, record, args)
    record["finished"] = now()
    (out / f"{side}.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(f"== wrote {out / f'{side}.json'}")
    return 0


def compare_cmd(args):
    cfg = load_config(args.config)
    out = Path(args.out).resolve() / args.session
    before_path = Path(args.out).resolve() / (args.before_session or args.session) / "before.json"
    after_path = out / "after.json"
    for p in (before_path, after_path):
        if not p.exists():
            die(f"regression_gate: {p} not found")
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = json.loads(after_path.read_text(encoding="utf-8"))
    r4_path = out / "r4-protocol.md"
    if not r4_path.exists():  # a filled protocol is never overwritten
        r4_path.write_text(r4_template(args.session, before, after, after.get("models", []), args.crow_repo),
                           encoding="utf-8")
    res = compare_sessions(before, after, cfg.get("vocab", VOCAB), r4_path.read_text(encoding="utf-8"))
    res.update({"session": args.session, "before": str(before_path), "after": str(after_path),
                "compared": now(), "r4_protocol": str(r4_path)})
    (out / "compare.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    for line in report_lines(res):
        print(line)
    print("Red means: report the cause; never change a default or a parameter to make it green.")
    return res["exit"]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("before", "after", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--session", default=datetime.date.today().isoformat(),
                       help="session name, the folder under --out and --raw (default: today)")
        p.add_argument("--out", default=str(REPO / "runs" / "regression-gate"), help="JSON results root")
        p.add_argument("--config", help="JSON config replacing the built-in models and items")
        p.add_argument("--before-session", help="take before.json (and the before binary) from this session")
        if name == "compare":
            p.add_argument("--crow-repo", default=r"C:\Users\robin\dev\Crow" if os.name == "nt" else "~/dev/Crow",
                           help="Crow checkout named in the R4 protocol")
            continue
        p.add_argument("--root", default=str(REPO), help="the crow-nest tree to test and build")
        p.add_argument("--data-root", default=str(REPO),
                       help="where converter/*.cnq and decode_out/*.json resolve; the cwd of decode")
        p.add_argument("--raw", default=str(REPO / "decode_out" / "regression-gate"), help="raw dump root")
        p.add_argument("--models", help="comma list (default: all configured)")
        p.add_argument("--skip", help="comma list of r1, r2, r3")
        p.add_argument("--no-build", action="store_true", help="use the tree's existing release decode bin")
        p.add_argument("--pairs", type=int, default=3, help="R3 A/B and A/A pairs (default 3)")
        p.add_argument("--timeout-s", type=int, default=3600)
        p.add_argument("--gpu-used-max-mib", type=int, default=2000,
                       help="refuse an engine start while the GPU holds this much (gate-linux.sh: 2000)")
        p.add_argument("--ram-gate-gib", type=float, default=50.5 if os.name == "nt" else 0.0,
                       help="Windows chain gate, free host RAM before an engine start (architecture.md 0.5)")
        p.add_argument("--ram-wait-s", type=int, default=600)
    args = ap.parse_args(argv)
    if args.cmd == "compare":
        return compare_cmd(args)
    return side_cmd(args, args.cmd)


if __name__ == "__main__":
    sys.exit(main())
