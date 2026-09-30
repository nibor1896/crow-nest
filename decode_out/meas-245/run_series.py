#!/usr/bin/env python3
"""Crow #245: reasoning budget 1024 vs 2048 on the 27B, Windows -- the runner PREREG.md names.

  python decode_out/meas-245/run_series.py            # the series (needs session-0922.json)
  python decode_out/meas-245/run_series.py --report   # the table from the kept JSONs
  python decode_out/meas-245/run_series.py --selftest SESSION --points K [--rounds-per-arm 1]
                                                       # plumbing check, own out dir, no sha pin

One probe invocation per (point, seed, arm), arms alternating ABBA by seed, a warm-up per point
(arm A body, max_tokens 1, not counted) whose prompt_tokens decide whether the point fits.
Everything the series produces stays under OUT: one JSON + one stderr log per round, plan.json,
engine-slice.log (the UTC lines of the series from Crow's engine.log).
"""
import argparse
import datetime
import hashlib
import json
import math
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
PROBE = os.path.join(REPO, "tools", "corruption-replay-probe.py")
HEAD = os.path.join(REPO, "tools", "corpora", "91-replay-diorama-0922-head.txt")
SESSION = os.path.join(HERE, "session-0922.json")
SESSION_SHA = "559bb1ed8e17ec4beecfc9ed2aba33f2538e1b53df446921160019441b3729a8"
CROW_CORE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Crow", "cli", "crow_core.py")
ENGINE_LOG = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Crow", "logs", "engine.log")
POINTS = [26, 69, 71, 109, 139]
ARMS = {"A1024": {}, "B2048": {"reasoning_budget_tokens": 2048}}
N_CTX, MAX_TOKENS = 65536, 16384
SEEDS = list(range(8)) + ["greedy"]


def now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%fZ")[:-4] + "Z"


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def gpu_processes():
    """Every running serve / llama-server / sd-server (PREREG: exactly one serve)."""
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "Get-Process -Name serve,llama-server,sd-server -ErrorAction SilentlyContinue"
                          " | ForEach-Object { $_.Name + ' ' + $_.Id }"],
                         capture_output=True, text=True).stdout.split()
    return [" ".join(out[i:i + 2]) for i in range(0, len(out), 2)]


def probe(session, k, seed, extra, label, out_dir, rounds_json=True):
    sampling = dict(extra)
    seed0 = 0
    if seed == "greedy":
        sampling["temperature"] = 0
    else:
        seed0 = seed
    js = os.path.join(out_dir, label + ".json")
    cmd = [sys.executable, PROBE, "--session", session, "--head-file", HEAD, "--at", str(k),
           "--rounds", "1", "--seed0", str(seed0), "--served-name", "auto",
           "--crow-core", CROW_CORE, "--home", "/home/nibor1896", "--label", label,
           "--sampling", json.dumps(sampling), "--json", js]
    with open(os.path.join(out_dir, label + ".err"), "w", encoding="utf-8") as err:
        rc = subprocess.run(cmd, stdout=err, stderr=err).returncode
    if rc != 0 or not os.path.exists(js):
        return {"label": label, "rc": rc, "error": "probe exit %d (see %s.err)" % (rc, label)}
    with open(js, encoding="utf-8") as fh:
        return json.load(fh)


def engine_slice(t0, t1, out_dir):
    kept = 0
    with open(ENGINE_LOG, encoding="utf-8", errors="replace") as fh, \
            open(os.path.join(out_dir, "engine-slice.log"), "w", encoding="utf-8") as out:
        for line in fh:
            if t0[:19] <= line[:19] <= t1[:19]:
                out.write(line)
                kept += 1
    return kept


def run(args):
    out_dir = args.out
    os.makedirs(out_dir, exist_ok=True)
    session = args.session
    sha = sha256(session)
    if not args.selftest and sha != SESSION_SHA:
        sys.exit("%s has sha256 %s, PREREG pins %s" % (session, sha[:16], SESSION_SHA[:16]))
    procs = gpu_processes()
    if len([p for p in procs if p.startswith("serve ")]) != 1 or len(procs) != 1:
        sys.exit("PREREG: exactly one serve.exe and nothing else - found %s" % procs)
    seeds = SEEDS[:args.rounds_per_arm] if args.rounds_per_arm else SEEDS
    plan = {"started_utc": now(), "session": session, "session_sha256": sha,
            "crow_core_sha256": sha256(CROW_CORE), "probe_sha256": sha256(PROBE),
            "processes_before": procs, "points": {}, "arms": ARMS, "seeds": seeds}
    print("series start %s, processes %s" % (plan["started_utc"], procs), flush=True)
    for k in args.points:
        warm = probe(session, k, 0, {"max_tokens": 1}, "warm-K%d" % k, out_dir)
        if "error" in warm:
            plan["points"][k] = {"in": False, "why": warm["error"]}
            print("K=%d OUT: %s" % (k, warm["error"]), flush=True)
            continue
        rd = (warm.get("rounds_detail") or [{}])[0]
        ptok = rd.get("prompt_tokens")
        if ptok is None or ptok + MAX_TOKENS > N_CTX:
            plan["points"][k] = {"in": False, "prompt_tokens": ptok,
                                 "why": "prompt + %d > n_ctx %d" % (MAX_TOKENS, N_CTX)}
            print("K=%d OUT: prompt %s tok does not fit" % (k, ptok), flush=True)
            continue
        plan["points"][k] = {"in": True, "prompt_tokens": ptok}
        print("K=%d IN: prompt %d tok" % (k, ptok), flush=True)
        for i, seed in enumerate(seeds):
            order = ["A1024", "B2048"] if (seed == "greedy" or seed % 2 == 0) else ["B2048", "A1024"]
            for arm in order:
                label = "%s-K%d-s%s" % (arm, k, seed)
                r = probe(session, k, seed, ARMS[arm], label, out_dir)
                d = (r.get("rounds_detail") or [{}])[0] if "error" not in r else {}
                print("  %-20s calls %s finish %s reasoning %s closed %s %.1f s%s" % (
                    label, d.get("n_calls"), d.get("finish"), d.get("reasoning_chunks"),
                    d.get("budget_closed"), d.get("seconds") or 0,
                    ("  ERROR " + (r.get("error") or d.get("error") or "")) if ("error" in r or "error" in d) else ""),
                    flush=True)
    plan["finished_utc"] = now()
    plan["processes_after"] = gpu_processes()
    plan["engine_log_lines"] = engine_slice(plan["started_utc"], plan["finished_utc"], out_dir)
    with open(os.path.join(out_dir, "plan.json"), "w", encoding="utf-8") as fh:
        json.dump(plan, fh, indent=1)
    print("series end %s, processes %s" % (plan["finished_utc"], plan["processes_after"]), flush=True)
    report(out_dir)


def wilson(x, n, z=1.96):
    if not n:
        return (None, None)
    p = x / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (round(max(0.0, c - h), 3), round(min(1.0, c + h), 3))


def median(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else (xs[m - 1] + xs[m]) / 2


def report(out_dir):
    rows = {}
    for name in sorted(os.listdir(out_dir)):
        if not name.endswith(".json") or name.startswith(("warm-", "plan")):
            continue
        arm, kk, _ = name[:-5].split("-", 2)
        with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
            doc = json.load(fh)
        for d in doc.get("rounds_detail") or []:
            rows.setdefault((arm, int(kk[1:])), []).append(d)
    lines = ["| arm | K | rounds ok | no call (95 % CI) | corrupt calls | schema-wrong | budget closed "
             "| reasoning chunks median [min-max] | s/round median [min-max] |",
             "|---|---|---|---|---|---|---|---|---|"]
    tot = {}
    for (arm, k) in sorted(rows, key=lambda x: (x[1], x[0])):
        ds = [d for d in rows[(arm, k)] if "error" not in d]
        nc = sum(1 for d in ds if not d["n_calls"])
        rc = [d.get("reasoning_chunks") for d in ds if d.get("reasoning_chunks") is not None]
        sec = [d["seconds"] for d in ds]
        t = tot.setdefault(arm, {"n": 0, "nc": 0, "corrupt": 0, "schema": 0, "closed": 0, "sec": [],
                                 "failed": 0})
        t["n"] += len(ds)
        t["failed"] += len(rows[(arm, k)]) - len(ds)
        t["nc"] += nc
        t["corrupt"] += sum(d["calls_with_error"] for d in ds)
        t["schema"] += sum(d["calls_with_schema_error"] for d in ds)
        t["closed"] += sum(1 for d in ds if d.get("budget_closed"))
        t["sec"] += sec
        lines.append("| %s | %d | %d/%d | %d %s | %d | %d | %d | %s [%s-%s] | %s [%s-%s] |" % (
            arm, k, len(ds), len(rows[(arm, k)]), nc, wilson(nc, len(ds)),
            sum(d["calls_with_error"] for d in ds), sum(d["calls_with_schema_error"] for d in ds),
            sum(1 for d in ds if d.get("budget_closed")),
            median(rc), min(rc, default=None), max(rc, default=None),
            median(sec), min(sec, default=None), max(sec, default=None)))
    lines.append("")
    lines.append("| arm | rounds | failed | no call (95 % CI) | corrupt | schema-wrong | budget closed | s/round median |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for arm, t in sorted(tot.items()):
        lines.append("| %s | %d | %d | %d %s | %d | %d | %d | %s |" % (
            arm, t["n"], t["failed"], t["nc"], wilson(t["nc"], t["n"]), t["corrupt"], t["schema"],
            t["closed"], median(t["sec"])))
    a, b = tot.get("A1024"), tot.get("B2048")
    if a and b and median(a["sec"]):
        rule = {"a_no_call": b["nc"] <= a["nc"], "b_corrupt": b["corrupt"] <= a["corrupt"],
                "c_wall_clock": median(b["sec"]) <= 1.5 * median(a["sec"])}
        lines.append("")
        lines.append("PREREG decision rule: %s -> %s" % (
            rule, "2048 replaces 1024" if all(rule.values()) else "1024 stays"))
    text = "\n".join(lines) + "\n"
    with open(os.path.join(out_dir, "TABLE.md"), "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true")
    ap.add_argument("--selftest", default=None, metavar="SESSION")
    ap.add_argument("--points", type=int, nargs="*", default=None)
    ap.add_argument("--rounds-per-arm", type=int, default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.report:
        return report(args.out or HERE)
    if args.selftest:
        args.session, args.out = args.selftest, args.out or os.path.join(HERE, "selftest")
        args.points = args.points or []
    else:
        args.session, args.out = SESSION, args.out or HERE
        args.points = args.points or POINTS
        if args.rounds_per_arm:
            sys.exit("--rounds-per-arm is for --selftest only (PREREG: 8 seeds + greedy)")
    run(args)


if __name__ == "__main__":
    sys.exit(main())
