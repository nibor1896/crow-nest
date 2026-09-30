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
# PREREG Amendment 1: the Windows session of 2026-09-15, its own head, Windows home
SESSION = os.path.join(HERE, "session-0915.json")
SESSION_SHA = "6ee99879f3982c78e5a5e9118ef50e57834d89201d6393bec15c76fc0f46a834"
HEAD = None
HOME = "C:/Users/robin"
CROW_CORE = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Crow", "cli", "crow_core.py")
ENGINE_LOG = os.path.join(os.environ.get("LOCALAPPDATA", ""), "Crow", "logs", "engine.log")
ARMS = {"A1024": {}, "B2048": {"reasoning_budget_tokens": 2048}}
N_CTX, MAX_TOKENS = 65536, 16384
FIT = N_CTX - MAX_TOKENS
SEEDS = list(range(8)) + ["greedy"]
SCREEN_SEED, SCREEN_MIN_K, MAX_POINTS = 100, 5, 5
# PREREG Amendment 2: the closing sentence at 1024 -- sentence B of Crow #245
SENT_B = "\n\nThat is enough analysis. I will now act on it.\n"
SERIES = {
    # arm names carry no "-": report() splits file names on it
    "budget": {"arms": ARMS, "seeds": SEEDS, "points": None, "out": HERE, "decide": "decide_budget"},
    "sentence": {"arms": {"SA1024": {}, "SB1024": {"reasoning_budget_message": SENT_B}},
                 "seeds": list(range(32)) + ["greedy"], "points": [10, 45],
                 "out": os.path.join(HERE, "sentence"), "decide": "decide_sentence"},
}


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
    cmd = [sys.executable, PROBE, "--session", session, "--at", str(k),
           "--rounds", "1", "--seed0", str(seed0), "--served-name", "auto",
           "--crow-core", CROW_CORE, "--home", HOME, "--label", label,
           "--sampling", json.dumps(sampling), "--json", js]
    if HEAD:
        cmd[4:4] = ["--head-file", HEAD]
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


def candidates(session, min_k):
    """PREREG Amendment 1, screening step 1: K with a user/tool turn before and an assistant
    turn WITH tool calls at K, ascending."""
    with open(session, encoding="utf-8") as fh:
        m = json.load(fh)["messages"]
    return [k for k in range(max(1, min_k), len(m))
            if m[k - 1]["role"] in ("user", "tool") and m[k]["role"] == "assistant"
            and m[k].get("tool_calls")]


def spread(ks, n=MAX_POINTS):
    """PREREG Amendment 1, step 4: n points evenly over the closers, indices round(i(N-1)/(n-1))."""
    if len(ks) <= n:
        return list(ks)
    return [ks[round(i * (len(ks) - 1) / (n - 1))] for i in range(n)]


def screen(session, out_dir, limit=None):
    """Steps 2-4: warm-up per candidate for its prompt size (the first that does not fit ends
    the list), one arm-A round at SCREEN_SEED, the points = the rounds the budget closed."""
    rows, closers = [], []
    for k in candidates(session, SCREEN_MIN_K)[:limit]:
        warm = probe(session, k, 0, {"max_tokens": 1}, "warm-K%d" % k, out_dir)
        rd = (warm.get("rounds_detail") or [{}])[0] if "error" not in warm else {}
        ptok = rd.get("prompt_tokens")
        if ptok is None or ptok > FIT:
            rows.append({"k": k, "prompt_tokens": ptok, "end": True,
                         "why": warm.get("error") or rd.get("error") or "prompt > %d" % FIT})
            print("screen K=%d: prompt %s tok - list ends" % (k, ptok), flush=True)
            break
        r = probe(session, k, SCREEN_SEED, ARMS["A1024"], "scr-A1024-K%d-s%d" % (k, SCREEN_SEED), out_dir)
        d = (r.get("rounds_detail") or [{}])[0] if "error" not in r else {"error": r["error"]}
        row = {"k": k, "prompt_tokens": ptok, "n_calls": d.get("n_calls"), "finish": d.get("finish"),
               "reasoning_chunks": d.get("reasoning_chunks"), "budget_closed": d.get("budget_closed"),
               "seconds": d.get("seconds"), "error": d.get("error")}
        rows.append(row)
        if d.get("budget_closed"):
            closers.append(k)
        print("screen K=%d: prompt %d, calls %s, reasoning %s, closed %s, %.1f s" % (
            k, ptok, row["n_calls"], row["reasoning_chunks"], row["budget_closed"],
            row["seconds"] or 0), flush=True)
    return rows, closers, spread(closers)


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
    series = SERIES[args.series]
    arms, names = series["arms"], list(series["arms"])
    seeds = series["seeds"][:args.rounds_per_arm] if args.rounds_per_arm else series["seeds"]
    if args.points is None:
        args.points = series["points"]
    plan = {"series": args.series, "started_utc": now(), "session": session, "session_sha256": sha,
            "crow_core_sha256": sha256(CROW_CORE), "probe_sha256": sha256(PROBE),
            "processes_before": procs, "points": {}, "arms": arms, "seeds": seeds}
    print("series start %s, processes %s" % (plan["started_utc"], procs), flush=True)
    if args.points is None:
        rows, closers, points = screen(session, out_dir, args.screen_limit)
        plan["screening"] = {"seed": SCREEN_SEED, "rows": rows, "closers": closers, "points": points}
        print("screening: %d rounds, %d closed %s -> points %s" % (
            sum(1 for r in rows if not r.get("end")), len(closers), closers, points), flush=True)
        if not points:
            print("PREREG Amendment 1 step 4: 1024 does not bind on the 27B in this session", flush=True)
        args.points = points
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
            order = names if (seed == "greedy" or seed % 2 == 0) else names[::-1]
            for arm in order:
                label = "%s-K%d-s%s" % (arm, k, seed)
                r = probe(session, k, seed, arms[arm], label, out_dir)
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
    report(out_dir, args.series)


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


def fisher_fewer(x_b, n_b, x_a, n_a):
    """One-sided Fisher exact p that B has so few events: P(X <= x_b), X hypergeometric with
    n_a + n_b rounds, x_a + x_b events, n_b drawn."""
    tot, ev = n_a + n_b, x_a + x_b
    if not n_b or not tot:
        return None
    den = math.comb(tot, n_b)
    return round(sum(math.comb(ev, i) * math.comb(tot - ev, n_b - i)
                     for i in range(0, min(x_b, ev) + 1)) / den, 4)


def decide_budget(tot, names):
    a, b = tot.get("A1024"), tot.get("B2048")
    if not (a and b and median(a["sec"])):
        return None
    rule = {"a_no_call": b["nc"] <= a["nc"], "b_corrupt": b["corrupt"] <= a["corrupt"],
            "c_wall_clock": median(b["sec"]) <= 1.5 * median(a["sec"])}
    return "PREREG decision rule: %s -> %s" % (rule, "2048 replaces 1024" if all(rule.values()) else "1024 stays")


def decide_sentence(tot, names):
    """PREREG Amendment 2: B only if strictly fewer no-call rounds, not worse on corrupt and
    schema, wall clock <= 1.5x; a pass is a PROPOSAL (one constant for every model)."""
    a, b = tot.get("SA1024"), tot.get("SB1024")
    if not (a and b and median(a["sec"])):
        return None
    rule = {"a_fewer_no_call": b["nc"] < a["nc"],
            "b_corrupt_schema": b["corrupt"] <= a["corrupt"] and b["schema"] <= a["schema"],
            "c_wall_clock": median(b["sec"]) <= 1.5 * median(a["sec"])}
    p = fisher_fewer(b["nc_closed"], b["closed"], a["nc_closed"], a["closed"])
    return ("closed rounds only: no call A %d/%d, B %d/%d, one-sided Fisher p (B fewer) %s\n\n"
            "PREREG decision rule: %s -> %s" % (
                a["nc_closed"], a["closed"], b["nc_closed"], b["closed"], p, rule,
                "sentence B is PROPOSED (robin decides; one constant for every model)"
                if all(rule.values()) else "sentence A stays"))


def determinism(out_dir):
    """PREREG Amendment 2: SA1024 at seeds 0..7 + greedy repeats the budget series' A1024 body."""
    same, n, diff = 0, 0, []
    for name in sorted(os.listdir(out_dir)):
        if not (name.startswith("SA1024-K") and name.endswith(".json")):
            continue
        old = os.path.join(HERE, "A1024-" + name[len("SA1024-"):])
        if not os.path.exists(old):
            continue
        with open(os.path.join(out_dir, name), encoding="utf-8") as fh:
            new_d = json.load(fh)["rounds_detail"][0]
        with open(old, encoding="utf-8") as fh:
            old_d = json.load(fh)["rounds_detail"][0]
        key = lambda d: (d.get("finish"), d.get("n_calls"), d.get("reasoning_chunks"))
        n += 1
        if key(new_d) == key(old_d):
            same += 1
        else:
            diff.append("%s %s vs %s" % (name[:-5], key(new_d), key(old_d)))
    return "determinism vs the budget series' A1024: %d of %d rounds identical (finish, calls, reasoning chunks)%s" % (
        same, n, (": differ " + "; ".join(diff)) if diff else "")


def report(out_dir, series="budget"):
    names = list(SERIES[series]["arms"])
    rows = {}
    for name in sorted(os.listdir(out_dir)):
        # only round files: <arm>-K<k>-s<seed>.json (session-0915.json, plan.json, warm-/scr- are not)
        if not name.endswith(".json") or not name.startswith(tuple(a + "-K" for a in names)):
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
                                 "failed": 0, "nc_closed": 0})
        t["n"] += len(ds)
        t["failed"] += len(rows[(arm, k)]) - len(ds)
        t["nc"] += nc
        t["nc_closed"] += sum(1 for d in ds if not d["n_calls"] and d.get("budget_closed"))
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
    verdict = globals()[SERIES[series]["decide"]](tot, names)
    if verdict:
        lines += ["", verdict]
    if series == "sentence":
        lines += ["", determinism(out_dir)]
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
    ap.add_argument("--screen-limit", type=int, default=None, help="--selftest only: screen the first N candidates")
    ap.add_argument("--series", choices=sorted(SERIES), default="budget",
                    help="budget = 1024 vs 2048 (PREREG + Amendment 1); sentence = A vs B at 1024 (Amendment 2)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.report:
        return report(args.out or SERIES[args.series]["out"], args.series)
    if args.selftest:
        args.session, args.out = args.selftest, args.out or os.path.join(HERE, "selftest")
    else:
        args.session, args.out = SESSION, args.out or SERIES[args.series]["out"]
        if args.rounds_per_arm or args.screen_limit or args.points is not None:
            sys.exit("--rounds-per-arm / --screen-limit / --points are for --selftest only "
                     "(PREREG: screening picks the points, 8 seeds + greedy)")
    run(args)


if __name__ == "__main__":
    sys.exit(main())
