#!/usr/bin/env python3
"""#214 re-check of 732369c, 2026-10-11 (robin's GPU go): serve, arm3.env, 3 boots per arm alternated
off, on, off, on, off, on (on = CROW_GLM_KV_LAZY=4096); one quick chat each, the acceptance's exact body
(../kv_serve.py chat(TXT, 128, 900, False): greedy, max_tokens 128, timings_per_token, no reasoning budget).
  quick     the series -> quick.json, serve-quick-*.log
  replay    optional: the C1 replay (first three requests of ../kv_serve.py replay), switch on -> replay.json
  selftest  paths, env, no GPU
The caller holds the GPU lock."""
import json, os, re, statistics, sys, time
R = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(R))
import kv_serve as k  # noqa: E402
g = k.g
PATS = k.PATS + ("[budget]", "staging set", "VRAM experts")


def parse(lines):
    o = {"staging_boot_line": any("staging set of 128 slots allocated at boot" in l for l in lines)}
    m = [l for l in lines if "[chat] KV lazy" in l]
    if m:
        l = m[-1]
        for key, pat in (("slots", r"VRAM slots (\d+)"), ("live", r"\((\d+) of \d+ elastic"), ("chunks", r"of (\d+) elastic chunks x"),
                         ("chunk_slots", r"elastic chunks x (\d+) live"), ("enter", r"elastic enter (\d+)"), ("exit", r" exit (\d+)"),
                         ("realloc_fail", r"realloc_fail (\d+)"), ("growths", r"growths (\d+)")):
            x = re.search(pat, l)
            o[key] = int(x.group(1)) if x else None
    return o


def quick():
    rows = []
    for i in range(1, 4):
        for on in (False, True):
            lab = f"quick-{'on' if on else 'off'}-r{i}"
            log = os.path.join(R, f"serve-{lab}.log")
            row = {"label": lab, "on": on, "start": time.strftime("%H:%M:%S"), "vram_before_mib": g.vram_used_mib()}
            k.set_arm(on)
            p, lf, row["boot_s"], row["boot_error"] = g.start_serve(log)
            if row["boot_error"] is None:
                row["vram_after_boot_mib"] = g.vram_used_mib()
                row["turn"] = k.chat(k.TXT, 128, 900, False)
                row["vram_after_turn_mib"] = g.vram_used_mib()
            row["stop"] = g.stop_serve(p, lf)
            row["log"] = k.log_lines(log, PATS)
            row.update(parse(row["log"]))
            t = row.get("turn") or {}
            print(json.dumps({"label": lab, "boot_s": row["boot_s"], "err": row["boot_error"] or t.get("error"), "prompt_n": t.get("prompt_n"),
                              "predicted_n": t.get("predicted_n"), "ms_per_token": t.get("ms_per_token"),
                              **{x: row.get(x) for x in ("staging_boot_line", "slots", "live", "chunks", "chunk_slots", "realloc_fail", "exit", "enter")}}), flush=True)
            rows.append(row)
            json.dump({"commit": g.git_head(), "binary": g.SERVE, "binary_mtime": time.ctime(os.path.getmtime(g.SERVE)),
                       "env": g.arm3_env()[1], "rows": rows}, open(os.path.join(R, "quick.json"), "w"), indent=1)
    for on in (False, True):
        ms = [r["turn"]["ms_per_token"] for r in rows if r["on"] == on and (r.get("turn") or {}).get("ms_per_token")]
        print(f"{'on ' if on else 'off'}: ms/token {[round(x, 2) for x in ms]} median {round(statistics.median(ms), 2) if ms else None}", flush=True)


def replay():
    """the acceptance replay's first three requests (warm-up, prefill8192, C1), same seed and nonce, switch on."""
    sys.path.insert(0, os.path.join(g.REPO, "tools"))
    import glm_sweep as sw
    seed, nonce = 1791698016, "a6077b493177720232c654608cbe174b"
    words = sw.vocab_words(g.TOK)
    q = lambda i: f"[cache-bust nonce: {nonce}-{i}]\nWrite a detailed, well-structured explanation of {sw.TOPICS[i % len(sw.TOPICS)]} (variant {i})."
    reqs = [("warm-up", sw.body(sw.rand_words(words, 512, seed), 1, False)),
            ("prefill8192", sw.body(sw.rand_words(words, 8192 - 12, seed + 1000 * 8192), 1, False)),
            ("C1", sw.body(q(0), sw.SERVE_MAX_TOKENS, True))]
    log = os.path.join(R, "serve-replay.log")
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": g.git_head(), "env": g.arm3_env()[1], "CROW_GLM_KV_LAZY": "4096", "requests": []}
    k.set_arm(True)
    p, lf, out["boot_s"], out["boot_error"] = g.start_serve(log)
    if out["boot_error"] is None:
        for name, b in reqs:
            t0, times, last, text = sw.post(g.URL, b)
            f = sw.final_fields(last)
            tm = f["timings"] or {}
            r = {"name": name, "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"), "finish": f["finish"]}
            if r["predicted_n"] and r["predicted_n"] > 1:
                r["ms_per_token"] = r["predicted_ms"] / r["predicted_n"]
            out["requests"].append(r)
            print(json.dumps(r), flush=True)
    out["stop"] = g.stop_serve(p, lf)
    out["log"] = k.log_lines(log, PATS)
    out["kv"] = parse(out["log"])
    json.dump(out, open(os.path.join(R, "replay.json"), "w"), indent=1)
    print(json.dumps(out["kv"]))


if __name__ == "__main__":
    c = sys.argv[1]
    if c == "quick":
        quick()
    elif c == "replay":
        replay()
    else:
        assert os.path.exists(g.SERVE) and os.path.exists(g.CNQ) and os.path.exists(g.TOK)
        print("selftest", g.git_head(), len(g.arm3_env()[1]), "health", g.health(), "vram", g.vram_used_mib(), "R", R)
        sample = open(os.path.join(os.path.dirname(R), "serve-quick-on-r1.log"), encoding="utf-8").read().splitlines()
        print(parse(sample))
