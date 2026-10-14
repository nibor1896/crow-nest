#!/usr/bin/env python3
"""#214 acceptance, 2026-10-11 (robin's GPU go): serve arms with arm3.env, CROW_GLM_KV_LAZY unset vs 4096.

  quick [--k 3]  K x (off, on) boots alternated; one chat of the quick decode question (57 ids in glm5_run), greedy,
                 max_tokens 128, thinking as served (no reasoning budget); predicted_ms / predicted_n, the
                 `[chat] KV lazy (#214)` line (VRAM slots)
  sweep          one boot, switch on: tools/glm_sweep.py exactly as g3g4-20261011/g4_run.py serve ran it (seq 4 there =
                 C1@32k, the 32,830-token turn, 76.79 ms/token)
  long [--words 190000]  one boot, switch on: one turn of a random-word prompt (glm_sweep's vocab_words, seed 214) of
                 >= 190,000 rows + max_tokens 256, thinking off
The caller holds the GPU lock. Writes <mode>*.json and serve-<mode>*.log here.
"""
import argparse, json, os, re, statistics, subprocess, sys, time, urllib.request

D = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, D)
import g4_run as g  # noqa: E402  (copy of g3g4-20261011/g4_run.py; its D is this directory)

LAZY = "CROW_GLM_KV_LAZY"
TXT = ("Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus "
       "write-through, and how false sharing hurts multithreaded code. Give concrete examples.")


def set_arm(on):
    os.environ.pop(LAZY, None)
    os.environ.pop("CROW_GLM_MTP", None)
    if on:
        os.environ[LAZY] = "4096"


def chat(content, max_tokens, timeout, think_off):
    b = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0, "max_tokens": max_tokens,
         "stream_options": {"include_usage": True}, "timings_per_token": True}
    if think_off:
        b["reasoning_budget_tokens"] = 0
    req = urllib.request.Request(g.URL + "/v1/chat/completions", data=json.dumps(b).encode(), headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, last, text, err = [], None, [], None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                last = json.loads(body)
                times.append(time.perf_counter())
                for ch in last.get("choices") or []:
                    d = ch.get("delta") or {}
                    text.append((d.get("content") or "") + (d.get("reasoning_content") or ""))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    tm = (last or {}).get("timings") or {}
    out = {"error": err, "wall_s": round(time.perf_counter() - t0, 3), "ttft_s": round(times[0] - t0, 3) if times else None,
           "frames": len(times), "text": "".join(text),
           "finish": next((c.get("finish_reason") for c in (last or {}).get("choices") or [] if c.get("finish_reason")), None)}
    for k in ("prompt_n", "cache_n", "prompt_ms", "prompt_per_second", "predicted_n", "predicted_ms", "predicted_per_second",
              "crow_expert_selections", "crow_expert_cold"):
        out[k] = tm.get(k)
    if out["predicted_n"]:
        out["ms_per_token"] = out["predicted_ms"] / out["predicted_n"]
    return out


def log_lines(path, pats):
    return [l.rstrip()[:600] for l in open(path, encoding="utf-8", errors="replace") if any(p in l for p in pats)]


PATS = ("KV lazy", "CROW_GLM_KV_LAZY", "VRAM experts per MoE layer", "glm5_next tiers per MoE layer", "out of memory", "OOM",
        "panicked", "refused", "\"event\":\"routing\"", "elastic")


def boot(on, log):
    set_arm(on)
    return g.start_serve(log)


def cmd_quick(k):
    rows = []
    order = [(False, True), (True, False)]
    for i in range(1, k + 1):
        for on in order[(i - 1) % 2]:
            lab = f"quick-{'on' if on else 'off'}-r{i}"
            log = os.path.join(D, f"serve-{lab}.log")
            row = {"label": lab, "on": on, "start": time.strftime("%H:%M:%S"), "vram_before_mib": g.vram_used_mib()}
            p, lf, row["boot_s"], row["boot_error"] = boot(on, log)
            if row["boot_error"] is None:
                row["vram_after_boot_mib"] = g.vram_used_mib()
                row["turn"] = chat(TXT, 128, 900, False)
                row["vram_after_turn_mib"] = g.vram_used_mib()
            row["stop"] = g.stop_serve(p, lf)
            row["log"] = log_lines(log, PATS)
            m = [re.search(r"VRAM slots (\d+)", l) for l in row["log"] if "[chat] KV lazy" in l]
            row["kv_slots"] = int(m[-1].group(1)) if m and m[-1] else None
            t = row.get("turn") or {}
            print(json.dumps({"label": lab, "boot_s": row["boot_s"], "err": row["boot_error"] or t.get("error"), "prompt_n": t.get("prompt_n"),
                              "predicted_n": t.get("predicted_n"), "ms_per_token": t.get("ms_per_token"), "kv_slots": row["kv_slots"]}), flush=True)
            rows.append(row)
            json.dump({"commit": g.git_head(), "binary": g.SERVE, "binary_mtime": time.ctime(os.path.getmtime(g.SERVE)),
                       "env": g.arm3_env()[1], "rows": rows}, open(os.path.join(D, "quick.json"), "w"), indent=1)
    for on in (False, True):
        ms = [r["turn"]["ms_per_token"] for r in rows if r["on"] == on and r.get("turn") and r["turn"].get("ms_per_token")]
        print(f"{'on ' if on else 'off'}: ms/token {[round(x, 2) for x in ms]} median {statistics.median(ms) if ms else None}")
    texts = {r["label"]: r["turn"]["text"] for r in rows if r.get("turn")}
    print("texts identical across all serve runs:", len(set(texts.values())) == 1, len(texts))


def cmd_sweep():
    head = g.git_head()
    log = os.path.join(D, "serve-sweep.log")
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": head, "env": g.arm3_env()[1], LAZY: "4096", "vram_before_mib": g.vram_used_mib()}
    p, lf, out["boot_s"], out["boot_error"] = boot(True, log)
    if out["boot_error"] is None:
        cfg = f"{head} arm3.env + CROW_GLM_KV_LAZY=4096, serve, 2026-10-11 #214 acceptance"
        sw = subprocess.run([sys.executable, "-I", os.path.join(g.REPO, "tools", "glm_sweep.py"), "--url", g.URL, "--card", "rtx5090",
                             "--config", cfg, "--tokenizer", g.TOK, "--prefill", "8192", "--conc", "1", "--reps", "1", "--dec-reps", "1",
                             "--no-power", "--out", os.path.join(D, "sweep32k.json")], capture_output=True, text=True, timeout=7200)
        open(os.path.join(D, "sweep32k.out"), "w").write(sw.stdout[-4000:] + "\n--stderr--\n" + sw.stderr[-4000:])
        out["sweep_rc"] = sw.returncode
        out["vram_after_mib"] = g.vram_used_mib()
    out["stop"] = g.stop_serve(p, lf)
    out["log"] = log_lines(log, PATS)
    for l in out["log"]:
        if '"seq":4' in l:
            j = json.loads(l[l.index("{"):])
            out["seq4"] = {k: j[k] for k in ("prompt_n", "predicted_n", "predicted_ms", "tok_s", "prompt_ms")}
            out["seq4"]["ms_per_token"] = j["predicted_ms"] / j["predicted_n"]
    json.dump(out, open(os.path.join(D, "sweep32k-run.json"), "w"), indent=1)
    print(json.dumps({k: out.get(k) for k in ("boot_s", "boot_error", "sweep_rc", "seq4", "stop")}))


def cmd_replay():
    """the first four requests of g3g4-20261011/serve-session.log, byte for byte: glm_sweep's seed of that run
    (1791698016, recovered from its prefill receipt sha256) and its nonce (sweep.json nonce_prefix); seq 4 = the
    32,830-token turn (76.79 ms/token there). Bodies as glm_sweep.body builds them."""
    sys.path.insert(0, os.path.join(g.REPO, "tools"))
    import glm_sweep as sw
    import hashlib
    seed, nonce = 1791698016, "a6077b493177720232c654608cbe174b"
    words = sw.vocab_words(g.TOK)
    q = lambda i: f"[cache-bust nonce: {nonce}-{i}]\nWrite a detailed, well-structured explanation of {sw.TOPICS[i % len(sw.TOPICS)]} (variant {i})."
    pre8 = sw.rand_words(words, 8192 - 12, seed + 1000 * 8192)
    assert hashlib.sha256(pre8.encode()).hexdigest() == "84abe346024c81f9d504bfea10292421c6b62abc5ee02c1cfd17812d0a335ca3"
    assert hashlib.sha256(q(9000).encode()).hexdigest() == "9ee109bc2d69aaa7930545c72bcb15884675c5e5a7635a54bd5ddcadaeec7260"
    reqs = [("warm-up", sw.body(sw.rand_words(words, 512, seed), 1, False)), ("prefill8192", sw.body(pre8, 1, False)),
            ("C1", sw.body(q(0), sw.SERVE_MAX_TOKENS, True)),
            ("C1@32k", sw.body(sw.rand_words(words, 32768, seed + 99) + "\n\n" + q(9000), sw.SERVE_MAX_TOKENS, True))]
    log = os.path.join(D, "serve-replay.log")
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": g.git_head(), "env": g.arm3_env()[1], LAZY: "4096",
           "seed": seed, "nonce": nonce, "vram_before_mib": g.vram_used_mib(), "requests": []}
    p, lf, out["boot_s"], out["boot_error"] = boot(True, log)
    if out["boot_error"] is None:
        for name, b in reqs:
            t0, times, last, text = sw.post(g.URL, b)
            f = sw.final_fields(last)
            tm = f["timings"] or {}
            r = {"name": name, "prompt_n": tm.get("prompt_n"), "predicted_n": tm.get("predicted_n"), "predicted_ms": tm.get("predicted_ms"),
                 "prompt_ms": tm.get("prompt_ms"), "finish": f["finish"], "text_sha256": hashlib.sha256(text.encode()).hexdigest()}
            if r["predicted_n"] and r["predicted_n"] > 1:
                r["ms_per_token"] = r["predicted_ms"] / r["predicted_n"]
            out["requests"].append(r)
            print(json.dumps(r), flush=True)
        out["vram_after_mib"] = g.vram_used_mib()
    out["stop"] = g.stop_serve(p, lf)
    out["log"] = log_lines(log, PATS)
    json.dump(out, open(os.path.join(D, "replay32k.json"), "w"), indent=1)


def cmd_long(words_n):
    sys.path.insert(0, os.path.join(g.REPO, "tools"))
    import glm_sweep as sw
    words = sw.vocab_words(g.TOK)
    content = sw.rand_words(words, words_n, 214) + "\n\nWrite a detailed, well-structured explanation of how a jet engine works."
    log = os.path.join(D, "serve-long.log")
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": g.git_head(), "env": g.arm3_env()[1], LAZY: "4096",
           "words": words_n, "seed": 214, "commit_limit_gib": round(g.commit()[1] / g.GIB, 3), "vram_before_mib": g.vram_used_mib()}
    with g.CommitSampler() as cs:
        p, lf, out["boot_s"], out["boot_error"] = boot(True, log)
        if out["boot_error"] is None:
            out["vram_after_boot_mib"] = g.vram_used_mib()
            t = chat(content, 256, 5400, True)
            t["text"] = t["text"][:300]
            out["turn"] = t
            out["vram_after_turn_mib"] = g.vram_used_mib()
            out["serve_alive_after_turn"] = p.poll() is None
        out["commit_peak_gib"] = round(cs.peak / g.GIB, 3)
    out["stop"] = g.stop_serve(p, lf)
    out["log"] = log_lines(log, PATS)
    t = out.get("turn") or {}
    out["rows_reached"] = (t.get("prompt_n") or 0) + (t.get("predicted_n") or 0)
    json.dump(out, open(os.path.join(D, "long190k.json"), "w"), indent=1)
    print(json.dumps({k: out.get(k) for k in ("boot_s", "boot_error", "rows_reached", "vram_after_turn_mib", "commit_peak_gib", "stop")}),
          json.dumps({k: t.get(k) for k in ("error", "prompt_n", "prompt_per_second", "predicted_n", "ms_per_token", "finish")}))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["quick", "sweep", "replay", "long", "selftest"])
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--words", type=int, default=190000)
    a = ap.parse_args()
    if a.cmd == "quick":
        cmd_quick(a.k)
    elif a.cmd == "sweep":
        cmd_sweep()
    elif a.cmd == "replay":
        cmd_replay()
    elif a.cmd == "long":
        cmd_long(a.words)
    else:
        assert os.path.exists(g.SERVE) and os.path.exists(g.CNQ) and os.path.exists(g.TOK)
        set_arm(True)
        print("env", len(g.arm3_env()[1]), os.environ.get(LAZY), "health", g.health(), "vram", g.vram_used_mib())
        sys.path.insert(0, os.path.join(g.REPO, "tools"))
        import glm_sweep as sw
        w = sw.vocab_words(g.TOK)
        print("vocab words", len(w), "sample", sw.rand_words(w, 5, 214))
        print("selftest ok")
