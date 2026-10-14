#!/usr/bin/env python3
"""G4 / #185 live run of 2026-10-11 (robin's GPU go): serve on the GLM-5.3-Flash MUL1K3 container with arm3.env.

  serve   one boot: tools/glm_sweep.py (one prefill 8192 rep + one C1 decode round), then N consecutive real-text
          requests in the same serve session (repo docs as text, greedy, thinking closed, max_tokens 256)
  boots   G4 series (PREREG G4): K boots at 200k context, each with one cold turn of prompt-33k.txt; VRAM after the
          turn (nvidia-smi), commit charge sampled every 0.5 s (peak) and the commit limit, engine m = cold/selections

Writes small JSON + one serve log per boot into this directory. Usage:
  python -I g4_run.py serve [--n 6]      python -I g4_run.py boots [--k 10]      python -I g4_run.py selftest
"""
import argparse
import ctypes
import json
import os
import statistics
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(D, "..", "..", ".."))
ENGINE = os.path.join(REPO, "engine")
SERVE = os.path.join(ENGINE, "target", "release", "serve.exe")
CNQ = os.path.join(REPO, "converter", "GLM-5.3-Flash-MUL1K3.cnq")
TOK = os.path.join(REPO, "models", "GLM-5.3-Flash-original", "tokenizer.json")
ARM3 = os.path.join(REPO, "runs", "glm53-flash", "quick", "arm3.env")
PROMPT33K = r"C:\Users\robin\dev\crow-lab\runs\2026-08-28-flash-next\prompt-33k.txt"
URL = "http://127.0.0.1:8099"
GIB = 1024 ** 3


def arm3_env():
    env = dict(os.environ)
    extra = {}
    for kv in open(ARM3).read().split():
        k, v = kv.split("=", 1)
        if v.startswith("/c/"):  # what MSYS does for quick.sh's `env $ARM` (b6-promptcount JSON: C:/Users/...)
            v = "C:/" + v[3:]
        extra[k] = v
    extra["CROW_CNQ"] = CNQ
    extra["CROW_TOKENIZER"] = TOK
    env.update(extra)
    return env, extra


def git_head():
    return subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()


def vram_used_mib():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True, timeout=20).stdout.split()
    return int(o[0]) if o else None


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def commit():
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    return pi.CommitTotal * pi.PageSize, pi.CommitLimit * pi.PageSize


class CommitSampler:
    def __enter__(self):
        self.peak, self.on = 0, True

        def run():
            while self.on:
                self.peak = max(self.peak, commit()[0])
                time.sleep(0.5)
        self.t = threading.Thread(target=run, daemon=True)
        self.t.start()
        return self

    def __exit__(self, *a):
        self.on = False
        self.t.join()


def health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


def start_serve(log_path):
    env, _ = arm3_env()
    log = open(log_path, "w", encoding="utf-8", errors="replace")
    p = subprocess.Popen([SERVE, "--port", "8099"], cwd=ENGINE, env=env, stdout=log, stderr=subprocess.STDOUT)
    t0 = time.perf_counter()
    while time.perf_counter() - t0 < 600:
        if p.poll() is not None:
            return p, log, None, f"exited rc {p.returncode} before /health"
        if health():
            return p, log, time.perf_counter() - t0, None
        time.sleep(2)
    return p, log, None, "no /health ok within 600 s"


def stop_serve(p, log):
    """Windows serve has no console-control handler (#126): taskkill while idle, then wait for exit and VRAM."""
    if p.poll() is None:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
    try:
        p.wait(60)
    except subprocess.TimeoutExpired:
        pass
    log.close()
    t0 = time.perf_counter()
    v = vram_used_mib()
    while v is not None and v > 3000 and time.perf_counter() - t0 < 120:
        time.sleep(2)
        v = vram_used_mib()
    return {"rc": p.returncode, "vram_after_exit_mib": v, "exit_wait_s": round(time.perf_counter() - t0, 1)}


def chat(content, max_tokens, timeout):
    b = {"messages": [{"role": "user", "content": content}], "stream": True, "temperature": 0,
         "max_tokens": max_tokens, "stream_options": {"include_usage": True}, "timings_per_token": True,
         "reasoning_budget_tokens": 0}
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(b).encode(),
                                 headers={"Content-Type": "application/json"})
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
                    text.append((ch.get("delta") or {}).get("content") or "")
    except Exception as e:  # timeout = a hang candidate; reset = serve died
        err = f"{type(e).__name__}: {e}"
    tm = (last or {}).get("timings") or {}
    usage = (last or {}).get("usage") or {}
    return {
        "error": err,
        "wall_s": round(time.perf_counter() - t0, 3),
        "ttft_s": round(times[0] - t0, 3) if times else None,
        "frames": len(times),
        "prompt_n": tm.get("prompt_n"), "cache_n": tm.get("cache_n"),
        "prompt_per_second": tm.get("prompt_per_second"), "prompt_ms": tm.get("prompt_ms"),
        "predicted_n": tm.get("predicted_n"), "predicted_per_second": tm.get("predicted_per_second"),
        "predicted_ms": tm.get("predicted_ms"),
        "crow_expert_selections": tm.get("crow_expert_selections"), "crow_expert_cold": tm.get("crow_expert_cold"),
        "finish": next((c.get("finish_reason") for c in (last or {}).get("choices") or [] if c.get("finish_reason")), None),
        "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
        "text_head": "".join(text)[:160],
    }


def grep_log(path):
    hits = []
    for line in open(path, encoding="utf-8", errors="replace"):
        if any(s in line for s in ("did not arrive", "panicked", "error", "Error", "refused")):
            hits.append(line.strip()[:300])
    return hits[:20]


# real text from the repo, one doc per request; each prompt a different document slice and question
REAL = [
    ("README.md", 0, 9000, "Summarize what this project is and how it is used, in five bullet points."),
    ("docs/glm5-model.md", 0, 12000, "Explain the layer order of a glm5_next layer as described here."),
    ("docs/glm5-tokenizer.md", 0, 12000, "What does serve still need according to this text? List the items."),
    ("CHANGELOG.md", 0, 12000, "Which changes concern GLM-5.3-Flash? Name three and what they do."),
    ("docs/glm5-mla.md", 0, 12000, "Describe how the MLA latent cache is laid out per token."),
    ("docs/glm5-kda.md", 0, 12000, "What is the KDA recurrent state and how is it carried across calls?"),
    ("docs/env.md", 0, 12000, "Pick five environment variables from this table and explain each in one sentence."),
    ("docs/glm5-head.md", 0, 12000, "How is the head computed from the four hyper-connection streams?"),
]


def cmd_serve(n):
    head = git_head()
    _, extra = arm3_env()
    out = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "commit": head, "binary": SERVE,
           "binary_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(SERVE))),
           "env": extra, "context": "serve default (CROW_CONTEXT unset, floor 200,000)", "vram_before_mib": vram_used_mib()}
    p, log, boot_s, why = start_serve(os.path.join(D, "serve-session.log"))
    out["boot_s"], out["boot_error"] = boot_s, why
    if why is None:
        out["vram_after_boot_mib"] = vram_used_mib()
        cfg = f"{head} arm3.env (runs/glm53-flash/quick/arm3.env), serve, 2026-10-11 G4 session"
        sw = subprocess.run([sys.executable, "-I", os.path.join(REPO, "tools", "glm_sweep.py"), "--url", URL,
                             "--card", "rtx5090", "--config", cfg, "--tokenizer", TOK, "--prefill", "8192",
                             "--conc", "1", "--reps", "1", "--dec-reps", "1", "--no-power",
                             "--out", os.path.join(D, "sweep.json")],
                            capture_output=True, text=True, timeout=7200)
        open(os.path.join(D, "sweep.log"), "w").write(sw.stdout[-4000:] + "\n--stderr--\n" + sw.stderr[-4000:])
        out["sweep_rc"] = sw.returncode
        reqs = []
        for i in range(n):
            path, a, b, q = REAL[i % len(REAL)]
            txt = open(os.path.join(REPO, path), encoding="utf-8", errors="replace").read()[a:b]
            r = chat(f"{txt}\n\n---\n{q}", 256, 900)
            r["i"], r["doc"] = i + 1, path
            reqs.append(r)
            print(json.dumps({k: r[k] for k in ("i", "doc", "error", "prompt_n", "prompt_per_second", "ttft_s",
                                                  "predicted_n", "predicted_per_second", "finish")}), flush=True)
            if r["error"] or p.poll() is not None:
                break
        out["requests"] = reqs
        ok = [r for r in reqs if not r["error"] and r["predicted_per_second"]]
        if ok:
            out["decode_tok_s_median"] = statistics.median(r["predicted_per_second"] for r in ok)
            out["decode_tok_s_after_first_median"] = statistics.median(r["predicted_per_second"] for r in ok[1:]) if len(ok) > 1 else None
        out["vram_after_requests_mib"] = vram_used_mib()
    out["stop"] = stop_serve(p, log)
    out["log_hits"] = grep_log(os.path.join(D, "serve-session.log"))
    json.dump(out, open(os.path.join(D, "serve-session.json"), "w"), indent=1)
    print("serve session done", json.dumps({k: out.get(k) for k in ("boot_s", "boot_error", "sweep_rc", "decode_tok_s_median", "stop")}))


def cmd_boots(k):
    head = git_head()
    _, extra = arm3_env()
    prompt = open(PROMPT33K, encoding="utf-8").read()
    limit = commit()[1]
    rows = []
    for i in range(1, k + 1):
        row = {"boot": i, "start": time.strftime("%H:%M:%S"), "vram_before_mib": vram_used_mib()}
        with CommitSampler() as cs:
            p, log, boot_s, why = start_serve(os.path.join(D, f"boot{i:02d}.log"))
            row["boot_s"], row["boot_error"] = boot_s, why
            if why is None:
                r = chat(prompt, 64, 1800)
                row["turn"] = r
                row["vram_after_turn_mib"] = vram_used_mib()
                row["commit_after_turn_gib"] = round(commit()[0] / GIB, 3)
            row["commit_peak_gib"] = round(cs.peak / GIB, 3)
        row["stop"] = stop_serve(p, log)
        row["log_hits"] = grep_log(os.path.join(D, f"boot{i:02d}.log"))
        t = row.get("turn") or {}
        row["ok"] = bool(why is None and t and not t["error"] and t["predicted_n"])
        if t.get("crow_expert_selections"):
            row["m"] = round(t["crow_expert_cold"] / t["crow_expert_selections"], 4)
        rows.append(row)
        print(json.dumps({"boot": i, "ok": row["ok"], "boot_s": boot_s, "err": why or t.get("error"),
                          "prompt_n": t.get("prompt_n"), "pp": t.get("prompt_per_second"),
                          "vram": row.get("vram_after_turn_mib"), "commit_peak": row["commit_peak_gib"],
                          "m": row.get("m")}), flush=True)
        json.dump({"date": time.strftime("%Y-%m-%d"), "commit": head, "binary": SERVE, "env": extra,
                   "prompt": PROMPT33K, "prompt_sha256": "62141cb4f43e4151c41bfbc0446821c42ec59509ffae2da8c5c47a9bca1bfcde",
                   "commit_limit_gib": round(limit / GIB, 3), "rows": rows},
                  open(os.path.join(D, "g4-boots.json"), "w"), indent=1)
    print(f"boots ok {sum(r['ok'] for r in rows)} / {len(rows)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["serve", "boots", "selftest"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--k", type=int, default=10)
    a = ap.parse_args()
    if a.cmd == "selftest":
        env, extra = arm3_env()
        assert os.path.exists(SERVE) and os.path.exists(CNQ) and os.path.exists(TOK) and os.path.exists(PROMPT33K)
        assert os.path.exists(extra["CROW_GLM_ARENA_WARM"]), extra["CROW_GLM_ARENA_WARM"]
        assert len(extra) == 35, len(extra)
        for path, *_ in REAL:
            assert os.path.exists(os.path.join(REPO, path)), path
        print("commit", commit()[0] / GIB, "limit", commit()[1] / GIB, "vram", vram_used_mib(), "health", health())
        print("selftest ok", len(extra), "env vars")
    elif a.cmd == "serve":
        cmd_serve(a.n)
    else:
        cmd_boots(a.k)
