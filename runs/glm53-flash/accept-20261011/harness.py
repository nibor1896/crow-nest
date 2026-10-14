"""Serve harness for the 2026-10-11 GPU acceptance block (one engine at a time under <home>/dev/.gpu-lock).

Every child is started with CREATE_NO_WINDOW (no console on robin's desktop). serve = arm4.env + the arm's switches,
`serve --port 8099`, CROW_CONTEXT unset (200,000), stdout+stderr to one log per boot.
Per request: the client result (text, reasoning, tool calls, finish, timings, usage, crow_ids when asked) plus the
serve log lines that request wrote ([cache], [chat], [glm5] prompt phase, routing JSON).
"""
import ctypes
import json
import os
import re
import subprocess
import time
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
ENGINE = os.path.join(REPO, "engine")
SERVE = os.path.join(ENGINE, "target", "release", "serve.exe")
CNQ = os.path.join(REPO, "converter", "GLM-5.3-Flash-MUL1K3.cnq")
TOK = os.path.join(REPO, "models", "GLM-5.3-Flash-original", "tokenizer.json")
ARM4 = os.path.join(REPO, "runs", "glm53-flash", "quick", "arm4.env")
URL = "http://127.0.0.1:8099"
NO_WINDOW = 0x08000000
SWITCHES = ("CROW_GLM_ANSWER_REUSE", "CROW_GLM_PREFIX_ANCHOR", "CROW_GLM_SMALL_PROMPT", "CROW_GLM_SMALL_PROMPT_ROWS",
            "CROW_GLM_MTP", "CROW_CONTEXT")


def arm4_env(extra_sw=None):
    env = dict(os.environ)
    for k in SWITCHES:
        env.pop(k, None)
    extra = {}
    for kv in open(ARM4).read().split():
        k, v = kv.split("=", 1)
        if v.startswith("/c/"):
            v = "C:/" + v[3:]
        extra[k] = v
    extra["CROW_CNQ"] = CNQ
    extra["CROW_TOKENIZER"] = TOK
    extra.update(extra_sw or {})
    env.update(extra)
    return env, extra


def run_hidden(args, **kw):
    return subprocess.run(args, capture_output=True, text=True, creationflags=NO_WINDOW, **kw)


def vram_used_mib():
    try:
        o = run_hidden(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], timeout=20).stdout.split()
        return int(o[0]) if o else None
    except Exception:
        return None


class PERF(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def perf():
    pi = PERF()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    g = pi.PageSize / 1024 ** 3
    return {"commit_gib": round(pi.CommitTotal * g, 2), "avail_phys_gib": round(pi.PhysicalAvailable * g, 2)}


def health():
    try:
        with urllib.request.urlopen(URL + "/health", timeout=5) as r:
            return r.status == 200
    except Exception:
        return False


class Serve:
    def __init__(self, log_path, switches=None, serve_bin=SERVE):
        self.log_path = log_path
        self.serve_bin = serve_bin
        self.env, self.extra = arm4_env(switches)
        self.switches = dict(switches or {})

    def __enter__(self):
        t0 = time.perf_counter()
        # wait for the GPU to be empty of a previous serve
        while (vram_used_mib() or 0) > 3000 and time.perf_counter() - t0 < 180:
            time.sleep(2)
        self.vram_before = vram_used_mib()
        self.perf_before = perf()
        self.log = open(self.log_path, "w", encoding="utf-8", errors="replace")
        self.p = subprocess.Popen([self.serve_bin, "--port", "8099"], cwd=ENGINE, env=self.env, stdout=self.log,
                                  stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, creationflags=NO_WINDOW)
        t0 = time.perf_counter()
        self.boot_err = None
        while True:
            if self.p.poll() is not None:
                self.boot_err = f"exited rc {self.p.returncode} before /health"
                break
            if health():
                break
            if time.perf_counter() - t0 > 900:
                self.boot_err = "no /health within 900 s"
                break
            time.sleep(2)
        self.boot_s = round(time.perf_counter() - t0, 1)
        return self

    def boot_lines(self):
        out = []
        for ln in open(self.log_path, encoding="utf-8", errors="replace"):
            if ln.startswith(("[serve]", "[budget] glm5_next tiers", "[budget] CROW_GLM", "[budget] device sampler")) \
                    or "device sampler" in ln or "#217" in ln or "#219" in ln or "#216" in ln:
                if "GET /health" not in ln and "GET /props" not in ln:
                    out.append(ln.rstrip()[:600])
        return out

    def __exit__(self, *a):
        if self.p.poll() is None:
            run_hidden(["taskkill", "/PID", str(self.p.pid), "/T", "/F"])
        try:
            self.p.wait(90)
        except subprocess.TimeoutExpired:
            pass
        self.log.close()
        t0 = time.perf_counter()
        while (vram_used_mib() or 0) > 3000 and time.perf_counter() - t0 < 180:
            time.sleep(2)
        self.alive_at_exit = self.p.returncode

    def chat(self, body, timeout=7200):
        off = os.path.getsize(self.log_path)
        t0 = time.perf_counter()
        res = post(body, timeout)
        res["wall_s"] = round(time.perf_counter() - t0, 3)
        time.sleep(1.0)
        self.log.flush()
        with open(self.log_path, "rb") as fh:
            fh.seek(off)
            new = fh.read().decode("utf-8", errors="replace").splitlines()
        res["log"] = [l for l in new if l.startswith(("[cache]", "[chat]", "[glm5]", "{")) or "panicked" in l
                      or "error" in l.lower()][:80]
        res["routing"] = next((json.loads(l) for l in new if l.startswith("{") and '"event":"routing"' in l), None)
        res["serve_alive"] = self.p.poll() is None
        return res


def post(body, timeout):
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    content, reasoning, calls, ids, last, finish, err = [], [], {}, [], None, None, None
    first_lp = []
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            if body.get("stream"):
                for raw in r:
                    line = raw.decode("utf-8").strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[5:].strip()
                    if data == "[DONE]":
                        break
                    last = json.loads(data)
                    for ch in last.get("choices") or []:
                        d = ch.get("delta") or {}
                        content.append(d.get("content") or "")
                        reasoning.append(d.get("reasoning_content") or "")
                        for tc in d.get("tool_calls") or []:
                            c = calls.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                            f = tc.get("function") or {}
                            c["name"] += f.get("name") or ""
                            c["arguments"] += f.get("arguments") or ""
                        for e in ((ch.get("logprobs") or {}).get("content") or []):
                            ids.append(e.get("crow_id"))
                            if not first_lp:
                                first_lp.append(e)
                        finish = ch.get("finish_reason") or finish
            else:
                last = json.loads(r.read())
                ch = last["choices"][0]
                m = ch.get("message") or {}
                content.append(m.get("content") or "")
                reasoning.append(m.get("reasoning_content") or "")
                for i, tc in enumerate(m.get("tool_calls") or []):
                    calls[i] = {"name": tc["function"]["name"], "arguments": tc["function"]["arguments"]}
                for e in ((ch.get("logprobs") or {}).get("content") or []):
                    ids.append(e.get("crow_id"))
                finish = ch.get("finish_reason")
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    tm = (last or {}).get("timings") or {}
    return {"error": err, "finish": finish, "content": "".join(content), "reasoning": "".join(reasoning),
            "tool_calls": [calls[k] for k in sorted(calls)], "ids": ids or None, "timings": tm,
            "usage": (last or {}).get("usage"), "first_logprobs": first_lp[0] if first_lp else None}


COUNTERS = re.compile(r"expert accesses vram (\d+) pinned (\d+) nvme (\d+), NVMe bytes (\d+)")
CHATP = re.compile(r"\[chat\] prompt (\d+) tok \((\d+) cached, (\d+) prefilled\), generated (\d+) tok, prefill ([\d.]+) ms"
                   r" \(([\d.]+) tok/s\), reset ([\d.]+) ms, decode ([\d.]+) ms, ([\d.]+) tok/s")


def counters(res):
    for l in res.get("log") or []:
        m = COUNTERS.search(l)
        if m:
            return [int(x) for x in m.groups()]
    return None


def chat_line(res):
    for l in res.get("log") or []:
        m = CHATP.search(l)
        if m:
            g = m.groups()
            return {"P": int(g[0]), "C": int(g[1]), "F": int(g[2]), "gen": int(g[3]), "prefill_ms": float(g[4]),
                    "prefill_tok_s": float(g[5]), "reset_ms": float(g[6]), "decode_ms": float(g[7]),
                    "decode_tok_s": float(g[8])}
    return None


def summary(res):
    r = res.get("routing") or {}
    c = chat_line(res) or {}
    pn = r.get("predicted_n") or 0
    return {"P": c.get("P"), "C": c.get("C"), "F": c.get("F"), "gen": pn, "prefill_ms": c.get("prefill_ms"),
            "decode_tok_s": c.get("decode_tok_s"),
            "ms_per_token": round(r["predicted_ms"] / pn, 3) if pn else None, "finish": res.get("finish"),
            "err": res.get("error"), "cache": next((l[:200] for l in res.get("log") or [] if l.startswith("[cache] ") and
                                                     ("COLD" in l or "WARM" in l or "LIVE" in l)), None)}


def jdump(path, obj):
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj) + "\n")


if __name__ == "__main__":
    print("selftest", os.path.exists(SERVE), os.path.exists(CNQ), os.path.exists(TOK), vram_used_mib(), perf(), health())
    e, x = arm4_env({"CROW_GLM_ANSWER_REUSE": "1"})
    print(len(x), "env vars;", x.get("CROW_GLM_ARENA_WARM"), os.path.exists(x["CROW_GLM_ARENA_WARM"]),
          [k for k in SWITCHES if k in e])
