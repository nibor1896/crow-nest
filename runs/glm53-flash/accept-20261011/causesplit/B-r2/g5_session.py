#!/usr/bin/env python3
"""G5 window session of 2026-10-11 (crow-nest #151, PREREG amendment 8, robin's GPU go): one serve boot for t1..t5.

  start     start serve DETACHED (arm4.env and nothing else, port 8099, CROW_CONTEXT unset -> floor 200,000),
            stdout+stderr to serve-tasks.log, metadata to serve-tasks.json; then start the sampler DETACHED
  sampler   read-only per-row recorder (amendment 8 "Per-row record"): every 2 s nvidia-smi memory.used, Windows
            commit charge / limit and available physical memory -> resources.jsonl; tails serve-tasks.log and,
            per routing JSON line, writes one row to rows.jsonl with that request's [cache] / [chat] lines
            verbatim, P = prompt_n, C = cached_n, F = P - C, C/P, prefill and decode tok/s, m = cold/selections,
            bytes_streamed and the last resource sample. Task ids / side-row flags are assigned afterwards
            against Crow's turn lines; this file only records.

Usage:  python -I g5_session.py start | sampler | sampler-start (restart the sampler alone, hidden)
"""
import ctypes
import json
import os
import re
import subprocess
import sys
import time

D = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(D, "..", "..", ".."))
ENGINE = os.path.join(REPO, "engine")
SERVE = os.path.join(ENGINE, "target", "release", "serve.exe")
CNQ = os.path.join(REPO, "converter", "GLM-5.3-Flash-MUL1K3.cnq")
TOK = os.path.join(REPO, "models", "GLM-5.3-Flash-original", "tokenizer.json")
ARM4 = os.path.join(REPO, "runs", "glm53-flash", "quick", "arm4.env")
LOG = os.path.join(D, "serve-tasks.log")
DETACHED = 0x00000008 | 0x00000200  # DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
BREAKAWAY = 0x01000000              # CREATE_BREAKAWAY_FROM_JOB
NO_WINDOW = 0x08000000              # CREATE_NO_WINDOW: no console window for any child (robin, 2026-10-11)


def arm4_env():
    env = dict(os.environ)
    extra = {}
    for kv in open(ARM4).read().split():
        k, v = kv.split("=", 1)
        if v.startswith("/c/"):  # as g4_run.py / quick.sh under MSYS
            v = "C:/" + v[3:]
        extra[k] = v
    extra["CROW_CNQ"] = CNQ
    extra["CROW_TOKENIZER"] = TOK
    env.pop("CROW_CONTEXT", None)
    env.update(extra)
    return env, extra


def vram_used_mib():
    try:
        o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=20, creationflags=NO_WINDOW).stdout.split()
        return int(o[0]) if o else None
    except Exception:
        return None


class PERFORMANCE_INFORMATION(ctypes.Structure):
    _fields_ = [("cb", ctypes.c_ulong), ("CommitTotal", ctypes.c_size_t), ("CommitLimit", ctypes.c_size_t),
                ("CommitPeak", ctypes.c_size_t), ("PhysicalTotal", ctypes.c_size_t),
                ("PhysicalAvailable", ctypes.c_size_t), ("SystemCache", ctypes.c_size_t),
                ("KernelTotal", ctypes.c_size_t), ("KernelPaged", ctypes.c_size_t),
                ("KernelNonpaged", ctypes.c_size_t), ("PageSize", ctypes.c_size_t),
                ("HandleCount", ctypes.c_ulong), ("ProcessCount", ctypes.c_ulong), ("ThreadCount", ctypes.c_ulong)]


def perf():
    pi = PERFORMANCE_INFORMATION()
    pi.cb = ctypes.sizeof(pi)
    ctypes.windll.psapi.GetPerformanceInfo(ctypes.byref(pi), pi.cb)
    g = pi.PageSize / 1024 ** 3
    return {"commit_gib": round(pi.CommitTotal * g, 3), "commit_limit_gib": round(pi.CommitLimit * g, 3),
            "avail_phys_gib": round(pi.PhysicalAvailable * g, 3)}


def spawn(args, out_path, base=DETACHED, **kw):
    out = open(out_path, "w", encoding="utf-8", errors="replace")
    for flags in (base | BREAKAWAY, base):
        try:
            return subprocess.Popen(args, stdout=out, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                    creationflags=flags, close_fds=True, **kw), flags
        except OSError:
            continue
    raise RuntimeError("could not spawn " + args[0])


def cmd_start():
    env, extra = arm4_env()
    meta = {"date": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "commit": subprocess.run(["git", "-C", REPO, "rev-parse", "--short", "HEAD"],
                                     capture_output=True, text=True, creationflags=NO_WINDOW).stdout.strip(),
            "binary": SERVE, "binary_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(SERVE))),
            "args": [SERVE, "--port", "8099"], "cwd": ENGINE, "env": extra,
            "context": "serve default (CROW_CONTEXT unset, floor 200,000)", "vram_before_mib": vram_used_mib(),
            "perf_before": perf(), "t_start_epoch": time.time()}
    p, flags = spawn([SERVE, "--port", "8099"], LOG, cwd=ENGINE, env=env)
    meta["serve_pid"], meta["serve_flags"] = p.pid, hex(flags)
    s, sflags = start_sampler()
    meta["sampler_pid"], meta["sampler_flags"] = s.pid, hex(sflags)
    json.dump(meta, open(os.path.join(D, "serve-tasks.json"), "w"), indent=1)
    print(json.dumps({"serve_pid": p.pid, "sampler_pid": s.pid, "flags": hex(flags)}))


def start_sampler():
    py = os.path.join(os.path.dirname(sys.executable), "python.exe")
    return spawn([py, "-I", os.path.abspath(__file__), "sampler"], os.path.join(D, "sampler.out"),
                 base=NO_WINDOW | 0x00000200, cwd=D)


def cmd_sampler_start():
    s, flags = start_sampler()
    m = json.load(open(os.path.join(D, "serve-tasks.json")))
    m["sampler_pid_old"], m["sampler_pid"], m["sampler_flags"] = m.get("sampler_pid"), s.pid, hex(flags)
    m["sampler_note"] = "restarted with CREATE_NO_WINDOW: the first sampler popped a console per nvidia-smi call"
    json.dump(m, open(os.path.join(D, "serve-tasks.json"), "w"), indent=1)
    print(json.dumps({"sampler_pid": s.pid, "flags": hex(flags)}))


ROUTE = re.compile(r'^\{.*"event":"routing"')


def cmd_sampler():
    res_f = open(os.path.join(D, "resources.jsonl"), "a", buffering=1)
    rows_f = open(os.path.join(D, "rows.jsonl"), "a", buffering=1)
    while not os.path.exists(LOG):
        time.sleep(1)
    fh = open(LOG, "r", encoding="utf-8", errors="replace")
    pend, buf, last_res, t_res, row_i = [], "", None, 0.0, 0
    while True:
        now = time.time()
        if now - t_res >= 2.0:
            last_res = {"ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)), "vram_mib": vram_used_mib(), **perf()}
            res_f.write(json.dumps(last_res) + "\n")
            t_res = now
        chunk = fh.read()
        if chunk:
            buf += chunk
            *lines, buf = buf.split("\n")
            for ln in lines:
                ln = ln.rstrip("\r")
                if ROUTE.match(ln):
                    try:
                        r = json.loads(ln)
                    except ValueError:
                        pend.append(ln)
                        continue
                    row_i += 1
                    P, C = r.get("prompt_n", 0), r.get("cached_n", 0)
                    chat = next((x for x in reversed(pend) if x.startswith("[chat] prompt ")), "")
                    pre = re.search(r"prefill ([\d.]+) ms \(([\d.]+) tok/s\)", chat)
                    dec = re.search(r"decode ([\d.]+) ms, ([\d.]+) tok/s", chat)
                    rst = re.search(r"reset ([\d.]+) ms", chat)
                    sel, cold = r.get("selections") or 0, r.get("cold") or 0
                    row = {"row": row_i, "ts": r.get("ts"), "seq": r.get("seq"), "route": r.get("route"),
                           "P_prompt_n": P, "C_cached_n": C, "F_prefilled": P - C,
                           "C_over_P": round(C / P, 4) if P else None,
                           "prefill_ms": r.get("prompt_ms"), "prefill_tok_s": float(pre.group(2)) if pre else None,
                           "predicted_n": r.get("predicted_n"), "predicted_ms": r.get("predicted_ms"),
                           "decode_tok_s": float(dec.group(2)) if dec else r.get("tok_s"),
                           "wall_ms": round((float(rst.group(1)) if rst else 0) + (float(pre.group(1)) if pre else 0)
                                            + (float(dec.group(1)) if dec else 0), 1) if chat else None,
                           "finish": r.get("finish"), "m": round(cold / sel, 4) if sel else None,
                           "selections": sel, "cold": cold, "bytes_streamed": r.get("bytes_streamed"),
                           "cache_lines": [x for x in pend if x.startswith("[cache]")],
                           "chat_lines": [x for x in pend if x.startswith("[chat]")],
                           "resources_at_end": last_res, "routing": r}
                    rows_f.write(json.dumps(row) + "\n")
                    pend = []
                elif ln.startswith("[cache]") or ln.startswith("[chat]"):
                    pend.append(ln)
        else:
            time.sleep(0.5)


if __name__ == "__main__":
    {"start": cmd_start, "sampler": cmd_sampler, "sampler-start": cmd_sampler_start}[sys.argv[1]]()
