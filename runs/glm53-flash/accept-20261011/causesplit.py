"""Cause split of the G5 row-1 decode gap (robin, 2026-10-11): G5 t1 request 1 ran at 8.4 tok/s in the G5 session,
the same body at <= 1,024 tokens ran at 13.5-15.5 tok/s in #215's P3 rows.

serve = local main 2a18fbc, arm4.env, all new switches off, a fresh boot per run. Request = G5 t1 request 1 as Crow
sent it (bodies.t1(1): temperature 1, top_p 0.95, min_p 0.01, top_k from the card = 20, seed 1408639702, thinking
max), max_tokens 2,500. Arms alternated A B A B:
  A  nothing else running
  B  the G5 resource sampler running as in the session: a copy of runs/glm53-flash/g5-20261011/g5_session.py in the
     run's own directory, `python -I g5_session.py sampler` (CREATE_NO_WINDOW), which tails that directory's
     serve-tasks.log (the serve log of this run) and polls nvidia-smi + GetPerformanceInfo every 2 s.
Per run: decode tok/s overall; per-window rates over stream chunks 1-1,024 and 1,025-end (chunk arrival times);
generated count; NVMe reads per decode token = (counter nvme - the prompt phase's NVMe reads) / generated.
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402
import harness as h  # noqa: E402

OUT = os.path.join(h.HERE, "causesplit")
G5PY = os.path.join(h.REPO, "runs", "glm53-flash", "g5-20261011", "g5_session.py")


def stream(body):
    req = urllib.request.Request(h.URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.perf_counter()
    times, last = [], None
    with urllib.request.urlopen(req, timeout=7200) as r:
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
                if d.get("content") or d.get("reasoning_content") or d.get("tool_calls"):
                    times.append(time.perf_counter() - t0)
    return times, last


def run(tag, sampler):
    d = os.path.join(OUT, tag)
    os.makedirs(d, exist_ok=True)
    log = os.path.join(d, "serve-tasks.log")
    sp = None
    with h.Serve(log) as sv:
        meta = {"tag": tag, "sampler": sampler, "boot_s": sv.boot_s, "boot_err": sv.boot_err,
                "vram_before": sv.vram_before, "perf": sv.perf_before}
        if sv.boot_err:
            h.jdump(os.path.join(OUT, "rows.jsonl"), meta)
            return
        if sampler:
            shutil.copy(G5PY, os.path.join(d, "g5_session.py"))
            sp = subprocess.Popen([sys.executable, "-I", os.path.join(d, "g5_session.py"), "sampler"], cwd=d,
                                  stdout=open(os.path.join(d, "sampler.out"), "w"), stderr=subprocess.STDOUT,
                                  stdin=subprocess.DEVNULL, creationflags=h.NO_WINDOW)
            time.sleep(4)
        off = os.path.getsize(log)
        times, last = stream(bodies.t1(1, max_tokens=2500))
        time.sleep(1.0)
        with open(log, "rb") as fh:
            fh.seek(off)
            new = fh.read().decode("utf-8", errors="replace").splitlines()
        if sp:
            sp.kill()
            sp.wait(30)
    routing = next((json.loads(l) for l in new if l.startswith("{") and '"event":"routing"' in l), {})
    chat = next((l for l in new if l.startswith("[chat] prompt ")), "")
    phase = next((l for l in new if l.startswith("[glm5] prompt phase")), "")
    cnt = next((h.COUNTERS.search(l) for l in new if h.COUNTERS.search(l)), None)
    pn = routing.get("predicted_n") or 0
    pre_reads = int(re.search(r"NVMe (\d+) reads", phase).group(1)) if phase else None
    nv = int(cnt.group(3)) if cnt else None
    # chunk windows: the first chunk carries the first generated token; rate over a window = chunks / its time
    first = times[0] if times else None
    def rate(a, b):
        seg = times[a - 1:b]
        return round((len(seg) - 1) / (seg[-1] - seg[0]), 2) if len(seg) > 1 else None
    row = dict(meta, chunks=len(times), predicted_n=pn, predicted_ms=routing.get("predicted_ms"),
               decode_tok_s=routing.get("tok_s"), prompt_ms=routing.get("prompt_ms"), finish=routing.get("finish"),
               ttft_s=round(first, 3) if first else None,
               win_1_1024=rate(1, 1024), win_1025_end=rate(1025, len(times)),
               nvme_total=nv, nvme_prefill=pre_reads,
               nvme_per_decode_token=round((nv - pre_reads) / pn, 3) if nv is not None and pre_reads is not None and pn else None,
               chat_line=chat[:400], phase_line=phase[:500],
               sampler_rows=sum(1 for _ in open(os.path.join(d, "resources.jsonl"))) if sampler and os.path.exists(os.path.join(d, "resources.jsonl")) else None)
    h.jdump(os.path.join(OUT, "rows.jsonl"), row)
    json.dump(times, open(os.path.join(d, "chunk_times.json"), "w"))
    print(json.dumps({k: row[k] for k in ("tag", "predicted_n", "decode_tok_s", "win_1_1024", "win_1025_end",
                                          "nvme_per_decode_token", "finish", "sampler_rows")}), flush=True)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for tag, s in [("A-r1", False), ("B-r1", True), ("A-r2", False), ("B-r2", True)]:
        run(tag, s)
