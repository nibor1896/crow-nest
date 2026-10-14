"""#215 A/B (device sampler), per the implementation comment issuecomment-6108430619.

One serve boot per (prompt, binary). Binaries: `before` = 6944330 (worktree crow-nest-wt-accept-6944330, host draw),
`after` = local main 2a18fbc (device sampler). arm4.env, port 8099, nothing else on the GPU.
Prompts: P1 the #214 quick question (57 ids), max_tokens 512; P2 #213's real-text request 2
(docs/glm5-model.md[0:12000] at 298092b + question), max_tokens 512; P3 the G5 t1 request 1 body, max_tokens 1024.
Arms (seed 1408639702 for every sampled arm):
  G  temperature 0                         H  G + logit_bias {"0": 0} (host readback, before binary only)
  S  temperature 1, top_p 0.95, top_k 20, min_p 0.01, seed + logit_bias {"0": 0}  (host draw, after binary)
  D  the S body without the bias (device draw, after binary)
  F  G + crow_force_ids = S's ids (captured)
Captures (not timed): <arm>cap = arm + logprobs true + crow_force_ids [] -> crow_id per generated id.
Order before: Gcap, G H H G G H.  Order after: Gcap Scap Dcap, G S F D D F S G G S F D.
"""
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import bodies  # noqa: E402
import harness as h  # noqa: E402

OUT = os.path.join(h.HERE, "215")
os.makedirs(OUT, exist_ok=True)
ROWS = os.path.join(OUT, "rows.jsonl")
BEFORE_BIN = r"C:\Users\robin\dev\crow-nest-wt-accept-6944330\engine\target\release\serve.exe"
SEED = 1408639702
TXT = ("Explain in detail how a modern CPU cache hierarchy works: L1, L2, L3, coherence protocols, write-back versus "
       "write-through, and how false sharing hurts multithreaded code. Give concrete examples.")


def p2_text():
    doc = subprocess.run(["git", "-C", h.REPO, "show", "298092b:docs/glm5-model.md"], capture_output=True,
                         creationflags=h.NO_WINDOW).stdout.decode("utf-8", errors="replace")
    return doc[0:12000] + "\n\n---\n" + "Explain the layer order of a glm5_next layer as described here."


def base(p):
    common = {"stream": True, "stream_options": {"include_usage": True}, "timings_per_token": True}
    if p == "P1":
        return dict(common, messages=[{"role": "user", "content": TXT}], max_tokens=512)
    if p == "P2":
        return dict(common, messages=[{"role": "user", "content": p2_text()}], max_tokens=512)
    return bodies.t1(1, max_tokens=1024)


def arm(p, a, s_ids=None):
    b = base(p)
    for k in ("temperature", "top_p", "top_k", "min_p", "seed"):
        b.pop(k, None)
    kind = a.rstrip("cap")
    if kind in ("G", "H", "F"):
        b["temperature"] = 0
    else:
        b.update(temperature=1, top_p=0.95, top_k=20, min_p=0.01, seed=SEED)
    if kind in ("H", "S"):
        b["logit_bias"] = {"0": 0}
    if kind == "F":
        b["crow_force_ids"] = list(s_ids)
    if a.endswith("cap"):
        b["logprobs"] = True
        b["crow_force_ids"] = []
    return b


def run(p, which):
    log = os.path.join(OUT, f"serve-{p}-{which}.log")
    binary = BEFORE_BIN if which == "before" else h.SERVE
    order = (["Gcap", "G", "H", "H", "G", "G", "H"] if which == "before" else
             ["Gcap", "Scap", "Dcap", "G", "S", "F", "D", "D", "F", "S", "G", "G", "S", "F", "D"])
    caps = {}
    with h.Serve(log, serve_bin=binary) as sv:
        h.jdump(ROWS, {"event": "boot", "prompt": p, "binary": which, "boot_s": sv.boot_s, "boot_err": sv.boot_err,
                       "vram_before": sv.vram_before, "perf": sv.perf_before, "boot_lines": sv.boot_lines()})
        if sv.boot_err:
            return
        prev = None
        for i, a in enumerate(order):
            if a == "F" and "S" not in caps:
                h.jdump(ROWS, {"event": "skip", "prompt": p, "arm": a, "why": "no S ids captured"})
                continue
            res = sv.chat(arm(p, a, caps.get("S")))
            if a.endswith("cap") and res.get("ids"):
                caps[a[:-3]] = res["ids"]
            c = h.counters(res)
            s = h.summary(res)
            row = {"event": "req", "prompt": p, "binary": which, "i": i, "arm": a, **s, "counters": c,
                   "nvme_delta": (c[2] - prev[2]) if c and prev else None,
                   "nvme_per_token": round((c[2] - prev[2]) / s["gen"], 3) if c and prev and s["gen"] else None,
                   "ids": res.get("ids"), "text": res.get("reasoning", "") + "\x1f" + res.get("content", ""),
                   "tool_calls": res.get("tool_calls"), "log": res.get("log"), "routing": res.get("routing")}
            prev = c or prev
            h.jdump(ROWS, row)
            print(p, which, a, s, flush=True)
            if not res.get("serve_alive"):
                h.jdump(ROWS, {"event": "crash", "prompt": p, "binary": which, "arm": a})
                return
    h.jdump(ROWS, {"event": "stop", "prompt": p, "binary": which, "rc": sv.alive_at_exit})


if __name__ == "__main__":
    plan = sys.argv[1:] or ["P1:before", "P1:after", "P2:before", "P2:after", "P3:before", "P3:after"]
    for item in plan:
        p, which = item.split(":")
        run(p, which)
