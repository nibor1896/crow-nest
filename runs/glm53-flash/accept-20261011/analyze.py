"""Summaries of the 2026-10-11 block's rows (215/rows.jsonl, 219/, 217/, 216/, combined/)."""
import json
import math
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def rows(t):
    p = os.path.join(HERE, t, "rows.jsonl")
    return [json.loads(l) for l in open(p, encoding="utf-8")] if os.path.exists(p) else []


def med(xs):
    xs = [x for x in xs if x is not None]
    return round(st.median(xs), 3) if xs else None


def a215():
    R = [r for r in rows("215") if r["event"] == "req"]
    for p in ("P1", "P2", "P3"):
        b = {a: [r for r in R if r["prompt"] == p and r["binary"] == "before" and r["arm"] == a] for a in ("Gcap", "G", "H")}
        f = {a: [r for r in R if r["prompt"] == p and r["binary"] == "after" and r["arm"] == a]
             for a in ("Gcap", "Scap", "Dcap", "G", "S", "F", "D")}
        if not f["G"] and not b["G"]:
            continue
        m = {("before", a): med([r["ms_per_token"] for r in b[a]]) for a in ("G", "H")}
        m.update({("after", a): med([r["ms_per_token"] for r in f[a]]) for a in ("G", "S", "F", "D")})
        print(f"== {p}: prompt {next((r['P'] for r in R if r['prompt'] == p), None)}")
        for k, v in m.items():
            runs = (b if k[0] == "before" else f)[k[1]]
            print(f"  {k[0]:6} {k[1]}: median {v} ms/token; runs {[r['ms_per_token'] for r in runs]}; gen "
                  f"{[r['gen'] for r in runs]}; nvme/token {[r['nvme_per_token'] for r in runs]}; finish {[r['finish'] for r in runs]}")
        ids = {a: (x[0]["ids"] if x else None) for a, x in list(f.items()) + [("Gcap_before", b["Gcap"])]}
        def eq(a, c):
            if not ids.get(a) or not ids.get(c):
                return None
            n = next((i for i, (x, y) in enumerate(zip(ids[a], ids[c])) if x != y), None)
            return "equal (%d ids)" % len(ids[a]) if n is None and len(ids[a]) == len(ids[c]) else f"differ at {n} (len {len(ids[a])} vs {len(ids[c])})"
        print("  ids Dcap vs Scap:", eq("Dcap", "Scap"), "| Gcap after vs before:", eq("Gcap", "Gcap_before"))
        for a in ("S", "D", "G", "F"):
            texts = {r["text"] for r in f[a]}
            print(f"  texts {a}: {len(texts)} distinct over {len(f[a])} runs", end="")
            cap = f.get(a + "cap") or ([] if a != "F" else f["Scap"])
            if cap:
                print(f"; equal to {'Scap' if a == 'F' else a + 'cap'}: {[r['text'] == cap[0]['text'] for r in f[a]]}", end="")
            print()
        print("  G text before == after:", {r["text"] for r in b["G"]} == {r["text"] for r in f["G"]})
        D, F_, S, G, Gb, H = m[("after", "D")], m[("after", "F")], m[("after", "S")], m[("after", "G")], m[("before", "G")], m[("before", "H")]
        if None not in (D, F_, S, G, Gb, H):
            print(f"  D <= F x 1.01: {D} <= {round(F_ * 1.01, 3)} -> {D <= F_ * 1.01}; D < S: {D} < {S} -> {D < S}; "
                  f"S - D = {round(S - D, 3)}; step0 H - G (6944330) = {round(H - Gb, 3)}; G after/before = {round(G / Gb, 4)}")
        routes = {a: sorted({re.sub(r"seed \d+", "seed N", l)[:150] for r in f[a] for l in r["log"]
                             if "sampling on the" in l or "draws on the HOST" in l}) for a in ("S", "D")}
        print("  route lines:", json.dumps(routes)[:600])


def lp_cos(a, b):
    """cosine over the union of the two top-20 lists (aligned by crow_id/token, missing -> its own minimum)."""
    if not a or not b:
        return None, None
    ka = {(e.get("crow_id") if e.get("crow_id") is not None else e["token"]): e["logprob"] for e in a["top_logprobs"]}
    kb = {(e.get("crow_id") if e.get("crow_id") is not None else e["token"]): e["logprob"] for e in b["top_logprobs"]}
    keys = sorted(set(ka) & set(kb), key=str)
    va, vb = [ka[k] for k in keys], [kb[k] for k in keys]
    pa, pb = [math.exp(x) for x in va], [math.exp(x) for x in vb]
    dot = sum(x * y for x, y in zip(pa, pb))
    cos = dot / math.sqrt(sum(x * x for x in pa) * sum(y * y for y in pb))
    top = lambda d: max(d, key=d.get)
    return round(cos, 7), (top(ka) == top(kb), len(keys))


def reqtable(t):
    R = rows(t)
    for r in R:
        if r["event"] == "boot":
            print(f"-- {r['tag']} boot {r['boot_s']} s {r['boot_err'] or ''}")
            for l in r["boot_lines"]:
                if any(k in l for k in ("#217", "#219", "#216", "device sampler", "prefix cache")):
                    print("   ", l[:300])
        elif r["event"] == "req":
            print(f"   {r['req']:6} P {r['P']} C {r['C']} F {r['F']} gen {r['gen']} prefill {r['prefill_ms']} ms decode "
                  f"{r['decode_tok_s']} tok/s ms/tok {r['ms_per_token']} finish {r['finish']} err {r['err']}")
            for l in r["log"]:
                if l.startswith("[cache]") or l.startswith("[glm5] prompt phase"):
                    print("      ", l[:400])
        else:
            print("   ", r)


def a219():
    R = [r for r in rows("219") if r["event"] == "req"]
    r2 = {r["tag"]: r for r in R if r["req"] == "r2"}
    for r in R:
        ph = next((l for l in r["log"] if l.startswith("[glm5] prompt phase")), "")
        m = re.search(r"NVMe (\d+) reads", ph)
        st_ = re.search(r"staged calls (\d+)", ph)
        print(f"{r['tag']:7} {r['req']} P {r['P']} C {r['C']} F {r['F']} prefill {r['prefill_ms']} gen {r['gen']} "
              f"ms/tok {r['ms_per_token']} finish {r['finish']} | nvme {m.group(1) if m else None} staged {st_.group(1) if st_ else None}")
        for l in r["log"]:
            if l.startswith("[cache] ") and ("COLD" in l or "WARM" in l or "LIVE" in l or "realigned" in l):
                print("        ", l[:230])
    off = [r2[k] for k in sorted(r2) if k.startswith("off")]
    on = [r2[k] for k in sorted(r2) if k.startswith("on")]
    print("B3 median prefill off", med([r["prefill_ms"] for r in off]), "on", med([r["prefill_ms"] for r in on]))
    def same(a, b):
        ia, ib = a.get("ids") or [], b.get("ids") or []
        n = next((i for i, (x, y) in enumerate(zip(ia, ib)) if x != y), None)
        return {"finish": a["finish"] == b["finish"], "calls": a["tool_calls"] == b["tool_calls"],
                "content": a["content"] == b["content"], "first_diff_id": n, "lens": (len(ia), len(ib)),
                "lp": lp_cos(a.get("first_logprobs"), b.get("first_logprobs"))}
    for a in off:
        for b in on:
            print("B2", a["tag"], "vs", b["tag"], same(a, b))
    for x, y in [(off[0], off[1]), (on[0], on[1])] if len(off) > 1 and len(on) > 1 else []:
        print("within", x["tag"], y["tag"], same(x, y))
    for r in off + on:
        print(r["tag"], "calls", json.dumps(r["tool_calls"])[:200])


def acomb():
    R = [r for r in rows("combined") if r["event"] == "req"]
    g5 = [json.loads(l) for l in open(os.path.join(HERE, "..", "g5-20261011", "rows.jsonl"))][:9]
    names = ["t1r1", "t1r2", "t1r3", "t2r1", "t2r2", "t2r3", "t2r4", "t2r5", "t2r6"]
    arms = {"G5": [{"req": n, "P": g["P_prompt_n"], "C": g["C_cached_n"], "F": g["F_prefilled"], "prefill_ms": g["prefill_ms"],
                    "gen": g["predicted_n"], "pred_ms": g["predicted_ms"]} for n, g in zip(names, g5)]}
    for tag in ("all-off", "all-on"):
        arms[tag] = [{"req": r["req"], "P": r["P"], "C": r["C"], "F": r["F"], "prefill_ms": r["prefill_ms"], "gen": r["gen"],
                      "pred_ms": (r.get("routing") or {}).get("predicted_ms"),
                      "cache": r["cache"], "phase": next((l for l in r["log"] if l.startswith("[glm5] prompt phase")), "")}
                     for r in R if r["tag"] == tag]
    for tag, rs in arms.items():
        print("==", tag)
        for r in rs:
            ph = r.get("phase") or ""
            m = re.search(r"route (\w+)", ph); nv = re.search(r"NVMe (\d+) reads", ph); stg = re.search(r"staged calls (\d+)", ph)
            dec = round(r["gen"] / r["pred_ms"] * 1000, 2) if r.get("pred_ms") else None
            print(f"  {r['req']} P {r['P']} C {r['C']} F {r['F']} prefill {r['prefill_ms']:.1f} ms gen {r['gen']} decode {dec} tok/s"
                  f" | {m.group(1) if m else ''} nvme {nv.group(1) if nv else ''} staged {stg.group(1) if stg else ''} | {(r.get('cache') or '')[:90]}")
        for chat in ("t1", "t2"):
            c = [r for r in rs if r["req"].startswith(chat)]
            F = sum(r["F"] for r in c); pm = sum(r["prefill_ms"] for r in c); g = sum(r["gen"] for r in c); dm = sum(r["pred_ms"] or 0 for r in c)
            small = [r["prefill_ms"] for r in c if r["F"] < 1024]
            print(f"  {chat} turn: prefilled {F} in {pm/1000:.2f} s ({F/pm*1000:.1f} tok/s), small rounds prefill ms {small}, decode {g} tok in {dm/1000:.1f} s ({g/dm*1000:.2f} tok/s), model time {(pm+dm)/1000:.1f} s")


if __name__ == "__main__":
    t = sys.argv[1]
    if t == "215":
        a215()
    elif t == "comb":
        acomb()
    elif t == "219a":
        a219()
    else:
        reqtable(t)
