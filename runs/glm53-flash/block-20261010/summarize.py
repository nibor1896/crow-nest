"""Summarize the clean block of 2026-10-10: per arm both passes; cold = rep 1, warm = reps 2-3.
Decode is reported as wall tok/s (tokens / decode wall) because MTP emits several tokens per step."""
import json
import pathlib
import statistics

D = pathlib.Path(__file__).parent
ARMS = ["base", "graph", "flags", "stager", "mtp1", "mtp2", "all_mtp1", "all_mtp2"]
PRE = ["pre_chunk1", "pre_chunk32", "pre121_chunk1", "pre121_chunk32"]


def load(p, name):
    f = D / f"p{p}-{name}.json"
    return json.loads(f.read_text()) if f.exists() else None


def rc(p, name):
    f = D / f"p{p}-{name}.log"
    if not f.exists():
        return "missing"
    for line in f.read_text(errors="replace").splitlines():
        if line.startswith("rc="):
            return line.split()[0][3:]
    return "?"


def ref_ids(name):
    for p in (1, 2):
        d = load(p, name)
        if d:
            return d["reps_detail"][0]["ids"]
    return None


med = lambda xs: statistics.median(xs) if xs else float("nan")
spread = lambda xs: (max(xs) / min(xs)) if xs and min(xs) > 0 else float("nan")


def table(arms, ref):
    base_ids, rows = ref_ids(ref), []
    for name in arms:
        cold, warm, cold_ttft, warm_ttft, pre_tok, acc, tps, h2d, zc, same, rcs = [], [], [], [], [], [], [], [], [], [], []
        for p in (1, 2):
            rcs.append(rc(p, name))
            d = load(p, name)
            if not d:
                continue
            reps = d["reps_detail"]
            for i, r in enumerate(reps):
                same.append(r["ids"] == base_ids)
                t = r["decode"]["timing"]
                (cold if i == 0 else warm).append(t["wall_tok_s"])
                (cold_ttft if i == 0 else warm_ttft).append(r["prefill"]["timing"]["ttft_s"])
                pre_tok.append(r["prefill"]["timing"]["wall_tok_s"])
                m = r["decode"].get("mtp")
                if m:
                    acc.append(m["acceptance_rate"] or 0.0)
                    tps.append(m["tokens_per_step"])
                if i > 0:
                    k = r["decode"]["counters"]
                    h2d.append(k["h2d_gb_per_token"])
                    zc.append(k.get("zero_copy_gb_per_token", float("nan")))
        rows.append((name, ",".join(rcs), all(same) if same else None, med(cold), med(warm), spread(warm),
                     med(cold_ttft), med(warm_ttft), med(pre_tok), med(acc), med(tps), med(h2d), med(zc)))
    return rows


hdr = "arm rc ids_same cold_dec warm_dec warm_spread cold_ttft warm_ttft prefill_tok_s mtp_accept tok_per_step h2d zc"
out = {}
for title, arms, ref in (("decode", ARMS, "base"), ("prefill", PRE, "pre_chunk1")):
    print(f"## {title} (ids vs {ref})\n{hdr}")
    rows = table(arms, ref)
    for r in rows:
        print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
    out[title] = [dict(zip(hdr.split(), r)) for r in rows]
json.dump(out, open(D / "summary.json", "w"), indent=1)
