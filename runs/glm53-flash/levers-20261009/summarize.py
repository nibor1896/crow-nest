"""Summarize the lever block of 2026-10-09: per arm, both passes; cold = rep 1, warm = reps 2-3."""
import json
import pathlib
import statistics
import sys

D = pathlib.Path(__file__).parent
ARMS = ["base", "ioring2", "flags", "lookahead", "zerocopy", "host", "host_zerocopy", "host_cpulane", "all_zerocopy", "all_cpulane"]


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


base_ids = None
for p in (1, 2):
    b = load(p, "base")
    if b:
        base_ids = b["reps_detail"][0]["ids"]
        break

rows = []
for name in ARMS:
    cold_dec, warm_dec, cold_ttft, warm_ttft, warm_r, warm_h2d, warm_d2h, warm_zc, lane, same = [], [], [], [], [], [], [], [], [], []
    rcs = []
    for p in (1, 2):
        rcs.append(rc(p, name))
        d = load(p, name)
        if not d:
            continue
        reps = d["reps_detail"]
        for r in reps:
            same.append(r["ids"] == base_ids)
        c = reps[0]
        cold_dec.append(c["decode"]["timing"]["tok_s_median_over_tokens"])
        cold_ttft.append(c["prefill"]["timing"]["ttft_s"])
        for w in reps[1:]:
            warm_dec.append(w["decode"]["timing"]["tok_s_median_over_tokens"])
            warm_ttft.append(w["prefill"]["timing"]["ttft_s"])
            k = w["decode"]["counters"]
            warm_r.append(k["r_nvme_reads_per_token"])
            warm_h2d.append(k["h2d_gb_per_token"])
            warm_d2h.append(k.get("d2h_gb_per_token", float("nan")))
            warm_zc.append(k.get("zero_copy_gb_per_token", float("nan")))
            lane.append(k.get("cpu_lane_per_token", 0.0))
    med = lambda xs: statistics.median(xs) if xs else float("nan")
    spread = lambda xs: (max(xs) / min(xs)) if xs and min(xs) > 0 else float("nan")
    rows.append((name, ",".join(rcs), all(same) if same else None, med(cold_dec), med(warm_dec), spread(warm_dec), min(warm_dec) if warm_dec else float("nan"), max(warm_dec) if warm_dec else float("nan"),
                 med(cold_ttft), med(warm_ttft), med(warm_r), med(warm_h2d), med(warm_d2h), med(warm_zc), med(lane)))

hdr = "arm rc ids_same cold_dec warm_dec warm_spread warm_min warm_max cold_ttft warm_ttft warm_r h2d d2h zc lane"
print(hdr)
for r in rows:
    print(" ".join(f"{x:.3f}" if isinstance(x, float) else str(x) for x in r))
json.dump([dict(zip(hdr.split(), r)) for r in rows], open(D / "summary.json", "w"), indent=1)
