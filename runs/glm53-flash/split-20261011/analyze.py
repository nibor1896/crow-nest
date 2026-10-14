import json, os, statistics as st, sys
OUT = sys.argv[1]; arms = sys.argv[2].split(",")
K = ["enter", "write_backs", "regrown", "lazy_slots", "lend_write_backs", "refilled", "refill_nvme"]
def m(v): return 1000 * st.mean(v)
for a in arms:
    print(f"## {a}")
    for r in (1, 2):
        p = os.path.join(OUT, f"{a}-r{r}", "decode.json")
        if not os.path.exists(p):
            print(f"  r{r}: no json"); continue
        d = json.load(open(p)); prev = {k: 0 for k in K}
        for x in d["reps_detail"]:
            t = x["decode"]["timing"]; c = x["decode"]["counters"]; lat = t["latency_s"]; e = x["arena"]["elastic"]
            nv = [w["nvme_reads"] for w in x["rows"] if not w.get("prompt")]
            delta = {k: e[k] - prev[k] for k in K}; prev = {k: e[k] for k in K}
            print(f"  r{r}.{x['rep']}{'c' if x['cache']=='cold' else 'w'} P{x['prefill']['timing']['tokens']} ttft {x['prefill']['timing']['ttft_s']:.2f}s"
                  f" wall {t['wall_tok_s']:.2f} tok/s ({1000*t['wall_s']/t['tokens']:.1f} ms) med {t['tok_s_median_over_tokens']:.2f}"
                  f" | ms 1-16/17-48/49+ {m(lat[:16]):.0f}/{m(lat[16:48]):.0f}/{m(lat[48:]):.1f}"
                  f" | nvme/tok {c['nvme_demand_reads_per_token']:.1f} ({st.mean(nv[:16]):.0f}/{st.mean(nv[16:48]):.0f}/{st.mean(nv[48:]):.1f})"
                  f" | " + " ".join(f"{k}={v}" for k, v in delta.items()))
