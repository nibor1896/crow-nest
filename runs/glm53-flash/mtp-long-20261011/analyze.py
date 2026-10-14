"""#192 follow-up 2026-10-11: per-arm decode figures of run.sh's A/B/C (glm5_run JSON + stripped row log).
Windows by decode token index (the ids after the first, which comes from the last prompt row):
1-256, 257-1024, 1025-end. tok/s = tokens / (at_s of the window's last row - at_s of the row before the window).
Tier hits from the log's `tiers v/p/n` (distinct experts per call over the 42 MoE layers; under MTP on a step's first id).
Usage: python -I analyze.py <dir> [labels...]  -> prints and writes <dir>/summary.json
"""
import json, os, re, sys

D = sys.argv[1]
LABELS = sys.argv[2:] or ["A", "B", "C"]
ROW = re.compile(r"glm5_run row\s+(\d+) (prompt|gen) .*?NVMe reads\s+(\d+) .*tiers v/p/n (\d+)/(\d+)/(\d+)")
WIN = [(1, 256), (257, 1024), (1025, None)]


def arm(label):
    j = json.load(open(os.path.join(D, label + ".json")))
    r = j["reps_detail"][0]
    rows = r["rows"]
    pre = [x for x in rows if x["prompt"]]
    dec = [x for x in rows if not x["prompt"]]
    tiers = [tuple(map(int, m.groups()[3:])) for m in (ROW.search(l) for l in open(os.path.join(D, label + ".log"), encoding="utf-8", errors="replace")) if m and m.group(2) == "gen"]
    assert len(tiers) == len(dec), (label, len(tiers), len(dec))
    layers = j["tiers"]["moe_layers"]
    t0 = pre[-1]["at_s"]
    out = {"commit": j["commit"], "vram_slots": j["tiers"]["vram_slots"], "pinned_slots": j["tiers"]["pinned_slots"],
           "pinned_use": j["tiers"]["pinned_use"], "prompt_ids": j["prompt"]["ids"], "decode_tokens": len(dec),
           "ttft_s": r["prefill"]["timing"].get("ttft_s", t0), "summary": j["summary"]}
    wins = []
    for a, b in WIN:
        if a > len(dec):
            continue
        b = min(b or len(dec), len(dec))
        seg = dec[a - 1:b]
        tseg = tiers[a - 1:b]
        start = t0 if a == 1 else dec[a - 2]["at_s"]
        n = len(seg)
        steps = [t for t in tseg if sum(t[:3]) > 0]
        w = {"tokens": f"{a}-{b}", "tok_s": n / (seg[-1]["at_s"] - start),
             "nvme_reads_per_tok": sum(x["nvme_reads"] for x in seg) / n,
             "vram_hits_per_tok": sum(t[0] for t in tseg) / n, "pinned_hits_per_tok": sum(t[1] for t in tseg) / n,
             "nvme_hits_per_tok": sum(t[2] for t in tseg) / n,
             "calls": len(steps), "tok_per_call": n / max(1, len(steps)),
             "experts_per_call_per_layer": sum(sum(t) for t in steps) / max(1, len(steps)) / layers}
        wins.append(w)
    alln = len(dec)
    steps = [t for t in tiers if sum(t) > 0]
    out["overall"] = {"tok_s": alln / (dec[-1]["at_s"] - t0), "nvme_reads_per_tok": sum(x["nvme_reads"] for x in dec) / alln,
                      "vram_hits_per_tok": sum(t[0] for t in tiers) / alln, "pinned_hits_per_tok": sum(t[1] for t in tiers) / alln,
                      "nvme_hits_per_tok": sum(t[2] for t in tiers) / alln, "calls": len(steps),
                      "experts_per_call_per_layer": sum(sum(t) for t in steps) / max(1, len(steps)) / layers}
    out["windows"] = wins
    out["counters"] = r["decode"]["counters"]
    out["mtp"] = r["decode"].get("mtp")
    out["ids"] = r["ids"]
    return out


res = {l: arm(l) for l in LABELS if os.path.exists(os.path.join(D, l + ".json"))}
for l, a in res.items():
    o = a["overall"]
    print(f"{l}: vram_slots {a['vram_slots']} pinned {a['pinned_slots']} {a['pinned_use']} ttft {a['ttft_s']:.2f} s, decode {a['decode_tokens']} tok "
          f"{o['tok_s']:.2f} tok/s, NVMe {o['nvme_reads_per_tok']:.2f}/tok, hits v/p/n {o['vram_hits_per_tok']:.1f}/{o['pinned_hits_per_tok']:.1f}/{o['nvme_hits_per_tok']:.1f} per tok, "
          f"experts/call/layer {o['experts_per_call_per_layer']:.2f}, median {a['summary'].get('decode_tok_s_median')}, wall {a['summary'].get('decode_wall_tok_s')}")
    for w in a["windows"]:
        print(f"   {w['tokens']:>10}: {w['tok_s']:.2f} tok/s, NVMe {w['nvme_reads_per_tok']:.2f}/tok, v/p/n {w['vram_hits_per_tok']:.1f}/{w['pinned_hits_per_tok']:.1f}/{w['nvme_hits_per_tok']:.1f} per tok, "
              f"tok/call {w['tok_per_call']:.2f}, experts/call/layer {w['experts_per_call_per_layer']:.2f}")
    if a["mtp"]:
        m = a["mtp"]
        print(f"   MTP: steps {m['steps']} tokens {m['tokens']} acceptance {m['acceptance_rate']:.4f} tok/step {m['tokens_per_step']:.3f} hist {m['accepted_hist']}")
ids = {l: a["ids"] for l, a in res.items()}


def same(x, y):
    a, b = ids.get(x), ids.get(y)
    if a is None or b is None:
        return None
    k = next((i for i, (p, q) in enumerate(zip(a, b)) if p != q), None)
    return {"identical": a == b, "len": [len(a), len(b)], "first_diff": k}


cmp = {"B_vs_C": same("B", "C"), "A_vs_B": same("A", "B"), "A_vs_C": same("A", "C")}
print("ids:", cmp)
for a in res.values():
    a.pop("ids")
json.dump({"arms": res, "ids": cmp}, open(os.path.join(D, "summary.json"), "w"), indent=1)
