"""Round-3 additions (profile-20261010l-decode, glm-integration 3de5eff, ARM2 + RT2 + the 12 switches of quick/b5-all12).

Re-run:  python -I analysis3.py [dir] [rows.log]
  rows.log defaults to decode-stdout.log: the profiled run's own per-token rows, taken from the nsys report
  (sqlite ProcessStreams -> StringIds; nsys keeps the target's stdout there, the shell redirect stays empty).
  Runs analysis2.py (which runs analysis.py) first, then:
  1. host gaps re-attributed by the kernel that waited: with CROW_GLM_RT2_GATHER_EARLY the gather is queued
     before the publish, so the host's flag -> plan -> stager API stretch ends at glm5_moe_tables_in, which
     analysis.py files under "launch-bound";
  2. an additive critical path per MoE layer on s7 with the new marks (router, topk -> early pass,
     early pass, early end -> late start = stager event, late pass, tail = lane + combine);
  3. per layer, the stager batch the host queued for it (API calls between the layer's topk launch and its
     tables_in launch): cuStreamWaitValue64 (NVMe reads not landed: demand + in-flight guesses), 9.47 MB
     H2D (pinned/landing -> stage), 9.47 MB D2D (stage -> VRAM), 9.47 MB D2H (write-back);
     the stager-event wait split by what the batch held, and a span regression on those counts;
  4. NVMe read latency (wait-value issue -> landed) against the drive's QD1 service time.
"""
import bisect
import os
import sys
from collections import defaultdict

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
ROWS = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "decode-stdout.log")
sys.argv = [sys.argv[0], D, ROWS]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis2.py")).read())
print()
print("=" * 100)
REC = 9474048

# ---------------------------------------------------------------- 1. host gaps by the waiting kernel
host_by = defaultdict(float)
for g0, g1 in gaps:
    i = bisect.bisect_left(starts, g1)
    if i >= len(start_index):
        continue
    j = start_index[i][1]
    api = api_by_cid.get(j[7])
    H = api[1] if api else g0
    host = max(0, min(H, g1) - g0)
    if host <= 0:
        continue
    if j[7] in comb_cids:
        name = "CPU lane (combine waits for the lane rows)"
    elif j[3] == "glm5_moe_tables_in" or j[7] in gather_cids:
        name = "routing step (flag -> plan -> stager API -> tables_in/early pass launch)"
    elif j[7] in first_of_token:
        name = "token turnaround (argmax readback -> next launch)"
    else:
        name = f"launch-bound before {j[3]}"
    host_by[name] += host
print("-- host-bound GPU idle by the kernel that waited (ms/token):")
for k_, v in sorted(host_by.items(), key=lambda x: -x[1])[:12]:
    print(f"  {k_:80s} {ms_(v):6.2f}")
print(f"  {'sum':80s} {ms_(sum(host_by.values())):6.2f}")

# ---------------------------------------------------------------- 2. per-layer marks and critical path
Rc3 = sorted((cid, s, e, n) for s, e, cid, n, _ in R)
R3cid = [x[0] for x in Rc3]
Mbycid = {m[5]: m for m in M}
lay3 = []
prev_end = A
for idx, (l, x) in enumerate(zip(moe_layers, L)):
    seg = l["seg"]
    rt = next((k for k in seg if k[3] == "gemv_bf16_b"), None)
    tk = next((k for k in seg if k[3] == "glm5_router_sig_topk"), None)
    ti = next((k for k in seg if k[3] == "glm5_moe_tables_in"), None)
    g, cb = l["gather"], l["combine"]
    early, late = l["early"], l["late"]
    if not (rt and tk and g and cb and early):
        continue
    es, ee = min(k[0] for k in early), max(k[1] for k in early)
    ls = min((k[0] for k in late), default=None)
    le = max((k[1] for k in late), default=None)
    # the host's stager batch for this layer: API calls between the topk launch and the tables_in launch
    lo = bisect.bisect_right(R3cid, tk[7])
    hi = bisect.bisect_left(R3cid, (ti or early[0])[7])
    wv = h2d = d2d = d2h = 0
    wv_issue = []
    for cid, s, e, n in Rc3[lo:hi]:
        if n == "cuStreamWaitValue64_v2":
            wv += 1
            wv_issue.append((cid, s))
        m = Mbycid.get(cid)
        if m and m[4] == REC:
            if m[3] == 1:
                h2d += 1
            elif m[3] == 8:
                d2d += 1
            elif m[3] == 2:
                d2h += 1
    lane = x["lane"]
    lay3.append(dict(idx=idx, t=x["t"], n=x["n"], v=x["v"], p=x["p"], lane_n=x["lane_n"], zc=x["zc"], E=x["E"], Lt=x["Lt"],
                     pre=rt[0] - prev_end, router=tk[1] - rt[0], to_early=es - tk[1], early=ee - es,
                     wait=(ls - ee) if ls else 0, late=(le - ls) if ls else 0, tail=cb[1] - (le if le else ee),
                     lane_after=max(0, (lane[1] if lane else 0) - (le if le else ee)), g0=g[0], es=es, ee=ee, ls=ls, le=le, ce=cb[1],
                     wv=wv, h2d=h2d, d2d=d2d, d2h=d2h, wv_issue=wv_issue))
    prev_end = cb[1]
NL = len(lay3)
print(f"\n-- critical path per token on s7, additive (ms/token), {NL / NT:.1f} MoE layers/token:")
parts = [("pre", "attention, mHC, dense, head, turnaround (previous combine end -> router start)"),
         ("router", "router (router start -> topk end)"),
         ("to_early", "topk end -> early pass start (gather, publish, shared expert; host flag -> plan -> stager API)"),
         ("early", "early pass (VRAM + zero-copy experts, under the lane)"),
         ("wait", "early end -> late start (stager event: NVMe landing, promotion copies)"),
         ("late", "late pass (landed / staged / written-back experts)"),
         ("tail", "late end -> combine end (lane rows, combine)")]
tot = 0
for key, name in parts:
    v = sum(y[key] for y in lay3)
    tot += v
    print(f"  {name:100s} {ms_(v):6.2f}")
print(f"  {'tail of the window (last combine end -> window end)':100s} {ms_(B - prev_end):6.2f}")
print(f"  {'sum':100s} {ms_(tot + B - prev_end):6.2f}  (window {ms_(B - A):.2f})")
print(f"  lane end after the late end: {ms_(sum(y['lane_after'] for y in lay3)):.2f} ms/token of the tail")

# ---------------------------------------------------------------- 3. stager batches
print("\n-- stager batch per layer (host API between topk launch and tables_in launch), per token:")
print(f"  wait values {sum(y['wv'] for y in lay3) / NT:.1f}, 9.47 MB H2D {sum(y['h2d'] for y in lay3) / NT:.1f}, D2D {sum(y['d2d'] for y in lay3) / NT:.1f}, D2H {sum(y['d2h'] for y in lay3) / NT:.1f}; demand reads (rows) {sum(y['n'] for y in lay3) / NT:.1f}")
cls = defaultdict(list)
for y in lay3:
    if y["n"] > 0:
        c_ = "a demand read (n >= 1)"
    elif y["wv"] > 0:
        c_ = "no demand read, waits on a guessed read in flight (join)"
    elif y["h2d"] > 0:
        c_ = "no NVMe wait, promotion H2D in the batch"
    elif y["d2h"] > 0 or y["d2d"] > 0:
        c_ = "no NVMe wait, no H2D, write-back / D2D only"
    else:
        c_ = "nothing staged (table only)"
    cls[c_].append(y)
print("  class                                                         layers/tok  wait ms/tok  late ms/tok  span us  wait us  late slots")
for c_, ys in sorted(cls.items(), key=lambda z: -sum(y["wait"] for y in z[1])):
    print(f"  {c_:62s} {len(ys) / NT:8.2f}  {ms_(sum(y['wait'] for y in ys)):10.2f}  {ms_(sum(y['late'] for y in ys)):10.2f}  {np.mean([y['ce'] - y['g0'] for y in ys]) / 1e3:7.0f}  {np.mean([y['wait'] for y in ys]) / 1e3:7.0f}  {np.mean([y['Lt'] for y in ys]):6.2f}")
X = np.array([[1, y["n"], max(0, y["wv"] - y["n"]), y["h2d"], y["d2h"], y["zc"], y["lane_n"], y["v"]] for y in lay3], float)
ysp = np.array([y["ce"] - y["g0"] for y in lay3], float) / 1e3
co = np.linalg.lstsq(X, ysp, rcond=None)[0]
r2 = 1 - ((ysp - X @ co) ** 2).sum() / ((ysp - ysp.mean()) ** 2).sum()
print(f"  span (gather start -> combine end) = {co[0]:.0f} + {co[1]:.0f} x demand + {co[2]:.0f} x joined guess (waits beyond demand) + {co[3]:.0f} x H2D + {co[4]:.0f} x D2H"
      f" + {co[5]:.0f} x zero-copy + {co[6]:.0f} x lane + {co[7]:.0f} x VRAM (us), R2 {r2:.3f}")
Xs = np.array([[1, y["n"], max(0, y["wv"] - y["n"])] for y in lay3], float)
cs = np.linalg.lstsq(Xs, ysp, rcond=None)[0]
print(f"  short form: span = {cs[0]:.0f} + {cs[1]:.0f} x demand + {cs[2]:.0f} x joined guess (us)  (round k: 597 + 1,447 x demand + 1,015 x join)")
mean_ = lambda key: np.mean([max(0, y["wv"] - y["n"]) if key == "j" else y[key] for y in lay3]) * 42
print(f"  per token: demand {mean_('n'):.1f}, joined waits {mean_('j'):.1f}, H2D {mean_('h2d'):.1f}, D2H {mean_('d2h'):.1f} -> base {cs[0] * 42 / 1e3:.2f} + demand {cs[1] * mean_('n') / 1e3:.2f} + joins {cs[2] * mean_('j') / 1e3:.2f} ms")

# ---------------------------------------------------------------- 4. NVMe read latency on the path
print("\n-- NVMe: per wait value, host issue -> the next s16 op start (landed), by layer class:")
s16_ops2 = sorted([(m[5], m[0]) for m in M if m[2] == 16])
s16c2 = [x[0] for x in s16_ops2]
lat_d, lat_j = [], []
for y in lay3:
    for cid, s in y["wv_issue"]:
        jj = bisect.bisect_right(s16c2, cid)
        if jj < len(s16_ops2):
            (lat_d if y["n"] > 0 else lat_j).append((s16_ops2[jj][1] - s) / 1e3)
for name, a_ in (("layers with a demand read", lat_d), ("join-only layers", lat_j)):
    a_ = np.array(a_)
    if len(a_):
        print(f"  {name:28s} {len(a_) / NT:5.1f}/token: p10 {np.percentile(a_, 10):5.0f} p50 {np.percentile(a_, 50):5.0f} p90 {np.percentile(a_, 90):5.0f} mean {a_.mean():5.0f} us"
              f"  (one 9.47 MB record at the drive's 8.14 GB/s QD1: 1,164 us)")
dj3 = json.load(open(os.path.join(D, "decode.json")))["reps_detail"][0]["decode"]["counters"]
recs = dj3["nvme_records_per_token"] - dj3["prefetch"].get("dropped_per_token", 0)  # dropped guesses never reach the drive
print(f"  drive: {recs:.1f} records read/token (issued minus dropped) x 1.164 ms = {recs * 1.164:.1f} ms busy per {ms_(B - A):.1f} ms token ({recs * 1.164 / ms_(B - A) * 100:.0f} %); "
      f"demand {dj3['nvme_demand_reads_per_token']:.1f}, guesses {dj3['nvme_speculative_reads_per_token']:.1f} (used {dj3['prefetch']['used_per_token']:.1f}, wasted {dj3['prefetch']['wasted_per_token']:.1f}, dropped before the drive {dj3['prefetch'].get('dropped_per_token', 0):.1f}), joins {dj3['prefetch']['joins_per_token']:.1f}")
print(f"  demand overtakes {dj3['prefetch'].get('demand_overtakes_per_token', 0):.3f}/token, promoted {dj3['prefetch'].get('promoted_per_token', 0):.2f}/token")

# ---------------------------------------------------------------- 5. what a layer costs by its count of demand reads, decomposed
print("\n-- per layer by demand reads n: mean us of each critical-path part (pre | router | topk->early | early | wait | late | tail)")
for nn in range(0, 4):
    ys = [y for y in lay3 if y["n"] == nn]
    if not ys:
        continue
    f = lambda k_: np.mean([y[k_] for y in ys]) / 1e3
    print(f"  n={nn}: {len(ys) / NT:5.2f}/token  {f('pre'):5.0f} | {f('router'):4.0f} | {f('to_early'):4.0f} | {f('early'):5.0f} | {f('wait'):5.0f} | {f('late'):4.0f} | {f('tail'):4.0f}"
          f"   late slots {np.mean([y['Lt'] for y in ys]):.2f}, waits {np.mean([y['wv'] for y in ys]):.2f}, H2D {np.mean([y['h2d'] for y in ys]):.2f}")

# ---------------------------------------------------------------- 6. inside the parts: GPU busy / idle and what the idle waited on
print("\n-- GPU busy and idle inside each critical-path part (ms/token); idle by the analysis.py class of its gap")
gap_cls = []  # (g0, g1, class) re-derived with the same rules, host part and device part separately
for g0, g1 in gaps:
    i = bisect.bisect_left(starts, g1)
    if i >= len(start_index):
        continue
    j = start_index[i][1]
    api = api_by_cid.get(j[7])
    H = api[1] if api else g0
    hp = min(H, g1)
    if hp > g0:
        gap_cls.append((g0, hp, "host"))
    if g1 > max(g0, hp):
        w = False
        if j[2] == 7:
            pc = PREV7.get(j[7], j[7] - 400)
            lo_ = bisect.bisect_left(waits7_cid, pc)
            for cid, eid in waits7[lo_:]:
                if cid >= j[7]:
                    break
                if record_stream(eid, cid) == 16:
                    w = True
        gap_cls.append((max(g0, hp), g1, "stager event" if w else ("s13 event / queue" if j[2] == 13 else "queued/launch latency")))
gap_cls.sort()
G0 = [x[0] for x in gap_cls]


def idle_in(a, b):
    out = defaultdict(float)
    i = max(0, bisect.bisect_left(G0, a) - 1)
    while i < len(gap_cls) and gap_cls[i][0] < b:
        s, e, c_ = gap_cls[i]
        ov = min(e, b) - max(s, a)
        if ov > 0:
            out[c_] += ov
        i += 1
    return out


bounds = {"pre": lambda y, prv: (prv, y["g0"] - (y["g0"] - y["es"]) - (y["es"] - y["g0"]))}
segs_ = defaultdict(lambda: defaultdict(float))
prv = A
for y in lay3:
    rt0 = y["es"] - y["to_early"] - y["router"]
    marks = [("pre", prv, rt0), ("router", rt0, rt0 + y["router"]), ("to_early", rt0 + y["router"], y["es"]), ("early", y["es"], y["ee"])]
    if y["ls"]:
        marks += [("wait", y["ee"], y["ls"]), ("late", y["ls"], y["le"]), ("tail", y["le"], y["ce"])]
    else:
        marks += [("tail", y["ee"], y["ce"])]
    for key, a, b in marks:
        segs_[key]["span"] += b - a
        for c_, v in idle_in(a, b).items():
            segs_[key][c_] += v
    prv = y["ce"]
for key, name in parts:
    s_ = segs_[key]
    idl = sum(v for c_, v in s_.items() if c_ != "span")
    det = ", ".join(f"{c_} {ms_(v):.2f}" for c_, v in sorted(s_.items(), key=lambda z: -z[1]) if c_ != "span" and v > 0)
    print(f"  {key:9s} span {ms_(s_['span']):6.2f} = busy {ms_(s_['span'] - idl):6.2f} + idle {ms_(idl):5.2f} ({det})")

# ---------------------------------------------------------------- 7. queued-not-started s7 kernels by name
print("\n-- s7 kernels launched before the GPU ran dry that still started >= 5 us after it ran dry, by kernel (top 12):")
qb = defaultdict(lambda: [0.0, 0])
for dur, j, H, g0, g1 in LQ:
    if dur < 5000 or j[2] != 7:
        continue
    qb[j[3]][0] += dur
    qb[j[3]][1] += 1
for k_, (v, n) in sorted(qb.items(), key=lambda z: -z[1][0])[:12]:
    print(f"  {k_:28s} {ms_(v):5.2f} ms/token  {n / NT:6.1f}/token  {v / n / 1e3:5.1f} us each")

# ---------------------------------------------------------------- 8. the host's routing step, per layer
print("\n-- host routing step per layer (us, mean / p50): GPU topk end -> host's first API after it -> first stager API -> tables_in launch end -> tables_in GPU start")
STG = {"cuStreamWaitValue64_v2", "cuStreamWriteValue64_v2", "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyAsync", "cuEventRecord", "cuStreamWaitEvent", "cuEventQuery"}
Rs = sorted((s, e, cid, n) for s, e, cid, n, _ in R)
Rs0 = [x[0] for x in Rs]
rows8 = []
for y in lay3:
    l = moe_layers[y["idx"]]
    seg = l["seg"]
    tk = next((k for k in seg if k[3] == "glm5_router_sig_topk"), None)
    ti = next((k for k in seg if k[3] == "glm5_moe_tables_in"), None)
    pb = next((k for k in seg if k[3] in ("glm5_publish", "glm5_publish_pred")), None)
    if not (tk and ti and pb):
        continue
    api_ti = api_by_cid.get(ti[7])
    api_pb = api_by_cid.get(pb[7])
    if not api_ti or not api_pb:
        continue
    # host calls after the publish-side launches, before tables_in (the stager batch and the host's plan)
    lo_ = bisect.bisect_right(R3cid, pb[7])
    hi_ = bisect.bisect_left(R3cid, ti[7])
    calls = [x for x in Rc3[lo_:hi_]]
    first = next((x for x in calls if x[1] >= tk[1]), None)
    if first is None:
        continue
    stg = [x for x in calls if x[3] in STG]
    s_first = stg[0][1] if stg else api_ti[0]
    rows8.append((first[1] - tk[1], s_first - first[1], api_ti[1] - s_first, ti[0] - api_ti[1], sum(x[2] - x[1] for x in stg), len(stg), ti[0] - tk[1]))
a8 = np.array(rows8, float) / 1e3
names8 = ["topk end -> first host API after it (flag seen + plan)", "-> first stager API", "first stager API -> tables_in launch end", "-> tables_in GPU start", "stager API time inside", "stager API calls (count)", "topk end -> tables_in start (total)"]
for i_, n_ in enumerate(names8):
    col = a8[:, i_] * (1e3 if i_ == 5 else 1)
    print(f"  {n_:60s} {col.mean():7.1f} / {np.median(col):7.1f}")

# ---------------------------------------------------------------- 9. per-layer what-if on the measured marks (not a sum of category totals)
# A layer's MoE part ends at ce = max(late end, lane end) + c, with late start = max(early end, F) + d, F = the stager
# event's fire time (end of the last s16 op queued before the batch's cuEventRecord on s16). A lever moves one mark;
# the others stay; the saving is old ce - new ce, so a shift that a longer chain (NVMe landing, the lane) absorbs counts 0.
EVs = {cid: st for st, cid, eid in EV}
s16_ops3 = sorted([(m[5], m[1], m[0]) for m in M if m[2] == 16])
s16c3 = [x[0] for x in s16_ops3]
SIM = []
for y in lay3:
    l = moe_layers[y["idx"]]
    seg = l["seg"]
    tk = next((k for k in seg if k[3] == "glm5_router_sig_topk"), None)
    ti = next((k for k in seg if k[3] == "glm5_moe_tables_in"), None)
    if not (tk and ti):
        continue
    lo_ = bisect.bisect_right(R3cid, tk[7])
    hi_ = bisect.bisect_left(R3cid, ti[7])
    batch = Rc3[lo_:hi_]
    rec = [x for x in batch if x[3] == "cuEventRecord" and EVs.get(x[0]) == 16]
    if y["ls"] and rec:
        j_ = bisect.bisect_left(s16c3, rec[-1][0]) - 1
        F = s16_ops3[j_][1] if j_ >= 0 else y["ee"]
    elif y["ls"]:
        continue
    else:
        F = None
    stg = [x for x in batch if x[3] in STG]
    plan_end = stg[0][1] if stg else ti[0]
    # landing of each wait value: the next s16 op start
    lands = []
    for cid, s, e, n in batch:
        if n == "cuStreamWaitValue64_v2":
            jj = bisect.bisect_right(s16c2, cid)
            if jj < len(s16_ops2):
                lands.append((s16_ops2[jj][1], s16_ops2[jj][1] - s))
    SIM.append(dict(y=y, F=F, plan_end=plan_end, ti0=ti[0], tk1=tk[1], lands=lands))
# lane end per layer from analysis2's join
shared_end = {}
for y in lay3:
    l = moe_layers[y["idx"]]
    sh = [k for k in l["seg"] if cat.get((k[0], k[7])) in ("shared expert",) or k[3] in ("glm5_publish", "glm5_moe_gather")]
    shared_end[id(y)] = max((k[1] for k in sh), default=y["g0"])
# exact lane end: match by gather start
lane_by_g = {x["g0"]: (x["lane"][1] if x["lane"] else None) for x in L}
for s in SIM:
    s["lane_end"] = lane_by_g.get(s["y"]["g0"])


def ce_of(y, ee, F, late_dur, lane_end, d, c):
    if F is None:  # no late pass
        return max(ee, lane_end or 0) + c
    ls = max(ee, F) + d
    le = ls + late_dur
    return max(le, lane_end or 0) + c


def run(shift_early=None, F_new=None):
    tot_old = tot_new = 0
    for s in SIM:
        y = s["y"]
        d = max(0, y["ls"] - max(y["ee"], s["F"])) if s["F"] is not None else 0
        c = y["ce"] - max(y["le"] if y["le"] else y["ee"], s["lane_end"] or 0)
        c = max(0, c)
        old = ce_of(y, y["ee"], s["F"], y["late"], s["lane_end"], d, c)
        sh = shift_early(s) if shift_early else 0
        F2 = F_new(s) if F_new else s["F"]
        new = ce_of(y, y["ee"] - sh, F2, y["late"], s["lane_end"], d, c)
        tot_old += old
        tot_new += new
    return (tot_old - tot_new) / 1e6 / NT


SL = [s for s in SIM if s["F"] is not None]
dF = np.array([(s["y"]["ls"] - max(s["y"]["ee"], s["F"])) / 1e3 for s in SL])
print(f"\n-- check: late start - max(early end, stager event fire) p10/p50/p90 = {np.percentile(dF, 10):.1f} / {np.percentile(dF, 50):.1f} / {np.percentile(dF, 90):.1f} us; the event fires after the early end in {np.mean([s['F'] > s['y']['ee'] for s in SL]) * 100:.0f} % of the {len(SL) / NT:.1f} layers/token with a late pass; summed {dF.sum() / 1e3 / NT:.2f} ms/token")
print(f"\n-- per-layer what-if on the measured marks ({len(SIM) / NT:.1f} layers/token)")
# R: the early pass launched right after the plan (before the stager API): starts at max(shared expert end, plan end + 8 us)
r_sh = lambda s: max(0, s["y"]["es"] - max(shared_end[id(s["y"])], s["plan_end"] + 8000))
print(f"  R  early pass launched before the stager API (start = max(shared/gather end, plan end + 8 us)): {run(shift_early=r_sh):.2f} ms/token"
      f"  (mean shift {np.mean([r_sh(s) for s in SIM]) / 1e3:.0f} us/layer)")
# N3: a demand read never waits behind a guess on the drive: its landing <= issue + 1,164 us + one 1 MiB piece (129 us)
def n3(s):
    y = s["y"]
    if s["F"] is None or y["n"] == 0 or not s["lands"]:
        return s["F"]
    cut = max(0, max(lat for _, lat in s["lands"]) - (1164e3 + 129e3) * 1)
    return max(y["ee"] - 1, s["F"] - cut) if cut else s["F"]
print(f"  N3 demand reads see an idle drive (landing <= issue + 1.29 ms; the guesses they pass are not charged): {run(F_new=n3):.2f} ms/token")
# J: every joined guess landed when its layer is routed (upper bound; the late pass still computes them)
def jn(s):
    y = s["y"]
    if s["F"] is None or y["n"] > 0 or y["wv"] == 0:
        return s["F"]
    return y["ee"] - 1
print(f"  J  joined guesses landed before their layer (join-only layers' stager wait -> 0; upper bound): {run(F_new=jn):.2f} ms/token")
both = lambda s: None if s["F"] is None else min(n3(s), jn(s))
print(f"  N3 + J together: {run(F_new=both):.2f} ms/token; with R as well: {run(shift_early=r_sh, F_new=both):.2f} ms/token")

# E: the late pass waits on the landed flags / copies directly (no s16 event -> s7 start latency): d -> 0
def run_d0():
    tot = 0
    for s in SIM:
        y = s["y"]
        if s["F"] is None:
            continue
        d = max(0, y["ls"] - max(y["ee"], s["F"]))
        c = y["ce"] - max(y["le"], s["lane_end"] or 0)
        old = ce_of(y, y["ee"], s["F"], y["late"], s["lane_end"], d, c)
        new = ce_of(y, y["ee"], s["F"], y["late"], s["lane_end"], 0, c)
        tot += old - new
    return tot / 1e6 / NT
print(f"  E  stager event -> late-pass start latency removed (upper bound): {run_d0():.2f} ms/token")
# lane binding: layers where the lane ends after the GPU's routed work
lb = [max(0, (s["lane_end"] or 0) - (s["y"]["le"] or s["y"]["ee"])) for s in SIM]
print(f"  lane ends after the GPU's routed work in {np.mean([v > 0 for v in lb]) * 100:.0f} % of layers, by {np.sum(lb) / 1e6 / NT:.2f} ms/token in total")
# R+: as R, and the CPU lane posted at the plan's end as well (finish_staged posts it after the stager batch today)
lane_start_g = {x["g0"]: (x["lane"][0] if x["lane"] else None) for x in L}
def run_r_lane():
    tot = 0
    for s in SIM:
        y = s["y"]
        d = max(0, y["ls"] - max(y["ee"], s["F"])) if s["F"] is not None else 0
        c = y["ce"] - max(y["le"] if y["le"] else y["ee"], s["lane_end"] or 0)
        old = ce_of(y, y["ee"], s["F"], y["late"], s["lane_end"], d, c)
        ls0 = lane_start_g.get(y["g0"])
        lsh = max(0, ls0 - (s["plan_end"] + 10000)) if (ls0 and s["lane_end"]) else 0
        new = ce_of(y, y["ee"] - r_sh(s), s["F"], y["late"], (s["lane_end"] - lsh) if s["lane_end"] else None, d, c)
        tot += old - new
    return tot / 1e6 / NT
print(f"  R+ R and the lane posted at the plan's end too: {run_r_lane():.2f} ms/token")
