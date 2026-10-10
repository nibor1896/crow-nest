"""Round-2 additions to analysis.py (profile-20261010k-decode, glm-integration 3836cea, ARM2 + RT2 + the four switches).

Re-run:  python -I analysis2.py [dir] [plain.log]
  runs analysis.py (the budget, unchanged method) on decode.sqlite/decode.json in [dir], then:
  1. splits "latency: launch/queue" into queued-not-started (>= 5 us) and back-to-back (< 5 us), and the
     >= 5 us part by what the kernel's stream waited on (s13 waiting on an s7 event, s7 waiting on an
     s13 event, nothing);
  2. joins every MoE sublayer of the trace with the per-token, per-layer tier counts v/p/n of the plain
     run's log (same ids, all 128 equal), so lane = 8 - early - late slots, zero-copy = p - lane;
  3. fits the zero-copy and lane costs per expert, alone and under each other (DDR5 contention);
  4. NVMe: per cuStreamWaitValue64 on the stager stream, host issue -> the s16 op behind it (landed);
     per-layer stager idle by demand reads.
"""
import bisect
import os
import re
import sys
from collections import defaultdict

import numpy as np

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
PLAIN = sys.argv[2] if len(sys.argv) > 2 else os.path.join(D, "plain.log")
sys.argv = [sys.argv[0], D]
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis.py")).read())
print()
print("=" * 100)
ms_ = lambda v: v / 1e6 / NT

# ---------------------------------------------------------------- 1. launch/queue latency split
SY_ALL = c.execute("select streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType = 2").fetchall()
waits_by_stream = defaultdict(list)
for st, cid, eid in SY_ALL:
    waits_by_stream[st].append((cid, eid))
for st in waits_by_stream:
    waits_by_stream[st].sort()
kc_by_stream = defaultdict(list)
for k in K:
    kc_by_stream[k[2]].append(k[7])
for st in kc_by_stream:
    kc_by_stream[st].sort()
# event record -> the op before it on that stream (by correlationId) -> its end = when the event fires
ops_by_stream = defaultdict(list)  # (cid, end)
for k in K:
    ops_by_stream[k[2]].append((k[7], k[1]))
for m in M:
    ops_by_stream[m[2]].append((m[5], m[1]))
for st in ops_by_stream:
    ops_by_stream[st].sort()


def fire_time(eid, before_cid):
    lst = ev_rec.get(eid, [])
    i = bisect.bisect_left(lst, (before_cid, -1)) - 1
    if i < 0:
        return None, None
    rcid, rst = lst[i]
    ops = ops_by_stream.get(rst, [])
    j = bisect.bisect_left(ops, (rcid, -1)) - 1
    return (ops[j][1] if j >= 0 else None), rst


lq_small = lq_big = 0
lq_cls = defaultdict(float)
lq_cnt = defaultdict(int)
for dur, j, H, g0, g1 in LQ:
    if dur < 5000:
        lq_small += dur
        continue
    lq_big += dur
    st = j[2]
    kc = kc_by_stream[st]
    ii = bisect.bisect_left(kc, j[7]) - 1
    prev = kc[ii] if ii >= 0 else j[7] - 400
    w = waits_by_stream.get(st, [])
    lo = bisect.bisect_left(w, (prev, -1))
    cls = None
    for cid, eid in w[lo:]:
        if cid >= j[7]:
            break
        ft, rst = fire_time(eid, cid)
        if rst is not None and rst != st:
            cls = f"s{st} kernel waits on an s{rst} event (cross-stream event -> start)"
    if cls is None:
        cls = f"s{st} {j[3] if st == 13 else 'kernel'}: launched, no wait, GPU idle (queued not started)"
    lq_cls[cls] += dur
    lq_cnt[cls] += 1
print(f"latency: launch/queue {ms_(lq_small + lq_big):.2f} = queued not started (>= 5 us) {ms_(lq_big):.2f} + back-to-back (< 5 us) {ms_(lq_small):.2f} ms/token")
for k_, v in sorted(lq_cls.items(), key=lambda x: -x[1]):
    print(f"    {k_:75s} {ms_(v):6.2f} ms/token ({lq_cnt[k_] / NT:5.1f}/token, {v / max(lq_cnt[k_], 1) / 1e3:6.1f} us each)")
n_small = sum(1 for x in LQ if x[0] < 5000) / NT
print(f"    back-to-back gaps: {n_small:.0f}/token")
# kernels per token and kernel count
print(f"kernels per token: s7 {len(k7) / NT:.0f}, s13 {len(k13) / NT:.0f}; cuLaunchKernel {sum(1 for r in R if r[3] == 'cuLaunchKernel' and A < r[0] < B) / NT:.0f}/token")

# ---------------------------------------------------------------- 2. per-layer join with the plain log
rows = {}
for line in open(PLAIN, encoding="utf-8", errors="replace"):
    m = re.search(r"row\s+(\d+) gen .*? ([\d.]+) s\s+NVMe reads\s+(\d+) .*?tiers v/p/n (\d+)/(\d+)/(\d+)\s+\[(.*)\]", line)
    if m:
        pos = int(m.group(1))
        lay = re.findall(r"l(\d+) (\d+)/(\d+)/(\d+)", m.group(7))
        rows[pos] = (float(m.group(2)), [(int(a), int(b), int(c_), int(d_)) for a, b, c_, d_ in lay])
pos0 = min(rows)  # row pos0 = trace token 1
assert len(moe_layers) == 42 * NT, len(moe_layers)
H2D16 = sorted((m[0], m[1]) for m in cop16 if m[3] == 1 and m[4] > 1_000_000)
H2D16_S = [x[0] for x in H2D16]
L = []  # one dict per MoE sublayer
for idx, l in enumerate(moe_layers):
    t, i = divmod(idx, 42)
    row = rows.get(pos0 + t)
    if row is None or len(row[1]) != 42:
        continue
    lay_no, v, p, n = row[1][i]
    E, Lt = l["n_early"], l["n_late"]
    lane_n = 8 - E - Lt
    zc = p - lane_n
    gat, comb = l["gather"], l["combine"]
    if not gat or not comb:
        continue
    early = l["early"]
    late = l["late"]
    gm = [k for k in early if k[3] == "mul1_gemv"]
    zc_ex = sum(max(0, (k[1] - k[0]) - (c0 + c1 * E)) for k in gm)
    lane = lanes.get(idx)
    e_end = max((k[1] for k in early), default=gat[1])
    e_start = min((k[0] for k in early), default=gat[1])
    l_start = min((k[0] for k in late), default=None)
    l_end = max((k[1] for k in late), default=None)
    i0 = bisect.bisect_left(H2D16_S, gat[0])
    i1 = bisect.bisect_left(H2D16_S, comb[0])
    L.append(dict(t=t, i=i, layer=lay_no, v=v, p=p, n=n, E=E, Lt=Lt, lane_n=lane_n, zc=zc, g0=gat[0], g1=gat[1], c0=comb[0], c1=comb[1],
                  e_start=e_start, e_end=e_end, early_gpu=sum(k[1] - k[0] for k in early), zc_ex=zc_ex, gemv=sum(k[1] - k[0] for k in gm), n_gemv=len(gm),
                  l_start=l_start, l_end=l_end, late_gpu=sum(k[1] - k[0] for k in late), lane=lane, h2d=i1 - i0, wait=l["wait_cid"] is not None))
print(f"\njoined MoE sublayers: {len(L)} of {len(moe_layers)}; negative zero-copy counts (join mismatch): {sum(1 for x in L if x['zc'] < 0)}")
Lg = [x for x in L if x["zc"] >= 0]
tot = lambda key: sum(x[key] for x in Lg) / NT
print(f"per token: VRAM {tot('v'):.1f}, pinned {tot('p'):.1f} (lane {tot('lane_n'):.1f} + zero-copy {tot('zc'):.1f}), NVMe {tot('n'):.1f}; early slots {tot('E'):.1f}, late {tot('Lt'):.1f}")

# ---------------------------------------------------------------- 3. contention fits
print("\n-- zero-copy cost per expert (early mul1_gemv excess over the all-VRAM model, 3 GEMVs per expert):")
def frac_under(iv, a, b):
    return length(clipu(LU, a, b)) / (b - a) if b > a else 0


def fit(xs, ys):
    X = np.c_[np.ones(len(xs)), xs]
    co = np.linalg.lstsq(X, ys, rcond=None)[0]
    return co


for name, sel in (("no lane in the layer", lambda x: x["lane_n"] == 0), ("lane in the layer", lambda x: x["lane_n"] > 0)):
    S_ = [x for x in Lg if sel(x) and x["n_gemv"] == 3]
    if len(S_) < 10:
        continue
    xs = np.array([x["zc"] for x in S_], float)
    ys = np.array([x["zc_ex"] for x in S_], float) / 1e3
    co = fit(xs, ys)
    by = defaultdict(list)
    for x in S_:
        by[x["zc"]].append(x["zc_ex"] / 1e3)
    tab = ", ".join(f"{k_}:{np.median(v_):.0f}us(n{len(v_)})" for k_, v_ in sorted(by.items()) if len(v_) >= 5)
    print(f"  {name:22s} {len(S_):5d} layers: excess = {co[0]:6.1f} + {co[1]:6.1f} us x zc  | median by zc {tab}")
print("-- CPU lane time per layer (start-in-GPU query -> busy query before the combine launch):")
for name, sel in (("no zero-copy", lambda x: x["zc"] == 0), ("with zero-copy", lambda x: x["zc"] > 0), ("all", lambda x: True)):
    S_ = [x for x in Lg if x["lane"] and x["lane_n"] > 0 and sel(x)]
    if len(S_) < 10:
        continue
    xs = np.array([x["lane_n"] for x in S_], float)
    ys = np.array([(x["lane"][1] - x["lane"][0]) for x in S_], float) / 1e3
    co = fit(xs, ys)
    by = defaultdict(list)
    for x in S_:
        by[x["lane_n"]].append((x["lane"][1] - x["lane"][0]) / 1e3)
    tab = ", ".join(f"{k_}:{np.median(v_):.0f}us(n{len(v_)})" for k_, v_ in sorted(by.items()) if len(v_) >= 5)
    print(f"  {name:22s} {len(S_):5d} layers: lane = {co[0]:6.1f} + {co[1]:6.1f} us x experts | median by count {tab}")

# the routed MoE span of a layer without NVMe work (no stager wait, no landing copy): gather start -> combine end
print("-- MoE span (gather start -> combine end), layers without an NVMe demand read or stager H2D:")
S_ = [x for x in Lg if x["n"] == 0 and x["h2d"] == 0]
print(f"  {len(S_)} layers ({len(S_) / NT:.1f}/token), mean span {np.mean([x['c1'] - x['g0'] for x in S_]) / 1e3:.0f} us")
X = np.array([[1, x["v"], x["zc"], x["lane_n"]] for x in S_], float)
y = np.array([x["c1"] - x["g0"] for x in S_], float) / 1e3
co = np.linalg.lstsq(X, y, rcond=None)[0]
pred = X @ co
r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
print(f"  span = {co[0]:.0f} + {co[1]:.1f} x VRAM + {co[2]:.1f} x zero-copy + {co[3]:.1f} x lane (us), R2 {r2:.3f}")
gpu_end = np.array([x["e_end"] - x["g0"] for x in S_]) / 1e3
lane_end = np.array([((x["lane"][1] - x["g0"]) if x["lane"] else 0) for x in S_]) / 1e3
print(f"  early GPU ends {np.mean(gpu_end):.0f} us after the gather, the lane {np.mean(lane_end[lane_end > 0]):.0f} us; lane last in {np.mean(lane_end > gpu_end) * 100:.0f} % of layers")
# host bytes rate when both run
both = [x for x in S_ if x["lane_n"] > 0 and x["zc"] > 0 and x["lane"]]
if both:
    by_ = sum((x["lane_n"] + x["zc"]) * 9474048 for x in both)
    tt = sum(max(x["e_end"], x["lane"][1]) - min(x["e_start"], x["lane"][0]) for x in both)
    print(f"  layers with lane AND zero-copy: {len(both) / NT:.1f}/token, host-served bytes / (first start -> last end) = {by_ / tt:.1f} GB/s")
only_zc = [x for x in S_ if x["lane_n"] == 0 and x["zc"] > 0]
if only_zc:
    by_ = sum(x["zc"] * 9474048 for x in only_zc)
    tt = sum(x["zc_ex"] for x in only_zc)
    print(f"  zero-copy only: {len(only_zc) / NT:.1f}/token, {by_ / tt:.1f} GB/s over the zero-copy excess")
only_lane = [x for x in S_ if x["lane_n"] > 0 and x["zc"] == 0 and x["lane"]]
if only_lane:
    by_ = sum(x["lane_n"] * 9474048 for x in only_lane)
    tt = sum(x["lane"][1] - x["lane"][0] for x in only_lane)
    print(f"  lane only: {len(only_lane) / NT:.1f}/token, {by_ / tt:.1f} GB/s over the lane time")

# ---------------------------------------------------------------- 4. NVMe
print("\n-- NVMe: stager-stream waits (cuStreamWaitValue64 on s16) -> the s16 op behind it")
api_sorted = sorted((cid, s, e, n) for s, e, cid, n, _ in R)
s16_ops = sorted([(m[5], m[0], m[1], m[3], m[4]) for m in M if m[2] == 16])
s16_cids = [x[0] for x in s16_ops]
lat = []
for cid, s, e, n in api_sorted:
    if n != "cuStreamWaitValue64_v2" or not (A < s < B):
        continue
    j = bisect.bisect_right(s16_cids, cid)
    if j >= len(s16_ops):
        continue
    op = s16_ops[j]
    # only the stager's waits: the op behind the wait is an s16 copy queued by the next APIs
    if op[0] - cid > 40:
        continue
    lat.append((op[1] - s) / 1e3)
lat = np.array(lat)
if len(lat):
    print(f"  {len(lat) / NT:.1f} waits/token; issue -> s16 op start: p10 {np.percentile(lat, 10):.0f} p50 {np.percentile(lat, 50):.0f} p90 {np.percentile(lat, 90):.0f} us, mean {lat.mean():.0f} us")
    print(f"  bins (us): " + ", ".join(f"<{b}: {np.mean(lat < b) * 100:.0f}%" for b in (100, 500, 1000, 1500, 2000, 3000, 5000)))
print("-- per layer by NVMe demand reads n (plain-log counts): GPU idle between gather and combine, span")
for nn in range(0, 5):
    S_ = [x for x in Lg if x["n"] == nn]
    if not S_:
        continue
    span = np.mean([x["c1"] - x["g0"] for x in S_]) / 1e3
    busy_ = np.mean([x["early_gpu"] + x["late_gpu"] for x in S_]) / 1e3
    print(f"  n={nn}: {len(S_) / NT:5.2f} layers/token, span {span:6.0f} us, routed GPU busy {busy_:5.0f} us, late wait present {np.mean([x['wait'] for x in S_]) * 100:3.0f} %")
ns = np.array([x["n"] for x in Lg], float)
spans = np.array([x["c1"] - x["g0"] for x in Lg], float) / 1e3
co = fit(ns, spans)
print(f"  span = {co[0]:.0f} + {co[1]:.0f} us x n  (every MoE layer)")

# ---------------------------------------------------------------- 5. token critical path on s7 (additive)
# token = sum over its kernels' timeline cut at: topk end, publish start, gather start, combine end
print("\n-- critical path per token (s7 timeline cut at the MoE layers' marks, ms/token):")
cp = defaultdict(float)
prev_end = A
spans_n = defaultdict(list)
for idx, l in enumerate(moe_layers):
    seg = l["seg"]
    tk = next((k for k in seg if k[3] == "glm5_router_sig_topk"), None)
    rt = next((k for k in seg if k[3] == "gemv_bf16_b"), None)
    pb = next((k for k in seg if k[3] == "glm5_publish_pred"), None)
    g, cb = l["gather"], l["combine"]
    if not (tk and rt and pb and g and cb):
        continue
    t_tok = bisect.bisect_left(TOK, rt[0])  # token boundary inside the stretch?
    cp["attention, mHC, dense, head, turnaround (previous combine end -> router start)"] += rt[0] - prev_end
    cp["router (router start -> topk end)"] += tk[1] - rt[0]
    cp["side-guess join (topk end -> publish start)"] += pb[0] - tk[1]
    cp["routing step (publish start -> gather start; shared expert runs here)"] += g[0] - pb[0]
    cp["MoE span (gather start -> combine end)"] += cb[1] - g[0]
    prev_end = cb[1]
cp["tail (last combine end -> window end)"] += B - prev_end
tot_cp = sum(cp.values())
for k_, v in cp.items():
    print(f"  {k_:80s} {ms_(v):6.2f}")
print(f"  {'sum':80s} {ms_(tot_cp):6.2f}  (window {ms_(B - A):.2f})")

# ---------------------------------------------------------------- 6. DDR5 contention on clean layers + split simulation
# clean = no demand read, no joined guess, no late slot: zero-copy = early slots - VRAM exactly, lane = 8 - early
print("\n-- clean layers (no demand read, no stager wait value, no late slot): zero-copy = early - VRAM, lane = 8 - early")
Rc2 = sorted((cid, n) for s, e, cid, n, _ in R)
rcids = [x[0] for x in Rc2]
prev_cb = None
for x, l in zip(L, moe_layers):
    lo = bisect.bisect_right(rcids, prev_cb) if prev_cb else bisect.bisect_left(rcids, l["gather"][7] - 300)
    hi = bisect.bisect_left(rcids, l["gather"][7])
    x["waits"] = sum(1 for y in Rc2[lo:hi] if y[1] == "cuStreamWaitValue64_v2")
    prev_cb = l["combine"][7]
C = [x for x in L if x["n"] == 0 and x["waits"] == 0 and x["Lt"] == 0 and x["lane"]]
for x in C:
    x["zc2"] = x["E"] - x["v"]
C = [x for x in C if x["zc2"] >= 0]
print(f"  {len(C) / NT:.1f} layers/token; mean VRAM {np.mean([x['v'] for x in C]):.2f}, zero-copy {np.mean([x['zc2'] for x in C]):.2f}, lane {np.mean([x['lane_n'] for x in C]):.2f}; span {np.mean([x['c1'] - x['g0'] for x in C]) / 1e3:.0f} us")
Xg = np.array([[1, x["v"], x["zc2"], x["lane_n"]] for x in C], float)
yg = np.array([x["e_end"] - x["g0"] for x in C], float) / 1e3
cg = np.linalg.lstsq(Xg, yg, rcond=None)[0]
Xl = np.array([[1, x["lane_n"], x["zc2"]] for x in C], float)
yl = np.array([x["lane"][1] - x["g0"] for x in C], float) / 1e3
cl = np.linalg.lstsq(Xl, yl, rcond=None)[0]
ys = np.array([x["c1"] - x["g0"] for x in C], float) / 1e3
print(f"  early GPU end (after gather start) = {cg[0]:.0f} + {cg[1]:.1f} x VRAM + {cg[2]:.1f} x zero-copy + {cg[3]:.1f} x lane experts (us)")
print(f"  lane end (after gather start)      = {cl[0]:.0f} + {cl[1]:.1f} x lane experts + {cl[2]:.1f} x zero-copy (us)")
print(f"  -> one expert: zero-copy {cg[2]:.0f} us on the GPU (+{cl[2]:.0f} us on the lane), lane {cl[1]:.0f} us (+{cg[3]:.0f} us on the GPU)")
by = defaultdict(list)
for x in C:
    by[(x["lane_n"], x["zc2"])].append((x["e_end"] - x["g0"], x["lane"][1] - x["g0"], x["c1"] - x["g0"]))
print("  (lane, zero-copy): n/token, GPU end / lane end / span us")
for k_, v_ in sorted(by.items(), key=lambda z: -len(z[1]))[:10]:
    a_ = np.array(v_) / 1e3
    print(f"    {k_}: {len(v_) / NT:4.1f}  {np.median(a_[:, 0]):5.0f} / {np.median(a_[:, 1]):5.0f} / {np.median(a_[:, 2]):5.0f}")
tail = np.mean(ys - np.maximum(yg, yl))
print(f"  span - max(GPU end, lane end) = {tail:.0f} us (combine launch + run)")
# split simulation with the fitted (contended) linear models: per layer, P = lane + zero-copy fixed
def T(v, nz, nl):
    g_ = cg[0] + cg[1] * v + cg[2] * nz + cg[3] * nl
    l_ = (cl[0] + cl[1] * nl + cl[2] * nz) if nl > 0 else 0
    return max(g_, l_)
now = sum(T(x["v"], x["zc2"], x["lane_n"]) for x in C)
best = sum(min(T(x["v"], x["lane_n"] + x["zc2"] - j, j) for j in range(0, x["lane_n"] + x["zc2"] + 1)) for x in C)
allzc = sum(T(x["v"], x["lane_n"] + x["zc2"], 0) for x in C)
alllane = sum(T(x["v"], 0, x["lane_n"] + x["zc2"]) for x in C)
print(f"  split simulation, sum of max(GPU, lane) over clean layers, ms/token: now {now / 1e3 / NT:.2f}, best split per layer {best / 1e3 / NT:.2f}, all zero-copy {allzc / 1e3 / NT:.2f}, all lane {alllane / 1e3 / NT:.2f}")
print(f"  per clean layer: now {now / len(C):.0f} us, best {best / len(C):.0f} us -> over 42 layers {(now - best) / len(C) * 42 / 1e3:.2f} ms/token")
hb = np.mean([(x["lane_n"] + x["zc2"]) * 9474048 for x in C])
print(f"  host-served bytes per clean layer {hb / 1e6:.1f} MB; at the measured {hb / (np.mean(np.maximum(yg, yl)) * 1e3):.1f} GB/s (bytes / max end)")
