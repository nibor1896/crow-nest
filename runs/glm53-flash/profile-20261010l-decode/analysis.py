"""Per-token decode budget (round-1 method), profile-20261010l-decode copy: glm-integration 3de5eff, ARM2 + RT2 + 12 switches.

Changes against profile-20261010k-decode/analysis.py (only what the new kernels need, method unchanged):
  - sublayer cut also at glm5_mhc_expand_mix_norm (CROW_GLM_ATTN_FUSE + HCFUSE);
  - glm5_publish / glm5_guess_publish count as router (CROW_GLM_SIDE_NOJOIN);
  - with the fused FFN the glm5_swiglu_clamp after the gather is the shared expert's (RT2_GATHER_EARLY), not routed;
  - CROW_GLM_MUL1_FUSE: slots per pass from mul1_gemv_o gridZ; the all-VRAM base per fused kernel
    (p5 per slot count of this trace): mul1_gemv_gu 11.3 + 9.7 us x n, mul1_gemv_o 6.7 + 6.3 us x n.


Re-run:  python -I analysis.py [dir]      (reads decode.sqlite + decode.json next to this file)

Window: the 127 generated tokens = (end of argmax_k #0, end of argmax_k #127]; token t = (argmax end t-1, argmax end t].

GPU busy = union of all kernels (s7 compute, s13 side guess). Each busy instant goes to the category of
the s7 kernel running then (s13 only where no s7 kernel runs). s7 kernels are grouped by sublayer: the
sequence is cut at every glm5_mhc_mix_norm (and at glm5_stream_mean_rms for the head); a sublayer with
gm_* is MLA attention, with kda_* KDA attention, with mul1_* / glm5_moe_gather the MoE FFN, with
glm5_gemv_fp4 grid 6144 the dense FFN of layers 0-2. Inside the MoE FFN the host's call order splits the
routed experts: mul1 launches before the cuStreamWaitEvent on the stager event = early pass (VRAM +
zero-copy slots), after it = late pass (staged: NVMe landings / write-back targets).

GPU idle = window - busy, cut into gaps; each gap is attributed to what the next kernel j waited on:
  host part   [g0, min(H, g1)] with H = end of j's cuLaunchKernel (correlationId): the host had not
              launched j yet. Named by j: combine_rows -> CPU lane; moe_gather -> routing step (flag,
              plan, launch); first kernel of a token -> token turnaround; else host launch-bound.
  device part [max(g0, H), g1]: copies on s16 (stager H2D landing->stage, D2D) and s14 (write-back
              D2H) inside it are copy time; if j's stream waited on an event recorded on s16 (the
              stager event), the rest before the last copy and any tail > 30 us after it is the NVMe
              landing (the stager's cuStreamWaitValue64 on the landed flag); s7 in-stream copies are
              their own line; what is left is launch/event latency.
"""
import bisect
import json
import os
import sqlite3
import sys
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))
c = sqlite3.connect(os.path.join(D, "decode.sqlite"))
S = dict(c.execute("select id, value from StringIds"))

K = c.execute("select start, end, streamId, shortName, gridX, gridY, gridZ, correlationId from CUPTI_ACTIVITY_KIND_KERNEL order by start").fetchall()
K = [(s, e, st, S[n], gx, gy, gz, cid) for s, e, st, n, gx, gy, gz, cid in K]
M = c.execute("select start, end, streamId, copyKind, bytes, correlationId from CUPTI_ACTIVITY_KIND_MEMCPY order by start").fetchall()
R = c.execute("select r.start, r.end, r.correlationId, r.nameId, r.globalTid from CUPTI_ACTIVITY_KIND_RUNTIME r order by r.start").fetchall()
R = [(s, e, cid, S[n], tid) for s, e, cid, n, tid in R]
api_by_cid = {cid: (s, e, n) for s, e, cid, n, _ in R}
SY = c.execute("select streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType = 2").fetchall()
EV = c.execute("select streamId, correlationId, eventId from CUPTI_ACTIVITY_KIND_CUDA_EVENT order by correlationId").fetchall()

am = [k for k in K if k[3] == "argmax_k"]
TOK = [k[1] for k in am]
A, B = TOK[0], TOK[-1]
NT = len(TOK) - 1

# event record lookup: eventId -> sorted [(record correlationId, stream)]
ev_rec = defaultdict(list)
for st, cid, eid in EV:
    ev_rec[eid].append((cid, st))
# stream waits on s7: sorted correlationIds with (eventId)
waits7 = sorted((cid, eid) for st, cid, eid in SY if st == 7)
waits7_cid = [w[0] for w in waits7]


def record_stream(eid, before_cid):
    lst = ev_rec.get(eid, [])
    i = bisect.bisect_left(lst, (before_cid, -1)) - 1
    return lst[i][1] if i >= 0 else None


# ---- s7 kernels in the window, sublayer parse
k7 = [k for k in K if k[2] == 7 and k[1] > A and k[0] < B]
k13 = [k for k in K if k[2] == 13 and k[1] > A and k[0] < B]
cat = {}  # (start, corr) -> category
segs = []
cur = []
for k in k7:
    if k[3] in ("glm5_mhc_mix_norm", "glm5_mhc_expand_mix_norm", "glm5_stream_mean_rms") and cur:
        segs.append(cur)
        cur = []
    cur.append(k)
segs.append(cur)
moe_layers = []  # per MoE sublayer: dict with early / late kernels
for seg in segs:
    names = {k[3] for k in seg}
    if any(n.startswith("gm_") or n.startswith("qsa_") for n in names):
        kind = "attn MLA"
    elif any(n.startswith("kda_") for n in names):
        kind = "attn KDA"
    elif "glm5_moe_gather" in names or any(n.startswith("mul1_") for n in names):
        kind = "moe"
    elif "argmax_k" in names or "glm5_stream_mean_rms" in names:
        kind = "head"
    elif any(n == "glm5_gemv_fp4" and k[4] == 6144 for k in seg for n in [k[3]]):
        kind = "dense FFN (l0-2)"
    else:
        kind = "other"
    if kind != "moe":
        for k in seg:
            c_ = "mHC/norm" if k[3].startswith("glm5_mhc") else kind
            cat[(k[0], k[7])] = c_
        continue
    # MoE sublayer: router, shared, gather/combine, early/late by the stream wait on the stager event
    lay = {"early": [], "late": [], "n_early": 0, "n_late": 0, "gather": None, "combine": None, "seg": seg}
    # the stager wait: a stream wait on s7 (between the gather and the combine) whose event was recorded on s16
    gat = next((k for k in seg if k[3] == "glm5_moe_gather"), None)
    comb = next((k for k in seg if k[3] == "glm5_moe_combine_rows"), None)
    wcid = None
    if gat and comb:
        i = bisect.bisect_left(waits7_cid, gat[7])
        while i < len(waits7) and waits7[i][0] < comb[7]:
            if record_stream(waits7[i][1], waits7[i][0]) == 16:
                wcid = waits7[i][0]
                break
            i += 1
    lay["gather"], lay["combine"], lay["wait_cid"] = gat, comb, wcid
    for k in seg:
        n = k[3]
        if n.startswith("glm5_mhc"):
            c_ = "mHC/norm"
        elif n in ("gemv_bf16_b", "glm5_router_sig_topk", "glm5_publish_pred", "glm5_pred_tag", "glm5_publish", "glm5_guess_publish"):
            c_ = "router"
        elif n in ("glm5_moe_gather", "glm5_moe_combine_rows", "glm5_moe_combine"):
            c_ = "moe gather/combine"
        elif n.startswith("mul1_") or (n == "glm5_swiglu_clamp" and gat and k[7] > gat[7] and "mul1_gemv_gu" not in names):  # fused FFN: the swiglu after the gather is the shared expert's (gather queued early)
            late = wcid is not None and k[7] > wcid
            c_ = "routed late (staged)" if late else "routed early (VRAM+zero-copy)"
            lay["late" if late else "early"].append(k)
            if n == "mul1_had_in":
                lay["n_late" if late else "n_early"] += k[6]
            elif n == "mul1_gemv_o":  # fused FFN: one down GEMV launch per pass, gridZ = slots
                lay["n_late" if late else "n_early"] += 3 * k[6]
        elif n in ("glm5_gemv_fp4", "glm5_swiglu_clamp"):
            c_ = "shared expert"
        else:
            c_ = "moe other"
        cat[(k[0], k[7])] = c_
    lay["n_early"] //= 3  # had_in runs for gate, up, down
    lay["n_late"] //= 3
    moe_layers.append(lay)
for k in k13:
    cat[(k[0], k[7])] = "router side guess (s13)"


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            if e > out[-1][1]:
                out[-1][1] = e
        else:
            out.append([s, e])
    return out


def clipu(iv, a, b):
    return [(max(s, a), min(e, b)) for s, e in iv if e > a and s < b]


def length(iv):
    return sum(e - s for s, e in iv)


def subtract(x, y):
    """x, y merged lists; x minus y"""
    out = []
    j = 0
    for s, e in x:
        cur = s
        while j < len(y) and y[j][1] <= cur:
            j += 1
        jj = j
        while jj < len(y) and y[jj][0] < e:
            if y[jj][0] > cur:
                out.append([cur, y[jj][0]])
            cur = max(cur, y[jj][1])
            jj += 1
        if cur < e:
            out.append([cur, e])
    return out


# ---- busy by category (s7 has priority, s13 only where no s7 kernel)
busy = defaultdict(float)
u7 = union([(max(k[0], A), min(k[1], B)) for k in k7])
by_cat = defaultdict(list)
for k in k7:
    by_cat[cat[(k[0], k[7])]].append((max(k[0], A), min(k[1], B)))
for cname, iv in by_cat.items():
    busy[cname] = length(union(iv))
u13 = union([(max(k[0], A), min(k[1], B)) for k in k13])
busy["router side guess (s13)"] = length(subtract(u13, u7))
u_all = union([(k[0], k[1]) for k in k7 + k13])
u_all = [list(x) for x in clipu(u_all, A, B)]

# ---- idle gaps
idle = defaultdict(float)
gaps = []
prev = A
for s, e in u_all:
    if s > prev:
        gaps.append((prev, s))
    prev = max(prev, e)
if prev < B:
    gaps.append((prev, B))
start_index = sorted((k[0], k) for k in k7 + k13)
starts = [x[0] for x in start_index]
cop16 = [m for m in M if m[2] == 16 and m[1] > A and m[0] < B]
cop14 = [m for m in M if m[2] == 14 and m[1] > A and m[0] < B]
cop7 = [m for m in M if m[2] == 7 and m[1] > A and m[0] < B]
gather_cids = {l["gather"][7] for l in moe_layers if l["gather"]}
comb_cids = {l["combine"][7] for l in moe_layers if l["combine"]}
first_of_token = set()
for t in TOK[:-1]:
    i = bisect.bisect_right(starts, t)
    if i < len(start_index):
        first_of_token.add(start_index[i][1][7])
late_first = {}
for l in moe_layers:
    if l["late"]:
        late_first[l["late"][0][7]] = l
c16h = union([(m[0], m[1]) for m in cop16 if m[3] == 1])
c16d = union([(m[0], m[1]) for m in cop16 if m[3] != 1])
c14 = union([(m[0], m[1]) for m in cop14])
c7 = union([(m[0], m[1]) for m in cop7])
COPYSETS = (("copy: s7 in-stream (x D2H, table H2D, token I/O)", c7), ("copy: stager H2D landing->stage (s16)", c16h), ("copy: stager D2D / table (s16)", c16d), ("copy: write-back D2H VRAM->pinned (s14)", c14))
CS_START = {id(iv): [x[0] for x in iv] for _, iv in COPYSETS}
COP_END = sorted([m[1] for m in cop16 + cop14])


def clipb(iv, a, b):
    st = CS_START[id(iv)]
    i = max(0, bisect.bisect_left(st, a) - 1)
    out = []
    while i < len(iv) and iv[i][0] < b:
        s, e = iv[i]
        if e > a:
            out.append((max(s, a), min(e, b)))
        i += 1
    return out


# the s7 predecessor (host call order) of every s7 kernel: a stream wait counts only for the first kernel after it
_k7c = sorted(k[7] for k in k7)
PREV7 = {b_: a_ for a_, b_ in zip(_k7c, _k7c[1:])}
C7 = sorted(cop7, key=lambda m: m[5])
C7_CID = [m[5] for m in C7]
gap_log = []
LQ = []
for g0, g1 in gaps:
    i = bisect.bisect_left(starts, g1)
    j = start_index[i][1] if i < len(start_index) else None
    if j is None:
        idle["other/unattributed"] += g1 - g0
        continue
    api = api_by_cid.get(j[7])
    H = api[1] if api else g0
    host = max(0, min(H, g1) - g0)
    if host > 0:
        if j[7] in comb_cids:
            hc = "host: CPU lane (combine waits for the lane rows)"
        elif j[7] in gather_cids:
            hc = "host: routing step (flag -> plan -> launch)"
        elif j[7] in first_of_token:
            hc = "host: token turnaround (argmax readback -> next launch)"
        else:
            hc = "host: launch-bound (kernel launched after the GPU ran dry)"
        idle[hc] += host
    d0 = max(g0, min(H, g1))
    if g1 <= d0:
        continue
    # (1) j's in-stream predecessor is an s7 copy that ran inside this gap: j waited for that copy, the
    #     copy waited in the copy queue behind the stager's copies / write-backs / its wait on a landed flag
    X = None
    if j[2] == 7:
        pc = PREV7.get(j[7], j[7] - 400)
        i7 = bisect.bisect_right(C7_CID, j[7]) - 1
        if i7 >= 0 and C7[i7][5] > pc and C7[i7][1] > d0:
            X = C7[i7]
    if X is not None:
        xs = max(X[0], d0)
        idle["copy: s7 in-stream (x D2H, table H2D, token I/O)"] += X[1] - xs
        idle["latency: copy/event -> kernel start"] += max(0, g1 - X[1])
        if xs > d0:
            blk = [[d0, xs]]
            for name, iv in COPYSETS[1:]:
                ivc = union([list(x) for x in clipb(iv, d0, xs)])
                part = length(ivc)
                if part > 0:
                    idle["queue: " + name[6:]] += part
                    blk = subtract(blk, ivc)
            bl = length(blk)
            if (xs - d0) > 30_000:
                idle["NVMe landing via copy queue (s7 copy behind the stager's landed-flag wait)"] += bl
            else:
                idle["latency: copy queue"] += bl
        continue
    # (2) j's stream waited on the stager event (recorded on s16) since its predecessor
    waits_stager = False
    if j[2] == 7:
        pc = PREV7.get(j[7], j[7] - 400)
        lo = bisect.bisect_left(waits7_cid, pc)
        for cid, eid in waits7[lo:]:
            if cid >= j[7]:
                break
            if record_stream(eid, cid) == 16:
                waits_stager = True
    if waits_stager:
        rest = [[d0, g1]]
        for name, iv in COPYSETS[1:]:
            ivc = union([list(x) for x in clipb(iv, d0, g1)])
            part = length(ivc)
            if part > 0:
                idle[name] += part
                rest = subtract(rest, ivc)
        rl = length(rest)
        ii = bisect.bisect_right(COP_END, g1) - 1
        last_copy_end = COP_END[ii] if ii >= 0 and COP_END[ii] >= d0 else d0
        tail = g1 - last_copy_end
        lat = tail if tail < 30_000 else 0
        idle["NVMe landing (late pass waits on the stager event)"] += rl - min(lat, rl)
        idle["latency: copy/event -> kernel start"] += min(lat, rl)
    elif j[3] == "glm5_publish_pred":
        idle["router: publish waits for the side guess (s13)"] += g1 - d0
    else:
        idle["latency: launch/queue (kernel launched before the GPU ran dry)"] += g1 - d0
        LQ.append((g1 - d0, j, H, g0, g1))

W = B - A
tot_b = sum(busy.values())
tot_i = sum(idle.values())
ms = lambda v: v / 1e6 / NT
print(f"window {W / 1e9:.3f} s, {NT} tokens, {ms(W):.2f} ms/token; busy {ms(tot_b):.2f} + idle {ms(tot_i):.2f} = {ms(tot_b + tot_i):.2f}")
print("GPU busy (ms/token):")
for k_, v in sorted(busy.items(), key=lambda x: -x[1]):
    print(f"  {k_:60s} {ms(v):7.2f}")
print("GPU idle (ms/token):")
for k_, v in sorted(idle.items(), key=lambda x: -x[1]):
    print(f"  {k_:60s} {ms(v):7.2f}")

# ---- MoE layer stats
ne = sum(l["n_early"] for l in moe_layers) / NT
nl = sum(l["n_late"] for l in moe_layers) / NT
print(f"MoE sublayers {len(moe_layers) / NT:.1f}/token; GPU routed slots per token: early {ne:.1f}, late {nl:.1f}; with a stager wait {sum(1 for l in moe_layers if l['wait_cid']) / NT:.1f}/token")
json.dump({"busy_ms": {k_: ms(v) for k_, v in busy.items()}, "idle_ms": {k_: ms(v) for k_, v in idle.items()}, "ms_per_token": ms(W), "tokens": NT,
           "early_slots_per_token": ne, "late_slots_per_token": nl}, open(os.path.join(D, "budget.json"), "w"), indent=1)

# ---- routed experts: zero-copy vs VRAM part of mul1_gemv, CPU lane, contention (inputs of the projection)
BASE = {"mul1_gemv": (2.5e3, 5.2e3), "mul1_gemv_gu": (11.3e3, 9.7e3), "mul1_gemv_o": (6.7e3, 6.3e3)}
c0, c1 = 2.5e3, 5.2e3  # all-VRAM gemv of n slots ~ c0 + c1*n ns (late n=1 median 8.5 us, early n=6 all-VRAM ~33.6 us)
Rc = sorted((cid, (st, en, n)) for st, en, cid, n, _ in R)
Rcid = [x[0] for x in Rc]
lanes = {}
for idx, l in enumerate(moe_layers):
    comb = l["combine"]
    if not comb:
        continue
    i = bisect.bisect_left(Rcid, comb[7]) - 1
    qs = []
    while i >= 0 and len(qs) < 3 and Rc[i][0] > comb[7] - 60:
        if Rc[i][1][2] == "cuEventQuery":
            qs.append(Rc[i][1])
        elif Rc[i][1][2] == "cuLaunchKernel" and qs:
            break
        i -= 1
    if len(qs) >= 2:  # the lane runs between the start-in-GPU query and the busy query before the combine launch
        lanes[idx] = (qs[1][1], qs[0][0])
LU = union(list(lanes.values()))
vr = ex_ov = ex_no = had = 0
calls = {"ov": [0, 0, 0], "no": [0, 0, 0]}
lane_gt_gpu = lane_gt_vram = 0
for idx, l in enumerate(moe_layers):
    gv = 0
    for part, n in (("early", l["n_early"]), ("late", l["n_late"])):
        for k in l[part]:
            d = k[1] - k[0]
            if k[3] not in BASE:
                had += d
                gv += d
                continue
            base = BASE[k[3]][0] + BASE[k[3]][1] * n
            ex = max(0, d - base)
            f = length(clipu(LU, k[0], k[1])) / d if d else 0
            vr += min(d, base)
            gv += min(d, base)
            ex_ov += ex * f
            ex_no += ex * (1 - f)
            if part == "early" and (f > 0.8 or f < 0.05):
                c_ = calls["ov" if f > 0.8 else "no"]
                c_[0] += ex
                c_[1] += 1
                c_[2] += n
    a_, b_ = lanes.get(idx, (0, 0))
    g = sum(k[1] - k[0] for k in l["early"] + l["late"])
    lane_gt_gpu += max(0, (b_ - a_) - g)
    lane_gt_vram += max(0, (b_ - a_) - gv)
dj = json.load(open(os.path.join(D, "decode.json")))
cn = dj["reps_detail"][0]["decode"]["counters"]
lane_exp = sum(8 - l["n_early"] - l["n_late"] for l in moe_layers) / NT
lane_ms = sum(b_ - a_ for a_, b_ in lanes.values()) / 1e6 / NT
zc_ex = (ex_ov + ex_no) / 1e6 / NT
slow = (calls["ov"][0] / max(calls["ov"][1], 1)) / max(calls["no"][0] / max(calls["no"][1], 1), 1)
print(f"routed mul1_gemv: VRAM-equivalent {vr / 1e6 / NT:.2f} + zero-copy excess {zc_ex:.2f} (under the lane {ex_ov / 1e6 / NT:.2f}, without {ex_no / 1e6 / NT:.2f}) + had/act {had / 1e6 / NT:.2f} ms/token")
print(f"zero-copy {cn['zero_copy_gb_per_token']:.3f} GB/token (decode.json) / {zc_ex:.2f} ms = {cn['zero_copy_gb_per_token'] / zc_ex * 1e3:.1f} GB/s effective")
for k_, v in calls.items():
    print(f"  early gemv calls {'under' if k_ == 'ov' else 'without'} the lane: {v[1] / NT:.1f}/token, mean slots {v[2] / max(v[1], 1):.2f}, zero-copy excess {v[0] / max(v[1], 1) / 1e3:.1f} us/call")
print(f"  -> zero-copy is {slow:.2f}x slower while the lane runs")
print(f"CPU lane: {lane_ms:.2f} ms/token, {lane_exp:.1f} experts/token, {lane_exp * 9474048 / (lane_ms * 1e6):.1f} GB/s; lane longer than the GPU's routed work: {lane_gt_gpu / 1e6 / NT:.2f} ms/token as measured, {lane_gt_vram / 1e6 / NT:.2f} if the GPU had no zero-copy")
print(f"host-served expert bytes/token: lane {lane_exp * 9474048 / 1e9:.2f} GB + zero-copy {cn['zero_copy_gb_per_token']:.2f} GB")
