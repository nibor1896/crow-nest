"""#196 Stage 3: time budget of one cold 8192-row prefill (glm-integration 28f459a, arm2.env + CROW_GLM_RT2=1).

Re-run:  python -I analysis.py [dir]      (reads prof1.sqlite + prof1.json in this directory)

Window: the prompt row = [end of the first argmax_k - row secs, end of the first argmax_k]
(row secs = prof1.json reps_detail[0].rows[0].secs; it is the same clock glm5_run reports).

Method (no API durations are used as causes):
* Stream 7 (compute) carries every kernel; its busy time is split by kernel group.
* Every idle stretch of stream 7 before op X is attributed to the LATEST of the things X could
  have waited for: the previous op's end, the host submitting X (API end of X's launch), and the
  completion (device timestamp, CUDA_EVENT) of every event a cuStreamWaitEvent on stream 7 made
  X wait for (CUPTI SYNCHRONIZATION type 2 -> eventId/eventSyncId -> CUDA_EVENT).
* An event recorded on stream 17 (the prefill copy stream) is a sub-batch's "copies landed";
  that wait is then split by what stream 17 was doing in the same stretch: H2D busy, or idle
  before a ring copy whose start is later than everything known (= the ring copy's
  cuStreamWaitValue64 on its NVMe landed flag), or idle before a copy the host had not yet
  submitted, or idle waiting for a staging half to be freed by the compute stream.
* A host-submit-late stretch is split by what the host thread did right before the submit:
  blocked in cuStreamSynchronize / cuEventSynchronize (the GPU it waited for is idle by
  construction, so the stretch after the block is host CPU work), or CPU work before the first
  kernel (embedding rows read from the container on the host).
"""
import json
import os
import sqlite3
import sys
from bisect import bisect_left, bisect_right
from collections import defaultdict

D = sys.argv[1] if len(sys.argv) > 1 and os.path.isdir(sys.argv[1]) else os.path.dirname(os.path.abspath(__file__))
TAG = "prof1"
MAIN_TID = None  # filled from ThreadNames ('main')


def union(iv):
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect(x, y):
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        s, e = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if s < e:
            out.append([s, e])
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


ATTN = ("gm_", "kda_", "conv_", "l2norm", "rmsnorm_gated", "split_qkv", "transpose_rt", "qsa_select")
EXPERT = ("mul1_", "glm5_swiglu_clamp", "glm5_moe_", "glm5_router_sig_topk")


def group(n, gx=None):
    if n.startswith(ATTN):
        return "attention (KDA/MLA/DSA kernels)"
    if n.startswith("glm5_mhc"):
        return "mHC mix/expand"
    if n == "gemv_bf16_b":
        return "router logits (gemv_bf16_b)"
    if n.startswith(EXPERT):
        return "routed experts (mul1_tc2 + combine)"
    if n in ("glm5_gemm_fp4_tc", "gemm_bf16_dense", "glm5_gemv_fp4", "glm5_gemv_fp4_x3", "gemv_bf16_w"):
        return "dense/shared GEMMs (NVFP4 TC + BF16)"
    return "head/other"


def main():
    global MAIN_TID
    c = sqlite3.connect(os.path.join(D, TAG + ".sqlite"))
    pj = json.load(open(os.path.join(D, TAG + ".json")))
    secs = pj["reps_detail"][0]["rows"][0]["secs"]
    names = {t: n for n, t in c.execute("select s.value,t.globalTid from ThreadNames t join StringIds s on s.id=t.nameId")}
    MAIN_TID = [t for t, n in names.items() if n == "main"][0]
    am = c.execute("select k.end from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where s.value='argmax_k' order by k.start").fetchall()
    b = am[0][0]
    a = b - int(secs * 1e9)
    W = b - a
    api = {}
    host = []  # main thread api calls (start, end, name)
    for s, e, tid, cid, n in c.execute("select r.start,r.end,r.globalTid,r.correlationId,s.value from CUPTI_ACTIVITY_KIND_RUNTIME r join StringIds s on s.id=r.nameId where r.end>? and r.start<?", (a - 2 * 10**9, b)):
        api[cid] = (s, e, tid, n)
        if tid == MAIN_TID:
            host.append((s, e, n, cid))
    host.sort()
    ops = defaultdict(list)  # stream -> [start,end,kind,name,cid,bytes]
    for s, e, st, n, cid in c.execute("select k.start,k.end,k.streamId,s.value,k.correlationId from CUPTI_ACTIVITY_KIND_KERNEL k join StringIds s on s.id=k.shortName where k.end>? and k.start<?", (a, b)):
        ops[st].append([s, e, "K", n, cid, 0])
    for s, e, st, ck, by, cid, sk in c.execute("select start,end,streamId,copyKind,bytes,correlationId,srcKind from CUPTI_ACTIVITY_KIND_MEMCPY where end>? and start<?", (a, b)):
        ops[st].append([s, e, "C", f"cp{ck}{'pg' if sk == 0 else ''}", cid, by])
    for s, e, st, by, cid in c.execute("select start,end,streamId,bytes,correlationId from CUPTI_ACTIVITY_KIND_MEMSET where end>? and start<?", (a, b)):
        ops[st].append([s, e, "M", "memset", cid, by])
    for st in ops:
        ops[st].sort()
    # events: (eventId, eventSyncId) -> (completion ts, stream)
    evc = {}
    for ts, st, cid, eid, esid in c.execute("select timestamp,streamId,correlationId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_CUDA_EVENT where timestamp>? and timestamp<?", (a - 2 * 10**9, b + 10**9)):
        evc[(eid, esid)] = (ts, st, api.get(cid, (0, 0))[0])
    # stream waits: stream -> sorted [(api_end, completion ts, src stream, eventId)]
    waits = defaultdict(list)
    for s, e, st, cid, eid, esid in c.execute("select start,end,streamId,correlationId,eventId,eventSyncId from CUPTI_ACTIVITY_KIND_SYNCHRONIZATION where syncType=2 and end>? and start<?", (a - 10**9, b)):
        ap = api.get(cid)
        ec = evc.get((eid, esid))
        if ap and ec:
            waits[st].append((ap[1], ec[0], ec[1], eid))
    for st in waits:
        waits[st].sort()

    # ----- ring vs pinned H2D on stream 17: a ring copy is enqueued right after a cuStreamWaitValue64
    hseq = [x for x in host]
    ring_cids = set()
    for i, (s, e, n, cid) in enumerate(hseq):
        if n == "cuMemcpyHtoDAsync_v2" and i > 0 and hseq[i - 1][2] == "cuStreamWaitValue64_v2":
            ring_cids.add(cid)
    # submit (API end) of every op
    def submit(op):
        x = api.get(op[4])
        return x[1] if x else 0

    print(f"window {W / 1e9:.4f} s (row secs {secs}); profiled TTFT {pj['reps_detail'][0]['prefill']['timing']['ttft_s']:.3f} s")
    # ================= stream 17 state per instant =================
    s17 = [o for o in ops.get(17, []) if o[0] < b]
    st17 = []  # [start, end, state]
    prev_end = a
    prev_sub = 0
    nring = npin = 0
    by_ring = by_pin = 0
    ring_busy = []
    pin_busy = []
    for o in s17:
        is_ring = o[4] in ring_cids
        if o[2] == "C":
            if is_ring:
                nring += 1; by_ring += o[5]; ring_busy.append([o[0], o[1]])
            else:
                npin += 1; by_pin += o[5]; pin_busy.append([o[0], o[1]])
        gap0 = max(prev_end, a)
        if o[0] > gap0:
            sub = submit(o)
            evs = [w for w in waits.get(17, []) if prev_sub <= w[0] <= sub]
            ev_ready = max([w[1] for w in evs], default=0)
            known = max(gap0, sub, ev_ready)
            # split the gap: up to min(known, start) by the binding known cause, the rest = NVMe landed flag (ring) / unexplained
            cut = min(known, o[0])
            if cut > gap0:
                if sub >= ev_ready and sub > gap0:
                    st17.append([gap0, cut, "host not yet submitted"])
                elif ev_ready > gap0:
                    st17.append([gap0, cut, "staging half not yet free (compute)"])
            if o[0] > cut:
                st17.append([max(cut, gap0), o[0], "NVMe landed flag (ring copy)" if is_ring else "unexplained"])
        st17.append([max(o[0], a), min(o[1], b), "H2D ring->stage" if is_ring else ("H2D pinned->stage" if o[2] == "C" else o[3])])
        prev_end = max(prev_end, o[1])
        prev_sub = submit(o)
    if prev_end < b:
        st17.append([prev_end, b, "no copy queued"])
    print(f"stream 17: {nring} ring copies {by_ring / 1e9:.2f} GB, {npin} pinned copies {by_pin / 1e9:.2f} GB; H2D busy {length(union(ring_busy + pin_busy)) / 1e9:.3f} s "
          f"({(by_ring + by_pin) / max(length(union(ring_busy + pin_busy)), 1):.1f} GB/s while busy)")
    agg = defaultdict(int)
    for s, e, k in st17:
        agg[k] += e - s
    print("stream 17 state over the window:", {k: round(v / 1e9, 3) for k, v in sorted(agg.items(), key=lambda x: -x[1])})

    def state17_in(s, e):
        out = defaultdict(int)
        i = bisect_right([x[0] for x in st17], s) - 1
        i = max(i, 0)
        while i < len(st17) and st17[i][0] < e:
            x0, x1, k = st17[i]
            ov = min(x1, e) - max(x0, s)
            if ov > 0:
                out[k] += ov
            i += 1
        rest = (e - s) - sum(out.values())
        if rest > 0:
            out["no copy queued"] += rest
        return out

    # ================= host (main thread) state per instant =================
    hst = []  # [start, end, label]
    for (s, e, n, cid), nxt in zip(host, host[1:] + [(b, b, "", 0)]):
        if n in ("cuStreamSynchronize", "cuEventSynchronize", "cuMemcpyDtoH_v2", "cuCtxSynchronize") and e - s > 50_000:
            hst.append([s, e, "host blocked in " + n])
        else:
            hst.append([s, e, "host issuing CUDA calls"])
        if nxt[0] - e > 1_000_000:
            hst.append([e, nxt[0], "host without CUDA calls (CPU or synchronous NVMe read)"])
        elif nxt[0] > e:
            hst.append([e, nxt[0], "host issuing CUDA calls"])
    hst.sort()
    hs0 = [x[0] for x in hst]

    def host_in(s, e):
        out = defaultdict(int)
        i = max(bisect_right(hs0, s) - 1, 0)
        while i < len(hst) and hst[i][0] < e:
            x0, x1, k = hst[i]
            ov = min(x1, e) - max(x0, s)
            if ov > 0:
                out[k] += ov
            i += 1
        rest = (e - s) - sum(out.values())
        if rest > 0:
            out["host issuing CUDA calls"] += rest
        return out

    def copy_in(s, e):
        """stream 17 state in [s, e]; its 'not yet submitted' parts resolved by the host state"""
        out = defaultdict(int)
        for k, v in state17_in(s, e).items():
            out[k] += v
        res = defaultdict(int)
        for k, v in out.items():
            if k == "NVMe landed flag (ring copy)":
                pass
            res[k] += v
        return res

    # relabel stream-17 waits before ring copies: > 0.3 ms = the NVMe landing, else the memop pair
    for x in st17:
        if x[2] == "NVMe landed flag (ring copy)" and x[1] - x[0] <= 300_000:
            x[2] = "ring memop pair (WaitValue64+WriteValue64), record already landed"
    # ================= stream 7: busy by group, idle by cause =================
    s7 = [o for o in ops[7]]
    routers = [o[0] for o in s7 if o[3] == "glm5_router_sig_topk"]
    phases = [("row start -> l3 router", a, routers[0]), ("l3-l6", routers[0], routers[4]), ("l7-l44 + head", routers[4], b)]

    def phase_of(t):
        for nme, lo, hi in phases:
            if lo <= t < hi:
                return nme
        return phases[-1][0]

    budget = defaultdict(int)
    budget_ph = defaultdict(lambda: defaultdict(int))
    for o in s7:
        g = group(o[3]) if o[2] == "K" else ("H2D on compute stream (pageable)" if o[3].endswith("pg") else "small copies/memsets on compute stream")
        s_, e_ = max(o[0], a), min(o[1], b)
        if e_ > s_:
            budget[g] += e_ - s_
            budget_ph[phase_of(s_)][g] += e_ - s_
    idle = defaultdict(int)
    idle_ph = defaultdict(lambda: defaultdict(int))
    idle_detail = []
    prev_end, prev_sub = a, 0

    def add(k, v, t):
        if v > 0:
            idle[k] += v
            idle_ph[phase_of(t)][k] += v

    for o in s7:
        gap0 = max(prev_end, a)
        if o[0] > gap0:
            sub = submit(o)
            evs = [w for w in waits.get(7, []) if prev_sub <= w[0] <= sub]
            ev = max(evs, key=lambda w: w[1], default=None)
            ev_ready = ev[1] if ev else 0
            gap_e = min(o[0], b)
            if ev and ev_ready >= sub and ev_ready > gap0:
                cut = min(ev_ready, gap_e)
                for k, v in state17_in(gap0, cut).items():
                    if k in ("host not yet submitted", "no copy queued"):
                        # find the host state in the part of the stretch the copy stream sat empty
                        pass
                    add("copy stream: " + k, v, gap0)
                add("launch latency after the event", gap_e - cut, gap0)
                idle_detail.append((gap_e - gap0, gap0, "copy event", o[3]))
            elif sub > gap0:
                cut = min(sub, gap_e)
                for k, v in host_in(gap0, cut).items():
                    add(k, v, gap0)
                add("launch latency", gap_e - cut, gap0)
                idle_detail.append((gap_e - gap0, gap0, "host submit", o[3]))
            else:
                add("unexplained (memop wait / scheduler)", gap_e - gap0, gap0)
                idle_detail.append((gap_e - gap0, gap0, "unexpl", o[3]))
        prev_end = max(prev_end, o[1])
        prev_sub = submit(o)
    if prev_end < b:
        add("after the last op", b - prev_end, prev_end)
    # copy stream 'host not yet submitted' parts: split by host state (second pass over st17)
    hn = defaultdict(int)
    for s_, e_, k in st17:
        if k == "host not yet submitted":
            for kk, v in host_in(s_, e_).items():
                hn[kk] += v
    tot_b = sum(budget.values())
    tot_i = sum(idle.values())
    print(f"\nstream 7 busy {tot_b / 1e9:.3f} s + idle {tot_i / 1e9:.3f} s = {(tot_b + tot_i) / 1e9:.3f} s (window {W / 1e9:.3f} s)")
    print("busy by group:")
    for g, v in sorted(budget.items(), key=lambda x: -x[1]):
        print(f"  {g:60s} {v / 1e9:7.3f} s")
    print("idle by the event waited on:")
    for g, v in sorted(idle.items(), key=lambda x: -x[1]):
        print(f"  {g:90s} {v / 1e9:7.3f} s")
    for nme, lo, hi in phases:
        bb = sum(budget_ph[nme].values()); ii = sum(idle_ph[nme].values())
        print(f"\nphase {nme}: {(hi - lo) / 1e9:.3f} s = busy {bb / 1e9:.3f} + idle {ii / 1e9:.3f}")
        print("  busy: " + "; ".join(f"{k} {v / 1e9:.3f}" for k, v in sorted(budget_ph[nme].items(), key=lambda x: -x[1])))
        print("  idle: " + "; ".join(f"{k} {v / 1e9:.3f}" for k, v in sorted(idle_ph[nme].items(), key=lambda x: -x[1]) if v > 1e6))
    print("\ncopy stream 'host not yet submitted' by host state:", {k: round(v / 1e9, 3) for k, v in hn.items()})
    idle_detail.sort(reverse=True)
    print("largest idle stretches (ms, at s, cause, next op):")
    for d, s, k, n in idle_detail[:12]:
        print(f"  {d / 1e6:8.2f} ms at {(s - a) / 1e9:6.3f} s  {k:12s} {n}")
    return locals()


if __name__ == "__main__":
    main()
