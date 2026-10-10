"""Time breakdown of the profile-20261010b nsys traces (glm-integration dfd9201, ARM env).

Re-run:  python -I analysis.py   (from this directory, or pass the directory as argv[1])

Windows
  prefill: the 8192-row prompt row = [end of the first argmax_k - row secs, end of the first argmax_k]
           (row secs from prefill.json reps_detail[0].rows[0].secs; argmax_k runs once per row)
  decode:  the 95 generated rows = (end of argmax_k #0, end of argmax_k #95] in decode.sqlite
Everything below is plain SQL on CUPTI_ACTIVITY_KIND_{KERNEL,MEMCPY,RUNTIME} + StringIds,
intervals merged in Python (union = time at least one item of the set is running).
"""
import json
import os
import sqlite3
import sys

D = sys.argv[1] if len(sys.argv) > 1 else os.path.dirname(os.path.abspath(__file__))

# kernel groups (names from *_stats_cuda_gpu_kern_sum.csv)
SPIN = {"glm5_ctl_wait"}  # one thread spins on a host-mapped reply word (glm5_flags.rs:262)
LANE = {"glm5_lane_merge"}  # copies the CPU lane's host-mapped rows into ye (glm5_flags.rs:253)
GROUPED = {"mul1_gemm_grp"}
EXPERT_GEMV = {"mul1_gemv", "mul1_had_in", "mul1_had_out", "glm5_swiglu_clamp", "glm5_moe_gather", "glm5_moe_combine", "glm5_router_sig_topk", "glm5_pred_tag", "glm5_ctl_publish"}
DENSE = {"glm5_gemm_fp4_tc", "glm5_gemm_fp4_tcs", "gemm_bf16_dense", "gemv_bf16_b", "gemv_bf16_w", "glm5_gemv_fp4", "glm5_gemv_fp4_x3"}
ATTN_PREFIX = ("gm_", "kda_", "conv_", "l2norm", "rmsnorm_gated", "split_qkv", "transpose_rt", "qsa_select")


def group(name):
    if name in SPIN:
        return "spin_wait(ctl_wait)"
    if name in LANE:
        return "cpu_lane_merge"
    if name in GROUPED:
        return "moe_grouped(mul1_gemm_grp)"
    if name in EXPERT_GEMV:
        return "moe_gemv+router"
    if name in DENSE:
        return "dense_gemm/gemv"
    if name.startswith(ATTN_PREFIX):
        return "attn/KDA/MLA"
    return "other"


def union(iv):
    iv = sorted(iv)
    tot, cur_s, cur_e = 0, None, None
    out = []
    for s, e in iv:
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                out.append((cur_s, cur_e))
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        out.append((cur_s, cur_e))
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def clip(iv, a, b):
    return [(max(s, a), min(e, b)) for s, e in iv if e > a and s < b]


def intersect(x, y):
    # both merged and sorted
    i = j = 0
    out = []
    while i < len(x) and j < len(y):
        s, e = max(x[i][0], y[j][0]), min(x[i][1], y[j][1])
        if s < e:
            out.append((s, e))
        if x[i][1] < y[j][1]:
            i += 1
        else:
            j += 1
    return out


def subtract(x, y):
    return length(x) - length(intersect(x, y))


def load(db):
    c = sqlite3.connect(os.path.join(D, db))
    k = c.execute(
        "SELECT k.start, k.end, k.streamId, s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id = k.shortName ORDER BY k.start"
    ).fetchall()
    m = c.execute("SELECT start, end, streamId, copyKind, srcKind, dstKind, bytes FROM CUPTI_ACTIVITY_KIND_MEMCPY ORDER BY start").fetchall()
    r = c.execute(
        "SELECT r.start, r.end, r.globalTid, s.value FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON s.id = r.nameId ORDER BY r.start"
    ).fetchall()
    return k, m, r


def report(tag, k, m, r, a, b, tokens):
    W = b - a
    print(f"\n=== {tag}: window {W / 1e9:.3f} s, {tokens} row(s), {W / tokens / 1e6:.2f} ms per row ===")
    kk = [(max(s, a), min(e, b), st, n) for s, e, st, n in k if e > a and s < b]
    all_k = union([(s, e) for s, e, _, _ in kk])
    real_k = union([(s, e) for s, e, _, n in kk if n not in SPIN])
    spin = union([(s, e) for s, e, _, n in kk if n in SPIN])
    groups = {}
    for s, e, _, n in kk:
        groups.setdefault(group(n), []).append((s, e))
    print("kernel groups (union per group; ms per row; share of window):")
    for g, iv in sorted(groups.items(), key=lambda x: -length(union(x[1]))):
        L = length(union(iv))
        print(f"  {g:28s} {L / 1e6 / tokens:9.2f} ms  {100 * L / W:5.1f} %  ({len(iv)} launches)")
    print(f"  any kernel                   {length(all_k) / 1e6 / tokens:9.2f} ms  {100 * length(all_k) / W:5.1f} %")
    print(f"  kernel except spin-wait      {length(real_k) / 1e6 / tokens:9.2f} ms  {100 * length(real_k) / W:5.1f} %")
    streams = {}
    for s, e, st, n in kk:
        streams.setdefault(st, []).append((s, e))
    print("per stream kernel occupancy:", {st: f"{100 * length(union(iv)) / W:.1f} %" for st, iv in streams.items()})
    # copies
    mm = [(max(s, a), min(e, b), st, ck, sk, dk, by) for s, e, st, ck, sk, dk, by in m if e > a and s < b]
    kinds = {1: "H2D", 2: "D2H", 8: "D2D"}
    for ck in (1, 2, 8):
        sel = [x for x in mm if x[3] == ck]
        if not sel:
            continue
        iv = union([(x[0], x[1]) for x in sel])
        by = sum(x[6] for x in sel)
        ov = length(intersect(iv, real_k))
        per_stream = {}
        for x in sel:
            per_stream.setdefault(x[2], [0, 0])
            per_stream[x[2]][0] += x[6]
            per_stream[x[2]][1] += x[1] - x[0]
        print(
            f"{kinds[ck]}: {by / 1e9:.2f} GB ({by / 1e9 / tokens:.4f} GB/row), busy {length(iv) / 1e6 / tokens:.2f} ms/row "
            f"({100 * length(iv) / W:.1f} %), {by / max(length(iv), 1):.2f} GB/s while busy, "
            f"{100 * ov / max(length(iv), 1):.0f} % of copy time under a non-spin kernel; per stream GB/GBps: "
            + ", ".join(f"s{st} {v[0] / 1e9:.2f}/{v[0] / max(v[1], 1):.1f}" for st, v in per_stream.items())
        )
    # GPU idle = no kernel and no copy
    busy = union([(s, e) for s, e, _, _ in kk] + [(x[0], x[1]) for x in mm])
    idle = W - length(busy)
    print(f"GPU fully idle (no kernel, no copy): {idle / 1e6 / tokens:.2f} ms/row ({100 * idle / W:.1f} %)")
    nothing_real = W - length(union(real_k + [(x[0], x[1]) for x in mm]))
    print(f"GPU without real work (idle or only spin-wait): {nothing_real / 1e6 / tokens:.2f} ms/row ({100 * nothing_real / W:.1f} %)")
    # gaps between real kernels
    gaps = []
    for (s0, e0), (s1, e1) in zip(real_k, real_k[1:]):
        gaps.append(s1 - e0)
    big = sorted(gaps, reverse=True)[:5]
    print("largest gaps between non-spin kernels (ms):", [round(g / 1e6, 1) for g in big])
    # host API time inside the window, by call (sum over threads; blocking calls dominate)
    api = {}
    for s, e, tid, n in r:
        if e > a and s < b:
            api.setdefault(n, [0, 0])
            api[n][0] += min(e, b) - max(s, a)
            api[n][1] += 1
    print("host CUDA API time inside the window (ms per row, calls):")
    for n, (t, c) in sorted(api.items(), key=lambda x: -x[1][0])[:8]:
        print(f"  {n:28s} {t / 1e6 / tokens:9.2f} ms  {c}")
    return groups


def main():
    pj = json.load(open(os.path.join(D, "prefill.json")))
    secs = pj["reps_detail"][0]["rows"][0]["secs"]
    k, m, r = load("prefill.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    b = am[0]
    a = b - int(secs * 1e9)
    g = report("PREFILL 8192-row chunk", k, m, r, a, b, 1)
    grp = [(s, e, n) for s, e, st, n in k if n == "mul1_gemm_grp" and a <= s < b]
    print(f"mul1_gemm_grp in window: {len(grp)} launches, {sum(e - s for s, e, _ in grp) / 1e9:.2f} s")

    dj = json.load(open(os.path.join(D, "decode.json")))
    k, m, r = load("decode.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    a, b = am[0], am[-1]
    report(f"DECODE {len(am) - 1} cold rows", k, m, r, a, b, len(am) - 1)
    c = dj["reps_detail"][0]["decode"]["counters"]
    print(
        "decode tiers per token (decode.json): visits", c["visits_per_token"],
        "vram", round(c["hits"]["vram"] / c["tokens"], 1),
        "pinned", round(c["hits"]["pinned"] / c["tokens"], 1),
        "(cpu lane", round(c["cpu_lane_per_token"], 1), ", zero-copy", round(c["moves"]["zero_copy"] / c["tokens"], 1), ")",
        "nvme", round(c["r_nvme_reads_per_token"], 1),
        "| cpu_lane_s/token", round(c["cpu_lane_s_per_token"] * 1e3, 1), "ms",
        "| zero-copy GB/token", round(c["zero_copy_gb_per_token"], 3),
    )
    pc = pj["reps_detail"][0]["prefill"]["counters"]
    print("prefill tiers (prefill.json):", pc["hits"], "nvme GB", round(pc["nvme_gb_per_token"] * pc["tokens"], 1), "moves n2p/p2n", pc["moves"]["n2p"], pc["moves"]["p2n"])


if __name__ == "__main__":
    main()


def decode_waits():
    """Split the two device waits of a controlled MoE layer (glm5_flags.rs Ctl::wait_reply = #1, the
    tier reply; Ctl::merge_lane = #2, the CPU-lane flag, the wait that is followed by glm5_lane_merge)
    and time the expert GEMVs between them."""
    k, m, r = load("decode.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    a, b = am[0], am[-1]
    T = len(am) - 1
    ks = [x for x in k if x[2] == 7 and a <= x[0] < b]
    w1 = w2 = gemv_between = n1 = n2 = 0
    pending = None
    for i, (s, e, st, n) in enumerate(ks):
        if n == "glm5_ctl_wait":
            nxt = ks[i + 1][3] if i + 1 < len(ks) else ""
            if nxt == "glm5_lane_merge":
                w2 += e - s
                n2 += 1
                if pending is not None:
                    gemv_between += s - pending
                pending = None
            else:
                w1 += e - s
                n1 += 1
                pending = e
    print(f"\n=== DECODE device waits per row ({T} rows) ===")
    print(f"wait #1 tier reply (controller serving the layer: tiers, NVMe, H2D): {w1 / 1e6 / T:.2f} ms ({n1} waits)")
    print(f"wait #2 CPU-lane flag: {w2 / 1e6 / T:.2f} ms ({n2} waits)")
    print(f"device time from reply to lane wait (GPU experts incl. zero-copy, overlaps the CPU lane): {gemv_between / 1e6 / T:.2f} ms")
    # gemv durations split by grid: 16x16x8 (gate/up, 2 per layer) and 32x16x8 (down)
    c = sqlite3.connect(os.path.join(D, "decode.sqlite"))
    q = ("SELECT k.gridX, count(*), sum(k.end-k.start) FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
         "WHERE s.value='mul1_gemv' AND k.start>=? AND k.start<? GROUP BY k.gridX")
    for gx, cnt, tot in c.execute(q, (a, b)):
        print(f"mul1_gemv gridX={gx}: {cnt} launches, {tot / 1e6 / T:.2f} ms/row")


def prefill_gaps():
    """Idle stretches of the prefill row: count and size of GPU gaps > 50 ms (one per MoE layer =
    the host reading the layer's NVMe experts before the grouped GEMM is queued)."""
    pj = json.load(open(os.path.join(D, "prefill.json")))
    secs = pj["reps_detail"][0]["rows"][0]["secs"]
    k, m, r = load("prefill.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    b = am[0]
    a = b - int(secs * 1e9)
    busy = union([(max(s, a), min(e, b)) for s, e, _, _ in k if e > a and s < b] + [(max(s, a), min(e, b)) for s, e, *_ in m if e > a and s < b])
    gaps = [(s1 - e0, e0) for (s0, e0), (s1, e1) in zip(busy, busy[1:]) if s1 - e0 > 50e6]
    nv = pj["reps_detail"][0]["prefill"]["counters"]
    gb = nv["nvme_gb_per_token"] * nv["tokens"]
    tot = sum(g for g, _ in gaps)
    print(f"\n=== PREFILL gaps > 50 ms: {len(gaps)} gaps, {tot / 1e9:.2f} s; NVMe {gb:.1f} GB -> {gb / (tot / 1e9):.2f} GB/s if read only in the gaps ===")
    # what precedes each gap: the last kernel before it
    names = {}
    for g, e0 in gaps:
        last = max((x for x in k if x[1] <= e0 + 1), key=lambda x: x[1])
        names[last[3]] = names.get(last[3], 0) + 1
    print("kernel ending right before each gap:", names)
    # grouped GEMM effective throughput: work items x matrix bytes / time (z = 2 launch: gate+up, 2/3 of the record; z = 1: down, 1/3)
    c = sqlite3.connect(os.path.join(D, "prefill.sqlite"))
    rec = pj["container"]["record_bytes"]
    q = ("SELECT k.gridY, k.gridZ, k.end-k.start FROM CUPTI_ACTIVITY_KIND_KERNEL k JOIN StringIds s ON s.id=k.shortName "
         "WHERE s.value='mul1_gemm_grp' AND k.start>=? AND k.start<?")
    tb = tt = 0
    for gy, gz, d in c.execute(q, (a, b)):
        tb += gy * rec * (2 / 3 if gz == 2 else 1 / 3)
        tt += d
    print(f"mul1_gemm_grp: work items x matrix bytes = {tb / 1e12:.2f} TB in {tt / 1e9:.1f} s = {tb / tt:.1f} GB/s "
          f"(the experts read once would be {gb:.0f} GB)")
    dc = sqlite3.connect(os.path.join(D, "decode.sqlite"))
    dj = json.load(open(os.path.join(D, "decode.json")))
    am2 = [e for s, e, st, n in load("decode.sqlite")[0] if n == "argmax_k"]
    tb = tt = 0
    for gy, gz, d in dc.execute(q, (0, am2[0])):
        tb += gy * rec * (2 / 3 if gz == 2 else 1 / 3)
        tt += d
    print(f"same kernel in the 64-row prompt of decode.sqlite (about one work item per expert): {tb / 1e9:.1f} GB in {tt / 1e9:.2f} s = {tb / tt:.1f} GB/s")


if __name__ == "__main__":
    decode_waits()
    prefill_gaps()


def decode_host_threads():
    """Host CUDA API time per thread inside the decode window, and which calls overlap the device's
    tier-reply waits (#1)."""
    k, m, r = load("decode.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    a, b = am[0], am[-1]
    T = len(am) - 1
    ks = [x for x in k if x[2] == 7 and a <= x[0] < b]
    w1 = []
    for i, (s, e, st, n) in enumerate(ks):
        if n == "glm5_ctl_wait" and not (i + 1 < len(ks) and ks[i + 1][3] == "glm5_lane_merge"):
            w1.append((s, e))
    w1 = union(w1)
    per = {}
    for s, e, tid, n in r:
        if e > a and s < b:
            iv = [(max(s, a), min(e, b))]
            d = per.setdefault(tid, {})
            x = d.setdefault(n, [0, 0, 0])
            x[0] += iv[0][1] - iv[0][0]
            x[1] += 1
            x[2] += length(intersect(iv, w1))
    print(f"\n=== DECODE host API per thread (ms per row; total / inside wait #1) ===")
    for tid, d in sorted(per.items(), key=lambda kv: -sum(v[0] for v in kv[1].values())):
        tot = sum(v[0] for v in d.values())
        if tot / 1e6 / T < 0.5:
            continue
        top = sorted(d.items(), key=lambda kv: -kv[1][0])[:4]
        print(f"tid {tid & 0xffffff}: {tot / 1e6 / T:.1f} ms; " + "; ".join(f"{n} {v[0] / 1e6 / T:.1f}/{v[2] / 1e6 / T:.1f} ({v[1]})" for n, v in top))


if __name__ == "__main__":
    decode_host_threads()


def decode_wait_vs_nvme():
    """Is the tier-reply wait (#1) NVMe time? Per row: wait #1 sum against the row's NVMe reads
    (decode.json rows[].nvme_reads); per MoE layer: wait #1 sum against the layer's NVMe reads and
    prefetch joins (decode.json layers[])."""
    dj = json.load(open(os.path.join(D, "decode.json")))
    rows = dj["reps_detail"][0]["rows"][1:]
    k, m, r = load("decode.sqlite")
    am = [e for s, e, st, n in k if n == "argmax_k"]
    ks = [x for x in k if x[2] == 7 and am[0] <= x[0] < am[-1]]
    w1 = []  # (start, dur)
    for i, (s, e, st, n) in enumerate(ks):
        if n == "glm5_ctl_wait" and not (i + 1 < len(ks) and ks[i + 1][3] == "glm5_lane_merge"):
            w1.append((s, e - s))
    per_row = [0.0] * (len(am) - 1)
    per_layer = [0.0] * 42
    j = 0
    for idx, (s, d) in enumerate(w1):
        while j + 1 < len(am) and s >= am[j + 1]:
            j += 1
        per_row[j] += d / 1e6
        per_layer[idx % 42] += d / 1e6 / len(per_row)
    nv = [x["nvme_reads"] for x in rows]

    def corr(x, y):
        n = len(x)
        mx, my = sum(x) / n, sum(y) / n
        sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
        sx = sum((a - mx) ** 2 for a in x) ** 0.5
        sy = sum((b - my) ** 2 for b in y) ** 0.5
        return sxy / (sx * sy)

    n = len(per_row)
    mx = sum(nv) / n
    slope = sum((a - mx) * (b - sum(per_row) / n) for a, b in zip(nv, per_row)) / sum((a - mx) ** 2 for a in nv)
    icpt = sum(per_row) / n - slope * mx
    print(f"\n=== DECODE wait #1 vs NVMe ===")
    print(f"per row: corr(wait #1 ms, NVMe reads) = {corr(per_row, nv):.2f}; fit wait = {icpt:.1f} ms + {slope:.2f} ms x reads "
          f"(mean {sum(per_row) / n:.1f} ms, {mx:.1f} reads)")
    lay = dj["reps_detail"][0]["decode"]["layers"]
    T = dj["reps_detail"][0]["decode"]["counters"]["tokens"]
    lnv = [x["nvme_reads_per_token"] for x in lay]
    print(f"per layer: corr(wait #1 ms/row, NVMe reads/row) = {corr(per_layer, lnv):.2f}; "
          f"min/median/max wait per layer {min(per_layer):.2f}/{sorted(per_layer)[21]:.2f}/{max(per_layer):.2f} ms")


if __name__ == "__main__":
    decode_wait_vs_nvme()
