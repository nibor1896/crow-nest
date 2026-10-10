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

main()
