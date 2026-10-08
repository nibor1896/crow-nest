#!/usr/bin/env python3
"""#147 (GLM-5.3-Flash step 8, gate G1): offline tier simulation over the layerwise runner's routing.

  # 1. corpus: Crow sessions through GLM's own chat template and tokenizer (oracle venv; reads
  #    downloaded files, so -I). Writes <name>-ids.json, <name>-mask.json and corpus.json.
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py corpus \
      --model models/GLM-5.3-Flash-original --out <dir> --cap 32768 --held <name> \
      --file <name> <task> <session.json> [--file ...]

  # 2. routing (step 7 runner on the dequantised FULL container, one out dir per corpus name):
  #    .venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py [--dry-run]  (the five passes, #147)

  # 3. simulation and the G1 verdict fields
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py sim --corpus <dir>/corpus.json --runs <runs> \
      [--step3 runs/glm53-flash/step03/<run>.json] [--windows 0,8,16,32] [--json <out.json>]

Policy per MoE layer (PREREG G1, ticket #147): the experts are ranked by their routed count over the
GENERATED positions of the calibration files (the `G` rule of #106: frequency order, ties lower id);
ranks 0..N-1 are the VRAM hot set, the next 83 the pinned tier, the rest live on the NVMe. With a
window of W slots per layer, an NVMe-tier expert read for one token stays in a W-slot LRU window
(Eliseev & Mazur, arXiv:2312.17238 §3.1) and a later visit to it is not an NVMe read. Belady's MIN
(with bypass) over the same window is printed as the ceiling no replacement policy can beat, never as
the policy. LRU is a stack algorithm (Mattson et al. 1970): its NVMe reads never grow with W, so
m at W = 0 bounds m at every W.

m = NVMe reads / routed visits per token (42 x 8 = 336), primary on the generated positions of the
held-out file, 95 % block bootstrap (non-overlapping 1,000-token blocks of consecutive generated
positions, 2,000 resamples; the form of tools/hotset-eval.py:80-86). G1 at N = 25: CI upper bound of
m <= m* = B / 190.3 GB/s AND 39.9 x B >= 150 tok/s. B comes only from a step-3 run of record whose
best reader count holds the 1.15 spread; without it the verdict is "G1 not answered".
"""
import argparse
import hashlib
import heapq
import importlib.util
import json
import math
import os
import statistics
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

SHAPE = (42, 288, 8)            # MoE layers 3..44, routed experts, top-k (config.json rev eb9eb208)
PINNED = 83                     # pinned tier per layer: 46 GiB / (42 x 13.5 MiB) = 83.07
N_RANGE = tuple(range(25, 38))  # hot set per layer, derived 25..37
GATE_N = 25
REPORT_N = (25, 31, 37)
BLOCK, RESAMPLES, SEED = 1000, 2000, 20261008
DEPTHS = ((0, 32000), (32000, 100000), (100000, None))
EXPERT_BYTES = 14_155_776
TOKEN_BYTES = 336 * EXPERT_BYTES            # 4,756,340,736 B if every visit came from disk
TOKEN_GBPS = 190.3                          # 40 tok/s x 4.7563 GB (PREREG G1)
PREFILL_FACTOR, PREFILL_LINE = 39.9, 150.0  # PREREG G1 / G5 cold-prefill line
SPREAD_MAX = 1.15                           # PREREG step 3
# Device-issued ceiling 51.6 GB/s and copy engine 54.6 GB/s: docs/architecture.md:1414. The default staging
# kernel is stage_cold_ca (CROW_STAGE_KERNEL default 2, engine/src/gen.rs); stage_cold (31.5 GB/s) is the
# CROW_STAGE_KERNEL=1 fallback. #167
STAGE_CEILINGS = (("stage_cold_ca kernel, default", 51.6), ("copy engine, opt-in", 54.6))
VRAM, PIN, NVME = 0, 1, 2


class SimError(Exception):
    """A refusal: the input cannot give a number of record."""


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def jdump(obj, path, **kw):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, **kw)


def sha256_ids(ids):
    return hashlib.sha256(json.dumps([int(i) for i in ids]).encode()).hexdigest()


# ------------------------------------------------------------------ inputs

def load_runner(out_dir, shape=SHAPE):
    """The runner's per-layer routing (docs/glm5-reference-runner.md §5) -> (manifest, uint16 [N][L][K]).
    Self-test: every routing file's sha256 equals the manifest's record, its shape is [N][K], every id
    lies in 0..E-1 and every row is strictly ascending (eight distinct experts, sorted as the runner
    writes them)."""
    L, E, K = shape
    man = jload(os.path.join(out_dir, "manifest.json"))
    kinds = man.get("layer_kinds") or []
    moe = [l for l, k in enumerate(kinds) if k.endswith("+moe")]
    if len(moe) != L:
        raise SimError("%s: %d MoE layers in layer_kinds, expected %d" % (out_dir, len(moe), L))
    n = len(man["ids"])
    routes = np.empty((n, L, K), np.uint16)
    for j, l in enumerate(moe):
        nm = "l%d-routing-ids.i32" % l
        rec = man.get("files", {}).get(nm)
        path = os.path.join(out_dir, nm)
        if rec is None or not os.path.exists(path):
            raise SimError("%s: %s missing (run not through layer %d)" % (out_dir, nm, l))
        if sha256_file(path) != rec["sha256"]:
            raise SimError("self-test: %s/%s sha256 differs from the manifest" % (out_dir, nm))
        a = np.fromfile(path, "<i4")
        if a.size != n * K or list(rec.get("shape", [n, K])) != [n, K]:
            raise SimError("self-test: %s/%s holds %d ids, expected [%d][%d]" % (out_dir, nm, a.size, n, K))
        a = a.reshape(n, K)
        if a.min() < 0 or a.max() >= E:
            raise SimError("self-test: %s/%s has an id outside 0..%d" % (out_dir, nm, E - 1))
        if K > 1 and not (np.diff(a, axis=1) > 0).all():
            raise SimError("self-test: %s/%s has a row that is not %d distinct ascending ids" % (out_dir, nm, K))
        routes[:, j, :] = a
    return man, routes


def source_reasons(man):
    """Why this run is not the routing source of record (PREREG G1), empty if it is."""
    out = []
    w = man.get("weights") or {}
    if str(w.get("weights", "")).split(" ")[0] != "cnq":
        out.append("routing not from the CNQ container (FP8 originals or other): plausibility only")
    if w.get("partial"):
        out.append("partial container (%s): plausibility only" % (w["partial"].get("filter") or "partial"))
    if not man.get("complete") or list(man.get("layers", [])) != [0, man.get("num_hidden_layers", -1)]:
        out.append("runner pass not complete over layers 0..%s" % man.get("num_hidden_layers"))
    return out


def gen_mask(path, n):
    m = jload(path)
    if m["tokens"] != n:
        raise SimError("%s: mask for %d tokens, routing has %d" % (path, m["tokens"], n))
    mask = np.zeros(n, bool)
    for a, b in m["spans"]:
        mask[a:b] = True
    return mask


def b_from_step3(path, readers=None):
    """(B in GB/s or None, reason). B only from a valid step-3 run whose best reader count (best median)
    holds the 1.15 spread, re-checked here from the raw repetitions. With `readers` (the count a PREREG
    amendment fixes; amendment 5: 1, #146) B is that count's median under the same validity and spread
    checks, and the run's own b_for_g1 is not used."""
    if not path:
        return None, "no step-3 run given"
    d = jload(path)
    v = d.get("verdict") or {}
    if not v.get("valid"):
        return None, "step-3 run VOID: %s" % "; ".join(v.get("void_reasons") or ["no verdict"])
    per = v.get("per_readers") or {}
    if not per:
        return None, "step-3 run has no reader counts"
    med = {r: statistics.median(p["rates_gbps"]) for r, p in per.items()}
    if readers is not None:
        r = str(readers)
        if r not in per:
            return None, "step-3 run has no %s-reader cell" % r
        rates = per[r]["rates_gbps"]
        spread = max(rates) / min(rates) if min(rates) > 0 else math.inf
        if spread > SPREAD_MAX:
            return None, "spread %.3f > %.2f at the fixed reader count %s" % (spread, SPREAD_MAX, r)
        return med[r], "B = %.3f GB/s at %s reader(s), fixed by PREREG amendment, spread %.3f (%s)" % (
            med[r], r, spread, os.path.basename(path))
    best = max(med, key=lambda r: med[r])
    rates = per[best]["rates_gbps"]
    spread = max(rates) / min(rates) if min(rates) > 0 else math.inf
    if spread > SPREAD_MAX:
        return None, "spread %.3f > %.2f at the best reader count %s (step 3: cache or driver noise)" % (
            spread, SPREAD_MAX, best)
    b = v.get("b_for_g1")
    if b is None:
        return None, v.get("g1_input") or "step-3 run carries no B for G1"
    if str(v.get("best_readers")) != best or abs(b - med[best]) > 1e-9:
        return None, "b_for_g1 %.3f is not the median %.3f of the best reader count %s" % (b, med[best], best)
    return b, "B = %.3f GB/s at %s reader(s), spread %.3f (%s)" % (b, best, spread, os.path.basename(path))


class Run:
    def __init__(self, name, task, routes, gen, man=None):
        self.name, self.task, self.routes, self.gen, self.man = name, task, routes, gen, man


def load_corpus(corpus_path, runs_dir, shape=SHAPE):
    """corpus.json + one runner out dir per name -> (held Run, [cal Run], [reasons the source is not of record])."""
    c = jload(corpus_path)
    base = os.path.dirname(os.path.abspath(corpus_path))
    check_corpus(c)
    runs, reasons = {}, []
    for f in c["files"]:
        man, routes = load_runner(os.path.join(runs_dir, f["name"]), shape)
        ids = jload(os.path.join(base, f["name"] + "-ids.json"))
        if man["ids"] != ids:
            raise SimError("%s: the runner's ids are not %s-ids.json of the corpus" % (f["name"], f["name"]))
        if sha256_ids(ids) != f["ids_sha256"]:
            raise SimError("%s-ids.json does not match corpus.json (sha256)" % f["name"])
        gen = gen_mask(os.path.join(base, f["name"] + "-mask.json"), len(routes))
        reasons += ["%s: %s" % (f["name"], r) for r in source_reasons(man)]
        runs[f["name"]] = Run(f["name"], f["task"], routes, gen, man)
        print("self-test %-24s positions %7d generated %6d  routing files == manifest sha256: True"
              % (f["name"], len(routes), gen.sum()))
    held = runs[c["held"]]
    cal = [runs[f["name"]] for f in c["files"] if f["role"] == "cal"]
    return held, cal, reasons


def check_corpus(c):
    """The held-out is one file of a task no calibration file has, never a calibration file."""
    names = [f["name"] for f in c["files"]]
    if len(set(names)) != len(names):
        raise SimError("corpus: a name appears twice")
    held = [f for f in c["files"] if f["role"] == "held"]
    cal = [f for f in c["files"] if f["role"] == "cal"]
    if len(held) != 1 or held[0]["name"] != c["held"] or len(held) + len(cal) != len(names):
        raise SimError("corpus: exactly one held-out file (role 'held', named by 'held'), the rest 'cal'")
    if not cal:
        raise SimError("corpus: no calibration file")
    h = held[0]
    for f in cal:
        if f["task"] == h["task"]:
            raise SimError("corpus: held-out task %r is also the task of calibration file %s" % (h["task"], f["name"]))
        if f["ids_sha256"] == h["ids_sha256"] or f["source_sha256"] == h["source_sha256"]:
            raise SimError("corpus: held-out %s is the same file as calibration file %s" % (h["name"], f["name"]))


# ------------------------------------------------------------------ the policy

def gen_counts(runs, shape=SHAPE):
    L, E, _ = shape
    c = np.zeros((L, E), np.int64)
    for r in runs:
        sel = r.routes[r.gen]
        for l in range(L):
            c[l] += np.bincount(sel[:, l, :].ravel(), minlength=E)
    return c


def rank_table(counts):
    """[L][E] -> rank of every expert in its layer: count descending, ties lower id (stable sort)."""
    L, E = counts.shape
    order = np.argsort(-counts, axis=1, kind="stable")
    inv = np.empty((L, E), np.int16)
    inv[np.arange(L)[:, None], order] = np.arange(E, dtype=np.int16)[None, :]
    return inv


def cut(cal_runs, held=None, shape=SHAPE):
    """The rank table of record: generated-position counts of the calibration files only."""
    if held is not None and any(r is held or r.name == held.name for r in cal_runs):
        raise SimError("the held-out file %s is in the calibration set" % held.name)
    return rank_table(gen_counts(cal_runs, shape))


def visit_ranks(routes, inv):
    L = routes.shape[1]
    return inv[np.arange(L)[None, :, None], routes]          # [N][L][K]


def shares(rk, n, pinned=PINNED):
    """Per position: (VRAM, pinned, NVMe) shares of the 336 visits at W = 0."""
    flat = rk.reshape(len(rk), -1)
    v = (flat < n).mean(1)
    nv = (flat >= n + pinned).mean(1)
    return v, 1.0 - v - nv, nv


def lru_reads(routes, nvme, w):
    """Per position NVMe reads with a W-slot LRU window per layer over the NVMe-tier visits, token order
    over every position of the file (prompt and generated)."""
    n, L, _ = routes.shape
    reads = nvme.reshape(n, -1).sum(1).astype(np.int64)
    if w == 0:
        return reads
    reads[:] = 0
    for l in range(L):
        pos, k = np.nonzero(nvme[:, l, :])
        ids = routes[pos, l, k].tolist()
        miss = np.zeros(len(ids), bool)
        win = OrderedDict()
        for i, e in enumerate(ids):
            if e in win:
                win.move_to_end(e)
            else:
                miss[i] = True
                if len(win) >= w:
                    win.popitem(last=False)
                win[e] = None
        np.add.at(reads, pos[miss], 1)
    return reads


def min_reads(routes, nvme, w):
    """Belady's MIN with bypass over a W-slot window per layer: the fewest NVMe reads any replacement
    policy can reach with W slots (a ceiling, never the operating policy)."""
    n, L, _ = routes.shape
    reads = nvme.reshape(n, -1).sum(1).astype(np.int64)
    if w == 0:
        return reads
    reads[:] = 0
    for l in range(L):
        pos, k = np.nonzero(nvme[:, l, :])
        ids = routes[pos, l, k].tolist()
        m = len(ids)
        nxt, last = [0] * m, {}
        for i in range(m - 1, -1, -1):
            nxt[i] = last.get(ids[i], m + i)          # never again: beyond the end, unique per access
            last[ids[i]] = i
        cache, heap = {}, []
        miss = np.zeros(m, bool)
        for i, e in enumerate(ids):
            nu = nxt[i]
            if e in cache:
                cache[e] = nu
                heapq.heappush(heap, (-nu, e))
                continue
            miss[i] = True
            if len(cache) < w:
                cache[e] = nu
                heapq.heappush(heap, (-nu, e))
                continue
            while cache.get(heap[0][1]) != -heap[0][0]:
                heapq.heappop(heap)
            far = -heap[0][0]
            if nu >= far:
                continue                              # bypass: needed no sooner than anything held
            del cache[heapq.heappop(heap)[1]]
            cache[e] = nu
            heapq.heappush(heap, (-nu, e))
        np.add.at(reads, pos[miss], 1)
    return reads


# ------------------------------------------------------------------ statistics

def boot_ci(x):
    """95 % block bootstrap of the mean: non-overlapping BLOCK-position blocks in order, RESAMPLES
    resamples, fixed seed (tools/hotset-eval.py:80-86). Fewer than two blocks: no CI."""
    x = np.asarray(x, float)
    nb = len(x) // BLOCK
    if nb < 2:
        return math.nan, math.nan
    blocks = x[:nb * BLOCK].reshape(nb, BLOCK).mean(1)
    idx = np.random.default_rng(SEED).integers(0, nb, size=(RESAMPLES, nb))
    lo, hi = np.percentile(blocks[idx].mean(1), [2.5, 97.5])
    return float(lo), float(hi)


def stat(x):
    lo, hi = boot_ci(x)
    return {"mean": float(np.mean(x)) if len(x) else math.nan, "ci": [lo, hi], "n": int(len(x))}


def g1_verdict(m_stat, b, b_reason, reasons):
    """The G1 verdict fields at N = 25: passed / failed / not answered, with the numbers."""
    out = {"n": GATE_N, "m": m_stat["mean"], "m_ci_hi": m_stat["ci"][1], "generated": m_stat["n"],
           "B": b, "B_source": b_reason, "m_star": None, "prefill_bound_tok_s": None}
    void = list(reasons)
    if b is None:
        void.append("B: " + b_reason)
    else:
        out["m_star"] = b / TOKEN_GBPS
        out["prefill_bound_tok_s"] = PREFILL_FACTOR * b
    if math.isnan(m_stat["ci"][1]):
        void.append("held-out has %d generated positions, fewer than two blocks of %d: no CI"
                    % (m_stat["n"], BLOCK))
    if void:
        out["verdict"] = "G1 not answered: " + "; ".join(void)
        return out
    fails = []
    if out["m_ci_hi"] > out["m_star"]:
        fails.append("m CI upper bound %.4f > m* %.4f" % (out["m_ci_hi"], out["m_star"]))
    if out["prefill_bound_tok_s"] < PREFILL_LINE:
        fails.append("prefill bound %.1f < %.0f tok/s" % (out["prefill_bound_tok_s"], PREFILL_LINE))
    out["verdict"] = "G1 failed: " + "; ".join(fails) if fails else (
        "G1 passed: m CI upper bound %.4f <= m* %.4f, prefill bound %.1f >= %.0f tok/s"
        % (out["m_ci_hi"], out["m_star"], out["prefill_bound_tok_s"], PREFILL_LINE))
    return out


def simulate(held, cal, b=None, b_reason="no step-3 run given", reasons=(), windows=(0, 8, 16, 32),
             gate_window=0, shape=SHAPE, out=print):
    """Every number of the step-8 report; returns them as a dict (the --json document)."""
    if gate_window not in windows:
        windows = tuple(sorted(set(windows) | {gate_window}))
    inv = cut(cal, held, shape)
    rk = visit_ranks(held.routes, inv)
    g = held.gen
    pos = np.arange(len(g))
    res = {"held": held.name, "cal": [r.name for r in cal], "positions": int(len(g)), "generated": int(g.sum()),
           "pinned_per_layer": PINNED, "block": BLOCK, "resamples": RESAMPLES, "seed": SEED}
    out("\nheld-out %s (%s): %d positions, %d generated; cut on the generated positions of %s"
        % (held.name, held.task, len(g), g.sum(), ", ".join(r.name for r in cal)))

    res["by_n"] = {}
    out("\nW = 0, generated positions (primary): visits per token on VRAM / pinned / NVMe, m with 95 % CI")
    for n in N_RANGE:
        v, p, nv = shares(rk, n)
        s = stat(nv[g])
        row = {"vram": float(v[g].mean()), "pinned": stat(p[g]), "m": s,
               "m_prefill": float(nv[~g].mean()) if (~g).any() else math.nan, "m_all": float(nv.mean())}
        res["by_n"][n] = row
        out("  N %2d  VRAM %.4f  pinned %.4f [%.4f, %.4f]  m %.4f [%.4f, %.4f]   m prefill positions %.4f  all %.4f"
            % (n, row["vram"], row["pinned"]["mean"], *row["pinned"]["ci"], s["mean"], *s["ci"],
               row["m_prefill"], row["m_all"]))

    res["windows"] = {}
    out("\nNVMe window, W slots per layer (LRU = the policy; MIN = ceiling), m on generated positions")
    for n in REPORT_N:
        nvme = rk >= n + PINNED
        for w in windows:
            lru = lru_reads(held.routes, nvme, w) / rk[0].size
            s = stat(lru[g])
            row = {"lru": s}
            line = "  N %2d  W %3d  LRU m %.4f [%.4f, %.4f]" % (n, w, s["mean"], *s["ci"])
            if w:
                mn = min_reads(held.routes, nvme, w) / rk[0].size
                row["min_mean"] = float(mn[g].mean())
                line += "   MIN ceiling %.4f" % row["min_mean"]
            res["windows"]["%d/%d" % (n, w)] = row
            out(line)

    nvme = rk >= GATE_N + PINNED
    m_gate = lru_reads(held.routes, nvme, gate_window) / rk[0].size
    res["per_layer_m"] = [float(x) for x in nvme[g].mean(axis=(0, 2))]
    out("\nper-layer m (layers 3..44), N %d, W 0, generated positions:" % GATE_N)
    out("  " + " ".join("%.3f" % x for x in res["per_layer_m"]))

    res["depth"] = []
    out("\ndepth buckets, generated positions, N %d, W %d" % (GATE_N, gate_window))
    for lo, hi in DEPTHS:
        sel = g & (pos >= lo) & (pos < (hi or 10 ** 12))
        s = stat(m_gate[sel])
        res["depth"].append({"from": lo, "to": hi, "m": s})
        out("  %6d-%-6s n %7d  m %s" % (lo, "" if hi is None else hi, s["n"],
                                        "no positions" if not s["n"] else "%.4f [%.4f, %.4f]" % (s["mean"], *s["ci"])))

    own = rank_table(gen_counts([held], shape))
    _, _, nv_own = shares(visit_ranks(held.routes, own), GATE_N)
    res["ceiling_own_cut_m"] = float(nv_own[g].mean())
    out("\nceiling (held-out cut on its own counts; never a verdict): m %.4f at N %d, W 0"
        % (res["ceiling_own_cut_m"], GATE_N))

    res["loo"] = []
    files = [held] + list(cal)
    out("\nleave-one-out (cut on all other files, score the generated positions of the one left out), W 0")
    for f in files:
        rest = [x for x in files if x is not f]
        rkf = visit_ranks(f.routes, cut(rest, f, shape))
        line = "  out %-24s n %7d" % (f.name, f.gen.sum())
        fold = {"left_out": f.name, "generated": int(f.gen.sum())}
        for n in REPORT_N:
            s = stat(shares(rkf, n)[2][f.gen])
            fold[n] = s
            line += "  N %d m %.4f [%.4f, %.4f]" % (n, s["mean"], *s["ci"])
        res["loo"].append(fold)
        out(line)

    gs = stat(m_gate[g])
    p_gate = res["by_n"][GATE_N]["pinned"]["mean"]
    res["pinned_traffic"] = {"p": p_gate, "gbps_at_40": p_gate * TOKEN_GBPS}
    out("\npinned traffic at 40 tok/s, N %d: p %.4f -> %.1f GB/s over PCIe (reported, no threshold)"
        % (GATE_N, p_gate, p_gate * TOKEN_GBPS))
    for name, r in STAGE_CEILINGS:
        floor = p_gate * TOKEN_BYTES / (r * 1e9) + (gs["mean"] * TOKEN_BYTES / (b * 1e9) if b else 0.0)
        res["pinned_traffic"][name] = {"ceiling_gbps": r, "fits": p_gate * TOKEN_GBPS <= r,
                                       "floor_ms": 1e3 * floor, "B_in_floor": b is not None}
        out("  %-28s %.1f GB/s: %s; per-token floor p x 4.756 GB / R%s = %.1f ms (%.1f tok/s)"
            % (name, r, "fits" if p_gate * TOKEN_GBPS <= r else "does NOT fit",
               " + m x 4.756 GB / B" if b else " (B missing: NVMe term left out)", 1e3 * floor,
               1.0 / floor if floor > 0 else math.inf))

    v = g1_verdict(gs, b, b_reason, reasons)
    v["window"] = gate_window
    res["g1"] = v
    out("\nG1 (PREREG, N %d, W %d, generated positions of the held-out):" % (GATE_N, gate_window))
    out("  m %.4f, 95 %% CI upper bound %.4f, n %d" % (v["m"], v["m_ci_hi"], v["generated"]))
    out("  m* = B / %.1f GB/s: %s;  prefill bound 39.9 x B: %s"
        % (TOKEN_GBPS, "%.4f" % v["m_star"] if v["m_star"] is not None else "-",
           "%.1f tok/s" % v["prefill_bound_tok_s"] if v["prefill_bound_tok_s"] is not None else "-"))
    out("  " + v["verdict"])
    return res


# ------------------------------------------------------------------ corpus

def _session_ids():
    spec = importlib.util.spec_from_file_location("session_ids", Path(__file__).with_name("session_ids.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gen_spans(tok, msgs, ids, render):
    """The generated spans of tools/session_ids.py: per assistant message, from the end of its prompt
    (render of the messages before it + generation prompt) to the end of its own render; a span whose
    prompt is not a token prefix of the longer render is skipped and counted."""
    spans, skipped = [], 0
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        prompt = render(tok, msgs[:i], gen=True)
        upto = render(tok, msgs[:i + 1])
        if upto[:len(prompt)] != prompt or ids[:len(upto)] != upto:
            skipped += 1
            continue
        spans.append([len(prompt), len(upto)])
    return spans, skipped


def clip(spans, cap):
    return [[a, min(b, cap)] for a, b in spans if a < cap]


def corpus_cmd(a):
    si = _session_ids()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True)
    if os.path.exists(os.path.join(a.out, "corpus.json")):
        sys.exit("%s/corpus.json exists - refusing to overwrite" % a.out)
    doc = {"tool": "tools/glm_tier_sim.py corpus", "model": a.model, "cap": a.cap, "held": a.held,
           "tokenizer_sha256": sha256_file(os.path.join(a.model, "tokenizer.json")),
           "chat_template_sha256": sha256_file(os.path.join(a.model, "chat_template.jinja")),
           "render": "session_ids.messages + apply_chat_template, thinking on (template default), no tool schemas",
           "files": []}
    for name, task, src in a.file:
        msgs = si.messages(src)
        ids = si.render(tok, msgs)
        spans, skipped = gen_spans(tok, msgs, ids, si.render)
        cap = len(ids) if a.cap <= 0 else min(a.cap, len(ids))
        routed, sp = ids[:cap], clip(spans, cap)
        mask = {"tokens": cap, "spans": sp, "skipped": skipped, "tokens_full": len(ids)}
        jdump(routed, os.path.join(a.out, name + "-ids.json"))
        jdump(mask, os.path.join(a.out, name + "-mask.json"))
        f = {"name": name, "task": task, "role": "held" if name == a.held else "cal", "source": src,
             "source_bytes": os.path.getsize(src), "source_sha256": sha256_file(src),
             "tokens_full": len(ids), "generated_full": sum(b - x for x, b in spans),
             "tokens": cap, "generated": sum(b - x for x, b in sp), "assistant_skipped": skipped,
             "ids_sha256": sha256_ids(routed),
             "mask_sha256": sha256_file(os.path.join(a.out, name + "-mask.json"))}
        doc["files"].append(f)
        print("%-24s %-6s tokens %7d (routed %6d)  generated %6d (routed %6d)  skipped %d  sha256 %s"
              % (name, f["role"], len(ids), cap, f["generated_full"], f["generated"], skipped, f["source_sha256"]))
    check_corpus(doc)
    jdump(doc, os.path.join(a.out, "corpus.json"), indent=1)
    print("wrote %s" % os.path.join(a.out, "corpus.json"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--model", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--cap", type=int, default=32768, help="route the first CAP tokens of each file (0: all)")
    c.add_argument("--held", required=True)
    c.add_argument("--file", nargs=3, action="append", required=True, metavar=("NAME", "TASK", "SESSION_JSON"))
    s = sub.add_parser("sim")
    s.add_argument("--corpus", required=True)
    s.add_argument("--runs", required=True)
    s.add_argument("--step3")
    s.add_argument("--readers", type=int, help="reader count fixed by a PREREG amendment (amendment 5: 1)")
    s.add_argument("--windows", default="0,8,16,32")
    s.add_argument("--gate-window", type=int, default=0)
    s.add_argument("--json")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "corpus":
            return corpus_cmd(a)
        held, cal, reasons = load_corpus(a.corpus, a.runs)
        b, why = b_from_step3(a.step3, a.readers)
        res = simulate(held, cal, b, why, reasons, tuple(int(x) for x in a.windows.split(",")), a.gate_window)
        if a.json:
            jdump(res, a.json, indent=1, default=float)
    except SimError as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
