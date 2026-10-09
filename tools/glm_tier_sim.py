#!/usr/bin/env python3
"""#147 (GLM-5.3-Flash step 8, gate G1): offline tier simulation over the layerwise runner's routing.

  # 1. corpus: Crow sessions through GLM's own chat template and tokenizer (oracle venv; reads
  #    downloaded files, so -I). Writes <name>-ids.json, <name>-mask.json and corpus.json.
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py corpus \
      --model models/GLM-5.3-Flash-original --out <dir> --cap 32768 --held <name> \
      --file <name> <task> <session.json> [--file ...]

  # 2. routing (step 7 runner, one out dir per corpus name; G1 ran on the 4.5-bit container, #147; G1d runs on
  #    the FP8 originals, #179, PREREG-dyn amendment 1):
  #    .venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py --fp8 models/GLM-5.3-Flash-original [--dry-run]
  #    PREREG-dyn amendment 2: the calibration files' routing from the conversion's capture (dyn --capture), so only
  #    the held-out needs a pass: ... glm_route_passes.py --fp8 models/GLM-5.3-Flash-original --files todo-1006

  # 3. simulation and the G1 verdict fields
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py sim --corpus <dir>/corpus.json --runs <runs> \
      [--step3 runs/glm53-flash/step03/<run>.json] [--windows 0,8,16,32] [--json <out.json>]

  # 4. dynamic expert-cache policies (#178, plan step 2; docs/glm-tier-simulation.md section 7)
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py dyn --corpus <dir>/corpus.json --runs <runs> \
      [--capture decode_out/glm-step8/capture-ids] (--slots V:P[,...] | --vram 25.6GB --pinned 46GiB) [--bpw 4.5,3.05,3.5] [--arena layer|global] \
      [--policies lru,clock,lfu] [--admit-max 64] [--prefetch none,oracle,0.5,0.7,0.9] [--depths 1,2,3] \
      [--pf-budget N] [--step3 <run>.json --readers 1] [--rates <rates.json>] [--json <out.json>]

  # 5. the judged G1d policy and verdict (PREREG-dyn, #178; docs/glm-tier-simulation.md section 8)
  .venv-oracle/Scripts/python.exe -I tools/glm_tier_sim.py g1d --corpus <dir>/corpus.json --runs <runs> \
      [--capture decode_out/glm-step8/capture-ids] --step3 <run>.json --readers 1 [--cells 3.05:46:V25|all] \
      [--jobs N] [--json <out.json>]

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
import random
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


# PREREG-dyn amendment 1 (2026-10-09, #179): the G1d routing comes from the FP8 originals of zai-org/GLM-5.3-Flash at
# this revision; tools/glm_route_passes.py --fp8 writes their identity to weights.json in every pass dir and to each
# passes.jsonl row. The G1 record (`sim`, source_reasons) is not changed by it.
FP8_REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
FP8_SHARDS = 62


def identity_sha256(ident):
    """sha256 of an identity's canonical JSON without its own identity_sha256, as tools/glm_route_passes.py
    identity_sha256 computes it (that module imports this one, so the two lines are repeated, not imported)."""
    body = {k: v for k, v in ident.items() if k != "identity_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def g1d_source_reasons(man, out_dir):
    """PREREG-dyn amendment 1, check 2, for one pass dir -> (reasons it is not the G1d source, identity sha256).
    It replaces source_reasons' "not CNQ" reason for G1d only: FP8 weights, not partial, complete over every layer,
    --state-dtype bf16 --prompt-chunk 512, and a weights.json of kind fp8-originals at FP8_REVISION with 62 shards,
    its identity sha256 matching its content and its index sha256 the runner manifest's."""
    out = []
    w = man.get("weights") or {}
    if str(w.get("weights", "")).split(" ")[0] != "fp8":
        out.append("routing not from the FP8 originals (G1d source, PREREG-dyn amendment 1)")
    if w.get("partial"):
        out.append("partial weights (%s)" % (w["partial"].get("filter") or "partial"))
    if not man.get("complete") or list(man.get("layers", [])) != [0, man.get("num_hidden_layers", -1)]:
        out.append("runner pass not complete over layers 0..%s" % man.get("num_hidden_layers"))
    if man.get("state_dtype") != "bf16" or man.get("prompt_chunk") != 512:
        out.append("runner pass not --state-dtype bf16 --prompt-chunk 512 (%s, %s)"
                   % (man.get("state_dtype"), man.get("prompt_chunk")))
    wp = os.path.join(out_dir, "weights.json")
    if not os.path.exists(wp):
        return out + ["no weights.json (the FP8 identity tools/glm_route_passes.py --fp8 writes)"], None
    ident = jload(wp)
    if ident.get("kind") != "fp8-originals" or ident.get("revision") != FP8_REVISION:
        out.append("weights.json is %s at %s, not fp8-originals at %s"
                   % (ident.get("kind"), ident.get("revision"), FP8_REVISION[:8]))
    if len(ident.get("shards") or {}) != FP8_SHARDS:
        out.append("weights.json names %d shards, not %d" % (len(ident.get("shards") or {}), FP8_SHARDS))
    if ident.get("identity_sha256") != identity_sha256(ident):
        out.append("weights.json identity_sha256 does not match its content")
    if ident.get("index_json_sha256") != w.get("index_json_sha256"):
        out.append("runner manifest index sha256 %s is not weights.json's %s"
                   % (w.get("index_json_sha256"), ident.get("index_json_sha256")))
    return out, ident.get("identity_sha256")


def g1d_book_reasons(runs_dir, names, identity):
    """Every passes.jsonl row of a corpus file carries the identity, and each file has an ok row."""
    book = os.path.join(runs_dir, "passes.jsonl")
    if not os.path.exists(book):
        return ["no passes.jsonl in %s (the pass rows of tools/glm_route_passes.py)" % runs_dir]
    out, ok = [], set()
    with open(book, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("name") not in names:
                continue
            if row.get("weights_identity_sha256") != identity:
                out.append("passes.jsonl row of %s carries identity %s, not %s"
                           % (row["name"], row.get("weights_identity_sha256"), identity))
            if row.get("ok"):
                ok.add(row["name"])
    out += ["passes.jsonl has no ok row for %s" % n for n in names if n not in ok]
    return out


# PREREG-dyn amendment 2 (2026-10-09, #179 / #178): the calibration files' G1d routing may come from the conversion's
# capture (#182, tools/glm_mul1_quantize.py capture: the same runner code, run_layer, on the same FP8 originals), saved
# per MoE layer as <capture>/L<ll>/ids.i32 (int32 [rows][8], the four files' rows one after another) + capture.json.
CAPTURE_FIRST_LAYER = 3     # MoE layers 3..44 -> L03..L44 (SHAPE[0] of them)


def load_capture(cap_dir, files, identity, shape=SHAPE, first=CAPTURE_FIRST_LAYER):
    """The conversion capture's routing of the calibration files -> ({name: uint16 [n][L][K]}, {"Lxx": ids sha256}).
    files: [(name, rows)] in the corpus' calibration order; identity: the FP8 identity sha256 of the held-out pass dir.
    Refused by name (SimError): a missing layer; per layer a capture.json of another layer, another identity, other
    files or another order, other rows or top-k, an ids.i32 whose sha256 is not capture.json's or whose size is not
    rows x K; the routing self-test of load_runner (ids 0..E-1, K distinct ascending per row)."""
    L, E, K = shape
    layers = list(range(first, first + L))
    missing = [l for l in layers if not all(os.path.isfile(os.path.join(cap_dir, "L%02d" % l, f))
                                            for f in ("capture.json", "ids.i32"))]
    if missing:
        raise SimError("capture %s: layer%s %s missing (ids.i32 + capture.json per MoE layer %d..%d)" % (
            cap_dir, "s" if len(missing) > 1 else "", ", ".join("L%02d" % l for l in missing), layers[0], layers[-1]))
    names, n_rows = [f for f, _ in files], [n for _, n in files]
    total = sum(n_rows)
    out = {f: np.empty((n, L, K), np.uint16) for f, n in files}
    shas = {}
    for j, l in enumerate(layers):
        d = os.path.join(cap_dir, "L%02d" % l)
        tag = "capture %s L%02d" % (cap_dir, l)
        cj = jload(os.path.join(d, "capture.json"))
        if cj.get("layer") != l:
            raise SimError("%s: capture.json is of layer %s" % (tag, cj.get("layer")))
        if cj.get("identity") != identity:
            raise SimError("%s: identity %s is not the FP8 identity %s of the held-out pass dir (weights.json)"
                           % (tag, cj.get("identity"), identity))
        got = [f.get("name") for f in cj.get("files") or []]
        if got != names:
            raise SimError("%s: files %s, the corpus' calibration files in order are %s" % (tag, got, names))
        rws = [f.get("rows") for f in cj["files"]]
        if rws != n_rows or cj.get("rows") != total or cj.get("top_k") != K:
            raise SimError("%s: rows %s (total %s, top-k %s), the corpus files have %s (total %d, top-k %d)"
                           % (tag, rws, cj.get("rows"), cj.get("top_k"), n_rows, total, K))
        p = os.path.join(d, "ids.i32")
        sha = sha256_file(p)
        if sha != cj.get("ids_sha256"):
            raise SimError("%s: ids.i32 sha256 %s is not capture.json's %s" % (tag, sha, cj.get("ids_sha256")))
        a = np.fromfile(p, "<i4")
        if a.size != total * K:
            raise SimError("%s: ids.i32 holds %d ids, expected [%d][%d]" % (tag, a.size, total, K))
        a = a.reshape(total, K)
        if a.min() < 0 or a.max() >= E:
            raise SimError("self-test: %s has an id outside 0..%d" % (tag, E - 1))
        if K > 1 and not (np.diff(a, axis=1) > 0).all():
            raise SimError("self-test: %s has a row that is not %d distinct ascending ids" % (tag, K))
        r0 = 0
        for f, n in files:
            out[f][:, j, :] = a[r0:r0 + n]
            r0 += n
        shas["L%02d" % l] = sha
    return out, shas


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
    def __init__(self, name, task, routes, gen, man=None, identity=None):
        self.name, self.task, self.routes, self.gen, self.man = name, task, routes, gen, man
        self.identity = identity


def load_corpus(corpus_path, runs_dir, shape=SHAPE, source="g1", capture=None):
    """corpus.json + one runner out dir per name -> (held Run, [cal Run], [reasons the source is not of record]).
    source "g1": the G1 record's rule (source_reasons, CNQ container); "g1d": PREREG-dyn amendment 1 (FP8 originals,
    g1d_source_reasons, one identity in every dir and in passes.jsonl), the identity on every Run.
    capture (PREREG-dyn amendment 2): the calibration files' routing from the conversion's capture dir (load_capture,
    checked against the held-out pass dir's weights.json identity); only the held-out needs a pass dir (and, for
    "g1d", a passes.jsonl row). Each such Run carries the per-layer ids sha256 as .capture. Under "g1" (`sim`, the
    descriptive check 4.2 of amendment 1) it adds a "plausibility only" reason, so G1 stays "not answered"."""
    c = jload(corpus_path)
    base = os.path.dirname(os.path.abspath(corpus_path))
    check_corpus(c)
    from_cap = [f for f in c["files"] if f["role"] == "cal"] if capture is not None else []
    runs, reasons, idents = {}, [], {}
    for f in c["files"]:
        if f in from_cap:
            continue
        man, routes = load_runner(os.path.join(runs_dir, f["name"]), shape)
        ids = jload(os.path.join(base, f["name"] + "-ids.json"))
        if man["ids"] != ids:
            raise SimError("%s: the runner's ids are not %s-ids.json of the corpus" % (f["name"], f["name"]))
        if sha256_ids(ids) != f["ids_sha256"]:
            raise SimError("%s-ids.json does not match corpus.json (sha256)" % f["name"])
        gen = gen_mask(os.path.join(base, f["name"] + "-mask.json"), len(routes))
        if source == "g1d":
            why, idents[f["name"]] = g1d_source_reasons(man, os.path.join(runs_dir, f["name"]))
        else:
            why = source_reasons(man)
        reasons += ["%s: %s" % (f["name"], r) for r in why]
        runs[f["name"]] = Run(f["name"], f["task"], routes, gen, man, idents.get(f["name"]))
        print("self-test %-24s positions %7d generated %6d  routing files == manifest sha256: True"
              % (f["name"], len(routes), gen.sum()))
    if source == "g1d":
        found = sorted(set(x for x in idents.values() if x))
        if len(found) > 1:
            reasons.append("weights.json differs between the pass dirs (%d identities)" % len(found))
        elif found:
            reasons += g1d_book_reasons(runs_dir, list(idents), found[0])
    if from_cap:
        ident = idents.get(c["held"])
        wp = os.path.join(runs_dir, c["held"], "weights.json")
        if source != "g1d" and os.path.exists(wp):
            ident = jload(wp).get("identity_sha256")
        if source != "g1d":   # `sim`, check 4.2 of PREREG-dyn amendment 1: descriptive, never G1's record
            reasons.append("calibration files: routing from the conversion's capture (FP8 originals): plausibility only")
        if not ident:
            raise SimError("capture %s: its identity cannot be checked, the held-out pass dir %s has no weights.json "
                           "identity" % (capture, os.path.join(runs_dir, c["held"])))
        ids = {}
        for f in from_cap:
            ids[f["name"]] = jload(os.path.join(base, f["name"] + "-ids.json"))
            if sha256_ids(ids[f["name"]]) != f["ids_sha256"]:
                raise SimError("%s-ids.json does not match corpus.json (sha256)" % f["name"])
        routes, shas = load_capture(capture, [(f["name"], len(ids[f["name"]])) for f in from_cap], ident, shape)
        for f in from_cap:
            r = routes[f["name"]]
            gen = gen_mask(os.path.join(base, f["name"] + "-mask.json"), len(r))
            runs[f["name"]] = Run(f["name"], f["task"], r, gen, None, ident)
            runs[f["name"]].capture = shas
            print("self-test %-24s positions %7d generated %6d  capture ids.i32 == capture.json sha256: True"
                  % (f["name"], len(r), gen.sum()))
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
        miss = _min_miss(routes[pos, l, k].tolist(), w)
        np.add.at(reads, pos[miss], 1)
    return reads


def _min_miss(ids, w):
    """Belady's MIN with bypass over one access sequence and w slots -> miss flag per access."""
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
        if w <= 0:
            continue
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
    return miss


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


# ------------------------------------------------------------------ dynamic expert cache (#178, plan step 2)

# Expert record bytes per bpw. 4.5: the NVFP4 record of the CNQ container (EXPERT_BYTES). 3.05: the EXL3 record of
# sybil-solutions/glm53-flash-offload (plan record, external). 3.5: the 3.05 record scaled by 3.5 / 3.05 (derived,
# not measured; same codec family).
BPW_BYTES = {"4.5": EXPERT_BYTES, "3.05": 9_474_048, "3.5": round(9_474_048 * 3.5 / 3.05)}
POLICIES = ("lru", "clock", "lfu")
DYN_COUNTERS = ("vram_hits", "pinned_hits", "zero_copy", "admissions", "write_backs", "nvme_demand", "nvme_prefetch")
_UNITS = {"": 1, "B": 1, "KB": 10 ** 3, "MB": 10 ** 6, "GB": 10 ** 9, "KIB": 2 ** 10, "MIB": 2 ** 20, "GIB": 2 ** 30}


def parse_bytes(s):
    """'46GiB', '25.6GB', '1000' -> bytes (int)."""
    t = str(s).strip()
    i = len(t)
    while i and t[i - 1].isalpha():
        i -= 1
    unit = t[i:].upper()
    if unit not in _UNITS or not t[:i]:
        raise SimError("cannot read %r as bytes (units B, KB, MB, GB, KiB, MiB, GiB)" % s)
    try:
        v = float(t[:i]) * _UNITS[unit]
    except ValueError:
        raise SimError("cannot read %r as bytes" % s) from None
    if v < 0:
        raise SimError("negative byte budget %r" % s)
    return int(v)


def capacity(vram_bytes, pinned_bytes, expert_bytes, arena, layers=SHAPE[0]):
    """Slots (VRAM, pinned) from byte budgets: per layer floor(budget / expert / layers), global
    floor(budget / expert)."""
    div = expert_bytes * (layers if arena == "layer" else 1)
    return int(vram_bytes // div), int(pinned_bytes // div)


class LruTier:
    def __init__(self, cap):
        self.cap, self.d = cap, OrderedDict()

    def __contains__(self, k):
        return k in self.d

    def touch(self, k):
        self.d.move_to_end(k)

    def remove(self, k):
        del self.d[k]

    def insert(self, k, protect=()):
        """-> (stored, victim). LRU ignores `protect`, as lru_reads does (equal to it at every capacity)."""
        if self.cap <= 0:
            return False, None
        v = self.d.popitem(last=False)[0] if len(self.d) >= self.cap else None
        self.d[k] = None
        return True, v


class ClockTier:
    """CLOCK as sybil-solutions/glm53-flash-offload glm53/expert_cache.py ec_step_k (df0b439): a hit sets the ref
    bit; an insert sweeps from the hand, skips slots that hold one of the current step's picks, clears set ref bits,
    takes the first slot with a clear bit, sets the new entry's bit and leaves the hand behind it; no victim within
    2 x cap steps -> not stored. An empty slot (never filled, or freed by a move to the other tier, which sybil's
    VRAM-only cache has not) is taken first, in slot order, without a sweep."""

    def __init__(self, cap):
        self.cap, self.owner, self.ref, self.slot, self.hand = cap, [None] * cap, [0] * cap, {}, 0
        self.free = list(range(cap - 1, -1, -1))

    def __contains__(self, k):
        return k in self.slot

    def touch(self, k):
        self.ref[self.slot[k]] = 1

    def remove(self, k):
        s = self.slot.pop(k)
        self.owner[s], self.ref[s] = None, 0
        self.free.append(s)

    def insert(self, k, protect=()):
        cap, owner, ref, h = self.cap, self.owner, self.ref, self.hand
        if cap <= 0:
            return False, None
        if self.free:
            s = self.free.pop()
            owner[s], ref[s], self.slot[k] = k, 1, s
            return True, None
        found = -1
        for _ in range(2 * cap):
            s = h
            h = h + 1 if h + 1 < cap else 0
            o = owner[s]
            if o is not None and o in protect:
                continue
            if ref[s]:
                ref[s] = 0
                continue
            found = s
            break
        self.hand = h
        if found < 0:
            return False, None
        o = owner[found]
        if o is not None:
            del self.slot[o]
        owner[found], ref[found], self.slot[k] = k, 1, found
        return True, o


class LfuTier:
    """LFU with exponential decay: the score of an expert is the sum over its accesses of 2^-(age in tokens /
    half-life), kept for every expert (also after eviction); the victim is the resident with the lowest score (ties:
    lower key), never one of the current step's picks. Scores live in a dict shared by both tiers."""

    def __init__(self, cap, score):
        self.cap, self.score, self.res, self.heap = cap, score, set(), []

    def __contains__(self, k):
        return k in self.res

    def _push(self, k):
        heapq.heappush(self.heap, (self.score.get(k, 0.0), k))
        if len(self.heap) > 4 * self.cap + 64:
            self.rebuild()

    def touch(self, k):
        self._push(k)

    def remove(self, k):
        self.res.discard(k)

    def rebuild(self):
        self.heap = [(self.score.get(k, 0.0), k) for k in self.res]
        heapq.heapify(self.heap)

    def insert(self, k, protect=()):
        if self.cap <= 0:
            return False, None
        v = None
        if len(self.res) >= self.cap:
            held = []
            while self.heap:
                s, x = heapq.heappop(self.heap)
                if x not in self.res or self.score.get(x, 0.0) != s:
                    continue                          # stale entry
                if x in protect:
                    held.append((s, x))
                    continue
                v = x
                break
            for e in held:
                heapq.heappush(self.heap, e)
            if v is None:
                return False, None
            self.res.discard(v)
        self.res.add(k)
        self._push(k)
        return True, v


def _put(tier, k, protect, pending):
    """Insert k into a tier -> (stored, 1 if the victim was a prefetched expert never used)."""
    ok, v = tier.insert(k, protect)
    if v is not None and v in pending:
        pending.discard(v)
        return ok, 1
    return ok, 0


def _predict(true, p, n_experts, rnd):
    """A router prediction of precision p: each true expert is kept with probability p, else replaced by a wrong
    expert of the same layer (uniform over the experts not routed and not yet predicted)."""
    out, taken = [], set(true)
    for e in true:
        if rnd.random() < p:
            out.append(e)
            continue
        while True:
            w = rnd.randrange(n_experts)
            if w not in taken:
                break
        taken.add(w)
        out.append(w)
    return out


def dyn_run(routes, cv, cp, policy="lru", arena="layer", admit_max=None, prefetch=None, depth=1, pf_budget=None,
            halflife=64.0, n_experts=SHAPE[1], seed=SEED):
    """One dynamic-cache run over a file's routing, token order over every position.

    Two exclusive tiers per arena (one per layer, or one for all layers): VRAM with cv slots and pinned RAM with cp
    slots, the rest on the NVMe. A visit is a VRAM hit; or a pinned hit, admitted to VRAM (one PCIe copy) when
    admission is on, else read zero-copy (one PCIe read); or an NVMe read, admitted to VRAM or landed in pinned and
    read zero-copy. A VRAM victim moves to pinned (one PCIe write-back), a pinned victim is dropped (its record stays
    on the NVMe). Admission is on when the step's picks per layer (K for one token) are <= admit_max, the gate of
    sybil's GLM53_EC_ADMIT_MAX (None: always). Prefetch (None, "oracle" or a precision p): after layer j, the
    experts predicted for layer j + depth (into the next token past the last layer) are read from the NVMe into
    pinned, at most pf_budget reads per token. Returns per-position counters (DYN_COUNTERS) and the prefetch totals.
    """
    n, L, K = routes.shape
    E = n_experts
    if policy not in POLICIES:
        raise SimError("policy %r: one of %s" % (policy, ", ".join(POLICIES)))
    if arena not in ("layer", "global"):
        raise SimError("arena %r: layer or global" % arena)
    if prefetch is not None and prefetch != "oracle" and not 0.0 < float(prefetch) <= 1.0:
        raise SimError("prefetch precision %r outside (0, 1]" % prefetch)
    if not 1 <= depth <= L:
        raise SimError("look-ahead %d outside 1..%d layers" % (depth, L))
    score = {}

    def mk(c):
        return LruTier(c) if policy == "lru" else ClockTier(c) if policy == "clock" else LfuTier(c, score)

    if arena == "layer":
        V, P = [mk(cv) for _ in range(L)], [mk(cp) for _ in range(L)]
    else:
        v1, p1 = mk(cv), mk(cp)
        V, P = [v1] * L, [p1] * L
    admit = cv > 0 and (admit_max is None or K <= admit_max)
    do_pf = prefetch is not None and cp > 0
    rnd = random.Random(seed)
    lfu = policy == "lfu"
    grow, inc = 2.0 ** (1.0 / halflife), 1.0
    pending, useful, wasted = set(), 0, 0
    out = {c: [0] * n for c in DYN_COUNTERS}
    R = routes.tolist()
    for t in range(n):
        vh = ph = zc = adm = wb = nd = npf = 0
        row = R[t]
        for j in range(L):
            Vj, Pj, base = V[j], P[j], j * E
            keys = [base + e for e in row[j]]
            protect = set(keys)
            for k in keys:
                if lfu:
                    score[k] = score.get(k, 0.0) + inc
                if k in Vj:
                    Vj.touch(k)
                    vh += 1
                    continue
                if k in Pj:
                    ph += 1
                    if k in pending:
                        pending.discard(k)
                        useful += 1
                    ok, v = Vj.insert(k, protect) if admit else (False, None)
                    if not ok:
                        Pj.touch(k)
                        zc += 1
                        continue
                    Pj.remove(k)
                else:
                    nd += 1
                    ok, v = Vj.insert(k, protect) if admit else (False, None)
                    if not ok:
                        zc += 1
                        if cp:
                            wasted += _put(Pj, k, protect, pending)[1]
                        continue
                adm += 1
                if v is not None and cp:
                    ok2, w = _put(Pj, v, protect, pending)
                    wasted += w
                    wb += ok2
            if do_pf:
                jj, tt = j + depth, t
                if jj >= L:
                    jj, tt = jj - L, t + 1
                if tt < n and (pf_budget is None or npf < pf_budget):
                    true = R[tt][jj]
                    preds = true if prefetch == "oracle" else _predict(true, float(prefetch), E, rnd)
                    Vt, Pt, b2 = V[jj], P[jj], jj * E
                    pkeys = [b2 + e for e in preds]
                    prot = set(pkeys)
                    for k in pkeys:
                        if k in Vt or k in Pt:
                            continue
                        if pf_budget is not None and npf >= pf_budget:
                            break
                        ok, w = _put(Pt, k, prot, pending)
                        wasted += w
                        if ok:
                            npf += 1
                            pending.add(k)
        for c, x in zip(DYN_COUNTERS, (vh, ph, zc, adm, wb, nd, npf)):
            out[c][t] = x
        if lfu:
            inc *= grow
            if inc > 1e100:                           # renormalise: same order, no overflow
                for k in score:
                    score[k] /= inc
                inc = 1.0
                for tier in set(V) | set(P):
                    tier.rebuild()
    res = {c: np.asarray(x, np.int64) for c, x in out.items()}
    res["prefetch_useful"], res["prefetch_wasted"] = useful, wasted + len(pending)
    return res


def dyn_min(routes, cap, arena="layer", n_experts=SHAPE[1]):
    """Belady's MIN with bypass at the total capacity (VRAM + pinned slots) of the arena -> NVMe reads per position:
    the fewest reads any policy, prefetching ones included, can reach with that many slots."""
    n, L, K = routes.shape
    if arena == "layer":
        return min_reads(routes, np.ones(routes.shape, bool), cap)
    keys = (routes.astype(np.int64) + (np.arange(L, dtype=np.int64) * n_experts)[None, :, None]).reshape(-1)
    return _min_miss(keys.tolist(), cap).reshape(n, L * K).sum(1).astype(np.int64)


def rates_from_file(path):
    """R_PCIe and R_DRAM (GB/s) from a JSON file {"R_pcie_gbps": x, "R_dram_gbps": y, "source": "..."}; either may be
    missing (that stage is then not given), a present value must be a positive number."""
    if not path:
        return {}, "no rates file given"
    d = jload(path)
    out = {}
    for key in ("R_pcie_gbps", "R_dram_gbps"):
        if key in d:
            x = d[key]
            if isinstance(x, bool) or not isinstance(x, (int, float)) or not x > 0:
                raise SimError("%s: %s = %r is not a positive number" % (path, key, x))
            out[key] = float(x)
    if not out:
        raise SimError("%s: neither R_pcie_gbps nor R_dram_gbps" % path)
    return out, "%s (%s)" % (os.path.basename(path), d.get("source", "no source named"))


def dyn_cost(row, expert_bytes, b, rates, visits=SHAPE[0] * SHAPE[2]):
    """Per-stage bytes per token and tok/s ceilings: NVMe = m (prefetch reads included) at B, PCIe = zero-copy +
    admissions + write-backs at R_PCIe, DRAM = NVMe data landed + every PCIe transfer at R_DRAM. The binding ceiling
    is the lowest given one (perfect overlap); the serial bound adds the stage times (no overlap), only when all
    three rates are given."""
    per = visits * expert_bytes
    out, ceil, serial = {}, [], 0.0
    for name, share, rate in (("nvme", row["m"]["mean"], b), ("pcie", row["pcie"], rates.get("R_pcie_gbps")),
                              ("dram", row["dram"], rates.get("R_dram_gbps"))):
        by = share * per
        c = None if rate is None else (math.inf if by == 0 else rate * 1e9 / by)
        out[name] = {"bytes_per_token": by, "rate_gbps": rate, "ceiling_tok_s": c}
        if c is not None:
            ceil.append((c, name))
        if serial is not None:
            serial = None if rate is None else serial + by / (rate * 1e9)
    out["binding"] = min(ceil)[1] if ceil else None
    out["ceiling_tok_s"] = min(ceil)[0] if ceil else None
    out["serial_tok_s"] = (math.inf if serial == 0 else 1.0 / serial) if serial is not None else None
    return out


def dyn_summary(res, gen, visits=SHAPE[0] * SHAPE[2]):
    """Shares of the visits per token on the generated positions; m (all NVMe reads) with the block-bootstrap CI."""
    sh = {c: res[c][gen] / visits for c in DYN_COUNTERS}
    pcie = sh["zero_copy"] + sh["admissions"] + sh["write_backs"]
    nv = sh["nvme_demand"] + sh["nvme_prefetch"]
    row = {"m": stat(nv), "m_demand": float(sh["nvme_demand"].mean()), "pcie": float(pcie.mean()),
           "dram": float((nv + pcie).mean())}
    row.update({c: float(sh[c].mean()) for c in DYN_COUNTERS})
    row["prefetch_useful"], row["prefetch_wasted"] = res["prefetch_useful"], res["prefetch_wasted"]
    return row


def dyn_check_min(name, res, mn):
    """A policy below MIN over the whole file is a simulator defect (MIN bounds total reads, not a subset's)."""
    got = int(res["nvme_demand"].sum() + res["nvme_prefetch"].sum())
    if got < int(mn.sum()):
        raise SimError("simulator defect: %s reads %d < Belady MIN %d at the same capacity"
                       % (name, got, int(mn.sum())))


def dyn_simulate(held, configs, policies=POLICIES, arena="layer", admit_max=None, prefetches=(None,), depths=(1,),
                 pf_budget=None, halflife=64.0, b=None, b_reason="no step-3 run given", rates=None,
                 rates_reason="no rates file given", reasons=(), out=print):
    """Every `dyn` row: configs = [(bpw, expert_bytes, cv, cp)]; returns the --json document."""
    rates = rates or {}
    routes, g = held.routes, held.gen
    L, K = routes.shape[1], routes.shape[2]
    res = {"held": held.name, "positions": int(len(g)), "generated": int(g.sum()), "arena": arena,
           "admit_max": admit_max, "pf_budget": pf_budget, "lfu_halflife": halflife, "B": b, "B_source": b_reason,
           "rates": rates, "rates_source": rates_reason, "source": "FP8 originals, PREREG-dyn amendment 1",
           "weights_identity_sha256": getattr(held, "identity", None), "source_reasons": list(reasons), "rows": []}
    out("\nheld-out %s (%s): %d positions, %d generated; arena %s, admission %s, prefetch budget %s per token"
        % (held.name, held.task, len(g), g.sum(), arena,
           "always" if admit_max is None else "at <= %d picks per step" % admit_max,
           "none" if pf_budget is None else pf_budget))
    out("B: %s;  rates: %s" % (b_reason, rates_reason))
    out("routing source (G1d, PREREG-dyn amendment 1: FP8 originals): %s, weights identity %s"
        % ("ok" if not reasons else "NOT the G1d source", getattr(held, "identity", None) or "-"))
    for r in reasons:
        out("  source: %s" % r)
    for bpw, eb, cv, cp in configs:
        mn = dyn_min(routes, cv + cp, arena)
        mrow = {"bpw": bpw, "expert_bytes": eb, "vram_slots": cv, "pinned_slots": cp, "policy": "min",
                "m": stat(mn[g] / (L * K))}
        res["rows"].append(mrow)
        out("\nbpw %s (%d B per expert), slots VRAM %d + pinned %d %s; MIN ceiling m %.4f [%.4f, %.4f]"
            % (bpw, eb, cv, cp, "per layer" if arena == "layer" else "in all", mrow["m"]["mean"], *mrow["m"]["ci"]))
        pfs = prefetches if cp > 0 else tuple(x for x in prefetches if x is None)
        if len(pfs) < len(prefetches):
            out("  prefetch rows skipped: prefetch lands in pinned and this config has no pinned slots")
        for pol in policies:
            for pf in pfs:
                for d in (depths if pf is not None else (None,)):
                    r = dyn_run(routes, cv, cp, pol, arena, admit_max, pf, d or 1, pf_budget, halflife)
                    dyn_check_min("%s/%s/d%s" % (pol, pf, d), r, mn)
                    row = {"bpw": bpw, "expert_bytes": eb, "vram_slots": cv, "pinned_slots": cp, "policy": pol,
                           "prefetch": pf, "depth": d}
                    row.update(dyn_summary(r, g, L * K))
                    row["cost"] = c = dyn_cost(row, eb, b, rates, L * K)
                    res["rows"].append(row)
                    out("  %-5s prefetch %-6s d %-2s m %.4f [%.4f, %.4f] (demand %.4f)  VRAM %.4f pinned %.4f  "
                        "PCIe %.4f (zero-copy %.4f adm %.4f wb %.4f)  DRAM %.4f  prefetch useful %d wasted %d"
                        % (pol, "none" if pf is None else pf, "-" if d is None else d, row["m"]["mean"],
                           *row["m"]["ci"], row["m_demand"], row["vram_hits"], row["pinned_hits"], row["pcie"],
                           row["zero_copy"], row["admissions"], row["write_backs"], row["dram"],
                           row["prefetch_useful"], row["prefetch_wasted"]))
                    out("        ceilings tok/s: %s;  binding %s;  serial %s"
                        % (", ".join("%s %s" % (s, "-" if c[s]["ceiling_tok_s"] is None
                                                else "%.1f" % c[s]["ceiling_tok_s"]) for s in ("nvme", "pcie", "dram")),
                           c["binding"] or "-", "-" if c["serial_tok_s"] is None else "%.1f" % c["serial_tok_s"]))
    return res


def dyn_configs(a, layers=SHAPE[0]):
    """[(bpw, expert_bytes, vram slots, pinned slots)] from --bpw with --slots or --vram/--pinned."""
    out = []
    for bpw in a.bpw.split(","):
        if bpw not in BPW_BYTES:
            raise SimError("bpw %s: one of %s" % (bpw, ", ".join(BPW_BYTES)))
        eb = BPW_BYTES[bpw]
        if a.slots:
            for s in a.slots.split(","):
                v, _, p = s.partition(":")
                out.append((bpw, eb, int(v), int(p or 0)))
        elif a.vram is not None and a.pinned is not None:
            out.append((bpw, eb) + capacity(parse_bytes(a.vram), parse_bytes(a.pinned), eb, a.arena, layers))
        else:
            raise SimError("dyn needs --slots V:P[,...] or both --vram and --pinned")
    return out


def calibration_source(a, held, cal):
    """Where the calibration routing came from (the JSON's calibration_routing), printed for a capture."""
    if not a.capture:
        return {"from": "pass dirs", "dir": a.runs, "files": [r.name for r in cal]}
    print("calibration routing: the conversion's capture %s (%d layers, identity %s), held-out from %s"
          % (a.capture, len(cal[0].capture), held.identity, os.path.join(a.runs, held.name)))
    return {"from": "capture (PREREG-dyn amendment 2)", "dir": a.capture, "files": [r.name for r in cal],
            "ids_sha256": cal[0].capture}


def dyn_cmd(a):
    configs = dyn_configs(a)
    held, cal, reasons = load_corpus(a.corpus, a.runs, source="g1d", capture=a.capture)
    cal_src = calibration_source(a, held, cal)
    b, why = b_from_step3(a.step3, a.readers)
    rates, rwhy = rates_from_file(a.rates)
    pfs = tuple(None if x == "none" else x if x == "oracle" else float(x) for x in a.prefetch.split(","))
    res = dyn_simulate(held, configs, tuple(a.policies.split(",")), a.arena, a.admit_max, pfs,
                       tuple(int(x) for x in a.depths.split(",")), a.pf_budget, a.lfu_halflife, b, why, rates, rwhy,
                       reasons)
    res["calibration_routing"] = cal_src
    if a.json:
        jdump(res, a.json, indent=1, default=float)
    return 0


# ------------------------------------------------------------------ G1d: judged policy and verdict (PREREG-dyn, #178)

# runs/glm53-flash/PREREG-dyn.md: "Cache model", "Parameter selection", "Judged metric" and amendments 1-3. S per bpw is
# PREREG-dyn's (3.5 nominal 11,010,048; 3.05 the plan figure of amendment 1), not dyn's BPW_BYTES.
G1D_S = {"4.5": 14_155_776, "3.5": 11_010_048, "3.05": 9_474_048}
G1D_V = {"V25": 14_863_564_800, "V37": 21_998_075_904}
G1D_R_GIB = (46, 50, 54)
G1D_PRIMARY = ("3.05", 46, "V25")           # amendment 1: the deciding cell (per-layer arena)
G1D_GRID_S, G1D_GRID_P, G1D_GRID_D = (0.0, 0.25, 0.5), (0, 4, 8, 16), (1, 2, 4)
G1D_TOK_S = 40.0
G1D_CLASSES = ("LRU", "SEED+LRU", "SEED+LRU+IDPF")
G1D_STATUS = ("bar, primary cell, bpw scenarios, grid and tie order are 'proposed, awaiting robin's confirmation' "
              "(PREREG-dyn, Awaiting confirmation 1-4); no G1d row may be taken before he confirms them")


def g1d_capacity(bpw, r_gib, v, layers=SHAPE[0]):
    """C slots per layer: floor((V + R) / S) over all layers, divided by 42 (PREREG-dyn "Capacity from bytes")."""
    return int((G1D_V[v] + r_gib * 2 ** 30) // G1D_S[bpw]) // layers


def g1d_bar(b, s_bytes, tok_s=G1D_TOK_S):
    """Reads per token at the bar r_hi x S <= B / 40 tok/s."""
    return b * 1e9 / tok_s / s_bytes


def g1d_grid():
    """Every grid point (class, s, P, d) in tie order: the simpler class (LRU < SEED+LRU < SEED+LRU+IDPF), then the
    smaller s, P, d. P = 0 has no d; s = 0 with P = 0 is LRU."""
    pts = [(2 if p else (1 if s else 0), s, p, d) for s in G1D_GRID_S for p in G1D_GRID_P
           for d in (G1D_GRID_D if p else (None,))]
    return sorted(pts, key=lambda x: (x[0], x[1], x[2], x[3] or 0))


def g1d_point_name(pt):
    return "%s s %.2f P %d d %s" % (G1D_CLASSES[pt[0]], pt[1], pt[2], "-" if pt[3] is None else pt[3])


def g1d_split(c, pt):
    """(seed slots floor(s x C), LRU slots C - seed - P); a point whose LRU part would be negative does not fit C."""
    seed = int(math.floor(pt[1] * c))
    return seed, c - seed - pt[2]


class G1dStats:
    """What a policy may use, fitted on the generated positions of the given (calibration) files only: per MoE layer the
    rank of every expert by routed count (cut's rule, ties lower id: the seed), and per look-ahead d the id-only table
    T[j][a, b] = generated positions with a among layer j's ids and b among layer j + d's ids (same token)."""

    def __init__(self, runs, shape=SHAPE, depths=G1D_GRID_D):
        L, E, _ = shape
        self.names = [r.name for r in runs]
        self.ranks = rank_table(gen_counts(runs, shape))
        sel = [r.routes[r.gen].astype(np.int64) for r in runs]
        self.tables = {}
        for d in depths:
            tab = []
            for j in range(L - d):
                c = np.zeros(E * E, np.int64)
                for x in sel:
                    c += np.bincount((x[:, j, :, None] * E + x[:, j + d, None, :]).ravel(), minlength=E * E)
                tab.append(c.reshape(E, E))
            self.tables[d] = tab


def g1d_keys(routes, j, d, stats):
    """Prefetch keys for layer j from the same token's layer j - d ids: score[e] = sum over the source ids a of
    T[j - d][a, e]; key = score x E + (E - 1 - e), so the highest key is the highest score, ties the lower id."""
    tab = stats.tables[d][j - d]
    E = tab.shape[0]
    src = routes[:, j - d, :]
    sc = np.zeros((len(routes), E), np.int64)
    for k in range(src.shape[1]):
        sc += tab[src[:, k]]
    return sc * E + (E - 1 - np.arange(E, dtype=np.int64))


def g1d_replay_layer(ids, seed, cl, p=0, keys=None):
    """One MoE layer of the per-layer arena in token order (ids [n][K] ascending, as the dumps hold them).
    Seed experts (bool [E]) are resident and never evicted; the LRU part holds cl experts. With p > 0, before the
    token's visit the p highest keys among the experts not resident fill the prefetch buffer, one NVMe read each; a
    buffered expert that is demanded joins the LRU part, the rest are dropped after the visit. A demand visit that is
    neither resident nor buffered is one NVMe read and joins the LRU part (with cl = 0 it is not kept).
    -> (NVMe reads per token, prefetch reads issued, buffered experts used)"""
    n, E = len(ids), len(seed)
    reads = np.zeros(n, np.int64)
    res = seed.copy()
    nres = int(res.sum())
    lru = OrderedDict()
    rows = ids.tolist()
    issued = used = 0
    for t in range(n):
        r, buf = 0, ()
        if p:
            q = min(p, E - nres)
            if q:
                k = np.where(res, -1, keys[t])
                buf = set(np.argpartition(k, E - q)[E - q:].tolist())
                r += q
                issued += q
        for e in rows[t]:
            if res[e]:
                if e in lru:
                    lru.move_to_end(e)
                continue
            if e in buf:
                used += 1
            else:
                r += 1
            if cl > 0:
                if len(lru) >= cl:
                    res[lru.popitem(last=False)[0]] = False
                    nres -= 1
                lru[e] = None
                res[e] = True
                nres += 1
        reads[t] = r
    return reads, issued, used


def g1d_replay(routes, c, pts, stats, start=0):
    """Replay a file from an empty arena at position `start` (0; the first generated position for the reset variant)
    under each point of pts (all with the same d) -> {point: (reads [n - start][L], prefetch issued, used)}."""
    n, L, _ = routes.shape
    ds = {pt[3] for pt in pts if pt[2]}
    if len(ds) > 1:
        raise SimError("g1d_replay: points of one call share d (%s)" % sorted(ds))
    out = {}
    for pt in pts:
        seed_n, cl = g1d_split(c, pt)
        if cl < 0:
            raise SimError("%s does not fit C %d (seed %d + P %d > C)" % (g1d_point_name(pt), c, seed_n, pt[2]))
        out[pt] = [np.zeros((n - start, L), np.int64), 0, 0]
    for j in range(L):
        keys = None
        for pt in pts:
            seed_n, cl = g1d_split(c, pt)
            p = pt[2] if pt[2] and j - pt[3] >= 0 else 0   # the source layer is a MoE layer of the same token
            if p and keys is None:
                keys = g1d_keys(routes, j, pt[3], stats)[start:]
            r, i, u = g1d_replay_layer(routes[start:, j, :], stats.ranks[j] < seed_n, cl, p, keys)
            o = out[pt]
            o[0][:, j] = r
            o[1] += i
            o[2] += u
    return {pt: tuple(v) for pt, v in out.items()}


def _g1d_task(args):
    routes, gen, c, pts, stats = args
    return {pt: float(v[0].sum(1)[gen].mean()) for pt, v in g1d_replay(routes, c, pts, stats).items()}


def g1d_scores(runs, c, pts, stats, jobs=1):
    """{point: [reads per generated token of each run, replayed alone from an empty arena]} (score = their mean)."""
    groups = {}
    for pt in pts:
        groups.setdefault(pt[3] if pt[2] else None, []).append(pt)
    tasks, keys = [], []
    for i, r in enumerate(runs):
        for g in groups.values():
            tasks.append((r.routes, r.gen, c, g, stats))
            keys.append(i)
    if jobs > 1 and len(tasks) > 1:
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(min(jobs, len(tasks))) as ex:
            got = list(ex.map(_g1d_task, tasks))
    else:
        got = [_g1d_task(t) for t in tasks]
    out = {pt: [None] * len(runs) for pt in pts}
    for i, res in zip(keys, got):
        for pt, v in res.items():
            out[pt][i] = v
    return out


def g1d_choose(mean, pts):
    """The lowest score; pts is in tie order, so an equal score keeps the earlier (simpler, smaller) point."""
    best = None
    for pt in pts:
        if pt in mean and (best is None or mean[pt] < mean[best]):
            best = pt
    return best


def g1d_select(cal, c, jobs=1, held=None, shape=SHAPE):
    """PREREG-dyn "Parameter selection" at capacity c: every grid point scored on the calibration files (statistics
    fitted on them), the chosen point, and the leave-one-out folds (point chosen on the other files with statistics
    fitted on those, scored on the one left out). The held-out never enters."""
    if held is not None and any(r is held or r.name == held.name for r in cal):
        raise SimError("the held-out file %s is in the calibration set" % held.name)
    pts = [pt for pt in g1d_grid() if g1d_split(c, pt)[1] >= 0]
    stats = G1dStats(cal, shape)
    per = g1d_scores(cal, c, pts, stats, jobs)
    mean = {pt: float(np.mean(v)) for pt, v in per.items()}
    chosen = g1d_choose(mean, pts)
    folds = []
    if len(cal) > 1:
        for f in cal:
            rest = [r for r in cal if r is not f]
            st = G1dStats(rest, shape)
            m = {pt: float(np.mean(v)) for pt, v in g1d_scores(rest, c, pts, st, jobs).items()}
            ch = g1d_choose(m, pts)
            folds.append({"left_out": f.name, "chosen": g1d_point_name(ch), "point": list(ch),
                          "score_on_rest": m[ch], "left_out_r": g1d_scores([f], c, [ch], st, jobs)[ch][0],
                          "equals_full_choice": ch == chosen})
    grid = [{"point": list(pt), "name": g1d_point_name(pt), "score": mean[pt],
             "per_file": dict(zip([r.name for r in cal], per[pt]))} for pt in pts]
    skipped = [g1d_point_name(pt) for pt in g1d_grid() if pt not in mean]
    return chosen, stats, {"grid": grid, "skipped_points": skipped, "folds": folds}


def g1d_verdict(r_stat, s_bytes, b, b_reason, reasons, primary=True):
    """PREREG-dyn "Judged metric": passed when r_hi x S <= B / 40 tok/s (r_hi the CI upper bound of reads per generated
    token of the held-out), else failed; not answered without B, without a CI, or on any source reason."""
    out = {"r": r_stat["mean"], "r_ci": r_stat["ci"], "r_hi": r_stat["ci"][1], "generated": r_stat["n"],
           "S": s_bytes, "B": b, "B_source": b_reason, "bar_reads": None, "bar_bytes": None, "r_hi_bytes": None,
           "gate": "G1d" if primary else "scenario (no gate role)", "status": G1D_STATUS}
    void = list(reasons)
    if b is None:
        void.append("B: " + b_reason)
    else:
        out["bar_reads"] = g1d_bar(b, s_bytes)
        out["bar_bytes"] = b * 1e9 / G1D_TOK_S
    if math.isnan(out["r_hi"]):
        void.append("held-out has %d generated positions, fewer than two blocks of %d: no CI" % (r_stat["n"], BLOCK))
    else:
        out["r_hi_bytes"] = out["r_hi"] * s_bytes
    word = out["gate"]
    if void:
        out["verdict"] = "%s not answered: %s" % (word, "; ".join(void))
    elif out["r_hi"] <= out["bar_reads"]:
        out["verdict"] = "%s passed: r_hi %.3f <= %.3f reads per token (r_hi x S %.2f MB <= B / 40 %.2f MB)" % (
            word, out["r_hi"], out["bar_reads"], out["r_hi_bytes"] / 1e6, out["bar_bytes"] / 1e6)
    else:
        out["verdict"] = "%s failed: r_hi %.3f > %.3f reads per token (r_hi x S %.2f MB > B / 40 %.2f MB)" % (
            word, out["r_hi"], out["bar_reads"], out["r_hi_bytes"] / 1e6, out["bar_bytes"] / 1e6)
    return out


def g1d_cells(spec):
    """'3.05:46:V25,...' or 'all' (bpw 3.05 / 3.5 / 4.5 x R 46 / 50 / 54 GiB x V25 / V37) -> [(bpw, R GiB, V)]"""
    if spec == "all":
        return [(b, r, v) for b in ("3.05", "3.5", "4.5") for r in G1D_R_GIB for v in G1D_V]
    out = []
    for x in spec.split(","):
        parts = x.split(":")
        if len(parts) != 3 or parts[0] not in G1D_S or parts[2] not in G1D_V or not parts[1].isdigit():
            raise SimError("cell %r: bpw:R_GiB:V with bpw %s and V %s" % (x, "/".join(G1D_S), "/".join(G1D_V)))
        out.append((parts[0], int(parts[1]), parts[2]))
    return out


def g1d_run(held, cal, cells, b=None, b_reason="no step-3 run given", reasons=(), jobs=1, choice_log=None,
            shape=SHAPE, out=print):
    """Every cell: selection on the calibration files, the chosen point written (choice_log) BEFORE the held-out is
    replayed, then the held-out row with that one point, the MIN reference and, for the primary cell, the verdict."""
    import datetime
    res = {"held": held.name, "cal": [r.name for r in cal], "arena": "layer", "B": b, "B_source": b_reason,
           "source_reasons": list(reasons), "status": G1D_STATUS, "grid": [g1d_point_name(p) for p in g1d_grid()],
           "cells": []}
    out("\nG1d (PREREG-dyn): held-out %s, calibration %s, per-layer arena; %s" % (
        held.name, ", ".join(r.name for r in cal), G1D_STATUS))
    out("B: %s" % b_reason)
    for r in reasons:
        out("  source: %s" % r)
    sel_cache = {}
    g = held.gen
    for bpw, rg, v in cells:
        c, s_bytes = g1d_capacity(bpw, rg, v, shape[0]), G1D_S[bpw]
        primary = (bpw, rg, v) == G1D_PRIMARY
        if c not in sel_cache:
            sel_cache[c] = g1d_select(cal, c, jobs, held, shape)
        chosen, stats, sel = sel_cache[c]
        cell = {"bpw": bpw, "R_GiB": rg, "V": v, "C": c, "S": s_bytes, "primary": primary,
                "chosen": g1d_point_name(chosen), "point": list(chosen)}
        out("\nbpw %s, R %d GiB, %s: C %d per layer, S %d B%s" % (bpw, rg, v, c, s_bytes,
                                                                  "  [primary cell]" if primary else ""))
        for row in sel["grid"]:
            out("  cal %-30s score %.4f  %s" % (row["name"], row["score"], "  ".join(
                "%s %.4f" % kv for kv in row["per_file"].items())))
        for f in sel["folds"]:
            out("  fold out %-20s chose %-30s rest %.4f  left-out r %.4f  %s" % (
                f["left_out"], f["chosen"], f["score_on_rest"], f["left_out_r"],
                "= full choice" if f["equals_full_choice"] else "differs from the full choice"))
        out("  chosen on the calibration: %s (score %.4f)" % (cell["chosen"], [x["score"] for x in sel["grid"]
                                                                             if x["point"] == list(chosen)][0]))
        if choice_log:
            with open(choice_log, "a", encoding="utf-8") as f:
                f.write(json.dumps({"utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
                                    "cell": [bpw, rg, v], "C": c, "chosen": cell["chosen"], "point": list(chosen),
                                    "held_out_scored": False}) + "\n")
        cell.update(sel)
        rr = g1d_replay(held.routes, c, [chosen], stats)[chosen]
        per_tok = rr[0].sum(1)
        cell["held_out"] = {"r": stat(per_tok[g]), "per_layer_r": [float(x) for x in rr[0][g].mean(0)],
                            "prefetch_issued_per_token": rr[1] / len(per_tok), "prefetch_used_per_token": rr[2] / len(per_tok)}
        pos = np.arange(len(g))
        cell["held_out"]["depth"] = [{"from": lo, "to": hi, "r": stat(per_tok[g & (pos >= lo) & (pos < (hi or 10 ** 12))])}
                                     for lo, hi in DEPTHS]
        gi = np.flatnonzero(g)
        cell["held_out"]["first_1000_generated"] = stat(per_tok[gi[:1000]])
        cell["held_out"]["after_first_1000"] = stat(per_tok[gi[1000:]])
        if len(gi):
            rz = g1d_replay(held.routes, c, [chosen], stats, start=int(gi[0]))[chosen][0].sum(1)
            cell["held_out"]["reset_at_first_generated"] = stat(rz[g[gi[0]:]])
        mn = dyn_min(held.routes, c)
        cell["min_r"] = float(mn[g].mean())
        cell["verdict"] = g1d_verdict(cell["held_out"]["r"], s_bytes, b, b_reason, reasons, primary)
        h = cell["held_out"]["r"]
        out("  held-out r %.3f [%.3f, %.3f] reads per token (n %d); Belady MIN %.3f; prefetch issued %.3f, used %.3f"
            % (h["mean"], *h["ci"], h["n"], cell["min_r"], cell["held_out"]["prefetch_issued_per_token"],
               cell["held_out"]["prefetch_used_per_token"]))
        out("  " + cell["verdict"]["verdict"])
        res["cells"].append(cell)
    return res

def g1d_cmd(a):
    cells = g1d_cells(a.cells)
    if a.jobs < 1:
        raise SimError("--jobs must be >= 1")
    held, cal, reasons = load_corpus(a.corpus, a.runs, source="g1d", capture=a.capture)
    cal_src = calibration_source(a, held, cal)
    b, why = b_from_step3(a.step3, a.readers)
    res = g1d_run(held, cal, cells, b, why, reasons, a.jobs, (a.json + ".choices.jsonl") if a.json else None)
    res["calibration_routing"] = cal_src
    if a.json:
        jdump(res, a.json, indent=1, default=float)
    return 0


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
    s.add_argument("--capture", help="calibration routing from the conversion's capture (PREREG-dyn amendment 2; "
                                     "plausibility only for G1)")
    s.add_argument("--step3")
    s.add_argument("--readers", type=int, help="reader count fixed by a PREREG amendment (amendment 5: 1)")
    s.add_argument("--windows", default="0,8,16,32")
    s.add_argument("--gate-window", type=int, default=0)
    s.add_argument("--json")
    d = sub.add_parser("dyn", help="dynamic expert-cache policies over the held-out routing (#178)")
    d.add_argument("--corpus", required=True)
    d.add_argument("--runs", required=True)
    d.add_argument("--capture", help="the calibration files' routing from the conversion's capture, L<ll>/ids.i32 + "
                                     "capture.json per MoE layer (PREREG-dyn amendment 2); --runs then needs only "
                                     "the held-out pass")
    d.add_argument("--step3")
    d.add_argument("--readers", type=int, help="reader count fixed by a PREREG amendment (amendment 5: 1)")
    d.add_argument("--rates", help='JSON {"R_pcie_gbps": x, "R_dram_gbps": y, "source": "..."}')
    d.add_argument("--bpw", default="4.5", help="comma list of %s" % ", ".join(BPW_BYTES))
    d.add_argument("--vram", help="VRAM budget for experts, e.g. 25.6GB")
    d.add_argument("--pinned", help="pinned budget for experts, e.g. 46GiB")
    d.add_argument("--slots", help="V:P[,V:P...] slots per layer (arena layer) or in all (arena global), "
                                   "instead of --vram/--pinned")
    d.add_argument("--arena", choices=("layer", "global"), default="layer")
    d.add_argument("--policies", default="lru,clock,lfu")
    d.add_argument("--admit-max", type=int, help="admission only at <= this many picks per step (sybil: 64)")
    d.add_argument("--lfu-halflife", type=float, default=64.0, help="LFU decay half-life in tokens")
    d.add_argument("--prefetch", default="none", help="comma list of none, oracle, precision in (0, 1]")
    d.add_argument("--depths", default="1", help="look-ahead in layers, comma list")
    d.add_argument("--pf-budget", type=int, help="prefetch NVMe reads per token")
    d.add_argument("--json")
    g = sub.add_parser("g1d", help="the judged G1d policy and verdict of PREREG-dyn (#178)")
    g.add_argument("--corpus", required=True)
    g.add_argument("--runs", required=True)
    g.add_argument("--capture", help="calibration routing from the conversion's capture (PREREG-dyn amendment 2)")
    g.add_argument("--step3")
    g.add_argument("--readers", type=int, help="reader count fixed by a PREREG amendment (amendment 5: 1)")
    g.add_argument("--cells", default="%s:%d:%s" % G1D_PRIMARY,
                   help="bpw:R_GiB:V[,...] or all (default the primary cell of PREREG-dyn amendment 1)")
    g.add_argument("--jobs", type=int, default=1, help="worker processes for the grid replays")
    g.add_argument("--json", help="the report; the chosen point per cell goes to <json>.choices.jsonl first")
    a = ap.parse_args(argv)
    try:
        if a.cmd == "corpus":
            return corpus_cmd(a)
        if a.cmd == "dyn":
            return dyn_cmd(a)
        if a.cmd == "g1d":
            return g1d_cmd(a)
        held, cal, reasons = load_corpus(a.corpus, a.runs, capture=a.capture)
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
