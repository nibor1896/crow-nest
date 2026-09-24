"""Evaluate hot-set variants on per-position routing dumps (hot-set recalibration
2026-09-24, criteria in decode_out/hotset-0924/PREREG.md).

  .venv-oracle/bin/python tools/hotset-eval.py <dump-dir> <mask-dir> <held-out-name> <cal-name>...

<dump-dir> holds <name>-routes.bin (CROW_ROUTE_DUMP_PREFILL) and <name>-counts.json
(routestats); <mask-dir> holds <name>-mask.json (tools/session_ids.py). Prints:
the self-test (dump counts == engine counters), every variant's held-out hit rate
on the GENERATED positions (primary) and on all positions, the bootstrap CI of the
difference against the current default, leave-one-out folds and depth buckets.
With --write <out.json> <variant> it also writes that variant's sidecar, cut on
the calibration files.
"""
import json
import os
import sys

import numpy as np

LAYERS, E, TOPK = 48, 512, 10
CURRENT = "decode_out/hotsets-M-longctx2100-n160.json"
OTHERS = {"tentasks": "decode_out/hotsets-M-tentasks-n160.json"}
NS = (155, 160)
BLOCK, RESAMPLES = 1000, 2000
LAMBDAS = (0.1, 0.3)


def load_routes(path):
    """-> uint16 [positions, LAYERS, TOPK]"""
    raw = np.fromfile(path, dtype=np.uint8)
    chunks, off, cur, layer_next = [], 0, None, 0
    while off < len(raw):
        l, t = np.frombuffer(raw[off:off + 8], dtype="<u4")
        off += 8
        ids = np.frombuffer(raw[off:off + t * TOPK * 2], dtype="<u2").reshape(t, TOPK)
        off += t * TOPK * 2
        assert l == layer_next, "record order: layer %d, expected %d" % (l, layer_next)
        if l == 0:
            cur = np.empty((t, LAYERS, TOPK), dtype=np.uint16)
        cur[:, l, :] = ids
        layer_next = (l + 1) % LAYERS
        if layer_next == 0:
            chunks.append(cur)
    assert layer_next == 0, "dump ends inside a chunk"
    return np.concatenate(chunks)


def gen_mask(path, n):
    m = json.load(open(path))
    assert m["tokens"] == n, "mask for %d tokens, dump has %d" % (m["tokens"], n)
    mask = np.zeros(n, dtype=bool)
    for a, b in m["spans"]:
        mask[a:b] = True
    return mask


def counts(routes, sel=None):
    r = routes if sel is None else routes[sel]
    return np.stack([np.bincount(r[:, l, :].ravel(), minlength=E) for l in range(LAYERS)])


def top(c, n):
    # frequency order, ties: lower id (the `decode warmup` rule)
    return [sorted(range(E), key=lambda e: (-c[l, e], e))[:n] for l in range(LAYERS)]


def member(sets, n):
    m = np.zeros((LAYERS, E), dtype=bool)
    for l, s in enumerate(sets):
        m[l, s[:n]] = True
    return m


def per_pos_hit(routes, sets, n):
    m = member(sets, n)
    hit = m[np.arange(LAYERS)[None, :, None], routes]      # [pos, L, K]
    return hit.reshape(len(routes), -1).mean(axis=1)


def boot_ci(diff, rng):
    nb = len(diff) // BLOCK
    blocks = diff[:nb * BLOCK].reshape(nb, BLOCK).mean(axis=1)
    idx = rng.integers(0, nb, size=(RESAMPLES, nb))
    means = blocks[idx].mean(axis=1)
    return np.percentile(means, [2.5, 97.5])


def variants(cal):
    """cal: list of (routes, mask) -> {name: sets at N 160 (frequency order)}"""
    P = sum(counts(r) for r, _ in cal)
    G = sum(counts(r, m) for r, m in cal)
    out = {"P": top(P, E), "G": top(G, E)}
    Ps = P / P.sum(axis=1, keepdims=True)
    Gs = G / G.sum(axis=1, keepdims=True)
    for lam in LAMBDAS:
        out["G+%gP" % lam] = top(Gs + lam * Ps, E)
    return out


def main():
    args = sys.argv[1:]
    write = None
    if "--write" in args:
        i = args.index("--write")
        write = (args[i + 1], args[i + 2])
        del args[i:i + 3]
    dump, maskdir, held, cal_names = args[0], args[1], args[2], args[3:]
    names = [held] + cal_names
    data = {}
    for nm in names:
        r = load_routes(os.path.join(dump, nm + "-routes.bin"))
        eng = np.array(json.load(open(os.path.join(dump, nm + "-counts.json")))["prefill_counts"])
        same = np.array_equal(counts(r), eng)
        mask = gen_mask(os.path.join(maskdir, nm + "-mask.json"), len(r))
        print("self-test %-28s positions %7d generated %6d  dump counts == engine counters: %s"
              % (nm, len(r), mask.sum(), same))
        assert same, "the dump disagrees with the engine's own counters"
        data[nm] = (r, mask)

    fixed = {"current": json.load(open(CURRENT))["sets"]}
    for k, p in OTHERS.items():
        fixed[k] = json.load(open(p))["sets"]
    rng = np.random.default_rng(20260924)

    def score(held_nm, cand, n):
        r, m = data[held_nm]
        return per_pos_hit(r[m], cand, n), per_pos_hit(r, cand, n)

    # ---- primary: held-out, variants cut on the calibration files ----
    var = variants([data[c] for c in cal_names])
    cands = dict(fixed, **var)
    r_h, m_h = data[held]
    oracle_g = top(counts(r_h, m_h), E)
    print("\nheld-out %s  (generated positions = primary; all positions = old primary)" % held)
    for n in NS:
        base_g, _ = score(held, fixed["current"], n)
        print("  N %d" % n)
        for name, sets in list(cands.items()) + [("oracle (held-out's own G)", oracle_g)]:
            g, a = score(held, sets, n)
            d = g - base_g
            lo, hi = boot_ci(d, rng)
            print("    %-26s gen %.3f  all %.3f  diff vs current %+.3f  95%% CI [%+.3f, %+.3f]"
                  % (name, g.mean(), a.mean(), d.mean(), lo, hi))

    # ---- leave-one-out over all four files ----
    print("\nleave-one-out (cut on three, score generated positions of the fourth), N 160")
    for out_nm in names:
        rest = [data[x] for x in names if x != out_nm]
        v = variants(rest)
        base = score(out_nm, fixed["current"], 160)[0].mean()
        line = "  held %-26s current %.3f" % (out_nm, base)
        for name, sets in v.items():
            line += "  %s %.3f" % (name, score(out_nm, sets, 160)[0].mean())
        print(line)

    # ---- depth buckets on the held-out ----
    print("\nheld-out by context depth (generated positions), N 160")
    pos = np.arange(len(r_h))
    for lo_, hi_ in ((0, 32000), (32000, 100000), (100000, 10 ** 9)):
        sel = m_h & (pos >= lo_) & (pos < hi_)
        if not sel.any():
            continue
        line = "  %6d-%-6s n %6d" % (lo_, "" if hi_ > 10 ** 8 else hi_, sel.sum())
        for name in ("current", "tentasks", "P", "G", "G+0.1P", "G+0.3P"):
            line += "  %s %.3f" % (name, per_pos_hit(r_h[sel], cands[name], 160).mean())
        print(line)

    if write:
        out, name = write
        if os.path.exists(out):
            sys.exit("%s exists - refusing to overwrite" % out)
        ref = json.load(open(CURRENT))
        doc = {k: v for k, v in ref.items() if k not in ("sets", "provenance")}
        doc["n_per_layer"] = 160
        doc["sets"] = [s[:160] for s in var[name]]
        doc["provenance"] = ("variant %s over %s (generated positions of Crow sessions, "
                             "CROW_ROUTE_DUMP_PREFILL), top-160/layer, frequency order; "
                             "tools/hotset-eval.py" % (name, ", ".join(cal_names)))
        json.dump(doc, open(out, "w"))
        print("\nwrote %s (%s)" % (out, name))


if __name__ == "__main__":
    main()
