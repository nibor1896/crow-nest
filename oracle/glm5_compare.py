"""glm5_compare.py — crow-nest #156 (GLM-5.3-Flash plan step 6): per-layer agreement of two
layerwise runs of oracle/glm5_layerwise.py over the same ids (e.g. back A = FP8 originals,
back B = the CNQ container).

  python -I oracle/glm5_compare.py <run A dir> <run B dir> [--json out.json]

Per layer both runs hold, on `l<k>-output.*` ([N][hc][H], all hidden streams, all rows):
  cosine        <a, b> / (|a| |b|) over the flattened tensor, in f64 (the PREREG G3 metric form)
  max_abs       max |a - b|; also split into prompt rows (0..T-1) and decode rows (T..N-1)
  rel_rms       |a - b| / |a|
MoE layers: routed top-8 sets per token, the share of the 8 ids both runs picked (mean, min) and
the rows with identical sets. DSA layers: rows with identical indexer selections. Each run's
chain is its own: layer k of run B reads run B's layer k-1, so a difference accumulates.
"""
import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def _load(d, man, name):
    f = man["files"][name]
    path = os.path.join(d, name)
    if f["dtype"] == "bf16":
        a = np.fromfile(path, dtype="<u2").astype(np.uint32) << 16
        return a.view(np.float32).reshape(f["shape"])
    dt = {"f32": "<f4", "i32": "<i4"}[f["dtype"]]
    return np.fromfile(path, dtype=dt).reshape(f["shape"])


def compare_runs(da, db):
    ma = json.load(open(os.path.join(da, "manifest.json")))
    mb = json.load(open(os.path.join(db, "manifest.json")))
    for k in ("ids", "T", "D"):
        assert ma[k] == mb[k], f"{k} differs between {da} and {db}"
    T = ma["T"]
    rows = []
    for l in range(ma["num_hidden_layers"]):
        na = next((n for n in (f"l{l}-output.f32", f"l{l}-output.bf16") if n in ma["files"]), None)
        nb = next((n for n in (f"l{l}-output.f32", f"l{l}-output.bf16") if n in mb["files"]), None)
        if not (na and nb):
            continue
        a = _load(da, ma, na).astype(np.float64)
        b = _load(db, mb, nb).astype(np.float64)
        d = np.abs(a - b)
        r = {
            "layer": l, "kind": ma["layer_kinds"][l],
            "cosine": float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b))),
            "max_abs": float(d.max()),
            "max_abs_prompt": float(d[:T].max()),
            "max_abs_decode": float(d[T:].max()) if d.shape[0] > T else None,
            "rel_rms": float(np.linalg.norm(a - b) / np.linalg.norm(a)),
            "ref_rms": float(np.sqrt((a * a).mean())),
            "a_sha256": ma["files"][na]["sha256"], "b_sha256": mb["files"][nb]["sha256"],
        }
        ri = f"l{l}-routing-ids.i32"
        if ri in ma["files"] and ri in mb["files"]:
            ia, ib = _load(da, ma, ri), _load(db, mb, ri)
            overlap = np.array([len(set(x) & set(y)) / len(x) for x, y in zip(ia.tolist(), ib.tolist())])
            r["routing_overlap_mean"] = float(overlap.mean())
            r["routing_overlap_min"] = float(overlap.min())
            r["routing_rows_identical"] = int((overlap == 1.0).sum())
            r["routing_rows"] = int(len(overlap))
        tk = f"l{l}-dsa-topk.i32"
        if tk in ma["files"] and tk in mb["files"]:
            ta, tb = _load(da, ma, tk), _load(db, mb, tk)
            same = [set(x[x >= 0].tolist()) == set(y[y >= 0].tolist()) for x, y in zip(ta, tb)]
            r["dsa_rows_identical"] = int(sum(same))
            r["dsa_rows"] = len(same)
        rows.append(r)
    return {"a": os.path.abspath(da), "b": os.path.abspath(db), "a_weights": ma["weights"], "b_weights": mb["weights"],
            "T": T, "D": ma["D"], "N": len(ma["ids"]), "layers": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--json", default=None)
    x = ap.parse_args(argv)
    res = compare_runs(x.a, x.b)
    for r in res["layers"]:
        extra = ""
        if "routing_rows" in r:
            extra += f"  routing overlap mean {r['routing_overlap_mean']:.4f} min {r['routing_overlap_min']:.3f} " \
                     f"identical {r['routing_rows_identical']}/{r['routing_rows']}"
        if "dsa_rows" in r:
            extra += f"  dsa identical {r['dsa_rows_identical']}/{r['dsa_rows']}"
        print(f"layer {r['layer']:2d} {r['kind']:9s} cosine {r['cosine']:.8f}  max|d| {r['max_abs']:.4e} "
              f"(prompt {r['max_abs_prompt']:.4e}, decode {r['max_abs_decode'] if r['max_abs_decode'] is None else format(r['max_abs_decode'], '.4e')})"
              f"  rel_rms {r['rel_rms']:.4e}" + extra)
    if x.json:
        with open(x.json, "w") as f:
            json.dump(res, f, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
