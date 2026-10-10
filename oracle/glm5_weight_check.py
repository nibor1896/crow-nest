"""glm5_weight_check.py — crow-nest #156 (GLM-5.3-Flash plan step 6): every tensor of a (partial)
glm5_next container against the FP8 originals it was converted from, plus the gate-0 summary of
its sidecar.

  python -I oracle/glm5_weight_check.py <container.cnq> <originals dir> --json out.json

Per tensor of the container:
  nvfp4    rel_rms = |W_cnq - W_fp8| / |W_fp8| (W_cnq through `converter dequant`, W_fp8 through
           the runner's FP8 back end, glm5_common.fp8_dequant), and mse_recomputed =
           mean((W_cnq - W_fp8)^2) next to the sidecar's `mse` (computed by the converter against
           ITS f32 of the FP8 source): equal to 1e-9 relative only if the converter and the oracle
           read the FP8 source to the same f32 (scale order, block size) - an indirect check, the
           converter's f32 is not dumped;
  fp8      for an FP8 source in addition: glm5_common.fp8_dequant == transformers Fp8Dequantize,
           bit for bit (torch.equal);
  bf16/f32 the container value == the original, exactly.
Sidecar (gate 0, `--scales mse`): per class tensors, values, MSE, MSE/ceil ratio, clipped count
and old-bound violation count, and every tensor with a violation by name (under `mse` the bound is
void by design: converter/README.md "Scale policy").
"""
import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402


def hf_fp8(w, s):
    """transformers' own FP8 block dequant (`Fp8Dequantize._dequantize_one`, the call the runner's
    proof uses in glm5_layerwise.write_hf_dequant_checkpoint), to f32"""
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    return Fp8Dequantize(None)._dequantize_one(w, s, torch.float32)


def sidecar_summary(path):
    by_class = defaultdict(lambda: {"tensors": 0, "n": 0, "sse": 0.0, "sse_ceil": 0.0, "clipped": 0, "violations": 0})
    viol, lines, other = [], {}, []
    with open(path) as f:
        for line in f:
            v = json.loads(line)
            if "record" in v:
                other.append(v)
                continue
            lines[v["name"]] = v
            if v.get("dtype") != "nvfp4":
                continue
            c = by_class[v["class"]]
            c["tensors"] += 1
            c["n"] += v["n"]
            c["sse"] += v["mse"] * v["n"]
            c["sse_ceil"] += v["mse_ceil"] * v["n"]
            c["clipped"] += v["max_abs_clipped"]
            c["violations"] += v["violations"]
            if v["violations"]:
                viol.append({"name": v["name"], "violations": v["violations"], "max_rel_err": v["max_rel_err"],
                             "clipped": v["max_abs_clipped"], "mse_ratio": v["mse_ratio"]})
    classes = {k: {"tensors": c["tensors"], "values": c["n"], "mse": c["sse"] / c["n"],
                   "mse_ratio_vs_ceil": c["sse"] / c["sse_ceil"], "clipped": c["clipped"],
                   "clipped_share": c["clipped"] / c["n"], "violations": c["violations"]}
               for k, c in sorted(by_class.items())}
    return classes, viol, lines, other


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cnq")
    ap.add_argument("originals")
    ap.add_argument("--json", required=True)
    a = ap.parse_args(argv)
    t0 = time.time()
    cs = G.ContainerSource(a.cnq)
    ws = G.WeightSource("fp8", a.originals)
    classes, viol, side, other = sidecar_summary(os.path.splitext(a.cnq)[0] + ".cnq.sidecar.jsonl")
    names = [t["name"] for t in cs.index["tensors"]]
    rows, bad = [], []
    n_fp8 = n_fp8_ok = n_keep = n_keep_ok = 0
    worst_mse_rel = 0.0
    for name, wc in zip(names, cs.get_many(names)):
        t = cs.tensors[name]
        wf = ws.get(name).reshape(wc.shape)
        r = {"name": name, "dtype": t["dtype"], "src_fp8": ws.is_fp8(name)}
        if ws.is_fp8(name):
            n_fp8 += 1
            w, s = _raw_pair(ws, name)
            same = torch.equal(G.fp8_dequant(w, s), hf_fp8(w, s))
            n_fp8_ok += same
            r["oracle_fp8_eq_hf"] = same
            if not same:
                bad.append(f"{name}: oracle fp8_dequant != HF Fp8Dequantize")
        if t["dtype"] == "nvfp4":
            d = (wc.double() - wf.double())
            r["rel_rms"] = float(d.norm() / wf.double().norm())
            r["mse_recomputed"] = float((d * d).mean())
            r["mse_sidecar"] = side[name]["mse"]
            rel = abs(r["mse_recomputed"] - r["mse_sidecar"]) / max(r["mse_sidecar"], 1e-300)
            r["mse_rel_diff"] = rel
            worst_mse_rel = max(worst_mse_rel, rel)
            if rel > 1e-9:
                bad.append(f"{name}: mse recomputed {r['mse_recomputed']:.6e} vs sidecar {r['mse_sidecar']:.6e}")
            r["class"] = side[name]["class"]
        else:
            n_keep += 1
            same = torch.equal(wc, wf)
            n_keep_ok += same
            r["exact"] = same
            if not same:
                bad.append(f"{name}: {t['dtype']} keep differs from the original")
        rows.append(r)
    per_class = defaultdict(list)
    for r in rows:
        if "rel_rms" in r:
            per_class[r["class"]].append(r["rel_rms"])
    rel_by_class = {k: {"tensors": len(v), "rel_rms_median": float(np.median(v)), "rel_rms_min": float(min(v)),
                        "rel_rms_max": float(max(v))} for k, v in sorted(per_class.items())}
    res = {
        "container": os.path.abspath(a.cnq), "originals": os.path.abspath(a.originals),
        "container_provenance": cs.provenance(), "tensors": len(rows),
        "nvfp4": sum(1 for r in rows if r["dtype"] == "nvfp4"), "keeps": n_keep, "keeps_exact": n_keep_ok,
        "fp8_sources": n_fp8, "fp8_oracle_eq_hf": n_fp8_ok, "mse_worst_rel_diff": worst_mse_rel,
        "problems": bad, "gate0_by_class": classes, "gate0_violation_tensors": len(viol),
        "gate0_violations_total": sum(v["violations"] for v in viol), "gate0_violations": viol,
        "rel_rms_by_class": rel_by_class, "sidecar_records": other, "per_tensor": rows,
        "seconds": round(time.time() - t0, 1),
    }
    with open(a.json, "w") as f:
        json.dump(res, f, indent=1)
    print(f"{len(rows)} tensors: {res['nvfp4']} nvfp4, keeps exact {n_keep_ok}/{n_keep}, FP8 oracle==HF {n_fp8_ok}/{n_fp8}, "
          f"worst |mse - sidecar mse| rel {worst_mse_rel:.2e}, problems {len(bad)}, {res['seconds']} s")
    for k, c in rel_by_class.items():
        g = classes[k]
        print(f"  {k:14s} {c['tensors']:4d} tensors  rel_rms median {c['rel_rms_median']:.4f} [{c['rel_rms_min']:.4f}, "
              f"{c['rel_rms_max']:.4f}]  mse/ceil {g['mse_ratio_vs_ceil']:.4f}  clipped {g['clipped']} "
              f"({100 * g['clipped_share']:.3f} %)  old-bound violations {g['violations']}")
    for p in bad[:20]:
        print("  PROBLEM", p)
    return 0 if not bad else 1


def _raw_pair(ws, name):
    with ws._open(name) as f:
        w = f.get_tensor(name)
    with ws._open(name + "_scale_inv") as f:
        s = f.get_tensor(name + "_scale_inv")
    return w, s


if __name__ == "__main__":
    sys.exit(main())
