"""Crow #300 phase 2 (decode_out/p2-lh): the DIAG arm's weights for `converter --scales diag`.
Reads calib_qwen35_stats.py's stats-H.pt and writes, per statistic group, the diagonal of H
(sum x_i^2 per input column) as f32 - exactly the vector nvfp4_sim.search("diag") used
(H f64 -> diagonal -> f32). JSON floats of f32 values round-trip exactly.

Run: .venv-oracle/bin/python oracle/export_diag_stats.py decode_out/p2-lh/stats-H.pt decode_out/p2-lh/diag-stats.json"""
import hashlib
import json
import sys

import numpy as np
import torch

src, dst = sys.argv[1], sys.argv[2]
d = torch.load(src)
groups = {}
for k, H in sorted(d["H"].items()):
    v = torch.diagonal(H, dim1=1, dim2=2).to(torch.float32).reshape(-1).numpy()
    groups[k] = [float(x) for x in v]
sha = hashlib.sha256(open(src, "rb").read()).hexdigest()
json.dump({"what": "Crow #300 p2-lh DIAG weights: diag(H) per input column, f32", "stats": src,
           "stats_sha256": sha, "total_tokens": d["total_tokens"], "groups": groups}, open(dst, "w"))
print(f"{len(groups)} groups, {sum(len(v) for v in groups.values())} values -> {dst}")
