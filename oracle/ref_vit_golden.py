"""ref_vit_golden.py — #VIT: the f32 vision-tower oracle.

Runs Qwen4ExpVisionModel (transformers 5.16.1) in f32 on CPU over the ENGINE's
own patch dumps (decode_out/vit-dump/imgN.patches.f32 — the exact normalized
patch rows the engine fed its tower), with weights dequantized from the CNQ
container (the same NVFP4 bytes the engine loads; orchestrator ruling
2026-09-14: the gate is math precision, weights identical both sides).

Writes per image:
  <dump>/imgN.vit-oracle.f32      the oracle visual embeddings [n_visual][2560]
  <dump>/vit-golden-stats.json    per-image + total parity stats vs the engine

Usage (repo root, oracle venv):
  .venv-oracle/Scripts/python.exe oracle/ref_vit_golden.py decode_out/vit-dump
"""
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnq_weights import CnqReader  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel

ROOT = os.path.join(HERE, "..")
CONTAINER = os.path.join(ROOT, "converter", "Qwen3.8-Flash-Next-CNQ4.5-M.cnq")
CFG = os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-original", "config.json")

dump = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "decode_out", "vit-dump")


def read_f32(path):
    raw = open(path, "rb").read()
    return np.frombuffer(raw, dtype=np.float32).copy()


print(f"container: {CONTAINER}")
cnq = CnqReader(CONTAINER)
cfg = json.load(open(CFG, encoding="utf-8"))
vcfg = Qwen4ExpVisionConfig.from_dict(cfg["vision_config"])
vcfg._attn_implementation = "eager"
model = Qwen4ExpVisionModel(vcfg).float().eval()

# load the container-dequantized weights; names map 1:1 after stripping
# the model.visual. prefix
sd = {}
missing = []
for name, p in model.state_dict().items():
    full = f"model.visual.{name}"
    if not cnq.has(full):
        missing.append(full)
        continue
    t = cnq.tensor_f32(full)
    sd[name] = t.reshape(model.state_dict()[name].shape)
assert not missing, f"missing in container: {missing}"
model.load_state_dict(sd, strict=True)
n_nvfp4 = sum(1 for t in cnq.tensors.values()
              if t.get("dtype") == "nvfp4" and t["name"].startswith("model.visual."))
print(f"vision tower loaded: {len(sd)} tensors ({n_nvfp4} nvfp4 in the vit section)")

stats = []
for meta_path in sorted(f for f in os.listdir(dump) if f.endswith(".meta.json")):
    img = meta_path[: -len(".meta.json")]
    meta = json.load(open(os.path.join(dump, meta_path), encoding="utf-8"))
    n_patches, n_visual = meta["n_patches"], meta["n_visual"]
    patches = read_f32(os.path.join(dump, f"{img}.patches.f32")).reshape(n_patches, 3 * 2 * 16 * 16)
    grid = torch.tensor([meta["grid"]], dtype=torch.long)
    with torch.no_grad():
        out = model(torch.from_numpy(patches), grid_thw=grid, return_dict=True)
    ref = out.pooler_output.contiguous().numpy().astype(np.float32)
    ref.tofile(os.path.join(dump, f"{img}.vit-oracle.f32"))
    eng_path = os.path.join(ROOT, "decode_out", "vit-dump", "vit-embeds.f32")
    row = {"image": img, "grid": meta["grid"], "n_visual": n_visual}
    eng_all_path = eng_path
    if os.path.exists(eng_all_path):
        # the engine file concatenates all images in order; slice by the
        # cumulative order of the meta files (sorted names = message order)
        eng = read_f32(eng_all_path)
        offs = [0]
        for m2 in sorted(f for f in os.listdir(dump) if f.endswith(".meta.json")):
            mm = json.load(open(os.path.join(dump, m2), encoding="utf-8"))
            offs.append(offs[-1] + mm["n_visual"])
        idx = sorted(f for f in os.listdir(dump) if f.endswith(".meta.json")).index(meta_path)
        seg = eng[offs[idx] * 2560: offs[idx + 1] * 2560].reshape(n_visual, 2560)
        d = np.abs(seg - ref)
        cos = (seg * ref).sum(1) / (np.linalg.norm(seg, axis=1) * np.linalg.norm(ref, axis=1) + 1e-30)
        row.update({
            "engine_max_abs": float(d.max()),
            "engine_median_abs": float(np.median(d)),
            "engine_rel_max": float((d / (np.abs(ref) + 1e-8)).max()),
            "engine_cos_min": float(cos.min()),
        })
    stats.append(row)
    print(f"{img}: grid {meta['grid']} n_visual {n_visual} "
          + (f"max_abs {row.get('engine_max_abs'):.4e} cos_min {row.get('engine_cos_min'):.6f}"
             if "engine_max_abs" in row else "(engine embeds not present yet)"))

out = os.path.join(dump, "vit-golden-stats.json")
json.dump(stats, open(out, "w", encoding="utf-8"), indent=1)
print(f"stats -> {out}")
