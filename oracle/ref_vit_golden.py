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

crow-nest #122 (the dense 27B): `--config <hf config.json>` picks the model
(qwen3_5 -> Qwen3_5VisionModel, qwen4_exp -> Qwen4ExpVisionModel, the merger
width from vision_config.out_hidden_size) and `--mmproj <gguf>` loads the
weights from llama.cpp's F16 projector instead of the container, so both sides
run the identical F16 weights the engine uploads:
  .venv-oracle/bin/python oracle/ref_vit_golden.py <dump> \
      --config models/Qwen3.8-27B/config.json --mmproj models/Qwen3.8-27B/mmproj-F16.gguf
"""
import argparse
import json
import os
import struct
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnq_weights import CnqReader  # noqa: E402

ROOT = os.path.join(HERE, "..")
CONTAINER = os.path.join(ROOT, "converter", "Qwen3.8-Flash-Next-CNQ4.5-M.cnq")
CFG = os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-original", "config.json")

ap = argparse.ArgumentParser()
ap.add_argument("dump", nargs="?", default=os.path.join(ROOT, "decode_out", "vit-dump"))
ap.add_argument("--config", default=CFG, help="the HF config.json (model_type picks the vision classes)")
ap.add_argument("--container", default=CONTAINER, help="weights: the container's vit section (dequantized)")
ap.add_argument("--mmproj", default=None, help="weights: llama.cpp's F16 projector instead of the container (#122)")
args = ap.parse_args()
dump = args.dump


def read_f32(path):
    raw = open(path, "rb").read()
    return np.frombuffer(raw, dtype=np.float32).copy()


cfg = json.load(open(args.config, encoding="utf-8"))
if cfg["model_type"] == "qwen3_5":
    from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig as VisionConfig
    from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel as VisionModel
else:
    from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig as VisionConfig
    from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpVisionModel as VisionModel
OUT = cfg["vision_config"]["out_hidden_size"]
vcfg = VisionConfig.from_dict(cfg["vision_config"])
vcfg._attn_implementation = "eager"
model = VisionModel(vcfg).float().eval()

sd = {}
missing = []
if args.mmproj:
    # the inverse of engine/src/vit.rs `mmproj_plan`: the two temporal halves of
    # the Conv3d kernel are stacked back on axis 2
    sys.path.insert(0, os.path.expanduser("~/.local/share/crow/src/llama.cpp/gguf-py"))
    from gguf import GGUFReader
    from check_mmproj_vs_hf import plan as mmproj_plan
    print(f"weights: {args.mmproj} (F16 projector)")
    gt = {t.name: t for t in GGUFReader(args.mmproj).tensors}
    halves = {}
    for gname, (hname, t) in mmproj_plan().items():
        a = torch.from_numpy(np.array(gt[gname].data).astype(np.float32))
        if t is not None:
            halves[t] = a
            continue
        sd[hname[len("model.visual."):]] = a
    sd["patch_embed.proj.weight"] = torch.stack([halves[0], halves[1]], dim=2)
    for name, p in model.state_dict().items():
        if name not in sd:
            missing.append(name)
        else:
            sd[name] = sd[name].reshape(p.shape)
    assert not missing, f"missing in the projector: {missing}"
    model.load_state_dict(sd, strict=True)
    print(f"vision tower loaded: {len(sd)} tensors from the projector")
else:
    print(f"container: {args.container}")
    cnq = CnqReader(args.container)
    # load the container-dequantized weights; names map 1:1 after stripping
    # the model.visual. prefix
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
    eng_path = os.path.join(dump, "vit-embeds.f32")
    assert ref.shape == (n_visual, OUT), f"{img}: oracle shape {ref.shape}, expected ({n_visual}, {OUT})"
    row = {"image": img, "grid": meta["grid"], "n_visual": n_visual, "out_hidden": OUT}
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
        seg = eng[offs[idx] * OUT: offs[idx + 1] * OUT].reshape(n_visual, OUT)
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
