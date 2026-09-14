"""ref_vit_stages.py — #VIT bisect: run the oracle tower stage by stage on the
engine's patch dump and compare each stage against the engine's stage dumps
(stage-pe.f32 = after patch_embed + pos_embed, stage-block0.f32 = after vision
block 0). Weights are the container-dequantized f32 (identical to the
engine's, per the orchestrator ruling).

Usage (repo root): .venv-oracle/Scripts/python.exe oracle/ref_vit_stages.py decode_out/vit-dump
"""
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnq_weights import CnqReader  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpVisionConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpVisionModel,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

ROOT = os.path.join(HERE, "..")
dump = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "decode_out", "vit-dump")

cnq = CnqReader(os.path.join(ROOT, "converter", "Qwen3.8-Flash-Next-CNQ4.5-M.cnq"))
cfg = json.load(open(os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-original", "config.json"), encoding="utf-8"))
vcfg = Qwen4ExpVisionConfig.from_dict(cfg["vision_config"])
vcfg._attn_implementation = "eager"
model = Qwen4ExpVisionModel(vcfg).float().eval()
sd = {}
for name, p in model.state_dict().items():
    t = cnq.tensor_f32(f"model.visual.{name}")
    sd[name] = t.reshape(model.state_dict()[name].shape)
model.load_state_dict(sd, strict=True)

meta = json.load(open(os.path.join(dump, "img0.meta.json"), encoding="utf-8"))
n = meta["n_patches"]
grid = torch.tensor([meta["grid"]], dtype=torch.long)
patches = torch.from_numpy(
    np.fromfile(os.path.join(dump, "img0.patches.f32"), dtype=np.float32).reshape(n, -1))


def rd(name):
    p = os.path.join(dump, name)
    return np.fromfile(p, dtype=np.float32) if os.path.exists(p) else None


def cmp(tag, eng, ref):
    d = np.abs(eng - ref)
    cos = float((eng * ref).sum() / ((np.linalg.norm(eng) * np.linalg.norm(ref)) + 1e-30))
    print(f"{tag}: max_abs {d.max():.4e}  median {np.median(d):.4e}  cos {cos:.6f}")
    return d.max()


with torch.no_grad():
    # stage 1: patch embed + interpolated pos embed (model.forward lines 1-2)
    hidden = model.patch_embed(patches)
    ii, iw = get_vision_interpolation_indices_and_weights(
        grid, num_grid_per_side=model.num_grid_per_side, mode="bilinear",
        align_corners=True, spatial_merge_size=model.spatial_merge_size)
    pos = (model.pos_embed(ii) * iw[:, :, None]).sum(1)
    hidden = hidden + pos.to(hidden.dtype)
    eng = rd("stage-pe.f32")
    print(f"stage-pe shapes: eng {None if eng is None else eng.size} oracle {hidden.numel()}")
    if eng is not None:
        cmp("stage-pe ", eng, hidden.numpy().ravel())

    # rotary tables per the model forward
    position_ids = get_vision_position_ids(grid, model.spatial_merge_size)
    rot = model.rotary_pos_emb(position_ids)
    emb = torch.cat((rot, rot), dim=-1)
    pe = (emb.cos(), emb.sin())
    cu = torch.tensor([0, n], dtype=torch.int32)

    for bi, blk in enumerate(model.blocks):
        hidden = blk(hidden, cu_seqlens=cu, position_embeddings=pe)
        if bi == 0:
            eng0 = rd("stage-block0.f32")
            if eng0 is not None:
                cmp("stage-blk0", eng0, hidden.numpy().ravel())
            print("block 0 done; continuing to the merger...")

    merged = model.merger(hidden)
    print(f"merger out: {tuple(merged.shape)}")
    engm = rd("vit-embeds.f32")
    if engm is not None:
        cmp("merger   ", engm, merged.numpy().ravel())
