# Stage-wise comparator for p14 (QSA indexer GPU probe) — p8 method.
# Re-computes the first 8 rows of every indexer stage in torch (f32, real
# weights) and compares against the GPU probe's stage dumps
# (probes/p14debug/gpu-*.f32). One max_abs line per stage.

import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextQSAIndexer,
    Qwen4ExpTextRotaryEmbedding,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-original")
DBG = os.path.join(HERE, "..", "probes", "p14debug")
LAYER = 3
T = 2560
N = 8  # rows compared per stage

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"

idx = Qwen4ExpTextQSAIndexer(text_cfg, layer_idx=LAYER)
rot = Qwen4ExpTextRotaryEmbedding(config=text_cfg)
index_json = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index_json["weight_map"]
prefix = f"model.language_model.layers.{LAYER}.self_attn.indexer."
state = {}
for name in idx.state_dict().keys():
    with safe_open(os.path.join(MODELS, wm[prefix + name]), framework="pt", device="cpu") as f:
        state[name] = f.get_tensor(prefix + name).to(torch.float32)
idx.load_state_dict(state, strict=True)
idx = idx.float().eval()
rot = rot.float().eval()

hidden = torch.from_numpy(np.fromfile(os.path.join(DBG, "qsa-hidden.f32"), dtype=np.float32).reshape(T, 2560)).clone()

NTOK = 32  # 8 blocks of pooling need the first 32 tokens

with torch.no_grad():
    h32 = hidden[:NTOK].unsqueeze(0)                   # [1, 32, 2560]
    pos = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rot(hidden.unsqueeze(0), pos)
    cos, sin = cos.float(), sin.float()                # [1, T, 64]

    qk32 = idx.index_qk_proj(h32)[0]                   # [32, 640]
    qk8 = qk32[:8]

    q, token_k = torch.split(qk8, [512, 128], dim=-1)
    q = idx.q_layernorm(q.reshape(N, 4, 128))
    raw_keys = qk32[:, 512:]                           # [32, 128] — 8 blocks worth

    def rope(x, positions):                            # x [..., 128], positions ints
        c = cos[0, positions, :]
        s = sin[0, positions, :]
        if x.dim() == 3:                               # [N, heads, 128] q case
            c = c.unsqueeze(1)
            s = s.unsqueeze(1)
        return torch.cat([
            x[..., :64] * c + torch.cat((-x[..., 32:64], x[..., :32]), dim=-1) * s,
            x[..., 64:],
        ], dim=-1)

    q_rot = rope(q, torch.arange(N))
    ncb = 8  # first 8 blocks (tokens 0..31)
    blocks = raw_keys.view(ncb, 4, 128)
    pooled_raw = blocks.float().mean(dim=1)            # [8, 128]
    pooled_normed = idx.k_layernorm(pooled_raw)
    pooled_rot = rope(pooled_normed, torch.arange(ncb) * 4)
    sc = torch.relu(torch.einsum("thd,bd->tbh", q_rot.float(), pooled_rot.float())).sum(dim=-1) / math.sqrt(128)

def gpu(tag):
    return torch.from_numpy(np.fromfile(os.path.join(DBG, f"gpu-{tag}.f32"), dtype=np.float32)).clone()

d = (gpu("qk") - qk8.reshape(-1)).abs().max().item()
print(f"  {'qk':14s} max_abs(gpu vs ref) = {d:.3e}")
for tag, ref in [
    ("q-normed", q.reshape(-1)),
    ("q-rot", q_rot.reshape(-1)),
    ("pooled-raw", pooled_raw.reshape(-1)),
    ("pooled-normed", pooled_normed.reshape(-1)),
    ("pooled-rot", pooled_rot.reshape(-1)),
]:
    d = (gpu(tag) - ref).abs().max().item()
    print(f"  {tag:14s} max_abs(gpu vs ref) = {d:.3e}")

g_scores = gpu("scores").reshape(N, 640)
r_scores = torch.nn.functional.pad(sc, (0, 640 - ncb), value=float("nan"))
mask = ~torch.isnan(r_scores)
d = (g_scores[mask] - r_scores[mask]).abs().max().item()
print(f"  {'scores':14s} max_abs(gpu vs ref) = {d:.3e}")
