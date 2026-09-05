# MoE stage partitioning for p8: given that mixed-m already matches (3e-6),
# decide whether router selection, routing weights, shared expert or the
# routed-expert chain diverges.

import json
import os

import numpy as np
import torch
from safetensors import safe_open

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextDecoderLayer

MODELS = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-Flash-Next-original")
DBG = os.path.join(os.path.dirname(__file__), "..", "probes", "p8debug")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
T, E, TOPK = 8, 512, 10

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"
layer = Qwen4ExpTextDecoderLayer(text_cfg, layer_idx=0)
index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]
state = {}
for name in layer.state_dict().keys():
    full = f"model.language_model.layers.0.{name}"
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as f:
        state[name] = f.get_tensor(full).to(torch.float32)
layer.load_state_dict(state, strict=True)
layer = layer.float().eval()


def gpu(tag, shape):
    return torch.from_numpy(np.fromfile(os.path.join(DBG, f"{tag}.f32"), dtype=np.float32).reshape(*shape)).clone()


x = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "layer0-input.f32"), dtype=np.float32).reshape(1, T, 10240)).clone()
inp = torch.load(os.path.join(GOLDEN, "layer0-input.pt"), map_location="cpu", weights_only=True)
cos, sin = inp["cos"].float(), inp["sin"].float()

with torch.no_grad():
    mixed_a, hyper_a, injw_a = layer.attn_hyper_connection(x)
    gdn = layer.linear_attn(mixed_a, cache_params=None, attention_mask=None)
    x1 = hyper_a + (gdn.unsqueeze(-2) * injw_a.unsqueeze(-1)).flatten(-2)
    mixed_m, hyper_m, injw_m = layer.mlp_hyper_connection(x1)
    ref_moe = layer.mlp(mixed_m)

    logits_ref, scores_ref, idx_ref = layer.mlp.gate(mixed_m)
    shared_ref = torch.sigmoid(layer.mlp.shared_expert_gate(mixed_m)) * layer.mlp.shared_expert(mixed_m)

logits_gpu = gpu("gpu-logits", (T, E))
idx_gpu = gpu("gpu-route-idx", (T, TOPK)).to(torch.int64)
w_gpu = gpu("gpu-route-w", (T, TOPK))
shared_gpu = gpu("gpu-shared", (1, T, 2560))

print(f"logits   max_abs = {(logits_gpu - logits_ref).abs().max():.3e}")
same_sets = [set(idx_gpu[t].tolist()) == set(idx_ref[t].tolist()) for t in range(T)]
print(f"top-10 sets equal per token: {same_sets}")
same_order = [idx_gpu[t].tolist() == idx_ref[t].tolist() for t in range(T)]
print(f"top-10 order equal per token: {same_order}")
print(f"routing weights max_abs = {(w_gpu - scores_ref).abs().max():.3e}")
print(f"shared   max_abs = {(shared_gpu - shared_ref).abs().max():.3e}")

# routed experts with the REFERENCE math but the GPU routing decision
with torch.no_grad():
    out_gpu_route = torch.zeros(1, T, 2560)
    mm = mixed_m.reshape(T, 2560)
    for t in range(T):
        for j in range(TOPK):
            e = idx_gpu[t][j].item()
            wgt = w_gpu[t][j]
            gu = layer.mlp.experts.gate_up_proj[e]
            gate, up = (mm[t] @ gu.T).chunk(2, dim=-1)
            h = torch.nn.functional.silu(gate) * up
            out_gpu_route[0, t] += wgt * (layer.mlp.experts.down_proj[e] @ h)
    moe_gpu_total = gpu("gpu-moe", (1, T, 2560))
    print(f"moe gpu dump vs (ref experts + gpu routing): max_abs = {(moe_gpu_total - (out_gpu_route + shared_ref)).abs().max():.3e}")
    print(f"moe gpu dump vs ref moe:                     max_abs = {(moe_gpu_total - ref_moe).abs().max():.3e}")
