# Stage-wise comparator for p8 (full layer-0 GPU probe).
#
# Re-runs the reference layer-0 forward stage by stage in torch (f32, real
# weights), asserts the recomposed output matches the golden, then compares
# each stage against the GPU probe's dumped intermediates (probes/p8debug/).
# Prints one max_abs line per stage — the diverging stage shows immediately.

import json
import os

import numpy as np
import torch
from safetensors import safe_open

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextRMSNorm,
)

MODELS = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-Flash-Next-original")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
DBG = os.path.join(os.path.dirname(__file__), "..", "probes", "p8debug")
T = 8
LAYER = 0

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"

layer = Qwen4ExpTextDecoderLayer(text_cfg, layer_idx=LAYER)
index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]
state = {}
prefix = f"model.language_model.layers.{LAYER}."
for name in layer.state_dict().keys():
    full = prefix + name
    assert full in wm, f"missing in checkpoint: {full}"
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as f:
        state[name] = f.get_tensor(full).to(torch.float32)
layer.load_state_dict(state, strict=True)
layer = layer.float().eval()
print(f"loaded {len(state)} tensors for {prefix}*")

x = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "layer0-input.f32"), dtype=np.float32).reshape(1, T, 10240)).clone()
golden = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "layer0-golden-output.f32"), dtype=np.float32).reshape(1, T, 10240)).clone()
inp = torch.load(os.path.join(GOLDEN, "layer0-input.pt"), map_location="cpu", weights_only=True)
cos, sin = inp["cos"].float(), inp["sin"].float()


def gpu(tag):
    return torch.from_numpy(np.fromfile(os.path.join(DBG, f"gpu-{tag}.f32"), dtype=np.float32).reshape(1, T, -1)).clone()


def gpu_hc(tag, shape):
    return torch.from_numpy(np.fromfile(os.path.join(DBG, f"hc-{tag}.f32"), dtype=np.float32).reshape(*shape)).clone()


def cmp(tag, ref):
    g = gpu(tag)
    d = (g - ref).abs().max().item()
    print(f"  {tag:10s} max_abs(gpu vs ref) = {d:.3e}")


def cmp_hc(tag, ref, shape):
    g = gpu_hc(tag, shape)
    d = (g - ref).abs().max().item()
    print(f"  {tag:14s} max_abs(gpu vs ref) = {d:.3e}")


with torch.no_grad():
    # hc internals first: normed / mixw / injw for the attn block ("a")
    ahc = layer.attn_hyper_connection
    x_flat = x.reshape(-1, 10240)
    normed_ref = ahc.hc_norm(x).float()
    low_ref = ahc.input_mix_weight_down(normed_ref)
    sil_ref = torch.nn.functional.silu(low_ref / ahc.hc_count)
    mixw_ref = torch.sigmoid(ahc.input_mix_weight_up(sil_ref))
    mixed_ref = (mixw_ref.unflatten(-1, (4, 2560)) * normed_ref.unflatten(-1, (4, 2560))).mean(dim=-2)
    injw_ref = 2 * torch.sigmoid(ahc.block_inject_weight(normed_ref) / ahc.hc_count)
    print("hc internals (attn block):")
    cmp_hc("normed-a", normed_ref, (1, T, 10240))
    cmp_hc("mixw-a", mixw_ref, (1, T, 10240))
    cmp_hc("injw-a", injw_ref, (1, T, 4))
    print(f"  low_ref stats: absmax={low_ref.abs().max():.4f}")

    # stage-by-stage, mirroring DecoderLayer.forward (layer 0: ple=None, GDN)
    mixed_a, hyper_a, injw_a = ahc(x)
    cmp("mixed-a", mixed_a)

    gdn = layer.linear_attn(mixed_a, cache_params=None, attention_mask=None)
    cmp("gdn-out", gdn)

    x1 = hyper_a + (gdn.unsqueeze(-2) * injw_a.unsqueeze(-1)).flatten(-2)
    mixed_m, hyper_m, injw_m = layer.mlp_hyper_connection(x1)
    cmp("x1", x1)
    cmp("mixed-m", mixed_m)

    moe = layer.mlp(mixed_m)
    cmp("moe", moe)

    out = hyper_m + (moe.unsqueeze(-2) * injw_m.unsqueeze(-1)).flatten(-2)

self_d = (out - golden).abs().max().item()
print(f"  reference recomposition vs golden: max_abs = {self_d:.3e}")
assert self_d < 5e-3, "reference re-run does not reproduce the golden — setup broken"
print("reference setup verified; stage-wise comparison above pinpoints the diverging stage")
