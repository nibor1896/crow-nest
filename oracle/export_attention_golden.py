# Export an attention-layer golden for the GPU port (crow-nest p7 / Crow #188).
#
# What: text decoder layer 3 (first full_attention layer) — the self_attn MIXER
# sub-block only, same sub-block boundary as the GDN golden: input [1, 8, 2560]
# (the attn_hyper_connection mix input) -> Qwen4ExpTextAttention -> output
# [1, 8, 2560]. Hyper-connections and MoE stay OUTSIDE the sub-block.
#
# QSA note: at T=8 the indexer is dense by construction (visible 8 tokens form
# 2 complete blocks of compress_ratio 4, block_topk 512 >= 2 -> everything
# selected; spec section 5.1, PR #27742 max logit delta 0.0). The exported
# golden therefore pins the whole attention path without indexer selection.
#
# Weights: real checkpoint tensors, BF16 -> f32 (exact), layer run in f32.
# Provenance: spec section 5.3 — every artifact carries its object.

import json
import math
import os
import struct

import torch
from safetensors import safe_open

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextAttention,
    Qwen4ExpTextRotaryEmbedding,
)

MODELS = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-Flash-Next-original")
GOLDEN = os.path.join(os.path.dirname(__file__), "golden")
LAYER = 3  # layer_types pattern linear×3 + full -> first full_attention layer
T = 8
SEED = 20260902

os.makedirs(GOLDEN, exist_ok=True)
torch.manual_seed(SEED)

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"  # deterministic eager path, softmax f32

attn = Qwen4ExpTextAttention(text_cfg, layer_idx=LAYER)
rotary = Qwen4ExpTextRotaryEmbedding(config=text_cfg)

# suffix-match this module's params against the checkpoint index
index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]
state = {}
prefix = f"model.language_model.layers.{LAYER}.self_attn."
for name, tensor in attn.state_dict().items():
    full = prefix + name
    assert full in wm, f"missing in checkpoint: {full}"
    shard = wm[full]
    with safe_open(os.path.join(MODELS, shard), framework="pt", device="cpu") as f:
        state[name] = f.get_tensor(full).to(torch.float32)
missing, unexpected = attn.load_state_dict(state, strict=True), None
attn = attn.float().eval()
rotary = rotary.float().eval()
n_loaded = len(state)
print(f"loaded {n_loaded} tensors for {prefix}*")

with torch.no_grad():
    x = torch.randn(1, T, text_cfg.hidden_size, dtype=torch.float32)

    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)  # text-only: T/H/W equal
    cos, sin = rotary(x, position_ids)
    assert cos.shape == (1, T, 64) and cos.dtype == torch.float32

    min_dtype = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
    mask = mask.masked_fill(torch.tril(torch.ones(T, T, dtype=torch.bool)) == 0, min_dtype)

    out, attn_weights = attn(x, (cos, sin), attention_mask=mask, past_key_values=None)

assert out.shape == (1, T, text_cfg.hidden_size)
assert out.dtype == torch.float32
assert not out.isnan().any()

# every selection had all 8 tokens visible -> dense; verify via weights row sums
assert torch.allclose(attn_weights.sum(-1), torch.ones(1, 24, T), atol=1e-5), "not dense?"

x_f32 = x.reshape(-1).contiguous()
out_f32 = out.reshape(-1).contiguous()
x_f32.numpy().tofile(os.path.join(GOLDEN, f"layer{LAYER}-attn-input.f32"))
out_f32.numpy().tofile(os.path.join(GOLDEN, f"layer{LAYER}-attn-output.f32"))

torch.save(
    {
        "input": x, "output": out, "cos": cos, "sin": sin,
        "attn_weights": attn_weights, "mask": mask,
        "state_dict": {k: v.clone() for k, v in attn.state_dict().items()},
    },
    os.path.join(GOLDEN, f"layer{LAYER}-attn-subblock.pt"),
)

manifest = {
    "what": f"oracle golden: text decoder layer {LAYER} (full_attention) self_attn mixer sub-block, REAL weights, deterministic input",
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "config_source": os.path.relpath(MODELS, os.path.dirname(__file__)),
    "dtype": "float32",
    "seed": SEED,
    "layer": LAYER,
    "layer_type": cfg["text_config"]["layer_types"][LAYER],
    "attn_implementation": "eager",
    "qsa_note": "T=8 < dense threshold: indexer selects all visible tokens (2 complete blocks, block_topk 512) — attention path pinned dense by construction",
    "input_shape": list(x.shape),
    "output_shape": list(out.shape),
    "weights": f"{prefix}* ({n_loaded} tensors, suffix-matched)",
}
with open(os.path.join(GOLDEN, f"layer{LAYER}-attn-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)

print("wrote layer3-attn-input.f32 / -output.f32 / -subblock.pt / -manifest.json")
print(f"out stats: absmax={out.abs().max():.4f} mean={out.mean():.6f}")
