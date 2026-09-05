# Reference side of p13 (f32 generation-loop gate).
#
# Re-runs the FULL 48-layer text model forward on CPU (transformers 5.16.1,
# eager, f32) over the EXACT token sequence the GPU loop generated
# (probes/p13debug/gen-sequence.json), then compares per-position logits
# against probes/p13debug/gpu-logits.f32 — tol 5e-3 + greedy argmax match.
#
# Probe shortcuts mirrored on BOTH sides:
# - PLE (layers.1) skipped: layer.ple = None here (mirrors the GPU side).
# - QSA indexer dense by construction (T=12 < 2048; block_topk 512 >= 3
#   complete blocks — spec section 5.1, p7 evidence).
# Weights stream layer-by-layer from the safetensors shards (one decoder
# layer in RAM at a time — the whole model never fits 64 GB).

import json
import os

import numpy as np
import torch
from safetensors import safe_open

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-original")
DBG = os.path.join(HERE, "..", "probes", "p13debug")
TOL = 5e-3

seq = json.load(open(os.path.join(DBG, "gen-sequence.json")))
# the GPU trace carries prompt + generated tokens PLUS the final greedy
# candidate (the argmax AFTER the last processed position) — the reference
# forward runs over the tokens actually PROCESSED (one per logit row)
ids = seq["all_ids"][: seq["prompt_len"] + seq["generated"]]
T = len(ids)
cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"

index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]


def fetch(full):
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as f:
        return f.get_tensor(full).to(torch.float32)


def load_into(module, prefix):
    state = {}
    for name in module.state_dict().keys():
        full = prefix + name
        assert full in wm, f"missing in checkpoint: {full}"
        state[name] = fetch(full)
    module.load_state_dict(state, strict=True)
    return module.float().eval()


print(f"p13 ref: sequence {ids} (T={T})")
embed = fetch("model.language_model.embed_tokens.weight")   # [V,2560] f32
lm_head_w = fetch("lm_head.weight")                          # [V,2560] f32
mixer = load_into(Qwen4ExpTextGatedResidual(text_cfg, use_combine=False),
                  "model.language_model.hyper_connection_mixer.")
rotary = Qwen4ExpTextRotaryEmbedding(config=text_cfg).float().eval()

with torch.no_grad():
    input_ids = torch.tensor([ids], dtype=torch.long)
    emb = embed[input_ids]                                    # [1,T,2560]
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)  # text-only
    cos, sin = rotary(emb, position_ids)
    min_dtype = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
    mask = mask.masked_fill(torch.tril(torch.ones(T, T, dtype=torch.bool)) == 0, min_dtype)
    h = emb.repeat(1, 1, text_cfg.hc_count)                   # [1,T,10240]

    for layer_idx in range(text_cfg.num_hidden_layers):
        layer = Qwen4ExpTextDecoderLayer(text_cfg, layer_idx)
        if layer.ple is not None:
            layer.ple = None  # PLE skipped on BOTH sides (gated separately)
        layer = load_into(layer, f"model.language_model.layers.{layer_idx}.")
        mixed_a, hyper_a, injw_a = layer.attn_hyper_connection(h)
        if layer.layer_type == "linear_attention":
            o = layer.linear_attn(mixed_a, cache_params=None, attention_mask=None)
        else:
            o, _ = layer.self_attn(mixed_a, (cos, sin), attention_mask=mask, past_key_values=None)
        h1 = hyper_a + (o.unsqueeze(-2) * injw_a.unsqueeze(-1)).flatten(-2)
        mixed_m, hyper_m, injw_m = layer.mlp_hyper_connection(h1)
        o2 = layer.mlp(mixed_m)
        h = hyper_m + (o2.unsqueeze(-2) * injw_m.unsqueeze(-1)).flatten(-2)
        del layer
        print(f"  layer {layer_idx:2d} done", flush=True)

    h = mixer(h)                                              # [1,T,2560]
    logits = torch.nn.functional.linear(h, lm_head_w)[0]      # [T,V]

ref = logits.contiguous().numpy()          # [T,V]
ref.tofile(os.path.join(DBG, "ref-logits.f32"))

gpu = np.fromfile(os.path.join(DBG, "gpu-logits.f32"), dtype=np.float32).reshape(T, -1)
V = gpu.shape[1]
print(f"\np13 gate: logits [T={T}][V={V}] — gpu vs reference f32, tol {TOL}")
ok = True
for t in range(T):
    d = np.abs(gpu[t] - ref[t])
    g_am = int(np.argmax(gpu[t]))
    r_am = int(np.argmax(ref[t]))
    srt = np.sort(ref[t])
    margin_r = srt[-1] - srt[-2]
    line_ok = d.max() < TOL and g_am == r_am
    ok &= line_ok
    print(
        f"  pos {t:2d}: max_abs={d.max():.3e}  argmax gpu={g_am} ref={r_am} "
        f"match={g_am == r_am}  ref_top2_margin={margin_r:.4f}  {'OK' if line_ok else 'FAIL'}",
        flush=True,
    )
print("nan gpu:", int(np.isnan(gpu).sum()), " nan ref:", int(np.isnan(ref).sum()))
if ok and not np.isnan(gpu).any():
    print(f"p13: PASS — generation-loop logits match the transformers reference (tol {TOL})")
else:
    print("p13: FAIL")
    raise SystemExit(1)
