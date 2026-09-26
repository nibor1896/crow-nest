# Calibration statistics for the calibrated NVFP4 scales of the dense Qwen3.8-27B
# (Crow #300 phase 2, decode_out/p2-lh/PREREG.md).
#
# The CNQ container's own forward (the model as it runs today), f32 on the GPU, LAYER-STREAMED
# like ref_qwen35_logits.py but over many sequences at once: every window's hidden states stay on
# the card, one decoder layer at a time is filled, run over every window, and freed. For the
# input of every NVFP4 projection it accumulates, per 16-column input block b,
#     H_b = sum over tokens of x_b x_b^T      (16 x 16, f64)
# NVIDIA Model-Optimizer's Local-Hessian statistic; its diagonal is the imatrix form. Inputs that
# several projections share are accumulated once: q/k/v (attn_in), gate/up (mlp_in), GDN
# in_proj_qkv/in_proj_z (gdn_in); o_proj (o_in), GDN out_proj (gdn_out_in), down_proj (down_in),
# and lm_head (head_in, the final norm's output).
#
# Attention runs as torch SDPA (causal, no mask tensor): an eager [T, T] mask and score matrix at
# the 8,192-token windows would not fit beside the hidden states.
#
# Run: .venv-oracle/bin/python oracle/calib_qwen35_stats.py --ids decode_out/p2-lh/calib-ids.json \
#          --out decode_out/p2-lh/stats-H.pt

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_common import LM, WeightSource, build_meta, text_config

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

ap = argparse.ArgumentParser()
ap.add_argument("--ids", required=True, help="calib-ids.json: {'windows': [{'ids': [...]}, ...]}")
ap.add_argument("--out", required=True)
ap.add_argument("--weights", default="cnq")
ap.add_argument("--limit", type=int, default=0, help="first n windows only (smoke)")
ap.add_argument("--layers", type=int, default=0, help="first n layers only (smoke)")
args = ap.parse_args()

torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device("cuda")

# the NVFP4 projection -> the shared-input group it reads
GROUP_OF = {
    "self_attn.q_proj": "attn_in", "self_attn.k_proj": "attn_in", "self_attn.v_proj": "attn_in",
    "self_attn.o_proj": "o_in",
    "linear_attn.in_proj_qkv": "gdn_in", "linear_attn.in_proj_z": "gdn_in",
    "linear_attn.out_proj": "gdn_out_in",
    "mlp.gate_proj": "mlp_in", "mlp.up_proj": "mlp_in",
    "mlp.down_proj": "down_in",
}
HOOK_ON = {"self_attn.q_proj", "self_attn.o_proj", "linear_attn.in_proj_qkv", "linear_attn.out_proj",
           "mlp.gate_proj", "mlp.down_proj"}

wins = json.load(open(args.ids))["windows"]
if args.limit:
    wins = wins[:args.limit]
ids = [torch.tensor(w["ids"], dtype=torch.long) for w in wins]
n_tok = sum(len(x) for x in ids)
tc = text_config()
tc._attn_implementation = "sdpa"
L = args.layers or tc.num_hidden_layers
ws = WeightSource(args.weights)
print(f"calib stats: {len(ids)} windows, {n_tok} tokens, layers 0..{L - 1}, weights {args.weights}", flush=True)

stats, counts = {}, {}


def accumulate(key, x):
    x = x.reshape(-1, x.shape[-1])
    xb = x.reshape(x.shape[0], -1, 16)
    h = torch.einsum("tbi,tbj->bij", xb, xb).double()
    if key in stats:
        stats[key] += h
    else:
        stats[key] = h
    counts[key] = counts.get(key, 0) + x.shape[0]


t_start = time.time()
with torch.no_grad():
    E = f"{LM}embed_tokens.weight"
    emb = ws.get(E)                                   # [V, H] f32 on the host (BF16 in the container)
    hs = [emb[x].to(dev) for x in ids]                # one [T, H] per window, on the card
    del emb
    Tmax = max(len(x) for x in ids)
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval().to(dev)
    cos_all, sin_all = rotary(hs[0][None], torch.arange(Tmax, device=dev).view(1, Tmax))

    for i in range(L):
        t0 = time.time()
        layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, i), f"{LM}layers.{i}.").to(dev)
        t1 = time.time()
        handles = []
        for name, mod in layer.named_modules():
            if name in HOOK_ON:
                key = f"layers.{i}.{GROUP_OF[name]}"
                handles.append(mod.register_forward_pre_hook(lambda m, inp, key=key: accumulate(key, inp[0])))
        for w in range(len(hs)):
            T = hs[w].shape[0]
            out = layer(hs[w][None], position_embeddings=(cos_all[:, :T], sin_all[:, :T]),
                        attention_mask=None, past_key_values=None)
            hs[w] = (out[0] if isinstance(out, tuple) else out)[0]
        for hd in handles:
            hd.remove()
        del layer
        torch.cuda.empty_cache()
        mx = max(float(h.abs().max()) for h in hs)
        assert all(not h.isnan().any() for h in hs), f"NaN after layer {i}"
        print(f"  layer {i:2d} load {t1 - t0:5.1f}s run {time.time() - t1:6.1f}s |h|max {mx:9.2f} "
              f"GPU {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)

    if L == tc.num_hidden_layers:
        norm = ws.load(build_meta(Qwen3_5RMSNorm, tc.hidden_size, tc.rms_norm_eps), f"{LM}norm.").to(dev)
        for h in hs:
            accumulate("head_in", norm(h))

torch.save({"H": {k: v.cpu() for k, v in stats.items()}, "tokens": counts,
            "windows": len(ids), "total_tokens": n_tok, "ids_file": os.path.abspath(args.ids),
            "weights": ws.provenance(), "attn": "sdpa", "dtype": "f32 forward, f64 sums"}, args.out)
print(f"calib stats done in {time.time() - t_start:.0f} s -> {args.out} ({len(stats)} groups)")
