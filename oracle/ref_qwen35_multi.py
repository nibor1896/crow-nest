# ref_qwen35_logits.py over several sequences in ONE layer-streamed pass (Crow #300 phase 2,
# decode_out/p2-lh): the weights of each layer are loaded once and run over every sequence, so
# an arm of the 3,470-row KLD set costs one model load instead of seven. Same math, same modules,
# same I/O per directory (<dir>/gen-sequence.json in, <dir>/ref-logits.f32 out); no parity report.
#
# Run: .venv-oracle/bin/python oracle/ref_qwen35_multi.py --weights <arm> --device cuda <dir> [<dir> ...]

import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_common import LM, WeightSource, build_meta, causal_mask, text_config

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="cnq")
ap.add_argument("--device", default="cuda")
ap.add_argument("--attn", default="eager")
ap.add_argument("--lm-head-chunk", type=int, default=16384)
ap.add_argument("dirs", nargs="+")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device(args.device)

seqs = []
for d in args.dirs:
    s = json.load(open(os.path.join(d, "gen-sequence.json")))
    row0 = s.get("row0_pos", 0)
    ids = s["all_ids"][:row0 + s["rows"]]
    seqs.append((d, ids, row0))
tc = text_config()
tc._attn_implementation = args.attn
ws = WeightSource(args.weights)
print(f"qwen35 multi: weights={args.weights} {len(seqs)} sequences, {sum(len(x[1]) for x in seqs)} tokens", flush=True)
t_start = time.time()

with torch.no_grad():
    E = f"{LM}embed_tokens.weight"
    hs = [torch.stack([ws.rows(E, i, i + 1)[0] for i in ids]).unsqueeze(0).to(dev) for _, ids, _ in seqs]
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval().to(dev)
    Tmax = max(len(x[1]) for x in seqs)
    cos_all, sin_all = rotary(hs[0], torch.arange(Tmax, device=dev).view(1, Tmax))
    for i in range(tc.num_hidden_layers):
        t0 = time.time()
        layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, i), f"{LM}layers.{i}.").to(dev)
        t1 = time.time()
        full = layer.block_type == "full_attention"
        for k, (_, ids, _) in enumerate(seqs):
            T = len(ids)
            mask = causal_mask(T).to(dev) if (full and args.attn == "eager") else None
            hs[k] = layer(hs[k], position_embeddings=(cos_all[:, :T], sin_all[:, :T]), attention_mask=mask,
                          past_key_values=None)
            assert not hs[k].isnan().any(), f"NaN after layer {i} in {seqs[k][0]}"
        del layer
        torch.cuda.empty_cache()
        print(f"  layer {i:2d} load {t1 - t0:5.1f}s run {time.time() - t1:5.2f}s", flush=True)

    norm = ws.load(build_meta(Qwen3_5RMSNorm, tc.hidden_size, tc.rms_norm_eps), f"{LM}norm.").to(dev)
    hn = [norm(h)[0][row0:] for h, (_, _, row0) in zip(hs, seqs)]
    V = ws.n_rows("lm_head.weight")
    logits = [torch.empty(h.shape[0], V, dtype=torch.float32) for h in hn]
    for r0 in range(0, V, args.lm_head_chunk):
        r1 = min(V, r0 + args.lm_head_chunk)
        w = ws.rows("lm_head.weight", r0, r1).to(dev).t()
        for k, h in enumerate(hn):
            logits[k][:, r0:r1] = (h @ w).cpu()

for (d, _, _), lg in zip(seqs, logits):
    lg.contiguous().numpy().astype("<f4").tofile(os.path.join(d, "ref-logits.f32"))
print(f"multi forward {time.time() - t_start:.0f} s -> {len(seqs)} x ref-logits.f32")
