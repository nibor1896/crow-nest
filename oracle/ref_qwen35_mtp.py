# The f32 reference of the dense Qwen3.8-27B's MTP head (crow-nest #95, decode_out/p2-mtp/PREREG.md C1).
#
# transformers ignores `^mtp.*` on load, so the head is assembled here from its modules, following
# vLLM `qwen3_5_mtp.py` (forward) and llama.cpp `qwen35.cpp` (graph_mtp):
#   x_p = fc(cat[pre_fc_norm_embedding(embed(t_{p+1})), pre_fc_norm_hidden(h_p)])
#   x   = mtp.layers.0(x)    one gated full-attention decoder layer with its own causal KV, RoPE at p
#   out = mtp.norm(x) -> the shared lm_head          (h_p = the main model's hidden after model.norm)
# Teacher-forced over one sequence: row p of the output is the head's draft for position p + 2 made
# from (h_p, t_{p+1}); rows 0..T-2. The main model runs first (layer-streamed, GPU f32 like
# ref_qwen35_multi.py) and its argmax per row gives the head's teacher-forced greedy acceptance:
# draft row p is accepted iff it equals the main model's argmax at row p + 1.
#
#   <dir>/gen-sequence.json in; <dir>/mtp-logits.f32 [T-1][248320] and <dir>/mtp-ref.json out
# Run: .venv-oracle/bin/python oracle/ref_qwen35_mtp.py [--weights cnq] <dir>

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
ap.add_argument("--lm-head-chunk", type=int, default=16384)
ap.add_argument("dir")
args = ap.parse_args()
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device(args.device)

seq = json.load(open(os.path.join(args.dir, "gen-sequence.json")))
ids = seq["all_ids"][:seq["rows"]]
T = len(ids)
tc = text_config()
ws = WeightSource(args.weights)
H = tc.hidden_size
# a full-attention layer index, so the module is built as gated full attention (`mtp.layers.0`'s shape)
FULL = next(i for i, t in enumerate(tc.layer_types) if t == "full_attention")
t_start = time.time()


def head(h):
    """[R, H] -> [R, V] logits through the shared lm_head, chunked over the vocabulary"""
    V = ws.n_rows("lm_head.weight")
    out = torch.empty(h.shape[0], V, dtype=torch.float32)
    for r0 in range(0, V, args.lm_head_chunk):
        r1 = min(V, r0 + args.lm_head_chunk)
        out[:, r0:r1] = (h @ ws.rows("lm_head.weight", r0, r1).to(dev).t()).cpu()
    return out


with torch.no_grad():
    E = f"{LM}embed_tokens.weight"
    emb = torch.stack([ws.rows(E, i, i + 1)[0] for i in ids]).to(dev)        # [T, H]
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval().to(dev)
    cos, sin = rotary(emb[None], torch.arange(T, device=dev).view(1, T))
    mask = causal_mask(T).to(dev)
    h = emb[None]
    for i in range(tc.num_hidden_layers):
        layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, i), f"{LM}layers.{i}.").to(dev)
        full = layer.block_type == "full_attention"
        h = layer(h, position_embeddings=(cos, sin), attention_mask=mask if full else None, past_key_values=None)
        del layer
    norm = ws.load(build_meta(Qwen3_5RMSNorm, H, tc.rms_norm_eps), f"{LM}norm.").to(dev)
    hn = norm(h)[0]                                                           # [T, H], after model.norm
    main_am = head(hn).argmax(-1)                                            # [T]
    t_main = time.time()

    # ---- the MTP head over the pairs (h_p, t_{p+1}), p = 0..T-2 ----
    R = T - 1
    nrm = lambda name: ws.load(build_meta(Qwen3_5RMSNorm, H, tc.rms_norm_eps), f"mtp.{name}.").to(dev)
    e = nrm("pre_fc_norm_embedding")(emb[1:])
    hh = nrm("pre_fc_norm_hidden")(hn[:-1])
    fc = ws.get("mtp.fc.weight").to(dev)                                     # [H, 2H]
    x = torch.cat([e, hh], dim=-1) @ fc.t()                                  # [R, H]
    layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, FULL), "mtp.layers.0.").to(dev)
    x = layer(x[None], position_embeddings=(cos[:, :R], sin[:, :R]), attention_mask=causal_mask(R).to(dev),
              past_key_values=None)
    out = nrm("norm")(x[0])
    mtp_logits = head(out)                                                   # [R, V]

mtp_logits.contiguous().numpy().astype("<f4").tofile(os.path.join(args.dir, "mtp-logits.f32"))
mtp_am = mtp_logits.argmax(-1)
main_am = main_am.cpu()
n = R - 1                                                                     # row p needs main row p + 1
acc_main = int((mtp_am[:n] == main_am[1:R]).sum())
acc_text = int((mtp_am[:T - 2] == torch.tensor(ids[2:])).sum())
# on the model's OWN continuation (rows from prompt_len - 1 on, when the sequence carries one): the
# acceptance speculative decoding sees
pl = seq.get("prompt_len", T)
gen = list(range(max(pl - 1, 0), n))
acc_gen = (sum(int(mtp_am[p] == main_am[p + 1]) for p in gen) / len(gen)) if gen else None
res = {"rows": R, "tokens": T, "weights": args.weights, "mtp_layer_built_as": FULL,
       "accept_on_generated": acc_gen, "generated_rows": len(gen),
       "accept_vs_main_argmax": acc_main / n, "accept_vs_text": acc_text / (T - 2),
       "main_s": round(t_main - t_start, 1), "total_s": round(time.time() - t_start, 1)}
json.dump(res, open(os.path.join(args.dir, "mtp-ref.json"), "w"), indent=1)
print(json.dumps(res))
