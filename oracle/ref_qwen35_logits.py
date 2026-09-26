# Reference side of the ENGINE end-to-end parity for the dense Qwen3.8-27B
# (qwen3_5_text) — Crow #300 phase 2. Same I/O contract as ref_engine_logits.py:
#
#   <dir>/gen-sequence.json   {"rows", "prompt_len", "all_ids"}: exactly `rows`
#                             tokens (all_ids[:rows]) produced the `rows` logit rows
#   <dir>/gpu-logits.f32      engine logits [rows][248320] f32 (optional)
#   <dir>/ref-logits.f32      written here, [rows][248320] f32
#   <dir> = $CROW_PARITY_DIR, default decode_out/
#
# The full 64-layer text forward on CPU f32 (transformers 5.16.1 modules, eager),
# LAYER-STREAMED: one Qwen3_5DecoderLayer at a time is built on the meta
# device, filled, run over all positions, and freed. Text-only positions: 2-D
# position_ids through Qwen3_5TextRotaryEmbedding (the interleaved mrope rows
# are all equal -> 1-D rope over dims 0..63). Embedding rows and lm_head chunks
# are read by seek, never the whole table.
#
#   --weights cnq   (default) the CNQ4.5 container DEQUANTIZED: identical numbers
#                   to the engine, so the deltas are engine math
#   --weights bf16  the BF16 originals: the deltas include the quantization error
#
# Per position: argmax match, top-5 overlap, max |dlogit|, KL(ref || gpu), the
# reference top-2 margin. Exit 1 on an argmax mismatch or NaN (like #11's gate).
#
# Run: CROW_PARITY_DIR=<dir> .venv-oracle/bin/python oracle/ref_qwen35_logits.py [--weights cnq|bf16]

import argparse
import json
import os
import resource
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_common import LM, MODEL_DIR, ROOT, WeightSource, build_meta, causal_mask, text_config

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default="cnq", help="cnq | bf16 | r1 | x126 | diag | lh | only-mlp | only-lmhead | only-gdn | only-attn")
ap.add_argument("--device", default="cpu", help="cpu (the reference of record) | cuda (TF32 off; decode_out/p2-lh check 1)")
ap.add_argument("--attn", default="eager", help="eager | sdpa")
ap.add_argument("--threads", type=int, default=int(os.environ.get("ORACLE_THREADS", "16")))
ap.add_argument("--lm-head-chunk", type=int, default=16384, help="lm_head rows dequantized per step")
args = ap.parse_args()
torch.set_num_threads(args.threads)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
torch.set_float32_matmul_precision("highest")
dev = torch.device(args.device)


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


OUT = os.environ.get("CROW_PARITY_DIR") or os.path.join(ROOT, "decode_out")
seq = json.load(open(os.path.join(OUT, "gen-sequence.json")))
rows = seq["rows"]
# Crow #300 phase 2: with CROW_PARITY_TAIL the engine's logits start at position `row0_pos`
row0 = seq.get("row0_pos", 0)
ids = seq["all_ids"][:row0 + rows]
T = len(ids)
assert rows == T - row0, f"logit rows {rows} != processed tokens {T} - row0 {row0}"

tc = text_config()
tc._attn_implementation = args.attn
ws = WeightSource(args.weights)
print(f"qwen35 ref: weights={args.weights} T={T} rows {row0}..{T - 1} prompt_len={seq.get('prompt_len')} ids={ids if T <= 256 else '[' + str(T) + ' ids]'}")
t_start = time.time()

with torch.no_grad():
    E = f"{LM}embed_tokens.weight"
    h = torch.stack([ws.rows(E, i, i + 1)[0] for i in ids]).unsqueeze(0).to(dev)  # [1,T,5120]
    rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval().to(dev)
    cos, sin = rotary(h, torch.arange(T, device=dev).view(1, T))
    mask = causal_mask(T).to(dev) if args.attn == "eager" else None

    for i in range(tc.num_hidden_layers):
        t0 = time.time()
        layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, i), f"{LM}layers.{i}.").to(dev)
        t1 = time.time()
        full = layer.block_type == "full_attention"
        # the GDN takes a 2-D padding mask only; a single unpadded sequence needs none
        h = layer(h, position_embeddings=(cos, sin), attention_mask=mask if full else None,
                  past_key_values=None)
        del layer
        assert not h.isnan().any(), f"NaN after layer {i}"
        print(f"  layer {i:2d} {'attn' if full else 'gdn '} load {t1 - t0:5.1f}s run {time.time() - t1:5.2f}s "
              f"|h|max {h.abs().max():9.2f}  peak RSS {rss_gb():.1f} GB", flush=True)

    norm = ws.load(build_meta(Qwen3_5RMSNorm, tc.hidden_size, tc.rms_norm_eps), f"{LM}norm.").to(dev)
    h = norm(h)[0]                                                           # [T,5120]
    # Crow #300 phase 2: `row0_pos` (decode parity with CROW_PARITY_TAIL) - the engine collected
    # logits for positions row0_pos..T only; the head runs on those rows (8k x 248k f32 = 8 GB)
    h = h[row0:]
    V = ws.n_rows("lm_head.weight")
    logits = torch.empty(T - row0, V, dtype=torch.float32)
    for r0 in range(0, V, args.lm_head_chunk):
        r1 = min(V, r0 + args.lm_head_chunk)
        logits[:, r0:r1] = (h @ ws.rows("lm_head.weight", r0, r1).to(dev).t()).cpu()

ref = logits.contiguous().numpy()
ref.astype("<f4").tofile(os.path.join(OUT, "ref-logits.f32"))
print(f"ref forward {time.time() - t_start:.0f} s, peak RSS {rss_gb():.1f} GB -> {os.path.join(OUT, 'ref-logits.f32')}")

try:
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    dec = lambda i: repr(tok.decode([int(i)]))
except Exception as e:  # the tokenizer is a nicety, not part of the gate
    print(f"(no tokenizer: {e})")
    dec = lambda i: ""
am = ref.argmax(-1)
print("ref greedy next token per position:", " ".join(f"{int(a)}{dec(a)}" for a in am))

gpu_path = os.path.join(OUT, "gpu-logits.f32")
if not os.path.exists(gpu_path):
    print(f"reference only: {T - row0} rows (no gpu-logits.f32 to gate against)")
    raise SystemExit(0)

R = T - row0
gpu = np.fromfile(gpu_path, dtype=np.float32).reshape(R, -1)
assert gpu.shape[1] == V, f"gpu-logits width {gpu.shape[1]} != vocab {V}"
print(f"ENGINE parity: [rows={R} from pos {row0}][V={V}] engine vs f32 reference ({args.weights} weights)")


def log_softmax64(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


worst, match, top5, kls, margins = 0.0, 0, [], [], []
for t in range(R):
    d = np.abs(gpu[t] - ref[t])
    worst = max(worst, float(d.max()))
    g_am, r_am = int(np.argmax(gpu[t])), int(np.argmax(ref[t]))
    srt = np.sort(ref[t])
    margins.append(float(srt[-1] - srt[-2]))
    o5 = len(set(np.argsort(ref[t])[-5:]) & set(np.argsort(gpu[t])[-5:]))
    lr, lg = log_softmax64(ref[t]), log_softmax64(gpu[t])
    kl = float((np.exp(lr) * (lr - lg)).sum())
    match += g_am == r_am
    top5.append(o5)
    kls.append(kl)
    print(f"  pos {row0 + t:5d}: max_abs={d.max():.3e} argmax gpu={g_am} ref={r_am} match={g_am == r_am} "
          f"top5={o5}/5 KL={kl:.3e} ref_top2_margin={margins[-1]:.4f}")
print(f"worst |dlogit| {worst:.3e} | argmax {match}/{R} | mean top-5 overlap {np.mean(top5):.2f}/5 | "
      f"mean KL {np.mean(kls):.3e} max KL {np.max(kls):.3e} | min ref margin {min(margins):.4f}")
nan = int(np.isnan(gpu).sum())
print("nan gpu:", nan)
if match == R and nan == 0:
    print("engine parity: PASS (argmax trace matches the f32 reference)")
else:
    print("engine parity: ARGMAX MISMATCH - investigate")
    raise SystemExit(1)
