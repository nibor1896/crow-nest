# Export a QSA indexer golden for the GPU port (crow-nest p14 / Crow #10).
#
# What: the Qwen4ExpTextQSAIndexer of layer 3 (first full_attention layer),
# REAL weights, deterministic input, run in the SPARSE regime so the top-k
# selection is non-trivial: T=2560 → 640 complete blocks, block_topk 512
# → 128 blocks dropped per query (below the 2048-token budget the indexer
# is dense by construction — that free gate is p13's evidence, not this one).
#
# Reference math (modeling_qwen4_exp.py L611-717, transcription checked):
#   qk = index_qk_proj(hidden)                  # 2560 → (4+1)*128
#   q  = q_layernorm(q)  (RMSNorm 1+w, per head over 128)
#   q  = rotary(q, pos t)                        # first 64 dims, theta 1e7
#   per query t (visible = 0..t, causal, no padding):
#     blocks b = tokens [4b, 4b+4);  ncb = (t+1) // 4
#     pooled[b] = k_layernorm(mean(raw_keys[4b:4b+4]))   # raw keys UNGNORMED
#     pooled[b] = rotary(pooled[b], pos 4b)              # block start position
#     score[b]  = Σ_h relu(Σ_d q[t,h,d]·pooled[b,d]) / sqrt(128)
#     topk(min(512, ncb)) blocks → 4 token ids each (in SCORE order,
#       torch.topk .indices semantics: value desc), plus the tail tokens
#       [4·ncb .. t] appended ascending, rest padded with -1.
#
# Dumps (probes/p14debug/):
#   qsa-hidden.f32      [T, 2560]  input
#   qsa-selected.i32    [T, 2051]  selected token ids per query (-1 padded)
#   qsa-scores.f32      [T, 640]   block scores per query (pooled keys shared
#                                  across queries — identical math per query)
#   qsa-manifest.json

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
SEED = 20260902

os.makedirs(DBG, exist_ok=True)
torch.manual_seed(SEED)

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"

indexer = Qwen4ExpTextQSAIndexer(text_cfg, layer_idx=LAYER)
rotary = Qwen4ExpTextRotaryEmbedding(config=text_cfg)

index_json = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index_json["weight_map"]
prefix = f"model.language_model.layers.{LAYER}.self_attn.indexer."
state = {}
for name in indexer.state_dict().keys():
    full = prefix + name
    assert full in wm, f"missing in checkpoint: {full}"
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as f:
        state[name] = f.get_tensor(full).to(torch.float32)
indexer.load_state_dict(state, strict=True)
indexer = indexer.float().eval()
rotary = rotary.float().eval()
print(f"loaded {len(state)} indexer tensors ({list(state.keys())})")

with torch.no_grad():
    hidden = torch.randn(1, T, text_cfg.hidden_size, dtype=torch.float32)
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rotary(hidden, position_ids)
    assert cos.shape == (1, T, 64)

    # additive causal mask (eager float, 0 visible / -inf hidden)
    min_dtype = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
    mask = mask.masked_fill(torch.tril(torch.ones(T, T, dtype=torch.bool)) == 0, min_dtype)

selected_mask = indexer(hidden, (cos, sin), mask, past_key_values=None)

# The real class returns only the additive mask (a SET per query). For an
# index-level golden (with torch.topk order) we rebuild the selection with
# the same loop the reference uses internally and ASSERT the set matches the
# class output per query — the rebuild is thereby pinned to the real path.
budget_width = text_cfg.indexer_budget + text_cfg.indexer_compress_ratio - 1  # 2051
# mask shape: (batch, 1, T, kv) additive float (0 = selected, finfo.min = not);
# NOTE: values are finfo.min, NOT -inf — compare against 0
vis = selected_mask[0, 0] == 0   # [T, T] bool
qk = indexer.index_qk_proj(hidden)
q, token_k = torch.split(qk, [4 * 128, 1 * 128], dim=-1)
q = indexer.q_layernorm(q.reshape(1, T, 4, 128))
q = torch.cat([  # manual apply_rotary_pos_emb, first 64 dims
    q[..., :64] * cos[:, :, None, :] + torch.cat((-q[..., 32:64], q[..., :32]), dim=-1) * sin[:, :, None, :],
    q[..., 64:],
], dim=-1)[0]  # [T, 4, 128]
raw_keys = token_k[0]                                             # [T, 128]

sel_rows = []
scores_rows = []
with torch.no_grad():
    for t in range(T):
        n = t + 1
        ncb = n // indexer.compress_ratio
        tail = torch.arange(ncb * indexer.compress_ratio, n, dtype=torch.int64)
        if ncb > 0:
            block_tokens = torch.arange(ncb * indexer.compress_ratio, dtype=torch.int64).view(ncb, indexer.compress_ratio)
            pooled = indexer.k_layernorm(raw_keys[block_tokens].float().mean(dim=1))
            starts = block_tokens[:, 0]
            pooled = torch.cat([  # rotary at block-start positions
                pooled[..., :64] * cos[0, starts, :] + torch.cat((-pooled[..., 32:64], pooled[..., :32]), dim=-1) * sin[0, starts, :],
                pooled[..., 64:],
            ], dim=-1)                                                # [ncb, 128]
            sc = torch.relu(torch.einsum("hd,bd->bh", q[t].float(), pooled.float())).sum(dim=-1) / math.sqrt(128)
            k = min(indexer.block_topk, ncb)
            chosen = torch.topk(sc, k).indices                        # value-desc order
            chosen_tokens = block_tokens.index_select(0, chosen).flatten()
            scores_rows.append(
                torch.nn.functional.pad(sc, (0, T // indexer.compress_ratio - ncb), value=float("nan"))
            )
        else:
            chosen_tokens = torch.tensor([], dtype=torch.int64)
            scores_rows.append(torch.full((T // indexer.compress_ratio,), float("nan")))
        selected = torch.cat([chosen_tokens, tail]).to(torch.int32)
        # pin the rebuild to the real class output (same set per query)
        ref_set = torch.nonzero(vis[t], as_tuple=False).flatten()
        assert torch.equal(torch.sort(selected)[0], ref_set.to(torch.int32)), f"rebuild diverges from class mask at t={t}"
        row = torch.full((budget_width,), -1, dtype=torch.int32)
        row[: selected.numel()] = selected
        sel_rows.append(row)
        if t % 512 == 0:
            print(f"  query {t}/{T}", flush=True)

sel = torch.stack(sel_rows)                                        # [T, 2051]
scores_t = torch.stack(scores_rows)                                # [T, 640] (nan = block not visible)
print(f"selection width per query: min={int((sel != -1).sum(-1).min())}, "
      f"max={int((sel != -1).sum(-1).max())}, budget={budget_width}")

hidden.numpy().tofile(os.path.join(DBG, "qsa-hidden.f32"))
sel.numpy().tofile(os.path.join(DBG, "qsa-selected.i32"))
scores_t.numpy().tofile(os.path.join(DBG, "qsa-scores.f32"))

manifest = {
    "what": "oracle golden: QSA indexer layer 3, SPARSE regime (T=2560, 640 blocks, topk 512), real weights, deterministic input",
    "transformers": __import__("transformers").__version__,
    "torch": torch.__version__,
    "seed": SEED,
    "T": T,
    "indexer": {"n_heads": 4, "kv_heads": 1, "head_dim": 128, "budget": 2048,
                 "compress_ratio": 4, "block_topk": 512},
    "weights": f"{prefix}* ({len(state)} tensors)",
    "note": "selected recovered from the additive mask (1:1 with selected_token_indices); "
            "scores re-computed with shared pooled keys (query-independent by construction)",
}
with open(os.path.join(DBG, "qsa-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote qsa-hidden.f32 / qsa-selected.i32 / qsa-scores.f32 / qsa-manifest.json")
