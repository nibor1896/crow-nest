# Export a PLE golden for the GPU port (crow-nest p15 / Crow #10).
#
# What: the PLELayer of layer index 1 (config ple_layer_ids [2], 1-based),
# real weights, f32, input = the p13 generation-loop token sequence (12
# tokens, no cache — previous n-gram context is EOS-filled, ngram_size 3).
#
# The n-gram index math (shift-right-ignore-EOS, multiplier XOR, prime-mod)
# is recomputed here using the CHECKPOINT's stored I64 tables
# (layer_multipliers, ngram_heads_vocab_sizes, ngram_heads_offsets) — nothing
# about the hashing is re-derived. Embedding rows are read LAZILY from the
# original safetensors shards (only the unique ids the sequence needs — the
# full table is 102 GB BF16 and never fits RAM).
#
# Dumps (probes/p15debug/, all f32 unless noted):
#   ple-token-ids.i32   [12]            input token ids
#   ple-ngram-ids.i64   [12, 16]        embedding row ids per token (gate object
#                                       for the Rust host-side index math)
#   ple-embeddings.f32  [12, 2560]      gathered n-gram embedding
#   ple-output.f32      [12, 10240]     PLE layer output (adds onto the HC stream)
#   ple-weights.npz     key_proj, value_proj, norm_key, norm_query, norm_conv,
#                       conv1d          f32 weights (all small)

import json
import math
import os

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-original")
DBG = os.path.join(HERE, "..", "probes", "p15debug")
SEQ = json.load(open(os.path.join(HERE, "..", "probes", "p13debug", "gen-sequence.json")))
ids = SEQ["all_ids"][: SEQ["prompt_len"] + SEQ["generated"]]  # 12 processed tokens
T = len(ids)
EOS = 248044
NGRAM = 3            # ngram_size
CONTEXT = NGRAM - 1  # 2
HEADS_PER_NGRAM = 8
NHEADS = (NGRAM - 1) * HEADS_PER_NGRAM  # 16
EMB_DIM = 160

os.makedirs(DBG, exist_ok=True)
index_json = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index_json["weight_map"]
P = "model.language_model.layers.1.ple."


def fetch(name):
    with safe_open(os.path.join(MODELS, wm[P + name]), framework="pt", device="cpu") as f:
        return f.get_tensor(P + name)


def fetch_f32(name):
    return fetch(name).to(torch.float32)

multipliers = fetch("ple_embedding.layer_multipliers").long()            # [3]
vocab_sizes = fetch("ple_embedding.ngram_heads_vocab_sizes").long()      # [16]
offsets = fetch("ple_embedding.ngram_heads_offsets").long()              # [16]
print(f"layer_multipliers: {multipliers.tolist()}")
print(f"head_vocab_sizes[:4]: {vocab_sizes[:4].tolist()} total={int(vocab_sizes.sum())}")
print(f"head_offsets[:4]: {offsets[:4].tolist()}")

key_proj = fetch_f32("key_proj.weight")             # [10240, 2560]
value_proj = fetch_f32("value_proj.weight")         # [2560, 2560]
norm_key = fetch_f32("norm_key.weight")             # [10240]
norm_query = fetch_f32("norm_query.weight")
norm_conv = fetch_f32("norm_conv.weight")
conv1d = fetch_f32("conv1d.weight")                 # [10240, 1, 4]


def rms_norm(x, w, group_size=None, eps=1e-6):
    if group_size is not None:
        xg = x.reshape(*x.shape[:-1], -1, group_size)
        out = xg * torch.rsqrt(xg.pow(2).mean(-1, keepdim=True) + eps)
        return (out * (1.0 + w.reshape(1, 1, -1, group_size))).flatten(-2)
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * (1.0 + w)


def shift_right_ignore_eos(token_ids, shift):
    if shift == 0:
        return token_ids
    positions = torch.arange(len(token_ids))
    eos_positions = torch.where(token_ids == EOS, positions, torch.tensor(-1, dtype=torch.long))
    prev_eos_inclusive = torch.cummax(eos_positions, dim=0).values
    previous_eos = torch.cat([torch.tensor([-1]), prev_eos_inclusive[:-1]])
    segment_start = previous_eos + 1
    position_in_segment = positions - segment_start
    source = positions - shift
    gather = source.clamp_min(0)
    shifted = token_ids[gather]
    valid = (position_in_segment >= shift) & (source >= 0)
    return torch.where(valid, shifted, torch.tensor(EOS, dtype=torch.long))


token_ids = torch.tensor([EOS] * CONTEXT + ids, dtype=torch.long)  # history + input
shifted = [shift_right_ignore_eos(token_ids, s) for s in range(NGRAM)]

blocks = []
for ngram in range(2, NGRAM + 1):
    start = (ngram - 2) * HEADS_PER_NGRAM
    mixed = shifted[0].long() * multipliers[0]
    for p in range(1, ngram):
        mixed = torch.bitwise_xor(mixed, shifted[p].long() * multipliers[p])
    vs = vocab_sizes[start:start + HEADS_PER_NGRAM]
    offs = offsets[start:start + HEADS_PER_NGRAM]
    ngram_ids = torch.remainder(mixed.unsqueeze(-1), vs.view(1, -1))
    blocks.append(ngram_ids + offs.view(1, -1))
ngram_ids = torch.cat(blocks, dim=-1)[-T:]           # [T, 16]
assert int(ngram_ids.min()) >= 0

# ---- lazy row gather from the original shards ----
flat = ngram_ids.reshape(-1)
unique, inverse = torch.unique(flat, return_inverse=True)
print(f"ngram lookups: {flat.numel()} total, {unique.numel()} unique rows")

# shard layout: each shard holds [2500012, 160] = 2,500,012 consecutive rows
# (128 shards × 2,500,012 = 320,001,536 = padded total vocab)
ROWS_PER_SHARD = 2_500_012


def read_shard_rows_raw(shard_idx, rows):
    """Read specific rows of a PLE shard with raw file access (no torch mmap
    handles — many concurrent mapped 2.7 GB files die silently on Windows).
    Returns {row: np.float32[160]}."""
    shard_name = f"ple_embedding.ngram_embedding.shard_{shard_idx}.weight"
    rel = wm[P + shard_name]
    path = os.path.join(MODELS, rel)
    with open(path, "rb") as fh:
        n8 = fh.read(8)
        hl = int.from_bytes(n8, "little")
        hdr = json.loads(fh.read(hl))
        base = 8 + hl
        info = hdr[P + shard_name]
        off = info["data_offsets"][0]
        out = {}
        for r in rows:
            fh.seek(base + off + r * EMB_DIM * 2)
            raw = fh.read(EMB_DIM * 2)
            u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
            f32 = (u16 << 16).view(np.float32)  # exact BF16 -> f32
            out[r] = torch.from_numpy(f32.copy())
        return out


rows_by_shard = {}
for i, uid in enumerate(unique.tolist()):
    s, r = divmod(uid, ROWS_PER_SHARD)
    rows_by_shard.setdefault(s, []).append((i, r))
print(f"rows spread over {len(rows_by_shard)} shards", flush=True)

embeddings = torch.zeros(T, NHEADS * EMB_DIM)
for s_done, (s, items) in enumerate(sorted(rows_by_shard.items())):
    vecs = read_shard_rows_raw(s, [r for (_, r) in items])
    for i, r in items:
        emb = vecs[r]
        sel = (inverse == i).nonzero().flatten()
        tok = sel // NHEADS
        head = sel % NHEADS
        embeddings[tok, head * EMB_DIM:(head + 1) * EMB_DIM] = emb
    print(f"  shard {s} done ({s_done + 1}/{len(rows_by_shard)})", flush=True)

# ---- PLE forward (f32) ----
hc, H = 4, 2560
hidden = torch.randn(1, T, hc * H).uniform_(-1, 1)   # stand-in HC stream (deterministic enough; seeded below)
g = torch.Generator().manual_seed(20260902)
hidden = torch.empty(1, T, hc * H).uniform_(-1, 1, generator=g)

with torch.no_grad():
    key_normed = rms_norm(embeddings @ key_proj.T, norm_key, group_size=H).reshape(1, T, hc, H)
    value = embeddings @ value_proj.T                                # [T, 2560]
    query_normed = rms_norm(hidden, norm_query, group_size=H).reshape(1, T, hc, H)
    gate = (key_normed * query_normed).sum(dim=-1, keepdim=True) / math.sqrt(H)
    gate = gate.abs().clamp_min(1e-6).sqrt() * gate.sign()
    gated_value = torch.sigmoid(gate) * value.unsqueeze(-2)          # [1, T, 4, 2560]
    gated_flat = gated_value.flatten(-2)                             # [1, T, 10240]
    gated_normed = rms_norm(gated_flat, norm_conv, group_size=H)

    # dilated depthwise conv (kernel 4, dilation 3) + silu — no cache: left pad
    # short_conv_state_len = (4-1)*3 = 9 zeros, then plain conv1d
    pad = 9
    x = gated_normed.transpose(1, 2)                                 # [1, 10240, T]
    xp = torch.nn.functional.pad(x, (pad, 0))
    w = conv1d.reshape(10240, 4)
    out_conv = torch.zeros(1, 10240, T)
    for k in range(4):
        out_conv += w[:, k].view(1, -1, 1) * xp[:, :, k * 3:k * 3 + T]
    out_conv = torch.nn.functional.silu(out_conv).transpose(1, 2)

    output = gated_flat + out_conv                                   # [1, T, 10240]

for tag, t_ in [("key-normed", key_normed.reshape(1, T, -1)), ("value", value.reshape(1, T, -1)),
                ("query-normed", query_normed.reshape(1, T, -1)), ("gate-signed", gate.reshape(1, T, 4)),
                ("gated", gated_flat), ("gated-normed", gated_normed), ("conv-out", out_conv)]:
    t_.detach().numpy().tofile(os.path.join(DBG, f"ref-{tag}.f32"))

embeddings.numpy().tofile(os.path.join(DBG, "ple-embeddings.f32"))
output.numpy().tofile(os.path.join(DBG, "ple-output.f32"))
np.asarray(ngram_ids.numpy().reshape(-1), dtype=np.int64).tofile(os.path.join(DBG, "ple-ngram-ids.i64"))
np.asarray(ids, dtype=np.int32).tofile(os.path.join(DBG, "ple-token-ids.i32"))
hidden.numpy().tofile(os.path.join(DBG, "ple-hidden.f32"))
np.savez(os.path.join(DBG, "ple-weights.npz"),
         key_proj=key_proj.numpy(), value_proj=value_proj.numpy(),
         norm_key=norm_key.numpy(), norm_query=norm_query.numpy(),
         norm_conv=norm_conv.numpy(), conv1d=conv1d.numpy(),
         multipliers=multipliers.numpy(), vocab_sizes=vocab_sizes.numpy(),
         offsets=offsets.numpy())

manifest = {
    "what": "oracle golden: PLE layer (layers.1), real weights, f32, input = p13 12-token sequence, no cache (EOS-filled n-gram context)",
    "ngram": {"ngram_size": NGRAM, "heads_per_ngram": HEADS_PER_NGRAM, "nheads": NHEADS,
               "emb_dim_per_head": EMB_DIM, "eos": EOS},
    "unique_rows": int(unique.numel()),
    "note": "index math from the checkpoint's stored I64 tables; embedding rows lazily read from the original shards; hidden stream is seeded randn (stand-in, PLE input side)",
}
with open(os.path.join(DBG, "ple-manifest.json"), "w") as f:
    json.dump(manifest, f, indent=1)
print("wrote ple-embeddings/output/ngram-ids/token-ids/hidden/weights")
print(f"output stats: absmax={output.abs().max():.4f}")
