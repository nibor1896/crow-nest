"""ref_image_prompt_logits.py — #VIT: the end-to-end image-prompt oracle.

Re-runs the FULL text model forward on CPU (transformers 5.16.1, eager, f32,
container-dequantized weights — the streaming ref_engine_logits pattern) over
the EXACT sequence the engine processed for an image request
(vit-gen-sequence.json), with:
  - inputs_embeds = embed gather SPLICED with the oracle's own visual
    embeddings (Qwen4ExpVisionModel f32 over the engine's patch dumps) at the
    visual rows — the inputs_embeds[mask] = image_features path;
  - position_ids = the interleaved 3-axis mrope (Qwen4ExpModel.get_rope_index
    port) so the rotary matches the oracle for image conversations;
  - the PLE layer at index 1 (PleRef row-gather, from ref_engine_logits).

Compares per-prompt-position logits against the engine's gpu-logits.f32:
max_abs delta + argmax match per row (the functional gate), and the oracle's
own greedy continuation vs the engine's generated ids.

Usage (repo root, oracle venv):
  .venv-oracle/Scripts/python.exe oracle/ref_image_prompt_logits.py decode_out/vit-dump
"""
import itertools
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from cnq_weights import CnqReader  # noqa: E402

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
    Qwen4ExpVisionConfig,
    Qwen4ExpVisionModel,
)
from transformers.vision_utils import (
    get_vision_interpolation_indices_and_weights,
    get_vision_position_ids,
)

ROOT = os.path.join(HERE, "..")
CONTAINER = os.path.join(ROOT, "converter", "Qwen3.8-Flash-Next-CNQ4.5-M.cnq")
CFGDIR = os.path.join(ROOT, "models", "Qwen3.8-Flash-Next-original")
DUMP = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "decode_out", "vit-dump")

cnq = CnqReader(CONTAINER)
cfg = json.load(open(os.path.join(CFGDIR, "config.json"), encoding="utf-8"))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"
# the batched experts path (same per-token matmuls as the eager loop, the
# top-k accumulation order aside — f32, inside the measured band); the eager
# 512-expert python loop costs about 3 min per layer on this machine
text_cfg._experts_implementation = "batched_mm"


def fetch(full, *shape):
    t = cnq.tensor_f32(full)
    return t if not shape else t.reshape(shape)


def load_into(module, prefix, skip=()):
    state = {}
    for name in module.state_dict().keys():
        if any(s in name for s in skip):
            continue
        full = prefix + name
        assert cnq.has(full), f"missing in container: {full}"
        t = fetch(full)
        state[name] = t.reshape(module.state_dict()[name].shape)
    module.load_state_dict(state, strict=False)
    return module.float().eval()


# ---- the oracle vision tower over the engine's patch dumps ----
def oracle_visual_embeds(grids):
    vcfg = Qwen4ExpVisionConfig.from_dict(cfg["vision_config"])
    vcfg._attn_implementation = "eager"
    vm = Qwen4ExpVisionModel(vcfg).float().eval()
    sd = {}
    for name, p in vm.state_dict().items():
        t = fetch(f"model.visual.{name}")
        sd[name] = t.reshape(vm.state_dict()[name].shape)
    vm.load_state_dict(sd, strict=True)
    outs = []
    for i in range(len(grids)):
        meta = json.load(open(os.path.join(DUMP, f"img{i}.meta.json"), encoding="utf-8"))
        n = meta["n_patches"]
        patches = torch.from_numpy(
            np.fromfile(os.path.join(DUMP, f"img{i}.patches.f32"), dtype=np.float32).reshape(n, -1))
        grid = torch.tensor([meta["grid"]], dtype=torch.long)
        with torch.no_grad():
            out = vm(patches, grid_thw=grid, return_dict=True)
        outs.append(out.pooler_output)
    return torch.cat(outs, dim=0)


# ---- PLE reference (p15 math, row-gather; rows dequantized from the container) ----
PLE_LAYER = 1
P = f"model.language_model.layers.{PLE_LAYER}.ple."
multipliers = fetch(P + "ple_embedding.layer_multipliers").to(torch.int64)
vocab_sizes = fetch(P + "ple_embedding.ngram_heads_vocab_sizes").to(torch.int64)
offsets = fetch(P + "ple_embedding.ngram_heads_offsets").to(torch.int64)
EOS = 248044
NGRAM = 3
H_PER = 8
NHEADS = 16
EMB_DIM = 160
SHARD = f"{P}ple_embedding.ngram_embedding"


def shard_row(uid: int) -> torch.Tensor:
    # the engine's row fill (gen.rs: read_range at row * 108, 108 bytes, NO
    # bounds check) is the byte path of record — mirrored here exactly,
    # including reading past the declared tensor len for the rows where the
    # 108 B stride exceeds it (gen.rs row*108 vs the 90 B flat packing).
    shard = uid // 2_500_012
    row = uid % 2_500_012
    full = f"{SHARD}.shard_{shard}.weight"
    t = cnq.tensors[full]
    with open(cnq.path, "rb") as f:
        f.seek(cnq.blob_off + t["offset"] + row * 108)
        raw_row = f.read(108)
    out = np.empty(192, dtype=np.float32)
    for bi in range(3):
        out[bi * 64:(bi + 1) * 64] = CnqReader._deq_block(
            raw_row[bi * 36:(bi + 1) * 36], t["global_scale"])
    return torch.from_numpy(out[:160].copy())


def ple_ids_for(ids_in):
    history = [EOS] * (NGRAM - 1) + list(ids_in)
    n = len(history)
    shifted = []
    for shift in range(NGRAM):
        eos_pos = [-1] * n
        for i, v in enumerate(history):
            if v == EOS:
                eos_pos[i] = i
        prev_incl = []
        run = -(2**63)
        for i in range(n):
            run = max(run, eos_pos[i])
            prev_incl.append(run)
        row = []
        for i in range(n):
            prev = -1 if i == 0 else prev_incl[i - 1]
            seg_start = prev + 1
            pos_in_seg = i - seg_start
            src = i - shift
            valid = pos_in_seg >= shift and src >= 0
            row.append(history[max(src, 0)] if valid else EOS)
        shifted.append(row)
    out = []
    for i in range(NGRAM - 1, n):
        r = []
        for ngram in range(2, NGRAM + 1):
            start = (ngram - 2) * H_PER
            mixed = (shifted[0][i] & 0xFFFFFFFFFFFFFFFF)
            mixed = (mixed * (multipliers[0].item() & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
            for p in range(1, ngram):
                mixed ^= (shifted[p][i] * (multipliers[p].item() & 0xFFFFFFFFFFFFFFFF)) & 0xFFFFFFFFFFFFFFFF
            m = mixed if mixed < 2**63 else mixed - 2**64
            for h in range(H_PER):
                vs = vocab_sizes[start + h].item()
                off = offsets[start + h].item()
                r.append(m % vs + off)
        out.append(r)
    return out


class PleRef(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.key_proj = fetch(P + "key_proj.weight", 10240, 2560).float()
        self.value_proj = fetch(P + "value_proj.weight", 2560, 2560).float()
        self.norm_key = fetch(P + "norm_key.weight", 10240).float()
        self.norm_query = fetch(P + "norm_query.weight", 10240).float()
        self.norm_conv = fetch(P + "norm_conv.weight", 10240).float()
        self.conv1d = fetch(P + "conv1d.weight", 4, 2560, 4).float()

    def rms(self, x, w):
        return torch.nn.functional.rms_norm(x, (x.shape[-1],), (1 + w), eps=1e-6)

    def forward(self, hidden, ids_in):
        T = hidden.shape[1]
        ng = ple_ids_for(ids_in)
        flat = [x for row in ng for x in row]
        uniq = sorted(set(flat))
        slot = {v: i for i, v in enumerate(uniq)}
        rows = torch.stack([shard_row(u) for u in uniq])  # [U,160]
        idx = torch.tensor([[slot[v] for v in row] for row in ng])
        emb = rows[idx.view(-1)].view(T, NHEADS, EMB_DIM).reshape(T, NHEADS * EMB_DIM)
        key = self.rms(emb @ self.key_proj.t(), self.norm_key)
        val = emb @ self.value_proj.t()
        q = self.rms(hidden[0], self.norm_query)
        g = (key.view(T, 4, 2560) * q.view(T, 4, 2560)).sum(-1) / 2560 ** 0.5
        gs = g.sign() * g.abs().clamp(min=1e-6).sqrt()
        sg = torch.sigmoid(gs)
        gated = (sg.view(T, 4, 1) * val.view(T, 1, 2560)).reshape(T, 10240)
        gn = self.rms(gated, self.norm_conv)
        gnv = gn.view(T, 4, 2560)
        w = self.conv1d.view(4, 2560, 4)
        gnp = torch.zeros(T + 9, 4, 2560)
        gnp[9:] = gnv
        acc = torch.zeros(T, 4, 2560)
        for k in range(4):
            acc += w[..., k].unsqueeze(0) * gnp[k * 3:k * 3 + T]
        acc = acc.reshape(T, 10240)
        out = gated + torch.nn.functional.silu(acc)
        return out.unsqueeze(0)


# ---- the interleaved mrope (Qwen4ExpModel.get_rope_index port) ----
def mrope_positions(types, grids):
    it = iter(grids)
    out = []
    cur = 0
    i = 0
    while i < len(types):
        ty = types[i]
        start = i
        while i < len(types) and types[i] == ty:
            i += 1
        ln = i - start
        if ty == 0:
            out.extend([[cur + k] * 3 for k in range(ln)])
            cur += ln
        else:
            _, hp, wp = next(it)
            gw = wp // 2
            for k in range(ln):
                bh = k // (gw * 4)
                rem = k % (gw * 4)
                bj = rem // 4
                r2 = rem % 4
                out.append([cur, cur + bh * 2 + r2 // 2, cur + bj * 2 + r2 % 2])
            cur += max(hp, wp) // 2
    mx = max(max(r) for r in out)
    return out, mx + 1 - len(types)


seq = json.load(open(os.path.join(DUMP, "vit-gen-sequence.json"), encoding="utf-8"))
ids = seq["ids"]
T = len(ids)
types = [1 if m >= 0 else 0 for m in seq["visual_map"]]
grids = [tuple(g) for g in seq["grids"]]
print(f"engine ref: sequence T={T}, {sum(types)} visual rows, grids {grids}")
pos, delta = mrope_positions(types, grids)
print(f"mrope delta oracle {delta} (engine {seq.get('delta', 'n/a')})")

print("oracle vision tower over the engine patches …")
vis = oracle_visual_embeds(grids)
assert vis.shape[0] == sum(types), f"visual rows {vis.shape[0]} != {sum(types)}"

embed = fetch("model.language_model.embed_tokens.weight", 248320, 2560)
lm_head_w = fetch("lm_head.weight", 248320, 2560)
mixer = load_into(Qwen4ExpTextGatedResidual(text_cfg, use_combine=False),
                  "model.language_model.hyper_connection_mixer.")
rotary = Qwen4ExpTextRotaryEmbedding(config=text_cfg).float().eval()
ple = PleRef().eval()

with torch.no_grad():
    emb = embed[torch.tensor(ids, dtype=torch.long)]
    emb = emb.clone()
    for r, m in enumerate(seq["visual_map"]):
        if m >= 0:
            emb[r] = vis[m]
    position_ids = torch.tensor(pos, dtype=torch.long).T.unsqueeze(1)  # (3, 1, T)
    cos, sin = rotary(emb, position_ids)
    min_dtype = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
    mask = mask.masked_fill(torch.tril(torch.ones(T, T, dtype=torch.bool)) == 0, min_dtype)
    h = emb.repeat(1, 1, text_cfg.hc_count)

    import copy
    tiny_cfg = copy.deepcopy(text_cfg)
    tiny_cfg.ngram_vocab_size_base = 64
    for layer_idx in range(text_cfg.num_hidden_layers):
        layer = Qwen4ExpTextDecoderLayer(tiny_cfg if layer_idx == PLE_LAYER else text_cfg, layer_idx)
        layer = load_into(layer, f"model.language_model.layers.{layer_idx}.", skip=("ple",))
        if layer.ple is not None:
            if os.environ.get("PLE_OFF") != "1":
                h = h + ple(h, ids)
            layer.ple = None
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
        if layer_idx % 8 == 0:
            print(f"  layer {layer_idx:2d}", flush=True)

    h = mixer(h)
    logits = torch.nn.functional.linear(h, lm_head_w)[0]

ref = logits.contiguous().numpy()
ref.tofile(os.path.join(DUMP, "ref-logits.f32"))

gpu = np.fromfile(os.path.join(DUMP, "gpu-logits.f32"), dtype=np.float32).reshape(T, -1)
print(f"ENGINE parity gate: [T={T}][V={gpu.shape[1]}] - production FP4 vs f32 reference")
worst = 0.0
am_match = 0
margins = []
first_dev = None
for t in range(T):
    d = np.abs(gpu[t] - ref[t])
    worst = max(worst, d.max())
    g_am = int(np.argmax(gpu[t]))
    r_am = int(np.argmax(ref[t]))
    srt = np.sort(ref[t])
    margins.append(srt[-1] - srt[-2])
    if g_am != r_am and first_dev is None:
        first_dev = t
    am_match += g_am == r_am
print(f"worst logit delta {worst:.3e} | argmax {am_match}/{T} | min ref margin {min(margins):.4f} | first argmax dev at row {first_dev}")
gen = seq.get("generated") or []
# the oracle computed prompt rows only; the comparable greedy ids are the
# engine's generated ids against the oracle's argmax at the rows that
# produced them (row T-1+k produced generated[k] during decode)
k = min(len(gen), T)
og = [int(np.argmax(ref[T - 1 + i])) if T - 1 + i < T else None for i in range(len(gen))]
first = int(np.argmax(ref[T - 1]))
print("engine generated:", gen[:16])
print("oracle argmax at the last prompt row:", first, "(engine's first generated:", gen[0] if gen else None, ")")
print("first-token greedy match:", first == (gen[0] if gen else None))
print("nan gpu:", int(np.isnan(gpu).sum()))
verdict = "PASS" if am_match == T and not np.isnan(gpu).any() else "CHECK"
print(f"engine image-prompt parity: {verdict}")
