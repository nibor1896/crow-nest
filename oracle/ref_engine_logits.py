# Reference side of the ENGINE end-to-end parity (#11).
#
# Re-runs the FULL 48-layer text model forward on CPU (transformers 5.16.1,
# eager, f32) over the exact token sequence the engine processed
# (decode_out/gen-sequence.json), INCLUDING the PLE layer at index 1
# (manual row-gather math per p15 — the 102 GB table never loads), and
# compares per-position logits against decode_out/gpu-logits.f32.
#
# The engine runs PRODUCTION precision (CNQ4.5 FP4 weights, NVFP4 PLE rows,
# FP8-KV); the reference is f32 originals — the comparison is therefore a
# MEASUREMENT (documented deltas), with the greedy argmax match as the
# functional gate (p13 discipline: ref top-2 margins vs f32-noise deltas).

import json
import os

import numpy as np
import torch

from transformers.models.qwen4_exp.configuration_qwen4_exp import Qwen4ExpTextConfig
from transformers.models.qwen4_exp.modeling_qwen4_exp import (
    Qwen4ExpTextDecoderLayer,
    Qwen4ExpTextGatedResidual,
    Qwen4ExpTextRotaryEmbedding,
)

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-original")
# CROW_PARITY_DIR=<dir> (#11, 2026-09-05): read gen-sequence.json / gpu-logits.f32 from
# and write ref-logits.f32 to that directory instead of decode_out
OUT = os.environ.get("CROW_PARITY_DIR") or os.path.join(HERE, "..", "decode_out")

seq = json.load(open(os.path.join(OUT, "gen-sequence.json")))
rows = seq["rows"]
prompt_len = seq["prompt_len"]
# all_ids carries prompt + generated + the FINAL greedy candidate (one extra);
# exactly `rows` tokens were processed to produce the `rows` logit rows
ids = seq["all_ids"][:rows]
T = len(ids)
assert rows == T, f"logit rows {rows} != processed tokens {T}"

cfg = json.load(open(os.path.join(MODELS, "config.json")))
text_cfg = Qwen4ExpTextConfig.from_dict(cfg["text_config"])
text_cfg._attn_implementation = "eager"

index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]


def fetch(full):
    import safetensors
    from safetensors import safe_open
    path = os.path.join(MODELS, wm[full])
    with safe_open(path, framework="pt", device="cpu") as f:
        return f.get_tensor(full).to(torch.float32)


def load_into(module, prefix, skip=()):
    state = {}
    for name in module.state_dict().keys():
        if any(s in name for s in skip):
            continue
        full = prefix + name
        assert full in wm, f"missing in checkpoint: {full}"
        state[name] = fetch(full)
    module.load_state_dict(state, strict=False)
    return module.float().eval()


# ---- PLE reference (p15 math, row-gather) ----
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


def shard_row(uid: int) -> torch.Tensor:
    shard = uid // 2_500_012
    row = uid % 2_500_012
    full = f"{P}ple_embedding.ngram_embedding.shard_{shard}.weight"
    from safetensors import safe_open
    path = os.path.join(MODELS, wm[full])
    with safe_open(path, framework="pt", device="cpu") as f:
        sl = f.get_slice(full)
        return sl[row:row + 1, :].to(torch.float32)[0]


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
        self.key_proj = fetch(P + "key_proj.weight").float()
        self.value_proj = fetch(P + "value_proj.weight").float()
        self.norm_key = fetch(P + "norm_key.weight").float()
        self.norm_query = fetch(P + "norm_query.weight").float()
        self.norm_conv = fetch(P + "norm_conv.weight").float()
        self.conv1d = fetch(P + "conv1d.weight").float()

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
        key = self.rms(emb @ self.key_proj.t(), self.norm_key)         # [T,10240]
        val = emb @ self.value_proj.t()                                 # [T,2560]
        q = self.rms(hidden[0], self.norm_query)
        g = (key.view(T, 4, 2560) * q.view(T, 4, 2560)).sum(-1) / 2560 ** 0.5
        gs = g.sign() * g.abs().clamp(min=1e-6).sqrt()
        sg = torch.sigmoid(gs)                                            # [T,4]
        gated = (sg.view(T, 4, 1) * val.view(T, 1, 2560)).reshape(T, 10240)  # value row broadcast to all 4 streams (p15)
        gn = self.rms(gated, self.norm_conv)
        # dilated depthwise conv k=4 dil=3, left state 9 zeros + silu + add
        gnv = gn.view(T, 4, 2560)
        w = self.conv1d.view(4, 2560, 4)                                  # [4,2560,4]
        gnp = torch.zeros(T + 9, 4, 2560)
        gnp[9:] = gnv
        acc = torch.zeros(T, 4, 2560)
        for k in range(4):
            acc += w[..., k].unsqueeze(0) * gnp[k * 3:k * 3 + T]
        acc = acc.reshape(T, 10240)
        out = gated + torch.nn.functional.silu(acc)
        return out.unsqueeze(0)


print(f"engine ref: sequence {ids} (T={T})")
embed = fetch("model.language_model.embed_tokens.weight")
lm_head_w = fetch("lm_head.weight")
mixer = load_into(Qwen4ExpTextGatedResidual(text_cfg, use_combine=False),
                  "model.language_model.hyper_connection_mixer.")
rotary = Qwen4ExpTextRotaryEmbedding(config=text_cfg).float().eval()
ple = PleRef().eval()

with torch.no_grad():
    input_ids = torch.tensor([ids], dtype=torch.long)
    emb = embed[input_ids]
    position_ids = torch.arange(T).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rotary(emb, position_ids)
    min_dtype = torch.finfo(torch.float32).min
    mask = torch.zeros(1, 1, T, T, dtype=torch.float32)
    mask = mask.masked_fill(torch.tril(torch.ones(T, T, dtype=torch.bool)) == 0, min_dtype)
    h = emb.repeat(1, 1, text_cfg.hc_count)

    # the PLE layer's constructor would allocate the full n-gram table in f32
    # (102 GB bf16 -> 207 GB commit, measured 2026-09-05: page-file thrash for
    # minutes) only to be dropped below — build it with a tiny n-gram vocab;
    # PleRef reads the real sizes/offsets/rows from the checkpoint itself
    import copy
    tiny_cfg = copy.deepcopy(text_cfg)
    tiny_cfg.ngram_vocab_size_base = 64
    for layer_idx in range(text_cfg.num_hidden_layers):
        layer = Qwen4ExpTextDecoderLayer(tiny_cfg if layer_idx == PLE_LAYER else text_cfg, layer_idx)
        layer = load_into(layer, f"model.language_model.layers.{layer_idx}.", skip=("ple",))
        if layer.ple is not None:
            import os as _os
            if _os.environ.get("PLE_OFF") != "1":
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
        print(f"  layer {layer_idx:2d} done", flush=True)

    h = mixer(h)
    logits = torch.nn.functional.linear(h, lm_head_w)[0]

ref = logits.contiguous().numpy()
ref.tofile(os.path.join(OUT, "ref-logits.f32"))
# #91: a reference-only run (tools/teacher-forced-oracle-seq.py prep) has no
# engine dump beside it - the reference rows are the product, the gate below needs both
if not os.path.exists(os.path.join(OUT, "gpu-logits.f32")):
    print(f"reference only: {T} rows -> {os.path.join(OUT, 'ref-logits.f32')} (no gpu-logits.f32 to gate against)")
    raise SystemExit(0)

gpu = np.fromfile(os.path.join(OUT, "gpu-logits.f32"), dtype=np.float32).reshape(T, -1)
print(f"ENGINE parity gate: [T={T}][V={gpu.shape[1]}] - production FP4/FP8 vs f32 reference")
worst = 0.0
am_match = 0
margins = []
for t in range(T):
    d = np.abs(gpu[t] - ref[t])
    worst = max(worst, d.max())
    g_am = int(np.argmax(gpu[t]))
    r_am = int(np.argmax(ref[t]))
    srt = np.sort(ref[t])
    margins.append(srt[-1] - srt[-2])
    am_match += g_am == r_am
    print(f"  pos {t:2d}: max_abs={d.max():.3e}  argmax gpu={g_am} ref={r_am} "
          f"match={g_am == r_am}  ref_top2_margin={srt[-1] - srt[-2]:.4f}")
print(f"worst logit delta {worst:.3e} | argmax {am_match}/{T} | min ref margin {min(margins):.4f}")
print("nan gpu:", int(np.isnan(gpu).sum()))
if am_match == T and not np.isnan(gpu).any():
    print("engine parity: PASS (greedy trace matches the f32 reference; deltas are the production-quant measurement)")
else:
    print("engine parity: ARGMAX MISMATCH - investigate before ten-task")
    raise SystemExit(1)
