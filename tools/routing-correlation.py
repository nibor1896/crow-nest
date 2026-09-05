#!/usr/bin/env python3
"""routing-correlation.py — crow-nest #8 pre-study: token-to-token expert routing
correlation, measured with the reference model's own routers (oracle venv).

Streams real text through all 48 decoder layers LAYER BY LAYER (weights loaded per
layer from the original safetensors, freed before the next — the model never fits in
RAM), records each layer's routed expert ids per token, then computes:

  1. lag-ℓ overlap:  |routed(T) ∩ routed(T+ℓ)| / 10  for ℓ = 1, 2, 4, 8
  2. PREFETCH GAIN (the strategy-C number): for each token, the COLD needs
     (routed ids outside the per-layer top-N hot set) — how many were routed in one
     of the PREVIOUS k tokens at the same layer, i.e. would have been pre-streamed
     by a speculative prefetcher. Reported for N=160 and k=1..4.

Approximations (documented, acceptable for a locality pre-study): PLE contribution
skipped (layer 1's ple module set to None — full table is 102.4 GB), attention run
without an explicit causal mask. Both perturb inputs slightly; temporal STRUCTURE
is what we measure.

Usage: .venv-oracle/Scripts/python.exe tools/routing-correlation.py [T_tokens]
Writes: oracle/routing-selections.npz, prints the analysis.
"""

import json
import os
import sys
import time

import numpy as np
import torch
from safetensors import safe_open

MODELS = os.path.join(os.path.dirname(__file__), "..", "models", "Qwen3.8-Flash-Next-original")
OUT = os.path.join(os.path.dirname(__file__), "..", "oracle", "routing-selections.npz")
TOPK = 10

TEXT = """# Crow session — Messreihe #159

User: Miss bitte den Betriebspunkt: 41 tok/s decode unter llama.cpp, 62 Handoffs
je Token, RAM-Bus bei 33.5 %. Was bringt das neue Flag?

Assistant: Ich fahre die Serie verschränkt in derselben Sitzung: Baseline A,
Variante B, Baseline A, Variante B — eine Variable, Auflösung genannt. Der
Cold-Start wird verworfen. Vorhersagen kommen nicht ins Log, nur Messwerte.

User: Und wenn die Latenz steigt?

Assistant: Dann ist das ein Messergebnis. Wir schneiden die Stufe neu, nicht
das Projekt ab. Error bars before conclusions.

User: Schreibe einen Commit. Assistant: git commit -m "bench: interleaved A/B,
one variable, resolution stated". Fertig. Noch etwas?

User: Erkläre den Unterschied zwischen Residenz und Prefetch. Assistant: Residenz
heißt: die heißen Experten wohnen dauerhaft im VRAM. Prefetch heißt: die kommenden
wandern schon während der GPU rechnet. Beides senkt die Zahl der Handoffs —
das eine strukturell, das andere spekulativ.
"""


def main():
    t_tokens = int(sys.argv[1]) if len(sys.argv) > 1 else 96
    t0 = time.time()
    from transformers import AutoConfig, AutoTokenizer
    from transformers.models.qwen4_exp.modeling_qwen4_exp import (
        Qwen4ExpTextDecoderLayer, Qwen4ExpTextRotaryEmbedding,
    )

    tok = AutoTokenizer.from_pretrained(MODELS)
    ids = tok(TEXT, return_tensors="pt")["input_ids"][0][:t_tokens]
    t_tokens = len(ids)
    print(f"text: {t_tokens} tokens")

    cfg = AutoConfig.from_pretrained(MODELS)
    tc = cfg.text_config
    n_layers = tc.num_hidden_layers
    hc = tc.hc_count
    lt = tc.layer_types

    # embeddings + rotary live for the whole pass
    idx = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
    wm = idx["weight_map"]

    shard_headers = {}

    def get_tensor(ck):
        # RAW-BYTES read path: torch storage.__getitem__ segfaulted on Windows when
        # reading these safetensors via safe_open (twice, at different handle
        # lifetimes) — so we bypass it: file seek + numpy BF16->f32. The Rust
        # converter and verify_roundtrip read the same bytes fine; it's the torch
        # interop, not the files.
        shard = wm[ck]
        if shard not in shard_headers:
            with open(os.path.join(MODELS, shard), "rb") as f:
                n = int.from_bytes(f.read(8), "little")
                shard_headers[shard] = (json.loads(f.read(n)), 8 + n)
        hdr, data_start = shard_headers[shard]
        info = hdr[ck]
        dt = info["dtype"]
        assert dt == "BF16", f"unexpected dtype {dt} for {ck}"
        off, end = info["data_offsets"]
        with open(os.path.join(MODELS, shard), "rb") as f:
            f.seek(data_start + off)
            raw = f.read(end - off)
        # chunked BF16->F32: the naive u32 intermediate quadruples peak RAM and
        # OOMs on the 3.35 GB expert tensors
        total = len(raw) // 2
        f32 = np.empty(total, dtype=np.float32)
        CH = 1 << 27  # 134M elements per chunk
        for i in range(0, total, CH):
            n = min(CH, total - i)
            u16 = np.frombuffer(raw, dtype=np.uint16, count=n, offset=i * 2)
            f32[i:i + n] = (u16.astype(np.uint32) << 16).view(np.float32)
        return torch.from_numpy(f32.reshape(info["shape"]))

    embed = get_tensor("model.language_model.embed_tokens.weight").float()
    print(f"embeddings: {tuple(embed.shape)} f32, {time.time()-t0:.0f}s")

    rot = Qwen4ExpTextRotaryEmbedding(config=tc)
    pos = torch.arange(t_tokens).unsqueeze(0)
    dummy = torch.zeros(1, t_tokens, 64)
    pe = rot(dummy, pos)
    if not isinstance(pe, tuple):
        pe = (pe, pe)

    h = embed[ids].unsqueeze(0).repeat(1, 1, hc)  # model-level hc expansion
    print(f"hidden in: {tuple(h.shape)}")
    # QSA attention uses attention_mask as its visible-token index (bool) — a causal
    # prefill mask (True = visible) is the no-padding case
    causal = torch.tril(torch.ones(1, 1, t_tokens, t_tokens, dtype=torch.bool))

    selections = np.zeros((n_layers, t_tokens, TOPK), dtype=np.int32)

    for li in range(n_layers):
        t_l = time.time()
        layer = Qwen4ExpTextDecoderLayer(tc, layer_idx=li).to(torch.float32)
        ref_keys = [k for k in layer.state_dict().keys()
                    if ".ple." not in k and not k.startswith("ple")]
        # ".ple." skipped wholesale: layer.ple = None below (pre-study approximation),
        # and its I64 metadata tables have no place in a weight load
        for key in ref_keys:
            ck = f"model.language_model.layers.{li}.{key}"
            layer.state_dict()[key].copy_(get_tensor(ck).float())
        layer.ple = None  # pre-study approximation: 102.4 GB PLE table skipped

        captured = {}

        def hook(module, args, output):
            # TopKRouter.forward returns (logits, scores, indices) — capture the
            # int64/32 indices tensor with last dim == TOPK, wherever it sits
            outs = output if isinstance(output, tuple) else (output,)
            for c in outs:
                if torch.is_tensor(c) and c.dtype in (torch.int64, torch.int32) and c.shape[-1] == TOPK:
                    captured["ids"] = c.detach()
                    return

        target = None
        for name, modu in layer.named_modules():
            if "router" in name.lower() or name.endswith("mlp.gate"):
                target = modu
                break
        assert target is not None, "router module not found"
        handle = target.register_forward_hook(hook)
        with torch.no_grad():
            layer(h, position_embeddings=pe, attention_mask=causal, cache_position=pos[0])
        handle.remove()
        assert "ids" in captured, f"no router capture at layer {li}"
        s = captured["ids"]
        selections[li] = s.reshape(-1, TOPK)[-t_tokens:].numpy() if s.dim() > 2 else s.numpy()
        del layer
        import gc; gc.collect()
        print(f"layer {li+1}/{n_layers} ({lt[li][:9]:<9}) captured, {time.time()-t0:.0f}s total")

    np.savez_compressed(OUT, selections=selections, layer_types=np.array(lt))
    print(f"selections saved: {OUT}  total {time.time()-t0:.0f}s")

    # ---- analysis ----
    print("\n==== PREFETCH GAIN ANALYSIS ====")
    for N_hot in (160,):
        print(f"\nhot set N={N_hot}/layer (frequency-based, same rule as the loader)")
        for k_win in (1, 2, 3, 4):
            gains = []
            for li in range(n_layers):
                routed = [set(selections[li, t].tolist()) for t in range(t_tokens)]
                freq = {}
                for r in routed:
                    for e in r:
                        freq[e] = freq.get(e, 0) + 1
                hot = set(sorted(freq, key=freq.get, reverse=True)[:N_hot])
                need_hit = need_total = 0
                for t in range(k_win, t_tokens):
                    needs = routed[t] - hot
                    window = set().union(*routed[max(0, t - k_win):t]) if t > 0 else set()
                    need_hit += len(needs & window)
                    need_total += len(needs)
                if need_total:
                    gains.append(need_hit / need_total)
            if not gains:
                print(f"  prefetch window k={k_win}: no cold needs at this T (T too small — all routed ids fit the hot set)")
                continue
            g = np.array(gains)
            print(f"  prefetch window k={k_win}: gain {g.mean():.1%} mean | {np.median(g):.1%} median "
                  f"| min {g.min():.1%} | max {g.max():.1%}  (cold needs pre-satisfied)")
    print("\nlag overlap (all layers, mean |routed(T) ∩ routed(T+ℓ)|/10):")
    for l_lag in (1, 2, 4, 8):
        ov = []
        for li in range(n_layers):
            r = [set(selections[li, t].tolist()) for t in range(t_tokens)]
            pairs = [len(r[t] & r[t + l_lag]) / TOPK for t in range(t_tokens - l_lag)]
            ov.append(np.mean(pairs))
        print(f"  ℓ={l_lag}: {np.mean(ov):.1%}")


if __name__ == "__main__":
    main()
