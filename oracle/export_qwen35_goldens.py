# Export sub-block goldens for the dense Qwen3.8-27B (qwen3_5_text) engine
# path — Crow #300 phase 2.
#
# REAL transformers modules (5.16.1), eager, f32, deterministic input:
#   l0-input-layernorm  Qwen3_5RMSNorm  layer 0 input_layernorm   (1+w)      [T][5120]
#   l0-gdn              Qwen3_5GatedDeltaNet layer 0               prompt + decode  [T+D][5120]
#   l0-mlp              Qwen3_5MLP layer 0                                       [T][5120]
#   l3-attn             Qwen3_5Attention layer 3 + Qwen3_5TextRotaryEmbedding    [T+D][5120]
#
# T = 40 prompt tokens at positions 0..39 (40 > 32: a multi-tile / multi-split
# attention path), from a zero state. The two mixer goldens carry D = 4 DECODE
# rows (positions 40..43): after the batched 40-token prompt the SAME module
# runs 4 single tokens against an HF DynamicCache (attention: KV cache; GDN:
# conv state + recurrent state, causal_conv1d_update + the recurrent rule).
# Rows 0..39 are the prompt (batched) outputs, rows 40..43 the decode steps.
# Self-check: a cache-free batched run over all 44 rows must agree.
#
# Weights: the CNQ4.5 container DEQUANTIZED (cnq_weights.CnqReader) — the
# numbers the engine loads — as `<name>-output.f32`; the BF16 originals as
# `<name>-output-bf16.f32` (the quantization-error mark). Same input for both.
# Files: raw little-endian f32, row-major [rows][width]; manifest.json holds
# versions, seed, shapes, sha256 of every file and the container provenance.
#
# Input draws (one torch.Generator, seed 20260926, in this order):
#   norm randn[T][5120]*3, gdn randn[T+D][5120], mlp randn[T][5120], attn randn[T+D][5120]
#
# Run: .venv-oracle/bin/python oracle/export_qwen35_goldens.py   (CPU only, ~1 min)

import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_common import (LM, ROOT, WeightSource, build_meta, causal_mask,
                           rope_1d_reference, sha256_file, text_config)

import transformers
from transformers.cache_utils import DynamicCache
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5Attention,
    Qwen3_5GatedDeltaNet,
    Qwen3_5MLP,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "golden", "qwen35-27b")
SEED = 20260926
T = 40
D = 4
N = T + D
GDN_LAYER = 0
ATTN_LAYER = 3

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "16")))
os.makedirs(OUT, exist_ok=True)
tc = text_config()
H = tc.hidden_size
assert tc.layer_types[GDN_LAYER] == "linear_attention" and tc.layer_types[ATTN_LAYER] == "full_attention"

gen = torch.Generator().manual_seed(SEED)
inputs = {
    "l0-input-layernorm": torch.randn(T, H, generator=gen) * 3.0,
    "l0-gdn": torch.randn(N, H, generator=gen),
    "l0-mlp": torch.randn(T, H, generator=gen),
    "l3-attn": torch.randn(N, H, generator=gen),
}
# drawn AFTER the four above, so their inputs (and files) are unchanged by it
inputs["l17-mlp"] = torch.randn(T, H, generator=gen)
# Crow #300 phase 2: layer 17's down_proj carries one NVFP4 scale byte 0x7F (the E4M3 NaN
# code); through the engine's MMA path it made the residual stream NaN until the load rule
# (gen.rs load_pw_x) rewrote it. This golden is the regression check for that rule.
MLP17_LAYER = 17

rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval()


def rope(pos0, n):
    """HF rotary with text-only position ids (2-D -> expanded to 3 equal mrope
    rows), checked against plain 1-D rope over the 64 rotary dims"""
    pid = torch.arange(pos0, pos0 + n).view(1, n)
    cos, sin = rotary(torch.zeros(1, n, H), pid)
    rc, rs = rope_1d_reference(rotary.inv_freq, pid[0])
    assert cos.shape == (1, n, 64), cos.shape
    assert torch.allclose(cos[0], rc, atol=1e-6) and torch.allclose(sin[0], rs, atol=1e-6), "mrope != 1-D rope"
    return cos, sin


def run_norm(ws):
    m = ws.load(build_meta(Qwen3_5RMSNorm, H, tc.rms_norm_eps), f"{LM}layers.{GDN_LAYER}.input_layernorm.")
    return m(inputs["l0-input-layernorm"][None])[0], {}


def run_mlp(ws):
    m = ws.load(build_meta(Qwen3_5MLP, tc, tc.intermediate_size), f"{LM}layers.{GDN_LAYER}.mlp.")
    return m(inputs["l0-mlp"][None])[0], {}


def run_mlp17(ws):
    m = ws.load(build_meta(Qwen3_5MLP, tc, tc.intermediate_size), f"{LM}layers.{MLP17_LAYER}.mlp.")
    return m(inputs["l17-mlp"][None])[0], {}


def run_gdn(ws):
    m = ws.load(build_meta(Qwen3_5GatedDeltaNet, tc, GDN_LAYER), f"{LM}layers.{GDN_LAYER}.linear_attn.")
    x = inputs["l0-gdn"][None]
    cache = DynamicCache(config=tc)
    rows = [m(x[:, :T], cache_params=cache)[0]]
    lay = cache.layers[GDN_LAYER]
    conv_p = lay.conv_states[0].clone()[0]          # [10240, kernel] pre-conv inputs
    rec_p = lay.recurrent_states[0].clone()[0]      # [48, 128k, 128v]
    for i in range(D):
        assert cache.has_previous_state(GDN_LAYER, state_idx=0)
        rows.append(m(x[:, T + i:T + i + 1], cache_params=cache)[0])
    out = torch.cat(rows, 0)
    full = m(x, cache_params=None)[0]               # cache-free batched (chunked rule) over all 44
    d = (out - full).abs().max().item()
    print(f"    gdn cached-decode vs batched-44: max_abs {d:.3e} (absmax {full.abs().max():.3f})")
    assert d < 1e-3 * max(1.0, full.abs().max().item()), "GDN decode path disagrees with the batched rule"
    aux = {"prompt-conv-state": conv_p, "prompt-recurrent-state": rec_p}
    return out, aux


def run_attn(ws):
    m = ws.load(build_meta(Qwen3_5Attention, tc, ATTN_LAYER), f"{LM}layers.{ATTN_LAYER}.self_attn.")
    x = inputs["l3-attn"][None]
    cache = DynamicCache(config=tc)
    out0, w = m(x[:, :T], rope(0, T), attention_mask=causal_mask(T), past_key_values=cache)
    assert torch.allclose(w.sum(-1), torch.ones_like(w.sum(-1)), atol=1e-5)
    rows = [out0[0]]
    for i in range(D):
        p = T + i
        o, _ = m(x[:, p:p + 1], rope(p, 1), attention_mask=causal_mask(1, past=p), past_key_values=cache)
        rows.append(o[0])
    assert cache.get_seq_length(ATTN_LAYER) == N
    out = torch.cat(rows, 0)
    full, _ = m(x, rope(0, N), attention_mask=causal_mask(N), past_key_values=None)
    d = (out - full[0]).abs().max().item()
    print(f"    attn cached-decode vs batched-44: max_abs {d:.3e} (absmax {full.abs().max():.3f})")
    assert d < 1e-4 * max(1.0, full.abs().max().item()), "attention decode path disagrees with the batched run"
    return out, {}


GOLDENS = [
    ("l0-input-layernorm", run_norm, "Qwen3_5RMSNorm", GDN_LAYER, f"{LM}layers.{GDN_LAYER}.input_layernorm.*",
     "rows 0..39: positions 0..39 (position-independent)"),
    ("l0-gdn", run_gdn, "Qwen3_5GatedDeltaNet", GDN_LAYER, f"{LM}layers.{GDN_LAYER}.linear_attn.*",
     "rows 0..39: batched prompt from a zero conv/recurrent state (chunked rule); rows 40..43: single-token decode "
     "steps with DynamicCache (causal_conv1d_update + recurrent rule). aux: states after the 40-token prompt — "
     "conv [10240][4] (last 4 PRE-conv in_proj_qkv values per channel, oldest first; the decode step uses the "
     "last 3 plus the new token), recurrent [48][128 k][128 v] (S, o = S^T q)"),
    ("l0-mlp", run_mlp, "Qwen3_5MLP", GDN_LAYER, f"{LM}layers.{GDN_LAYER}.mlp.*",
     "rows 0..39: positions 0..39 (position-independent)"),
    ("l3-attn", run_attn, "Qwen3_5Attention + Qwen3_5TextRotaryEmbedding", ATTN_LAYER,
     f"{LM}layers.{ATTN_LAYER}.self_attn.*",
     "rows 0..39: batched prompt, positions 0..39, causal mask; rows 40..43: single-token decode at positions "
     "40..43 against the DynamicCache KV of all previous rows. Text-only rope: position_ids 2-D -> 3 equal mrope "
     "rows == plain 1-D NeoX rope over dims 0..63 (asserted)"),
    ("l17-mlp", run_mlp17, "Qwen3_5MLP", MLP17_LAYER, f"{LM}layers.{MLP17_LAYER}.mlp.*",
     "rows 0..39: positions 0..39 (position-independent); down_proj carries one scale byte 0x7F, read as 0x7E "
     "(the engine's load rule) on the cnq side"),
]


def write(path, t):
    a = t.detach().to(torch.float32).contiguous().numpy()
    a.astype("<f4").tofile(path)
    return list(a.shape)


manifest = {
    "what": "Crow #300 phase 2: dense Qwen3.8-27B (qwen3_5_text) sub-block goldens, real HF modules, eager f32",
    "transformers": transformers.__version__,
    "torch": torch.__version__,
    "numpy": np.__version__,
    "dtype": "float32 (raw little-endian, row-major [rows][width])",
    "attn_implementation": "eager",
    "device": "cpu",
    "seed": SEED,
    "input_draws": "one torch.Generator(seed): norm randn[T][5120]*3, gdn randn[T+D][5120], mlp randn[T][5120], attn randn[T+D][5120], then l17-mlp randn[T][5120]",
    "T_prompt": T,
    "D_decode": D,
    "config": os.path.relpath(os.path.join(ROOT, "models", "Qwen3.8-27B", "config.json"), ROOT),
    "weight_sources": {},
    "goldens": {},
}

outs = {}
SOURCES = os.environ.get("GOLDEN_SOURCES", "cnq,bf16").split(",")  # debugging aid; the product is both
for kind in SOURCES:
    t0 = time.time()
    ws = WeightSource(kind)
    manifest["weight_sources"][kind] = ws.provenance()
    suffix = "" if kind == "cnq" else "-bf16"
    print(f"weights: {kind}")
    with torch.no_grad():
        for name, fn, cls, layer, prefix, rows in GOLDENS:
            out, aux = fn(ws)
            assert not out.isnan().any()
            outs[(name, kind)] = out
            g = manifest["goldens"].setdefault(name, {
                "module": cls, "layer": layer, "layer_type": tc.layer_types[layer], "weights": prefix,
                "rows": rows, "files": {},
            })
            if kind == "cnq":
                f = f"{name}-input.f32"
                g["input_shape"] = write(os.path.join(OUT, f), inputs[name])
                g["files"]["input"] = f
            f = f"{name}-output{suffix}.f32"
            g["output_shape"] = write(os.path.join(OUT, f), out)
            g["files"][f"output_{kind}"] = f
            for k, v in aux.items():
                f = f"{name}-{k}{suffix}.f32"
                g.setdefault("aux_shapes", {})[k] = write(os.path.join(OUT, f), v)
                g["files"][f"{k}_{kind}"] = f
            print(f"  {name}: out {tuple(out.shape)} absmax {out.abs().max():.4f}")
    del ws
    print(f"  {kind} done in {time.time() - t0:.1f} s")

# quantization-error mark: dequantized-CNQ vs BF16-original outputs, same input
for name, *_ in (GOLDENS if len(SOURCES) == 2 else []):
    a, b = outs[(name, "cnq")], outs[(name, "bf16")]
    d = a - b
    manifest["goldens"][name]["cnq_vs_bf16"] = {
        "max_abs": float(d.abs().max()),
        "rel_rms": float(d.pow(2).mean().sqrt() / b.pow(2).mean().sqrt()),
    }
    print(f"  cnq vs bf16 {name}: max_abs {d.abs().max():.3e} rel_rms {manifest['goldens'][name]['cnq_vs_bf16']['rel_rms']:.3e}")

manifest["sha256"] = {f: sha256_file(os.path.join(OUT, f)) for f in sorted(os.listdir(OUT)) if f.endswith(".f32")}
with open(os.path.join(OUT, "manifest.json"), "w") as fh:
    json.dump(manifest, fh, indent=1)
print(f"wrote {len(manifest['sha256'])} .f32 + manifest.json -> {os.path.relpath(OUT, ROOT)}")
import resource  # noqa: E402
print(f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")
