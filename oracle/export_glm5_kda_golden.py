"""export_glm5_kda_golden.py — crow-nest #162 (GLM-5.3-Flash plan step 13b, KDA part): the
goldens of ONE KDA sub-block (Glm5NextTextLinearAttention, transformers 5.16.1, eager, f32,
CPU) at the REAL block shapes of GLM-5.3-Flash (hidden 4096, 64 heads x 128, conv 4, gate
lower bound -5), on SYNTHETIC weights.

The weights (192 MiB for q/k/v alone) are not stored: they and the input rows come from a
counter-based generator (splitmix64) that engine/src/glm5_kda.rs reproduces bit for bit
(`synth`); every value is q * 2^-p with an integer q in [-128, 127], so it is exact in BF16
and in f32 on both sides. The manifest carries the sha256 of every generated tensor (f32
little-endian), the Rust side asserts them first.

HF call plan (the one oracle/glm5_layerwise.py run_layer uses for a layer): the prompt rows
0..T-1 in ONE call (HF's chunked form, chunk 64, `chunk_kimi_delta_attention`), then the
decode rows T..N-1 one by one against the same DynamicCache (recurrent form,
`recurrent_kimi_delta_attention`, `causal_conv1d_update`).

  python -I oracle/export_glm5_kda_golden.py [--out engine/tests/fixtures/glm5/kda]

Output (raw little-endian f32, row-major; shapes and sha256 in manifest.json):
  out.f32              [N][4096]        the sub-block output o_proj(...) of every row
  conv-prompt.f32      [24576][3]       the last 3 pre-conv q|k|v inputs per channel after the prompt
  state-prompt.f32     [4][128][128]    recurrent state S[h][key][value] after the prompt, heads STATE_HEADS
  state-final.f32      [4][128][128]    the same after the last decode row
  manifest.json        also the per-head Frobenius norm of S (all 64 heads) at both points and the
                       max |difference| of HF's own prompt split into calls of SPLIT rows
"""
import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm5_common as G  # noqa: E402

import transformers  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextLinearAttention  # noqa: E402

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "2")))

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CONFIG = os.path.join(ROOT, "engine", "tests", "fixtures", "GLM-5.3-Flash", "config.json")
OUT = os.path.join(ROOT, "engine", "tests", "fixtures", "glm5", "kda")

T = 96  # prompt rows: one full 64-row chunk of HF's chunked form plus a partial one
D = 8  # decode rows
SPLIT = (40, 2, 54)  # HF's own multi-call prompt (incl. a call shorter than the conv window)
STATE_HEADS = (0, 21, 42, 63)
SEED = 0x6C6D35_4B4441  # "glm5" "KDA"

# ---------------------------------------------------------------- the shared generator

GOLD = np.uint64(0x9E3779B97F4A7C15)
C1 = np.uint64(0xBF58476D1CE4E5B9)
C2 = np.uint64(0x94D049BB133111EB)


def ints(tensor_id, n, chunk=1 << 22):
    """q[i] in [-128, 127], i = 0..n-1: the top byte of splitmix64(seed_t + (i + 1) * GOLD) minus 128,
    seed_t = SEED + tensor_id * 2^32 (all arithmetic mod 2^64)"""
    seed = np.uint64((SEED + tensor_id * (1 << 32)) % (1 << 64))
    out = np.empty(n, dtype=np.int16)
    with np.errstate(over="ignore"):
        for a in range(0, n, chunk):
            i = np.arange(a + 1, min(a + chunk, n) + 1, dtype=np.uint64)
            z = seed + i * GOLD
            z = (z ^ (z >> np.uint64(30))) * C1
            z = (z ^ (z >> np.uint64(27))) * C2
            z = z ^ (z >> np.uint64(31))
            out[a:a + len(i)] = (z >> np.uint64(56)).astype(np.int16) - 128
    return out


# (id, module name, shape, value rule); p: v = q * 2^-p; "dt": q * 2^-5 - 2; "norm": (64 + (q >> 2)) * 2^-6
TENSORS = [
    (1, "q_proj.weight", (8192, 4096), 13),
    (2, "k_proj.weight", (8192, 4096), 13),
    (3, "v_proj.weight", (8192, 4096), 13),
    (4, "conv1d.weight", (24576, 1, 4), 8),
    (5, "forget_gate.f_a_proj.weight", (128, 4096), 13),
    (6, "forget_gate.f_b_proj.weight", (8192, 128), 9),
    (7, "forget_gate.dt_bias", (8192,), "dt"),
    (8, "forget_gate.A_log", (64,), 7),
    (9, "b_proj.weight", (64, 4096), 13),
    (10, "g_a_proj.weight", (128, 4096), 13),
    (11, "g_b_proj.weight", (8192, 128), 9),
    (12, "o_norm.weight", (128,), "norm"),
    (13, "o_proj.weight", (4096, 8192), 14),
]
X_ID, X_P = 0, 6  # input rows [N][4096], q * 2^-6 in [-2, 2)


def values(tensor_id, shape, rule):
    n = int(np.prod(shape))
    q = ints(tensor_id, n).astype(np.float32)
    if rule == "dt":
        v = q * np.float32(2.0 ** -5) - np.float32(2.0)
    elif rule == "norm":
        v = (np.float32(64.0) + np.floor(q / np.float32(4.0))) * np.float32(2.0 ** -6)
    else:
        v = q * np.float32(2.0 ** -rule)
    v = v.astype(np.float32).reshape(shape)
    # every value is exact in BF16: the low 16 bits of its f32 pattern are zero
    assert not (v.view(np.uint32) & np.uint32(0xFFFF)).any(), f"tensor {tensor_id} not BF16-exact"
    return v


def sha(a):
    return hashlib.sha256(np.ascontiguousarray(a, dtype="<f4").tobytes()).hexdigest()


# ---------------------------------------------------------------- the HF sub-block

def build(tc):
    attn = G.build_meta(Glm5NextTextLinearAttention, tc, 0)
    sd, shas = {}, {}
    for tid, name, shape, rule in TENSORS:
        v = values(tid, shape, rule)
        shas[name] = sha(v)
        sd[name] = torch.from_numpy(v)
    attn.load_state_dict(sd, assign=True, strict=True)
    return attn.eval(), shas


def run(attn, tc, x, calls):
    """x [N][H]; calls: list of (r0, r1). Returns (out [N][H], conv / state snapshots after each call)"""
    cache = DynamicCache(config=tc)
    out = torch.empty_like(x)
    snaps = []
    with torch.no_grad():
        for r0, r1 in calls:
            out[r0:r1] = attn(x[None, r0:r1].contiguous(), cache_params=cache,
                              attention_mask=torch.ones(1, r1 - r0, dtype=torch.bool))[0]
            layer = cache.layers[0]
            snaps.append((r1, layer.conv_states[0][0].clone(), layer.recurrent_states[0][0].clone()))
    return out, snaps


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args(argv)
    t0 = time.time()
    with open(CONFIG) as f:
        tc = G.text_config_from_dict(json.load(f))
    assert tc.layer_types[0] == "linear_attention"
    lin = dict(hidden=tc.hidden_size, heads=tc.linear_num_heads, head_dim=tc.linear_head_dim,
               conv=tc.linear_conv_kernel_dim, lower_bound=tc.linear_lower_bound, eps=tc.rms_norm_eps,
               act=tc.hidden_act)
    assert (lin["hidden"], lin["heads"], lin["head_dim"], lin["conv"], lin["lower_bound"]) == (4096, 64, 128, 4, -5.0), lin
    attn, shas = build(tc)
    N = T + D
    x = values(X_ID, (N, 4096), X_P)
    shas["x"] = sha(x)
    x = torch.from_numpy(x)

    one = [(0, T)] + [(r, r + 1) for r in range(T, N)]
    out, snaps = run(attn, tc, x, one)
    _, conv_p, s_p = snaps[0]
    _, _, s_f = snaps[-1]
    assert conv_p.shape == (24576, 4), conv_p.shape  # HF keeps the last conv_kernel_size inputs

    cuts = np.cumsum((0,) + SPLIT)
    assert cuts[-1] == T
    split = [(int(cuts[i]), int(cuts[i + 1])) for i in range(len(SPLIT))] + [(r, r + 1) for r in range(T, N)]
    out_s, snaps_s = run(attn, tc, x, split)
    s_ps = snaps_s[len(SPLIT) - 1][2]

    os.makedirs(a.out, exist_ok=True)
    files = {
        "out.f32": out.numpy(),
        "conv-prompt.f32": conv_p[:, 1:].numpy(),
        "state-prompt.f32": s_p[list(STATE_HEADS)].numpy(),
        "state-final.f32": s_f[list(STATE_HEADS)].numpy(),
    }
    man = {
        "ticket": "crow-nest #162",
        "producer": "oracle/export_glm5_kda_golden.py",
        "transformers": transformers.__version__,
        "torch": torch.__version__,
        "config": "engine/tests/fixtures/GLM-5.3-Flash/config.json (zai-org/GLM-5.3-Flash @ eb9eb208), layer 0",
        "kda": lin,
        "T": T, "D": D, "N": N,
        "calls": "prompt rows 0..T-1 in one call, then one call per decode row",
        "seed": SEED, "x": {"id": X_ID, "p": X_P},
        "tensors": [{"id": tid, "name": name, "shape": list(shape), "rule": rule} for tid, name, shape, rule in TENSORS],
        "generated_sha256": shas,
        "state_heads": list(STATE_HEADS),
        "state_norm_prompt": [float(v) for v in s_p.flatten(1).norm(dim=1)],
        "state_norm_final": [float(v) for v in s_f.flatten(1).norm(dim=1)],
        "hf_split": {"calls": list(SPLIT),
                     "out_max_abs": float((out_s - out).abs().max()),
                     "state_prompt_max_abs": float((s_ps - s_p).abs().max())},
        "out_rms": float(out.pow(2).mean().sqrt()),
        "files": {},
    }
    for name, arr in files.items():
        arr = np.ascontiguousarray(arr, dtype="<f4")
        arr.tofile(os.path.join(a.out, name))
        man["files"][name] = {"shape": list(arr.shape), "sha256": sha(arr)}
    with open(os.path.join(a.out, "manifest.json"), "w", newline="\n") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}: N {N}, out rms {man['out_rms']:.4f}, HF split max_abs {man['hf_split']['out_max_abs']:.3e} "
          f"(state {man['hf_split']['state_prompt_max_abs']:.3e}), {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
