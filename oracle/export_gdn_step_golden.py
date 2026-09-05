# Export step-mode goldens for the GDN decode kernel (p11).
#
# The reference recurrence (torch_recurrent_gated_delta_rule semantics, pinned
# by p6's batched gate) is re-run step by step in torch f32 over the SAME 8
# tokens as the p6 golden input. Dumps:
#   core_per_step.f32   [T, 48, 128]  core attention output per step (PRE-rmsnorm)
#   s_final.f32         [48, 128, 128] recurrent state after the last token
#   conv_state_final.f32 [10240, 3]   last 3 pre-conv mixed_qkv values per channel
#   normed_per_step.f32 [T, 6144]     post RMSNormGated per step
#   y_per_step.f32      [T, 2560]     post out_proj per step (== batched golden rows)
#
# The stepping semantics: per token t — decay S *= exp(g_t) FIRST, then
# kv = S·k, delta = (v − kv)·β, S += k⊗delta, out = S·q. conv_state holds the
# last kernel-1 (=3) pre-conv inputs; the step output is
# w0·cs0 + w1·cs1 + w2·cs2 + w3·x_t followed by silu, then cs shifts.

import json
import os

import numpy as np
import torch

HERE = os.path.dirname(__file__)
GOLDEN = os.path.join(HERE, "golden")
T = 8

from safetensors import safe_open

MODELS = os.path.join(HERE, "..", "models", "Qwen3.8-Flash-Next-original")
index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]
state = {}
for suffix in [
    "in_proj_qkv.weight", "conv1d.weight", "in_proj_z.weight", "in_proj_b.weight",
    "in_proj_a.weight", "A_log", "dt_bias", "norm.weight", "out_proj.weight",
]:
    full = f"model.language_model.layers.0.linear_attn.{suffix}"
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as f:
        state[suffix] = f.get_tensor(full).to(torch.float32)

x = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "layer0-gdn-input.f32"), dtype=np.float32).reshape(T, 2560)).clone()
batched_out = torch.from_numpy(np.fromfile(os.path.join(GOLDEN, "layer0-gdn-output.f32"), dtype=np.float32).reshape(T, 2560)).clone()


def l2norm(t, eps=1e-6):
    return t / torch.sqrt((t * t).sum(-1, keepdim=True) + eps)


w_qkv = state["in_proj_qkv.weight"].float()          # [10240, 2560]
w_conv = state["conv1d.weight"].float().reshape(-1, 4)  # [10240, 4]
w_z = state["in_proj_z.weight"].float()              # [6144, 2560]
w_b = state["in_proj_b.weight"].float()              # [48, 2560]
w_a = state["in_proj_a.weight"].float()              # [48, 2560]
a_log = state["A_log"].float()                       # [48]
dt_bias = state["dt_bias"].float()                   # [48]
w_norm = state["norm.weight"].float()                # [128] shared
w_out = state["out_proj.weight"].float()             # [2560, 6144]

KEY_DIM, VALUE_DIM, NUM_V = 2048, 6144, 48

S = torch.zeros(NUM_V, 128, 128)
conv_state = torch.zeros(10240, 3)

core_steps = torch.zeros(T, NUM_V, 128)
normed_steps = torch.zeros(T, VALUE_DIM)
y_steps = torch.zeros(T, 2560)

for t in range(T):
    xt = x[t]
    mq = w_qkv @ xt                                   # [10240]
    # causal conv step with state + silu
    conv_in = torch.cat([conv_state, mq.unsqueeze(-1)], dim=-1)   # [10240, 4]
    c = (w_conv * conv_in).sum(-1)
    c = torch.nn.functional.silu(c)
    conv_state = conv_in[:, 1:]
    q, k, v = c[:KEY_DIM], c[KEY_DIM:2 * KEY_DIM], c[2 * KEY_DIM:]
    q = q.reshape(16, 128)
    k = k.reshape(16, 128)
    v = v.reshape(NUM_V, 128)
    beta = torch.sigmoid(w_b @ xt)                    # [48]
    g = -torch.exp(a_log) * torch.nn.functional.softplus(w_a @ xt + dt_bias)
    qn = l2norm(q).repeat_interleave(3, dim=0) * (128 ** -0.5)   # [48,128]
    kn = l2norm(k).repeat_interleave(3, dim=0)
    S = S * torch.exp(g).view(-1, 1, 1)
    kv = torch.einsum("hkd,hk->hd", S, kn)
    delta = (v - kv) * beta.view(-1, 1)
    S = S + torch.einsum("hd,he->hde", kn, delta)
    core = torch.einsum("hkd,hk->hd", S, qn)
    core_steps[t] = core
    z = (w_z @ xt).reshape(NUM_V, 128)
    rms = torch.rsqrt((core * core).mean(-1, keepdim=True) + 1e-6)
    normed = w_norm * core * rms * torch.sigmoid(z)
    normed_steps[t] = normed.reshape(-1)
    y_steps[t] = w_out @ normed.reshape(-1)

# sanity: the step-mode outputs must equal the batched p6 golden rows
d = (y_steps - batched_out).abs().max().item()
print("first rows gpu-golden:", batched_out[0][:4].tolist(), " step:", y_steps[0][:4].tolist())
print(f"step vs batched golden: max_abs = {d:.3e}")
assert d < 5e-3, "stepping does not reproduce the batched golden"

np.asarray(core_steps.reshape(-1)).tofile(os.path.join(GOLDEN, "layer0-gdn-step-core.f32"))
np.asarray(S.reshape(-1)).tofile(os.path.join(GOLDEN, "layer0-gdn-step-s.f32"))
np.asarray(conv_state.reshape(-1)).tofile(os.path.join(GOLDEN, "layer0-gdn-step-conv.f32"))
np.asarray(normed_steps.reshape(-1)).tofile(os.path.join(GOLDEN, "layer0-gdn-step-normed.f32"))
np.asarray(y_steps.reshape(-1)).tofile(os.path.join(GOLDEN, "layer0-gdn-step-y.f32"))
print("wrote layer0-gdn-step-{core,s,conv,normed,y}.f32")
