
import json, os, struct
import numpy as np
import torch
from safetensors import safe_open

MODELS = "models/Qwen3.8-Flash-Next-original"
T = 8
CNQ = "converter/Qwen3.8-Flash-Next-CNQ4.5.cnq"
f = open(CNQ, 'rb'); f.seek(0,2); end = f.tell()
f.seek(end-8); idx_len = struct.unpack('<Q', f.read(8))[0]
f.seek(end-8-idx_len); idx = json.loads(f.read(idx_len))
blob = idx['blob_offset']
tmap = {(t['name'], t['section']): t for t in idx['tensors']}
MAG = np.array([0.0,0.5,1.0,1.5,2.0,3.0,4.0,6.0], dtype=np.float32)
def ue4m3_arr(b):
    e = (b >> 3) & 0xF; m = (b & 7).astype(np.float32)
    return np.where(e==0, m*1.953125e-3, (1.0+m/8.0)*np.exp2((e-7).astype(np.float32)))
def dequant_tensor(name, section, rows, k):
    t = tmap[(name, section)]
    f.seek(blob + t['offset'])
    raw = np.frombuffer(f.read((t['n_values']+63)//64*36), dtype=np.uint8).reshape(rows*k//64, 36)
    scales = ue4m3_arr(raw[:, :4]).repeat(16, axis=1) * np.float32(t['global_scale'])
    nib = np.empty((raw.shape[0], 64), dtype=np.uint8)
    lo = raw[:, 4:] & 0xF; hi = (raw[:, 4:] >> 4) & 0xF
    nib[:, 0::2] = lo; nib[:, 1::2] = hi
    vals = MAG[nib & 7] * np.where(nib & 8, -1.0, 1.0)
    return (vals * scales).reshape(rows, k)
index = json.load(open(os.path.join(MODELS, "model.safetensors.index.json")))
wm = index["weight_map"]
def f32true(full):
    with safe_open(os.path.join(MODELS, wm[full]), framework="pt", device="cpu") as fo:
        return fo.get_tensor(full).to(torch.float32).numpy()
def read_tag(tag):
    return torch.from_numpy(np.fromfile(f"probes/engine-p8debug/gpu-{tag}.f32", dtype=np.float32).reshape(T, -1))
x = torch.from_numpy(np.fromfile("oracle/golden/layer0-input.f32", dtype=np.float32).reshape(T, 10240)).clone()
prefix = "model.language_model.layers.0.attn_hyper_connection."
sec = "text"
norm = torch.from_numpy(f32true(prefix + "hc_norm.weight"))
n_ref = torch.nn.functional.rms_norm(x, (10240,), (1 + norm), eps=1e-6)
normed_e = read_tag("hc-normed-a")
print("normed  engine vs torch-f32:", float((normed_e - n_ref).abs().max()))
low_e = read_tag("hc-low")
w_dn = dequant_tensor(prefix + "input_mix_weight_down.weight", sec, 320, 10240)
low_fp4 = torch.from_numpy(n_ref.numpy() @ w_dn.T)
w_dn_true = f32true(prefix+'input_mix_weight_down.weight')
print("low     engine vs torch-FP4-dequant:", float((low_e - low_fp4).abs().max()),
      "| torch-FP4 vs torch-f32:", float((low_fp4 - n_ref @ w_dn_true.T).abs().max()))
w_up = dequant_tensor(prefix + "input_mix_weight_up.weight", sec, 10240, 320)
sil = torch.nn.functional.silu(low_fp4 / 4)
mixw_fp4 = torch.from_numpy(sil.numpy() @ w_up.T)
mixw_e = read_tag("hc-mixw-a")
print("mixw    engine vs torch(silu(low_fp4)@up_fp4) pre-sigmoid:", float((mixw_e - mixw_fp4).abs().max()))
