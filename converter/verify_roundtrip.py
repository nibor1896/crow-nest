#!/usr/bin/env python3
"""verify_roundtrip.py — crow-nest #5: decode a CNQ v1 file and check the RTN error bounds.

Bound per element: |deq - orig| <= 1.15 * (max|orig|/6 over its 16-wide sub-block).
1.0 is the half-step of the widest E2M1 gap (4 -> 6), measured against the STORED
scale. The stored scale is CEILING-rounded (smallest ue4m3 >= raw, so values never
clamp); the worst-case stored/raw ratio is one subnormal/normal ladder step up,
+12.5% — hence 1.15. Measured max on synthetic data: 1.054 (round-to-nearest era),
ceiling policy expected similar. Also asserts the container is exactly 4.5 bpw.
"""
import json, struct, sys
import numpy as np
import torch
from safetensors.torch import load_file

cnq, st_file = sys.argv[1], sys.argv[2]
raw = open(cnq, "rb").read()
assert raw[:4] == b"CNQ1", "magic"
idx_len = struct.unpack_from("<Q", raw, len(raw)-8)[0]
idx = json.loads(raw[len(raw)-8-idx_len:len(raw)-8])
blob_off = idx["blob_offset"]
blob = raw[blob_off:blob_off + (len(raw)-8-idx_len-blob_off)]
E2M1 = np.array([0, .5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)

def deq(block_bytes, g):
    block = np.frombuffer(block_bytes, dtype=np.uint8)
    e = ((block[:4].astype(np.uint32) >> 3) & 0xF); m = (block[:4] & 7).astype(np.float32)
    s = np.where(e == 0, m * 2.0**-6 / 8, (1 + m / 8) * 2.0 ** (e.astype(np.int32) - 7)).astype(np.float32) * g
    vals = block[4:]; lo = (vals & 0xF).astype(np.uint32); hi = ((vals >> 4) & 0xF).astype(np.uint32)
    nib = np.empty(64, dtype=np.uint32); nib[0::2] = lo; nib[1::2] = hi
    mag = E2M1[nib & 7]; out = np.where(nib & 8, -mag, mag)
    return out * np.repeat(s, 16)

orig = {k: (v.float().numpy() if v.dtype in (torch.bfloat16, torch.float16) else v.numpy())
        for k, v in load_file(st_file).items()}

ok = True
for t in idx["tensors"]:
    if t["dtype"] != "nvfp4":
        continue  # bf16 keeps pass through byte-exact (spot-checked separately)
    o = orig[t["name"]].reshape(-1)
    b = blob[t["offset"]:t["offset"]+t["len"]]
    d = np.concatenate([deq(b[i*36:(i+1)*36], t["global_scale"]) for i in range(len(b)//36)])
    scales = np.array([np.abs(o[k*16:(k+1)*16]).max()/6.0 for k in range(len(o)//16)], dtype=np.float32)
    err = np.abs(d - o); bound = 1.15 * np.repeat(scales, 16) + 1e-6
    viol = int((err > bound).sum()); bpw = t["len"] * 8 / t["n_values"]
    line_ok = viol == 0 and abs(bpw - 4.5) < 0.01
    ok &= line_ok
    if not line_ok:
        k = int(np.argmax(err - bound))
        print(f"FAIL {t['name']}: viol={viol}/{t['n_values']} bpw={bpw:.2f} "
              f"worst idx {k}: orig={o[k]:.6f} deq={d[k]:.6f} err={err[k]:.6f} "
              f"scale={scales[k//16]:.6e} bound={bound[k]:.6e}")
    else:
        print(f"{t['name']}: viol={viol}/{t['n_values']} "
              f"max_rel_err={(err/(np.repeat(scales,16)+1e-9)).max():.3f} bpw={bpw:.2f}")
print("ROUND-TRIP:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
