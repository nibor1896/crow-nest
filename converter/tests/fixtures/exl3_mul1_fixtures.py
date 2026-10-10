"""crow-nest #181: the exllamav3 reference fixtures of `converter/src/mul1.rs` (MUL1 trellis codec).

Run with the exllamav3 venv (exllamav3 1.6.0 cu132 / torch 2.13.0+cu132, CUDA GPU), isolated:

    .venv-exl3/Scripts/python.exe -I converter/tests/fixtures/exl3_mul1_fixtures.py converter/tests/fixtures
    (optional: --glm-out DIR  also writes one quantized GLM-shaped expert, too large to commit)

Writes
- exl3-mul1-q-*.bin: one routed "expert" (gate/up [hidden, inter], down [inter, hidden]) quantized by
  exllamav3's own `quantize_exl3` with the mul1 codebook from seeded N(0, 0.02^2) weights and an identity
  Hessian, stored as one UNPADDED record: [gate.trellis][up.trellis][down.trellis][gate.suh][gate.svh]
  [up.suh][up.svh][down.suh][down.svh], little endian;
- exl3-mul1-decode.tsv: per case the sha256 of the fp16 LE bytes of exllamav3's `reconstruct` (the
  CUDA decode kernel) for gate, up and down. "synth:<seed>" cases decode synthetic trellis streams
  (`synth_words`, the same formula as `mul1.rs` tests) at GLM-5.3-Flash's routed-expert shapes;
- exl3-mul1-glm-sizes.tsv: the tensor byte sizes `quantize_exl3` writes for one GLM-5.3-Flash expert
  (hidden 4096, inter 2048) at K = 3.
"""
import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import torch

from exllamav3.version import __version__ as exl3_version
from exllamav3.ext import exllamav3_ext as ext
from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3

# (name, K, hidden, inter, source): quantizer cases are small enough to commit
CASES = [
    ("q-k3", 3, 256, 128, "quant"),
    ("q-k2", 2, 384, 256, "quant"),
    ("q-k4", 4, 128, 128, "quant"),
    ("q-k3.5", 3.5, 256, 128, "quant"),
    ("glm-k3", 3, 4096, 2048, "synth:11"),
    ("glm-k2", 2, 4096, 2048, "synth:23"),
    ("glm-k3.5", 3.5, 4096, 2048, "synth:37"),
]


def synth_words(count, seed):
    i = np.arange(count, dtype=np.uint64)
    x = (i * 0x9E3779B1 + seed * 0x85EBCA6B) & 0xFFFFFFFF
    x ^= x >> 15
    x = (x * 0x2C1B3C6D) & 0xFFFFFFFF
    x ^= x >> 12
    return (x & 0xFFFF).astype(np.uint16)


def words_per_tile(K):
    return int(16 * K)


def shapes(hidden, inter):
    return [(hidden, inter), (hidden, inter), (inter, hidden)]


def identity_h(k):
    return {
        "H": torch.eye(k, dtype=torch.float32, device="cuda:0"),
        "first_key": "synthetic",
        "count": 1,
        "finalized": False,
        "num_total": 1,
        "inf_nan": torch.zeros(2, dtype=torch.long, device="cuda:0"),
        "device": torch.device("cuda:0"),
    }


def quantize(k, n, K, seed):
    g = torch.Generator().manual_seed(seed)
    w = (torch.randn(k, n, generator=g) * 0.02).float().cuda()
    qa = {"K": K, "mul1": True, "seed": seed, "devices": [0], "apply_out_scales": None}
    _, _, out = quantize_exl3(w, identity_h(k), qa, False)
    tr, suh, svh = out["trellis"], out["suh"], out["svh"]
    assert tr.dtype == torch.int16 and tuple(tr.shape) == (k // 16, n // 16, words_per_tile(K)), tr.shape
    assert suh.dtype == torch.half and suh.numel() == k and svh.dtype == torch.half and svh.numel() == n
    return tr.contiguous(), suh.contiguous(), svh.contiguous()


def decode_sha(tr, k, n, K):
    w = torch.empty((k, n), dtype=torch.half, device="cuda:0")
    ext.reconstruct(w, tr.cuda(), K, False, True)
    torch.cuda.synchronize()
    return hashlib.sha256(w.cpu().numpy().tobytes()).hexdigest()


def le_bytes(t):
    return t.cpu().view(torch.int16).numpy().astype("<i2").tobytes()


def quant_expert(K, hidden, inter, seed0):
    mats = [quantize(k, n, K, seed0 + i) for i, (k, n) in enumerate(shapes(hidden, inter))]
    raw = b"".join(le_bytes(m[0]) for m in mats) + b"".join(le_bytes(m[1]) + le_bytes(m[2]) for m in mats)
    return mats, raw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out", type=Path)
    ap.add_argument("--glm-out", type=Path)
    a = ap.parse_args()
    dev = torch.cuda.get_device_name(0)
    rows = []
    for ci, (name, K, hidden, inter, source) in enumerate(CASES):
        shas = []
        if source == "quant":
            mats, raw = quant_expert(K, hidden, inter, 1000 + 10 * ci)
            fn = f"exl3-mul1-{name}.bin"
            (a.out / fn).write_bytes(raw)
            for (tr, _, _), (k, n) in zip(mats, shapes(hidden, inter)):
                shas.append(decode_sha(tr, k, n, K))
            src = fn
        else:
            seed = int(source.split(":")[1])
            wpt = words_per_tile(K)
            for i, (k, n) in enumerate(shapes(hidden, inter)):
                words = synth_words(k // 16 * (n // 16) * wpt, seed + 3 * i)
                tr = torch.from_numpy(words.view(np.int16).copy()).view(k // 16, n // 16, wpt)
                shas.append(decode_sha(tr, k, n, K))
            src = source
        rows.append(f"{name}\t{src}\t{K}\t{hidden}\t{inter}\t" +"\t".join(shas))
        print(name, "done", flush=True)
    head = (f"# case\tsource\tK\thidden\tinter\tsha256 of exllamav3 reconstruct fp16 LE: gate\tup\tdown"
            f"   (exllamav3 {exl3_version}, torch {torch.__version__}, {dev})\n")
    (a.out / "exl3-mul1-decode.tsv").write_text(head + "\n".join(rows) + "\n", newline="\n")

    # one GLM-5.3-Flash routed expert at K = 3 through the quantizer: tensor sizes (and, on request, the bytes)
    mats, raw = quant_expert(3, 4096, 2048, 4242)
    names = ["gate", "up", "down"]
    lines = [f"# matrix\ttrellis shape\ttrellis bytes\tsuh bytes\tsvh bytes   (exllamav3 {exl3_version} "
             f"quantize_exl3, mul1, K = 3, identity Hessian, {dev})"]
    for nm, (tr, suh, svh) in zip(names, mats):
        lines.append(f"{nm}\t{list(tr.shape)}\t{tr.numel() * 2}\t{suh.numel() * 2}\t{svh.numel() * 2}")
    (a.out / "exl3-mul1-glm-sizes.tsv").write_text("\n".join(lines) + "\n", newline="\n")
    print("glm expert record payload bytes:", len(raw))
    if a.glm_out:
        a.glm_out.mkdir(parents=True, exist_ok=True)
        (a.glm_out / "glm-k3.bin").write_bytes(raw)
        shas = [decode_sha(tr, k, n, 3) for (tr, _, _), (k, n) in zip(mats, shapes(4096, 2048))]
        (a.glm_out / "glm-k3.tsv").write_text("\t".join(shas) + "\n", newline="\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
