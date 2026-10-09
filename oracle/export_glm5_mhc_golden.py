"""export_glm5_mhc_golden.py — crow-nest #161 (GLM-5.3-Flash plan step 14, mHC): goldens for the
engine's mHC block (`engine/src/glm5_mhc.rs`, `engine/src/kernels_glm5_mhc.cu`).

The reference is HF's own module, `Glm5NextTextHyperConnection` (transformers 5.16.1,
`modeling_glm5_next.py:219-295`), built from the checkpoint config (rev eb9eb208,
`engine/tests/fixtures/GLM-5.3-Flash/config.json`: hc_mult 4, hc_eps 1e-6, hc_sinkhorn_iters 20,
rms_norm_eps 1e-5) and filled with SYNTHETIC weights (BF16-representable, as the checkpoint stores
them), run in f32 on the CPU like the layerwise runner `oracle/glm5_layerwise.py` (#158). The
stream mix after the sublayer is the decoder layer's expression (`modeling_glm5_next.py:1316-1318`)
with dtype = f32, the runner's dtype.

  python -I oracle/export_glm5_mhc_golden.py [--out engine/tests/fixtures/glm5/mhc]

Writes (raw little-endian, row-major; shapes, dtypes and sha256 in manifest.json):
  real/  the real block shapes, H 4096 (16,384 values per row), T rows:
         fn.bf16 [24][16384], base.f32 [24], scale.f32 [3], x.f32 [T][4][4096] (the streams),
         y.f32 [T][4096] (a stand-in sublayer output), logits.f32 [T][24], pre.f32 [T][4],
         post.f32 [T][4], comb.f32 [T][4][4], collapsed.f32 [T][4096], expanded.f32 [T][4][4096]
         Row 0 has four identical streams (the embedding broadcast of modeling :1477).
  tiny.json  three cases with H 8 (32 values per row), mild / sharp / extreme comb scale, every
         input and output as JSON numbers (f32 values, exact): the exact-op fixture.
"""
import argparse
import hashlib
import json
import os
import sys
import types

import numpy as np
import torch

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "2")))

from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperConnection  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
CONFIG = os.path.join(REPO, "engine", "tests", "fixtures", "GLM-5.3-Flash", "config.json")
SEED = 20261009


def bf16(t):
    return t.to(torch.bfloat16).float()


def hc_config(hidden):
    c = json.load(open(CONFIG))
    t = c.get("text_config", c)
    assert t["mhc"] and t["hc_mult"] == 4, t
    return types.SimpleNamespace(hc_mult=t["hc_mult"], hc_sinkhorn_iters=t["hc_sinkhorn_iters"],
                                 hc_eps=t["hc_eps"], rms_norm_eps=t["rms_norm_eps"], hidden_size=hidden)


def module(hidden, g, fn_std, scale):
    cfg = hc_config(hidden)
    hc = Glm5NextTextHyperConnection(cfg)
    mix = (2 + cfg.hc_mult) * cfg.hc_mult
    with torch.no_grad():
        hc.fn.copy_(bf16(torch.randn(mix, cfg.hc_mult * hidden, generator=g) * fn_std))
        hc.base.copy_(bf16(torch.randn(mix, generator=g) * 0.5))
        hc.scale.copy_(bf16(torch.tensor(scale)))
    return hc.float().eval(), cfg


def streams(T, hidden, g, broadcast_row0=True):
    x = torch.randn(T, 4, hidden, generator=g)
    x = x * torch.tensor([1.0, 2.5, 0.6, 4.0]).view(1, 4, 1)  # unequal stream magnitudes
    x[:, :, 7] += 30.0  # one massive-activation channel, as residual streams carry
    if broadcast_row0:
        x[0] = x[0, 0].unsqueeze(0).expand(4, -1) * 0.05  # layer 0: the embedding in all 4 streams
    return x


def run(hc, x, y):
    """x [T][4][H], y [T][H]: one batch of T rows, the [B, S, hc, H] layout HF's forward takes."""
    x, y = x.unsqueeze(0), y.unsqueeze(0)
    with torch.no_grad():
        # the module's own first two lines (modeling :278-279), for the logits the engine also exposes
        m = torch.nn.functional.linear(hc.input_norm(x.flatten(start_dim=2).float()), hc.fn.float())
        post, comb, collapsed = hc(x)
        dtype = x.dtype
        expanded = post.to(dtype).unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(comb.to(dtype).transpose(-1, -2), x)
        pre = torch.sigmoid(m[..., :4] * hc.scale[0] + hc.base[:4]) + hc.hc_eps  # modeling :283, not returned by HF
    return tuple(t.squeeze(0) for t in (m, pre, post, comb, collapsed, expanded))


def write(out, name, t, dtype, files):
    path = os.path.join(out, name)
    a = t.detach().contiguous()
    raw = (a.to(torch.bfloat16).view(torch.int16).numpy().astype("<i2") if dtype == "bf16"
           else a.float().numpy().astype("<f4")).tobytes()
    with open(path, "wb") as f:
        f.write(raw)
    files[name] = {"shape": list(a.shape), "dtype": dtype, "sha256": hashlib.sha256(raw).hexdigest()}


def real(out, T=5):
    g = torch.Generator().manual_seed(SEED)
    hc, cfg = module(4096, g, 0.02, [0.7, 1.3, 2.0])
    x = streams(T, 4096, g)
    y = torch.randn(T, 4096, generator=g)
    m, pre, post, comb, collapsed, expanded = run(hc, x, y)
    d = os.path.join(out, "real")
    os.makedirs(d, exist_ok=True)
    files = {}
    write(d, "fn.bf16", hc.fn, "bf16", files)
    for n, t in [("base", hc.base), ("scale", hc.scale), ("x", x), ("y", y), ("logits", m), ("pre", pre),
                 ("post", post), ("comb", comb), ("collapsed", collapsed), ("expanded", expanded)]:
        write(d, f"{n}.f32", t, "f32", files)
    man = {"ticket": "crow-nest#161", "seed": SEED, "T": T, "hidden_size": 4096, "hc_mult": cfg.hc_mult,
           "hc_eps": cfg.hc_eps, "hc_sinkhorn_iters": cfg.hc_sinkhorn_iters, "rms_norm_eps": cfg.rms_norm_eps,
           "torch": torch.__version__, "files": files}
    json.dump(man, open(os.path.join(d, "manifest.json"), "w"), indent=1)
    print(f"real: T {T}, comb row0 {comb[1, 0].tolist()}, pre row1 {pre[1].tolist()}")


def tiny(out):
    g = torch.Generator().manual_seed(SEED + 1)
    cases = []
    for name, sc in [("mild", 0.5), ("sharp", 4.0), ("extreme", 12.0)]:
        hc, cfg = module(8, g, 0.4, [0.7, 1.3, sc])
        x = streams(2, 8, g, broadcast_row0=(name == "mild"))
        y = torch.randn(2, 8, generator=g)
        m, pre, post, comb, collapsed, expanded = run(hc, x, y)
        L = lambda t: [float(v) for v in t.detach().float().flatten().tolist()]  # noqa: E731
        cases.append({"name": name, "hidden": 8, "T": 2, "fn": L(hc.fn), "base": L(hc.base), "scale": L(hc.scale),
                      "x": L(x), "y": L(y), "logits": L(m), "pre": L(pre), "post": L(post), "comb": L(comb),
                      "collapsed": L(collapsed), "expanded": L(expanded)})
    json.dump({"ticket": "crow-nest#161", "seed": SEED + 1, "torch": torch.__version__, "cases": cases},
              open(os.path.join(out, "tiny.json"), "w"), indent=0)
    print("tiny: " + ", ".join(c["name"] for c in cases))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=os.path.join(REPO, "engine", "tests", "fixtures", "glm5", "mhc"))
    a = p.parse_args(argv)
    os.makedirs(a.out, exist_ok=True)
    real(a.out)
    tiny(a.out)


if __name__ == "__main__":
    sys.exit(main())
