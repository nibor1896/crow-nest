# Export the glm5_next head goldens - crow-nest #165 (plan step 13e, gate G3 head part).
#
# What the head computes (docs/glm5-next-recipe.md section 3 rows 4-6; modeling_glm5_next.py
# of transformers 5.16.1 in .venv-oracle):
#   h      = Glm5NextTextHyperHead(X)      unweighted mean over the 4 streams   (:298-302, :1493)
#   normed = Glm5NextTextRMSNorm(h)        weight * x_hat, eps 1e-5, f32        (:66-80, :1421)
#   logits = normed @ lm_head.T            [154880, 4096], untied               (:2075, :2179-2181)
# The two HF modules run as they are (no re-implementation); the lm_head product is the one
# oracle/glm5_layerwise.py writes its logits-anchor files with (h @ W.T in f32, row chunks).
#
# SYNTHETIC weights, REAL shapes: hidden 4096, hc_mult 4, the full vocab of 154,880, eps 1e-5.
# The lm_head (1.27 GB in BF16) is not stored: both sides generate it from a counter hash,
# integer-exact, so Python and Rust (engine/src/glm5_head.rs, `testkit`) build the same bits:
#   splitmix64(seed, i) = mix(seed + (i + 1) * 0x9E3779B97F4A7C15)            (wrapping u64)
#   mix(z) = z ^= z >> 30; z *= 0xBF58476D1CE4E5B9; z ^= z >> 27; z *= 0x94D049BB133111EB; z ^ z >> 31
#   lm_head[r][c] = ((h >> 56) - 128) * 2^-9         h = splitmix64(SEED_LM, r * 4096 + c)
#   norm_w[d]     = 0.5 + (h >> 57) * 2^-7           h = splitmix64(SEED_NORM, d)
# Every value is exact in BF16 (at most 8 significant bits), so the BF16 tensor and its f32
# widening are the same numbers. The input X [T][4][4096] f32 is stored (x.f32): four anchors
# of different character (see ROWS), the streams distinct so the mean is no single stream.
#
# Output: engine/tests/fixtures/glm5/head/{x.f32, normed.f32, logits.f32, manifest.json}, raw
# little-endian row-major; manifest = shapes, seeds, versions, sha256, pinned hash samples
# (the Rust generator is asserted against them), per-anchor top-1 and the f32-vs-f64 noise of
# the golden itself.
#
# Run (CPU, torch threads 2, ~1 min):
#   .venv-oracle/Scripts/python.exe -I oracle/export_glm5_head_golden.py
# Deterministic: no clock in the output.

import hashlib
import json
import os

import numpy as np
import torch
import transformers
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextHyperHead, Glm5NextTextRMSNorm

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "2")))

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "engine", "tests", "fixtures", "glm5", "head")
H, HC, V, EPS = 4096, 4, 154880, 1e-5
SEED_X, SEED_LM, SEED_NORM = 20261009, 0x165_0001, 0x165_0002
GAMMA = np.uint64(0x9E3779B97F4A7C15)
CHUNK = 8192  # lm_head rows per generated block
ROWS = [
    "plain: streams a_s * randn + b_s, a_s, b_s distinct per stream",
    "large: row 0's character x 300 (the norm must not depend on the scale)",
    "tiny: mean(m^2) ~ eps, so eps 1e-5 (vs 1e-6) and mean (vs sum) change the result",
    "skewed: stream 3 x 40, streams 0-2 opposite in sign (the mean is no stream)",
]


def splitmix64(seed, idx):
    """idx: uint64 array of counters; wrapping u64 arithmetic (numpy arrays wrap silently)."""
    with np.errstate(over="ignore"):
        z = np.uint64(seed) + (idx + np.uint64(1)) * GAMMA
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def lm_rows(r0, r1):
    idx = np.arange(r0 * H, r1 * H, dtype=np.uint64)
    k = (splitmix64(SEED_LM, idx) >> np.uint64(56)).astype(np.int32) - 128
    return torch.from_numpy((k.astype(np.float32) * np.float32(2.0 ** -9)).reshape(r1 - r0, H))


def norm_weight():
    k = (splitmix64(SEED_NORM, np.arange(H, dtype=np.uint64)) >> np.uint64(57)).astype(np.float32)
    return torch.from_numpy(np.float32(0.5) + k * np.float32(2.0 ** -7))


def inputs():
    g = torch.Generator().manual_seed(SEED_X)
    a = torch.tensor([1.0, 0.6, 1.7, 0.3]).view(HC, 1)
    b = torch.tensor([0.4, -0.9, 0.1, 1.3]).view(HC, 1)
    base = a * torch.randn(HC, H, generator=g) + b
    tiny = (torch.randn(HC, H, generator=g) + b) * 2.5e-3
    sk = torch.randn(HC, H, generator=g)
    sk[:3] = -sk[:3].abs() * torch.tensor([0.5, 1.0, 2.0]).view(3, 1)
    sk[3] = sk[3].abs() * 40.0
    return torch.stack([base, base * 300.0, tiny, sk]).float().contiguous()


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def main():
    os.makedirs(OUT, exist_ok=True)
    x = inputs()
    T = x.shape[0]
    w = norm_weight()
    head = Glm5NextTextHyperHead()
    norm = Glm5NextTextRMSNorm(H, eps=EPS)
    with torch.no_grad():
        norm.weight.copy_(w)
        normed = norm(head(x.unsqueeze(0)))[0].contiguous()  # [1, T, hc, H] -> [T, H]
        logits = torch.empty(T, V)
        logits64 = torch.empty(T, V, dtype=torch.float64)
        # the golden's own f32 noise: the same math in f64
        m64 = x.double().mean(dim=1)
        n64 = w.double() * (m64 * torch.rsqrt(m64.pow(2).mean(-1, keepdim=True) + EPS))
        for r0 in range(0, V, CHUNK):
            r1 = min(V, r0 + CHUNK)
            wl = lm_rows(r0, r1)
            logits[:, r0:r1] = normed @ wl.T
            logits64[:, r0:r1] = n64 @ wl.double().T
    files = {"x.f32": x, "normed.f32": normed, "logits.f32": logits}
    man = {
        "ticket": "crow-nest #165 (plan step 13e)",
        "generator": "oracle/export_glm5_head_golden.py",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "threads": torch.get_num_threads(),
        "hidden": H, "hc_mult": HC, "vocab": V, "rms_norm_eps": EPS, "anchors": T, "rows": ROWS,
        "seed_x": SEED_X, "seed_lm": SEED_LM, "seed_norm": SEED_NORM,
        "lm_head": "((splitmix64(seed_lm, r*hidden+c) >> 56) - 128) * 2^-9",
        "norm_weight": "0.5 + (splitmix64(seed_norm, d) >> 57) * 2^-7",
        # pinned samples of the generator (raw u64 as decimal strings, values as f32)
        "hash_samples": [{"seed": SEED_LM, "i": i, "u64": str(int(splitmix64(SEED_LM, np.array([i], dtype=np.uint64))[0]))}
                         for i in (0, 1, 4095, 4096, 123456789, V * H - 1)],
        "lm_head_samples": [{"r": r, "c": c, "v": float(lm_rows(r, r + 1)[0, c])}
                            for r, c in ((0, 0), (0, 4095), (77, 1234), (V - 1, 4095))],
        "norm_weight_samples": [{"d": d, "v": float(w[d])} for d in (0, 1, 2047, 4095)],
        "top1": [int(i) for i in logits.argmax(dim=1)],
        "golden_vs_f64_max_abs": {
            "normed": float((normed.double() - n64).abs().max()),
            "logits": float((logits.double() - logits64).abs().max()),
        },
        "golden_vs_f64_min_cosine_logits": float(torch.nn.functional.cosine_similarity(logits.double(), logits64, dim=1).min()),
        "files": {},
    }
    for nm, t in files.items():
        p = os.path.join(OUT, nm)
        t.numpy().astype("<f4").tofile(p)
        man["files"][nm] = {"shape": list(t.shape), "dtype": "f32", "sha256": sha256(p)}
    with open(os.path.join(OUT, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(json.dumps({k: man[k] for k in ("top1", "golden_vs_f64_max_abs", "golden_vs_f64_min_cosine_logits")}))


if __name__ == "__main__":
    main()
