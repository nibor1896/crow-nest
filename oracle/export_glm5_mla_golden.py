"""export_glm5_mla_golden.py - crow-nest #163 (GLM-5.3-Flash plan step 14, MLA/DSA part): the golden of
one glm5_next MLA + DSA attention sub-block on SYNTHETIC weights with the real block shapes.

HF's own `Glm5NextTextAttention` (transformers 5.16.1, eager, f32, CPU) at GLM-5.3-Flash's shapes
(hidden 4096, 64 heads x 256, kv_lora 512, q_lora 1536, indexer 32 x 128, kpool 4, index_topk 2048) runs
T prompt rows in calls of --prompt-chunk rows, then D decode rows one by one, against the layer cache of
glm5_layerwise (`_AppendIndexedLayer`, the DSA slot of #158/#147). T > 2051 reaches the sparse regime
(the indexer drops pools from ncb = 513 complete pools on, row p >= 2051).

Weights and the input rows come from a counter-based generator (splitmix64 of (seed, stream, index)),
written out nowhere: the engine test (`engine/src/glm5_mla.rs`, `synth`) regenerates the same values bit
for bit. Every weight value is BF16-representable (the engine holds the projections in BF16; the
checkpoint's kv_b and indexer are BF16 at source), the input rows are plain f32.

Two goldens, same weights, same input:
  f32       HF as it is in the oracle: every cache value f32.
  bf16kv    HF with the ENGINE's cache precision: the MLA latent c (kv_a_layernorm output) and the indexer
            cache row [key | gate | valid] rounded to BF16 where HF stores them (the 1,024 B and 514 B
            per token of the #159 planner, `Glm5Geo::latent_bytes_per_token` / `indexer_bytes_per_token`).
            Nothing else changes: q, q_resid, indexer q and weights, softmax, o_proj stay f32.
The engine is gated against bf16kv (same stored values, so the selections are the same up to f32
summation order); the f32 golden gives the cost of the BF16 cache (selection rows that differ, cosine).

  python -I oracle/export_glm5_mla_golden.py --out engine/tests/fixtures/glm5/mla   (ORACLE_THREADS=2)

Output (raw little-endian; shapes and sha256 in manifest.json):
  golden-<g>-anchors.f32   [A][4096]   sub-block output at the anchor rows, g in {bf16kv, f32}
  latent-<g>-anchors.f32   [A][512]    the latent c at the anchor rows (before the BF16 store)
  topk-<g>-sparse.u16      [S][512]    rows p >= 2051: the selected pool ids, ascending (pool P = tokens
                                       4P..4P+3); every dense row's selection is checked here to be the
                                       full causal set 0..p and is not stored
  scores-<g>-sparse-anchors.f32 [As][P]  the indexer pool scores of the sparse anchor rows (P = pools of
                                       the last row; entries past a row's own pools are 0)
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np

os.environ.setdefault("ORACLE_THREADS", "2")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch  # noqa: E402
import glm5_common as G  # noqa: E402
import glm5_layerwise as LW  # noqa: E402
from transformers.cache_utils import DynamicCache  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextAttention  # noqa: E402

SEED = 163
LAYER = 3  # the first DSA layer of the schedule (layer % 4 == 3)
MASK64 = (1 << 64) - 1

# ---------------------------------------------------------------- the generator (engine twin: glm5_mla::synth)


def splitmix64(z):
    """numpy uint64 array, wrapping arithmetic"""
    z = z + np.uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return z ^ (z >> np.uint64(31))


def uniform(seed, stream, n):
    """n values in [-1, 1), exact in f32: (top 24 bits of splitmix64(key) - 2^23) / 2^23,
    key = ((stream << 32) | i) ^ (seed * 0xD1B54A32D192ED03 mod 2^64)"""
    salt = np.uint64((seed * 0xD1B54A32D192ED03) & MASK64)
    out = np.empty(n, dtype=np.float32)
    step = 1 << 22
    with np.errstate(over="ignore"):
        for i0 in range(0, n, step):
            i = np.arange(i0, min(n, i0 + step), dtype=np.uint64)
            z = splitmix64((np.uint64(stream) << np.uint64(32) | i) ^ salt)
            m = (z >> np.uint64(40)).astype(np.int64) - (1 << 23)
            out[i0:i0 + len(i)] = m.astype(np.float32) / np.float32(1 << 23)
    return out


def bf16_round(a):
    """f32 -> nearest BF16 (ties to even), returned as f32; no NaN/Inf in these values"""
    b = a.view(np.uint32).astype(np.uint64)
    b = (b + np.uint64(0x7FFF) + ((b >> np.uint64(16)) & np.uint64(1))) & np.uint64(0xFFFF0000)
    return b.astype(np.uint32).view(np.float32)


# (stream id, module key, shape, kind, parameter): kind "lin" = uniform * sqrt(3 / fan_in) (unit-variance
# output for unit-variance input), "one" = 1 + p * u, "lin0" = p * u; every value rounded to BF16
STREAMS = [
    (1, "q_a_proj.weight", (1536, 4096), "lin", None),
    (2, "q_a_layernorm.weight", (1536,), "one", 0.1),
    (3, "q_b_proj.weight", (64 * 256, 1536), "lin", None),
    (4, "kv_a_proj_with_mqa.weight", (512, 4096), "lin", None),
    (5, "kv_a_layernorm.weight", (512,), "one", 0.1),
    (6, "kv_b_proj.weight", (64 * 512, 512), "lin", None),
    (7, "o_proj.weight", (4096, 64 * 256), "lin", None),
    (8, "indexer.wq_b.weight", (32 * 128, 1536), "lin", None),
    (9, "indexer.wk.weight", (128, 4096), "lin", None),
    (10, "indexer.k_norm.weight", (128,), "one", 0.1),
    (11, "indexer.k_norm.bias", (128,), "lin0", 0.1),
    (12, "indexer.weights_proj.weight", (32, 4096), "lin", None),
    (13, "indexer.index_kpool_compress_gate", (128, 4096), "lin", None),
    (14, "indexer.index_kpool_compress_ape", (4, 128), "lin0", 0.8),
]
X_STREAM = 100  # input rows [N][4096]: uniform * sqrt(3), f32, not rounded


def gen_tensor(seed, stream, shape, kind, p):
    n = int(np.prod(shape))
    u = uniform(seed, stream, n)
    if kind == "lin":
        v = u * np.float32(math.sqrt(3.0 / shape[1]))
    elif kind == "one":
        v = np.float32(1.0) + np.float32(p) * u
    else:
        v = np.float32(p) * u
    return bf16_round(v.astype(np.float32)).reshape(shape)


def gen_x(seed, n_rows, hidden=4096):
    return (uniform(seed, X_STREAM, n_rows * hidden) * np.float32(math.sqrt(3.0))).reshape(n_rows, hidden)


def probes(seed):
    """4 values per stream (index 0, 1, 12345 mod n, n-1) as f32 bit patterns: the engine's generator
    test reproduces them without a GPU"""
    out = {}
    for sid, key, shape, kind, p in STREAMS + [(X_STREAM, "x", (8, 4096), "x", None)]:
        t = gen_x(seed, 8) if kind == "x" else gen_tensor(seed, sid, shape, kind, p)
        f = t.reshape(-1)
        n = f.size
        out[key] = [[int(i), int(f[i].view(np.uint32))] for i in (0, 1, 12345 % n, n - 1)]
    return out


# ---------------------------------------------------------------- HF side


def real_config():
    text, vision = LW.mini_config("real", 2048)
    return G.text_config_from_dict(LW._config_dict(text, vision))


class _Bf16IndexedLayer(LW._AppendIndexedLayer):
    """the DSA slot with the engine's indexer cache precision: the packed row [key | gate | valid]
    is stored as BF16 values (valid = 1 is exact)"""

    def update_indexer(self, indexer_key_states):
        return super().update_indexer(indexer_key_states.to(torch.bfloat16).float())


def build_attention(tc, seed):
    attn = G.build_meta(Glm5NextTextAttention, tc, LAYER)
    sd = {key: torch.from_numpy(gen_tensor(seed, sid, shape, kind, p).copy())
          for sid, key, shape, kind, p in STREAMS}
    attn.load_state_dict(sd, strict=True, assign=True)
    return attn.eval()


def run_golden(attn, tc, x, T, chunk, bf16kv):
    """x [N][4096] f32. Returns (y [N][4096], latent [N][512], topk [N][2051] i32, scores per call
    [N][pools] (None-padded), tie rows)."""
    N = x.shape[0]
    cache = DynamicCache(config=tc)
    layer_cls = _Bf16IndexedLayer if bf16kv else LW._AppendIndexedLayer
    cache.layers[LAYER] = layer_cls(N)
    rec = {"topk": [], "lat": [], "scores": []}
    hooks = [attn.indexer.register_forward_hook(lambda m, i, o: rec["topk"].append(o[0].detach().clone()))]

    def lat_hook(m, i, o):
        rec["lat"].append(o[0].detach().clone())
        return o.to(torch.bfloat16).float() if bf16kv else o

    hooks.append(attn.kv_a_layernorm.register_forward_hook(lat_hook))
    LW._watch_index_ties(attn.indexer, rec)
    # the pool scores: index_scores.topk's input (the masked [B, S, P] scores), wrapped like the tie watch
    inner = attn.indexer.forward

    def forward(*a, **kw):
        real_topk = torch.Tensor.topk
        seen = [False]  # the tie watch calls topk twice on the same scores (k + 1, then k)

        def topk(t, k, *ta, **tk):
            if t.dim() == 3 and not seen[0]:
                rec["scores"].append(t[0].detach().clone())
                seen[0] = True
            return real_topk(t, k, *ta, **tk)

        torch.Tensor.topk = topk
        try:
            return inner(*a, **kw)
        finally:
            torch.Tensor.topk = real_topk

    attn.indexer.forward = forward
    y = torch.empty(N, x.shape[1])
    try:
        with torch.no_grad():
            for r0, r1 in LW.call_plan(N, T, chunk):
                rec["r0"] = r0
                out, _, _ = attn(x[None, r0:r1].contiguous(), attention_mask=torch.ones(1, r1 - r0, dtype=torch.bool),
                                 past_key_values=cache, prev_topk_indices=None)
                y[r0:r1] = out[0]
    finally:
        for h in hooks:
            h.remove()
        del attn.indexer.forward
    return (y, torch.cat(rec["lat"], 0), torch.cat(rec["topk"], 0).to(torch.int32), rec["scores"],
            sorted(set(rec["ties"])))


def selections(topk, kpool=4, sel_pools=512):
    """per row: the selected pool ids (ascending) of a sparse row, after checking the row's structure:
    every pool of the row's selection appears with all 4 tokens, the tail is tokens 4*ncb..p, and a dense
    row (ncb <= sel_pools) selects exactly 0..p"""
    sparse = {}
    for p in range(topk.shape[0]):
        row = topk[p][topk[p] >= 0].tolist()
        ncb = (p + 1) // kpool
        s = sorted(row)
        assert len(s) == len(set(s)), f"row {p}: duplicate token ids"
        tail = [t for t in s if t >= kpool * ncb]
        assert tail == list(range(kpool * ncb, p + 1)), f"row {p}: tail {tail}"
        ss = set(s)
        pools = sorted({t // kpool for t in s if t < kpool * ncb})
        for P in pools:
            assert all(kpool * P + j in ss for j in range(kpool)), f"row {p}: pool {P} incomplete"
        assert len(pools) == min(ncb, sel_pools), (p, len(pools), ncb)
        if ncb <= sel_pools:
            assert s == list(range(p + 1)), f"dense row {p} is not the causal set"
        else:
            sparse[p] = pools
    return sparse


def cosine(a, b):
    a, b = a.double(), b.double()
    return float((a * b).sum() / (a.norm() * b.norm()))


def write(out_dir, files, name, arr, dtype):
    path = os.path.join(out_dir, name)
    np.ascontiguousarray(arr).astype(dtype).tofile(path)
    files[name] = {"shape": list(arr.shape), "dtype": np.dtype(dtype).name,
                   "sha256": hashlib.sha256(open(path, "rb").read()).hexdigest()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--T", type=int, default=2200)
    ap.add_argument("--D", type=int, default=8)
    ap.add_argument("--prompt-chunk", type=int, default=512)
    ap.add_argument("--seed", type=int, default=SEED)
    a = ap.parse_args(argv)
    t0 = time.time()
    torch.set_num_threads(int(os.environ["ORACLE_THREADS"]))
    T, D, N = a.T, a.D, a.T + a.D
    tc = real_config()
    assert tc.layer_types[LAYER] == "deepseek_sparse_attention" and tc.index_kpool == 4 and tc.index_topk == 2048
    x = torch.from_numpy(gen_x(a.seed, N).copy())
    os.makedirs(a.out, exist_ok=True)
    anchors = sorted({0, 1, 2, 3, 4, 511, 512, 1023, 1500, 2047, 2050, 2051, 2052, 2100, 2150, T - 1}
                     | set(range(T, N, max(1, D // 4))) | {N - 1})
    anchors = [p for p in anchors if p < N]
    sparse_rows = [p for p in range(N) if (p + 1) // 4 > 512]
    sparse_anchors = [p for p in anchors if p in sparse_rows]
    files, res = {}, {}
    for g in ("bf16kv", "f32"):
        attn = build_attention(tc, a.seed)
        t1 = time.time()
        y, lat, topk, scores, ties = run_golden(attn, tc, x, T, a.prompt_chunk, g == "bf16kv")
        del attn
        sp = selections(topk)
        assert sorted(sp) == sparse_rows, (sorted(sp)[:3], sparse_rows[:3])
        write(a.out, files, f"golden-{g}-anchors.f32", y[anchors].numpy(), "<f4")
        write(a.out, files, f"latent-{g}-anchors.f32", lat[anchors].numpy(), "<f4")
        write(a.out, files, f"topk-{g}-sparse.u16", np.array([sp[p] for p in sparse_rows]), "<u2")
        # the pool scores of the sparse anchors, from the call that held the row
        P = (N // 4)
        sc = np.zeros((len(sparse_anchors), P), dtype=np.float32)
        rows_seen, r = [], 0
        for s in scores:
            rows_seen.append((r, r + s.shape[0], s))
            r += s.shape[0]
        for i, p in enumerate(sparse_anchors):
            for r0, r1, s in rows_seen:
                if r0 <= p < r1:
                    n = (p + 1) // 4
                    sc[i, :n] = s[p - r0, :n].numpy()
        write(a.out, files, f"scores-{g}-sparse-anchors.f32", sc, "<f4")
        res[g] = {"y": y, "sp": sp, "ties": ties, "seconds": round(time.time() - t1, 1)}
        print(f"{g}: {res[g]['seconds']} s, {len(sp)} sparse rows, tie rows {len(ties)}", flush=True)
    diff_rows = [p for p in sparse_rows if res["bf16kv"]["sp"][p] != res["f32"]["sp"][p]]
    cos = [cosine(res["bf16kv"]["y"][p], res["f32"]["y"][p]) for p in range(N)]
    man = {
        "generator": "oracle/export_glm5_mla_golden.py (crow-nest #163)",
        "torch": torch.__version__, "transformers": __import__("transformers").__version__,
        "threads": torch.get_num_threads(), "attn_implementation": tc._attn_implementation,
        "seed": a.seed, "layer": LAYER, "T": T, "D": D, "N": N, "prompt_chunk": a.prompt_chunk,
        "shapes": {"hidden": 4096, "heads": 64, "nope": 256, "v": 256, "kv_lora": 512, "q_lora": 1536,
                   "index_heads": 32, "index_dim": 128, "kpool": 4, "index_topk": 2048},
        "anchors": anchors, "sparse_rows": [sparse_rows[0], sparse_rows[-1]] if sparse_rows else [], "sparse_anchors": sparse_anchors,
        "scores_width": N // 4,
        "streams": [[sid, key, list(shape), kind, p] for sid, key, shape, kind, p in STREAMS],
        "x_stream": X_STREAM, "probes": probes(a.seed),
        "tie_rows": {g: res[g]["ties"] for g in res},
        "bf16kv_vs_f32": {
            "selection_rows_differ": len(diff_rows), "rows": diff_rows[:200],
            "pools_differ": int(sum(len(set(res["bf16kv"]["sp"][p]) ^ set(res["f32"]["sp"][p])) // 2
                                    for p in diff_rows)),
            "cosine_min": min(cos), "cosine_min_row": int(np.argmin(cos)),
            "max_abs": float((res["bf16kv"]["y"] - res["f32"]["y"]).abs().max()),
            "ref_rms": float(res["f32"]["y"].pow(2).mean().sqrt()),
        },
        "seconds": {g: res[g]["seconds"] for g in res},
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    print(json.dumps(man["bf16kv_vs_f32"], indent=1))
    print(f"wrote {len(files)} files to {a.out}, {time.time() - t0:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
