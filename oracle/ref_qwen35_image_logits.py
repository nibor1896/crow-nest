# Reference side of the IMAGE-PROMPT end-to-end parity for the dense Qwen3.8-27B
# (qwen3_5) — crow-nest #122, P2 of decode_out/p3-vit27b/PREREG.md.
#
# The layer-streamed 64-layer CPU f32 forward of ref_qwen35_logits.py, run over the
# exact sequence the engine prefilled for one image request, with the image parts of
# ref_image_prompt_logits.py (Flash-Next) ported to the 27B:
#   - inputs_embeds = embed gather, the visual embeddings SPLICED at the image_pad rows
#     (Qwen3_5Model.forward: inputs_embeds.masked_scatter(image_mask, image_embeds));
#   - position_ids = the 3-D (3, 1, T) interleaved mrope ids of
#     Qwen3_5Model.get_rope_index (transformers 5.16.1), fed to
#     Qwen3_5TextRotaryEmbedding (mrope_section [11, 11, 10], interleaved).
#
# Input: the CROW_VIT_DUMP=<dir> artifacts of ONE image request (serve.rs
# write_vit_dump, vit.rs build_plan):
#   vit-gen-sequence.json  {ids (expanded), rows, prompt_len, cached_n, visual_map,
#                           grids [[t, h, w] in patches], n_visual}
#   gpu-logits.f32         engine logits [rows][248320], positions cached_n..prompt_len-1
#   vit-embeds.f32         engine tower output, all images concatenated [n_visual][5120]
#   imgN.patches.f32 / imgN.meta.json   per image run through the tower
#   imgN.vit-oracle.f32    (only for --embeds oracle) the f32 HF tower output
#                          written by the 27B tower golden (P1)
#
#   --embeds engine    (default) splice the engine's vit-embeds.f32: the compare
#                      isolates the text side (splice, mrope, 64 layers, head)
#   --embeds oracle    splice imgN.vit-oracle.f32: tower + text end to end
#   --weights cnq      (default) the dequantized CNQ4.5 container; bf16 = the originals
#   --positions hf     (default) HF get_rope_index (checked against the HF function
#                      on every run); `engine` = a port of engine/src/vit.rs
#                      mrope_positions, for attribution only
#
# Output (in <dir>, or --out): ref-logits.f32 [rows][248320] (the gpu rows only) and
# ref-image-logits-stats.json: per-row argmax match, top-5 overlap, max |dlogit|,
# KL(ref || gpu), and the P2 verdict (argmax >= 99 % of prompt rows, mean top-5
# overlap >= 4.5, no NaN). Exit 1 on FAIL.
#
# Run:  .venv-oracle/bin/python oracle/ref_qwen35_image_logits.py <dump-dir> [--embeds engine|oracle] [--weights cnq|bf16]
# Self-test (no dump, no weights): .venv-oracle/bin/python oracle/ref_qwen35_image_logits.py --self-test

import argparse
import itertools
import json
import os
import resource
import sys
import time
import types as _types

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwen35_common import LM, MODEL_DIR, ROOT, build_meta, causal_mask, text_config

from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5Model,
    Qwen3_5RMSNorm,
    Qwen3_5TextRotaryEmbedding,
)

CFG = json.load(open(os.path.join(MODEL_DIR, "config.json")))
IMAGE_PAD = CFG["image_token_id"]            # 248056
VISION_START = CFG["vision_start_token_id"]  # 248053
VISION_END = CFG["vision_end_token_id"]      # 248054
MERGE = CFG["vision_config"]["spatial_merge_size"]  # 2

# the PREREG P2 thresholds (decode_out/p3-vit27b/PREREG.md, fixed 2026-09-27)
P2_ARGMAX_FRAC = 0.99
P2_TOP5_MEAN = 4.5


# ---- positions -------------------------------------------------------------------
def hf_positions(types, grids):
    """port of Qwen3_5Model.get_rope_index (5.16.1) for one unpadded sequence:
    text groups arange + cur on all three axes; an image group (grid t, h, w in
    patches) gets T = cur, H = cur + row, W = cur + col over the MERGED grid
    (h/2 x w/2, raster order = the merger's output order); then
    cur += max(h, w) // 2. Returns [T][3] and the mrope delta."""
    it = iter(grids)
    out, cur = [], 0
    for ty, grp in itertools.groupby(types):
        ln = len(list(grp))
        if ty == 0:
            out.extend([[cur + k] * 3 for k in range(ln)])
            cur += ln
        else:
            t, h, w = next(it)
            lh, lw = h // MERGE, w // MERGE
            assert ln == t * lh * lw, f"image group of {ln} rows vs grid {t, h, w} ({t * lh * lw} merged tokens)"
            for tt in range(t):
                for r in range(lh):
                    for c in range(lw):
                        out.append([cur + tt, cur + r, cur + c])
            cur += max(h, w) // MERGE
    mx = max(max(r) for r in out)
    return out, mx + 1 - len(types)


def engine_positions(types, grids):
    """port of engine/src/vit.rs mrope_positions after #123 (merged-grid raster
    order); before #123 (a434d4d) it decomposed k in block-major patch order"""
    it = iter(grids)
    out, cur = [], 0
    for ty, grp in itertools.groupby(types):
        ln = len(list(grp))
        if ty == 0:
            out.extend([[cur + k] * 3 for k in range(ln)])
            cur += ln
        else:
            _, hp, wp = next(it)
            gw = wp // MERGE
            for k in range(ln):
                row, col = divmod(k, gw)
                out.append([cur, cur + row, cur + col])
            cur += max(hp, wp) // MERGE
    mx = max(max(r) for r in out)
    return out, mx + 1 - len(types)


def hf_get_rope_index(ids, types, grids):
    """the transformers function itself, on CPU, without building the 27B: it reads
    only config.vision_config.spatial_merge_size and self.get_vision_position_ids"""
    fake = _types.SimpleNamespace(config=_types.SimpleNamespace(
        vision_config=_types.SimpleNamespace(spatial_merge_size=MERGE)))
    fake.get_vision_position_ids = lambda *a, **k: Qwen3_5Model.get_vision_position_ids(fake, *a, **k)
    g = torch.tensor(grids, dtype=torch.long) if grids else None
    pos, delta = Qwen3_5Model.get_rope_index(fake, torch.tensor([ids]), torch.tensor([types]), image_grid_thw=g)
    return pos, int(delta.item())  # (3, 1, T)


def to_pid(pos):
    return torch.tensor(pos, dtype=torch.long).T.unsqueeze(1).contiguous()  # (3, 1, T)


# ---- self-test ------------------------------------------------------------------------
def synth(pre, grids, post):
    ids = list(range(1000, 1000 + pre))
    for (t, h, w) in grids:
        ids += [VISION_START] + [IMAGE_PAD] * (t * (h // MERGE) * (w // MERGE)) + [VISION_END] + [1100, 1101]
    ids += list(range(2000, 2000 + post))
    return ids, [1 if i == IMAGE_PAD else 0 for i in ids]


def self_test():
    ok = True
    cases = [("one image [1,32,58]", 5, [(1, 32, 58)], 4),
             ("tall image [1,58,32]", 3, [(1, 58, 32)], 7),
             ("two images [1,32,58]+[1,16,16]", 11, [(1, 32, 58), (1, 16, 16)], 5),
             ("800x450 render [1,32,58]", 20, [(1, 32, 58)], 1)]
    for name, pre, grids, post in cases:
        ids, ty = synth(pre, grids, post)
        mine, d_mine = hf_positions(ty, grids)
        hf, d_hf = hf_get_rope_index(ids, ty, grids)
        eng, d_eng = engine_positions(ty, grids)
        eq = torch.equal(to_pid(mine), hf)
        n_eng = int((to_pid(eng) != hf).any(0).sum())
        vis = [r for r, t in enumerate(ty) if t]
        mx = hf[:, 0, vis].max(1).values.tolist()
        print(f"[self-test] {name}: T={len(ids)} visual={len(vis)} port==HF {eq} delta port {d_mine} HF {d_hf} | "
              f"HF image max (t,h,w) {mx} | engine port: {n_eng} rows differ, delta {d_eng}")
        ok &= eq and d_mine == d_hf
        if name.startswith("one image"):
            s = vis[0]
            at = {r: hf[:, 0, r].tolist() for r in (s - 1, s, s + 1, s + 28, s + 29, vis[-1], vis[-1] + 1)}
            print(f"            HF (t,h,w) at rows {at} (vision_start, first visual, .., last visual, vision_end)")
    # no image: the positions and the rotary reduce to ref_qwen35_logits.py
    tc = text_config()
    rot = Qwen3_5TextRotaryEmbedding(config=tc).float().eval()
    for T in (1, 37, 1100):
        ids = list(range(3000, 3000 + T))
        ty = [0] * T
        mine, d_mine = hf_positions(ty, [])
        hf, d_hf = hf_get_rope_index(ids, ty, [])
        x = torch.zeros(1, T, tc.hidden_size)
        c3, s3 = rot(x, to_pid(mine))
        c2, s2 = rot(x, torch.arange(T).view(1, T))  # ref_qwen35_logits.py's call
        eq = torch.equal(to_pid(mine), hf) and d_mine == 0 == d_hf
        req = torch.equal(c3, c2) and torch.equal(s3, s2)
        print(f"[self-test] text only T={T}: port==HF {eq} (delta {d_hf}), rotary (3,1,T) == ref_qwen35 2-D arange: {req} "
              f"cos {tuple(c3.shape)}")
        ok &= eq and req
    # the image rows really change the rotary (the H/W slots of the interleave)
    ids, ty = synth(5, [(1, 32, 58)], 4)
    mine, _ = hf_positions(ty, [(1, 32, 58)])
    x = torch.zeros(1, len(ids), tc.hidden_size)
    _, s3 = rot(x, to_pid(mine))
    tpos = torch.tensor([p[0] for p in mine]).view(1, -1)
    _, s1 = rot(x, tpos)  # sin: the low-frequency slots round cos to 1.0 in f32
    diff_slots = sorted(set((s3[0] != s1[0]).nonzero()[:, 1].tolist()) & set(range(32)))
    print(f"[self-test] image rotary vs T-only: differing freq slots {diff_slots} "
          f"(expect H 1,4,..,31 and W 2,5,..,29)")
    exp = sorted(set(range(1, 33, 3)) | set(range(2, 30, 3)))
    ok &= diff_slots == exp
    print("[self-test]", "PASS" if ok else "FAIL")
    return ok


# ---- the parity run ----------------------------------------------------------------------
def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def read_f32(path):
    return np.fromfile(path, dtype="<f4")


def log_softmax64(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def top5(x):
    return set(np.argpartition(x, -5)[-5:].tolist())


def main(args):
    from qwen35_common import WeightSource

    D = args.dump
    OUT = args.out or D
    os.makedirs(OUT, exist_ok=True)
    seq = json.load(open(os.path.join(D, "vit-gen-sequence.json")))
    ids = [int(i) for i in seq["ids"]]
    T = len(ids)
    vmap = seq["visual_map"]
    grids = [tuple(int(v) for v in g) for g in seq["grids"]]
    rows, cached_n = int(seq["rows"]), int(seq["cached_n"])
    n_visual = int(seq["n_visual"])
    assert len(vmap) == T, f"visual_map {len(vmap)} != ids {T}"
    assert seq["prompt_len"] == T and cached_n + rows == T, f"prompt_len {seq['prompt_len']} cached_n {cached_n} rows {rows} T {T}"
    assert rows > 0, "no gpu logit rows in the dump"
    ty = [1 if m >= 0 else 0 for m in vmap]
    for r in range(T):
        assert (ids[r] == IMAGE_PAD) == (vmap[r] >= 0), f"row {r}: id {ids[r]} vs visual_map {vmap[r]}"
    assert [m for m in vmap if m >= 0] == list(range(n_visual)), "visual_map is not 0..n_visual-1 in order"
    assert sum(t * (h // MERGE) * (w // MERGE) for t, h, w in grids) == n_visual, f"grids {grids} vs n_visual {n_visual}"

    tc = text_config()
    tc._attn_implementation = "eager"
    Hd = tc.hidden_size

    # positions: the port, checked against the HF function on this very sequence
    pos_hf, delta_hf = hf_positions(ty, grids)
    hf_ref, hf_delta = hf_get_rope_index(ids, ty, grids)
    assert torch.equal(to_pid(pos_hf), hf_ref) and delta_hf == hf_delta, "hf_positions port != Qwen3_5Model.get_rope_index"
    pos_eng, delta_eng = engine_positions(ty, grids)
    n_pos_diff = int((to_pid(pos_eng) != hf_ref).any(0).sum())
    pos = pos_hf if args.positions == "hf" else pos_eng
    print(f"qwen35 image ref: T={T} rows {cached_n}..{T - 1} visual {n_visual} grids {grids} weights={args.weights} "
          f"embeds={args.embeds} positions={args.positions} | mrope delta HF {delta_hf} engine-port {delta_eng}, "
          f"{n_pos_diff} rows where the engine port differs from HF")

    # visual embeddings
    if args.embeds == "engine":
        vis = read_f32(os.path.join(D, "vit-embeds.f32"))
        assert vis.size == n_visual * Hd, f"vit-embeds.f32 has {vis.size} floats, want {n_visual} x {Hd}"
        vis = torch.from_numpy(vis.reshape(n_visual, Hd).copy())
    else:
        parts = []
        for i, (t, h, w) in enumerate(grids):
            p = os.path.join(D, f"img{i}.vit-oracle.f32")
            assert os.path.exists(p), f"{p} missing (run the tower golden first; a vit-cache HIT image has no dump)"
            a = read_f32(p)
            nv = t * (h // MERGE) * (w // MERGE)
            assert a.size == nv * Hd, f"{p}: {a.size} floats, want {nv} x {Hd}"
            parts.append(torch.from_numpy(a.reshape(nv, Hd).copy()))
        vis = torch.cat(parts)
    assert not vis.isnan().any(), "NaN in the visual embeddings"

    ws = WeightSource(args.weights)
    t_start = time.time()
    with torch.no_grad():
        E = f"{LM}embed_tokens.weight"
        uniq = sorted(set(i for i in ids if i != IMAGE_PAD))
        rowmap = {i: ws.rows(E, i, i + 1)[0] for i in uniq}
        h = torch.empty(T, Hd, dtype=torch.float32)
        for r, i in enumerate(ids):
            h[r] = vis[vmap[r]] if vmap[r] >= 0 else rowmap[i]
        h = h.unsqueeze(0)
        rotary = Qwen3_5TextRotaryEmbedding(config=tc).float().eval()
        cos, sin = rotary(h, to_pid(pos))
        mask = causal_mask(T)

        for li in range(tc.num_hidden_layers):
            t0 = time.time()
            layer = ws.load(build_meta(Qwen3_5DecoderLayer, tc, li), f"{LM}layers.{li}.")
            t1 = time.time()
            full = layer.block_type == "full_attention"
            h = layer(h, position_embeddings=(cos, sin), attention_mask=mask if full else None,
                      past_key_values=None)
            del layer
            assert not h.isnan().any(), f"NaN after layer {li}"
            print(f"  layer {li:2d} {'attn' if full else 'gdn '} load {t1 - t0:5.1f}s run {time.time() - t1:5.2f}s "
                  f"|h|max {h.abs().max():9.2f}  peak RSS {rss_gb():.1f} GB", flush=True)

        norm = ws.load(build_meta(Qwen3_5RMSNorm, tc.hidden_size, tc.rms_norm_eps), f"{LM}norm.")
        h = norm(h)[0][cached_n:]
        V = ws.n_rows("lm_head.weight")
        logits = torch.empty(rows, V, dtype=torch.float32)
        for r0 in range(0, V, args.lm_head_chunk):
            r1 = min(V, r0 + args.lm_head_chunk)
            logits[:, r0:r1] = h @ ws.rows("lm_head.weight", r0, r1).t()

    ref = logits.contiguous().numpy()
    ref_path = os.path.join(OUT, "ref-logits.f32")
    ref.astype("<f4").tofile(ref_path)
    print(f"ref forward {time.time() - t_start:.0f} s, peak RSS {rss_gb():.1f} GB -> {ref_path}")

    gpu = read_f32(os.path.join(D, "gpu-logits.f32"))
    assert gpu.size == rows * V, f"gpu-logits.f32 has {gpu.size} floats, want {rows} x {V}"
    gpu = gpu.reshape(rows, V)
    per_row = []
    for k in range(rows):
        r = cached_n + k
        g, f = gpu[k], ref[k]
        g_nan = bool(np.isnan(g).any())
        d = float(np.abs(g - f).max()) if not g_nan else float("nan")
        g_am, r_am = int(np.argmax(g)), int(np.argmax(f))
        lr, lg = log_softmax64(f), log_softmax64(g)
        kl = float((np.exp(lr) * (lr - lg)).sum())
        srt = np.partition(f, -2)[-2:]
        per_row.append({"pos": r, "id": ids[r], "visual": vmap[r] >= 0, "argmax_gpu": g_am, "argmax_ref": r_am,
                        "match": g_am == r_am, "top5": len(top5(g) & top5(f)), "max_abs": d, "kl": kl,
                        "ref_margin": float(srt[1] - srt[0]), "nan": g_nan})

    def summ(sel):
        if not sel:
            return None
        return {"rows": len(sel), "argmax_match": sum(p["match"] for p in sel),
                "argmax_frac": sum(p["match"] for p in sel) / len(sel),
                "top5_mean": float(np.mean([p["top5"] for p in sel])),
                "max_abs": float(np.nanmax([p["max_abs"] for p in sel])),
                "kl_mean": float(np.mean([p["kl"] for p in sel])), "kl_max": float(np.max([p["kl"] for p in sel]))}

    allr = summ(per_row)
    nan_gpu = int(np.isnan(gpu).sum())
    nan_ref = int(np.isnan(ref).sum())
    passed = allr["argmax_frac"] >= P2_ARGMAX_FRAC and allr["top5_mean"] >= P2_TOP5_MEAN and nan_gpu == 0 and nan_ref == 0
    first_dev = next((p["pos"] for p in per_row if not p["match"]), None)
    stats = {
        "ticket": "crow-nest #122 P2", "dump": os.path.abspath(D), "weights": args.weights, "embeds": args.embeds,
        "positions": args.positions, "T": T, "cached_n": cached_n, "rows": rows, "n_visual": n_visual, "grids": grids,
        "mrope_delta_hf": delta_hf, "mrope_delta_engine_port": delta_eng, "rows_engine_port_differs_from_hf": n_pos_diff,
        "thresholds": {"argmax_frac": P2_ARGMAX_FRAC, "top5_mean": P2_TOP5_MEAN, "nan": 0},
        "all": allr, "text_rows": summ([p for p in per_row if not p["visual"]]),
        "visual_rows": summ([p for p in per_row if p["visual"]]),
        "nan_gpu": nan_gpu, "nan_ref": nan_ref, "first_argmax_dev_pos": first_dev,
        "last_row_argmax": {"gpu": per_row[-1]["argmax_gpu"], "ref": per_row[-1]["argmax_ref"]},
        "verdict": "PASS" if passed else "FAIL", "provenance": ws.provenance(),
        "per_row": per_row,
    }
    sp = os.path.join(OUT, "ref-image-logits-stats.json")
    json.dump(stats, open(sp, "w"), indent=1)
    t_, v_ = stats["text_rows"], stats["visual_rows"]
    print(f"P2 [{args.embeds} embeds, {args.positions} positions, {args.weights}]: argmax {allr['argmax_match']}/{rows} "
          f"({100 * allr['argmax_frac']:.2f} %, need {100 * P2_ARGMAX_FRAC:.0f} %) | top-5 mean {allr['top5_mean']:.3f} "
          f"(need {P2_TOP5_MEAN}) | max |dlogit| {allr['max_abs']:.3e} | KL mean {allr['kl_mean']:.3e} max {allr['kl_max']:.3e} "
          f"| nan gpu {nan_gpu} ref {nan_ref} | first argmax dev at pos {first_dev}")
    if t_:
        print(f"  text rows   {t_['argmax_match']}/{t_['rows']} top-5 {t_['top5_mean']:.3f} KL mean {t_['kl_mean']:.3e}")
    if v_:
        print(f"  visual rows {v_['argmax_match']}/{v_['rows']} top-5 {v_['top5_mean']:.3f} KL mean {v_['kl_mean']:.3e}")
    print(f"P2 verdict: {stats['verdict']} -> {sp}")
    return passed


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dump", nargs="?", default=os.environ.get("CROW_VIT_DUMP") or os.path.join(ROOT, "decode_out", "vit-dump"))
    ap.add_argument("--embeds", default="engine", choices=["engine", "oracle"])
    ap.add_argument("--weights", default="cnq", help="cnq | bf16 (qwen35_common.WeightSource kinds)")
    ap.add_argument("--positions", default="hf", choices=["hf", "engine"])
    ap.add_argument("--out", default=None, help="output dir (default: the dump dir)")
    ap.add_argument("--threads", type=int, default=int(os.environ.get("ORACLE_THREADS", "16")))
    ap.add_argument("--lm-head-chunk", type=int, default=16384)
    ap.add_argument("--self-test", action="store_true", help="position/rotary checks against transformers, no dump")
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.set_float32_matmul_precision("highest")
    if a.self_test:
        sys.exit(0 if self_test() else 1)
    sys.exit(0 if main(a) else 1)
