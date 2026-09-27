"""crow-nest #122 P0: every tensor of a llama.cpp mmproj-F16.gguf against the HF
checkpoint's model.visual.* tensor it was converted from (CPU, headers + ~1 GB).

Name map = engine/src/vit.rs `mmproj_plan`; the Conv3d patch kernel is stored as
two temporal halves `v.patch_embd.weight` (t=0) and `.1` (t=1).

usage: python oracle/check_mmproj_vs_hf.py <mmproj.gguf> <hf model dir> [threshold]
"""
import json, os, sys
sys.path.insert(0, os.path.expanduser("~/.local/share/crow/src/llama.cpp/gguf-py"))
import numpy as np
import torch
from gguf import GGUFReader
from safetensors import safe_open


def plan(blocks=27):
    v = {
        "v.patch_embd.weight": ("model.visual.patch_embed.proj.weight", 0),
        "v.patch_embd.weight.1": ("model.visual.patch_embed.proj.weight", 1),
        "v.patch_embd.bias": ("model.visual.patch_embed.proj.bias", None),
        "v.position_embd.weight": ("model.visual.pos_embed.weight", None),
        "v.post_ln.weight": ("model.visual.merger.norm.weight", None),
        "v.post_ln.bias": ("model.visual.merger.norm.bias", None),
        "mm.0.weight": ("model.visual.merger.linear_fc1.weight", None),
        "mm.0.bias": ("model.visual.merger.linear_fc1.bias", None),
        "mm.2.weight": ("model.visual.merger.linear_fc2.weight", None),
        "mm.2.bias": ("model.visual.merger.linear_fc2.bias", None),
    }
    pairs = [("ln1", "norm1"), ("ln2", "norm2"), ("attn_qkv", "attn.qkv"), ("attn_out", "attn.proj"),
             ("ffn_up", "mlp.linear_fc1"), ("ffn_down", "mlp.linear_fc2")]
    for b in range(blocks):
        for g, h in pairs:
            for s in ("weight", "bias"):
                v[f"v.blk.{b}.{g}.{s}"] = (f"model.visual.blocks.{b}.{h}.{s}", None)
    return v


def main():
    gpath, hf = sys.argv[1], sys.argv[2]
    thr = float(sys.argv[3]) if len(sys.argv) > 3 else 0.9999
    idx = json.load(open(os.path.join(hf, "model.safetensors.index.json")))["weight_map"]
    hf_names = sorted(k for k in idx if k.startswith("model.visual."))
    r = GGUFReader(gpath)
    gt = {t.name: t for t in r.tensors}
    p = plan()
    missing_g = sorted(set(p) - set(gt))
    extra_g = sorted(set(gt) - set(p))
    mapped = sorted({h for h, _ in p.values()})
    missing_h = sorted(set(mapped) - set(hf_names))
    unmapped_h = sorted(set(hf_names) - set(mapped))
    handles = {}

    def hft(name):
        f = idx[name]
        if f not in handles:
            handles[f] = safe_open(os.path.join(hf, f), "pt")
        return handles[f].get_tensor(name).to(torch.float64)

    rows, worst = [], (2.0, "")
    for gname, (hname, t) in sorted(p.items()):
        if gname not in gt or hname not in idx:
            continue
        a = torch.from_numpy(np.array(gt[gname].data).astype(np.float64))
        b = hft(hname)
        if t is not None:
            b = b[:, :, t]
        if a.numel() != b.numel():
            rows.append({"gguf": gname, "hf": hname, "cos": -1.0, "max_abs": float("inf"),
                         "note": f"numel {a.numel()} vs {b.numel()}"})
            worst = min(worst, (-1.0, gname))
            continue
        if a.shape != b.shape:
            a = a.reshape(b.shape)
        a, b = a.flatten(), b.flatten()
        cos = float(a @ b / (a.norm() * b.norm()))
        mx = float((a - b).abs().max())
        rows.append({"gguf": gname, "hf": hname, "cos": cos, "max_abs": mx})
        worst = min(worst, (cos, gname))
    ok = (not missing_g and not extra_g and not missing_h and not unmapped_h
          and len(mapped) == len(hf_names) and all(r["cos"] >= thr for r in rows))
    out = {
        "gguf": gpath, "hf": hf, "threshold_cos": thr,
        "gguf_tensors": len(gt), "hf_visual_tensors": len(hf_names), "hf_mapped": len(mapped),
        "compared": len(rows), "missing_in_gguf": missing_g, "extra_in_gguf": extra_g,
        "missing_in_hf": missing_h, "hf_unmapped": unmapped_h,
        "worst_cos": worst[0], "worst_tensor": worst[1],
        "max_abs_overall": max(r["max_abs"] for r in rows),
        "pass": ok, "rows": rows,
    }
    print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=1))
    if len(sys.argv) > 4:
        json.dump(out, open(sys.argv[4], "w"), indent=1)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
