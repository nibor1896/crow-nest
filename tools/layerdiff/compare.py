# layerdiff: crow-nest decode dumps vs llama.cpp cb_eval dumps at one position
import json, os, sys, numpy as np
R = os.path.join(os.path.abspath(os.environ.get("LAYERDIFF_DIR", "decode_out/layerdiff")), "")  # sites.json + dump dirs
H, HC, L = 2560, 4, 48
def lf(d, name, k=0):
    for ext in ("f32", "i32"):
        p = f"{d}/{name}#{k}.{ext}"
        if os.path.exists(p): return np.fromfile(p, dtype=np.float32 if ext == "f32" else np.int32)
    return None
def cf(d, tag, l):
    for ext in ("f32", "i32"):
        p = f"{d}/L{l:02d}-{tag}.{ext}"
        if os.path.exists(p): return np.fromfile(p, dtype=np.float32 if ext == "f32" else np.int32)
    return None
def cos(a, b): return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
def rel(a, b): return float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-30))
def smean(x): return x.reshape(HC, H).mean(0)
def lsm(x): m = x.max(); return x - m - np.log(np.exp(x - m).sum())
def llama_layer(ld, l):
    d = {}
    d["h_out"] = lf(ld, f"l_last-{l}")
    d["attn_mixed"] = lf(ld, f"hc_mixed-{l}", 0); d["ffn_mixed"] = lf(ld, f"hc_mixed-{l}", 1)
    inj0, inj1 = lf(ld, f"hc_inject-{l}", 0), lf(ld, f"hc_inject-{l}", 1)
    d["attn_injw"] = 2 / (1 + np.exp(-inj0 / HC)) if inj0 is not None else None
    d["ffn_injw"] = 2 / (1 + np.exp(-inj1 / HC)) if inj1 is not None else None
    la = lf(ld, f"linear_attn_out-{l}")
    d["sub_out"] = la if la is not None else lf(ld, f"attn_output-{l}")
    d["kind"] = "gdn" if la is not None else "attn"
    d["x1"] = lf(ld, f"hc_combine-{l}", 0)
    d["moe_out"] = lf(ld, f"ffn_out-{l}", 0)
    d["routed"] = lf(ld, f"ffn_moe_out-{l}", 0)
    d["shared"] = lf(ld, f"ffn_shexp_gated-{l}", 0)
    d["rids"] = lf(ld, f"ffn_moe_topk-{l}", 0)
    if l == 1:
        g, c = lf(ld, "ple_gated_value-1"), lf(ld, "ple_conv_out-1")
        d["ple_add"] = g + c
    return d
def crow_layer(cd, l):
    d = {k: cf(cd, k, l) for k in ("h_out", "attn_mixed", "ffn_mixed", "attn_injw", "ffn_injw", "sub_out", "x1", "moe_out", "rids")}
    eo, w, sd, sg = cf(cd, "eo", l), cf(cd, "rwts", l), cf(cd, "sdown", l), cf(cd, "sgv", l)
    d["routed"] = (eo.reshape(10, H) * w[:, None]).sum(0)
    d["shared"] = sd / (1 + np.exp(-sg[0]))
    if l == 1:
        d["ple_add"] = cf(cd, "h_after_ple", 1) - cf(cd, "h_in", 0 + 1) if False else cf(cd, "h_after_ple", 1) - cf(cd, "h_out", 0)
    return d
def compare(ld, cd, site):
    rows = []
    prev_l = prev_c = None
    for l in range(L):
        a, b = llama_layer(ld, l), crow_layer(cd, l)
        r = {"layer": l, "kind": a["kind"]}
        for k in ("h_out", "attn_mixed", "sub_out", "x1", "ffn_mixed", "moe_out", "routed", "shared", "ple_add"):
            if a.get(k) is None or b.get(k) is None: continue
            r[k] = (round(cos(b[k], a[k]), 5), round(rel(b[k], a[k]), 4))
        r["h_mean"] = (round(cos(smean(b["h_out"]), smean(a["h_out"])), 5), round(rel(smean(b["h_out"]), smean(a["h_out"])), 4))
        r["streams_cos"] = [round(cos(b["h_out"].reshape(HC, H)[s], a["h_out"].reshape(HC, H)[s]), 4) for s in range(HC)]
        r["attn_injw"] = [np.round(a["attn_injw"], 3).tolist(), np.round(b["attn_injw"], 3).tolist()]
        r["ffn_injw"] = [np.round(a["ffn_injw"], 3).tolist(), np.round(b["ffn_injw"], 3).tolist()]
        ra, rb = set(a["rids"].tolist()), set(b["rids"].tolist())
        r["experts_common"] = len(ra & rb)
        if prev_l is not None:
            r["dl_llama"] = round(rel(a["h_out"], prev_l), 4); r["dl_crow"] = round(rel(b["h_out"], prev_c), 4)
        r["norm_llama"] = round(float(np.linalg.norm(a["h_out"])), 2); r["norm_crow"] = round(float(np.linalg.norm(b["h_out"])), 2)
        # block-output size relative to the residual
        r["sub_frac"] = round(float(np.linalg.norm(a["sub_out"]) / np.linalg.norm(smean(a["x1"]) + 1e-30)), 4)
        prev_l, prev_c = a["h_out"], b["h_out"]
        rows.append(r)
    fin = {}
    rl, rc = lf(ld, "result_norm"), np.fromfile(f"{cd}/L99-mixed_final.f32", dtype=np.float32)
    fin["result_norm"] = (round(cos(rc, rl), 5), round(rel(rc, rl), 4))
    return rows, fin
def margins(ld, cd, cid, xid):
    out = {}
    for name, p in (("llama", f"{ld}/logits.f32"), ("crow", f"{cd}/L99-logits.f32")):
        lg = np.fromfile(p, dtype=np.float32)[:248320]; lp = lsm(lg.astype(np.float64))
        out[name] = {"lp_correct": round(float(lp[cid]), 3), "lp_corrupt": round(float(lp[xid]), 3),
                     "margin": round(float(lp[cid] - lp[xid]), 3), "top1": int(lg.argmax())}
    return out
if __name__ == "__main__":
    arm = sys.argv[1]
    meta = json.load(open(R + "sites.json"))
    res = {}
    for sid, m in meta.items():
        res[sid] = {}
        for p in range(m["n_total"] - 4, m["n_total"]):
            ld, cd = f"{R}llama-{sid}/p{p}", f"{R}cn-{arm}-{sid}/p{p}"
            if not (os.path.isdir(ld) and os.path.isdir(cd)): continue
            rows, fin = compare(ld, cd, sid)
            mg = margins(ld, cd, m["correct_id"], m["corrupt_id"]) if p == m["n_total"] - 1 else None
            res[sid][p] = {"site": p == m["n_total"] - 1, "rows": rows, "final": fin, "margins": mg}
    json.dump(res, open(R + f"compare-{arm}.json", "w"), indent=1)
    print("wrote", R + f"compare-{arm}.json")
