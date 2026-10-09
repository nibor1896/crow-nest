"""glm5_mtp.py — crow-nest #182 (GLM-5.3-Flash open point O1): the MTP (NextN) block of glm5_next
as a reference forward, on top of HF's own blocks (transformers 5.16.1, eager, f32, CPU).

HF has no MTP forward: it drops checkpoint layer 45 on load (`_keys_to_ignore_on_load_unexpected`,
modeling_glm5_next.py:1359). The formula below is the one three serving stacks run for this model
(docs/glm5-mtp.md "Sources": vLLM `vllm/models/glm5next/common/mtp.py`, SGLang
`models/glm5_next_nextn.py` -> `deepseek_nextn.py`, llama.cpp `src/models/glm5-next.cpp` graph_mtp,
PR #29928); the blocks are HF's, so only the glue is new:

  h_i    = norm(mean over the 4 streams of the trunk's last layer output)   the trunk's lm_head input
  x_i    = eh_proj( [ enorm(embed(t_{i+1})) | hnorm(h_i) ] )                 eh_proj [H, 2H], e FIRST
  a_i    = MLA+DSA(input_layernorm(x))_i            Glm5NextTextAttention, its own full indexer
  y_i    = x_i + a_i
  z_i    = y_i + MoE(post_attention_layernorm(y))_i Glm5NextTextMoE: router + 288 experts + shared
  n_i    = shared_head.norm(z_i)                    the draft's lm_head input (and the recycled h)
  draft  = argmax(n_i . lm_head^T)                  the TRUNK's lm_head (no shared_head.head stored)

Row i (i = 0 .. N-2) pairs trunk row i with the NEXT token and drafts t_{i+2}; it sits at position i
of the MTP block's own causal cache (vLLM/SGLang shift: ids rotated left, positions kept). No mHC: the
block is a plain pre-norm residual layer on one stream. Where the stacks differ (row 0, chained
steps) the variants of `VARIANTS` run the difference instead of guessing it.

  run       python -I oracle/glm5_mtp.py run --weights container <file.cnq> --fallback fp8-originals <dir>
                --trunk <runner out dir with --capture-head> --out <dir> [--prompt-chunk C] [--no-variants]
            The MTP routed experts come from the container (section `mtp`, MUL1 K=3); every other
            tensor of layer 45 is not in the container (the cnq4.5-glm5-next row omits the block) and
            comes from --fallback. The trunk dir must hold head-norm.f32 / head-logits.f32 (#165).
  selftest  python -I oracle/glm5_mtp.py selftest
            synthetic mini checkpoint + an MTP layer in the original naming and FP8 format, the trunk
            from HF's full model, the oracle against an independent plain-torch formula (manual_mtp).

Output (raw little-endian f32 / i32, R = N - 1 rows; shapes and sha256 in manifest.json):
  mtp-embed.f32 [R][H] embed(t_{i+1})       mtp-h.f32 [R][H] the trunk's h_i (head-norm rows 0..R-1)
  mtp-eh.f32 [R][H] x                       mtp-attn-out.f32 [R][H] a      mtp-ffn-in.f32 [R][H] y
  mtp-ffn-out.f32 [R][H] MoE out            mtp-out.f32 [R][H] z           mtp-head-norm.f32 [R][H] n
  mtp-logits.f32 [R][V]                     mtp-routing-ids.i32 / -weights.f32 [R][top_k] (ascending ids)
  mtp-dsa-topk.i32 [R][W]                   mtp-draft-top1.i32 [R]         trunk-next-top1.i32 [R]
"""
import argparse
import copy
import gc
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402
import glm5_layerwise as LW  # noqa: E402

import transformers  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextAttention,
    Glm5NextTextMoE,
    Glm5NextTextRMSNorm,
)

SEED = 20261009
TOL = 1e-4  # oracle (HF blocks, eager) vs the plain-torch formula, f32, absolute

# How the stacks treat the pairing (docs/glm5-mtp.md, "Differences"). `pos0`: what row 0 sees;
# `order`: the eh_proj input order; `h`: which trunk state feeds hnorm.
VARIANTS = {
    # SGLang deepseek_nextn.py (torch.cat path, no masking); the DeepSeek-V3 recurrence without a mask
    "sglang": dict(pos0="keep", order="eh", h="norm"),
    # vLLM deepseek_mtp.py / glm5next fused_eh_norm: embeds zeroed where position == 0
    "vllm": dict(pos0="zero_embed", order="eh", h="norm"),
    # llama.cpp speculative.cpp: an extra leading row (t_0, h = 0); row i+1 = (t_{i+1}, h_i) at position i+1
    "llamacpp": dict(pos0="lead_row", order="eh", h="norm"),
    # the DeepSeek-V3 paper's written order, eq. (21) [RMSNorm(h); RMSNorm(Emb)] (no code uses it)
    "paper-order": dict(pos0="keep", order="he", h="norm"),
    # the trunk state BEFORE the final norm (no stack uses it)
    "prenorm-h": dict(pos0="keep", order="eh", h="mean"),
}
PRIMARY = "sglang"

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "16")))


# ---------------------------------------------------------------- the block

def mtp_text_config(tc):
    """(a copy of the text config with the MTP block as one more layer, its layer index). The block is a
    DSA layer with a full indexer and a MoE FFN (llama.cpp graph_mtp: `il >= n_layer()` always runs the
    full indexer); the trunk's lists are untouched, so a DynamicCache over this config has the trunk's
    slots plus the block's."""
    L = tc.num_hidden_layers
    m = copy.deepcopy(tc)
    m.num_hidden_layers = L + 1
    m.layer_types = list(tc.layer_types) + ["deepseek_sparse_attention"]
    m.mlp_layer_types = list(tc.mlp_layer_types) + ["sparse"]
    m.indexer_types = list(tc.indexer_types) + ["full"]
    return m, L


class SharedHead(nn.Module):
    def __init__(self, tc):
        super().__init__()
        self.norm = Glm5NextTextRMSNorm(tc.hidden_size, tc.rms_norm_eps)


class Glm5MtpBlock(nn.Module):
    """Checkpoint layer L = num_hidden_layers. Its state_dict keys are the checkpoint names under
    `model.language_model.layers.L.` (so WeightSource.load fills it strictly): enorm, hnorm, eh_proj,
    input_layernorm, self_attn.* (HF's MLA + DSA indexer), post_attention_layernorm, mlp.* (HF's MoE),
    shared_head.norm. No hc_* (no mHC), no embed_tokens / shared_head.head (the trunk's are used)."""

    def __init__(self, tc_mtp, layer_idx):
        super().__init__()
        H, eps = tc_mtp.hidden_size, tc_mtp.rms_norm_eps
        self.layer_idx = layer_idx
        self.enorm = Glm5NextTextRMSNorm(H, eps)
        self.hnorm = Glm5NextTextRMSNorm(H, eps)
        self.eh_proj = nn.Linear(2 * H, H, bias=False)
        self.input_layernorm = Glm5NextTextRMSNorm(H, eps)
        self.self_attn = Glm5NextTextAttention(tc_mtp, layer_idx)
        self.post_attention_layernorm = Glm5NextTextRMSNorm(H, eps)
        self.mlp = Glm5NextTextMoE(tc_mtp)
        self.shared_head = SharedHead(tc_mtp)

    def eh(self, e, h, order="eh"):
        """eh_proj over the concatenation of the two normed inputs (rows x H each)"""
        a, b = self.enorm(e), self.hnorm(h)
        return self.eh_proj(torch.cat([a, b] if order == "eh" else [b, a], dim=-1))


def load_block(ws, tc):
    """(the block filled from `ws`, the MTP text config)"""
    tc_m, L = mtp_text_config(tc)
    block = G.build_meta(Glm5MtpBlock, tc_m, L)
    ws.load(block, f"{G.LM}layers.{L}.")
    return block, tc_m


class MixedSource:
    """`primary` where it holds a name, `fallback` otherwise, request order kept (one converter call per
    run of consecutive primary names). For the 3-bit container: the MTP experts are in it (section
    `mtp`), the rest of layer 45 is not (docs/glm-mul1-conversion.md "Container") and comes from the FP8
    originals."""

    def __init__(self, primary, fallback):
        self.primary, self.fallback = primary, fallback
        self.kind = f"{primary.kind}+{fallback.kind}"
        self.taken = {"primary": [], "fallback": []}

    def _src(self, name):
        if self.primary.has(name):
            return "primary"
        if self.fallback.has(name):
            return "fallback"
        raise G.ContainerError(f"{name}: in neither weight source")

    def has(self, name):
        return self.primary.has(name) or self.fallback.has(name)

    def get_many(self, names):
        runs = []
        for n in names:
            s = self._src(n)
            self.taken[s].append(n)
            if runs and runs[-1][0] == s:
                runs[-1][1].append(n)
            else:
                runs.append((s, [n]))
        for s, ns in runs:
            yield from getattr(self, s).get_many(ns)

    def get(self, name):
        return next(self.get_many([name]))

    def rows(self, name, r0, r1):
        return getattr(self, self._src(name)).rows(name, r0, r1)

    def n_rows(self, name):
        return getattr(self, self._src(name)).n_rows(name)

    def embed(self, ids, name=G.LM + "embed_tokens.weight"):
        return getattr(self, self._src(name)).embed(ids, name)

    def config_dict(self):
        return self.primary.config_dict()

    load = G.WeightSource.load

    def provenance(self):
        return {"primary": self.primary.provenance(), "fallback": self.fallback.provenance(),
                "rule": "a tensor comes from `primary` when it holds the name, else from `fallback`",
                "from_primary": len(self.taken["primary"]), "from_fallback": len(self.taken["fallback"]),
                "fallback_names": sorted(self.taken["fallback"])}


# ---------------------------------------------------------------- the forward

def pair_rows(ids, h_norm, h_mean, embed, variant):
    """the block's input rows for one variant: (e [R'][H], h [R'][H], zero-embed row mask [R'], index of
    the first draft row). Draft row i (output index lead + i) pairs trunk row i with t_{i+1}."""
    v = VARIANTS[variant] if isinstance(variant, str) else variant
    h = h_norm if v["h"] == "norm" else h_mean
    N = len(ids)
    e = embed[1:N]
    hh = h[0:N - 1]
    zero = torch.zeros(N - 1, dtype=torch.bool)
    if v["pos0"] == "zero_embed":
        zero[0] = True
    if v["pos0"] == "lead_row":
        e = torch.cat([embed[0:1], e], 0)
        hh = torch.cat([torch.zeros_like(hh[:1]), hh], 0)
        return e, hh, torch.zeros(N, dtype=torch.bool), 1
    return e, hh, zero, 0


def mtp_forward(block, tc_m, e, h, T_rows, prompt_chunk=None, order="eh", zero_embed=None):
    """e, h: [R][H] f32 (row r at position r of the block's cache). Prompt rows 0..T_rows-1 in calls of
    `prompt_chunk` rows (None = one call), then rows T_rows..R-1 one by one, against the block's own
    cache (the runner's call plan, glm5_layerwise.call_plan). Returns a dict of [R][..] tensors:
    eh, attn_out, ffn_in, ffn_out, out, head_norm, routing (ids, weights ascending), dsa_topk."""
    R, H = e.shape
    if zero_embed is not None and bool(zero_embed.any()):
        e = torch.where(zero_embed[:, None], torch.zeros_like(e), e)
    L = block.layer_idx
    rec = {"route": [], "topk": []}
    hooks = [block.mlp.gate.register_forward_hook(
        lambda m, i, o: rec["route"].append((o[2].detach().clone(), o[1].detach().clone()))),
        block.self_attn.indexer.register_forward_hook(lambda m, i, o: rec["topk"].append(o.detach().clone()))]
    cache = LW.append_in_place(LW.new_layer_cache(tc_m), tc_m, R)
    assert isinstance(cache.layers[L], LW._AppendIndexedLayer), "the MTP slot of the cache is not a DSA slot"
    out = {k: torch.empty(R, H) for k in ("eh", "attn_out", "ffn_in", "ffn_out", "out", "head_norm")}
    try:
        with torch.no_grad():
            for r0, r1 in LW.call_plan(R, T_rows, prompt_chunk):
                x = block.eh(e[None, r0:r1], h[None, r0:r1], order)
                a, _, _ = block.self_attn(
                    hidden_states=block.input_layernorm(x),
                    attention_mask=torch.ones(1, r1 - r0, dtype=torch.bool),
                    position_ids=torch.arange(r0, r1)[None],
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=None,
                    prev_topk_indices=None,
                )
                y = x + a
                f = block.mlp(block.post_attention_layernorm(y))
                z = y + f
                for k, t in (("eh", x), ("attn_out", a), ("ffn_in", y), ("ffn_out", f), ("out", z),
                             ("head_norm", block.shared_head.norm(z))):
                    out[k][r0:r1] = t[0]
    finally:
        for hk in hooks:
            hk.remove()
    del cache
    ids = torch.cat([r[0] for r in rec["route"]], 0)
    w = torch.cat([r[1] for r in rec["route"]], 0)
    o = ids.argsort(dim=-1)
    out["routing"] = (ids.gather(1, o).to(torch.int32), w.gather(1, o).float())
    out["dsa_topk"] = torch.cat([t[0] for t in rec["topk"]], 0).to(torch.int32)
    return out


def lm_head_logits(ws, hn, chunk=LW.LM_HEAD_CHUNK):
    V = ws.n_rows("lm_head.weight")
    logits = torch.empty(hn.shape[0], V)
    for r0 in range(0, V, chunk):
        r1 = min(V, r0 + chunk)
        logits[:, r0:r1] = hn @ ws.rows("lm_head.weight", r0, r1).T
    return logits


class LmHead:
    """the trunk's lm_head held once in RAM (f32 [V][H]) for several variants"""

    def __init__(self, ws, chunk=LW.LM_HEAD_CHUNK):
        V = ws.n_rows("lm_head.weight")
        parts = [ws.rows("lm_head.weight", r0, min(V, r0 + chunk)) for r0 in range(0, V, chunk)]
        self.w = torch.cat(parts, 0)

    def __call__(self, hn):
        return hn @ self.w.T


# ---------------------------------------------------------------- the independent formula (selftest)

def manual_mtp(ws, tc, ids, h, order="eh"):
    """The MTP block in plain torch from the checkpoint tensors (no HF module): full causal attention, so
    only valid while every row selects all its rows (index_topk >= R). Returns (head_norm [R][H],
    logits [R][V]) for the SGLang pairing (row i = (t_{i+1}, h_i))."""
    L = tc.num_hidden_layers
    p = f"{G.LM}layers.{L}."
    g = ws.get
    eps = tc.rms_norm_eps

    def rms(x, w):
        return w * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps))

    N = len(ids)
    e = ws.embed(ids)[1:N]
    hh = h[0:N - 1]
    R = N - 1
    a_, b_ = rms(e, g(p + "enorm.weight")), rms(hh, g(p + "hnorm.weight"))
    x = torch.cat([a_, b_] if order == "eh" else [b_, a_], -1) @ g(p + "eh_proj.weight").T
    # MLA, expanded, causal
    xa = rms(x, g(p + "input_layernorm.weight"))
    nh, dn, dv, kl = tc.num_attention_heads, tc.qk_nope_head_dim, tc.v_head_dim, tc.kv_lora_rank
    qr = rms(xa @ g(p + "self_attn.q_a_proj.weight").T, g(p + "self_attn.q_a_layernorm.weight"))
    q = (qr @ g(p + "self_attn.q_b_proj.weight").T).view(R, nh, dn)
    c = rms(xa @ g(p + "self_attn.kv_a_proj_with_mqa.weight").T, g(p + "self_attn.kv_a_layernorm.weight"))
    kv = (c @ g(p + "self_attn.kv_b_proj.weight").T).view(R, nh, dn + dv)
    k, v = kv[..., :dn], kv[..., dn:]
    s = torch.einsum("qhd,khd->hqk", q, k) * dn ** -0.5
    s = s.masked_fill(~torch.ones(R, R, dtype=torch.bool).tril()[None], float("-inf"))
    o = torch.einsum("hqk,khd->qhd", s.softmax(-1), v).reshape(R, nh * dv)
    y = x + o @ g(p + "self_attn.o_proj.weight").T
    # MoE
    xf = rms(y, g(p + "post_attention_layernorm.weight"))
    sc = (xf @ g(p + "mlp.gate.weight").T).sigmoid()
    top = (sc + g(p + "mlp.gate.e_score_correction_bias")).topk(tc.num_experts_per_tok, -1).indices
    w = sc.gather(1, top)
    w = w / (w.sum(-1, keepdim=True) + 1e-20) * tc.routed_scaling_factor
    lim = tc.swiglu_limit

    def mlp(xr, pre):
        gt = (xr @ g(pre + "gate_proj.weight").T).clamp(max=lim)
        up = (xr @ g(pre + "up_proj.weight").T).clamp(-lim, lim)
        return (F.silu(gt) * up) @ g(pre + "down_proj.weight").T

    f = mlp(xf, p + "mlp.shared_experts.")
    for r in range(R):
        for j in range(top.shape[1]):
            f[r] += w[r, j] * mlp(xf[r:r + 1], f"{p}mlp.experts.{int(top[r, j])}.")[0]
    z = y + f
    hn = rms(z, g(p + "shared_head.norm.weight"))
    return hn, hn @ ws.get("lm_head.weight").T


# ---------------------------------------------------------------- the golden

def read_trunk(trunk_dir):
    """(manifest, ids, T, D, head-norm [N][H], head-mean [N][H], head-logits top-1 [N]) of a runner dir
    written with --capture-head (crow-nest #165)"""
    with open(os.path.join(trunk_dir, "manifest.json")) as f:
        man = json.load(f)
    if not man.get("complete") or "head" not in man:
        raise SystemExit(f"{trunk_dir}: no complete pass with --capture-head (manifest `head` missing); the MTP "
                         "golden needs the trunk's head-norm.f32 / head-logits.f32 of every row")
    ld = lambda role: LW.load_file(trunk_dir, man, man["head"][role])  # noqa: E731
    hn, hm = ld("norm"), ld("mean")
    lg = ld("logits")
    top1 = lg.argmax(-1).to(torch.int32)
    del lg
    return man, man["ids"], man["T"], man["D"], hn, hm, top1


def _text_cfg(cfg):
    return cfg.get("text_config", cfg)


def agreement(draft_top1, trunk_top1, T_rows):
    """draft row i against the trunk's own pick at row i+1 (the token the draft guesses): overall, prompt
    rows, decode rows"""
    hit = (draft_top1 == trunk_top1[1:1 + draft_top1.shape[0]])
    R = hit.shape[0]
    f = lambda m: {"rows": int(m.numel()), "agree": int(m.sum()), "rate": round(float(m.float().mean()), 4) if m.numel() else None}  # noqa: E731
    return {"all": f(hit), "prompt": f(hit[:T_rows]), "decode": f(hit[T_rows:R])}


def run_golden(ws, trunk_dir, out_dir, prompt_chunk=None, variants=True, log=print):
    t0 = time.time()
    tman, ids, T, D, hn, hm, trunk_top1 = read_trunk(trunk_dir)
    N = len(ids)
    R, T_rows = N - 1, T - 1  # draft rows; the prompt-side rows are those whose t_{i+1} is a prompt token
    tc = G.text_config_from_dict(ws.config_dict())
    emb = ws.embed(ids)
    if "embed.f32" in tman.get("files", {}):
        te = LW.load_file(trunk_dir, tman, "embed.f32")
        assert torch.equal(te, emb), "the embedding rows differ from the trunk's embed.f32"
    block, tc_m = load_block(ws, tc)
    t1 = time.time()
    rss_load, _ = LW._rss()
    head = LmHead(ws)
    t2 = time.time()
    res, agr = {}, {}
    for name in (VARIANTS if variants else [PRIMARY]):
        v = VARIANTS[name]
        e, h, zero, lead = pair_rows(ids, hn, hm, emb, name)
        o = mtp_forward(block, tc_m, e, h, T_rows + lead, prompt_chunk, v["order"], zero)
        lg = head(o["head_norm"][lead:])
        top1 = lg.argmax(-1).to(torch.int32)
        agr[name] = dict(agreement(top1, trunk_top1, T_rows), **v)
        log(f"variant {name:12s} draft top-1 = trunk next top-1 on {agr[name]['all']['agree']}/{R} rows "
            f"({agr[name]['all']['rate']})")
        if name == PRIMARY:
            res = {k: (t[lead:] if torch.is_tensor(t) else t) for k, t in o.items()}
            res["logits"], res["top1"], res["e"], res["h"] = lg, top1, e, h
        del o, lg
        gc.collect()
    t3 = time.time()
    os.makedirs(out_dir, exist_ok=True)
    files = {}

    def put(nm, t, dt):
        LW.write_raw(os.path.join(out_dir, nm), t)
        files[nm] = {"shape": list(t.shape), "dtype": dt, "sha256": G.sha256_file(os.path.join(out_dir, nm))}

    for nm, key in (("mtp-embed.f32", "e"), ("mtp-h.f32", "h"), ("mtp-eh.f32", "eh"), ("mtp-attn-out.f32", "attn_out"),
                    ("mtp-ffn-in.f32", "ffn_in"), ("mtp-ffn-out.f32", "ffn_out"), ("mtp-out.f32", "out"),
                    ("mtp-head-norm.f32", "head_norm"), ("mtp-logits.f32", "logits")):
        put(nm, res[key].float().contiguous(), "f32")
    put("mtp-routing-ids.i32", res["routing"][0], "i32")
    put("mtp-routing-weights.f32", res["routing"][1], "f32")
    put("mtp-dsa-topk.i32", res["dsa_topk"], "i32")
    put("mtp-draft-top1.i32", res["top1"], "i32")
    put("trunk-next-top1.i32", trunk_top1[1:N].contiguous(), "i32")
    _, peak = LW._rss()
    tm = os.path.join(trunk_dir, "manifest.json")
    man = {
        "runner": "oracle/glm5_mtp.py (crow-nest #182)",
        "torch": torch.__version__, "transformers": transformers.__version__, "threads": torch.get_num_threads(),
        "attn_implementation": tc._attn_implementation, "experts_implementation": tc._experts_implementation,
        "weights": ws.provenance(),
        "trunk": {"dir": os.path.abspath(trunk_dir), "manifest_sha256": G.sha256_file(tm),
                  "head_norm": tman["files"][tman["head"]["norm"]]["sha256"],
                  "head_logits": tman["files"][tman["head"]["logits"]]["sha256"],
                  "weights_index_json_sha256": tman["weights"].get("index_json_sha256")},
        "ids": [int(i) for i in ids], "N": N, "T": T, "D": D, "rows": R, "prompt_rows": T_rows,
        "hidden_size": tc.hidden_size, "mtp_layer": tc.num_hidden_layers,
        "pairing": "draft row i = (embed(ids[i+1]), trunk head-norm row i) at position i of the block's own cache; "
                   "drafts ids[i+2]; compared with the trunk's argmax at row i+1",
        "primary_variant": PRIMARY, "variants": agr,
        # read, not used: with one draft step the block always runs its own indexer (docs/glm5-mtp.md)
        "index_share_for_mtp_iteration": _text_cfg(ws.config_dict()).get("index_share_for_mtp_iteration"),
        "draft_steps": 1, "prompt_chunk": int(prompt_chunk or 0),
        "timing_s": {"load": round(t1 - t0, 1), "lm_head": round(t2 - t1, 1), "variants": round(t3 - t2, 1)},
        "rss_after_load_gib": rss_load, "peak_wset_gib": peak, "files": files,
    }
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


# ---------------------------------------------------------------- synthetic MTP layer (the proof)

def _init_block(block, seed):
    """values for every tensor of a freshly built block (HF leaves experts empty, norms at 1, the router
    and the score bias at 0): matrices N(0, 1/fan_in), norms 1 + 0.1 N, kpool ape 0.5 N, gate 0.02 N"""
    gen = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for n, t in list(block.named_parameters()) + list(block.named_buffers()):
            r = torch.randn(t.shape, generator=gen)
            if n.endswith("index_kpool_compress_ape"):
                t.copy_(r * 0.5)
            elif n.endswith("index_kpool_compress_gate"):
                t.copy_(r * 0.02)
            elif n.endswith("e_score_correction_bias"):
                t.copy_(r * 0.05)
            elif n.endswith(".bias"):
                t.copy_(r * 0.1)
            elif t.dim() == 1:
                t.copy_(1.0 + 0.1 * r)
            else:
                t.copy_(r * t.shape[-1] ** -0.5)


def add_synthetic_mtp(ckpt_dir, seed=SEED):
    """append an MTP layer (checkpoint layer L = num_hidden_layers) to a synthetic checkpoint written by
    glm5_layerwise.make_synthetic, in the ORIGINAL naming and format (FP8 + weight_scale_inv where the
    real checkpoint has them: attention q_a/q_b/kv_a/o, experts, shared expert; eh_proj and the rest
    BF16, the score bias F32), and set `num_nextn_predict_layers` 1. Returns the number of names added."""
    from safetensors.torch import save_file
    with open(os.path.join(ckpt_dir, "config.json")) as f:
        cfg = json.load(f)
    tc = G.text_config_from_dict(cfg)
    tc_m, L = mtp_text_config(tc)
    torch.manual_seed(seed)
    block = Glm5MtpBlock(tc_m, L).float()
    _init_block(block, seed)
    shard, added = {}, []
    for k, v in block.state_dict().items():
        for n, t in G.module_to_ckpt(k, v.detach()).items():
            name = f"{G.LM}layers.{L}.{n}"
            if G.SYN_FP8.search(name):
                q, s = G.fp8_quant(t.float())
                shard[name], shard[name + "_scale_inv"] = q.contiguous(), s.contiguous()
                added += [name, name + "_scale_inv"]
            elif G.SYN_F32.search(name):
                shard[name] = t.float().contiguous()
                added.append(name)
            else:
                shard[name] = t.to(torch.bfloat16).contiguous()
                added.append(name)
    fn = "model-mtp.safetensors"
    save_file(shard, os.path.join(ckpt_dir, fn), metadata={"format": "pt"})
    idx = os.path.join(ckpt_dir, "model.safetensors.index.json")
    with open(idx) as f:
        im = json.load(f)
    for n in added:
        im["weight_map"][n] = fn
    with open(idx, "w") as f:
        json.dump(im, f, indent=1)
    cfg.setdefault("text_config", cfg)["num_nextn_predict_layers"] = 1
    cfg["text_config"]["index_share_for_mtp_iteration"] = True
    with open(os.path.join(ckpt_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    return len(added)


def hf_trunk_dir(hf, ids, n_decode, out_dir):
    """a runner-format trunk dir from HF's full model (prompt call, then teacher-forced decode rows):
    head-mean / head-norm / head-logits of every row (hooks on the final norm) and a manifest with `head`"""
    lm = hf.model.language_model
    rec = {"mean": [], "norm": []}
    def take(m, i, o):  # returns None: a hook's return value would replace the output
        rec["mean"].append(i[0][0].detach().clone())
        rec["norm"].append(o[0].detach().clone())

    hk = lm.norm.register_forward_hook(take)
    N = len(ids)
    T = N - n_decode
    cache = LW.DynamicCache(config=hf.config)
    logits = []
    x = torch.tensor(ids)[None]
    try:
        with torch.no_grad():
            for r0, r1 in [(0, T)] + [(r, r + 1) for r in range(T, N)]:
                logits.append(hf(input_ids=x[:, r0:r1], past_key_values=cache, use_cache=True).logits[0].float())
    finally:
        hk.remove()
    os.makedirs(out_dir, exist_ok=True)
    files, head = {}, {}
    for role, t in (("mean", torch.cat(rec["mean"])), ("norm", torch.cat(rec["norm"])), ("logits", torch.cat(logits))):
        nm = f"head-{role}.f32"
        LW.write_raw(os.path.join(out_dir, nm), t.float().contiguous())
        files[nm] = {"shape": list(t.shape), "dtype": "f32", "sha256": G.sha256_file(os.path.join(out_dir, nm))}
        head[role] = nm
    man = {"runner": "glm5_mtp.hf_trunk_dir (HF full model)", "ids": [int(i) for i in ids], "T": T, "D": n_decode,
           "complete": True, "head": head, "files": files, "weights": {"index_json_sha256": None}}
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(man, f, indent=1)
    return man


def make_synthetic_mtp(wd, shapes="small", index_topk=2048, T=24, D=4, seed=SEED):
    """(fp8 checkpoint dir with an MTP layer, trunk dir from HF's full model, ids)"""
    from transformers import Glm5NextForConditionalGeneration
    fp8_dir, deq_dir, trunk = (os.path.join(wd, d) for d in ("fp8", "hf-deq", "trunk"))
    text, vision, _ = LW.make_synthetic(shapes, fp8_dir, seed, index_topk)
    LW.write_hf_dequant_checkpoint(fp8_dir, deq_dir)  # the trunk sees no MTP tensor (HF drops it anyway)
    add_synthetic_mtp(fp8_dir, seed)
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()
    transformers.utils.logging.disable_progress_bar()
    try:
        hf = Glm5NextForConditionalGeneration.from_pretrained(deq_dir, dtype=torch.float32, attn_implementation="eager",
                                                              experts_implementation="eager").eval()
    finally:
        transformers.logging.set_verbosity(verbosity)
        transformers.utils.logging.enable_progress_bar()
    g = torch.Generator().manual_seed(seed + 2)
    ids = torch.randint(1, text["vocab_size"], (T + D,), generator=g).tolist()
    hf_trunk_dir(hf, ids, D, trunk)
    del hf
    gc.collect()
    shutil.rmtree(deq_dir, ignore_errors=True)
    return fp8_dir, trunk, ids


def selftest(shapes="small", workdir=None, keep=False, log=print):
    """the proof of the oracle: on a synthetic checkpoint with an MTP layer, `run_golden` (HF's blocks,
    prompt call + decode rows against the block's cache) equals `manual_mtp` (plain torch, one call) to
    TOL at every row, head norm and logits. Returns (ok, rows)."""
    wd = workdir or tempfile.mkdtemp(prefix="glm5-mtp-selftest-")
    try:
        fp8_dir, trunk, ids = make_synthetic_mtp(wd, shapes)
        ws = G.WeightSource("fp8", fp8_dir)
        man = run_golden(ws, trunk, os.path.join(wd, "golden"), prompt_chunk=7, log=log)
        tc = G.text_config_from_dict(ws.config_dict())
        tman, _, _, _, hn, _, _ = read_trunk(trunk)
        ref_hn, ref_lg = manual_mtp(ws, tc, ids, hn)
        out = os.path.join(wd, "golden")
        got_hn = LW.load_file(out, man, "mtp-head-norm.f32")
        got_lg = LW.load_file(out, man, "mtp-logits.f32")
        rows = {"head_norm_max_abs": float((got_hn - ref_hn).abs().max()),
                "logits_max_abs": float((got_lg - ref_lg).abs().max()),
                "head_norm_rms": float(ref_hn.pow(2).mean().sqrt()), "rows": int(got_hn.shape[0])}
        ok = rows["head_norm_max_abs"] <= TOL and rows["logits_max_abs"] <= 10 * TOL
        log(f"selftest {shapes}: head-norm max|d| {rows['head_norm_max_abs']:.2e}, logits max|d| "
            f"{rows['logits_max_abs']:.2e} over {rows['rows']} rows: {'PASS' if ok else 'FAIL'}")
        return ok, rows
    finally:
        if not keep and workdir is None:
            shutil.rmtree(wd, ignore_errors=True)


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the MTP golden over real weights and a runner trunk dir")
    r.add_argument("--weights", nargs=2, metavar=("KIND", "PATH"), required=True,
                   help="container <file.cnq> | fp8-originals <dir>")
    r.add_argument("--fallback", nargs=2, metavar=("KIND", "PATH"), default=None,
                   help="where the tensors --weights lacks come from (fp8-originals <dir>)")
    r.add_argument("--trunk", required=True, help="a glm5_layerwise run dir written with --capture-head")
    r.add_argument("--out", required=True)
    r.add_argument("--prompt-chunk", type=int, default=0, help="prompt rows per call (0 = one call)")
    r.add_argument("--no-variants", action="store_true", help="run the primary pairing only")
    s = sub.add_parser("selftest", help="the proof on a synthetic checkpoint with an MTP layer")
    s.add_argument("--shapes", choices=("small", "real"), default="small")
    s.add_argument("--workdir", default=None)
    a = ap.parse_args(argv)
    if a.cmd == "selftest":
        ok, _ = selftest(a.shapes, a.workdir, keep=a.workdir is not None)
        return 0 if ok else 1
    kinds = {"fp8-originals": "fp8", "container": "cnq"}
    try:
        ws = G.open_weights(kinds[a.weights[0]], a.weights[1])
        if a.fallback:
            ws = MixedSource(ws, G.open_weights(kinds[a.fallback[0]], a.fallback[1]))
    except (G.ContainerError, KeyError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    man = run_golden(ws, a.trunk, a.out, a.prompt_chunk or None, not a.no_variants)
    print(f"wrote {len(man['files'])} files to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
