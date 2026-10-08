"""glm5_layerwise.py — crow-nest #158 (GLM-5.3-Flash plan step 7): the layerwise HF
reference runner for glm5_next (transformers 5.16.1, eager, f32, CPU).

GLM-5.3-Flash has 321 G parameters; this machine has 63 GiB of RAM. The runner holds
ONE decoder layer at a time: build it on the meta device, fill it from the weights
(FP8 originals dequantized per 128x128 block, glm5_common.WeightSource), run the prompt
rows and then the decode rows one by one against a cache of that layer only, write the
4-stream hidden state [N][4][H] to disk, free the layer, read the state back for the
next layer. Same HF modules and call sequence as Glm5NextTextModel.forward
(modeling_glm5_next.py:1477-1493), so the result is the HF full model's.

  run       python -I oracle/glm5_layerwise.py run --weights fp8-originals <dir> --ids ids.json
                --decode D --out <dir> [--layers A:B] [--anchors P,..] [--state-dtype f32|bf16]
                [--prompt-chunk C (default 512; 0 = one call)] [--delete-states-behind]
            python -I oracle/glm5_layerwise.py run --weights container <file.cnq> ...   (crow-nest #156:
                the container's weights as `converter dequant` decodes them; a partial container
                runs only the layers it holds)
  selftest  python -I oracle/glm5_layerwise.py selftest [--shapes small|real]
            the proof of the runner (abort criterion of plan step 7): a synthetic mini config,
            written as a checkpoint in the ORIGINAL naming and FP8 format, run by this runner
            and by HF's full Glm5NextForConditionalGeneration (loaded by from_pretrained from
            the same checkpoint, FP8 dequantized by HF's Fp8Dequantize). Every layer output,
            the top-8 routing, the DSA top-k and the logits must agree to 1e-5 (f32, absolute).

Output directory (all raw little-endian, row-major; shapes, dtypes and sha256 in manifest.json;
the format is documented in docs/glm5-reference-runner.md):
  embed.f32                 [N][H]          token embeddings (the input of layer 0)
  l<k>-output.f32           [N][hc][H]      decoder layer k output = the state handed to layer k+1
                                            (.bf16 instead with --state-dtype bf16)
  l<k>-routing-ids.i32      [N][top_k]      MoE layers: routed expert ids per token, ascending
  l<k>-routing-weights.f32  [N][top_k]      their weights (normalized, x routed_scaling_factor), same order
  l<k>-dsa-topk.i32         [N][W]          DSA layers: the indexer's token selection, -1 = empty,
                                            W = index_topk + index_kpool - 1
  logits-anchor-<p>.f32     [V]             logits of row p (only when the last layer was run)
Rows 0..T-1 are the prompt, rows T..N-1 the decode steps (teacher-forced, one row per call).
The prompt runs in calls of --prompt-chunk rows against the layer's cache (crow-nest #147): KDA carries
its conv and recurrent state, the DSA slot appends K/V and indexer keys in place into N preallocated
rows (_AppendIndexedLayer), and causal attention gives every row the same context as in one call.
With --delete-states-behind, l<k-1>-output.* goes once l<k>-output.* is written and recorded
(manifest `deleted_states`), so a pass holds at most two states on disk and resumes from the last.
"""
import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402

import transformers  # noqa: E402
from transformers.cache_utils import DynamicCache, DynamicIndexedLayer  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextDecoderLayer,
    Glm5NextTextRMSNorm,
)

SEED = 20261008
TOL = 1e-5
# chunked vs one prompt call (crow-nest #147): another call split is another f32 summation order (matmul
# row counts, KDA's 64-row blocks cut elsewhere), not another computation. Measured on the real-shape
# selftest 2026-10-08 (T 96 in calls of 40): <= 5.2e-5 absolute at residual RMS 2.9-5.3 (<= 1.1e-5
# relative), every routing id and DSA set identical. A cache defect is > 1e-3 (test_glm5_layerwise).
CHUNK_TOL = 1e-4
LM_HEAD_CHUNK = 16384

torch.set_num_threads(int(os.environ.get("ORACLE_THREADS", "16")))


# ---------------------------------------------------------------- state files

def write_raw(path, t):
    a = t.detach().contiguous()
    if a.dtype == torch.bfloat16:
        a.view(torch.int16).numpy().tofile(path)
    else:
        a.numpy().tofile(path)


def write_state(out_dir, k, y, state_dtype):
    ext = "bf16" if state_dtype == "bf16" else "f32"
    path = os.path.join(out_dir, f"l{k}-output.{ext}")
    write_raw(path, y.to(torch.bfloat16) if ext == "bf16" else y.float())
    return path


def read_state(path, shape):
    """the hand-over state of one layer, widened to f32"""
    if path.endswith(".bf16"):
        a = np.fromfile(path, dtype="<i2").reshape(shape)
        return torch.from_numpy(a.copy()).view(torch.bfloat16).float()
    return torch.from_numpy(np.fromfile(path, dtype="<f4").reshape(shape).copy())


class _AppendIndexedLayer(DynamicIndexedLayer):
    """The DSA slot of the layer cache (crow-nest #147): HF's DynamicIndexedLayer `torch.cat`s the whole
    expanded MLA K/V (128 KiB per token at GLM's shapes) and the indexer keys on every call
    (cache_utils.py:144-145, :350). This one allocates `capacity` rows once, on the first call, and
    writes each call's rows in place behind the previous ones. `keys` / `values` / `indexer_keys` are
    views of the rows written so far, so everything HF reads from the layer (shape[-2] as the kv length,
    get_seq_length) is what the DynamicIndexedLayer would hold, value for value."""

    def __init__(self, capacity):
        super().__init__()
        self.capacity = int(capacity)
        self._kv = None   # (K buffer, V buffer) [B][heads][capacity][d]
        self._ix = None   # indexer buffer [B][capacity][d]
        self.n_kv = 0
        self.n_ix = 0

    def update(self, key_states, value_states, *args, **kwargs):
        s = key_states.shape[-2]
        if self._kv is None:
            self.lazy_initialization(key_states, value_states)
            self._kv = tuple(t.new_empty(*t.shape[:-2], self.capacity, t.shape[-1]) for t in (key_states, value_states))
        if self.n_kv + s > self.capacity:
            raise RuntimeError(f"layer cache: {self.n_kv} + {s} rows > capacity {self.capacity}")
        self._kv[0][..., self.n_kv:self.n_kv + s, :].copy_(key_states)
        self._kv[1][..., self.n_kv:self.n_kv + s, :].copy_(value_states)
        self.n_kv += s
        self.keys = self._kv[0][..., :self.n_kv, :]
        self.values = self._kv[1][..., :self.n_kv, :]
        return self.keys, self.values

    def update_indexer(self, indexer_key_states):
        s = indexer_key_states.shape[1]
        if self._ix is None:
            self.lazy_initialization_indexer(indexer_key_states)
            t = indexer_key_states
            self._ix = t.new_empty(t.shape[0], self.capacity, *t.shape[2:])
        if self.n_ix + s > self.capacity:
            raise RuntimeError(f"indexer cache: {self.n_ix} + {s} rows > capacity {self.capacity}")
        self._ix[:, self.n_ix:self.n_ix + s].copy_(indexer_key_states)
        self.n_ix += s
        self.indexer_keys = self._ix[:, :self.n_ix]
        return self.indexer_keys


def new_layer_cache(tc):
    # one DynamicCache per layer: only slot `layer_idx` is ever touched (KDA conv +
    # recurrent state, MLA K/V, indexer packed keys); the other slots stay empty
    return DynamicCache(config=tc)


def append_in_place(cache, tc, n_rows):
    """the DSA slots of `cache` append in place into n_rows preallocated rows (_AppendIndexedLayer)
    instead of torch.cat over the history per call; the KDA slots hold fixed-size conv / recurrent
    states and need no change"""
    if cache is not None:
        for i, t in enumerate(tc.layer_types):
            if t == "deepseek_sparse_attention" and i < len(cache.layers):
                cache.layers[i] = _AppendIndexedLayer(n_rows)
    return cache


def call_plan(N, T, prompt_chunk):
    """the calls of one layer: prompt rows 0..T-1 in calls of `prompt_chunk` rows (0 / None = one call),
    then rows T..N-1 one by one (the decode rows)"""
    c = int(prompt_chunk or 0) or T
    return [(r0, min(r0 + c, T)) for r0 in range(0, T, c)] + [(r, r + 1) for r in range(T, N)]


# ---------------------------------------------------------------- one layer

def _watch_index_ties(indexer, rec):
    """Record the rows whose DSA selection boundary is an exact tie: the select_k-th and the next
    pool score are equal (and valid). torch.topk breaks exact ties in an order that depends on the
    tensor's shape, so on such a row the selected SET can differ between call splits (HF's own as
    well); on every other row it cannot. The indexer's ReLU makes all-zero pool scores possible
    (modeling_glm5_next.py, Glm5NextTextIndexer.forward). Only index_scores.topk runs inside the
    indexer's forward, so Tensor.topk is wrapped for that call only."""
    rec["ties"], rec["r0"] = [], 0
    orig = type(indexer).forward

    def forward(*a, **kw):
        real_topk = torch.Tensor.topk

        def topk(t, k, *ta, **tk):
            if t.dim() == 3 and k < t.shape[-1]:
                v = real_topk(t, k + 1, dim=-1).values  # descending
                tie = (v[..., k - 1] == v[..., k]) & (v[..., k - 1] > torch.finfo(t.dtype).min)
                rec["ties"] += (tie[0].nonzero().flatten() + rec["r0"]).tolist()
            return real_topk(t, k, *ta, **tk)

        torch.Tensor.topk = topk
        try:
            return orig(indexer, *a, **kw)
        finally:
            torch.Tensor.topk = real_topk

    indexer.forward = forward


def run_layer(layer, tc, l, x, T, prompt_chunk=None, timings=None, stats=None):
    """x: [N][hc][H] f32. Prompt rows 0..T-1 in calls of `prompt_chunk` rows against this layer's cache
    (None / 0 = one call), then rows T..N-1 one by one. Causal attention, so a row sees rows 0..itself
    in every split. `timings`, a list, gets the seconds of each prompt call; `stats`, a dict, gets
    "dsa_tie_rows" (DSA layers: rows whose selection boundary is an exact tie, _watch_index_ties).
    Returns (y [N][hc][H], routing (ids, weights) or None, dsa topk [N][W] or None)."""
    N = x.shape[0]
    rec = {"route": [], "topk": []}
    hooks = []
    if tc.mlp_layer_types[l] == "sparse":
        hooks.append(layer.mlp.gate.register_forward_hook(
            lambda m, i, o: rec["route"].append((o[2].detach().clone(), o[1].detach().clone()))))
    if tc.layer_types[l] == "deepseek_sparse_attention":
        if tc.indexer_types[l] != "full":
            raise NotImplementedError(
                f"layer {l}: indexer_types '{tc.indexer_types[l]}' (shared top-k) is not supported; the config of "
                "rev eb9eb208 has 'full' on all 45 layers")
        hooks.append(layer.self_attn.indexer.register_forward_hook(
            lambda m, i, o: rec["topk"].append(o.detach().clone())))
        _watch_index_ties(layer.self_attn.indexer, rec)
    cache = append_in_place(new_layer_cache(tc), tc, N)
    y = torch.empty_like(x)
    try:
        with torch.no_grad():
            for r0, r1 in call_plan(N, T, prompt_chunk):
                tc0 = time.perf_counter()
                rec["r0"] = r0
                out, _ = layer(
                    x[None, r0:r1].contiguous(),
                    attention_mask=torch.ones(1, r1 - r0, dtype=torch.bool),
                    position_ids=torch.arange(r0, r1)[None],
                    past_key_values=cache,
                    use_cache=True,
                    position_embeddings=None,
                    prev_topk_indices=None,
                )
                y[r0:r1] = out[0]
                del out
                if timings is not None and r1 <= T:
                    timings.append(time.perf_counter() - tc0)
    finally:
        for h in hooks:
            h.remove()
        if "ties" in rec:
            del layer.self_attn.indexer.forward  # back to the class method
    del cache
    if stats is not None and "ties" in rec:
        stats["dsa_tie_rows"] = rec["ties"]
    routing = None
    if rec["route"]:
        ids = torch.cat([r[0] for r in rec["route"]], 0)
        w = torch.cat([r[1] for r in rec["route"]], 0)
        order = ids.argsort(dim=-1)
        routing = (ids.gather(1, order).to(torch.int32), w.gather(1, order).float())
    topk = torch.cat([t[0] for t in rec["topk"]], 0).to(torch.int32) if rec["topk"] else None
    return y, routing, topk


def layer_kind(tc, l):
    a = "dsa" if tc.layer_types[l] == "deepseek_sparse_attention" else "kda"
    return f"{a}+{'moe' if tc.mlp_layer_types[l] == 'sparse' else 'dense'}"


# ---------------------------------------------------------------- the runner

def _rss():
    try:
        import psutil
        mi = psutil.Process().memory_info()
        return round(mi.rss / 2**30, 3), round(getattr(mi, "peak_wset", 0) / 2**30, 3) or None
    except ImportError:
        return None, None


def run_layerwise(ws, tc, ids, n_decode, out_dir, start=0, stop=None, anchors=None, state_dtype="f32",
                  log=print, extra=None, prompt_chunk=None, delete_states_behind=False):
    """ids: list[int] of N tokens, the last n_decode run as decode rows. Writes the files of the
    module docstring into out_dir and returns the manifest dict.
    prompt_chunk: prompt rows per call (None / 0 = all T rows in one call).
    delete_states_behind: once layer k's state is on disk and recorded, delete layer k-1's state (and the
    last layer's after the pass); routing, DSA top-k, embed, logits and the manifest stay."""
    assert state_dtype in ("f32", "bf16"), state_dtype
    L, H, hc = tc.num_hidden_layers, tc.hidden_size, tc.hc_mult
    stop = L if stop is None else stop
    N = len(ids)
    T = N - n_decode
    # start == stop == L: no layer, only the logits from l<L-1>-output (a pass that stopped after its last layer)
    assert T >= 1 and 0 <= start <= stop <= L and (start < stop or stop == L), (T, start, stop, L)
    anchors = sorted(set(anchors if anchors is not None else [T - 1] + list(range(T, N))))
    assert all(0 <= p < N for p in anchors), anchors
    os.makedirs(out_dir, exist_ok=True)
    files = {}
    man = {
        "runner": "oracle/glm5_layerwise.py (crow-nest #158)",
        "torch": torch.__version__, "transformers": transformers.__version__,
        "threads": torch.get_num_threads(), "attn_implementation": tc._attn_implementation,
        "experts_implementation": tc._experts_implementation,
        "weights": ws.provenance(), "ids": [int(i) for i in ids], "T": T, "D": n_decode,
        "layers": [start, stop], "num_hidden_layers": L, "state_dtype": state_dtype,
        "anchors": anchors if stop == L else [], "hidden_size": H, "hc_mult": hc,
        "layer_kinds": [layer_kind(tc, l) for l in range(L)], "files": files, "per_layer": [],
        "prompt_chunk": int(prompt_chunk or 0), "delete_states_behind": bool(delete_states_behind),
        "deleted_states": [], "complete": False,
    }
    if extra:
        man.update(extra)

    def record(name, shape, dtype):
        files[name] = {"shape": list(shape), "dtype": dtype,
                       "sha256": G.sha256_file(os.path.join(out_dir, name))}

    def save_manifest():
        with open(os.path.join(out_dir, "manifest.json"), "w") as f:
            json.dump(man, f, indent=1)

    if start == 0:
        e = ws.embed(ids)
        write_raw(os.path.join(out_dir, "embed.f32"), e)
        record("embed.f32", e.shape, "f32")
        x = e.unsqueeze(1).expand(-1, hc, -1).contiguous()  # modeling :1477
        del e
    else:
        prev = [f for f in (f"l{start - 1}-output.f32", f"l{start - 1}-output.bf16")
                if os.path.exists(os.path.join(out_dir, f))]
        assert prev, f"--layers {start}:{stop}: l{start - 1}-output.* is not in {out_dir}"
        x = read_state(os.path.join(out_dir, prev[0]), (N, hc, H))
        mp = os.path.join(out_dir, "manifest.json")
        if os.path.exists(mp):  # a split run: keep the records of the earlier layers
            with open(mp) as f:
                old = json.load(f)
            for key in ("ids", "T", "D", "state_dtype", "num_hidden_layers"):
                assert old[key] == man[key], f"--layers {start}:{stop}: {key} differs from {mp}"
            assert old["weights"]["index_json_sha256"] == man["weights"]["index_json_sha256"],                 f"--layers {start}:{stop}: other weights than {mp}"
            files.update({k: v for k, v in old["files"].items() if not k.startswith("logits-")})
            man["per_layer"] = [r for r in old["per_layer"] if r["layer"] < start]
            man["layers"] = [old["layers"][0], stop]
            man["deleted_states"] = [d for d in old.get("deleted_states", [])
                                     if int(d[1:].split("-")[0]) < start - 1]
    save_manifest()

    def delete_state(k):
        for ext in ("f32", "bf16"):
            nm = f"l{k}-output.{ext}"
            p = os.path.join(out_dir, nm)
            if os.path.exists(p):
                os.remove(p)
                files.pop(nm, None)
                man["deleted_states"].append(nm)
        save_manifest()

    for l in range(start, stop):
        t0 = time.time()
        layer = G.build_meta(Glm5NextTextDecoderLayer, tc, l)
        ws.load(layer, f"{G.LM}layers.{l}.")
        t1 = time.time()
        rss_load, _ = _rss()
        call_s, stats = [], {}
        y, routing, topk = run_layer(layer, tc, l, x, T, prompt_chunk, call_s, stats)
        t2 = time.time()
        del layer
        gc.collect()
        path = write_state(out_dir, l, y, state_dtype)
        record(os.path.basename(path), y.shape, state_dtype)
        if routing is not None:
            for nm, t, dt in ((f"l{l}-routing-ids.i32", routing[0], "i32"),
                              (f"l{l}-routing-weights.f32", routing[1], "f32")):
                write_raw(os.path.join(out_dir, nm), t)
                record(nm, t.shape, dt)
        if topk is not None:
            nm = f"l{l}-dsa-topk.i32"
            write_raw(os.path.join(out_dir, nm), topk)
            record(nm, topk.shape, "i32")
        x = read_state(path, (N, hc, H))  # the next layer sees what is on disk
        del y
        _, peak = _rss()
        man["per_layer"].append({"layer": l, "kind": layer_kind(tc, l), "load_s": round(t1 - t0, 3),
                                 "compute_s": round(t2 - t1, 3), "rss_after_load_gib": rss_load,
                                 "peak_wset_gib": peak, "prompt_chunk": int(prompt_chunk or 0),
                                 "prompt_call_s": [round(s, 3) for s in call_s]})
        if "dsa_tie_rows" in stats:
            ties = stats["dsa_tie_rows"]
            man["per_layer"][-1].update({"dsa_tie_rows_n": len(ties), "dsa_tie_rows": ties[:1000]})
        save_manifest()  # layer l is recorded before the state behind it goes
        if delete_states_behind and l > 0:
            delete_state(l - 1)
        log(f"layer {l:2d} {layer_kind(tc, l):9s} load {t1 - t0:7.2f} s  compute {t2 - t1:7.2f} s"
            + (f"  ({len(call_s)} prompt calls)" if len(call_s) > 1 else "")
            + (f"  rss {rss_load:.2f} GiB" if rss_load else ""))

    if stop == L and anchors:
        norm = G.build_meta(Glm5NextTextRMSNorm, H, tc.rms_norm_eps)
        ws.load(norm, f"{G.LM}norm.")
        with torch.no_grad():
            h = norm(x[anchors].mean(dim=1))  # hc_head = mean over the streams, then norm (:1493, :298-302)
            V = ws.n_rows("lm_head.weight")
            logits = torch.empty(len(anchors), V)
            for r0 in range(0, V, LM_HEAD_CHUNK):
                r1 = min(V, r0 + LM_HEAD_CHUNK)
                logits[:, r0:r1] = h @ ws.rows("lm_head.weight", r0, r1).T
        for i, p in enumerate(anchors):
            nm = f"logits-anchor-{p}.f32"
            write_raw(os.path.join(out_dir, nm), logits[i])
            record(nm, (V,), "f32")
    if delete_states_behind and stop == L:
        delete_state(L - 1)  # nothing reads the last state after the logits
    man["complete"] = True
    save_manifest()
    return man


def load_file(out_dir, man, name):
    f = man["files"][name]
    path = os.path.join(out_dir, name)
    if f["dtype"] == "bf16":
        return read_state(path, f["shape"])
    dt = {"f32": "<f4", "i32": "<i4"}[f["dtype"]]
    return torch.from_numpy(np.fromfile(path, dtype=dt).reshape(f["shape"]).copy())


# ---------------------------------------------------------------- the proof (selftest)

def mini_config(shapes, index_topk):
    """(text, vision) config dicts of the synthetic mini model. 8 layers = 3 dense KDA, then the real
    schedule (DSA at 3 and 7, KDA elsewhere, MoE from layer 3), 16 routed experts, top-8 kept.
    `real`: the per-block shapes of GLM-5.3-Flash (hidden 4096, 64 heads x 256, kv_lora 512, q_lora 1536,
    KDA 64 x 128, conv 4, moe_intermediate 2048, intermediate 12288, vocab 154880, indexer 32 x 128,
    kpool 4). `small`: the same structure at small widths (FP8 tensors still span several 128x128 blocks)."""
    common = dict(num_hidden_layers=8, n_routed_experts=16, num_experts_per_tok=8, n_shared_experts=1,
                  n_group=1, topk_group=1, index_topk=index_topk, index_kpool=4, hc_mult=4,
                  hc_sinkhorn_iters=20, linear_conv_kernel_dim=4, rms_norm_eps=1e-5, qk_rope_head_dim=0,
                  pad_token_id=0, tie_word_embeddings=False)
    if shapes == "real":
        text = dict(common, vocab_size=154880, hidden_size=4096, intermediate_size=12288,
                    moe_intermediate_size=2048, num_attention_heads=64, num_key_value_heads=64,
                    kv_lora_rank=512, q_lora_rank=1536, qk_nope_head_dim=256, v_head_dim=256,
                    index_head_dim=128, index_n_heads=32, linear_head_dim=128, linear_num_heads=64)
    else:
        assert shapes == "small", shapes
        text = dict(common, vocab_size=1000, hidden_size=256, intermediate_size=384,
                    moe_intermediate_size=128, num_attention_heads=4, num_key_value_heads=4,
                    kv_lora_rank=128, q_lora_rank=256, qk_nope_head_dim=64, v_head_dim=64,
                    index_head_dim=64, index_n_heads=4, linear_head_dim=64, linear_num_heads=4)
    vision = dict(depth=1, hidden_size=32, num_heads=2, intermediate_size=64,
                  out_hidden_size=text["hidden_size"], projection_intermediate_size=64)
    return text, vision


def _config_dict(text, vision):
    return {"model_type": "glm5_next", "architectures": ["Glm5NextForConditionalGeneration"],
            "text_config": text, "vision_config": vision, "tie_word_embeddings": False,
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128],
                                    "activation_scheme": "dynamic"}}


def _perturb(model, seed):
    """HF init leaves norms at 1, A_log / e_score_correction_bias / kpool ape at 0 and the kpool gate at 1:
    a test that cannot see a swapped or dropped tensor. Give every such tensor its own values."""
    g = torch.Generator().manual_seed(seed + 1)
    with torch.no_grad():
        # std 1 instead of 0.02: the 1e-5 bound is absolute, so a larger signal makes it stricter
        emb = model.model.language_model.embed_tokens.weight
        emb.copy_(torch.randn(emb.shape, generator=g))
        for n, p in model.model.language_model.named_parameters():
            if n.endswith("index_kpool_compress_ape"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.5)
            elif n.endswith("index_kpool_compress_gate"):
                p.copy_(torch.randn(p.shape, generator=g) * 0.02)
            elif p.dim() == 1:
                p.add_(torch.randn(p.shape, generator=g) * 0.1)
        for n, b in model.model.language_model.named_buffers():
            if n.endswith("e_score_correction_bias"):
                b.copy_(torch.randn(b.shape, generator=g) * 0.05)


def write_hf_dequant_checkpoint(src, dst):
    """the HF side of the proof: the same checkpoint, original naming, every FP8 tensor
    dequantized by HF's own Fp8Dequantize, stored f32 (one output shard per input shard)"""
    from safetensors import safe_open
    from safetensors.torch import save_file
    from transformers.integrations.finegrained_fp8 import Fp8Dequantize
    os.makedirs(dst, exist_ok=True)
    with open(os.path.join(src, "model.safetensors.index.json")) as f:
        wm = json.load(f)["weight_map"]
    dq = Fp8Dequantize(None)
    out_map = {}
    for shard in sorted(set(wm.values())):
        out = {}
        for n in [n for n, s in wm.items() if s == shard and not n.endswith("_scale_inv")]:
            with safe_open(os.path.join(src, shard), "pt") as f:
                w = f.get_tensor(n)
            if n + "_scale_inv" in wm:
                with safe_open(os.path.join(src, wm[n + "_scale_inv"]), "pt") as f:
                    s = f.get_tensor(n + "_scale_inv")
                out[n] = dq._dequantize_one(w, s, torch.float32).contiguous()
            else:
                out[n] = w.float().contiguous()
            out_map[n] = shard
        save_file(out, os.path.join(dst, shard), metadata={"format": "pt"})
        del out
        gc.collect()
    with open(os.path.join(dst, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {}, "weight_map": out_map}, f)
    with open(os.path.join(src, "config.json")) as f:
        cfg = json.load(f)
    cfg.pop("quantization_config", None)
    with open(os.path.join(dst, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)


def hf_reference(model, ids, n_decode):
    """HF's full model, prompt then teacher-forced decode against one DynamicCache; hooks record
    every decoder layer output, router top-k and indexer selection per row"""
    lm = model.model.language_model
    tc = model.config.text_config
    rec = {"y": {}, "route": {}, "topk": {}}
    hooks = []
    for l, layer in enumerate(lm.layers):
        rec["y"][l], rec["route"][l], rec["topk"][l] = [], [], []
        hooks.append(layer.register_forward_hook(lambda m, i, o, l=l: rec["y"][l].append(o[0][0].detach().clone())))
        if tc.mlp_layer_types[l] == "sparse":
            hooks.append(layer.mlp.gate.register_forward_hook(
                lambda m, i, o, l=l: rec["route"][l].append((o[2].detach().clone(), o[1].detach().clone()))))
        if tc.layer_types[l] == "deepseek_sparse_attention":
            hooks.append(layer.self_attn.indexer.register_forward_hook(
                lambda m, i, o, l=l: rec["topk"][l].append(o[0].detach().clone())))
    N = len(ids)
    T = N - n_decode
    x = torch.tensor(ids)[None]
    cache = DynamicCache(config=model.config)
    logits = []
    try:
        with torch.no_grad():
            for r0, r1 in [(0, T)] + [(r, r + 1) for r in range(T, N)]:
                out = model(input_ids=x[:, r0:r1], past_key_values=cache, use_cache=True)
                logits.append(out.logits[0].float())
    finally:
        for h in hooks:
            h.remove()
    ref = {"y": {}, "route": {}, "topk": {}, "logits": torch.cat(logits, 0)}
    for l in rec["y"]:
        ref["y"][l] = torch.cat(rec["y"][l], 0)
        if rec["route"][l]:
            ids_ = torch.cat([r[0] for r in rec["route"][l]], 0)
            w = torch.cat([r[1] for r in rec["route"][l]], 0)
            order = ids_.argsort(dim=-1)
            ref["route"][l] = (ids_.gather(1, order).to(torch.int32), w.gather(1, order).float())
        if rec["topk"][l]:
            ref["topk"][l] = torch.cat(rec["topk"][l], 0).to(torch.int32)
    return ref


def canon_topk(t):
    """DSA top-k rows as sets: valid ids ascending, -1 behind. The indexer's row LAYOUT depends on the
    call split (select_k = min(index_topk / kpool, pools in the cache), the tail behind it), the set
    of selected tokens does not, and the attention mask is built from the set (scatter)."""
    big = torch.iinfo(torch.int32).max
    s = torch.where(t < 0, torch.full_like(t, big), t).sort(dim=1).values
    return torch.where(s == big, torch.full_like(s, -1), s)


def compare(ref, out_dir, man, tc, T, raw_layout=True, tol=TOL):
    """one row per layer + one logits row; ok = every number within `tol` and every id identical.
    DSA top-k: the selected set per row must be identical; with raw_layout also the row layout
    (only meaningful when both sides split the prompt into the same calls)."""
    rows, ok = [], True
    for l in range(tc.num_hidden_layers):
        y = load_file(out_dir, man, f"l{l}-output.f32")
        d = (y - ref["y"][l]).abs()
        row = {"layer": l, "kind": layer_kind(tc, l), "max_abs_prompt": float(d[:T].max()),
               "max_abs_decode": float(d[T:].max()) if d.shape[0] > T else 0.0,
               "ref_rms": float(ref["y"][l].pow(2).mean().sqrt()), "finite": bool(torch.isfinite(y).all())}
        if l in ref["route"]:
            ids_ = load_file(out_dir, man, f"l{l}-routing-ids.i32")
            w = load_file(out_dir, man, f"l{l}-routing-weights.f32")
            row["route_ids_mismatch_tokens"] = int((ids_ != ref["route"][l][0]).any(dim=1).sum())
            row["route_w_max_abs"] = float((w - ref["route"][l][1]).abs().max())
        if l in ref["topk"]:
            tk = load_file(out_dir, man, f"l{l}-dsa-topk.i32")
            same_shape = tuple(tk.shape) == tuple(ref["topk"][l].shape)
            row["dsa_topk_mismatch_rows"] = (int((canon_topk(tk) != canon_topk(ref["topk"][l])).any(dim=1).sum())
                                             if same_shape else -1)
            if raw_layout:
                row["dsa_topk_layout_mismatch_rows"] = (int((tk != ref["topk"][l]).any(dim=1).sum())
                                                        if same_shape else -1)
            row["dsa_selected_per_row"] = float((ref["topk"][l] >= 0).sum(1).float().mean())
            pl = [p for p in man.get("per_layer", []) if p["layer"] == l]
            row["dsa_tie_rows_n"] = pl[0].get("dsa_tie_rows_n", 0) if pl else 0
        row["ok"] = (row["finite"] and row["max_abs_prompt"] <= tol and row["max_abs_decode"] <= tol
                     and row.get("route_ids_mismatch_tokens", 0) == 0 and row.get("route_w_max_abs", 0) <= tol
                     and row.get("dsa_topk_mismatch_rows", 0) == 0
                     and row.get("dsa_topk_layout_mismatch_rows", 0) == 0)
        ok &= row["ok"]
        rows.append(row)
    dl = 0.0
    for p in man["anchors"]:
        lg = load_file(out_dir, man, f"logits-anchor-{p}.f32")
        dl = max(dl, float((lg - ref["logits"][p]).abs().max()))
    lrow = {"layer": "logits", "max_abs": dl, "anchors": len(man["anchors"]), "ok": dl <= tol and len(man["anchors"]) > 0}
    ok &= lrow["ok"]
    rows.append(lrow)
    return rows, ok


def print_table(rows, index_topk, log=print):
    log(f"index_topk {index_topk}")
    log(f"{'layer':>6} {'kind':9} {'max|d| prompt':>13} {'max|d| decode':>13} {'ref rms':>9} "
        f"{'route ids':>9} {'route w':>9} {'dsa topk':>8} {'sel/row':>7} {'ties':>4}  ok")
    for r in rows:
        if r["layer"] == "logits":
            log(f"{'logits':>6} {'':9} {r['max_abs']:13.3e} {'':13} {'':9} {'':9} {'':9} {'':8} {'':7} {'':4}  "
                f"{'yes' if r['ok'] else 'NO'}  ({r['anchors']} anchors)")
            continue
        rid = "" if "route_ids_mismatch_tokens" not in r else f"{r['route_ids_mismatch_tokens']} bad"
        rw = "" if "route_w_max_abs" not in r else f"{r['route_w_max_abs']:.1e}"
        tk = "" if "dsa_topk_mismatch_rows" not in r else f"{r['dsa_topk_mismatch_rows']} bad"
        sel = "" if "dsa_selected_per_row" not in r else f"{r['dsa_selected_per_row']:.1f}"
        ties = "" if "dsa_tie_rows_n" not in r else str(r["dsa_tie_rows_n"])
        log(f"{r['layer']:>6} {r['kind']:9} {r['max_abs_prompt']:13.3e} {r['max_abs_decode']:13.3e} "
            f"{r['ref_rms']:9.3e} {rid:>9} {rw:>9} {tk:>8} {sel:>7} {ties:>4}  {'yes' if r['ok'] else 'NO'}")


def make_synthetic(shapes, fp8_dir, seed=SEED, index_topk=32):
    """the synthetic mini model, seeded, written as an FP8 checkpoint in the original naming;
    returns (text config dict, vision config dict, number of tensors)"""
    from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
    text, vision = mini_config(shapes, index_topk)
    torch.manual_seed(seed)
    model = Glm5NextForConditionalGeneration(Glm5NextConfig(text_config=text, vision_config=vision)).float().eval()
    _perturb(model, seed)
    n = G.write_synthetic_checkpoint(model, fp8_dir, _config_dict(text, vision))
    del model
    gc.collect()
    return text, vision, n


def ref_from_run(out_dir, man, tc):
    """a runner output dir as a reference for compare() (the chunked-vs-unchunked check)"""
    ref = {"y": {}, "route": {}, "topk": {}, "logits": {}}
    for l in range(tc.num_hidden_layers):
        ref["y"][l] = load_file(out_dir, man, f"l{l}-output.f32")
        if f"l{l}-routing-ids.i32" in man["files"]:
            ref["route"][l] = (load_file(out_dir, man, f"l{l}-routing-ids.i32"),
                               load_file(out_dir, man, f"l{l}-routing-weights.f32"))
        if f"l{l}-dsa-topk.i32" in man["files"]:
            ref["topk"][l] = load_file(out_dir, man, f"l{l}-dsa-topk.i32")
    for p in man["anchors"]:
        ref["logits"][p] = load_file(out_dir, man, f"logits-anchor-{p}.f32")
    return ref


def selftest(shapes="small", T=96, D=4, topks=(32, 2048), seed=SEED, workdir=None, keep=False, log=print,
             hf_experts="eager", prompt_chunk=None):
    """the proof of the runner; returns (ok, {index_topk: rows}). The HF side runs eager attention
    (the runner's path; from_pretrained would pick sdpa) and `hf_experts` ("eager" = the runner's
    per-expert loop, "grouped_mm" = HF's default kernel when it can dispatch).
    prompt_chunk C: the runner also runs the prompt in calls of C rows (crow-nest #147); that run is
    compared with HF's full model (one prompt call) and with the runner's own one-call run, under the
    rule CHUNK_TOL per value, routing ids identical, DSA top-k identical as a set per row; the extra
    rows are in tables["<k> chunked vs HF"] and tables["<k> chunked vs unchunked"]."""
    from transformers import Glm5NextForConditionalGeneration
    wd = workdir or tempfile.mkdtemp(prefix="glm5-selftest-")
    os.makedirs(wd, exist_ok=True)
    t0 = time.time()
    fp8_dir, deq_dir = os.path.join(wd, "fp8"), os.path.join(wd, "hf-deq")
    text, vision, n = make_synthetic(shapes, fp8_dir, seed, topks[0])
    write_hf_dequant_checkpoint(fp8_dir, deq_dir)
    log(f"synthetic checkpoint: {n} tensors in the original naming ({shapes} shapes), {time.time() - t0:.1f} s")
    verbosity = transformers.logging.get_verbosity()
    transformers.logging.set_verbosity_error()  # the load report lists the vision tower, checked below
    transformers.utils.logging.disable_progress_bar()
    try:
        hf, info = Glm5NextForConditionalGeneration.from_pretrained(
            deq_dir, dtype=torch.float32, output_loading_info=True, attn_implementation="eager",
            experts_implementation=hf_experts)
    finally:
        transformers.logging.set_verbosity(verbosity)
        transformers.utils.logging.enable_progress_bar()
    hf.eval()
    log(f"HF full model: attn {hf.config.text_config._attn_implementation}, "
        f"experts {hf.config.text_config._experts_implementation}; runner: attn eager, experts eager")
    bad = [k for k in info["missing_keys"] if not k.startswith("model.visual.")] + list(info["unexpected_keys"]) \
        + list(info["mismatched_keys"])
    assert not bad, f"HF loader: {bad[:8]}"
    shutil.rmtree(deq_dir, ignore_errors=True)
    g = torch.Generator().manual_seed(seed + 2)
    ids = torch.randint(1, text["vocab_size"], (T + D,), generator=g).tolist()
    ws = G.WeightSource("fp8", fp8_dir)
    ok_all, tables = True, {}
    try:
        refs, hf_s = {}, {}
        for k in topks:  # every HF reference first, then the HF model goes before the runner passes
            for m in hf.modules():
                if type(m).__name__ == "Glm5NextTextIndexer":
                    m.index_topk = k
            hf.config.text_config.index_topk = k
            t1 = time.time()
            refs[k] = hf_reference(hf, ids, D)
            hf_s[k] = time.time() - t1
        del hf
        gc.collect()
        for k in topks:
            ref = refs.pop(k)
            cfg = _config_dict(dict(text, index_topk=k), vision)
            tc = G.text_config_from_dict(cfg)
            sel = {"selftest": {"shapes": shapes, "seed": seed, "index_topk": k}}
            t2 = time.time()
            out_dir = os.path.join(wd, f"run-topk{k}")
            man = run_layerwise(ws, tc, ids, D, out_dir, anchors=list(range(T + D)), log=lambda *_: None,
                                extra=sel)
            t3 = time.time()
            rows, ok = compare(ref, out_dir, man, tc, T)
            print_table(rows, k, log)
            log(f"  T {T} prompt (one call) + D {D} decode rows; HF full model {hf_s[k]:.1f} s, "
                f"layerwise {t3 - t2:.1f} s")
            ok_all &= ok
            tables[k] = rows
            if prompt_chunk:
                ch_dir = os.path.join(wd, f"run-topk{k}-chunk{prompt_chunk}")
                ch = run_layerwise(ws, tc, ids, D, ch_dir, anchors=list(range(T + D)), log=lambda *_: None,
                                   extra=sel, prompt_chunk=prompt_chunk)
                t4 = time.time()
                for label, r, raw in ((f"{k} chunked vs HF", ref, False),
                                      (f"{k} chunked vs unchunked", ref_from_run(out_dir, man, tc), False)):
                    rows, ok = compare(r, ch_dir, ch, tc, T, raw_layout=raw, tol=CHUNK_TOL)
                    print_table(rows, f"{k}, runner with --prompt-chunk {prompt_chunk}: {label.split(' ', 1)[1]}",
                                log)
                    ok_all &= ok
                    tables[label] = rows
                calls = [len(p["prompt_call_s"]) for p in ch["per_layer"]]
                log(f"  T {T} prompt in {calls[0]} calls of <= {prompt_chunk} rows + D {D} decode rows; "
                    f"layerwise {t4 - t3:.1f} s")
            del ref
            gc.collect()
    finally:
        if not keep and workdir is None:
            shutil.rmtree(wd, ignore_errors=True)
    log(f"selftest {shapes}: {'PASS' if ok_all else 'FAIL'} (tolerance {TOL:g} absolute, f32"
        + (f"; chunked vs one call {CHUNK_TOL:g}, every routing id and DSA set identical" if prompt_chunk else "")
        + f"), {time.time() - t0:.1f} s")
    return ok_all, tables


# ---------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="the layerwise reference over real weights")
    r.add_argument("--weights", nargs=2, metavar=("KIND", "PATH"), required=True,
                   help="fp8-originals <dir> | container <file.cnq>")
    r.add_argument("--ids", required=True, help="JSON list of token ids (prompt + decode rows)")
    r.add_argument("--decode", type=int, default=0, help="the last D ids run as decode rows")
    r.add_argument("--out", required=True)
    r.add_argument("--layers", default=None, help="A:B runs layers A..B-1 (input: l<A-1>-output.* in --out); L: (L = last layer + 1) only the logits")
    r.add_argument("--anchors", default=None, help="comma list of rows for logits (default: last prompt row + decode rows)")
    r.add_argument("--state-dtype", choices=("f32", "bf16"), default="f32")
    r.add_argument("--prompt-chunk", type=int, default=512,
                   help="prompt rows per call against the layer cache (0 = all prompt rows in one call)")
    r.add_argument("--delete-states-behind", action="store_true",
                   help="delete layer k-1's hand-over state once layer k's is written (and the last one after "
                        "the pass); routing, DSA top-k, logits and the manifest stay")
    s = sub.add_parser("selftest", help="the proof of the runner on a synthetic mini config")
    s.add_argument("--shapes", choices=("small", "real"), default="small")
    s.add_argument("--T", type=int, default=96)
    s.add_argument("--D", type=int, default=4)
    s.add_argument("--topks", default="32,2048")
    s.add_argument("--workdir", default=None, help="keep the synthetic checkpoint and runs here")
    s.add_argument("--hf-experts", choices=("eager", "grouped_mm"), default="eager",
                   help="experts kernel of the HF full model (the runner always runs eager)")
    s.add_argument("--prompt-chunk", type=int, default=0,
                   help="also run the runner with the prompt in calls of this many rows and compare it with HF "
                        "and with the one-call run (0 = off)")
    a = ap.parse_args(argv)

    if a.cmd == "selftest":
        ok, _ = selftest(a.shapes, a.T, a.D, tuple(int(k) for k in a.topks.split(",")), workdir=a.workdir,
                         keep=a.workdir is not None, hf_experts=a.hf_experts, prompt_chunk=a.prompt_chunk or None)
        return 0 if ok else 1
    if a.prompt_chunk < 0:
        ap.error("--prompt-chunk must be >= 0")

    kind, path = a.weights
    kinds = {"fp8-originals": "fp8", "container": "cnq"}
    if kind not in kinds:
        ap.error(f"--weights {kind}: expected one of {', '.join(kinds)}")
    try:
        ws = G.open_weights(kinds[kind], path)
    except G.ContainerError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    tc = G.text_config_from_dict(ws.config_dict())
    with open(a.ids) as f:
        ids = json.load(f)
    start, stop = 0, tc.num_hidden_layers
    if a.layers:
        s0, s1 = a.layers.split(":")
        start, stop = int(s0 or 0), int(s1 or tc.num_hidden_layers)
    part = getattr(ws, "partial", None)
    if part and not set(range(start, stop)) <= set(part["layers"]):
        print(f"error: --layers {start}:{stop}: {path} is a PARTIAL container ({part['filter']}); "
              f"it holds layers {part['layers']} only", file=sys.stderr)
        return 2
    anchors = [int(p) for p in a.anchors.split(",")] if a.anchors else None
    man = run_layerwise(ws, tc, ids, a.decode, a.out, start, stop, anchors, a.state_dtype,
                        prompt_chunk=a.prompt_chunk, delete_states_behind=a.delete_states_behind)
    print(f"wrote {len(man['files'])} files to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
