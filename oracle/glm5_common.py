"""glm5_common.py — crow-nest #158 (GLM-5.3-Flash plan step 7): shared plumbing for
the layerwise HF reference of glm5_next (transformers 5.16.1,
models/glm5_next/modeling_glm5_next.py).

One weight source, module-side names on the way out, CHECKPOINT names on disk:

  fp8   the FP8 originals of zai-org/GLM-5.3-Flash (safetensors shards +
        model.safetensors.index.json + config.json in one directory). A tensor
        `X.weight` with a sibling `X.weight_scale_inv` is E4M3 with one f32 scale
        per 128x128 block (quantization_config.weight_block_size); every other
        tensor (BF16 / F32) is widened to f32 exactly.
  cnq   a CNQ container of glm5_next written by the converter (crow-nest #156), also a
        partial one (`--layers 0-3 --with-embed-head`). ContainerSource reads every
        weight through `converter dequant`: the converter's own decode, the one gate 0
        (the sidecar) measures the written encoding with, so the reference sees exactly
        what the converter wrote. Checkpoint names are kept in the container, so the
        name table below serves both back ends. `open_weights(kind, path)` picks one.

Checkpoint names differ from module names (conversion_mapping.py "glm5_next"):
  self_attn.{f_a_proj,f_b_proj}.weight, self_attn.{dt_bias,A_log}  -> self_attn.forget_gate.*
  hc_{attn,ffn}_{fn,base,scale}                                     -> {attn,ffn}_hc.{fn,base,scale}
  mlp.experts.<e>.{gate,up}_proj.weight   stack(e) then cat(dim 1)  -> mlp.experts.gate_up_proj [E, 2I, H]
  mlp.experts.<e>.down_proj.weight        stack(e)                  -> mlp.experts.down_proj    [E, H, I]
  self_attn.{q,k,v}_conv1d.weight         cat(dim 0)                -> self_attn.conv1d.weight  [3*qkv, 1, K]
`module_to_ckpt` (writer side) and `WeightSource.load` (reader side) are the two
directions of that table; test_glm5_layerwise.py checks them against HF's own
loader (from_pretrained over a synthetic checkpoint in the original naming).

Modules are built on the meta device and filled with load_state_dict(assign=True,
strict=True) — the qwen35_common.py pattern: no random init of the big
matrices, a missing or extra tensor is an error. Shards are opened per call
(safe_open), never cached: a kept handle keeps its mmap in RSS.
"""
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys

import numpy as np
import torch
from safetensors import safe_open

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_DIR = os.path.join(ROOT, "models", "GLM-5.3-Flash-original")
LM = "model.language_model."
FP8_BLOCK = 128  # quantization_config.weight_block_size [128, 128] of rev eb9eb208
FP8_MAX = 448.0  # largest finite E4M3 (fn) value


CONVERTER = os.environ.get("CROW_CONVERTER") or os.path.join(
    ROOT, "converter", "target", "release", "converter.exe" if sys.platform == "win32" else "converter")


class ContainerError(RuntimeError):
    pass


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def text_config_from_dict(cfg):
    """Glm5NextTextConfig from a config.json dict (nested `text_config` or flat), eager f32"""
    from transformers.models.glm5_next.configuration_glm5_next import Glm5NextConfig, Glm5NextTextConfig
    if "text_config" in cfg:
        tc = Glm5NextConfig(**{k: v for k, v in cfg.items() if k not in ("architectures",)}).text_config
    else:
        tc = Glm5NextTextConfig(**cfg)
    tc._attn_implementation = "eager"  # deterministic eager path, softmax f32
    # the plain per-expert loop of Glm5NextTextExperts.forward; HF's full model picks
    # "grouped_mm" when it can dispatch it (modeling_utils.get_correct_experts_implementation)
    tc._experts_implementation = "eager"
    return tc


def text_config(model_dir):
    with open(os.path.join(model_dir, "config.json")) as f:
        return text_config_from_dict(json.load(f))


def build_meta(cls, *args):
    with torch.device("meta"):
        return cls(*args)


# ---------------------------------------------------------------- FP8 (E4M3, 128x128 block scales)

def fp8_dequant(w, s_inv, block=FP8_BLOCK):
    """E4M3 [R, C] x f32 scale_inv [ceil(R/b), ceil(C/b)] -> f32 [R, C].
    Block (i, j) covers rows i*b..i*b+b-1, cols j*b..j*b+b-1; the last block of a
    dimension that is not a multiple of b is partial (DeepSeek-V3 weight_dequant rule).
    HF's Fp8Dequantize derives the block from the grid (rows // scale_rows); both
    rules agree whenever every dim is a multiple of b or has one block — true for
    every FP8 tensor of glm5_next (dims 512..32768 in steps of 128; b_proj 64 and
    indexer.weights_proj 32 rows have one block)."""
    assert w.dim() == 2 and s_inv.dim() == 2, (tuple(w.shape), tuple(s_inv.shape))
    R, C = w.shape
    assert tuple(s_inv.shape) == (math.ceil(R / block), math.ceil(C / block)), \
        f"scale grid {tuple(s_inv.shape)} does not fit weight {tuple(w.shape)} with {block}x{block} blocks"
    s = s_inv.float().repeat_interleave(block, 0)[:R].repeat_interleave(block, 1)[:, :C]
    return w.float() * s


def fp8_quant(w, block=FP8_BLOCK):
    """f32 [R, C] -> (E4M3 [R, C], f32 scale_inv grid): amax per block / 448 (synthetic checkpoints only)"""
    R, C = w.shape
    sr, sc = math.ceil(R / block), math.ceil(C / block)
    pad = torch.zeros(sr * block, sc * block, dtype=torch.float32)
    pad[:R, :C] = w.float()
    amax = pad.abs().view(sr, block, sc, block).amax(dim=(1, 3))
    s_inv = (amax / FP8_MAX).clamp(min=1e-12)
    s = s_inv.repeat_interleave(block, 0)[:R].repeat_interleave(block, 1)[:, :C]
    return (w.float() / s).to(torch.float8_e4m3fn), s_inv


# ---------------------------------------------------------------- module <-> checkpoint names

_FORGET = re.compile(r"self_attn\.forget_gate\.(f_a_proj\.weight|f_b_proj\.weight|dt_bias|A_log)")
_HC = re.compile(r"(attn|ffn)_hc\.(fn|base|scale)")


def ckpt_recipe(key):
    """module key (relative to a decoder layer) -> how to read it from the checkpoint:
    ("one", name) | ("experts_gate_up",) | ("experts_down",) | ("cat0", [names])"""
    m = _FORGET.fullmatch(key)
    if m:
        return ("one", "self_attn." + m.group(1))
    m = _HC.fullmatch(key)
    if m:
        return ("one", f"hc_{m.group(1)}_{m.group(2)}")
    if key == "mlp.experts.gate_up_proj":
        return ("experts_gate_up",)
    if key == "mlp.experts.down_proj":
        return ("experts_down",)
    if key == "self_attn.conv1d.weight":
        return ("cat0", [f"self_attn.{p}_conv1d.weight" for p in "qkv"])
    return ("one", key)


def module_to_ckpt(key, t):
    """writer side of ckpt_recipe: one module tensor -> {checkpoint name: tensor} (layer-relative)"""
    r = ckpt_recipe(key)
    if r[0] == "one":
        return {r[1]: t}
    if r[0] == "experts_gate_up":
        I = t.shape[1] // 2
        out = {}
        for e in range(t.shape[0]):
            out[f"mlp.experts.{e}.gate_proj.weight"] = t[e, :I]
            out[f"mlp.experts.{e}.up_proj.weight"] = t[e, I:]
        return out
    if r[0] == "experts_down":
        return {f"mlp.experts.{e}.down_proj.weight": t[e] for e in range(t.shape[0])}
    parts = t.chunk(len(r[1]), dim=0)
    return dict(zip(r[1], parts))


# ---------------------------------------------------------------- weight source

def load_plan(module, prefix):
    """the checkpoint names a module reads, in load order: [(module key, shape, recipe, [names])].
    Both back ends fill a module in exactly this order (`WeightSource.load`)."""
    items = []
    for k, v in module.state_dict().items():
        shp = tuple(v.shape)
        r = ckpt_recipe(k)
        if r[0] == "one":
            names = [prefix + r[1]]
        elif r[0] == "cat0":
            names = [prefix + n for n in r[1]]
        elif r[0] == "experts_gate_up":
            names = [f"{prefix}mlp.experts.{e}.{p}_proj.weight" for e in range(shp[0]) for p in ("gate", "up")]
        else:
            names = [f"{prefix}mlp.experts.{e}.down_proj.weight" for e in range(shp[0])]
        items.append((k, shp, r, names))
    return items


class WeightSource:
    """the FP8 originals (`fp8`); `ContainerSource` below is the `cnq` back end"""

    def __init__(self, kind, path=MODEL_DIR):
        assert kind == "fp8", f"WeightSource reads the FP8 originals; use open_weights('{kind}', ...)"
        self.kind = kind
        self.path = path
        idx = os.path.join(path, "model.safetensors.index.json")
        assert os.path.exists(idx), f"{idx}: missing (expected the FP8 originals of zai-org/GLM-5.3-Flash)"
        with open(idx) as f:
            self.wm = json.load(f)["weight_map"]

    def config_dict(self):
        with open(os.path.join(self.path, "config.json")) as f:
            return json.load(f)

    def _open(self, name):
        # opened per call, never cached: a kept handle keeps its mmap, and every
        # touched page of the shards would stay in this process's RSS
        return safe_open(os.path.join(self.path, self.wm[name]), framework="pt", device="cpu")

    def has(self, name):
        return name in self.wm

    def is_fp8(self, name):
        return name.endswith(".weight") and (name + "_scale_inv") in self.wm

    def dtype_of(self, name):
        with self._open(name) as f:
            return str(f.get_slice(name).get_dtype())

    def shape_of(self, name):
        with self._open(name) as f:
            return list(f.get_slice(name).get_shape())

    def get(self, name):
        """the whole tensor, f32, in its checkpoint shape (FP8 dequantized)"""
        assert self.has(name), f"{self.kind}: missing {name}"
        with self._open(name) as f:
            w = f.get_tensor(name)
        if self.is_fp8(name):
            with self._open(name + "_scale_inv") as f:
                s = f.get_tensor(name + "_scale_inv")
            assert w.dtype == torch.float8_e4m3fn, f"{name}: has a scale_inv but dtype {w.dtype}"
            return fp8_dequant(w, s)
        assert w.dtype != torch.float8_e4m3fn, f"{name}: E4M3 without a weight_scale_inv"
        return w.to(torch.float32).clone()

    def get_many(self, names):
        """the tensors of `names`, in that order, one at a time (a generator)"""
        for n in names:
            yield self.get(n)

    def rows(self, name, r0, r1):
        """rows [r0, r1) of a 2-D tensor, f32 — the embedding lookup and the chunked
        lm_head never hold the whole table. FP8 rows are read in whole 128-row blocks."""
        assert self.has(name), f"{self.kind}: missing {name}"
        if self.is_fp8(name):
            b0, b1 = r0 // FP8_BLOCK, math.ceil(r1 / FP8_BLOCK)
            with self._open(name) as f:
                w = f.get_slice(name)[b0 * FP8_BLOCK:b1 * FP8_BLOCK]
            with self._open(name + "_scale_inv") as f:
                s = f.get_slice(name + "_scale_inv")[b0:b1]
            return fp8_dequant(w, s)[r0 - b0 * FP8_BLOCK:r1 - b0 * FP8_BLOCK].contiguous()
        with self._open(name) as f:
            return f.get_slice(name)[r0:r1].to(torch.float32).clone()

    def n_rows(self, name):
        return self.shape_of(name)[0]

    def embed(self, ids, name=LM + "embed_tokens.weight"):
        """[len(ids), H] f32: one row read per distinct id"""
        uniq = sorted(set(int(i) for i in ids))
        rows = {i: self.rows(name, i, i + 1)[0] for i in uniq}
        return torch.stack([rows[int(i)] for i in ids])

    def load(self, module, prefix):
        """fill a (meta-built) module from the checkpoint names under `prefix`, strict;
        the routed experts are written into one preallocated [E, ...] tensor so a
        288-expert layer never holds a second copy. The names are requested in
        `load_plan` order through `get_many` (one converter call per module for `cnq`)."""
        plan = load_plan(module, prefix)
        it = self.get_many([n for _, _, _, names in plan for n in names])
        state = {}
        for k, shp, r, names in plan:
            if r[0] == "one":
                t = next(it)
            elif r[0] == "cat0":
                t = torch.cat([next(it) for _ in names], dim=0)
                assert t.numel() == math.prod(shp), f"{prefix}{k}: {tuple(t.shape)} vs module {shp}"
                t = t.reshape(shp)
            else:
                t = torch.empty(shp, dtype=torch.float32)
                I = shp[1] // 2
                for e in range(shp[0]):
                    if r[0] == "experts_gate_up":
                        t[e, :I] = next(it)
                        t[e, I:] = next(it)
                    else:
                        t[e] = next(it)
                assert not self.has(f"{prefix}mlp.experts.{shp[0]}.down_proj.weight"), \
                    f"{prefix}: the checkpoint has more than {shp[0]} routed experts"
            assert tuple(t.shape) == shp, f"{prefix}{k}: {tuple(t.shape)} vs module {shp}"
            state[k] = t
        assert next(it, None) is None, f"{prefix}: load_plan and load disagree"
        module.load_state_dict(state, strict=True, assign=True)
        return module.float().eval()

    def provenance(self):
        idx = os.path.join(self.path, "model.safetensors.index.json")
        cfg = os.path.join(self.path, "config.json")
        st = os.stat(idx)
        try:
            rel = os.path.relpath(self.path, ROOT)
        except ValueError:  # another drive (Windows)
            rel = self.path
        return {
            "weights": "fp8 (FP8 E4M3 originals, 128x128 weight_scale_inv blocks, dequantized to f32; BF16/F32 widened)",
            "path": rel,
            "index_json_sha256": sha256_file(idx),
            "config_json_sha256": sha256_file(cfg),
            "index_json_mtime": int(st.st_mtime),
            "n_tensors": len(self.wm),
            "n_fp8": sum(1 for n in self.wm if self.is_fp8(n)),
        }


class ContainerSource(WeightSource):
    """`--weights container <file.cnq>`: a glm5_next CNQ container (crow-nest #156), whole or
    partial. Every tensor comes out of `converter dequant` (`converter/src/dequant.rs`): NVFP4
    decoded by the same `nvfp4_scale` / `nvfp4_value` gate 0 measures the written encoding with,
    BF16 widened exactly, F32 as stored. The index trailer is read here only for names, shapes
    and the checkpoint's config.json (carried verbatim in `model.config_json`)."""

    def __init__(self, path, converter=None):
        self.kind = "cnq"
        self.path = path
        self.converter = converter or CONVERTER
        if not os.path.isfile(path):
            raise ContainerError(f"--weights container {path}: no such file")
        if not os.path.isfile(self.converter):
            raise ContainerError(f"{self.converter}: the converter binary is missing (cd converter && cargo build "
                                 "--release, or set CROW_CONVERTER); the container is decoded by it, not here")
        with open(path, "rb") as f:
            if f.read(4) != b"CNQ1":
                raise ContainerError(f"{path}: not a CNQ1 container")
            f.seek(-8, 2)
            n = struct.unpack("<Q", f.read(8))[0]
            f.seek(-(8 + n), 2)
            raw = f.read(n)
        self.index_sha256 = hashlib.sha256(raw).hexdigest()
        self.index = json.loads(raw)
        if self.index.get("format_version") != 2 or self.index.get("recipe") != "cnq4.5-glm5-next":
            raise ContainerError(f"{path}: index format {self.index.get('format_version')} recipe "
                                 f"{self.index.get('recipe')}, expected an index v2 of the cnq4.5-glm5-next row")
        self.tensors = {t["name"]: t for t in self.index["tensors"]}
        self.partial = self.index.get("partial")

    def config_dict(self):
        return json.loads(self.index["model"]["config_json"])

    def has(self, name):
        return name in self.tensors

    def is_fp8(self, name):
        return False

    def dtype_of(self, name):
        return self.tensors[name]["dtype"]

    def shape_of(self, name):
        return list(self.tensors[name]["shape"])

    def _missing(self, name):
        part = f" (a PARTIAL container: {self.partial['filter']})" if self.partial else ""
        return ContainerError(f"{name}: not in {self.path}{part}")

    def _dequant(self, specs):
        """[(spec, n_values, shape)] -> generator of f32 tensors, one converter process"""
        if not specs:
            return
        p = subprocess.Popen([self.converter, "dequant", self.path, "--names", "-"], stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            p.stdin.write(("\n".join(s for s, _, _ in specs) + "\n").encode())
            p.stdin.close()
            for spec, n, shape in specs:
                buf = p.stdout.read(4 * n)
                if len(buf) != 4 * n:
                    p.wait()
                    raise ContainerError(f"converter dequant {spec}: {len(buf)} of {4 * n} B, rc {p.returncode}: "
                                         f"{p.stderr.read().decode(errors='replace').strip()}")
                yield torch.from_numpy(np.frombuffer(buf, dtype="<f4").copy()).reshape(shape)
            if p.stdout.read(1):
                raise ContainerError("converter dequant wrote more than was asked for")
            if p.wait() != 0:
                raise ContainerError(f"converter dequant: rc {p.returncode}: {p.stderr.read().decode(errors='replace')}")
        finally:
            if p.poll() is None:
                p.kill()
                p.wait()
            p.stdout.close()
            p.stderr.close()

    def get_many(self, names):
        for n in names:
            if n not in self.tensors:
                raise self._missing(n)
        return self._dequant([(n, self.tensors[n]["n_values"], self.tensors[n]["shape"]) for n in names])

    def get(self, name):
        return next(self.get_many([name]))

    def rows(self, name, r0, r1):
        if name not in self.tensors:
            raise self._missing(name)
        cols = self.tensors[name]["shape"][1]
        return next(self._dequant([(f"{name}:{r0}:{r1}", (r1 - r0) * cols, [r1 - r0, cols])]))

    def embed(self, ids, name=LM + "embed_tokens.weight"):
        if name not in self.tensors:
            raise self._missing(name)
        cols = self.tensors[name]["shape"][1]
        uniq = sorted(set(int(i) for i in ids))
        got = self._dequant([(f"{name}:{i}:{i + 1}", cols, [cols]) for i in uniq])
        rows = dict(zip(uniq, got))
        return torch.stack([rows[int(i)] for i in ids])

    def provenance(self):
        st = os.stat(self.path)
        try:
            rel = os.path.relpath(self.path, ROOT)
        except ValueError:
            rel = self.path
        m = self.index["model"]
        dts = [t["dtype"] for t in self.index["tensors"]]
        return {
            "weights": "cnq (CNQ container, decoded by `converter dequant`, the converter's gate-0 decode)",
            "path": rel,
            "index_json_sha256": self.index_sha256,
            "config_json_sha256": m.get("config_json_sha256"),
            "container_bytes": st.st_size,
            "container_mtime": int(st.st_mtime),
            "recipe": self.index.get("recipe"),
            "scales": self.index.get("scales"),
            "source": {"repo": m["source"]["repo"], "revision": m["source"]["revision"]},
            "partial": self.partial,
            "converter": {"path": self.converter, "sha256": sha256_file(self.converter)},
            "n_tensors": len(dts),
            "n_nvfp4": dts.count("nvfp4"),
        }


def open_weights(kind, path):
    """`fp8` -> WeightSource over the FP8 originals, `cnq` -> ContainerSource"""
    if kind == "fp8":
        return WeightSource("fp8", path)
    if kind == "cnq":
        return ContainerSource(path)
    raise ValueError(kind)


# ---------------------------------------------------------------- synthetic checkpoints (the proof)

# the FP8 split observed in the index of rev eb9eb208 (HF web view, 2026-10-08): dense and
# shared/routed MLP projections and the DSA q_a/q_b/kv_a/o projections carry a weight_scale_inv;
# KDA projections, kv_b_proj, the indexer, router, norms, hc_* and embed/lm_head do not.
# KDA b_proj (num_heads rows < 128: one partial block) is FP8 here on purpose, synthetic only,
# so the partial-block path of fp8_dequant is exercised.
SYN_FP8 = re.compile(
    r"\.(mlp\.(gate|up|down)_proj|mlp\.shared_experts\.(gate|up|down)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj"
    r"|self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj|b_proj))\.weight$")
SYN_F32 = re.compile(r"(e_score_correction_bias|\.A_log|\.dt_bias|_conv1d\.weight)$")
# crow-nest #156: the dtypes the shard headers of rev eb9eb208 carry (headers/, read 2026-10-08),
# which the converter's cnq4.5-glm5-next whitelist accepts: b_proj and the conv weights BF16,
# A_log / dt_bias / e_score_correction_bias / hc_*_base / hc_*_scale F32. (o_proj is FP8 for DSA
# layers only on disk; the row converts either source, so FP8 everywhere is kept here.)
CNQ_FP8 = re.compile(
    r"\.(mlp\.(gate|up|down)_proj|mlp\.shared_experts\.(gate|up|down)_proj|mlp\.experts\.\d+\.(gate|up|down)_proj"
    r"|self_attn\.(q_a_proj|q_b_proj|kv_a_proj_with_mqa|o_proj))\.weight$")
CNQ_F32 = re.compile(r"(e_score_correction_bias|\.A_log|\.dt_bias|\.hc_(attn|ffn)_(base|scale))$")


def write_synthetic_checkpoint(model, out_dir, config_dict, shard_bytes=2 << 30, fp8_re=SYN_FP8, f32_re=SYN_F32):
    """model: a Glm5NextForConditionalGeneration (f32). Writes its text model + lm_head in the
    ORIGINAL naming and format (per-expert tensors, hc_*, q/k/v_conv1d, E4M3 + weight_scale_inv
    for `fp8_re`, BF16 for the rest, F32 for `f32_re`) plus index and config. The vision tower
    is not written. Returns the number of tensors."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    tm = model.model.language_model
    items = []  # (ckpt name, tensor) in order
    for k, v in tm.state_dict().items():
        m = re.match(r"layers\.(\d+)\.(.*)", k)
        if m:
            for n, t in module_to_ckpt(m.group(2), v.detach()).items():
                items.append((f"{LM}layers.{m.group(1)}.{n}", t))
        else:
            items.append((LM + k, v.detach()))
    items.append(("lm_head.weight", model.lm_head.weight.detach()))
    weight_map, shard, size, n_shard, total = {}, {}, 0, 0, 0

    def flush():
        nonlocal shard, size, n_shard
        if shard:
            fn = f"model-{n_shard:05d}.safetensors"
            save_file(shard, os.path.join(out_dir, fn), metadata={"format": "pt"})
            for n in shard:
                weight_map[n] = fn
            n_shard += 1
            shard, size = {}, 0

    for name, t in items:
        if fp8_re.search(name):
            q, s = fp8_quant(t.float())
            add = {name: q.contiguous(), name + "_scale_inv": s.contiguous()}
        elif f32_re.search(name):
            add = {name: t.float().contiguous()}
        else:
            add = {name: t.to(torch.bfloat16).contiguous()}
        for n, x in add.items():
            shard[n] = x
            size += x.numel() * x.element_size()
            total += x.numel() * x.element_size()
        if size >= shard_bytes:
            flush()
    flush()
    with open(os.path.join(out_dir, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"total_size": total}, "weight_map": weight_map}, f, indent=1)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config_dict, f, indent=1)
    return len(weight_map)
