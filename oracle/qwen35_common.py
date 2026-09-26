"""qwen35_common.py — Crow #300 phase 2: shared plumbing for the dense
Qwen3.8-27B (qwen3_5_text) oracle scripts (export_qwen35_goldens.py,
ref_qwen35_logits.py).

One weight source, two back ends, same tensor names (the HF names — the
converter keeps them verbatim):

  cnq   the CNQ4.5 container DEQUANTIZED (cnq_weights.CnqReader): the exact
        numbers the engine loads, so a comparison isolates engine math
  bf16  the original BF16 safetensors shards (widened to f32, exact): the
        quantization-error mark

Modules are built on the meta device and filled with load_state_dict(assign=True,
strict=True): no random init of the 17408x5120 MLP matrices, and a missing or
extra tensor is an error, not a silent default.
"""
import hashlib
import json
import os

import torch
from safetensors import safe_open

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from cnq_weights import CnqReader

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
MODEL_DIR = os.path.join(ROOT, "models", "Qwen3.8-27B")
CNQ_PATH = os.path.join(ROOT, "converter", "Qwen3.8-27B-CNQ4.5.cnq")
LM = "model.language_model."


def text_config():
    cfg = json.load(open(os.path.join(MODEL_DIR, "config.json")))
    tc = Qwen3_5TextConfig.from_dict(cfg["text_config"])
    tc._attn_implementation = "eager"  # deterministic eager path, softmax f32
    return tc


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


class WeightSource:
    def __init__(self, kind, cnq_path=CNQ_PATH, model_dir=MODEL_DIR):
        assert kind in ("cnq", "bf16"), kind
        self.kind = kind
        if kind == "cnq":
            self.path = cnq_path
            # a container still being written (or cut short) has no index trailer yet
            assert has_trailer(cnq_path), f"{cnq_path}: no index trailer - conversion unfinished?"
            self.cnq = CnqReader(cnq_path, sanitize_sf=True)  # the engine's load rule (gen.rs load_pw_x)
        else:
            self.model_dir = model_dir
            self.wm = json.load(open(os.path.join(model_dir, "model.safetensors.index.json")))["weight_map"]
            self.path = model_dir

    def _open(self, name):
        # opened per call, never cached: a kept handle keeps its mmap, and every
        # touched page of the 54 GB of shards would stay in this process's RSS
        return safe_open(os.path.join(self.model_dir, self.wm[name]), framework="pt", device="cpu")

    def has(self, name):
        return self.cnq.has(name) if self.kind == "cnq" else name in self.wm

    def get(self, name):
        """the whole tensor, f32, in its checkpoint shape"""
        assert self.has(name), f"{self.kind}: missing {name}"
        if self.kind == "cnq":
            return self.cnq.tensor(name)
        with self._open(name) as f:
            return f.get_tensor(name).to(torch.float32).clone()

    def rows(self, name, r0, r1):
        """rows [r0, r1) of a 2-D tensor, f32 — never the whole table"""
        if self.kind == "cnq":
            return self.cnq.rows_f32(name, r0, r1)
        with self._open(name) as f:
            return f.get_slice(name)[r0:r1].to(torch.float32).clone()

    def n_rows(self, name):
        if self.kind == "cnq":
            return self.cnq.tensors[name]["shape"][0]
        with self._open(name) as f:
            return f.get_slice(name).get_shape()[0]

    def dtype_of(self, name):
        if self.kind == "cnq":
            return self.cnq.tensors[name]["dtype"]
        with self._open(name) as f:
            return str(f.get_slice(name).get_dtype())

    def load(self, module, prefix):
        """fill a (meta-built) module from `prefix + param name`, strict"""
        state = {k: self.get(prefix + k) for k in module.state_dict().keys()}
        for k, v in module.state_dict().items():
            assert tuple(v.shape) == tuple(state[k].shape), f"{prefix}{k}: {tuple(state[k].shape)} vs module {tuple(v.shape)}"
        module.load_state_dict(state, strict=True, assign=True)
        return module.float().eval()

    def provenance(self):
        if self.kind == "cnq":
            st = os.stat(self.path)
            idx = self.cnq.index
            return {
                "weights": "cnq (dequantized CNQ4.5 container)",
                "path": os.path.relpath(self.path, ROOT),
                "file_size": st.st_size,
                "file_mtime": int(st.st_mtime),
                "index_sha256": self.cnq.index_sha256,
                "index_len": self.cnq.index_len,
                "format_version": idx.get("format_version"),
                "recipe": idx.get("recipe"),
                "scales": idx.get("scales"),
            }
        st = os.stat(os.path.join(self.model_dir, "model.safetensors.index.json"))
        return {
            "weights": "bf16 (original safetensors shards, widened to f32)",
            "path": os.path.relpath(self.model_dir, ROOT),
            "index_json_sha256": sha256_file(os.path.join(self.model_dir, "model.safetensors.index.json")),
            "config_json_sha256": sha256_file(os.path.join(self.model_dir, "config.json")),
            "index_json_mtime": int(st.st_mtime),
        }


def has_trailer(path):
    import struct
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        f.seek(-8, 2)
        n = struct.unpack("<Q", f.read(8))[0]
        if not 0 < n < size - 12:
            return False
        f.seek(-(8 + n), 2)
        return f.read(1) == b"{"


def build_meta(cls, *args):
    with torch.device("meta"):
        return cls(*args)


def causal_mask(T, past=0):
    """additive [1,1,T,past+T] f32 mask: query i sees keys 0..past+i"""
    min_dtype = torch.finfo(torch.float32).min
    q = torch.arange(T).view(T, 1) + past
    k = torch.arange(past + T).view(1, -1)
    m = torch.zeros(T, past + T, dtype=torch.float32).masked_fill(k > q, min_dtype)
    return m.view(1, 1, T, past + T)


def rope_1d_reference(inv_freq, positions):
    """plain 1-D NeoX rope over the rotary dims: the text-only reduction of the
    interleaved mrope (all three position rows equal)"""
    freqs = positions.float().view(-1, 1) * inv_freq.float().view(1, -1)
    emb = torch.cat([freqs, freqs], dim=-1)
    return emb.cos(), emb.sin()
