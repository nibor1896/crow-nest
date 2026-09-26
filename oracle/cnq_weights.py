"""cnq_weights.py — #VIT: read tensors out of a CNQ1 container, dequantized.

The oracle weight source of record for the image path (orchestrator ruling
2026-09-14): the original bf16 safetensors shards are no longer on disk, and
the gate is math precision — the engine and the oracle must see IDENTICAL
weights, so the oracle dequantizes the same NVFP4 bytes the engine loads.

Layout (converter/verify_roundtrip.py is the reference of record): the file
starts with the magic CNQ1, the last 8 bytes hold the u64 length of a JSON
index that names every tensor (name, dtype, offset, len, n_values,
global_scale) over one blob region. NVFP4 tensors are 36-byte blocks per 64
values: FOUR ue4m3 scale bytes (one per 16-value sub-block) + 32 pack bytes
(low nibble = even value, high nibble = odd value). bf16 keeps pass through.
The 104 GB file is SEEKED, never buffered whole.

Index v2 (Crow #300 C6) adds dtype "f32" (raw little-endian f32, e.g. the
27B's A_log) and a `shape` per tensor record; `tensor()` / `rows_f32()` use it.
"""
import hashlib
import json
import struct

import numpy as np
import torch

E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6], dtype=np.float32)


class CnqReader:
    def __init__(self, path, sanitize_sf=False):
        # sanitize_sf (Crow #300 phase 2): read an NVFP4 scale byte 0x7F (the E4M3 NaN
        # code) as 0x7E, the rule the engine applies at load to every NVFP4 weight of a
        # non-Flash-Next container (gen.rs `load_pw_x`), so both sides see the same weights
        self.sanitize_sf = sanitize_sf
        self.path = path
        f = open(path, "rb")
        f.seek(-8, 2)
        idx_len = struct.unpack("<Q", f.read(8))[0]
        f.seek(-(8 + idx_len), 2)
        idx_raw = f.read(idx_len)
        f.close()
        idx = json.loads(idx_raw)
        self.index = idx
        self.index_len = idx_len
        self.index_sha256 = hashlib.sha256(idx_raw).hexdigest()
        self.blob_off = idx["blob_offset"]
        self.tensors = {t["name"]: t for t in idx["tensors"] if "name" in t}

    def raw_bytes(self, name):
        t = self.tensors[name]
        with open(self.path, "rb") as f:
            f.seek(self.blob_off + t["offset"])
            return f.read(t["len"])

    @staticmethod
    def _deq_block(block_bytes, g):
        block = np.frombuffer(block_bytes, dtype=np.uint8)
        e = ((block[:4].astype(np.uint32) >> 3) & 0xF)
        m = (block[:4] & 7).astype(np.float32)
        s = np.where(e == 0, m * np.float32(2.0 ** -6 / 8),
                     (1 + m / 8) * np.float32(2.0 ** (e.astype(np.int32) - 7))).astype(np.float32) * g
        vals = block[4:]
        lo = (vals & 0xF).astype(np.uint32)
        hi = ((vals >> 4) & 0xF).astype(np.uint32)
        nib = np.empty(64, dtype=np.uint32)
        nib[0::2] = lo
        nib[1::2] = hi
        mag = E2M1[nib & 7]
        out = np.where(nib & 8, -mag, mag)
        return out * np.repeat(s, 16)

    def tensor_i64(self, name) -> torch.Tensor:
        t = self.tensors[name]
        assert t["dtype"] == "i64", f"{name}: dtype {t['dtype']}"
        raw = self.raw_bytes(name)
        v = np.frombuffer(raw, dtype=np.int64).copy()
        return torch.from_numpy(v)

    def tensor_f32(self, name) -> torch.Tensor:
        """one tensor as a flat f32 torch tensor, dequantized exactly like the engine"""
        t = self.tensors[name]
        raw = self.raw_bytes(name)
        if t["dtype"] == "i64":
            v = np.frombuffer(raw, dtype=np.int64).astype(np.float32)
            return torch.from_numpy(np.ascontiguousarray(v))
        if t["dtype"] == "f32":
            return torch.from_numpy(np.frombuffer(raw, dtype="<f4").astype(np.float32).copy())
        if t["dtype"] == "bf16":
            return torch.from_numpy(self._bf16_to_f32(raw))
        assert t["dtype"] == "nvfp4", f"{name}: dtype {t['dtype']}"
        return torch.from_numpy(self._deq_flat(raw, t["global_scale"])[:t["n_values"]].copy())

    @staticmethod
    def _bf16_to_f32(raw):
        v = np.frombuffer(raw, dtype="<u2").astype(np.uint32) << np.uint32(16)
        return v.view(np.float32).copy()

    def tensor(self, name) -> torch.Tensor:
        """tensor_f32 reshaped to the index record's `shape` (index v2)"""
        t = self.tensors[name]
        return self.tensor_f32(name).reshape(t["shape"])

    def rows_f32(self, name, r0, r1) -> torch.Tensor:
        """rows [r0, r1) of a 2-D tensor as f32 [r1-r0, cols], read by seek —
        the embedding lookup and the chunked lm_head never hold the whole
        table. NVFP4 needs cols % 64 == 0 (a row is whole blocks)."""
        t = self.tensors[name]
        rows, cols = t["shape"]
        assert 0 <= r0 <= r1 <= rows, (name, r0, r1, rows)
        n = (r1 - r0) * cols
        if t["dtype"] == "nvfp4":
            assert cols % 64 == 0, f"{name}: row of {cols} is not whole 64-value blocks"
            bpr = cols // 64 * 36
            off, ln = r0 * bpr, (r1 - r0) * bpr
        else:
            w = {"bf16": 2, "f32": 4}[t["dtype"]]
            off, ln = r0 * cols * w, n * w
        with open(self.path, "rb") as f:
            f.seek(self.blob_off + t["offset"] + off)
            raw = f.read(ln)
        if t["dtype"] == "nvfp4":
            v = self._deq_flat(raw, t["global_scale"]).astype(np.float32)
        elif t["dtype"] == "bf16":
            v = self._bf16_to_f32(raw)
        else:
            v = np.frombuffer(raw, dtype="<f4").astype(np.float32)
        return torch.from_numpy(v.reshape(r1 - r0, cols))

    def _deq_flat(self, raw, g):
        """vectorized whole-tensor dequant — the per-block math of
        _deq_block without the python loop (the routed-expert tensors are
        ~1.7B values; the loop form costs minutes per layer)"""
        n_blocks = len(raw) // 36
        b = np.frombuffer(raw[:n_blocks * 36], dtype=np.uint8).reshape(n_blocks, 36)
        # scales: 4 ue4m3 bytes per block, one per 16-value sub-block
        sc = b[:, 0:4].astype(np.uint32)
        if self.sanitize_sf:
            sc = np.where(sc == 0x7F, np.uint32(0x7E), sc)
        e = (sc >> 3) & 0xF
        m = (sc & 7).astype(np.float32)
        s = np.where(e == 0, m * np.float32(2.0 ** -6 / 8),
                     (1 + m / 8) * np.float32(2.0 ** (e.astype(np.int32) - 7))).astype(np.float32) * g
        # nibbles: 32 pack bytes, low nibble = even value, high = odd
        pk = b[:, 4:36]
        nib = np.empty((n_blocks, 64), dtype=np.uint8)
        nib[:, 0::2] = pk & 0xF
        nib[:, 1::2] = pk >> 4
        mag = E2M1[nib & 7]
        vals = np.where(nib & 8, -mag, mag)
        # sub-block scales repeated over their 16 values
        return (vals.reshape(-1) * np.repeat(s.reshape(-1), 16))

    def has(self, name):
        return name in self.tensors
