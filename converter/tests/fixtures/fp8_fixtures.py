"""crow-nest #155: the FP8 reference fixtures of `converter/src/fp8.rs`, from torch.

Run with the oracle venv (torch 2.13.0+cpu, transformers 5.16.1), isolated:

    .venv-oracle/Scripts/python.exe -I converter/tests/fixtures/fp8_fixtures.py converter/tests/fixtures

Writes
- fp8-e4m3fn-lut.tsv: code -> f32 bits of torch.float8_e4m3fn (256 rows, "nan" for 0x7F/0xFF);
- fp8-dequant.tsv: sha256 of the f32 LE bytes of two dequantized synthetic tensors:
  300 x 200 by the DeepSeek-V3 `weight_dequant` formula (ceil grid, masked edge tiles,
  `x.to(f32) * s`), 256 x 256 by transformers `Fp8Dequantize._dequantize_one`.
The synthetic input is the same formula as `fp8.rs` `tests::synth`.
"""
import hashlib
import struct
import sys
from pathlib import Path

import torch
from transformers.integrations.finegrained_fp8 import Fp8Dequantize


def synth(rows, cols):
    raw = bytearray()
    for i in range(rows * cols):
        b = ((i * 2654435761) >> 7) & 0xFF
        if b & 0x7F == 0x7F:
            b ^= 0x01
        raw.append(b)
    sr, sc = -(-rows // 128), -(-cols // 128)
    scale = [struct.unpack("<f", struct.pack("<I", 0x3A000000 + ((j * 7919) % 0x00800000) * 3))[0] for j in range(sr * sc)]
    q = torch.frombuffer(raw, dtype=torch.uint8).view(torch.float8_e4m3fn).reshape(rows, cols)
    s = torch.tensor(scale, dtype=torch.float32).reshape(sr, sc)
    return q, s


def deepseek_weight_dequant(q, s):
    # DeepSeek-V3 inference/kernel.py weight_dequant: one f32 scale per 128x128 tile,
    # s[pid_m * cdiv(N, 128) + pid_n], y = x.to(f32) * s, edge tiles masked (= cropped here)
    rows, cols = q.shape
    full = s.repeat_interleave(128, dim=0).repeat_interleave(128, dim=1)[:rows, :cols]
    return q.to(torch.float32) * full


def sha(t):
    return hashlib.sha256(t.contiguous().numpy().astype("<f4").tobytes()).hexdigest()


def last_bits(t):
    return "%08x" % struct.unpack("<I", struct.pack("<f", float(t.reshape(-1)[-1])))[0]


def main(out):
    out = Path(out)
    lines = ["# code\tf32 bits of torch.float8_e4m3fn (torch %s)" % torch.__version__]
    for code in range(256):
        v = torch.tensor([code], dtype=torch.uint8).view(torch.float8_e4m3fn).to(torch.float32)[0]
        lines.append("%d\t%s" % (code, "nan" if torch.isnan(v) else "%08x" % struct.unpack("<I", struct.pack("<f", float(v)))[0]))
    (out / "fp8-e4m3fn-lut.tsv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    rows = ["# who\trows\tcols\tsha256 of f32 LE\tlast value bits (torch %s)" % torch.__version__]
    q, s = synth(300, 200)
    y = deepseek_weight_dequant(q, s)
    rows.append("deepseek_weight_dequant\t300\t200\t%s\t%s" % (sha(y), last_bits(y)))
    q, s = synth(256, 256)
    y = Fp8Dequantize(None)._dequantize_one(q, s, output_dtype=torch.float32)
    assert y.dtype == torch.float32
    rows.append("transformers_Fp8Dequantize\t256\t256\t%s\t%s" % (sha(y), last_bits(y)))
    (out / "fp8-dequant.tsv").write_text("\n".join(rows) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1])
