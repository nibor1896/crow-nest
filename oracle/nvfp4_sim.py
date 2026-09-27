"""nvfp4_sim.py — Crow #300 phase 2: the converter's NVFP4 encoder in torch, with an exhaustive
sub-block scale search under three objectives (decode_out/p2-lh/PREREG.md).

Mirrors `converter/src/main.rs`:
  - global scale = max over 16-value sub-blocks of max|x|/6, divided by 448 (`quantize_nvfp4`)
  - a sub-block's scale = decode_ue4m3(byte) * global, f32
  - E2M1 round-to-nearest-midpoint on |x| * (1 / scale) (`quant_dequant`, `e2m1_index`)
Sub-blocks are 16 consecutive values of the row-major tensor; every NVFP4 tensor of the 27B has
K % 64 == 0, so a sub-block is 16 consecutive input columns of one output row.

Objectives of the byte search over the 126 finite codes 1..0x7E (ties keep the smaller byte;
an all-zero sub-block keeps byte 0):
  x126  plain SSE of the dequantized weights
  diag  SSE weighted by h_i = sum x_i^2 of the sub-block's input columns (the imatrix form)
  lh    e^T H_b e, H_b = sum x_b x_b^T over calibration tokens (NVIDIA's Local-Hessian objective)
"""
import torch

E2M1_GRID = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32)
E2M1_MID = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)


def ue4m3_table():
    """decode of the ue4m3 bytes 0..126 (0x7F is NaN and never chosen), f32, as `decode_ue4m3`"""
    out = []
    for b in range(127):
        e, m = b >> 3, b & 7
        out.append(m * 2.0 ** -9 if e == 0 else (1.0 + m / 8.0) * 2.0 ** (e - 7))
    return torch.tensor(out, dtype=torch.float32)


UE4M3 = ue4m3_table()


def fdiv(a, b):
    """IEEE f32 division a / b, correctly rounded as Rust's f32 `/` (CUDA's division by a scalar is
    a multiply by the reciprocal, 1 ulp off; f64 then f32 is exact: 53 >= 2 * 24 + 2 bits)"""
    return (a.double() / b).float()


def global_scale(w):
    """the converter's global scale of a whole tensor (f32 scalar tensor)"""
    sub = fdiv(w.reshape(-1, 16).abs().amax(dim=1), 6.0)
    mx = sub.max()
    return fdiv(mx, 448.0) if mx > 0 else torch.tensor(1.0, dtype=torch.float32, device=w.device)


def dequant_at(w16, s):
    """w16 [..., 16] f32 quantized to E2M1 at per-sub-block scale s [..., 1] (f32), dequantized"""
    inv = fdiv(torch.ones_like(s), s.double()) if s.dim() else fdiv(torch.ones((), device=s.device), s)
    m = w16.abs() * inv
    idx = torch.zeros(m.shape, dtype=torch.uint8, device=m.device)
    for mid in E2M1_MID.tolist():
        idx += m >= mid
    q = E2M1_GRID.to(w16.device)[idx.long()]
    return torch.where(w16 < 0, -q, q) * s


def dequant_bytes(w, g, bytes_, table=None, row_chunk=4096):
    """w [N, K] f32, global g, bytes_ [N, K/16] -> the dequantized tensor the converter would write"""
    N, K = w.shape
    table = (UE4M3 if table is None else table).to(w.device)
    out = torch.empty_like(w)
    for r0 in range(0, N, row_chunk):
        w16 = w[r0:r0 + row_chunk].reshape(-1, K // 16, 16)
        s = (table[bytes_[r0:r0 + row_chunk].long()] * g).unsqueeze(-1)
        out[r0:r0 + row_chunk] = dequant_at(w16, s).reshape(-1, K)
    return out


def search(w, objective, H=None, row_chunk=2048):
    """exhaustive byte search; w [N, K] f32 on the compute device; H [K/16, 16, 16] (lh) or
    [K/16, 16] (diag) f32/f64 on the same device. Returns (bytes [N, K/16] uint8, dequantized [N, K])."""
    assert objective in ("x126", "diag", "lh")
    N, K = w.shape
    B = K // 16
    g = global_scale(w)
    table = UE4M3.to(w.device)
    if objective == "lh":
        Hf = H.to(torch.float32)
    elif objective == "diag":
        d = H.to(torch.float32)
    best_b = torch.zeros(N, B, dtype=torch.uint8, device=w.device)
    for r0 in range(0, N, row_chunk):
        w16 = w[r0:r0 + row_chunk].reshape(-1, B, 16)
        best_err = torch.full(w16.shape[:2], float("inf"), device=w.device, dtype=torch.float32)
        bb = torch.zeros(w16.shape[:2], dtype=torch.uint8, device=w.device)
        for b in range(1, 127):
            s = table[b] * g
            e = dequant_at(w16, s.view(1, 1, 1).expand(*w16.shape[:2], 1)) - w16
            if objective == "x126":
                err = (e * e).sum(-1)
            elif objective == "diag":
                err = (e * e * d.unsqueeze(0)).sum(-1)
            else:
                err = torch.einsum("nbi,bij,nbj->nb", e, Hf, e)
            better = err < best_err
            best_err = torch.where(better, err, best_err)
            bb = torch.where(better, torch.full_like(bb, b), bb)
        zero = w16.abs().amax(-1) == 0
        bb[zero] = 0
        best_b[r0:r0 + row_chunk] = bb
    return best_b, dequant_bytes(w, g, best_b)
