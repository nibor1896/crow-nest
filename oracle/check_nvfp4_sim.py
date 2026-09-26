"""PREREG check 2 (decode_out/p2-lh/PREREG.md): nvfp4_sim fed the container's own scale bytes
reproduces the container's dequantized values bit-exactly; the global scale matches the index."""
import sys
import numpy as np
import torch
from qwen35_common import LM, WeightSource
import nvfp4_sim as ns

names = sys.argv[1:] or [f"{LM}layers.0.mlp.gate_proj.weight", f"{LM}layers.0.mlp.down_proj.weight",
    f"{LM}layers.3.self_attn.q_proj.weight", f"{LM}layers.0.linear_attn.in_proj_qkv.weight",
    f"{LM}layers.11.self_attn.o_proj.weight", "lm_head.weight"]
dev = "cuda"
bf = WeightSource("bf16")
cq = WeightSource("cnq")
table128 = torch.cat([ns.UE4M3, torch.tensor([480.0])])  # 0x7F read as 480 = the unsanitized scalar decode
ok = True
for n in names:
    t = cq.cnq.tensors[n]
    w = bf.get(n).to(dev)
    N, K = w.shape
    g = ns.global_scale(w)
    g_idx = np.float32(t["global_scale"])
    raw = np.frombuffer(cq.cnq.raw_bytes(n), dtype=np.uint8).reshape(-1, 36)
    by = torch.from_numpy(raw[:, :4].copy().reshape(N, K // 16)).to(dev)
    n7f = int((by == 0x7F).sum())
    mine = ns.dequant_bytes(w, g, by, table=table128)
    # the sanitized reader and the raw bytes differ only where 0x7F sits: compare against the raw decode
    cq.cnq.sanitize_sf = False
    cont = torch.from_numpy(cq.cnq._deq_flat(cq.cnq.raw_bytes(n), g_idx)[:N * K].astype(np.float32)).reshape(N, K).to(dev)
    cq.cnq.sanitize_sf = True
    same_g = float(g) == float(g_idx)
    diff = int((mine != cont).sum())
    # x126 must not be worse than the container's written choice (plain SSE) per tensor
    _, dq126 = ns.search(w, "x126")
    sse_c = float(((cont - w) ** 2).sum()); sse_x = float(((dq126 - w) ** 2).sum())
    print(f"{n}: [{N}x{K}] global mine {float(g):.9e} index {float(g_idx):.9e} same={same_g} | "
          f"values differing {diff} of {N * K} | 0x7F bytes {n7f} | SSE container {sse_c:.6e} x126 {sse_x:.6e} "
          f"({(sse_x / sse_c - 1) * 100:+.2f} %)", flush=True)
    ok &= same_g and diff == 0 and sse_x <= sse_c * (1 + 1e-6)
    del w, mine, cont, dq126
    torch.cuda.empty_cache()
print("CHECK 2:", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
