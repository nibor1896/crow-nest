"""ring copy cadence where the copy stream waits on landed flags (> 0.3 ms): landing interval per record"""
import sys, os, io, contextlib, statistics as S
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
with contextlib.redirect_stdout(io.StringIO()):
    v = A.main()
a, b, ops, st17, ring_cids, host = v["a"], v["b"], v["ops"], v["st17"], v["ring_cids"], v["host"]
RB = 9474048
rc = [o for o in ops[17] if o[4] in ring_cids]
wait_before = {}
for s, e, k in st17:
    if k.startswith("NVMe"):
        wait_before[e] = e - s
phases = [(0.74, 1.89, "l3-l6 (host path active)"), (1.89, 8.05, "l7-l44")]
for lo, hi, name in phases:
    iv = []
    for x, y in zip(rc, rc[1:]):
        if a + lo * 1e9 <= y[0] < a + hi * 1e9 and wait_before.get(y[0], 0) > 300_000 and wait_before.get(x[0], 0) > 300_000:
            iv.append(y[0] - x[0])
    if iv:
        m = S.median(iv)
        print(f"{name}: {len(iv)} back-to-back landed waits, median interval {m/1e6:.3f} ms = {RB/m:.2f} GB/s, p10 {RB/S.quantiles(iv, n=10)[-1]:.2f} p90 {RB/S.quantiles(iv, n=10)[0]:.2f} GB/s")
# host-path stretches on the main thread (no CUDA call > 2 ms) in l3-l6
hs = host
gaps = [(x[1], y[0] - x[1]) for x, y in zip(hs, hs[1:]) if y[0] - x[1] > 2_000_000 and a + 0.74e9 <= x[1] < a + 1.89e9]
print(f"main thread without CUDA calls in l3-l6: {len(gaps)} stretches, {sum(g for _, g in gaps)/1e9:.3f} s")
gaps2 = [(x[1], y[0] - x[1]) for x, y in zip(hs, hs[1:]) if y[0] - x[1] > 2_000_000 and a + 1.89e9 <= x[1] < b]
print(f"main thread without CUDA calls in l7-l44: {len(gaps2)} stretches, {sum(g for _, g in gaps2)/1e9:.3f} s; sizes ms: {sorted(round(g/1e6,1) for _, g in gaps2)[-10:]}")
# copy-stream ring records per segment l3..l6 and time
