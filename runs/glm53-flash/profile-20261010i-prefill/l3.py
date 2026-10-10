import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
with contextlib.redirect_stdout(io.StringIO()):
    v = A.main()
a, b, ops, st17, ring_cids, host = v["a"], v["b"], v["ops"], v["st17"], v["ring_cids"], v["host"]
lo, hi = float(sys.argv[1]), float(sys.argv[2])
lo, hi = a + int(lo * 1e9), a + int(hi * 1e9)
ev = []
for h in host:
    if lo <= h[0] < hi and (h[1] - h[0] > 300_000 or h[2] in ("cuStreamSynchronize", "cuEventSynchronize")):
        ev.append((h[0], "HOST", h[2], h[1] - h[0]))
# host idle stretches (no CUDA call) > 2 ms on the main thread
hs = [h for h in host if lo - 1e9 <= h[0] < hi]
for x, y in zip(hs, hs[1:]):
    if y[0] - x[1] > 2_000_000 and lo <= x[1] < hi:
        ev.append((x[1], "HOST", "-- no CUDA call (CPU / NVMe on host) --", y[0] - x[1]))
# collapse stream 17 states
cur = None
for s, e, k in st17:
    if s < lo or s >= hi:
        continue
    if cur and cur[2] == k and s - cur[1] < 50_000:
        cur[1] = e; cur[3] += 1
    else:
        if cur: ev.append((cur[0], "S17", f"{cur[2]} x{cur[3]}", cur[1] - cur[0]))
        cur = [s, e, k, 1]
if cur: ev.append((cur[0], "S17", f"{cur[2]} x{cur[3]}", cur[1] - cur[0]))
cur = None
for o in ops[7]:
    if o[0] < lo or o[0] >= hi:
        continue
    g = A.group(o[3]) if o[2] == "K" else o[3]
    if cur and cur[2] == g and o[0] - cur[1] < 200_000:
        cur[1] = o[1]; cur[3] += 1
    else:
        if cur: ev.append((cur[0], "S7", f"{cur[2]} x{cur[3]}", cur[1] - cur[0]))
        cur = [o[0], o[1], g, 1]
if cur: ev.append((cur[0], "S7", f"{cur[2]} x{cur[3]}", cur[1] - cur[0]))
ev.sort()
for t, w, n, d in ev:
    if d > 500_000 or w == "HOST":
        print(f"{(t - a) / 1e9:8.4f} {w:4s} {n[:70]:70s} {d / 1e6:8.2f} ms")
