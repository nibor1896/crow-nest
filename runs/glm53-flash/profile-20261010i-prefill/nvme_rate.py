"""NVMe throughput inferred from the copy stream: at a ring copy that waited on its landed flag the
consumer has caught up with the reader, so (records consumed) is the records landed there.
NVMe records consumed = ring copies + host-path landing copies (layers 3-6: NVMe experts beyond
the plan's 192-slot row, read synchronously on the main thread)."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
with contextlib.redirect_stdout(io.StringIO()):
    v = A.main()
a, b, ops, st17, ring_cids = v["a"], v["b"], v["ops"], v["st17"], v["ring_cids"]
RB = 9474048
s7 = ops[7]
routers = [o[0] for o in s7 if o[3] == "glm5_router_sig_topk"]
# host-path NVMe copies: non-ring copies on 17 in layers 3-5 segments (no pinned tier there) + 16 in l6
hostpath = []
seg = routers + [b]
for i, (s, e) in enumerate(zip(seg, seg[1:])):
    nr = [o for o in ops[17] if s <= o[0] < e and o[2] == "C" and o[4] not in ring_cids]
    if i < 3:
        hostpath += nr
nv = sorted([o for o in ops[17] if o[4] in ring_cids] + hostpath)
print(f"NVMe records consumed: {len(nv)} (ring {sum(1 for o in nv if o[4] in ring_cids)}, host path layers 3-5 {len(hostpath)}; + 16 host-path in l6 not separable from pinned there)")
waits = [x for x in st17 if x[2].startswith("NVMe")]
wset = sorted(x[1] for x in waits)  # copy start after a landed wait
import bisect
starts = [o[0] for o in nv]
pts = [(t, bisect.bisect_right(starts, t)) for t in wset]
print(f"landed-flag waits on the copy stream: {len(waits)}, {sum(x[1]-x[0] for x in waits)/1e9:.3f} s")
t0, c0 = pts[0]; t1, c1 = pts[-1]
print(f"first wait at {(t0-a)/1e9:.3f} s (consumed {c0}), last at {(t1-a)/1e9:.3f} s (consumed {c1}): {(c1-c0)*RB/1e9:.2f} GB in {(t1-t0)/1e9:.3f} s = {(c1-c0)*RB/(t1-t0):.3f} GB/s")
# per 4-layer windows between waits
print("rate between landed waits spaced >= 0.5 s:")
last = pts[0]
for t, cc in pts[1:]:
    if t - last[0] >= 0.5e9:
        print(f"  {(last[0]-a)/1e9:6.3f}-{(t-a)/1e9:6.3f} s: {(cc-last[1])*RB/(t-last[0]):.3f} GB/s")
        last = (t, cc)
print(f"first NVMe copy at {(starts[0]-a)/1e9:.3f} s; last ring copy starts {(starts[-1]-a)/1e9:.3f} s, ends {(nv[-1][1]-a)/1e9:.3f}; window end {(b-a)/1e9:.3f} s")
# tail: after the last ring copy, what runs on stream 7
tail = [o for o in s7 if o[0] >= nv[-1][1]]
print(f"tail after the last NVMe record is on the GPU: {(b-nv[-1][1])/1e9:.3f} s, kernels {sum(o[1]-o[0] for o in tail)/1e9:.3f} s")
