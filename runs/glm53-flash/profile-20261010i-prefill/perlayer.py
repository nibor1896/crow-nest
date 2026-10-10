"""per MoE layer view of the prefill window: segment L = [router kernel of L, router kernel of L+1)"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
import io, contextlib
with contextlib.redirect_stdout(io.StringIO()):
    v = A.main()
a, b, ops, st17, ring_cids, host = v["a"], v["b"], v["ops"], v["st17"], v["ring_cids"], v["host"]
s7 = ops[7]
routers = [o[0] for o in s7 if o[3] == "glm5_router_sig_topk"]
bounds = routers + [b]
print(f"first router at {(routers[0]-a)/1e9:.3f} s; {len(routers)} routers")
print(" L  seg_s  kern_s  attn+  experts  idle7  | ring n  nvme-wait_s  ring-h2d_s pin n  pin-h2d_s | host-sync->first copy ms")
tot = {}
for i, (s, e) in enumerate(zip(bounds, bounds[1:])):
    ks = [o for o in s7 if s <= o[0] < e]
    kb = sum(min(o[1], e) - o[0] for o in ks)
    ex = sum(o[1] - o[0] for o in ks if o[2] == "K" and A.group(o[3]).startswith("routed"))
    st = [x for x in st17 if s <= x[0] < e]
    rn = sum(1 for o in ops[17] if s <= o[0] < e and o[4] in ring_cids)
    pn = sum(1 for o in ops[17] if s <= o[0] < e and o[2] == "C" and o[4] not in ring_cids)
    nw = sum(x[1] - x[0] for x in st if x[2].startswith("NVMe"))
    rh = sum(x[1] - x[0] for x in st if x[2] == "H2D ring->stage")
    ph = sum(x[1] - x[0] for x in st if x[2] == "H2D pinned->stage")
    # host: end of the last cuStreamSynchronize > 1 ms in the segment, then first copy on 17
    syncs = [h for h in host if s <= h[1] < e and h[2] == "cuStreamSynchronize" and h[1] - h[0] > 1e6]
    fc = [o[0] for o in ops[17] if s <= o[0] < e]
    lag = ((fc[0] - syncs[-1][1]) / 1e6) if syncs and fc and fc[0] > syncs[-1][1] else float('nan')
    print(f"{i+3:2d} {(e-s)/1e9:6.3f} {kb/1e9:7.3f} {(kb-ex)/1e9:6.3f} {ex/1e9:7.3f} {(e-s-kb)/1e9:6.3f} | {rn:6d} {nw/1e9:10.3f} {rh/1e9:10.3f} {pn:5d} {ph/1e9:9.3f} | {lag:7.1f}")
