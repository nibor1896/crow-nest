"""Is the ring reader the limit after layer 6? Ring copy g (time order = plan order) cannot start
before read g landed. (1) lower bound on the reader rate: records i..j must land between the
moment slot of i was free (copy end of record i-192) and the start of copy j.
(2) slack of every copy start against a reader that never stalls: t0 + g * RB / R."""
import sys, os, io, contextlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A
with contextlib.redirect_stdout(io.StringIO()):
    v = A.main()
a, b, ops, ring_cids = v["a"], v["b"], v["ops"], v["ring_cids"]
RB, SLOTS = 9474048, 192
rc = [o for o in ops[17] if o[4] in ring_cids]
S = [o[0] for o in rc]; E = [o[1] for o in rc]
n = len(rc)
# (1) best lower bound over pairs with j - i >= 192 (steady state: i from the l7 start)
i0 = next(i for i, s in enumerate(S) if s > a + 1.9e9)
best = (0, 0, 0)
for i in range(max(i0, SLOTS), n, 8):
    T = E[i - SLOTS]
    for j in range(i + 50, min(n, i + 1200), 8):
        r = (j - i + 1) * RB / max(S[j] - T, 1)
        if r > best[0]:
            best = (r, i, j)
r, i, j = best
print(f"(1) l7-l44: reader rate >= {r:.3f} GB/s (records {i}..{j}, slot of {i} free at {(E[i-SLOTS]-a)/1e9:.3f} s, copy {j} at {(S[j]-a)/1e9:.3f} s)")
# (2) slack against a never-stalling reader at R, from the plan start (first copy) and fitted t0
for R in (6.994, 7.05, 7.2, 7.7):
    dt = RB / R  # ns per record
    t0 = min(S[g] - g * dt for g in range(i0, n))
    sl = [S[g] - (t0 + g * dt) for g in range(i0, n)]
    near = sum(1 for x in sl if x < 2e6)
    print(f"(2) R {R:5.3f} GB/s: copies within 2 ms of the reader front: {near}/{len(sl)}; median slack {sorted(sl)[len(sl)//2]/1e6:.1f} ms; t0 {(t0-a)/1e9:.3f} s")
