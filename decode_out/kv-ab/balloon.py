#!/usr/bin/env python3
"""kv-ab operational helper: force the NVIDIA driver's lazy pinned pool back.

After every engine exit the ~46 GiB pinned tier sits in the driver pool
(documented #15/#82, HANDOFF-2026-09-20-pm 'Ballon-Beweis'). The next engine's
loader then sees MemAvailable ~8 GiB and refuses config (manager.rs:203) or the
pre-pin gate (residency.rs:247). Sustained anonymous-memory pressure forces the
pool back within seconds to a minute; one shallow pass is sometimes not enough,
so the deep pattern (stop at 1.5 GiB, 2 s pace, up to 60 GiB) repeats until the
target MemAvailable is reached or three rounds pass."""
import sys, time

TARGET_GIB = float(sys.argv[1]) if len(sys.argv) > 1 else 46.0

def memavail():
    with open('/proc/meminfo') as f:
        for l in f:
            if l.startswith('MemAvailable'):
                return int(l.split()[1]) * 1024

for rnd in range(3):
    if memavail() >= TARGET_GIB * 2**30:
        print("memavail %.1f GiB >= target" % (memavail()/2**30)); sys.exit(0)
    balloons, total = [], 0
    try:
        for step in range(64):
            if memavail() < 1100 * 2**20:
                time.sleep(30)  # sustained hold at pressure: kswapd/driver need time
                if memavail() < 1100 * 2**20: break
            if total >= 62 * 2**30: break
            c = bytearray(1 * 2**30)
            for i in range(0, len(c), 8192):
                c[i] = 1
            balloons.append(c); total += len(c)
            time.sleep(1)
            if memavail() > 40 * 2**30: break
        time.sleep(4)
    finally:
        balloons.clear()
    print("round %d: balloon %.0f GiB -> %.1f GiB" % (rnd, total/2**30, memavail()/2**30))
print("final memavail %.1f GiB" % (memavail()/2**30))
sys.exit(0 if memavail() >= TARGET_GIB * 2**30 else 1)
