#!/usr/bin/env python3
"""coverage-curve.py — crow-nest #3: expert coverage curve from Crow's own routing logs.

Reads llama.cpp `print_locality` blocks (written by crow-lab's moe-stream build) from
crow-lab run logs READ-ONLY and derives the expert-residency coverage curve that sizes
the hot set (issue #7, spec section "memory layout").

What the logs contain: per request, the fraction of experts covering 50/80/95 % of all
routed selections over the 48 expert layers, plus the Gini coefficient. They do NOT
contain per-token per-layer selections — every statement below that would need those is
labeled an assumption, not a measurement.

Method:
  1. Parse all blocks, report min/median/max of the three coverage points and the Gini.
  2. Fit coverage c(N) = a + b*ln(N) least-squares through the MEDIAN points
     (three points in, three parameters out — the fit is a smooth interpolation of the
     measured points, not a model claim; monotone over the relevant range).
  3. P(all 10 routed experts resident) = c(N)^10 under an INDEPENDENCE assumption;
     selections within a layer are correlated in reality, so treat this as an estimate
     with stated sign uncertainty (correlation lowers it, per-layer top-N selection
     raises coverage versus the global curve).
  4. VRAM cost per hot-set size at 2.76 MB per expert per layer (NVFP4, 48 layers).

No tok/s anywhere. Bytes and raw transfer time only, transfer time labeled serialized.
"""

import glob
import math
import re
import statistics
import sys
from pathlib import Path

LAB_RUNS = Path(r"C:\Users\robin\dev\crow-lab\runs")
N_EXPERTS = 512
N_LAYERS = 48
ROUTED = 10
BYTES_PER_EXPERT_LAYER = 2_760_000  # 2.76 MB NVFP4 per expert per layer (research strand 1/3)
EXPERT_VRAM_BUDGET_GB = 23.5  # 32 GiB minus dense 4.5, KV FP8 ~3.0 GiB, GDN/QSA ~0.25, PLE rows ~1.0, pools ~2.0
CANDIDATE_N = [32, 64, 96, 128, 160, 176, 192, 256]

BLOCK_HEAD = re.compile(r"print_locality: (\w+): moe stream: expert locality over (\d+) layers, (\d+) experts each")
COVER = re.compile(
    r"print_locality: (\w+): moe stream:\s+50% of selections covered by ([\d.]+)% of experts, "
    r"80% by ([\d.]+)%, 95% by ([\d.]+)%"
)
GINI = re.compile(r"print_locality: (\w+): moe stream:\s+Gini = ([\d.]+)")


def parse_logs():
    blocks = []
    for log in sorted(LAB_RUNS.glob("*/*-server.log")):
        role = None
        entry = {}
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            m = BLOCK_HEAD.search(line)
            if m:
                role = m.group(1)
                entry = {"log": str(log.relative_to(LAB_RUNS)), "role": role}
                continue
            m = COVER.search(line)
            if m and m.group(1) == role:
                entry["pct50"], entry["pct80"], entry["pct95"] = (
                    float(m.group(2)), float(m.group(3)), float(m.group(4)))
                continue
            m = GINI.search(line)
            if m and m.group(1) == role:
                entry["gini"] = float(m.group(2))
                blocks.append(entry)
                entry = {}
    return [b for b in blocks if {"pct50", "pct80", "pct95", "gini"} <= set(b)]


def fit(points):
    """Least-squares c = a + b*ln(N) through (N_i, c_i)."""
    xs = [math.log(n) for n, _ in points]
    ys = [c for _, c in points]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    a = my - b * mx
    return a, b


def main():
    blocks = parse_logs()
    if not blocks:
        sys.exit("no print_locality blocks found")
    print(f"parsed {len(blocks)} print_locality blocks from {LAB_RUNS}\n")

    print("per-block measurements (experts = % of 512):")
    for blk in blocks:
        print(
            f"  {blk['log']:<42} 50%:{blk['pct50']:>5.1f}% ({blk['pct50']*N_EXPERTS/100:5.1f} experts)"
            f"  80%:{blk['pct80']:>5.1f}% ({blk['pct80']*N_EXPERTS/100:6.1f})"
            f"  95%:{blk['pct95']:>5.1f}% ({blk['pct95']*N_EXPERTS/100:6.1f})"
            f"  Gini {blk['gini']:.3f}"
        )

    med = lambda k: statistics.median(b[k] for b in blocks)
    points = [
        (med("pct50") / 100 * N_EXPERTS, 0.50),
        (med("pct80") / 100 * N_EXPERTS, 0.80),
        (med("pct95") / 100 * N_EXPERTS, 0.95),
    ]
    a, b = fit(points)
    print(f"\nmedian points: " + ", ".join(f"({n:.0f} experts, {c:.0%})" for n, c in points))
    print(f"fit: c(N) = {a:.4f} + {b:.4f} * ln(N)   (capped at 0.995)")

    budget_experts = EXPERT_VRAM_BUDGET_GB * 1e9 / (BYTES_PER_EXPERT_LAYER * N_LAYERS)
    print(
        f"expert VRAM budget {EXPERT_VRAM_BUDGET_GB:.1f} GB -> max {budget_experts:.0f} experts/layer"
        f" (48 x {BYTES_PER_EXPERT_LAYER/1e6:.2f} MB)\n"
    )

    print(f"{'N/layer':>7} {'VRAM GB':>8} {'coverage':>9} {'P(all 10)':>10} {'cold bytes/token':>17} {'serial H2D':>11}  fits")
    for n in CANDIDATE_N:
        c = min(0.995, a + b * math.log(n))
        p_skip = c ** ROUTED
        vram = n * BYTES_PER_EXPERT_LAYER * N_LAYERS / 1e9
        cold_bytes = (1 - c) * ROUTED * N_LAYERS * BYTES_PER_EXPERT_LAYER
        serial_ms = cold_bytes / 25e9 * 1000  # ~25 GB/s pinned H2D, serialized — an input, not a prediction
        print(
            f"{n:>7} {vram:>8.1f} {c:>9.1%} {p_skip:>10.1%} {cold_bytes/1e6:>13.0f} MB"
            f" {serial_ms:>9.1f} ms  {'YES' if n <= budget_experts else 'no'}"
        )

    print("\nassumptions (read before quoting any number above):")
    print("  1. coverage measured on Crow traffic through the UD-Q2_K_XL quant, aggregated over 48 layers;")
    print("     per-layer top-N residency typically achieves >= this global curve")
    print("  2. P(all 10 resident) = c^10 assumes independence; correlation within a layer lowers it,")
    print("     per-layer selection raises coverage — refine from engine telemetry, do not promise")
    print("  3. c(N) is an interpolation through three measured points, not a model")
    print("  4. 'serial H2D' is a bandwidth input (25 GB/s pinned), not a throughput prediction")


if __name__ == "__main__":
    main()
