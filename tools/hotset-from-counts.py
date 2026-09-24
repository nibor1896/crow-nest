"""Cut a hot set from `decode routestats` prefill counts, and score sidecars on
held-out counts. The companion of tools/hotset-calibrate.sh.

  python3 tools/hotset-from-counts.py cut <out-sidecar.json> <N> <counts.json>...
      sums the prefill_counts of every file, keeps the top N experts per layer
      in frequency order (ties: lower id, as `decode warmup`), and writes the
      sidecar in persist_sidecar's format, its slab bytes copied from the
      current default sidecar. Refuses to overwrite.
  python3 tools/hotset-from-counts.py score <counts.json> <sidecar.json>...
      the share of the held-out file's routed choices that land in each
      sidecar's set at N = 160 and at the N the engine actually keeps
      (read from the run's log, `n_hot`) -- 1 - cold/selections, the same
      quantity serve reports as hit_rate, for prefill routing.
"""
import json
import os
import sys

REFERENCE = "decode_out/hotsets-M-longctx2100-n160.json"


def counts_of(path):
    return json.load(open(path))["prefill_counts"]


def cut(out, n, files):
    if os.path.exists(out):
        sys.exit("%s exists - refusing to overwrite" % out)
    total = None
    for f in files:
        c = counts_of(f)
        total = c if total is None else [[a + b for a, b in zip(x, y)]
                                          for x, y in zip(total, c)]
    sets = [sorted(range(len(row)), key=lambda e: (-row[e], e))[:n] for row in total]
    ref = json.load(open(REFERENCE))
    doc = {k: v for k, v in ref.items() if k not in ("sets", "provenance", "n_per_layer")}
    doc["n_per_layer"] = n
    doc["provenance"] = ("routestats prefill counts summed over %s, top-%d/layer, "
                         "frequency order (tools/hotset-from-counts.py)"
                         % (", ".join(os.path.basename(f) for f in files), n))
    doc["sets"] = sets
    json.dump(doc, open(out, "w"))
    hit = sum(row[e] for row, s in zip(total, sets) for e in s)
    tot = sum(sum(row) for row in total)
    print("%s: in-sample coverage %.1f %% of %d selections" % (out, 100.0 * hit / tot, tot))


def score(counts_path, sidecars):
    held = counts_of(counts_path)
    run = json.load(open(counts_path))
    n_run = run.get("n_hot")
    tot = sum(sum(row) for row in held)
    print("held-out %s: %d selections, engine n_hot %s" % (counts_path, tot, n_run))
    for sc in sidecars:
        sets = json.load(open(sc))["sets"]
        line = []
        for n in sorted({160, n_run} - {None}):
            hit = sum(row[e] for row, s in zip(held, sets) for e in s[:n])
            line.append("N %d: %.3f" % (n, hit / tot))
        print("  %-60s %s" % (os.path.basename(sc), "  ".join(line)))


if __name__ == "__main__":
    if sys.argv[1] == "cut":
        cut(sys.argv[2], int(sys.argv[3]), sys.argv[4:])
    elif sys.argv[1] == "score":
        score(sys.argv[2], sys.argv[3:])
    else:
        sys.exit(__doc__)
