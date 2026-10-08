#!/usr/bin/env python3
"""#148 (GLM-5.3-Flash step 10, PREREG gate G2): the entropy census over the converter's sidecar.

  # the census and, only on a finished container, the G2 verdict
  python -I tools/glm_entropy_report.py census converter/GLM-5.3-Flash-CNQ4.5.cnq.sidecar.jsonl \
      [--decoder-draft runs/glm53-flash/step10/decoder-draft.json] [--block L:E ...] [--json out.json]

  # where one routed expert block sits in a finished container (absolute file offset, length)
  python -I tools/glm_entropy_report.py locate <container.cnq> --layer 3 --expert 0

What is counted (PREREG G2, `runs/glm53-flash/PREREG.md`): one static order-0 coder per routed expert
block (gate, up and down projection of one expert of one layer = 14,155,776 B raw), codes (16
classes) and scale bytes (256 classes) both coded, the code-length tables included (8 B + 128 B per
block), summed over all 12,096 blocks (42 MoE layers x 288). saving = 1 - coded / raw.

The coded size is the converter's (`converter/src/entropy.rs`): canonical Huffman, length-limited to
12 bits (codes) and 15 bits (scales), each stream rounded up to whole bytes. Only the optimal COST of
a length-limited prefix code enters a size, and that cost is unique (any optimal code has it), so this
tool computes it by the cost-only form of the same package-merge (Larmore & Hirschberg 1990): the sum
of the weights of the 2n-2 cheapest items after `limit` package/merge rounds. On a finished container
the total is checked against the converter's own `code_summary` `expert_blocks` record byte for byte;
a mismatch refuses the verdict.

Guards:
- every NVFP4 line: sum(h_codes) == n and sum(h_scales) == n / 16 (the step-5 test), or refused;
- every expert block has gate, up and down once, 25,165,824 values, or refused;
- the verdict needs the finished container: index trailer present and parseable (format_version 2,
  not partial), the resume journal gone (the converter removes it after the trailer,
  `converter/README.md` "Resume"), 12,096 blocks, the converter's `code_summary` lines present and
  equal. Before that the tool prints PROVISIONAL partial numbers and never a G2 line (exit 3).

Order-0 limit: the entropy bound is printed beside the coded size; a context coder may save more
and costs more decode time (PREREG G2). Stdlib only.
"""
import argparse
import heapq
import json
import math
import os
import struct
import sys

CODE_LEN_LIMIT = 12
SCALE_LEN_LIMIT = 15
CODE_TABLE_BYTES = 8
SCALE_TABLE_BYTES = 128
TABLE_BYTES = CODE_TABLE_BYTES + SCALE_TABLE_BYTES
EXPERT_VALUES = 3 * 4096 * 2048
BLOCK_RAW = EXPERT_VALUES // 2 + EXPERT_VALUES // 16  # 14,155,776
MOE_LAYERS = list(range(3, 45))
EXPERTS = 288
N_BLOCKS = len(MOE_LAYERS) * EXPERTS  # 12,096
THRESHOLD = 0.08
PINNED_PER_LAYER = 83
B_STEP3_1READER = 6.994e9  # step 3, 1 reader, spread 1.003 (runs/glm53-flash/step03/20261008T001819Z.md)
PROJ = ("expert_gate", "expert_up", "expert_down")
PROVISIONAL_EXIT = 3


class Refused(Exception):
    pass


# ---------------------------------------------------------------- coder sizes


def entropy_bits(h):
    n = sum(h)
    if n == 0:
        return 0.0
    return -sum(c * math.log2(c / n) for c in h if c > 0)


def huffman_cost_bits(h, limit):
    """Optimal length-limited prefix-code cost sum(count * length), package-merge, cost only.
    0 symbols -> 0; 1 symbol -> length 1 (the converter's convention)."""
    w = sorted(c for c in h if c > 0)
    n = len(w)
    if n == 0:
        return 0
    if n == 1:
        return w[0]
    if (1 << limit) < n:
        raise ValueError("%d symbols do not fit a %d-bit limit" % (n, limit))
    lst = list(w)
    for _ in range(1, limit):
        packages = [lst[i] + lst[i + 1] for i in range(0, len(lst) - 1, 2)]
        lst = list(heapq.merge(w, packages))
    return sum(lst[: 2 * n - 2])


def coded(hc, hs):
    """(raw, coded incl. tables, codes coded, scales coded, entropy bytes) of one table over hc/hs."""
    raw = sum(hc) // 2 + sum(hs)
    cc = (huffman_cost_bits(hc, CODE_LEN_LIMIT) + 7) // 8
    cs = (huffman_cost_bits(hs, SCALE_LEN_LIMIT) + 7) // 8
    ent = (entropy_bits(hc) + entropy_bits(hs)) / 8.0
    return {"raw": raw, "coded": cc + cs + TABLE_BYTES, "codes": cc, "scales": cs, "entropy": ent,
            "raw_codes": sum(hc) // 2, "raw_scales": sum(hs)}


class Acc:
    def __init__(self):
        self.raw = self.coded = self.codes = self.scales = self.raw_codes = self.raw_scales = 0
        self.tables = 0
        self.entropy = 0.0
        self.n = 0
        self.lo = math.inf
        self.hi = -math.inf

    def add(self, c):
        self.raw += c["raw"]
        self.coded += c["coded"]
        self.codes += c["codes"]
        self.scales += c["scales"]
        self.raw_codes += c["raw_codes"]
        self.raw_scales += c["raw_scales"]
        self.tables += TABLE_BYTES
        self.entropy += c["entropy"]
        self.n += 1
        s = 1 - c["coded"] / c["raw"]
        self.lo = min(self.lo, s)
        self.hi = max(self.hi, s)

    def saving(self):
        return 1 - self.coded / self.raw if self.raw else 0.0

    def row(self):
        return {"units": self.n, "raw_bytes": self.raw, "coded_bytes": self.coded, "table_bytes": self.tables,
                "coded_codes_bytes": self.codes, "coded_scales_bytes": self.scales,
                "entropy_bytes": self.entropy, "saving": self.saving(),
                "saving_entropy": 1 - self.entropy / self.raw if self.raw else 0.0,
                "saving_codes": 1 - self.codes / self.raw_codes if self.raw_codes else 0.0,
                "saving_scales": 1 - self.scales / self.raw_scales if self.raw_scales else 0.0,
                "saving_min": self.lo, "saving_max": self.hi}


def add_hist(a, b):
    if a is None:
        return list(b)
    return [x + y for x, y in zip(a, b)]


# ---------------------------------------------------------------- container state


def read_trailer(cnq):
    """The index trailer as a dict, or None while the converter is still writing (no valid trailer)."""
    try:
        size = os.path.getsize(cnq)
        if size < 20:
            return None
        with open(cnq, "rb") as f:
            f.seek(size - 8)
            n = struct.unpack("<Q", f.read(8))[0]
            if n == 0 or n > size - 20 or n > (1 << 31):
                return None
            f.seek(size - 8 - n)
            ix = json.loads(f.read(n).decode("utf-8"))
    except (OSError, ValueError, UnicodeDecodeError):
        return None
    if not isinstance(ix, dict) or ix.get("format_version") != 2 or not isinstance(ix.get("tensors"), list):
        return None
    return ix


def container_state(sidecar):
    suffix = ".sidecar.jsonl"
    if not sidecar.endswith(suffix):
        raise Refused("sidecar name must end in %s: %s" % (suffix, sidecar))
    cnq = sidecar[: -len(suffix)]
    journal = cnq + ".journal.jsonl"
    st = {"container": cnq, "container_exists": os.path.exists(cnq),
          "container_bytes": os.path.getsize(cnq) if os.path.exists(cnq) else None,
          "journal_exists": os.path.exists(journal)}
    ix = read_trailer(cnq) if st["container_exists"] else None
    st["trailer"] = ix is not None
    st["partial"] = bool(ix and ix.get("partial"))
    st["index"] = ix
    return st


# ---------------------------------------------------------------- census


def read_sidecar(path, tolerate_tail):
    lines, bad_tail = [], False
    with open(path, "rb") as f:
        data = f.read()
    raw_lines = data.split(b"\n")
    if raw_lines and raw_lines[-1] == b"":
        raw_lines.pop()
    for i, ln in enumerate(raw_lines):
        try:
            lines.append(json.loads(ln))
        except ValueError:
            if tolerate_tail and i == len(raw_lines) - 1:
                bad_tail = True  # a line the running converter is writing right now
                continue
            raise Refused("sidecar line %d is not JSON" % (i + 1))
    return lines, bad_tail


def census(lines, want_blocks=()):
    blocks = {}  # (layer, expert) -> {class: (hc, hs)}
    shared = {}  # layer -> {name: (hc, hs)}
    per_tensor = {}  # class -> Acc (one table per tensor)
    pooled = {}  # class -> [hc, hs] (one table per class)
    summaries = []
    nv_lines = 0
    for r in lines:
        if r.get("record") == "code_summary":
            summaries.append(r)
            continue
        if r.get("record") or "h_codes" not in r:
            continue
        nv_lines += 1
        hc, hs, n = r["h_codes"], r["h_scales"], r["n"]
        if len(hc) != 16 or len(hs) != 256:
            raise Refused("%s: histogram widths %d/%d, not 16/256" % (r.get("name"), len(hc), len(hs)))
        if sum(hc) != n or sum(hs) * 16 != n:
            raise Refused("%s: histogram sum %d codes / %d scales != value count %d (step-5 test)"
                          % (r.get("name"), sum(hc), sum(hs), n))
        cls = r["class"]
        per_tensor.setdefault(cls, Acc()).add(coded(hc, hs))
        p = pooled.setdefault(cls, [None, None])
        p[0], p[1] = add_hist(p[0], hc), add_hist(p[1], hs)
        if cls in PROJ:
            key = (r["layer"], r["expert"])
            b = blocks.setdefault(key, {})
            if cls in b:
                raise Refused("expert block L%d E%d has %s twice" % (key[0], key[1], cls))
            b[cls] = (hc, hs)
        elif cls == "shared_expert":
            shared.setdefault(r["layer"], {})[r["name"]] = (hc, hs)
    total, by_layer, by_block = Acc(), {}, {}
    incomplete = []
    for key in sorted(blocks):
        b = blocks[key]
        if set(b) != set(PROJ):
            incomplete.append(key)
            continue
        hc = hs = None
        for c in PROJ:
            hc, hs = add_hist(hc, b[c][0]), add_hist(hs, b[c][1])
        c = coded(hc, hs)
        if c["raw"] != BLOCK_RAW:
            raise Refused("expert block L%d E%d: %d raw bytes, not %d" % (key[0], key[1], c["raw"], BLOCK_RAW))
        total.add(c)
        by_layer.setdefault(key[0], Acc()).add(c)
        if key in want_blocks:
            by_block[key] = c
    sh = Acc()
    for layer in sorted(shared):
        t = shared[layer]
        if len(t) != 3:
            continue
        hc = hs = None
        for hc1, hs1 in t.values():
            hc, hs = add_hist(hc, hc1), add_hist(hs, hs1)
        sh.add(coded(hc, hs))
    pooled_rows = {}
    for cls, (hc, hs) in pooled.items():
        a = Acc()
        a.add(coded(hc, hs))
        pooled_rows[cls] = a.row()
    return {"nvfp4_lines": nv_lines, "total": total, "by_layer": by_layer, "by_block": by_block,
            "incomplete_blocks": incomplete, "shared": sh,
            "per_tensor": {k: v.row() for k, v in sorted(per_tensor.items())},
            "pooled": dict(sorted(pooled_rows.items())), "summaries": summaries}


def check_against_converter(res):
    """The converter's own expert_blocks record must equal ours (sizes exact, entropy to 1e-6 rel)."""
    rec = [s for s in res["summaries"] if s.get("scope") == "expert_blocks"]
    if not rec:
        return "no code_summary expert_blocks record in the sidecar"
    r, t = rec[0], res["total"]
    for k, mine in (("blocks", t.n), ("raw_bytes", t.raw), ("coded_bytes", t.coded), ("table_bytes", t.tables)):
        if r.get(k) != mine:
            return "converter %s %s != census %s" % (k, r.get(k), mine)
    if abs(r["entropy_bytes"] - t.entropy) > 1e-6 * t.entropy:
        return "converter entropy_bytes %r != census %r" % (r["entropy_bytes"], t.entropy)
    return None


def effects(saving):
    return {"pinned_experts_per_layer": PINNED_PER_LAYER / (1 - saving),
            "bytes_per_nvme_visit": BLOCK_RAW * (1 - saving),
            "nvme_ms_raw": BLOCK_RAW / B_STEP3_1READER * 1e3,
            "nvme_ms_coded": BLOCK_RAW * (1 - saving) / B_STEP3_1READER * 1e3,
            "nvme_ms_saved": BLOCK_RAW * saving / B_STEP3_1READER * 1e3}


def verdict(saving, draft):
    """G2 per PREREG: saving >= 8 % incl. tables AND a decoder draft naming the decode time per cold visit."""
    named = bool(draft) and isinstance(draft.get("decode_ms_per_cold_visit"), (int, float))
    if saving < THRESHOLD:
        return "failed", "saving %.3f %% < 8 %% incl. tables" % (100 * saving)
    if not named:
        return "not answered", "saving %.3f %% >= 8 %%, but no decoder draft names the decode time" % (100 * saving)
    return "passed", "saving %.3f %% >= 8 %% incl. tables and decode time %.3f ms per cold visit named" % (
        100 * saving, draft["decode_ms_per_cold_visit"])


def pct(x):
    return "%.3f %%" % (100 * x)


def cmd_census(a):
    st = container_state(a.sidecar)
    finished = st["trailer"] and not st["journal_exists"] and not st["partial"]
    lines, bad_tail = read_sidecar(a.sidecar, tolerate_tail=not finished)
    want = set()
    for b in a.block or ():
        l, e = b.split(":")
        want.add((int(l), int(e)))
    res = census(lines, want)
    t = res["total"]
    reasons = []
    if not st["trailer"]:
        reasons.append("container has no index trailer (still writing)")
    if st["journal_exists"]:
        reasons.append("resume journal still present (conversion running or interrupted)")
    if st["partial"]:
        reasons.append("container is partial (%s)" % st["index"]["partial"].get("filter"))
    if t.n != N_BLOCKS:
        reasons.append("%d of %d expert blocks complete" % (t.n, N_BLOCKS))
    if res["incomplete_blocks"]:
        reasons.append("%d expert blocks missing a projection" % len(res["incomplete_blocks"]))
    has_summary = any(s.get("scope") == "expert_blocks" for s in res["summaries"])
    mismatch = check_against_converter(res) if (has_summary or not reasons) else None
    if mismatch:
        reasons.append(mismatch)
    if not reasons:
        ix = st["index"]
        n_ix = sum(1 for x in ix["tensors"] if ".mlp.experts." in x["name"] and x["dtype"] == "nvfp4")
        if n_ix != 3 * N_BLOCKS:
            reasons.append("index has %d routed-expert NVFP4 tensors, not %d" % (n_ix, 3 * N_BLOCKS))
    final = not reasons
    draft = None
    if a.decoder_draft:
        with open(a.decoder_draft, encoding="utf-8") as f:
            draft = json.load(f)

    tag = "" if final else "PROVISIONAL "
    out = []
    p = out.append
    p("%sentropy census of %s" % (tag, a.sidecar))
    p("container: %s, %s B, trailer %s, journal %s" % (
        st["container"], st["container_bytes"], "present" if st["trailer"] else "absent",
        "present" if st["journal_exists"] else "absent"))
    p("NVFP4 sidecar lines: %d, every histogram sum == value count%s" % (
        res["nvfp4_lines"], " (one unfinished tail line skipped)" if bad_tail else ""))
    if not final:
        p("PROVISIONAL, not G2: " + "; ".join(reasons))
    if t.n:
        r = t.row()
        p("%srouted expert blocks: %d x %d B, one static order-0 table per block (Huffman, 12/15-bit limit)" % (
            tag, t.n, BLOCK_RAW))
        p("  raw %d B, coded %d B incl. %d B tables: saving %s" % (t.raw, t.coded, t.tables, pct(r["saving"])))
        p("  codes %d -> %d B (%s), scales %d -> %d B (%s), tables %d B" % (
            t.raw_codes, t.codes, pct(r["saving_codes"]), t.raw_scales, t.scales, pct(r["saving_scales"]), t.tables))
        p("  order-0 entropy bound (no tables, no rounding): %s; a context coder may save more and costs more decode time"
          % pct(r["saving_entropy"]))
        p("  scales-only variant (codes raw, one 128-B table per block): %s" % pct(
            1 - (t.raw_codes + t.scales + SCALE_TABLE_BYTES * t.n) / t.raw))
        p("  per block min-max: %s .. %s" % (pct(t.lo), pct(t.hi)))
        ls = {l: a_.saving() for l, a_ in res["by_layer"].items()}
        lo_l, hi_l = min(ls, key=ls.get), max(ls, key=ls.get)
        p("  per layer min-max: %s (L%d) .. %s (L%d) over %d layers" % (pct(ls[lo_l]), lo_l, pct(ls[hi_l]), hi_l, len(ls)))
        for l in sorted(res["by_layer"]):
            x = res["by_layer"][l]
            p("    L%-2d %3d blocks  saving %s  (blocks %s .. %s)" % (l, x.n, pct(x.saving()), pct(x.lo), pct(x.hi)))
        e = effects(r["saving"])
        p("  effect: pinned tier %.1f instead of %d experts per layer; %.0f B per NVMe visit = %.3f ms instead of %.3f ms at B = 6.994 GB/s (step 3, 1 reader; not the G1 B, which is not answered)" % (
            e["pinned_experts_per_layer"], PINNED_PER_LAYER, e["bytes_per_nvme_visit"], e["nvme_ms_coded"], e["nvme_ms_raw"]))
    for k in sorted(res["by_block"]):
        c = res["by_block"][k]
        p("  block L%d E%d: raw %d B, codes %d B, scales %d B, tables %d B, coded %d B, saving %s" % (
            k[0], k[1], c["raw"], c["codes"], c["scales"], TABLE_BYTES, c["coded"], pct(1 - c["coded"] / c["raw"])))
    p("%sper class, one table per tensor (min-max over tensors) | one table per class:" % tag)
    for cls, r in res["per_tensor"].items():
        p("  %-14s %5d tensors  %s (%s .. %s) | %s" % (cls, r["units"], pct(r["saving"]), pct(r["saving_min"]),
                                                     pct(r["saving_max"]), pct(res["pooled"][cls]["saving"])))
    if res["shared"].n:
        r = res["shared"].row()
        p("%sshared expert blocks (not in the 12,096): %d, saving %s (%s .. %s)" % (
            tag, r["units"], pct(r["saving"]), pct(r["saving_min"]), pct(r["saving_max"])))
    v = None
    if has_summary and not mismatch:
        p("converter code_summary expert_blocks: equal (blocks, raw, coded, table bytes; entropy)")
    if final:
        v = verdict(t.saving(), draft)
        p("G2 %s: %s" % v)
        if draft:
            p("decoder draft: %s" % draft.get("summary", ""))
    print("\n".join(out))
    if a.json:
        doc = {"final": final, "reasons": reasons, "sidecar": a.sidecar,
               "container": {k: v_ for k, v_ in st.items() if k != "index"},
               "nvfp4_lines": res["nvfp4_lines"], "skipped_tail_line": bad_tail,
               "expert_blocks": t.row() if t.n else None,
               "expert_blocks_layer": {str(l): x.row() for l, x in sorted(res["by_layer"].items())},
               "per_class_tensor_tables": res["per_tensor"], "per_class_pooled": res["pooled"],
               "shared_expert_blocks": res["shared"].row() if res["shared"].n else None,
               "effects": effects(t.saving()) if t.n else None,
               "blocks": {"L%d:E%d" % k: c for k, c in sorted(res["by_block"].items())},
               "g2": {"verdict": v[0], "why": v[1]} if v else None,
               "decoder_draft": a.decoder_draft}
        with open(a.json, "w", encoding="utf-8", newline="\n") as f:
            json.dump(doc, f, indent=1)
            f.write("\n")
    return 0 if final else PROVISIONAL_EXIT


def cmd_locate(a):
    ix = read_trailer(a.container)
    if ix is None:
        raise Refused("%s has no index trailer" % a.container)
    base = ix.get("blob_offset", 12)
    pre = "model.language_model.layers.%d.mlp.experts.%d." % (a.layer, a.expert)
    t = {x["name"]: x for x in ix["tensors"] if x["name"].startswith(pre)}
    parts = [t.get(pre + p + "_proj.weight") for p in ("gate", "up", "down")]
    if None in parts:
        raise Refused("expert L%d E%d not in the index" % (a.layer, a.expert))
    for x, y in zip(parts, parts[1:]):
        if x["offset"] + x["len"] != y["offset"]:
            raise Refused("expert L%d E%d: projections are not back to back" % (a.layer, a.expert))
    off = base + parts[0]["offset"]
    length = sum(x["len"] for x in parts)
    print(json.dumps({"container": a.container, "layer": a.layer, "expert": a.expert,
                      "offset": off, "len": length, "aligned_4096": off % 4096 == 0}))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("census")
    c.add_argument("sidecar")
    c.add_argument("--decoder-draft")
    c.add_argument("--block", action="append", help="L:E, print that block's sizes (e.g. 3:0)")
    c.add_argument("--json")
    l = sub.add_parser("locate")
    l.add_argument("container")
    l.add_argument("--layer", type=int, required=True)
    l.add_argument("--expert", type=int, required=True)
    a = ap.parse_args(argv)
    try:
        return cmd_census(a) if a.cmd == "census" else cmd_locate(a)
    except Refused as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
