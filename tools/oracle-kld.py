#!/usr/bin/env python3
"""The reference-based reading of a quantization: top-1 agreement and KL divergence
against an f32 oracle, in llama.cpp's units (issue #78, step 4 of the requant series).

    tools/oracle-kld.py --ref decode_out/oracle-tf298/ref-logits.f32 \
        --rows 0:298 --prompt-rows 235 \
        --arm none=decode_out/oracle-tf298/none/gpu-logits.f32 \
        --arm orig=decode_out/oracle-tf298/orig/gpu-logits.f32 \
        --paired orig,none

stdlib only, no numpy, no GPU, no server. It reads flat `rows x vocab` f32 little-endian
logit dumps - the form `decode parity` writes and the form the 2026-09-05 transformers f32
oracle was stored in - and reports, per arm, exactly the statistics llama.cpp's
`tools/perplexity` prints for `--kl-divergence`, so a number here is comparable IN KIND with
the rows of that tool's README scoreboard.

WHAT IS COPIED FROM llama.cpp, and where it is in that source
(`~/.local/share/crow/src/llama.cpp`, pin 6c84c7d5d + PR #27880 + PR #28040):

  * the KL divergence itself, `tools/perplexity/perplexity.cpp:222-231`:
        KLD = sum_i p_ref(i) * (log p_ref(i) - log q_arm(i))
    over every i whose REFERENCE log-probability is above -16 in the natural log. It is
    KL(P_reference || Q_arm) - the reference is the first argument, the arm the second -
    and nothing here is symmetric. Natural log, nats, never bits.
  * the -16 support cut, same lines (`if (p_log_base > -16.f)`). `--min-log-prob` moves it.
  * the uncertainty on a mean, `perplexity.cpp:1770-1778`: sqrt((sum2/n - mean^2)/(n-1)),
    and zero when n <= 10 - so a mean over ten rows or fewer is printed without one.
  * the percentile, `perplexity.cpp:1954-1960`: linear interpolation at fraction*(n-1),
    NOT a nearest-rank percentile. The median is the mean of the two middle values when n
    is even (`perplexity.cpp:1951`).
  * same top-1 and its uncertainty, `perplexity.cpp:2004-2005`: the Gaussian approximation
    of the binomial, sqrt(p(1-p)/(n-1)).

WHAT IS NOT llama.cpp's: its `Delta p` is the change of the probability of the CORPUS's next
token; there is no corpus here, only a token sequence the reference itself was teacher-forced
on, so `--dp` here is the change of the probability the arm gives the REFERENCE's top-1
token. Positive means the arm is more certain of it than the reference was; the two columns
that matter are the share of rows that lost more than 10 and more than 50 percentage points
of it. The name is kept because the quantity is the same kind of quantity; the definition is
spelled out in every report this tool prints.

TWO ARM KINDS, ONE ESTIMATOR. `--arm NAME=PATH` is a full logit dump. `--topn-arm NAME=PATH`
is a JSON file of per-row top-N probabilities - what llama-server's `/completion` returns for
`n_probs`, which is the only honest way to get a per-position distribution out of a GGUF
without a second engine on the card. A top-N arm cannot answer for a token outside its N, so
BOTH kinds are then run through the same TRUNCATED estimator: `--truncate M` restricts the
sum to the reference's M most probable tokens, `--arm-topn N` gives a full dump the same
blindness beyond its own N, and a token the arm cannot answer for is charged at the arm's
smallest known probability - which makes the reported KLD a LOWER BOUND, never an estimate
from above. The report names the reference mass the truncation covers and how often the
blindness actually bound.

SPARSE DUMPS (#90). A source whose path carries a `<path>.rows.json` sidecar is read as a
SPARSE dump: the sidecar lists the ABSOLUTE row ids present, in file order, and the f32 file
holds exactly those rows. That is the long-context form - 178k positions of context with 64
rows sampled per depth - and it is what `oracle/ref_longctx_logits.py` writes. `--rows A:B`
then bounds ABSOLUTE row ids and the run collects the rows of the intersection; an arm has to
carry every row the reference asked for, exactly as a top-N arm does. A dense dump with no
sidecar is read exactly as before, byte for byte.

ROW GROUPS AND THE POSITION CURVE (#90). `--row-groups FILE` names groups of absolute rows -
the depth blocks of `decode_out/oracle-longctx/row-plan.json`, the per-text groups of
`decode_out/oracle-en/en-row-groups.json`, or any {"groups": [{"name", "rows"}]} file - and
the report gains one block per group per arm: the per-corpus / per-depth summary.
`--kld-vs-position N` bins the collected rows by ABSOLUTE position into N equal-width bins
and prints mean KLD and same-top-1 per bin per arm: the KLD-vs-position curve. Both land in
`--json` under "groups" and "position_curve".
"""

import argparse
import array
import json
import math
import os
import sys

VOCAB_DEFAULT = 248320
MIN_LOG_PROB = -16.0      # perplexity.cpp:222
LSE_CUTOFF = 60.0         # exp(-60) = 8.8e-27; dropping those terms is below f64 resolution


# --------------------------------------------------------------------------- pure numerics

def log_sum_exp(logits, cutoff=LSE_CUTOFF):
    """log sum_i exp(logit_i), the shift that turns logits into log-probabilities."""
    m = max(logits)
    thr = m - cutoff
    s = sum(map(math.exp, [x - m for x in logits if x > thr]))
    return m + math.log(s)


def argmax(logits):
    """The FIRST index carrying the maximum - llama.cpp's loop keeps the first too."""
    m = max(logits)
    return logits.index(m)


def support_indices(logits, lse, min_log_prob=MIN_LOG_PROB):
    """The i with log p(i) > min_log_prob, in index order (perplexity.cpp:222)."""
    thr = lse + min_log_prob
    return [i for i, x in enumerate(logits) if x > thr]


def truncate_support(logits, sup, m):
    """The m entries of `sup` with the largest logit, largest first."""
    if m is None or m <= 0 or m >= len(sup):
        return sup
    return sorted(sup, key=lambda i: logits[i], reverse=True)[:m]


def nth_largest(logits, n):
    """The n-th largest value of `logits` (n >= 1), or None if n exceeds its length."""
    if n is None or n <= 0 or n > len(logits):
        return None
    return sorted(logits, reverse=True)[n - 1]


def percentile(values_sorted, fraction):
    """perplexity.cpp:1954-1960 - linear interpolation at fraction*(n-1)."""
    if not values_sorted:
        raise ValueError("percentile of an empty sample")
    if fraction <= 0:
        return values_sorted[0]
    if fraction >= 1:
        return values_sorted[-1]
    p = fraction * (len(values_sorted) - 1)
    ip = int(p)
    p -= ip
    return (1 - p) * values_sorted[ip] + p * values_sorted[min(ip + 1, len(values_sorted) - 1)]


def median(values_sorted):
    """perplexity.cpp:1951 - the mean of the two middle values when n is even."""
    n = len(values_sorted)
    if n == 0:
        raise ValueError("median of an empty sample")
    if n % 2 == 0:
        return 0.5 * (values_sorted[n // 2] + values_sorted[n // 2 - 1])
    return values_sorted[n // 2]


def mean_and_uncertainty(values):
    """perplexity.cpp:1770-1778. Returns (mean, uncertainty); uncertainty 0 for n <= 10."""
    n = len(values)
    if n < 1:
        return (0.0, 0.0)
    f = sum(values) / n
    df = sum(v * v for v in values) / n - f * f
    df = math.sqrt(df / (n - 1)) if (df > 0 and n > 10) else 0.0
    return (f, df)


def share_and_uncertainty(k, n):
    """perplexity.cpp:2004-2005 - a share with the Gaussian binomial uncertainty."""
    if n < 1:
        return (0.0, 0.0)
    p = k / n
    if n < 2:
        return (p, 0.0)
    return (p, math.sqrt(p * (1.0 - p) / (n - 1)))


def sign_test(diffs, eps=0.0):
    """Two-sided exact sign test at p = 1/2. Returns (n_pos, n_neg, n_tie, p_value)."""
    pos = sum(1 for d in diffs if d > eps)
    neg = sum(1 for d in diffs if d < -eps)
    tie = len(diffs) - pos - neg
    n = pos + neg
    if n == 0:
        return (pos, neg, tie, 1.0)
    k = min(pos, neg)
    tail = sum(math.comb(n, i) for i in range(k + 1))
    return (pos, neg, tie, min(1.0, 2.0 * tail / (1 << n)))


# ------------------------------------------------------------------------------ arm rows

class FullRow:
    """One row of an arm that dumped the complete logit vector."""

    __slots__ = ("logits", "lse", "top1", "floor_logit")

    def __init__(self, logits, arm_topn=None):
        self.logits = logits
        self.lse = log_sum_exp(logits)
        self.top1 = argmax(logits)
        self.floor_logit = nth_largest(logits, arm_topn) if arm_topn else None

    def log_q(self, i):
        x = self.logits[i]
        if self.floor_logit is not None and x < self.floor_logit:
            return None
        return x - self.lse

    def floor_log_q(self):
        if self.floor_logit is None:
            return None
        return self.floor_logit - self.lse

    def known_mass(self):
        if self.floor_logit is None:
            return 1.0
        thr = self.floor_logit
        return sum(math.exp(x - self.lse) for x in self.logits if x >= thr)


class TopNRow:
    """One row of an arm that could only report its top N probabilities."""

    __slots__ = ("probs", "top1", "min_p", "mass", "n")

    def __init__(self, ids, probs):
        if not ids:
            raise ValueError("a top-N row with no entries")
        self.probs = dict(zip(ids, probs))
        self.top1 = max(zip(probs, ids))[1] if len(ids) > 1 else ids[0]
        self.min_p = min(p for p in probs if p > 0.0)
        self.mass = sum(probs)
        self.n = len(ids)

    def log_q(self, i):
        q = self.probs.get(i)
        if q is None or q <= 0.0:
            return None
        return math.log(q)

    def floor_log_q(self):
        return math.log(self.min_p)

    def known_mass(self):
        return self.mass


def row_kld(ref_logits, ref_lse, sup, arm):
    """KL(P_ref || Q_arm) over `sup`. Returns (kld, n_missing, ref_mass_of_sup)."""
    total = 0.0
    missing = 0
    mass = 0.0
    floor = arm.floor_log_q()
    for i in sup:
        lp = ref_logits[i] - ref_lse
        p = math.exp(lp)
        mass += p
        lq = arm.log_q(i)
        if lq is None:
            if floor is None:
                raise ValueError("an arm with a complete vector reported no value for %d" % i)
            lq = floor
            missing += 1
        total += p * (lp - lq)
    return (total, missing, mass)


# ---------------------------------------------------------------------------- file access

class LogitFile:
    """A flat `rows x vocab` f32 little-endian dump, read one row at a time."""

    def __init__(self, path, vocab):
        self.path = path
        self.vocab = vocab
        size = os.path.getsize(path)
        stride = vocab * 4
        if size % stride:
            raise SystemExit("%s: %d bytes is not a whole number of %d-wide f32 rows"
                             % (path, size, vocab))
        self.rows = size // stride
        self.handle = open(path, "rb")

    def read(self, row):
        if row >= self.rows:
            raise SystemExit("%s: row %d beyond the %d rows in the file" % (self.path, row, self.rows))
        self.handle.seek(row * self.vocab * 4)
        a = array.array("f")
        a.fromfile(self.handle, self.vocab)
        if sys.byteorder != "little":
            a.byteswap()
        return a

    def close(self):
        self.handle.close()


class SparseLogitFile:
    """A dump of SAMPLED rows (#90): the f32 file plus a `.rows.json` sidecar that lists
    the absolute row ids present, in file order. read(row) is by ABSOLUTE row id."""

    def __init__(self, path, vocab):
        side = path + ".rows.json"
        with open(side, "r", encoding="utf-8") as fh:
            doc = json.load(fh)
        rows = [int(r) for r in doc["rows"]]
        if not rows:
            raise SystemExit("%s lists no rows" % side)
        if any(rows[i] >= rows[i + 1] for i in range(len(rows) - 1)):
            raise SystemExit("%s is not strictly increasing - the sparse form is written "
                             "in ascending row order" % side)
        self.rows = rows
        self.index = {r: i for i, r in enumerate(rows)}
        self.file = LogitFile(path, vocab)
        self.sparse = True

    def read(self, row):
        i = self.index.get(row)
        if i is None:
            raise SystemExit("this sparse dump has no row %d" % row)
        return self.file.read(i)

    def close(self):
        self.file.close()


def open_source(path, vocab):
    """A dump source: sparse when the `.rows.json` sidecar exists, dense otherwise."""
    if os.path.exists(path + ".rows.json"):
        return SparseLogitFile(path, vocab)
    src = LogitFile(path, vocab)
    src.sparse = False
    return src


def load_row_groups(path):
    """The #90 row-groups file: {"groups": [{"name": ..., "rows": [...]}]} - which is
    also the shape of decode_out/oracle-longctx/row-plan.json."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    groups = doc["groups"] if isinstance(doc, dict) else doc
    out = []
    for g in groups:
        out.append((str(g["name"]), [int(r) for r in g["rows"]]))
    if not out:
        raise SystemExit("%s names no row groups" % path)
    return out


def load_topn(path):
    """Read a top-N probability file. Returns {row index: (ids, probs)} and its header."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    rows = {}
    for entry in doc["rows"]:
        rows[int(entry["row"])] = (entry["ids"], entry["probs"])
    return rows, doc


def parse_range(text, total):
    """`A:B` (half open), `A:` or `:B`. Returns (first, last_exclusive)."""
    if text is None:
        return (0, total)
    if ":" not in text:
        raise SystemExit("--rows wants A:B, got %r" % text)
    a, b = text.split(":", 1)
    first = int(a) if a.strip() else 0
    last = int(b) if b.strip() else total
    if not (0 <= first < last <= total):
        raise SystemExit("--rows %r is not inside 0:%d" % (text, total))
    return (first, last)


def parse_named(pairs, flag):
    out = []
    for item in pairs or []:
        if "=" not in item:
            raise SystemExit("%s wants NAME=PATH, got %r" % (flag, item))
        name, path = item.split("=", 1)
        out.append((name, path))
    return out


# ------------------------------------------------------------------------------- the run

class Accumulator:
    def __init__(self, name, kind):
        self.name = name
        self.kind = kind
        self.rows = []          # row index
        self.kld = []
        self.same_top1 = []     # bool per row
        self.dp = []            # q_arm(top1_ref) - p_ref(top1_ref)
        self.missing = []       # support entries the arm could not answer for
        self.sup_mass = []      # reference mass the support carried
        self.arm_mass = []      # the arm's own known mass


def collect(ref_path, vocab, first, last, full_arms, topn_arms, min_log_prob,
            truncate, arm_topn, per_row_path=None):
    ref = open_source(ref_path, vocab)
    if ref.sparse:
        rows_to_do = [r for r in ref.rows if first <= r < last]
        if not rows_to_do:
            raise SystemExit("--rows %d:%d matches no row of the sparse reference %s "
                             "(it carries %d rows, %d..%d)"
                             % (first, last, ref_path, len(ref.rows), ref.rows[0],
                                ref.rows[-1]))
    else:
        if last > ref.rows:
            raise SystemExit("--rows asks for %d rows, the reference has %d" % (last, ref.rows))
        rows_to_do = list(range(first, last))

    files = [(name, open_source(path, vocab)) for name, path in full_arms]
    for name, handle in files:
        if handle.sparse:
            missing = [r for r in rows_to_do if r not in handle.index]
            if missing:
                raise SystemExit("sparse arm %s has no row %d of the %d the reference asks "
                                 "for" % (name, missing[0], len(rows_to_do)))
        elif last > handle.rows:
            raise SystemExit("arm %s has %d rows, the range asks for %d" % (name, handle.rows, last))
    topn = []
    for name, path in topn_arms:
        rows, header = load_topn(path)
        topn.append((name, rows, header))

    acc = {}
    for name, _ in full_arms:
        acc[name] = Accumulator(name, "full")
    for name, _, header in topn:
        acc[name] = Accumulator(name, "top-%d" % int(header.get("n_probs", 0)))

    per_row = []
    for row in rows_to_do:
        p = ref.read(row)
        lse = log_sum_exp(p)
        top1 = argmax(p)
        sup = support_indices(p, lse, min_log_prob)
        sup = truncate_support(p, sup, truncate)
        p_top1 = math.exp(p[top1] - lse)
        record = {"row": row, "ref_top1": top1, "ref_p_top1": p_top1,
                  "support": len(sup), "arms": {}}

        for name, handle in files:
            arm = FullRow(handle.read(row), arm_topn)
            _score(acc[name], row, p, lse, sup, top1, p_top1, arm, record)
        for name, rows, _header in topn:
            if row not in rows:
                raise SystemExit("arm %s has no row %d" % (name, row))
            ids, probs = rows[row]
            arm = TopNRow(ids, probs)
            _score(acc[name], row, p, lse, sup, top1, p_top1, arm, record)
        per_row.append(record)

    ref.close()
    for _, handle in files:
        handle.close()
    if per_row_path:
        with open(per_row_path, "w", encoding="utf-8") as fh:
            json.dump({"reference": ref_path, "vocab": vocab, "first_row": first,
                       "last_row": last, "min_log_prob": min_log_prob,
                       "truncate": truncate, "arm_topn": arm_topn, "rows": per_row},
                      fh, indent=1)
    return acc, per_row


def _score(a, row, p, lse, sup, top1, p_top1, arm, record):
    kld, missing, mass = row_kld(p, lse, sup, arm)
    lq = arm.log_q(top1)
    if lq is None:
        lq = arm.floor_log_q()
    q_top1 = math.exp(lq) if lq is not None else 0.0
    a.rows.append(row)
    a.kld.append(kld)
    a.same_top1.append(arm.top1 == top1)
    a.dp.append(q_top1 - p_top1)
    a.missing.append(missing)
    a.sup_mass.append(mass)
    a.arm_mass.append(arm.known_mass())
    record["arms"][a.name] = {"kld": kld, "top1": arm.top1, "same_top1": arm.top1 == top1,
                              "dp": q_top1 - p_top1, "missing": missing}


def summarize(a, mask=None):
    """The statistics of one arm over the rows selected by `mask` (a list of bool)."""
    idx = [i for i in range(len(a.kld)) if (mask is None or mask[i])]
    n = len(idx)
    if n == 0:
        return None
    kld = sorted(a.kld[i] for i in idx)
    dp = [a.dp[i] for i in idx]
    same = sum(1 for i in idx if a.same_top1[i])
    mean, unc = mean_and_uncertainty([a.kld[i] for i in idx])
    dp_mean, dp_unc = mean_and_uncertainty(dp)
    share, share_unc = share_and_uncertainty(same, n)
    return {
        "n": n,
        "same_top1": same,
        "same_top1_share": share,
        "same_top1_unc": share_unc,
        "kld_mean": mean,
        "kld_unc": unc,
        "kld_median": median(kld),
        "kld_p90": percentile(kld, 0.90),
        "kld_p99": percentile(kld, 0.99),
        "kld_p999": percentile(kld, 0.999),
        "kld_max": kld[-1],
        "kld_min": kld[0],
        "dp_mean": dp_mean,
        "dp_unc": dp_unc,
        "dp_lost_10pp": sum(1 for d in dp if d < -0.10),
        "dp_lost_50pp": sum(1 for d in dp if d < -0.50),
        "rows_truncated": sum(1 for i in idx if a.missing[i] > 0),
        "support_mass_mean": sum(a.sup_mass[i] for i in idx) / n,
        "arm_mass_mean": sum(a.arm_mass[i] for i in idx) / n,
    }


def paired(a, b):
    """Per-row KLD difference a - b, its mean with a standard error, and a sign test."""
    diffs = [x - y for x, y in zip(a.kld, b.kld)]
    mean, unc = mean_and_uncertainty(diffs)
    pos, neg, tie, pval = sign_test(diffs)
    return {"a": a.name, "b": b.name, "n": len(diffs), "mean": mean, "unc": unc,
            "a_above": pos, "b_above": neg, "ties": tie, "sign_test_p": pval,
            "median": median(sorted(diffs))}


# --------------------------------------------------------------------------------- report

def fmt_arm_line(name, s):
    return ("%-22s %5d  %6.2f +- %4.2f %%  %9.6f +- %8.6f  %8.5f %8.4f %8.4f %8.4f %9.4f"
            % (name, s["n"], 100 * s["same_top1_share"], 100 * s["same_top1_unc"],
               s["kld_mean"], s["kld_unc"], s["kld_median"], s["kld_p90"],
               s["kld_p99"], s["kld_p999"], s["kld_max"]))


HEADER = ("%-22s %5s  %-19s  %-24s %8s %8s %8s %8s %9s"
          % ("arm", "rows", "same top-1 as ref", "mean KLD", "median", "p90", "p99", "p99.9", "max"))


def report(args, acc, order, prompt_rows, out=None):
    out = out or sys.stdout
    n_rows = None
    for name in order:
        n_rows = len(acc[name].kld)
    rows_done = acc[order[0]].rows
    contiguous = n_rows == rows_done[-1] - rows_done[0] + 1
    print("reference        %s" % args.ref, file=out)
    if (rows_done[0], rows_done[-1] + 1) != (args.first, args.last) or not contiguous:
        print("rows             %d..%d (%d collected, of --rows %d:%d on absolute row ids)"
              % (rows_done[0], rows_done[-1], n_rows, args.first, args.last), file=out)
    else:
        print("rows             %d..%d (%d)" % (args.first, args.last - 1, args.last - args.first),
              file=out)
    if prompt_rows is not None:
        print("split            prompt rows < %d, answer rows >= %d" % (prompt_rows, prompt_rows), file=out)
    print("support          natural log p_ref > %.1f  (llama.cpp perplexity.cpp:222)" % args.min_log_prob,
          file=out)
    if args.truncate:
        print("truncation       the reference's top %d only; a token an arm cannot answer for is"
              " charged at that arm's smallest known probability, so every KLD below is a LOWER BOUND"
              % args.truncate, file=out)
    if args.arm_topn:
        print("arm blindness    every arm limited to its own top %d, so the two arm kinds are"
              " the same estimator" % args.arm_topn, file=out)
    print("direction        KL(P_reference || Q_arm), natural log, nats", file=out)
    print("", file=out)

    blocks = [("all rows", None)]
    if prompt_rows is not None:
        rows = acc[order[0]].rows
        blocks.append(("prompt rows", [rows[i] < prompt_rows for i in range(n_rows)]))
        blocks.append(("answer rows", [rows[i] >= prompt_rows for i in range(n_rows)]))

    summaries = {}
    for label, mask in blocks:
        print("== %s" % label, file=out)
        print(HEADER, file=out)
        for name in order:
            s = summarize(acc[name], mask)
            if s is None:
                continue
            summaries.setdefault(label, {})[name] = s
            print(fmt_arm_line(name, s), file=out)
        print("", file=out)

    groups_out = {}
    if getattr(args, "row_groups", None):
        have = set(rows_done)
        for name, grows in args.row_groups:
            sel = set(r for r in grows if r in have)
            if not sel:
                print("== %s - no row of this group is in the run" % name, file=out)
                print("", file=out)
                continue
            mask = [r in sel for r in rows_done]
            print("== %s (%d rows)" % (name, len(sel)), file=out)
            print(HEADER, file=out)
            groups_out[name] = {}
            for arm in order:
                s = summarize(acc[arm], mask)
                if s is None:
                    continue
                groups_out[name][arm] = s
                print(fmt_arm_line(arm, s), file=out)
            print("", file=out)

    curve = []
    if getattr(args, "kld_vs_position", None):
        lo, hi = rows_done[0], rows_done[-1]
        span = hi - lo + 1
        print("== KLD vs position, %d equal-width bins over absolute rows %d..%d"
              % (args.kld_vs_position, lo, hi), file=out)
        print("%-14s %5s  %-19s  %s" % ("bin", "rows", "mean KLD per arm", "same top-1 per arm"),
              file=out)
        for b in range(args.kld_vs_position):
            blo = lo + (span * b) // args.kld_vs_position
            bhi = lo + (span * (b + 1)) // args.kld_vs_position - 1
            mask = [blo <= r <= bhi for r in rows_done]
            entry = {"lo": blo, "hi": bhi, "arms": {}}
            line = ["%-14s %5d" % ("%d..%d" % (blo, bhi), sum(mask))]
            for arm in order:
                s = summarize(acc[arm], mask)
                if s is None:
                    continue
                entry["arms"][arm] = {"n": s["n"], "kld_mean": s["kld_mean"],
                                      "kld_unc": s["kld_unc"],
                                      "same_top1_share": s["same_top1_share"],
                                      "same_top1_unc": s["same_top1_unc"]}
                line.append("%-22s %9.6f  %6.2f %%"
                            % (arm, s["kld_mean"], 100 * s["same_top1_share"]))
            curve.append(entry)
            print("   ".join(line), file=out)
        print("", file=out)

    print("== the reference's top-1 probability, and what the arm does to it "
          "(dp = q_arm(ref top-1) - p_ref(ref top-1))", file=out)
    print("%-22s %-24s %12s %12s" % ("arm", "mean dp", "lost > 10 pp", "lost > 50 pp"), file=out)
    for name in order:
        s = summaries["all rows"][name]
        print("%-22s %8.3f +- %6.3f %%   %5d = %5.1f %%  %5d = %5.1f %%"
              % (name, 100 * s["dp_mean"], 100 * s["dp_unc"],
                 s["dp_lost_10pp"], 100 * s["dp_lost_10pp"] / s["n"],
                 s["dp_lost_50pp"], 100 * s["dp_lost_50pp"] / s["n"]), file=out)
    print("", file=out)

    if args.truncate or args.arm_topn:
        print("== what the truncation covers", file=out)
        print("%-22s %18s %18s %14s" % ("arm", "ref mass in support", "arm mass known",
                                        "rows truncated"), file=out)
        for name in order:
            s = summaries["all rows"][name]
            print("%-22s %17.5f  %17.5f  %6d = %5.1f %%"
                  % (name, s["support_mass_mean"], s["arm_mass_mean"],
                     s["rows_truncated"], 100 * s["rows_truncated"] / s["n"]), file=out)
        print("", file=out)

    pairs = []
    for spec in args.paired or []:
        a, b = spec.split(",", 1)
        for name in (a, b):
            if name not in acc:
                raise SystemExit("--paired names %r, which is not an arm" % name)
        pairs.append(paired(acc[a], acc[b]))
    if pairs:
        print("== paired, per row: KLD(A) - KLD(B) on the SAME row", file=out)
        print("%-30s %5s %-24s %10s %10s %12s" % ("A - B", "rows", "mean difference",
                                                  "median", "A worse", "sign test p"), file=out)
        for p in pairs:
            print("%-30s %5d %9.6f +- %9.6f %10.6f %5d/%-4d %12.3g"
                  % ("%s - %s" % (p["a"], p["b"]), p["n"], p["mean"], p["unc"], p["median"],
                     p["a_above"], p["n"] - p["ties"], p["sign_test_p"]), file=out)
        print("", file=out)
    return summaries, pairs, groups_out, curve


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ref", required=True, help="the f32 reference logit dump")
    ap.add_argument("--vocab", type=int, default=VOCAB_DEFAULT)
    ap.add_argument("--rows", default=None, help="A:B, half open; default every row of the reference")
    ap.add_argument("--prompt-rows", type=int, default=None,
                    help="rows below this index are prompt rows, the rest answer rows")
    ap.add_argument("--arm", action="append", default=[], metavar="NAME=PATH",
                    help="an arm that dumped the complete logit vector")
    ap.add_argument("--topn-arm", action="append", default=[], metavar="NAME=PATH",
                    help="an arm that could only report its top N probabilities (JSON)")
    ap.add_argument("--min-log-prob", type=float, default=MIN_LOG_PROB)
    ap.add_argument("--truncate", type=int, default=None,
                    help="sum over the reference's top M tokens only")
    ap.add_argument("--arm-topn", type=int, default=None,
                    help="give every full arm the same blindness beyond its own top N")
    ap.add_argument("--paired", action="append", default=[], metavar="A,B")
    ap.add_argument("--row-groups", default=None, metavar="FILE",
                    help="a {\"groups\": [{\"name\", \"rows\"}]} file - the depth blocks of "
                         "the #90 row plan or the per-text groups of the English corpus; "
                         "adds one summary block per group per arm")
    ap.add_argument("--kld-vs-position", type=int, default=None, metavar="BINS",
                    help="bin the collected rows by absolute position into BINS equal-width "
                         "bins and report mean KLD and same-top-1 per bin per arm")
    ap.add_argument("--per-row", default=None, help="write the per-row detail to this JSON file")
    ap.add_argument("--json", dest="json_out", default=None, help="write the summary to this JSON file")
    args = ap.parse_args(argv)

    full_arms = parse_named(args.arm, "--arm")
    topn_arms = parse_named(args.topn_arm, "--topn-arm")
    if not full_arms and not topn_arms:
        raise SystemExit("no arms: give at least one --arm or --topn-arm")
    if topn_arms and not args.truncate:
        raise SystemExit("a --topn-arm needs --truncate M: the sum has to be restricted to a set "
                         "the top-N arm can answer for, and the same restriction has to reach "
                         "every other arm")

    if args.row_groups:
        args.row_groups = load_row_groups(args.row_groups)

    # for a sparse reference the total is its last absolute row + 1, so `--rows A:B`
    # bounds absolute row ids in both forms
    if os.path.exists(args.ref + ".rows.json"):
        with open(args.ref + ".rows.json", "r", encoding="utf-8") as fh:
            total = max(int(r) for r in json.load(fh)["rows"]) + 1
    else:
        total = os.path.getsize(args.ref) // (args.vocab * 4)
    args.first, args.last = parse_range(args.rows, total)

    acc, _ = collect(args.ref, args.vocab, args.first, args.last, full_arms, topn_arms,
                     args.min_log_prob, args.truncate, args.arm_topn, args.per_row)
    order = [n for n, _ in full_arms] + [n for n, _ in topn_arms]
    summaries, pairs, groups_out, curve = report(args, acc, order, args.prompt_rows)

    if args.json_out:
        doc = {"reference": args.ref, "first_row": args.first, "last_row": args.last,
               "prompt_rows": args.prompt_rows, "min_log_prob": args.min_log_prob,
               "truncate": args.truncate, "arm_topn": args.arm_topn,
               "summaries": summaries, "paired": pairs}
        if groups_out:
            doc["groups"] = groups_out
        if curve:
            doc["position_curve"] = curve
        with open(args.json_out, "w", encoding="utf-8") as fh:
            json.dump(doc, fh, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
