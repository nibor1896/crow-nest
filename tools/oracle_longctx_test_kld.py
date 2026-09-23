#!/usr/bin/env python3
"""#90: the unit tests of the oracle-kld EXTENSIONS - sparse dumps, row groups and
the KLD-vs-position curve. The 55 tests of tools/test_oracle_kld.py stay untouched
and green; everything here is the new surface only.

Run:  python3 tools/oracle_longctx_test_kld.py      (no GPU, no server, no model)
"""

import array
import importlib.util
import json
import math
import os
import struct
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("oracle_kld", TOOLS / "oracle-kld.py")
ok = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ok)

VOCAB = 8
ROWS = 12


def write_dense(path, rows, vocab=VOCAB):
    a = array.array("f")
    for row in rows:
        a.extend(row)
    if os.path.isfile(path):
        os.unlink(path)
    with open(path, "wb") as fh:
        a.tofile(fh)
    return path


def write_sparse(path, rows, keep, vocab=VOCAB):
    write_dense(path, [rows[i] for i in keep])
    with open(path + ".rows.json", "w", encoding="utf-8") as fh:
        json.dump({"rows": list(keep)}, fh)
    return path


def tiny_rows():
    """ROWS rows of VOCAB logits, deterministic and spread out enough to differ."""
    rows = []
    for r in range(ROWS):
        row = [math.log(1.0 + ((r * 7 + i * 3) % 11)) for i in range(VOCAB)]
        row[(r * 5) % VOCAB] += 3.0          # a clear top-1 that moves with the row
        rows.append(row)
    return rows


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="oracle90-")
        self.rows = tiny_rows()
        # a second dump: the same rows with a small perturbation, top-1 kept
        self.other = [[x + 0.01 for x in row] for row in self.rows]
        self.ref = write_dense(os.path.join(self.dir, "ref.f32"), self.rows)
        self.arm = write_dense(os.path.join(self.dir, "arm.f32"), self.other)


class SparseSources(Base):
    def test_a_sparse_read_reproduces_the_dense_per_row_kld(self):
        keep = [0, 3, 7, 11]
        sref = write_sparse(os.path.join(self.dir, "sref.f32"), self.rows, keep)
        sarm = write_sparse(os.path.join(self.dir, "sarm.f32"), self.other, keep)
        dense = ok.collect(self.ref, VOCAB, 0, ROWS, [("arm", self.arm)], [],
                           ok.MIN_LOG_PROB, None, None)[0]["arm"]
        sparse = ok.collect(sref, VOCAB, 0, ROWS, [("arm", sarm)], [],
                            ok.MIN_LOG_PROB, None, None)[0]["arm"]
        self.assertEqual(sparse.rows, keep)
        for d, s in zip(dense.kld, sparse.kld):
            self.assertAlmostEqual(d, s, places=9)
        self.assertEqual(sum(sparse.same_top1), len(keep))

    def test_the_sparse_reference_answers_only_rows_it_carries(self):
        keep = [2, 5]
        sref = write_sparse(os.path.join(self.dir, "s2.f32"), self.rows, keep)
        sarm = write_sparse(os.path.join(self.dir, "a2.f32"), self.other, keep)
        acc = ok.collect(sref, VOCAB, 0, ROWS, [("arm", sarm)], [],
                         ok.MIN_LOG_PROB, None, None)[0]["arm"]
        self.assertEqual(acc.rows, keep)
        # --rows bounds absolute ids: 3:9 keeps only row 5
        acc = ok.collect(sref, VOCAB, 3, 9, [("arm", sarm)], [],
                         ok.MIN_LOG_PROB, None, None)[0]["arm"]
        self.assertEqual(acc.rows, [5])
        with self.assertRaises(SystemExit):
            ok.collect(sref, VOCAB, 20, 30, [("arm", sarm)], [],
                       ok.MIN_LOG_PROB, None, None)

    def test_a_sparse_arm_must_carry_every_reference_row(self):
        sref = write_sparse(os.path.join(self.dir, "s3.f32"), self.rows, [1, 4, 8])
        sarm = write_sparse(os.path.join(self.dir, "a3.f32"), self.other, [1, 8])
        with self.assertRaises(SystemExit):
            ok.collect(sref, VOCAB, 0, ROWS, [("arm", sarm)], [],
                       ok.MIN_LOG_PROB, None, None)

    def test_a_sparse_dump_with_broken_sidecar_is_refused(self):
        path = write_sparse(os.path.join(self.dir, "s4.f32"), self.rows, [3, 1, 2])
        with self.assertRaises(SystemExit):
            ok.SparseLogitFile(path, VOCAB)

    def test_paired_works_over_sparse_rows(self):
        keep = list(range(0, ROWS, 2))
        sref = write_sparse(os.path.join(self.dir, "s5.f32"), self.rows, keep)
        sa = write_sparse(os.path.join(self.dir, "a5.f32"),
                          [[x - 0.05 for x in r] for r in self.rows], keep)
        sb = write_sparse(os.path.join(self.dir, "b5.f32"),
                          [[x + 0.05 for x in r] for r in self.rows], keep)
        acc = ok.collect(sref, VOCAB, 0, ROWS, [("a", sa), ("b", sb)], [],
                         ok.MIN_LOG_PROB, None, None)[0]
        p = ok.paired(acc["a"], acc["b"])
        self.assertEqual(p["n"], len(keep))


class RowGroupsAndCurve(Base):
    def GROUPS(self):
        g = os.path.join(self.dir, "groups.json")
        with open(g, "w", encoding="utf-8") as fh:
            json.dump({"groups": [{"name": "shallow", "rows": [0, 1, 2, 3]},
                                  {"name": "deep", "rows": [8, 9, 10, 11]}]}, fh)
        return g

    def run_main(self, extra):
        import contextlib
        import io
        buf = io.StringIO()
        j = os.path.join(self.dir, "out.json")
        with contextlib.redirect_stdout(buf):
            rc = ok.main(["--ref", self.ref, "--vocab", str(VOCAB),
                          "--arm", "arm=" + self.arm] + extra + ["--json", j])
        self.assertEqual(rc, 0)
        return json.load(open(j, encoding="utf-8")), buf.getvalue()

    def test_row_groups_land_in_blocks_and_json(self):
        doc, printed = self.run_main(["--row-groups", self.GROUPS()])
        self.assertIn("== shallow (4 rows)", printed)
        self.assertIn("== deep (4 rows)", printed)
        self.assertEqual(doc["groups"]["shallow"]["arm"]["n"], 4)
        self.assertEqual(doc["groups"]["deep"]["arm"]["n"], 4)
        self.assertNotIn("groups", doc["summaries"])   # the 55-test contract holds

    def test_the_position_curve_bins_by_absolute_row(self):
        doc, printed = self.run_main(["--kld-vs-position", "3"])
        self.assertIn("KLD vs position", printed)
        self.assertEqual(len(doc["position_curve"]), 3)
        self.assertEqual(doc["position_curve"][0]["lo"], 0)
        self.assertEqual(doc["position_curve"][-1]["hi"], ROWS - 1)
        total = sum(e["arms"]["arm"]["n"] for e in doc["position_curve"])
        self.assertEqual(total, ROWS)

    def test_curve_and_groups_over_a_sparse_reference(self):
        keep = [0, 1, 5, 6, 11]
        sref = write_sparse(os.path.join(self.dir, "s6.f32"), self.rows, keep)
        sarm = write_sparse(os.path.join(self.dir, "a6.f32"), self.other, keep)
        g = self.GROUPS()
        with open(g, "w", encoding="utf-8") as fh:
            json.dump({"groups": [{"name": "ends", "rows": [0, 1, 11]},
                                  {"name": "middle", "rows": [5, 6]}]}, fh)
        import contextlib
        import io
        buf = io.StringIO()
        j = os.path.join(self.dir, "sparse.json")
        with contextlib.redirect_stdout(buf):
            rc = ok.main(["--ref", sref, "--vocab", str(VOCAB),
                          "--arm", "arm=" + sarm, "--row-groups", g,
                          "--kld-vs-position", "2", "--json", j])
        self.assertEqual(rc, 0)
        doc = json.load(open(j, encoding="utf-8"))
        self.assertEqual(doc["groups"]["ends"]["arm"]["n"], 3)
        self.assertEqual(doc["groups"]["middle"]["arm"]["n"], 2)
        self.assertEqual(sum(e["arms"]["arm"]["n"] for e in doc["position_curve"]), 5)
        self.assertIn("(5 collected", buf.getvalue())


class SubsetTool(unittest.TestCase):
    def test_subset_round_trips_through_the_sparse_reader(self):
        import importlib.util as ilu
        spec = ilu.spec_from_file_location(
            "rows_tool", TOOLS / "oracle_longctx_rows.py")
        rt = ilu.module_from_spec(spec)
        spec.loader.exec_module(rt)   # argparse only fires in __main__

        d = tempfile.mkdtemp(prefix="oracle90-subset-")
        rows = tiny_rows()
        dense = write_dense(os.path.join(d, "dense.f32"), rows)
        keep = [1, 4, 9]
        keep_file = os.path.join(d, "keep.json")
        with open(keep_file, "w", encoding="utf-8") as fh:
            json.dump(keep, fh)
        rc = rt.main(["subset", "--source", dense, "--out", os.path.join(d, "sp.f32"),
                      "--rows-file", keep_file, "--vocab", str(VOCAB)])
        self.assertEqual(rc, 0)
        with open(os.path.join(d, "sp.f32.rows.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["rows"], keep)
        src = ok.SparseLogitFile(os.path.join(d, "sp.f32"), VOCAB)
        for i, r in enumerate(keep):
            got = src.read(r).tolist()
            want = struct.unpack("<%df" % VOCAB,
                                 struct.pack("<%df" % VOCAB, *rows[r]))
            self.assertEqual(got, list(want))


if __name__ == "__main__":
    unittest.main(verbosity=2)
