#!/usr/bin/env python3
"""#147: tests of tools/glm_tier_sim.py on synthetic runner dumps with known answers.

  python -I tools/test_glm_tier_sim.py

No GPU, no weights, no tokenizer: the dumps are written in the runner's own layout
(docs/glm5-reference-runner.md §5: manifest.json + l<k>-routing-ids.i32 [N][8] for MoE layers 3..44).
The guards: the held-out file never enters the cut, and a missing / invalid / partial-source input
gives "G1 not answered", never a verdict.
"""
import copy
import hashlib
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_tier_sim as ts  # noqa: E402

L, E, K = ts.SHAPE
REPO = Path(__file__).resolve().parent.parent
STEP3 = REPO / "runs" / "glm53-flash" / "step03" / "20261008T001819Z.json"
KINDS = ["kda+dense"] * 3 + ["dsa+moe" if l % 4 == 3 else "kda+moe" for l in range(3, 45)]


def write_run(d, routes, ids=None, weights="cnq (CNQ container)", partial=None, complete=True):
    """A runner out dir: manifest + one routing file per MoE layer, sha256 recorded as the runner does."""
    os.makedirs(d, exist_ok=True)
    n = len(routes)
    ids = list(range(n)) if ids is None else ids
    files = {}
    for j, l in enumerate(range(3, 45)):
        nm = "l%d-routing-ids.i32" % l
        np.ascontiguousarray(routes[:, j, :], dtype="<i4").tofile(os.path.join(d, nm))
        files[nm] = {"shape": [n, K], "dtype": "i32", "sha256": ts.sha256_file(os.path.join(d, nm))}
    w = {"weights": weights}
    if partial:
        w["partial"] = partial
    man = {"ids": ids, "T": n, "D": 0, "layers": [0, 45], "num_hidden_layers": 45, "layer_kinds": KINDS,
           "files": files, "weights": w, "complete": complete}
    ts.jdump(man, os.path.join(d, "manifest.json"))
    return man


def rows(n, picks):
    """[n][L][K] with the same sorted 8 ids at every position and layer."""
    r = np.empty((n, L, K), np.uint16)
    r[:] = np.array(sorted(picks), np.uint16)
    return r


# cal routes only experts 0..7, so every layer ranks expert e at rank e (ties: lower id)
CAL_PICKS = range(8)
# held-out: 2 ids in VRAM at every N in 25..37, 2 pinned (40..107 at N 25..37 -> ids 40, 41) and
# 4 always on the NVMe (>= 37 + 83 = 120): VRAM 0.25, pinned 0.25, m 0.5 exactly
HELD_PICKS = (0, 1, 40, 41, 200, 201, 202, 203)


def step3_doc(rates, valid=True):
    per = {}
    for r, xs in rates.items():
        per[str(r)] = {"rates_gbps": xs}
    meds = {r: sorted(xs)[1] for r, xs in rates.items()}
    best = max(meds, key=lambda r: meds[r])
    s = max(rates[best]) / min(rates[best])
    return {"verdict": {"valid": valid, "void_reasons": [] if valid else ["foreign reads"], "per_readers": per,
                        "best_readers": best, "b_for_g1": meds[best] if s <= 1.15 and valid else None,
                        "g1_input": "x"}}


class Corpus:
    """A corpus dir + runs dir on disk in the layout `sim` reads."""

    def __init__(self, root, held_routes, cal_routes, held_gen=None, held_task="coding", cal_task="ops",
                 **run_kw):
        self.root = root
        self.cdir, self.rdir = os.path.join(root, "corpus"), os.path.join(root, "runs")
        os.makedirs(self.cdir)
        files = []
        for i, (name, routes, role) in enumerate([("held", held_routes, "held")]
                                                 + [("cal%d" % j, r, "cal") for j, r in enumerate(cal_routes)]):
            n = len(routes)
            ids = [i * 100000 + t for t in range(n)]
            ts.jdump(ids, os.path.join(self.cdir, name + "-ids.json"))
            spans = held_gen if (role == "held" and held_gen is not None) else [[0, n]]
            ts.jdump({"tokens": n, "spans": spans}, os.path.join(self.cdir, name + "-mask.json"))
            write_run(os.path.join(self.rdir, name), routes, ids, **run_kw)
            files.append({"name": name, "task": held_task if role == "held" else cal_task, "role": role,
                          "source_sha256": hashlib.sha256(name.encode()).hexdigest(),
                          "ids_sha256": ts.sha256_ids(ids)})
        self.doc = {"held": "held", "files": files}
        self.path = os.path.join(self.cdir, "corpus.json")
        ts.jdump(self.doc, self.path)


def quiet(*_a, **_k):
    pass


class TestPolicy(unittest.TestCase):
    def test_rank_ties_lower_id(self):
        c = np.zeros((1, 6), np.int64)
        c[0] = [5, 9, 9, 0, 5, 1]
        self.assertEqual(ts.rank_table(c)[0].tolist(), [2, 0, 1, 5, 3, 4])

    def test_known_shares_every_n(self):
        held = ts.Run("h", "a", rows(10, HELD_PICKS), np.ones(10, bool))
        inv = ts.cut([ts.Run("c", "b", rows(10, CAL_PICKS), np.ones(10, bool))], held)
        self.assertEqual(inv[0, :10].tolist(), list(range(10)))
        rk = ts.visit_ranks(held.routes, inv)
        for n in ts.N_RANGE:
            v, p, nv = ts.shares(rk, n)
            self.assertEqual((v[0], p[0], nv[0]), (0.25, 0.25, 0.5), n)

    def test_cut_counts_generated_positions_only(self):
        r = rows(4, range(100, 108))
        r[2:] = np.array(range(8), np.uint16)           # positions 2, 3 route 0..7
        cal = ts.Run("c", "b", r, np.array([False, False, True, True]))
        inv = ts.cut([cal])
        self.assertTrue((inv[:, :8] < 8).all())         # prompt positions (100..107) did not count
        self.assertTrue((inv[:, 100:108] >= 8).all())


class TestHeldOutGuard(unittest.TestCase):
    """Guard: the held-out file never enters the calibration cut."""

    def test_cut_refuses_held_in_calibration(self):
        held = ts.Run("h", "a", rows(10, HELD_PICKS), np.ones(10, bool))
        cal = ts.Run("c", "b", rows(10, CAL_PICKS), np.ones(10, bool))
        with self.assertRaises(ts.SimError):
            ts.cut([cal, held], held)
        with self.assertRaises(ts.SimError):            # same name, another object (loaded twice)
            ts.cut([cal, ts.Run("h", "a", rows(10, HELD_PICKS), np.ones(10, bool))], held)

    def test_corpus_refuses_same_task_or_same_file(self):
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(d, rows(10, HELD_PICKS), [rows(10, CAL_PICKS)])
            ts.check_corpus(c.doc)
            same_task = copy.deepcopy(c.doc)
            same_task["files"][1]["task"] = same_task["files"][0]["task"]
            with self.assertRaisesRegex(ts.SimError, "also the task"):
                ts.check_corpus(same_task)
            same_file = copy.deepcopy(c.doc)
            same_file["files"][1]["source_sha256"] = same_file["files"][0]["source_sha256"]
            with self.assertRaisesRegex(ts.SimError, "same file"):
                ts.check_corpus(same_file)
            two_held = copy.deepcopy(c.doc)
            two_held["files"][1]["role"] = "held"
            with self.assertRaises(ts.SimError):
                ts.check_corpus(two_held)

    def test_held_out_counts_change_nothing(self):
        """The cut of record is identical whatever the held-out routes."""
        cal = [ts.Run("c", "b", rows(10, CAL_PICKS), np.ones(10, bool))]
        a = ts.cut(cal, ts.Run("h", "a", rows(10, HELD_PICKS), np.ones(10, bool)))
        b = ts.cut(cal, ts.Run("h", "a", rows(10, range(280, 288)), np.ones(10, bool)))
        self.assertTrue(np.array_equal(a, b))


class TestB(unittest.TestCase):
    """Guard: no valid B -> "G1 not answered"."""

    def test_step3_of_record_gives_no_b(self):
        b, why = ts.b_from_step3(str(STEP3))
        self.assertIsNone(b)
        self.assertIn("spread 1.232", why)

    def test_tampered_b_with_bad_spread_is_refused(self):
        d = ts.jload(STEP3)
        d["verdict"]["b_for_g1"] = d["verdict"]["per_readers"]["2"]["median_gbps"]   # claims B = 9.765
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "s3.json")
            ts.jdump(d, p)
            b, why = ts.b_from_step3(p)
        self.assertIsNone(b)
        self.assertIn("spread 1.232 > 1.15", why)

    def test_amendment5_fixed_reader_count(self):
        b, why = ts.b_from_step3(str(STEP3), readers=1)   # PREREG amendment 5 (#146)
        self.assertAlmostEqual(b, 6.9936611328)
        self.assertIn("spread 1.003", why)
        b, why = ts.b_from_step3(str(STEP3), readers=2)   # a fixed count still holds the spread rule
        self.assertIsNone(b)
        self.assertIn("spread 1.232 > 1.15", why)
        self.assertIsNone(ts.b_from_step3(str(STEP3), readers=8)[0])

    def test_missing_void_and_valid(self):
        self.assertIsNone(ts.b_from_step3(None)[0])
        with tempfile.TemporaryDirectory() as t:
            p = os.path.join(t, "s3.json")
            ts.jdump(step3_doc({1: [7.0, 7.1, 6.9], 2: [6.0, 6.1, 6.2]}, valid=False), p)
            self.assertIn("VOID", ts.b_from_step3(p)[1])
            ts.jdump(step3_doc({1: [7.0, 7.1, 6.9], 2: [6.0, 6.1, 6.2]}), p)
            self.assertEqual(ts.b_from_step3(p)[0], 7.0)

    def test_verdict_not_answered_without_b(self):
        s = {"mean": 0.01, "ci": [0.005, 0.015], "n": 5000}
        v = ts.g1_verdict(s, None, "spread 1.232 > 1.15 at the best reader count 2", [])
        self.assertTrue(v["verdict"].startswith("G1 not answered"), v["verdict"])
        self.assertIsNone(v["m_star"])

    def test_verdict_pass_and_fail(self):
        b = 7.0                                          # m* = 0.03678, prefill 279.3
        ok = ts.g1_verdict({"mean": 0.02, "ci": [0.01, 0.03], "n": 5000}, b, "x", [])
        self.assertTrue(ok["verdict"].startswith("G1 passed"), ok["verdict"])
        bad = ts.g1_verdict({"mean": 0.03, "ci": [0.02, 0.04], "n": 5000}, b, "x", [])
        self.assertTrue(bad["verdict"].startswith("G1 failed: m CI upper bound"), bad["verdict"])
        slow = ts.g1_verdict({"mean": 0.0, "ci": [0.0, 0.0], "n": 5000}, 3.0, "x", [])
        self.assertIn("prefill bound 119.7 < 150", slow["verdict"])

    def test_verdict_not_answered_partial_or_short(self):
        s = {"mean": 0.0, "ci": [0.0, 0.0], "n": 5000}
        v = ts.g1_verdict(s, 7.0, "x", ["h: partial container (--layers 0-3): plausibility only"])
        self.assertTrue(v["verdict"].startswith("G1 not answered: h: partial"), v["verdict"])
        v = ts.g1_verdict({"mean": 0.0, "ci": [math.nan, math.nan], "n": 1500}, 7.0, "x", [])
        self.assertIn("fewer than two blocks", v["verdict"])


class TestWindow(unittest.TestCase):
    SEQ = [10, 11, 10, 12, 11]                           # A B A C B

    def one_layer(self, seq):
        return np.array(seq, np.uint16).reshape(-1, 1, 1), np.ones((len(seq), 1, 1), bool)

    def test_lru_known(self):
        r, m = self.one_layer(self.SEQ)
        self.assertEqual(ts.lru_reads(r, m, 2).tolist(), [1, 1, 0, 1, 1])   # 4 reads
        self.assertEqual(ts.lru_reads(r, m, 0).tolist(), [1, 1, 1, 1, 1])
        self.assertEqual(ts.lru_reads(r, m, 3).tolist(), [1, 1, 0, 1, 0])

    def test_min_known(self):
        r, m = self.one_layer(self.SEQ)
        self.assertEqual(ts.min_reads(r, m, 2).tolist(), [1, 1, 0, 1, 0])   # 3 reads: C bypassed
        self.assertEqual(ts.min_reads(r, m, 1).tolist(), [1, 1, 0, 1, 1])

    def test_monotone_and_ceiling(self):
        rng = np.random.default_rng(1)
        n = 400
        r = np.sort(np.stack([np.stack([rng.choice(E, K, replace=False, p=None) for _ in range(L)])
                              for _ in range(n)]).astype(np.uint16), axis=2)
        nvme = r >= 108
        prev = None
        for w in (0, 4, 8, 16, 32):
            lru = ts.lru_reads(r, nvme, w).sum()
            mn = ts.min_reads(r, nvme, w).sum()
            self.assertLessEqual(mn, lru)
            if prev is not None:
                self.assertLessEqual(lru, prev)
            prev = lru
        self.assertEqual(ts.lru_reads(r, nvme, 0).sum(), nvme.sum())


class TestBootstrap(unittest.TestCase):
    def test_constant_and_short(self):
        self.assertEqual(ts.boot_ci(np.full(5000, 0.5)), (0.5, 0.5))
        self.assertTrue(all(math.isnan(x) for x in ts.boot_ci(np.zeros(1999))))

    def test_blocks_in_order(self):
        x = np.concatenate([np.zeros(1000), np.ones(1000), np.zeros(1000), np.ones(1000), np.full(999, 9.0)])
        lo, hi = ts.boot_ci(x)                           # the 999-position tail is not a block
        self.assertGreaterEqual(lo, 0.0)
        self.assertLessEqual(hi, 1.0)
        self.assertLess(lo, 0.5)
        self.assertGreater(hi, 0.5)


class TestRunnerDump(unittest.TestCase):
    def test_roundtrip_and_self_test(self):
        r = rows(20, HELD_PICKS)
        with tempfile.TemporaryDirectory() as d:
            write_run(d, r)
            got = ts.load_runner(d)[1]
            self.assertTrue(np.array_equal(got, r))
            with open(os.path.join(d, "l20-routing-ids.i32"), "r+b") as f:
                f.write(b"\x01\x00\x00\x00")
            with self.assertRaisesRegex(ts.SimError, "sha256 differs"):
                ts.load_runner(d)
            bad = r.copy()
            bad[3, 5, :2] = 7                            # a duplicate id in one row
            write_run(d, bad)
            with self.assertRaisesRegex(ts.SimError, "distinct ascending"):
                ts.load_runner(d)
            bad = r.astype(np.int32)
            bad[0, 0, 7] = 288
            write_run(d, bad)
            with self.assertRaisesRegex(ts.SimError, "outside"):
                ts.load_runner(d)

    def test_source_of_record(self):
        self.assertEqual(ts.source_reasons({"weights": {"weights": "cnq (x)"}, "complete": True,
                                            "layers": [0, 45], "num_hidden_layers": 45}), [])
        self.assertTrue(ts.source_reasons({"weights": {"weights": "fp8 (FP8 E4M3 originals)"}, "complete": True,
                                           "layers": [0, 45], "num_hidden_layers": 45}))
        self.assertTrue(ts.source_reasons({"weights": {"weights": "cnq (x)", "partial": {"filter": "--layers 0-3"}},
                                           "complete": True, "layers": [0, 4], "num_hidden_layers": 45}))


class TestEndToEnd(unittest.TestCase):
    def run_sim(self, c, step3=None):
        held, cal, reasons = ts.load_corpus(c.path, c.rdir)
        b, why = ts.b_from_step3(step3)
        return ts.simulate(held, cal, b, why, reasons, windows=(0, 8), out=quiet)

    def test_known_m_and_not_answered_without_b(self):
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(d, rows(3000, HELD_PICKS), [rows(500, CAL_PICKS), rows(500, CAL_PICKS)])
            res = self.run_sim(c, str(STEP3))
            for n in ts.N_RANGE:
                self.assertEqual(res["by_n"][n]["m"]["mean"], 0.5)
                self.assertEqual(res["by_n"][n]["pinned"]["mean"], 0.25)
            self.assertEqual(res["by_n"][25]["m"]["ci"], [0.5, 0.5])
            # the same 4 NVMe experts at every position: a window of 8 holds them after the first read
            self.assertAlmostEqual(res["windows"]["25/8"]["lru"]["mean"], 4 * L / 336 / 3000)
            self.assertEqual(res["g1"]["generated"], 3000)
            self.assertTrue(res["g1"]["verdict"].startswith("G1 not answered: B: spread 1.232"))
            self.assertEqual(len(res["loo"]), 3)
            self.assertEqual(res["depth"][1]["m"]["n"], 0)

    def test_pass_fail_with_valid_b(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s3.json")
            ts.jdump(step3_doc({1: [7.0, 7.1, 6.9], 2: [6.0, 6.1, 6.2]}), p)
            c = Corpus(os.path.join(d, "a"), rows(3000, CAL_PICKS), [rows(500, CAL_PICKS)])
            self.assertTrue(self.run_sim(c, p)["g1"]["verdict"].startswith("G1 passed"))
            c = Corpus(os.path.join(d, "b"), rows(3000, HELD_PICKS), [rows(500, CAL_PICKS)])
            self.assertTrue(self.run_sim(c, p)["g1"]["verdict"].startswith("G1 failed"))

    def test_partial_container_not_answered(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "s3.json")
            ts.jdump(step3_doc({1: [7.0, 7.1, 6.9]}), p)
            c = Corpus(os.path.join(d, "a"), rows(3000, CAL_PICKS), [rows(500, CAL_PICKS)],
                       partial={"filter": "--layers 0-3 --with-embed-head"})
            v = self.run_sim(c, p)["g1"]["verdict"]
            self.assertTrue(v.startswith("G1 not answered: held: partial container"), v)

    def test_generated_mask_is_primary(self):
        r = rows(4000, CAL_PICKS)
        r[2000:] = np.array(HELD_PICKS, np.uint16)
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(d, r, [rows(500, CAL_PICKS)], held_gen=[[2000, 4000]])
            res = self.run_sim(c)
            self.assertEqual(res["generated"], 2000)
            self.assertEqual(res["by_n"][25]["m"]["mean"], 0.5)
            self.assertEqual(res["by_n"][25]["m_prefill"], 0.0)

    def test_ids_must_be_the_corpus_file(self):
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(d, rows(100, HELD_PICKS), [rows(100, CAL_PICKS)])
            write_run(os.path.join(c.rdir, "held"), rows(100, HELD_PICKS), ids=list(range(5, 105)))
            with self.assertRaisesRegex(ts.SimError, "not held-ids.json"):
                ts.load_corpus(c.path, c.rdir)

    def test_cli_exit_code_on_refusal(self):
        with tempfile.TemporaryDirectory() as d:
            c = Corpus(d, rows(100, HELD_PICKS), [rows(100, CAL_PICKS)], held_task="ops")
            self.assertEqual(ts.main(["sim", "--corpus", c.path, "--runs", c.rdir]), 2)


class TestCorpusSpans(unittest.TestCase):
    @staticmethod
    def render(_tok, msgs, gen=False):
        out = []
        for m in msgs:
            out += [{"system": 1, "user": 2, "assistant": 3, "tool": 4}[m["role"]]] + [ord(ch) for ch in m["content"]]
        return out + ([3] if gen else [])

    def test_spans_and_clip(self):
        msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "abc"}, {"role": "tool", "content": "t"},
                {"role": "assistant", "content": "de"}]
        ids = self.render(None, msgs)
        spans, skipped = ts.gen_spans(None, msgs, ids, self.render)
        self.assertEqual((spans, skipped), ([[6, 9], [12, 14]], 0))
        self.assertEqual([ids[a - 1] for a, _ in spans], [3, 3])     # each span follows its role token
        self.assertEqual(ts.clip(spans, 10), [[6, 9]])
        self.assertEqual(ts.clip(spans, 13), [[6, 9], [12, 13]])


class TestStageCeilingDefault(unittest.TestCase):
    """#167: the kernel the report calls "default" is the one engine/src/gen.rs documents as the default."""

    def test_default_kernel_matches_gen_rs(self):
        import re
        src = (REPO / "engine" / "src" / "gen.rs").read_text(encoding="utf-8")
        m = re.search(r"CROW_STAGE_KERNEL \(default (\d+) = (\w+)", src)
        self.assertIsNotNone(m, "gen.rs no longer documents the CROW_STAGE_KERNEL default as 'default N = name'")
        defaults = [lab for lab, _ in ts.STAGE_CEILINGS if lab.endswith(", default")]
        self.assertEqual(len(defaults), 1, ts.STAGE_CEILINGS)
        self.assertEqual(defaults[0].split(" kernel")[0], m.group(2),
                         "STAGE_CEILINGS names a different default kernel than gen.rs (CROW_STAGE_KERNEL default %s)"
                         % m.group(1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
