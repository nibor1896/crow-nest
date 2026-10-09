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
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_tier_sim as ts  # noqa: E402

L, E, K = ts.SHAPE
REPO = Path(__file__).resolve().parent.parent
STEP3 = REPO / "runs" / "glm53-flash" / "step03" / "20261008T001819Z.json"
KINDS = ["kda+dense"] * 3 + ["dsa+moe" if l % 4 == 3 else "kda+moe" for l in range(3, 45)]


def write_run(d, routes, ids=None, weights="cnq (CNQ container)", partial=None, complete=True, index_sha=None,
              extra=None):
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
    if index_sha:
        w["index_json_sha256"] = index_sha
    man = {"ids": ids, "T": n, "D": 0, "layers": [0, 45], "num_hidden_layers": 45, "layer_kinds": KINDS,
           "files": files, "weights": w, "complete": complete}
    man.update(extra or {})
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


def uniform_routes(n, seed, layers=L):
    """[n][layers][K]: every position and layer routes a uniform random set of K distinct experts of E."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.random((n, layers, E)).argsort(axis=2)[:, :, :K], axis=2).astype(np.uint16)


def one_layer(seq):
    return np.array(seq, np.uint16).reshape(-1, 1, 1)


class TestDynPolicies(unittest.TestCase):
    """#178: the `dyn` simulator's policies against hand sequences, MIN, the static path and uniform routing."""

    SEQ = [0, 1, 2, 1, 0, 3, 2, 1, 4, 3]                 # A B C B A D C B E D, one layer, one pick per token

    def reads(self, res):
        return (res["nvme_demand"] + res["nvme_prefetch"]).tolist()

    def test_clock_hand_sequence(self):
        # 3 slots: A B C fill (bit set); B, A hit; D sweeps A, B, C (bits cleared), evicts A, hand at B; C, B hit
        # (bits set); E sweeps B, C, D (D's bit was set on insert) and evicts B; D hits
        r = one_layer(self.SEQ)
        want = [1, 1, 1, 0, 0, 1, 0, 0, 1, 0]
        self.assertEqual(self.reads(ts.dyn_run(r, 3, 0, "clock", n_experts=8)), want)
        self.assertEqual(self.reads(ts.dyn_run(r, 0, 3, "clock", n_experts=8)), want)
        self.assertEqual(self.reads(ts.dyn_run(r, 3, 0, "lru", n_experts=8)), [1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

    def test_clock_never_evicts_the_step_picks(self):
        # 2 slots holding 0 and 1; the step picks 0, 1, 2: no victim outside the picks -> 2 is not stored
        t = ts.ClockTier(2)
        t.insert(10), t.insert(11)
        self.assertEqual(t.insert(12, protect={10, 11, 12}), (False, None))
        self.assertEqual(t.insert(12, protect={10, 12}), (True, 11))

    def test_lru_is_the_window_and_capacity_0_is_static(self):
        routes = uniform_routes(300, 5)
        every = np.ones(routes.shape, bool)
        static = ts.lru_reads(routes, every, 0)          # the static path at W = 0: every visit an NVMe read
        self.assertTrue(np.array_equal(ts.dyn_run(routes, 0, 0, "lru")["nvme_demand"], static))
        self.assertTrue((static == L * K).all())
        for cv, cp in ((8, 0), (0, 8), (108, 0), (25, 83), (83, 25)):
            got = ts.dyn_run(routes, cv, cp, "lru")["nvme_demand"]
            self.assertTrue(np.array_equal(got, ts.lru_reads(routes, every, cv + cp)), (cv, cp))

    def test_min_bounds_every_policy(self):
        routes = uniform_routes(150, 6, layers=4)
        for arena, (cv, cp) in (("layer", (10, 30)), ("global", (40, 120))):
            mn = ts.dyn_min(routes, cv + cp, arena).sum()
            for pol in ts.POLICIES:
                for pf, d in ((None, 1), ("oracle", 1), (0.5, 2), (0.9, 3)):
                    for adm in (None, 4):
                        r = ts.dyn_run(routes, cv, cp, pol, arena, adm, pf, d)
                        self.assertGreaterEqual(int((r["nvme_demand"] + r["nvme_prefetch"]).sum()), int(mn),
                                                (arena, pol, pf, d, adm))
        self.assertEqual(ts.dyn_min(one_layer(self.SEQ), 3).tolist(), [1, 1, 1, 0, 0, 1, 0, 0, 1, 0])

    def test_min_guard_refuses_a_policy_below_min(self):
        fake = {"nvme_demand": np.array([1, 0]), "nvme_prefetch": np.array([0, 0])}
        with self.assertRaisesRegex(ts.SimError, "simulator defect"):
            ts.dyn_check_min("x", fake, np.array([1, 1]))

    def test_precision_one_is_the_oracle(self):
        routes = uniform_routes(200, 7, layers=6)
        for pol, arena, cv, cp in (("lru", "layer", 4, 12), ("clock", "global", 20, 60), ("lfu", "layer", 4, 12)):
            a = ts.dyn_run(routes, cv, cp, pol, arena, prefetch="oracle", depth=2)
            b = ts.dyn_run(routes, cv, cp, pol, arena, prefetch=1.0, depth=2)
            for c in ts.DYN_COUNTERS:
                self.assertTrue(np.array_equal(a[c], b[c]), (pol, c))
            self.assertEqual((a["prefetch_useful"], a["prefetch_wasted"]), (b["prefetch_useful"], b["prefetch_wasted"]))
            low = ts.dyn_run(routes, cv, cp, pol, arena, prefetch=0.5, depth=2)
            self.assertGreater(low["prefetch_wasted"], 0)

    def test_oracle_prefetch_per_layer_hides_every_later_read(self):
        # per-layer arena, policies that protect the step's picks: nothing touches layer j + d's arena between the
        # prefetch and the use, so only the first d layers of token 0 stall and nothing is wasted
        routes = uniform_routes(100, 8, layers=6)
        for pol in ("clock", "lfu"):
            for d in (1, 2, 3):
                pf = ts.dyn_run(routes, 4, 12, pol, prefetch="oracle", depth=d)
                self.assertEqual(int(pf["nvme_demand"].sum()), d * K, (pol, d))
                self.assertEqual(pf["prefetch_wasted"], 0)
                self.assertEqual(pf["prefetch_useful"], int(pf["nvme_prefetch"].sum()))
        for pol in ts.POLICIES:                          # a budget of 0 reads per token is no prefetch
            base = ts.dyn_run(routes, 4, 12, pol)
            capped = ts.dyn_run(routes, 4, 12, pol, prefetch="oracle", depth=1, pf_budget=0)
            self.assertTrue(all(np.array_equal(capped[c], base[c]) for c in ts.DYN_COUNTERS), pol)

    def test_uniform_routing_matches_the_analytic_m(self):
        routes = uniform_routes(1200, 9)
        gen = np.arange(1200) >= 200                     # past the cold start
        c = 108
        lru = sum((E - c) / (E - k) for k in range(K)) / K   # per pick in order: 0.63273
        protect = (E - c) / E                                # the step's picks protected: 0.625
        self.assertAlmostEqual(lru, 0.63273, places=5)
        for pol, cv, cp, want in (("lru", 108, 0, lru), ("lru", 25, 83, lru), ("clock", 25, 83, protect),
                                  ("lfu", 25, 83, protect)):
            m = ts.dyn_run(routes, cv, cp, pol)["nvme_demand"][gen].mean() / (L * K)
            self.assertAlmostEqual(m, want, delta=0.004, msg=(pol, cv, cp, m, want))


class TestDynTraffic(unittest.TestCase):
    def test_capacity_from_bytes(self):
        pinned = ts.parse_bytes("46GiB")
        self.assertEqual(pinned, 46 * 2 ** 30)
        self.assertEqual(ts.capacity(0, pinned, ts.BPW_BYTES["4.5"], "layer"), (0, 83))
        self.assertEqual(ts.capacity(0, pinned, ts.BPW_BYTES["3.05"], "layer"), (0, 124))
        self.assertEqual(ts.capacity(ts.parse_bytes("25.6GB"), 0, ts.BPW_BYTES["3.05"], "layer"), (64, 0))
        self.assertEqual(ts.capacity(ts.parse_bytes("25.6GB"), 0, ts.BPW_BYTES["3.05"], "global"), (2702, 0))
        self.assertEqual(ts.BPW_BYTES["3.5"], 10_871_858)
        with self.assertRaises(ts.SimError):
            ts.parse_bytes("12 parsecs")

    def test_pcie_is_zero_copy_admissions_write_backs(self):
        routes = uniform_routes(200, 10, layers=4)
        for adm in (None, 4):
            r = ts.dyn_run(routes, 6, 20, "clock", admit_max=adm)
            visits = 4 * K
            # every visit that is not a VRAM hit crosses PCIe once (copy or zero-copy read); write-backs on top
            self.assertTrue(np.array_equal(r["zero_copy"] + r["admissions"], visits - r["vram_hits"]))
            self.assertTrue((r["write_backs"] <= r["admissions"]).all())
        off = ts.dyn_run(routes, 6, 20, "lru", admit_max=4)     # 8 picks per step > 4: admission off
        self.assertEqual(int(off["admissions"].sum() + off["vram_hits"].sum() + off["write_backs"].sum()), 0)
        on = ts.dyn_run(routes, 6, 20, "lru", admit_max=64)
        ref = ts.dyn_run(routes, 6, 20, "lru")
        self.assertTrue(all(np.array_equal(on[c], ref[c]) for c in ts.DYN_COUNTERS))

    def test_hand_traffic(self):
        # VRAM 1 slot, pinned 1 slot, one layer: A (NVMe, admitted) B (NVMe, admitted, A written back)
        # A (pinned hit, admitted, B written back) C (NVMe, admitted, A written back, B dropped)
        r = ts.dyn_run(one_layer([0, 1, 0, 2]), 1, 1, "lru", n_experts=4)
        self.assertEqual(r["nvme_demand"].tolist(), [1, 1, 0, 1])
        self.assertEqual(r["pinned_hits"].tolist(), [0, 0, 1, 0])
        self.assertEqual(r["admissions"].tolist(), [1, 1, 1, 1])
        self.assertEqual(r["write_backs"].tolist(), [0, 1, 1, 1])
        z = ts.dyn_run(one_layer([0, 1, 0, 2]), 0, 1, "lru", n_experts=4)   # no VRAM: zero-copy from pinned
        self.assertEqual(z["zero_copy"].tolist(), [1, 1, 1, 1])
        self.assertEqual(z["nvme_demand"].tolist(), [1, 1, 1, 1])

    def test_cost_model_and_rates_file(self):
        row = {"m": {"mean": 0.1}, "pcie": 0.4, "dram": 0.5}
        eb = ts.BPW_BYTES["3.05"]
        c = ts.dyn_cost(row, eb, 6.994, {"R_pcie_gbps": 51.6, "R_dram_gbps": 89.6})
        per = 336 * eb
        self.assertAlmostEqual(c["nvme"]["ceiling_tok_s"], 6.994e9 / (0.1 * per))
        self.assertAlmostEqual(c["pcie"]["ceiling_tok_s"], 51.6e9 / (0.4 * per))
        self.assertEqual(c["binding"], "nvme")
        self.assertAlmostEqual(1 / c["serial_tok_s"], 0.1 * per / 6.994e9 + 0.4 * per / 51.6e9 + 0.5 * per / 89.6e9)
        bare = ts.dyn_cost(row, eb, None, {})
        self.assertIsNone(bare["ceiling_tok_s"])
        self.assertIsNone(bare["serial_tok_s"])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rates.json")
            ts.jdump({"R_pcie_gbps": 47.78, "source": "docs/architecture.md:610"}, p)
            rates, why = ts.rates_from_file(p)
            self.assertEqual(rates, {"R_pcie_gbps": 47.78})
            self.assertIn("architecture.md:610", why)
            ts.jdump({"R_pcie_gbps": -1}, p)
            with self.assertRaises(ts.SimError):
                ts.rates_from_file(p)
        self.assertEqual(ts.rates_from_file(None), ({}, "no rates file given"))


class TestDynCli(unittest.TestCase):
    def test_dyn_end_to_end_and_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            held = uniform_routes(2500, 11)
            c = Corpus(d, held, [rows(100, CAL_PICKS)])
            out = os.path.join(d, "dyn.json")
            rc = ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir, "--slots", "108:0,25:83", "--policies",
                          "lru,clock", "--prefetch", "none,0.7", "--depths", "1", "--step3", str(STEP3),
                          "--readers", "1", "--json", out])
            self.assertEqual(rc, 0)
            doc = ts.jload(out)
            self.assertEqual(doc["held"], "held")
            self.assertAlmostEqual(doc["B"], 6.9936611328)
            lru = [r for r in doc["rows"] if r["policy"] == "lru" and r["prefetch"] is None]
            want = ts.lru_reads(held, np.ones(held.shape, bool), 108).mean() / 336
            self.assertAlmostEqual(lru[0]["m"]["mean"], want)
            # 108:0 has no pinned slots, so no prefetch rows: MIN + 2 policies; 25:83: MIN + 2 policies x 2 prefetch
            self.assertEqual(len(doc["rows"]), (1 + 2) + (1 + 2 * 2))
            self.assertIsNotNone(lru[0]["cost"]["nvme"]["ceiling_tok_s"])
            self.assertIsNone(lru[0]["cost"]["pcie"]["ceiling_tok_s"])   # no rates file: no R_PCIe
            self.assertEqual(ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir]), 2)   # no capacity
            self.assertEqual(ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir, "--slots", "8:8",
                                      "--policies", "fifo"]), 2)


FP8_WORD = "fp8 (FP8 E4M3 originals, 128x128 weight_scale_inv blocks, dequantized to f32; BF16/F32 widened)"
FP8_INDEX = "ab" * 32
FP8_PASS = {"state_dtype": "bf16", "prompt_chunk": 512}


def fp8_identity(index_sha=FP8_INDEX, shards=62):
    """weights.json as tools/glm_route_passes.py fp8_identity writes it (#179)."""
    ident = {"kind": "fp8-originals", "repo": "zai-org/GLM-5.3-Flash", "revision": ts.FP8_REVISION,
             "index_json_sha256": index_sha, "config_json_sha256": "cd" * 32,
             "shards": {"model-%05d-of-00062.safetensors" % i: "%064x" % i for i in range(1, shards + 1)}}
    ident["identity_sha256"] = ts.identity_sha256(ident)
    return ident


def fp8_corpus(root, held, cal, ident=None, book=True):
    """A corpus whose five-pass dirs are FP8-original routing with weights.json and passes.jsonl rows (#179)."""
    ident = ident or fp8_identity()
    c = Corpus(root, held, cal, weights=FP8_WORD, index_sha=FP8_INDEX, extra=FP8_PASS)
    with open(os.path.join(c.rdir, "passes.jsonl"), "w", encoding="utf-8") as bk:
        for f in c.doc["files"]:
            ts.jdump(ident, os.path.join(c.rdir, f["name"], "weights.json"))
            if book:
                bk.write(json.dumps({"name": f["name"], "ok": True, "weights": "fp8-originals",
                                        "weights_identity_sha256": ident["identity_sha256"]}) + "\n")
    return c, ident


class TestG1dSource(unittest.TestCase):
    """#178 / PREREG-dyn amendment 1: `dyn` takes FP8-original routing as the G1d source; G1's `sim` rule stays."""

    def test_dyn_takes_fp8_routing_and_sim_keeps_g1(self):
        with tempfile.TemporaryDirectory() as d:
            c, ident = fp8_corpus(d, rows(3000, HELD_PICKS), [rows(500, CAL_PICKS)])
            out = os.path.join(d, "dyn.json")
            self.assertEqual(ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir, "--slots", "8:8",
                                      "--policies", "lru", "--json", out]), 0)
            doc = ts.jload(out)
            self.assertEqual(doc["source_reasons"], [])
            self.assertEqual(doc["weights_identity_sha256"], ident["identity_sha256"])
            # the G1 record's rule is unchanged: FP8 routing is plausibility only there, G1 is not answered
            held, cal, reasons = ts.load_corpus(c.path, c.rdir)
            self.assertEqual(len(reasons), 2)
            self.assertTrue(all("plausibility only" in r for r in reasons), reasons)
            p = os.path.join(d, "s3.json")
            ts.jdump(step3_doc({1: [7.0, 7.1, 6.9]}), p)
            b, why = ts.b_from_step3(p)
            v = ts.simulate(held, cal, b, why, reasons, windows=(0,), out=quiet)["g1"]["verdict"]
            self.assertTrue(v.startswith("G1 not answered: held: routing not from the CNQ container"), v)

    def test_g1d_source_refusals(self):
        with tempfile.TemporaryDirectory() as d:
            c, ident = fp8_corpus(os.path.join(d, "ok"), rows(100, HELD_PICKS), [rows(100, CAL_PICKS)])
            held_dir = os.path.join(c.rdir, "held")
            man = ts.jload(os.path.join(held_dir, "manifest.json"))
            self.assertEqual(ts.g1d_source_reasons(man, held_dir), ([], ident["identity_sha256"]))

            def why(m, ide=None):
                if ide is not None:
                    ts.jdump(ide, os.path.join(held_dir, "weights.json"))
                return " | ".join(ts.g1d_source_reasons(m, held_dir)[0])

            cnq = copy.deepcopy(man)
            cnq["weights"]["weights"] = "cnq (CNQ container)"
            self.assertIn("not from the FP8 originals", why(cnq))
            chunk = dict(man, prompt_chunk=0)
            self.assertIn("--prompt-chunk 512", why(chunk))
            idx = copy.deepcopy(man)
            idx["weights"]["index_json_sha256"] = "ef" * 32
            self.assertIn("is not weights.json's", why(idx))
            tampered = fp8_identity()
            del tampered["shards"]["model-00062-of-00062.safetensors"]   # identity not resealed
            r = why(man, tampered)
            self.assertIn("61 shards", r)
            self.assertIn("does not match its content", r)
            other = fp8_identity()
            other["revision"] = "0" * 40
            other["identity_sha256"] = ts.identity_sha256(other)
            self.assertIn("not fp8-originals at eb9eb208", why(man, other))
            os.remove(os.path.join(held_dir, "weights.json"))
            self.assertIn("no weights.json", why(man))
            # across dirs and the book
            c2, _ = fp8_corpus(os.path.join(d, "mixed"), rows(100, HELD_PICKS), [rows(100, CAL_PICKS)])
            alt = fp8_identity()
            alt["config_json_sha256"] = "00" * 32
            alt["identity_sha256"] = ts.identity_sha256(alt)
            ts.jdump(alt, os.path.join(c2.rdir, "cal0", "weights.json"))
            self.assertIn("weights.json differs between the pass dirs", " | ".join(ts.load_corpus(
                c2.path, c2.rdir, source="g1d")[2]))
            c3, _ = fp8_corpus(os.path.join(d, "nobook"), rows(100, HELD_PICKS), [rows(100, CAL_PICKS)], book=False)
            os.remove(os.path.join(c3.rdir, "passes.jsonl"))
            self.assertIn("no passes.jsonl", " | ".join(ts.load_corpus(c3.path, c3.rdir, source="g1d")[2]))
            c4, _ = fp8_corpus(os.path.join(d, "badrow"), rows(100, HELD_PICKS), [rows(100, CAL_PICKS)],
                               ident=fp8_identity(), book=False)
            with open(os.path.join(c4.rdir, "passes.jsonl"), "w", encoding="utf-8") as bk:
                bk.write(json.dumps({"name": "held", "ok": True, "weights_identity_sha256": "12" * 32}) + "\n")
            r = " | ".join(ts.load_corpus(c4.path, c4.rdir, source="g1d")[2])
            self.assertIn("passes.jsonl row of held carries identity", r)
            self.assertIn("passes.jsonl has no ok row for cal0", r)


def write_capture(d, cal, identity, layers=range(3, 45)):
    """The conversion's capture as the copier keeps it (#182): per MoE layer L<ll>/ids.i32, the calibration files' rows
    one after another (int32 [rows][8]), and capture.json as tools/glm_mul1_quantize.py capture writes it."""
    for j, l in enumerate(layers):
        ld = os.path.join(d, "L%02d" % l)
        os.makedirs(ld)
        a = np.ascontiguousarray(np.concatenate([r[:, j, :] for _, r in cal]), dtype="<i4")
        a.tofile(os.path.join(ld, "ids.i32"))
        ts.jdump({"layer": l, "rows": len(a), "hidden": 4096, "top_k": K,
                  "files": [{"name": n, "rows": len(r)} for n, r in cal], "moe_in_sha256": "00" * 32,
                  "ids_sha256": ts.sha256_file(os.path.join(ld, "ids.i32")), "identity": identity,
                  "what": "post_attention_layernorm output ...; ids = the router's top-k, ascending"},
                 os.path.join(ld, "capture.json"))


def capture_corpus(root):
    """PREREG-dyn amendment 2: the held-out from one FP8 pass dir (+ its passes.jsonl row), the two calibration files
    only in the capture dir (their pass dirs removed). The files have different lengths, so the row split matters."""
    held, cal = rows(300, HELD_PICKS), [uniform_routes(120, 5), uniform_routes(80, 6)]
    c, ident = fp8_corpus(root, held, cal, book=False)
    with open(os.path.join(c.rdir, "passes.jsonl"), "w", encoding="utf-8") as bk:
        bk.write(json.dumps({"name": "held", "ok": True, "weights": "fp8-originals",
                             "weights_identity_sha256": ident["identity_sha256"]}) + "\n")
    for n in ("cal0", "cal1"):
        shutil.rmtree(os.path.join(c.rdir, n))
    cap = os.path.join(root, "capture-ids")
    write_capture(cap, [("cal0", cal[0]), ("cal1", cal[1])], ident["identity_sha256"])
    return c, ident, cap, cal


def edit_capture(cap, layer, fn):
    p = os.path.join(cap, "L%02d" % layer, "capture.json")
    cj = ts.jload(p)
    fn(cj)
    ts.jdump(cj, p)


class TestG1dCapture(unittest.TestCase):
    """PREREG-dyn amendment 2 (#178, #179): `dyn --capture` takes the calibration files' routing from the conversion's
    capture (L<ll>/ids.i32 + capture.json), checked by name against the held-out pass dir's FP8 identity."""

    def test_capture_accepted_and_split_per_file(self):
        with tempfile.TemporaryDirectory() as d:
            c, ident, cap, cal = capture_corpus(d)
            held, got, reasons = ts.load_corpus(c.path, c.rdir, source="g1d", capture=cap)
            self.assertEqual(reasons, [])
            self.assertEqual([r.name for r in got], ["cal0", "cal1"])
            for r, want in zip(got, cal):
                self.assertTrue(np.array_equal(r.routes, want))
                self.assertTrue(r.gen.all())                      # the corpus mask: [[0, n]]
                self.assertEqual(r.identity, ident["identity_sha256"])
                self.assertEqual(sorted(r.capture), ["L%02d" % l for l in range(3, 45)])
            self.assertEqual(held.identity, ident["identity_sha256"])
            out = os.path.join(d, "dyn.json")
            self.assertEqual(ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir, "--capture", cap, "--slots", "8:8",
                                      "--policies", "lru", "--json", out]), 0)
            doc = ts.jload(out)
            self.assertEqual(doc["source_reasons"], [])
            self.assertTrue(doc["calibration_routing"]["from"].startswith("capture"))
            self.assertEqual(doc["calibration_routing"]["files"], ["cal0", "cal1"])
            self.assertEqual(len(doc["calibration_routing"]["ids_sha256"]), 42)
            # `sim` (check 4.2 of amendment 1) takes it too, as plausibility only: G1 stays "not answered"
            held, got, reasons = ts.load_corpus(c.path, c.rdir, capture=cap)
            self.assertTrue(np.array_equal(got[1].routes, cal[1]))
            self.assertTrue(any(r.startswith("calibration files: routing from the conversion's capture") for r in reasons))
            self.assertTrue(all("plausibility only" in r for r in reasons), reasons)
            self.assertEqual(ts.main(["sim", "--corpus", c.path, "--runs", c.rdir, "--capture", cap, "--windows", "0"]), 0)

    def test_capture_refusals_by_name(self):
        cases = [
            ("identity", lambda cap: edit_capture(cap, 17, lambda cj: cj.update(identity="12" * 32)),
             ("L17", "is not the FP8 identity")),
            ("order", lambda cap: edit_capture(cap, 5, lambda cj: cj["files"].reverse()),
             ("L05", "calibration files in order")),
            ("rows", lambda cap: edit_capture(cap, 9, lambda cj: cj.update(
                files=[{"name": "cal0", "rows": 119}, {"name": "cal1", "rows": 81}])), ("L09", "rows [119, 81]")),
            ("total rows", lambda cap: edit_capture(cap, 10, lambda cj: cj.update(rows=201)), ("L10", "total 201")),
            ("sha", lambda cap: np.tile(np.arange(8, dtype="<i4"), (200, 1)).tofile(os.path.join(cap, "L20", "ids.i32")),
             ("L20", "ids.i32 sha256")),
            ("missing layer", lambda cap: shutil.rmtree(os.path.join(cap, "L44")), ("L44", "missing")),
            ("missing ids", lambda cap: os.remove(os.path.join(cap, "L03", "ids.i32")), ("layer L03 missing",)),
            ("other layer", lambda cap: edit_capture(cap, 30, lambda cj: cj.update(layer=31)), ("L30", "of layer 31")),
        ]
        with tempfile.TemporaryDirectory() as d:
            for i, (what, breakit, needles) in enumerate(cases):
                c, _ident, cap, _cal = capture_corpus(os.path.join(d, str(i)))
                breakit(cap)
                with self.assertRaises(ts.SimError, msg=what) as cm:
                    ts.load_corpus(c.path, c.rdir, source="g1d", capture=cap)
                for nd in needles:
                    self.assertIn(nd, str(cm.exception), what)
                if what == "identity":   # the CLI refuses with exit 2
                    self.assertEqual(ts.main(["dyn", "--corpus", c.path, "--runs", c.rdir, "--capture", cap,
                                              "--slots", "8:8", "--policies", "lru"]), 2)
            # a row that is not 8 distinct ascending ids, with a matching sha256: the routing self-test
            c, _ident, cap, _cal = capture_corpus(os.path.join(d, "selftest"))
            p = os.path.join(cap, "L12", "ids.i32")
            a = np.fromfile(p, "<i4").reshape(-1, 8)
            a[3] = a[3][::-1]
            a.tofile(p)
            edit_capture(cap, 12, lambda cj: cj.update(ids_sha256=ts.sha256_file(p)))
            with self.assertRaises(ts.SimError) as cm:
                ts.load_corpus(c.path, c.rdir, source="g1d", capture=cap)
            self.assertIn("self-test", str(cm.exception))
            # no identity to check against: the held-out pass dir lacks weights.json
            c, _ident, cap, _cal = capture_corpus(os.path.join(d, "noident"))
            os.remove(os.path.join(c.rdir, "held", "weights.json"))
            with self.assertRaises(ts.SimError) as cm:
                ts.load_corpus(c.path, c.rdir, source="g1d", capture=cap)
            self.assertIn("cannot be checked", str(cm.exception))


G1D_TABLE = {   # PREREG-dyn "Capacity from bytes": C per layer at R 46 / 50 / 54 GiB, V25 then V37
    ("4.5", "V25"): (108, 115, 122), ("3.5", "V25"): (138, 148, 157), ("3.05", "V25"): (161, 172, 183),
    ("4.5", "V37"): (120, 127, 134), ("3.5", "V37"): (154, 163, 172), ("3.05", "V37"): (179, 190, 201)}


def k1(seq):
    """one layer, one pick per token -> ids [n][1]"""
    return np.array(seq, np.int64)[:, None]


def mask(e, on):
    m = np.zeros(e, bool)
    m[list(on)] = True
    return m


def small_run(name, routes, gen=None):
    return ts.Run(name, "t", routes, np.ones(len(routes), bool) if gen is None else gen)


class TestG1dModel(unittest.TestCase):
    """PREREG-dyn cache model, policies and selection (#178), known answers on hand traces."""

    def test_capacity_and_bar_are_the_prereg_table(self):
        for (bpw, v), cs in G1D_TABLE.items():
            self.assertEqual(tuple(ts.g1d_capacity(bpw, r, v) for r in (46, 50, 54)), cs, (bpw, v))
        b = 6.9936611328                    # amendment 5: 1 reader
        self.assertEqual(round(ts.g1d_bar(b, ts.G1D_S["3.05"]), 2), 18.45)
        self.assertEqual(round(ts.g1d_bar(b, ts.G1D_S["4.5"]), 2), 12.35)
        self.assertEqual(round(ts.g1d_bar(b, ts.G1D_S["3.5"]), 2), 15.88)
        self.assertEqual(round(b * 1e9 / 40 / 1e6, 2), 174.84)   # B / 40 tok/s in MB
        self.assertEqual(ts.G1D_PRIMARY, ("3.05", 46, "V25"))
        self.assertEqual(len(ts.g1d_cells("all")), 18)

    def test_grid_and_tie_order(self):
        g = ts.g1d_grid()
        self.assertEqual(len(g), 3 + 3 * 3 * 3)        # s x (P 0 + P {4,8,16} x d {1,2,4})
        self.assertEqual(g[:3], [(0, 0.0, 0, None), (1, 0.25, 0, None), (1, 0.5, 0, None)])
        self.assertEqual(g[3], (2, 0.0, 4, 1))
        self.assertEqual(ts.g1d_choose({pt: 1.0 for pt in g}, g), (0, 0.0, 0, None))   # all equal: LRU
        eq = {pt: (0.5 if pt[0] == 2 and pt[1] == 0.25 and pt[2] == 8 else 2.0) for pt in g}
        self.assertEqual(ts.g1d_choose(eq, g), (2, 0.25, 8, 1))                          # then the smaller d
        eq = {pt: (0.5 if pt[0] == 1 else 2.0) for pt in g}
        self.assertEqual(ts.g1d_choose(eq, g), (1, 0.25, 0, None))                      # then the smaller s
        eq = {pt: (0.5 if pt[0] == 2 and pt[1] == 0.5 else 0.9 if pt[0] == 2 else 2.0) for pt in g}
        self.assertEqual(ts.g1d_choose(eq, g), (2, 0.5, 4, 1))                          # then the smaller P
        self.assertEqual(ts.g1d_split(161, (1, 0.25, 0, None)), (40, 121))               # floor(s x C)
        self.assertEqual(ts.g1d_split(161, (2, 0.5, 16, 2)), (80, 65))

    def test_seed_lru_hand_trace(self):
        seq = [0, 1, 2, 0, 3, 1, 2]
        r, _, _ = ts.g1d_replay_layer(k1(seq), mask(6, ()), 3)
        self.assertEqual(r.tolist(), [1, 1, 1, 0, 1, 1, 1])           # LRU of 3
        r, _, _ = ts.g1d_replay_layer(k1(seq), mask(6, {0}), 2)
        self.assertEqual(r.tolist(), [0, 1, 1, 0, 1, 1, 1])           # expert 0 seeded, LRU of 2
        r, _, _ = ts.g1d_replay_layer(k1([1, 2, 3, 4, 5, 0]), mask(6, {0}), 1)
        self.assertEqual(r.tolist(), [1, 1, 1, 1, 1, 0])              # the seed is never evicted
        r, _, _ = ts.g1d_replay_layer(k1([1, 1, 1]), mask(6, ()), 0)
        self.assertEqual(r.tolist(), [1, 1, 1])                       # no LRU slot: nothing kept

    def test_prefetch_hand_trace(self):
        E = 6
        seq = [1, 2, 3, 4]

        def keys_for(targets):            # key rows whose highest key is the target expert
            k = np.tile(np.arange(E, 0, -1, dtype=np.int64), (len(targets), 1))
            for t, e in enumerate(targets):
                k[t, e] = 100
            return k

        r, iss, used = ts.g1d_replay_layer(k1(seq), mask(E, ()), 2, 1, keys_for(seq))
        self.assertEqual((r.tolist(), iss, used), ([1, 1, 1, 1], 4, 4))   # one read each, all used, no demand miss
        r, iss, used = ts.g1d_replay_layer(k1(seq), mask(E, ()), 2, 1, keys_for([5, 5, 0, 0]))
        self.assertEqual((r.tolist(), iss, used), ([2, 2, 2, 2], 4, 0))   # wasted prefetches count as reads
        # a buffered expert not demanded is dropped after the visit: 5 prefetched at t0, demanded at t1 -> a miss
        r, _, _ = ts.g1d_replay_layer(k1([1, 5]), mask(E, ()), 2, 1, keys_for([5, 1]))
        self.assertEqual(r.tolist(), [2, 2])
        # a resident expert is never proposed: at t1, 1 is in the LRU part, so the next key (expert 0) is read and used
        r, iss, used = ts.g1d_replay_layer(k1([1, 0]), mask(E, ()), 2, 1, keys_for([1, 1]))
        self.assertEqual((r.tolist(), iss, used), ([1, 1], 2, 2))
        # a used prefetch joins the LRU part: 3 prefetched + used at t0 is an LRU hit at t1 without prefetch reads
        r, _, _ = ts.g1d_replay_layer(k1([3, 3]), mask(E, ()), 1, 1, keys_for([3, 3]))
        self.assertEqual(r.tolist(), [1, 1])                          # t1: 3 resident, the prefetch reads key 0

    def test_idpf_table_keys_and_layers(self):
        shape = (6, 10, 2)
        # layer j+1 holds layer j's ids + 1 (mod 10); calibration generated positions only
        base = np.array([[0, 5], [2, 7], [1, 3]], np.int64)
        rts = np.zeros((3, 6, 2), np.uint16)
        for j in range(6):
            rts[:, j, :] = np.sort((base + j) % 10, axis=1)
        gen = np.array([True, True, False])
        st = ts.G1dStats([small_run("c", rts, gen)], shape)
        t = st.tables[1][0]
        self.assertEqual(int(t.sum()), 2 * 2 * 2)                   # 2 generated positions x 2 x 2 pairs
        self.assertEqual((t[0, 1], t[0, 6], t[5, 1], t[1, 2]), (1, 1, 1, 0))  # the prompt row [1, 3] is not counted
        self.assertEqual(len(st.tables[4]), 2)                      # source layers 0..L-1-d
        keys = ts.g1d_keys(rts, 1, 1, st)
        self.assertEqual(int(np.argmax(keys[0])), 1)                # from [0, 5]: scores 1 at 1 and 6, tie -> lower id
        self.assertEqual(sorted(np.argsort(-keys[0])[:2].tolist()), [1, 6])
        # layer j < d gets no prefetch: with P 1, d 4 the first four layers read only demand
        res = ts.g1d_replay(rts, 6, [(2, 0.0, 1, 4)], st)[(2, 0.0, 1, 4)]
        self.assertEqual(res[0][:, :4].tolist(), [[2] * 4, [2] * 4, [2] * 4])
        self.assertEqual(res[1], 3 * 2)                              # 3 tokens x layers 4, 5
        with self.assertRaises(ts.SimError):                         # seed + P above C
            ts.g1d_replay(rts, 3, [(2, 0.5, 4, 1)], st)

    def test_selection_loo_and_held_guard(self):
        shape = (6, 40, 2)
        rng = np.random.default_rng(3)

        def rr(n):
            return np.sort(np.stack([np.stack([rng.choice(40, 2, replace=False) for _ in range(6)])
                                     for _ in range(n)]), axis=2).astype(np.uint16)

        cal = [small_run("c%d" % i, rr(60)) for i in range(3)]
        held = small_run("held", rr(60))
        chosen, stats, sel = ts.g1d_select(cal, 30, held=held, shape=shape)
        self.assertEqual(stats.names, ["c0", "c1", "c2"])
        pts = [p for p in ts.g1d_grid() if ts.g1d_split(30, p)[1] >= 0]
        self.assertEqual(len(sel["grid"]), len(pts))                 # s .5 + P 16 = 31 > 30: skipped
        self.assertEqual(sel["skipped_points"], [ts.g1d_point_name(p) for p in ts.g1d_grid() if p not in pts])
        # the score is the mean over files of each file's reads per generated token, each replayed alone
        row = [x for x in sel["grid"] if x["point"] == list(chosen)][0]
        own = [float(ts.g1d_replay(r.routes, 30, [chosen], stats)[chosen][0].sum(1).mean()) for r in cal]
        self.assertAlmostEqual(row["score"], float(np.mean(own)))
        self.assertEqual(chosen, ts.g1d_choose({tuple(x["point"]): x["score"] for x in sel["grid"]}, pts))
        # folds: chosen on the other two with their own statistics, scored on the one left out
        self.assertEqual([f["left_out"] for f in sel["folds"]], ["c0", "c1", "c2"])
        f0 = sel["folds"][0]
        st0 = ts.G1dStats(cal[1:], shape)
        ch0 = tuple(f0["point"])
        self.assertAlmostEqual(f0["left_out_r"], ts.g1d_scores([cal[0]], 30, [ch0], st0)[ch0][0])
        self.assertEqual(f0["equals_full_choice"], ch0 == chosen)
        # the held-out never enters: other held-out routing, same choice and grid; held-out in the cal set refused
        chosen2, _, sel2 = ts.g1d_select(cal, 30, held=small_run("held", rr(60)), shape=shape)
        self.assertEqual((chosen2, sel2["grid"]), (chosen, sel["grid"]))
        with self.assertRaises(ts.SimError):
            ts.g1d_select(cal + [held], 30, held=held, shape=shape)
        # only generated positions are scored
        g = cal[0].gen.copy()
        g[:30] = False
        part = small_run("c0", cal[0].routes, g)
        sc = ts.g1d_scores([part], 30, [chosen], stats)[chosen][0]
        full = ts.g1d_replay(part.routes, 30, [chosen], stats)[chosen][0].sum(1)
        self.assertAlmostEqual(sc, float(full[30:].mean()))
        # worker processes give the same scores as one process
        self.assertEqual(ts.g1d_scores(cal, 30, pts, stats, jobs=3), ts.g1d_scores(cal, 30, pts, stats))

    def test_verdict(self):
        s = ts.G1D_S["3.05"]
        ok = {"mean": 10.0, "ci": [9.0, 18.4], "n": 26599}
        v = ts.g1d_verdict(ok, s, 6.9936611328, "B", [])
        self.assertTrue(v["verdict"].startswith("G1d passed: r_hi 18.400 <= 18.455"), v["verdict"])
        self.assertIn("awaiting robin's confirmation", v["status"])
        bad = dict(ok, ci=[9.0, 18.5])
        self.assertTrue(ts.g1d_verdict(bad, s, 6.9936611328, "B", [])["verdict"].startswith("G1d failed"))
        self.assertTrue(ts.g1d_verdict(ok, s, None, "no step-3 run given", [])["verdict"].startswith(
            "G1d not answered: B: no step-3 run given"))
        self.assertIn("no CI", ts.g1d_verdict(dict(ok, ci=[math.nan, math.nan], n=900), s, 7.0, "B", [])["verdict"])
        self.assertIn("held: no weights.json", ts.g1d_verdict(ok, s, 7.0, "B", ["held: no weights.json"])["verdict"])
        v = ts.g1d_verdict(ok, ts.G1D_S["4.5"], 6.9936611328, "B", [], primary=False)
        self.assertTrue(v["verdict"].startswith("scenario (no gate role) failed: r_hi 18.400 > 12.351"), v["verdict"])


class TestG1dCli(unittest.TestCase):
    def test_g1d_end_to_end_choice_before_held_out(self):
        with tempfile.TemporaryDirectory() as d:
            held = rows(2000, HELD_PICKS)            # the same 8 experts at every position: 336 reads at token 0 only
            cal = [uniform_routes(150, 5), uniform_routes(120, 6)]
            c, ident = fp8_corpus(d, held, cal, book=False)
            with open(os.path.join(c.rdir, "passes.jsonl"), "w", encoding="utf-8") as bk:
                bk.write(json.dumps({"name": "held", "ok": True, "weights_identity_sha256": ident["identity_sha256"]})
                         + "\n")
            for n in ("cal0", "cal1"):
                shutil.rmtree(os.path.join(c.rdir, n))
            cap = os.path.join(d, "capture-ids")
            write_capture(cap, [("cal0", cal[0]), ("cal1", cal[1])], ident["identity_sha256"])
            out = os.path.join(d, "g1d.json")
            seen = []
            real = ts.g1d_replay

            def spy(routes, *a, **k):                # the choice is in the book before the held-out is replayed
                if len(routes) == 2000:
                    seen.append(os.path.exists(out + ".choices.jsonl"))
                return real(routes, *a, **k)

            with mock.patch.object(ts, "g1d_replay", side_effect=spy):
                rc = ts.main(["g1d", "--corpus", c.path, "--runs", c.rdir, "--capture", cap, "--step3", str(STEP3),
                              "--readers", "1", "--json", out])
            self.assertEqual(rc, 0)
            self.assertTrue(seen and all(seen), seen)
            doc = ts.jload(out)
            cell = doc["cells"][0]
            self.assertEqual((cell["bpw"], cell["R_GiB"], cell["V"], cell["C"], cell["primary"]), ("3.05", 46, "V25", 161, True))
            self.assertEqual(len(cell["grid"]), 30)
            self.assertEqual(len(cell["folds"]), 2)
            self.assertNotEqual(cell["point"][0], 2)  # uniform calibration routing: id-only prefetch only wastes reads
            h = cell["held_out"]["r"]
            # token 0 reads every held-out pick that is not seeded, nothing after (C 161 >= 8, no prefetch)
            st = ts.G1dStats([small_run("cal0", cal[0]), small_run("cal1", cal[1])])
            nseed = ts.g1d_split(161, tuple(cell["point"]))[0]
            want = sum(int(st.ranks[j][e] >= nseed) for j in range(L) for e in HELD_PICKS)
            self.assertAlmostEqual(h["mean"], want / 2000)
            self.assertLessEqual(h["ci"][1], 336 / 1000 + 1e-9)       # block 0 holds the only reads
            self.assertTrue(cell["verdict"]["verdict"].startswith("G1d passed"), cell["verdict"]["verdict"])
            self.assertAlmostEqual(cell["verdict"]["bar_reads"], ts.g1d_bar(6.9936611328, 9_474_048), places=6)
            self.assertEqual(doc["calibration_routing"]["files"], ["cal0", "cal1"])
            with open(out + ".choices.jsonl", encoding="utf-8") as f:
                ch = [json.loads(x) for x in f]
            self.assertEqual((ch[0]["chosen"], ch[0]["held_out_scored"]), (cell["chosen"], False))
            self.assertEqual(ts.main(["g1d", "--corpus", c.path, "--runs", c.rdir, "--capture", cap,
                                      "--cells", "3.1:46:V25"]), 2)

if __name__ == "__main__":
    unittest.main(verbosity=2)
