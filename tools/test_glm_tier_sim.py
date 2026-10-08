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


if __name__ == "__main__":
    unittest.main(verbosity=2)
