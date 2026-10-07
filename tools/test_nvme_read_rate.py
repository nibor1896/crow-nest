#!/usr/bin/env python3
"""#146: tests of tools/nvme_read_rate.py.

  python -I tools/test_nvme_read_rate.py     # pure parts anywhere; the IO parts on Windows only

The pure parts decide what the tool may claim: the sector rules of FILE_FLAG_NO_BUFFERING, the
offset sampler (aligned, in bounds, every slot reachable), the PREREG step-3 verdict (median,
spread <= 1.15, B for G1 only at the best reader count and only when the run is not void), m* and
the prefill bound, and the foreign-IO arithmetic on the disk counters. The Windows half runs the
tool end to end on a synthetic temp file it deletes afterwards (a functional test, never a
measurement row), and checks the void paths: --disk-busy, a writer running beside it, a file still
held open for writing.
"""

import contextlib
import io
import json
import os
import random
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import nvme_read_rate as nr  # noqa: E402

MiB = 1 << 20


class Geometry(unittest.TestCase):
    def test_expert_block_fits_this_drive(self):
        nr.check_geometry(nr.EXPERT_BLOCK, nr.ALIGN, 512, 4096)
        self.assertEqual(nr.EXPERT_BLOCK, 3 * 4096 * 2048 * 36 // 64)
        self.assertEqual(nr.EXPERT_BLOCK % 4096, 0)

    def test_refusals(self):
        for block, align, logical, physical in [(4097, 4096, 512, 4096), (8192, 1000, 512, 4096),
                                                (8192, 4096, 0, 4096), (8192, 4096, 512, 0),
                                                (6144, 2048, 512, 4096), (8192 + 2048, 4096, 512, 2048)]:
            with self.assertRaises(nr.Refused, msg=(block, align, logical, physical)):
                nr.check_geometry(block, align, logical, physical)


class Sampler(unittest.TestCase):
    def test_every_slot_is_aligned_in_bounds_and_distinct(self):
        sizes, block, align = [10 * 4096 + 5, 3 * 4096, 2 * 4096], 2 * 4096, 4096
        s = nr.OffsetSampler(sizes, block, align)
        self.assertEqual(s.total, 9 + 2 + 1)

        class Seq:                                   # a fake rng that walks every slot once
            def __init__(self):
                self.i = -1

            def randrange(self, n):
                self.i += 1
                return self.i

        rng, seen = Seq(), set()
        for _ in range(s.total):
            f, off = s.pick(rng)
            self.assertEqual(off % align, 0)
            self.assertLessEqual(off + block, sizes[f])
            seen.add((f, off))
        self.assertEqual(len(seen), s.total)
        self.assertIn((0, 8 * 4096), seen)           # the last start that fits in file 0
        self.assertIn((2, 0), seen)                  # a file of exactly one block

    def test_files_weighted_by_slots(self):
        s = nr.OffsetSampler([1000 * 4096, 100 * 4096], 4096, 4096)
        rng = random.Random(7)
        hits = sum(1 for _ in range(20000) if s.pick(rng)[0] == 0)
        self.assertAlmostEqual(hits / 20000, 1000 / 1100, delta=0.01)

    def test_file_smaller_than_block_refused(self):
        with self.assertRaises(nr.Refused):
            nr.OffsetSampler([nr.EXPERT_BLOCK - 4096], nr.EXPERT_BLOCK, 4096)


def cells(table):
    return [{"readers": r, "gbps": g} for r, rates in table.items() for g in rates]


class Verdict(unittest.TestCase):
    def test_spread_boundary(self):
        self.assertTrue(nr.summarize([1.0, 1.15, 1.1])["spread_ok"])
        self.assertFalse(nr.summarize([1.0, 1.151, 1.1])["spread_ok"])

    def test_b_for_g1_is_best_median(self):
        v = nr.verdict(cells({1: [5.0, 5.1, 4.9], 2: [9.0, 9.4, 9.2], 4: [9.1, 9.0, 8.9]}), [1, 2, 4], 3, [])
        self.assertTrue(v["valid"])
        self.assertEqual(v["best_readers"], 2)
        self.assertEqual(v["b_for_g1"], 9.2)
        self.assertAlmostEqual(v["per_readers"]["2"]["m_star"], 9.2 / 190.3)
        self.assertAlmostEqual(v["per_readers"]["2"]["prefill_bound_tok_s"], 39.9 * 9.2)

    def test_best_count_with_bad_spread_leaves_g1_unanswered(self):
        v = nr.verdict(cells({1: [5.0, 5.1, 4.9], 2: [7.0, 9.4, 9.2], 4: [9.1, 9.0, 8.9]}), [1, 2, 4], 3, [])
        self.assertEqual(v["best_readers"], 2)
        self.assertIsNone(v["b_for_g1"])
        self.assertIn("not answered", v["g1_input"])

    def test_void_never_yields_b(self):
        v = nr.verdict(cells({1: [5.0] * 3, 2: [9.0] * 3, 4: [9.0] * 3}), [1, 2, 4], 3, ["disk writes 50 MB/s"])
        self.assertFalse(v["valid"])
        self.assertIsNone(v["b_for_g1"])
        self.assertTrue(v["g1_input"].startswith("VOID"))

    def test_too_few_reps_or_missing_counts_is_not_prereg(self):
        v = nr.verdict(cells({1: [5.0, 5.0], 2: [9.0, 9.0], 4: [9.0, 9.0]}), [1, 2, 4], 2, [])
        self.assertIsNone(v["b_for_g1"])
        v = nr.verdict(cells({1: [5.0] * 3, 2: [9.0] * 3}), [1, 2], 3, [])
        self.assertIsNone(v["b_for_g1"])
        self.assertIn("[4]", v["g1_input"])

    def test_derived_constants(self):
        self.assertAlmostEqual(nr.derived(1.0)["m_star"], 0.005255, places=6)   # 0.53 % per GB/s
        self.assertTrue(nr.derived(3.76)["prefill_line_ok"])                     # 150.02 tok/s
        self.assertFalse(nr.derived(3.75)["prefill_line_ok"])                    # 149.6 tok/s

    def test_parse_readers(self):
        self.assertEqual(nr.parse_readers("1,2,4"), [1, 2, 4])
        for bad in ("", "0,1", "1,1", "a"):
            with self.assertRaises(nr.Refused):
                nr.parse_readers(bad)


class DownloadSigns(unittest.TestCase):
    def test_partial_files_and_locks(self):
        names = ["model-00001-of-00062.safetensors", "model-00002-of-00062.safetensors.part",
                 "fetch.lock", "fetch.log", "config.json", "x.TMP", "a.incomplete", "b.partial"]
        self.assertEqual(nr.download_signs(names), ["a.incomplete", "b.partial", "fetch.lock",
                                                    "model-00002-of-00062.safetensors.part", "x.TMP"])
        self.assertEqual(nr.download_signs(["model-00001-of-00062.safetensors", "fetch.log"]), [])


class ForeignIO(unittest.TestCase):
    def test_own_reads_subtracted(self):
        f = nr.foreign_io({"read": 0, "written": 0}, {"read": 10_000_000_000, "written": 1_000_000},
                          own_bytes=9_990_000_000, seconds=10, max_mbps=8)
        self.assertEqual(f["foreign_read_bytes"], 10_000_000)
        self.assertEqual(f["void_reasons"], [])

    def test_reader_or_writer_beside_us_voids(self):
        f = nr.foreign_io({"read": 0, "written": 0}, {"read": 500_000_000, "written": 0}, 0, 10, 8)
        self.assertEqual(len(f["void_reasons"]), 1)
        f = nr.foreign_io({"read": 0, "written": 0}, {"read": 0, "written": 200_000_000}, 0, 10, 8)
        self.assertIn("disk writes", f["void_reasons"][0])


@unittest.skipUnless(sys.platform == "win32", "FILE_FLAG_NO_BUFFERING path is Windows only")
class Functional(unittest.TestCase):
    """End to end on a synthetic file in a temp dir; small cells, so the rates mean nothing."""

    FAST = ["--block", str(MiB), "--secs", "0.2", "--warmup", "0.05", "--idle-check", "0.2"]

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.file = self.dir / "synthetic.bin"
        with open(self.file, "wb") as f:
            for _ in range(32):
                f.write(os.urandom(MiB))
        with open(self.dir / "tail.bin", "wb") as f:       # a size that is no sector multiple
            f.write(os.urandom(3 * MiB + 1234))

    def tearDown(self):
        self.tmp.cleanup()

    def run_tool(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = nr.main([str(a) for a in argv])
        return rc, out.getvalue(), err.getvalue()

    def test_end_to_end(self):
        js = self.dir / "run.json"
        rc, out, err = self.run_tool(*self.FAST, "--readers", "1,2,4", "--reps", "3", "--seq-passes", "2",
                                     "--allow-void", "--json", js, self.file, self.dir / "tail.bin")
        self.assertIn(rc, (nr.EXIT_OK, nr.EXIT_VOID), err)
        r = json.loads(js.read_text(encoding="utf-8"))
        v = r["verdict"]
        self.assertEqual(rc == nr.EXIT_OK, v["valid"])
        self.assertEqual(len(r["cells"]), 9)
        self.assertEqual([c["readers"] for c in r["cells"]], [1, 2, 4] * 3)     # interleaved
        for c in r["cells"]:
            self.assertGreater(c["gbps"], 0)
            self.assertEqual(c["short_reads"], 0)
            self.assertEqual(c["bytes"] % MiB, 0)
        total = 32 * MiB + 3 * MiB + 1234
        self.assertEqual([s["bytes"] for s in r["sequential"]], [total, total])
        self.assertEqual(set(v["per_readers"]), {"1", "2", "4"})
        self.assertEqual(r["env"]["sector_physical"] % 512, 0)
        for key in ("B = ", "spread", "<= 1.15?", "m* = B/190.3", "prefill bound 39.9 x B", "G1 input:"):
            self.assertIn(key, out)
        self.assertEqual(sum(1 for line in out.splitlines() if line.startswith("rep ")), 9)

    def test_disk_busy_refuses_before_reading(self):
        js = self.dir / "busy.json"
        rc, out, err = self.run_tool(*self.FAST, "--reps", "1", "--disk-busy", "--json", js, self.file)
        self.assertEqual(rc, nr.EXIT_VOID)
        self.assertIn("--disk-busy", err)
        self.assertNotIn("rep 0", out)
        self.assertFalse(js.exists())

    def test_disk_busy_with_allow_void_is_labelled_void(self):
        js = self.dir / "busy.json"
        rc, out, _ = self.run_tool(*self.FAST, "--reps", "3", "--seq-passes", "0", "--disk-busy",
                                   "--allow-void", "--json", js, self.file)
        self.assertEqual(rc, nr.EXIT_VOID)
        v = json.loads(js.read_text(encoding="utf-8"))["verdict"]
        self.assertFalse(v["valid"])
        self.assertIsNone(v["b_for_g1"])
        self.assertIn("run VOID", out)
        self.assertNotIn("run VALID", out)

    def test_writer_beside_the_run_voids_it(self):
        stop = threading.Event()
        other = tempfile.TemporaryDirectory()                # not beside the shard: counters only
        self.addCleanup(other.cleanup)
        victim = Path(other.name) / "writer.bin"

        def writer():
            chunk = os.urandom(8 * MiB)
            with open(victim, "wb") as f:
                while not stop.is_set():
                    f.seek(0)
                    f.write(chunk)
                    f.flush()
                    os.fsync(f.fileno())

        t = threading.Thread(target=writer)
        t.start()
        try:
            time.sleep(0.2)
            js = self.dir / "w.json"
            rc, _, _ = self.run_tool("--block", str(MiB), "--secs", "1.0", "--warmup", "0.05",
                                     "--idle-check", "0.5", "--readers", "1", "--reps", "1",
                                     "--seq-passes", "0", "--allow-void", "--json", js, self.file)
        finally:
            stop.set()
            t.join()
        self.assertEqual(rc, nr.EXIT_VOID)
        reasons = json.loads(js.read_text(encoding="utf-8"))["verdict"]["void_reasons"]
        self.assertTrue(any("disk writes" in x for x in reasons), reasons)

    def test_download_beside_the_shards_refuses_before_reading(self):
        (self.dir / "model-00002-of-00062.safetensors.part").write_bytes(b"x")
        rc, out, err = self.run_tool(*self.FAST, "--reps", "1", self.file)
        self.assertEqual(rc, nr.EXIT_VOID)
        self.assertIn("download in progress", err)
        self.assertNotIn("rep 0", out)

    def test_entry_appearing_beside_the_shards_voids_the_run(self):
        def later():
            time.sleep(0.4)
            (self.dir / "late.bin").write_bytes(b"x")

        t = threading.Thread(target=later)
        t.start()
        js = self.dir / "late.json"
        rc, _, _ = self.run_tool("--block", str(MiB), "--secs", "1.0", "--warmup", "0.05", "--idle-check", "0.1",
                                 "--readers", "1", "--reps", "1", "--seq-passes", "0", "--allow-void",
                                 "--json", js, self.file)
        t.join()
        self.assertEqual(rc, nr.EXIT_VOID)
        reasons = json.loads(js.read_text(encoding="utf-8"))["verdict"]["void_reasons"]
        self.assertTrue(any("late.bin" in x for x in reasons), reasons)

    def test_file_open_for_writing_is_refused(self):
        with open(self.file, "r+b"):
            rc, _, err = self.run_tool(*self.FAST, "--reps", "1", "--allow-void", self.file)
        self.assertEqual(rc, nr.EXIT_REFUSED)
        self.assertIn("open for writing", err)

    def test_unbuffered_flag_takes(self):
        self.assertTrue(nr.unbuffered_took(str(self.file)))

    def test_refusals(self):
        rc, _, err = self.run_tool("--block", "4097", "--reps", "1", self.file)
        self.assertEqual(rc, nr.EXIT_REFUSED, err)
        rc, _, err = self.run_tool("--block", str(64 * MiB), "--reps", "1", self.file)
        self.assertEqual(rc, nr.EXIT_REFUSED)
        self.assertIn("smaller than one block", err)


if __name__ == "__main__":
    unittest.main()
