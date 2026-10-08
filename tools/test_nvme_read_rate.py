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

#171 adds the queue-depth grid (FILE_FLAG_OVERLAPPED, depth per reader, IOCP or IoRing). Its pure
parts are tested with a fake backend and a fake clock (the in-flight loop keeps exactly `depth`
reads outstanding, counts only completions inside the window, drains after the deadline), plus the
block presets, the depth list, the B(depth, readers) table and the drive-temperature parser. On
Windows a 32 MiB synthetic file runs the grid end to end.
"""

import contextlib
import io
import json
import os
import random
import struct
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


class BlockPresets(unittest.TestCase):
    def test_presets_are_the_two_record_sizes_and_sector_multiples(self):
        self.assertEqual(nr.BLOCK_PRESETS, {"3.05bit": 9_474_048, "4.5bit": 14_155_776})
        self.assertEqual(nr.BLOCK_PRESETS["4.5bit"], nr.EXPERT_BLOCK)
        for size in nr.BLOCK_PRESETS.values():
            self.assertEqual(size % nr.ALIGN, 0)
            nr.check_geometry(size, nr.ALIGN, 512, 4096)
        self.assertEqual(nr.BLOCK_PRESETS["3.05bit"], 2313 * 4096)

    def test_parse_block(self):
        self.assertEqual(nr.parse_block("3.05bit"), 9_474_048)
        self.assertEqual(nr.parse_block("4.5bit"), 14_155_776)
        self.assertEqual(nr.parse_block("1048576"), 1 << 20)
        self.assertEqual(nr.parse_block("1_048_576"), 1 << 20)
        self.assertEqual(nr.parse_block(nr.EXPERT_BLOCK), nr.EXPERT_BLOCK)     # the argparse default
        for bad in ("", "0", "-4096", "5bit", "3.05", "x"):
            with self.assertRaises(nr.Refused, msg=bad):
                nr.parse_block(bad)


class Depths(unittest.TestCase):
    def test_parse_depths(self):
        self.assertEqual(nr.parse_depths("1,4,8,16"), [1, 4, 8, 16])
        for bad in ("", "0,1", "4,4", "a", "-1"):
            with self.assertRaises(nr.Refused, msg=bad) as cm:
                nr.parse_depths(bad)
            self.assertIn("--depths", str(cm.exception))


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeBackend:
    """FIFO device: reap() advances the clock by `step` and completes the oldest read."""

    def __init__(self, clock, block, step=1.0, short_at=()):
        self.clock, self.block, self.step, self.short_at = clock, block, step, set(short_at)
        self.queue, self.submitted, self.reaped, self.max_inflight = [], [], [], 0

    def submit(self, slot, file_index, offset):
        assert slot not in self.queue, "slot submitted twice while in flight"
        self.queue.append(slot)
        self.submitted.append((slot, file_index, offset, self.clock()))
        self.max_inflight = max(self.max_inflight, len(self.queue))

    def reap(self):
        self.clock.now += self.step
        slot = self.queue.pop(0)
        n = self.block - 4096 if len(self.reaped) in self.short_at else self.block
        self.reaped.append(slot)
        return [(slot, n)]


class QueueLoop(unittest.TestCase):
    BLOCK = 8192

    def run_loop(self, depth, t_meas=2.0, t_end=6.0, **kw):
        clock = FakeClock()
        be = FakeBackend(clock, self.BLOCK, **kw)
        count = iter(range(10_000))
        r = nr.run_queue(be, depth, lambda: (next(count) % 2, 4096 * next(count)), self.BLOCK, clock, t_meas, t_end)
        return be, r

    def test_keeps_exactly_depth_in_flight_and_drains(self):
        for depth in (1, 2, 4):
            be, r = self.run_loop(depth)
            self.assertEqual(be.max_inflight, depth)
            self.assertEqual(be.queue, [])                              # drained
            self.assertEqual(len(be.submitted), len(be.reaped))
            self.assertEqual(r["issued"], len(be.reaped) * self.BLOCK)

    def test_no_submit_at_or_after_the_deadline(self):
        be, _ = self.run_loop(2)
        self.assertTrue(all(t < 6.0 for *_, t in be.submitted), be.submitted)
        self.assertEqual(len(be.submitted), 7)                          # at t = 0, 0, 1, 2, 3, 4, 5

    def test_only_completions_inside_the_window_count(self):
        be, r = self.run_loop(2)
        # completions land at t = 1..7; the window is [2, 6): t = 2, 3, 4, 5 count
        self.assertEqual(r["counted"], 4 * self.BLOCK)
        self.assertEqual(len(r["lat"]), 4)
        self.assertEqual(r["lat"], [2.0] * 4)                           # submit to reap, per block
        self.assertEqual(r["issued"], 7 * self.BLOCK)                   # warm-up and drain are still own IO
        self.assertEqual(r["short"], 0)

    def test_depth_one_is_the_synchronous_case(self):
        be, r = self.run_loop(1, t_meas=1.0, t_end=4.0)
        self.assertEqual(be.max_inflight, 1)
        self.assertEqual(r["lat"], [1.0] * 3)                           # completions at t = 1, 2, 3

    def test_offsets_come_from_pick_unchanged(self):
        be, _ = self.run_loop(2)
        self.assertTrue(all(off % 4096 == 0 for _, _, off, _ in be.submitted))
        self.assertEqual(len({off for _, _, off, _ in be.submitted}), len(be.submitted))

    def test_short_read_is_not_counted_but_is_own_io(self):
        be, r = self.run_loop(2, short_at={2})
        self.assertEqual(r["short"], 1)
        self.assertEqual(r["counted"], 3 * self.BLOCK)
        self.assertEqual(r["issued"], 7 * self.BLOCK - 4096)

    def test_backend_error_propagates(self):
        class Boom(FakeBackend):
            def reap(self):
                raise OSError(5, "boom")
        with self.assertRaises(OSError):
            nr.run_queue(Boom(FakeClock(), 8192), 2, lambda: (0, 0), 8192, FakeClock(), 0.0, 1.0)


def gcells(table):
    """{(depth, readers): [(gbps, p50, p99), ...]} -> grid cells."""
    return [{"depth": d, "readers": r, "gbps": g, "lat_p50_ms": p50, "lat_p99_ms": p99, "blocks": 100}
            for (d, r), reps in table.items() for g, p50, p99 in reps]


class GridTable(unittest.TestCase):
    TABLE = {(1, 1): [(5.0, 2.0, 3.0), (5.1, 2.1, 3.5), (4.9, 1.9, 3.2)],
             (1, 2): [(9.0, 3.0, 5.0), (9.4, 3.1, 4.0), (9.2, 3.2, 4.5)],
             (4, 1): [(8.0, 8.0, 11.0), (8.2, 8.1, 12.5), (8.1, 8.2, 12.0)],
             (4, 2): [(9.5, 15.0, 30.0), (7.0, 16.0, 31.0), (9.4, 14.0, 29.0)]}

    def summary(self):
        return nr.grid_summary(gcells(self.TABLE), [1, 4], [1, 2])

    def test_one_row_per_depth_and_readers_in_grid_order(self):
        s = self.summary()
        self.assertEqual([(x["depth"], x["readers"]) for x in s], [(1, 1), (1, 2), (4, 1), (4, 2)])

    def test_median_spread_and_latency_per_cell(self):
        x = {(c["depth"], c["readers"]): c for c in self.summary()}
        c = x[(1, 1)]
        self.assertEqual(c["rates_gbps"], [5.0, 5.1, 4.9])
        self.assertEqual(c["median_gbps"], 5.0)
        self.assertAlmostEqual(c["spread"], 5.1 / 4.9)
        self.assertTrue(c["spread_ok"])
        self.assertEqual(c["lat_p50_ms"], 2.0)                          # median of the reps' p50
        self.assertEqual(c["lat_p99_ms"], 3.5)                          # worst rep's p99
        self.assertFalse(x[(4, 2)]["spread_ok"])                        # 9.5 / 7.0 = 1.357

    def test_matrix_is_depth_rows_by_reader_columns_of_medians(self):
        self.assertEqual(nr.grid_matrix(self.summary(), [1, 4], [1, 2]), [[5.0, 9.2], [8.1, 9.4]])

    def test_missing_cell_is_none_not_zero(self):
        s = nr.grid_summary(gcells({(1, 1): [(5.0, 1.0, 2.0)] * 3}), [1, 4], [1, 2])
        self.assertEqual(nr.grid_matrix(s, [1, 4], [1, 2]), [[5.0, None], [None, None]])

    def test_format_prints_the_matrix_spread_and_latency(self):
        text = "\n".join(nr.format_grid(self.summary(), [1, 4], [1, 2], True, "iocp", 9_474_048))
        for key in ("queue-depth grid", "backend iocp", "9,474,048", "depth", "readers", "9.200", "p50", "p99",
                    "spread", "<= 1.15?", "NO"):
            self.assertIn(key, text)
        self.assertNotIn("VOID", text)

    def test_format_labels_a_void_run(self):
        text = "\n".join(nr.format_grid(self.summary(), [1, 4], [1, 2], False, "iocp", 9_474_048))
        self.assertIn("VOID", text)

    def test_grid_only_verdict_claims_no_g1_input(self):
        v = nr.grid_only_verdict([])
        self.assertTrue(v["valid"])
        self.assertIsNone(v["b_for_g1"])
        self.assertEqual(v["per_readers"], {})
        self.assertIn("--no-sync-arm", v["g1_input"])
        v = nr.grid_only_verdict(["grid rep 0 depth 4 readers 2: disk writes 9.0 MB/s > 4.0 MB/s"])
        self.assertFalse(v["valid"])
        self.assertTrue(v["g1_input"].startswith("VOID"))


def temp_blob(sensors, critical=87, warning=84, size=None):
    head = struct.pack("<IIhhH2x8x", 24, size or 24 + 16 * len(sensors), critical, warning, len(sensors))
    return head + b"".join(struct.pack("<HhhhBBBBI", i, t, 0, 0, 0, 0, 0, 0, 0) for i, t in enumerate(sensors))


class Temperature(unittest.TestCase):
    def test_parse_one_and_several_sensors(self):
        t = nr.parse_temperature(temp_blob([41]))
        self.assertEqual((t["readable"], t["celsius"], t["critical_c"], t["warning_c"]), (True, 41, 87, 84))
        t = nr.parse_temperature(temp_blob([41, 55, 38]))
        self.assertEqual(t["celsius"], 55)                              # the hottest sensor
        self.assertEqual([s["celsius"] for s in t["sensors"]], [41, 55, 38])

    def test_unreadable_shapes_raise(self):
        for bad in (b"", temp_blob([41])[:20], temp_blob([]), temp_blob([41, 42])[:30],
                    temp_blob([300]), temp_blob([-90])):
            with self.assertRaises(ValueError, msg=bad):
                nr.parse_temperature(bad)

    def test_text(self):
        self.assertEqual(nr.temperature_text(nr.parse_temperature(temp_blob([41]))),
                         "41 C (warning 84 C, critical 87 C)")
        self.assertEqual(nr.temperature_text({"readable": False, "error": "Win32 error 5"}),
                         "not readable (Win32 error 5)")
        self.assertEqual(nr.temperature_text(None), "not readable")


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

    GRID = ["--no-sync-arm", "--seq-passes", "0", "--allow-void"]

    def test_grid_iocp_end_to_end(self):
        js = self.dir / "grid.json"
        rc, out, err = self.run_tool(*self.FAST, "--depths", "1,4", "--readers", "1,2", "--reps", "2",
                                     *self.GRID, "--json", js, self.file, self.dir / "tail.bin")
        self.assertIn(rc, (nr.EXIT_OK, nr.EXIT_VOID), err)
        r = json.loads(js.read_text(encoding="utf-8"))
        g = r["grid"]
        self.assertEqual(g["backend"], "iocp")
        self.assertEqual([(c["depth"], c["readers"]) for c in g["cells"]], [(1, 1), (1, 2), (4, 1), (4, 2)] * 2)
        for c in g["cells"]:
            self.assertGreater(c["gbps"], 0)
            self.assertEqual(c["short_reads"], 0)
            self.assertEqual(c["bytes"] % MiB, 0)
            self.assertGreater(c["blocks"], 0)
            self.assertGreater(c["lat_p99_ms"], 0)
        self.assertEqual(len(g["summary"]), 4)
        self.assertEqual(set(g["temperature"]), {"before", "after"})
        self.assertEqual(r["cells"], [])                                  # --no-sync-arm: no PREREG cells
        self.assertEqual(r["verdict"]["per_readers"], {})
        self.assertEqual(rc == nr.EXIT_OK, r["verdict"]["valid"])
        for key in ("queue-depth grid", "backend iocp", "temperature", "<= 1.15?"):
            self.assertIn(key, out)

    def test_grid_ioring_when_the_system_has_it(self):
        js = self.dir / "ring.json"
        rc, out, err = self.run_tool(*self.FAST, "--depths", "1,4", "--readers", "1", "--reps", "1",
                                     "--backend", "ioring", *self.GRID, "--json", js, self.file)
        if rc == nr.EXIT_REFUSED and "IoRing" in err:
            self.skipTest(err.strip())
        self.assertIn(rc, (nr.EXIT_OK, nr.EXIT_VOID), err)
        g = json.loads(js.read_text(encoding="utf-8"))["grid"]
        self.assertEqual(g["backend"], "ioring")
        for c in g["cells"]:
            self.assertGreater(c["gbps"], 0)
            self.assertEqual(c["short_reads"], 0)

    def test_grid_block_preset_and_void_label(self):
        js = self.dir / "preset.json"
        rc, out, _ = self.run_tool("--block", "3.05bit", "--depths", "2", "--readers", "1", "--reps", "1",
                                   "--secs", "0.3", "--warmup", "0.05", "--idle-check", "0.2", "--disk-busy",
                                   *self.GRID, "--json", js, self.file)
        self.assertEqual(rc, nr.EXIT_VOID)                                # --disk-busy voids the grid too
        r = json.loads(js.read_text(encoding="utf-8"))
        self.assertEqual(r["env"]["block"], 9_474_048)
        self.assertFalse(r["verdict"]["valid"])
        self.assertIn("VOID", out)

    def test_no_depths_means_no_grid(self):
        js = self.dir / "plain.json"
        rc, out, err = self.run_tool(*self.FAST, "--readers", "1", "--reps", "1", "--seq-passes", "0",
                                     "--allow-void", "--json", js, self.file)
        self.assertIn(rc, (nr.EXIT_OK, nr.EXIT_VOID), err)
        r = json.loads(js.read_text(encoding="utf-8"))
        self.assertNotIn("grid", r)
        self.assertEqual(len(r["cells"]), 1)
        self.assertNotIn("queue-depth grid", out)

    def test_grid_refusals(self):
        for argv in (["--depths", "0"], ["--depths", "1,1"], ["--block", "5bit"]):
            rc, _, err = self.run_tool(*self.FAST, *argv, "--reps", "1", self.file)
            self.assertEqual(rc, nr.EXIT_REFUSED, (argv, err))


if __name__ == "__main__":
    unittest.main()
