#!/usr/bin/env python3
"""#101: the unit tests of the stale-rollback log replay.

  python tools/test_cache_stale_rollbacks.py      # no engine, no GPU

The defect case is the 2026-09-22 session as #101's Evidence table quotes it (07:02:17 to
07:02:35 UTC): a rollback below the snapshot at 50567, which stayed listed and was then
restored. The fixed case is the same history as the engine logs it since 9e46fc1.
"""

import gzip
import tempfile
import unittest
from pathlib import Path

import cache_stale_rollbacks as csr


def warm(t: str, held: int, point: int, snaps: list) -> str:
    listing = ", ".join("Some(%d)" % s if s is not None else "None" for s in snaps)
    return ("2026-09-22T%s  INFO cache: [cache] WARM L %d (held 51000), P %d, snapshots [%s], "
            "reusable [%s], prefill 1 of 1 tok, reset 1.0 ms" % (t, held, point, listing, listing))


def snap(t: str, pos: int) -> str:
    return "2026-09-22T%s  INFO cache: [cache] snapshot point 1 (after prompt) at pos %d, DtoH 1.0 ms" % (t, pos)


# #101 Evidence, the first stale rollback, with the snapshots its lines imply in between.
DEFECT = [
    warm("07:02:17.541Z", 49849, 49241, [50567, 49241, 49063]),
    snap("07:02:18.000Z", 49933),
    warm("07:02:23.814Z", 50175, 49933, [49933, 50567, 49241]),
    snap("07:02:24.000Z", 50428),
    warm("07:02:35.198Z", 50953, 50567, [50428, 49933, 50567]),
]

# The same edit with the fix: 50567 is gone from the next line on, and the next turn
# restores the newest prefix of the new branch.
FIXED = [
    warm("07:02:17.541Z", 49849, 49241, [50567, 49241, 49063]),
    snap("07:02:18.000Z", 49933),
    warm("07:02:23.814Z", 50175, 49933, [49933, 49241, 49063]),
    snap("07:02:24.000Z", 50428),
    warm("07:02:35.198Z", 50953, 50428, [50428, 49933, 49241]),
]


class Scan(unittest.TestCase):
    def test_the_2026_09_22_defect_is_found(self):
        r = csr.scan(DEFECT)
        # 07:02:23 bypasses the stale 50567 a second time (L 50175)
        self.assertEqual(r["rollbacks_below_newest"], 2)
        self.assertEqual([(v["kind"], v["pos"]) for v in r["violations"]],
                         [("held", 50567), ("chosen", 50567), ("held", 50567)])

    def test_the_fixed_engine_is_clean(self):
        r = csr.scan(FIXED)
        self.assertEqual((r["decisions"], r["warm"], r["rollbacks_below_newest"]), (3, 3, 1))
        self.assertEqual(r["violations"], [])

    def test_a_fresh_snapshot_at_the_same_position_is_not_stale(self):
        lines = [warm("07:00:00.000Z", 100, 90, [120, 90, None]),
                 snap("07:00:01.000Z", 120),
                 warm("07:00:02.000Z", 130, 120, [120, 90, None])]
        self.assertEqual(csr.scan(lines)["violations"], [])

    def test_a_parked_conversation_is_not_a_rollback(self):
        """engine-20261002-061441-000.log.gz, 10:53:47 to 10:54:13 UTC (27B, Windows): a
        second conversation parks the first one's slots, which stay under `snapshots` but
        not under `reusable`. Read from `snapshots`, 16735 and 15727 looked bypassed."""
        head = "2026-10-02T%s  INFO cache: [cache] "
        lines = [
            head % "10:53:47.195Z" + "COLD L 6570 (held 17949), P 0, snapshots [Some(16735), "
            "Some(15727), Some(12843)], reusable [Some(16735), Some(15727), Some(12843)], prefill "
            "6627 of 6627 tok, held conversation parked (snapshots [Some(16735), Some(15727), Some(12843)])",
            head % "10:53:52.841Z" + "snapshot point 1 (after prompt) at pos 6627, DtoH 21.414 ms",
            head % "10:53:55.536Z" + "WARM L 6727 (held 6727), P 6627, snapshots [Some(6627), "
            "Some(16735), Some(15727)], reusable [Some(6627), None, None], parked [None, "
            "Some(16735), Some(15727)], prefill 1397 of 8024 tok",
            head % "10:53:56.768Z" + "snapshot point 1 (after prompt) at pos 8024, DtoH 13.784 ms",
            head % "10:54:13.636Z" + "WARM L 9164 (held 9164), P 8024, snapshots [Some(8024), "
            "Some(16735), Some(15727)], reusable [Some(8024), None, None], parked [None, "
            "Some(16735), Some(15727)], prefill 1208 of 9232 tok",
        ]
        r = csr.scan(lines)
        self.assertEqual((r["warm"], r["rollbacks_below_newest"], r["violations"]), (2, 0, []))

    def test_a_cold_line_and_an_engine_start_clear_the_slate(self):
        cold = ("2026-09-22T07:00:01.000Z  INFO cache: [cache] COLD L 0 (held 0), P 0, "
                "snapshots [None, None, None], reusable [None, None, None], prefill 5 of 5 tok")
        start = "2026-09-22T07:00:01.000Z  INFO log: [log] file engine.log (rotate at 64.000 MiB)"
        for reset in (cold, start):
            lines = [warm("07:00:00.000Z", 100, 90, [120, 90, None]), reset,
                     warm("07:00:02.000Z", 130, 120, [120, None, None])]
            self.assertEqual(csr.scan(lines)["violations"], [], reset)

    def test_a_directory_reads_the_rotated_files_first(self):
        with tempfile.TemporaryDirectory() as d:
            with gzip.open(Path(d) / "engine-20260922-070000-000.log.gz", "wt", encoding="utf-8") as fh:
                fh.write("\n".join(DEFECT[:3]) + "\n")
            (Path(d) / "engine.log").write_text("\n".join(DEFECT[3:]) + "\n", encoding="utf-8")
            files = csr.expand(d)
            self.assertEqual([f.name for f in files], ["engine-20260922-070000-000.log.gz", "engine.log"])
            self.assertEqual(csr.main([d]), 1)


if __name__ == "__main__":
    unittest.main()
