#!/usr/bin/env python3
"""#101: replay engine logs and find prefix-cache snapshots that outlived a rollback.

  python tools/cache_stale_rollbacks.py <engine.log | dir | *.log.gz> ...

Every `[cache] WARM|COLD` line lists the held snapshots BEFORE its decision, `L` is the
prefix shared with the request, `P` the point it restores. The check reads `reusable`, the
slots of the conversation this request continues; a slot of a PARKED conversation (listed
under `parked`, its KV rows copied to the host) is another history that no rollback touches.
Logs from before `reusable` existed are read from `snapshots`. A snapshot at `q > L` is
bypassed: the rollback to `P <= L` rewrites the KV rows above `P`, so since #101
(9e46fc1, v0.4.0) `PrefixCache::rollback` forgets it. Two findings, each a violation:

- `held`:   a bypassed position is listed again by a later decision line;
- `chosen`: a later decision restores a bypassed position (the 2026-09-22 defect).

A `snapshot point ... at pos q` line makes `q` fresh again. A COLD line, a parked or
restored conversation, and an engine start (`[log] file`) begin a clean slate.

`rollbacks_below_newest` counts the decisions that bypassed at least one snapshot: only
those exercise #101, so a report with 0 of them proves nothing about it.

Files are read in the order given; a directory contributes its rotated `engine-*.log.gz`
in name order, then `engine.log`, which is how the engine rotates one stream.
Exit 1 when a violation is found, 0 otherwise.
"""

import gzip
import re
import sys
from pathlib import Path

DECISION = re.compile(r"\[cache\] (WARM|COLD) L (\d+) .*?, P (\d+), snapshots \[([^\]]*)\]"
                      r"(?:, reusable \[([^\]]*)\])?")
SNAPSHOT = re.compile(r"\[cache\] snapshot point \d+ .*? at pos (\d+)")
RESET = re.compile(r"\[log\] file |held conversation parked|restor")


def positions(listing: str) -> list[int]:
    return [int(n) for n in re.findall(r"Some\((\d+)\)", listing)]


def scan(lines) -> dict:
    """Pure: the report over one stream of log lines."""
    stale: set[int] = set()
    report = {"decisions": 0, "warm": 0, "rollbacks_below_newest": 0, "violations": []}
    for line in lines:
        if "[cache]" not in line and "[log] file " not in line:
            continue
        stamp = line[:24]
        m = DECISION.search(line)
        if m:
            kind, held_len, point = m.group(1), int(m.group(2)), int(m.group(3))
            listing = m.group(5) if m.group(5) is not None else m.group(4)
            held = positions(listing)
            report["decisions"] += 1
            if kind == "COLD":
                stale.clear()
            else:
                report["warm"] += 1
                if point in stale:
                    report["violations"].append({"at": stamp, "kind": "chosen", "pos": point})
                for q in sorted(stale.intersection(held)):
                    report["violations"].append({"at": stamp, "kind": "held", "pos": q})
                bypassed = [q for q in held if q > held_len]
                if bypassed:
                    report["rollbacks_below_newest"] += 1
                    stale.update(bypassed)
            if RESET.search(line):
                stale.clear()
            continue
        if RESET.search(line):
            stale.clear()
            continue
        s = SNAPSHOT.search(line)
        if s:
            stale.discard(int(s.group(1)))
    return report


def expand(arg: str) -> list[Path]:
    p = Path(arg)
    if p.is_dir():
        return sorted(p.glob("engine-*.log.gz")) + ([p / "engine.log"] if (p / "engine.log").is_file() else [])
    return [p]


def read_lines(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:
        yield from fh


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__.strip().splitlines()[2].strip())
        return 2
    files = [f for a in argv for f in expand(a)]
    report = scan(line for f in files for line in read_lines(f))
    for f in files:
        print("read", f)
    print("decisions %(decisions)d, warm %(warm)d, rollbacks below the newest snapshot "
          "%(rollbacks_below_newest)d, violations %(n)d" % dict(report, n=len(report["violations"])))
    for v in report["violations"]:
        print("  %(at)s %(kind)s %(pos)d" % v)
    return 1 if report["violations"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
