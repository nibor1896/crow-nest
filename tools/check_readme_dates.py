#!/usr/bin/env python3
"""Guard: no number in a README without a machine and a date (E6, issue crow-nest #47).

The rule
--------
- Record rule (spec section 0.5, ``docs/architecture.md:58``): performance is a
  measurement, not a condition. A measured number without its date is not a
  measurement, it reads as a property of the product.
- So: every README line that carries a digit run of two or more digits must
  carry a date ``20\\d\\d-\\d\\d-\\d\\d`` in the SAME line, or sit inside a fenced
  code block, or consist only of digit runs from the exempt list below.

Exempt digit runs (they are identifiers, not measurements)
----------------------------------------------------------
- issue numbers ``#47``
- versions ``v0.1.0`` and dotted triples ``1.97.0``, ``0.19.9``
- source anchors ``serve.rs:446``, ``:446-447``, ``:23,25``
- byte and width units ``64 KiB``, ``16 MiB``, ``105 GB``, ``4.5 bpw``
- architecture names ``sm_120``, ``compute_120a``
- toolchain and device names ``RTX 5090``, ``CUDA 13.3``, ``Rust 1.97.0``
- a port ``port 8099``
- an inline code span without whitespace: a path, a flag, a symbol, a file name.
  It is code, the same way a fenced block is code.
- a data type or format name: ``BF16``, ``FP8``, ``NVFP4``, ``E2M1``, ``ue4m3``,
  ``f32``, ``CNQ4.5``, ``Q2_K_XL``.

Deviation from the E6 brief (2026-09-11)
----------------------------------------
- The brief enumerated the first eight exemptions. The last two were added while
  the READMEs were written, because a file name and a data type name carry no
  measurement, and dating them would be a false provenance.
- An inline code span WITH whitespace stays checked, so a phrase in backticks
  still needs its date.

Usage
-----
- ``python tools/check_readme_dates.py`` checks the three READMEs, exit 0 or 1.
- ``python tools/check_readme_dates.py <path> [...]`` checks the given files
  instead (negative control: point it at a file with an undated number).
- Output is one summary line plus one line per offender.

How E7 runs it
--------------
- ``python tools/check_readme_dates.py`` in CI, no GPU needed, exit code is the gate.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_FILES = ["README.md", "engine/README.md", "converter/README.md"]

FENCE_RE = re.compile(r"^\s*```")
DATE_RE = re.compile(r"20\d\d-\d\d-\d\d")
RUN_RE = re.compile(r"\d{2,}")

ALLOW = [
    re.compile(r"#\d+"),                                     # issue number
    re.compile(r"\bv\d+(?:\.\d+)*\b"),                       # v0.1.0
    re.compile(r"\b\d+\.\d+\.\d+\b"),                        # 1.97.0, 0.19.9
    re.compile(r":\d+(?:\s*[-,]\s*\d+)*"),                   # file:line anchors
    re.compile(
        r"\b\d[\d,]*(?:\.\d+)?\s?(?:B|KB|MB|MiB|GB|GiB|TB|bpw)\b"
    ),                                                       # sizes and widths
    re.compile(r"\bsm_\d+\b"),
    re.compile(r"\bcompute_\d+[a-z]?\b"),
    re.compile(r"\bRTX \d+\b"),
    re.compile(r"\bCUDA \d+(?:\.\d+)*\b"),
    re.compile(r"\bRust \d+(?:\.\d+)*\b"),
    re.compile(r"\bports?\D{0,4}`?\d{4,5}`?"),               # port 8099
    re.compile(r"`(?=[^`\s]*[A-Za-z_/])[^`\s]+`"),           # identifier in backticks (a purely numeric span is not exempt)
    re.compile(
        r"\b(?:BF|FP|NVFP|IQ|INT|UINT)\d+(?:\.\d+)?\b"
        r"|\b[EeMm]\d+[Mm]\d+\b"
        r"|\bue\d+m\d+\b"
        r"|\bf\d+\b"
        r"|\bCNQ\d+(?:\.\d+)?\b"
        r"|\bQ\d+_K(?:_[A-Z]+)*\b"
    ),                                                       # dtype and format names
]


def scan(path):
    """Return (checked, in_code, dated, exempt, offenders) for one file."""
    text = path.read_text(encoding="utf-8", errors="replace")
    in_code = False
    checked = code = dated = exempt = 0
    offenders = []
    for number, raw in enumerate(text.splitlines(), start=1):
        checked += 1
        if FENCE_RE.match(raw):
            in_code = not in_code
            code += 1
            continue
        if in_code:
            code += 1
            continue
        if not RUN_RE.search(raw):
            continue
        if DATE_RE.search(raw):
            dated += 1
            continue
        rest = raw
        for pattern in ALLOW:
            rest = pattern.sub(" ", rest)
        if RUN_RE.search(rest):
            offenders.append((number, raw.strip()))
        else:
            exempt += 1
    return checked, code, dated, exempt, offenders


def main(argv):
    targets = argv[1:] or DEFAULT_FILES
    total = [0, 0, 0, 0]
    offenders = []
    missing = []
    for rel in targets:
        path = Path(rel)
        if not path.is_absolute():
            path = REPO / rel
        if not path.is_file():
            missing.append(rel)
            continue
        checked, code, dated, exempt, bad = scan(path)
        total[0] += checked
        total[1] += code
        total[2] += dated
        total[3] += exempt
        offenders.extend((rel, n, t) for n, t in bad)
    for rel in missing:
        print(f"MISSING {rel}")
    for rel, number, text in offenders:
        print(f"{rel}:{number}: number without a date: {text}")
    print(
        f"checked {total[0]} lines in {len(targets) - len(missing)} files: "
        f"{total[1]} in code blocks, {total[2]} with a date, "
        f"{total[3]} exempt by pattern, {len(offenders)} offenders"
    )
    return 1 if (offenders or missing) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
