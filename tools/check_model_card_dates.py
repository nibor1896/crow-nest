#!/usr/bin/env python3
"""Guard: every measured number in the HF model card carries a date (F4, issue crow-nest #57).

The rule
--------
- Same record rule as ``tools/check_readme_dates.py`` (spec section 0.5,
  ``docs/architecture.md:58``): performance is a measurement, not a condition. A
  measured number without its date is not a measurement, it reads as a property
  of the product. The card must not present the 972/42 targets or any perf
  number as a property (decision record, spec section 0.1).
- So: every line of ``hf-package/README.md`` that carries a digit run of two or
  more digits must carry a date ``20\\d\\d-\\d\\d-\\d\\d`` in the SAME line, or sit
  inside a fenced code block (the converter command and its log line), or consist
  only of digit runs from the exempt list. A markdown table row IS one line in
  the raw text, so "same line or same table row" is one and the same check here.

Exempt digit runs (identifiers, not measurements)
-------------------------------------------------
The list is inherited from the E6 checker ``tools/check_readme_dates.py``:
issue numbers ``#44``, versions ``v0.1.0`` and dotted triples, source anchors
``serve.rs:446``, byte and width units ``105 GB`` and ``4.5 bpw``, architecture
names ``sm_120``, toolchain and device names ``RTX 5090``, a port
``port 8099``, an inline code span without whitespace, and data type or format
names ``BF16``, ``NVFP4``, ``ue4m3``, ``CNQ4.5``, ``Q2_K_XL``.

Deviation from the E6 checker
-----------------------------
- A Hugging Face card starts with a YAML frontmatter block between a leading
  ``---`` line and the closing ``---`` line. Those lines are hub metadata (for
  example the tag ``rtx-5090``), read by the hub as tags, not as prose, so the
  block is skipped and counted separately.

Where the card lives (2026-09-18, F5, issue crow-nest #64)
---------------------------------------------------------
- The card of record is ``docs/model-card.md``, TRACKED, and it is the byte source
  of the ``README.md`` uploaded to
  ``https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M``.
- It used to live only in ``hf-package/README.md``, which ``.gitignore`` ignores
  (F3, issue #58: that directory is hard links to the 105 GB payload). This guard
  was written for that path in F4 and was itself never committed, so
  ``tools/gate-linux.sh`` printed it as "not in this tree - skipped" on every
  Linux run. Both halves are fixed here: the guard is tracked and it checks the
  tracked card. ``hf-package/README.md`` is still checked when it exists, so a
  staged upload copy is covered too.

Usage
-----
- ``python tools/check_model_card_dates.py`` checks ``docs/model-card.md`` plus
  ``hf-package/README.md`` when that one exists, exit 0 or 1.
- ``python tools/check_model_card_dates.py <path> [...]`` checks the given files
  instead (negative control: point it at a file with an undated number).
- Output: one summary line with the number-of-numbers (the denominator, count of
  lines that carry at least one checked digit run) plus one line per offender.
  Exit 0 only at 0 offenders.
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# the tracked card of record first; the gitignored upload staging copy only when it is
# there, so the guard is green on a fresh checkout (see "Where the card lives")
DEFAULT_FILES = [
    rel
    for rel in ("docs/model-card.md", "hf-package/README.md")
    if rel == "docs/model-card.md" or (REPO / rel).is_file()
]

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
    re.compile(r"`[^`\s]+`"),                                # identifier in backticks
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
    """Return (number_lines, in_code, frontmatter, dated, exempt, offenders)."""
    text = path.read_text(encoding="utf-8", errors="replace")
    in_code = False
    frontmatter_done = False
    in_frontmatter = False
    number_lines = code = fm = dated = exempt = 0
    offenders = []
    for number, raw in enumerate(text.splitlines(), start=1):
        if not frontmatter_done:
            stripped = raw.strip()
            if number == 1 and stripped == "---":
                in_frontmatter = True
                fm += 1
                continue
            if in_frontmatter:
                if stripped == "---":
                    in_frontmatter = False
                    frontmatter_done = True
                fm += 1
                continue
            frontmatter_done = True
        if FENCE_RE.match(raw):
            in_code = not in_code
            code += 1
            continue
        if in_code:
            code += 1
            continue
        if not RUN_RE.search(raw):
            continue
        number_lines += 1
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
    return number_lines, code, fm, dated, exempt, offenders


def main(argv):
    targets = argv[1:] or DEFAULT_FILES
    total = [0, 0, 0, 0, 0]
    offenders = []
    missing = []
    for rel in targets:
        path = Path(rel)
        if not path.is_absolute():
            path = REPO / rel
        if not path.is_file():
            missing.append(rel)
            continue
        number_lines, code, fm, dated, exempt, bad = scan(path)
        total[0] += number_lines
        total[1] += code
        total[2] += fm
        total[3] += dated
        total[4] += exempt
        offenders.extend((rel, n, t) for n, t in bad)
    for rel in missing:
        print(f"MISSING {rel}")
    for rel, number, text in offenders:
        print(f"{rel}:{number}: number without a date: {text}")
    print(
        f"number-of-numbers {total[0]} number lines in {len(targets) - len(missing)} file(s): "
        f"{total[1]} in code blocks, {total[2]} in YAML frontmatter, "
        f"{total[3]} with a date, {total[4]} exempt by pattern, "
        f"{len(offenders)} offenders"
    )
    return 1 if (offenders or missing) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
