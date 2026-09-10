#!/usr/bin/env python3
"""Guard: every CROW_* name in the engine sources has a row in docs/env.md.

What it checks
--------------
- code set: every distinct token matching ``CROW_[A-Z0-9_]+`` in ``engine/src``
  and ``converter/src``, whether it is read by ``std::env::var``, by a helper
  (``env_on``, ``num``, ``f``, ``u``) or only named in a comment.
- doc set: the first cell of every Markdown table row in ``docs/env.md`` whose
  first cell is exactly one backticked ``CROW_*`` name.
- both differences must be empty; any difference exits 1.

Notes
-----
- ``engine/src/cuda.rs`` holds non-UTF-8 bytes, so every file is read with
  ``errors='replace'``.
- No third-party imports. Paths are resolved relative to this file, so the
  script runs from any working directory, on Windows and on Linux.

How E7 runs it
--------------
- ``python tools/check_env_docs.py`` in CI, no GPU needed, exit code is the gate.
- Adding a ``CROW_*`` name to the sources without a row in ``docs/env.md``
  fails that job, and so does a row for a name that no longer exists.

Options
-------
- ``--list``            print the sorted code list, one name per line, exit 0.
- ``--doc <path>``      check against another copy of the doc (negative control).
"""

import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SRC_DIRS = ["engine/src", "converter/src"]
DEFAULT_DOC = "docs/env.md"

NAME_RE = re.compile(r"CROW_[A-Z0-9_]+")
ROW_RE = re.compile(r"^\s*\|\s*`(CROW_[A-Z0-9_]+)`\s*\|")


def code_names(repo):
    """Every distinct CROW_* token in the engine and converter sources."""
    found = set()
    for rel in SRC_DIRS:
        root = repo / rel
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file():
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            found.update(NAME_RE.findall(text))
    return found


def doc_names(doc_path):
    """Every CROW_* name that owns a table row in the doc."""
    text = doc_path.read_text(encoding="utf-8", errors="replace")
    found = set()
    for line in text.splitlines():
        m = ROW_RE.match(line)
        if m:
            found.add(m.group(1))
    return found


def main(argv):
    doc_path = REPO / DEFAULT_DOC
    if "--doc" in argv:
        i = argv.index("--doc")
        if i + 1 >= len(argv):
            print("error: --doc needs a path")
            return 2
        doc_path = Path(argv[i + 1])
        if not doc_path.is_absolute():
            doc_path = (Path.cwd() / doc_path).resolve()

    code = code_names(REPO)

    if "--list" in argv:
        for name in sorted(code):
            print(name)
        return 0

    if not doc_path.is_file():
        print("error: doc not found: %s" % doc_path)
        return 2

    doc = doc_names(doc_path)
    only_code = sorted(code - doc)
    only_doc = sorted(doc - code)

    print("doc: %s" % doc_path)
    print("code %d, doc %d" % (len(code), len(doc)))
    print("code-doc: %s" % only_code)
    print("doc-code: %s" % only_doc)

    if only_code or only_doc:
        print("FAIL: code and doc differ")
        return 1
    print("OK: code and doc agree")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
