#!/usr/bin/env python3
"""The engine package's third-party notices: THIRD-PARTY-NOTICES.txt.

`serve` (crow-nest-engine-<v>-win-x64.zip, crow-nest-engine-<v>-linux-x64.tar.gz,
tools/pack-engine.ps1 / .sh) compiles in Rust crates from crates.io, and through
onig_sys the Oniguruma C library. Their MIT, BSD, ISC, Zlib, Unicode, BSL and
Apache terms allow a binary only together with the notice, so the package carries
this file beside NOTICE.

The file is generated, never written by hand:

  1. `cargo tree -e normal` for the `crow-nest-engine` package (engine/Cargo.toml)
     with its default features -- what `cargo build --release --bin serve` links;
     Cargo has no per-binary dependencies, so every bin of the package shares
     this set -- on each target the engine ships for. Procedural macros count
     (their output is compiled in), build scripts and dev-dependencies do not;
  2. every licence file the crate source carries (LICENSE*, COPYING*, NOTICE*,
     UNLICENSE*, the manifest's license-file), plus EXTRA_FILES for code a crate
     keeps below its root;
  3. identical texts (whitespace aside) are printed once and listed with every
     crate that uses them; an Apache-2.0 file is split into the licence terms,
     printed once per wording, and what the crate adds -- its filled-in copyright
     line, or text before or after the terms (split_apache, apache_own).

Checked here: the file is what the current engine/Cargo.lock produces, every
crate has at least one text and its SPDX expression is covered by them ("A AND
B" needs both texts), every MPL-2.0 crate gets a source line (MPL 3.2(a)), NOTICE
points at the file, the pack scripts stage no NVRTC file (the package holds no NVIDIA
file), and NOTICE names every NVRTC file tools/fetch-nvrtc.* places beside serve.

Usage:  engine_notices.py [--write] [REPO]
Exit 0 = all green (or written).  1 = at least one check failed.  2 = setup error.
"""

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = "crow-nest-engine"
TARGETS = (("x86_64-pc-windows-msvc", "Windows"), ("x86_64-unknown-linux-gnu", "Linux"))
OUT = "THIRD-PARTY-NOTICES.txt"
LICENCE_FILE = re.compile(r"(?i)^(licen[cs]e|copying|notice|unlicense)([-._].*)?$")
# Texts a crate keeps below its root: onig_sys compiles the bundled Oniguruma C
# library in (its build.rs does unless RUSTONIG_SYSTEM_LIBONIG / RUSTONIG_DYNAMIC_LIBONIG
# is set), whose own licence is oniguruma/COPYING (K.Kosako, BSD-2-Clause); the
# crate's LICENSE.md covers the Rust bindings.
EXTRA_FILES = {"onig_sys": ["oniguruma/COPYING"]}


def cargo(engine, *args):
    r = subprocess.run(["cargo", *args, "--locked", "--manifest-path",
                        os.path.join(engine, "Cargo.toml")],
                       capture_output=True, text=True, encoding="utf-8")
    if r.returncode != 0:
        raise RuntimeError("cargo %s: exit %d\n%s" % (" ".join(args), r.returncode, r.stderr[-2000:]))
    return r.stdout


def shipped(engine):
    """{(name, version): [target labels]} for the crates in the release binary."""
    out = {}
    for triple, label in TARGETS:
        tree = cargo(engine, "tree", "-p", PACKAGE, "--target", triple, "-e", "normal",
                     "--prefix", "none", "--no-dedupe", "--format", "{p}")
        for line in tree.splitlines():
            parts = line.split()
            if len(parts) < 2 or not parts[1].startswith("v"):
                continue
            key = (parts[0], parts[1][1:])
            labels = out.setdefault(key, [])
            if label not in labels:
                labels.append(label)
    return out


def read_text(path):
    with open(path, "rb") as fh:
        raw = fh.read()
    text = raw.decode("utf-8", errors="replace").replace("\r\n", "\n").replace("\r", "\n")
    return "\n".join(line.rstrip() for line in text.split("\n")).strip("\n")


# What a licence text is, by phrases from the licence itself. A crate's SPDX
# expression is then evaluated against the kinds its texts show: "A AND B" needs
# both, "A OR B" one of them. ISC and 0BSD share their grant sentence, MIT and
# MIT-0 theirs; an id with no entry here fails the check until it gets one.
KINDS = (
    ("Apache-2.0", (r"Apache License", r"Version 2\.0")),
    ("MIT", (r"Permission is hereby granted, free of charge",)),
    ("ISC", (r"Permission to use, copy, modify, and(/or)? distribute this software for any",)),
    ("BSD", (r"Redistribution and use in source and binary forms",)),
    ("Zlib", (r"Permission is granted to anyone to use this software for any purpose",
              r"(?i)altered source versions must be plainly marked")),
    ("Unicode", (r"(?i)unicode,? inc|UNICODE LICENSE",)),
    ("Unlicense", (r"This is free and unencumbered software released into the public domain",)),
    ("CC0", (r"(?i)Creative Commons", r"CC0")),
    ("BSL", (r"Boost Software License - Version 1\.0",)),
    ("MPL-2.0", (r"Mozilla Public License,? [Vv]ersion 2\.0",)),
)
SPDX_KIND = {
    "Apache-2.0": "Apache-2.0", "MIT": "MIT", "MIT-0": "MIT", "ISC": "ISC", "0BSD": "ISC",
    "BSD-2-Clause": "BSD", "BSD-3-Clause": "BSD", "Zlib": "Zlib", "Unicode-3.0": "Unicode",
    "Unicode-DFS-2016": "Unicode", "Unlicense": "Unlicense", "CC0-1.0": "CC0",
    "BSL-1.0": "BSL", "MPL-2.0": "MPL-2.0",
}
# MPL-2.0 is file-level copyleft, not a notice licence: section 3.2(a) has whoever
# distributes the Executable Form tell its recipients where to get the Source Code
# Form. The notices therefore name, for every MPL crate, the unmodified .crate on
# crates.io and its sha256 from Cargo.lock.
CRATE_URL = "https://static.crates.io/crates/{name}/{name}-{version}.crate"


def lock_checksums(engine):
    """{(name, version): sha256 of the .crate} from engine/Cargo.lock."""
    with open(os.path.join(engine, "Cargo.lock"), encoding="utf-8") as fh:
        lock = fh.read()
    out = {}
    for block in lock.split("[[package]]"):
        n = re.search(r'^name = "([^"]+)"', block, re.M)
        v = re.search(r'^version = "([^"]+)"', block, re.M)
        c = re.search(r'^checksum = "([0-9a-f]{64})"', block, re.M)
        if n and v and c:
            out[(n.group(1), v.group(1))] = c.group(1)
    return out


def kinds_of(text):
    return {k for k, needles in KINDS if all(re.search(n, text) for n in needles)}


def covered(expr, kinds):
    """True when the SPDX expression is satisfied by texts of these kinds.
    Raises ValueError on an id SPDX_KIND does not know."""
    tokens = re.findall(r"\(|\)|[A-Za-z0-9.+-]+", expr.replace("/", " OR "))
    pos = 0

    def atom():
        nonlocal pos
        tok = tokens[pos]
        pos += 1
        if tok == "(":
            v = disj()
            pos += 1                         # ")"
            return v
        if tok.endswith("+"):
            tok = tok[:-1]
        if tok not in SPDX_KIND:
            raise ValueError("unknown licence id %r in %r" % (tok, expr))
        return SPDX_KIND[tok] in kinds

    def conj():
        nonlocal pos
        v = atom()
        while pos < len(tokens) and tokens[pos] == "AND":
            pos += 1
            v = atom() and v
        return v

    def disj():
        nonlocal pos
        v = conj()
        while pos < len(tokens) and tokens[pos] == "OR":
            pos += 1
            v = conj() or v
        return v

    return disj()


def norm(text):
    return " ".join(text.split())


APACHE_END = "END OF TERMS AND CONDITIONS"


def split_apache(text):
    """(before, terms, appendix) of an Apache-2.0 licence file, or None.

    Apache-2.0 asks for one copy of the licence with the work (4a) and the
    work's NOTICE file (4d). Crates ship the same terms over and over, so the
    notices print each wording of the terms once; what a crate adds before or
    after them stays with that crate (apache_own)."""
    i = text.find("Apache License")
    j = text.find(APACHE_END)
    if i < 0 or j < i or "TERMS AND CONDITIONS FOR USE" not in text[i:j]:
        return None
    j += len(APACHE_END)
    return text[:i].strip("\n"), text[i:j], text[j:].strip("\n")


def apache_own(appendix):
    """What of an Apache APPENDIX is the crate's own. The appendix is the
    licence's template for applying it ("Copyright [yyyy] [name of copyright
    owner]"); when it is only that, the crate's own part is the copyright line
    it filled in, if any. An appendix that carries more is kept whole."""
    n = norm(appendix)
    if n.startswith("APPENDIX: How to apply the Apache License") and \
            n.endswith("limitations under the License."):
        return "\n".join(line.strip() for line in appendix.splitlines()
                         if re.match(r"\s*Copyright\b", line)
                         and not re.search(r"[\[{]yyyy[\]}]", line))
    return appendix


def build(repo):
    """(file text, [(check, ok, detail)], summary)."""
    engine = os.path.join(repo, "engine")
    checks = []
    cargo(engine, "fetch")
    meta = json.loads(cargo(engine, "metadata", "--format-version", "1"))
    packages = {(p["name"], p["version"]): p for p in meta["packages"]}
    crates = shipped(engine)
    checksums = lock_checksums(engine)
    mpl = []                                 # [(name, version, repository, sha256)]

    texts = {}                               # whitespace-normalised text -> index
    order = []                               # [(title, text, users)]
    rows = []
    missing = []
    uncovered = []

    def add(title, text):
        key = norm(text)                     # texts that differ only in spacing are one
        if key not in texts:
            texts[key] = len(order) + 1
            order.append((title, text, []))
        return texts[key]

    def add_licence(title, text):
        parts = split_apache(text)
        if parts is None:
            return [add(title, text)]
        before, terms, appendix = parts
        refs = [add("Apache License 2.0, terms (as in %s)" % title, terms)]
        own = "\n\n".join(x for x in (before, apache_own(appendix)) if x.strip())
        if own:
            refs.append(add("%s -- besides the Apache-2.0 terms" % title, own))
        return refs

    for (name, version) in sorted(crates, key=lambda k: (k[0].lower(), k[1])):
        p = packages.get((name, version))
        if p is None:
            missing.append("%s %s (not in cargo metadata)" % (name, version))
            continue
        if p["source"] is None:
            continue                         # a path crate: crow-nest's own code
        root = os.path.dirname(p["manifest_path"])
        files = sorted(f for f in os.listdir(root)
                       if LICENCE_FILE.match(f) and os.path.isfile(os.path.join(root, f)))
        if p.get("license_file") and p["license_file"] not in files:
            files.append(p["license_file"])
        extra = EXTRA_FILES.get(name, [])
        for f in extra:
            if os.path.isfile(os.path.join(root, f)):
                files.append(f)
            else:
                missing.append("%s %s (EXTRA_FILES %s not in the crate)" % (name, version, f))
        found = [("%s %s: %s" % (name, version, f.replace("\\", "/")), read_text(os.path.join(root, f)))
                 for f in files]
        if not found:
            missing.append("%s %s (%s)" % (name, version, p.get("license")))
        refs = []
        for title, text in found:
            refs += [r for r in add_licence(title, text) if r not in refs]
        kinds = set().union(*(kinds_of(text) for _, text in found))
        try:
            if not covered(p.get("license") or "", kinds):
                uncovered.append("%s %s: %s, texts show %s"
                                 % (name, version, p.get("license"), sorted(kinds) or "nothing"))
        except (ValueError, IndexError) as exc:
            uncovered.append("%s %s: %s" % (name, version, exc))
        if "MPL" in (p.get("license") or ""):
            mpl.append((name, version, p.get("repository") or "-", checksums.get((name, version))))
        for r in refs:
            order[r - 1][2].append("%s %s" % (name, version))
        rows.append((name, version, p.get("license") or "?", crates[(name, version)], refs))
    checks.append(("every compiled-in crate has a licence text", not missing, "; ".join(missing)))
    checks.append(("every crate's licence expression is covered by its texts", not uncovered,
                   "; ".join(uncovered)))
    checks.append(("every MPL-2.0 crate has a checksum in Cargo.lock for its source line",
                   all(m[3] for m in mpl), ", ".join("%s %s" % m[:2] for m in mpl if not m[3])))

    lines = [
        "crow-nest engine (serve) -- third-party notices",
        "",
        "Generated by tools/engine_notices.py from engine/Cargo.lock. Do not edit;",
        "run `python tools/engine_notices.py --write` after a dependency change.",
        "",
        "crow-nest's own code is Apache-2.0 (LICENSE). The release binary `serve`",
        "compiles in the %d crates below (Windows and Linux builds together," % len(rows),
        "procedural macros included); onig_sys also compiles in the Oniguruma C",
        "library, whose own licence is listed with it. Each keeps its own terms.",
        "Where a crate offers a choice (\"A OR B\"), crow-nest takes it under any one",
        "of them; every text the crate ships is reproduced.",
        "",
        "The package holds no NVIDIA file; NVIDIA's NVRTC, which serve loads at run",
        "time, is not compiled in and is not covered here; see NOTICE.",
        "",
        "=" * 80,
        "COMPONENTS",
        "=" * 80,
        "",
    ]
    for name, version, lic, labels, refs in rows:
        targets = "" if len(labels) == len(TARGETS) else "  [%s only]" % "/".join(labels)
        lines.append("%s %s -- %s%s" % (name, version, lic, targets))
        lines.append("    text %s" % ", ".join("[%d]" % r for r in refs))
    if mpl:
        lines += ["", "=" * 80, "SOURCE CODE OF THE MPL-2.0 COMPONENTS", "=" * 80, "",
                  "These crates are under the Mozilla Public License 2.0 and are compiled in",
                  "unmodified. Their Source Code Form is the published crate, downloadable at",
                  "no charge from crates.io; the sha256 is the one Cargo.lock pins.", ""]
        for name, version, repo_url, digest in mpl:
            lines += ["%s %s" % (name, version),
                      "    %s" % CRATE_URL.format(name=name, version=version),
                      "    sha256 %s" % digest,
                      "    repository %s" % repo_url]
    lines += ["", "=" * 80, "LICENCE TEXTS", "=" * 80]
    for i, (title, text, users) in enumerate(order, 1):
        lines += ["", "-" * 80, "[%d] %s" % (i, title)]
        if len(users) > 1:
            lines.append("    also used by: %s" % ", ".join(u for u in users if not title.startswith(u + ":")))
        lines += ["-" * 80, "", text]
    body = "\n".join(lines) + "\n"
    summary = "%d crates, %d distinct texts, %d MPL-2.0" % (len(rows), len(order), len(mpl))
    return body, checks, summary


def read_tool(repo, name):
    with open(os.path.join(repo, "tools", name), encoding="utf-8") as fh:
        return fh.read()


def staged_nvrtc_names(repo):
    """The NVRTC file names the pack scripts still put into the package: the old
    $NVRTC_DLLS / NVRTC_LIBS lists. There must be none."""
    names = []
    m = re.search(r"^\$NVRTC_DLLS\s*=\s*@\(([^)]*)\)", read_tool(repo, "pack-engine.ps1"), re.M)
    if m:
        names += re.findall(r"'([^']+)'", m.group(1))
    m = re.search(r"^NVRTC_LIBS=\(([^)]*)\)", read_tool(repo, "pack-engine.sh"), re.M)
    if m:
        names += [e.split(":", 1)[1] for e in m.group(1).split() if ":" in e]
    return names


def fetched_nvrtc_names(repo):
    """The file names tools/fetch-nvrtc.ps1 / .sh place beside serve."""
    names = []
    m = re.search(r"^\$NVRTC_MEMBERS\s*=\s*\[ordered\]@\{(.*?)^\}", read_tool(repo, "fetch-nvrtc.ps1"), re.M | re.S)
    if m:
        names += re.findall(r"=\s*'([^']+)'", m.group(1))
    m = re.search(r"^MEMBERS='([^']*)'", read_tool(repo, "fetch-nvrtc.sh"), re.M)
    if m:
        names += [e.split(":", 1)[1] for e in m.group(1).split() if ":" in e]
    return names


def main(argv):
    args = [a for a in argv[1:] if a != "--write"]
    write = "--write" in argv[1:]
    repo = os.path.abspath(args[0]) if args else os.path.dirname(HERE)
    if not os.path.isfile(os.path.join(repo, "engine", "Cargo.lock")):
        print("SETUP ERROR: no engine/Cargo.lock under %s" % repo)
        return 2
    try:
        body, checks, summary = build(repo)
        staged = staged_nvrtc_names(repo)
        fetched = fetched_nvrtc_names(repo)
    except (OSError, ValueError, RuntimeError) as exc:
        print("SETUP ERROR: %s" % exc)
        return 2
    out = os.path.join(repo, OUT)
    if write:
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(body)
        print("wrote %s (%d bytes)" % (OUT, len(body.encode("utf-8"))))
    print(summary)
    try:
        with open(out, encoding="utf-8", newline="") as fh:
            have = fh.read()
    except OSError:
        have = None
    checks.append(("%s is what engine/Cargo.lock produces" % OUT, have == body,
                   "run: python tools/engine_notices.py --write"))
    try:
        with open(os.path.join(repo, "NOTICE"), encoding="utf-8") as fh:
            notice = fh.read()
    except OSError as exc:
        notice = ""
        checks.append(("NOTICE readable", False, str(exc)))
    checks.append(("NOTICE points at %s" % OUT, OUT in notice, "NOTICE"))
    checks.append(("the pack scripts stage no NVRTC file",
                   not staged, "tools/pack-engine.ps1 $NVRTC_DLLS / tools/pack-engine.sh NVRTC_LIBS still list: %s" % staged))
    checks.append(("the fetch scripts' NVRTC file names were found", len(fetched) >= 4,
                   "tools/fetch-nvrtc.ps1 $NVRTC_MEMBERS / tools/fetch-nvrtc.sh MEMBERS: %s" % fetched))
    absent = [n for n in fetched if n not in notice]
    checks.append(("NOTICE names every NVRTC file the fetch scripts place", not absent,
                   "missing from NOTICE: %s" % ", ".join(absent)))
    checks.append(("NOTICE says crow-nest distributes no NVIDIA file",
                   "does not distribute any NVIDIA file" in notice, "NOTICE"))
    failed = 0
    for name, ok, detail in checks:
        if ok:
            print("  OK       %s" % name)
        else:
            failed += 1
            print("  FAILED   %s\n             %s" % (name, detail))
    print()
    print("RESULT: %s" % ("PASS" if not failed else "%d FAILED" % failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
