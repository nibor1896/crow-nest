#!/usr/bin/env python3
"""#90: the engine arm script's deep-anchor wiring, and the `subset --row0` it needs.

`tools/oracle_longctx_engine_arm.sh --dry-run` prints, per anchor and arm, the `decode parity`
command it would run and does nothing else: no flock, no engine, no GPU, no output directory.
These tests run it in a throwaway repo layout (a copy of the script, a synthetic row plan, ids
files of the plan's lengths), so they need neither the model nor the real dumps.
`EngineArmRun` runs the real path of the script in the same layout with stubs for everything heavy
(`decode`, the subset tool, flock, pgrep, sleep): the manifest step and what a failed step leaves behind.

Run:  python3 tools/oracle_longctx_test_engine_arm.py       (needs bash; no GPU, no engine)
"""

import array
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
BASH = shutil.which("bash")
BLOCK = 64
# anchor -> ids in its prefix file. 9000 mimics the real anchor 178553: the plan clamps it to
# the last valid row, so its prefix is one id SHORT of the anchor.
PREFIX = {1000: 1001, 2564: 2565, 2565: 2566, 50000: 50001, 9000: 8999}
OLD_ENV_KEYS = ["CROW_CNQ", "CROW_HOTSETS", "CROW_GRAPH", "CROW_MMA", "CROW_PINNED_BUDGET_GB",
                "LD_LIBRARY_PATH"]


def make_repo(root, prefix=PREFIX, plan_edit=None):
    """A repo layout the script can cd into: tools/<script>, row-plan.json, the ids files."""
    od = Path(root) / "decode_out" / "oracle-longctx"
    (Path(root) / "tools").mkdir(parents=True)
    od.mkdir(parents=True)
    # a Windows checkout with core.autocrlf=true holds the script with CRLF; bash cannot run that
    src = (TOOLS / "oracle_longctx_engine_arm.sh").read_bytes().replace(b"\r\n", b"\n")
    script = Path(root) / "tools" / "oracle_longctx_engine_arm.sh"
    script.write_bytes(src)
    groups = []
    for anchor, p in prefix.items():
        groups.append({"name": "d%d" % anchor, "anchor": anchor, "prefix_tokens": p,
                       "rows": list(range(p - BLOCK, p))})
        (od / ("longctx-170k-a%d-ids.json" % anchor)).write_text(json.dumps([0] * p))
    if plan_edit:
        plan_edit(groups)
    (od / "row-plan.json").write_text(json.dumps({"groups": groups}))
    return script


@unittest.skipUnless(BASH, "needs bash")
class EngineArmDryRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # the script calls python3; point it at the interpreter running this test
        cls.shim = tempfile.mkdtemp(prefix="arm90-shim-")
        py3 = Path(cls.shim) / "python3"
        py3.write_text('#!/bin/sh\nexec "%s" "$@"\n' % sys.executable.replace("\\", "/"))
        py3.chmod(py3.stat().st_mode | stat.S_IXUSR)

    def run_script(self, *args, plan_edit=None):
        root = tempfile.mkdtemp(prefix="arm90-repo-")
        script = make_repo(root, plan_edit=plan_edit)
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"])
        p = subprocess.run([BASH, str(script), *args], cwd=root, env=env, timeout=60,
                           capture_output=True, text=True)
        self.root = root
        return p

    @staticmethod
    def commands(stdout):
        """{(anchor, arm): argv} from the `dry-run anchor A arm X ...` header + `  env ...` pairs."""
        lines, found = stdout.splitlines(), {}
        for i, line in enumerate(lines):
            m = re.match(r"dry-run anchor (\d+) arm (\S+)", line)
            if m:
                found[(int(m.group(1)), m.group(2))] = shlex.split(lines[i + 1])
        return found

    def test_anchors_up_to_2564_keep_their_command(self):
        p = self.run_script("--dry-run", "--anchors", "1000 2564", "--arms", "none kvbf16")
        self.assertEqual(p.returncode, 0, p.stderr)
        found = self.commands(p.stdout)
        self.assertEqual(sorted(found), [(a, arm) for a in (1000, 2564) for arm in ("kvbf16", "none")])
        for (a, arm), argv in found.items():
            env = [t for t in argv if re.match(r"[A-Z_]+=", t)]
            self.assertEqual([e.split("=")[0] for e in env],
                             OLD_ENV_KEYS + (["CROW_KV"] if arm == "kvbf16" else []))
            vals = dict(e.split("=", 1) for e in env)
            self.assertEqual((vals["CROW_GRAPH"], vals["CROW_MMA"], vals["CROW_PINNED_BUDGET_GB"]),
                             ("1", "1", "50"))      # the first rung of the 50 36 24 16 ladder
            self.assertTrue(vals["CROW_CNQ"].endswith("/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq"))
            self.assertTrue(vals["CROW_HOTSETS"].endswith("/decode_out/hotsets-M-longctx2100-n160.json"))
            self.assertEqual(argv[0], "env")
            self.assertEqual(argv[len(env) + 1:],
                             ["engine/target/release/decode", "parity",
                              "decode_out/oracle-longctx/longctx-170k-a%d-ids.json" % a,
                              "decode_out/oracle-longctx/engine/a%d/%s" % (a, arm)])
            if arm == "kvbf16":
                self.assertEqual(vals["CROW_KV"], "bf16")

    def test_anchors_above_2564_run_in_the_tail_form(self):
        p = self.run_script("--dry-run", "--anchors", "2564 2565 50000 9000", "--arms", "none kvbf16")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertNotIn("REFUSED", p.stdout + p.stderr)
        found = self.commands(p.stdout)
        self.assertEqual(len(found), 8)
        for (a, arm), argv in found.items():
            tails = [t for t in argv if t.startswith("CROW_PARITY_TAIL=")]
            if a == 2564:                                  # the boundary: still every row
                self.assertEqual(tails, [])
                continue
            self.assertEqual(tails, ["CROW_PARITY_TAIL=%d" % BLOCK])   # the plan block, all of it
            self.assertEqual(argv[-2:], ["decode_out/oracle-longctx/longctx-170k-a%d-ids.json" % a,
                                         "decode_out/oracle-longctx/engine/a%d/%s" % (a, arm)])
            # and nothing else differs from the dense form of the same arm
            dense = found[(2564, arm)]
            self.assertEqual([t for t in argv[:-2] if t not in tails], dense[:-2])

    def test_a_dry_run_writes_nothing(self):
        p = self.run_script("--dry-run", "--anchors", "1000 50000")
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertFalse((Path(self.root) / "decode_out" / "oracle-longctx" / "engine").exists())

    def test_a_deep_anchor_that_is_not_the_prefix_tail_is_refused(self):
        def shifted(groups):                    # rows one short of the prefix's last row
            g = [g for g in groups if g["anchor"] == 50000][0]
            g["rows"] = [r - 1 for r in g["rows"]]
        p = self.run_script("--dry-run", "--anchors", "50000", plan_edit=shifted)
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("50000", p.stderr)
        self.assertNotIn("CROW_PARITY_TAIL", p.stdout)

    def test_a_deep_anchor_missing_from_the_plan_is_refused(self):
        p = self.run_script("--dry-run", "--anchors", "3000")
        self.assertEqual(p.returncode, 2, p.stdout + p.stderr)
        self.assertIn("3000", p.stderr)

    def test_no_anchor_at_all_is_still_an_error(self):
        # characterization: the same before and after the tail wiring
        p = self.run_script("--anchors", "")
        self.assertEqual(p.returncode, 2)
        self.assertIn("no runnable anchor left", p.stderr)


# the subset tool of the throwaway repo: the real one needs 248,320-wide rows; this one writes the
# files the script looks for, or dies half way when ARM_STUB_SUBSET=fail
STUB_SUBSET = '''import os, sys
out = sys.argv[sys.argv.index("--out") + 1]
if os.environ.get("ARM_STUB_SUBSET") == "fail":
    open(out, "wb").write(b"partial")
    sys.exit("stub subset: failing on purpose")
open(out, "wb").write(b"stub-rows")
open(out + ".rows.json", "w").write("{}")
'''
STUB_DECODE = '#!/bin/sh\nmkdir -p "$3"\nprintf stub-dump > "$3/gpu-logits.f32"\necho "decode/parity: stub"\n'


@unittest.skipUnless(BASH, "needs bash")
class EngineArmRun(unittest.TestCase):
    """The real run path (no --dry-run) with every heavy part stubbed: `decode` writes a tiny dump,
    flock and pgrep do nothing (the real flock would take /tmp/crow-gpu.lock), `sleep` records the
    10-minute backoff and ends the script (a run that cannot finish would retry for 8 hours), and
    the subset tool is the stub above. No GPU, no engine, no model."""

    @classmethod
    def setUpClass(cls):
        cls.shim = tempfile.mkdtemp(prefix="arm90-shim-")
        for name, body in (("python3", 'exec "%s" "$@"' % sys.executable.replace("\\", "/")),
                           ("flock", "exit 0"), ("pgrep", "exit 1"),
                           ("sleep", 'echo "$@" >> "$ARM_MARK"\nkill "$PPID"')):
            f = Path(cls.shim) / name
            f.write_text("#!/bin/sh\n%s\n" % body)
            f.chmod(f.stat().st_mode | stat.S_IXUSR)

    def run_script(self, *args, subset_fails=False, before=None):
        self.root = Path(tempfile.mkdtemp(prefix="arm90-run-"))
        script = make_repo(self.root)
        rel = self.root / "engine" / "target" / "release"
        rel.mkdir(parents=True)
        (rel / "decode").write_text(STUB_DECODE)
        (rel / "decode").chmod(0o755)
        (self.root / "tools" / "oracle_longctx_rows.py").write_text(STUB_SUBSET)
        self.mark = self.root / "sleeps.txt"
        if before:
            before(self.root)
        env = dict(os.environ, PATH=self.shim + os.pathsep + os.environ["PATH"], CROW_LOCK="0",
                   ARM_MARK=self.mark.as_posix(), ARM_STUB_SUBSET="fail" if subset_fails else "")
        return subprocess.run([BASH, str(script), *args], cwd=self.root, env=env, timeout=60,
                              capture_output=True, text=True)

    def run_dir(self, anchor, arm="none"):
        return self.root / "decode_out" / "oracle-longctx" / "engine" / ("a%d" % anchor) / arm

    def test_the_manifest_is_written_when_two_or_more_runs_finish(self):
        p = self.run_script("--anchors", "1000 2564", "--arms", "none kvbf16")
        self.assertEqual(p.returncode, 0, p.stdout + p.stderr)
        self.assertIn("engine arm complete", p.stdout)
        doc = json.loads((self.root / "decode_out" / "oracle-longctx" / "engine"
                          / "manifest.json").read_text())
        self.assertEqual([(r["anchor"], r["arm"]) for r in doc["runs"]],
                         [("a1000", "kvbf16"), ("a1000", "none"), ("a2564", "kvbf16"), ("a2564", "none")])
        self.assertTrue(all(r["done"] for r in doc["runs"]))

    def test_a_failed_trim_keeps_the_dump_and_does_not_claim_success(self):
        p = self.run_script("--anchors", "1000", "--arms", "none", subset_fails=True)
        d = self.run_dir(1000)
        self.assertIn("trim FAILED", p.stderr)
        self.assertNotIn("plan rows written", p.stdout)
        self.assertTrue((d / "gpu-logits.f32").exists(), "the dump was deleted after a failed trim")
        self.assertFalse((d / "plan-rows.f32").exists(), "a partial trim reads as the done marker")
        self.assertFalse((d / "SHA256SUMS").exists())
        self.assertTrue(self.mark.exists(), "the run counted as done instead of pending")

    def test_a_failed_hash_keeps_the_dump_and_does_not_claim_success(self):
        p = self.run_script("--anchors", "1000", "--arms", "none",
                            before=lambda root: (root / "decode_out" / "oracle-longctx"
                                                 / "longctx-170k-a1000-ids.json").unlink())
        d = self.run_dir(1000)
        self.assertIn("hashing FAILED", p.stderr)
        self.assertNotIn("plan rows written", p.stdout)
        self.assertTrue((d / "gpu-logits.f32").exists())
        self.assertFalse((d / "plan-rows.f32").exists())
        self.assertTrue(self.mark.exists())

    def test_an_anchor_without_a_plan_group_stops_before_the_trim(self):
        p = self.run_script("--anchors", "1500", "--arms", "none")   # 1500 is not in the plan
        d = self.run_dir(1500)
        self.assertIn("no row-plan group", p.stderr)
        self.assertNotIn("plan rows written", p.stdout)
        self.assertTrue((d / "gpu-logits.f32").exists())
        self.assertFalse((d / "plan-rows.f32").exists())


class SubsetRow0(unittest.TestCase):
    """`subset --row0`: carving the plan rows out of a CROW_PARITY_TAIL dump, which starts at
    absolute row `row0_pos`, not at row 0."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("rows_tool", TOOLS / "oracle_longctx_rows.py")
        cls.rt = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.rt)     # argparse only fires in __main__

    VOCAB, N, ROW0 = 4, 6, 100              # a tail dump of absolute rows 100..105

    def dump(self, d):
        rows = [[float(r * 10 + i) for i in range(self.VOCAB)] for r in range(self.N)]
        a = array.array("f")
        for row in rows:
            a.extend(row)
        path = os.path.join(d, "tail.f32")
        with open(path, "wb") as fh:
            a.tofile(fh)
        return path, rows

    def subset(self, d, dump, keep):
        keep_file = os.path.join(d, "keep.json")
        with open(keep_file, "w", encoding="utf-8") as fh:
            json.dump(keep, fh)
        return self.rt.main(["subset", "--source", dump, "--out", os.path.join(d, "sp.f32"),
                             "--rows-file", keep_file, "--vocab", str(self.VOCAB),
                             "--row0", str(self.ROW0)])

    def test_rows_are_carved_by_absolute_id(self):
        d = tempfile.mkdtemp(prefix="arm90-subset-")
        dump, rows = self.dump(d)
        self.assertEqual(self.subset(d, dump, [101, 103, 104]), 0)
        with open(os.path.join(d, "sp.f32.rows.json"), encoding="utf-8") as fh:
            self.assertEqual(json.load(fh)["rows"], [101, 103, 104])   # absolute, as the reader wants
        got = array.array("f")
        with open(os.path.join(d, "sp.f32"), "rb") as fh:
            got.fromfile(fh, 3 * self.VOCAB)
        self.assertEqual(got.tolist(), rows[1] + rows[3] + rows[4])

    def test_a_row_outside_the_dump_is_refused_by_name(self):
        d = tempfile.mkdtemp(prefix="arm90-subset-")
        dump, _ = self.dump(d)
        with self.assertRaises(SystemExit) as before:
            self.subset(d, dump, [99, 101])
        self.assertIn("before row0", str(before.exception))
        with self.assertRaises(SystemExit) as beyond:
            self.subset(d, dump, [101, 106])
        self.assertIn("beyond", str(beyond.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
