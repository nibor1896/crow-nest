#!/usr/bin/env python3
"""#174: unit tests of tools/regression_gate.py - no GPU, no model, no cargo.

  python -I tools/test_regression_gate.py

Fixtures stand in for what the real runs produce: f32 logits dumps [rows][vocab] (vocab 8 here
instead of 248,320), libtest summary lines, `decode mtpspec` stdout, R3 run lists and whole
before/after records. What has to hold: one differing logit byte is RED and named by row and
column; an A/B delta outside the A/A window is RED in both directions; the libtest counts are
summed per binary and a FAILED binary, a build without any summary line and fewer passes after the
change are RED; the MTP C2 lines and draft counters must match, the timing in those lines must not.
StubEngineTests run record (`before`) -> check (`after` + `compare`) end to end with a stub engine.
"""

import contextlib
import datetime
import importlib.util
import io
import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("regression_gate", TOOLS / "regression_gate.py")
rg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rg)

V = 8


def write_logits(path, rows):
    with open(path, "wb") as f:
        for row in rows:
            f.write(struct.pack(f"<{len(row)}f", *row))
    return rg.file_rec(path)


def base_rows():
    return [[float(r * V + c) / 7.0 for c in range(V)] for r in range(3)]


class LogitsTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_byte_identical_dumps_are_green(self):
        a = write_logits(self.d / "a.f32", base_rows())
        b = write_logits(self.d / "b.f32", base_rows())
        self.assertTrue(rg.compare_logits(a, b, V)["identical"])
        v = rg.compare_r2_item({"logits": a, "ids": [1, 2]}, {"logits": b, "ids": [1, 2]}, V)
        self.assertEqual(v["verdict"], rg.GREEN, v)

    def test_one_differing_logit_byte_is_red_and_named(self):
        a = write_logits(self.d / "a.f32", base_rows())
        raw = bytearray((self.d / "a.f32").read_bytes())
        off = (2 * V + 5) * 4 + 1  # row 2, column 5, the second byte of that f32
        raw[off] ^= 0x01
        (self.d / "b.f32").write_bytes(bytes(raw))
        b = rg.file_rec(self.d / "b.f32")
        res = rg.compare_logits(a, b, V)
        self.assertFalse(res["identical"])
        self.assertEqual((res["offset"], res["row"], res["col"]), (off, 2, 5))
        self.assertNotEqual(res["before_f32"], res["after_f32"])
        v = rg.compare_r2_item({"logits": a, "ids": [1, 2]}, {"logits": b, "ids": [1, 2]}, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("row 2, column 5", v["causes"][0])

    def test_sha_mismatch_without_raw_dumps_is_still_red(self):
        a = {"sha256": "a" * 64, "bytes": 96, "path": str(self.d / "gone-a")}
        b = {"sha256": "b" * 64, "bytes": 96, "path": str(self.d / "gone-b")}
        res = rg.compare_logits(a, b, V)
        self.assertFalse(res["identical"])
        self.assertIn("sha256", res["cause"])

    def test_a_shorter_dump_is_red(self):
        a = write_logits(self.d / "a.f32", base_rows())
        b = write_logits(self.d / "b.f32", base_rows()[:2])
        res = rg.compare_logits(a, b, V)
        self.assertFalse(res["identical"])
        self.assertEqual(res["offset"], 2 * V * 4)

    def test_first_diff_crosses_chunk_boundaries(self):
        a = write_logits(self.d / "a.f32", base_rows())
        raw = bytearray((self.d / "a.f32").read_bytes())
        raw[50] ^= 0x80
        (self.d / "b.f32").write_bytes(bytes(raw))
        self.assertEqual(rg.first_diff_bytes(self.d / "a.f32", self.d / "b.f32", chunk=7), 50)
        self.assertIsNone(rg.first_diff_bytes(self.d / "a.f32", self.d / "a.f32", chunk=7))


class IdsTests(unittest.TestCase):
    def test_identical(self):
        self.assertTrue(rg.compare_ids([5, 6, 7], [5, 6, 7])["identical"])

    def test_first_difference_is_named(self):
        r = rg.compare_ids([5, 6, 7, 8], [5, 6, 7, 9])
        self.assertFalse(r["identical"])
        self.assertEqual(r["index"], 3)

    def test_length_difference_is_red(self):
        self.assertFalse(rg.compare_ids([5, 6], [5, 6, 7])["identical"])

    def test_run_item_ids_red(self):
        v = rg.compare_r2_item({"ids": [1, 2, 3]}, {"ids": [1, 2, 4]}, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("index 2", v["causes"][0])

    def test_an_error_on_one_side_is_red(self):
        v = rg.compare_r2_item({"ids": [1]}, {"error": "decode exit 101"}, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("decode exit 101", v["causes"][0])


def runs(b_ab, n_ab, aa, sha="x"):
    out = []
    for i, (b, n) in enumerate(zip(b_ab, n_ab), 1):
        out += [{"block": "ab", "pair": i, "arm": "B", "ms": b, "ids_sha": sha},
                {"block": "ab", "pair": i, "arm": "N", "ms": n, "ids_sha": sha}]
    for i in range(0, len(aa), 2):
        out += [{"block": "aa", "pair": i // 2 + 1, "arm": "B", "ms": aa[i], "ids_sha": sha},
                {"block": "aa", "pair": i // 2 + 1, "arm": "B2", "ms": aa[i + 1], "ids_sha": sha}]
    return out


class R3Tests(unittest.TestCase):
    AA = [22.10, 22.15, 22.12, 22.18, 22.11, 22.14]  # window 0.08 ms

    def test_delta_inside_the_aa_window_is_green(self):
        v = rg.r3_verdict(runs([22.12, 22.13, 22.14], [22.15, 22.16, 22.17], self.AA))
        self.assertEqual(v["verdict"], rg.GREEN, v)
        self.assertAlmostEqual(v["aa_window_ms"], 0.08, places=6)
        self.assertAlmostEqual(v["delta_ms"], 0.03, places=6)

    def test_slower_outside_the_window_is_red(self):
        v = rg.r3_verdict(runs([22.12, 22.13, 22.14], [22.40, 22.41, 22.42], self.AA))
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("slower", v["causes"][0])

    def test_faster_outside_the_window_is_red_too(self):
        v = rg.r3_verdict(runs([22.12, 22.13, 22.14], [21.80, 21.81, 21.82], self.AA))
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("faster", v["causes"][0])

    def test_ids_differing_between_timed_runs_is_red(self):
        r = runs([22.12] * 3, [22.13] * 3, self.AA)
        r[1]["ids_sha"] = "y"
        v = rg.r3_verdict(r)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("ids differ", v["causes"][0])

    def test_a_failed_run_is_red(self):
        r = runs([22.12] * 3, [22.13] * 3, self.AA)
        r[0]["error"] = "decode exit 1"
        self.assertEqual(rg.r3_verdict(r)["verdict"], rg.RED)

    def test_schedule_is_warmup_then_interleaved_pairs(self):
        s = rg.schedule(3)
        self.assertEqual(s[0], ("warmup", 0, "B"))
        self.assertEqual(len(s), 1 + 4 * 3)
        self.assertEqual(s[1:5], [("ab", 1, "B"), ("ab", 1, "N"), ("aa", 1, "B"), ("aa", 1, "B2")])
        self.assertEqual(sum(1 for x in s if x[2] == "N"), 3)


CARGO_OK = """
   Compiling crow-nest-engine v0.1.0
running 300 tests
test result: ok. 300 passed; 0 failed; 2 ignored; 0 measured; 0 filtered out; finished in 4.12s

running 150 tests
test result: ok. 150 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 1.00s

running 12 tests
test result: ok. 12 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s
"""
CARGO_FAILED = """
running 300 tests
test gen::tests::x ... FAILED
test result: FAILED. 299 passed; 1 failed; 2 ignored; 0 measured; 0 filtered out; finished in 4.12s
"""
CARGO_BUILD_ERROR = """
error[E0425]: cannot find value `x` in this scope
error: could not compile `crow-nest-engine` (lib test) due to 1 previous error
"""


class CargoTests(unittest.TestCase):
    def test_counts_are_summed_per_binary(self):
        r = rg.parse_cargo_test(CARGO_OK, 0)
        self.assertEqual((r["passed"], r["failed"], r["ignored"], r["binaries"]), (462, 0, 2, 3))

    def test_equal_counts_are_green(self):
        a = rg.parse_cargo_test(CARGO_OK, 0)
        self.assertEqual(rg.compare_r1(a, a)["verdict"], rg.GREEN)

    def test_more_tests_after_is_green(self):
        a = rg.parse_cargo_test(CARGO_OK, 0)
        b = rg.parse_cargo_test(CARGO_OK.replace("12 passed", "14 passed"), 0)
        self.assertEqual(rg.compare_r1(a, b)["verdict"], rg.GREEN)

    def test_a_failed_binary_is_red(self):
        a = rg.parse_cargo_test(CARGO_OK, 0)
        b = rg.parse_cargo_test(CARGO_FAILED, 101)
        self.assertEqual((b["passed"], b["failed"], b["failed_binaries"]), (299, 1, 1))
        v = rg.compare_r1(a, b)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertTrue(any("1 failed" in c for c in v["causes"]))

    def test_a_build_without_any_summary_is_red(self):
        a = rg.parse_cargo_test(CARGO_OK, 0)
        b = rg.parse_cargo_test(CARGO_BUILD_ERROR, 101)
        self.assertEqual(b["binaries"], 0)
        v = rg.compare_r1(a, b)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertTrue(any("no libtest summary" in c for c in v["causes"]))

    def test_fewer_passes_after_is_red(self):
        a = rg.parse_cargo_test(CARGO_OK, 0)
        b = rg.parse_cargo_test(CARGO_OK.replace("12 passed", "10 passed"), 0)
        v = rg.compare_r1(a, b)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertIn("462 -> 460", v["causes"][0])


def mtp_stdout(c2=("true", "true", "true"), passes=160, rate="120.3", trace=(13, 248046, 198)):
    return f"""[load] MTP head loaded
mtpspec: 8 prompt tokens, 512 generated, k 3
mtpspec: C2 greedy ids identical: {c2[0]}
mtpspec: passes {passes}, tokens per pass 3.194, acceptance per chain position [0.850 (136/160), 0.700 (95/136), 0.600 (57/95)]
mtpspec: plain 72.{rate[-1]} tok/s, spec (row-by-row verify, no speed-up by design) 30.1 tok/s
mtpspec-batched: C2 greedy ids identical: {c2[1]}
mtpspec-batched: passes {passes}, tokens per pass 3.194, acceptance [0.850 (136/160), 0.700 (95/136), 0.600 (57/95)], {rate} tok/s (plain 72.3)
mtpspec-step: C2 ids identical over 528 tokens (spec_step x 512, spec_finish, decode_step x 16): {c2[2]}
mtpspec-step: passes {passes}, tokens per pass 3.194, k histogram [0, 2, 5, 153], {rate} tok/s (plain 72.3)
trace-plain: {list(trace)}
"""


class MtpTests(unittest.TestCase):
    def test_parse(self):
        m = rg.parse_mtpspec(mtp_stdout())
        self.assertEqual(m["ids"], [13, 248046, 198])
        self.assertEqual(m["c2"], {"mtpspec": True, "mtpspec-batched": True, "mtpspec-step": True})
        self.assertEqual(m["step_tok_s"], 120.3)
        self.assertEqual(m["plain_tok_s"], 72.3)
        self.assertNotIn("tok/s", m["draft"]["mtpspec-step"])

    def test_same_drafts_different_timing_is_green(self):
        a = {"mtp": rg.parse_mtpspec(mtp_stdout(rate="120.3"))}
        b = {"mtp": rg.parse_mtpspec(mtp_stdout(rate="119.8"))}
        a["ids"], b["ids"] = a["mtp"]["ids"], b["mtp"]["ids"]
        v = rg.compare_r2_item(a, b, V)
        self.assertEqual(v["verdict"], rg.GREEN, v)

    def test_a_false_c2_line_is_red(self):
        a = {"mtp": rg.parse_mtpspec(mtp_stdout())}
        b = {"mtp": rg.parse_mtpspec(mtp_stdout(c2=("true", "false (first difference at token 7: plain 1 batched 2)", "true")))}
        v = rg.compare_r2_item(a, b, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertTrue(any("mtpspec-batched C2 line false" in c for c in v["causes"]))

    def test_moved_draft_counters_are_red(self):
        a = {"mtp": rg.parse_mtpspec(mtp_stdout(passes=160))}
        b = {"mtp": rg.parse_mtpspec(mtp_stdout(passes=161))}
        v = rg.compare_r2_item(a, b, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertTrue(any("draft counters differ" in c for c in v["causes"]))

    def test_differing_plain_trace_is_red(self):
        a = {"mtp": rg.parse_mtpspec(mtp_stdout())}
        b = {"mtp": rg.parse_mtpspec(mtp_stdout(trace=(13, 248046, 199)))}
        v = rg.compare_r2_item(a, b, V)
        self.assertEqual(v["verdict"], rg.RED)
        self.assertTrue(any("index 2" in c for c in v["causes"]))


def side(r1_text=CARGO_OK, ids=(1, 2, 3), r3=None):
    rec = {"git": {"head": "e6f901a0000", "dirty": False}, "models": ["flash-next"],
           "r1": {c: rg.parse_cargo_test(r1_text, 0) for c in rg.CRATES},
           "r2": {"flash-next": {"greedy512": {"ids": list(ids)}}}}
    if r3 is not None:
        rec["r3"] = {"flash-next": {"run512": {"runs": r3}}}
    return rec


class SessionTests(unittest.TestCase):
    GOOD_R3 = runs([22.12] * 3, [22.13] * 3, R3Tests.AA)

    def test_all_green_with_r4_pending_exits_3(self):
        res = rg.compare_sessions(side(), side(r3=self.GOOD_R3), V, "R4 verdict: PENDING")
        self.assertEqual((res["overall"], res["exit"]), ("R1-R3 GREEN, R4 PENDING", 3))

    def test_all_green_with_r4_pass_exits_0(self):
        res = rg.compare_sessions(side(), side(r3=self.GOOD_R3), V, "x\nR4 verdict: PASS\n")
        self.assertEqual((res["overall"], res["exit"]), ("ALL GREEN", 0))

    def test_one_red_item_exits_1(self):
        res = rg.compare_sessions(side(), side(ids=(1, 2, 4), r3=self.GOOD_R3), V, "R4 verdict: PASS")
        self.assertEqual((res["overall"], res["exit"]), (rg.RED, 1))
        self.assertTrue(any(l.startswith("RED") and "greedy512" in l for l in rg.report_lines(res)))

    def test_skipped_r3_is_incomplete(self):
        res = rg.compare_sessions(side(), side(), V, "R4 verdict: PASS")
        self.assertEqual((res["overall"], res["exit"]), ("INCOMPLETE", 3))

    def test_cli_compare_writes_json_and_never_overwrites_a_filled_protocol(self):
        with tempfile.TemporaryDirectory() as t:
            s = Path(t) / "s1"
            s.mkdir()
            (s / "before.json").write_text(json.dumps(side()), encoding="utf-8")
            (s / "after.json").write_text(json.dumps(side(r3=self.GOOD_R3)), encoding="utf-8")
            self.assertEqual(rg.main(["compare", "--session", "s1", "--out", t]), 3)
            proto = s / "r4-protocol.md"
            self.assertIn("R4 verdict: PENDING", proto.read_text(encoding="utf-8"))
            self.assertIn("crow_gui.py --base-url http://127.0.0.1:8099/v1", proto.read_text(encoding="utf-8"))
            proto.write_text(proto.read_text(encoding="utf-8").replace("R4 verdict: PENDING", "R4 verdict: PASS"),
                             encoding="utf-8")
            self.assertEqual(rg.main(["compare", "--session", "s1", "--out", t]), 0)
            self.assertIn("R4 verdict: PASS", proto.read_text(encoding="utf-8"))
            self.assertEqual(json.loads((s / "compare.json").read_text(encoding="utf-8"))["overall"], "ALL GREEN")


class EnvTests(unittest.TestCase):
    def test_inherited_crow_vars_are_stripped(self):
        old = os.environ.get("CROW_MTP")
        os.environ["CROW_MTP"] = "0"
        try:
            env = rg.clean_env({"CROW_GRAPH": "1"})
        finally:
            if old is None:
                del os.environ["CROW_MTP"]
            else:
                os.environ["CROW_MTP"] = old
        self.assertNotIn("CROW_MTP", env)
        self.assertEqual(env["CROW_GRAPH"], "1")

    def test_the_27b_runs_with_mtp_on_and_container_paths_resolve(self):
        mc = rg.DEFAULT_CONFIG["models"]["qwen27b"]
        env = rg.model_env(mc, {"env": rg.TF_ENV}, Path("/data"))
        self.assertNotIn("CROW_MTP", env)
        self.assertTrue(os.path.isabs(env["CROW_CNQ"]))
        self.assertEqual(env["CROW_GRAPH"], "0")
        names = [i["name"] for i in mc["r2"]]
        self.assertEqual(names, ["logits512", "logits512-tf", "greedy512", "mtp512", "mtp-logits"])


# A stand-in for engine/target/release/decode: the four modes the gate calls, with the output files
# and stdout lines of engine/src/bin/decode.rs (parity :143-159, run :318-328, mtpspec :618-681,
# mtpgolden :689-696 at be4a250), vocab 8. The values depend on the container name and CROW_GRAPH,
# so the two models and the teacher-forced item differ from each other as the real ones do.
# MUTATE marks a build whose numerics moved: "id" changes greedy id 300 of `run`, "logit" flips one
# byte of the teacher-forced parity dump (row 2, column 5).
STUB_DECODE = r'''
import json, os, struct, sys
MUTATE = None
V = 8
cnq = os.environ.get("CROW_CNQ")
if not cnq or "CROW_MTP" in os.environ:
    sys.exit(3)
salt = len(os.path.basename(cnq)) + (5 if os.environ.get("CROW_GRAPH") == "0" else 0)

def write_f32(path, rows):
    data = bytearray(b"".join(struct.pack("<%df" % V, *r) for r in rows))
    if MUTATE == "logit" and os.path.basename(path) == "gpu-logits.f32" and os.environ.get("CROW_GRAPH") == "0":
        data[(2 * V + 5) * 4 + 1] ^= 0x01
    with open(path, "wb") as f:
        f.write(data)

def logits(rows, extra):
    return [[((r * V + c) * 31 + salt + extra) % 1000 / 7.0 for c in range(V)] for r in range(rows)]

def trace(n):
    return [(i * 13 + salt) % 248320 for i in range(n)]

mode = sys.argv[1]
if mode == "parity":
    ids = json.load(open(sys.argv[2]))
    out = sys.argv[3]
    with open(os.path.join(out, "gen-sequence.json"), "w") as f:
        json.dump({"all_ids": ids + [(i * 7 + salt) % V for i in range(4)], "rows": len(ids), "nan": 0}, f)
    write_f32(os.path.join(out, "gpu-logits.f32"), logits(len(ids), 0))
elif mode == "run":
    t = trace(int(sys.argv[3]))
    if MUTATE == "id":
        t[300] += 1
    os.makedirs("decode_out", exist_ok=True)
    with open("decode_out/run.json", "w") as f:
        json.dump({"warmup_ms": 30.0, "mean_ms": 22.1, "p50_ms": 22.0, "trace": t}, f)
elif mode == "mtpspec":
    n = int(sys.argv[3])
    print("mtpspec: C2 greedy ids identical: true")
    print("mtpspec: passes 200, tokens per pass 2.560, acceptance per chain position [0.910, 0.740, 0.520]")
    print("mtpspec-batched: C2 greedy ids identical: true")
    print("mtpspec-batched: passes 200, tokens per pass 2.560, acceptance [0.910, 0.740, 0.520], 41.3 tok/s (plain 30.2)")
    print("mtpspec-step: C2 ids identical over %d tokens (spec_step x %d, spec_finish, decode_step x 16): true" % (n + 16, n))
    print("mtpspec-step: passes 200, tokens per pass 2.560, k histogram [20, 40, 140], 40.9 tok/s (plain 30.2)")
    print("trace-plain: %s" % json.dumps(trace(n)))
elif mode == "mtpgolden":
    d = sys.argv[2]
    rows = json.load(open(os.path.join(d, "gen-sequence.json")))["rows"]
    write_f32(os.path.join(d, "mtp-gpu-logits.f32"), logits(rows - 1, 3))
else:
    sys.exit(4)
'''


class StubEngineTests(unittest.TestCase):
    """End to end through main(): `before` records a reference, `after` + `compare` check a build
    against it. The real path runs (CLI, binary snapshot, child env, every decode mode, the dumps
    on disk, compare.json); only the engine is the stub above, started through the interpreter."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        self.data = self.d / "data"
        (self.data / "decode_out").mkdir(parents=True)
        (self.data / "decode_out" / "real512-ids.json").write_text(json.dumps([11, 12, 13, 14, 15, 16]))
        (self.data / "decode_out" / "parity-ids.json").write_text(json.dumps([1, 2, 3, 4, 5, 6, 7, 8]))
        self.out = self.d / "out"
        self.config = self.d / "config.json"  # the built-in models and items, vocab 8
        self.config.write_text(json.dumps(dict(rg.DEFAULT_CONFIG, vocab=V)), encoding="utf-8")
        self.saved = (rg.run_logged, rg.engines_alive, rg.gpu_used_mib)
        real_run = rg.run_logged

        def via_interpreter(cmd, cwd, env, log, timeout=None):
            if cmd and str(cmd[0]).endswith(rg.EXE):  # the snapshot of the stub, not a real exe
                cmd = [sys.executable, "-I", *cmd]
            return real_run(cmd, cwd, env, log, timeout)

        rg.run_logged = via_interpreter
        rg.engines_alive = lambda: []  # the host's own engines are not this test's business
        rg.gpu_used_mib = lambda: None

    def tearDown(self):
        rg.run_logged, rg.engines_alive, rg.gpu_used_mib = self.saved
        self.tmp.cleanup()

    def tree(self, name, mutate=None):
        root = self.d / name
        exe = root / "engine" / "target" / "release" / rg.EXE
        exe.parent.mkdir(parents=True)
        exe.write_text(STUB_DECODE.replace("MUTATE = None", f"MUTATE = {mutate!r}"), encoding="utf-8")
        return root

    def gate(self, *argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            rc = rg.main([str(a) for a in argv])
        return rc, buf.getvalue()

    def side(self, cmd, session, root, *extra):
        return self.gate(cmd, "--session", session, "--root", root, "--data-root", self.data,
                         "--out", self.out, "--raw", self.d / "raw", "--config", self.config, "--no-build",
                         "--skip", "r1,r3", "--ram-gate-gib", "0", "--timeout-s", "120", *extra)

    def record(self, root=None):
        rc, _ = self.side("before", "ref", root or self.tree("ref"))
        self.assertEqual(rc, 0)
        return json.loads((self.out / "ref" / "before.json").read_text(encoding="utf-8"))

    def check(self, mutate=None):
        self.side("after", "chk", self.tree("chk", mutate), "--before-session", "ref")
        rc, text = self.gate("compare", "--session", "chk", "--before-session", "ref", "--out", self.out,
                             "--config", self.config)
        res = json.loads((self.out / "chk" / "compare.json").read_text(encoding="utf-8"))
        return rc, text, res

    def test_record_holds_binary_sha_commit_date_and_every_item(self):
        root = self.tree("ref")
        git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t"]
        if subprocess.run(git + ["init", "-q"]).returncode or \
                subprocess.run(git + ["commit", "-q", "--allow-empty", "-m", "ref"]).returncode:
            self.skipTest("git cannot make a fixture commit here")
        head = subprocess.run(git + ["rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
        rec = self.record(root)
        self.assertEqual(rec["git"]["head"], head)
        self.assertTrue(rec["started"].startswith(str(datetime.date.today().year)))
        exe = root / "engine" / "target" / "release" / rg.EXE
        self.assertEqual(rec["binary"]["sha256"], rg.sha256_file(exe))
        r2 = rec["r2"]
        self.assertEqual(list(r2), ["flash-next", "qwen27b"])
        self.assertEqual(list(r2["qwen27b"]), ["logits512", "logits512-tf", "greedy512", "mtp512", "mtp-logits"])
        for model, items in r2.items():
            for name, item in items.items():
                self.assertNotIn("error", item, f"{model}/{name}: {item.get('error')}")
        self.assertEqual(len(r2["flash-next"]["greedy512"]["ids"]), 512)
        self.assertEqual(len(r2["qwen27b"]["logits512"]["logits"]["sha256"]), 64)
        self.assertEqual(r2["qwen27b"]["mtp512"]["mtp"]["c2"], dict.fromkeys(rg.C2_PATHS, True))

    def test_same_build_passes_every_r2_item(self):
        self.record()
        rc, text, res = self.check()
        self.assertEqual(rc, 3, text)  # R1 and R3 skipped: INCOMPLETE, never 0, never 1
        self.assertEqual(res["overall"], "INCOMPLETE")
        items = [v for m in res["r2"].values() for v in m.values()]
        self.assertEqual(len(items), 8)
        self.assertTrue(all(v["verdict"] == rg.GREEN for v in items), text)
        self.assertNotIn("RED", text)

    def test_a_changed_greedy_id_fails_with_its_index(self):
        self.record()
        rc, text, res = self.check(mutate="id")
        self.assertEqual(rc, 1, text)
        for model in ("flash-next", "qwen27b"):
            v = res["r2"][model]["greedy512"]
            self.assertEqual(v["verdict"], rg.RED)
            self.assertIn("first differing id at index 300", v["causes"][0])
            self.assertEqual(res["r2"][model]["logits512"]["verdict"], rg.GREEN)
        self.assertIn("RED      R2 qwen27b/greedy512", text)

    def test_a_changed_logit_byte_fails_with_row_and_column(self):
        self.record()
        rc, text, res = self.check(mutate="logit")
        self.assertEqual(rc, 1, text)
        for model in ("flash-next", "qwen27b"):
            v = res["r2"][model]["logits512-tf"]
            self.assertEqual(v["verdict"], rg.RED)
            self.assertIn(f"first differing byte at offset {(2 * V + 5) * 4 + 1}: row 2, column 5", v["causes"][0])
            self.assertEqual(res["r2"][model]["logits512"]["verdict"], rg.GREEN)
            self.assertEqual(res["r2"][model]["greedy512"]["verdict"], rg.GREEN)

    def test_a_missing_reference_is_refused_by_name(self):
        self.side("after", "chk", self.tree("chk"))
        missing = self.out.resolve() / "nope" / "before.json"
        for argv in (["compare", "--session", "chk", "--before-session", "nope", "--out", self.out],
                     ["after", "--session", "chk2", "--before-session", "nope", "--root", self.d / "chk",
                      "--data-root", self.data, "--out", self.out, "--raw", self.d / "raw", "--no-build",
                      "--skip", "r1,r2", "--ram-gate-gib", "0"]):
            err = io.StringIO()
            with self.assertRaises(SystemExit) as cm, contextlib.redirect_stderr(err), \
                    contextlib.redirect_stdout(io.StringIO()):
                rg.main([str(a) for a in argv])
            self.assertEqual(cm.exception.code, 2, argv[0])
            self.assertIn(str(missing), err.getvalue(), argv[0])
        self.assertFalse((self.out / "chk" / "compare.json").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
