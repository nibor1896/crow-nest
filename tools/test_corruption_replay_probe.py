#!/usr/bin/env python3
"""#91: the unit tests of the replay probe's grader.

  python tools/test_corruption_replay_probe.py      # no server, no GPU, no Crow import

Every check of `tools/corruption-replay-probe.py` is a pure function of the tool call, the
declared tools and the context -- except `digit_near_miss`, which asks the filesystem, so
its cases build their own directory. The three live corruptions of 2026-09-22 (session
messages [2], [26], [69]) are pinned here in their stored shape.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("replay_probe", TOOLS / "corruption-replay-probe.py")
rp = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rp)


def _fn(name, props, required):
    return {"type": "function", "function": {"name": name, "description": "",
            "parameters": {"type": "object", "properties": props, "required": required}}}


S = {"type": "string"}
DECL = [
    _fn("run_command", {"command": S, "cwd": S}, ["command"]),
    _fn("write_file", {"path": S, "content": S}, ["path", "content"]),
    _fn("edit_file", {"path": S, "old": S, "new": S}, ["path", "old", "new"]),
    _fn("delegate", {"task": S, "context": S}, ["task"]),
    _fn("read_file", {"path": S, "start_line": {"type": "integer"}}, ["path"]),
]
HOME = "/home/nibor1896"


def call(name, **args):
    return {"name": name, "arguments": json.dumps(args)}


def kinds(graded):
    return sorted({e["kind"] for e in graded["errors"]})


class LiveShapes(unittest.TestCase):
    """The stored answers of 2026-09-22, graded as the probe grades them."""

    ctx = rp.context_index([{"role": "user", "content":
                             "root /home/nibor1896/Projects/localconf/testcases/diorama-test"}])

    def test_the_11896_cwd_is_a_home_mismatch_and_the_clean_one_is_not(self):
        bad = rp.grade_call(call("run_command", command="ls", cwd="/home/nibor11896/three-staging"),
                            DECL, self.ctx, HOME)
        good = rp.grade_call(call("run_command", command="ls",
                                  cwd="/home/nibor1896/Projects/localconf/testcases/diorama-test"),
                             DECL, self.ctx, HOME)
        self.assertIn("home_mismatch", kinds(bad))
        self.assertEqual([e["edit"] for e in bad["errors"] if e["kind"] == "home_mismatch"], ["insertion"])
        self.assertEqual(good["errors"], [])

    def test_a_newline_in_a_path_is_a_control_char(self):
        g = rp.grade_call(call("write_file", path="/\n/home/nibor11896/x/water.js", content="//"),
                          DECL, self.ctx, HOME)
        # digit_near_miss too: this context spells nibor1896 only (the live [69] context also
        # carried the corrupt nibor11896 from [2]/[3], so there it is home_mismatch alone)
        self.assertEqual(kinds(g), ["control_char", "digit_near_miss", "home_mismatch"])
        self.assertEqual(sum(e["kind"] == "home_mismatch" for e in g["errors"]), 1)

    def test_placeholder_literals(self):
        g = rp.grade_call(call("delegate", task="parameter_placeholder", context="context_placeholder"),
                          DECL, self.ctx, HOME)
        self.assertEqual([e["value"] for e in g["errors"]], ["parameter_placeholder", "context_placeholder"])
        clean = rp.grade_call(call("delegate", task="find the water shader", context="three 0.186"),
                              DECL, self.ctx, HOME)
        self.assertEqual(clean["errors"], [])


class Schema(unittest.TestCase):
    ctx = rp.context_index([])

    def test_old_string_is_an_unknown_arg_and_old_is_missing(self):
        g = rp.grade_call(call("edit_file", path="/tmp/a", old_string="x", new_string="y"), DECL, self.ctx, HOME)
        self.assertEqual(kinds(g), ["missing_required", "unknown_arg"])
        a = rp.grade_answer("", [call("edit_file", path="/tmp/a", old_string="x", new_string="y")],
                            "tool_calls", DECL, self.ctx, HOME)
        self.assertEqual((a["calls_with_error"], a["calls_with_schema_error"]), (0, 1))

    def test_invalid_json_and_unknown_tool(self):
        g = rp.grade_call({"name": "read_file", "arguments": '{"path": "/tmp/a'}, DECL, self.ctx, HOME)
        self.assertEqual(kinds(g), ["json_invalid"])
        g = rp.grade_call(call("run_image", path="/tmp/a"), DECL, self.ctx, HOME)
        self.assertEqual(kinds(g), ["unknown_tool"])

    def test_types(self):
        g = rp.grade_call(call("read_file", path="/tmp/a", start_line="12"), DECL, self.ctx, HOME)
        self.assertEqual(g["errors"], [])
        self.assertEqual([i["kind"] for i in g["info"] if i["kind"] == "numeric_string"], ["numeric_string"])
        g = rp.grade_call(call("read_file", path="/tmp/a", start_line=[1]), DECL, self.ctx, HOME)
        self.assertEqual(kinds(g), ["type_mismatch"])

    def test_a_truncated_call_is_not_an_unknown_arg(self):
        g = rp.grade_call(call("write_file", path="/tmp/a", _truncated=True), DECL, self.ctx, HOME)
        self.assertEqual(g["errors"], [])
        self.assertIn("truncated", [i["kind"] for i in g["info"]])


class DigitNearMiss(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = os.path.realpath(self.tmp.name)
        os.makedirs(os.path.join(self.root, "proj1896", "src"))
        self.ctx = rp.context_index([{"role": "user", "content": "work in %s/proj1896/src" % self.root}])

    def tearDown(self):
        self.tmp.cleanup()

    def test_an_inserted_digit_is_flagged_when_the_filesystem_confirms_it(self):
        g = rp.grade_call(call("read_file", path="%s/proj11896/src/a.js" % self.root), DECL, self.ctx, HOME)
        hit = [e for e in g["errors"] if e["kind"] == "digit_near_miss"]
        self.assertEqual([(e["got"], e["context"], e["edit"]) for e in hit],
                         [("proj11896", "proj1896", "insertion")])

    def test_transposition(self):
        self.assertEqual(rp.edit1("proj1986", "proj1896"), "transposition")
        self.assertEqual(rp.edit1("proj1897", "proj1896"), "substitution")
        self.assertIsNone(rp.edit1("proj2987", "proj1896"))

    def test_a_counter_is_not_a_copy_error(self):
        # /tmp/shot13 beside a context /tmp/shot12: neither exists -> unconfirmed, info only
        ctx = rp.context_index([{"role": "tool", "content": "wrote %s/shot12/index.html" % self.root}])
        g = rp.grade_call(call("run_command", command="mkdir -p %s/shot13" % self.root), DECL, ctx, HOME)
        self.assertEqual(g["errors"], [])
        self.assertIn("digit_near_miss_unconfirmed", [i["kind"] for i in g["info"]])


class Answer(unittest.TestCase):
    def test_no_call_and_markup_are_notes_not_errors(self):
        a = rp.grade_answer("<tool_call><function=read_file>", [], "stop", DECL, rp.context_index([]), HOME)
        self.assertEqual(a["calls_with_error"], 0)
        self.assertEqual(sorted(n["kind"] for n in a["notes"]), ["markup_in_content", "no_tool_call"])


if __name__ == "__main__":
    unittest.main()
