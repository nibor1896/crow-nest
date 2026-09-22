#!/usr/bin/env python3
"""#91: the unit tests of the replay probe's grader.

  python tools/test_corruption_replay_probe.py      # no engine, no GPU, no Crow import

The #91 logprobs cases (`Logprobs`) run `post` against a STUB server on 127.0.0.1 that answers
the wire form serve sends with `logprobs: true` (architecture 7.11.22), streamed and as one
document, and drive the divergence report on a synthetic near-tie at the 11896 byte.

Every check of `tools/corruption-replay-probe.py` is a pure function of the tool call, the
declared tools and the context -- except `digit_near_miss`, which asks the filesystem, so
its cases build their own directory. The three live corruptions of 2026-09-22 (session
messages [2], [26], [69]) are pinned here in their stored shape.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import http.server
import importlib.util
import io
import json
import os
import tempfile
import threading
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


def _e(tok, lp, top):
    """one OpenAI logprobs entry as serve writes it"""
    b = list(tok.encode("utf-8"))
    return {"token": tok, "logprob": lp, "bytes": b,
            "top_logprobs": [{"token": t, "logprob": l, "bytes": list(t.encode("utf-8"))} for t, l in top]}


# `/home/nibor11896/three-staging` as the model would spell it, the second `1` a near-tie
# with `8` (the correct continuation): margin +0.02 nats
CORRUPT = [
    _e("<tool_call>", -0.01, [("<tool_call>", -0.01), ("I", -5.0)]),
    _e("\n<function=run_command>\n<parameter=cwd>\n", -0.02, [("\n<function=run_command>\n<parameter=cwd>\n", -0.02)]),
    _e("/home", -0.001, [("/home", -0.001), ("/tmp", -7.5)]),
    _e("/n", -0.002, [("/n", -0.002), ("/r", -8.0)]),
    _e("ibor", -0.0, [("ibor", -0.0), ("ib", -9.0)]),
    _e("1", -0.01, [("1", -0.01), ("2", -6.0)]),
    _e("1", -0.68, [("1", -0.68), ("8", -0.70), ("9", -4.0)]),
    _e("896", -0.05, [("896", -0.05), ("89", -3.2)]),
    _e("/three-staging", -0.3, [("/three-staging", -0.3), ("/three", -1.9)]),
    _e("\n</parameter>\n</function>\n</tool_call>", -0.01, [("\n</parameter>\n</function>\n</tool_call>", -0.01)]),
]


class _Stub(http.server.BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.server.bodies.append(body)
        args = '{"command": "ls", "cwd": "/home/nibor11896/three-staging"}'
        base = {"id": "chatcmpl-1", "object": "chat.completion.chunk", "created": 1, "model": "crow"}
        if not body.get("stream"):
            doc = {"id": "chatcmpl-1", "object": "chat.completion", "created": 1, "model": "crow",
                   "choices": [{"index": 0, "finish_reason": "tool_calls",
                                "message": {"role": "assistant", "content": "",
                                            "tool_calls": [{"id": "c0", "type": "function",
                                                            "function": {"name": "run_command", "arguments": args}}]},
                                "logprobs": {"content": CORRUPT, "refusal": None}}],
                   "usage": {"prompt_tokens": 6738, "completion_tokens": len(CORRUPT)}}
            raw = json.dumps(doc).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        frames = [dict(base, choices=[{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}])]
        for e in CORRUPT:      # serve: one logprobs chunk per id, before the id's text
            frames.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": None,
                                               "logprobs": {"content": [e], "refusal": None}}]))
        frames.append(dict(base, choices=[{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "id": "c0", "type": "function", "function": {"name": "run_command", "arguments": ""}}]}}]))
        frames.append(dict(base, choices=[{"index": 0, "finish_reason": None, "delta": {"tool_calls": [
            {"index": 0, "function": {"arguments": args}}]}}]))
        frames.append(dict(base, choices=[{"index": 0, "delta": {}, "finish_reason": "tool_calls"}],
                           usage={"prompt_tokens": 6738}))
        for f in frames:
            self.wfile.write(b"data: " + json.dumps(f).encode() + b"\n\n")
        self.wfile.write(b"data: [DONE]\n\n")


class Logprobs(unittest.TestCase):
    """#91: `--top-logprobs N` against a stub of serve's wire form."""

    @classmethod
    def setUpClass(cls):
        cls.srv = http.server.HTTPServer(("127.0.0.1", 0), _Stub)
        cls.srv.bodies = []
        threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cls.url = "http://127.0.0.1:%d/v1/chat/completions" % cls.srv.server_address[1]

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()

    def test_both_wire_forms_give_the_same_entries(self):
        for stream in (True, False):
            content, calls, finish, usage, lps = rp.post(self.url, {"stream": stream, "logprobs": True,
                                                                    "top_logprobs": 3})
            self.assertEqual(lps, CORRUPT, "stream=%s" % stream)
            self.assertEqual((finish, calls[0]["name"]), ("tool_calls", "run_command"))
            self.assertIn("nibor11896", calls[0]["arguments"])
        self.assertEqual(self.srv.bodies[-1]["top_logprobs"], 3)

    def test_the_first_diverging_token_and_its_margin(self):
        _, calls, finish, _, lps = rp.post(self.url, {"stream": True})
        g = rp.grade_answer("", calls, finish, DECL, LiveShapes.ctx, HOME)
        self.assertEqual(g["calls_with_error"], 1)
        rep = rp.logprob_report(lps, g, HOME)
        [s] = [x for x in rep["spans"] if x["kind"] == "home_mismatch"]
        self.assertEqual((s["produced"], s["correct"]), ("/home/nibor11896", "/home/nibor1896"))
        self.assertEqual(s["token_index"], 6, "the SECOND 1 is where 11896 leaves 1896")
        self.assertEqual(s["first_diverging_token"]["token"], "1")
        self.assertAlmostEqual(s["margin"], 0.02)
        self.assertEqual(s["correct_alt"]["token"], "8")
        self.assertAlmostEqual(s["correct_alt"]["margin_vs_correct"], 0.02)
        self.assertEqual([t["i"] for t in s["window"]], list(range(0, 10)))
        # the clean-round view: the same token is the narrowest margin inside the call
        self.assertEqual(rep["narrowest_in_tool_calls"][0]["i"], 6)
        buf = io.StringIO()
        rp.print_report(rep, 2, 0, out=buf)
        self.assertIn("first diverging token #6 '1' logprob -0.6800, margin +0.0200, correct '8'", buf.getvalue())

    def test_edges(self):
        # a span the tokens do not contain is named, not guessed
        self.assertIsNone(rp.divergence(CORRUPT, "/home/other", "/home/nibor1896"))
        # a produced strict prefix diverges on the byte right after it
        d = rp.divergence(CORRUPT, "/home/nibor1", "/home/nibor1896")
        self.assertEqual(d["token_index"], 6)
        # no alternative in the list -> no margin; a placeholder has no correct form
        self.assertIsNone(rp.margin(_e("x", -0.1, [("x", -0.1)])))
        self.assertIsNone(rp.divergence(CORRUPT, "ibor", None)["correct"])
        self.assertIsNone(rp.error_pair({"kind": "json_invalid"}, HOME))



TFSPEC = importlib.util.spec_from_file_location("tf_compare", TOOLS / "teacher-forced-compare.py")
tfc = importlib.util.module_from_spec(TFSPEC)
TFSPEC.loader.exec_module(tfc)


def _ide(tok, tid, lp, top):
    return {"token": tok, "crow_id": tid, "logprob": lp, "bytes": list(tok.encode()),
            "top_logprobs": [{"token": t, "crow_id": i, "logprob": l, "bytes": list(t.encode())}
                             for t, i, l in top]}


class TeacherForced(unittest.TestCase):
    """#91: the --force-ids input and the compare tool, no server."""

    def _dump(self, d, name, entries, forced=0):
        path = os.path.join(d, name)
        with open(path, "w") as fh:
            json.dump({"rounds": [{"ids": [e["crow_id"] for e in entries], "entries": entries,
                                   "forced": forced, "prompt_tokens": 6738, "cached_tokens": 0}]}, fh)
        return path

    def test_force_ids_from_a_list_or_a_dump(self):
        with tempfile.TemporaryDirectory() as d:
            lst = os.path.join(d, "ids.json")
            with open(lst, "w") as fh:
                json.dump([5, 6, 7], fh)
            self.assertEqual(rp.load_force_ids(lst), [5, 6, 7])
            self.assertEqual(rp.load_force_ids(lst, 2), [5, 6])
            dump = self._dump(d, "dump.json", [_ide("a", 9, -0.1, []), _ide("b", 10, -0.2, [])])
            self.assertEqual(rp.load_force_ids(dump), [9, 10])
            with open(dump, "w") as fh:          # a dump written without crow_id
                json.dump({"rounds": [{"ids": [None, None], "entries": []}]}, fh)
            with self.assertRaises(SystemExit):
                rp.load_force_ids(dump)

    def test_compare_reads_the_digit_and_refuses_a_different_sequence(self):
        ref = [_ide("/n", 1, -0.0, [("/n", 1, -0.0)]),
               _ide("1", 16, -0.013, [("1", 16, -0.013), ("8", 23, -4.41), ("9", 24, -6.0)])]
        arm = [_ide("/n", 1, -0.001, [("/n", 1, -0.001)]),
               _ide("1", 16, -2.0, [("8", 23, -0.2), ("1", 16, -2.0)])]
        with tempfile.TemporaryDirectory() as d:
            r = self._dump(d, "ref.json", ref)
            a = self._dump(d, "arm.json", arm, forced=2)
            out = os.path.join(d, "out.json")
            buf = io.StringIO()
            import contextlib
            with contextlib.redirect_stdout(buf):
                rc = tfc.main(["--ref", "bare=" + r, "--arm", "x=" + a, "--at", "1", "--json", out])
            self.assertEqual(rc, 0)
            doc = json.load(open(out))
            [b, x] = doc["dumps"]
            self.assertAlmostEqual(b["margin_got_minus_want"], -0.013 + 4.41)
            self.assertAlmostEqual(x["margin_got_minus_want"], -2.0 + 0.2)
            self.assertIn("*", buf.getvalue(), "the top-1 flip at #1 is marked")
            # a token outside the top-N is a bound, not a number
            self.assertEqual(tfc.tok_lp(ref[1], "7"), (-6.0, False))
            bad = self._dump(d, "bad.json", [ref[0], _ide("2", 17, -1.0, [])], forced=2)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                self.assertEqual(tfc.main(["--ref", "bare=" + r, "--arm", "y=" + bad]), 2)

if __name__ == "__main__":
    unittest.main()
