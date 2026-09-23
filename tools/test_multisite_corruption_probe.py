#!/usr/bin/env python3
"""#91: unit tests of tools/multisite-corruption-probe.py (no engine, no GPU, no tokenizer).

  python3 tools/test_multisite_corruption_probe.py

What is pinned:
  - `locate_site` finds the first differing char of produced vs correct inside the named
    parameter of the rendered call (o[113] vs o[13] -> the second `1`), honours `occurrence`,
    and refuses a produced span outside the parameter or a produced span that is a prefix;
  - `forced_body` strips the two fields serve refuses together with `crow_force_ids`
    (reasoning_budget_tokens / _message) and makes the request greedy, grammar-free, unstreamed,
    with logprobs and max_tokens = forced + 1, without touching messages or tools;
  - `read_site` reads lp(corrupt) from the forced entry, lp(correct) from the alternatives by
    crow_id (None when absent), and refuses an entry whose forced id is not the corrupt id;
  - `summarize` gives the corrupt-win fraction and mean margin;
  - `run` against a STUB serve: the wire form serve returns for `crow_force_ids` +
    `logprobs` (entry and alternatives with `crow_id`), including the second request that
    forces the correct id when it is not in the top 20;
  - `count_digit_errors` on the live corruption shapes (o[113], mat44, 0.90&0.20, nibor1196).
The site set itself (tools/corpora/91-multisite-0923.json) is checked for shape: every site's
token_index lies inside its request's forced ids and the forced id there is the corrupt id.
"""
import http.server
import importlib.util
import json
import os
import tempfile
import threading
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("multisite", TOOLS / "multisite-corruption-probe.py")
ms = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ms)

CALL = ("<think>\nr\n</think>\n\n<tool_call>\n<function=write_file>\n<parameter=content>\n"
        "o[12]=p[0]; o[113]=p[1]; o[14]=p[2]; o[115]=1;\n</parameter>\n<parameter=path>\nsrc/math.js\n"
        "</parameter>\n</function>\n</tool_call>")


class LocateSite(unittest.TestCase):
    def test_first_difference_is_the_inserted_digit(self):
        cs = CALL.find("<tool_call>")
        at, div, cp = ms.locate_site(CALL, cs, "content", "o[113]=p[1]", "o[13]=p[1]")
        self.assertEqual(CALL[at:at + 6], "o[113]")
        self.assertEqual(cp, 3)                      # "o[1" is common
        self.assertEqual(CALL[div], "1")             # the corrupt char
        self.assertEqual(CALL[div - 3:div], "o[1")

    def test_occurrence_and_parameter_scope(self):
        cs = CALL.find("<tool_call>")
        at0, _, _ = ms.locate_site(CALL, cs, "content", "o[1", "o[X")
        at1, _, _ = ms.locate_site(CALL, cs, "content", "o[1", "o[X", occurrence=1)
        self.assertLess(at0, at1)
        with self.assertRaises(ValueError):          # the path is not inside `content`
            ms.locate_site(CALL, cs, "content", "src/math.js", "src/mat.js")
        with self.assertRaises(ValueError):          # produced is a prefix of correct
            ms.locate_site(CALL, cs, "content", "o[1", "o[13")
        with self.assertRaises(ValueError):
            ms.locate_site(CALL, cs, "missing", "o[1", "o[2")


class ForcedBody(unittest.TestCase):
    def test_fields(self):
        live = {"model": "crow", "messages": [{"role": "user", "content": "x"}], "tools": [1, 2],
                "temperature": 1.0, "stream": True, "stream_options": {"include_usage": True},
                "reasoning_effort": "high", "reasoning_budget_tokens": 1024,
                "reasoning_budget_message": "m", "seed": 7, "max_tokens": 16384}
        b = ms.forced_body(live, [5, 6, 7])
        for k in ("reasoning_budget_tokens", "reasoning_budget_message", "stream_options", "seed"):
            self.assertNotIn(k, b)
        self.assertEqual(b["crow_force_ids"], [5, 6, 7])
        self.assertEqual(b["max_tokens"], 4)
        self.assertEqual((b["temperature"], b["tool_choice"], b["stream"], b["logprobs"], b["top_logprobs"]),
                         (0, "none", False, True, 20))
        self.assertEqual(b["reasoning_effort"], "high")          # the render input stays
        self.assertIs(b["messages"], live["messages"])
        self.assertIn("reasoning_budget_tokens", live)           # the live body is not mutated


SITE = {"id": "s", "corrupt_id": 16, "correct_id": 18, "corrupt_token": "1", "correct_token": "3"}


def entry(forced, lp, alts):
    return {"token": "t", "crow_id": forced, "logprob": lp,
            "top_logprobs": [{"token": "a%d" % i, "crow_id": i, "logprob": l} for i, l in alts]}


class ReadSite(unittest.TestCase):
    def test_bare_like(self):
        g = ms.read_site(entry(16, -0.01, [(16, -0.01), (18, -4.4)]), SITE)
        self.assertEqual((g["lp_corrupt"], g["lp_correct"], g["top1_id"]), (-0.01, -4.4, 16))

    def test_correct_absent(self):
        self.assertIsNone(ms.read_site(entry(16, -9.0, [(1, -0.1), (2, -2.0)]), SITE)["lp_correct"])

    def test_wrong_forced_id_is_refused(self):
        with self.assertRaises(ValueError):
            ms.read_site(entry(99, -1.0, [(16, -1.0)]), SITE)

    def test_summary(self):
        rows = [{"lp_correct": -4.0, "lp_corrupt": -0.1, "margin": -3.9, "corrupt_wins": True,
                 "top1_id": 16, "correct_id": 18, "corrupt_id": 16},
                {"lp_correct": -0.1, "lp_corrupt": -6.0, "margin": 5.9, "corrupt_wins": False,
                 "top1_id": 18, "correct_id": 18, "corrupt_id": 16}]
        s = ms.summarize(rows)
        self.assertEqual((s["corrupt_wins"], s["corrupt_win_fraction"], s["mean_margin"], s["correct_top1"]),
                         (1, 0.5, 1.0, 1))


class Stub(http.server.BaseHTTPRequestHandler):
    bodies = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        Stub.bodies.append(body)
        ids = body["crow_force_ids"]
        ents = []
        for i, x in enumerate(ids):
            # position 2 is the site: the correct id 18 is outside the top list unless forced
            alts = [(x, -0.5), (1, -1.0)] if not (i == 2 and x == 16) else [(16, -0.2), (1, -1.0)]
            ents.append(entry(x, -0.5 if x != 18 else -3.25, alts))
        doc = {"choices": [{"message": {"content": ""}, "finish_reason": "length",
                            "logprobs": {"content": ents}}],
               "usage": {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 0}}}
        raw = json.dumps(doc).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


class RunAgainstStub(unittest.TestCase):
    def test_run(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            crow = d / "crow_core.py"
            crow.write_text(
                "DEFAULT_MODEL='crow'\n"
                "def _post_stream(*a, **k): pass\n"
                "class Conversation:\n"
                "    def __init__(self): self._messages=[]\n"
                "    def payload(self): return [dict(m) for m in self._messages]\n"
                "def sampling_for(m): return {'temperature':1.0,'top_p':0.95,'min_p':0.0,'top_k':20}\n"
                "def stream_reply(conv, **k):\n"
                "    _post_stream('u', {'model':'crow','messages':conv.payload(),'tools':[{'type':'function'}],'reasoning_effort':'high',\n"
                "                  'reasoning_budget_tokens':1024,'reasoning_budget_message':'m'}, '', 10)\n")
            sess = d / "seg.json"
            sess.write_text(json.dumps({"messages": [{"role": "system", "content": "s"},
                                                     {"role": "user", "content": "u"},
                                                     {"role": "assistant", "content": "a"}]}))
            spec = d / "spec.json"
            spec.write_text("{}")
            sites = d / "sites.json"
            sites.write_text(json.dumps({
                "spec": str(spec), "served_model": "m", "crow_core": str(crow),
                "crow_core_sha256": ms.sha256_file(crow),
                "segments": {"1": {"file": str(sess), "sha256": ms.sha256_file(sess)}},
                "sites": [dict(SITE, segment="1", at=2, token_index=2, **{"class": "c"}, produced="p",
                               correct="q", rendered_prompt_tokens=100, live_prompt_tokens=100)],
                "requests": [{"segment": "1", "at": 2, "force_ids": [5, 6, 16], "sites": ["s"]}]}))
            srv = http.server.HTTPServer(("127.0.0.1", 0), Stub)
            th = threading.Thread(target=srv.serve_forever, daemon=True)
            th.start()
            try:
                out = d / "arm.json"
                ms.main(["run", "--sites", str(sites), "--port", str(srv.server_port), "--label", "t",
                         "--json", str(out)])
            finally:
                srv.shutdown()
            doc = json.loads(out.read_text())
            row = doc["rows"][0]
            self.assertEqual(row["lp_corrupt"], -0.5)
            self.assertTrue(row["correct_forced"])            # 18 was not in the top list
            self.assertEqual(row["lp_correct"], -3.25)       # read from the second, forced request
            self.assertTrue(row["corrupt_wins"])
            self.assertEqual(row["prompt_tokens_delta_vs_live"], 0)
            self.assertEqual(Stub.bodies[0]["crow_force_ids"], [5, 6, 16])
            self.assertEqual(Stub.bodies[1]["crow_force_ids"], [5, 6, 18])
            self.assertNotIn("reasoning_budget_tokens", Stub.bodies[0])
            self.assertEqual(doc["summary"]["corrupt_win_fraction"], 1.0)


class DigitErrors(unittest.TestCase):
    def test_live_shapes(self):
        n, kinds, nidx = ms.count_digit_errors(
            "o[12]=p[0]; o[113]=p[1]; o[15]=1;\nuniform mat44 uVP;\n[128,82,52, 0.90&0.20]\n/home/nibor1196/x")
        self.assertEqual(n, 4)
        self.assertEqual(kinds["mat4 index > 15"], ["o[113]"])
        self.assertEqual(nidx, 3)
        self.assertEqual(ms.count_digit_errors("o[0]=1; o[15]=2; mat4 m; /home/nibor1896/y")[0], 0)


class SiteSet(unittest.TestCase):
    def test_shape(self):
        p = TOOLS / "corpora" / "91-multisite-0923.json"
        if not p.exists():
            self.skipTest("site set not built")
        doc = json.loads(p.read_text())
        reqs = {(r["segment"], r["at"]): r for r in doc["requests"]}
        self.assertGreaterEqual(len(doc["sites"]), 20)
        for s in doc["sites"]:
            r = reqs[(s["segment"], s["at"])]
            self.assertIn(s["id"], r["sites"])
            self.assertLess(s["token_index"], len(r["force_ids"]))
            self.assertEqual(r["force_ids"][s["token_index"]], s["corrupt_id"])
            self.assertNotEqual(s["corrupt_id"], s["correct_id"])
            self.assertIsNotNone(s["live_prompt_tokens"], s["id"])


if __name__ == "__main__":
    unittest.main()
