#!/usr/bin/env python3
"""#148: tests of tools/glm_entropy_report.py.

  python -I tools/test_glm_entropy_report.py

Synthetic sidecars and containers in the converter's own layout (sidecar lines with h_codes/h_scales,
`code_summary` records, a CNQ1 file with or without an index trailer, a resume journal beside it).
The guards: no G2 line while the trailer is missing, the journal exists, the container is partial,
the block count is short or the converter's own summary disagrees; a histogram whose sum is not the
value count is refused. The coder sizes are checked by hand-computable histograms and, where the
step-6 partial container is on disk, against the converter's own `expert_blocks` record.
"""
import contextlib
import io
import json
import math
import os
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import glm_entropy_report as ger  # noqa: E402

STEP6 = Path("C:/Users/robin/dev/crow-nest/models/GLM-5.3-Flash-step06/GLM-5.3-Flash-CNQ4.5-L0-3.cnq.sidecar.jsonl")
N_PROJ = 4096 * 2048


def hist(skew):
    """A 16/256 histogram pair for one projection of 8,388,608 values; `skew` puts mass on few codes."""
    w = [skew ** -i for i in range(16)]
    hc = [int(N_PROJ * x / sum(w)) for x in w]
    hc[0] += N_PROJ - sum(hc)
    ns = N_PROJ // 16
    if skew == 1.0:  # flat: every scale byte equally often, nothing to save
        return hc, [ns // 256] * 256
    hs = [0] * 256
    hs[0x60], hs[0x61], hs[0x62] = ns // 2, ns // 4, ns - ns // 2 - ns // 4
    return hc, hs


def line(cls, layer, expert, hc, hs, n=N_PROJ):
    name = ("model.language_model.layers.%d.mlp.experts.%d.%s_proj.weight" % (layer, expert, cls.split("_")[1])
            if expert is not None else "model.language_model.layers.%d.mlp.shared_experts.%s" % (layer, cls))
    return {"name": name, "dtype": "nvfp4", "class": cls, "layer": layer, "expert": expert, "n": n,
            "h_codes": hc, "h_scales": hs, "section": "text"}


def write_case(d, skew=1.0, trailer=True, journal=False, partial=False, summary=True, bad_summary=False,
               blocks=((3, 0), (3, 1)), draft=None):
    side = os.path.join(d, "m.cnq.sidecar.jsonl")
    lines, tensors = [], []
    for (l, e) in blocks:
        for c in ger.PROJ:
            hc, hs = hist(skew)
            lines.append(line(c, l, e, hc, hs))
            tensors.append({"name": lines[-1]["name"], "dtype": "nvfp4", "offset": 0, "len": 4718592})
    res = ger.census(lines)
    if summary:
        t = res["total"]
        lines.append({"record": "code_summary", "scope": "expert_blocks", "blocks": t.n, "raw_bytes": t.raw,
                      "coded_bytes": t.coded + (1 if bad_summary else 0), "table_bytes": t.tables,
                      "entropy_bytes": t.entropy})
    with open(side, "w", encoding="utf-8") as f:
        for x in lines:
            f.write(json.dumps(x) + "\n")
    cnq = os.path.join(d, "m.cnq")
    with open(cnq, "wb") as f:
        f.write(b"CNQ1" + bytes(8) + os.urandom(4096))
        if trailer:
            ix = {"format_version": 2, "blob_offset": 12, "tensors": tensors}
            if partial:
                ix["partial"] = {"filter": "--layers 3"}
            b = json.dumps(ix).encode()
            f.write(b + struct.pack("<Q", len(b)))
    if journal:
        open(cnq + ".journal.jsonl", "w").close()
    args = ["census", side]
    if draft is not None:
        p = os.path.join(d, "draft.json")
        with open(p, "w") as f:
            json.dump(draft, f)
        args += ["--decoder-draft", p]
    return args


def run(args):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = ger.main(args)
    return rc, buf.getvalue()


class Small(unittest.TestCase):
    """N_BLOCKS patched to the two synthetic blocks; everything else as for GLM."""

    def setUp(self):
        self._n = ger.N_BLOCKS
        ger.N_BLOCKS = 2
        self.d = tempfile.TemporaryDirectory()

    def tearDown(self):
        ger.N_BLOCKS = self._n
        self.d.cleanup()

    def g2_lines(self, out):
        return [x for x in out.splitlines() if x.startswith("G2 ")]


class TestCoder(unittest.TestCase):
    def test_hand_computable_huffman(self):
        h = [0] * 16
        h[1], h[2], h[3] = 512, 256, 256
        self.assertEqual(ger.huffman_cost_bits(h, 12), 1536)
        self.assertAlmostEqual(ger.entropy_bits(h), 1536.0)
        s = [0] * 256
        s[0x38] = 64
        self.assertEqual(ger.huffman_cost_bits(s, 15), 64)  # a lone symbol costs 1 bit each
        c = ger.coded(h, s)
        self.assertEqual(c["raw"], 1024 // 2 + 64)
        self.assertEqual(c["coded"], 1536 // 8 + 64 // 8 + 8 + 128)  # entropy.rs test, same numbers

    def test_length_limit_binds(self):
        h, (a, b) = [], (1, 1)
        for _ in range(16):
            h.append(a)
            a, b = b, a + b
        self.assertEqual(ger.huffman_cost_bits(h, 4), 4 * sum(h))  # flat 4-bit code is optimal
        free = ger.huffman_cost_bits(h, 15)
        self.assertLess(free, 4 * sum(h))
        self.assertLess(free, ger.entropy_bits(h) + sum(h))
        self.assertGreaterEqual(free, ger.entropy_bits(h))

    def test_expert_block_summary_matches_the_converter_test(self):
        # entropy.rs the_expert_block_summary_has_one_table_per_block: 24+2+136 + 8+1+136
        hc, hs = [0] * 16, [0] * 256
        hc[0], hs[1] = 64, 4
        b0 = [hc[i] * 3 for i in range(16)], [hs[i] * 3 for i in range(256)]
        self.assertEqual(ger.coded(*b0)["coded"], 24 + 2 + 136)
        self.assertEqual(ger.coded(hc, hs)["coded"], 8 + 1 + 136)

    @unittest.skipUnless(STEP6.exists(), "step-6 partial container not on this machine")
    def test_equal_to_the_converter_on_the_step6_sidecar(self):
        lines, _ = ger.read_sidecar(str(STEP6), tolerate_tail=False)
        res = ger.census(lines)
        self.assertIsNone(ger.check_against_converter(res))
        self.assertEqual(res["total"].coded, 3803962857)
        self.assertEqual(res["total"].n, 288)


class TestGuards(Small):
    def test_histogram_sum_is_checked(self):
        args = write_case(self.d.name)
        side = args[1]
        with open(side) as f:
            rows = [json.loads(x) for x in f]
        rows[0]["h_codes"][0] += 1
        with open(side, "w") as f:
            f.writelines(json.dumps(x) + "\n" for x in rows)
        rc, _ = run(args)
        self.assertEqual(rc, 2)

    def test_no_trailer_and_journal_is_provisional(self):
        rc, out = run(write_case(self.d.name, skew=3.0, trailer=False, journal=True, summary=False,
                                 draft={"decode_ms_per_cold_visit": 1.0}))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("PROVISIONAL, not G2", out)
        self.assertEqual(self.g2_lines(out), [])

    def test_missing_trailer_alone_blocks_the_verdict(self):
        rc, out = run(write_case(self.d.name, trailer=False))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("no index trailer", out)
        self.assertEqual(self.g2_lines(out), [])

    def test_journal_alone_blocks_the_verdict(self):
        rc, out = run(write_case(self.d.name, journal=True))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("resume journal still present", out)
        self.assertEqual(self.g2_lines(out), [])

    def test_partial_container_blocks_the_verdict(self):
        rc, out = run(write_case(self.d.name, partial=True))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertEqual(self.g2_lines(out), [])

    def test_short_block_count_blocks_the_verdict(self):
        rc, out = run(write_case(self.d.name, blocks=((3, 0),)))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("1 of 2 expert blocks", out)

    def test_missing_or_unequal_converter_summary_blocks_the_verdict(self):
        rc, out = run(write_case(self.d.name, summary=False))
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("no code_summary", out)
        with tempfile.TemporaryDirectory() as d2:
            rc, out = run(write_case(d2, bad_summary=True))
            self.assertEqual(rc, ger.PROVISIONAL_EXIT)
            self.assertIn("converter coded_bytes", out)
            self.assertEqual(self.g2_lines(out), [])

    def test_unfinished_tail_line_is_skipped_only_while_running(self):
        args = write_case(self.d.name, trailer=False, journal=True)
        with open(args[1], "a") as f:
            f.write('{"name": "half')
        rc, out = run(args)
        self.assertEqual(rc, ger.PROVISIONAL_EXIT)
        self.assertIn("one unfinished tail line skipped", out)


class TestVerdict(Small):
    def test_flat_codes_fail(self):
        rc, out = run(write_case(self.d.name, skew=1.0, draft={"decode_ms_per_cold_visit": 1.0}))
        self.assertEqual(rc, 0)
        g = self.g2_lines(out)
        self.assertEqual(len(g), 1)
        self.assertTrue(g[0].startswith("G2 failed"), g)

    def test_skewed_codes_pass_only_with_a_named_decode_time(self):
        rc, out = run(write_case(self.d.name, skew=3.0))
        self.assertEqual(rc, 0)
        self.assertTrue(self.g2_lines(out)[0].startswith("G2 not answered"), out)
        with tempfile.TemporaryDirectory() as d2:
            rc, out = run(write_case(d2, skew=3.0, draft={"decode_ms_per_cold_visit": 28.2, "summary": "x"}))
            self.assertTrue(self.g2_lines(out)[0].startswith("G2 passed"), out)

    def test_threshold_is_inclusive_at_8_percent(self):
        self.assertEqual(ger.verdict(0.08, {"decode_ms_per_cold_visit": 1.0})[0], "passed")
        self.assertEqual(ger.verdict(math.nextafter(0.08, 0), {"decode_ms_per_cold_visit": 1.0})[0], "failed")


class TestLocate(unittest.TestCase):
    def test_locate_finds_the_contiguous_block(self):
        with tempfile.TemporaryDirectory() as d:
            cnq = os.path.join(d, "m.cnq")
            pre = "model.language_model.layers.3.mlp.experts.0."
            ts = [{"name": pre + p + "_proj.weight", "offset": 4084 + k * 4718592, "len": 4718592, "dtype": "nvfp4"}
                  for k, p in enumerate(("gate", "up", "down"))]
            b = json.dumps({"format_version": 2, "blob_offset": 12, "tensors": ts}).encode()
            with open(cnq, "wb") as f:
                f.write(b"CNQ1" + bytes(8) + bytes(64) + b + struct.pack("<Q", len(b)))
            rc, out = run(["locate", cnq, "--layer", "3", "--expert", "0"])
            self.assertEqual(rc, 0)
            r = json.loads(out)
            self.assertEqual((r["offset"], r["len"], r["aligned_4096"]), (4096, 3 * 4718592, True))


if __name__ == "__main__":
    unittest.main()
