"""crow-nest #158: tests of the layerwise glm5_next reference runner (CPU, ~1 min).

Run: .venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v

- the proof (small shapes): runner == HF's full Glm5NextForConditionalGeneration to 1e-5 per layer,
  routing ids, DSA top-k and logits, at index_topk 32 and 2048, prompt and decode rows;
- the proof can go red: a collapsed mHC hand-over and decode rows without the layer cache both fail it;
- FP8 unpack == HF's Fp8Dequantize on whole blocks and one-block partial tensors;
- the CLI: --weights fp8-originals with a layer split (0:3 then 3:8) writes the same files as one run;
  --state-dtype bf16; --weights container exits 2 with a clear error.
"""
import io
import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402
import glm5_layerwise as LW  # noqa: E402


def quiet(*_):
    pass


def _silent(fn, *a, **k):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class Proof(unittest.TestCase):
    def test_small_shapes_match_hf_full_model(self):
        ok, tables = _silent(LW.selftest, "small", log=quiet)
        self.assertTrue(ok, tables)
        for k, rows in tables.items():
            layers = [r for r in rows if r["layer"] != "logits"]
            self.assertEqual(len(layers), 8)
            self.assertEqual({r["kind"] for r in layers}, {"kda+dense", "dsa+moe", "kda+moe"})
            for r in layers:
                self.assertLessEqual(r["max_abs_prompt"], LW.TOL)
                self.assertLessEqual(r["max_abs_decode"], LW.TOL)
            dsa = [r for r in layers if r["kind"].startswith("dsa")]
            self.assertTrue(all(r["dsa_topk_mismatch_rows"] == 0 for r in dsa))
            if k == 32:  # the indexer really drops pools: <= 32 + 3 tail tokens of up to 100 visible
                self.assertTrue(all(r["dsa_selected_per_row"] < 36 for r in dsa))
            self.assertLessEqual(rows[-1]["max_abs"], LW.TOL)
            self.assertEqual(rows[-1]["anchors"], 100)

    def test_collapsed_mhc_handover_fails_the_proof(self):
        real = LW.read_state

        def collapsed(path, shape):  # hands over the stream mean in all 4 streams
            x = real(path, shape)
            return x.mean(dim=1, keepdim=True).expand_as(x).contiguous()

        with mock.patch.object(LW, "read_state", collapsed):
            ok, tables = _silent(LW.selftest, "small", topks=(32,), log=quiet)
        self.assertFalse(ok)
        rows = tables[32]
        self.assertTrue(rows[0]["ok"])  # layer 0 reads the embeddings, not a hand-over
        self.assertGreater(rows[1]["max_abs_prompt"], 1e-3)

    def test_decode_rows_without_the_layer_cache_fail_the_proof(self):
        with mock.patch.object(LW, "new_layer_cache", lambda tc: None):
            ok, tables = _silent(LW.selftest, "small", topks=(32,), log=quiet)
        self.assertFalse(ok)
        rows = tables[32]
        self.assertEqual(rows[0]["max_abs_prompt"], 0.0)  # the prompt rows need no cache
        self.assertGreater(rows[0]["max_abs_decode"], 1e-3)


class Fp8(unittest.TestCase):
    def _hf(self, q, s):
        from transformers.integrations.finegrained_fp8 import Fp8Dequantize
        return Fp8Dequantize(None)._dequantize_one(q, s, torch.float32)

    def test_dequant_matches_hf_on_whole_and_one_block_tensors(self):
        g = torch.Generator().manual_seed(1)
        for shape in [(256, 384), (128, 128), (64, 256), (32, 4096)]:
            w = torch.randn(shape, generator=g) * torch.rand(shape, generator=g)
            q, s = G.fp8_quant(w)
            self.assertEqual(q.dtype, torch.float8_e4m3fn)
            self.assertEqual(tuple(s.shape), (math.ceil(shape[0] / 128), math.ceil(shape[1] / 128)))
            mine = G.fp8_dequant(q, s)
            self.assertTrue(torch.equal(mine, self._hf(q, s)), shape)
            self.assertLess(float(((mine - w).norm() / w.norm())), 0.05)

    def test_partial_blocks_follow_the_128_grid(self):
        # 192 rows = one full and one partial block; HF would infer a 96-row block from the grid,
        # the DeepSeek-V3 rule (and the FP8 originals) use 128 rows: row 150 takes scale row 1
        q = torch.ones(192, 128).to(torch.float8_e4m3fn)
        s = torch.tensor([[2.0], [3.0]])
        d = G.fp8_dequant(q, s)
        self.assertEqual(float(d[127, 0]), 2.0)
        self.assertEqual(float(d[128, 0]), 3.0)
        self.assertEqual(float(d[150, 0]), 3.0)
        with self.assertRaises(AssertionError):
            G.fp8_dequant(q, torch.ones(3, 1))

    def test_names_round_trip(self):
        t = torch.arange(2 * 6 * 3, dtype=torch.float32).view(2, 6, 3)
        parts = G.module_to_ckpt("mlp.experts.gate_up_proj", t)
        self.assertEqual(sorted(parts), ["mlp.experts.0.gate_proj.weight", "mlp.experts.0.up_proj.weight",
                                         "mlp.experts.1.gate_proj.weight", "mlp.experts.1.up_proj.weight"])
        self.assertTrue(torch.equal(parts["mlp.experts.1.up_proj.weight"], t[1, 3:]))
        self.assertEqual(G.ckpt_recipe("self_attn.forget_gate.A_log"), ("one", "self_attn.A_log"))
        self.assertEqual(G.ckpt_recipe("ffn_hc.scale"), ("one", "hc_ffn_scale"))
        self.assertEqual(G.ckpt_recipe("self_attn.q_a_proj.weight"), ("one", "self_attn.q_a_proj.weight"))


class Cli(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="glm5-cli-")
        cls.ck = os.path.join(cls.wd, "fp8")
        _silent(LW.make_synthetic, "small", cls.ck)
        cls.ids = os.path.join(cls.wd, "ids.json")
        with open(cls.ids, "w") as f:
            json.dump([5, 17, 900, 3, 3, 42, 7, 512, 1, 999, 64, 65], f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def _run(self, *args):
        return _silent(LW.main, ["run", "--weights", "fp8-originals", self.ck, "--ids", self.ids,
                                 "--decode", "3", *args])

    def test_layer_split_equals_one_run(self):
        one, split = os.path.join(self.wd, "one"), os.path.join(self.wd, "split")
        self.assertEqual(self._run("--out", one), 0)
        self.assertEqual(self._run("--out", split, "--layers", "0:3"), 0)
        self.assertEqual(self._run("--out", split, "--layers", "3:"), 0)
        a = json.load(open(os.path.join(one, "manifest.json")))
        b = json.load(open(os.path.join(split, "manifest.json")))
        self.assertTrue(a["complete"] and b["complete"])
        self.assertEqual(a["anchors"], [8, 9, 10, 11])
        names = set(a["files"])
        self.assertIn("l3-dsa-topk.i32", names)
        self.assertIn("l7-routing-ids.i32", names)
        self.assertNotIn("l2-routing-ids.i32", names)  # dense layer
        self.assertNotIn("l4-dsa-topk.i32", names)  # KDA layer
        self.assertEqual(set(b["files"]), names)
        self.assertEqual(b["layers"], [0, 8])
        self.assertEqual([r["layer"] for r in b["per_layer"]], list(range(8)))
        for n in names:
            self.assertEqual(a["files"][n]["sha256"], b["files"][n]["sha256"], n)
        ids = LW.load_file(one, a, "l5-routing-ids.i32")
        self.assertEqual(tuple(ids.shape), (12, 8))
        self.assertTrue(bool((ids[:, 1:] > ids[:, :-1]).all()))  # ascending, distinct

    def test_bf16_state(self):
        f32, bf = os.path.join(self.wd, "f32"), os.path.join(self.wd, "bf16")
        self.assertEqual(self._run("--out", f32, "--layers", "0:2"), 0)
        self.assertEqual(self._run("--out", bf, "--layers", "0:2", "--state-dtype", "bf16"), 0)
        a = json.load(open(os.path.join(f32, "manifest.json")))
        b = json.load(open(os.path.join(bf, "manifest.json")))
        self.assertIn("l0-output.bf16", b["files"])
        y32 = LW.load_file(f32, a, "l0-output.f32")
        y16 = LW.load_file(bf, b, "l0-output.bf16")
        self.assertTrue(torch.equal(y16, y32.to(torch.bfloat16).float()))  # same input, rounded output

    def test_a_missing_container_is_a_clear_error(self):
        # crow-nest #156: the stub this test pinned (#158) is replaced by the container back end
        # (test_glm5_container.py); a missing file still exits 2 with a clear error
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = LW.main(["run", "--weights", "container", os.path.join(self.wd, "x.cnq"), "--ids", self.ids,
                          "--out", os.path.join(self.wd, "c")])
        self.assertEqual(rc, 2)
        self.assertIn("no such file", err.getvalue())
        with self.assertRaises(G.ContainerError):
            G.open_weights("cnq", "x.cnq")


if __name__ == "__main__":
    unittest.main()
