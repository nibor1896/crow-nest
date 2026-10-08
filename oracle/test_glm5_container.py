"""crow-nest #156: tests of the `--weights container` back end of the layerwise glm5_next runner.

Run: .venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v
Needs the converter binary (cd converter && cargo build --release, or CROW_CONVERTER).

A synthetic glm5_next with the real schedule cut to nothing (45 layers, small widths, 16 experts),
written in the original naming with the dtypes of rev eb9eb208's headers, is converted by the real
converter with `--scales mse --layers 0-3 --with-embed-head`. Then:
- every container weight the back end hands out is the converter's decode, bit for bit against an
  independent decoder (oracle/cnq_weights.py); BF16 keeps and F32 carries equal the originals;
- the runner runs layers 0-3 on both back ends; the container run differs from the FP8 run (the
  quantisation is really there) and agrees with it to cosine > 0.99 per layer;
- a layer the partial container does not hold is refused before anything runs.
"""
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402
import glm5_compare as C  # noqa: E402
import glm5_layerwise as LW  # noqa: E402
from cnq_weights import CnqReader  # noqa: E402

L3 = G.LM + "layers.3."


def _silent(fn, *a, **k):
    with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
        return fn(*a, **k)


class Container(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert os.path.isfile(G.CONVERTER), f"{G.CONVERTER}: build the converter first (cargo build --release)"
        from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
        cls.wd = tempfile.mkdtemp(prefix="glm5-cnq-")
        cls.ck = os.path.join(cls.wd, "fp8")
        text, vision = LW.mini_config("small", 32)
        text = dict(text, num_hidden_layers=45, num_nextn_predict_layers=1, model_type="glm5_next_text")
        torch.manual_seed(LW.SEED)
        model = Glm5NextForConditionalGeneration(Glm5NextConfig(text_config=text, vision_config=vision)).float().eval()
        LW._perturb(model, LW.SEED)
        G.write_synthetic_checkpoint(model, cls.ck, LW._config_dict(text, vision), fp8_re=G.CNQ_FP8, f32_re=G.CNQ_F32)
        del model
        with open(os.path.join(cls.ck, "generation_config.json"), "w") as f:
            f.write("{}")
        cls.cnq = os.path.join(cls.wd, "glm-l0-3.cnq")
        p = subprocess.run([G.CONVERTER, "--scales", "mse", "--source-repo", "crow-nest/glm-synth", "--revision", "synth",
                            "--layers", "0-3", "--with-embed-head", cls.ck, cls.cnq], capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        cls.convert_log = p.stderr
        cls.ids = os.path.join(cls.wd, "ids.json")
        with open(cls.ids, "w") as f:
            json.dump([5, 17, 900, 3, 3, 42, 7, 512, 1, 999, 64, 65, 300, 301, 2, 77], f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def _run(self, kind, path, out, *args):
        return _silent(LW.main, ["run", "--weights", kind, path, "--ids", self.ids, "--decode", "3",
                                 "--out", os.path.join(self.wd, out), *args])

    def test_the_converter_wrote_a_partial_container(self):
        self.assertIn("coverage check (PARTIAL container, --layers 0-3 --with-embed-head)", self.convert_log)
        ws = G.ContainerSource(self.cnq)
        self.assertEqual(ws.partial["layers"], [0, 1, 2, 3])
        self.assertTrue(ws.has(L3 + "mlp.experts.15.down_proj.weight"))
        self.assertFalse(ws.has(G.LM + "layers.4.input_layernorm.weight"))
        prov = ws.provenance()
        self.assertEqual((prov["recipe"], prov["scales"]), ("cnq4.5-glm5-next", "mse"))
        self.assertEqual(G.text_config_from_dict(ws.config_dict()).num_hidden_layers, 45)

    def test_weights_are_the_converter_decode_bit_for_bit(self):
        ws, fp8 = G.ContainerSource(self.cnq), G.WeightSource("fp8", self.ck)
        ref = CnqReader(self.cnq)  # an independent decoder of the same bytes
        nv = [n for n, t in ws.tensors.items() if t["dtype"] == "nvfp4"]
        self.assertGreater(len(nv), 50)
        got = dict(zip(nv, ws.get_many(nv)))
        for n in nv:
            self.assertTrue(torch.equal(got[n], ref.tensor(n)), n)
            self.assertEqual(list(got[n].shape), ws.shape_of(n))
        # the quantisation is there: NVFP4 is not the FP8 original
        n = L3 + "mlp.experts.0.gate_proj.weight"
        self.assertFalse(torch.equal(got[n], fp8.get(n)))
        # keeps and carries are the originals
        for n in (L3 + "mlp.gate.weight", L3 + "mlp.gate.e_score_correction_bias", G.LM + "layers.0.self_attn.A_log",
                  G.LM + "layers.0.hc_attn_base", G.LM + "layers.0.hc_attn_fn", "lm_head.weight"):
            self.assertIn(ws.dtype_of(n), ("bf16", "f32"), n)
            self.assertTrue(torch.equal(ws.get(n), fp8.get(n)), n)
        self.assertTrue(torch.equal(ws.rows("lm_head.weight", 10, 20), fp8.get("lm_head.weight")[10:20]))
        ids = [5, 17, 5, 999]
        self.assertTrue(torch.equal(ws.embed(ids), fp8.embed(ids)))

    def test_both_back_ends_run_layers_0_to_3(self):
        self.assertEqual(self._run("fp8-originals", self.ck, "a", "--layers", "0:4"), 0)
        self.assertEqual(self._run("container", self.cnq, "b", "--layers", "0:4"), 0)
        a, b = os.path.join(self.wd, "a"), os.path.join(self.wd, "b")
        ma, mb = (json.load(open(os.path.join(d, "manifest.json"))) for d in (a, b))
        self.assertEqual(set(ma["files"]), set(mb["files"]))
        self.assertIn("l3-routing-ids.i32", mb["files"])
        self.assertIn("l3-dsa-topk.i32", mb["files"])
        self.assertTrue(mb["weights"]["weights"].startswith("cnq"))
        self.assertEqual(mb["weights"]["partial"]["layers"], [0, 1, 2, 3])
        self.assertEqual(ma["files"]["embed.f32"]["sha256"], mb["files"]["embed.f32"]["sha256"])  # BF16 keep
        res = C.compare_runs(a, b)
        self.assertEqual([r["layer"] for r in res["layers"]], [0, 1, 2, 3])
        for r in res["layers"]:
            self.assertGreater(r["max_abs"], 0.0, r)  # NVFP4 moved the layer
            self.assertGreater(r["cosine"], 0.99, r)

    def test_layers_outside_the_partial_container_are_refused(self):
        err = io.StringIO()
        with redirect_stderr(err), redirect_stdout(io.StringIO()):
            rc = LW.main(["run", "--weights", "container", self.cnq, "--ids", self.ids,
                          "--out", os.path.join(self.wd, "c"), "--layers", "0:5"])
        self.assertEqual(rc, 2)
        self.assertIn("PARTIAL container", err.getvalue())
        self.assertFalse(os.path.exists(os.path.join(self.wd, "c")))
        with self.assertRaises(G.ContainerError):
            G.ContainerSource(self.cnq).get(G.LM + "layers.4.input_layernorm.weight")


if __name__ == "__main__":
    unittest.main()
