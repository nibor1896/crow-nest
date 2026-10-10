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

The 3-bit container (class Mul1Container): the same checkpoint converted with `--experts-mul1` over
a store of synthetic MUL1 K=3 records (trellis words from a seeded generator, fp16 suh/svh of the
quantizer's kind). The back end hands out every routed-expert projection in the original basis:
undoing `diag(suh) H . H diag(svh) / 128` on it lands every value on the mul1 codebook (computed
here from its formula; the trellis decode itself stays the converter's), and the runner runs
layers 0-3 on the container with --capture-subblocks.
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

import numpy as np
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


def _mul1_codebook():
    """every fp16 value the mul1 codebook decodes a 16-bit state to (exllamav3 codebook.cuh, #181
    `mul1_decode`): sum = 0x6400 + bytesum(state * 0x83DCD12D), w = rne16(half(sum) * half(0x1eee) +
    half(0xc931)); the product and sum are exact in f64, so one f64 -> f16 cast is the rounding"""
    s = np.arange(1 << 16, dtype=np.uint64)
    x = (s * np.uint64(0x83DCD12D)) & np.uint64(0xFFFFFFFF)
    tot = np.uint64(0x6400) + sum((x >> np.uint64(8 * i)) & np.uint64(0xFF) for i in range(4))
    f16 = lambda b: np.array([b], dtype=np.uint16).view(np.float16).astype(np.float64)[0]  # noqa: E731
    h = tot.astype(np.uint16).view(np.float16).astype(np.float64)
    return np.unique((h * f16(0x1EEE) + f16(0xC931)).astype(np.float16).astype(np.float64))


def _sylvester(n=128):
    h = np.ones((1, 1))
    while h.shape[0] < n:
        h = np.block([[h, h], [h, -h]])
    return h


class Mul1Container(unittest.TestCase):
    """the back end on a container whose routed experts are MUL1 K=3 records (#181/#182)"""

    @classmethod
    def setUpClass(cls):
        assert os.path.isfile(G.CONVERTER), f"{G.CONVERTER}: build the converter first (cargo build --release)"
        from safetensors.torch import save_file
        from transformers import Glm5NextConfig, Glm5NextForConditionalGeneration
        cls.wd = tempfile.mkdtemp(prefix="glm5-mul1-")
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
        # the store (converter/src/mul1_store.rs): store.json, L03/E<eee>.safetensors, journal.jsonl
        H, I, E = text["hidden_size"], text["moe_intermediate_size"], text["n_routed_experts"]
        cls.store = os.path.join(cls.wd, "store")
        os.makedirs(os.path.join(cls.store, "L03"))
        rec = 3 * (H // 16) * (I // 16) * 48 * 2 + 3 * 2 * (H + I)
        head = {"format": "crow-nest mul1 store", "version": 1, "k": 3, "hidden": H, "inter": I, "n_experts": E,
                "record_bytes": -(-rec // 4096) * 4096, "quantizer": {"name": "synthetic (oracle tests)"},
                "calibration": {"files": [], "tokens": 0}}
        with open(os.path.join(cls.store, "store.json"), "w") as f:
            json.dump(head, f)
        g = torch.Generator().manual_seed(LW.SEED + 7)
        cls.scales = {}
        journal = []
        for e in range(E):
            ts = {}
            for p, (k, n) in (("gate", (H, I)), ("up", (H, I)), ("down", (I, H))):
                ts[p + ".trellis"] = torch.randint(-32768, 32768, (k // 16, n // 16, 48), dtype=torch.int16, generator=g)
                for s, m in (("suh", k), ("svh", n)):
                    # fp16 of random sign, magnitude (1 + f) 2^-3, f in [0, 1)
                    v = (1 + torch.rand(m, generator=g)) * 0.125 * (torch.randint(0, 2, (m,), generator=g) * 2 - 1)
                    ts[f"{p}.{s}"] = v.half()
                cls.scales[(e, p)] = (ts[p + ".suh"].double().numpy(), ts[p + ".svh"].double().numpy())
            rel = f"L03/E{e:03d}.safetensors"
            save_file(ts, os.path.join(cls.store, rel))
            journal.append({"layer": 3, "expert": e, "file": rel, "sha256": G.sha256_file(os.path.join(cls.store, rel))})
        with open(os.path.join(cls.store, "journal.jsonl"), "w", newline="\n") as f:
            f.write("".join(json.dumps(j) + "\n" for j in journal))
        cls.cnq = os.path.join(cls.wd, "glm3-l0-3.cnq")
        p = subprocess.run([G.CONVERTER, "--scales", "mse", "--source-repo", "crow-nest/glm-synth", "--revision", "synth",
                            "--layers", "0-3", "--with-embed-head", "--experts-mul1", cls.store, cls.ck, cls.cnq],
                           capture_output=True, text=True)
        assert p.returncode == 0, p.stderr[-3000:]
        cls.ids = os.path.join(cls.wd, "ids.json")
        with open(cls.ids, "w") as f:
            json.dump([5, 17, 900, 3, 3, 42, 7, 512, 1, 999, 64, 65, 300, 301, 2, 77], f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def test_mul1_experts_come_out_in_the_original_basis(self):
        ws = G.ContainerSource(self.cnq)
        names = [n for n, t in ws.tensors.items() if t["dtype"] == "mul1"]
        self.assertEqual(len(names), 3 * 16)
        prov = ws.provenance()
        self.assertEqual((prov["n_mul1"], prov["expert_codec"]["k"], prov["expert_codec"]["records"]), (48, 3.0, 16))
        cb, had = _mul1_codebook(), _sylvester()
        # byte sums 0..1020 index the engine's 1021-entry table (engine/src/cpu_mul1.rs); not all are reached
        self.assertTrue(100 < len(cb) <= 1021, len(cb))
        got = dict(zip(names, ws.get_many(names)))
        for n in names:
            e, p = int(n.split(".experts.")[1].split(".")[0]), n.split(".")[-2].split("_")[0]
            w = got[n].double().numpy()
            self.assertEqual(list(w.shape), ws.shape_of(n), n)
            self.assertTrue(np.isfinite(w).all(), n)
            suh, svh = self.scales[(e, p)]
            wk = w.T / suh[:, None] / svh[None, :]  # [in, out] = H W_hat H / 128
            k, m = wk.shape
            w_hat = np.einsum("ab,xbyd,dc->xayc", had, wk.reshape(k // 128, 128, m // 128, 128), had).reshape(k, m) / 128
            dist = np.abs(w_hat[..., None] - cb[np.clip(np.searchsorted(cb, w_hat), 1, len(cb) - 1)[..., None] + [-1, 0]]).min(-1)
            self.assertLess(dist.max(), 1e-5, n)
            if e == 0:  # the control: without the left Hadamard the values are off the codebook
                off = np.einsum("ibjc,cd->ibjd", wk.reshape(k // 128, 128, m // 128, 128), had).reshape(k, m) / 128
                d2 = np.abs(off[..., None] - cb[np.clip(np.searchsorted(cb, off), 1, len(cb) - 1)[..., None] + [-1, 0]]).min(-1)
                self.assertGreater(np.median(d2), 1e-4, n)
        n = names[0]
        self.assertTrue(torch.equal(ws.rows(n, 3, 70), got[n][3:70]))

    def test_the_runner_runs_layers_0_to_3_on_the_mul1_container(self):
        out = os.path.join(self.wd, "m")
        rc = _silent(LW.main, ["run", "--weights", "container", self.cnq, "--ids", self.ids, "--decode", "3",
                               "--out", out, "--layers", "0:4", "--capture-subblocks"])
        self.assertEqual(rc, 0)
        man = json.load(open(os.path.join(out, "manifest.json")))
        self.assertEqual(man["weights"]["n_mul1"], 48)
        for f in ("l3-output.f32", "l3-routing-ids.i32", "l3-dsa-topk.i32", "l3-ffn_hc-collapsed.f32"):
            self.assertIn(f, man["files"])
        y = np.fromfile(os.path.join(out, "l3-ffn-out.f32"), dtype="<f4")
        self.assertTrue(np.isfinite(y).all() and np.abs(y).max() > 0)


if __name__ == "__main__":
    unittest.main()
