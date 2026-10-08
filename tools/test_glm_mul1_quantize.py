#!/usr/bin/env python3
"""#182: tests of tools/glm_mul1_quantize.py (the MUL1 K=3 expert quantizer driver). No real weights.

  python -I tools/test_glm_mul1_quantize.py                              the pure tests (stdlib)
  .venv-oracle/Scripts/python.exe -I tools/test_glm_mul1_quantize.py     + `capture` on the runner's synthetic small
                                                                           checkpoint against HF's full model (CPU)
  .venv-exl3/Scripts/python.exe -I tools/test_glm_mul1_quantize.py       + `quantize` through exllamav3 on the GPU
                                                                           (seconds) and the converter reading the store

- the record size (9,474,048 B for GLM-5.3-Flash) and the expert count (43 x 288);
- the layer spec, the held-out file, a work or store dir inside the FP8 originals, and the disk check refuse;
- a torn journal line is cut; pruning deletes only records whose `.done` names the given container;
- `plan` prints the partial (layers 0-3) and the full commands;
- capture: the dumped MoE inputs equal HF's full model's `mlp` input on every row (BF16 rounding), the routed
  ids its router's; a capture interrupted after a layer resumes to the same files;
- quantize: every expert file holds exllamav3's 9 tensors at the record's shapes, its journal line its sha256; a
  killed run resumes without a duplicate; the MTP layer is quantized with an identity Hessian; `routed` works;
  the converter turns the store into a container with one 4096-aligned MUL1 record per expert, and pruning
  then frees exactly those records.
"""
import hashlib
import importlib.util
import json
import math
import os
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
sys.path.insert(0, str(HERE))
import glm_mul1_quantize as Q  # noqa: E402


def quiet(*_):
    pass


def has(*mods):
    try:
        for m in mods:
            importlib.import_module(m)
        return True
    except Exception:
        return False


def cuda():
    if not has("torch", "exllamav3"):
        return False
    import torch
    return torch.cuda.is_available()


class Pure(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="mul1-pure-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_record_size_and_expert_count(self):
        self.assertEqual(Q.record_bytes(), 9_474_048)
        self.assertEqual(Q.record_bytes() % 4096, 0)
        self.assertEqual(Q.record_bytes(128, 256), 40_960)  # the converter miniature's record (mul1_store tests)
        layers = Q.expert_layers(Q.parse_layers(None))
        self.assertEqual((len(layers), layers[0], layers[-1]), (43, 3, 45))
        self.assertEqual(len(layers) * Q.N_EXPERTS, 12_384)

    def test_layer_spec(self):
        self.assertEqual(Q.parse_layers("0-3"), [0, 1, 2, 3])
        self.assertEqual(Q.expert_layers(Q.parse_layers("0-3")), [3])
        self.assertEqual(Q.parse_layers("45,3"), [3, 45])
        for bad in ("3-1", "x", "46", "0-50"):
            with self.assertRaises(Q.Refusal, msg=bad):
                Q.parse_layers(bad)

    def test_the_held_out_file_never_calibrates(self):
        with self.assertRaises(Q.Refusal) as c:
            Q.check_calibration(["lenis-0830", "todo-1006"])
        self.assertIn("held-out", str(c.exception))
        with self.assertRaises(Q.Refusal):
            Q.check_calibration(["somefile"])
        Q.check_calibration(list(Q.CAL_NAMES))
        self.assertNotIn(Q.HELD, Q.CAL_NAMES)

    def test_no_work_or_store_inside_the_originals(self):
        fp8 = os.path.join(self.d, "fp8")
        os.makedirs(fp8)
        for inside in (fp8, os.path.join(fp8, "store"), os.path.join(fp8, "a", "b")):
            with self.assertRaises(Q.Refusal):
                Q.check_outside(fp8, os.path.join(self.d, "work"), inside)
        Q.check_outside(fp8, os.path.join(self.d, "work"), os.path.join(self.d, "fp8-store"))
        # the CLI refuses before it looks at the weights
        self.assertEqual(Q.main(["capture", "--fp8", fp8, "--work", os.path.join(fp8, "w")]), 2)
        self.assertEqual(Q.main(["quantize", "--fp8", fp8, "--work", os.path.join(self.d, "w"),
                                 "--store", os.path.join(fp8, "s")]), 2)
        self.assertEqual(os.listdir(fp8), [])

    def test_disk_check(self):
        with self.assertRaises(Q.Refusal) as c:
            Q.disk_check(self.d, 100 * Q.GIB, 16 * Q.GIB, "full", free=110 * Q.GIB)
        self.assertIn("100.0 GiB to write + 16.0 GiB reserve, 110.0 GiB free", str(c.exception))
        self.assertTrue(Q.disk_check(self.d, 100 * Q.GIB, 16 * Q.GIB, "full", free=116 * Q.GIB).endswith(": ok"))

    def test_a_torn_journal_line_is_cut(self):
        Q.append_journal(self.d, {"layer": 3, "expert": 0, "file": Q.rel_record(3, 0), "sha256": "a"})
        Q.append_journal(self.d, {"layer": 3, "expert": 1, "file": Q.rel_record(3, 1), "sha256": "b"})
        with open(os.path.join(self.d, "journal.jsonl"), "a") as f:
            f.write('{"layer": 3, "exp')
        self.assertEqual(sorted(Q.read_journal(self.d)), [(3, 0), (3, 1)])
        Q.append_journal(self.d, {"layer": 3, "expert": 2, "file": Q.rel_record(3, 2), "sha256": "c"})
        self.assertEqual(sorted(Q.read_journal(self.d)), [(3, 0), (3, 1), (3, 2)])

    def test_prune_deletes_only_records_the_named_container_holds(self):
        store, fp8 = os.path.join(self.d, "store"), os.path.join(self.d, "fp8")
        os.makedirs(os.path.join(store, "L03"))
        os.makedirs(fp8)
        Path(fp8, "model-00001-of-00062.safetensors").write_bytes(b"x" * 10)
        a, b = os.path.join(self.d, "a.cnq"), os.path.join(self.d, "b.cnq")
        for e, out in ((0, a), (1, b), (2, None)):
            Path(store, Q.rel_record(3, e)).write_bytes(b"r" * 100)
            if out:
                Path(store, Q.rel_record(3, e)[:-len(".safetensors")] + ".done").write_text(json.dumps({"out": out}))
        self.assertEqual(Q.prune_consumed(store, a, quiet), 100)
        self.assertEqual(sorted(os.listdir(os.path.join(store, "L03"))),
                         ["E000.done", "E001.done", "E001.safetensors", "E002.safetensors"])
        self.assertEqual(os.listdir(fp8), ["model-00001-of-00062.safetensors"])

    def test_plan_prints_both_commands(self):
        t = Q.plan_text(free=168 * Q.GIB)
        self.assertIn("131072 tokens", t)
        self.assertIn("held out: todo-1006", t)
        self.assertIn("43 x 288 = 12384", t)
        lines = [ln.strip() for ln in t.splitlines()]
        conv = [ln for ln in lines if "--experts-mul1" in ln]
        self.assertEqual(len(conv), 2)
        self.assertIn("--layers 0-3 --with-embed-head", conv[0])
        self.assertTrue(conv[0].endswith("converter/GLM-5.3-Flash-MUL1K3-L0-3.cnq"))
        self.assertIn("--mul1-wait", conv[1])
        self.assertIn("--scales mse --source-repo zai-org/GLM-5.3-Flash --revision " + Q.REVISION, conv[1])
        self.assertTrue(any("quantize" in ln and "--prune-consumed converter/GLM-5.3-Flash-MUL1K3.cnq" in ln for ln in lines))
        self.assertTrue(any("capture" in ln and ln.endswith("--layers 0-3") for ln in lines))
        self.assertEqual(t.count("-> ok"), 2)
        self.assertIn("REFUSED", Q.plan_text(free=100 * Q.GIB))


@unittest.skipUnless(has("torch", "transformers") and has("transformers.models.glm5_next"),
                     "capture needs the oracle venv (transformers 5.16.1)")
class Capture(unittest.TestCase):
    """the runner's synthetic small checkpoint: 8 layers, MoE from layer 3, 16 experts, hidden 256"""

    @classmethod
    def setUpClass(cls):
        import torch
        sys.path.insert(0, str(REPO / "oracle"))
        import glm5_common as G
        import glm5_layerwise as LW
        import transformers
        from transformers import Glm5NextForConditionalGeneration
        cls.G, cls.LW = G, LW
        cls.d = tempfile.mkdtemp(prefix="mul1-capture-")
        cls.ck = os.path.join(cls.d, "fp8")
        LW.make_synthetic("small", cls.ck)
        cls.ids = {"a": [7, 900, 3, 3, 42] * 8, "b": [5, 17, 1, 999] * 6}
        deq = os.path.join(cls.d, "deq")
        LW.write_hf_dequant_checkpoint(cls.ck, deq)
        transformers.logging.set_verbosity_error()
        hf = Glm5NextForConditionalGeneration.from_pretrained(deq, dtype=torch.float32, attn_implementation="eager",
                                                              experts_implementation="eager").eval()
        cls.ref = {}
        lm = hf.model.language_model
        for name, ids in cls.ids.items():
            rec = {l: [] for l in (3, 4)}
            route = {l: [] for l in (3, 4)}
            hooks = [lm.layers[l].mlp.register_forward_pre_hook(lambda m, a, l=l: rec[l].append(a[0][0].clone()))
                     for l in (3, 4)]
            hooks += [lm.layers[l].mlp.gate.register_forward_hook(lambda m, i, o, l=l: route[l].append(o[2].clone()))
                      for l in (3, 4)]
            with torch.no_grad():
                hf(input_ids=torch.tensor([ids]))
            for h in hooks:
                h.remove()
            cls.ref[name] = {l: (torch.cat(rec[l], 0), torch.cat(route[l], 0).sort(dim=1).values) for l in (3, 4)}
        del hf

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def run_capture(self, work, stop_after=None, state_dtype="f32"):
        ws = self.G.WeightSource("fp8", self.ck)
        tc = self.G.text_config_from_dict(ws.config_dict())
        return Q.capture_layers(ws, tc, self.ids, work, [0, 1, 2, 3, 4], prompt_chunk=16, state_dtype=state_dtype,
                                max_ahead=0, log=quiet, stop_after=stop_after)

    def load(self, work, l):
        import numpy as np
        import torch
        cap = Q.jload(os.path.join(work, "L%02d" % l, "capture.json"))
        x = np.fromfile(os.path.join(work, "L%02d" % l, "moe-in.bf16"), dtype="<i2").reshape(cap["rows"], cap["hidden"])
        ids = np.fromfile(os.path.join(work, "L%02d" % l, "ids.i32"), dtype="<i4").reshape(cap["rows"], cap["top_k"])
        return cap, torch.from_numpy(x.copy()).view(torch.bfloat16).float(), torch.from_numpy(ids.copy())

    def test_the_dump_is_hfs_moe_input(self):
        import torch
        work = os.path.join(self.d, "work")
        self.assertEqual(self.run_capture(work), [3, 4])
        for l in (3, 4):
            cap, x, ids = self.load(work, l)
            self.assertEqual(cap["rows"], 64)
            self.assertEqual([f["name"] for f in cap["files"]], ["a", "b"])
            r0 = 0
            for name in ("a", "b"):
                ref, route = self.ref[name][l]
                n = ref.shape[0]
                got = x[r0:r0 + n]
                # BF16 storage of an f32 value that agrees with HF to ~1e-5: one BF16 step at most
                tol = ref.abs() * 2 ** -7 + 1e-4
                self.assertTrue(bool(((got - ref).abs() <= tol).all()), "layer %d file %s: max |d| %.3g" % (
                    l, name, float((got - ref).abs().max())))
                self.assertTrue(torch.equal(ids[r0:r0 + n].long(), route.long()), "layer %d file %s: routing" % (l, name))
                r0 += n
        # the last layer's states stay for a later, longer run; the ones behind are gone
        self.assertEqual(sorted(os.listdir(os.path.join(work, "states", "a"))), ["l4-output.f32"])

    def test_an_interrupted_capture_resumes_to_the_same_files(self):
        w1, w2 = os.path.join(self.d, "w1"), os.path.join(self.d, "w2")
        self.run_capture(w1, state_dtype="bf16")
        self.assertEqual(self.run_capture(w2, stop_after=3, state_dtype="bf16"), [3])
        self.assertFalse(os.path.exists(os.path.join(w2, "L04")))
        self.assertEqual(self.run_capture(w2, state_dtype="bf16"), [4])
        for l in (3, 4):
            for f in ("moe-in.bf16", "ids.i32"):
                self.assertEqual(Q.sha256_file(os.path.join(w1, "L%02d" % l, f)),
                                 Q.sha256_file(os.path.join(w2, "L%02d" % l, f)), "L%02d %s" % (l, f))


# ------------------------------------------------------------------ quantize (GPU)

HID, INT_ = 128, 256   # the converter miniature's expert shapes (converter/src/main.rs tests)


def glm_config():
    lt = ["deepseek_sparse_attention" if i % 4 == 3 else "linear_attention" for i in range(45)]
    return {"text_config": {"model_type": "glm5_next_text", "num_hidden_layers": 45, "num_nextn_predict_layers": 1,
                            "hidden_size": HID, "vocab_size": 64, "layer_types": lt, "n_routed_experts": 2,
                            "num_experts_per_tok": 1, "moe_intermediate_size": INT_, "first_k_dense_replace": 3,
                            "num_attention_heads": 2, "num_key_value_heads": 2},
            "quantization_config": {"quant_method": "fp8", "fmt": "e4m3", "weight_block_size": [128, 128]}}


def write_mini_checkpoint(d):
    """a GLM-5.3-Flash miniature the converter accepts: embed, lm_head, a layer-44 norm (45 text layers), the
    routed experts 0 and 1 of layer 3 and of the MTP layer 45 in FP8 with 128x128 block scales"""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(182)
    t = {"lm_head.weight": torch.randn(64, HID, generator=g).to(torch.bfloat16),
         Q.LM + "embed_tokens.weight": torch.randn(64, HID, generator=g).to(torch.bfloat16),
         Q.LM + "layers.44.post_attention_layernorm.weight": torch.ones(HID, dtype=torch.bfloat16)}
    for l in (3, 45):
        for e in (0, 1):
            for p, shp in (("gate", (INT_, HID)), ("up", (INT_, HID)), ("down", (HID, INT_))):
                n = "%slayers.%d.mlp.experts.%d.%s_proj.weight" % (Q.LM, l, e, p)
                t[n] = (torch.randn(*shp, generator=g) * 2).to(torch.float8_e4m3fn)
                t[n + "_scale_inv"] = torch.full((math.ceil(shp[0] / 128), math.ceil(shp[1] / 128)), 0.01)
    os.makedirs(d, exist_ok=True)
    shard = "model-00001-of-00001.safetensors"
    save_file(t, os.path.join(d, shard))
    Path(d, "model.safetensors.index.json").write_text(json.dumps({"weight_map": {n: shard for n in t}}))
    Path(d, "config.json").write_text(json.dumps(glm_config()))
    Path(d, "generation_config.json").write_text("{}")
    size = os.path.getsize(os.path.join(d, shard))
    Path(d, "hf-revision.json").write_text(json.dumps({"sha": "mul1-synth", "siblings": [
        {"rfilename": shard, "size": size, "lfs": {"sha256": Q.sha256_file(os.path.join(d, shard)), "size": size}}]}))


def write_capture(work, l, rows=600):
    import numpy as np
    import torch
    g = torch.Generator().manual_seed(l)
    x = torch.randn(rows, HID, generator=g).to(torch.bfloat16)
    ids = torch.randint(0, 2, (rows, 1), generator=g, dtype=torch.int32)
    ld = os.path.join(work, "L%02d" % l)
    os.makedirs(ld, exist_ok=True)
    x.view(torch.int16).numpy().tofile(os.path.join(ld, "moe-in.bf16"))
    ids.numpy().tofile(os.path.join(ld, "ids.i32"))
    Q.write_json_atomic(os.path.join(ld, "capture.json"), {
        "layer": l, "rows": rows, "hidden": HID, "top_k": 1, "files": [{"name": "synthetic", "rows": rows}],
        "moe_in_sha256": Q.sha256_file(os.path.join(ld, "moe-in.bf16")),
        "ids_sha256": Q.sha256_file(os.path.join(ld, "ids.i32")), "identity": None})


@unittest.skipUnless(cuda(), "quantize needs the exllamav3 venv and a CUDA GPU")
class Quantize(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "oracle"))
        import glm5_common as G
        cls.G = G
        cls.d = tempfile.mkdtemp(prefix="mul1-quant-")
        cls.ck = os.path.join(cls.d, "fp8")
        write_mini_checkpoint(cls.ck)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def fresh(self, tag, hessian="all"):
        work, store = os.path.join(self.d, tag, "work"), os.path.join(self.d, tag, "store")
        write_capture(work, 3)
        Q.open_store(store, Q.store_head(HID, INT_, 2, {"files": ["synthetic"], "tokens": 600}, {"repo": "t"}, hessian))
        return work, store

    def run_q(self, work, store, hessian="all", **kw):
        return Q.quantize_layers(self.G.WeightSource("fp8", self.ck), work, store, [3, 45], hessian, log=quiet,
                                 free=1 << 50, **kw)

    def test_store_files_journal_resume_and_mtp(self):
        import torch
        from safetensors import safe_open
        work, store = self.fresh("main")
        self.assertEqual(self.run_q(work, store, stop_after=1), 1)   # a kill after the first expert
        self.assertEqual(self.run_q(work, store), 3)
        self.assertEqual(self.run_q(work, store), 0)                 # everything journalled: nothing again
        j = Q.read_journal(store)
        self.assertEqual(sorted(j), [(3, 0), (3, 1), (45, 0), (45, 1)])
        with open(os.path.join(store, "journal.jsonl")) as f:
            self.assertEqual(len(f.read().splitlines()), 4, "no duplicate line after the resume")
        for (l, e), line in j.items():
            p = os.path.join(store, line["file"])
            self.assertEqual(Q.sha256_file(p), line["sha256"])
            self.assertEqual(line["hessian"], "identity" if l == 45 else "all rows")
            self.assertTrue(all(math.isfinite(x) for x in line["proxy_err"]))
            with safe_open(p, "pt") as f:
                self.assertEqual(len(list(f.keys())), 9)
                for proj, (k, n) in (("gate", (HID, INT_)), ("up", (HID, INT_)), ("down", (INT_, HID))):
                    tr = f.get_tensor(proj + ".trellis")
                    self.assertEqual((tr.dtype, tuple(tr.shape)), (torch.int16, (k // 16, n // 16, 48)))
                    self.assertEqual((f.get_tensor(proj + ".suh").dtype, f.get_tensor(proj + ".suh").numel()), (torch.half, k))
                    self.assertEqual((f.get_tensor(proj + ".svh").dtype, f.get_tensor(proj + ".svh").numel()), (torch.half, n))
        self.assertTrue(os.path.exists(os.path.join(work, "L03", "quantized.json")))
        self.assertFalse(os.path.exists(os.path.join(work, "L03", "moe-in.bf16")), "the capture is derived and goes")
        # the converter reads the store: one 4096-aligned MUL1 record per expert
        conv = REPO / "converter" / "target" / "release" / ("converter.exe" if os.name == "nt" else "converter")
        if not conv.exists():
            self.skipTest("converter not built (cd converter && cargo build --release)")
        out = os.path.join(self.d, "main", "mini.cnq")
        r = subprocess.run([str(conv), "--scales", "mse", "--source-repo", "crow-nest/mul1-synth", "--experts-mul1", store,
                            "--disk-reserve-gib", "0", self.ck, out], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        b = Path(out).read_bytes()
        n = struct.unpack("<Q", b[-8:])[0]
        idx = json.loads(b[-8 - n:-8])
        self.assertEqual(idx["expert_codec"]["records"], 4)
        self.assertEqual(idx["expert_codec"]["record_bytes"], Q.record_bytes(HID, INT_))
        gates = [t for t in idx["tensors"] if t["dtype"] == "mul1" and t["name"].endswith("gate_proj.weight")]
        self.assertEqual(len(gates), 4)
        self.assertTrue(all((12 + t["offset"]) % 4096 == 0 for t in gates))
        held = sum(os.path.getsize(os.path.join(store, v["file"])) for v in j.values())
        self.assertEqual(Q.prune_consumed(store, out, quiet), held)
        self.assertFalse(any(os.path.exists(os.path.join(store, v["file"])) for v in j.values()))

    def test_routed_rows(self):
        work, store = self.fresh("routed", "routed")
        self.assertEqual(self.run_q(work, store, "routed"), 4)
        rows = {k: v["rows"] for k, v in Q.read_journal(store).items() if k[0] == 3}
        self.assertEqual(rows[(3, 0)] + rows[(3, 1)], 600, "top-1 routing: every row goes to one of the two experts")


if __name__ == "__main__":
    unittest.main(verbosity=2)
