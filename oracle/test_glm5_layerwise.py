"""crow-nest #158: tests of the layerwise glm5_next reference runner (CPU, ~1 min).

Run: .venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v

- the proof (small shapes): runner == HF's full Glm5NextForConditionalGeneration to 1e-5 per layer,
  routing ids, DSA top-k and logits, at index_topk 32 and 2048, prompt and decode rows;
- the proof can go red: a collapsed mHC hand-over and decode rows without the layer cache both fail it;
- FP8 unpack == HF's Fp8Dequantize on whole blocks and one-block partial tensors;
- the CLI: --weights fp8-originals with a layer split (0:3 then 3:8) writes the same files as one run;
  --state-dtype bf16; --weights container exits 2 with a clear error;
- crow-nest #147: the prompt in calls of C rows routes as one call (vs HF and vs the one-call run;
  differing DSA rows only at recorded exact ties); a DSA cache slot that drops its history fails that
  proof; the DSA slot is allocated once (HF's DynamicIndexedLayer grows per call); --delete-states-behind
  keeps the routing and resumes; --layers L: computes only the logits.
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


class Chunked(unittest.TestCase):
    """crow-nest #147: the prompt in calls of C rows against the layer cache routes every row as one call does"""

    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="glm5-chunk-")
        cls.ck = os.path.join(cls.wd, "fp8")
        _silent(LW.make_synthetic, "small", cls.ck)
        cls.tc = G.text_config(cls.ck)
        cls.ws = G.WeightSource("fp8", cls.ck)
        g = torch.Generator().manual_seed(7)
        cls.ids = torch.randint(1, 1000, (150,), generator=g).tolist()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def _run(self, name, chunk, **kw):
        out = os.path.join(self.wd, name)
        man = LW.run_layerwise(self.ws, self.tc, self.ids, 2, out, anchors=[147, 148, 149], log=quiet,
                               prompt_chunk=chunk, **kw)
        return out, man

    def test_chunked_prompt_matches_hf_and_the_one_call_run(self):
        ok, tables = _silent(LW.selftest, "small", prompt_chunk=40, log=quiet)
        self.assertTrue(ok, tables)
        for k in (32, 2048):
            for label in (f"{k} chunked vs HF", f"{k} chunked vs unchunked"):
                layers = [r for r in tables[label] if r["layer"] != "logits"]
                self.assertEqual(len(layers), 8)
                for r in layers:
                    self.assertLessEqual(r["max_abs_prompt"], LW.TOL, (label, r))
                    self.assertEqual(r.get("route_ids_mismatch_tokens", 0), 0, (label, r))
                    self.assertEqual(r.get("dsa_topk_mismatch_rows", 0), 0, (label, r))

    def test_every_chunk_size_routes_as_one_call(self):
        """1 (every prompt row its own call), 7 (KDA's 64-row blocks cut mid-block), 40, 64 (aligned), 100.
        Up to the first DSA layer where a selected set differs, every layer is within TOL with identical
        routing ids; the rows where the set differs are rows whose selection boundary is an exact tie
        (torch.topk's tie order depends on the call shape; the small config's 4 indexer heads with ReLU
        give all-zero pool scores). Measured 2026-10-08: 40 and 100 identical everywhere; 1, 7, 64 differ
        on one row of layer 7, a recorded tie (scores 0.0 = 0.0 at the boundary)."""
        one, m1 = self._run("one", 0)
        ref = LW.ref_from_run(one, m1, self.tc)
        identical = []
        for c in (1, 7, 40, 64, 100):
            out, man = self._run(f"c{c}", c)
            self.assertEqual(len(man["per_layer"][0]["prompt_call_s"]), math.ceil(148 / c))
            rows, ok = LW.compare(ref, out, man, self.tc, 148, raw_layout=False)
            if ok:
                identical.append(c)
                continue
            flip = next(r["layer"] for r in rows if r.get("dsa_topk_mismatch_rows", 0) != 0)
            for r in rows[:flip]:
                self.assertTrue(r["ok"], (c, r))
            self.assertEqual(rows[flip].get("route_ids_mismatch_tokens", 0), 0, c)
            a = LW.canon_topk(LW.load_file(one, m1, f"l{flip}-dsa-topk.i32"))
            b = LW.canon_topk(LW.load_file(out, man, f"l{flip}-dsa-topk.i32"))
            differ = set((a != b).any(1).nonzero().flatten().tolist())
            ties = set()
            for m in (m1, man):
                ties |= set(next(p for p in m["per_layer"] if p["layer"] == flip)["dsa_tie_rows"])
            self.assertTrue(differ and differ <= ties, (c, flip, differ, ties))
        self.assertIn(40, identical)
        self.assertIn(100, identical)

    def test_a_cache_that_drops_the_history_fails_the_chunk_proof(self):
        def no_history(self, key_states, value_states, *a, **k):  # every call starts at row 0 again
            self.n_kv = 0
            return real(self, key_states, value_states, *a, **k)

        real = LW._AppendIndexedLayer.update
        with mock.patch.object(LW._AppendIndexedLayer, "update", no_history):
            ok, tables = _silent(LW.selftest, "small", topks=(32,), prompt_chunk=40, log=quiet)
        self.assertFalse(ok)
        rows = tables["32 chunked vs unchunked"]
        self.assertTrue(all(r["ok"] for r in rows[:3]))  # KDA layers 0-2 do not use the DSA slot
        self.assertGreater(rows[3]["max_abs_prompt"], 1e-3)  # the first DSA layer loses rows 0..39

    def _dsa_layer_calls(self, chunk):
        """run DSA layer 3 alone with the prompt in calls of `chunk`; returns (y, the byte size of the
        storage behind the K the cache hands back, per call)"""
        from transformers.cache_utils import Cache
        from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer
        layer = G.build_meta(Glm5NextTextDecoderLayer, self.tc, 3)
        self.ws.load(layer, f"{G.LM}layers.3.")
        x = torch.randn(150, 4, self.tc.hidden_size, generator=torch.Generator().manual_seed(3))
        ptrs, real = [], Cache.update

        def spy(cache, k, v, layer_idx, *a, **kw):
            out = real(cache, k, v, layer_idx, *a, **kw)
            ptrs.append(out[0].untyped_storage().nbytes())
            return out

        with mock.patch.object(Cache, "update", spy):
            y, _, _ = LW.run_layer(layer, self.tc, 3, x, 148, chunk)
        return y, ptrs

    def test_the_dsa_cache_is_allocated_once(self):
        y, sizes = self._dsa_layer_calls(16)
        self.assertEqual(len(sizes), 10 + 2)  # 148 rows in calls of 16, then 2 decode rows
        full = 150 * self.tc.num_attention_heads * (self.tc.qk_nope_head_dim + self.tc.qk_rope_head_dim) * 4
        self.assertEqual(set(sizes), {full})  # one buffer of all 150 rows from the first call: no torch.cat
        with mock.patch.object(LW, "append_in_place", lambda cache, tc, n: cache):  # HF's DynamicIndexedLayer
            y_cat, sizes_cat = self._dsa_layer_calls(16)
        self.assertEqual(len(set(sizes_cat)), 12)  # a new, longer K tensor per call
        self.assertLessEqual(float((y - y_cat).abs().max()), LW.TOL)


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

    def test_delete_states_behind_keeps_routing_and_resumes(self):
        keep, dele = os.path.join(self.wd, "keep"), os.path.join(self.wd, "del")
        self.assertEqual(self._run("--out", keep, "--prompt-chunk", "4"), 0)
        self.assertEqual(self._run("--out", dele, "--prompt-chunk", "4", "--delete-states-behind",
                                   "--layers", "0:3"), 0)
        states = sorted(f for f in os.listdir(dele) if "-output." in f)
        self.assertEqual(states, ["l2-output.f32"])  # the one a resume needs
        self.assertEqual(self._run("--out", dele, "--prompt-chunk", "4", "--delete-states-behind",
                                   "--layers", "3:"), 0)
        self.assertEqual([f for f in os.listdir(dele) if "-output." in f], [])
        with open(os.path.join(keep, "manifest.json")) as f:
            a = json.load(f)
        with open(os.path.join(dele, "manifest.json")) as f:
            b = json.load(f)
        self.assertEqual(b["deleted_states"], [f"l{k}-output.f32" for k in range(8)])
        self.assertTrue(b["complete"] and b["delete_states_behind"])
        self.assertEqual(b["prompt_chunk"], 4)
        self.assertEqual(set(b["files"]), {n for n in a["files"] if "-output." not in n})
        for n in b["files"]:
            self.assertEqual(a["files"][n]["sha256"], b["files"][n]["sha256"], n)
            self.assertTrue(os.path.exists(os.path.join(dele, n)), n)

    def test_logits_only_resume(self):
        # a pass that stopped after its last layer, before the logits: --layers 8: computes only them
        full, cut = os.path.join(self.wd, "full"), os.path.join(self.wd, "cut")
        self.assertEqual(self._run("--out", full), 0)
        shutil.copytree(full, cut)
        with open(os.path.join(cut, "manifest.json")) as f:
            m = json.load(f)
        for n in [n for n in m["files"] if n.startswith("logits-")]:
            os.remove(os.path.join(cut, n))
            del m["files"][n]
        m["complete"] = False
        with open(os.path.join(cut, "manifest.json"), "w") as f:
            json.dump(m, f)
        self.assertEqual(self._run("--out", cut, "--layers", "8:"), 0)
        with open(os.path.join(full, "manifest.json")) as f:
            a = json.load(f)
        with open(os.path.join(cut, "manifest.json")) as f:
            b = json.load(f)
        self.assertTrue(b["complete"])
        self.assertEqual(b["layers"], [0, 8])
        self.assertEqual({n: v["sha256"] for n, v in a["files"].items()},
                         {n: v["sha256"] for n, v in b["files"].items()})

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


class Subblocks(unittest.TestCase):
    """crow-nest #156: --capture-subblocks writes the mHC sub-block files the golden harness of #161 reads
    (attn_hc / ffn_hc input, post, comb, collapsed, the sub-layer output; expanded = the next site's input
    and the layer output), and they are HF's full model's values"""

    @classmethod
    def setUpClass(cls):
        from transformers import Glm5NextForConditionalGeneration
        cls.wd = tempfile.mkdtemp(prefix="glm5-sub-")
        cls.ck, deq = os.path.join(cls.wd, "fp8"), os.path.join(cls.wd, "hf-deq")
        _silent(LW.make_synthetic, "small", cls.ck)
        _silent(LW.write_hf_dequant_checkpoint, cls.ck, deq)
        cls.tc = G.text_config(cls.ck)
        cls.ws = G.WeightSource("fp8", cls.ck)
        cls.T, cls.D = 40, 3
        g = torch.Generator().manual_seed(11)
        cls.ids = torch.randint(1, 1000, (cls.T + cls.D,), generator=g).tolist()
        verbosity = LW.transformers.logging.get_verbosity()
        LW.transformers.logging.set_verbosity_error()  # the load report lists the vision tower (not written)
        try:
            hf = _silent(Glm5NextForConditionalGeneration.from_pretrained, deq, dtype=torch.float32,
                         attn_implementation="eager", experts_implementation="eager").eval()
        finally:
            LW.transformers.logging.set_verbosity(verbosity)
        cls.hf = {}  # HF's full model, prompt in one call then one call per decode row (the runner's plan)
        hooks = []
        for l, layer in enumerate(hf.model.language_model.layers):
            cls.hf[l] = {"y": []}
            hooks += LW.subblock_hooks(layer, cls.hf[l])
            hooks.append(layer.register_forward_hook(
                lambda m, i, o, l=l: cls.hf[l]["y"].append(o[0][0].detach().clone())))
        x = torch.tensor(cls.ids)[None]
        cache = LW.DynamicCache(config=hf.config)
        with torch.no_grad():
            for r0, r1 in [(0, cls.T)] + [(r, r + 1) for r in range(cls.T, cls.T + cls.D)]:
                hf(input_ids=x[:, r0:r1], past_key_values=cache, use_cache=True)
        for h in hooks:
            h.remove()
        del hf, cache
        for l in cls.hf:
            cls.hf[l] = {k: torch.cat(v, 0) for k, v in cls.hf[l].items()}
        cls.ids_path = os.path.join(cls.wd, "ids.json")
        with open(cls.ids_path, "w") as f:
            json.dump(cls.ids, f)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def _cli(self, out, *args):
        return _silent(LW.main, ["run", "--weights", "fp8-originals", self.ck, "--ids", self.ids_path,
                                 "--decode", str(self.D), "--out", out, *args])

    def test_captures_are_hf_full_model_values(self):
        out = os.path.join(self.wd, "cap")
        self.assertEqual(self._cli(out, "--capture-subblocks", "--prompt-chunk", "0"), 0)
        with open(os.path.join(out, "manifest.json")) as f:
            man = json.load(f)
        N, hc, H = self.T + self.D, self.tc.hc_mult, self.tc.hidden_size
        shapes = {"in": (N, hc, H), "post": (N, hc), "comb": (N, hc, hc), "collapsed": (N, H), "out": (N, H),
                  "expanded": (N, hc, H)}
        self.assertEqual(sorted(man["subblocks"], key=int), [str(l) for l in range(8)])
        for l in range(8):
            for site in ("attn", "ffn"):
                roles = man["subblocks"][str(l)][site]
                self.assertEqual(set(roles), set(shapes))
                for role, nm in roles.items():
                    self.assertEqual(tuple(man["files"][nm]["shape"]), shapes[role], nm)
                    got = LW.load_file(out, man, nm)
                    if (site, role) == ("ffn", "expanded"):
                        ref = self.hf[l]["y"]  # the decoder layer's output
                    elif (site, role) == ("attn", "expanded"):
                        ref = self.hf[l]["ffn.in"]
                    else:
                        ref = self.hf[l][f"{site}.{role}"]
                    self.assertEqual(tuple(ref.shape), shapes[role], (l, site, role))
                    self.assertLessEqual(float((got - ref).abs().max()), LW.TOL, (l, site, role))
                    self.assertGreater(float(ref.abs().max()), 0.0, (l, site, role))

    def test_captures_are_consistent_and_change_nothing(self):
        cap, plain = os.path.join(self.wd, "cap7"), os.path.join(self.wd, "plain7")
        self.assertEqual(self._cli(cap, "--capture-subblocks", "--prompt-chunk", "7"), 0)
        self.assertEqual(self._cli(plain, "--prompt-chunk", "7"), 0)
        with open(os.path.join(cap, "manifest.json")) as f:
            a = json.load(f)
        with open(os.path.join(plain, "manifest.json")) as f:
            b = json.load(f)
        self.assertNotIn("subblocks", b)
        for n, v in b["files"].items():  # the hooks only read: every other file is byte-identical
            self.assertEqual(a["files"][n]["sha256"], v["sha256"], n)
        self.assertEqual(len(a["files"]) - len(b["files"]), 8 * 10)
        prev = LW.load_file(cap, a, "embed.f32").unsqueeze(1).expand(-1, self.tc.hc_mult, -1)
        for l in range(8):
            r = {s: {k: LW.load_file(cap, a, nm) for k, nm in a["subblocks"][str(l)][s].items()}
                 for s in ("attn", "ffn")}
            self.assertTrue(torch.equal(r["attn"]["in"], prev), l)  # the layer's input, rows in call order
            for s in ("attn", "ffn"):
                x, p, c, y = r[s]["in"], r[s]["post"], r[s]["comb"], r[s]["out"]
                expand = p.unsqueeze(-1) * y.unsqueeze(-2) + torch.matmul(c.transpose(-1, -2), x)
                self.assertLessEqual(float((expand - r[s]["expanded"]).abs().max()), LW.TOL, (l, s))
            prev = LW.load_file(cap, a, f"l{l}-output.f32")

    def test_capture_refuses_states_it_cannot_point_at(self):
        for extra in (("--state-dtype", "bf16"), ("--delete-states-behind",)):
            with self.assertRaises(SystemExit) as e:
                self._cli(os.path.join(self.wd, "bad"), "--capture-subblocks", *extra)
            self.assertEqual(e.exception.code, 2)
        with self.assertRaises(ValueError):
            LW.run_layerwise(self.ws, self.tc, self.ids, self.D, os.path.join(self.wd, "bad"), state_dtype="bf16",
                             log=quiet, capture_subblocks=True)


if __name__ == "__main__":
    unittest.main()
