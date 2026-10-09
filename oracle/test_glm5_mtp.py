"""crow-nest #182: tests of the MTP (NextN) reference of glm5_next (CPU, ~30 s).

Run: .venv-oracle/Scripts/python.exe -I -m unittest discover -s oracle -p "test_glm5_*.py" -v

- the proof (small shapes, a synthetic checkpoint with an MTP layer in the original naming and FP8
  format, the trunk from HF's full model): the oracle (HF's attention and MoE blocks, the prompt in
  calls of 7 rows, decode rows one by one against the block's cache) equals a plain-torch formula
  (`manual_mtp`, one call, no HF module) at every row;
- the proof can go red: the paper's [h; e] order, the trunk state before the final norm, and a block
  cache that forgets its history each break it;
- the block's names are layer 45 of the real index (minus the scale companions), and a checkpoint
  without the layer is refused;
- the variants differ exactly where the stacks differ (row 0, the lead row);
- a sparse DSA selection (index_topk 8) routes the same in one call and in decode-shaped calls;
- MixedSource takes the experts from the primary and the rest from the fallback, value for value;
- the CLI refuses a trunk dir without --capture-head.
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
from unittest import mock

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import glm5_common as G  # noqa: E402
import glm5_layerwise as LW  # noqa: E402
import glm5_mtp as MTP  # noqa: E402


def quiet(*_):
    pass


def originals_dir():
    """G.MODEL_DIR, or the same path in the main checkout when this runs in a git worktree"""
    cands = [G.MODEL_DIR]
    try:
        common = subprocess.run(["git", "rev-parse", "--git-common-dir"], cwd=G.ROOT, capture_output=True,
                                text=True, check=True).stdout.strip()
        cands.append(os.path.join(os.path.dirname(os.path.abspath(os.path.join(G.ROOT, common))), "models",
                                  "GLM-5.3-Flash-original"))
    except (OSError, subprocess.CalledProcessError):
        pass
    return next((d for d in cands if os.path.exists(os.path.join(d, "model.safetensors.index.json"))), None)


class Synth:
    """one synthetic checkpoint + trunk per class"""
    index_topk = 2048

    @classmethod
    def setUpClass(cls):
        cls.wd = tempfile.mkdtemp(prefix="glm5-mtp-test-")
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            cls.fp8, cls.trunk, cls.ids = MTP.make_synthetic_mtp(cls.wd, "small", cls.index_topk)
        cls.ws = G.WeightSource("fp8", cls.fp8)
        cls.tc = G.text_config_from_dict(cls.ws.config_dict())
        cls.tman, _, cls.T, cls.D, cls.hn, cls.hm, cls.trunk_top1 = MTP.read_trunk(cls.trunk)
        cls.block, cls.tc_m = MTP.load_block(cls.ws, cls.tc)
        cls.emb = cls.ws.embed(cls.ids)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.wd, ignore_errors=True)

    def forward(self, variant="sglang", chunk=None, h=None):
        v = MTP.VARIANTS[variant]
        e, hh, zero, lead = MTP.pair_rows(self.ids, self.hn if h is None else h, self.hm, self.emb, variant)
        return MTP.mtp_forward(self.block, self.tc_m, e, hh, self.T - 1 + lead, chunk, v["order"], zero), lead


class Proof(Synth, unittest.TestCase):
    def test_oracle_equals_the_plain_formula(self):
        out = os.path.join(self.wd, "golden")
        man = MTP.run_golden(self.ws, self.trunk, out, prompt_chunk=7, log=quiet)
        ref_hn, ref_lg = MTP.manual_mtp(self.ws, self.tc, self.ids, self.hn)
        hn = LW.load_file(out, man, "mtp-head-norm.f32")
        lg = LW.load_file(out, man, "mtp-logits.f32")
        R = len(self.ids) - 1
        self.assertEqual(tuple(hn.shape), (R, self.tc.hidden_size))
        self.assertLessEqual(float((hn - ref_hn).abs().max()), MTP.TOL)
        self.assertLessEqual(float((lg - ref_lg).abs().max()), 10 * MTP.TOL)
        self.assertGreater(float(ref_hn.pow(2).mean().sqrt()), 0.5)  # a real signal, not zeros
        # the draft picks and the trunk's next picks are what the manifest says
        top1 = LW.load_file(out, man, "mtp-draft-top1.i32")
        self.assertTrue(torch.equal(top1, ref_lg.argmax(-1).to(torch.int32)))
        nxt = LW.load_file(out, man, "trunk-next-top1.i32")
        self.assertTrue(torch.equal(nxt, self.trunk_top1[1:]))
        self.assertEqual(man["variants"]["sglang"]["all"]["agree"], int((top1 == nxt).sum()))
        self.assertEqual(set(man["variants"]), set(MTP.VARIANTS))
        self.assertTrue(man["index_share_for_mtp_iteration"])
        # inputs recorded: row i = (embed(ids[i+1]), head-norm row i)
        self.assertTrue(torch.equal(LW.load_file(out, man, "mtp-embed.f32"), self.emb[1:]))
        self.assertTrue(torch.equal(LW.load_file(out, man, "mtp-h.f32"), self.hn[:-1]))

    def test_the_proof_can_go_red(self):
        o, _ = self.forward(chunk=7)
        for order, h in (("he", self.hn), ("eh", self.hm)):  # the paper's order; the pre-norm trunk state
            ref_hn, _ = MTP.manual_mtp(self.ws, self.tc, self.ids, h, order=order)
            self.assertGreater(float((o["head_norm"] - ref_hn).abs().max()), 1e-2, order)

    def test_a_cache_that_forgets_fails(self):
        ok, _ = self.forward(chunk=7)
        real = LW._AppendIndexedLayer.update

        def forget(layer, k, v, *a, **kw):  # keeps only the call's own rows
            layer.n_kv = 0
            return real(layer, k, v, *a, **kw)

        with mock.patch.object(LW._AppendIndexedLayer, "update", forget):
            bad, _ = self.forward(chunk=7)
        T = self.T - 1
        self.assertEqual(float((bad["head_norm"][:7] - ok["head_norm"][:7]).abs().max()), 0.0)  # first call
        self.assertGreater(float((bad["head_norm"][T:] - ok["head_norm"][T:]).abs().max()), 1e-3)

    def test_chunked_equals_one_call(self):
        a, _ = self.forward(chunk=None)
        b, _ = self.forward(chunk=1)
        self.assertLessEqual(float((a["head_norm"] - b["head_norm"]).abs().max()), LW.CHUNK_TOL)
        self.assertTrue(torch.equal(a["routing"][0], b["routing"][0]))

    def test_variants_differ_where_the_stacks_differ(self):
        base, _ = self.forward("sglang")
        vl, _ = self.forward("vllm")
        R = len(self.ids) - 1
        # vLLM zeroes row 0's embedding only: eh differs on row 0 alone, later rows through attention
        d = (vl["eh"] - base["eh"]).abs().amax(-1)
        self.assertGreater(float(d[0]), 1e-3)
        self.assertEqual(float(d[1:].max()), 0.0)
        self.assertGreater(float((vl["head_norm"][1:] - base["head_norm"][1:]).abs().max()), 1e-6)
        lc, lead = self.forward("llamacpp")
        self.assertEqual((lead, lc["head_norm"].shape[0]), (1, R + 1))
        self.assertTrue(torch.equal(lc["eh"][1:], base["eh"]))  # same pairs, one row later
        self.assertGreater(float((lc["head_norm"][1:] - base["head_norm"]).abs().max()), 1e-6)


class Names(unittest.TestCase):
    def test_block_names_are_layer_45_of_the_real_index(self):
        md = originals_dir()
        if md is None:
            self.skipTest("the FP8 originals are not on this machine")
        idx = os.path.join(md, "model.safetensors.index.json")
        with open(os.path.join(md, "config.json")) as f:
            tc = G.text_config_from_dict(json.load(f))
        tc_m, L = MTP.mtp_text_config(tc)
        self.assertEqual(L, 45)
        block = G.build_meta(MTP.Glm5MtpBlock, tc_m, L)
        names = {n for _, _, _, ns in G.load_plan(block, f"{G.LM}layers.{L}.") for n in ns}
        with open(idx) as f:
            wm = json.load(f)["weight_map"]
        real = {n for n in wm if n.startswith(f"{G.LM}layers.{L}.") and not n.endswith("_scale_inv")}
        self.assertEqual(names, real)
        self.assertEqual(len(real), 25 + 3 * 288)  # 25 non-expert tensors (7 FP8 ones carry a scale)

    def test_a_checkpoint_without_the_layer_is_refused(self):
        wd = tempfile.mkdtemp(prefix="glm5-mtp-nolayer-")
        try:
            d = os.path.join(wd, "fp8")
            with redirect_stdout(io.StringIO()):
                LW.make_synthetic("small", d)
            ws = G.WeightSource("fp8", d)
            with self.assertRaises(AssertionError):
                MTP.load_block(ws, G.text_config_from_dict(ws.config_dict()))
        finally:
            shutil.rmtree(wd, ignore_errors=True)


class Sparse(Synth, unittest.TestCase):
    index_topk = 8  # 2 pools + tail: the indexer drops rows from position 11 on

    def test_sparse_selection_routes_as_one_call(self):
        a, _ = self.forward(chunk=None)
        b, _ = self.forward(chunk=5)
        sel = (LW.canon_topk(a["dsa_topk"]) >= 0).sum(1)
        self.assertLess(int(sel[-1]), len(self.ids) - 1)  # the last row really drops rows
        self.assertTrue(torch.equal(LW.canon_topk(a["dsa_topk"]), LW.canon_topk(b["dsa_topk"])))
        self.assertLessEqual(float((a["head_norm"] - b["head_norm"]).abs().max()), LW.CHUNK_TOL)


class Mixed(Synth, unittest.TestCase):
    def test_experts_from_primary_rest_from_fallback(self):
        class Only(G.WeightSource):  # the container's view: the routed experts only
            def has(self, name):
                return ".mlp.experts." in name and super().has(name)

        ms = MTP.MixedSource(Only("fp8", self.fp8), G.WeightSource("fp8", self.fp8))
        block, _ = MTP.load_block(ms, self.tc)
        for (k, a), (_, b) in zip(block.state_dict().items(), self.block.state_dict().items()):
            self.assertTrue(torch.equal(a, b), k)
        p = ms.provenance()
        E = self.tc.n_routed_experts
        self.assertEqual(p["from_primary"], 3 * E)
        self.assertTrue(all(".mlp.experts." not in n for n in p["fallback_names"]))
        self.assertIn(f"{G.LM}layers.{self.tc.num_hidden_layers}.eh_proj.weight", p["fallback_names"])


class Cli(unittest.TestCase):
    def test_a_trunk_without_the_head_is_refused(self):
        wd = tempfile.mkdtemp(prefix="glm5-mtp-cli-")
        try:
            with open(os.path.join(wd, "manifest.json"), "w") as f:
                json.dump({"complete": True, "ids": [1, 2], "T": 2, "D": 0, "files": {}}, f)
            with self.assertRaises(SystemExit) as cm:
                MTP.read_trunk(wd)
            self.assertIn("--capture-head", str(cm.exception))
        finally:
            shutil.rmtree(wd, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
