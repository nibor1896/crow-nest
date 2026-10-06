#!/usr/bin/env python3
"""#138: the f32 oracles' PLE norms against the model definition, no checkpoint needed.

transformers builds the PLE layer's norm_key / norm_query / norm_conv as
`Qwen4ExpTextRMSNorm(hc_hidden_size, group_size=hidden_size)`: one RMS per 2,560-value
hyper-connection stream. Each oracle script carries its own hand-ported `PleRef.rms`. The
scripts load the checkpoint at import, so the method is read out of the source with `ast` and
run on its own, against transformers' class, on inputs whose four streams have different scales
(the case where one RMS over all 10,240 values and four per-stream RMS differ).

  .venv-oracle/Scripts/python tools/test_oracle_ple_norm.py     # Windows
  .venv-oracle/bin/python tools/test_oracle_ple_norm.py         # Linux
"""
import ast
import os
import types
import unittest

import torch
from transformers.models.qwen4_exp.modeling_qwen4_exp import Qwen4ExpTextRMSNorm

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["oracle/ref_engine_logits.py", "oracle/ref_image_prompt_logits.py", "oracle/ref_longctx_logits.py"]
H, HC = 2560, 4


def ple_rms(script):
    """`PleRef.rms` of `script`, compiled alone (the module body is never run)."""
    with open(os.path.join(REPO, script), encoding="utf-8") as fh:
        tree = ast.parse(fh.read())
    cls = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == "PleRef")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "rms")
    ns = {"torch": torch}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), script, "exec"), ns)
    return ns["rms"]


class PleNorm(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(138)
        # four streams at scales 0.01, 1, 10, 100: per-stream and whole-row RMS disagree
        scale = torch.tensor([0.01, 1.0, 10.0, 100.0]).repeat_interleave(H)
        self.x = torch.randn(3, 5, HC * H, generator=g) * scale
        self.w = torch.randn(HC * H, generator=g) * 0.1
        ref = Qwen4ExpTextRMSNorm(HC * H, group_size=H, eps=1e-6)
        with torch.no_grad():
            ref.weight.copy_(self.w)
            self.want = ref(self.x)

    def test_each_oracle_matches_transformers(self):
        for script in SCRIPTS:
            with self.subTest(script=script):
                got = ple_rms(script)(types.SimpleNamespace(hidden=H), self.x, self.w)
                err = (got - self.want).abs().max().item()
                self.assertLessEqual(err, 1e-5, "%s: max_abs %.3e against Qwen4ExpTextRMSNorm(group_size=%d)"
                                     % (script, err, H))


if __name__ == "__main__":
    unittest.main()
