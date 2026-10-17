#!/usr/bin/env python3
"""#240: tests of tools/prequal.py and tools/stage_ceiling.py. Stdlib unittest, no network, no GPU.

    python -I tools/test_prequal.py
    PREQUAL_CONVERTER=<converter binary> PREQUAL_STATES=<states binary> python -I tools/test_prequal.py

The model dirs are synthetic, laid out as the header helper writes them: headers/<shard>.json
({"shard", "size", "data_start", "header"}), a weight_map index, the engine fixture configs. GLM-5.3-Flash
from converter/tests/fixtures/glm53-flash-tensors.tsv (experts expanded from {a..b}, contiguous
data_offsets, data_start and size per shard from its `# shard` lines); the 27B from
converter/tests/fixtures/qwen3.8-27b-tensors.tsv (name, shape, dtype; one synthetic shard). The tests that
run the real converter / states binaries run only when PREQUAL_CONVERTER / PREQUAL_STATES name them.
"""
import contextlib
import importlib.util
import io
import json
import math
import os
import re
import shutil
import sys
import tempfile
import types
import unittest
from unittest import mock

TOOLS = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(TOOLS)


def _load(name, filename):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return mod


pq = _load("prequal", "prequal.py")
sc = _load("stage_ceiling", "stage_ceiling.py")

GLM_TSV = os.path.join(REPO, "converter", "tests", "fixtures", "glm53-flash-tensors.tsv")
Q27_TSV = os.path.join(REPO, "converter", "tests", "fixtures", "qwen3.8-27b-tensors.tsv")
GLM_CFG = os.path.join(REPO, "engine", "tests", "fixtures", "GLM-5.3-Flash")
Q27_CFG = os.path.join(REPO, "engine", "tests", "fixtures", "Qwen3.8-27B")
CONVERTER = os.environ.get("PREQUAL_CONVERTER")
STATES = os.environ.get("PREQUAL_STATES")
RANGE = re.compile(r"\{(\d+)\.\.(\d+)\}")

# converter plan --headers on the 62 real GLM-5.3-Flash headers (7b0fea8, 2026-10-17): the totals block, verbatim
PLAN_TOTALS = """\
totals    (GiB = 2^30 B, GB = 10^9 B; nvfp4 = 36 B per 64 values, plus one f32 global scale per tensor in the index, not counted)
  GPU, text weights NVFP4             175373070336 B   163.329 GiB
  GPU, text keeps (bf16/f32/i64)        1836742904 B     1.711 GiB  (without the token embedding)
  GPU, text subtotal                  177209813240 B   165.039 GiB
  host RAM, token embedding             1268776960 B     1.182 GiB
  routed experts (tiered)             171228266496 B   159.469 GiB
  dense resident part                   5981546744 B     5.571 GiB    5.982 GB  (text without routed experts and the token embedding)
  optional section mtp                           0 B     0.000 GiB
  optional section vit                           0 B     0.000 GiB
  container payload                   178478590200 B   166.221 GiB
  omitted, not written                1760 tensors,      7493399168 B source     6.979 GiB  (MTP block (layer 45) not converted in v1 (GLM-5.3-Flash plan: no vision tower and no MTP in v1; step 21))
  omitted, not written                 347 tensors,      1127254016 B source     1.050 GiB  (vision tower not converted in v1 (GLM-5.3-Flash plan: no vision tower and no MTP in v1))
  FP8 E4M3 weights read              36467 tensors with their weight_scale_inv (74956800 B of scales, read and not written); pairs across two shards: 0
  write units                        13342 (12096 aligned to 4096 B), alignment zeros 28412 B; most shards one unit reads: 1
  container without index trailer     178478618624 B   166.221 GiB   178.479 GB  (12 B head + payload + alignment zeros)
"""

# states --plan --expert-bytes 9474048 --expert-codec mul1, CROW_CONTEXT=200000, HOST_PINNED_CAP (7b0fea8, 2026-10-17)
STATES_OUT = """\
states --plan: C:/Users/robin/dev/crow-nest/models/GLM-5.3-Flash-original/config.json (38 constants verified against zai-org/GLM-5.3-Flash @ eb9eb208 config.json); no CUDA, no container read
--- glm5_next plan (#159): context 200000, three expert tiers ---
inputs
  dense part                 5981546744 B     5.57 GiB (converter dry run, GLM measurement book)
  expert block                  9474048 B     9.04 MiB  one expert of one MoE layer (--expert-bytes, no container read; codec mul1; config derives 14155776 B at nvfp4)
  MoE layers x experts                     42 x 288 (top-8), MTP layer 45 not executed (step 21): no cache, no experts
  KV per token                    16918 B  11 DSA layers x (MLA latent 1024 B BF16 + indexer 514 B HF layout)
  card VRAM                 34190917632 B    31.84 GiB (RTX 5090 of record, docs/system-landscape.md:12; not free VRAM)
  pinned budget             49392123904 B    46.00 GiB (HOST_PINNED_CAP; the boot takes min(cap, free RAM - CROW_RAM_MARGIN_GB 1 GiB), free RAM not read here)
VRAM
  plan ceiling              32043433984 B    29.84 GiB  card - headroom 2.00 GiB, cap 31.90 GiB (Stability::GLM5_NEXT, #176)
  dense part                 5981546744 B     5.57 GiB
  kv_b at BF16                265289728 B     0.25 GiB  11 DSA layers: kv_b decoded at load, BF16 369098752 B - container NVFP4 103809024 B (#161)
  MLA latent                 2252800000 B     2.10 GiB  11 x 200000 tokens x 1024 B
  indexer cache              1130800000 B     1.05 GiB  11 x 200000 tokens x 514 B
  KDA state + conv            152633344 B     0.14 GiB  34 layers x (4194304 + 294912 B) per sequence
  cold staging               1515847680 B     1.41 GiB  160 slots (decode and prefill sets apart, #176)
  launch slack + safety       671088640 B     0.62 GiB
  activations/scratch  not measured (GLM widths; the step-14 boot measures them against free VRAM)
  room for experts          20073427848 B    18.69 GiB
expert tiers per MoE layer (unit = 397910016 B = 42 layers x one block)
  VRAM    N =  50   17.4 %       19895500800 B    18.53 GiB
  pinned  P = 124   43.1 %       49340841984 B    45.95 GiB
  NVMe        114   39.6 %       45361741824 B    42.25 GiB  (CROW_NVME_TIER, #149)
  resident (N + P) / 288 = 60.4 %; expert cache policy default lru (CROW_EXPERT_CACHE, #175)
  G1d capacities (plan step 3): not measured yet - these are the planner's own numbers
"""

GLM_CONTAINER = 178_478_618_624
GLM_PAYLOAD = 178_478_590_200
GLM_ROUTED = 171_228_266_496
MUL1_RECORD = 9_474_048


# ------------------------------------------------------------------ synthetic model dirs

def expand(name):
    m = RANGE.search(name)
    if not m:
        return [name]
    return [name[:m.start()] + str(i) + name[m.end():] for i in range(int(m.group(1)), int(m.group(2)) + 1)]


def write_dir(d, cfg_dir, rows, shard_info, config=None):
    """rows: (name, dtype, shape, shard key, bytes); shard_info: key -> (file, data_start, size)"""
    os.makedirs(os.path.join(d, "headers"))
    for f in ("config.json", "generation_config.json"):
        shutil.copyfile(os.path.join(cfg_dir, f), os.path.join(d, f))
    if config is not None:
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config, f)
    headers, offs, wm = {}, {}, {}
    for name, dt, shape, key, b in rows:
        o = offs.get(key, 0)
        headers.setdefault(key, {})[name] = {"dtype": dt, "shape": shape, "data_offsets": [o, o + b]}
        offs[key] = o + b
        wm[name] = shard_info[key][0]
    for key, h in headers.items():
        file, start, size = shard_info[key]
        with open(os.path.join(d, "headers", file + ".json"), "w", encoding="utf-8") as f:
            json.dump({"shard": file, "size": size, "data_start": start, "header": h}, f)
    with open(os.path.join(d, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
        json.dump({"metadata": {}, "weight_map": wm}, f)
    return d


def write_glm_dir(d, config=None):
    shards, rows = {}, []
    with open(GLM_TSV, encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if line.startswith("# shard\t"):
                _, nr, start, size = line.split("\t")
                shards[nr] = ("model-%05d-of-00062.safetensors" % int(nr), int(start), int(size))
                continue
            if not line or line.startswith("#"):
                continue
            name, dt, shape, nr, b = line.split("\t")
            for n in expand(name):
                rows.append((n, dt, [int(x) for x in shape.split("x")], nr, int(b)))
    return write_dir(d, GLM_CFG, rows, shards, config)


def write_q27_dir(d, config=None):
    rows = []
    with open(Q27_TSV, encoding="utf-8") as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            name, shape, dt = line.rstrip("\n").split("\t")
            shp = [int(x) for x in shape.split("x")]
            rows.append((name, dt, shp, "1", math.prod(shp) * pq.DTYPE_BYTES[dt]))
    start = 8 + 200_000
    info = {"1": ("model-00001-of-00001.safetensors", start, start + sum(r[4] for r in rows))}
    return write_dir(d, Q27_CFG, rows, info, config)


def run_main(argv):
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        rc = pq.main(argv)
    return rc, out.getvalue(), err.getvalue()


def write_rates(d, **kw):
    p = os.path.join(d, "rates.json")
    doc = {"B_nvme_gbps": 6.994, "R_pcie_gbps": 47.78, "R_dram_gbps": 89.6, "R_vram_gbps": 1792,
           "source": "test rates"}
    doc.update(kw)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(doc, f)
    return p


class _Dirs(unittest.TestCase):
    """one synthetic GLM dir and one 27B dir for every test that needs them (built once)"""
    tmp = glm = q27 = rates = None

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="test-prequal-")
        cls.glm = write_glm_dir(os.path.join(cls.tmp, "glm"))
        cls.q27 = write_q27_dir(os.path.join(cls.tmp, "q27"))
        cls.rates = write_rates(cls.tmp)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def json_of(self, argv):
        fd, out = tempfile.mkstemp(suffix=".json", dir=self.tmp)
        os.close(fd)
        rc, md, err = run_main(argv + ["--json", out])
        self.assertEqual(rc, 0, err)
        with open(out, encoding="utf-8") as f:
            return json.load(f), md


# ------------------------------------------------------------------ stage ceilings

class TestStageCeiling(unittest.TestCase):
    def test_ceilings_equal_the_old_dyn_cost_arithmetic(self):
        # the case of tools/test_glm_tier_sim.py:546-566 (m 0.1, PCIe 0.4, DRAM 0.5 of 336 visits x 3.05-bpw record)
        per = 336 * MUL1_RECORD
        c = sc.ceilings([("nvme", 0.1 * per, 6.994), ("pcie", 0.4 * per, 51.6), ("dram", 0.5 * per, 89.6)])
        self.assertAlmostEqual(c["nvme"]["ceiling_tok_s"], 6.994e9 / (0.1 * per))
        self.assertAlmostEqual(c["pcie"]["ceiling_tok_s"], 51.6e9 / (0.4 * per))
        self.assertEqual(c["binding"], "nvme")
        self.assertEqual(c["ceiling_tok_s"], c["nvme"]["ceiling_tok_s"])
        self.assertAlmostEqual(1 / c["serial_tok_s"], 0.1 * per / 6.994e9 + 0.4 * per / 51.6e9 + 0.5 * per / 89.6e9)
        bare = sc.ceilings([("nvme", 0.1 * per, None), ("pcie", 0.4 * per, None)])
        self.assertIsNone(bare["ceiling_tok_s"])
        self.assertIsNone(bare["serial_tok_s"])
        self.assertIsNone(bare["binding"])
        z = sc.ceilings([("vram", 0, 1792.0), ("nvme", 0, 6.994)])
        self.assertEqual(z["vram"]["ceiling_tok_s"], math.inf)
        self.assertEqual(z["serial_tok_s"], math.inf)

    def test_read_rates_contract(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "rates.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump({"R_pcie_gbps": 47.78, "B_nvme_gbps": 6.994, "source": "docs/architecture.md:789"}, f)
            rates, why = sc.read_rates(p, ("R_pcie_gbps", "R_dram_gbps"))
            self.assertEqual(rates, {"R_pcie_gbps": 47.78})
            self.assertEqual(why, "rates.json (docs/architecture.md:789)")
            self.assertEqual(sc.read_rates(p, ("B_nvme_gbps",))[0], {"B_nvme_gbps": 6.994})
            for bad in (-1, 0, True, "7"):
                with open(p, "w", encoding="utf-8") as f:
                    json.dump({"R_pcie_gbps": bad}, f)
                with self.assertRaises(KeyError):
                    sc.read_rates(p, ("R_pcie_gbps",), KeyError)
        self.assertEqual(sc.read_rates(None, ("R_pcie_gbps",)), ({}, "no rates file given"))

    def test_nvme_ceiling_at_given_m(self):
        # B 6.994 GB/s, MUL1 record 9,474,048 B, 42 x 8 = 336 visits: m 0.11 / 0.125 -> 19.97 / 17.58 tok/s
        rows = pq.decode_rows(moe=True, dense=0, kv=0, rates={"B_nvme_gbps": 6.994}, m_values=[0.11, 0.125],
                              placement={"hot": 50, "pinned": 124, "nvme": 114}, record=MUL1_RECORD, moe_layers=42,
                              topk=8, experts=288)
        self.assertEqual([r["visits"] for r in rows], [336, 336])
        self.assertEqual([round(r["ceil"]["nvme"]["ceiling_tok_s"], 2) for r in rows], [19.97, 17.58])
        self.assertEqual([r["ceil"]["binding"] for r in rows], ["nvme", "nvme"])
        # the rest 1 - m is split N:P between VRAM and pinned
        self.assertAlmostEqual(rows[0]["ceil"]["pcie"]["bytes_per_token"] / (336 * 0.89 * 124 / 174 * MUL1_RECORD), 1.0,
                               places=12)
        self.assertAlmostEqual(rows[0]["ceil"]["dram"]["bytes_per_token"],
                               rows[0]["ceil"]["pcie"]["bytes_per_token"] + rows[0]["ceil"]["nvme"]["bytes_per_token"])


# ------------------------------------------------------------------ evaluator and source constants

class TestEvaluator(unittest.TestCase):
    def test_accepts_rust_integer_expressions(self):
        self.assertEqual(pq.eval_int("46 << 30"), 46 << 30)
        self.assertEqual(pq.eval_int("32_607"), 32607)
        self.assertEqual(pq.eval_int("319 * (1 << 30) / 10"), 319 * (1 << 30) // 10)
        self.assertEqual(pq.eval_int("512u64 << 20"), 512 << 20)
        self.assertEqual(pq.eval_int("A * 3 - 1", {"A": 2}), 5)

    def test_refuses_calls_names_and_other_nodes(self):
        for bad in ("len(x)", "foo", "__import__('os')", "a.b", "1.5", "2 ** 3", "-1", "1 / 0", "3 - 4", "1 if 1 else 2"):
            with self.assertRaises(ValueError, msg=bad):
                pq.eval_int(bad)


class TestSourceConstants(unittest.TestCase):
    def test_real_engine_sources(self):
        c = pq.engine_constants(REPO)
        self.assertEqual(c["HOST_PINNED_CAP"]["value"], 46 << 30)
        self.assertEqual(c["HOST_PINNED_CAP"]["source"], "engine/src/geo.rs:42")
        want = {"CARD_MIB_OF_RECORD": 32_607, "SAFETY": 512 << 20, "LAUNCH_SLACK": 128 << 20, "RAM_MARGIN_DEFAULT_GB": 1,
                "PF_TG": 64, "CHUNK_CAP": 4096}
        for name, v in want.items():
            self.assertEqual(c[name]["value"], v, name)
            rel, line = c[name]["source"].rsplit(":", 1)
            with open(os.path.join(REPO, rel), encoding="utf-8") as f:
                self.assertIn(name, f.read().splitlines()[int(line) - 1], c[name]["source"])
        g, o = c["stability"]["GLM5_NEXT"], c["stability"]["OF_RECORD"]
        self.assertEqual((g["vram_headroom"], g["vram_cap"], g["decode_stage_rows"]), (2 << 30, 319 * (1 << 30) // 10, 4))
        self.assertEqual((o["vram_headroom"], o["vram_cap"], o["decode_stage_rows"]), (0, None, None))
        self.assertEqual(c["stability_of"]["map"], {"FlashNext": "OF_RECORD", "Qwen35Dense": "OF_RECORD",
                                                    "Glm5Next": "GLM5_NEXT"})
        self.assertGreaterEqual(c["PF_ASYNC_DEFAULT"]["value"], 1)
        fams = pq.converter_families(REPO)
        self.assertEqual({k: v["family"] for k, v in fams.items()},
                         {"qwen4_exp_text": "FlashNext", "qwen3_5_text": "Qwen35Dense", "glm5_next_text": "Glm5Next"})

    def test_missing_constant_exits_3_naming_it(self):
        with tempfile.TemporaryDirectory() as root:
            for rel in ("engine/src/geo.rs", "engine/src/manager.rs", "engine/src/gen.rs", "engine/src/bin/states.rs",
                        "converter/src/recipe.rs"):
                os.makedirs(os.path.dirname(os.path.join(root, rel)), exist_ok=True)
                shutil.copyfile(os.path.join(REPO, rel), os.path.join(root, rel))
            geo = os.path.join(root, "engine/src/geo.rs")
            with open(geo, encoding="utf-8") as f:
                text = f.read()
            with open(geo, "w", encoding="utf-8") as f:
                f.write(re.sub(r"^pub const HOST_PINNED_CAP.*$", "", text, flags=re.M))
            d = write_q27_dir(os.path.join(root, "q27"))
            with mock.patch.object(pq, "SRC_ROOT", root):
                rc, out, err = run_main(["--model-dir", d, "--context", "190", "--generic"])
        self.assertEqual(rc, 3)
        self.assertIn("HOST_PINNED_CAP", err)
        self.assertEqual(out, "")


# ------------------------------------------------------------------ KV cache

class TestKv(unittest.TestCase):
    def test_gqa(self):
        tc = {"num_hidden_layers": 4, "num_attention_heads": 8, "num_key_value_heads": 2, "hidden_size": 512}
        kv = pq.kv_cache(tc, 1000, "bf16")
        self.assertEqual(kv["per_token"], 4 * 2 * 2 * 64 * 2)
        self.assertEqual(kv["at_ctx"], 4 * 2 * 2 * 64 * 2 * 1000)
        self.assertEqual(pq.kv_cache(tc, 1000, "fp8")["at_ctx"], kv["at_ctx"] // 2)

    def test_mla(self):
        tc = {"num_hidden_layers": 3, "num_attention_heads": 16, "kv_lora_rank": 512, "qk_rope_head_dim": 64}
        self.assertEqual(pq.kv_cache(tc, 10, "bf16")["at_ctx"], 3 * (512 + 64) * 2 * 10)

    def test_hybrid_with_linear_layers(self):
        with open(os.path.join(Q27_CFG, "config.json"), encoding="utf-8") as f:
            q = json.load(f)["text_config"]
        kv = pq.kv_cache(q, 190, "bf16")   # 16 full (GQA 4 x 256) + 48 GDN layers
        self.assertEqual(kv["attention"], 16 * 2 * 4 * 256 * 2 * 190)
        self.assertEqual(kv["linear"], 48 * (48 * 128 * 128 * 4 + (2 * 16 * 128 + 48 * 128) * 3 * 4))
        with open(os.path.join(GLM_CFG, "config.json"), encoding="utf-8") as f:
            g = json.load(f)["text_config"]
        kv = pq.kv_cache(g, 200_000, "bf16")   # 11 MLA + 34 KDA: the latent and KDA lines of states --plan
        self.assertEqual(kv["attention"], 2_252_800_000)
        self.assertEqual(kv["linear"], 152_633_344)
        self.assertTrue(any("indexer" in s for s in kv["not_derived"]))
        nd = pq.kv_cache({"num_hidden_layers": 2, "num_attention_heads": 2, "head_dim": 8,
                          "layer_types": ["mamba", "attention"]}, 10, "bf16")
        self.assertEqual(nd["layers"], {"full": 1, "sliding": 0, "linear": 1})
        self.assertTrue(nd["not_derived"])

    def test_sliding_window(self):
        tc = {"num_hidden_layers": 6, "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 32,
              "sliding_window": 100, "sliding_window_pattern": 3}
        kv = pq.kv_cache(tc, 1000, "bf16")
        self.assertEqual(kv["layers"], {"full": 2, "sliding": 4, "linear": 0})
        self.assertEqual(kv["at_ctx"], 2 * 128 * 1000 + 4 * 128 * 100)
        lt = dict(tc, layer_types=["sliding_attention", "full_attention"] * 3, sliding_window=50)
        del lt["sliding_window_pattern"]
        self.assertEqual(pq.kv_cache(lt, 1000, "bf16")["at_ctx"], 3 * 128 * 1000 + 3 * 128 * 50)
        self.assertEqual(pq.kv_cache(lt, 20, "bf16")["at_ctx"], 6 * 128 * 20)


# ------------------------------------------------------------------ the two tool parsers

class TestParsers(unittest.TestCase):
    def test_converter_plan_totals(self):
        t = pq.parse_plan_totals("plan      x: family Glm5Next\n...\n" + PLAN_TOTALS)
        self.assertEqual(t["container"], GLM_CONTAINER)
        self.assertEqual(t["payload"], GLM_PAYLOAD)
        self.assertEqual(t["routed"], GLM_ROUTED)
        self.assertEqual(t["dense"], 5_981_546_744)
        self.assertEqual(t["subtotal"], 177_209_813_240)
        self.assertEqual(t["embed"], 1_268_776_960)
        self.assertEqual([(n, b) for _, n, b in t["omitted"]], [(1760, 7_493_399_168), (347, 1_127_254_016)])
        with self.assertRaises(pq.ToolFailure):
            pq.parse_plan_totals(PLAN_TOTALS.replace("container without index trailer", "container"))

    def test_states_table(self):
        s = pq.parse_states_table(STATES_OUT)
        self.assertEqual((s["hot"], s["pinned"], s["nvme"]), (50, 124, 114))
        self.assertEqual((s["unit"], s["unit_layers"], s["slots"]), (397_910_016, 42, 160))
        self.assertEqual((s["ceiling"], s["room"]), (32_043_433_984, 20_073_427_848))
        self.assertEqual((s["kv_per_token"], s["kv_total"]), (16_918, 2_252_800_000 + 1_130_800_000 + 152_633_344))
        self.assertEqual((s["moe_layers"], s["experts"], s["record"]), (42, 288, MUL1_RECORD))
        with self.assertRaises(pq.ToolFailure):
            pq.parse_states_table(STATES_OUT.split("expert tiers")[0])


# ------------------------------------------------------------------ the generic path on the fixtures

class TestGeneric(_Dirs):
    def test_glm_classification_and_size(self):
        m = pq.read_model_dir(self.glm)
        self.assertEqual((len(m["tensors"]), len(m["shards"])), (76_108, 62))
        c = pq.classify(m["tensors"], m["tc"])
        self.assertEqual(c["routed"], GLM_ROUTED)
        self.assertEqual((c["records"], len(c["moe_layers"]), c["experts"]), (12_096, 42, 288))
        self.assertEqual(sorted((n, b) for n, b in c["omitted"].values()), [(347, 1_127_254_016), (1760, 7_493_399_168)])
        self.assertEqual(c["embed"], 1_268_776_960)
        self.assertLess(abs(12 + c["payload"] - GLM_CONTAINER) / GLM_CONTAINER, 0.05)
        self.assertEqual((c["mul1"]["record"], c["mul1"]["exact"], c["mul1"]["fallback"]), (MUL1_RECORD, 12_096, 0))

    def test_glm_generic_prints_estimate_and_splits_within_10_percent(self):
        j, md = self.json_of(["--model-dir", self.glm, "--context", "200000", "--generic", "--rates", self.rates])
        self.assertIn("estimate: no recipe", md)
        self.assertIn("band: not given (--band LO,HI; the calibration lives outside this repo)", md)
        self.assertLess(abs(j["size"]["container"]["value"] - GLM_CONTAINER) / GLM_CONTAINER, 0.05)
        mul = [c for c in j["candidates"] if c["name"] == pq.MUL1][0]
        self.assertEqual(mul["record"]["value"], MUL1_RECORD)
        got = [mul["placement"][k]["value"] for k in ("hot", "pinned", "nvme")]
        for g, want in zip(got, (50, 124, 114)):
            self.assertLessEqual(abs(g - want) / want, 0.10, got)
        self.assertEqual(mul["placement"]["slots"]["value"], 160)

    def test_27b_all_in_vram_ceiling_near_120(self):
        j, md = self.json_of(["--model-dir", self.q27, "--context", "190", "--generic", "--rates", self.rates,
                              "--band", "0.5,0.8"])
        self.assertFalse(j["moe"])
        self.assertEqual(len(j["candidates"]), 1)
        c = j["candidates"][0]
        self.assertTrue(c["fits"].startswith("yes"))
        v = c["decode"][0]["stages"]["vram"]["ceiling_tok_s"]
        self.assertLessEqual(abs(v - 120) / 120, 0.10, v)
        self.assertIn("**Prefill:** compute-bound, not modelled here.", md)
        self.assertEqual(j["stability"]["name"], "OF_RECORD")

    def test_unknown_family_takes_the_generic_path(self):
        with open(os.path.join(Q27_CFG, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["text_config"]["model_type"] = "mistral9_text"
        d = write_q27_dir(os.path.join(self.tmp, "unknown"), cfg)
        j, md = self.json_of(["--model-dir", d, "--context", "190", "--converter", os.path.join(self.tmp, "absent")])
        self.assertIn("mistral9_text; estimate: no recipe", md)
        self.assertEqual(j["stability"]["name"], "GLM5_NEXT")
        self.assertIsNone(j["model"]["family"])

    def test_commit_and_free_ram(self):
        j, md = self.json_of(["--model-dir", self.glm, "--context", "200000", "--generic", "--free-ram-gib", "30",
                              "--commit-gib", "40"])
        self.assertEqual(j["pinned_budget"]["value"], 29 << 30)
        mul = [c for c in j["candidates"] if c["name"] == pq.MUL1][0]
        unit = mul["placement"]["unit"]["value"]
        self.assertEqual(mul["placement"]["pinned"]["value"], (29 << 30) // unit)
        self.assertEqual(mul["commit"]["used"], mul["placement"]["vram_used"]["value"] + mul["placement"]["pinned_used"]["value"])
        self.assertFalse(mul["commit"]["ok"])
        self.assertIn("commit over the limit", mul["fits"])


# ------------------------------------------------------------------ refusals

class FakeHeaderError(Exception):
    pass


def fake_helper(result=None, error=None, calls=None):
    def fetch_model(repo, revision, out_dir, curl=None, log=None):
        if calls is not None:
            calls.append((repo, revision, out_dir))
        if error:
            raise FakeHeaderError(error)
        return result
    return types.SimpleNamespace(HeaderError=FakeHeaderError, fetch_model=fetch_model)


class TestRefusals(_Dirs):
    def test_model_dir_without_index_and_without_single_file(self):
        d = os.path.join(self.tmp, "noindex")
        os.makedirs(d, exist_ok=True)
        shutil.copyfile(os.path.join(Q27_CFG, "config.json"), os.path.join(d, "config.json"))
        rc, out, err = run_main(["--model-dir", d, "--context", "190"])
        self.assertEqual(rc, 2)
        self.assertIn("no model.safetensors.index.json and no single-file header", err)

    def test_header_error_is_exit_2_with_the_reason(self):
        why = "zai-org/GLM-5.3-Flash: revision deadbeef not found (HTTP 404)"
        with mock.patch.object(pq, "HELPER_LOADER", lambda: fake_helper(error=why)):
            rc, out, err = run_main(["--repo", "zai-org/GLM-5.3-Flash", "--revision", "deadbeef", "--context", "1000",
                                     "--work", os.path.join(self.tmp, "w")])
        self.assertEqual(rc, 2)
        self.assertIn(why, err)
        rc, out, err = run_main(["--repo", "zai-org/GLM-5.3-Flash", "--context", "1000"])
        self.assertEqual(rc, 2)
        self.assertIn("--revision", err)

    def test_repo_path_reports_bytes_transferred(self):
        calls = []
        res = {"repo": "Qwen/Qwen3.8-27B", "revision": "1d4bf0f2ff60", "dir": self.q27,
               "headers_dir": os.path.join(self.q27, "headers"), "shards": 1, "bytes_transferred": 1_234_567,
               "requests": 3, "cached": False}
        work = os.path.join(self.tmp, "work")
        with mock.patch.object(pq, "HELPER_LOADER", lambda: fake_helper(result=res, calls=calls)):
            rc, out, err = run_main(["--repo", "Qwen/Qwen3.8-27B", "--revision", "1d4bf0f2ff6012fd", "--context", "190",
                                     "--generic", "--work", work])
        self.assertEqual(rc, 0, err)
        self.assertEqual(calls, [("Qwen/Qwen3.8-27B", "1d4bf0f2ff6012fd", work)])
        self.assertIn("**Model:** Qwen/Qwen3.8-27B@1d4bf0f2ff60;", out)
        self.assertIn("1,234,567 B transferred (3 requests)", out)

    def test_bad_flag_values_exit_2(self):
        base = ["--model-dir", self.q27, "--context", "190", "--generic"]
        for extra in (["--band", "0.8,0.5"], ["--band", "x,1"], ["--band", "0.5"], ["--m", "1.5"], ["--m", "0"],
                      ["--m", "a"], ["--m", "0.1,0.2,0.3"], ["--kv-dtype", "q8"], ["--free-ram-gib", "-1"]):
            rc, out, err = run_main(base + extra)
            self.assertEqual(rc, 2, extra)
            self.assertTrue(err.strip(), extra)
        rc, out, err = run_main(["--model-dir", self.q27, "--context", "0"])
        self.assertEqual(rc, 2)

    def test_unknown_dtype_exit_2(self):
        d = write_q27_dir(os.path.join(self.tmp, "q4k"))
        p = os.path.join(d, "headers", "model-00001-of-00001.safetensors.json")
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        h["header"]["lm_head.weight"]["dtype"] = "Q4_K"
        with open(p, "w", encoding="utf-8") as f:
            json.dump(h, f)
        rc, out, err = run_main(["--model-dir", d, "--context", "190", "--generic"])
        self.assertEqual(rc, 2)
        self.assertIn("Q4_K", err)

    def test_known_family_without_converter_exits_3_naming_the_build(self):
        rc, out, err = run_main(["--model-dir", self.q27, "--context", "190", "--converter",
                                 os.path.join(self.tmp, "absent", "converter")])
        self.assertEqual(rc, 3)
        self.assertIn("cargo build --release", err)


# ------------------------------------------------------------------ the real binaries (opt-in)

@unittest.skipUnless(CONVERTER, "PREQUAL_CONVERTER not set: the converter-backed tests need the binary")
class TestConverterBinary(_Dirs):
    def test_glm_known_family_takes_converter_totals(self):
        args = ["--model-dir", self.glm, "--context", "200000", "--converter", CONVERTER]
        if not STATES:
            args += ["--states", os.path.join(self.tmp, "absent")]
            rc, out, err = run_main(args)
            self.assertEqual(rc, 3)   # glm5_next needs states for the placement
            self.skipTest("PREQUAL_STATES not set")
        j, md = self.json_of(args + ["--states", STATES])
        self.assertEqual(j["size"]["payload"]["value"], GLM_PAYLOAD)
        self.assertEqual(j["size"]["routed"]["value"], GLM_ROUTED)
        self.assertNotIn("estimate: no recipe", md)
        self.assertIn("recipe cnq4.5-glm5-next (converter plan)", md)

    def test_27b_known_family(self):
        j, md = self.json_of(["--model-dir", self.q27, "--context", "190", "--converter", CONVERTER,
                              "--rates", self.rates])
        self.assertIn("Qwen35Dense, recipe cnq4.5-qwen35-dense (converter plan)", md)
        v = j["candidates"][0]["decode"][0]["stages"]["vram"]["ceiling_tok_s"]
        self.assertLessEqual(abs(v - 120) / 120, 0.10, v)

    def test_converter_refusal_exits_3(self):
        with open(os.path.join(GLM_CFG, "config.json"), encoding="utf-8") as f:
            cfg = json.load(f)
        cfg["text_config"]["num_hidden_layers"] = 44
        d = write_glm_dir(os.path.join(self.tmp, "glm44"), cfg)
        rc, out, err = run_main(["--model-dir", d, "--context", "200000", "--converter", CONVERTER,
                                 "--states", STATES or os.path.join(self.tmp, "absent")])
        self.assertEqual(rc, 3)
        self.assertIn("converter plan refused" if STATES else "states binary not found", err)


@unittest.skipUnless(CONVERTER and STATES, "PREQUAL_CONVERTER and PREQUAL_STATES not set: the states-backed test needs both")
class TestStatesBinary(_Dirs):
    def test_glm_mul1_placement_is_states_plan(self):
        j, md = self.json_of(["--model-dir", self.glm, "--context", "200000", "--converter", CONVERTER,
                              "--states", STATES, "--rates", self.rates, "--m", "0.11,0.125"])
        mul = [c for c in j["candidates"] if c["name"] == pq.MUL1][0]
        self.assertEqual([mul["placement"][k]["value"] for k in ("hot", "pinned", "nvme")], [50, 124, 114])
        self.assertEqual(mul["placement"]["unit"]["value"], 397_910_016)
        self.assertEqual([round(d["stages"]["nvme"]["ceiling_tok_s"], 2) for d in mul["decode"]], [19.97, 17.58])
        self.assertEqual(j["kv"]["states"]["kv_per_token"]["value"], 16_918)


if __name__ == "__main__":
    unittest.main(verbosity=1)
