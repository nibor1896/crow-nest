#!/usr/bin/env python3
"""#147: tests of tools/glm_route_passes.py (the step-8 routing passes). No GPU, no real weights:

  .venv-oracle/Scripts/python.exe -I tools/test_glm_route_passes.py

- the pinned amendment-1 table equals the one in runs/glm53-flash/PREREG.md;
- a container without its index trailer (or with the converter's journal beside it, partial, of
  another recipe or revision, or missing a layer) is refused;
- corpus files with other sha256 or another length are refused;
- end to end on the synthetic small checkpoint of the runner's selftest (8 layers, 5 MoE layers, 16
  experts): a pass writes routing the tier simulator's self-test accepts and leaves no hand-over state;
  an interrupted pass resumes from its last state and ends with the same routing files; a done pass is
  skipped; a run dir of other weights is refused;
- #179, the FP8 originals as weights: a directory is accepted only when tools/fetch-glm.py's record
  (hf-revision.json of rev eb9eb208) and a matching `.verified` marker stand behind config, index and
  all 62 shards; a missing or wrong marker, another revision or a changed index is refused by name; the
  pass dir records the weights' identity, resumes under it and refuses other weights.
"""
import hashlib
import importlib.util
import json
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import glm_route_passes as R  # noqa: E402
import glm_tier_sim as ts  # noqa: E402

_SPEC = importlib.util.spec_from_file_location("fetch_glm_test", HERE / "fetch-glm.py")
fg = importlib.util.module_from_spec(_SPEC)  # the verification record's own writer (#179)
_SPEC.loader.exec_module(fg)

REPO = HERE.parent
PREREG = REPO / "runs" / "glm53-flash" / "PREREG.md"


def quiet(*_):
    pass


def book(runs):
    with open(os.path.join(runs, "passes.jsonl"), encoding="utf-8") as f:
        return [json.loads(x) for x in f]


def write_container(path, layers=3, partial=None, revision=R.REVISION, recipe=R.RECIPE, trailer=True,
                    drop=None):
    names = ["%slayers.%d.mlp.gate.weight" % (R.LM, l) for l in range(layers)]
    names += [R.LM + "embed_tokens.weight", R.LM + "norm.weight", "lm_head.weight"]
    names = [n for n in names if n != drop]
    idx = {"format_version": 2, "recipe": recipe, "partial": partial,
           "model": {"source": {"repo": "zai-org/GLM-5.3-Flash", "revision": revision},
                     "config_json": json.dumps({"text_config": {"num_hidden_layers": layers}})},
           "tensors": [{"name": n, "dtype": "bf16", "shape": [2], "n_values": 2} for n in names]}
    raw = json.dumps(idx).encode()
    with open(path, "wb") as f:
        f.write(b"CNQ1" + b"\0" * 8 + os.urandom(4096))
        if trailer:
            f.write(raw + struct.pack("<Q", len(raw)))


class Amendment(unittest.TestCase):
    def test_pinned_table_is_prereg_amendment_1(self):
        text = PREREG.read_text(encoding="utf-8")
        amend = text[text.index("## Amendment 1"):]
        rows = re.findall(r"^\| `([a-z0-9-]+)` \| \**(held-out|calibration)\**.*\| `([0-9a-f]{64})` \| `([0-9a-f]{64})` \|$",
                          amend, re.M)
        self.assertEqual([(n, "held" if r == "held-out" else "cal", i, m) for n, r, i, m in rows],
                         [tuple(r) for r in R.AMENDMENT_1])
        self.assertIn("`corpus.json` there has sha256 `%s`" % R.CORPUS_JSON_SHA256, amend)


class Container(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="route-cnq-")
        self.p = os.path.join(self.d, "x.cnq")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def refused(self, needle, **kw):
        write_container(self.p, **kw)
        with self.assertRaises(R.Refusal) as cm:
            R.container_index(self.p)
        self.assertIn(needle, str(cm.exception))

    def test_complete_container_is_accepted(self):
        write_container(self.p)
        idx, sha = R.container_index(self.p)
        self.assertEqual(len(sha), 64)
        self.assertEqual(idx["recipe"], R.RECIPE)

    def test_no_trailer_is_refused(self):
        # a container still being written ends in tensor bytes, not in an index
        self.refused("no index trailer", trailer=False)

    def test_journal_beside_it_is_refused(self):
        write_container(self.p)
        open(self.p + ".journal.jsonl", "w").close()
        with self.assertRaises(R.Refusal) as cm:
            R.container_index(self.p)
        self.assertIn("still running", str(cm.exception))

    def test_partial_other_recipe_other_revision_missing_layer(self):
        self.refused("PARTIAL", partial={"filter": "--layers 0-3"})
        self.refused("expected v2", recipe="cnq4.5-qwen35-dense")
        self.refused("source revision", revision="0" * 40)
        self.refused("lacks 1", drop="%slayers.1.mlp.gate.weight" % R.LM)
        self.refused("lacks lm_head.weight", drop="lm_head.weight")

    def test_missing_file(self):
        with self.assertRaises(R.Refusal):
            R.container_index(os.path.join(self.d, "none.cnq"))


def write_corpus(d, files):
    """files: {name: ids} -> (table, corpus sha) in amendment-1 form"""
    os.makedirs(d, exist_ok=True)
    table = []
    for name, ids in files.items():
        ts.jdump(ids, os.path.join(d, name + "-ids.json"))
        ts.jdump({"tokens": len(ids), "spans": [[0, len(ids)]]}, os.path.join(d, name + "-mask.json"))
        table.append((name, "cal", ts.sha256_ids(ids), ts.sha256_file(os.path.join(d, name + "-mask.json"))))
    ts.jdump({"files": [{"name": n} for n in files]}, os.path.join(d, "corpus.json"))
    return tuple(table), ts.sha256_file(os.path.join(d, "corpus.json"))


class Corpus(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="route-corpus-")

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def test_match_and_refusals(self):
        table, csha = write_corpus(self.d, {"a": list(range(10)), "b": list(range(5, 15))})
        got = R.check_corpus(self.d, ["a", "b"], tokens=10, table=table, corpus_sha=csha)
        self.assertEqual(sorted(got), ["a", "b"])
        with self.assertRaises(R.Refusal):
            R.check_corpus(self.d, ["a"], tokens=10, table=table, corpus_sha="0" * 64)
        with self.assertRaises(R.Refusal):  # not the routed length
            R.check_corpus(self.d, ["a"], tokens=12, table=table, corpus_sha=csha)
        ts.jdump(list(range(1, 11)), os.path.join(self.d, "a-ids.json"))
        with self.assertRaises(R.Refusal) as cm:
            R.check_corpus(self.d, ["a"], tokens=10, table=table, corpus_sha=csha)
        self.assertIn("ids sha256", str(cm.exception))
        with self.assertRaises(R.Refusal):
            R.check_corpus(self.d, ["c"], tokens=10, table=table, corpus_sha=csha)

    def test_cli_refuses_the_running_conversion(self):
        p = os.path.join(self.d, "x.cnq")
        write_container(p, trailer=False)
        r = subprocess.run([sys.executable, "-I", str(HERE / "glm_route_passes.py"), "--container", p,
                            "--corpus", self.d, "--runs", os.path.join(self.d, "runs"), "--dry-run"],
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("refused: ", r.stderr)
        self.assertFalse(os.path.exists(os.path.join(self.d, "runs")))


def mark_verified(d, revision=R.REVISION):
    """hf-revision.json over the files in d (plus the small files fetch-glm.py requires, recorded but
    absent), and a `.verified` marker beside each present file, as tools/fetch-glm.py writes them"""
    sib = []
    present = sorted(n for n in os.listdir(d) if n.endswith((".json", ".safetensors")) and n != fg.API_FILE)
    for n in present:
        data = Path(d, n).read_bytes()
        s256, s1 = hashlib.sha256(data).hexdigest(), fg.git_blob_sha1(data, len(data))
        row = {"rfilename": n, "size": len(data), "blobId": s1}
        if n.endswith(".safetensors"):
            row["lfs"] = {"size": len(data), "sha256": s256}
        sib.append(row)
        Path(d, n + ".verified").write_text(json.dumps({
            "file": n, "size": len(data), "sha256": s256, "blob": s1, "repo": fg.REPO,
            "revision": revision, "verified_at": "2026-10-08T23:00:00"}) + "\n", encoding="utf-8")
    for n in fg.SMALL_FILES:
        if n not in present:
            sib.append({"rfilename": n, "size": 3, "blobId": "0" * 40})
    Path(d, fg.API_FILE).write_text(json.dumps({"id": fg.REPO, "sha": revision, "siblings": sib}),
                                      encoding="utf-8")


def write_fp8(d, n=62):
    """a fake FP8 checkout as fetch-glm.py leaves it: config, index and n verified shards"""
    os.makedirs(d, exist_ok=True)
    names = ["model-%05d-of-%05d.safetensors" % (i, n) for i in range(1, n + 1)]
    for nm in names:
        Path(d, nm).write_bytes(nm.encode() * 3)
    Path(d, "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 3}}), encoding="utf-8")
    Path(d, "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": names[0]}}),
                                                       encoding="utf-8")
    mark_verified(d)
    return names


class Fp8(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp(prefix="route-fp8-")
        self.names = write_fp8(self.d)

    def tearDown(self):
        shutil.rmtree(self.d, ignore_errors=True)

    def refused(self, needle):
        with self.assertRaises(R.Refusal) as cm:
            R.fp8_identity(self.d)
        self.assertIn(needle, str(cm.exception))

    def test_complete_markers_are_accepted(self):
        ident = R.fp8_identity(self.d)
        self.assertEqual((ident["kind"], ident["revision"], ident["repo"]),
                         ("fp8-originals", R.REVISION, "zai-org/GLM-5.3-Flash"))
        self.assertEqual(sorted(ident["shards"]), self.names)
        self.assertEqual(ident["index_json_sha256"],
                         ts.sha256_file(os.path.join(self.d, "model.safetensors.index.json")))
        self.assertEqual(ident["shards"][self.names[5]], hashlib.sha256(self.names[5].encode() * 3).hexdigest())
        self.assertEqual(ident["identity_sha256"], R.identity_sha256(ident))

    def test_missing_marker_is_refused(self):
        os.remove(os.path.join(self.d, self.names[40] + ".verified"))
        self.refused(self.names[40])

    def test_wrong_marker_is_refused(self):
        mp = os.path.join(self.d, self.names[7] + ".verified")
        rec = json.loads(Path(mp).read_text(encoding="utf-8"))
        Path(mp).write_text(json.dumps(dict(rec, sha256="0" * 64)), encoding="utf-8")
        self.refused(self.names[7])
        Path(mp).write_text(json.dumps(dict(rec, revision="0" * 40)), encoding="utf-8")
        self.refused("revision " + "0" * 40)

    def test_other_revision_is_refused(self):
        mark_verified(self.d, revision="1" * 40)
        self.refused("revision")

    def test_missing_shard_changed_index_shard_count(self):
        os.remove(os.path.join(self.d, self.names[0]))  # e.g. deleted by glm-stage.py
        self.refused(self.names[0])
        write_fp8(self.d)
        ip = Path(self.d, "model.safetensors.index.json")
        ip.write_text(ip.read_text(encoding="utf-8").replace('"x"', '"y"'), encoding="utf-8")
        self.refused("not the %s tools/fetch-glm.py verified" % json.loads(  # same size, other bytes
            Path(str(ip) + ".verified").read_text(encoding="utf-8"))["sha256"])
        shutil.rmtree(self.d)
        write_fp8(self.d, n=61)
        self.refused("61 shards")

    def test_cli_refuses_unverified_fp8_and_both_kinds(self):
        os.remove(os.path.join(self.d, self.names[61] + ".verified"))
        runs = os.path.join(self.d, "runs")
        r = subprocess.run([sys.executable, "-I", str(HERE / "glm_route_passes.py"), "--fp8", self.d,
                            "--corpus", self.d, "--runs", runs, "--dry-run"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2, r.stderr)
        self.assertIn("refused: ", r.stderr)
        self.assertIn(self.names[61], r.stderr)
        self.assertFalse(os.path.exists(runs))
        r = subprocess.run([sys.executable, "-I", str(HERE / "glm_route_passes.py"), "--fp8", self.d,
                            "--container", "x.cnq", "--dry-run"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 2)
        self.assertIn("not allowed with", r.stderr)


class EndToEnd(unittest.TestCase):
    """the driver over the runner's synthetic small checkpoint (FP8 weights, so of_record=False)"""

    @classmethod
    def setUpClass(cls):
        sys.path.insert(0, str(REPO / "oracle"))
        import glm5_layerwise as LW  # torch + transformers: the oracle venv
        import glm5_common as G
        cls.d = tempfile.mkdtemp(prefix="route-e2e-")
        cls.ck = os.path.join(cls.d, "fp8")
        LW.make_synthetic("small", cls.ck)
        cls.sha = G.sha256_file(os.path.join(cls.ck, "model.safetensors.index.json"))
        cls.corpus = os.path.join(cls.d, "corpus")
        cls.table, _ = write_corpus(cls.corpus, {"a": [7, 900, 3, 3, 42] * 8, "b": [5, 17, 1, 999] * 10})
        cls.ids = {n: os.path.join(cls.corpus, n + "-ids.json") for n in ("a", "b")}
        cls.shape = (5, 16, 8)
        mark_verified(cls.ck)  # #179: as tools/fetch-glm.py leaves the originals
        cls.nshards = len([n for n in os.listdir(cls.ck) if n.endswith(".safetensors")])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.d, ignore_errors=True)

    def passes(self, runs, names=("a",)):
        return R.run_passes(list(names), self.ids, ("fp8-originals", self.ck), runs, self.sha, tokens=40,
                            prompt_chunk=16, shape=self.shape, of_record=False, log=quiet)

    def routing(self, out):
        man = ts.jload(os.path.join(out, "manifest.json"))
        return {n: v["sha256"] for n, v in man["files"].items() if "routing" in n}

    def test_pass_resume_skip(self):
        runs = os.path.join(self.d, "runs")
        self.assertEqual(self.passes(runs, ("a", "b")), 0)
        a = os.path.join(runs, "a")
        self.assertEqual(R.verify(a, self.shape, of_record=False), [])
        self.assertEqual(R.verify(a, self.shape, of_record=True),
                         ["routing not from the CNQ container (FP8 originals or other): plausibility only"])
        self.assertEqual([f for f in os.listdir(a) if "-output." in f], [])
        man = ts.jload(os.path.join(a, "manifest.json"))
        self.assertEqual((man["prompt_chunk"], man["state_dtype"], man["anchors"]), (16, "bf16", [39]))
        self.assertEqual(len(man["per_layer"][0]["prompt_call_s"]), 3)  # 40 rows in calls of 16
        rows = book(runs)
        self.assertEqual([(r["name"], r["rc"], r["ok"]) for r in rows], [("a", 0, True), ("b", 0, True)])

        # an interrupted pass (layers 0..2 done, state l2 on disk) resumes at layer 3, same routing
        cut = os.path.join(self.d, "runs-cut")
        cmd = R.runner_cmd(sys.executable, ("fp8-originals", self.ck), self.ids["a"], os.path.join(cut, "a"),
                           40, 16) + ["--layers", "0:3"]
        subprocess.run(cmd, check=True, capture_output=True)
        self.assertEqual(R.plan(os.path.join(cut, "a"), ts.jload(self.ids["a"]), self.sha), ("resume", 3))
        self.assertEqual(self.passes(cut), 0)
        self.assertEqual(self.routing(os.path.join(cut, "a")), self.routing(a))
        row = book(cut)[-1]
        self.assertEqual(row["resumed_from_layer"], 3)

        # a done pass is skipped (no new book row); a dir of other weights is refused
        self.assertEqual(R.plan(a, ts.jload(self.ids["a"]), self.sha), ("done", None))
        self.assertEqual(self.passes(runs), 0)
        self.assertEqual(len(book(runs)), 2)
        with self.assertRaises(R.Refusal):
            R.plan(a, ts.jload(self.ids["a"]), "f" * 64)
        with self.assertRaises(R.Refusal):
            R.plan(a, ts.jload(self.ids["b"]), self.sha)

    def test_fp8_identity_pass_resume_and_other_weights(self):
        with mock.patch.object(R.fg, "N_SHARDS", self.nshards):
            ident = R.fp8_identity(self.ck)
        self.assertEqual(ident["index_json_sha256"], self.sha)

        def go(runs, idn=ident):
            return R.run_passes(["a"], self.ids, ("fp8-originals", self.ck), runs, idn["index_json_sha256"],
                                tokens=40, prompt_chunk=16, shape=self.shape, of_record=True, log=quiet,
                                identity=idn)

        runs = os.path.join(self.d, "runs-fp8")
        self.assertEqual(go(runs), 0)  # of record: the FP8 source check, not the CNQ one
        a = os.path.join(runs, "a")
        self.assertEqual(R.verify(a, self.shape, of_record=True, kind="fp8-originals"), [])
        self.assertEqual(ts.jload(os.path.join(a, "weights.json")), ident)
        row = book(runs)[-1]
        self.assertEqual((row["weights"], row["weights_identity_sha256"], row["revision"], row["ok"]),
                         ("fp8-originals", ident["identity_sha256"], R.REVISION, True))

        # resume under the same identity, same routing as the uninterrupted pass
        cut = os.path.join(self.d, "runs-fp8-cut")
        cmd = R.runner_cmd(sys.executable, ("fp8-originals", self.ck), self.ids["a"], os.path.join(cut, "a"),
                           40, 16) + ["--layers", "0:3"]
        subprocess.run(cmd, check=True, capture_output=True)
        R.write_identity(os.path.join(cut, "a"), ident)
        self.assertEqual(R.plan(os.path.join(cut, "a"), ts.jload(self.ids["a"]), self.sha, ident), ("resume", 3))
        self.assertEqual(go(cut), 0)
        self.assertEqual(book(cut)[-1]["resumed_from_layer"], 3)
        self.assertEqual(self.routing(os.path.join(cut, "a")), self.routing(a))
        self.assertEqual(R.plan(a, ts.jload(self.ids["a"]), self.sha, ident), ("done", None))

        # other weights: another verified shard set, or the container kind, is refused by name
        other = dict(ident, shards=dict(ident["shards"], **{"model-00000.safetensors": "e" * 64}))
        other["identity_sha256"] = R.identity_sha256(other)
        with self.assertRaises(R.Refusal) as cm:
            R.plan(a, ts.jload(self.ids["a"]), self.sha, other)
        self.assertIn("other weights", str(cm.exception))
        cnq = {"kind": "container", "index_json_sha256": self.sha}
        cnq["identity_sha256"] = R.identity_sha256(cnq)
        os.remove(os.path.join(a, "weights.json"))  # a pre-#179 dir: the runner manifest's kind decides
        with self.assertRaises(R.Refusal) as cm:
            R.plan(a, ts.jload(self.ids["a"]), self.sha, cnq)
        self.assertIn("fp8", str(cm.exception))

    def test_no_state_to_resume_from_starts_again(self):
        runs = os.path.join(self.d, "runs-gone")
        cmd = R.runner_cmd(sys.executable, ("fp8-originals", self.ck), self.ids["b"], os.path.join(runs, "b"),
                           40, 16) + ["--layers", "0:2"]
        subprocess.run(cmd, check=True, capture_output=True)
        os.remove(os.path.join(runs, "b", "l1-output.bf16"))
        self.assertEqual(R.plan(os.path.join(runs, "b"), ts.jload(self.ids["b"]), self.sha), ("fresh", None))


if __name__ == "__main__":
    unittest.main()
