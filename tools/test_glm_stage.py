#!/usr/bin/env python3
"""#157: unit tests of tools/glm-stage.py (the delete-behind supervisor) - no network, no converter.

  python -I tools/test_glm_stage.py

A small world on disk stands in for `models/GLM-5.3-Flash-original/`: the HF record
(`hf-revision.json`, 62 shards), `config.json`, the header cache, shard files with the fetcher's
`.verified` markers, and a container plus journal written in the converter's format (magic,
reserved bytes, alignment zeros, one journal record per tensor with seq/pad/sha256/entry). What
has to hold: a shard is deleted only after its `.done`, only when `.done` names this container,
only when the shard is verified, and only when every tensor the recipe writes from it is in the
journal's verified prefix (container bytes re-hashed) or, after the trailer, in the index.
"""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("glm_stage", TOOLS / "glm-stage.py")
gs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(gs)
fg = gs.fg

L = "model.language_model.layers."


def shard_name(i):
    return f"model-{i:05d}-of-00062.safetensors"


# shard 1: embed + layer 0 + an FP8 pair; shard 2: layer 1 + MTP layer 45 + vision (omitted);
# shard 3: only omitted tensors; the rest: one tensor each
TENSORS = {
    1: ["model.language_model.embed_tokens.weight", L + "0.mlp.gate_proj.weight",
        L + "0.mlp.gate_proj.weight_scale_inv"],
    2: [L + "1.self_attn.o_proj.weight", L + "45.enorm.weight", "model.visual.blocks.0.attn.qkv.weight"],
    3: [L + "45.eh_proj.weight", L + "45.eh_proj.weight_scale_inv"],
}


class World:
    def __init__(self, root):
        self.dest = Path(root) / "models" / "GLM-5.3-Flash-original"
        self.dest.mkdir(parents=True)
        (self.dest / fg.HEADER_DIR).mkdir()
        self.out = Path(root) / "converter" / "GLM.cnq"
        self.out.parent.mkdir()
        self.logs = []
        self.data = {}
        sib = [{"rfilename": n, "blobId": "0" * 40, "size": 10} for n in fg.SMALL_FILES]
        for i in range(1, 63):
            names = TENSORS.get(i, [L + f"{i + 3}.post_attention_layernorm.weight"])
            body = bytes([i]) * (100 + i)
            self.data[i] = body
            sha = hashlib.sha256(body).hexdigest()
            sib.append({"rfilename": shard_name(i), "blobId": "1" * 40, "size": len(body),
                        "lfs": {"sha256": sha, "size": len(body)}})
            header = {n: {"dtype": "BF16", "shape": [1], "data_offsets": [0, 0]} for n in names}
            header["__metadata__"] = {"format": "pt"}
            (self.dest / fg.HEADER_DIR / (shard_name(i) + ".json")).write_text(
                json.dumps({"shard": shard_name(i), "size": len(body), "data_start": 8, "header": header}))
        (self.dest / fg.API_FILE).write_text(json.dumps({"sha": fg.REVISION, "siblings": sib}))
        (self.dest / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 45}}))

    def table(self):
        return fg.file_table(json.loads((self.dest / fg.API_FILE).read_text()))

    def fetch(self, i, sha=None):
        """As fetch-glm.py leaves a verified shard."""
        p = self.dest / shard_name(i)
        p.write_bytes(self.data[i])
        rec = {"file": p.name, "size": len(self.data[i]),
               "sha256": sha or hashlib.sha256(self.data[i]).hexdigest()}
        fg.marker_of(p).write_text(json.dumps(rec))

    def done(self, i, out=None):
        """As the converter writes `<shard>.done`."""
        (self.dest / (shard_name(i) + ".done")).write_text(json.dumps({"shard": shard_name(i), "out": str(out or self.out)}))

    def convert(self, names, pads=None, trailer=False):
        """Container + journal as the converter writes them, for `names` in order."""
        pads = pads or {}
        blob = b""
        lines = [json.dumps({"journal": "crow-nest converter", "version": 1, "recipe": "cnq4.5-glm5-next",
                             "scales": "mse", "tensors": 99, "order_sha256": "0" * 64})]
        entries = []
        for k, n in enumerate(names):
            pad = pads.get(n, 0)
            payload = hashlib.sha256(n.encode()).digest() * 3
            off = len(blob) + pad
            blob += bytes(pad) + payload
            entry = {"name": n, "offset": off, "len": len(payload)}
            entries.append(entry)
            lines.append(json.dumps({"seq": k, "pad": pad, "sha256": hashlib.sha256(payload).hexdigest(),
                                     "entry": entry, "sidecar": {}, "acc": {}}))
        body = b"CNQ1" + bytes(8) + blob
        jp = gs.journal_path(self.out)
        if trailer:
            idx = json.dumps({"format_version": 2, "tensors": entries}).encode()
            body += idx + struct.pack("<Q", len(idx))
            jp.unlink(missing_ok=True)
        else:
            jp.write_text("\n".join(lines) + "\n")
        self.out.write_bytes(body)

    def stage(self, **kw):
        return gs.Stage(self.dest, self.out, self.logs.append, bases=(self.dest.parent.parent,), **kw)


ALL_1 = ["model.language_model.embed_tokens.weight", L + "0.mlp.gate_proj.weight"]


class Pure(unittest.TestCase):
    def test_omitted_is_the_vision_tower_and_the_mtp_layers(self):
        self.assertTrue(gs.omitted("model.visual.blocks.0.attn.qkv.weight", 45))
        self.assertTrue(gs.omitted(L + "45.enorm.weight", 45))
        self.assertFalse(gs.omitted(L + "44.mlp.gate.weight", 45))
        self.assertFalse(gs.omitted(L + "4.mlp.gate.weight", 45))      # 4 is not 45
        self.assertFalse(gs.omitted("lm_head.weight", 45))

    def test_required_tensors_count_a_scale_through_its_weight(self):
        need = gs.required_tensors(["__metadata__", L + "0.mlp.up_proj.weight_scale_inv", "lm_head.weight",
                                    L + "45.eh_proj.weight_scale_inv", "model.visual.merger.weight"], 45)
        self.assertEqual(need, {L + "0.mlp.up_proj.weight", "lm_head.weight"})

    def test_text_layers_from_text_config(self):
        self.assertEqual(gs.text_layers({"text_config": {"num_hidden_layers": 45}}), 45)
        with self.assertRaises(ValueError):
            gs.text_layers({"text_config": {}})

    def test_journal_path_is_the_converters(self):
        self.assertEqual(gs.journal_path(Path("c/GLM.cnq")).name, "GLM.cnq.journal.jsonl")

    def test_same_path(self):
        r = Path(Path.cwd().anchor) / "r"          # absolute on Windows (drive) and POSIX
        x = str(r / "converter" / "x.cnq")
        self.assertTrue(gs.same_path(str(r / "a" / ".." / "converter" / "x.cnq"), x))
        self.assertTrue(gs.same_path("converter/x.cnq", x, bases=(r,)))
        self.assertFalse(gs.same_path("converter/y.cnq", x, bases=(r,)))
        self.assertFalse(gs.same_path("", x, bases=(r,)))

    def test_delete_verdict_needs_every_guard(self):
        need, written = {"a", "b"}, {"a", "b", "c"}
        self.assertEqual(gs.delete_verdict("s", True, True, True, True, need, written)[0], True)
        self.assertEqual(gs.delete_verdict("s", True, True, False, True, need, written), (False, "no .done"))
        self.assertFalse(gs.delete_verdict("s", True, True, True, False, need, written)[0])
        self.assertFalse(gs.delete_verdict("s", True, False, True, True, need, written)[0])
        self.assertFalse(gs.delete_verdict("s", True, True, True, True, need, None)[0])
        ok, why = gs.delete_verdict("s", True, True, True, True, need, {"a"})
        self.assertFalse(ok)
        self.assertIn("1 of 2 tensors not journalled (first b)", why)


class JournalPrefix(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = World(self.tmp.name)

    def test_the_verified_prefix_holds_every_record(self):
        self.w.convert(["a", "b", "c"], pads={"b": 7})
        j = gs.Journal(gs.journal_path(self.w.out), self.w.out)
        self.assertEqual(j.refresh(), {"a", "b", "c"})
        self.assertIsNone(j.stuck)

    def test_a_corrupt_container_byte_stops_the_prefix_there(self):
        self.w.convert(["a", "b", "c"])
        raw = bytearray(self.w.out.read_bytes())
        raw[12 + 96 + 5] ^= 1                     # inside "b" (each payload is 96 B)
        self.w.out.write_bytes(bytes(raw))
        j = gs.Journal(gs.journal_path(self.w.out), self.w.out)
        self.assertEqual(j.refresh(), {"a"})
        self.assertEqual(j.stuck[0], 1)
        self.assertIn("journal says", j.stuck[1])

    def test_nonzero_alignment_bytes_stop_the_prefix(self):
        self.w.convert(["a", "b"], pads={"b": 4})
        raw = bytearray(self.w.out.read_bytes())
        raw[12 + 96 + 1] = 1
        self.w.out.write_bytes(bytes(raw))
        j = gs.Journal(gs.journal_path(self.w.out), self.w.out)
        self.assertEqual(j.refresh(), {"a"})

    def test_a_half_written_line_is_not_read_and_a_rewrite_starts_over(self):
        self.w.convert(["a", "b"])
        jp = gs.journal_path(self.w.out)
        with open(jp, "a") as f:
            f.write('{"seq": 2, "pad')                  # a kill mid-line
        j = gs.Journal(jp, self.w.out)
        self.assertEqual(j.refresh(), {"a", "b"})
        self.w.convert(["x"])                           # a resume that rewrote the journal
        self.assertEqual(j.refresh(), {"x"})

    def test_a_foreign_file_is_not_a_journal(self):
        jp = gs.journal_path(self.w.out)
        self.w.out.write_bytes(b"CNQ1" + bytes(8))
        jp.write_text('{"something": "else"}\n')
        with self.assertRaisesRegex(ValueError, "not a crow-nest converter journal"):
            gs.Journal(jp, self.w.out).refresh()

    def test_the_trailer_names_the_tensors(self):
        self.w.convert(["a", "b"], trailer=True)
        self.assertEqual(gs.read_trailer(self.w.out), {"a", "b"})
        self.w.out.write_bytes(b"CNQ1" + bytes(20))
        self.assertIsNone(gs.read_trailer(self.w.out))


class DeleteBehind(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.w = World(self.tmp.name)
        self.p1 = self.w.dest / shard_name(1)

    def test_a_converted_shard_is_deleted_once_and_marked(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1)
        st = self.w.stage()
        self.assertEqual(st.one_pass(), 1)
        self.assertFalse(self.p1.exists())
        rec = json.loads(fg.deleted_marker_of(self.p1).read_text())
        self.assertEqual((rec["shard"], rec["tensors"], rec["journal_records"]), (shard_name(1), 2, 2))
        self.assertTrue(fg.marker_of(self.p1).exists())                       # .verified kept
        self.assertTrue((self.w.dest / (shard_name(1) + ".done")).exists())   # .done kept
        self.assertTrue(any(l.startswith(f"DELETED {shard_name(1)}") for l in self.w.logs))
        # idempotent: a second pass and a restarted supervisor change nothing
        self.assertEqual(st.one_pass(), 0)
        self.assertEqual(self.w.stage().one_pass(), 0)

    def test_without_done_a_shard_is_never_deleted(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.assertEqual(self.w.stage().one_pass(), 0)
        self.assertTrue(self.p1.exists())
        self.assertFalse(fg.deleted_marker_of(self.p1).exists())

    def test_a_done_of_another_container_is_not_enough(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1, out=self.w.out.with_name("step6.cnq"))
        self.assertEqual(self.w.stage().one_pass(), 0)
        self.assertTrue(self.p1.exists())
        self.assertTrue(any("another container" in l for l in self.w.logs))

    def test_a_relative_out_in_done_resolves_against_the_repo_root(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1, out="converter/GLM.cnq")
        self.assertEqual(self.w.stage().one_pass(), 1)

    def test_a_tensor_missing_from_the_journal_keeps_the_shard(self):
        self.w.fetch(1)
        self.w.convert(ALL_1[:1])                 # the FP8 weight (and its scale) not yet written
        self.w.done(1)
        self.assertEqual(self.w.stage().one_pass(), 0)
        self.assertTrue(self.p1.exists())
        self.assertTrue(any("1 of 2 tensors not journalled" in l for l in self.w.logs))

    def test_a_journalled_tensor_whose_bytes_do_not_rehash_keeps_the_shard(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1)
        raw = bytearray(self.w.out.read_bytes())
        raw[-1] ^= 1                              # the last tensor's bytes
        self.w.out.write_bytes(bytes(raw))
        self.assertEqual(self.w.stage().one_pass(), 0)
        self.assertTrue(self.p1.exists())

    def test_a_shard_whose_marker_disagrees_with_the_hf_record_is_kept(self):
        self.w.fetch(1, sha="f" * 64)
        self.w.convert(ALL_1)
        self.w.done(1)
        self.assertEqual(self.w.stage().one_pass(), 0)
        self.assertTrue(self.p1.exists())
        self.assertTrue(any("not verified" in l for l in self.w.logs))

    def test_omitted_only_shards_go_after_done_and_other_shards_stay(self):
        for i in (1, 2, 3, 4):
            self.w.fetch(i)
        self.w.convert(ALL_1 + [L + "1.self_attn.o_proj.weight"])
        for i in (1, 2, 3):
            self.w.done(i)
        self.assertEqual(self.w.stage().one_pass(), 3)
        self.assertTrue((self.w.dest / shard_name(4)).exists())           # no .done: stays

    def test_a_kill_between_marker_and_delete_is_completed(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1)
        fg.deleted_marker_of(self.p1).write_text("{}")                     # marker written, then killed
        self.assertEqual(self.w.stage().one_pass(), 1)
        self.assertFalse(self.p1.exists())

    def test_after_the_trailer_the_index_is_the_record(self):
        self.w.fetch(1)
        self.w.convert(ALL_1, trailer=True)
        self.w.done(1)
        self.assertEqual(self.w.stage().one_pass(), 1)

    def test_dry_run_deletes_nothing(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1)
        self.w.stage(dry_run=True).one_pass()
        self.assertTrue(self.p1.exists())
        self.assertFalse(fg.deleted_marker_of(self.p1).exists())
        self.assertTrue(any(l.startswith(f"DRY {shard_name(1)}") for l in self.w.logs))

    def test_the_status_line(self):
        self.w.fetch(1)
        self.w.fetch(2)
        (self.w.dest / (shard_name(3) + ".part")).write_bytes(b"x" * 5)
        self.w.convert(ALL_1)
        self.w.done(1)
        self.w.stage().one_pass()
        line = gs.status_line(self.w.dest, self.w.out, self.w.table(), fg.check_table(self.w.table()))
        self.assertIn("verified 2/62, on disk 1, done 1/62, deleted 1/62", line)
        self.assertIn(f"shards on disk {len(self.w.data[2])} B + .part 5 B", line)
        self.assertIn(f"container {self.w.out.stat().st_size:,} B", line)
        self.assertIn("journal 2 tensors", line)

    def test_the_fetcher_skips_what_the_supervisor_deleted(self):
        self.w.fetch(1)
        self.w.convert(ALL_1)
        self.w.done(1)
        self.w.stage().one_pass()
        logs, calls = [], []
        ctx = {"log": logs.append, "lock": None, "stats": fg.Stats(), "curl": "curl", "wait_space": True}
        orig = fg.run_curl
        fg.run_curl = lambda *a, **k: calls.append(a) or (22, {"http": 404, "retries": 0, "bytes": 0, "exit": 22}, "")
        try:
            rc = fg.mode_shards(self.w.dest, self.w.table(), [shard_name(1)], ctx)
        finally:
            fg.run_curl = orig
        self.assertEqual((rc, calls), (0, []))


if __name__ == "__main__":
    unittest.main(verbosity=1)
