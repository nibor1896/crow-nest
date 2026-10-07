#!/usr/bin/env python3
"""#154/#156/#157: unit tests of tools/fetch-glm.py - no network, no curl.

  python -I tools/test_fetch_glm.py

The pure parts (API table, shard and layer specs, shard selection from the index, header
parsing, resume decision, disk verdict, lock staleness, curl argv) are tested directly. The
outer download loop is tested against a fake curl that writes bytes into the `.part` file the
way a dropping connection does - what has to hold is the contract: resume until the byte count
equals the API size, never keep a file whose hash differs, delete an overlong file, refuse a
shard that would leave less than 20 GB, stop on a permanent HTTP error.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import hashlib
import importlib.util
import json
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fetch_glm", TOOLS / "fetch-glm.py")
fg = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fg)


def api_record(extra=None, n_shards=62):
    sib = [{"rfilename": n, "blobId": "0" * 40, "size": 10} for n in fg.SMALL_FILES]
    for i in range(1, n_shards + 1):
        sib.append({"rfilename": f"model-{i:05d}-of-{n_shards:05d}.safetensors", "blobId": "1" * 40,
                    "size": 1000 + i, "lfs": {"sha256": f"{i:064x}", "size": 1000 + i}})
    sib += extra or []
    return {"sha": fg.REVISION, "siblings": sib}


def safetensors_bytes(tensors):
    header, body = {}, b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": shape, "data_offsets": [len(body), len(body) + len(payload)]}
        body += payload
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + body


class ApiTable(unittest.TestCase):
    def test_table_has_sizes_hashes_and_62_sorted_shards(self):
        table = fg.file_table(api_record())
        shards = fg.check_table(table)
        self.assertEqual(len(shards), 62)
        self.assertEqual(shards[0], "model-00001-of-00062.safetensors")
        self.assertEqual(table[shards[1]]["sha256"], f"{2:064x}")
        self.assertIsNone(table["config.json"]["sha256"])

    def test_a_foreign_revision_is_refused(self):
        rec = api_record()
        rec["sha"] = "deadbeef"
        with self.assertRaisesRegex(ValueError, "not " + fg.REVISION):
            fg.file_table(rec)

    def test_a_shard_without_sha256_or_a_missing_shard_is_refused(self):
        rec = api_record()
        del rec["siblings"][-1]["lfs"]
        with self.assertRaisesRegex(ValueError, "no lfs.sha256"):
            fg.check_table(fg.file_table(rec))
        with self.assertRaisesRegex(ValueError, "61 shards"):
            fg.check_table(fg.file_table(api_record(n_shards=61)))

    def test_lfs_size_disagreeing_with_size_is_refused(self):
        rec = api_record()
        rec["siblings"][-1]["lfs"]["size"] = 1
        with self.assertRaisesRegex(ValueError, "lfs.size"):
            fg.file_table(rec)


class Specs(unittest.TestCase):
    shards = [f"model-{i:05d}-of-00062.safetensors" for i in range(1, 63)]

    def test_shard_numbers_ranges_and_names(self):
        self.assertEqual(fg.parse_shard_spec("2,1,61-62", self.shards),
                         [self.shards[0], self.shards[1], self.shards[60], self.shards[61]])
        self.assertEqual(fg.parse_shard_spec(self.shards[4], self.shards), [self.shards[4]])

    def test_shard_out_of_range_or_backwards_is_refused(self):
        for bad in ("0", "63", "5-3", ""):
            with self.assertRaises(ValueError):
                fg.parse_shard_spec(bad, self.shards)

    def test_layers(self):
        self.assertEqual(fg.parse_layers("0-3"), [0, 1, 2, 3])
        self.assertEqual(fg.parse_layers("5,0-1,5"), [0, 1, 5])
        for bad in ("", "3-1", "-1"):
            with self.assertRaises(ValueError):
                fg.parse_layers(bad)


class Selection(unittest.TestCase):
    weight_map = {
        "model.language_model.embed_tokens.weight": "s01",
        "model.language_model.layers.0.mlp.gate_proj.weight": "s01",
        "model.language_model.layers.1.mlp.gate_proj.weight": "s02",
        "model.language_model.layers.10.mlp.gate_proj.weight": "s02",   # 10 is not 1
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight": "s03",
        "model.language_model.layers.3.mlp.experts.0.down_proj.weight_scale_inv": "s04",
        "model.language_model.layers.2.input_layernorm.weight": "s03",
        "model.language_model.layers.45.enorm.weight": "s09",           # MTP, not wanted
        "model.language_model.norm.weight": "s07",
        "lm_head.weight": "s08",
        "model.visual.blocks.0.attn.qkv.weight": "s05",
        "model.visual.layers.0.norm.weight": "s05",
    }

    def test_layers_0_3_with_embed_head(self):
        shards, counts, reasons = fg.select_shards(self.weight_map, [0, 1, 2, 3], True)
        self.assertEqual(shards, ["s01", "s02", "s03", "s04", "s07", "s08"])
        self.assertEqual(counts, {"embed": 1, "layer 0": 1, "layer 1": 1, "layer 2": 1, "layer 3": 2,
                                  "head": 1, "final_norm": 1})
        self.assertEqual(reasons["s01"], ["embed", "layer 0"])

    def test_without_embed_head_only_layers(self):
        shards, _, _ = fg.select_shards(self.weight_map, [1], False)
        self.assertEqual(shards, ["s02"])

    def test_a_layer_without_tensors_is_refused(self):
        with self.assertRaisesRegex(ValueError, "layer 4"):
            fg.select_shards(self.weight_map, [3, 4], False)

    def test_kinds(self):
        self.assertIsNone(fg.lang_layer_of("model.visual.layers.0.norm.weight"))
        self.assertEqual(fg.lang_layer_of("model.language_model.layers.45.x"), 45)
        self.assertEqual(fg.tensor_kind("model.language_model.norm.weight"), "final_norm")
        self.assertIsNone(fg.tensor_kind("model.language_model.layers.0.input_layernorm.weight"))
        self.assertIsNone(fg.tensor_kind("model.visual.embed_tokens.weight"))


class Headers(unittest.TestCase):
    def test_prefix_that_covers_the_header_parses(self):
        raw = safetensors_bytes({"a": ("BF16", [2], b"\x01" * 4), "b": ("F8_E4M3", [4], b"\x02" * 4)})
        hl = struct.unpack("<Q", raw[:8])[0]
        header, start = fg.parse_safetensors_header(raw[:8 + hl])
        self.assertEqual(start, 8 + hl)
        self.assertEqual(fg.header_summary(header, start, len(raw)), (2, 8))

    def test_short_prefix_asks_for_exactly_the_missing_length(self):
        raw = safetensors_bytes({"a": ("BF16", [2], b"\x01" * 4)})
        hl = struct.unpack("<Q", raw[:8])[0]
        with self.assertRaises(fg.NeedMore) as ctx:
            fg.parse_safetensors_header(raw[:20])
        self.assertEqual(ctx.exception.total, 8 + hl)
        with self.assertRaises(fg.NeedMore) as ctx:
            fg.parse_safetensors_header(raw[:3])
        self.assertEqual(ctx.exception.total, 8)

    def test_a_header_that_does_not_tile_the_file_is_refused(self):
        raw = safetensors_bytes({"a": ("BF16", [2], b"\x01" * 4)})
        header, start = fg.parse_safetensors_header(raw)
        with self.assertRaisesRegex(ValueError, "file size"):
            fg.header_summary(header, start, len(raw) + 1)
        header["b"] = {"dtype": "BF16", "shape": [1], "data_offsets": [6, 8]}   # hole at 4..6
        with self.assertRaisesRegex(ValueError, "contiguous"):
            fg.header_summary(header, start, start + 8)

    def test_an_absurd_header_length_is_refused(self):
        with self.assertRaisesRegex(ValueError, "cap"):
            fg.parse_safetensors_header(struct.pack("<Q", 10 ** 9) + b"{}")


class Pure(unittest.TestCase):
    def test_git_blob_sha1_matches_git(self):
        # `git hash-object` of "hello\n" (git object format: "blob <size>\0" + bytes)
        self.assertEqual(fg.git_blob_sha1(b"hello\n", 6), "ce013625030ba8dba906f756967f9e9ca394464a")
        self.assertEqual(fg.git_blob_sha1([b"hel", b"lo\n"], 6), "ce013625030ba8dba906f756967f9e9ca394464a")

    def test_decide(self):
        self.assertEqual(fg.decide(0, 10), "resume")
        self.assertEqual(fg.decide(10, 10), "done")
        self.assertEqual(fg.decide(11, 10), "overlong")

    def test_backoff_grows_and_caps(self):
        seq, b = [], fg.BACKOFF_START
        for _ in range(8):
            seq.append(b)
            b = fg.next_backoff(b)
        self.assertEqual(seq[:6], [10, 20, 40, 80, 160, 300])
        self.assertEqual(max(seq), fg.BACKOFF_MAX)

    def test_disk_verdict_keeps_20_gb(self):
        self.assertEqual(fg.disk_verdict(30 * fg.GB, 10 * fg.GB), (True, 20 * fg.GB))
        ok, left = fg.disk_verdict(30 * fg.GB, 10 * fg.GB + 1)
        self.assertFalse(ok)

    def test_lock_staleness(self):
        self.assertFalse(fg.lock_is_stale(1000, 1000 + fg.LOCK_TTL))
        self.assertTrue(fg.lock_is_stale(1000, 1001 + fg.LOCK_TTL))

    def test_write_out(self):
        self.assertEqual(fg.parse_write_out("206 2 262144 0"), {"http": 206, "retries": 2, "bytes": 262144, "exit": 0})
        self.assertEqual(fg.parse_write_out(""), {"http": 0, "retries": 0, "bytes": 0, "exit": 0})

    def test_curl_argv_carries_the_owner_flags(self):
        argv = fg.curl_file_argv("curl", "https://x/f", "f.part")
        s = " ".join(argv)
        for flag in ("-L", "-f", "-C -", "--retry 20", "--retry-all-errors", "--retry-delay 10",
                     "--speed-limit", "--speed-time"):
            self.assertIn(flag, s)
        r = fg.curl_range_argv("curl", "https://x/f", "o", 100, 300)
        self.assertIn("100-299", r)
        self.assertEqual(r[r.index("--max-filesize") + 1], "200")
        self.assertNotIn("-C", r)

    def test_status_line(self):
        line = fg.status_line("a", 1000, 2000, False)
        self.assertRegex(line, r"1,000 / 2,000\s+verified no$")
        self.assertTrue(fg.status_line("a", 2, 2, True).endswith("verified yes"))


class FakeCurl:
    """Writes `payload` into the .part file in steps, like a connection that drops."""

    def __init__(self, payload, step, http=206, rc_drop=56):
        self.payload, self.step, self.http, self.rc_drop = payload, step, http, rc_drop
        self.calls = 0

    def __call__(self, argv, watch, log, lock, label, expected=None):
        self.calls += 1
        have = watch.stat().st_size if watch.exists() else 0
        chunk = self.payload[have:have + self.step]
        with open(watch, "ab") as f:
            f.write(chunk)
        done = have + len(chunk) >= len(self.payload)
        return (0 if done else self.rc_drop), {"http": self.http, "retries": 1, "bytes": len(chunk), "exit": 0}, ""


class OuterLoop(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dest = Path(self.tmp.name)
        self.logs = []
        self.ctx = {"log": self.logs.append, "lock": mock.Mock(), "stats": fg.Stats(), "curl": "curl"}
        self.payload = bytes(range(256)) * 40
        self.meta = {"size": len(self.payload), "blob": None,
                     "sha256": hashlib.sha256(self.payload).hexdigest()}
        p = mock.patch.object(fg.time, "sleep", lambda s: None)
        p.start()
        self.addCleanup(p.stop)
        self.addCleanup(self.tmp.cleanup)

    def run_with(self, fake, **ctx):
        self.ctx.update(ctx)
        with mock.patch.object(fg, "run_curl", fake):
            return fg.fetch_file("f.safetensors", self.meta, self.dest, self.ctx)

    def test_resumes_until_the_size_is_reached_then_verifies(self):
        fake = FakeCurl(self.payload, step=3000)
        self.assertTrue(self.run_with(fake))
        self.assertEqual(fake.calls, 4)
        self.assertEqual(self.ctx["stats"].restarts, 3)
        final = self.dest / "f.safetensors"
        self.assertEqual(final.read_bytes(), self.payload)
        self.assertFalse((self.dest / "f.safetensors.part").exists())
        self.assertTrue(fg.is_verified(final, self.meta))
        # a second run fetches nothing
        again = FakeCurl(self.payload, step=3000)
        self.assertTrue(self.run_with(again))
        self.assertEqual(again.calls, 0)

    def test_a_hash_mismatch_is_deleted_and_fetched_again(self):
        bad = bytearray(self.payload)
        bad[5] ^= 1
        fakes = [FakeCurl(bytes(bad), step=len(bad)), FakeCurl(self.payload, step=len(bad))]
        calls = iter([fakes[0], fakes[1]])
        current = {}

        def fake(*a, **k):
            if not (self.dest / "f.safetensors.part").exists():
                current["f"] = next(calls)
            return current["f"](*a, **k)

        self.assertTrue(self.run_with(fake))
        self.assertEqual(self.ctx["stats"].mismatches, 1)
        self.assertEqual((self.dest / "f.safetensors").read_bytes(), self.payload)

    def test_three_mismatches_stop_and_keep_nothing(self):
        bad = bytes(len(self.payload))
        self.assertFalse(self.run_with(FakeCurl(bad, step=len(bad))))
        self.assertEqual(self.ctx["stats"].mismatches, fg.MAX_MISMATCHES)
        self.assertFalse((self.dest / "f.safetensors").exists())
        self.assertFalse((self.dest / "f.safetensors.part").exists())

    def test_an_overlong_part_is_deleted(self):
        (self.dest / "f.safetensors.part").write_bytes(self.payload + b"x")
        fake = FakeCurl(self.payload, step=len(self.payload))
        self.assertTrue(self.run_with(fake))
        self.assertTrue(any("OVERLONG" in l for l in self.logs))

    def test_a_permanent_http_error_stops(self):
        fake = FakeCurl(self.payload, step=0, http=404, rc_drop=22)
        self.assertFalse(self.run_with(fake))
        self.assertEqual(fake.calls, 1)

    def test_a_shard_that_would_leave_less_than_20_gb_is_refused(self):
        usage = mock.Mock(free=fg.MIN_FREE_AFTER + len(self.payload) - 1)
        with mock.patch.object(fg.shutil, "disk_usage", return_value=usage):
            fake = FakeCurl(self.payload, step=len(self.payload))
            self.assertFalse(self.run_with(fake, disk_check=True))
        self.assertEqual(fake.calls, 0)
        self.assertTrue(self.ctx["refused"])

    def test_an_unmarked_final_file_is_rehashed_not_trusted(self):
        (self.dest / "f.safetensors").write_bytes(bytes(len(self.payload)))   # right size, wrong bytes
        fake = FakeCurl(self.payload, step=len(self.payload))
        calls = {"n": 0}

        def wrapped(*a, **k):
            calls["n"] += 1
            return fake(*a, **k)

        self.assertTrue(self.run_with(wrapped))
        self.assertEqual(self.ctx["stats"].mismatches, 1)
        self.assertEqual((self.dest / "f.safetensors").read_bytes(), self.payload)


if __name__ == "__main__":
    unittest.main(verbosity=1)
