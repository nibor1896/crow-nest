#!/usr/bin/env python3
"""#76: the unit tests of the fetch tool's pure parts.

  python tools/test_fetch_dense_originals.py     # no network, no venv, no GPU

Three things in `tools/fetch-dense-originals.py` decide whether the download is right, and
all three are pure: reading a safetensors header out of a prefix of bytes, coalescing tensor
ranges into requests, and writing a header whose offsets are contiguous. The network half is
tested against a fake opener that never opens a socket - what has to hold there is the
contract (a 206 of exactly the asked length, a retry that honours `Retry-After`, a 200
refused), not Hugging Face's behaviour.

The module is loaded by path because the tool's file name carries a hyphen.
"""

import importlib.util
import io
import json
import struct
import tempfile
import unittest
import urllib.error
from datetime import datetime, timedelta, timezone
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("fetch_dense_originals", TOOLS / "fetch-dense-originals.py")
fd = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fd)


def safetensors_bytes(tensors, metadata=None):
    """A real little safetensors file: {name: (dtype, shape, payload)} in the given order."""
    header = {}
    if metadata is not None:
        header["__metadata__"] = metadata
    body = b""
    for name, (dtype, shape, payload) in tensors.items():
        header[name] = {"dtype": dtype, "shape": list(shape), "data_offsets": [len(body), len(body) + len(payload)]}
        body += payload
    blob = json.dumps(header).encode("utf-8")
    return struct.pack("<Q", len(blob)) + blob + body


class HeaderParsing(unittest.TestCase):
    def test_a_whole_file_parses_to_its_tensors_and_data_start(self):
        raw = safetensors_bytes({"a": ("BF16", [2, 2], b"\x01" * 8)}, metadata={"format": "pt"})
        header, data_start = fd.parse_safetensors_header(raw)
        self.assertEqual(header["a"]["dtype"], "BF16")
        self.assertEqual(header["a"]["data_offsets"], [0, 8])
        self.assertEqual(raw[data_start:], b"\x01" * 8)

    def test_a_prefix_that_covers_the_header_is_enough(self):
        # this is the whole point of the tool: the first request never sees the payload
        raw = safetensors_bytes({"a": ("BF16", [8192], b"\x02" * 16384)})
        header_len = struct.unpack("<Q", raw[:8])[0]
        header, data_start = fd.parse_safetensors_header(raw[: 8 + header_len])
        self.assertEqual(data_start, 8 + header_len)
        self.assertIn("a", header)

    def test_a_short_prefix_raises_instead_of_guessing(self):
        raw = safetensors_bytes({"a": ("BF16", [4], b"\x03" * 8)})
        with self.assertRaises(ValueError):
            fd.parse_safetensors_header(raw[:12])
        with self.assertRaises(ValueError):
            fd.parse_safetensors_header(b"\x00\x00\x00")

    def test_metadata_is_a_key_like_any_other(self):
        raw = safetensors_bytes({"a": ("F32", [1], b"\x00" * 4)}, metadata={"format": "pt"})
        header, _ = fd.parse_safetensors_header(raw)
        self.assertEqual(header["__metadata__"], {"format": "pt"})

    def test_a_header_that_is_not_an_object_is_refused(self):
        blob = b"[1,2,3]"
        with self.assertRaises(ValueError):
            fd.parse_safetensors_header(struct.pack("<Q", len(blob)) + blob)


class Selection(unittest.TestCase):
    LINES = [
        json.dumps({"name": "m.layers.0.linear_attn.out_proj.weight", "section": "text", "dtype": "nvfp4", "n": 64}),
        json.dumps({"name": "m.layers.0.mlp.experts.down_proj", "section": "text", "dtype": "nvfp4", "n": 128}),
        json.dumps({"name": "m.layers.0.input_layernorm.weight", "section": "text", "dtype": "bf16", "n": 8}),
        json.dumps({"name": "m.visual.blocks.0.attn.qkv.weight", "section": "vit", "dtype": "nvfp4", "n": 64}),
        json.dumps({"record": "section_summary", "section": "text", "dtype": "nvfp4", "n": 999}),
        "",
    ]

    def test_only_dense_text_nvfp4_survives(self):
        got = fd.select_dense_tensors(self.LINES)
        self.assertEqual([t["name"] for t in got], ["m.layers.0.linear_attn.out_proj.weight"])
        self.assertEqual(got[0]["n"], 64)

    def test_a_summary_record_has_no_name_and_is_skipped(self):
        # the four `record: section_summary` lines of the real sidecar carry an `n` of
        # billions; counting one of them would put the plan out by the whole model
        self.assertEqual(sum(t["n"] for t in fd.select_dense_tensors(self.LINES)), 64)

    def test_the_order_of_the_sidecar_is_kept(self):
        lines = [json.dumps({"name": n, "section": "text", "dtype": "nvfp4", "n": 64}) for n in ("z", "a", "m")]
        self.assertEqual([t["name"] for t in fd.select_dense_tensors(lines)], ["z", "a", "m"])


class Coalescing(unittest.TestCase):
    def items(self, *pairs):
        return [{"begin": a, "end": b} for a, b in pairs]

    def test_touching_ranges_become_one_request(self):
        g = fd.coalesce_ranges(self.items((0, 100), (100, 200)), max_gap=0, max_span=1 << 30)
        self.assertEqual(len(g), 1)
        self.assertEqual((g[0]["begin"], g[0]["end"]), (0, 200))
        self.assertEqual(len(g[0]["members"]), 2)

    def test_a_gap_above_the_threshold_splits(self):
        g = fd.coalesce_ranges(self.items((0, 100), (1101, 1200)), max_gap=1000, max_span=1 << 30)
        self.assertEqual([(x["begin"], x["end"]) for x in g], [(0, 100), (1101, 1200)])

    def test_a_gap_at_the_threshold_still_merges(self):
        g = fd.coalesce_ranges(self.items((0, 100), (1100, 1200)), max_gap=1000, max_span=1 << 30)
        self.assertEqual([(x["begin"], x["end"]) for x in g], [(0, 1200)])

    def test_the_span_cap_wins_over_the_gap(self):
        g = fd.coalesce_ranges(self.items((0, 100), (110, 900)), max_gap=1000, max_span=500)
        self.assertEqual([(x["begin"], x["end"]) for x in g], [(0, 100), (110, 900)])

    def test_input_order_does_not_matter(self):
        g = fd.coalesce_ranges(self.items((300, 400), (0, 100), (100, 200)), max_gap=0, max_span=1 << 30)
        self.assertEqual([(x["begin"], x["end"]) for x in g], [(0, 200), (300, 400)])

    def test_every_member_is_kept_exactly_once(self):
        pairs = [(0, 10), (10, 20), (5000, 5100), (5100, 5200), (5200, 5300)]
        g = fd.coalesce_ranges(self.items(*pairs), max_gap=64, max_span=1 << 30)
        members = [(m["begin"], m["end"]) for grp in g for m in grp["members"]]
        self.assertEqual(sorted(members), sorted(pairs))

    def test_the_empty_shard_is_no_request(self):
        self.assertEqual(fd.coalesce_ranges([], max_gap=1, max_span=1), [])


class HeaderWriting(unittest.TestCase):
    ENTRIES = [
        {"name": "a.weight", "dtype": "BF16", "shape": [4, 8]},
        {"name": "b.weight", "dtype": "BF16", "shape": [64]},
        {"name": "c.weight", "dtype": "F32", "shape": [2, 2]},
    ]

    def test_offsets_are_contiguous_with_no_holes(self):
        head, offsets, data = fd.build_safetensors_header(self.ENTRIES)
        self.assertEqual(offsets, [0, 64, 192])
        self.assertEqual(data, 208)
        parsed, _ = fd.parse_safetensors_header(head + b"\x00" * data)
        spans = sorted(v["data_offsets"] for k, v in parsed.items() if k != "__metadata__")
        self.assertEqual(spans, [[0, 64], [64, 192], [192, 208]])
        for lo, hi in zip(spans, spans[1:]):
            self.assertEqual(lo[1], hi[0], "a hole between two tensors")

    def test_the_file_it_writes_parses_back_to_the_same_tensors(self):
        head, offsets, data = fd.build_safetensors_header(self.ENTRIES, metadata={"format": "pt", "rev": "abc"})
        blob = head + bytes(range(256))[:data]
        parsed, data_start = fd.parse_safetensors_header(blob)
        self.assertEqual(data_start, len(head))
        self.assertEqual(parsed["__metadata__"], {"format": "pt", "rev": "abc"})
        self.assertEqual(parsed["a.weight"]["shape"], [4, 8])
        for e, off in zip(self.ENTRIES, offsets):
            lo, hi = parsed[e["name"]]["data_offsets"]
            self.assertEqual(lo, off)
            self.assertEqual(blob[data_start + lo : data_start + hi], blob[len(head) + lo : len(head) + hi])

    def test_the_header_length_is_a_multiple_of_eight(self):
        # the reference writer pads the JSON with spaces so the payload starts aligned
        for names in (["x"], ["x", "yy"], ["a" * 37]):
            entries = [{"name": n, "dtype": "BF16", "shape": [2]} for n in names]
            head, _, _ = fd.build_safetensors_header(entries)
            self.assertEqual(struct.unpack("<Q", head[:8])[0] % 8, 0)
            self.assertEqual(len(head) % 8, 0)

    def test_metadata_values_are_strings(self):
        head, _, _ = fd.build_safetensors_header(self.ENTRIES, metadata={"n": 7})
        parsed, _ = fd.parse_safetensors_header(head + b"\x00" * 208)
        self.assertEqual(parsed["__metadata__"], {"n": "7"})

    def test_an_empty_selection_writes_a_header_and_no_payload(self):
        head, offsets, data = fd.build_safetensors_header([])
        self.assertEqual((offsets, data), ([], 0))
        parsed, start = fd.parse_safetensors_header(head)
        self.assertEqual((parsed, start), ({}, len(head)))


class RetryAfter(unittest.TestCase):
    def test_a_plain_number_is_seconds(self):
        self.assertEqual(fd.parse_retry_after("30"), 30.0)
        self.assertEqual(fd.parse_retry_after(" 5 "), 5.0)

    def test_an_http_date_becomes_the_distance_to_now(self):
        now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
        when = now + timedelta(seconds=90)
        header = when.strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.assertAlmostEqual(fd.parse_retry_after(header, now=now), 90.0, delta=1.0)

    def test_a_date_in_the_past_never_goes_negative(self):
        now = datetime(2026, 9, 18, 12, 0, 0, tzinfo=timezone.utc)
        header = (now - timedelta(hours=1)).strftime("%a, %d %b %Y %H:%M:%S GMT")
        self.assertEqual(fd.parse_retry_after(header, now=now), 0.0)

    def test_nothing_and_nonsense_are_None(self):
        self.assertIsNone(fd.parse_retry_after(None))
        self.assertIsNone(fd.parse_retry_after(""))
        self.assertIsNone(fd.parse_retry_after("soon"))

    def test_the_backoff_is_bounded(self):
        waits = [fd.backoff_seconds(a) for a in range(1, 12)]
        self.assertEqual(waits[:4], [2.0, 4.0, 8.0, 16.0])
        self.assertTrue(all(w <= 60.0 for w in waits))


class FakeResponse(io.BytesIO):
    def __init__(self, data, status=206):
        super().__init__(data)
        self.status = status

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()
        return False


class FetcherContract(unittest.TestCase):
    """The network half against a fake opener - no socket is ever opened."""

    def opener_for(self, script):
        calls = []

        def opener(req, timeout=None):
            calls.append(req.headers.get("Range") or req.headers.get("range"))
            action = script.pop(0)
            if isinstance(action, Exception):
                raise action
            return FakeResponse(*action) if isinstance(action, tuple) else FakeResponse(action)

        return opener, calls

    def test_a_206_of_the_asked_length_is_returned(self):
        opener, calls = self.opener_for([b"0123456789"])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None)
        self.assertEqual(f.get("shard", 100, 110), b"0123456789")
        self.assertEqual(calls, ["bytes=100-109"])
        self.assertEqual((f.requests, f.retries, f.bytes), (1, 0, 10))

    def test_a_200_is_refused_and_not_retried_into_a_whole_shard(self):
        opener, _ = self.opener_for([(b"x" * 10, 200)] * 6)
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None, max_tries=2)
        with self.assertRaises(RuntimeError):
            f.get("shard", 0, 10)

    def test_a_short_body_is_a_failure_not_a_silent_truncation(self):
        opener, _ = self.opener_for([b"short", b"short"])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None, max_tries=2)
        with self.assertRaises(RuntimeError):
            f.get("shard", 0, 10)

    def test_a_429_is_retried_after_the_header_says_so(self):
        slept = []
        err = urllib.error.HTTPError("u", 429, "slow down", {"Retry-After": "7"}, None)
        opener, _ = self.opener_for([err, b"abcd"])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=slept.append)
        self.assertEqual(f.get("shard", 0, 4), b"abcd")
        self.assertEqual(slept, [7.0])
        self.assertEqual((f.retries, f.rate_limited), (1, 1))

    def test_a_404_is_not_retried(self):
        err = urllib.error.HTTPError("u", 404, "gone", {}, None)
        opener, calls = self.opener_for([err])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None)
        with self.assertRaises(urllib.error.HTTPError):
            f.get("shard", 0, 4)
        self.assertEqual(len(calls), 1)

    def test_a_dropped_connection_is_retried_with_backoff(self):
        slept = []
        opener, _ = self.opener_for([ConnectionResetError("reset"), b"ok"])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=slept.append)
        self.assertEqual(f.get("shard", 0, 2), b"ok")
        self.assertEqual(slept, [2.0])

    def test_the_last_try_raises(self):
        opener, calls = self.opener_for([ConnectionResetError("reset")] * 3)
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None, max_tries=3)
        with self.assertRaises(ConnectionResetError):
            f.get("shard", 0, 2)
        self.assertEqual(len(calls), 3)

    def test_a_header_is_read_from_the_cache_the_second_time(self):
        raw = safetensors_bytes({"a": ("BF16", [4], b"\x01" * 8)})
        opener, calls = self.opener_for([raw + b"\x00" * (fd.HEADER_PROBE - len(raw))])
        f = fd.RangeFetcher("https://example/", opener=opener, sleep=lambda s: None)
        with tempfile.TemporaryDirectory() as d:
            h1, s1 = f.read_header("shard", cache_dir=d)
            h2, s2 = f.read_header("shard", cache_dir=d)
        self.assertEqual((h1, s1), (h2, s2))
        self.assertEqual(len(calls), 1, "the second read must not touch the network")


class Sha256OfASlice(unittest.TestCase):
    def test_it_hashes_exactly_the_window(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"AAAABBBBCCCC")
            self.assertEqual(fd.sha256_of(p, 4, 4), hashlib.sha256(b"BBBB").hexdigest())
            self.assertEqual(fd.sha256_of(p, 0, 12), hashlib.sha256(b"AAAABBBBCCCC").hexdigest())

    def test_a_window_past_the_end_is_an_error(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "f.bin"
            p.write_bytes(b"AAAA")
            with self.assertRaises(RuntimeError):
                fd.sha256_of(p, 0, 8)




class ExpertSelection(unittest.TestCase):
    """#79: the routed-expert half of the same derivation."""

    LINES = [
        json.dumps({"name": "m.layers.0.mlp.experts.gate_up_proj", "section": "text", "dtype": "nvfp4", "n": 16}),
        json.dumps({"name": "m.layers.7.mlp.experts.gate_up_proj", "section": "text", "dtype": "nvfp4", "n": 32}),
        json.dumps({"name": "m.layers.7.mlp.experts.down_proj", "section": "text", "dtype": "nvfp4", "n": 64}),
        json.dumps({"name": "m.layers.7.linear_attn.out_proj.weight", "section": "text", "dtype": "nvfp4", "n": 8}),
        json.dumps({"name": "m.layers.7.mlp.experts.gate_up_proj", "section": "vit", "dtype": "nvfp4", "n": 999}),
        json.dumps({"record": "section_summary", "section": "text", "dtype": "nvfp4", "n": 999}),
    ]

    def test_only_the_asked_layers_survive(self):
        got = fd.select_expert_tensors(self.LINES, [7])
        self.assertEqual([t["name"] for t in got],
                         ["m.layers.7.mlp.experts.gate_up_proj", "m.layers.7.mlp.experts.down_proj"])
        self.assertEqual([t["layer"] for t in got], [7, 7])
        self.assertEqual(sum(t["n"] for t in got), 96)

    def test_the_dense_rule_and_the_expert_rule_partition_the_text_nvfp4_tensors(self):
        # the two selections may never overlap and may never leave a tensor out: that is what
        # makes "the experts were not touched" of #76 true and #79's file the rest of it
        dense = {t["name"] for t in fd.select_dense_tensors(self.LINES)}
        experts = {t["name"] for t in fd.select_expert_tensors(self.LINES, range(64))}
        self.assertEqual(dense & experts, set())
        self.assertEqual(dense | experts, {
            "m.layers.0.mlp.experts.gate_up_proj", "m.layers.7.mlp.experts.gate_up_proj",
            "m.layers.7.mlp.experts.down_proj", "m.layers.7.linear_attn.out_proj.weight"})

    def test_a_layer_nobody_asked_for_is_not_in_the_plan(self):
        self.assertEqual(fd.select_expert_tensors(self.LINES, [3]), [])

    def test_the_layer_number_comes_out_of_the_name(self):
        self.assertEqual(fd.layer_of("model.language_model.layers.43.mlp.experts.down_proj"), 43)
        self.assertEqual(fd.layer_of("model.language_model.layers.0.ple.conv1d.weight"), 0)
        self.assertIsNone(fd.layer_of("model.visual.blocks.3.attn.qkv.weight"))
        self.assertIsNone(fd.layer_of("model.language_model.layers.x.foo"))
        self.assertIsNone(fd.layer_of("model.language_model.layers."))

    def test_the_layer_spec_takes_a_list_and_a_range(self):
        self.assertEqual(fd.parse_layers("1,7,13"), [1, 7, 13])
        self.assertEqual(fd.parse_layers("13,1,7,1"), [1, 7, 13])
        self.assertEqual(fd.parse_layers("2-5,9"), [2, 3, 4, 5, 9])
        self.assertEqual(fd.parse_layers(" 3 "), [3])

    def test_a_spec_that_is_not_layers_is_refused(self):
        for bad in ("", ",", "a", "5-2", "-1", "1..3"):
            with self.assertRaises(ValueError, msg=bad):
                fd.parse_layers(bad)


class ChunkedRequests(unittest.TestCase):
    """#79: a 3.36 GB tensor is one range too big to hold, so a span is cut into requests.

    The contract that matters is that the chunk size is invisible in the result: the same
    bytes land at the same offsets and every sha256 is the same. Run against a fake shard,
    through the REAL `fetch_group`, so the output file and its manifest are the real ones.
    """

    SHARD = "model-00004-of-00131.safetensors"

    def setUp(self):
        payload_a = bytes((i * 7 + 3) & 0xFF for i in range(4096))
        payload_b = bytes((i * 11 + 5) & 0xFF for i in range(2048))
        self.raw = safetensors_bytes({
            "m.layers.7.mlp.experts.gate_up_proj": ("BF16", [2, 1024], payload_a),
            "m.layers.7.mlp.experts.down_proj": ("BF16", [1024], payload_b),
        })
        self.payloads = {"m.layers.7.mlp.experts.gate_up_proj": payload_a,
                         "m.layers.7.mlp.experts.down_proj": payload_b}
        self.requests = 0

    def opener(self, req, timeout=None):
        self.requests += 1
        lo, hi = req.headers["Range"].split("=")[1].split("-")
        lo, hi = int(lo), int(hi) + 1
        blob = self.raw[lo:hi]
        if len(blob) < hi - lo:  # the 256 KiB header probe asks past the end of this little shard
            blob += b"\x00" * (hi - lo - len(blob))
        return FakeResponse(blob)

    def group_for(self, out_dir):
        return {
            "layer": 7,
            "selection": [{"name": n, "n": len(self.payloads[n]) // 2, "layer": 7, "shard": self.SHARD}
                          for n in sorted(self.payloads)],
            "out_file": Path(out_dir) / "layer-07.safetensors",
            "manifest": Path(out_dir) / "layer-07.manifest.json",
            "header_cache": Path(out_dir) / ".shard-headers",
            "metadata": {"format": "pt"},
            "rule": "test",
        }

    def args_for(self, chunk):
        import argparse
        return argparse.Namespace(max_tries=3, max_gap=fd.DEFAULT_MAX_GAP, max_span=fd.DEFAULT_MAX_SPAN,
                                  chunk_bytes=chunk, skip_verify=False, repo="r", revision="v",
                                  sidecar=Path("s.jsonl"))

    def fetch_with(self, chunk, out_dir):
        real = fd.RangeFetcher
        opener = self.opener
        fd.RangeFetcher = lambda url, **kw: real(url, opener=opener, sleep=lambda s: None, **kw)
        try:
            rc, stats = fd.fetch_group(self.group_for(out_dir), self.args_for(chunk), {}, "https://example/")
        finally:
            fd.RangeFetcher = real
        self.assertEqual(rc, 0)
        return stats

    def test_the_chunk_size_changes_the_request_count_and_nothing_else(self):
        results = {}
        for chunk in (1 << 20, 1024, 100):
            with tempfile.TemporaryDirectory() as d:
                stats = self.fetch_with(chunk, d)
                out = (Path(d) / "layer-07.safetensors").read_bytes()
                man = json.loads((Path(d) / "layer-07.manifest.json").read_text())
                results[chunk] = (out, {t["name"]: t["sha256"] for t in man["tensors"]}, stats["requests"])
        big, small, tiny = results[1 << 20], results[1024], results[100]
        self.assertEqual(big[0], small[0])
        self.assertEqual(big[0], tiny[0])
        self.assertEqual(big[1], small[1])
        self.assertEqual(big[1], tiny[1])
        self.assertLess(big[2], tiny[2], "a 100 B chunk must cost more requests than a 1 MiB one")

    def test_the_payload_lands_where_the_written_header_says(self):
        import hashlib
        with tempfile.TemporaryDirectory() as d:
            self.fetch_with(1000, d)
            out = (Path(d) / "layer-07.safetensors").read_bytes()
            header, data_start = fd.parse_safetensors_header(out)
            man = json.loads((Path(d) / "layer-07.manifest.json").read_text())
            recorded = {t["name"]: t["sha256"] for t in man["tensors"]}
            for name, payload in self.payloads.items():
                lo, hi = header[name]["data_offsets"]
                self.assertEqual(out[data_start + lo : data_start + hi], payload, name)
                self.assertEqual(recorded[name], hashlib.sha256(payload).hexdigest(), name)

    def test_a_second_run_over_the_same_directory_fetches_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            first = self.fetch_with(1 << 20, d)
            self.requests = 0
            second = self.fetch_with(1 << 20, d)
            self.assertEqual(first["fetched"], 2)
            self.assertEqual(second["fetched"], 0)
            self.assertEqual(second["verified"], 2)
            self.assertEqual(self.requests, 0, "the shard header is cached and the tensors are on disk")

if __name__ == "__main__":
    unittest.main(verbosity=2)
