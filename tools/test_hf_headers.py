#!/usr/bin/env python3
"""#240: unit tests of tools/hf_headers.py - no network, no curl.

  python -I tools/test_hf_headers.py

A fake hub stands in for huggingface.co behind `hh.curl_call` (the one place the helper runs
curl): it answers the API record, whole small files and byte ranges the way the hub does
(404 with a reason body, 401 for gated, 206 for ranges; with `-f` an HTTP error is exit 22 and no
body). What has to hold: one probe range per shard and a second one only for a longer header,
the converter's `--headers` format, refusals with the reason named, a rerun that reuses the
files and transfers 0 B, an exact byte counter, an HF_TOKEN that never reaches argv, a log line,
an error text or a file, and `tools/fetch-glm.py --headers` reading through the same core.

The modules are loaded by path: `python -I` puts no script directory on sys.path.
"""

import contextlib
import hashlib
import importlib.util
import io
import json
import os
import struct
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from urllib.parse import unquote

TOOLS = Path(__file__).resolve().parent


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, TOOLS / file)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


hh = load("hf_headers", "hf_headers.py")
fg = load("fetch_glm_for_hf_headers", "fetch-glm.py")

REPO = "acme/Tiny-MoE"
REV = "a" * 40
TOKEN = "hf_TESTtoken0123456789abcdef"


def st_bytes(tensors, pad=0):
    """A safetensors file: `tensors` = [(name, nbytes)], `pad` B of __metadata__ to lengthen the header."""
    header, body = {}, b""
    if pad:
        header["__metadata__"] = {"pad": "x" * pad}
    for i, (name, n) in enumerate(tensors):
        header[name] = {"dtype": "BF16", "shape": [n // 2], "data_offsets": [len(body), len(body) + n]}
        body += bytes([i % 251 + 1]) * n
    blob = json.dumps(header).encode()
    return struct.pack("<Q", len(blob)) + blob + body


def header_len(data):
    return 8 + struct.unpack("<Q", data[:8])[0]


def sibling(name, data):
    s = {"rfilename": name, "size": len(data),
         "blobId": hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()}
    if name.endswith((".safetensors", ".gguf")):
        s["lfs"] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "pointerSize": 134}
    return s


def sharded_repo(n=2, pad=0):
    shards = {f"model-{i:05d}-of-{n:05d}.safetensors": st_bytes([(f"layers.{i}.w", 64), (f"layers.{i}.b", 32)], pad)
              for i in range(1, n + 1)}
    wm = {}
    for name, data in shards.items():
        hdr, _ = hh.parse_safetensors_header(data)
        wm.update({t: name for t in hdr if t != "__metadata__"})
    files = {"config.json": b'{"model_type": "tiny_moe", "num_hidden_layers": 2}\n',
             "generation_config.json": b'{"eos_token_id": 2}\n',
             "README.md": b"# tiny\n",
             "model.safetensors.index.json": json.dumps({"metadata": {}, "weight_map": wm}).encode()}
    files.update(shards)
    return files


class FakeHub:
    """huggingface.co for one repo behind hh.curl_call. Records (argv, config) of every call."""

    def __init__(self, files, repo=REPO, revision=REV):
        self.files, self.repo, self.revision = dict(files), repo, revision
        self.api = {"id": repo, "sha": revision, "private": False,
                    "siblings": [sibling(n, d) for n, d in self.files.items()]}
        self.status = {}      # url substring -> (http, body)
        self.calls = []
        self.sent = 0         # body bytes written into -o files

    def __call__(self, argv, config=None):
        self.calls.append((list(argv), config))
        url = next(a for a in argv if a.startswith("https://"))
        fail = "-f" in argv
        for key, (http, body) in self.status.items():
            if key in url:
                return self.answer(argv, http, body, fail)
        api = f"https://huggingface.co/api/models/{self.repo}/revision/"
        if url.startswith(api):
            rev = url[len(api):].split("?")[0]
            if rev != self.revision:
                return self.answer(argv, 404, json.dumps({"error": f"Invalid rev id: {rev}"}).encode(), fail)
            return self.answer(argv, 200, json.dumps(self.api).encode(), fail)
        res = f"https://huggingface.co/{self.repo}/resolve/{self.revision}/"
        name = unquote(url[len(res):]) if url.startswith(res) else None
        if name not in self.files:
            return self.answer(argv, 404, b"Entry not found", fail)
        data = self.files[name]
        if "-r" in argv:
            b, e = (int(x) for x in argv[argv.index("-r") + 1].split("-"))
            return self.answer(argv, 206, data[b:e + 1], fail)
        return self.answer(argv, 200, data, fail)

    def answer(self, argv, http, body, fail):
        if http >= 400 and fail:
            return 22, f"{http} 0 0 22", f"curl: (22) The requested URL returned error: {http}"
        Path(argv[argv.index("-o") + 1]).write_bytes(body)
        self.sent += len(body)
        return 0, f"{http} 0 {len(body)} 0", ""

    def range_specs(self, shard):
        return [a[a.index("-r") + 1] for a, _ in self.calls if "-r" in a and any(x.endswith("/" + shard) for x in a)]


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.out = Path(self.tmp.name) / "out"
        self.logs = []
        env = mock.patch.dict(os.environ)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("HF_TOKEN", None)

    def run_hub(self, hub, revision=REV, repo=REPO):
        with mock.patch.object(hh, "curl_call", hub):
            return hh.fetch_model(repo, revision, self.out, curl="curl", log=self.logs.append)

    def refused(self, hub, pattern, **kw):
        with self.assertRaises(hh.HeaderError) as ctx:
            self.run_hub(hub, **kw)
        self.assertRegex(str(ctx.exception), pattern)
        return str(ctx.exception)


class HeaderRead(Base):
    def test_a_short_header_costs_one_probe_range_per_shard(self):
        hub = FakeHub(sharded_repo(2))
        r = self.run_hub(hub)
        for shard in r["shards"]:
            size = len(hub.files[shard])
            self.assertLess(size, hh.HEADER_PROBE)
            self.assertEqual(hub.range_specs(shard), [f"0-{size - 1}"])

    def test_the_probe_is_256_kib_on_a_large_shard(self):
        big = st_bytes([("w", hh.HEADER_PROBE * 2)])
        hub = FakeHub({"config.json": b"{}", "model.safetensors": big})
        self.run_hub(hub)
        self.assertEqual(hub.range_specs("model.safetensors"), [f"0-{hh.HEADER_PROBE - 1}"])

    def test_a_header_longer_than_the_probe_costs_a_second_range_for_the_rest(self):
        files = sharded_repo(1, pad=hh.HEADER_PROBE + 1000)
        hub = FakeHub(files)
        r = self.run_hub(hub)
        shard = r["shards"][0]
        hl = header_len(files[shard])
        self.assertGreater(hl, hh.HEADER_PROBE)
        self.assertEqual(hub.range_specs(shard), [f"0-{hh.HEADER_PROBE - 1}", f"{hh.HEADER_PROBE}-{hl - 1}"])
        rec = json.loads((self.out / "headers" / (shard + ".json")).read_text(encoding="utf-8"))
        self.assertEqual(rec["data_start"], hl)

    def test_a_header_over_the_cap_is_refused(self):
        bad = struct.pack("<Q", hh.MAX_HEADER + 1) + b"{}" + bytes(100)
        hub = FakeHub({"config.json": b"{}", "model.safetensors": bad})
        self.refused(hub, r"model\.safetensors: header length 100000001 B exceeds the safetensors cap")
        self.assertEqual(len(hub.range_specs("model.safetensors")), 1)

    def test_a_header_that_does_not_tile_its_shard_is_refused(self):
        data = st_bytes([("a", 8), ("b", 8)]) + b"\x00"     # one byte past the last tensor
        hub = FakeHub({"config.json": b"{}", "model.safetensors": data})
        self.refused(hub, r"model\.safetensors: header \d+ B \+ tensors 16 B != file size")
        self.assertFalse((self.out / "headers" / "model.safetensors.json").exists())

    def test_a_header_longer_than_its_file_is_refused(self):
        data = struct.pack("<Q", 5000) + b"{}"
        hub = FakeHub({"config.json": b"{}", "model.safetensors": data})
        self.refused(hub, r"header needs 5008 B, the file has 10 B")


class Refusals(Base):
    def test_an_api_answer_for_another_revision_is_refused(self):
        hub = FakeHub(sharded_repo())
        hub.api["sha"] = "b" * 40
        msg = self.refused(hub, "the API answers revision 'b{40}' for acme/Tiny-MoE, not the requested 'a{40}'")
        self.assertIn("full commit sha", msg)
        self.assertFalse((self.out / hh.API_FILE).exists())

    def test_404_names_repo_and_revision(self):
        hub = FakeHub(sharded_repo())
        msg = self.refused(hub, "HTTP 404 on the API record: repo acme/Tiny-MoE or revision 0{40} not found",
                           revision="0" * 40)
        self.assertIn("Invalid rev id: " + "0" * 40, msg)

    def test_401_says_accept_the_licence_and_set_hf_token(self):
        hub = FakeHub(sharded_repo())
        hub.status["/resolve/"] = (401, b"Access to model acme/Tiny-MoE is restricted. Please log in.")
        msg = self.refused(hub, r"HTTP 401 on config\.json: acme/Tiny-MoE is gated, private or does not exist")
        self.assertIn("Accept the licence on https://huggingface.co/acme/Tiny-MoE and set HF_TOKEN", msg)
        self.assertIn("server: Access to model acme/Tiny-MoE is restricted", msg)

    def test_403_on_the_api_is_refused_the_same_way(self):
        hub = FakeHub(sharded_repo())
        hub.status["/api/models/"] = (403, b'{"error": "forbidden"}')
        self.refused(hub, "HTTP 403 on the API record: .*Accept the licence")

    def test_no_index_and_no_single_file_is_refused_and_lists_files(self):
        hub = FakeHub({"README.md": b"# gguf\n", "Tiny-Q4_K_M.gguf": b"GGUF" + bytes(60),
                       "Tiny-Q8_0.gguf": b"GGUF" + bytes(90), ".gitattributes": b"*.gguf filter=lfs\n"})
        msg = self.refused(hub, "no model.safetensors.index.json and no single model.safetensors")
        self.assertIn("the repo has: Tiny-Q4_K_M.gguf, Tiny-Q8_0.gguf, .gitattributes, README.md", msg)
        self.assertEqual(len(hub.calls), 1)          # the API record only

    def test_an_index_naming_a_shard_the_api_does_not_list_is_refused(self):
        files = sharded_repo(2)
        del files["model-00002-of-00002.safetensors"]
        hub = FakeHub(files)
        self.refused(hub, "names 1 shard\\(s\\) the API does not list: model-00002-of-00002.safetensors")

    def test_a_small_file_that_does_not_match_its_record_is_refused(self):
        hub = FakeHub(sharded_repo())
        hub.files["config.json"] = b'{"model_type": "other"}\n'     # API record keeps the old size/blob
        self.refused(hub, "config.json: size 24 B != API size 51 B")
        self.assertFalse((self.out / "config.json").exists())


class Layouts(Base):
    def test_a_sharded_repo_writes_record_config_index_and_one_header_per_shard(self):
        files = sharded_repo(3)
        hub = FakeHub(files)
        r = self.run_hub(hub)
        names = [f"model-{i:05d}-of-00003.safetensors" for i in (1, 2, 3)]
        self.assertEqual(r["shards"], names)
        for f in ("config.json", "generation_config.json", "model.safetensors.index.json"):
            self.assertEqual((self.out / f).read_bytes(), files[f])
        self.assertEqual(json.loads((self.out / hh.API_FILE).read_bytes()), hub.api)
        self.assertFalse((self.out / "README.md").exists())
        self.assertEqual(sorted(p.name for p in (self.out / "headers").iterdir()), [n + ".json" for n in names])
        self.assertEqual((r["tensors"], r["header_bytes"]), (6, sum(header_len(files[n]) for n in names)))
        self.assertEqual((r["repo"], r["revision"], r["dir"], r["headers_dir"]),
                         (REPO, REV, str(self.out), str(self.out / "headers")))

    def test_a_single_file_repo_gets_headers_model_safetensors_json_and_no_index(self):
        data = st_bytes([("w", 40), ("b", 8)])
        hub = FakeHub({"config.json": b"{}", "model.safetensors": data})
        r = self.run_hub(hub)
        self.assertEqual(r["shards"], ["model.safetensors"])
        self.assertTrue((self.out / "headers" / "model.safetensors.json").exists())
        self.assertFalse((self.out / "model.safetensors.index.json").exists())
        self.assertFalse((self.out / "generation_config.json").exists())

    def test_headers_are_in_the_converters_format_byte_for_byte_as_fetch_glm_writes_them(self):
        files = sharded_repo(1)
        hub = FakeHub(files)
        r = self.run_hub(hub)
        shard = r["shards"][0]
        raw = (self.out / "headers" / (shard + ".json")).read_bytes()
        rec = json.loads(raw)
        self.assertEqual(list(rec), ["shard", "size", "data_start", "header"])
        hdr, start = hh.parse_safetensors_header(files[shard])
        self.assertEqual(rec, {"shard": shard, "size": len(files[shard]), "data_start": start, "header": hdr})
        # tools/fetch-glm.py mode_headers at 7b0fea8: json.dumps({...}) written as utf-8 text
        self.assertEqual(raw, json.dumps({"shard": shard, "size": len(files[shard]), "data_start": start,
                                          "header": hdr}).encode("utf-8"))


class Cache(Base):
    def test_a_rerun_reuses_every_file_and_transfers_0_bytes(self):
        hub = FakeHub(sharded_repo(2))
        first = self.run_hub(hub)
        self.assertGreater(first["bytes_transferred"], 0)
        again = FakeHub(sharded_repo(2))
        second = self.run_hub(again)
        self.assertEqual(again.calls, [])
        self.assertEqual((second["bytes_transferred"], second["requests"]), (0, 0))
        self.assertEqual(second["cached"], 1 + 3 + 2)    # record, config + generation + index, 2 headers
        self.assertEqual((second["shards"], second["tensors"], second["header_bytes"]),
                         (first["shards"], first["tensors"], first["header_bytes"]))

    def test_a_cached_file_that_does_not_verify_is_fetched_again(self):
        files = sharded_repo(2)
        self.run_hub(FakeHub(files))
        (self.out / "config.json").write_bytes(b'{"tampered": 1}')
        shard = "model-00002-of-00002.safetensors"
        (self.out / "headers" / (shard + ".json")).write_text(
            json.dumps({"shard": shard, "size": 1, "data_start": 8, "header": {}}), encoding="utf-8")
        hub = FakeHub(files)
        r = self.run_hub(hub)
        self.assertEqual((self.out / "config.json").read_bytes(), files["config.json"])
        self.assertEqual(len(hub.calls), 2)              # config.json and the one header
        self.assertEqual(r["cached"], 1 + 2 + 1)
        self.assertEqual(json.loads((self.out / "headers" / (shard + ".json")).read_text())["size"], len(files[shard]))


class Accounting(Base):
    def test_bytes_transferred_is_every_byte_received_and_requests_every_curl_run(self):
        files = sharded_repo(2, pad=hh.HEADER_PROBE)
        hub = FakeHub(files)
        r = self.run_hub(hub)
        shards = r["shards"]
        expect = (len(json.dumps(hub.api)) + len(files["config.json"]) + len(files["generation_config.json"])
                  + len(files["model.safetensors.index.json"]) + sum(header_len(files[s]) for s in shards))
        self.assertEqual(r["bytes_transferred"], expect)
        self.assertEqual(r["bytes_transferred"], hub.sent)
        self.assertEqual(r["requests"], 4 + 2 * len(shards))   # api, 3 small files, probe + rest per shard
        self.assertEqual(r["requests"], len(hub.calls))

    def test_failed_requests_count_too(self):
        fake = FakeHub(sharded_repo())
        hub = hh.Hub(REPO, "0" * 40, "curl", Path(self.tmp.name) / "dl")
        with mock.patch.object(hh, "curl_call", fake), self.assertRaises(hh.HeaderError):
            hub.get(hub.api_url(), "the API record")
        body = json.dumps({"error": "Invalid rev id: " + "0" * 40})
        self.assertEqual((hub.received, hub.requests), (len(body), 1))


class Token(Base):
    def test_the_token_goes_through_stdin_never_argv_logs_errors_or_files(self):
        os.environ["HF_TOKEN"] = TOKEN
        files = sharded_repo(2)
        hub = FakeHub(files)
        r = self.run_hub(hub)
        self.assertEqual(len(r["shards"]), 2)
        self.assertTrue(hub.calls)
        for argv, config in hub.calls:
            self.assertFalse(any(TOKEN in a for a in argv), argv)
            self.assertEqual(argv[-2:], ["-K", "-"])
            self.assertEqual(config, f'header = "Authorization: Bearer {TOKEN}"\n'.encode())
        self.assertTrue(self.logs)
        self.assertFalse(any(TOKEN in line for line in self.logs))
        for p in self.out.rglob("*"):
            if p.is_file():
                self.assertNotIn(TOKEN.encode(), p.read_bytes(), p)
        # an error body that echoes the token never reaches the reason
        bad = FakeHub(files)
        bad.status["/api/models/"] = (401, json.dumps({"error": f"token {TOKEN} has no access"}).encode())
        self.out = Path(self.tmp.name) / "out2"
        msg = self.refused(bad, "HF_TOKEN is set; its account has no access")
        self.assertNotIn(TOKEN, msg)
        self.assertIn("token <HF_TOKEN> has no access", msg)
        self.assertFalse(any(TOKEN in line for line in self.logs))

    def test_without_hf_token_no_config_is_sent(self):
        hub = FakeHub(sharded_repo(1))
        self.run_hub(hub)
        for argv, config in hub.calls:
            self.assertIsNone(config)
            self.assertNotIn("-K", argv)


class Cli(Base):
    def cli(self, hub, *extra):
        out, err = io.StringIO(), io.StringIO()
        with mock.patch.object(hh, "curl_call", hub), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = hh.main(["--repo", REPO, "--out", str(self.out), "--curl", "curl", *extra])
        return rc, out.getvalue(), err.getvalue()

    def test_a_refusal_exits_2_with_the_reason_on_stderr(self):
        rc, out, err = self.cli(FakeHub(sharded_repo()), "--revision", "0" * 40)
        self.assertEqual(rc, 2)
        self.assertEqual(out, "")
        self.assertRegex(err, r"^hf_headers: HTTP 404 on the API record: repo acme/Tiny-MoE or revision 0{40} not found")

    def test_success_prints_one_summary_line_and_exits_0(self):
        rc, out, err = self.cli(FakeHub(sharded_repo(2)), "--revision", REV)
        self.assertEqual(rc, 0)
        self.assertEqual(len(out.splitlines()), 1)
        self.assertRegex(out, r"acme/Tiny-MoE@aaaaaaaaaaaa: 2 shards, 4 tensors, header bytes [\d,]+ B, "
                              r"bytes_transferred [\d,]+ B, requests 6, cached 0 -> ")


class FetchGlmThroughHelper(unittest.TestCase):
    """tools/fetch-glm.py --headers reads through hh.read_header; the moved names stay on it."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.dest = Path(self.tmp.name)
        self.logs = []
        self.ctx = {"log": self.logs.append, "lock": mock.Mock(), "stats": fg.Stats(), "curl": "curl"}

    def test_the_moved_names_are_the_helpers(self):
        self.assertEqual(Path(fg.hh.__file__).resolve(), TOOLS / "hf_headers.py")
        for name in ("parse_safetensors_header", "NeedMore", "header_summary", "parse_write_out", "curl_range_argv",
                     "HEADER_PROBE", "MAX_HEADER", "CURL_RETRY", "WRITE_OUT", "CONNECT_TIMEOUT"):
            self.assertIs(getattr(fg, name), getattr(fg.hh, name), name)

    def fake_curl(self, data, specs):
        def run_curl(argv, watch, log, lock, label, expected=None, floor=None):
            spec = argv[argv.index("-r") + 1]
            specs.append(spec)
            b, e = (int(x) for x in spec.split("-"))
            Path(watch).write_bytes(data[b:e + 1])
            return 0, {"http": 206, "retries": 0, "bytes": e + 1 - b, "exit": 0}, ""
        return run_curl

    def test_a_long_header_is_read_in_two_ranges_and_cached_as_the_helper_writes_it(self):
        shard = "model-00001-of-00062.safetensors"
        data = st_bytes([("w", 64)], pad=hh.HEADER_PROBE + 10)
        table = {shard: {"size": len(data), "blob": None, "sha256": hashlib.sha256(data).hexdigest()}}
        specs = []
        with mock.patch.object(fg, "run_curl", self.fake_curl(data, specs)):
            rc = fg.mode_headers(self.dest, table, [shard], self.ctx)
        hl = header_len(data)
        self.assertEqual(rc, 0)
        self.assertEqual(specs, [f"0-{hh.HEADER_PROBE - 1}", f"{hh.HEADER_PROBE}-{hl - 1}"])
        hdr, start = hh.parse_safetensors_header(data)
        self.assertEqual((self.dest / "headers" / (shard + ".json")).read_text(encoding="utf-8"),
                         hh.header_record(shard, len(data), start, hdr))
        self.assertTrue(any(l.startswith(f"headers: 1/1 shards, 1 tensors, header bytes {hl:,} B") for l in self.logs))

    def test_a_non_tiling_shard_is_refused_by_the_shared_check(self):
        shard = "model-00001-of-00062.safetensors"
        data = st_bytes([("w", 64)])
        table = {shard: {"size": len(data) + 4, "blob": None, "sha256": "0" * 64}}
        with mock.patch.object(fg, "run_curl", self.fake_curl(data + bytes(4), [])):
            with self.assertRaisesRegex(ValueError, "!= file size"):
                fg.mode_headers(self.dest, table, [shard], self.ctx)
        self.assertFalse((self.dest / "headers" / (shard + ".json")).exists())

    def test_a_verified_shard_on_disk_is_read_locally_without_network(self):
        shard = "model-00001-of-00062.safetensors"
        data = st_bytes([("w", 64), ("b", 16)], pad=hh.HEADER_PROBE + 10)
        meta = {"size": len(data), "blob": None, "sha256": hashlib.sha256(data).hexdigest()}
        (self.dest / shard).write_bytes(data)
        fg.marker_of(self.dest / shard).write_text(json.dumps({"size": len(data), "sha256": meta["sha256"]}),
                                                   encoding="utf-8")

        def no_network(*a, **k):
            raise AssertionError("network used for a verified local shard")
        with mock.patch.object(fg, "run_curl", no_network):
            self.assertEqual(fg.mode_headers(self.dest, {shard: meta}, [shard], self.ctx), 0)
        rec = json.loads((self.dest / "headers" / (shard + ".json")).read_text(encoding="utf-8"))
        self.assertEqual((rec["data_start"], len(rec["header"])), (header_len(data), 3))


if __name__ == "__main__":
    unittest.main(verbosity=1)
