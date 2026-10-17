#!/usr/bin/env python3
"""#240: the safetensors shard headers of any Hugging Face model by HTTP range, no tensor bytes.

  python -I tools/hf_headers.py --repo ORG/NAME --revision SHA --out DIR [--curl PATH] [-v]

Writes into DIR: `hf-revision.json` (the API record `/api/models/<repo>/revision/<rev>?blobs=true`
as received), `config.json`, `generation_config.json` (if the repo has it),
`model.safetensors.index.json` (if the model is sharded) and `headers/<shard>.json` per shard in
the converter's `--headers` format `{"shard", "size", "data_start", "header"}`
(converter/README.md, "Staged conversion"). A repo with one `model.safetensors` and no index gets
`headers/model.safetensors.json`.

Every small file is checked against the API record (size, and `lfs.sha256` or the git `blobId`);
every header must tile its shard (tensor byte ranges contiguous from 0, data_start + tensor bytes
== shard size). Files already in DIR that pass these checks are reused without network, so a
rerun transfers 0 B. A header costs one range request of min(256 KiB, shard size); a second one
for the rest only when the header is longer (the pattern of `HfApi.parse_safetensors_file_metadata`).

If the environment has `HF_TOKEN`, it is sent as `Authorization: Bearer ...` through curl's
config on stdin (`-K -`): never on curl's argv, in a log line, an error text or a file.

Prints one summary line, exit 0. A refusal (gated or missing repo, wrong revision, no safetensors
index, a header over the cap or not tiling its shard, ...) exits 2 with `hf_headers: <reason>` on
stderr. `tools/fetch-glm.py` reads GLM-5.3-Flash's headers through `read_header` here.
Standard library + curl; downloaded files are only read as data (run with `python -I`).
Tests: `tools/test_hf_headers.py` (no network, no curl).
"""

import argparse
import hashlib
import json
import os
import re
import shutil
import struct
import subprocess
import sys
from pathlib import Path, PurePosixPath
from urllib.parse import quote

HF = "https://huggingface.co"
API_FILE = "hf-revision.json"
HEADER_DIR = "headers"
INDEX = "model.safetensors.index.json"
SINGLE = "model.safetensors"
CONFIG = "config.json"
GENERATION = "generation_config.json"
TMP = ".hf_headers.download"

HEADER_PROBE = 256 * 1024         # first range request of a shard header
MAX_HEADER = 100 * 1000 * 1000    # safetensors caps the header at 100 MB

CURL_RETRY = ["--retry", "20", "--retry-all-errors", "--retry-delay", "10"]   # range and small GET requests only
CONNECT_TIMEOUT = 30
WRITE_OUT = "%{http_code} %{num_retries} %{size_download} %{exitcode}"
PERMANENT_HTTP = {401, 403, 404, 410}
RANGE_TRIES = 3
WEIGHT_SUFFIXES = (".safetensors", ".gguf", ".bin", ".pt", ".pth", ".ckpt", ".onnx", ".h5", ".msgpack")
REPO_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
REVISION_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class HeaderError(Exception):
    """A refusal; str(e) is the reason shown to the user."""


# ---------------------------------------------------------------- pure parts


class NeedMore(Exception):
    def __init__(self, total):
        super().__init__(f"need {total} bytes")
        self.total = total


def parse_safetensors_header(blob):
    """(header dict, data_start) from a prefix of a safetensors file.

    Raises NeedMore(total) when the prefix is shorter than 8 + header length.
    """
    if len(blob) < 8:
        raise NeedMore(8)
    header_len = struct.unpack("<Q", blob[:8])[0]
    if header_len > MAX_HEADER:
        raise ValueError(f"header length {header_len} B exceeds the safetensors cap of {MAX_HEADER} B")
    if len(blob) < 8 + header_len:
        raise NeedMore(8 + header_len)
    header = json.loads(blob[8:8 + header_len])
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header, 8 + header_len


def header_summary(header, data_start, file_size):
    """(tensor count, tensor bytes) of one shard; refuses a header that does not tile the file."""
    spans = sorted(tuple(v["data_offsets"]) for k, v in header.items() if k != "__metadata__")
    cursor = 0
    for b, e in spans:
        if b != cursor or e < b:
            raise ValueError(f"tensor bytes are not contiguous at offset {b} (expected {cursor})")
        cursor = e
    if data_start + cursor != file_size:
        raise ValueError(f"header {data_start} B + tensors {cursor} B != file size {file_size} B")
    return len(spans), cursor


def read_header(read, size):
    """(header, data_start, tensors, tensor bytes) of a safetensors file of `size` B.

    `read(begin, end)` returns exactly the bytes [begin, end) of the file. The first read is
    [0, min(HEADER_PROBE, size)); a second read for the rest only when the header is longer.
    Raises ValueError for a header over MAX_HEADER, one longer than the file, or one that does
    not tile the file.
    """
    blob = read(0, min(HEADER_PROBE, size))
    try:
        header, data_start = parse_safetensors_header(blob)
    except NeedMore as more:
        if more.total > size:
            raise ValueError(f"header needs {more.total} B, the file has {size} B") from None
        blob += read(len(blob), more.total)
        header, data_start = parse_safetensors_header(blob)
    n, tensor_bytes = header_summary(header, data_start, size)
    return header, data_start, n, tensor_bytes


def header_record(shard, size, data_start, header):
    """The converter's `--headers` file of one shard, byte for byte as tools/fetch-glm.py writes it."""
    return json.dumps({"shard": shard, "size": size, "data_start": data_start, "header": header})


def parse_write_out(text):
    """curl -w WRITE_OUT -> {"http", "retries", "bytes", "exit"} (ints; missing -> 0)."""
    parts = (text or "").strip().split()
    vals = []
    for i in range(4):
        try:
            vals.append(int(parts[i]))
        except (IndexError, ValueError):
            vals.append(0)
    return dict(zip(("http", "retries", "bytes", "exit"), vals))


def curl_range_argv(curl, url, out, begin, end):
    """Bytes [begin, end) into `out`. `--max-filesize` stops a server that ignores the range
    from sending the whole 5 GB shard."""
    return [curl, "-L", "-f", "-sS", *CURL_RETRY, "--connect-timeout", str(CONNECT_TIMEOUT),
            "--max-time", "300", "--max-filesize", str(end - begin),
            "-r", f"{begin}-{end - 1}", "-w", WRITE_OUT, "-o", str(out), url]


def curl_get_argv(curl, url, out):
    """One small GET (API record, config). No `-f`: a 4xx answer comes back at once with its body
    (the reason) instead of being retried; 408/429/5xx and dropped connections are retried."""
    return [curl, "-L", "-sS", *CURL_RETRY, "--connect-timeout", str(CONNECT_TIMEOUT),
            "--max-time", "300", "-w", WRITE_OUT, "-o", str(out), url]


def auth_config(token):
    """curl config text for `-K -` (stdin): the token never reaches argv."""
    tok = token.replace("\\", "\\\\").replace('"', '\\"')
    return f'header = "Authorization: Bearer {tok}"\n'.encode("utf-8")


def api_files(api):
    """{name: {"size", "blob", "sha256"}} from the revision API record."""
    out = {}
    for s in api.get("siblings") or []:
        lfs = s.get("lfs") or {}
        if lfs and lfs.get("size") != s.get("size"):
            raise HeaderError(f"API record: {s.get('rfilename')}: size {s.get('size')} != lfs.size {lfs.get('size')}")
        if "size" not in s:
            raise HeaderError(f"API record: {s.get('rfilename')} has no size (blobs=true missing?)")
        out[s["rfilename"]] = {"size": int(s["size"]), "blob": s.get("blobId"), "sha256": lfs.get("sha256")}
    return out


def some_files(table, n=8):
    """A few of the repo's files for a refusal, weight-like files first."""
    names = sorted(table, key=lambda x: (not x.lower().endswith(WEIGHT_SUFFIXES), x))
    more = f" (+{len(names) - n} more)" if len(names) > n else ""
    return (", ".join(names[:n]) or "no files") + more


def check_bytes(data, meta):
    """None if `data` matches its API record (size, then lfs.sha256 or git blobId), else the reason."""
    if len(data) != meta["size"]:
        return f"size {len(data)} B != API size {meta['size']} B"
    if meta["sha256"]:
        got = hashlib.sha256(data).hexdigest()
        return None if got == meta["sha256"] else f"sha256 {got} != lfs.sha256 {meta['sha256']}"
    if meta["blob"]:
        got = hashlib.sha1(b"blob %d\x00" % len(data) + data).hexdigest()
        return None if got == meta["blob"] else f"git blob {got} != blobId {meta['blob']}"
    return None


def safe_name(name):
    """A repo file name usable under DIR (no absolute path, no `..`)."""
    p = PurePosixPath(name)
    if not name or "\\" in name or p.is_absolute() or ".." in p.parts:
        raise HeaderError(f"refusing the file name {name!r}")
    return name


def index_shards(index, table):
    """Sorted distinct shard names of a `model.safetensors.index.json`; each must be in the API record."""
    wm = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(wm, dict) or not wm:
        raise HeaderError(f"{INDEX} has no weight_map")
    shards = sorted(set(wm.values()))
    missing = [s for s in shards if s not in table]
    if missing:
        raise HeaderError(f"{INDEX} names {len(missing)} shard(s) the API does not list: {', '.join(missing[:3])}")
    return [safe_name(s) for s in shards]


def summary_line(r):
    return (f"{r['repo']}@{r['revision'][:12]}: {len(r['shards'])} shards, {r['tensors']:,} tensors, "
            f"header bytes {r['header_bytes']:,} B, bytes_transferred {r['bytes_transferred']:,} B, "
            f"requests {r['requests']}, cached {r['cached']} -> {r['headers_dir']}")


# ---------------------------------------------------------------- io


def curl_call(argv, config=None):
    """Run one curl. `config` (bytes) goes to its stdin for `-K -`. (exit code, stdout, stderr)."""
    kw = {"input": config} if config is not None else {"stdin": subprocess.DEVNULL}
    p = subprocess.run(argv, capture_output=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), **kw)
    return p.returncode, p.stdout.decode("ascii", "replace"), p.stderr.decode("utf-8", "replace")


def server_message(body):
    """The reason a hub error body gives (JSON `error` or the first line), short."""
    text = body.decode("utf-8", "replace").strip() if body else ""
    try:
        obj = json.loads(text)
        if isinstance(obj, dict) and obj.get("error"):
            text = str(obj["error"])
    except ValueError:
        pass
    return (text.splitlines() or [""])[0][:200]


def write_atomic(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    part = path.with_name(path.name + ".part")
    part.write_bytes(data)
    os.replace(part, path)


class Hub:
    """curl against huggingface.co for one repo@revision; counts requests and received bytes."""

    def __init__(self, repo, revision, curl, tmp, log=None, token=None):
        self.repo, self.revision, self.curl, self.tmp = repo, revision, curl, Path(tmp)
        self.token = token or None
        self._log = log
        self.requests = 0
        self.received = 0

    def scrub(self, text):
        text = str(text)
        return text.replace(self.token, "<HF_TOKEN>") if self.token else text

    def log(self, msg):
        if self._log:
            self._log(self.scrub(msg))

    def fail(self, msg):
        return HeaderError(self.scrub(msg))

    def api_url(self):
        return f"{HF}/api/models/{self.repo}/revision/{self.revision}?blobs=true"

    def file_url(self, name):
        return f"{HF}/{self.repo}/resolve/{self.revision}/{quote(name)}"

    def call(self, argv):
        """One curl run into self.tmp: (exit, write-out dict, stderr tail, body bytes)."""
        config = None
        if self.token:
            argv = argv + ["-K", "-"]
            config = auth_config(self.token)
        self.tmp.unlink(missing_ok=True)
        try:
            rc, out, err = curl_call(argv, config)
        except OSError as e:
            raise self.fail(f"cannot run curl {self.curl!r}: {e}") from None
        wo = parse_write_out(out)
        data = self.tmp.read_bytes() if self.tmp.exists() else b""
        self.tmp.unlink(missing_ok=True)
        self.requests += 1
        self.received += max(wo["bytes"], len(data))
        tail = self.scrub(" | ".join(err.strip().splitlines()[-3:]))
        return rc, wo, tail, data

    def refusal(self, http, label, rc, tail, body=b""):
        said = server_message(body)
        said = f" (server: {said})" if said else ""
        if http in (401, 403):
            return self.fail(
                f"HTTP {http} on {label}: {self.repo} is gated, private or does not exist{said}. Accept the licence "
                f"on {HF}/{self.repo} and set HF_TOKEN"
                + (" (HF_TOKEN is set; its account has no access)" if self.token else ""))
        if http in (404, 410):
            return self.fail(f"HTTP {http} on {label}: repo {self.repo} or revision {self.revision} not found{said}")
        return self.fail(f"{label}: curl exit {rc}, HTTP {http}{said}" + (f", curl: {tail}" if tail else ""))

    def get(self, url, label):
        """A whole small resource: its bytes, or HeaderError with the reason."""
        rc, wo, tail, data = self.call(curl_get_argv(self.curl, url, self.tmp))
        self.log(f"GET {label}: exit {rc}, http {wo['http']}, {len(data):,} B")
        if rc == 0 and wo["http"] == 200:
            return data
        raise self.refusal(wo["http"], label, rc, tail, data)

    def get_range(self, name, begin, end, size, tries=RANGE_TRIES):
        """Exactly the bytes [begin, end) of the repo file `name` of `size` B, or HeaderError."""
        label = f"{name} [{begin},{end})"
        rc = http = 0
        for attempt in range(1, tries + 1):
            rc, wo, tail, data = self.call(curl_range_argv(self.curl, self.file_url(name), self.tmp, begin, end))
            http = wo["http"]
            whole = begin == 0 and end == size
            if rc == 0 and len(data) == end - begin and (http == 206 or (http == 200 and whole)):
                self.log(f"range {label}: http {http}, {len(data):,} B")
                return data
            if http in PERMANENT_HTTP:
                raise self.refusal(http, label, rc, tail)
            self.log(f"range {label}: exit {rc}, http {http}, got {len(data):,} B (try {attempt}/{tries})"
                     + (f", curl: {tail}" if tail else ""))
        raise self.fail(f"{label}: failed {tries} times (last: curl exit {rc}, HTTP {http})")


def load_api(hub, out, counts):
    """The revision's API record: `hf-revision.json` in `out` if it is this revision, else fetched."""
    path = out / API_FILE
    if path.exists():
        try:
            api = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            api = None
        if isinstance(api, dict) and api.get("sha") == hub.revision:
            counts["cached"] += 1
            hub.log(f"cached {API_FILE}")
            return api
        hub.log(f"{API_FILE} on disk is not revision {hub.revision} -> read the API again")
    data = hub.get(hub.api_url(), "the API record")
    try:
        api = json.loads(data)
    except ValueError:
        raise hub.fail(f"the API answer for {hub.repo}@{hub.revision} is not JSON") from None
    if not isinstance(api, dict):
        raise hub.fail(f"the API answer for {hub.repo}@{hub.revision} is not a JSON object")
    if api.get("sha") != hub.revision:
        raise hub.fail(f"the API answers revision {api.get('sha')!r} for {hub.repo}, not the requested "
                       f"{hub.revision!r} (pass the full commit sha)")
    write_atomic(path, data)
    return api


def small_file(hub, out, name, meta, counts):
    """A small repo file checked against its API record, from `out` if it verifies there."""
    path = out / name
    if path.exists():
        data = path.read_bytes()
        bad = check_bytes(data, meta)
        if bad is None:
            counts["cached"] += 1
            hub.log(f"cached {name}")
            return data
        hub.log(f"{name} on disk does not verify ({bad}) -> fetch again")
    data = hub.get(hub.file_url(name), name)
    bad = check_bytes(data, meta)
    if bad:
        raise hub.fail(f"{name}: {bad}")
    write_atomic(path, data)
    return data


def cached_header(path, shard, size):
    """(data_start, tensors, tensor bytes, header) of a `headers/<shard>.json` that fits the shard, else None."""
    try:
        rec = json.loads(path.read_text(encoding="utf-8"))
        if rec.get("shard") != shard or rec.get("size") != size:
            return None
        n, tb = header_summary(rec["header"], rec["data_start"], size)
        return rec["data_start"], n, tb, rec["header"]
    except (OSError, ValueError, KeyError, TypeError, AttributeError):
        return None


def fetch_model(repo, revision, out_dir, curl=None, log=None):
    """Read the API record, config, index and every shard header of repo@revision into out_dir.

    Returns {"repo", "revision", "dir", "headers_dir", "shards", "bytes_transferred", "requests",
    "cached", "tensors", "header_bytes", "tensor_bytes"}. Raises HeaderError with the reason.
    """
    if not REPO_RE.match(repo or ""):
        raise HeaderError(f"--repo {repo!r} is not ORG/NAME")
    if not REVISION_RE.match(revision or ""):
        raise HeaderError(f"--revision {revision!r} is not a commit sha")
    curl = curl or shutil.which("curl")
    if not curl:
        raise HeaderError("curl not found on PATH (pass --curl)")
    out = Path(out_dir)
    hdir = out / HEADER_DIR
    hdir.mkdir(parents=True, exist_ok=True)
    hub = Hub(repo, revision, curl, out / TMP, log, os.environ.get("HF_TOKEN", "").strip())
    counts = {"cached": 0}

    table = api_files(load_api(hub, out, counts))
    sharded = INDEX in table
    if not sharded and SINGLE not in table:
        raise hub.fail(f"{repo}@{revision}: no {INDEX} and no single {SINGLE}; the repo has: {some_files(table)}")
    if CONFIG not in table:
        raise hub.fail(f"{repo}@{revision}: no {CONFIG}; the repo has: {some_files(table)}")
    small_file(hub, out, CONFIG, table[CONFIG], counts)
    if GENERATION in table:
        small_file(hub, out, GENERATION, table[GENERATION], counts)
    if sharded:
        raw = small_file(hub, out, INDEX, table[INDEX], counts)
        try:
            index = json.loads(raw)
        except ValueError:
            raise hub.fail(f"{INDEX} is not JSON") from None
        shards = index_shards(index, table)
    else:
        shards = [SINGLE]

    tensors = header_bytes = tensor_bytes = 0
    for shard in shards:
        size = table[shard]["size"]
        path = hdir / (shard + ".json")
        hit = cached_header(path, shard, size) if path.exists() else None
        if hit:
            data_start, n, tb, _ = hit
            counts["cached"] += 1
        else:
            try:
                header, data_start, n, tb = read_header(lambda b, e: hub.get_range(shard, b, e, size), size)
            except ValueError as err:
                raise hub.fail(f"{shard}: {err}") from None
            write_atomic(path, header_record(shard, size, data_start, header).encode("utf-8"))
            hub.log(f"header {shard}: {data_start} B header, {n} tensors, {tb:,} B tensors (tiles the file)")
        tensors += n
        header_bytes += data_start
        tensor_bytes += tb
    return {"repo": repo, "revision": revision, "dir": str(out), "headers_dir": str(hdir), "shards": shards,
            "bytes_transferred": hub.received, "requests": hub.requests, "cached": counts["cached"],
            "tensors": tensors, "header_bytes": header_bytes, "tensor_bytes": tensor_bytes}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="ORG/NAME on huggingface.co")
    ap.add_argument("--revision", required=True, help="the full commit sha")
    ap.add_argument("--out", required=True, help="directory for hf-revision.json, config, index, headers/")
    ap.add_argument("--curl", help="curl binary (default: curl on PATH)")
    ap.add_argument("-v", "--verbose", action="store_true", help="one progress line per request on stderr")
    args = ap.parse_args(argv)
    log = (lambda m: print(m, file=sys.stderr, flush=True)) if args.verbose else None
    try:
        r = fetch_model(args.repo, args.revision, args.out, curl=args.curl, log=log)
    except HeaderError as err:
        print(f"hf_headers: {err}", file=sys.stderr)
        return 2
    print(summary_line(r))
    return 0


if __name__ == "__main__":
    sys.exit(main())
