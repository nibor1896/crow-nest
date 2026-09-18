#!/usr/bin/env python3
"""#76/#79: fetch the BF16 originals back out of the Hugging Face repo, by HTTP range.

  tools/fetch-dense-originals.py --dry-run     # the plan, no network at all
  tools/fetch-dense-originals.py               # fetch what is missing, verify everything
  tools/fetch-dense-originals.py               # a second run fetches nothing and re-verifies

  tools/fetch-dense-originals.py --experts 1,7,13,19,25,31,37,43     # #79: the ROUTED EXPERTS
                                               # of those layers, one safetensors file each

Step 2 of the requant series. The 360 GB of `Qwen/Qwen3.8-Flash-Next` are on neither disk
anymore, and the reference-based metrics of #75 (mean KLD, top-1 agreement against BF16) and
any higher-precision rebuild of the dense path need the originals. Only about 2.58 B values
of them - about 5.2 GB in BF16 - and safetensors lets a reader take single tensors out of a
shard by HTTP range: the first 8 bytes are a little-endian u64 header length, the next N bytes
are a JSON header naming every tensor with `dtype`, `shape` and `data_offsets`, and those
offsets are RELATIVE to the end of the header. One tensor is therefore one range request.

WHICH tensors is DERIVED, never typed: every line of the container's own sidecar that is
`section == "text"` and `dtype == "nvfp4"` and whose name does not contain `.mlp.experts.`.
The routed experts are 97 % of the bytes and they stay as they are.

This tool only downloads. It proves nothing: the proof that the bytes are the ones the
container was built from is `converter requant-check` (see docs/dense-originals.md).

Nothing here needs a GPU, a venv or a token. Standard library only, sequential requests,
`Retry-After` honoured, resumable per tensor, idempotent. Unit tests for the pure parts in
`tools/test_fetch_dense_originals.py` (no network).
"""

import argparse
import email.utils
import hashlib
import http.client
import json
import os
import socket
import struct
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

DEFAULT_REPO = "Qwen/Qwen3.8-Flash-Next"
DEFAULT_REVISION = "de4b8e4d43b917e7706784d8bb445c9af86a3540"
DEFAULT_SIDECAR = ROOT / "converter" / "Qwen3.8-Flash-Next-CNQ4.5-M.cnq.sidecar.jsonl"
DEFAULT_INDEX = ROOT / "models" / "Qwen3.8-Flash-Next-original" / "model.safetensors.index.json"
DEFAULT_OUT = ROOT / "models" / "Qwen3.8-Flash-Next-original" / "dense"
OUT_NAME = "dense.safetensors"
MANIFEST_NAME = "manifest.json"

# #79: the routed experts, one file per layer, beside the dense one
DEFAULT_EXPERT_OUT = ROOT / "models" / "Qwen3.8-Flash-Next-original" / "experts"
EXPERT_NAME = "layer-{layer:02d}.safetensors"
EXPERT_MANIFEST = "layer-{layer:02d}.manifest.json"

DENSE_RULE = 'section == "text" and dtype == "nvfp4" and ".mlp.experts." not in name'
DENSE_META = "text section, nvfp4 in CNQ4.5-M, not .mlp.experts."
EXPERT_RULE = 'section == "text" and dtype == "nvfp4" and ".mlp.experts." in name'
EXPERT_META = "text section, nvfp4 in CNQ4.5-M, .mlp.experts."

USER_AGENT = "crow-nest-fetch-dense-originals/1 (+https://github.com/nibor1896/crow-nest)"

# One round trip costs about half a second on this line, which is worth roughly 5 MB of
# transfer at the 11 MB/s measured on 2026-09-18. Coalescing two tensor ranges that are
# closer than this therefore pays even though the gap between them is downloaded and thrown
# away; the span cap keeps one response in a bounded buffer.
DEFAULT_MAX_GAP = 2 * 1024 * 1024
DEFAULT_MAX_SPAN = 128 * 1024 * 1024
# The largest single range request. 128 MiB is 16 s at the 8 MB/s of this line, so the half
# second a round trip costs is 3 % - and a timeout re-fetches 128 MiB rather than 3.36 GB.
DEFAULT_CHUNK = 128 * 1024 * 1024
HEADER_PROBE = 256 * 1024

ELEM_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}

RETRY_STATUS = {408, 425, 429, 500, 502, 503, 504}


# ---------------------------------------------------------------- pure parts


def parse_safetensors_header(blob):
    """(header dict, data_start) from the first bytes of a safetensors file.

    Raises ValueError if `blob` is too short to hold the whole header - the caller then
    knows exactly how many bytes it still has to ask for.
    """
    if len(blob) < 8:
        raise ValueError(f"need 8 bytes for the header length, got {len(blob)}")
    header_len = struct.unpack("<Q", blob[:8])[0]
    if len(blob) < 8 + header_len:
        raise ValueError(f"header is {header_len} B, only {len(blob) - 8} B after the length field")
    header = json.loads(blob[8 : 8 + header_len])
    if not isinstance(header, dict):
        raise ValueError("safetensors header is not a JSON object")
    return header, 8 + header_len


def select_dense_tensors(sidecar_lines):
    """The DERIVED list: text-section nvfp4 tensors that are not routed experts.

    One dict per tensor with `name` and `n`, in the sidecar's own order. Lines without a
    `name` are the per-section summary records and are skipped.
    """
    out = []
    for line in sidecar_lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "name" not in rec:
            continue
        if rec.get("section") != "text" or rec.get("dtype") != "nvfp4":
            continue
        if ".mlp.experts." in rec["name"]:
            continue
        out.append({"name": rec["name"], "n": rec["n"]})
    return out


def layer_of(name):
    """The layer number in `model.language_model.layers.<N>....`, or None if there is none.

    The same rule the converter and the engine derive a kind from; here only the number is
    wanted, because #79 fetches the routed experts of CHOSEN layers.
    """
    marker = ".layers."
    at = name.find(marker)
    if at < 0:
        return None
    rest = name[at + len(marker) :]
    dot = rest.find(".")
    if dot <= 0:
        return None
    digits = rest[:dot]
    if not digits.isdigit():
        return None
    return int(digits)


def select_expert_tensors(sidecar_lines, layers):
    """#79: the ROUTED EXPERT tensors of the given layers - the mirror of the dense rule.

    Same two container facts (`section == "text"`, `dtype == "nvfp4"`), but the name MUST
    contain `.mlp.experts.` and its layer must be one of `layers`. One dict per tensor with
    `name`, `n` and `layer`, in the sidecar's own order.
    """
    want = set(layers)
    out = []
    for line in sidecar_lines:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        if "name" not in rec:
            continue
        if rec.get("section") != "text" or rec.get("dtype") != "nvfp4":
            continue
        if ".mlp.experts." not in rec["name"]:
            continue
        layer = layer_of(rec["name"])
        if layer is None or layer not in want:
            continue
        out.append({"name": rec["name"], "n": rec["n"], "layer": layer})
    return out


def parse_layers(spec):
    """`1,7,13` or `1-3,7` -> a sorted list of distinct layer numbers. Refuses anything else."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = part.split("-", 1)
            lo, hi = int(lo), int(hi)
            if hi < lo:
                raise ValueError(f"layer range {part!r} runs backwards")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    if not out:
        raise ValueError("no layers selected")
    if min(out) < 0:
        raise ValueError("layer numbers are not negative")
    return sorted(out)


def coalesce_ranges(items, max_gap, max_span):
    """Group byte ranges of ONE shard into as few requests as possible.

    `items` is a list of dicts with `begin` and `end`; the result is a list of
    {"begin", "end", "members"} with members in ascending order. Two groups are merged when
    the gap between them is at most `max_gap` and the merged span stays at or below
    `max_span`. The gap bytes ARE downloaded and discarded - that is the trade the caller
    picks the thresholds for.
    """
    groups = []
    for it in sorted(items, key=lambda i: (i["begin"], i["end"])):
        if groups:
            last = groups[-1]
            gap = it["begin"] - last["end"]
            span = max(last["end"], it["end"]) - last["begin"]
            if gap <= max_gap and span <= max_span:
                last["end"] = max(last["end"], it["end"])
                last["members"].append(it)
                continue
        groups.append({"begin": it["begin"], "end": it["end"], "members": [it]})
    return groups


def build_safetensors_header(entries, metadata=None):
    """The header bytes of a fresh safetensors file plus the offset each tensor lands at.

    `entries` is a list of dicts with `name`, `dtype` and `shape`, and the order IS the
    layout: offsets are contiguous from 0, no holes, no overlap. Returns
    (header_bytes, [out_offset per entry], data_bytes). The JSON is padded with spaces to a
    multiple of 8, the way the reference writer pads it.
    """
    header = {}
    if metadata is not None:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offsets = []
    cursor = 0
    for e in entries:
        n = 1
        for d in e["shape"]:
            n *= d
        size = n * ELEM_SIZE[e["dtype"]]
        header[e["name"]] = {"dtype": e["dtype"], "shape": list(e["shape"]), "data_offsets": [cursor, cursor + size]}
        offsets.append(cursor)
        cursor += size
    body = json.dumps(header, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    pad = (-len(body)) % 8
    body += b" " * pad
    return struct.pack("<Q", len(body)) + body, offsets, cursor


def parse_retry_after(value, now=None):
    """Seconds to wait from a `Retry-After` header, or None. Accepts both forms."""
    if not value:
        return None
    value = value.strip()
    try:
        return max(0.0, float(int(value)))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    now = now or datetime.now(timezone.utc)
    return max(0.0, (when - now).total_seconds())


def backoff_seconds(attempt):
    """Sleep before retry number `attempt` (1-based), capped."""
    return min(60.0, 2.0 ** attempt)


# ---------------------------------------------------------------- the network


class RangeFetcher:
    """Sequential range reads with retry. Counts every request, retry and byte."""

    def __init__(self, base_url, max_tries=6, timeout=120.0, sleep=time.sleep, opener=None):
        self.base_url = base_url.rstrip("/") + "/"
        self.max_tries = max_tries
        self.timeout = timeout
        self._sleep = sleep
        self._opener = opener or urllib.request.urlopen
        self.requests = 0
        self.retries = 0
        self.bytes = 0
        self.rate_limited = 0
        self.notes = []

    def get(self, path, begin, end):
        """Bytes [begin, end) of `path`. Raises after the last try."""
        want = end - begin
        last = None
        for attempt in range(1, self.max_tries + 1):
            req = urllib.request.Request(
                self.base_url + path,
                headers={"Range": f"bytes={begin}-{end - 1}", "User-Agent": USER_AGENT, "Accept-Encoding": "identity"},
            )
            try:
                self.requests += 1
                with self._opener(req, timeout=self.timeout) as resp:
                    status = getattr(resp, "status", None) or resp.getcode()
                    if status != 206:
                        # a 200 means the server ignored the Range and is about to hand us
                        # the whole shard - refuse instead of downloading 1 GB by accident
                        raise RuntimeError(f"{path}: expected 206 for a range request, got {status}")
                    data = resp.read()
                if len(data) != want:
                    raise RuntimeError(f"{path}: asked for {want} B at {begin}, got {len(data)} B")
                self.bytes += len(data)
                return data
            except urllib.error.HTTPError as err:
                last = err
                retry_after = parse_retry_after(err.headers.get("Retry-After") if err.headers else None)
                if err.code == 429:
                    self.rate_limited += 1
                if err.code not in RETRY_STATUS or attempt == self.max_tries:
                    raise
                wait = retry_after if retry_after is not None else backoff_seconds(attempt)
                self.notes.append(f"HTTP {err.code} on {path} [{begin},{end}) - waiting {wait:.1f}s")
                self.retries += 1
                self._sleep(wait)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError,
                    http.client.HTTPException, RuntimeError) as err:
                last = err
                if attempt == self.max_tries:
                    raise
                wait = backoff_seconds(attempt)
                self.notes.append(f"{type(err).__name__} on {path} [{begin},{end}) - waiting {wait:.1f}s")
                self.retries += 1
                self._sleep(wait)
        raise last  # unreachable: the loop either returns or raises

    def read_header(self, path, cache_dir=None):
        """(header, data_start) of a shard, from the cache if it is there."""
        cached = None if cache_dir is None else Path(cache_dir) / (path + ".json")
        if cached is not None and cached.exists():
            rec = json.loads(cached.read_text(encoding="utf-8"))
            return rec["header"], rec["data_start"]
        blob = self.get(path, 0, HEADER_PROBE)
        try:
            header, data_start = parse_safetensors_header(blob)
        except ValueError:
            header_len = struct.unpack("<Q", blob[:8])[0]
            blob += self.get(path, len(blob), 8 + header_len)
            header, data_start = parse_safetensors_header(blob)
        if cached is not None:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"header": header, "data_start": data_start}), encoding="utf-8")
        return header, data_start


# ---------------------------------------------------------------- the run


def sha256_of(path, offset, length, chunk=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        left = length
        while left:
            block = f.read(min(chunk, left))
            if not block:
                raise RuntimeError(f"{path}: short read at offset {offset + length - left}")
            h.update(block)
            left -= len(block)
    return h.hexdigest()


def human(n):
    return f"{n:,} B ({n / 1e9:.3f} GB)"


def fetch_group(group, args, weight_map, base_url):
    """Fetch ONE output safetensors file. Returns (rc, stats) and prints its own lines.

    `group` names the selection, the output file, the manifest, the header cache and the
    metadata; everything below this line is what #76 did for the dense file, unchanged, and
    #79 calls it once per pilot layer instead of once.
    """
    selection = group["selection"]
    out_file = group["out_file"]
    manifest_path = group["manifest"]
    header_cache = group["header_cache"]

    out_file.parent.mkdir(parents=True, exist_ok=True)
    fetch = RangeFetcher(base_url, max_tries=args.max_tries)
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")
    shards = sorted({t["shard"] for t in selection})

    # every shard header once (cached), and with it the real dtype, shape and byte range
    print(f"headers:   reading {len(shards)} shard headers ({header_cache})")
    headers = {}
    for shard in shards:
        headers[shard] = fetch.read_header(shard, cache_dir=header_cache)
    header_requests = fetch.requests

    entries = []
    for t in selection:
        header, data_start = headers[t["shard"]]
        info = header.get(t["name"])
        if info is None:
            print(f"{t['name']} is not in {t['shard']}", file=sys.stderr)
            return 2, None
        dtype = info["dtype"]
        shape = list(info["shape"])
        n = 1
        for d in shape:
            n *= d
        begin, end = info["data_offsets"]
        nbytes = end - begin
        if n != t["n"]:
            print(f"{t['name']}: shard says {n} values, the container says {t['n']}", file=sys.stderr)
            return 2, None
        if nbytes != n * ELEM_SIZE.get(dtype, 0):
            print(f"{t['name']}: {nbytes} B does not match {n} x {dtype}", file=sys.stderr)
            return 2, None
        entries.append({
            "name": t["name"], "dtype": dtype, "shape": shape, "n": n,
            "shard": t["shard"], "shard_begin": data_start + begin, "shard_end": data_start + end,
            "nbytes": nbytes,
        })

    dtypes = sorted({e["dtype"] for e in entries})
    print(f"dtype:     {', '.join(dtypes)} in the shard headers")

    header_bytes, out_offsets, data_bytes = build_safetensors_header(entries, metadata=group["metadata"])
    for e, off in zip(entries, out_offsets):
        e["out_offset"] = len(header_bytes) + off
    total_file = len(header_bytes) + data_bytes
    print(f"output:    {out_file}  header {len(header_bytes)} B + payload {human(data_bytes)}")

    # the file is created at its final size once; a tensor is only ever recorded in the
    # manifest after its bytes are on disk and hashed, so an interrupted run loses at most
    # the tensor it was in the middle of.
    fresh = True
    if out_file.exists() and out_file.stat().st_size == total_file:
        with open(out_file, "rb") as f:
            fresh = f.read(len(header_bytes)) != header_bytes
    if fresh:
        existed = out_file.exists()
        with open(out_file, "wb") as f:
            f.write(header_bytes)
            f.truncate(total_file)
        print("output:    " + ("recreated - the layout on disk was not this plan" if existed else "created"))

    done = {}
    if manifest_path.exists() and not fresh:
        old = json.loads(manifest_path.read_text(encoding="utf-8"))
        for rec in old.get("tensors", []):
            done[rec["name"]] = rec
        runs = old.get("runs", [])
    else:
        runs = []

    todo = []
    verified = 0
    for e in entries:
        rec = done.get(e["name"])
        ok = (rec is not None
              and rec.get("sha256")
              and rec.get("out_offset") == e["out_offset"]
              and rec.get("nbytes") == e["nbytes"]
              and rec.get("shard_begin") == e["shard_begin"])
        if ok and not args.skip_verify:
            ok = sha256_of(out_file, e["out_offset"], e["nbytes"]) == rec["sha256"]
            if not ok:
                print(f"  re-fetch {e['name']}: the bytes on disk do not match the manifest")
        if ok:
            e["sha256"] = rec["sha256"]
            verified += 1
        else:
            todo.append(e)

    print(f"state:     {verified} tensors already on disk and verified, {len(todo)} to fetch")

    by_shard = {}
    for e in todo:
        by_shard.setdefault(e["shard"], []).append(e)

    fetched_bytes = 0
    groups_total = 0
    fh = open(out_file, "r+b")
    try:
        for si, shard in enumerate(sorted(by_shard), 1):
            items = [{"begin": e["shard_begin"], "end": e["shard_end"], "entry": e} for e in by_shard[shard]]
            groups = coalesce_ranges(items, args.max_gap, args.max_span)
            groups_total += len(groups)
            shard_bytes = 0
            t_shard = time.monotonic()
            for g in groups:
                # ONE request per group would buffer the whole span in RAM and make a retry
                # cost all of it - which the 5 dense tensors never noticed (2 MB to 500 MB)
                # and the 3.36 GB `experts.gate_up_proj` of #79 would. The span is cut into
                # requests of at most `--chunk-bytes`; the bytes go straight to their place
                # in the output file and each tensor's sha256 is fed in arrival order, which
                # is ascending because the slices are.
                digests = {id(m["entry"]): hashlib.sha256() for m in g["members"]}
                pos = g["begin"]
                while pos < g["end"]:
                    stop = min(g["end"], pos + args.chunk_bytes)
                    blob = fetch.get(shard, pos, stop)
                    for m in g["members"]:
                        e = m["entry"]
                        lo = max(pos, e["shard_begin"])
                        hi = min(stop, e["shard_end"])
                        if hi <= lo:
                            continue
                        part = blob[lo - pos : hi - pos]
                        fh.seek(e["out_offset"] + (lo - e["shard_begin"]))
                        fh.write(part)
                        digests[id(e)].update(part)
                        shard_bytes += len(part)
                    del blob
                    pos = stop
                for m in g["members"]:
                    e = m["entry"]
                    e["sha256"] = digests[id(e)].hexdigest()
            fh.flush()
            fetched_bytes += shard_bytes
            dt = max(1e-9, time.monotonic() - t_shard)
            print(f"  [{si:2d}/{len(by_shard)}] {shard}  {len(by_shard[shard]):3d} tensors in "
                  f"{len(groups)} request(s)  {shard_bytes:>12,} B  {shard_bytes / dt / 1e6:6.1f} MB/s", flush=True)
            # the manifest is rewritten after every shard: an interrupted run resumes here
            write_manifest(manifest_path, args, group, entries, header_bytes, total_file, runs, started, fetch, t0,
                           fetched_bytes, header_requests, groups_total, partial=True)
    finally:
        fh.close()

    elapsed = time.monotonic() - t0
    write_manifest(manifest_path, args, group, entries, header_bytes, total_file, runs, started, fetch, t0,
                   fetched_bytes, header_requests, groups_total, partial=False)

    rate = fetched_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
    print(f"done:      {len(entries)} tensors, {verified} re-verified, {len(todo)} fetched")
    print(f"           {fetch.requests} requests ({header_requests} headers, {groups_total} data), "
          f"{fetch.retries} retries, {fetch.rate_limited} HTTP 429")
    print(f"           {human(fetched_bytes)} in {elapsed:.1f} s, mean {rate:.1f} MB/s")
    print(f"           {out_file} ({total_file:,} B), {manifest_path}", flush=True)
    for note in fetch.notes:
        print(f"           note: {note}")
    return 0, {
        "tensors": len(entries), "verified": verified, "fetched": len(todo),
        "requests": fetch.requests, "retries": fetch.retries, "http_429": fetch.rate_limited,
        "fetched_bytes": fetched_bytes, "elapsed": elapsed, "file_bytes": total_file,
        "notes": list(fetch.notes),
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR, help="the container sidecar the list is derived from")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="model.safetensors.index.json (tensor -> shard)")
    ap.add_argument("--out", type=Path, default=None, help="output directory (default: dense/ or experts/ beside it)")
    ap.add_argument("--experts", default=None, metavar="LAYERS",
                    help="#79: fetch the ROUTED EXPERTS of these layers (e.g. 1,7,13) instead of the dense path, "
                         "one safetensors file per layer")
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network at all")
    ap.add_argument("--skip-verify", action="store_true", help="trust the manifest instead of re-hashing what is on disk")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    ap.add_argument("--max-span", type=int, default=DEFAULT_MAX_SPAN)
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--chunk-bytes", type=int, default=DEFAULT_CHUNK,
                    help="largest single range request; a coalesced span is cut into these")
    ap.add_argument("--limit", type=int, default=0, help="stop after N tensors (for a short live smoke test)")
    args = ap.parse_args(argv)

    sidecar_lines = args.sidecar.read_text(encoding="utf-8").splitlines()
    weight_map = json.loads(args.index.read_text(encoding="utf-8"))["weight_map"]

    if args.experts is None:
        selection = select_dense_tensors(sidecar_lines)
        rule, meta_sel = DENSE_RULE, DENSE_META
        out_dir = args.out or DEFAULT_OUT
        print(f"selection: {len(selection)} tensors, {sum(t['n'] for t in selection):,} values, "
              f"{human(sum(t['n'] for t in selection) * 2)} at BF16")
        print(f"           derived from {args.sidecar.name}: section text, dtype nvfp4, not .mlp.experts.")
        plans = [(None, selection)]
    else:
        layers = parse_layers(args.experts)
        selection = select_expert_tensors(sidecar_lines, layers)
        rule, meta_sel = EXPERT_RULE, EXPERT_META
        out_dir = args.out or DEFAULT_EXPERT_OUT
        found = sorted({t["layer"] for t in selection})
        if found != layers:
            missing = [layer for layer in layers if layer not in found]
            print(f"the container has no routed experts for layer(s) {missing}", file=sys.stderr)
            return 2
        print(f"selection: {len(selection)} tensors over {len(found)} layers, "
              f"{sum(t['n'] for t in selection):,} values, {human(sum(t['n'] for t in selection) * 2)} at BF16")
        print(f"           derived from {args.sidecar.name}: section text, dtype nvfp4, .mlp.experts., layers {found}")
        plans = [(layer, [t for t in selection if t["layer"] == layer]) for layer in found]

    missing = [t["name"] for t in selection if t["name"] not in weight_map]
    if missing:
        print(f"{len(missing)} selected tensors are not in the model index, first: {missing[0]}", file=sys.stderr)
        return 2
    for t in selection:
        t["shard"] = weight_map[t["name"]]

    groups = []
    for layer, sel in plans:
        # the layout of each output file: name order, contiguous, no holes
        sel = sorted(sel, key=lambda t: t["name"])
        if args.limit:
            sel = sel[: args.limit]
        meta = {
            "format": "pt",
            "crow_nest_source_repo": args.repo,
            "crow_nest_revision": args.revision,
            "crow_nest_selection": meta_sel if layer is None else f"{meta_sel} layer {layer}",
        }
        groups.append({
            "layer": layer,
            "selection": sel,
            "out_file": out_dir / (OUT_NAME if layer is None else EXPERT_NAME.format(layer=layer)),
            "manifest": out_dir / (MANIFEST_NAME if layer is None else EXPERT_MANIFEST.format(layer=layer)),
            "header_cache": out_dir / ".shard-headers",
            "metadata": meta,
            "rule": rule if layer is None else f"{rule} layer {layer}",
        })

    all_shards = sorted({t["shard"] for t in selection})
    print(f"shards:    {len(all_shards)} of 131")

    if args.dry_run:
        for g in groups:
            per_shard = {}
            for t in g["selection"]:
                e = per_shard.setdefault(t["shard"], [0, 0])
                e[0] += 1
                e[1] += t["n"] * 2
            if g["layer"] is not None:
                print(f"  -> {g['out_file'].name}  {len(g['selection'])} tensors, "
                      f"{sum(v[1] for v in per_shard.values()):,} B")
            for shard in sorted(per_shard):
                cnt, by = per_shard[shard]
                print(f"  {shard}  {cnt:3d} tensors  {by:>12,} B")
        total = sum(t["n"] for t in selection) * 2
        print(f"total:     {human(total)} of tensor payload, plus {len(all_shards)} shard headers")
        print("dry run: nothing fetched. The exact byte ranges come from the shard headers at fetch time.")
        return 0

    base_url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    t_all = time.monotonic()
    totals = {"tensors": 0, "verified": 0, "fetched": 0, "requests": 0, "retries": 0,
              "http_429": 0, "fetched_bytes": 0}
    for gi, g in enumerate(groups, 1):
        if g["layer"] is not None:
            print(f"== [{gi}/{len(groups)}] layer {g['layer']}: {len(g['selection'])} tensors "
                  f"-> {g['out_file'].name}", flush=True)
        rc, stats = fetch_group(g, args, weight_map, base_url)
        if rc != 0:
            return rc
        for k in totals:
            totals[k] += stats[k]
    if len(groups) > 1:
        elapsed = time.monotonic() - t_all
        rate = totals["fetched_bytes"] / elapsed / 1e6 if elapsed > 0 else 0.0
        print(f"ALL:       {len(groups)} files, {totals['tensors']} tensors, {totals['verified']} re-verified, "
              f"{totals['fetched']} fetched")
        print(f"           {totals['requests']} requests, {totals['retries']} retries, "
              f"{totals['http_429']} HTTP 429")
        print(f"           {human(totals['fetched_bytes'])} in {elapsed:.1f} s, mean {rate:.1f} MB/s", flush=True)
    return 0


def write_manifest(path, args, group, entries, header_bytes, total_file, runs, started, fetch, t0,
                   fetched_bytes, header_requests, groups, partial):
    recs = [e for e in entries if e.get("sha256")]
    run = {
        "started_utc": started,
        "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "partial": partial,
        "requests": fetch.requests,
        "header_requests": header_requests,
        "data_requests": groups,
        "retries": fetch.retries,
        "http_429": fetch.rate_limited,
        "fetched_bytes": fetched_bytes,
        "wall_seconds": round(time.monotonic() - t0, 1),
        "notes": list(fetch.notes),
    }
    doc = {
        "tool": "tools/fetch-dense-originals.py",
        "repo": args.repo,
        "revision": args.revision,
        "base_url": f"https://huggingface.co/{args.repo}/resolve/{args.revision}/",
        "derived_from": str(args.sidecar.relative_to(ROOT)) if args.sidecar.is_relative_to(ROOT) else str(args.sidecar),
        "selection_rule": group["rule"],
        "file": group["out_file"].name,
        "layer": group["layer"],
        "file_bytes": total_file,
        "header_bytes": len(header_bytes),
        "tensors_total": len(entries),
        "tensors_recorded": len(recs),
        "values": sum(e["n"] for e in entries),
        "payload_bytes": sum(e["nbytes"] for e in entries),
        "runs": runs + [run],
        "tensors": [
            {
                "name": e["name"], "dtype": e["dtype"], "shape": e["shape"], "n": e["n"],
                "shard": e["shard"], "shard_begin": e["shard_begin"], "shard_end": e["shard_end"],
                "nbytes": e["nbytes"], "sha256": e["sha256"], "out_offset": e["out_offset"],
            }
            for e in recs
        ],
    }
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=1), encoding="utf-8")
    os.replace(tmp, path)


if __name__ == "__main__":
    sys.exit(main())
