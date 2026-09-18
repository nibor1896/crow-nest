#!/usr/bin/env python3
"""#76: fetch the BF16 originals of the DENSE text path back out of the Hugging Face repo.

  tools/fetch-dense-originals.py --dry-run     # the plan, no network at all
  tools/fetch-dense-originals.py               # fetch what is missing, verify everything
  tools/fetch-dense-originals.py               # a second run fetches nothing and re-verifies

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

USER_AGENT = "crow-nest-fetch-dense-originals/1 (+https://github.com/nibor1896/crow-nest)"

# One round trip costs about half a second on this line, which is worth roughly 5 MB of
# transfer at the 11 MB/s measured on 2026-09-18. Coalescing two tensor ranges that are
# closer than this therefore pays even though the gap between them is downloaded and thrown
# away; the span cap keeps one response in a bounded buffer.
DEFAULT_MAX_GAP = 2 * 1024 * 1024
DEFAULT_MAX_SPAN = 128 * 1024 * 1024
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


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--repo", default=DEFAULT_REPO)
    ap.add_argument("--revision", default=DEFAULT_REVISION)
    ap.add_argument("--sidecar", type=Path, default=DEFAULT_SIDECAR, help="the container sidecar the list is derived from")
    ap.add_argument("--index", type=Path, default=DEFAULT_INDEX, help="model.safetensors.index.json (tensor -> shard)")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and exit; no network at all")
    ap.add_argument("--skip-verify", action="store_true", help="trust the manifest instead of re-hashing what is on disk")
    ap.add_argument("--max-gap", type=int, default=DEFAULT_MAX_GAP)
    ap.add_argument("--max-span", type=int, default=DEFAULT_MAX_SPAN)
    ap.add_argument("--max-tries", type=int, default=6)
    ap.add_argument("--limit", type=int, default=0, help="stop after N tensors (for a short live smoke test)")
    args = ap.parse_args(argv)

    selection = select_dense_tensors(args.sidecar.read_text(encoding="utf-8").splitlines())
    weight_map = json.loads(args.index.read_text(encoding="utf-8"))["weight_map"]
    missing = [t["name"] for t in selection if t["name"] not in weight_map]
    if missing:
        print(f"{len(missing)} selected tensors are not in the model index, first: {missing[0]}", file=sys.stderr)
        return 2
    for t in selection:
        t["shard"] = weight_map[t["name"]]

    # the layout of the output file: name order, contiguous, no holes
    selection.sort(key=lambda t: t["name"])
    if args.limit:
        selection = selection[: args.limit]
    shards = sorted({t["shard"] for t in selection})
    total_values = sum(t["n"] for t in selection)
    expected_bytes = total_values * 2  # BF16 per the container's record; checked against every header

    print(f"selection: {len(selection)} tensors, {total_values:,} values, {human(expected_bytes)} at BF16")
    print(f"           derived from {args.sidecar.name}: section text, dtype nvfp4, not .mlp.experts.")
    print(f"shards:    {len(shards)} of 131")

    if args.dry_run:
        per_shard = {}
        for t in selection:
            e = per_shard.setdefault(t["shard"], [0, 0])
            e[0] += 1
            e[1] += t["n"] * 2
        for shard in shards:
            cnt, by = per_shard[shard]
            print(f"  {shard}  {cnt:3d} tensors  {by:>12,} B")
        print(f"total:     {human(expected_bytes)} of tensor payload, plus {len(shards)} shard headers")
        print("dry run: nothing fetched. The exact byte ranges come from the shard headers at fetch time.")
        return 0

    args.out.mkdir(parents=True, exist_ok=True)
    out_file = args.out / OUT_NAME
    manifest_path = args.out / MANIFEST_NAME
    header_cache = args.out / ".shard-headers"

    base_url = f"https://huggingface.co/{args.repo}/resolve/{args.revision}/"
    fetch = RangeFetcher(base_url, max_tries=args.max_tries)
    t0 = time.monotonic()
    started = datetime.now(timezone.utc).isoformat(timespec="seconds")

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
            return 2
        dtype = info["dtype"]
        shape = list(info["shape"])
        n = 1
        for d in shape:
            n *= d
        begin, end = info["data_offsets"]
        nbytes = end - begin
        if n != t["n"]:
            print(f"{t['name']}: shard says {n} values, the container says {t['n']}", file=sys.stderr)
            return 2
        if nbytes != n * ELEM_SIZE.get(dtype, 0):
            print(f"{t['name']}: {nbytes} B does not match {n} x {dtype}", file=sys.stderr)
            return 2
        entries.append({
            "name": t["name"], "dtype": dtype, "shape": shape, "n": n,
            "shard": t["shard"], "shard_begin": data_start + begin, "shard_end": data_start + end,
            "nbytes": nbytes,
        })

    dtypes = sorted({e["dtype"] for e in entries})
    print(f"dtype:     {', '.join(dtypes)} in the shard headers")

    meta = {
        "format": "pt",
        "crow_nest_source_repo": args.repo,
        "crow_nest_revision": args.revision,
        "crow_nest_selection": "text section, nvfp4 in CNQ4.5-M, not .mlp.experts.",
    }
    header_bytes, out_offsets, data_bytes = build_safetensors_header(entries, metadata=meta)
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
                blob = fetch.get(shard, g["begin"], g["end"])
                for m in g["members"]:
                    e = m["entry"]
                    lo = e["shard_begin"] - g["begin"]
                    chunk = blob[lo : lo + e["nbytes"]]
                    fh.seek(e["out_offset"])
                    fh.write(chunk)
                    e["sha256"] = hashlib.sha256(chunk).hexdigest()
                    shard_bytes += e["nbytes"]
                del blob
            fh.flush()
            fetched_bytes += shard_bytes
            dt = max(1e-9, time.monotonic() - t_shard)
            print(f"  [{si:2d}/{len(by_shard)}] {shard}  {len(by_shard[shard]):3d} tensors in "
                  f"{len(groups)} request(s)  {shard_bytes:>12,} B  {shard_bytes / dt / 1e6:6.1f} MB/s")
            # the manifest is rewritten after every shard: an interrupted run resumes here
            write_manifest(manifest_path, args, entries, header_bytes, total_file, runs, started, fetch, t0,
                           fetched_bytes, header_requests, groups_total, partial=True)
    finally:
        fh.close()

    elapsed = time.monotonic() - t0
    write_manifest(manifest_path, args, entries, header_bytes, total_file, runs, started, fetch, t0,
                   fetched_bytes, header_requests, groups_total, partial=False)

    rate = fetched_bytes / elapsed / 1e6 if elapsed > 0 else 0.0
    print(f"done:      {len(entries)} tensors, {verified} re-verified, {len(todo)} fetched")
    print(f"           {fetch.requests} requests ({header_requests} headers, {groups_total} data), "
          f"{fetch.retries} retries, {fetch.rate_limited} HTTP 429")
    print(f"           {human(fetched_bytes)} in {elapsed:.1f} s, mean {rate:.1f} MB/s")
    print(f"           {out_file} ({total_file:,} B), {manifest_path}")
    for note in fetch.notes:
        print(f"           note: {note}")
    return 0


def write_manifest(path, args, entries, header_bytes, total_file, runs, started, fetch, t0,
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
        "selection_rule": 'section == "text" and dtype == "nvfp4" and ".mlp.experts." not in name',
        "file": OUT_NAME,
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
