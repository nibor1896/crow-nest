#!/usr/bin/env python3
"""#154 / #156 / #157: fetch GLM-5.3-Flash from its original source, resumable, every file verified.

  python -I tools/fetch-glm.py --small                          # the 9 small files (config, index, tokenizer, ...)
  python -I tools/fetch-glm.py --headers                        # 62 shard headers by HTTP range -> headers/
  python -I tools/fetch-glm.py --shards 1,2,61-62               # whole shards (numbers or file names)
  python -I tools/fetch-glm.py --for-layers 0-3 --with-embed-head          # shards that hold those tensors
  python -I tools/fetch-glm.py --for-layers 0-3 --with-embed-head --plan   # only print that shard set
  python -I tools/fetch-glm.py --status                         # one line per file, no network

Source: `zai-org/GLM-5.3-Flash` at revision `eb9eb208eb0d988989d07a6a12d0fdeb5f52574a`, nothing else.
The reference for every file is the HF API of that revision
(`/api/models/<repo>/revision/<rev>?blobs=true`), saved once as `hf-revision.json` beside the
data: per file `size`, the git `blobId` (SHA-1 over "blob <size>\\0" + bytes) and for LFS files
`lfs.sha256`. A file is only ever called verified after its size AND its hash equal that record;
a mismatch deletes the file and fetches it again (at most `MAX_MISMATCHES` times, then stop).

Hugging Face drops long transfers (TLS resets, WinError 10054, curl exit 35), so every whole file
goes through two retry layers:
  1. curl itself: `-L -f -C - --retry 0`, plus stall detection `--speed-limit`/`--speed-time`.
     curl's own retry is OFF on purpose: it does not resume, it truncates the output back to
     where that curl run started (curl docs/TODO.md, "--retry should resume"), so every drop
     would cost the bytes of the current run - up to a whole shard.
  2. this script: the ONLY retry for whole files - an unlimited outer loop that restarts curl
     with `-C -` (which DOES resume from the bytes on disk) with backoff 10 s -> 300 s, until the
     byte count equals the API size, one log line per restart; plus a watchdog that kills a
     curl whose file has not grown for `STALL_SEC`. Small range requests (headers, API) keep
     curl's `--retry 20 --retry-all-errors`, where a rewind costs at most 256 KiB.
Before each shard the free space is checked: a shard that would leave less than 20 GB free on
the destination volume is refused and the run stops.

Everything is logged with a timestamp to `models/GLM-5.3-Flash-original/fetch.log`.
Standard library only; downloaded files are only read as data (run with `python -I`).
Unit tests for the pure parts: `tools/test_fetch_glm.py` (no network). Docs: docs/glm-download.md.
"""

import argparse
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

REPO = "zai-org/GLM-5.3-Flash"
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
API_URL = f"https://huggingface.co/api/models/{REPO}/revision/{REVISION}?blobs=true"
RESOLVE_URL = f"https://huggingface.co/{REPO}/resolve/{REVISION}/"
DEFAULT_DEST = ROOT / "models" / "GLM-5.3-Flash-original"
API_FILE = "hf-revision.json"
LOG_FILE = "fetch.log"
LOCK_FILE = "fetch.lock"
HEADER_DIR = "headers"

# the plan's 9 small files (28,715,382 B at this revision); `.gitattributes` and
# `.eval_results/` are repo bookkeeping, not model files
SMALL_FILES = (
    "LICENSE",
    "README.md",
    "chat_template.jinja",
    "config.json",
    "generation_config.json",
    "model.safetensors.index.json",
    "processor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
)
N_SHARDS = 62

GB = 1000 ** 3
MIN_FREE_AFTER = 20 * GB          # owner rule: never leave less than 20 GB free
HEADER_PROBE = 256 * 1024         # first range request of a shard header
MAX_HEADER = 100 * 1000 * 1000    # safetensors caps the header at 100 MB

CURL_RETRY = ["--retry", "20", "--retry-all-errors", "--retry-delay", "10"]   # small range requests only
# whole files: curl must never retry by itself - its retry rewinds to the start of the run
# (curl docs/TODO.md, "--retry should resume"); the outer `-C -` loop is the only retry
CURL_FILE_RETRY = ["--retry", "0"]
SPEED_LIMIT = 256 * 1024          # B/s; below this for SPEED_TIME s curl aborts (and retries)
SPEED_TIME = 120
CONNECT_TIMEOUT = 30
STALL_SEC = 600                   # watchdog: no growth for this long while curl lives -> kill
POLL_SEC = 10
PROGRESS_SEC = 300                # one progress line per file per 5 minutes
BACKOFF_START = 10
BACKOFF_MAX = 300
MAX_MISMATCHES = 3
LOCK_TTL = 120                    # a lock whose heartbeat is older than this is stale
PERMANENT_HTTP = {401, 403, 404, 410}
WRITE_OUT = "%{http_code} %{num_retries} %{size_download} %{exitcode}"

LANG_LAYERS = "model.language_model.layers."
FINAL_NORMS = ("model.language_model.norm.", "model.norm.")


# ---------------------------------------------------------------- pure parts


def file_table(api):
    """{name: {"size", "blob", "sha256"}} from the revision API JSON; refuses a foreign revision."""
    if api.get("sha") != REVISION:
        raise ValueError(f"API answer is for revision {api.get('sha')!r}, not {REVISION}")
    out = {}
    for s in api.get("siblings", []):
        lfs = s.get("lfs") or {}
        if lfs and lfs.get("size") != s.get("size"):
            raise ValueError(f"{s['rfilename']}: size {s.get('size')} != lfs.size {lfs.get('size')}")
        out[s["rfilename"]] = {"size": int(s["size"]), "blob": s.get("blobId"), "sha256": lfs.get("sha256")}
    return out


def shard_names(table):
    """The model shards of the table, sorted (= numeric order for `model-000NN-of-000MM`)."""
    return sorted(n for n in table if n.startswith("model-") and n.endswith(".safetensors"))


def check_table(table):
    """Refuse a table that does not have the 9 small files and 62 shards with hashes."""
    missing = [n for n in SMALL_FILES if n not in table]
    if missing:
        raise ValueError(f"API has no {', '.join(missing)}")
    shards = shard_names(table)
    if len(shards) != N_SHARDS:
        raise ValueError(f"API lists {len(shards)} shards, expected {N_SHARDS}")
    nohash = [n for n in shards if not table[n]["sha256"]]
    if nohash:
        raise ValueError(f"no lfs.sha256 for {nohash[0]} (+{len(nohash) - 1})")
    return shards


def git_blob_sha1(data_or_chunks, size):
    """Git blob id: SHA-1 over b"blob <size>\\0" + bytes. Accepts bytes or an iterable of chunks."""
    h = hashlib.sha1(b"blob %d\x00" % size)
    if isinstance(data_or_chunks, (bytes, bytearray)):
        h.update(data_or_chunks)
    else:
        for c in data_or_chunks:
            h.update(c)
    return h.hexdigest()


def parse_shard_spec(spec, shards):
    """`1,3-5` or file names -> list of shard names (sorted, distinct). 1-based numbers."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if part in shards:
            out.add(part)
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"shard range {part!r} runs backwards")
            nums = range(lo, hi + 1)
        else:
            nums = [int(part)]
        for n in nums:
            if not 1 <= n <= len(shards):
                raise ValueError(f"shard {n} is not in 1..{len(shards)}")
            out.add(shards[n - 1])
    if not out:
        raise ValueError("no shards selected")
    return sorted(out)


def parse_layers(spec):
    """`0-3` or `0,1,5-7` -> sorted distinct layer numbers. Refuses anything else."""
    out = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(x) for x in part.split("-", 1))
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


def lang_layer_of(name):
    """N of `model.language_model.layers.N.…`, else None (vision blocks never match)."""
    if not name.startswith(LANG_LAYERS):
        return None
    head = name[len(LANG_LAYERS):].split(".", 1)[0]
    return int(head) if head.isdigit() else None


def tensor_kind(name):
    """'embed' / 'head' / 'final_norm' for the non-layer tensors step 6 needs, else None."""
    if lang_layer_of(name) is not None:
        return None
    if name.startswith("model.visual.") or name.startswith("visual."):
        return None
    if name.endswith("embed_tokens.weight"):
        return "embed"
    if name == "lm_head.weight" or name.endswith(".lm_head.weight"):
        return "head"
    if name.startswith(FINAL_NORMS):
        return "final_norm"
    return None


def select_shards(weight_map, layers, with_embed_head):
    """(sorted shard names, {what: tensor count}, {shard: [what...]}) for the given layers.

    Refuses a requested layer without a tensor, and (with `with_embed_head`) a missing
    embedding, head or final norm - an empty selection is a typo, not a plan.
    """
    want = set(layers)
    counts = {}
    reasons = {}
    for name, shard in weight_map.items():
        layer = lang_layer_of(name)
        if layer is not None and layer in want:
            what = f"layer {layer}"
        elif with_embed_head and tensor_kind(name):
            what = tensor_kind(name)
        else:
            continue
        counts[what] = counts.get(what, 0) + 1
        reasons.setdefault(shard, set()).add(what)
    missing = [f"layer {n}" for n in sorted(want) if f"layer {n}" not in counts]
    if with_embed_head:
        missing += [k for k in ("embed", "head", "final_norm") if k not in counts]
    if missing:
        raise ValueError(f"no tensor in the index for: {', '.join(missing)}")
    return sorted(reasons), counts, {s: sorted(r) for s, r in reasons.items()}


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


class NeedMore(Exception):
    def __init__(self, total):
        super().__init__(f"need {total} bytes")
        self.total = total


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


def decide(have, expected):
    """What to do with a partial file of `have` bytes: 'done', 'resume' or 'overlong'."""
    if have == expected:
        return "done"
    if have > expected:
        return "overlong"
    return "resume"


def next_backoff(current):
    return min(BACKOFF_MAX, max(BACKOFF_START, current * 2))


def disk_verdict(free, remaining, floor=MIN_FREE_AFTER):
    """(ok, free after the file) - a file that would leave less than `floor` is refused."""
    left = free - remaining
    return left >= floor, left


def lock_is_stale(lock_mtime, now, ttl=LOCK_TTL):
    return now - lock_mtime > ttl


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


def curl_file_argv(curl, url, out):
    """One whole-file curl run: resume from the bytes on disk, no curl-internal retry (it would
    rewind the file to where this run started), abort on a stall."""
    return [curl, "-L", "-f", "-sS", "-C", "-", *CURL_FILE_RETRY,
            "--connect-timeout", str(CONNECT_TIMEOUT),
            "--speed-limit", str(SPEED_LIMIT), "--speed-time", str(SPEED_TIME),
            "-w", WRITE_OUT, "-o", str(out), url]


def curl_range_argv(curl, url, out, begin, end):
    """Bytes [begin, end) into `out`. `--max-filesize` stops a server that ignores the range
    from sending the whole 5 GB shard."""
    return [curl, "-L", "-f", "-sS", *CURL_RETRY, "--connect-timeout", str(CONNECT_TIMEOUT),
            "--max-time", "300", "--max-filesize", str(end - begin),
            "-r", f"{begin}-{end - 1}", "-w", WRITE_OUT, "-o", str(out), url]


def fmt_bytes(n):
    return f"{n:,}"


def status_line(name, have, expected, verified):
    return f"{name:<36} {fmt_bytes(have):>15} / {fmt_bytes(expected):<15} verified {'yes' if verified else 'no'}"


# ---------------------------------------------------------------- io


class Log:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg):
        line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        try:
            print(line, flush=True)
        except (OSError, ValueError, AttributeError):
            pass  # detached without a console


class Lock:
    """One downloader per destination. Heartbeat by mtime; a stale lock is taken over."""

    def __init__(self, path, log):
        self.path = Path(path)
        self.log = log
        self.held = False

    def acquire(self):
        for _ in range(2):
            try:
                fd = os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(fd, f"{os.getpid()} {datetime.now().isoformat(timespec='seconds')}\n".encode())
                os.close(fd)
                self.held = True
                return
            except FileExistsError:
                try:
                    st = self.path.stat()
                except FileNotFoundError:
                    continue
                if lock_is_stale(st.st_mtime, time.time()):
                    self.log(f"lock {self.path.name}: stale (heartbeat {time.time() - st.st_mtime:.0f} s old) -> taking over")
                    self.path.unlink(missing_ok=True)
                    continue
                owner = self.path.read_text(encoding="utf-8", errors="replace").strip()
                raise SystemExit(f"another fetch-glm run holds {self.path} ({owner}); --status works meanwhile")
        raise SystemExit(f"could not take {self.path}")

    def beat(self):
        if self.held:
            try:
                os.utime(self.path)
            except OSError:
                pass

    def release(self):
        if self.held:
            self.path.unlink(missing_ok=True)
            self.held = False


class Stats:
    def __init__(self):
        self.curl_runs = 0
        self.curl_retries = 0
        self.restarts = 0
        self.mismatches = 0
        self.bytes_added = 0
        self.requests = 0


def file_chunks(path, chunk=8 << 20):
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                return
            yield b


def hashes_of(path, size):
    """(sha256, git blob sha1) of a file in one pass."""
    s256 = hashlib.sha256()
    s1 = hashlib.sha1(b"blob %d\x00" % size)
    for c in file_chunks(path):
        s256.update(c)
        s1.update(c)
    return s256.hexdigest(), s1.hexdigest()


def check_hashes(meta, sha256, sha1):
    """None if the file matches its API record, else the reason."""
    if meta["sha256"] and sha256 != meta["sha256"]:
        return f"sha256 {sha256} != lfs.sha256 {meta['sha256']}"
    if not meta["sha256"] and meta["blob"] and sha1 != meta["blob"]:
        return f"git blob {sha1} != blobId {meta['blob']}"
    return None


def marker_of(path):
    return Path(str(path) + ".verified")


def is_verified(path, meta):
    """Cheap check (no hashing): final file of the right size with a matching marker."""
    path = Path(path)
    m = marker_of(path)
    if not path.exists() or not m.exists() or path.stat().st_size != meta["size"]:
        return False
    try:
        rec = json.loads(m.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if rec.get("size") != meta["size"]:
        return False
    if meta["sha256"]:
        return rec.get("sha256") == meta["sha256"]
    return rec.get("blob") == meta["blob"]


def which_curl(arg):
    curl = arg or shutil.which("curl")
    if not curl:
        raise SystemExit("curl not found on PATH (pass --curl)")
    return curl


def run_curl(argv, watch, log, lock, label, expected=None):
    """Run one curl, watching `watch` grow. Returns (exit code, write-out dict, stderr tail).

    Logs progress every PROGRESS_SEC; kills curl when the file has not grown for STALL_SEC.
    """
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    t0 = last_grow = last_prog = time.monotonic()
    size0 = last_size = prog_size = watch.stat().st_size if watch.exists() else 0
    out = err = b""
    killed = False
    try:
        while True:
            try:
                out, err = proc.communicate(timeout=POLL_SEC)
                break
            except subprocess.TimeoutExpired:
                pass
            lock.beat()
            now = time.monotonic()
            size = watch.stat().st_size if watch.exists() else 0
            if size != last_size:
                last_size, last_grow = size, now
            if now - last_prog >= PROGRESS_SEC:
                rate = (size - prog_size) / (now - last_prog) / 1e6
                tot = f" / {fmt_bytes(expected)}" if expected else ""
                log(f"progress {label}: {fmt_bytes(size)}{tot} B, {rate:.1f} MB/s over the last {now - last_prog:.0f} s")
                last_prog, prog_size = now, size
            if now - last_grow >= STALL_SEC:
                log(f"STALL {label}: no growth for {now - last_grow:.0f} s at {fmt_bytes(size)} B -> kill curl pid {proc.pid}")
                proc.kill()
                killed = True
                out, err = proc.communicate()
                break
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.communicate()
    wo = parse_write_out(out.decode("ascii", "replace"))
    tail = " | ".join(err.decode("utf-8", "replace").strip().splitlines()[-3:])
    rc = proc.returncode if not killed else -9
    size = watch.stat().st_size if watch.exists() else 0
    secs = time.monotonic() - t0
    log(f"curl {label}: exit {rc}, http {wo['http']}, curl-retries {wo['retries']}, "
        f"+{fmt_bytes(size - size0)} B on disk in {secs:.0f} s ({(size - size0) / max(secs, 1e-9) / 1e6:.1f} MB/s)"
        + (f", stderr: {tail}" if tail else ""))
    return rc, wo, tail


def fetch_file(name, meta, dest, ctx, final=None):
    """Download one whole file resumably and verify it. True on success, False on refusal."""
    log, stats = ctx["log"], ctx["stats"]
    final = Path(final or dest / name)
    part = Path(str(final) + ".part")
    if is_verified(final, meta):
        log(f"skip {name}: verified ({fmt_bytes(meta['size'])} B)")
        return True
    if final.exists():
        # a final file without a marker: hash it once, keep it only if it matches
        log(f"check {name}: on disk without marker, hashing")
        part.unlink(missing_ok=True)
        os.replace(final, part)
    final.parent.mkdir(parents=True, exist_ok=True)
    url = RESOLVE_URL + name
    backoff = BACKOFF_START
    mismatches = 0
    while True:
        have = part.stat().st_size if part.exists() else 0
        state = decide(have, meta["size"])
        if state == "overlong":
            log(f"OVERLONG {name}: {fmt_bytes(have)} B > {fmt_bytes(meta['size'])} B -> delete .part, refetch")
            part.unlink()
            continue
        if state == "done":
            t0 = time.monotonic()
            sha256, sha1 = hashes_of(part, meta["size"])
            bad = check_hashes(meta, sha256, sha1)
            if bad:
                stats.mismatches += 1
                mismatches += 1
                log(f"MISMATCH {name} ({mismatches}/{MAX_MISMATCHES}): {bad} -> delete, never keep")
                part.unlink()
                if mismatches >= MAX_MISMATCHES:
                    log(f"STOP {name}: {mismatches} hash mismatches in a row")
                    return False
                continue
            os.replace(part, final)
            marker_of(final).write_text(json.dumps({
                "file": name, "size": meta["size"], "sha256": sha256, "blob": sha1,
                "repo": REPO, "revision": REVISION,
                "verified_at": datetime.now().isoformat(timespec="seconds")}) + "\n", encoding="utf-8")
            how = "lfs.sha256" if meta["sha256"] else "git blobId"
            log(f"VERIFIED {name}: {fmt_bytes(meta['size'])} B, sha256 {sha256} ({how} match, hashed in {time.monotonic() - t0:.0f} s)")
            return True
        if ctx.get("disk_check"):
            free = shutil.disk_usage(final.parent).free
            ok, left = disk_verdict(free, meta["size"] - have)
            if not ok:
                log(f"REFUSED {name}: needs {fmt_bytes(meta['size'] - have)} B more, {fmt_bytes(free)} B free "
                    f"would leave {left / GB:.1f} GB < {MIN_FREE_AFTER / GB:.0f} GB")
                ctx["refused"] = True
                return False
        if have:
            log(f"resume {name} at {fmt_bytes(have)} / {fmt_bytes(meta['size'])} B")
        else:
            log(f"start {name}: {fmt_bytes(meta['size'])} B")
        stats.curl_runs += 1
        rc, wo, _ = run_curl(curl_file_argv(ctx["curl"], url, part), part, log, ctx["lock"], name, meta["size"])
        stats.curl_retries += wo["retries"]
        now_have = part.stat().st_size if part.exists() else 0
        stats.bytes_added += max(0, now_have - have)
        if rc == 0 and now_have == meta["size"]:
            continue  # -> verify
        if wo["http"] in PERMANENT_HTTP:
            log(f"STOP {name}: HTTP {wo['http']} is permanent (wrong repo/revision/access), no retry")
            return False
        if wo["http"] == 416:
            log(f"{name}: HTTP 416 at {fmt_bytes(now_have)} B (range not satisfiable) -> re-check size after backoff")
        if now_have > have:
            backoff = BACKOFF_START
        stats.restarts += 1
        log(f"restart {name} in {backoff} s (outer loop; {fmt_bytes(now_have)} / {fmt_bytes(meta['size'])} B)")
        time.sleep(backoff)
        backoff = next_backoff(backoff)


def fetch_range(name, begin, end, tmp, ctx, tries=5):
    """Bytes [begin, end) of a remote file, exactly, or an exception."""
    log, stats = ctx["log"], ctx["stats"]
    backoff = BACKOFF_START
    for attempt in range(1, tries + 1):
        tmp.unlink(missing_ok=True)
        stats.requests += 1
        stats.curl_runs += 1
        rc, wo, tail = run_curl(curl_range_argv(ctx["curl"], RESOLVE_URL + name, tmp, begin, end),
                                tmp, log, ctx["lock"], f"{name} [{begin},{end})")
        stats.curl_retries += wo["retries"]
        data = tmp.read_bytes() if tmp.exists() else b""
        tmp.unlink(missing_ok=True)
        if rc == 0 and wo["http"] == 206 and len(data) == end - begin:
            stats.bytes_added += len(data)
            return data
        if wo["http"] in PERMANENT_HTTP:
            raise RuntimeError(f"{name}: HTTP {wo['http']}")
        log(f"range {name} [{begin},{end}): exit {rc}, http {wo['http']}, got {len(data)} B (try {attempt}/{tries})")
        if attempt < tries:
            stats.restarts += 1
            time.sleep(backoff)
            backoff = next_backoff(backoff)
    raise RuntimeError(f"{name}: range [{begin},{end}) failed {tries} times")


def load_api(dest, ctx, refresh=False):
    """The revision's API record, from `hf-revision.json` or fetched once (immutable revision)."""
    path = dest / API_FILE
    if refresh or not path.exists():
        tmp = Path(str(path) + ".part")
        ctx["log"](f"api: GET {API_URL}")
        argv = [ctx["curl"], "-L", "-f", "-sS", *CURL_RETRY, "--connect-timeout", str(CONNECT_TIMEOUT),
                "-w", WRITE_OUT, "-o", str(tmp), API_URL]
        rc, wo, tail = run_curl(argv, tmp, ctx["log"], ctx["lock"], API_FILE)
        if rc != 0:
            raise SystemExit(f"API request failed: exit {rc}, http {wo['http']} {tail}")
        table = file_table(json.loads(tmp.read_text(encoding="utf-8")))
        check_table(table)
        os.replace(tmp, path)
        ctx["log"](f"api: saved {path.name} ({len(table)} files)")
    api = json.loads(path.read_text(encoding="utf-8"))
    table = file_table(api)
    return table, check_table(table)


# ---------------------------------------------------------------- modes


def mode_small(dest, table, ctx):
    ok = True
    for name in SMALL_FILES:
        ok &= fetch_file(name, table[name], dest, ctx)
    total = sum(table[n]["size"] for n in SMALL_FILES)
    ctx["log"](f"small: {sum(is_verified(dest / n, table[n]) for n in SMALL_FILES)}/{len(SMALL_FILES)} verified, {fmt_bytes(total)} B expected")
    return 0 if ok else 3


def read_local_prefix(path, n):
    with open(path, "rb") as f:
        return f.read(n)


def mode_headers(dest, table, shards, ctx):
    log = ctx["log"]
    hdir = dest / HEADER_DIR
    hdir.mkdir(parents=True, exist_ok=True)
    count = header_bytes = tensor_bytes = tensors = 0
    for shard in shards:
        size = table[shard]["size"]
        cache = hdir / (shard + ".json")
        if cache.exists():
            rec = json.loads(cache.read_text(encoding="utf-8"))
            header, data_start = rec["header"], rec["data_start"]
        else:
            local = dest / shard
            if is_verified(local, table[shard]):
                blob = read_local_prefix(local, HEADER_PROBE)
            else:
                blob = fetch_range(shard, 0, min(HEADER_PROBE, size), hdir / (shard + ".tmp"), ctx)
            try:
                header, data_start = parse_safetensors_header(blob)
            except NeedMore as more:
                if is_verified(local, table[shard]):
                    blob = read_local_prefix(local, more.total)
                else:
                    blob += fetch_range(shard, len(blob), more.total, hdir / (shard + ".tmp"), ctx)
                header, data_start = parse_safetensors_header(blob)
        n, tb = header_summary(header, data_start, size)
        if not cache.exists():
            cache.write_text(json.dumps({"shard": shard, "size": size, "data_start": data_start,
                                         "header": header}), encoding="utf-8")
            log(f"header {shard}: {data_start} B header, {n} tensors, {fmt_bytes(tb)} B tensors (tiles the file)")
        count += 1
        header_bytes += data_start
        tensor_bytes += tb
        tensors += n
    total = sum(table[s]["size"] for s in shards)
    log(f"headers: {count}/{len(shards)} shards, {tensors} tensors, header bytes {fmt_bytes(header_bytes)} B "
        f"(8-byte length incl.), tensor bytes {fmt_bytes(tensor_bytes)} B, sum {fmt_bytes(header_bytes + tensor_bytes)} "
        f"{'==' if header_bytes + tensor_bytes == total else '!='} shard bytes {fmt_bytes(total)}")
    return 0 if header_bytes + tensor_bytes == total else 3


def mode_shards(dest, table, names, ctx):
    log = ctx["log"]
    total = sum(table[n]["size"] for n in names)
    log(f"shards: {len(names)} to fetch, {fmt_bytes(total)} B: {', '.join(names)}")
    ctx["disk_check"] = True
    for name in names:
        if not fetch_file(name, table[name], dest, ctx):
            log(f"shards: stopped at {name}")
            return 4 if ctx.get("refused") else 3
    log(f"shards: {sum(is_verified(dest / n, table[n]) for n in names)}/{len(names)} verified")
    return 0


def resolve_layers(dest, table, layers, with_embed_head):
    index = dest / "model.safetensors.index.json"
    if not is_verified(index, table["model.safetensors.index.json"]):
        raise SystemExit(f"{index} is not verified - run --small first")
    weight_map = json.loads(index.read_text(encoding="utf-8"))["weight_map"]
    return select_shards(weight_map, layers, with_embed_head)


def mode_status(dest, table, shards):
    lines = []
    done = expected = nver = 0
    for name in list(SMALL_FILES) + shards:
        meta = table[name]
        final = dest / name
        part = Path(str(final) + ".part")
        have = final.stat().st_size if final.exists() else (part.stat().st_size if part.exists() else 0)
        ver = is_verified(final, meta)
        nver += ver
        done += have
        expected += meta["size"]
        lines.append(status_line(name, have, meta["size"], ver))
    hdir = dest / HEADER_DIR
    nh = sum((hdir / (s + ".json")).exists() for s in shards)
    lines.append(f"headers cached: {nh}/{len(shards)}")
    lines.append(f"total: {fmt_bytes(done)} / {fmt_bytes(expected)} B, {nver}/{len(SMALL_FILES) + len(shards)} files verified")
    print("\n".join(lines))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--small", action="store_true", help="the 9 small files")
    ap.add_argument("--headers", action="store_true", help="62 shard headers by HTTP range -> headers/")
    ap.add_argument("--shards", help="shard numbers (1-based, ranges) or file names")
    ap.add_argument("--for-layers", help="language-model layers, e.g. 0-3; resolves the shard set from the index")
    ap.add_argument("--with-embed-head", action="store_true", help="with --for-layers: add embed_tokens, lm_head, final norm")
    ap.add_argument("--plan", action="store_true", help="with --for-layers: print the shard set, fetch nothing")
    ap.add_argument("--status", action="store_true", help="one line per file, no network")
    ap.add_argument("--dest", default=str(DEFAULT_DEST))
    ap.add_argument("--curl", help="curl binary (default: curl on PATH)")
    ap.add_argument("--refresh-api", action="store_true", help="re-read the API record instead of hf-revision.json")
    args = ap.parse_args(argv)

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    if not any([args.small, args.headers, args.shards, args.for_layers, args.status]):
        ap.error("pick a mode: --small, --headers, --shards, --for-layers or --status")

    if args.status:
        if not (dest / API_FILE).exists():
            raise SystemExit(f"no {API_FILE} in {dest} yet - run --small first")
        table = file_table(json.loads((dest / API_FILE).read_text(encoding="utf-8")))
        return mode_status(dest, table, check_table(table))

    if args.for_layers and args.plan:
        table = file_table(json.loads((dest / API_FILE).read_text(encoding="utf-8")))
        check_table(table)
        names, counts, reasons = resolve_layers(dest, table, parse_layers(args.for_layers), args.with_embed_head)
        for s in names:
            print(f"{s}  {fmt_bytes(table[s]['size']):>15} B  {', '.join(reasons[s])}")
        print(f"{len(names)} shards, {fmt_bytes(sum(table[s]['size'] for s in names))} B; tensors: "
              + ", ".join(f"{k} {v}" for k, v in sorted(counts.items())))
        return 0

    log = Log(dest / LOG_FILE)
    lock = Lock(dest / LOCK_FILE, log)
    lock.acquire()
    stats = Stats()
    ctx = {"log": log, "lock": lock, "stats": stats, "curl": which_curl(args.curl)}
    t0 = time.monotonic()
    rc = 0
    mode = " ".join(a for a in (argv if argv is not None else sys.argv[1:]))
    try:
        ver = subprocess.run([ctx["curl"], "--version"], capture_output=True, text=True).stdout.splitlines()[:1]
        log(f"run start (pid {os.getpid()}): {mode} | {ctx['curl']} {ver[0] if ver else '?'} | {REPO}@{REVISION}")
        table, shards = load_api(dest, ctx, refresh=args.refresh_api)
        if args.small:
            rc = rc or mode_small(dest, table, ctx)
        if args.headers and not rc:
            rc = mode_headers(dest, table, shards, ctx)
        if args.shards and not rc:
            rc = mode_shards(dest, table, parse_shard_spec(args.shards, shards), ctx)
        if args.for_layers and not rc:
            names, counts, reasons = resolve_layers(dest, table, parse_layers(args.for_layers), args.with_embed_head)
            log(f"for-layers {args.for_layers}{' +embed/head/norm' if args.with_embed_head else ''}: "
                + "; ".join(f"{s} ({', '.join(reasons[s])})" for s in names))
            rc = mode_shards(dest, table, names, ctx)
    except KeyboardInterrupt:
        log("interrupted")
        rc = 130
    except Exception as err:  # logged, then re-raised for the exit code and traceback
        log(f"ERROR {type(err).__name__}: {err}")
        raise
    finally:
        secs = time.monotonic() - t0
        log(f"run end: rc {rc}, {secs:.0f} s, +{fmt_bytes(stats.bytes_added)} B, {stats.curl_runs} curl runs "
            f"({stats.requests} range requests), {stats.curl_retries} curl-internal retries, "
            f"{stats.restarts} outer restarts, {stats.mismatches} hash mismatches")
        lock.release()
    return rc


if __name__ == "__main__":
    sys.exit(main())
