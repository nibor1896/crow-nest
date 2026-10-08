#!/usr/bin/env python3
"""#157: the delete-behind supervisor of the staged GLM-5.3-Flash conversion.

  python -I tools/glm-stage.py --out <container.cnq>              # supervise until 62/62 shards are deleted
  python -I tools/glm-stage.py --out <container.cnq> --once       # one pass, then exit
  python -I tools/glm-stage.py --out <container.cnq> --dry-run    # say what would be deleted, delete nothing
  python -I tools/glm-stage.py --out <container.cnq> --status     # one status line, no change, no lock

Three processes share `models/GLM-5.3-Flash-original/`: `tools/fetch-glm.py --shards 1-62
--wait-for-space` writes `<shard>` + `<shard>.verified` (size and sha256 equal to the HF record)
and parks at the 20 GB reserve; `converter --headers … --consume …` reads a shard after its
`.verified`, and writes `<shard>.done` once every tensor that reads it is written and journalled;
this script deletes a shard after that - and only then:

  1. `<shard>.done` exists and names this conversion's container (`out`),
  2. the shard is verified (`.verified` record == the HF record, file size == the record),
  3. the conversion's journal `<out>.journal.jsonl` holds every tensor of the shard that the
     `cnq4.5-glm5-next` recipe writes (the shard's header from `headers/<shard>.json`, minus the
     omitted vision tower and MTP layers >= `text_config.num_hidden_layers`; an FP8
     `X.weight_scale_inv` counts through its `X.weight`), inside the journal's verified prefix:
     records in sequence, contiguous offsets, zero alignment bytes, and the sha256 of the
     container bytes equal to the record's - the converter's own resume criterion, so a resume
     never needs a deleted shard again. After the trailer is written the journal is gone; then
     the container's index trailer (format_version 2) is the record instead.

Then it writes `<shard>.deleted` (size, sha256, journal position, time) and deletes the shard;
`fetch-glm.py` never fetches a shard with that marker again. A marker whose shard still exists
(a kill between the two steps) is completed on the next pass after the same checks. `.verified`,
`.done` and `.part` files are never deleted; nothing but the 62 shard files of the HF record is.
Every delete and every refusal is logged to `models/GLM-5.3-Flash-original/stage.log`; a status
line goes there every `--status-every` seconds (default 1500 = 25 minutes). Restartable at any
point: all state is in the files. Docs: docs/glm-staged-conversion.md. Tests: tools/test_glm_stage.py.
"""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import struct
import sys
import time
from datetime import datetime
from pathlib import Path

TOOLS = Path(__file__).resolve().parent
ROOT = TOOLS.parent
_SPEC = importlib.util.spec_from_file_location("fetch_glm", TOOLS / "fetch-glm.py")
fg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fg)

STAGE_LOG = "stage.log"
STAGE_LOCK = "stage.lock"
JOURNAL_HEAD = "crow-nest converter"
POLL_SEC = 30
STATUS_SEC = 1500                 # the owner's 25-minute size check (#157)
SCALE_SUFFIX = "_scale_inv"
MAGIC = b"CNQ1"
BLOB_START = 12
READ_CHUNK = 8 << 20


# ---------------------------------------------------------------- pure parts


def text_layers(config):
    """`text_config.num_hidden_layers` (GLM-5.3-Flash: 45); layers at or above it are MTP."""
    n = (config.get("text_config") or {}).get("num_hidden_layers", config.get("num_hidden_layers"))
    if not isinstance(n, int) or n <= 0:
        raise ValueError("config.json has no text_config.num_hidden_layers")
    return n


def lm_layer(name):
    """L of `model.language_model.layers.L.…`, else None (the converter's `glm_parts`)."""
    rest = name[len("model.language_model.layers."):] if name.startswith("model.language_model.layers.") else None
    if rest is None:
        return None
    head = rest.split(".", 1)[0]
    return int(head) if head.isdigit() else None


def omitted(name, n_layers):
    """True for what the `cnq4.5-glm5-next` recipe does not write (converter `recipe::omitted`):
    the vision tower (section `vit`) and the MTP block (language layers >= n_layers)."""
    if ".visual." in name or name.startswith("model.visual"):
        return True
    layer = lm_layer(name)
    return layer is not None and layer >= n_layers


def required_tensors(header_names, n_layers):
    """The names that must be in the journal before a shard holding `header_names` may go.

    An FP8 `X.weight_scale_inv` is read only through `X.weight` and is never written itself, so
    it requires `X.weight` - wherever that weight lives. Omitted tensors require nothing.
    """
    need = set()
    for n in header_names:
        if n == "__metadata__" or omitted(n, n_layers):
            continue
        if n.endswith(SCALE_SUFFIX):
            w = n[:-len(SCALE_SUFFIX)]
            if not omitted(w, n_layers):
                need.add(w)
            continue
        need.add(n)
    return need


def same_path(a, b, bases=()):
    """`a` (as the converter wrote it into `.done`) names the file `b`. A relative `a` is tried
    against each of `bases` (the converter's working directory is not recorded)."""
    want = os.path.normcase(os.path.abspath(b))
    cands = [Path(a)] if Path(a).is_absolute() else [Path(base) / a for base in bases]
    return any(os.path.normcase(os.path.abspath(c)) == want for c in cands)


def delete_verdict(shard, on_disk, verified, done, done_out_ok, need, written):
    """(may delete, reason). The guards, in order; every one must hold."""
    if not done:
        return False, "no .done"
    if not done_out_ok:
        return False, ".done names another container"
    if on_disk and not verified:
        return False, "not verified against the HF record"
    if written is None:
        return False, "no journal and no index trailer to check against"
    missing = sorted(need - written)
    if missing:
        return False, f"{len(missing)} of {len(need)} tensors not journalled (first {missing[0]})"
    return True, f"{len(need)} tensors journalled"


def journal_path(out):
    """`<out>` with its extension replaced by `cnq.journal.jsonl`, as the converter names it
    (Rust `out_path.with_extension("cnq.journal.jsonl")`)."""
    return Path(out).with_suffix(".cnq.journal.jsonl")


def fmt_gb(n):
    return f"{n / 1e9:.1f} GB"


# ---------------------------------------------------------------- journal / trailer


class Journal:
    """The verified prefix of `<out>.journal.jsonl`, extended incrementally.

    Mirrors the converter's `read_journal`: line 1 is the head; then records in `seq` order whose
    `entry.offset == previous end + pad`, whose alignment bytes in the container are zero and
    whose container bytes hash to `sha256`. The prefix stops at the first record that fails
    (a half-written last line has no newline yet and is not read). If the journal shrinks or the
    line at the end of the prefix changed (a converter resume truncated it), it starts over.
    """

    def __init__(self, journal_path, out_path):
        self.path = Path(journal_path)
        self.out = Path(out_path)
        self.reset()

    def reset(self):
        self.head = None
        self.pos = 0
        self.last_line = b""
        self.blob_end = 0
        self.n = 0
        self.names = set()
        self.stuck = None          # (seq, reason) of the first record that did not verify

    def refresh(self):
        """Extend the prefix. Returns the set of verified tensor names, or None (no journal)."""
        if not self.path.exists():
            return None
        size = self.path.stat().st_size
        with open(self.path, "rb") as f:
            if self.pos:
                f.seek(self.pos - len(self.last_line))
                if size < self.pos or f.read(len(self.last_line)) != self.last_line:
                    self.reset()
            f.seek(self.pos)
            data = f.read(size - self.pos)
        start = 0
        container = None
        try:
            while True:
                nl = data.find(b"\n", start)
                if nl < 0:
                    break
                line = data[start:nl + 1]
                if self.head is None:
                    head = json.loads(line)
                    if head.get("journal") != JOURNAL_HEAD:
                        raise ValueError(f"{self.path.name}: line 1 is not a {JOURNAL_HEAD} journal")
                    self.head = head
                else:
                    if container is None:
                        try:
                            container = open(self.out, "rb")
                        except OSError as e:
                            self.stuck = (self.n, f"container: {e}")
                            break
                    try:
                        rec = json.loads(line)
                    except ValueError:
                        rec = None
                    why = self._check(rec, container) if isinstance(rec, dict) else "not a JSON record"
                    if why:
                        self.stuck = (self.n, why)
                        break
                self.pos += len(line)
                self.last_line = line
                start = nl + 1
        finally:
            if container is not None:
                container.close()
        if self.stuck and self.stuck[0] != self.n:
            self.stuck = None
        return set(self.names)

    def _check(self, rec, f):
        """None if the record extends the verified prefix (and it is taken), else the reason."""
        try:
            seq, pad, sha = rec["seq"], rec["pad"], rec["sha256"]
            name, off, ln = rec["entry"]["name"], rec["entry"]["offset"], rec["entry"]["len"]
        except (KeyError, TypeError):
            return "record without seq/pad/sha256/entry"
        if seq != self.n:
            return f"seq {seq}, expected {self.n}"
        if off != self.blob_end + pad:
            return f"offset {off} != end {self.blob_end} + pad {pad}"
        f.seek(BLOB_START + self.blob_end)
        if pad and f.read(pad) != bytes(pad):
            return "alignment bytes are not zero"
        h = hashlib.sha256()
        left = ln
        while left:
            b = f.read(min(READ_CHUNK, left))
            if not b:
                return f"container ends inside {name}"
            h.update(b)
            left -= len(b)
        if h.hexdigest() != sha:
            return f"{name}: container bytes hash to {h.hexdigest()}, journal says {sha}"
        self.n += 1
        self.blob_end = off + ln
        self.names.add(name)
        return None


def read_trailer(out_path):
    """Tensor names of a finished container's index trailer (format_version 2), else None."""
    out = Path(out_path)
    try:
        size = out.stat().st_size
        with open(out, "rb") as f:
            if f.read(4) != MAGIC or size < BLOB_START + 8:
                return None
            f.seek(size - 8)
            n = struct.unpack("<Q", f.read(8))[0]
            if n <= 0 or BLOB_START + n + 8 > size:
                return None
            f.seek(size - 8 - n)
            idx = json.loads(f.read(n))
    except (OSError, ValueError):
        return None
    if not isinstance(idx, dict) or idx.get("format_version") != 2 or not isinstance(idx.get("tensors"), list):
        return None
    return {t["name"] for t in idx["tensors"] if isinstance(t, dict) and "name" in t}


# ---------------------------------------------------------------- the pass


class Stage:
    def __init__(self, dest, out, log, dry_run=False, bases=()):
        self.dest = Path(dest)
        self.out = Path(out)
        self.log = log
        self.dry = dry_run
        self.bases = tuple(bases) or (Path.cwd(), self.dest.parent.parent)
        api = json.loads((self.dest / fg.API_FILE).read_text(encoding="utf-8"))
        self.table = fg.file_table(api)
        self.shards = fg.check_table(self.table)
        cfg = json.loads((self.dest / "config.json").read_text(encoding="utf-8"))
        self.n_layers = text_layers(cfg)
        self.need = {}
        for s in self.shards:
            hp = self.dest / fg.HEADER_DIR / (s + ".json")
            if not hp.exists():
                raise SystemExit(f"{hp} is missing - run fetch-glm.py --headers first")
            self.need[s] = required_tensors(json.loads(hp.read_text(encoding="utf-8"))["header"], self.n_layers)
        self.journal = Journal(journal_path(self.out), self.out)
        self.blocked = {}            # shard -> last logged reason (log a refusal once, not every pass)
        self.deleted_now = 0

    def written(self):
        """(names, source): the verified journal prefix, else the trailer, else (None, None)."""
        names = self.journal.refresh()
        if names is not None:
            if self.journal.stuck:
                seq, why = self.journal.stuck
                key = ("journal", seq, why)
                if self.blocked.get("__journal__") != key:
                    self.log(f"journal: verified prefix stops at record {seq}: {why}")
                    self.blocked["__journal__"] = key
            return names, f"journal {self.journal.n} records"
        names = read_trailer(self.out)
        if names is not None:
            return names, f"index trailer {len(names)} tensors"
        return None, None

    def done_record(self, shard):
        p = self.dest / (shard + ".done")
        if not p.exists():
            return None
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}

    def one_pass(self):
        """Delete every shard that passes all guards. Returns the number deleted in this pass."""
        written, source = self.written()
        n = 0
        for s in self.shards:
            path = self.dest / s
            marker = fg.deleted_marker_of(path)
            on_disk = path.exists()
            if marker.exists() and not on_disk:
                continue
            rec = self.done_record(s)
            if rec is None:
                continue                        # not converted yet: never touched
            meta = self.table[s]
            verified = fg.is_verified(path, meta) if on_disk else fg.marker_of(path).exists()
            out_ok = same_path(rec.get("out", ""), self.out, self.bases) if rec else False
            ok, why = delete_verdict(s, on_disk, verified, True, out_ok, self.need[s], written)
            if not ok:
                if self.blocked.get(s) != why:
                    self.log(f"KEEP {s}: .done present but {why}")
                    self.blocked[s] = why
                continue
            self.blocked.pop(s, None)
            if not on_disk:
                if not marker.exists() and not self.dry:
                    self._write_marker(s, meta, written, source, recovered=True)
                    self.log(f"marked {s}: already gone, {why} ({source}) -> {marker.name} written")
                continue
            if self.dry:
                self.log(f"DRY {s}: would delete {fg.fmt_bytes(meta['size'])} B, {why} ({source})")
                continue
            self._write_marker(s, meta, written, source)
            path.unlink()
            n += 1
            self.deleted_now += 1
            free = shutil.disk_usage(self.dest).free
            self.log(f"DELETED {s}: {fg.fmt_bytes(meta['size'])} B, sha256 {meta['sha256']}, {why} "
                     f"({source}); free now {fg.fmt_bytes(free)} B")
        return n

    def _write_marker(self, shard, meta, written, source, recovered=False):
        rec = {"shard": shard, "size": meta["size"], "sha256": meta["sha256"], "out": str(self.out),
               "tensors": len(self.need[shard]), "checked_against": source,
               "journal_records": self.journal.n, "recovered": recovered,
               "deleted_at": datetime.now().isoformat(timespec="seconds")}
        p = fg.deleted_marker_of(self.dest / shard)
        tmp = Path(str(p) + ".tmp")
        tmp.write_text(json.dumps(rec) + "\n", encoding="utf-8")
        os.replace(tmp, p)

    def all_deleted(self):
        return all(fg.deleted_marker_of(self.dest / s).exists() and not (self.dest / s).exists()
                   for s in self.shards)


def status_line(dest, out, table, shards):
    """shards verified / on disk / done / deleted, bytes on disk, free space, container bytes."""
    dest, out = Path(dest), Path(out)
    nver = ndisk = ndone = ndel = 0
    on_disk = part = 0
    for s in shards:
        p = dest / s
        nver += fg.marker_of(p).exists()
        ndone += (dest / (s + ".done")).exists()
        gone = fg.deleted_marker_of(p).exists() and not p.exists()
        ndel += gone
        if p.exists():
            ndisk += 1
            on_disk += p.stat().st_size
        pp = Path(str(p) + ".part")
        if pp.exists():
            part += pp.stat().st_size
    free = shutil.disk_usage(dest).free
    cbytes = out.stat().st_size if out.exists() else 0
    jpath = journal_path(out)
    if jpath.exists():
        with open(jpath, "rb") as f:
            lines = sum(chunk.count(b"\n") for chunk in iter(lambda: f.read(READ_CHUNK), b""))
        jtxt = f"journal {max(0, lines - 1)} tensors"
    elif cbytes and read_trailer(out) is not None:
        jtxt = "index trailer written"
    else:
        jtxt = "no journal"
    n = len(shards)
    return (f"stage: verified {nver}/{n}, on disk {ndisk}, done {ndone}/{n}, deleted {ndel}/{n} | "
            f"shards on disk {fg.fmt_bytes(on_disk)} B + .part {fg.fmt_bytes(part)} B | "
            f"free {fg.fmt_bytes(free)} B ({fmt_gb(free)}) | container {fg.fmt_bytes(cbytes)} B ({fmt_gb(cbytes)}), {jtxt}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True, help="the container the converter writes (as passed to it)")
    ap.add_argument("--dest", default=str(fg.DEFAULT_DEST), help="the shard directory (= converter --consume)")
    ap.add_argument("--once", action="store_true", help="one pass, then exit")
    ap.add_argument("--dry-run", action="store_true", help="log what would be deleted, delete nothing")
    ap.add_argument("--status", action="store_true", help="print one status line and exit (no lock, no change)")
    ap.add_argument("--poll", type=float, default=POLL_SEC)
    ap.add_argument("--status-every", type=float, default=STATUS_SEC)
    args = ap.parse_args(argv)
    dest, out = Path(args.dest), Path(args.out)

    api = json.loads((dest / fg.API_FILE).read_text(encoding="utf-8"))
    table = fg.file_table(api)
    shards = fg.check_table(table)
    if args.status:
        print(status_line(dest, out, table, shards))
        return 0

    log = fg.Log(dest / STAGE_LOG)
    lock = fg.Lock(dest / STAGE_LOCK, log)
    lock.acquire()
    rc = 0
    try:
        stage = Stage(dest, out, log, dry_run=args.dry_run)
        log(f"stage start (pid {os.getpid()}): out {out}, dest {dest}, {len(shards)} shards, "
            f"{sum(len(v) for v in stage.need.values())} tensor names required, text layers {stage.n_layers}"
            f"{', DRY RUN' if args.dry_run else ''}")
        log(status_line(dest, out, table, shards))
        last_status = time.monotonic()
        while True:
            stage.one_pass()
            lock.beat()
            if args.once:
                break
            if stage.all_deleted():
                log("stage: all shards converted and deleted")
                break
            if time.monotonic() - last_status >= args.status_every:
                log(status_line(dest, out, table, shards))
                last_status = time.monotonic()
            time.sleep(args.poll)
    except KeyboardInterrupt:
        log("interrupted")
        rc = 130
    except Exception as err:
        log(f"ERROR {type(err).__name__}: {err}")
        raise
    finally:
        try:
            log(status_line(dest, out, table, shards))
            log(f"stage end: rc {rc}, {stage.deleted_now if 'stage' in locals() else 0} shards deleted in this run")
        finally:
            lock.release()
    return rc


if __name__ == "__main__":
    sys.exit(main())
