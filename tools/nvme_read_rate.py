#!/usr/bin/env python3
"""#146 (PREREG step 3): sustained NVMe read rate B for expert-sized blocks, unbuffered.

  python -I tools/nvme_read_rate.py --json runs/glm53-flash/step03/<stamp>.json <shard> [<shard> ...]

What it measures (runs/glm53-flash/PREREG.md, "Step 3"):
- random arm: blocks of --block B (default 14,155,776 = one GLM-5.3-Flash expert) at
  --align-aligned (default 4096) uniform random offsets over all given files, opened with
  FILE_FLAG_NO_BUFFERING, one handle per reader and file, one buffer per reader from
  VirtualAlloc; 1, 2, 4 readers; --reps repetitions, interleaved (1,2,4, then again);
- sequential arm: one reader, unbuffered, the files front to back, --seq-passes passes; reported
  beside B, no threshold.

Per reader count it prints every repetition's rate, the median (= B for that count), min, max,
the spread max/min and whether it is <= 1.15, m* = B / 190.3 GB/s and the prefill bound
39.9 x B tok/s. B for G1 is the median at the best reader count, only if that count's spread holds.

A run is VOID (PREREG: "a run with a second process reading or writing the disk") when
- --disk-busy says so, or
- the physical disk's own counters (IOCTL_DISK_PERFORMANCE) show reads beyond this tool's own
  bytes, or writes, above --foreign-max-mbps (default 4) in the idle check or in any cell, or
- a directory holding a measured file shows a download in progress (`*.part`, `*.partial`,
  `*.incomplete`, `*.tmp`, `fetch.lock`), or any entry of such a directory changed size or mtime
  during the run, or a file is open for writing elsewhere (the open shares read only, so a
  writer's handle refuses it).
Known void conditions at the start stop the tool before the first read unless --allow-void.
A VOID run still prints its numbers, labelled VOID, and exits 3; it never prints a valid verdict.

Exit codes: 0 valid run, 2 refused (usage, alignment, unbuffered flag did not take), 3 VOID.
Windows only (the operating point of the series); elsewhere it refuses with exit 2.
"""

import argparse
import bisect
import json
import os
import random
import statistics
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

# PREREG constants (runs/glm53-flash/PREREG.md, "Fixed for the whole series", "Step 3", "G1").
EXPERT_BLOCK = 14_155_776          # 3 x 4096 x 2048 weights, 36 B per 64 values
ALIGN = 4096
PREREG_READERS = (1, 2, 4)
PREREG_MIN_REPS = 3
SPREAD_MAX = 1.15
TOKEN_GBPS = 190.3                 # 40 tok/s x 4.7563 GB per token if every visit came from disk
PREFILL_FACTOR = 39.9              # 4096 tokens / 102.7 GB per chunk
G5_PREFILL_LINE = 150.0            # tok/s, cold prefill line of G5, checked in G1

EXIT_OK, EXIT_REFUSED, EXIT_VOID = 0, 2, 3


class Refused(Exception):
    """A condition under which no number may be produced at all."""


# ---------------------------------------------------------------- pure parts (tested anywhere)

def check_geometry(block, align, logical, physical):
    """The FILE_FLAG_NO_BUFFERING rules: size and offset multiples of the sector size."""
    if block <= 0 or align <= 0:
        raise Refused(f"block {block} and align {align} must be positive")
    for name, sector in (("logical", logical), ("physical", physical)):
        if sector <= 0:
            raise Refused(f"{name} sector size {sector} unknown; no fallback")
        if block % sector:
            raise Refused(f"block {block} is not a multiple of the {name} sector {sector}")
        if align % sector:
            raise Refused(f"align {align} is not a multiple of the {name} sector {sector}")
    if block % align:
        raise Refused(f"block {block} is not a multiple of align {align}")


class OffsetSampler:
    """Uniform over every aligned start in every file at which a whole block fits."""

    def __init__(self, sizes, block, align):
        self.block, self.align = block, align
        self.cum = []
        total = 0
        for i, size in enumerate(sizes):
            if size < block:
                raise Refused(f"file {i} has {size} B, smaller than one block of {block} B")
            total += (size - block) // align + 1
            self.cum.append(total)
        self.total = total

    def pick(self, rng):
        slot = rng.randrange(self.total)
        f = bisect.bisect_right(self.cum, slot)
        first = self.cum[f - 1] if f else 0
        return f, (slot - first) * self.align


def summarize(rates):
    """Median, min, max and spread of one reader count's repetitions (GB/s)."""
    lo, hi = min(rates), max(rates)
    spread = hi / lo if lo > 0 else float("inf")
    return {"rates_gbps": list(rates), "median_gbps": statistics.median(rates), "min_gbps": lo,
            "max_gbps": hi, "spread": spread, "spread_ok": spread <= SPREAD_MAX}


def derived(b_gbps):
    """PREREG G1 inputs from B: m* (share of visits that may come from NVMe) and the prefill bound."""
    prefill = PREFILL_FACTOR * b_gbps
    return {"m_star": b_gbps / TOKEN_GBPS, "prefill_bound_tok_s": prefill,
            "prefill_line_ok": prefill >= G5_PREFILL_LINE}


def foreign_io(before, after, own_bytes, seconds, max_mbps):
    """Disk-counter delta minus this tool's own reads; void above max_mbps (10^6 B/s) either way."""
    read = max(0, (after["read"] - before["read"]) - own_bytes)
    write = max(0, after["written"] - before["written"])
    secs = max(seconds, 1e-9)
    r_mbps, w_mbps = read / secs / 1e6, write / secs / 1e6
    reasons = []
    if r_mbps > max_mbps:
        reasons.append(f"foreign reads {r_mbps:.1f} MB/s > {max_mbps} MB/s")
    if w_mbps > max_mbps:
        reasons.append(f"disk writes {w_mbps:.1f} MB/s > {max_mbps} MB/s")
    return {"foreign_read_bytes": read, "write_bytes": write, "foreign_read_mbps": r_mbps,
            "write_mbps": w_mbps, "void_reasons": reasons}


def verdict(cells, readers, reps, void_reasons):
    """Per reader count summary plus B for G1; nothing is valid once a void reason exists."""
    out = {"valid": not void_reasons, "void_reasons": list(void_reasons), "per_readers": {}}
    if reps < PREREG_MIN_REPS:
        out["prereg_notes"] = [f"reps {reps} < {PREREG_MIN_REPS}: not a PREREG row"]
    missing = [r for r in PREREG_READERS if r not in readers]
    if missing:
        out.setdefault("prereg_notes", []).append(f"reader counts {missing} not measured: not a PREREG row")
    for r in readers:
        rates = [c["gbps"] for c in cells if c["readers"] == r]
        s = summarize(rates)
        s.update(derived(s["median_gbps"]))
        out["per_readers"][str(r)] = s
    best = max(readers, key=lambda r: out["per_readers"][str(r)]["median_gbps"])
    bs = out["per_readers"][str(best)]
    out["best_readers"] = best
    if void_reasons:
        out["b_for_g1"] = None
        out["g1_input"] = "VOID: " + "; ".join(void_reasons)
    elif "prereg_notes" in out:
        out["b_for_g1"] = None
        out["g1_input"] = "not answered: " + "; ".join(out["prereg_notes"])
    elif not bs["spread_ok"]:
        out["b_for_g1"] = None
        out["g1_input"] = (f"not answered: spread {bs['spread']:.3f} > {SPREAD_MAX} at the best reader "
                           f"count {best} (cache or driver noise)")
    else:
        out["b_for_g1"] = bs["median_gbps"]
        out["g1_input"] = f"B = {bs['median_gbps']:.3f} GB/s at {best} reader(s), binding for step 14"
    return out


DOWNLOAD_SIGNS = (".part", ".partial", ".incomplete", ".tmp")
DOWNLOAD_LOCKS = ("fetch.lock",)


def download_signs(names):
    """Entries that say a download is (or was left) in progress beside the measured files."""
    return sorted(n for n in names if n.lower().endswith(DOWNLOAD_SIGNS) or n.lower() in DOWNLOAD_LOCKS)


def dir_state(dirs):
    """(dir, name, size, mtime) of every entry in the directories that hold measured files."""
    out = set()
    for d in dirs:
        for e in os.scandir(d):
            try:
                st = e.stat()
            except OSError:
                continue
            out.add((d, e.name, st.st_size, st.st_mtime_ns))
    return out


def parse_readers(text):
    try:
        vals = [int(x) for x in text.split(",") if x.strip()]
    except ValueError:
        raise Refused(f"--readers {text!r}: comma-separated positive integers expected")
    if not vals or any(v <= 0 for v in vals) or len(set(vals)) != len(vals):
        raise Refused(f"--readers {text!r}: distinct positive integers expected")
    return vals


# ---------------------------------------------------------------- Windows IO

if sys.platform == "win32":
    import ctypes
    import struct
    from ctypes import wintypes as w

    _k = ctypes.WinDLL("kernel32", use_last_error=True)   # WinDLL releases the GIL per call
    _k.CreateFileW.restype = w.HANDLE
    _k.CreateFileW.argtypes = [w.LPCWSTR, w.DWORD, w.DWORD, ctypes.c_void_p, w.DWORD, w.DWORD, w.HANDLE]
    _k.CloseHandle.argtypes = [w.HANDLE]
    _k.ReadFile.argtypes = [w.HANDLE, ctypes.c_void_p, w.DWORD, ctypes.POINTER(w.DWORD), ctypes.c_void_p]
    _k.DeviceIoControl.argtypes = [w.HANDLE, w.DWORD, ctypes.c_void_p, w.DWORD, ctypes.c_void_p, w.DWORD,
                                   ctypes.POINTER(w.DWORD), ctypes.c_void_p]
    _k.VirtualAlloc.restype = ctypes.c_void_p
    _k.VirtualAlloc.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD, w.DWORD]
    _k.VirtualFree.argtypes = [ctypes.c_void_p, ctypes.c_size_t, w.DWORD]
    _k.GetVolumePathNameW.argtypes = [w.LPCWSTR, w.LPWSTR, w.DWORD]
    _k.GetDiskFreeSpaceExW.argtypes = [w.LPCWSTR, ctypes.POINTER(ctypes.c_ulonglong),
                                       ctypes.POINTER(ctypes.c_ulonglong), ctypes.POINTER(ctypes.c_ulonglong)]

    GENERIC_READ = 0x8000_0000
    FILE_SHARE_READ, FILE_SHARE_WRITE = 1, 2
    OPEN_EXISTING = 3
    FILE_FLAG_NO_BUFFERING = 0x2000_0000
    INVALID_HANDLE = ctypes.c_void_p(-1).value
    ERROR_HANDLE_EOF, ERROR_INVALID_PARAMETER, ERROR_SHARING_VIOLATION = 38, 87, 32
    MEM_COMMIT_RESERVE, MEM_RELEASE, PAGE_READWRITE = 0x3000, 0x8000, 0x04
    IOCTL_DISK_PERFORMANCE = 0x0007_0020                 # CTL_CODE(DISK, 0x08, BUFFERED, ANY)
    IOCTL_STORAGE_QUERY_PROPERTY = 0x002D_1400           # CTL_CODE(MASS_STORAGE, 0x500, ...)
    IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS = 0x0056_0000   # CTL_CODE('V', 0, ...)
    StorageDeviceProperty, StorageAccessAlignmentProperty = 0, 6

    class OVERLAPPED(ctypes.Structure):
        _fields_ = [("Internal", ctypes.c_size_t), ("InternalHigh", ctypes.c_size_t),
                    ("Offset", w.DWORD), ("OffsetHigh", w.DWORD), ("hEvent", w.HANDLE)]

    def _err(what):
        e = ctypes.get_last_error()
        return OSError(e, f"{what}: Win32 error {e} ({ctypes.FormatError(e).strip()})")

    def _open(path, flags, share, access=GENERIC_READ):
        h = _k.CreateFileW(str(path), access, share, None, OPEN_EXISTING, flags, None)
        if h is None or h == INVALID_HANDLE:
            raise _err(f"CreateFileW {path}")
        return h

    def _ioctl(path, code, inbuf=b"", outsize=1024):
        h = _open(path, 0, FILE_SHARE_READ | FILE_SHARE_WRITE, access=0)
        try:
            out = ctypes.create_string_buffer(outsize)
            n = w.DWORD()
            ib = ctypes.create_string_buffer(inbuf, len(inbuf)) if inbuf else None
            if not _k.DeviceIoControl(h, code, ib, len(inbuf), out, outsize, ctypes.byref(n), None):
                raise _err(f"DeviceIoControl {hex(code)} on {path}")
            return out.raw[:n.value]
        finally:
            _k.CloseHandle(h)

    def volume_of(path):
        buf = ctypes.create_unicode_buffer(260)
        if not _k.GetVolumePathNameW(str(Path(path).resolve()), buf, 260):
            raise _err(f"GetVolumePathNameW {path}")
        return buf.value                                     # e.g. "C:\\"

    def _volume_device(vol):
        return "\\\\.\\" + vol.rstrip("\\")

    def physical_disk_of(vol):
        raw = _ioctl(_volume_device(vol), IOCTL_VOLUME_GET_VOLUME_DISK_EXTENTS)
        (count,) = struct.unpack_from("<I", raw, 0)
        if count != 1:
            raise Refused(f"volume {vol} spans {count} disk extents; one physical disk expected")
        (disk,) = struct.unpack_from("<I", raw, 8)
        return disk

    def sector_sizes(vol):
        raw = _ioctl(_volume_device(vol), IOCTL_STORAGE_QUERY_PROPERTY,
                     struct.pack("<iiB3x", StorageAccessAlignmentProperty, 0, 0))
        f = struct.unpack_from("<7I", raw)
        return f[4], f[5]                                    # BytesPerLogicalSector, BytesPerPhysicalSector

    def drive_model(disk):
        raw = _ioctl(f"\\\\.\\PhysicalDrive{disk}", IOCTL_STORAGE_QUERY_PROPERTY,
                     struct.pack("<iiB3x", StorageDeviceProperty, 0, 0))
        vo, po, ro = struct.unpack_from("<3I", raw, 12)      # vendor, product, revision (no serial)

        def s(o):
            return raw[o:raw.index(b"\0", o)].decode("ascii", "replace").strip() if o else ""
        return " ".join(x for x in (s(vo), s(po)) if x), s(ro)

    def disk_counters(disk):
        raw = _ioctl(f"\\\\.\\PhysicalDrive{disk}", IOCTL_DISK_PERFORMANCE)
        read, written = struct.unpack_from("<2q", raw, 0)
        return {"read": read, "written": written}

    def free_bytes(vol):
        free = ctypes.c_ulonglong()
        _k.GetDiskFreeSpaceExW(vol, ctypes.byref(free), None, None)
        return free.value

    class Buffer:
        def __init__(self, size):
            self.size = size
            self.ptr = _k.VirtualAlloc(None, size, MEM_COMMIT_RESERVE, PAGE_READWRITE)
            if not self.ptr:
                raise _err("VirtualAlloc")

        def close(self):
            _k.VirtualFree(self.ptr, 0, MEM_RELEASE)

    def open_unbuffered(path):
        """Share read only: a handle someone holds for writing refuses the open (file still written)."""
        try:
            return _open(path, FILE_FLAG_NO_BUFFERING, FILE_SHARE_READ)
        except OSError as e:
            if e.errno == ERROR_SHARING_VIOLATION:
                raise Refused(f"{path} is open for writing by another process (sharing violation)")
            raise

    def read_at(handle, buf, offset, length):
        """Positional read on a synchronous handle; returns bytes read (short only at EOF)."""
        ov = OVERLAPPED()
        ov.Offset, ov.OffsetHigh = offset & 0xFFFF_FFFF, offset >> 32
        n = w.DWORD()
        if not _k.ReadFile(handle, buf.ptr, length, ctypes.byref(n), ctypes.byref(ov)):
            if ctypes.get_last_error() == ERROR_HANDLE_EOF:
                return 0
            raise _err(f"ReadFile at {offset} len {length}")
        return n.value

    def close_handle(h):
        _k.CloseHandle(h)

    def unbuffered_took(path):
        """With FILE_FLAG_NO_BUFFERING a read at offset 1 must fail with ERROR_INVALID_PARAMETER."""
        h = open_unbuffered(path)
        buf = Buffer(ALIGN)
        try:
            read_at(h, buf, 1, ALIGN)
            return False
        except OSError as e:
            return e.errno == ERROR_INVALID_PARAMETER
        finally:
            buf.close()
            close_handle(h)


# ---------------------------------------------------------------- the arms

def file_state(paths):
    return [(os.stat(p).st_size, os.stat(p).st_mtime_ns) for p in paths]


def random_cell(paths, sampler, block, readers, secs, warmup, seed):
    """`readers` threads, each with its own handles and buffer, random blocks until the deadline."""
    results = [None] * readers
    errors = [None] * readers
    clock = {}

    def set_clock():                          # runs once, when every reader holds its handles
        now = time.perf_counter()
        clock["meas"], clock["end"] = now + warmup, now + warmup + secs

    start_gate = threading.Barrier(readers + 1, action=set_clock)

    def run(i):
        handles, buf = [], None
        try:
            for p in paths:
                handles.append(open_unbuffered(p))
            buf = Buffer(block)
        except BaseException as e:            # noqa: BLE001 - handed to the main thread
            errors[i] = e
            start_gate.abort()
            for h in handles:
                close_handle(h)
            return
        rng = random.Random(seed * 1_000_003 + i)
        counted = issued = 0
        lat = []
        short = 0
        try:
            start_gate.wait()
            t_meas, t_end = clock["meas"], clock["end"]
            while True:
                t0 = time.perf_counter()
                if t0 >= t_end:
                    break
                f, off = sampler.pick(rng)
                n = read_at(handles[f], buf, off, block)
                t1 = time.perf_counter()
                issued += n
                if n != block:
                    short += 1
                elif t_meas <= t1 < t_end:
                    counted += n
                    lat.append(t1 - t0)
        except BaseException as e:            # noqa: BLE001 - handed to the main thread
            errors[i] = e
        finally:
            buf.close()
            for h in handles:
                close_handle(h)
        results[i] = {"counted": counted, "issued": issued, "lat": lat, "short": short}

    threads = [threading.Thread(target=run, args=(i,)) for i in range(readers)]
    for t in threads:
        t.start()
    try:
        start_gate.wait()
    except threading.BrokenBarrierError:
        pass
    for t in threads:
        t.join()
    for e in errors:
        if isinstance(e, threading.BrokenBarrierError):
            continue
        if e is not None:
            raise e
    if any(e is not None for e in errors):
        raise Refused("a reader thread failed before the start")
    lat = sorted(x for r in results for x in r["lat"])
    counted = sum(r["counted"] for r in results)
    return {"readers": readers, "bytes": counted, "gbps": counted / secs / 1e9,
            "blocks": len(lat), "short_reads": sum(r["short"] for r in results),
            "own_bytes": sum(r["issued"] for r in results),
            "lat_p50_ms": 1e3 * lat[len(lat) // 2] if lat else None,
            "lat_p99_ms": 1e3 * lat[min(len(lat) - 1, int(0.99 * len(lat)))] if lat else None}


def sequential_pass(paths, chunk):
    """One reader, unbuffered, every file front to back in `chunk`-sized aligned reads."""
    buf = Buffer(chunk)
    total = 0
    t0 = time.perf_counter()
    try:
        for p in paths:
            h = open_unbuffered(p)
            try:
                off = 0
                while True:
                    n = read_at(h, buf, off, chunk)
                    total += n
                    off += n
                    if n < chunk:
                        break
            finally:
                close_handle(h)
    finally:
        buf.close()
    secs = time.perf_counter() - t0
    return {"bytes": total, "secs": secs, "gbps": total / secs / 1e9}


# ---------------------------------------------------------------- driver

def git_commit():
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def measure(args, log=print):
    if sys.platform != "win32":
        raise Refused("Windows only (FILE_FLAG_NO_BUFFERING path of the operating point)")
    readers = parse_readers(args.readers)
    paths = [str(Path(p).resolve()) for p in args.files]
    if not paths:
        raise Refused("no files given")
    for p in paths:
        if not Path(p).is_file():
            raise Refused(f"{p} is not a file")
    vols = {volume_of(p) for p in paths}
    disks = {physical_disk_of(v) for v in vols}
    if len(disks) != 1:
        raise Refused(f"files lie on physical disks {sorted(disks)}; one disk per run")
    disk = disks.pop()
    vol = sorted(vols)[0]
    logical, physical = sector_sizes(vol)
    check_geometry(args.block, args.align, logical, physical)
    if args.seq_chunk % physical or args.seq_chunk % logical:
        raise Refused(f"--seq-chunk {args.seq_chunk} is not a multiple of the sector size")
    sizes = [os.stat(p).st_size for p in paths]
    sampler = OffsetSampler(sizes, args.block, args.align)
    for p in paths:
        if not unbuffered_took(p):
            raise Refused(f"{p}: a read at offset 1 succeeded; FILE_FLAG_NO_BUFFERING did not take")
    model, firmware = drive_model(disk)
    env = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "commit": git_commit(),
           "python": sys.version.split()[0], "drive": model, "firmware": firmware, "physical_disk": disk,
           "volume": vol, "free_bytes": free_bytes(vol), "sector_logical": logical,
           "sector_physical": physical, "block": args.block, "align": args.align, "readers": readers,
           "reps": args.reps, "secs": args.secs, "warmup": args.warmup, "seed": args.seed,
           "seq_passes": args.seq_passes, "seq_chunk": args.seq_chunk,
           "foreign_max_mbps": args.foreign_max_mbps, "disk_busy_flag": args.disk_busy,
           "files": [{"path": p, "size": s} for p, s in zip(paths, sizes)]}
    log(f"nvme_read_rate  {env['date_utc']}  commit {env['commit']}  {model} fw {firmware}  "
        f"PhysicalDrive{disk} {vol}  free {env['free_bytes']:,} B  sectors {logical}/{physical}")
    log(f"block {args.block:,} B  align {args.align}  readers {readers}  reps {args.reps}  "
        f"{args.secs} s/cell (+{args.warmup} s warm-up)  {len(paths)} file(s), {sum(sizes):,} B")

    void = []
    if args.disk_busy:
        void.append("--disk-busy: another process uses the disk")
    dirs = sorted({str(Path(p).parent) for p in paths})
    for d in dirs:
        signs = download_signs(os.listdir(d))
        if signs:
            void.append(f"download in progress in {d}: {', '.join(signs)}")
    dirs0 = dir_state(dirs)
    c0 = disk_counters(disk)
    time.sleep(args.idle_check)
    idle = foreign_io(c0, disk_counters(disk), 0, args.idle_check, args.foreign_max_mbps)
    log(f"idle check {args.idle_check} s: reads {idle['foreign_read_mbps']:.2f} MB/s, "
        f"writes {idle['write_mbps']:.2f} MB/s (limit {args.foreign_max_mbps} MB/s)")
    void += [f"idle check: {r}" for r in idle["void_reasons"]]
    if void and not args.allow_void:
        raise VoidStart(void)

    state0 = file_state(paths)
    cells = []
    for rep in range(args.reps):
        for r in readers:
            before = disk_counters(disk)
            t0 = time.perf_counter()
            cell = random_cell(paths, sampler, args.block, r, args.secs, args.warmup,
                               seed=args.seed * 10_007 + rep * 101 + r)
            wall = time.perf_counter() - t0
            fio = foreign_io(before, disk_counters(disk), cell["own_bytes"], wall, args.foreign_max_mbps)
            cell.update(rep=rep, foreign=fio)
            cells.append(cell)
            void += [f"rep {rep} readers {r}: {x}" for x in fio["void_reasons"]]
            mark = "  VOID " + "; ".join(fio["void_reasons"]) if fio["void_reasons"] else ""
            log(f"rep {rep}  readers {r:>2}  {cell['gbps']:7.3f} GB/s  blocks {cell['blocks']:>6}  "
                f"p50 {cell['lat_p50_ms'] or 0:6.2f} ms  p99 {cell['lat_p99_ms'] or 0:6.2f} ms  "
                f"short {cell['short_reads']}  foreign r/w {fio['foreign_read_mbps']:.1f}/"
                f"{fio['write_mbps']:.1f} MB/s{mark}")
    seq = []
    for p in range(args.seq_passes):
        before = disk_counters(disk)
        s = sequential_pass(paths, args.seq_chunk)
        fio = foreign_io(before, disk_counters(disk), s["bytes"], s["secs"], args.foreign_max_mbps)
        s.update(rep=p, foreign=fio)
        seq.append(s)
        void += [f"sequential pass {p}: {x}" for x in fio["void_reasons"]]
        log(f"sequential pass {p}  {s['gbps']:7.3f} GB/s  {s['bytes']:,} B in {s['secs']:.2f} s  "
            f"foreign r/w {fio['foreign_read_mbps']:.1f}/{fio['write_mbps']:.1f} MB/s")
    if file_state(paths) != state0:
        void.append("a measured file changed size or mtime during the run")
    changed = sorted({f"{d}/{n}" for d, n, _, _ in dir_state(dirs) ^ dirs0})
    if changed:
        void.append(f"directory entries changed during the run: {', '.join(changed[:5])}"
                    + (f" (+{len(changed) - 5})" if len(changed) > 5 else ""))

    v = verdict(cells, readers, args.reps, void)
    if seq:
        v["sequential"] = {"rates_gbps": [s["gbps"] for s in seq],
                           "median_gbps": statistics.median(s["gbps"] for s in seq)}
    return {"env": env, "idle": idle, "cells": cells, "sequential": seq, "verdict": v}


class VoidStart(Exception):
    def __init__(self, reasons):
        super().__init__("; ".join(reasons))
        self.reasons = reasons


def print_verdict(v, log=print):
    tag = "run VALID (no foreign disk IO seen)" if v["valid"] else "run VOID"
    log(f"\n== PREREG step 3 verdict: {tag} ==")
    for reason in v["void_reasons"]:
        log(f"  void: {reason}")
    for note in v.get("prereg_notes", []):
        log(f"  note: {note}")
    for r, s in v["per_readers"].items():
        reps = ", ".join(f"{x:.3f}" for x in s["rates_gbps"])
        log(f"readers {r:>2}: B = {s['median_gbps']:.3f} GB/s (median; reps {reps})  "
            f"min {s['min_gbps']:.3f}  max {s['max_gbps']:.3f}  spread {s['spread']:.3f} "
            f"<= {SPREAD_MAX}? {'yes' if s['spread_ok'] else 'NO'}  "
            f"m* = B/{TOKEN_GBPS} = {100 * s['m_star']:.2f} %  "
            f"prefill bound {PREFILL_FACTOR} x B = {s['prefill_bound_tok_s']:.0f} tok/s "
            f"(>= {G5_PREFILL_LINE:.0f}? {'yes' if s['prefill_line_ok'] else 'no'})"
            + ("" if v["valid"] else "  [VOID]"))
    if "sequential" in v:
        seq = ", ".join(f"{x:.3f}" for x in v["sequential"]["rates_gbps"])
        log(f"sequential: median {v['sequential']['median_gbps']:.3f} GB/s (passes {seq}; no threshold)")
    log(f"G1 input: {v['g1_input']}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="+", help="shard files to read (one physical disk)")
    ap.add_argument("--block", type=int, default=EXPERT_BLOCK, help="bytes per read (default one expert)")
    ap.add_argument("--align", type=int, default=ALIGN, help="offset alignment (default 4096)")
    ap.add_argument("--readers", default=",".join(map(str, PREREG_READERS)), help="reader counts, e.g. 1,2,4")
    ap.add_argument("--reps", type=int, default=PREREG_MIN_REPS, help="repetitions per reader count")
    ap.add_argument("--secs", type=float, default=20.0, help="measured seconds per cell")
    ap.add_argument("--warmup", type=float, default=2.0, help="unmeasured seconds before each cell")
    ap.add_argument("--seq-passes", type=int, default=3, help="sequential passes over all files")
    ap.add_argument("--seq-chunk", type=int, default=16 << 20, help="sequential read size")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--idle-check", type=float, default=3.0, help="seconds of disk counters before the run")
    ap.add_argument("--foreign-max-mbps", type=float, default=4.0,
                    help="foreign disk reads or writes above this (10^6 B/s) void the run")
    ap.add_argument("--disk-busy", action="store_true", help="another process uses the disk: the run is VOID")
    ap.add_argument("--allow-void", action="store_true", help="run anyway when void at the start (functional tests)")
    ap.add_argument("--json", help="write every cell, the env and the verdict here")
    args = ap.parse_args(argv)
    if args.reps < 1 or args.secs <= 0 or args.warmup < 0 or args.seq_passes < 0:
        print("refused: --reps >= 1, --secs > 0, --warmup >= 0, --seq-passes >= 0", file=sys.stderr)
        return EXIT_REFUSED
    try:
        result = measure(args)
    except VoidStart as e:
        print("VOID before the first read (pass --allow-void to run anyway, labelled VOID):", file=sys.stderr)
        for r in e.reasons:
            print(f"  {r}", file=sys.stderr)
        return EXIT_VOID
    except Refused as e:
        print(f"refused: {e}", file=sys.stderr)
        return EXIT_REFUSED
    print_verdict(result["verdict"])
    if args.json:
        Path(args.json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json).write_text(json.dumps(result, indent=1), encoding="utf-8")
        print(f"json: {args.json}")
    return EXIT_OK if result["verdict"]["valid"] else EXIT_VOID


if __name__ == "__main__":
    sys.exit(main())
