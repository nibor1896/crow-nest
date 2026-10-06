#!/usr/bin/env bash
# Fetches NVIDIA's NVRTC runtime for the crow-nest engine straight from NVIDIA's own
# PyPI wheel and places it beside `serve`. crow-nest does not distribute NVRTC.
# Linux twin of tools/fetch-nvrtc.ps1.
#
# serve loads NVRTC at run time to compile its CUDA kernels. The engine package holds
# no NVIDIA file (tools/pack-engine.sh stages none), so this script gets them from the
# source NVIDIA publishes: the `nvidia-cuda-nvrtc` 13.3.33 wheel on PyPI, the exact
# NVRTC build the engine's PTX manifest was recorded with. It is NVIDIA's software
# under NVIDIA's licence; fetching it is your own act, not a redistribution by crow-nest.
#
#  1. downloads the pinned wheel from files.pythonhosted.org (or takes --wheel FILE);
#  2. refuses unless the wheel's size and sha256 are the pinned ones;
#  3. reads only the two members it needs (never extracts a path from the archive);
#  4. refuses unless each member's sha256 and size equal the entry in the wheel's own
#     *.dist-info/RECORD (urlsafe base64, no padding);
#  5. only when every check passed, writes libnvrtc.so (the bytes of the wheel's
#     libnvrtc.so.13) and libnvrtc-builtins.so.13.3 into --target. On any mismatch
#     nothing is written.
#
# libnvrtc.so.13 is written under the unversioned name `libnvrtc.so` because cudarc
# tries that name FIRST over all search paths: a system CUDA providing libnvrtc.so
# would beat a libnvrtc.so.13 beside serve (#133, Crow #341). libnvrtc dlopens the
# builtins by the exact name libnvrtc-builtins.so.13.3, so that one keeps its name.
# Real files, no symlinks, no duplicate. Put the folder on LD_LIBRARY_PATH (libnvrtc
# loads the builtins from there; it has no RUNPATH).
#
# Usage:  tools/fetch-nvrtc.sh --target DIR [--wheel FILE]
#         tools/fetch-nvrtc.sh --selftest      offline, on a synthetic wheel
# Needs python3 (or python) and, for the download, network access to files.pythonhosted.org.
set -euo pipefail

# nvidia-cuda-nvrtc 13.3.33, manylinux_2_12_x86_64, as published by NVIDIA on PyPI
WHEEL_URL='https://files.pythonhosted.org/packages/8b/2c/86916c8a34dcdb0c3ddd1c0e30545041bd781184e437b9cb76fcda70560b/nvidia_cuda_nvrtc-13.3.33-py3-none-manylinux2010_x86_64.manylinux_2_12_x86_64.whl'
WHEEL_SHA256='82530788b8c6164a54d3fd9ae8bcca8893d397c4aeb998861982a03bbe41e204'
WHEEL_SIZE=51110910
# wheel member : name beside serve
MEMBERS='nvidia/cu13/lib/libnvrtc.so.13:libnvrtc.so nvidia/cu13/lib/libnvrtc-builtins.so.13.3:libnvrtc-builtins.so.13.3'

PY=""; for c in python3 python; do if command -v "$c" >/dev/null 2>&1 && "$c" -c 'import sys' >/dev/null 2>&1; then PY="$c"; break; fi; done
[ -n "$PY" ] || { echo "python3 not found" >&2; exit 1; }

target=""; wheel=""; mode=fetch
while [ $# -gt 0 ]; do
  case "$1" in
    --selftest) mode=selftest; shift ;;
    --target) target="$2"; shift 2 ;;
    --wheel) wheel="$2"; shift 2 ;;
    -h|--help) sed -n 2,30p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done
if [ "$mode" = fetch ] && [ -z "$target" ]; then
  echo "pass --target <the folder holding serve>" >&2; exit 2
fi

exec "$PY" - "$mode" "$target" "$wheel" "$WHEEL_URL" "$WHEEL_SHA256" "$WHEEL_SIZE" "$MEMBERS" <<'PY'
import base64, hashlib, io, os, re, shutil, stat, sys, tempfile, urllib.request, zipfile

mode, target, wheel_arg, URL, SHA, SIZE, MEMBERS = sys.argv[1:8]
SIZE = int(SIZE)
MEMBERS = dict(m.split(":", 1) for m in MEMBERS.split())  # wheel member -> name beside serve


class Refused(Exception):
    pass


def record_hash(data):
    """The RECORD form of a digest: urlsafe base64, no padding."""
    return base64.urlsafe_b64encode(hashlib.sha256(data).digest()).decode().rstrip("=")


def read_record(text):
    rec = {}
    for line in text.splitlines():
        m = re.match(r"^(.*),([^,]*),([^,]*)$", line)
        if m:
            h = m.group(2)
            rec[m.group(1).strip('"')] = (h[7:] if h.startswith("sha256=") else "", m.group(3))
    return rec


def install(wheel, sha, size, members, target_dir):
    """Verify the wheel and its members, then write them. Raises Refused (and writes
    nothing) on any mismatch. Returns the written file names."""
    if not os.path.isfile(wheel):
        raise Refused("no wheel at %s" % wheel)
    n = os.path.getsize(wheel)
    if n != size:
        raise Refused("the wheel is %d bytes, expected %d" % (n, size))
    got = hashlib.sha256(open(wheel, "rb").read()).hexdigest()
    if got != sha.lower():
        raise Refused("the wheel's sha256 is %s, expected %s" % (got, sha))
    verified = {}
    with zipfile.ZipFile(wheel) as z:
        recs = [i for i in z.namelist() if re.match(r"^[^/]+\.dist-info/RECORD$", i)]
        if len(recs) != 1:
            raise Refused("the wheel has %d dist-info/RECORD files, expected exactly 1" % len(recs))
        record = read_record(z.read(recs[0]).decode("utf-8"))
        for member, name in members.items():
            infos = [i for i in z.infolist() if i.filename == member]
            if len(infos) != 1:
                raise Refused("the wheel has no member %s" % member)
            if stat.S_ISLNK(infos[0].external_attr >> 16):
                raise Refused("%s is a symlink in the wheel, expected a real file" % member)
            if member not in record:
                raise Refused("RECORD has no entry for %s" % member)
            data = z.read(infos[0])
            want_hash, want_size = record[member]
            if record_hash(data) != want_hash:
                raise Refused("%s sha256=%s, RECORD says %s" % (member, record_hash(data), want_hash))
            if str(len(data)) != want_size:
                raise Refused("%s is %d bytes, RECORD says %s" % (member, len(data), want_size))
            verified[name] = data
    # everything verified: only now touch the target
    os.makedirs(target_dir, exist_ok=True)
    for name, data in verified.items():
        dest = os.path.join(target_dir, name)
        part = dest + ".part"
        with open(part, "wb") as fh:
            fh.write(data)
        os.chmod(part, 0o755)
        if os.path.islink(dest):
            os.remove(dest)
        os.replace(part, dest)
    return list(verified)


# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

ok = red = 0


def check(name, passed):
    global ok, red
    if passed:
        print("  ok   " + name); ok += 1
    else:
        print("  FAIL " + name); red += 1


def synth_wheel(path, members, record=None, record_name="synth-1.0.dist-info/RECORD", symlink=None):
    if record is None:
        record = "".join("%s,sha256=%s,%d\n" % (k, record_hash(v), len(v)) for k, v in members.items())
        record += "synth-1.0.dist-info/RECORD,,\n"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        for k, v in members.items():
            zi = zipfile.ZipInfo(k)
            if k == symlink:
                zi.external_attr = (stat.S_IFLNK | 0o777) << 16
            z.writestr(zi, v)
        if record_name:
            z.writestr(record_name, record)


def wheel_pin(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest(), os.path.getsize(path)


def selftest():
    print("fetch-nvrtc selftest (synthetic wheel, offline)")
    tmp = tempfile.mkdtemp(prefix="crow-nest-fetch-selftest-")
    try:
        a, b = b"synthetic libnvrtc payload", b"synthetic builtins payload"
        members = {"p/lib/libnvrtc.so.13": a, "p/lib/libnvrtc-builtins.so.13.3": b, "p/lib/unrelated.txt": b"x"}
        want = {"p/lib/libnvrtc.so.13": "libnvrtc.so", "p/lib/libnvrtc-builtins.so.13.3": "libnvrtc-builtins.so.13.3"}
        wheel = os.path.join(tmp, "synth.whl")
        synth_wheel(wheel, members)
        wsha, wsize = wheel_pin(wheel)

        # the pins themselves
        check("the pinned URL is on files.pythonhosted.org over https", re.match(r"^https://files\.pythonhosted\.org/packages/", URL) is not None)
        check("the pinned sha256 is 64 lower-case hex", re.fullmatch(r"[0-9a-f]{64}", SHA) is not None)
        check("the pin writes libnvrtc.so (cudarc's first name) and the builtins under the name libnvrtc opens",
              sorted(MEMBERS.values()) == ["libnvrtc-builtins.so.13.3", "libnvrtc.so"])
        check("the RECORD hash of 'abc' is urlsafe base64 without padding", record_hash(b"abc") == "ungWv48Bz-pBQUDeXa4iI7ADYaOWF3qctBD_YfIAFa0")

        # happy path: both members land, byte for byte, nothing else
        t1 = os.path.join(tmp, "t1")
        names = install(wheel, wsha, wsize, want, t1)
        check("a valid wheel installs both members", sorted(names) == ["libnvrtc-builtins.so.13.3", "libnvrtc.so"])
        check("libnvrtc.so holds the bytes of libnvrtc.so.13", open(os.path.join(t1, "libnvrtc.so"), "rb").read() == a)
        check("the builtins keep their name and bytes", open(os.path.join(t1, "libnvrtc-builtins.so.13.3"), "rb").read() == b)
        check("only the two files are placed (no .part, no libnvrtc.so.13 duplicate, no unrelated member)",
              sorted(os.listdir(t1)) == ["libnvrtc-builtins.so.13.3", "libnvrtc.so"])
        check("the placed files are real files, not symlinks", not any(os.path.islink(os.path.join(t1, n)) for n in os.listdir(t1)))

        # refusals; each must leave the target untouched
        t2 = os.path.join(tmp, "t2")

        def refuses(label, **kw):
            args = dict(wheel=wheel, sha=wsha, size=wsize, members=want, target_dir=t2)
            args.update(kw)
            try:
                install(**args)
                threw = False
            except Refused:
                threw = True
            check("%s (refused, nothing written)" % label, threw and (not os.path.exists(t2) or not os.listdir(t2)))

        refuses("a wheel with the wrong sha256", sha="0" * 64)
        refuses("a wheel with the wrong size", size=wsize + 1)
        refuses("a missing wheel file", wheel=os.path.join(tmp, "nope.whl"))

        def variant(name, **kw):
            p = os.path.join(tmp, name + ".whl")
            m = kw.pop("members", members)
            synth_wheel(p, m, **kw)
            s, z = wheel_pin(p)
            return dict(wheel=p, sha=s, size=z)

        lie = "p/lib/libnvrtc.so.13,sha256=%s,%d\np/lib/libnvrtc-builtins.so.13.3,sha256=%s,%d\n" % (record_hash(a), len(a), record_hash(a), len(b))
        refuses("a member whose sha256 differs from RECORD", **variant("lie", record=lie))
        bs = "p/lib/libnvrtc.so.13,sha256=%s,%d\np/lib/libnvrtc-builtins.so.13.3,sha256=%s,%d\n" % (record_hash(a), len(a), record_hash(b), len(b) + 1)
        refuses("a member whose size differs from RECORD", **variant("size", record=bs))
        refuses("a member RECORD does not list", **variant("norec", record="p/lib/libnvrtc.so.13,sha256=%s,%d\n" % (record_hash(a), len(a))))
        refuses("a wheel without the second member", **variant("nomember", members={"p/lib/libnvrtc.so.13": a}))
        refuses("a wheel without a RECORD", **variant("nodist", record_name=""))
        refuses("a RECORD entry with another digest algorithm", **variant("md5", record="p/lib/libnvrtc.so.13,md5=AAAA,%d\np/lib/libnvrtc-builtins.so.13.3,md5=AAAA,%d\n" % (len(a), len(b))))
        swapped = dict(members); swapped["p/lib/libnvrtc.so.13"] = b"swapped after RECORD was made"
        orig = "p/lib/libnvrtc.so.13,sha256=%s,%d\np/lib/libnvrtc-builtins.so.13.3,sha256=%s,%d\n" % (record_hash(a), len(a), record_hash(b), len(b))
        refuses("a swapped member (right wheel pin, RECORD from the original)", **variant("swap", members=swapped, record=orig))
        refuses("a member that is a symlink in the wheel", **variant("link", symlink="p/lib/libnvrtc.so.13"))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    print("selftest: %d ok, %d failed" % (ok, red))
    return 1 if red else 0


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------

def fetch():
    tmp = None
    try:
        if wheel_arg:
            wheel = wheel_arg
        else:
            tmp = tempfile.mkdtemp(prefix="crow-nest-nvrtc-")
            wheel = os.path.join(tmp, "nvidia_cuda_nvrtc.whl")
            print("downloading " + URL)
            with urllib.request.urlopen(URL, timeout=120) as r, open(wheel, "wb") as fh:
                shutil.copyfileobj(r, fh)
        names = install(wheel, SHA, SIZE, MEMBERS, target)
        d = os.path.abspath(target)
        for n in names:
            data = open(os.path.join(d, n), "rb").read()
            print("  %s  %d bytes  sha256 %s  (matches the wheel's RECORD)" % (n, len(data), hashlib.sha256(data).hexdigest()))
        print("RESULT: NVRTC 13.3.33 from NVIDIA's wheel placed in %s" % d)
        return 0
    except Refused as e:
        print("REFUSED: %s" % e, file=sys.stderr)
        return 1
    finally:
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


sys.exit(selftest() if mode == "selftest" else fetch())
PY
