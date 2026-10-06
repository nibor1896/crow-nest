#!/usr/bin/env bash
# Builds `serve` for shipping and packs it, then proves the package carries no trace of
# the machine it was built on (#133) and no NVIDIA file. NVRTC is NOT in the package:
# the user fetches it from NVIDIA's own PyPI wheel with tools/fetch-nvrtc.sh.
# Linux twin of tools/pack-engine.ps1 (#131).
#
# A plain `cargo build` serve is not portable (#133, measured 2026-10-02 at 65ce4d4):
# it carried the builder's home path 255 times (panic locations under ~/.cargo/registry
# and the compiled-in lock path). This script:
#
#  1. builds `serve` with `--remap-path-prefix` for $HOME and the repo root, through
#     CARGO_ENCODED_RUSTFLAGS (0x1f separated, so a path with spaces stays one
#     argument). The last matching remap wins, so the repo root, which usually lies
#     inside $HOME, comes second;
#  2. refuses when serve's highest GLIBC_ symbol version is above GLIBC_FLOOR (2.34,
#     measured 2026-10-02); the floor is written into MANIFEST.json;
#  3. stages serve and LICENSE, NOTICE and THIRD-PARTY-NOTICES.txt (the crates' licence
#     texts), refusing first when tools/engine_notices.py says THIRD-PARTY-NOTICES.txt
#     is not what engine/Cargo.lock produces, and refusing when the stage holds any
#     NVIDIA file (nvrtc, nvidia, cuda, cublas, cudart in a name) or anything else
#     than those four. NVRTC (libnvrtc.so, the bytes of NVIDIA's libnvrtc.so.13, and
#     libnvrtc-builtins.so.13.3) is placed beside serve by tools/fetch-nvrtc.sh, which
#     keeps the naming reasons (cudarc dlopens the bare libnvrtc.so first: #133, Crow #341);
#  4. scans every staged file, as UTF-8 and as UTF-16LE at both byte alignments, for
#     $HOME, any `/home/<name>/`, the bare $USER (any context, case-insensitive) and
#     the host name, and REFUSES on any hit (no override);
#  5. writes MANIFEST.json and dist/crow-nest-engine-<version>-linux-x64.tar.gz.
#
# MANIFEST.json shape: {"glibc_min": "2.34", "files": [{path, bytes, sha256}, ...]}.
# `files` is the Windows array (path with forward slashes, sha256 upper-case hex,
# MANIFEST.json itself not listed); the object wraps it to carry the glibc floor.
#
# The consumer unpacks, runs tools/fetch-nvrtc.sh --target <that folder> and puts the
# folder on LD_LIBRARY_PATH (libnvrtc-builtins is loaded by libnvrtc and has no RUNPATH,
# so serve itself needs no rpath).
#
# Usage:  tools/pack-engine.sh [--version V] [--out DIR]
#         tools/pack-engine.sh --scan FILE...   scan files with this machine's identity
#         tools/pack-engine.sh --selftest       checks on synthetic inputs, no build
# Default version: `git describe --tags --always --dirty` without the leading `v`
# (engine/Cargo.toml carries 0.1.0, which is not a release number).
set -euo pipefail

GLIBC_FLOOR="2.34"
# The package holds no NVIDIA file: nothing is staged from a CUDA install.
nvrtc_staged_names() { :; }
# from the repo root: crow-nest's licence, NOTICE and the crates' texts
SHIP_FILES=(LICENSE NOTICE THIRD-PARTY-NOTICES.txt)
# the files a package holds, MANIFEST.json aside
package_files() { echo serve; printf '%s\n' "${SHIP_FILES[@]}"; }
# stage_nvidia_files DIR: every file under DIR whose name looks like NVIDIA's (nvrtc,
# nvidia, cuda, cublas, cudart; any case); prints them, exit 0 if there is any
stage_nvidia_files() {
  local hits; hits="$(find "$1" \( -type f -o -type l \) | grep -iE '(nvrtc|nvidia|cuda|cublas|cudart)[^/]*$' || true)"
  if [ -n "$hits" ]; then echo "$hits"; return 0; fi
  return 1
}
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ---------------------------------------------------------------------------
# Checks (the selftest calls exactly these)
# ---------------------------------------------------------------------------

# scan_file FILE HOME USER HOST: print one line per (needle, encoding) hit as
# "<needle label>\t<encoding>\t<count>\t<first match>"; exit 1 if any.
# Needles: HOME, /home/<any>/, bare USER, HOST. Case-insensitive; UTF-8 (the raw
# bytes) and UTF-16LE at offset 0 and 1. One byte <-> one char (latin-1 view).
scan_file() {
  python3 - "$@" <<'PY'
import re, sys
path, home, user, host = sys.argv[1:5]
data = open(path, 'rb').read()
needles = []
if home:
    h = home.rstrip('/')
    needles.append(("HOME " + h, re.escape(h)))
needles.append(("/home/<name>/", r'/home/[^/\x00-\x1f"<>|:*?]{1,64}/'))
if user:
    needles.append(("USER " + user + " (bare)", re.escape(user)))
if host:
    needles.append(("HOSTNAME " + host, re.escape(host)))
views = [("utf-8", data.decode('latin-1'))]
for name, off in (("utf-16le", 0), ("utf-16le+1", 1)):
    b = data[off:]
    b = b[:len(b) - (len(b) % 2)]
    views.append((name, b.decode('utf-16-le', errors='replace')))
total = 0
for enc, text in views:
    for label, rx in needles:
        ms = list(re.finditer(rx, text, re.I))
        if ms:
            total += len(ms)
            print(f"{label}\t{enc}\t{len(ms)}\t{ms[0].group(0)!r}")
sys.exit(1 if total else 0)
PY
}

# scan_bytes STRING-OR-FILE-CONTENT via printf; selftest helper
glibc_max() { # glibc_max FILE -> highest GLIBC_x.y[.z] version, empty if none
  objdump -T "$1" 2>/dev/null | grep -o 'GLIBC_[0-9][0-9.]*' | sed 's/^GLIBC_//' | sort -V | tail -1
}

# version_le A B: A <= B in version order
version_le() { [ "$(printf '%s\n%s\n' "$1" "$2" | sort -V | head -1)" = "$1" ]; }

pack_name() { echo "crow-nest-engine-$1-linux-x64.tar.gz"; }

# manifest_json STAGE: the MANIFEST.json text for every file except MANIFEST.json
manifest_json() {
  python3 - "$1" "$GLIBC_FLOOR" <<'PY'
import hashlib, json, os, sys
stage, floor = sys.argv[1], sys.argv[2]
files = []
for root, _, names in os.walk(stage):
    for n in names:
        p = os.path.join(root, n)
        rel = os.path.relpath(p, stage).replace(os.sep, '/')
        if rel == 'MANIFEST.json':
            continue
        d = open(p, 'rb').read()
        files.append({"path": rel, "bytes": len(d), "sha256": hashlib.sha256(d).hexdigest().upper()})
files.sort(key=lambda f: f["path"])
print(json.dumps({"glibc_min": floor, "files": files}, indent=2))
PY
}

host_name() { hostname 2>/dev/null || cat /proc/sys/kernel/hostname; }

# ---------------------------------------------------------------------------
# Selftest
# ---------------------------------------------------------------------------

ok=0; red=0
check() { # check NAME EXPECT(hit|clean) FILE
  local name="$1" expect="$2" f="$3" got
  if scan_file "$f" /home/builder builder buildbox >/dev/null; then got=clean; else got=hit; fi
  if [ "$got" = "$expect" ]; then echo "  ok   $name"; ok=$((ok+1)); else echo "  FAIL $name (got $got)"; red=$((red+1)); fi
}
check_true() { if [ "$2" = 1 ]; then echo "  ok   $1"; ok=$((ok+1)); else echo "  FAIL $1"; red=$((red+1)); fi; }

selftest() {
  echo "pack-engine selftest (synthetic identity: /home/builder, builder, buildbox)"
  local t; t="$(mktemp -d "${TMPDIR:-/tmp}/crow-nest-pack-selftest.XXXXXX")"
  trap 'rm -rf "$t"' RETURN
  w() { printf '%s' "$2" > "$t/$1"; }
  w clean "panicked at ~/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/serde-1.0/src/de.rs"
  check "a clean binary has no hit" clean "$t/clean"
  w home "x /home/builder/.cargo/registry y"
  check "HOME in UTF-8 is a hit" hit "$t/home"
  w homecase "/HOME/BUILDER/x"
  check "HOME in another case is a hit" hit "$t/homecase"
  printf '%s' "/home/builder/src" | iconv -f UTF-8 -t UTF-16LE > "$t/u16"
  check "HOME in UTF-16LE is a hit" hit "$t/u16"
  { printf 'A'; cat "$t/u16"; } > "$t/u16odd"
  check "UTF-16LE at an odd offset is a hit" hit "$t/u16odd"
  w other "/home/someone/src/lib.rs"
  check "ANOTHER user's /home/<name>/ is a hit" hit "$t/other"
  check "/home/<name>/ needs the closing slash (/home/someone alone is not)" clean <(printf '%s' "/home/someone") 
  w word "rebuilder and builders"
  check "the user name inside a word is a hit (bare-name rule)" hit "$t/word"
  w nouser "the owner's decision 2026-09-25"
  check "text without the user name is NOT a hit" clean "$t/nouser"
  w prose "default since 2026-09-09 (Builder's decision)"
  check "the bare user name in prose is a hit (UTF-8)" hit "$t/prose"
  printf '%s' "decided by BUILDER" | iconv -f UTF-8 -t UTF-16LE > "$t/prose16"
  check "the bare user name in prose is a hit (UTF-16LE)" hit "$t/prose16"
  w host "host=buildbox.local"
  check "the host name is a hit" hit "$t/host"
  w nohost "host=otherbox.local"
  check "another host name is NOT a hit" clean "$t/nohost"
  : > "$t/empty"
  check "an empty file has no hit" clean "$t/empty"

  # manifest shape: forward slashes, upper-case hex, glibc floor, MANIFEST.json not listed
  mkdir -p "$t/st/sub"; printf abc > "$t/st/sub/abc.txt"; printf '[]' > "$t/st/MANIFEST.json"
  local m; m="$(manifest_json "$t/st")"
  check_true "the manifest lists the file, not itself" "$(python3 -c 'import json,sys; print(int(len(json.loads(sys.argv[1])["files"])==1))' "$m")"
  check_true "the manifest path uses forward slashes" "$(python3 -c 'import json,sys; print(int(json.loads(sys.argv[1])["files"][0]["path"]=="sub/abc.txt"))' "$m")"
  check_true "the manifest carries the byte count" "$(python3 -c 'import json,sys; print(int(json.loads(sys.argv[1])["files"][0]["bytes"]==3))' "$m")"
  check_true "the manifest sha256 is upper-case hex" "$(python3 -c 'import json,sys; print(int(json.loads(sys.argv[1])["files"][0]["sha256"]=="BA7816BF8F01CFEA414140DE5DAE2223B00361A396177A9CB410FF61F20015AD"))' "$m")"
  check_true "the manifest carries the glibc floor" "$(python3 -c 'import json,sys; print(int(json.loads(sys.argv[1])["glibc_min"]==sys.argv[2]))' "$m" "$GLIBC_FLOOR")"

  # glibc floor gate
  check_true "2.34 is within the 2.34 floor" "$(version_le 2.34 2.34 && echo 1 || echo 0)"
  check_true "2.9 is within the 2.34 floor (version order, not text)" "$(version_le 2.9 2.34 && echo 1 || echo 0)"
  check_true "2.35 is above the 2.34 floor" "$(version_le 2.35 2.34 && echo 0 || echo 1)"
  check_true "the highest of 2.2.5, 2.34, 2.17 is 2.34" "$([ "$(printf '2.2.5\n2.34\n2.17\n' | sort -V | tail -1)" = 2.34 ] && echo 1 || echo 0)"
  check_true "NVRTC is staged as libnvrtc.so (cudarc tries that name first)" "$(nvrtc_staged_names | grep -qx 'libnvrtc.so' && echo 1 || echo 0)"
  check_true "no versioned libnvrtc.so.<n> is staged (it would be a duplicate)" "$(nvrtc_staged_names | grep -qE '^libnvrtc\.so\.' && echo 0 || echo 1)"
  check_true "the builtins keep the name libnvrtc dlopens" "$(nvrtc_staged_names | grep -qx 'libnvrtc-builtins.so.13.3' && echo 1 || echo 0)"
  check_true "the pack name" "$([ "$(pack_name 0.8.0)" = crow-nest-engine-0.8.0-linux-x64.tar.gz ] && echo 1 || echo 0)"

  # the package holds no NVIDIA file
  check_true "the package list names no NVIDIA file (nvrtc, nvidia, cuda in a name)" "$(package_files | grep -qiE 'nvrtc|nvidia|cuda' && echo 0 || echo 1)"
  check_true "the package is exactly serve, LICENSE, NOTICE, THIRD-PARTY-NOTICES.txt (+ MANIFEST.json)" "$([ "$(package_files | sort | tr '\n' ' ')" = 'LICENSE NOTICE THIRD-PARTY-NOTICES.txt serve ' ] && echo 1 || echo 0)"
  mkdir -p "$t/nv_ok" "$t/nv_bad"
  : > "$t/nv_ok/serve"; : > "$t/nv_ok/LICENSE"; : > "$t/nv_ok/NOTICE"; : > "$t/nv_ok/THIRD-PARTY-NOTICES.txt"
  check_true "the stage check passes the four package files" "$(stage_nvidia_files "$t/nv_ok" >/dev/null && echo 0 || echo 1)"
  : > "$t/nv_bad/libnvrtc.so"
  check_true "the stage check refuses a staged libnvrtc.so" "$(stage_nvidia_files "$t/nv_bad" >/dev/null && echo 1 || echo 0)"
  rm -f "$t/nv_bad/libnvrtc.so"; : > "$t/nv_bad/libnvrtc-builtins.so.13.3"
  check_true "the stage check refuses a staged libnvrtc-builtins.so.13.3" "$(stage_nvidia_files "$t/nv_bad" >/dev/null && echo 1 || echo 0)"
  rm -f "$t/nv_bad/libnvrtc-builtins.so.13.3"; : > "$t/nv_bad/NVRTC64_130_0.DLL"
  check_true "the stage check refuses a Windows NVRTC DLL in any case" "$(stage_nvidia_files "$t/nv_bad" >/dev/null && echo 1 || echo 0)"
  rm -f "$t/nv_bad/NVRTC64_130_0.DLL"; : > "$t/nv_bad/libcudart.so.13"
  check_true "the stage check refuses another CUDA runtime file" "$(stage_nvidia_files "$t/nv_bad" >/dev/null && echo 1 || echo 0)"

  # the notices travel in the package and the privacy scan covers them
  local f
  for f in LICENSE NOTICE THIRD-PARTY-NOTICES.txt; do
    check_true "the package carries $f" "$(package_files | grep -qx "$f" && echo 1 || echo 0)"
  done
  check_true "the package is 6 distinct files + MANIFEST.json" "$([ "$(package_files | sort -u | wc -l)" -eq 6 ] && [ "$(package_files | wc -l)" -eq 6 ] && echo 1 || echo 0)"
  w NOTICE "see /home/builder/dev/NOTICE"
  check "the scan finds a leak in a NOTICE" hit "$t/NOTICE"
  printf '%s' "built on buildbox" | iconv -f UTF-8 -t UTF-16LE > "$t/THIRD-PARTY-NOTICES.txt"
  check "the scan finds a leak in THIRD-PARTY-NOTICES.txt (UTF-16LE)" hit "$t/THIRD-PARTY-NOTICES.txt"

  # the repo's real texts, scanned with THIS machine's identity, as the build will
  local repo_root; repo_root="$(cd "$SELF_DIR/.." && pwd)"
  for f in "${SHIP_FILES[@]}"; do
    check_true "$f exists in the repo root" "$([ -f "$repo_root/$f" ] && echo 1 || echo 0)"
    if [ -f "$repo_root/$f" ]; then
      check_true "$f passes this machine's privacy scan" "$(scan_file "$repo_root/$f" "${HOME%/}" "${USER:-$(id -un)}" "$(host_name)" >/dev/null && echo 1 || echo 0)"
    fi
  done

  echo "selftest: $ok ok, $red failed"
  [ "$red" -eq 0 ]
}

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

version=""; out=""
while [ $# -gt 0 ]; do
  case "$1" in
    --selftest) selftest; exit $? ;;
    --scan)
      shift; rc=0
      for f in "$@"; do
        n=0
        if res="$(scan_file "$f" "${HOME:-}" "${USER:-$(id -un)}" "$(host_name)")"; then :; else
          echo "$res" | sed "s|^|$f: |"
          n="$(echo "$res" | awk -F'\t' '{s+=$3} END{print s+0}')"; rc=1
        fi
        echo "scan $f: $n hit(s)"
      done
      exit $rc ;;
    --version) version="$2"; shift 2 ;;
    --out) out="$2"; shift 2 ;;
    -h|--help) sed -n 2,41p "$0"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

repo="$(cd "$SELF_DIR/.." && pwd)"
if [ -z "$version" ]; then
  d="$(git -C "$repo" describe --tags --always --dirty 2>/dev/null || true)"
  [ -n "$d" ] || { echo "git describe gave no version - pass --version" >&2; exit 1; }
  version="${d#v}"
fi
[ -n "$out" ] || out="$repo/dist"
mkdir -p "$out"; out="$(cd "$out" && pwd)"
[ -n "${HOME:-}" ] || { echo "HOME is not set - the remap and the privacy scan need it" >&2; exit 1; }
home="${HOME%/}"
user="${USER:-$(id -un)}"
host="$(host_name)"

# the crates' licence texts must be the ones this Cargo.lock compiles in
python3 "$repo/tools/engine_notices.py" "$repo" || {
  echo "REFUSED: tools/engine_notices.py failed - THIRD-PARTY-NOTICES.txt or NOTICE is stale; nothing was built" >&2; exit 1; }

echo "packing crow-nest engine $version"
echo "  repo   : $repo"
echo "  out    : $out"

target="$repo/engine/target_pack"
US=$'\x1f'
flags="--remap-path-prefix=$home=~${US}--remap-path-prefix=$repo=crow-nest"
( cd "$repo/engine" && unset RUSTFLAGS && CARGO_ENCODED_RUSTFLAGS="$flags" cargo build --release --bin serve --target-dir "$target" )
serve="$target/release/serve"
[ -f "$serve" ] || { echo "no serve at $serve" >&2; exit 1; }

gmax="$(glibc_max "$serve")"
echo "  glibc  : highest GLIBC_${gmax:-none} (floor $GLIBC_FLOOR)"
if [ -n "$gmax" ] && ! version_le "$gmax" "$GLIBC_FLOOR"; then
  echo "REFUSED: serve needs GLIBC_$gmax, above the recorded floor $GLIBC_FLOOR" >&2; exit 1
fi

stage="$out/crow-nest-engine-$version-linux-x64"
rm -rf "$stage"; mkdir -p "$stage"
cp "$serve" "$stage/serve"; chmod 755 "$stage/serve"
for f in "${SHIP_FILES[@]}"; do cp "$repo/$f" "$stage/$f"; done
if nvidia="$(stage_nvidia_files "$stage")"; then
  echo "REFUSED: the stage holds an NVIDIA file ($(echo "$nvidia" | tr '\n' ' ')) - crow-nest does not distribute NVRTC; tools/fetch-nvrtc.sh fetches it from NVIDIA" >&2; exit 1
fi
if [ "$(ls -A "$stage" | sort)" != "$(package_files | sort)" ]; then
  echo "REFUSED: the stage holds $(ls -A "$stage" | tr '\n' ' ')- expected $(package_files | tr '\n' ' ')" >&2; exit 1
fi

# privacy scan: refuse before anything is packed
total=0
for f in "$stage"/*; do
  if res="$(scan_file "$f" "$home" "$user" "$host")"; then :; else
    echo "$res" | sed "s|^|  $(basename "$f"): |" >&2
    total=$((total + $(echo "$res" | awk -F'\t' '{s+=$3} END{print s+0}')))
  fi
done
if [ "$total" -gt 0 ]; then
  echo "REFUSED: privacy scan: $total hit(s) in the staged files - nothing was packed" >&2; exit 1
fi
echo "  privacy scan: 0 hits in $(ls "$stage" | wc -l) files (4 needle classes, UTF-8 and UTF-16LE)"

manifest_json "$stage" > "$stage/MANIFEST.json"
tarball="$out/$(pack_name "$version")"
rm -f "$tarball"
tar --sort=name --owner=0 --group=0 --numeric-owner -C "$stage" -czf "$tarball" .
nfiles="$(python3 -c 'import json,sys; print(len(json.load(open(sys.argv[1]))["files"]))' "$stage/MANIFEST.json")"
echo "RESULT: $nfiles files in the manifest (+ MANIFEST.json), glibc_min $GLIBC_FLOOR, $(du -h "$tarball" | cut -f1) packed"
echo "  $tarball"
