#!/usr/bin/env python3
"""#147 (GLM-5.3-Flash step 8): the routing passes of PREREG amendment 1, one file after another,
through the layerwise runner (oracle/glm5_layerwise.py, #158) on the full CNQ container or (#179) on
the FP8 originals.

  .venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py [--dry-run]
      [--container converter/GLM-5.3-Flash-CNQ4.5.cnq | --fp8 models/GLM-5.3-Flash-original]
      [--corpus decode_out/glm-step8/corpus] [--runs decode_out/glm-step8/runs] [--only <name>,...]
      [--prompt-chunk 512]

Before anything runs (exit 2 with the reason otherwise):
  - the container is complete: no `<cnq>.journal.jsonl` beside it (the converter deletes it only after
    the index trailer is written, converter/src/main.rs), magic CNQ1, an index trailer that parses
    (index v2, recipe cnq4.5-glm5-next, not partial, source rev eb9eb208), every text layer 0..L-1 and
    the embedding, final norm and lm_head in the index;
  - or (--fp8, #179) the FP8 originals are verified by tools/fetch-glm.py: its revision record
    hf-revision.json is that of rev eb9eb208 with 62 shards, and config.json, model.safetensors.index.json
    and every shard have a `.verified` marker equal to that record and of that revision (fetch-glm.py's
    `is_verified`; no shard is read or hashed here); the index still has its marker's sha256;
  - corpus.json and each file's ids and mask have the sha256 of amendment 1 (pinned below, checked
    against runs/glm53-flash/PREREG.md by the tests), and each ids file holds the routed 32,768 tokens.
Per file, in the amendment's order, the runner pass of the amendment's dump plan plus the prompt-chunk
and delete-behind options (#147):
  oracle/glm5_layerwise.py run --weights container <cnq> --ids <corpus>/<name>-ids.json --anchors 32767
      --state-dtype bf16 --prompt-chunk 512 --delete-states-behind --out <runs>/<name>
(`--weights fp8-originals <dir>` with --fp8). Each pass dir gets weights.json, the weights' identity
(kind, revision, index sha256, for FP8 the 62 verified shard sha256), before the runner starts.
Resumable per file: a pass that is complete over every layer is skipped; an interrupted one continues
with `--layers k+1:` from the last recorded state l<k>-output.* (the runner records layer k before it
deletes the state behind it); one with no state to continue from starts again at layer 0. A run dir made
with other weights (identity, index sha256 or kind) or other ids is refused, never overwritten.
After each pass: the self-test of `tools/glm_tier_sim.py sim` over the out dir (routing sha256 ==
manifest, [N][8], ids 0..287 ascending) and the source check: for the container the sim's (CNQ, not
partial, complete), for --fp8 the runner manifest's weights are FP8 and the pass is complete. One line
per pass, also a failed one, goes to <runs>/passes.jsonl (the measurement-book row).
"""
import argparse
import datetime
import hashlib
import importlib.util
import json
import os
import struct
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import glm_tier_sim as ts  # noqa: E402

# #179: tools/fetch-glm.py owns the FP8 originals' verification record (hf-revision.json + `.verified`
# markers); its checks are reused, not copied (as tools/glm-stage.py does)
_SPEC = importlib.util.spec_from_file_location("fetch_glm", os.path.join(HERE, "fetch-glm.py"))
fg = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(fg)

RUNNER = os.path.join(ROOT, "oracle", "glm5_layerwise.py")
CONTAINER = os.path.join(ROOT, "converter", "GLM-5.3-Flash-CNQ4.5.cnq")
FP8 = os.path.join(ROOT, "models", "GLM-5.3-Flash-original")
CORPUS = os.path.join(ROOT, "decode_out", "glm-step8", "corpus")
RUNS = os.path.join(ROOT, "decode_out", "glm-step8", "runs")
TOKENS = 32768
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
RECIPE = "cnq4.5-glm5-next"
LM = "model.language_model."
FP8_SMALL = ("config.json", "model.safetensors.index.json")  # the small files the runner reads
MANIFEST_KIND = {"container": "cnq", "fp8-originals": "fp8"}  # runner kind -> manifest weights word

# PREREG amendment 1 (2026-10-08, runs/glm53-flash/PREREG.md): corpus.json and, per file in the
# table's order, (name, role, ids sha256 of the routed 32,768 ids, mask file sha256)
CORPUS_JSON_SHA256 = "8fb9560f43a58ed5a1791ed2c458130a0febf4236f39ef882c12e665faae5f26"
AMENDMENT_1 = (
    ("todo-1006", "held", "0c30a34d5d212fc9f072f803dd964a6b45f56335e157247bfe7fb09874f359e3",
     "c306f565542c1f7d6c42605a25cea37439711a08e17d514f60e5c9a0aa5c2674"),
    ("omarchy-0915a", "cal", "083567463f108b8a562d2d398013b24164573bbfe4808d0a5f9043f83c7e7380",
     "5398ae6c1691bdfa921ce285e4ec4a8aee4ad1b683b7ff87d33f60935ab7325b"),
    ("lenis-0830", "cal", "7f411cf6a2a44bbd90714ca123201c4a4bd875849b3fdae25ddf3444c0e83085",
     "01b384512eb3620a0df2ba56b705f26b6337a18971833a4fe54c3c90a374b037"),
    ("ctx7-0830", "cal", "d73e494ed333be8af6b5fc1c07d9da82d873676b453a49cbb467e234c2790044",
     "b84763b3335b0899cc00cefd666f8bcfcc010109096549811ce34137e9cd4960"),
    ("zetalab-0829", "cal", "4e96d0d09f84c070c4f914dfcd3d0580693d8782e540db74cce162f258953bf7",
     "7cf87ef8191f33f105b9fde090d1e85e51933853e172426abbeb277f09fa0a51"),
)


class Refusal(Exception):
    """The passes cannot start (or continue) on these inputs."""


# ------------------------------------------------------------------ preflight

def container_index(path):
    """the index of a COMPLETE container, or Refusal with the reason. Reads the magic, the last 8 bytes
    and the index JSON only."""
    journal = path + ".journal.jsonl"
    if not os.path.isfile(path):
        raise Refusal("%s: no such file" % path)
    if os.path.exists(journal):
        raise Refusal("%s exists: the conversion is still running or was interrupted (the converter deletes "
                      "its journal after writing the index trailer)" % journal)
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        if f.read(4) != b"CNQ1":
            raise Refusal("%s: not a CNQ1 container" % path)
        if size < 12 + 8:
            raise Refusal("%s: %d B, too short for an index trailer" % (path, size))
        f.seek(-8, 2)
        n = struct.unpack("<Q", f.read(8))[0]
        if not 2 <= n <= size - 12 - 8:
            raise Refusal("%s: no index trailer (trailer length %d does not fit a %d B file)" % (path, n, size))
        f.seek(-(8 + n), 2)
        raw = f.read(n)
    try:
        idx = json.loads(raw)
    except (UnicodeDecodeError, ValueError) as e:
        raise Refusal("%s: no index trailer (the last %d B are not JSON: %s)" % (path, n, e))
    if not isinstance(idx, dict) or idx.get("format_version") != 2 or idx.get("recipe") != RECIPE:
        raise Refusal("%s: index format %s recipe %s, expected v2 %s" % (
            path, idx.get("format_version") if isinstance(idx, dict) else "?",
            idx.get("recipe") if isinstance(idx, dict) else "?", RECIPE))
    if idx.get("partial"):
        raise Refusal("%s: a PARTIAL container (%s); G1 needs the full one" % (
            path, idx["partial"].get("filter") if isinstance(idx["partial"], dict) else idx["partial"]))
    rev = ((idx.get("model") or {}).get("source") or {}).get("revision")
    if rev != REVISION:
        raise Refusal("%s: source revision %s, expected %s" % (path, rev, REVISION))
    names = {t["name"] for t in idx.get("tensors", [])}
    cfg = json.loads(idx["model"]["config_json"])
    L = (cfg.get("text_config") or cfg)["num_hidden_layers"]
    missing = [l for l in range(L) if not any(nm.startswith("%slayers.%d." % (LM, l)) for nm in names)]
    missing += [nm for nm in (LM + "embed_tokens.weight", LM + "norm.weight", "lm_head.weight") if nm not in names]
    if missing:
        raise Refusal("%s: the index lacks %s" % (path, ", ".join(map(str, missing[:8]))))
    return idx, hashlib.sha256(raw).hexdigest()


def identity_sha256(ident):
    """sha256 of the identity's canonical JSON, without its own identity_sha256"""
    body = {k: v for k, v in ident.items() if k != "identity_sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _sealed(ident):
    ident["identity_sha256"] = identity_sha256(ident)
    return ident


def container_identity(index_sha):
    return _sealed({"kind": "container", "revision": REVISION, "index_json_sha256": index_sha})


def fp8_identity(path):
    """#179: the identity of FP8 originals every file of which tools/fetch-glm.py verified against the HF
    record of rev eb9eb208, or Refusal naming what is not. Reads hf-revision.json, the markers, the index
    and config.json; the shards are only stat'ed (fetch-glm.py's `is_verified`), never read."""
    if fg.REVISION != REVISION:
        raise Refusal("tools/fetch-glm.py verifies revision %s, the passes need %s" % (fg.REVISION, REVISION))
    if not os.path.isdir(path):
        raise Refusal("%s: no such directory" % path)
    api = os.path.join(path, fg.API_FILE)
    if not os.path.isfile(api):
        raise Refusal("%s: no such file (tools/fetch-glm.py saves the HF revision record there)" % api)
    try:
        with open(api, encoding="utf-8") as f:
            table = fg.file_table(json.load(f))  # refuses another revision
        shards = fg.check_table(table)  # the 9 small files and 62 shards with lfs.sha256
    except (OSError, ValueError, KeyError, TypeError) as e:
        raise Refusal("%s: %s" % (api, e))
    names = FP8_SMALL + tuple(shards)
    bad, shas = [], {}
    for name in names:
        p = os.path.join(path, name)
        mk = fg.marker_of(p)
        if not os.path.isfile(p):
            bad.append("%s (missing)" % name)
        elif not mk.is_file():
            bad.append("%s (no .verified marker: not verified by tools/fetch-glm.py)" % name)
        elif not fg.is_verified(p, table[name]):
            bad.append("%s (.verified marker or size differs from the HF record)" % name)
        else:
            rec = json.loads(mk.read_text(encoding="utf-8"))
            if rec.get("revision") != REVISION:
                bad.append("%s (marker of revision %s)" % (name, rec.get("revision")))
            else:
                shas[name] = rec.get("sha256")
    if bad:
        raise Refusal("%s: %d of %d files not verified at rev %s: %s%s" % (
            path, len(bad), len(names), REVISION[:8], "; ".join(bad[:4]), " ..." if len(bad) > 4 else ""))
    small = {}
    for name in FP8_SMALL:  # the runner reads these two; they must still be the verified bytes
        small[name] = ts.sha256_file(os.path.join(path, name))
        if small[name] != shas[name]:
            raise Refusal("%s: sha256 %s, not the %s tools/fetch-glm.py verified" % (
                os.path.join(path, name), small[name], shas[name]))
    return _sealed({"kind": "fp8-originals", "repo": fg.REPO, "revision": REVISION,
                    "index_json_sha256": small["model.safetensors.index.json"],
                    "config_json_sha256": small["config.json"],
                    "shards": {n: shas[n] for n in shards}})


def write_identity(out_dir, identity):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "weights.json"), "w", encoding="utf-8") as f:
        json.dump(identity, f, indent=1, sort_keys=True)
        f.write("\n")


def check_corpus(corpus_dir, names, tokens=TOKENS, table=AMENDMENT_1, corpus_sha=CORPUS_JSON_SHA256):
    """amendment 1's sha256 for corpus.json and each file -> {name: ids path}"""
    cj = os.path.join(corpus_dir, "corpus.json")
    if not os.path.isfile(cj):
        raise Refusal("%s: no such file" % cj)
    if ts.sha256_file(cj) != corpus_sha:
        raise Refusal("%s: sha256 is not amendment 1's %s" % (cj, corpus_sha))
    rows = {r[0]: r for r in table}
    out = {}
    for name in names:
        if name not in rows:
            raise Refusal("%s is not a file of amendment 1 (%s)" % (name, ", ".join(rows)))
        _, _, ids_sha, mask_sha = rows[name]
        ip, mp = (os.path.join(corpus_dir, "%s-%s.json" % (name, k)) for k in ("ids", "mask"))
        for p in (ip, mp):
            if not os.path.isfile(p):
                raise Refusal("%s: no such file" % p)
        ids = ts.jload(ip)
        if len(ids) != tokens:
            raise Refusal("%s: %d ids, the pass routes the first %d and tools/glm_tier_sim.py compares the "
                          "runner's ids with the whole file" % (ip, len(ids), tokens))
        if ts.sha256_ids(ids) != ids_sha:
            raise Refusal("%s: ids sha256 is not amendment 1's %s" % (ip, ids_sha))
        if ts.sha256_file(mp) != mask_sha:
            raise Refusal("%s: sha256 is not amendment 1's %s" % (mp, mask_sha))
        out[name] = ip
    return out


# ------------------------------------------------------------------ one pass

def plan(out_dir, ids, index_sha, identity=None):
    """('done', None) | ('resume', k) | ('fresh', None); Refusal for a dir of other weights or ids
    (#179: another identity in its weights.json, or another weights kind in the runner manifest)"""
    wp = os.path.join(out_dir, "weights.json")
    if identity is not None and os.path.exists(wp):
        got = ts.jload(wp)
        if got.get("identity_sha256") != identity["identity_sha256"]:
            raise Refusal("%s: made with other weights (%s, identity sha256 %s; now %s, %s); move it away" % (
                out_dir, got.get("kind"), got.get("identity_sha256"), identity["kind"],
                identity["identity_sha256"]))
    mp = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(mp):
        return "fresh", None
    man = ts.jload(mp)
    if identity is not None:
        word = str((man.get("weights") or {}).get("weights", "")).split(" ")[0]
        if word != MANIFEST_KIND[identity["kind"]]:
            raise Refusal("%s: made with %s weights, this run uses %s (%s); move it away" % (
                out_dir, word or "unknown", MANIFEST_KIND[identity["kind"]], identity["kind"]))
    got = (man.get("weights") or {}).get("index_json_sha256")
    if index_sha is not None and got != index_sha:
        raise Refusal("%s: made with other weights (index sha256 %s, these weights %s); move it away" % (
            out_dir, got, index_sha))
    if man.get("ids") != ids:
        raise Refusal("%s: made over other ids than this corpus file; move it away" % out_dir)
    L = man.get("num_hidden_layers")
    if man.get("complete") and list(man.get("layers", [])) == [0, L]:
        return "done", None
    done = sorted(r["layer"] for r in man.get("per_layer", []))
    if done and done == list(range(done[-1] + 1)):
        k = done[-1]
        if any(os.path.exists(os.path.join(out_dir, "l%d-output.%s" % (k, e))) for e in ("bf16", "f32")):
            return "resume", k + 1
    return "fresh", None


def runner_cmd(python, weights, ids_path, out_dir, tokens, prompt_chunk, start=None, runner=RUNNER,
               state_dtype="bf16"):
    kind, path = weights
    cmd = [python, "-I", runner, "run", "--weights", kind, path, "--ids", ids_path,
           "--anchors", str(tokens - 1), "--state-dtype", state_dtype, "--prompt-chunk", str(prompt_chunk),
           "--delete-states-behind", "--out", out_dir]
    if start:
        cmd += ["--layers", "%d:" % start]
    return cmd


def source_problems(man, kind="container"):
    """the source check of a pass: the sim's for the container (CNQ, not partial, complete); for the FP8
    originals (#179) the runner manifest's weights are FP8 and the pass is complete over every layer"""
    if kind == "container":
        return ts.source_reasons(man)
    word = str((man.get("weights") or {}).get("weights", "")).split(" ")[0]
    out = [] if word == MANIFEST_KIND[kind] else ["routing not from the FP8 originals (manifest weights %s)" % (
        word or "unknown")]
    if not man.get("complete") or list(man.get("layers", [])) != [0, man.get("num_hidden_layers", -1)]:
        out.append("runner pass not complete over layers 0..%s" % man.get("num_hidden_layers"))
    return out


def verify(out_dir, shape=ts.SHAPE, of_record=True, kind="container"):
    """the sim's own self-test of the routing files (+ the source check when of_record) -> [problems]"""
    try:
        man, _ = ts.load_runner(out_dir, shape)
    except ts.SimError as e:
        return [str(e)]
    return source_problems(man, kind) if of_record else []


def _git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except OSError:
        return None


def run_passes(names, ids_paths, weights, runs_dir, index_sha, tokens=TOKENS, prompt_chunk=512,
               python=sys.executable, runner=RUNNER, shape=ts.SHAPE, of_record=True, log=print, dry_run=False,
               state_dtype="bf16", identity=None):
    """one pass per name, in order; returns 0 when every pass is done and verified, 1 at the first failure.
    identity (#179): the weights' identity, written to <out>/weights.json before the runner starts"""
    os.makedirs(runs_dir, exist_ok=True)
    book = os.path.join(runs_dir, "passes.jsonl")
    kind = weights[0]
    for name in names:
        out_dir = os.path.join(runs_dir, name)
        ids = ts.jload(ids_paths[name])
        what, start = plan(out_dir, ids, index_sha, identity)
        if what == "done":
            bad = verify(out_dir, shape, of_record, kind)
            if bad:
                log("%s: complete, but: %s" % (name, "; ".join(bad)))
                return 1
            log("%s: done (verified), skipped" % name)
            continue
        cmd = runner_cmd(python, weights, ids_paths[name], out_dir, tokens, prompt_chunk, start, runner,
                         state_dtype)
        log("%s: %s -> %s" % (name, "resume at layer %d" % start if start else "fresh pass",
                              " ".join(cmd)))
        if dry_run:
            continue
        os.makedirs(out_dir, exist_ok=True)
        if identity is not None and not os.path.exists(os.path.join(out_dir, "weights.json")):
            write_identity(out_dir, identity)
        t0, s0 = time.time(), datetime.datetime.now(datetime.timezone.utc)
        with open(os.path.join(out_dir, "runner.log"), "a", encoding="utf-8") as lf:
            lf.write("# %s %s\n" % (s0.isoformat(timespec="seconds"), " ".join(cmd)))
            with subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                  encoding="utf-8", errors="replace", bufsize=1) as p:
                for line in p.stdout:
                    lf.write(line)
                    lf.flush()
                    log("  %s | %s" % (name, line.rstrip()))
                rc = p.wait()
        wall = time.time() - t0
        bad = verify(out_dir, shape, of_record, kind) if rc == 0 else ["runner rc %d (see %s)" % (
            rc, os.path.join(out_dir, "runner.log"))]
        row = {"name": name, "start_utc": s0.isoformat(timespec="seconds"), "wall_s": round(wall, 1),
               "rc": rc, "resumed_from_layer": start, "prompt_chunk": prompt_chunk, "tokens": tokens,
               "weights": kind, "index_json_sha256": index_sha,
               "weights_identity_sha256": identity["identity_sha256"] if identity else None,
               "revision": identity.get("revision") if identity else None, "commit": _git_head(),
               "threads": os.environ.get("ORACLE_THREADS", "16"), "ok": not bad, "problems": bad}
        mp = os.path.join(out_dir, "manifest.json")
        if os.path.exists(mp):
            pl = ts.jload(mp).get("per_layer", [])
            peaks = [r["peak_wset_gib"] for r in pl if r.get("peak_wset_gib")]
            row["peak_wset_gib"] = max(peaks) if peaks else None
            row["load_s"] = round(sum(r["load_s"] for r in pl), 1)
            row["compute_s"] = round(sum(r["compute_s"] for r in pl), 1)
        with open(book, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
        if bad:
            log("%s: FAILED after %.0f s: %s" % (name, wall, "; ".join(bad)))
            return 1
        log("%s: pass done in %.0f s, routing verified" % (name, wall))
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--container", default=None, help="the full CNQ container (default %s)" % os.path.relpath(
        CONTAINER, ROOT))
    src.add_argument("--fp8", default=None, metavar="DIR",
                     help="#179: the FP8 originals as tools/fetch-glm.py verified them (e.g. %s)" % os.path.relpath(
                         FP8, ROOT))
    ap.add_argument("--corpus", default=CORPUS)
    ap.add_argument("--runs", default=RUNS)
    ap.add_argument("--only", default=None, help="comma list of corpus names (default: all five, amendment order)")
    ap.add_argument("--prompt-chunk", type=int, default=512)
    ap.add_argument("--python", default=sys.executable, help="the oracle venv's python (default: this one)")
    ap.add_argument("--dry-run", action="store_true", help="check everything and print the passes, run nothing")
    a = ap.parse_args(argv)
    names = [r[0] for r in AMENDMENT_1]
    if a.only:
        names = [n for n in names if n in a.only.split(",")]
        unknown = set(a.only.split(",")) - set(names)
        if unknown:
            print("refused: not files of amendment 1: %s" % ", ".join(sorted(unknown)), file=sys.stderr)
            return 2
    if a.prompt_chunk < 1:
        ap.error("--prompt-chunk must be >= 1")
    try:
        if a.fp8:
            identity = fp8_identity(a.fp8)
            index_sha, weights = identity["index_json_sha256"], ("fp8-originals", a.fp8)
            seen = "FP8 originals %s: %d shards verified by tools/fetch-glm.py at rev %s, index sha256 %s, " \
                   "identity %s" % (a.fp8, len(identity["shards"]), REVISION[:8], index_sha,
                                    identity["identity_sha256"])
        else:
            container = a.container or CONTAINER
            _, index_sha = container_index(container)
            identity, weights = container_identity(index_sha), ("container", container)
            seen = "container %s: complete, index sha256 %s" % (container, index_sha)
        ids_paths = check_corpus(a.corpus, names)
        for n in names:  # refuse up front, not after hours of earlier passes
            plan(os.path.join(a.runs, n), ts.jload(ids_paths[n]), index_sha, identity)
    except Refusal as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    print(seen)
    print("corpus %s: corpus.json and %d files match amendment 1" % (a.corpus, len(names)))
    return run_passes(names, ids_paths, weights, a.runs, index_sha, prompt_chunk=a.prompt_chunk,
                      python=a.python, dry_run=a.dry_run, identity=identity)


if __name__ == "__main__":
    sys.exit(main())
