#!/usr/bin/env python3
"""#147 (GLM-5.3-Flash step 8): the routing passes of PREREG amendment 1, one file after another,
through the layerwise runner (oracle/glm5_layerwise.py, #158) on the full CNQ container.

  .venv-oracle/Scripts/python.exe -I tools/glm_route_passes.py [--dry-run]
      [--container converter/GLM-5.3-Flash-CNQ4.5.cnq] [--corpus decode_out/glm-step8/corpus]
      [--runs decode_out/glm-step8/runs] [--only <name>,...] [--prompt-chunk 512]

Before anything runs (exit 2 with the reason otherwise):
  - the container is complete: no `<cnq>.journal.jsonl` beside it (the converter deletes it only after
    the index trailer is written, converter/src/main.rs), magic CNQ1, an index trailer that parses
    (index v2, recipe cnq4.5-glm5-next, not partial, source rev eb9eb208), every text layer 0..L-1 and
    the embedding, final norm and lm_head in the index;
  - corpus.json and each file's ids and mask have the sha256 of amendment 1 (pinned below, checked
    against runs/glm53-flash/PREREG.md by the tests), and each ids file holds the routed 32,768 tokens.
Per file, in the amendment's order, the runner pass of the amendment's dump plan plus the prompt-chunk
and delete-behind options (#147):
  oracle/glm5_layerwise.py run --weights container <cnq> --ids <corpus>/<name>-ids.json --anchors 32767
      --state-dtype bf16 --prompt-chunk 512 --delete-states-behind --out <runs>/<name>
Resumable per file: a pass that is complete over every layer is skipped; an interrupted one continues
with `--layers k+1:` from the last recorded state l<k>-output.* (the runner records layer k before it
deletes the state behind it); one with no state to continue from starts again at layer 0. A run dir made
with other weights or other ids is refused, never overwritten.
After each pass: the self-test of `tools/glm_tier_sim.py sim` over the out dir (routing sha256 ==
manifest, [N][8], ids 0..287 ascending) and its source check (CNQ, not partial, complete). One line per
pass, also a failed one, goes to <runs>/passes.jsonl (the measurement-book row).
"""
import argparse
import datetime
import hashlib
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

RUNNER = os.path.join(ROOT, "oracle", "glm5_layerwise.py")
CONTAINER = os.path.join(ROOT, "converter", "GLM-5.3-Flash-CNQ4.5.cnq")
CORPUS = os.path.join(ROOT, "decode_out", "glm-step8", "corpus")
RUNS = os.path.join(ROOT, "decode_out", "glm-step8", "runs")
TOKENS = 32768
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
RECIPE = "cnq4.5-glm5-next"
LM = "model.language_model."

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

def plan(out_dir, ids, index_sha):
    """('done', None) | ('resume', k) | ('fresh', None); Refusal for a dir of other weights or ids"""
    mp = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(mp):
        return "fresh", None
    man = ts.jload(mp)
    got = (man.get("weights") or {}).get("index_json_sha256")
    if index_sha is not None and got != index_sha:
        raise Refusal("%s: made with other weights (index sha256 %s, this container %s); move it away" % (
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


def verify(out_dir, shape=ts.SHAPE, of_record=True):
    """the sim's own self-test of the routing files (+ its source check when of_record) -> [problems]"""
    try:
        man, _ = ts.load_runner(out_dir, shape)
    except ts.SimError as e:
        return [str(e)]
    return ts.source_reasons(man) if of_record else []


def _git_head():
    try:
        return subprocess.run(["git", "-C", ROOT, "rev-parse", "--short", "HEAD"], capture_output=True,
                              text=True, timeout=10).stdout.strip() or None
    except OSError:
        return None


def run_passes(names, ids_paths, weights, runs_dir, index_sha, tokens=TOKENS, prompt_chunk=512,
               python=sys.executable, runner=RUNNER, shape=ts.SHAPE, of_record=True, log=print, dry_run=False,
               state_dtype="bf16"):
    """one pass per name, in order; returns 0 when every pass is done and verified, 1 at the first failure"""
    os.makedirs(runs_dir, exist_ok=True)
    book = os.path.join(runs_dir, "passes.jsonl")
    for name in names:
        out_dir = os.path.join(runs_dir, name)
        ids = ts.jload(ids_paths[name])
        what, start = plan(out_dir, ids, index_sha)
        if what == "done":
            bad = verify(out_dir, shape, of_record)
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
        bad = verify(out_dir, shape, of_record) if rc == 0 else ["runner rc %d (see %s)" % (
            rc, os.path.join(out_dir, "runner.log"))]
        row = {"name": name, "start_utc": s0.isoformat(timespec="seconds"), "wall_s": round(wall, 1),
               "rc": rc, "resumed_from_layer": start, "prompt_chunk": prompt_chunk, "tokens": tokens,
               "weights": weights[0], "index_json_sha256": index_sha, "commit": _git_head(),
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
    ap.add_argument("--container", default=CONTAINER)
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
        _, index_sha = container_index(a.container)
        ids_paths = check_corpus(a.corpus, names)
        for n in names:  # refuse up front, not after hours of earlier passes
            plan(os.path.join(a.runs, n), ts.jload(ids_paths[n]), index_sha)
    except Refusal as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2
    print("container %s: complete, index sha256 %s" % (a.container, index_sha))
    print("corpus %s: corpus.json and %d files match amendment 1" % (a.corpus, len(names)))
    return run_passes(names, ids_paths, ("container", a.container), a.runs, index_sha,
                      prompt_chunk=a.prompt_chunk, python=a.python, dry_run=a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
