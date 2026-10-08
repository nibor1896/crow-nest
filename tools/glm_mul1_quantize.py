#!/usr/bin/env python3
"""glm_mul1_quantize.py — crow-nest #182 (GLM-5.3-Flash plan steps 11/12): the routed experts of
GLM-5.3-Flash as MUL1 K=3 trellis records (exllamav3's quantizer), Hessians from the layerwise
runner's MoE inputs on calibration tokens. Three subcommands, each resumable:

  plan      any python -I (stdlib only): the calibration set, the disk verdict, and the exact commands of
            (a) the step-11 partial conversion of layers 0-3 and (b) the full conversion, with derived runtimes
  capture   .venv-oracle (CPU, transformers 5.16.1): the four calibration files of PREREG amendment 1 layer by
            layer through oracle/glm5_layerwise.py's run_layer, one layer load for all files; per MoE layer the
            expert input (post_attention_layernorm output) -> <work>/L<ll>/moe-in.bf16 [rows][hidden] + ids.i32
            [rows][8] + capture.json. Hand-over states <work>/states/<file>/l<k>-output.bf16 (derived) are
            deleted behind; at most --max-ahead captured layers wait for the quantizer.
  quantize  .venv-exl3 (GPU, exllamav3 1.6.0): per expert gate/up/down dequantized from FP8, H_gu = X^T X over
            the layer's rows (exllamav3's calibration_all_experts; --hessian-tokens routed: routed rows only),
            H_down = A^T A, A = silu(min(X Wg^T, 10)) * clamp(X Wu^T, -10, 10), then quantize_exl3 (mul1, K=3,
            sigma_reg 0.025, apply_out_scales auto) -> <store>/L<ll>/E<eee>.safetensors + a synced line in
            <store>/journal.jsonl. The MTP layer 45 has no forward in the runner: identity Hessian. A record the
            converter has journalled gets <store>/L<ll>/E<eee>.done; --prune-consumed <container> deletes the
            record files whose .done names that container.

The converter reads the store: converter --experts-mul1 <store> [--mul1-wait] (converter/README.md).
The FP8 originals are only read: a work or store directory inside them is refused, and nothing here
deletes a file outside <work>/states, <work>/L<ll>/ (captures) and <store>/L<ll>/ (consumed records).
Docs: docs/glm-mul1-conversion.md.
"""
import argparse
import hashlib
import json
import os
import shutil
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# ------------------------------------------------------------------ the model and the codec

HIDDEN, INTER, N_EXPERTS, TOP_K = 4096, 2048, 288, 8      # config.json rev eb9eb208 (docs/glm5-next-recipe.md 1, 10.2)
TEXT_LAYERS = 45                                          # num_hidden_layers; layer 45 is the MTP block
MOE_LAYERS = tuple(range(3, 45))                          # mlp_layer_types sparse 3..44
MTP_LAYER = 45
K = 3                                                     # MUL1 bitrate of every routed expert (#181)
EXPERT_ALIGN = 4096
REVISION = "eb9eb208eb0d988989d07a6a12d0fdeb5f52574a"
REPO = "zai-org/GLM-5.3-Flash"
LM = "model.language_model."

# PREREG amendment 1 (runs/glm53-flash/PREREG.md): the calibration files, never the held-out one
HELD = "todo-1006"
CAL_NAMES = ("omarchy-0915a", "lenis-0830", "ctx7-0830", "zetalab-0829")
TOKENS = 32768

STORE_FORMAT = "crow-nest mul1 store"
EXL3_COMMIT = "151539c7"
SIGMA_REG = 0.025

FP8 = os.path.join(ROOT, "models", "GLM-5.3-Flash-original")
CORPUS = os.path.join(ROOT, "decode_out", "glm-step8", "corpus")
WORK = os.path.join(ROOT, "decode_out", "glm-mul1", "work")
STORE = os.path.join(ROOT, "decode_out", "glm-mul1", "store")
OUT_FULL = os.path.join(ROOT, "converter", "GLM-5.3-Flash-MUL1K3.cnq")
OUT_PARTIAL = os.path.join(ROOT, "converter", "GLM-5.3-Flash-MUL1K3-L0-3.cnq")
RESERVE_GIB = 16

GIB = 1 << 30


class Refusal(Exception):
    """The step cannot start (or continue) on these inputs."""


def record_bytes(hidden=HIDDEN, inter=INTER, k=K):
    """one expert's record: 3 trellis of hidden/16 x inter/16 tiles x 16K u16 words, six fp16 scale vectors,
    zeros to 4096 (converter/src/mul1.rs RecordLayout)"""
    trellis = (hidden // 16) * (inter // 16) * 16 * k * 2
    payload = 3 * trellis + 3 * 2 * (hidden + inter)
    return -(-payload // EXPERT_ALIGN) * EXPERT_ALIGN


def parse_layers(spec):
    """'0-3', '3,7,45', 'all' or None (= every MoE layer and the MTP layer) -> sorted list"""
    if spec in (None, "", "all"):
        return list(MOE_LAYERS) + [MTP_LAYER]
    out = set()
    for part in spec.split(","):
        part = part.strip()
        try:
            if "-" in part:
                a, b = (int(x) for x in part.split("-", 1))
                if a > b:
                    raise Refusal("--layers %s: range %d-%d runs backwards" % (spec, a, b))
                out.update(range(a, b + 1))
            else:
                out.add(int(part))
        except ValueError:
            raise Refusal("--layers %s: `%s` is not a layer number" % (spec, part))
    bad = [l for l in out if not 0 <= l <= MTP_LAYER]
    if bad:
        raise Refusal("--layers %s: layer %d is outside 0..%d" % (spec, bad[0], MTP_LAYER))
    return sorted(out)


def expert_layers(layers):
    """the layers of `layers` that have routed experts (MoE 3..44 and the MTP block 45)"""
    return [l for l in layers if l in MOE_LAYERS or l == MTP_LAYER]


def rel_record(layer, expert):
    return "L%02d/E%03d.safetensors" % (layer, expert)


def check_calibration(names):
    if HELD in names:
        raise Refusal("%s is the held-out file of PREREG amendment 1; it never calibrates" % HELD)
    bad = [n for n in names if n not in CAL_NAMES]
    if bad:
        raise Refusal("%s is not a calibration file of PREREG amendment 1 (%s)" % (bad[0], ", ".join(CAL_NAMES)))


def _real(p):
    return os.path.normcase(os.path.realpath(os.path.abspath(p)))


def check_outside(fp8, *dirs):
    """robin's rule: the originals are never written or deleted - no work dir inside them"""
    f = _real(fp8)
    for d in dirs:
        r = _real(d)
        if r == f or r.startswith(f + os.sep):
            raise Refusal("%s lies inside the FP8 originals %s; derived files go elsewhere (the originals are never "
                          "written or deleted)" % (d, fp8))


def free_bytes(path):
    """free bytes on the volume of `path` (its first existing ancestor)"""
    p = os.path.abspath(path)
    while not os.path.exists(p):
        q = os.path.dirname(p)
        if q == p:
            break
        p = q
    return shutil.disk_usage(p).free


def disk_check(path, need, reserve, what, free=None):
    """Refusal unless the volume of `path` has `need` + `reserve` bytes free"""
    free = free_bytes(path) if free is None else free
    line = "disk check (%s): %.1f GiB to write + %.1f GiB reserve, %.1f GiB free on the volume of %s" % (
        what, need / GIB, reserve / GIB, free / GIB, path)
    if free < need + reserve:
        raise Refusal(line)
    return line + ": ok"


def sha256_file(path, chunk=1 << 22):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def write_json_atomic(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def jload(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ the store journal

def read_journal(store):
    """{(layer, expert): line} of <store>/journal.jsonl; a torn last line (a kill mid-write) is cut off"""
    p = os.path.join(store, "journal.jsonl")
    if not os.path.exists(p):
        return {}
    with open(p, "rb") as f:
        raw = f.read()
    keep = raw[:raw.rfind(b"\n") + 1]
    if len(keep) != len(raw):
        with open(p, "r+b") as f:
            f.truncate(len(keep))
    out = {}
    for line in keep.splitlines():
        try:
            v = json.loads(line)
            out[(int(v["layer"]), int(v["expert"]))] = v
        except (ValueError, KeyError, TypeError):
            continue
    return out


def append_journal(store, line):
    with open(os.path.join(store, "journal.jsonl"), "a", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(line, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def prune_consumed(store, container, log=print):
    """delete record files whose .done (written by the converter after journalling the record) names
    `container`; returns the bytes freed. Only E<eee>.safetensors under <store>/L<ll>/ are touched."""
    want = _real(container)
    freed = 0
    for d in sorted(os.listdir(store)) if os.path.isdir(store) else []:
        ld = os.path.join(store, d)
        if not (d.startswith("L") and d[1:].isdigit() and os.path.isdir(ld)):
            continue
        for n in sorted(os.listdir(ld)):
            if not (n.startswith("E") and n.endswith(".done")):
                continue
            rec = os.path.join(ld, n[:-len(".done")] + ".safetensors")
            if not os.path.exists(rec):
                continue
            try:
                out = jload(os.path.join(ld, n)).get("out")
            except (OSError, ValueError):
                continue
            if out is None or _real(out) != want:
                continue
            freed += os.path.getsize(rec)
            os.remove(rec)
    if freed:
        log("prune: %.2f GiB of records the container %s holds" % (freed / GIB, container))
    return freed


# ------------------------------------------------------------------ plan (stdlib only)

# derived runtime inputs (docs/glm-mul1-conversion.md, section "Runtime"):
PASS_FP8_S = 4650            # one 32,768-token pass through the runner on the FP8 originals (derived, glm5-reference-runner.md 8)
LOAD_MOE_S, LOAD_DENSE_S = 7.8, 2.0   # one layer load: FP8 MoE layer 3 measured once (step 6); dense from the container
QUANT_EXPERT_S = 1.04        # quantize_exl3 gate + up + down, measured 2026-10-09 (RTX 5090, n = 2 reps, warm)
EXPERT_EXTRA_S = 0.4         # FP8 read + dequant, H_down, file write and sha256 per expert (assumed)
CONVERT_DENSE_S = 250        # step 6: 244 s for 7.22 GB at --scales mse (measured once), the dense part is ~7.25 GB
WRITE_GBPS = 1.0             # converter record copy + two sha256 passes, GB/s (assumed)
NON_EXPERT_BYTES = 7_250_323_716        # 178,478,618,624 - 171,228,266,496 - 28,412 (converter plan, 2026-10-08)
PARTIAL_DENSE_BYTES = 3_143_000_000     # step 6 payload 7.22 GB minus 288 NVFP4 experts x 14,155,776 B (derived)


def estimate(layers_captured, expert_layers_n, dense_bytes):
    compute_per_layer = (PASS_FP8_S - (42 * LOAD_MOE_S + 3 * LOAD_DENSE_S)) / TEXT_LAYERS
    loads = sum(LOAD_MOE_S if l >= 3 else LOAD_DENSE_S for l in layers_captured)
    capture = loads + len(layers_captured) * len(CAL_NAMES) * compute_per_layer
    quant = expert_layers_n * N_EXPERTS * (QUANT_EXPERT_S + EXPERT_EXTRA_S)
    recs = expert_layers_n * N_EXPERTS * record_bytes()
    convert = CONVERT_DENSE_S * dense_bytes / NON_EXPERT_BYTES + recs / (WRITE_GBPS * 1e9)
    return {"capture_s": capture, "quantize_s": quant, "convert_s": convert, "container_bytes": dense_bytes + recs,
            "records": expert_layers_n * N_EXPERTS}


def commands(python_oracle, python_exl3, converter, fp8, work, store, out, partial):
    rel = lambda p: os.path.relpath(p, ROOT).replace(os.sep, "/") if os.path.isabs(p) else p
    lay = ["--layers", "0-3"] if partial else []
    cap = [rel(python_oracle), "-I", "tools/glm_mul1_quantize.py", "capture", "--fp8", rel(fp8), "--work", rel(work)] + lay
    qua = [rel(python_exl3), "-I", "tools/glm_mul1_quantize.py", "quantize", "--fp8", rel(fp8), "--work", rel(work),
           "--store", rel(store)] + lay
    con = [rel(converter), "--scales", "mse", "--source-repo", REPO, "--revision", REVISION, "--experts-mul1", rel(store)]
    if partial:
        con += ["--layers", "0-3", "--with-embed-head"]
    else:
        qua += ["--wait", "--prune-consumed", rel(out)]
        con += ["--mul1-wait"]
    con += [rel(fp8), rel(out)]
    return [" ".join(c) for c in (cap, qua, con)]


def plan_text(fp8=FP8, work=WORK, store=STORE, out_full=OUT_FULL, out_partial=OUT_PARTIAL, reserve_gib=RESERVE_GIB,
              free=None):
    py_o = ".venv-oracle/Scripts/python.exe" if os.name == "nt" else ".venv-oracle/bin/python"
    py_e = ".venv-exl3/Scripts/python.exe" if os.name == "nt" else ".venv-exl3/bin/python"
    conv = "converter/target/release/converter" + (".exe" if os.name == "nt" else "")
    free = free_bytes(out_full) if free is None else free
    reserve = reserve_gib * GIB
    states = 2 * len(CAL_NAMES) * TOKENS * 4 * HIDDEN * 2
    cap_layer = len(CAL_NAMES) * TOKENS * (HIDDEN * 2 + TOP_K * 4)
    rows = []
    p = estimate(range(0, 4), 1, PARTIAL_DENSE_BYTES)
    f = estimate(range(0, TEXT_LAYERS), len(MOE_LAYERS) + 1, NON_EXPERT_BYTES)
    need_p = p["container_bytes"] + states + cap_layer + p["records"] * record_bytes()
    need_f = f["container_bytes"] + states + 2 * cap_layer + 2 * N_EXPERTS * record_bytes()
    L = rows.append
    L("glm_mul1_quantize plan (crow-nest #182) - nothing runs; every runtime below is DERIVED, not measured")
    L("calibration: PREREG amendment 1 files %s, %d x %d = %d tokens (exllamav3 default 250 x 2048 = 512,000); "
      "held out: %s" % (", ".join(CAL_NAMES), len(CAL_NAMES), TOKENS, len(CAL_NAMES) * TOKENS, HELD))
    L("record: MUL1 K=%d, %d B per expert; experts: 42 MoE layers + MTP layer 45 = 43 x %d = %d"
      % (K, record_bytes(), N_EXPERTS, 43 * N_EXPERTS))
    for name, est, need, out, partial in (("(a) step 11, partial container, layers 0-3", p, need_p, out_partial, True),
                                          ("(b) full container", f, need_f, out_full, False)):
        verdict = "ok" if free >= need + reserve else "REFUSED (free < need + reserve)"
        L("")
        L(name)
        L("  container %.1f GiB (%d records); peak disk %.1f GiB (container + %.1f GiB hand-over states + captures + "
          "record backlog) + reserve %d GiB vs %.1f GiB free -> %s" % (est["container_bytes"] / GIB, est["records"],
                                                                      need / GIB, states / GIB, reserve_gib, free / GIB, verdict))
        wall = (max(est["capture_s"], est["quantize_s"]) + est["convert_s"] / 4) if not partial else \
            (est["capture_s"] + est["quantize_s"] + est["convert_s"])
        L("  runtime (derived): capture %.1f h, quantize %.1f h, convert %.2f h; wall %.1f h %s" % (
            est["capture_s"] / 3600, est["quantize_s"] / 3600, est["convert_s"] / 3600, wall / 3600,
            "(one after the other)" if partial else "(the three run side by side; quantize trails capture by <= 2 layers)"))
        cmds = commands(py_o, py_e, conv, fp8, work, store, out, partial)
        for i, c in enumerate(cmds, 1):
            L("  %d. %s" % (i, c))
    return "\n".join(rows)


# ------------------------------------------------------------------ capture (.venv-oracle)

def capture_layers(ws, tc, ids_by_name, work, layers, prompt_chunk=512, state_dtype="bf16", max_ahead=2,
                   identity=None, log=print, stop_after=None, poll=30.0):
    """Advance every calibration file through layers 0..max(layers) (the MTP block excluded) and dump the
    MoE input of each MoE layer in `layers`. Resumable: <work>/states/progress.json names the last layer
    whose states are on disk for every file. Returns the list of layers captured in this call."""
    import gc
    import numpy as np
    import torch
    sys.path.insert(0, os.path.join(ROOT, "oracle"))
    import glm5_common as G
    import glm5_layerwise as LW
    from transformers.models.glm5_next.modeling_glm5_next import Glm5NextTextDecoderLayer

    names = list(ids_by_name)
    H, hc = tc.hidden_size, tc.hc_mult
    run = [l for l in layers if l < tc.num_hidden_layers]
    if not run:
        return []
    last = max(run)
    sdir = os.path.join(work, "states")
    os.makedirs(sdir, exist_ok=True)
    prog_p = os.path.join(sdir, "progress.json")
    prog = jload(prog_p) if os.path.exists(prog_p) else {"layer_done": -1, "files": names}
    if prog["files"] != names:
        raise Refusal("%s: states of files %s, this run calibrates on %s" % (prog_p, prog["files"], names))
    ext = "bf16" if state_dtype == "bf16" else "f32"
    done_cap = []

    def captured_waiting():
        n = 0
        for l in MOE_LAYERS:
            d = os.path.join(work, "L%02d" % l)
            if os.path.exists(os.path.join(d, "capture.json")) and not os.path.exists(os.path.join(d, "quantized.json")):
                n += 1
        return n

    for l in range(prog["layer_done"] + 1, last + 1):
        cap = l in layers and tc.mlp_layer_types[l] == "sparse"
        ldir = os.path.join(work, "L%02d" % l)
        if cap and os.path.exists(os.path.join(ldir, "capture.json")):
            cap = False  # captured by an earlier run whose states went on (a resume after a crash in between)
        said = False
        while cap and max_ahead and captured_waiting() >= max_ahead:
            if not said:
                log("capture: %d captured layers wait for the quantizer (--max-ahead %d)" % (max_ahead, max_ahead))
                said = True
            time.sleep(poll)
        t0 = time.time()
        layer = G.build_meta(Glm5NextTextDecoderLayer, tc, l)
        ws.load(layer, "%slayers.%d." % (LM, l))
        rows = []
        hook = layer.mlp.register_forward_pre_hook(lambda m, a: rows.append(a[0][0].detach().to(torch.bfloat16).clone())) \
            if cap else None
        ids_all = []
        try:
            for name in names:
                N = len(ids_by_name[name])
                if l == 0:
                    x = ws.embed(ids_by_name[name]).unsqueeze(1).expand(-1, hc, -1).contiguous()
                else:
                    x = LW.read_state(os.path.join(sdir, name, "l%d-output.%s" % (l - 1, ext)), (N, hc, H))
                y, routing, _ = LW.run_layer(layer, tc, l, x, N, prompt_chunk)
                os.makedirs(os.path.join(sdir, name), exist_ok=True)
                LW.write_state(os.path.join(sdir, name), l, y, state_dtype)
                if cap:
                    ids_all.append(routing[0])
                del x, y
        finally:
            if hook is not None:
                hook.remove()
        del layer
        gc.collect()
        if cap:
            os.makedirs(ldir, exist_ok=True)
            X = torch.cat(rows, 0)
            ids = torch.cat(ids_all, 0).to(torch.int32)
            assert X.shape == (sum(len(v) for v in ids_by_name.values()), H), tuple(X.shape)
            X.view(torch.int16).numpy().tofile(os.path.join(ldir, "moe-in.bf16"))
            ids.numpy().tofile(os.path.join(ldir, "ids.i32"))
            write_json_atomic(os.path.join(ldir, "capture.json"), {
                "layer": l, "rows": int(X.shape[0]), "hidden": H, "top_k": int(ids.shape[1]),
                "files": [{"name": n, "rows": len(ids_by_name[n])} for n in names],
                "moe_in_sha256": sha256_file(os.path.join(ldir, "moe-in.bf16")),
                "ids_sha256": sha256_file(os.path.join(ldir, "ids.i32")), "identity": identity,
                "what": "post_attention_layernorm output = the routed experts' input (forward pre-hook on layer.mlp), "
                        "BF16; ids = the router's top-k, ascending (glm5_layerwise.run_layer)"})
            done_cap.append(l)
            del X, rows
        prog["layer_done"] = l
        write_json_atomic(prog_p, prog)
        if l > 0:
            for name in names:  # derived hand-over states behind the one just written
                p = os.path.join(sdir, name, "l%d-output.%s" % (l - 1, ext))
                if os.path.exists(p):
                    os.remove(p)
        log("capture layer %2d %s %6.1f s" % (l, "MoE input dumped" if cap else "advanced", time.time() - t0))
        if stop_after is not None and l >= stop_after:
            break
    return done_cap


# ------------------------------------------------------------------ quantize (.venv-exl3)

def quant_args(layer, expert, proj):
    return {"K": K, "mul1": True, "seed": (layer * N_EXPERTS + expert) * 3 + proj, "devices": [0],
            "apply_out_scales": None, "sigma_reg": SIGMA_REG}


def store_head(hidden, inter, n_experts, calibration, source, hessian_tokens):
    import torch
    try:
        from exllamav3.version import __version__ as exl3_version
    except ImportError:
        exl3_version = None
    return {"format": STORE_FORMAT, "version": 1, "k": K, "hidden": hidden, "inter": inter, "n_experts": n_experts,
            "record_bytes": record_bytes(hidden, inter),
            "quantizer": {"name": "exllamav3 quantize_exl3", "version": exl3_version, "commit": EXL3_COMMIT,
                          "torch": torch.__version__, "codebook": "mul1", "sigma_reg": SIGMA_REG,
                          "apply_out_scales": "auto (None)", "seed": "(layer * n_experts + expert) * 3 + {0 gate, 1 up, 2 down}",
                          "hessian_tokens": hessian_tokens, "mtp_layer_45": "identity Hessian (no forward in the runner, recipe O1)",
                          "h_down": "A^T A, A = silu(min(X Wg^T, 10)) * clamp(X Wu^T, -10, 10) with the FP8-dequantized weights"},
            "calibration": calibration, "source": source}


def open_store(store, head):
    os.makedirs(store, exist_ok=True)
    p = os.path.join(store, "store.json")
    if os.path.exists(p):
        old = jload(p)
        if old != head:
            diff = sorted(k for k in set(old) | set(head) if old.get(k) != head.get(k))
            raise Refusal("%s was written by another quantizer setup (differs in %s); use another --store" % (p, ", ".join(diff)))
    else:
        write_json_atomic(p, head)


def _hdata(H, count, device):
    import torch
    return {"H": H, "first_key": "glm-mul1", "count": count, "finalized": False, "num_total": count * H.shape[0],
            "inf_nan": torch.zeros(2, dtype=torch.long, device=device), "device": device}


def gram(X, chunk=16384):
    """X^T X in f32, X [rows][k] (any dtype) on the GPU"""
    import torch
    k = X.shape[1]
    H = torch.zeros(k, k, dtype=torch.float32, device=X.device)
    for r0 in range(0, X.shape[0], chunk):
        x = X[r0:r0 + chunk].float()
        H.addmm_(x.T, x)
    return H


def down_input_gram(X, Wg, Wu, limit=10.0, chunk=16384):
    """A^T A for A = silu(min(X Wg^T, limit)) * clamp(X Wu^T, -limit, limit) (recipe section 10)"""
    import torch
    n = Wg.shape[0]
    H = torch.zeros(n, n, dtype=torch.float32, device=X.device)
    for r0 in range(0, X.shape[0], chunk):
        x = X[r0:r0 + chunk].float()
        g = torch.clamp(x @ Wg.T, max=limit)
        u = torch.clamp(x @ Wu.T, min=-limit, max=limit)
        a = torch.nn.functional.silu(g) * u
        H.addmm_(a.T, a)
    return H


def quantize_expert(ws, layer, expert, X, H_gu, count, device):
    """gate, up, down of one expert through quantize_exl3 -> ({name: tensor}, meta). X: the Hessian rows
    (None for an identity Hessian); H_gu: X^T X (shared by gate and up)."""
    import torch
    from exllamav3.modules.quant.exl3_lib.quantize import quantize_exl3
    W = [ws.get("%slayers.%d.mlp.experts.%d.%s_proj.weight" % (LM, layer, expert, p)).to(device)
         for p in ("gate", "up", "down")]                       # torch [out, in], f32
    hidden, inter = W[0].shape[1], W[0].shape[0]
    if X is None:
        h_gu, h_d = _hdata(torch.eye(hidden, device=device), 1, device), _hdata(torch.eye(inter, device=device), 1, device)
    else:
        h_gu = _hdata(H_gu.clone(), count, device)
        h_d = _hdata(down_input_gram(X, W[0], W[1]), X.shape[0], device)
    tensors, proxy, fallback = {}, [], []
    for pi, (p, w, h) in enumerate(zip(("gate", "up", "down"), W, (h_gu, h_gu, h_d))):
        qa = quant_args(layer, expert, pi)
        _, perr, out = quantize_exl3(w.T.contiguous(), h, qa, False)
        tr, suh, svh = out["trellis"], out["suh"], out["svh"]
        k, n = w.shape[1], w.shape[0]
        assert tr.dtype == torch.int16 and tuple(tr.shape) == (k // 16, n // 16, 16 * K), (p, tuple(tr.shape))
        assert suh.dtype == torch.half and suh.numel() == k and svh.dtype == torch.half and svh.numel() == n
        tensors[p + ".trellis"] = tr.contiguous().cpu()
        tensors[p + ".suh"] = suh.contiguous().cpu()
        tensors[p + ".svh"] = svh.contiguous().cpu()
        proxy.append(float(perr))
        fallback.append(bool(qa.get("q_fallback")))
    return tensors, {"proxy_err": proxy, "q_fallback": fallback}


def write_record_file(path, tensors, meta):
    from safetensors.torch import save_file
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    save_file(tensors, tmp, metadata={k: json.dumps(v) for k, v in meta.items()})
    with open(tmp, "rb+") as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)


def quantize_layers(ws, work, store, layers, hessian_tokens="all", wait=False, keep_capture=False, prune_for=None,
                    reserve=RESERVE_GIB * GIB, device="cuda:0", log=print, poll=30.0, free=None, stop_after=None):
    """quantize every expert of the expert layers in `layers` not yet in the store journal. Returns the
    number of experts written in this call."""
    import numpy as np
    import torch
    torch.backends.cuda.matmul.allow_tf32 = False
    head = jload(os.path.join(store, "store.json"))
    hidden, inter, n_exp = head["hidden"], head["inter"], head["n_experts"]
    rec = record_bytes(hidden, inter)
    journal = read_journal(store)
    written = 0
    for l in expert_layers(layers):
        todo = [e for e in range(n_exp) if (l, e) not in journal]
        ldir = os.path.join(work, "L%02d" % l)
        if not todo:
            continue
        X = H_gu = ids = None
        if l != MTP_LAYER:
            cj = os.path.join(ldir, "capture.json")
            said = False
            while not os.path.exists(cj):
                if not wait:
                    raise Refusal("%s: layer %d is not captured; run `capture` first (or pass --wait)" % (cj, l))
                if not said:
                    log("quantize: waiting for %s" % cj)
                    said = True
                time.sleep(poll)
            cap = jload(cj)
            if cap["hidden"] != hidden:
                raise Refusal("%s: hidden %s, the store is for %d" % (cj, cap["hidden"], hidden))
            for f, key in (("moe-in.bf16", "moe_in_sha256"), ("ids.i32", "ids_sha256")):
                if sha256_file(os.path.join(ldir, f)) != cap[key]:
                    raise Refusal("%s: sha256 differs from %s" % (os.path.join(ldir, f), cj))
            a = np.fromfile(os.path.join(ldir, "moe-in.bf16"), dtype="<i2").reshape(cap["rows"], hidden)
            X = torch.from_numpy(a).view(torch.bfloat16).to(device)
            ids = torch.from_numpy(np.fromfile(os.path.join(ldir, "ids.i32"), dtype="<i4").reshape(cap["rows"], -1)).to(device)
            if hessian_tokens == "all":
                H_gu = gram(X)
        t_layer = time.time()
        for i, e in enumerate(todo):
            if prune_for and i % 16 == 0:
                prune_consumed(store, prune_for, log)
            fr = free_bytes(store) if free is None else free
            if fr < rec + reserve:
                raise Refusal("disk check (quantize): %.1f GiB free at %s, a record needs %.4f GiB above the %.1f GiB "
                              "reserve - stopped before layer %d expert %d (resumable)" % (fr / GIB, store, rec / GIB,
                                                                                            reserve / GIB, l, e))
            t0 = time.time()
            if X is None:
                tensors, meta = quantize_expert(ws, l, e, None, None, 0, device)
                rows, hkind = 0, "identity"
            elif hessian_tokens == "routed":
                Xe = X[(ids == e).any(dim=1)]
                tensors, meta = quantize_expert(ws, l, e, Xe, gram(Xe), Xe.shape[0], device)
                rows, hkind = int(Xe.shape[0]), "routed rows"
            else:
                tensors, meta = quantize_expert(ws, l, e, X, H_gu, X.shape[0], device)
                rows, hkind = int(X.shape[0]), "all rows"
            path = os.path.join(store, rel_record(l, e))
            write_record_file(path, tensors, dict(meta, layer=l, expert=e, rows=rows, hessian=hkind))
            line = dict(meta, layer=l, expert=e, file=rel_record(l, e), sha256=sha256_file(path), rows=rows,
                        hessian=hkind, seconds=round(time.time() - t0, 3))
            append_journal(store, line)
            journal[(l, e)] = line
            written += 1
            if i % 16 == 0 or i == len(todo) - 1:
                el = time.time() - t_layer
                log("quantize layer %2d expert %3d (%d/%d of this layer) proxy %s, %.2f s/expert" % (
                    l, e, i + 1, len(todo), ["%.5f" % x for x in meta["proxy_err"]], el / (i + 1)))
            if stop_after is not None and written >= stop_after:
                return written
        write_json_atomic(os.path.join(ldir, "quantized.json") if l != MTP_LAYER else os.path.join(store, "L45.quantized.json"),
                          {"layer": l, "experts": n_exp, "store": os.path.abspath(store)})
        if X is not None and not keep_capture:
            del X, H_gu
            for f in ("moe-in.bf16", "ids.i32"):  # derived: the layer's records are in the store now
                p = os.path.join(ldir, f)
                if os.path.exists(p):
                    os.remove(p)
        torch.cuda.empty_cache()
    if prune_for:
        prune_consumed(store, prune_for, log)
    return written


# ------------------------------------------------------------------ CLI

def _fp8_checks(fp8):
    sys.path.insert(0, HERE)
    import glm_route_passes as R
    try:
        return R.fp8_identity(fp8)
    except R.Refusal as e:
        raise Refusal(str(e))


def _corpus(corpus, names):
    sys.path.insert(0, HERE)
    import glm_route_passes as R
    import glm_tier_sim as ts
    try:
        paths = R.check_corpus(corpus, names)
    except R.Refusal as e:
        raise Refusal(str(e))
    return {n: ts.jload(paths[n]) for n in names}, {n: ts.sha256_file(paths[n]) for n in names}


def cmd_plan(a):
    print(plan_text(a.fp8, a.work, a.store, a.out, a.out_partial, a.reserve_gib))
    return 0


def cmd_capture(a):
    layers = parse_layers(a.layers)
    names = a.calib.split(",") if a.calib else list(CAL_NAMES)
    check_calibration(names)
    check_outside(a.fp8, a.work)
    ident = _fp8_checks(a.fp8)
    ids, shas = _corpus(a.corpus, names)
    states = 2 * len(names) * TOKENS * 4 * HIDDEN * (2 if a.state_dtype == "bf16" else 4)
    cap_layer = len(names) * TOKENS * (HIDDEN * 2 + TOP_K * 4)
    print(disk_check(a.work, states + max(a.max_ahead, 1) * cap_layer, a.reserve_gib * GIB, "capture"))
    os.makedirs(a.work, exist_ok=True)
    cj = os.path.join(a.work, "calibration.json")
    calib = {"files": names, "ids_file_sha256": shas, "tokens": sum(len(v) for v in ids.values()), "held_out": HELD,
             "corpus": "PREREG amendment 1 (runs/glm53-flash/PREREG.md)", "fp8_identity_sha256": ident["identity_sha256"],
             "revision": REVISION}
    if os.path.exists(cj) and jload(cj) != calib:
        raise Refusal("%s: this work dir holds a capture of other files or weights; use another --work" % cj)
    write_json_atomic(cj, calib)
    if a.dry_run:
        print("dry run: %d files, %d tokens, layers %s - nothing ran" % (len(names), calib["tokens"], layers))
        return 0
    sys.path.insert(0, os.path.join(ROOT, "oracle"))
    import glm5_common as G
    ws = G.WeightSource("fp8", a.fp8)
    tc = G.text_config_from_dict(ws.config_dict())  # eager attention and experts, as the runner's `run`
    capture_layers(ws, tc, ids, a.work, layers, a.prompt_chunk, a.state_dtype, a.max_ahead, ident["identity_sha256"])
    return 0


def cmd_quantize(a):
    layers = parse_layers(a.layers)
    check_outside(a.fp8, a.work, a.store)
    ident = _fp8_checks(a.fp8)
    cj = os.path.join(a.work, "calibration.json")
    if not os.path.exists(cj):
        raise Refusal("%s: no capture in this work dir; run `capture` first" % cj)
    calib = jload(cj)
    if calib["fp8_identity_sha256"] != ident["identity_sha256"]:
        raise Refusal("%s: captured from other weights" % cj)
    todo = len(expert_layers(layers)) * N_EXPERTS
    need = (2 * N_EXPERTS if a.prune_consumed else todo) * record_bytes()
    print(disk_check(a.store, need, a.reserve_gib * GIB, "quantize"))
    head = store_head(HIDDEN, INTER, N_EXPERTS, calib, {"repo": REPO, "revision": REVISION,
                                                        "fp8_identity_sha256": ident["identity_sha256"]}, a.hessian_tokens)
    open_store(a.store, head)
    if a.dry_run:
        print("dry run: store %s ready, layers %s - nothing ran" % (a.store, expert_layers(layers)))
        return 0
    sys.path.insert(0, os.path.join(ROOT, "oracle"))
    import glm5_common as G
    quantize_layers(G.WeightSource("fp8", a.fp8), a.work, a.store, layers, a.hessian_tokens, a.wait, a.keep_capture,
                    a.prune_consumed, a.reserve_gib * GIB)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--fp8", default=FP8)
    common.add_argument("--work", default=WORK)
    common.add_argument("--reserve-gib", type=int, default=RESERVE_GIB)
    p = sub.add_parser("plan", parents=[common])
    p.add_argument("--store", default=STORE)
    p.add_argument("--out", default=OUT_FULL)
    p.add_argument("--out-partial", default=OUT_PARTIAL)
    c = sub.add_parser("capture", parents=[common])
    c.add_argument("--layers", default=None, help="layers to capture (0-3, all); the states advance through every layer before")
    c.add_argument("--corpus", default=CORPUS)
    c.add_argument("--calib", default=None, help="comma list of amendment-1 calibration files (default: all four)")
    c.add_argument("--prompt-chunk", type=int, default=512)
    c.add_argument("--state-dtype", choices=("bf16", "f32"), default="bf16")
    c.add_argument("--max-ahead", type=int, default=2, help="captured layers that may wait for the quantizer (0 = no limit)")
    c.add_argument("--dry-run", action="store_true")
    q = sub.add_parser("quantize", parents=[common])
    q.add_argument("--store", default=STORE)
    q.add_argument("--layers", default=None)
    q.add_argument("--hessian-tokens", choices=("all", "routed"), default="all")
    q.add_argument("--wait", action="store_true", help="wait for layers the capture has not dumped yet")
    q.add_argument("--keep-capture", action="store_true", help="keep a layer's MoE-input dump after its records")
    q.add_argument("--prune-consumed", default=None, metavar="CONTAINER",
                   help="delete record files the converter has journalled into CONTAINER (their .done names it)")
    q.add_argument("--dry-run", action="store_true")
    a = ap.parse_args(argv)
    try:
        return {"plan": cmd_plan, "capture": cmd_capture, "quantize": cmd_quantize}[a.cmd](a)
    except Refusal as e:
        print("refused: %s" % e, file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
