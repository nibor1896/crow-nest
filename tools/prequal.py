#!/usr/bin/env python3
"""tools/prequal.py - crow-nest #240: pre-qualification of a model without a download and without a GPU.

    python -I tools/prequal.py (--repo ORG/NAME --revision SHA | --model-dir DIR) --context N
        [--rates rates.json] [--band LO,HI] [--m LO[,HI]] [--free-ram-gib N] [--commit-gib X] [--vram-mib N]
        [--kv-dtype bf16|fp8] [--generic] [--work DIR] [--converter PATH] [--states PATH] [--json out.json]

From the shard headers and config.json only it forecasts, per candidate (CNQ4.5; for a MoE also MUL1 K=3
experts + CNQ4.5 dense): the container size, the KV cache at the context, the placement over VRAM / pinned host
RAM / NVMe, and the decode ceiling per stage. Docs: docs/prequal.md.

- A family the converter has a recipe for (model_type strings parsed from converter/src/recipe.rs) goes through
  `converter plan --headers` and, for glm5_next, `states --plan`; their numbers are taken unchanged. Any other
  family, or --generic, goes through the generic path below, printed as `estimate: no recipe`.
- Every engine constant is parsed at run time from its source file (a regex plus a whitelisted integer evaluator,
  never eval) and carries its path:line; a constant that is not found exits 3, never a silent default.
- Stdlib only (+ curl through tools/hf_headers.py for --repo). Run with -I: it parses downloaded JSON.

Exit codes: 0 forecast printed; 2 refused input (reason named); 3 a tool it calls failed or a source constant
was not found (named). This is a forecast, not a measurement and not a gate.
"""
import argparse
import ast
import datetime
import importlib.util
import inspect
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile

TOOLS = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(TOOLS)   # the repo: engine/ and converter/ sources, default binaries, models/
EXE = ".exe" if os.name == "nt" else ""
GIB = 1 << 30
MIB = 1 << 20

NVFP4_BLOCK_VALUES, NVFP4_BLOCK_BYTES = 64, 36   # DtypeOut::bytes, converter/src/recipe.rs: n.div_ceil(64) * 36
EXPERT_ALIGN = 4096                              # a routed-expert record starts on a 4096-B sector
MUL1_K = 3
MUL1_FALLBACK_BPW = 3.05
CNQ = "CNQ4.5"
MUL1 = "MUL1 K=3 experts + CNQ4.5 dense"
NOT_COMPUTED = "path A/B, engine gap, external best, licence, disk (runbook, by hand)"
UNIFORM_NOTE = "uniform routing; hot sets need a model run (tools/glm_tier_sim.py, tools/hotset-eval.py)"
M_NOTE = "m given, not simulated; hot sets need a model run (tools/glm_tier_sim.py, tools/hotset-eval.py)"
PREFILL_NOTE = ("uncalibrated: the GLM bound did not hold, 1,137.5 tok/s measured above it "
                "(runs/glm53-flash/quick/README.md:38)")

# safetensors dtypes (huggingface/safetensors README "Format"; sub-byte F4/F6 as one entry each) -> bytes per value
DTYPE_BYTES = {"BOOL": 1, "U8": 1, "I8": 1, "F8_E5M2": 1, "F8_E4M3": 1, "F8_E8M0": 1, "I16": 2, "U16": 2,
               "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "F64": 8, "I64": 8, "U64": 8,
               "F4": 0.5, "F6_E2M3": 0.75, "F6_E3M2": 0.75}
INT_DTYPES = {"BOOL", "U8", "I8", "I16", "U16", "I32", "U32", "I64", "U64"}
SCALE_SUFFIXES = ("weight_scale_inv", "weight_scale", "input_scale")
VISION_SEGMENTS = {"visual", "vision", "vision_tower", "vision_model", "vision_encoder", "vision_language_adapter",
                   "multi_modal_projector", "mm_projector", "patch_merger", "image_newline", "audio", "audio_tower",
                   "audio_model", "audio_encoder"}
EMBED_SEGMENTS = {"embed_tokens", "tok_embeddings", "word_embeddings", "wte"}
ROUTER_MARKS = ("mlp.gate.weight", "router", "block_sparse_moe.gate", "shared_expert_gate")
EXPERT_RE = re.compile(r"\.experts\.(\d+)\.")
LAYER_RE = re.compile(r"(?:^|\.)layers\.(\d+)\.")
EXPERT_COUNT_KEYS = ("n_routed_experts", "num_experts", "num_local_experts")
TOPK_KEYS = ("num_experts_per_tok", "num_experts_per_token", "moe_topk", "top_k")
RATE_KEYS = ("B_nvme_gbps", "R_pcie_gbps", "R_dram_gbps", "R_vram_gbps")


class Refusal(Exception):
    """exit 2: the input is refused, the reason is the message"""


class ToolFailure(Exception):
    """exit 3: a tool prequal calls failed, or a source constant was not found"""


def _load(name, filename):
    mod = sys.modules.get(name)
    if mod is None:
        spec = importlib.util.spec_from_file_location(name, os.path.join(TOOLS, filename))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        sys.modules[name] = mod
    return mod


stage_ceiling = _load("stage_ceiling", "stage_ceiling.py")


def load_header_helper():
    """tools/hf_headers.py (#240 step 1): fetch_model(repo, revision, out_dir, curl=None, log=None), HeaderError"""
    if not os.path.isfile(os.path.join(TOOLS, "hf_headers.py")):
        raise ToolFailure("tools/hf_headers.py not found: --repo needs the header helper of #240 step 1 "
                          "(use --model-dir with headers read another way)")
    return _load("hf_headers", "hf_headers.py")


HELPER_LOADER = load_header_helper   # the tests put a fake helper here; nothing else replaces it


def mul1_module():
    return _load("glm_mul1_quantize", "glm_mul1_quantize.py")


# ------------------------------------------------------------------ the integer evaluator and the source constants

_BINOPS = {ast.Add: lambda a, b: a + b, ast.Sub: lambda a, b: a - b, ast.Mult: lambda a, b: a * b,
           ast.LShift: lambda a, b: a << b, ast.RShift: lambda a, b: a >> b}
_RUST_SUFFIX = re.compile(r"\b(\d[\d_]*)(?:u8|u16|u32|u64|u128|usize|i8|i16|i32|i64|i128|isize)\b")


def eval_int(expr, names=None):
    """A Rust constant expression of whole numbers -> int: int literals (with `_`), `<< >> * / + -` and
    parentheses, `/` as u64 integer division. A name is taken from `names` only; any other node (a call, an
    attribute, a float, a power, a unary minus) is refused with ValueError. Never eval."""
    names = names or {}
    src = _RUST_SUFFIX.sub(r"\1", expr.strip())
    try:
        tree = ast.parse(src, mode="eval")
    except SyntaxError as e:
        raise ValueError("%r is not an integer expression (%s)" % (expr, e.msg)) from None

    def ev(n):
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) is int:
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in _BINOPS:
            return _BINOPS[type(n.op)](ev(n.left), ev(n.right))
        if isinstance(n, ast.BinOp) and isinstance(n.op, (ast.Div, ast.FloorDiv)):
            a, b = ev(n.left), ev(n.right)
            if b == 0:
                raise ValueError("%r divides by zero" % expr)
            return a // b
        if isinstance(n, ast.Name) and n.id in names:
            return names[n.id]
        raise ValueError("%r: %s is not allowed in a constant expression (int literals, << >> * / + -, parentheses)"
                         % (expr, ast.dump(n) if not isinstance(n, ast.Name) else "the name " + n.id))

    v = ev(tree)
    if v < 0:
        raise ValueError("%r is negative (u64)" % expr)
    return v


_TEXTS = {}


def _source(root, rel):
    path = os.path.join(root, rel)
    if path not in _TEXTS:
        try:
            with open(path, encoding="utf-8") as f:
                _TEXTS[path] = f.read()
        except OSError as e:
            raise ToolFailure("%s: cannot read the source file (%s)" % (rel, e.strerror or e)) from None
    return _TEXTS[path]


def _line(text, pos):
    return text.count("\n", 0, pos) + 1


def source_const(root, rel, name, depth=0):
    """`[pub] const NAME: T = <expr>;` in root/rel -> {"value", "source": "rel:line", "expr"}; a name inside the
    expression is resolved the same way in the same file."""
    text = _source(root, rel)
    m = re.search(r"^[ \t]*(?:pub(?:\([^)]*\))?[ \t]+)?const[ \t]+%s[ \t]*:[ \t]*[\w:<>]+[ \t]*=[ \t]*([^;]+);"
                  % re.escape(name), text, re.M)
    if not m:
        raise ToolFailure("source constant %s not found in %s (prequal parses it; did it move?)" % (name, rel))
    return {"value": _eval_source(root, rel, m.group(1), depth, name), "source": "%s:%d" % (rel, _line(text, m.start())),
            "expr": " ".join(m.group(1).split())}


def _eval_source(root, rel, expr, depth, what):
    try:
        tree = ast.parse(_RUST_SUFFIX.sub(r"\1", expr.strip()), mode="eval")
    except SyntaxError:
        raise ToolFailure("%s in %s: %r is not an integer expression" % (what, rel, expr)) from None
    names = {}
    for n in ast.walk(tree):
        if isinstance(n, ast.Name) and depth < 4:
            names[n.id] = source_const(root, rel, n.id, depth + 1)["value"]
    try:
        return eval_int(expr, names)
    except ValueError as e:
        raise ToolFailure("%s in %s: %s" % (what, rel, e)) from None


def _split_top(body):
    out, depth, cur = [], 0, ""
    for ch in body:
        if ch in "([{":
            depth += 1
        elif ch in ")]}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return [p.strip() for p in out if p.strip()]


STABILITY_FIELDS = ("vram_headroom", "vram_cap", "decode_stage_rows")


def source_stability(root, rel, name):
    """`pub const NAME: Stability = Stability { vram_headroom: .., vram_cap: None|Some(..), decode_stage_rows: .. };`"""
    text = _source(root, rel)
    m = re.search(r"^[ \t]*pub[ \t]+const[ \t]+%s[ \t]*:[ \t]*Stability[ \t]*=[ \t]*Stability[ \t]*\{(.*?)\}[ \t]*;"
                  % re.escape(name), text, re.M | re.S)
    if not m:
        raise ToolFailure("source constant Stability::%s not found in %s" % (name, rel))
    fields = {}
    for part in _split_top(m.group(1)):
        k, _, v = part.partition(":")
        k, v = k.strip(), v.strip()
        if v == "None":
            fields[k] = None
        elif v.startswith("Some(") and v.endswith(")"):
            fields[k] = _eval_source(root, rel, v[5:-1], 0, "Stability::%s.%s" % (name, k))
        else:
            fields[k] = _eval_source(root, rel, v, 0, "Stability::%s.%s" % (name, k))
    miss = [f for f in STABILITY_FIELDS if f not in fields]
    if miss:
        raise ToolFailure("Stability::%s in %s: field %s not found" % (name, rel, ", ".join(miss)))
    fields["source"] = "%s:%d" % (rel, _line(text, m.start()))
    return fields


def _fn_body(text, signature_re, what, rel):
    m = re.search(signature_re, text)
    if not m:
        raise ToolFailure("%s not found in %s (prequal parses it; did it move?)" % (what, rel))
    end = text.find("\n    }\n", m.end())
    return text[m.end():end if end > 0 else len(text)], _line(text, m.start())


def converter_families(root):
    """{model_type: {"family", "recipe"}} from converter/src/recipe.rs `Family::model_type` / `Family::recipe`"""
    rel = "converter/src/recipe.rs"
    text = _source(root, rel)
    mt, mt_line = _fn_body(text, r"pub fn model_type\(self\)[^{]*\{", "Family::model_type", rel)
    rc, _ = _fn_body(text, r"pub fn recipe\(self\)[^{]*\{", "Family::recipe", rel)
    types = re.findall(r"Family::(\w+)\s*=>\s*\"([^\"]+)\"", mt)
    recipes = dict(re.findall(r"Family::(\w+)\s*=>\s*\"([^\"]+)\"", rc))
    if not types:
        raise ToolFailure("%s: Family::model_type has no `Family::X => \"...\"` arm" % rel)
    src = "%s:%d (Family::model_type)" % (rel, mt_line)
    return {t: {"family": f, "recipe": recipes.get(f), "source": src} for f, t in types}


def engine_constants(root):
    """every engine constant prequal uses, parsed from its source: {name: {"value", "source"}}"""
    geo, man, gen, st = "engine/src/geo.rs", "engine/src/manager.rs", "engine/src/gen.rs", "engine/src/bin/states.rs"
    c = {}
    for name, rel in (("HOST_PINNED_CAP", geo), ("CHUNK_CAP", geo), ("CARD_MIB_OF_RECORD", st), ("SAFETY", man),
                      ("LAUNCH_SLACK", man), ("RAM_MARGIN_DEFAULT_GB", man), ("PF_TG", gen)):
        c[name] = source_const(root, rel, name)
    text = _source(root, gen)
    m = re.search(r"env_parse\(\"CROW_PF_ASYNC\"\)\.unwrap_or\((\d+)\)", text)
    if not m:
        raise ToolFailure("the CROW_PF_ASYNC default (pf_async_mode) not found in %s" % gen)
    c["PF_ASYNC_DEFAULT"] = {"value": int(m.group(1)), "source": "%s:%d" % (gen, _line(text, m.start())),
                             "expr": "pf_async_on() = pf_async_mode() >= 1: two prefill group sets"}
    stab = {n: source_stability(root, geo, n) for n in ("OF_RECORD", "GLM5_NEXT")}
    body, line = _fn_body(_source(root, geo), r"pub fn of\(family: Family\) -> Stability \{", "Stability::of", geo)
    of = {}
    for fams, s in re.findall(r"((?:Family::\w+\s*\|\s*)*Family::\w+)\s*=>\s*Stability::(\w+)", body):
        for f in re.findall(r"Family::(\w+)", fams):
            of[f] = s
    if not of:
        raise ToolFailure("%s: Stability::of has no `Family::X => Stability::Y` arm" % geo)
    c["stability"] = stab
    c["stability_of"] = {"map": of, "source": "%s:%d (Stability::of)" % (geo, line)}
    return c


def stability_for(consts, family):
    """the family's Stability; an unknown family takes GLM5_NEXT, the only row with a measured cap (31.9 GiB, G4)"""
    name = consts["stability_of"]["map"].get(family) if family else None
    why = "Stability::of(%s), %s" % (family, consts["stability_of"]["source"]) if name else \
        "unknown family: GLM5_NEXT, the only row with the measured 31.9 GiB cap (G4)"
    name = name or "GLM5_NEXT"
    s = dict(consts["stability"][name])
    s["name"], s["why"] = name, why
    return s


# ------------------------------------------------------------------ the model input

def _jload(path, what):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except OSError as e:
        raise Refusal("%s: cannot read %s (%s)" % (path, what, e.strerror or e)) from None
    except ValueError as e:
        raise Refusal("%s: %s is not JSON (%s)" % (path, what, e)) from None


def read_model_dir(d, headers_dir=None):
    """the model as the header helper lays it out: config.json, generation_config.json (optional), the index or a
    single-file header, hf-revision.json (optional), headers/<shard>.json. Read only."""
    d = os.path.abspath(d)
    if not os.path.isdir(d):
        raise Refusal("model dir %s: not a directory" % d)
    hdir = os.path.abspath(headers_dir or os.path.join(d, "headers"))
    config = _jload(os.path.join(d, "config.json"), "config.json")
    if not isinstance(config, dict):
        raise Refusal("%s: config.json is not a JSON object" % d)
    index_path = os.path.join(d, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        idx = _jload(index_path, "the safetensors index")
        wm = idx.get("weight_map") if isinstance(idx, dict) else None
        if not isinstance(wm, dict) or not wm:
            raise Refusal("%s: no weight_map" % index_path)
        shards, layout = sorted(set(wm.values())), "index"
    else:
        singles = sorted(f[:-5] for f in os.listdir(hdir) if f.endswith(".safetensors.json")) \
            if os.path.isdir(hdir) else []
        if len(singles) != 1:
            raise Refusal("%s: no model.safetensors.index.json and no single-file header (%s/<file>.safetensors.json, "
                          "found %d)" % (d, hdir, len(singles)))
        shards, layout = singles, "single"
    tensors = []
    for s in shards:
        hp = os.path.join(hdir, s + ".json")
        if not os.path.isfile(hp):
            raise Refusal("%s: the header of shard %s is missing (%s)" % (d, s, hp))
        h = _jload(hp, "a shard header")
        if not isinstance(h, dict) or not isinstance(h.get("header"), dict):
            raise Refusal("%s: no header object" % hp)
        if h.get("shard") not in (None, s):
            raise Refusal("%s: the header is for %s, not %s" % (hp, h.get("shard"), s))
        for name, info in h["header"].items():
            if name == "__metadata__":
                continue
            dt = info.get("dtype") if isinstance(info, dict) else None
            if dt not in DTYPE_BYTES:
                raise Refusal("%s: tensor %s has dtype %r, not a safetensors dtype this tool knows (%s)"
                              % (hp, name, dt, ", ".join(sorted(DTYPE_BYTES))))
            shape, off = info.get("shape"), info.get("data_offsets")
            if not isinstance(shape, list) or not all(isinstance(x, int) and x >= 0 for x in shape) or \
                    not isinstance(off, list) or len(off) != 2:
                raise Refusal("%s: tensor %s: shape %r / data_offsets %r" % (hp, name, shape, off))
            tensors.append((name, dt, tuple(shape), off[1] - off[0], s))
    rev = None
    rp = os.path.join(d, "hf-revision.json")
    if os.path.isfile(rp):
        r = _jload(rp, "hf-revision.json")
        if isinstance(r, dict):
            rev = {"repo": r.get("id") or r.get("modelId"), "revision": r.get("sha")}
    tc = config.get("text_config") if isinstance(config.get("text_config"), dict) else config
    if tc is not config and tc.get("model_type"):
        mt, mt_key = tc.get("model_type"), "text_config.model_type"
    else:
        mt, mt_key = config.get("model_type"), "model_type"
    return {"dir": d, "headers_dir": hdir, "config": config, "tc": tc, "model_type": mt, "mt_key": mt_key, "layout": layout,
            "shards": shards, "tensors": tensors, "hf": rev,
            "generation_config": os.path.isfile(os.path.join(d, "generation_config.json"))}


def _int(tc, keys):
    for k in keys:
        v = tc.get(k)
        if isinstance(v, int) and not isinstance(v, bool) and v > 0:
            return v, k
    return None, None


def _prod(shape):
    n = 1
    for x in shape:
        n *= x
    return n


def nvfp4_bytes(n):
    return -(-n // NVFP4_BLOCK_VALUES) * NVFP4_BLOCK_BYTES


def keep_bytes(n, dt):
    """a keep: BF16 (2 B), an F32 source carried as f32 (4 B), integer metadata carried raw"""
    if dt == "F32" or dt == "F64":
        return n * 4
    if dt in INT_DTYPES:
        return int(n * DTYPE_BYTES[dt])
    return n * 2


def _segments(name):
    return name.split(".")


def classify(tensors, tc):
    """the generic recipe (keep rules of converter/src/recipe.rs decide_flash_next :224, decide_qwen35_dense :272,
    decide_glm5_next :420, omitted :175) over every header tensor -> sizes and the routed-expert geometry"""
    L, _ = _int(tc, ("num_hidden_layers",))
    E_cfg, e_key = _int(tc, EXPERT_COUNT_KEYS)
    hidden, _ = _int(tc, ("hidden_size",))
    r = {"quant": 0, "keep": 0, "embed": 0, "scales": [0, 0], "omitted": {}, "empty": 0}
    per_expert, shapes, fused_values = {}, {}, {}
    omit_v, omit_m = "vision / audio tower or projector (not written)", "MTP / NextN layers (not written)"
    for name, dt, shape, nb, _shard in tensors:
        n = _prod(shape)
        if n == 0:
            r["empty"] += 1
            continue
        seg = _segments(name)
        lm = LAYER_RE.search(name)
        layer = int(lm.group(1)) if lm else None
        if VISION_SEGMENTS.intersection(seg):
            o = r["omitted"].setdefault(omit_v, [0, 0])
            o[0], o[1] = o[0] + 1, o[1] + nb
            continue
        if "mtp" in seg or any("nextn" in s for s in seg) or (layer is not None and L is not None and layer >= L):
            o = r["omitted"].setdefault(omit_m, [0, 0])
            o[0], o[1] = o[0] + 1, o[1] + nb
            continue
        if name.endswith(SCALE_SUFFIXES):
            r["scales"][0] += 1
            r["scales"][1] += nb
            continue
        if EMBED_SEGMENTS.intersection(seg):
            r["embed"] += keep_bytes(n, dt)
            continue
        em = EXPERT_RE.search(name)
        if em and layer is not None:
            b = nvfp4_bytes(n) if len(shape) >= 2 and n % NVFP4_BLOCK_VALUES == 0 else keep_bytes(n, dt)
            key = (layer, int(em.group(1)))
            per_expert[key] = per_expert.get(key, 0) + b
            if len(shape) == 2:
                shapes.setdefault(key, []).append(shape)
            continue
        if ".experts." in name and len(shape) == 3 and E_cfg and shape[0] == E_cfg and layer is not None:
            v = n // E_cfg
            b = nvfp4_bytes(v) if v % NVFP4_BLOCK_VALUES == 0 else keep_bytes(v, dt)
            for e in range(E_cfg):
                per_expert[(layer, e)] = per_expert.get((layer, e), 0) + b
            fused_values[layer] = fused_values.get(layer, 0) + v
            continue
        is_keep = (len(shape) <= 1 or n % NVFP4_BLOCK_VALUES != 0 or "norm" in name or "lm_head" in name
                   or seg[0] == "output" or any(k in name for k in ROUTER_MARKS))
        if is_keep:
            r["keep"] += keep_bytes(n, dt)
        else:
            r["quant"] += nvfp4_bytes(n)
    routed = sum(per_expert.values())
    moe_layers = sorted({l for l, _ in per_expert})
    E = E_cfg or (max((sum(1 for k in per_expert if k[0] == l) for l in moe_layers), default=0))
    r.update({
        "routed": routed, "records": len(per_expert), "moe_layers": moe_layers, "experts": E,
        "experts_key": e_key, "text_nvfp4": r["quant"] + sum(per_expert.values()),
        "subtotal": r["quant"] + r["keep"] + routed, "dense": r["quant"] + r["keep"],
        "payload": r["quant"] + r["keep"] + routed + r["embed"],
        "mul1": mul1_records(per_expert, shapes, fused_values, hidden) if per_expert else None,
    })
    return r


def mul1_records(per_expert, shapes, fused_values, hidden):
    """one MUL1 K=3 record per routed expert per layer: record_bytes(hidden, inter, 3) of
    tools/glm_mul1_quantize.py (the Python twin of converter/src/mul1.rs RecordLayout) when the expert is a
    gate/up/down triple whose two dims are multiples of 128; else 3.05 bpw of its values, rounded up to the
    4096-B record alignment (an approximation, said so in the output)."""
    mq = mul1_module()
    line = inspect.getsourcelines(mq.record_bytes)[1]
    exact, fallback, sizes = 0, 0, {}

    def geometry(key):
        if key[0] in fused_values:
            v = fused_values[key[0]]
            if hidden and v % (3 * hidden) == 0:
                return hidden, v // (3 * hidden), 3 * v
            return None, None, v
        sh = shapes.get(key, [])
        v = sum(a * b for a, b in sh)
        if len(sh) == 3:
            for x in set(sh):
                y = (x[1], x[0])
                if sh.count(x) == 2 and sh.count(y) == 1 and (hidden is None or x[1] == hidden):
                    return x[1], x[0], v
        return None, None, v

    total = 0
    for key in per_expert:
        h, i, v = geometry(key)
        if h and i and h % 128 == 0 and i % 128 == 0:
            b = mq.record_bytes(h, i, MUL1_K)
            exact += 1
            sizes[b] = sizes.get(b, 0) + 1
        else:
            b = -(-math.ceil(v * MUL1_FALLBACK_BPW / 8) // EXPERT_ALIGN) * EXPERT_ALIGN
            fallback += 1
            sizes[b] = sizes.get(b, 0) + 1
        total += b
    record = max(sizes, key=lambda k: (sizes[k], k))
    how = ("record_bytes(hidden, inter, %d), tools/glm_mul1_quantize.py:%d" % (MUL1_K, line)) if not fallback else \
        ("%d records by record_bytes (tools/glm_mul1_quantize.py:%d), %d at %.2f bpw rounded to %d B: not a "
         "gate/up/down triple of multiples of 128" % (exact, line, fallback, MUL1_FALLBACK_BPW, EXPERT_ALIGN))
    return {"record": record, "total": total, "records": len(per_expert), "exact": exact, "fallback": fallback,
            "how": how, "distinct": sorted(sizes)}


# ------------------------------------------------------------------ KV cache

def _layer_kinds(tc, L):
    lt = tc.get("layer_types")
    if isinstance(lt, list) and len(lt) >= L and all(isinstance(s, str) for s in lt[:L]):
        kinds = []
        for s in lt[:L]:
            s = s.lower()
            if any(k in s for k in ("linear", "mamba", "recurrent", "ssm", "conv")):
                kinds.append("linear")
            elif any(k in s for k in ("sliding", "local", "chunked")):
                kinds.append("sliding")
            else:
                kinds.append("full")
        return kinds, "layer_types"
    k, _ = _int(tc, ("full_attention_interval",))
    if k:
        return ["full" if (i + 1) % k == 0 else "linear" for i in range(L)], "full_attention_interval"
    win, _ = _int(tc, ("sliding_window",))
    if win and tc.get("use_sliding_window") is not False:
        p, _ = _int(tc, ("sliding_window_pattern",))
        if p:
            return ["full" if (i + 1) % p == 0 else "sliding" for i in range(L)], "sliding_window_pattern"
        return ["sliding"] * L, "sliding_window (every layer)"
    return ["full"] * L, "no layer_types: every layer full attention"


def kv_cache(tc, ctx, kv_dtype):
    """the generic KV cache at ctx tokens from config.json (text_config if nested)"""
    eb = {"bf16": 2, "fp8": 1}[kv_dtype]
    L, _ = _int(tc, ("num_hidden_layers",))
    if not L:
        return {"at_ctx": None, "per_token": None, "lines": [], "keys": [],
                "not_derived": ["no num_hidden_layers in config.json"]}
    kinds, kinds_src = _layer_kinds(tc, L)
    n_full, n_slide, n_lin = kinds.count("full"), kinds.count("sliding"), kinds.count("linear")
    keys, lines, nd = ["num_hidden_layers", kinds_src], [], []
    lora, _ = _int(tc, ("kv_lora_rank",))
    if lora:
        rope = tc.get("qk_rope_head_dim") or 0
        per = (lora + rope) * eb
        form = "MLA (kv_lora_rank %d + qk_rope_head_dim %d) x %d B" % (lora, rope, eb)
        keys += ["kv_lora_rank", "qk_rope_head_dim"]
    else:
        heads, _ = _int(tc, ("num_attention_heads",))
        kvh, kk = _int(tc, ("num_key_value_heads", "num_attention_heads"))
        hd, hk = _int(tc, ("head_dim",))
        hidden, _ = _int(tc, ("hidden_size",))
        if not hd and hidden and heads:
            hd, hk = hidden // heads, "hidden_size / num_attention_heads"
        if not (kvh and hd):
            return {"at_ctx": None, "per_token": None, "lines": [], "keys": keys,
                    "not_derived": ["attention KV: num_key_value_heads / head_dim not in config.json"]}
        per = 2 * kvh * hd * eb
        form = "GQA/MHA 2 x %d kv heads (%s) x head_dim %d (%s) x %d B" % (kvh, kk, hd, hk, eb)
        keys += [kk, hk]
    win, wk = _int(tc, ("sliding_window", "attention_chunk_size"))
    full_b = n_full * per * ctx
    slide_rows = min(ctx, win) if win else ctx
    slide_b = n_slide * per * slide_rows
    c_ = lambda n: format(n, ",")  # noqa: E731
    lines.append("%d full-attention layers x %s x %s tokens = %s B" % (n_full, form, c_(ctx), c_(full_b)))
    if n_slide:
        lines.append("%d sliding layers x %d B x min(ctx, %s %s) = %s B" % (n_slide, per, wk or "no window key",
                                                                            c_(win or ctx), c_(slide_b)))
        keys.append(wk or "sliding_window")
    lin_b = 0
    if n_lin:
        lac = tc.get("linear_attn_config") if isinstance(tc.get("linear_attn_config"), dict) else None
        vh, _ = _int(tc, ("linear_num_value_heads",))
        kh, _ = _int(tc, ("linear_num_key_heads",))
        kd, _ = _int(tc, ("linear_key_head_dim",))
        vd, _ = _int(tc, ("linear_value_head_dim",))
        kc, _ = _int(tc, ("linear_conv_kernel_dim",))
        if vh and kh and kd and vd and kc:
            s, conv = vh * kd * vd * 4, (2 * kh * kd + vh * vd) * (kc - 1) * 4
            lin_b = n_lin * (s + conv)
            lines.append("%d linear layers x (GDN state %d x %d x %d f32 = %s B + conv (2 x %d x %d + %d x %d) x %d f32 = %s B)"
                         " = %s B, constant (engine StateSizes::plan)" % (n_lin, vh, kd, vd, c_(s), kh, kd, vh, vd, kc - 1,
                                                                           c_(conv), c_(lin_b)))
            keys += ["linear_num_value_heads", "linear_num_key_heads", "linear_key_head_dim", "linear_value_head_dim",
                     "linear_conv_kernel_dim"]
        elif lac and all(isinstance(lac.get(k), int) for k in ("num_heads", "head_dim", "short_conv_kernel_size")):
            h, d_, k_ = lac["num_heads"], lac["head_dim"], lac["short_conv_kernel_size"]
            s, conv = h * d_ * d_ * 4, 3 * h * d_ * (k_ - 1) * 4
            lin_b = n_lin * (s + conv)
            lines.append("%d linear layers x (KDA state %d x %d x %d f32 = %s B + conv 3 x %d x %d x %d f32 = %s B) = %s B,"
                         " constant (engine Glm5Geo)" % (n_lin, h, d_, d_, c_(s), h, d_, k_ - 1, c_(conv), c_(lin_b)))
            keys += ["linear_attn_config.num_heads", "linear_attn_config.head_dim",
                     "linear_attn_config.short_conv_kernel_size"]
        else:
            nd.append("%d linear / recurrent layers: state not derived (no GDN linear_* keys, no linear_attn_config)" % n_lin)
    if any(k in tc for k in ("index_head_dim", "index_n_heads")):
        nd.append("a sparse-attention indexer cache (index_head_dim) is not in the generic formula")
    total = full_b + slide_b + lin_b
    return {"at_ctx": total, "attention": full_b + slide_b, "linear": lin_b, "per_token": (n_full + n_slide) * per,
            "layers": {"full": n_full, "sliding": n_slide, "linear": n_lin}, "lines": lines,
            "keys": sorted(set(keys)), "not_derived": nd, "dtype": kv_dtype}


# ------------------------------------------------------------------ converter plan and states --plan

PLAN_LABELS = (("GPU, text weights NVFP4", "text_nvfp4"), ("GPU, text keeps (bf16/f32/i64)", "text_keeps"),
               ("GPU, text subtotal", "subtotal"), ("host RAM, token embedding", "embed"),
               ("routed experts (tiered)", "routed"), ("dense resident part", "dense"),
               ("optional section mtp", "mtp"), ("optional section vit", "vit"),
               ("section ple (Flash-Next tier)", "ple"), ("container payload", "payload"),
               ("container without index trailer", "container"))
PLAN_REQUIRED = ("subtotal", "embed", "payload", "container")


def parse_plan_totals(text):
    """the totals block of `converter plan` (converter/src/main.rs, `totals` ... `container without index trailer`)"""
    i = text.find("\ntotals ")
    if i < 0 and not text.startswith("totals "):
        raise ToolFailure("converter plan: no totals block in its output")
    block = text[i + 1:] if i >= 0 else text
    out = {"omitted": []}
    for label, key in PLAN_LABELS:
        m = re.search(r"^  %s\s+(\d+) B\b" % re.escape(label), block, re.M)
        if m:
            out[key] = int(m.group(1))
    for m in re.finditer(r"^  omitted, not written\s+(\d+) tensors,\s+(\d+) B source.*?\((.*)\)\s*$", block, re.M):
        out["omitted"].append((m.group(3), int(m.group(1)), int(m.group(2))))
    miss = [k for k in PLAN_REQUIRED if k not in out]
    if miss:
        raise ToolFailure("converter plan: totals line(s) %s not found" % ", ".join(miss))
    return out


STATES_LINES = (("hot", r"^\s*VRAM\s+N\s*=\s*(\d+)\s"), ("pinned", r"^\s*pinned\s+P\s*=\s*(\d+)\s"),
                ("nvme", r"^\s*NVMe\s+(\d+)\s+[\d.]+ %"), ("unit", r"unit = (\d+) B"),
                ("unit_layers", r"unit = \d+ B = (\d+) layers"), ("ceiling", r"^\s*plan ceiling\s+(\d+) B"),
                ("room", r"^\s*room for experts\s+(\d+) B"), ("kv_per_token", r"^\s*KV per token\s+(\d+) B"),
                ("latent", r"^\s*MLA latent\s+(\d+) B"), ("indexer", r"^\s*indexer cache\s+(\d+) B"),
                ("kda", r"^\s*KDA state \+ conv\s+(\d+) B"), ("staging", r"^\s*cold staging\s+(\d+) B"),
                ("slots", r"^\s*cold staging\s+\d+ B\s+[\d.]+ GiB\s+(\d+) slots"),
                ("card", r"^\s*card VRAM\s+(\d+) B"), ("pinned_budget", r"^\s*pinned budget\s+(\d+) B"),
                ("record", r"^\s*expert block\s+(\d+) B"), ("kv_b", r"^\s*kv_b at BF16\s+(\d+) B"),
                ("dense", r"^\s*dense part\s+(\d+) B"), ("context", r"plan \(#159\): context (\d+)"),
                ("experts", r"MoE layers x experts\s+(\d+) x (\d+)"))
STATES_REQUIRED = ("hot", "pinned", "nvme", "unit", "ceiling", "room")


def parse_states_table(text):
    """`states --plan`'s table (engine/src/manager.rs glm5_plan_table) -> its numbers"""
    out = {}
    for key, rx in STATES_LINES:
        m = re.search(rx, text, re.M)
        if m and key == "experts":
            out["moe_layers"], out["experts"] = int(m.group(1)), int(m.group(2))
        elif m:
            out[key] = int(m.group(1))
    miss = [k for k in STATES_REQUIRED if k not in out]
    if miss:
        raise ToolFailure("states --plan: table line(s) %s not found in its output" % ", ".join(miss))
    out["kv_total"] = sum(out.get(k, 0) for k in ("latent", "indexer", "kda"))
    return out


def _run(argv, env=None):
    try:
        p = subprocess.run(argv, capture_output=True, env=env)
    except OSError as e:
        raise ToolFailure("%s: cannot run (%s)" % (argv[0], e.strerror or e)) from None
    return p.returncode, p.stdout.decode("utf-8", "replace"), p.stderr.decode("utf-8", "replace")


def run_converter_plan(converter, model):
    """`converter plan --headers <headers> <model dir>`: its totals, unchanged. A single-file model is staged in a
    temporary dir (config files + a one-shard index), never written into the model dir."""
    with tempfile.TemporaryDirectory(prefix="prequal-") as tmp:
        target = model["dir"]
        if model["layout"] == "single":
            target = os.path.join(tmp, "model")
            os.makedirs(target)
            for f in ("config.json", "generation_config.json", "hf-revision.json"):
                if os.path.isfile(os.path.join(model["dir"], f)):
                    shutil.copyfile(os.path.join(model["dir"], f), os.path.join(target, f))
            wm = {t[0]: model["shards"][0] for t in model["tensors"]}
            with open(os.path.join(target, "model.safetensors.index.json"), "w", encoding="utf-8") as f:
                json.dump({"metadata": {}, "weight_map": wm}, f)
        rc, out, err = _run([converter, "plan", "--headers", model["headers_dir"], target])
    if rc != 0:
        raise ToolFailure("converter plan refused the %s family (exit %d): %s" % (model["model_type"], rc, err.strip() or out.strip()))
    return parse_plan_totals(out)


def run_states_plan(states, model_dir, ctx, record, codec, pinned_gib, vram_mib):
    """`states --plan --expert-bytes <record> --expert-codec <codec>` -> (table, None) or (None, refusal). The
    engine defaults are planned: CROW_PF_TG / CROW_PF_ASYNC are not passed on, CROW_PINNED_BUDGET_GB only when
    --free-ram-gib is given."""
    env = {k: v for k, v in os.environ.items() if k not in ("CROW_PINNED_BUDGET_GB", "CROW_PF_TG", "CROW_PF_ASYNC")}
    env["CROW_MODEL_DIR"], env["CROW_CONTEXT"] = model_dir, str(ctx)
    if pinned_gib is not None:
        env["CROW_PINNED_BUDGET_GB"] = str(pinned_gib)
    argv = [states, "--plan", "--expert-bytes", str(record), "--expert-codec", codec]
    if vram_mib:
        argv += ["--vram-mib", str(vram_mib)]
    rc, out, err = _run(argv, env)
    if rc != 0:
        return None, (err.strip() or out.strip() or "exit %d" % rc)
    return parse_states_table(out), None


# ------------------------------------------------------------------ placement and ceilings

def stage_slots(stab, topk, consts):
    """Stability::stage_slots (engine/src/geo.rs): decode rows x topk and PF_TG (x 2 with CROW_PF_ASYNC >= 1)
    held apart, or 2 x topk shared with the prefill set (OF_RECORD)"""
    prefill = consts["PF_TG"]["value"] * (2 if consts["PF_ASYNC_DEFAULT"]["value"] >= 1 else 1)
    if stab["decode_stage_rows"] is None:
        return max(2 * topk, prefill)
    return stab["decode_stage_rows"] * topk + prefill


def generic_split(*, consts, stab, card, dense, kv, moe, record=0, moe_layers=0, experts=0, topk=0, pinned_budget):
    """the generic three-tier split (the arithmetic of engine/src/manager.rs plan_three_tiers, without the GLM-only
    terms: kv_b decode, indexer cache)"""
    ceiling = min(card - stab["vram_headroom"], stab["vram_cap"] if stab["vram_cap"] is not None else card)
    parts = [("dense resident part" if moe else "GPU text subtotal", dense), ("KV cache", kv),
             ("SAFETY", consts["SAFETY"]["value"]), ("LAUNCH_SLACK", consts["LAUNCH_SLACK"]["value"])]
    slots = 0
    if moe:
        slots = stage_slots(stab, topk, consts)
        parts.append(("staging %d slots x record" % slots, slots * record))
    fixed = sum(b for _, b in parts)
    p = {"ceiling": ceiling, "fixed": fixed, "parts": parts, "slots": slots, "fits": fixed <= ceiling,
         "how": "generic split"}
    if not moe:
        p.update({"vram_used": fixed, "pinned_used": 0})
        return p
    unit = moe_layers * record
    if fixed > ceiling:
        p.update({"unit": unit, "hot": 0, "pinned": 0, "nvme": experts, "vram_used": fixed, "pinned_used": 0})
        return p
    hot = min(experts, (ceiling - fixed) // unit)
    pinned = min(experts - hot, pinned_budget // unit)
    p.update({"unit": unit, "hot": hot, "pinned": pinned, "nvme": experts - hot - pinned,
              "vram_used": fixed + hot * unit, "pinned_used": pinned * unit, "room": ceiling - fixed})
    return p


def decode_rows(*, moe, dense, kv, rates, m_values=None, placement=None, record=0, moe_layers=0, topk=0, experts=0):
    """per-stage bytes per token -> stage_ceiling.ceilings; one row for uniform routing or one per given m"""
    rv, rp, rn, rd = (rates.get(k) for k in ("R_vram_gbps", "R_pcie_gbps", "B_nvme_gbps", "R_dram_gbps"))
    if not moe:
        return [{"label": "all in VRAM", "m": None, "ceil": stage_ceiling.ceilings([("vram", dense + kv, rv)])}]
    visits = moe_layers * topk
    n, p, nv = placement["hot"], placement["pinned"], placement["nvme"]
    rows = []
    if m_values:
        shares = []
        for m in m_values:
            rest = 1.0 - m
            sv, sp = (rest * n / (n + p), rest * p / (n + p)) if n + p else (0.0, 0.0)
            shares.append(("m = %g given, rest split N:P = %d:%d" % (m, n, p), m, sv, sp))
    else:
        shares = [("uniform routing, m = NVMe/E = %d/%d" % (nv, experts), nv / experts, n / experts, p / experts)]
    for label, sn, sv, sp in shares:
        vram_b = dense + kv + visits * sv * record
        pcie_b = visits * sp * record
        nvme_b = visits * sn * record
        rows.append({"label": label, "m": sn, "visits": visits, "ceil": stage_ceiling.ceilings(
            [("vram", vram_b, rv), ("pcie", pcie_b, rp), ("nvme", nvme_b, rn), ("dram", pcie_b + nvme_b, rd)])})
    return rows


# ------------------------------------------------------------------ formatting

def fb(b):
    return "%s B (%.2f GiB)" % (format(int(b), ","), b / GIB)


def ft(x):
    if x is None:
        return "-"
    if x == math.inf:
        return "inf"
    return "%.1f" % x if x >= 100 else "%.2f" % x


def _num(v, src):
    return {"value": v, "source": src}


def _jsonable(o):
    if isinstance(o, float) and (math.isinf(o) or math.isnan(o)):
        return str(o)
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    return o


def parse_pair(s, flag, n_max, lo_hi=True, bound=None):
    try:
        vals = [float(x) for x in s.split(",")]
    except ValueError:
        raise Refusal("%s %r: not a number list (%s)" % (flag, s, "LO,HI" if n_max == 2 else "LO[,HI]")) from None
    if not 1 <= len(vals) <= n_max or (n_max == 2 and lo_hi and len(vals) != 2):
        raise Refusal("%s %r: %s" % (flag, s, "two numbers LO,HI" if lo_hi and n_max == 2 else "one or two numbers"))
    if any(not (v > 0) or math.isinf(v) for v in vals) or (bound is not None and any(v > bound for v in vals)):
        raise Refusal("%s %r: every value must be > 0%s" % (flag, s, " and <= %g" % bound if bound else ""))
    if len(vals) == 2 and vals[0] > vals[1]:
        raise Refusal("%s %r: LO > HI" % (flag, s))
    return vals


# ------------------------------------------------------------------ main

def build_parser():
    ap = argparse.ArgumentParser(prog="prequal.py", description="#240: size, placement and decode ceiling of a model "
                                 "from its shard headers and config.json; no download, no GPU (docs/prequal.md)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--repo", help="ORG/NAME on Hugging Face (headers by HTTP range through tools/hf_headers.py)")
    src.add_argument("--model-dir", help="a dir laid out as the header helper writes it (read only)")
    ap.add_argument("--revision", help="commit sha of --repo (required with --repo)")
    ap.add_argument("--context", type=int, required=True, help="context in tokens")
    ap.add_argument("--rates", help="JSON {B_nvme_gbps, R_pcie_gbps, R_dram_gbps, R_vram_gbps, source}")
    ap.add_argument("--band", help="LO,HI: calibration band applied to the binding ceiling")
    ap.add_argument("--m", help="LO[,HI]: NVMe share of the routed visits per token (else uniform)")
    ap.add_argument("--free-ram-gib", type=float, help="free host RAM (else HOST_PINNED_CAP, free RAM not read)")
    ap.add_argument("--commit-gib", type=float, help="Windows commit limit for the commit check")
    ap.add_argument("--vram-mib", type=int, help="card VRAM (else CARD_MIB_OF_RECORD)")
    ap.add_argument("--kv-dtype", choices=("bf16", "fp8"), default="bf16")
    ap.add_argument("--generic", action="store_true", help="the generic path also for a known family")
    ap.add_argument("--work", help="--repo: where the headers go (default models/prequal/<org>--<name>-<rev12>)")
    ap.add_argument("--converter", help="converter binary (default converter/target/release/converter%s)" % EXE)
    ap.add_argument("--states", help="states binary (default engine/target/release/states%s)" % EXE)
    ap.add_argument("--json", help="write the numbers with their sources here")
    return ap


def _log(msg):
    print("prequal: %s" % msg, file=sys.stderr)


def main(argv=None):
    ap = build_parser()
    try:
        a = ap.parse_args(argv)
    except SystemExit as e:
        return e.code if isinstance(e.code, int) else 2
    try:
        return run(a)
    except Refusal as e:
        print("prequal refused: %s" % e, file=sys.stderr)
        return 2
    except ToolFailure as e:
        print("prequal failed: %s" % e, file=sys.stderr)
        return 3


def _check_args(a):
    if a.context <= 0:
        raise Refusal("--context %d: a positive number of tokens" % a.context)
    if a.repo is not None:
        if not re.fullmatch(r"[\w.-]+/[\w.-]+", a.repo):
            raise Refusal("--repo %r: ORG/NAME" % a.repo)
        if not a.revision:
            raise Refusal("--repo %s needs --revision <commit sha> (a branch moves; the forecast names one revision)" % a.repo)
    elif a.revision:
        raise Refusal("--revision goes with --repo; --model-dir carries its own hf-revision.json")
    for flag, v in (("--free-ram-gib", a.free_ram_gib), ("--commit-gib", a.commit_gib), ("--vram-mib", a.vram_mib)):
        if v is not None and not (v > 0 and not math.isinf(v)):
            raise Refusal("%s %r: a positive number" % (flag, v))
    band = parse_pair(a.band, "--band", 2) if a.band else None
    m_values = parse_pair(a.m, "--m", 2, lo_hi=False, bound=1.0) if a.m else None
    return band, m_values


def run(a):
    band, m_values = _check_args(a)
    consts = engine_constants(SRC_ROOT)
    families = converter_families(SRC_ROOT)
    try:
        rates, rates_why = stage_ceiling.read_rates(a.rates, RATE_KEYS, Refusal)
    except (OSError, ValueError) as e:
        raise Refusal("--rates %s: %s" % (a.rates, e)) from None
    fetched = None
    if a.repo:
        helper = HELPER_LOADER()
        org, name = a.repo.split("/")
        out_dir = a.work or os.path.join(SRC_ROOT, "models", "prequal", "%s--%s-%s" % (org, name, a.revision[:12]))
        try:
            fetched = helper.fetch_model(a.repo, a.revision, out_dir, curl=None, log=_log)
        except helper.HeaderError as e:
            raise Refusal(str(e)) from None
        except Exception as e:   # noqa: BLE001 - any other helper failure is a tool failure, named
            raise ToolFailure("tools/hf_headers.py fetch_model failed: %s: %s" % (type(e).__name__, e)) from None
        if not isinstance(fetched, dict) or not fetched.get("dir"):
            raise ToolFailure("tools/hf_headers.py fetch_model returned %r, not {\"dir\", \"headers_dir\", ...}" % (fetched,))
        model = read_model_dir(fetched["dir"], fetched.get("headers_dir"))
    else:
        model = read_model_dir(a.model_dir)
    tc = model["tc"]
    fam = families.get(model["model_type"]) if model["model_type"] else None
    family = fam["family"] if fam else None
    use_recipe = fam is not None and not a.generic
    stab = stability_for(consts, family)
    notes = []
    cls = classify(model["tensors"], tc)
    moe = cls["records"] > 0
    topk, topk_key = _int(tc, TOPK_KEYS)

    # ---- binaries
    conv_bin = a.converter or os.path.join(SRC_ROOT, "converter", "target", "release", "converter" + EXE)
    states_bin = a.states or os.path.join(SRC_ROOT, "engine", "target", "release", "states" + EXE)
    is_glm = family == "Glm5Next"
    if use_recipe and not os.path.isfile(conv_bin):
        raise ToolFailure("converter binary not found at %s: build it with `cargo build --release --manifest-path "
                          "converter/Cargo.toml` (or pass --converter, or --generic)" % conv_bin)
    if use_recipe and is_glm and not os.path.isfile(states_bin):
        raise ToolFailure("states binary not found at %s: build it with `cargo build --release --manifest-path "
                          "engine/Cargo.toml --bin states` (or pass --states, or --generic)" % states_bin)

    # ---- size
    hsrc = "%s/*.json (%d shard headers)" % (model["headers_dir"].replace("\\", "/"), len(model["shards"]))
    if use_recipe:
        plan = run_converter_plan(conv_bin, model)
        routed = plan.get("routed", 0)
        moe = routed > 0
        dense = plan["dense"] if moe else plan["subtotal"]
        size = {"container": plan["container"], "payload": plan["payload"], "subtotal": plan["subtotal"],
                "routed": routed, "dense": dense, "embed": plan["embed"], "omitted": plan["omitted"],
                "src": "converter plan --headers (%s), totals block" % conv_bin.replace("\\", "/")}
        path_label = "%s, recipe %s (converter plan)" % (family, fam["recipe"])
    else:
        size = {"container": 12 + cls["payload"], "payload": cls["payload"], "subtotal": cls["subtotal"],
                "routed": cls["routed"], "dense": cls["dense"], "embed": cls["embed"],
                "omitted": [(k, v[0], v[1]) for k, v in sorted(cls["omitted"].items())],
                "src": "generic classification of %s; container = 12 B head + payload, alignment zeros not estimated" % hsrc}
        path_label = ("%s, " % family if family else "") + "estimate: no recipe"
    moe_layers, experts = len(cls["moe_layers"]), cls["experts"]
    if moe and not (moe_layers and experts):
        raise ToolFailure("routed experts in the totals, but the headers give no MoE layers / expert count")
    if moe and not topk:
        notes.append("decode ceilings not derived: no %s in config.json" % " / ".join(TOPK_KEYS))
    candidates = [{"name": CNQ, "size": size["container"], "payload": size["payload"], "derived": False}]
    if moe:
        nv_record = -(-size["routed"] // (moe_layers * experts))
        nv_record = -(-nv_record // EXPERT_ALIGN) * EXPERT_ALIGN
        candidates[0]["record"], candidates[0]["codec"] = nv_record, "nvfp4"
        mul = cls["mul1"]
        records = moe_layers * experts
        mul_total = mul["total"] if mul["records"] == records else records * mul["record"]
        candidates.append({"name": MUL1, "size": size["container"] - size["routed"] + mul_total,
                           "payload": size["payload"] - size["routed"] + mul_total, "derived": True,
                           "record": mul["record"], "codec": "mul1", "mul1": mul, "records": records})

    # ---- KV
    kv = kv_cache(tc, a.context, a.kv_dtype)
    if kv["at_ctx"] is None:
        notes.append("KV cache not derived: %s" % "; ".join(kv["not_derived"]))
    kv_generic = kv["at_ctx"] or 0

    # ---- placement per candidate
    card_mib = a.vram_mib or consts["CARD_MIB_OF_RECORD"]["value"]
    card = card_mib << 20
    cap = consts["HOST_PINNED_CAP"]["value"]
    if a.free_ram_gib is not None:
        free = int(a.free_ram_gib * GIB)
        pinned_budget = max(0, min(cap, free - (consts["RAM_MARGIN_DEFAULT_GB"]["value"] << 30)))
        pinned_gib = pinned_budget >> 30
        pinned_why = "min(HOST_PINNED_CAP, --free-ram-gib %g - RAM_MARGIN_DEFAULT_GB); states --plan gets the whole GiB "                      "below it, CROW_PINNED_BUDGET_GB=%d" % (a.free_ram_gib, pinned_gib)
    else:
        pinned_budget, pinned_why, pinned_gib = cap, "HOST_PINNED_CAP; free RAM not read", None
    states_kv = None
    for c in candidates:
        pl = None
        if moe and use_recipe and is_glm:
            table, why = run_states_plan(states_bin, model["dir"], a.context, c["record"], c["codec"], pinned_gib, a.vram_mib)
            if table:
                fixed = table["ceiling"] - table["room"]
                pl = {"how": "states --plan", "ceiling": table["ceiling"], "fixed": fixed, "unit": table["unit"],
                      "hot": table["hot"], "pinned": table["pinned"], "nvme": table["nvme"], "fits": True,
                      "vram_used": fixed + table["hot"] * table["unit"], "pinned_used": table["pinned"] * table["unit"],
                      "slots": table.get("slots"), "kv": table["kv_total"], "table": table, "room": table["room"]}
                states_kv = table
            else:
                notes.append("states --plan refused %s (%s); the generic split is used for it" % (c["name"], why))
        if pl is None:
            pl = generic_split(consts=consts, stab=stab, card=card, dense=size["dense"], kv=kv_generic, moe=moe,
                               record=c.get("record", 0), moe_layers=moe_layers, experts=experts, topk=topk or 0,
                               pinned_budget=pinned_budget)
            pl["kv"] = kv_generic
            if moe and use_recipe and is_glm:
                pl["how"] = "generic split (states --plan refused)"
        c["placement"] = pl
        if a.commit_gib is not None:
            limit = int(a.commit_gib * GIB)
            used = pl["vram_used"] + pl["pinned_used"]
            c["commit"] = {"used": used, "limit": limit, "ok": used <= limit}
        else:
            c["commit"] = None
        if not pl["fits"]:
            c["fits"] = "no: does not fit, fixed VRAM %s > ceiling %s" % (fb(pl["fixed"]), fb(pl["ceiling"]))
        elif not moe:
            c["fits"] = "yes, all in VRAM (%.2f of %.2f GiB)" % (pl["fixed"] / GIB, pl["ceiling"] / GIB)
        elif pl["nvme"] == 0:
            c["fits"] = "yes, every expert resident (VRAM + pinned)"
        else:
            c["fits"] = "yes, %d of %d experts per MoE layer read from NVMe" % (pl["nvme"], experts)
        if c["commit"] is not None and not c["commit"]["ok"]:
            c["fits"] += "; commit over the limit"
        # ---- decode
        c["decode"] = []
        if pl["fits"] and (not moe or topk):
            c["decode"] = decode_rows(moe=moe, dense=size["dense"], kv=pl["kv"], rates=rates, m_values=m_values,
                                      placement=pl, record=c.get("record", 0), moe_layers=moe_layers, topk=topk or 0,
                                      experts=experts)
        # ---- prefill (tiered: the chunk bound only)
        c["prefill"] = None
        if moe and pl["fits"] and pl["nvme"] > 0 and rates.get("B_nvme_gbps"):
            nb = pl["nvme"] * moe_layers * c["record"]
            c["prefill"] = consts["CHUNK_CAP"]["value"] / (nb / (rates["B_nvme_gbps"] * 1e9))
            c["prefill_bytes"] = nb
    if m_values and not moe:
        notes.append("--m ignored: a dense model has no expert tier")

    report = {
        "date": datetime.date.today().isoformat(), "context": a.context, "kv_dtype": a.kv_dtype,
        "model": {"input": ("%s@%s" % (a.repo, a.revision)) if a.repo else model["dir"].replace("\\", "/"),
                  "fetched": bool(fetched), "hf": model["hf"], "model_type": model["model_type"],
                  "mt_key": model["mt_key"], "family": family,
                  "recipe": fam["recipe"] if fam else None, "path": path_label,
                  "tensors": len(model["tensors"]), "shards": len(model["shards"]),
                  "bytes_transferred": fetched.get("bytes_transferred") if fetched else 0,
                  "requests": fetched.get("requests") if fetched else 0, "cached": fetched.get("cached") if fetched else None,
                  "estimate": not use_recipe},
        "moe": moe, "moe_layers": moe_layers, "experts": experts, "topk": topk, "topk_key": topk_key,
        "size": size, "classification": {k: cls[k] for k in ("quant", "keep", "embed", "routed", "records", "scales", "empty")},
        "candidates": candidates, "kv": kv, "states_kv": states_kv, "rates": rates, "rates_why": rates_why,
        "band": band, "m": m_values, "stability": stab, "card": card, "card_mib": card_mib,
        "card_src": "--vram-mib" if a.vram_mib else consts["CARD_MIB_OF_RECORD"]["source"],
        "pinned_budget": pinned_budget, "pinned_why": pinned_why, "constants": consts, "notes": notes,
        "hsrc": hsrc, "families_src": next(iter(families.values()))["source"],
        "states_bin": states_bin.replace("\\", "/") if (use_recipe and is_glm) else None,
    }
    md = render(report)
    print(md)
    if a.json:
        with open(a.json, "w", encoding="utf-8") as f:
            json.dump(_jsonable(to_json(report)), f, indent=1)
    return 0


def render(r):
    m, out, src = r["model"], [], []
    out.append("## Pre-qualification (no download, no GPU) — %s" % r["date"])
    out.append("")
    ident = m["input"]
    if m["fetched"]:
        repo, rev = ident.split("@", 1)
        ident = "%s@%s" % (repo, rev[:12])
        moved = "%s B transferred (%s requests%s)" % (format(m["bytes_transferred"] or 0, ","), m["requests"],
                                                       ", headers cached" if m["cached"] else "")
    else:
        if m["hf"] and m["hf"].get("revision"):
            ident += " (%s@%s, hf-revision.json)" % (m["hf"].get("repo"), str(m["hf"]["revision"])[:12])
        moved = "0 B transferred (--model-dir)"
    out.append("**Model:** %s; %s; %s; %s tensors in %d shard%s; %s." % (
        ident, m["model_type"], m["path"], format(m["tensors"], ","), m["shards"], "" if m["shards"] == 1 else "s", moved))
    out.append("")
    out.append("| Candidate | Size | VRAM / pinned / NVMe | Commit | Fits? |")
    out.append("|---|---|---|---|---|")
    for c in r["candidates"]:
        pl = c["placement"]
        size = fb(c["size"]) + (", derived" if c["derived"] else "") + (", estimate: no recipe" if m["estimate"] else "")
        if r["moe"] and pl["fits"]:
            tiers = "%d / %d / %d experts per MoE layer (%s)" % (pl["hot"], pl["pinned"], pl["nvme"], pl["how"])
        elif not r["moe"]:
            tiers = "%.2f GiB / 0 / 0 (no expert tier; %s)" % (pl["vram_used"] / GIB, pl["how"])
        else:
            tiers = "- (%s)" % pl["how"]
        if c["commit"]:
            cm = "%.2f of %.2f GiB: %s" % (c["commit"]["used"] / GIB, c["commit"]["limit"] / GIB,
                                          "ok" if c["commit"]["ok"] else "over")
        else:
            cm = "not checked (--commit-gib)"
        out.append("| %s | %s | %s | %s | %s |" % (c["name"], size, tiers, cm, c["fits"]))
    out.append("")
    kv = r["kv"]
    kv_txt = "Generic: " + ("; ".join(kv["lines"]) if kv["lines"] else "-")
    if kv["at_ctx"] is not None:
        kv_txt += "; total %s" % fb(kv["at_ctx"])
    if kv["not_derived"]:
        kv_txt += "; not derived: %s" % "; ".join(kv["not_derived"])
    if r["states_kv"]:
        s = r["states_kv"]
        kv_txt = ("states --plan: MLA latent %s + indexer %s + KDA state + conv %s = %s (%s B per token). " % (
            format(s.get("latent", 0), ","), format(s.get("indexer", 0), ","), format(s.get("kda", 0), ","),
            fb(s["kv_total"]), format(s.get("kv_per_token", 0), ","))) + kv_txt
    out.append("**KV cache at %s (%s):** %s." % (format(r["context"], ","), r["kv_dtype"], kv_txt))
    out.append("")
    for c in r["candidates"]:
        for row in c["decode"]:
            ce = row["ceil"]
            if ce["binding"] is None:
                out.append("**Decode ceiling, %s (%s):** not computed: no rate for any stage (--rates with %s)." % (
                    c["name"], row["label"], ", ".join(RATE_KEYS)))
                continue
            stages = ", ".join("%s %s" % (k, ft(ce[k]["ceiling_tok_s"])) for k in ("vram", "pcie", "nvme", "dram") if k in ce)
            if r["band"]:
                lo, hi = r["band"]
                tail = "× band %g–%g → **%s–%s tok/s**" % (lo, hi, ft(ce["ceiling_tok_s"] * lo), ft(ce["ceiling_tok_s"] * hi))
            else:
                tail = "band: not given (--band LO,HI; the calibration lives outside this repo)"
            out.append("**Decode ceiling, %s (%s):** binding %s %s tok/s (%s; serial %s) %s; %s." % (
                c["name"], row["label"], ce["binding"] or "-", ft(ce["ceiling_tok_s"]), stages, ft(ce["serial_tok_s"]),
                tail, (M_NOTE if row["label"].startswith("m =") else UNIFORM_NOTE) if r["moe"] else
                "all weights and the KV cache read from VRAM once per token"))
    if not any(c["decode"] for c in r["candidates"]):
        out.append("**Decode ceiling:** not derived (%s)." % ("; ".join(r["notes"]) or "does not fit"))
    out.append("")
    if r["moe"]:
        parts = []
        for c in r["candidates"]:
            if c["prefill"] is not None:
                parts.append("%s %s / (%d NVMe experts × %d MoE layers × %s B / %s GB/s) = %s tok/s" % (
                    c["name"], format(r["constants"]["CHUNK_CAP"]["value"], ","), c["placement"]["nvme"], r["moe_layers"],
                    format(c["record"], ","), "%g" % r["rates"]["B_nvme_gbps"], ft(c["prefill"])))
            elif c["placement"]["fits"] and c["placement"].get("nvme") == 0:
                parts.append("%s: no NVMe tier, compute-bound, not modelled here" % c["name"])
            else:
                parts.append("%s: not computed (%s)" % (c["name"], "no B_nvme_gbps" if not r["rates"].get("B_nvme_gbps") else "does not fit"))
        out.append("**Prefill:** chunk bound (CHUNK_CAP tokens per NVMe pass), %s; %s." % ("; ".join(parts), PREFILL_NOTE))
    else:
        out.append("**Prefill:** compute-bound, not modelled here.")
    out.append("")
    out.append("**Not computed here:** %s." % NOT_COMPUTED)
    if r["notes"]:
        out.append("")
        out.append("**Notes:** %s." % "; ".join(r["notes"]))
    out.append("")
    # ---- sources, one line per number
    k = r["constants"]
    sz = r["size"]
    src.append("tensors, shapes, dtypes: %s" % r["hsrc"])
    src.append("model_type: config.json %s = %s; families: %s" % (m["mt_key"], m["model_type"], r["families_src"]))
    for c in r["candidates"]:
        if c["derived"]:
            mul = c["mul1"]
            src.append("%s size %s: %s − routed NVFP4 %s + %s records × %s B (%s)" % (
                c["name"], format(c["size"], ","), "converter container" if not m["estimate"] else "estimated container",
                format(sz["routed"], ","), format(c["records"], ","), format(c["record"], ","), mul["how"]))
        else:
            src.append("%s size %s: %s" % (c["name"], format(c["size"], ","), sz["src"]))
    src.append("GPU text subtotal %s, dense resident %s, routed %s, token embedding (host RAM) %s: %s" % (
        format(sz["subtotal"], ","), format(sz["dense"], ","), format(sz["routed"], ","), format(sz["embed"], ","), sz["src"]))
    for why, n, b in sz["omitted"]:
        src.append("omitted, not written: %s tensors, %s B source (%s)" % (format(n, ","), format(b, ","), why))
    if r["moe"]:
        src.append("MoE layers %d × experts %d: routed-expert tensors in the headers (%s); top-k %s: config.json %s" % (
            r["moe_layers"], r["experts"], r["candidates"][0].get("codec"), r["topk"], r["topk_key"]))
        src.append("NVFP4 record %s B: routed bytes / (MoE layers × experts), 4096-aligned" % format(r["candidates"][0]["record"], ","))
    src.append("KV keys: config.json %s%s" % (", ".join(kv["keys"]), "; states --plan KV lines (%s)" % r["states_bin"] if r["states_kv"] else ""))
    for c in r["candidates"]:
        pl = c["placement"]
        if pl["how"] == "states --plan":
            src.append("%s placement %d / %d / %d, unit %s B, plan ceiling %s B, room %s B: states --plan (%s) "
                       "--expert-bytes %d --expert-codec %s, CROW_CONTEXT=%d, pinned %s" % (
                           c["name"], pl["hot"], pl["pinned"], pl["nvme"], format(pl["unit"], ","), format(pl["ceiling"], ","),
                           format(pl["room"], ","), r["states_bin"], c["record"], c["codec"], r["context"], r["pinned_why"]))
        else:
            parts = " + ".join("%s %s" % (n, format(b, ",")) for n, b in pl["parts"])
            src.append("%s placement (%s): ceiling %s B = Stability::%s.vram_ceiling(card %s B) (%s); fixed %s B = %s%s" % (
                c["name"], pl["how"], format(pl["ceiling"], ","), r["stability"]["name"], format(r["card"], ","),
                r["stability"]["why"], format(pl["fixed"], ","), parts,
                "; unit %s B; pinned budget %s B (%s)" % (format(pl["unit"], ","), format(r["pinned_budget"], ","), r["pinned_why"])
                if r["moe"] and "unit" in pl else ""))
    src.append("card %s MiB: %s" % (format(r["card_mib"], ","), r["card_src"]))
    for name in ("HOST_PINNED_CAP", "RAM_MARGIN_DEFAULT_GB", "CARD_MIB_OF_RECORD", "SAFETY", "LAUNCH_SLACK", "PF_TG",
                 "PF_ASYNC_DEFAULT", "CHUNK_CAP"):
        src.append("%s = %s (%s): %s" % (name, format(k[name]["value"], ","), k[name].get("expr", ""), k[name]["source"]))
    for name, s in k["stability"].items():
        src.append("Stability::%s = headroom %s, cap %s, decode stage rows %s: %s" % (
            name, format(s["vram_headroom"], ","), "None" if s["vram_cap"] is None else format(s["vram_cap"], ","),
            s["decode_stage_rows"], s["source"]))
    src.append("Stability of this model: %s (%s)" % (r["stability"]["name"], r["stability"]["why"]))
    src.append("rates: %s%s" % (r["rates_why"], (" " + json.dumps(r["rates"], sort_keys=True)) if r["rates"] else ""))
    src.append("band: %s" % ("--band %g,%g" % tuple(r["band"]) if r["band"] else "not given"))
    out.append("Sources:")
    out.extend("- " + s for s in src)
    out.append("")
    out.append("This is a forecast, not a measurement and not a gate.")
    return "\n".join(out)


def to_json(r):
    """the numbers of the block, each with its source"""
    m, sz, k = r["model"], r["size"], r["constants"]
    j = {"date": r["date"], "model": m, "context": r["context"], "kv_dtype": r["kv_dtype"], "moe": r["moe"],
         "moe_layers": _num(r["moe_layers"], "routed-expert tensors in %s" % r["hsrc"]),
         "experts": _num(r["experts"], "config.json / headers"), "topk": _num(r["topk"], "config.json %s" % r["topk_key"]),
         "size": {kk: _num(sz[kk], sz["src"]) for kk in ("container", "payload", "subtotal", "routed", "dense", "embed")},
         "omitted": [{"reason": w, "tensors": n, "source_bytes": b} for w, n, b in sz["omitted"]],
         "kv": {"generic": {"at_ctx": _num(r["kv"]["at_ctx"], "config.json " + ", ".join(r["kv"]["keys"])),
                            "per_token": _num(r["kv"]["per_token"], "config.json"), "lines": r["kv"]["lines"],
                            "not_derived": r["kv"]["not_derived"]},
                "states": ({kk: _num(r["states_kv"].get(kk), "states --plan (%s)" % r["states_bin"])
                            for kk in ("latent", "indexer", "kda", "kv_total", "kv_per_token")} if r["states_kv"] else None)},
         "card": _num(r["card"], r["card_src"]), "pinned_budget": _num(r["pinned_budget"], r["pinned_why"]),
         "stability": {kk: r["stability"][kk] for kk in ("name", "why", "vram_headroom", "vram_cap", "decode_stage_rows", "source")},
         "constants": {n: {"value": v["value"], "source": v["source"]} for n, v in k.items() if isinstance(v, dict) and "value" in v},
         "rates": {kk: _num(v, r["rates_why"]) for kk, v in r["rates"].items()}, "band": r["band"], "m": r["m"],
         "notes": r["notes"], "candidates": []}
    for c in r["candidates"]:
        pl = c["placement"]
        cj = {"name": c["name"], "size": _num(c["size"], "derived: CNQ4.5 container - routed NVFP4 + records x MUL1 record"
                                                if c["derived"] else sz["src"]),
              "record": _num(c.get("record"), (c.get("mul1") or {}).get("how", "routed / (MoE layers x experts)")),
              "placement": {kk: _num(pl.get(kk), pl["how"]) for kk in ("ceiling", "fixed", "unit", "hot", "pinned", "nvme",
                                                                       "vram_used", "pinned_used", "slots") if kk in pl},
              "fits": c["fits"], "commit": c["commit"],
              "decode": [{"label": row["label"], "m": row["m"], "stages": row["ceil"]} for row in c["decode"]],
              "prefill_tok_s": _num(c["prefill"], "CHUNK_CAP %s / NVMe pass at B; %s" % (k["CHUNK_CAP"]["source"], PREFILL_NOTE))}
        if r["band"] and c["decode"]:
            cj["band_tok_s"] = [[row["ceil"]["ceiling_tok_s"] * r["band"][0] if row["ceil"]["ceiling_tok_s"] is not None else None,
                                 row["ceil"]["ceiling_tok_s"] * r["band"][1] if row["ceil"]["ceiling_tok_s"] is not None else None]
                                for row in c["decode"]]
        j["candidates"].append(cj)
    return j


if __name__ == "__main__":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.exit(main())
