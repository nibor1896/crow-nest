"""export_glm5_moe_golden.py — crow-nest #164 (GLM-5.3-Flash plan step 15, gate G3): the goldens of
the glm5_next FFN sub-block on SYNTHETIC weights with the real block shapes (hidden 4096, 288
experts top-8, expert width 2048, shared 2048, dense 12,288), computed by HF's own modules
(transformers 5.16.1, modeling_glm5_next.py, eager, f32, CPU):

  router  Glm5NextTextTopkRouter (M:158-183) on 64 router rows and on the 4 MoE rows
  MoE     Glm5NextTextMoE.forward (M:200-207): router -> Glm5NextTextExperts (M:108-142) -> + shared
          Glm5NextTextMLP (M:84-104), one row per call
  dense   Glm5NextTextMLP at intermediate_size 12,288 (layers 0-2) on 2 rows

Inputs: the directory `glm5_moe_write_oracle_inputs` writes (engine/src/glm5_moe.rs), every
encoded weight the engine reads decoded to f32 in HF's [out][in] layout - the routed experts are
#181 synthetic MUL1 records (K = 3) decoded by the #181 decoder (cpu_mul1::testkit::RefLinear),
the shared and dense FFN synthetic NVFP4 matrices, the router BF16, the bias f32.

The experts module holds only the experts a row routes to (288 full experts would be 29 GB of
f32): Remap renames the 8 global ids of the row to 0..7 before Glm5NextTextExperts.forward, a
bijection on the hit set, so the module's math runs unchanged.

  python -I oracle/export_glm5_moe_golden.py --inputs <dump dir> [--out engine/tests/fixtures/glm5/moe]

Threads: ORACLE_THREADS (default 2). Output (raw little-endian, row-major; manifest.json has shapes,
sha256, versions, the input sha256 the Rust test re-checks, the smallest 8th/9th choice gap):
  router-ids.i32 [64][8] ascending, router-weights.f32 [64][8] same order
  moe-ids.i32 [4][8], moe-weights.f32 [4][8], moe-y.f32 [4][4096], dense-y.f32 [2][4096]
"""
import argparse
import json
import os
import platform
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import glm5_common as G  # noqa: E402

import transformers  # noqa: E402
from transformers.models.glm5_next.modeling_glm5_next import (  # noqa: E402
    Glm5NextTextExperts,
    Glm5NextTextMLP,
    Glm5NextTextMoE,
    Glm5NextTextTopkRouter,
)

ROOT = os.path.abspath(os.path.join(HERE, ".."))
CONFIG = os.path.join(ROOT, "engine", "tests", "fixtures", "GLM-5.3-Flash", "config.json")
OUT = os.path.join(ROOT, "engine", "tests", "fixtures", "glm5", "moe")


def load(d, name, shape, dtype="<f4"):
    a = np.fromfile(os.path.join(d, name), dtype=dtype)
    n = int(np.prod(shape))
    if a.size != n:
        raise SystemExit(f"{name}: {a.size} values, want {shape}")
    return torch.from_numpy(a.reshape(shape).copy())


def load_bf16(d, name, shape):
    u = np.fromfile(os.path.join(d, name), dtype="<u2").astype(np.uint32) << 16
    return torch.from_numpy(u.view(np.float32).reshape(shape).copy())


class Remap(torch.nn.Module):
    """global expert ids -> the compact index of an experts module holding only the row's experts"""

    def __init__(self, experts, lut):
        super().__init__()
        self.experts = experts
        self.lut = lut

    def forward(self, hidden_states, top_k_index, top_k_weights):
        return self.experts(hidden_states, self.lut[top_k_index], top_k_weights)


def mlp(tc, inter, d, prefix):
    m = G.build_meta(Glm5NextTextMLP, tc, inter)
    H = tc.hidden_size
    m.load_state_dict({
        "gate_proj.weight": load(d, f"{prefix}_gate.f32", (inter, H)),
        "up_proj.weight": load(d, f"{prefix}_up.f32", (inter, H)),
        "down_proj.weight": load(d, f"{prefix}_down.f32", (H, inter)),
    }, assign=True, strict=True)
    return m


def canon(ids, w):
    order = ids.argsort(dim=-1)
    return ids.gather(1, order).to(torch.int32), w.gather(1, order).float()


def write(out, files, name, t):
    a = t.detach().contiguous().numpy()
    a = a.astype("<i4" if a.dtype == np.int32 else "<f4")
    p = os.path.join(out, name)
    a.tofile(p)
    files[name] = {"shape": list(a.shape), "dtype": "i32" if a.dtype.kind == "i" else "f32", "sha256": G.sha256_file(p)}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--config", default=CONFIG)
    a = ap.parse_args(argv)
    threads = int(os.environ.get("ORACLE_THREADS", "2"))
    torch.set_num_threads(threads)
    with open(a.config) as f:
        tc = G.text_config_from_dict(json.load(f))
    with open(os.path.join(a.inputs, "inputs.json")) as f:
        inp = json.load(f)
    H, E, K, I, DI = tc.hidden_size, tc.n_routed_experts, tc.num_experts_per_tok, tc.moe_intermediate_size, tc.intermediate_size
    TR, TM, TD = inp["tokens"]["router"], inp["tokens"]["moe"], inp["tokens"]["dense"]
    assert (H, E, K, I, DI) == (4096, 288, 8, 2048, 12288), (H, E, K, I, DI)

    router = Glm5NextTextTopkRouter(tc)
    with torch.no_grad():
        router.weight.copy_(load_bf16(a.inputs, "router.bf16", (E, H)))
        router.e_score_correction_bias.copy_(load(a.inputs, "bias.f32", (E,)))
    x_route = load(a.inputs, "x_route.f32", (TR, H))
    x_moe = load(a.inputs, "x_moe.f32", (TM, H))
    x_dense = load(a.inputs, "x_dense.f32", (TD, H))

    gaps = []

    def route(x):
        with torch.no_grad():
            logits, w, idx = router(x)
            choice = logits.sigmoid() + router.e_score_correction_bias
            top = choice.sort(dim=-1, descending=True).values
            gaps.extend((top[:, K - 1] - top[:, K]).tolist())
        return canon(idx, w)

    ids_r, w_r = route(x_route)
    ids_m, w_m = route(x_moe)

    moe = G.build_meta(Glm5NextTextMoE, tc)
    moe.gate = router
    moe.shared_experts = mlp(tc, I * tc.n_shared_experts, a.inputs, "shared")
    ys = []
    for t in range(TM):
        hit = sorted(set(ids_m[t].tolist()))
        for e in hit:
            if not os.path.exists(os.path.join(a.inputs, f"expert_{e}_gate_up.f32")):
                raise SystemExit(f"row {t}: HF routes to expert {e}, the engine lane dumped no record for it "
                                 f"(lane ids {inp['lane_moe_ids'][t * K:(t + 1) * K]}) - a routing mismatch")
        ex = G.build_meta(Glm5NextTextExperts, tc)
        ex.gate_up_proj = torch.nn.Parameter(torch.stack([load(a.inputs, f"expert_{e}_gate_up.f32", (2 * I, H)) for e in hit]), requires_grad=False)
        ex.down_proj = torch.nn.Parameter(torch.stack([load(a.inputs, f"expert_{e}_down.f32", (H, I)) for e in hit]), requires_grad=False)
        ex.num_experts = len(hit)
        lut = torch.zeros(E, dtype=torch.long)
        lut[torch.tensor(hit)] = torch.arange(len(hit))
        moe.experts = Remap(ex, lut)
        with torch.no_grad():
            ys.append(moe(x_moe[t:t + 1])[0])
        del ex, moe.experts
    y_moe = torch.stack(ys)

    dense = mlp(tc, DI, a.inputs, "dense")
    with torch.no_grad():
        y_dense = dense(x_dense)

    os.makedirs(a.out, exist_ok=True)
    files = {}
    write(a.out, files, "router-ids.i32", ids_r)
    write(a.out, files, "router-weights.f32", w_r)
    write(a.out, files, "moe-ids.i32", ids_m)
    write(a.out, files, "moe-weights.f32", w_m)
    write(a.out, files, "moe-y.f32", y_moe)
    write(a.out, files, "dense-y.f32", y_dense)
    lane = [sorted(inp["lane_moe_ids"][t * K:(t + 1) * K]) for t in range(TM)]
    used = sorted({int(e) for e in ids_m.flatten().tolist()})
    man = {
        "ticket": "crow-nest #164 (plan step 15, gate G3)",
        "generator": "oracle/export_glm5_moe_golden.py",
        "inputs_writer": "engine/src/glm5_moe.rs glm5_moe_write_oracle_inputs",
        "reference": "transformers %s modeling_glm5_next.py: Glm5NextTextTopkRouter, Glm5NextTextMoE (experts compact, Remap), Glm5NextTextMLP" % transformers.__version__,
        "versions": {"transformers": transformers.__version__, "torch": torch.__version__, "numpy": np.__version__, "python": platform.python_version()},
        "torch_threads": threads,
        "experts_implementation": tc._experts_implementation,
        "config": {"path": os.path.relpath(a.config, ROOT).replace(os.sep, "/"), "sha256": G.sha256_file(a.config)},
        "tokens": inp["tokens"],
        "synthetic": {k: inp[k] for k in ("seed", "record_seed", "record_seed_step", "x_amp", "router_amp", "bias_amp", "nvfp4_global_scale")},
        "inputs_sha256": inp["inputs_sha256"],
        "records_sha256": {str(e): inp["records_sha256"][str(e)] for e in used},
        "router_min_gap": min(gaps),
        "router_gap_ties": sum(1 for g in gaps if g == 0.0),
        "lane_routing_equal_at_dump": lane == ids_m.tolist(),
        "clamp_fraction": inp["clamp_fraction"],
        "moe_y_rms": float(y_moe.pow(2).mean().sqrt()),
        "dense_y_rms": float(y_dense.pow(2).mean().sqrt()),
        "files": files,
    }
    with open(os.path.join(a.out, "manifest.json"), "w", newline="\n") as f:
        json.dump(man, f, indent=1)
        f.write("\n")
    print(json.dumps({k: man[k] for k in ("router_min_gap", "router_gap_ties", "lane_routing_equal_at_dump", "clamp_fraction", "moe_y_rms", "dense_y_rms")}))


if __name__ == "__main__":
    main()
