# Pre-qualification of a new model: `tools/prequal.py`

`tools/prequal.py` (issue #240) forecasts, before any download and without a GPU, what a model costs on
this machine: the container size per candidate, the KV cache at a context, the placement of the routed
experts over VRAM / pinned host RAM / NVMe, and the decode ceiling per stage. It reads the shard headers
and `config.json` only. It is an orchestrator: for a family the converter has a recipe for, the numbers
come from `converter plan` and `states --plan` unchanged; only an unknown family goes through the generic
path in this file. The result is a forecast, not a measurement and not a gate.

```text
python -I tools/prequal.py (--repo ORG/NAME --revision SHA | --model-dir DIR) --context N \
    [--rates rates.json] [--band LO,HI] [--m LO[,HI]] [--free-ram-gib N] [--commit-gib X] [--vram-mib N] \
    [--kv-dtype bf16|fp8] [--generic] [--work DIR] [--converter PATH] [--states PATH] [--json out.json]
```

Stdlib only, cross-platform, `python -I` (it parses downloaded JSON). The stage arithmetic is shared with
`tools/glm_tier_sim.py` through `tools/stage_ceiling.py`.

## Input

| flag | what is read |
|---|---|
| `--model-dir DIR` | `DIR/config.json`, `DIR/generation_config.json` (optional), `DIR/model.safetensors.index.json` or one single-file header, `DIR/hf-revision.json` (optional), `DIR/headers/<shard>.json` = `{"shard", "size", "data_start", "header"}`. Nothing is written into `DIR`. |
| `--repo ORG/NAME --revision SHA` | `tools/hf_headers.py` `fetch_model(repo, revision, out_dir)` reads the headers by HTTP range into `--work` (default `models/prequal/<org>--<name>-<rev[:12]>`, git-ignored), then as `--model-dir`. The block reports the helper's `bytes_transferred`. A `HeaderError` (missing revision, no index, ...) exits 2 with its reason. |

## What it computes

**Family.** `text_config.model_type` (else the top-level `model_type`) against the strings of
`Family::model_type` in `converter/src/recipe.rs`, parsed at run time. Known (`qwen4_exp_text`,
`qwen3_5_text`, `glm5_next_text` at `7b0fea8`): `converter plan --headers <dir>/headers <dir>` and its totals
block. A converter refusal exits 3; there is no silent fallback. Unknown, or `--generic`: the generic path,
printed as `estimate: no recipe`.

**Generic size.** Every header tensor, by name and shape, after the keep rules of `recipe.rs`
(`decide_flash_next`, `decide_qwen35_dense`, `decide_glm5_next`, `omitted`):

- not written, counted apart: vision / audio towers and projectors; MTP / NextN layers (`mtp` / `nextn` in
  the name, or a layer index ≥ `num_hidden_layers`);
- read, not written: FP8 scales (`*weight_scale_inv`, `*weight_scale`, `*input_scale`);
- token embedding: host RAM, its own line;
- routed experts: `.experts.<int>.` tensors, or a fused 3-D `.experts.` tensor whose first dim is
  `n_routed_experts` / `num_experts` / `num_local_experts`;
- keep (BF16 2 B per value, F32 sources 4 B): `lm_head`, router (`mlp.gate.weight`, `router`,
  `block_sparse_moe.gate`, `shared_expert_gate`), norms, every 1-D tensor, every tensor whose element count
  is not a multiple of 64;
- everything else NVFP4, 36 B per 64 values (`DtypeOut::bytes`).

The container estimate is the 12-B head plus the payload; alignment zeros are not estimated.

**Candidates.** `CNQ4.5`, and for a MoE `MUL1 K=3 experts + CNQ4.5 dense`: container − routed NVFP4 +
records × MUL1 record (labelled `derived`). The record is `record_bytes(hidden, inter, 3)` of
`tools/glm_mul1_quantize.py` (the Python twin of `converter/src/mul1.rs` `RecordLayout`; 9,474,048 B for
hidden 4096 / inter 2048). An expert that is not a gate/up/down triple of multiples of 128 falls back to
3.05 bpw, rounded up to a 4096-B record, and the source line says so. The NVFP4 record is routed bytes /
(MoE layers × experts), 14,155,776 B for GLM-5.3-Flash.

**KV cache** from `config.json` (`text_config` if nested), at `--kv-dtype` (default bf16):

| layers | bytes |
|---|---|
| GQA / MHA | `2 · L_full · kv_heads · head_dim · ctx · bytes` |
| MLA | `L · (kv_lora_rank + qk_rope_head_dim) · ctx · bytes` |
| sliding window | as above with `min(ctx, sliding_window)` |
| linear / recurrent (`layer_types`, `full_attention_interval`) | a constant f32 state per layer: GDN (`linear_*` keys) or KDA (`linear_attn_config`); anything else is named as not derived |

For glm5_next the KV lines of `states --plan` (MLA latent, indexer, KDA) are printed and used, the generic
KV beside them. The decode ceiling counts the whole KV cache plus the linear states as read once per token.

**Placement.**

- glm5_next (known): per candidate `states --plan --expert-bytes <record> --expert-codec nvfp4|mul1`
  with `CROW_MODEL_DIR`, `CROW_CONTEXT` and, when `--free-ram-gib` is given, `CROW_PINNED_BUDGET_GB` (whole
  GiB); `--vram-mib` passed through. `CROW_PF_TG` / `CROW_PF_ASYNC` are not passed on, so the engine
  defaults are planned. A refusal is printed and the generic split is used, labelled.
- Generic split (every other family, or `--generic`): ceiling = `Stability::vram_ceiling(card)`; card =
  `CARD_MIB_OF_RECORD << 20` or `--vram-mib`. Flash-Next and the 27B take `Stability::OF_RECORD`;
  glm5_next and every unknown family `Stability::GLM5_NEXT`, the only row with the measured 31.9 GiB cap.
  Fixed VRAM = dense resident part (dense model: GPU text subtotal) + KV + `SAFETY` + `LAUNCH_SLACK` + for
  a MoE the staging slots × record (`Stability::stage_slots`: decode rows × top-k + `PF_TG` × 2).
  unit = MoE layers × record; N = min(E, ⌊(ceiling − fixed) / unit⌋); pinned budget = min(`HOST_PINNED_CAP`,
  free RAM − `RAM_MARGIN_DEFAULT_GB`) with `--free-ram-gib`, else the cap ("free RAM not read");
  P = min(E − N, ⌊pinned / unit⌋); NVMe = E − N − P. Fixed > ceiling: "does not fit". It leaves out what only
  the GLM planner books (kv_b decoded to BF16, the indexer cache), so N comes out a few units higher.
- Commit check with `--commit-gib`: VRAM used + pinned used ≤ limit, both 1:1.

Routing is uniform (share = NVMe / E), or `--m` gives the NVMe share and the rest 1 − m is split N:P.
Hot sets need a model run (`tools/glm_tier_sim.py`, `tools/hotset-eval.py`).

**Ceilings.** `stage_ceiling.ceilings`, the semantics `glm_tier_sim.dyn_cost` had: per stage
rate · 10⁹ / bytes per token; the lowest given ceiling binds (full overlap); the serial bound adds the stage
times, only when every rate is given. visits = MoE layers × top-k.

| stage | bytes per token | rate key |
|---|---|---|
| vram | dense resident + KV at ctx + visits · N/E · record | `R_vram_gbps` |
| pcie | visits · P/E · record | `R_pcie_gbps` |
| nvme | visits · share · record | `B_nvme_gbps` |
| dram | pcie + nvme bytes (data landed plus every PCIe transfer) | `R_dram_gbps` |

A dense model has the vram stage only (GPU text subtotal + KV). `--band LO,HI` multiplies the binding
ceiling; without it the block says `band: not given`. The calibration lives outside this repo; the repo
holds no copy of it.

**Prefill.** Tiered models: only the chunk bound `CHUNK_CAP` tokens / (NVMe experts × MoE layers × record /
B), labelled uncalibrated (the GLM bound did not hold: 1,137.5 tok/s measured above it,
`runs/glm53-flash/quick/README.md:38`). Resident models: "compute-bound, not modelled here".

**Not computed:** path A/B, engine gap, external best, licence, disk (runbook, by hand).

## Rates file

`{"B_nvme_gbps": x, "R_pcie_gbps": y, "R_dram_gbps": z, "R_vram_gbps": w, "source": "..."}`, GB/s (10⁹ B).
A present value must be a positive number (else exit 2); a missing one leaves its stage without a ceiling.
`glm_tier_sim.py dyn --rates` reads the same file and ignores the two extra keys.

## Sources of the numbers

Every engine constant is parsed from its source file at run time (a regex and a whitelisted integer
evaluator: int literals with `_`, `<< >> * / + -`, parentheses, `/` as u64 integer division; never
`eval`) and carries its `path:line`. A constant that is not found exits 3 with its name.

| constant | file |
|---|---|
| `HOST_PINNED_CAP`, `CHUNK_CAP`, `Stability::OF_RECORD`, `Stability::GLM5_NEXT` (and the constants it names), `Stability::of` | `engine/src/geo.rs` |
| `SAFETY`, `LAUNCH_SLACK`, `RAM_MARGIN_DEFAULT_GB` | `engine/src/manager.rs` |
| `PF_TG`, the `CROW_PF_ASYNC` default | `engine/src/gen.rs` |
| `CARD_MIB_OF_RECORD` | `engine/src/bin/states.rs` |
| `Family::model_type`, `Family::recipe` | `converter/src/recipe.rs` |

The block ends with one source line per number: header files, config keys, the constants above,
`converter plan`, `states --plan`, the rates file.

## Exit codes

| code | meaning |
|---|---|
| 0 | the block is printed (and `--json` written: the same numbers, each with its source) |
| 2 | refused input, reason named: missing revision, a model dir without index and without single-file header, a missing shard header, an unknown dtype, a bad `--band` / `--m` / `--context` / `--rates` value |
| 3 | a tool it calls failed or a source constant was not found, named: a converter refusal, a missing binary (the message names the `cargo build` command), a states table it cannot parse |

Default binaries: `converter/target/release/converter(.exe)` and `engine/target/release/states(.exe)`;
`--converter` / `--states` override them. `--generic` needs neither.

## Tests

`python -I tools/test_prequal.py` (stdlib unittest, no network, no GPU). Synthetic model dirs are built
from `converter/tests/fixtures/glm53-flash-tensors.tsv` (experts expanded, contiguous data_offsets,
data_start and size per shard) and `converter/tests/fixtures/qwen3.8-27b-tensors.tsv` with the engine
fixture configs. Covered: generic classification and size on GLM (CNQ4.5 within ±5 % of
178,478,618,624 B, `estimate: no recipe`); the `converter plan` totals parser and the `states --plan` table
parser on captured real output (178,478,618,624 B; 50 / 124 / 114); the KV formulas (GQA, MLA, GDN and KDA
hybrids, sliding window); the constants on the real sources (`HOST_PINNED_CAP = 46 << 30`,
`engine/src/geo.rs:42`) and on a copy without it (exit 3 naming it); the evaluator refusing calls and names;
`ceilings()` against the old `dyn_cost` arithmetic; the NVMe ceiling at m 0.11 / 0.125 (19.97 /
17.58 tok/s); the 27B all-in-VRAM ceiling within ±10 % of 120 tok/s; the generic GLM split within ±10 % of
50 / 124 / 114; the refusals. With `PREQUAL_CONVERTER` / `PREQUAL_STATES` set to the binaries, four more
tests run them (else skipped with that reason). Every test was red with its hunk broken (2026-10-17).

## Checked against the reference numbers (2026-10-17, crow-nest `7b0fea8` binaries)

Rates: B 6.994 GB/s (`docs/nvme-read-rate.md:16`), PCIe 47.78 GB/s (`docs/architecture.md:789`), DRAM 89.6
GB/s (nominal, not measured), VRAM 1,792 GB/s (datasheet, not measured).

| run | result | reference |
|---|---|---|
| GLM-5.3-Flash, 62 real headers, ctx 200,000, both binaries | CNQ4.5 178,478,618,624 B; MUL1 placement 50 / 124 / 114; NVMe ceiling 19.97 / 17.58 tok/s at m 0.11 / 0.125 | converter plan (exact); states --plan (exact; the ticket's 51 / 124 / 113 predates #161); 0xSero-style plan 18–20 tok/s |
| same, `--generic` | CNQ4.5 178,143,013,220 B (−0.19 %); MUL1 split 54 / 124 / 110 (+8.0 % / 0.0 % / −3.5 %); CNQ4.5 split 35 / 83 / 170 vs 32 / 83 / 173 | ±5 % size, ±10 % per tier |
| Qwen3.8-27B, synthetic fixture headers, converter plan, ctx 190 | all in VRAM, ceiling 122.6 tok/s (+2.2 %) | ≈ 120 tok/s (nibor1896/Crow#300) |
| same, ctx 200,000 | fits: 26.44 of 31.84 GiB, ceiling 64.66 tok/s | — |

The decode ceilings over-forecast measurements by a factor of about 0.5–0.8 (#240 evidence table); that is
what `--band` is for.
