# Issue #91, phase 1 — the llama.cpp layer-rule arms, as converter capability + artifacts

Built 2026-09-20/21, machine blocked for GPU work (no engine binary was run — every
command below the converter line is a POST-REBOOT instruction, not a result). Scope of this
phase: the converter capability, the overlay artifacts it can build from the data that
actually exists on this disk, the verification of those artifacts, and the exact measurement
plan for when the machine is back. The #79 stop rule is binding on all of it: nothing here
was tuned, scored or selected on weight-space MSE, and nothing here will be — the arms are
fixed shapes copied from llama.cpp's rule table; their only acceptance instrument is
oracle-KLD (#90), per the hard rule now in `docs/improve-loop.md`.

## 1. What was ported, from where

`src/llama-quant.cpp` of ggml-org/llama.cpp (master, fetched 2026-09-20) shapes mixed
quantization by tensor CATEGORY and LAYER POSITION:

```cpp
auto use_more_bits = [](int i_layer, int n_layers) -> bool {
    return i_layer < n_layers/8 || i_layer >= 7*n_layers/8 || (i_layer - n_layers/8)%3 == 2;
};
```

plus: `output.weight` → Q6_K by default; `attn_v` (and the v-like categories) is the most
sensitive attention tensor and gets the top tier short of f16 — Q6_K under `Q4_K_M`/`Q5_K_M`,
Q8_0 outright for the 8-expert model ("trades just ~128MB", their comment).

On this container the only tier above NVFP4 that exists — in the CNQ1 format AND in the
engine's overlay refusal table — is `bf16`. So "one tier up" IS bf16 here, expressed as a
#77-style overlay (`CROW_CNQ_OVERLAY`, `boot.rs` front door, no second 105 GB file). The
`use_more_bits` set over the base's own 48 layers is exactly 24 layers:
`0-5` (first eighth), `42-47` (last eighth), `8, 11, 14, 17, 20, 23, 26, 29, 32, 35, 38, 41`
(every third middle layer, counting from `n/8 + 2`).

## 2. The capability: `converter layer-rule-overlay`

New file `converter/src/layer_rule_overlay.rs` (plus `mod`/dispatch/HELP lines in
`converter/src/main.rs`, and `blocks_to_bf16` made `pub` in `dense_overlay.rs` so the two
controls dequant by ONE arithmetic, not two copies). Additive in the house style: the word
`layer-rule-overlay` is taken off the front of the argument list and the conversion path
never sees it; `converter <model-dir> <out.cnq>` and `requant-check` read and write exactly
what they did before (`requant-check` re-run tonight: **495 of 495 byte-identical including
the global scale**, 2,583,306,240 values, 7 s).

```
converter layer-rule-overlay --base <container.cnq> --out <overlay.cnq> \
                             (--from-originals <dense.safetensors> | --from-container <base.cnq>) \
                             --arm attn-v-out|ffn-down-rule|ffn-down-all
```

Conventions kept from #77/#79: `--from-container` must be the same file as `--base`
(canonicalized) — the control must be built from the container it shadows; a selected tensor
missing from the originals is a hard error, not a skip; the originals must be dtype BF16;
shapes and value counts must match the base index; an arm kind that selects no tensor of
this base is refused (the empty-overlay hazard); `--arm` has no default and an unknown name
prints the arm table. The selection derives from the container's own index trailer
(`is_dense_text`: text section, nvfp4, not a routed expert) — never from an out-of-band
list. Seven unit tests pin the port (`cargo test --release`: 46 passed, 0 failed, was 39).

The written trailer carries `kind: "layer-rule-bf16"`, `issue: 91`, the arm name, the layer
rule, the ACTUAL selected layer list, the counts and `bf16_inexact_values` for the control.
The engine needs zero changes for these overlays: `overlay_refusal` gates on
`base_name`/`base_bytes` and per-tensor rules only (bf16 dense tensors are #77's accepted
path), and `overlay.kind` is free-form — verified against `engine/src/cnq.rs:597-676` and
`boot.rs:30-58`, not assumed.

## 3. The arms that were built

All six artifacts live in `converter/`, built against `Qwen3.8-Flash-Next-CNQ4.5-M.cnq`
(104,727,179,972 B), source `models/Qwen3.8-Flash-Next-original/dense/dense.safetensors`
(5,166,682,768 B, #76, re-proven tonight by the 495/495 requant-check).

| arm | rule (llama.cpp shape) | tensors | values | file | bytes |
|---|---|---|---|---|---|
| `attn-v-out` | `self_attn.v_proj` (12) + `self_attn.o_proj` (12) + `linear_attn.out_proj` (36) → bf16, ALL layers | 60 | 770,703,360 | `layer91-attn-v-out-originals.cnq` | 1,541,423,944 |
| same, CONTROL | same tensors, base's own NVFP4 dequantized to bf16 | 60 | 770,703,360 | `layer91-attn-v-out-control.cnq` | 1,541,423,901 |
| `ffn-down-rule` | `mlp.shared_expert.down_proj` → bf16 on the 24 `use_more_bits` layers | 24 | 39,321,600 | `layer91-ffn-down-rule-originals.cnq` | 78,650,697 |
| same, CONTROL | same 24 tensors dequantized | 24 | 39,321,600 | `layer91-ffn-down-rule-control.cnq` | 78,650,653 |
| `ffn-down-all` | `mlp.shared_expert.down_proj` → bf16 on all 48 layers | 48 | 78,643,200 | `layer91-ffn-down-all-originals.cnq` | 157,300,602 |
| same, CONTROL | same 48 tensors dequantized | 48 | 78,643,200 | `layer91-ffn-down-all-control.cnq` | 157,300,558 |

Total new bytes ≈ 3.55 GB, file sizes as listed (budget was ≤ 15 GB; 33 GB free after). VRAM
cost at load, computed from the byte delta against the NVFP4 they replace: `attn-v-out`
+1.03 GiB, `ffn-down-rule` +56.5 MB, `ffn-down-all` +113 MB — all far below #77's +3.36 GiB,
so the default operating point should hold; if not, `CROW_PINNED_BUDGET_GB` is the switch
and the refusal path is part of the report (issue item 3).

The router half of issue arm (a) needs no artifact and cannot have one: `mlp.gate.weight`
and `shared_expert_gate` are ALREADY `bf16` in the base container (sidecar of record; the
tool prints "97 router/gate/head tensors already bf16, nothing to elevate" when it builds
`attn-v-out`), and `lm_head` — llama.cpp's `output.weight` → Q6_K — is bf16 too. That half
of the arm is the status quo, which is itself a result of the 2026-09-03 keep-set.

`ffn-down-all` exists because `ffn-down-rule` alone cannot separate "the layer rule
matters" from "the tensor matters" — the pair of arms is the ablation, read as a pair.

### 3.1 Verification of the artifacts (CPU, run tonight)

1. The converter's own pass (built into the tool, runs after every write): re-read the
   finished overlay, re-derive every tensor from its claimed source, compare byte for byte.
   All six: `verified re-read N tensors, B bytes — every byte is its source's`.
2. Independent trailer readout (stdlib python, not the converter): kind/arm/layer list
   correct, payload bounds exact (`max(offset+len) == overlay.bytes`), all tensors bf16/text.
3. Independent dequant spot check (python re-implementation of `e2m1 × ue4m3 × global` +
   bf16 RNE): layer 0 `mlp.shared_expert.down_proj` from the CONTROL overlay is
   byte-identical to the independent dequant of the base container's own bytes; the
   ORIGINALS overlay's copy is byte-identical to `dense.safetensors`.
4. `converter requant-check` re-run: 495/495 (the source file is still the one of record).

The control rounding residue — wiring information, #77's discipline, NOT a quality number:
80.92 % of the `attn-v-out` values, 78.88 % of `ffn-down-rule`, 82.53 % of `ffn-down-all`
did not fit bf16 exactly and were rounded nearest-even (an NVFP4 value carries more
mantissa than bf16's 8 bits; the residue is ~2^-9 relative against NVFP4's ~2^-4).

## 4. What is NOT built, and exactly why

**Arm (b) of the issue — routed experts elevated to bf16 on the
first-eighth/last-eighth/every-third-middle layer rule — is not expressible tonight.** Two
independent blockers, either of which alone would stop it:

1. **The BF16 source data for routed experts is gone from this disk.**
   `models/Qwen3.8-Flash-Next-original/experts/` carries ONLY the eight
   `layer-NN.manifest.json` records (sha256, byte ranges, provenance of #79's fetch of
   layers 1,7,13,19,25,31,37,43) and the `.shard-headers` — the 37.5 GiB of
   `layer-NN.safetensors` payloads were deleted after #79 closed. `dense/dense.safetensors`
   carries dense-path originals only (selection rule: `section == "text" and dtype ==
   "nvfp4" and ".mlp.experts." not in name`) — verified, it holds 495 tensors, zero experts.
   No placeholder or synthetic weights were written anywhere. Re-fetching 8 layers is
   ~5.0 GB × 8 at the measured 10.0 MB/s ≈ 67 min of download + 37.5 GiB of disk (fits the
   33 GB free only after deleting something); all 48 layers is 201 GB — does not fit.
2. **The engine refuses bf16 routed experts by design.**
   `engine/src/cnq.rs::overlay_refusal` (the #79 table): *"bf16 — a routed expert may only
   be shadowed as nvfp4 of the same byte length, the expert slabs and the kernels are cut
   from it"*. `residency` cuts per-expert slabs out of `byte_len / 512` and hands them to
   kernels that index 36 B per 64 values; a bf16 expert would be a silently wrong-size slab.
   Expressing arm (b) needs ENGINE slab work plus a re-fetch — engine agents own that half.

For the same reasons the "one tier up" for the ROUTED `mlp.experts.down_proj` (issue arm
(c)'s expert half) is blocked: within the container's dtypes the tier above NVFP4 is bf16,
and a q8_0-style intermediate tier does not exist in CNQ1 or in the kernels. The issue
itself gates that work correctly (its item 4: re-quant formats only if the measurements
show the NVFP4 GRID is the binding constraint) — and #79's stop-rule result already showed
that re-rounding experts onto the SAME grid moved KLD the wrong way, which is exactly the
evidence phase 2 must weigh before any of it is built.

## 5. Post-reboot measurement plan (exact commands)

Instrument: `tools/oracle-kld.py` (#78, formulas pinned to llama.cpp `perplexity.cpp`) and
the #90 long-context form. Paired baseline is `none` in every case; the 0.025 paired
threshold of `docs/oracle-kld.md` decides; controls run BEFORE their originals arm is
quoted. ONE engine on the card at a time, GPU flock discipline of
`tools/oracle_longctx_engine_arm.sh`.

### 5.1 The short-context 607-row set (`decode_out/oracle-t2b`, ref logits exist)

```bash
cd /home/nibor1896/Projects/crow-nest
for ARM in layer91-attn-v-out-control layer91-attn-v-out-originals \
           layer91-ffn-down-rule-control layer91-ffn-down-rule-originals \
           layer91-ffn-down-all-control  layer91-ffn-down-all-originals; do
  OUT=decode_out/oracle-t2b/91-${ARM#layer91-}
  systemd-run --user --scope --slice=session.slice --quiet -p MemorySwapMax=0 \
      -p MemoryHigh=56G -p MemoryMax=58G \
      env CROW_CNQ=$PWD/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
          CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json CROW_GRAPH=1 CROW_MMA=1 \
          LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
          CROW_CNQ_OVERLAY=$PWD/converter/${ARM}.cnq \
      engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json "$OUT" \
      > "$OUT.log" 2>&1
done
# the `none` dump of record already exists (decode_out/oracle-t2b/none/gpu-logits.f32);
# re-run it only if the engine binary changed since #78

tools/oracle-kld.py --ref decode_out/oracle-t2b/ref-logits.f32 --rows 0:607 --prompt-rows 596 \
    --arm none=decode_out/oracle-t2b/none/gpu-logits.f32 \
    --arm a-ctl=decode_out/oracle-t2b/91-attn-v-out-control/gpu-logits.f32 \
    --arm a-org=decode_out/oracle-t2b/91-attn-v-out-originals/gpu-logits.f32 \
    --arm c-r-ctl=decode_out/oracle-t2b/91-ffn-down-rule-control/gpu-logits.f32 \
    --arm c-r-org=decode_out/oracle-t2b/91-ffn-down-rule-originals/gpu-logits.f32 \
    --arm c-a-ctl=decode_out/oracle-t2b/91-ffn-down-all-control/gpu-logits.f32 \
    --arm c-a-org=decode_out/oracle-t2b/91-ffn-down-all-originals/gpu-logits.f32 \
    --paired a-org,none --paired a-ctl,none --paired c-r-org,none \
    --paired c-a-org,none --paired c-r-org,c-a-org \
    --json decode_out/91/kld-t2b.json
```

Read it the #79 way: an arm is accepted only if its paired mean KLD vs `none` is negative
beyond 0.025; `a-ctl - none` at 0.000000 proves the wiring (any non-zero there is a kernel
path difference to fix BEFORE quoting the originals arm); `c-r-org - c-a-org` answers
whether llama.cpp's layer rule specifically earns its keep on this architecture.

### 5.2 The #90 long-context form (anchors 1000 and 2564 only — parity collects all rows)

`decode_out/oracle-longctx/engine/a{1000,2564}/{none,kvbf16}/` exist but are EMPTY (the #90
engine arm was interrupted by the machine block), so the baseline runs too. The #90 script
has no overlay passthrough; run the arms in its exact `run_one` shape:

```bash
for A in 1000 2564; do
  IDS=decode_out/oracle-longctx/longctx-170k-a${A}-ids.json
  for ARM in layer91-attn-v-out-originals layer91-ffn-down-rule-originals; do
    OUT=decode_out/oracle-longctx/engine/a${A}/91-${ARM#layer91-}
    mkdir -p "$OUT"
    systemd-run --user --scope --slice=session.slice --quiet -p MemorySwapMax=0 \
        -p MemoryHigh=56G -p MemoryMax=58G \
        env CROW_CNQ=$PWD/converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
            CROW_HOTSETS=$PWD/decode_out/hotsets-M-longctx2100-n160.json \
            CROW_GRAPH=1 CROW_MMA=1 CROW_PINNED_BUDGET_GB=50 \
            LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
            CROW_CNQ_OVERLAY=$PWD/converter/${ARM}.cnq \
        engine/target/release/decode parity "$IDS" "$OUT" > "$OUT/parity.log" 2>&1
    python3 tools/oracle_longctx_rows.py subset --source "$OUT/gpu-logits.f32" \
        --out "$OUT/plan-rows.f32" \
        --rows-file <(python3 -c "import json;print(json.dumps([g for g in json.load(open('decode_out/oracle-longctx/row-plan.json'))['groups'] if g['anchor']==$A][0]['rows']))")
    rm -f "$OUT/gpu-logits.f32"   # the plan rows are what the instrument reads
  done
done
```

Then the reading per `tools/oracle_longctx_engine_arm.sh`'s conventions (f32 oracle rows
for these anchors come from the #90 CPU oracle side; arm names as above, `--paired
91-arm,none` per anchor). If #90's German corpus rows exist by then, the same loop with the
German ids file is the third corpus — issue #91 names German explicitly, and the German
quality probe (`tools/quality-probe.py --label 91-<arm> --base-url …` under
`CROW_CNQ_OVERLAY=…`, #77 §5.4 style, paired against A1) is the symptom-side reading if the
KLD corpus is still missing.

### 5.3 The economics lines (issue item 3)

Adjacent `decode run decode_out/srv-a5-t1read-ids.json 256` per accepted arm, `[budget]`
copied into the report: VRAM delta, planned/resident N, pinned budget (default vs
`CROW_PINNED_BUDGET_GB`), prefill and decode rates, cold experts per token — exactly the
#77 §5.2/§5.3 and #79 §7.3 tables, so the arms' costs sit beside their KLD gains.

## 6. Expected reading, stated in advance so it cannot be moved after the fact

The dense control of #77 bounds everything here: BF16 over the WHOLE dense path closed
15-22 % of the German symptom gap. These arms are subsets of that set (attn-v-out is 30 %
of the dense values, ffn-down-all 3 %), so their ceiling is the #77 result and their
expected paired KLD gain is a fraction of it; anything at or beyond the #77 bound would
point at layer-position concentration being the whole story — worth knowing, and exactly
what the `c-r-org - c-a-org` pair tests. If NO arm clears −0.025, the quantization lever on
the dense path is spent at these shapes and the grid itself (issue item 4) becomes the only
remaining quant question.

## 7. Files of this phase

- `converter/src/layer_rule_overlay.rs` — new subcommand, arm table, `use_more_bits` port,
  self-verification pass, 7 unit tests.
- `converter/src/main.rs` — `mod`, dispatch, HELP line (additive; nothing else moved).
- `converter/src/dense_overlay.rs` — `blocks_to_bf16` made `pub` (one word, no behavior).
- `converter/layer91-*.cnq` — the six artifacts (§3).
- `docs/improve-loop.md` — the hard rule (issue item 1).
- this file.

---

## Orchestrator verification (appended 2026-09-21, all checks passed)

1. Six containers on disk with the reported byte sizes; originals-vs-control deltas consistent (kind-field only). Containers stay untracked (*.cnq ignored, like dense-bf16-originals.cnq) — the acceptance doc + manifest carry identity.
2. `cd converter && cargo test` → 46 passed / 0 failed (was 39; +7 = the layer-rule unit tests).
3. Engine-zero-change claim corroborated: overlay kind is free-form per engine/src/cnq.rs:597-676 refusal table; no engine file modified by this unit (git status).
4. The check_env_docs failure at this instant is correctly attributed: all 11 missing rows are the in-flight wave-2 sampler unit's knobs (CROW_DRY_*, CROW_MIROSTAT_*, CROW_TOP_N_SIGMA, CROW_TYPICAL_P, CROW_XTC_*) — that unit owes the env.md rows at its own acceptance.
5. Blockers verified honest: models/experts/ holds manifests only; dense/dense.safetensors carries the 495 dense tensors, zero routed experts; the engine's #79 refusal table bans bf16 routed experts by construction. Arm (b) and the routed half of (c) are data- and design-blocked, documented — not silently dropped.
6. Post-reboot measurement commands present and concrete (oracle-t2b 607-row paired runs + #90 anchors 1000/2564 against the none baseline, 0.025 paired floor).
