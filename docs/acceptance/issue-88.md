# Acceptance protocol — issue #88 phase 1 (KV cache precision: e4m3-LUT vs bf16 vs non-LUT, measurement only)

- Date: 2026-09-20 (late). Executed by: fleet subagent (issue #88 unit). Repo at `e27f005`, decode binary of 2026-09-20 08:28 (fresher than every source file in the binary; the only newer sources — `serve.rs`, `ctx_reset_probe.rs` — belong to a parallel unit and are not in this binary).
- Scope: PHASE 1 IS MEASUREMENT ONLY. No engine source was changed, no kernel was added, `CROW_KV` was NOT plumbed through `serve` (that file belongs to a parallel unit). Phase 2 (per-256-element block scales for e4m3, or a bf16-KV-for-12-layers planner option) is implemented nowhere — the issue's own trigger condition ("If fp8 moves KLD") was not met. What is accepted here is the measurement record and its verdict.

## The instrument

`tools/oracle-kld.py` against `decode_out/oracle-t2b/` (reference 611 rows = 611 x 248,320 f32, analysis rows `0:607`, prompt-rows 596), exactly as `docs/oracle-kld.md` §8 prescribes. Paired noise floor of the instrument (§6): **0.025 mean KLD / 1.5 pp top-1**. Arms are `decode parity` dumps (rows x vocab f32); every engine run went through `flock /tmp/crow-gpu.lock`, one engine at a time, cgroup `MemoryHigh=56G MemoryMax=58G`. Full commands, env and per-run notes: `decode_out/kv-ab/log.md`.

## Why two matrices (the LUT lever is invisible on the standard form)

On the standard parity form, rows 0..606 come from the PREFILL path (`attn_sel_s8l` — always LUT) and only rows 607..610 come from the decode path; analysis stops at 607, so `CROW_ATTN_LUT` can move nothing the instrument reads. The assigned matrix (A) was run anyway — it reproduces the arms of record and carries the bit-parity proof — and a second matrix (D) was collected on the teacher-forced form (`CROW_GRAPH=0 CROW_PARITY_PREFILL=8`, the `p8tf` gate form): rows 0..7 prefill, rows 8..606 teacher-forced DECODE steps, so 599 analyzed rows run `attn_sel_split_l` vs `attn_sel_split` and the decode-path `store_kv`. A third probe (L) ran `decode layercheck3` against the f32 layer-3 golden, per KV dtype, with and without the BF16-originals overlay — the issue's own "expected result" names the 2.9 % layer-3 residual.

## Matrix A — standard form (the assigned matrix)

| arm | env on top of the none row | sha256 (first 16) of gpu-logits.f32 | N | mean KLD, 607 rows | same top-1 |
|---|---|---|---|---|---|
| A1 default (fp8 KV + LUT) | — | `79a4d2b682b4db9e` | 158 | 0.460689 ± 0.048404 | 82.37 ± 1.55 % |
| A2 fp8 KV + LUT off | `CROW_ATTN_LUT=0` | `79a4d2b682b4db9e` | 158 | 0.460689 ± 0.048404 | 82.37 ± 1.55 % |
| A3 bf16 KV + LUT on | `CROW_KV=bf16` (+`CROW_PINNED_BUDGET_GB=48`) | `04ba4ee86ed4e609` | 138 | 0.472301 ± 0.051684 | 82.04 ± 1.56 % |
| A4 bf16 KV + LUT off | `CROW_KV=bf16 CROW_ATTN_LUT=0` (+`CROW_PINNED_BUDGET_GB=48`) | `04ba4ee86ed4e609` | 139 | 0.472301 ± 0.051684 | 82.04 ± 1.56 % |

Paired, per row (KLD(A) − KLD(B), 607 rows):

| A − B | mean difference | A worse on | sign test p | vs floor 0.025 |
|---|---|---|---|---|
| A2 − A1 | **0.000000 ± 0.000000** | 0/0 | 1 | byte-identical dumps |
| A3 − A1 | +0.011612 ± 0.014840 | 287/607 | 0.194 | inside, wrong direction |
| A4 − A1 | +0.011612 ± 0.014840 | 287/607 | 0.194 | inside, wrong direction |
| A4 − A3 | **0.000000 ± 0.000000** | 0/0 | 1 | byte-identical dumps |

Byte identities (whole 611-row files, `cmp`): **A1 == the `none` arm of record** (`79a4d2b682b4…`, the §6 determinism sha) — the baseline reproduces across days. **A3 == the `kvbf16` arm of record** (`04ba4ee86ed4…`) although the loader clamped N to 138 vs the record's 142 — hot/cold expert placement is proven numerics-neutral for this form. The paired A3−A1 row reproduces the #78 `kvbf16 − none` row of `docs/oracle-kld.md` §5.4 to the sixth decimal.

## Matrix D — teacher-forced decode path (`CROW_GRAPH=0 CROW_PARITY_PREFILL=8`)

| arm | env on top of D1 | sha256 (first 16) | N | mean KLD, 607 rows | same top-1 |
|---|---|---|---|---|---|
| D1 decode-path default | — | `542850166419454b` | 160 | 0.476149 ± 0.051302 | 82.54 ± 1.54 % |
| D2 decode-path LUT off | `CROW_ATTN_LUT=0` | `542850166419454b` | 160 | 0.476149 ± 0.051302 | 82.54 ± 1.54 % |
| D3 decode-path bf16 KV | `CROW_KV=bf16` (+`CROW_PINNED_BUDGET_GB=48`) | `87ccfd3de2ee5ee3` | 143 | 0.465485 ± 0.051410 | 82.21 ± 1.55 % |
| D4 decode-path bf16 + LUT off | `CROW_KV=bf16 CROW_ATTN_LUT=0` (+`CROW_PINNED_BUDGET_GB=48`) | `87ccfd3de2ee5ee3` | 143 | 0.465485 ± 0.051410 | 82.21 ± 1.55 % |

Paired, per row:

| A − B | mean difference | A worse on | sign test p | vs floor 0.025 |
|---|---|---|---|---|
| D2 − D1 | **0.000000 ± 0.000000** | 0/0 | 1 | byte-identical dumps |
| D3 − D1 | **−0.010664 ± 0.018694** | 240/607 | 2.86e-07 | inside (right direction, 0.43x floor) |
| D4 − D1 | −0.010664 ± 0.018694 | 240/607 | 2.86e-07 | inside |
| D4 − D3 | **0.000000 ± 0.000000** | 0/0 | 1 | byte-identical dumps |

Context: D1 − A1 (decode-path form vs prefill-path form, same ids) = +0.015460 ± 0.011911 — the collection form itself moves the reading by more than the bf16 KV does, and it too is inside the floor. The D3 sign test is read the way §6 reads its own sign tests: on 607 correlated rows a "significant" sign test with a mean difference inside its own floor decides nothing.

## Probe L — the layer-3 attention sub-block vs the f32 golden (`decode layercheck3`, cwd `engine/`)

| run | weights | KV | rel_L2 | max_abs | corr |
|---|---|---|---|---|---|
| L1 | plain FP4 | fp8 | 0.1282 | 0.4473 | 0.99179 |
| L2 | plain FP4 | bf16 | 0.1282 | 0.4473 | 0.99179 |
| L3 | BF16 originals overlay | fp8 | **0.0290** | 0.1092 | 0.99958 |
| L4 | BF16 originals overlay | bf16 | **0.0290** | 0.1092 | 0.99958 |

L3 reproduces the 2.9 % residual of record (the "engine numerics" residual of `docs/expert-requant.md` §8). L4 is identical at print precision: **the 2.9 % layer-3 residual is not the KV store/decode path** — doubling KV precision moves it by less than one digit of the printed value. Stepwise (single-token, persistent KV) equals batched in every run (the p11/p12 pin).

## Verdict

1. **The e4m3 LUT decode (`attn_sel_split_l`, default since #61g) is bit-identical to the non-LUT kernel by MEASUREMENT**, not only by construction: byte-identical dumps on the standard form (all 611 rows incl. the 4 decode rows), on the decode-path form (603 decode rows), and under bf16 KV. It is a pure speed lever (-6.28 % per token, #61g); it owes no quality gate and it has no quality cost. `CROW_ATTN_LUT=0` stays what it is — the reachable fallback of record.
2. **bf16 KV vs fp8 e4m3-no-scale KV is inside the instrument's noise floor in BOTH collection forms**: +0.0116 ± 0.0148 (standard form, the wrong direction) and −0.0107 ± 0.0187 (decode path, the right direction) against a 0.025 floor — and the two forms disagree on the SIGN, which is the floor talking. Same top-1 moves 0.33 resp. 0.33 pp against a 1.5 pp floor. The layer-3 golden agrees: nothing moves at print precision. **The KV cache precision — both the storage dtype and the decode LUT — is measured innocent** of CNQ4.5-M's distance to the f32 oracle on this form, and of the 2.9 % layer-3 residual.
3. **Phase 2 is NOT implemented.** The issue's trigger ("If fp8 moves KLD: add per-256-element block scales … or flip the 12 full-attention layers to bf16 KV") did not fire: no arm cleared 0.025. Implementing scales or a planner option anyway would spend kernel/layout work on a lever this instrument cannot justify — the measurement-first rule of the ticket is what rules here.
4. **What this does NOT decide** (and says so, per the issue's own context): the 100k–200k sparse QSA regime — the form the #68 degeneration flip lives in — is not reachable by this instrument (needs #T8); the kvbf16 control of #78 was already known to be short-context-only, and matrices A/D/L inherit that limit unchanged. `CROW_KV` remains decode-only (serve does not read it); plumbing it through serve only becomes interesting once a long-context instrument exists that could see a difference.

## Deliverables

| artifact | what it is |
|---|---|
| `decode_out/kv-ab/log.md` | the run log: every command with env, GPU-lock discipline, the between-run lazy-pool recovery protocol, byte-parity table, verdict |
| `decode_out/kv-ab/{a1-none,a2-lut0,a3-kvbf16,a4-kvbf16-lut0}/` | matrix A dumps (`gpu-logits.f32`, `gen-sequence.json`) + `.log` per run |
| `decode_out/kv-ab/{d1-tf-none,d2-tf-lut0,d3-tf-kvbf16,d4-tf-kvbf16-lut0}/` | matrix D dumps (teacher-forced decode path) |
| `decode_out/kv-ab/l[1-4]-lc3-*.log` | the layercheck3 runs |
| `decode_out/kv-ab/kld-a.json`, `kld-a-per-row.json`, `kld-d.json`, `kld-d-per-row.json` | the instrument's summaries and per-row details |
| `decode_out/kv-ab/balloon.py` | the operational lazy-pool recovery helper (documented #15/#82 driver behaviour; no engine touch) |
| `docs/acceptance/issue-88.md` | this protocol |

No repository source file was modified. `tools/oracle-kld.py` was used as-is (not edited — owned by a parallel unit).

## Exact repro (from the repo root; every engine run under the GPU lock)

```
# --- matrix A (standard form) -------------------------------------------------
flock /tmp/crow-gpu.lock -c "systemd-run --user --scope --slice=session.slice --quiet \
    -p MemorySwapMax=0 -p MemoryHigh=56G -p MemoryMax=58G \
    env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
        CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json \
        CROW_GRAPH=1 CROW_MMA=1 \
        LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
    engine/target/release/decode parity decode_out/oracle-t2b/t2b-tf-ids.json \
        decode_out/kv-ab/a1-none" > decode_out/kv-ab/a1-none.log 2>&1
#   A2 adds CROW_ATTN_LUT=0 and out a2-lut0
#   A3 adds CROW_KV=bf16 CROW_PINNED_BUDGET_GB=48 and out a3-kvbf16
#   A4 adds CROW_KV=bf16 CROW_ATTN_LUT=0 CROW_PINNED_BUDGET_GB=48 and out a4-kvbf16-lut0

# --- matrix D (decode-path form) ----------------------------------------------
#   same wrapper with CROW_GRAPH=0 CROW_PARITY_PREFILL=8 instead of CROW_GRAPH=1:
#   D1 no extra env; D2 + CROW_ATTN_LUT=0;
#   D3 + CROW_KV=bf16 CROW_PINNED_BUDGET_GB=48; D4 + all three

# --- between runs: return the lazy pinned pool (see log.md; #15/#82) ----------
python3 decode_out/kv-ab/balloon.py 46

# --- the reading (no GPU) ------------------------------------------------------
python3 tools/oracle-kld.py --ref decode_out/oracle-t2b/ref-logits.f32 --rows 0:607 \
    --prompt-rows 596 \
    --arm a1-none=decode_out/kv-ab/a1-none/gpu-logits.f32 \
    --arm a2-lut0=decode_out/kv-ab/a2-lut0/gpu-logits.f32 \
    --arm a3-kvbf16=decode_out/kv-ab/a3-kvbf16/gpu-logits.f32 \
    --arm a4-kvbf16-lut0=decode_out/kv-ab/a4-kvbf16-lut0/gpu-logits.f32 \
    --paired a2-lut0,a1-none --paired a3-kvbf16,a1-none \
    --paired a4-kvbf16-lut0,a1-none --paired a4-kvbf16-lut0,a3-kvbf16 \
    --per-row decode_out/kv-ab/kld-a-per-row.json --json decode_out/kv-ab/kld-a.json
#   the D matrix identically with the d*-tf-* arms -> kld-d.json

# --- probe L (layer-3 golden; cwd engine/) ------------------------------------
cd engine && flock /tmp/crow-gpu.lock -c "systemd-run --user --scope --slice=session.slice --quiet \
    -p MemorySwapMax=0 -p MemoryHigh=56G -p MemoryMax=58G \
    env CROW_GRAPH=0 CROW_MMA=1 LD_LIBRARY_PATH=$HOME/.local/share/crow/cuda/lib \
    ../engine/target/release/decode layercheck3"
#   L2 + CROW_KV=bf16 CROW_PINNED_BUDGET_GB=48
#   L3 + CROW_CNQ_OVERLAY=/home/nibor1896/Projects/crow-nest/converter/dense-bf16-originals.cnq CROW_PINNED_BUDGET_GB=49
#   L4 + both
```

Machine notes for whoever re-runs this: the loader's host budget is MemAvailable-based whenever any other CUDA process is alive (a desktop chromium was all day), and after every engine exit the ~46 GiB pinned tier sits in the driver's lazy pool until memory pressure returns it — `decode_out/kv-ab/balloon.py` (deep pass, holds ~30 s at ~1 GiB) is the reliable form; shallow balloons sometimes do not release it. bf16-KV arms also need ~2.3 GiB more VRAM: with a desktop holding ~1 GiB the two-sided planner refuses at the default 46 GiB host cap, so the bf16 arms pin 48 (the doc §8 overlay precedent); the resulting N difference (138-143 vs 158-160) is proven numerics-neutral by the A3 byte-identity against the N=142 arm of record.

## Test cases for robin's live acceptance

| # | what to check | how | expected |
|---|---|---|---|
| 1 | the baseline really is the arm of record | `cd /home/nibor1896/Projects/crow-nest && sha256sum decode_out/kv-ab/a1-none/gpu-logits.f32 decode_out/oracle-t2b/none/gpu-logits.f32` | both `79a4d2b682b4…` |
| 2 | the LUT lever is bit-identical | `cmp decode_out/kv-ab/a2-lut0/gpu-logits.f32 decode_out/kv-ab/a1-none/gpu-logits.f32 && cmp decode_out/kv-ab/d2-tf-lut0/gpu-logits.f32 decode_out/kv-ab/d1-tf-none/gpu-logits.f32` | both silent (identical) |
| 3 | the A-matrix KLD is the #78 number | `python3 -c "import json;d=json.load(open('decode_out/kv-ab/kld-a.json'));print(d['paired'])"` and diff against the `kvbf16 - none` row of docs/oracle-kld.md §5.4 | +0.0116 ± 0.0148, 287/607, p 0.194 |
| 4 | the decode-path bf16 number | same on `kld-d.json` | d3−d1 = −0.010664 ± 0.018694 |
| 5 | the layer-3 residual is KV-independent | `grep rel_L2 decode_out/kv-ab/l3-lc3-orig-fp8.log decode_out/kv-ab/l4-lc3-orig-bf16.log` | both 0.0290 / 0.0290 |
| 6 | nothing in the engine changed | `git status --short engine/src` | only the parallel units' files (serve.rs, ctx_reset_probe.rs, Cargo.toml), none of them this unit's |
| 7 | (spot) re-run arm A1 with the repro command above after a balloon | sha256 | `79a4d2b682b4…` — run-to-run zero, the §6 discipline |

## Remainder of #88 (not accepted here)

- Phase 2 (per-block e4m3 scales or bf16-KV-for-12-layers planner option): BLOCKED on a long-context instrument (#T8 form) that can actually see the 100k–200k regime — this protocol measured the trigger condition and it did not fire at 607 tokens.
- Plumbing `CROW_KV` through serve: deferred with it — serve.rs belongs to a parallel unit in this wave, and a serve knob has no measured effect to expose until #T8 says otherwise.

---

## Orchestrator verification (appended 2026-09-21, all checks passed)

1. Artifact numbers cross-checked in `decode_out/kv-ab/kld-a.json` / `kld-d.json`: a1 `kld_mean` 0.460688827693011 / same_top1 500 of 607 = 82.37 % — byte-for-byte the values of record; a2 identical to a1 in EVERY statistic (bit-identity corroborated numerically); a3 delta +0.011612; a4 = a3. D-matrix present with its own summaries.
2. Change discipline: `git status` shows zero engine-source changes from this unit (the tree's remaining modifications belong to robin's #82 and the parallel #90 instrument extension of `tools/oracle-kld.py`, which was assigned to it).
3. Phase 2 correctly NOT implemented — the issue's own trigger ("if and only if an arm moves KLD beyond the threshold") did not fire; no arm cleared 0.025.
4. Baseline integrity: A1 proven byte-identical to the `none` arm of record (dump hash `79a4d2b682b4…`) — the matrix measured what it claims.
5. Artifacts: 4.6 GB of dumps stay git-ignored (house practice — `decode_out/oracle-t2b` is untracked alike); the SMALL evidence (log.md with full commands + env + balloon protocol, both kld summaries, per-row files, balloon.py) is whitelisted and tracked with this acceptance.
6. GPU-lock discipline: every run under `flock /tmp/crow-gpu.lock` per log.md — no interference with the parallel #90 engine-arm runs.

Live-acceptance note for robin: spot-checks 1/2/3/5 of the seven cases above were re-run green during acceptance; the remaining cases (4, 6, 7) involve the bf16 planner-budget interaction and are one-command re-runs from log.md.
