# GLM-5.3-Flash: entropy census and gate G2

`tools/glm_entropy_report.py` is step 10 of the GLM-5.3-Flash plan (crow-nest #148, parent nibor1896/Crow#362).
It answers gate G2 of `runs/glm53-flash/PREREG.md`: does a lossless static order-0 entropy coder per
routed expert block shrink the expert bytes by at least 8 % including its code tables (option B of the
plan)? `tools/glm-huff-draft/` is the decoder draft the gate also asks for. Neither changes `engine/`,
the converter or a container. Results of record: `runs/glm53-flash/step10/README.md`.

**Verdict of record (2026-10-08, 07:41 UTC, full container):** G2 failed, saving 6.688 % < 8 % incl. tables
(order-0 entropy bound 7.522 %); option B stays unbuilt.

## 1. Commands

```
# the census; the G2 line only on a finished container (exit 0), else PROVISIONAL numbers (exit 3)
python -I tools/glm_entropy_report.py census converter/GLM-5.3-Flash-CNQ4.5.cnq.sidecar.jsonl \
    --decoder-draft runs/glm53-flash/step10/decoder-draft.json [--block 3:0] [--json out.json]

# where routed expert L/E sits in a finished container (absolute offset, length, 4096-aligned or not)
python -I tools/glm_entropy_report.py locate <container.cnq> --layer 3 --expert 0

# the decoder draft on one block: code it, decode it cold N times per thread count, check every decode bit for bit
cd tools/glm-huff-draft && cargo build --release
target/release/glm-huff-draft <container.cnq> <offset> 14155776 --reps 9 --threads 1,2,4,8,16,24 --json out.json

# tests: 16 Python tests (stdlib only), 2 Rust tests
python -I tools/test_glm_entropy_report.py
cd tools/glm-huff-draft && cargo test --release
```

## 2. What is counted

| item | value |
|---|---|
| unit | one routed expert block = gate + up + down of one expert of one layer, 25,165,824 values, 14,155,776 B raw (393,216 NVFP4 blocks of 36 B) |
| census | all 12,096 blocks (MoE layers 3–44 × 288 experts); shared experts reported apart, not in the 12,096 |
| input | the converter's sidecar: per NVFP4 tensor `h_codes` (16 nibble values, sign included) and `h_scales` (256 ue4m3 bytes), counted from the written bytes (`converter/README.md`, "Code histograms") |
| coder | one static table per block, two streams (codes, scale bytes), canonical Huffman length-limited to 12 / 15 bits, each stream rounded up to whole bytes, plus the code-length tables 8 B + 128 B |
| metric | saving = 1 − Σ coded bytes incl. tables / Σ raw bytes over the blocks (PREREG G2) |
| order-0 limit | the order-0 entropy (no tables, no rounding) is printed beside it; a context coder (nibble pairs, neighbouring scale bytes) may save more and costs more decode time; not estimated |
| split | codes vs scale bytes; a scales-only variant (codes left raw) |
| robustness | per layer (min–max over layers and over the blocks of each layer), per class gate / up / down / shared |

The size of a length-limited prefix code is its optimal cost Σ count × length, which is the same for
every optimal code, so the tool computes the cost by the cost-only form of the converter's package-merge
(`converter/src/entropy.rs`: the weights of the 2n − 2 cheapest items after `limit` rounds). On the
step-6 partial container (layer 3, 288 blocks) it equals the converter's own `code_summary`
`expert_blocks` record byte for byte (coded 3,803,962,857 B; test
`test_equal_to_the_converter_on_the_step6_sidecar`), and on a finished container the tool refuses the
verdict unless the two agree again.

Per class the tool prints two forms: one table per tensor (the best static order-0 Huffman at tensor
grain; a table shared per class and layer and applied per expert cannot code a tensor in fewer bits than
that tensor's own optimal table, so it is bounded by this number plus 136 B per tensor, 0.003 %) and one
table per class.

## 3. When it gives a verdict

| state | output |
|---|---|
| container without index trailer, or `<out>.cnq.journal.jsonl` present (the converter removes it after writing the trailer, `converter/README.md` "Resume") | `PROVISIONAL` numbers over the blocks written so far, reasons listed, no `G2` line, exit 3; an unfinished last sidecar line is skipped |
| partial container (`--layers`), fewer than 12,096 blocks, a block without one of its projections, no converter `code_summary` or one that differs | the same: `PROVISIONAL`, exit 3 |
| a histogram whose sum is not the value count (`sum(h_codes) == n`, `sum(h_scales) == n / 16`) | refused, exit 2 |
| finished container, sums checked, 12,096 blocks, converter summary equal, index holds 36,288 routed-expert NVFP4 tensors | `G2 passed / failed / not answered`, exit 0 |

The verdict follows PREREG G2 as written: saving ≥ 8 % including tables AND a decoder draft that names
the decode time per cold visit (`decode_ms_per_cold_visit` in the `--decoder-draft` JSON) → `passed`
(conditional step 19); saving < 8 % → `failed` (option B stays unbuilt); saving ≥ 8 % without a named
decode time → `not answered`.

## 4. The decoder draft (`tools/glm-huff-draft/`)

| item | value |
|---|---|
| where it runs | the CPU staging path: after the NVMe read of a cold expert, before the H2D copy; the engine is not changed |
| format | the G2 coder exactly: one table per block, two LSB-first bit streams; framing apart from the G2 size: 384 segments of 1,024 NVFP4 blocks (36,864 B out), two u32 bit offsets each = 3,072 B per block (0.022 %) |
| decode | a 4,096-entry pair table emits one packed code byte (two nibbles) per lookup when both codes fit 12 bits, else two single lookups; a 32,768-entry table for scale bytes; unaligned 8-byte peeks; T threads on disjoint segments (`std::thread::scope`, spawn and join inside the timing) |
| cold visit | a 512 MiB buffer is written before every timed decode (the CPU's L3 is 36 MiB), so the coded input comes from DRAM; the output buffer is pre-faulted, as a pinned staging buffer is |
| check | every decode is compared with the original block bytes (lossless, bit for bit); the coded size equals the census figure for the same block |
| package-merge | a copy of `converter/src/entropy.rs` `huffman_lengths` (the step-10 file set allows no converter change) |

What a cold visit costs on the CPU path is the decode time against the NVMe read of the same block at
the step-3 one-reader rate, 6.994 GB/s (spread 1.003; `runs/glm53-flash/step03/20261008T001819Z.md`):
14,155,776 B read in 2.024 ms. That rate is not the G1 B: step 3's best reader count (2) failed the
1.15 spread rule, so the G1 input stays "not answered", and 6.994 GB/s is used here only as the measured
one-reader rate. The pinned-RAM path (`stage_cold`, kernel ceiling 31.5 GB/s, `engine/src/gen.rs`) would
need a GPU decoder; none is drafted or measured here.

Measured numbers, conditions and the comparison: `runs/glm53-flash/step10/README.md`.
