# Step 10 — entropy census and gate G2 (crow-nest #148), 2026-10-08

Plan step 10 of the GLM-5.3-Flash series, PREREG `runs/glm53-flash/PREREG.md` section G2 (blob sha256
375de4d8… as committed in `2228d4c`; amendment 1 since then touches G1 only). Tool and method:
`docs/glm-entropy.md`. Commit of the tools: `17e7e46` (branch `glm-step10` on crow-nest `97ce69f`).

## Verdict

**G2 failed: saving 6.688 % < 8 % incl. tables.** Option B (lossless entropy coding of the expert codes)
stays unbuilt; step 19 is not filed; the plan continues on path A. The decoder draft named its decode
time (28.195 ms per cold visit on one core), but the AND fails on the saving.

## Census (`census.txt`, `census.json`)

| item | value |
|---|---|
| input | `converter/GLM-5.3-Flash-CNQ4.5.cnq.sidecar.jsonl`, 43,405,963 B, sha256 `0f26db69947af536eec8d2acf1a584affc92b5891b6da71ae9fbfc0697c2c449`, 36,716 NVFP4 lines, every histogram sum == value count |
| container | `converter/GLM-5.3-Flash-CNQ4.5.cnq`, 178,490,593,097 B, index trailer present, journal absent (stage log 09:38:36 CEST: "all shards converted and deleted", "index trailer written", rc 0) |
| run | 2026-10-08 07:41:40 UTC, `python -I tools/glm_entropy_report.py census <sidecar> --decoder-draft runs/glm53-flash/step10/decoder-draft.json --json runs/glm53-flash/step10/census.json`, exit 0 |
| blocks | 12,096 routed expert blocks × 14,155,776 B = 171,228,266,496 B raw |
| coded | 159,777,173,847 B incl. 1,645,056 B tables: **saving 6.688 %** |
| split | codes 152,202,903,552 → 151,672,590,199 B (0.348 %); scale bytes 19,025,362,944 → 8,102,938,592 B (57.410 %) |
| order-0 entropy bound | 7.522 % (no tables, no rounding): no static order-0 coder per block reaches 8 % |
| scales-only variant | 6.378 % |
| per block | 4.236 % … 7.112 % |
| per layer | 5.387 % (L44) … 6.797 % (L5); L3–L39 6.615 … 6.797 %, falling to L40–L44 6.651 / 6.582 / 6.487 / 6.409 / 5.387 % |
| per class, one table per tensor | gate 6.853 % (5.809 … 7.207), up 6.891 % (5.794 … 7.207), down 6.922 % (5.487 … 7.211); the bound for any static order-0 Huffman at tensor grain, also with tables shared per class and layer |
| per class, one table per class | gate 6.434 %, up 6.561 %, down 6.380 % |
| shared experts (not in the 12,096) | 42 blocks, 5.755 % (4.581 … 6.369 %) |
| check against the converter | the converter's own `code_summary` `expert_blocks` record: equal (blocks, raw, coded, table bytes; entropy) |
| effect had it been built | pinned tier 88.9 instead of 83 experts per layer; 13,209,092 B per NVMe visit = 1.889 ms instead of 2.024 ms at 6.994 GB/s |

Order-0 limit: a context coder (pairs of nibbles, neighbouring scale bytes) may save more and costs more
decode time; not estimated. The codes themselves are at the floor (0.348 %); the slack is in the scale
bytes, which are 11.1 % of a block.

## Decoder draft (`decoder-draft.json`, `decoder-run-*.json`)

Block: layer 3 expert 0 of the step-6 partial container (`models/GLM-5.3-Flash-step06/GLM-5.3-Flash-CNQ4.5-L0-3.cnq`,
offset 3,045,916,672, 14,155,776 B); coded 13,268,642 B (codes 12,548,872, scales 719,634, tables 136; framing
3,072 B apart), equal to `census --block 3:0` on the step-6 sidecar. Every decode lossless (9/9 per row).
Intel Core Ultra 9 285K (8 P + 16 E cores, L3 36 MiB), Windows 11, rustc 1.97.0 release, 9 cold reps (512 MiB
eviction before each), median.

| threads | converter waiting, 01:57:03 UTC (ms) | converter converting on 0.985 cores, 02:00:04 UTC (ms) |
|---|---|---|
| 1 | 28.164 (spread 1.015) | **28.195** (1.011) |
| 2 | 14.490 | 14.341 |
| 4 | 7.496 | 7.577 |
| 8 | 4.383 (1.226) | 4.308 (1.151) |
| 16 | 2.893 (1.213) | 2.784 (1.137) |
| 24 | 2.360 (1.077) | 2.657 (1.357) |
| memcpy raw block, 1 thread | 0.361 | 0.409 |

Other load in both runs: the GLM downloader (python) and the staging driver; CPU load 11 % (first run) and 6 %
(after the second). Against the NVMe read of the same block at the step-3 one-reader rate 6.994 GB/s (spread 1.003):
2.024 ms raw, 1.897 ms coded, 0.127 ms saved per visit. One core decodes in 13.9 × the raw read time; all 24
threads on one block are still slower than reading it raw. 6.994 GB/s is not the G1 B: step 3's best reader
count failed the 1.15 spread rule, so G1's input stays "not answered". No GPU decoder for the pinned-RAM path
was drafted or measured.
