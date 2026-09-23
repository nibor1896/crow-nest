# #91 corruption ladder, 100k context (2026-09-21/22)

tools/corruption-arms.sh with CORRUPTION_CTX_TOKENS=100000 (tools/corruption-probe-long.py,
position end: ~101.8k tokens of seeded session noise, then the 40-literal copy task; 8 rounds,
seeds 1-8, ~102750 prompt tokens per round; rounds 1-3 resumed from the prefix cache, 4-8 cold).
One arm per boot (the #82 pool state). Raw JSONs alongside; ctx984 is the short probe of record.

| arm       | overlay                          | badlines/320 | hexerr | line_err_rate | error seeds       |
|-----------|----------------------------------|-------------:|-------:|--------------:|-------------------|
| baseline  | none                             |            8 |      0 |        0.025  | 2                 |
| attn-ctrl | attn-v-out-control (placebo)     |            2 |      0 |        0.0063 | 2                 |
| attn-arm  | attn-v-out-originals (bf16 v/o)  |           28 |      2 |        0.0875 | 2 (11), 4 (14), 5 (3) |
| rule-arm  | ffn-down-rule-originals          |            2 |      0 |        0.0063 | 2                 |
| all-arm   | ffn-down-all-originals           |            2 |      0 |        0.0063 | 2                 |

Short context (984 tokens, same seeds): every arm 1/320 = 0.0031 (all-arm 2/320) - the floor.

Reading:
- attn-arm - the serve-linux.sh DEFAULT since 723d18f - is the only arm that gets WORSE with depth
  beyond the seed-2 event: 3.5x baseline, errors on three seeds, the only hex-char errors.
- the placebo is weight-identical to baseline yet scores 2 vs 8 on seed 2: that gap is run-to-run
  noise at depth, so rule-arm/all-arm (also 2) are NOT shown to beat baseline - they sit on the placebo.
- errors are almost all DROPPED lines, not wrong digits.
