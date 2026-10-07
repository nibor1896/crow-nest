# PREREG: G3b, the q8 KV of the dense 27B against BF16 with a noise control (crow-nest #88)

Written 2026-10-07, before any run of this form. robin chose the criterion on 2026-10-07 ("Dein Vorschlag ist genau gut … Das können wir so machen").

## Why a second G3

G3 of `decode_out/q8-kld/PREREG.md` (`23f2fc0`, amendment 2 `63f7f50`) failed on 2026-10-07: mean KL(BF16 ‖ q8) 1.963 (1k) and 0.532 (2.5k) against the 0.073 line, pass at 50k-178k. That form had no noise control. Measured afterwards on the same anchors (raw: `crow-lab/runs/q8-gate-27b-win-20261007/`), BF16 against BF16 with only the attention kernel changed (`CROW_P2_FA=0`, the split kernel, against the default FA kernel; same BF16 cache, other rounding order) gives 0.836 (1k), 0.057 (2.5k), 0.0029 (50k), 0.00047 (100k). At 1k the harmless rounding change alone breaks 0.073. The G3 verdict stands as recorded; this form does not overturn it, it asks the question again on rows nobody has read.

What was seen before writing this: the four q8 / noise ratios above are 2.3, 9.3, 15.0 and 5.7. With the criterion below, the old anchors would have failed at 3 of 4.

## Form

- Session: the #90 / #68 libghost history, `decode_out/oracle-longctx/longctx-170k-a178553-ids.json` (178,552 ids).
- Anchors (new, not used by any earlier run): 1500, 6000, 25000, 75000, 130000, 170000. Ids are the first A+1 ids of that session, as the old anchor files are (checked: every old file equals the same prefix). Files and sha256 (first 16 hex): `ids/a1500-ids.json` 65ffeae0221aee70, `a6000` 23a037fe21b2e748, `a25000` f5b2ba3737be5f65, `a75000` b1b3920e6ecf332a, `a130000` a18bad72892c443b, `a170000` 7bef343351c13732. Their 64 tail rows overlap no tail of the old anchors.
- Rows: the tail form, `CROW_PARITY_TAIL=64`, KL over dump rows 0..63 (n = 64 per anchor), `tools/oracle-kld.py`, natural log.
- Arms, all `decode.exe parity` from `engine/target/release` (`cfbf7ba` = v0.10.0, engine unchanged since), the installed 27B container (sha256 ba014aeb…), `CROW_MMA=1 CROW_GRAPH=1 CROW_CONTEXT=180000`:
  - **R** reference: `CROW_KV=bf16`, default kernels (FA).
  - **N** noise: `CROW_KV=bf16`, `CROW_P2_FA=0` (split kernels).
  - **Q** candidate: `CROW_KV=q8`, default kernels (FA), the configuration Crow would ship.
- Order: anchor by anchor, R then N then Q; one engine on the card at a time.

## Criterion (fixed here)

Per anchor: noise = mean KL(R ‖ N), q8 = mean KL(R ‖ Q). The anchor passes when

    q8 ≤ max(0.073, 3 × noise)

G3b passes when all six anchors pass. One failing anchor fails G3b. A run that does not finish (rc ≠ 0, NaN in a dump) fails its anchor.

## What follows

- G3b pass: G4 (200k boot with `CROW_KV=q8`) and G5 (speed) as in the first PREREG, then robin's acceptance; only then may Crow's `stack.json` flip the 27B to 200,000.
- G3b fail: the 27B stays at 131,072; the finding goes to #88.
