# Regression gate R

`tools/regression_gate.py` checks that an engine change left the two shipped models unchanged: Flash-Next and the dense 27B, with MTP on (the 27B's default). It came out of crow-nest #174, plan step 7 under root #169. Every GLM engine step runs it before it lands.

**Red means: report the cause.** Never change a default, a parameter, a hot set or an env value to turn a check green. A red R2 is a numeric change, and the change that caused it is the finding.

## Commands

Run from the checkout that holds the containers (`converter/*.cnq`) and the prompt files (`decode_out/*.json`). `--root` names the tree that is tested and built.

```
git worktree add ../crow-nest-r-before <before-sha>
python tools/regression_gate.py before  --session S --root ../crow-nest-r-before
python tools/regression_gate.py after   --session S --root .
python tools/regression_gate.py compare --session S
```

- **`before` / `after`:** record the tree's HEAD and dirty state, then:
  1. run R1;
  2. run `cargo build --release --bin decode`;
  3. copy the binary into the session;
  4. run R2 with that copy.
- **`after` also runs R3:** it needs the before binary of the same session, or of the session `--before-session` names.
- **`compare`:** pure, no GPU. Writes `compare.json`, prints one GREEN/RED line per check, and writes the R4 protocol if it is missing.
- **GPU use:** `before` and `after` boot both models many times. On robin's machines they run only after his go.

| option | default | meaning |
|---|---|---|
| `--session` | today's date | folder under `--out` and `--raw` |
| `--out` | `runs/regression-gate` | JSON results: `before.json`, `after.json`, `compare.json`, `r4-protocol.md` |
| `--raw` | `decode_out/regression-gate` (git-ignored) | logits dumps (~0.5 GB per item), logs, the two binaries |
| `--root` / `--data-root` | this checkout | the tree to test and build / where containers and prompts resolve (the cwd of `decode`) |
| `--models` | `flash-next,qwen27b` | subset |
| `--skip` | none | `r1`, `r2`, `r3` |
| `--before-session` | the same session | read `before.json` and the before binary from another session |
| `--pairs` | 3 | R3 A/B and A/A pairs |
| `--no-build` | off | use the tree's existing `engine/target/release/decode` |
| `--gpu-used-max-mib` | 2000 | refuse an engine start on a busy GPU (`tools/gate-linux.sh` precheck) |
| `--ram-gate-gib` | 50.5 on Windows, 0 on Linux | free host RAM before an engine start (`docs/architecture.md` 0.5 rule 1) |
| `--config` | built in | JSON with `vocab` and `models`, the same shape as `DEFAULT_CONFIG` |

**Child environment:** every inherited `CROW_*` variable is stripped, and only the configured ones are set. The env of record is `CROW_GRAPH=1 CROW_MMA=1`. Flash-Next uses `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` with `hotsets-M-longctx2100-n160.json`; the 27B uses `Qwen3.8-27B-CNQ4.5.cnq`.

**Exit codes:**
- **`compare`:**
  - 0: all green and R4 PASS;
  - 1: any red;
  - 3: R1-R3 green with R4 pending, or a part skipped.
- **Any subcommand:** 2 on a setup error (missing input, failed build).

## What each check compares

| check | form | red when |
|---|---|---|
| R1 | `cargo test --release` in `engine/` and `converter/`. Counts are the sum of libtest's `test result:` lines | either side exits non-zero, counts a failure, prints no summary line, or the after side passes fewer tests than before |
| R2 `logits512` | `decode parity decode_out/real512-ids.json` | any byte of `gpu-logits.f32` differs; the cause names the offset, row, column and both f32 values. Also red if the ids in `gen-sequence.json` differ |
| R2 `logits512-tf` | the same with `CROW_GRAPH=0 CROW_PARITY_PREFILL=8`, the decode path teacher-forced | as above |
| R2 `greedy512` | `decode run decode_out/parity-ids.json 512` | the 512 greedy ids differ |
| R2 `mtp512` (27B) | `decode mtpspec decode_out/parity-ids.json 512 3`, the #95 C2 form | `trace-plain` differs; any of the three C2 lines is not `true` on either side; the draft counters (passes, acceptance, k histogram) differ |
| R2 `mtp-logits` (27B) | `decode mtpgolden` over `logits512`'s sequence | any byte of `mtp-gpu-logits.f32` differs |
| R3 | `decode run` `mean_ms`, and for the 27B also the `mtpspec-step` rate as ms = 1000 / tok/s | \|mean(N) - mean(B)\| over the A/B pairs exceeds the A/A window, faster or slower. Also red if the ids differ between timed runs |
| R4 | `r4-protocol.md`, filled in by a human | `R4 verdict: FAIL` |

**How R3 runs:**
- The order is: warm-up (discarded), then AB1 AA1 AB2 AA2 AB3 AA3. One fresh process per run, one engine at a time.
  - A/B pair: before binary, then after binary.
  - A/A pair: before binary twice.
- The A/A window is max - min of the six A/A runs. It is the "B spread window" of the adjacent-pair tables in `docs/architecture.md`, measured in the same session.

**R4: Crow's real path.** Per model, the human:
1. starts `serve` on port 8099;
2. runs `python cli\crow_gui.py --base-url http://127.0.0.1:8099/v1` and judges at least 3 turns;
3. checks that `git -C <Crow> diff --stat -- manifests/` prints nothing.

Then set `R4 verdict: PASS` or `FAIL` in the file. `compare` never overwrites a filled protocol.

## Self-test

The gate is only trusted after it has caught a deliberate change.
1. In a throwaway worktree, raise the decode attention split count `pub const ATTN_SPLITS: usize = 8;` in `engine/src/gen.rs` to 16, without committing. The const has to be edited: the gate strips a runtime `CROW_ATTN_SPLITS`.
2. Run:
   ```
   python tools/regression_gate.py after   --session S-selftest --root ../crow-nest-r-selftest --before-session S --skip r1,r3
   python tools/regression_gate.py compare --session S-selftest --before-session S
   ```
3. R2 must be RED on the decode-path items (`logits512-tf` at least) for both models. If it stays green, the gate is broken.

## Limits

- Before and after must run on the same machine, driver and toolchain. A different NVRTC/driver JIT moves logit bytes by itself (`tools/gate-linux.sh` header, the Linux vs Windows 512-row values).
- The prompts are fixed and short (8 and 512 ids), so long-context paths (QSA at more than 2,048 tokens, the 27B at 30k) are not covered. A change aimed at those needs its own measurement beside the gate.
- `tools/test_regression_gate.py` covers the comparison logic only, with fixtures. The `decode` calls themselves are exercised by the first real run.
