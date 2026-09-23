# Acceptance protocol — issue #90 (oracle-KLD at 100k+ context + English-prose corpus)

- Date: 2026-09-20 (night). Executed by: fleet subagent (build), robin (live acceptance below).
- Issue spec: long-context form from the clean #68 history (1k/50k/100k/158k/178k depth
  sampling), a prose corpus, the KLD-vs-position curve and per-corpus summary in
  `tools/oracle-kld.py`, everything runnable/verifiable. **Scope change 2026-09-20 by
  robin's direct order: ENGLISH ONLY — the German-prose corpus component was dropped
  entirely and replaced by English long-prose under `decode_out/oracle-en/`; nothing
  German was built or fetched.** Reference-mode adjustment (coordinator, same night):
  the f32 checkpoint is gone machine-wide, so the oracle is delivered IMPLEMENTED and
  VALIDATED but not RUN; tonight's reference is paired-baseline mode.
- Companion document: `docs/oracle-longctx.md` (the full record this protocol accepts).

## What exists (all paths repo-relative)

| artifact | what it is | state |
|---|---|---|
| `tools/oracle_longctx_rows.py` | rebuilds the 178,553-id history + 5 probe prompts, VERIFIES them against the recorded gate run; writes the depth row plan; `subset` carves sparse dumps | run; all five id counts exact |
| `decode_out/oracle-longctx/longctx-170k-p{1..5}-ids.json` | the verified ids (178,553 / 178,608 / 178,664 / 178,728 / 178,798) + `ids-manifest.json` with sha256s | written |
| `decode_out/oracle-longctx/row-plan.json` | 6 depth anchors (1000, 2564, 50000, 100000, 158000, 178553) x 64 contiguous rows; also a `--row-groups` file | written |
| `decode_out/oracle-longctx/longctx-170k-a<anchor>-ids.json` | per-anchor prefix ids for the engine arm | written |
| `oracle/ref_longctx_logits.py` | the chunked, resumable f32 oracle (sparse output, PLE carry, DynamicCache, vectorized uid table); `check-weights`, `run`, `verify`, `self-test` | self-test PASS; RUN blocked by the checkpoint gap |
| `tools/oracle-kld.py` | extended: sparse dumps (`.rows.json` sidecar auto-detect), `--row-groups`, `--kld-vs-position`; 55 old tests green, 9 new | done, validated on real dumps |
| `tools/oracle_longctx_test_kld.py` | the 9 new unit tests | green |
| `tools/corpora/` | 5 Gutenberg English books (raw + stripped + `SOURCES.json`, sha256s, strip rule) | fetched, stripped |
| `decode_out/oracle-en/` | 5 x 512 teacher-forced prose rows as id sets + `en-manifest.json` + `en-row-groups.json` | written |
| `tools/oracle_longctx_engine_arm.sh` | crow-engine arm, GPU-flock wrapped, budget-ladder retries, sparse trim, manifests | running in background (fleet RAM squeeze); see live status |
| `tools/oracle_longctx_llama.sh` | llama-arm harness — one command when GGUF shard 1 is whole | harness only (GGUF broken) |
| `decode_out/oracle-longctx/validate/` | the sparse-reader validation against the REAL tf298 dumps (per-row KLD diff exactly 0.0) | done |

## The blocker, precisely (measured, not assumed)

`oracle/ref_longctx_logits.py check-weights` on this machine:
index total_size **359,999,963,128 B**; **131 shard files referenced, 0 on disk**;
**1294/1294 text tensors unloadable**; **33 PLE shards referenced, 0 on disk**. The
Linux tree holds only the 5.17 GB dense originals of #76 and expert manifests; both
HF caches hold only GGUFs; the read-only Windows tree's model dir is 23 MB; Linux
disk has ~26 GB free. Robin's options: **(A)** re-download the 360 GB (479 GB free
on the Windows NTFS partition — robin's call), **(B)** run
`oracle/ref_longctx_logits.py run` on a box that has the checkpoint (resumable,
one command), **(C)** paired-baseline mode without new f32 rows (tonight's mode).
No f32 rows were fabricated; the honesty rule was applied.

## Live acceptance — test cases for robin (exact commands, in order)

Expected wall time: ~2 minutes total, no GPU, no server, no checkpoint needed for
1-5; case 6 needs the GPU free and ~5-10 minutes.

1. **The ids are the recorded history.**
   ```
   .venv-oracle/bin/python tools/oracle_longctx_rows.py ids
   ```
   EXPECT: five lines `probe n: <ids> == recorded prompt_tokens`, then "verified".
   Any MISMATCH aborts (that is the tool refusing to write unverified ids).

2. **The unit tests.**
   ```
   python3 tools/test_oracle_kld.py          # 55 tests, the #78 contract unchanged
   python3 tools/oracle_longctx_test_kld.py  # 9 tests: sparse, groups, curve, subset
   ```
   EXPECT: `OK` twice.

3. **The chunked oracle is the unchunked oracle.**
   ```
   .venv-oracle/bin/python oracle/ref_longctx_logits.py self-test
   ```
   EXPECT: `uid table: identical`, `chunked vs full: ... max |delta logit| < 1e-05,
   same top-1 256/256`, `resumed run: ... `, `SELF-TEST PASS`.

4. **The checkpoint gap is what the docs say.**
   ```
   .venv-oracle/bin/python oracle/ref_longctx_logits.py check-weights
   ```
   EXPECT: `131 referenced, 0 on disk, 131 MISSING`, `the f32 oracle CANNOT run here`,
   exit code 1. (After option A or B, this same command must say "looks complete".)

5. **The sparse reader on the REAL reference.**
   ```
   python3 tools/oracle-kld.py \
       --ref decode_out/oracle-longctx/validate/tf298-ref-sparse.f32 \
       --arm orig=decode_out/oracle-longctx/validate/tf298-orig-sparse.f32 \
       --row-groups decode_out/oracle-longctx/row-plan.json --kld-vs-position 3
   ```
   EXPECT: a summary table over 100 rows with `orig 87.00 +- 3.38 %  0.302573`, the
   `== KLD vs position` block, and depth-group blocks that say "no row of this group
   is in the run" for the anchors outside 0..297 (honest empty, not fake).

6. **The engine arm (if not already finished overnight).**
   ```
   tail decode_out/oracle-longctx/engine-arm.log
   ls decode_out/oracle-longctx/engine/a*/none/plan-rows.f32 2>/dev/null
   ```
   The background script retries for up to 8 h against the fleet's RAM squeeze; when
   it has the card you should see `anchor 1000 arm none: plan rows written` and the
   same for `kvbf16` and anchor `2564`. Then the first REAL paired reading:
   ```
   python3 tools/oracle-kld.py \
       --ref decode_out/oracle-longctx/engine/a2564/none/plan-rows.f32 \
       --arm kvbf16=decode_out/oracle-longctx/engine/a2564/kvbf16/plan-rows.f32 \
       --paired kvbf16,none --kld-vs-position 4
   ```
   EXPECT: KL(none || kvbf16) over the first sparse rows with a sign test — the
   first long-context noise-floor datum of this engine.

7. **The English corpus.**
   ```
   .venv-oracle/bin/python tools/oracle_longctx_corpus.py emit
   ```
   EXPECT: five `... -> slice ... (512 rows)` lines, idempotent; then
   `cat tools/corpora/SOURCES.json` shows five sha256s and the strip rule.

## Honest gaps (what this acceptance does NOT claim)

- No f32 reference rows at any depth exist yet (checkpoint gap; options A/B/C above).
- No engine or llama dump exists above anchor 2564: `decode parity` and
  `llama-row-probs.py` both write rows 0..last contiguously (50 GB at anchor 50000);
  both need a row-window write mode — an engine/tools change outside this issue's
  ownership, documented here as the follow-up.
- The llama arm has not run: GGUF shard 1 is broken; `tools/oracle_longctx_llama.sh`
  is the one command for when it is whole.
- No KLD number at 100k+ is quoted anywhere, because none has been measured.

---

## Orchestrator verification (appended 2026-09-21, all checks passed)

1. Deliverables present: 5 scripts + ref oracle + 2 docs + corpora (SOURCES.json with sha256s) + decode_out/oracle-longctx/ (verified ids + manifests + row-plan) + decode_out/oracle-en/.
2. Test suites re-run by orchestrator: `tools/oracle_longctx_test_kld.py` OK (new), `tools/test_oracle_kld.py` OK (the OLD suite — the extend-without-breaking instruction held).
3. Provenance discipline verified: the ids builder aborts on gate-run mismatch and the manifest carries sha256s; the five probe lengths (178,553/608/664/728/798) match the #68 gate record.
4. The engine-arm "parity FAILED" at anchor 2564/kvbf16 investigated: NOT an instrument fault — the engine's planner correctly refused the config under the fleet's RAM squeeze (manager.rs:203, "no hot-set size fits BOTH budgets"); the script's documented 10-min retry ladder then backed off. Instrument behavior as designed.
5. f32 blocker independently confirmed earlier by orchestrator (1 of 131 shards machine-wide; Downloads dir empty; HF caches GGUF-only). No rows fabricated — paired-baseline mode is the standing mode per robin's no-download order.
6. Footprint: only `tools/oracle-kld.py` modified among pre-existing files (assigned); engine/ untouched by this unit.
7. Tracked vs ignored: ids jsons + manifests + row-plan + engine-arm.log + corpora tracked (~17 MB text); engine dumps (185 MB) and validate replay dumps (190 MB) stay ignored, results documented.

Live-acceptance note: the engine arm keeps running overnight via the RAM ladder (PID file /tmp/fleet-monitor/90-oracle.pids); case 5 of the seven cases was verified against an actual run by the agent, cases re-runnable one-command from docs/acceptance/issue-90.md.

---

## Addendum 2026-09-21 12:10 — the engine arm: 3 of 4 runs done, the 4th is a VRAM edge, documented

- DONE and hashed: a1000/none, a1000/kvbf16, a2564/none (`plan-rows.f32` + `.rows.json` + `SHA256SUMS` each, dense dumps hashed then removed — the marker the arm script respects).
- BLOCKED: **a2564/kvbf16**. The engine refuses config: with bf16 KV the state side grows ~2.5 GiB (states are sized by the 200k CONTEXT FLOOR, not the 2565-row prompt), and the 5090's ~22.2 GiB free VRAM at a running desktop leaves the planner no N (the fp8 arm booted with 0.57 GiB to spare at n=153). Measured facts: `CROW_CHUNK` is IGNORED by parity mode (chunk = round4(prompt len), empirically verified 12:08 with the env set and the boot line still reading 2565), so the staging lever does not exist there; `CROW_RAM_MARGIN_GB=1` works (three runs booted at MemAvailable 12–20 GiB — free_for_pin counts the reclaimable pool — the pin gate forces the release on demand).
- The clean engine-side fix (follow-up, not tonight): parity mode should respect a context cap for short prompts (states sized to the prompt, not CONTEXT_FLOOR) — one config line, gate re-run, byte-identity trivially held (states beyond the prompt are never read).
- The instrument is USABLE now: the paired-baseline at anchor 1000 has both arms, anchor 2564 has the none arm; the missing cell only narrows the bf16-vs-fp8 comparison at the sparse-QSA boundary, it blocks nothing else.
- Run evidence: `/tmp/90-arm*.log` (robin's terminal) + `decode_out/oracle-longctx/engine/*/SHA256SUMS`.
