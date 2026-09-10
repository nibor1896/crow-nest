# Changelog

- Format: one section per version, subsections Added, Changed, Measured, Known limitations.
- Every item names its issue number in `crow-nest`.
- Every number names its date; the machine is `docs/system-landscape.md` unless another one is named.

## v0.1.0 (unreleased, tag follows E9)

- Scope: one model, one GPU, one client, Windows.
- Branch `release-v0.1`, never pushed as of 2026-09-11.
- E9 is the tag step of the release plan under issue #1; this section is written before it.

### Added

#### Stage A, the HTTP server (`engine/src/bin/serve.rs`)

- `#23` A1: spec section 7, the prefix cache design for KV, QSA ring, GDN and PLE.
- `#24` A2: `GET /health` and `GET /props`, the operating point as a JSON document.
- `#25` A3: in-engine tokenizer and chat template, ids identical to `tools/tokenize_ids.py --chat`.
- `#26` A4: streaming `POST /v1/chat/completions` with content deltas and `[DONE]`.
- `#27` A5: `usage` and `timings` on the final chunk, behind `stream_options.include_usage` and `timings_per_token`.
- `#28` A6: sampling and EOS stop on the server path, per-request seed, `min_p` accepted and ignored.
- `#29` A7: `tool_calls` streamed in fragments a client can reassemble, parser in `engine/src/toolcall.rs`.
- `#30` A8: expert counters in the `timings` block, cumulative.
- `#31` A9: the prefix cache reuses KV, QSA ring, GDN and PLE conv across turns.
- `#32` A10: `GET /slots` and `POST /slots/0?action=save|restore`, behind `--slot-save-path`.
- `#33` A11: spec section 7 corrected against the built server, `cargo test` green in engine and converter.

#### Stage B, the client contract

- `#39` B3a: `stream:false` answers one `chat.completion` document from the same generation loop as the SSE path.

#### Stage C and M2, measurement only, no code

- `#35` M2a: the `serve` rate measured against `decode run` at identical ids; the gap is the missing hot-set tick, filed as `#37`, and the run-position drift as `#38`.
- `#40` C1: sampling series on seeds 3 and 4, ten tasks, one reader.
- `#44` C2: sampling series on seeds 5 and 6, then one reader over all six series.

#### Stage E, the release surface

- `#41` E1: measured that a 125.28 MB probe blob makes GitHub refuse the first push (GH001, limit 100.00 MB). (measured 2026-09-10)
- `#46` E5: `docs/env.md` with a row per `CROW_*` name, guarded by `tools/check_env_docs.py`.
- `#50` E7, 2026-09-11: `.github/workflows/ci.yml`, four jobs on windows-latest (build, test, clippy non-blocking at 1314 warnings, guards), engine tests 72 of 80 lib and 55 of 57 serve on the runner because 10 tokenizer tests need `../models/`, first run pending the first push.
- `#47` E6: `README.md`, `engine/README.md`, `converter/README.md`, `LICENSE`, this file, and `tools/check_readme_dates.py`.

#### Licensing

- `LICENSE`: Apache License 2.0 for the code in this repository, decision E-a of the release plan.
- The model weights and every container derived from them keep the Qwen Community License 1.0; they are not in this repository.

### Changed

- `#51` E8, 2026-09-11: `decode` and `parity` default to `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` and `decode_out/hotsets-M-longctx2100-n160.json`, the container and sidecar of `serve.rs:446-447`; five lines, `CROW_CNQ` and `CROW_HOTSETS` still override; closes `#48`.
- `#42` E2, 2026-09-10: the history was rewritten with `git filter-repo` to drop the probe debug blobs. Every commit sha changed: `release-v0.1` head `4519f6f` became `aa6dd04`, `main` head `d36353a` became `e9e53cc`. Tracked bytes went from 253,232,175 to 38,397,227, blobs over 50 MB 0 after (before: one, the 125.28 MB blob of `#41`), `Co-Authored-By` trailers from 19 to 0. A pre-rewrite mirror was kept outside the repository.
- `#43` E3, 2026-09-10: `converter/target` (103 files) and 216 of 235 `decode_out` records untracked; 19 gate inputs kept; tracked bytes 38,397,227 to 5,208,050.
- `#43` E3, 2026-09-11: `decode_out/README.md` added; tracked `decode_out` files 19 to 20. The ten `final4-*-run0-crow.json` records stay because they are the greedy reference answers the identity gate (B4/C1/E3/E8) compares against, not raw run output.
- `#45` E4, 2026-09-10: `.gitignore` ignores every build tree with `**/target*/` instead of listing them.
- `#36` M2b, 2026-09-10: one snapshot slot per process; the after-answer snapshot was dropped because its rows are decode rows, not prefill rows.
- `#10`, 2026-09-09: `CROW_PF_ASYNC=2` and `CROW_PF_TG=64` became engine defaults; `CROW_PF_ASYNC=0 CROW_PF_TG=32` restores the previous behaviour.
- `#10`, 2026-09-06: `attn_sel_s8l` became the default attention kernel; `CROW_ATTN_R=0` restores the previous one. A latent `red[0]` barrier race was fixed in `attn_sel`, `attn_sel_r` and `attn_sel_split`.
- `#22`, 2026-09-05: the 1024-row parity became the third parity gate after PLE row-cache collisions within a chunk were fixed.

### Measured

| what | value | date | issue |
|---|---|---|---|
| ten-task quality, greedy | crow-nest 2 Pass / 5 Partial / 3 Fail, llama.cpp 2 / 6 / 2, of 10 each | 2026-09-10 | #11 |
| decode, engine arm | crow-nest 35.5 to 46.8 tok/s, llama.cpp 44.4 to 48.2 tok/s | 2026-09-10 | #11 |
| prefill, engine arm | crow-nest 110 to 706 tok/s, llama.cpp 265 to 846 tok/s | 2026-09-10 | #11 |
| decode through `serve` | 22.42 tok/s against 38.31 tok/s for `decode run` on the same ids | 2026-09-10 | #37 |
| run-to-run drift of a `serve` rate | 26 %, run 1 30.45 against run 6 22.42 tok/s | 2026-09-10 | #38 |
| cold prefill of 16,064 ids on `serve` | 21.6 to 22.0 s | 2026-09-10 | #31 |
| warm turn through the prefix cache | 404 ms for 95 of 16,159 ids, 99.41 % reused | 2026-09-10 | #31 |
| slot save and restore at 16k | `n_saved` = `n_restored` = 16064, file 353 MB | 2026-09-10 | #32 |
| ten-task quality, sampled | gate met in 1 of 6 seeds, 4 Pass / 37 Partial / 19 Fail of 60 | 2026-09-10 | #44 |
| parity, 8 and 512 rows | byte-identical against the installed build `d211ab52ad2b` | 2026-09-10 | #43 |
| start commands of `README.md` | 2 of 2 shells, `/health` ok and one chat answer with HTTP 200 | 2026-09-11 | #47 |

- Targets stay targets: decode at least 42 tok/s, prefill at least 972 tok/s, context floor 200,000 tokens. None of them is a pass or fail gate (`docs/architecture.md:18-19`).

### Known limitations

- Windows only; the Linux environment is unverified and stays open as `#15`.
- Prefill is below its target; the remaining gap is exposed cold-tier copy plus dense GEMM, open as `#10`.
- A `serve` rate is below the `decode run` rate on the same ids because `serve` never ticks the hot set, open as `#37`.
- A `serve` tok/s number drifts with run position on one machine, open as `#38`; no `serve` number enters the spec before it is re-measured.
- The ten-task quality gate is not met on greedy, and sampling meets it on 1 of 6 seeds, open as `#11` and `#44`.
- `min_p` is parsed and ignored: the device sampler implements top_k, top_p and presence only; decision M2 in `#28` (closed 2026-09-09), no tracking issue.
- Closed by `#51` on 2026-09-11: `decode.rs:40`, `decode.rs:45`, `parity.rs:151`, `parity.rs:156` and `parity.rs:341` now name the same container and sidecar as `serve.rs:446-447` (found 2026-09-10 in stage B, `#48`).
- A ragged hot-set sidecar makes `residency.rs:413` assert, so the container sidecars cannot be used unchanged (found 2026-09-10 in stage B, `#49`).
- `parity.rs` pipes UTF-8 into the Python oracle without `PYTHONIOENCODING`, open as `#34`; the chains export it and are unaffected.
- Engine logging is not started, open as `#13`; the architecture diagrams are stale, open as `#14`.
- Ampere and Ada are not planned: the fallback stage was closed for want of a card, `#12`.
- The clippy job renders red while non-blocking: 1314 warnings, 1085 `unnecessary_cast`, measured 2026-09-11 (`#50`).
