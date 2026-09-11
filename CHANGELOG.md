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

- `#37`, 2026-09-11: `serve` ticks the stream trickle once per `decode_step`, the mirror of `bin/decode.rs:224-231`; `adapt_tick` stays uncalled, the trickle is drained after the last step, the request-local swap count goes to the `[chat]` stderr line as `crow_trickle_swaps`, and the wire `timings` block is unchanged. Measured in two six-run chains on the same day, serve and `decode run` D1 alternating: 23.13 to 25.57 tok/s before against 26.43 to 26.77 after, `decode run` 32.82 to 34.50 in both; the 255 generated ids are identical in all 12 runs.
- `#37` fix round 1, 2026-09-11: `serve` sets `CROW_ADAPT_WINDOW=1` when it is unset, in the same loop that already sets `CROW_GRAPH` and `CROW_MMA` (`bin/serve.rs:2313`), so the stream trickle ranks its swaps by the decayed selections since the last tick instead of the prefill-dominated cumulative count; `CROW_ADAPT_WINDOW=0` still restores the old ranking. Measured in a third six-run chain of the same form: `serve` 31.97 to 32.32 tok/s against `decode run` 32.71 to 33.09, three adjacent pairs at -3.39 %, -1.87 % and -2.03 %, all within 5 %; 4,595 trickle swaps in both arms; the 255 generated ids are identical in all 18 runs of the three chains.
- `#55` C2, 2026-09-11: the `serve` sampler default is confirmed as built (`#28`): a request without `temperature` is greedy, `temperature > 0` samples with the data-sheet defaults; the ten-task gate under sampling is met in 1 of 6 seeds (`#44`), under greedy 0 of 1 (`#11`); no code change.
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

- The two arms run different weights: crow-nest CNQ4.5-M (NVFP4, 4.5 bpw); llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL (GGUF, 2.4 bpw). Every comparison names both.

| what | crow-nest | llama.cpp | delta | date | issue |
|---|---|---|---|---|---|
| ten-task quality, greedy | 2 Pass / 5 Partial / 3 Fail of 10 | 2 Pass / 6 Partial / 2 Fail of 10 | n/a | 2026-09-10 | #11 |
| decode, engine arm | 35.5 to 46.8 tok/s | 44.4 to 48.2 tok/s | n/a | 2026-09-10 | #11 |
| prefill, engine arm | 110 to 706 tok/s | 265 to 846 tok/s | n/a | 2026-09-10 | #11 |
| decode through `serve`, before the hot-set tick | 23.13 to 25.57 tok/s against 33.06 to 34.50 for `decode run`, three adjacent pairs | n/a | n/a | 2026-09-11 | #37 |
| decode through `serve`, with the hot-set tick | 26.43 to 26.77 tok/s against 32.82 to 32.98 for `decode run`, three adjacent pairs | n/a | n/a | 2026-09-11 | #37 |
| cold experts per timed token on `serve` | 292.1 before the tick, 212.9 with it, 140.0 with it plus `CROW_ADAPT_WINDOW=1`, against 137.3 for `decode run` D1 | n/a | n/a | 2026-09-11 | #37 |
| decode through `serve` with `CROW_ADAPT_WINDOW=1` | 32.20 to 32.33 tok/s against 32.78 to 32.88 for `decode run`, two adjacent pairs | n/a | n/a | 2026-09-11 | #37 |
| decode through `serve`, tick plus the `CROW_ADAPT_WINDOW` default | 31.97 to 32.32 tok/s against 32.71 to 33.09 for `decode run`, three adjacent pairs at -3.39, -1.87 and -2.03 % | n/a | n/a | 2026-09-11 | #37 |
| run-to-run drift of a `serve` rate | 26 %, run 1 30.45 against run 6 22.42 tok/s | n/a | n/a | 2026-09-10 | #38 |
| cold prefill of 16,064 ids on `serve` | 21.6 to 22.0 s | n/a | n/a | 2026-09-10 | #31 |
| warm turn through the prefix cache | 404 ms for 95 of 16,159 ids, 99.41 % reused | n/a | n/a | 2026-09-10 | #31 |
| slot save and restore at 16k | `n_saved` = `n_restored` = 16064, file 353 MB | n/a | n/a | 2026-09-10 | #32 |
| ten-task quality, sampled | gate met in 1 of 6 seeds, 4 Pass / 37 Partial / 19 Fail of 60 | n/a | n/a | 2026-09-10 | #44 |
| parity, 8 and 512 rows | byte-identical against the installed build `d211ab52ad2b` | n/a | n/a | 2026-09-10 | #43 |
| start commands of `README.md` | 2 of 2 shells, `/health` ok and one chat answer with HTTP 200 | n/a | n/a | 2026-09-11 | #47 |

- Delta is n/a on every row: each value above is a range or a Pass/Partial/Fail distribution, not a single point number with a sourced ratio, so no ratio is computed here.

- Targets stay targets: decode at least 42 tok/s, prefill at least 972 tok/s, context floor 200,000 tokens. None of them is a pass or fail gate (`docs/architecture.md:18-19`).

### Known limitations

- Windows only; the Linux environment is unverified and stays open as `#15`.
- Prefill is below its target; the remaining gap is exposed cold-tier copy plus dense GEMM, open as `#10`.
- A `serve` rate is within 5 % of the `decode run` rate on the same ids since 2026-09-11 (`#37`): the mean delta to the adjacent `decode run` D1 moved from -28.55 % without the tick to -19.17 % with it and to -2.43 % with the tick plus the `CROW_ADAPT_WINDOW` default, three pairs each. Cold experts per timed token followed: 292.1, 212.9, 140.0, against 137.3 for `decode run` D1.
- A `serve` tok/s number drifts with run position on one machine, open as `#38`; no `serve` number enters the spec before it is re-measured.
- The ten-task quality gate is not met on greedy, and sampling meets it on 1 of 6 seeds, open as `#11` and `#44`; decision on the default recorded in `#55`.
- `min_p` is parsed and ignored: the device sampler implements top_k, top_p and presence only; decision M2 in `#28` (closed 2026-09-09), no tracking issue.
- Closed by `#51` on 2026-09-11: `decode.rs:40`, `decode.rs:45`, `parity.rs:151`, `parity.rs:156` and `parity.rs:341` now name the same container and sidecar as `serve.rs:446-447` (found 2026-09-10 in stage B, `#48`).
- A ragged hot-set sidecar makes `residency.rs:413` assert, so the container sidecars cannot be used unchanged (found 2026-09-10 in stage B, `#49`).
- `parity.rs` pipes UTF-8 into the Python oracle without `PYTHONIOENCODING`, open as `#34`; the chains export it and are unaffected.
- Engine logging is not started, open as `#13`; the architecture diagrams are stale, open as `#14`.
- Ampere and Ada are not planned: the fallback stage was closed for want of a card, `#12`.
- The clippy job renders red while non-blocking: 1314 warnings, 1085 `unnecessary_cast`, measured 2026-09-11 (`#50`).
