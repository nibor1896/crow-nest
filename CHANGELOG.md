# Changelog

- Format: one section per version, subsections Added, Changed, Fixed (since 2026-09-17), Measured, Known limitations.
- Every item names its issue number in `crow-nest`, or the commit it landed in when the work had no issue.
- Every number names its date; the machine is `docs/system-landscape.md` unless another one is named.

## v0.3.1 (unreleased) — the reasoning filter, and what the 170k session really was

- Branch `main`, opened 2026-09-18 on top of `b0102c0` (v0.3.0). Sixteen issues so far: `#67` (the
  reasoning filter, `667b68b`), the engine side of `#68` (the long-context measurement, `f14e557`, and robin's decision of
  the same day: the cross-turn repeat counter as observability plus the de-duplicated replay
  that answers its open question 1),
  `#49` (the ragged hot-set sidecar, `0adbe6a`), `#60` (the `parity` record header per arm, and
  the last two bins that hard-coded the pre-`#51` container, `784bd64`), `#54` (the gone-client
  probe of the `stream:false` path, `20bc121`), `#65` (the bounded retry around the harness's
  oracle python children, `6b5025e`), `#64` (F5, the quant package's own self-test and the
  model card that carries its numbers, `cea9406`), `#14` (the living diagrams brought back to
  HEAD, `5a58e0b`), `#13` (engine logging: `tracing`, rotation, the routing line and the
  operating-point report, `788fb64`) and `#38` (the run-position drift of a `serve` rate: the
  Linux chain that answers it, `tools/drift-chain.sh` and the record in
  `docs/measurement-coverage.md`, `77c4d40`, and later the same day robin's decision on the
  issue's three consequence rules with the chain rerun at the new default — no engine code
  in either), `#61` (the decode kernel
  decomposition at the Linux operating point, the `CROW_ATTN_LUT` lever it named, and the flip of
  that lever to the DEFAULT later the same day, 61g) and
  `#62` (the GDN row taken apart per kernel, 2026-09-18 — no engine code) and `#19` (the
  cold-expert staging row taken apart, and the opt-in `CROW_STAGE_PAR` lever it named, 2026-09-18)
  and `#69` (the layer-3 sub-block check that F5 found returning zeros, repaired and added to the
  package self-test, 2026-09-18) and `#10` (the router GEMM second probe, and the ten-task quality
  reference it re-based, 2026-09-18 — no engine code) and `#71` (`#62` lever 1: the z slab out of
  the grouped GDN input launch, and the opt-in `CROW_GDN_SPLIT_Z` it rides on, 2026-09-18) and
  `#72` (the vit reserve held at boot instead of planned, and the post-plan ledger, 2026-09-18).
  The machine is the second environment block of `docs/system-landscape.md` unless a row names another one.
- The crate version field stays `0.1.0`, as it has for every release: this file is the record.

### Fixed

- **The vit reserve was planned but never HELD, and the first image request of a session found
  35.7 MiB free** (`#72`, seen by robin 2026-09-18, fixed the same day). Since `8ff2055` the
  planner set 277.3 MB aside for the image path (tower scratch 228.5 + mrope span 48.8) before it
  chose N — and then nothing allocated it: the scratch was "lazy, allocated on the first image
  request". Between `[serve] listening` and robin's seventh round the free VRAM the reserve was
  supposed to name (30.52 of 31.37 GiB used, ~0.85 GiB slack) was spent by the allocations that
  happen AFTER the plan, and the tower answered a named 503:
  `[vit] scratch allocation refused after 23 buffer(s)`, `CUDA_ERROR_OUT_OF_MEMORY allocating the
  vit patch input (24.0 MiB); free VRAM 35.7 MiB`. The engine stayed up (`8ff2055`'s path worked),
  but `README.md`'s "an image request allocates no per-request VRAM at all" only holds if the
  reserve is real.

  **The reserve is now an allocation.** `Engine::load` calls `vit::arm_scratch` right after the
  tower weights and takes the two interleaved-mrope span tables at `n_ctx` in the same place —
  before the budget verify, so `free0` excludes them exactly as it excludes the dense weights, and
  `pending` carries only what is LEFT of the reserve (nothing, at the derived value). The device
  sampler's five buffers (0.27 MB, the last thing the engine took after the plan) are held beside
  them and handed to `enable_dev_sampler` on demand. `begin_vision` and `ensure_scratch` keep
  their lazy paths as the fallback; `CROW_VIT_RESERVE_MB=0` is still the way back to the
  pre-`8ff2055` behaviour and now also turns the hold off.

  **The post-plan ledger, and the one number the issue got wrong.** A new `[budget]` line lists
  every allocation that used to happen after N was chosen, with its side of the bus:
  `post-plan allocations held at boot: vit tower scratch 228.5 MB + vit mrope span 48.8 MB +
  device sampler 0.3 MB = 277.6 MB VRAM; host RAM only (never on the card): vit image cache 256.0 MB`,
  followed by `free VRAM after load 0.54 GiB >= floor 0.25 GiB` (`manager::POST_PLAN_FLOOR`, what
  the decode graph and the driver pools still have to fit in). The prefix-cache snapshots
  (3 x 124.6 MiB) were the issue's prime suspect and they are **host RAM**, not VRAM — `Vec<f32>`
  per `cache.rs`'s memory section — so they stay out of the VRAM total; subtracting them would
  have cost about 150 hot experts for nothing. Serve's boot line says so now.

  Live, at the full operating point (`n_ctx` 200,000, prefix cache on, 3 snapshots, 2026-09-18):
  `N 160 -> 155` and `148 of 155` as before the change (the reserve moved from `pending` to
  resident, so the plan is the same), free VRAM after load 551 MiB, three text requests then one
  image request then a four-image request, all 200, no `[vit] scratch allocated` line at any of
  them (the tower armed from the held scratch) and free VRAM 544.1 MiB at both image requests with
  `engine live allocs` unchanged at 2243 — an image request allocates no VRAM at all. Tests
  206 -> 212 (six pure planning-arithmetic tests), clippy 1421.

- **`decode layercheck3` returned an ALL-ZERO `o_proj` and it read as a measurement** (`#69`,
  found by F5 on 2026-09-18, fixed the same day). On the `-M` container the layer-3
  full-attention sub-block check printed `max_abs` 5.2512 and `mean_abs` 0.5188 — exactly
  `max|golden|` and `mean|golden|` of `oracle/golden/layer3-attn-output.f32` — at `rel_L2`
  1.0000, `corr` 0.00000 and NaN 0. Those are the numbers a zero output produces, and nothing
  about the engine was measured by them, which is why F5 shipped the package self-test with one
  check instead of two.

  **The root cause is one missing launch, and it was introduced by a default, not by a kernel.**
  `Engine::run_attn_subblock` uploads the golden `mixed` row set from the host and calls the
  production `attn_prompt` — the same kernels, buffers and splits a prefill chunk runs layer 3
  with. What it skips by entering there is `hc_run`, and since `#19g` (2026-09-13, `CROW_QFUSE`
  default on) `hc_run`'s last launch is `mix_streams_q`, which writes `mixed` **and** the NVFP4
  activation cascade `xq_m`. Under `CROW_MMA=1` — the configuration of record, the one
  `tools/gate-linux.sh` and `tools/selftest.sh` set — the v projection and the QSA indexer read
  `xq_m` and nothing else; `attn_prompt` quantizes `mixed` itself only when `CROW_QFUSE=0`,
  precisely because the fused producer is the default. So `xq_m` stayed at its allocation zeros:
  measured **0 of 34,560 bytes non-zero**, with `mixed` at `max_abs` 3.80 and the BF16 `q`
  projection at 5.08 while `v` and the indexer `qk` were 0.0 — after which attention had nothing
  to weight and the gate, `xq_v` and `o_proj` were zero in turn. The same binary with `CROW_MMA`
  unset read `max_abs` 0.4475 / `corr` 0.99180 against the same golden at HEAD, which is what
  pinned the cause to the cascade and not to the QSA ring, the selection, the `#61` split or the
  kernels.

  **The fix** is that launch: `run_attn_subblock` emits `quant_x_fp4` into `xq_m` right after the
  upload — the documented bit-identical twin of the cascade `mix_streams_q` fuses (same amax →
  ue4m3 ceiling → RNE nibble → residual per 16-wide sub-block) and the launch `attn_prompt` makes
  itself when the fusion is off, so the sub-block is fed what the production path feeds it under
  either default. **Measured 2026-09-18** (RTX 5090 / Arch Linux / driver 610.57.04 / CUDA 13.3.1
  / NVRTC 13.3.33): `decode layercheck3` reads `max_abs` **0.4473**, `rel_L2` **0.1282**, `corr`
  **0.99179**, NaN 0, and the stepwise repeat (one token at a time, advancing position) reads the
  same two numbers, which is the batched == stepped pin.

  **So the package self-test ships two checks now** (`selftest/manifest.json`, `+ layer3-attn-
  input.f32` and `layer3-attn-output.f32`, 81,920 B each; `selftest/` is 825,424 B = 0.79 MiB
  against the issue's 100 MB bound, and the four sums are in `SHA256SUMS` with a dated block in
  `SHA256SUMS.log`). Layer 0 is a GDN layer, so until now no attention kernel of the 12
  full-attention layers was under the package self-test at all. The layer-3 gate is **0.625**,
  read off the measured distribution and not carried over from layer 0: `|err|` p50 0.0550, p90
  0.1391, p99 0.2355, p999 0.3133, max 0.4473 over 20,480 values, against a reference that is f32
  where the whole sub-block is NVFP4 at 4.5 bpw — the p16 chain measured this very golden at
  `max_abs` 0.582 with the q and k projections FP4 too (this container keeps them BF16). 0.625 =
  5 × the layer-0 gate sits just above that mark and carries the same headroom the layer-0 gate
  carries: 71.6 percent of the gate measured, against layer 0's 73.5. Both arms are ALL GREEN
  **PASS 2 of 2** on 2026-09-18 — from the repository with `--with-originals` (27.1 s) and from
  `/home/nibor1896/pkgtest-69`, a hard-linked package copy outside the repository with no
  `models/` (32.0 s) — at identical numbers, and the control still fires the moment a `models/`
  directory exists.

  **And the failure mode itself is now a named refusal.** `decode selftest` fails a check whose
  engine output is IDENTICALLY ZERO before it consults the gate, printing `output is identically
  zero`: zeros against a golden report the golden's own numbers back (`max_abs` = `max|golden|`,
  `rel_L2` exactly 1, `corr` exactly 0) and on a gate wide enough they would PASS. Two unit tests
  pin it and the shipped manifest's two checks with no GPU and no package, so
  `tools/gate-linux.sh` `TESTS` goes **200 → 202** (113 lib + 78 serve + 6 parity + 5 decode),
  clippy unchanged at 1421.

- **The ten-task harness lost a whole phase to one python child that died with an EMPTY stderr**
  (`#65`, three occurrences — chains 19f and 19g on 2026-09-13 and the 62b pairs preflight, fixed
  2026-09-18). `tools/tokenize_ids.py --chat` runs once per task for the crow arm's token stream;
  at task 5 of ten it exited non-zero with nothing on stderr, the fail-closed assert stopped the
  phase, and every retry on the identical binaries and the identical frozen prompt ran green. No
  wrong data was ever recorded — the cost was a full re-run, about 15 min.

  **The root cause, as far as the record can carry it: environmental, and the harness is what made
  it unpinnable.** The call site was `assert!(out.status.success(), "tokenize stderr: {}", ..)` —
  the child's stderr and nothing else. No exit status, no stdout length, no size of what had been
  sent, no name of the task. An empty stderr is not a Python error: a Python error writes its
  traceback to that pipe. An empty stderr with a non-zero status is a process that died WITHOUT
  running Python's own error path, and the name of that death is the exit code, which is the one
  thing the harness threw away. Every cause inside the harness is excluded with code evidence
  (architecture 8.9 carries the table): the prompt goes through STDIN and not through `argv`, so
  the 32,767-char `CreateProcess` cap is not reachable; `wait_with_output` drains both output pipes
  while it waits and drops the parent's stdin before it reads, so neither a full pipe nor a missing
  EOF is possible — and both of those would HANG, while all three failures were a fast non-zero
  EXIT; there is no timeout and no `kill` anywhere on the path; stderr has exactly one reader; the
  UTF-8 transport has been set on the process since `#34` and an encoding mismatch moves the ID
  COUNT, not the exit status. What is left is a child that started, exited and said nothing, which
  on Windows means `STATUS_DLL_INIT_FAILED` (`0xC0000142`), `STATUS_ACCESS_VIOLATION`,
  `STATUS_NO_MEMORY` / `STATUS_COMMITMENT_LIMIT` or an outside kill — all environmental, all of
  them named only by the exit code, and all of them likelier in the window the child was spawned
  in: between the previous task's container purge (a 104 GB whole-file standby purge on Windows)
  and the release of its ~44.6 GB pinned tier, and the next task's `Cnq::open` + `Ctx::init`.
  Starting a load in that window is what `engine/README.md`'s Windows 50.5 GiB rule is about
  (`#38`); starting an interpreter that imports `transformers` in it is the same bet. That it was
  task 5 on all three chains and never task 1 fits pressure that accumulates over a phase.
  **WHICH of those it was is not recoverable from what was recorded**, and that is now the one
  thing the guard below fixes for good.

  **The guard: a bounded retry per CALL, not per phase.** Both oracle children go through
  `oracle_child` — three attempts, 2 s and then 5 s between them (seconds, because a host resource
  that is being reclaimed needs wall-clock time; 7 s worst case against the ~15 min a lost phase
  costs). The phase retries the single task and goes on, and is never restarted. It stays
  fail-closed: after the third attempt the harness panics with the whole table of attempts and
  records nothing for that task, and each failed attempt is printed the moment it happens because
  the process may not live to write the record. A retry that succeeded lands in the run record as
  `oracle_retries` on that task's measurement row (`oracle_retries_warmup` / `oracle_retries_detok`
  in the meta block) with the task, the child, the attempt, the bytes sent, the milliseconds and
  the diagnosis — and those keys appear ONLY when something was retried, so a clean phase writes
  exactly the record it wrote before. The diagnosis leads with the exit code and names it where
  the name is the whole message (`exit_code_name`: the four NTSTATUS values above plus CPython's
  silent `120`), and an empty stderr is called empty in those words.

  **Three real defects fixed with it.** The write to the child's stdin is no longer `unwrap`ed: the
  CHILD knows why its read end went away, so the error is kept, the child is waited for anyway and
  its own stderr and exit code are reported with it — before, a child that died early was reported
  as a bare `Broken pipe` and its reason was discarded. A partial write now fails the attempt even
  behind a zero exit, a zero exit with an EMPTY stdout fails, and `tokenize` refuses an empty id
  list: a child that read a truncated stdin must not be recorded as a measurement of a prompt that
  was never sent. And `crow_complete` tokenizes BEFORE `open_model` instead of between it and
  `Engine::load`, which takes the python child out of the worst host-memory window of the process —
  numerically inert, since `tokenize` runs no kernel, reads no config, and the chunk policy still
  sees the same `ids.len()` before the load.

  **Live on Linux, 2026-09-18.** The oracle venv rebuilt at transformers 5.16.1 and checked against
  the in-engine tokenizer over the ten frozen prompts: **10 of 10 identical id streams**, with
  t4-prose at 9,398 ids — the `#34` value of record, which is what says the venv is the oracle and
  not a lookalike. Then the accident itself, with a stub interpreter that dies the #65 death
  (non-zero exit, nothing on stderr, after reading its whole stdin) on the phase's SECOND tokenize
  call: `[oracle] tokenize_ids.py --chat: attempt 1 of 3 FAILED after 4 ms — exit code 1
  (0x00000001); 0 B on stdout; stderr EMPTY …`, then `attempt 2 of 3 succeeded (1140 B on stdout)`,
  the task ran at `prompt_tokens 220` — the oracle's own count — and the phase exited 0 with the
  failed attempt in its record. The same phase run clean writes a record with no oracle keys at
  all and the same eight answer ids, `[1919, 3377, 7225, 34579, 1064, 2020, 18959, 364]`
  (`decode_out/gate65/`).

  Tests 183 -> 187, all four in `bin/parity.rs`: the pure verdict table of `child_verdict` (the #65
  shape, the Windows exit codes whose name is the whole message, the zero exit with nothing on
  stdout, a child that DID reach Python's error path, and the unix signal arm), and three against a
  real `/bin/sh` child — one that fails once with an empty stderr and is retried, one that fails
  every attempt and still fails the task, and a 60,290-byte payload that reaches the child whole
  through STDIN. No oracle venv, no llama-server and no GPU. Not covered: a child that HANGS
  instead of exiting still hangs the phase, which no occurrence has ever done.

- **A `stream:false` request whose client disconnected held the one slot for its whole
  `max_tokens` budget** (`#54`, found 2026-09-11 in the whole-branch review before v0.1.0, fixed
  2026-09-18). `CollectSink`'s three methods returned `true` unconditionally, so the document path
  could not notice a client that left. The stream path learns it from the first failed flush
  (`sse_send`) because it writes and flushes per token; the document path writes NOTHING until the
  generation is over. Measured here before the fix: a `max_tokens: 512` POST dropped one second in
  ran `generated 512 tok, decode 8097.1 ms, finish length` to the end and the next request waited
  **7.463 s**. Not exploitable in the shipped configuration (localhost, one client, Crow), and it
  was the one asymmetry between the two sinks.

  **One `poll(POLLRDHUP)` between two decode steps, and a BASELINE for what a FIN means.** The
  probe is `poll` on the request socket's descriptor with `timeout 0`, `events = POLLRDHUP`: no
  byte read, no byte written, nothing on the wire (a zero-byte write sends no segment, so it could
  not tell a closed peer from a live one anyway). What a probe cannot do on its own is separate
  the two meanings of a FIN, and that was measured over five client shapes before anything was
  written: `close()`, a `SIGKILL`ed process and a `shutdown(SHUT_WR)` client that is still waiting
  for its answer ALL give `POLLIN|POLLRDHUP` and `recv(MSG_PEEK) == 0`, with the socket in
  `CLOSE_WAIT` — only a WRITE discovers the difference, and the document path has nothing it may
  write yet. So the decision is made at the HTTP layer: `ClientProbe::new` probes once when the
  sink is built, after the body, the head, the template render and the tokenizer and before the
  prefill. An EOF that is ALREADY there is a client that finished sending and gets its document;
  an EOF that appears LATER is a client that was there when the request started and has closed the
  connection while it was our turn to speak, and the generation ends. A reset
  (`POLLERR`/`POLLHUP`/`POLLNVAL`) ends it at any time. `nginx` and `cpp-httplib`
  (llama-server's HTTP layer) both treat the half-closed client as gone; this server does not, and
  pays one baseline probe for it. Details, the measurement table and the anchors are
  architecture 7.11.18.

  **The cadence is EVERY step, and the cost is why.** One probe is 0.10 us mean and 0.14 us worst
  over 1,000,000 calls on this machine, against a 13.6 ms token: one 136,000th of the budget. A
  cadence of k would buy nothing measurable and would pay for it with k-1 further steps of a
  generation nobody is reading, so `PROBE_EVERY` is 1 and no environment variable was added
  (`docs/env.md` stays 82 = 82). Read off the server's own `timings.predicted_per_token_ms` at
  steady state, same prompt and same budget on both builds: 13.592 / 13.617 / 13.592 / 13.604 ms
  before, 13.585 / 13.589 / 13.595 ms after — below the reading's own spread. Linux only
  (`POLLRDHUP` is Linux's bit): elsewhere the probe is inert, the Windows build compiles, and the
  SSE path detects a gone client by its failed flush on every platform.

  **What it looks like live** (`tools/replay-toolcalls.py --gone-client`, new: a raw-socket
  `stream:false` POST with `max_tokens 512`, a plain `close()` after one second, then the next
  request timed from the drop): the probe fires at step 62, the generation stops ONE step later at
  63 tokens with `[chat] the client is gone at step 62: the peer closed the connection
  mid-generation (POLLRDHUP; its read side was open at the first probe)`, the summary line ends in
  `client gone`, `[serve]` names `200 OK (client gone)` and the next request is served **0.172 s**
  after the drop. The same run half-closes a second client with `shutdown(SHUT_WR)`: it logs
  `the client had already closed its write side when this request started` and gets its whole
  document, 32 of 32 tokens. The streamed answer to the same greedy prompt is byte-identical on
  both builds (355 B, `sha256 39cbf9a3fd5d2d61`), and `finish_reason` gained no new value: a
  client that is not reading does not get a new wire contract.

  Tests 179 -> 183, all four in `bin/serve.rs`: the pure decision table (`peer_from_poll` and
  `gone_reason`, with the four `revents` constants asserted against `libc` on Linux), the
  `PROBE_EVERY` cadence, the probe against a REAL loopback socket in its three shapes (live,
  `shutdown(Write)`, closed — the last two being the same wire event is what the test pins), and
  the no-socket `CollectSink` that never stops a loop, which is why nothing changed for a client
  that stays.

- **The `parity` record header could stamp a sampler profile on the llama arm's greedy answers,
  and two bins still opened the pre-`#51` container** (`#60`, both found 2026-09-11 in the review
  of `HARN1`, fixed 2026-09-18). Two harness defects, both off the numeric path by construction:
  no kernel, no id and no byte of a parity form is reachable from either.

  **The header is per ARM now.** `operating_point()` read `CROW_SAMPLE` and nothing else, while
  `llama_complete` sends `"temperature": 0.0` on every request and reads no engine variable at all.
  On this machine the two arms cannot be co-resident — the pinned cold tier and the llama-server
  model do not fit in 64 GB / 32 GB — so the ten-task gate runs them ARM-PHASED, which means one
  shell and one environment for both phases. `CROW_SAMPLE=1` set for a crow phase and left in place
  would therefore have written `sample: temp 0.7 top_p 0.8 top_k 20 presence 1.5 seed 0` into the
  record of answers drawn at temperature 0, at both sites (the `phase` meta block and the
  per-measurement row of `run`). Latent: no llama record on this branch carries it. `operating_point`
  takes the arm now and the llama arm always stamps the greedy string; the rule itself is `point_for`,
  the pure half — the arm and the sampler as arguments, no environment — so it is unit tested with
  no llama-server, no oracle venv and no GPU. Tests 177 -> 179, the first two `bin/parity.rs` has
  ever had: the arm rule against a `Sampler::new(4)` built without env, and the same rule through
  the env-reading front door with `CROW_SAMPLE=1` in the environment, which is the accident itself.

  **`plecheck` and `states` read `CROW_CNQ` now.** Both hard-coded
  `../converter/Qwen3.8-Flash-Next-CNQ4.5.cnq` — the container of record before `#51` — and read no
  variable, because `#52` (2026-09-11) named only `residency` and `sf_scan`. That file is not in this
  tree (only its per-tensor `*.cnq.sidecar.jsonl` report is), so both bins panicked in `Cnq::open`
  on this machine, and the `states` acceptance demo would otherwise have printed the expert geometry
  and the whole allocation plan of a container that is no longer of record. They take the
  `geo::DEFAULT_CNQ` + `from_engine_dir` pair the other two generator bins use, with `CROW_CNQ`
  overriding it, and `states` names the container it opened in its first line. `grep -n 'CNQ4.5'
  engine/src/bin/*.rs` now answers with `-M` in every binary, and `docs/env.md` row `CROW_CNQ` lists
  five read sites and no exception.

- **A ragged hot-set sidecar asserted instead of naming the row it could not read** (`#49`,
  found 2026-09-10 in stage B, fixed 2026-09-18). Rows of unequal length in the file
  `CROW_HOTSETS` names survived the load and died two hundred lines further down in
  `residency.rs` at `assert_eq!(s.len(), n)` with `47 != 155` — no file, no row, no length — and
  whether they died at all depended on ROW 0: the loader read the sidecar's N as `sets[0].len()`
  and adapted every row only when that one number differed from the config N. So the same ragged
  file aborted the process at one N and loaded at another.

  **Padded, not refused, and the rule is per row now.** `residency::sidecar_sets` (new, pure,
  unit tested) parses the file and applies the loader's own adapt rule to EVERY row: a short row
  is padded with the lowest unused expert ids, a long one is truncated, and each adapted row is
  named with its own length (`row 3 has 47 ids: padded with the 113 lowest unused expert ids`).
  Padding is what a short row can mean here: the planner gives every layer the same N hot slots
  whatever the file says, so a short row does not describe a smaller layer, it leaves slots the
  run has already paid for unspecified — and on the default tier residency is numerically
  invisible (7.5 condition 2, the pointer table decides only WHERE an expert is read from), so
  filling them is a placement choice and not a numeric one. Leaving them EMPTY would strand that
  VRAM and put `spare_free` out of step with the occupied slots; a row LONGER than the stride
  would hand the stream trickle a slot that holds a live expert. The padding is deterministic and
  is appended after the file's own ids, so the sidecar's frequency order stays first and one
  sidecar at one N always yields one hot set — which is what the lossy `CROW_COLD_TIER` tier
  needs, where the hot set IS numeric.

  **Refused by name, because no padding gives them a meaning**: a file that is not one JSON object
  with a `sets` array of 48 rows, an entry that is not an expert id, an id outside `0..512`, and
  an id named twice in one row (a duplicate would leave that layer's cold tier, sized
  `E - sets[l].len()`, one slot short of the ids indexed into it). The ticket's own example lands
  there: the converter writes no hot-set file at all — `converter/*.cnq.sidecar.jsonl` is the
  per-tensor quantization report, one JSON object per LINE — so `CROW_HOTSETS` pointed at it now
  answers `not one JSON object (trailing characters at line 2 column 1); a hot-set sidecar is one
  JSON object with a "sets" array of 48 rows of expert ids` instead of dying in an `unwrap`. The
  ragged file of the finding was the engine's OWN pre-`#52` warm-up output at
  `<container>.hotsets.json` (gitignored, and absent on this machine since `#52` moved that path
  to `CROW_HOTSETS_OUT`).

  Tests 174 -> 177, all three in `residency.rs`: the ragged file with row 0 exactly at N (the case
  the old loader could not see), the single-N file that keeps its one adapted line, and the
  refusals including the converter's JSONL. Parity is untouched: the id-sorted
  `decode_out/hotsets-M-longctx2100-n160.json` is 48 rows of 160 and takes the same path it did
  before.

- **`serve` streamed the model's `</think>` as content, and the client re-sent it every turn**
  (`#67`, opened 2026-09-17 against `487128d`, fixed 2026-09-18). `serve` renders the chat
  template with `enable_thinking false` and had no reasoning parser at all: whatever the model
  emitted left as `delta.content`, llama.cpp's chat parser strips exactly this. Crow stores the
  turn verbatim and re-sends the whole history every turn (`crow_core.py:3754-3761`,
  `:3783-3785`), so one stray tag was re-fed on every later turn and the model imitated it —
  seen live from about 118k of 200k context on, right after a large code paste, and on **67 of
  273** assistant turns of the `#68` session.

  **What the template does with a stored `</think>` — measured, and the issue's assumption is
  REFUTED.** `#67` expected the HF history rule `content.split('</think>')[-1]`, under which a
  stored turn that ENDS with the tag would render as an EMPTY assistant turn and the session
  would lose its own answers silently. This model's template has no such rule: prior reasoning
  comes from the separate `reasoning_content` field, and `content` is rendered **verbatim**
  (`|trim` only) inside the template's own `<think>\n\n</think>\n\n` header. So no text is ever
  lost — instead the assistant turn the model reads back carries **two** closing tags, and that
  nested shape is what it imitates. Pinned by
  `a_history_whose_last_assistant_turn_ends_with_the_tag_keeps_its_text`, which renders the
  history through the real template and asserts both halves.

  **End 1, the generation path** (`bin/serve.rs` `ThinkFilter`, `send_emits`, `chunk_reasoning`):
  a leading `<think>...</think>` block leaves as `delta.reasoning_content` and every bare
  `</think>` is dropped; neither tag can leave as `content` on any path, the malformed-tool-call
  path included. The tag is ordinary text, not an added token, so it arrives across several
  deltas: a tail that is a prefix of a candidate tag is held back until the next piece resolves
  it, and a prefix that never completes leaves as content at the flush, so no byte of an answer
  is lost. It sits between the tool-call parser and the sink and touches `Emit::Content` only,
  so the `arguments` contract of 7.11.14 stays byte-identical. `stream:false` carries the same
  text as `message.reasoning_content`.

  **End 2, the history** (`normalize_messages`, `strip_stored_think`): a stored assistant
  `content` that begins with a `<think>...</think>` block, or ends with `</think>`, is stripped
  before the render, so a client that already stored one cannot poison its next turns; a
  `</think>` in the middle, a `<think>` that never closes, any non-assistant role and
  `reasoning_content` itself are left byte-identical. Every strip logs one `[chat] normalised`
  line.

  **OFF the numeric path, by construction**: the filter reads the decoded text and writes to the
  sink, nothing it does reaches `decode_step`, the sampler or the id vector. No env switch — the
  project keeps flags for perf levers and this is a correctness fix, so `docs/env.md` stays at 82
  rows (`check_env_docs` exit 0, 82 = 82).

### Added

- **`CROW_GDN_SPLIT_Z` — the z slab leaves the grouped GDN input launch and runs beside it**
  (`#71`, `#62` lever 1, 2026-09-18, DEFAULT OFF, opt-in). `CROW_GDN_SPLIT_Z=1` keeps the grouping
  for qkv + b + a — `gemv_fp4_mma_g32[323x1]` over 10240 + 48 + 48 = 10336 rows, the kernel's
  fourth slab slot given ZERO rows through the device scalar `p.zero`, so no warp ever maps into
  it and its pointers are never read — and launches z's 6144 rows through the
  `gemv_fp4_mma_d32[192x1]` the per-slab fallback already uses, verbatim, beside it. The GDN layer
  goes from 7 launches per decode token to 8. **No new kernel and no kernel source change**:
  `KERNEL_SRC` still defines 117 `__global__`s and the host still resolves 111. One `[gdn]` boot
  line per process names the launch shape in effect.

  **Bit-identical by construction**, by exactly `#62b`'s argument: every row keeps its k split
  (`bpb = ceil(bpr/ks_n)`, `ks_n = 4` in both launches), its `bpr` loop bounds, its mma order, its
  two residual levels, its fixed ascending smem slice reduce and its own per-slab `gs` at the
  store; 10240 and 10240 + 48 are both multiples of 16, so the warp-granular group selection never
  spans two groups. Only WHICH launch carries the z rows moves. Proven with the flag ON against
  the three Linux values of record (8 rows `bceba6ff7724…` at 11,919,360 B, 512 rows
  `838723470927…` and the P8 teacher-forced `3bb3e69edf90…`, the form that matters here because 8
  and 512 are prefill-only while this is a DECODE-path change) plus the sparse regime: `decode run`
  on t1-read carries the ids sha256 `56305eee11d6` of record in **12 of 12 runs** across both arms.

  **Measured** (RTX 5090 / Arch Linux, 2026-09-18, t1-read 16,064 ids, 256 tokens, 255 timed steps,
  one fresh process per run, W + 3 adjacent pairs, `CROW_ATTN_LUT` and `CROW_STAGE_PAR` unset in
  both arms, `decode_out/71/`): B mean **25.1812 ms per decode token = 39.71 tok/s** against N
  **24.9942 = 40.01 tok/s**, **-0.1869 ms = -0.74 percent**, 3 of 3 pairs favour N (-0.1236 /
  -0.2074 / -0.2298) at 1.9 B spread windows (B window 0.0969 ms, N 0.0419), W 25.1207. Under
  `CROW_KPROF=1 CROW_PROFILE=1 CROW_GRAPH=0` the two launches read 18.2 + 14.4 us per call against
  the grouped 32.7 — floor-removed 22.6 us per GDN layer against 27.7, **-5.1 us per layer =
  -0.184 ms per decode token**, which is the pairs' number taken a different way (1.5 percent
  apart) and 86 percent of the -0.21 ms `docs/architecture.md` 4.7 predicted from `#62`'s separate
  slab readings; the missing 0.8 us is the b and a rows now riding inside the qkv launch. Default
  NOT flipped; the record is `docs/architecture.md` 4.7, `docs/env.md` row (88 -> 89).

- **`CROW_STAGE_PAR` — the cold-expert staging copy runs beside the shared expert instead of in
  front of it** (`#19`, 2026-09-18, DEFAULT OFF, opt-in). `CROW_STAGE_PAR=1` issues `stage_cold_ca`
  on a side stream and joins the compute stream again immediately before the routed gate|up GEMV,
  the first reader of the combo pointers the staging kernel rewrites: `cuEventRecord` on the compute
  stream, `cuStreamWaitEvent` on the side stream, the launch, and the pair back. Inside the decode
  graph the fork and the join are capture nodes, so the captured graph gains **one parallel branch
  per MoE layer** and no host work; outside it the same two events order the same two streams. The
  window it fills is the shared-expert chain of the same layer — `sh_gate_up_q`, the `sgv` `gemv_b`
  and the fused down `gemv_fp4_mma_dg`, about 22 us per layer = **1.05 ms per decode token** — which
  reads no staged byte. The fork event carries both hazards at once, because it depends on
  everything already on the compute stream: `router_top10` of this layer and the routed GEMVs of the
  previous one.

  **Bit-identical by construction**: the staging kernel is a pure copy into its own slots, the
  shared expert reads and writes disjoint buffers, and no floating-point operation changes kernel,
  operand or order — so the lever owes the parity gate and not the ten-task quality gate. Proven
  with the flag ON against the three Linux values of record (8 rows `bceba6ff7724…` at 11,919,360 B,
  512 rows `838723470927…`, and the P8 teacher-forced `3bb3e69edf90…`, the form that puts the DECODE
  path under the contract) plus the 32 generated ids of `decode run`.

  **Measured** (RTX 5090 / Arch Linux, 2026-09-18, t1-read 16,064 ids, 255 timed steps, one fresh
  process per run, W + 3 adjacent pairs, `decode_out/19/`): B mean **25.2962 ms per decode token =
  39.53 tok/s** against N **24.4182 = 40.95 tok/s**, **-0.8779 ms = -3.47 percent**, 3 of 3 pairs
  favour N at 10.0 B spread windows (B window 0.0878 ms), ids sha256 `56305eee11d6` in 7 of 7 runs.
  Both arms report the same 212.6 cold experts per timed decode token and the same 3,964 trickle
  swaps: the flag moves the schedule and nothing else, and -0.8779 ms is **84 percent** of the
  1.05 ms window. Under `CROW_KPROF` the row does not move (244.7 us per call against 242.7) — it
  must not, because the profiler syncs around every launch, which is the serialization this flag
  removes; that arm is the control proving no kernel got faster. No kernel source changed, so
  `KERNEL_SRC` is unchanged. Default NOT flipped; the record is `docs/architecture.md` 4.8.1.

- **`CROW_ATTN_LUT` — the decode attention kernel reads its e4m3 KV bytes out of a table**
  (`#61`, 2026-09-18, opt-in when it landed, THE DEFAULT since 61g the same day).
  `CROW_ATTN_LUT=1` launches `attn_sel_split_l` instead
  of `attn_sel_split`: the same kernel, one template on `LUT`, the KV byte taken through
  `kv_ld<LUT>` and, at `LUT = 1`, out of a shared 256-entry table filled once per block with
  `dec_e4m3(b)` for every byte. It is the change `attn_sel_s8l` has carried against `attn_sel_s8`
  since `#10`, on the one decode kernel that never got it — and the decomposition below is what
  named it: `dec_e4m3` is two `ldexpf`, a float divide and three branches **per KV byte**, and
  `attn_sel_split` walks 512 of them per selected token.

  **Bit-identical by construction**: the table holds the same float for the same byte, so the fma
  chains, the `e` order, the shuffle tree, the `expf`, the IEEE divide and the `j` order are those
  of `attn_sel_split`. Proven, not only argued — parity with the flag ON reproduces all three Linux
  values of record (8 rows `bceba6ff7724…` at 11,919,360 B, 512 rows `838723470927…`, and the P8
  teacher-forced `3bb3e69edf90…`, which is the form that puts the DECODE path under the contract:
  504 of the 512 rows go through `decode_step`), and the four LUT-on `decode run 256` runs on
  t1-read carry the ids sha256 `56305eee11d6` of record, which is the SPARSE regime the parity
  forms never reach (2,050 of 16,320 tokens selected).

  **Measured** (RTX 5090 / Arch Linux, 2026-09-18, t1-read 16,064 ids, 255 timed steps, one fresh
  process per run, W + 3 adjacent pairs, `decode_out/61/`): B mean **25.2638 ms per decode token =
  39.58 tok/s** against N **23.6772 = 42.23 tok/s**, **-1.5866 ms = -6.28 percent**, 3 of 3 pairs
  favour N at 15.0 B spread windows (B window 0.1055 ms), ids identical in 7 of 7 runs. Under
  `CROW_KPROF=1` the row itself reads 206.4 us per attention layer against 64.9 — 2.477 to 0.779 ms
  per decode token, **-3.18 x** — and the `[profile]` attention bucket 5.07 to 3.35 ms per step.

  `KERNEL_SRC` CHANGED in this commit: `attn_sel_split` became `attn_sel_split_body<LUT>` plus two
  `extern "C"` wrappers, so the module now defines 117 `__global__`s and the host resolves 111
  (was 116 / 110). The OFF path is unchanged where it counts — the same binary with the flag unset
  reproduces 206.4 us per call and the ids of record, and `tools/gate-linux.sh` is ALL GREEN with
  the flag OFF. The default was NOT flipped in this commit: robin decides — and robin decided the
  same day, so this lever is the DEFAULT since 61g (see **Changed** below). One `[attn]` boot line
  per process names the kernel it runs. `docs/architecture.md` 4.6.1, `docs/env.md` row (86 -> 87).

- **Engine logging: `tracing` as the single facade, a rotating gzipping file, one routing line per
  request and the operating point as one JSON line** (`#13`, 2026-09-18). Before this commit every
  line the engine said was an `eprintln!` — 153 sites in `engine/src`, always on, never levelled,
  never in a file, synchronous on the calling thread — and `[chat] ids [...]` printed the full id
  list of every answer, so a redirected `serve` stderr grew without bound. All seven requirements
  of the ticket are met; `docs/architecture.md` section 9 is the record.

  **The rule that shaped it: the stderr lines are an interface.** `tools/replay-toolcalls.py`
  matches `[chat] prompt ` and `[chat] the client is gone at step`, `docs/long-context-goalmode.md`
  counts `[chat] normalised` and `[chat] ids` lines, and every chain log in this repository is read
  by someone who knows those prefixes. So the conversion is mechanical and text-preserving: each
  site is now a `tracing` event with a per-component target (23 of them, from the bracket prefix
  the message already carried) and the SAME format string byte for byte — verified site by site
  against `5a58e0b`, 152 of the 153 matched in source order, levels 119 `info` / 18 `warn` /
  14 `error` / 1 `debug`; the one left alone is a test helper inside `#[cfg(test)]`, where no
  subscriber is installed and an event would swallow a failing test's diagnosis — and the mirror is
  formatted **message only** — no timestamp, no level, no target. Measured: a `decode run 32` at
  the default level writes the stderr of `5a58e0b` with exactly two lines added (the `[log]` line
  that names the file, and the boot report), and nothing else but the run-to-run values. The file
  is the machine form, `<ISO-8601 UTC>  <LEVEL> <target>: <message>`.

  **Never in the hot path, and the pair that proves it.** Both sinks sit behind
  `tracing_appender::non_blocking` — one worker thread each, `lossy(true)`, 131,072 lines buffered
  — so a call site never waits on a write and never waits on a full queue. Rotation, gzip and prune
  run on that worker. Five `decode run 64` runs in the gate's own environment, `decode`'s own
  `mean` over the 63 timed steps: **25.08 / 24.99 ms at `info`** (the new per-token event filtered
  out), **24.98 / 24.98 ms at `info,decode=trace`** (64 lines emitted), **25.00 ms on a build where
  that same line is a raw synchronous `eprintln!`** — all five inside a 0.10 ms band, which is the
  spread of the two readings of the SAME configuration, and 0.4 % of a token against the
  13.6–22 ms/token budget. TRACE measured 0.055 ms FASTER than INFO. The per-call truth, where it
  is measurable (100,000 calls, three runs, no GPU): a synchronous `writeln!` to a file
  **333–354 ns**, an enabled `tracing` event through the non-blocking rotating file
  **347–373 ns**, a **disabled** event — what `gen::decode_step` pays on every operator run —
  **0.6 ns**, one 40-millionth of a 24 ms token. The 64 generated ids are bit-identical in all five
  runs and their first 32 are the 32 ids of record.

  **Rotation.** `log::RotatingFile`, about 200 lines in this tree instead of a crate, because the
  maintained ones each pay more than they give: `rolling-file` and `tracing-rolling-file` roll by
  size but never compress and never prune by count, `file-rotate` does all three but brings
  `chrono` for its timestamp suffixes — and the suffix here is nine lines of Hinnant's
  `civil_from_days`, which the UTC day boundary needs anyway. Size limit (`CROW_LOG_ROTATE_MB`,
  default 64, decimals accepted), the day boundary, retention N (`CROW_LOG_KEEP`, default 8) and
  gzip on every rotated file, through `flate2`, which costs **zero** new crates: `image`'s `png`
  feature already pulls it. Proven live at a 1,048 B limit with `CROW_LOG_KEEP=3`: six rotations,
  the newest three kept as real gzips, contiguous and in ascending time order with the live file.
  **That proof found a bug and closed it**: the first form of the archive name added the collision
  counter only when two rotations fell inside one second, and `-` (0x2D) sorts before `.` (0x2E),
  so `…-033802-1.log.gz` sorted BEFORE `…-033802.log.gz` and the prune — which sorts by name —
  deleted the second archive of a second instead of the oldest; scanning for the first FREE number
  then reused a number the prune had just released and handed the newest archive the oldest name.
  The counter is always present and zero padded now, and monotone inside a second; two unit tests
  pin it.

  **Level routing without a rebuild.** `CROW_LOG` is `RUST_LOG` syntax through
  `tracing-subscriber`'s `EnvFilter`, default `info` for operators. `info,chat=debug` brings back
  the full `[chat] ids` list (DEBUG since this commit), `info,decode=trace` turns on the per-token
  decode forensics, `info,routing=debug` adds one line per prefill chunk. A string this build
  cannot parse installs the default and says so on a WARN line that names it — an operating switch
  with a typo may never silence a running server.

  **Machine telemetry next to the human lines.** One `routing` line per request, as JSON, drained
  from the counters that already existed and had no drain: the `[48][2]` device block of expert
  selections and cold selections, `Ple::req`/`Ple::miss`, and the trickle's swap count. Every
  counter in this engine is cumulative and never reset, so `Srv` holds the block the previous
  request left and the line is the difference — the first request-local numbers this server has
  logged: `selections`, `cold`, `hit_rate`, `layers_cold`, `bytes_streamed`, `ple_rows`,
  `ple_fills`, `ple_miss_rate`, `trickle_swaps`, `counters_ms`, beside the id counts and the
  milliseconds the `timings` block carries. One requirement of the ticket's list is **reported as
  absent instead of invented**: there is no ring round-trip counter in this tree to drain.

  **The operating point as ONE line.** `Engine::boot_point` + `log::boot`, target `boot`, at INFO,
  in addition to the eight human boot lines `serve` already printed: container, hot sets and their
  source, `n_ctx`, prompt chunk, N residency and the stride, KV dtype, kernel path, the cold tier
  and its policy with `hot_per_layer` for all 48 layers, QSA ring rows, PLE cache bytes, vision and
  prefix cache. `serve` and `decode run` emit it.

  **Windows and Linux, one code path.** `log::log_dir_from` takes the OS as an argument, so the
  Windows rule (`%LOCALAPPDATA%\crow\logs`) is unit tested on Linux and the Linux rule
  (`$XDG_STATE_HOME/crow/logs`, else `~/.local/state/crow/logs`) on Windows. `flate2`'s pure-Rust
  backend needs no C toolchain on either. No Windows process has written a rotating log yet, which
  section 9.6 says out loud.

  Ten unit tests in `log.rs` (200 = 113 lib + 78 serve + 6 parity + 3 decode), four new `CROW_LOG*`
  rows in `docs/env.md` (86 = 86), 16 new crates in `engine/Cargo.lock` — eight of them from
  `tracing-appender`, whose `rolling` module is deliberately unused — and clippy **1,421**, one
  FEWER than the 1,422 of record: `redundant reference in eprintln! argument` at `gen.rs:2890` is
  gone because that line is a `tracing` event now, and nothing new warns.

- **`docs/diagrams.md` diagram 8, the verification picture** (`#14`, 2026-09-18). The gates had no
  diagram at all, while three commits of this release built around them: `tools/gate-linux.sh`
  with its nine items and the two host-side values they pin (`TESTS` 190, `CLIPPY` 1422,
  architecture 8.7), `decode selftest` behind `tools/selftest.sh` with the `test ! -d models`
  refusal and the sha256 of the shipped golden (8.10, `#64`), `oracle_child`'s three attempts with
  the exit code leading the diagnosis (8.9, `#65`), and the three replay commands that reproduce
  the live bugs of `#67`, `#54` and `#68` (`tools/replay-toolcalls.py --think` / `--gone-client`,
  `tools/replay-session.py`, `tools/longctx-gate.py`; 7.11.16 to 7.11.18). The 1024-row parity
  form is drawn outside the script, where 8.7 puts it. Mermaid only, render-checked with
  `@mermaid-js/mermaid-cli` 11.17.0 like the other seven.

- **The quant package verifies itself, with no originals anywhere near it** (`#64`, F5,
  2026-09-18). Until now the only numeric gates on the container were the layer-wise oracle,
  which needs `models/` and a torch environment, and the parity forms, which need this
  repository's reference dumps — so a downloader had no check at all beyond `sha256sum`. The
  package now ships `selftest/`: `layer0-input.f32` and `layer0-golden-output.f32` (327,680 B
  each, `[8][10240]` f32 little-endian) plus `manifest.json`, **658,998 B = 0.63 MiB** beside a
  105 GB container, against the issue's 100 MB bound. The two arrays are the p10 layer-0 golden
  the oracle produced from the UNQUANTIZED originals on 2026-09-02 (transformers 5.16.1, torch
  2.13.0+cpu, f32, 24 of 24 layer-`0` tensors suffix-matched), copied byte-identically out of the
  Windows working tree with their sha256 in `SHA256SUMS` and a dated block in `SHA256SUMS.log`:
  `oracle/golden/` is gitignored and never was in the repository, and the originals' safetensors
  do not exist on this Linux box (`models/Qwen3.8-Flash-Next-original` holds the config and the
  tokenizer, 23 MB, no weights), so the goldens could be shipped here and not re-derived here.
  That is stated rather than smoothed over.

  **New `decode selftest [<golden_dir>]`** (`bin/decode.rs`): the golden directory is an argument
  and both model paths come from `CROW_CNQ` / `CROW_HOTSETS`, so the mode runs from any working
  directory and reads no `oracle/golden/`, no `models/` and no python. The manifest names a list
  of checks with their layer, shapes and `max_abs_gate`; the mode loads the engine once, prints
  one `max_abs` / `rel_L2` / NaN line per layer and `PASS n of n checks`, and its EXIT CODE is
  the verdict — the only mode of that bin where that is true. Every manifest field is required (a
  missing `max_abs_gate` would otherwise read as a gate of 0) and a golden whose byte length does
  not match its declared shape is refused BEFORE the compare, because a truncated download read
  as a measurement reports a delta against the wrong rows. **New `tools/selftest.sh`** runs three
  items, GREEN/RED each: the `test ! -d models` control, the golden set's own sha256 (`grep
  ' selftest/' SHA256SUMS | sha256sum -c -`, no read of the container; `--full` reads it) and the
  engine. The control is a REFUSAL and not a warning — with a `models/` directory present and no
  `--with-originals` the script exits 1 and the engine is never started, because a run in a tree
  that holds the originals cannot be presented as a run without them whatever its numbers say.

  **Measured 2026-09-18**, RTX 5090 / Arch Linux / driver 610.57.04 / CUDA 13.3.1 / NVRTC
  13.3.33, inside the memory-bounded scope, one engine at a time. `decode layercheck` against
  `oracle/golden/` reads `max_abs` **9.184837e-2**, NaN 0 — the first Linux reading of the gate
  that has been hard at 0.125 since 2026-09-04. The self-test reproduces it to all six digits in
  both arms: from the repository with `--with-originals`, ALL GREEN at `max_abs` 9.184837e-2,
  `rel_L2` 1.3671e-2, NaN 0, 24.2 s; and from `/home/nibor1896/pkgtest-f5`, a hard-linked package
  copy OUTSIDE the repository with no `models/`, ALL GREEN at the same numbers, 32.0 s of which
  the container load is 23 s. The control fired on that same copy the moment an empty `models/`
  existed: RED, exit 1, engine not started. `max_abs` is 73.5 percent of the bound. Three unit
  tests pin the pure half with no GPU and no package (the manifest contract and its four named
  refusals, the inclusive bound with the NaN rule — `f32::max` drops a NaN operand, so only the
  count sees one — and the length refusal), so `tools/gate-linux.sh` `TESTS` goes **187 → 190**
  (103 lib + 78 serve + 6 parity + 3 decode), clippy unchanged at 1422.

  **Not shipped, and a finding.** `oracle/golden/` also holds a layer-3 full-attention sub-block
  golden (the p7 chain). It is deliberately not a check: on the `-M` container `decode
  layercheck3` returns an ALL-ZERO `o_proj` output — `max_abs` 5.2512 and `mean_abs` 0.5188,
  which are exactly `max|golden|` and `mean|golden|` of that file, at `rel_L2` 1.0000 and `corr`
  0.00000, NaN 0, measured 2026-09-18. The debug path is stale and would gate nothing about the
  quant, and the production attention path is under the parity contract instead, where the logits
  are byte-identical. It owes its own issue. — That issue is `#69`, opened and closed out the same
  day: the path was one missing cascade launch, and the layer-3 check ships (see the first item of
  this section).

- **The model card is tracked, and its dates are back** (`#64`, 2026-09-18). `docs/model-card.md`
  is the Hugging Face card of record and the byte source of that repository's `README.md`; it
  used to live only in `hf-package/README.md`, which `.gitignore` ignores (F3, `#58`). Its
  Self-test section was a `<!-- SELFTEST: F5 pending -->` placeholder and now carries the
  procedure, the gate, the numbers above, the control and what the check does NOT cover, plus
  three rows in the Files table and a second verify command for the golden set alone. Two
  defects found on the way. **`tools/check_model_card_dates.py` was never committed**: written in
  F4, referenced by `tools/gate-linux.sh` and named in a dozen commit messages, it existed only
  as an untracked file in the Windows tree, so every Linux gate run printed it as "not in this
  tree - skipped". It is tracked now and it checks the tracked card. **And the card on the Hub
  had every date stripped out of it.** The copy fetched from the Hub on 2026-09-18 carries 0
  dated lines against the Windows copy's 37, and the strip was mechanical rather than an edit —
  it left "Spec section 1 of the engine, approved ." and turned the upstream `lastModified`
  `2026-08-27T05:03:36Z` into `T05:03:36Z`. Under the record rule (spec section 0.5) that made 25
  lines offenders, which is precisely the failure the guard exists to catch and could not,
  because it was not in the tree. Every date is restored from the Windows copy of record or from
  this repository (the measured rows carry the date of the commit they cite, 2026-09-17), five
  lines the F4 run never saw are dated for the first time (the Vision section, added 2026-09-14),
  and the stale "until it lands" row about the image path is replaced by what landed. The guard
  now reads **83 number lines, 49 dated, 34 exempt, 0 offenders**, and its negative control exits
  1.

- `delta.reasoning_content` on the stream and `message.reasoning_content` on the non-streaming
  document (`#67`, 2026-09-18), present only when the filter stripped a block the model opened
  itself. Both were listed as "never emitted" until now; Crow reads the key
  (`crow_core.py:5045`), shows it behind `--show-reasoning`, stores it (`:3755`) and re-sends it,
  and this template renders a stored one into the assistant turn's think block — so the round
  trip is the template's own form and not a second copy of the answer.
- `tools/replay-toolcalls.py --think` (`#67`, 2026-09-18): the live shape in one command — a
  ~3 KB code paste in the first user turn, then three ordinary turns, the whole history re-sent
  every turn the way Crow does it, no `tools` at all. Every round must answer 200 with no
  `<think>` and no `</think>` anywhere in its streamed content; the reader also accumulates
  `reasoning_content` and stores it on the turn, as Crow does.
- Six tests in `bin/serve.rs` (`cargo test --release` 165 → **171**, 98 lib + 73 serve): the
  byte-identical passthrough of a stream without a tag, the stray closing tag at EVERY split
  point of the live line, the leading think block, the tool-call fragments the filter must not
  touch, what the real template does with a stored `</think>`, and the normaliser's table.
  `tools/gate-linux.sh` carries the new count with its provenance.
- `tools/replay-session.py` (`#68`, 2026-09-18): robin's stored goal-mode session replayed against
  a running `serve` at a chosen context size and with a chosen sampling row. It cuts the
  574-message conversation of `decode_out/sessions/2026-09-17-goalmode/` after a USER turn, sends
  it whole the way Crow does (`crow_core.py:3783-3785`), renders `tools` from Crow's own
  25-function `TOOLS` constant (exec-extracted out of `crow_core.py`, 14,541 B of JSON), feeds the
  engine's own `tool_calls` back with the live session's own tool results, and re-sends the goal
  nudge as the next user turn — the shape 105 of the 114 live user turns had. Per round it records
  prompt tokens (cached / prefilled), the generated text, an echo verdict (32-character shingle
  overlap with the nudge plus the live marker string), a degeneration verdict in BOTH live forms
  (one unit repeated inside an answer, and an answer of at most two tokens) and both rates out of
  `timings`. Three rows are built in: `greedy`, `card` (0.7 / 0.8 / 20 / 1.5) and `crow`
  (1.0 / 0.95 / 20 / 1.5, seed 0). Since the same day it also has `--dedup-nudges`: consecutive
  byte-identical USER turns of the stored history are collapsed (574 messages -> 475, 114 user
  turns -> 15, 105 goal nudges -> 6) while every assistant turn and tool result stays where it
  was, which is the arm that answers open question 1 (Measured, below).
- `tools/longctx-gate.py` (`#68`, 2026-09-18): the long-context quality gate the ten-task gate
  never had — one agentic session shape (a synthetic `libghost` crate read file by file through
  `read_file` tool calls, generated from the file index alone so it is byte-stable and cannot
  drift with the tree), then five probe turns on the same session, greedy, scored the way
  `docs/ten-tasks.md` scores: three constants planted at ~5 %, ~50 % and ~95 % depth (value plus
  the file that declares it), the three-value arithmetic across those depths
  (`5137 + 9281 − 4409 = 10009`), and the one function whose body contradicts its own doc comment.
  Degeneration is a Fail on its own (`docs/ten-task-expected.md` §1). Recorded expectation, so it
  can become a standing gate: **>= 4 of 5 Pass and 0 degenerate**.
- `docs/long-context-goalmode.md` (`#68`, 2026-09-18): the measurement record of the replay, the
  penalty scope, the quality gate, the cause separation and the open questions.
- **A cross-turn repeat counter on the `[chat]` and `routing` lines** (`#68`, robin's decision of
  2026-09-18 on open question 4 of `docs/long-context-goalmode.md`): pure observability, no
  sampling change and no new knob. `serve` keeps a ring of the last 8 answer HASHES per process
  (FNV-1a over the GENERATED IDS — what the model produced, before the detokenizer, the tool-call
  parser or the `#67` filter) and reports per completed request `repeat_of` (how many answers back
  the most recent identical answer is, 0 = none in the ring), `repeat_run` (identical answers in a
  row, 1 = none, NOT capped by the ring) and `single_token` (exactly one generated id and the model
  ended the answer itself — a one-id answer that hit the client's own `max_tokens` budget is
  `finish length` and does not count). All three are fields of the `routing` JSON line on every
  request; the `[chat]` summary line gains `, repeat run N` only when N > 1 and
  `, single-token answer` only when true, so a healthy session's line is byte-identical to the line
  `tools/replay-toolcalls.py`, `tools/drift-chain.sh` and `tools/gate-linux.sh` grep. At three
  identical answers in a row — or three single-token ones — ONE WARN line on target `chat` says
  `the client is looping: N identical answers in a row (#68)` and nothing else happens: no 4xx, no
  brake, no sampling change, the wire untouched. Why it is not a brake: nothing INSIDE one request
  can see the live stage-3 shape — 48 of the 293 answers of the goal-mode session were the single
  id 18 (`3`) with `finish stop` and every one of those requests was correct on its own — but the
  PROCESS can, because `serve` is stateful (one client, one held conversation). SCOPE is per
  process on purpose: there is no session id on the wire and an identical re-send is a COLD prefill
  by construction, which is exactly the case the counter exists to see. Documented in
  `docs/architecture.md` 7.11.19 and 9.3, `engine/README.md`.
- Four tests (`cargo test --release` 202 → **206**, 113 lib + 82 serve + 6 parity + 5 decode), all
  pure host logic in `bin/serve.rs` (`#68`, 2026-09-18): the ring's run and its distance back (a
  run is CONSECUTIVE, and it is not capped by the ring — 48 identical answers report 48), the hash
  over the generated ids (order and length matter, an answer older than the eight-deep ring is out
  of it), the single-token rule (one id AND `finish stop`), and the WARN threshold plus the
  `[chat]` suffix (a healthy answer adds nothing to that line).
  `tools/gate-linux.sh` carries the new count with its provenance.
- Three tests (`cargo test --release` 171 → **174**, 100 lib + 74 serve): `sample.rs`
  `the_presence_penalty_is_applied_once_per_distinct_token` (presence and frequency pick DIFFERENT
  tokens on the test's logits, so it cannot pass under a count-scaled penalty) and
  `the_penalty_set_is_this_answers_tokens_only` (a fresh `Sampler` penalizes nothing, and
  `Sampler::new` is what the server builds per request), plus `bin/serve.rs`
  `the_sampling_line_says_which_values_the_request_carried` (`#68`, 2026-09-18).
  `tools/gate-linux.sh` carries the new count with its provenance.

### Changed

- **The three consequence rules of `#38` are decided, and the serve rate of record enters the
  spec** (`#38`, robin 2026-09-18, after the Linux chains of that morning; docs only, no engine
  code, so no gate run is owed). The issue had carried three rules since 2026-09-10, all of them
  written when a `serve` rate on the Windows box drifted 26.4 % with run position.

  **1. The host-RAM gate is per OS.** *Windows*: the 50.5 GiB gate stands (wait for more than
  50.5 GiB free host RAM before a load, rule since 2026-09-10). *Linux*: it does not apply and
  stays REPLACED by the pinned budget the engine derives at boot and prints on its `[budget]`
  line (`#15`, v0.3.0, `docs/architecture.md` 8.8 point 2). Nothing in the code moves; what
  changed is that no document now implies the 50.5 GiB figure gates a Linux run. The twenty
  engine starts of this day's three chains all passed on the derived budget at 46.00 GiB, and a
  `MemAvailable` reading of the Windows gate would have refused every one of them (9.4 to
  11.6 GiB available against 60.2 GiB free for pinning).

  **2. "A `serve` tok/s is quoted only next to an adjacent `decode run` measured in the same
  chain" is KEPT, on Linux too, with a new reason.** Not the drift — that is absent on this box —
  but the operating point: `serve` pins prompt chunk 2048, gets N = 149 hot experts per layer and
  ticks the stream trickle every decode step, `decode run` lets the policy pick 4096, gets N = 142
  and does not tick, so a lone serve rate invites a comparison the configuration does not support.
  The reason is written where the rule is stated: `docs/architecture.md` 0.5 and 3.4,
  `engine/README.md` "Machine rules", `README.md`.

  **3. "No serve decode number enters `docs/architecture.md`" is RELAXED on Linux to the
  drift-chain form.** A serve rate may enter when it carries all of: at least 3 counted serve runs
  in one chain, one fresh process per run, the generated-ids sha256 identical across them, an
  adjacent `decode run` arm in the SAME chain, and the figure quoted as its arm mean with its
  max-over-min spread beside that decode arm's mean and spread, naming the chain log. A number
  without that form is refused, which is why the rule text in `docs/architecture.md` 0.5 names the
  form. Windows keeps the bar until the M2a form is rerun there.

  **The number that satisfies rule 3 at the CURRENT default** (the morning's 49.82 / 1.0056 was
  measured before the `#61g` `CROW_ATTN_LUT` flip, which moves both arms). `tools/drift-chain.sh
  c3-sdsd-61g SDSDSDSD` at HEAD `6c87054`, RTX 5090 / Arch Linux, 2026-09-18, one fresh process per
  run, `CROW_ATTN_LUT`, `CROW_STAGE_PAR` and `CROW_GDN_SPLIT_Z` unset,
  `decode_out/38/c3-sdsd-61g/chain.log`: **`serve` 53.32 tok/s mean over 4 counted runs (53.282 /
  53.412 / 53.209 / 53.375), within-arm spread 1.0038**, next to the adjacent **`decode run` arm's
  42.56 tok/s mean (42.583 / 42.542 / 42.581 / 42.544), spread 1.0010** — generated ids
  `e7c17e064ea2` 4 of 4 and `56305eee11d6` 4 of 4, `serve[1:] == decode[:255]` TRUE, serve counters
  4,494,905 cold of 7,833,120 selections and 261,104 PLE rows identical in all four runs. The drift
  is absent at the new default too, and tighter than in the morning chain (1.0038 against 1.0056;
  the `decode run` arm 1.0010 against 1.0012). The `decode run` arm's 23.4950 ms per token
  reproduces the `#61g` confirmation mean (23.5193) to 0.10 %, which is what puts the serve number
  on the operating point of record; against the morning chain at the pre-flip default both arms
  move by the flip, serve +7.02 % and `decode run` +6.86 %, a cross-chain reading under rule 11 of
  the 38a discipline and not the lever's own pair (−6.28 %, 61f). The record with both tables, the
  eight machine blocks and the rule texts is `docs/measurement-coverage.md`; the figure of record
  and the form that admits it are `docs/architecture.md` 4.6.1 and 0.5, and it replaces the
  single-run 53.31 reading the `#61g` commit had put there, which stays as the lever's own serve
  A/B.

- **`CROW_ATTN_LUT` is the DEFAULT — the split decode attention kernel reads its e4m3 KV bytes out
  of the shared table** (`#61`, 61g, 2026-09-18, robin's call after the 61f numbers). `attn_lut_on()`
  goes from the 61f opt-in `== Ok("1")` to the house `!= Ok("0")` pattern (`gen.rs:1707`): unset or
  any value but `0` launches `attn_sel_split_l`, and `CROW_ATTN_LUT=0` is the fallback of record that
  launches the pre-61f `attn_sel_split`. The `[attn]` boot line names the default. `gen.rs` only — no
  kernel changed, so `KERNEL_SRC` is unchanged (117 `__global__`s / 111 launched).

  **Why it needs no quality gate**: the lever is bit-identical BY CONSTRUCTION (the table holds
  `dec_e4m3(b)` for every byte, so the fma chains, the `e` order, the shuffle tree, the `expf`, the
  IEEE divide and the `j` order are those of `attn_sel_split`) and was measured so in 61f on all
  three parity forms and on the sparse `decode run` ids. This commit re-proves it AT THE NEW DEFAULT:
  `tools/gate-linux.sh decode_out/gate61g` is ALL GREEN nine of nine with NO env — parity 8
  `bceba6ff7724`, 512 `838723470927`, P8 teacher-forced `3bb3e69edf90`, the 32 `decode run` ids of
  record, `cargo test --release` 202 / 0, clippy 1421 and the three doc guards — and the three parity
  forms run once more with `CROW_ATTN_LUT=0` reproduce the same three values.

  **Measured** (RTX 5090 / Arch Linux, 2026-09-18, t1-read 16,064 ids, 256 tokens, 255 timed steps,
  one fresh process per run, `CROW_STAGE_PAR` and `CROW_GDN_SPLIT_Z` unset, `decode_out/61g/`). The
  W + 3N form a flip takes, with no adjacent B arm because the no-env arm now IS the lever: W 23.5697,
  N1 23.5571, N2 23.4800, N3 23.5207, **N mean 23.5193 ms per decode token = 42.52 tok/s** at a
  0.0771 ms spread, against the one `CROW_ATTN_LUT=0` fallback run at **25.2044 = 39.68 tok/s** — the
  25.2 to 25.3 ms arm the flip left behind, the same one #19, #61 and #71 measured on this machine
  today. The ids sha256 is `56305eee11d6` in **5 of 5** runs: the flip moved the rate and not one
  generated id. Through `serve`, which is what Crow sees (one POST `/v1/chat/completions`, the
  t1-read prompt, `max_tokens` 256, greedy, `stream` false, one fresh process per arm), the default
  reads **4,783.7 ms predicted = 53.31 tok/s** against the `=0` arm's **5,112.7 ms = 49.88 tok/s**,
  **-329.0 ms = -6.44 percent**, at identical generated ids `e7c17e064ea2` — and that `=0` arm lands
  on the #38 serve figure of record (49.82 tok/s at 5,115 ms). The lever's own adjacent-pair reading
  stays the 61f one (25.2638 -> 23.6772, -1.5866 ms = **-6.28 percent**, 3 of 3 pairs); it is not
  re-run here. `docs/architecture.md` 4.6.1 carries the flip, `docs/env.md` the row.

- **The living diagrams are current again** (`#14`, 2026-09-18): the eight commits that landed
  after the 2026-09-12 pass — v0.3.0 (`9f12429`..`487128d`) and the seven v0.3.1 commits — were
  audited against every diagram in `docs/diagrams.md`, and five were redrawn. **1, system
  overview**: the loader box says the host pinned budget is derived and the vision reserve is
  subtracted before N is chosen, the engine box says Linux since 2026-09-17 (`#15`), and the quant
  package with its travelling `selftest/` golden and the bounded launcher are new boxes (2.1, 7.13,
  8.8, 8.10). **2, decode path**: the PLE prep names the batched `cnq::Warm` row fetch instead of
  one mapping fault per row and the trickle box the 8-token re-cut that replaced `#17`'s 16 (7.14),
  and the sampler box carries the `#68` penalty scope (7.11.17); the three defaults of `#19e`,
  `#63c` and `#61a`/`#61b` are untouched. **3, VRAM pie**: the 277.3 MB vision reserve is a planned
  slice now, out of the headroom it used to live in (2.1, 7.13). **4, residency**, renamed
  "Residency and load": the derived budget with its `/dev/nvidia-uvm` fallback, the two-sided clamp
  with the reserve in `pending`, the one ascending sweep with `fadvise_consumed`, the `ple`-sparing
  exit purge, the per-row sidecar rule of `#49` and the PLE row path with `page_runs` and the
  16-thread reader pool (2.2, 7.14, 8.4, 8.8). **5, serve**, now the whole request path: the
  normaliser's two jobs, `check_messages` before the render, the `ThinkFilter` and its three states,
  the sink split with `CollectSink`'s `ClientProbe(POLLRDHUP)` and its baseline, the two
  sampling-provenance lines, and `guarded` answering a request-scoped allocation failure with a
  named 503 (7.11.13 to 7.11.18, 7.13, 8.5). Unchanged, with the reason in their status lines: **6,
  the converter** (no converter stage since 2026-09-02; the container facts that moved are
  reader-side) and **7, the module graph** (the `use crate::` edges at `cea9406` are edge for edge
  those of `487128d` — only `sample.rs` and `residency.rs` were touched in the library and
  everything else landed in a bin, which that graph does not draw). Every box cites the section it
  renders, each diagram carries the commit it renders, and all eight were rendered locally with
  `@mermaid-js/mermaid-cli` 11.17.0 before the commit.

- The `[chat] sampling on the device` line names the SOURCE of every value (`#68`, 2026-09-18):
  `temperature 1 (request) top_p 0.95 (request) top_k 20 (data sheet) presence_penalty 1.5 (data
  sheet) seed 0 (data sheet)`, and a second line says what the penalty applies over
  (`cleared for this request, generated tokens only`). Why: the live `#68` line was read as
  "sampling as sent by Crow", and Crow never sent three of those five values — its wire list is
  `SAMPLING_FIELDS = ("temperature", "top_p", "min_p", "top_k")` (`crow_core.py:716`, build of
  2026-09-16) and the string `presence_penalty` does not occur in its source at all, so 1.5 was
  `DEFAULT_PRESENCE` of `bin/serve.rs`. `ChatReq::sampling_sent` (four bools) carries it, a field
  counts as sent when the body has it as a non-null value — the same condition every reader uses
  for "absent" — and the sampler never sees the struct. The defaults themselves do not change
  (`#28` A6 decided them) and nothing on the wire moves.

### Measured

- **The router GEMM second probe: RED greedy on the degeneration clause, GREEN sampled — and the
  reference the first verdict used does not exist on this platform** (`#10` 10e, 2026-09-18, RTX 5090 / Arch Linux,
  HEAD `3feec4d`, chain `decode_out/10e/srv-10e.log`, record `docs/architecture.md` 5.4).
  `CROW_ROUTER_GEMM=1` was stood down on 2026-09-14 (10d, Windows) because the ten-task quality
  gate read 0 Pass / 7 Partial / 3 Fail against the series record of 2 / 5 / 3 — the improve-loop
  RED line is a pass count below reference minus one. robin commissioned the second probe.
  **No engine source change, and no default flipped**; the switch stays opt-in with default off,
  and the numeric contract of `docs/architecture.md` 8.7 moves whenever it is on.

  **What "a second sample" can be.** Greedy is a pure function of the logits, so a repeat greedy
  run is the same sample — and both repeats prove it: the control and the switch each reproduce
  **byte-identically, 10 of 10**, ids and text, degeneration included. The second samples that do
  exist are the Linux arm itself (this toolchain re-rolls the stream on its own, so the Linux ON
  arm is an independent realization with its own control in the same chain, which 10d never had)
  and a sampled draw at the data-sheet non-thinking profile with one seed on both arms.

  **The control is a Linux value of record, and it is not `final4`.** The no-env arm is
  byte-identical, ids and text, on all ten tasks to `decode_out/final/ten-run0-crow.json`
  (2026-09-17, `0667e0b`) across nine commits and two tokenizer paths — and it differs from the
  WINDOWS `final4` record on **nine of ten tasks with no flag set at all** (first differing index
  5 to 138; only `t6b-reason-multi` reproduces it, over 1024 ids and 2006 characters). Judged with
  the 10d rubric the Linux default reads **0 Pass / 5 Partial / 5 Fail**. A judging control holds:
  re-judging the two tasks the record scores Pass, from the tracked `final4` texts, returns Pass on
  both, so the 0 is the answers and not the judge. The 10d pass clause never discriminated the
  lever.

  **The lever's own greedy arm reads 1 Pass / 5 Partial / 4 Fail** — strictly better than its
  control on counts, with the only Pass either greedy arm produced (`t4-prose`, which names the
  constant-slot-cost assumption and quotes the text's own driver-spill warning) and three upgrades
  against one downgrade. The gate is RED on the degeneration clause alone: `t5-agent` runs its whole
  1536-token budget as a repetition loop (106 repeats, 901 backticked items) and never reaches the
  assumptions list. It is not a new failure mode — all six measured arms write the same sentence
  with the same false premise, five emit "etc." and finish, this one has no exit token.

  **And the sampled pair is GREEN**: at the data-sheet non-thinking profile with one seed on both
  arms the control reads 0 / 7 / 3 and the switch **1 / 6 / 3**, no degeneration in either, every
  clause of the line held. The same default engine reads 0 / 5 / 5 greedy and 0 / 7 / 3 sampled, so
  the operating point alone moves the counts by two steps; and 6 of 10 task verdicts move across six
  equally defensible arms of the same engine. C1 is answered in the affirmative — a second sample
  did land differently, and it landed green.

  **And the house had already measured this instrument's noise.** `#40` / `#44` (section 7.12 rows
  C1 and C2, robin-decided 2026-09-11 in `#55`) ran the ten tasks over six sampling seeds of the
  UNCHANGED default, 60 answers, one reader plus a reviewer: smpv1 1/6/3, smpv2 1/6/3, smp3 0/6/4,
  smp4 1/7/2, smp5 **0/7/3**, smp6 1/5/4 — Pass ranges 0 to 2 and Fail 2 to 4 with nothing changed
  but the seed. **10d's ON arm read 0 / 7 / 3, to the digit the line the unchanged default produced
  on seed 5**; this chain's sampled control reads the same line, its sampled ON arm reads 1 / 6 / 3
  (smpv1 and smpv2) and its greedy ON arm 1 / 5 / 4. Every judged arm of this lever, on both
  platforms, falls inside the spread a seed change alone already produces.

  **The prefill the 10a plan estimated at 0.8-1.0 s is measured at -1.750 s** (t1-read 16,064 ids,
  F49 form, W + 3 adjacent pairs: B 16.184 s = 993 tok/s against N 14.434 s = 1113 tok/s,
  **-10.81 %**, 3 of 3 pairs, within-arm spreads 0.052 / 0.041 s), and **-0.225 s = -10.88 %** on
  the 2,100-token standing prompt, 3 of 3 pairs, spreads 0.001 / 0.002 s. Ids stable within each
  arm, hot set identical in both.

  **C3 has a number.** With `CROW_DUMP_H` on the 2,100-token prompt (one chunk; layer 0, the only
  layer whose router input is still bit-identical between the arms — checked): the real masked
  max-rel is **1.094e-5** against the synthetic probe's 3.777e-3, i.e. the probe was pessimistic by
  **345x** (the same probe binary re-run here reproduces the Windows numbers to every digit, so the
  gap is the activation distribution, not the toolchain). Real top-10 boundary margins are indeed
  tight — **1171 of 2100 tokens (55.8 %) sit inside the synthetic error band** — but only **10 of
  2100 (0.48 %)** are within reach of this switch's actual error, and the top-10 sets and their
  order are **identical on 2100 of 2100**. What propagates is the continuous channel: the softmax
  weights over the unchanged top-10 still move by up to 3.755e-6.

  **The numeric contract moves, measured**: with the switch OFF all three parity forms are GREEN
  IDENTICAL at the `docs/architecture.md` 8.7 values (8 rows `bceba6ff7724`, 512 `838723470927`,
  P8 teacher-forced `3bb3e69edf90`), and with it ON all three differ. Row-wise, a 1.2e-4
  perturbation of the layer-0 router logits comes out of 48 layers as max `|d|` 27.5 on the 8-row
  form (argmax differs on 3 of 12 rows) and 8.33 on the 512-row form (**41 of 516 rows, 7.9 %**) —
  larger than the Windows-to-Linux drift, which reads 7.0 with the ids identical in all 517
  positions.

  One standing cost belongs on the record: **`MoeW::router_bf` is loaded unconditionally**
  (`gen.rs:863`), 512 x 2560 x 2 B x 48 layers = **120 MiB of VRAM resident for a switch that is off
  by default**, on top of the 240 MiB of the f32 router the default path uses.

  **Recommendation (robin decides): keep the switch opt-in, do not remove it and do not make it
  default.** What this chain establishes is that the ten-task gate, at one greedy sample per task,
  cannot decide a numeric-drift lever: it turns a continuous 1e-4-class logit perturbation into a
  coin flip on a handful of near-ties, and the Windows-to-Linux toolchain change moved nine of ten
  answers and cost the same two Passes with no code change at all.

- **The cold-expert staging row is the PCIe link, not the kernel and not the launches**
  (`#19`, 2026-09-18, RTX 5090 / Arch Linux, HEAD `cb1895a`, `decode run` on t1-read, 16,064 ids,
  greedy, 256 tokens / 255 timed steps, context 16,320; logs `decode_out/19/`, record
  `docs/architecture.md` 4.8). `stage_cold_ca` is rank 1 of the whole decode step in the `#61`
  table — 11.468 ms per decode token, 38.2 percent of what the profiler attributes. This is that row
  taken apart. The flag-free control at HEAD reads **25.32 ms per decode token = 39.5 tok/s at ids
  sha256 `56305eee11d6`**. The row needs no prefill differencing: the staging branch is taken only
  when `t * TOPK <= stage.max`, so `calls/step` reads exactly 48.0 — one per MoE layer, nothing from
  the 16,064-token prefill — and two runs of the same arm reproduce **244.7 us/call to four digits**.

  **The bytes.** One cold expert is `gate_up [1280 x 2560]` + `down [2560 x 640]` at 4.5 bpw =
  1,843,200 + 921,600 = **2,764,800 B**. At the operating point the engine's own counter reads
  **212.6 cold experts of 480 selections per timed decode token = 587.8 MB over PCIe per token**,
  4.43 per layer per launch, and the row costs **11.746 ms** (11.506 floor-removed) = **50.0 GB/s
  raw, 51.1 GB/s floor-removed**.

  **The latency term, fitted.** The stream trickle makes the cold count a known function of run
  length, so six runs of the identical prompt at `gen` 8 / 16 / 32 / 64 / 128 / 256 sweep the
  per-launch expert count with everything else fixed. `a + b x experts` per layer fits all six to
  **rms 0.100 us = 0.04 percent**: `a` = **4.74 us per launch** (the profiler's own floor is 5.0 us
  of that, so the kernel's fixed cost is 0 to 5.5 us = **0.00 to 0.26 ms per decode token**) and
  `b` = **54.07 us per cold expert = 51.1 GB/s**. A launch costs 4.7 us at 0 cold experts, 59 at 1,
  113 at 2, 221 at 4, 437 at 8. Out of sample: `CROW_CHUNK=2048` (N 149 of 156 slots, the `serve`
  figure) drops the cold count to 202.4 per token, the fit predicts **234.0 us/call** with no
  refitting and the run measures **234.0**, ids unchanged.

  **The ceiling of this box, measured the same morning** (`pcie_probe`, 1 GiB per row, 5 reps, two
  runs, every row byte-checked; the 19b/19c tables were Windows, these are the LINUX values of
  record on PCIe 5.0 x16): best DEVICE-ISSUED **51.6 GB/s** (`cp.async.cg` 4 KB tiles at 40 x 256 —
  the exact shape `stage_cold_ca` runs), every other device-issued form 50.1 to 50.9 (TMA bulk,
  `ld.global.v8.b32`, plain 16 B loads, write-combined or cacheable alike), copy engine **54.6**,
  VRAM to VRAM 746.5, host `memcpy` one thread 7.5, two and four streams no gain. **The staging
  kernel runs at 99 percent of the device-issued ceiling**; the same bytes at that ceiling would be
  11.39 ms against the 11.50 measured, so the whole distance to a perfect copy is **0.34 ms per
  decode token = 2.9 percent**.

  **Three independent ways the kernel shape does not matter**, each one `CROW_KPROF` arm at the same
  211.8 cold experts per token against the default's 244.7 us/call: `CROW_STAGE_BLOCKS` 8 / 20 / 40 /
  80 / 160 / 320 reads 258.1 / **240.7** / 244.7 / 245.0 / 245.2 / 244.9 — flat to 1.8 percent over a
  16 x block range; `CROW_STAGE_KERNEL=1`, the pre-`#19e` `stage_cold` with a completely different
  read shape, reads **244.7, the same number to four digits** (on Windows the same comparison was
  1.47 x, `#19d`) — **the `#19d`/`#19e` lever, which is why `stage_cold_ca` is the default at all,
  buys nothing on this box**; and `CROW_PINNED_WC=0` reads 244.5, so the 47.6-against-24 GB/s
  write-combined figure in `residency.rs` is a 2026-09-04 Windows measurement of the old kernel and
  does not describe this path on Linux.

  **Verdict: bandwidth bound.** Not latency bound (the issue body's "latency bound at 2 to 3 cold
  experts per layer" was a 2026-09-04 Windows reading at context 528; here the launch term is at most
  2 percent of the row), not occupancy bound, not per-expert-chain bound (linear in the expert count
  to 0.04 percent), not a property of the read instruction. What that rules out: a faster staging
  kernel (0.15 ms left in it), the copy engine (6 percent faster but `#19b` measured it losing 6.49 ms
  to the decode graph to win 0.78), and fewer launches (48 cannot become 1 — layer L+1's routing needs
  layer L's output — and they cost 0.23 ms in total). What is left is **fewer bytes**, which is `#8`'s
  territory (`CROW_CHUNK=2048` buys 5 to 7 hot slots per layer and -0.51 ms of staging, and the reason
  `decode run` sits at N 142 where `serve` sits at 149 is the chunk policy: 4,096 against 2,048, whose
  scratch plus a 4,100-row QSA ring is exactly the VRAM the loader then cannot give the hot set), and
  **overlap**, which is the `CROW_STAGE_PAR` lever above.

  **The external leg** (source read 2026-09-18): llama.cpp does not stream expert weights over PCIe
  at all — `-ot` / `--cpu-moe` / `--n-cpu-moe` set a tensor's BUFFER TYPE to the CPU backend and
  ggml's scheduler runs those matmuls on the CPU, so only activations cross the bus, and there is no
  expert cache or weight prefetch anywhere in `ggml-cuda`. The `--moe-stream` VRAM expert cache with
  a host-RAM L2 tier is a `crow` patch, not upstream, and it is the shape of crow-nest's residency
  model. ktransformers and Fiddler make the same CPU-compute trade; PowerInfer predicts the hot set
  online where crow-nest measures it offline (`#8`). Nobody streams 588 MB per token over PCIe at
  decode — and the single biggest term in that number is the weight format: llama.cpp's arm of record
  is 2.4 bpw against crow's 4.5, so the same 212.6 cold experts are 313 MB for it and 588 MB for
  crow, 6.07 ms against 11.39 at this machine's own measured rate.

- **The GDN row taken apart per kernel, and what the +1.64 ms of this issue actually is**
  (`#62`, 2026-09-18, RTX 5090 / Arch Linux, HEAD `156e2fe`, `decode run` on t1-read, 16,064 ids,
  greedy, 256 tokens / 255 timed steps, context 16,320; logs `decode_out/62/`, record
  `docs/architecture.md` 4.7). The `#62e` re-decomposition that "stays open" is this. Same method as
  `#61` above — `CROW_KPROF=1 CROW_PROFILE=1 CROW_GRAPH=0`, prefill removed by differencing `gen`
  256 against `gen` 8 over the 248 extra decode steps, a 5.0 us per-launch floor subtracted. The
  flag-free control at HEAD reads **25.3066 ms per decode token = 39.52 tok/s at ids sha256
  `56305eee11d6`**, the value of record; all four `gen` 256 runs carry it, including the
  `CROW_GDN_FUSE_IN=0` arm, which is the first time the per-slab fallback has reproduced the ids in
  the SPARSE decode regime (`#62b` proved it at the parity forms only).

  **Seven launches per GDN layer per decode token, 252 per token** — the issue's own first step,
  answered. The row is 1.954 ms floor-removed (3.214 raw) against 2.280 in `#59`:
  `gemv_fp4_mma_g32` (the grouped input projection) **1.000**, `gemv_fp4_mma_d32` (out) **0.658**,
  `delta_rule_step_r` 0.148, `l2norm_repeat` 0.059, `rmsnorm_gated_q` 0.044, `conv_step` 0.028,
  `beta_g` 0.017. **85 % of the row is the two projection launches**; all five recurrent-step
  kernels together are 0.296 ms, and `delta_rule_step_r` is already at 1,527 GB/s = **85 % of this
  card's peak** on the 6.29 MB of state it rewrites per layer. The five projections measured
  separately (`CROW_GDN_FUSE_IN=0`): qkv 320 blocks **1,180 GB/s**, z 192 blocks **948**, b and a
  one block each (7.06 us for 0.069 MB, pure latency), out 80 blocks **483**; grouping the four
  inputs is worth **-0.294 ms** per decode token (per-slab 1.294 against the grouped 1.000),
  `#62b`'s lever measured a third way.

  **The headline comparison of this issue was never like for like.** At batch 1 the GDN projections
  read every weight byte exactly once: 1.1728 GB per decode token at CNQ4.5-M's 4.5 bpw, whose hard
  floor at 1,792 GB/s is **0.6545 ms**. The llama.cpp arm of record is `UD-Q2_K_XL` GGUF at **2.4
  bpw** (`docs/architecture.md` 0.3) = 0.6255 GB per decode token. Reading llama's 0.642 ms against
  CROW's bytes would need **1,827 GB/s = 102 % of this card's peak**, which is impossible; against
  its own bytes it is 974 GB/s = 54 % of peak, where crow's projections run at 707 = 39.5 %. Of the
  1.312 ms that still separates the rows, **0.774 (59 %) is the weight format**, 0.296 (23 %) is
  crow's five recurrent launches and 0.242 (18 %) is projection efficiency — and the `#62a` mapping
  caveat still runs the same way, so 0.642 probably understates llama's true GDN cost.

  **The launch count is not the lever and crow is already ahead on it.** llama.cpp's Qwen3-Next
  linear-attention layer (`src/models/qwen3next.cpp`, read at source 2026-09-18) is 13 graph nodes —
  THREE input-projection `mul_mat`s (`wqkv`, `wqkv_gate`, `ssm_beta_alpha`) against crow's one
  grouped launch, plus `ssm_conv`, `silu`, two l2 norms, `sigmoid`, `softplus`, `mul`, the fused
  `ggml_gated_delta_net`, `build_norm_gated` and the `ssm_out` `mul_mat` — 13 kernels or more before
  ggml's elementwise fusion, against crow's 7.

  **The lever this pass built, measured and did NOT land.** `gemv_fp4_mma_d16` — one 16-row group
  per block instead of two, so the out projection (the one GDN launch that leaves 90 of 170 SMs
  idle) runs 160 blocks of `32*KS` threads instead of 80 of `64*KS`, per-row arithmetic bit-identical
  by construction. It doubles the BLOCK count and keeps the WARP count at 640, and the row does not
  move: **19.00 us per call against 18.27, 466 GB/s against 484**, ids `56305eee11d6` and the
  `run 32` ids of record in both arms on the same binary. Throughput on this kernel body rises with
  total warps and is flat in the block count (640 warps 484 GB/s, 1536 948, 2560 1,180) — the same
  shape `#61` found for `attn_sel_split` from 96 to 768 blocks. **It is not in the tree**: it would
  add a kernel to `KERNEL_SRC` and an env row and buy nothing. At `CROW_MMA_KS` = 4 a 2560-row slab's
  warp count is pinned at 640 by the row geometry, and the only bit-identical way to raise it — more
  k slices — moves the fixed ascending smem reduce. The three levers the decomposition names instead,
  with their numbers, are in `docs/architecture.md` 4.7; the largest is splitting the grouped input
  launch into qkv+b+a (323 blocks) plus the existing z `d32[192x1]`, about **-0.21 ms per decode
  token**, no new kernel needed.

  No engine code changed in this pass and no default moved. `tools/gate-linux.sh decode_out/gate62`
  ALL GREEN.

- **The decode kernel decomposition at the Linux operating point — the attention row of `#61`
  re-measured after the levers, and the GDN row of `#62` with it** (`#61`, 2026-09-18, RTX 5090 /
  Arch Linux, `decode run` on t1-read, 16,064 ids, greedy, 256 tokens / 255 timed steps, context
  16,320; logs `decode_out/61/`). The pre-lever table of `#59` (Windows, nsys, 2026-09-11) was the
  state of record and nsys is not installed on this box, so the pass ran the engine's own
  `CROW_KPROF=1` per-kernel profiler, which charges kernel time plus one launch latency and
  accumulates from load. Prefill is removed EXACTLY by differencing two runs of the same prompt at
  `gen` 256 and `gen` 8 — bit-identical prefill — over their 248 extra decode steps; the per-launch
  floor `CROW_KPROF` adds is 5.0 us, read off the cheapest rows, and the floor-removed column is the
  one comparable with nsys. Two independent run pairs agree to better than 0.4 percent.

  The profile arm sits on the operating point of record: `CROW_GRAPH=1` reads **25.1064 ms per token
  = 39.83 tok/s** at ids sha256 `56305eee11d6`, against the `#38` chain's 39.81 to 39.86 tok/s over
  four D runs at the same sha (`decode_out/38/c1-sdsd.out`). `CROW_GRAPH=0` (which `CROW_KPROF`
  requires) reads 25.8363, and the profiled run 33.5811 / 33.5880 — **`CROW_KPROF` is not a tok/s
  form**, that 7.7 ms is its two syncs per launch.

  | row, ms per decode token | `#59` (Windows, nsys) | this pass (Linux, floor removed) | llama.cpp (`#59`) |
  |---|---|---|---|
  | the 12 attention layers | 4.665 | **3.595** (4.840 raw) | 0.866 |
  | of which `attn_sel_split` | 2.439 | 2.420 | — |
  | of which the selection | `qsa_select_fast` 1.110 | `qsa_select_par_h` + `_e` 0.075 | — |
  | the 36 GDN layers | 2.280 | **1.954** (3.214 raw) | 0.642 |

  The attention row is **-23 percent** and the whole move is the `#61b` selection lever;
  `attn_sel_split` itself is unchanged to 0.8 percent, which is also the two methods agreeing. The
  GDN row is **-14 percent**, the `#62e` geometry lever measured a second way (62e's own pairs read
  -0.447 ms end to end). Ratios to llama.cpp: 4.2 x and 3.0 x, from 5.4 x and 3.6 x.

  **Rank 1 of the whole decode step is neither**: `stage_cold_ca`, the cold-expert staging of `#19`,
  is **11.468 ms per token at 48 calls, 38.2 percent** of the 30.05 ms `CROW_KPROF` attributes.
  Then `attn_sel_split` 2.480 (8.3 percent), `hc_down_inj` 1.522, `gemv_fp4_mma_g32` 1.174,
  `rms_group` 1.004, `gemv_bf16_ws` 0.991, `gemv_fp4_mma[20x10]` 0.980, `sh_gate_up_q` 0.962,
  `gemv_fp4_mma_d32` 0.841, `gemv_bf16_w[31040x1]` 0.762. The full 15-row table and both row
  breakdowns are `docs/architecture.md` 4.6, for `#62` and `#19` to reuse.

- **Why `attn_sel_split` costs what it costs, and what it is NOT** (`#61`, 2026-09-18). Sweeping
  `CROW_ATTN_SPLITS` (4 / 8 / 16 / 32 = 96 / 192 / 384 / 768 blocks on 170 SMs) reads
  **406.4 / 206.7 / 108.6 / 59.7 us** per attention layer, and the fit `8.6 us + 0.776 us x
  ceil(sel_n/S)` holds to better than 1.2 percent at every point — **flat in the block count** from
  0.56 waves to 4.5 waves. So it is not tail-effect and not occupancy bound. It is not bandwidth
  bound: per layer per token the kernel requests 25.26 MB of KV and touches 2.10 MB of UNIQUE KV (a
  **12.0 x re-read, exactly the GQA ratio** — one block per query head, 24 over 2 KV heads), which
  at 201.7 us is 125 GB/s of requests and 10.4 GB/s of unique bytes against this card's 1,792 GB/s.
  Nor compute bound: 50.4 MFLOP per layer in 201.7 us is 0.25 TFLOP/s. What it IS: **0.776 us per
  selected token per block, about 2,250 clocks, for 512 e4m3 BYTE decodes** — `dec_e4m3` is two
  `ldexpf`, a divide and three branches per byte. Hence the `CROW_ATTN_LUT` lever above, which takes
  that term to 0.219 us.

  External leg, read from source on 2026-09-18: llama.cpp does **not** use `flash_attn_ext_vec` at
  this shape (`fattn.cu:617` excludes vec when `gqa_ratio > 4 && K->ne[1] >= 8192`); it picks the
  MMA kernel with `ncols2 = 8` (`fattn.cu:242`), so **one CTA covers 8 query heads and the KV tile
  is re-read 2 x per token where crow re-reads it 12 x**, at 128 threads, `nbatch_fa` = 64 KV rows
  per iteration and cp.async double-buffering (`fattn-mma-f16.cuh:70`). FlashInfer makes the GQA
  group a block dimension (`decode.cuh:685`, `bdy = GROUP_SIZE`) so a KV row is read once for the
  group, with 16-byte vectorised loads and a 256-token floor on the split. vLLM's PagedAttention V2
  is essentially crow's current geometry and has been deleted from vLLM main. In bytes: llama.cpp
  moves about 401 MB of unique KV per token over the 12 layers if its KV is f16 (213 MB at Q8_0; the
  `#59` csv does not name the type), crow moves **25.2 MB** — the sparse
  selection already buys a 16 x cut and crow was still 5.4 x slower, so **the byte count is not the
  lever here, the cost per byte is**. The two byte-level levers the others hold and crow does not
  (one CTA per GQA group, 16-byte vectorised KV loads) both move the reduction order or the lane
  mapping, so both would need the ten-task quality gate, not only parity. They stay open on `#61`.

- `tools/gate-linux.sh` ALL GREEN at this commit — 8 rows `bceba6ff7724…`, 512 rows
  `8387234709271515…`, P8 teacher-forced `3bb3e69edf90…` and the 32 ids of record all unchanged,
  which is what "the filter is off the numeric path" means in bytes. `cargo test --release` 171
  passed, 0 failed; clippy unchanged at **1,422**; `check_env_docs` exit 0 (82 = 82);
  `check_readme_dates` 0 offenders. Measured 2026-09-18.
- `tools/replay-toolcalls.py --think` against this build: 4 rounds, all 200, **0** rounds with a
  reasoning tag in the streamed content. Measured 2026-09-18.
- `tools/gate-linux.sh` ALL GREEN again at the `#68` commit (2026-09-18) — the same four byte
  values of record, `cargo test --release` **174** passed / 0 failed (100 lib + 74 serve), clippy
  **1,422**, `check_env_docs` exit 0 (82 = 82), `check_readme_dates` 0 offenders. The `#68` change
  is one log line and three tests: nothing it does can reach `decode_step`, the sampler or the id
  vector, and the gate is what says so in bytes.
- **The `presence_penalty` scope of `#68`: measured, and there is nothing to fix** (2026-09-18).
  The penalty set is the tokens THIS request generated and nothing else — `enable_dev_sampler`
  (`gen.rs:3641`) uploads a zeroed `mask[V]` and `kernels.rs sample_k` sets `mask[tok] = 1` for the
  token it just drew, so the prompt is never in it, there is no last-n window, a token drawn ten
  times is penalized once (the mask is a `u8` and cannot count), and `arm_sampler` runs for EVERY
  sampled request, so turn 300 of a prefix-cached session starts with an empty set. That is the
  HF/vLLM semantics the card's `presence_penalty` is written in (`probes/p5_STATUS.md:539-545`);
  llama.cpp's windowed `repeat_penalty` is a different knob and the engine implements none.
  Greedy arms no sampler at all, which is why the parity ids cannot move under any of this.
  Consequence recorded with it: `presence_penalty` cannot brake a loop that spans TURNS — the live
  late answers are ONE token long and the set is empty when that token is drawn.
- **The `#68` session replayed through `serve`, `tools/replay-session.py`** (2026-09-18, the whole
  record with every answer is `docs/long-context-goalmode.md`). The stored 574-message history,
  images stripped, renders as **168,928** prompt ids against the live 178,779:

  | context | sampling row | echo of the nudge | single-token answer |
  |---|---|---|---|
  | 120,924 | greedy | 0 of 3 | 0 of 3 |
  | 120,924 | card 0.7 / 0.8 / 20 / 1.5 | 0 of 3 | 0 of 3 |
  | 120,924 | Crow 1.0 / 0.95 / 20 / 1.5 seed 0 | 0 of 3 | 0 of 3 |
  | 168,928 | Crow 1.0 / 0.95 / 20 / 1.5 seed 0 | **1 of 3**, round 0 verbatim | **2 of 3** (`3`, `finish stop`) |
  | 168,928 | **greedy** | 0 of 3 | **3 of 3** (`3`, 0.54 s per turn) |

  So **both late stages of `#68` reproduce with the `#67` tags gone, and greedy is the worst arm**:
  the degeneration is neither the reasoning tag nor the sampler. The greedy onset on this history
  is between **153,755** (healthy: 270 tokens and a `run_command` call) and **163,401** (one token
  `3`); the live session with its 17 images held to prompt 172,599. Prefill 834 to 923 tok/s cold
  at these sizes, decode 37 to 48 tok/s at 124k to 169k of context.
- **The long-context quality gate, `tools/longctx-gate.py`, greedy** (2026-09-18): **5 of 5 Pass,
  0 degenerate at 104,433** prompt ids (98 files of material) and **4 of 5 Pass, 0 degenerate at
  178,553** ids (167 files, 669 history messages) — the first quality reading this engine has above
  16k of context. Decode 35.7 to 42.2 tok/s at 104k and 37.5 to 40.9 tok/s at 178.5k; prefill 894
  to 908 tok/s cold. The one Fail is the shape such a gate exists for: at 178,553 ids the answer
  names the right function, the right operator and the right reasoning but puts it in `f166.rs`
  where the material has `f116.rs` — two digits transposed.
- **What that means for `#68`** (2026-09-18): a CLEAN agentic history of 178,553 ids, which is the
  live session's own context size, does NOT degenerate, while robin's history degenerates from
  about 163k on under greedy. Length is the enabling condition; the content of those tokens — 300
  churning turns, 105 byte-identical nudges, contradictory half-finished tool output — is the cause.
  The engine's decode at 178k is not broken.
- The `#67` filter caught a LIVE tag during the `#68` replay (2026-09-18): on 32 of the 33 requests
  of that session the `[chat]` line reads `think tags stripped 0`, and on the 33rd (the card row at
  120,924 ids, a 404-token answer) the model emitted a `</think>` itself mid-answer and the filter
  removed it — `reasoning chunks 0`, one loud `[chat] reasoning filter` line, the generated ids
  untouched. The normaliser meanwhile strips exactly **67** stored tags per full-history request,
  which is the ticket's own count of tagged assistant turns.
- **Stage 3 of `#68` re-read off the artefact** (2026-09-18): **48 of the 293 answers** of the live
  session are the single token `3` (id 18) with `finish stop`, the first at prompt 172,599
  (`serve.log:6139`), 48 of the last 56 answers; 59 answers are at most 3 ids, and 9 of the 273
  assistant turns repeat the nudge text. The "digit written non-stop" of the report is the client
  concatenating 48 one-token answers, not one runaway generation — every request is correct on its
  own, which is why loop detection belongs to the client.

- **`#38`, the run-position drift of a `serve` rate: measured on Linux, and it is ABSENT here**
  (2026-09-18, HEAD `788fb64`, no engine change; full record with both tables, the twelve machine
  blocks and the log paths in `docs/measurement-coverage.md`). Two chains through the new
  `tools/drift-chain.sh`, one fresh process per run, the t1-read prompt of record (16,064 ids,
  greedy, `max_tokens` 256, 255 timed steps), no warmup discarded because the question IS whether
  run 1 differs from run 4. Chain 1 alternates the arms S D S D S D S D, chain 2 is the four
  consecutive fresh serve loads the issue asked for. Within-arm spread, max over min: **serve
  1.0056** (49.636 / 49.891 / 49.845 / 49.914 tok/s, mean 49.8215) and **`decode run` 1.0012**
  (39.861 / 39.824 / 39.813 / 39.823, mean 39.8304) in chain 1, **serve 1.0009** (49.851 to
  49.896, mean 49.8635) in chain 2 — against the 26.4 % of `decode_out/srv-m2a.log:824` on the
  Windows box (2026-09-10) and the 1.105 largest spread of the four `#37` chains (2026-09-11).
  Neither arm is monotonic in run index; chain 1's first serve run is its SLOWEST by 0.372 % (the
  direction M2a had backwards) and chain 2 does not reproduce even that (run 1 is 0.015 % off its
  mean). The generated ids are one sha256 per arm across every run
  (`e7c17e064ea2…` serve 8 of 8, `56305eee11d6…` `decode run` 4 of 4, and
  `serve[1:] == decode[:255]`), and the engine counters are identical to the digit inside each
  arm — serve 4,494,905 cold of **7,833,120** expert selections, **261,104** PLE rows and 107,227
  fills per request in 8 of 8 runs. Those two counts are the M2a serve figures to the digit
  (issue `#38` table); only the cold share differs, −3.5 %, because the hot set is not the same
  one (N 149 with the `#37` trickle tick). So the cold-bytes hypothesis of the issue cannot be
  tested here — there is no drift left to scale with copy volume — and what the chain bounds
  instead is the converse: **with the copy volume held exactly fixed, run position buys at most
  0.56 % on this box.** Against a clock explanation, harder than `#37` could put it: the sm clock
  before the load read 360 MHz at the first start of each chain and 1,260 to 2,917 MHz at the
  other ten, power 35.6 to 112.4 W, temperature 31 to 53 °C, and both arms held inside 0.6 %
  through all of it. Nothing here concludes anything about the Windows M2a outlier, which needs
  the same form rerun on that box. The `#38` consequence rules are untouched by this commit and a
  recommendation on each of them is in the record; no `serve` number entered
  `docs/architecture.md`.

- **The de-duplicated replay: the 105 identical nudges are NOT what flips the model** (`#68`,
  2026-09-18, RTX 5090, `main` at `b70310a`, greedy, 3 answered turns per point, images stripped;
  `tools/replay-session.py --dedup-nudges`, record in `docs/long-context-goalmode.md` §8,
  artefacts `decode_out/68b/`). The flag collapses consecutive byte-identical USER turns of the
  stored session and leaves every assistant turn and tool result where it was: 574 messages → 475,
  114 user turns → 15, 105 goal nudges → 6.

  | run | history | ids | user turns (nudges) | echo | single-token | decode tok/s |
  |---|---|---|---|---|---|---|
  | F | cut 483, de-duplicated | 158,717 | 14 (6) | 0 of 3 | 0 of 3 | 40.7 / 49.3 / 53.8 |
  | G | cut 573, de-duplicated (the whole session) | 160,589 | 16 (7) | 0 of 3 | 0 of 3 | 46.7 / 37.2 / 38.1 |
  | H | cut 413, nudges KEPT — length-matched control | 158,639 | 39 (33, 30 byte-identical) | 0 of 3 | 0 of 3 | 44.0 / 43.9 / 49.6 |
  | E (cited) | cut 573, nudges kept | 168,928 | 114 (105) | 0 of 3 | **3 of 3** | 44.5 / 42.7 / 43.0 |

  **At matched length the repetition makes no difference**: H (158,639 ids, 30 byte-identical
  nudges) and F (158,717 ids, the repeats collapsed) are 78 ids apart and both answer three sound,
  on-task turns with a tool call each. What the 105 nudges really contribute is LENGTH — about 84
  ids each, 8,339 over the session, 4 % of the 200,000 budget — and removing them moves the full
  history from 168,928 ids, where it degenerates on 3 of 3 turns (run E), to 160,589, where it
  degenerates on none. The de-duplicated arm can never be tested at 163k because that is the whole
  of it. **The flip band with the nudges kept narrows to 158,639 healthy → 163,401 degenerate**
  (it was 153,755 → 163,401). §4's finding stands: a clean 178,553-id agentic history degenerates
  on nothing, so it is the CONTENT of robin's tokens — 300 turns of churn and contradictory
  half-finished tool output — and not the repeated sentence. E is cited rather than re-run because
  the decode path has not moved: `tools/gate-linux.sh decode_out/gate68b` at `b70310a` is ALL GREEN
  at the same four byte values of record (parity 8 `bceba6ff7724`, 512 `838723470927`, P8
  `3bb3e69edf90`, the 32 ids), and the new counter is a log line that touches no id.
- The `#68` repeat counter's first sighting on real material (2026-09-18): round 2 of run F is
  byte-identical to round 1 (`Let me look at the actual rendered picture …` plus the same
  `read_image` call), and the `[chat]` line said `repeat run 2` for it — the shape no single
  request can see, seen.

### Known limitations

- The filter owns a **leading** `<think>` block and **every bare** `</think>`; a `<think>` that
  the model opens in the MIDDLE of an answer is left in the content as ordinary text, because at
  that point the answer has started and the tag can be text the user asked about. If a live
  session ever shows that shape it is a one-line change to the state table
  (`docs/architecture.md` 7.11.16), not a redesign.
- A `<think>` block the model opens and never closes ends the turn as `reasoning_content` with an
  empty `content`. On a stream that decision cannot be taken back — the reasoning deltas are
  already on the wire — and with `enable_thinking false` the block should not be opened at all.
- `#68` (the 300-turn degeneration at 178k context): the tag was its first stage, not its only one.
  The engine side is measured now (`docs/long-context-goalmode.md`) and what is left is not a
  `serve` bug — the open items below.
- The onset of the long-context degeneration is bracketed, not curved (`#68`, 2026-09-18): healthy
  at **158,639** prompt ids (section 8's length-matched control of the same day narrowed it from
  153,755), degenerate at 163,401, under greedy on one history. A proper curve (every
  10k from 120k to 180k, three turns per point) is about 40 minutes of GPU and was not run.
- Nothing separates "this model at 170k" from "this quant at 170k" (`#68`, 2026-09-18): CNQ4.5-M
  has never been compared against a higher-bit container or against llama.cpp above 16k of context.
- There is no numeric (logit) statement above 1,024 rows: the parity forms are 8, 512, P8 and 1,024
  rows, and `tools/longctx-gate.py` scores ANSWERS, not bytes. A quality gate is not a parity gate.
- The engine offers a client no cross-turn repetition signal (`#68`, 2026-09-18). Whether it should
  — a repeated-answer counter on the `[chat]` line, or a deliberately chosen `repeat_penalty` — is
  a feature decision for robin, not a bug fix, and it would be the first knob outside the card row.
- The `#68` replay is text-only: the 17 `image_url` parts of the live history (~11k visual tokens,
  874 to 1,000 each) are dropped by default, so the image path at 178k context stays unmeasured.

## 2026-09-17 — v0.3.0: Linux, the host-memory fix, the refactor, and the two prefill floors

- Branch `main` at `487128d`; fourteen commits `9f12429` to `487128d`, all on one day. The day in order: the Linux port (`9f12429`), the host-memory fix (`0c9feb5`), three refactor cuts (`74c79f2`, `bb9d2ca`, `7ddd296`), the CUDA Rust evaluation (`0667e0b`), the Linux gate script and the code map (`0cf1de5`, `c1a68cd`), the page-cache finding and two hardenings (`f8f75c0`), the PLE prefill floor (`1032bc5`), the per-turn prefill floor (`4004e66`), and three bugs out of robin's live sessions: the tool-call session poison (`e2b9845`), the image-request panic and its VRAM reserve (`8ff2055`), the removed splice buffer (`487128d`).
- The machine is the second environment block of `docs/system-landscape.md` (RTX 5090, driver 610.57.04, Arch Linux 7.2.3-arch1-3, 62.17 GiB RAM, CUDA 13.3.1, NVRTC 13.3.33, rustc 1.98.1) unless a row names Windows. Every engine run is inside the memory-bounded scope, one engine at a time.
- The crate version field stays `0.1.0`, as it did for v0.2.0 and v0.2.1: this file is the release record, `engine/Cargo.toml` has never been bumped.
- The numeric contract held across all fourteen commits: `KERNEL_SRC` byte-identical, no launch geometry, k-order, reduce order or `CROW_*` semantics changed. The Linux values of record are `bceba6ff7724…` (8 rows, = Windows), `8387234709271515…` (512 rows), `3bb3e69edf90…` (P8 teacher-forced) and `117dd8d9d8dc…` (1024 rows).

### Added

- `#15` the Linux port (`9f12429`): `cnq.rs` maps the container with `mmap(PROT_READ, MAP_SHARED)` on unix and keeps the Windows ordering in `Drop`; `posix_fadvise(DONTNEED)` replaces the `FILE_FLAG_NO_BUFFERING` purge and `POSIX_FADV_SEQUENTIAL` the sequential-scan hint; `cuda.rs` loads the graph API from `libcuda.so.1` instead of `nvcuda.dll`; `gen.rs` reads `/proc/<pid>` for `pid_alive`; `hybrid.rs` uses `FileExt::write_at`; `parity.rs` picks the oracle venv python per OS; `libc 0.2` is the one dependency the port added. The Windows path is byte-for-byte the old code under `#[cfg(windows)]`.
- `tools/serve-linux.sh` (`0c9feb5`): the memory-bounded launcher — `systemd-run --user --scope --slice=session.slice`, `MemorySwapMax=0`, `MemoryHigh=MemTotal-8G`, `MemoryMax=MemTotal-6G` computed from `/proc/meminfo`, `CUDA_LIB` for `LD_LIBRARY_PATH`, every `CROW_*` and argument passed through. The same shape Crow uses for `llama-server` on Linux.
- `gen::assert_kernel_defines()` with `kernels::define_u32` (`74c79f2`): `QSA_PAR_BINS`, `SAMPLE_MAXK`, `SAMPLE_PARTS`, `SAMPLE_THREADS` are parsed out of the frozen `KERNEL_SRC` and compared with their Rust twins once per `Engine::load`.
- `engine/src/weights.rs` (`bb9d2ca`): `Fp4` and the six tensor loaders, so `gen` and `vit` share them without reaching into each other.
- `engine/src/boot.rs` (`7ddd296`): `open_model(cnq_default, sidecar_default)` — container, CUDA context and starting `Config` for `decode`, `parity` and `serve`. The returned order IS the drop order, which is what the three bins wrote by hand.
- Fifteen named `Engine` methods for the bins (`7ddd296`): `residency`, `weights`, `logits`, `ple`, `pos`, `history`, `route_log`, `n_ctx`, `qsa_ring_rows`, `has_vision`, `vision_plan`, `build_vision_plan`, `arm_sampler`, `park_sampler`, `unpark_sampler`.
- `docs/cuda-rust-evaluation.md` and `engine/src/cutile_pilot.rs` behind the default-off `cutile-pilot` feature (`0667e0b`), an evaluation artefact that nothing in the engine calls.
- `tools/gate-linux.sh` (`0cf1de5`): the Linux parity gate in one script — parity 8 / 512 / P8 teacher-forced, `decode run` over 32 ids, `cargo test`, clippy and both doc guards, against the values of record, GREEN or RED per item, non-zero exit on any RED. Every expected value carries its provenance in the header.
- `docs/architecture.md` section 8 (the code map: module graph, module by module, the `Engine` API surface, the boot and request paths, the sources of truth, the numeric contract on Linux, the host-memory model), `docs/diagrams.md` diagram 7 (the module graph), the module map at the top of `engine/src/lib.rs`, and the Linux environment block in `docs/system-landscape.md`.
- `cnq::Warm`, `Cnq::warm()`, `cnq::page_runs()` and the process-wide row-fetch reader pool (`1032bc5`): a chunk's PLE row misses are one batch of coalesced, page-deduplicated `pread` reads on a second descriptor with `POSIX_FADV_RANDOM`, 16 threads by default, two queues so a background prefetch never blocks a chunk, poison-tolerant locks.
- `tools/replay-toolcalls.py` (`e2b9845`): a Crow-shaped tool loop against a running `serve` — the engine's own streamed `tool_calls` are fed back as the assistant turn, verbatim, the way `crow_core.py` stores and re-sends them, so a poisoned history shows up as the 400 storm it caused live. It finds the truncating budget itself; `--poison` splices the exact broken turn of the live log, `--refusals` sends four shapes the template cannot render. `8ff2055` added `--write-file` (a multi-line HTML `content` parameter) and `--dump-dir`, and made it print `serde_json`'s own byte window. Exit non-zero when any round is refused.
- `cuda::AllocFailed { what, bytes, free }`, `cuda::try_alloc_zeroed` and `cuda::RequestScope` (`8ff2055`): every `cuMemAlloc_v2` of the engine names itself, its byte count and the free VRAM on failure, and inside a request the refusal is a payload `serve::guarded` can answer instead of a process panic.
- `vit::reserve_bytes(context)` and `vit::reserve_line(context)` (`8ff2055`, final value in `487128d`): the VRAM the planner sets aside for the image path before it chooses N, with its own `[budget]` boot line.
- `docs/env.md` rows `CROW_PINNED_BUDGET_GB` and `CROW_VIT_CACHE_MB` (`0c9feb5`), `CROW_PLE_FETCH` (`1032bc5`) and `CROW_VIT_RESERVE_MB` (`8ff2055`); the table is 82 rows against 82 names in the code, and every `Read at` line number in it was re-read off the `487128d` tree.

### Changed

- **The pinned budget is derived, not assumed** (`0c9feb5`): `manager::derive_host_pinned_budget(cap, log) = min(46 GiB cap, free_for_pin - CROW_RAM_MARGIN_GB)`, applied in `Engine::load` before the planner, with one `[budget]` boot line naming the value and its basis. `geo::HOST_PINNED_CAP` is the CAP now, not the budget; a smaller host is an input to the two-sided planner loop, not a refusal.
- **`free_for_pin` is not `MemAvailable` on Linux** (`0c9feb5`): `cuda::free_physical_ram_parts` returns `MemTotal - (AnonPages + Shmem + SUnreclaim + KernelStack + PageTables + Percpu)`, because the NVIDIA driver's pinned-page pool sits in no `/proc/meminfo` class after an exit yet is reclaimable and is served straight back to the next `cuMemHostAlloc`. Live reading with the pool present, 2026-09-17: free for pinning 60.76 GiB against `MemAvailable` 10.89 GiB.
- **The load leaves no page-cache trail** (`0c9feb5`): the cold-tier fill and the hot-slab staging are ONE ascending sweep per expert tensor (same bytes, same destinations) and `Cnq::fadvise_consumed` drops the pages behind the cursor in 64 MiB batches for every section but `ple` (`CROW_CNQ_PURGE=0` disables it). The order is what closed it: the naive per-range `DONTNEED` only got the page cache from 33 to 17.8 GiB, because the readahead window sailed over every skipped hot expert.
- **`slot::restore` streams** the payload through one reusable buffer instead of a 2.70 GiB `read_to_end`, with the length checked against the size on disk first; **`Vit::image_cache` is bounded** (LRU, `CROW_VIT_CACHE_MB`, default 256 MiB, previously unbounded); **`vit::prep_image` builds one channel plane at a time** (12 → 4 bytes per source pixel; patches and `cs` hashes identical on a 97x61 PNG). All `0c9feb5`.
- **Refactor cut 1** (`74c79f2`), the risk-free duplication and the dependent constants: `cuda.rs` `vram_info` / `to_dev` / `dtoh_t` / `Pinned::alloc_flags` / `write_le<T>` for the ten hand-rolled LE dumps; `cnq.rs` `read_at` behind `read_bytes` / `read_range`, `mag_index`, `best_scale_and_codes`; `residency.rs` `fill_layer_table` and `cold_ptrs` at all six sites; `gen.rs` `mma_ks`, `with_nblk`, `decay_window`, the five `Drop` preambles; `vit.rs` `trace_dump` for the seven trace blocks; `sample.rs` `Rng::from_state` shared by the three probes. Constants: `SAMPLE_*` drive the sampler allocation and both launches, the eleven ViT literals derive from five roots, `geo::CHUNK_ROUND` / `CHUNK_CAP` / `TRICKLE_CHUNK_THRESHOLD` (with `bin/serve.rs::SERVE_CHUNK` deriving from the threshold, `42e2b67`'s lesson in the code), `sample::EOS_IDS` from `geo::PLE_EOS`, `geo::DEFAULT_CNQ` / `DEFAULT_HOTSETS` for twelve path literals, `geo::MIB` / `GIB` for 54 inline divisors. `engine/src` 24,707 → 24,641 lines, code 15,439 → 15,234 (−205), clippy 1494 → 1480; bytes identical.
- **Refactor cut 2** (`bb9d2ca`), dead code out and both module cycles broken: `launch_v` + `launch_sync` and the per-kernel profile moved into `kernels.rs` beside the kernel table, the loaders and `Fp4` into the new `weights.rs`; `gen <-> residency` and `gen <-> vit` are gone and the crate is six clean layers, with no re-export left behind and no launch argument, grid, block, stream or order changed. Removed: `run_single_layer`, `LoadReport`, `manager::kv_row_offset`, `gen::gdn_reg_on`, `tokenizer::decode_with_specials`, `coldtier::e2m1_mag`, `bin/residency::totals`, `mma_probe`'s dead E2M1 table, `residency::SIDECAR_SUFFIX`, three dead `let _ =`, six written-never-read fields, and six kernel registrations never launched (`gemv_f32`, `dequant_fp4_flat`, `bf16_to_f32`, `acc_scale`, `gemv_fp4_mma_g`, `gemv_fp4_vit` — host table only; `KERNEL_SRC` byte-identical and the CUDA source stays). `engine/src` 24,641 → 24,595 (code −110 net of +135 new safeguard and test code), clippy 1480 → 1426, build warnings 16 → 1, `#[allow(dead_code)]` 2 → 1.
- **The two-engines safeguard** (`bb9d2ca`, follow-up to `0c9feb5`): `cuda::other_cuda_fd` scans `/proc/<pid>/fd` for another process holding `/dev/nvidia-uvm`; when it finds one the budget falls back to `MemAvailable` and the boot line says so. `/dev/nvidia0` and `/dev/nvidiactl` are the wrong test — measured 2026-09-17, Hyprland, quickshell, Xwayland and GTK hold them permanently and own no pinned pool, and testing for those refused the engine's own operating point. Measured alone: budget 46.00 GiB, free for pinning 60.69 GiB; with a second engine alive (`CROW_LOCK=0`): budget 6.18 GiB from `MemAvailable` 9.18 GiB — a refusal instead of a second 45 GiB pin.
- **Refactor cut 3** (`7ddd296`), the `Engine` API surface: 43 `pub` fields and no private ones became `cfg` `pub`, 8 `pub(crate)` (`st`, `ple`, `pos`, `history`, `done_blocks`, `graph_exec`, `cap_stream`, `route_log`) and 33 private, plus the fifteen methods above; one of the old `pub` fields was dead (`cnq: *mut Cnq`, null at construction, never read — `pub` had hidden the lint). The R3 band: the sixteen env reads inside the 48-layer loop are five `OnceLock`s, `env_flag!` collapses twelve four-line bodies, `geo::env_parse<T>` serves the eighteen number parses with every filter and clamp kept at the site, and `Engine::upload_chunk_scalars` parameterizes the three per-chunk scalar blocks without ever uploading a superset. Host code +7 net (96 removed, 103 added), clippy 1426 → 1422.
- Docs: `docs/env.md` names `env_flag!` / `env_parse` instead of the deleted helpers, `CROW_DUMP_H` points at `gen::dump_h()`, the `CROW_CNQ` / `CROW_HOTSETS` read sites point at `boot::open_model`, and the stale `CROW_CHUNK` note that said `serve` pins 4096 is corrected (it pins 2048, and `SERVE_CHUNK` derives from `geo::TRICKLE_CHUNK_THRESHOLD`). `README.md` and `docs/system-landscape.md` carry the platform truth; `.github/workflows/ci.yml` runs its four jobs on ubuntu-latest. After the code map the doc work continued with the day: `docs/architecture.md` 7.11.14 and 7.11.15 (the `arguments` contract and the image path), 7.13 (the vit reserve), 7.14 (the PLE row path and what a warm turn pays), 2.1 (the host-memory and VRAM model), and `docs/diagrams.md` diagram 7.

### Fixed

- Three Windows runtime dependencies that compiled on Linux and then misbehaved silently (`9f12429`): `nvcuda.dll` for the graph API (`serve` forces `CROW_GRAPH=1` and panicked), `kernel32.dll` `GlobalMemoryStatusEx` returning 0 (which silently disabled the pinned-RAM guard), and `tasklist` returning `Err` → `true` (a stale `.engine.lock` panicked every start).
- `engine/src/cuda.rs` is LF with `\0` escapes instead of CRLF with eight raw NUL bytes (`9f12429`): plain `grep` treated it as binary and skipped it, which hid it from every refactor sweep.
- `#15` **the engine refused every second start on Linux**: the `MemAvailable`-based RAM gate could not see the driver's pinned-page pool, so a second boot read ~9 GiB free against a 43.51 GiB tier. With the derived budget, HEAD booted eight times in a row in that state, including two back-to-back boots whose second ran at `MemAvailable` 9.30 GiB (`0c9feb5`, `bb9d2ca`; gate battery item 11).
- `#15` **the Windows boot of 2026-09-14 died in the same gate** ("refusing to pin 44.62 GiB with only 46.47 GiB physical RAM free", `decode_out/hotfix-serve.log`). That gate is now the backstop behind a budget that is derived from the host it runs on.
- The boot line under-reported ViT scratch by 68 MiB: the hand-computed 41,024 bytes per patch is replaced by the allocator's own count of 58,496 B over the twelve allocations (`74c79f2`).
- One blocking D2H per decode token instead of two: the redundant `_prof_tok` `dtoh_i32` is gone, ids and logits proven unchanged (`bb9d2ca`).
- `ThreeStates::allocate` plans the state sizes once instead of on every clamp iteration — same value, about 1,000 fewer `getenv` calls per boot (`bb9d2ca`).
- **The 1024-row parity form's throughput reading was a page-cache reading, not an engine reading** (re-measured 2026-09-17, 13:22–14:22 local, HEAD `c1a68cd` against `0c9feb5`, 49 engine runs plus two full gate batteries, logs in `decode_out/taskg/` with the payload hashes in `SHAS.txt`). That form's wall clock on this machine is HOST-side I/O: **10,192 of its 16,400 PLE row requests miss** the in-process row cache during prefill and **16 of 16 miss per decode step**, and every miss is a read out of the container's 26.82 GiB `ple` section. Whether a miss costs ~0.9 ms (NVMe) or ~0.09 ms (page cache) is decided by the PREVIOUS process, because `Cnq::drop` → `cnq::purge_cache` issues `posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED)` over the whole container at every exit unless `CROW_CNQ_PURGE=0` — unchanged since `9f12429`, and the comment at `engine/src/cnq.rs:54` already said it ("the purge also cools the PLE rows the next process would have found in standby"). So the number a 1024-parity run prints belongs to its predecessor, not to its own commit. The `0c9feb5` arm of the bisect of record was the one run that followed `D-pool-nopurge`, the only run in that battery with `CROW_CNQ_PURGE=0` and therefore the only one that did not cool the file; the `9f12429` reference read 266.7 tok/s because the pre-fix load left a ~33 GiB page-cache trail on a machine at `MemAvailable` 49.6 GiB. The GPU side never moved: with `CROW_PROFILE=1` the GPU half of a decode step reads **26.85 ms/step on `0c9feb5` cold, 26.83 on HEAD cold and 26.84 on HEAD warm** while only the host `ple` half moves (25.32 / 21.80 / 13.04 ms/step) and prefill swings 10.69 s → 1.38 s at identical miss counts; the `CROW_KPROF=1` per-kernel tables of the two builds are the same table (total **225.4 against 224.1 ms/step** in matched state, every grid tag and every call count identical — `qsa_scores_par[512x1]`, `qsa_select_par_h[32x1]`, `gemv_fp4_mma_g32[515x1]`, `stage_cold_ca[40x1]` on both, which is also what clears the `QSA_PAR_BLOCKS` → `QSA_SCORES_BLOCKS` rename and `mma_ks()` — and no entry off by more than 0.86 ms, that one in HEAD's favour). Nothing in refactor cut 1 is involved and **no code changed**: what changed is the reading and the record.
- **`CROW_KPROF=1` under `CROW_GRAPH=1` no longer aborts mid-run**: the profile syncs the stream before and after every launch, a sync inside the open decode-graph capture is `CUDA_ERROR_STREAM_CAPTURE_UNSUPPORTED`, and the pair used to die on the first captured decode token. `Engine::load` now refuses it before anything is allocated — milliseconds instead of a 24 s load — with one message naming both switches, the CUDA reason and the escape (`CROW_GRAPH=0`). No new switch; `CROW_STAGE_DMA=1` already forces the graph off through `graph_on()`, so that combination still profiles. `docs/env.md` carries the rule on both rows.
- **A CUDA error on a teardown path no longer turns a panic into `abort` plus a gigabyte of core**: `cuda::ck` panics as before, except while `std::thread::panicking()` is on. During an unwind the only code that runs is the destructors, and a second panic raised there is "panic in a destructor during cleanup" — an immediate SIGABRT (measured 2026-09-17: 958.7 MB and 958.6 MB of core from the two runs of the trap above). The teardown error is logged as `[drop] <call>: CUDA error: … - ignored, the unwind continues` and the process exits by the original panic; `free_dev` and `Pinned::free` name their call in that line. Every free is byte-for-byte the same `cuMemFree_v2` / `cuMemFreeHost`, and no non-unwinding path changes. Verified by re-running the old trap with the refusal patched out: before, two panics and SIGABRT; after, **one panic, one `[drop]` line, exit 101, no core**.

- **The PLE prefill floor: the row read was a page fault, and a chunk's misses were serial** (TASK H, 2026-09-17, on the `-M` container rewritten uncompressed; every run in the memory-bounded scope, one engine at a time). `Ple::ensure_rows` reads each miss as one 108-byte `Cnq::read_range` at a random offset in the 26.82 GiB `ple` section, and until now that went through the container mapping. A COLD fault of that mapping costs **~1.09 ms** against **~0.20 ms** for a `pread` of the same row, of which 0.094 ms is the device: the fault runs the file's readahead state, and `Cnq::open` sets `POSIX_FADV_SEQUENTIAL` on that descriptor, which doubles the window — a quarter megabyte read per 108-byte row. Two changes, no byte of output different: `Cnq::read_at` reads `ple` with `pread` on unix (windows keeps the mapping, where it IS the read path), and every miss of a chunk is fetched TOGETHER, because a chunk's row addresses are a pure function of its ids and are known before the forward pass (`Cnq::warm` / `page_runs` / the reader pool, `CROW_PLE_FETCH`, default 16 threads; the existing next-chunk prefetch thread issues the same batch on a second, lower-priority queue). 1024-token form, cold, chunk 1024, **10,192 of 16,400 rows missing on every arm**:

| arm | prefill | PLE host time per decode step | decode step |
|---|---|---|---|
| `f8f75c0`, mapping, page cache cold | 11.14 s, 92 tok/s | 27.07 ms | 54.56 ms, 18.3 tok/s |
| `f8f75c0`, mapping, the `ple` pages warm | 1.41 s, 728 tok/s | 0.08 ms | 26.46 ms, 37.8 tok/s |
| `f8f75c0` with `CROW_MMAP=0` (`pread`), cold | 2.04 s, 502 tok/s | 1.49 ms | 27.52 ms, 36.3 tok/s |
| this build, `CROW_PLE_FETCH=0` (no batch), cold | 2.05 s, 499 tok/s | 1.54 ms | 27.56 ms, 36.3 tok/s |
| this build, 8 threads, cold | 1.44 s, 711 tok/s | 0.63 ms | 26.81 ms, 37.3 tok/s |
| this build, **16 threads (default)**, cold | 1.39 s, 735 tok/s | 0.53 ms | 26.62 ms, 37.6 tok/s |
| this build, 32 threads, cold | 1.37 s, 747 tok/s | 0.45 ms | 26.67 ms, 37.5 tok/s |
| this build, `madv` (`MADV_WILLNEED`), cold | 1.38 s, 741 tok/s | 0.31 ms | 26.61 ms, 37.6 tok/s |

  **The fixed build cold is the old build warm** (735 against 728 tok/s): what the fix removes is the whole cold-start penalty, and the old binary was never slow at READING PLE rows, only at MISSING them. The warm row had to be built deliberately — the pages were left behind by a run of the fixed build, because `f8f75c0`'s own exit purge takes them away — and an earlier attempt at it (two runs with `CROW_CNQ_PURGE=0`, reading 10.66 s) was void, because `CROW_CNQ_PURGE=0` also switches off `fadvise_consumed`, so the 100 GB load flooded the page cache and evicted the 40 MB of PLE rows before the prefill asked for them. Three runs each for the 16 / 32 / `madv` rows; `16` is the default because it sits inside the spread of `32` and, unlike `madv`, needs no mapping and reads the bytes instead of hinting at them. Device floor, 2000 random 4 KiB reads over the same section: 10,607 IOPS on one thread, 201,305 at 16, 280,198 at 32, 251,588 at 64, 209,652 at 128. Paired cold-state runs, `f8f75c0` against this build, alternating, twice each: **parity 1024 rows 94.3 / 94.4 against 739.6 / 740.6 tok/s prefill with sha256 `117dd8d9d8dc…` and 1,021,091,840 B on all four**; `decode run decode_out/srv-a5-t1read-ids.json 128` (16,064 ids) 26.88 s / 598 tok/s and 26.71 s / 601 against 16.60 s / 968 and 16.66 s / 964 prefill, 27.66 / 27.69 ms against 27.19 / 27.22 ms per decode token (36.2 / 36.1 against 36.8 / 36.7 tok/s), 257,040 rows and 104,652 misses on all four.
- **The exit purge keeps the `ple` pages** (TASK H, 2026-09-17): `Cnq::drop` issues its `POSIX_FADV_DONTNEED` around the `ple` byte range instead of over the whole file, so the one section the next process reads at random is the one it finds warm; `(0, 0)` — no `ple` section — purges everything as before, `CROW_CNQ_PURGE=0` still purges nothing, and windows is unchanged. It costs the next process nothing, because page cache is not part of the reclaim-aware `free_for_pin` of `0c9feb5`. Measured on the 1024-token form, run 2 starting with whatever run 1 left behind: `f8f75c0` 10.84 s → 10.71 s (94 → 96 tok/s, the mapping ignores the page cache either way), this build **1.39 s → 1.36 s (734 → 753 tok/s)**.
- **`CROW_PLE_CACHE_MB` stays at 128 MB** (TASK H, 2026-09-17), decided by measurement and not by the F49 chains' 512. On the six-turn serve replay the whole session requests 63,472 PLE rows against **1,198,372 slots** at 128 MB, so capacity is never the binding constraint and the misses are compulsory — first touches of hash-scattered rows. Miss rate 10.64 % at 128 MB, 10.66 % at 256, 10.65 % at 512, while the planner's hot set falls **N 155 → 154 → 152**: the bigger cache buys nothing and costs two hot experts per layer. Mean per-turn prefill over the three: 267.2 / 268.1 / 271.0 ms.

- **A warm serve turn's prefill is a cold-tier cost, not a PLE cost.** Replayed in the reported shape (a 3,296-token cached prefix, six turns of 39 to 101 new ids, greedy, `CROW_CHUNK=2048`, one chunk per turn), `f8f75c0` spends 251.5 to 319.0 ms of prefill per turn (mean 276.2 / 277.6 over two runs) and this build 241.4 to 311.1 ms (mean 267.2 / 267.5). The per-turn prefill does not follow the PLE miss count: the turn with 1,218 misses is the fastest at 241.4 ms and the turn with 128 misses the slowest at 311.1 ms. Prefill against new ids reads as **~225 ms fixed per prefill call plus ~0.43 ms per token**, and a 64-token cold prefill measures 0.27 s on its own — that floor is the per-chunk pass over the cold expert tier, which `Engine::prefill` already names in a comment ("every chunk costs one full PCIe pass over the cold tier"). The PLE work is now a few ms of it. The next measurement on multi-turn latency belongs there. **Answered by TASK I** (the next item): it is the staged cold-expert bytes of the chunk, and the fixed-plus-linear fit was an artefact of a byte count that saturates in the first ~40 tokens.

- **The warm-turn floor was the hot set, not a fixed cost per prefill call** (TASK I, 2026-09-17, same machine, same replay, every run in the memory-bounded scope, one engine at a time). TASK H left a warm serve turn reading as "~225 ms fixed per prefill call plus ~0.43 ms per token". Instrumented with CUDA events around the four spans of every prefill chunk and a byte counter on the staging copies, that fixed cost is **bytes**: a chunk stages every cold expert its tokens route to, once per layer, over PCIe — 10.1 GB for a 42-token turn, 3,661 copies of 2.76 MB, 76 cold experts per layer of the 144 distinct ones its 420 combos route to, at 48-49 GB/s against the 55 GB/s `pcie_probe` measures for that copy. It is three quarters of the turn, and it does not follow the token count: over the six turns the staged bytes read 10,122 / 11,504 / 9,704 / 8,737 / 8,247 / 7,462 MB for 42 / 66 / 62 / 39 / 101 / 85 new ids, so the 101-token turn moves 19 % fewer bytes than the 42-token one. Three changes, no byte of output different (the 6-turn replay's generated ids are identical, `1032bc5` and this build, four runs):
  - **the stream trickle ticks every 8 decode tokens, not every 16** (`geo::TRICKLE_EVERY`, the long-context policy of #17). Every expert the hot set does not hold is 2.76 MB per layer per turn for as long as the conversation stays on the same material, and #17 cut the interval on single-prompt evidence, where the hot set has one prefill to be wrong about. Ticking twice as often halves the tokens the hot set needs to follow the session: staged bytes on the last turn 7,462 → 6,301 MB. Arms measured on the replay (prefill ms mean over turns 1-6, decode ms over the six turns, swaps): every 16 → 267.7 / 5,219 / 4,030; every 8 → 231.3 / 5,033 / 9,160; every 4 → 209.3 / 5,128 / 16,134; `CROW_ADAPT_STREAM=0` (no trickle, N 148 → 155) → 312.5 / 6,056 / 0, with the staged bytes GROWING over that session, 9,790 → 12,809 MB. `every 4` is measured and not taken: 22 ms more prefill for twice the swap traffic again, a total turn time inside 0.6 %, and a four-token ranking window. Both stay reachable through `CROW_ADAPT_STREAM=1` plus `CROW_ADAPT_EVERY`. On the single-prompt long-context form the same change is neutral-to-better (`decode run` on the 16k t1-read prompt, 128 tokens: 27.21 / 27.21 → 27.09 / 27.09 ms per decode token, cold experts per token 236.1 → 222.5).
  - **the prefill tile-group loop runs over the tiles that exist** (`gen.rs`, the `CROW_PF_ASYNC=2` branch): the host bound was the worst case a host that has not seen the plan must assume (`E + t * TOPK / 8` = 9 groups of 64 for a 42-token chunk) while the plan holds 144 tiles = 3 groups, and the six empty groups per layer launched five kernels each that every block exited on `ti >= n_tiles` and made the compute and staging streams wait on each other's events for nothing. The `ce` branch reads the plan back anyway, so the bound is now the tile count. 2.3 ms per warm turn, 1,440 launches per chunk not issued.
  - **the prefix-cache slots are faulted in at boot** (`cache.rs`): `vec![0f32; n]` is a `calloc`, so the 130.6 MB of a slot was the shared zero page until the first `snapshot` WROTE it, inside a request. Measured DtoH of snapshot 1: 45.5 / 43.7 / 38.4 ms for the first three requests of a process against 8.8 ms once the pages exist. One volatile store per 4 KiB at allocation moves those faults to the load.

  Paired cold-state runs, `1032bc5` against this build, alternating, twice each. The 6-turn replay (3,296-token cached prefix, turns of 42 / 66 / 62 / 39 / 101 / 85 new ids, greedy, `CROW_CHUNK=2048`), means over the warm turns 1-6:

| | prefill ms | time to first token | decode ms | whole session |
|---|---|---|---|---|
| `1032bc5` | 267.1 / 267.1 | 296.2 / 295.5 | 870.4 / 870.3 | 10,939.7 / 10,960.2 ms |
| this build | 228.3 / 228.7 | 247.4 / 247.1 | 839.8 / 839.0 | 10,461.2 / 10,458.0 ms |

  Time to first token is `reset` + `prefill` + the snapshot DtoH, which is what the caller waits for; turn 1 alone goes 324.1 → 271.8 ms. The cold turn 0 is unchanged (3,413.5 / 3,436.5 → 3,429.9 / 3,431.9 ms), `decode run` on the 16k t1-read prompt is unchanged in prefill (16.53 / 16.48 → 16.49 / 16.50 s) and 0.12 ms per token better in decode, and the 1024-row parity form holds its value of record on all four runs (739.8 / 738.3 → 740.2 / 742.8 tok/s, sha256 `117dd8d9d8dc…`, 1,021,091,840 B).
- **One truncated tool call killed a whole Crow session, and the log said only `400 Bad Request`** (TASK J, 2026-09-17; reproduced without robin with `tools/replay-toolcalls.py`, logs in `decode_out/taskj/`). After 79 tool rounds in goal mode, EVERY `POST /v1/chat/completions` answered `400 {"error":"chat template render failed: invalid operation: cannot convert value into pairs (in chat:136)"}` with the body growing ~350 B per turn (358,654 → 386,444 B). **The shape**: a tool call cut off at `max_tokens` left the streamed `arguments` fragments UNTERMINATED — `{"path":"` — on purpose, so that Crow's `json.loads` would fail rather than run half a command. Crow concatenates those fragments into a string (`crow_core.py:5068`), stores the assistant turn with that exact string (`:3756-3761`), re-sends the whole history every turn (`:3783-3785`, `:4866`) and, being append-only, can neither repair nor drop it after a 400 (`:13485-13487`, `:3592-3607`) — while `chat_template.jinja:136` iterates `tool_call.arguments|items`, which needs a mapping. One bad turn, and every later turn of that session was refused. Measured A/B on the replay harness at the truncating budget: pre-fix round 0 stores `arguments=NOT JSON '{"path":"'` and rounds 1–3 are three consecutive 400s at 657 / 741 / 823 B; post-fix the same round 0 stores `'{"path":"","_truncated":true}'` and rounds 1–3 are 200. **The fix is on both ends.** The engine can no longer produce it: an abandoned call (truncated at `max_tokens`, or markup the parser gives up on) closes its own arguments object through `toolcall::ToolStream::close_args_truncated` — the value in flight is closed the way `</parameter>` would have closed it, then `,"_truncated":true}` — so the concatenation of the `Emit::Args` fragments of every index the parser NAMED is a parseable JSON object for every input there is. The marker keeps the safety the old contract bought: no tool declares `_truncated`, so half a command is still not runnable, and `finish_reason` still stays `stop`/`length` for a malformed call. And `serve::normalize_messages` no longer lets a non-mapping reach that template line at all: a string that parses to an object becomes that object (the A7 path, unchanged), the empty string and an absent value stay as they are (the template's own `arguments != ''` guard skips them), JSON `null` becomes `{}`, and **any other value becomes `{"_raw": "<the value, verbatim>"}`** — the model must still see what the previous turn asked for, and the leading underscore says it is not a declared parameter. **The new diagnostic**: every rewrite writes `[chat] normalised: message <i> tool_call <j> function.arguments is <kind> the template cannot iterate, rendered as ...: <first 200 bytes>`, and every messages 400 now logs the reason plus one line per message (role, content kind, and per tool call the name and the first 200 bytes of `arguments`) — the live log had only `400 Bad Request`, which is why this cost a session to find. The neighbouring template hazards were measured the same way and each one now either renders or is refused by `serve::check_messages` BEFORE the render with a body that names the message index and the field: a `content` that is not a string, a list of parts or null; a content list part that is not a text or `image_url` block; a `tool_calls` that is not an array (the template silently DROPPED it); a tool call whose `function` or `function.name` is wrong; a role the template does not render; a system message that is not first. A text-only content list, a tool result carrying an image (`crow_core.py:13739-13741`), a `null` content and a `reasoning_content` of any type all render and are served. Six new tests (three in `toolcall.rs`, three in `bin/serve.rs`; `cargo test --release` 147 → 153), clippy unchanged at 1422, the A7 oracle fixture renders byte-identical, and the serve smoke is identical modulo id/created/timings with the slot at `2b7d129afdc4…`, 131,820,320 B. Contract tables: `docs/architecture.md` 7.11.14 and the `serve.rs` / `toolcall.rs` module docs.
- **TASK K, an image request could take the whole server down** (robin's goal-mode session, 2026-09-17): after two cached images a `POST /v1/chat/completions` ended in `thread 'main' panicked at src/cuda.rs:252:5: CUDA error: CUDA_ERROR_OUT_OF_MEMORY`, and Crow got `Connection refused` for the rest of the session. Both images were cache HITS, so the tower — and with it the lazy `ensure_scratch` — never ran; the only device allocation left in that window is the per-request spliced embedding buffer, `sum(n_visual) * 2560` f32 = 1,660 x 2560 x 4 = **17,000,000 B (16.21 MiB)**. The cause is that NOTHING in the #VIT path was planned: the cap-sized tower scratch (239,599,616 B = 228.5 MiB), that splice buffer and the interleaved-mrope span tables (48.8 MiB at `n_ctx` 200,000) are all allocated inside a REQUEST, while the planner had already maximised the hot set against free VRAM at boot — the image path was living in the ~400 MiB that `manager::SAFETY` (512 MiB) plus `LAUNCH_SLACK` (128 MiB) had left after the ~220 MB NVRTC module load. Two fixes. **(1) The planner reserves it up front** (`vit::reserve_bytes`, added to `pending` in `Engine::load` when `CROW_VIT` is on, `CROW_VIT_RESERVE_MB` to pin or disable it): 317.3 MB at first (tower scratch + a four-image splice allowance + the mrope span), named on its own `[budget]` line, cost **N 157 -> 155** measured on two boots of the same binary at the same free-at-start (22.86 GiB), VRAM used 30.83 -> 30.58 GiB; `CROW_VIT=0` reserves nothing. `serve` now also clamps the generation budget BEFORE it arms the mrope tables, so the span is at most `n_ctx` (it used to use the raw `max_tokens`, and a request refused with 413 had already allocated its tables). **(2) A request may no longer panic the process**: every `cuMemAlloc_v2` goes through `cuda::try_alloc_zeroed(what, bytes)`, the failure line and the response body name the allocation, its byte count and the free VRAM, and inside a `cuda::RequestScope` the refusal raises `cuda::AllocFailed` as the panic payload — `serve::guarded` catches exactly that, frees what was taken, resets the engine and answers **503**. Reproduced both ways with a second CUDA process holding the card at ~50 MiB free: `e2b9845` panics at `cuda.rs:252` and `/health` is DEAD, this build answers two 503s naming `the vit block MLP scratch (70516736 B = 67.2 MiB); free VRAM 76.6 MiB` and `/health` is ALIVE. New tests (`cuda.rs` `mod alloc_failure` injects the failure without a GPU; `vit.rs` `mod reserve` pins the reserve's byte counts and the `[budget]` line, and `487128d` updated both to the two-row reserve). **The 12-image follow-up, `487128d`, the same day**: with the reserve in place robin's next session answered 503 on every turn - Crow re-sends its whole image history, 12 images = 9,939 visual tokens, and the per-request spliced embedding buffer (97.1 MiB) did not fit the 4-image reserve. That device buffer was never read: the prefill splices from the plan's host copy row by row into the chunk's embedding upload. It is removed (`vit.rs` `VisionPlan` holds host rows only), the reserve is tower scratch + mrope span (277.3 MB), and no per-request VRAM is allocated for images at all - the image count is bounded by the context, not by VRAM.
- **TASK K, the second `arguments` case was not the parser** (the same log's `[chat] normalised: message 46 tool_call 0 ... rendered as {"_raw": ...}`, 3,112 bytes of `write_file` arguments with an HTML `content` parameter). The visible prefix is correctly escaped and the call carries no `_truncated` marker, so `ToolStream` was the suspect; it is not. The invariant of `e2b9845` is now SWEPT rather than argued: `toolcall.rs` `mod arguments_contract` runs ten hand-written hazards (a full HTML5 document with quotes, backslashes, tabs, a `<script>` block, a non-BMP emoji, U+007F and a vertical tab; a value that itself contains `</parameter>`; a value containing `<parameter=`; an undeclared tool; a parameter name with a quote, a backslash and a tab; empty, CRLF, trailing-backslash, no-separator and cut-mid-HTML values) at eight piece sizes with the types declared and undeclared, plus 4,000 random markups over the marker fragments, control characters, `U+2028`, `U+FFFD` and non-BMP scalars at four piece sizes — every index concatenates to a parseable JSON OBJECT, for every input. End to end, `tools/replay-toolcalls.py --write-file` drives three `write_file` rounds through the real model and the real server (HTML document, markdown with a Windows path, JSON config with escaped quotes): all 200, all `arguments=object`. What changed is that the contract is now CHECKED where it is produced — `serve` accumulates the fragments per call index exactly as Crow does and logs `[chat] BUG: the arguments of tool call N are not a JSON object - serde_json: ...` after the flush, and the `stream:false` document repairs a violation to `{"_raw": ...}` before it leaves (`args_object_or_raw`) — and DIAGNOSABLE where it is consumed: the `[chat] normalised` line carries `serde_json`'s own message and the byte window it stopped in (`error_window`), so the next occurrence names the defect instead of only its coordinates. Four new `bin/serve.rs` tests; with `toolcall.rs` `mod arguments_contract`, `cuda.rs` `mod alloc_failure` and `vit.rs` `mod reserve` that is twelve, and `cargo test --release` 153 → **165 passed, 0 failed** (98 lib + 67 serve).

### Measured

The final battery ran 2026-09-17, 12:11–12:52 local, HEAD `0667e0b` against the reference build `9f12429` (the last commit before the fix and the three cuts), full record in `decode_out/final/GATES.md`.

| gate | expected | got | verdict |
|---|---|---|---|
| parity 8 rows | `bceba6ff7724…`, 11,919,360 B | identical | GREEN |
| parity 512 rows | `8387234709271515…`, 512,532,480 B | identical | GREEN |
| P8 teacher-forced | `3bb3e69edf90…`, 512,532,480 B | identical | GREEN |
| parity 1024 rows | no Linux value existed | reference first: `117dd8d9d8dc…`, 1,021,091,840 B, then HEAD byte-identical twice | GREEN, new Linux value of record |
| cross form `CROW_CHUNK=2048` on the 512 form | the 512 value | identical | GREEN |
| `decode run` over 32 ids | the 32 ids of record | 32 of 32 | GREEN |
| ten-task gate at ids level against the `final4` WINDOWS record | — | 1 of 10 byte-identical, 9 flip on a near-tie | see the note below |
| serve smoke | the responses of record, slot `2b7d129afdc4…` | identical modulo id, created and timings; slot 131,820,320 B | GREEN |
| throughput | no regression | paired against `0c9feb5` on three forms in matched machine state, six runs each on the 16k form: HEAD equal or faster on every one (re-measurement of 2026-09-17, below) | GREEN |
| tests, clippy, doc guards | 144 / 1422 / exit 0 | 144 passed 0 failed, 1422, `code 80, doc 80`, 0 offenders | GREEN |
| two back-to-back boots, no reclaim | both boot and match | both `bceba6ff7724…`, the second at `MemAvailable` 9.30 GiB | GREEN |

Ten of the eleven items are green. The ten-task item is the toolchain, not the refactor, and the evidence is quantitative: HEAD's logits are bit-identical to the `9f12429` reference on every form both builds can run (1024 rows measured on both today; 8, 512 and P8 at the reference's own values of record), greedy ids are a pure function of those logits, the `final4` records are a WINDOWS record, and each flip is a near-tie two orders of magnitude inside the documented platform drift — teacher-forced probes score the exact flip position at **0.0371** logits for t2-write (`264` at 20.383991 against the record's `1407` at 20.346920) and **0.0758** for t3b-debug-syn (`31626` at 19.788471 against `33041` at 19.712648), against a measured Windows-vs-Linux drift of up to 7.0. `t6b-reason-multi` reproduces the Windows record over 1,024 ids and 2,006 characters byte for byte, which no broken prompt, tokenizer, template or harness could do. What would close it formally is a ten-task run on the `9f12429` build; it is blocked (below), and the exact command is `GATES.md` section 6.

Host memory, 2026-09-17, 1 s poll during the tier fill, same binary with `CROW_CNQ_PURGE` as the A/B switch, both arms `bceba6ff7724`:

| quantity | before | after |
|---|---|---|
| min `MemFree` during the load | 1.15 GiB | 7.70 GiB |
| max `Cached` during the load | 33.04 GiB | 5.14 GiB |
| 8-row load | 44 s | 25 s |
| second start in the same state | refused by the RAM gate | boots, `[budget]` reads free for pinning 59.31 GiB against `MemAvailable` 9.25 GiB |

Throughput, HEAD `0667e0b`, `decode run … 128` on `decode_out/srv-a5-t1read-ids.json` (16,064 ids), twice, identical id traces: prefill 25.38 s = **633 tok/s** and 25.44 s = **631 tok/s**; decode mean 27.17 ms = **36.8 tok/s** and 27.12 ms = **36.9 tok/s** at context 16,192. The Windows `final4` t1-read record reads 36.80 tok/s decode on the same prompt (2026-09-05; its harness counts the first token in, so the two are within noise of each other).

Throughput paired against `0c9feb5`, 2026-09-17 13:22–14:22, every run inside the bounded scope, one engine at a time, alternating builds (and with the order flipped on the 16k form as a control), each measured run preceded by a purging run so that all of them see the same cold container:

| form | `0c9feb5` | HEAD `c1a68cd` |
|---|---|---|
| parity 1024, prefill | 95.7 / 93.4 tok/s | **95.8 / 94.4 tok/s** |
| parity 1024, 4 decode steps | 60.3 / 55.7 / 59.7 / 52.1 and 57.0 / 51.0 / 51.9 / 46.5 ms | **54.9 / 52.1 / 54.3 / 48.2 and 52.1 / 54.0 / 54.5 / 47.8 ms** |
| parity 512, prefill | 269.5 / 278.1 tok/s | **275.8 / 280.8 tok/s** |
| parity 512, 4 decode steps | 19.0 / 15.4 / 15.5 / 15.5 and 18.9 / 15.3 / 15.5 / 15.3 ms | **19.0 / 15.5 / 15.3 / 15.5 and 18.9 / 15.3 / 15.6 / 15.3 ms** |
| `decode run … 128`, 16,064 ids, prefill (6 runs) | 631.95 tok/s mean (627.8–636.4) | **630.55 tok/s mean (625.9–633.7)** |
| `decode run … 128`, decode (6 runs) | 27.180 ms mean = 36.8 tok/s | **27.167 ms mean = 36.8 tok/s** |
| repeated after the two hardening items, 2 runs each | 1024 96.3 / 512 280.0 / 16k 625.3, 628.1 tok/s, decode 27.48, 27.53 ms | **1024 94.9 / 512 290.6 / 16k 625.3, 623.5 tok/s, decode 27.52, 27.53 ms** |

All twelve parity runs above printed `117dd8d9d8dc…` (1024 rows, 1,021,091,840 B) or `8387234709271515…` (512 rows), and all sixteen `decode run … 128` runs printed the same 128-id trace. The 1024 form in the WARM state reads 748.2 tok/s on `0c9feb5` and 578.0 / 748.8 / 192.2 tok/s on HEAD across three attempts — the spread is the point: how much of the `ple` section survives the next load's 43.51 GiB pin is not reproducible, so only the cold state is a measurement. `tools/gate-linux.sh` at `c1a68cd`: ALL GREEN (`bceba6ff7724` / `838723470927` / `3bb3e69edf90` / the 32 ids / 144 tests / 1422 clippy / both doc guards).

The refactor series, all on 2026-09-17: clippy 1494 → 1480 → 1426 → **1422** warnings (the `--all-targets` form), `engine/src` 24,707 → 24,641 → 24,595 lines with cut 3 at +7 host lines net, build warnings 16 → 1, `cargo test --release` 142 → **144** passed 0 failed (84 lib + 60 serve).

The CUDA Rust evaluation (`docs/cuda-rust-evaluation.md`, `0667e0b`): `gelu_tanh` ported to cuTile 0.3.1 and run against the production NVRTC kernel over 1,048,576 f32 with the sixteen edge patterns — **1,047,516 values bit-equal (99.899 %)**, 1,060 differ, max 33,135 ulp in the cancelling negative tail at max abs 2.38e-7, and against an f64 reference neither form is more accurate (16,570 against 16,565 ulp): the two `tanh` implementations simply round differently, so only maps of exact IEEE operations could move with an identity claim. Steady-state cost is the same (2.76 µs per launch either way), but the cuTile first launch costs 82.4 ms through the `tileiras` subprocess and the feature-on build costs +63 % wall and +71 crates. Recommendation: port nothing now.


The gate after that battery, one run per commit, every value in the commit it landed in: `tools/gate-linux.sh` ALL GREEN at `f8f75c0` (twice), `1032bc5` (twice), `4004e66` (twice), `e2b9845`, `0cf1de5`/`c1a68cd` and `8ff2055` — 8 rows `bceba6ff7724…`, 512 rows `8387234709271515…`, P8 teacher-forced `3bb3e69edf90…`, the 32 ids of record, `check_env_docs` exit 0 and `check_readme_dates` 0 offenders on every one. `cargo test --release` 144 → 147 (the three `cnq::tests::page_runs_*` of TASK H) → 153 (the six of TASK J) → **165 passed, 0 failed** (98 lib + 67 serve, the twelve of TASK K), clippy unchanged at **1,422** throughout. The `serve` smoke is byte-identical across the whole series with the slot file at `2b7d129afdc4…`, 131,820,320 B, including across the vit reserve, which moved N 158 → 155 and not one byte of the smoke. The 1024-row parity form holds `117dd8d9d8dc…` and 1,021,091,840 B on every run of every build, at 739.6 / 740.6 (`1032bc5`), 740.2 / 742.8 (`4004e66`) and 740.9 tok/s (`8ff2055`).

The Linux values of record at `487128d`, the contract for every later commit: parity 8 `bceba6ff7724…`, parity 512 `8387234709271515…`, P8 teacher-forced `3bb3e69edf90…`, parity 1024 `117dd8d9d8dc…` at 740 tok/s; `decode run` on the 16,064-id t1-read prompt 968 / 964 tok/s prefill and 27.19 / 27.22 ms = 36.8 / 36.7 tok/s decode at context 16,192; the six-turn `serve` replay 228.3 ms of prefill and 247.4 ms to first token, mean over the warm turns.

### Known limitations

- **The ten-task gate has no Linux value of record.** Today's run is the first one, and `final4` cannot be reproduced at ids level on this toolchain for long generations. The closer is one `9f12429` ten-task run after a reboot (`decode_out/final/GATES.md` section 6). It could not be run on the day: after any engine exit about 45 GiB stays in the driver's pinned pool, invisible to `MemAvailable`, so the pre-fix build's own RAM gate refused every start but one — which is the `#15` failure reproduced on the pre-fix build, and the reason the reference serve smoke is also missing. The throughput arm was closed on 2026-09-17 against `0c9feb5` instead, the last commit before the three cuts and the first one that boots twice (Fixed, above); a `9f12429` arm still needs a reboot.
- **The Windows and Linux values of record differ on the 512-row and 1024-row forms** and will keep differing until the toolchains match: Windows NVRTC 13.3.73 with driver 616.56 against Linux NVRTC 13.3.33 with driver 610.57. The 8-row form is identical on both. This is a property of the JIT, not of the engine (`9f12429` proved it over four configurations), and `docs/architecture.md` section 8.7 holds the four values.
- **A warm turn's prefill is a PCIe transfer, and what remains of it is physics at this hot-set size** (TASK I, 2026-09-17). After the three changes above, three quarters of a warm turn is still the staging of the cold experts the chunk routes to: 9,431 MB on turn 1 of the replay and 192.8 ms of that chunk's 256.4 ms in the instrumented arm, at 48-49 GB/s against the 55 GB/s of `pcie_probe` variant f. Nothing in that loop is compute-bound, and the 55 ms of pre-MoE compute per turn is time the copy engine cannot use, because the routing of layer `l` is not known until layer `l` has run. The levers left are all "fewer bytes": a hot set that follows the session further (`CROW_ADAPT_EVERY=4` measured 209.3 ms of prefill and is not taken — see the Fixed item), more VRAM for the hot set (the planner already maximizes N against a 262k-context KV budget: N=155 slots, 148 logical with the trickle's seven spares), or a cold tier smaller per expert, which is not bit-identical and therefore not this. `#10` (prefill against its target) is the issue this belongs to.
- **`CROW_KPROF` can only profile a `CROW_GRAPH=0` run**, because no sync may run inside an open graph capture, so the per-kernel table never covers the captured decode path as `serve` replays it. The pair is refused at load since 2026-09-17; the decode-path numbers in this file come from the `CROW_PROFILE` section table, which does not sync.
- **Clippy is still red while non-blocking**: 1,422 warnings at HEAD, down from 1,494 at the start of the branch and measured in the `--all-targets` form the series was counted with.
- **No run of the CI workflow is recorded in this repository**: the four jobs moved to ubuntu-latest with the port, and the counts quoted in `README.md` are still the local Windows proof of 2026-09-11.
- The hardware probes of 2026-09-01/02 were never re-run on Linux, and `docs/measurement-handoff.md` still owes its Linux retest of the job-ring round trip; its numbers are WDDM numbers.
- **The btrfs compression finding was real about the filesystem and empty about the number.** The container sat on `compress=zstd:3` with 27,665 of 31,146 extents compressed, and `chattr +m` plus `btrfs filesystem defragment` is a NO-OP on an already compressed file (27,659 extents still encoded); only a full rewrite cleared it (0 encoded extents, 2,817 extents, sha256 unchanged). On the 1024-token cold form with the `f8f75c0` binary that rewrite changed nothing — 95.7 / 93.4 tok/s compressed against 92 tok/s uncompressed — because the mapping fault dominated both. Keeping the container off a compressed mount is a rule for the FIXED engine, where the device read is on the critical path, and it is not backed by a before/after of that engine on a compressed container.
- **`serve` has no reasoning filter, and one stray `</think>` poisons the client's history** (`#67`, opened 2026-09-17 against `487128d`). `serve` renders the chat template with `enable_thinking false` and streams whatever the model emits as `content`; llama.cpp's chat parser strips `<think>…</think>` and a bare `</think>` before the client sees it, ours does not. Crow stores the turn verbatim and re-sends the whole history, so one tag is re-fed every turn; and the model's own template cuts prior assistant turns at `content.split('</think>')[-1]`, so a stored turn that ENDS with the tag renders as an EMPTY assistant turn on the next request. Seen from about 118k of 200k context on, after a large code paste. **FIXED 2026-09-18, and the template claim in this item is REFUTED**: this model's template renders a stored `content` verbatim inside its own think block instead of splitting on the tag, so the turn is not emptied - it carries two closing tags, which is what the model imitates. See the v0.3.1 section above.
- **A long goal-mode session at 170k+ context degenerates** (`#68`, opened 2026-09-17 against `487128d`). One 40-minute goal: 574 messages, 273 assistant turns, 186 tool calls, the last prompt 178,779 of 200,000 tokens. Three stages: the `</think>` tag on 67 of the 273 turns (`#67`), then the model echoing the client's goal nudge verbatim (105 of the 114 user turns ARE that nudge; 9 assistant turns repeat its text at 65 to 76 tokens each), then a single repeated digit token. The engine errored on none of it — 318 completions, all 200 after the `487128d` fix, 17 images in the last request with 16 cache hits — and at 178k context the `[chat]` lines read prefill of 114 to 333 new tokens at 138 to 266 tok/s, reset 12 to 14 ms, decode 24.0 to 28.5 tok/s. What is the engine's, what is Crow's and what is the model's is to be measured, not assumed; the artefacts are secured under `decode_out/sessions/2026-09-17-goalmode/` (`serve.log`, 7,976 lines; `crow-session/session.json`, 574 messages; `crow-log/`; `booted.json`). Not started.
- Everything open before this release stays open: prefill against its target (`#10`), the run-position drift of a `serve` rate (`#38`), engine logging (`#13`), Ampere and Ada (`#12`), and the ten-task quality gate itself (`#11`, `#44`).

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

- `#15` the Linux host-memory model, 2026-09-17: the engine boots reliably on a 62.17 GiB Arch box (RTX 5090, systemd-oomd, zram, `vm.swappiness=150`) without refusing itself and without taking the desktop with it. Nothing in the numeric contract moves: `KERNEL_SRC`, launch geometry, k-order and reduce order are untouched, and every gate below is byte-identical to the Linux values of record of `9f12429`.
  - **the pinned budget is derived, not assumed** (`manager::derive_host_pinned_budget`, called from `Engine::load`): `min(46 GiB cap, free_for_pin - CROW_RAM_MARGIN_GB)`. The 46 GiB of `geo.rs:110` is now the CAP, the measured ceiling of the 64 GB machine the default was written on. A smaller host is not a refusal any more - it is an input to the two-sided planner loop, which raises N (more experts hot in VRAM, fewer pinned) and refuses only when no N satisfies both sides (`planner_refusal_msg`, test kept). `CROW_PINNED_BUDGET_GB` holds the budget fixed for a measurement. One `[budget]` boot line names the value, its basis, free for pinning and `MemAvailable`. The Windows boot of 2026-09-14 died in the gate this replaces (`decode_out/hotfix-serve.log`: "refusing to pin 44.62 GiB with only 46.47 GiB physical RAM free").
  - **`free_physical_ram` stops reading `MemAvailable` on Linux** (`cuda::free_physical_ram_parts`): it now counts what cannot be RECLAIMED and subtracts it from `MemTotal` - `AnonPages + Shmem + SUnreclaim + KernelStack + PageTables + Percpu`; `Unevictable`/`Mlocked` are deliberately not subtracted (they are already inside `AnonPages`/`Shmem`). The NVIDIA driver's pinned-page pool sits in no /proc/meminfo class after a process exits, so `MemAvailable` cannot see it, yet the next `cuMemHostAlloc` is served from it and it is given back under pressure. Measured 2026-09-17 with the pool present: free for pinning **60.78 GiB** against `MemAvailable` **12.44 GiB** (`MemTotal` 62.17, `AnonPages` 0.60, `Shmem` 0.17, `SUnreclaim` 0.57, `KernelStack`/`PageTables`/`Percpu` 0.05 GiB). Before this, the gate refused every second start. The Windows branch is unchanged.
  - **the load leaves no page-cache trail** (`Cnq::fadvise_consumed`, `residency::Residency::build`): every consumed container range is dropped with `posix_fadvise(POSIX_FADV_DONTNEED)`, coalesced into 64 MiB batches while the reads stay contiguous, with the `ple` section exempt (it is designed to live in page cache). The cold tier and the hot slabs are now filled by ONE ascending sweep per expert tensor instead of two passes over disjoint id sets - same bytes into the same destinations, but the tensor is read whole and in order, so the kernel's readahead is consumed instead of stranded over the 1.76 MB gaps the old cold pass skipped. Measured on the 8-row form, same binary, `CROW_CNQ_PURGE` as the A/B switch: drop OFF peaks at **12.67 GiB** page cache with `MemFree` down to **1.18 GiB**, drop ON at **5.14 GiB** with `MemFree` never below **7.70 GiB**, both `bceba6ff7724`. Against the pre-fix trace of `9f12429` (`decode_out/linux-port/rss512.log`): max `Cached` **33.04 -> 5.14 GiB**, min `MemFree` **1.15 -> 7.70 GiB**. The 8-row load also fell from 44 s to 25 s.
  - **`slot::restore` streams**: the payload is checked against the size on disk before the first upload (the whole length check, earlier than the `read_to_end` bound it replaces) and then read into the destinations through one reusable buffer sized by the largest consumer, about 51 MB. A full-context restore was a single **2.70 GiB** anonymous allocation on top of the 47.7 GiB steady state, the worst instantaneous host-RAM event in the engine.
  - **`Vit::image_cache` is bounded**: LRU with a byte ceiling, `CROW_VIT_CACHE_MB`, default 256 MiB. It was per-process, unbounded, never evicted, up to 10 MiB per distinct image - about 1 GiB per 100 screenshots in a long serve session.
  - **`vit::prep_image` builds one channel plane at a time** instead of all three: the f32 copy of the decoded raster falls from `12 * h0 * w0` to `4 * h0 * w0` bytes (up to about 2 GiB from a single 16 MiB PNG, the decoder's own cap being 512 MiB). Identical bytes by construction - `plane` carries exactly the slice `planes[c*h0*w0..]` carried - and checked: a throwaway test hashed `prep_image`'s `patches` and `cs` for a 97x61 PNG on both forms and read the same values (`3068936281624539976` / `17244354145405810823`, 308 patches, grid (1, 14, 22)). The `planes`-after-downscale variant of the research note is not possible: the downscale IS the resize, and its source is the original raster. Image smoke through the launcher: one 96x96 PNG, MISS runs the tower in 187.0 ms, the re-send is a 0.4 ms HIT, `[vit-cache]` reports 1 entry / 0.6 MiB of 256 MiB.
  - **`tools/serve-linux.sh`**: starts `serve` in a transient scope (`systemd-run --user --scope --slice=session.slice -p MemorySwapMax=0 -p MemoryHigh=<MemTotal-8G> -p MemoryMax=<MemTotal-6G>`) with the limits computed from `/proc/meminfo`, `CUDA_LIB` for `LD_LIBRARY_PATH`, and every `CROW_*` and argument passed through. Documented in `README.md` and referenced from the host-memory rows of `docs/env.md`.
  - gates (RTX 5090, Arch Linux, all runs inside the memory-bounded scope, 2026-09-17): `cargo test --release` **142 passed 0 failed**; parity 8 rows `bceba6ff772431dedf57631e83c7418d3f0bca3ab7e33da828f8da5d12a122a2` at 11,919,360 B, three runs incl. the `CROW_CNQ_PURGE=0` arm; parity 512 rows `8387234709271515b091b1c4dbd0d59c66550d0e3feab551a6418d30b55c9105` at 512,532,480 B; **two 8-row runs back to back with no balloon and no manual reclaim, both booted** (the second-start failure is closed); `serve` through the launcher from a host holding 12.66 GiB of page cache - ready in 26 s, `N 151` logical of 158 slots, pinned cold tier **44.62 GiB** under the 46.00 GiB derived budget, `VmHWM` **46.23 GiB**, `MemFree` never below 2.67 GiB, page cache FALLING 12.66 -> 5.36 GiB across the load, `/props` and `/health` 200, one `max_tokens` 8 completion answered "Ready" (`finish_reason` stop).
  - known limitation: `free_for_pin` counts the driver's pinned-page pool as available because it is reclaimable, so it also counts a pool another process is actively using. The scope of `tools/serve-linux.sh` is what bounds that case; the derivation does not read the cgroup's own `memory.max`.

- `#62` 62e, 2026-09-13: the GDN big-slab GEMV geometry lands, 62a lever 2 (the resumed #62d lever) - the three big GDN decode projections run 32 rows per block instead of 64
  - kernels: `gemv_fp4_mma_d32` (`kernels.rs:649`) and `gemv_fp4_mma_g32` (`kernels.rs:723`), the 64-row forms verbatim with only the warp map moved (rg = warp & 1, ks = warp >> 1, block 64 x KS threads = `mma_bx32()`, `gen.rs:1265`); `CROW_MMA_KS` stays 4 and the k split with its fixed smem reduce order is untouched, so per-row arithmetic is bit-identical by construction; registered in the `Kernels::new` names list
  - launch sites (`engine/src/gen.rs`): the grouped input launch (default since #19g) goes `gemv_fp4_mma_g` 258 -> `gemv_fp4_mma_g32` 515 blocks (`gen.rs:1945`), the per-slab fallback (`CROW_GDN_FUSE_IN=0`) goes qkv 160 -> 320 and z 96 -> 192 on `gemv_fp4_mma_d32` (`gen.rs:1952`, `:1955`), and the out projection in gdn_step goes 40 -> 80 (`gen.rs:1995`); the b and a launches, gdn_prompt, every non-GDN `gemv_fp4_mma_d` user, `gemm_fp4_dense` and the fallback branches are untouched; NO new env variable
  - verification (`decode_out/srv-62e.log`, RTX 5090, 2026-09-13, PROTOCOL v2: smoke first, VRAM headroom gate before the PX form, expensive forms early, phase resume): unit lib 81/81, serve 57/57, `qsa_probe` 4928 rows / 0 differences; parity 9 of 9 forms byte-identical to the reference `d211ab52ad2b` at the 61b sha256 values of record (8 rows `bceba6ff7724`, 512 `14c8628acbec` x3 + cross, 1024 `b2e87b2bf99a`, P8/P8FUSE `b7f6419203b4`, PFB0 `bceba6ff7724`, PXFUSE 16,056 teacher-forced rows `f217e1c55926` at 15,960,023,040 bytes under the > 26 GiB headroom gate - the parked 62d attempt's VRAM OOM did not reproduce); ten tasks 10 of 10 equal the standing splits-8 record
  - measured (t1-read 16,064 ids, 255 timed decode steps, W + 3 adjacent pairs): the 62e binary reads 22.1761 ms per token = 45.1 tok/s (N1 22.1327 / N2 22.1627 / N3 22.2330) against the 19i-state B arm's 22.6231 = 44.2, -0.4470 ms per token = -1.98 percent, 3 of 3 pairs favour N, ids `5098f885ab3a` in 7 of 7 runs incl. W; THE DECODE LINE IS CROSSED: 22.1761 < 22.27, the llama.cpp row of record, the first engine default under it
  - B-arm note of record: the pairs B arm is a same-source `be7bb60` rebuild `a7d850803e627432f00083e4dee6c461429ed9bf` (the 19i binary of record `5c7919276203` is not byte-reproducible on the day's toolchain; the source state is pinned by `decode_out/19i-out/head.txt`)
  - bookkeeping: the lever halves rode parked in `decode_out/62d-wip.patch` since the `52a922e` incident (repaired by `be7bb60`); the 62e letter keeps the resumed attempt's log/records unambiguous against the parked 62d attempt (`decode_out/srv-62d.log`)
- `#19` 19i, 2026-09-13: the #19h shared-expert fusion becomes the decode DEFAULT — unset or any value but `0` runs cascade ON + hc chain FUSED + shared chain FUSED, the completed launch-fusion default set
  - switch: `sh_fuse_on()` (`gen.rs:1327`) converts from the 19h opt-in `== Ok("1")` to the house `!= Ok("0")` pattern (`1` stays accepted and now redundant; `0` = all fallbacks of record); the `[hc]` boot line names the new default and the `0` fallback (`gen.rs:759`); `engine/src/gen.rs` only, kernels and launch sites untouched
  - incident of record: the first 19i commit `52a922e` accidentally swept the parked #62d gen.rs geometry hunks without their kernels.rs half — every engine start panicked (`kernel missing gemv_fp4_mma_d32`); repaired by `be7bb60` (gen.rs back to the 19h state plus ONLY the 19i flip hunks, the 62d halves re-parked in `decode_out/62d-wip.patch`); ALL 19i gates ran on the repair build `5c7919276203`
  - verification (`decode_out/srv-19i.log`, RTX 5090, 2026-09-13): smoke run exit 0 with the fused boot lines; the trimmed 19g battery (no PXBOTH) with NO env — 8 rows `bceba6ff7724`, 512 `14c8628acbec` x3, 1024 `b2e87b2bf99a`, P8 `b7f6419203b4` — 20 of 20 subchecks GREEN, every no-env form byte-identical to the reference `d211ab52ad2b` at the 61b sha256 values of record, plus the `CROW_QFUSE=0` fallback 8 rows at the absolute `bceba6ff7724` with the unfused/`0` boot lines (the 19g INFO basis: the cascade-off dump equals the cascade-on dump); ten tasks 10 of 10 equal the standing splits-8 record (t19h = t19g, cross-checked in-log); unit lib 80/80, serve 57/57, `qsa_probe` 4928 rows / 0 differences
  - measured (the coordinator-approved W + 3N form — NO adjacent B arm post-flip: `CROW_QFUSE=0` also disables the NVFP4 cascade, the 19g artifact; binary archaeology skipped by ruling): t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, ids `5098f885ab3a` in 4 of 4 runs incl. W; the all-fused default reads 22.4266 ms per token = 44.6 tok/s (N1 22.4336 / N2 22.4179 / N3 22.4284, spread 0.0157 ms), hard gate < 22.6999 MET, against the 19g combined default's 22.7999 = -0.3733 ms = -1.64 percent and the 19h N arm 22.5584 = -0.1318 ms = -0.58 percent; the lever's own adjacent-pair reading (-0.2286 ms = -1.00 percent, 3 of 3 pairs) stands from the 19h chain
- `#19` 19h, 2026-09-13: the shared-expert MoE decode chain fuses behind the SAME `CROW_QFUSE` overload, EXACT `1` only (unset keeps the post-19g default of record: fused hc chain + unfused shared chain; `0` stays all off)
  - `sh_gate_up_q` (`kernels.rs:3047`): ONE launch replaces the two shared gate|up `gemv_fp4_mma_d` GEMVs plus `silu_mul640_q`, the mma_d body twice verbatim (same lane split, same k walk, same ks-split reduce) over the gate and up slabs against the same `xq_gu` row, the silu_mul640_q math warp-wide on the finished accumulator pairs (the standalone 16-consecutive-j warp layout kept, so the quant16_store shuffle groups are unchanged), writes `sh2` + `xq_s`; `gemv_fp4_mma_dg` (`kernels.rs:3143`): the down GEMV gains the `gate_shared` epilogue at the store (`moe_out = sigmoid(sgv) * down`, ASSIGN, still the first writer, no memset); the `sgv` `gemv_b` hoists before the down launch; 6 -> 3 launches per layer x 48 = 144 launches removed, decode `t < 8` + the mma/dense path only, prefill and the `gemv_fp4_bs` fallback verbatim; `sh_fuse_on` (`gen.rs:1327`) next to `hc_fuse_on`, the `[hc]` boot line names the shared state
  - bit identity: by construction (epilogue folding only) and measured (`decode_out/srv-19h.log`): parity 12 of 12 GREEN (OFF 8 rows byte-identical to the 19g-committed binary `42c6419f2cd6` with sha `bceba6ff7724`, ON 8 rows byte-identical to OFF, the P8FUSE form no-env vs `CROW_QFUSE=1` byte-identical with BOTH dumps at the teacher-forced sha256 of record `b7f6419203b4`), ten tasks 10 of 10 equal the standing splits-8 records (t19g = t19f, cross-checked)
  - tests on the 19h build, 2026-09-13: lib 80/80, serve 57/57, `qsa_probe` 4928 rows / 0 differences
  - measured 2026-09-13, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs, B = no env (fused hc + unfused shared) / N = `CROW_QFUSE=1` (all fused): crow 22.8290 / 22.7769 / 22.7551 against 22.5472 / 22.5571 / 22.5708 ms per token, means 22.7870 against 22.5584 = 43.88 against 44.33 tok/s, -0.2286 ms per token = -1.00 percent at 3.1x the B spread window, 3 of 3 pairs favour the fused path, ids `5098f885ab3a` in 7 of 7 runs of both arms (`decode_out/srv-19h.log`); the delta lands inside the #19c-pre rank-2 estimate band of 0.20 to 0.30 ms
- `#19` 19g, 2026-09-13 (with `#62`): BOTH proven bit-identical levers become the decode default in one gated flip — the hc-chain fusion and the grouped GDN input projections
  - switches: `hc_fuse_on()` and `gdn_fuse_in_on()` convert from the opt-in `== Ok("1")` to the house `!= Ok("0")` pattern — unset or any value but `0` runs the fused/grouped form; `0` selects the fallback of record (the unfused 8-launch hc chain / the four per-slab `gemv_fp4_mma_d` launches). `CROW_QFUSE` semantics of record: unset now means NVFP4 cascade ON + hc chain FUSED, `0` turns both off; the documented overload remains debt and since 19g cascade-on + unfused is no longer reachable (a rename to distinct switches costs one rebuild plus the gate forms); both doc comments and the `[hc]`/`[gdn]` boot lines name the new default and the `0` fallback; launch sites and kernels untouched, `engine/src/gen.rs` only
  - COMBINED parity (the 19f C5 concern discharged — the two levers had never run together; every no-env form exercises both): 16 of 16 subchecks GREEN against `d211ab52ad2b` with the 61b sha256 values of record — 8 rows `bceba6ff7724`, 512 `14c8628acbec` x3, 1024 `b2e87b2bf99a`, P8 `b7f6419203b4`, PXBOTH (16,056 teacher-forced rows, 16 GB per side) `f217e1c55926bca26ae...`, and the double-fallback `CROW_QFUSE=0 CROW_GDN_FUSE_IN=0` 8 rows `bceba6ff7724` (the old default path intact)
  - identity: ten-task with no env 10 of 10 equal BOTH the t63c and t61b splits-8 records including the EOS stops; pairs ids `5098f885ab3a` 7 of 7; boot lines assert the arm's form per run; unit lib 80/80, serve 57/57, `qsa_probe` 4928 rows / 0 differences (`decode_out/srv-19g.log`, RTX 5090, 2026-09-13)
  - measured, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run: the new combined default reads 22.7999 ms per token = 43.9 tok/s against the previous default of record 23.94 ms = 41.8 tok/s (the 61b row), an honest gain of -1.14 ms per token = -4.8 percent, ids identical; the pair-chain basis is the 19f pairs (-0.9806 ms per token) plus the 62b clean pairs (-0.33 ms); the same-chain double-fallback arm measured 26.8559 ms per token = 37.2 tok/s, but `CROW_QFUSE=0` also disables the NVFP4 cascade, so that figure is the overload artifact, not the previous default (named incident, `.superpowers/sdd/task-19g-report.md` C2); operating point 22.80 ms per token against llama.cpp 22.27 ms per token = 44.9 tok/s, gap ~0.53 ms
- `#19` 19f, 2026-09-13: the hc_run hyper-connection decode chain fuses opt-in behind `CROW_QFUSE=1`
  - launch shape: ONE hc block goes from 8 launches to 4 — `hc_down_inj` (`kernels.rs:871`) carries the down GEMV + `silu_div4` + the 4-row inject `gemv_fp4_b1k` + `sig2_div4` in one launch (the b1k 1024-slot reduce emulated bit for bit on 256 threads), `gemv_bf16_ws` (`kernels.rs:947`) stores the `sigmoid_el` epilogue with the up row; `head_run` fuses the same epilogues; the elementwise row of the #19c inventory (`silu_div4` 97 + `sigmoid_el` 97 + `gemv_fp4_b1k` 96 + `sig2_div4` 96 launches per token) folds away, decode `t < 8` only
  - switch: the exact value `1` of the EXISTING `CROW_QFUSE` additionally opts into the fusion (value table in `docs/env.md`: unset = cascade on + fusion off, `1` = both on, `0` = both off); unset keeps the of-record unfused chain; one `[hc]` boot line names the form (`gen.rs:758`); documented debt, a rename to a distinct switch would cost one rebuild plus the gate forms
  - bit identity: by construction (epilogues elementwise on the finished accumulator, merged grid keeps every row's dot product) and measured (`decode_out/srv-19f.log`): switch-OFF parity 12 of 12 subchecks GREEN against `d211ab52ad2b` reproducing the 61b sha256 values of record (8 rows `bceba6ff7724`, 512 rows `14c8628acbec`, 1024 rows `b2e87b2bf99a`, P8 teacher-forced `b7f6419203b4`) — per orchestrator directive this doubles as the #61e NO-env revert proof; switch-ON P8FUSE and PXFUSE (16,056 teacher-forced rows) byte-identical with `b7f6419203b4` / `f217e1c55926`; ten-task switch-off 10 of 10 equal to the t63c splits-8 record
  - tests on the 19f build, 2026-09-13: lib 80/80, serve 57/57, `qsa_probe` 4928 rows / 0 differences
  - measured 2026-09-13, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs, B = switch off (no env) / N = `CROW_QFUSE=1`: crow 24.3094 / 24.3840 / 24.3598 ms per token against 23.3383 / 23.4522 / 23.3210 fused, means 24.3511 against 23.3705 ms per token = 41.07 against 42.79 tok/s, -0.9806 ms per token = -4.03 percent at 13.1x the B spread window, 3 of 3 pairs favour the fused path, ids `5098f885ab3a` in 7 of 7 runs of both arms, cold experts 122.2 of 480 in every run (`decode_out/srv-19f.log`); the delta lands inside the #19c rank-1 estimate of 0.8 to 1.2 ms
- `#61` 61e, 2026-09-13: the split-count default rolls back to 8, the #61d flip is withdrawn
  - selection: `CROW_ATTN_SPLITS` unset or any value but `4`, `8`, `16`, `32` runs the `ATTN_SPLITS` const, back to 8; `CROW_ATTN_SPLITS=32` reproduces the 61d experiment (`gen.rs:1302`, env read `gen.rs:1323-1327`, boot line `gen.rs:741`, engine commit `fdc00c4`)
  - reason of record: the #61d ten-task quality verdict - 0 Pass / 7 Partial / 3 Fail at 32 against the crow record 2 / 5 / 3 at 8 and the llama reference 2 / 6 / 2 (`.superpowers/sdd/task-61d-quality-report.md`); the improvement-loop quality rule outranks the performance-over-ids ruling of 2026-09-12, which stays recorded
  - the 61d measurement stays the measurement of record for the knob: 22.8545 against 24.1325 ms per token = 43.8 against 41.4 tok/s = -1.278 ms = -5.3 percent at 32.3 x the fallback spread, B ids `5098f885ab3a` 3 of 3, N ids `c65969f7793a` 3 of 3 (`decode_out/srv-61d.log`); `16` and `32` stay measurement only, the splits-8 stream of record is again final4-identical
  - boot line: every engine process prints one `[attn]` line naming the split count it runs and the restored default ("default 8, restored by 61e"), with 32 marked as the rolled-back experiment
  - tests on the 61e build, 2026-09-13: lib 80/80, serve 57/57, `qsa_probe` 4928 rows / 0 differences (`decode_out/srv-61e.log`); the NO-env revert proof (parity forms reproducing the 61b sha256 values of record) rides the next engine chain per orchestrator directive
- `#61` 61d, 2026-09-13: the decode attention runs 32 splits by default (ROLLED BACK by #61e the same day, see above; the measured numbers stand)
  - selection: `CROW_ATTN_SPLITS` unset or any value but `4`, `8`, `16` runs the `ATTN_SPLITS` const, now 32; `CROW_ATTN_SPLITS=8` restores the configuration of record before the flip (`gen.rs:1280`, read site `gen.rs:1301`, boot line `gen.rs:742`)
  - boot line: every engine process prints one `[attn]` line naming the split count it runs, next to the `[qsa]` line
  - measured 2026-09-13, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs: crow-nest 23.23 / 22.63 / 22.71 ms per token with the new default against 24.11 / 24.14 / 24.15 with `CROW_ATTN_SPLITS=8`, means 22.8545 against 24.1325 ms per token = 43.8 against 41.4 tok/s, 1.278 ms per token gained = 5.3 percent at 32.3 fallback spread windows, B ids `5098f885ab3a` 3 of 3, N ids `c65969f7793a` 3 of 3, the 61a/61c S32 value (`decode_out/srv-61d.log`)
  - llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL on the same machine, 2026-09-11: 22.27 ms per token = 44.9 tok/s (`decode_out/srv-59b.log`); the engine gap fell from 1.67 to 0.58 ms per token
  - gates: parity 8 of 8 FORCED-8 forms byte-identical against the installed build `d211ab52ad2b` with the 61b sha256 values of record, so the flip is the const plus the boot line and nothing else; the new-default splits 32 stream is recorded fresh, PX long form sha256 of record `43de2126b85c03f21565a619425b14dec99dbebde70856ac3172a5c3a30f62bb` (pxn1 and pxn2 byte-identical), the ids change of record against the splits 8 `f217e1c55926`; ten-task 10 of 10 recorded with per-task shas as the new baseline of record under the documented ids-change rule (0 of 10 still equal `final4`); A9B GATE PASS with in-chain new-default reference shas; A10 smoke 6 of 6
  - quality verdict of record: the ten-task quality bar is NOT held at splits 32, judged 0 Pass / 7 Partial / 3 Fail against the crow record 2 / 5 / 3 at splits 8 and the llama reference 2 / 6 / 2, no degeneration mode, one greedy sample per task (`.superpowers/sdd/task-61d-quality-report.md`); the flip rests on robin's performance-over-ids ruling of 2026-09-12, `CROW_ATTN_SPLITS=8` restores the record stream
  - the prefill split-count form is untouched: a different split count lives in the decode attention only
- `#61` 61b, 2026-09-12: the decode QSA top-k runs as `qsa_select_par` by default
  - selection: `CROW_QSA_PAR` unset or any value but `0` runs `qsa_select_par_h` over `CROW_QSA_PAR_BLOCKS` = 32 blocks plus one `qsa_select_par_e` emit block, same list in the same order; `CROW_QSA_PAR=0` restores the `qsa_select_fast` single-block fallback (`gen.rs:1302`, boot line `gen.rs:731`)
  - boot line: every engine process prints one `[qsa]` line naming the selection it runs, next to the `[stage]` and `[trickle]` lines
  - measured 2026-09-12, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs: crow-nest 23.94 / 23.93 / 23.95 ms per token with the new default against 24.84 / 24.89 / 24.88 with `CROW_QSA_PAR=0`, means 23.94 against 24.87 ms per token = 41.8 against 40.2 tok/s, 0.93 ms per token gained at 19.2 baseline spread windows, ids `5098f885ab3a` in 6 of 6 counted runs (`decode_out/srv-61b.log`)
  - llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL on the same machine, 2026-09-11: 22.27 ms per token = 44.9 tok/s (`decode_out/srv-59b.log`)
  - gates: parity 8 of 8 forms byte-identical against the installed build `d211ab52ad2b`, including the new PX form, teacher forced over the 16,064 id t1-read prompt so the radix path runs at logit level, sha256 of record `f217e1c55926bca26ae32867a62eebf3ab796e05f689469b70032ee1e1acd695`; ten-task greedy ids equal `final4` 10 of 10; A9 parts 1 to 3 PASS with reference shas 10 of 10; A10 smoke 6 of 6
  - review fixes in the same engine commit: the emit block's 1024 thread contract is a named constant asserted in the only launcher, the h1 pairing invariant is written at the launch site, `qsa_probe` covers 4928 rows with the production cap 2051
  - `CROW_ATTN_SPLITS` stays 8 and measurement only: 16 and 32 change the generated ids, robin decides; `attn_sel_split` at 2.44 ms per token stays the open row of #61
- `#63` 63c, 2026-09-12: the stream trickle's copies are issued AFTER the graph launch by default
  - selection: `CROW_TRICKLE_DEFER` unset or any value but `0` parks the side-stream copies in `trickle_tick` and issues them in `decode_step` right after the launch (`gen.rs:1223`, drain site `gen.rs:3145`, `Engine::trickle_drain_after_launch` at `gen.rs:3449`)
  - fallback: `CROW_TRICKLE_DEFER=0` restores the previous order, the issue inside `trickle_tick` before the launch (`gen.rs:3412-3422`)
  - boot line: every engine process prints one `[trickle]` line naming the order it runs (`gen.rs:721`), next to the `[stage]` line
  - scope: the copies move in host issue order only; the table flip and `event_record(ev_commit)` stay before the launch, so both settings are byte-identical in parity
  - measured 2026-09-12, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs: crow-nest 24.78 / 24.79 / 24.78 ms per token with the new default against 26.35 / 26.45 / 26.43 with `CROW_TRICKLE_DEFER=0`, means 24.78 against 26.41 ms per token = 40.35 against 37.87 tok/s, 1.63 ms per token gained at 16.0 baseline spread windows (`decode_out/srv-63c.log`)
  - llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL on the same machine, 2026-09-11: 22.27 ms per token = 44.9 tok/s (`decode_out/srv-59b.log`)
  - nsys class table of the deferred order, 2026-09-12: 83.96 % of the copy ms falls inside a graph span, exposed copy time 0.51 ms per token (`decode_out/srv-63b-crow-ND.sqlite`)
  - nsys class table of the eager order, 2026-09-11: 0.00 % inside a graph span, exposed copy time 2.60 ms per token (`decode_out/srv-19d-crow-NK.sqlite`)
  - gates: parity 7 of 7 forms byte-identical against the installed build `d211ab52ad2b`, ten-task greedy ids equal `final4` 10 of 10, A9 parts 1 to 3 PASS with reference shas 10 of 10, A10 smoke 6 of 6, ids sha `5098f885ab3a` in 7 of 7 pair runs

- `#19` 19e, 2026-09-12: `stage_cold_ca` is the default decode staging kernel
  - selection: `CROW_STAGE_KERNEL` unset or any value but `1` selects it (`gen.rs:542`), `CROW_STAGE_BLOCKS` default `40`
  - fallback: `CROW_STAGE_KERNEL=1` restores `stage_cold` and keeps `CROW_STAGE_SPLIT` meaningful, the new kernel ignores it
  - requirement: both staged slab byte counts are multiples of 4096, asserted at load (`gen.rs:683`) and at the launch site (`gen.rs:2048`); this container is 1,843,200 B = 450 tiles and 921,600 B = 225 tiles
  - boot line: every engine process prints one `[stage]` line naming the kernel it runs (`gen.rs:697`)
  - measured 2026-09-12, RTX 5090, t1-read 16,064 ids, 255 timed decode steps, one fresh process per run, three adjacent pairs: crow-nest 26.40 / 26.42 / 26.42 ms per token with the new default against 29.33 / 29.86 / 29.84 with `CROW_STAGE_KERNEL=1`, means 26.42 against 29.68 ms per token = 37.86 against 33.70 tok/s, 3.26 ms per token gained at 6.2 baseline spread windows (`decode_out/srv-19e.log`)
  - llama.cpp Qwen3.8-Flash-Next-UD-Q2_K_XL on the same machine, 2026-09-11: 22.27 ms per token = 44.9 tok/s (`decode_out/srv-59b.log`)
  - nsys staging row, 2026-09-11: 7.07 ms per token at 47.78 GB/s against 10.42 ms at 32.45 GB/s over 338 MB per token (`decode_out/srv-19d.log`)
  - gates: parity 7 of 7 forms identical against `d211ab52ad2b` (8 rows, 512 twice plus cross, 1024, the teacher-forced decode regime, 8 rows on the fallback), ten-task greedy ids equal to `final4` 10 of 10, A9 parts 1 to 3 PASS with reference shas 10 of 10, A10 6 of 6, the 255 generated ids identical in 7 of 7 pair runs (`decode_out/srv-19e.log`)
- `#34`, 2026-09-11: the `parity` harness sets `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` on the two Python oracle processes it spawns (`parity.rs:54-55` for `tools/tokenize_ids.py`, `parity.rs:77-78` for `tools/detokenize_ids.py`), so the id stream no longer depends on the shell that starts the harness. Before the fix a bare invocation tokenized t4-prose to 9,522 ids instead of 9,398 (measured 2026-09-09); 4 of 10 frozen prompts hold non-ASCII text.
- `#52`, 2026-09-11: the two generator bins default to the production container, `../converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` (`residency.rs:46`, `sf_scan.rs:9`), and the `residency` warm-up writes its sidecar to an output path of its own, `CROW_HOTSETS_OUT`, default `../decode_out/residency-warmup.hotsets.json` (`residency.rs:50`). The path is never `<container>.hotsets.json` any more, so the ragged sidecar of `#49` stays byte-unchanged.
- `#53`, 2026-09-11: both `operating_point` fields of a `parity` record are derived from `sample::Sampler::from_env` (`parity.rs:287`, record sites `parity.rs:370` and `parity.rs:458`). A greedy run keeps `200k floor, -np 1, greedy, temperature 0`; a sampled run carries `200k floor, -np 1, sample: temp <t> top_p <p> top_k <k> presence <pr> seed <s>`. Before the fix every record said greedy, also the six C2 sampling series.
- `#37`, 2026-09-11: `serve` ticks the stream trickle once per `decode_step`, the mirror of `bin/decode.rs:224-231`; `adapt_tick` stays uncalled, the trickle is drained after the last step, the request-local swap count goes to the `[chat]` stderr line as `crow_trickle_swaps`, and the wire `timings` block is unchanged. Measured in two six-run chains on the same day, serve and `decode run` D1 alternating: 23.13 to 25.57 tok/s before against 26.43 to 26.77 after, `decode run` 32.82 to 34.50 in both; the 255 generated ids are identical in all 12 runs.
- `#37` fix round 1, 2026-09-11: `serve` sets `CROW_ADAPT_WINDOW=1` when it is unset, in the same loop that already sets `CROW_GRAPH` and `CROW_MMA` (`bin/serve.rs:2313`), so the stream trickle ranks its swaps by the decayed selections since the last tick instead of the prefill-dominated cumulative count; `CROW_ADAPT_WINDOW=0` still restores the old ranking. Measured in a third six-run chain of the same form: `serve` 31.97 to 32.32 tok/s against `decode run` 32.71 to 33.09, three adjacent pairs at -3.39 %, -1.87 % and -2.03 %, all within 5 %; 4,595 trickle swaps in both arms; the 255 generated ids are identical in all 18 runs of the three chains.
- `#55` C2, 2026-09-11: the `serve` sampler default is confirmed as built (`#28`): a request without `temperature` is greedy, `temperature > 0` samples with the data-sheet defaults; the ten-task gate under sampling is met in 1 of 6 seeds (`#44`), under greedy 0 of 1 (`#11`); no code change.
- `#51` E8, 2026-09-11: `decode` and `parity` default to `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` and `decode_out/hotsets-M-longctx2100-n160.json`, the container and sidecar of `serve.rs:446-447`; five lines, `CROW_CNQ` and `CROW_HOTSETS` still override; closes `#48`.
- `#42` E2, 2026-09-10: the history was rewritten with `git filter-repo` to drop the probe debug blobs. Every commit sha changed: `release-v0.1` head `4519f6f` became `aa6dd04`, `main` head `d36353a` became `e9e53cc`. Tracked bytes went from 253,232,175 to 38,397,227, blobs over 50 MB 0 after (before: one, the 125.28 MB blob of `#41`), `Co-Authored-By` trailers from 19 to 0. A pre-rewrite mirror was kept outside the repository.
- `#43` E3, 2026-09-10: `converter/target` (103 files) and 216 of 235 `decode_out` records untracked; 19 gate inputs kept; tracked bytes 38,397,227 to 5,208,050.
- `#43` E3, 2026-09-11: `decode_out/README.md` added; tracked `decode_out` files 19 to 20. The ten `final4-*-run0-crow.json` records stay because they are the greedy reference answers the identity gate (B4/C1/E3/E8) compares against, not raw run output.
- `#45` E4, 2026-09-10: `.gitignore` ignores every build tree with `**/target*/` instead of listing them.
- `#36` M2b, 2026-09-10: one snapshot slot per process; the after-answer snapshot was dropped because its rows are decode rows, not prefill rows.
- `#VIT` 2026-09-14: the engine image path landed — the container's `vit` section (27 vision blocks x 12 tensors + patch embed + learned position table + merger; 112 NVFP4 + 221 bf16 keeps) loads beside the text sections with `CROW_VIT` unset (default ON; `0` = the text-only placeholder of record), serve answers Crow's image wire (`image_url` data-URL blocks, no resize client-side), and `/props` reports `modalities.vision` per the switch so Crow's `refuse_images` sends or refuses for real. New `engine/src/vit.rs` (weights, HF-fast-processor preprocessing with smart_resize factor 32 and antialiased bicubic, the f32 tower on the NVFP4 weights via the new 16-wide `gemv_fp4_vit` whose global sub-block addressing covers the k_dim-4304 rows, LayerNorm/rope/non-causal-attention/gelu kernels, the interleaved-mrope span tables and the host-side embedding splice at the expanded `<|image_pad|>` rows) plus 8 new kernels in `engine/src/kernels.rs`; the `image` crate rides as a decode-only dependency. Gates (`decode_out/srv-vit.log`, RTX 5090; reference `d211ab52ad2b` untouched): text parity with the tower LOADED (no env) AND with `CROW_VIT=0` byte-identical in BOTH arms at the 61b sha256 values of record — 8 `bceba6ff7724`, 512 `14c8628acbec` x2, 1024 `b2e87b2bf99a`, and the teacher-forced 16,064-row PX form `f217e1c55926` at 15,960,023,040 bytes under the > 26 GiB VRAM headroom gate — 23 of 23 subchecks GREEN with the `[vit]` boot line exactly once per process; ten tasks 10 of 10 identical to final4 with the tower loaded; ViT embeddings vs the f32 container-dequant oracle (identical weights both sides per the orchestrator ruling, so the band is math precision only) max_abs 3.43e-06 at cos 1.000000, with a stage-by-stage bisect (`oracle/ref_vit_stages.py`) that localized and fixed four tower bugs to exactness (the vision-rope w-axis frequency reuse, the attention V-walk that saw one key per thread per tile, two elementwise-count bugs, the fc2 k_dim); image-prompt oracle (trimmed prompt, T = 98): prompt-row argmax 88/98 with every deviation inside the production-FP4-vs-f32 noise band (ref top-2 margins < 3.3, max_abs delta 12.5) and the oracle's last-prompt-row argmax equal to the engine's first generated token; image-prompt pairs (text-only 26-token prompt vs the 224-token image prompt, W + 3 adjacent pairs, fresh serve process per run, RAM gate every start): B 406.1 ms vs N 1,626.2 ms prefill (N spread 4.9 ms), pair delta mean +1,220.1 ms, plus the vision window of 35.3 s per request — the tower GEMVs run the text-style per-token shape and are the known optimization lever. The llama.cpp mmproj comparison column was deferred to the B-series (orchestrator ruling; the oracle is THE gate). The f32 oracle chain lives in `oracle/` (`cnq_weights.py`, `ref_vit_golden.py`, `ref_vit_stages.py`, `ref_image_prompt_logits.py`) over the SAME container-dequantized weights. Report: `.superpowers/sdd/task-vit-report.md`.
- `#10` 10d, 2026-09-14: the built router GEMM switch `CROW_ROUTER_GEMM` was gated in the tolerance form and NOT LANDED, the ten-task quality gate came back RED: the numeric probe GREEN (`engine/src/bin/router_probe.rs`, the qsa_probe pattern, new; synthetic production-scale inputs, one 2,048-token chunk plus the t = 8 boundary, both dense forms vs `gemv_b` with f64 spot arbitration), masked max-rel 3.777e-3 against the 1e-2 line, max-abs 1.059e-4, top-10 expert sets IDENTICAL on 2048 of 2048 tokens; ten-task greedy with the switch on moved all ten answers (first differing token index 4 to 267 against final4) and the judgement against `docs/ten-task-expected.md` gave 0 Pass / 7 Partial / 3 Fail against the 2/5/3 record of the series (t1-read Pass to Fail: the trap assertion the protocol pre-declares false; t4-prose Pass to Partial: the constant-slot-cost assumption and the 312.44 MiB figure absent; t2b-write-refactor Fail to Partial: the record's 21-token EOS stop is gone), no degeneration, but pass 0 is below the bar of reference minus one (`decode_out/srv-10d.log`, RTX 5090); the switch stays an opt-in measurement switch, the default path untouched (parity 5 of 5 GREEN at the 61b sha256 values of record on the 10c build), pairs and the combined probe not run (the stand-down rule).
- `#10` 10c, 2026-09-14: the prefill dense GEMM variant B landed as `CROW_PF_GEMM_B` (exact `1`, default off): `gemm_fp4_dense_b` and `gemm_bf16_dense_b` widen the 8-token tile to 32 tokens (four independent n-tiles per warp, four accumulator sets, the KS smem reduce per tile) and load the weight fragments as aligned 48-byte uint4 windows with per-lane word picks (the 36 B block layout is only 4B aligned; the window of the last block ends exactly at the slab end because `bpr % 4 == 0`); per-token math is identical to the 8-token forms (same per-row k chain, levels, KS reduce). Gates (`decode_out/srv-10c.log`, RTX 5090): parity 29 of 29 GREEN with the switch OFF (8 rows `bceba6ff7724`, 512 `14c8628acbec` x3, 1024 `b2e87b2bf99a` against the reference `d211ab52ad2b`, plus `CROW_PF_GEMM_B=0` 8 rows) and ON (the same four forms byte-identical at the same values of record, plus the teacher-forced 16,064-row PX form `f217e1c55926` at 15,960,023,040 bytes under the > 26 GiB VRAM headroom gate); ten tasks 10 of 10 equal final4 with no env; KPROF dense group 2.00x (568.9 vs 284.7 ms per step; the acceptance line: 5.30 s serialized x 0.5004 = 2.65 s estimated saving, an estimate); pairs (t1-read 16,064 ids, W + 3 adjacent pairs, one chain, fresh process per run, purged container): N 18.437 s = 871.3 tok/s (spread 0.3 percent) against the default-of-record B plateau 20.726 s = 775 tok/s (B2 B3; reproduction gap -0.49 percent against the 10b stage-1 N arm 20.828 s), the two clean pairs -2.270 and -2.330 s = -11.0 and -11.2 percent (B1 18.491 s is a documented whole-run outlier kept in the log, 38a rule 9); decode watch 32.9 against 33.4 tok/s (prefill-shaped switch). One `[pf-gemm-b]` boot line per process names the form. Context: the #10 10b stage-1 scratch diet and the chunk-policy cap 4096 are committed as `41169f0` (2026-09-13); the 10b stage-2 wavefront is parked (`decode_out/10b-wip3.patch`), the pairs B arm here is therefore the 46c234b no-env default.
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
- Closed by `#51` on 2026-09-11: `decode.rs:40`, `decode.rs:45`, `parity.rs:159`, `parity.rs:164` and `parity.rs:371` now name the same container and sidecar as `serve.rs:454-455` (found 2026-09-10 in stage B, `#48`).
- Closed by `#49` on 2026-09-18: a ragged sidecar is adapted row by row and every adapted row is named (`residency::sidecar_sets`), and a file that is not a hot set at all — the converter's per-tensor `*.cnq.sidecar.jsonl` among them — is refused by name instead of asserting (found 2026-09-10 in stage B, `#49`).
- Closed by `#34` on 2026-09-11: `parity.rs:54-55` and `parity.rs:77-78` set `PYTHONIOENCODING=utf-8` and `PYTHONUTF8=1` on the oracle process itself, so a bare invocation tokenizes the same ids as a chain (found 2026-09-09 in stage A, `#25`).
- Closed by `#53` on 2026-09-11: a `parity` record header names the sampler that produced it, so the six C2 sampling series no longer read as greedy (found 2026-09-11 in the whole-branch review, first noted in `#40`).
- Closed by `#52` on 2026-09-11: `residency.rs:46` and `sf_scan.rs:9` name the `-M` container, and the `residency` warm-up sidecar moved to `CROW_HOTSETS_OUT` (found 2026-09-11 in the E8 follow-up, `#51`).
- Engine logging is not started, open as `#13`; the architecture diagrams are stale, open as `#14`.
- Ampere and Ada are not planned: the fallback stage was closed for want of a card, `#12`.
- The clippy job renders red while non-blocking: 1314 warnings, 1085 `unnecessary_cast`, measured 2026-09-11 (`#50`).

## 2026-09-14 — v0.2.1: the prefix cache survives a history edit

- **M3 multi-snapshot cache (#36 follow-up)**: `SLOTS` 1 -> 3 — the engine holds the last three turn-prompt snapshots (newest in slot 0, older aged by `rotate_right`) instead of one, so a history edit that diverges below the newest position rolls back to the previous turn's snapshot (partial reuse at the deepest common prefix `L`) instead of collapsing into a full cold prefill. The live `serve` log (2026-09-14) showed the exact collapse: an edit that dropped a 1024-token `finish=length` answer forced `COLD L 87389, prefill 87510 of 87510 tok` = 215 s, while the 28 turns around it reused warm. Save/restore stays on slot 0 (slot file byte-compatible with v0.2.0); `slot.rs::restore` invalidates the older slots first. Tests: lib 82/82 (new `an_edit_below_the_newest_snapshot_reuses_the_previous_turn` pins the log's edit case), serve 60/60. Cost: +260 MiB pageable host RAM (3 x 130 MiB), no VRAM change.

## 2026-09-14 — v0.2.0: the decode line is crossed, and the engine sees

- **62e**: the GDN big-slab GEMV geometry (32 rows per block) — decode **22.1761 ms = 45.1 tok/s** vs llama.cpp 22.27 = 44.9 on the same model, quant and machine (RTX 5090), greedy ids bit-identical (`5098f885ab3a`, 7/7). The first engine default under the llama.cpp row.
- **10b stage 1**: the per-chunk scratch diet + chunk cap 4096 — prefill 22.359 -> **20.828 s = 771.3 tok/s** (F49 pairs, 3/3). Stage 2 (the two-chunk wavefront `CROW_PF_WAVE`) is parked: correct-wave parity was byte-identical incl. the PX form, but the TG sweep measured the WDDM cross-chunk edge cost at **+40.8 to +45.5 percent SLOWER than legacy** — the lever needs a pipeline redesign, not a bugfix (parks: `decode_out/10b-wip3.patch`).
- **10c**: the dense GEMM variant B behind `CROW_PF_GEMM_B=1` — **bit-identical** (29/29 gates incl. the PX teacher-forced form), KPROF dense group 2.00x, prefill **18.437 s = 871.3 tok/s** opt-in (-11 percent, 2 clean pairs + documented outlier).
- **10d**: the router GEMM verdict — probe GREEN (max-rel 3.777e-3, top-10 expert sets identical 2048/2048) but the ten-task quality gate read **0/7/3 vs the record 2/5/3** -> NOT landed, per the loop rule; the `router_probe` harness ships as the measurement tool.
- **#VIT (#66)**: the engine image path — Crow's vision on the CNQ container's own `vit` section (332+ tensors, NVFP4): oracle parity max_abs 3.43e-06 / cos 1.000000, text parity 23/23 byte-identical, ten-task untouched. Interactive-test hotfix batch the same day (no gate chains yet, robin-tested): the tiled `gemm_fp4_f32x` projections, a per-process image-embedding cache (a re-sent history image costs a hash hit, not a tower run), the patch cap 4096 with downscale instead of a 400 refusal, serve chunk 2048 -> 4096, and log lines from request arrival. Open on the reset list: multi-image order verification (A/B), attention tiling, the planner RAM auto-fit, stop/gone-client propagation.
