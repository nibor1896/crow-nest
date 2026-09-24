#!/usr/bin/env bash
# Usage, from the repo root:  tools/gate-linux.sh [outdir]   (default decode_out/gate) - runs the five
# cheap gates of record (parity 8 / 512 / P8 teacher-forced, decode run 32, host-side checks), prints
# GREEN/RED per item and exits non-zero on any RED.
#
# Why this script exists: every commit on this branch has to reproduce the SAME bytes, and the values
# below are the values of record, not tuning knobs. Nothing here may be edited to make a gate pass.
#
# Provenance of every expected value (all measured on RTX 5090 / Arch Linux / driver 610.57.04 /
# CUDA 13.3.1 / NVRTC 13.3.33, inside the memory-bounded scope this script sets up):
#
#   8 rows    bceba6ff7724...  commit 9f12429 "Linux port": byte-identical to the WINDOWS reference
#                              decode_out/parity-62d-ref8 - the one form that survived the toolchain
#                              drift. 11,919,360 B = 12 rows x 248,320 f32. Re-confirmed at 0c9feb5,
#                              74c79f2, bb9d2ca, 7ddd296, 0667e0b.
#   512 rows  838723470927...  commit 9f12429: the LINUX value of record. The Windows bytes differ
#                              (logits drift from row 23, ids identical in all 517 positions) - that
#                              is the NVRTC/driver JIT, not the port (9f12429 proved it over four
#                              configurations). Re-confirmed at 0c9feb5, 74c79f2, bb9d2ca, 7ddd296.
#   P8 tf     3bb3e69edf90...  commit 9f12429, decode_out/linux-port/parity-p8tf: the teacher-forced
#                              form (prefill 8 ids, the other 504 fed through decode_step) - it puts
#                              the DECODE path under the parity contract. Re-confirmed at 74c79f2,
#                              bb9d2ca, 7ddd296.
#   run 32    the 32 ids       commit bb9d2ca ("the 32 generated ids identical to the 74c79f2
#                              binary") and 7ddd296 ("decode run 32 ids identical").
#   tests 171 / clippy 1422    commit 7ddd296 and 0667e0b gave 144 = 84 lib + 60 serve; TASK H
#                              (2026-09-17) added the three `cnq::tests::page_runs_*` unit tests of
#                              the PLE row fetch, so 147 = 87 lib + 60 serve. TASK J (2026-09-17)
#                              added six tests of the tool-call `arguments` contract - three in
#                              `toolcall.rs` for the invariant that an abandoned call still leaves a
#                              parseable JSON object, three in `bin/serve.rs` for the normalization,
#                              the rewrite note and the neighbouring template hazards - so
#                              153 = 90 lib + 63 serve. TASK K (2026-09-17) added twelve more:
#                              `toolcall.rs` `mod arguments_contract` (the HTML hazards and the
#                              4,000-case random sweep of the same invariant), `cuda.rs`
#                              `mod alloc_failure` (the named allocation failure, injected without a
#                              GPU, and the request scope that turns it into a 503), `vit.rs`
#                              `mod reserve` (the three byte counts the planner subtracts and the
#                              [budget] line that names them) and four in `bin/serve.rs` (the
#                              producer-end repair, the broken-string note, the serde byte window
#                              and the per-index accumulator) - so 165 = 98 lib + 67 serve.
#                              #67 (2026-09-18) added six in `bin/serve.rs` for the reasoning
#                              filter - the passthrough of a stream without a tag, the stray
#                              closing tag at every split point, the leading think block, the
#                              tool-call fragments the filter must not touch, what the REAL
#                              template does with a stored `</think>` and what the normaliser
#                              makes of it - so 171 = 98 lib + 73 serve. #68 (2026-09-18) added
#                              three that pin the presence-penalty CONTRACT and the new sampling
#                              provenance line: `sample.rs`
#                              `the_presence_penalty_is_applied_once_per_distinct_token` (presence
#                              and frequency pick different tokens on its logits) and
#                              `the_penalty_set_is_this_answers_tokens_only` (a fresh `Sampler`
#                              penalizes nothing, which is what the server builds per request),
#                              plus `bin/serve.rs`
#                              `the_sampling_line_says_which_values_the_request_carried` - so
#                              174 = 100 lib + 74 serve. #49 (2026-09-18) added three in
#                              `residency.rs` for the hot-set sidecar loader: the RAGGED file
#                              (rows of unequal length, row 0 exactly at N - the case the old
#                              loader could not see) adapted row by row with every adapted row
#                              named, the single-N file that keeps its one adapted line, and the
#                              refusals of a file that is not a hot set at all (the converter's
#                              per-tensor `*.cnq.sidecar.jsonl`, a missing `sets`, 47 rows, an id
#                              outside 0..512, an id twice in one row) - so 177 = 103 lib +
#                              74 serve. #60 (2026-09-18) added the first two tests `bin/parity.rs`
#                              has ever had, for the arm rule of the record header: the llama arm's
#                              `operating_point` is the greedy string whatever sampler the series
#                              used (`point_for`, the pure half - no server, no GPU, no env), and
#                              the same through the env-reading front door with `CROW_SAMPLE=1` in
#                              the environment, which is the accident the ticket names - so
#                              179 = 103 lib + 74 serve + 2 parity. #54 (2026-09-18) added
#                              four in `bin/serve.rs` for the gone-client probe of the
#                              `stream:false` path: the pure decision table of
#                              `peer_from_poll` / `gone_reason` (with the four `revents`
#                              constants asserted against `libc` on Linux), the `PROBE_EVERY`
#                              cadence, the probe against a REAL loopback socket in its three
#                              shapes (live, `shutdown(Write)`, closed - the last two are the
#                              same wire event, which is the whole reason the baseline exists)
#                              and the no-socket `CollectSink` that never stops a loop - so
#                              183 = 103 lib + 78 serve + 2 parity. #65 (2026-09-18) added
#                              four more in `bin/parity.rs` for the bounded retry around the
#                              two oracle python children: the pure verdict table of
#                              `child_verdict` (the #65 shape itself - a non-zero exit with
#                              an EMPTY stderr - plus the Windows exit codes whose name is
#                              the whole message and the zero exit with nothing on stdout),
#                              and three against a REAL `/bin/sh` stub child, which needs no
#                              oracle venv and no GPU: a child that fails ONCE with an empty
#                              stderr is retried and the task goes on with the failed
#                              attempt in the record, a child that fails EVERY attempt still
#                              fails the task (fail-closed), and a 60,290-byte payload - the
#                              longest of the ten frozen prompts - reaches the child whole
#                              through STDIN and never through the command line - so
#                              187 = 103 lib + 78 serve + 6 parity. #64 (2026-09-18) added
#                              the first three tests `bin/decode.rs` has ever had, for the
#                              pure half of the new `decode selftest` mode (F5, the quant
#                              package's own self-test) - the manifest contract with its
#                              four named refusals (not JSON, no `checks`, empty `checks`,
#                              a check without `max_abs_gate`, which a default would read
#                              as a gate of 0), the INCLUSIVE 0.125 bound together with
#                              the NaN rule (`f32::max` drops a NaN operand, so a NaN
#                              output leaves max_abs small and only the count sees it) and
#                              the refusal of a golden whose byte length does not match
#                              the shape the manifest declares - so
#                              190 = 103 lib + 78 serve + 6 parity + 3 decode. All three
#                              run without a GPU and without a package.
#                              #13 (2026-09-18) added ten in the new `log.rs`, all of them
#                              without a GPU, a model or an installed subscriber: the
#                              `CROW_LOG` filter rule (unset is INFO, a per-target string
#                              passes through, a string EnvFilter refuses falls back WITH a
#                              note instead of silencing the process), the two rotation
#                              knobs with their clamps, both per-OS default log directories
#                              (`log_dir_from` takes the OS as an argument, so the Windows
#                              rule is tested on Linux), the Gregorian calendar the archive
#                              names and the file timestamps are built from, the rotation
#                              DECISION (size, the UTC day boundary, the empty file that
#                              never rotates, and the NAME ORDER of the collision counter -
#                              the bug the live proof found: `-1` sorted before the
#                              unsuffixed name of its own second and the prune deleted the
#                              wrong archive), the writer itself against a real temp
#                              directory at a 1 KiB limit (k of N kept, no plain rotated
#                              file left behind, a real gzip whose lines come back uncut,
#                              and the kept window contiguous with the live file), the
#                              restart that appends to `engine.log` instead of starting a
#                              fresh 64 MiB, the boot report as ONE valid JSON line with
#                              48 `hot_per_layer` entries, the routing line's fields and its
#                              two derived rates, and the call-site cost of one line the
#                              three ways this engine can pay for it (a synchronous
#                              `writeln!` 333-354 ns, an ENABLED tracing event through the
#                              non-blocking rotating file 347-373 ns, a DISABLED one 0.6 ns;
#                              loose ceilings, as a regression guard against a call site
#                              that starts BLOCKING) - so
#                              200 = 113 lib + 78 serve + 6 parity + 3 decode.
#                              #69 (2026-09-18) added two more in `bin/decode.rs`, both
#                              without a GPU and without a package: the manifest this
#                              repository SHIPS parses into its TWO checks with the kinds
#                              `decode selftest` implements and the two gates of record
#                              (0.125 layer 0, 0.625 layer 3), and the zero-output refusal -
#                              an identically zero engine output is a FAIL of its own before
#                              the gate is consulted, because zeros against a golden report
#                              the golden's own numbers back (max_abs = max|golden|, rel_L2
#                              exactly 1, corr exactly 0) and would PASS a gate wide enough,
#                              which is how a dead debug path read as a measurement for
#                              sixteen days - so 202 = 113 lib + 78 serve + 6 parity +
#                              5 decode.
#                              #68 (2026-09-18) added four in `bin/serve.rs` for the
#                              cross-turn repeat counter, all pure host logic with no GPU
#                              and no model: the ring's run and its distance back (a run is
#                              CONSECUTIVE, so an answer that returns after another one
#                              starts its run over, and the run itself is NOT capped by the
#                              ring - the live session's 48 identical answers report 48),
#                              the hash over the GENERATED IDS (order and length matter,
#                              and an answer older than the eight-deep ring is out of it),
#                              the single-token rule (one id AND `finish stop`; the client's
#                              own `max_tokens 1` budget is `length` and says nothing about
#                              the model), and the WARN threshold plus the `[chat]` suffix -
#                              a healthy answer adds NOTHING to that line, which is what
#                              keeps `replay-toolcalls.py`, `drift-chain.sh` and this script
#                              reading the line of record - so
#                              206 = 113 lib + 82 serve + 6 parity + 5 decode.
#   tests 206 -> 212           #72 (2026-09-18) adds SIX library tests for the planning
#                              arithmetic the fix rests on, all pure and GPU-free: in
#                              `vit.rs`, that the derived reserve IS what `Engine::load`
#                              holds at boot (the twelve scratch buffers at the 4096-patch
#                              cap plus the two mrope span tables at n_ctx, nothing left
#                              pending) and that the `[budget]` line says HELD, lazy or
#                              partly pending; in `manager.rs`, the post-plan ledger - its
#                              VRAM total (277.6 MB = the 277.3 MB reserve + the 0.27 MB
#                              device sampler), the empty case, and the one that pays for
#                              the whole issue: the prefix-cache snapshots (3 x 124.6 MiB)
#                              are HOST RAM and must stay OUT of the VRAM total, or the
#                              planner would drop ~150 hot experts for nothing - plus the
#                              headroom floor (256 MiB) with the issue's own 35.7 MiB as
#                              the SHORT case. So 212 = 119 lib + 82 serve + 6 parity + 5 decode.
#   tests 212 -> 217           the toolcall array/object JSON repair (robin's live session 2026-09-18, no issue): five lib tests. 217 = 124 lib + 82 serve + 6 parity + 5 decode.
#   tests 217 -> 221           #73 (2026-09-18) adds FOUR library tests in `vit.rs`, all pure host
#                              arithmetic with no GPU and no CUDA: the channel-major patch row
#                              with its duplicated temporal frame, the solid-colour row in R G B
#                              order (the assertion `red -> Black` would have failed FIRST if the
#                              cause had been the channels), the align-corners bilinear taps of
#                              the learned position table in h0w0 order, and the rotary vector's
#                              first half as the patch ROW - each against hand-computed values. The bug itself was a missing stream sync, not a layout
#                              error - these pin the layout the symptom IMITATED, so the
#                              suspect it looked like stays cleared. 221 = 128 lib + 82 serve
#                              + 6 parity + 5 decode.
#   tests 221 -> 228           #74 (2026-09-18) adds ONE library test in `tokenizer.rs` and SIX in
#                              `bin/serve.rs` for the thinking path, all pure - the template and
#                              the tokenizer, no GPU and no model. `tokenizer.rs`: the third
#                              template variable, with UNDEFINED as the render of record (the
#                              template's own `xhigh` default), `low` and `medium` as their own
#                              renders, the words THIS template does not have (`high`, `none`,
#                              `max`) raising, and thinking OFF rendering the same bytes for
#                              every word. `bin/serve.rs`: the twelve-row resolution table of
#                              the two doors plus the eight refusals, the ids of record for a
#                              request that names no level (against the frozen oracle ids, not
#                              against a second call), the `[chat]` provenance line, the filter
#                              starting `Inside` for a thinking request (with the same bytes
#                              through a `Lead` filter as the negative control - that is the
#                              bug), the budget that runs out mid-thought in both request
#                              forms, and a stored `reasoning_content` through the REAL
#                              template. So 228 = 129 lib + 88 serve + 6 parity + 5 decode.
#   tests 228 -> 231           #77 (2026-09-18) adds THREE library tests in `cnq.rs` for the
#                              dense-BF16 overlay, all pure host logic with no GPU, no
#                              container and no overlay file: the KIND rule `--kinds` and the
#                              boot log count by (it has to be the twin of the converter's, or
#                              an ablation would select nothing and read as "the originals
#                              change nothing"), the TABLE OF REFUSALS `Cnq::attach_overlay`
#                              applies before a single byte is loaded - the accepting case
#                              first, then a missing `overlay` block, a different base name, a
#                              different base byte count, no tensors, a dtype that is not
#                              bf16, a name the base does not carry, the same name in another
#                              SECTION, a different value count and a different shape - and
#                              `byte_len` on a shadowed tensor (2 B per value against 36 B per
#                              64: the 16 / 4.5 the residency planner has to see). So
#                              231 = 132 lib + 88 serve + 6 parity + 5 decode.
#   tests 231 -> 232           #79 (2026-09-19) adds ONE library test in `cnq.rs` for the
#                              ROUTED-EXPERT overlay, pure host logic with no GPU, no
#                              container and no overlay file:
#                              `an_expert_overlay_is_accepted_as_nvfp4_of_the_same_byte_length_and_refused_otherwise`
#                              - the accepting case (nvfp4 over nvfp4, same byte length, its
#                              OWN global scale), then the three refusals #79 added to the
#                              table: a routed expert offered as bf16 (the residency planner
#                              cuts per-expert slabs out of `byte_len / 512` and hands them to
#                              kernels that index 36 B per 64 values, so a bf16 expert would be
#                              a silently wrong-size slab), an nvfp4 overlay over a bf16 base
#                              keep, and a global scale that is zero, negative or NaN. The
#                              existing ten-row refusal test changed by ONE line and did not
#                              grow: its `nvfp4` case USED to be the refusal "bf16 only" and is
#                              now the accepted #79 case, so the refused dtype there is `f32`.
#   tests 232 -> 234           #81 (2026-09-20) adds TWO serve tests for the reasoning
#                              budget: the integer dialect of `reasoning_budget_tokens` and
#                              `is_inside` counting only what the block still holds.
#                              So 234 = 133 lib + 90 serve + 6 parity + 5 decode.
#   clippy 1458 -> 1459        #82 (2026-09-20) adds NO clippy warning: measured 1459 at
#                              7586234 in a clean worktree BEFORE the SIGHUP/SIGQUIT watcher
#                              edit, and 1459 after it - the +1 is TOOLCHAIN DRIFT from the
#                              arch system update of 2026-09-20 07:22 (new clippy lints such
#                              as manual_div_ceil), not the engine. #79's note below stays
#                              for the record.
#   clippy 1458 (unchanged)    #79 adds no clippy warning: the engine change is one widened
#                              match in `overlay_refusal`, two fields on `OverlayReport`, one
#                              boot line and one panic in `residency.rs`. The converter is a
#                              separate crate and clippy lints this package only.
#   clippy 1421 -> 1458        #77 RAISED the count by 37, and every one of them is the
#                              --all-targets noise this tree already carries in bulk: +29
#                              `casting to the same type is unnecessary (u64 -> u64)` from the
#                              launch arguments the seventeen dense kinds now pass through
#                              their `PW::or_bf16*` wrappers (`Dev` IS `u64`; the call lines
#                              themselves were written without the casts, which is why it is
#                              29 and not 140), +3 `unsafe function's docs are missing a
#                              # Safety section` for those three wrappers, +2 `manually
#                              reimplementing div_ceil` and +1 `this function has too many
#                              arguments (8/7)` for `or_bf16_s`, plus 2 in the neighbouring
#                              rewritten blocks. No new lint KIND appears and no warning was
#                              removed. Counted the same way as the 1494 -> 1480 -> 1426 ->
#                              1422 -> 1421 series: grep -cE '^warning: ' on --all-targets.
#   clippy 1421 (history)      #13 (2026-09-18) LOWERED the count of record by one, and by
#                              exactly one: `warning: redundant reference in `eprintln!`
#                              argument` at `gen.rs:2890` is gone because that line is a
#                              `tracing` event now. The other 153 converted sites and the
#                              whole of `log.rs` add no warning, and neither do the 16 new
#                              crates (clippy lints this package only). It is the
#                              --all-targets form counted as grep -cE '^warning: ', the form
#                              the 1494 -> 1480 -> 1426 -> 1422 -> 1421 series was counted
#                              with.
#
# Environment: CROW_CNQ / CROW_HOTSETS / CROW_GRAPH / CROW_MMA are set here exactly as the runs of
# record had them; CUDA_LIB names the CUDA runtime directory (default ~/.local/share/crow/cuda/lib).
# Every engine run goes inside `systemd-run --user --scope` with MemorySwapMax=0 / MemoryHigh=52G /
# MemoryMax=54G - the same bounded cgroup tools/serve-linux.sh uses (issue #15). Runs are SEQUENTIAL
# on purpose: the RAM gate refuses a second engine while the first one holds the pinned tier.

set -u

root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
out="${1:-$root/decode_out/gate}"
case "$out" in /*) ;; *) out="$root/$out" ;; esac
bin="$root/engine/target/release/decode"
cuda_lib="${CUDA_LIB:-$HOME/.local/share/crow/cuda/lib}"

# ---- the values of record (see the provenance block above) --------------------------------------
SHA8="bceba6ff772431dedf57631e83c7418d3f0bca3ab7e33da828f8da5d12a122a2"
BYTES8="11919360"
SHA512="8387234709271515b091b1c4dbd0d59c66550d0e3feab551a6418d30b55c9105"
SHAP8="3bb3e69edf90a6c3839222d1ceae7fe06aed1ba49813daa1f7487e3c6e7cff2d"
IDS32="[13, 248046, 198, 248045, 74455, 198, 248068, 198, 760, 1156, 682, 3766, 264, 11316, 25, 328, 760, 3841, 13477, 37550, 33075, 888, 279, 15217, 5388, 1149, 271, 1919, 7701, 310, 381, 264]"
TESTS="387"
CLIPPY="1505"
#   tests 358 -> 387   v0.5.0 (2026-09-24, `9c9fd51`): #106, #107/#108/#109, #111, #112, #113, #114, #85/#92.
#                              Measured 2026-09-24 by running the test binaries built at `9c9fd51`: lib 259 / 0 /
#                              3 ignored, serve 117, decode 5, parity 6 = 387 / 0. Clippy NOT re-counted.
#   tests 335 -> 358   integration-0923 + 0923 fixes (2026-09-23, release-2026-09-23): #103 cold tier /
#                              free_for_pin, #102 CROW_KV + unread-CROW_* warning, #100/#101 prefix cache,
#                              the ue4m3 0x7F encoder test (5c6891a), tests_act_prescale (488a840),
#                              the toolgrammar closer/owed-parameter cases (492f137) and tests_ple_row
#                              (85a48e7). Measured 2026-09-23: lib 237 / 0 / 1 ignored, decode 5, parity 6,
#                              serve 110 = 358 / 0. Clippy NOT re-counted on this branch.
#   NOTE (2026-09-23): the per-row activation pre-scale (488a840) and the PLE row-offset fix
#                              (85a48e7) change the engine's numerics ON PURPOSE, so the three parity
#                              shas and the run32 ids above are expected to move. They are the values
#                              of the pre-0923 engine; new values of record need a gate run on the GPU,
#                              which has not happened yet. Until then items 1, 2, 3 and 6 are expected RED.
#   tests 320 -> 335 / clippy 1505 (unchanged)   #93 (2026-09-22): eleven toolgrammar
#                              tests (the Crow 3dbc015 tools compile; the byte machine on well-formed
#                              calls and on the observed failure shapes; the JSON subset; required /
#                              parallel; masks on the real vocabulary agree with the per-token check and
#                              are never empty; a random walk finds no dead end; the parser turns an
#                              accepted call into the declared arguments; the CPU cost measurement),
#                              one sample.rs test (rebook_plan against a host model of the device accept)
#                              and three serve tests (tool_choice / parallel_tool_calls and their 400s,
#                              the gate per switch and tool_choice, the redraw landing on `old` where the
#                              rows prefer `old_string`). Measured 334 / 0 / 1 ignored, clippy 1505.
#   tests 319 -> 320   #91 (2026-09-22): one serve test (`crow_force_ids` parse,
#                              its named 400s, the `crow_id`-carrying logprob entry).
#   tests 313 -> 319 / clippy 1505 (unchanged)   #91 logprobs (2026-09-22): two sample.rs tests
#                              (pos_logprobs against an f64 log-softmax reference; exact ties, NaN and -inf),
#                              one tokenizer test (token_bytes concatenate to the exact text, split UTF-8
#                              included) and three serve tests (the request fields and their named 400s,
#                              the OpenAI entry in both wire forms, no `logprobs` key anywhere when off).
#                              Measured 318 / 0 with the NVRTC ptx test skipped (no NVRTC run for this
#                              task; 319 with it), clippy 1505, no new warning.
#   tests 305 -> 313 / clippy 1505 (unchanged)   #99 (2026-09-22): six toolcall tests (the live
#                              markup-only shapes, the named cut, both give-up names, the split sweep of
#                              the records) and two serve tests (decide_finish, crow_malformed_calls on
#                              the final chunk and the document). Measured 313 / 0 and 1505, no new warning.
#   tests 265 -> 305 / clippy 1480 -> 1505   wave 2 of the quality fleet (2026-09-21): #85 adds the DRY
#                              suite, #92 the tier suite, #86 the nine stopstr cases plus the serve pipeline
#                              pins, #96 the scaled-table and warn tests, the probes' regression cases.
#                              The 25 new clippy warnings are the SAME lint classes as before (u64->u64,
#                              div_ceil, doc indentation) in the new code - measured, no new class.
#   tests 234 -> 265        wave 1 of the quality fleet (2026-09-20/21): #83/#84 add eighteen
#                              (six min_p + one serve row in sample.rs/serve.rs, eight penalty +
#                              three serve), #94 phase 1 adds thirteen (meta.rs). All measured,
#                              265 = 234 + 31, zero failures.
#   clippy 1459 -> 1480     same wave: the 21 new warnings are the SAME lint classes the engine
#                              already carries engine-wide after the 2026-09-20 07:22 toolchain
#                              drift (u64->u64 casts, manual div_ceil, doc indentation) hitting
#                              the new code of meta.rs / sample.rs / the sampler serve plumbing -
#                              measured 1480, no new lint class introduced.

red=0
green() { printf 'GREEN  %-28s %s\n' "$1" "${2:-}"; }
fail()  { printf 'RED    %-28s %s\n' "$1" "${2:-}"; red=1; }

[ -x "$bin" ] || { echo "gate-linux.sh: no $bin - build it with 'cd engine && cargo build --release'" >&2; exit 2; }
[ -d "$cuda_lib" ] || { echo "gate-linux.sh: no CUDA runtime directory at $cuda_lib - set CUDA_LIB" >&2; exit 2; }
mkdir -p "$out"
cd "$root"

# one engine at a time, and never on a busy GPU: a second pinned tier is exactly what the RAM gate
# is there to refuse, and a busy GPU would make the run fail for a reason that is not the code.
precheck() {
    local used procs
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)
    # -x: match the process NAME exactly. Without it the pattern matched the
    # comm of an unrelated `tmux: server` and refused every engine item
    # (2026-09-17, TASK H) - the gate then printed RED for a machine that was idle.
    procs=$(pgrep -a -x 'serve|decode|parity|llama.*' | tr '\n' ' ')
    if [ "${used:-9999}" -ge 2000 ]; then echo "  precheck: GPU holds ${used} MiB - refusing to run" >&2; return 1; fi
    if [ -n "$procs" ]; then echo "  precheck: an engine is alive ($procs) - refusing to run" >&2; return 1; fi
    return 0
}

# scope_run <logfile> <extra env assignments...> -- <args to decode>
scope_run() {
    local log="$1"; shift
    local extra=()
    while [ "$1" != "--" ]; do extra+=("$1"); shift; done
    shift
    systemd-run --user --scope --slice=session.slice --quiet \
        -p MemorySwapMax=0 -p MemoryHigh=52G -p MemoryMax=54G \
        env CROW_CNQ=converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq \
            CROW_HOTSETS=decode_out/hotsets-M-longctx2100-n160.json \
            CROW_GRAPH=1 CROW_MMA=1 "LD_LIBRARY_PATH=$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" \
            ${extra[@]+"${extra[@]}"} \
            "$bin" "$@" > "$log" 2>&1
}

# #103 (2026-09-23): every engine item is gated on the engine's own
# free_for_pin (tools/pin-room.sh -> engine ramcheck) instead of MemAvailable +
# balloon. The NVIDIA driver pool an earlier engine left is reclaimable and
# counted as free there; the engine's allocation takes it back itself.
. "$root/tools/pin-room.sh"
pool_recover() {
    pin_room 46 || echo "  pool_recover: free_for_pin below 46 GiB - live processes hold the RAM (see above); item may refuse"
}

# parity_item <name> <ids.json> <expected sha> <expected bytes|-> <extra env...>
parity_item() {
    local name="$1" ids="$2" want="$3" wantb="$4"; shift 4
    local dir="$out/$name" log="$out/$name.log"
    precheck || { fail "$name" "precheck refused"; return; }
    pool_recover
    rm -rf "$dir"
    local t0 t1
    t0=$(date +%s%3N)
    scope_run "$log" "$@" -- parity "$ids" "$dir"
    local rc=$?
    # placement fallback (2026-09-21): the -M default tier needs free_for_pin
    # >= cold 43.51 + margin 3 = 46.5 GiB; a sessionized machine can sit ~1 GiB
    # under that without anything being wrong. Retry ONCE with a 42 GiB pinned
    # budget on exactly the residency refusal - placement is numerics-neutral
    # (#88 proved dumps byte-identical across hot-set placements), so the sha
    # contract is unchanged; the fallback is LOUD.
    if [ $rc -ne 0 ] && grep -q "refusing to pin" "$log"; then
        echo "  $name: default tier refused at the margin - retrying with CROW_RAM_MARGIN_GB=1 (the planner keeps the default tier; only the safety margin yields - outputs byte-identical)"
        scope_run "$log" CROW_RAM_MARGIN_GB=1 "$@" -- parity "$ids" "$dir"
        rc=$?
    fi
    t1=$(date +%s%3N)
    local wall; wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}')
    if [ $rc -ne 0 ]; then fail "$name" "decode exit $rc, see $log"; return; fi
    local got bytes
    got=$(sha256sum "$dir/gpu-logits.f32" | cut -d' ' -f1)
    bytes=$(stat -c%s "$dir/gpu-logits.f32")
    if [ "$got" != "$want" ]; then fail "$name" "sha256 $got != $want (${wall}s)"; return; fi
    if [ "$wantb" != "-" ] && [ "$bytes" != "$wantb" ]; then fail "$name" "$bytes B != $wantb B (${wall}s)"; return; fi
    green "$name" "sha256 ${got:0:12} $bytes B  ${wall}s"
}

echo "== gate-linux.sh  $(date -Is)  $(git -C "$root" rev-parse --short HEAD 2>/dev/null)  -> $out"

# 1) parity 8 rows - the form that is byte-identical to Windows
parity_item parity8 decode_out/parity-ids.json "$SHA8" "$BYTES8"

# 2) parity 512 rows - the Linux value of record
parity_item parity512 decode_out/real512-ids.json "$SHA512" -

# 3) P8 teacher-forced - the decode path under the parity contract (graph off by construction)
parity_item p8tf decode_out/real512-ids.json "$SHAP8" - CROW_GRAPH=0 CROW_PARITY_PREFILL=8

# 6) decode run 32 - the 32 generated ids of record
if precheck; then
    pool_recover
    log="$out/run32.log"
    t0=$(date +%s%3N)
    scope_run "$log" -- run decode_out/parity-ids.json 32 "$out/run32"
    rc=$?
    if [ $rc -ne 0 ] && grep -q "refusing to pin" "$log"; then
        echo "  run32: default tier refused at the margin - retrying with CROW_RAM_MARGIN_GB=1 (the planner keeps the default tier; only the safety margin yields - outputs byte-identical)"
        scope_run "$log" CROW_RAM_MARGIN_GB=1 -- run decode_out/parity-ids.json 32 "$out/run32"
        rc=$?
    fi
    t1=$(date +%s%3N)
    wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", (b-a)/1000}')
    cp -f "$root/decode_out/run.json" "$out/run32.json" 2>/dev/null
    if [ $rc -ne 0 ]; then
        fail run32 "decode exit $rc, see $log"
    else
        gotids=$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['trace'])" "$out/run32.json")
        if [ "$gotids" = "$IDS32" ]; then green run32 "32 ids of record  ${wall}s"; else fail run32 "ids differ: $gotids"; fi
    fi
else
    fail run32 "precheck refused"
fi

# 10) host-side checks - no GPU, no model
( cd "$root/engine" && env LD_LIBRARY_PATH="$cuda_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" cargo test --release > "$out/cargo-test.log" 2>&1 )
tpass=$(grep -hoE '^test result: ok\. [0-9]+ passed' "$out/cargo-test.log" | awk '{s+=$4} END{print s+0}')
tfail=$(grep -hoE '[0-9]+ failed' "$out/cargo-test.log" | awk '{s+=$1} END{print s+0}')
if [ "$tpass" = "$TESTS" ] && [ "$tfail" = "0" ]; then green "cargo test" "$tpass passed, 0 failed"
else fail "cargo test" "$tpass passed / $tfail failed, expected $TESTS / 0"; fi

( cd "$root/engine" && cargo clippy --release --all-targets > "$out/clippy.log" 2>&1 )
cw=$(grep -cE '^warning: ' "$out/clippy.log")
if [ "$cw" = "$CLIPPY" ]; then green clippy "$cw warnings"; else fail clippy "$cw warnings, expected $CLIPPY"; fi

for t in check_env_docs check_readme_dates check_model_card_dates; do
    if [ -f "$root/tools/$t.py" ]; then
        if PYTHONIOENCODING=utf-8 python3 "$root/tools/$t.py" > "$out/$t.log" 2>&1
        then green "$t" "exit 0"; else fail "$t" "exit $?, see $out/$t.log"; fi
    else
        green "$t" "not in this tree - skipped"
    fi
done

echo "== gate-linux.sh: $([ $red -eq 0 ] && echo 'ALL GREEN' || echo 'RED - see above')"
exit $red
