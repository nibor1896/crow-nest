# Acceptance protocol — issue #83 (min_p, the log-space tail filter)

- Date: 2026-09-20. Executed by: fleet subagent (implementation + unit tests). Live/GPU acceptance: robin.
- Issue: implement min_p exactly as llama.cpp has it (PR ggml-org/llama.cpp#3841, paper arXiv:2407.01082) and make the model's real operating point (temp 1.0 / top_p 0.95 / min_p 0.01) expressible; end the "accepted and ignored" era of #28.

## What changed (file:line, all lines post-change)

| file | where | what |
|---|---|---|
| `engine/src/sample.rs:69` | `Sampler.min_p` field | `pub min_p: f32`, doc pinned to the llama.cpp form; `0` (and absent) disables, `(0,1]` filters, `min_keep 1` |
| `engine/src/sample.rs:105` | `from_env` | `CROW_MIN_P` env knob, default `0.0` (docs/env.md row added) |
| `engine/src/sample.rs:141` | `Sampler::ln_min_p` | the ONE host-computed `ln(min_p)` f32, uploaded to the device so both samplers add the same threshold constant (no libm disagreement can move the boundary) |
| `engine/src/sample.rs:183` | `Sampler::sample` | the filter: AFTER the top-k sort, BEFORE the f64 temperature softmax — `thr = cand[0].1 + ln_min_p`, keep the prefix with `value >= thr` (list is sorted desc), `truncate(keep.max(1))` is min_keep and covers a degenerate `min_p > 1` |
| `engine/src/kernels.rs:4036-4052` | sampler region comment | the 36 B params block layout `{temp, top_p, presence, min_p, ln_min_p} f32 + {top_k} i32 + 4 B reserved for #84` |
| `engine/src/kernels.rs:4174` | `sample_k` (device twin) | the same filter on `cv` between the top-k rounds and the softmax: `k2 = 1; while (k2 < k && cv[k2] >= thr) k2++;` then softmax/nucleus/draw over `k2` |
| `engine/src/gen.rs:1023, 4189` | params buffer 16 → 36 B | boot hold + fallback alloc; `enable_dev_sampler` uploads `min_p` at `[16..20]` and `ln_min_p` at `[20..24]` (`gen.rs:4199-4208`) |
| `engine/src/gen.rs:1898` | `sampler_bytes()` | VRAM ledger mirror, +20 B |
| `engine/src/bin/serve.rs:1326` | `parse_chat` | `min_p` parse unchanged since #28 (default `0.0`, strict on type), now HONORED downstream; `SamplingSent.min_p` added (`serve.rs:1350`) |
| `engine/src/bin/serve.rs:1073` | `sampling_line` (new pure fn) | the `[chat]` provenance line carries `min_p {} ({request|data sheet})`; the old "accepted and ignored" log block is REMOVED |
| `engine/src/bin/serve.rs:1456` | `sampler_from` | `s.min_p = req.min_p` — the field reaches the sampler that is armed |
| `engine/src/manager.rs:498` | ledger test pin | `sampler_bytes` moved 281_112 → 281_132 (+20 B, the params growth) — see Deviations |
| `docs/env.md` | Sampler table | `CROW_MIN_P` row (9 rows now), summary list updated |

## Spec-adherence checklist (issue text → implementation)

| #83 requirement | where it lives |
|---|---|
| "Host reference: after top-k candidate selection and BEFORE the f64 temperature softmax, filter candidates with `c.logit >= max_logit + ln(min_p)`" | `sample.rs:183-195` — exactly that predicate, on the sorted candidate list, prefix keep |
| "min_p ∈ (0,1]; ≤ 0 or absent = disabled" | host `if self.min_p > 0.0`; device `if (params[4] > 0.0f)`; serve default `0.0` |
| "Keep at least `min_keep = 1` candidate" | `cand.truncate(keep.max(1))` / device `k2 = 1;` before the while |
| "Then proceed with the existing temperature softmax → top-p → draw" | unchanged code path below the filter (f64 softmax, nucleus, xorshift64* draw) |
| "Device twin … same log-space filter on the candidate list; respect the existing SAMPLE_MAXK=64 clamp discipline; keep the f64 path untouched" | `kernels.rs sample_k` — filter sits between `sample_rounds` and the unchanged `double pr[SAMPLE_MAXK]` block; k clamps unchanged |
| "Wire-through: serve.rs already parses min_p — replace 'accepted and ignored' with real plumbing; extend the `[chat]` provenance line; make it a first-class field" | `sampler_from` plumbs it; `sampling_line` names it with provenance; the ignored-log block is gone |
| "Bit-parity gate as usual: CROW_SAMPLE_HOST=1 cross-check device vs host on the golden prompts" | unit-level: `sample::tests::the_device_mirror_draws_what_the_host_draws_min_p` (a line-for-line host mirror of the CUDA, written from the kernel source, agreeing token-for-token over 6 profile x 12 draws). The GPU run itself is robin's live gate (case 4 below) |
| llama.cpp reference math: `threshold = max_logit + logf(p_base)`, implemented in log space without softmax; `min_p <= 0` disables | implemented verbatim; the one deliberate difference: `logf` is computed ONCE on the host (`ln_min_p`) and shipped, so host and device threshold against identical bytes instead of two libms |

## Unit tests (all green, `cargo test --lib` + `--bin serve`)

| test | pins |
|---|---|
| `sample::tests::min_p_disabled_is_the_old_sampler_exactly` | min_p 0/absent = the pre-#83 draw sequence byte for byte (golden captured from the old code, 20 draws, seed 42) |
| `sample::tests::min_p_filters_exactly_the_log_space_set` | survivors are exactly `{v >= max + ln(min_p)}`, all reachable; the `>=` boundary uses the host's own f32 threshold |
| `sample::tests::min_p_one_keeps_only_the_top_and_a_degenerate_still_keeps_one` | min_p 1 = only the max; min_p > 1 keeps 1 (min_keep) |
| `sample::tests::greedy_ignores_min_p` | greedy argmax is untouched by the filter |
| `sample::tests::ln_min_p_is_the_host_computed_threshold_constant` | disabled = 0.0 (never NaN); enabled = `0.01f32.ln()` |
| `sample::tests::the_device_mirror_draws_what_the_host_draws_min_p` | host == device-mirror on random 300-wide vectors, 6 profiles x 12 sequential draws, min_p 0/0.003/0.01/0.05/0.2/1.0 |
| `serve::tests::min_p_is_plumbed_into_the_sampler_and_the_line` | Crow's own request (temp 1.0, top_p 0.95, min_p 0.01) yields `s.min_p == 0.01`, `ln_min_p == 0.01f32.ln()`, and the `[chat]` line says `min_p 0.01 (request)`; absent = `min_p 0 (data sheet)` |
| `serve::tests::the_sampling_line_says_which_values_the_request_carried` | `SamplingSent.min_p` flag (Crow sends it → `request`) |

## Test cases for robin's LIVE acceptance

Start serve (one engine process; wrap in the GPU lock if others are queued):

```
flock /tmp/crow-gpu.lock -c 'cargo run --release --bin serve -- --port 8099' 2>serve.log &
```

| # | what to check | how | expected |
|---|---|---|---|
| 1 | min_p is honored and echoed | `curl -s http://127.0.0.1:8099/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}],"temperature":1.0,"top_p":0.95,"min_p":0.01,"max_tokens":8}'`; then `grep "sampling on the device" serve.log` | the line carries `min_p 0.01 (request)` — the old `min_p accepted and ignored` line never appears again |
| 2 | absent min_p = old behavior | same curl WITHOUT `min_p`; `grep "min_p" serve.log` | `min_p 0 (data sheet)`; the answer of an identical request sent before this change (same seed row) is byte-identical |
| 3 | min_p changes the draw (it is real) | same request twice, once `"min_p":0.01`, once `"min_p":0.9`, seed fixed | the `min_p 0.9` answer is at least as deterministic/greedy-leaning (0.9 keeps only tokens within ln(0.9)~0.1 of the max); with `"min_p":1.0` only the argmax token repeats every step |
| 4 | host/device bit parity (the gate of the issue) | run one golden prompt sampled with the device sampler, then again with `CROW_SAMPLE_HOST=1` on the host path (decode/parity harness) | identical token streams |
| 5 | the quality probe (the issue's expected result) | `python3 tools/quality-probe.py --label A-minp --base-url http://127.0.0.1:8099` after sending `min_p: 0.01` from the client arm | German long-prose non-words/1000 drops from 16.40 toward the llama band (target ≤ 10 hand-checked) |
| 6 | unit suite | `cd engine && cargo test` | all green (counts in the wave-1 report) |

## Deviations from the issue text (with justification)

1. **`logf` on the device was replaced by a host-computed constant.** llama.cpp computes `max_logit + logf(p_base)` inside the sampler; CUDA `logf` and Rust `f32::ln` are different libms and could disagree in the last ulp, which would break the file's bit-parity discipline at the filter boundary. `enable_dev_sampler` uploads `ln(min_p)` once per request; both samplers then do the identical IEEE add. The filtered SET is llama.cpp's; only the provenance of the constant differs.
2. **`manager.rs` ledger pin moved (281_112 → 281_132).** The 36 B params block is device VRAM the ledger test pins by byte count; the pin's job is to mirror `sampler_bytes()` and it moved by exactly the +20 B the buffer grew. Confirmed with the orchestrator (coordinator instruction, 2026-09-20): this constant is this unit's consequence. No other manager.rs line was touched.
3. **Chain order.** llama.cpp's default chain is `top-k → top-p → min-p → temp`; issue #83's own crow-nest spec places the filter after top-k and before the (single) temperature softmax + nucleus, and the issue text wins: host and device both run `top-k → min_p → softmax → top_p → draw`. With both top_p and min_p active the filtered set can differ from llama.cpp's order in principle; against the llama ARM via llama-server the client should send `min_p` explicitly (it does, #83 point 4), and the operating point (0.01) sits far from any order-sensitive boundary.

---

## Orchestrator verification (appended 2026-09-20, all checks passed)

1. Full suite re-run by orchestrator: `cargo test` over all targets — **265 passed, 0 failed** (matches the agent's claim exactly; pre-existing suites still green).
2. Formula conformance vs the llama.cpp reference (src/llama-sampler.cpp min-p): log-space threshold `max_logit + ln(min_p)`, filter after top-k / before the temperature softmax, min_keep 1, `<= 0` disabled — all verified by direct code read (sample.rs:320-334, kernels.rs:4040ff). The ONE justified deviation (ln computed host-side once and uploaded so both sides cut at the same bytes) strengthens bit-parity rather than weakening it; chain position follows THIS issue's spec, which is binding.
3. The "accepted and ignored" era is over for min_p: the 4 remaining `grep` hits are unrelated contexts (unknown-field policy, tools); the min_p ignore block is gone (serve.rs:289 documents the honor).
4. Golden neutrality pinned by test `min_p_disabled_is_the_old_sampler_exactly` and `windowed_defaults_are_neutral_and_goldens_stay_byte_identical` — no implicit row change.
5. Footprint audit: all edits inside the ownership map (sample.rs, kernels sampler region 4037-4254 only, gen.rs device-sampler hunks only, serve.rs sampler plumbing, manager.rs ledger pins — coordinator-approved class). env.md row present; `tools/check_env_docs.py` OK after the CROW_MODEL_DIR row was added separately.
6. GPU-level parity (`CROW_SAMPLE_HOST=1` A/B on the container) intentionally deferred to robin's live acceptance / the wave-1 gate — unit-level host==device-mirror proof exists (6 profiles × 12 draws).

## robin's live-acceptance short list

| # | check | how | expected |
|---|---|---|---|
| 1 | min_p honored end-to-end | `curl serve /v1/chat/completions` with `"min_p": 0.01, "temperature": 1.0, "top_p": 0.95` | provenance line reads `min_p 0.01 (request)`; no "ignored" warning |
| 2 | disabled = old behavior | same request without `min_p`, greedy | byte-identical to a pre-#83 build's answer (golden discipline) |
| 3 | host vs device bit-parity | `CROW_SAMPLE_HOST=1` vs default, same seed/row | identical token streams |
| 4 | the symptom | re-run the quality probe at the real operating point | non-word rate moves toward the llama band (issue #83's headline target) |
