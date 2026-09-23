# Issue #94, phase 1 — the metadata gate (boot-time parse + assert-equal, ZERO numeric change)

Built 2026-09-20. Scope of this phase: the plumbing and the gate, exactly as the
issue's step 2 prescribes — "Do NOT change numerics today: boot-time assert
that every parsed value equals the current pinned constant. Green on the current
checkpoint proves the plumbing byte-identical; mismatch on a future checkpoint
becomes a loud error instead of silent corruption (the llama.cpp discipline)."

## What was built

- **`engine/src/meta.rs`** (new): `ModelMeta` — the parsed checkpoint metadata
  (`models/<name>/config.json` + `generation_config.json`), `checks()` / `verify()`
  (every constant compared against the pin, `verify()` returns the named
  mismatches), `from_container()` (locates the config next to the container) and
  `assert_pinned()` (the boot door: INFO line / WARN line / loud panic).
- **`engine/src/lib.rs`**: module registration only (`pub mod meta;`).
- **`engine/src/boot.rs`**: `open_model` calls `meta::assert_pinned(&cnq_path)`
  FIRST — before `Cnq::open` maps the 32 GiB container and before the CUDA
  context exists — so a mismatched checkpoint dies at the front door having
  touched nothing. `decode`, `parity` and `serve` all come through this one door.
- Nothing else. `sample.rs`, `kernels.rs`, `manager.rs`, `gen.rs`, `geo.rs`,
  `serve.rs`, `bin/decode.rs`, `tools/` untouched; no pinned value, no kernel,
  no math changed.

The truth source is the checkpoint dir, not the container: the CNQ trailer JSON
carries only quant geometry (`blob_offset`, `block_geometry`, `format`,
`sections`, `tensors`, `version` — read out of `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`
2026-09-20), so `models/…/config.json` is where the constants live. The dir is
found the way the bins find the container: repo root = the container's
directory's parent (both cwd forms — `converter/x.cnq` from the repo root,
`../converter/x.cnq` from `engine/` — give it), then `models/*/config.json`; a
lone candidate wins, several candidates are disambiguated by the model name in
the container's file stem (`…-CNQ…` split), and `CROW_MODEL_DIR` overrides all
of it. Layer constants are read from `text_config` (Qwen multimodal nesting)
with a flat top level as the fallback of the same read.

## The constant table (what the gate asserts, every boot)

| # | name | pinned value (engine site) | config source key |
|---|------|----------------------------|-------------------|
| 1 | `rms_norm_eps` | `1e-6` (kernels.rs, every rms + LayerNorm site: 1426, 1444, 1599, 1769, 2352, 3413, 4289) | `text_config.rms_norm_eps` |
| 2 | `rope_theta` | `1e7` (manager.rs:304 boot RoPE table) | `text_config.rope_parameters.rope_theta` (fallback `text_config.rope_theta`) |
| 3 | `rope_pairs` | `32` (`geo::ROPE_PAIRS`) | `partial_rotary_factor × head_dim / 2` = 0.25 × 256 / 2 |
| 4 | `attention_scale` | `0.0625` (kernels.rs, 5 attention variants: 1860, 1932, 2096, 2194, 3709) | `1/sqrt(text_config.head_dim)` |
| 5 | `gqa_ratio` | `12` (kernels.rs `head / 12`: 1850, 1918, 2075, 3695) | `num_attention_heads / num_key_value_heads` = 24 / 2 |
| 6 | `hidden_size` | `2560` (`geo::H`) | `text_config.hidden_size` |
| 7 | `head_dim` | `256` (`geo::AHD`) | `text_config.head_dim` |
| 8 | `q_heads` | `24` (`geo::NQ`) | `text_config.num_attention_heads` |
| 9 | `kv_heads` | `2` (`geo::NKV`) | `text_config.num_key_value_heads` |
| 10 | `num_hidden_layers` | `48` (`geo::LAYERS`) | `text_config.num_hidden_layers` |
| 11 | `layer_layout` | `layer % 4 == 3` is full attention: 12 attn + 36 gdn (`geo::is_attn`, `ATTN_LAYERS`, `GDN_LAYERS`) | `text_config.layer_types` (positions AND counts; an unknown type string is named) |
| 12 | `full_attention_interval` | `4` (`geo::is_attn`'s `layer % 4`) | `text_config.full_attention_interval` |
| 13 | `vocab_size` | `248320` (`geo::V`) | `text_config.vocab_size` |
| 14 | `max_position_embeddings` | `262144` (`geo::Config::default().context`) | `text_config.max_position_embeddings` |
| 15 | `eos_ids` | `[248046, 248044]` (`sample::EOS_IDS`, read live — the pin stays written once, in sample) | `generation_config.json eos_token_id` (fallback `text_config.eos_token_id`) |
| 16 | `ple_eos` | `248044` (`geo::PLE_EOS`, the PLE shard end marker / gen.rs filler) | `text_config.eos_token_id` |
| 17 | `eos_ids_in_vocab` | `0 <= id < 248320` (llama.cpp special-id range check) | eos ids vs `text_config.vocab_size` |
| 18 | `bos_id_in_vocab` | `0 <= id < 248320` | `generation_config.json bos_token_id` vs vocab |
| 19 | `bos_id_agrees` | text and generation configs agree | `text_config.bos_token_id` vs `generation_config.json bos_token_id` |
| 20 | `mrope_section_pairs` | `32` = `geo::ROPE_PAIRS` (the table manager.rs builds) | `text_config.rope_parameters.mrope_section` sum = 11+11+10 |

20 checks on the checkpoint of record; the boot line reports the live count.

## How robin live-tests it

1. **The green line.** Boot any engine-loading bin from the repo:
   `cd engine && ./target/release/decode badmode` (a nonsense mode: `open_model`
   runs, the usage line prints, nothing loads). The first `meta` line on stderr:
   `meta: 20 constants verified against config.json (zero numeric change) [../models/Qwen3.8-Flash-Next-original/config.json]`
   Same line appears in the serve/parity boot logs (target `meta`).
2. **The loud panic.** Corrupt a COPY of the config, point the gate at it:
   ```
   mkdir -p /tmp/crow-meta-doctored
   cp models/Qwen3.8-Flash-Next-original/{config,generation_config}.json /tmp/crow-meta-doctored/
   # doctor /tmp/crow-meta-doctored/config.json: rms_norm_eps 1e-5, head_dim 128, vocab_size 151936 …
   cd engine && CROW_MODEL_DIR=/tmp/crow-meta-doctored ./target/release/decode badmode
   ```
   → panic BEFORE the container is mapped or the GPU touched, one row per wrong
   constant (verified live 2026-09-20; a 5-field doctoring produced):
   ```
   [meta] 10 of 20 constants differ from the engine pins - refusing to boot (issue #94):
     rms_norm_eps: pinned 1e-6 (…), config 1e-5 (text_config.rms_norm_eps)
     rope_theta: pinned 10000000.0 (…), config 5000000.0 (…)
     rope_pairs: pinned 32 (…), config 16 (0.25 x head_dim 128 / 2)
     attention_scale: pinned 0.0625 (…), config 0.08838834764831843 (1/sqrt(…))
     …
   ```
3. **The package shape (WARN, not panic).** A tree without `models/` — the
   selftest download package — boots on with
   `meta: no config.json beside <container> - 0 constants verified, the pins stand unchecked (set CROW_MODEL_DIR …)`.
   Verified live by pointing `CROW_CNQ` at a bare directory.
4. **Unit tests.** `cd engine && cargo test` — 13 tests in `meta::tests`: the
   real config all-green (the phase-1 proof), both cwd conventions resolving,
   and one doctored-copy test per check name (eps, theta, partial rotary, head
   dim → scale+pairs, hidden, head counts → GQA, layer count/layout/interval,
   vocab, max positions, eos, bos, out-of-range ids, missing-field error).

## Deviations and honest gaps

- **Missing config.json is a WARN, not a refusal** — deliberate: the selftest
  package's "without the originals" control REQUIRES no `models/` dir, so a hard
  failure would break the release gate on the download package. This is the
  issue's own llama.cpp "missing fields" warning class; a PRESENT-but-deviating,
  unparseable or field-missing config is the hard-error class and panics.
- **`log.rs`'s target table was not extended** with `meta` (file owned by the
  logging discipline, not this ticket); the target works regardless — targets
  without an entry fall through to the default filter, so the line lands in the
  file log and on the stderr mirror at default INFO.
- **No BOS pin exists in the engine** (nothing in the codebase hardcodes a bos
  id), so bos is validated (range + cross-file agreement) but not compared to a
  pin.
- **`CONTEXT_FLOOR` 200_000 is NOT checked** — it is an engine policy (a floor
  under the chunk/adapt machinery), not a model constant; only the checkpoint's
  `max_position_embeddings` == the default context is asserted.
- Not covered by any pin (parsed nowhere, future phase-2 material if a site ever
  needs them): `moe_intermediate_size`, `num_experts`, `num_experts_per_tok`,
  the GDN head counts (`linear_num_key/value_heads`), `hc_count`/`hc_lowrank`,
  the PLE fields (`ngram_size`, `heads_per_ngram`, `ple_layer_ids`), `mtp.*`,
  and the vit tower constants. The CNQ trailer carries no model metadata
  (verified), so the container itself can never be this gate's source.
- Phase 2 (per the issue) migrates the constants site by site onto a runtime
  `ModelMeta` — `assert_pinned` already returns the parsed `ModelMeta` for that;
  every kernel-side scalar then stops being a compile-time pin. Not done here,
  by design: this phase buys the insurance, zero numeric change.

---

## Orchestrator verification (appended 2026-09-20, all checks passed)

1. Diffs inspected: `lib.rs` +5 lines (module registration + comment only), `boot.rs` +10 lines (one `use`, one call with comment — before `Cnq::open` and CUDA init, exactly the front-door position the issue demanded), `meta.rs` new (679 lines, 13 `#[test]`). No other file touched by this unit.
2. `cargo test --lib meta::` → **13 passed, 0 failed** (139 filtered).
3. The one failing full-suite test (`manager::tests_72::…sampler`, 291080748 vs 291080728) reproduced independently as NOT #94's: the 20-byte delta is the concurrent #83/#84 sampler-state growth; the sampler agent has been tasked with the ledger update.
4. GPU-live claims (green boot line, doctored-config panic, package-tree WARN) accepted per the agent's locked-run logs; re-run happens anyway in the wave-1 gate.
5. Honest-gap list reviewed: CNQ trailer metadata absence verified by direct trailer read; WARN-not-panic for missing config justified by the selftest package control; phase-2 scope documented.

Live-acceptance test cases for robin (in addition to the agent's recipes above): run any engine bin (e.g. `decode` smoke) and observe the `meta: 20 constants verified …` INFO line; `CROW_MODEL_DIR=/tmp/doctored-models … decode …` → loud named panic table before container/CUDA; move `models/` away (selftest-package shape) → WARN + normal continue.
