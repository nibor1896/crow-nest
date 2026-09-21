# Issue #96 — YaRN/NTK rope scaling + the exceed-training-context warning

Built 2026-09-20. Scope, per the issue's four points: (1) the boot table
builder extended with YaRN (corr dims from beta_fast/beta_slow, low/high-
frequency mixing, the mscale attention factor threaded to the softmax scale),
(2) config-driven through the `rope_scaling` object of the checkpoint config —
absent (the checkpoint of record) = byte-identical behavior, gate-proven,
(3) the CHEAP warning first: one loud boot WARN when the effective context
exceeds the original training context and no scaling is armed, and (4) no
validation tonight — that rides on the #90 long-context instrument when the
machine is back (this session ran NO engine binary; unit tests only).

## What was built

- **`engine/src/meta.rs`** — the config side. `RopeKind` (default / linear /
  yarn / ntk-aware) + `RopeScaling` (factor, original_context with the
  llama.cpp `n_ctx_orig_yarn` fallback, beta_fast/beta_slow with the llama.cpp
  defaults 32/1, optional `attention_factor`) parsed from
  `text_config.rope_scaling` (the legacy `"type"` spelling accepted, `rope_type`
  preferred). An unsupported type (longrope's factor tables, dynamic, su-rope)
  or a missing `rope_type`/`factor` is a NAMED parse error — the #94
  hard-error class, never a silently ignored object. New check row `rope_type`:
  `text_config.rope_parameters.rope_type` must say `"default"` (it does, on the
  checkpoint of record; a different value there is a checkpoint announcing
  scaling through a channel nothing reads). The green `assert_pinned` path
  stashes the scaling + the training context in two `OnceLock`s
  (`meta::boot_rope_scaling()` / `meta::boot_training_context()`) — the seam
  that hands the truth to the two boot-time readers whose call sites this issue
  does not own (`ThreeStates::allocate` in gen.rs, `Kernels::new` in gen.rs).
  `ModelMeta::training_context()` = `rope_scaling.original_context` when the
  object carries one, else `max_position_embeddings`.
- **`engine/src/manager.rs`** — `build_rope_table(context, scaling)`, the boot
  table factored out as pure host f32 math (no GPU, no container), plus
  `exceed_training_warning(context, training, scaling)` as a pure function.
  `allocate` reads the stash, fires the warn, builds the table through the
  builder, and prints one `[rope] rope_scaling armed — …` report line when a
  non-default scaling is in force (silent otherwise; the `RoPE tbl` line keeps
  its exact bytes).
- **`engine/src/kernels.rs`** — the mscale threading. A `__device__ float
  d_attn_scale = 0.0625f` global + `attn_scale_src<RT>()` (a template
  specialization: RT=0 returns the compile-time literal, RT=1 reads the global)
  + a one-boot-write `set_attn_scale` kernel (the p5 rule: the scalar lives in
  a device buffer). The five attention-scale sites (attn_sel, attn_sel_r_body,
  attn_sel_s_body, g_score, attn_sel_split_body) became `RT`-templated bodies;
  every existing wrapper keeps `RT=0`, and ten `_y` twins carry `RT=1`.
  `Kernels::new`, when `boot_rope_scaling()` has an mscale != 1, uploads
  `0.0625 · mscale` once and swaps the ten canonical names in its function map
  to the `_y` handles — gen.rs, every launch site, every signature, every
  graph capture unchanged.
- No new env knob (nothing here reads the environment), so `docs/env.md` gains
  no row; its `CROW_MODEL_DIR` note was touched only to re-count the gate's
  constants (20 → 21, the `rope_type` row). `geo.rs`, `boot.rs`, `gen.rs`,
  `sample.rs`, `serve.rs`, `toolcall.rs` untouched.

## The reference math (llama.cpp provenance, as the issue prescribes)

- corr dims — `ggml.c ggml_rope_yarn_corr_dims`:
  `dim·ln(ctx / (beta·2π)) / (2·ln base)`, floor at beta_fast, ceil at
  beta_slow, clamped `[0, dim-1]`. For this checkpoint (dim 64, base 1e7,
  ctx 262144, beta 32/1): **[14, 22]** — pairs ≤ 14 extrapolate, ≥ 22
  interpolate, between ramps. The bounds are in DIM units against a PAIR
  index, exactly as llama.cpp and HF ship it.
- the blend — `ops.cpp rope_yarn` with `ext_factor` pinned at the llama.cpp
  yarn default 1.0: `theta = interp·(1−mix) + extrap·mix`,
  `interp = freq_scale·extrap`, `mix = rope_yarn_ramp(pair)` =
  `1 − clamp((pair−lo)/max(0.001, hi−lo), 0, 1)`.
- the mscale — `get_mscale(factor, 1.0)` = `1 + 0.1·ln(1/freq_scale)` =
  `1 + 0.1·ln(factor)` (0.0625-side multiplier; factor 4 → 1.13863), times the
  optional HF `attention_factor`. Applied on the ATTENTION side per the issue
  text ("threaded to the softmax scale"). NOTE: current llama.cpp instead bakes
  mscale into the rope cache and cancels it on `yarn_attn_factor`
  (`llama-context.cpp`, the 2025 rework) — net default identity here would be
  the same logits-by-construction argument does NOT transfer, so this engine
  follows the issue's attention-side form, which is also the HF/YaRN-paper
  form. A one-line change in `Kernels::new` if that is ever revisited.
- NTK-aware — the theta rewrite the llama.cpp converter bakes into the base:
  `base·factor^(dim/(dim−2))`; linear — `freq_scale` on every pair, no mscale.

## The byte-identity contract and its gates

With `rope_scaling` absent (the checkpoint of record):

1. **Table bytes** — `manager::tests_96::the_unscaled_table_is_byte_identical_to_the_loop`
   builds the table through the new builder and through the pre-#96 loop
   (copied verbatim into the test as the oracle) and asserts `Vec` equality on
   cos and sin; a present-but-`"default"` rope_scaling object is asserted to
   move nothing either.
2. **Kernel scale constant-folded** — the RT=0 template specialization returns
   the literal, so the kernels of record compile to the same folded multiply;
   `kernels::tests_96::the_source_compiles_and_the_scale_split_is_visible_in_the_ptx`
   NVRTC-compiles the frozen KERNEL_SRC host-side (no context, no GPU, no
   pinned RAM) and asserts in the PTX that `attn_sel`/`attn_sel_r`/`attn_sel_s8l`/`
   attn_sel_g`/`attn_sel_split_l` carry the `0f3E000000` (0.0625f) immediate
   and never reference `d_attn_scale`, while the `_y` twins load it.
3. **Launches and logits** — by construction: no upload, no map substitution
   and no new kernel runs unarmed; every launch resolves the same function
   with the same arguments; a multiply by the same f32 0.0625 (a power of two —
   exact) yields identical bits whether immediate or constant-bank operand.

Plus the #94 gate itself stays green: the checkpoint of record parses
scaling-free, `rope_type` "default", 21/21 checks.

## How robin tests it tonight (unit tests only — machine blocked)

1. `cd engine && cargo test tests_96` — the full set: the byte-identity gate,
   the YaRN shape test (pairs ≤ 14 byte-equal to the unscaled table, pairs
   ≥ 22 the interpolated angle, the ramp midpoint the half-and-half angle,
   corr range [14, 22]), linear-scales-every-pair + NTK-base-rewrite, the
   warn's truth table, and the PTX fold/load split. Meta-side:
   `cargo test meta::` — the 13 #94 tests still green plus 6 new (real config
   scaling-free, doctored rope_type fires by name, yarn parses with the
   reference math, legacy `"type"` spelling + defaults, linear/ntk/no-mscale,
   longrope + missing-factor named errors).
2. **Boot warn at oversized context (live, when the machine is back).** The
   warn fires from `ThreeStates::allocate` whenever `cfg.context` exceeds the
   parsed training context with nothing armed. Today every bin boots
   `CONTEXT_FLOOR` 200_000 < 262_144, so the line is dormant by design — the
   honest trigger is any future context raise above 262_144 (a `states`-bin
   shape or a one-line boot change; this issue owned neither). The decision
   function is fully pinned by
   `manager::tests_96::the_exceed_training_warning_fires_only_unscaled_and_oversized`:
   fires at 300_000/262_144/unscaled with the words "quality WILL be
   degraded"; silent at 200_000, at equality, under armed yarn, and when no
   config was seen at all (the selftest package).
3. **Config-driven YaRN smoke (live, when the machine is back).** Doctor a
   COPY, never the original:
   ```
   mkdir -p /tmp/crow-yarn-smoke
   cp models/Qwen3.8-Flash-Next-original/{config,generation_config}.json /tmp/crow-yarn-smoke/
   # add to text_config of /tmp/crow-yarn-smoke/config.json:
   #   "rope_scaling": { "rope_type": "yarn", "factor": 4.0,
   #                     "original_max_position_embeddings": 262144 }
   cd engine && CROW_MODEL_DIR=/tmp/crow-yarn-smoke ./target/release/decode <mode>
   ```
   The #94 gate stays green (rope_scaling pins nothing), the boot log carries
   `[rope] rope_scaling armed — YaRN: factor 4 (262144 training positions ->
   …), corr dims [14, 22] of 64 (beta 32 / 1), mscale 1.1386 on the attention
   scale` next to the `RoPE tbl` line, and `[kernels] #96 attention scale
   0.0625 -> 0.070941 …` from the arm. Remove the object again and both lines
   vanish — the default boot is the byte-identical one.

## Deviations and honest gaps

- **No engine binary ran tonight** (pinned-RAM hard leak, machine blocked):
  every claim above is unit-test proven; the two live recipes wait for the
  machine, and the KLD-vs-position validation rides on #90's instrument, as
  the issue itself prescribes.
- **The warn is dormant at today's boot shapes** — effective context 200_000
  never exceeds 262_144. That is correct behavior (nothing is being
  overflowed), and the wiring is in place for whichever change next raises a
  context past the training window.
- **The boot table is shared**: `rope`/`rope_p` (text attention) AND `rope64`
  (the QSA indexer keys) read the same cos/sin rows, so a scaled table scales
  both. The reference behavior of the indexer under YaRN is unverified for
  this architecture (llama.cpp has no QSA); flagged for the #90 validation to
  answer, not silently assumed either way.
- **`ext_factor` is pinned at the llama.cpp yarn default 1.0** (not
  configurable tonight); the corr bounds keep the llama.cpp dim-units-vs-pair-
  index convention deliberately.
- **llama.cpp's mscale-in-rope-cache form is NOT followed** (see the math
  section) — the issue's own text says attention side, and that is what is
  implemented.
- **The vit tower keeps its own tables** (vit.rs builds its own rope) — long
  context is a text-path concern; the tower never sees 200k positions.
- **longrope / dynamic / su-rope refuse the boot by name** rather than
  degrading to none — implementing them is future work with its own factor
  tables.
- A real yarn checkpoint would still die at OTHER #94 pins (its
  `max_position_embeddings` would be the original training context, not
  262_144) — porting such a checkpoint stays a conscious act; the mechanism
  and its tests are in place for when that happens.
- Phase 2 of #94 (constants migrating onto a runtime `ModelMeta`) will
  eventually make the attention scale a plain runtime scalar (#T12); the `_y`
  twin + device-global here is the conservative bridge that keeps tonight's
  default path literally constant-folded.
