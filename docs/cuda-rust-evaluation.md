# CUDA Rust for crow-nest — evaluation, 2026-09-17

NVIDIA published *Introducing CUDA Rust: Two Tracks for Writing GPU Kernels* on 2026-09-08
(https://developer.nvidia.com/blog/introducing-cuda-rust-two-tracks-for-writing-gpu-kernels/).
This document weighs both tracks against the engine as it stands at `7ddd296`, and reports one
pilot kernel built and measured on this machine on 2026-09-17. It decides nothing about the
production path; the recommendation in section 6 is the only claim it makes about the future.

Every number below was measured on 2026-09-17 on the box of `docs/system-landscape.md`
(RTX 5090, `sm_120`, driver 610.57.04, CUDA 13.3.1, NVRTC 13.3.33, ptxas V13.3.73,
rustc 1.98.1, Arch Linux), unless a different source is named.

---

## 1 — The two tracks

| | SIMT track | Tile track |
|---|---|---|
| project | `cuda-oxide` (NVlabs) | `cutile-rs` (NVlabs), crates.io `cutile` |
| model | one thread, like CUDA C++ | one tile block over one sub-tensor |
| compiler | "a custom rustc codegen backend … routes `#[kernel]` functions through Rust MIR, the community Pliron IR framework, and LLVM IR down to PTX" | "`#[cutile::module]` … embeds the kernel's AST in the host binary and JIT-compiles it through CUDA Tile IR … when the kernel is first needed" |
| toolchain | "Linux, a GPU with compute capability 8.0 or later, a CUDA toolkit (12.x or newer), clang with its libclang headers, and the pinned nightly toolchain"; the install line is `cargo +nightly-2026-04-03 install --git … cargo-oxide` (blog, *The SIMT track*) | "stable Rust 1.89 or newer, and Linux, but no nightly toolchain and no LLVM of your own" (blog, *The Tile track*) |
| CUDA | 12.x or newer | 13.2 for `sm_8x`/`sm_100+`, 13.3 for `sm_90`; 13.3 recommended, "FP4 packing and block-scaled MMA require 13.3" (repo README, Setup/Requirements) |
| GPUs | compute capability 8.0+ | compute capability 8.0+; below `sm_80` unsupported |
| safety | checked per launch call | ownership follows the tensors across the launch boundary — "the stronger of the two claims" (blog, *What the compiler catches*) |
| maturity | "cuda-oxide is early alpha" | "further along, published on crates.io and already used outside NVIDIA in HuggingFace's Grout inference engine and in mistral.rs" (both blog, *Where the projects stand*) |

NVIDIA's own verdict, verbatim: **"Both projects are early-stage and neither is
production-ready. … Coverage is incomplete and APIs will move."** (blog, *Where the projects stand*). The
`cutile-rs` README repeats it: "The software is in an early stage and under active development:
you should expect bugs, incomplete features, and API breakage as we work to improve it."

Only the Tile track was evaluated further. A pinned nightly and a private LLVM are disqualifying
for an engine whose build is a gate.

## 2 — What crow-nest does today

- One CUDA C++ translation unit, `KERNEL_SRC` (`engine/src/kernels.rs:15`, 4,464 lines),
  NVRTC-compiled once per process at `engine/src/gen.rs:990` for `compute_120a` — plain `sm_120`
  is rejected by ptxas (`docs/architecture.md:351-352`).
- **110 kernels registered** in the launched set (`engine/src/kernels.rs:4509`); six more are
  defined and have no launch site left. Every launch goes through one shim,
  `kernels::launch_v` (`engine/src/kernels.rs:4593`), with scalar arguments passed as device
  buffers only.
- The NVFP4 GEMV/GEMM forms use `mma.sync…kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64`
  with block-16 `ue4m3` scales, a **fixed k order**, a residual level cascade and a **fixed KS
  smem slice reduce order** (`engine/src/kernels.rs:787`).
- The parity contract is byte-identity of the logit dump, not a tolerance: 8 rows
  `bceba6ff7724…`, 512 rows `8387234709271515…`, P8 teacher-forced `3bb3e69edf90…`
  (the Linux values of record, commit `9f12429`).
- The toolchain, not the engine, already moves those bytes. The Linux port (`9f12429`) measured
  the 8-row form byte-identical to Windows and the 512-row form **bit-identical for rows 0-22,
  drifting from row 23** (max |d| 7.0) with **all 517 ids identical**, deterministic across four
  configurations. The port surface was exonerated: the difference is
  **Windows NVRTC 13.3.73 + driver 616.56 JIT against Linux NVRTC 13.3.33 + driver 610.57 JIT**.
- What that costs at boot, measured here with `CUDA_CACHE_DISABLE=1` (no driver cache):
  **NVRTC source → PTX 7.91 s, driver JIT PTX → SASS 16.79 s, 24.70 s total**. With
  `~/.nv/ComputeCache` warm (21 MiB on this box) the same two steps are **0.06 s + 0.01 s**.
  The boot compile is therefore a once-per-toolchain cost, not a once-per-start cost.

## 3 — Kernel families against cuTile

"Bit-identical" below means: the same f32 bits as the current kernel, which only a
reduction-free kernel whose every operation is IEEE-exact in both forms can promise. A
reduction gives the Tile compiler the association order, and the engine's contract is the
order, not the value.

| family (kernels.rs) | expressible in cuTile 0.3.1 | closest example | bit-identical? | port cost |
|---|---|---|---|---|
| NVFP4 GEMV mma (`gemv_fp4_mma*`, `gemv_fp4*`, `quant_x_fp4`) | yes in principle — `mmaf_scaled`, `f4e2m1fnx2`, `f8e4m3fn` are all in 0.3.1 | `nvfp4.rs` (`linear_tile`, block-16 e4m3 scales) | **no** — Tile IR owns the k loop and the accumulation; the residual cascade and the KS reduce order are not expressible as such | new values of record for 8/512/1024 rows + P8 + layercheck + ten-task |
| NVFP4 dense GEMM (`gemm_fp4_dense*`, `gemm_fp4_tiles`, `gemm_fp4_f32x`) | yes | `gemm.rs`, `persistent_gemm.rs`, `nvfp4.rs` | no (reduction) | same battery |
| bf16 dense (`gemv_bf16*`, `gemm_bf16_dense*`) | yes | `gemm.rs`, `gemm_static.rs` | no (reduction) | same battery |
| attention prompt/step/sel (`attn_sel*`, `attn_sel_split`, `attn_merge`, `vit_attn`) | yes | `flash_attention.rs`, `flash_attention_causal.rs` | no — the split count and the merge order are the contract (`CROW_ATTN_SPLITS`, default 8) | same battery + the split/merge knobs re-pinned |
| GDN prompt/step (`delta_rule_persist*`, `delta_rule_step*`, `conv_*`, `l2norm_repeat`, `beta_g`, `transpose_rt`) | partly — the recurrent state loop has no example, and the persistent form wants device globals, which 0.4.0 is still changing (`Global` became a breaking API change) | none | no | same battery; the highest-risk family |
| PLE (`gather_ple_fp4`, `ple_conv`, `ple_state_update`, `ple_conv_step`) | yes | `embeddings.rs` (`embedding_batch_f16`) | the **gather** yes (reduction-free); the 3-tap conv only if the summation order is written out | gather alone: P8 + 8 rows |
| QSA scores/select (`qsa_scores*`, `qsa_select*`, `router_top10`) | scores yes, select yes | `argmax.rs`, `softmax.rs` | no — scores reduce; select's tie-break is the contract | same battery |
| sampler top-k (`argmax_k`, `sample_topk_part`, `sample_k`) | yes | `argmax.rs` (two-stage block reduce) | no — the partial/final split and the tie-break decide the id | full battery; ids are the output |
| rmsnorm / rope (`rms_group`, `rmsnorm_1pw`, `rms128`, `rmsnorm_gated*`, `rope*`) | yes | `rms_norm.rs`, `positional.rs` (`rope_seq_f16`) | no for rmsnorm (row reduction); rope is reduction-free but needs `sin`/`cos` to match `__sincosf` | 8 + 512 rows |
| elementwise maps (`add_flat`, `mix_streams`, `inject_residual`, `gate_mul`, `silu_*`, `sigmoid_el`, `vit_add_bias`, `cast/dec_e4m3_flat`, `gelu_tanh`) | yes | `pointwise.rs`, and the pilot below | **yes for the exact ones** (add, mul, select); *measured* for `gelu_tanh` in section 5 | the 8-row form is enough for a map that measures bit-identical |
| `gelu_erf` | **no** | — | — | cuTile 0.3.1 has no `erf`/`erfc` op at all — its whole float op set is `absf addf ceil cos cosh divf exp exp2 floor fma log log2 maxf minf mulf negf pow rsqrt sin sinh sqrt subf tan tanh`, plus `reduce`/`scan`/`select`/`mma*` (`cutile-0.3.1/src/_core.rs`). A hand-rolled polynomial would not be bit-identical. |
| ViT tower (`vit_ln`, `vit_rope`, `vit_pe_add`, `vit_attn`, `gemm_fp4_f32x`) | mixed, as the rows above | `rms_norm.rs`, `flash_attention.rs` | per row | the vision smoke plus the 8-row form |
| staging, residency, MoE plumbing (`stage_cold*`, `stage_tiles*`, `moe_count/plan/scatter`, `expand_slab`, `swap_pairs`, `d2d_block`, `store_kv`, `qk_k_append`, `pool4_cache`) | **wrong tool** — these are byte movement and index arithmetic, not tile math | — | n/a | do not port |

Counting the table: of 110 registered kernels, the reduction-free maps are the only ones that can
enter the engine without a new value of record. Everything that carries the model's arithmetic
needs the full gate battery **and** a decision to re-baseline the parity contract.

## 4 — Honest weighing, for this engine

**What cuTile would actually buy**

- *A boot step removed.* The NVRTC + driver-JIT cost is 24.70 s cold, 0.07 s warm (section 2).
  cuTile's own JIT does not remove it — it replaces it (section 5: 82.4 ms for one kernel, and
  every kernel pays separately unless the disk cache is enabled). The real AOT win would be
  precompiled cubins, which cuTile supports as a compile-only API, not as the default path.
- *Host and device share one type system.* `assert_kernel_defines()` (`gen.rs:992`) exists
  because four `#define`s in the frozen CUDA text have Rust twins that nothing else checks; cut 1
  turned that into a boot assertion by parsing the string (`kernels.rs:4489`). In cuTile those
  four are ordinary Rust `const`s and the assertion is unnecessary. That is a real, small win.
- *Aliasing checked at compile time.* The engine passes every kernel argument as a bare
  `CUdeviceptr` through `launch_v(f, …, &[u64])` — 110 kernels' worth of positional `u64`s with
  no type on them at all. cuTile's `Partition`/`Arc<Tensor>` split makes the output-aliases-input
  mistake a compile error. This is the strongest argument in the list, and it is about the host
  code, not the math.
- *Architecture portability.* The engine is `compute_120a` only; cuTile infers the target from
  the device and `tileiras` accepts `sm_80 … sm_121`. Relevant only if the Ampere/Ada stage
  (`docs/architecture.md:478`) is ever opened.
- *Versioned codegen against driver-JIT drift.* This is the argument the Windows/Linux drift
  invites, and it does **not** survive contact: cuTile's cache key already contains
  `compiler_version` and `tileiras_fingerprint` (the cuTile JIT guide, "Kernel Cache Key"), which
  is an admission that a toolkit upgrade changes the code it generates too. A Tile IR port trades
  one versioned code generator for another; it does not make the bytes stable.

**What it costs**

- *The parity contract.* Byte-identity is the engine's correctness definition. Every family
  except the reduction-free maps would need new values of record, and a re-baseline is a decision
  robin makes, not a refactor.
- *Two driver bindings in one process.* cuTile ships its own `cuda-bindings`/`cuda-core` and does
  not use cudarc. Both `dlopen` `libcuda.so.1`; the pilot proves they coexist and share the
  primary context. It is still two binding crates to version and two error types to translate.
- *Maturity.* NVIDIA says not production-ready (section 1), the 0.3.1 changelog lists eleven
  breaking API changes in one point release, and `main` is already 0.4.0 with more.
- *No speed win by itself.* Measured: 2.76 µs per launch, both forms, same stream (section 5).
- *Build cost.* 71 extra crates including `bindgen` and `clang-sys` (section 5).

## 5 — The pilot

`engine/src/cutile_pilot.rs` (274 lines), behind the default-off cargo feature `cutile-pilot`
(`engine/Cargo.toml`, `cutile = { version = "=0.3.1", optional = true }`). It is an evaluation
artefact and its header says so; nothing in the engine calls it and `KERNEL_SRC` is untouched.

**Kernel chosen: `gelu_tanh`** (`kernels.rs:4473`), the vision-MLP activation launched at
`vit.rs:394`. It is the reduction-free kernel with the simplest signature that the engine really
launches on a plain f32 buffer — `(float* x, const int* n_p)`, in place, one output per element.
`gelu_erf` (`kernels.rs:4464`) has the same signature and was the first choice, but cuTile 0.3.1
has no `erf` op (section 3). `add_flat` and `vit_add_bias` are simpler still but exercise nothing
beyond an add. The pilot slices the CUDA text for `gelu_tanh` **out of `KERNEL_SRC` at runtime**,
so it NVRTC-compiles the production source rather than a copy of it.

**Interoperation.** The engine allocates with `cuda::alloc_zeroed` (`cuda.rs:336`), the pilot
wraps the raw `CUdeviceptr` in an `EngineBuffer` implementing cuTile's `DeviceAllocation`
(pointer, byte length, ordinal — nothing else) and hands it to `Tensor::from_foreign`. The
wrapper frees nothing; the engine frees. The engine's primary context and its non-blocking stream
are borrowed with `Device::borrow_raw` and `Stream::borrow_raw`, so both kernels run on the same
stream: `cuda::set_stream(s)` makes `launch_v` use it and `.sync_on(&st)` / `.execute(&ctx)` make
cuTile use it.

**Measured** (`cargo test --release --features cutile-pilot -- --nocapture`, n = 1,048,576 f32,
tile 256, input = 16 edge bit patterns (±0, denormals, `f32::MAX/MIN`, ±inf, quiet and signalling
NaN, ±1) then a xorshift32 stream alternating raw f32 bit patterns with a [-12, 12] sweep):

| quantity | value |
|---|---|
| NVRTC compile, whole `KERNEL_SRC`, cold | 7.91 s + 16.79 s driver JIT |
| NVRTC compile, whole `KERNEL_SRC`, warm cache | 0.06 s + 0.01 s |
| NVRTC compile, `gelu_tanh` alone | 5.7 ms |
| cuTile first launch (JIT + launch) | 82.4 ms — frontend 12.0 ms, `tileiras` subprocess 51.3 ms, module load 3.3 ms |
| per launch, 1,000 launches, one stream, cuEvent-timed | NVRTC **2.76 µs**, cuTile **2.76 µs** |
| bit-equal outputs | **1,047,516 of 1,048,576** (99.899 %) |
| NaN against NaN with a different payload | 0 |
| differing | 1,060; ulp histogram 1 / 2 / 3-4 / 5-16 / >16 = 328 / 93 / 56 / 374 / 209 |
| largest difference | 33,135 ulp — but **2.38e-7 in absolute terms** |
| against an f64 evaluation of the same f32 constants | NVRTC 16,570 ulp, cuTile 16,565 ulp |

**So: not bit-identical, and the ulp table is the honest outcome.** The reading is in the last
two rows. Every differing element sits in the negative tail, where GELU evaluates
`0.5·v·(1 + tanh(z))` with `tanh(z) → -1`: the sum cancels to a number of order 1e-5, so one ulp
of disagreement inside `tanh` comes out as thousands of ulp in the result. The largest absolute
difference over the whole million elements is 2.38e-7, and measured against an f64 reference
neither form is the more accurate one (16,570 against 16,565 ulp). The two `tanh`
implementations — NVRTC's `tanhf` and Tile IR's `cuda_tile.tanh` — simply round differently, and
cuTile's own bytecode notes call out a `tanh` rounding attribute among the field
layouts that were only pinned at bytecode 13.2 (`cutile-ir/src/bytecode/enums.rs`). A kernel of exact operations only (`add_flat`, `vit_add_bias`) would be
bit-identical; a kernel with a transcendental is not, and no flag makes it so.

**Build cost of the feature.** Clean builds, separate `CARGO_TARGET_DIR`, same box:

| | feature off | feature on | delta |
|---|---|---|---|
| wall | 11.79 s | 19.22 s | +63 % |
| CPU | 2 m 34 s | 4 m 46 s | +86 % |
| target dir | 389 MiB | 923 MiB | +534 MiB |
| crates in the graph | 109 | 180 | +71 (`bindgen`, `clang-sys`, `futures`, `uuid`, …) |
| clippy, `--all-targets` | 1422 | 1422 | 0 (the pilot file raises none) |
| `cargo test --release` | 144 passed | 145 passed | the pilot test |

**What the JIT needs on this box.** This took the longest and is the most reusable finding.

- Build time: `cuda-bindings` needs `CUDA_TOOLKIT_PATH` or `CUDA_HOME` pointing at a toolkit root
  with `include/cuda.h` **and `include/curand.h`** (its `wrapper.h` includes both), plus
  `libclang` for `bindgen`. Crow's toolkit at `~/.local/share/crow/cuda` has no cuRAND headers;
  they come from the `libcurand` 10.4.3.29 redistributable.
- Run time: the Tile JIT shells out to **`tileiras`**, the CUDA Tile IR assembler, at
  `<toolkit>/bin/tileiras`. Crow's toolkit does not ship it; it is the `cuda_tileiras` 13.3.36
  redistributable. `libnvJitLink` is present and is **not** what the Tile JIT uses.
- **`tileiras` resolves `ptxas` and `nvvm/lib64/libnvvm.so` relative to its own executable path.**
  A symlink at `<toolkit>/bin/tileiras` resolves to the real file's directory, those two are not
  found, and every compile fails with `error: failed to compile Tile IR program` (exit status 5)
  — on every `--gpu-name`, every `--opt-level` and every bytecode version, with no further
  diagnostic, including for NVIDIA's own `saxpy` example from `NVlabs/cutile-rs@main`. **Copy the
  binary into the toolkit's `bin/`, do not symlink it.** With that one change the JIT works.
- `tileiras` 13.2.86 rejects `sm_120` ("invalid GPU architecture: 120"); 13.3.36 and 13.4.92
  accept it. `tileiras --list-versions` on 13.3.36 reports bytecode 13.1/13.2/13.3.
- cuTile needs no `LD_LIBRARY_PATH` — it `dlopen`s `libcuda.so.1` itself, like cudarc (checked:
  a cuTile-only binary runs with the variable unset). It does need `CUDA_TOOLKIT_PATH` or
  `CUDA_HOME` **at run time**, to find `tileiras`; the pilot test needs `LD_LIBRARY_PATH` only
  for its NVRTC half.

**Gates, with the feature off** (`7ddd296` plus this change): `cargo build --release` green,
`cargo test --release` **144 passed**, `cargo clippy --release --all-targets` **1422**, and the
8-row parity form inside the memory-bounded scope **`bceba6ff772431de…a122a2`, 11,919,360 B** —
the value of record. `tools/check_env_docs.py` exits 0 (80 = 80), `tools/check_readme_dates.py`
reports 0 offenders. The release binaries are **not** byte-identical to `7ddd296`
(`decode` `ca14924c…` → `9a0bc8ff…`): a `[features]` section changes the crate's metadata hash
and therefore its symbol mangling. The parity bytes are what the contract is about, and they
did not move.

Every count in this document is the count at `0667e0b`, the commit it was written in. They moved later the same day and the moves are not this document's subject: `cargo test --release` went 144 → 147 (TASK H) → 153 (TASK J) → **165** (TASK K, 98 lib + 67 serve), so the feature-on arm is **166**; `tools/check_env_docs.py` reads **82 = 82** at `487128d` instead of 80 = 80. Clippy is unchanged at **1422** at `487128d`, and the 8-row parity value of record is unchanged (`bceba6ff772431de…a122a2`, 11,919,360 B). Nothing in the pilot measurements above was re-run.

## 6 — Recommendation

**Do not port anything to cuTile now.** NVIDIA says neither track is production-ready, the pilot
shows no speed to gain (2.76 µs against 2.76 µs) and the only family that can move without a new
parity baseline is the elementwise maps, which are already the cheapest kernels in the engine.
Keep the pilot as the standing measurement it is, revisit when NVIDIA calls the Tile track
production-ready and the crates reach a version that does not break eleven APIs in a point release.

**If it is ever revisited, in this order.** (a) The elementwise maps of exact operations first —
`add_flat`, `vit_add_bias`, `inject_residual`, `mix_streams`: they can be proven bit-identical
element for element, so the 8-row form is the whole gate. (b) The PLE gather, also
reduction-free. (c) Nothing else until a re-baseline of the parity contract is a decision robin
has taken, because every remaining family needs new values of record for 8, 512 and 1024 rows,
P8 teacher-forced, `layercheck` and the ten-task gate — and mixing a re-baseline with a port
means neither can be blamed when the bytes move.

**Do not do**

- Do not port a reduction to cuTile and claim identity. The Tile compiler owns the association
  order; the engine's contract *is* the order.
- Do not port a kernel with a transcendental and expect bit-equality. Measured above: `tanh`
  alone costs 1,060 of 1,048,576 elements, and `erf` is not implemented at all.
- Do not put the `tileiras` subprocess on the engine's boot path. 51.3 ms per kernel
  specialization against 0.07 s for all 110 kernels once the driver cache is warm.
- Do not let cuTile allocate the engine's memory. The residency planner owns every byte of VRAM
  and of the pinned host tier; `DeviceAllocation` + `from_foreign` is the only correct direction.
- Do not enable `cutile-pilot` in CI or in any measurement chain. It adds 71 crates, 534 MiB and
  a toolkit requirement that no other part of the build has.
- Do not touch `KERNEL_SRC` for an experiment. The pilot reads its CUDA out of the frozen string
  at runtime precisely so that the experiment cannot change it.
