---
license: other
license_name: qwen-community-1.0
license_link: LICENSE
base_model: Qwen/Qwen3.8-Flash-Next
base_model_relation: quantized
pipeline_tag: text-generation
language:
- en
- de
tags:
- nvfp4
- cnq
- quantized
- moe
- vision
- multimodal
- windows
- linux
- rtx-5090
- crow-nest
---

# Qwen3.8-Flash-Next-CNQ4.5-M

Qwen3.8-Flash-Next quantized to CNQ4.5-M: one NVFP4 container at 4.5 bpw with a BF16 keep set and MSE sub-block scales (`--scales mse`), weights for the crow-nest engine on Linux and Windows.

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-Flash-Next`, revision of record `de4b8e4d43b917e7706784d8bb445c9af86a3540` (see Provenance) |
| quantization | CNQ4.5-M: NVFP4 at 4.5 bpw, round to nearest, calibration free; BF16 keeps for embeddings, `lm_head`, router, shared expert gate, all norms and every 1-D tensor; sub-block scales by `--scales mse` |
| container | one file, `Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, 104,727,179,972 B (about 105 GB) |
| engine | crow-nest, `https://github.com/nibor1896/crow-nest`: own HTTP server, own container format, no GGUF, no transformers |
| vision | the container carries the FULL vision tower, same quant policy as the text tower (see Vision) |
| platform | Linux and Windows (crow-nest v0.3.0), CUDA, NVIDIA Blackwell (`sm_120`); every measured number on this card comes from one RTX 5090 |
| licence | model weights: Qwen Community License 1.0 (see License); engine and converter code: Apache-2.0 |

## Files

| file | size | `sha256` | purpose |
|---|---|---|---|
| `Qwen3.8-Flash-Next-CNQ4.5-M.cnq` | 104,727,179,972 B | `7c058e555667b1c3d8f7d804d4a3393ba3664161dd307305718d85e4f33646b7` | the container |
| `Qwen3.8-Flash-Next-CNQ4.5-M.cnq.sidecar.jsonl` | 436,716 B | `af85905fbf46c12fa9528d22f2932b8785c55731a8c8fb013b2e91eecc6f0692` | verification sidecar: one JSON line per tensor with max and mean error against the sub-block scales, gate 0 of the measurement ladder |
| `hotsets-M-longctx2100-n160.json` | 83,705 B | `4a408907d553518ee4421e59eae09e243db82ac0cd677557596dfdbaa4b099dc` | hot-set manifest the engine loads so adapted experts stay resident; this is the manifest to use, not a sidecar derived from the container name (engine issue #49) |
| `selftest/layer0-input.f32` | 327,680 B | `65907fe567547704dd644eea9212c4c71b0b02f2fd954f23ecd28c7f5804d974` | self-test input: the `[8][10240]` f32 layer-0 activation the golden below belongs to (see Self-test) |
| `selftest/layer0-golden-output.f32` | 327,680 B | `e98292a0413d7aaf6825ab23e80cfae52f2424b5524f835624e9d76e90d7c705` | self-test golden: the `[8][10240]` f32 output of the UNQUANTIZED layer 0 on that input |
| `selftest/manifest.json` | 3,638 B | `9fb2d5d5b270d1976c17da5b63b4edae226a348611e6b651f4c8a8a732f13388` | the self-test's check list, its gate, the two sums above and the goldens' full provenance |
| `SHA256SUMS` | 589 B | n/a, it carries the sums | the six sums above; the first three were each read twice when the package was formed (F1, engine issue #57), the self-test lines were added with the self-test (F5, engine issue #64) |
| `LICENSE` | 3,235 B | `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08` | Qwen Community License 1.0, verbatim from the upstream revision |

- Verify the download with the first command below, in the package directory; it must report 6 of 6 OK and covers the container, the sidecar, the hot-set manifest and the three self-test files. The second command checks the self-test files alone, which costs no read of the 105 GB container.

```
sha256sum -c SHA256SUMS
grep ' selftest/' SHA256SUMS | sha256sum -c -
```

## Format

- The container is the CNQ v1 format written by the `converter` in the crow-nest repository (`converter/src/main.rs`, module comment of record `main.rs:1-59`). That converter reproduces this package from the original safetensors; there is no second format description, only this summary and the code.
- Spec section 1 of the engine, approved 2026-09-02. The tables below mirror `converter/README.md` of the engine repo.

### Container layout

| range | content | source |
|---|---|---|
| `[0..4)` | magic `CNQ1` | `main.rs:53`, spec section 1 approved 2026-09-02 |
| `[4..12)` | reserved, zeros | `main.rs:54`, spec section 1 approved 2026-09-02 |
| `[12..)` | payload blob, streamed | `main.rs:55`, spec section 1 approved 2026-09-02 |
| `[end-8-index_len .. end-8)` | index JSON, UTF-8, a TRAILER | `main.rs:56`, spec section 1 approved 2026-09-02 |
| `[end-8 .. end)` | `u64` little endian `index_len` | `main.rs:57`, spec section 1 approved 2026-09-02 |

- The index is a trailer, not a header: the payload streams to disk without knowing the index size up front, so the model is never held in RAM (`main.rs:5-7`).
- Tensor offsets in the index are relative to blob start (`main.rs:58`).

### NVFP4 geometry

| quantity | value | source |
|---|---|---|
| values per block | 64 | `main.rs:9`, spec section 1 approved 2026-09-02 |
| sub-block scales per block | 4, `ue4m3`, one per 16-wide k-block | `main.rs:9-10`, spec section 1 approved 2026-09-02 |
| packed `E2M1` payload per block | 32 B | `main.rs:10`, spec section 1 approved 2026-09-02 |
| stored bytes per block | 36 B | `main.rs:10`, spec section 1 approved 2026-09-02 |
| effective width | 4.5 bpw | `main.rs:10`, spec section 1 approved 2026-09-02 |
| second level | one global `f32` scale per tensor, two level scaling | `main.rs:11`, spec section 1 approved 2026-09-02 |
| calibration | none, round to nearest | `main.rs:11`, spec section 1 approved 2026-09-02 |

### BF16 keep set

- Approved as spec 1.2 on 2026-09-02 (`main.rs:13-17`).
- Embeddings and `lm_head`, the router GEMM, `shared_expert_gate`, all norms.
- Every 1-D tensor rides along: biases, `A_log`, `dt_bias`, gates. Under 0.1 % of the bytes, and a 1-D recurrence parameter must not go through a GEMM quant path.
- Any tensor whose length is not a multiple of `64` stays BF16 (`main.rs:17`).

### Section marks

- Decided 2026-09-02 (`main.rs:19-21`).

| mark | content | load |
|---|---|---|
| `text` | the default section | always |
| `ple` | `ngram_embedding`, NVFP4, block exchangeable to FP8 | always |
| `vit` | `model.visual` | carried in the container, optional to load |
| `mtp` | carried in the container | optional to load |

### Verification sidecar

- File name `<out>.cnq.sidecar.jsonl`: one JSON line per tensor with max and mean error against the sub-block scales, computed during quantization by dequantizing in place (`main.rs:23-26`).
- It is gate 0 of the measurement ladder; the converter exits 1 on any bound violation in the mode that has a bound.

### Scale policy: this container is `mse`

| policy | sub-block scale | bound | this package |
|---|---|---|---|
| `ceil`, the converter default | smallest `ue4m3` ladder step at or above the sub-block max | stored at or above raw always, so values never clamp and `max_rel <= 1.0` holds | not used |
| `mse`, used here | the `ue4m3` ladder step minimizing the summed squared error of the 16 `e2m1` values, by analytic pre-selection plus local refinement | clipping is deliberate: elements above six times the scale are cut, so no fixed relative bound holds | used, see the command below (`main.rs:32-41`, spec section 1 approved 2026-09-02) |

- Under `mse` the rel-bound exit gate is disabled and quality is reported instead: every NVFP4 sidecar line gains `mse`, `mse_ceil`, `mse_ratio` and `max_abs_clipped` (a count of clipped elements), plus one `section_summary` line per NVFP4 section (`main.rs:41-46`).
- The global tensor scale stays max based in both modes, so ladder utilization is unchanged; only the sub-block scale choice differs (`main.rs:47-48`).

## Vision

The container includes the complete vision tower of the base model, quantized with the same policy as the text tower. Nothing was dropped at conversion.

| item | value |
|---|---|
| section | `vit`, the section mark `model.visual`, optional to load |
| tensors | 334 records: 27 vision blocks with 12 tensors each, plus the patch-embed convolution, the position embedding and the merger (norm, fc1, fc2); counted in the container index 2026-09-14 |
| quant | 112 tensors NVFP4 (4.5 bpw, `--scales mse`, per-tensor error stats in the sidecar under `"section":"vit"`), 221 tensors BF16 keeps; counted 2026-09-14 |
| geometry | hidden 1152, 16 heads; patch conv 1536 to 1152; pos embed 2304 x 1152; merger fc2 4608 to 2560; `gelu_pytorch_tanh` in the blocks, exact-erf GELU in the merger; read from the container index 2026-09-14 |
| tokenizer | the `image_pad` token (`248056`) and the vision markers are part of the base tokenizer; image prompts splice the tower output at the placeholder positions |
| engine support | the crow-nest engine image path landed 2026-09-14 (engine `#VIT`, issue #66): the tower loads beside the text sections by default and `serve` answers image requests; ViT embeddings against the f32 oracle over the same container weights read max_abs 3.43e-06 at cos 1.000000 that day |

- There is no separate vision file to download and no projector to fetch: the tower rides inside the single container file, verified by the same `SHA256SUMS` check as everything else.
- Vision quality rows against another engine are not claimed here: the llama.cpp `mmproj` comparison was deferred on 2026-09-14 and the oracle is the gate that ran instead.

## Converter command of record

```
converter --scales mse <model-dir> converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq
wrote converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq: 1658 tensors (843 nvfp4, 815 bf16-keep), payload 104.73 GB, scales mse, violations 9178206, elapsed 5205 s
```

- The first line is the command that produced this container; the second line is the result line of the generation log, verbatim (`converter/converter_mse_vollauf.log:24`).
- The `violations` figure counts elements above the old max-rel bound; under `--scales mse` that is expected, not a defect (see Scale policy).

## Measured numbers

<!-- NUMBERS-OF-RECORD refresh before upload -->

Current numbers: crow-nest v0.3.0, one RTX 5090, Arch Linux, driver 610.57.04, CUDA 13.3.1, this container. llama.cpp runs the same model as `Qwen3.8-Flash-Next-UD-Q2_K_XL` (GGUF, 2.4 bpw, 73 GB); its latest numbers are Crow's Linux placement and the paired Windows prefill (17.41 s = 922.5 tok/s, 2026-09-11; no Linux prefill number exists for it).

| metric | crow-nest CNQ4.5-M | llama.cpp UD-Q2_K_XL | form | source |
|---|---|---|---|---|
| prefill, 16,064 id prompt, cold | **968 / 964 tok/s** | 922.5 tok/s (Windows) | completed-prompt average, two runs | engine commit 1032bc5, 2026-09-17; engine issue #10 |
| decode at 16k context | **36.8 tok/s** (27.19 / 27.22 ms per token) | 41.8 tok/s (Linux, `--load-mode mmap`, `-ncmoe 31 -t 24`, short prompts; 36.7 with `--load-mode none`) | 128 steps after the prompt above vs Crow's placement measurement in the window; not a pair | engine commit 1032bc5, 2026-09-17; `nibor1896/Crow` `docs/user-guide/linux.md`, which carries no date in this repository |
| prefill, 1024 id prompt, cold | **740 tok/s** | n/a | `decode parity`, two runs (92 tok/s before the same-day PLE fix) | engine commit 1032bc5, 2026-09-17 |
| warm short turns through `serve` | **228 ms prefill, 247 ms to the first token** | n/a | 3,296 id cached prefix, 39 to 101 new ids, mean of six turns, two runs | engine commit 4004e66, 2026-09-17 |
| first token after a restart, 3,928 id prompt in Crow | 819 tok/s prefill | n/a | robin's live session, one reading | engine issue #68 artefacts, 2026-09-17 |

<!-- end of the numbers of record -->

- crow-nest moves about 1.9x the expert bytes per token of the 2.4 bpw GGUF.
- The 972 tok/s prefill and 42 tok/s decode figures of the engine's decision record are targets (spec section 0.1, approved 2026-09-02); prefill is above that target on Linux since v0.3.0, decode at 16k context is below it.
- Earlier numbers (Windows, v0.2.0) are in the engine's `CHANGELOG.md` and its v0.2.0 release notes.

## Quality, the ten-task gate

The two arms run different weights, and every comparison names both: llama.cpp runs this model on this machine only as `Qwen3.8-Flash-Next-UD-Q2_K_XL` (GGUF, 2.4 bpw, 73 GB, `-ncmoe 40`, the arm of `decode_out/srv-59b.log`, 2026-09-11); crow-nest runs CNQ4.5-M (NVFP4, 4.5 bpw, 105 GB).

| check | crow-nest CNQ4.5-M | llama.cpp UD-Q2_K_XL | reader | machine | source |
|---|---|---|---|---|---|
| ten tasks, greedy, strict 400 word cap on `t4-prose` | 2 Pass / 5 Partial / 3 Fail of 10 | 2 Pass / 6 Partial / 2 Fail of 10 | same reader for both arms, Rev3 | RTX 5090 | 2026-09-10, B4, engine issue #11 (comment), `nibor1896/Crow` issue #192 (comment) |
| ten tasks, sampled, six seeds, `temperature` above 0 | gate met in 1 of 6 seeds; 4 Pass / 37 Partial / 19 Fail of 60 answers; degeneration 0 of 60 | the reference is the greedy row above | one reader over all six rows, Rev3 | RTX 5090 | 2026-09-10, C2, engine issue #44 |

- No sugarcoating: greedy crow-nest is 2 Pass / 5 Partial / 3 Fail, one task below the llama.cpp reference 2 / 6 / 2 on this reading, and sampling met the gate in 1 of 6 seeds (RTX 5090, 2026-09-10, engine issue #44).
- On the C2 basis greedy met the gate in 0 of 1 seed (decision record, 2026-09-11, comment on engine issue #1).
- Default since the C2 decision, option a, robin, 2026-09-11, comment on engine issue #1: `serve` is request decides. A request without `temperature` or with `temperature <= 0` runs greedy; a request with `temperature > 0` samples with top_p 0.8, top_k 20, presence_penalty 1.5, seed 0 (`serve.rs:1041` of the engine). Greedy stays the identity gate; the measured basis is the rows above.

## Requirements

| item | value | source |
|---|---|---|
| engine | crow-nest, `https://github.com/nibor1896/crow-nest`; the server binary is `serve`, started from the engine repo root as its README Start section shows | engine README |
| OS | Linux (x86_64, tested on Arch Linux, kernel 7.2, driver 610.57.04) and Windows x64. Linux needs the CUDA 13.3 runtime libraries on `LD_LIBRARY_PATH` and the container on a path WITHOUT filesystem compression (`chattr +m` on btrfs `compress=` mounts); `tools/serve-linux.sh` of the engine starts `serve` inside a memory-bounded `systemd-run` scope | engine README, Platform and Run on Linux; engine issue #15, closed 2026-09-17 |
| GPU | one NVIDIA Blackwell GPU, `sm_120`, `compute_120a` target; every measured number here is one RTX 5090 | `docs/system-landscape.md:12` of the engine repo |
| host RAM | 62 to 64 GB class: the engine pins up to 46 GiB of host memory for the cold expert tier, sized at boot from the RAM that can be pinned (on Linux the NVIDIA driver's freed pinned-page pool and the page cache count as free; another live CUDA process makes the gate conservative); the chain gate (engine issue #38) still applies to measurements; the budget as built was measured 2026-09-17 | `docs/system-landscape.md` and `docs/architecture.md` 2.1 of the engine repo |
| CUDA | CUDA 13.3 toolkit. Windows: `nvrtc64_133_0.dll` needs the toolkit bin directory on `PATH`. Linux: `libnvrtc.so.13` and the runtime on `LD_LIBRARY_PATH` (never the `lib/stubs` directory); the driver's own `libcuda.so.1` | `docs/system-landscape.md` of the engine repo |
| container placement | the engine default path is `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq` inside the engine repo, or `CROW_CNQ` names any path | `geo.rs` `DEFAULT_CNQ` and `boot.rs` of the engine, `docs/env.md`, row `CROW_CNQ` |
| hot-set manifest placement | the engine default path is `decode_out/hotsets-M-longctx2100-n160.json`, or `CROW_HOTSETS` names any path | `geo.rs` `DEFAULT_HOTSETS` and `boot.rs` of the engine, `docs/env.md`, row `CROW_HOTSETS` |
| server | binds `127.0.0.1`, default port 8099, one request at a time, one engine per machine via `engine/.engine.lock` | `serve.rs:445` of the engine, engine README |

## Self-test

This package verifies itself. `selftest/` ships a golden that the layer-wise oracle produced from the UNQUANTIZED originals, and the engine compares its own layer output against it — no original safetensors, no python, no oracle environment, no network. It is the one numeric check a download can actually run (engine issue #64, first run 2026-09-18).

| item | value |
|---|---|
| what runs | `decode selftest <golden-dir>` of the crow-nest engine: one container load and one layer, 32.0 s wall in total on 2026-09-18, of which the load is 23 s |
| what is compared | the WHOLE text decoder layer `0` — hyper-connection mix, the GDN linear-attention mixer, the second mix, the `352`-expert MoE, both residual injections — on a fixed 8-token input, against the f32 reference of the unquantized weights |
| the golden | produced 2026-09-02 by the engine repo's `oracle/` chain from `Qwen/Qwen3.8-Flash-Next` at the revision of record: transformers 5.16.1, torch 2.13.0+cpu, f32, real unquantized weights, 24 of 24 layer-`0` tensors suffix-matched. It is data, not code: the engine reads the two arrays and the manifest, and nothing else |
| gate | `max_abs <= 0.125` and `NaN == 0`, the bound inclusive — the engine's `layercheck` gate of record since 2026-09-04, where `0.125` is the measured spread of genuine NVFP4 round-to-nearest noise on this golden |
| measured, this package | `max_abs 9.184837e-2`, `rel_L2 1.3671e-2`, `NaN 0` — `73.5` percent of the gate — on 2026-09-18, RTX 5090, driver 610.57.04, CUDA 13.3.1, NVRTC 13.3.33, from a package directory holding nothing but the files in the table above |
| positive control | `test ! -d models`: in a directory that holds the originals the self-test REFUSES, before the engine is even started, because such a run says nothing about a download. Both arms run on 2026-09-18 — refused with `models/` present, `ALL GREEN` without it, identical `max_abs` in the arm that was allowed |
| what it does not cover | one layer on one 8-token input. It catches a container that was corrupted, truncated, converted with the wrong scales or loaded by a broken build; it is not the engine's full numeric contract, which is the byte-identical parity gate of the engine repo and needs that repo's reference dumps, not this package |

- The engine repo's `tools/selftest.sh` runs all three items — the control, the sums and the engine — and exits non-zero on any of them:

```
tools/selftest.sh /path/to/this/package
```

## Not measured

| item | status |
|---|---|
| Linux, the numeric contract | (rows above). The Linux build reproduces the Windows 8-row parity dump byte for byte; on the 512 and 1024 row forms the logits drift from the Windows references (NVRTC 13.3.33 + driver 610.57 vs 13.3.73 + 616.56, the kernels are driver-JIT'd PTX) while the generated ids stay identical over 517 positions; the ten-task Windows records reproduce 1 of 10 answers on Linux, the other 9 flip on near-ties (0.037 to 0.076 logits at the flip position). Linux values of record exist per form in the engine repo (`tools/gate-linux.sh`), established 2026-09-17 |
| long agentic sessions | not measured beyond 16k context; a 300-turn Crow goal-mode session at 170k+ context degenerated (engine issues #67 `</think>` tags streamed as content, fixed 2026-09-18, and #68 the degeneration record, 2026-09-17) |
| Ampere and Ada GPUs | the fallback stage is not planned (engine issue #12) |
| the ten-task gate with the upstream thinking mode sampler profile | not measured; C2 measured the six seeds above |
| sharded or split variants of this container | none exist |
| trademark review of the repo name | not done; the licence regulates no naming for derivatives (see License) |

## Provenance

| item | value |
|---|---|
| base model | `Qwen/Qwen3.8-Flash-Next` on Hugging Face, public, not gated |
| revision of record | `de4b8e4d43b917e7706784d8bb445c9af86a3540`, upstream `lastModified` 2026-08-27T05:03:36Z |
| verification | the upstream licence text was fetched and checked against the Hub on 2026-09-11 at the revision above; the local download was byte-identical |
| upstream citation | the upstream model card carries two BibTeX entries, `qwen2026design` (techreport) and `qwen3.8flashnext` (misc, url `https://qwen.ai/blog?id=qwen3.8-flash-next`), at `README.md:653-680` of the upstream repo; they are the citation of record for this derivative |

## License

| item | value |
|---|---|
| licence | Qwen Community License 1.0 for the model weights; name and version verified verbatim against the upstream repo on 2026-09-11, revision `de4b8e4d43b917e7706784d8bb445c9af86a3540` |
| text | ships as `LICENSE` next to this card: 3,235 B, 15 lines, sha256 `a0dc422560841fd68e06d974907f8b4c709bca44a67daad2b528437bdf676c08`, byte-identical to the upstream file at the revision above, verified 2026-09-11 |
| duty, every copy | keep the copyright notice and the permission notice in all copies or substantial portions (licence section 1, sentence 1) |
| duty, large deployments | above 100,000,000 monthly active users or above US$ 20,000,000 monthly revenue, the model name must be displayed prominently in the product UI (licence section 1, sentence 2; licence verified 2026-09-11) |
| duty, hosted service | a Model as a Service or AI Work Assistant offering needs a separate licence from Qwen before commercial use; internal use without third-party access is exempt (licence section 2) |
| attribution | the licence prescribes no attribution formula and no naming rule for derivatives; this card therefore states provenance plus the upstream BibTeX (see Provenance) and invents no formula |
| scope | this package is weights for local inference with the crow-nest engine; no hosted inference endpoint is offered |
| engine code | the crow-nest engine and converter are Apache-2.0, see the `LICENSE` of the engine repo |
