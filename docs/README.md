# docs index

Every page under `docs/`, grouped; one line each, taken from the page's own title and opening lines. The repository's
front page is [`../README.md`](../README.md). GLM-5.3-Flash (glm5_next) is the third model family and
Crow's third operating point (robin, 2026-10-11, Crow #365); its operating set is `runs/glm53-flash/quick/arm4.env`.

## Getting started and status

- [getting-started.md](getting-started.md): full build, run and check instructions (moved out of the README on 2026-09-23).
- [status.md](status.md): the per-issue status table (open items, live checks, scope, history).
- [repository.md](repository.md): repository layout, environment and quality gates.
- [system-landscape.md](system-landscape.md): the machine and environment every measurement in the repo refers to (2026-09-02).

## Measurements

- [measurements.md](measurements.md): the measured figures and the targets (moved out of the README on 2026-09-23).
- [long-context-goalmode.md](long-context-goalmode.md): the 300-turn Crow goal-mode session at 170k context, replayed through `serve` (#68).
- [measurement-coverage.md](measurement-coverage.md): expert coverage curve from Crow's routing logs (#3, 2026-09-02).
- [measurement-handoff.md](measurement-handoff.md): job-ring handoff benchmark, device-side publish to a host stager (#7, 2026-09-02).
- [ten-tasks.md](ten-tasks.md): the ten-task gate, fixed from Crow's real workload (spec 5.2).
- [ten-task-expected.md](ten-task-expected.md): expected answers and grading protocol of the ten-task gate (German).
- [ten-task-linux.md](ten-task-linux.md): the Linux ten-task record, both engines at the real operating row (#80).
- [ten-task-linux-report.md](ten-task-linux-report.md): grading report of that record (#80, #87); worksheet [ten-task-grades-worksheet.json](ten-task-grades-worksheet.json), prompts [ten-task-prompts-crowlab.json](ten-task-prompts-crowlab.json).

## Architecture and engine

- [architecture.md](architecture.md): the engine spec, sections 0-9 (approved by robin 2026-09-02, later sections added), with a table of contents.
- [diagrams.md](diagrams.md): the eight living architecture diagrams of the spec (#14, rendered at `8bad310`, v0.3.1).
- [env.md](env.md): every `CROW_*` environment variable the engine reads (190, guarded by `tools/check_env_docs.py`), with mode, default and evidence.
- [numerics-diff.md](numerics-diff.md): GDN normalization and QSA sparse attention against the HF reference and llama.cpp (#89).
- [cuda-rust-evaluation.md](cuda-rust-evaluation.md): NVIDIA's CUDA Rust tracks weighed against the engine (2026-09-17).

## Flash-Next / Qwen

- [model-card.md](model-card.md): the Hugging Face model card of Qwen3.8-Flash-Next-CNQ4.5-M (Flash-Next, NVFP4), with its front-matter.
- [hotset-calibration.md](hotset-calibration.md): how the hot-set manifest `serve` loads is cut and accepted (#106).
- [quality-probe.md](quality-probe.md): the answer-quality probe against any OpenAI-compatible endpoint (#75, requant step 1).
- [dense-originals.md](dense-originals.md): the BF16 originals of the dense path and their proof (#76, requant step 2).
- [dense-overlay.md](dense-overlay.md): the dense path in BF16 as an overlay container (#77, requant step 3).
- [oracle-kld.md](oracle-kld.md): the f32 oracle and the KLD of a quantization against it (#78, requant step 4).
- [expert-requant.md](expert-requant.md): the routed experts re-quantized with an importance matrix (#79, requant step 5).
- [oracle-longctx.md](oracle-longctx.md): the oracle-KLD instrument at long context and the English-prose corpus (#90).

## GLM-5.3-Flash (glm5_next, measurement)

Plan and conversion:

- [glm5-next-recipe.md](glm5-next-recipe.md): what one glm5_next forward pass computes, block by block (plan step 2, #153).
- [nvme-read-rate.md](nvme-read-rate.md): NVMe read rate for expert-sized blocks (step 3, #146).
- [dram-bandwidth.md](dram-bandwidth.md): DRAM read bandwidth per core class (step 4, #170).
- [glm-download.md](glm-download.md): downloading and verifying the originals (`tools/fetch-glm.py`).
- [glm5-reference-runner.md](glm5-reference-runner.md): the layerwise HF reference runner and its goldens (step 7, #158).
- [glm-tier-simulation.md](glm-tier-simulation.md): tier simulation and gate G1 (step 8, #147).
- [glm-staged-conversion.md](glm-staged-conversion.md): the staged CNQ4.5 conversion of step 9 (#157; that container was deleted on 2026-10-08).
- [glm-entropy.md](glm-entropy.md): entropy census and gate G2, failed (step 10, #148).
- [glm-mul1-conversion.md](glm-mul1-conversion.md): the 3-bit MUL1 container the engine runs, and the MTP overlay (#182).
- [glm5-tokenizer.md](glm5-tokenizer.md): tokenizer, chat template and tool-call markup (step 12, #160).

Engine path:

- [glm5-model.md](glm5-model.md): the layer path, the `glmgolden` harness (G3) and the three expert tiers (`glm5_tiers`, `glm5_run`).
- [glm5-mhc.md](glm5-mhc.md): the mHC residual (#161).
- [glm5-kda.md](glm5-kda.md): the KDA sub-block on the GPU (#162).
- [glm5-mla.md](glm5-mla.md): MLA latent cache and DSA indexer (#163).
- [glm5-moe.md](glm5-moe.md): router, clamped MUL1 experts, shared expert, dense FFN (#164).
- [glm5-head.md](glm5-head.md): stream mean, final norm, lm_head (#165).
- [glm5-mtp.md](glm5-mtp.md): the MTP block (NextN, checkpoint layer 45; MTP stays off, #192).
- [mul1-gemv.md](mul1-gemv.md): MUL1 trellis expert GEMV and FFN (#180).
- [cpu-nvfp4.md](cpu-nvfp4.md): CPU NVFP4 expert FFN (#173).
- [expert-cache.md](expert-cache.md): the dynamic expert cache over VRAM and pinned tiers (#175).
- [glm-stability.md](glm-stability.md): stability policy: VRAM headroom, decode staging, the 0x7F scale cap, the 2-hour soak (not run).

Switches and figures: the GLM rows of [env.md](env.md) (sections GLM-5.3-Flash, NVMe tier, GLM operating set) and
`runs/glm53-flash/quick/README.md`.

## Tooling and process

- [regression-gate.md](regression-gate.md): regression gate R, Flash-Next and the 27B unchanged by an engine change (#174).
- [prequal.md](prequal.md): `tools/prequal.py`, size, placement and decode ceiling of a new model from its shard headers, no download, no GPU (#240).
- [improve-loop.md](improve-loop.md): the improve loop, canonical definition (German, 2026-09-03).
- [sota-research-2026-09-03.md](sota-research-2026-09-03.md): SOTA research on FP4 inference on sm_120a / WDDM (German, 2026-09-03).
- [acceptance/](acceptance/): acceptance protocols per issue (#83 to #96).

## Archive

- [archive/README-v0.6.0.md](archive/README-v0.6.0.md): the README as it stood at v0.6.0, archived 2026-09-25; do not edit.
