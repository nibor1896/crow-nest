# crow-nest

<a href="https://github.com/nibor1896/crow-nest/releases"><img src="https://img.shields.io/github/v/release/nibor1896/crow-nest?style=flat-square&logo=github&logoColor=ffffff&labelColor=000000" alt="release"></a>
<a href="https://github.com/nibor1896/crow-nest/actions"><img src="https://img.shields.io/github/actions/workflow/status/nibor1896/crow-nest/ci.yml?branch=main&style=flat-square&logo=githubactions&logoColor=ffffff&labelColor=000000" alt="ci"></a>
<a href="engine/"><img src="https://img.shields.io/badge/Rust-1.98-555555?style=flat-square&logo=rust&logoColor=ffffff&labelColor=000000" alt="rust"></a>
<a href="docs/architecture.md"><img src="https://img.shields.io/badge/CUDA-13.3%20%C2%B7%20Blackwell%20sm__120-555555?style=flat-square&logo=nvidia&logoColor=76B900&labelColor=000000" alt="cuda"></a>
<a href="README.md#run-on-linux-from-the-repository-root"><img src="https://img.shields.io/badge/Linux-x86__64-555555?style=flat-square&logo=linux&logoColor=ffffff&labelColor=000000" alt="linux"></a>
<a href="README.md"><img src="https://img.shields.io/badge/Windows-x64-555555?style=flat-square&labelColor=000000" alt="windows"></a>
<a href="https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M"><img src="https://img.shields.io/badge/model-Qwen3.8--Flash--Next--CNQ4.5--M-555555?style=flat-square&logo=huggingface&logoColor=FFD21E&labelColor=000000" alt="model on hugging face"></a>
<a href="https://github.com/nibor1896/Crow"><img src="https://img.shields.io/badge/client-Crow-555555?style=flat-square&logo=github&logoColor=ffffff&labelColor=000000" alt="crow"></a>
<a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-555555?style=flat-square&labelColor=000000" alt="license"></a>

| item | value |
|---|---|
| product | inference engine for one model on one GPU, own quantization, own container, thin CUDA kernels in Rust |
| model | Qwen3.8-Flash-Next as CNQ4.5-M (NVFP4, 4.5 bpw), converted from the original safetensors; the container carries the visual tower (`vit` section) and serve answers image requests (`CROW_VIT`, default on) |
| platform | Linux and Windows, NVIDIA Blackwell (`sm_120`), CUDA only |
| license | code Apache-2.0 (`LICENSE`); the model files carry the Qwen Community License 1.0 |


| | |
|---|---|
| Engine | one model, one GPU, Rust + thin CUDA kernels |
| Model | Qwen3.8-Flash-Next, NVFP4 container CNQ4.5-M |
| API | OpenAI-compatible HTTP, client [Crow](https://github.com/nibor1896/Crow) (2026-09-23) |
| Release | v0.4.0 (2026-09-23), [CHANGELOG.md](CHANGELOG.md) |

## Status (2026-09-23)

| | |
|---|---|
| #91 output corruption | fixed: PLE n-gram rows read at the wrong container offset |
| Live agent run | 60 → 0 corrupt tokens (2026-09-23) |
| 23-site corruption set | 15/23 → 4/23 (llama.cpp UD-Q2_K_XL: 4/23) (2026-09-23) |
| Full table | [docs/status.md](docs/status.md) |

## Requirements (2026-09-23)

| | |
|---|---|
| GPU | NVIDIA Blackwell `sm_120`, RTX 5090 32 GB |
| Host RAM | 64 GB |
| CUDA | 13.3 runtime (NVRTC) (2026-09-23) |
| Rust | stable |
| OS | Linux, Windows |
| Container | `converter/Qwen3.8-Flash-Next-CNQ4.5-M.cnq`, 104.7 GB, [Hugging Face](https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M) (2026-09-23) |

## Quick start

```
# build
cd engine && cargo build --release --bin serve && cd ..

# run, Linux
tools/serve-linux.sh --port 8099

# run, Windows
engine/target/release/serve.exe --port 8099

# ask
curl -s http://127.0.0.1:8099/v1/chat/completions -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hello."}],"max_tokens":32}'

# use from Crow
crow --base-url http://127.0.0.1:8099/v1
```

## Documentation

| Topic | File |
|---|---|
| Build, run and check in full | [docs/getting-started.md](docs/getting-started.md) |
| Environment variables | [docs/env.md](docs/env.md) |
| Architecture | [docs/architecture.md](docs/architecture.md) |
| Measurements and targets | [docs/measurements.md](docs/measurements.md) |
| Repository layout, quality gates | [docs/repository.md](docs/repository.md) |
| Status per issue | [docs/status.md](docs/status.md) |
| Model card | [docs/model-card.md](docs/model-card.md) |
| Changelog | [CHANGELOG.md](CHANGELOG.md) |

## License

| | |
|---|---|
| Code | Apache-2.0, [LICENSE](LICENSE) |
| Model weights and containers | Qwen Community License 1.0 |
