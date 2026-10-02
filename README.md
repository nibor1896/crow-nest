<picture>
  <source media="(max-width: 700px) and (prefers-color-scheme: dark)" srcset="docs/images/readme/crow-nest-mobile-dark.svg">
  <source media="(max-width: 700px)" srcset="docs/images/readme/crow-nest-mobile-light.svg">
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/crow-nest-dark.svg">
  <img src="docs/images/readme/crow-nest-light.svg" width="100%" alt="crow-nest: one GPU, its own quant, two model families. An inference engine in Rust and CUDA for Qwen3.8-Flash-Next as CNQ4.5-M and the dense Qwen3.8-27B as CNQ4.5 with MTP and its own F16 vision projector, OpenAI-compatible, the engine behind Crow.">
</picture>

**Build** (Flash-Next container: 104.7 GB from Hugging Face)

```bash
git clone https://github.com/nibor1896/crow-nest && cd crow-nest
hf download nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M Qwen3.8-Flash-Next-CNQ4.5-M.cnq --local-dir converter
cd engine && cargo build --release --bin serve && cd ..
```

**Run, Linux**

```bash
tools/serve-linux.sh --port 8099
```

**The 27B** (container, tokenizer, vision projector)

```bash
hf download nibor1896/Qwen3.8-27B-CNQ4.5 Qwen3.8-27B-CNQ4.5.cnq --local-dir converter
hf download Qwen/Qwen3.8-27B tokenizer.json tokenizer_config.json --local-dir models/Qwen3.8-27B
hf download unsloth/Qwen3.8-27B-GGUF mmproj-F16.gguf --local-dir models/Qwen3.8-27B
```

**Run the 27B, Linux**

```bash
CROW_CNQ=converter/Qwen3.8-27B-CNQ4.5.cnq tools/serve-linux.sh --port 8099
```

**Run the 27B, Windows**

```powershell
$env:CROW_CNQ = "converter\Qwen3.8-27B-CNQ4.5.cnq"; engine\target\release\serve.exe --port 8099
```

**Run, Windows**

```powershell
engine/target/release/serve.exe --port 8099
```

**Windows engine without building** (`serve.exe` + NVRTC from the release, run in the checkout)

```powershell
irm https://github.com/nibor1896/crow-nest/releases/download/v0.8.0/crow-nest-engine-0.8.0-win-x64.zip -OutFile engine.zip; Expand-Archive engine.zip crow-nest-engine
$env:CROW_CNQ = "converter\Qwen3.8-27B-CNQ4.5.cnq"; crow-nest-engine\serve.exe --port 8099
```

**Linux engine without building** (`serve` + NVRTC from the release, glibc 2.34+)

```bash
curl -LO https://github.com/nibor1896/crow-nest/releases/download/v0.9.0/crow-nest-engine-0.9.0-linux-x64.tar.gz
mkdir crow-nest-engine && tar -xzf crow-nest-engine-0.9.0-linux-x64.tar.gz -C crow-nest-engine
CROW_CNQ=converter/Qwen3.8-27B-CNQ4.5.cnq LD_LIBRARY_PATH=$PWD/crow-nest-engine crow-nest-engine/serve --port 8099
```

**Use from Crow**

```bash
crow --base-url http://127.0.0.1:8099/v1
```

<p align="center">
<a href="docs/getting-started.md">Getting started</a> ·
<a href="docs/env.md">Environment</a> ·
<a href="docs/architecture.md">Architecture</a> ·
<a href="docs/measurements.md">Measurements</a> ·
<a href="docs/repository.md">Repository</a> ·
<a href="docs/status.md">Status</a> ·
<a href="docs/model-card.md">Model card</a> ·
<a href="CHANGELOG.md">Changelog</a> ·
<a href="docs/archive/README-v0.6.0.md">Previous README</a>
</p>

<p align="center"><sub>
Apache-2.0 · <a href="https://github.com/nibor1896/crow-nest">nibor1896/crow-nest</a> ·
Model: <a href="https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M">Qwen3.8-Flash-Next CNQ4.5-M</a> (Qwen Community License 1.0) ·
<a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a> (Apache-2.0) ·
Client: <a href="https://github.com/nibor1896/Crow">Crow</a>
</sub></p>

<p align="center">
<a href="https://ko-fi.com/nibor1896"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/kofi-dark.svg">
  <img src="docs/images/readme/kofi-light.svg" height="44" alt="Support crow-nest on Ko-fi">
</picture></a>
</p>
