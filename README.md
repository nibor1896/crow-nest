<picture>
  <source media="(max-width: 700px) and (prefers-color-scheme: dark)" srcset="docs/images/readme/crow-nest-mobile-dark.svg?v=20261012">
  <source media="(max-width: 700px)" srcset="docs/images/readme/crow-nest-mobile-light.svg?v=20261012">
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/crow-nest-dark.svg?v=20261012">
  <img src="docs/images/readme/crow-nest-light.svg?v=20261012" width="100%" alt="crow-nest: one GPU, its own quant, three model families. An inference engine in Rust and CUDA for Qwen3.8-Flash-Next as CNQ4.5-M, the dense Qwen3.8-27B as CNQ4.5 with MTP and its own F16 vision projector, and GLM-5.3-Flash as MUL1K3 (EXL3 3.05 bpw experts), OpenAI-compatible, the engine behind Crow.">
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

**Windows engine without building** (`serve.exe` from the release, NVRTC from NVIDIA's own wheel)

```powershell
irm https://github.com/nibor1896/crow-nest/releases/download/v0.13.1/crow-nest-engine-0.13.1-win-x64.zip -OutFile engine.zip; Expand-Archive engine.zip crow-nest-engine
powershell -ExecutionPolicy Bypass -File crow-nest-engine\fetch-nvrtc.ps1 -Target crow-nest-engine   # NVRTC 13.3.33 from NVIDIA's PyPI wheel, verified; the zip holds no NVIDIA file
$env:CROW_CNQ = "converter\Qwen3.8-27B-CNQ4.5.cnq"; crow-nest-engine\serve.exe --port 8099
```

**Linux engine without building**: no Linux package at the moment (the v0.9.0 tarball carried NVRTC and is withdrawn); build from source as above, then `tools/fetch-nvrtc.sh --target engine/target/release`.

**GLM-5.3-Flash**, Crow's third operating point (3-bit container: 124.6 GB from Hugging Face, [conversion](docs/glm-mul1-conversion.md); operating set [`arm4.env`](runs/glm53-flash/quick/README.md))

```bash
hf download nibor1896/GLM-5.3-Flash-MUL1K3 GLM-5.3-Flash-MUL1K3.cnq --local-dir converter
hf download zai-org/GLM-5.3-Flash tokenizer.json tokenizer_config.json chat_template.jinja --local-dir models/GLM-5.3-Flash-original
cd engine && cargo build --release --bin serve --bin glm5_run && cd ..
# one prompt, from the repository root
env $(cat runs/glm53-flash/quick/arm4.env) CROW_GLM_ARENA_WARM=runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json   engine/target/release/glm5_run --cnq converter/GLM-5.3-Flash-MUL1K3.cnq --cold --reps 1 --tokenizer models/GLM-5.3-Flash-original/tokenizer.json --prompt "Explain how a CPU cache works." -n 128
# serve (Linux unmeasured)
env $(cat runs/glm53-flash/quick/arm4.env) CROW_GLM_ARENA_WARM=runs/glm53-flash/arena-warm/glm53-cal4-decode-counts.json CROW_CNQ=converter/GLM-5.3-Flash-MUL1K3.cnq tools/serve-linux.sh --port 8099
```

**Run GLM-5.3-Flash, Windows** (from the repository root)

```powershell
Get-Content runs\glm53-flash\quick\arm4.env | ForEach-Object { $_ -split '\s+' } | Where-Object { $_ } | ForEach-Object { $k, $v = $_ -split '=', 2; Set-Item "env:$k" $v }
$env:CROW_GLM_ARENA_WARM = "runs\glm53-flash\arena-warm\glm53-cal4-decode-counts.json"; $env:CROW_CNQ = "converter\GLM-5.3-Flash-MUL1K3.cnq"; engine\target\release\serve.exe --port 8099
```

| GLM-5.3-Flash container | format |
|---|---|
| routed experts | exllamav3's EXL3 (MUL1 codebook, K=3, 3.05 bpw): [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3); trellis method from QTIP (Tseng et al. 2024): [arXiv:2406.11235](https://arxiv.org/abs/2406.11235) |
| dense part | crow-nest `cnq4.5-glm5-next` (NVFP4) |

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
<a href="docs/model-card.md">Model card (Flash-Next)</a> ·
<a href="docs/README.md">All docs</a> ·
<a href="CHANGELOG.md">Changelog</a> ·
<a href="docs/archive/README-v0.6.0.md">Previous README</a>
</p>

<p align="center"><sub>
Apache-2.0 · <a href="https://github.com/nibor1896/crow-nest">nibor1896/crow-nest</a> ·
Model: <a href="https://huggingface.co/nibor1896/Qwen3.8-Flash-Next-CNQ4.5-M">Qwen3.8-Flash-Next CNQ4.5-M</a> (Qwen Community License 1.0) ·
<a href="https://huggingface.co/Qwen/Qwen3.8-27B">Qwen3.8-27B</a> (Apache-2.0) ·
<a href="https://huggingface.co/nibor1896/GLM-5.3-Flash-MUL1K3">GLM-5.3-Flash MUL1K3</a> from <a href="https://huggingface.co/zai-org/GLM-5.3-Flash">zai-org/GLM-5.3-Flash</a> (MIT) ·
Client: <a href="https://github.com/nibor1896/Crow">Crow</a>
</sub></p>

<p align="center">
<a href="https://ko-fi.com/nibor1896"><picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/images/readme/kofi-dark.svg">
  <img src="docs/images/readme/kofi-light.svg" height="44" alt="Support crow-nest on Ko-fi">
</picture></a>
</p>
