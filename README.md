<p align="center">
  <img src="soup.png" alt="Soup" width="280">
</p>

<h1 align="center">Soup</h1>

<p align="center">
  <strong>Fine-tune and post-train LLMs in one command. No SSH, no config hell.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> &middot;
  <a href="#configuration">Config</a> &middot;
  <a href="#documentation">Docs</a> &middot;
  <a href="docs/commands.md">Commands</a> &middot;
  <a href="docs/models.md">Models</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10--3.12-blue" alt="Python 3.10-3.12">
  <img src="https://img.shields.io/badge/license-Apache--2.0-blue" alt="Apache-2.0 License">
</p>

> **This is a fork** of [MakazhanAlpamys/Soup](https://github.com/MakazhanAlpamys/Soup) —
> all credit for the original design, the layer-streaming approach, and the core CLI goes
> to the upstream project and its contributors. This fork adds a Web UI, RAM-prefetch
> streaming, additional quantization strategies (AWQ/GPTQ/k-quants/i-quants), an optional
> compression pipeline (activation scan → merge/SVD compress → distill, all opt-in and
> declared in the training YAML), multi-dataset training/validation/calibration support,
> and a pre-training data-verification gate. See [`README-FORK.md`](README-FORK.md) for the
> full list of fork-specific changes and how they map onto upstream, and please direct
> anything not specific to this fork's own changes (bug reports against core training
> behavior, feature requests unrelated to the items above) to the
> [upstream repository](https://github.com/MakazhanAlpamys/Soup) instead — this fork tracks
> upstream but isn't a substitute for it.

---

Soup turns the pain of LLM fine-tuning into a simple workflow. One config, one command, done.
It also ships a Web UI and a set of optional model-compression tools (neuron-importance
ranking, similar-neuron merging, SVD compression, a bridge into distillation) on top of the
core CLI — all opt-in, all covered below.

```bash
git clone https://github.com/serpis172/Soup.git
cd Soup
pip install -e ".[train]"   # add [train] to fine-tune; bare -e "." is the light CLI
soup init --template chat
soup train
```


**Fine-tune an 8B model on a 4 GB laptop GPU.** Layer streaming keeps the frozen base out of
VRAM and feeds it to the GPU one decoder layer at a time. Measured on an RTX 3050 Laptop 4 GB:
Llama-3.1-8B-Instruct + NF4 at **119.6 tok/s, 3.32 GB peak** — bit-exact against a normal
resident run, and reproduced independently on an H100 at 113.00 tok/s in the same 3.32 GB.
(The tok/s figure was measured on v0.72.2, before the v0.73.0 correctness repair that cost
−4.8% at 32B; it has not been re-run on a 4 GB card since.) Opt-in (`stream_layers: true`)
and still BETA —
[how it works](docs/performance-and-quantization.md#layer-streaming-beta-v0720-nf4-v0722-disk--wider-archs-v0723-preference-losses-v0724) ·
[all measurements](benchmarks/) ·
**[check it yourself on a free Colab T4](notebooks/proof-4gb.ipynb)** (caps the process to
4 GB, then asserts a streamed model is bit-identical to a normal one)

<p align="center">
  <a href="https://youtu.be/T1LCErE943E"><img src="docs/assets/layer-streaming.gif" alt="soup train pre-flight for Llama-3.1-8B on a 4 GB card: a 3.60 GB base store pinned in RAM across 32 layers and two 113 MB VRAM buffers, then a measured peak of 3.32 GB at 119.6 tok/s, stopping short of the 4 GB line"></a><br>
  <sub>Llama-3.1-8B-Instruct + NF4, LoRA, batch 1, seq 512 on an RTX 3050 Laptop 4 GB — <b>3.32 GB peak, 119.6 tok/s</b>. <a href="https://youtu.be/T1LCErE943E">Full video (90s)</a></sub>
</p>

## Why Soup?

Training LLMs is still painful. Even experienced teams spend 30-50% of their time fighting
infrastructure instead of improving models. Soup fixes that.

- **Zero SSH.** Never SSH into a broken GPU box again.
- **One config.** A simple YAML file is all you need.
- **Auto everything.** Batch size, GPU detection, quantization — handled.
- **Works locally.** Train on your own GPU with QLoRA. No cloud required.

## What's New

**v0.73.3 — every pull request in this release came from someone other than the
maintainer.** All 24 of them, from eight people, five of whom appear here for the first
time. What they found is the interesting part: four separate flags that were validated,
documented, and then read by nothing.

- **Assistant-only masking trained on zero tokens, with a normal loss curve.** A
  tokenizer returning `BatchEncoding` — which is not a `dict` — slipped past the guard,
  so the label mask was built from the mapping's **key strings**. No exception, no
  warning, a loss curve that looks like training. Found by reading the type, not by
  hitting the bug.
- **On Apple Silicon, `quantization: 4bit` was silently rewritten to `none`.**
  `detect_device()` did not know MLX, so every run reported "CPU (no GPU detected)" and
  quietly downgraded. The label was never the harm; the quantization decision is now
  explicit and testable instead of hidden inside a 900-line function.
- **`soup train --no-reexec` printed a launch command with your own flags missing** —
  follow it literally and you trained without `--fsdp`, and **the run succeeded**, so
  nothing pointed back at the hint. Two hand-maintained copies of "what the user typed";
  the printed one is deleted, and the hint now derives from the argv that actually
  launches the run.
- **`training.bnb_4bit_use_double_quant` was read by nothing.** Every 4-bit path
  hardcoded `True`, so setting it to `false` changed your config fingerprint and nothing
  else. Fixing it correctly also meant *not* defaulting the field: a plain `True` breaks
  round-tripping for 21 of 173 shipped configs.
- **On Windows, a process that genuinely exits with code 259 read as alive forever**,
  because that is also `STILL_ACTIVE`. It defeated run reconciliation and could wedge the
  MCP execution cap shut with no error an operator could act on.
- **New: `soup mcp serve --allow-execute`** runs a planned training or export behind a
  single-use, server-generated confirmation token — no command, no argv, no
  client-supplied environment — with the config snapshotted at plan time and protected
  paths digested by content, so a model cannot be swapped between planning and running.

The measurement record for the earlier VRAM work, published as written — including the
**three readings withdrawn during it** — is
[`benchmarks/gate-v0.73.1-measured-vram-fit.md`](benchmarks/gate-v0.73.1-measured-vram-fit.md).

```yaml
# soup.yaml — then just `soup train --config soup.yaml`
training:
  stream_layers: true      # base streams out of VRAM; only the adapter trains
  quantization: 4bit       # NF4 — ~4x smaller store, so 8B fits a 4 GB card
  batch_size: 4            # bigger batches amortise the weight read
  stream_source: auto      # RAM when it fits, NVMe disk when it does not
  seed: 1234               # new in v0.73.0
```

> Python **3.10–3.12** only. v0.73.0 adds the upper bound that was missing: on 3.13+, pip
> used to resolve untested PyTorch wheels that crash in the native extension before Soup
> runs at all.

## Web UI & model-compression tools

A Web UI and a set of optional model-compression tools sit on top of the core CLI.
Everything below is opt-in — plain `soup train` on the CLI works exactly as described above
whether or not you touch any of this.

### Web UI

```bash
chmod +x start_ui.sh
./start_ui.sh              # first run: builds the image (10-20 min, compiles llama.cpp)
                            # later runs: only rebuilds layers that actually changed
./start_ui.sh --rebuild    # force a clean image if you suspect it's stale/corrupted
./start_ui.sh --skip-build # relaunch the existing image, no docker build at all
```

A FastAPI backend (`soup_cli/ui/app.py`) serves the dashboard — the same one `soup ui` runs
outside Docker — with a Bearer token gating every action that starts training, downloads a
model, or writes a checkpoint (printed to the console on launch). Prerequisites: Docker with
NVIDIA Container Toolkit, ~16 GB RAM, an NVIDIA GPU with ≥4 GB VRAM for the Docker path (the
plain `soup ui` CLI path has no GPU requirement of its own — it shells out to `soup train`,
which has the normal requirements below).

Pages: **Dashboard** (runs, live CPU/RAM/GPU, a doctor-style health banner), **New
Training** (template picker + YAML editor, plus a **Streaming & Quantization** panel — RAM
prefetch slider, quantization format/bit picker, with a live diff of what changed), **Data**,
**Chat**, **Tool Outputs**, **Model Hub** (search + download models/datasets/benchmarks from
Hugging Face, with the Hub's own filters), and **Compress** (below). Background jobs
(downloads, compress scans) toast a notification on completion even if you've navigated
away.

### Layer RAM prefetch

```yaml
training:
  stream_layers: true
  ram_cache_gb: 8         # 0 = disabled
```

A single background thread reads upcoming decoder layers ahead of need into pinned host
RAM, direction-aware (it follows the forward-then-backward zigzag gradient checkpointing
already produces). Not an LRU cache — that turned out to be the wrong model for a fully
deterministic access pattern; see `soup_cli/memory/layer_cache.py` for why. The requested
GB is clamped to a safe fraction of *actually available* system RAM (via `psutil`), so a
value that doesn't fit gets reduced with a printed warning instead of feeding the OOM
killer.

### Quantization

```yaml
training:
  custom_quant_strategy: awq   # none | awq | gptq | k-quants | i-quants | qat
  custom_quant_detail: "4"     # bits for awq/gptq; a GGUF type name for k/i-quants
```

AWQ/GPTQ/GGUF are post-training formats — they need the *trained* checkpoint, not the base
model, so setting this just prints the exact `soup export` command to run once training
finishes, rather than pretending to quantize a model that hasn't been trained yet. QAT
points at the existing BitNet trainer (`soup train --trainer bitnet`) instead of a generic,
unvalidated `torch.ao.quantization` hook. `custom_quant_detail` is validated against the
real format registry in `soup_cli/utils/gguf_quant.py` — the same one the WebUI's dropdown
is populated from, so the two can't drift apart.

### `soup compress` — optional model-density tools

None of this runs automatically, and nothing writes a checkpoint without `--apply`.

- **`soup compress importance --model <path>`** — ranks output neurons by weight
  magnitude (default, streamed, no data needed) or `--metric wanda` (activation-weighted,
  needs the model loaded + `--calibration-data`, catches neurons whose large weights
  cancel out on real inputs).
- **`soup compress neurons --model <path> [--apply --output-dir <dir>]`** — finds
  MLP/FFN intermediate neurons that are near-duplicates (gate_proj **and** up_proj rows
  both highly similar — that joint condition is what makes the merge safe without
  calibration data) and merges them, folding the dropped neuron's output contribution into
  the kept one's. `--eval-data` compares perplexity before/after on real text as part of
  the same run.
- **`soup compress svd --model <path> --mode denoise|factorize [--apply --output-dir <dir>]`**
  — `denoise` replaces a matrix with its best low-rank reconstruction at the *same shape*
  (always loads with stock `transformers`, doesn't shrink the file, removes low-signal
  noise — the same idea `soup spectrum` already uses for SNR-based LoRA targeting, applied
  to compression). `factorize` splits a matrix into two smaller ones for a genuine size
  reduction; the result needs custom loading code (`svd_manifest.json` documents exactly
  how) and is not loadable by plain `AutoModelForCausalLM.from_pretrained`.
- **`soup compress distill-config --student <compressed> --teacher <original>`** — Soup
  already has a full token/sequence-level distillation trainer (`task: distill`,
  `soup_cli/trainer/distill.py`); this generates a ready-to-run config pointing it at your
  compressed model as the student and the pre-compression original as the teacher, for a
  quality-recovery pass. Doesn't reimplement training — just wires the existing trainer to
  the natural next step after pruning/merging.

Scope, stated plainly: MLP/FFN neurons only (attention-head merging has more structure —
per-head grouping, GQA, RoPE — and isn't attempted here); merge quality is bounded, not
guaranteed (measured on synthetic weights: ~0.3% median output error at 0.998 similarity,
~1.3% at 0.97 — `--eval-data` lets you check on your own text instead of trusting that
number).

<details>
<summary>Previous release — v0.72.4, align on a laptop (DPO / ORPO / SimPO / KTO over layer streaming)</summary>

Layer streaming used to support supervised fine-tuning only; v0.72.4 opened it to the
preference losses. The risk was one thing: DPO needs a reference model, and a second copy
would double memory and defeat the point. Soup uses *the same streamed base with its
adapters switched off* — measured at **0.914×** the SFT peak, where forcing a real second
instance cost **+730 MB, exactly one copy of the weights**. Bit-exact against a normal
non-streamed run for all four. Honest cost: free in *memory*, not in *time* — DPO reads the
layer stack **1.52×** as often per step. `grpo` / `ppo` stay excluded on purpose.

> **Trained with `stream_layers: true` on v0.72.0?** That adapter is inert — its tensors were
> saved under keys with an extra `.inner.` segment, so every loader returned the untuned base.
> Fixed in v0.72.1; re-run or re-save. Check with:
> `python -c "from safetensors.torch import load_file; print([k for k in load_file('adapter_model.safetensors') if '.inner.' in k][:3])"`

</details>

<details>
<summary>Previous release — v0.71.40, soup reward synth (generate a reward verifier from your data)</summary>

Point `soup reward synth` at a JSONL of reference outputs and it infers a deterministic verifier,
writes a readable / committable `.py` reward function, and — the part nobody else does — *refuses* to
emit one that can't tell your references from bad answers (four families: `numeric` / `json_schema` /
`regex` / `tool_call`; a mandatory calibration report is the moat). Reward ensembles
(`reward_fn: "accuracy,format"`) also train now. (#311)

```bash
soup reward synth references.jsonl -o reward.py --output-report calib.json
```

</details>

<details>
<summary>Previous release — v0.71.39, CI for weights not prompts (emit + provenance-bind the ship verdict)</summary>

`soup ship`'s verdict became emittable, committable, and provenance-bound: `--emit-evidence` makes a
run replay into an identical verdict, `eval.ship` in `soup.yaml` + `--config` makes the gate policy
reviewable, and `--config` binds evidence to the exact recipe that produced it (stale evidence → exit 3).
`soup ship --push owner/repo#N` posts the SHIP / DON'T-SHIP card on the PR.

</details>

<details>
<summary>Previous release — v0.71.38, The gate grows teeth (real leg-2 regression gate)</summary>

`soup ship`'s regression leg became real: a fixed, extraction-based scorer over seven bundled,
offline suites (MCQ · arithmetic · tool-calling · JSON validity · safety/refusal). A tune that
wins your task but quietly breaks tool-calling now gets a **DON'T SHIP**. Zero new deps.

```bash
soup ship --base ./base --adapter ./my-lora --task-eval my_task.jsonl
#   exit 0 = SHIP · 2 = DON'T SHIP · 3 = bad flags · 1 = runtime error
```

</details>

Full history: [CHANGELOG.md](CHANGELOG.md).

## Quick Start

### 1. Install

```bash
git clone <this-repository-url>
cd soup

# Light core: CLI + config + data tools, no PyTorch
pip install -e "."

# Add the training stack (torch, transformers, peft, trl, datasets, …)
pip install -e ".[train]"

# Everything (train + serve + ui + data) in one shot
pip install -e ".[all]"
```

The full extras table (`fast`, `mlx`, `serve`, `eval`, `ui`, `vision`, `audio`, …) lives in
[`docs/models.md`](docs/models.md#optional-extras).

> **Double quotes, not single.** `".[train]"` is the only spelling that works in every
> shell — `cmd.exe`, PowerShell, bash and zsh. If you copied `'.[train]'` from an older
> tutorial and pip rejected it, that is the reason:
> [why, and the exact error](docs/models.md#quoting-the-extra).

`soup init`, `soup data …`, and the other data/inspection commands work on the light install.
Fine-tuning (`soup train`) needs the `[train]` extra.

### 2. Create a config

```bash
soup init                       # interactive wizard
soup init --template chat       # or start from a template
```

Templates: `chat`, `code`, `tool-calling`, `medical`, `reasoning`, `vision`, `kto`, `orpo`,
`simpo`, `ipo`, `bco`, `rlhf`, `pretrain`, `moe`, `longcontext`, `embedding`, `audio`.

### 3. Train, test, ship

```bash
soup train --config soup.yaml                 # LoRA, quantization, batching — all handled
soup chat  --model ./output                    # talk to your model
soup push  --model ./output --repo you/my-model

soup merge  --adapter ./output                              # merge LoRA into the base
soup export --model ./output --format gguf --quant q4_k_m   # GGUF for Ollama / llama.cpp
```

More export targets (ONNX, TensorRT, AWQ, GPTQ, BitNet) and deployment options live in
[`docs/serving-and-export.md`](docs/serving-and-export.md).

## Configuration

A complete `soup.yaml`:

```yaml
base: meta-llama/Llama-3.1-8B-Instruct
task: sft
# backend: unsloth  # 2-5x faster, pip install -e ".[fast]"

data:
  train: ./data/train.jsonl
  format: alpaca
  val_split: 0.1

training:
  epochs: 3
  lr: 2e-5
  batch_size: auto
  lora:
    r: 64
    alpha: 16
  quantization: 4bit

output: ./output
```

`config/schema.py` is the single source of truth for every field. Advanced data, training,
and PEFT options are documented under [Documentation](#documentation).

## Documentation

The full feature reference lives in [`docs/`](docs/). Start here:

| Guide | Covers |
|---|---|
| [Training tasks & methods](docs/training.md) | SFT, DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/BCO, tool-calling, PRM, pre-training, distillation, classification, vision/audio/TTS, unlearning, RAFT/RA-DIT, loop-hardening detectors |
| [PEFT, long context & efficiency](docs/peft-and-efficiency.md) | DoRA, LoRA+, rsLoRA, VeRA, OLoRA, NEFTune, PiSSA, ReLoRA, optimizer & PEFT zoo, LLaMA Pro, GaLore, YaRN/LongLoRA, packing, curriculum, auto-tuning |
| [Performance & quantization](docs/performance-and-quantization.md) | QAT, FP8, Quant Menu (I + II), KV-cache, NVFP4, save formats, Cut Cross-Entropy, gradient checkpointing, kernels, activation offloading, layer streaming, multi-GPU / DeepSpeed / FSDP |
| [Data engineering](docs/data.md) | Formats, the Axolotl/LF-parity pipeline, data tools, synthetic generation & forge, quality scorecards, trace tooling, remote datasets, mixing, recipe DAGs |
| [Evaluation & probes](docs/evaluation.md) | Eval design/gate, eval-gated training, benchmarks, NLG metrics, calibration, Elo arena, diagnose, post-train X-ray probes, A/B, drift, tunability, `soup advise` |
| [Serving & export](docs/serving-and-export.md) | OpenAI-compatible server, batch inference, benchmarking, merge/export, Anthropic Messages endpoint, speculative decoding (train + measure your own draft), deploy autopilot, Web UI, Agent Forge |
| [Adapters, registry & governance](docs/adapters-and-governance.md) | Adapter lifecycle/management, model registry, Soup Cans, the data flywheel (`soup loop`), knowledge editing, steering, supply-chain controls (scan/sign/BOM/attest/audit/airgap) |
| [Compliance & governance quickstart](docs/compliance.md) | HIPAA/SOC2/EU-AI-Act/SR-11-7 `init` templates, provenance (BOM/attest/repro-receipt), audit log, air-gap, model-card autogen (`soup card`), CI gate (`soup ci init`) |
| [Backends, platform & ops](docs/backends-and-ops.md) | MLX/Unsloth backends, alternative hubs, HF Hub integration, autopilot, experiment tracking, plan/apply, env lockfiles, hardware-fit, completions, plugins, utility commands |
| [Command reference](docs/commands.md) | The full `soup` command list |
| [Supported models & extras](docs/models.md) | Recommended model families, the VRAM size guide, the pip extras matrix |

## Data Formats

Alpaca, ShareGPT, ChatML, preference pairs (DPO / ORPO / SimPO / IPO / KTO), vision, audio,
ASR, plaintext, embedding, RAFT and more — all auto-detected from JSONL, JSON, CSV, Parquet or
TXT, so in most cases you point `data.train` at a file and nothing else changes. Schemas with a
worked example per format, plus the data pipeline (remote URIs, streaming, sharding,
interleaving, vocab expansion, document ingestion), are in
[`docs/data.md`](docs/data.md#data-formats).

## Common Commands

```bash
soup train  --config soup.yaml        # train (SFT/DPO/GRPO/PPO/KTO/ORPO/SimPO/IPO/...)
soup infer  --model ./output --input prompts.jsonl   # batch inference
soup chat   --model ./output          # interactive chat
soup serve  --model ./output          # OpenAI-compatible API server
soup merge  --adapter ./output        # merge LoRA into the base model
soup export --model ./output --format gguf           # export for deployment
soup eval   benchmark --model ./output               # evaluate
soup data   inspect ./data/train.jsonl               # dataset stats
soup recipes list                     # 100+ ready-made model recipes
soup autopilot --model <id> --data d.jsonl --goal chat  # zero-config
soup doctor                           # check GPU / deps / environment
```

The complete command list is in [`docs/commands.md`](docs/commands.md).

## Supported Models

Soup works with **any** text-generation model on the
[HuggingFace Hub](https://huggingface.co/models?pipeline_tag=text-generation) — if it loads with
`AutoModelForCausalLM`, it works, zero config changes. Llama 3.x/4, Qwen 2.5/3, Gemma 3, Mistral,
Mixtral, DeepSeek R1/V3, Phi-4, and 100+ others ship as ready-made recipes (`soup recipes list`).

| VRAM | Max model (QLoRA 4-bit) | Example |
|---|---|---|
| 8 GB | ~7B | Llama-3.1-8B, Mistral-7B |
| 16 GB | ~14B | Phi-4-14B, Qwen2.5-14B |
| 24 GB | ~34B | CodeLlama-34B, Yi-1.5-34B |
| 48 GB | ~70B | Llama-3.3-70B |
| 80 GB+ | 70B+ (full) or MoE | Mixtral-8x22B, DeepSeek-V3 |

Full model + vision tables and the optional-extras matrix are in [`docs/models.md`](docs/models.md).

## Docker

Run Soup without installing CUDA or PyTorch locally — build the image from this repo (see
`Dockerfile.ui` for the Web UI image, `./start_ui.sh` to build and run it in one step):

```bash
docker build -f Dockerfile.ui -t soup .
docker run --gpus all -v $(pwd):/workspace soup train --config soup.yaml
docker compose up   # or build locally
```

## Requirements

- Python 3.10, 3.11 or 3.12 (those are the versions CI tests; 3.13+ is not supported yet
  because the PyTorch stack has not been validated there)
- GPU with CUDA (recommended), Apple Silicon (MPS), or CPU (experimental — very slow)
- 8 GB+ VRAM for 7B models with QLoRA

All training tasks run on CPU for testing (quantization auto-disabled). Optional extras
(`train`, `all`, `fast`, `vision`, `qat`, `serve`, `serve-fast`, `ui`, `eval`, `deepspeed`,
`liger`, `mlx`, `onnx`, `tensorrt`, …) are listed in
[`docs/models.md`](docs/models.md#optional-extras).

## Troubleshooting

```bash
soup doctor    # GPU, system resources, dependencies, and version in one place
```

- **`ImportError: DLL load failed while importing _C` (Windows)** — reinstall PyTorch for your
  CUDA version: `pip install torch --index-url https://download.pytorch.org/whl/cu121`.
- **`soup version` ≠ `pip show soup-cli`** — multiple Python installs; use a virtualenv.

## Development

```bash
git clone <this-repository-url>
cd soup
pip install -e ".[dev]"

ruff check src/soup_cli/ tests/    # lint
pytest tests/ -v                   # unit tests (fast, no GPU)
pytest tests/ -m smoke -v          # smoke tests (downloads a tiny model, trains)

pre-commit install                 # optional: ruff lint+format on commit
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow and [SECURITY.md](SECURITY.md) to
report a vulnerability.

## Contributing

Soup is Apache-2.0 and free — and stays that way. See [CONTRIBUTING.md](CONTRIBUTING.md) for
the full workflow and [SECURITY.md](SECURITY.md) to report a vulnerability. Bugs and feature
requests belong in the issue tracker; the [Code of Conduct](CODE_OF_CONDUCT.md) applies
throughout.

## License

[Apache-2.0](LICENSE). Copyright © the Soup contributors.
