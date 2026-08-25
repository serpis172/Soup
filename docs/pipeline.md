# Compression pipeline, multi-objective training & multi-dataset support

This page covers features added in this fork (see [README-FORK.md](../README-FORK.md) for
the full list). Everything here is opt-in — a config that doesn't mention any of these
fields behaves exactly as it did before they existed.

## The `training.pipeline` block

Three optional, ordered stages that run **before** the normal train stage:
`activation_scan` → `compress` → `distill`. The order is fixed (not configurable) because
each stage depends on the previous one's output making sense: scanning after compressing
would score an already-pruned model, and distilling before compressing wastes the
distillation run on weights about to be discarded.

```yaml
training:
  pipeline:
    activation_scan:
      enabled: true
      metric: wanda            # "magnitude" (weight-only) or "wanda" (activation-aware)
      modules: "mlp,attn"
      bottom_k: 10
      calibration_samples_per_dataset: 64
    compress:
      enabled: true
      strategy: merge           # "merge" (neuron merging) or "svd" (low-rank compression)
      merge_threshold: 0.95     # cosine-similarity threshold for merge candidates
    distill:
      enabled: true              # requires task: distill — see below
```

**Run it explicitly**, before `soup train`:

```bash
soup pipeline run config.yaml
# or, if base: isn't already a local directory:
soup pipeline run config.yaml --checkpoint-dir /path/to/local/snapshot
```

This is deliberately a separate command, not something `soup train` runs automatically.
`activation_scan`/`compress` write real, irreversible changes to a new checkpoint directory
using code paths (`neuron_compress.py`, `svd_compress.py`) that had no automated test
coverage in upstream before this fork added it here — silently rewriting a checkpoint as a
side effect of `soup train` starting felt like the wrong default until that path has real
mileage. `soup pipeline run` prints exactly what each stage did and where the result landed;
point `base:` there and run `soup train` normally.

### `activation_scan`

Ranks weight rows/neurons by importance, writes a JSON report
(`<output>/pipeline_activation_scan.json` by default). Two metrics:

- **`magnitude`**: weight-only (L2 norm of each row). No model loading beyond streaming the
  raw safetensors files — fast, no calibration data needed.
- **`wanda`**: activation-aware (Sun et al., ICLR 2024). Needs the full model loaded and
  calibration text — pulled from `data.calibration` if set (recommended: point this at a
  small representative sample, not your whole training set), falling back to `data.train`
  otherwise. Requires `torch`/`transformers`.

### `compress`

- **`strategy: merge`**: finds near-duplicate MLP intermediate neurons (cosine similarity ≥
  `merge_threshold`) and merges them, shrinking `intermediate_size`. Set
  `merge_allow_nonuniform: true` to allow a per-layer-varying width (needs custom loading
  code on the consuming side — see `apply_merges_to_checkpoint`'s docstring).
- **`strategy: svd`**: low-rank SVD compression per matrix, keeping enough singular values to
  retain `svd_energy_threshold` of the energy. `svd_mode: denoise` (default) keeps the
  original tensor shape (always stock-loadable); `svd_mode: factorize` writes real, smaller
  U/V factor tensors (needs custom loading code).

### `distill`

This does **not** reimplement distillation — that already exists in upstream Soup as
`task: distill` plus the top-level `training.distill_divergence` /
`training.distill_temperature` / `training.distill_mode` / `training.teacher_model` fields.
Setting `pipeline.distill.enabled: true` is a position marker + consistency check: it
confirms the rest of the config is actually set up for distillation (`task: distill` is
required, or config validation rejects it) so a compress-then-distill run has one place that
says "yes, this order is intentional."

## `training.objectives` — declaring more than one training domain

```yaml
training:
  objectives: [code, tool_call, reasoning]   # freely combinable
```

`code`, `tool_call`, `reasoning`, `chat`, `general` are SFT-style single-response domains —
freely combinable with each other under `task: sft` (or `task: distill`). Combining them
doesn't change the loss function; it documents which domains `data.train` mixes, for
bookkeeping, eval routing, or a model card.

`orpo` is a distinct preference-pair objective (`task: orpo`) and **cannot** be combined with
anything else: ORPO trains on `(prompt, chosen, rejected)` triplets, not the single-response
rows the SFT-style objectives assume.

```yaml
task: orpo
training:
  objectives: [orpo]     # must be alone
```

Config validation rejects `orpo` mixed with SFT-style objectives, and rejects any objective
combination whose `task` doesn't match (e.g. `objectives: [code]` under `task: dpo`).

## Multi-dataset train / val / calibration

`data.train`, `data.val`, and `data.calibration` each accept a single path/HF-dataset-name/
remote-URI, **or a list of them**:

```yaml
data:
  train:
    - ./data/code_samples.jsonl
    - ./data/chat_samples.jsonl
    - ./data/domain_corpus.jsonl
  val: ./data/held_out.jsonl        # explicit val set — replaces val_split's carve-out
  calibration:                       # feeds the activation_scan stage's wanda metric
    - ./data/calib_a.jsonl
    - ./data/calib_b.jsonl
```

Each source is loaded and format-detected independently, then concatenated in the order
given — two datasets in different formats (e.g. one Alpaca, one ShareGPT) don't need to be
pre-merged with `soup data mix` first just to train on both.

`data.val`, when set, **replaces** the `val_split`-based carve-out entirely: the validation
set is exactly what's loaded from `val:`, and `train:` is not shrunk to make room for it. Omit
`val:` to keep the original `val_split` behavior unchanged.

### Pre-training verification (`soup data inspect`, automatically)

```yaml
data:
  verify_before_training: true   # default
```

Before training starts, every **local** source in `train`/`val`/`calibration` is checked with
the same logic as `soup data inspect` — row count, duplicate ratio, empty fields — and a hard
failure (missing file, zero usable rows, a fully-degenerate file where every row is
identical) aborts before the model and optimizer are allocated, instead of after. HF dataset
names and remote URIs are skipped (nothing local to check) and reported as skipped, not
silently passed as if verified. Set `verify_before_training: false` to opt out (e.g. CI runs
against fixtures that intentionally don't look like real data).

## Quantization bit-width

`training.custom_quant_strategy` (`awq` / `gptq` / `k-quants` / `i-quants`) +
`training.custom_quant_detail`:

| Strategy | Valid `custom_quant_detail` | Why |
|---|---|---|
| `awq` | `"4"` only | `autoawq`'s activation-aware scale search is defined for a 4-bit grid — there is no 2/3/8-bit AWQ kernel in any mainstream implementation. |
| `gptq` | `"2"`, `"3"`, `"4"`, `"8"` | `auto-gptq`/GPTQModel genuinely support all four. |
| `k-quants` | a standard GGUF quant type (e.g. `"Q4_K_M"`) | |
| `i-quants` | an advanced GGUF format (e.g. `"IQ4_XS"`, `"UD-Q4_K_XL"`) | |

Same domains are enforced in `soup export --bits` and the Web UI's quantization panel, so the
three can't drift apart again.
