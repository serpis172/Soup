# How to apply this changeset

Every file in this zip keeps its path relative to the repo root. Copy the whole tree over
your `Soup/` checkout, overwriting existing files at those paths:

```bash
cd /path/to/your/Soup
unzip -o soup-fork-changes.zip -d /tmp/soup-fork-changes
cp -r /tmp/soup-fork-changes/soup-fork-changes/* .
rm -f MANIFEST.md   # this file — not part of the repo
```

## Two files to delete manually (a zip can't represent deletions)

```bash
git rm Dockerfile.ui.orig Dockerfile.ui.rej
```

Leftover `patch`/`.orig`/`.rej` artifacts from an earlier merge, already superseded by the
real `Dockerfile.ui` — repo hygiene, unrelated to any feature below.

## After copying

```bash
pip install -e ".[train]"
pytest tests/test_config.py tests/test_examples_configs.py tests/test_ui_config_builder.py \
       tests/test_loader.py tests/test_data.py tests/test_data_split.py \
       tests/test_awq_gptq_export.py tests/test_pipeline_orchestrator.py \
       tests/test_ui_live_monitor.py tests/test_ui.py tests/test_tracker.py
```

226 tests passed, 2 skipped (one needs `torch`, one an environment-specific extra) when this
was built. The rest of the existing suite (thousands of other tests across the repo) was
**not** re-run end-to-end — only the files this changeset touches and their direct neighbors
were exercised. Run your full suite before merging.

## What changed and why

Full detail with rationale: `CHANGELOG.md`'s `[Unreleased]` section (top of the file).
Feature docs: `docs/pipeline.md`. Short version, in the order these were requested:

**Round 1 — config/pipeline/data:**
- GPTQ export widened to 2/3/4/8-bit; AWQ correctly locked to 4-bit-only (was silently
  accepting 8 and failing later, inside the quant builder, with a confusing error).
- `data.train` / `data.val` (new) / `data.calibration` (new): single source or a list;
  `data.verify_before_training` (default on) runs `soup data inspect`-equivalent checks
  before training starts.
- Wanda calibration accepts multiple dataset files, pooled via a new shared
  `neuron_compress.load_calibration_texts`.
- `training.pipeline`: optional, ordered `activation_scan → compress → distill` stages, run
  via `soup pipeline run config.yaml` (deliberately NOT auto-run by `soup train` — see
  `pipeline_orchestrator.py`'s docstring: the underlying merge/SVD functions had zero
  pre-existing test coverage in this repo before this changeset added it).
- `training.objectives`: declare/validate training-domain combinations
  (code/tool_call/reasoning/chat/general freely combinable; `orpo` alone only).

**Round 2 — Web UI (the actual "Start Training does nothing" fix + everything requested after):**
- **Root cause of "Start Training does nothing, just PID"**: two separate real bugs.
  (1) `connectTrainingSSE()`/`updateProgressBar()` existed in the code but were never called
  from `startTraining()` — now they are, immediately on start. (2) The training subprocess's
  stdout was piped but nothing read it unless someone had the logs page open; once the OS
  pipe buffer filled, the process would block on its next `print()` and hang silently,
  indistinguishable from "still training". Fixed with an always-on background drain thread.
- Pause/Resume via `SIGSTOP`/`SIGCONT` (frees compute, not VRAM — documented as such).
- `/api/train/progress` now returns phase (parsed from real log markers), and auto-discovers
  a run's live step/loss/speed/ETA from just the tracked subprocess's PID (new
  `ExperimentTracker.find_run_by_pid()` + `mark_running()` call in `soup train` itself —
  nothing in the frontend previously tracked a run_id at all).
- Compress section merged into New Training (collapsible, was a separate top-level page).
- Calculator section: estimates checkpoint size after quantization/compression, reusing the
  existing `model_size_from_name()` helper.
- Help & Tutorial page: step-by-step beginner walkthrough (8 sections — what Soup is, first
  run, what each section does, phases/pause, a quantization decision table, when Compress is
  needed, HF Hub filter semantics, common problems).
- Quick Reference moved to the bottom of New Training and expanded to match the current
  schema (was stale/partial).
- "Model Hub" renamed to "HF Hub" everywhere.
- HF Hub filter fixes: Language filter was shown on the Models tab but silently dropped
  server-side — now actually wired (via a `language:<code>` tag filter, same mechanism as
  library/license). Task field now shows tab-appropriate suggestions (models vs datasets use
  different HF taxonomies — a value valid for one silently returned zero results on the other).

## What this changeset does NOT include

- Full integration testing of `training.pipeline`'s `activation_scan` with `metric='wanda'`
  against a real model — needs `torch`/`transformers` and a genuinely loadable checkpoint,
  neither available in the sandbox this was built in. `magnitude` scan and both `compress`
  strategies (`merge`, `svd`) ARE exercised end-to-end against a synthetic checkpoint in
  `tests/test_pipeline_orchestrator.py`.
- Live, in-browser testing of the Web UI changes (no browser available in the build sandbox)
  — verified via: JS syntax checks (`node --check`), HTML tag-balance checks, FastAPI
  `TestClient` requests confirming routes register and serve expected content, and direct
  unit/integration tests of every new backend endpoint and the tracker/orchestrator logic
  those endpoints depend on. Click through it yourself before relying on it for a real run.
- An exhaustive dropdown of every valid HF Hub task/library/license/language value — the
  `<datalist>` suggestions are common values, not a restriction; anything valid on
  huggingface.co's own filters works.
