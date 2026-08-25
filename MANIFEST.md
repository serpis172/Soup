# How to apply this changeset

Every file in this zip keeps its path relative to the repo root — copy them over your
`Soup/` checkout, overwriting the existing files at those paths, then run the two deletions
below by hand (a zip can't represent "delete this file").

## Files in this zip (26 total: 24 modified, 2 added under new paths already shown)

Just copy the whole tree over your working copy:

```bash
cd /path/to/your/Soup
unzip -o soup-fork-changes.zip -d /tmp/soup-fork-changes
cp -r /tmp/soup-fork-changes/soup-fork-changes/* .
rm -f MANIFEST.md   # this file — not part of the repo
```

## Two files to delete manually (not in this zip — deletions, not additions)

```bash
git rm Dockerfile.ui.orig Dockerfile.ui.rej
```

These were leftover `patch`/`.orig`/`.rej` artifacts from an earlier merge, already superseded
by the real `Dockerfile.ui` — repo hygiene, unrelated to any feature above.

## After copying

```bash
pip install -e ".[train]"     # if you don't already have it installed editable
pytest tests/test_config.py tests/test_examples_configs.py tests/test_ui_config_builder.py \
       tests/test_loader.py tests/test_data.py tests/test_data_split.py \
       tests/test_awq_gptq_export.py tests/test_pipeline_orchestrator.py
```

All 140 tests in that set passed (2 skipped — one needs `transformers` optional extras
already covered if installed with the export, one needs `torch`) when this changeset was
built. The rest of the existing suite (thousands of other tests across the full repo) was
**not** re-run end-to-end here — only the files touched by this changeset and their direct
neighbors were exercised. Run your full suite before merging.

## What changed and why

See `CHANGELOG.md`'s `[Unreleased]` section (top of the file) for the complete list with
rationale, and `docs/pipeline.md` for the new `training.pipeline` / `training.objectives` /
multi-dataset features in detail. Short version:

- GPTQ export widened to 2/3/4/8-bit; AWQ correctly locked to 4-bit-only (was silently
  accepting 8 and failing later).
- Web UI: RAM Prefetch and Quantization split into separate cards/endpoints; a session-token
  recovery banner now appears instead of every button failing with an unrecoverable
  `Unauthorized`.
- Wanda calibration accepts multiple dataset files (CLI `--calibration-data` repeatable, UI
  multi-line input), pooled via a new shared `neuron_compress.load_calibration_texts`.
- `data.train` / `data.val` (new) / `data.calibration` (new) accept a single source or a
  list; `data.verify_before_training` (default on) runs `soup data inspect`-equivalent
  checks before training starts.
- `training.pipeline`: optional, ordered `activation_scan` → `compress` → `distill` stages,
  run via the new `soup pipeline run config.yaml` command (not auto-run by `soup train` —
  see `src/soup_cli/utils/pipeline_orchestrator.py`'s docstring for why).
- `training.objectives`: declare and validate combinations of training domains
  (code/tool_call/reasoning/chat/general, freely combinable; `orpo` alone only).
- Docs: README fork attribution + link to upstream, `README-FORK.md` rewritten and corrected
  (it previously described a Gradio UI that no longer matches the actual FastAPI+JS UI),
  CODE_OF_CONDUCT / SECURITY / CONTRIBUTING fork notices, `docs/pipeline.md` added.
- Removed stray `Dockerfile.ui.orig` / `.rej` patch artifacts (see deletion step above).

## What this changeset does NOT include

- Broader Web UI/UX reorganization beyond the RAM-prefetch/quantization split and the
  auth banner — the original request ("riordina la UI e UX") is partially addressed, not
  exhaustively.
- Integration testing of `training.pipeline`'s `activation_scan` (metric='wanda') against a
  real model — it requires `torch`/`transformers` and a genuinely loadable checkpoint,
  neither available in the sandbox this was built in. The `magnitude` metric and both
  `compress` strategies (`merge`, `svd`) ARE exercised end-to-end against a synthetic
  checkpoint in `tests/test_pipeline_orchestrator.py`.
