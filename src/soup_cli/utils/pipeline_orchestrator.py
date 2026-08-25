"""Pipeline orchestrator — runs training.pipeline's stages in fixed order.

This session's addition: `training.pipeline.{activation_scan,compress,distill}`
(see config/schema.py's PipelineConfig) lets a single training YAML declare
"scan for importance, then compress, then distill, then train" instead of
those being separate manual CLI invocations against whatever checkpoint the
user remembered to point them at.

Deliberately NOT auto-invoked inside `soup train`. Two reasons:

1. `activation_scan` (Wanda) and `compress` (merge/SVD) call into
   soup_cli.utils.neuron_compress / svd_compress — real, correctly-wired
   functions, but ones with NO pre-existing test coverage anywhere in this
   codebase as of this session (checked: zero hits for
   apply_merges_to_checkpoint/find_merge_candidates/analyze_svd/
   apply_svd_to_checkpoint across tests/). Auto-running an untested,
   irreversible checkpoint rewrite as a silent side effect of `soup train`
   is a worse failure mode than requiring one extra explicit step.
2. compress writes a NEW checkpoint directory — the user should get to look
   at it (or at least see the merge/rank-reduction summary this prints)
   before a possibly-long, possibly-expensive training run starts against
   it, not discover a bad compression choice after the fact.

So this is exposed as its own CLI surface (`soup pipeline run
<config.yaml>`, see commands/pipeline.py) that a person runs deliberately.
Its output is a checkpoint directory + a suggested `base:` override — wiring
THAT into `soup train` automatically remains a natural follow-up once this
path has real mileage on it.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from soup_cli.config.schema import SoupConfig


@dataclass(frozen=True)
class StageResult:
    stage: str
    ran: bool
    detail: str
    output_path: Optional[str] = None


def run_pipeline_stages(
    cfg: "SoupConfig",
    *,
    checkpoint_dir: Optional[str] = None,
) -> List[StageResult]:
    """Run every enabled stage in `cfg.training.pipeline`, in fixed order:
    activation_scan -> compress -> distill (position-marker only, see
    DistillStageConfig).

    Args:
        cfg: a validated SoupConfig. If `cfg.training.pipeline` is None or
            every stage is unset/disabled, returns `[]` immediately —
            existing configs that don't mention `pipeline:` are completely
            unaffected by this function existing.
        checkpoint_dir: local directory of the checkpoint to scan/compress.
            Defaults to `cfg.base` — this only works if `cfg.base` is
            already a local directory (a HF Hub model id needs resolving to
            a local snapshot first; that resolution is deliberately left to
            the caller via this parameter rather than done implicitly here,
            since downloading multi-GB weights is not something a "run the
            configured pipeline stages" call should do as a surprise side
            effect).

    Returns:
        One StageResult per stage that was actually enabled, in the order
        they ran. A stage that's unset or `enabled=False` is skipped
        entirely (no StageResult for it) rather than reported as a no-op,
        so `len(results)` is a direct answer to "how many stages ran".

    Raises:
        ValueError: checkpoint_dir doesn't exist, or a stage's own
            operation fails (e.g. no merge candidates at the given
            threshold) — propagated as-is from the underlying
            neuron_compress/svd_compress function, not wrapped, so the
            caller sees the real error.
    """
    pipeline = cfg.training.pipeline
    if pipeline is None:
        return []

    weights_dir = checkpoint_dir or cfg.base
    results: List[StageResult] = []
    current_dir = weights_dir

    if pipeline.activation_scan is not None and pipeline.activation_scan.enabled:
        results.append(_run_activation_scan(cfg, current_dir))

    if pipeline.compress is not None and pipeline.compress.enabled:
        compress_result = _run_compress(cfg, current_dir)
        results.append(compress_result)
        if compress_result.output_path:
            # Downstream stages (distill's position check, and — once a
            # caller wires it — the actual train stage) operate on the
            # compressed output, not the original checkpoint.
            current_dir = compress_result.output_path

    if pipeline.distill is not None and pipeline.distill.enabled:
        # Position-marker only (see DistillStageConfig docstring + the
        # schema-level `_validate_pipeline_stages` check that task='distill'
        # is actually set). Nothing to execute here — the real distillation
        # trainer runs as the normal train stage, against `current_dir`.
        results.append(
            StageResult(
                stage="distill",
                ran=True,
                detail=(
                    f"distill stage acknowledged — actual distillation runs "
                    f"as the normal train stage (task='distill') against "
                    f"{current_dir}"
                ),
                output_path=current_dir,
            )
        )

    return results


def _run_activation_scan(cfg: "SoupConfig", weights_dir: str) -> StageResult:
    stage_cfg = cfg.training.pipeline.activation_scan
    output_dir = cfg.output or "."
    output_path = stage_cfg.output_path or os.path.join(
        output_dir, "pipeline_activation_scan.json"
    )

    if stage_cfg.metric == "wanda":
        from soup_cli.utils.neuron_compress import (
            load_calibration_texts,
            rank_importance_wanda,
        )

        calib_sources = cfg.data.calibration or cfg.data.train
        if isinstance(calib_sources, str):
            calib_sources = [calib_sources]
        texts = load_calibration_texts(
            list(calib_sources),
            max_samples_per_file=stage_cfg.calibration_samples_per_dataset,
        )
        if not texts:
            raise ValueError(
                "activation_scan metric='wanda' found no usable calibration "
                "text from data.calibration / data.train"
            )
        results = rank_importance_wanda(
            weights_dir, texts, modules=stage_cfg.modules, max_length=stage_cfg.max_length
        )
        report: Dict[str, Any] = {
            "metric": "wanda",
            "n_calibration_samples": len(texts),
            "results": [
                {
                    "param_name": r.param_name,
                    "group": r.group,
                    "module_type": r.module_type,
                    "bottom_k_indices": _bottom_k_indices(r.row_norms, stage_cfg.bottom_k),
                }
                for r in results
            ],
        }
    else:
        from soup_cli.utils.neuron_compress import rank_importance

        results = rank_importance(weights_dir, modules=stage_cfg.modules)
        report = {
            "metric": "magnitude",
            "results": [
                {
                    "param_name": r.param_name,
                    "group": r.group,
                    "module_type": r.module_type,
                    "bottom_k_indices": _bottom_k_indices(r.row_norms, stage_cfg.bottom_k),
                }
                for r in results
            ],
        }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    return StageResult(
        stage="activation_scan",
        ran=True,
        detail=f"{stage_cfg.metric} scan over {len(report['results'])} matrices",
        output_path=output_path,
    )


def _bottom_k_indices(row_norms: "tuple[float, ...]", k: int) -> List[int]:
    """Indices of the k lowest-norm rows — the merge/prune candidates."""
    order = sorted(range(len(row_norms)), key=lambda i: row_norms[i])
    return order[: min(k, len(order))]


def _run_compress(cfg: "SoupConfig", weights_dir: str) -> StageResult:
    stage_cfg = cfg.training.pipeline.compress
    output_dir = stage_cfg.output_dir or os.path.join(
        cfg.output or ".", "pipeline_compressed_checkpoint"
    )

    if stage_cfg.strategy == "merge":
        from soup_cli.utils.neuron_compress import (
            apply_merges_to_checkpoint,
            find_merge_candidates,
        )

        candidates = find_merge_candidates(weights_dir, threshold=stage_cfg.merge_threshold)
        summary = apply_merges_to_checkpoint(
            weights_dir,
            output_dir,
            candidates,
            allow_nonuniform=stage_cfg.merge_allow_nonuniform,
        )
        detail = f"merged {len(summary)} MLP layer(s) at threshold={stage_cfg.merge_threshold}"
    else:
        from soup_cli.utils.svd_compress import analyze_svd, apply_svd_to_checkpoint

        analyses = analyze_svd(
            weights_dir, energy_thresholds=(stage_cfg.svd_energy_threshold,)
        )
        plan = {
            a.param_name: a.rank_at_energy[stage_cfg.svd_energy_threshold]
            for a in analyses
            if stage_cfg.svd_energy_threshold in a.rank_at_energy
        }
        apply_svd_to_checkpoint(weights_dir, output_dir, plan, mode=stage_cfg.svd_mode)
        detail = (
            f"SVD-compressed {len(plan)} matrices at "
            f"energy_threshold={stage_cfg.svd_energy_threshold}, mode={stage_cfg.svd_mode}"
        )

    return StageResult(stage="compress", ran=True, detail=detail, output_path=output_dir)
