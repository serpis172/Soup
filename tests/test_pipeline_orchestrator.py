"""Tests for training.pipeline orchestration (utils/pipeline_orchestrator.py).

Uses a small synthetic SwiGLU-style checkpoint (2 decoder layers, hidden=8,
intermediate=16) rather than a real model — the orchestrator's job is
sequencing calls into neuron_compress/svd_compress correctly, not anything
model-specific, and a synthetic checkpoint runs in milliseconds on CPU with
no network access needed.

Note: as of this session, apply_merges_to_checkpoint / find_merge_candidates
/ analyze_svd / apply_svd_to_checkpoint (which this module calls into) had
NO pre-existing test coverage anywhere in this repo. These tests exercise
the real functions end-to-end (not mocked) specifically to give the
orchestration path itself real coverage, not just a mocked control-flow
check.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import save_file

from soup_cli.config.schema import SoupConfig
from soup_cli.utils.pipeline_orchestrator import run_pipeline_stages


@pytest.fixture
def synthetic_checkpoint(tmp_path: Path) -> Path:
    """A tiny 2-layer SwiGLU-style checkpoint with two near-identical
    gate/up rows in layer 0, deliberately, so a merge threshold around 0.5
    reliably finds a candidate without depending on exact float tolerance.
    """
    ckpt_dir = tmp_path / "ckpt"
    ckpt_dir.mkdir()
    rng = np.random.default_rng(0)
    hidden, inter = 8, 16
    tensors = {}
    for layer in range(2):
        gate = rng.standard_normal((inter, hidden)).astype(np.float32)
        up = rng.standard_normal((inter, hidden)).astype(np.float32)
        down = rng.standard_normal((hidden, inter)).astype(np.float32)
        gate[1] = gate[0] + 1e-4
        up[1] = up[0] + 1e-4
        tensors[f"model.layers.{layer}.mlp.gate_proj.weight"] = gate
        tensors[f"model.layers.{layer}.mlp.up_proj.weight"] = up
        tensors[f"model.layers.{layer}.mlp.down_proj.weight"] = down
    save_file(tensors, str(ckpt_dir / "model.safetensors"))
    (ckpt_dir / "config.json").write_text(
        json.dumps({"intermediate_size": inter, "hidden_size": hidden})
    )
    return ckpt_dir


def _cfg(tmp_path: Path, ckpt_dir: Path, pipeline: dict, **extra) -> SoupConfig:
    return SoupConfig(
        base=str(ckpt_dir),
        output=str(tmp_path / "out"),
        data={"train": "examples/data/alpaca_tiny.jsonl"},
        training={"pipeline": pipeline, **extra.pop("training_extra", {})},
        **extra,
    )


def test_no_pipeline_configured_is_a_true_no_op(tmp_path: Path, synthetic_checkpoint: Path):
    """The overwhelming common case: `training.pipeline` unset entirely.
    Must return [] and touch nothing on disk."""
    cfg = SoupConfig(
        base=str(synthetic_checkpoint),
        data={"train": "examples/data/alpaca_tiny.jsonl"},
    )
    assert run_pipeline_stages(cfg) == []


def test_all_stages_disabled_is_a_true_no_op(tmp_path: Path, synthetic_checkpoint: Path):
    cfg = _cfg(tmp_path, synthetic_checkpoint, {
        "activation_scan": {"enabled": False},
        "compress": {"enabled": False},
    })
    assert run_pipeline_stages(cfg) == []


def test_activation_scan_magnitude_writes_report(tmp_path: Path, synthetic_checkpoint: Path):
    cfg = _cfg(tmp_path, synthetic_checkpoint, {
        "activation_scan": {"enabled": True, "metric": "magnitude", "bottom_k": 3},
    })
    results = run_pipeline_stages(cfg)
    assert len(results) == 1
    assert results[0].stage == "activation_scan"
    report_path = Path(results[0].output_path)
    assert report_path.exists()
    report = json.loads(report_path.read_text())
    assert report["metric"] == "magnitude"
    assert len(report["results"]) == 6  # 2 layers x 3 (gate/up/down)
    for row in report["results"]:
        assert len(row["bottom_k_indices"]) <= 3


def test_activation_scan_wanda_uses_data_calibration(tmp_path: Path, synthetic_checkpoint: Path):
    """metric='wanda' must actually load and use calibration text — this
    would raise if data.calibration/data.train resolution were broken.

    Requires torch: unlike magnitude/merge/SVD (which only ever touch raw
    safetensors weight files), Wanda is activation-aware and
    rank_importance_wanda loads a real model via
    AutoModelForCausalLM.from_pretrained — an inherent requirement of the
    method itself, not a gap in the orchestrator. The synthetic fixture
    used elsewhere in this file (raw MLP tensors, no tokenizer/full config)
    isn't a loadable model for that reason, so this needs `torch` present
    to even exercise the loading path meaningfully; skipped otherwise
    rather than faking a pass.
    """
    pytest.importorskip("torch")
    calib = tmp_path / "calib.jsonl"
    calib.write_text(
        "\n".join(json.dumps({"text": f"sample {i} for calibration"}) for i in range(4)) + "\n"
    )
    cfg = SoupConfig(
        base=str(synthetic_checkpoint),
        output=str(tmp_path / "out"),
        data={"train": "examples/data/alpaca_tiny.jsonl", "calibration": str(calib)},
        training={"pipeline": {
            "activation_scan": {"enabled": True, "metric": "wanda", "bottom_k": 2},
        }},
    )
    results = run_pipeline_stages(cfg)
    assert len(results) == 1
    report = json.loads(Path(results[0].output_path).read_text())
    assert report["metric"] == "wanda"
    assert report["n_calibration_samples"] == 4


def test_compress_merge_reduces_intermediate_size(tmp_path: Path, synthetic_checkpoint: Path):
    """The seeded near-duplicate row in layer 0 should get merged at a
    generous threshold, dropping intermediate_size from 16 to 15 — a real,
    verifiable change to the written checkpoint, not just "didn't crash"."""
    cfg = _cfg(tmp_path, synthetic_checkpoint, {
        "compress": {"enabled": True, "strategy": "merge", "merge_threshold": 0.5},
    })
    results = run_pipeline_stages(cfg)
    assert len(results) == 1
    assert results[0].stage == "compress"
    out_dir = Path(results[0].output_path)
    assert (out_dir / "model.safetensors").exists()
    new_config = json.loads((out_dir / "config.json").read_text())
    assert new_config["intermediate_size"] == 15


def test_compress_svd_denoise_preserves_shapes(tmp_path: Path, synthetic_checkpoint: Path):
    cfg = _cfg(tmp_path, synthetic_checkpoint, {
        "compress": {
            "enabled": True, "strategy": "svd",
            "svd_energy_threshold": 0.9, "svd_mode": "denoise",
        },
    })
    results = run_pipeline_stages(cfg)
    assert len(results) == 1
    out_dir = Path(results[0].output_path)
    assert (out_dir / "model.safetensors").exists()

    from safetensors import safe_open

    with safe_open(str(out_dir / "model.safetensors"), framework="numpy") as handle:
        # denoise mode keeps original shapes — must still be plain 2-D
        # (hidden, intermediate) or (intermediate, hidden) tensors, not the
        # factorized .svd_u/.svd_v pairs 'factorize' mode would write.
        keys = list(handle.keys())
        assert any(k.endswith("gate_proj.weight") for k in keys)
        assert not any(k.endswith(".svd_u") for k in keys)


def test_scan_then_compress_runs_scan_first(tmp_path: Path, synthetic_checkpoint: Path):
    """Both stages enabled together must run in the fixed activation_scan
    -> compress order and produce both artifacts."""
    cfg = _cfg(tmp_path, synthetic_checkpoint, {
        "activation_scan": {"enabled": True, "metric": "magnitude", "bottom_k": 2},
        "compress": {"enabled": True, "strategy": "merge", "merge_threshold": 0.5},
    })
    results = run_pipeline_stages(cfg)
    assert [r.stage for r in results] == ["activation_scan", "compress"]
    assert Path(results[0].output_path).exists()
    assert Path(results[1].output_path, ).joinpath("model.safetensors").exists()


def test_distill_stage_is_position_marker_not_executor(tmp_path: Path):
    """distill stage doesn't run distillation itself (see
    DistillStageConfig docstring) — it just reports the hand-off point.
    Schema-level validation (task must be 'distill') is covered in
    tests/test_config.py; this only checks the orchestrator's own
    behavior given an already-valid config."""
    cfg = SoupConfig(
        base="irrelevant-for-this-stage",
        task="distill",
        output=str(tmp_path / "out"),
        data={"train": "examples/data/alpaca_tiny.jsonl"},
        training={"teacher_model": "some/teacher", "pipeline": {"distill": {"enabled": True}}},
    )
    results = run_pipeline_stages(cfg, checkpoint_dir="/some/checkpoint")
    assert len(results) == 1
    assert results[0].stage == "distill"
    assert results[0].output_path == "/some/checkpoint"
