"""Tests for config loading and validation."""

from pathlib import Path

import pytest

from soup_cli.config.loader import load_config
from soup_cli.config.schema import SoupConfig


def test_load_valid_config(sample_config: Path):
    """Valid config should parse without errors."""
    cfg = load_config(sample_config)
    assert isinstance(cfg, SoupConfig)
    assert cfg.base == "meta-llama/Llama-3.1-8B-Instruct"
    assert cfg.task == "sft"
    assert cfg.training.epochs == 1
    assert cfg.training.lora.r == 8
    assert cfg.training.quantization == "4bit"


def test_config_defaults():
    """Config should fill in defaults for optional fields."""
    cfg = SoupConfig(
        base="some-model",
        data={"train": "./data.jsonl"},
    )
    assert cfg.task == "sft"
    assert cfg.training.epochs == 3
    assert cfg.training.lr == 2e-5
    assert cfg.training.batch_size == "auto"
    assert cfg.training.lora.r == 64
    assert cfg.training.quantization == "4bit"
    assert cfg.output == "./output"


def test_config_invalid_task():
    """Invalid task should raise validation error."""
    with pytest.raises(Exception):
        SoupConfig(
            base="some-model",
            task="invalid_task",
            data={"train": "./data.jsonl"},
        )


def test_config_val_split_bounds():
    """val_split must be between 0 and 0.5."""
    with pytest.raises(Exception):
        SoupConfig(
            base="some-model",
            data={"train": "./data.jsonl", "val_split": 0.9},
        )


def test_config_dpo_task():
    """DPO task config should parse and have dpo_beta default."""
    cfg = SoupConfig(
        base="some-model",
        task="dpo",
        data={"train": "./data.jsonl", "format": "dpo"},
    )
    assert cfg.task == "dpo"
    assert cfg.training.dpo_beta == 0.1


def test_config_dpo_beta_custom():
    """Custom dpo_beta should be accepted."""
    cfg = SoupConfig(
        base="some-model",
        task="dpo",
        data={"train": "./data.jsonl", "format": "dpo"},
        training={"dpo_beta": 0.5},
    )
    assert cfg.training.dpo_beta == 0.5


# --- training.objectives compatibility matrix (this session) -----------


def test_config_objectives_sft_combo_allowed():
    """SFT-style objectives (code/tool_call/reasoning/chat/general) may be
    freely combined with each other under task='sft'."""
    cfg = SoupConfig(
        base="some-model",
        task="sft",
        data={"train": "./data.jsonl"},
        training={"objectives": ["code", "tool_call", "reasoning"]},
    )
    assert cfg.training.objectives == ["code", "tool_call", "reasoning"]


def test_config_objectives_orpo_alone_allowed():
    """'orpo' is allowed on its own under task='orpo'."""
    cfg = SoupConfig(
        base="some-model",
        task="orpo",
        data={"train": "./data.jsonl"},
        training={"objectives": ["orpo"]},
    )
    assert cfg.training.objectives == ["orpo"]


def test_config_objectives_orpo_cannot_combine_with_sft_style():
    """'orpo' operates on preference triplets, not the single-response rows
    the SFT-style objectives assume — combining them must be rejected."""
    with pytest.raises(Exception, match="non è combinabile"):
        SoupConfig(
            base="some-model",
            task="orpo",
            data={"train": "./data.jsonl"},
            training={"objectives": ["orpo", "code"]},
        )


def test_config_objectives_require_matching_task():
    """SFT-style objectives declared under an incompatible task (e.g. dpo)
    must be rejected rather than silently accepted."""
    with pytest.raises(Exception, match="richiede task"):
        SoupConfig(
            base="some-model",
            task="dpo",
            data={"train": "./data.jsonl", "format": "dpo"},
            training={"objectives": ["code"]},
        )


def test_config_objectives_orpo_requires_task_orpo():
    """objectives=['orpo'] under a non-orpo task must be rejected."""
    with pytest.raises(Exception, match="richiede task='orpo'"):
        SoupConfig(
            base="some-model",
            task="sft",
            data={"train": "./data.jsonl"},
            training={"objectives": ["orpo"]},
        )


def test_config_objectives_unset_by_default():
    """objectives is opt-in — unset by default, no behavior change for
    every existing config that doesn't mention it."""
    cfg = SoupConfig(base="some-model", data={"train": "./data.jsonl"})
    assert cfg.training.objectives is None


# --- data.train/val/calibration multi-source (this session) ------------


def test_config_data_train_accepts_list():
    cfg = SoupConfig(
        base="some-model",
        data={"train": ["./a.jsonl", "./b.jsonl"]},
    )
    assert cfg.data.train == ["./a.jsonl", "./b.jsonl"]


def test_config_data_val_and_calibration_accept_single_or_list():
    cfg = SoupConfig(
        base="some-model",
        data={
            "train": "./a.jsonl",
            "val": "./val.jsonl",
            "calibration": ["./c1.jsonl", "./c2.jsonl"],
        },
    )
    assert cfg.data.val == "./val.jsonl"
    assert cfg.data.calibration == ["./c1.jsonl", "./c2.jsonl"]


def test_config_data_verify_before_training_defaults_true():
    cfg = SoupConfig(base="some-model", data={"train": "./a.jsonl"})
    assert cfg.data.verify_before_training is True
