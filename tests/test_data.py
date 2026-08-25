"""Tests for data loading, format detection, and validation."""

import json
from pathlib import Path

import pytest

from soup_cli.data.formats import detect_format, format_to_messages
from soup_cli.data.loader import load_raw_data
from soup_cli.data.validator import validate_and_stats


def test_load_jsonl(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    assert len(data) == 3
    assert "instruction" in data[0]
    assert "output" in data[0]


def test_detect_alpaca_format():
    data = [{"instruction": "test", "input": "", "output": "result"}]
    assert detect_format(data) == "alpaca"


def test_detect_sharegpt_format():
    data = [{"conversations": [{"from": "human", "value": "hi"}]}]
    assert detect_format(data) == "sharegpt"


def test_detect_chatml_format():
    data = [{"messages": [{"role": "user", "content": "hi"}]}]
    assert detect_format(data) == "chatml"


def test_convert_alpaca():
    row = {"instruction": "Explain AI", "input": "", "output": "AI is..."}
    result = format_to_messages(row, "alpaca")
    assert result is not None
    assert len(result["messages"]) == 2
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_convert_alpaca_with_input():
    row = {"instruction": "Translate", "input": "hello", "output": "привет"}
    result = format_to_messages(row, "alpaca")
    assert "hello" in result["messages"][0]["content"]


def test_convert_sharegpt():
    row = {
        "conversations": [
            {"from": "human", "value": "What is 2+2?"},
            {"from": "gpt", "value": "4"},
        ]
    }
    result = format_to_messages(row, "sharegpt")
    assert result["messages"][0]["role"] == "user"
    assert result["messages"][1]["role"] == "assistant"


def test_validate_stats(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    stats = validate_and_stats(data)
    assert stats["total"] == 3
    assert "instruction" in stats["columns"]
    assert stats["avg_length"] > 0


def test_validate_with_format(sample_alpaca_data: Path):
    data = load_raw_data(sample_alpaca_data)
    stats = validate_and_stats(data, expected_format="alpaca")
    assert stats["valid_rows"] == 3
    assert len(stats["issues"]) == 0  # no issues for valid data


def test_detect_dpo_format():
    data = [{"prompt": "What is AI?", "chosen": "AI is...", "rejected": "I don't know"}]
    assert detect_format(data) == "dpo"


def test_convert_dpo():
    row = {"prompt": "Explain gravity", "chosen": "Gravity is a force", "rejected": "No idea"}
    result = format_to_messages(row, "dpo")
    assert result is not None
    assert result["prompt"] == "Explain gravity"
    assert result["chosen"] == "Gravity is a force"
    assert result["rejected"] == "No idea"


# --- verify_dataset_sources (this session's pre-training data gate) -----


def test_verify_dataset_sources_passes_good_file(tmp_path: Path):
    from soup_cli.commands.data import verify_dataset_sources

    f = tmp_path / "good.jsonl"
    f.write_text(
        "\n".join(
            json.dumps({"instruction": f"Q{i}", "output": f"A{i}"}) for i in range(5)
        )
        + "\n"
    )
    verify_dataset_sources([str(f)], "train")  # should not raise


def test_verify_dataset_sources_rejects_missing_file(tmp_path: Path):
    from soup_cli.commands.data import verify_dataset_sources

    with pytest.raises(ValueError, match="not found"):
        verify_dataset_sources([str(tmp_path / "nope.jsonl")], "val")


def test_verify_dataset_sources_rejects_empty_file(tmp_path: Path):
    from soup_cli.commands.data import verify_dataset_sources

    f = tmp_path / "empty.jsonl"
    f.write_text("")
    with pytest.raises(ValueError, match="zero usable rows"):
        verify_dataset_sources([str(f)], "train")


def test_verify_dataset_sources_rejects_fully_degenerate_file(tmp_path: Path):
    """Every row identical => only 1 unique row => hard failure, not just a
    'Duplicates: 4' stat nobody looks at until training already burned GPU
    time on it."""
    from soup_cli.commands.data import verify_dataset_sources

    f = tmp_path / "dup.jsonl"
    f.write_text(
        "\n".join(json.dumps({"instruction": "same", "output": "same"}) for _ in range(5))
        + "\n"
    )
    with pytest.raises(ValueError, match="only 1 unique row"):
        verify_dataset_sources([str(f)], "calibration")


def test_verify_dataset_sources_allows_partial_duplicates(tmp_path: Path):
    """Some repeated rows among otherwise-real data is normal and must NOT
    be treated as the degenerate/fully-duplicated case above."""
    from soup_cli.commands.data import verify_dataset_sources

    f = tmp_path / "mixed.jsonl"
    rows = [{"instruction": "a", "output": "a"}] * 2 + [{"instruction": "b", "output": "b"}]
    f.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    verify_dataset_sources([str(f)], "train")  # should not raise


def test_verify_dataset_sources_skips_hf_and_remote_sources():
    """HF dataset names and remote URIs have nothing local to inspect —
    must be skipped, not treated as missing files."""
    from soup_cli.commands.data import verify_dataset_sources

    verify_dataset_sources(["some-org/some-dataset"], "train")  # no suffix => HF-style
    verify_dataset_sources(["s3://bucket/data.jsonl"], "train")  # remote URI
