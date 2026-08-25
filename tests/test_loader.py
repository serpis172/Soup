"""Tests for data loading (loader.py)."""

import json
from pathlib import Path

import pytest

from soup_cli.data.loader import load_raw_data


def test_load_jsonl(sample_alpaca_data: Path):
    """Load JSONL file should return list of dicts."""
    data = load_raw_data(sample_alpaca_data)
    assert len(data) == 3
    assert data[0]["instruction"] == "What is Python?"


def test_load_json(tmp_path: Path):
    """Load JSON array file."""
    path = tmp_path / "data.json"
    records = [
        {"instruction": "Q1", "input": "", "output": "A1"},
        {"instruction": "Q2", "input": "", "output": "A2"},
    ]
    path.write_text(json.dumps(records))
    data = load_raw_data(path)
    assert len(data) == 2
    assert data[0]["instruction"] == "Q1"


def test_load_json_not_array(tmp_path: Path):
    """JSON file with object (not array) should raise ValueError."""
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"key": "value"}))
    with pytest.raises(ValueError, match="list"):
        load_raw_data(path)


def test_load_csv(tmp_path: Path):
    """Load CSV file with headers."""
    path = tmp_path / "data.csv"
    path.write_text("instruction,input,output\nWhat is AI,,AI is...\nExplain ML,,ML is...\n")
    data = load_raw_data(path)
    assert len(data) == 2
    assert data[0]["instruction"] == "What is AI"
    assert data[1]["output"] == "ML is..."


def test_load_nonexistent_file(tmp_path: Path):
    """Loading nonexistent file should raise FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_raw_data(tmp_path / "nonexistent.jsonl")


def test_load_unsupported_extension(tmp_path: Path):
    """Unsupported file extension should raise ValueError."""
    path = tmp_path / "data.xyz"
    path.write_text("hello")
    with pytest.raises(ValueError, match="Unsupported"):
        load_raw_data(path)


def test_load_jsonl_with_empty_lines(tmp_path: Path):
    """JSONL loader should skip empty lines."""
    path = tmp_path / "data.jsonl"
    content = (
        json.dumps({"instruction": "Q1", "output": "A1"}) + "\n"
        + "\n"
        + json.dumps({"instruction": "Q2", "output": "A2"}) + "\n"
        + "\n"
    )
    path.write_text(content)
    data = load_raw_data(path)
    assert len(data) == 2


def test_load_jsonl_with_invalid_line(tmp_path: Path):
    """JSONL loader should skip invalid JSON lines with a warning."""
    path = tmp_path / "data.jsonl"
    content = (
        json.dumps({"instruction": "Q1", "output": "A1"}) + "\n"
        + "this is not json\n"
        + json.dumps({"instruction": "Q2", "output": "A2"}) + "\n"
    )
    path.write_text(content)
    data = load_raw_data(path)
    assert len(data) == 2


# --- Multi-source train/val/calibration (this session) -----------------
#
# DataConfig.train/val/calibration each accept a single path OR a list of
# paths (config/schema.py); load_dataset() concatenates every source in
# `train` and, when `val` is set, uses it verbatim instead of carving val
# out of train via val_split. These tests exercise that through the real
# `load_dataset()` entry point, not just schema validation, since the
# schema accepting a list and the loader actually using every element of
# it are two different things that could each be wrong independently.


def _write_jsonl(path: Path, n: int, prefix: str = "Q") -> Path:
    rows = [
        {"instruction": f"{prefix}{i}", "input": "", "output": f"A{i}"}
        for i in range(n)
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_load_dataset_single_train_source_unchanged(tmp_path: Path):
    """A plain string `train` (the pre-existing shape) still behaves
    exactly as before — no regression from widening the field to
    Union[str, List[str]]."""
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    f = _write_jsonl(tmp_path / "a.jsonl", 10)
    dc = DataConfig(train=str(f), val_split=0.0)
    result = load_dataset(dc)
    assert len(result["train"]) == 10


def test_load_dataset_multi_train_sources_concatenated(tmp_path: Path):
    """train: [a.jsonl, b.jsonl] loads and concatenates both, in order."""
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    a = _write_jsonl(tmp_path / "a.jsonl", 5, prefix="A")
    b = _write_jsonl(tmp_path / "b.jsonl", 7, prefix="B")
    dc = DataConfig(train=[str(a), str(b)], val_split=0.0)
    result = load_dataset(dc)
    assert len(result["train"]) == 12


def test_load_dataset_explicit_val_bypasses_split(tmp_path: Path):
    """When `val` is set, the val set is exactly what's loaded from it —
    not a val_split-based slice of `train` — and `train` is not shrunk to
    make room for it."""
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    train_f = _write_jsonl(tmp_path / "train.jsonl", 10)
    val_f = _write_jsonl(tmp_path / "val.jsonl", 4)
    dc = DataConfig(train=str(train_f), val=str(val_f), val_split=0.5)
    result = load_dataset(dc)
    assert len(result["train"]) == 10  # not halved by val_split
    assert len(result["val"]) == 4


def test_load_dataset_multi_val_sources_concatenated(tmp_path: Path):
    """`val` also accepts a list, concatenated the same way as `train`."""
    from soup_cli.config.schema import DataConfig
    from soup_cli.data.loader import load_dataset

    train_f = _write_jsonl(tmp_path / "train.jsonl", 6)
    val_a = _write_jsonl(tmp_path / "val_a.jsonl", 2)
    val_b = _write_jsonl(tmp_path / "val_b.jsonl", 3)
    dc = DataConfig(train=str(train_f), val=[str(val_a), str(val_b)])
    result = load_dataset(dc)
    assert len(result["val"]) == 5


def test_data_config_calibration_field_accepts_single_and_list(tmp_path: Path):
    """`calibration` is schema-validated the same way as train/val even
    though load_dataset() doesn't consume it directly — the pipeline's
    activation-scan stage does (see config/schema.py PipelineConfig)."""
    from soup_cli.config.schema import DataConfig

    f = _write_jsonl(tmp_path / "calib.jsonl", 3)
    dc = DataConfig(train=str(f), calibration=str(f))
    assert dc.calibration == str(f)
    dc2 = DataConfig(train=str(f), calibration=[str(f), str(f)])
    assert dc2.calibration == [str(f), str(f)]


def test_data_config_empty_source_list_rejected():
    """train/val/calibration each reject `[]` at validation time rather
    than silently producing zero rows deep in the loader."""
    from pydantic import ValidationError

    from soup_cli.config.schema import DataConfig

    with pytest.raises(ValidationError, match="cannot be an empty list"):
        DataConfig(train=[])
    with pytest.raises(ValidationError, match="cannot be an empty list"):
        DataConfig(train="x.jsonl", val=[])
    with pytest.raises(ValidationError, match="cannot be an empty list"):
        DataConfig(train="x.jsonl", calibration=[])
