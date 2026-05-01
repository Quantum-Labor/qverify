"""Unit tests for the dataset download functions.

The HuggingFace `datasets.load_dataset` call is monkey-patched to return a
canned in-memory iterable; no network access happens in CI.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pytest

from qverify.eval import datasets as datasets_mod


def _fake_load_dataset(records: dict[str, list[dict[str, Any]]]) -> Any:
    """Return a callable that mimics datasets.load_dataset(name, config, split=...)."""

    def loader(name: str, config: str, *, split: str) -> Iterable[dict[str, Any]]:
        del name, config
        return records.get(split, [])

    return loader


_PROOFWRITER_FAKE = {
    "train": [
        {
            "theory": "Robin is a bird. Bird flies.",
            "questions": [
                {"id": "q0", "text": "Robin flies.", "label": True},
                {"id": "q1", "text": "Robin swims.", "label": False},
                {"id": "q2", "text": "Robin sings.", "label": "Unknown"},
            ],
        }
    ],
    "validation": [
        {
            "theory": "Tom is a cat.",
            "questions": [
                {"id": "v0", "text": "Tom is a cat.", "label": True},
            ],
        }
    ],
    "test": [],
}


def test_download_proofwriter_writes_split_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset(_PROOFWRITER_FAKE))

    cache_dir = datasets_mod.download_proofwriter(depth=1)
    assert cache_dir == tmp_path / "proofwriter" / "depth-1"
    for split in ("train", "validation", "test"):
        assert (cache_dir / f"{split}.json").exists()


def test_download_proofwriter_records_match_fixture_shape(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset(_PROOFWRITER_FAKE))

    cache_dir = datasets_mod.download_proofwriter(depth=1)
    train = json.loads((cache_dir / "train.json").read_text(encoding="utf-8"))
    assert isinstance(train, list)
    # Two well-labeled questions; the "Unknown" one is dropped
    assert len(train) == 2
    assert all(
        set(rec.keys()) >= {"id", "premises", "hypothesis", "label", "source"} for rec in train
    )
    assert train[0]["label"] == "consistent"
    assert train[1]["label"] == "inconsistent"
    assert train[0]["source"] == "proofwriter"


def test_download_proofwriter_premises_split_on_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset(_PROOFWRITER_FAKE))

    cache_dir = datasets_mod.download_proofwriter(depth=1)
    train = json.loads((cache_dir / "train.json").read_text(encoding="utf-8"))
    assert train[0]["premises"] == ["Robin is a bird.", "Bird flies."]


def test_download_skips_when_cached_unless_forced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    call_count = {"n": 0}

    def fake(name: str, config: str, *, split: str) -> Iterable[dict[str, Any]]:
        del name, config, split
        call_count["n"] += 1
        return []

    monkeypatch.setattr("datasets.load_dataset", fake)

    # Pre-create the expected files
    cache_dir = tmp_path / "proofwriter" / "depth-1"
    cache_dir.mkdir(parents=True)
    for split in ("train", "validation", "test"):
        (cache_dir / f"{split}.json").write_text("[]", encoding="utf-8")

    datasets_mod.download_proofwriter(depth=1, force=False)
    assert call_count["n"] == 0  # all splits cached, no calls

    datasets_mod.download_proofwriter(depth=1, force=True)
    assert call_count["n"] == 3  # forced re-download for all 3 splits


def test_download_ruletaker_uses_distinct_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        "datasets.load_dataset", _fake_load_dataset({"train": [], "validation": [], "test": []})
    )

    pw_cache = datasets_mod.download_proofwriter(depth=1)
    rt_cache = datasets_mod.download_ruletaker(depth=1)
    assert pw_cache != rt_cache
    assert "ruletaker" in str(rt_cache)
    assert "depth-1" in str(rt_cache)


def test_download_propagates_load_dataset_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)

    def boom(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("network down")

    monkeypatch.setattr("datasets.load_dataset", boom)
    with pytest.raises(RuntimeError, match="Failed to load"):
        datasets_mod.download_proofwriter(depth=1)


def test_download_all_returns_both_dataset_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr(
        "datasets.load_dataset",
        _fake_load_dataset({"train": [], "validation": [], "test": []}),
    )
    paths = datasets_mod.download_all(depth=1)
    assert set(paths.keys()) == {"proofwriter", "ruletaker"}
    assert all(isinstance(p, Path) for p in paths.values())


def test_label_mapping_handles_strings_and_ints() -> None:
    from qverify.eval.datasets import _label_from_truth

    assert _label_from_truth(True) == "consistent"
    assert _label_from_truth(False) == "inconsistent"
    assert _label_from_truth(1) == "consistent"
    assert _label_from_truth(0) == "inconsistent"
    assert _label_from_truth("True") == "consistent"
    assert _label_from_truth("FALSE") == "inconsistent"
    assert _label_from_truth("Unknown") is None
    assert _label_from_truth(None) is None


def test_loader_reads_back_downloaded_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(datasets_mod, "_CACHE_ROOT", tmp_path)
    monkeypatch.setattr("datasets.load_dataset", _fake_load_dataset(_PROOFWRITER_FAKE))
    cache_dir = datasets_mod.download_proofwriter(depth=1)

    # The existing loader, given the path of a downloaded split, should
    # parse it without modification.
    examples = list(datasets_mod.load_proofwriter(path=cache_dir / "train.json"))
    assert len(examples) == 2
    labels = sorted(e.label for e in examples)
    assert labels == ["consistent", "inconsistent"]
