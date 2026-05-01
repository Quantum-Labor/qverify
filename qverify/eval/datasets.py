"""Dataset loaders for benchmark evaluation.

Three loaders, one per dataset, each yielding :class:`DatasetExample`
instances. A loader either reads from a fixture path supplied by the
caller or falls back to ``~/.cache/qverify/datasets/<name>/<split>.json``.
Network downloads are out of scope for this module: a missing cache file
raises :class:`FileNotFoundError` with a hint on how to populate it. Tests
exercise loaders against shipped fixtures so CI never touches the network.

The fixture / cache schema is a JSON list of objects with this shape:

    {
      "id": "ex_001",
      "premises": ["Bird(robin)", "All robins are birds."],
      "hypothesis": "Bird(robin)",
      "label": "consistent",
      "source": "proofwriter",
      "rendered_cnf": {
        "clauses": [
          [{"predicate": "Bird", "args": ["robin"], "negated": false}]
        ]
      }
    }

``rendered_cnf`` is optional; when present, the runner can skip the
translator (and therefore the GPU). Phase 6 ships fixtures with
``rendered_cnf`` populated so reproducibility does not require a GPU.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from qverify.translator.cnf import CNF, Clause
from qverify.translator.cnf import Literal as CNFLiteral

DatasetLabel = Literal["consistent", "inconsistent"]
"""Two-class label used by the consistency-mode verifier.

The runner drops examples whose original gold label is "unknown" because
consistency mode does not distinguish "unentailed" from "consistent with
the negation".
"""

_CACHE_ROOT = Path.home() / ".cache" / "qverify" / "datasets"


@dataclass(frozen=True)
class DatasetExample:
    """One benchmark example after label normalization."""

    id: str
    premises: tuple[str, ...]
    hypothesis: str
    label: DatasetLabel
    source: str
    rendered_cnf: CNF | None = field(default=None)


def _resolve_cache_path(dataset_name: str, split: str) -> Path:
    """Return the conventional cache file for a (dataset, split) pair."""
    return _CACHE_ROOT / dataset_name / f"{split}.json"


def _parse_rendered_cnf(raw: Any) -> CNF | None:
    """Decode a rendered_cnf dict into a :class:`CNF`. Tolerant of None."""
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(f"rendered_cnf must be an object, got {type(raw).__name__}")
    raw_clauses = raw.get("clauses")
    if not isinstance(raw_clauses, list):
        raise ValueError("rendered_cnf.clauses must be a list")
    clauses: list[Clause] = []
    for clause_raw in raw_clauses:
        if not isinstance(clause_raw, list):
            raise ValueError("each clause must be a list of literals")
        literals: list[CNFLiteral] = []
        for lit_raw in clause_raw:
            if not isinstance(lit_raw, dict):
                raise ValueError("each literal must be an object")
            literals.append(
                CNFLiteral(
                    predicate=lit_raw["predicate"],
                    args=tuple(lit_raw.get("args", ())),
                    negated=bool(lit_raw.get("negated", False)),
                )
            )
        clauses.append(Clause(literals=tuple(literals)))
    return CNF(clauses=tuple(clauses))


def _coerce_example(record: dict[str, Any], default_source: str) -> DatasetExample:
    """Decode one record dict into a :class:`DatasetExample`. Raises on bad records."""
    label = record["label"]
    if label not in ("consistent", "inconsistent"):
        raise ValueError(f"label must be 'consistent' or 'inconsistent', got {label!r}")
    return DatasetExample(
        id=str(record["id"]),
        premises=tuple(record["premises"]),
        hypothesis=str(record["hypothesis"]),
        label=label,
        source=str(record.get("source", default_source)),
        rendered_cnf=_parse_rendered_cnf(record.get("rendered_cnf")),
    )


def _iter_records(
    *,
    dataset_name: str,
    split: str,
    path: Path | None,
    max_examples: int | None,
) -> Iterator[DatasetExample]:
    """Yield examples from ``path`` or from the conventional cache location.

    Skips records that fail to decode and counts them in
    ``_iter_records.skipped`` (a function attribute so tests can read it).
    """
    resolved = path if path is not None else _resolve_cache_path(dataset_name, split)
    if not resolved.exists():
        raise FileNotFoundError(
            f"No cached fixture for {dataset_name}/{split} at {resolved}. "
            f"Place a JSON list of examples there, or pass an explicit path."
        )

    raw = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError(f"{resolved}: expected a list of records, got {type(raw).__name__}")

    skipped = 0
    yielded = 0
    for record in raw:
        if max_examples is not None and yielded >= max_examples:
            break
        try:
            example = _coerce_example(record, default_source=dataset_name)
        except (KeyError, ValueError, TypeError):
            skipped += 1
            continue
        yielded += 1
        yield example

    _iter_records.skipped = skipped  # type: ignore[attr-defined]


def load_proofwriter(
    *,
    split: str = "validation",
    depth: int = 1,
    max_examples: int | None = None,
    path: Path | None = None,
) -> Iterator[DatasetExample]:
    """Yield ProofWriter (CWA) examples. ``depth`` is informational here."""
    del depth  # routed through the cache path by callers, no runtime filter
    yield from _iter_records(
        dataset_name="proofwriter",
        split=split,
        path=path,
        max_examples=max_examples,
    )


def load_ruletaker(
    *,
    split: str = "validation",
    depth: int = 1,
    max_examples: int | None = None,
    path: Path | None = None,
) -> Iterator[DatasetExample]:
    """Yield RuleTaker default-split examples."""
    del depth
    yield from _iter_records(
        dataset_name="ruletaker",
        split=split,
        path=path,
        max_examples=max_examples,
    )


def load_folio(
    *,
    split: str = "validation",
    max_examples: int | None = None,
    path: Path | None = None,
) -> Iterator[DatasetExample]:
    """FOLIO is intentionally not included in v0.1.

    Raises :class:`NotImplementedError` because FOLIO ships under CC BY-SA
    4.0, whose share-alike clause would force any derivative benchmark
    report into CC BY-SA. This conflicts with the project's Apache 2.0
    licensing default and is out of scope for v0.1. The signature is kept
    for forward-compatibility so callers can be migrated cleanly when this
    constraint is revisited; see ``benchmarks/LICENSE-DATA.md`` for the
    reasoning.
    """
    del split, max_examples, path
    raise NotImplementedError(
        "FOLIO is not included in v0.1 due to CC BY-SA share-alike licensing "
        "constraints; see docs/benchmarks.md"
    )
    yield  # pragma: no cover - keeps the function a generator for typing


_PROOFWRITER_HF_NAME = "tasksource/proofwriter"
_RULETAKER_HF_NAME = "tasksource/ruletaker"


def _download_root(dataset_name: str, depth: int) -> Path:
    """Per-dataset cache root used by the download functions."""
    return _CACHE_ROOT / dataset_name / f"depth-{depth}"


def _label_from_truth(truth: Any) -> DatasetLabel | None:
    """Map a HuggingFace dataset's truth field to the two-class label.

    True / "True" / 1 -> consistent (hypothesis follows from premises).
    False / "False" / 0 -> inconsistent.
    Anything else (Unknown, None, NaN) -> None, signalling "drop".
    """
    if isinstance(truth, bool):
        return "consistent" if truth else "inconsistent"
    if isinstance(truth, int) and truth in (0, 1):
        return "consistent" if truth == 1 else "inconsistent"
    if isinstance(truth, str):
        normalized = truth.strip().lower()
        if normalized in ("true", "yes", "entail", "entails", "entailed"):
            return "consistent"
        if normalized in ("false", "no", "contradict", "contradicts"):
            return "inconsistent"
    return None


def _split_theory_into_premises(theory: str) -> list[str]:
    """Split a ProofWriter/RuleTaker `theory` string into individual premises."""
    if not theory:
        return []
    parts = [p.strip() for p in theory.replace("\n", " ").split(".")]
    return [p + "." for p in parts if p]


def _hf_record_to_examples(
    record: dict[str, Any], example_prefix: str, source: str
) -> list[dict[str, Any]]:
    """Convert one HF record into zero or more fixture-shaped example dicts.

    ProofWriter and RuleTaker share the same shape: a `theory` string plus a
    `questions` list whose entries each carry `text` and a truth label
    (`label` or `answer` depending on the configuration).
    """
    theory = record.get("theory") or record.get("context") or ""
    premises = _split_theory_into_premises(theory)
    questions = record.get("questions") or []

    out: list[dict[str, Any]] = []
    for q_idx, question in enumerate(questions):
        if not isinstance(question, dict):
            continue
        hypothesis = question.get("text") or question.get("question") or ""
        truth = question.get("label", question.get("answer"))
        label = _label_from_truth(truth)
        if label is None:
            continue
        qid = question.get("id") or f"{example_prefix}_q{q_idx}"
        out.append(
            {
                "id": str(qid),
                "premises": premises,
                "hypothesis": hypothesis,
                "label": label,
                "source": source,
            }
        )
    return out


def _materialize(
    *,
    dataset_name: str,
    hf_name: str,
    config_name: str,
    splits: tuple[str, ...],
    depth: int,
    force: bool,
) -> Path:
    """Download a HuggingFace dataset and write per-split JSON files."""
    cache_dir = _download_root(dataset_name, depth)
    cache_dir.mkdir(parents=True, exist_ok=True)

    if not force:
        existing = [cache_dir / f"{s}.json" for s in splits]
        if all(p.exists() for p in existing):
            return cache_dir

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError(
            "The `datasets` package is required for downloads. "
            "Install with `pip install datasets>=2.20`."
        ) from exc

    for split in splits:
        try:
            ds = load_dataset(hf_name, config_name, split=split)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load {hf_name}/{config_name} split={split}: {exc}"
            ) from exc

        records: list[dict[str, Any]] = []
        for i, raw in enumerate(ds):
            assert isinstance(raw, dict)
            prefix = f"{dataset_name}_{split}_{i}"
            records.extend(_hf_record_to_examples(raw, prefix, source=dataset_name))

        out_path = cache_dir / f"{split}.json"
        out_path.write_text(json.dumps(records, indent=2), encoding="utf-8")

    return cache_dir


def download_proofwriter(
    *,
    depth: int = 1,
    force: bool = False,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> Path:
    """Download ProofWriter (CWA) at the requested depth into the cache.

    Returns the cache directory path. ``force=True`` re-downloads even if
    JSON files for every requested split already exist.
    """
    return _materialize(
        dataset_name="proofwriter",
        hf_name=_PROOFWRITER_HF_NAME,
        config_name=f"CWA_depth-{depth}",
        splits=splits,
        depth=depth,
        force=force,
    )


def download_ruletaker(
    *,
    depth: int = 1,
    force: bool = False,
    splits: tuple[str, ...] = ("train", "validation", "test"),
) -> Path:
    """Download RuleTaker at the requested depth into the cache."""
    return _materialize(
        dataset_name="ruletaker",
        hf_name=_RULETAKER_HF_NAME,
        config_name=f"depth-{depth}",
        splits=splits,
        depth=depth,
        force=force,
    )


def download_all(*, depth: int = 1, force: bool = False) -> dict[str, Path]:
    """Download both ProofWriter and RuleTaker."""
    return {
        "proofwriter": download_proofwriter(depth=depth, force=force),
        "ruletaker": download_ruletaker(depth=depth, force=force),
    }


def last_skip_count() -> int:
    """Return the skip count from the most recent loader call.

    Loaders intentionally swallow malformed records so a single bad row
    does not abort a long benchmark run. Callers that care about data
    quality can read this value after iteration completes.
    """
    return getattr(_iter_records, "skipped", 0)
