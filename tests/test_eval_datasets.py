"""Unit tests for dataset loaders."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from qverify.eval.datasets import (
    DatasetExample,
    last_skip_count,
    load_folio,
    load_proofwriter,
    load_ruletaker,
)
from qverify.translator.cnf import CNF

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
PROOFWRITER = FIXTURES / "proofwriter_tiny.json"
RULETAKER = FIXTURES / "ruletaker_tiny.json"


def test_load_proofwriter_yields_dataset_examples() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    assert len(examples) == 3
    assert all(isinstance(e, DatasetExample) for e in examples)


def test_load_proofwriter_first_example_shape() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    first = examples[0]
    assert first.id == "pw_001"
    assert first.premises == ("Bird(robin)",)
    assert first.hypothesis == "Bird(robin)"
    assert first.label == "consistent"
    assert first.source == "proofwriter"


def test_load_proofwriter_decodes_rendered_cnf() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    assert isinstance(examples[0].rendered_cnf, CNF)
    cnf = examples[0].rendered_cnf
    assert cnf is not None
    assert len(cnf.clauses) == 1
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_load_proofwriter_skips_malformed_records() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    assert len(examples) == 3  # the 4th is malformed and should be skipped
    assert last_skip_count() == 1


def test_load_proofwriter_max_examples_truncates() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER, max_examples=2))
    assert len(examples) == 2


def test_load_proofwriter_inconsistent_label_decoded() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    inconsistent = [e for e in examples if e.label == "inconsistent"]
    assert len(inconsistent) == 1
    assert inconsistent[0].id == "pw_002"


def test_load_ruletaker_yields_examples() -> None:
    examples = list(load_ruletaker(path=RULETAKER))
    assert len(examples) == 2
    assert examples[0].source == "ruletaker"


def test_load_qverify_mini_yields_50_examples() -> None:
    from qverify.eval.datasets import load_qverify_mini

    examples = list(load_qverify_mini())
    assert len(examples) == 50
    n_consistent = sum(1 for e in examples if e.label == "consistent")
    n_inconsistent = sum(1 for e in examples if e.label == "inconsistent")
    assert n_consistent == 25
    assert n_inconsistent == 25
    assert all(e.rendered_cnf is not None for e in examples)


def test_load_folio_raises_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="CC BY-SA"):
        list(load_folio())


def test_loader_missing_file_raises_filenotfound(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist.json"
    with pytest.raises(FileNotFoundError, match="No cached fixture"):
        list(load_proofwriter(path=missing))


def test_loader_non_list_root_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps({"not": "a list"}), encoding="utf-8")
    with pytest.raises(ValueError, match="expected a list"):
        list(load_proofwriter(path=bad))


def test_loader_invalid_label_record_is_skipped(tmp_path: Path) -> None:
    bad = tmp_path / "with_bad_label.json"
    bad.write_text(
        json.dumps(
            [
                {
                    "id": "ok",
                    "premises": ["P(a)"],
                    "hypothesis": "P(a)",
                    "label": "consistent",
                },
                {
                    "id": "bad_label",
                    "premises": ["P(a)"],
                    "hypothesis": "P(a)",
                    "label": "unknown",  # not in {consistent, inconsistent}
                },
            ]
        ),
        encoding="utf-8",
    )
    examples = list(load_proofwriter(path=bad))
    assert len(examples) == 1
    assert examples[0].id == "ok"
    assert last_skip_count() == 1


def test_loader_rendered_cnf_optional(tmp_path: Path) -> None:
    no_cnf = tmp_path / "no_cnf.json"
    no_cnf.write_text(
        json.dumps(
            [
                {
                    "id": "x",
                    "premises": ["A"],
                    "hypothesis": "A",
                    "label": "consistent",
                }
            ]
        ),
        encoding="utf-8",
    )
    examples = list(load_proofwriter(path=no_cnf))
    assert examples[0].rendered_cnf is None


def test_loader_rendered_cnf_supports_negated_literals() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    inconsistent = next(e for e in examples if e.id == "pw_002")
    cnf = inconsistent.rendered_cnf
    assert cnf is not None
    negated_lits = [lit for cl in cnf.clauses for lit in cl.literals if lit.negated]
    assert len(negated_lits) == 1
    assert negated_lits[0].predicate == "Bird"
