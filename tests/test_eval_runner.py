"""Unit tests for the benchmark runner."""

from __future__ import annotations

from pathlib import Path

import pytest

from qverify.eval.datasets import DatasetExample, load_proofwriter
from qverify.eval.metrics import DatasetReport, ExampleResult
from qverify.eval.runner import evaluate
from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.backends import PennyLaneBackend

FIXTURES = Path(__file__).parent / "fixtures" / "eval"
PROOFWRITER = FIXTURES / "proofwriter_tiny.json"


def _atom(name: str, neg: bool = False) -> Literal:
    return Literal(predicate=name, negated=neg)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def test_runner_returns_dataset_report_on_fixtures() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=128)
    assert isinstance(report, DatasetReport)
    assert report.dataset == "proofwriter"
    assert report.n_examples == 3
    assert all(isinstance(r, ExampleResult) for r in report.results)


def test_runner_accuracy_is_one_when_verifier_agrees_with_oracle() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=512)
    # All shipped fixtures are simple enough that the simulator always
    # agrees with PySAT.
    assert report.accuracy == 1.0


def test_runner_predicted_label_matches_dataset_label_for_consistent_examples() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=512)
    by_id = {r.example_id: r for r in report.results}
    assert by_id["pw_001"].predicted == "consistent"
    assert by_id["pw_001"].dataset_label == "consistent"
    assert by_id["pw_002"].predicted == "inconsistent"
    assert by_id["pw_002"].dataset_label == "inconsistent"


def test_runner_skips_examples_without_rendered_cnf_when_no_translator() -> None:
    examples = [
        DatasetExample(
            id="needs_translation",
            premises=("a sentence",),
            hypothesis="another sentence",
            label="consistent",
            source="x",
            rendered_cnf=None,
        )
    ]
    report = evaluate(examples, dataset="x", backend=PennyLaneBackend(), shots=128)
    assert report.n_examples == 0
    assert report.n_skipped == 1


def test_runner_uses_translator_callback_when_provided() -> None:
    fake_cnf = CNF(clauses=(_clause(_atom("Q")),))
    examples = [
        DatasetExample(
            id="needs_translation",
            premises=("a sentence",),
            hypothesis="another sentence",
            label="consistent",
            source="x",
            rendered_cnf=None,
        )
    ]
    report = evaluate(
        examples,
        dataset="x",
        backend=PennyLaneBackend(),
        shots=128,
        translate=lambda _ex: fake_cnf,
    )
    assert report.n_examples == 1
    assert report.n_skipped == 0
    assert report.results[0].predicted == "consistent"


def test_runner_skips_examples_when_translator_raises() -> None:
    examples = [
        DatasetExample(
            id="bad",
            premises=("x",),
            hypothesis="y",
            label="consistent",
            source="x",
            rendered_cnf=None,
        )
    ]

    def boom(_ex: DatasetExample) -> CNF:
        raise RuntimeError("translator unavailable")

    report = evaluate(
        examples,
        dataset="x",
        backend=PennyLaneBackend(),
        shots=128,
        translate=boom,
    )
    assert report.n_examples == 0
    assert report.n_skipped == 1


def test_runner_max_examples_caps_processing() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    report = evaluate(
        examples,
        dataset="proofwriter",
        backend=PennyLaneBackend(),
        shots=128,
        max_examples=2,
    )
    assert report.n_examples == 2


def test_runner_records_qubit_count_and_iterations() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER, max_examples=1))
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=128)
    r = report.results[0]
    assert r.n_qubits >= 1
    assert r.n_iterations >= 0


def test_runner_handles_empty_dataset() -> None:
    report = evaluate([], dataset="empty", backend=PennyLaneBackend(), shots=128)
    assert report.n_examples == 0
    assert report.n_skipped == 0
    assert report.accuracy == 0.0


def test_runner_avg_and_p95_seconds_non_negative() -> None:
    examples = list(load_proofwriter(path=PROOFWRITER))
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=128)
    assert report.avg_seconds >= 0.0
    assert report.p95_seconds >= 0.0
    assert report.p95_seconds >= report.avg_seconds or report.n_examples == 1


def test_runner_skips_examples_whose_cnf_exceeds_max_variables() -> None:
    # CNF with 3 distinct atoms — exceeds max_variables=2 and is skipped.
    big_cnf = CNF(
        clauses=(
            _clause(_atom("A"), _atom("B")),
            _clause(_atom("C")),
        )
    )
    examples = [
        DatasetExample(
            id="too_big",
            premises=("A or B",),
            hypothesis="C",
            label="consistent",
            source="x",
            rendered_cnf=big_cnf,
        )
    ]
    report = evaluate(
        examples,
        dataset="x",
        backend=PennyLaneBackend(),
        shots=64,
        max_variables=2,
    )
    assert report.n_examples == 0
    assert report.n_skipped_too_large == 1
    # Not a failure — just a size skip — so the regular n_skipped stays 0
    assert report.n_skipped == 0


def test_runner_with_max_variables_above_threshold_runs_normally() -> None:
    big_cnf = CNF(
        clauses=(
            _clause(_atom("A")),
            _clause(_atom("B")),
        )
    )
    examples = [
        DatasetExample(
            id="ok",
            premises=("A",),
            hypothesis="B",
            label="consistent",
            source="x",
            rendered_cnf=big_cnf,
        )
    ]
    report = evaluate(
        examples,
        dataset="x",
        backend=PennyLaneBackend(),
        shots=64,
        max_variables=8,
    )
    assert report.n_examples == 1
    assert report.n_skipped_too_large == 0


def test_runner_skips_oracle_failure_gracefully(monkeypatch: pytest.MonkeyPatch) -> None:
    examples = list(load_proofwriter(path=PROOFWRITER, max_examples=1))

    def boom(_cnf: CNF) -> bool:
        raise RuntimeError("oracle down")

    monkeypatch.setattr("qverify.eval.runner.pysat_satisfies", boom)
    report = evaluate(examples, dataset="proofwriter", backend=PennyLaneBackend(), shots=128)
    assert report.n_examples == 0
    assert report.n_skipped == 1
