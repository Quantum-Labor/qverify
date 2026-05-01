"""Smoke tests for chart renderers."""

from __future__ import annotations

from pathlib import Path

from qverify.eval.charts import (
    render_accuracy_chart,
    render_latency_chart,
    render_qubit_distribution,
)
from qverify.eval.metrics import DatasetReport, ExampleResult


def _toy_report(dataset: str = "demo", backend: str = "pennylane") -> DatasetReport:
    results = (
        ExampleResult(
            example_id="a",
            dataset_label="consistent",
            expected="consistent",
            predicted="consistent",
            agrees=True,
            verify_seconds=0.012,
            n_qubits=2,
            n_iterations=1,
        ),
        ExampleResult(
            example_id="b",
            dataset_label="inconsistent",
            expected="inconsistent",
            predicted="inconsistent",
            agrees=True,
            verify_seconds=0.045,
            n_qubits=4,
            n_iterations=2,
        ),
    )
    return DatasetReport(
        dataset=dataset,
        backend=backend,
        n_examples=2,
        n_skipped=0,
        accuracy=1.0,
        avg_seconds=0.0285,
        p95_seconds=0.045,
        results=results,
    )


def test_render_accuracy_chart_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "acc.png"
    render_accuracy_chart([_toy_report()], out)
    assert out.exists() and out.stat().st_size > 0


def test_render_latency_chart_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "lat.png"
    render_latency_chart([_toy_report()], out)
    assert out.exists() and out.stat().st_size > 0


def test_render_qubit_distribution_writes_png(tmp_path: Path) -> None:
    out = tmp_path / "qubits.png"
    render_qubit_distribution([_toy_report()], out)
    assert out.exists() and out.stat().st_size > 0
