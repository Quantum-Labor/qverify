"""Metric models for benchmark runs."""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from qverify.eval.datasets import DatasetLabel


class ExampleResult(BaseModel):
    """Outcome of running the verifier on a single example."""

    model_config = ConfigDict(frozen=True)

    example_id: str
    dataset_label: DatasetLabel = Field(
        ..., description="Gold label shipped with the dataset; recorded for audit."
    )
    expected: DatasetLabel = Field(
        ..., description="Ground truth per the PySAT oracle; what predicted is judged against."
    )
    predicted: DatasetLabel = Field(
        ..., description="Verifier output mapped to the two-class label."
    )
    agrees: bool = Field(..., description="True iff predicted == expected.")
    verify_seconds: float = Field(..., ge=0.0)
    n_qubits: int = Field(..., ge=0)
    n_iterations: int = Field(..., ge=0)


class DatasetReport(BaseModel):
    """Aggregate metrics over a single dataset run."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    backend: str
    n_examples: int = Field(..., ge=0)
    n_skipped: int = Field(..., ge=0)
    accuracy: float = Field(..., ge=0.0, le=1.0)
    avg_seconds: float = Field(..., ge=0.0)
    p95_seconds: float = Field(..., ge=0.0)
    n_translated: int = Field(default=0, ge=0)
    n_translation_failed: int = Field(default=0, ge=0)
    avg_translation_seconds: float = Field(default=0.0, ge=0.0)
    n_skipped_too_large: int = Field(default=0, ge=0)
    results: tuple[ExampleResult, ...] = Field(default_factory=tuple)


def build_report(
    *,
    dataset: str,
    backend: str,
    results: Iterable[ExampleResult],
    n_skipped: int,
    n_translated: int = 0,
    n_translation_failed: int = 0,
    avg_translation_seconds: float = 0.0,
    n_skipped_too_large: int = 0,
) -> DatasetReport:
    """Aggregate per-example results into a :class:`DatasetReport`."""
    results_tuple = tuple(results)
    n = len(results_tuple)
    if n == 0:
        return DatasetReport(
            dataset=dataset,
            backend=backend,
            n_examples=0,
            n_skipped=n_skipped,
            accuracy=0.0,
            avg_seconds=0.0,
            p95_seconds=0.0,
            n_translated=n_translated,
            n_translation_failed=n_translation_failed,
            avg_translation_seconds=avg_translation_seconds,
            n_skipped_too_large=n_skipped_too_large,
            results=(),
        )
    accuracy = sum(1 for r in results_tuple if r.agrees) / n
    times = sorted(r.verify_seconds for r in results_tuple)
    avg = sum(times) / n
    # Nearest-rank P95: ceil(0.95 * n) - 1, clamped to [0, n-1]
    p95_idx = max(0, min(n - 1, -(-(95 * n) // 100) - 1))
    p95 = times[p95_idx]
    return DatasetReport(
        dataset=dataset,
        backend=backend,
        n_examples=n,
        n_skipped=n_skipped,
        accuracy=accuracy,
        avg_seconds=avg,
        p95_seconds=p95,
        n_translated=n_translated,
        n_translation_failed=n_translation_failed,
        avg_translation_seconds=avg_translation_seconds,
        n_skipped_too_large=n_skipped_too_large,
        results=results_tuple,
    )
