"""End-to-end benchmark runner.

The runner takes an iterable of :class:`DatasetExample` plus a verifier
backend, runs the verifier on each example's CNF, compares the verifier's
output against the PySAT oracle, and returns a :class:`DatasetReport`.

Translation policy
------------------

If an example carries a ``rendered_cnf``, the runner uses it directly and
skips translation. This is how Phase 6 results are reproduced without a
GPU: the shipped fixtures (and any user-prepared cache) include
pre-encoded CNFs.

If ``rendered_cnf`` is missing, the runner calls a caller-supplied
``translate`` function. Examples whose translation raises are skipped and
counted in ``n_skipped``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterable

from qverify.eval.datasets import DatasetExample, DatasetLabel
from qverify.eval.metrics import DatasetReport, ExampleResult, build_report
from qverify.eval.oracle import pysat_satisfies
from qverify.translator.cnf import CNF
from qverify.utils.logging import get_logger
from qverify.verifier import verify
from qverify.verifier.backends import Backend, PennyLaneBackend

_log = get_logger("qverify.eval")

TranslateFn = Callable[[DatasetExample], CNF]
"""A function that converts a dataset example to a CNF on demand."""


def _label_from_sat(is_sat: bool) -> DatasetLabel:
    """Map a SAT verdict to the two-class consistency label."""
    return "consistent" if is_sat else "inconsistent"


def _verifier_label(contradiction_found: bool) -> DatasetLabel:
    """Map the verifier's contradiction flag to the two-class label.

    In consistency mode (the default since Phase 6.8),
    contradiction_found=False means the formula is consistent.
    """
    return "inconsistent" if contradiction_found else "consistent"


def evaluate(
    examples: Iterable[DatasetExample],
    *,
    dataset: str,
    backend: Backend | None = None,
    shots: int = 1024,
    translate: TranslateFn | None = None,
    max_examples: int | None = None,
) -> DatasetReport:
    """Run the verifier over ``examples`` and return the aggregated report."""
    if backend is None:
        backend = PennyLaneBackend()

    results: list[ExampleResult] = []
    n_skipped = 0
    n_processed = 0

    for example in examples:
        if max_examples is not None and n_processed >= max_examples:
            break

        cnf = example.rendered_cnf
        if cnf is None:
            if translate is None:
                _log.info("Skipping %s: no rendered_cnf and no translate fn provided", example.id)
                n_skipped += 1
                continue
            try:
                cnf = translate(example)
            except Exception as exc:
                _log.info("Skipping %s: translate failed (%s)", example.id, exc)
                n_skipped += 1
                continue

        try:
            oracle_sat = pysat_satisfies(cnf)
        except Exception as exc:
            _log.info("Skipping %s: oracle failed (%s)", example.id, exc)
            n_skipped += 1
            continue

        start = time.monotonic()
        try:
            verification = verify(cnf, backend=backend, shots=shots, mode="consistency")
        except Exception as exc:
            _log.info("Skipping %s: verify failed (%s)", example.id, exc)
            n_skipped += 1
            continue
        elapsed = time.monotonic() - start

        predicted = _verifier_label(verification.contradiction_found)
        expected = _label_from_sat(oracle_sat)
        results.append(
            ExampleResult(
                example_id=example.id,
                dataset_label=example.label,
                expected=expected,
                predicted=predicted,
                agrees=(predicted == expected),
                verify_seconds=elapsed,
                n_qubits=verification.n_variables,
                n_iterations=verification.n_grover_iterations,
            )
        )
        n_processed += 1

    return build_report(
        dataset=dataset,
        backend=backend.name,
        results=results,
        n_skipped=n_skipped,
    )
