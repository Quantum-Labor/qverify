"""Regression lock for the qverify-mini-50 benchmark (audit pass).

The README pins verifier accuracy at 100% (50/50) on the hand-crafted,
PySAT-validated qverify-mini suite. This module locks that in so a future
change that silently breaks a verdict fails CI.

Two layers:

* ``test_mini_fast_subset_stays_100pct`` (default suite) — every example whose
  total simulated register (atoms + clauses + 1 flag) is <= 16 wires, which is
  all 50 except ``e04`` (22 wires, ~150s). Fast guard for every push.
* ``test_mini_full_suite_stays_100pct`` (``slow``) — the complete 50-example
  ``evaluate()`` run including ``e04``; asserts accuracy == 1.0 and 0 skipped.

Only imports from ``qverify``; no source is modified.
"""

from __future__ import annotations

import pytest

from qverify.eval import evaluate, load_qverify_mini
from qverify.verifier import verify
from qverify.verifier.backends import PennyLaneBackend
from qverify.verifier.encoding import AtomEncoder

_MAX_WIRES_FAST = 16  # atoms + clauses + 1 flag; excludes e04 (22) only


def _total_wires(cnf) -> int:
    return AtomEncoder(cnf).n_qubits + len(cnf.clauses) + 1


def _predicted_label(contradiction_found: bool) -> str:
    return "inconsistent" if contradiction_found else "consistent"


def test_mini_fast_subset_stays_100pct() -> None:
    backend = PennyLaneBackend()
    checked = 0
    for ex in load_qverify_mini():
        cnf = ex.rendered_cnf
        assert cnf is not None, f"{ex.id} has no rendered_cnf"
        if _total_wires(cnf) > _MAX_WIRES_FAST:
            continue  # the single 22-wire monster (e04) lives in the slow lane
        result = verify(cnf, backend=backend, mode="consistency")
        assert _predicted_label(result.contradiction_found) == ex.label, (
            f"{ex.id}: verifier said {_predicted_label(result.contradiction_found)!r}, "
            f"gold label is {ex.label!r}"
        )
        checked += 1
    # All 50 except e04 must be in the fast subset.
    assert checked == 49, f"expected 49 fast examples, checked {checked}"


@pytest.mark.slow
def test_mini_full_suite_stays_100pct() -> None:
    report = evaluate(load_qverify_mini(), dataset="qverify-mini", shots=1024)
    assert report.n_examples == 50
    assert report.n_skipped == 0
    assert report.accuracy == 1.0, f"accuracy regressed to {report.accuracy}"
