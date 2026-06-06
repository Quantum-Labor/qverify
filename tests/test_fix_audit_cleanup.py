"""Regression tests for the fix/audit-cleanup pass.

Each test locks one fix documented in audit/REPORT.md "Bugs found". Tests are
added alongside the commit that introduces the corresponding fix.
"""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier import verify
from qverify.verifier.backends import MAX_SIMULATOR_QUBITS, PennyLaneBackend
from qverify.verifier.encoding import VerifierError


def _atom(name: str) -> Literal:
    return Literal(predicate=name)


# --- Commit 1: simulator total-wire guard -----------------------------------


def test_low_atom_high_clause_cnf_is_rejected_by_wire_budget() -> None:
    # 12 atoms (<= MAX_VARIABLES=16, so the atom-count guard passes) but
    # 12 clauses -> 12 + 12 + 1 = 25 wires, over the simulator budget of 24.
    clauses = tuple(Clause(literals=(_atom(f"P{i}"),)) for i in range(12))
    cnf = CNF(clauses=clauses)
    with pytest.raises(VerifierError, match="budget"):
        verify(cnf, backend=PennyLaneBackend(), mode="consistency")


def test_within_budget_multi_clause_cnf_still_runs() -> None:
    # 6 atoms + 7 clauses = 14 wires (<= 24): a normal multi-clause case must
    # run unaffected by the guard. (Kept small so the fast lane stays fast; the
    # 22-wire e04 boundary is asserted via the constant below and exercised in
    # full by the qverify-mini regression suite.)
    atoms = [f"P{i}" for i in range(6)]
    clauses = tuple(Clause(literals=(_atom(a),)) for a in atoms)  # 6 atoms, 6 clauses
    extra = (Clause(literals=(Literal(predicate=atoms[0], negated=True),)),)
    cnf = CNF(clauses=clauses + extra)  # 6 atoms, 7 clauses -> 14 wires
    result = verify(cnf, backend=PennyLaneBackend(), mode="consistency")
    assert result.n_variables == 6


def test_budget_constant_permits_the_mini_benchmark() -> None:
    # Guards the chosen budget value: e04's 22 wires must stay <= the budget.
    assert MAX_SIMULATOR_QUBITS >= 22


# --- Commit 2: orphaned IBM dead code removed + docstring fixed --------------


def _load_space_app():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "space" / "app.py"
    spec = importlib.util.spec_from_file_location("_qv_app_fixtest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_app_fixtest"] = module
    spec.loader.exec_module(module)
    return module


def test_orphaned_ibm_helpers_are_gone() -> None:
    app = _load_space_app()
    for name in (
        "check_job_status",
        "_build_verification_result",
        "_final_payload",
        "_lookup_job",
        "_decode_counts",
        "_PreparedJob",
        "POLL_INTERVAL_SECONDS",
        "LIVE_POLL_TIMEOUT_SECONDS",
        "IBM_TERMINAL_STATUSES",
    ):
        assert not hasattr(app, name), f"dead symbol {name!r} still present"


def test_docstring_no_longer_claims_polling_or_recovery() -> None:
    app = _load_space_app()
    doc = app.__doc__ or ""
    assert "polls the job" not in doc
    assert "recover by job ID" not in doc


def test_simulator_handler_still_works_after_cleanup() -> None:
    app = _load_space_app()
    res = app.verify_on_simulator(app.DEFAULT_LABEL, "consistency")
    assert res["status"] == "completed"
    assert res["backend"] == "default.qubit"


# --- Commit 3: per-IP table is bounded by eviction --------------------------


def _load_safety():
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parent.parent / "space" / "safety.py"
    spec = importlib.util.spec_from_file_location("_qv_safety_fixtest", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_safety_fixtest"] = module
    spec.loader.exec_module(module)
    return module


def test_per_ip_table_is_bounded_by_eviction() -> None:
    from datetime import UTC, datetime, timedelta

    rate_limiter_cls = _load_safety().RateLimiter
    rl = rate_limiter_cls(window_seconds=300, daily_cap=10_000)
    base = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
    # 50 distinct IPs spaced 60s apart span 2940s, far past the 300s window.
    for i in range(50):
        rl.check_and_register(ip=f"10.0.0.{i}", now=base + timedelta(seconds=60 * i))
    # Only IPs seen within the last window survive (5 at 60s spacing + current).
    assert len(rl._last_ip) <= 6


def test_window_zero_keeps_table_empty() -> None:
    from datetime import UTC, datetime

    rate_limiter_cls = _load_safety().RateLimiter
    rl = rate_limiter_cls(window_seconds=0, daily_cap=10_000)
    now = datetime(2026, 6, 6, 12, 0, tzinfo=UTC)
    for i in range(20):
        rl.check_and_register(ip=f"10.0.1.{i}", now=now)
    # Each call clears the table then records only the current IP, so it never
    # accumulates regardless of how many distinct IPs are seen.
    assert len(rl._last_ip) <= 1
