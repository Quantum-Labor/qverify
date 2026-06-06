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
