"""Unit tests for the PySAT oracle."""

from __future__ import annotations

from qverify.eval.oracle import pysat_satisfies
from qverify.translator.cnf import CNF, Clause, Literal


def _atom(name: str, neg: bool = False) -> Literal:
    return Literal(predicate=name, negated=neg)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def test_empty_cnf_is_sat() -> None:
    assert pysat_satisfies(CNF(clauses=())) is True


def test_single_positive_literal_is_sat() -> None:
    cnf = CNF(clauses=(_clause(_atom("P")),))
    assert pysat_satisfies(cnf) is True


def test_contradiction_is_unsat() -> None:
    cnf = CNF(clauses=(_clause(_atom("P")), _clause(_atom("P", neg=True))))
    assert pysat_satisfies(cnf) is False


def test_two_clauses_satisfied_by_common_assignment() -> None:
    # (P or Q) and (not P or Q) — Q=true satisfies both
    cnf = CNF(
        clauses=(
            _clause(_atom("P"), _atom("Q")),
            _clause(_atom("P", neg=True), _atom("Q")),
        )
    )
    assert pysat_satisfies(cnf) is True


def test_three_variable_unsat() -> None:
    # (P) and (Q) and (not P or not Q) — UNSAT
    cnf = CNF(
        clauses=(
            _clause(_atom("P")),
            _clause(_atom("Q")),
            _clause(_atom("P", neg=True), _atom("Q", neg=True)),
        )
    )
    assert pysat_satisfies(cnf) is False


def test_predicate_with_args_treated_as_distinct_variable() -> None:
    cat_tom = Literal(predicate="Cat", args=("tom",))
    cat_jerry = Literal(predicate="Cat", args=("jerry",))
    cnf = CNF(clauses=(_clause(cat_tom), _clause(cat_jerry)))
    assert pysat_satisfies(cnf) is True


def test_pigeon_hole_3_in_2_unsat() -> None:
    # 3 pigeons, 2 holes; each pigeon in some hole; no two pigeons share a hole
    p = [[Literal(predicate=f"H{p}{h}") for h in range(2)] for p in range(3)]
    clauses: list[Clause] = []
    for row in p:
        clauses.append(Clause(literals=tuple(row)))
    for h in range(2):
        for i in range(3):
            for j in range(i + 1, 3):
                clauses.append(
                    Clause(
                        literals=(
                            Literal(predicate=f"H{i}{h}", negated=True),
                            Literal(predicate=f"H{j}{h}", negated=True),
                        )
                    )
                )
    assert pysat_satisfies(CNF(clauses=tuple(clauses))) is False


def test_satisfiable_after_removing_one_clause() -> None:
    base = (
        _clause(_atom("A")),
        _clause(_atom("B")),
        _clause(_atom("A", neg=True), _atom("B", neg=True)),
    )
    assert pysat_satisfies(CNF(clauses=base)) is False
    # Drop the third (incompatibility) clause, now SAT
    assert pysat_satisfies(CNF(clauses=base[:2])) is True
