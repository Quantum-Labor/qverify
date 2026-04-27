"""Unit tests for the classical CNF satisfaction checker."""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.classical_check import satisfies


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def test_atomic_positive_satisfied() -> None:
    cnf = _cnf(_clause(_atom("P")))
    assert satisfies(cnf, {"P": True})


def test_atomic_positive_unsatisfied() -> None:
    cnf = _cnf(_clause(_atom("P")))
    assert not satisfies(cnf, {"P": False})


def test_atomic_negation_satisfied_when_atom_false() -> None:
    cnf = _cnf(_clause(_atom("P", neg=True)))
    assert satisfies(cnf, {"P": False})


def test_atomic_negation_unsatisfied_when_atom_true() -> None:
    cnf = _cnf(_clause(_atom("P", neg=True)))
    assert not satisfies(cnf, {"P": True})


def test_two_clause_conjunction_both_required() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    assert satisfies(cnf, {"P": True, "Q": True})
    assert not satisfies(cnf, {"P": True, "Q": False})
    assert not satisfies(cnf, {"P": False, "Q": True})


def test_p_and_not_p_unsat_for_every_assignment() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("P", neg=True)))
    assert not satisfies(cnf, {"P": True})
    assert not satisfies(cnf, {"P": False})


def test_disjunction_at_least_one_literal_must_be_true() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q")))
    assert satisfies(cnf, {"P": True, "Q": False})
    assert satisfies(cnf, {"P": False, "Q": True})
    assert satisfies(cnf, {"P": True, "Q": True})
    assert not satisfies(cnf, {"P": False, "Q": False})


def test_empty_cnf_trivially_satisfied() -> None:
    assert satisfies(CNF(clauses=()), {})
    assert satisfies(CNF(clauses=()), {"P": True})  # ignores extra atoms


def test_assignment_missing_atom_raises_key_error() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q")))
    with pytest.raises(KeyError, match="missing atom 'Q'"):
        satisfies(cnf, {"P": False})


def test_extra_atoms_in_assignment_are_ignored() -> None:
    cnf = _cnf(_clause(_atom("P")))
    assert satisfies(cnf, {"P": True, "Q": False, "R": True})


def test_canonical_atom_name_with_args_required() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin")))
    assert satisfies(cnf, {"Bird(penguin)": True})
    with pytest.raises(KeyError):
        satisfies(cnf, {"Bird": True})  # wrong key form


def test_universal_implication_clause() -> None:
    # CNF: (¬Bird(x) ∨ Flies(x)) — but use ground constants
    cnf = _cnf(
        _clause(
            _atom("Bird", "tweety", neg=True),
            _atom("Flies", "tweety"),
        )
    )
    assert satisfies(cnf, {"Bird(tweety)": False, "Flies(tweety)": False})
    assert satisfies(cnf, {"Bird(tweety)": True, "Flies(tweety)": True})
    assert not satisfies(cnf, {"Bird(tweety)": True, "Flies(tweety)": False})


def test_three_clause_unsat_triplet() -> None:
    # Bird, ¬Flies, ¬Bird ∨ Flies — the spec's "penguin paradox" CNF
    cnf = _cnf(
        _clause(_atom("Bird", "penguin")),
        _clause(_atom("Flies", "penguin", neg=True)),
        _clause(
            _atom("Bird", "penguin", neg=True),
            _atom("Flies", "penguin"),
        ),
    )
    for b in (True, False):
        for f in (True, False):
            assert not satisfies(cnf, {"Bird(penguin)": b, "Flies(penguin)": f}), (
                f"unexpected SAT for Bird={b}, Flies={f}"
            )


def test_clause_satisfied_by_any_one_literal() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q"), _atom("R")))
    assert satisfies(cnf, {"P": False, "Q": False, "R": True})
    assert satisfies(cnf, {"P": False, "Q": True, "R": False})
    assert satisfies(cnf, {"P": True, "Q": False, "R": False})
    assert not satisfies(cnf, {"P": False, "Q": False, "R": False})
