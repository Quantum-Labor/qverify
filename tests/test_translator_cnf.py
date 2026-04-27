"""Unit tests for the CNF data structures."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from qverify.translator.cnf import CNF, Clause, Literal

# ---------------------------------------------------------------------------
# Literal construction and validation
# ---------------------------------------------------------------------------


def test_literal_atomic_no_args() -> None:
    lit = Literal(predicate="Rain")
    assert lit.predicate == "Rain"
    assert lit.args == ()
    assert lit.negated is False


def test_literal_with_single_arg() -> None:
    lit = Literal(predicate="Bird", args=("penguin",))
    assert lit.args == ("penguin",)


def test_literal_with_multiple_args() -> None:
    lit = Literal(predicate="Loves", args=("alice", "bob"))
    assert lit.args == ("alice", "bob")


def test_literal_negated() -> None:
    lit = Literal(predicate="Flies", args=("tweety",), negated=True)
    assert lit.negated is True


def test_literal_args_default_is_empty_tuple() -> None:
    assert Literal(predicate="P").args == ()


# ---------------------------------------------------------------------------
# Predicate name validation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["Bird", "IsBird", "Warm_Blooded", "P", "X1", "Has_2_Wings"],
)
def test_literal_predicate_valid_names_accepted(name: str) -> None:
    Literal(predicate=name)


@pytest.mark.parametrize(
    "name",
    ["bird", "isBird", "_Bird", "1Bird", "Is bird", "Bird!", ""],
)
def test_literal_predicate_invalid_names_rejected(name: str) -> None:
    with pytest.raises(ValidationError):
        Literal(predicate=name)


# ---------------------------------------------------------------------------
# Literal.negate
# ---------------------------------------------------------------------------


def test_negate_flips_positive_to_negative() -> None:
    lit = Literal(predicate="P")
    flipped = lit.negate()
    assert flipped.negated is True
    assert lit.negated is False


def test_negate_flips_negative_to_positive() -> None:
    lit = Literal(predicate="P", negated=True)
    flipped = lit.negate()
    assert flipped.negated is False


def test_negate_preserves_predicate_and_args() -> None:
    lit = Literal(predicate="Loves", args=("a", "b"))
    flipped = lit.negate()
    assert flipped.predicate == "Loves"
    assert flipped.args == ("a", "b")


def test_double_negation_round_trip() -> None:
    lit = Literal(predicate="P", args=("x",))
    assert lit.negate().negate() == lit


# ---------------------------------------------------------------------------
# Literal.__str__
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "lit, expected",
    [
        (Literal(predicate="Rain"), "Rain"),
        (Literal(predicate="Rain", negated=True), "¬Rain"),
        (Literal(predicate="Bird", args=("penguin",)), "Bird(penguin)"),
        (Literal(predicate="Bird", args=("x",), negated=True), "¬Bird(x)"),
        (Literal(predicate="Loves", args=("a", "b")), "Loves(a, b)"),
    ],
)
def test_literal_str(lit: Literal, expected: str) -> None:
    assert str(lit) == expected


# ---------------------------------------------------------------------------
# Frozen models — mutation must raise
# ---------------------------------------------------------------------------


def test_literal_is_frozen() -> None:
    lit = Literal(predicate="P")
    with pytest.raises(ValidationError):
        lit.predicate = "Q"  # type: ignore[misc]


def test_clause_is_frozen() -> None:
    clause = Clause(literals=(Literal(predicate="P"),))
    with pytest.raises(ValidationError):
        clause.literals = ()  # type: ignore[misc]


def test_cnf_is_frozen() -> None:
    cnf = CNF(clauses=())
    with pytest.raises(ValidationError):
        cnf.clauses = ()  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Clause
# ---------------------------------------------------------------------------


def test_clause_with_single_literal() -> None:
    clause = Clause(literals=(Literal(predicate="P"),))
    assert len(clause.literals) == 1


def test_clause_empty_rejected() -> None:
    with pytest.raises(ValidationError):
        Clause(literals=())


def test_clause_str_single_literal() -> None:
    clause = Clause(literals=(Literal(predicate="P"),))
    assert str(clause) == "(P)"


def test_clause_str_multiple_literals() -> None:
    clause = Clause(
        literals=(
            Literal(predicate="Bird", args=("x",), negated=True),
            Literal(predicate="Flies", args=("x",)),
        )
    )
    assert str(clause) == "(¬Bird(x) ∨ Flies(x))"


# ---------------------------------------------------------------------------
# CNF
# ---------------------------------------------------------------------------


def test_cnf_empty_str_returns_top() -> None:
    assert str(CNF(clauses=())) == "⊤"


def test_cnf_str_single_clause() -> None:
    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="P"),)),))
    assert str(cnf) == "(P)"


def test_cnf_str_joins_with_and() -> None:
    cnf = CNF(
        clauses=(
            Clause(literals=(Literal(predicate="P"),)),
            Clause(literals=(Literal(predicate="Q"),)),
        )
    )
    assert str(cnf) == "(P) ∧ (Q)"


def test_cnf_variables_distinct_predicate_symbols() -> None:
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="Bird", args=("x",), negated=True),
                    Literal(predicate="Flies", args=("x",)),
                )
            ),
            Clause(literals=(Literal(predicate="Bird", args=("penguin",)),)),
        )
    )
    assert cnf.variables == frozenset({"Bird", "Flies"})


def test_cnf_variables_empty_for_empty_formula() -> None:
    assert CNF(clauses=()).variables == frozenset()


# ---------------------------------------------------------------------------
# DIMACS rendering
# ---------------------------------------------------------------------------


def test_to_dimacs_atomic_single_literal() -> None:
    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="Bird", args=("penguin",)),)),))
    expected = "p cnf 1 1\n1 0\n"
    assert cnf.to_dimacs() == expected


def test_to_dimacs_universal_implication() -> None:
    # ¬Bird(x) ∨ Flies(x)
    # Lex order: Bird(x) -> 1, Flies(x) -> 2
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="Bird", args=("x",), negated=True),
                    Literal(predicate="Flies", args=("x",)),
                )
            ),
        )
    )
    expected = "p cnf 2 1\n-1 2 0\n"
    assert cnf.to_dimacs() == expected


def test_to_dimacs_negation_only() -> None:
    cnf = CNF(
        clauses=(Clause(literals=(Literal(predicate="Flies", args=("tweety",), negated=True),)),)
    )
    expected = "p cnf 1 1\n-1 0\n"
    assert cnf.to_dimacs() == expected


def test_to_dimacs_pure_disjunction() -> None:
    # Bird(tweety) ∨ Bat(tweety)
    # Lex: Bat(tweety) -> 1, Bird(tweety) -> 2
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="Bird", args=("tweety",)),
                    Literal(predicate="Bat", args=("tweety",)),
                )
            ),
        )
    )
    expected = "p cnf 2 1\n2 1 0\n"
    assert cnf.to_dimacs() == expected


def test_to_dimacs_two_clause_conjunction() -> None:
    # Mammal(sk1), LaysEggs(sk1)
    # Lex: LaysEggs(sk1) -> 1, Mammal(sk1) -> 2
    cnf = CNF(
        clauses=(
            Clause(literals=(Literal(predicate="Mammal", args=("sk1",)),)),
            Clause(literals=(Literal(predicate="LaysEggs", args=("sk1",)),)),
        )
    )
    expected = "p cnf 2 2\n2 0\n1 0\n"
    assert cnf.to_dimacs() == expected


def test_to_dimacs_distinct_arg_tuples_get_distinct_ids() -> None:
    # Bird(x) and Bird(penguin) are different DIMACS variables
    cnf = CNF(
        clauses=(
            Clause(literals=(Literal(predicate="Bird", args=("x",)),)),
            Clause(literals=(Literal(predicate="Bird", args=("penguin",)),)),
        )
    )
    out = cnf.to_dimacs()
    # Header: 2 vars, 2 clauses
    assert out.startswith("p cnf 2 2\n")
    # Each clause uses its own variable id
    lines = out.strip().splitlines()
    assert lines[1] != lines[2]


def test_to_dimacs_with_provided_var_map() -> None:
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="Bird", args=("x",), negated=True),
                    Literal(predicate="Flies", args=("x",)),
                )
            ),
        )
    )
    var_map = {"Bird(x)": 7, "Flies(x)": 3}
    expected = "p cnf 7 1\n-7 3 0\n"
    assert cnf.to_dimacs(var_map=var_map) == expected


def test_to_dimacs_var_map_missing_atom_raises() -> None:
    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="P"),)),))
    with pytest.raises(ValueError, match="missing"):
        cnf.to_dimacs(var_map={"Q": 1})


def test_to_dimacs_var_map_zero_id_rejected() -> None:
    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="P"),)),))
    with pytest.raises(ValueError, match="positive"):
        cnf.to_dimacs(var_map={"P": 0})
