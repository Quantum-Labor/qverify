"""Unit tests for Universe and finite-domain grounding."""

from __future__ import annotations

import subprocess
import sys

import pytest
from pydantic import ValidationError

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier import verify
from qverify.verifier.grounding import (
    GroundingError,
    Universe,
    _collect_free_variables,
    _substitute_clause,
    ground_cnf,
)


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


# ---------------------------------------------------------------------------
# Universe validation
# ---------------------------------------------------------------------------


def test_universe_empty_constants_allowed() -> None:
    u = Universe(constants=())
    assert u.constants == ()


def test_universe_single_constant_allowed() -> None:
    u = Universe(constants=("Tom",))
    assert u.constants == ("Tom",)


def test_universe_sorts_constants_lexicographically() -> None:
    u = Universe(constants=("zebra", "alice", "bob_cat"))
    assert u.constants == ("alice", "bob_cat", "zebra")


def test_universe_rejects_duplicate_constants() -> None:
    with pytest.raises(ValidationError, match="duplicate"):
        Universe(constants=("alice", "alice"))


def test_universe_rejects_constant_matching_variable_pattern() -> None:
    with pytest.raises(ValidationError, match="free-variable pattern"):
        Universe(constants=("x",))


def test_universe_rejects_short_lowercase_skolem_like_name() -> None:
    with pytest.raises(ValidationError, match="free-variable pattern"):
        Universe(constants=("sk1",))


def test_universe_rejects_empty_string_constant() -> None:
    with pytest.raises(ValidationError):
        Universe(constants=("",))


def test_universe_rejects_non_alphanumeric_constant() -> None:
    with pytest.raises(ValidationError, match="alphanumeric"):
        Universe(constants=("foo bar",))


def test_universe_accepts_uppercase_short_name() -> None:
    # `Tom` is 3 chars but starts uppercase, so it is NOT a free variable.
    u = Universe(constants=("Tom",))
    assert u.constants == ("Tom",)


def test_universe_accepts_long_lowercase_name() -> None:
    u = Universe(constants=("alice", "tweety", "penguin"))
    assert "alice" in u.constants


def test_universe_is_frozen() -> None:
    u = Universe(constants=("tom_cat",))
    with pytest.raises(ValidationError):
        u.constants = ("other",)  # type: ignore[misc]


# ---------------------------------------------------------------------------
# _collect_free_variables
# ---------------------------------------------------------------------------


def test_collect_free_variables_empty_cnf() -> None:
    assert _collect_free_variables(CNF(clauses=())) == frozenset()


def test_collect_free_variables_pure_propositional() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q", neg=True)))
    assert _collect_free_variables(cnf) == frozenset()


def test_collect_free_variables_one_variable() -> None:
    cnf = _cnf(_clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")))
    assert _collect_free_variables(cnf) == frozenset({"x"})


def test_collect_free_variables_shared_across_clauses() -> None:
    cnf = _cnf(
        _clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")),
        _clause(_atom("Animal", "x")),
    )
    assert _collect_free_variables(cnf) == frozenset({"x"})


def test_collect_free_variables_two_distinct() -> None:
    cnf = _cnf(_clause(_atom("Loves", "x", "y", neg=True)))
    assert _collect_free_variables(cnf) == frozenset({"x", "y"})


def test_collect_free_variables_in_negated_literal_detected() -> None:
    cnf = _cnf(_clause(_atom("Bird", "z", neg=True)))
    assert _collect_free_variables(cnf) == frozenset({"z"})


def test_collect_free_variables_ignores_long_constants() -> None:
    cnf = _cnf(_clause(_atom("Bird", "penguin"), _atom("Cat", "tweety")))
    assert _collect_free_variables(cnf) == frozenset()


# ---------------------------------------------------------------------------
# _substitute_clause
# ---------------------------------------------------------------------------


def test_substitute_clause_no_substitution_unchanged() -> None:
    clause = _clause(_atom("P"), _atom("Q"))
    out = _substitute_clause(clause, {})
    assert out == clause


def test_substitute_clause_renames_single_variable() -> None:
    clause = _clause(_atom("Cat", "x", neg=True), _atom("Fur", "x"))
    out = _substitute_clause(clause, {"x": "Tom"})
    assert out.literals[0].args == ("Tom",)
    assert out.literals[1].args == ("Tom",)


def test_substitute_clause_variable_appears_twice_in_one_literal() -> None:
    clause = _clause(_atom("Loves", "x", "x"))
    out = _substitute_clause(clause, {"x": "alice"})
    assert out.literals[0].args == ("alice", "alice")


def test_substitute_clause_unmapped_arg_unchanged() -> None:
    clause = _clause(_atom("Cat", "x", neg=True), _atom("Fur", "tweety"))
    out = _substitute_clause(clause, {"x": "Tom"})
    assert out.literals[0].args == ("Tom",)
    assert out.literals[1].args == ("tweety",)


def test_substitute_clause_preserves_predicate_and_negation() -> None:
    clause = _clause(_atom("Cat", "x", neg=True))
    out = _substitute_clause(clause, {"x": "Tom"})
    assert out.literals[0].predicate == "Cat"
    assert out.literals[0].negated is True


# ---------------------------------------------------------------------------
# ground_cnf
# ---------------------------------------------------------------------------


def test_ground_cnf_pure_propositional_is_identity() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q", neg=True)))
    out = ground_cnf(cnf, Universe(constants=()))
    assert out == cnf


def test_ground_cnf_one_variable_one_constant() -> None:
    # forall x. Cat(x) -> Fur(x); ground over {Tom}.
    cnf = _cnf(_clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")))
    out = ground_cnf(cnf, Universe(constants=("Tom",)))
    assert out == _cnf(_clause(_atom("Cat", "Tom", neg=True), _atom("Fur", "Tom")))


def test_ground_cnf_one_variable_three_constants() -> None:
    cnf = _cnf(_clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")))
    out = ground_cnf(cnf, Universe(constants=("Tom", "Whiskers", "Felix")))
    # 3 constants × 1 clause = 3 grounded clauses (post-sort: Felix, Tom, Whiskers).
    assert len(out.clauses) == 3
    grounded_args = {c.literals[0].args[0] for c in out.clauses}
    assert grounded_args == {"Felix", "Tom", "Whiskers"}


def test_ground_cnf_two_variables_two_constants() -> None:
    # forall x. forall y. Loves(x, y); ground over {a_lice, b_ob}.
    cnf = _cnf(_clause(_atom("Loves", "x", "y")))
    out = ground_cnf(cnf, Universe(constants=("alice", "bobby")))
    # 2 vars × 2 constants = 4 grounded copies of the clause.
    assert len(out.clauses) == 4
    pairs = {c.literals[0].args for c in out.clauses}
    assert pairs == {
        ("alice", "alice"),
        ("alice", "bobby"),
        ("bobby", "alice"),
        ("bobby", "bobby"),
    }


def test_ground_cnf_empty_universe_with_variables_raises() -> None:
    cnf = _cnf(_clause(_atom("Cat", "x")))
    with pytest.raises(GroundingError, match="no constants"):
        ground_cnf(cnf, Universe(constants=()))


def test_ground_cnf_deduplicates_identical_grounded_clauses() -> None:
    # Two distinct variables x, y both grounded over the same singleton
    # universe collapse to identical clauses (Loves(Tom, Tom) appears twice).
    cnf = _cnf(_clause(_atom("Loves", "x", "y")))
    out = ground_cnf(cnf, Universe(constants=("Tom",)))
    assert len(out.clauses) == 1
    assert out.clauses[0].literals[0].args == ("Tom", "Tom")


def test_ground_cnf_mixed_variable_and_constant_in_same_literal() -> None:
    # forall x. Knows(x, alice)
    cnf = _cnf(_clause(_atom("Knows", "x", "alice")))
    out = ground_cnf(cnf, Universe(constants=("bobby", "carol")))
    pairs = {c.literals[0].args for c in out.clauses}
    assert pairs == {("bobby", "alice"), ("carol", "alice")}


def test_ground_cnf_multi_clause_first_order() -> None:
    # forall x. Cat(x) -> Fur(x), AND Cat(Tom)
    cnf = _cnf(
        _clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")),
        _clause(_atom("Cat", "Tom")),
    )
    out = ground_cnf(cnf, Universe(constants=("Tom", "Whiskers")))
    # The universal expands into 2 clauses; the ground Cat(Tom) clause stays as 1.
    # Per-assignment, both clauses are emitted; for x=Tom we get ¬Cat(Tom)∨Fur(Tom)
    # plus Cat(Tom); for x=Whiskers we get ¬Cat(Whiskers)∨Fur(Whiskers) plus
    # Cat(Tom) — the latter dedupes.
    assert len(out.clauses) == 3


def test_ground_cnf_is_deterministic() -> None:
    cnf = _cnf(_clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")))
    universe = Universe(constants=("Felix", "Tom", "Whiskers"))
    out1 = ground_cnf(cnf, universe)
    out2 = ground_cnf(cnf, universe)
    # Bit-exact clause ordering, not just set equality.
    assert out1.clauses == out2.clauses


def test_ground_cnf_returns_same_object_for_propositional_input() -> None:
    cnf = _cnf(_clause(_atom("P")))
    out = ground_cnf(cnf, Universe(constants=()))
    # Identity for the propositional case is documented.
    assert out is cnf


# ---------------------------------------------------------------------------
# End-to-end: hand-grounded syllogism through verify()
# ---------------------------------------------------------------------------


def test_grounded_syllogism_runs_through_verify() -> None:
    """All cats have fur; Tom is a cat; therefore Tom has fur."""
    first_order_cnf = _cnf(
        # forall x. Cat(x) -> Fur(x)
        _clause(_atom("Cat", "x", neg=True), _atom("Fur", "x")),
        # Cat(Tom)
        _clause(_atom("Cat", "Tom")),
        # ¬Fur(Tom)  — the negated conclusion
        _clause(_atom("Fur", "Tom", neg=True)),
    )
    universe = Universe(constants=("Tom", "Whiskers"))
    grounded = ground_cnf(first_order_cnf, universe)

    result = verify(grounded, shots=512, seed=42)
    # Premises ∧ ¬conclusion is unsatisfiable here, so the verifier reports
    # "no contradiction found" — i.e. the conclusion is entailed.
    assert result.contradiction_found is False
    assert result.counter_model is None
    # Grounding produced the expected propositional shape.
    assert grounded.variables == {"Cat", "Fur"}


# ---------------------------------------------------------------------------
# Lazy-load contract — strict subprocess check
# ---------------------------------------------------------------------------


def test_grounding_module_does_not_load_heavy_deps() -> None:
    # Spec-required leakage check: torch / transformers / qiskit must NOT
    # appear in sys.modules after importing qverify.verifier.grounding.
    # PennyLane is loaded transitively by qverify.verifier.__init__ for the
    # default simulator backend, which is acceptable — the controller's
    # outer lazy-load contract (Phase 5) is what protects torch.
    code = (
        "import sys; "
        "import qverify.verifier.grounding; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "assert 'transformers' not in sys.modules, 'transformers leaked'; "
        "assert 'qiskit' not in sys.modules, 'qiskit leaked'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
