"""End-to-end tests for the Grover-based verifier on the PennyLane simulator."""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier import (
    MAX_VARIABLES,
    CounterModel,
    VerificationResult,
    VerifierError,
    optimal_iterations,
    verify,
)
from qverify.verifier.classical_check import satisfies

DEFAULT_SHOTS = 512


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


# ---------------------------------------------------------------------------
# Pure helper test
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_qubits,m,expected",
    [(0, 1, 0), (1, 1, 1), (2, 1, 2), (3, 1, 2), (4, 1, 3), (6, 1, 6)],
)
def test_optimal_iterations(n_qubits: int, m: int, expected: int) -> None:
    assert optimal_iterations(n_qubits, m) == expected


def test_optimal_iterations_zero_solution_estimate_returns_zero() -> None:
    assert optimal_iterations(4, 0) == 0


# ---------------------------------------------------------------------------
# Result type contract
# ---------------------------------------------------------------------------


def test_verify_returns_verification_result() -> None:
    cnf = _cnf(_clause(_atom("P")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert isinstance(result, VerificationResult)


def test_counter_model_actually_satisfies_cnf() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert satisfies(cnf, result.counter_model.assignment)


# ---------------------------------------------------------------------------
# SAT cases under entailment mode (counter-model expected)
# ---------------------------------------------------------------------------


def test_simple_sat_p_or_q_with_implication_finds_q_true() -> None:
    # CNF: (P ∨ Q) ∧ (¬P ∨ Q). Both satisfying assignments share Q=True.
    cnf = _cnf(
        _clause(_atom("P"), _atom("Q")),
        _clause(_atom("P", neg=True), _atom("Q")),
    )
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert result.counter_model.assignment["Q"] is True


def test_three_sat_unique_solution() -> None:
    # CNF: (P) ∧ (Q) ∧ (¬R). Unique satisfying assignment: P=T, Q=T, R=F.
    cnf = _cnf(
        _clause(_atom("P")),
        _clause(_atom("Q")),
        _clause(_atom("R", neg=True)),
    )
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert result.counter_model.assignment == {"P": True, "Q": True, "R": False}


def test_four_var_multiple_solutions_finds_a_satisfying_one() -> None:
    # CNF: (P ∨ Q) ∧ (R ∨ S). 9 of 16 assignments satisfy.
    cnf = _cnf(
        _clause(_atom("P"), _atom("Q")),
        _clause(_atom("R"), _atom("S")),
    )
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert satisfies(cnf, result.counter_model.assignment)


def test_unjustified_step_finds_counter_model() -> None:
    # Premise: P. Step (negated): Q. Premises ∧ ¬step = {P, ¬Q}. SAT with P=T, Q=F.
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q", neg=True)))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert result.counter_model.assignment == {"P": True, "Q": False}


# ---------------------------------------------------------------------------
# UNSAT cases under entailment mode (no counter-model — step is entailed)
# ---------------------------------------------------------------------------


def test_simple_unsat_p_and_not_p() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("P", neg=True)))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert not result.contradiction_found
    assert result.counter_model is None


def test_penguin_paradox_three_clause_is_unsat() -> None:
    # The spec's "penguin paradox" CNF: Bird, ¬Flies, ¬Bird ∨ Flies.
    # No assignment satisfies all three — the verifier returns UNSAT
    # (no counter-model to the implied step). This is the correct outcome.
    cnf = _cnf(
        _clause(_atom("Bird", "penguin")),
        _clause(_atom("Flies", "penguin", neg=True)),
        _clause(
            _atom("Bird", "penguin", neg=True),
            _atom("Flies", "penguin"),
        ),
    )
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert not result.contradiction_found
    assert result.counter_model is None
    assert result.n_variables == 2
    assert result.n_clauses == 3


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_cnf_is_consistent() -> None:
    # Empty CNF is trivially satisfied. Consistency-mode default reads
    # this as "the step is consistent with the premises", so
    # contradiction_found is False and counter_model is None.
    result = verify(CNF(clauses=()), shots=DEFAULT_SHOTS, seed=42)
    assert not result.contradiction_found
    assert result.counter_model is None
    assert result.n_variables == 0
    assert result.n_clauses == 0
    assert result.n_grover_iterations == 0


def test_empty_cnf_under_entailment_mode_is_also_consistent() -> None:
    # Both modes agree on the empty CNF: trivially SAT means
    # consistent / no contradiction. counter_model is left None to keep
    # the result faithful to "no satisfying assignment was searched for".
    result = verify(CNF(clauses=()), shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert not result.contradiction_found
    assert result.counter_model is None


def test_qubit_cap_above_max_raises() -> None:
    clauses = tuple(_clause(_atom(f"P{i}")) for i in range(MAX_VARIABLES + 1))
    cnf = CNF(clauses=clauses)
    with pytest.raises(VerifierError, match="at most"):
        verify(cnf)


def test_free_variable_in_input_raises_via_encoder() -> None:
    cnf = _cnf(
        _clause(_atom("Bird", "x", neg=True), _atom("Flies", "x")),
    )
    with pytest.raises(VerifierError, match="free first-order"):
        verify(cnf)


# ---------------------------------------------------------------------------
# Result-field plumbing
# ---------------------------------------------------------------------------


def test_top_measurements_truncated_to_at_most_5() -> None:
    cnf = _cnf(
        _clause(_atom("P"), _atom("Q"), _atom("R")),  # SAT for 7 of 8
    )
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert len(result.top_measurements) <= 5


def test_top_measurements_descending_count() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    counts = [c for _bs, c in result.top_measurements]
    assert counts == sorted(counts, reverse=True)


def test_n_grover_iterations_matches_optimal() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")), _clause(_atom("R")))
    # 3 atoms => optimal_iterations(3, 1) == 2
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert result.n_grover_iterations == optimal_iterations(3, 1)


def test_backend_name_recorded() -> None:
    cnf = _cnf(_clause(_atom("P")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert result.backend_name == "default.qubit"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_top_measurements() -> None:
    cnf = _cnf(
        _clause(_atom("P"), _atom("Q")),
        _clause(_atom("P", neg=True), _atom("Q")),
    )
    r1 = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    r2 = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert r1.top_measurements == r2.top_measurements


def test_different_seeds_produce_different_top_measurements() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    r1 = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    r2 = verify(cnf, shots=DEFAULT_SHOTS, seed=99)
    # The two measurement histograms should differ on at least one bitstring count.
    assert dict(r1.top_measurements) != dict(r2.top_measurements)


# ---------------------------------------------------------------------------
# Pydantic model invariant
# ---------------------------------------------------------------------------


def test_counter_model_str_renders_truth_values() -> None:
    cm = CounterModel(assignment={"P": True, "Q": False})
    assert str(cm) == "{P=T, Q=F}"


def test_verification_result_consistency_rejection_allows_no_counter_model() -> None:
    # Phase 6.8: consistency-mode rejections (UNSAT) legitimately have
    # contradiction_found=True with counter_model=None. No exception.
    result = VerificationResult(
        contradiction_found=True,
        counter_model=None,
        n_variables=1,
        n_clauses=1,
        n_grover_iterations=1,
        backend_name="x",
        shots=1,
    )
    assert result.contradiction_found is True
    assert result.counter_model is None


def test_verification_result_invariant_no_contradiction_forbids_counter_model() -> None:
    with pytest.raises(ValueError, match="forbids"):
        VerificationResult(
            contradiction_found=False,
            counter_model=CounterModel(assignment={"P": True}),
            n_variables=1,
            n_clauses=1,
            n_grover_iterations=1,
            backend_name="x",
            shots=1,
        )


# ---------------------------------------------------------------------------
# Phase 6.8: mode parameter — entailment vs consistency
# ---------------------------------------------------------------------------


def test_verify_default_mode_is_consistency() -> None:
    # Satisfiable CNF: under default (consistency) mode, SAT means
    # consistent → contradiction_found=False and no counter_model.
    cnf = _cnf(_clause(_atom("P")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42)
    assert result.contradiction_found is False
    assert result.counter_model is None


def test_verify_consistency_mode_satisfiable_cnf() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="consistency")
    assert result.contradiction_found is False
    assert result.counter_model is None


def test_verify_consistency_mode_unsatisfiable_cnf() -> None:
    # {P, ¬P} is UNSAT — under consistency mode, that means the step
    # contradicts the premises, so contradiction_found=True. Counter-model
    # is None because no satisfying assignment exists.
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("P", neg=True)))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="consistency")
    assert result.contradiction_found is True
    assert result.counter_model is None


def test_verify_entailment_mode_preserves_old_behaviour() -> None:
    # Under entailment mode, the SAT case yields a counter-model.
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q", neg=True)))
    result = verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="entailment")
    assert result.contradiction_found is True
    assert result.counter_model is not None
    assert result.counter_model.assignment == {"P": True, "Q": False}


def test_verify_mode_invalid_raises() -> None:
    cnf = _cnf(_clause(_atom("P")))
    with pytest.raises(ValueError, match="mode must be"):
        verify(cnf, shots=DEFAULT_SHOTS, seed=42, mode="bogus")  # type: ignore[arg-type]
