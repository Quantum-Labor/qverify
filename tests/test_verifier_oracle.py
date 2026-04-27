"""Unit tests for the CNF satisfiability oracle."""

from __future__ import annotations

import math

import numpy as np
import pennylane as qml

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.classical_check import satisfies
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.oracle import build_sat_oracle, required_ancillas


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def _state_after_oracle(cnf: CNF) -> tuple[np.ndarray, AtomEncoder, int, int]:
    enc = AtomEncoder(cnf)
    n_qubits = enc.n_qubits
    n_anc = required_ancillas(len(cnf.clauses))
    total = n_qubits + n_anc
    encoded = enc.encode_clauses()
    oracle = build_sat_oracle(encoded, n_qubits)

    dev = qml.device("default.qubit", wires=total)

    @qml.qnode(dev)
    def circuit() -> np.ndarray:
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        oracle()
        return qml.state()

    return np.asarray(circuit()), enc, n_qubits, n_anc


def _check_oracle_marks_correctly(cnf: CNF) -> None:
    state, enc, n, n_anc = _state_after_oracle(cnf)
    n_states = 2**n
    expected_amp_mag = 1.0 / math.sqrt(n_states)
    for x in range(n_states):
        bitstring = format(x, f"0{n}b") if n > 0 else ""
        asn = enc.bitstring_to_assignment(bitstring)
        sat = satisfies(cnf, asn)
        idx = x * (2**n_anc)  # ancillas all 0 in low bits
        sign = -1 if sat else 1
        assert math.isclose(state[idx].real, sign * expected_amp_mag, abs_tol=1e-9), (
            f"assignment {bitstring} (sat={sat}) wrong amplitude {state[idx]}"
        )
        assert math.isclose(state[idx].imag, 0.0, abs_tol=1e-9)
    # All non-zero-ancilla indices should have ~0 amplitude
    for x in range(n_states):
        for anc in range(1, 2**n_anc):
            idx = x * (2**n_anc) + anc
            assert abs(state[idx]) < 1e-9, f"ancilla leak at idx={idx}: {state[idx]}"


def test_oracle_one_var_atomic_positive() -> None:
    # CNF: (P) — sat iff P=1
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P"))))


def test_oracle_one_var_atomic_negation() -> None:
    # CNF: (¬P) — sat iff P=0
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P", neg=True))))


def test_oracle_two_var_conjunction() -> None:
    # CNF: (P) ∧ (Q) — sat only at P=Q=1
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P")), _clause(_atom("Q"))))


def test_oracle_two_var_disjunction() -> None:
    # CNF: (P ∨ Q) — sat for 3 of 4
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P"), _atom("Q"))))


def test_oracle_two_var_implication() -> None:
    # CNF: (¬P ∨ Q) i.e. P → Q
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P", neg=True), _atom("Q"))))


def test_oracle_three_var_disjunction() -> None:
    # CNF: (P ∨ Q ∨ R) — sat for 7 of 8
    _check_oracle_marks_correctly(_cnf(_clause(_atom("P"), _atom("Q"), _atom("R"))))


def test_oracle_three_var_three_unit_clauses() -> None:
    # CNF: (P) ∧ (Q) ∧ (R) — sat only at 111
    _check_oracle_marks_correctly(
        _cnf(_clause(_atom("P")), _clause(_atom("Q")), _clause(_atom("R")))
    )


def test_oracle_unsat_p_and_not_p() -> None:
    # No assignment satisfies — every state should remain unflipped
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("P", neg=True)))
    state, _enc, n, n_anc = _state_after_oracle(cnf)
    expected = 1.0 / math.sqrt(2**n)
    for x in range(2**n):
        idx = x * (2**n_anc)
        assert math.isclose(state[idx].real, expected, abs_tol=1e-9)


def test_oracle_four_var_two_clauses() -> None:
    # CNF: (P ∨ Q) ∧ (R ∨ S)
    _check_oracle_marks_correctly(
        _cnf(
            _clause(_atom("P"), _atom("Q")),
            _clause(_atom("R"), _atom("S")),
        )
    )


def test_oracle_ancillas_uncomputed_after_call() -> None:
    # After running oracle once, the ancilla wires should be back to |0⟩.
    # We test by checking that the marginal probability on ancilla wires is 0
    # for any non-zero ancilla pattern.
    state, _enc, n, n_anc = _state_after_oracle(
        _cnf(_clause(_atom("P"), _atom("Q")), _clause(_atom("R"), _atom("S")))
    )
    # Sum |amp|^2 for all states with any ancilla bit set
    mass_with_ancilla = 0.0
    for x in range(2**n):
        for anc in range(1, 2**n_anc):
            idx = x * (2**n_anc) + anc
            mass_with_ancilla += float(abs(state[idx]) ** 2)
    assert mass_with_ancilla < 1e-15


def test_oracle_norm_preserved() -> None:
    state, _enc, _n, _n_anc = _state_after_oracle(
        _cnf(_clause(_atom("P"), _atom("Q")), _clause(_atom("R", neg=True)))
    )
    assert math.isclose(float(np.linalg.norm(state)), 1.0, abs_tol=1e-9)


def test_oracle_phase_pattern_complex_three_clause() -> None:
    # Spec's penguin paradox: Bird, ¬Flies, ¬Bird ∨ Flies — UNSAT
    cnf = _cnf(
        _clause(_atom("Bird", "penguin")),
        _clause(_atom("Flies", "penguin", neg=True)),
        _clause(
            _atom("Bird", "penguin", neg=True),
            _atom("Flies", "penguin"),
        ),
    )
    _check_oracle_marks_correctly(cnf)
