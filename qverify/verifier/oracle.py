"""Quantum oracle for CNF satisfiability."""

from __future__ import annotations

from collections.abc import Callable

import pennylane as qml


def required_ancillas(n_clauses: int) -> int:
    """Number of ancillas needed: one per clause plus one final flag qubit."""
    return n_clauses + 1


def build_sat_oracle(
    encoded_clauses: tuple[tuple[tuple[int, bool], ...], ...],
    n_qubits: int,
    n_ancillas: int | None = None,
) -> Callable[[], None]:
    """Return a function that applies a phase-flip oracle for the CNF.

    The oracle flips the phase of computational basis states ``|x⟩`` such
    that ``x`` satisfies every clause, leaving all other states unchanged.
    Per-clause ancillas occupy wires ``n_qubits .. n_qubits + n_clauses - 1``
    and the final flag qubit is at ``n_qubits + n_clauses``. All ancillas
    are uncomputed at the end so the oracle is safe to use repeatedly inside
    a Grover iteration.
    """
    n_clauses = len(encoded_clauses)
    expected = required_ancillas(n_clauses)
    if n_ancillas is not None and n_ancillas != expected:
        raise ValueError(
            f"n_ancillas={n_ancillas} does not match required {expected} "
            f"({n_clauses} clauses + 1 flag)"
        )

    flag_wire = n_qubits + n_clauses
    clause_ancilla_wires = tuple(range(n_qubits, n_qubits + n_clauses))

    def oracle() -> None:
        for clause_idx, clause in enumerate(encoded_clauses):
            _apply_clause_or(clause, clause_ancilla_wires[clause_idx])

        if n_clauses == 0:
            # Empty CNF is trivially satisfied — flip phase of every state.
            qml.PauliZ(wires=flag_wire)
        else:
            qml.MultiControlledX(wires=[*clause_ancilla_wires, flag_wire])
            qml.PauliZ(wires=flag_wire)
            qml.MultiControlledX(wires=[*clause_ancilla_wires, flag_wire])

        for clause_idx in range(n_clauses - 1, -1, -1):
            _uncompute_clause_or(
                encoded_clauses[clause_idx],
                clause_ancilla_wires[clause_idx],
            )

    return oracle


def _apply_clause_or(
    clause: tuple[tuple[int, bool], ...],
    ancilla_wire: int,
) -> None:
    """Set ``ancilla_wire = OR of literals`` (assumes ancilla starts at |0⟩).

    For OR via De Morgan: AND of negated literals = NOT(OR). The MCX fires
    when every literal is unsatisfied (i.e. the clause is False), flipping
    ancilla to 1; the trailing X then inverts so ancilla = OR.
    """
    control_wires = [q for (q, _polarity) in clause]
    # MCX should fire when every literal is unsatisfied.
    # Positive literal satisfied iff qubit=1, so unsatisfied iff qubit=0 -> control_value=0.
    # Negated literal satisfied iff qubit=0, so unsatisfied iff qubit=1 -> control_value=1.
    control_values = [0 if pol else 1 for (_q, pol) in clause]
    qml.MultiControlledX(
        wires=[*control_wires, ancilla_wire],
        control_values=control_values,
    )
    qml.PauliX(wires=ancilla_wire)


def _uncompute_clause_or(
    clause: tuple[tuple[int, bool], ...],
    ancilla_wire: int,
) -> None:
    """Inverse of :func:`_apply_clause_or` — restores the ancilla to |0⟩."""
    qml.PauliX(wires=ancilla_wire)
    control_wires = [q for (q, _polarity) in clause]
    control_values = [0 if pol else 1 for (_q, pol) in clause]
    qml.MultiControlledX(
        wires=[*control_wires, ancilla_wire],
        control_values=control_values,
    )
