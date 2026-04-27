"""Qiskit Grover circuit builder."""

from __future__ import annotations

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from qverify.verifier.oracle import required_ancillas


def build_grover_qiskit_circuit(
    encoded_clauses: tuple[tuple[tuple[int, bool], ...], ...],
    n_qubits: int,
    n_iterations: int,
) -> QuantumCircuit:
    """Construct the Grover circuit as a Qiskit ``QuantumCircuit``.

    Layout: qubits ``0..n_qubits-1`` are the assignment register; the next
    ``n_clauses`` qubits are clause-OR ancillas; the final qubit is the
    flag, initialized in ``|−⟩`` for phase kickback. The classical register
    is named ``meas`` and bit ``i`` records the measurement of qubit ``i``.

    Bit-order note: Qiskit's ``Result.get_counts()`` returns bitstrings
    little-endian by classical-bit index, i.e. ``cr[N-1]`` is leftmost.
    Callers (notably :class:`qverify.utils.ibm_client.IBMRuntimeClient`)
    must reverse the strings before handing them to
    :meth:`qverify.verifier.encoding.AtomEncoder.bitstring_to_assignment`,
    which expects MSB-by-qubit-index (qubit 0 = leftmost).
    """
    if n_qubits <= 0:
        raise ValueError("n_qubits must be >= 1")

    n_clauses = len(encoded_clauses)
    n_anc = required_ancillas(n_clauses)
    total = n_qubits + n_anc

    qr = QuantumRegister(total, "q")
    cr = ClassicalRegister(n_qubits, "meas")
    qc = QuantumCircuit(qr, cr)

    flag_wire = n_qubits + n_clauses
    clause_ancilla_wires = list(range(n_qubits, n_qubits + n_clauses))

    for w in range(n_qubits):
        qc.h(w)

    if n_clauses > 0:
        # Flag in |−⟩ for phase kickback: X then H.
        qc.x(flag_wire)
        qc.h(flag_wire)

    for _ in range(n_iterations):
        for clause_idx, clause in enumerate(encoded_clauses):
            _apply_clause_or(qc, clause, clause_ancilla_wires[clause_idx])

        if n_clauses == 1:
            qc.cx(clause_ancilla_wires[0], flag_wire)
        elif n_clauses > 1:
            qc.mcx(clause_ancilla_wires, flag_wire)
        # n_clauses == 0: empty CNF — caller short-circuits before reaching here.

        for clause_idx in range(n_clauses - 1, -1, -1):
            _uncompute_clause_or(qc, encoded_clauses[clause_idx], clause_ancilla_wires[clause_idx])

        _apply_diffusion(qc, list(range(n_qubits)))

    qc.measure(list(range(n_qubits)), list(range(n_qubits)))
    return qc


def _apply_clause_or(
    qc: QuantumCircuit,
    clause: tuple[tuple[int, bool], ...],
    ancilla_wire: int,
) -> None:
    """Set ``ancilla_wire = OR of literals`` (assumes ancilla starts at |0⟩)."""
    control_wires = [q for (q, _pol) in clause]
    bit_values = [0 if pol else 1 for (_q, pol) in clause]

    # X-sandwich any control that should fire on 0 (positive literal,
    # since OR=0 requires the literal to be unsatisfied).
    for q, bv in zip(control_wires, bit_values, strict=True):
        if bv == 0:
            qc.x(q)

    if len(control_wires) == 1:
        qc.cx(control_wires[0], ancilla_wire)
    else:
        qc.mcx(control_wires, ancilla_wire)

    for q, bv in zip(control_wires, bit_values, strict=True):
        if bv == 0:
            qc.x(q)

    qc.x(ancilla_wire)


def _uncompute_clause_or(
    qc: QuantumCircuit,
    clause: tuple[tuple[int, bool], ...],
    ancilla_wire: int,
) -> None:
    """Inverse of :func:`_apply_clause_or` — restores the ancilla to |0⟩."""
    qc.x(ancilla_wire)

    control_wires = [q for (q, _pol) in clause]
    bit_values = [0 if pol else 1 for (_q, pol) in clause]

    for q, bv in zip(control_wires, bit_values, strict=True):
        if bv == 0:
            qc.x(q)

    if len(control_wires) == 1:
        qc.cx(control_wires[0], ancilla_wire)
    else:
        qc.mcx(control_wires, ancilla_wire)

    for q, bv in zip(control_wires, bit_values, strict=True):
        if bv == 0:
            qc.x(q)


def _apply_diffusion(qc: QuantumCircuit, wires: list[int]) -> None:
    """Grover diffusion ``H X MCZ X H`` over ``wires``."""
    for w in wires:
        qc.h(w)
    for w in wires:
        qc.x(w)

    if len(wires) == 1:
        qc.z(wires[0])
    else:
        qc.h(wires[-1])
        qc.mcx(wires[:-1], wires[-1])
        qc.h(wires[-1])

    for w in wires:
        qc.x(w)
    for w in wires:
        qc.h(w)
