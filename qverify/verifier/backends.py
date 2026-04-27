"""Quantum backend abstraction."""

from __future__ import annotations

from collections import Counter
from typing import Any, Protocol, runtime_checkable

import numpy as np
import pennylane as qml

from qverify.verifier.diffusion import build_diffusion
from qverify.verifier.oracle import build_sat_oracle, required_ancillas


@runtime_checkable
class Backend(Protocol):
    """Anything that can run a Grover circuit and return measurement counts."""

    name: str

    def execute_grover(
        self,
        encoded_clauses: tuple[tuple[tuple[int, bool], ...], ...],
        n_qubits: int,
        n_iterations: int,
        *,
        shots: int,
        seed: int,
    ) -> tuple[dict[str, int], dict[str, object]]:
        """Run the Grover circuit and return ``(counts, metadata)``.

        ``counts`` keys are bitstrings, MSB-first by qubit index, length
        ``n_qubits`` — the same convention as
        :meth:`qverify.verifier.encoding.AtomEncoder.bitstring_to_assignment`.
        ``metadata`` carries backend-specific information such as
        ``backend_name`` and (for hardware) ``job_id``.
        """
        ...


class PennyLaneBackend:
    """Local statevector simulator (``default.qubit``).

    Constructs a fresh PennyLane device per call so the caller's ``seed``
    fully controls the RNG stream.
    """

    name: str = "default.qubit"

    def execute_grover(
        self,
        encoded_clauses: tuple[tuple[tuple[int, bool], ...], ...],
        n_qubits: int,
        n_iterations: int,
        *,
        shots: int,
        seed: int,
    ) -> tuple[dict[str, int], dict[str, object]]:
        n_clauses = len(encoded_clauses)
        n_anc = required_ancillas(n_clauses)
        total_wires = n_qubits + n_anc

        oracle = build_sat_oracle(encoded_clauses, n_qubits)
        diffusion = build_diffusion(n_qubits)

        device = qml.device("default.qubit", wires=total_wires, seed=seed)

        @qml.set_shots(shots=shots)
        @qml.qnode(device)
        def circuit() -> Any:
            for w in range(n_qubits):
                qml.Hadamard(wires=w)
            for _ in range(n_iterations):
                oracle()
                diffusion()
            return qml.sample(wires=list(range(n_qubits)))

        samples = np.asarray(circuit())
        if samples.ndim == 1:
            samples = samples.reshape(-1, 1)

        counter: Counter[str] = Counter()
        for row in samples:
            bitstring = "".join(str(int(b)) for b in row)
            counter[bitstring] += 1

        metadata: dict[str, object] = {"backend_name": self.name}
        return dict(counter), metadata
