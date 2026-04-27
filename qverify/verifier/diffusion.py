"""Grover diffusion operator."""

from __future__ import annotations

import math
from collections.abc import Callable

import pennylane as qml


def build_diffusion(n_qubits: int) -> Callable[[], None]:
    """Return a function that applies the Grover diffusion operator.

    Implements ``D = 2|s⟩⟨s| - I`` where ``|s⟩`` is the uniform
    superposition over wires ``0..n_qubits-1``. The construction is
    ``H X MCZ X H`` (which equals ``-D``) followed by a global ``-1``
    phase to match the canonical sign. For ``n_qubits = 1`` the MCZ
    degenerates to a single-qubit Z.
    """
    if n_qubits < 1:
        raise ValueError(f"diffusion requires n_qubits >= 1, got {n_qubits}")

    wires = list(range(n_qubits))
    last_wire = n_qubits - 1

    def diffusion() -> None:
        for w in wires:
            qml.Hadamard(wires=w)
        for w in wires:
            qml.PauliX(wires=w)

        if n_qubits == 1:
            qml.PauliZ(wires=0)
        else:
            qml.Hadamard(wires=last_wire)
            qml.MultiControlledX(wires=wires)
            qml.Hadamard(wires=last_wire)

        for w in wires:
            qml.PauliX(wires=w)
        for w in wires:
            qml.Hadamard(wires=w)

        # The H-X-MCZ-X-H sandwich computes -(2|s⟩⟨s| - I); the global phase
        # of pi restores the canonical sign of the diffusion operator.
        qml.GlobalPhase(math.pi)

    return diffusion
