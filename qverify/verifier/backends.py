"""Quantum backend abstraction."""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import pennylane as qml


@runtime_checkable
class Backend(Protocol):
    """Anything that can produce a PennyLane-compatible device."""

    name: str

    def device(self, n_wires: int, *, shots: int, seed: int) -> Any: ...


class PennyLaneBackend:
    """Local simulator (``default.qubit``).

    A fresh :class:`pennylane.Device` is created per call so the supplied
    ``seed`` always controls the entire RNG stream — re-running with the same
    seed and shot count produces identical samples.
    """

    name: str = "default.qubit"

    def device(self, n_wires: int, *, shots: int, seed: int) -> Any:
        del shots  # shots are applied via qml.set_shots on the QNode
        return qml.device("default.qubit", wires=n_wires, seed=seed)
