"""Quantum backend abstraction."""

from __future__ import annotations

from collections import Counter
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

import numpy as np
import pennylane as qml

from qverify.verifier.diffusion import build_diffusion
from qverify.verifier.encoding import VerifierError
from qverify.verifier.oracle import build_sat_oracle, required_ancillas

if TYPE_CHECKING:
    from qverify.utils.ibm_client import IBMRuntimeClient


# Maximum total wires (atoms + clause ancillas + flag) the statevector
# simulator will allocate. Statevector cost is 2**total_wires, so this caps
# memory and runtime. The atom-only MAX_VARIABLES cap in
# qverify.verifier.grover does not bound the per-clause ancillas, so a
# low-atom / high-clause CNF can still blow past it; this is the backstop.
MAX_SIMULATOR_QUBITS: int = 24


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

        if total_wires > MAX_SIMULATOR_QUBITS:
            raise VerifierError(
                f"Grover circuit needs {total_wires} wires ({n_qubits} atoms + "
                f"{n_clauses} clause ancillas + 1 flag), over the simulator "
                f"budget of {MAX_SIMULATOR_QUBITS} (statevector cost is 2**wires). "
                f"Reduce atoms or clauses, or use a hardware backend."
            )

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


class IBMQuantumBackend:
    """IBM Quantum hardware backend (Heron r2 by default).

    Construction is cheap and never touches the network. The
    :class:`~qverify.utils.ibm_client.IBMRuntimeClient` is built lazily on
    the first call to :meth:`execute_grover` from
    :data:`~qverify.utils.config.Settings.ibm_quantum_token` and
    :data:`~qverify.utils.config.Settings.ibm_quantum_instance` if no
    explicit ``client`` was supplied.
    """

    def __init__(
        self,
        backend_name: str | None = None,
        *,
        client: IBMRuntimeClient | None = None,
        optimization_level: int = 3,
        min_qubits: int = 5,
    ) -> None:
        self._explicit_backend_name: str | None = backend_name
        self._client: IBMRuntimeClient | None = client
        self._optimization_level: int = optimization_level
        self._min_qubits: int = min_qubits
        self._resolved_backend_name: str = ""

    @property
    def name(self) -> str:
        """Resolved backend name (e.g. ``ibm_kingston``); empty until first run."""
        if self._resolved_backend_name:
            return self._resolved_backend_name
        if self._explicit_backend_name:
            return self._explicit_backend_name
        return ""

    def _ensure_client(self) -> IBMRuntimeClient:
        if self._client is not None:
            return self._client
        from qverify.utils.config import load_settings
        from qverify.utils.ibm_client import IBMRuntimeClient

        settings = load_settings()
        token = settings.require("ibm_quantum_token")
        instance = settings.require("ibm_quantum_instance")
        self._client = IBMRuntimeClient(token=token, instance=instance)
        return self._client

    def execute_grover(
        self,
        encoded_clauses: tuple[tuple[tuple[int, bool], ...], ...],
        n_qubits: int,
        n_iterations: int,
        *,
        shots: int,
        seed: int,
    ) -> tuple[dict[str, int], dict[str, object]]:
        del seed  # IBM hardware is intrinsically non-deterministic.
        from qverify.verifier.grover_circuit import build_grover_qiskit_circuit

        circuit = build_grover_qiskit_circuit(encoded_clauses, n_qubits, n_iterations)

        client = self._ensure_client()

        backend_name = self._explicit_backend_name
        if backend_name is None:
            backend_name = client.least_busy_heron(min_qubits=self._min_qubits)

        result = client.run(
            circuit,
            backend_name=backend_name,
            shots=shots,
            optimization_level=self._optimization_level,
        )

        self._resolved_backend_name = result.backend_name

        metadata: dict[str, object] = {
            "backend_name": result.backend_name,
            "job_id": result.job_id,
        }
        metadata.update(result.raw_metadata)
        return result.counts, metadata
