"""IBM Quantum runtime client wrapper."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from qverify.utils.logging import get_logger

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService

_log = get_logger("qverify.utils.ibm_client")


@dataclass(frozen=True)
class IBMRunResult:
    """Counts and metadata returned by a single IBM Quantum job."""

    job_id: str
    backend_name: str
    counts: dict[str, int]
    raw_metadata: dict[str, Any] = field(default_factory=dict)


class IBMRuntimeClient:
    """Lazy IBM Quantum runtime client.

    Construction never touches the network. The ``QiskitRuntimeService`` is
    instantiated only when :meth:`get_service`, :meth:`least_busy_heron`, or
    :meth:`run` is called.
    """

    def __init__(
        self,
        token: str,
        instance: str,
        channel: str = "ibm_quantum_platform",
    ) -> None:
        self._token: str = token
        self._instance: str = instance
        self._channel: str = channel
        self._service: QiskitRuntimeService | None = None

    def get_service(self) -> QiskitRuntimeService:
        """Return the cached ``QiskitRuntimeService``, instantiating it on first call."""
        if self._service is None:
            from qiskit_ibm_runtime import QiskitRuntimeService

            self._service = QiskitRuntimeService(
                channel=self._channel,
                token=self._token,
                instance=self._instance,
            )
        return self._service

    def least_busy_heron(self, min_qubits: int = 5) -> str:
        """Return the name of the least-busy operational Heron-class backend.

        Delegates filtering to :meth:`QiskitRuntimeService.least_busy`, which
        accepts the ``min_num_qubits`` / ``simulator`` / ``operational`` /
        ``filters`` kwargs directly. The ``filters`` callable rejects every
        backend whose processor family is not Heron, using
        :func:`_extract_processor_family` to normalize the dict-shaped vs
        string-shaped ``processor_type`` attribute that different
        ``qiskit-ibm-runtime`` versions expose.
        """
        service = self.get_service()
        backend = service.least_busy(
            min_num_qubits=min_qubits,
            simulator=False,
            operational=True,
            filters=lambda b: _extract_processor_family(b) == "heron",
        )
        return str(backend.name)

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        backend_name: str,
        shots: int,
        optimization_level: int = 3,
    ) -> IBMRunResult:
        """Transpile to ``backend_name`` and submit via SamplerV2; block until done.

        Counts in the returned :class:`IBMRunResult` are normalized to
        MSB-by-qubit-index (qubit 0 leftmost) — Qiskit's native ``get_counts``
        is the opposite order, and we reverse here so consumers can hand the
        bitstrings straight to
        :meth:`qverify.verifier.encoding.AtomEncoder.bitstring_to_assignment`.
        """
        from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
        from qiskit_ibm_runtime import SamplerV2

        service = self.get_service()
        backend = service.backend(backend_name)

        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        transpiled = pass_manager.run(circuit)

        sampler = SamplerV2(mode=backend)
        job = sampler.run([transpiled], shots=shots)
        # Capture and log the job_id immediately so the user sees it even when
        # the job sits in queue for a while.
        job_id = str(job.job_id())
        _log.info("IBM job submitted: %s on %s", job_id, backend_name)
        result = job.result()
        pub_result = result[0]

        # Qiskit get_counts() returns bitstrings little-endian by classical-bit
        # index (cr[N-1] leftmost). Reverse so qubit 0 ends up leftmost,
        # matching AtomEncoder.bitstring_to_assignment.
        raw_counts = pub_result.data.meas.get_counts()
        counts: dict[str, int] = {bs[::-1]: int(c) for bs, c in raw_counts.items()}

        raw_metadata: dict[str, Any] = {
            "shots": shots,
            "optimization_level": optimization_level,
            "transpiled_depth": int(transpiled.depth()),
            "transpiled_n_qubits": int(transpiled.num_qubits),
        }
        return IBMRunResult(
            job_id=job_id,
            backend_name=backend_name,
            counts=counts,
            raw_metadata=raw_metadata,
        )


def _extract_processor_family(backend: Any) -> str:
    """Return the processor family string in lowercase, or '' when unknown."""
    try:
        processor_type = backend.processor_type
    except (AttributeError, KeyError):
        return ""
    if isinstance(processor_type, dict):
        return str(processor_type.get("family", "")).lower()
    return str(processor_type).lower()
