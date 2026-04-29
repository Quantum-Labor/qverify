"""Unit tests for the Backend abstraction and concrete backends."""

from __future__ import annotations

from typing import Any

import pytest
from qiskit import QuantumCircuit
from qiskit_aer import AerSimulator

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.utils.ibm_client import IBMRunResult, IBMRuntimeClient
from qverify.verifier.backends import (
    Backend,
    IBMQuantumBackend,
    PennyLaneBackend,
)
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.grover import optimal_iterations, run_grover
from qverify.verifier.grover_circuit import build_grover_qiskit_circuit


def _atom(pred: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=pred, args=args, negated=neg)


def _cnf(*clauses: Clause) -> CNF:
    return CNF(clauses=clauses)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


# ---------------------------------------------------------------------------
# PennyLaneBackend regression
# ---------------------------------------------------------------------------


def test_pennylane_backend_implements_protocol() -> None:
    assert isinstance(PennyLaneBackend(), Backend)


def test_pennylane_backend_returns_counts_and_metadata() -> None:
    backend = PennyLaneBackend()
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    enc = AtomEncoder(cnf)
    counts, metadata = backend.execute_grover(
        enc.encode_clauses(),
        enc.n_qubits,
        optimal_iterations(enc.n_qubits, 1),
        shots=512,
        seed=42,
    )
    assert isinstance(counts, dict)
    assert all(isinstance(k, str) and isinstance(v, int) for k, v in counts.items())
    assert metadata.get("backend_name") == "default.qubit"


def test_pennylane_backend_counts_sum_to_shots() -> None:
    backend = PennyLaneBackend()
    cnf = _cnf(_clause(_atom("P"), _atom("Q")))
    enc = AtomEncoder(cnf)
    counts, _meta = backend.execute_grover(
        enc.encode_clauses(),
        enc.n_qubits,
        optimal_iterations(enc.n_qubits, 1),
        shots=512,
        seed=42,
    )
    assert sum(counts.values()) == 512


def test_pennylane_backend_seed_determinism() -> None:
    backend = PennyLaneBackend()
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    enc = AtomEncoder(cnf)
    counts1, _ = backend.execute_grover(enc.encode_clauses(), enc.n_qubits, 2, shots=512, seed=42)
    counts2, _ = backend.execute_grover(enc.encode_clauses(), enc.n_qubits, 2, shots=512, seed=42)
    assert counts1 == counts2


def test_pennylane_backend_via_run_grover_finds_satisfying_assignment() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = run_grover(cnf, shots=512, seed=42, backend=PennyLaneBackend(), mode="entailment")
    assert result.contradiction_found
    assert result.counter_model is not None
    assert result.counter_model.assignment == {"P": True, "Q": True}


# ---------------------------------------------------------------------------
# Qiskit Grover circuit (simulator-only)
# ---------------------------------------------------------------------------


def test_qiskit_circuit_has_expected_register_layout() -> None:
    cnf = _cnf(_clause(_atom("P"), _atom("Q")), _clause(_atom("R")))
    enc = AtomEncoder(cnf)
    qc = build_grover_qiskit_circuit(enc.encode_clauses(), enc.n_qubits, 1)
    # 3 assignment + 2 clause-OR ancillas + 1 flag = 6 qubits, 3 classical bits.
    assert qc.num_qubits == 6
    assert qc.num_clbits == 3
    assert qc.cregs[0].name == "meas"


def test_qiskit_circuit_zero_qubits_raises() -> None:
    with pytest.raises(ValueError, match="n_qubits"):
        build_grover_qiskit_circuit((), 0, 0)


def test_qiskit_circuit_runs_on_aer_and_finds_satisfier() -> None:
    # (P) ∧ (Q) ∧ (¬R) — unique solution P=Q=1, R=0. n=3, optimal iter=2.
    cnf = _cnf(
        _clause(_atom("P")),
        _clause(_atom("Q")),
        _clause(_atom("R", neg=True)),
    )
    enc = AtomEncoder(cnf)
    qc = build_grover_qiskit_circuit(
        enc.encode_clauses(), enc.n_qubits, optimal_iterations(enc.n_qubits, 1)
    )
    sim = AerSimulator()
    result = sim.run(qc, shots=1024, seed_simulator=42).result()
    raw_counts = result.get_counts()
    counts = {bs[::-1]: c for bs, c in raw_counts.items()}
    # Top measurement should be the unique satisfying assignment.
    top = max(counts.items(), key=lambda kv: kv[1])
    assignment = enc.bitstring_to_assignment(top[0])
    assert assignment == {"P": True, "Q": True, "R": False}


def test_qiskit_circuit_unsat_returns_no_satisfier_in_top_measurements() -> None:
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("P", neg=True)))
    enc = AtomEncoder(cnf)
    qc = build_grover_qiskit_circuit(
        enc.encode_clauses(), enc.n_qubits, optimal_iterations(enc.n_qubits, 1)
    )
    sim = AerSimulator()
    result = sim.run(qc, shots=512, seed_simulator=42).result()
    raw_counts = result.get_counts()
    counts = {bs[::-1]: c for bs, c in raw_counts.items()}
    from qverify.verifier.classical_check import satisfies

    for bs in counts:
        assert not satisfies(cnf, enc.bitstring_to_assignment(bs))


# ---------------------------------------------------------------------------
# IBMQuantumBackend with stubs (no network)
# ---------------------------------------------------------------------------


class _StubIBMClient:
    """Mimics IBMRuntimeClient without touching the network."""

    def __init__(
        self,
        counts: dict[str, int],
        backend_name: str = "ibm_kingston",
        job_id: str = "stub-job-id",
    ) -> None:
        self.counts = counts
        self.backend_name = backend_name
        self.job_id = job_id
        self.run_calls: list[dict[str, Any]] = []
        self.least_busy_calls: list[int] = []

    def least_busy_heron(self, min_qubits: int = 5) -> str:
        self.least_busy_calls.append(min_qubits)
        return self.backend_name

    def run(
        self,
        circuit: QuantumCircuit,
        *,
        backend_name: str,
        shots: int,
        optimization_level: int = 3,
    ) -> IBMRunResult:
        self.run_calls.append(
            {
                "backend_name": backend_name,
                "shots": shots,
                "optimization_level": optimization_level,
                "circuit_qubits": circuit.num_qubits,
            }
        )
        return IBMRunResult(
            job_id=self.job_id,
            backend_name=backend_name,
            counts=self.counts,
            raw_metadata={"shots": shots, "optimization_level": optimization_level},
        )


class _FakeSettingsMissingTokens:
    """Settings stub whose require() always raises like the real one would."""

    def require(self, field_name: str) -> str:
        raise RuntimeError(
            f"Required setting '{field_name}' is not set. Add it to your environment or .env file."
        )


def test_ibm_backend_construction_does_not_hit_network() -> None:
    # Even with no client and no settings prep, construction must not error.
    backend = IBMQuantumBackend()
    assert backend.name == ""


def test_ibm_backend_no_token_raises_at_execute_not_construct(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qverify.utils import config as config_mod

    monkeypatch.setattr(config_mod, "load_settings", lambda: _FakeSettingsMissingTokens())

    backend = IBMQuantumBackend()  # no error here
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    with pytest.raises(RuntimeError, match="ibm_quantum_token"):
        backend.execute_grover(enc.encode_clauses(), enc.n_qubits, 1, shots=4, seed=42)


def test_ibm_backend_name_empty_before_run_populated_after() -> None:
    stub = _StubIBMClient(counts={"00": 256, "11": 256}, backend_name="ibm_kingston")
    backend = IBMQuantumBackend(client=stub)
    assert backend.name == ""
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    enc = AtomEncoder(cnf)
    backend.execute_grover(
        enc.encode_clauses(),
        enc.n_qubits,
        optimal_iterations(enc.n_qubits, 1),
        shots=4,
        seed=42,
    )
    assert backend.name == "ibm_kingston"


def test_ibm_backend_explicit_backend_name_skips_least_busy_lookup() -> None:
    stub = _StubIBMClient(counts={"0": 4}, backend_name="ibm_overridden")
    backend = IBMQuantumBackend(backend_name="ibm_brisbane", client=stub)
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    counts, metadata = backend.execute_grover(
        enc.encode_clauses(), enc.n_qubits, 1, shots=4, seed=42
    )
    assert stub.least_busy_calls == []
    assert stub.run_calls[0]["backend_name"] == "ibm_brisbane"
    assert metadata["backend_name"] == "ibm_brisbane"
    assert counts == {"0": 4}


def test_ibm_backend_resolves_least_busy_lazily_when_no_explicit_name() -> None:
    stub = _StubIBMClient(counts={"1": 4}, backend_name="ibm_least_busy_target")
    backend = IBMQuantumBackend(client=stub)
    cnf = _cnf(_clause(_atom("P")))
    enc = AtomEncoder(cnf)
    backend.execute_grover(enc.encode_clauses(), enc.n_qubits, 1, shots=4, seed=42)
    assert stub.least_busy_calls == [backend._min_qubits]


def test_ibm_backend_execute_grover_returns_counts_and_metadata_with_job_id() -> None:
    stub = _StubIBMClient(
        counts={"11": 800, "00": 100, "10": 80, "01": 44},
        backend_name="ibm_kingston",
        job_id="stub-job-12345",
    )
    backend = IBMQuantumBackend(client=stub)
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    enc = AtomEncoder(cnf)
    counts, metadata = backend.execute_grover(
        enc.encode_clauses(),
        enc.n_qubits,
        optimal_iterations(enc.n_qubits, 1),
        shots=1024,
        seed=42,
    )
    assert counts == {"11": 800, "00": 100, "10": 80, "01": 44}
    assert metadata["job_id"] == "stub-job-12345"
    assert metadata["backend_name"] == "ibm_kingston"


def test_ibm_backend_through_run_grover_picks_top_satisfying_bitstring() -> None:
    # Stub returns a counts histogram mimicking a real run on (P)∧(Q):
    # most-frequent bitstring '11' is the unique satisfier.
    stub = _StubIBMClient(
        counts={"11": 700, "00": 120, "10": 110, "01": 94},
        backend_name="ibm_kingston",
    )
    backend = IBMQuantumBackend(client=stub)
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = run_grover(cnf, shots=1024, seed=42, backend=backend, mode="entailment")
    assert result.contradiction_found is True
    assert result.counter_model is not None
    assert result.counter_model.assignment == {"P": True, "Q": True}
    assert result.backend_name == "ibm_kingston"


def test_ibm_backend_through_run_grover_unsat_when_no_measured_satisfier() -> None:
    stub = _StubIBMClient(counts={"00": 600, "01": 200, "10": 200})
    backend = IBMQuantumBackend(client=stub)
    cnf = _cnf(_clause(_atom("P")), _clause(_atom("Q")))
    result = run_grover(cnf, shots=1000, seed=42, backend=backend, mode="entailment")
    assert result.contradiction_found is False
    assert result.counter_model is None


# ---------------------------------------------------------------------------
# IBMRuntimeClient.least_busy_heron — regression for the kwargs-shape bug
# ---------------------------------------------------------------------------


class _FakeBackend:
    """Stand-in for a Qiskit ``BackendV2`` with a ``processor_type`` attribute."""

    def __init__(self, name: str, processor_type: Any) -> None:
        self.name = name
        self.processor_type = processor_type


class _RecordingService:
    """QiskitRuntimeService stub that records least_busy() calls verbatim."""

    def __init__(self, return_backend: _FakeBackend) -> None:
        self._return_backend = return_backend
        self.least_busy_calls: list[dict[str, Any]] = []

    def least_busy(self, **kwargs: Any) -> _FakeBackend:
        self.least_busy_calls.append(kwargs)
        return self._return_backend


def test_least_busy_heron_uses_filters_kwarg() -> None:
    """Regression: must call service.least_busy with kwargs, not a list of backends."""
    fake_target = _FakeBackend("ibm_kingston", {"family": "Heron"})
    fake_service = _RecordingService(fake_target)

    client = IBMRuntimeClient(token="dummy", instance="dummy")
    client._service = fake_service  # type: ignore[assignment]

    name = client.least_busy_heron(min_qubits=5)

    assert name == "ibm_kingston"
    assert len(fake_service.least_busy_calls) == 1

    call = fake_service.least_busy_calls[0]
    assert call.get("min_num_qubits") == 5
    assert call.get("simulator") is False
    assert call.get("operational") is True
    assert callable(call.get("filters"))


def test_least_busy_heron_filter_accepts_heron_rejects_others() -> None:
    """The captured `filters` callable must distinguish Heron from non-Heron."""
    fake_target = _FakeBackend("ibm_kingston", {"family": "Heron"})
    fake_service = _RecordingService(fake_target)

    client = IBMRuntimeClient(token="dummy", instance="dummy")
    client._service = fake_service  # type: ignore[assignment]
    client.least_busy_heron(min_qubits=5)

    filt = fake_service.least_busy_calls[0]["filters"]

    heron_dict = _FakeBackend("ibm_a", {"family": "Heron"})
    heron_string = _FakeBackend("ibm_b", "heron")
    eagle_dict = _FakeBackend("ibm_c", {"family": "Eagle"})
    falcon_string = _FakeBackend("ibm_d", "Falcon")

    assert filt(heron_dict) is True
    assert filt(heron_string) is True
    assert filt(eagle_dict) is False
    assert filt(falcon_string) is False


def test_least_busy_heron_min_qubits_passed_through() -> None:
    fake_target = _FakeBackend("ibm_kingston", {"family": "Heron"})
    fake_service = _RecordingService(fake_target)

    client = IBMRuntimeClient(token="dummy", instance="dummy")
    client._service = fake_service  # type: ignore[assignment]
    client.least_busy_heron(min_qubits=12)

    assert fake_service.least_busy_calls[0]["min_num_qubits"] == 12
