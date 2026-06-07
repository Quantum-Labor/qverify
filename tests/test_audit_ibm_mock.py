"""Mock IBM backend smoke test: submit -> poll -> retrieve, no credentials.

Fakes the qiskit / qiskit-ibm-runtime boundary so the real
``qverify.utils.ibm_client.IBMRuntimeClient`` and
``qverify.verifier.backends.IBMQuantumBackend`` code paths run end to end
without a token or network:

* ``IBMRuntimeClient.run`` — transpile (faked passthrough) -> SamplerV2 (faked)
  -> job_id capture -> result fetch -> little-endian -> qubit-0-leftmost
  reversal -> metadata.
* ``least_busy_heron`` — backend selection via the faked service.
* A submit -> poll(QUEUED->RUNNING->DONE) -> retrieve lifecycle using the same
  fakes, mirroring the intended live-recovery flow.
* ``IBMQuantumBackend.execute_grover`` with an injected client.

Only imports from ``qverify``; no source is modified.
"""

from __future__ import annotations

from typing import Any

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.utils.ibm_client import IBMRunResult, IBMRuntimeClient
from qverify.verifier.backends import IBMQuantumBackend
from qverify.verifier.encoding import AtomEncoder

# Qiskit get_counts() is little-endian by classical-bit index; the client
# reverses each key so qubit 0 ends up leftmost.
_RAW_COUNTS = {"01": 100, "10": 24, "11": 4}
_EXPECTED_COUNTS = {"10": 100, "01": 24, "11": 4}


class _FakeData:
    class meas:  # noqa: N801 - mirrors qiskit's pub_result.data.meas
        @staticmethod
        def get_counts() -> dict[str, int]:
            return dict(_RAW_COUNTS)


class _FakePubResult:
    data = _FakeData()


class _FakeJob:
    """A fake IBM job with a QUEUED -> RUNNING -> DONE status progression."""

    def __init__(self) -> None:
        self._statuses = ["QUEUED", "RUNNING", "DONE"]
        self._i = 0

    def job_id(self) -> str:
        return "mock_job_0001"

    def status(self) -> str:
        s = self._statuses[min(self._i, len(self._statuses) - 1)]
        self._i += 1
        return s

    def result(self) -> list[_FakePubResult]:
        return [_FakePubResult()]


class _FakeSampler:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def run(self, _pubs: Any, *, shots: int) -> _FakeJob:
        return _FakeJob()


class _FakeBackend:
    def __init__(self, name: str = "ibm_mock_heron") -> None:
        self.name = name
        self.processor_type = {"family": "Heron"}


class _FakeService:
    def __init__(self, *_args: Any, **_kwargs: Any) -> None:
        pass

    def backend(self, name: str) -> _FakeBackend:
        return _FakeBackend(name)

    def least_busy(self, **_kwargs: Any) -> _FakeBackend:
        return _FakeBackend("ibm_mock_heron")


def _passthrough_pass_manager(*_args: Any, **_kwargs: Any):
    class _PM:
        @staticmethod
        def run(circuit):
            return circuit  # real QuantumCircuit -> .depth()/.num_qubits work

    return _PM()


@pytest.fixture
def _patched_ibm(monkeypatch: pytest.MonkeyPatch) -> None:
    # IBMRuntimeClient.get_service() does `from qiskit_ibm_runtime import
    # QiskitRuntimeService`; run() imports SamplerV2 and the preset pass
    # manager locally, so patching the module attributes intercepts them.
    monkeypatch.setattr("qiskit_ibm_runtime.QiskitRuntimeService", _FakeService)
    monkeypatch.setattr("qiskit_ibm_runtime.SamplerV2", _FakeSampler)
    monkeypatch.setattr(
        "qiskit.transpiler.preset_passmanagers.generate_preset_pass_manager",
        _passthrough_pass_manager,
    )


def _simple_circuit():
    from qverify.verifier.grover_circuit import build_grover_qiskit_circuit

    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="A"), Literal(predicate="B"))),))
    enc = AtomEncoder(cnf)
    return build_grover_qiskit_circuit(enc.encode_clauses(), enc.n_qubits, 2)


def test_ibm_client_run_submit_and_retrieve(_patched_ibm: None) -> None:
    client = IBMRuntimeClient(token="fake-token", instance="fake/instance")
    result = client.run(_simple_circuit(), backend_name="ibm_mock_heron", shots=128)

    assert isinstance(result, IBMRunResult)
    assert result.job_id == "mock_job_0001"
    assert result.backend_name == "ibm_mock_heron"
    # Bit order reversed from the raw little-endian counts.
    assert result.counts == _EXPECTED_COUNTS
    assert result.raw_metadata["shots"] == 128
    assert result.raw_metadata["transpiled_n_qubits"] >= 1


def test_least_busy_heron_selection(_patched_ibm: None) -> None:
    client = IBMRuntimeClient(token="fake-token", instance="fake/instance")
    assert client.least_busy_heron(min_qubits=5) == "ibm_mock_heron"


def test_submit_poll_retrieve_lifecycle(_patched_ibm: None) -> None:
    """Submit, poll status until terminal, then decode results."""
    terminal = frozenset({"DONE", "ERROR", "CANCELLED"})
    sampler = __import__("qiskit_ibm_runtime").SamplerV2(mode=_FakeBackend())
    job = sampler.run([_simple_circuit()], shots=64)

    assert job.job_id() == "mock_job_0001"

    seen: list[str] = []
    for _ in range(10):
        status = job.status()
        seen.append(status)
        if status in terminal:
            break
    assert seen[0] == "QUEUED"
    assert "RUNNING" in seen
    assert seen[-1] == "DONE"

    raw = job.result()[0].data.meas.get_counts()
    decoded = {bs[::-1]: c for bs, c in raw.items()}
    assert decoded == _EXPECTED_COUNTS


def test_ibm_quantum_backend_with_injected_client() -> None:
    """IBMQuantumBackend.execute_grover with a stand-in client (no network)."""

    class _StubClient:
        def least_busy_heron(self, min_qubits: int = 5) -> str:
            return "ibm_stub"

        def run(self, circuit: Any, *, backend_name: str, shots: int, **_: Any) -> IBMRunResult:
            del circuit
            return IBMRunResult(
                job_id="stub_job",
                backend_name=backend_name,
                counts={"10": shots},
                raw_metadata={"shots": shots},
            )

    backend = IBMQuantumBackend(client=_StubClient())  # type: ignore[arg-type]
    cnf = CNF(clauses=(Clause(literals=(Literal(predicate="A"), Literal(predicate="B"))),))
    enc = AtomEncoder(cnf)
    counts, metadata = backend.execute_grover(
        enc.encode_clauses(), enc.n_qubits, 2, shots=256, seed=0
    )
    assert counts == {"10": 256}
    assert metadata["job_id"] == "stub_job"
    assert metadata["backend_name"] == "ibm_stub"
