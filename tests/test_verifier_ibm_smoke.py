"""IBM Quantum hardware smoke tests.

Marked ``slow`` and excluded from CI. Run manually after placing
``IBM_QUANTUM_TOKEN`` and ``IBM_QUANTUM_INSTANCE`` in ``.env``:

    .venv/bin/pytest tests/test_verifier_ibm_smoke.py -v -m slow

Skips if either env var is missing.
"""

from __future__ import annotations

import os

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier import IBMQuantumBackend, verify

pytestmark = [pytest.mark.slow]


@pytest.fixture
def ibm_backend() -> IBMQuantumBackend:
    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        pytest.skip("IBM_QUANTUM_TOKEN not set")
    if not os.environ.get("IBM_QUANTUM_INSTANCE"):
        pytest.skip("IBM_QUANTUM_INSTANCE not set")
    return IBMQuantumBackend()


def test_simple_sat_on_real_hardware(ibm_backend: IBMQuantumBackend) -> None:
    """(P ∨ Q) ∧ (¬P ∨ Q) — 3 of 4 assignments satisfy.

    In consistency mode, SAT means the formula is consistent, so
    contradiction_found=False and counter_model=None.
    """
    cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="P"),
                    Literal(predicate="Q"),
                )
            ),
            Clause(
                literals=(
                    Literal(predicate="P", negated=True),
                    Literal(predicate="Q"),
                )
            ),
        )
    )
    result = verify(cnf, backend=ibm_backend, shots=1024, seed=42)

    assert result.backend_name.startswith("ibm_")
    assert result.contradiction_found is False
    assert result.counter_model is None
    assert result.n_grover_iterations >= 1
    assert result.shots == 1024
    assert len(result.top_measurements) == 4
    assert "job_id" in result.metadata
    job_id = result.metadata["job_id"]
    assert isinstance(job_id, str) and job_id, "job_id must be a non-empty string"

    print(f"\nIBM job ran on: {result.backend_name}")
    print(f"IBM job_id: {job_id}")
    print(f"Top measurements: {result.top_measurements}")
