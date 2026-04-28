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
    """(P ∨ Q) ∧ (¬P ∨ Q) — both satisfying assignments share Q=True."""
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
    assert result.contradiction_found is True
    assert result.counter_model is not None
    assert result.counter_model.assignment["Q"] is True
    print(f"\nIBM job ran on: {result.backend_name}")
    print(f"Top measurements: {result.top_measurements[:5]}")
    print(f"IBM job_id: {result.metadata['job_id']}")
    assert result.metadata["job_id"], "job_id missing from metadata"
    job_id = result.metadata["job_id"]
    assert len(job_id) >= 16, f"job_id too short: {job_id!r}"
    assert job_id.isalnum(), f"job_id has unexpected chars: {job_id!r}"
