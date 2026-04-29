"""Quantum verification via Grover's search."""

from qverify.verifier.backends import IBMQuantumBackend, PennyLaneBackend
from qverify.verifier.encoding import VerifierError
from qverify.verifier.grover import (
    MAX_VARIABLES,
    VerifyMode,
    optimal_iterations,
    run_grover,
)
from qverify.verifier.types import CounterModel, VerificationResult

verify = run_grover

__all__ = [
    "MAX_VARIABLES",
    "CounterModel",
    "IBMQuantumBackend",
    "PennyLaneBackend",
    "VerificationResult",
    "VerifierError",
    "VerifyMode",
    "optimal_iterations",
    "run_grover",
    "verify",
]
