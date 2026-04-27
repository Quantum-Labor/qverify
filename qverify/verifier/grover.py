"""Grover's search runner over a CNF."""

from __future__ import annotations

import math
from collections import Counter
from typing import Any

import numpy as np
import pennylane as qml

from qverify.translator.cnf import CNF
from qverify.utils.logging import get_logger
from qverify.verifier.backends import Backend, PennyLaneBackend
from qverify.verifier.classical_check import satisfies
from qverify.verifier.diffusion import build_diffusion
from qverify.verifier.encoding import AtomEncoder, VerifierError
from qverify.verifier.oracle import build_sat_oracle, required_ancillas
from qverify.verifier.types import CounterModel, VerificationResult

MAX_VARIABLES: int = 16
TOP_MEASUREMENTS_KEEP: int = 5

_log = get_logger("qverify.verifier")


def optimal_iterations(n_qubits: int, n_solutions_estimate: int = 1) -> int:
    """Return ``round(π/4 · sqrt(N/M))``, clamped to at least 1 when ``n_qubits > 0``.

    Returns 0 when there are no qubits or when the solution estimate is
    non-positive (so the caller can short-circuit).
    """
    if n_qubits <= 0 or n_solutions_estimate <= 0:
        return 0
    n_states = 2**n_qubits
    raw = math.pi / 4 * math.sqrt(n_states / n_solutions_estimate)
    return max(1, round(raw))


def run_grover(
    cnf: CNF,
    *,
    shots: int = 1024,
    seed: int = 42,
    n_solutions_estimate: int = 1,
    backend: Backend | None = None,
) -> VerificationResult:
    """Execute Grover's search on the given CNF.

    Picks the most-frequent measurement bitstring and verifies classically
    that it satisfies the CNF; only then sets ``contradiction_found=True``.
    If the top measurement does NOT satisfy, falls back to UNSAT (within
    shot-noise limits).
    """
    if backend is None:
        backend = PennyLaneBackend()

    encoder = AtomEncoder(cnf)  # may raise VerifierError on free variables
    n_qubits = encoder.n_qubits
    n_clauses = len(cnf.clauses)

    if n_qubits > MAX_VARIABLES:
        raise VerifierError(
            f"CNF has {n_qubits} variables; the Phase 3 verifier accepts at "
            f"most {MAX_VARIABLES} (state-vector simulation cost is exponential). "
            "Reduce the variable count or wait for a hardware backend."
        )

    # Empty CNF is trivially satisfied by the empty assignment.
    if n_qubits == 0:
        return VerificationResult(
            contradiction_found=True,
            counter_model=CounterModel(assignment={}),
            n_variables=0,
            n_clauses=n_clauses,
            n_grover_iterations=0,
            backend_name=backend.name,
            shots=shots,
            top_measurements=(),
        )

    encoded = encoder.encode_clauses()
    n_anc = required_ancillas(n_clauses)
    total_wires = n_qubits + n_anc
    n_iter = optimal_iterations(n_qubits, n_solutions_estimate)

    oracle = build_sat_oracle(encoded, n_qubits)
    diffusion = build_diffusion(n_qubits)

    device = backend.device(total_wires, shots=shots, seed=seed)

    @qml.set_shots(shots=shots)
    @qml.qnode(device)
    def circuit() -> Any:
        for w in range(n_qubits):
            qml.Hadamard(wires=w)
        for _ in range(n_iter):
            oracle()
            diffusion()
        return qml.sample(wires=list(range(n_qubits)))

    samples = np.asarray(circuit())
    if samples.ndim == 1:
        samples = samples.reshape(-1, 1)

    counts: Counter[str] = Counter()
    for row in samples:
        bitstring = "".join(str(int(b)) for b in row)
        counts[bitstring] += 1

    top = counts.most_common(TOP_MEASUREMENTS_KEEP)
    top_measurements: tuple[tuple[str, int], ...] = tuple((bs, c) for bs, c in top)

    contradiction_found = False
    counter_model: CounterModel | None = None

    # Classical post-check: scan measured bitstrings in descending count order
    # and return the first satisfier. When M is close to N/2 the amplitudes
    # remain near-uniform, so the most-frequent bitstring is essentially
    # random — but a satisfying outcome will still appear high in the count
    # ranking, so a short scan is enough.
    for bits, _count in counts.most_common():
        candidate = encoder.bitstring_to_assignment(bits)
        if satisfies(cnf, candidate):
            contradiction_found = True
            counter_model = CounterModel(assignment=candidate)
            _log.info(
                "Grover found a counter-model after %d iterations: %s",
                n_iter,
                counter_model,
            )
            break
    else:
        if top:
            _log.info(
                "No measured bitstring satisfies the CNF (top: %s); reporting UNSAT",
                top[0][0],
            )

    return VerificationResult(
        contradiction_found=contradiction_found,
        counter_model=counter_model,
        n_variables=n_qubits,
        n_clauses=n_clauses,
        n_grover_iterations=n_iter,
        backend_name=backend.name,
        shots=shots,
        top_measurements=top_measurements,
    )
