"""Grover's search runner over a CNF."""

from __future__ import annotations

import math
from collections import Counter
from typing import Literal

from qverify.translator.cnf import CNF
from qverify.utils.logging import get_logger
from qverify.verifier.backends import Backend, PennyLaneBackend
from qverify.verifier.classical_check import satisfies
from qverify.verifier.encoding import AtomEncoder, VerifierError
from qverify.verifier.types import CounterModel, VerificationResult

MAX_VARIABLES: int = 16
TOP_MEASUREMENTS_KEEP: int = 5

VerifyMode = Literal["entailment", "consistency"]
"""How :func:`run_grover` interprets the satisfying-bitstring outcome.

Both modes share the same Grover circuit (which always searches for a
satisfying assignment of the supplied CNF). They differ only in what
"contradiction" means:

* ``"consistency"`` (default since Phase 6.8): the supplied CNF is
  expected to be ``premises ∧ step``; finding a satisfying assignment
  means the step is consistent with the premises, so
  ``contradiction_found = False``. UNSAT means the step contradicts the
  premises, so ``contradiction_found = True``. ``counter_model`` is
  always ``None`` — for the consistent branch we don't surface the
  satisfying assignment, and for the inconsistent branch there isn't
  one.
* ``"entailment"`` (legacy Phase 3 behaviour): the supplied CNF is
  expected to be ``premises ∧ ¬step``; finding a satisfying assignment
  is a counter-model showing the step is NOT entailed by the premises,
  so ``contradiction_found = True`` and ``counter_model`` carries the
  witness. UNSAT means the step IS entailed.
"""

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
    mode: VerifyMode = "consistency",
) -> VerificationResult:
    """Execute Grover's search on the given CNF and interpret per ``mode``.

    The Grover circuit is identical across modes: it searches for a
    satisfying assignment of ``cnf``. ``mode`` decides what the outcome
    *means*. See :data:`VerifyMode` for the full semantics of each mode.

    The classical post-check (walking the most-frequent measured
    bitstrings until one satisfies the CNF) is unchanged — Grover with
    near-uniform amplitudes still benefits from a top-K rescan.
    """
    if mode not in ("entailment", "consistency"):
        raise ValueError(f"mode must be 'entailment' or 'consistency', got {mode!r}")

    if backend is None:
        backend = PennyLaneBackend()

    encoder = AtomEncoder(cnf)  # may raise VerifierError on free variables
    n_qubits = encoder.n_qubits
    n_clauses = len(cnf.clauses)

    if n_qubits > MAX_VARIABLES:
        raise VerifierError(
            f"CNF has {n_qubits} variables; the verifier accepts at most "
            f"{MAX_VARIABLES} (state-vector simulation cost is exponential)."
        )

    # Empty CNF is trivially satisfied. Both modes agree: no contradiction.
    if n_qubits == 0:
        return VerificationResult(
            contradiction_found=False,
            counter_model=None,
            n_variables=0,
            n_clauses=n_clauses,
            n_grover_iterations=0,
            backend_name=backend.name,
            shots=shots,
            top_measurements=(),
        )

    encoded = encoder.encode_clauses()
    n_iter = optimal_iterations(n_qubits, n_solutions_estimate)

    counts, metadata = backend.execute_grover(encoded, n_qubits, n_iter, shots=shots, seed=seed)

    counter: Counter[str] = Counter(counts)
    top = counter.most_common(TOP_MEASUREMENTS_KEEP)
    top_measurements: tuple[tuple[str, int], ...] = tuple((bs, c) for bs, c in top)

    # Walk the histogram in descending order of frequency, looking for a
    # bitstring whose decoded assignment classically satisfies the CNF.
    found_satisfying_assignment = False
    satisfying_assignment: dict[str, bool] | None = None
    for bits, _count in counter.most_common():
        candidate = encoder.bitstring_to_assignment(bits)
        if satisfies(cnf, candidate):
            found_satisfying_assignment = True
            satisfying_assignment = candidate
            break

    if mode == "consistency":
        contradiction_found = not found_satisfying_assignment
        counter_model: CounterModel | None = None
        if contradiction_found:
            _log.info(
                "Consistency check FAILED after %d iterations: no satisfying "
                "assignment measured (top: %s)",
                n_iter,
                top[0][0] if top else "<none>",
            )
        else:
            _log.info(
                "Consistency check passed after %d iterations: satisfying assignment found",
                n_iter,
            )
    else:  # entailment
        contradiction_found = found_satisfying_assignment
        counter_model = (
            CounterModel(assignment=satisfying_assignment)
            if found_satisfying_assignment and satisfying_assignment is not None
            else None
        )
        if found_satisfying_assignment:
            _log.info(
                "Entailment counter-model found after %d iterations: %s",
                n_iter,
                counter_model,
            )
        else:
            _log.info(
                "Entailment check passed after %d iterations: no counter-model",
                n_iter,
            )

    backend_name_value = metadata.get("backend_name") or backend.name
    backend_name = backend_name_value if isinstance(backend_name_value, str) else backend.name

    return VerificationResult(
        contradiction_found=contradiction_found,
        counter_model=counter_model,
        n_variables=n_qubits,
        n_clauses=n_clauses,
        n_grover_iterations=n_iter,
        backend_name=backend_name,
        shots=shots,
        top_measurements=top_measurements,
        metadata=dict(metadata),
    )
