"""Performance-regression guard for Grover circuit/oracle construction.

Two parts:

* ``test_benchmark_oracle_build`` — uses ``pytest-benchmark`` to record the
  cost of ``build_grover_qiskit_circuit`` (the full oracle + diffusion build)
  at 3 / 6 / 9 / 12 atoms. Always passes; produces timings for ``--benchmark``
  reporting.
* ``test_oracle_build_no_regression`` — recomputes machine-normalized scaling
  ratios (t_n / t_3) and fails if any exceeds the committed baseline in
  ``tests/perf_baseline.json`` by more than 20%. Normalizing by the 3-atom
  build makes the gate independent of raw CPU speed; it catches algorithmic
  blow-ups in gate emission, not slower hardware.

Skipped automatically under coverage / any sys.settrace tool, where timing is
unreliable. Only imports from ``qverify``; no source is modified.
"""

from __future__ import annotations

import json
import sys
import timeit
from pathlib import Path

import pytest

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.grover import optimal_iterations
from qverify.verifier.grover_circuit import build_grover_qiskit_circuit

_BASELINE_PATH = Path(__file__).parent / "perf_baseline.json"
_ATOM_COUNTS = (3, 6, 9, 12)
_TOLERANCE = 1.20  # fail on >20% slowdown vs baseline ratio


def _build_inputs(n_atoms: int):  # noqa: ANN201 - tuple, test helper
    """One clause per atom; deterministic gate count for stable timing."""
    clauses = tuple(Clause(literals=(Literal(predicate=f"P{i}"),)) for i in range(n_atoms))
    cnf = CNF(clauses=clauses)
    enc = AtomEncoder(cnf)
    return enc.encode_clauses(), enc.n_qubits, optimal_iterations(enc.n_qubits)


@pytest.mark.parametrize("n_atoms", _ATOM_COUNTS)
def test_benchmark_oracle_build(benchmark, n_atoms: int) -> None:  # noqa: ANN001
    encoded, n_qubits, n_iter = _build_inputs(n_atoms)
    circuit = benchmark(build_grover_qiskit_circuit, encoded, n_qubits, n_iter)
    # The build must produce the assignment register + clause ancillas + flag.
    assert circuit.num_qubits == n_qubits + n_atoms + 1


def _median_build_seconds(n_atoms: int, *, number: int = 15, repeat: int = 5) -> float:
    encoded, n_qubits, n_iter = _build_inputs(n_atoms)
    samples = timeit.repeat(
        lambda: build_grover_qiskit_circuit(encoded, n_qubits, n_iter),
        number=number,
        repeat=repeat,
    )
    return min(samples) / number  # min = least noise-contaminated estimate


def test_oracle_build_no_regression() -> None:
    if sys.gettrace() is not None:
        pytest.skip("timing unreliable under coverage / tracer")
    if not _BASELINE_PATH.exists():
        pytest.skip(f"no baseline at {_BASELINE_PATH}; run once to create it")

    baseline = json.loads(_BASELINE_PATH.read_text())["ratios_vs_3atoms"]

    t3 = _median_build_seconds(3)
    assert t3 > 0
    for n in (6, 9, 12):
        ratio = _median_build_seconds(n) / t3
        allowed = float(baseline[str(n)]) * _TOLERANCE
        assert ratio <= allowed, (
            f"{n}-atom build scaling regressed: ratio {ratio:.2f} vs "
            f"baseline {baseline[str(n)]:.2f} (allowed {allowed:.2f})"
        )
