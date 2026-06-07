"""cProfile harness for QVerify critical paths.

Produces .prof files under audit/profiles/ for:
  1. cnf_parse      - translator.parser.parse_llm_output (defensive + fast path)
  2. oracle_build   - verifier.oracle.build_sat_oracle (closure construction)
  3. grover_sim     - verifier.grover.run_grover end-to-end on the PennyLane
                      simulator (exercises oracle apply + diffuser + statevector)
  4. ibm_submit_cpu - verifier.grover_circuit.build_grover_qiskit_circuit
                      (the offline-measurable CPU portion of the IBM submit
                      path; the network transpile+submit cannot run without
                      credentials)

Run: .venv/bin/python audit/profile_harness.py
This is an audit artifact; it imports qverify but does not modify it.
"""

from __future__ import annotations

import cProfile
import json
import pstats
import time
from pathlib import Path

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.translator.parser import parse_llm_output
from qverify.verifier.backends import PennyLaneBackend
from qverify.verifier.encoding import AtomEncoder
from qverify.verifier.grover import run_grover
from qverify.verifier.grover_circuit import build_grover_qiskit_circuit
from qverify.verifier.oracle import build_sat_oracle

OUT = Path(__file__).resolve().parent / "profiles"
OUT.mkdir(parents=True, exist_ok=True)

_SAMPLE_JSON = json.dumps(
    {
        "entities": ["Tom", "Whiskers"],
        "clauses": [
            {"literals": [{"predicate": "Cat", "args": ["x"], "negated": True},
                          {"predicate": "HasFur", "args": ["x"], "negated": False}]},
            {"literals": [{"predicate": "Cat", "args": ["Tom"], "negated": False}]},
        ],
    }
)


def _mk_cnf(rendered: dict) -> CNF:
    return CNF(
        clauses=tuple(
            Clause(literals=tuple(Literal(**lit) for lit in clause))
            for clause in rendered["clauses"]
        )
    )


def _load_example(example_id: str) -> CNF:
    data = json.loads(Path("benchmarks/qverify_mini/dataset.json").read_text())
    for ex in data:
        if ex["id"] == example_id:
            return _mk_cnf(ex["rendered_cnf"])
    raise KeyError(example_id)


def _profile(name: str, fn, repeats: int = 1) -> float:
    pr = cProfile.Profile()
    t0 = time.monotonic()
    pr.enable()
    for _ in range(repeats):
        fn()
    pr.disable()
    dt = time.monotonic() - t0
    path = OUT / f"{name}.prof"
    pr.dump_stats(str(path))
    print(f"\n=== {name}  ({repeats}x, {dt:.3f}s total, {dt / repeats * 1e3:.3f} ms/call) ===")
    st = pstats.Stats(pr).sort_stats("cumulative")
    st.print_stats(8)
    return dt


def main() -> None:
    # 1. CNF parse (2000x to get a stable profile)
    _profile("cnf_parse", lambda: parse_llm_output(_SAMPLE_JSON), repeats=2000)

    # 2. Oracle closure construction (10000x; tiny per-call, profile shows shape)
    cnf6 = _load_example("g09")  # 6 atoms, 7 clauses
    enc6 = AtomEncoder(cnf6)
    encoded6 = enc6.encode_clauses()
    _profile(
        "oracle_build",
        lambda: build_sat_oracle(encoded6, enc6.n_qubits),
        repeats=10000,
    )

    # 3. Grover end-to-end on simulator (6 atoms / 7 clauses / 14 wires).
    #    Exercises oracle apply + diffuser + statevector sampling.
    _profile(
        "grover_sim_6atoms",
        lambda: run_grover(cnf6, backend=PennyLaneBackend(), shots=1024, seed=42),
        repeats=1,
    )

    # 4. IBM-submit CPU portion: build the Qiskit circuit (no network).
    _profile(
        "ibm_submit_cpu",
        lambda: build_grover_qiskit_circuit(encoded6, enc6.n_qubits, 6),
        repeats=200,
    )


if __name__ == "__main__":
    main()
