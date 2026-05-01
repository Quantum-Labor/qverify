"""CLI entry point for the QVerify benchmark harness.

Example:

    python scripts/run_benchmarks.py \\
        --dataset proofwriter \\
        --backend simulator \\
        --max-examples 100 \\
        --output benchmarks/results/proofwriter_simulator

Writes ``<output>/report.json`` plus three PNG charts (accuracy, latency,
qubits). Reads from a fixture/cache JSON when one is supplied; otherwise
falls back to the conventional ``~/.cache/qverify/datasets/<name>/<split>.json``
path (download is out of scope).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qverify.eval.charts import (
    render_accuracy_chart,
    render_latency_chart,
    render_qubit_distribution,
)
from qverify.eval.datasets import (
    last_skip_count,
    load_proofwriter,
    load_ruletaker,
)
from qverify.eval.runner import evaluate
from qverify.utils.logging import get_logger
from qverify.verifier.backends import Backend, IBMQuantumBackend, PennyLaneBackend

_log = get_logger("qverify.eval.cli")

LOADERS = {
    "proofwriter": load_proofwriter,
    "ruletaker": load_ruletaker,
}

# ProofWriter publishes its dev split as "validation"; RuleTaker as "dev".
# When the user does not pass --split, we pick the right name per dataset.
DEFAULT_SPLIT = {
    "proofwriter": "validation",
    "ruletaker": "dev",
}


def _build_backend(name: str) -> Backend:
    if name == "simulator":
        return PennyLaneBackend()
    if name == "ibm":
        return IBMQuantumBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'simulator' or 'ibm'")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QVerify benchmark runner")
    parser.add_argument("--dataset", required=True, choices=sorted(LOADERS.keys()))
    parser.add_argument("--backend", default="simulator", choices=("simulator", "ibm"))
    parser.add_argument(
        "--split",
        default=None,
        help="Split name. Defaults to 'validation' for proofwriter and 'dev' for ruletaker.",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument("--shots", type=int, default=1024)
    parser.add_argument(
        "--input-path",
        type=Path,
        default=None,
        help="Optional explicit fixture/cache JSON path. Defaults to "
        "~/.cache/qverify/datasets/<dataset>/depth-<N>/<split>.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output directory; report.json and chart PNGs go here.",
    )
    args = parser.parse_args(argv)

    loader = LOADERS[args.dataset]
    split = args.split or DEFAULT_SPLIT[args.dataset]
    examples = list(
        loader(
            split=split,
            depth=args.depth,
            max_examples=args.max_examples,
            path=args.input_path,
        )
    )
    skipped_in_load = last_skip_count()
    _log.info(
        "Loaded %d examples for %s/%s (%d malformed records skipped)",
        len(examples),
        args.dataset,
        split,
        skipped_in_load,
    )

    backend = _build_backend(args.backend)
    report = evaluate(
        examples,
        dataset=args.dataset,
        backend=backend,
        shots=args.shots,
    )

    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")

    render_accuracy_chart([report], args.output / "accuracy.png")
    render_latency_chart([report], args.output / "latency.png")
    render_qubit_distribution([report], args.output / "qubits.png")

    summary = {
        "dataset": report.dataset,
        "backend": report.backend,
        "n_examples": report.n_examples,
        "n_skipped_load": skipped_in_load,
        "n_skipped_run": report.n_skipped,
        "accuracy": report.accuracy,
        "avg_seconds": report.avg_seconds,
        "p95_seconds": report.p95_seconds,
        "report_path": str(report_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
