"""CLI entry point for the QVerify benchmark harness.

Example (simulator only, fixture/cache must already carry rendered_cnf):

    python scripts/run_benchmarks.py \\
        --dataset proofwriter \\
        --backend simulator \\
        --max-examples 100 \\
        --output benchmarks/results/proofwriter_simulator

Real natural-language datasets need a translator. Pass
``--translate gemma-e2b`` and the runner will lazily load Gemma 4 E2B,
translate each example's premises + hypothesis into a propositional CNF,
ground it, and verify. Translations are cached on disk under
``~/.cache/qverify/translations/<dataset>/depth-<N>/<split>.json``;
``--force-translate`` bypasses the cache.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from qverify.eval.charts import (
    render_accuracy_chart,
    render_latency_chart,
    render_qubit_distribution,
)
from qverify.eval.datasets import (
    DatasetExample,
    last_skip_count,
    load_proofwriter,
    load_ruletaker,
)
from qverify.eval.runner import TranslateFn, evaluate
from qverify.translator.cnf import CNF, Clause
from qverify.translator.cnf import Literal as CNFLiteral
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

TRANSLATE_CHOICES = ("gemma-e2b",)
"""Translator backends accepted by ``--translate``. Add more as wired."""

_TRANSLATION_CACHE_ROOT = Path.home() / ".cache" / "qverify" / "translations"


def _build_backend(name: str) -> Backend:
    if name == "simulator":
        return PennyLaneBackend()
    if name == "ibm":
        return IBMQuantumBackend()
    raise ValueError(f"unknown backend {name!r}; expected 'simulator' or 'ibm'")


def _cnf_to_dict(cnf: CNF) -> dict[str, Any]:
    """Serialize a CNF to the rendered_cnf dict shape used by fixtures."""
    return {
        "clauses": [
            [
                {
                    "predicate": lit.predicate,
                    "args": list(lit.args),
                    "negated": lit.negated,
                }
                for lit in cl.literals
            ]
            for cl in cnf.clauses
        ]
    }


def _cnf_from_dict(raw: dict[str, Any]) -> CNF:
    """Inverse of :func:`_cnf_to_dict` — used to read translation cache hits."""
    clauses: list[Clause] = []
    for clause_raw in raw.get("clauses", []):
        literals = [
            CNFLiteral(
                predicate=lit["predicate"],
                args=tuple(lit.get("args", ())),
                negated=bool(lit.get("negated", False)),
            )
            for lit in clause_raw
        ]
        clauses.append(Clause(literals=tuple(literals)))
    return CNF(clauses=tuple(clauses))


def _translation_cache_path(dataset: str, depth: int, split: str) -> Path:
    return _TRANSLATION_CACHE_ROOT / dataset / f"depth-{depth}" / f"{split}.json"


def _load_translation_cache(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if not isinstance(raw, dict):
        return {}
    return raw


def _save_translation_cache(path: Path, cache: dict[str, dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cache, indent=2), encoding="utf-8")


def _build_gemma_e2b_translate(
    *,
    dataset: str,
    depth: int,
    split: str,
    force_translate: bool,
) -> TranslateFn:
    """Wire the Gemma 4 E2B translator with disk-backed caching.

    Lazy imports torch and transformers so users running simulator-only
    benchmarks without ``--translate`` never pay the GPU/torch cost.
    """
    # Lazy GPU/torch imports. Importing torch alone is fast; loading the
    # model below is the slow part.
    from qverify.translator import Translator
    from qverify.translator.llm import GemmaE2BBackend
    from qverify.utils.models import TRANSLATOR_MODEL_ID
    from qverify.verifier._universe import Universe
    from qverify.verifier.grounding import ground_cnf

    try:
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        device_name = torch.cuda.get_device_name(0) if device == "cuda" else "cpu"
    except Exception:
        device = "cpu"
        device_name = "cpu"

    _log.info("Loading translator %s on device=%s (%s)", TRANSLATOR_MODEL_ID, device, device_name)
    load_start = time.monotonic()
    backend = GemmaE2BBackend(model_id=TRANSLATOR_MODEL_ID)
    translator = Translator(backend, max_retries=3)
    _log.info("Translator loaded in %.1fs", time.monotonic() - load_start)

    cache_path = _translation_cache_path(dataset, depth, split)
    cache = {} if force_translate else _load_translation_cache(cache_path)
    _log.info("Translation cache at %s (%d entries loaded)", cache_path, len(cache))

    dirty = False

    def translate(example: DatasetExample) -> CNF:
        nonlocal dirty

        if example.id in cache:
            return _cnf_from_dict(cache[example.id])

        all_clauses: list[Clause] = []
        all_constants: list[str] = []
        for statement in (*example.premises, example.hypothesis):
            result = translator.translate(statement)
            all_clauses.extend(result.cnf.clauses)
            for const in result.universe.constants:
                if const not in all_constants:
                    all_constants.append(const)

        merged = CNF(clauses=tuple(all_clauses))
        universe = Universe(constants=tuple(all_constants))
        grounded = ground_cnf(merged, universe)

        cache[example.id] = _cnf_to_dict(grounded)
        dirty = True

        # Periodic flush so a crash mid-run does not throw away progress.
        if len(cache) % 16 == 0:
            _save_translation_cache(cache_path, cache)

        return grounded

    # Return the translate fn, but expose a flush hook on it so the
    # outer driver can persist any unsaved entries when the run ends.
    def _flush() -> None:
        if dirty:
            _save_translation_cache(cache_path, cache)

    translate.flush = _flush  # type: ignore[attr-defined]
    return translate


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
        "--translate",
        default=None,
        choices=TRANSLATE_CHOICES,
        help="If set, translate each example's premises+hypothesis into a CNF "
        "before verifying. Currently 'gemma-e2b' is the only supported value.",
    )
    parser.add_argument(
        "--force-translate",
        action="store_true",
        help="Bypass the on-disk translation cache and re-translate every example.",
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

    translate_fn: TranslateFn | None = None
    if args.translate == "gemma-e2b":
        translate_fn = _build_gemma_e2b_translate(
            dataset=args.dataset,
            depth=args.depth,
            split=split,
            force_translate=args.force_translate,
        )

    backend = _build_backend(args.backend)
    try:
        report = evaluate(
            examples,
            dataset=args.dataset,
            backend=backend,
            shots=args.shots,
            translate=translate_fn,
        )
    finally:
        # Persist any unsaved translations even if evaluate() raises.
        if translate_fn is not None and hasattr(translate_fn, "flush"):
            translate_fn.flush()  # type: ignore[attr-defined]

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
        "n_translated": report.n_translated,
        "n_translation_failed": report.n_translation_failed,
        "avg_translation_seconds": report.avg_translation_seconds,
        "report_path": str(report_path),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
