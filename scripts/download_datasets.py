"""Download benchmark datasets to the local cache.

Example:

    python scripts/download_datasets.py --datasets proofwriter,ruletaker

ProofWriter (Apache 2.0) and RuleTaker (CC BY 4.0) are fetched from
HuggingFace Datasets, transformed into the JSON shape that
:mod:`qverify.eval.datasets` loaders expect, and written to
``~/.cache/qverify/datasets/<name>/depth-<N>/<split>.json``.

FOLIO is intentionally not supported (CC BY-SA share-alike licensing); see
``benchmarks/LICENSE-DATA.md``.
"""

from __future__ import annotations

import argparse
import sys

from qverify.eval.datasets import download_proofwriter, download_ruletaker
from qverify.utils.logging import get_logger

_log = get_logger("qverify.eval.download")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="QVerify dataset downloader")
    parser.add_argument(
        "--datasets",
        default="proofwriter,ruletaker",
        help="Comma-separated subset of {proofwriter, ruletaker}",
    )
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download even if cached JSON files already exist.",
    )
    args = parser.parse_args(argv)

    requested = [d.strip() for d in args.datasets.split(",") if d.strip()]
    valid = {"proofwriter", "ruletaker"}
    unknown = [d for d in requested if d not in valid]
    if unknown:
        print(f"error: unknown datasets requested: {unknown}", file=sys.stderr)
        return 2

    failures: list[tuple[str, str]] = []
    for name in requested:
        _log.info("Downloading %s (depth=%d, force=%s)", name, args.depth, args.force)
        try:
            if name == "proofwriter":
                cache_path = download_proofwriter(depth=args.depth, force=args.force)
            else:
                cache_path = download_ruletaker(depth=args.depth, force=args.force)
        except Exception as exc:
            print(f"error: {name}: {type(exc).__name__}: {exc}", file=sys.stderr)
            failures.append((name, str(exc)))
            continue
        print(f"{name}: cached at {cache_path}")

    if failures:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
