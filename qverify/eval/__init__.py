"""Benchmark loaders and evaluation runners."""

from qverify.eval.datasets import (
    DatasetExample,
    DatasetLabel,
    download_all,
    download_proofwriter,
    download_ruletaker,
    last_skip_count,
    load_folio,
    load_proofwriter,
    load_ruletaker,
)
from qverify.eval.metrics import DatasetReport, ExampleResult, build_report
from qverify.eval.oracle import pysat_satisfies
from qverify.eval.runner import evaluate

__all__ = [
    "DatasetExample",
    "DatasetLabel",
    "DatasetReport",
    "ExampleResult",
    "build_report",
    "download_all",
    "download_proofwriter",
    "download_ruletaker",
    "evaluate",
    "last_skip_count",
    "load_folio",
    "load_proofwriter",
    "load_ruletaker",
    "pysat_satisfies",
]
