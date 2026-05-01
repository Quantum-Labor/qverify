"""Matplotlib chart renderers for benchmark reports.

All renderers are pure functions: same input, same output bytes. No
timestamps or random seeds in chart styling.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless rendering, no display required

import matplotlib.pyplot as plt

from qverify.eval.metrics import DatasetReport


def render_accuracy_chart(reports: Iterable[DatasetReport], out_path: Path) -> Path:
    """Bar chart of per-dataset accuracy. One bar per report."""
    reports_tuple = tuple(reports)
    labels = [r.dataset for r in reports_tuple]
    values = [r.accuracy * 100 for r in reports_tuple]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color="#1f77b4")
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Verifier vs PySAT oracle")
    for i, v in enumerate(values):
        ax.text(i, v + 1, f"{v:.1f}%", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_latency_chart(reports: Iterable[DatasetReport], out_path: Path) -> Path:
    """Box plot of per-example verify_seconds, one box per (dataset, backend)."""
    reports_tuple = tuple(reports)
    data = [[r.verify_seconds for r in rep.results] for rep in reports_tuple]
    labels = [f"{r.dataset}\n({r.backend})" for r in reports_tuple]

    fig, ax = plt.subplots(figsize=(6, 4))
    if any(d for d in data):
        ax.boxplot(data, tick_labels=labels, showfliers=True)
    ax.set_ylabel("verify() seconds")
    ax.set_title("Per-example verifier latency")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path


def render_qubit_distribution(reports: Iterable[DatasetReport], out_path: Path) -> Path:
    """Histogram of n_qubits across every example in every report."""
    reports_tuple = tuple(reports)
    qubits = [r.n_qubits for rep in reports_tuple for r in rep.results]

    fig, ax = plt.subplots(figsize=(6, 4))
    if qubits:
        max_q = max(qubits)
        bins = list(range(0, max(2, max_q + 2)))
        ax.hist(qubits, bins=bins, color="#2ca02c", edgecolor="black", align="left")
    ax.set_xlabel("Qubits per example")
    ax.set_ylabel("Number of examples")
    ax.set_title("Grounded CNF qubit count distribution")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path
