"""Pure rendering for the "Verified on IBM Quantum Hardware" gallery.

No Gradio and no I/O beyond :func:`load_runs`. Everything returns Markdown so
the same output renders inside ``gr.Markdown()`` in the Space and can be
exported into the README later. The single source of truth is
``space/data/hardware_runs.json``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

WORKLOADS_URL = "https://quantum.cloud.ibm.com/workloads?search="

# depth >= threshold -> "deep Grover" group; below -> "shallow".
DEEP_DEPTH_THRESHOLD = 200

_DEFAULT_DATA = Path(__file__).resolve().parent / "data" / "hardware_runs.json"


def load_runs(path: str | Path | None = None) -> list[dict[str, Any]]:
    """Load gallery runs from the JSON data file.

    Returns an empty list when the file is missing, unreadable, or malformed,
    so the renderer can show a clean empty state instead of raising.
    """
    target = Path(path) if path is not None else _DEFAULT_DATA
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    if not isinstance(data, list):
        return []
    return [dict(entry) for entry in data if isinstance(entry, dict)]


def group_by_depth(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Split runs into ``deep`` and ``shallow`` Grover groups, each newest-first."""
    deep = [r for r in runs if _depth(r) >= DEEP_DEPTH_THRESHOLD]
    shallow = [r for r in runs if _depth(r) < DEEP_DEPTH_THRESHOLD]
    deep.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    shallow.sort(key=lambda r: str(r.get("date", "")), reverse=True)
    return {"deep": deep, "shallow": shallow}


def render_featured_card(run: dict[str, Any]) -> str:
    """Render a featured run as Markdown: CNF, mode, verdict, counts, notes."""
    job_id = str(run.get("job_id", ""))
    counts = run.get("counts_top") or []
    counts_str = " · ".join(f"`{bs}`×{n}" for bs, n in counts) if counts else "n/a"
    return (
        f"**`{run.get('cnf', '')}`** — {run.get('mode', '')} mode -> "
        f"**{run.get('verdict', '')}**  \n"
        f"`{run.get('backend', '')}` · {run.get('atoms', '?')} atoms · "
        f"{run.get('shots', '?')} shots · depth {run.get('depth', '?')} · "
        f"{run.get('date', '')}  \n"
        f"Top measurements: {counts_str}  \n"
        f"{run.get('notes', '')}  \n"
        f"[Verify job `{job_id}` on IBM Quantum]({WORKLOADS_URL}{job_id})"
    )


def render_verified_card(run: dict[str, Any]) -> str:
    """Render a verified run as one honest line — no invented CNF or verdict."""
    job_id = str(run.get("job_id", ""))
    return (
        f"- **Verified on hardware** — {run.get('atoms', '?')}-atom CNF · "
        f"{run.get('shots', '?')} shots · depth {run.get('depth', '?')} · "
        f"{run.get('date', '')} · `{run.get('backend', '')}` · "
        f"[`{job_id}`]({WORKLOADS_URL}{job_id})"
    )


def render_gallery(runs: list[dict[str, Any]] | None = None) -> str:
    """Render the full gallery section as Markdown.

    Pass ``runs`` explicitly (used by tests) or omit to load the default data
    file. An empty or missing list yields a friendly empty-state message with
    no broken cards.
    """
    if runs is None:
        runs = load_runs()
    if not runs:
        return (
            "## Verified on IBM Quantum Hardware\n\n_No verified hardware runs are recorded yet._\n"
        )

    groups = group_by_depth(runs)
    parts: list[str] = [
        "## Verified on IBM Quantum Hardware",
        "",
        "Every run below executed a Grover verification circuit on an IBM Heron "
        "r2 processor and is publicly verifiable at the linked Job ID.",
        "",
    ]

    for key, title in (("deep", "Deep Grover circuits"), ("shallow", "Shallow Grover circuits")):
        bucket = groups[key]
        if not bucket:
            continue
        parts.append(f"### {title} (depth {_depth_range(bucket)})")
        parts.append("")
        for run in bucket:
            if run.get("tier") == "featured":
                parts.append(render_featured_card(run))
            else:
                parts.append(render_verified_card(run))
            parts.append("")

    return "\n".join(parts).rstrip() + "\n"


def _depth(run: dict[str, Any]) -> int:
    try:
        return int(run.get("depth", 0))
    except (TypeError, ValueError):
        return 0


def _depth_range(runs: list[dict[str, Any]]) -> str:
    depths = [_depth(r) for r in runs]
    return f"{min(depths)}-{max(depths)}" if depths else "n/a"
