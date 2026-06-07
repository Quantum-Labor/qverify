"""Tests for the IBM hardware gallery: data file + pure renderer.

Pure data and string assertions only — no GPU, no IBM credentials, no network.
``space/gallery.py`` is loaded by file path (space/ is not a package).
"""

from __future__ import annotations

import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

_ROOT = Path(__file__).resolve().parent.parent
_DATA = _ROOT / "space" / "data" / "hardware_runs.json"
_GALLERY = _ROOT / "space" / "gallery.py"

BASE_FIELDS = {"job_id", "backend", "date", "shots", "depth", "atoms", "tier"}
FEATURED_FIELDS = BASE_FIELDS | {"cnf", "mode", "verdict", "notes", "counts_top"}


def _load_gallery() -> Any:
    spec = importlib.util.spec_from_file_location("_qv_gallery_test", _GALLERY)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["_qv_gallery_test"] = module
    spec.loader.exec_module(module)
    return module


gallery = _load_gallery()
RUNS: list[dict[str, Any]] = json.loads(_DATA.read_text(encoding="utf-8"))


def test_entry_counts() -> None:
    assert len(RUNS) == 14
    assert sum(1 for r in RUNS if r["tier"] == "featured") == 2
    assert sum(1 for r in RUNS if r["tier"] == "verified") == 12


def test_schema_per_tier() -> None:
    for r in RUNS:
        assert set(r) >= BASE_FIELDS, f"{r.get('job_id')} missing base fields"
        if r["tier"] == "featured":
            assert set(r) >= FEATURED_FIELDS, f"{r['job_id']} missing featured fields"
            assert isinstance(r["counts_top"], list) and r["counts_top"]
        else:
            assert r["tier"] == "verified"


def test_job_ids_unique_and_well_formed() -> None:
    ids = [r["job_id"] for r in RUNS]
    assert len(ids) == len(set(ids)), "job_ids not unique"
    for jid in ids:
        assert re.fullmatch(r"[a-z0-9]{20}", jid), f"malformed job_id: {jid!r}"


def test_render_gallery_contains_all_ids_and_both_backends() -> None:
    md = gallery.render_gallery(RUNS)
    for r in RUNS:
        assert r["job_id"] in md, f"missing {r['job_id']} in output"
    assert "ibm_fez" in md
    assert "ibm_kingston" in md


def test_render_gallery_loads_default_data_when_no_arg() -> None:
    md = gallery.render_gallery()
    assert "Verified on IBM Quantum Hardware" in md
    assert len(md) > 100


def test_empty_state_is_friendly_and_nonempty() -> None:
    md = gallery.render_gallery([])
    assert isinstance(md, str) and md.strip()
    assert "No verified hardware runs" in md


def test_group_by_depth_split_and_sorted() -> None:
    groups = gallery.group_by_depth(RUNS)
    assert len(groups["deep"]) == 4
    assert len(groups["shallow"]) == 10
    for bucket in groups.values():
        dates = [r["date"] for r in bucket]
        assert dates == sorted(dates, reverse=True), "group not newest-first"


def test_featured_cards_show_cnf_and_verdict() -> None:
    md = gallery.render_gallery(RUNS)
    assert "(P ∨ Q) ∧ (¬P ∨ Q)" in md
    assert "consistent" in md


def test_verified_cards_invent_no_cnf_or_verdict() -> None:
    for r in RUNS:
        if r["tier"] == "verified":
            card = gallery.render_verified_card(r)
            assert "∨" not in card, "verified card must not show a formula"
            assert "verdict" not in card.lower()
            assert r["job_id"] in card
