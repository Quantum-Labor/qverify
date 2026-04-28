"""Shared variable/constant identification helper."""

from __future__ import annotations


def is_free_variable(arg: str) -> bool:
    """Return True if ``arg`` looks like a free first-order variable.

    The Phase 3 heuristic treats short lowercase tokens (length < 4) as
    variables — matches the conventional ``x``, ``y``, ``z``, ``xs``, etc.
    Reserves longer or uppercase-led tokens for ground constants.
    """
    return bool(arg) and arg[0].islower() and len(arg) < 4
