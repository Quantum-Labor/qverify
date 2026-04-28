"""Shared variable/constant identification helper."""

from __future__ import annotations


def is_free_variable(arg: str) -> bool:
    """Return True if ``arg`` looks like a free first-order variable.

    Recognised as variables:

    - Any single alphabetic character regardless of case — ``x``, ``y``,
      ``z``, ``X``, ``Y``, ``Z``. Constrained-generation backends often
      pick uppercase ``X`` for a universal quantifier even when the
      few-shot examples used lowercase, so we accept both.
    - Short lowercase tokens (length < 4) — ``xs``, ``abc``, ``sk1``.

    Everything else (longer tokens, capitalised proper nouns like
    ``Tom``, all-caps acronyms like ``IBM``) is treated as a ground
    constant and must be declared in the universe.
    """
    if not arg:
        return False
    if len(arg) == 1 and arg.isalpha():
        return True
    return arg[0].islower() and len(arg) < 4
