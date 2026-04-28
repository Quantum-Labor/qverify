"""Defensive parser from raw LLM output to validated TranslationResult."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from qverify.translator.cnf import CNF
from qverify.translator.types import TranslationResult
from qverify.utils.logging import get_logger
from qverify.verifier._universe import Universe
from qverify.verifier._vars import is_free_variable

_FENCE_OPEN = re.compile(r"^```(?:json|JSON)?\s*", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$", re.MULTILINE)

_log = get_logger("qverify.translator.parser")


class TranslationParseError(ValueError):
    """Raised when LLM output cannot be parsed into a valid CNF."""

    def __init__(self, message: str, raw: str = "") -> None:
        super().__init__(message)
        self.raw: str = raw

    def __str__(self) -> str:
        base = super().__str__()
        if self.raw:
            preview = self.raw if len(self.raw) <= 200 else self.raw[:200] + "..."
            return f"{base} | raw output: {preview!r}"
        return base


def parse_llm_output(raw: str) -> TranslationResult:
    """Extract and validate a TranslationResult from raw LLM output.

    Strips a UTF-8 BOM, surrounding whitespace, markdown code fences, and any
    prose before the first ``{`` or after the matching ``}``. Then validates
    the JSON against the schema:

    .. code-block:: json

        {"entities": ["Tom", "Whiskers"],
         "clauses": [{"literals": [...]}]}

    The ``entities`` list becomes a :class:`Universe`. When absent, an
    empty universe is assumed and a warning is logged. Every constant
    appearing as a literal argument must be in ``entities``; every
    free-variable-shaped argument must NOT be. Mismatch raises
    :class:`TranslationParseError` with the full raw output attached.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise TranslationParseError("LLM output is empty", raw=raw or "")

    cleaned = raw.lstrip("﻿").strip()
    cleaned = _FENCE_OPEN.sub("", cleaned, count=1)
    cleaned = _FENCE_CLOSE.sub("", cleaned, count=1)
    cleaned = cleaned.strip()

    json_text = _extract_first_json_object(cleaned)
    if json_text is None:
        raise TranslationParseError("no JSON object found in output", raw=raw)

    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        raise TranslationParseError(f"invalid JSON: {exc.msg}", raw=raw) from exc

    if not isinstance(data, dict):
        raise TranslationParseError(
            f"top-level JSON must be an object, got {type(data).__name__}",
            raw=raw,
        )

    cnf = _build_cnf(data, raw)
    universe = _build_universe(data, raw)
    _validate_entities_vs_literal_args(cnf, universe, raw)
    return TranslationResult(cnf=cnf, universe=universe)


def _build_cnf(data: dict[str, object], raw: str) -> CNF:
    if "clauses" not in data:
        raise TranslationParseError(
            "output does not match CNF schema: missing 'clauses' field",
            raw=raw,
        )
    cnf_payload = {"clauses": data["clauses"]}
    try:
        return CNF.model_validate(cnf_payload)
    except ValidationError as exc:
        raise TranslationParseError(
            f"output does not match CNF schema: {_format_validation_error(exc)}",
            raw=raw,
        ) from exc


def _build_universe(data: dict[str, object], raw: str) -> Universe:
    if "entities" not in data:
        _log.warning("translator output missing 'entities' field; defaulting to empty universe")
        return Universe(constants=())
    entities = data["entities"]
    if not isinstance(entities, list):
        raise TranslationParseError(
            f"'entities' must be a JSON array, got {type(entities).__name__}",
            raw=raw,
        )
    if not all(isinstance(e, str) for e in entities):
        raise TranslationParseError(
            "'entities' must contain only strings",
            raw=raw,
        )
    try:
        return Universe(constants=tuple(entities))
    except ValidationError as exc:
        raise TranslationParseError(
            f"invalid universe: {_format_validation_error(exc)}",
            raw=raw,
        ) from exc


def _validate_entities_vs_literal_args(cnf: CNF, universe: Universe, raw: str) -> None:
    entities_set = set(universe.constants)
    for clause in cnf.clauses:
        for lit in clause.literals:
            for arg in lit.args:
                if is_free_variable(arg):
                    if arg in entities_set:
                        raise TranslationParseError(
                            f"argument {arg!r} matches the free-variable pattern but "
                            f"appears in 'entities' — variables must not be declared "
                            f"as constants",
                            raw=raw,
                        )
                else:
                    if arg not in entities_set:
                        raise TranslationParseError(
                            f"constant {arg!r} appears in a literal but is missing "
                            f"from 'entities' (declared entities: "
                            f"{sorted(entities_set)})",
                            raw=raw,
                        )


def _extract_first_json_object(text: str) -> str | None:
    """Return the first balanced ``{...}`` substring, honouring string escapes."""
    start = text.find("{")
    if start == -1:
        return None

    depth = 0
    in_string = False
    escape = False
    for i in range(start, len(text)):
        ch = text[i]
        if escape:
            escape = False
            continue
        if ch == "\\":
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : i + 1]
    return None


def _format_validation_error(exc: ValidationError) -> str:
    """Render a Pydantic ValidationError as a single short line."""
    parts: list[str] = []
    for err in exc.errors()[:3]:
        loc = ".".join(str(x) for x in err.get("loc", ()))
        parts.append(f"{loc}: {err.get('msg', '')}")
    return "; ".join(parts)
