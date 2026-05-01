"""Defensive parser from raw LLM output to validated TranslationResult."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from qverify.translator.cnf import CNF
from qverify.translator.schema import TranslationSchema
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


def _capitalize_first(s: str) -> str:
    """Capitalize the first character only, leaving the rest untouched.

    Unlike :py:meth:`str.capitalize` which lowercases the tail (so ``"IBM"``
    becomes ``"Ibm"``), this preserves any internal capitalization the
    model chose deliberately. Empty strings pass through unchanged.
    """
    return (s[:1].upper() + s[1:]) if s else s


def _normalize_entity_casing(data: dict[str, object]) -> dict[str, object]:
    """Capitalize lowercase entity names so they pass Universe validation.

    RuleTaker premises use bare lowercase animal/person names ("cat",
    "cow", "tom") which the verifier's :func:`is_free_variable` heuristic
    treats as free variables, and which :class:`Universe` rejects as
    constants. We capitalize the first character of every declared entity
    and rewrite every literal arg whose lowercased form matches a
    declared entity to use the same capitalized form. Args that do *not*
    match a declared entity (free variables like ``x``) are left alone.

    The returned dict is a shallow copy; nested structures may be
    rebuilt as new lists/dicts so the caller can mutate freely.
    """
    if not isinstance(data, dict):
        return data

    raw_entities = data.get("entities")
    if not isinstance(raw_entities, list) or not all(isinstance(e, str) for e in raw_entities):
        return data

    rename: dict[str, str] = {}
    new_entities: list[str] = []
    for entity in raw_entities:
        capped = _capitalize_first(entity)
        rename[entity.lower()] = capped
        new_entities.append(capped)

    out = dict(data)
    out["entities"] = new_entities

    raw_clauses = data.get("clauses")
    if isinstance(raw_clauses, list):
        new_clauses: list[object] = []
        for clause in raw_clauses:
            if not isinstance(clause, dict):
                new_clauses.append(clause)
                continue
            new_clause = dict(clause)
            literals = clause.get("literals")
            if isinstance(literals, list):
                new_literals: list[object] = []
                for lit in literals:
                    if not isinstance(lit, dict):
                        new_literals.append(lit)
                        continue
                    new_lit = dict(lit)
                    args = lit.get("args")
                    if isinstance(args, list):
                        new_lit["args"] = [
                            rename[a.lower()] if isinstance(a, str) and a.lower() in rename else a
                            for a in args
                        ]
                    new_literals.append(new_lit)
                new_clause["literals"] = new_literals
            new_clauses.append(new_clause)
        out["clauses"] = new_clauses

    return out


def parse_llm_output(raw: str) -> TranslationResult:
    """Extract and validate a TranslationResult from raw LLM output.

    Two paths:

    1. **Fast path** — when the constrained-generation backend
       (:class:`qverify.translator.llm.Gemma4StructuredBackend`) emits
       output, the entire string is already a valid
       :class:`TranslationSchema` JSON document. The fast path validates
       directly against the schema with no fence-stripping.
    2. **Defensive path** — for stub outputs and any free-form-emitting
       backend, strip BOM, markdown fences, and surrounding prose; pull
       out the first balanced ``{...}``; then re-run the schema
       validation. This is the original Phase 4.5 logic, kept as a
       safety net.

    Either path ends in the same entities↔literal-args consistency check
    so semantic mismatches still raise :class:`TranslationParseError`
    even when the syntax was guaranteed by constrained generation.
    """
    if not isinstance(raw, str) or not raw.strip():
        raise TranslationParseError("LLM output is empty", raw=raw or "")

    schema = _try_parse_schema_directly(raw)
    if schema is None:
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

        data = _normalize_entity_casing(data)
        cnf = _build_cnf(data, raw)
        universe = _build_universe(data, raw)
    else:
        # Round-trip the schema through dict so the same casing
        # normalization runs on the constrained-generation fast path.
        normalized = _normalize_entity_casing(schema.model_dump())
        cnf = _build_cnf(normalized, raw)
        universe = _build_universe(normalized, raw)

    _validate_entities_vs_literal_args(cnf, universe, raw)
    return TranslationResult(cnf=cnf, universe=universe)


def _try_parse_schema_directly(raw: str) -> TranslationSchema | None:
    """Attempt the constrained-generation fast path.

    Returns the validated :class:`TranslationSchema` when the entire
    stripped string parses cleanly, ``None`` otherwise. We swallow the
    exception so the defensive path can run; the defensive path will
    surface a clearer error if both fail.
    """
    stripped = raw.strip()
    if not stripped or not stripped.startswith("{"):
        return None
    try:
        return TranslationSchema.model_validate_json(stripped)
    except ValidationError:
        return None


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
