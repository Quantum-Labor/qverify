"""Defensive parser from raw LLM output to validated CNF."""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from qverify.translator.cnf import CNF

_FENCE_OPEN = re.compile(r"^```(?:json|JSON)?\s*", re.MULTILINE)
_FENCE_CLOSE = re.compile(r"\s*```\s*$", re.MULTILINE)


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


def parse_llm_output(raw: str) -> CNF:
    """Extract and validate a CNF from raw LLM output.

    Strips a UTF-8 BOM, surrounding whitespace, markdown code fences, and any
    prose before the first ``{`` or after the matching ``}``. Then validates
    the JSON against the CNF schema. Raises :class:`TranslationParseError`
    with the full raw output attached on any failure.
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

    try:
        return CNF.model_validate(data)
    except ValidationError as exc:
        raise TranslationParseError(
            f"output does not match CNF schema: {_format_validation_error(exc)}",
            raw=raw,
        ) from exc


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
