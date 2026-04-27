"""Tests for the defensive LLM-output -> CNF parser."""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF
from qverify.translator.few_shot import EXAMPLES
from qverify.translator.parser import TranslationParseError, parse_llm_output

VALID_JSON = '{"clauses":[{"literals":[{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'


def test_clean_json_parses() -> None:
    cnf = parse_llm_output(VALID_JSON)
    assert isinstance(cnf, CNF)
    assert len(cnf.clauses) == 1
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_json_in_markdown_fence_parses() -> None:
    fenced = f"```json\n{VALID_JSON}\n```"
    cnf = parse_llm_output(fenced)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_json_in_unlabeled_fence_parses() -> None:
    fenced = f"```\n{VALID_JSON}\n```"
    cnf = parse_llm_output(fenced)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_leading_prose_stripped() -> None:
    raw = f"Here is the CNF:\n{VALID_JSON}"
    cnf = parse_llm_output(raw)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_trailing_prose_stripped() -> None:
    raw = f"{VALID_JSON}\n\nLet me know if you need anything else."
    cnf = parse_llm_output(raw)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_leading_and_trailing_prose_stripped() -> None:
    raw = f"Here you go:\n{VALID_JSON}\nAll set."
    cnf = parse_llm_output(raw)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_bom_stripped() -> None:
    raw = "﻿" + VALID_JSON
    cnf = parse_llm_output(raw)
    assert cnf.clauses[0].literals[0].predicate == "Bird"


def test_empty_input_raises() -> None:
    with pytest.raises(TranslationParseError, match="empty"):
        parse_llm_output("")


def test_whitespace_only_input_raises() -> None:
    with pytest.raises(TranslationParseError, match="empty"):
        parse_llm_output("   \n\t  ")


def test_no_json_object_raises() -> None:
    with pytest.raises(TranslationParseError, match="no JSON object"):
        parse_llm_output("This is just prose with no braces at all.")


def test_malformed_json_raises() -> None:
    with pytest.raises(TranslationParseError, match="invalid JSON"):
        parse_llm_output('{"clauses":[}')


def test_schema_mismatch_missing_field_raises() -> None:
    with pytest.raises(TranslationParseError, match="schema"):
        parse_llm_output('{"wrong_key":[]}')


def test_schema_mismatch_lowercase_predicate_raises() -> None:
    bad = '{"clauses":[{"literals":[{"predicate":"bird","args":["penguin"],"negated":false}]}]}'
    with pytest.raises(TranslationParseError):
        parse_llm_output(bad)


def test_schema_mismatch_empty_clause_raises() -> None:
    bad = '{"clauses":[{"literals":[]}]}'
    with pytest.raises(TranslationParseError):
        parse_llm_output(bad)


def test_raw_is_preserved_on_parse_error() -> None:
    raw = "totally bogus input"
    try:
        parse_llm_output(raw)
    except TranslationParseError as exc:
        assert exc.raw == raw
    else:  # pragma: no cover - should always raise
        pytest.fail("expected TranslationParseError")


def test_raw_truncated_in_str_for_long_input() -> None:
    raw = "x" * 1000
    try:
        parse_llm_output(raw)
    except TranslationParseError as exc:
        rendered = str(exc)
        assert "..." in rendered
        assert len(rendered) < len(raw)


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.statement[:50])
def test_every_few_shot_example_parses(example: object) -> None:
    """Sanity check: every demonstration we show the LLM is itself valid CNF."""
    from qverify.translator.few_shot import FewShotExample

    assert isinstance(example, FewShotExample)
    cnf = parse_llm_output(example.cnf_json)
    assert isinstance(cnf, CNF)
    assert len(cnf.clauses) >= 1
