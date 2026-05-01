"""Tests for the defensive LLM-output -> TranslationResult parser."""

from __future__ import annotations

import pytest

from qverify.translator.cnf import CNF
from qverify.translator.few_shot import EXAMPLES
from qverify.translator.parser import TranslationParseError, parse_llm_output
from qverify.translator.types import TranslationResult
from qverify.verifier.grounding import Universe

VALID_JSON = (
    '{"entities":["penguin"],'
    '"clauses":[{"literals":['
    '{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'
)


# ---------------------------------------------------------------------------
# Happy path — clean / fenced / prose-wrapped input
# ---------------------------------------------------------------------------


def test_clean_json_parses() -> None:
    result = parse_llm_output(VALID_JSON)
    assert isinstance(result, TranslationResult)
    assert isinstance(result.cnf, CNF)
    assert isinstance(result.universe, Universe)
    # Lowercase entities are auto-capitalized by the parser so they pass
    # Universe validation (length-3 lowercase tokens look like free vars).
    assert result.universe.constants == ("Penguin",)
    assert len(result.cnf.clauses) == 1
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"
    assert result.cnf.clauses[0].literals[0].args == ("Penguin",)


def test_json_in_markdown_fence_parses() -> None:
    fenced = f"```json\n{VALID_JSON}\n```"
    result = parse_llm_output(fenced)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


def test_json_in_unlabeled_fence_parses() -> None:
    fenced = f"```\n{VALID_JSON}\n```"
    result = parse_llm_output(fenced)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


def test_leading_prose_stripped() -> None:
    raw = f"Here is the CNF:\n{VALID_JSON}"
    result = parse_llm_output(raw)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


def test_trailing_prose_stripped() -> None:
    raw = f"{VALID_JSON}\n\nLet me know if you need anything else."
    result = parse_llm_output(raw)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


def test_leading_and_trailing_prose_stripped() -> None:
    raw = f"Here you go:\n{VALID_JSON}\nAll set."
    result = parse_llm_output(raw)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


def test_bom_stripped() -> None:
    raw = "﻿" + VALID_JSON
    result = parse_llm_output(raw)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


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


def test_schema_mismatch_missing_clauses_raises() -> None:
    with pytest.raises(TranslationParseError, match=r"schema|clauses"):
        parse_llm_output('{"wrong_key":[]}')


def test_schema_mismatch_lowercase_predicate_raises() -> None:
    bad = (
        '{"entities":["penguin"],'
        '"clauses":[{"literals":[{"predicate":"bird","args":["penguin"],"negated":false}]}]}'
    )
    with pytest.raises(TranslationParseError):
        parse_llm_output(bad)


def test_schema_mismatch_empty_clause_raises() -> None:
    bad = '{"entities":[],"clauses":[{"literals":[]}]}'
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


# ---------------------------------------------------------------------------
# Entities-vs-literal-args consistency
# ---------------------------------------------------------------------------


def test_entities_field_populates_universe() -> None:
    raw = (
        '{"entities":["alice","bobby"],'
        '"clauses":[{"literals":['
        '{"predicate":"Loves","args":["alice","bobby"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    # Entities are normalized so the resulting Universe accepts them.
    assert result.universe.constants == ("Alice", "Bobby")
    assert result.cnf.clauses[0].literals[0].args == ("Alice", "Bobby")


def test_lowercase_entities_get_capitalized() -> None:
    """RuleTaker uses bare lowercase animal names (cat, cow). The parser
    normalizes them so they survive the Universe constant validator."""
    raw = (
        '{"entities":["cat","cow"],'
        '"clauses":[{"literals":['
        '{"predicate":"Chases","args":["cat","cow"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    assert result.universe.constants == ("Cat", "Cow")
    assert result.cnf.clauses[0].literals[0].args == ("Cat", "Cow")


def test_already_capitalized_entities_pass_through_unchanged() -> None:
    """Capitalized constants should not be touched."""
    raw = (
        '{"entities":["Tom","IBM"],'
        '"clauses":[{"literals":['
        '{"predicate":"Owns","args":["Tom","IBM"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    # Order is implementation-defined (Universe may sort/dedupe).
    assert set(result.universe.constants) == {"Tom", "IBM"}


def test_free_variables_in_args_are_not_capitalized() -> None:
    """An arg matching a declared entity is rewritten; a bare variable like
    ``x`` (not declared as an entity) must be left alone so the verifier's
    free-variable detector still recognizes it."""
    raw = (
        '{"entities":["cat"],'
        '"clauses":[{"literals":['
        '{"predicate":"Pet","args":["cat"],"negated":false},'
        '{"predicate":"Owns","args":["x","cat"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    assert result.universe.constants == ("Cat",)
    # The "x" variable was not in entities, so it stays lowercase.
    second_lit_args = result.cnf.clauses[0].literals[1].args
    assert second_lit_args == ("x", "Cat")


def test_missing_entities_field_defaults_to_empty_universe() -> None:
    """Backward compatibility: old-format JSON parses with empty universe.

    A warning is logged via the project logger (qverify.utils.logging) but
    that logger has propagation disabled and writes to its own stdout
    handler captured at import time, so neither caplog nor capsys can
    reliably observe the message in tests. The behaviour is verified
    through the empty-Universe outcome instead.
    """
    raw = '{"clauses":[]}'
    result = parse_llm_output(raw)
    assert result.universe.constants == ()


def test_entities_must_be_list_of_strings() -> None:
    raw = '{"entities":"penguin","clauses":[]}'
    with pytest.raises(TranslationParseError, match="must be a JSON array"):
        parse_llm_output(raw)


def test_entities_with_non_string_element_raises() -> None:
    raw = '{"entities":[1,"two"],"clauses":[]}'
    with pytest.raises(TranslationParseError, match="strings"):
        parse_llm_output(raw)


def test_constant_in_literal_must_be_in_entities() -> None:
    bad = (
        '{"entities":[],'
        '"clauses":[{"literals":[{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'
    )
    with pytest.raises(TranslationParseError, match="missing from 'entities'"):
        parse_llm_output(bad)


def test_variable_must_not_be_in_entities() -> None:
    """Variables (lowercase length<4) must not be declared as entities."""
    bad = (
        '{"entities":["x"],'
        '"clauses":[{"literals":[{"predicate":"Bird","args":["x"],"negated":true}]}]}'
    )
    # Universe rejects "x" before we reach the consistency check.
    with pytest.raises(TranslationParseError, match=r"invalid universe|free-variable"):
        parse_llm_output(bad)


def test_propositional_cnf_with_zero_arg_predicates_accepts_empty_entities() -> None:
    raw = (
        '{"entities":[],'
        '"clauses":[{"literals":['
        '{"predicate":"Rain","args":[],"negated":true},'
        '{"predicate":"Wet","args":[],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    assert result.universe.constants == ()
    assert len(result.cnf.clauses) == 1


def test_universal_with_free_variable_accepts_empty_entities() -> None:
    raw = (
        '{"entities":[],'
        '"clauses":[{"literals":['
        '{"predicate":"Cat","args":["x"],"negated":true},'
        '{"predicate":"Fur","args":["x"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    assert result.universe.constants == ()
    assert len(result.cnf.clauses) == 1


# ---------------------------------------------------------------------------
# Few-shot example self-consistency
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("example", EXAMPLES, ids=lambda e: e.statement[:50])
def test_every_few_shot_example_parses(example: object) -> None:
    """Sanity check: every demonstration we show the LLM is itself valid."""
    from qverify.translator.few_shot import FewShotExample

    assert isinstance(example, FewShotExample)
    result = parse_llm_output(example.cnf_json)
    assert isinstance(result, TranslationResult)
    assert len(result.cnf.clauses) >= 1
