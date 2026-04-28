"""Unit tests for the LLM output schema."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from qverify.translator.parser import parse_llm_output
from qverify.translator.schema import TranslationSchema, _ClauseSchema, _LiteralSchema
from qverify.translator.types import TranslationResult

# ---------------------------------------------------------------------------
# TranslationSchema validation
# ---------------------------------------------------------------------------


def test_empty_entities_and_clauses_validates() -> None:
    schema = TranslationSchema(entities=[], clauses=[])
    assert schema.entities == []
    assert schema.clauses == []


def test_entities_default_factory_to_empty_list() -> None:
    schema = TranslationSchema(clauses=[])
    assert schema.entities == []


def test_entities_with_valid_identifiers() -> None:
    schema = TranslationSchema(entities=["Tom", "Whiskers", "tom_cat"], clauses=[])
    assert "Tom" in schema.entities


def test_literal_predicate_only_validates() -> None:
    lit = _LiteralSchema(predicate="Rain")
    assert lit.predicate == "Rain"
    assert lit.args == []
    assert lit.negated is False


def test_literal_with_args_validates() -> None:
    lit = _LiteralSchema(predicate="Loves", args=["alice", "bobby"])
    assert lit.args == ["alice", "bobby"]


def test_literal_negated_true_validates() -> None:
    lit = _LiteralSchema(predicate="Flies", args=["tweety"], negated=True)
    assert lit.negated is True


def test_clause_with_multiple_literals_validates() -> None:
    clause = _ClauseSchema(
        literals=[
            _LiteralSchema(predicate="Cat", args=["x"], negated=True),
            _LiteralSchema(predicate="Fur", args=["x"]),
        ]
    )
    assert len(clause.literals) == 2


def test_full_schema_validates_for_first_order_implication() -> None:
    schema = TranslationSchema(
        entities=[],
        clauses=[
            _ClauseSchema(
                literals=[
                    _LiteralSchema(predicate="Cat", args=["x"], negated=True),
                    _LiteralSchema(predicate="Fur", args=["x"]),
                ]
            )
        ],
    )
    assert schema.entities == []
    assert len(schema.clauses) == 1


def test_schema_negated_field_coerces_truthy_strings_to_bool() -> None:
    """Pydantic v2 lax mode happily coerces ``'yes'`` / ``'true'`` to ``True``.

    This is fine for our purposes — the constrained-generation backend
    only ever emits real JSON booleans. The test pins the documented
    behaviour so a future stricter-mode flip can't silently change it.
    """
    lit = _LiteralSchema(predicate="P", negated="yes")  # type: ignore[arg-type]
    assert lit.negated is True


def test_schema_predicate_must_be_string() -> None:
    with pytest.raises(ValidationError):
        _LiteralSchema(predicate=42)  # type: ignore[arg-type]


def test_schema_args_must_be_list_of_strings() -> None:
    with pytest.raises(ValidationError):
        _LiteralSchema(predicate="P", args="not a list")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        _LiteralSchema(predicate="P", args=[1, 2])  # type: ignore[list-item]


def test_schema_json_roundtrip_is_identity() -> None:
    original = TranslationSchema(
        entities=["Tom"],
        clauses=[
            _ClauseSchema(
                literals=[_LiteralSchema(predicate="Cat", args=["Tom"])],
            ),
            _ClauseSchema(
                literals=[
                    _LiteralSchema(predicate="Cat", args=["x"], negated=True),
                    _LiteralSchema(predicate="Fur", args=["x"]),
                ]
            ),
        ],
    )
    rendered = original.model_dump_json()
    parsed = TranslationSchema.model_validate_json(rendered)
    assert parsed == original


# ---------------------------------------------------------------------------
# Schema-direct fast path through parse_llm_output
# ---------------------------------------------------------------------------


def test_schema_fast_path_produces_translation_result() -> None:
    """Output that is already valid against TranslationSchema (e.g. from
    constrained generation) flows through the fast path with no fence
    stripping or brace extraction."""
    schema = TranslationSchema(
        entities=["Tom"],
        clauses=[_ClauseSchema(literals=[_LiteralSchema(predicate="Cat", args=["Tom"])])],
    )
    raw = schema.model_dump_json()
    result = parse_llm_output(raw)
    assert isinstance(result, TranslationResult)
    assert result.universe.constants == ("Tom",)
    assert result.cnf.clauses[0].literals[0].predicate == "Cat"


def test_schema_fast_path_matches_freeform_path() -> None:
    """Producing the same TranslationResult through both paths."""
    schema_json = json.dumps(
        {
            "entities": ["penguin"],
            "clauses": [
                {"literals": [{"predicate": "Bird", "args": ["penguin"], "negated": False}]}
            ],
        }
    )
    fenced = f"Here is the CNF:\n```json\n{schema_json}\n```"
    fast = parse_llm_output(schema_json)
    fenced_path = parse_llm_output(fenced)
    assert fast.cnf == fenced_path.cnf
    assert fast.universe == fenced_path.universe


def test_schema_fast_path_real_gemma_4_e2b_sample() -> None:
    """A representative output from a successful constrained-generation run
    on Gemma 4 E2B for the syllogism premise 'Tom is a cat.'"""
    raw = (
        '{"entities":["Tom"],"clauses":'
        '[{"literals":[{"predicate":"Cat","args":["Tom"],"negated":false}]}]}'
    )
    result = parse_llm_output(raw)
    assert result.universe.constants == ("Tom",)
    assert result.cnf.clauses[0].literals[0].predicate == "Cat"
    assert result.cnf.clauses[0].literals[0].args == ("Tom",)


def test_schema_fast_path_rejects_lowercase_predicate() -> None:
    """The schema is permissive but the strict CNF model rejects a
    lowercase-led predicate; this should surface as TranslationParseError."""
    raw = (
        '{"entities":["penguin"],"clauses":'
        '[{"literals":[{"predicate":"bird","args":["penguin"],"negated":false}]}]}'
    )
    from qverify.translator.parser import TranslationParseError

    with pytest.raises(TranslationParseError):
        parse_llm_output(raw)


def test_backward_compat_old_format_without_entities_still_parses() -> None:
    """Old free-form output (no 'entities' key) still produces a valid
    TranslationResult with an empty Universe."""
    raw = '{"clauses":[]}'
    result = parse_llm_output(raw)
    assert result.universe.constants == ()
    assert result.cnf.clauses == ()
