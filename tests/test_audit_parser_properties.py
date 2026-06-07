"""Property-based tests for the translator parser (audit pass).

Uses Hypothesis to generate schema-valid translator payloads and asserts:

* round-trip   — ``parse_llm_output(json(cnf, entities))`` reconstructs the
  same ``CNF`` and a sorted/deduped ``Universe`` of the declared entities;
* idempotence  — parsing the same payload twice yields equal results, and
  re-serializing a parsed result and parsing again is stable;
* defensive path — the same payload wrapped in markdown fences / surrounding
  prose parses to the identical result as the bare-JSON fast path.

These tests only import from ``qverify``; they do not modify any source.
"""

from __future__ import annotations

import json

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from qverify.translator.cnf import CNF
from qverify.translator.parser import parse_llm_output

# --- strategies -------------------------------------------------------------

# Entity / constant: starts uppercase, alnum, length >= 3 so it never matches
# the free-variable heuristic (single alpha char, or lowercase len < 4).
_entity = st.from_regex(r"[A-Z][A-Za-z0-9]{2,6}", fullmatch=True)
# Predicate: CamelCase / snake-free alnum starting uppercase (CNF validator).
_predicate = st.from_regex(r"[A-Z][A-Za-z0-9]{0,5}", fullmatch=True)
# Free variable: single lowercase letter -> is_free_variable() is True and it
# can never collide with an entity (entities are length >= 3).
_freevar = st.sampled_from(["x", "y", "z", "w"])


@st.composite
def _translations(draw: st.DrawFn) -> dict:
    """A schema-valid {entities, clauses} payload with consistent args.

    Every constant argument is drawn from the declared entities, and every
    free variable is a single lowercase letter not present in entities, so
    the parser's entities<->args consistency check always passes.
    """
    entities = draw(st.lists(_entity, min_size=0, max_size=4, unique=True))
    term = st.sampled_from(entities) | _freevar if entities else _freevar

    literal = st.fixed_dictionaries(
        {
            "predicate": _predicate,
            "args": st.lists(term, min_size=0, max_size=2),
            "negated": st.booleans(),
        }
    )
    clause = st.fixed_dictionaries({"literals": st.lists(literal, min_size=1, max_size=3)})
    clauses = draw(st.lists(clause, min_size=1, max_size=4))
    return {"entities": entities, "clauses": clauses}


_SETTINGS = settings(
    max_examples=150,
    deadline=None,  # pydantic warmup can exceed the default 200ms deadline
    suppress_health_check=[HealthCheck.too_slow],
)


# --- properties -------------------------------------------------------------


@_SETTINGS
@given(payload=_translations())
def test_parser_round_trip(payload: dict) -> None:
    raw = json.dumps(payload)
    result = parse_llm_output(raw)

    expected_cnf = CNF.model_validate({"clauses": payload["clauses"]})
    assert result.cnf == expected_cnf
    # Universe is sorted + deduped on construction.
    assert result.universe.constants == tuple(sorted(set(payload["entities"])))


@_SETTINGS
@given(payload=_translations())
def test_parser_idempotent(payload: dict) -> None:
    raw = json.dumps(payload)
    first = parse_llm_output(raw)
    second = parse_llm_output(raw)
    assert first.cnf == second.cnf
    assert first.universe == second.universe

    # Re-serialize the parsed result back into the schema shape and parse
    # again: the fixed point must be stable.
    reserialized = json.dumps(
        {
            "entities": list(first.universe.constants),
            "clauses": first.cnf.model_dump()["clauses"],
        }
    )
    third = parse_llm_output(reserialized)
    assert third.cnf == first.cnf
    assert third.universe == first.universe


@_SETTINGS
@given(payload=_translations())
def test_parser_defensive_path_matches_fast_path(payload: dict) -> None:
    raw = json.dumps(payload)
    fast = parse_llm_output(raw)

    fenced = f"Here is the translation:\n```json\n{raw}\n```\nThat is all."
    prose = f"The model says: {raw}"
    for variant in (fenced, prose):
        defensive = parse_llm_output(variant)
        assert defensive.cnf == fast.cnf
        assert defensive.universe == fast.universe
