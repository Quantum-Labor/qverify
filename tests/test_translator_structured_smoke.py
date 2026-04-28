"""Real-Gemma constrained-generation smoke test.

Marked ``slow`` and ``gpu`` — excluded from CI. Run manually on a CUDA host
with at least 12 GB VRAM after accepting the gated Gemma 4 license at
https://huggingface.co/google/gemma-4-E2B-it and running ``hf auth login``:

    .venv/bin/pytest tests/test_translator_structured_smoke.py -v -m "slow and gpu" -s

The first call compiles the schema into outlines's token-level FSM
(~few seconds) and downloads the model weights if not cached. Subsequent
generates are normal-speed.

PASSED  → constrained generation works on real Gemma 4 E2B; the parser's
          fast path will accept every output by construction.
FAILED  → either outlines/transformers integration is broken, or the
          model is producing wrong *content* despite correct shape — the
          latter is a model-capability issue addressed by switching to
          E4B for translation, not by tweaking the constrained-generation
          setup.
"""

from __future__ import annotations

import pytest

from qverify.translator.few_shot import build_prompt
from qverify.translator.llm import Gemma4StructuredBackend
from qverify.translator.schema import TranslationSchema

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def backend() -> Gemma4StructuredBackend:
    """Real Gemma 4 E2B backend. Module-scoped — schema compilation happens once."""
    return Gemma4StructuredBackend()


def _generate_and_validate(backend: Gemma4StructuredBackend, statement: str) -> TranslationSchema:
    prompt = build_prompt(statement)
    raw = backend.generate(prompt, max_new_tokens=512)
    print(f"\n--- generated for {statement!r} ---")
    print(raw)
    schema = TranslationSchema.model_validate_json(raw)
    return schema


def test_simple_propositional_translation(backend: Gemma4StructuredBackend) -> None:
    """'It rains.' produces clauses with empty entities."""
    schema = _generate_and_validate(backend, "It rains.")
    assert schema.entities == []
    assert len(schema.clauses) >= 1


def test_first_order_translation(backend: Gemma4StructuredBackend) -> None:
    """'All cats have fur.' produces clauses with at least one variable arg."""
    from qverify.verifier._vars import is_free_variable

    schema = _generate_and_validate(backend, "All cats have fur.")
    # Every Cat / Fur / IsCat / HasFur predicate is acceptable — we just
    # require that the universal-quantifier shape (a free variable) is
    # present somewhere in the args. The model may pick lowercase ``x``
    # or uppercase ``X`` for the variable; both are valid per the
    # ``is_free_variable`` heuristic.
    args_seen: set[str] = set()
    for clause in schema.clauses:
        for lit in clause.literals:
            args_seen.update(lit.args)
    assert any(is_free_variable(a) for a in args_seen), (
        f"expected at least one free variable in args; saw {args_seen}"
    )


def test_constant_translation(backend: Gemma4StructuredBackend) -> None:
    """'Tom is a cat.' produces a schema with 'Tom' (or 'tom') in entities."""
    schema = _generate_and_validate(backend, "Tom is a cat.")
    constants = {e.lower() for e in schema.entities}
    assert "tom" in constants, f"expected 'Tom' in entities; got {schema.entities}"
    # The literal arg should match one of the declared entities.
    arg_lowers: set[str] = set()
    for clause in schema.clauses:
        for lit in clause.literals:
            arg_lowers.update(a.lower() for a in lit.args)
    assert "tom" in arg_lowers, f"expected 'Tom' in literal args; got {arg_lowers}"
