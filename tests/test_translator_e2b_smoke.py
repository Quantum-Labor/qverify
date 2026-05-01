"""Real-LLM smoke test for the Gemma E2B translator backend.

Marked ``slow`` and ``gpu`` — excluded from CI. Run manually on a CUDA host:

    .venv/bin/pytest tests/test_translator_e2b_smoke.py -v -m "slow and gpu"

Loads the model id from ``qverify.utils.models.TRANSLATOR_MODEL_ID`` (Gemma 4
E2B instruct, gated) and translates two well-known statements. Asserts that
the output is a valid CNF; does not assert specific predicate names because a
chat-tuned LLM may choose ``Bird`` vs ``IsBird`` vs ``IsABird`` — all are
correct.
"""

from __future__ import annotations

import pytest

from qverify.translator import CNF, TranslationResult, Translator
from qverify.translator.llm import GemmaE2BBackend
from qverify.utils.models import TRANSLATOR_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def translator() -> Translator:
    backend = GemmaE2BBackend(model_id=TRANSLATOR_MODEL_ID)
    return Translator(backend, max_retries=3)


def test_atomic_statement_round_trip(translator: Translator) -> None:
    result = translator.translate("The penguin is a bird.")
    assert isinstance(result, TranslationResult)
    assert isinstance(result.cnf, CNF)
    assert len(result.cnf.clauses) >= 1
    # Case-insensitive: parser auto-capitalizes lowercase entities so
    # the arg may surface as "Penguin" rather than "penguin".
    assert any(
        any(arg.lower() == "penguin" for arg in lit.args)
        for cl in result.cnf.clauses
        for lit in cl.literals
    )
    print(f"\nCNF: {result.cnf}")
    print(f"Universe: {result.universe}")


def test_universal_statement_round_trip(translator: Translator) -> None:
    result = translator.translate("All birds can fly.")
    assert isinstance(result, TranslationResult)
    assert isinstance(result.cnf, CNF)
    assert len(result.cnf.clauses) >= 1
    # A correct universal-implication encoding has at least one clause with
    # exactly one negative and one positive literal sharing a variable.
    has_implication_shape = any(
        len(cl.literals) == 2 and {lit.negated for lit in cl.literals} == {True, False}
        for cl in result.cnf.clauses
    )
    assert has_implication_shape, f"unexpected CNF shape: {result.cnf}"
    print(f"\nCNF: {result.cnf}")
    print(f"Universe: {result.universe}")


@pytest.mark.xfail(
    reason="known: Gemma 4 E2B sometimes emits ungrounded variable names for universals",
    strict=False,
)
def test_universal_statement_well_formed_args(translator: Translator) -> None:
    """Stricter version of the universal-statement test: every literal arg must
    be either a non-trivial identifier or a declared constant in the universe.

    Gemma 4 E2B at this scale occasionally emits ``" x"`` (leading space) or
    ``"_"`` as a variable name when grounding universals, which the translator
    accepts as a literal arg even though downstream grounding cannot expand it.
    Tracked here as xfail so the failure mode is visible without blocking CI.
    """
    result = translator.translate("All birds can fly.")
    assert isinstance(result.cnf, CNF)
    for cl in result.cnf.clauses:
        for lit in cl.literals:
            for arg in lit.args:
                assert arg.strip() == arg, f"arg has surrounding whitespace: {arg!r}"
                assert len(arg) >= 2 or arg in result.universe.constants, (
                    f"single-character arg {arg!r} is not a declared constant in "
                    f"universe {result.universe.constants}"
                )
    print(f"\nCNF: {result.cnf}")
    print(f"Universe: {result.universe}")
