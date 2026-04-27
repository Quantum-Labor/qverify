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

from qverify.translator import CNF, Translator
from qverify.translator.llm import GemmaE2BBackend
from qverify.utils.models import TRANSLATOR_MODEL_ID

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def translator() -> Translator:
    backend = GemmaE2BBackend(model_id=TRANSLATOR_MODEL_ID)
    return Translator(backend, max_retries=3)


def test_atomic_statement_round_trip(translator: Translator) -> None:
    cnf = translator.translate("The penguin is a bird.")
    assert isinstance(cnf, CNF)
    assert len(cnf.clauses) >= 1
    assert any("penguin" in lit.args for cl in cnf.clauses for lit in cl.literals)


def test_universal_statement_round_trip(translator: Translator) -> None:
    cnf = translator.translate("All birds can fly.")
    assert isinstance(cnf, CNF)
    assert len(cnf.clauses) >= 1
    # A correct universal-implication encoding has at least one clause with
    # exactly one negative and one positive literal sharing a variable.
    has_implication_shape = any(
        len(cl.literals) == 2 and {lit.negated for lit in cl.literals} == {True, False}
        for cl in cnf.clauses
    )
    assert has_implication_shape, f"unexpected CNF shape: {cnf}"
