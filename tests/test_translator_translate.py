"""Integration tests for the Translator using stub LLM backends."""

from __future__ import annotations

import pytest

from qverify.translator import CNF, TranslationError, Translator
from qverify.translator.few_shot import build_prompt
from qverify.translator.llm import StubBackend

# ---------------------------------------------------------------------------
# Test infrastructure
# ---------------------------------------------------------------------------


class SequentialBackend:
    """Returns canned responses in order, one per ``generate`` call.

    Used to drive the Translator's retry path, where the same prompt may be
    issued multiple times but should receive different responses.
    """

    def __init__(self, responses: list[str]) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
        self.calls.append(prompt)
        if not self._responses:
            raise RuntimeError("SequentialBackend exhausted")
        return self._responses.pop(0)


ATOMIC_JSON = '{"clauses":[{"literals":[{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'
UNIVERSAL_JSON = (
    '{"clauses":[{"literals":['
    '{"predicate":"Bird","args":["x"],"negated":true},'
    '{"predicate":"Flies","args":["x"],"negated":false}]}]}'
)
NEGATION_JSON = (
    '{"clauses":[{"literals":[{"predicate":"Flies","args":["tweety"],"negated":true}]}]}'
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_atomic_statement_translates() -> None:
    statement = "The penguin is a bird."
    backend = StubBackend({build_prompt(statement): ATOMIC_JSON})
    cnf = Translator(backend).translate(statement)
    assert isinstance(cnf, CNF)
    assert cnf.clauses[0].literals[0].predicate == "Bird"
    assert cnf.clauses[0].literals[0].args == ("penguin",)
    assert cnf.clauses[0].literals[0].negated is False


def test_universal_implication_translates() -> None:
    statement = "All birds can fly."
    backend = StubBackend({build_prompt(statement): UNIVERSAL_JSON})
    cnf = Translator(backend).translate(statement)
    lits = cnf.clauses[0].literals
    assert len(lits) == 2
    assert lits[0].predicate == "Bird" and lits[0].negated is True
    assert lits[1].predicate == "Flies" and lits[1].negated is False


def test_negation_translates() -> None:
    statement = "Tweety cannot fly."
    backend = StubBackend({build_prompt(statement): NEGATION_JSON})
    cnf = Translator(backend).translate(statement)
    assert cnf.clauses[0].literals[0].negated is True


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_second_attempt() -> None:
    backend = SequentialBackend(["this is not JSON at all", ATOMIC_JSON])
    cnf = Translator(backend, max_retries=3).translate("The penguin is a bird.")
    assert cnf.clauses[0].literals[0].predicate == "Bird"
    assert len(backend.calls) == 2


def test_retry_succeeds_on_third_attempt_with_stricter_prompt() -> None:
    backend = SequentialBackend(["bad output 1", "still bad", ATOMIC_JSON])
    translator = Translator(backend, max_retries=3)
    cnf = translator.translate("The penguin is a bird.")
    assert cnf.clauses[0].literals[0].predicate == "Bird"
    assert len(backend.calls) == 3
    # The third prompt should differ from the first two (it appends feedback).
    assert backend.calls[0] == backend.calls[1]
    assert backend.calls[2] != backend.calls[0]
    assert "previous output" in backend.calls[2].lower()


def test_gives_up_after_max_retries_and_raises() -> None:
    backend = SequentialBackend(["bad", "still bad", "yet another bad"])
    with pytest.raises(TranslationError) as excinfo:
        Translator(backend, max_retries=3).translate("The penguin is a bird.")
    assert len(excinfo.value.attempts) == 3
    assert all("bad" in attempt[0] for attempt in excinfo.value.attempts)


def test_translation_error_carries_all_attempts() -> None:
    backend = SequentialBackend(["bogus 1", "bogus 2", "bogus 3"])
    with pytest.raises(TranslationError) as excinfo:
        Translator(backend, max_retries=3).translate("Hello.")
    raw_outputs = [a[0] for a in excinfo.value.attempts]
    assert raw_outputs == ["bogus 1", "bogus 2", "bogus 3"]


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_empty_statement_raises_meaningful_error() -> None:
    backend = StubBackend({})
    with pytest.raises(TranslationError, match="empty"):
        Translator(backend).translate("")


def test_whitespace_only_statement_raises() -> None:
    backend = StubBackend({})
    with pytest.raises(TranslationError, match="empty"):
        Translator(backend).translate("   \n  ")


def test_multi_sentence_input_rejected() -> None:
    backend = StubBackend({})
    with pytest.raises(TranslationError, match="multiple sentences"):
        Translator(backend).translate("All birds fly. Penguins are birds.")


def test_question_then_statement_rejected() -> None:
    backend = StubBackend({})
    with pytest.raises(TranslationError, match="multiple sentences"):
        Translator(backend).translate("Is this a bird? It is.")


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_max_retries_must_be_positive() -> None:
    with pytest.raises(ValueError, match="max_retries"):
        Translator(StubBackend({}), max_retries=0)


def test_backend_failure_wraps_in_translation_error() -> None:
    class BoomBackend:
        def generate(self, prompt: str, max_new_tokens: int = 512) -> str:
            raise RuntimeError("network down")

    with pytest.raises(TranslationError, match="backend failed"):
        Translator(BoomBackend()).translate("The penguin is a bird.")
