"""Integration tests for the Translator using stub LLM backends."""

from __future__ import annotations

import subprocess
import sys

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


ATOMIC_JSON = (
    '{"entities":["penguin"],'
    '"clauses":[{"literals":[{"predicate":"Bird","args":["penguin"],"negated":false}]}]}'
)
UNIVERSAL_JSON = (
    '{"entities":[],'
    '"clauses":[{"literals":['
    '{"predicate":"Bird","args":["x"],"negated":true},'
    '{"predicate":"Flies","args":["x"],"negated":false}]}]}'
)
NEGATION_JSON = (
    '{"entities":["tweety"],'
    '"clauses":[{"literals":[{"predicate":"Flies","args":["tweety"],"negated":true}]}]}'
)


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_atomic_statement_translates() -> None:
    statement = "The penguin is a bird."
    backend = StubBackend({build_prompt(statement): ATOMIC_JSON})
    result = Translator(backend).translate(statement)
    assert isinstance(result.cnf, CNF)
    assert result.universe.constants == ("penguin",)
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"
    assert result.cnf.clauses[0].literals[0].args == ("penguin",)
    assert result.cnf.clauses[0].literals[0].negated is False


def test_universal_implication_translates() -> None:
    statement = "All birds can fly."
    backend = StubBackend({build_prompt(statement): UNIVERSAL_JSON})
    result = Translator(backend).translate(statement)
    assert result.universe.constants == ()
    lits = result.cnf.clauses[0].literals
    assert len(lits) == 2
    assert lits[0].predicate == "Bird" and lits[0].negated is True
    assert lits[1].predicate == "Flies" and lits[1].negated is False


def test_negation_translates() -> None:
    statement = "Tweety cannot fly."
    backend = StubBackend({build_prompt(statement): NEGATION_JSON})
    result = Translator(backend).translate(statement)
    assert result.universe.constants == ("tweety",)
    assert result.cnf.clauses[0].literals[0].negated is True


def test_translate_returns_translation_result() -> None:
    """The new return type wraps both the CNF and the universe."""
    from qverify.translator.types import TranslationResult

    statement = "The penguin is a bird."
    backend = StubBackend({build_prompt(statement): ATOMIC_JSON})
    result = Translator(backend).translate(statement)
    assert isinstance(result, TranslationResult)


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_retry_succeeds_on_second_attempt() -> None:
    backend = SequentialBackend(["this is not JSON at all", ATOMIC_JSON])
    result = Translator(backend, max_retries=3).translate("The penguin is a bird.")
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"
    assert len(backend.calls) == 2


def test_retry_succeeds_on_third_attempt_with_stricter_prompt() -> None:
    backend = SequentialBackend(["bad output 1", "still bad", ATOMIC_JSON])
    translator = Translator(backend, max_retries=3)
    result = translator.translate("The penguin is a bird.")
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"
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


# ---------------------------------------------------------------------------
# Constrained-generation backend integration with Translator (Phase 6)
# ---------------------------------------------------------------------------


def test_stub_returning_schema_conformant_string_succeeds_no_retries() -> None:
    """The constrained-generation fast path: a schema-valid raw output
    flows through parse_llm_output without fence-stripping or retry."""
    statement = "Tom is a cat."
    schema_json = (
        '{"entities":["Tom"],'
        '"clauses":[{"literals":[{"predicate":"Cat","args":["Tom"],"negated":false}]}]}'
    )
    backend = StubBackend({build_prompt(statement): schema_json})
    result = Translator(backend, max_retries=3).translate(statement)
    assert result.universe.constants == ("Tom",)
    assert result.cnf.clauses[0].literals[0].predicate == "Cat"


def test_retry_logic_still_works_for_malformed_stub_outputs() -> None:
    """Even with the schema fast path, the retry/feedback loop must still
    handle backends that emit garbage on attempt one."""
    backend = SequentialBackend(
        [
            "this is garbage and will not parse",
            ATOMIC_JSON,
        ]
    )
    result = Translator(backend, max_retries=3).translate("The penguin is a bird.")
    assert result.cnf.clauses[0].literals[0].predicate == "Bird"
    assert len(backend.calls) == 2


def test_gemma_e2b_alias_resolves_to_structured_backend() -> None:
    """``GemmaE2BBackend`` is now an alias for the constrained-generation
    backend; existing imports must continue to resolve."""
    from qverify.translator.llm import Gemma4StructuredBackend, GemmaE2BBackend

    assert GemmaE2BBackend is Gemma4StructuredBackend


def test_importing_qverify_translator_does_not_load_outlines() -> None:
    """Spawn a fresh interpreter to assert lazy-load — neither outlines
    nor transformers nor torch may appear in sys.modules just from
    importing qverify.translator."""
    code = (
        "import sys; "
        "import qverify.translator; "
        "leaked = [m for m in sys.modules if any("
        "    m == x or m.startswith(x + '.') for x in ('outlines', 'transformers', 'torch')"
        ")]; "
        "assert not leaked, f'leaked: {leaked}'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
