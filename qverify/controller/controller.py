"""Controller: stream Gemma reasoning, verify each step, retry on contradiction."""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Protocol

from qverify.controller.correction import format_counter_model_prompt
from qverify.controller.llm import LLMBackend, StubGemmaBackend
from qverify.controller.types import (
    ControllerEvent,
    ControllerResult,
    FinalAnswer,
    ReasoningStepCommitted,
    ReasoningStepGaveUp,
    ReasoningStepRejected,
    ReasoningStepStarted,
    ReasoningStepVerified,
    RejectedStep,
    StreamChunk,
)
from qverify.translator import CNF, Clause
from qverify.translator.translator import Translator
from qverify.utils.logging import get_logger
from qverify.verifier import verify as default_verify
from qverify.verifier.backends import Backend
from qverify.verifier.types import VerificationResult


class _TranslatorLike(Protocol):
    """Anything with a ``translate(statement) -> CNF`` method."""

    def translate(self, statement: str) -> CNF: ...


VerifyFn = Callable[[str, list[str]], VerificationResult]
"""Test-injection point: skip translator+verifier and return a canned result.

Takes ``(step_text, current_premises)`` and returns a
:class:`VerificationResult`. When ``None`` (the production default), the
controller calls the real translator and verifier instead.
"""


_log = get_logger("qverify.controller")


def reason_with_verification(
    problem: str,
    *,
    llm: LLMBackend | None = None,
    translator: _TranslatorLike | None = None,
    verifier_backend: Backend | None = None,
    verify_fn: VerifyFn | None = None,
    max_retries_per_step: int = 3,
    emit: Callable[[ControllerEvent], None] | None = None,
    seed: int | None = None,
    max_new_tokens: int = 4096,
) -> ControllerResult:
    """Run a thinking-mode LLM with verification of every emitted reasoning step.

    See :class:`Controller` for parameter details.
    """
    controller = Controller(
        llm=llm,
        translator=translator,
        verifier_backend=verifier_backend,
        verify_fn=verify_fn,
        max_retries_per_step=max_retries_per_step,
    )
    return controller.reason(
        problem,
        emit=emit,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )


class Controller:
    """Orchestrates LLM streaming, per-step verification, and rewrite injection."""

    def __init__(
        self,
        *,
        llm: LLMBackend | None = None,
        translator: _TranslatorLike | None = None,
        verifier_backend: Backend | None = None,
        verify_fn: VerifyFn | None = None,
        max_retries_per_step: int = 3,
    ) -> None:
        if max_retries_per_step < 0:
            raise ValueError(f"max_retries_per_step must be >= 0, got {max_retries_per_step}")
        self._llm: LLMBackend = llm if llm is not None else _default_llm()
        self._translator: _TranslatorLike | None = translator
        self._verifier_backend: Backend | None = verifier_backend
        self._verify_fn: VerifyFn | None = verify_fn
        self._max_retries_per_step: int = max_retries_per_step
        self._premise_cnf_cache: dict[str, CNF] = {}

    def reason(
        self,
        problem: str,
        *,
        emit: Callable[[ControllerEvent], None] | None = None,
        seed: int | None = None,
        max_new_tokens: int = 4096,
    ) -> ControllerResult:
        emit_event = emit if emit is not None else _noop_emit
        start_wall = time.monotonic()

        messages: list[dict[str, str]] = [{"role": "user", "content": problem}]

        premises: list[str] = []
        committed_steps: list[str] = []
        rejected_records: list[RejectedStep] = []
        gave_up_steps: list[str] = []
        total_verifications = 0
        total_contradictions_found = 0

        answer_buffer = ""
        thinking_buffer = ""

        chunk_iter = self._llm.stream_reasoning(messages, max_new_tokens=max_new_tokens, seed=seed)

        for chunk in chunk_iter:
            if chunk.phase == "answer":
                answer_buffer += chunk.text
                continue

            thinking_buffer += chunk.text
            while "\n\n" in thinking_buffer:
                step, thinking_buffer = thinking_buffer.split("\n\n", 1)
                step = step.strip()
                if not step:
                    continue

                outcome = self._process_step(
                    step=step,
                    premises=premises,
                    emit_event=emit_event,
                    seed=seed,
                )
                total_verifications += outcome.verifications
                total_contradictions_found += outcome.contradictions
                if outcome.committed_text is not None:
                    committed_steps.append(outcome.committed_text)
                    premises.append(outcome.committed_text)
                if outcome.rejected_record is not None:
                    rejected_records.append(outcome.rejected_record)
                if outcome.gave_up_text is not None:
                    gave_up_steps.append(outcome.gave_up_text)

        # Flush any trailing partial step that lacked a final blank line.
        trailing = thinking_buffer.strip()
        if trailing:
            outcome = self._process_step(
                step=trailing,
                premises=premises,
                emit_event=emit_event,
                seed=seed,
            )
            total_verifications += outcome.verifications
            total_contradictions_found += outcome.contradictions
            if outcome.committed_text is not None:
                committed_steps.append(outcome.committed_text)
                premises.append(outcome.committed_text)
            if outcome.rejected_record is not None:
                rejected_records.append(outcome.rejected_record)
            if outcome.gave_up_text is not None:
                gave_up_steps.append(outcome.gave_up_text)

        final_answer = answer_buffer.strip()
        emit_event(FinalAnswer(text=final_answer, timestamp=time.monotonic()))

        return ControllerResult(
            final_answer=final_answer,
            committed_steps=tuple(committed_steps),
            rejected_steps=tuple(rejected_records),
            gave_up_steps=tuple(gave_up_steps),
            total_verifications=total_verifications,
            total_contradictions_found=total_contradictions_found,
            wall_clock_seconds=time.monotonic() - start_wall,
        )

    # ----- internal step machinery -----------------------------------------

    def _process_step(
        self,
        *,
        step: str,
        premises: list[str],
        emit_event: Callable[[ControllerEvent], None],
        seed: int | None,
    ) -> _StepOutcome:
        emit_event(ReasoningStepStarted(step=step, timestamp=time.monotonic()))

        verifications = 0
        contradictions = 0
        first_counter_model = None
        last_counter_model = None
        original_step = step

        attempt = 0
        while True:
            result = self._verify_step(step=step, premises=premises)
            verifications += 1
            emit_event(
                ReasoningStepVerified(
                    step=step,
                    attempt=attempt,
                    contradiction_found=result.contradiction_found,
                    timestamp=time.monotonic(),
                )
            )

            if not result.contradiction_found:
                emit_event(
                    ReasoningStepCommitted(step=step, attempt=attempt, timestamp=time.monotonic())
                )
                rejected_record = None
                if first_counter_model is not None:
                    rejected_record = RejectedStep(
                        original_step=original_step,
                        counter_model=first_counter_model,
                        fixed_at_attempt=attempt,
                        final_accepted_rewrite=step,
                    )
                return _StepOutcome(
                    committed_text=step,
                    rejected_record=rejected_record,
                    gave_up_text=None,
                    verifications=verifications,
                    contradictions=contradictions,
                )

            assert result.counter_model is not None
            contradictions += 1
            last_counter_model = result.counter_model
            if first_counter_model is None:
                first_counter_model = result.counter_model
            emit_event(
                ReasoningStepRejected(
                    step=step,
                    counter_model=result.counter_model,
                    attempt=attempt,
                    timestamp=time.monotonic(),
                )
            )

            if attempt >= self._max_retries_per_step:
                emit_event(
                    ReasoningStepGaveUp(
                        step=step,
                        last_counter_model=last_counter_model,
                        attempts=attempt + 1,
                        timestamp=time.monotonic(),
                    )
                )
                rejected_record = RejectedStep(
                    original_step=original_step,
                    counter_model=first_counter_model,
                    fixed_at_attempt=None,
                    final_accepted_rewrite=None,
                )
                return _StepOutcome(
                    committed_text=None,
                    rejected_record=rejected_record,
                    gave_up_text=original_step,
                    verifications=verifications,
                    contradictions=contradictions,
                )

            # Ask the LLM for a rewrite using a focused mini-conversation.
            step = self._request_step_rewrite(
                rejected_step=step,
                counter_model=result.counter_model,
                premises=premises,
                step_index=len(premises) + 1,
                seed=seed,
            )
            attempt += 1

    def _verify_step(
        self,
        *,
        step: str,
        premises: list[str],
    ) -> VerificationResult:
        if self._verify_fn is not None:
            return self._verify_fn(step, list(premises))

        translator = self._translator if self._translator is not None else _default_translator()
        cnf = self._build_consistency_cnf(premises=premises, step=step, translator=translator)
        return default_verify(cnf, backend=self._verifier_backend)

    def _build_consistency_cnf(
        self,
        *,
        premises: list[str],
        step: str,
        translator: _TranslatorLike,
    ) -> CNF:
        all_clauses: list[Clause] = []
        for premise in premises:
            cached = self._premise_cnf_cache.get(premise)
            if cached is None:
                cached = translator.translate(premise)
                self._premise_cnf_cache[premise] = cached
            all_clauses.extend(cached.clauses)

        # Translate the negation of the candidate step. We rely on the
        # translator to handle the natural-language negation rather than
        # negating CNFs algebraically (which converts to DNF in general).
        neg_cnf = translator.translate(f"It is not the case that {step}")
        all_clauses.extend(neg_cnf.clauses)
        return CNF(clauses=tuple(all_clauses))

    def _request_step_rewrite(
        self,
        *,
        rejected_step: str,
        counter_model: object,
        premises: list[str],
        step_index: int,
        seed: int | None,
    ) -> str:
        from qverify.verifier.types import CounterModel

        assert isinstance(counter_model, CounterModel)
        prompt = format_counter_model_prompt(
            step=rejected_step,
            counter_model=counter_model,
            premises=premises,
            step_index=step_index,
        )
        messages = [{"role": "user", "content": prompt}]
        rewrite = _consume_first_paragraph(self._llm.stream_reasoning(messages, seed=seed))
        return rewrite or rejected_step


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _StepOutcome:
    """Internal record returned by ``_process_step`` for the main reason() loop."""

    __slots__ = (
        "committed_text",
        "contradictions",
        "gave_up_text",
        "rejected_record",
        "verifications",
    )

    def __init__(
        self,
        *,
        committed_text: str | None,
        rejected_record: RejectedStep | None,
        gave_up_text: str | None,
        verifications: int,
        contradictions: int,
    ) -> None:
        self.committed_text = committed_text
        self.rejected_record = rejected_record
        self.gave_up_text = gave_up_text
        self.verifications = verifications
        self.contradictions = contradictions


def _noop_emit(_event: ControllerEvent) -> None:
    return None


def _consume_first_paragraph(stream: Iterator[StreamChunk]) -> str:
    """Pull from a stream until ``\\n\\n`` appears in the thinking buffer."""
    buf = ""
    for chunk in stream:
        if chunk.phase != "thinking":
            continue
        buf += chunk.text
        if "\n\n" in buf:
            paragraph, _ = buf.split("\n\n", 1)
            return paragraph.strip()
    return buf.strip()


def _default_llm() -> LLMBackend:
    """Return a stub LLM with an empty script.

    The production default is ``Gemma4ThinkingBackend()``; we don't pull that
    in here because most callers will inject either a real Gemma backend or
    a test stub. If someone really wants the default-default, they can pass
    ``llm=Gemma4ThinkingBackend()`` explicitly.
    """
    return StubGemmaBackend(scripts=[])


def _default_translator() -> Translator:
    """Build a Translator with the default GemmaE2BBackend (lazy-loaded)."""
    from qverify.translator.llm import GemmaE2BBackend

    return Translator(backend=GemmaE2BBackend())


__all__ = [
    "Controller",
    "VerifyFn",
    "reason_with_verification",
]
