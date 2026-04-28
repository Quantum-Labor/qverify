"""Public translation API."""

from __future__ import annotations

import logging
import re

from qverify.translator.few_shot import build_prompt
from qverify.translator.llm import TranslationBackend
from qverify.translator.parser import TranslationParseError, parse_llm_output
from qverify.translator.types import TranslationResult
from qverify.utils.logging import get_logger

_SENTENCE_SPLIT = re.compile(r"[.!?]+")


class TranslationError(RuntimeError):
    """Raised when translation fails after all retry attempts."""

    def __init__(
        self,
        message: str,
        attempts: list[tuple[str, str]] | None = None,
    ) -> None:
        super().__init__(message)
        self.attempts: list[tuple[str, str]] = list(attempts) if attempts else []


class Translator:
    """Convert natural-language statements into validated CNF formulas."""

    def __init__(
        self,
        backend: TranslationBackend,
        max_retries: int = 3,
        logger: logging.Logger | None = None,
    ) -> None:
        if max_retries < 1:
            raise ValueError(f"max_retries must be >= 1, got {max_retries}")
        self._backend: TranslationBackend = backend
        self._max_retries: int = max_retries
        self._log: logging.Logger = logger or get_logger("qverify.translator")

    def translate(self, statement: str) -> TranslationResult:
        """Translate one natural-language statement into a TranslationResult.

        Retries up to ``max_retries`` on parse failure. The first re-attempt
        uses the same prompt; subsequent attempts append the previous bad
        output and the parse error to the prompt with an explicit instruction
        to emit JSON only. Raises :class:`TranslationError` with all attempted
        outputs attached if every retry fails.

        The returned :class:`TranslationResult` exposes both the CNF and the
        :class:`Universe` of declared constants — the controller merges
        universes across translator calls and grounds the combined CNF
        before handing it to the verifier.
        """
        self._validate_input(statement)

        base_prompt = build_prompt(statement)
        attempts: list[tuple[str, str]] = []

        for attempt_idx in range(self._max_retries):
            if attempt_idx >= 2 and attempts:
                last_output, last_error = attempts[-1]
                prompt = self._stricter_prompt(base_prompt, last_output, last_error)
            else:
                prompt = base_prompt

            try:
                raw = self._backend.generate(prompt)
            except Exception as exc:
                raise TranslationError(
                    f"backend failed during attempt {attempt_idx + 1}: {exc}",
                    attempts=attempts,
                ) from exc

            try:
                result = parse_llm_output(raw)
            except TranslationParseError as exc:
                attempts.append((raw, str(exc)))
                self._log.warning(
                    "translator attempt %d/%d failed: %s",
                    attempt_idx + 1,
                    self._max_retries,
                    exc,
                )
                continue

            self._log.debug(
                "translator attempt %d succeeded: %d clauses, %d entities",
                attempt_idx + 1,
                len(result.cnf.clauses),
                len(result.universe.constants),
            )
            return result

        raise TranslationError(
            f"translation failed after {self._max_retries} attempts",
            attempts=attempts,
        )

    @staticmethod
    def _validate_input(statement: str) -> None:
        if not isinstance(statement, str) or not statement.strip():
            raise TranslationError("statement is empty")
        sentences = [s for s in _SENTENCE_SPLIT.split(statement.strip()) if s.strip()]
        if len(sentences) > 1:
            raise TranslationError("translate() accepts a single statement; got multiple sentences")

    @staticmethod
    def _stricter_prompt(base_prompt: str, last_output: str, last_error: str) -> str:
        return (
            f"{base_prompt}\n\n"
            f"Your previous output was rejected.\n"
            f"Previous output:\n{last_output}\n\n"
            f"Reason: {last_error}\n\n"
            "Emit ONLY the JSON object on the next line — no markdown fences, "
            "no prose, no explanation. Try again.\n"
            "CNF:"
        )
