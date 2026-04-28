"""Streaming LLM backend Protocol with real Gemma 4 and stub implementations."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Thread
from typing import Any, Protocol, runtime_checkable

from qverify.controller.types import StreamChunk
from qverify.utils.logging import get_logger
from qverify.utils.models import REASONER_E4B_MODEL_ID

# Gemma 4 thinking-mode channel markers. Per the official model cards
# (google/gemma-4-{E2B,E4B,31B}-it), thinking-mode output is structured as:
#     <|channel|>thought\n[Internal reasoning]<|channel|>[Final answer]
# The END marker is a substring of the START marker, so the phase parser
# must consume the start before it begins scanning for the end.
THINKING_START_MARKER = "<|channel|>thought\n"
THINKING_END_MARKER = "<|channel|>"


@runtime_checkable
class LLMBackend(Protocol):
    """A streaming LLM backend with explicit thinking-mode support."""

    name: str

    def stream_reasoning(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 4096,
        seed: int | None = None,
    ) -> Iterator[StreamChunk]:
        """Yield :class:`StreamChunk` objects with text and phase tags."""
        ...


def _split_thinking_answer(
    raw: str,
    *,
    start_marker: str = THINKING_START_MARKER,
    end_marker: str = THINKING_END_MARKER,
) -> tuple[str, str]:
    """Split a complete Gemma 4 thinking-mode response into (thinking, answer).

    Recognizes the channel-token structure
    ``<|channel|>thought\\n...<|channel|>...``. When ``start_marker`` is
    absent the entire string is treated as the answer; when ``start_marker``
    is present but ``end_marker`` never appears after it, every byte after
    the start marker is treated as thinking and the answer comes back empty.
    Both halves are stripped of surrounding whitespace.
    """
    if start_marker not in raw:
        return "", raw.strip()
    after_start = raw.split(start_marker, 1)[1]
    if end_marker not in after_start:
        return after_start.strip(), ""
    thinking, answer = after_start.split(end_marker, 1)
    return thinking.strip(), answer.strip()


def _stream_with_phase(
    text_stream: Iterator[str],
    *,
    start_marker: str = THINKING_START_MARKER,
    end_marker: str = THINKING_END_MARKER,
) -> Iterator[StreamChunk]:
    """Wrap a raw text-chunk stream into phase-tagged :class:`StreamChunk` events.

    Three states:

    1. ``pre-thinking`` — searching for ``start_marker``. Bytes seen here are
       discarded (Gemma 4 with ``enable_thinking=True`` emits the start
       marker first thing, so this state is empty in practice; non-thinking
       output where the start marker never arrives is treated as a final
       answer at end-of-stream).
    2. ``thinking`` — searching for ``end_marker`` and emitting buffered
       text as ``thinking`` phase.
    3. ``answer`` — every subsequent chunk is emitted as ``answer`` phase.

    The lookback window per state ensures the markers are detected even when
    a chunk boundary falls inside one.
    """
    state = "pre-thinking"
    pending = ""
    start_lookback = max(0, len(start_marker) - 1)
    end_lookback = max(0, len(end_marker) - 1)

    for chunk in text_stream:
        if state == "answer":
            if chunk:
                yield StreamChunk(text=chunk, phase="answer")
            continue

        pending += chunk

        if state == "pre-thinking":
            if start_marker in pending:
                _before, after = pending.split(start_marker, 1)
                pending = after
                state = "thinking"
                # fall through to thinking-state handling below
            else:
                if len(pending) > start_lookback:
                    pending = pending[-start_lookback:] if start_lookback else ""
                continue

        if state == "thinking":
            if end_marker in pending:
                before, after = pending.split(end_marker, 1)
                if before:
                    yield StreamChunk(text=before, phase="thinking")
                state = "answer"
                if after:
                    yield StreamChunk(text=after, phase="answer")
                pending = ""
                continue

            if len(pending) > end_lookback:
                emit_now = pending[:-end_lookback] if end_lookback else pending
                pending = pending[-end_lookback:] if end_lookback else ""
                if emit_now:
                    yield StreamChunk(text=emit_now, phase="thinking")

    if pending:
        if state == "pre-thinking":
            # No structured thinking ever began — treat the whole leftover
            # buffer as a final answer so the controller doesn't drop it.
            yield StreamChunk(text=pending, phase="answer")
        else:
            yield StreamChunk(text=pending, phase=state)  # type: ignore[arg-type]


class StubGemmaBackend:
    """Deterministic backend for tests — replays a sequence of pre-recorded scenes.

    Each call to :meth:`stream_reasoning` consumes one scene from
    ``scripts``. ``fail_after`` simulates mid-stream errors by raising
    :class:`RuntimeError` after that many chunks have been yielded.
    """

    name: str = "stub.gemma"

    def __init__(
        self,
        scripts: list[list[StreamChunk]],
        *,
        fail_after: int | None = None,
    ) -> None:
        self._scripts: list[list[StreamChunk]] = list(scripts)
        self._fail_after: int | None = fail_after
        self.call_count: int = 0
        self.last_messages: list[dict[str, str]] = []
        self.last_seed: int | None = None
        self.last_max_new_tokens: int | None = None

    def stream_reasoning(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 4096,
        seed: int | None = None,
    ) -> Iterator[StreamChunk]:
        self.last_messages = list(messages)
        self.last_seed = seed
        self.last_max_new_tokens = max_new_tokens
        if self.call_count >= len(self._scripts):
            raise RuntimeError(
                f"StubGemmaBackend exhausted: called {self.call_count + 1} times, "
                f"only {len(self._scripts)} scenes scripted"
            )
        scene = self._scripts[self.call_count]
        self.call_count += 1
        for i, chunk in enumerate(scene):
            if self._fail_after is not None and i >= self._fail_after:
                raise RuntimeError(f"StubGemmaBackend simulated failure after {i} chunks")
            yield chunk


class Gemma4ThinkingBackend:
    """Production backend: ``google/gemma-4-E4B-it`` with thinking mode enabled.

    Both processor and model are loaded lazily on the first
    :meth:`stream_reasoning` call so importing this module never touches
    torch, transformers, or HuggingFace. Greedy decoding by default for
    determinism on a fixed seed; pass ``seed`` per call to seed both
    ``torch.manual_seed`` and ``transformers.set_seed`` before generation.

    The ``THINKING_START_MARKER`` / ``THINKING_END_MARKER`` class attributes
    are the channel tokens that bracket the thinking section in the model's
    output. Override on a subclass if a future Gemma revision changes them.
    """

    name: str = "gemma-4-E4B-it"
    THINKING_START_MARKER: str = THINKING_START_MARKER
    THINKING_END_MARKER: str = THINKING_END_MARKER

    def __init__(
        self,
        model_id: str = REASONER_E4B_MODEL_ID,
        device: str = "auto",
    ) -> None:
        self._model_id: str = model_id
        self._device: str = device
        self._processor: Any = None
        self._model: Any = None
        self._log = get_logger("qverify.controller.gemma4")

    def _load(self) -> None:
        if self._model is not None:
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoProcessor

        self._log.info("loading reasoner model %s", self._model_id)
        self._processor = AutoProcessor.from_pretrained(self._model_id)
        self._model = AutoModelForCausalLM.from_pretrained(
            self._model_id,
            torch_dtype=torch.bfloat16,
            device_map=self._device,
        )
        self._model.eval()

    def stream_reasoning(
        self,
        messages: list[dict[str, str]],
        *,
        max_new_tokens: int = 4096,
        seed: int | None = None,
    ) -> Iterator[StreamChunk]:
        self._load()

        import torch
        from transformers import TextIteratorStreamer, set_seed

        if seed is not None:
            torch.manual_seed(seed)
            set_seed(seed)

        prompt = self._processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        inputs = self._processor(text=prompt, return_tensors="pt").to(self._model.device)

        tokenizer = getattr(self._processor, "tokenizer", self._processor)
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=False,
        )

        gen_kwargs: dict[str, Any] = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "streamer": streamer,
            "do_sample": False,
        }

        thread = Thread(target=self._model.generate, kwargs=gen_kwargs)
        thread.start()

        try:
            yield from _stream_with_phase(
                iter(streamer),
                start_marker=self.THINKING_START_MARKER,
                end_marker=self.THINKING_END_MARKER,
            )
        finally:
            thread.join()
