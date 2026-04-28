"""Streaming LLM backend Protocol with real Gemma 4 and stub implementations."""

from __future__ import annotations

from collections.abc import Iterator
from threading import Thread
from typing import Any, Protocol, runtime_checkable

from qverify.controller.types import StreamChunk
from qverify.utils.logging import get_logger
from qverify.utils.models import REASONER_E4B_MODEL_ID

# Gemma 4 thinking-mode delimiter that closes the chain-of-thought channel.
# Adjust if a future Gemma revision changes the marker; the controller just
# needs *some* token sequence that reliably signals the start of the answer.
THINKING_END_MARKER = "<end_of_thinking>"


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


def _split_thinking_answer(raw: str, marker: str = THINKING_END_MARKER) -> tuple[str, str]:
    """Split a complete model response into (thinking, answer) halves.

    If ``marker`` does not appear, the entire string is treated as the answer
    and the thinking half is returned empty. Both halves are stripped of
    surrounding whitespace.
    """
    if marker in raw:
        thinking, answer = raw.split(marker, 1)
        return thinking.strip(), answer.strip()
    return "", raw.strip()


def _stream_with_phase(
    text_stream: Iterator[str],
    marker: str = THINKING_END_MARKER,
) -> Iterator[StreamChunk]:
    """Wrap a raw text-chunk stream into :class:`StreamChunk` events.

    Buffers a small lookback window so the marker is detected even when it
    straddles two raw chunks.
    """
    phase: str = "thinking"
    pending = ""
    lookback = max(0, len(marker) - 1)

    for chunk in text_stream:
        if phase == "answer":
            if chunk:
                yield StreamChunk(text=chunk, phase="answer")
            continue

        pending += chunk
        if marker in pending:
            before, after = pending.split(marker, 1)
            if before:
                yield StreamChunk(text=before, phase="thinking")
            phase = "answer"
            if after:
                yield StreamChunk(text=after, phase="answer")
            pending = ""
            continue

        if len(pending) > lookback:
            emit_now = pending[:-lookback] if lookback else pending
            pending = pending[-lookback:] if lookback else ""
            if emit_now:
                yield StreamChunk(text=emit_now, phase="thinking")

    if pending:
        yield StreamChunk(text=pending, phase=phase)  # type: ignore[arg-type]


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
    """

    name: str = "gemma-4-E4B-it"

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
            yield from _stream_with_phase(iter(streamer))
        finally:
            thread.join()
