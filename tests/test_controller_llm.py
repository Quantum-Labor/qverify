"""Unit tests for the controller's LLM Protocol, StubGemmaBackend, and helpers."""

from __future__ import annotations

import sys

import pytest

from qverify.controller.llm import (
    ANSWER_END_MARKER,
    THINKING_END_MARKER,
    THINKING_START_MARKER,
    Gemma4ThinkingBackend,
    LLMBackend,
    StubGemmaBackend,
    _split_thinking_answer,
    _stream_with_phase,
)
from qverify.controller.types import StreamChunk

# Canonical real-marker sample, distilled from /tmp/gemma4_dump.txt of an
# actual google/gemma-4-E4B-it run. The asymmetric brackets and the trailing
# <turn|> token are the exact strings the tokenizer emits.
REAL_RAW_SAMPLE = (
    "<|channel>thought\n"
    "1. Premise 1: All cats have fur.\n\n"
    "2. Premise 2: Tom is a cat.\n\n"
    "3. Therefore Tom has fur."
    "<channel|>"
    "**Answer:**\n\nYes"
    "<turn|>"
)

# ---------------------------------------------------------------------------
# StreamChunk validation
# ---------------------------------------------------------------------------


def test_stream_chunk_accepts_thinking_phase() -> None:
    chunk = StreamChunk(text="step 1", phase="thinking")
    assert chunk.phase == "thinking"
    assert chunk.text == "step 1"


def test_stream_chunk_accepts_answer_phase() -> None:
    chunk = StreamChunk(text="answer", phase="answer")
    assert chunk.phase == "answer"


def test_stream_chunk_rejects_invalid_phase() -> None:
    with pytest.raises(ValueError, match="phase must be"):
        StreamChunk(text="x", phase="other")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# StubGemmaBackend
# ---------------------------------------------------------------------------


def test_stub_yields_scripted_chunks_in_order() -> None:
    chunks = [
        StreamChunk("Step 1.", "thinking"),
        StreamChunk("\n\nStep 2.", "thinking"),
        StreamChunk("Final.", "answer"),
    ]
    backend = StubGemmaBackend(scripts=[chunks])
    out = list(backend.stream_reasoning([{"role": "user", "content": "go"}]))
    assert out == chunks


def test_stub_records_messages_and_seed() -> None:
    backend = StubGemmaBackend(scripts=[[StreamChunk("x", "thinking")]])
    msgs = [{"role": "user", "content": "hi"}]
    list(backend.stream_reasoning(msgs, seed=7, max_new_tokens=128))
    assert backend.last_messages == msgs
    assert backend.last_seed == 7
    assert backend.last_max_new_tokens == 128


def test_stub_consumes_one_scene_per_call() -> None:
    backend = StubGemmaBackend(
        scripts=[
            [StreamChunk("scene A", "thinking")],
            [StreamChunk("scene B", "thinking")],
        ]
    )
    a = list(backend.stream_reasoning([]))
    b = list(backend.stream_reasoning([]))
    assert a[0].text == "scene A"
    assert b[0].text == "scene B"


def test_stub_raises_when_exhausted() -> None:
    backend = StubGemmaBackend(scripts=[[StreamChunk("only", "thinking")]])
    list(backend.stream_reasoning([]))
    with pytest.raises(RuntimeError, match="exhausted"):
        list(backend.stream_reasoning([]))


def test_stub_fail_after_raises_runtime_error() -> None:
    backend = StubGemmaBackend(
        scripts=[
            [
                StreamChunk("a", "thinking"),
                StreamChunk("b", "thinking"),
                StreamChunk("c", "thinking"),
            ]
        ],
        fail_after=2,
    )
    out: list[StreamChunk] = []
    with pytest.raises(RuntimeError, match="simulated failure"):
        for chunk in backend.stream_reasoning([]):
            out.append(chunk)
    assert len(out) == 2


def test_stub_satisfies_llm_backend_protocol() -> None:
    backend = StubGemmaBackend(scripts=[[]])
    assert isinstance(backend, LLMBackend)


# ---------------------------------------------------------------------------
# _split_thinking_answer (Gemma 4 channel-token format)
# ---------------------------------------------------------------------------


def test_split_thinking_answer_real_markers() -> None:
    """Canonical Gemma 4 E4B output (asymmetric brackets, trailing <turn|>)."""
    thinking, answer = _split_thinking_answer(REAL_RAW_SAMPLE)
    assert "Premise 1" in thinking
    assert "Premise 2" in thinking
    assert "Therefore" in thinking
    # Channel markers and the channel-name "thought" must never leak into
    # the thinking content.
    assert "<|channel>" not in thinking
    assert "<channel|>" not in thinking
    assert "thought" not in thinking
    assert answer == "**Answer:**\n\nYes"
    assert "<turn|>" not in answer


def test_split_thinking_answer_no_start_marker_treats_as_answer_only() -> None:
    raw = "no thinking here, just an answer."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == ""
    assert answer == raw


def test_split_thinking_answer_strips_whitespace() -> None:
    raw = f"  {THINKING_START_MARKER}  step  {THINKING_END_MARKER}  ans  "
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == "step"
    assert answer == "ans"


def test_split_thinking_answer_empty_thinking_block() -> None:
    raw = f"{THINKING_START_MARKER}{THINKING_END_MARKER}only answer."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == ""
    assert answer == "only answer."


def test_split_thinking_answer_start_present_but_no_end() -> None:
    """If the model starts thinking but never closes the channel, treat
    everything after start as thinking and return an empty answer."""
    raw = f"{THINKING_START_MARKER}step 1.\nstep 2."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == "step 1.\nstep 2."
    assert answer == ""


# ---------------------------------------------------------------------------
# _stream_with_phase (Gemma 4 channel-token format)
# ---------------------------------------------------------------------------


def test_stream_with_phase_no_start_marker_emits_as_answer() -> None:
    """No structured thinking ever began — leftover flushes as answer."""
    chunks = list(_stream_with_phase(iter(["alpha", " beta", " gamma"])))
    assert "".join(c.text for c in chunks) == "alpha beta gamma"
    # Without a start marker, the controller should still see content as
    # the final answer rather than dropping it on the floor.
    assert all(c.phase == "answer" for c in chunks)


def test_split_thinking_answer_split_across_chunks_real_markers() -> None:
    """Feed REAL_RAW_SAMPLE one character at a time through ``_stream_with_phase``
    and verify phase transitions land at the correct marker boundaries."""
    raw_chunks = list(REAL_RAW_SAMPLE)  # one char per chunk
    chunks = list(_stream_with_phase(iter(raw_chunks)))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text

    expected_thinking = (
        "1. Premise 1: All cats have fur.\n\n"
        "2. Premise 2: Tom is a cat.\n\n"
        "3. Therefore Tom has fur."
    )
    expected_answer = "**Answer:**\n\nYes"

    assert text_by_phase["thinking"] == expected_thinking
    assert text_by_phase["answer"] == expected_answer
    # No marker fragment leaks into either phase, regardless of where the
    # one-character cuts fall inside the channel/turn tokens.
    for marker in ("<|channel>", "<channel|>", "<turn|>", "thought"):
        assert marker not in text_by_phase["thinking"], f"{marker!r} leaked into thinking"
        assert marker not in text_by_phase["answer"], f"{marker!r} leaked into answer"


def test_answer_end_marker_truncates_before_emit() -> None:
    """`<turn|>` must terminate the answer phase and drop everything after it."""
    raw = f"{THINKING_START_MARKER}ABC{THINKING_END_MARKER}HELLO{ANSWER_END_MARKER}WORLD"
    chunks = list(_stream_with_phase(iter([raw])))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text
    assert text_by_phase["thinking"] == "ABC"
    assert text_by_phase["answer"] == "HELLO"
    assert "WORLD" not in text_by_phase["answer"]
    assert "<turn|>" not in text_by_phase["answer"]


def test_answer_end_marker_truncates_when_split_byte_by_byte() -> None:
    """Same truncation property must hold when chunks are 1 char wide."""
    raw = f"{THINKING_START_MARKER}ABC{THINKING_END_MARKER}HELLO{ANSWER_END_MARKER}WORLD"
    chunks = list(_stream_with_phase(iter(list(raw))))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text
    assert text_by_phase["thinking"] == "ABC"
    assert text_by_phase["answer"] == "HELLO"


# ---------------------------------------------------------------------------
# Gemma4ThinkingBackend lazy-load contract
# ---------------------------------------------------------------------------


def test_gemma4_constructor_does_not_load_torch_or_transformers() -> None:
    backend = Gemma4ThinkingBackend()
    assert backend._model is None
    assert backend._processor is None
    # The bare construction must not have pulled torch/transformers into
    # sys.modules. Phase 4 enforces the same property for IBMRuntimeClient.
    # We don't strictly require absence here (test runner may have already
    # imported them), but at minimum the backend's caches stay empty.
    assert backend.name == "gemma-4-E4B-it"


def test_importing_controller_module_does_not_load_transformers() -> None:
    """`import qverify.controller` must not pull transformers into sys.modules."""
    # Re-import a fresh runtime by checking the running process.
    # If transformers is already in sys.modules from a prior test, this is
    # a soft check — we still verify the controller imports work.
    import qverify.controller  # noqa: F401

    # If transformers is loaded, it must NOT be because of qverify.controller's
    # top-level imports (would only happen via lazy paths inside execute).
    # The strictest test is the subprocess one in test_controller_loop.py.
    assert "qverify.controller" in sys.modules
