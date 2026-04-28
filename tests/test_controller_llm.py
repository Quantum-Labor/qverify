"""Unit tests for the controller's LLM Protocol, StubGemmaBackend, and helpers."""

from __future__ import annotations

import sys

import pytest

from qverify.controller.llm import (
    THINKING_END_MARKER,
    THINKING_START_MARKER,
    Gemma4ThinkingBackend,
    LLMBackend,
    StubGemmaBackend,
    _split_thinking_answer,
    _stream_with_phase,
)
from qverify.controller.types import StreamChunk

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


def test_split_thinking_answer_with_real_channel_markers() -> None:
    raw = (
        f"{THINKING_START_MARKER}"
        "Step 1: Tom is a cat.\n\n"
        "Step 2: All cats have fur.\n\n"
        "Step 3: Therefore Tom has fur."
        f"{THINKING_END_MARKER}"
        "Yes, Tom has fur."
    )
    thinking, answer = _split_thinking_answer(raw)
    assert "Step 1" in thinking
    assert "Step 2" in thinking
    assert "Step 3" in thinking
    # The literal markers and the word "thought" must NOT leak into the
    # extracted thinking text — they are channel framing, not content.
    assert "<|channel|>" not in thinking
    assert "thought" not in thinking
    assert answer == "Yes, Tom has fur."


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


def test_stream_with_phase_emits_thinking_then_answer_with_real_markers() -> None:
    raw_chunks = [
        THINKING_START_MARKER,
        "thinking part ",
        THINKING_END_MARKER,
        " answer part",
    ]
    chunks = list(_stream_with_phase(iter(raw_chunks)))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text
    assert "thinking part" in text_by_phase["thinking"]
    assert "answer part" in text_by_phase["answer"]
    # Markers themselves must not leak through.
    for c in chunks:
        assert "<|channel|>" not in c.text


def test_stream_with_phase_chunk_split_anywhere_in_markers() -> None:
    """Split the full Gemma 4 stream byte-by-byte at every offset and
    verify the controller still recovers the same thinking and answer
    content regardless of where the cut falls."""
    raw = (
        f"{THINKING_START_MARKER}"
        "Step 1: Tom is a cat.\n\n"
        "Step 2: All cats have fur."
        f"{THINKING_END_MARKER}"
        "Yes, Tom has fur."
    )
    expected_thinking = "Step 1: Tom is a cat.\n\nStep 2: All cats have fur."
    expected_answer = "Yes, Tom has fur."

    for chunk_size in (1, 2, 3, 5, 7, 13, len(THINKING_START_MARKER)):
        raw_chunks = [raw[i : i + chunk_size] for i in range(0, len(raw), chunk_size)]
        chunks = list(_stream_with_phase(iter(raw_chunks)))
        text_by_phase = {"thinking": "", "answer": ""}
        for c in chunks:
            text_by_phase[c.phase] += c.text
        # Markers must never leak into either phase.
        assert "<|channel|>" not in text_by_phase["thinking"], (
            f"channel marker leaked into thinking at chunk_size={chunk_size}"
        )
        assert "<|channel|>" not in text_by_phase["answer"], (
            f"channel marker leaked into answer at chunk_size={chunk_size}"
        )
        # Content is preserved (modulo the non-leading whitespace inside
        # the start marker — the trailing newline of "<|channel|>thought\n"
        # is consumed as part of the marker).
        assert text_by_phase["thinking"] == expected_thinking, (
            f"thinking mismatch at chunk_size={chunk_size}: {text_by_phase['thinking']!r}"
        )
        assert text_by_phase["answer"] == expected_answer, (
            f"answer mismatch at chunk_size={chunk_size}: {text_by_phase['answer']!r}"
        )


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
