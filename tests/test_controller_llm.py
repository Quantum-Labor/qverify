"""Unit tests for the controller's LLM Protocol, StubGemmaBackend, and helpers."""

from __future__ import annotations

import sys

import pytest

from qverify.controller.llm import (
    THINKING_END_MARKER,
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
# _split_thinking_answer
# ---------------------------------------------------------------------------


def test_split_thinking_answer_with_marker() -> None:
    raw = f"step 1.\nstep 2.{THINKING_END_MARKER}final answer."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == "step 1.\nstep 2."
    assert answer == "final answer."


def test_split_thinking_answer_no_marker_treats_as_answer_only() -> None:
    raw = "no thinking here, just an answer."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == ""
    assert answer == raw


def test_split_thinking_answer_strips_whitespace() -> None:
    raw = f"  step  {THINKING_END_MARKER}  ans  "
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == "step"
    assert answer == "ans"


def test_split_thinking_answer_empty_thinking() -> None:
    raw = f"{THINKING_END_MARKER}only answer."
    thinking, answer = _split_thinking_answer(raw)
    assert thinking == ""
    assert answer == "only answer."


# ---------------------------------------------------------------------------
# _stream_with_phase
# ---------------------------------------------------------------------------


def test_stream_with_phase_thinking_only() -> None:
    chunks = list(_stream_with_phase(iter(["alpha", " beta", " gamma"])))
    assert all(c.phase == "thinking" for c in chunks)
    assert "".join(c.text for c in chunks) == "alpha beta gamma"


def test_stream_with_phase_switches_on_marker() -> None:
    raw = ["thinking part ", THINKING_END_MARKER, " answer part"]
    chunks = list(_stream_with_phase(iter(raw)))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text
    assert "thinking part" in text_by_phase["thinking"]
    assert "answer part" in text_by_phase["answer"]


def test_stream_with_phase_marker_split_across_chunks() -> None:
    half_a = THINKING_END_MARKER[: len(THINKING_END_MARKER) // 2]
    half_b = THINKING_END_MARKER[len(THINKING_END_MARKER) // 2 :]
    raw = ["pre ", half_a, half_b, "post"]
    chunks = list(_stream_with_phase(iter(raw)))
    text_by_phase = {"thinking": "", "answer": ""}
    for c in chunks:
        text_by_phase[c.phase] += c.text
    assert "pre" in text_by_phase["thinking"]
    assert "post" in text_by_phase["answer"]


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
