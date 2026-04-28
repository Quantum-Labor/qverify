"""Sanity check: controller + translator + verifier on scripted thinking.

This test bypasses the real Gemma 4 reasoner by feeding the controller a
StubGemmaBackend with a hand-written reasoning trace. The Translator and
Verifier are real — Gemma E2B for translation, PennyLane for the Grover
search. Goal: prove the lower layers work end-to-end on clean syllogism
input, isolating any future failures to either step parsing or NL->CNF
translation.

Marked ``slow`` and ``gpu`` — excluded from CI. Run manually after
accepting the gated Gemma 4 license at
https://huggingface.co/google/gemma-4-E2B-it and running ``hf auth login``:

    .venv/bin/pytest tests/test_controller_pipeline_sanity.py -v -m "slow and gpu" -s

PASSED  → verifier + controller + translator all work on clean input;
          any real-Gemma E4B failures are isolated to step parsing or
          translator robustness on noisy NL input.
FAILED  → there's a deeper issue in the controller-translator-verifier
          chain that needs fixing before any translator refactor.
"""

from __future__ import annotations

import pytest

from qverify.controller import (
    StreamChunk,
    StubGemmaBackend,
    reason_with_verification,
)
from qverify.translator.llm import GemmaE2BBackend
from qverify.translator.translator import Translator
from qverify.verifier.backends import PennyLaneBackend

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture
def scripted_llm() -> StubGemmaBackend:
    """Hand-written thinking trace using simple, translator-friendly sentences.

    StubGemmaBackend takes a list of *scenes* (one per call to
    ``stream_reasoning``). For this sanity test the controller only ever
    makes one stream call (no retries are scripted), so we wrap our scene
    in an outer list of length one.
    """
    scene = [
        # Thinking phase — three clean syllogism steps separated by \n\n.
        StreamChunk(text="All cats have fur.", phase="thinking"),
        StreamChunk(text="\n\n", phase="thinking"),
        StreamChunk(text="Tom is a cat.", phase="thinking"),
        StreamChunk(text="\n\n", phase="thinking"),
        StreamChunk(text="Therefore Tom has fur.", phase="thinking"),
        # Answer phase.
        StreamChunk(text="Yes, Tom has fur.", phase="answer"),
    ]
    return StubGemmaBackend(scripts=[scene])


@pytest.fixture
def translator() -> Translator:
    """Real Gemma E2B translator (lazy-loaded on first translate call)."""
    return Translator(backend=GemmaE2BBackend())


def test_pipeline_sanity_with_scripted_thinking(
    scripted_llm: StubGemmaBackend,
    translator: Translator,
) -> None:
    """Controller processes 3 clean steps, verifier accepts each, final answer
    contains 'fur' or 'yes'."""
    result = reason_with_verification(
        problem=("Premises: All cats have fur. Tom is a cat. Question: does Tom have fur?"),
        llm=scripted_llm,
        translator=translator,
        verifier_backend=PennyLaneBackend(),
        max_retries_per_step=2,
        seed=42,
    )

    print("\n=== Sanity check result ===")
    print(f"final_answer:        {result.final_answer!r}")
    print(f"committed_steps:     {len(result.committed_steps)}")
    print(f"rejected_steps:      {len(result.rejected_steps)}")
    print(f"gave_up_steps:       {len(result.gave_up_steps)}")
    print(f"total_verifications: {result.total_verifications}")
    print(f"total_groundings:    {result.total_groundings}")
    print(f"wall_clock_seconds:  {result.wall_clock_seconds:.1f}")
    for i, step in enumerate(result.committed_steps):
        print(f"  committed[{i}]: {step!r}")
    for i, rejected in enumerate(result.rejected_steps):
        print(f"  rejected[{i}]:  {rejected.original_step!r}")
        print(f"                 -> counter_model={rejected.counter_model}")
    for i, gave_up in enumerate(result.gave_up_steps):
        print(f"  gave_up[{i}]:   {gave_up!r}")

    lowered = result.final_answer.lower()
    assert ("yes" in lowered) or ("fur" in lowered), (
        f"unexpected final_answer: {result.final_answer!r}"
    )
    # The point of the sanity check: at least one step was actually verified.
    assert result.total_verifications >= 1, (
        "no verifications happened — translator or controller is broken"
    )
    # Phase 4.5: at least one grounding pass should have occurred too.
    assert result.total_groundings >= 1, (
        "no groundings happened — controller's grounding integration is broken"
    )
