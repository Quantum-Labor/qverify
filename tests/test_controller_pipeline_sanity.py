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
        # Phase 6.7: thinking is captured but not split for verification;
        # the verifier operates on the answer phase. Keep a brief thinking
        # chunk to exercise both phases.
        StreamChunk(text="Let me reason step by step about this.", phase="thinking"),
        StreamChunk(text="\n", phase="thinking"),
        # Answer phase — three numbered declarative steps + final answer.
        StreamChunk(text="1. All cats have fur.\n", phase="answer"),
        StreamChunk(text="2. Tom is a cat.\n", phase="answer"),
        StreamChunk(text="3. Therefore Tom has fur.\n", phase="answer"),
        StreamChunk(text="\nYes, Tom has fur.", phase="answer"),
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
    print(f"final_answer:                   {result.final_answer!r}")
    print(f"committed_steps:                {len(result.committed_steps)}")
    print(f"rejected_steps:                 {len(result.rejected_steps)}")
    print(f"gave_up_steps:                  {len(result.gave_up_steps)}")
    print(f"total_verifications:            {result.total_verifications}")
    print(f"total_groundings:               {result.total_groundings}")
    print(f"total_answer_steps_extracted:   {result.total_answer_steps_extracted}")
    print(f"initial_universe_size:          {result.initial_universe_size}")
    print(f"wall_clock_seconds:             {result.wall_clock_seconds:.1f}")
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
    # Phase 6.5: the pre-pass on the problem statement should have
    # extracted at least one entity (Tom) from "Premises: ... Tom is a
    # cat. Question: ...". Without this, step 1's free-variable
    # universal couldn't be grounded.
    assert result.initial_universe_size >= 1, (
        "expected at least one entity (Tom) extracted from the problem"
    )
    # Phase 6.7: the controller now extracts numbered steps from the
    # answer phase; the scripted scene emits exactly three.
    assert result.total_answer_steps_extracted >= 1, (
        "no answer-phase steps extracted — the new verification path is broken"
    )
    # Phase 6.8: under consistency-mode verification, all three clean
    # syllogism steps should be accepted: each is consistent with what
    # came before. Anything dropped here means consistency-mode
    # verification is mis-rejecting.
    assert len(result.committed_steps) == 3, (
        f"expected all 3 syllogism steps to commit under consistency mode; "
        f"got {len(result.committed_steps)}: {result.committed_steps}"
    )
    assert len(result.gave_up_steps) == 0, f"expected no gave-up steps; got {result.gave_up_steps}"
    # Consistency-mode rejections carry counter_model=None (UNSAT — no
    # satisfying assignment to display). Vacuously true when the loop
    # accepted everything, but documents the new shape of the data.
    for r in result.rejected_steps:
        assert r.counter_model is None, (
            f"unexpected counter_model on consistency-mode rejection: {r}"
        )
