"""Real-Gemma controller smoke test.

Marked ``slow`` and ``gpu`` — excluded from CI. Run manually on a CUDA host
with at least 12 GB VRAM after accepting the gated Gemma 4 license at
https://huggingface.co/google/gemma-4-E4B-it and running ``hf auth login``:

    .venv/bin/pytest tests/test_controller_e4b_smoke.py -v -m "slow and gpu" -s

The first run downloads ~5–8 GB of weights into ``~/.cache/huggingface/``;
subsequent runs are warm and complete in under a minute.
"""

from __future__ import annotations

import pytest

from qverify.controller import (
    Gemma4ThinkingBackend,
    reason_with_verification,
)
from qverify.translator import Translator
from qverify.translator.llm import GemmaE2BBackend
from qverify.verifier.backends import PennyLaneBackend

pytestmark = [pytest.mark.slow, pytest.mark.gpu]


@pytest.fixture(scope="module")
def llm_backend() -> Gemma4ThinkingBackend:
    return Gemma4ThinkingBackend()


@pytest.fixture(scope="module")
def translator() -> Translator:
    return Translator(backend=GemmaE2BBackend())


def test_simple_syllogism_runs_end_to_end(
    llm_backend: Gemma4ThinkingBackend,
    translator: Translator,
) -> None:
    problem = (
        "Premises: All cats have fur. Tom is a cat.\n"
        "Question: does Tom have fur? Reason step by step then give a yes/no answer."
    )
    result = reason_with_verification(
        problem=problem,
        llm=llm_backend,
        translator=translator,
        verifier_backend=PennyLaneBackend(),
        max_retries_per_step=2,
        seed=42,
        max_new_tokens=1024,
    )

    lowered = result.final_answer.lower()
    assert ("yes" in lowered) or ("fur" in lowered), (
        f"unexpected final answer: {result.final_answer!r}"
    )
    assert result.total_verifications >= 1
    assert result.wall_clock_seconds > 0.0

    print("\n--- controller smoke test summary ---")
    print(f"final_answer       : {result.final_answer}")
    print(f"committed_steps    : {len(result.committed_steps)}")
    print(f"rejected_steps     : {len(result.rejected_steps)}")
    print(f"gave_up_steps      : {len(result.gave_up_steps)}")
    print(f"total_verifs       : {result.total_verifications}")
    print(f"answer_steps       : {result.total_answer_steps_extracted}")
    print(f"contradictions     : {result.total_contradictions_found}")
    print(f"initial_universe   : {result.initial_universe_size}")
    print(f"wall_clock         : {result.wall_clock_seconds:.2f}s")
