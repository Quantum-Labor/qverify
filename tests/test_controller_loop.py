"""End-to-end tests for the controller's verify-and-rewrite loop."""

from __future__ import annotations

import subprocess
import sys

import pytest

from qverify.controller import (
    ControllerEvent,
    FinalAnswer,
    ReasoningStepCommitted,
    ReasoningStepGaveUp,
    ReasoningStepRejected,
    ReasoningStepStarted,
    ReasoningStepVerified,
    StubGemmaBackend,
    reason_with_verification,
)
from qverify.controller.types import StreamChunk
from qverify.verifier.types import CounterModel, VerificationResult

# ---------------------------------------------------------------------------
# StubVerifier — accepts a queue of canned VerificationResults
# ---------------------------------------------------------------------------


def _make_result(
    *,
    contradiction: bool,
    counter_model_assignment: dict[str, bool] | None = None,
) -> VerificationResult:
    cm = (
        CounterModel(assignment=counter_model_assignment)
        if (contradiction and counter_model_assignment is not None)
        else CounterModel(assignment={"P": True})
        if contradiction
        else None
    )
    return VerificationResult(
        contradiction_found=contradiction,
        counter_model=cm,
        n_variables=1,
        n_clauses=1,
        n_grover_iterations=1,
        backend_name="stub",
        shots=4,
    )


class StubVerifier:
    """Records every (step, premises) call and returns canned results in order."""

    def __init__(self, results: list[VerificationResult]) -> None:
        self._results = list(results)
        self.calls: list[tuple[str, list[str]]] = []

    def __call__(self, step: str, premises: list[str]) -> VerificationResult:
        self.calls.append((step, list(premises)))
        if not self._results:
            raise RuntimeError(f"StubVerifier exhausted after {len(self.calls)} calls")
        return self._results.pop(0)


# ---------------------------------------------------------------------------
# Helpers for building stub LLM scripts
# ---------------------------------------------------------------------------


def _thinking_scene(*paragraphs: str, answer: str | None = None) -> list[StreamChunk]:
    """Build a single LLM scene from a list of thinking paragraphs and an answer."""
    chunks: list[StreamChunk] = []
    body = "\n\n".join(paragraphs)
    if body:
        chunks.append(StreamChunk(text=body + "\n\n", phase="thinking"))
    if answer is not None:
        chunks.append(StreamChunk(text=answer, phase="answer"))
    return chunks


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


def test_happy_path_all_steps_verify() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene(
                "Step 1: All cats have fur.",
                "Step 2: Tom is a cat.",
                "Step 3: Therefore Tom has fur.",
                answer="Yes, Tom has fur.",
            )
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=False),
            _make_result(contradiction=False),
            _make_result(contradiction=False),
        ]
    )

    result = reason_with_verification(
        problem="Does Tom have fur?",
        llm=llm,
        verify_fn=verifier,
        max_retries_per_step=3,
    )

    assert result.final_answer == "Yes, Tom has fur."
    assert len(result.committed_steps) == 3
    assert len(result.rejected_steps) == 0
    assert len(result.gave_up_steps) == 0
    assert result.total_verifications == 3
    assert result.total_contradictions_found == 0


def test_committed_steps_become_premises_for_subsequent_verifies() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("Step A.", "Step B.", answer="done")])
    verifier = StubVerifier(
        [
            _make_result(contradiction=False),
            _make_result(contradiction=False),
        ]
    )
    reason_with_verification(problem="?", llm=llm, verify_fn=verifier)

    # Second verify call should see Step A in its premises.
    assert verifier.calls[0][1] == []
    assert verifier.calls[1][1] == ["Step A."]


# ---------------------------------------------------------------------------
# Retry behaviour
# ---------------------------------------------------------------------------


def test_one_contradiction_fixed_on_retry_one() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("Step 1: All birds fly.", answer="ans"),
            _thinking_scene("Step 1: Most birds can fly.", answer=""),
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=True, counter_model_assignment={"Bird": True}),
            _make_result(contradiction=False),
        ]
    )

    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=3
    )

    assert result.final_answer == "ans"
    assert result.committed_steps == ("Step 1: Most birds can fly.",)
    assert len(result.rejected_steps) == 1
    rec = result.rejected_steps[0]
    assert rec.original_step == "Step 1: All birds fly."
    assert rec.fixed_at_attempt == 1
    assert rec.final_accepted_rewrite == "Step 1: Most birds can fly."
    assert result.total_contradictions_found == 1
    assert result.total_verifications == 2


def test_contradiction_fixed_on_last_allowed_retry() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("S1: bad.", answer="ans"),
            _thinking_scene("S1: still bad."),
            _thinking_scene("S1: still bad too."),
            _thinking_scene("S1: finally ok."),
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=True),
            _make_result(contradiction=True),
            _make_result(contradiction=True),
            _make_result(contradiction=False),
        ]
    )
    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=3
    )
    assert result.committed_steps == ("S1: finally ok.",)
    assert result.gave_up_steps == ()
    assert result.rejected_steps[0].fixed_at_attempt == 3
    assert result.total_verifications == 4


def test_all_retries_fail_records_gave_up() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("S1: bad.", answer="ans"),
            _thinking_scene("S1: rewrite 1."),
            _thinking_scene("S1: rewrite 2."),
            _thinking_scene("S1: rewrite 3."),
        ]
    )
    verifier = StubVerifier([_make_result(contradiction=True) for _ in range(4)])

    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=3
    )

    assert result.committed_steps == ()
    assert result.gave_up_steps == ("S1: bad.",)
    assert result.rejected_steps[0].fixed_at_attempt is None
    assert result.rejected_steps[0].final_accepted_rewrite is None
    assert result.total_contradictions_found == 4


def test_max_retries_zero_means_immediate_gave_up() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("Bad step.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=True)])

    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=0
    )

    assert result.gave_up_steps == ("Bad step.",)
    assert result.total_verifications == 1
    assert result.total_contradictions_found == 1


def test_multiple_contradictions_in_sequence_do_not_cascade() -> None:
    """Two adjacent steps both rejected then fixed — controller stays alive."""
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("Step A bad.", "Step B bad.", answer="ans"),
            _thinking_scene("Step A good."),
            _thinking_scene("Step B good."),
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=True),
            _make_result(contradiction=False),
            _make_result(contradiction=True),
            _make_result(contradiction=False),
        ]
    )
    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=3
    )
    assert result.committed_steps == ("Step A good.", "Step B good.")
    assert len(result.rejected_steps) == 2


# ---------------------------------------------------------------------------
# Output and metadata
# ---------------------------------------------------------------------------


def test_empty_thinking_yields_immediate_answer() -> None:
    llm = StubGemmaBackend(scripts=[[StreamChunk(text="just the answer", phase="answer")]])
    verifier = StubVerifier([])
    result = reason_with_verification(problem="?", llm=llm, verify_fn=verifier)
    assert result.final_answer == "just the answer"
    assert result.committed_steps == ()
    assert result.total_verifications == 0


def test_total_verifications_matches_verifier_calls() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("a.", "b.", "c.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=False) for _ in range(3)])
    result = reason_with_verification(problem="?", llm=llm, verify_fn=verifier)
    assert result.total_verifications == len(verifier.calls)


def test_emit_callback_receives_events_in_correct_order() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("step.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=False)])
    events: list[ControllerEvent] = []
    reason_with_verification(problem="?", llm=llm, verify_fn=verifier, emit=events.append)
    types = [type(e) for e in events]
    assert types == [
        ReasoningStepStarted,
        ReasoningStepVerified,
        ReasoningStepCommitted,
        FinalAnswer,
    ]


def test_emit_callback_records_rejection_and_committed_for_retried_step() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("bad.", answer="ans"),
            _thinking_scene("good."),
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=True),
            _make_result(contradiction=False),
        ]
    )
    events: list[ControllerEvent] = []
    reason_with_verification(problem="?", llm=llm, verify_fn=verifier, emit=events.append)
    types = [type(e) for e in events]
    assert ReasoningStepRejected in types
    assert ReasoningStepCommitted in types
    assert types[-1] is FinalAnswer


def test_emit_callback_records_gave_up_for_exhausted_retries() -> None:
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("bad.", answer="ans"),
            _thinking_scene("rewrite 1."),
        ]
    )
    verifier = StubVerifier([_make_result(contradiction=True), _make_result(contradiction=True)])
    events: list[ControllerEvent] = []
    reason_with_verification(
        problem="?",
        llm=llm,
        verify_fn=verifier,
        emit=events.append,
        max_retries_per_step=1,
    )
    assert any(isinstance(e, ReasoningStepGaveUp) for e in events)


def test_wall_clock_seconds_is_positive_and_small_for_stub_runs() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("a.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=False)])
    result = reason_with_verification(problem="?", llm=llm, verify_fn=verifier)
    assert 0.0 < result.wall_clock_seconds < 5.0


def test_seed_is_plumbed_to_llm_backend() -> None:
    llm = StubGemmaBackend(scripts=[_thinking_scene("a.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=False)])
    reason_with_verification(problem="?", llm=llm, verify_fn=verifier, seed=4242)
    assert llm.last_seed == 4242


def test_max_retries_negative_raises_value_error() -> None:
    llm = StubGemmaBackend(scripts=[])
    verifier = StubVerifier([])
    with pytest.raises(ValueError, match="max_retries_per_step"):
        reason_with_verification(problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=-1)


def test_premises_only_contain_committed_steps_after_run() -> None:
    """gave_up steps must NOT appear in committed_steps and must NOT be premises."""
    llm = StubGemmaBackend(
        scripts=[
            _thinking_scene("good 1.", "bad.", "good 2.", answer="ans"),
            _thinking_scene("rewrite of bad."),  # also fails
        ]
    )
    verifier = StubVerifier(
        [
            _make_result(contradiction=False),  # good 1 -> committed
            _make_result(contradiction=True),  # bad -> rejected
            _make_result(contradiction=True),  # rewrite still fails -> gave_up
            _make_result(contradiction=False),  # good 2 -> committed
        ]
    )
    result = reason_with_verification(
        problem="?", llm=llm, verify_fn=verifier, max_retries_per_step=1
    )
    assert result.committed_steps == ("good 1.", "good 2.")
    assert result.gave_up_steps == ("bad.",)
    # Premises seen on the LAST verify call should be just the prior committed.
    last_call_premises = verifier.calls[-1][1]
    assert last_call_premises == ["good 1."]


# ---------------------------------------------------------------------------
# Lazy-load contract — strict subprocess check
# ---------------------------------------------------------------------------


def test_importing_qverify_controller_does_not_load_transformers() -> None:
    """Spawn a fresh interpreter to assert lazy-load — the parent's sys.modules
    can be polluted by prior tests, but a child process is clean."""
    code = (
        "import sys; "
        "import qverify.controller as c; "
        "assert 'transformers' not in sys.modules, 'transformers leaked'; "
        "assert 'torch' not in sys.modules, 'torch leaked'; "
        "print('ok')"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert proc.returncode == 0, proc.stderr
    assert "ok" in proc.stdout
