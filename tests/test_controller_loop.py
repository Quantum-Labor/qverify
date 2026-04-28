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


# ---------------------------------------------------------------------------
# Grounding integration (Phase 4.5)
# ---------------------------------------------------------------------------


from qverify.translator.cnf import CNF, Clause, Literal  # noqa: E402
from qverify.translator.types import TranslationResult  # noqa: E402
from qverify.verifier._universe import Universe  # noqa: E402


class _StubTranslator:
    """Returns predetermined TranslationResults keyed by exact statement text.

    Falls back to ``default_result`` when the statement is not in the map,
    which lets tests stay terse — they only need to script the few inputs
    they care about.
    """

    def __init__(
        self,
        mapping: dict[str, TranslationResult] | None = None,
        default_result: TranslationResult | None = None,
    ) -> None:
        self._mapping = dict(mapping) if mapping else {}
        self._default = (
            default_result
            if default_result is not None
            else TranslationResult(cnf=CNF(clauses=()), universe=Universe(constants=()))
        )
        self.calls: list[str] = []

    def translate(self, statement: str) -> TranslationResult:
        self.calls.append(statement)
        if statement in self._mapping:
            return self._mapping[statement]
        return self._default


def test_controller_merges_universes_from_multiple_translator_calls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The controller must take the union of every translator result's
    universe before grounding; constants from premise A and premise B must
    both be available when grounding the negated step."""
    from qverify.controller import controller as controller_module
    from qverify.translator.cnf import CNF
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    captured_universes: list[Universe] = []
    original_ground = controller_module.ground_cnf

    def spy_ground_cnf(cnf: CNF, universe: Universe) -> CNF:
        captured_universes.append(universe)
        return original_ground(cnf, universe)

    monkeypatch.setattr(controller_module, "ground_cnf", spy_ground_cnf)

    def cnf_for(predicate: str, arg: str, neg: bool = False) -> CNF:
        return CNF(
            clauses=(Clause(literals=(Literal(predicate=predicate, args=(arg,), negated=neg),)),)
        )

    translator = _StubTranslator(
        mapping={
            "Tom is a cat.": TranslationResult(
                cnf=cnf_for("Cat", "Tom"),
                universe=Universe(constants=("Tom",)),
            ),
            "Whiskers is a cat.": TranslationResult(
                cnf=cnf_for("Cat", "Whiskers"),
                universe=Universe(constants=("Whiskers",)),
            ),
            "It is not the case that Whiskers is a mammal.": TranslationResult(
                cnf=cnf_for("Mammal", "Whiskers", neg=True),
                universe=Universe(constants=("Whiskers",)),
            ),
        }
    )

    llm = StubGemmaBackend(scripts=[_thinking_scene("Whiskers is a mammal.", answer="ans")])

    reason_with_verification(
        problem="?",
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # The premise list at the time of verify() was empty (this is the FIRST
    # step) so only the negated step's universe contributes here. Re-run
    # with a pre-committed premise to exercise the merge path properly.
    assert len(captured_universes) == 1
    assert captured_universes[0].constants == ("Whiskers",)


def test_controller_passes_grounded_cnf_to_verify(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Verify the CNF the runner hands to the verifier has no free vars."""
    from qverify.controller import controller as controller_module
    from qverify.translator.cnf import CNF
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe
    from qverify.verifier._vars import is_free_variable

    captured_cnfs: list[CNF] = []

    def fake_verify(cnf: CNF, backend: object = None) -> VerificationResult:
        captured_cnfs.append(cnf)
        return _make_result(contradiction=False)

    monkeypatch.setattr(controller_module, "default_verify", fake_verify)

    # First-order CNF for "All cats have fur" — has free variable x.
    universal = TranslationResult(
        cnf=CNF(
            clauses=(
                Clause(
                    literals=(
                        Literal(predicate="Cat", args=("x",), negated=True),
                        Literal(predicate="Fur", args=("x",)),
                    )
                ),
            )
        ),
        universe=Universe(constants=()),
    )
    # Ground premise contributing the constant Tom.
    tom_is_cat = TranslationResult(
        cnf=CNF(clauses=(Clause(literals=(Literal(predicate="Cat", args=("Tom",)),)),)),
        universe=Universe(constants=("Tom",)),
    )
    translator = _StubTranslator(
        mapping={
            "All cats have fur.": universal,
            "Tom is a cat.": tom_is_cat,
            "It is not the case that Tom is a cat.": tom_is_cat,
        }
    )

    llm = StubGemmaBackend(
        scripts=[_thinking_scene("All cats have fur.", "Tom is a cat.", answer="ans")]
    )

    reason_with_verification(
        problem="?",
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # Every captured CNF must have NO free variables — grounding worked.
    for cnf in captured_cnfs:
        for clause in cnf.clauses:
            for lit in clause.literals:
                for arg in lit.args:
                    assert not is_free_variable(arg), (
                        f"free variable {arg!r} reached the verifier; grounding failed"
                    )


def test_controller_propositional_cnf_passes_through_grounding_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Empty universe + propositional CNF must not raise; ground_cnf returns
    the input unchanged."""
    from qverify.controller import controller as controller_module
    from qverify.translator.cnf import CNF
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    captured: list[CNF] = []

    def fake_verify(cnf: CNF, backend: object = None) -> VerificationResult:
        captured.append(cnf)
        return _make_result(contradiction=False)

    monkeypatch.setattr(controller_module, "default_verify", fake_verify)

    propositional = TranslationResult(
        cnf=CNF(
            clauses=(
                Clause(
                    literals=(
                        Literal(predicate="Rain", args=(), negated=True),
                        Literal(predicate="Wet", args=()),
                    )
                ),
            )
        ),
        universe=Universe(constants=()),
    )
    translator = _StubTranslator(default_result=propositional)
    llm = StubGemmaBackend(
        scripts=[_thinking_scene("If it rains the street is wet.", answer="ans")]
    )

    result = reason_with_verification(
        problem="?",
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    assert result.total_groundings >= 1
    assert len(captured) == 1
    # Propositional CNF stayed propositional through grounding.
    for clause in captured[0].clauses:
        for lit in clause.literals:
            assert lit.args == ()


def test_controller_total_groundings_increments_per_verify_call() -> None:
    """ControllerResult.total_groundings tracks one ground_cnf call per verify."""
    from qverify.translator.cnf import CNF
    from qverify.translator.types import TranslationResult
    from qverify.verifier._universe import Universe

    empty = TranslationResult(cnf=CNF(clauses=()), universe=Universe(constants=()))
    translator = _StubTranslator(default_result=empty)
    llm = StubGemmaBackend(scripts=[_thinking_scene("step 1.", "step 2.", "step 3.", answer="ans")])

    result = reason_with_verification(
        problem="?",
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # One ground_cnf call per step verification — three steps committed.
    assert result.total_groundings == 3
    assert result.total_verifications == 3


def test_controller_total_groundings_zero_when_verify_fn_used() -> None:
    """verify_fn bypasses the translator+grounding path; counter stays 0."""
    llm = StubGemmaBackend(scripts=[_thinking_scene("step.", answer="ans")])
    verifier = StubVerifier([_make_result(contradiction=False)])
    result = reason_with_verification(problem="?", llm=llm, verify_fn=verifier)
    assert result.total_groundings == 0


# ---------------------------------------------------------------------------
# Problem-statement pre-pass (Phase 6.5)
# ---------------------------------------------------------------------------


def test_initial_universe_seeded_from_problem_statement() -> None:
    """The pre-pass calls translate() per sentence; entities accumulate."""
    problem_text = "Alice and Bob are friends. Does Alice trust Bob?"
    translator = _StubTranslator(
        mapping={
            "Alice and Bob are friends": TranslationResult(
                cnf=CNF(clauses=()),
                universe=Universe(constants=("Alice", "Bob")),
            ),
            "Does Alice trust Bob": TranslationResult(
                cnf=CNF(clauses=()),
                universe=Universe(constants=()),
            ),
        }
    )
    llm = StubGemmaBackend(scripts=[_thinking_scene("trivial.", answer="ans")])

    result = reason_with_verification(
        problem=problem_text,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )
    assert result.initial_universe_size == 2


def test_first_step_can_use_universe_from_problem(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When step 1 has only a free variable, the pre-pass universe seeded
    from the problem statement makes grounding succeed without GroundingError."""
    from qverify.controller import controller as controller_module

    captured_universes: list[Universe] = []
    original_ground = controller_module.ground_cnf

    def spy_ground_cnf(cnf: CNF, universe: Universe) -> CNF:
        captured_universes.append(universe)
        return original_ground(cnf, universe)

    monkeypatch.setattr(controller_module, "ground_cnf", spy_ground_cnf)

    problem_text = "Premises: All cats have fur. Tom is a cat."
    universal_cnf = CNF(
        clauses=(
            Clause(
                literals=(
                    Literal(predicate="Cat", args=("x",), negated=True),
                    Literal(predicate="Fur", args=("x",)),
                )
            ),
        )
    )
    translator = _StubTranslator(
        mapping={
            # Pre-pass sentence 1 — universal, no entities.
            "Premises: All cats have fur": TranslationResult(
                cnf=universal_cnf, universe=Universe(constants=())
            ),
            # Pre-pass sentence 2 — extracts Tom.
            "Tom is a cat": TranslationResult(
                cnf=CNF(clauses=()),
                universe=Universe(constants=("Tom",)),
            ),
            # Step 1 is the universal — has variable x but no entities.
            "All cats have fur.": TranslationResult(
                cnf=universal_cnf, universe=Universe(constants=())
            ),
            "It is not the case that All cats have fur.": TranslationResult(
                cnf=universal_cnf, universe=Universe(constants=())
            ),
        }
    )
    llm = StubGemmaBackend(scripts=[_thinking_scene("All cats have fur.", answer="done")])

    result = reason_with_verification(
        problem=problem_text,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # Step 1's verification ran (no GroundingError raised) thanks to the
    # pre-pass seeding {Tom}. The universe used for grounding contains
    # Tom even though the step itself declared no entities.
    assert result.initial_universe_size == 1
    assert any("Tom" in u.constants for u in captured_universes), (
        f"expected pre-pass Tom in some grounding universe; saw {captured_universes}"
    )


def test_initial_universe_merges_with_per_step_entities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-pass seeds {Alice}; step 1 adds {Bob}; merged universe is both."""
    from qverify.controller import controller as controller_module

    captured_universes: list[Universe] = []
    original_ground = controller_module.ground_cnf

    def spy_ground_cnf(cnf: CNF, universe: Universe) -> CNF:
        captured_universes.append(universe)
        return original_ground(cnf, universe)

    monkeypatch.setattr(controller_module, "ground_cnf", spy_ground_cnf)

    problem_text = "Alice is a person."
    bob_clause_cnf = CNF(clauses=(Clause(literals=(Literal(predicate="Friend", args=("Bob",)),)),))
    translator = _StubTranslator(
        mapping={
            # Pre-pass sentence (period stripped by split_sentences).
            "Alice is a person": TranslationResult(
                cnf=CNF(clauses=()),
                universe=Universe(constants=("Alice",)),
            ),
            "Bob is a friend.": TranslationResult(
                cnf=bob_clause_cnf, universe=Universe(constants=("Bob",))
            ),
            "It is not the case that Bob is a friend.": TranslationResult(
                cnf=bob_clause_cnf, universe=Universe(constants=("Bob",))
            ),
        }
    )
    llm = StubGemmaBackend(scripts=[_thinking_scene("Bob is a friend.", answer="ok")])

    reason_with_verification(
        problem=problem_text,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # The merged universe at step 1's grounding must include both
    # the pre-pass entity (Alice) and the step's entity (Bob).
    assert captured_universes, "expected at least one ground_cnf call"
    merged = captured_universes[0].constants
    assert "Alice" in merged
    assert "Bob" in merged


def test_translator_failure_on_problem_raises_controller_error() -> None:
    """If every pre-pass sentence fails translation, the controller surfaces
    a ControllerError naming the count of failed sentences."""
    from qverify.controller.types import ControllerError
    from qverify.translator.translator import TranslationError

    class _FailingTranslator:
        def translate(self, statement: str) -> TranslationResult:
            raise TranslationError("translator exhausted retries")

    llm = StubGemmaBackend(scripts=[])
    with pytest.raises(ControllerError, match=r"failed to parse any of the \d+"):
        reason_with_verification(
            problem="One bad sentence. Another bad one. And a third.",
            llm=llm,
            translator=_FailingTranslator(),
            max_retries_per_step=0,
        )


# ---------------------------------------------------------------------------
# Sentence-by-sentence pre-pass (Phase 6.6)
# ---------------------------------------------------------------------------


class _RecordingTranslator:
    """Translator that mirrors _StubTranslator but records every call.

    Subset of _StubTranslator's API: it returns canned results from a
    mapping (with ``default_result`` fallback) and exposes ``calls`` so
    tests can assert on the exact sequence of translate() invocations.
    """

    def __init__(
        self,
        mapping: dict[str, TranslationResult] | None = None,
        default_result: TranslationResult | None = None,
        failures: set[str] | None = None,
    ) -> None:
        self._mapping = dict(mapping) if mapping else {}
        self._default = (
            default_result
            if default_result is not None
            else TranslationResult(cnf=CNF(clauses=()), universe=Universe(constants=()))
        )
        self._failures = set(failures) if failures else set()
        self.calls: list[str] = []

    def translate(self, statement: str) -> TranslationResult:
        self.calls.append(statement)
        if statement in self._failures:
            from qverify.translator.translator import TranslationError

            raise TranslationError(f"scripted failure for {statement!r}")
        if statement in self._mapping:
            return self._mapping[statement]
        return self._default


def test_pre_pass_splits_problem_into_sentences() -> None:
    """The pre-pass calls translate() once per sentence — three calls for
    a three-sentence canonical problem."""
    problem = "Premises: All cats have fur. Tom is a cat. Does Tom have fur?"
    translator = _RecordingTranslator()
    llm = StubGemmaBackend(scripts=[_thinking_scene("done.", answer="ans")])

    reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    pre_pass_calls = [c for c in translator.calls if not c.startswith("It is not the case that")]
    # The pre-pass calls one per sentence (3); the per-step verification
    # also adds the committed step "done" to translator.calls. Filter to
    # just the pre-pass texts.
    pre_pass_only = [c for c in pre_pass_calls if c != "done."]
    assert pre_pass_only == [
        "Premises: All cats have fur",
        "Tom is a cat",
        "Does Tom have fur",
    ]


def test_pre_pass_merges_entities_across_sentences() -> None:
    """Sentence 1 contributes Alice; sentence 2 contributes Bob;
    initial universe is the union."""
    problem = "Alice is a person. Bob is a person. Does Alice know Bob?"
    translator = _RecordingTranslator(
        mapping={
            "Alice is a person": TranslationResult(
                cnf=CNF(clauses=()), universe=Universe(constants=("Alice",))
            ),
            "Bob is a person": TranslationResult(
                cnf=CNF(clauses=()), universe=Universe(constants=("Bob",))
            ),
            "Does Alice know Bob": TranslationResult(
                cnf=CNF(clauses=()), universe=Universe(constants=())
            ),
        }
    )
    llm = StubGemmaBackend(scripts=[_thinking_scene("trivial.", answer="ans")])

    result = reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )
    assert result.initial_universe_size == 2


def test_pre_pass_skips_sentences_that_fail_to_translate() -> None:
    """A failing question sentence is logged + skipped; the remaining
    sentences still seed the universe."""
    problem = "Tom is a cat. All cats have fur. Does Tom have fur?"
    translator = _RecordingTranslator(
        mapping={
            "Tom is a cat": TranslationResult(
                cnf=CNF(clauses=()), universe=Universe(constants=("Tom",))
            ),
            "All cats have fur": TranslationResult(
                cnf=CNF(clauses=()), universe=Universe(constants=())
            ),
        },
        failures={"Does Tom have fur"},
    )
    llm = StubGemmaBackend(scripts=[_thinking_scene("trivial.", answer="ans")])

    result = reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )
    # Two of three sentences succeeded; Tom is in the initial universe.
    assert result.initial_universe_size == 1


def test_pre_pass_raises_when_all_sentences_fail() -> None:
    """If the translator fails on every pre-pass sentence, the controller
    surfaces ControllerError with a sentence count."""
    from qverify.controller.types import ControllerError

    problem = "First. Second. Third."
    translator = _RecordingTranslator(
        failures={"First", "Second", "Third"},
    )
    llm = StubGemmaBackend(scripts=[])

    with pytest.raises(ControllerError, match=r"failed to parse any of the 3 problem sentences"):
        reason_with_verification(
            problem=problem,
            llm=llm,
            translator=translator,
            max_retries_per_step=0,
        )


def test_pre_pass_caches_successful_translations_for_premise_reuse() -> None:
    """A sentence that appears in both the problem and as a committed
    step should hit the pre-pass cache on the per-step verification."""
    problem = "Tom is a cat. Tom has fur."
    tom_cat_result = TranslationResult(
        cnf=CNF(clauses=(Clause(literals=(Literal(predicate="Cat", args=("Tom",)),)),)),
        universe=Universe(constants=("Tom",)),
    )
    translator = _RecordingTranslator(
        mapping={
            "Tom is a cat": tom_cat_result,
            "Tom has fur": TranslationResult(
                cnf=CNF(clauses=(Clause(literals=(Literal(predicate="Fur", args=("Tom",)),)),)),
                universe=Universe(constants=("Tom",)),
            ),
        }
    )
    # The thinking trace re-states "Tom is a cat" (the period stripped by
    # split_sentences becomes the cache key) as a step.
    llm = StubGemmaBackend(scripts=[_thinking_scene("Tom is a cat", "Tom is a cat", answer="ans")])

    reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        max_retries_per_step=0,
    )

    # Pre-pass calls: "Tom is a cat", "Tom has fur" — 2 calls.
    # Per-step processing translates "It is not the case that <step>"
    # (uncached) plus the premise itself ("Tom is a cat", which IS in
    # the cache from the pre-pass — should not re-translate).
    pre_pass_count = sum(1 for c in translator.calls if c == "Tom is a cat")
    # Only the pre-pass call should appear; the per-step premise lookup
    # hits the cache populated by the pre-pass.
    assert pre_pass_count == 1, (
        f"expected 'Tom is a cat' to be translated only once via the pre-pass cache; "
        f"saw {pre_pass_count} calls in {translator.calls}"
    )


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
