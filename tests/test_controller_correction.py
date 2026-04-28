"""Tests for the correction-prompt formatter."""

from __future__ import annotations

from qverify.controller.correction import format_counter_model_prompt
from qverify.verifier.types import CounterModel


def test_counter_model_variables_appear_in_prompt() -> None:
    cm = CounterModel(assignment={"Bird": True, "Flies": False})
    prompt = format_counter_model_prompt(step="All birds can fly.", counter_model=cm, premises=[])
    assert "Bird=true" in prompt
    assert "Flies=false" in prompt


def test_original_step_text_appears_verbatim() -> None:
    step = "Therefore the penguin flies through the clouds."
    cm = CounterModel(assignment={"Penguin": True, "Flies": False})
    prompt = format_counter_model_prompt(step=step, counter_model=cm, premises=[])
    assert step in prompt


def test_tone_is_neutral_no_adversarial_language() -> None:
    cm = CounterModel(assignment={"P": True})
    prompt = format_counter_model_prompt(step="P is false.", counter_model=cm, premises=[])
    lowered = prompt.lower()
    forbidden_phrases = [
        "you are wrong",
        "you're wrong",
        "incorrect",
        "stupid",
        "fix your",
        "your mistake",
    ]
    for phrase in forbidden_phrases:
        assert phrase not in lowered, f"prompt contains adversarial phrase {phrase!r}"


def test_premises_listed_when_provided() -> None:
    cm = CounterModel(assignment={"X": True})
    premises = ["All birds can fly.", "Penguins are birds."]
    prompt = format_counter_model_prompt(step="X is true.", counter_model=cm, premises=premises)
    for premise in premises:
        assert premise in prompt


def test_empty_premises_list_renders_sensibly() -> None:
    cm = CounterModel(assignment={"X": True})
    prompt = format_counter_model_prompt(step="X is true.", counter_model=cm, premises=[])
    assert "no prior premises" in prompt


def test_step_index_is_used_when_provided() -> None:
    cm = CounterModel(assignment={"P": False})
    prompt = format_counter_model_prompt(
        step="P is true.", counter_model=cm, premises=[], step_index=3
    )
    assert "step 3" in prompt


def test_empty_assignment_does_not_break_prompt() -> None:
    cm = CounterModel(assignment={})
    prompt = format_counter_model_prompt(step="X.", counter_model=cm, premises=["A."])
    assert "(empty assignment" in prompt
