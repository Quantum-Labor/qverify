"""Unit tests for the controller's answer-phase step extractor."""

from __future__ import annotations

from qverify.controller.utils import extract_answer_steps


def test_empty_string_returns_empty_list() -> None:
    assert extract_answer_steps("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert extract_answer_steps("   \n\t  \n") == []


def test_no_numbered_structure_returns_empty_list() -> None:
    assert extract_answer_steps("Yes, Tom has fur.") == []


def test_single_numbered_step_with_period_marker() -> None:
    assert extract_answer_steps("1. Tom is a cat.") == ["Tom is a cat."]


def test_three_numbered_steps_with_period_markers() -> None:
    text = "1. All cats have fur.\n2. Tom is a cat.\n3. Therefore Tom has fur."
    assert extract_answer_steps(text) == [
        "All cats have fur.",
        "Tom is a cat.",
        "Therefore Tom has fur.",
    ]


def test_strips_canonical_gemma4_bold_label_pattern() -> None:
    text = "1.  **Identify the General Rule (Premise 1):** The first premise states fur."
    assert extract_answer_steps(text) == ["The first premise states fur."]


def test_mixed_paren_marker_format() -> None:
    text = "1) First step.\n2) Second step.\n3) Third step."
    assert extract_answer_steps(text) == [
        "First step.",
        "Second step.",
        "Third step.",
    ]


def test_parenthesized_number_marker_format() -> None:
    text = "(1) First.\n(2) Second."
    assert extract_answer_steps(text) == ["First.", "Second."]


def test_indented_numbered_lines_with_tabs_and_spaces() -> None:
    text = "    1. Indented with spaces.\n\t2. Indented with tab."
    assert extract_answer_steps(text) == [
        "Indented with spaces.",
        "Indented with tab.",
    ]


def test_strips_leftover_markdown_noise_from_content() -> None:
    text = "1. **A bold** thing with `code` and _italic_ noise."
    assert extract_answer_steps(text) == ["A bold thing with code and italic noise."]


def test_skips_non_numbered_lines_between_numbered_steps() -> None:
    text = "1. First step.\n2. Second step.\n**Answer:** Yes\n3. Third step."
    assert extract_answer_steps(text) == [
        "First step.",
        "Second step.",
        "Third step.",
    ]


def test_real_gemma4_answer_phase_sample() -> None:
    """Distilled from /tmp/gemma4_dump.txt — real Gemma 4 E4B output for
    the syllogism in the sanity test."""
    text = (
        "**Reasoning Step by Step:**\n\n"
        "1.  **Identify the General Rule (Premise 1):** The first premise "
        'states that the entire group of "cats" possesses the property of '
        'having "fur."\n'
        "2.  **Identify the Specific Case (Premise 2):** The second premise "
        'identifies "Tom" as a member of the group "cats."\n'
        "3.  **Apply the Logic:** Since Tom belongs to the group (cats) "
        "that is defined by having fur, Tom must possess that property.\n\n"
        "**Answer:**\n\n"
        "Yes"
    )
    steps = extract_answer_steps(text)
    assert len(steps) == 3
    for step in steps:
        assert "**" not in step, f"bold marker leaked: {step!r}"
        assert not step.startswith("Identify"), f"bold-label not stripped from: {step!r}"
    # Substantive content survives.
    assert "first premise" in steps[0].lower()
    assert "second premise" in steps[1].lower()
    assert "tom must possess" in steps[2].lower()


def test_multi_line_continuation_only_first_line_kept() -> None:
    """Documented limitation: a step that wraps across lines is truncated
    at the first newline. Phase 7+ may extend the regex if benchmarks need
    multi-line support."""
    text = "1. First step continues\n   onto a second line that we drop.\n2. Second step."
    assert extract_answer_steps(text) == ["First step continues", "Second step."]


def test_numbered_list_followed_by_unnumbered_answer_line() -> None:
    text = "1. First.\n2. Second.\n3. Third.\n\nAnswer: Yes"
    steps = extract_answer_steps(text)
    assert steps == ["First.", "Second.", "Third."]
    assert "Yes" not in steps


def test_internal_colon_not_truncated_when_no_bold_markers() -> None:
    """Plain colons inside content (no surrounding asterisks) must not
    trigger the inline-label stripper."""
    text = "1. The fact is: Tom has fur."
    assert extract_answer_steps(text) == ["The fact is: Tom has fur."]


def test_bold_label_with_dotted_subject_stripped() -> None:
    """Label content can contain non-asterisk, non-colon characters
    including parentheses, digits, and dots within reason."""
    text = "1. **Step 2.1 Goal:** Establish the predicate."
    assert extract_answer_steps(text) == ["Establish the predicate."]
