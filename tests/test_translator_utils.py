"""Unit tests for the shared translator utilities."""

from __future__ import annotations

from qverify.translator.utils import split_sentences


def test_empty_string_returns_empty_list() -> None:
    assert split_sentences("") == []


def test_whitespace_only_returns_empty_list() -> None:
    assert split_sentences("   \n\t  ") == []


def test_single_sentence_with_trailing_period() -> None:
    assert split_sentences("Tom is a cat.") == ["Tom is a cat"]


def test_two_sentences_split_on_period() -> None:
    assert split_sentences("All cats have fur. Tom is a cat.") == [
        "All cats have fur",
        "Tom is a cat",
    ]


def test_mixed_punctuation_period_question_exclamation() -> None:
    assert split_sentences("It rains. Does Tom flee? Yes!") == [
        "It rains",
        "Does Tom flee",
        "Yes",
    ]


def test_consecutive_whitespace_between_sentences_handled() -> None:
    assert split_sentences("First.   \n\n  Second.\tThird.") == [
        "First",
        "Second",
        "Third",
    ]


def test_no_trailing_punctuation_returns_single_sentence() -> None:
    assert split_sentences("This has no period") == ["This has no period"]


def test_canonical_sanity_problem_split() -> None:
    """The canonical sanity-test problem has two periods and one question
    mark — three sentence-ending tokens, three resulting sentences. (The
    Phase 6.6 spec counted four; the regex sees three because there's no
    sentence terminator after 'Question:' before the colon-introduced
    clause.)"""
    problem = "Premises: All cats have fur. Tom is a cat. Question: does Tom have fur?"
    assert split_sentences(problem) == [
        "Premises: All cats have fur",
        "Tom is a cat",
        "Question: does Tom have fur",
    ]
