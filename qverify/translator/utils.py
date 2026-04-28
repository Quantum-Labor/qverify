"""Translator utility helpers shared across the translator and controller."""

from __future__ import annotations

import re

# Regex source-of-truth for sentence boundaries. Splits on any run of
# ``.``, ``?``, or ``!`` characters — sufficient for the academic
# / benchmark inputs QVerify targets (ProofWriter, RuleTaker, FOLIO,
# textbook syllogisms). Smarter splitters (NLTK, spaCy) would add
# heavy model dependencies for negligible gain on these inputs.
_SENTENCE_SPLIT_RE = re.compile(r"[.!?]+")


def split_sentences(text: str) -> list[str]:
    """Split ``text`` into a list of non-empty stripped sentences.

    Empty / whitespace-only inputs return ``[]``. Trailing punctuation
    is consumed by the splitter so individual sentences don't carry the
    period / question mark / exclamation point. Sentences shorter than
    one non-whitespace character (e.g. a stray ``"!!"``) are dropped.
    """
    if not text or not text.strip():
        return []
    return [s.strip() for s in _SENTENCE_SPLIT_RE.split(text) if s.strip()]
