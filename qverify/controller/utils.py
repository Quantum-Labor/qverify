"""Helpers for parsing the structured-answer phase of a Gemma 4 response.

The answer phase typically follows a recognizable pattern::

    [optional "**Reasoning Step by Step:**" header]
    1. [optional bold label] [content sentence].
    2. ...
    3. ...
    [blank line]
    [optional "**Answer:**" header]
    [final answer text]

We extract the numbered content sentences, stripped of markdown bold,
inline labels, and surrounding bullets. Each returned step is a single
declarative sentence suitable for ``translator.translate()``.
"""

from __future__ import annotations

import re

# Matches ``1.``, ``1)``, ``(1)`` at the start of a line (possibly indented).
# Captures the rest of the line as group 1 — ``(.*)`` does not span newlines
# even under ``re.MULTILINE``, so multi-line continuations are not picked up.
_NUMBERED_LINE_RE = re.compile(r"^[ \t]*(?:\(?\d+\)?[.)]\s+)(.*)$", re.MULTILINE)

# Strips a leading ``**Bold Label:**`` or ``*Bold Label:*`` from a step's
# captured content. The asterisks on BOTH sides are required so that plain
# inline colons inside content (``"The fact is: Tom has fur."``) are not
# accidentally truncated. The character cap on the label body avoids
# greedy matches into substantive text on long sentences.
_INLINE_LABEL_RE = re.compile(r"^\*+\s*[^*:\n]{1,80}\s*:\*+\s*")

# Strips leftover markdown markers (asterisk, underscore, backtick, hash).
# Punctuation, brackets, parentheses, and LaTeX-like ``\command`` tokens
# are kept — they're often legitimate content.
_MARKDOWN_NOISE_RE = re.compile(r"\*+|_+|`+|#+")


def extract_answer_steps(answer_text: str) -> list[str]:
    """Return the declarative reasoning steps from a Gemma 4 answer phase.

    Returns an empty list if no numbered structure is found — callers
    should treat this as "the model gave a freeform answer with no
    intermediate reasoning to verify". The final answer text itself is
    not in the returned list; extract it separately if needed.
    """
    if not answer_text:
        return []

    steps: list[str] = []
    for match in _NUMBERED_LINE_RE.finditer(answer_text):
        raw = match.group(1).strip()
        if not raw:
            continue
        cleaned = _INLINE_LABEL_RE.sub("", raw, count=1).strip()
        cleaned = _MARKDOWN_NOISE_RE.sub("", cleaned).strip()
        cleaned = " ".join(cleaned.split())
        if cleaned:
            steps.append(cleaned)
    return steps
