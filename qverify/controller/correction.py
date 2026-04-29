"""Correction-prompt formatter for the controller's retry loop."""

from __future__ import annotations

from qverify.verifier.types import CounterModel


def _format_assignment(counter_model: CounterModel) -> str:
    """Render a counter-model as ``P=true, Q=false`` style fragments."""
    if not counter_model.assignment:
        return "(empty assignment — the formula is satisfied vacuously)"
    parts = [
        f"{atom}={'true' if value else 'false'}"
        for atom, value in sorted(counter_model.assignment.items())
    ]
    return ", ".join(parts)


def format_counter_model_prompt(
    step: str,
    counter_model: CounterModel | None,
    premises: list[str],
    *,
    step_index: int | None = None,
) -> str:
    """Build a neutral, pedagogical correction prompt for the LLM.

    The prompt explains which step is inconsistent, recaps the current
    premise list, and asks for a rewrite of just that step. When the
    verifier ran in entailment mode and surfaced a witness assignment,
    that concrete counter-model is shown to the LLM; consistency-mode
    rejections (UNSAT — there is no satisfying assignment) instead get a
    generic explanation. Tone is intentionally non-adversarial — no
    "you are wrong" language — so retries don't push the model toward
    defensive or evasive rewordings.
    """
    label = f"step {step_index}" if step_index is not None else "this step"
    premise_block = "\n".join(f"  - {p}" for p in premises) if premises else "  (no prior premises)"

    if counter_model is None:
        explanation = (
            "The step contradicts the established premises (no consistent "
            "assignment of the underlying atoms exists)."
        )
    else:
        assignment = _format_assignment(counter_model)
        explanation = (
            f"A counter-example: assigning {assignment} satisfies the premises "
            f"but makes {label} false."
        )

    return (
        f'The reasoning {label} "{step}" is inconsistent with the premises '
        f"established so far.\n"
        f"\n"
        f"Premises so far:\n"
        f"{premise_block}\n"
        f"\n"
        f"{explanation} Please reconsider {label} and rewrite it as one short "
        f"paragraph. Do not restate the premises; emit only the corrected step."
    )
