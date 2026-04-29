"""QVerify HuggingFace Space - interactive quantum-assisted verifier."""

from __future__ import annotations

import os
from typing import Any

import gradio as gr

from qverify.controller import (
    StreamChunk,
    StubGemmaBackend,
    reason_with_verification,
)
from qverify.translator import Translator
from qverify.translator.llm import Gemma4StructuredBackend
from qverify.verifier.backends import IBMQuantumBackend, PennyLaneBackend

EXAMPLE_PROBLEM = "Premises: All cats have fur. Tom is a cat. Question: does Tom have fur?"
EXAMPLE_STEPS = "All cats have fur.\nTom is a cat.\nTherefore Tom has fur."


def _build_scene(steps: list[str]) -> list[StreamChunk]:
    """Wrap the user-supplied step list in a stub LLM scene."""
    scene: list[StreamChunk] = [
        StreamChunk(text="Reasoning provided by user.", phase="thinking"),
        StreamChunk(text="\n", phase="thinking"),
    ]
    for i, step in enumerate(steps, start=1):
        scene.append(StreamChunk(text=f"{i}. {step}\n", phase="answer"))
    scene.append(StreamChunk(text="\nDone.", phase="answer"))
    return scene


def _format_result(result: Any) -> dict[str, Any]:
    """Project a ControllerResult into a JSON-serialisable dict for Gradio."""
    return {
        "committed_steps": list(result.committed_steps),
        "rejected_steps": [r.original_step for r in result.rejected_steps],
        "total_verifications": result.total_verifications,
        "total_contradictions_found": result.total_contradictions_found,
        "total_groundings": result.total_groundings,
        "total_answer_steps_extracted": result.total_answer_steps_extracted,
        "initial_universe_size": result.initial_universe_size,
        "wall_clock_seconds": round(result.wall_clock_seconds, 1),
    }


def verify_on_simulator(problem: str, steps_text: str) -> dict[str, Any]:
    """Verify the user's structured reasoning on the PennyLane simulator."""
    steps = [s.strip() for s in steps_text.splitlines() if s.strip()]
    if not steps:
        return {"error": "No reasoning steps provided."}

    llm = StubGemmaBackend(scripts=[_build_scene(steps)])
    translator = Translator(backend=Gemma4StructuredBackend())

    result = reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        verifier_backend=PennyLaneBackend(),
        max_retries_per_step=1,
    )
    return _format_result(result)


def verify_on_ibm(problem: str, steps_text: str) -> dict[str, Any]:
    """Run a single verification on IBM Quantum hardware. Slow, opt-in."""
    if not os.environ.get("IBM_QUANTUM_TOKEN"):
        return {"error": "IBM_QUANTUM_TOKEN not set in this Space's secrets."}

    steps = [s.strip() for s in steps_text.splitlines() if s.strip()]
    if not steps:
        return {"error": "No reasoning steps provided."}

    llm = StubGemmaBackend(scripts=[_build_scene(steps)])
    translator = Translator(backend=Gemma4StructuredBackend())

    result = reason_with_verification(
        problem=problem,
        llm=llm,
        translator=translator,
        verifier_backend=IBMQuantumBackend(),
        max_retries_per_step=0,
    )
    return _format_result(result)


with gr.Blocks(title="QVerify") as demo:
    gr.Markdown(
        "# QVerify\n\n"
        "Quantum-assisted verification of structured logical reasoning. "
        "Built with Gemma 4 and PennyLane / IBM Quantum."
    )

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Input")
            problem_box = gr.Textbox(
                label="Problem statement",
                value=EXAMPLE_PROBLEM,
                lines=3,
                info=("The problem in natural language. Used to seed the universe of constants."),
            )
            steps_box = gr.Textbox(
                label="Reasoning steps (one per line)",
                value=EXAMPLE_STEPS,
                lines=8,
                info=("Each line is one declarative statement. Premises and inferences both work."),
            )
            with gr.Row():
                btn_sim = gr.Button("Verify on simulator (PennyLane)", variant="primary")
                btn_hw = gr.Button(
                    "Verify on IBM Heron r2 (real hardware, slow)",
                    variant="secondary",
                )

        with gr.Column():
            gr.Markdown("### Output")
            output = gr.JSON(label="Result")

    btn_sim.click(verify_on_simulator, inputs=[problem_box, steps_box], outputs=output)
    btn_hw.click(verify_on_ibm, inputs=[problem_box, steps_box], outputs=output)

    gr.Markdown("---")
    gr.Markdown(
        "### About\n\n"
        "Each reasoning step is translated to first-order CNF via Gemma 4 E4B "
        "with grammar-constrained generation, grounded in a finite universe of "
        "constants extracted from the problem statement, then verified for "
        "consistency using Grover's search on the chosen backend.\n\n"
        "[Repository](https://github.com/Quantum-Labor/qverify) · "
        "[Documentation](https://github.com/Quantum-Labor/qverify/tree/main/docs)"
    )


if __name__ == "__main__":
    demo.launch()
