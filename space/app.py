"""QVerify HuggingFace Space - verifier-only demo, runs on CPU Basic.

This Space exposes the QVerify *verifier* component: a CNF is fed
straight to Grover's search, on either the PennyLane simulator (local
to this Space's CPU container) or IBM Heron r2 hardware (delegated to
IBM Cloud over the network, account credentials read from Space
Secrets).

The *translator* component (Gemma 4 E4B with grammar-constrained
generation) is intentionally not loaded here: it needs a GPU, which
CPU Basic does not provide. The full pipeline runs locally from the
GitHub repository, see https://github.com/Quantum-Labor/qverify.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any

import gradio as gr

from qverify.translator.cnf import CNF, Clause, Literal
from qverify.verifier import verify
from qverify.verifier._universe import Universe
from qverify.verifier.backends import IBMQuantumBackend, PennyLaneBackend
from qverify.verifier.grounding import ground_cnf

IBM_CONSOLE_BASE = "https://quantum.cloud.ibm.com/workloads?search="


@dataclass(frozen=True)
class CnfExample:
    """One ready-to-run CNF example surfaced in the Space's dropdown."""

    label: str
    description: str
    cnf: CNF
    universe: Universe


def _atom(predicate: str, *args: str, neg: bool = False) -> Literal:
    return Literal(predicate=predicate, args=args, negated=neg)


def _clause(*lits: Literal) -> Clause:
    return Clause(literals=lits)


def _build_examples() -> dict[str, CnfExample]:
    """Three pre-built CNFs covering propositional, ground first-order, and grounded universal."""
    cat_fur_consistency = CnfExample(
        label="Cat and HasFur consistency",
        description=(
            "Two propositional atoms, both asserted: Cat AND HasFur. The "
            "verifier should report contradiction_found=False (the formula "
            "is consistent: setting both to True satisfies it)."
        ),
        cnf=CNF(
            clauses=(
                _clause(_atom("Cat")),
                _clause(_atom("HasFur")),
            )
        ),
        universe=Universe(constants=()),
    )

    tom_is_a_cat = CnfExample(
        label="Tom is a cat",
        description=(
            "A single ground first-order atom: Cat(Tom). Universe = {Tom}. "
            "Trivially satisfiable; consistency mode reports no contradiction."
        ),
        cnf=CNF(clauses=(_clause(_atom("Cat", "Tom")),)),
        universe=Universe(constants=("Tom",)),
    )

    universal_rule = CnfExample(
        label="Universal cat-fur rule",
        description=(
            "First-order rule 'forall x. Cat(x) -> HasFur(x)' grounded over "
            "{Tom, Whiskers}. The grounded CNF has two clauses, one per "
            "constant. Consistent: any assignment that makes both Cat(Tom) "
            "and Cat(Whiskers) false satisfies the formula."
        ),
        cnf=CNF(
            clauses=(
                _clause(
                    _atom("Cat", "x", neg=True),
                    _atom("HasFur", "x"),
                ),
            )
        ),
        universe=Universe(constants=("Tom", "Whiskers")),
    )

    return {
        cat_fur_consistency.label: cat_fur_consistency,
        tom_is_a_cat.label: tom_is_a_cat,
        universal_rule.label: universal_rule,
    }


EXAMPLES = _build_examples()
EXAMPLE_LABELS = list(EXAMPLES.keys())
DEFAULT_LABEL = EXAMPLE_LABELS[0]

IBM_TOKEN_PRESENT = bool(os.environ.get("IBM_QUANTUM_TOKEN"))
IBM_INSTANCE_PRESENT = bool(os.environ.get("IBM_QUANTUM_INSTANCE"))
IBM_AVAILABLE = IBM_TOKEN_PRESENT and IBM_INSTANCE_PRESENT


def _format_cnf(cnf: CNF) -> str:
    """Render a CNF as a readable multi-line string."""
    if not cnf.clauses:
        return "(empty CNF, trivially satisfied)"
    lines = []
    for c in cnf.clauses:
        lits = []
        for lit in c.literals:
            args = f"({', '.join(lit.args)})" if lit.args else ""
            lits.append(f"{'¬' if lit.negated else ''}{lit.predicate}{args}")
        lines.append("(" + " ∨ ".join(lits) + ")")
    return "\n".join(lines)


def _on_example_change(label: str) -> tuple[str, str, str]:
    """Update the CNF preview, universe preview, and description on dropdown change."""
    ex = EXAMPLES[label]
    universe_str = (
        ", ".join(ex.universe.constants) if ex.universe.constants else "(empty)"
    )
    return _format_cnf(ex.cnf), universe_str, ex.description


def _run_verify(label: str, mode: str, backend_kind: str) -> dict[str, Any]:
    """Ground the example CNF, run Grover's search, and return a JSON-friendly result."""
    if label not in EXAMPLES:
        return {"error": f"unknown example: {label}"}

    ex = EXAMPLES[label]
    grounded = ground_cnf(ex.cnf, ex.universe)

    if backend_kind == "pennylane":
        backend = PennyLaneBackend()
    elif backend_kind == "ibm":
        if not IBM_AVAILABLE:
            return {
                "error": (
                    "IBM_QUANTUM_TOKEN and/or IBM_QUANTUM_INSTANCE are not "
                    "set in this Space's Secrets. The IBM hardware path is "
                    "unavailable; use the simulator button."
                )
            }
        backend = IBMQuantumBackend()
    else:
        return {"error": f"unknown backend kind: {backend_kind}"}

    start = time.monotonic()
    result = verify(grounded, backend=backend, mode=mode)
    wall = round(time.monotonic() - start, 1)

    counter_model: Any
    if result.counter_model is None:
        if mode == "consistency":
            counter_model = (
                "(consistency mode: no satisfying assignment is surfaced; "
                "this is expected when the formula is consistent because "
                "the verifier does not display a model in this mode, and "
                "when inconsistent because UNSAT means there is none)"
            )
        else:
            counter_model = "no counter-model found (formula is entailed)"
    else:
        counter_model = result.counter_model.assignment

    out: dict[str, Any] = {
        "example": label,
        "mode": mode,
        "backend": result.backend_name,
        "contradiction_found": result.contradiction_found,
        "counter_model": counter_model,
        "n_variables": result.n_variables,
        "n_clauses": result.n_clauses,
        "n_grover_iterations": result.n_grover_iterations,
        "shots": result.shots,
        "wall_clock_seconds": wall,
    }

    job_id = result.metadata.get("job_id") if result.metadata else None
    if isinstance(job_id, str) and job_id:
        out["ibm_job_id"] = job_id
        out["ibm_job_url"] = IBM_CONSOLE_BASE + job_id

    return out


def verify_on_simulator(label: str, mode: str) -> dict[str, Any]:
    return _run_verify(label, mode, "pennylane")


def verify_on_ibm(label: str, mode: str) -> dict[str, Any]:
    return _run_verify(label, mode, "ibm")


_initial_cnf, _initial_universe, _initial_description = _on_example_change(DEFAULT_LABEL)

_ibm_status_md = (
    "IBM Quantum credentials detected in Space Secrets. "
    "The hardware button is enabled; runs consume your free-tier "
    "monthly budget (10 minutes shared)."
    if IBM_AVAILABLE
    else (
        "IBM Quantum credentials are not configured. "
        "Add `IBM_QUANTUM_TOKEN` and `IBM_QUANTUM_INSTANCE` to this "
        "Space's Settings -> Variables and secrets to enable hardware runs."
    )
)


with gr.Blocks(title="QVerify") as demo:
    gr.Markdown(
        "# QVerify  ·  quantum-assisted CNF verifier\n\n"
        "This Space exposes the QVerify **verifier** component: a CNF is "
        "fed directly to Grover's search, on either a CPU-side PennyLane "
        "simulator or IBM Quantum's Heron r2 processor.\n\n"
        "Pick a pre-built example, choose a verification mode, then run."
    )

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 1. Pick a CNF")
            example_dd = gr.Dropdown(
                label="Example",
                choices=EXAMPLE_LABELS,
                value=DEFAULT_LABEL,
            )
            description_box = gr.Markdown(_initial_description)
            cnf_box = gr.Textbox(
                label="CNF (read-only)",
                value=_initial_cnf,
                lines=4,
                interactive=False,
            )
            universe_box = gr.Textbox(
                label="Universe of constants",
                value=_initial_universe,
                interactive=False,
            )

            gr.Markdown("### 2. Choose a mode")
            mode_radio = gr.Radio(
                label="Mode",
                choices=["consistency", "entailment"],
                value="consistency",
                info=(
                    "consistency: SAT means the CNF is consistent. "
                    "entailment: SAT yields a counter-model showing the "
                    "step is not entailed by the premises."
                ),
            )

            gr.Markdown("### 3. Run")
            with gr.Row():
                btn_sim = gr.Button(
                    "Verify on PennyLane simulator", variant="primary"
                )
                btn_hw = gr.Button(
                    "Verify on IBM Heron r2",
                    variant="secondary",
                    interactive=IBM_AVAILABLE,
                )
            gr.Markdown(_ibm_status_md)

        with gr.Column(scale=1):
            gr.Markdown("### Result")
            output = gr.JSON(label="Verification result")

    example_dd.change(
        _on_example_change,
        inputs=[example_dd],
        outputs=[cnf_box, universe_box, description_box],
    )
    btn_sim.click(
        verify_on_simulator,
        inputs=[example_dd, mode_radio],
        outputs=output,
    )
    btn_hw.click(
        verify_on_ibm,
        inputs=[example_dd, mode_radio],
        outputs=output,
    )

    gr.Markdown("---")
    gr.Markdown(
        "### About\n\n"
        "**This Space demos the verifier component of QVerify.** A CNF "
        "is grounded over its declared universe, encoded into a Grover "
        "circuit, and run on the chosen backend. The classical "
        "post-check walks the most-frequent measured bitstrings until "
        "one satisfies the CNF.\n\n"
        "**The full pipeline (translator + grounding + verifier) "
        "requires a GPU and runs locally**: it loads Gemma 4 E4B with "
        "grammar-constrained generation to convert natural-language "
        "premises into CNF before verification. CPU Basic Spaces cannot "
        "host the translator, so this Space is verifier-only. See the "
        "[GitHub repository](https://github.com/Quantum-Labor/qverify) "
        "for the end-to-end pipeline.\n\n"
        "**Real quantum hardware runs are routed through your IBM "
        "Quantum free-tier account** (Open Plan, ~10 minutes of quantum "
        "time per month, shared across users). The Space orchestrates "
        "the call; the actual circuit executes on `ibm_fez` (Heron r2, "
        "156 qubits) via IBM Cloud. A previously verified hardware run "
        "is documented [here](https://github.com/Quantum-Labor/qverify"
        "#hardware-run)."
    )


if __name__ == "__main__":
    demo.launch()
